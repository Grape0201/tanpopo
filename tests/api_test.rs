use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Method, Request, StatusCode};
use serde_json::{Value, json};
use tanpopo::index::IndexManager;
use tanpopo::server::build_router;
use tanpopo::tokenizer::{
    SplitMode, SudachiTokenizer, SudachiTokenizerConfig, TokenForm, build_fallback_dictionary,
    build_fallback_system_dic_bytes, discover_dictionaries, load_dictionary_from_dir,
};
use tempfile::TempDir;
use tower::ServiceExt;

async fn send_json(
    app: &axum::Router,
    method: Method,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let body_str = body.map(|v| v.to_string()).unwrap_or_default();
    send_raw(app, method, uri, "application/json", body_str).await
}

async fn send_raw(
    app: &axum::Router,
    method: Method,
    uri: &str,
    content_type: &str,
    body: String,
) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", content_type)
        .body(Body::from(body))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let val: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, val)
}

#[test]
fn test_sudachi_tokenizer_modes_and_normalization() {
    let dict = build_fallback_dictionary().unwrap();

    // Mode C keeps compound noun "関西国際空港" together
    let tok_c = SudachiTokenizer::new(dict.clone(), SudachiTokenizerConfig::with_mode(SplitMode::C));
    let tokens_c: Vec<String> = tok_c
        .analyze("関西国際空港に行く。")
        .into_iter()
        .map(|t| t.token)
        .collect();
    assert_eq!(tokens_c, vec!["関西国際空港", "に", "行く"]);

    // Mode A splits into short units: "関西", "国際", "空港"
    let tok_a = SudachiTokenizer::new(dict.clone(), SudachiTokenizerConfig::with_mode(SplitMode::A));
    let tokens_a: Vec<String> = tok_a
        .analyze("関西国際空港に行く。")
        .into_iter()
        .map(|t| t.token)
        .collect();
    assert_eq!(tokens_a, vec!["関西", "国際", "空港", "に", "行く"]);

    // Mode B splits into middle units: "関西", "国際空港"
    let tok_b = SudachiTokenizer::new(dict.clone(), SudachiTokenizerConfig::with_mode(SplitMode::B));
    let tokens_b: Vec<String> = tok_b
        .analyze("関西国際空港に行く。")
        .into_iter()
        .map(|t| t.token)
        .collect();
    assert_eq!(tokens_b, vec!["関西", "国際空港", "に", "行く"]);

    // Normalized form unifies spelling variants (打込む -> 打ち込む, しょうゆ -> 醤油, 蒲公英 -> タンポポ)
    let norm_tokens: Vec<String> = tok_a
        .analyze("しょうゆを打込む、蒲公英")
        .into_iter()
        .map(|t| t.token)
        .collect();
    assert_eq!(norm_tokens, vec!["醤油", "を", "打ち込む", "タンポポ"]);

    // Surface form preserves original surface
    let tok_surface = SudachiTokenizer::new(
        dict,
        SudachiTokenizerConfig {
            mode: SplitMode::A,
            form: TokenForm::Surface,
            ..Default::default()
        },
    );
    let surf_tokens: Vec<String> = tok_surface
        .analyze("しょうゆを打込む")
        .into_iter()
        .map(|t| t.token)
        .collect();
    assert_eq!(surf_tokens, vec!["しょうゆ", "を", "打込む"]);
}

#[tokio::test]
async fn test_rest_api_e2e_and_persistence() {
    let tmp_dir = TempDir::new().unwrap();
    let dict = build_fallback_dictionary().unwrap();

    // 1. Initialize server
    {
        let mgr = Arc::new(
            IndexManager::new(tmp_dir.path().to_path_buf(), dict.clone(), 15_000_000).unwrap(),
        );
        let app = build_router(mgr);

        // Analyze endpoint
        let (status, health_res) = send_json(&app, Method::GET, "/_cluster/health", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(health_res["status"], "green");
        assert_eq!(health_res["cluster_name"], "tanpopo");

        let (status, analyze_res) = send_json(
            &app,
            Method::POST,
            "/_analyze",
            Some(json!({
                "text": "全文検索エンジン",
                "mode": "A"
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let analyzed_tokens: Vec<&str> = analyze_res["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["token"].as_str().unwrap())
            .collect();
        assert_eq!(analyzed_tokens, vec!["全文", "検索", "エンジン"]);

        // Create index with explicit mappings & custom analyzer
        let (status, _) = send_json(
            &app,
            Method::PUT,
            "/articles",
            Some(json!({
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "ja_short": {
                                "mode": "A",
                                "form": "normalized",
                                "stop_tags": ["補助記号", "空白", "助詞", "助動詞"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "title": { "type": "text", "analyzer": "ja_short" },
                        "body": { "type": "text", "analyzer": "sudachi_a" },
                        "category": { "type": "keyword" },
                        "views": { "type": "long" },
                        "published": { "type": "boolean" },
                        "created_at": { "type": "date" }
                    }
                }
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // Bulk index documents
        let bulk_ndjson = [
            r#"{"index": {"_id": "1"}}"#,
            r#"{"title": "関西国際空港の案内", "body": "大阪と京都から関西国際空港に行く方法。", "category": "travel", "views": 120, "published": true, "created_at": "2026-10-01T10:00:00Z"}"#,
            r#"{"index": {"_id": "2"}}"#,
            r#"{"title": "日本語の全文検索エンジン", "body": "省メモリで高速な形態素解析と検索を実現します。", "category": "tech", "views": 500, "published": true, "created_at": "2026-10-15T12:00:00Z"}"#,
            r#"{"index": {"_id": "3"}}"#,
            r#"{"title": "たんぽぽとしょうゆ", "body": "蒲公英の観察記録を打込む。", "category": "nature", "views": 30, "published": false, "created_at": "2026-10-20T08:00:00Z"}"#,
        ]
        .join("\n")
            + "\n";

        let (status, bulk_res) = send_raw(
            &app,
            Method::POST,
            "/articles/_bulk",
            "application/x-ndjson",
            bulk_ndjson,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(bulk_res["errors"], false);
        assert_eq!(bulk_res["items"].as_array().unwrap().len(), 3);

        // Match search using Mode A sub-token ("空港" hits "関西国際空港")
        let (status, search_res) = send_json(
            &app,
            Method::POST,
            "/articles/_search",
            Some(json!({
                "query": {
                    "match": { "title": "空港" }
                }
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(search_res["hits"]["total"]["value"], 1);
        assert_eq!(search_res["hits"]["hits"][0]["_id"], "1");

        // Normalized form search ("醤油" matches "しょうゆ", "打ち込む" matches "打込む")
        let (status, norm_res) = send_json(
            &app,
            Method::POST,
            "/articles/_search",
            Some(json!({
                "query": {
                    "bool": {
                        "must": [
                            { "match": { "title": "醤油" } },
                            { "match": { "body": "打ち込む" } }
                        ]
                    }
                }
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(norm_res["hits"]["total"]["value"], 1);
        assert_eq!(norm_res["hits"]["hits"][0]["_id"], "3");

        // Bool + Range + Term filter
        let (status, bool_res) = send_json(
            &app,
            Method::POST,
            "/articles/_search",
            Some(json!({
                "query": {
                    "bool": {
                        "must": { "term": { "published": true } },
                        "filter": { "range": { "views": { "gte": 200 } } }
                    }
                }
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(bool_res["hits"]["total"]["value"], 1);
        assert_eq!(bool_res["hits"]["hits"][0]["_id"], "2");
    }

    // 2. Re-open IndexManager from disk and verify persistence
    {
        let mgr = Arc::new(
            IndexManager::new(tmp_dir.path().to_path_buf(), dict, 15_000_000).unwrap(),
        );
        let app = build_router(mgr);

        let (status, doc_res) = send_json(&app, Method::GET, "/articles/_doc/2", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(doc_res["found"], true);
        assert_eq!(doc_res["_source"]["category"], "tech");

        // Delete doc 3 and verify count drops to 2
        let (status, _) = send_json(&app, Method::DELETE, "/articles/_doc/3", None).await;
        assert_eq!(status, StatusCode::OK);

        let (status, all_res) = send_json(&app, Method::GET, "/articles/_search", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(all_res["hits"]["total"]["value"], 2);

        // 3. Test /_mapping and /{index}/_mapping for Grafana field discovery
        let (status, map_all) = send_json(&app, Method::GET, "/_mapping", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            map_all["articles"]["mappings"]["properties"]["created_at"]["type"],
            "date"
        );
        assert_eq!(
            map_all["articles"]["mappings"]["properties"]["category"]["type"],
            "keyword"
        );

        let (status, map_idx) = send_json(&app, Method::GET, "/articles/_mapping", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            map_idx["articles"]["mappings"]["properties"]["views"]["type"],
            "long"
        );

        let (status, _) = send_json(&app, Method::GET, "/nonexistent/_mapping", None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        // 4. Test /_msearch and /{index}/_msearch with Grafana query patterns
        let msearch_ndjson = [
            // Query 1: Grafana raw_data query with empty index (searches across all indices), unmapped @timestamp range, query_string, sort, docvalue_fields
            r#"{"search_type":"query_then_fetch","ignore_unavailable":true,"index":""}"#,
            r#"{"size":500,"query":{"bool":{"filter":[{"range":{"@timestamp":{"gte":1790000000000,"lte":1800000000000,"format":"epoch_millis"}}},{"query_string":{"analyze_wildcard":true,"query":"空港"}}]}},"sort":[{"created_at":{"order":"desc","unmapped_type":"boolean"}},{"_doc":{"order":"desc"}}],"docvalue_fields":[{"field":"created_at","format":"epoch_millis"}]}"#,
            // Query 2: Grafana date_histogram + terms + avg metric aggregation on explicit index
            r#"{"search_type":"query_then_fetch","ignore_unavailable":true,"index":["articles"]}"#,
            r#"{"size":0,"query":{"bool":{"filter":[{"range":{"created_at":{"gte":"2026-10-01T00:00:00Z","lte":"2026-10-31T23:59:59Z"}}}]}},"aggs":{"by_cat":{"terms":{"field":"category","size":10},"aggs":{"avg_views":{"avg":{"field":"views"}}}},"over_time":{"date_histogram":{"field":"created_at","fixed_interval":"1d","min_doc_count":0}}}}"#,
            // Query 3: Missing index with ignore_unavailable: false returns 404 item in responses
            r#"{"index":"missing_idx","ignore_unavailable":false}"#,
            r#"{"query":{"match_all":{}}}"#,
        ]
        .join("\n")
            + "\n";

        let (status, msearch_res) = send_raw(
            &app,
            Method::POST,
            "/_msearch?max_concurrent_shard_requests=5",
            "application/x-ndjson",
            msearch_ndjson,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let responses = msearch_res["responses"].as_array().unwrap();
        assert_eq!(responses.len(), 3);

        // Verify Query 1 response
        assert_eq!(responses[0]["status"], 200);
        assert_eq!(responses[0]["hits"]["total"]["value"], 1);
        assert_eq!(responses[0]["hits"]["hits"][0]["_id"], "1");
        assert!(responses[0]["hits"]["hits"][0]["fields"]["created_at"][0].is_i64());

        // Verify Query 2 aggregations response
        assert_eq!(responses[1]["status"], 200);
        assert_eq!(responses[1]["hits"]["total"]["value"], 2);
        let cat_buckets = responses[1]["aggregations"]["by_cat"]["buckets"]
            .as_array()
            .unwrap();
        assert_eq!(cat_buckets.len(), 2);
        let time_buckets = responses[1]["aggregations"]["over_time"]["buckets"]
            .as_array()
            .unwrap();
        assert_eq!(time_buckets.len(), 2);

        // Verify Query 3 error item
        assert_eq!(responses[2]["status"], 404);
        assert_eq!(
            responses[2]["error"]["type"],
            "index_not_found_exception"
        );
    }
}

#[test]
fn test_discover_dictionaries() {
    let temp_dir = TempDir::new().unwrap();
    let dir = temp_dir.path();

    // Create various .dic and non-.dic files
    std::fs::write(dir.join("readme.txt"), "info").unwrap();
    std::fs::write(dir.join("system.dic"), b"fake_sys").unwrap();
    std::fs::write(dir.join("user_b.dic"), b"fake_u2").unwrap();
    std::fs::write(dir.join("user_a.dic"), b"fake_u1").unwrap();
    std::fs::write(dir.join("system_core.dic"), b"fake_core").unwrap();

    let discovered = discover_dictionaries(dir).unwrap();
    // system.dic has highest priority (0) over system_core.dic (1)
    assert_eq!(
        discovered.system_dict.as_deref(),
        Some(dir.join("system.dic").as_path())
    );

    // Other .dic files should be treated as user dicts and sorted alphabetically
    let user_names: Vec<String> = discovered
        .user_dicts
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    assert_eq!(
        user_names,
        vec!["system_core.dic".to_string(), "user_a.dic".to_string(), "user_b.dic".to_string()]
    );
}

#[tokio::test]
async fn test_user_dictionary_from_dir_and_rest_analyze() {
    use sudachi::dic::binary_loader::LoadedDictionary;
    use sudachi::dic::build::DictBuilder;

    let dict_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();

    // 1. Compile minimal system dictionary and write to system.dic
    let sys_bytes = build_fallback_system_dic_bytes().unwrap();
    let sys_dic_path = dict_dir.path().join("system.dic");
    std::fs::write(&sys_dic_path, &sys_bytes).unwrap();

    // 2. Build a user dictionary based on the compiled system dictionary
    let loaded_sys = LoadedDictionary::load_system(&sys_bytes).unwrap();
    let mut user_builder = DictBuilder::new_user(&loaded_sys);

    // Add a custom domain word "たんぽぽ検索" as a single noun
    let user_csv = concat!(
        "たんぽぽ検索,1,1,1000,たんぽぽ検索,名詞,固有名詞,一般,*,*,*,タンポポケンサク,たんぽぽ検索,*,A,*,*,*,*\n",
        "超高速検索,1,1,1000,超高速検索,名詞,固有名詞,一般,*,*,*,チョウコウソクケンサク,超高速検索,*,A,*,*,*,*\n"
    );
    user_builder.read_lexicon(user_csv.as_bytes()).unwrap();
    user_builder.resolve().unwrap();

    let mut user_bytes = Vec::new();
    user_builder.compile(&mut user_bytes).unwrap();
    let user_dic_path = dict_dir.path().join("custom_user.dic");
    std::fs::write(&user_dic_path, user_bytes).unwrap();

    // 3. Load dictionary from directory (simulating TANPOPO_SUDACHI_DICT_DIR)
    let dict = load_dictionary_from_dir(Some(dict_dir.path()), None).unwrap();

    // Verify tokenizer recognizes user dictionary words
    let tok = SudachiTokenizer::new(
        dict.clone(),
        SudachiTokenizerConfig::with_mode(SplitMode::C),
    );
    let analyzed = tok.analyze("たんぽぽ検索で超高速検索を実現する");
    let tokens: Vec<String> = analyzed.into_iter().map(|t| t.token).collect();

    // "たんぽぽ検索" and "超高速検索" should each be tokenized as a single word from user dict
    assert!(
        tokens.contains(&"たんぽぽ検索".to_string()),
        "Tokens should contain user dict word 'たんぽぽ検索': {:?}",
        tokens
    );
    assert!(
        tokens.contains(&"超高速検索".to_string()),
        "Tokens should contain user dict word '超高速検索': {:?}",
        tokens
    );

    // 4. Verify end-to-end via Tanpopo REST API `POST /_analyze`
    let mgr = Arc::new(IndexManager::new(data_dir.path().to_path_buf(), dict, 15_000_000).unwrap());
    let app = build_router(mgr);

    let (status, res) = send_json(
        &app,
        Method::POST,
        "/_analyze",
        Some(json!({
            "text": "たんぽぽ検索を導入",
            "mode": "C"
        })),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let resp_tokens = res["tokens"].as_array().unwrap();
    assert_eq!(resp_tokens[0]["token"], "たんぽぽ検索");
    assert_eq!(resp_tokens[0]["normalized_form"], "たんぽぽ検索");

    // 5. Verify index creation, indexing, and search with user dictionary vocabulary
    let (status, _) = send_json(
        &app,
        Method::PUT,
        "/articles",
        Some(json!({
            "mappings": {
                "properties": {
                    "title": { "type": "text", "analyzer": "sudachi" }
                }
            }
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, _) = send_json(
        &app,
        Method::POST,
        "/articles/_doc/1",
        Some(json!({ "title": "たんぽぽ検索の導入事例について" })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let (status, _) = send_json(
        &app,
        Method::POST,
        "/articles/_doc/2",
        Some(json!({ "title": "一般的な全文検索エンジンの比較" })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    send_json(&app, Method::POST, "/articles/_refresh", None).await;

    let (status, search_res) = send_json(
        &app,
        Method::POST,
        "/articles/_search",
        Some(json!({
            "query": { "match": { "title": "たんぽぽ検索" } }
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(search_res["hits"]["total"]["value"], 1);
    assert_eq!(search_res["hits"]["hits"][0]["_id"], "1");
}
