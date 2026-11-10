use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Method, Request, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{Value, json};
use tanpopo::engine::HybridManager;
use tanpopo::index::IndexManager;
use tanpopo::server::build_router;
use tanpopo::tokenizer::build_fallback_dictionary;
use tanpopo::vector::{ChunkConfig, EmbedConfig, Embedder, HttpEmbedder};
use tempfile::TempDir;
use tower::ServiceExt;

async fn send_json(
    app: &Router,
    method: Method,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let body_str = body.map(|v| v.to_string()).unwrap_or_default();
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body_str))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let val: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, val)
}

/// Deterministic semantic mock embedding server:
/// - axis 0: aviation / airport ("空港", "関空", "伊丹", "フライト", "滑走路")
/// - axis 1: seismic / nuclear safety ("耐震", "地震", "活断層", "揺れ", "地盤")
/// - axis 2: shielding / radiation ("遮蔽", "MCNP", "放射線", "中性子")
/// - axis 3: baseline
fn semantic_vector_for_text(text: &str) -> [f32; 4] {
    let mut v = [0.01f32, 0.01, 0.01, 0.05];
    if text.contains("空港")
        || text.contains("関空")
        || text.contains("伊丹")
        || text.contains("フライト")
        || text.contains("滑走路")
    {
        v[0] += 1.0;
    }
    if text.contains("耐震")
        || text.contains("地震")
        || text.contains("活断層")
        || text.contains("揺れ")
        || text.contains("地盤")
    {
        v[1] += 1.0;
    }
    if text.contains("遮蔽")
        || text.contains("MCNP")
        || text.contains("放射線")
        || text.contains("中性子")
    {
        v[2] += 1.0;
    }
    v
}

async fn spawn_mock_embedding_server() -> String {
    let app = Router::new().route(
        "/v1/embeddings",
        post(|Json(body): Json<Value>| async move {
            let arr = body["input"].as_array().unwrap();
            let mut data = Vec::with_capacity(arr.len());
            for (idx, item) in arr.iter().enumerate() {
                let s = item.as_str().unwrap_or("");
                let emb = semantic_vector_for_text(s);
                data.push(json!({
                    "object": "embedding",
                    "index": idx,
                    "embedding": emb,
                }));
            }
            Json(json!({
                "object": "list",
                "data": data,
                "model": body["model"],
            }))
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}/v1/embeddings")
}

#[tokio::test]
async fn test_hybrid_search_rrf_chunk_maxp_filter_and_mcp() {
    let embed_url = spawn_mock_embedding_server().await;
    let tmp = TempDir::new().unwrap();
    let dict = build_fallback_dictionary().unwrap();
    let bm25_mgr = Arc::new(IndexManager::new(tmp.path().to_path_buf(), dict, 15_000_000).unwrap());

    let mut embed_cfg = EmbedConfig::new(embed_url, "cl-nagoya/ruri-v3-310m");
    embed_cfg.chunk_config = ChunkConfig {
        max_chars: 60,
        overlap_lines: 0,
    };
    let embedder: Arc<dyn Embedder> = Arc::new(HttpEmbedder::new(embed_cfg));
    let hybrid_mgr = Arc::new(HybridManager::new(bm25_mgr, Some(embedder)).unwrap());
    let app = build_router(hybrid_mgr);

    // 1. Create index
    let (status, _) = send_json(
        &app,
        Method::PUT,
        "/hearings",
        Some(json!({
            "mappings": {
                "properties": {
                    "title": { "type": "text", "analyzer": "sudachi_search" },
                    "category": { "type": "keyword" },
                    "content": { "type": "text", "analyzer": "sudachi_search" }
                }
            }
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // 2. Index doc1 (matches both BM25 "関西国際空港" and seismic vector in chunk 2)
    let doc1_content = "\
# 第1章 施設概要
関西国際空港の連絡橋に関する点検記録です。

# 第2章 地盤と揺れの評価
活断層による激しい揺れと液状化に対する安全性を検証しました。";

    let (status, _) = send_json(
        &app,
        Method::PUT,
        "/hearings/_doc/doc1",
        Some(json!({
            "title": "関西国際空港 点検報告",
            "category": "report",
            "content": doc1_content
        })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    // 3. Index doc2 (does NOT contain the word "関西国際空港" or "耐震", but mentions "地震" and "地盤" -> semantic match on seismic!)
    let (status, _) = send_json(
        &app,
        Method::PUT,
        "/hearings/_doc/doc2",
        Some(json!({
            "title": "川内発電所 構造審査",
            "category": "hearing",
            "content": "第1行: 概要説明\n第2行: 活断層に伴う地盤安定性評価を実施。"
        })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    // 4. Index doc3 (unrelated radiation shielding doc)
    let (status, _) = send_json(
        &app,
        Method::PUT,
        "/hearings/_doc/doc3",
        Some(json!({
            "title": "MCNP 遮蔽計算書",
            "category": "hearing",
            "content": "中性子およびガンマ線の遮蔽解析結果。"
        })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    // 5. Hybrid search: query = "関西国際空港の耐震"
    // - doc1 matches BM25 ("関西国際空港") AND matches seismic vector on lines 4-5!
    // - doc2 has 0 BM25 overlap with "関西国際空港の耐震", but matches the seismic vector!
    let (status, res) = send_json(
        &app,
        Method::POST,
        "/hearings/_search",
        Some(json!({
            "query": {
                "match": { "content": "関西国際空港の耐震" }
            },
            "highlight": true
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let hits = res["hits"]["hits"].as_array().unwrap();
    assert!(hits.len() >= 2);
    // doc1 must be rank #1 because it matched BOTH BM25 and Vector (RRF boost)
    assert_eq!(hits[0]["_id"], "doc1");
    assert_eq!(hits[0]["_hybrid"]["bm25_rank"], 1);
    assert!(hits[0]["_hybrid"]["vector_rank"].is_number());
    // doc1's matched_lines should point to the second chunk (# 第2章 地盤と揺れの評価, lines 4..=5)
    let matched_lines = hits[0]["_hybrid"]["matched_lines"].as_array().unwrap();
    assert_eq!(matched_lines[0], 4);
    assert_eq!(matched_lines[1], 5);

    // doc2 was found purely via dense vector similarity and includes a line-numbered highlight!
    let doc2_hit = hits.iter().find(|h| h["_id"] == "doc2").expect("doc2 in hybrid hits");
    assert!(doc2_hit["_hybrid"]["bm25_rank"].is_null());
    assert!(doc2_hit["_hybrid"]["vector_rank"].is_number());
    let doc2_hl = doc2_hit["highlight"]["content"][0].as_str().unwrap();
    assert!(doc2_hl.contains("[L1-L2]"));

    // 6. Test Tantivy Filter Pushdown into Vector Search (`category == "hearing"`)
    let (status, filtered_res) = send_json(
        &app,
        Method::POST,
        "/hearings/_search",
        Some(json!({
            "query": {
                "bool": {
                    "must": { "match": { "content": "耐震" } },
                    "filter": { "term": { "category": "hearing" } }
                }
            }
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let f_hits = filtered_res["hits"]["hits"].as_array().unwrap();
    // doc1 has category="report" so it must be excluded by the pre-filter!
    assert!(f_hits.iter().all(|h| h["_id"] != "doc1"));
    assert_eq!(f_hits[0]["_id"], "doc2");

    // 7. Test MCP list_indices and search tools
    let (status, mcp_list) = send_json(
        &app,
        Method::POST,
        "/mcp",
        Some(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "list_indices",
                "arguments": {}
            }
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let list_text = mcp_list["result"]["content"][0]["text"].as_str().unwrap();
    let list_val: Value = serde_json::from_str(list_text).unwrap();
    assert_eq!(list_val[0]["docs_count"], 3);
    assert_eq!(list_val[0]["vector_docs_count"], 3);
    assert_eq!(list_val[0]["vector_chunks_count"], 4); // doc1 has 2 chunks, doc2 has 1, doc3 has 1
    assert_eq!(list_val[0]["vector_dim"], 4);

    // 8. Test _bulk indexing + delete synchronization across BM25 and MmapVectorStore
    let bulk_ndjson = concat!(
        r#"{"index":{"_id":"doc4"}}"#, "\n",
        r#"{"title":"伊丹空港 滑走路整備","category":"report","content":"滑走路の舗装および耐震補強工事。"}"#, "\n",
        r#"{"delete":{"_id":"doc3"}}"#, "\n"
    );
    let bulk_req = Request::builder()
        .method(Method::POST)
        .uri("/hearings/_bulk")
        .header("content-type", "application/x-ndjson")
        .body(Body::from(bulk_ndjson))
        .unwrap();
    let bulk_resp = app.clone().oneshot(bulk_req).await.unwrap();
    assert_eq!(bulk_resp.status(), StatusCode::OK);

    let (status, idx_info) = send_json(&app, Method::GET, "/hearings", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(idx_info["hearings"]["docs_count"], 3); // doc1, doc2, doc4 (doc3 deleted)
    assert_eq!(idx_info["hearings"]["vector_docs_count"], 3);
}
