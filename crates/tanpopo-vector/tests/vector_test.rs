use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use axum::routing::post;
use axum::{Json, Router};
use serde_json::{Value, json};
use tanpopo_vector::{
    ChunkConfig, ChunkVectorInput, EmbedConfig, EmbedPreset, Embedder, HttpEmbedder,
    MmapVectorStore, chunk_document_json, chunk_text_by_lines,
};
use tempfile::TempDir;

#[test]
fn test_line_aware_chunking() {
    let md = "\
# 第1章 概要
関西国際空港は大阪湾の人工島に建設された空港です。
24時間運用可能な国際拠点空港として機能しています。

# 第2章 耐震設計
護岸および連絡橋の耐震設計基準について記述します。
基準地震動 Ss に対する構造健全性を確認しています。
地盤改良工事と液状化対策の詳細データ。";

    let chunks = chunk_text_by_lines(
        md,
        ChunkConfig {
            max_chars: 75,
            overlap_lines: 0,
        },
    );
    assert!(chunks.len() >= 2);
    assert_eq!(chunks[0].start_line, 1);
    assert!(chunks[0].text.contains("第1章"));

    let sec2 = chunks.iter().find(|c| c.text.contains("第2章")).unwrap();
    assert_eq!(sec2.start_line, 5);
    assert!(sec2.end_line >= 6);
}

#[test]
fn test_ruri_and_gemma_presets() {
    let ruri = EmbedPreset::from_model_name("cl-nagoya/ruri-v3-310m");
    assert_eq!(ruri, EmbedPreset::Ruri);
    assert_eq!(ruri.format_query("耐震設計"), "検索クエリ: 耐震設計");
    assert_eq!(
        ruri.format_passage(Some("川内原発審査資料"), "基準地震動の評価"),
        "検索文書: 川内原発審査資料\n基準地震動の評価"
    );

    let gemma = EmbedPreset::from_model_name("google/embeddinggemma-300m");
    assert_eq!(gemma, EmbedPreset::Gemma);
    assert_eq!(
        gemma.format_query("耐震設計"),
        "task: search result | query: 耐震設計"
    );
    assert_eq!(
        gemma.format_passage(Some("川内原発審査資料"), "基準地震動の評価"),
        "title: 川内原発審査資料 | text: 基準地震動の評価"
    );
}

#[test]
fn test_mmap_vector_store_sq8_maxp_persistence_and_compaction() {
    let tmp = TempDir::new().unwrap();

    // 1. Open store and insert multi-chunk documents
    {
        let store = MmapVectorStore::open(tmp.path()).unwrap();
        store
            .upsert_doc(
                "doc1",
                &[
                    ChunkVectorInput {
                        vector: vec![1.0, 0.0, 0.0, 0.0],
                        start_line: 1,
                        end_line: 30,
                    },
                    ChunkVectorInput {
                        vector: vec![0.0, 1.0, 0.0, 0.0],
                        start_line: 31,
                        end_line: 65,
                    },
                ],
            )
            .unwrap();

        store
            .upsert_doc(
                "doc2",
                &[ChunkVectorInput {
                    vector: vec![0.0, 0.0, 1.0, 0.0],
                    start_line: 1,
                    end_line: 20,
                }],
            )
            .unwrap();

        assert_eq!(store.active_doc_count(), 2);
        assert_eq!(store.active_vector_count(), 3);

        // Query close to doc1's second chunk (lines 31..=65)
        let hits = store.search(&[0.05, 0.98, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc1");
        assert_eq!(hits[0].start_line, 31);
        assert_eq!(hits[0].end_line, 65);
        assert!(hits[0].score > 0.95);

        // Pre-filter to only allow doc2
        let mut allowed = HashSet::new();
        allowed.insert("doc2".to_string());
        let filtered_hits = store
            .search(&[0.05, 0.98, 0.0, 0.0], 10, Some(&allowed))
            .unwrap();
        assert_eq!(filtered_hits.len(), 1);
        assert_eq!(filtered_hits[0].doc_id, "doc2");
    }

    // 2. Reopen from disk and test update + delete + compaction
    {
        let store = MmapVectorStore::open(tmp.path()).unwrap();
        assert_eq!(store.dim(), 4);
        assert_eq!(store.active_doc_count(), 2);
        assert_eq!(store.active_vector_count(), 3);

        // Re-upsert doc1 with a single updated chunk
        store
            .upsert_doc(
                "doc1",
                &[ChunkVectorInput {
                    vector: vec![0.0, 0.0, 0.0, 1.0],
                    start_line: 1,
                    end_line: 15,
                }],
            )
            .unwrap();
        assert_eq!(store.active_vector_count(), 2);

        // Delete doc2
        assert!(store.delete_doc("doc2").unwrap());
        assert_eq!(store.active_doc_count(), 1);
        assert_eq!(store.active_vector_count(), 1);

        // Compact and verify
        store.compact().unwrap();
        let hits = store.search(&[0.0, 0.0, 0.0, 1.0], 10, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc1");
        assert_eq!(hits[0].start_line, 1);
        assert_eq!(hits[0].end_line, 15);
    }
}

#[tokio::test]
async fn test_http_embedder_with_batching() {
    let seen_inputs: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_clone = seen_inputs.clone();

    let app = Router::new().route(
        "/v1/embeddings",
        post(move |Json(body): Json<Value>| {
            let seen = seen_clone.clone();
            async move {
                let arr = body["input"].as_array().unwrap();
                let mut data = Vec::new();
                for (i, item) in arr.iter().enumerate() {
                    let s = item.as_str().unwrap().to_string();
                    seen.lock().unwrap().push(s);
                    data.push(json!({
                        "object": "embedding",
                        "index": i,
                        "embedding": [1.0, (i as f32) + 0.5, 0.0, 0.0]
                    }));
                }
                Json(json!({
                    "object": "list",
                    "data": data,
                    "model": body["model"],
                }))
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let mut cfg = EmbedConfig::new(
        format!("http://{addr}/v1/embeddings"),
        "cl-nagoya/ruri-v3-310m",
    );
    cfg.batch_size = 2;
    let embedder = HttpEmbedder::new(cfg);

    let q_vec = embedder.embed_query("MCNP 解析").await.unwrap();
    assert_eq!(q_vec.len(), 4);

    let doc = json!({
        "title": "遮蔽解析説明書",
        "content": "第1行\n第2行"
    });
    let (title, chunks) = chunk_document_json(&doc, embedder.chunk_config());
    let texts: Vec<String> = chunks.into_iter().map(|c| c.text).collect();
    let p_vecs = embedder
        .embed_passages(title.as_deref(), &texts)
        .await
        .unwrap();
    assert_eq!(p_vecs.len(), 1);

    let log = seen_inputs.lock().unwrap();
    assert_eq!(log[0], "検索クエリ: MCNP 解析");
    assert_eq!(log[1], "検索文書: 遮蔽解析説明書\n第1行\n第2行");
}
