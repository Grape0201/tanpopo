use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use axum::routing::post;
use axum::{Json, Router};
use serde_json::{Value, json};
use tanpopo_vector::{
    DocText, EmbedPreset, EmbedConfig, Embedder, HttpEmbedder, MmapVectorStore,
    doc_representative_text,
};
use tempfile::TempDir;

#[test]
fn test_doc_representative_text() {
    // summary wins over content; title returned separately
    let d = json!({"title": "T", "summary": "要約", "content": "本文"});
    assert_eq!(
        doc_representative_text(&d, 100),
        Some(DocText { title: Some("T".into()), text: "要約".into() })
    );
    // content truncated by chars (multibyte safe)
    let d = json!({"title": "T", "content": "あいうえお"});
    assert_eq!(doc_representative_text(&d, 3).unwrap().text, "あいう");
    // title-only fallback, and empty doc
    assert_eq!(doc_representative_text(&json!({"title": "T"}), 10).unwrap().text, "T");
    assert!(doc_representative_text(&json!({"n": 1}), 10).is_none());
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
fn test_mmap_vector_store_sq8_persistence_and_compaction() {
    let tmp = TempDir::new().unwrap();

    {
        let store = MmapVectorStore::open(tmp.path()).unwrap();
        store.upsert_doc("doc1", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        store.upsert_doc("doc2", &[0.0, 0.0, 1.0, 0.0]).unwrap();
        assert_eq!(store.active_doc_count(), 2);

        let hits = store.search(&[0.05, 0.98, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc1");
        assert!(hits[0].score > 0.95);

        let mut allowed = HashSet::new();
        allowed.insert("doc2".to_string());
        let filtered = store
            .search(&[0.05, 0.98, 0.0, 0.0], 10, Some(&allowed))
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].doc_id, "doc2");
    }

    // Reopen from disk, then update + delete + compaction
    {
        let store = MmapVectorStore::open(tmp.path()).unwrap();
        assert_eq!(store.dim(), 4);
        assert_eq!(store.active_doc_count(), 2);

        store.upsert_doc("doc1", &[0.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(store.active_doc_count(), 2);

        assert!(store.delete_doc("doc2").unwrap());
        assert_eq!(store.active_doc_count(), 1);
        assert!(!store.delete_doc("doc2").unwrap());

        store.compact().unwrap();
        let hits = store.search(&[0.0, 0.0, 0.0, 1.0], 10, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc1");
    }

    // Replay after compaction/tombstones yields the same state
    let store = MmapVectorStore::open(tmp.path()).unwrap();
    assert_eq!(store.active_doc_count(), 1);
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

    let p_vecs = embedder
        .embed_documents(&[DocText {
            title: Some("遮蔽解析説明書".into()),
            text: "第1行\n第2行".into(),
        }])
        .await
        .unwrap();
    assert_eq!(p_vecs.len(), 1);

    let log = seen_inputs.lock().unwrap();
    assert_eq!(log[0], "検索クエリ: MCNP 解析");
    assert_eq!(log[1], "検索文書: 遮蔽解析説明書\n第1行\n第2行");
}
