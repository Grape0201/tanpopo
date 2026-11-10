use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use serde_json::{Map, Value, json};
use tanpopo_bm25::index::{AppError, CreateIndexRequest, IndexManager, ManagedIndex};
use tanpopo_bm25::query::{
    SearchRequest, evaluate_filter_doc_ids, execute_search_arc, extract_search_text,
    filter_and_truncate_source, format_span_snippet,
};
use tanpopo_vector::{
    ChunkVectorInput, Embedder, MmapVectorStore, chunk_document_json,
};

use crate::fusion::{
    FusedCandidate, FusionStrategy, RankedCandidate, linear_combination_fusion,
    reciprocal_rank_fusion,
};

/// Coordinates `tanpopo-bm25` (Sudachi + Tantivy) and per-index `MmapVectorStore` instances.
pub struct HybridManager {
    bm25: Arc<IndexManager>,
    embedder: Option<Arc<dyn Embedder>>,
    vector_stores: RwLock<BTreeMap<String, Arc<MmapVectorStore>>>,
}

impl HybridManager {
    pub fn new(bm25: Arc<IndexManager>, embedder: Option<Arc<dyn Embedder>>) -> Result<Self, AppError> {
        let mgr = Self {
            bm25,
            embedder,
            vector_stores: RwLock::new(BTreeMap::new()),
        };
        // Pre-open existing vector stores for all existing indices
        for idx in mgr.bm25.resolve_indices(None)? {
            let _ = mgr.get_or_open_vector_store(&idx.name)?;
        }
        Ok(mgr)
    }

    pub fn bm25(&self) -> &Arc<IndexManager> {
        &self.bm25
    }

    pub fn has_embedder(&self) -> bool {
        self.embedder.is_some()
    }

    pub fn get_or_open_vector_store(&self, index_name: &str) -> Result<Arc<MmapVectorStore>, AppError> {
        if let Ok(guard) = self.vector_stores.read() {
            if let Some(store) = guard.get(index_name) {
                return Ok(store.clone());
            }
        }
        let dir_path = self.bm25.data_dir().join(index_name);
        let store = Arc::new(
            MmapVectorStore::open(&dir_path).map_err(|e| AppError::Internal(e.to_string()))?,
        );
        if let Ok(mut guard) = self.vector_stores.write() {
            guard.insert(index_name.to_string(), store.clone());
        }
        Ok(store)
    }

    pub fn create_index(
        &self,
        name: &str,
        req: CreateIndexRequest,
    ) -> Result<Arc<ManagedIndex>, AppError> {
        let managed = self.bm25.create_index(name, req)?;
        let _ = self.get_or_open_vector_store(name)?;
        Ok(managed)
    }

    pub fn delete_index(&self, name: &str) -> Result<(), AppError> {
        if let Ok(mut guard) = self.vector_stores.write() {
            guard.remove(name);
        }
        self.bm25.delete_index(name)
    }

    pub fn refresh_index(&self, name: &str) -> Result<(), AppError> {
        let managed = self.bm25.get_index(name)?;
        managed.refresh()?;
        if let Ok(store) = self.get_or_open_vector_store(name) {
            store
                .compact_if_needed()
                .map_err(|e| AppError::Internal(e.to_string()))?;
        }
        Ok(())
    }

    /// Indexes a document into `tanpopo-bm25` and, if an `Embedder` is configured,
    /// chunks the document by lines, embeds all chunks, and stores them in `MmapVectorStore`.
    pub async fn index_document(
        &self,
        index_name: &str,
        id: Option<String>,
        doc_json: Value,
        refresh: bool,
    ) -> Result<(String, bool), AppError> {
        let managed = self.bm25.get_or_create_from_doc(index_name, &doc_json)?;
        let (doc_id, created) = managed.index_document(id, doc_json.clone(), refresh)?;

        if let Some(ref embedder) = self.embedder {
            let chunk_inputs = self.embed_single_doc(embedder.as_ref(), &doc_json).await?;
            if !chunk_inputs.is_empty() {
                let store = self.get_or_open_vector_store(index_name)?;
                store
                    .upsert_doc(&doc_id, &chunk_inputs)
                    .map_err(|e| AppError::Internal(e.to_string()))?;
            }
        }

        Ok((doc_id, created))
    }

    pub fn delete_document(
        &self,
        index_name: &str,
        id: &str,
        refresh: bool,
    ) -> Result<bool, AppError> {
        let managed = self.bm25.get_index(index_name)?;
        let existed = managed.delete_document(id, refresh)?;
        if let Ok(store) = self.get_or_open_vector_store(index_name) {
            let _ = store.delete_doc(id);
            if refresh {
                let _ = store.compact_if_needed();
            }
        }
        Ok(existed)
    }

    /// Executes an Elasticsearch-compatible `_bulk` request across both BM25 and Vector stores.
    /// Batches chunk embeddings across documents per index for high throughput.
    pub async fn execute_bulk(
        &self,
        default_index: Option<&str>,
        body: &str,
        refresh: bool,
    ) -> Result<Value, AppError> {
        let start = Instant::now();
        let mut lines = body.lines().filter(|l| !l.trim().is_empty());
        let mut items = Vec::new();
        let mut has_errors = false;
        let mut touched_indices: BTreeMap<String, Arc<ManagedIndex>> = BTreeMap::new();
        let mut pending_docs_per_index: BTreeMap<String, Vec<(String, Value)>> = BTreeMap::new();

        while let Some(action_line) = lines.next() {
            let action_json: Value = serde_json::from_str(action_line)
                .map_err(|e| AppError::BadRequest(format!("invalid bulk action JSON: {e}")))?;
            let action_obj = action_json
                .as_object()
                .and_then(|o| o.iter().next())
                .ok_or_else(|| AppError::BadRequest("empty bulk action object".into()))?;

            let (op_type, meta_val) = action_obj;
            let target_index = meta_val
                .get("_index")
                .and_then(|v| v.as_str())
                .or(default_index)
                .ok_or_else(|| AppError::BadRequest("bulk operation missing _index".into()))?;
            let target_id = meta_val
                .get("_id")
                .and_then(|v| v.as_str())
                .map(String::from);

            match op_type.as_str() {
                "index" | "create" => {
                    let Some(doc_line) = lines.next() else {
                        return Err(AppError::BadRequest(
                            "unexpected end of bulk body after index/create action".into(),
                        ));
                    };
                    let doc_val: Value = serde_json::from_str(doc_line)
                        .map_err(|e| AppError::BadRequest(format!("invalid bulk doc JSON: {e}")))?;

                    let managed = if let Some(idx) = touched_indices.get(target_index) {
                        idx.clone()
                    } else {
                        let idx = self.bm25.get_or_create_from_doc(target_index, &doc_val)?;
                        touched_indices.insert(target_index.to_string(), idx.clone());
                        idx
                    };

                    match managed.index_document(target_id, doc_val.clone(), false) {
                        Ok((doc_id, created)) => {
                            if self.embedder.is_some() {
                                pending_docs_per_index
                                    .entry(target_index.to_string())
                                    .or_default()
                                    .push((doc_id.clone(), doc_val));
                            }
                            items.push(json!({
                                op_type: {
                                    "_index": target_index,
                                    "_id": doc_id,
                                    "status": if created { 201 } else { 200 },
                                    "result": if created { "created" } else { "updated" },
                                }
                            }));
                        }
                        Err(e) => {
                            has_errors = true;
                            items.push(json!({
                                op_type: {
                                    "_index": target_index,
                                    "status": 400,
                                    "error": { "type": "mapper_parsing_exception", "reason": e.to_string() }
                                }
                            }));
                        }
                    }
                }
                "delete" => {
                    let Some(doc_id) = target_id else {
                        return Err(AppError::BadRequest("bulk delete requires _id".into()));
                    };
                    let managed = if let Some(idx) = touched_indices.get(target_index) {
                        idx.clone()
                    } else {
                        let idx = self.bm25.get_index(target_index)?;
                        touched_indices.insert(target_index.to_string(), idx.clone());
                        idx
                    };
                    let _ = managed.delete_document(&doc_id, false)?;
                    if let Ok(store) = self.get_or_open_vector_store(target_index) {
                        let _ = store.delete_doc(&doc_id);
                    }
                    items.push(json!({
                        "delete": {
                            "_index": target_index,
                            "_id": doc_id,
                            "status": 200,
                            "result": "deleted"
                        }
                    }));
                }
                other => {
                    return Err(AppError::BadRequest(format!(
                        "unsupported bulk action [{other}]"
                    )));
                }
            }
        }

        if let Some(ref embedder) = self.embedder {
            for (index_name, docs) in pending_docs_per_index {
                let mut batch_upserts = Vec::with_capacity(docs.len());
                for (doc_id, doc_val) in docs {
                    let chunk_vecs = self.embed_single_doc(embedder.as_ref(), &doc_val).await?;
                    if !chunk_vecs.is_empty() {
                        batch_upserts.push((doc_id, chunk_vecs));
                    }
                }
                if !batch_upserts.is_empty() {
                    let store = self.get_or_open_vector_store(&index_name)?;
                    store
                        .upsert_batch(&batch_upserts)
                        .map_err(|e| AppError::Internal(e.to_string()))?;
                }
            }
        }

        if refresh {
            for name in touched_indices.keys() {
                self.refresh_index(name)?;
            }
        }

        Ok(json!({
            "took": start.elapsed().as_millis() as u64,
            "errors": has_errors,
            "items": items,
        }))
    }

    async fn embed_single_doc(
        &self,
        embedder: &dyn Embedder,
        doc_json: &Value,
    ) -> Result<Vec<ChunkVectorInput>, AppError> {
        let (title, chunks) = chunk_document_json(doc_json, embedder.chunk_config());
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let passages: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let vectors = embedder
            .embed_passages(title.as_deref(), &passages)
            .await
            .map_err(|e| AppError::Internal(e.to_string()))?;

        let out = chunks
            .into_iter()
            .zip(vectors.into_iter())
            .map(|(chunk, vector)| ChunkVectorInput {
                vector,
                start_line: chunk.start_line,
                end_line: chunk.end_line,
            })
            .collect();
        Ok(out)
    }

    /// Executes a unified hybrid search across `tanpopo-bm25` and `MmapVectorStore`.
    /// Automatically falls back to pure BM25 when no `Embedder` is configured, when no vectors exist,
    /// when a custom field `sort` is specified, or when `mode == "bm25"`.
    pub async fn search(
        &self,
        index_pattern: Option<&str>,
        req_val: Value,
        q_param: Option<&str>,
        from_param: Option<usize>,
        size_param: Option<usize>,
    ) -> Result<Value, AppError> {
        let start = Instant::now();
        let indices = self.bm25.resolve_indices(index_pattern)?;
        let strategy = FusionStrategy::from_request_json(&req_val);

        let mut bm25_req: SearchRequest = if req_val.is_null() {
            SearchRequest::default()
        } else {
            serde_json::from_value(req_val.clone())
                .map_err(|e| AppError::BadRequest(format!("invalid search request JSON: {e}")))?
        };
        if bm25_req.from.is_none() {
            bm25_req.from = from_param;
        }
        if bm25_req.size.is_none() {
            bm25_req.size = size_param;
        }

        let from = bm25_req.from.unwrap_or(0);
        let size = bm25_req.size.unwrap_or(10);
        let has_custom_sort = bm25_req
            .sort
            .as_ref()
            .is_some_and(|s| !s.is_null() && s.as_array().map_or(true, |a| !a.is_empty()));

        let search_text = extract_search_text(bm25_req.query.as_ref(), q_param);

        // Check if any target index actually has vectors
        let has_any_vectors = indices.iter().any(|idx| {
            self.get_or_open_vector_store(&idx.name)
                .map(|s| s.active_vector_count() > 0)
                .unwrap_or(false)
        });

        // Fast path: Pure BM25 when embedder is absent, strategy is Bm25, size == 0, custom sort, or no vectors/query text
        let Some(ref embedder) = self.embedder else {
            return execute_search_arc(&indices, bm25_req, q_param);
        };
        if strategy == FusionStrategy::Bm25
            || size == 0
            || has_custom_sort
            || !has_any_vectors
            || search_text.is_none()
        {
            return execute_search_arc(&indices, bm25_req, q_param);
        }

        let query_text = search_text.unwrap();
        let window_size = (from + size).max(50);

        // 1. Embed query text & run BM25 candidate retrieval
        let query_vec = embedder
            .embed_query(&query_text)
            .await
            .map_err(|e| AppError::Internal(e.to_string()))?;

        let mut candidate_bm25_req = bm25_req.clone();
        candidate_bm25_req.from = Some(0);
        candidate_bm25_req.size = Some(window_size);
        let mut bm25_resp = execute_search_arc(&indices, candidate_bm25_req, q_param)?;

        let bm25_candidates: Vec<RankedCandidate> = if strategy == FusionStrategy::Vector {
            Vec::new()
        } else {
            extract_bm25_candidates(&bm25_resp)
        };

        // 2. Run Vector search across resolved indices (with Tantivy filter pushdown if present)
        let mut vector_candidates = Vec::new();
        for idx in &indices {
            let store = self.get_or_open_vector_store(&idx.name)?;
            if store.active_vector_count() == 0 {
                continue;
            }
            let filter_ids = evaluate_filter_doc_ids(idx, bm25_req.query.as_ref())?;
            let v_hits = store
                .search(&query_vec, window_size, filter_ids.as_ref())
                .map_err(|e| AppError::Internal(e.to_string()))?;
            for vh in v_hits {
                vector_candidates.push(RankedCandidate {
                    index_name: idx.name.clone(),
                    doc_id: vh.doc_id,
                    score: vh.score,
                    start_line: vh.start_line,
                    end_line: vh.end_line,
                    bm25_hit: None,
                });
            }
        }
        if indices.len() > 1 {
            vector_candidates.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            vector_candidates.truncate(window_size);
        }

        // 3. Fuse BM25 and Vector rankings
        let fused_all: Vec<FusedCandidate> = match strategy {
            FusionStrategy::Rrf { k } => {
                reciprocal_rank_fusion(&bm25_candidates, &vector_candidates, k, from + size)
            }
            FusionStrategy::LinearCombination { alpha } => {
                linear_combination_fusion(&bm25_candidates, &vector_candidates, alpha, from + size)
            }
            FusionStrategy::Vector => vector_candidates
                .into_iter()
                .enumerate()
                .take(from + size)
                .map(|(i, c)| FusedCandidate {
                    index_name: c.index_name,
                    doc_id: c.doc_id,
                    fused_score: c.score as f64,
                    bm25_rank: None,
                    bm25_score: None,
                    vector_rank: Some(i + 1),
                    vector_score: Some(c.score),
                    matched_lines: (c.start_line > 0)
                        .then_some((c.start_line, c.end_line.max(c.start_line))),
                    bm25_hit: None,
                })
                .collect(),
            FusionStrategy::Bm25 => unreachable!(),
        };

        let max_score = fused_all.first().map(|f| f.fused_score);
        let index_map: BTreeMap<&str, &Arc<ManagedIndex>> =
            indices.iter().map(|i| (i.name.as_str(), i)).collect();
        let highlight_enabled = bm25_req
            .highlight
            .as_ref()
            .is_some_and(|v| v.as_bool().unwrap_or(true));

        let mut final_hits = Vec::new();
        for f in fused_all.into_iter().skip(from).take(size) {
            let mut hit_obj = if let Some(existing_hit) = f.bm25_hit {
                existing_hit
            } else {
                // Document was found via vector similarity; load _source from Tantivy
                let Some(managed) = index_map.get(f.index_name.as_str()) else {
                    continue;
                };
                let Some(raw_source) = managed.get_document(&f.doc_id)? else {
                    continue;
                };
                let mut h = json!({
                    "_index": f.index_name,
                    "_id": f.doc_id,
                });
                if highlight_enabled {
                    if let Some((start_l, end_l)) = f.matched_lines {
                        if let Some((field_name, snippet)) =
                            format_span_snippet(&raw_source, start_l, end_l, 240)
                        {
                            let mut hl_map = Map::new();
                            hl_map.insert(field_name, json!([snippet]));
                            h["highlight"] = Value::Object(hl_map);
                        }
                    }
                }
                if let Some(filtered_source) = filter_and_truncate_source(
                    &raw_source,
                    bm25_req.source.as_ref(),
                    bm25_req.exclude_fields.as_deref(),
                    bm25_req.max_field_chars,
                ) {
                    h["_source"] = filtered_source;
                }
                h
            };

            // If BM25 hit didn't produce highlights but vector matched a line span, attach vector span snippet
            if highlight_enabled && hit_obj.get("highlight").is_none() {
                if let Some((start_l, end_l)) = f.matched_lines {
                    if let Some(managed) = index_map.get(f.index_name.as_str()) {
                        if let Ok(Some(raw_source)) = managed.get_document(&f.doc_id) {
                            if let Some((field_name, snippet)) =
                                format_span_snippet(&raw_source, start_l, end_l, 240)
                            {
                                let mut hl_map = Map::new();
                                hl_map.insert(field_name, json!([snippet]));
                                hit_obj["highlight"] = Value::Object(hl_map);
                            }
                        }
                    }
                }
            }

            hit_obj["_score"] = json!(f.fused_score);
            hit_obj["_hybrid"] = json!({
                "bm25_rank": f.bm25_rank,
                "bm25_score": f.bm25_score,
                "vector_rank": f.vector_rank,
                "vector_score": f.vector_score,
                "matched_lines": f.matched_lines.map(|(s, e)| vec![s, e]),
            });
            final_hits.push(hit_obj);
        }

        let bm25_total = bm25_resp["hits"]["total"]["value"].as_u64().unwrap_or(0) as usize;
        let total_val = bm25_total.max(final_hits.len() + from);

        bm25_resp["took"] = json!(start.elapsed().as_millis() as u64);
        bm25_resp["hits"]["total"]["value"] = json!(total_val);
        bm25_resp["hits"]["max_score"] = json!(max_score);
        bm25_resp["hits"]["hits"] = Value::Array(final_hits);

        Ok(bm25_resp)
    }
}

fn extract_bm25_candidates(resp: &Value) -> Vec<RankedCandidate> {
    let mut out = Vec::new();
    let Some(arr) = resp
        .get("hits")
        .and_then(|h| h.get("hits"))
        .and_then(|a| a.as_array())
    else {
        return out;
    };
    for hit in arr {
        let index_name = hit
            .get("_index")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let doc_id = hit
            .get("_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let score = hit
            .get("_score")
            .and_then(|v| v.as_f64())
            .map(|s| s as f32)
            .unwrap_or(0.0);
        out.push(RankedCandidate {
            index_name,
            doc_id,
            score,
            start_line: 0,
            end_line: 0,
            bm25_hit: Some(hit.clone()),
        });
    }
    out
}
