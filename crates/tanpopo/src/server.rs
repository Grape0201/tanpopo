use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use tanpopo_bm25::index::{AppError, CreateIndexRequest};
use tanpopo_bm25::server::{AnalyzeRequest, UI_HTML, run_analyze};

use crate::engine::HybridManager;

pub type SharedState = Arc<HybridManager>;

async fn ui_handler() -> axum::response::Html<&'static str> {
    axum::response::Html(UI_HTML)
}

pub fn build_router(state: SharedState) -> Router {
    build_router_with_ui(state, true)
}

pub fn build_router_with_ui(state: SharedState, ui_enabled: bool) -> Router {
    let router = if ui_enabled {
        Router::new().route("/_ui", get(ui_handler))
    } else {
        Router::new()
    };
    router
        .route("/", get(root_handler))
        .route("/mcp", post(mcp_http_handler))
        .route("/_cluster/health", get(cluster_health_handler))
        .route("/_cluster/health/{index}", get(cluster_health_handler))
        .route("/_cat/indices", get(cat_indices_handler))
        .route("/_mapping", get(get_all_mappings_handler))
        .route("/_mappings", get(get_all_mappings_handler))
        .route("/_analyze", post(analyze_global_handler).get(analyze_global_handler))
        .route("/_bulk", post(bulk_global_handler).put(bulk_global_handler))
        .route("/_search", get(search_global_handler).post(search_global_handler))
        .route("/_msearch", get(msearch_global_handler).post(msearch_global_handler))
        .route(
            "/{index}",
            put(create_index_handler)
                .get(get_index_handler)
                .delete(delete_index_handler),
        )
        .route("/{index}/_mapping", get(get_index_mapping_handler))
        .route("/{index}/_mappings", get(get_index_mapping_handler))
        .route("/{index}/_analyze", post(analyze_index_handler).get(analyze_index_handler))
        .route("/{index}/_refresh", post(refresh_index_handler).get(refresh_index_handler))
        .route("/{index}/_bulk", post(bulk_index_handler).put(bulk_index_handler))
        .route("/{index}/_doc", post(post_doc_auto_id_handler))
        .route(
            "/{index}/_doc/{id}",
            put(put_doc_handler)
                .post(put_doc_handler)
                .get(get_doc_handler)
                .delete(delete_doc_handler),
        )
        .route("/{index}/_search", get(search_handler).post(search_handler))
        .route(
            "/{index}/_msearch",
            get(msearch_index_handler).post(msearch_index_handler),
        )
        .layer(axum::middleware::map_response(add_elastic_header))
        .with_state(state)
}

async fn add_elastic_header(mut res: Response) -> Response {
    res.headers_mut().insert(
        "x-elastic-product",
        axum::http::HeaderValue::from_static("Elasticsearch"),
    );
    res
}

async fn mcp_http_handler(State(mgr): State<SharedState>, Json(payload): Json<Value>) -> Response {
    match crate::mcp::handle_mcp_value(&mgr, payload).await {
        Some(res) => Json(res).into_response(),
        None => StatusCode::ACCEPTED.into_response(),
    }
}

async fn root_handler(State(mgr): State<SharedState>) -> Json<Value> {
    Json(json!({
        "name": "tanpopo",
        "cluster_name": "tanpopo",
        "version": {
            "number": "8.10.0",
            "tanpopo_version": env!("CARGO_PKG_VERSION"),
            "build_flavor": "default",
            "engine": "tantivy+sudachi.rs+mmap-sq8",
            "hybrid_enabled": mgr.has_embedder(),
        },
        "tagline": "You Know, for Lightweight Japanese Hybrid Search"
    }))
}

async fn cluster_health_handler(State(mgr): State<SharedState>) -> Result<Json<Value>, AppError> {
    let indices = mgr.bm25().list_indices()?;
    let active_shards = indices.len();
    Ok(Json(json!({
        "cluster_name": "tanpopo",
        "status": "green",
        "timed_out": false,
        "number_of_nodes": 1,
        "number_of_data_nodes": 1,
        "active_primary_shards": active_shards,
        "active_shards": active_shards,
        "relocating_shards": 0,
        "initializing_shards": 0,
        "unassigned_shards": 0,
        "delayed_unassigned_shards": 0,
        "number_of_pending_tasks": 0,
        "number_of_in_flight_fetch": 0,
        "task_max_waiting_in_queue_millis": 0,
        "active_shards_percent_as_number": 100.0,
    })))
}

async fn cat_indices_handler(State(mgr): State<SharedState>) -> Result<Json<Value>, AppError> {
    let mut list = mgr.bm25().list_indices()?;
    for item in &mut list {
        if let Some(name) = item.get("index").and_then(|v| v.as_str()).map(String::from) {
            if let Ok(store) = mgr.get_or_open_vector_store(&name) {
                item["vectors.count"] = json!(store.active_vector_count());
                item["vectors.dim"] = json!(store.dim());
            }
        }
    }
    Ok(Json(Value::Array(list)))
}

async fn analyze_global_handler(
    State(mgr): State<SharedState>,
    Json(req): Json<AnalyzeRequest>,
) -> Result<Json<Value>, AppError> {
    run_analyze(mgr.bm25(), None, req)
}

async fn analyze_index_handler(
    State(mgr): State<SharedState>,
    Path(index): Path<String>,
    Json(req): Json<AnalyzeRequest>,
) -> Result<Json<Value>, AppError> {
    let managed = mgr.bm25().get_index(&index)?;
    run_analyze(mgr.bm25(), Some(&managed), req)
}

async fn create_index_handler(
    State(mgr): State<SharedState>,
    Path(index): Path<String>,
    body: Option<Json<CreateIndexRequest>>,
) -> Result<Json<Value>, AppError> {
    let req = body.map(|Json(r)| r).unwrap_or_default();
    mgr.create_index(&index, req)?;
    Ok(Json(json!({
        "acknowledged": true,
        "shards_acknowledged": true,
        "index": index,
    })))
}

async fn get_index_handler(
    State(mgr): State<SharedState>,
    Path(index): Path<String>,
) -> Result<Json<Value>, AppError> {
    let managed = mgr.bm25().get_index(&index)?;
    let (vec_docs, vec_chunks, vec_dim) = mgr
        .get_or_open_vector_store(&index)
        .map(|s| (s.active_doc_count(), s.active_vector_count(), s.dim()))
        .unwrap_or((0, 0, 0));
    Ok(Json(json!({
        index: {
            "settings": &managed.meta.settings,
            "mappings": &managed.meta.mappings,
            "docs_count": managed.doc_count(),
            "vector_docs_count": vec_docs,
            "vector_chunks_count": vec_chunks,
            "vector_dim": vec_dim,
        }
    })))
}

async fn delete_index_handler(
    State(mgr): State<SharedState>,
    Path(index): Path<String>,
) -> Result<Json<Value>, AppError> {
    mgr.delete_index(&index)?;
    Ok(Json(json!({ "acknowledged": true })))
}

async fn refresh_index_handler(
    State(mgr): State<SharedState>,
    Path(index): Path<String>,
) -> Result<Json<Value>, AppError> {
    mgr.refresh_index(&index)?;
    Ok(Json(json!({
        "_shards": { "total": 1, "successful": 1, "failed": 0 }
    })))
}

#[derive(Debug, Default, Deserialize)]
struct RefreshQuery {
    #[serde(default)]
    refresh: Option<String>,
}

impl RefreshQuery {
    fn should_refresh(&self, default: bool) -> bool {
        match self.refresh.as_deref() {
            Some("false") => false,
            Some("true" | "wait_for" | "") => true,
            _ => default,
        }
    }
}

async fn post_doc_auto_id_handler(
    State(mgr): State<SharedState>,
    Path(index): Path<String>,
    Query(q): Query<RefreshQuery>,
    Json(doc): Json<Value>,
) -> Result<(StatusCode, Json<Value>), AppError> {
    let (id, created) = mgr
        .index_document(&index, None, doc, q.should_refresh(true))
        .await?;
    let status = if created {
        StatusCode::CREATED
    } else {
        StatusCode::OK
    };
    Ok((
        status,
        Json(json!({
            "_index": index,
            "_id": id,
            "result": if created { "created" } else { "updated" },
        })),
    ))
}

async fn put_doc_handler(
    State(mgr): State<SharedState>,
    Path((index, id)): Path<(String, String)>,
    Query(q): Query<RefreshQuery>,
    Json(doc): Json<Value>,
) -> Result<(StatusCode, Json<Value>), AppError> {
    let (doc_id, created) = mgr
        .index_document(&index, Some(id), doc, q.should_refresh(true))
        .await?;
    let status = if created {
        StatusCode::CREATED
    } else {
        StatusCode::OK
    };
    Ok((
        status,
        Json(json!({
            "_index": index,
            "_id": doc_id,
            "result": if created { "created" } else { "updated" },
        })),
    ))
}

async fn get_doc_handler(
    State(mgr): State<SharedState>,
    Path((index, id)): Path<(String, String)>,
) -> Result<(StatusCode, Json<Value>), AppError> {
    let managed = mgr.bm25().get_index(&index)?;
    match managed.get_document(&id)? {
        Some(source) => Ok((
            StatusCode::OK,
            Json(json!({
                "_index": index,
                "_id": id,
                "found": true,
                "_source": source,
            })),
        )),
        None => Ok((
            StatusCode::NOT_FOUND,
            Json(json!({
                "_index": index,
                "_id": id,
                "found": false,
            })),
        )),
    }
}

async fn delete_doc_handler(
    State(mgr): State<SharedState>,
    Path((index, id)): Path<(String, String)>,
    Query(q): Query<RefreshQuery>,
) -> Result<(StatusCode, Json<Value>), AppError> {
    let existed = mgr.delete_document(&index, &id, q.should_refresh(true))?;
    let status = if existed {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    };
    Ok((
        status,
        Json(json!({
            "_index": index,
            "_id": id,
            "result": if existed { "deleted" } else { "not_found" },
        })),
    ))
}

async fn bulk_global_handler(
    State(mgr): State<SharedState>,
    Query(q): Query<RefreshQuery>,
    body: String,
) -> Result<Json<Value>, AppError> {
    let res = mgr.execute_bulk(None, &body, q.should_refresh(true)).await?;
    Ok(Json(res))
}

async fn bulk_index_handler(
    State(mgr): State<SharedState>,
    Path(index): Path<String>,
    Query(q): Query<RefreshQuery>,
    body: String,
) -> Result<Json<Value>, AppError> {
    let res = mgr
        .execute_bulk(Some(&index), &body, q.should_refresh(true))
        .await?;
    Ok(Json(res))
}

#[derive(Debug, Default, Deserialize)]
struct SearchQueryParams {
    #[serde(default)]
    q: Option<String>,
    #[serde(default)]
    from: Option<usize>,
    #[serde(default)]
    size: Option<usize>,
}

async fn search_handler(
    State(mgr): State<SharedState>,
    Path(index): Path<String>,
    Query(params): Query<SearchQueryParams>,
    body: String,
) -> Result<Json<Value>, AppError> {
    // Validate single index existence unless pattern has wildcards/commas
    if !index.contains('*') && !index.contains(',') && index != "_all" {
        let _ = mgr.bm25().get_index(&index)?;
    }
    let req_val: Value = if body.trim().is_empty() {
        Value::Null
    } else {
        serde_json::from_str(&body)
            .map_err(|e| AppError::BadRequest(format!("invalid search request JSON: {e}")))?
    };
    let result = mgr
        .search(
            Some(&index),
            req_val,
            params.q.as_deref(),
            params.from,
            params.size,
        )
        .await?;
    Ok(Json(result))
}

async fn search_global_handler(
    State(mgr): State<SharedState>,
    Query(params): Query<SearchQueryParams>,
    body: String,
) -> Result<Json<Value>, AppError> {
    let req_val: Value = if body.trim().is_empty() {
        Value::Null
    } else {
        serde_json::from_str(&body)
            .map_err(|e| AppError::BadRequest(format!("invalid search request JSON: {e}")))?
    };
    let result = mgr
        .search(
            None,
            req_val,
            params.q.as_deref(),
            params.from,
            params.size,
        )
        .await?;
    Ok(Json(result))
}

async fn get_all_mappings_handler(
    State(mgr): State<SharedState>,
) -> Result<Json<Value>, AppError> {
    let indices = mgr.bm25().resolve_indices(None)?;
    let mut out = serde_json::Map::with_capacity(indices.len());
    for idx in indices {
        out.insert(
            idx.name.clone(),
            json!({ "mappings": &idx.meta.mappings }),
        );
    }
    Ok(Json(Value::Object(out)))
}

async fn get_index_mapping_handler(
    State(mgr): State<SharedState>,
    Path(index): Path<String>,
) -> Result<Json<Value>, AppError> {
    let indices = mgr.bm25().resolve_indices(Some(&index))?;
    if indices.is_empty() && index != "_all" && index != "*" {
        return Err(AppError::IndexNotFound(index));
    }
    let mut out = serde_json::Map::with_capacity(indices.len());
    for idx in indices {
        out.insert(
            idx.name.clone(),
            json!({ "mappings": &idx.meta.mappings }),
        );
    }
    Ok(Json(Value::Object(out)))
}

async fn msearch_global_handler(
    State(mgr): State<SharedState>,
    body: String,
) -> Result<Json<Value>, AppError> {
    execute_msearch(&mgr, None, &body).await
}

async fn msearch_index_handler(
    State(mgr): State<SharedState>,
    Path(index): Path<String>,
    body: String,
) -> Result<Json<Value>, AppError> {
    execute_msearch(&mgr, Some(&index), &body).await
}

async fn execute_msearch(
    mgr: &HybridManager,
    default_index: Option<&str>,
    body: &str,
) -> Result<Json<Value>, AppError> {
    let start = Instant::now();
    let mut lines = body.lines().filter(|l| !l.trim().is_empty());
    let mut responses = Vec::new();

    while let Some(header_line) = lines.next() {
        let Some(body_line) = lines.next() else {
            return Err(AppError::BadRequest(
                "unexpected end of _msearch body after header line".into(),
            ));
        };
        let header: Value = serde_json::from_str(header_line)
            .map_err(|e| AppError::BadRequest(format!("invalid _msearch header JSON: {e}")))?;
        let req_val: Value = serde_json::from_str(body_line)
            .map_err(|e| AppError::BadRequest(format!("invalid _msearch request JSON: {e}")))?;

        let header_index = header
            .get("index")
            .or_else(|| header.get("indices"))
            .and_then(|v| match v {
                Value::String(s) => Some(s.clone()),
                Value::Array(arr) => {
                    let joined: Vec<&str> = arr
                        .iter()
                        .filter_map(|x| x.as_str())
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .collect();
                    Some(joined.join(","))
                }
                _ => None,
            })
            .filter(|s| !s.trim().is_empty());

        let target_pattern = header_index.as_deref().or(default_index);
        let ignore_unavailable = header
            .get("ignore_unavailable")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let indices = mgr.bm25().resolve_indices(target_pattern)?;
        if indices.is_empty()
            && let Some(pat) =
                target_pattern.filter(|p| !p.is_empty() && *p != "*" && *p != "_all")
            && !ignore_unavailable
        {
            responses.push(json!({
                "error": {
                    "type": "index_not_found_exception",
                    "reason": format!("index [{pat}] not found"),
                },
                "status": 404,
            }));
            continue;
        }

        match mgr.search(target_pattern, req_val, None, None, None).await {
            Ok(mut res) => {
                if let Some(obj) = res.as_object_mut() {
                    obj.insert("status".into(), json!(200));
                }
                responses.push(res);
            }
            Err(e) => {
                responses.push(json!({
                    "error": {
                        "type": "search_phase_execution_exception",
                        "reason": e.to_string(),
                    },
                    "status": 400,
                }));
            }
        }
    }

    Ok(Json(json!({
        "took": start.elapsed().as_millis() as u64,
        "responses": responses,
    })))
}
