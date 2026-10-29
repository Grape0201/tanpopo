use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use tantivy::schema::Term;

use crate::index::{AppError, CreateIndexRequest, IndexManager, ManagedIndex};
use crate::query::{SearchRequest, execute_search, execute_search_arc};
use crate::tokenizer::{SplitMode, SudachiTokenizer, SudachiTokenizerConfig, TokenForm};

pub type SharedState = Arc<IndexManager>;

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type) = match &self {
            AppError::IndexNotFound(_) => (StatusCode::NOT_FOUND, "index_not_found_exception"),
            AppError::DocumentNotFound(_) => (StatusCode::NOT_FOUND, "document_missing_exception"),
            AppError::IndexAlreadyExists(_) => {
                (StatusCode::BAD_REQUEST, "resource_already_exists_exception")
            }
            AppError::BadRequest(_) => (StatusCode::BAD_REQUEST, "illegal_argument_exception"),
            AppError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_server_error"),
        };
        let body = json!({
            "error": {
                "type": error_type,
                "reason": self.to_string(),
            },
            "status": status.as_u16(),
        });
        (status, Json(body)).into_response()
    }
}

pub fn build_router(state: SharedState) -> Router {
    Router::new()
        .route("/", get(root_handler))
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
        .route(
            "/{index}/_search",
            get(search_handler).post(search_handler),
        )
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

async fn root_handler() -> Json<Value> {
    Json(json!({
        "name": "tanpopo",
        "cluster_name": "tanpopo",
        "version": {
            "number": "8.10.0",
            "tanpopo_version": env!("CARGO_PKG_VERSION"),
            "build_flavor": "default",
            "engine": "tantivy+sudachi.rs",
        },
        "tagline": "You Know, for Lightweight Japanese Search"
    }))
}

async fn cluster_health_handler(State(mgr): State<SharedState>) -> Result<Json<Value>, AppError> {
    let indices = mgr.list_indices()?;
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
    let list = mgr.list_indices()?;
    Ok(Json(Value::Array(list)))
}

#[derive(Debug, Default, Deserialize)]
struct AnalyzeRequest {
    text: Value,
    #[serde(default)]
    analyzer: Option<String>,
    #[serde(default)]
    mode: Option<SplitMode>,
    #[serde(default)]
    form: Option<TokenForm>,
    #[serde(default)]
    stop_tags: Option<Vec<String>>,
    #[serde(default)]
    stop_words: Option<HashSet<String>>,
    #[serde(default)]
    lowercase: Option<bool>,
}

async fn analyze_global_handler(
    State(mgr): State<SharedState>,
    Json(req): Json<AnalyzeRequest>,
) -> Result<Json<Value>, AppError> {
    run_analyze(&mgr, None, req)
}

async fn analyze_index_handler(
    State(mgr): State<SharedState>,
    Path(index): Path<String>,
    Json(req): Json<AnalyzeRequest>,
) -> Result<Json<Value>, AppError> {
    let managed = mgr.get_index(&index)?;
    run_analyze(&mgr, Some(&managed), req)
}

fn run_analyze(
    mgr: &IndexManager,
    managed: Option<&Arc<ManagedIndex>>,
    req: AnalyzeRequest,
) -> Result<Json<Value>, AppError> {
    let mut cfg = if let Some(analyzer_name) = req.analyzer.as_deref() {
        if let Some(idx) = managed {
            if let Some(custom) = idx.meta.settings.analysis.analyzer.get(analyzer_name) {
                custom.clone()
            } else {
                preset_analyzer_config(analyzer_name)?
            }
        } else {
            preset_analyzer_config(analyzer_name)?
        }
    } else {
        SudachiTokenizerConfig::default()
    };

    if let Some(m) = req.mode {
        cfg.mode = m;
    }
    if let Some(f) = req.form {
        cfg.form = f;
    }
    if let Some(tags) = req.stop_tags {
        cfg.stop_tags = tags;
    }
    if let Some(words) = req.stop_words {
        cfg.stop_words = words;
    }
    if let Some(lc) = req.lowercase {
        cfg.lowercase = lc;
    }

    let tok = SudachiTokenizer::new(mgr.dict.clone(), cfg);
    let mut all_tokens = Vec::new();
    match &req.text {
        Value::String(s) => {
            all_tokens.extend(tok.analyze(s));
        }
        Value::Array(arr) => {
            for item in arr {
                if let Some(s) = item.as_str() {
                    all_tokens.extend(tok.analyze(s));
                }
            }
        }
        _ => {
            return Err(AppError::BadRequest(
                "'text' must be a string or array of strings".into(),
            ));
        }
    }

    Ok(Json(json!({ "tokens": all_tokens })))
}

fn preset_analyzer_config(name: &str) -> Result<SudachiTokenizerConfig, AppError> {
    match name {
        "sudachi" | "sudachi_c" | "default" => Ok(SudachiTokenizerConfig::with_mode(SplitMode::C)),
        "sudachi_a" => Ok(SudachiTokenizerConfig::with_mode(SplitMode::A)),
        "sudachi_b" => Ok(SudachiTokenizerConfig::with_mode(SplitMode::B)),
        "sudachi_search" => Ok(SudachiTokenizerConfig {
            mode: SplitMode::A,
            form: TokenForm::Normalized,
            stop_tags: vec![
                "補助記号".to_string(),
                "空白".to_string(),
                "助詞".to_string(),
                "助動詞".to_string(),
            ],
            stop_words: HashSet::new(),
            lowercase: true,
        }),
        _ => Err(AppError::BadRequest(format!("unknown analyzer [{name}]"))),
    }
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
    let managed = mgr.get_index(&index)?;
    Ok(Json(json!({
        index: {
            "settings": &managed.meta.settings,
            "mappings": &managed.meta.mappings,
            "docs_count": managed.doc_count(),
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
    let managed = mgr.get_index(&index)?;
    managed.refresh()?;
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
    let managed = mgr.get_or_create_from_doc(&index, &doc)?;
    let (id, created) = managed.index_document(None, doc, q.should_refresh(true))?;
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
    let managed = mgr.get_or_create_from_doc(&index, &doc)?;
    let (doc_id, created) = managed.index_document(Some(id), doc, q.should_refresh(true))?;
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
    let managed = mgr.get_index(&index)?;
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
    let managed = mgr.get_index(&index)?;
    let existed = managed.delete_document(&id, q.should_refresh(true))?;
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
    execute_bulk(&mgr, None, &body, q.should_refresh(true))
}

async fn bulk_index_handler(
    State(mgr): State<SharedState>,
    Path(index): Path<String>,
    Query(q): Query<RefreshQuery>,
    body: String,
) -> Result<Json<Value>, AppError> {
    execute_bulk(&mgr, Some(&index), &body, q.should_refresh(true))
}

fn execute_bulk(
    mgr: &IndexManager,
    default_index: Option<&str>,
    body: &str,
    refresh: bool,
) -> Result<Json<Value>, AppError> {
    let start = Instant::now();
    let mut lines = body.lines().filter(|l| !l.trim().is_empty());
    let mut items = Vec::new();
    let mut has_errors = false;
    let mut touched_indices: BTreeMap<String, Arc<ManagedIndex>> = BTreeMap::new();

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
                    let idx = mgr.get_or_create_from_doc(target_index, &doc_val)?;
                    touched_indices.insert(target_index.to_string(), idx.clone());
                    idx
                };

                match managed.index_document(target_id, doc_val, false) {
                    Ok((doc_id, created)) => {
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
                    let idx = mgr.get_index(target_index)?;
                    touched_indices.insert(target_index.to_string(), idx.clone());
                    idx
                };
                let id_term = Term::from_field_text(managed.id_field(), &doc_id);
                if let Ok(writer) = managed.writer.lock() {
                    writer.delete_term(id_term);
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

    if refresh {
        for managed in touched_indices.values() {
            managed.refresh()?;
        }
    }

    Ok(Json(json!({
        "took": start.elapsed().as_millis() as u64,
        "errors": has_errors,
        "items": items,
    })))
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
    let managed = mgr.get_index(&index)?;
    let mut req: SearchRequest = if body.trim().is_empty() {
        SearchRequest::default()
    } else {
        serde_json::from_str(&body)
            .map_err(|e| AppError::BadRequest(format!("invalid search request JSON: {e}")))?
    };
    if req.from.is_none() {
        req.from = params.from;
    }
    if req.size.is_none() {
        req.size = params.size;
    }
    let result = execute_search(&managed, req, params.q.as_deref())?;
    Ok(Json(result))
}

async fn search_global_handler(
    State(mgr): State<SharedState>,
    Query(params): Query<SearchQueryParams>,
    body: String,
) -> Result<Json<Value>, AppError> {
    let indices = mgr.resolve_indices(None)?;
    let mut req: SearchRequest = if body.trim().is_empty() {
        SearchRequest::default()
    } else {
        serde_json::from_str(&body)
            .map_err(|e| AppError::BadRequest(format!("invalid search request JSON: {e}")))?
    };
    if req.from.is_none() {
        req.from = params.from;
    }
    if req.size.is_none() {
        req.size = params.size;
    }
    let result = execute_search_arc(&indices, req, params.q.as_deref())?;
    Ok(Json(result))
}

async fn get_all_mappings_handler(
    State(mgr): State<SharedState>,
) -> Result<Json<Value>, AppError> {
    let indices = mgr.resolve_indices(None)?;
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
    let indices = mgr.resolve_indices(Some(&index))?;
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
    execute_msearch(&mgr, None, &body)
}

async fn msearch_index_handler(
    State(mgr): State<SharedState>,
    Path(index): Path<String>,
    body: String,
) -> Result<Json<Value>, AppError> {
    execute_msearch(&mgr, Some(&index), &body)
}

fn execute_msearch(
    mgr: &IndexManager,
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
        let req: SearchRequest = serde_json::from_str(body_line)
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

        let indices = mgr.resolve_indices(target_pattern)?;
        if indices.is_empty()
            && let Some(pat) = target_pattern.filter(|p| !p.is_empty() && *p != "*" && *p != "_all")
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

        match execute_search_arc(&indices, req, None) {
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
