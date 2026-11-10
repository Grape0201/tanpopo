use std::io::{BufRead, Write};
use std::sync::Arc;

use serde_json::{Value, json};
use tanpopo_bm25::index::AppError;
use tanpopo_bm25::mcp::{
    DEFAULT_MCP_MAX_FIELD_CHARS, PROTOCOL_VERSION, handle_get_document, handle_search_in_document,
};
use tanpopo_bm25::server::{AnalyzeRequest, run_analyze};

use crate::engine::HybridManager;

fn tool_definitions() -> Value {
    json!([
        {
            "name": "list_indices",
            "description": "List indices with document counts, vector counts/dimensions, field mappings and analyzer settings. Call this first to learn the schema.",
            "inputSchema": { "type": "object", "properties": {
                "index": { "type": "string", "description": "Index name, wildcard or comma-separated list. Default: all." }
            }}
        },
        {
            "name": "search",
            "description": "Hybrid Japanese search combining BM25 (Sudachi + Tantivy) and dense vector similarity (RRF) with filters, aggregations, line-numbered highlights, and token diagnostics.",
            "inputSchema": { "type": "object", "properties": {
                "index": { "type": "string", "description": "Index name, wildcard or comma-separated list. Default: all." },
                "q": { "type": "string", "description": "Simple query string (e.g. 空港 AND 京都, or natural language query)." },
                "query": { "type": "object", "description": "Elasticsearch Query DSL (match, multi_match, bool, range, term, ...)." },
                "mode": { "type": "string", "enum": ["hybrid", "bm25", "vector"], "description": "Search mode (default: 'hybrid' when embeddings are available)." },
                "minimum_should_match": { "description": "Minimum number or percentage of terms that must match (e.g. 2, -1, '75%')." },
                "fields": { "description": "Fields to include in _source (array of field names/wildcards, boolean, or {includes, excludes})." },
                "exclude_fields": { "type": "array", "items": { "type": "string" }, "description": "Fields to exclude from _source (e.g. ['content'])." },
                "max_field_chars": { "type": "integer", "description": "Max characters per string field in _source (default: 500; 0 = unlimited)." },
                "highlight": { "description": "Highlight matched snippets with [Lstart-Lend] line numbers (default: true when query is present)." },
                "explain_tokens": { "type": "boolean", "description": "Return Sudachi query token breakdown and document frequencies (default: true)." },
                "size": { "type": "integer" },
                "from": { "type": "integer" },
                "sort": {},
                "aggs": { "type": "object" }
            }}
        },
        {
            "name": "get_document",
            "description": "Get a document by index and id. Supports reading specific line ranges (start_line/end_line), character ranges (offset/length), field filtering, or extracting a Markdown heading outline (outline=true).",
            "inputSchema": { "type": "object", "required": ["index", "id"], "properties": {
                "index": { "type": "string" },
                "id": { "type": "string" },
                "fields": { "type": "array", "items": { "type": "string" } },
                "field": { "type": "string" },
                "start_line": { "type": "integer" },
                "end_line": { "type": "integer" },
                "offset": { "type": "integer" },
                "length": { "type": "integer" },
                "outline": { "type": "boolean" }
            }}
        },
        {
            "name": "search_in_document",
            "description": "Search (grep) within a single document using Sudachi morphological normalization and substring matching.",
            "inputSchema": { "type": "object", "required": ["index", "id", "query"], "properties": {
                "index": { "type": "string" },
                "id": { "type": "string" },
                "query": { "type": "string" },
                "field": { "type": "string" },
                "context_lines": { "type": "integer" },
                "max_matches": { "type": "integer" }
            }}
        },
        {
            "name": "analyze_text",
            "description": "Run Sudachi morphological analysis: tokens, normalized form, reading, part of speech.",
            "inputSchema": { "type": "object", "required": ["text"], "properties": {
                "text": { "type": "string" },
                "index": { "type": "string" },
                "analyzer": { "type": "string" },
                "mode": { "type": "string", "enum": ["A", "B", "C"] }
            }}
        }
    ])
}

async fn call_tool(mgr: &HybridManager, name: &str, args: &Value) -> Result<Value, AppError> {
    match name {
        "list_indices" => {
            let pattern = args.get("index").and_then(|v| v.as_str());
            let indices = mgr.bm25().resolve_indices(pattern)?;
            let mut out = Vec::with_capacity(indices.len());
            for i in indices {
                let (vec_docs, vec_chunks, vec_dim) = mgr
                    .get_or_open_vector_store(&i.name)
                    .map(|s| (s.active_doc_count(), s.active_vector_count(), s.dim()))
                    .unwrap_or((0, 0, 0));
                out.push(json!({
                    "index": i.name,
                    "docs_count": i.doc_count(),
                    "vector_docs_count": vec_docs,
                    "vector_chunks_count": vec_chunks,
                    "vector_dim": vec_dim,
                    "settings": i.meta.settings,
                    "mappings": i.meta.mappings,
                }));
            }
            Ok(Value::Array(out))
        }
        "search" => {
            let pattern = args.get("index").and_then(|v| v.as_str());
            let q_param = args.get("q").and_then(|v| v.as_str());
            let mut req_val = args.clone();
            if let Some(obj) = req_val.as_object_mut() {
                let has_query = q_param.is_some_and(|s| !s.trim().is_empty())
                    || obj
                        .get("query")
                        .is_some_and(|q| !q.is_null() && q.as_object().is_some_and(|o| !o.is_empty()));
                if !obj.contains_key("max_field_chars") {
                    obj.insert("max_field_chars".into(), json!(DEFAULT_MCP_MAX_FIELD_CHARS));
                }
                if !obj.contains_key("highlight") && has_query {
                    obj.insert("highlight".into(), Value::Bool(true));
                }
                if !obj.contains_key("explain_tokens") && has_query {
                    obj.insert("explain_tokens".into(), Value::Bool(true));
                }
            }
            mgr.search(pattern, req_val, q_param, None, None).await
        }
        "get_document" => handle_get_document(mgr.bm25(), args),
        "search_in_document" => handle_search_in_document(mgr.bm25(), args),
        "analyze_text" => {
            let req: AnalyzeRequest = serde_json::from_value(args.clone())
                .map_err(|e| AppError::BadRequest(format!("invalid analyze arguments: {e}")))?;
            let managed = args
                .get("index")
                .and_then(|v| v.as_str())
                .map(|i| mgr.bm25().get_index(i))
                .transpose()?;
            Ok(run_analyze(mgr.bm25(), managed.as_ref(), req)?.0)
        }
        other => Err(AppError::BadRequest(format!("unknown tool [{other}]"))),
    }
}

async fn handle_one(mgr: &HybridManager, msg: &Value) -> Option<Value> {
    let id = msg.get("id")?.clone();
    let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or(Value::Null);

    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "tanpopo", "version": env!("CARGO_PKG_VERSION") },
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tool_definitions() })),
        "tools/call" => {
            let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
            Ok(match call_tool(mgr, name, &args).await {
                Ok(v) => json!({
                    "content": [{ "type": "text", "text": v.to_string() }],
                    "isError": false,
                }),
                Err(e) => json!({
                    "content": [{ "type": "text", "text": e.to_string() }],
                    "isError": true,
                }),
            })
        }
        other => Err(format!("method not found: {other}")),
    };

    Some(match result {
        Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
        Err(m) => json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32601, "message": m } }),
    })
}

pub async fn handle_mcp_value(mgr: &HybridManager, payload: Value) -> Option<Value> {
    match payload {
        Value::Array(items) => {
            let mut out = Vec::new();
            for m in items {
                if let Some(res) = handle_one(mgr, &m).await {
                    out.push(res);
                }
            }
            (!out.is_empty()).then_some(Value::Array(out))
        }
        other => handle_one(mgr, &other).await,
    }
}

pub fn run_stdio_loop(mgr: Arc<HybridManager>) -> std::io::Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(v) => rt.block_on(handle_mcp_value(&mgr, v)),
            Err(e) => Some(json!({
                "jsonrpc": "2.0", "id": null,
                "error": { "code": -32700, "message": format!("parse error: {e}") }
            })),
        };
        if let Some(r) = reply {
            writeln!(stdout, "{r}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}
