use std::io::{BufRead, Write};
use std::sync::Arc;

use serde_json::{Value, json};

use crate::index::{AppError, IndexManager};
use crate::query::{SearchRequest, execute_search_arc};
use crate::server::{AnalyzeRequest, run_analyze};

const PROTOCOL_VERSION: &str = "2025-03-26";

fn tool_definitions() -> Value {
    json!([
        {
            "name": "list_indices",
            "description": "List indices with document counts, field mappings and analyzer settings. Call this first to learn the schema.",
            "inputSchema": { "type": "object", "properties": {
                "index": { "type": "string", "description": "Index name, wildcard or comma-separated list. Default: all." }
            }}
        },
        {
            "name": "search",
            "description": "Japanese full-text search (Sudachi + BM25) with filters and aggregations. Returns matching documents with full _source.",
            "inputSchema": { "type": "object", "properties": {
                "index": { "type": "string", "description": "Index name, wildcard or comma-separated list. Default: all." },
                "q": { "type": "string", "description": "Simple query string (e.g. 空港 AND 京都)." },
                "query": { "type": "object", "description": "Elasticsearch Query DSL (match, bool, range, term, ...)." },
                "size": { "type": "integer" },
                "from": { "type": "integer" },
                "sort": {},
                "aggs": { "type": "object" }
            }}
        },
        {
            "name": "get_document",
            "description": "Get a full document by index and id.",
            "inputSchema": { "type": "object", "required": ["index", "id"], "properties": {
                "index": { "type": "string" },
                "id": { "type": "string" }
            }}
        },
        {
            "name": "analyze_text",
            "description": "Run Sudachi morphological analysis: tokens, normalized form, reading, part of speech.",
            "inputSchema": { "type": "object", "required": ["text"], "properties": {
                "text": { "type": "string" },
                "index": { "type": "string", "description": "Use custom analyzers defined on this index." },
                "analyzer": { "type": "string" },
                "mode": { "type": "string", "enum": ["A", "B", "C"] }
            }}
        }
    ])
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(|v| v.as_str())
}

fn call_tool(mgr: &IndexManager, name: &str, args: &Value) -> Result<Value, AppError> {
    match name {
        "list_indices" => {
            let indices = mgr.resolve_indices(str_arg(args, "index"))?;
            let out: Vec<Value> = indices
                .iter()
                .map(|i| {
                    json!({
                        "index": i.name,
                        "docs_count": i.doc_count(),
                        "settings": i.meta.settings,
                        "mappings": i.meta.mappings,
                    })
                })
                .collect();
            Ok(Value::Array(out))
        }
        "search" => {
            let indices = mgr.resolve_indices(str_arg(args, "index"))?;
            let req: SearchRequest = serde_json::from_value(args.clone())
                .map_err(|e| AppError::BadRequest(format!("invalid search arguments: {e}")))?;
            execute_search_arc(&indices, req, str_arg(args, "q"))
        }
        "get_document" => {
            let index = str_arg(args, "index")
                .ok_or_else(|| AppError::BadRequest("'index' is required".into()))?;
            let id = str_arg(args, "id")
                .ok_or_else(|| AppError::BadRequest("'id' is required".into()))?;
            let managed = mgr.get_index(index)?;
            let source = managed
                .get_document(id)?
                .ok_or_else(|| AppError::DocumentNotFound(id.to_string()))?;
            Ok(json!({ "_index": index, "_id": id, "_source": source }))
        }
        "analyze_text" => {
            let req: AnalyzeRequest = serde_json::from_value(args.clone())
                .map_err(|e| AppError::BadRequest(format!("invalid analyze arguments: {e}")))?;
            let managed = str_arg(args, "index").map(|i| mgr.get_index(i)).transpose()?;
            Ok(run_analyze(mgr, managed.as_ref(), req)?.0)
        }
        other => Err(AppError::BadRequest(format!("unknown tool [{other}]"))),
    }
}

fn handle_one(mgr: &IndexManager, msg: &Value) -> Option<Value> {
    let id = msg.get("id")?.clone(); // no id => notification, no response
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
            // Tool errors are returned as isError results so the agent can fix its query.
            Ok(match call_tool(mgr, name, &args) {
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

/// Handles a JSON-RPC payload (single or batch). `None` means nothing to send back.
pub fn handle_mcp_value(mgr: &IndexManager, payload: Value) -> Option<Value> {
    match payload {
        Value::Array(items) => {
            let out: Vec<Value> = items.iter().filter_map(|m| handle_one(mgr, m)).collect();
            (!out.is_empty()).then_some(Value::Array(out))
        }
        other => handle_one(mgr, &other),
    }
}

pub fn run_stdio_loop(mgr: Arc<IndexManager>) -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(v) => handle_mcp_value(&mgr, v),
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
