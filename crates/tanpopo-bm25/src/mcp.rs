use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::sync::Arc;

use serde_json::{Value, json};

use crate::index::{AppError, IndexManager, ManagedIndex};
use crate::query::{SearchRequest, execute_search_arc, filter_and_truncate_source};
use crate::server::{AnalyzeRequest, run_analyze};

pub const PROTOCOL_VERSION: &str = "2025-03-26";
pub const DEFAULT_MCP_MAX_FIELD_CHARS: usize = 500;

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
            "description": "Japanese full-text search (Sudachi + BM25) with filters, aggregations, line-numbered highlights, and token diagnostics. Long text fields in _source are automatically truncated to max_field_chars (default: 500; pass 0 for full text) to protect context window.",
            "inputSchema": { "type": "object", "properties": {
                "index": { "type": "string", "description": "Index name, wildcard or comma-separated list. Default: all." },
                "q": { "type": "string", "description": "Simple query string (e.g. 空港 AND 京都, or space-separated terms)." },
                "query": { "type": "object", "description": "Elasticsearch Query DSL (match, multi_match, bool, range, term, ...). Supports field^boost and minimum_should_match." },
                "minimum_should_match": { "description": "Minimum number or percentage of terms that must match (e.g. 2, -1, '75%')." },
                "fields": { "description": "Fields to include in _source (array of field names/wildcards, boolean, or {includes, excludes})." },
                "exclude_fields": { "type": "array", "items": { "type": "string" }, "description": "Fields to exclude from _source (e.g. ['content'])." },
                "max_field_chars": { "type": "integer", "description": "Max characters per string field in _source (default: 500; 0 = unlimited)." },
                "highlight": { "description": "Highlight matched snippets with [Lstart-Lend] line numbers (default: true when query is present; or object {fields, fragment_size, number_of_fragments})." },
                "explain_tokens": { "type": "boolean", "description": "Return Sudachi query token breakdown and document frequencies in diagnostics.query_tokens (default: true)." },
                "size": { "type": "integer" },
                "from": { "type": "integer" },
                "sort": {},
                "aggs": { "type": "object" }
            }}
        },
        {
            "name": "get_document",
            "description": "Get a document by index and id. Supports reading specific line ranges (start_line/end_line), character ranges (offset/length), field filtering, or extracting a Markdown heading outline (outline=true) for progressive disclosure of long documents.",
            "inputSchema": { "type": "object", "required": ["index", "id"], "properties": {
                "index": { "type": "string" },
                "id": { "type": "string" },
                "fields": { "type": "array", "items": { "type": "string" }, "description": "Optional list of _source fields to return." },
                "field": { "type": "string", "description": "Target field for line/char slicing or outline extraction (default: 'content' or longest string field)." },
                "start_line": { "type": "integer", "description": "1-based start line number (inclusive) to read from the target field." },
                "end_line": { "type": "integer", "description": "1-based end line number (inclusive) to read from the target field." },
                "offset": { "type": "integer", "description": "0-based character offset to read from the target field." },
                "length": { "type": "integer", "description": "Maximum number of characters to read when using offset." },
                "outline": { "type": "boolean", "description": "If true, extracts Markdown headings (# ...) and section titles with line numbers instead of full text." }
            }}
        },
        {
            "name": "search_in_document",
            "description": "Search (grep) within a single document using Sudachi morphological normalization and substring matching. Returns matching line numbers and surrounding context lines so you can inspect long documents without loading them in full.",
            "inputSchema": { "type": "object", "required": ["index", "id", "query"], "properties": {
                "index": { "type": "string" },
                "id": { "type": "string" },
                "query": { "type": "string", "description": "Keywords or phrase to search inside the document." },
                "field": { "type": "string", "description": "Specific field to search in (default: all text fields)." },
                "context_lines": { "type": "integer", "description": "Number of context lines before and after each match (default: 2)." },
                "max_matches": { "type": "integer", "description": "Maximum number of match windows to return (default: 20)." }
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

fn usize_arg(args: &Value, key: &str) -> Option<usize> {
    args.get(key)
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
}

fn bool_arg(args: &Value, key: &str) -> Option<bool> {
    args.get(key).and_then(|v| v.as_bool())
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
            let mut req: SearchRequest = serde_json::from_value(args.clone())
                .map_err(|e| AppError::BadRequest(format!("invalid search arguments: {e}")))?;
            let q_param = str_arg(args, "q");
            let has_query = q_param.is_some_and(|s| !s.trim().is_empty())
                || req
                    .query
                    .as_ref()
                    .is_some_and(|q| !q.is_null() && q.as_object().is_some_and(|o| !o.is_empty()));

            if req.max_field_chars.is_none() {
                req.max_field_chars = Some(DEFAULT_MCP_MAX_FIELD_CHARS);
            }
            if req.highlight.is_none() && has_query {
                req.highlight = Some(Value::Bool(true));
            }
            if req.explain_tokens.is_none() && has_query {
                req.explain_tokens = Some(true);
            }
            execute_search_arc(&indices, req, q_param)
        }
        "get_document" => handle_get_document(mgr, args),
        "search_in_document" => handle_search_in_document(mgr, args),
        "analyze_text" => {
            let req: AnalyzeRequest = serde_json::from_value(args.clone())
                .map_err(|e| AppError::BadRequest(format!("invalid analyze arguments: {e}")))?;
            let managed = str_arg(args, "index").map(|i| mgr.get_index(i)).transpose()?;
            Ok(run_analyze(mgr, managed.as_ref(), req)?.0)
        }
        other => Err(AppError::BadRequest(format!("unknown tool [{other}]"))),
    }
}

fn select_target_field(source: &Value, explicit_field: Option<&str>) -> Option<String> {
    if let Some(f) = explicit_field {
        return Some(f.to_string());
    }
    let obj = source.as_object()?;
    if obj.get("content").and_then(|v| v.as_str()).is_some() {
        return Some("content".to_string());
    }
    if obj.get("body").and_then(|v| v.as_str()).is_some() {
        return Some("body".to_string());
    }
    obj.iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k, s.len())))
        .max_by_key(|(_, len)| *len)
        .map(|(k, _)| k.clone())
}

pub fn handle_get_document(mgr: &IndexManager, args: &Value) -> Result<Value, AppError> {
    let index = str_arg(args, "index")
        .ok_or_else(|| AppError::BadRequest("'index' is required".into()))?;
    let id = str_arg(args, "id")
        .ok_or_else(|| AppError::BadRequest("'id' is required".into()))?;
    let managed = mgr.get_index(index)?;
    let source = managed
        .get_document(id)?
        .ok_or_else(|| AppError::DocumentNotFound(id.to_string()))?;

    let fields_spec = args.get("fields");
    let explicit_field = str_arg(args, "field");
    let start_line = usize_arg(args, "start_line");
    let end_line = usize_arg(args, "end_line");
    let offset = usize_arg(args, "offset");
    let length = usize_arg(args, "length");
    let outline = bool_arg(args, "outline").unwrap_or(false);

    let has_progressive_opts = fields_spec.is_some()
        || explicit_field.is_some()
        || start_line.is_some()
        || end_line.is_some()
        || offset.is_some()
        || length.is_some()
        || outline;

    if !has_progressive_opts {
        return Ok(json!({ "_index": index, "_id": id, "_source": source }));
    }

    let target_field = select_target_field(&source, explicit_field);
    let mut res = json!({
        "_index": index,
        "_id": id,
    });

    if outline {
        if let Some(ref tf) = target_field {
            if let Some(text) = source.get(tf).and_then(|v| v.as_str()) {
                let lines: Vec<&str> = text.lines().collect();
                let total_lines = lines.len();
                let total_chars = text.chars().count();
                let mut headings = Vec::new();
                for (idx, line) in lines.iter().enumerate() {
                    let trimmed = line.trim();
                    if is_heading_line(trimmed) {
                        headings.push(json!({
                            "line": idx + 1,
                            "heading": trimmed,
                        }));
                    }
                }
                res["outline"] = json!({
                    "field": tf,
                    "total_lines": total_lines,
                    "total_chars": total_chars,
                    "headings": headings,
                });
            }
        }
        let excludes = target_field.clone().map(|f| vec![f]);
        if let Some(filtered) = filter_and_truncate_source(
            &source,
            fields_spec,
            excludes.as_deref(),
            Some(DEFAULT_MCP_MAX_FIELD_CHARS),
        ) {
            res["_source"] = filtered;
        }
        return Ok(res);
    }

    if start_line.is_some() || end_line.is_some() {
        let tf = target_field
            .ok_or_else(|| AppError::BadRequest("no string field found to slice".into()))?;
        let text = source
            .get(&tf)
            .and_then(|v| v.as_str())
            .ok_or_else(|| AppError::BadRequest(format!("field [{tf}] is not a string")))?;
        let lines: Vec<&str> = text.lines().collect();
        let total_lines = lines.len();
        let total_chars = text.chars().count();
        let s_line = start_line.unwrap_or(1).max(1);
        let e_line = end_line
            .unwrap_or_else(|| (s_line + 99).min(total_lines.max(1)))
            .max(s_line)
            .min(total_lines.max(1));

        let mut numbered = String::new();
        if total_lines > 0 && s_line <= total_lines {
            for line_no in s_line..=e_line {
                if !numbered.is_empty() {
                    numbered.push('\n');
                }
                numbered.push_str(&format!("{line_no}: {}", lines[line_no - 1]));
            }
        }
        res["slice"] = json!({
            "field": tf,
            "start_line": s_line,
            "end_line": e_line,
            "total_lines": total_lines,
            "total_chars": total_chars,
            "content": numbered,
        });
        let excludes = vec![tf];
        if let Some(filtered) = filter_and_truncate_source(
            &source,
            fields_spec,
            Some(&excludes),
            Some(DEFAULT_MCP_MAX_FIELD_CHARS),
        ) {
            res["_source"] = filtered;
        }
        return Ok(res);
    }

    if offset.is_some() || length.is_some() {
        let tf = target_field
            .ok_or_else(|| AppError::BadRequest("no string field found to slice".into()))?;
        let text = source
            .get(&tf)
            .and_then(|v| v.as_str())
            .ok_or_else(|| AppError::BadRequest(format!("field [{tf}] is not a string")))?;
        let total_chars = text.chars().count();
        let total_lines = text.lines().count();
        let off = offset.unwrap_or(0).min(total_chars);
        let len = length.unwrap_or(2000);
        let sliced: String = text.chars().skip(off).take(len).collect();
        res["slice"] = json!({
            "field": tf,
            "offset": off,
            "length": sliced.chars().count(),
            "total_chars": total_chars,
            "total_lines": total_lines,
            "content": sliced,
        });
        let excludes = vec![tf];
        if let Some(filtered) = filter_and_truncate_source(
            &source,
            fields_spec,
            Some(&excludes),
            Some(DEFAULT_MCP_MAX_FIELD_CHARS),
        ) {
            res["_source"] = filtered;
        }
        return Ok(res);
    }

    if let Some(filtered) = filter_and_truncate_source(&source, fields_spec, None, None) {
        res["_source"] = filtered;
    }
    Ok(res)
}

fn is_heading_line(trimmed: &str) -> bool {
    if trimmed.is_empty() || trimmed.chars().count() > 120 {
        return false;
    }
    if trimmed.starts_with('#') {
        return true;
    }
    // Also detect numbered Japanese section headers like "1. はじめに", "第1章 ...", "２．評価方法"
    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if first == '第' && (trimmed.contains('章') || trimmed.contains('節')) {
        return true;
    }
    if first.is_ascii_digit() || ('０'..='９').contains(&first) {
        let rest: String = chars.take(6).collect();
        if rest.contains('.') || rest.contains('．') || rest.contains('、') {
            return true;
        }
    }
    false
}

pub fn handle_search_in_document(mgr: &IndexManager, args: &Value) -> Result<Value, AppError> {
    let index = str_arg(args, "index")
        .ok_or_else(|| AppError::BadRequest("'index' is required".into()))?;
    let id = str_arg(args, "id")
        .ok_or_else(|| AppError::BadRequest("'id' is required".into()))?;
    let query_str = str_arg(args, "query")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| AppError::BadRequest("'query' is required".into()))?;
    let target_field = str_arg(args, "field");
    let context_lines = usize_arg(args, "context_lines").unwrap_or(2);
    let max_matches = usize_arg(args, "max_matches").unwrap_or(20).max(1);

    let managed = mgr.get_index(index)?;
    let source = managed
        .get_document(id)?
        .ok_or_else(|| AppError::DocumentNotFound(id.to_string()))?;
    let obj = source
        .as_object()
        .ok_or_else(|| AppError::Internal("document _source is not an object".into()))?;

    let mut matches = Vec::new();
    let mut total_matches = 0usize;

    for (field_name, field_val) in obj {
        if let Some(tf) = target_field {
            if field_name != tf {
                continue;
            }
        }
        let Some(text) = field_val.as_str() else {
            continue;
        };
        if text.is_empty() {
            continue;
        }

        let field_matches = search_field_lines(
            &managed,
            field_name,
            text,
            query_str,
            context_lines,
            max_matches.saturating_sub(matches.len()),
            &mut total_matches,
        );
        matches.extend(field_matches);
    }

    Ok(json!({
        "_index": index,
        "_id": id,
        "query": query_str,
        "total_matches": total_matches,
        "matches": matches,
    }))
}

fn search_field_lines(
    managed: &ManagedIndex,
    field_name: &str,
    text: &str,
    query_str: &str,
    context_lines: usize,
    remaining_cap: usize,
    total_matches: &mut usize,
) -> Vec<Value> {
    let analyzer_name = managed
        .meta
        .mappings
        .properties
        .get(field_name)
        .map(|m| m.index_analyzer())
        .unwrap_or("sudachi");

    let mut query_tokens: HashSet<String> = HashSet::new();
    for part in query_str.split_whitespace() {
        let lower = part.trim().to_lowercase();
        if !lower.is_empty() {
            query_tokens.insert(lower);
        }
    }

    if let Some(mut analyzer) = managed
        .index
        .tokenizers()
        .get(analyzer_name)
        .or_else(|| managed.index.tokenizers().get("sudachi"))
    {
        let mut stream = analyzer.token_stream(query_str);
        stream.process(&mut |tok| {
            let lower = tok.text.to_lowercase();
            if !lower.is_empty() {
                query_tokens.insert(lower);
            }
        });
    }

    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut matched_line_spans: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
    let mut analyzer_opt = managed
        .index
        .tokenizers()
        .get(analyzer_name)
        .or_else(|| managed.index.tokenizers().get("sudachi"));

    for (line_idx, &line) in lines.iter().enumerate() {
        if line.is_empty() {
            continue;
        }
        let mut spans: Vec<(usize, usize)> = Vec::new();
        if let Some(ref mut analyzer) = analyzer_opt {
            let mut stream = analyzer.token_stream(line);
            stream.process(&mut |tok| {
                if query_tokens.contains(&tok.text.to_lowercase())
                    && tok.offset_from < tok.offset_to
                    && tok.offset_to <= line.len()
                    && line.is_char_boundary(tok.offset_from)
                    && line.is_char_boundary(tok.offset_to)
                {
                    spans.push((tok.offset_from, tok.offset_to));
                }
            });
        }

        let lower_line = line.to_lowercase();
        if lower_line.len() == line.len() {
            for q in &query_tokens {
                if q.is_empty() {
                    continue;
                }
                let mut from = 0;
                while let Some(pos) = lower_line[from..].find(q.as_str()) {
                    let s = from + pos;
                    let e = s + q.len();
                    if line.is_char_boundary(s) && line.is_char_boundary(e) {
                        spans.push((s, e));
                    }
                    from = e.max(from + 1);
                    if from >= lower_line.len() {
                        break;
                    }
                }
            }
        } else {
            for q in &query_tokens {
                if q.is_empty() {
                    continue;
                }
                let mut from = 0;
                while let Some(pos) = line[from..].find(q.as_str()) {
                    let s = from + pos;
                    let e = s + q.len();
                    spans.push((s, e));
                    from = e.max(from + 1);
                    if from >= line.len() {
                        break;
                    }
                }
            }
        }

        if !spans.is_empty() {
            spans.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
            let mut merged: Vec<(usize, usize)> = Vec::new();
            for sp in spans {
                if let Some(last) = merged.last_mut() {
                    if sp.0 <= last.1 {
                        last.1 = last.1.max(sp.1);
                        continue;
                    }
                }
                merged.push(sp);
            }
            matched_line_spans.push((line_idx, merged));
        }
    }

    *total_matches += matched_line_spans.len();
    if matched_line_spans.is_empty() || remaining_cap == 0 {
        return Vec::new();
    }

    let total_lines = lines.len();
    let mut windows: Vec<Value> = Vec::new();
    let mut i = 0;

    while i < matched_line_spans.len() && windows.len() < remaining_cap {
        let first_line_idx = matched_line_spans[i].0;
        let win_start = first_line_idx.saturating_sub(context_lines);
        let mut win_end = (first_line_idx + context_lines).min(total_lines - 1);
        let mut j = i;

        while j + 1 < matched_line_spans.len() {
            let next_idx = matched_line_spans[j + 1].0;
            if next_idx.saturating_sub(context_lines) <= win_end + 1 {
                j += 1;
                win_end = (next_idx + context_lines).min(total_lines - 1);
            } else {
                break;
            }
        }

        let mut match_line_numbers: Vec<usize> = Vec::new();
        for &(l_idx, _) in &matched_line_spans[i..=j] {
            match_line_numbers.push(l_idx + 1);
        }

        let mut snippet = String::new();
        for l_idx in win_start..=win_end {
            if !snippet.is_empty() {
                snippet.push('\n');
            }
            let line_no = l_idx + 1;
            let raw_line = lines[l_idx];
            if let Some((_, spans)) = matched_line_spans[i..=j]
                .iter()
                .find(|(idx, _)| *idx == l_idx)
            {
                let mut highlighted = String::new();
                let mut cur = 0;
                for &(s, e) in spans {
                    if s > cur {
                        highlighted.push_str(&raw_line[cur..s]);
                    }
                    highlighted.push_str("<em>");
                    highlighted.push_str(&raw_line[s..e]);
                    highlighted.push_str("</em>");
                    cur = e;
                }
                if cur < raw_line.len() {
                    highlighted.push_str(&raw_line[cur..]);
                }
                snippet.push_str(&format!("{line_no}: {highlighted}"));
            } else {
                snippet.push_str(&format!("{line_no}: {raw_line}"));
            }
        }

        windows.push(json!({
            "field": field_name,
            "start_line": win_start + 1,
            "end_line": win_end + 1,
            "match_lines": match_line_numbers,
            "total_lines": total_lines,
            "snippet": snippet,
        }));

        i = j + 1;
    }

    windows
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
