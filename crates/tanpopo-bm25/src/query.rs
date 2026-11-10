use std::collections::{BTreeMap, HashSet};
use std::ops::Bound;
use std::sync::Arc;
use std::time::Instant;

use serde::Deserialize;
use serde_json::{Map, Value, json};
use tantivy::collector::{Count, TopDocs};
use tantivy::query::{
    AllQuery, BooleanQuery, BoostQuery, EmptyQuery, Occur, PhraseQuery, Query, QueryParser,
    RangeQuery, TermQuery,
};
use tantivy::schema::{Field, IndexRecordOption, TantivyDocument, Term, Value as TValue};
use tantivy::tokenizer::Token;

use crate::index::{AppError, FIELD_ALL, FieldKind, ManagedIndex, parse_datetime};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SearchRequest {
    #[serde(default)]
    pub query: Option<Value>,
    #[serde(default)]
    pub from: Option<usize>,
    #[serde(default)]
    pub size: Option<usize>,
    #[serde(default)]
    pub sort: Option<Value>,
    #[serde(default, alias = "aggregations")]
    pub aggs: Option<Map<String, Value>>,
    #[serde(default)]
    pub docvalue_fields: Option<Vec<Value>>,
    #[serde(default, rename = "_source", alias = "fields")]
    pub source: Option<Value>,
    #[serde(default)]
    pub exclude_fields: Option<Vec<String>>,
    #[serde(default)]
    pub highlight: Option<Value>,
    #[serde(default)]
    pub max_field_chars: Option<usize>,
    #[serde(default)]
    pub minimum_should_match: Option<Value>,
    #[serde(default)]
    pub explain_tokens: Option<bool>,
}

struct MatchedDoc {
    index_name: String,
    doc_id: String,
    score: f32,
    source: Value,
}

pub fn execute_search(
    managed: &ManagedIndex,
    req: SearchRequest,
    q_param: Option<&str>,
) -> Result<Value, AppError> {
    execute_search_indices(&[managed], req, q_param)
}

pub fn execute_search_arc(
    indices: &[Arc<ManagedIndex>],
    req: SearchRequest,
    q_param: Option<&str>,
) -> Result<Value, AppError> {
    let refs: Vec<&ManagedIndex> = indices.iter().map(|a| a.as_ref()).collect();
    execute_search_indices(&refs, req, q_param)
}

pub fn execute_search_indices(
    indices: &[&ManagedIndex],
    req: SearchRequest,
    q_param: Option<&str>,
) -> Result<Value, AppError> {
    let start = Instant::now();
    let from = req.from.unwrap_or(0);
    let size = req.size.unwrap_or(10);
    let has_aggs = req.aggs.as_ref().is_some_and(|m| !m.is_empty());
    let sort_specs = parse_sort_specs(req.sort.as_ref());

    if indices.is_empty() {
        let mut res = json!({
            "took": start.elapsed().as_millis() as u64,
            "timed_out": false,
            "_shards": { "total": 0, "successful": 0, "skipped": 0, "failed": 0 },
            "hits": {
                "total": { "value": 0, "relation": "eq" },
                "max_score": null,
                "hits": []
            }
        });
        if let Some(aggs_map) = &req.aggs {
            res["aggregations"] = compute_aggregations(&[], indices, aggs_map)?;
        }
        return Ok(res);
    }

    let shards_total = indices.len();

    if size == 0 && !has_aggs {
        let mut total = 0usize;
        let mut token_stats: BTreeMap<(String, String), u64> = BTreeMap::new();
        for managed in indices {
            let query = build_index_query(
                managed,
                req.query.as_ref(),
                q_param,
                req.minimum_should_match.as_ref(),
            )?;
            let searcher = managed.reader.searcher();
            collect_query_token_df(managed, &searcher, &*query, &mut token_stats);
            total += searcher
                .search(&query, &Count)
                .map_err(|e| AppError::Internal(e.to_string()))?;
        }
        let mut res = json!({
            "took": start.elapsed().as_millis() as u64,
            "timed_out": false,
            "_shards": { "total": shards_total, "successful": shards_total, "skipped": 0, "failed": 0 },
            "hits": {
                "total": { "value": total, "relation": "eq" },
                "max_score": null,
                "hits": []
            }
        });
        if (req.explain_tokens.unwrap_or(false) || total == 0) && !token_stats.is_empty() {
            res["diagnostics"] = format_token_diagnostics(&token_stats);
        }
        return Ok(res);
    }

    let mut all_docs: Vec<MatchedDoc> = Vec::new();
    let mut total_hits = 0usize;
    let fetch_all = has_aggs || !sort_specs.is_empty() || indices.len() > 1;
    let mut token_stats: BTreeMap<(String, String), u64> = BTreeMap::new();
    let mut index_map: BTreeMap<String, &ManagedIndex> = BTreeMap::new();

    for managed in indices {
        index_map.insert(managed.name.clone(), *managed);
        let query = build_index_query(
            managed,
            req.query.as_ref(),
            q_param,
            req.minimum_should_match.as_ref(),
        )?;
        let searcher = managed.reader.searcher();
        collect_query_token_df(managed, &searcher, &*query, &mut token_stats);
        let num_docs = searcher.num_docs() as usize;
        if num_docs == 0 {
            continue;
        }
        let (limit, offset) = if fetch_all {
            (num_docs.max(1), 0)
        } else {
            (size.max(1), from)
        };
        let (top_docs, count) = searcher
            .search(
                &query,
                &(
                    TopDocs::with_limit(limit).and_offset(offset).order_by_score(),
                    Count,
                ),
            )
            .map_err(|e| AppError::Internal(e.to_string()))?;
        total_hits += count;

        let id_field = managed.id_field();
        let source_field = managed.source_field();
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(doc_address)
                .map_err(|e| AppError::Internal(e.to_string()))?;
            let doc_id = doc
                .get_first(id_field)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let source: Value = doc
                .get_first(source_field)
                .and_then(|v| v.as_str())
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or(Value::Null);
            all_docs.push(MatchedDoc {
                index_name: managed.name.clone(),
                doc_id,
                score,
                source,
            });
        }
    }

    let aggs_val = if let Some(aggs_map) = &req.aggs {
        let doc_refs: Vec<&Value> = all_docs.iter().map(|d| &d.source).collect();
        Some(compute_aggregations(&doc_refs, indices, aggs_map)?)
    } else {
        None
    };

    if size == 0 {
        let mut res = json!({
            "took": start.elapsed().as_millis() as u64,
            "timed_out": false,
            "_shards": { "total": shards_total, "successful": shards_total, "skipped": 0, "failed": 0 },
            "hits": {
                "total": { "value": total_hits, "relation": "eq" },
                "max_score": null,
                "hits": []
            }
        });
        if let Some(aggs) = aggs_val {
            res["aggregations"] = aggs;
        }
        if (req.explain_tokens.unwrap_or(false) || total_hits == 0) && !token_stats.is_empty() {
            res["diagnostics"] = format_token_diagnostics(&token_stats);
        }
        return Ok(res);
    }

    if !sort_specs.is_empty() {
        all_docs.sort_by(|a, b| compare_docs_by_sort(a, b, &sort_specs));
    } else if indices.len() > 1 {
        all_docs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }

    let max_score = if sort_specs.is_empty() {
        all_docs.first().map(|d| d.score)
    } else {
        None
    };

    let page_docs = if fetch_all {
        all_docs.into_iter().skip(from).take(size).collect::<Vec<_>>()
    } else {
        all_docs
    };

    let highlight_cfg = HighlightConfig::from_value(req.highlight.as_ref());
    let query_terms_set: HashSet<String> = token_stats
        .keys()
        .map(|(_, tok)| tok.to_lowercase())
        .collect();

    let dv_specs = parse_docvalue_specs(req.docvalue_fields.as_deref());
    let mut hits = Vec::with_capacity(page_docs.len());
    for d in page_docs {
        let raw_source = d.source;
        let mut hit = json!({
            "_index": d.index_name,
            "_id": d.doc_id,
            "_score": if sort_specs.is_empty() { json!(d.score) } else { Value::Null },
        });
        if !dv_specs.is_empty() {
            let mut fields_map = Map::new();
            for (field_name, is_epoch_ms) in &dv_specs {
                if let Some(val) = get_nested_value(&raw_source, field_name) {
                    let formatted = if *is_epoch_ms {
                        parse_datetime(val)
                            .map(|dt| json!(dt.into_timestamp_millis()))
                            .unwrap_or_else(|_| val.clone())
                    } else {
                        val.clone()
                    };
                    fields_map.insert(field_name.clone(), json!([formatted]));
                }
            }
            if !fields_map.is_empty() {
                hit["fields"] = Value::Object(fields_map);
            }
        }
        if !sort_specs.is_empty() {
            let sort_vals: Vec<Value> = sort_specs
                .iter()
                .map(|(f, _)| extract_sort_json_value(&raw_source, f, d.score))
                .collect();
            hit["sort"] = Value::Array(sort_vals);
        }
        if let Some(hl_cfg) = &highlight_cfg {
            if !query_terms_set.is_empty() {
                if let Some(managed) = index_map.get(&d.index_name) {
                    let hl_map = build_doc_highlights(
                        managed,
                        &raw_source,
                        &query_terms_set,
                        hl_cfg,
                    );
                    if !hl_map.is_empty() {
                        hit["highlight"] = Value::Object(hl_map);
                    }
                }
            }
        }
        if let Some(filtered_source) = filter_and_truncate_source(
            &raw_source,
            req.source.as_ref(),
            req.exclude_fields.as_deref(),
            req.max_field_chars,
        ) {
            hit["_source"] = filtered_source;
        }
        hits.push(hit);
    }

    let mut res = json!({
        "took": start.elapsed().as_millis() as u64,
        "timed_out": false,
        "_shards": { "total": shards_total, "successful": shards_total, "skipped": 0, "failed": 0 },
        "hits": {
            "total": { "value": total_hits, "relation": "eq" },
            "max_score": max_score,
            "hits": hits,
        }
    });
    if let Some(aggs) = aggs_val {
        res["aggregations"] = aggs;
    }
    if (req.explain_tokens.unwrap_or(false) || total_hits == 0) && !token_stats.is_empty() {
        res["diagnostics"] = format_token_diagnostics(&token_stats);
    }
    Ok(res)
}

fn build_index_query(
    managed: &ManagedIndex,
    query_val: Option<&Value>,
    q_param: Option<&str>,
    top_min_should_match: Option<&Value>,
) -> Result<Box<dyn Query>, AppError> {
    if let Some(q_str) = q_param.filter(|s| !s.trim().is_empty()) {
        build_query_string(managed, q_str, None, top_min_should_match)
    } else if let Some(qv) = query_val {
        build_query_from_dsl_with_msm(managed, qv, top_min_should_match)
    } else {
        Ok(Box::new(AllQuery))
    }
}

pub fn build_query_from_dsl(
    managed: &ManagedIndex,
    dsl: &Value,
) -> Result<Box<dyn Query>, AppError> {
    build_query_from_dsl_with_msm(managed, dsl, None)
}

fn build_query_from_dsl_with_msm(
    managed: &ManagedIndex,
    dsl: &Value,
    default_msm: Option<&Value>,
) -> Result<Box<dyn Query>, AppError> {
    let obj = dsl
        .as_object()
        .ok_or_else(|| AppError::BadRequest("query must be a JSON object".into()))?;

    if obj.is_empty() || obj.contains_key("match_all") {
        return Ok(Box::new(AllQuery));
    }
    if let Some(m) = obj.get("match") {
        return build_match_query(managed, m, false, default_msm);
    }
    if let Some(mp) = obj.get("match_phrase") {
        return build_match_query(managed, mp, true, default_msm);
    }
    if let Some(mm) = obj.get("multi_match") {
        return build_multi_match_query(managed, mm, default_msm);
    }
    if let Some(t) = obj.get("term") {
        return build_term_query(managed, t);
    }
    if let Some(ts) = obj.get("terms") {
        return build_terms_query(managed, ts);
    }
    if let Some(w) = obj.get("wildcard") {
        return build_wildcard_query(managed, w);
    }
    if let Some(r) = obj.get("range") {
        return build_range_query(managed, r);
    }
    if let Some(b) = obj.get("bool") {
        return build_bool_query(managed, b, default_msm);
    }
    if let Some(qs) = obj.get("query_string") {
        let q = qs
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AppError::BadRequest("query_string requires 'query' string".into()))?;
        let df = qs.get("default_field").and_then(|v| v.as_str());
        let msm = qs.get("minimum_should_match").or(default_msm);
        return build_query_string(managed, q, df, msm);
    }

    Err(AppError::BadRequest(format!(
        "unsupported query clause in {obj:?}"
    )))
}

fn parse_field_and_boost(raw_field: &str) -> (&str, Option<f32>) {
    if let Some((f, b_str)) = raw_field.rsplit_once('^') {
        if let Ok(b) = b_str.parse::<f32>() {
            return (f, Some(b));
        }
    }
    (raw_field, None)
}

fn wrap_boost(query: Box<dyn Query>, boost: Option<f32>) -> Box<dyn Query> {
    match boost {
        Some(b) if (b - 1.0).abs() > f32::EPSILON => Box::new(BoostQuery::new(query, b)),
        _ => query,
    }
}

fn resolve_minimum_should_match(spec: &Value, total_clauses: usize) -> Option<usize> {
    if total_clauses == 0 {
        return None;
    }
    let raw_str = match spec {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let resolved = if i < 0 {
                    (total_clauses as i64 + i).max(1) as usize
                } else {
                    i as usize
                };
                return Some(resolved.clamp(1, total_clauses));
            }
            return None;
        }
        Value::String(s) => s.trim(),
        _ => return None,
    };

    if let Some(pct_str) = raw_str.strip_suffix('%') {
        if let Ok(pct) = pct_str.trim().parse::<f64>() {
            let resolved = if pct < 0.0 {
                let drop = ((total_clauses as f64) * (-pct / 100.0)).floor() as isize;
                (total_clauses as isize - drop).max(1) as usize
            } else {
                ((total_clauses as f64) * (pct / 100.0)).floor() as usize
            };
            return Some(resolved.clamp(1, total_clauses));
        }
    } else if let Ok(i) = raw_str.parse::<i64>() {
        let resolved = if i < 0 {
            (total_clauses as i64 + i).max(1) as usize
        } else {
            i as usize
        };
        return Some(resolved.clamp(1, total_clauses));
    }
    None
}

fn build_match_query(
    managed: &ManagedIndex,
    val: &Value,
    is_phrase: bool,
    default_msm: Option<&Value>,
) -> Result<Box<dyn Query>, AppError> {
    let obj = val
        .as_object()
        .ok_or_else(|| AppError::BadRequest("match clause must be an object".into()))?;
    let (raw_field_name, spec) = obj
        .iter()
        .next()
        .ok_or_else(|| AppError::BadRequest("match clause is empty".into()))?;
    let (field_name, field_boost) = parse_field_and_boost(raw_field_name);

    let mut operator_and = false;
    let mut analyzer_override: Option<&str> = None;
    let mut slop: u32 = 0;
    let mut boost = field_boost;
    let mut msm_spec: Option<&Value> = default_msm;

    let query_text = match spec {
        Value::String(s) => s.as_str(),
        Value::Object(map) => {
            if let Some(op) = map.get("operator").and_then(|v| v.as_str()) {
                operator_and = op.eq_ignore_ascii_case("and");
            }
            analyzer_override = map.get("analyzer").and_then(|v| v.as_str());
            if let Some(s) = map.get("slop").and_then(|v| v.as_u64()) {
                slop = s as u32;
            }
            if let Some(b) = map.get("boost").and_then(value_as_f64) {
                boost = Some(b as f32);
            }
            if let Some(msm) = map.get("minimum_should_match") {
                msm_spec = Some(msm);
            }
            map.get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AppError::BadRequest("match object missing 'query' string".into()))?
        }
        _ => {
            return Err(AppError::BadRequest(
                "match field value must be a string or object".into(),
            ));
        }
    };

    let q = build_tokenized_field_query_with_msm(
        managed,
        field_name,
        query_text,
        operator_and,
        is_phrase,
        slop,
        analyzer_override,
        msm_spec,
    )?;
    Ok(wrap_boost(q, boost))
}

fn build_multi_match_query(
    managed: &ManagedIndex,
    val: &Value,
    default_msm: Option<&Value>,
) -> Result<Box<dyn Query>, AppError> {
    let obj = val
        .as_object()
        .ok_or_else(|| AppError::BadRequest("multi_match must be an object".into()))?;
    let query_text = obj
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::BadRequest("multi_match requires 'query'".into()))?;
    let operator_and = obj
        .get("operator")
        .and_then(|v| v.as_str())
        .is_some_and(|op| op.eq_ignore_ascii_case("and"));
    let msm_spec = obj.get("minimum_should_match").or(default_msm);
    let top_boost = obj.get("boost").and_then(value_as_f64).map(|b| b as f32);

    let fields: Vec<String> = obj
        .get("fields")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if fields.is_empty() {
        let q = build_tokenized_field_query_with_msm(
            managed,
            FIELD_ALL,
            query_text,
            operator_and,
            false,
            0,
            None,
            msm_spec,
        )?;
        return Ok(wrap_boost(q, top_boost));
    }

    let mut subqueries: Vec<(Occur, Box<dyn Query>)> = Vec::with_capacity(fields.len());
    for raw_f in &fields {
        let (f, field_boost) = parse_field_and_boost(raw_f);
        let q = build_tokenized_field_query_with_msm(
            managed,
            f,
            query_text,
            operator_and,
            false,
            0,
            None,
            msm_spec,
        )?;
        subqueries.push((Occur::Should, wrap_boost(q, field_boost)));
    }
    Ok(wrap_boost(Box::new(BooleanQuery::new(subqueries)), top_boost))
}

fn build_tokenized_field_query(
    managed: &ManagedIndex,
    field_name: &str,
    query_text: &str,
    operator_and: bool,
    is_phrase: bool,
    slop: u32,
    analyzer_override: Option<&str>,
) -> Result<Box<dyn Query>, AppError> {
    build_tokenized_field_query_with_msm(
        managed,
        field_name,
        query_text,
        operator_and,
        is_phrase,
        slop,
        analyzer_override,
        None,
    )
}

fn build_tokenized_field_query_with_msm(
    managed: &ManagedIndex,
    field_name: &str,
    query_text: &str,
    operator_and: bool,
    is_phrase: bool,
    slop: u32,
    analyzer_override: Option<&str>,
    msm_spec: Option<&Value>,
) -> Result<Box<dyn Query>, AppError> {
    let schema = managed.index.schema();
    let (field, analyzer_name) = if let Ok(f) = schema.get_field(field_name) {
        let an = analyzer_override.unwrap_or_else(|| {
            managed
                .meta
                .mappings
                .properties
                .get(field_name)
                .map(|m| m.effective_search_analyzer())
                .unwrap_or("sudachi")
        });
        (f, an)
    } else {
        (managed.all_field(), analyzer_override.unwrap_or("sudachi"))
    };

    let mut analyzer = managed
        .index
        .tokenizers()
        .get(analyzer_name)
        .or_else(|| managed.index.tokenizers().get("sudachi"))
        .ok_or_else(|| AppError::BadRequest(format!("unknown analyzer [{analyzer_name}]")))?;

    let mut tokens: Vec<Token> = Vec::new();
    let mut stream = analyzer.token_stream(query_text);
    stream.process(&mut |tok| tokens.push(tok.clone()));

    if tokens.is_empty() {
        return Ok(Box::new(EmptyQuery));
    }

    if tokens.len() == 1 {
        let term = Term::from_field_text(field, &tokens[0].text);
        return Ok(Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)));
    }

    if is_phrase {
        let terms: Vec<(usize, Term)> = tokens
            .into_iter()
            .map(|t| (t.position, Term::from_field_text(field, &t.text)))
            .collect();
        let mut pq = PhraseQuery::new_with_offset(terms);
        pq.set_slop(slop);
        return Ok(Box::new(pq));
    }

    let occur = if operator_and {
        Occur::Must
    } else {
        Occur::Should
    };
    let num_tokens = tokens.len();
    let clauses: Vec<(Occur, Box<dyn Query>)> = tokens
        .into_iter()
        .map(|t| {
            let term = Term::from_field_text(field, &t.text);
            let q: Box<dyn Query> = Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs));
            (occur, q)
        })
        .collect();

    if !operator_and {
        if let Some(msm) = msm_spec.and_then(|s| resolve_minimum_should_match(s, num_tokens)) {
            if msm > 1 {
                return Ok(Box::new(BooleanQuery::with_minimum_required_clauses(
                    clauses, msm,
                )));
            }
        }
    }
    Ok(Box::new(BooleanQuery::new(clauses)))
}

fn build_term_query(managed: &ManagedIndex, val: &Value) -> Result<Box<dyn Query>, AppError> {
    let obj = val
        .as_object()
        .ok_or_else(|| AppError::BadRequest("term clause must be an object".into()))?;
    let (raw_field_name, raw_spec) = obj
        .iter()
        .next()
        .ok_or_else(|| AppError::BadRequest("term clause is empty".into()))?;
    let (field_name, field_boost) = parse_field_and_boost(raw_field_name);

    if managed.index.schema().get_field(field_name).is_err() {
        return Ok(Box::new(EmptyQuery));
    }

    let mut boost = field_boost;
    let target_val = if let Some(map) = raw_spec.as_object() {
        if let Some(b) = map.get("boost").and_then(value_as_f64) {
            boost = Some(b as f32);
        }
        map.get("value").unwrap_or(raw_spec)
    } else {
        raw_spec
    };

    let term = make_term_for_field(managed, field_name, target_val)?;
    let q: Box<dyn Query> = Box::new(TermQuery::new(term, IndexRecordOption::Basic));
    Ok(wrap_boost(q, boost))
}

fn build_terms_query(managed: &ManagedIndex, val: &Value) -> Result<Box<dyn Query>, AppError> {
    let obj = val
        .as_object()
        .ok_or_else(|| AppError::BadRequest("terms clause must be an object".into()))?;
    let (field_name, arr_val) = obj
        .iter()
        .next()
        .ok_or_else(|| AppError::BadRequest("terms clause is empty".into()))?;
    let arr = arr_val
        .as_array()
        .ok_or_else(|| AppError::BadRequest("terms field value must be an array".into()))?;

    if arr.is_empty() || managed.index.schema().get_field(field_name).is_err() {
        return Ok(Box::new(EmptyQuery));
    }

    let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::with_capacity(arr.len());
    for item in arr {
        let term = make_term_for_field(managed, field_name, item)?;
        clauses.push((
            Occur::Should,
            Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
        ));
    }
    Ok(Box::new(BooleanQuery::new(clauses)))
}

fn build_wildcard_query(managed: &ManagedIndex, val: &Value) -> Result<Box<dyn Query>, AppError> {
    let obj = val
        .as_object()
        .ok_or_else(|| AppError::BadRequest("wildcard clause must be an object".into()))?;
    let (field_name, raw_spec) = obj
        .iter()
        .next()
        .ok_or_else(|| AppError::BadRequest("wildcard clause is empty".into()))?;
    let pat = raw_spec
        .as_str()
        .or_else(|| {
            raw_spec
                .as_object()
                .and_then(|m| m.get("value").or_else(|| m.get("wildcard")))
                .and_then(|v| v.as_str())
        })
        .unwrap_or("*");
    let cleaned = pat.trim_matches('*');
    if cleaned.is_empty() {
        return Ok(Box::new(AllQuery));
    }
    build_tokenized_field_query(managed, field_name, cleaned, false, false, 0, None)
}

fn build_range_query(managed: &ManagedIndex, val: &Value) -> Result<Box<dyn Query>, AppError> {
    let obj = val
        .as_object()
        .ok_or_else(|| AppError::BadRequest("range clause must be an object".into()))?;
    let (field_name, bounds_val) = obj
        .iter()
        .next()
        .ok_or_else(|| AppError::BadRequest("range clause is empty".into()))?;
    let bounds_map = bounds_val
        .as_object()
        .ok_or_else(|| AppError::BadRequest("range field value must be an object".into()))?;

    // shortcut: ignore range filter on unmapped fields (e.g. Grafana default @timestamp when index uses another date field), map or alias @timestamp if strict unmapped range filtering is needed
    if managed.index.schema().get_field(field_name).is_err() {
        return Ok(Box::new(AllQuery));
    }

    let include_lower = bounds_map
        .get("include_lower")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let include_upper = bounds_map
        .get("include_upper")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let lower = if let Some(v) = bounds_map.get("gte").filter(|v| !v.is_null()) {
        Bound::Included(make_term_for_field(managed, field_name, v)?)
    } else if let Some(v) = bounds_map.get("gt").filter(|v| !v.is_null()) {
        Bound::Excluded(make_term_for_field(managed, field_name, v)?)
    } else if let Some(v) = bounds_map.get("from").filter(|v| !v.is_null()) {
        let term = make_term_for_field(managed, field_name, v)?;
        if include_lower {
            Bound::Included(term)
        } else {
            Bound::Excluded(term)
        }
    } else {
        Bound::Unbounded
    };

    let upper = if let Some(v) = bounds_map.get("lte").filter(|v| !v.is_null()) {
        Bound::Included(make_term_for_field(managed, field_name, v)?)
    } else if let Some(v) = bounds_map.get("lt").filter(|v| !v.is_null()) {
        Bound::Excluded(make_term_for_field(managed, field_name, v)?)
    } else if let Some(v) = bounds_map.get("to").filter(|v| !v.is_null()) {
        let term = make_term_for_field(managed, field_name, v)?;
        if include_upper {
            Bound::Included(term)
        } else {
            Bound::Excluded(term)
        }
    } else {
        Bound::Unbounded
    };

    if matches!(lower, Bound::Unbounded) && matches!(upper, Bound::Unbounded) {
        return Err(AppError::BadRequest(
            "range query requires at least one of gte, gt, lte, lt, from, to".into(),
        ));
    }

    Ok(Box::new(RangeQuery::new(lower, upper)))
}

fn build_bool_query(
    managed: &ManagedIndex,
    val: &Value,
    default_msm: Option<&Value>,
) -> Result<Box<dyn Query>, AppError> {
    let obj = val
        .as_object()
        .ok_or_else(|| AppError::BadRequest("bool clause must be an object".into()))?;

    let msm_spec = obj.get("minimum_should_match").or(default_msm);
    let must_clauses = parse_clause_list(managed, obj.get("must"), default_msm)?;
    let filter_clauses = parse_clause_list(managed, obj.get("filter"), default_msm)?;
    let should_clauses = parse_clause_list(managed, obj.get("should"), default_msm)?;
    let must_not_clauses = parse_clause_list(managed, obj.get("must_not"), default_msm)?;

    let mut all: Vec<(Occur, Box<dyn Query>)> = Vec::new();
    let has_positive = !must_clauses.is_empty() || !filter_clauses.is_empty();
    let num_should = should_clauses.len();
    let resolved_msm = msm_spec.and_then(|s| resolve_minimum_should_match(s, num_should));

    for q in must_clauses {
        all.push((Occur::Must, q));
    }
    for q in filter_clauses {
        all.push((Occur::Must, q));
    }

    if let Some(min_should) = resolved_msm.filter(|&k| k > 0 && num_should > 0) {
        let should_pairs: Vec<(Occur, Box<dyn Query>)> = should_clauses
            .into_iter()
            .map(|q| (Occur::Should, q))
            .collect();
        if all.is_empty() && must_not_clauses.is_empty() {
            return Ok(Box::new(BooleanQuery::with_minimum_required_clauses(
                should_pairs,
                min_should,
            )));
        }
        all.push((
            Occur::Must,
            Box::new(BooleanQuery::with_minimum_required_clauses(
                should_pairs,
                min_should,
            )),
        ));
    } else {
        for q in should_clauses {
            all.push((Occur::Should, q));
        }
    }

    if !has_positive && all.is_empty() && !must_not_clauses.is_empty() {
        all.push((Occur::Must, Box::new(AllQuery)));
    }
    for q in must_not_clauses {
        all.push((Occur::MustNot, q));
    }

    if all.is_empty() {
        return Ok(Box::new(AllQuery));
    }
    Ok(Box::new(BooleanQuery::new(all)))
}

fn parse_clause_list(
    managed: &ManagedIndex,
    clause_val: Option<&Value>,
    default_msm: Option<&Value>,
) -> Result<Vec<Box<dyn Query>>, AppError> {
    let Some(v) = clause_val else {
        return Ok(Vec::new());
    };
    match v {
        Value::Array(arr) => arr
            .iter()
            .map(|item| build_query_from_dsl_with_msm(managed, item, default_msm))
            .collect(),
        Value::Object(_) => Ok(vec![build_query_from_dsl_with_msm(
            managed,
            v,
            default_msm,
        )?]),
        _ => Err(AppError::BadRequest(
            "bool sub-clause must be an object or array of objects".into(),
        )),
    }
}

fn has_lucene_syntax(s: &str) -> bool {
    s.contains("AND")
        || s.contains("OR")
        || s.contains("NOT")
        || s.contains(':')
        || s.contains('"')
        || s.contains('(')
        || s.contains('+')
        || s.contains('-')
}

fn build_query_string(
    managed: &ManagedIndex,
    q_str: &str,
    default_field: Option<&str>,
    msm_spec: Option<&Value>,
) -> Result<Box<dyn Query>, AppError> {
    let trimmed = q_str.trim();
    if trimmed.is_empty() || trimmed == "*" || trimmed == "*:*" {
        return Ok(Box::new(AllQuery));
    }

    let schema = managed.index.schema();
    let mut default_fields: Vec<Field> = Vec::new();
    if let Some(df) = default_field {
        if let Ok(f) = schema.get_field(df) {
            default_fields.push(f);
        }
    }
    if default_fields.is_empty() {
        default_fields.push(managed.all_field());
        for (name, prop) in &managed.meta.mappings.properties {
            if matches!(prop.kind, FieldKind::Text | FieldKind::Keyword) {
                if let Ok(f) = schema.get_field(name) {
                    default_fields.push(f);
                }
            }
        }
    }

    if msm_spec.is_some() && !has_lucene_syntax(trimmed) {
        if let Some(df) = default_field {
            return build_tokenized_field_query_with_msm(
                managed, df, trimmed, false, false, 0, None, msm_spec,
            );
        }
        return build_multi_field_msm_query(managed, &default_fields, trimmed, msm_spec);
    }

    let parser = QueryParser::for_index(&managed.index, default_fields);
    match parser.parse_query(trimmed) {
        Ok(q) => Ok(q),
        Err(_) => build_tokenized_field_query_with_msm(
            managed,
            default_field.unwrap_or(FIELD_ALL),
            trimmed,
            false,
            false,
            0,
            None,
            msm_spec,
        ),
    }
}

fn build_multi_field_msm_query(
    managed: &ManagedIndex,
    fields: &[Field],
    query_text: &str,
    msm_spec: Option<&Value>,
) -> Result<Box<dyn Query>, AppError> {
    let analyzer_name = managed
        .meta
        .mappings
        .properties
        .values()
        .find(|m| m.kind == FieldKind::Text)
        .map(|m| m.effective_search_analyzer())
        .unwrap_or("sudachi");

    let mut analyzer = managed
        .index
        .tokenizers()
        .get(analyzer_name)
        .or_else(|| managed.index.tokenizers().get("sudachi"))
        .ok_or_else(|| AppError::BadRequest(format!("unknown analyzer [{analyzer_name}]")))?;

    let mut tokens: Vec<String> = Vec::new();
    let mut stream = analyzer.token_stream(query_text);
    stream.process(&mut |tok| {
        if !tok.text.is_empty() {
            tokens.push(tok.text.clone());
        }
    });

    if tokens.is_empty() {
        return Ok(Box::new(EmptyQuery));
    }

    let num_tokens = tokens.len();
    let clauses: Vec<(Occur, Box<dyn Query>)> = tokens
        .into_iter()
        .map(|tok_str| {
            let field_queries: Vec<(Occur, Box<dyn Query>)> = fields
                .iter()
                .map(|&f| {
                    let term = Term::from_field_text(f, &tok_str);
                    (
                        Occur::Should,
                        Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs))
                            as Box<dyn Query>,
                    )
                })
                .collect();
            let q: Box<dyn Query> = if field_queries.len() == 1 {
                field_queries.into_iter().next().unwrap().1
            } else {
                Box::new(BooleanQuery::new(field_queries))
            };
            (Occur::Should, q)
        })
        .collect();

    if let Some(msm) = msm_spec.and_then(|s| resolve_minimum_should_match(s, num_tokens)) {
        if msm > 1 {
            return Ok(Box::new(BooleanQuery::with_minimum_required_clauses(
                clauses, msm,
            )));
        }
    }
    Ok(Box::new(BooleanQuery::new(clauses)))
}

fn make_term_for_field(
    managed: &ManagedIndex,
    field_name: &str,
    val: &Value,
) -> Result<Term, AppError> {
    let schema = managed.index.schema();
    let field = schema
        .get_field(field_name)
        .map_err(|_| AppError::BadRequest(format!("unknown field [{field_name}]")))?;

    if field_name == "_id" {
        let s = val
            .as_str()
            .ok_or_else(|| AppError::BadRequest("_id must be string".into()))?;
        return Ok(Term::from_field_text(field, s));
    }

    let kind = managed
        .meta
        .mappings
        .properties
        .get(field_name)
        .map(|m| m.kind)
        .unwrap_or(FieldKind::Text);

    match kind {
        FieldKind::Text | FieldKind::Keyword => {
            if let Some(s) = val.as_str() {
                Ok(Term::from_field_text(field, s))
            } else {
                Ok(Term::from_field_text(field, &val.to_string()))
            }
        }
        FieldKind::Long => {
            let n = val
                .as_i64()
                .or_else(|| val.as_str().and_then(|s| s.parse::<i64>().ok()))
                .ok_or_else(|| AppError::BadRequest(format!("invalid integer term {val}")))?;
            Ok(Term::from_field_i64(field, n))
        }
        FieldKind::Double => {
            let n = val
                .as_f64()
                .or_else(|| val.as_str().and_then(|s| s.parse::<f64>().ok()))
                .ok_or_else(|| AppError::BadRequest(format!("invalid float term {val}")))?;
            Ok(Term::from_field_f64(field, n))
        }
        FieldKind::Boolean => {
            let b = val
                .as_bool()
                .or_else(|| val.as_str().and_then(|s| s.parse::<bool>().ok()))
                .ok_or_else(|| AppError::BadRequest(format!("invalid bool term {val}")))?;
            Ok(Term::from_field_bool(field, b))
        }
        FieldKind::Date => {
            let dt = parse_datetime(val)?;
            Ok(Term::from_field_date(field, dt))
        }
    }
}

fn get_nested_value<'a>(source: &'a Value, path: &str) -> Option<&'a Value> {
    if let Some(v) = source.get(path) {
        return Some(v);
    }
    let mut cur = source;
    for part in path.split('.') {
        cur = cur.get(part)?;
    }
    Some(cur)
}

fn resolve_date_value<'a>(
    source: &'a Value,
    field: &str,
    indices: &[&ManagedIndex],
) -> Option<&'a Value> {
    if let Some(v) = get_nested_value(source, field) {
        return Some(v);
    }
    if field == "@timestamp" {
        for idx in indices {
            for (name, prop) in &idx.meta.mappings.properties {
                if prop.kind == FieldKind::Date {
                    if let Some(v) = get_nested_value(source, name) {
                        return Some(v);
                    }
                }
            }
        }
    }
    None
}

fn parse_sort_specs(sort_val: Option<&Value>) -> Vec<(String, bool)> {
    let Some(val) = sort_val else {
        return Vec::new();
    };
    let items: Vec<&Value> = match val {
        Value::Array(arr) => arr.iter().collect(),
        other => vec![other],
    };
    let mut specs = Vec::new();
    for item in items {
        match item {
            Value::String(s) => {
                if s != "_doc" {
                    specs.push((s.clone(), s == "_score"));
                }
            }
            Value::Object(map) => {
                if let Some((field, cfg)) = map.iter().next() {
                    if field == "_doc" {
                        continue;
                    }
                    let desc = match cfg {
                        Value::String(dir) => dir.eq_ignore_ascii_case("desc"),
                        Value::Object(m) => m
                            .get("order")
                            .and_then(|v| v.as_str())
                            .is_some_and(|dir| dir.eq_ignore_ascii_case("desc")),
                        _ => false,
                    };
                    specs.push((field.clone(), desc));
                }
            }
            _ => {}
        }
    }
    specs
}

fn compare_docs_by_sort(
    a: &MatchedDoc,
    b: &MatchedDoc,
    specs: &[(String, bool)],
) -> std::cmp::Ordering {
    for (field, desc) in specs {
        let ord = if field == "_score" {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            compare_json_field(
                get_nested_value(&a.source, field),
                get_nested_value(&b.source, field),
            )
        };
        let ord = if *desc { ord.reverse() } else { ord };
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    std::cmp::Ordering::Equal
}

fn compare_json_field(a: Option<&Value>, b: Option<&Value>) -> std::cmp::Ordering {
    match (a, b) {
        (None, None) => std::cmp::Ordering::Equal,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (Some(va), Some(vb)) => {
            if let (Ok(da), Ok(db)) = (parse_datetime(va), parse_datetime(vb)) {
                return da.into_timestamp_millis().cmp(&db.into_timestamp_millis());
            }
            if let (Some(na), Some(nb)) = (va.as_f64(), vb.as_f64()) {
                return na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal);
            }
            let sa = va.as_str().map(String::from).unwrap_or_else(|| va.to_string());
            let sb = vb.as_str().map(String::from).unwrap_or_else(|| vb.to_string());
            sa.cmp(&sb)
        }
    }
}

fn extract_sort_json_value(source: &Value, field: &str, score: f32) -> Value {
    if field == "_score" {
        return json!(score);
    }
    let Some(val) = get_nested_value(source, field) else {
        return Value::Null;
    };
    if let Ok(dt) = parse_datetime(val) {
        return json!(dt.into_timestamp_millis());
    }
    val.clone()
}

fn parse_docvalue_specs(dv: Option<&[Value]>) -> Vec<(String, bool)> {
    let Some(arr) = dv else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in arr {
        match item {
            Value::String(s) => out.push((s.clone(), false)),
            Value::Object(map) => {
                if let Some(f) = map.get("field").and_then(|v| v.as_str()) {
                    let is_epoch = map
                        .get("format")
                        .and_then(|v| v.as_str())
                        .is_some_and(|fmt| fmt == "epoch_millis");
                    out.push((f.to_string(), is_epoch));
                }
            }
            _ => {}
        }
    }
    out
}

fn compute_aggregations(
    docs: &[&Value],
    indices: &[&ManagedIndex],
    aggs_map: &Map<String, Value>,
) -> Result<Value, AppError> {
    let mut result = Map::new();
    for (agg_name, agg_spec) in aggs_map {
        let Some(spec_obj) = agg_spec.as_object() else {
            continue;
        };
        let sub_aggs = spec_obj
            .get("aggs")
            .or_else(|| spec_obj.get("aggregations"))
            .and_then(|v| v.as_object());

        let agg_val = if let Some(dh) = spec_obj.get("date_histogram") {
            compute_date_histogram(docs, indices, dh, sub_aggs)?
        } else if let Some(h) = spec_obj.get("histogram") {
            compute_histogram(docs, indices, h, sub_aggs)?
        } else if let Some(t) = spec_obj.get("terms") {
            compute_terms_agg(docs, indices, t, sub_aggs)?
        } else if let Some(f) = spec_obj.get("filters") {
            compute_filters_agg(docs, indices, f, sub_aggs)?
        } else if spec_obj.contains_key("geohash_grid") || spec_obj.contains_key("nested") {
            json!({ "buckets": [] })
        } else {
            compute_metric_agg(docs, spec_obj)
        };
        result.insert(agg_name.clone(), agg_val);
    }
    Ok(Value::Object(result))
}

fn compute_date_histogram(
    docs: &[&Value],
    indices: &[&ManagedIndex],
    dh: &Value,
    sub_aggs: Option<&Map<String, Value>>,
) -> Result<Value, AppError> {
    let field = dh
        .get("field")
        .and_then(|v| v.as_str())
        .unwrap_or("@timestamp");
    let interval_str = dh
        .get("fixed_interval")
        .or_else(|| dh.get("calendar_interval"))
        .or_else(|| dh.get("interval"))
        .and_then(|v| v.as_str())
        .unwrap_or("1d");
    let interval_ms = parse_interval_ms(interval_str).max(1);
    let min_doc_count = dh
        .get("min_doc_count")
        .and_then(|v| v.as_u64().or_else(|| v.as_str()?.parse().ok()))
        .unwrap_or(0) as usize;

    let mut groups: BTreeMap<i64, Vec<&Value>> = BTreeMap::new();
    for doc in docs {
        let Some(val) = resolve_date_value(doc, field, indices) else {
            continue;
        };
        let Ok(dt) = parse_datetime(val) else {
            continue;
        };
        let ts = dt.into_timestamp_millis();
        let bucket_key = ts.div_euclid(interval_ms) * interval_ms;
        groups.entry(bucket_key).or_default().push(*doc);
    }

    let mut buckets = Vec::new();
    for (key, bucket_docs) in groups {
        if bucket_docs.len() < min_doc_count {
            continue;
        }
        let mut bucket_obj = Map::new();
        bucket_obj.insert("key_as_string".into(), Value::String(key.to_string()));
        bucket_obj.insert("key".into(), json!(key));
        bucket_obj.insert("doc_count".into(), json!(bucket_docs.len()));
        if let Some(subs) = sub_aggs {
            if let Value::Object(sub_res) = compute_aggregations(&bucket_docs, indices, subs)? {
                for (k, v) in sub_res {
                    bucket_obj.insert(k, v);
                }
            }
        }
        buckets.push(Value::Object(bucket_obj));
    }

    Ok(json!({ "buckets": buckets }))
}

fn compute_histogram(
    docs: &[&Value],
    indices: &[&ManagedIndex],
    h: &Value,
    sub_aggs: Option<&Map<String, Value>>,
) -> Result<Value, AppError> {
    let field = h.get("field").and_then(|v| v.as_str()).unwrap_or_default();
    let interval = h
        .get("interval")
        .and_then(|v| v.as_f64().or_else(|| v.as_str()?.parse().ok()))
        .unwrap_or(10.0)
        .max(f64::EPSILON);
    let min_doc_count = h
        .get("min_doc_count")
        .and_then(|v| v.as_u64().or_else(|| v.as_str()?.parse().ok()))
        .unwrap_or(0) as usize;

    let mut groups: BTreeMap<i64, Vec<&Value>> = BTreeMap::new();
    for doc in docs {
        let Some(val) = get_nested_value(doc, field).and_then(value_as_f64) else {
            continue;
        };
        let slot = (val / interval).floor() as i64;
        groups.entry(slot).or_default().push(*doc);
    }

    let mut buckets = Vec::new();
    for (slot, bucket_docs) in groups {
        if bucket_docs.len() < min_doc_count {
            continue;
        }
        let key = (slot as f64) * interval;
        let mut bucket_obj = Map::new();
        bucket_obj.insert("key".into(), json!(key));
        bucket_obj.insert("doc_count".into(), json!(bucket_docs.len()));
        if let Some(subs) = sub_aggs {
            if let Value::Object(sub_res) = compute_aggregations(&bucket_docs, indices, subs)? {
                for (k, v) in sub_res {
                    bucket_obj.insert(k, v);
                }
            }
        }
        buckets.push(Value::Object(bucket_obj));
    }

    Ok(json!({ "buckets": buckets }))
}

fn compute_terms_agg(
    docs: &[&Value],
    indices: &[&ManagedIndex],
    t: &Value,
    sub_aggs: Option<&Map<String, Value>>,
) -> Result<Value, AppError> {
    let field = t.get("field").and_then(|v| v.as_str()).unwrap_or_default();
    let size = t
        .get("size")
        .and_then(|v| v.as_u64().or_else(|| v.as_str()?.parse().ok()))
        .unwrap_or(10) as usize;
    let size = if size == 0 { 500 } else { size };
    let min_doc_count = t
        .get("min_doc_count")
        .and_then(|v| v.as_u64().or_else(|| v.as_str()?.parse().ok()))
        .unwrap_or(1) as usize;
    let missing_val = t.get("missing");

    let mut groups: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for doc in docs {
        match get_nested_value(doc, field) {
            Some(Value::Array(arr)) => {
                for item in arr {
                    if let Some(s) = scalar_to_string(item) {
                        groups.entry(s).or_default().push(*doc);
                    }
                }
            }
            Some(v) if !v.is_null() => {
                if let Some(s) = scalar_to_string(v) {
                    groups.entry(s).or_default().push(*doc);
                }
            }
            _ => {
                if let Some(mv) = missing_val.and_then(scalar_to_string) {
                    groups.entry(mv).or_default().push(*doc);
                }
            }
        }
    }

    let mut entries: Vec<(String, Vec<&Value>)> = groups
        .into_iter()
        .filter(|(_, ds)| ds.len() >= min_doc_count)
        .collect();

    let (order_by_key, order_asc) = t
        .get("order")
        .and_then(|v| v.as_object())
        .and_then(|m| m.iter().next())
        .map(|(k, v)| {
            let asc = v.as_str().is_some_and(|s| s.eq_ignore_ascii_case("asc"));
            (k == "_key" || k == "_term", asc)
        })
        .unwrap_or((false, false));

    entries.sort_by(|(ka, da), (kb, db)| {
        let ord = if order_by_key {
            ka.cmp(kb)
        } else {
            da.len().cmp(&db.len()).then_with(|| kb.cmp(ka))
        };
        if order_asc { ord } else { ord.reverse() }
    });
    entries.truncate(size);

    let mut buckets = Vec::with_capacity(entries.len());
    for (key, bucket_docs) in entries {
        let mut bucket_obj = Map::new();
        bucket_obj.insert("key".into(), Value::String(key));
        bucket_obj.insert("doc_count".into(), json!(bucket_docs.len()));
        if let Some(subs) = sub_aggs {
            if let Value::Object(sub_res) = compute_aggregations(&bucket_docs, indices, subs)? {
                for (k, v) in sub_res {
                    bucket_obj.insert(k, v);
                }
            }
        }
        buckets.push(Value::Object(bucket_obj));
    }

    Ok(json!({
        "doc_count_error_upper_bound": 0,
        "sum_other_doc_count": 0,
        "buckets": buckets
    }))
}

fn compute_filters_agg(
    docs: &[&Value],
    indices: &[&ManagedIndex],
    f: &Value,
    sub_aggs: Option<&Map<String, Value>>,
) -> Result<Value, AppError> {
    let mut buckets = Map::new();
    if let Some(filters_obj) = f.get("filters").and_then(|v| v.as_object()) {
        for (label, filter_dsl) in filters_obj {
            let q_str = filter_dsl
                .get("query_string")
                .and_then(|qs| qs.get("query"))
                .and_then(|v| v.as_str())
                .unwrap_or("*");
            let matched: Vec<&Value> = if q_str == "*" || q_str.trim().is_empty() {
                docs.to_vec()
            } else {
                docs.iter()
                    .copied()
                    .filter(|d| d.to_string().contains(q_str))
                    .collect()
            };
            let mut bucket_obj = Map::new();
            bucket_obj.insert("doc_count".into(), json!(matched.len()));
            if let Some(subs) = sub_aggs {
                if let Value::Object(sub_res) = compute_aggregations(&matched, indices, subs)? {
                    for (k, v) in sub_res {
                        bucket_obj.insert(k, v);
                    }
                }
            }
            buckets.insert(label.clone(), Value::Object(bucket_obj));
        }
    }
    Ok(json!({ "buckets": buckets }))
}

fn compute_metric_agg(docs: &[&Value], spec_obj: &Map<String, Value>) -> Value {
    let Some((agg_type, cfg)) = spec_obj
        .iter()
        .find(|(k, _)| *k != "aggs" && *k != "aggregations" && *k != "meta")
    else {
        return json!({ "value": null });
    };
    let field = cfg.get("field").and_then(|v| v.as_str()).unwrap_or_default();

    match agg_type.as_str() {
        "value_count" => {
            let count = docs
                .iter()
                .filter(|d| get_nested_value(d, field).is_some_and(|v| !v.is_null()))
                .count();
            json!({ "value": count })
        }
        "cardinality" => {
            let mut set = HashSet::new();
            for d in docs {
                if let Some(v) = get_nested_value(d, field).and_then(scalar_to_string) {
                    set.insert(v);
                }
            }
            json!({ "value": set.len() })
        }
        "avg" | "sum" | "min" | "max" | "stats" | "extended_stats" | "percentiles" => {
            let mut nums: Vec<f64> = docs
                .iter()
                .filter_map(|d| get_nested_value(d, field).and_then(value_as_f64))
                .collect();
            let count = nums.len();
            let sum: f64 = nums.iter().sum();
            let min = nums.iter().copied().reduce(f64::min);
            let max = nums.iter().copied().reduce(f64::max);
            let avg = if count > 0 {
                Some(sum / count as f64)
            } else {
                None
            };
            match agg_type.as_str() {
                "sum" => json!({ "value": sum }),
                "avg" => json!({ "value": avg }),
                "min" => json!({ "value": min }),
                "max" => json!({ "value": max }),
                "percentiles" => {
                    nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let percents: Vec<f64> = cfg
                        .get("percents")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(value_as_f64).collect())
                        .unwrap_or_else(|| vec![1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]);
                    let mut values = Map::new();
                    for p in percents {
                        let v = if nums.is_empty() {
                            Value::Null
                        } else {
                            let idx =
                                ((p / 100.0) * (nums.len() - 1) as f64).round() as usize;
                            json!(nums[idx.min(nums.len() - 1)])
                        };
                        values.insert(format!("{p:.1}"), v);
                    }
                    json!({ "values": values })
                }
                _ => {
                    let avg_v = avg.unwrap_or(0.0);
                    let variance = if count > 0 {
                        nums.iter().map(|x| (x - avg_v).powi(2)).sum::<f64>() / count as f64
                    } else {
                        0.0
                    };
                    let std_dev = variance.sqrt();
                    json!({
                        "count": count,
                        "min": min,
                        "max": max,
                        "avg": avg,
                        "sum": sum,
                        "sum_of_squares": nums.iter().map(|x| x * x).sum::<f64>(),
                        "variance": variance,
                        "std_deviation": std_dev,
                        "std_deviation_bounds": {
                            "upper": avg.map(|a| a + 2.0 * std_dev),
                            "lower": avg.map(|a| a - 2.0 * std_dev),
                        }
                    })
                }
            }
        }
        _ => json!({ "value": null }),
    }
}

fn parse_interval_ms(s: &str) -> i64 {
    let s = s.trim();
    if let Some(ms) = s.strip_suffix("ms") {
        return ms.parse().unwrap_or(60_000);
    }
    let (num_part, unit) = s.split_at(
        s.find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(s.len()),
    );
    let n: f64 = if num_part.is_empty() {
        1.0
    } else {
        num_part.parse().unwrap_or(1.0)
    };
    let mult: f64 = match unit {
        "s" | "second" => 1_000.0,
        "m" | "minute" => 60_000.0,
        "h" | "hour" => 3_600_000.0,
        "d" | "day" => 86_400_000.0,
        "w" | "week" => 7.0 * 86_400_000.0,
        "M" | "month" => 30.0 * 86_400_000.0,
        "q" | "quarter" => 90.0 * 86_400_000.0,
        "y" | "year" => 365.0 * 86_400_000.0,
        _ => 1.0,
    };
    (n * mult) as i64
}

fn value_as_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_i64().map(|n| n as f64))
        .or_else(|| v.as_str()?.parse().ok())
}

fn scalar_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

pub fn parse_query_params_map(params: &Map<String, Value>) -> Option<&str> {
    params.get("q").and_then(|v| v.as_str())
}

fn collect_query_token_df(
    managed: &ManagedIndex,
    searcher: &tantivy::Searcher,
    query: &dyn Query,
    out: &mut BTreeMap<(String, String), u64>,
) {
    let schema = managed.index.schema();
    let mut visited: Vec<(Field, String, Term)> = Vec::new();
    query.query_terms(&mut |term, _| {
        if let Some(s) = term.value().as_str() {
            visited.push((term.field(), s.to_string(), term.clone()));
        }
    });

    for (field, token_str, term) in visited {
        let field_name = schema.get_field_name(field).to_string();
        let df = searcher.doc_freq(&term).unwrap_or(0);
        *out.entry((field_name, token_str)).or_insert(0) += df;
    }
}

fn format_token_diagnostics(token_stats: &BTreeMap<(String, String), u64>) -> Value {
    // If `_all` is present alongside expanded default fields for the same token,
    // deduplicate by token preferring `_all` (or max doc_freq) to keep output concise.
    let has_all_for_token: HashSet<&str> = token_stats
        .keys()
        .filter(|(f, _)| f == FIELD_ALL)
        .map(|(_, t)| t.as_str())
        .collect();

    let mut items = Vec::new();
    for ((field, token), df) in token_stats {
        if field != FIELD_ALL && has_all_for_token.contains(token.as_str()) {
            continue;
        }
        items.push(json!({
            "field": field,
            "token": token,
            "doc_freq": df,
        }));
    }
    json!({ "query_tokens": items })
}

pub fn matches_field_pattern(field: &str, pattern: &str) -> bool {
    if pattern == "*" || field == pattern {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return field.starts_with(prefix);
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return field.ends_with(suffix);
    }
    field.starts_with(&format!("{pattern}."))
}

pub fn filter_and_truncate_source(
    source: &Value,
    source_spec: Option<&Value>,
    extra_excludes: Option<&[String]>,
    max_field_chars: Option<usize>,
) -> Option<Value> {
    if let Some(Value::Bool(false)) = source_spec {
        return None;
    }

    let mut includes: Vec<String> = Vec::new();
    let mut excludes: Vec<String> = Vec::new();

    if let Some(spec) = source_spec {
        match spec {
            Value::String(s) => includes.push(s.clone()),
            Value::Array(arr) => {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        includes.push(s.to_string());
                    }
                }
            }
            Value::Object(obj) => {
                if let Some(inc) = obj.get("includes").or_else(|| obj.get("include")) {
                    match inc {
                        Value::String(s) => includes.push(s.clone()),
                        Value::Array(arr) => {
                            for item in arr {
                                if let Some(s) = item.as_str() {
                                    includes.push(s.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if let Some(exc) = obj.get("excludes").or_else(|| obj.get("exclude")) {
                    match exc {
                        Value::String(s) => excludes.push(s.clone()),
                        Value::Array(arr) => {
                            for item in arr {
                                if let Some(s) = item.as_str() {
                                    excludes.push(s.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(extra) = extra_excludes {
        excludes.extend(extra.iter().cloned());
    }

    let mut out = if includes.is_empty() && excludes.is_empty() {
        source.clone()
    } else if let Some(obj) = source.as_object() {
        let mut filtered = Map::new();
        for (k, v) in obj {
            let inc_ok =
                includes.is_empty() || includes.iter().any(|p| matches_field_pattern(k, p));
            let exc_hit = excludes.iter().any(|p| matches_field_pattern(k, p));
            if inc_ok && !exc_hit {
                filtered.insert(k.clone(), v.clone());
            }
        }
        Value::Object(filtered)
    } else {
        source.clone()
    };

    if let Some(max_chars) = max_field_chars.filter(|&n| n > 0) {
        truncate_json_strings(&mut out, max_chars);
    }

    Some(out)
}

fn truncate_json_strings(val: &mut Value, max_chars: usize) {
    match val {
        Value::String(s) => {
            let total_chars = s.chars().count();
            if total_chars > max_chars {
                let total_lines = s.lines().count().max(1);
                let byte_end = s
                    .char_indices()
                    .nth(max_chars)
                    .map(|(idx, _)| idx)
                    .unwrap_or(s.len());
                let mut prefix = s[..byte_end].to_string();
                prefix.push_str(&format!(
                    "... [truncated: total {total_chars} chars, {total_lines} lines]"
                ));
                *s = prefix;
            }
        }
        Value::Array(arr) => {
            for item in arr {
                truncate_json_strings(item, max_chars);
            }
        }
        Value::Object(map) => {
            for (_, v) in map.iter_mut() {
                truncate_json_strings(v, max_chars);
            }
        }
        _ => {}
    }
}

struct HighlightConfig {
    fields: Vec<String>,
    fragment_size: usize,
    number_of_fragments: usize,
    pre_tag: String,
    post_tag: String,
    include_line_numbers: bool,
}

impl HighlightConfig {
    fn from_value(val: Option<&Value>) -> Option<Self> {
        let v = val?;
        let mut cfg = Self {
            fields: Vec::new(),
            fragment_size: 160,
            number_of_fragments: 3,
            pre_tag: "<em>".to_string(),
            post_tag: "</em>".to_string(),
            include_line_numbers: true,
        };
        match v {
            Value::Bool(false) => return None,
            Value::Bool(true) => return Some(cfg),
            Value::Object(map) => {
                if let Some(fs) = map.get("fragment_size").and_then(|x| x.as_u64()) {
                    cfg.fragment_size = (fs as usize).max(20);
                }
                if let Some(nf) = map.get("number_of_fragments").and_then(|x| x.as_u64()) {
                    cfg.number_of_fragments = (nf as usize).max(1);
                }
                if let Some(pre) = map
                    .get("pre_tags")
                    .and_then(|x| x.as_array())
                    .and_then(|a| a.first())
                    .and_then(|x| x.as_str())
                {
                    cfg.pre_tag = pre.to_string();
                }
                if let Some(post) = map
                    .get("post_tags")
                    .and_then(|x| x.as_array())
                    .and_then(|a| a.first())
                    .and_then(|x| x.as_str())
                {
                    cfg.post_tag = post.to_string();
                }
                if let Some(ln) = map.get("include_line_numbers").and_then(|x| x.as_bool()) {
                    cfg.include_line_numbers = ln;
                }
                if let Some(fields_val) = map.get("fields") {
                    match fields_val {
                        Value::Object(fmap) => {
                            for (k, fspec) in fmap {
                                cfg.fields.push(k.clone());
                                if let Some(fobj) = fspec.as_object() {
                                    if let Some(fs) =
                                        fobj.get("fragment_size").and_then(|x| x.as_u64())
                                    {
                                        cfg.fragment_size = (fs as usize).max(20);
                                    }
                                    if let Some(nf) =
                                        fobj.get("number_of_fragments").and_then(|x| x.as_u64())
                                    {
                                        cfg.number_of_fragments = (nf as usize).max(1);
                                    }
                                }
                            }
                        }
                        Value::Array(arr) => {
                            for item in arr {
                                if let Some(s) = item.as_str() {
                                    cfg.fields.push(s.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Some(cfg)
            }
            _ => None,
        }
    }
}

fn build_doc_highlights(
    managed: &ManagedIndex,
    source: &Value,
    query_terms: &HashSet<String>,
    cfg: &HighlightConfig,
) -> Map<String, Value> {
    let mut out = Map::new();
    let Some(obj) = source.as_object() else {
        return out;
    };

    for (field_name, field_val) in obj {
        if !cfg.fields.is_empty()
            && !cfg
                .fields
                .iter()
                .any(|pat| matches_field_pattern(field_name, pat))
        {
            continue;
        }
        let Some(text) = field_val.as_str() else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        let snippets = extract_field_snippets(managed, field_name, text, query_terms, cfg);
        if !snippets.is_empty() {
            out.insert(
                field_name.clone(),
                Value::Array(snippets.into_iter().map(Value::String).collect()),
            );
        }
    }
    out
}

fn extract_field_snippets(
    managed: &ManagedIndex,
    field_name: &str,
    text: &str,
    query_terms: &HashSet<String>,
    cfg: &HighlightConfig,
) -> Vec<String> {
    let analyzer_name = managed
        .meta
        .mappings
        .properties
        .get(field_name)
        .map(|m| m.index_analyzer())
        .unwrap_or("sudachi");

    let mut spans: Vec<(usize, usize)> = Vec::new();
    if let Some(mut analyzer) = managed
        .index
        .tokenizers()
        .get(analyzer_name)
        .or_else(|| managed.index.tokenizers().get("sudachi"))
    {
        let mut stream = analyzer.token_stream(text);
        stream.process(&mut |tok| {
            let lower = tok.text.to_lowercase();
            if query_terms.contains(&lower)
                && tok.offset_from < tok.offset_to
                && tok.offset_to <= text.len()
                && text.is_char_boundary(tok.offset_from)
                && text.is_char_boundary(tok.offset_to)
            {
                spans.push((tok.offset_from, tok.offset_to));
            }
        });
    }

    // Fallback literal substring match (case-insensitive for ASCII)
    let lower_text = text.to_lowercase();
    if lower_text.len() == text.len() {
        for q in query_terms {
            if q.is_empty() {
                continue;
            }
            let mut search_from = 0;
            while let Some(pos) = lower_text[search_from..].find(q.as_str()) {
                let start = search_from + pos;
                let end = start + q.len();
                if text.is_char_boundary(start) && text.is_char_boundary(end) {
                    spans.push((start, end));
                }
                search_from = end.max(search_from + 1);
                if search_from >= lower_text.len() {
                    break;
                }
            }
        }
    } else {
        for q in query_terms {
            if q.is_empty() {
                continue;
            }
            let mut search_from = 0;
            while let Some(pos) = text[search_from..].find(q.as_str()) {
                let start = search_from + pos;
                let end = start + q.len();
                spans.push((start, end));
                search_from = end.max(search_from + 1);
                if search_from >= text.len() {
                    break;
                }
            }
        }
    }

    if spans.is_empty() {
        return Vec::new();
    }

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

    let line_starts: Vec<usize> = std::iter::once(0)
        .chain(text.match_indices('\n').map(|(i, _)| i + 1))
        .collect();
    let byte_to_line = |b: usize| -> usize {
        line_starts.partition_point(|&start| start <= b).max(1)
    };

    let char_indices: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let byte_to_char_idx = |b: usize| -> usize {
        char_indices.partition_point(|&idx| idx < b)
    };
    let char_idx_to_byte = |c_idx: usize| -> usize {
        if c_idx >= char_indices.len() {
            text.len()
        } else {
            char_indices[c_idx]
        }
    };

    let half_window = (cfg.fragment_size / 2).max(10);
    let mut fragments = Vec::new();
    let mut i = 0;

    while i < merged.len() && fragments.len() < cfg.number_of_fragments {
        let first_span = merged[i];
        let first_char = byte_to_char_idx(first_span.0);
        let win_start_char = first_char.saturating_sub(half_window);
        let max_end_char = win_start_char + cfg.fragment_size;

        let mut j = i;
        while j + 1 < merged.len() {
            let next_end_char = byte_to_char_idx(merged[j + 1].1);
            if next_end_char <= max_end_char {
                j += 1;
            } else {
                break;
            }
        }

        let last_span = merged[j];
        let last_char = byte_to_char_idx(last_span.1);
        let win_end_char = (last_char + half_window).min(char_indices.len());

        let win_start = char_idx_to_byte(win_start_char);
        let win_end = char_idx_to_byte(win_end_char);

        let start_line = byte_to_line(first_span.0);
        let end_line = byte_to_line(last_span.1);

        let mut snippet = String::new();
        if cfg.include_line_numbers {
            if start_line == end_line {
                snippet.push_str(&format!("[L{start_line}] "));
            } else {
                snippet.push_str(&format!("[L{start_line}-L{end_line}] "));
            }
        }
        if win_start > 0 {
            snippet.push_str("...");
        }

        let mut cursor = win_start;
        for &(s_start, s_end) in &merged[i..=j] {
            let clamped_start = s_start.clamp(win_start, win_end);
            let clamped_end = s_end.clamp(win_start, win_end);
            if clamped_start > cursor {
                snippet.push_str(&text[cursor..clamped_start]);
            }
            if clamped_end > clamped_start {
                snippet.push_str(&cfg.pre_tag);
                snippet.push_str(&text[clamped_start..clamped_end]);
                snippet.push_str(&cfg.post_tag);
            }
            cursor = clamped_end;
        }
        if cursor < win_end {
            snippet.push_str(&text[cursor..win_end]);
        }
        if win_end < text.len() {
            snippet.push_str("...");
        }

        fragments.push(snippet);
        i = j + 1;
    }

    fragments
}

/// Extracts natural-language query text from `q` or an Elasticsearch Query DSL object
/// (`match`, `match_phrase`, `multi_match`, `query_string`, or `bool.must`/`bool.should`).
pub fn extract_search_text(query_val: Option<&Value>, q_param: Option<&str>) -> Option<String> {
    if let Some(q) = q_param.map(str::trim).filter(|s| !s.is_empty() && *s != "*" && *s != "*:*") {
        return Some(q.to_string());
    }
    let mut parts = Vec::new();
    if let Some(qv) = query_val {
        collect_dsl_search_text(qv, &mut parts);
    }
    let joined = parts.join(" ").trim().to_string();
    (!joined.is_empty()).then_some(joined)
}

fn collect_dsl_search_text(dsl: &Value, out: &mut Vec<String>) {
    let Some(obj) = dsl.as_object() else {
        return;
    };
    for key in ["match", "match_phrase"] {
        if let Some(m) = obj.get(key).and_then(|v| v.as_object()) {
            for (_, v) in m {
                if let Some(s) = v.as_str() {
                    out.push(s.to_string());
                } else if let Some(s) = v.get("query").and_then(|q| q.as_str()) {
                    out.push(s.to_string());
                }
            }
        }
    }
    if let Some(mm) = obj.get("multi_match").and_then(|v| v.as_object()) {
        if let Some(s) = mm.get("query").and_then(|q| q.as_str()) {
            out.push(s.to_string());
        }
    }
    if let Some(qs) = obj.get("query_string").and_then(|v| v.as_object()) {
        if let Some(s) = qs.get("query").and_then(|q| q.as_str()) {
            if s.trim() != "*" && s.trim() != "*:*" {
                out.push(s.to_string());
            }
        }
    }
    if let Some(b) = obj.get("bool").and_then(|v| v.as_object()) {
        for sub_key in ["must", "should"] {
            if let Some(sub) = b.get(sub_key) {
                match sub {
                    Value::Array(arr) => {
                        for item in arr {
                            collect_dsl_search_text(item, out);
                        }
                    }
                    Value::Object(_) => collect_dsl_search_text(sub, out),
                    _ => {}
                }
            }
        }
    }
}

/// Evaluates filter constraints (`bool.filter`, `bool.must_not`, or top-level `term`/`terms`/`range`)
/// against Tantivy and returns the set of matching `_id`s if any filter constraint is present.
pub fn evaluate_filter_doc_ids(
    managed: &ManagedIndex,
    query_val: Option<&Value>,
) -> Result<Option<HashSet<String>>, AppError> {
    let Some(obj) = query_val.and_then(|v| v.as_object()) else {
        return Ok(None);
    };

    let filter_query: Option<Box<dyn Query>> = if obj.contains_key("term")
        || obj.contains_key("terms")
        || obj.contains_key("range")
    {
        Some(build_query_from_dsl(managed, query_val.unwrap())?)
    } else if let Some(b) = obj.get("bool").and_then(|v| v.as_object()) {
        let filter_clauses = parse_clause_list(managed, b.get("filter"), None)?;
        let must_not_clauses = parse_clause_list(managed, b.get("must_not"), None)?;
        if filter_clauses.is_empty() && must_not_clauses.is_empty() {
            None
        } else {
            let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
            if filter_clauses.is_empty() {
                clauses.push((Occur::Must, Box::new(AllQuery)));
            } else {
                for q in filter_clauses {
                    clauses.push((Occur::Must, q));
                }
            }
            for q in must_not_clauses {
                clauses.push((Occur::MustNot, q));
            }
            Some(Box::new(BooleanQuery::new(clauses)))
        }
    } else {
        None
    };

    let Some(fq) = filter_query else {
        return Ok(None);
    };

    let searcher = managed.reader.searcher();
    let num_docs = searcher.num_docs() as usize;
    if num_docs == 0 {
        return Ok(Some(HashSet::new()));
    }
    let top_docs = searcher
        .search(&fq, &TopDocs::with_limit(num_docs.max(1)).order_by_score())
        .map_err(|e| AppError::Internal(e.to_string()))?;
    let id_field = managed.id_field();
    let mut ids = HashSet::with_capacity(top_docs.len());
    for (_, addr) in top_docs {
        let doc: TantivyDocument = searcher
            .doc(addr)
            .map_err(|e| AppError::Internal(e.to_string()))?;
        if let Some(id) = doc.get_first(id_field).and_then(|v| v.as_str()) {
            ids.insert(id.to_string());
        }
    }
    Ok(Some(ids))
}

/// Extracts a `[Lstart-Lend]` snippet from the primary text field of `source` for a given 1-based line range.
pub fn format_span_snippet(
    source: &Value,
    start_line: u32,
    end_line: u32,
    max_chars: usize,
) -> Option<(String, String)> {
    let obj = source.as_object()?;
    let (field_name, text) = if let Some(s) = obj.get("content").and_then(|v| v.as_str()) {
        ("content".to_string(), s)
    } else if let Some(s) = obj.get("body").and_then(|v| v.as_str()) {
        ("body".to_string(), s)
    } else {
        obj.iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s)))
            .max_by_key(|(_, s)| s.len())?
    };
    if text.is_empty() {
        return None;
    }
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return None;
    }
    let s = (start_line.max(1) as usize).min(lines.len());
    let e = (end_line.max(start_line).max(1) as usize).min(lines.len());
    let joined = lines[s - 1..e].join("\n");
    let total_chars = joined.chars().count();
    let body = if max_chars > 0 && total_chars > max_chars {
        let byte_end = joined
            .char_indices()
            .nth(max_chars)
            .map(|(i, _)| i)
            .unwrap_or(joined.len());
        format!("{}...", &joined[..byte_end])
    } else {
        joined
    };
    let prefix = if s == e {
        format!("[L{s}] ")
    } else {
        format!("[L{s}-L{e}] ")
    };
    Some((field_name, format!("{prefix}{body}")))
}
