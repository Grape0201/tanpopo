use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sudachi::dic::dictionary::JapaneseDictionary;
use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::query::TermQuery;
use tantivy::schema::{
    DateOptions, DateTimePrecision, FAST, Field, IndexRecordOption, NumericOptions, STORED,
    STRING, Schema, TantivyDocument, Term, TextFieldIndexing, TextOptions, Value as TValue,
};
use tantivy::{DateTime, Index, IndexReader, IndexWriter, ReloadPolicy};
use thiserror::Error;

use crate::tokenizer::{SudachiTokenizer, SudachiTokenizerConfig, register_default_tokenizers};

pub const FIELD_ID: &str = "_id";
pub const FIELD_SOURCE: &str = "_source";
pub const FIELD_ALL: &str = "_all";
const META_FILENAME: &str = "tanpopo_meta.json";

static DOC_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Error)]
pub enum AppError {
    #[error("index [{0}] not found")]
    IndexNotFound(String),
    #[error("document [{0}] not found")]
    DocumentNotFound(String),
    #[error("index [{0}] already exists")]
    IndexAlreadyExists(String),
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("internal error: {0}")]
    Internal(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FieldKind {
    #[default]
    Text,
    Keyword,
    #[serde(alias = "integer", alias = "i64", alias = "int")]
    Long,
    #[serde(alias = "float", alias = "f64")]
    Double,
    #[serde(alias = "bool")]
    Boolean,
    Date,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMapping {
    #[serde(rename = "type", default)]
    pub kind: FieldKind,
    #[serde(default)]
    pub analyzer: Option<String>,
    #[serde(default)]
    pub search_analyzer: Option<String>,
    #[serde(default = "default_true")]
    pub stored: bool,
    #[serde(default = "default_true")]
    pub indexed: bool,
}

fn default_true() -> bool {
    true
}

impl Default for FieldMapping {
    fn default() -> Self {
        Self {
            kind: FieldKind::Text,
            analyzer: None,
            search_analyzer: None,
            stored: true,
            indexed: true,
        }
    }
}

impl FieldMapping {
    pub fn index_analyzer(&self) -> &str {
        self.analyzer.as_deref().unwrap_or(match self.kind {
            FieldKind::Text => "sudachi",
            FieldKind::Keyword => "keyword",
            _ => "raw",
        })
    }

    pub fn effective_search_analyzer(&self) -> &str {
        self.search_analyzer
            .as_deref()
            .unwrap_or_else(|| self.index_analyzer())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MappingsConfig {
    #[serde(default)]
    pub properties: BTreeMap<String, FieldMapping>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalysisSettings {
    #[serde(default)]
    pub analyzer: BTreeMap<String, SudachiTokenizerConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexSettings {
    #[serde(default)]
    pub analysis: AnalysisSettings,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CreateIndexRequest {
    #[serde(default)]
    pub settings: IndexSettings,
    #[serde(default)]
    pub mappings: MappingsConfig,
}

pub struct ManagedIndex {
    pub name: String,
    pub dir_path: PathBuf,
    pub meta: CreateIndexRequest,
    pub index: Index,
    pub writer: Mutex<IndexWriter>,
    pub reader: IndexReader,
}

impl ManagedIndex {
    pub fn id_field(&self) -> Field {
        self.index.schema().get_field(FIELD_ID).expect("_id field")
    }

    pub fn source_field(&self) -> Field {
        self.index
            .schema()
            .get_field(FIELD_SOURCE)
            .expect("_source field")
    }

    pub fn all_field(&self) -> Field {
        self.index
            .schema()
            .get_field(FIELD_ALL)
            .expect("_all field")
    }

    pub fn refresh(&self) -> Result<(), AppError> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|e| AppError::Internal(e.to_string()))?;
        writer
            .commit()
            .map_err(|e| AppError::Internal(e.to_string()))?;
        drop(writer);
        self.reader
            .reload()
            .map_err(|e| AppError::Internal(e.to_string()))?;
        Ok(())
    }

    pub fn doc_count(&self) -> u64 {
        self.reader.searcher().num_docs()
    }

    pub fn index_document(
        &self,
        id: Option<String>,
        doc_json: Value,
        refresh: bool,
    ) -> Result<(String, bool), AppError> {
        let obj = doc_json
            .as_object()
            .ok_or_else(|| AppError::BadRequest("document body must be a JSON object".into()))?;

        let doc_id = id
            .filter(|s| !s.is_empty())
            .or_else(|| {
                obj.get("_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(generate_doc_id);

        let existed = self.get_document(&doc_id)?.is_some();
        let tantivy_doc = self.build_tantivy_doc(&doc_id, &doc_json)?;

        let id_term = Term::from_field_text(self.id_field(), &doc_id);
        let mut writer = self
            .writer
            .lock()
            .map_err(|e| AppError::Internal(e.to_string()))?;
        writer.delete_term(id_term);
        writer
            .add_document(tantivy_doc)
            .map_err(|e| AppError::Internal(e.to_string()))?;
        if refresh {
            writer
                .commit()
                .map_err(|e| AppError::Internal(e.to_string()))?;
            drop(writer);
            self.reader
                .reload()
                .map_err(|e| AppError::Internal(e.to_string()))?;
        }
        Ok((doc_id, !existed))
    }

    pub fn get_document(&self, id: &str) -> Result<Option<Value>, AppError> {
        let searcher = self.reader.searcher();
        let term = Term::from_field_text(self.id_field(), id);
        let query = TermQuery::new(term, IndexRecordOption::Basic);
        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(1).order_by_score())
            .map_err(|e| AppError::Internal(e.to_string()))?;
        let Some((_, doc_address)) = top_docs.into_iter().next() else {
            return Ok(None);
        };
        let retrieved: TantivyDocument = searcher
            .doc(doc_address)
            .map_err(|e| AppError::Internal(e.to_string()))?;
        let source_str = retrieved
            .get_first(self.source_field())
            .and_then(|v| v.as_str())
            .ok_or_else(|| AppError::Internal("missing _source in document".into()))?;
        let val: Value =
            serde_json::from_str(source_str).map_err(|e| AppError::Internal(e.to_string()))?;
        Ok(Some(val))
    }

    pub fn delete_document(&self, id: &str, refresh: bool) -> Result<bool, AppError> {
        let existed = self.get_document(id)?.is_some();
        let id_term = Term::from_field_text(self.id_field(), id);
        let mut writer = self
            .writer
            .lock()
            .map_err(|e| AppError::Internal(e.to_string()))?;
        writer.delete_term(id_term);
        if refresh {
            writer
                .commit()
                .map_err(|e| AppError::Internal(e.to_string()))?;
            drop(writer);
            self.reader
                .reload()
                .map_err(|e| AppError::Internal(e.to_string()))?;
        }
        Ok(existed)
    }

    pub fn build_tantivy_doc(&self, doc_id: &str, doc_json: &Value) -> Result<TantivyDocument, AppError> {
        let obj = doc_json
            .as_object()
            .ok_or_else(|| AppError::BadRequest("document must be a JSON object".into()))?;

        let mut doc = TantivyDocument::default();
        doc.add_text(self.id_field(), doc_id);
        let raw_source =
            serde_json::to_string(doc_json).map_err(|e| AppError::Internal(e.to_string()))?;
        doc.add_text(self.source_field(), &raw_source);

        let schema = self.index.schema();
        let all_field = self.all_field();

        for (key, value) in obj {
            if key == FIELD_ID || key == FIELD_SOURCE || key == FIELD_ALL {
                continue;
            }
            add_value_to_all(&mut doc, all_field, value);

            let Some(mapping) = self.meta.mappings.properties.get(key) else {
                continue;
            };
            let Ok(field) = schema.get_field(key) else {
                continue;
            };
            add_mapped_value(&mut doc, field, &mapping.kind, value)?;
        }

        Ok(doc)
    }
}

fn add_value_to_all(doc: &mut TantivyDocument, all_field: Field, value: &Value) {
    match value {
        Value::String(s) => doc.add_text(all_field, s),
        Value::Array(arr) => {
            for item in arr {
                add_value_to_all(doc, all_field, item);
            }
        }
        Value::Object(map) => {
            for v in map.values() {
                add_value_to_all(doc, all_field, v);
            }
        }
        _ => {}
    }
}

fn add_mapped_value(
    doc: &mut TantivyDocument,
    field: Field,
    kind: &FieldKind,
    value: &Value,
) -> Result<(), AppError> {
    if value.is_null() {
        return Ok(());
    }
    if let Value::Array(items) = value {
        for item in items {
            add_mapped_value(doc, field, kind, item)?;
        }
        return Ok(());
    }

    match kind {
        FieldKind::Text | FieldKind::Keyword => {
            if let Some(s) = value.as_str() {
                doc.add_text(field, s);
            } else {
                doc.add_text(field, &value.to_string());
            }
        }
        FieldKind::Long => {
            let n = value
                .as_i64()
                .or_else(|| value.as_str().and_then(|s| s.parse::<i64>().ok()))
                .ok_or_else(|| AppError::BadRequest(format!("expected integer for field, got {value}")))?;
            doc.add_i64(field, n);
        }
        FieldKind::Double => {
            let n = value
                .as_f64()
                .or_else(|| value.as_str().and_then(|s| s.parse::<f64>().ok()))
                .ok_or_else(|| AppError::BadRequest(format!("expected float for field, got {value}")))?;
            doc.add_f64(field, n);
        }
        FieldKind::Boolean => {
            let b = value
                .as_bool()
                .or_else(|| value.as_str().and_then(|s| s.parse::<bool>().ok()))
                .ok_or_else(|| AppError::BadRequest(format!("expected boolean for field, got {value}")))?;
            doc.add_bool(field, b);
        }
        FieldKind::Date => {
            let dt = parse_datetime(value)?;
            doc.add_date(field, dt);
        }
    }
    Ok(())
}

pub fn parse_datetime(value: &Value) -> Result<DateTime, AppError> {
    if let Some(ms) = value.as_i64() {
        return Ok(DateTime::from_timestamp_millis(ms));
    }
    if let Some(s) = value.as_str() {
        if let Ok(ms) = s.parse::<i64>() {
            return Ok(DateTime::from_timestamp_millis(ms));
        }
        // Try RFC3339 or YYYY-MM-DD via time/offset format in tantivy
        // Tantivy re-exports time::OffsetDateTime via DateTime::from_utc
        if let Ok(dt) = parse_iso8601_to_millis(s) {
            return Ok(DateTime::from_timestamp_millis(dt));
        }
    }
    Err(AppError::BadRequest(format!(
        "invalid date value {value}; expected epoch millis or ISO-8601 (YYYY-MM-DD[THH:MM:SSZ])"
    )))
}

fn parse_iso8601_to_millis(s: &str) -> Result<i64, String> {
    // Supports "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS[.sss][Z|+HH:MM]"
    let s = s.trim();
    if s.len() < 10 {
        return Err("too short".into());
    }
    let year: i32 = s[0..4].parse().map_err(|_| "year")?;
    let month: u32 = s[5..7].parse().map_err(|_| "month")?;
    let day: u32 = s[8..10].parse().map_err(|_| "day")?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err("range".into());
    }
    let days = days_from_civil(year, month, day);
    if s.len() == 10 {
        return Ok(days * 86_400_000);
    }
    let rest = &s[10..];
    let rest = rest.strip_prefix('T').or_else(|| rest.strip_prefix(' ')).ok_or("T")?;
    if rest.len() < 8 {
        return Err("time".into());
    }
    let hour: i64 = rest[0..2].parse().map_err(|_| "hour")?;
    let min: i64 = rest[3..5].parse().map_err(|_| "min")?;
    let sec: i64 = rest[6..8].parse().map_err(|_| "sec")?;
    let mut total_ms = ((days * 24 + hour) * 3600 + min * 60 + sec) * 1000;

    let mut tz_part = &rest[8..];
    if let Some(after_dot) = tz_part.strip_prefix('.') {
        let digits_len = after_dot.bytes().take_while(|b| b.is_ascii_digit()).count();
        let frac_str = &after_dot[..digits_len];
        if !frac_str.is_empty() {
            let mut padded = frac_str.chars().take(3).collect::<String>();
            while padded.len() < 3 {
                padded.push('0');
            }
            if let Ok(ms) = padded.parse::<i64>() {
                total_ms += ms;
            }
        }
        tz_part = &after_dot[digits_len..];
    }

    if !tz_part.is_empty() && tz_part != "Z" && tz_part != "z" {
        let sign = if tz_part.starts_with('+') {
            1i64
        } else if tz_part.starts_with('-') {
            -1i64
        } else {
            return Err("tz".into());
        };
        let tz_rest = &tz_part[1..];
        if tz_rest.len() >= 5 {
            let tz_h: i64 = tz_rest[0..2].parse().map_err(|_| "tzh")?;
            let tz_m: i64 = tz_rest[3..5].parse().map_err(|_| "tzm")?;
            total_ms -= sign * (tz_h * 3600 + tz_m * 60) * 1000;
        }
    }
    Ok(total_ms)
}

// Howard Hinnant's civil-to-days algorithm (days since 1970-01-01)
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = y as i64 - if m <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m = m as i64;
    let d = d as i64;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn generate_doc_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = DOC_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:x}-{seq:x}")
}

fn build_tantivy_schema(meta: &CreateIndexRequest) -> Schema {
    let mut builder = Schema::builder();
    builder.add_text_field(FIELD_ID, STRING | STORED | FAST);
    builder.add_text_field(FIELD_SOURCE, STORED);

    let all_indexing = TextFieldIndexing::default()
        .set_tokenizer("sudachi")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    builder.add_text_field(
        FIELD_ALL,
        TextOptions::default().set_indexing_options(all_indexing),
    );

    for (name, prop) in &meta.mappings.properties {
        if name == FIELD_ID || name == FIELD_SOURCE || name == FIELD_ALL {
            continue;
        }
        match prop.kind {
            FieldKind::Text => {
                let mut opts = TextOptions::default();
                if prop.stored {
                    opts = opts.set_stored();
                }
                if prop.indexed {
                    let indexing = TextFieldIndexing::default()
                        .set_tokenizer(prop.index_analyzer())
                        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
                    opts = opts.set_indexing_options(indexing);
                }
                builder.add_text_field(name, opts);
            }
            FieldKind::Keyword => {
                let mut opts = STRING | FAST;
                if prop.stored {
                    opts = opts | STORED;
                }
                builder.add_text_field(name, opts);
            }
            FieldKind::Long => {
                let mut opts = NumericOptions::default().set_fast();
                if prop.indexed {
                    opts = opts.set_indexed();
                }
                if prop.stored {
                    opts = opts.set_stored();
                }
                builder.add_i64_field(name, opts);
            }
            FieldKind::Double => {
                let mut opts = NumericOptions::default().set_fast();
                if prop.indexed {
                    opts = opts.set_indexed();
                }
                if prop.stored {
                    opts = opts.set_stored();
                }
                builder.add_f64_field(name, opts);
            }
            FieldKind::Boolean => {
                let mut opts = NumericOptions::default().set_fast();
                if prop.indexed {
                    opts = opts.set_indexed();
                }
                if prop.stored {
                    opts = opts.set_stored();
                }
                builder.add_bool_field(name, opts);
            }
            FieldKind::Date => {
                let mut opts = DateOptions::default()
                    .set_fast()
                    .set_precision(DateTimePrecision::Milliseconds);
                if prop.indexed {
                    opts = opts.set_indexed();
                }
                if prop.stored {
                    opts = opts.set_stored();
                }
                builder.add_date_field(name, opts);
            }
        }
    }

    builder.build()
}

pub fn infer_mappings_from_doc(doc_json: &Value) -> CreateIndexRequest {
    let mut properties = BTreeMap::new();
    if let Some(obj) = doc_json.as_object() {
        for (k, v) in obj {
            if k.starts_with('_') {
                continue;
            }
            let kind = match v {
                Value::Bool(_) => FieldKind::Boolean,
                Value::Number(n) if n.is_i64() || n.is_u64() => FieldKind::Long,
                Value::Number(_) => FieldKind::Double,
                _ => FieldKind::Text,
            };
            properties.insert(
                k.clone(),
                FieldMapping {
                    kind,
                    ..FieldMapping::default()
                },
            );
        }
    }
    CreateIndexRequest {
        settings: IndexSettings::default(),
        mappings: MappingsConfig { properties },
    }
}

pub struct IndexManager {
    data_dir: PathBuf,
    pub dict: Arc<JapaneseDictionary>,
    writer_heap_bytes: usize,
    indices: RwLock<BTreeMap<String, Arc<ManagedIndex>>>,
}

impl IndexManager {
    pub fn new(
        data_dir: PathBuf,
        dict: Arc<JapaneseDictionary>,
        writer_heap_bytes: usize,
    ) -> Result<Self, AppError> {
        std::fs::create_dir_all(&data_dir).map_err(|e| AppError::Internal(e.to_string()))?;
        let mgr = Self {
            data_dir,
            dict,
            writer_heap_bytes: writer_heap_bytes.max(15_000_000),
            indices: RwLock::new(BTreeMap::new()),
        };
        mgr.load_existing_indices()?;
        Ok(mgr)
    }

    fn load_existing_indices(&self) -> Result<(), AppError> {
        let entries =
            std::fs::read_dir(&self.data_dir).map_err(|e| AppError::Internal(e.to_string()))?;
        let mut map = self
            .indices
            .write()
            .map_err(|e| AppError::Internal(e.to_string()))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() || !path.join("meta.json").exists() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()).map(String::from) else {
                continue;
            };
            let meta_path = path.join(META_FILENAME);
            let meta: CreateIndexRequest = if meta_path.exists() {
                let bytes =
                    std::fs::read(&meta_path).map_err(|e| AppError::Internal(e.to_string()))?;
                serde_json::from_slice(&bytes).map_err(|e| AppError::Internal(e.to_string()))?
            } else {
                CreateIndexRequest::default()
            };
            let managed = self.open_or_init_index(&name, &path, meta)?;
            map.insert(name, Arc::new(managed));
        }
        Ok(())
    }

    fn open_or_init_index(
        &self,
        name: &str,
        dir_path: &Path,
        meta: CreateIndexRequest,
    ) -> Result<ManagedIndex, AppError> {
        std::fs::create_dir_all(dir_path).map_err(|e| AppError::Internal(e.to_string()))?;
        let schema = build_tantivy_schema(&meta);
        let mmap_dir =
            MmapDirectory::open(dir_path).map_err(|e| AppError::Internal(e.to_string()))?;
        let index = Index::open_or_create(mmap_dir, schema)
            .map_err(|e| AppError::Internal(e.to_string()))?;

        register_default_tokenizers(index.tokenizers(), &self.dict);
        for (analyzer_name, cfg) in &meta.settings.analysis.analyzer {
            index.tokenizers().register(
                analyzer_name,
                SudachiTokenizer::new(self.dict.clone(), cfg.clone()),
            );
        }

        // 1 worker thread with bounded heap keeps RSS minimal
        let writer = index
            .writer_with_num_threads(1, self.writer_heap_bytes)
            .map_err(|e| AppError::Internal(e.to_string()))?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e: tantivy::TantivyError| AppError::Internal(e.to_string()))?;

        let meta_bytes =
            serde_json::to_vec_pretty(&meta).map_err(|e| AppError::Internal(e.to_string()))?;
        std::fs::write(dir_path.join(META_FILENAME), meta_bytes)
            .map_err(|e| AppError::Internal(e.to_string()))?;

        Ok(ManagedIndex {
            name: name.to_string(),
            dir_path: dir_path.to_path_buf(),
            meta,
            index,
            writer: Mutex::new(writer),
            reader,
        })
    }

    pub fn create_index(
        &self,
        name: &str,
        req: CreateIndexRequest,
    ) -> Result<Arc<ManagedIndex>, AppError> {
        validate_index_name(name)?;
        let mut map = self
            .indices
            .write()
            .map_err(|e| AppError::Internal(e.to_string()))?;
        if map.contains_key(name) {
            return Err(AppError::IndexAlreadyExists(name.to_string()));
        }
        let dir_path = self.data_dir.join(name);
        let managed = Arc::new(self.open_or_init_index(name, &dir_path, req)?);
        map.insert(name.to_string(), managed.clone());
        Ok(managed)
    }

    pub fn get_index(&self, name: &str) -> Result<Arc<ManagedIndex>, AppError> {
        let map = self
            .indices
            .read()
            .map_err(|e| AppError::Internal(e.to_string()))?;
        map.get(name)
            .cloned()
            .ok_or_else(|| AppError::IndexNotFound(name.to_string()))
    }

    pub fn get_or_create_from_doc(
        &self,
        name: &str,
        doc: &Value,
    ) -> Result<Arc<ManagedIndex>, AppError> {
        if let Ok(existing) = self.get_index(name) {
            return Ok(existing);
        }
        let inferred = infer_mappings_from_doc(doc);
        match self.create_index(name, inferred) {
            Ok(idx) => Ok(idx),
            Err(AppError::IndexAlreadyExists(_)) => self.get_index(name),
            Err(e) => Err(e),
        }
    }

    pub fn delete_index(&self, name: &str) -> Result<(), AppError> {
        let mut map = self
            .indices
            .write()
            .map_err(|e| AppError::Internal(e.to_string()))?;
        let removed = map
            .remove(name)
            .ok_or_else(|| AppError::IndexNotFound(name.to_string()))?;
        let dir_path = removed.dir_path.clone();
        drop(removed);
        if dir_path.exists() {
            std::fs::remove_dir_all(&dir_path).map_err(|e| AppError::Internal(e.to_string()))?;
        }
        Ok(())
    }

    pub fn list_indices(&self) -> Result<Vec<Value>, AppError> {
        let map = self
            .indices
            .read()
            .map_err(|e| AppError::Internal(e.to_string()))?;
        let mut list = Vec::with_capacity(map.len());
        for (name, idx) in map.iter() {
            list.push(json!({
                "index": name,
                "health": "green",
                "status": "open",
                "docs.count": idx.doc_count(),
            }));
        }
        Ok(list)
    }

    pub fn resolve_indices(
        &self,
        pattern: Option<&str>,
    ) -> Result<Vec<Arc<ManagedIndex>>, AppError> {
        let map = self
            .indices
            .read()
            .map_err(|e| AppError::Internal(e.to_string()))?;
        let pat = pattern.map(str::trim).unwrap_or("");
        if pat.is_empty() || pat == "_all" || pat == "*" {
            return Ok(map.values().cloned().collect());
        }
        let mut matched = BTreeMap::new();
        for part in pat.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if part == "_all" || part == "*" {
                for (k, v) in map.iter() {
                    matched.insert(k.clone(), v.clone());
                }
            } else if part.contains('*') {
                for (k, v) in map.iter() {
                    if wildcard_match(part, k) {
                        matched.insert(k.clone(), v.clone());
                    }
                }
            } else if let Some(idx) = map.get(part) {
                matched.insert(part.to_string(), idx.clone());
            }
        }
        Ok(matched.into_values().collect())
    }
}

fn wildcard_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let mut rem = text;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            let Some(r) = rem.strip_prefix(part) else {
                return false;
            };
            rem = r;
        } else if i == parts.len() - 1 {
            return rem.ends_with(part);
        } else {
            let Some(pos) = rem.find(part) else {
                return false;
            };
            rem = &rem[pos + part.len()..];
        }
    }
    true
}

fn validate_index_name(name: &str) -> Result<(), AppError> {
    if name.is_empty()
        || name.starts_with('_')
        || name.starts_with('.')
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
    {
        return Err(AppError::BadRequest(format!("invalid index name [{name}]")));
    }
    Ok(())
}
