use std::env;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::chunker::ChunkConfig;
use crate::{Result, VectorError};

/// Built-in prompt templates for Japanese / multilingual embedding models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedPreset {
    /// Nagoya Univ. `ruri` (`ruri-v3`, `ruri-base`, `ruri-large`, `ruri-small`).
    /// Uses `"検索クエリ: "` and `"検索文書: "`.
    Ruri,
    /// Google `EmbeddingGemma` (`embeddinggemma-300m`).
    /// Uses `"task: search result | query: "` and `"title: {title} | text: "`.
    Gemma,
    /// Custom or raw pass-through prefixes.
    Custom {
        query_prefix: String,
        passage_prefix: String,
    },
}

impl EmbedPreset {
    pub fn from_model_name(model: &str) -> Self {
        let lower = model.to_ascii_lowercase();
        if lower.contains("ruri") {
            Self::Ruri
        } else if lower.contains("gemma") {
            Self::Gemma
        } else {
            Self::Custom {
                query_prefix: String::new(),
                passage_prefix: String::new(),
            }
        }
    }

    pub fn format_query(&self, query: &str) -> String {
        match self {
            Self::Ruri => format!("検索クエリ: {}", query.trim()),
            Self::Gemma => format!("task: search result | query: {}", query.trim()),
            Self::Custom { query_prefix, .. } => {
                if query_prefix.is_empty() {
                    query.trim().to_string()
                } else {
                    format!("{query_prefix}{}", query.trim())
                }
            }
        }
    }

    pub fn format_passage(&self, title: Option<&str>, text: &str) -> String {
        let clean_title = title.map(str::trim).filter(|t| !t.is_empty());
        let clean_text = text.trim();
        match self {
            Self::Ruri => match clean_title {
                Some(t) => format!("検索文書: {t}\n{clean_text}"),
                None => format!("検索文書: {clean_text}"),
            },
            Self::Gemma => {
                let t = clean_title.unwrap_or("none");
                format!("title: {t} | text: {clean_text}")
            }
            Self::Custom { passage_prefix, .. } => match clean_title {
                Some(t) => format!("{passage_prefix}{t}\n{clean_text}"),
                None => format!("{passage_prefix}{clean_text}"),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbedConfig {
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub preset: EmbedPreset,
    pub batch_size: usize,
    pub chunk_config: ChunkConfig,
}

impl EmbedConfig {
    pub fn new(url: impl Into<String>, model: impl Into<String>) -> Self {
        let url = url.into();
        let model = model.into();
        let preset = EmbedPreset::from_model_name(&model);
        Self {
            url,
            model,
            api_key: None,
            preset,
            batch_size: 32,
            chunk_config: ChunkConfig::default(),
        }
    }

    /// Reads embedding configuration from `TANPOPO_EMBED_*` environment variables.
    /// Returns `None` if `TANPOPO_EMBED_URL` is unset or empty.
    pub fn from_env() -> Option<Self> {
        let url = env::var("TANPOPO_EMBED_URL")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())?;
        let model = env::var("TANPOPO_EMBED_MODEL")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "cl-nagoya/ruri-v3-310m".to_string());
        let api_key = env::var("TANPOPO_EMBED_API_KEY")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let preset = if env::var("TANPOPO_EMBED_QUERY_PREFIX").is_ok()
            || env::var("TANPOPO_EMBED_PASSAGE_PREFIX").is_ok()
        {
            EmbedPreset::Custom {
                query_prefix: env::var("TANPOPO_EMBED_QUERY_PREFIX").unwrap_or_default(),
                passage_prefix: env::var("TANPOPO_EMBED_PASSAGE_PREFIX").unwrap_or_default(),
            }
        } else {
            EmbedPreset::from_model_name(&model)
        };

        let batch_size = env::var("TANPOPO_EMBED_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(32)
            .max(1);

        let max_chars = env::var("TANPOPO_CHUNK_CHARS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1000)
            .max(100);
        let overlap_lines = env::var("TANPOPO_CHUNK_OVERLAP_LINES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3);

        Some(Self {
            url,
            model,
            api_key,
            preset,
            batch_size,
            chunk_config: ChunkConfig {
                max_chars,
                overlap_lines,
            },
        })
    }
}

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>>;
    async fn embed_passages(&self, title: Option<&str>, passages: &[String]) -> Result<Vec<Vec<f32>>>;
    fn chunk_config(&self) -> ChunkConfig {
        ChunkConfig::default()
    }
}

/// Lightweight OpenAI-compatible `/v1/embeddings` client.
pub struct HttpEmbedder {
    cfg: EmbedConfig,
}

impl HttpEmbedder {
    pub fn new(cfg: EmbedConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &EmbedConfig {
        &self.cfg
    }

    async fn embed_raw_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let payload = json!({
            "model": self.cfg.model,
            "input": inputs,
        });
        let body_bytes =
            serde_json::to_vec(&payload).map_err(|e| VectorError::Embed(e.to_string()))?;

        let resp_bytes = http_post_json(&self.cfg.url, self.cfg.api_key.as_deref(), &body_bytes).await?;

        #[derive(Deserialize)]
        struct EmbedItem {
            index: usize,
            embedding: Vec<f32>,
        }
        #[derive(Deserialize)]
        struct EmbedResponse {
            data: Vec<EmbedItem>,
        }

        let mut parsed: EmbedResponse = serde_json::from_slice(&resp_bytes).map_err(|e| {
            let preview = String::from_utf8_lossy(&resp_bytes);
            VectorError::Embed(format!("invalid /v1/embeddings response ({e}): {preview}"))
        })?;

        parsed.data.sort_by_key(|d| d.index);
        if parsed.data.len() != inputs.len() {
            return Err(VectorError::Embed(format!(
                "expected {} embeddings, got {}",
                inputs.len(),
                parsed.data.len()
            )));
        }
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait]
impl Embedder for HttpEmbedder {
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let formatted = self.cfg.preset.format_query(query);
        let mut res = self.embed_raw_batch(&[formatted]).await?;
        res.pop()
            .ok_or_else(|| VectorError::Embed("empty embedding response".into()))
    }

    async fn embed_passages(&self, title: Option<&str>, passages: &[String]) -> Result<Vec<Vec<f32>>> {
        if passages.is_empty() {
            return Ok(Vec::new());
        }
        let formatted: Vec<String> = passages
            .iter()
            .map(|p| self.cfg.preset.format_passage(title, p))
            .collect();

        let mut all_vectors = Vec::with_capacity(formatted.len());
        for batch in formatted.chunks(self.cfg.batch_size.max(1)) {
            let vecs = self.embed_raw_batch(batch).await?;
            all_vectors.extend(vecs);
        }
        Ok(all_vectors)
    }

    fn chunk_config(&self) -> ChunkConfig {
        self.cfg.chunk_config
    }
}

async fn http_post_json(url: &str, api_key: Option<&str>, body: &[u8]) -> Result<Vec<u8>> {
    // shortcut: HTTPS URLs delegate to system curl to avoid pulling in TLS crates; use http:// for local TEI/vLLM/Ollama
    if url.starts_with("https://") {
        let url_owned = url.to_string();
        let key_owned = api_key.map(String::from);
        let body_owned = body.to_vec();
        return tokio::task::spawn_blocking(move || {
            use std::io::Write;
            let mut cmd = std::process::Command::new("curl");
            cmd.arg("-sS")
                .arg("--fail-with-body")
                .arg("-X")
                .arg("POST")
                .arg(&url_owned)
                .arg("-H")
                .arg("Content-Type: application/json");
            if let Some(key) = key_owned {
                cmd.arg("-H").arg(format!("Authorization: Bearer {key}"));
            }
            cmd.arg("--data-binary").arg("@-");
            cmd.stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            let mut child = cmd
                .spawn()
                .map_err(|e| VectorError::Embed(format!("failed to spawn curl for https: {e}")))?;
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(&body_owned)?;
            }
            let out = child.wait_with_output()?;
            if !out.status.success() {
                let err = String::from_utf8_lossy(&out.stderr);
                return Err(VectorError::Embed(format!("HTTPS embedding error: {err}")));
            }
            Ok(out.stdout)
        })
        .await
        .map_err(|e| VectorError::Embed(e.to_string()))?;
    }

    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| VectorError::Embed(format!("unsupported URL scheme: {url}")))?;
    let (host_port, path) = match rest.find('/') {
        Some(idx) => (&rest[..idx], &rest[idx..]),
        None => (rest, "/"),
    };
    let connect_addr = if host_port.contains(':') {
        host_port.to_string()
    } else {
        format!("{host_port}:80")
    };

    let mut stream = tokio::time::timeout(Duration::from_secs(30), TcpStream::connect(&connect_addr))
        .await
        .map_err(|_| VectorError::Embed(format!("timeout connecting to {connect_addr}")))?
        .map_err(|e| VectorError::Embed(format!("failed connecting to {connect_addr}: {e}")))?;

    let mut req_header = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    if let Some(key) = api_key {
        req_header.push_str(&format!("Authorization: Bearer {key}\r\n"));
    }
    req_header.push_str("\r\n");

    stream.write_all(req_header.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await?;

    let header_end = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| VectorError::Embed("malformed HTTP response from embedding server".into()))?;

    let headers_str = String::from_utf8_lossy(&raw[..header_end]);
    let status_line = headers_str.lines().next().unwrap_or("");
    let status_code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let payload_slice = &raw[header_end + 4..];
    let is_chunked = headers_str
        .lines()
        .any(|l| l.to_ascii_lowercase().contains("transfer-encoding: chunked"));

    let decoded = if is_chunked {
        decode_chunked(payload_slice)?
    } else {
        payload_slice.to_vec()
    };

    if !(200..300).contains(&status_code) {
        let msg = String::from_utf8_lossy(&decoded);
        return Err(VectorError::Embed(format!(
            "embedding server returned HTTP {status_code}: {msg}"
        )));
    }

    Ok(decoded)
}

fn decode_chunked(mut input: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let Some(line_end) = input.windows(2).position(|w| w == b"\r\n") else {
            break;
        };
        let size_str = std::str::from_utf8(&input[..line_end])
            .unwrap_or("0")
            .split(';')
            .next()
            .unwrap_or("0")
            .trim();
        let chunk_size = usize::from_str_radix(size_str, 16)
            .map_err(|e| VectorError::Embed(format!("invalid chunked size: {e}")))?;
        input = &input[line_end + 2..];
        if chunk_size == 0 {
            break;
        }
        if input.len() < chunk_size {
            return Err(VectorError::Embed("truncated chunked HTTP body".into()));
        }
        out.extend_from_slice(&input[..chunk_size]);
        input = &input[chunk_size..];
        if input.starts_with(b"\r\n") {
            input = &input[2..];
        }
    }
    Ok(out)
}
