use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TextChunk {
    pub text: String,
    /// 1-based inclusive start line number in the primary text field (0 if not line-specific).
    pub start_line: u32,
    /// 1-based inclusive end line number in the primary text field (0 if not line-specific).
    pub end_line: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    pub max_chars: usize,
    pub overlap_lines: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_chars: 1000,
            overlap_lines: 3,
        }
    }
}

/// Splits multi-line text into line-bounded chunks while preserving 1-based `[start_line, end_line]`.
/// Prefers breaking at Markdown headings (`#`) or blank lines when a chunk exceeds 60% of `max_chars`.
pub fn chunk_text_by_lines(text: &str, cfg: ChunkConfig) -> Vec<TextChunk> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let max_chars = cfg.max_chars.max(32);
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return vec![TextChunk {
            text: trimmed.to_string(),
            start_line: 1,
            end_line: 1,
        }];
    }

    let total_chars: usize = text.chars().count();
    if total_chars <= max_chars {
        return vec![TextChunk {
            text: trimmed.to_string(),
            start_line: 1,
            end_line: lines.len() as u32,
        }];
    }

    let soft_break_threshold = (max_chars * 3) / 5;
    let mut chunks = Vec::new();
    let mut start_idx = 0usize;

    while start_idx < lines.len() {
        // Skip leading blank lines for a clean chunk start
        while start_idx < lines.len() && lines[start_idx].trim().is_empty() {
            start_idx += 1;
        }
        if start_idx >= lines.len() {
            break;
        }

        let mut end_idx = start_idx;
        let mut char_acc = 0usize;
        let mut seen_non_heading = false;

        while end_idx < lines.len() {
            let line = lines[end_idx];
            let trimmed_line = line.trim();
            let is_heading = trimmed_line.starts_with('#');
            let line_chars = line.chars().count() + 1;

            if end_idx > start_idx && char_acc + line_chars > max_chars {
                break;
            }
            // Keep Markdown headings attached to the section that follows them
            if end_idx > start_idx && is_heading && seen_non_heading {
                break;
            }
            if end_idx > start_idx && char_acc >= soft_break_threshold && trimmed_line.is_empty() {
                break;
            }

            if !trimmed_line.is_empty() && !is_heading {
                seen_non_heading = true;
            }
            char_acc += line_chars;
            end_idx += 1;
        }

        // Trim trailing blank lines from the chunk range
        let mut actual_end = end_idx;
        while actual_end > start_idx + 1 && lines[actual_end - 1].trim().is_empty() {
            actual_end -= 1;
        }

        let slice = &lines[start_idx..actual_end];
        let body = slice.join("\n").trim().to_string();
        if !body.is_empty() {
            chunks.push(TextChunk {
                text: body,
                start_line: (start_idx + 1) as u32,
                end_line: actual_end as u32,
            });
        }

        if end_idx >= lines.len() {
            break;
        }

        let step = (end_idx - start_idx).saturating_sub(cfg.overlap_lines).max(1);
        start_idx += step;
    }

    chunks
}

/// Extracts `(optional_title, chunks)` from an arbitrary JSON document.
/// Prioritizes `content` or `body` as the primary line-numbered text field; falls back to
/// concatenating other string fields if neither is present.
pub fn chunk_document_json(doc: &Value, cfg: ChunkConfig) -> (Option<String>, Vec<TextChunk>) {
    let Some(obj) = doc.as_object() else {
        return (None, Vec::new());
    };

    let title = obj
        .get("title")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from);

    let primary_text = obj
        .get("content")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            obj.get("body")
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
        });

    if let Some(text) = primary_text {
        let chunks = chunk_text_by_lines(text, cfg);
        if !chunks.is_empty() {
            return (title, chunks);
        }
    }

    // Fallback: combine remaining non-underscore string fields into a single text block
    let mut parts = Vec::new();
    for (k, v) in obj {
        if k.starts_with('_') || k == "title" || k == "content" || k == "body" {
            continue;
        }
        if let Some(s) = v.as_str().map(str::trim).filter(|s| !s.is_empty()) {
            parts.push(s.to_string());
        }
    }

    if parts.is_empty() {
        if let Some(ref t) = title {
            return (
                title.clone(),
                vec![TextChunk {
                    text: t.clone(),
                    start_line: 1,
                    end_line: 1,
                }],
            );
        }
        return (None, Vec::new());
    }

    let combined = parts.join("\n");
    (title, chunk_text_by_lines(&combined, cfg))
}
