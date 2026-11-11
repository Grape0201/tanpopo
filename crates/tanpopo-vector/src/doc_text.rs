use serde_json::Value;

/// Representative text of a whole document, embedded as a single vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocText {
    pub title: Option<String>,
    pub text: String,
}

fn non_empty_str<'a>(obj: &'a serde_json::Map<String, Value>, key: &str) -> Option<&'a str> {
    obj.get(key)
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// Builds the text embedded for a document (1 document = 1 vector).
///
/// `title` is returned separately (the embed preset prepends it). The body is `summary` if present,
/// else the first `max_chars` characters of `content` / `body`, else the other string fields joined.
/// Falls back to the title alone; returns `None` when the document has no usable text.
pub fn doc_representative_text(doc: &Value, max_chars: usize) -> Option<DocText> {
    let obj = doc.as_object()?;
    let title = non_empty_str(obj, "title").map(String::from);

    let body: String = non_empty_str(obj, "summary")
        .or_else(|| non_empty_str(obj, "content"))
        .or_else(|| non_empty_str(obj, "body"))
        .map(String::from)
        .unwrap_or_else(|| {
            obj.iter()
                .filter(|(k, _)| !k.starts_with('_') && !matches!(k.as_str(), "title" | "summary"))
                .filter_map(|(_, v)| v.as_str().map(str::trim).filter(|s| !s.is_empty()))
                .collect::<Vec<_>>()
                .join("\n")
        });

    let text: String = body.chars().take(max_chars).collect();
    if !text.is_empty() {
        return Some(DocText { title, text });
    }
    title.clone().map(|t| DocText {
        title,
        text: t,
    })
}
