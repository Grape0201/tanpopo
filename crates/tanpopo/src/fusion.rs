use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion: `score(d) = w_bm25 / (k + rank_bm25(d)) + w_vec / (k + rank_vec(d))`
    Rrf {
        #[serde(default = "default_rrf_k")]
        k: usize,
    },
    /// Convex combination of Min-Max normalized BM25 and Vector scores.
    LinearCombination {
        #[serde(default = "default_alpha")]
        alpha: f32,
    },
    /// Pure BM25 lexical search.
    Bm25,
    /// Pure dense vector search.
    Vector,
}

fn default_rrf_k() -> usize {
    60
}

fn default_alpha() -> f32 {
    0.5
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::Rrf { k: 60 }
    }
}

impl FusionStrategy {
    /// Parses optional search strategy from request JSON (`"mode": "hybrid"|"bm25"|"vector"` or `"rank": {"rrf": ...}`).
    pub fn from_request_json(req: &Value) -> Self {
        if let Some(mode) = req.get("mode").and_then(|v| v.as_str()) {
            match mode.to_ascii_lowercase().as_str() {
                "bm25" | "lexical" | "text" => return Self::Bm25,
                "vector" | "dense" | "semantic" | "knn" => return Self::Vector,
                "linear" => {
                    let alpha = req
                        .get("alpha")
                        .and_then(|v| v.as_f64())
                        .map(|a| a as f32)
                        .unwrap_or(0.5);
                    return Self::LinearCombination { alpha };
                }
                _ => {}
            }
        }
        if let Some(rrf) = req.get("rank").and_then(|r| r.get("rrf")) {
            let k = rrf
                .get("rank_constant")
                .or_else(|| rrf.get("k"))
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(60);
            return Self::Rrf { k: k.max(1) };
        }
        Self::default()
    }
}

#[derive(Debug, Clone)]
pub struct RankedCandidate {
    pub index_name: String,
    pub doc_id: String,
    pub score: f32,
    pub start_line: u32,
    pub end_line: u32,
    pub bm25_hit: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct FusedCandidate {
    pub index_name: String,
    pub doc_id: String,
    pub fused_score: f64,
    pub bm25_rank: Option<usize>,
    pub bm25_score: Option<f32>,
    pub vector_rank: Option<usize>,
    pub vector_score: Option<f32>,
    pub matched_lines: Option<(u32, u32)>,
    pub bm25_hit: Option<Value>,
}

pub fn reciprocal_rank_fusion(
    bm25_list: &[RankedCandidate],
    vector_list: &[RankedCandidate],
    k: usize,
    limit: usize,
) -> Vec<FusedCandidate> {
    let k = k.max(1);
    let mut map: HashMap<(String, String), FusedCandidate> = HashMap::new();

    for (idx, c) in bm25_list.iter().enumerate() {
        let rank = idx + 1;
        let rrf_score = 1.0 / ((k + rank) as f64);
        let key = (c.index_name.clone(), c.doc_id.clone());
        let entry = map.entry(key).or_insert_with(|| FusedCandidate {
            index_name: c.index_name.clone(),
            doc_id: c.doc_id.clone(),
            fused_score: 0.0,
            bm25_rank: None,
            bm25_score: None,
            vector_rank: None,
            vector_score: None,
            matched_lines: None,
            bm25_hit: c.bm25_hit.clone(),
        });
        entry.fused_score += rrf_score;
        entry.bm25_rank = Some(rank);
        entry.bm25_score = Some(c.score);
        if entry.bm25_hit.is_none() && c.bm25_hit.is_some() {
            entry.bm25_hit = c.bm25_hit.clone();
        }
    }

    for (idx, c) in vector_list.iter().enumerate() {
        let rank = idx + 1;
        let rrf_score = 1.0 / ((k + rank) as f64);
        let key = (c.index_name.clone(), c.doc_id.clone());
        let entry = map.entry(key).or_insert_with(|| FusedCandidate {
            index_name: c.index_name.clone(),
            doc_id: c.doc_id.clone(),
            fused_score: 0.0,
            bm25_rank: None,
            bm25_score: None,
            vector_rank: None,
            vector_score: None,
            matched_lines: None,
            bm25_hit: None,
        });
        entry.fused_score += rrf_score;
        entry.vector_rank = Some(rank);
        entry.vector_score = Some(c.score);
        if c.start_line > 0 {
            entry.matched_lines = Some((c.start_line, c.end_line.max(c.start_line)));
        }
    }

    let mut out: Vec<FusedCandidate> = map.into_values().collect();
    out.sort_by(|a, b| {
        b.fused_score
            .partial_cmp(&a.fused_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    out.truncate(limit);
    out
}

pub fn linear_combination_fusion(
    bm25_list: &[RankedCandidate],
    vector_list: &[RankedCandidate],
    alpha: f32,
    limit: usize,
) -> Vec<FusedCandidate> {
    let alpha = alpha.clamp(0.0, 1.0) as f64;
    let beta = 1.0 - alpha;

    let norm_bm25 = min_max_scores(bm25_list);
    let norm_vec = min_max_scores(vector_list);

    let mut map: HashMap<(String, String), FusedCandidate> = HashMap::new();

    for (idx, (c, norm)) in bm25_list.iter().zip(norm_bm25.into_iter()).enumerate() {
        let key = (c.index_name.clone(), c.doc_id.clone());
        let entry = map.entry(key).or_insert_with(|| FusedCandidate {
            index_name: c.index_name.clone(),
            doc_id: c.doc_id.clone(),
            fused_score: 0.0,
            bm25_rank: None,
            bm25_score: None,
            vector_rank: None,
            vector_score: None,
            matched_lines: None,
            bm25_hit: c.bm25_hit.clone(),
        });
        entry.fused_score += alpha * (norm as f64);
        entry.bm25_rank = Some(idx + 1);
        entry.bm25_score = Some(c.score);
    }

    for (idx, (c, norm)) in vector_list.iter().zip(norm_vec.into_iter()).enumerate() {
        let key = (c.index_name.clone(), c.doc_id.clone());
        let entry = map.entry(key).or_insert_with(|| FusedCandidate {
            index_name: c.index_name.clone(),
            doc_id: c.doc_id.clone(),
            fused_score: 0.0,
            bm25_rank: None,
            bm25_score: None,
            vector_rank: None,
            vector_score: None,
            matched_lines: None,
            bm25_hit: None,
        });
        entry.fused_score += beta * (norm as f64);
        entry.vector_rank = Some(idx + 1);
        entry.vector_score = Some(c.score);
        if c.start_line > 0 {
            entry.matched_lines = Some((c.start_line, c.end_line.max(c.start_line)));
        }
    }

    let mut out: Vec<FusedCandidate> = map.into_values().collect();
    out.sort_by(|a, b| {
        b.fused_score
            .partial_cmp(&a.fused_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    out.truncate(limit);
    out
}

fn min_max_scores(list: &[RankedCandidate]) -> Vec<f32> {
    if list.is_empty() {
        return Vec::new();
    }
    let min_s = list
        .iter()
        .map(|c| c.score)
        .fold(f32::INFINITY, |a, b| a.min(b));
    let max_s = list
        .iter()
        .map(|c| c.score)
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let range = max_s - min_s;
    list.iter()
        .map(|c| {
            if range <= f32::EPSILON {
                1.0
            } else {
                (c.score - min_s) / range
            }
        })
        .collect()
}
