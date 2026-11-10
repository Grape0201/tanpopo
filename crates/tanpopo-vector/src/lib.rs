pub mod chunker;
pub mod embed;
pub mod store;

pub use chunker::{ChunkConfig, TextChunk, chunk_document_json, chunk_text_by_lines};
pub use embed::{EmbedConfig, EmbedPreset, Embedder, HttpEmbedder};
pub use store::{ChunkVectorInput, MmapVectorStore, VectorHit};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VectorError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("embedding error: {0}")]
    Embed(String),
    #[error("corrupted vector store: {0}")]
    Corrupted(String),
}

pub type Result<T> = std::result::Result<T, VectorError>;
