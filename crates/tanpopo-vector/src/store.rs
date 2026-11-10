use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{Result, VectorError};

const META_MAGIC: &[u8; 4] = b"TPV1";
const VEC_FILENAME: &str = "vectors.sq8";
const META_FILENAME: &str = "vectors.meta";

/// A single chunk vector to insert for a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkVectorInput {
    pub vector: Vec<f32>,
    /// 1-based start line in the document (`0` if whole-document).
    #[serde(default)]
    pub start_line: u32,
    /// 1-based end line in the document (`0` if whole-document).
    #[serde(default)]
    pub end_line: u32,
}

/// A document-level hit returned by `MmapVectorStore::search` after MaxP chunk aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorHit {
    pub doc_id: String,
    pub score: f32,
    /// Start line of the best-matching chunk (0 if not chunked).
    pub start_line: u32,
    /// End line of the best-matching chunk (0 if not chunked).
    pub end_line: u32,
}

/// Compact 16-byte in-memory descriptor per vector slot.
#[derive(Debug, Clone, Copy)]
struct SlotMeta {
    doc_idx: u32,
    start_line: u32,
    end_line: u32,
    deleted: bool,
}

struct StoreInner {
    dim: usize,
    stride: usize,
    vec_file: File,
    meta_file: File,
    mmap: Option<Mmap>,
    slots: Vec<SlotMeta>,
    doc_ids: Vec<String>,
    doc_to_idx: HashMap<String, u32>,
    doc_slots: Vec<Vec<u32>>,
    active_slots: usize,
}

/// Disk-backed `mmap` vector store using per-vector symmetric `SQ8` (`i8` + `f32` scale) quantization.
///
/// - **Memory Footprint**: Vector payloads stay in OS page cache via `memmap2::Mmap` (`0` bytes on heap).
///   Per-slot RAM metadata is only 16 bytes (~4.8 MB for 300,000 vectors).
/// - **Throughput**: Parallel `rayon` + 8-lane unrolled `i8 -> i32` SIMD dot product.
/// - **Granularity**: Supports 1-to-N chunk vectors per document and aggregates via **MaxP** at search time.
pub struct MmapVectorStore {
    dir_path: PathBuf,
    inner: RwLock<StoreInner>,
}

impl MmapVectorStore {
    pub fn open(dir_path: impl AsRef<Path>) -> Result<Self> {
        let dir_path = dir_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir_path)?;

        let vec_path = dir_path.join(VEC_FILENAME);
        let meta_path = dir_path.join(META_FILENAME);

        let vec_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&vec_path)?;
        let meta_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&meta_path)?;

        let mut inner = StoreInner {
            dim: 0,
            stride: 0,
            vec_file,
            meta_file,
            mmap: None,
            slots: Vec::new(),
            doc_ids: Vec::new(),
            doc_to_idx: HashMap::new(),
            doc_slots: Vec::new(),
            active_slots: 0,
        };

        inner.load_existing()?;
        Ok(Self {
            dir_path,
            inner: RwLock::new(inner),
        })
    }

    pub fn dim(&self) -> usize {
        self.inner.read().map(|g| g.dim).unwrap_or(0)
    }

    pub fn active_vector_count(&self) -> usize {
        self.inner.read().map(|g| g.active_slots).unwrap_or(0)
    }

    pub fn active_doc_count(&self) -> usize {
        self.inner
            .read()
            .map(|g| g.doc_slots.iter().filter(|s| !s.is_empty()).count())
            .unwrap_or(0)
    }

    /// Upsert chunk vectors for a single document, tombstoning any previous chunks for `doc_id`.
    pub fn upsert_doc(&self, doc_id: &str, chunks: &[ChunkVectorInput]) -> Result<()> {
        self.upsert_batch(&[(doc_id.to_string(), chunks.to_vec())])
    }

    /// Upsert chunk vectors for a batch of documents in a single disk flush & mmap reload.
    pub fn upsert_batch(&self, batch: &[(String, Vec<ChunkVectorInput>)]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let mut guard = self
            .inner
            .write()
            .map_err(|_| VectorError::Corrupted("poisoned lock".into()))?;
        guard.append_batch(batch)
    }

    /// Tombstone all chunk vectors belonging to `doc_id`.
    pub fn delete_doc(&self, doc_id: &str) -> Result<bool> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| VectorError::Corrupted("poisoned lock".into()))?;
        guard.mark_doc_deleted(doc_id, true)
    }

    /// Compact `vectors.sq8` and `vectors.meta` if deleted slots exceed 30% (and at least 256 slots).
    pub fn compact_if_needed(&self) -> Result<()> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| VectorError::Corrupted("poisoned lock".into()))?;
        let total = guard.slots.len();
        let deleted = total.saturating_sub(guard.active_slots);
        if deleted >= 256 && deleted * 10 >= total * 3 {
            guard.compact(&self.dir_path)?;
        }
        Ok(())
    }

    /// Force compaction of deleted vector slots.
    pub fn compact(&self) -> Result<()> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| VectorError::Corrupted("poisoned lock".into()))?;
        guard.compact(&self.dir_path)
    }

    /// Search for top-`limit` documents by maximum chunk cosine similarity (MaxP),
    /// optionally pre-filtered by `allowed_doc_ids`.
    pub fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        allowed_doc_ids: Option<&HashSet<String>>,
    ) -> Result<Vec<VectorHit>> {
        if limit == 0 || query_vector.is_empty() {
            return Ok(Vec::new());
        }

        let guard = self
            .inner
            .read()
            .map_err(|_| VectorError::Corrupted("poisoned lock".into()))?;

        if guard.active_slots == 0 || guard.dim == 0 {
            return Ok(Vec::new());
        }
        if query_vector.len() != guard.dim {
            return Err(VectorError::DimensionMismatch {
                expected: guard.dim,
                got: query_vector.len(),
            });
        }
        let Some(ref mmap) = guard.mmap else {
            return Ok(Vec::new());
        };
        let raw_bytes: &[u8] = mmap.as_ref();
        let dim = guard.dim;
        let stride = guard.stride;

        let (q_scale, q_i8) = quantize_l2_normalized(query_vector);
        if q_scale == 0.0 {
            return Ok(Vec::new());
        }

        // Fast path when pre-filter is selective: score only slots belonging to allowed_doc_ids!
        if let Some(allowed) = allowed_doc_ids {
            if allowed.is_empty() {
                return Ok(Vec::new());
            }
            if allowed.len() * 4 <= guard.doc_ids.len().max(1) {
                let mut hits = Vec::with_capacity(allowed.len());
                for doc_id in allowed {
                    let Some(&doc_idx) = guard.doc_to_idx.get(doc_id.as_str()) else {
                        continue;
                    };
                    let slots = &guard.doc_slots[doc_idx as usize];
                    if slots.is_empty() {
                        continue;
                    }
                    let mut best_score = f32::NEG_INFINITY;
                    let mut best_span = (0u32, 0u32);
                    for &slot_idx in slots {
                        let slot = &guard.slots[slot_idx as usize];
                        if slot.deleted {
                            continue;
                        }
                        let score = score_slot(raw_bytes, slot_idx as usize, stride, dim, q_scale, &q_i8);
                        if score > best_score {
                            best_score = score;
                            best_span = (slot.start_line, slot.end_line);
                        }
                    }
                    if best_score > f32::NEG_INFINITY {
                        hits.push(VectorHit {
                            doc_id: doc_id.clone(),
                            score: best_score,
                            start_line: best_span.0,
                            end_line: best_span.1,
                        });
                    }
                }
                hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                hits.truncate(limit);
                return Ok(hits);
            }
        }

        // Build allowed doc_idx mask if a broader filter was supplied
        let allowed_mask: Option<Vec<bool>> = allowed_doc_ids.map(|allowed| {
            let mut mask = vec![false; guard.doc_ids.len()];
            for id in allowed {
                if let Some(&idx) = guard.doc_to_idx.get(id.as_str()) {
                    mask[idx as usize] = true;
                }
            }
            mask
        });

        let slots = &guard.slots;
        let num_slots = slots.len();

        // Parallel scan across mmap blocks when collection is non-trivial
        let best_per_doc: HashMap<u32, (f32, u32, u32)> = if num_slots >= 2048 {
            let chunk_size = 2048;
            (0..num_slots.div_ceil(chunk_size))
                .into_par_iter()
                .map(|blk| {
                    let start = blk * chunk_size;
                    let end = (start + chunk_size).min(num_slots);
                    let mut local: HashMap<u32, (f32, u32, u32)> = HashMap::new();
                    for (slot_idx, slot) in slots[start..end].iter().enumerate() {
                        if slot.deleted {
                            continue;
                        }
                        if let Some(ref mask) = allowed_mask {
                            if !mask[slot.doc_idx as usize] {
                                continue;
                            }
                        }
                        let abs_idx = start + slot_idx;
                        let score = score_slot(raw_bytes, abs_idx, stride, dim, q_scale, &q_i8);
                        local
                            .entry(slot.doc_idx)
                            .and_modify(|e| {
                                if score > e.0 {
                                    *e = (score, slot.start_line, slot.end_line);
                                }
                            })
                            .or_insert((score, slot.start_line, slot.end_line));
                    }
                    local
                })
                .reduce(HashMap::new, |mut acc, map| {
                    for (doc_idx, cand) in map {
                        acc.entry(doc_idx)
                            .and_modify(|e| {
                                if cand.0 > e.0 {
                                    *e = cand;
                                }
                            })
                            .or_insert(cand);
                    }
                    acc
                })
        } else {
            let mut map: HashMap<u32, (f32, u32, u32)> = HashMap::with_capacity(guard.doc_ids.len());
            for (slot_idx, slot) in slots.iter().enumerate() {
                if slot.deleted {
                    continue;
                }
                if let Some(ref mask) = allowed_mask {
                    if !mask[slot.doc_idx as usize] {
                        continue;
                    }
                }
                let score = score_slot(raw_bytes, slot_idx, stride, dim, q_scale, &q_i8);
                map.entry(slot.doc_idx)
                    .and_modify(|e| {
                        if score > e.0 {
                            *e = (score, slot.start_line, slot.end_line);
                        }
                    })
                    .or_insert((score, slot.start_line, slot.end_line));
            }
            map
        };

        let mut hits: Vec<VectorHit> = best_per_doc
            .into_iter()
            .map(|(doc_idx, (score, start_line, end_line))| VectorHit {
                doc_id: guard.doc_ids[doc_idx as usize].clone(),
                score,
                start_line,
                end_line,
            })
            .collect();

        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(limit);
        Ok(hits)
    }
}

impl StoreInner {
    fn compute_stride(dim: usize) -> usize {
        4 + ((dim + 3) & !3)
    }

    fn load_existing(&mut self) -> Result<()> {
        let meta_len = self.meta_file.metadata()?.len();
        if meta_len == 0 {
            return Ok(());
        }

        self.meta_file.seek(SeekFrom::Start(0))?;
        let mut reader = BufReader::new(&self.meta_file);
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != META_MAGIC {
            return Err(VectorError::Corrupted("invalid vectors.meta magic header".into()));
        }

        let mut dim_buf = [0u8; 4];
        reader.read_exact(&mut dim_buf)?;
        let dim = u32::from_le_bytes(dim_buf) as usize;
        self.dim = dim;
        self.stride = Self::compute_stride(dim);

        self.slots.clear();
        self.doc_ids.clear();
        self.doc_to_idx.clear();
        self.doc_slots.clear();
        self.active_slots = 0;

        loop {
            let mut tag = [0u8; 1];
            match reader.read_exact(&mut tag) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(VectorError::Io(e)),
            }

            match tag[0] {
                // Record type 1: New slot [1, start_line: u32, end_line: u32, id_len: u16, id_bytes]
                1 => {
                    let mut header = [0u8; 10];
                    reader.read_exact(&mut header)?;
                    let start_line = u32::from_le_bytes(header[0..4].try_into().unwrap());
                    let end_line = u32::from_le_bytes(header[4..8].try_into().unwrap());
                    let id_len = u16::from_le_bytes(header[8..10].try_into().unwrap()) as usize;
                    let mut id_bytes = vec![0u8; id_len];
                    reader.read_exact(&mut id_bytes)?;
                    let doc_id = String::from_utf8(id_bytes)
                        .map_err(|e| VectorError::Corrupted(e.to_string()))?;

                    let doc_idx = intern_doc_id(
                        &mut self.doc_ids,
                        &mut self.doc_to_idx,
                        &mut self.doc_slots,
                        &doc_id,
                    );
                    let slot_idx = self.slots.len() as u32;
                    self.slots.push(SlotMeta {
                        doc_idx,
                        start_line,
                        end_line,
                        deleted: false,
                    });
                    self.doc_slots[doc_idx as usize].push(slot_idx);
                    self.active_slots += 1;
                }
                // Record type 2: Tombstone doc [2, id_len: u16, id_bytes]
                2 => {
                    let mut len_buf = [0u8; 2];
                    reader.read_exact(&mut len_buf)?;
                    let id_len = u16::from_le_bytes(len_buf) as usize;
                    let mut id_bytes = vec![0u8; id_len];
                    reader.read_exact(&mut id_bytes)?;
                    let doc_id = String::from_utf8(id_bytes)
                        .map_err(|e| VectorError::Corrupted(e.to_string()))?;
                    if let Some(&doc_idx) = self.doc_to_idx.get(&doc_id) {
                        for &slot_idx in &self.doc_slots[doc_idx as usize] {
                            let s = &mut self.slots[slot_idx as usize];
                            if !s.deleted {
                                s.deleted = true;
                                self.active_slots = self.active_slots.saturating_sub(1);
                            }
                        }
                        self.doc_slots[doc_idx as usize].clear();
                    }
                }
                other => {
                    return Err(VectorError::Corrupted(format!(
                        "unknown record tag {other} in vectors.meta"
                    )));
                }
            }
        }

        drop(reader);
        self.reload_mmap()?;
        Ok(())
    }

    fn ensure_initialized(&mut self, dim: usize) -> Result<()> {
        if self.dim == 0 {
            self.dim = dim;
            self.stride = Self::compute_stride(dim);
            self.meta_file.seek(SeekFrom::Start(0))?;
            self.meta_file.write_all(META_MAGIC)?;
            self.meta_file.write_all(&(dim as u32).to_le_bytes())?;
            self.meta_file.flush()?;
        } else if self.dim != dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.dim,
                got: dim,
            });
        }
        Ok(())
    }

    fn mark_doc_deleted(&mut self, doc_id: &str, flush_meta: bool) -> Result<bool> {
        let Some(&doc_idx) = self.doc_to_idx.get(doc_id) else {
            return Ok(false);
        };
        if self.doc_slots[doc_idx as usize].is_empty() {
            return Ok(false);
        }
        for &slot_idx in &self.doc_slots[doc_idx as usize] {
            let slot = &mut self.slots[slot_idx as usize];
            if !slot.deleted {
                slot.deleted = true;
                self.active_slots = self.active_slots.saturating_sub(1);
            }
        }
        self.doc_slots[doc_idx as usize].clear();

        let id_bytes = doc_id.as_bytes();
        self.meta_file.seek(SeekFrom::End(0))?;
        self.meta_file.write_all(&[2u8])?;
        self.meta_file
            .write_all(&(id_bytes.len() as u16).to_le_bytes())?;
        self.meta_file.write_all(id_bytes)?;
        if flush_meta {
            self.meta_file.flush()?;
        }
        Ok(true)
    }

    fn append_batch(&mut self, batch: &[(String, Vec<ChunkVectorInput>)]) -> Result<()> {
        let first_dim = batch
            .iter()
            .flat_map(|(_, chunks)| chunks.iter())
            .map(|c| c.vector.len())
            .find(|&d| d > 0);
        let Some(dim) = first_dim else {
            return Ok(());
        };
        self.ensure_initialized(dim)?;

        self.vec_file.seek(SeekFrom::End(0))?;
        self.meta_file.seek(SeekFrom::End(0))?;

        let stride = self.stride;
        let mut vec_writer = BufWriter::new(&mut self.vec_file);
        let mut meta_writer = BufWriter::new(&mut self.meta_file);
        let mut row_buf = vec![0u8; stride];

        for (doc_id, chunks) in batch {
            for c in chunks {
                if c.vector.len() != dim {
                    return Err(VectorError::DimensionMismatch {
                        expected: dim,
                        got: c.vector.len(),
                    });
                }
            }

            // If doc already had chunks, tombstone them first
            if let Some(&doc_idx) = self.doc_to_idx.get(doc_id.as_str()) {
                if !self.doc_slots[doc_idx as usize].is_empty() {
                    for &slot_idx in &self.doc_slots[doc_idx as usize] {
                        let slot = &mut self.slots[slot_idx as usize];
                        if !slot.deleted {
                            slot.deleted = true;
                            self.active_slots = self.active_slots.saturating_sub(1);
                        }
                    }
                    self.doc_slots[doc_idx as usize].clear();
                    let id_bytes = doc_id.as_bytes();
                    meta_writer.write_all(&[2u8])?;
                    meta_writer.write_all(&(id_bytes.len() as u16).to_le_bytes())?;
                    meta_writer.write_all(id_bytes)?;
                }
            }

            let doc_idx = intern_doc_id(
                &mut self.doc_ids,
                &mut self.doc_to_idx,
                &mut self.doc_slots,
                doc_id,
            );
            let id_bytes = doc_id.as_bytes();

            for chunk in chunks {
                if chunk.vector.is_empty() {
                    continue;
                }
                let (scale, q_vec) = quantize_l2_normalized(&chunk.vector);
                row_buf.fill(0);
                row_buf[0..4].copy_from_slice(&scale.to_le_bytes());
                let data_bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(q_vec.as_ptr() as *const u8, dim) };
                row_buf[4..4 + dim].copy_from_slice(data_bytes);
                vec_writer.write_all(&row_buf)?;

                meta_writer.write_all(&[1u8])?;
                meta_writer.write_all(&chunk.start_line.to_le_bytes())?;
                meta_writer.write_all(&chunk.end_line.to_le_bytes())?;
                meta_writer.write_all(&(id_bytes.len() as u16).to_le_bytes())?;
                meta_writer.write_all(id_bytes)?;

                let slot_idx = self.slots.len() as u32;
                self.slots.push(SlotMeta {
                    doc_idx,
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    deleted: false,
                });
                self.doc_slots[doc_idx as usize].push(slot_idx);
                self.active_slots += 1;
            }
        }

        vec_writer.flush()?;
        meta_writer.flush()?;
        drop(vec_writer);
        drop(meta_writer);

        self.reload_mmap()?;
        Ok(())
    }

    fn reload_mmap(&mut self) -> Result<()> {
        let len = self.vec_file.metadata()?.len();
        if len == 0 {
            self.mmap = None;
            return Ok(());
        }
        let expected = (self.slots.len() * self.stride) as u64;
        if len < expected {
            return Err(VectorError::Corrupted(format!(
                "vectors.sq8 truncated: len={len}, expected={expected}"
            )));
        }
        let mmap = unsafe { Mmap::map(&self.vec_file)? };
        self.mmap = Some(mmap);
        Ok(())
    }

    fn compact(&mut self, dir_path: &Path) -> Result<()> {
        if self.dim == 0 {
            return Ok(());
        }
        let tmp_vec_path = dir_path.join("vectors.sq8.tmp");
        let tmp_meta_path = dir_path.join("vectors.meta.tmp");

        {
            let mut tmp_vec = BufWriter::new(File::create(&tmp_vec_path)?);
            let mut tmp_meta = BufWriter::new(File::create(&tmp_meta_path)?);

            tmp_meta.write_all(META_MAGIC)?;
            tmp_meta.write_all(&(self.dim as u32).to_le_bytes())?;

            let mut new_slots = Vec::with_capacity(self.active_slots);
            let mut new_doc_ids = Vec::new();
            let mut new_doc_to_idx: HashMap<String, u32> = HashMap::new();
            let mut new_doc_slots: Vec<Vec<u32>> = Vec::new();

            if let Some(ref mmap) = self.mmap {
                let raw = mmap.as_ref();
                let stride = self.stride;
                for (old_idx, slot) in self.slots.iter().enumerate() {
                    if slot.deleted {
                        continue;
                    }
                    let doc_id = &self.doc_ids[slot.doc_idx as usize];
                    let new_doc_idx = if let Some(&idx) = new_doc_to_idx.get(doc_id) {
                        idx
                    } else {
                        let idx = new_doc_ids.len() as u32;
                        new_doc_ids.push(doc_id.clone());
                        new_doc_to_idx.insert(doc_id.clone(), idx);
                        new_doc_slots.push(Vec::new());
                        idx
                    };

                    let byte_offset = old_idx * stride;
                    tmp_vec.write_all(&raw[byte_offset..byte_offset + stride])?;

                    let id_bytes = doc_id.as_bytes();
                    tmp_meta.write_all(&[1u8])?;
                    tmp_meta.write_all(&slot.start_line.to_le_bytes())?;
                    tmp_meta.write_all(&slot.end_line.to_le_bytes())?;
                    tmp_meta.write_all(&(id_bytes.len() as u16).to_le_bytes())?;
                    tmp_meta.write_all(id_bytes)?;

                    let new_slot_idx = new_slots.len() as u32;
                    new_slots.push(SlotMeta {
                        doc_idx: new_doc_idx,
                        start_line: slot.start_line,
                        end_line: slot.end_line,
                        deleted: false,
                    });
                    new_doc_slots[new_doc_idx as usize].push(new_slot_idx);
                }
            }

            tmp_vec.flush()?;
            tmp_meta.flush()?;

            self.slots = new_slots;
            self.doc_ids = new_doc_ids;
            self.doc_to_idx = new_doc_to_idx;
            self.doc_slots = new_doc_slots;
            self.active_slots = self.slots.len();
        }

        self.mmap = None;
        let vec_path = dir_path.join(VEC_FILENAME);
        let meta_path = dir_path.join(META_FILENAME);
        std::fs::rename(&tmp_vec_path, &vec_path)?;
        std::fs::rename(&tmp_meta_path, &meta_path)?;

        self.vec_file = OpenOptions::new().read(true).write(true).open(&vec_path)?;
        self.meta_file = OpenOptions::new().read(true).write(true).open(&meta_path)?;
        self.reload_mmap()?;
        Ok(())
    }
}

/// L2-normalizes `v` and quantizes it into symmetric `SQ8` (`scale: f32`, `Vec<i8>`).
fn intern_doc_id(
    doc_ids: &mut Vec<String>,
    doc_to_idx: &mut HashMap<String, u32>,
    doc_slots: &mut Vec<Vec<u32>>,
    doc_id: &str,
) -> u32 {
    if let Some(&idx) = doc_to_idx.get(doc_id) {
        return idx;
    }
    let idx = doc_ids.len() as u32;
    doc_ids.push(doc_id.to_string());
    doc_to_idx.insert(doc_id.to_string(), idx);
    doc_slots.push(Vec::new());
    idx
}

fn quantize_l2_normalized(v: &[f32]) -> (f32, Vec<i8>) {
    let mut norm_sq = 0.0f32;
    for &x in v {
        norm_sq += x * x;
    }
    if norm_sq <= f32::EPSILON {
        return (0.0, vec![0i8; v.len()]);
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    let mut max_abs = 0.0f32;
    for &x in v {
        let a = (x * inv_norm).abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs <= f32::EPSILON {
        return (0.0, vec![0i8; v.len()]);
    }
    let scale = max_abs / 127.0;
    let factor = inv_norm / scale;
    let mut out = Vec::with_capacity(v.len());
    for &x in v {
        let q = (x * factor).round().clamp(-127.0, 127.0) as i8;
        out.push(q);
    }
    (scale, out)
}

#[inline(always)]
fn score_slot(
    raw_bytes: &[u8],
    slot_idx: usize,
    stride: usize,
    dim: usize,
    q_scale: f32,
    q_i8: &[i8],
) -> f32 {
    let offset = slot_idx * stride;
    let v_scale = f32::from_le_bytes([
        raw_bytes[offset],
        raw_bytes[offset + 1],
        raw_bytes[offset + 2],
        raw_bytes[offset + 3],
    ]);
    if v_scale == 0.0 {
        return 0.0;
    }
    let v_u8 = &raw_bytes[offset + 4..offset + 4 + dim];
    let v_i8: &[i8] = unsafe { std::slice::from_raw_parts(v_u8.as_ptr() as *const i8, dim) };
    let dot = dot_i8_unrolled(q_i8, v_i8);
    (dot as f32) * (q_scale * v_scale)
}

/// 8-lane unrolled integer dot product (`i8 -> i32`) auto-vectorized to NEON (`sdot`) / AVX2.
#[inline(always)]
fn dot_i8_unrolled(a: &[i8], b: &[i8]) -> i32 {
    let len = a.len();
    let chunks = len / 8;
    let mut s0 = 0i32;
    let mut s1 = 0i32;
    let mut s2 = 0i32;
    let mut s3 = 0i32;
    let mut s4 = 0i32;
    let mut s5 = 0i32;
    let mut s6 = 0i32;
    let mut s7 = 0i32;

    for i in 0..chunks {
        let base = i * 8;
        s0 += (a[base] as i32) * (b[base] as i32);
        s1 += (a[base + 1] as i32) * (b[base + 1] as i32);
        s2 += (a[base + 2] as i32) * (b[base + 2] as i32);
        s3 += (a[base + 3] as i32) * (b[base + 3] as i32);
        s4 += (a[base + 4] as i32) * (b[base + 4] as i32);
        s5 += (a[base + 5] as i32) * (b[base + 5] as i32);
        s6 += (a[base + 6] as i32) * (b[base + 6] as i32);
        s7 += (a[base + 7] as i32) * (b[base + 7] as i32);
    }

    let mut total = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    for i in (chunks * 8)..len {
        total += (a[i] as i32) * (b[i] as i32);
    }
    total
}
