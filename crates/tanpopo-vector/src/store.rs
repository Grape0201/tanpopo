use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{Result, VectorError};

const META_MAGIC: &[u8; 4] = b"TPV2";
const VEC_FILENAME: &str = "vectors.sq8";
const META_FILENAME: &str = "vectors.meta";

/// A document-level hit returned by `MmapVectorStore::search`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorHit {
    pub doc_id: String,
    pub score: f32,
}

#[derive(Debug, Clone, Copy)]
struct SlotMeta {
    doc_idx: u32,
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
    /// Active slot per document (`None` if the document has no live vector).
    doc_slot: Vec<Option<u32>>,
    active_slots: usize,
}

/// Disk-backed `mmap` vector store using per-vector symmetric `SQ8` (`i8` + `f32` scale) quantization.
///
/// One document = one vector. Vector payloads stay in OS page cache via `memmap2::Mmap`;
/// search is a parallel `rayon` + unrolled `i8 -> i32` dot product over all live slots.
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
            doc_slot: Vec::new(),
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

    /// Number of documents that currently have a live vector.
    pub fn active_doc_count(&self) -> usize {
        self.inner.read().map(|g| g.active_slots).unwrap_or(0)
    }

    /// Upsert the vector for a single document, tombstoning any previous vector for `doc_id`.
    pub fn upsert_doc(&self, doc_id: &str, vector: &[f32]) -> Result<()> {
        self.upsert_batch(&[(doc_id.to_string(), vector.to_vec())])
    }

    /// Upsert vectors for a batch of documents in a single disk flush & mmap reload.
    pub fn upsert_batch(&self, batch: &[(String, Vec<f32>)]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let mut guard = self
            .inner
            .write()
            .map_err(|_| VectorError::Corrupted("poisoned lock".into()))?;
        guard.append_batch(batch)
    }

    /// Tombstone the vector belonging to `doc_id`.
    pub fn delete_doc(&self, doc_id: &str) -> Result<bool> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| VectorError::Corrupted("poisoned lock".into()))?;
        guard.mark_doc_deleted(doc_id)
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

    /// Search for top-`limit` documents by cosine similarity,
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

        let mut scored: Vec<(f32, u32)> = if let Some(allowed) = allowed_doc_ids {
            if allowed.is_empty() {
                return Ok(Vec::new());
            }
            if allowed.len() * 4 <= guard.doc_ids.len().max(1) {
                // Selective filter: score only the allowed documents.
                allowed
                    .iter()
                    .filter_map(|id| {
                        let doc_idx = *guard.doc_to_idx.get(id.as_str())?;
                        let slot_idx = guard.doc_slot[doc_idx as usize]?;
                        let score = score_slot(raw_bytes, slot_idx as usize, stride, dim, q_scale, &q_i8);
                        Some((score, doc_idx))
                    })
                    .collect()
            } else {
                let mut mask = vec![false; guard.doc_ids.len()];
                for id in allowed {
                    if let Some(&idx) = guard.doc_to_idx.get(id.as_str()) {
                        mask[idx as usize] = true;
                    }
                }
                scan_slots(&guard.slots, raw_bytes, stride, dim, q_scale, &q_i8, Some(&mask))
            }
        } else {
            scan_slots(&guard.slots, raw_bytes, stride, dim, q_scale, &q_i8, None)
        };

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored
            .into_iter()
            .map(|(score, doc_idx)| VectorHit {
                doc_id: guard.doc_ids[doc_idx as usize].clone(),
                score,
            })
            .collect())
    }
}

/// Scores every live slot (optionally restricted by a per-doc mask); parallel for large stores.
fn scan_slots(
    slots: &[SlotMeta],
    raw_bytes: &[u8],
    stride: usize,
    dim: usize,
    q_scale: f32,
    q_i8: &[i8],
    mask: Option<&Vec<bool>>,
) -> Vec<(f32, u32)> {
    let score_one = |(slot_idx, slot): (usize, &SlotMeta)| -> Option<(f32, u32)> {
        if slot.deleted || mask.is_some_and(|m| !m[slot.doc_idx as usize]) {
            return None;
        }
        Some((
            score_slot(raw_bytes, slot_idx, stride, dim, q_scale, q_i8),
            slot.doc_idx,
        ))
    };
    if slots.len() >= 2048 {
        slots.par_iter().enumerate().filter_map(score_one).collect()
    } else {
        slots.iter().enumerate().filter_map(score_one).collect()
    }
}

impl StoreInner {
    fn compute_stride(dim: usize) -> usize {
        4 + ((dim + 3) & !3)
    }

    fn read_doc_id(reader: &mut impl Read) -> Result<String> {
        let mut len_buf = [0u8; 2];
        reader.read_exact(&mut len_buf)?;
        let mut id_bytes = vec![0u8; u16::from_le_bytes(len_buf) as usize];
        reader.read_exact(&mut id_bytes)?;
        String::from_utf8(id_bytes).map_err(|e| VectorError::Corrupted(e.to_string()))
    }

    fn intern_doc_id(&mut self, doc_id: &str) -> u32 {
        if let Some(&idx) = self.doc_to_idx.get(doc_id) {
            return idx;
        }
        let idx = self.doc_ids.len() as u32;
        self.doc_ids.push(doc_id.to_string());
        self.doc_to_idx.insert(doc_id.to_string(), idx);
        self.doc_slot.push(None);
        idx
    }

    /// Tombstones the live slot of `doc_idx` in memory. Returns whether one existed.
    fn tombstone_in_memory(&mut self, doc_idx: u32) -> bool {
        let Some(slot_idx) = self.doc_slot[doc_idx as usize].take() else {
            return false;
        };
        self.slots[slot_idx as usize].deleted = true;
        self.active_slots = self.active_slots.saturating_sub(1);
        true
    }

    /// Registers a new live slot for `doc_id`, replacing any previous one.
    fn push_slot(&mut self, doc_id: &str) {
        let doc_idx = self.intern_doc_id(doc_id);
        self.tombstone_in_memory(doc_idx);
        let slot_idx = self.slots.len() as u32;
        self.slots.push(SlotMeta {
            doc_idx,
            deleted: false,
        });
        self.doc_slot[doc_idx as usize] = Some(slot_idx);
        self.active_slots += 1;
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
            return Err(VectorError::Corrupted(
                "invalid vectors.meta magic header (old chunk-based store? delete the vector files and re-index)".into(),
            ));
        }

        let mut dim_buf = [0u8; 4];
        reader.read_exact(&mut dim_buf)?;
        let dim = u32::from_le_bytes(dim_buf) as usize;

        let mut records = Vec::new();
        loop {
            let mut tag = [0u8; 1];
            match reader.read_exact(&mut tag) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(VectorError::Io(e)),
            }
            match tag[0] {
                1 | 2 => records.push((tag[0], Self::read_doc_id(&mut reader)?)),
                other => {
                    return Err(VectorError::Corrupted(format!(
                        "unknown record tag {other} in vectors.meta"
                    )));
                }
            }
        }
        drop(reader);

        self.dim = dim;
        self.stride = Self::compute_stride(dim);
        self.slots.clear();
        self.doc_ids.clear();
        self.doc_to_idx.clear();
        self.doc_slot.clear();
        self.active_slots = 0;

        for (tag, doc_id) in records {
            if tag == 1 {
                // Record type 1: new slot [1, id_len: u16, id_bytes]
                self.push_slot(&doc_id);
            } else if let Some(&doc_idx) = self.doc_to_idx.get(&doc_id) {
                // Record type 2: tombstone doc [2, id_len: u16, id_bytes]
                self.tombstone_in_memory(doc_idx);
            }
        }

        self.reload_mmap()
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

    fn mark_doc_deleted(&mut self, doc_id: &str) -> Result<bool> {
        let Some(&doc_idx) = self.doc_to_idx.get(doc_id) else {
            return Ok(false);
        };
        if !self.tombstone_in_memory(doc_idx) {
            return Ok(false);
        }

        let id_bytes = doc_id.as_bytes();
        self.meta_file.seek(SeekFrom::End(0))?;
        self.meta_file.write_all(&[2u8])?;
        self.meta_file
            .write_all(&(id_bytes.len() as u16).to_le_bytes())?;
        self.meta_file.write_all(id_bytes)?;
        self.meta_file.flush()?;
        Ok(true)
    }

    fn append_batch(&mut self, batch: &[(String, Vec<f32>)]) -> Result<()> {
        let Some(dim) = batch.iter().map(|(_, v)| v.len()).find(|&d| d > 0) else {
            return Ok(());
        };
        self.ensure_initialized(dim)?;
        if batch.iter().any(|(_, v)| !v.is_empty() && v.len() != dim) {
            let got = batch.iter().map(|(_, v)| v.len()).find(|&l| l != dim).unwrap_or(0);
            return Err(VectorError::DimensionMismatch { expected: dim, got });
        }

        self.vec_file.seek(SeekFrom::End(0))?;
        self.meta_file.seek(SeekFrom::End(0))?;

        let stride = self.stride;
        let mut vec_writer = BufWriter::new(&mut self.vec_file);
        let mut meta_writer = BufWriter::new(&mut self.meta_file);
        let mut row_buf = vec![0u8; stride];
        let mut new_ids: Vec<&str> = Vec::with_capacity(batch.len());

        for (doc_id, vector) in batch {
            if vector.is_empty() {
                continue;
            }
            let (scale, q_vec) = quantize_l2_normalized(vector);
            row_buf.fill(0);
            row_buf[0..4].copy_from_slice(&scale.to_le_bytes());
            let data_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(q_vec.as_ptr() as *const u8, dim) };
            row_buf[4..4 + dim].copy_from_slice(data_bytes);
            vec_writer.write_all(&row_buf)?;

            let id_bytes = doc_id.as_bytes();
            meta_writer.write_all(&[1u8])?;
            meta_writer.write_all(&(id_bytes.len() as u16).to_le_bytes())?;
            meta_writer.write_all(id_bytes)?;
            new_ids.push(doc_id.as_str());
        }

        vec_writer.flush()?;
        meta_writer.flush()?;
        drop(vec_writer);
        drop(meta_writer);

        // Replaying order matches the on-disk record order (later record replaces earlier one).
        for id in new_ids {
            self.push_slot(id);
        }

        self.reload_mmap()
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

        let mut live: Vec<String> = Vec::with_capacity(self.active_slots);
        {
            let mut tmp_vec = BufWriter::new(File::create(&tmp_vec_path)?);
            let mut tmp_meta = BufWriter::new(File::create(&tmp_meta_path)?);

            tmp_meta.write_all(META_MAGIC)?;
            tmp_meta.write_all(&(self.dim as u32).to_le_bytes())?;

            if let Some(ref mmap) = self.mmap {
                let raw = mmap.as_ref();
                let stride = self.stride;
                for (old_idx, slot) in self.slots.iter().enumerate() {
                    if slot.deleted {
                        continue;
                    }
                    let doc_id = &self.doc_ids[slot.doc_idx as usize];
                    let byte_offset = old_idx * stride;
                    tmp_vec.write_all(&raw[byte_offset..byte_offset + stride])?;

                    let id_bytes = doc_id.as_bytes();
                    tmp_meta.write_all(&[1u8])?;
                    tmp_meta.write_all(&(id_bytes.len() as u16).to_le_bytes())?;
                    tmp_meta.write_all(id_bytes)?;
                    live.push(doc_id.clone());
                }
            }

            tmp_vec.flush()?;
            tmp_meta.flush()?;
        }

        self.slots.clear();
        self.doc_ids.clear();
        self.doc_to_idx.clear();
        self.doc_slot.clear();
        self.active_slots = 0;
        for id in &live {
            self.push_slot(id);
        }

        self.mmap = None;
        let vec_path = dir_path.join(VEC_FILENAME);
        let meta_path = dir_path.join(META_FILENAME);
        std::fs::rename(&tmp_vec_path, &vec_path)?;
        std::fs::rename(&tmp_meta_path, &meta_path)?;

        self.vec_file = OpenOptions::new().read(true).write(true).open(&vec_path)?;
        self.meta_file = OpenOptions::new().read(true).write(true).open(&meta_path)?;
        self.reload_mmap()
    }
}

/// L2-normalizes `v` and quantizes it into symmetric `SQ8` (`scale: f32`, `Vec<i8>`).
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
