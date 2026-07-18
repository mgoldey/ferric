//! Memory-budgeted aux-blocked 3-index (P|μν) integral source.
//!
//! Serves RAW (un-dressed) 3-center integrals in aux-blocks under a fixed byte
//! budget. In-core when the full tensor fits the budget; disk-spill otherwise.
//! Consumers apply their own metric (V^{-1} for J, V^{-1/2} for K).

use crate::basis_bridge::PreparedBasis;
use crate::operator::Operator;
use ferric_core::FerricError;
use ndarray::{Array2, Array3, ArrayView3};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;

/// `FERRIC_OOC_TRACE` descriptor: 3-index in-core/spill decision trace (env-only
/// debug toggle). NOTE behavior change: previously `.is_ok()` (any value, incl.
/// `=0`, enabled it); now `=0`/`false`/`off` disable it.
static OOC_TRACE: ferric_core::config::ConfigVar<bool> = ferric_core::config::ConfigVar {
    env_name: "FERRIC_OOC_TRACE",
    default: false,
    parse: ferric_core::config::parse_toggle,
    validate: ferric_core::config::accept_any,
};
fn ooc_trace() -> bool {
    OOC_TRACE.toggle()
}

/// Largest number of aux rows whose (block_naux × nao × nao × 8) bytes fit the
/// budget; at least 1 (a single aux row must always be representable).
fn block_naux_for(budget_bytes: usize, nao: usize) -> usize {
    let row_bytes = nao.saturating_mul(nao).saturating_mul(8).max(1);
    (budget_bytes / row_bytes).max(1)
}

/// Spill-path block size honoring the double-buffered pipeline: the compute
/// thread holds one block (N+1) while the write thread holds another (N), so at
/// most TWO blocks are resident at once. Sizing each block to *half* the budget
/// keeps that pair inside the ceiling. The `scratch` read-back buffer allocated
/// for `DiskSpill` is one block of the same size, so a later streaming pass
/// (one live scratch block) also stays well within budget.
fn spill_block_naux_for(budget_bytes: usize, nao: usize) -> usize {
    block_naux_for(budget_bytes / 2, nao)
}

/// Evict the file's pages from the OS page cache.
///
/// CRITICAL for memory safety under a cgroup budget: written/read file pages are
/// charged to the cgroup's `memory.current` (the `file` component of
/// `memory.stat`) until the kernel reclaims them. When spilling a tensor far
/// larger than the budget (e.g. a 15 GB temp file under an 8 GB cap), that page
/// cache accumulates unbounded and OOM-kills the process even though our heap
/// (`anon`) stays within budget. We flush dirty pages to disk then advise the
/// kernel to drop the whole file from cache, keeping the cgroup footprint bound
/// to the heap working set. Best-effort: errors are ignored (cache eviction is
/// an optimization, not a correctness requirement on systems where it's a no-op).
fn drop_page_cache(file: &File) {
    // DONTNEED only drops CLEAN pages; flush dirty pages to disk first.
    let _ = file.sync_data();
    let fd = file.as_raw_fd();
    // offset 0, len 0 == "to end of file".
    unsafe {
        libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

/// One aux-block of raw (P|μν), rows [p0, p0+data.shape()[0]).
pub struct AuxBlock<'a> {
    pub p0: usize,
    pub data: ArrayView3<'a, f64>,
}

enum Backend {
    InCore(Array3<f64>),
    DiskSpill { file: File, scratch: Array3<f64> },
}

pub struct ThreeIndexSource {
    naux: usize,
    nao: usize,
    block_naux: usize,
    backend: Backend,
}

impl ThreeIndexSource {
    /// `budget_bytes` is the hard ceiling for the resident raw 3-index footprint.
    pub fn build(
        op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis, budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        let naux = dfbs.nbasis();
        let nao = obs.nbasis();
        let needed = naux.saturating_mul(nao).saturating_mul(nao).saturating_mul(8);
        if ooc_trace() {
            eprintln!(
                "[OOC build] naux={naux} nao={nao} needed={:.2}GB budget={:.2}GB -> {}",
                needed as f64 / 1e9, budget_bytes as f64 / 1e9,
                if needed <= budget_bytes { "InCore" } else { "Spill" },
            );
        }
        if needed <= budget_bytes {
            let eri = crate::threeindex::eri3_tensor(op, obs, dfbs)?;
            Ok(Self { naux, nao, block_naux: naux, backend: Backend::InCore(eri) })
        } else {
            // Double-buffered spill: a producer thread computes block N+1 (via the
            // rayon-parallel `eri3_block`) while this thread writes block N to
            // disk, so compute and I/O overlap instead of serializing. A rendezvous
            // `sync_channel(0)` bounds the pipeline to exactly TWO live blocks at
            // any instant (one being written, one being computed) — the producer's
            // `send` blocks until the writer takes the previous block, so it never
            // runs more than one block ahead. `spill_block_naux_for` sizes each
            // block to half the budget so that pair stays inside the ceiling
            // (budget-honest). The write thread does pure I/O — no rayon here.
            //
            // Block content is byte-identical to the old serial loop: `eri3_block`
            // is write-once per element (see threeindex.rs) and blocks are written
            // in the same p0-ascending order, so the on-disk file — and thus the
            // read-back path in `for_each_block` — is unchanged.
            let block_naux = spill_block_naux_for(budget_bytes, nao);
            let mut file = tempfile::tempfile()
                .map_err(|e| FerricError::General(format!("tempfile: {e}")))?;

            // Channel carries either a computed block or a producer-side error.
            let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Array3<f64>, FerricError>>(0);

            std::thread::scope(|s| -> Result<(), FerricError> {
                // Producer: compute blocks in ascending p0 order, hand each off.
                s.spawn(move || {
                    let mut p0 = 0;
                    while p0 < naux {
                        let p1 = (p0 + block_naux).min(naux);
                        let blk = crate::threeindex::eri3_block(op, obs, dfbs, p0, p1);
                        let is_err = blk.is_err();
                        // If the receiver hung up (writer hit an I/O error and
                        // returned early), stop producing.
                        if tx.send(blk).is_err() || is_err {
                            return;
                        }
                        p0 = p1;
                    }
                });

                // Consumer (this thread): write each block as it arrives. Pure I/O.
                for blk in rx.iter() {
                    let blk = blk?;
                    let bytes: &[u8] = bytemuck::cast_slice(blk.as_slice().unwrap());
                    file.write_all(bytes)
                        .map_err(|e| FerricError::General(format!("spill write: {e}")))?;
                    // Evict just-written pages so the cgroup-charged page cache does
                    // not accumulate the whole (>budget) file. See drop_page_cache.
                    drop_page_cache(&file);
                }
                Ok(())
            })?;

            file.flush().ok();
            drop_page_cache(&file);
            let scratch = Array3::<f64>::zeros((block_naux, nao, nao));
            Ok(Self { naux, nao, block_naux, backend: Backend::DiskSpill { file, scratch } })
        }
    }

    /// Build a DRESSED source: out[P,:,:] = Σ_Q m[P,Q] · raw[Q,:,:], honoring budget.
    /// `raw` is consumed (streamed) and `m` is (naux, naux).
    pub fn build_dressed(
        raw: &mut ThreeIndexSource, m: &Array2<f64>, budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        let naux = raw.naux();
        let nao = raw.nao();
        let needed = naux.saturating_mul(nao).saturating_mul(nao).saturating_mul(8);
        let block_naux = block_naux_for(budget_bytes, nao);
        let in_core = needed <= budget_bytes;
        if ooc_trace() {
            eprintln!(
                "[OOC dress] naux={naux} nao={nao} needed={:.2}GB budget={:.2}GB block_naux={block_naux} -> {}",
                needed as f64 / 1e9, budget_bytes as f64 / 1e9,
                if in_core { "InCore" } else { "Spill" },
            );
        }
        let mut out_incore: Option<Array3<f64>> =
            if in_core { Some(Array3::zeros((naux, nao, nao))) } else { None };
        let mut file: Option<File> =
            if in_core { None } else {
                Some(tempfile::tempfile().map_err(|e| FerricError::General(format!("tempfile: {e}")))?)
            };
        let mut p0 = 0;
        while p0 < naux {
            let p1 = (p0 + block_naux).min(naux);
            let b = p1 - p0;
            let mut accum = Array3::<f64>::zeros((b, nao, nao));
            raw.for_each_block(|rb| {
                let rb_b = rb.data.shape()[0];
                let raw_flat = rb.data.into_shape_with_order((rb_b, nao * nao))
                    .map_err(|e| FerricError::General(format!("raw reshape: {e}")))?;
                let msub = m.slice(ndarray::s![p0..p1, rb.p0..rb.p0 + rb_b]); // (b, rb_b)
                let contrib = msub.dot(&raw_flat); // (b, nao*nao)
                let mut acc_flat = accum.view_mut().into_shape_with_order((b, nao * nao)).unwrap();
                acc_flat += &contrib;
                Ok(())
            })?;
            if let Some(arr) = out_incore.as_mut() {
                arr.slice_mut(ndarray::s![p0..p1, .., ..]).assign(&accum);
            } else if let Some(f) = file.as_mut() {
                let bytes: &[u8] = bytemuck::cast_slice(accum.as_slice().unwrap());
                f.write_all(bytes).map_err(|e| FerricError::General(format!("dress write: {e}")))?;
                drop_page_cache(f);
            }
            p0 = p1;
        }
        let backend = match (out_incore, file) {
            (Some(arr), _) => Backend::InCore(arr),
            (None, Some(f)) => {
                f.sync_all().ok();
                Backend::DiskSpill { file: f, scratch: Array3::zeros((block_naux, nao, nao)) }
            }
            _ => unreachable!(),
        };
        Ok(Self { naux, nao, block_naux: if in_core { naux } else { block_naux }, backend })
    }

    pub fn naux(&self) -> usize { self.naux }
    pub fn nao(&self) -> usize { self.nao }
    pub fn n_blocks(&self) -> usize {
        self.naux.div_ceil(self.block_naux.max(1))
    }
    pub fn block_naux(&self) -> usize { self.block_naux }

    /// Primary iteration API. Calls `f` once per raw aux-block, in order.
    pub fn for_each_block(
        &mut self,
        mut f: impl FnMut(AuxBlock<'_>) -> Result<(), FerricError>,
    ) -> Result<(), FerricError> {
        match &mut self.backend {
            Backend::InCore(eri) => {
                let nb = self.naux.div_ceil(self.block_naux.max(1));
                for i in 0..nb {
                    let p0 = i * self.block_naux;
                    let p1 = (p0 + self.block_naux).min(self.naux);
                    let view = eri.slice(ndarray::s![p0..p1, .., ..]);
                    f(AuxBlock { p0, data: view })?;
                }
                Ok(())
            }
            Backend::DiskSpill { file, scratch } => {
                file.seek(SeekFrom::Start(0)).map_err(|e| FerricError::General(format!("seek: {e}")))?;
                let nb = self.naux.div_ceil(self.block_naux.max(1));
                for i in 0..nb {
                    let p0 = i * self.block_naux;
                    let p1 = (p0 + self.block_naux).min(self.naux);
                    let b = p1 - p0;
                    let elems = b * self.nao * self.nao;
                    let buf = scratch.as_slice_mut().unwrap();
                    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf[..elems]);
                    file.read_exact(bytes).map_err(|e| FerricError::General(format!("spill read: {e}")))?;
                    let view = scratch.slice(ndarray::s![0..b, .., ..]);
                    f(AuxBlock { p0, data: view })?;
                }
                // Reads also populate the cgroup-charged page cache; drop them so
                // a full streaming pass doesn't pull the entire file into cache.
                drop_page_cache(file);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Operator;
    use crate::basis_bridge::PreparedBasis;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    fn water() -> (Molecule,) { (Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap(),) }

    #[test]
    fn block_naux_respects_budget() {
        // nao=10 → one aux row is 10*10*8 = 800 bytes.
        // budget 4000 bytes → block_naux = 4000/800 = 5.
        assert_eq!(block_naux_for(4000, 10), 5);
        // budget smaller than one row → at least 1.
        assert_eq!(block_naux_for(500, 10), 1);
    }

    #[test]
    fn spill_block_sizing_counts_both_live_blocks() {
        // Double-buffered spill holds two blocks at once (one computing, one
        // writing): the pair must fit the budget. nao=10 → row = 800 bytes.
        // budget 4000 → 2 × (2 rows × 800) = 3200 ≤ 4000. block_naux_for
        // would have said 5 rows (4000 bytes), whose pair would bust the budget.
        assert_eq!(spill_block_naux_for(4000, 10), 2);
        // Degenerate floor: even a budget below one row yields 1 (a single aux
        // row must always be representable) — pre-existing behavior.
        assert_eq!(spill_block_naux_for(500, 10), 1);
    }

    #[test]
    fn spill_single_row_blocks_equal_dense_eri3() {
        // Degenerate pipeline: budget below one aux row forces block_naux = 1,
        // maximizing producer/consumer handoffs (one rendezvous per aux row).
        // Content must still be bit-identical to the dense build.
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let dense = crate::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let (naux, nao, _) = dense.dim();
        let mut src = ThreeIndexSource::build(op, &obs, &dfbs, 1).unwrap();
        assert_eq!(src.n_blocks(), naux, "budget=1 byte should force 1-row blocks");
        let mut reassembled = ndarray::Array3::<f64>::zeros((naux, nao, nao));
        src.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            reassembled.slice_mut(ndarray::s![blk.p0..blk.p0 + b, .., ..]).assign(&blk.data);
            Ok(())
        }).unwrap();
        let n_diff = reassembled.iter().zip(dense.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(n_diff, 0, "1-row spill blocks differ bitwise from dense eri3");
    }

    #[test]
    fn spill_blocks_equal_dense_eri3() {
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let dense = crate::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let (naux, nao, _) = dense.dim();
        // Tiny budget → force spill into several blocks.
        let tiny = nao * nao * 8 * 3; // ~3 aux rows per block
        let mut src = ThreeIndexSource::build(op, &obs, &dfbs, tiny).unwrap();
        assert!(src.n_blocks() > 1, "expected spill into >1 block, got {}", src.n_blocks());
        let mut reassembled = ndarray::Array3::<f64>::zeros((naux, nao, nao));
        src.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            reassembled.slice_mut(ndarray::s![blk.p0..blk.p0 + b, .., ..]).assign(&blk.data);
            Ok(())
        }).unwrap();
        let maxdiff = (&reassembled - &dense).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(maxdiff == 0.0, "spill blocks != dense eri3, maxdiff={maxdiff}");
    }

    #[test]
    fn in_core_block_equals_dense_eri3() {
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let dense = crate::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        // Huge budget → single in-core block, raw (un-dressed).
        let mut src = ThreeIndexSource::build(op, &obs, &dfbs, usize::MAX).unwrap();
        assert_eq!(src.n_blocks(), 1);
        let mut reassembled = ndarray::Array3::<f64>::zeros(dense.dim());
        src.for_each_block(|blk| {
            reassembled.slice_mut(ndarray::s![blk.p0..blk.p0 + blk.data.shape()[0], .., ..])
                .assign(&blk.data);
            Ok(())
        }).unwrap();
        let maxdiff = (&reassembled - &dense).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(maxdiff == 0.0, "in-core raw block != dense eri3, maxdiff={maxdiff}");
    }
}
