//! Opt-in single-precision storage of the RI-J raw tensor (`dfj-f32`).
//!
//! RI-J reads `B[P, k] = (P|μν)` twice per Fock build (`d_P = Σ_k B_Pk w_k`,
//! `J_k = Σ_P B_Pk c_P`). Holding `B` as f32 halves the resident tensor; the
//! consumer (`ferric_scf::df_j`) multiplies the f32 entries by f64 vectors with
//! f64 accumulation, so the ONLY single-precision rounding in a build is the one
//! rounding of each stored entry made here.
//!
//! ```text
//!   tier              resident             columns
//!   InCorePacked      naux·n(n+1)/2·8      every packed pair
//!   InCorePairScr.    naux·ncols·8         Schwarz-kept pairs
//!   InCoreF32         naux·cols·4          either layout (screen None / Some)
//! ```
//!
//! Rounding contract (IEEE 754 round to nearest): for a finite `x` with
//! `|x| ≤ f32::MAX`, `x̂ = x as f32` satisfies
//! `|x̂ − x| ≤ u32·|x| + η`, `u32 = 2⁻²⁴`, `η = 2⁻¹⁵⁰` ([`F32_UNDERFLOW_ABS`],
//! half the smallest f32 subnormal; it is 0 for `|x| ≥ f32::MIN_POSITIVE`). A
//! finite value beyond `f32::MAX` would round to infinity: the build refuses it
//! ([`F32Declined::OutOfRange`]) and the caller keeps the f64 tiers. A Coulomb
//! 3-index integral is bounded by `Q_P·Q_μν` (Schwarz), orders of magnitude
//! below that, so the refusal exists to be honest, not because it is expected.
//!
//! The f32 tier is in-core only and RI-J only: it cannot be streamed, unpacked
//! or dressed (DF-K needs the f64 tensor), and it is taken only under the
//! memory policies (`[scf] jk_storage = "auto"` or `"memory"`), the user's
//! `disk`/`direct` being an instruction to keep the tensor out of core.
//!
//! Memory: the tensor is hard-charged at 4 bytes per element under labels that
//! end in "f32". The transient f64 block it is computed through (at most
//! `PACKED_SCRATCH_CAP_BYTES`, at least one row) is uncharged, as the f64 packed
//! build's own block is, and the fit check leaves the same 64-row headroom for
//! it that `packed_in_core_fits` leaves.
use super::{
    charge_three_index, jk_storage, pack_lower_triangle, packed_pair_len, pair_screen, Backend,
    JkStorage, Operator, PairScreen, PreparedBasis, ThreeIndexSource, PACKED_MIN_SCRATCH_ROWS,
    PACKED_SCRATCH_CAP_BYTES,
};
use ferric_core::FerricError;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use std::sync::Arc;

/// `η = 2⁻¹⁵⁰`: the largest absolute error of rounding a value below
/// `f32::MIN_POSITIVE` to the nearest f32 (half the smallest subnormal).
pub const F32_UNDERFLOW_ABS: f64 = 7.006_492_321_624_085e-46;

/// How each f64 integral is rounded into the f32 tier.
// `TowardZero` is hidden because it is a test mutation, not because the enum is
// meant to grow: clippy's non-exhaustive heuristic misreads that.
#[allow(clippy::manual_non_exhaustive)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32Rounding {
    /// IEEE round to nearest, ties to even (`x as f32`): the contract above.
    Nearest,
    /// Toward zero. A MUTATION, never a production mode: the defect the
    /// error-bound tests of `ferric_scf::df_j` must catch (`TRUNCATE_DFJ_F32`).
    #[doc(hidden)]
    TowardZero,
}

impl F32Rounding {
    /// `x` rounded to f32 in this mode.
    #[inline]
    pub fn round(self, x: f64) -> f32 {
        let r = x as f32;
        match self {
            F32Rounding::Nearest => r,
            F32Rounding::TowardZero => {
                if r.is_finite() && f64::from(r).abs() > x.abs() {
                    // One ulp toward zero (sign-magnitude: the bits shrink for either sign).
                    f32::from_bits(r.to_bits() - 1)
                } else {
                    r
                }
            }
        }
    }
}

/// Why the f32 RI-J tensor was not built; the caller then keeps the f64 tiers.
#[derive(Clone, Debug, PartialEq)]
pub enum F32Declined {
    /// `[scf] jk_storage` is `disk` or `direct`.
    Policy(JkStorage),
    /// The f32 tensor plus its build block does not fit the budget or the pool.
    DoesNotFit {
        need: usize,
        budget: usize,
        pool_free: Option<usize>,
    },
    /// A finite integral beyond `f32::MAX` (global aux row, stored column).
    OutOfRange { row: usize, col: usize, value: f64 },
}

impl std::fmt::Display for F32Declined {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            F32Declined::Policy(p) => {
                write!(f, "jk_storage = \"{p}\" keeps the tensor out of core")
            }
            F32Declined::DoesNotFit {
                need,
                budget,
                pool_free,
            } => {
                write!(
                    f,
                    "the f32 tensor needs {:.3} GB with its build block, the budget is {:.3} GB",
                    *need as f64 / 1e9,
                    *budget as f64 / 1e9
                )?;
                if let Some(p) = pool_free {
                    write!(f, " and the pool has {:.3} GB free", *p as f64 / 1e9)?;
                }
                Ok(())
            }
            F32Declined::OutOfRange { row, col, value } => {
                write!(f, "B[{row}, {col}] = {value:e} is beyond the f32 range")
            }
        }
    }
}

/// Outcome of an f32 build or conversion that did not fail with an error.
pub type F32Outcome<T> = Result<T, F32Declined>;

/// Round `src` into `dst`; `Err(i)` names the first finite value that
/// overflowed.
fn round_into(src: &[f64], dst: &mut [f32], mode: F32Rounding) -> Result<(), usize> {
    for (i, (d, &s)) in dst.iter_mut().zip(src).enumerate() {
        let r = mode.round(s);
        if s.is_finite() && !r.is_finite() {
            return Err(i);
        }
        *d = r;
    }
    Ok(())
}

/// Round `rows` (f64, `cols` per row, first row = global aux row `p0`) into
/// the matching rows of `dst`, in parallel over rows.
fn round_rows(
    src: &[f64],
    dst: &mut [f32],
    cols: usize,
    p0: usize,
    mode: F32Rounding,
) -> F32Outcome<()> {
    let cols = cols.max(1);
    dst.par_chunks_mut(cols)
        .zip(src.par_chunks(cols))
        .enumerate()
        .try_for_each(|(i, (d, s))| {
            round_into(s, d, mode).map_err(|k| F32Declined::OutOfRange {
                row: p0 + i,
                col: k,
                value: s[k],
            })
        })
}

/// Bytes the shared pool has free, `None` when no pool is installed.
fn pool_free() -> Option<usize> {
    ferric_core::memory::pool::global_available_bytes()
}

/// Does `tensor` bytes plus `min(64, rows)` rows of `scratch_row` f64 bytes
/// (capped like the packed tier's headroom) fit the budget and the pool?
fn f32_fits(tensor: usize, rows: usize, scratch_row: usize, budget: usize) -> F32Outcome<()> {
    let head = scratch_row
        .saturating_mul(PACKED_MIN_SCRATCH_ROWS.min(rows.max(1)))
        .min(PACKED_SCRATCH_CAP_BYTES);
    let need = tensor.saturating_add(head);
    let free = pool_free();
    if need <= budget && need <= free.unwrap_or(usize::MAX) {
        Ok(())
    } else {
        Err(F32Declined::DoesNotFit {
            need,
            budget,
            pool_free: free,
        })
    }
}

/// Rows per f64 build block: the budget left after the f32 tensor, capped at
/// `PACKED_SCRATCH_CAP_BYTES`, at least one row.
fn block_rows(tensor: usize, scratch_row: usize, budget: usize) -> usize {
    let head = budget.saturating_sub(tensor).min(PACKED_SCRATCH_CAP_BYTES);
    (head / scratch_row.max(1)).max(1)
}

/// Aux row ranges covering `[p0, p1)`, each cut at an aux shell boundary once it
/// holds at least `step` rows (so no aux shell's integrals are computed twice).
fn shell_aligned_edges(
    offsets: &[usize],
    (p0, p1): (usize, usize),
    step: usize,
) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    if p0 >= p1 {
        return edges;
    }
    let mut q0 = p0;
    let cuts = offsets
        .iter()
        .copied()
        .filter(|&o| o > p0 && o < p1)
        .chain([p1]);
    for b in cuts {
        if b - q0 >= step || b == p1 {
            edges.push((q0, b));
            q0 = b;
        }
    }
    edges
}

/// The packed column list `(μ, ν ≤ μ)` in row-major order.
fn packed_cols(nao: usize) -> Vec<(u32, u32)> {
    let mut cols = Vec::with_capacity(packed_pair_len(nao));
    for mu in 0..nao as u32 {
        for nu in 0..=mu {
            cols.push((mu, nu));
        }
    }
    cols
}

/// One block of f64 rows `[q0, q1)` in the stored column layout: the packed
/// triangle (`screen` None) or the kept pairs.
fn f64_rows(
    (op, obs, dfbs): (Operator, &PreparedBasis, &PreparedBasis),
    (q0, q1): (usize, usize),
    screen: Option<&PairScreen>,
) -> Result<Array2<f64>, FerricError> {
    let nao = obs.nbasis();
    match screen {
        Some(s) => {
            let mut out = Array2::<f64>::zeros((q1 - q0, s.ncols()));
            pair_screen::fill_eri3_pairs(op, obs, dfbs, (q0, q1), s, &mut out)?;
            Ok(out)
        }
        None => {
            let blk = crate::threeindex::eri3_block_screened(op, obs, dfbs, q0, q1, None)?;
            let pair = packed_pair_len(nao);
            let mut out = Array2::<f64>::zeros((q1 - q0, pair));
            let dst = out
                .as_slice_mut()
                .ok_or_else(|| FerricError::General("packed rows not contiguous".into()))?;
            pack_lower_triangle(&blk.view(), nao, dst);
            Ok(out)
        }
    }
}

impl ThreeIndexSource {
    /// Bytes of an f32 tensor of `band` rows and `cols` stored columns.
    pub fn ri_j_f32_bytes(band: usize, cols: usize) -> usize {
        band.saturating_mul(cols).saturating_mul(4)
    }

    /// Build the band `[band_p0, band_p1)` directly as the f32 RI-J tier:
    /// the packed triangle (`screen` None) or `screen`'s kept pairs, computed in
    /// f64 one block at a time and rounded by `rounding`. The f64 tensor is never
    /// whole in memory, so the peak is the f32 tensor plus one f64 block.
    ///
    /// Every stored value is the rounding of the value the matching f64 tier
    /// stores (same engine calls: `eri3_block_screened(.., None)` for the packed
    /// layout, `fill_eri3_pairs` for the kept pairs), so it is bit-identical to
    /// rounding that tier's tensor.
    ///
    /// `Ok(Err(why))` when not taken: a `disk`/`direct` policy, a tensor that
    /// does not fit the budget or the pool, or a value beyond `f32::MAX`.
    pub fn build_ri_j_f32(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
        (band_p0, band_p1): (usize, usize),
        screen: Option<Arc<PairScreen>>,
        rounding: F32Rounding,
    ) -> Result<F32Outcome<Self>, FerricError> {
        let policy = jk_storage();
        if !matches!(policy, JkStorage::Auto | JkStorage::Memory) {
            return Ok(Err(F32Declined::Policy(policy)));
        }
        pair_screen::check_band(dfbs, (band_p0, band_p1))?;
        let (naux, nao) = (dfbs.nbasis(), obs.nbasis());
        let rows = band_p1 - band_p0;
        let cols = screen.as_ref().map_or(packed_pair_len(nao), |s| s.ncols());
        let bytes = Self::ri_j_f32_bytes(rows, cols);
        // The f64 block: an unpacked eri3 block (rows × n²) for the packed
        // layout, the compressed rows (rows × ncols) for the kept pairs.
        let scratch_row = match &screen {
            None => nao.saturating_mul(nao).saturating_mul(8),
            Some(_) => cols.saturating_mul(8),
        };
        if let Err(why) = f32_fits(bytes, rows, scratch_row, budget_bytes) {
            return Ok(Err(why));
        }
        let label = if screen.is_some() {
            "DF 3-index (P|mn) in-core packed, pair-screened, f32"
        } else {
            "DF 3-index (P|mn) in-core packed, f32"
        };
        let _charge = charge_three_index(label, bytes)?;
        let step = block_rows(bytes, scratch_row, budget_bytes).min(rows.max(1));
        if super::ooc_trace() {
            eprintln!(
                "[OOC build] jk_storage={policy} nao={nao} band=[{band_p0},{band_p1}) budget={:.2}GB \
                 -> InCoreF32 ({cols} columns{}, {:.2}GB, build block {step} rows)",
                budget_bytes as f64 / 1e9,
                if screen.is_some() { ", pair-screened" } else { "" },
                bytes as f64 / 1e9,
            );
        }
        let mut data = Array2::<f32>::zeros((rows, cols));
        let edges = shell_aligned_edges(dfbs.shell_offsets(), (band_p0, band_p1), step);
        for (q0, q1) in edges {
            let blk = f64_rows((op, obs, dfbs), (q0, q1), screen.as_deref())?;
            let src = blk
                .as_slice()
                .ok_or_else(|| FerricError::General("f64 rows not contiguous".into()))?;
            let dst = data
                .slice_mut(ndarray::s![q0 - band_p0..q1 - band_p0, ..])
                .into_slice()
                .ok_or_else(|| FerricError::General("f32 rows not contiguous".into()))?;
            if let Err(why) = round_rows(src, dst, cols, q0, rounding) {
                return Ok(Err(why));
            }
        }
        Ok(Ok(Self {
            naux,
            nao,
            block_naux: rows.max(1),
            band_p0,
            band_p1,
            backend: Backend::InCoreF32 { data, screen },
            _charge,
        }))
    }

    /// Convert this (f64) source into the f32 RI-J tier: the pair-screened tier
    /// keeps its columns, every other tier becomes the packed triangle (an
    /// unpacked tensor is packed, a spilled or recomputed one streamed once). An
    /// f32 source is returned as it is.
    ///
    /// Used where the raw tensor was built in f64 for another consumer first
    /// (the shared RI-JK path, where DF-K dresses it). The f32 charge is taken
    /// with `try_reserve` while the f64 tensor still holds its own, so a pool
    /// that cannot hold both at once declines. Returns the source (the f32 one,
    /// or this one unchanged) and, when it was not converted, why. On success
    /// the f64 tensor and its charge are dropped.
    pub fn into_ri_j_f32(
        mut self,
        budget_bytes: usize,
        rounding: F32Rounding,
    ) -> Result<(Self, Option<F32Declined>), FerricError> {
        if matches!(self.backend, Backend::InCoreF32 { .. }) {
            return Ok((self, None));
        }
        let policy = jk_storage();
        if !matches!(policy, JkStorage::Auto | JkStorage::Memory) {
            return Ok((self, Some(F32Declined::Policy(policy))));
        }
        let (rows, nao, p0) = (self.band_naux(), self.nao, self.band_p0);
        let screen = match &self.backend {
            Backend::InCorePairScreened { screen, .. } => Some(Arc::clone(screen)),
            _ => None,
        };
        let cols = screen.as_ref().map_or(packed_pair_len(nao), |s| s.ncols());
        let bytes = Self::ri_j_f32_bytes(rows, cols);
        let free = pool_free();
        let declined = F32Declined::DoesNotFit {
            need: bytes,
            budget: budget_bytes,
            pool_free: free,
        };
        if bytes > budget_bytes {
            return Ok((self, Some(declined)));
        }
        let label = if screen.is_some() {
            "DF 3-index (P|mn) in-core packed, pair-screened, f32"
        } else {
            "DF 3-index (P|mn) in-core packed, f32"
        };
        let Some(charge) = ferric_core::memory::pool::try_reserve_global(label, bytes) else {
            return Ok((self, Some(declined)));
        };
        enum Kind {
            InMemory,
            Stream,
        }
        let kind = match &self.backend {
            Backend::InCorePairScreened { .. } | Backend::InCore(_) => Kind::InMemory,
            _ => Kind::Stream,
        };
        let mut data = Array2::<f32>::zeros((rows, cols));
        let filled = {
            let dst = data
                .as_slice_mut()
                .ok_or_else(|| FerricError::General("f32 rows not contiguous".into()))?;
            if matches!(kind, Kind::Stream) {
                let mut why: Option<F32Declined> = None;
                let streamed = self.for_each_packed_block(1, |blk| {
                    let std = blk.data.as_standard_layout();
                    let src = std.as_slice().ok_or_else(|| {
                        FerricError::General("packed block not contiguous".into())
                    })?;
                    let l0 = blk.p0 - p0;
                    let d = &mut dst[l0 * cols..l0 * cols + src.len()];
                    if let Err(w) = round_rows(src, d, cols, blk.p0, rounding) {
                        why = Some(w);
                        return Err(FerricError::General("f32 range".into()));
                    }
                    Ok(())
                });
                match (why, streamed) {
                    (Some(w), _) => Err(w),
                    (None, Err(e)) => return Err(e),
                    (None, Ok(())) => Ok(()),
                }
            } else {
                match &self.backend {
                    Backend::InCorePairScreened { data: d64, .. } => {
                        let std = d64.as_standard_layout();
                        let src = std.as_slice().ok_or_else(|| {
                            FerricError::General("pair rows not contiguous".into())
                        })?;
                        round_rows(src, dst, cols, p0, rounding)
                    }
                    Backend::InCore(eri) => {
                        // Pack one aux row at a time (no second f64 tensor).
                        let mut out = Ok(());
                        let mut row = vec![0.0f64; cols];
                        for i in 0..rows {
                            let one = eri.slice(ndarray::s![i..i + 1, .., ..]);
                            pack_lower_triangle(&one, nao, &mut row);
                            let d = &mut dst[i * cols..(i + 1) * cols];
                            if let Err(why) = round_rows(&row, d, cols, p0 + i, rounding) {
                                out = Err(why);
                                break;
                            }
                        }
                        out
                    }
                    _ => unreachable!("kind is read from the backend above"),
                }
            }
        };
        if let Err(why) = filled {
            drop(charge);
            return Ok((self, Some(why)));
        }
        let me = Self {
            naux: self.naux,
            nao,
            block_naux: rows.max(1),
            band_p0: self.band_p0,
            band_p1: self.band_p1,
            backend: Backend::InCoreF32 { data, screen },
            _charge: charge,
        };
        drop(self);
        Ok((me, None))
    }

    /// The f32 RI-J tensor `(band_naux, cols)` and its pair screen (`None`: the
    /// packed triangle in `(μ, ν ≤ μ)` row-major order); `None` unless this
    /// source is the f32 tier. Rows are band-local.
    pub fn ri_j_f32(&self) -> Option<(ArrayView2<'_, f32>, Option<&PairScreen>)> {
        match &self.backend {
            Backend::InCoreF32 { data, screen } => Some((data.view(), screen.as_deref())),
            _ => None,
        }
    }

    /// Is this source the f32 RI-J tier?
    pub fn is_ri_j_f32(&self) -> bool {
        matches!(self.backend, Backend::InCoreF32 { .. })
    }

    /// Bytes this source's shared-pool charge holds (0 when no pool is
    /// installed: the charge is then inert).
    pub fn charged_bytes(&self) -> usize {
        self._charge.bytes()
    }

    /// Bytes of the tensor itself resident in this source (4 per element for
    /// the f32 tier, 8 for the f64 in-core tiers, 0 for spill and recompute).
    pub fn resident_tensor_bytes(&self) -> usize {
        self.tensor_resident_bytes()
    }

    /// Recompute this f32 source's rows in f64, in its own column layout, one
    /// block of global aux rows at a time (ascending, `f(p0, rows)`), without
    /// storing them: the f64 exit pass of `ferric_scf::df_j`. The bases must be
    /// the ones the tensor was built from. Errors unless this is the f32 tier.
    ///
    /// The values are those of the f64 tier with the same layout, bit for bit
    /// (same engine calls as [`Self::build_ri_j_f32`]). Blocks are sized to the
    /// budget left beside the f32 tensor (capped at 512 MiB, at least one row).
    pub fn for_each_ri_j_f64_block_recomputed(
        &self,
        (op, obs, dfbs): (Operator, &PreparedBasis, &PreparedBasis),
        budget_bytes: usize,
        mut f: impl FnMut(usize, ArrayView2<'_, f64>) -> Result<(), FerricError>,
    ) -> Result<(), FerricError> {
        let Backend::InCoreF32 { data, screen } = &self.backend else {
            return Err(FerricError::General(
                "the f64 exit pass needs the f32 RI-J tier".into(),
            ));
        };
        if obs.nbasis() != self.nao || dfbs.nbasis() != self.naux {
            return Err(FerricError::General(format!(
                "f64 exit pass: bases (nao {}, naux {}) do not match the tensor (nao {}, naux {})",
                obs.nbasis(),
                dfbs.nbasis(),
                self.nao,
                self.naux
            )));
        }
        let cols = data.ncols();
        let scratch_row = match screen {
            None => self.nao.saturating_mul(self.nao).saturating_mul(8),
            Some(_) => cols.saturating_mul(8),
        };
        let bytes = data.len().saturating_mul(4);
        let step = block_rows(bytes, scratch_row, budget_bytes).min(self.band_naux().max(1));
        let _charge = super::charge_three_index_soft(
            "DF 3-index f64 exit block",
            step.saturating_mul(scratch_row),
        );
        for (q0, q1) in shell_aligned_edges(dfbs.shell_offsets(), self.band(), step) {
            let blk = f64_rows((op, obs, dfbs), (q0, q1), screen.as_deref())?;
            f(q0, blk.view())?;
        }
        Ok(())
    }

    /// The stored rows as f64 `(band_naux, cols)` and their `(μ, ν)` columns:
    /// the f32 tier upcast exactly, the pair-screened tier as stored, every other
    /// tier as its packed triangle. Test hook.
    #[doc(hidden)]
    pub fn stored_rows_f64_for_test(
        &mut self,
    ) -> Result<(Array2<f64>, Vec<(u32, u32)>), FerricError> {
        let (rows, nao, p0) = (self.band_naux(), self.nao, self.band_p0);
        match &self.backend {
            Backend::InCoreF32 { data, screen } => {
                let cols = screen
                    .as_ref()
                    .map_or_else(|| packed_cols(nao), |s| s.cols().to_vec());
                return Ok((data.mapv(f64::from), cols));
            }
            Backend::InCorePairScreened { data, screen } => {
                return Ok((data.clone(), screen.cols().to_vec()));
            }
            Backend::InCore(eri) => {
                let pair = packed_pair_len(nao);
                let mut out = Array2::<f64>::zeros((rows, pair));
                let dst = out
                    .as_slice_mut()
                    .ok_or_else(|| FerricError::General("packed rows not contiguous".into()))?;
                pack_lower_triangle(&eri.view(), nao, dst);
                return Ok((out, packed_cols(nao)));
            }
            _ => {}
        }
        let mut out = Array2::<f64>::zeros((rows, packed_pair_len(nao)));
        self.for_each_packed_block(1, |blk| {
            let l0 = blk.p0 - p0;
            out.slice_mut(ndarray::s![l0..l0 + blk.data.nrows(), ..])
                .assign(&blk.data);
            Ok(())
        })?;
        Ok((out, packed_cols(nao)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Catches: a toward-zero mutant that rounds the wrong way for negative
    /// values, and a nearest mode that is not `as f32`.
    #[test]
    fn rounding_modes() {
        for x in [
            1.0 / 3.0,
            -1.0 / 3.0,
            2.0f64.powi(-140) / 3.0,
            1e30 / 7.0,
            0.0,
            -0.0,
        ] {
            let n = F32Rounding::Nearest.round(x);
            assert_eq!(n.to_bits(), (x as f32).to_bits());
            let t = F32Rounding::TowardZero.round(x);
            assert!(
                f64::from(t).abs() <= x.abs(),
                "{x:e}: toward zero went outward"
            );
            assert!(
                t.to_bits() == n.to_bits() || t.to_bits() + 1 == n.to_bits(),
                "{x:e}"
            );
            // Contract of nearest: |x̂ − x| ≤ u32|x| + η.
            let err = (f64::from(n) - x).abs();
            assert!(err <= 2f64.powi(-24) * x.abs() + F32_UNDERFLOW_ABS, "{x:e}");
        }
        assert_eq!(F32_UNDERFLOW_ABS, 2f64.powi(-150));
    }

    /// Catches: a range check that lets a finite value become infinity.
    #[test]
    fn overflow_is_refused_with_its_index() {
        let src = [1.0, f64::from(f32::MAX) * 2.0, 3.0];
        let mut dst = [0.0f32; 3];
        assert_eq!(round_into(&src, &mut dst, F32Rounding::Nearest), Err(1));
    }

    /// Catches: edges that miss a row, overlap, or cut inside an aux shell.
    #[test]
    fn shell_aligned_edges_cover_the_band_at_shell_cuts() {
        let offsets = [0usize, 3, 4, 9, 10, 15];
        for (band, step) in [
            ((0usize, 20usize), 4usize),
            ((2, 12), 1),
            ((0, 20), 100),
            ((5, 5), 3),
        ] {
            let e = shell_aligned_edges(&offsets, band, step);
            let mut at = band.0;
            for &(q0, q1) in &e {
                assert_eq!(q0, at);
                assert!(q1 > q0);
                assert!(
                    q1 == band.1 || offsets.contains(&q1),
                    "{band:?} cut at {q1}"
                );
                at = q1;
            }
            assert_eq!(at, band.1.max(band.0));
        }
    }
}
