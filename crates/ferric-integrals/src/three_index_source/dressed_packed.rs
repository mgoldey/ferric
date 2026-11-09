//! DF-K's V^{-1/2}-dressed band held in core as the PACKED lower μν triangle,
//! at f64 or f32 (`ferric_scf::df_k::RiKPrecision`).
//!
//! The dressed tensor `B[P,μν] = Σ_Q V^{-1/2}_{PQ} (Q|μν)` is exactly symmetric
//! in μν: the dressing computes the packed lower triangle only and expands it
//! (`super::dress_band_in_core`). The
//! square in-core tier therefore stores every off-diagonal twice. Every consumer
//! of the dressed tensor reads the lower triangle and nothing else, or can be
//! fed it unpacked one chunk at a time:
//!
//! * DF-K's occupied path calls DSYMM on each `B[P]` with Fortran `UPLO = 'U'`,
//!   which on a row-major matrix is the row-major LOWER triangle, exactly the
//!   packed row. Unpacked into the lower triangle of an `n × n` buffer, DSYMM
//!   reads the identical values, so K is bit-for-bit the square tier's.
//! * DF-K's density path (a guess density, ROHF spin densities, fractional
//!   occupations) unpacks each aux chunk to the full square, mirrored, which is
//!   the square tier's chunk exactly.
//!
//! ```text
//!   layout       resident (band rows)    K vs the square tier
//!   Square       band·n²·8               (the reference)
//!   PackedF64    band·n(n+1)/2·8         bit-identical (same values read)
//!   PackedF32    band·n(n+1)/2·4         one f32 rounding per stored element
//! ```
//!
//! # The dressing into a packed band
//!
//! The square in-core dressing (`super::dress_band_in_core`) already
//! accumulates the packed lower triangle and only then expands it to squares.
//! The f64 packed tier runs the same accumulation
//! (`super::dress_resident_into_packed`) and skips the expansion, so its
//! values are the square tier's lower triangle bit for bit. The f32 tier issues
//! the same GEMMs per `super::DRESS_COL_BLOCK` packed column tile (all band
//! rows, ascending `super::DRESS_K_BLOCK` Q blocks) into a per-worker f64
//! tile, then rounds the tile once (to nearest) into the f32 band; only the
//! output's leading dimension differs from the f64 route, which moves no bit.
//! So the ONLY single-precision operation is that one rounding per stored
//! element: `|B̂ − B| ≤ u32·|B| + η` (`u32 = 2⁻²⁴`, `η = 2⁻¹⁵⁰`,
//! [`super::F32_UNDERFLOW_ABS`]). A value beyond the f32 range declines the
//! f32 tier (never expected: `|B|` is bounded by `‖V^{-1/2}‖·max(Q|μν)`).
//!
//! # When this tier is taken
//!
//! Only with the raw tensor resident as the full square or the packed triangle
//! (the shared RI-JK raw of `build_for_jk` under the memory tiers): a spilled or
//! recomputed raw keeps the existing dressing paths (square in core or packed
//! on disk), as does a band that does not fit the budget and the pool. Memory:
//! the band is hard-charged at its stored size; the f32 tier's per-worker f64
//! column tile (and, for a dense raw, gathered Q panel) is soft-charged.
use super::{
    charge_three_index_soft, dress_resident_into_packed, dress_tile, packed_column_runs,
    packed_pair_len, Backend, F32Rounding, ThreeIndexSource, DRESS_COL_BLOCK, DRESS_K_BLOCK,
};
use ferric_core::FerricError;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// How a [`ThreeIndexSource`] holds DF-K's dressed band in core.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DressedLayout {
    /// The full `(band, n, n)` square, f64 (the device DF-K path reads it).
    Square,
    /// The packed lower triangle `(band, n(n+1)/2)`, f64: half the square,
    /// the same values.
    PackedF64,
    /// The packed lower triangle, f32: a quarter of the square, one rounding
    /// to nearest per stored element.
    PackedF32,
}

/// The resident packed dressed band, rows band-local, columns in `(μ, ν ≤ μ)`
/// row-major order.
#[derive(Clone, Copy, Debug)]
pub enum DressedRows<'a> {
    F64(ArrayView2<'a, f64>),
    F32(ArrayView2<'a, f32>),
}

impl DressedRows<'_> {
    /// Stored rows (the band height).
    pub fn nrows(&self) -> usize {
        match self {
            DressedRows::F64(a) => a.nrows(),
            DressedRows::F32(a) => a.nrows(),
        }
    }

    /// Write row `r`'s lower triangle into the LOWER triangle of the row-major
    /// `n × n` buffer `dst` (as f64); the upper triangle is not touched.
    pub fn unpack_lower_into(&self, r: usize, dst: &mut [f64], n: usize) {
        match self {
            DressedRows::F64(a) => unpack_lower(a.row(r).as_slice().expect("packed row"), dst, n),
            DressedRows::F32(a) => unpack_lower(a.row(r).as_slice().expect("packed row"), dst, n),
        }
    }

    /// Write row `r` into the row-major `n × n` buffer `dst` as the full
    /// symmetric matrix (both triangles, as f64).
    pub fn unpack_square_into(&self, r: usize, dst: &mut [f64], n: usize) {
        match self {
            DressedRows::F64(a) => unpack_square(a.row(r).as_slice().expect("packed row"), dst, n),
            DressedRows::F32(a) => unpack_square(a.row(r).as_slice().expect("packed row"), dst, n),
        }
    }
}

/// A stored element widened to f64 (exact for both types).
trait Widen: Copy {
    fn widen(self) -> f64;
}
impl Widen for f64 {
    #[inline]
    fn widen(self) -> f64 {
        self
    }
}
impl Widen for f32 {
    #[inline]
    fn widen(self) -> f64 {
        f64::from(self)
    }
}

fn unpack_lower<T: Widen>(src: &[T], dst: &mut [f64], n: usize) {
    let mut k = 0usize;
    for mu in 0..n {
        let row = &mut dst[mu * n..mu * n + mu + 1];
        for (d, s) in row.iter_mut().zip(&src[k..k + mu + 1]) {
            *d = s.widen();
        }
        k += mu + 1;
    }
}

fn unpack_square<T: Widen>(src: &[T], dst: &mut [f64], n: usize) {
    let mut k = 0usize;
    for mu in 0..n {
        for nu in 0..=mu {
            let v = src[k].widen();
            dst[mu * n + nu] = v;
            dst[nu * n + mu] = v;
            k += 1;
        }
    }
}

/// One f32 column tile `[c0, c0 + w)` of the dressed band: the GEMMs of
/// [`dress_resident_into_packed`] for that tile into the f64 scratch `tile`
/// `(band, w)`, ascending Q.
fn dress_tile_f64(
    (dense, packed): (Option<ArrayView2<'_, f64>>, Option<ArrayView2<'_, f64>>),
    m: &Array2<f64>,
    (band_p0, band): (usize, usize),
    (c0, w, nao): (usize, usize, usize),
    (panel_buf, tile): (&mut Vec<f64>, &mut [f64]),
) {
    let naux = m.ncols();
    let panel = match (&packed, &dense) {
        (Some(p), _) => p.slice(ndarray::s![.., c0..c0 + w]),
        (None, Some(d)) => {
            panel_buf.resize(naux * w, 0.0);
            let runs = packed_column_runs(c0, c0 + w, nao);
            for (q, dst) in panel_buf.chunks_exact_mut(w).enumerate() {
                let src = d.row(q);
                let src = src.as_slice().expect("dense raw rows are contiguous");
                for &(s0, d0, len) in &runs {
                    dst[d0..d0 + len].copy_from_slice(&src[s0..s0 + len]);
                }
            }
            ArrayView2::from_shape((naux, w), &panel_buf[..]).expect("panel sized naux x w")
        }
        (None, None) => unreachable!("checked by the caller"),
    };
    tile.fill(0.0);
    let mut t = ndarray::ArrayViewMut2::from_shape((band, w), tile).expect("tile sized band x w");
    let m_rows = m.slice(ndarray::s![band_p0..band_p0 + band, ..]);
    for qa in (0..naux).step_by(DRESS_K_BLOCK) {
        let qb = (qa + DRESS_K_BLOCK).min(naux);
        dress_tile(
            &m_rows.slice(ndarray::s![.., qa..qb]),
            &panel.slice(ndarray::s![qa..qb, ..]),
            &mut t,
        );
    }
}

/// Why an f32 block was not stored.
enum F32BlockError {
    OutOfRange { row: usize, col: usize, value: f64 },
}

/// What [`ThreeIndexSource::build_dressed_band_packed`] did.
pub enum PackedDressOutcome {
    /// The packed band, in the requested layout.
    Built(Box<ThreeIndexSource>),
    /// The f32 tier was requested but a dressed value lies beyond the f32
    /// range (global aux row, packed column, value); nothing was kept.
    F32OutOfRange { row: usize, col: usize, value: f64 },
    /// Not taken: the raw tensor is not resident in a layout this path reads,
    /// or the band does not fit the budget and the pool. The caller uses the
    /// square / spill dressing.
    NotTaken,
}

impl ThreeIndexSource {
    /// Bytes the packed dressed band of `band` rows holds at `layout`
    /// (`PackedF64` or `PackedF32`; the square size for `Square`).
    pub fn dressed_band_bytes(band: usize, nao: usize, layout: DressedLayout) -> usize {
        let pair = packed_pair_len(nao);
        match layout {
            DressedLayout::Square => band.saturating_mul(nao * nao).saturating_mul(8),
            DressedLayout::PackedF64 => band.saturating_mul(pair).saturating_mul(8),
            DressedLayout::PackedF32 => band.saturating_mul(pair).saturating_mul(4),
        }
    }

    /// Dress `raw` (the FULL aux range) into the band `[band_p0, band_p1)` held
    /// as the packed lower triangle at `layout` (`PackedF64` or `PackedF32`).
    /// `rounding` is the f32 rounding mode (a test mutation besides `Nearest`).
    /// See the module docs for when this declines ([`PackedDressOutcome::NotTaken`]).
    pub fn build_dressed_band_packed(
        raw: &ThreeIndexSource,
        m: &Array2<f64>,
        budget_bytes: usize,
        (band_p0, band_p1): (usize, usize),
        layout: DressedLayout,
        rounding: F32Rounding,
    ) -> Result<PackedDressOutcome, FerricError> {
        let naux = raw.naux();
        assert!(
            raw.band_p0 == 0 && raw.band_p1 == naux,
            "build_dressed_band_packed requires a FULL raw source (all Q)"
        );
        assert!(band_p0 <= band_p1 && band_p1 <= naux, "invalid aux band");
        assert!(
            layout != DressedLayout::Square,
            "the square layout is not packed"
        );
        let nao = raw.nao();
        let pair = packed_pair_len(nao);
        let band = band_p1 - band_p0;
        if raw.incore_flat().is_none() && raw.packed_flat().is_none() {
            return Ok(PackedDressOutcome::NotTaken);
        }
        let needed = Self::dressed_band_bytes(band, nao, layout);
        let n_tiles = pair.div_ceil(DRESS_COL_BLOCK).max(1);
        let workers = rayon::current_num_threads().max(1).min(n_tiles);
        // Per worker: the f64 column tile (f32 tier) and the gathered Q panel
        // of a dense raw (both tiers; the f64 route gathers it too).
        let tile_w = DRESS_COL_BLOCK.min(pair.max(1));
        let panel = if raw.incore_flat().is_some() {
            naux * tile_w * 8
        } else {
            0
        };
        let tile = if layout == DressedLayout::PackedF32 {
            band * tile_w * 8
        } else {
            0
        };
        let scratch = workers * (panel + tile);
        if needed.saturating_add(scratch) > budget_bytes {
            return Ok(PackedDressOutcome::NotTaken);
        }
        let label = match layout {
            DressedLayout::PackedF32 => "DF 3-index dressed B[P,mn] packed f32",
            _ => "DF 3-index dressed B[P,mn] packed",
        };
        // Admit against what the pool has FREE (raw stays resident beside it).
        let Some(charge) = ferric_core::memory::pool::try_reserve_global(label, needed) else {
            return Ok(PackedDressOutcome::NotTaken);
        };
        let _scratch_charge = charge_three_index_soft("DF packed dressing scratch", scratch);
        if super::ooc_trace() {
            eprintln!(
                "[OOC dress] naux={naux} band=[{band_p0},{band_p1}) nao={nao} needed={:.2}GB \
                 budget={:.2}GB -> {layout:?}",
                needed as f64 / 1e9,
                budget_bytes as f64 / 1e9,
            );
        }
        let backend = match layout {
            DressedLayout::PackedF64 => {
                let mut out = Array2::<f64>::zeros((band, pair));
                let t_st = ferric_core::timing::Stage::start("dfk:dress-gemm");
                dress_resident_into_packed(raw, m, band_p0, out.view_mut())?;
                t_st.end();
                Backend::InCorePacked {
                    packed: out,
                    scratch: None,
                }
            }
            DressedLayout::PackedF32 => {
                let mut out = Array2::<f32>::zeros((band, pair));
                let (dense, packed) = (raw.incore_flat(), raw.packed_flat());
                // Column tiles of the f32 band: each task owns `w` columns of
                // every row (disjoint views), computes them in an f64 tile and
                // rounds them in place.
                let mut tiles = Vec::new();
                let (mut rest, mut c0) = (out.view_mut(), 0usize);
                while rest.ncols() > 0 {
                    let w = DRESS_COL_BLOCK.min(rest.ncols());
                    let (t, r) = rest.split_at(ndarray::Axis(1), w);
                    tiles.push((c0, t));
                    rest = r;
                    c0 += w;
                }
                let t_st = ferric_core::timing::Stage::start("dfk:dress-gemm");
                let rounded: Result<(), F32BlockError> = tiles
                    .into_par_iter()
                    .map_init(
                        || (Vec::<f64>::new(), Vec::<f64>::new()),
                        |(panel_buf, tile), (c0, mut dst)| {
                            let w = dst.ncols();
                            tile.resize(band * w, 0.0);
                            dress_tile_f64(
                                (dense, packed),
                                m,
                                (band_p0, band),
                                (c0, w, nao),
                                (panel_buf, &mut tile[..band * w]),
                            );
                            for (i, (d, &x)) in dst.iter_mut().zip(tile.iter()).enumerate() {
                                let r = rounding.round(x);
                                if x.is_finite() && !r.is_finite() {
                                    return Err(F32BlockError::OutOfRange {
                                        row: band_p0 + i / w,
                                        col: c0 + i % w,
                                        value: x,
                                    });
                                }
                                *d = r;
                            }
                            Ok(())
                        },
                    )
                    .collect();
                t_st.end();
                match rounded {
                    Ok(()) => {}
                    Err(F32BlockError::OutOfRange { row, col, value }) => {
                        return Ok(PackedDressOutcome::F32OutOfRange { row, col, value })
                    }
                }
                Backend::InCoreF32 {
                    data: out,
                    screen: None,
                }
            }
            DressedLayout::Square => unreachable!("asserted above"),
        };
        Ok(PackedDressOutcome::Built(Box::new(ThreeIndexSource {
            naux,
            nao,
            block_naux: band.max(1),
            band_p0,
            band_p1,
            backend,
            _charge: charge,
        })))
    }

    /// The packed dressed band (f64 or f32) when this source holds one in core;
    /// `None` for every other tier. Rows band-local.
    ///
    /// A packed f64 RAW tensor answers too (it is the same storage): only call
    /// this on a dressed source.
    pub fn dressed_packed(&self) -> Option<DressedRows<'_>> {
        match &self.backend {
            Backend::InCorePacked { packed, .. } => Some(DressedRows::F64(packed.view())),
            Backend::InCoreF32 { data, screen: None } => Some(DressedRows::F32(data.view())),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpacking_a_packed_row_reproduces_the_symmetric_square() {
        let n = 5;
        let pair = packed_pair_len(n);
        let src: Vec<f64> = (0..pair).map(|k| k as f64 + 0.5).collect();
        let mut sq = vec![0.0; n * n];
        unpack_square(&src, &mut sq, n);
        let mut lo = vec![-1.0; n * n];
        unpack_lower(&src, &mut lo, n);
        for mu in 0..n {
            for nu in 0..n {
                assert_eq!(sq[mu * n + nu], sq[nu * n + mu]);
                if nu <= mu {
                    assert_eq!(lo[mu * n + nu], sq[mu * n + nu]);
                    assert_eq!(sq[mu * n + nu], (mu * (mu + 1) / 2 + nu) as f64 + 0.5);
                } else {
                    assert_eq!(lo[mu * n + nu], -1.0, "upper triangle written");
                }
            }
        }
    }
}
