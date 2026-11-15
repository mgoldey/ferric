//! Exchange matrices of many factored densities from one pass over the RI-K
//! tensor.
//!
//! The coupled-perturbed solves of the Hessian need `K[D1_x]` for every
//! coordinate `x` of a lockstep block at each iteration, and every such
//! density has the form `D = X C_oᵀ + C_o Xᵀ` for one fixed occupied block
//! `C_o` (`n × o`): the rotation response `occ · C_v U` and the occupied-block
//! density `−(occ/2) C_o S_oo` are both `X`. With `B_P` symmetric,
//!
//! ```text
//!   K[D] = Σ_P B_P D B_P = G + Gᵀ,   G = Σ_P (B_P X)(B_P C_o)ᵀ = Σ_P M_P L_Pᵀ
//! ```
//!
//! so [`DfK::occupied_half_transform`] builds `L_P = B_P C_o` once per solve
//! (`naux · n · o` doubles) and [`DfK::build_factored_many`] needs, per density
//! and auxiliary function, one `n × n × o` DSYMM (`M_P = B_P X`) and, per
//! auxiliary chunk, one GEMM `G_kᵀ = L_c M_c^k` of inner length `c · o`:
//! `4 naux n² o` flops per density against the density path's `4 naux n³`
//! (and none of its per-density repacks of the tensor). Each auxiliary
//! function is unpacked once per pass for all densities.
//!
//! Every chunk's partial is added in chunk order
//! ([`crate::reduce::grouped_deterministic_fold_columns`]) and the chunk edges
//! are those of the other DF-K paths ([`df_chunk_width`]), so the matrices do
//! not depend on the thread count, and each density gets the same DSYMMs and
//! GEMMs, of the same shapes, whatever else is in its pass: a pass of many
//! densities equals passes of one density bit for bit (`tests` asserts both).
//! Against the density path the sums are regrouped, which costs rounding
//! only.
//!
//! Memory: `L` is charged to the global pool while it lives. A pass holds,
//! per density, its share of at most [`MAX_FOLD_PARTIALS`] chunk partials
//! (`n²` each), of their `M_c` panels (`c · o · n` each) and of the
//! accumulator (`n²`); [`factored_pass_width`] keeps that within the DF-K band
//! budget (`crate::reduce::resolve_band_bytes`). Neither reads the thread
//! count.

use super::{dsymm_, occ_half_panel, DfK, OccChunk};
use crate::reduce::MAX_FOLD_PARTIALS;
use ferric_core::memory::pool::Reservation;
use ferric_core::FerricError;
use ferric_integrals::three_index_source::df_chunk_width;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, ArrayView2, ArrayViewMut2};

/// `L_P = B_P C_o` of every auxiliary function of a [`DfK`], stored per
/// auxiliary chunk (in the order [`DfK::build_factored_many`] visits them) as
/// the `n × c·o` panel `[L_{P0} | L_{P0+1} | …]`.
pub struct OccHalfTransform {
    c_occ: Array2<f64>,
    panels: Vec<Array2<f64>>,
    _charge: Reservation,
}

impl OccHalfTransform {
    /// The occupied block `C_o` the panels were built from.
    pub fn c_occ(&self) -> &Array2<f64> {
        &self.c_occ
    }

    /// Bytes of the panels for `naux` auxiliary functions, `n` basis
    /// functions and `nocc` occupied orbitals.
    pub fn bytes(naux: usize, n: usize, nocc: usize) -> usize {
        naux.saturating_mul(n)
            .saturating_mul(nocc)
            .saturating_mul(std::mem::size_of::<f64>())
    }
}

/// Densities per pass of [`DfK::build_factored_many`] for `n` basis
/// functions, `nocc` occupied orbitals and chunk width `chunk` under
/// `band_bytes`: [`MAX_FOLD_PARTIALS`] chunk partials and their panels per
/// density plus its accumulator. At least one. A pure function of the sizes
/// and the budget, never of the thread count.
pub fn factored_pass_width(n: usize, nocc: usize, chunk: usize, band_bytes: usize) -> usize {
    let per_density = ((MAX_FOLD_PARTIALS + 1) * n * n + MAX_FOLD_PARTIALS * chunk * nocc * n)
        .max(1)
        * std::mem::size_of::<f64>();
    (band_bytes / per_density).max(1)
}

/// Chunk partials held at once for a pass of `m` densities.
fn fold_band(m: usize, n: usize, nocc: usize, chunk: usize, band_bytes: usize) -> usize {
    let per_partial = (m * n * n + chunk * nocc * m * n).max(1) * std::mem::size_of::<f64>();
    (band_bytes / per_partial).clamp(1, MAX_FOLD_PARTIALS)
}

impl DfK<'_> {
    /// `L_P = B_P C_o` for every auxiliary function, built once for
    /// [`Self::build_factored_many`] (charged to the global pool while it
    /// lives; a typed error when it does not fit).
    pub fn occupied_half_transform(
        &mut self,
        c_occ: ArrayView2<'_, f64>,
    ) -> Result<OccHalfTransform, FerricError> {
        let n = self.dressed.nao();
        if c_occ.nrows() != n {
            return Err(FerricError::General(format!(
                "DfK::occupied_half_transform: C_occ rows ({}) != nao ({n})",
                c_occ.nrows()
            )));
        }
        let c_occ = c_occ.as_standard_layout().to_owned();
        let charge = ferric_core::memory::pool::reserve_global(
            "RI-K CPHF occupied half-transformed tensor B_P C_occ",
            OccHalfTransform::bytes(self.dressed.band_naux(), n, c_occ.ncols()),
        )?;
        let chunk = df_chunk_width(n);
        let mut panels = Vec::new();
        if c_occ.ncols() == 0 {
            return Ok(OccHalfTransform {
                c_occ,
                panels,
                _charge: charge,
            });
        }
        self.for_each_chunk_source(chunk, |srcs| {
            use rayon::prelude::*;
            let got: Vec<Array2<f64>> = srcs
                .par_iter()
                .map(|src| occ_half_panel(n, &c_occ, src))
                .collect::<Result<_, _>>()?;
            panels.extend(got);
            Ok(())
        })?;
        Ok(OccHalfTransform {
            c_occ,
            panels,
            _charge: charge,
        })
    }

    /// `K[X_k C_oᵀ + C_o X_kᵀ]` for every `X_k` of `xs` (`n × o` each), in
    /// order, with `C_o` and `L_P` from `half` (see the module doc). Passes of
    /// at most [`factored_pass_width`] densities.
    pub fn build_factored_many(
        &mut self,
        half: &OccHalfTransform,
        xs: &[&Array2<f64>],
    ) -> Result<Vec<Array2<f64>>, FerricError> {
        let n = self.dressed.nao();
        let nocc = half.c_occ.ncols();
        if let Some(x) = xs.iter().find(|x| x.dim() != (n, nocc)) {
            return Err(FerricError::General(format!(
                "DfK::build_factored_many: factor is {:?}, expected ({n}, {nocc})",
                x.dim()
            )));
        }
        let chunk = df_chunk_width(n);
        let band_bytes = crate::reduce::resolve_band_bytes(self.budget_bytes);
        let width = factored_pass_width(n, nocc, chunk, band_bytes);
        let mut out = Vec::with_capacity(xs.len());
        for pass in xs.chunks(width) {
            out.extend(self.factored_pass(half, pass, chunk, band_bytes)?);
        }
        Ok(out)
    }

    /// One pass of [`Self::build_factored_many`]: the accumulator holds
    /// `[G_1ᵀ | … | G_mᵀ]` (`n × m·n`), each chunk's partial added in chunk
    /// order.
    fn factored_pass(
        &mut self,
        half: &OccHalfTransform,
        xs: &[&Array2<f64>],
        chunk: usize,
        band_bytes: usize,
    ) -> Result<Vec<Array2<f64>>, FerricError> {
        let n = self.dressed.nao();
        let nocc = half.c_occ.ncols();
        let m = xs.len();
        if nocc == 0 || m == 0 {
            let mut zeros: Vec<Array2<f64>> = vec![Array2::zeros((n, n)); m];
            zeros.iter_mut().for_each(|k| self.reduce_k_across_ranks(k));
            return Ok(zeros);
        }
        // Xᵀ, row-major (o, n) = the Fortran (n, o) operand of DSYMM.
        let xts: Vec<Array2<f64>> = xs
            .iter()
            .map(|x| x.t().as_standard_layout().to_owned())
            .collect();
        let band = fold_band(m, n, nocc, chunk, band_bytes);
        let mut acc = vec![0.0f64; n * m * n];
        let mut scratch: Vec<Vec<f64>> = Vec::new();
        let mut first = 0usize;
        self.for_each_chunk_source(chunk, |srcs| {
            let panels = half.panels.get(first..first + srcs.len()).ok_or_else(|| {
                FerricError::General("DfK factored K: half-transform chunk mismatch".into())
            })?;
            let fail = std::sync::Mutex::new(None::<FerricError>);
            crate::reduce::grouped_deterministic_fold_columns(
                &mut acc,
                srcs.len(),
                band,
                &mut scratch,
                |g, y| {
                    if let Err(e) = factored_chunk(n, &xts, &panels[g], &srcs[g], y) {
                        *fail.lock().expect("factored K error slot") = Some(e);
                    }
                },
            );
            first += srcs.len();
            match fail.into_inner().expect("factored K error slot") {
                Some(e) => Err(e),
                None => Ok(()),
            }
        })?;
        if first != half.panels.len() {
            return Err(FerricError::General(
                "DfK factored K: the tensor's chunks changed since its half transform".into(),
            ));
        }
        let gt = ArrayView2::from_shape((n, m * n), &acc)
            .map_err(|e| FerricError::General(format!("factored K accumulator: {e}")))?;
        let mut out = Vec::with_capacity(m);
        for k in 0..m {
            let g = gt.slice(ndarray::s![.., k * n..(k + 1) * n]);
            let mut kmat = &g + &g.t();
            self.reduce_k_across_ranks(&mut kmat);
            out.push(kmat);
        }
        Ok(out)
    }

    /// Visit the dressed tensor as aux chunks of width `chunk` (the chunk edges
    /// of the other DF-K paths): the packed tiers as one resident band, the
    /// square tiers block by block (streamed when spilled).
    fn for_each_chunk_source(
        &mut self,
        chunk: usize,
        mut f: impl FnMut(&[OccChunk<'_>]) -> Result<(), FerricError>,
    ) -> Result<(), FerricError> {
        if let Some(rows) = self.dressed.dressed_packed() {
            let b = rows.nrows();
            let srcs: Vec<OccChunk<'_>> = (0..b.div_ceil(chunk))
                .map(|ci| OccChunk::Packed(rows, ci * chunk, ((ci + 1) * chunk).min(b)))
                .collect();
            return f(&srcs);
        }
        self.dressed.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            let srcs: Vec<OccChunk<'_>> = (0..b.div_ceil(chunk))
                .map(|ci| {
                    let (q0, q1) = (ci * chunk, ((ci + 1) * chunk).min(b));
                    OccChunk::Square(blk.data.slice(ndarray::s![q0..q1, .., ..]))
                })
                .collect();
            f(&srcs)
        })?;
        self.dressed.release_scratch();
        Ok(())
    }
}

/// One aux chunk's partial `[G_1ᵀ | … | G_mᵀ]` of [`DfK::build_factored_many`]
/// into `y` (`n × m·n`, overwritten): per auxiliary function `P` and density
/// `k` the DSYMM `M_P^k = B_P X_k`, stored transposed as rows `P·o..(P+1)·o`,
/// columns `k·n..(k+1)·n` of `M_c` (`c·o × m·n`), then per density the GEMM
/// `G_kᵀ = L_c M_c^k` into columns `k·n..(k+1)·n` of `y`.
#[allow(clippy::manual_c_str_literals)] // b"L\0" byte literals match dsymm_'s *const u8 args
fn factored_chunk(
    n: usize,
    xts: &[Array2<f64>],
    l_panel: &Array2<f64>,
    src: &OccChunk<'_>,
    y: &mut [f64],
) -> Result<(), FerricError> {
    let m = xts.len();
    let nocc = xts[0].nrows();
    let c = match src {
        OccChunk::Square(v) => v.shape()[0],
        OccChunk::Packed(_, q0, q1) => q1 - q0,
    };
    if l_panel.dim() != (n, c * nocc) {
        return Err(FerricError::General(
            "DfK factored K: half-transform panel shape mismatch".into(),
        ));
    }
    let ldc = (m * n) as i32;
    let (ni, no) = (n as i32, nocc as i32);
    let (one, zero) = (1.0f64, 0.0f64);
    let mut mc = Array2::<f64>::zeros((c * nocc, m * n));
    let mut unpacked = match src {
        OccChunk::Square(_) => Vec::new(),
        OccChunk::Packed(..) => vec![0.0f64; n * n],
    };
    for p in 0..c {
        let bp: &[f64] = match src {
            OccChunk::Square(v) => {
                v.slice_move(ndarray::s![p, .., ..])
                    .to_slice()
                    .ok_or_else(|| {
                        FerricError::General("DfK factored K: non-contiguous B row".into())
                    })?
            }
            OccChunk::Packed(rows, q0, _) => {
                rows.unpack_lower_into(q0 + p, &mut unpacked, n);
                &unpacked
            }
        };
        for (k, xt) in xts.iter().enumerate() {
            // Fortran C(n, o) = B_P (n×n, the row-major lower triangle is the
            // Fortran upper one DSYMM reads) · Xf (n, o) = Xᵀ row-major, so C is
            // the row-major (o, n) block (B_P X)ᵀ at row P·o, column k·n of M_c.
            let off = p * nocc * m * n + k * n;
            // SAFETY: bp is n×n, xt is (o, n) row-major (Fortran (n, o), ld n);
            // the (n, o) Fortran block written with ldc = m·n starting at
            // row p·o, column k·n of the (c·o, m·n) buffer stays inside it.
            unsafe {
                dsymm_(
                    b"L\0".as_ptr(),
                    b"U\0".as_ptr(),
                    &ni,
                    &no,
                    &one,
                    bp.as_ptr(),
                    &ni,
                    xt.as_ptr(),
                    &ni,
                    &zero,
                    mc.as_mut_ptr().add(off),
                    &ldc,
                );
            }
        }
    }
    let mut out = ArrayViewMut2::from_shape((n, m * n), y)
        .map_err(|e| FerricError::General(format!("factored K partial: {e}")))?;
    // One GEMM per density, of the same shape whatever the pass holds: a GEMM
    // stacked over the densities is not bitwise the one-density GEMM (OpenBLAS
    // picks its kernels by the output width).
    for k in 0..m {
        let cols = ndarray::s![.., k * n..(k + 1) * n];
        general_mat_mul(1.0, l_panel, &mc.slice(cols), 0.0, &mut out.slice_mut(cols));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Bars. The factored product and the density path compute the same
    //! exact `K = Σ_P B_P (X C_oᵀ + C_o Xᵀ) B_P` with the sums regrouped: the
    //! longest sum either path forms is `L = 2n + naux·o + 2` terms (the
    //! density path's two contractions of length `n` and its `naux` chunk
    //! fold; the factored path's DSYMM of length `n`, GEMM of length `c·o` and
    //! chunk fold, plus forming `D`), each over terms bounded elementwise by
    //! `A = Σ_P |B_P| (|X||C_o|ᵀ + |C_o||X|ᵀ) |B_P|`, so both are within
    //! `γ_L A` of the exact matrix and of each other within `2 γ_L A`
    //! (`γ_L = L u / (1 − L u)`, `u = ε/2`).
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;

    struct Case {
        dfk: DfK<'static>,
        n: usize,
        naux: usize,
        c_occ: Array2<f64>,
        xs: Vec<Array2<f64>>,
    }

    /// Distorted water / cc-pVDZ with def2-universal-jkfit (packed f64
    /// tensor, the production tier), a non-orthonormal `C_o` and three
    /// factors.
    fn case() -> Case {
        let xyz = "3\nH2O\nO 0.02 -0.01 0.1173\nH 0.0 0.80 -0.45\nH 0.05 -0.74 -0.50\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let aux =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let dfk = DfK::new(Operator::coulomb(), &obs, &aux, 1 << 30).unwrap();
        assert!(dfk.is_packed(), "the anchor must reach the packed f64 tier");
        let (n, nocc) = (obs.nbasis(), 5);
        let probe = |s: f64| {
            Array2::from_shape_fn((n, nocc), |(i, j)| ((i * 31 + j * 17) as f64 * s).sin())
        };
        Case {
            dfk,
            n,
            naux: aux.nbasis(),
            c_occ: probe(0.37),
            xs: (1..=3).map(|k| probe(0.11 * k as f64)).collect(),
        }
    }

    fn refs(xs: &[Array2<f64>]) -> Vec<&Array2<f64>> {
        xs.iter().collect()
    }

    /// `A = Σ_P |B_P| E |B_P|`, `E = |X||C|ᵀ + |C||X|ᵀ` (module bar).
    fn abs_bound(dfk: &DfK<'_>, x: &Array2<f64>, c: &Array2<f64>, n: usize) -> Array2<f64> {
        let (xa, ca) = (x.mapv(f64::abs), c.mapv(f64::abs));
        let e = xa.dot(&ca.t()) + ca.dot(&xa.t());
        let rows = dfk.dressed_rows_for_test().expect("packed rows");
        let mut buf = vec![0.0; n * n];
        let mut a = Array2::<f64>::zeros((n, n));
        for p in 0..rows.nrows() {
            rows.unpack_lower_into(p, &mut buf, n);
            let b = Array2::from_shape_fn((n, n), |(i, j)| buf[i.max(j) * n + i.min(j)].abs());
            a += &b.dot(&e).dot(&b);
        }
        a
    }

    /// The factored K of each `X` equals the density path's `K[X C_oᵀ + C_o Xᵀ]`
    /// within `2 γ_L A` elementwise (module bar).
    #[test]
    fn factored_k_matches_the_density_path() {
        let mut c = case();
        let half = c.dfk.occupied_half_transform(c.c_occ.view()).unwrap();
        let got = c.dfk.build_factored_many(&half, &refs(&c.xs)).unwrap();
        let len = 2 * c.n + c.naux * c.c_occ.ncols() + 2;
        let u = f64::EPSILON / 2.0;
        let gamma = len as f64 * u / (1.0 - len as f64 * u);
        for (k, (x, g)) in c.xs.iter().zip(&got).enumerate() {
            let d = x.dot(&c.c_occ.t()) + c.c_occ.dot(&x.t());
            let mut want = Array2::zeros((c.n, c.n));
            crate::fock::KBuilder::build(&mut c.dfk, &d, &mut want).unwrap();
            let bar = abs_bound(&c.dfk, x, &c.c_occ, c.n) * (2.0 * gamma);
            let (mut worst, mut ratio) = (0.0f64, 0.0f64);
            for ((a, b), t) in g.iter().zip(&want).zip(&bar) {
                worst = worst.max((a - b).abs());
                ratio = ratio.max((a - b).abs() / t);
            }
            eprintln!("factor #{k}: max |K_fac - K_dens| {worst:.3e}, max diff/bar {ratio:.3e}");
            assert!(ratio <= 1.0, "factor #{k}: exceeds the bar by {ratio:.3e}x");
            assert!(
                g.iter().zip(g.t().iter()).all(|(a, b)| a == b),
                "K not symmetric"
            );
        }
    }

    /// A pass of three densities equals three one-density passes bit for bit,
    /// under 1, 2 and 12 worker threads (the same bits in every pool).
    #[test]
    fn factored_block_equals_one_density_passes_in_every_pool() {
        let mut first: Option<Vec<Array2<f64>>> = None;
        for threads in [1usize, 2, 12] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut c = case();
                let half = c.dfk.occupied_half_transform(c.c_occ.view()).unwrap();
                let block = c.dfk.build_factored_many(&half, &refs(&c.xs)).unwrap();
                for (k, x) in c.xs.iter().enumerate() {
                    let one = c.dfk.build_factored_many(&half, &[x]).unwrap();
                    assert_eq!(block[k], one[0], "{threads} threads, factor #{k}");
                }
                match &first {
                    None => first = Some(block),
                    Some(f) => assert_eq!(f, &block, "{threads} threads differ from 1"),
                }
            });
        }
    }

    /// The pass width reads the sizes and the budget only: the same under
    /// every pool, at least one, and the live set it admits stays within the
    /// band whenever more than one density fits.
    #[test]
    fn factored_pass_width_is_charged_and_ignores_the_pool() {
        for threads in [1usize, 2, 12] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for (n, nocc) in [(24usize, 5usize), (106, 17), (250, 17), (2000, 120)] {
                    let chunk = df_chunk_width(n);
                    for band in [1usize << 20, 64 << 20, 512 << 20] {
                        let w = factored_pass_width(n, nocc, chunk, band);
                        assert!(w >= 1);
                        let live = 8
                            * w
                            * ((MAX_FOLD_PARTIALS + 1) * n * n
                                + MAX_FOLD_PARTIALS * chunk * nocc * n);
                        assert!(w == 1 || live <= band, "n {n} nocc {nocc} band {band}");
                        let b = fold_band(w, n, nocc, chunk, band);
                        assert!((1..=MAX_FOLD_PARTIALS).contains(&b));
                    }
                }
                assert_eq!(factored_pass_width(106, 17, 38, 512 << 20), 34);
            });
        }
    }

    /// Zero occupied orbitals: every K is exactly zero.
    #[test]
    fn factored_k_with_no_occupied_orbitals_is_zero() {
        let mut c = case();
        let empty = Array2::<f64>::zeros((c.n, 0));
        let half = c.dfk.occupied_half_transform(empty.view()).unwrap();
        let x = Array2::<f64>::zeros((c.n, 0));
        let k = c.dfk.build_factored_many(&half, &[&x, &x]).unwrap();
        assert!(k.iter().all(|m| m.iter().all(|v| *v == 0.0)));
    }
}
