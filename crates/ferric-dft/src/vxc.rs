//! Semilocal V_xc and E_xc assembly on a Becke-Lebedev grid.
//!
//! For an LDA functional:
//!   V_xc_μν = Σ_g w_g · v_ρ(r_g) · χ_μ(r_g) · χ_ν(r_g)
//!   E_xc   = Σ_g w_g · ρ(r_g) · ε_xc(r_g)
//!
//! For a GGA functional, add a v_σ-coupled term:
//!   V_xc_μν += Σ_g w_g · 2 · v_σ(r_g) · Σ_a ∇ρ_a(r_g) ·
//!                [χ_μ(r_g) · ∂χ_ν/∂a(r_g) + χ_ν(r_g) · ∂χ_μ/∂a(r_g)]
//!
//! The implementation uses one GEMM per term — χ (or ∇χ) is pre-scaled
//! by a per-grid-point factor into a scratch buffer, then contracted.
//! A single (nbf, npts) scratch serves all terms (refilled per use), and
//! callers that build V_xc every SCF iteration can hold a `VxcScratch`
//! across iterations to skip the per-iteration allocation entirely.
//!
//! Hybrid GGA and range-separated GGA functionals use the same semilocal
//! eval path as plain GGA — the exact-exchange mixing is the SCF's job.

use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ndarray::{Array1, Array2, Array3, Axis, Zip};
use rayon::prelude::*;

/// Below this density, libxc-returned v_ρ / v_σ may diverge; skip grid points
/// to keep V_xc well-conditioned. Matches libxc's internal `dens_threshold` default.
const DENSITY_FLOOR: f64 = 1e-10;

use crate::density_on_grid::{DensityGrid, UksDensityGrid};
use crate::grid::GridPoint;
use crate::libxc::{FunctionalFamily, XcDef};

/// Below this many grid points, run point loops serially — rayon overhead
/// dwarfs the work on tiny grids (free-atom SAD solves). Pure function of
/// grid size only, matching the `ao_grid`/`vv10`/`density_on_grid` guards.
const PAR_MIN_PTS: usize = 512;

/// Fixed group count for the deterministic chunked `E_xc` reduction below —
/// mirrors `ferric_scf::reduce::TARGET_GROUPS`. The point range is split into
/// at most this many contiguous, equal-ish groups; group boundaries are a
/// pure function of `npts`, never of thread count.
const EXC_SUM_GROUPS: usize = 256;

/// Deterministic, thread-count-invariant Σ_g `term(g)` over `0..npts`.
///
/// Splits `0..npts` into a fixed number of contiguous groups (a pure function
/// of `npts`), sums each group serially (in ascending-index order) — in
/// parallel across groups — then folds the group partials in ascending group
/// order. The total addition order is always: within a group, ascending `g`;
/// across groups, ascending group index. That order never depends on the
/// worker count, so the result is bit-identical across thread counts, and
/// (because it's the same order the flat serial `(0..npts).map(term).sum()`
/// would use within each group, just re-associated at group boundaries)
/// numerically matches the serial sum to machine precision — see the
/// `deterministic_sum_matches_serial_fold` test.
fn deterministic_point_sum<F>(npts: usize, term: F) -> f64
where
    F: Fn(usize) -> f64 + Sync,
{
    if npts < PAR_MIN_PTS {
        return (0..npts).map(term).sum();
    }
    let group_size = npts.div_ceil(EXC_SUM_GROUPS).max(1);
    let n_groups = npts.div_ceil(group_size);
    let partials: Vec<f64> = (0..n_groups)
        .into_par_iter()
        .map(|grp| {
            let g0 = grp * group_size;
            let g1 = (g0 + group_size).min(npts);
            (g0..g1).map(&term).sum::<f64>()
        })
        .collect();
    partials.into_iter().sum()
}

/// Reusable (nbf, npts) scratch holding the pre-scaled χ GEMM operand.
///
/// One buffer serves the LDA piece and all three GGA axes — each use refills
/// it with a single fused pass (`scale_columns_into`), replacing the previous
/// per-call `chi.clone()` + strided in-place scaling. Hold this across SCF
/// iterations (e.g. inside `KsXc`) to also amortize the allocation.
#[derive(Debug)]
pub struct VxcScratch {
    buf: Array2<f64>,
}

impl Default for VxcScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl VxcScratch {
    pub fn new() -> Self {
        Self { buf: Array2::zeros((0, 0)) }
    }

    /// Buffer of exactly `dim`, reallocating only on shape change.
    pub(crate) fn ensure(&mut self, dim: (usize, usize)) -> &mut Array2<f64> {
        if self.buf.dim() != dim {
            self.buf = Array2::zeros(dim);
        }
        &mut self.buf
    }
}

/// `out[(μ, g)] = chi[(μ, g)] · factors[g]` in one fused row-major pass.
///
/// `chi` is taken as a view so both owned `Array2` (via `.view()`) and a
/// borrowed axis slice of a 3-D `∇χ` (the meta-GGA τ term) can feed it without
/// an intermediate `.to_owned()` clone.
pub(crate) fn scale_columns_into(
    chi: ndarray::ArrayView2<f64>,
    factors: &Array1<f64>,
    out: &mut Array2<f64>,
) {
    debug_assert_eq!(chi.ncols(), factors.len());
    debug_assert_eq!(out.dim(), chi.dim());
    Zip::from(out)
        .and(chi)
        .and_broadcast(factors)
        .for_each(|o, &c, &s| *o = c * s);
}

/// Closed-shell semilocal exchange-correlation energy and potential.
///
/// Returns (E_xc, V_xc). V_xc is symmetrized before return.
///
/// Convenience wrapper over [`semilocal_vxc_closed_scratch`] that allocates a
/// fresh scratch — fine for one-shot callers; SCF loops should hold a
/// [`VxcScratch`] and call the `_scratch` variant.
///
/// `tau` (the total kinetic-energy density τ = ½ Σ_i|∇φ_i|², from
/// [`crate::density_on_grid::eval_tau_closed`]) is required for meta-GGA
/// functionals and ignored for LDA/GGA. Pass `None` for a pure LDA/GGA call.
pub fn semilocal_vxc_closed(
    grid: &[GridPoint],
    chi: &Array2<f64>,         // (nbf, npts)
    dchi: &Array3<f64>,        // (3, nbf, npts)
    dens: &DensityGrid,
    tau: Option<&Array1<f64>>,
    xc: &XcDef,
) -> (f64, Array2<f64>) {
    semilocal_vxc_closed_scratch(grid, chi, dchi, dens, tau, xc, &mut VxcScratch::new())
}

/// Closed-shell semilocal V_xc with caller-owned scratch (see [`VxcScratch`]).
///
/// For meta-GGA functionals `tau` must be `Some(&τ)` (total kinetic-energy
/// density on the grid); it is ignored for LDA/GGA. A meta-GGA call with
/// `tau == None` panics (a programming error at the call site).
pub fn semilocal_vxc_closed_scratch(
    grid: &[GridPoint],
    chi: &Array2<f64>,         // (nbf, npts)
    dchi: &Array3<f64>,        // (3, nbf, npts)
    dens: &DensityGrid,
    tau: Option<&Array1<f64>>,
    xc: &XcDef,
    scratch: &mut VxcScratch,
) -> (f64, Array2<f64>) {
    let (nbf, npts) = chi.dim();
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));

    let has_mgga = xc.funcs.iter().any(|f| matches!(f.family(), FunctionalFamily::MetaGga));

    let w: Array1<f64> = grid.iter().map(|g| g.weight).collect();

    let mut exc_total    = Array1::<f64>::zeros(npts);
    let mut vrho_total   = Array1::<f64>::zeros(npts);
    let mut vsigma_total = Array1::<f64>::zeros(npts);
    let mut vtau_total   = Array1::<f64>::zeros(npts);

    let rho_slice   = dens.rho.as_slice().expect("rho is contiguous");
    let sigma_slice = dens.sigma.as_slice().expect("sigma is contiguous");
    // τ input for meta-GGA. Required when a meta-GGA component is present; the
    // empty fallback is never read on the LDA/GGA path (only meta-GGA eval
    // touches `tau_slice`).
    let empty_tau: Array1<f64> = Array1::zeros(0);
    let tau_slice: &[f64] = if has_mgga {
        tau.expect("meta-GGA V_xc requires tau (kinetic-energy density)")
            .as_slice()
            .expect("tau is contiguous")
    } else {
        empty_tau.as_slice().unwrap()
    };

    // Accumulate ε_xc, v_ρ, v_σ across all component functionals. Left
    // serial: the O(npts) `+=` merges below are ~1 flop/point, dwarfed by the
    // libxc eval call that fills `exc`/`vrho`/`vsigma`; `xc.funcs` itself is
    // typically 1-3 iterations, so there's no compute here worth fanning out
    // (rayon collect/join overhead would exceed the work).
    for func in &xc.funcs {
        let mut exc  = vec![0.0_f64; npts];
        let mut vrho = vec![0.0_f64; npts];
        match func.family() {
            FunctionalFamily::Lda => {
                func.eval_lda_unpolarized(rho_slice, &mut exc, &mut vrho);
            }
            FunctionalFamily::Gga
            | FunctionalFamily::HybridGga
            | FunctionalFamily::RangeSepGga => {
                let mut vsigma = vec![0.0_f64; npts];
                func.eval_gga_unpolarized(
                    rho_slice, sigma_slice,
                    &mut exc, &mut vrho, &mut vsigma,
                );
                vsigma_total
                    .iter_mut()
                    .zip(&vsigma)
                    .for_each(|(t, &v)| *t += v);
            }
            FunctionalFamily::MetaGga => {
                let mut vsigma = vec![0.0_f64; npts];
                let mut vtau = vec![0.0_f64; npts];
                func.eval_mgga_unpolarized(
                    rho_slice, sigma_slice, tau_slice,
                    &mut exc, &mut vrho, &mut vsigma, &mut vtau,
                );
                vsigma_total
                    .iter_mut()
                    .zip(&vsigma)
                    .for_each(|(t, &v)| *t += v);
                vtau_total
                    .iter_mut()
                    .zip(&vtau)
                    .for_each(|(t, &v)| *t += v);
            }
        }
        exc_total.iter_mut().zip(&exc).for_each(|(t, &v)| *t += v);
        vrho_total.iter_mut().zip(&vrho).for_each(|(t, &v)| *t += v);
    }

    // E_xc = Σ_g w_g · ρ(r_g) · ε_xc(r_g). Deterministic grouped reduction —
    // see `deterministic_point_sum`: bit-identical across thread counts.
    let e_xc: f64 = deterministic_point_sum(npts, |g| w[g] * dens.rho[g] * exc_total[g]);

    let buf = scratch.ensure((nbf, npts));

    // ──────────────────────────────────────────────────────────────────────
    // LDA piece: V_lda_μν = Σ_g (w_g v_ρ_g) · χ_μg · χ_νg
    //
    // Pre-scale χ by (w · v_ρ) per grid point, then GEMM: buf @ chiᵀ.
    // ──────────────────────────────────────────────────────────────────────
    let s: Array1<f64> = Zip::from(&w)
        .and(&dens.rho)
        .and(&vrho_total)
        .map_collect(|&w, &r, &v| if r > DENSITY_FLOOR { w * v } else { 0.0 });
    scale_columns_into(chi.view(), &s, buf);
    // Digestion GEMM (nbf, npts)·(npts, nbf), outside any rayon region — this
    // whole function runs serially at the top level of one SCF/grid-response
    // iteration. Opt-in BLAS raise via FERRIC_BLAS_THREADS (default 1,
    // unchanged behavior); opt_in_blas_threads()'s rayon-worker self-guard
    // also protects any caller reached from inside a rayon pool (e.g.
    // free-atom SAD grid builds under run_serial_pool).
    let mut vxc: Array2<f64> =
        with_blas_threads(opt_in_blas_threads(), || buf.dot(&chi.t()));

    // ──────────────────────────────────────────────────────────────────────
    // GGA piece: V_gga_μν = Σ_g (2 w_g v_σ_g) ·
    //              Σ_axis ∇ρ_axis_g · [χ_μg (∂_axis χ_ν)_g + χ_νg (∂_axis χ_μ)_g]
    //
    // Define f_axis_g = 2 · w_g · v_σ_g · ∇ρ_axis(r_g). Then for each axis:
    //   M_axis = (f_axis ⊙ χ) · (∂_axis χ)ᵀ
    //   V_gga += M_axis + M_axisᵀ
    // ──────────────────────────────────────────────────────────────────────
    // The σ = |∇ρ|² coupling term is present for GGA and meta-GGA (both carry
    // a v_σ); only pure LDA skips it.
    let has_gga = xc.funcs.iter().any(|f| !matches!(f.family(), FunctionalFamily::Lda));
    if has_gga {
        for axis in 0..3 {
            let dchi_axis = dchi.index_axis(Axis(0), axis);
            let grad_axis = dens.grad.index_axis(Axis(0), axis);
            let f_ax: Array1<f64> = Zip::from(&w)
                .and(&dens.rho)
                .and(&vsigma_total)
                .and(&grad_axis)
                .map_collect(|&w, &r, &v, &gr| {
                    if r > DENSITY_FLOOR { 2.0 * w * v * gr } else { 0.0 }
                });
            scale_columns_into(chi.view(), &f_ax, buf);
            // Same opt-in-raise digestion GEMM as the LDA piece above.
            let m_axis: Array2<f64> =
                with_blas_threads(opt_in_blas_threads(), || buf.dot(&dchi_axis.t()));
            vxc = vxc + &m_axis + &m_axis.t();
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Meta-GGA τ piece:
    //   V^τ_μν = ∂E_xc/∂D_μν via τ = ½ Σ_axis Σ_κλ D_κλ ∂χ_κ ∂χ_λ
    //          = ½ Σ_g w_g v_τ(g) Σ_axis ∂_axis χ_μ(g) · ∂_axis χ_ν(g)
    //
    // Already symmetric in μν per axis (∂χ_μ ∂χ_ν = ∂χ_ν ∂χ_μ), so no extra
    // symmetrization is needed beyond the final ½(V+Vᵀ) (a no-op for it).
    // Build via: buf = (½ w v_τ) ⊙ ∂_axis χ, then buf · (∂_axis χ)ᵀ.
    // ──────────────────────────────────────────────────────────────────────
    if has_mgga {
        for axis in 0..3 {
            let dchi_axis = dchi.index_axis(Axis(0), axis);
            let f_ax: Array1<f64> = Zip::from(&w)
                .and(&dens.rho)
                .and(&vtau_total)
                .map_collect(|&w, &r, &vt| {
                    if r > DENSITY_FLOOR { 0.5 * w * vt } else { 0.0 }
                });
            scale_columns_into(dchi_axis, &f_ax, buf);
            let m_axis: Array2<f64> =
                with_blas_threads(opt_in_blas_threads(), || buf.dot(&dchi_axis.t()));
            vxc += &m_axis;
        }
    }

    // Symmetrize: V_xc ← ½(V_xc + V_xcᵀ)
    let vxc_sym = 0.5 * (&vxc + &vxc.t());

    (e_xc, vxc_sym)
}

/// Spin-polarized (UKS) semilocal exchange-correlation energy and potentials.
///
/// Returns `(E_xc, V_α, V_β)`. Each `V_σ` is symmetrized before return.
///
/// Convenience wrapper over [`semilocal_vxc_polarized_scratch`]; SCF loops
/// should hold a [`VxcScratch`] and call the `_scratch` variant.
pub fn semilocal_vxc_polarized(
    grid: &[GridPoint],
    chi: &Array2<f64>,
    dchi: &Array3<f64>,
    dens: &UksDensityGrid,
    tau: Option<(&Array1<f64>, &Array1<f64>)>,
    xc: &XcDef,
) -> (f64, Array2<f64>, Array2<f64>) {
    semilocal_vxc_polarized_scratch(grid, chi, dchi, dens, tau, xc, &mut VxcScratch::new())
}

/// Spin-polarized semilocal V_xc with caller-owned scratch.
///
/// libxc polarized interleaved layouts:
///   `rho_in[2g+0]   = ρ_α`,  `rho_in[2g+1]   = ρ_β`
///   `sigma_in[3g+0] = σ_αα`, `sigma_in[3g+1] = σ_αβ`, `sigma_in[3g+2] = σ_ββ`
///   `vrho[2g+0]     = v_α`,  `vrho[2g+1]     = v_β`
///   `vsigma[3g+0]   = v_σαα`,`vsigma[3g+1]   = v_σαβ`, `vsigma[3g+2]   = v_σββ`
///
/// V^α_μν includes a σ_αβ cross-term proportional to ∇ρ_β (and vice versa).
///
/// For meta-GGA functionals `tau` must be `Some((&τ_α, &τ_β))` — the per-spin
/// kinetic-energy densities (each τ_σ = ½ Σ_i∈σ |∇φ_i|², from
/// [`crate::density_on_grid::eval_tau_uks`]); it is ignored for LDA/GGA. A
/// meta-GGA call with `tau == None` panics.
pub fn semilocal_vxc_polarized_scratch(
    grid: &[GridPoint],
    chi: &Array2<f64>,
    dchi: &Array3<f64>,
    dens: &UksDensityGrid,
    tau: Option<(&Array1<f64>, &Array1<f64>)>,
    xc: &XcDef,
    scratch: &mut VxcScratch,
) -> (f64, Array2<f64>, Array2<f64>) {
    let (nbf, npts) = chi.dim();
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));

    let has_mgga = xc.funcs.iter().any(|f| matches!(f.family(), FunctionalFamily::MetaGga));

    let w: Array1<f64> = grid.iter().map(|g| g.weight).collect();

    // Build interleaved rho / sigma (and τ, for meta-GGA) input for libxc.
    // Left serial: a handful of loads/stores per point, one pass, no reduction.
    let mut rho_in = vec![0.0_f64; 2 * npts];
    let mut sigma_in = vec![0.0_f64; 3 * npts];
    for g in 0..npts {
        rho_in[2 * g + 0]     = dens.rho_a[g];
        rho_in[2 * g + 1]     = dens.rho_b[g];
        sigma_in[3 * g + 0]   = dens.sigma[(0, g)];
        sigma_in[3 * g + 1]   = dens.sigma[(1, g)];
        sigma_in[3 * g + 2]   = dens.sigma[(2, g)];
    }
    // Interleaved per-spin τ (`tau_in[2g+0]=τ_α`, `tau_in[2g+1]=τ_β`), only for
    // meta-GGA. Empty otherwise.
    let tau_in: Vec<f64> = if has_mgga {
        let (ta, tb) = tau.expect("meta-GGA UKS V_xc requires (tau_a, tau_b)");
        let mut v = vec![0.0_f64; 2 * npts];
        for g in 0..npts {
            v[2 * g + 0] = ta[g];
            v[2 * g + 1] = tb[g];
        }
        v
    } else {
        Vec::new()
    };

    let mut exc_total    = Array1::<f64>::zeros(npts);
    let mut vrho_a_total = Array1::<f64>::zeros(npts);
    let mut vrho_b_total = Array1::<f64>::zeros(npts);
    let mut vsigma_aa_total = Array1::<f64>::zeros(npts);
    let mut vsigma_ab_total = Array1::<f64>::zeros(npts);
    let mut vsigma_bb_total = Array1::<f64>::zeros(npts);
    let mut vtau_a_total = Array1::<f64>::zeros(npts);
    let mut vtau_b_total = Array1::<f64>::zeros(npts);

    // Left serial: same reasoning as the closed-shell accumulation above —
    // the O(npts) merges are trivial next to the libxc eval call itself.
    for func in &xc.funcs {
        let mut exc = vec![0.0_f64; npts];
        let mut vrho = vec![0.0_f64; 2 * npts];
        match func.family() {
            FunctionalFamily::Lda => {
                func.eval_lda_polarized(&rho_in, &mut exc, &mut vrho);
            }
            FunctionalFamily::Gga
            | FunctionalFamily::HybridGga
            | FunctionalFamily::RangeSepGga => {
                let mut vsigma = vec![0.0_f64; 3 * npts];
                func.eval_gga_polarized(
                    &rho_in, &sigma_in,
                    &mut exc, &mut vrho, &mut vsigma,
                );
                for g in 0..npts {
                    vsigma_aa_total[g] += vsigma[3 * g + 0];
                    vsigma_ab_total[g] += vsigma[3 * g + 1];
                    vsigma_bb_total[g] += vsigma[3 * g + 2];
                }
            }
            FunctionalFamily::MetaGga => {
                let mut vsigma = vec![0.0_f64; 3 * npts];
                let mut vtau = vec![0.0_f64; 2 * npts];
                func.eval_mgga_polarized(
                    &rho_in, &sigma_in, &tau_in,
                    &mut exc, &mut vrho, &mut vsigma, &mut vtau,
                );
                for g in 0..npts {
                    vsigma_aa_total[g] += vsigma[3 * g + 0];
                    vsigma_ab_total[g] += vsigma[3 * g + 1];
                    vsigma_bb_total[g] += vsigma[3 * g + 2];
                    vtau_a_total[g] += vtau[2 * g + 0];
                    vtau_b_total[g] += vtau[2 * g + 1];
                }
            }
        }
        for g in 0..npts {
            exc_total[g]    += exc[g];
            vrho_a_total[g] += vrho[2 * g + 0];
            vrho_b_total[g] += vrho[2 * g + 1];
        }
    }

    // E_xc = Σ_g w_g · (ρ_α + ρ_β) · ε_xc. Deterministic grouped reduction —
    // see `deterministic_point_sum`: bit-identical across thread counts.
    let e_xc: f64 = deterministic_point_sum(npts, |g| {
        w[g] * (dens.rho_a[g] + dens.rho_b[g]) * exc_total[g]
    });

    let has_gga = xc.funcs.iter().any(|f| !matches!(f.family(), FunctionalFamily::Lda));

    let buf = scratch.ensure((nbf, npts));

    // Build V^σ for σ ∈ {α, β}. One shared scratch buffer, refilled per term.
    let mut build = |vrho_sigma: &Array1<f64>,
                     vsigma_self: &Array1<f64>,
                     vsigma_cross: &Array1<f64>,
                     vtau_sigma: &Array1<f64>,
                     grad_self: &Array2<f64>,
                     grad_cross: &Array2<f64>,
                     rho_floor_ref: &Array1<f64>| -> Array2<f64> {
        // LDA piece: V^σ_μν = Σ_g (w v_ρσ) · χ_μg · χ_νg
        let s: Array1<f64> = Zip::from(&w)
            .and(rho_floor_ref)
            .and(vrho_sigma)
            .map_collect(|&w, &r, &v| if r > DENSITY_FLOOR { w * v } else { 0.0 });
        scale_columns_into(chi.view(), &s, buf);
        let mut v: Array2<f64> = buf.dot(&chi.t());

        if has_gga {
            // GGA piece for spin σ:
            //   V^σ_μν += Σ_g (2 w_g v_σσσ) · ∇ρ_σ · (χ_μ ∇χ_ν + χ_ν ∇χ_μ)
            //           + Σ_g (  w_g v_σαβ) · ∇ρ_other · (χ_μ ∇χ_ν + χ_ν ∇χ_μ)
            // (the αβ cross-coupling enters with coefficient 1, not 2, because
            // ∂σ_αβ/∂(∇ρ_α) = ∇ρ_β rather than 2∇ρ_β.)
            for axis in 0..3 {
                let dchi_axis = dchi.index_axis(Axis(0), axis);
                let gs = grad_self.index_axis(Axis(0), axis);
                let gc = grad_cross.index_axis(Axis(0), axis);
                let mut f_ax = Array1::<f64>::zeros(npts);
                Zip::from(&mut f_ax)
                    .and(&w)
                    .and(rho_floor_ref)
                    .and(vsigma_self)
                    .and(&gs)
                    .for_each(|f, &w, &r, &vs, &g_self| {
                        if r > DENSITY_FLOOR {
                            *f = 2.0 * w * vs * g_self;
                        }
                    });
                Zip::from(&mut f_ax)
                    .and(&w)
                    .and(rho_floor_ref)
                    .and(vsigma_cross)
                    .and(&gc)
                    .for_each(|f, &w, &r, &vc, &g_cross| {
                        if r > DENSITY_FLOOR {
                            *f += w * vc * g_cross;
                        }
                    });
                scale_columns_into(chi.view(), &f_ax, buf);
                let m_axis: Array2<f64> = buf.dot(&dchi_axis.t());
                v = v + &m_axis + &m_axis.t();
            }
        }

        if has_mgga {
            // Meta-GGA τ piece for spin σ (per-spin, no cross-coupling):
            //   V^τσ_μν = ½ Σ_g w_g v_τσ(g) Σ_axis ∂_axis χ_μ · ∂_axis χ_ν
            // ∂τ_σ/∂D^σ_μν = ½ Σ_axis ∂χ_μ ∂χ_ν (see the closed-shell derivation).
            // Already symmetric per axis; the final ½(V+Vᵀ) leaves it unchanged.
            for axis in 0..3 {
                let dchi_axis = dchi.index_axis(Axis(0), axis);
                let f_ax: Array1<f64> = Zip::from(&w)
                    .and(rho_floor_ref)
                    .and(vtau_sigma)
                    .map_collect(|&w, &r, &vt| {
                        if r > DENSITY_FLOOR { 0.5 * w * vt } else { 0.0 }
                    });
                scale_columns_into(dchi_axis, &f_ax, buf);
                let m_axis: Array2<f64> = buf.dot(&dchi_axis.t());
                v += &m_axis;
            }
        }

        0.5 * (&v + &v.t())
    };

    // Floor each spin block on its own ρ_σ (libxc treats v_ρσ as ill-defined
    // where ρ_σ → 0). For the αβ cross-term, gate on the smaller of the two.
    let v_a = build(
        &vrho_a_total, &vsigma_aa_total, &vsigma_ab_total, &vtau_a_total,
        &dens.grad_a, &dens.grad_b, &dens.rho_a,
    );
    let v_b = build(
        &vrho_b_total, &vsigma_bb_total, &vsigma_ab_total, &vtau_b_total,
        &dens.grad_b, &dens.grad_a, &dens.rho_b,
    );

    (e_xc, v_a, v_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(g: usize) -> f64 {
        // Irregular magnitudes so the sum is genuinely order-sensitive in
        // floating point.
        ((g as f64) + 0.7).sin() / ((g % 97) as f64 + 1.3)
    }

    #[test]
    fn deterministic_sum_bit_identical_across_thread_counts() {
        // Sized above PAR_MIN_PTS so the parallel grouped path is exercised.
        let npts = 4 * PAR_MIN_PTS + 37;
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| deterministic_point_sum(npts, term))
        };
        let r1 = run(1);
        let r4 = run(4);
        let r8 = run(8);
        assert_eq!(r1.to_bits(), r4.to_bits(), "1 vs 4 threads: {r1:e} vs {r4:e}");
        assert_eq!(r1.to_bits(), r8.to_bits(), "1 vs 8 threads: {r1:e} vs {r8:e}");

        // Grouped re-association may differ from the flat serial fold in the
        // last ulps, but must agree to near machine precision.
        let serial: f64 = (0..npts).map(term).sum();
        assert!(
            (r1 - serial).abs() <= 1e-12 * serial.abs().max(1.0),
            "grouped {r1:e} vs flat serial {serial:e}"
        );
    }

    #[test]
    fn deterministic_sum_below_threshold_is_flat_serial() {
        let npts = PAR_MIN_PTS - 1;
        let expected: f64 = (0..npts).map(term).sum();
        let got = deterministic_point_sum(npts, term);
        assert_eq!(got.to_bits(), expected.to_bits());
    }
}
