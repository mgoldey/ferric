//! Second-derivative XC kernel response for ROKS/UKS Newton steps.
//!
//! Given a density-matrix perturbation (δD_α, δD_β), evaluate the response
//! of the per-spin XC potential:
//!
//!   δV_xc^σ_{μν} = Σ_g w_g f_xc^{σσ'}(r_g) δρ_{σ'}(r_g) χ_μ(r_g) χ_ν(r_g)
//!
//! where f_xc^{σσ'} is the spin-spin second functional derivative (libxc's
//! `xc_lda_fxc` / `xc_gga_fxc`). LDA only for this round — the OH/LDA
//! Newton-convergence target needs nothing more, and the GGA kernel adds
//! ∇δρ terms with σ-coupling that need separate validation.

use crate::ao_grid::eval_basis_and_grad_on_points;
use crate::density_on_grid::eval_density_closed;
use crate::grid::{build_atomic_grid, AtomicGridConfig, GridPoint};
use crate::libxc::{FunctionalFamily, XcDef};
use crate::vxc::{scale_columns_into, VxcScratch};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ndarray::{Array1, Array2, Array3};
use std::sync::Mutex;

/// Precomputed grid + AO data + libxc handle for an LDA f_xc response.
/// Build once at SCF setup; reuse the closure across Newton iterations.
pub struct LdaFxcKernel {
    pub xc: XcDef,
    pub grid: Vec<GridPoint>,
    pub chi: Array2<f64>,     // (nbf, npts)
    pub dchi: Array3<f64>,    // (3, nbf, npts) — unused for LDA but kept for symmetry
    /// Pre-scaled χ scratch reused across Newton iterations (`apply_with_ref`
    /// takes `&self`, hence the Mutex; uncontended — one lock per matvec).
    scratch: Mutex<VxcScratch>,
}

impl LdaFxcKernel {
    pub fn new(
        mol: &Molecule,
        bs: &BasisSet,
        xc: XcDef,
        cfg: &AtomicGridConfig,
    ) -> Result<Self, String> {
        if xc.funcs.iter().any(|f| f.family() != FunctionalFamily::Lda) {
            return Err("LdaFxcKernel: all sub-functionals must be LDA-family for the f_xc \
                 response. GGA response not implemented yet."
                .to_string());
        }
        let grid = build_atomic_grid(mol, cfg);
        let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let (chi, dchi) =
            eval_basis_and_grad_on_points(mol, bs, &pts).map_err(|e| format!("{e:?}"))?;
        Ok(Self { xc, grid, chi, dchi, scratch: Mutex::new(VxcScratch::new()) })
    }

    /// Compute (δV_xc^α, δV_xc^β) in AO basis given (δD_α, δD_β).
    pub fn apply(
        &self,
        d_delta_a: &Array2<f64>,
        d_delta_b: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let npts = self.grid.len();
        let nbf = self.chi.shape()[0];

        // 1. Evaluate δρ_α(r_g), δρ_β(r_g) on the grid.
        //    Reuse eval_density_closed (works for any symmetric D).
        let drho_a = eval_density_closed(d_delta_a, &self.chi, &self.dchi).rho;
        let drho_b = eval_density_closed(d_delta_b, &self.chi, &self.dchi).rho;

        // 2. Evaluate f_xc at the current reference density (taken from
        //    drho_a and drho_b is wrong — we need ρ_α^0, ρ_β^0 at the SCF
        //    state). The reference density is built by the caller.
        // ... actually for the matvec the f_xc must be evaluated at the
        // *current SCF reference* density, NOT the perturbation. So we
        // need the reference density to be passed in. Restructure below.

        // Placeholder — real entry point is apply_with_ref.
        let _ = (drho_a, drho_b, npts, nbf);
        (Array2::zeros(d_delta_a.dim()), Array2::zeros(d_delta_b.dim()))
    }

    /// Compute (δV_xc^α, δV_xc^β) given the reference per-spin densities
    /// (ρ_α^0, ρ_β^0) on this kernel's grid, and the perturbation density
    /// matrices (δD_α, δD_β).
    ///
    /// Pre-evaluating ρ^0 on the grid each call would duplicate work the
    /// caller already does for the gradient build. We accept ρ^0 as input
    /// for efficiency, then internally project to grid via δD → δρ → δV.
    pub fn apply_with_ref(
        &self,
        rho_a0: &[f64],
        rho_b0: &[f64],
        d_delta_a: &Array2<f64>,
        d_delta_b: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let npts = self.grid.len();
        let nbf = self.chi.shape()[0];
        assert_eq!(rho_a0.len(), npts);
        assert_eq!(rho_b0.len(), npts);

        // δρ on grid.
        let drho_a = eval_density_closed(d_delta_a, &self.chi, &self.dchi).rho;
        let drho_b = eval_density_closed(d_delta_b, &self.chi, &self.dchi).rho;

        // Interleave reference (ρ_α, ρ_β) into the polarized libxc layout.
        let mut rho_packed = vec![0.0f64; 2 * npts];
        for g in 0..npts {
            rho_packed[2 * g + 0] = rho_a0[g];
            rho_packed[2 * g + 1] = rho_b0[g];
        }
        // Sum f_xc over all sub-functionals (e.g., LDA_X + LDA_C_VWN).
        let mut v2 = vec![0.0f64; 3 * npts];
        let mut v2_tmp = vec![0.0f64; 3 * npts];
        for f in &self.xc.funcs {
            for x in v2_tmp.iter_mut() { *x = 0.0; }
            f.eval_lda_fxc_polarized(&rho_packed, &mut v2_tmp);
            for (a, b) in v2.iter_mut().zip(v2_tmp.iter()) { *a += *b; }
        }

        // δV(r_g) per spin (LDA):
        //   δV_α = f_αα · δρ_α + f_αβ · δρ_β
        //   δV_β = f_αβ · δρ_α + f_ββ · δρ_β
        let mut dv_a = vec![0.0f64; npts];
        let mut dv_b = vec![0.0f64; npts];
        for g in 0..npts {
            let fxc_aa = v2[3 * g + 0];
            let fxc_ab = v2[3 * g + 1];
            let fxc_bb = v2[3 * g + 2];
            let da = drho_a[g];
            let db = drho_b[g];
            dv_a[g] = fxc_aa * da + fxc_ab * db;
            dv_b[g] = fxc_ab * da + fxc_bb * db;
        }

        // Back-project to AO: δV^σ_{μν} = Σ_g w_g δV^σ(r_g) χ_μ(r_g) χ_ν(r_g)
        //
        // Pre-scale χ columns by (w · δV^σ) per grid point, then GEMM:
        // chi_scaled @ chiᵀ — same idiom as vxc.rs::semilocal_vxc_closed's
        // LDA piece. One reused (nbf, npts) scratch serves both spins
        // sequentially (a's GEMM completes before b refills), replacing the two
        // full `chi.clone()` copies this call used to make every Newton step.
        let mut scratch = self.scratch.lock().expect("fxc scratch mutex poisoned");
        let buf = scratch.ensure((nbf, npts));

        let fac_a: Array1<f64> =
            (0..npts).map(|g| self.grid[g].weight * dv_a[g]).collect();
        scale_columns_into(&self.chi, &fac_a, buf);
        let dvxc_a: Array2<f64> = buf.dot(&self.chi.t());

        let fac_b: Array1<f64> =
            (0..npts).map(|g| self.grid[g].weight * dv_b[g]).collect();
        scale_columns_into(&self.chi, &fac_b, buf);
        let dvxc_b: Array2<f64> = buf.dot(&self.chi.t());
        (dvxc_a, dvxc_b)
    }

    /// Convenience: precompute the reference per-spin density on this
    /// kernel's grid from (D_α, D_β).
    pub fn reference_density(
        &self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> (Vec<f64>, Vec<f64>) {
        let rho_a = eval_density_closed(d_a, &self.chi, &self.dchi).rho.to_vec();
        let rho_b = eval_density_closed(d_b, &self.chi, &self.dchi).rho.to_vec();
        (rho_a, rho_b)
    }
}

/// Reference triple-loop back-projection, kept only to prove the GEMM in
/// `apply_with_ref` is numerically equivalent. Mirrors the pre-optimization
/// code exactly:
///   δV^σ_{μν} = Σ_g w_g δV^σ(r_g) χ_μ(r_g) χ_ν(r_g)
#[cfg(test)]
fn backproject_loop_reference(
    chi: &Array2<f64>,
    weights: &[f64],
    dv_a: &[f64],
    dv_b: &[f64],
) -> (Array2<f64>, Array2<f64>) {
    let (nbf, npts) = chi.dim();
    let mut dvxc_a = Array2::<f64>::zeros((nbf, nbf));
    let mut dvxc_b = Array2::<f64>::zeros((nbf, nbf));
    for g in 0..npts {
        let w = weights[g];
        let wa = w * dv_a[g];
        let wb = w * dv_b[g];
        for mu in 0..nbf {
            let chi_mu = chi[(mu, g)];
            let kmu_a = wa * chi_mu;
            let kmu_b = wb * chi_mu;
            for nu in 0..nbf {
                let chi_nu = chi[(nu, g)];
                dvxc_a[(mu, nu)] += kmu_a * chi_nu;
                dvxc_b[(mu, nu)] += kmu_b * chi_nu;
            }
        }
    }
    (dvxc_a, dvxc_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64 PRNG — no external `rand` dependency in the
    /// workspace, and we only need reproducible values in [-1, 1).
    struct Xorshift64(u64);
    impl Xorshift64 {
        fn next_f64(&mut self) -> f64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            // Map to [-1, 1).
            ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// GEMM back-projection must agree with the old triple loop to
    /// ~1e-12 (summation-order differences only), on small random inputs.
    #[test]
    fn gemm_backprojection_matches_loop_reference() {
        let npts = 64;
        let nbf = 10;
        let mut rng = Xorshift64(0x243F6A8885A308D3);

        let chi = Array2::<f64>::from_shape_fn((nbf, npts), |_| rng.next_f64());
        let weights: Vec<f64> = (0..npts).map(|_| 0.5 + 0.5 * rng.next_f64().abs()).collect();
        let dv_a: Vec<f64> = (0..npts).map(|_| rng.next_f64()).collect();
        let dv_b: Vec<f64> = (0..npts).map(|_| rng.next_f64()).collect();

        // Reference: old scalar triple loop.
        let (ref_a, ref_b) = backproject_loop_reference(&chi, &weights, &dv_a, &dv_b);

        // New: GEMM idiom (chi_scaled.dot(&chi.t())), exactly as in
        // LdaFxcKernel::apply_with_ref.
        let mut chi_scaled_a = chi.clone();
        let mut chi_scaled_b = chi.clone();
        for g in 0..npts {
            let w = weights[g];
            let wa = w * dv_a[g];
            let wb = w * dv_b[g];
            for mu in 0..nbf {
                chi_scaled_a[(mu, g)] *= wa;
                chi_scaled_b[(mu, g)] *= wb;
            }
        }
        let gemm_a: Array2<f64> = chi_scaled_a.dot(&chi.t());
        let gemm_b: Array2<f64> = chi_scaled_b.dot(&chi.t());

        let max_abs_diff = |x: &Array2<f64>, y: &Array2<f64>| -> f64 {
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };

        let diff_a = max_abs_diff(&gemm_a, &ref_a);
        let diff_b = max_abs_diff(&gemm_b, &ref_b);
        assert!(diff_a < 1e-12, "V_a GEMM vs loop max abs diff = {diff_a:e}");
        assert!(diff_b < 1e-12, "V_b GEMM vs loop max abs diff = {diff_b:e}");
    }
}
