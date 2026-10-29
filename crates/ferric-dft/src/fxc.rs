//! Second-derivative XC kernel response for ROKS/UKS Newton steps.
//!
//! Given a density-matrix perturbation (δD_α, δD_β), evaluate the response
//! of the per-spin XC potential:
//!
//!   δV_xc^σ_{μν} = Σ_g w_g f_xc^{σσ'}(r_g) δρ_{σ'}(r_g) χ_μ(r_g) χ_ν(r_g)
//!
//! where f_xc^{σσ'} is the spin-spin second functional derivative (libxc's
//! `xc_lda_fxc` / `xc_gga_fxc`).
//!
//! Two kernels live here:
//!   - `LdaFxcKernel`: purely local response, δV_σ = f_ρρ · δρ.
//!   - `GgaFxcKernel`: adds the ∇ρ / σ = |∇ρ|² coupling terms (v2rhosigma,
//!     v2sigma2). This is the general (semilocal) GGA/hybrid/RSH-GGA kernel —
//!     the exact-exchange fraction of a hybrid/RSH is the SCF's job (folded in
//!     via `k_mix_sr` at the call site), so the fxc kernel only ever sees the
//!     DFT semilocal piece. `GgaFxcKernel` also transparently handles pure-LDA
//!     sub-functionals (their σ-derivatives vanish), so a mixed LDA+GGA
//!     definition still evaluates correctly.
//!   - `MggaFxcKernel`: the τ-only meta-GGA kernel (SCAN / r2SCAN / TPSS, no
//!     Laplacian, as the SCF evaluates them), adding v2rhotau / v2sigmatau /
//!     v2tau2. It differentiates the polarized `vxc.rs` V_xc as implemented,
//!     density floor included, and on τ-free functionals reduces to
//!     `GgaFxcKernel` exactly (`mgga_kernel_on_gga_functional_equals_gga_kernel`).
//!     VV10's nonlocal response is in none of these kernels.
//!
//! The GGA response is validated by a finite-difference cross-check against the
//! (already-validated) first-derivative GGA V_xc builder in `vxc.rs` — see the
//! `gga_fxc_matches_finite_difference_of_vxc` test.

use crate::ao_grid::eval_basis_and_grad_on_points;
use crate::density_on_grid::{eval_density_closed, eval_density_uks, eval_tau_uks};
use crate::grid::{build_atomic_grid, AtomicGridConfig, GridPoint};
use crate::libxc::{FunctionalFamily, XcDef};
use crate::vxc::{scale_columns_into, VxcScratch, DENSITY_FLOOR};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ndarray::{Array1, Array2, Array3, Axis};
use std::sync::Mutex;

/// Precomputed grid + AO data + libxc handle for an LDA f_xc response.
/// Build once at SCF setup; reuse the closure across Newton iterations.
#[derive(Debug)]
pub struct LdaFxcKernel {
    pub xc: XcDef,
    pub grid: Vec<GridPoint>,
    pub chi: Array2<f64>,  // (nbf, npts)
    pub dchi: Array3<f64>, // (3, nbf, npts) — unused for LDA but kept for symmetry
    /// Pre-scaled χ scratch reused across Newton iterations (`apply_with_ref`
    /// takes `&self`, hence the Mutex; uncontended — one lock per matvec).
    scratch: Mutex<VxcScratch>,
}

impl LdaFxcKernel {
    /// Build the LDA f_xc kernel from a molecular grid and functional definition.
    pub fn new(
        mol: &Molecule,
        bs: &BasisSet,
        xc: XcDef,
        cfg: &AtomicGridConfig,
    ) -> Result<Self, String> {
        if xc.funcs.iter().any(|f| f.family() != FunctionalFamily::Lda) {
            return Err(
                "LdaFxcKernel: all sub-functionals must be LDA-family for the f_xc \
                 response. GGA response not implemented yet."
                    .to_string(),
            );
        }
        let grid = build_atomic_grid(mol, cfg);
        let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let (chi, dchi) =
            eval_basis_and_grad_on_points(mol, bs, &pts).map_err(|e| format!("{e:?}"))?;
        Ok(Self {
            xc,
            grid,
            chi,
            dchi,
            scratch: Mutex::new(VxcScratch::new()),
        })
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
        (
            Array2::zeros(d_delta_a.dim()),
            Array2::zeros(d_delta_b.dim()),
        )
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
        for (i, f) in self.xc.funcs.iter().enumerate() {
            let w_i = self.xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
            for x in v2_tmp.iter_mut() {
                *x = 0.0;
            }
            f.eval_lda_fxc_polarized(&rho_packed, &mut v2_tmp);
            for (a, b) in v2.iter_mut().zip(v2_tmp.iter()) {
                *a += w_i * *b;
            }
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
        let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let buf = scratch.ensure((nbf, npts));

        let fac_a: Array1<f64> = (0..npts).map(|g| self.grid[g].weight * dv_a[g]).collect();
        scale_columns_into(self.chi.view(), &fac_a, buf);
        // Digestion GEMM (nbf, npts)·(npts, nbf), outside any rayon region —
        // apply_with_ref is called once per matvec from the serial ROHF
        // AH-Newton solver (rohf_newton.rs/rohf_ah.rs; neither uses rayon).
        // Opt-in BLAS raise via FERRIC_BLAS_THREADS (default 1, unchanged
        // behavior); mirrors vxc.rs's semilocal_vxc_closed_scratch idiom.
        let dvxc_a: Array2<f64> =
            with_blas_threads(opt_in_blas_threads(), || buf.dot(&self.chi.t()));

        let fac_b: Array1<f64> = (0..npts).map(|g| self.grid[g].weight * dv_b[g]).collect();
        scale_columns_into(self.chi.view(), &fac_b, buf);
        // Same opt-in-raise digestion GEMM as the alpha-spin piece above.
        let dvxc_b: Array2<f64> =
            with_blas_threads(opt_in_blas_threads(), || buf.dot(&self.chi.t()));
        (dvxc_a, dvxc_b)
    }

    /// Convenience: precompute the reference per-spin density on this
    /// kernel's grid from (D_α, D_β).
    pub fn reference_density(&self, d_a: &Array2<f64>, d_b: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
        let rho_a = eval_density_closed(d_a, &self.chi, &self.dchi).rho.to_vec();
        let rho_b = eval_density_closed(d_b, &self.chi, &self.dchi).rho.to_vec();
        (rho_a, rho_b)
    }
}

/// Precomputed grid + AO data + libxc handle for a GGA f_xc response.
///
/// Parallel to [`LdaFxcKernel`] but retains and uses `dchi` (the AO gradients),
/// which the LDA kernel leaves dead. Handles GGA, hybrid-GGA and RSH-GGA
/// semilocal kernels; pure-LDA sub-functionals are also accepted and evaluate
/// correctly (their σ-derivatives are identically zero, and `xc_gga_fxc`
/// returns them as such, so the σ-coupling terms vanish).
#[derive(Debug)]
pub struct GgaFxcKernel {
    pub xc: XcDef,
    pub grid: Vec<GridPoint>,
    pub chi: Array2<f64>,  // (nbf, npts)
    pub dchi: Array3<f64>, // (3, nbf, npts)
    /// Pre-scaled χ / ∇χ scratch reused across Newton iterations.
    scratch: Mutex<VxcScratch>,
}

impl GgaFxcKernel {
    /// Build the kernel. Accepts GGA-family functionals (GGA / hybrid-GGA /
    /// RSH-GGA) and, transparently, pure LDA. Meta-GGA (which needs the
    /// τ-dependent second derivative) is out of scope for the f_xc Newton
    /// kernel and is rejected here — the SCF Newton gate already skips meta-GGA
    /// (falls back to DIIS), so this is a defensive belt-and-suspenders check.
    pub fn new(
        mol: &Molecule,
        bs: &BasisSet,
        xc: XcDef,
        cfg: &AtomicGridConfig,
    ) -> Result<Self, String> {
        if xc
            .funcs
            .iter()
            .any(|f| f.family() == FunctionalFamily::MetaGga)
        {
            return Err(
                "GgaFxcKernel: meta-GGA f_xc response (τ second derivative) \
                 is not implemented — meta-GGA SCF must use the DIIS path"
                    .to_string(),
            );
        }
        let grid = build_atomic_grid(mol, cfg);
        let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let (chi, dchi) =
            eval_basis_and_grad_on_points(mol, bs, &pts).map_err(|e| format!("{e:?}"))?;
        Ok(Self {
            xc,
            grid,
            chi,
            dchi,
            scratch: Mutex::new(VxcScratch::new()),
        })
    }

    /// Precompute the reference per-spin density (ρ, ∇ρ, σ channels) on this
    /// kernel's grid from the converged (D_α, D_β). Held by the caller and
    /// passed to [`apply_with_ref`](Self::apply_with_ref) each matvec so the
    /// (expensive) reference evaluation happens once per Newton solve, not once
    /// per matrix-vector product.
    pub fn reference_density(
        &self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> crate::density_on_grid::UksDensityGrid {
        eval_density_uks(d_a, d_b, &self.chi, &self.dchi)
    }

    /// Compute (δV_xc^α, δV_xc^β) in AO basis given the reference per-spin
    /// density grid (from [`reference_density`](Self::reference_density)) and
    /// the perturbation density matrices (δD_α, δD_β).
    ///
    /// # Derivation (spin σ, other spin σ̄)
    ///
    /// The first-derivative GGA potential (see `vxc.rs`) is
    ///
    ///   V^σ_{μν} = ∫ [ v_ρσ χ_μχ_ν
    ///                + ( 2 v_{σσσ} ∇ρ_σ + v_{σαβ} ∇ρ_σ̄ ) · ∇(χ_μχ_ν) ] dr
    ///
    /// with ∇(χ_μχ_ν) = χ_μ∇χ_ν + χ_ν∇χ_μ. Linearising in (δρ_α, δρ_β) gives a
    /// scalar term u^σ (multiplying χ_μχ_ν) and a 3-vector term w^σ
    /// (multiplying ∇(χ_μχ_ν)):
    ///
    ///   u^σ = δv_ρσ
    ///   w^σ = 2 (δv_{σσσ}) ∇ρ_σ + 2 v_{σσσ} ∇δρ_σ
    ///       +   (δv_{σαβ}) ∇ρ_σ̄ +   v_{σαβ} ∇δρ_σ̄
    ///
    /// where δv_ρσ and δv_{σc} are the first-order changes of the libxc
    /// derivatives, built from the second-derivative arrays:
    ///
    ///   δv_ρσ  = Σ_σ' (∂v_ρσ/∂ρ_σ') δρ_σ' + Σ_c (∂v_ρσ/∂σ_c) δσ_c   [v2rho2, v2rhosigma]
    ///   δv_{σc}= Σ_σ' (∂v_σc/∂ρ_σ') δρ_σ' + Σ_c'(∂v_σc/∂σ_c') δσ_c'  [v2rhosigma, v2sigma2]
    ///
    /// and the σ-channel perturbations are
    ///
    ///   δσ_αα = 2 ∇ρ_α·∇δρ_α
    ///   δσ_ββ = 2 ∇ρ_β·∇δρ_β
    ///   δσ_αβ =   ∇ρ_α·∇δρ_β + ∇ρ_β·∇δρ_α
    ///
    /// The final AO matrix element is
    ///
    ///   δV^σ_{μν} = Σ_g w_g [ u^σ(r_g) χ_μχ_ν
    ///                        + w^σ(r_g) · (χ_μ∇χ_ν + χ_ν∇χ_μ) ]
    ///
    /// assembled with the same one-GEMM-per-term idiom as `vxc.rs`.
    pub fn apply_with_ref(
        &self,
        ref_dens: &crate::density_on_grid::UksDensityGrid,
        d_delta_a: &Array2<f64>,
        d_delta_b: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let npts = self.grid.len();
        let nbf = self.chi.shape()[0];
        assert_eq!(ref_dens.rho_a.len(), npts);
        assert_eq!(ref_dens.rho_b.len(), npts);

        // Perturbation density on grid: δρ_σ and ∇δρ_σ (per spin).
        let ddens = eval_density_uks(d_delta_a, d_delta_b, &self.chi, &self.dchi);

        // Interleave reference (ρ_α, ρ_β) and (σ_αα, σ_αβ, σ_ββ) for libxc.
        let mut rho_packed = vec![0.0f64; 2 * npts];
        let mut sigma_packed = vec![0.0f64; 3 * npts];
        for g in 0..npts {
            rho_packed[2 * g] = ref_dens.rho_a[g];
            rho_packed[2 * g + 1] = ref_dens.rho_b[g];
            sigma_packed[3 * g] = ref_dens.sigma[(0, g)];
            sigma_packed[3 * g + 1] = ref_dens.sigma[(1, g)];
            sigma_packed[3 * g + 2] = ref_dens.sigma[(2, g)];
        }

        // Sum the second-derivative arrays over all sub-functionals.
        let mut v2rho2 = vec![0.0f64; 3 * npts];
        let mut v2rhosigma = vec![0.0f64; 6 * npts];
        let mut v2sigma2 = vec![0.0f64; 6 * npts];
        let mut t_rr = vec![0.0f64; 3 * npts];
        let mut t_rs = vec![0.0f64; 6 * npts];
        let mut t_ss = vec![0.0f64; 6 * npts];
        for (i, f) in self.xc.funcs.iter().enumerate() {
            let w_i = self.xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
            for x in t_rr.iter_mut() {
                *x = 0.0;
            }
            for x in t_rs.iter_mut() {
                *x = 0.0;
            }
            for x in t_ss.iter_mut() {
                *x = 0.0;
            }
            match f.family() {
                FunctionalFamily::Lda => {
                    f.eval_lda_fxc_polarized(&rho_packed, &mut t_rr);
                }
                FunctionalFamily::Gga
                | FunctionalFamily::HybridGga
                | FunctionalFamily::RangeSepGga => {
                    f.eval_gga_fxc_polarized(
                        &rho_packed,
                        &sigma_packed,
                        &mut t_rr,
                        &mut t_rs,
                        &mut t_ss,
                    );
                }
                FunctionalFamily::MetaGga => unreachable!(
                    "GgaFxcKernel built with a meta-GGA functional (rejected in ::new)"
                ),
            }
            for (a, b) in v2rho2.iter_mut().zip(&t_rr) {
                *a += w_i * *b;
            }
            for (a, b) in v2rhosigma.iter_mut().zip(&t_rs) {
                *a += w_i * *b;
            }
            for (a, b) in v2sigma2.iter_mut().zip(&t_ss) {
                *a += w_i * *b;
            }
        }

        // Reference and perturbation gradients as (3, npts) views.
        let grad_ref_a = &ref_dens.grad_a;
        let grad_ref_b = &ref_dens.grad_b;
        let grad_del_a = &ddens.grad_a;
        let grad_del_b = &ddens.grad_b;

        // Per-point scalar u^σ and vector w^σ (3 components), both spins.
        let mut u_a = vec![0.0f64; npts];
        let mut u_b = vec![0.0f64; npts];
        // w stored as (3, npts) to match the axis-major GEMM back-projection.
        let mut w_a = Array2::<f64>::zeros((3, npts));
        let mut w_b = Array2::<f64>::zeros((3, npts));

        for g in 0..npts {
            // Reference / perturbation gradient vectors.
            let gra = [grad_ref_a[(0, g)], grad_ref_a[(1, g)], grad_ref_a[(2, g)]];
            let grb = [grad_ref_b[(0, g)], grad_ref_b[(1, g)], grad_ref_b[(2, g)]];
            let gda = [grad_del_a[(0, g)], grad_del_a[(1, g)], grad_del_a[(2, g)]];
            let gdb = [grad_del_b[(0, g)], grad_del_b[(1, g)], grad_del_b[(2, g)]];

            let dra = ddens.rho_a[g];
            let drb = ddens.rho_b[g];

            // σ-channel perturbations δσ_c.
            let dot = |x: &[f64; 3], y: &[f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
            let dsig_aa = 2.0 * dot(&gra, &gda);
            let dsig_bb = 2.0 * dot(&grb, &gdb);
            let dsig_ab = dot(&gra, &gdb) + dot(&grb, &gda);

            // Second-derivative components at this point.
            // v2rho2: (αα, αβ, ββ)
            let frr_aa = v2rho2[3 * g];
            let frr_ab = v2rho2[3 * g + 1];
            let frr_bb = v2rho2[3 * g + 2];
            // v2rhosigma: (α·αα, α·αβ, α·ββ, β·αα, β·αβ, β·ββ)
            let frs_a_aa = v2rhosigma[6 * g];
            let frs_a_ab = v2rhosigma[6 * g + 1];
            let frs_a_bb = v2rhosigma[6 * g + 2];
            let frs_b_aa = v2rhosigma[6 * g + 3];
            let frs_b_ab = v2rhosigma[6 * g + 4];
            let frs_b_bb = v2rhosigma[6 * g + 5];
            // v2sigma2: (aa·aa, aa·ab, aa·bb, ab·ab, ab·bb, bb·bb)
            let fss_aa_aa = v2sigma2[6 * g];
            let fss_aa_ab = v2sigma2[6 * g + 1];
            let fss_aa_bb = v2sigma2[6 * g + 2];
            let fss_ab_ab = v2sigma2[6 * g + 3];
            let fss_ab_bb = v2sigma2[6 * g + 4];
            let fss_bb_bb = v2sigma2[6 * g + 5];

            // δv_ρσ = Σ_σ' (∂v_ρσ/∂ρ_σ') δρ_σ' + Σ_c (∂v_ρσ/∂σ_c) δσ_c
            let dv_rho_a = frr_aa * dra
                + frr_ab * drb
                + frs_a_aa * dsig_aa
                + frs_a_ab * dsig_ab
                + frs_a_bb * dsig_bb;
            let dv_rho_b = frr_ab * dra
                + frr_bb * drb
                + frs_b_aa * dsig_aa
                + frs_b_ab * dsig_ab
                + frs_b_bb * dsig_bb;

            // δv_σc = Σ_σ' (∂v_σc/∂ρ_σ') δρ_σ' + Σ_c'(∂v_σc/∂σ_c') δσ_c'
            //   ∂v_σc/∂ρ_σ' is v2rhosigma (same array, ρ↔σ symmetric).
            let dv_sig_aa = frs_a_aa * dra
                + frs_b_aa * drb
                + fss_aa_aa * dsig_aa
                + fss_aa_ab * dsig_ab
                + fss_aa_bb * dsig_bb;
            let dv_sig_ab = frs_a_ab * dra
                + frs_b_ab * drb
                + fss_aa_ab * dsig_aa
                + fss_ab_ab * dsig_ab
                + fss_ab_bb * dsig_bb;
            let dv_sig_bb = frs_a_bb * dra
                + frs_b_bb * drb
                + fss_aa_bb * dsig_aa
                + fss_ab_bb * dsig_ab
                + fss_bb_bb * dsig_bb;

            // u^σ = δv_ρσ (multiplies χ_μχ_ν).
            u_a[g] = dv_rho_a;
            u_b[g] = dv_rho_b;

            // w^σ (multiplies ∇(χ_μχ_ν)) — the δv_σ · ∇ρ_ref pieces:
            //   w^α += 2 δv_σαα ∇ρ_α + δv_σαβ ∇ρ_β
            //   w^β += 2 δv_σββ ∇ρ_β + δv_σαβ ∇ρ_α
            // The remaining v_σ_ref · ∇δρ pieces are added in the second pass
            for ax in 0..3 {
                w_a[(ax, g)] = 2.0 * dv_sig_aa * gra[ax] + dv_sig_ab * grb[ax];
                w_b[(ax, g)] = 2.0 * dv_sig_bb * grb[ax] + dv_sig_ab * gra[ax];
            }
        }

        // Reference first-derivative v_σ channels (vsigma) at the reference
        // density, summed over sub-functionals — needed for the
        // 2 v_σσσ ∇δρ_σ and v_σαβ ∇δρ_σ̄ contributions to w^σ.
        let mut vsig_aa = vec![0.0f64; npts];
        let mut vsig_ab = vec![0.0f64; npts];
        let mut vsig_bb = vec![0.0f64; npts];
        {
            let mut exc = vec![0.0f64; npts];
            let mut vrho = vec![0.0f64; 2 * npts];
            let mut vsigma = vec![0.0f64; 3 * npts];
            for (i, f) in self.xc.funcs.iter().enumerate() {
                let w_i = self.xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
                match f.family() {
                    FunctionalFamily::Lda => { /* no σ contribution */ }
                    FunctionalFamily::Gga
                    | FunctionalFamily::HybridGga
                    | FunctionalFamily::RangeSepGga => {
                        for x in exc.iter_mut() {
                            *x = 0.0;
                        }
                        for x in vrho.iter_mut() {
                            *x = 0.0;
                        }
                        for x in vsigma.iter_mut() {
                            *x = 0.0;
                        }
                        f.eval_gga_polarized(
                            &rho_packed,
                            &sigma_packed,
                            &mut exc,
                            &mut vrho,
                            &mut vsigma,
                        );
                        for g in 0..npts {
                            vsig_aa[g] += w_i * vsigma[3 * g];
                            vsig_ab[g] += w_i * vsigma[3 * g + 1];
                            vsig_bb[g] += w_i * vsigma[3 * g + 2];
                        }
                    }
                    FunctionalFamily::MetaGga => unreachable!(
                        "GgaFxcKernel built with a meta-GGA functional (rejected in ::new)"
                    ),
                }
            }
        }
        for g in 0..npts {
            let vaa = vsig_aa[g];
            let vab = vsig_ab[g];
            let vbb = vsig_bb[g];
            for ax in 0..3 {
                // + 2 v_σαα ∇δρ_α + v_σαβ ∇δρ_β  (α spin)
                w_a[(ax, g)] += 2.0 * vaa * grad_del_a[(ax, g)] + vab * grad_del_b[(ax, g)];
                // + 2 v_σββ ∇δρ_β + v_σαβ ∇δρ_α  (β spin)
                w_b[(ax, g)] += 2.0 * vbb * grad_del_b[(ax, g)] + vab * grad_del_a[(ax, g)];
            }
        }

        // ── Back-project to AO basis, one GEMM per term (mirrors vxc.rs). ──
        let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let buf = scratch.ensure((nbf, npts));

        // Build V^σ = (scalar u term) + Σ_axis (w-vector axis term).
        let build = |u: &[f64], w: &Array2<f64>, buf: &mut Array2<f64>| -> Array2<f64> {
            // Scalar (LDA-like) piece: Σ_g (w_g u_g) χ_μg χ_νg.
            let fac_u: Array1<f64> = (0..npts).map(|g| self.grid[g].weight * u[g]).collect();
            scale_columns_into(self.chi.view(), &fac_u, buf);
            let mut v: Array2<f64> =
                with_blas_threads(opt_in_blas_threads(), || buf.dot(&self.chi.t()));

            // Vector piece: for each axis, M = (w_axis ⊙ χ) · (∂_axis χ)ᵀ,
            //   V += M + Mᵀ  (from the χ_μ∇χ_ν + χ_ν∇χ_μ symmetrization).
            for axis in 0..3 {
                let dchi_axis = self.dchi.index_axis(Axis(0), axis);
                let w_ax = w.index_axis(Axis(0), axis);
                let fac_w: Array1<f64> = (0..npts).map(|g| self.grid[g].weight * w_ax[g]).collect();
                scale_columns_into(self.chi.view(), &fac_w, buf);
                let m_axis: Array2<f64> =
                    with_blas_threads(opt_in_blas_threads(), || buf.dot(&dchi_axis.t()));
                v = v + &m_axis + &m_axis.t();
            }
            // Symmetrize (the reference V_xc builder does the same).
            0.5 * (&v + &v.t())
        };

        let dvxc_a = build(&u_a, &w_a, buf);
        let dvxc_b = build(&u_b, &w_b, buf);
        (dvxc_a, dvxc_b)
    }
}

// ── Meta-GGA (τ-dependent) f_xc kernel ─────────────────────────────────────

/// Positions of the seven polarized semilocal variables in a per-point
/// Hessian: (ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ, τ_α, τ_β).
const RA: usize = 0;
const RB: usize = 1;
const SAA: usize = 2;
const SAB: usize = 3;
const SBB: usize = 4;
const TA: usize = 5;
const TB: usize = 6;
/// Number of polarized semilocal variables (τ-only meta-GGA, no Laplacian).
const NVAR: usize = 7;

/// `∂²f/∂x_i∂x_j` at one grid point over the [`NVAR`] semilocal variables.
pub(crate) type PointHessian = [[f64; NVAR]; NVAR];

/// The σ-channel pairs of libxc's packed upper-triangular `v2sigma2`.
const SIGMA2_PAIRS: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

/// `h[i][j] += v` and its mirror (once on the diagonal).
fn add_sym(h: &mut PointHessian, i: usize, j: usize, v: f64) {
    h[i][j] += v;
    if i != j {
        h[j][i] += v;
    }
}

/// Scatter one point's ρ/σ second derivatives (libxc packing, already
/// weighted) into `h`.
fn scatter_rho_sigma(h: &mut PointHessian, rr: &[f64], rs: &[f64], ss: &[f64]) {
    add_sym(h, RA, RA, rr[0]);
    add_sym(h, RA, RB, rr[1]);
    add_sym(h, RB, RB, rr[2]);
    for s in 0..2 {
        for c in 0..3 {
            add_sym(h, RA + s, SAA + c, rs[3 * s + c]);
        }
    }
    for (k, &(c1, c2)) in SIGMA2_PAIRS.iter().enumerate() {
        add_sym(h, SAA + c1, SAA + c2, ss[k]);
    }
}

/// Scatter one point's τ second derivatives (ρ×τ, σ×τ, τ×τ) into `h`.
fn scatter_tau(h: &mut PointHessian, rt: &[f64], st: &[f64], tt: &[f64]) {
    for t in 0..2 {
        for s in 0..2 {
            add_sym(h, RA + s, TA + t, rt[2 * s + t]);
        }
        for c in 0..3 {
            add_sym(h, SAA + c, TA + t, st[2 * c + t]);
        }
    }
    add_sym(h, TA, TA, tt[0]);
    add_sym(h, TA, TB, tt[1]);
    add_sym(h, TB, TB, tt[2]);
}

/// Scale every element of a libxc output buffer by the mixing weight.
fn weighted(w: f64, v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| w * x).collect()
}

/// Accumulate one LDA/GGA sub-functional (weight `w`) into the per-point
/// Hessians and first σ-derivatives.
fn accumulate_semilocal(
    f: &crate::libxc::XcFunctional,
    w: f64,
    (rho, sigma): (&[f64], &[f64]),
    hess: &mut [PointHessian],
    vsigma: &mut [[f64; 3]],
) {
    let n = hess.len();
    let mut rr = vec![0.0; 3 * n];
    if f.family() == FunctionalFamily::Lda {
        f.eval_lda_fxc_polarized(rho, &mut rr);
        let rr = weighted(w, &rr);
        for (g, h) in hess.iter_mut().enumerate() {
            scatter_rho_sigma(h, &rr[3 * g..], &[0.0; 6], &[0.0; 6]);
        }
        return;
    }
    let (mut rs, mut ss) = (vec![0.0; 6 * n], vec![0.0; 6 * n]);
    f.eval_gga_fxc_polarized(rho, sigma, &mut rr, &mut rs, &mut ss);
    let (mut exc, mut vrho, mut vs) = (vec![0.0; n], vec![0.0; 2 * n], vec![0.0; 3 * n]);
    f.eval_gga_polarized(rho, sigma, &mut exc, &mut vrho, &mut vs);
    let (rr, rs, ss) = (weighted(w, &rr), weighted(w, &rs), weighted(w, &ss));
    for g in 0..n {
        scatter_rho_sigma(&mut hess[g], &rr[3 * g..], &rs[6 * g..], &ss[6 * g..]);
        for c in 0..3 {
            vsigma[g][c] += w * vs[3 * g + c];
        }
    }
}

/// Accumulate one τ-only meta-GGA sub-functional (weight `w`).
fn accumulate_mgga(
    f: &crate::libxc::XcFunctional,
    w: f64,
    (rho, sigma, tau): (&[f64], &[f64], &[f64]),
    hess: &mut [PointHessian],
    vsigma: &mut [[f64; 3]],
) -> Result<(), String> {
    let n = hess.len();
    let mut d2 = crate::libxc::MggaFxcPolarized::zeros(n);
    f.eval_mgga_fxc_polarized(rho, sigma, tau, &mut d2)
        .map_err(|e| format!("{e}"))?;
    let (mut exc, mut vrho, mut vs, mut vtau) = (
        vec![0.0; n],
        vec![0.0; 2 * n],
        vec![0.0; 3 * n],
        vec![0.0; 2 * n],
    );
    f.eval_mgga_polarized(rho, sigma, tau, &mut exc, &mut vrho, &mut vs, &mut vtau);
    let rr = weighted(w, &d2.v2rho2);
    let rs = weighted(w, &d2.v2rhosigma);
    let ss = weighted(w, &d2.v2sigma2);
    let rt = weighted(w, &d2.v2rhotau);
    let st = weighted(w, &d2.v2sigmatau);
    let tt = weighted(w, &d2.v2tau2);
    for g in 0..n {
        let h = &mut hess[g];
        scatter_rho_sigma(h, &rr[3 * g..], &rs[6 * g..], &ss[6 * g..]);
        scatter_tau(h, &rt[4 * g..], &st[6 * g..], &tt[3 * g..]);
        for c in 0..3 {
            vsigma[g][c] += w * vs[3 * g + c];
        }
    }
    Ok(())
}

/// Per-point Hessians of `f = ρ ε_xc` in (ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ, τ_α, τ_β)
/// and the first σ-derivatives (v_σαα, v_σαβ, v_σββ), summed over the
/// sub-functionals of `xc` with their weights. Inputs are libxc-interleaved
/// polarized buffers (`rho` 2/pt, `sigma` 3/pt, `tau` 2/pt).
pub(crate) fn point_hessians(
    xc: &XcDef,
    rho: &[f64],
    sigma: &[f64],
    tau: &[f64],
) -> Result<(Vec<PointHessian>, Vec<[f64; 3]>), String> {
    let n = rho.len() / 2;
    let mut hess = vec![[[0.0; NVAR]; NVAR]; n];
    let mut vsigma = vec![[0.0; 3]; n];
    for (i, f) in xc.funcs.iter().enumerate() {
        let w = xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
        if f.family() == FunctionalFamily::MetaGga {
            accumulate_mgga(f, w, (rho, sigma, tau), &mut hess, &mut vsigma)?;
        } else {
            accumulate_semilocal(f, w, (rho, sigma), &mut hess, &mut vsigma);
        }
    }
    Ok((hess, vsigma))
}

fn dot3(x: &[f64; 3], y: &[f64; 3]) -> f64 {
    x[0] * y[0] + x[1] * y[1] + x[2] * y[2]
}

/// One grid point's reference and perturbation data, per spin `[α, β]`.
struct PointPerturbation {
    /// Reference ∇ρ_σ.
    grad: [[f64; 3]; 2],
    /// Perturbation δρ_σ, ∇δρ_σ, δτ_σ.
    drho: [f64; 2],
    dgrad: [[f64; 3]; 2],
    dtau: [f64; 2],
}

/// The linearised potential coefficients at one point, per spin: `u^σ`
/// (multiplies χ_μχ_ν), `w^σ` (multiplies ∇(χ_μχ_ν)) and `t^σ` (multiplies
/// ½ ∇χ_μ·∇χ_ν). See [`MggaFxcKernel::apply_with_ref`] for the derivation.
fn linearize_point(
    h: &PointHessian,
    vs: &[f64; 3],
    p: &PointPerturbation,
) -> ([f64; 2], [[f64; 3]; 2], [f64; 2]) {
    let [ga, gb] = &p.grad;
    let [da, db] = &p.dgrad;
    let mut dx = [0.0; NVAR];
    dx[RA] = p.drho[0];
    dx[RB] = p.drho[1];
    dx[SAA] = 2.0 * dot3(ga, da);
    dx[SAB] = dot3(ga, db) + dot3(gb, da);
    dx[SBB] = 2.0 * dot3(gb, db);
    dx[TA] = p.dtau[0];
    dx[TB] = p.dtau[1];
    let mut dv = [0.0; NVAR];
    for (i, row) in h.iter().enumerate() {
        dv[i] = row.iter().zip(&dx).map(|(a, b)| a * b).sum();
    }
    let mut w = [[0.0; 3]; 2];
    for ax in 0..3 {
        w[0][ax] =
            2.0 * dv[SAA] * ga[ax] + dv[SAB] * gb[ax] + 2.0 * vs[0] * da[ax] + vs[1] * db[ax];
        w[1][ax] =
            2.0 * dv[SBB] * gb[ax] + dv[SAB] * ga[ax] + 2.0 * vs[2] * db[ax] + vs[1] * da[ax];
    }
    ([dv[RA], dv[RB]], w, [dv[TA], dv[TB]])
}

/// Per-spin grid coefficients of the linearised potential, `(u, w, t)` with
/// `u`, `t` of length npts and `w` of shape (3, npts).
struct SpinCoefficients {
    u: Vec<f64>,
    w: Array2<f64>,
    t: Vec<f64>,
}

impl SpinCoefficients {
    fn zeros(npts: usize) -> Self {
        Self {
            u: vec![0.0; npts],
            w: Array2::zeros((3, npts)),
            t: vec![0.0; npts],
        }
    }
}

/// Reference state of an [`MggaFxcKernel`]: the per-point Hessians and σ
/// first derivatives at (D_α, D_β), the reference gradients, and the
/// per-spin density-floor mask production V_xc applies. Built once per
/// response solve; every matvec reuses it.
#[derive(Debug, Clone)]
pub struct MggaFxcReference {
    grad_a: Array2<f64>,
    grad_b: Array2<f64>,
    hess: Vec<PointHessian>,
    vsigma: Vec<[f64; 3]>,
    /// `ρ_σ > DENSITY_FLOOR` per spin: production V^σ drops every term at a
    /// point below it, so its derivative there is zero.
    live: [Vec<bool>; 2],
}

/// Precomputed grid + AO data for the τ-only meta-GGA f_xc response
/// (SCAN / r2SCAN / TPSS, as ferric's SCF evaluates them: no Laplacian).
///
/// It differentiates the polarized semilocal V_xc of `vxc.rs` as implemented,
/// INCLUDING its per-spin `DENSITY_FLOOR` mask: δV^σ is zero at points where
/// the reference ρ_σ ≤ 1e-10. LDA and GGA sub-functionals are accepted too
/// (their τ blocks are zero), so a mixed definition evaluates correctly; that
/// is also the exactness anchor (`mgga_kernel_on_gga_functional_equals_gga_kernel`).
#[derive(Debug)]
pub struct MggaFxcKernel {
    pub xc: XcDef,
    pub grid: Vec<GridPoint>,
    pub chi: Array2<f64>,  // (nbf, npts)
    pub dchi: Array3<f64>, // (3, nbf, npts)
    scratch: Mutex<VxcScratch>,
}

impl MggaFxcKernel {
    /// Build the kernel. Every meta-GGA sub-functional must have a libxc
    /// second derivative (libxc would otherwise terminate the process).
    pub fn new(
        mol: &Molecule,
        bs: &BasisSet,
        xc: XcDef,
        cfg: &AtomicGridConfig,
    ) -> Result<Self, String> {
        if let Some(f) = xc
            .funcs
            .iter()
            .find(|f| f.family() == FunctionalFamily::MetaGga && !f.has_fxc())
        {
            return Err(format!(
                "MggaFxcKernel: libxc has no second derivative for {f:?}"
            ));
        }
        let grid = build_atomic_grid(mol, cfg);
        let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let (chi, dchi) =
            eval_basis_and_grad_on_points(mol, bs, &pts).map_err(|e| format!("{e:?}"))?;
        Ok(Self {
            xc,
            grid,
            chi,
            dchi,
            scratch: Mutex::new(VxcScratch::new()),
        })
    }

    /// Evaluate the reference state at the converged (D_α, D_β).
    pub fn reference(
        &self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<MggaFxcReference, String> {
        let dens = eval_density_uks(d_a, d_b, &self.chi, &self.dchi);
        let (tau_a, tau_b) = eval_tau_uks(d_a, d_b, &self.dchi);
        let npts = self.grid.len();
        let (mut rho, mut sigma, mut tau) = (
            vec![0.0; 2 * npts],
            vec![0.0; 3 * npts],
            vec![0.0; 2 * npts],
        );
        for g in 0..npts {
            rho[2 * g] = dens.rho_a[g];
            rho[2 * g + 1] = dens.rho_b[g];
            for c in 0..3 {
                sigma[3 * g + c] = dens.sigma[(c, g)];
            }
            tau[2 * g] = tau_a[g];
            tau[2 * g + 1] = tau_b[g];
        }
        let (hess, vsigma) = point_hessians(&self.xc, &rho, &sigma, &tau)?;
        let live = [
            dens.rho_a.iter().map(|&r| r > DENSITY_FLOOR).collect(),
            dens.rho_b.iter().map(|&r| r > DENSITY_FLOOR).collect(),
        ];
        Ok(MggaFxcReference {
            grad_a: dens.grad_a,
            grad_b: dens.grad_b,
            hess,
            vsigma,
            live,
        })
    }

    /// (δV_xc^α, δV_xc^β) for the perturbation (δD_α, δD_β).
    ///
    /// # Derivation (spin σ, other spin σ̄)
    ///
    /// The polarized meta-GGA potential (`vxc.rs`) is
    ///
    ///   V^σ_{μν} = Σ_g w_g [ v_ρσ χ_μχ_ν
    ///                      + (2 v_{σσσ} ∇ρ_σ + v_{σαβ} ∇ρ_σ̄)·∇(χ_μχ_ν)
    ///                      + ½ v_τσ ∇χ_μ·∇χ_ν ].
    ///
    /// With x = (ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ, τ_α, τ_β) and H = ∂²f/∂x∂x
    /// (libxc v2rho2, v2rhosigma, v2rhotau, v2sigma2, v2sigmatau, v2tau2),
    /// the first-derivative changes are δv = H δx where
    ///
    ///   δσ_αα = 2∇ρ_α·∇δρ_α,  δσ_αβ = ∇ρ_α·∇δρ_β + ∇ρ_β·∇δρ_α,
    ///   δσ_ββ = 2∇ρ_β·∇δρ_β,  δτ_σ = ½ Σ_axis Σ_κλ δD^σ_κλ ∂χ_κ ∂χ_λ.
    ///
    /// τ is LINEAR in D, so v_τ itself never appears — only δv_τ. Then
    ///
    ///   δV^σ_{μν} = Σ_g w_g [ u^σ χ_μχ_ν + w^σ·∇(χ_μχ_ν) + ½ t^σ ∇χ_μ·∇χ_ν ],
    ///   u^σ = δv_ρσ,  t^σ = δv_τσ,
    ///   w^σ = 2 δv_{σσσ} ∇ρ_σ + δv_{σαβ} ∇ρ_σ̄ + 2 v_{σσσ} ∇δρ_σ + v_{σαβ} ∇δρ_σ̄,
    ///
    /// with every term at a point zeroed where the reference ρ_σ is below the
    /// production density floor.
    pub fn apply_with_ref(
        &self,
        r: &MggaFxcReference,
        d_delta_a: &Array2<f64>,
        d_delta_b: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let npts = self.grid.len();
        assert_eq!(r.hess.len(), npts, "reference built on another grid");
        let dd = eval_density_uks(d_delta_a, d_delta_b, &self.chi, &self.dchi);
        let (dtau_a, dtau_b) = eval_tau_uks(d_delta_a, d_delta_b, &self.dchi);
        let mut coef = [SpinCoefficients::zeros(npts), SpinCoefficients::zeros(npts)];
        let col = |a: &Array2<f64>, g: usize| [a[(0, g)], a[(1, g)], a[(2, g)]];
        for g in 0..npts {
            let p = PointPerturbation {
                grad: [col(&r.grad_a, g), col(&r.grad_b, g)],
                drho: [dd.rho_a[g], dd.rho_b[g]],
                dgrad: [col(&dd.grad_a, g), col(&dd.grad_b, g)],
                dtau: [dtau_a[g], dtau_b[g]],
            };
            let (u, w, t) = linearize_point(&r.hess[g], &r.vsigma[g], &p);
            for s in 0..2 {
                if !r.live[s][g] {
                    continue;
                }
                coef[s].u[g] = u[s];
                coef[s].t[g] = t[s];
                for ax in 0..3 {
                    coef[s].w[(ax, g)] = w[s][ax];
                }
            }
        }
        let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let buf = scratch.ensure((self.chi.nrows(), npts));
        let va = self.backproject(&coef[0], buf);
        let vb = self.backproject(&coef[1], buf);
        (va, vb)
    }

    /// Σ_g w_g [u χ_μχ_ν + w·(χ_μ∇χ_ν + χ_ν∇χ_μ) + ½ t ∇χ_μ·∇χ_ν], one GEMM
    /// per term (the `vxc.rs` idiom), symmetrized.
    fn backproject(&self, c: &SpinCoefficients, buf: &mut Array2<f64>) -> Array2<f64> {
        let wg = |g: usize| self.grid[g].weight;
        let npts = self.grid.len();
        let gemm = |a: &Array2<f64>, b: ndarray::ArrayView2<f64>| -> Array2<f64> {
            with_blas_threads(opt_in_blas_threads(), || a.dot(&b.t()))
        };
        let fac: Array1<f64> = (0..npts).map(|g| wg(g) * c.u[g]).collect();
        scale_columns_into(self.chi.view(), &fac, buf);
        let mut v = gemm(buf, self.chi.view());
        for axis in 0..3 {
            let dchi_ax = self.dchi.index_axis(Axis(0), axis);
            let fac: Array1<f64> = (0..npts).map(|g| wg(g) * c.w[(axis, g)]).collect();
            scale_columns_into(self.chi.view(), &fac, buf);
            let m = gemm(buf, dchi_ax);
            v = v + &m + &m.t();
            let fac: Array1<f64> = (0..npts).map(|g| 0.5 * wg(g) * c.t[g]).collect();
            scale_columns_into(dchi_ax, &fac, buf);
            v += &gemm(buf, dchi_ax);
        }
        0.5 * (&v + &v.t())
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
        // Kernel construction transitively resolves FERRIC_MEM_BUDGET_GB
        // (checked AO-grid eval); hold the crate-wide env lock so budget-
        // mutating tests in ks/ao_grid can't race this read.
        let _env_guard = crate::TEST_BUDGET_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let npts = 64;
        let nbf = 10;
        let mut rng = Xorshift64(0x243F6A8885A308D3);

        let chi = Array2::<f64>::from_shape_fn((nbf, npts), |_| rng.next_f64());
        let weights: Vec<f64> = (0..npts)
            .map(|_| 0.5 + 0.5 * rng.next_f64().abs())
            .collect();
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

    // ── GGA f_xc finite-difference-vs-analytic validation ───────────────────
    //
    // The load-bearing correctness proof for `GgaFxcKernel`. We finite-
    // difference the (already-validated) first-derivative GGA V_xc builder in
    // `vxc.rs` and compare its central difference against the analytic f_xc
    // response, at several ε, checking a clean quadratic (Richardson) error
    // trend. `ferric-scf` is a dev-dependency, so we obtain a genuine physical
    // spin-polarized reference density from a real UHF solve.

    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;

    /// Floor-free polarized GGA V_xc oracle for the FD check.
    ///
    /// This reproduces `vxc.rs::semilocal_vxc_polarized`'s math EXACTLY but
    /// WITHOUT its `DENSITY_FLOOR` masking. The floor is a production
    /// well-conditioning safeguard that zeroes contributions where ρ_σ < 1e-10;
    /// under a finite perturbation it flips on/off between the ±ε evaluations
    /// at tail points, injecting O(1)-magnitude discontinuity noise into the
    /// central difference (the ~1% non-converging plateau a floored oracle
    /// gives). The analytic kernel is likewise floor-free, so a floor-free
    /// oracle is the correct apples-to-apples FD reference.
    fn vxc_polarized_no_floor(
        grid: &[GridPoint],
        chi: &Array2<f64>,
        dchi: &Array3<f64>,
        xc: &XcDef,
        dens: &crate::density_on_grid::UksDensityGrid,
    ) -> (Array2<f64>, Array2<f64>) {
        use crate::libxc::FunctionalFamily;
        let (nbf, npts) = chi.dim();
        let w: Vec<f64> = grid.iter().map(|g| g.weight).collect();

        // Interleaved libxc inputs.
        let mut rho_in = vec![0.0f64; 2 * npts];
        let mut sigma_in = vec![0.0f64; 3 * npts];
        for g in 0..npts {
            rho_in[2 * g] = dens.rho_a[g];
            rho_in[2 * g + 1] = dens.rho_b[g];
            sigma_in[3 * g] = dens.sigma[(0, g)];
            sigma_in[3 * g + 1] = dens.sigma[(1, g)];
            sigma_in[3 * g + 2] = dens.sigma[(2, g)];
        }

        let mut vrho_a = vec![0.0f64; npts];
        let mut vrho_b = vec![0.0f64; npts];
        let mut vs_aa = vec![0.0f64; npts];
        let mut vs_ab = vec![0.0f64; npts];
        let mut vs_bb = vec![0.0f64; npts];
        for (i, f) in xc.funcs.iter().enumerate() {
            let w_i = xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
            let mut exc = vec![0.0f64; npts];
            let mut vrho = vec![0.0f64; 2 * npts];
            match f.family() {
                FunctionalFamily::Lda => {
                    f.eval_lda_polarized(&rho_in, &mut exc, &mut vrho);
                }
                _ => {
                    let mut vsigma = vec![0.0f64; 3 * npts];
                    f.eval_gga_polarized(&rho_in, &sigma_in, &mut exc, &mut vrho, &mut vsigma);
                    for g in 0..npts {
                        vs_aa[g] += w_i * vsigma[3 * g];
                        vs_ab[g] += w_i * vsigma[3 * g + 1];
                        vs_bb[g] += w_i * vsigma[3 * g + 2];
                    }
                }
            }
            for g in 0..npts {
                vrho_a[g] += w_i * vrho[2 * g];
                vrho_b[g] += w_i * vrho[2 * g + 1];
            }
        }

        let build = |vrho_sig: &[f64],
                     vs_self: &[f64],
                     vs_cross: &[f64],
                     grad_self: &Array2<f64>,
                     grad_cross: &Array2<f64>|
         -> Array2<f64> {
            // LDA piece.
            let mut chi_s = chi.clone();
            for g in 0..npts {
                let s = w[g] * vrho_sig[g];
                for mu in 0..nbf {
                    chi_s[(mu, g)] = chi[(mu, g)] * s;
                }
            }
            let mut v: Array2<f64> = chi_s.dot(&chi.t());
            // GGA piece.
            for axis in 0..3 {
                let dchi_axis = dchi.index_axis(Axis(0), axis);
                let gs = grad_self.index_axis(Axis(0), axis);
                let gc = grad_cross.index_axis(Axis(0), axis);
                let mut buf = chi.clone();
                for g in 0..npts {
                    let f_ax = 2.0 * w[g] * vs_self[g] * gs[g] + w[g] * vs_cross[g] * gc[g];
                    for mu in 0..nbf {
                        buf[(mu, g)] = chi[(mu, g)] * f_ax;
                    }
                }
                let m: Array2<f64> = buf.dot(&dchi_axis.t());
                v = v + &m + &m.t();
            }
            0.5 * (&v + &v.t())
        };

        let va = build(&vrho_a, &vs_aa, &vs_ab, &dens.grad_a, &dens.grad_b);
        let vb = build(&vrho_b, &vs_bb, &vs_ab, &dens.grad_b, &dens.grad_a);
        (va, vb)
    }

    /// Central-difference the floor-free polarized GGA V_xc (first derivative)
    /// w.r.t. a density-matrix perturbation:  δV ≈ [V(D+εδD) − V(D−εδD)]/(2ε).
    fn fd_vxc_response(
        grid: &[GridPoint],
        chi: &Array2<f64>,
        dchi: &Array3<f64>,
        xc: &XcDef,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
        dd_a: &Array2<f64>,
        dd_b: &Array2<f64>,
        eps: f64,
    ) -> (Array2<f64>, Array2<f64>) {
        use crate::density_on_grid::eval_density_uks;

        let dens_plus = eval_density_uks(&(d_a + &(eps * dd_a)), &(d_b + &(eps * dd_b)), chi, dchi);
        let dens_minus =
            eval_density_uks(&(d_a - &(eps * dd_a)), &(d_b - &(eps * dd_b)), chi, dchi);

        let (va_p, vb_p) = vxc_polarized_no_floor(grid, chi, dchi, xc, &dens_plus);
        let (va_m, vb_m) = vxc_polarized_no_floor(grid, chi, dchi, xc, &dens_minus);

        let inv = 1.0 / (2.0 * eps);
        ((&va_p - &va_m) * inv, (&vb_p - &vb_m) * inv)
    }

    fn max_abs(a: &Array2<f64>) -> f64 {
        a.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
    }
    fn max_abs_diff2(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |m, (&x, &y)| m.max((x - y).abs()))
    }
    fn fro(a: &Array2<f64>) -> f64 {
        a.iter().fold(0.0_f64, |s, &x| s + x * x).sqrt()
    }
    fn fro_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |s, (&x, &y)| s + (x - y) * (x - y))
            .sqrt()
    }

    /// PBE GGA f_xc response must match the central difference of the vxc.rs
    /// first-derivative builder, with a clean O(ε²) Richardson error trend.
    #[test]
    fn gga_fxc_matches_finite_difference_of_vxc() {
        // Kernel construction transitively resolves FERRIC_MEM_BUDGET_GB
        // (checked AO-grid eval); hold the crate-wide env lock so budget-
        // mutating tests in ks/ao_grid can't race this read.
        let _env_guard = crate::TEST_BUDGET_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // OH radical (doublet), 6-31G — small, converges fast, D_α ≠ D_β so all
        // spin channels (αα, αβ, ββ) of the kernel are exercised.
        let mol = Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
        let bs = basis::bundled("6-31g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        // A plain UHF density is a perfectly good *physical* reference for the
        // kernel FD check — the f_xc response is evaluated at whatever (D_α,D_β)
        // we pass; it need not be the PBE self-consistent density.
        let ucfg = ferric_scf::uhf::UhfConfig {
            energy_conv: 1e-9,
            density_conv: 1e-7,
            ..Default::default()
        };
        let res = ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &ucfg).unwrap();
        let d_a = res.density_alpha.clone();
        let d_b = res.density_beta.clone().unwrap();

        // PBE (pure GGA — exercises v2rho2 + v2rhosigma + v2sigma2).
        let cfg = AtomicGridConfig {
            n_radial: 50,
            n_angular: 110,
            ..Default::default()
        };
        let xc_def = crate::libxc::xc_def_from_name_nspin("PBE", 2).unwrap();
        let kernel = GgaFxcKernel::new(&mol, &bs, xc_def, &cfg).unwrap();

        // A second XcDef handle for the vxc oracle (kernel consumed its own).
        let xc_oracle = crate::libxc::xc_def_from_name_nspin("PBE", 2).unwrap();

        // Reference density on the kernel grid, and a random symmetric δD.
        // Scale δD to a small fraction of ‖D‖ so the central difference sits in
        // the clean quadratic (truncation-dominated) regime rather than a
        // round-off / density-floor-masking regime: the FD truncation error is
        // ∝ ε²·‖δD‖³·V''', so an over-large δD both inflates that error and
        // pushes many grid points across the vxc DENSITY_FLOOR between the ±
        // evaluations, producing the noisy non-monotone ladder a raw O(1) δD
        // gives.
        let ref_dens = kernel.reference_density(&d_a, &d_b);
        let nbf = d_a.nrows();
        let d_scale = max_abs(&d_a).max(max_abs(&d_b)).max(1e-6);
        let dd_target = 1e-3 * d_scale; // ‖δD‖∞ ≈ 0.1% of ‖D‖∞
        let mut rng = Xorshift64(0xC0FFEE123456789);
        let mut mk_sym = || -> Array2<f64> {
            let mut m = Array2::<f64>::from_shape_fn((nbf, nbf), |_| rng.next_f64());
            m = 0.5 * (&m + &m.t());
            let mx = max_abs(&m).max(1e-30);
            (dd_target / mx) * &m
        };
        let dd_a = mk_sym();
        let dd_b = mk_sym();

        // Analytic response.
        let (an_a, an_b) = kernel.apply_with_ref(&ref_dens, &dd_a, &dd_b);
        // Frobenius scale of the (α,β) response stacked — robust to the single
        // near-zero matrix elements that make a max-abs relative metric noisy.
        let scale = (fro(&an_a).powi(2) + fro(&an_b).powi(2)).sqrt().max(1e-30);

        // The central difference has truncation error → 0 as ε → 0. Because ρ
        // (hence σ = |∇ρ|², hence V_xc) is nonlinear in D, that error is a
        // superposition of O(ε²), O(ε⁴)… terms whose α/β-block contributions can
        // partially cancel in a Frobenius sum at intermediate ε — so the ladder
        // is only guaranteed monotone once ε is small enough that the leading
        // term dominates. We halve ε across a small-ε ladder and require (a) a
        // tight final agreement and (b) a clean, monotone, faster-than-linear
        // shrink of the residual. This is the load-bearing correctness proof.
        let rel_at = |eps: f64| -> f64 {
            let (fd_a, fd_b) = fd_vxc_response(
                &kernel.grid,
                &kernel.chi,
                &kernel.dchi,
                &xc_oracle,
                &d_a,
                &d_b,
                &dd_a,
                &dd_b,
                eps,
            );
            let rel =
                (fro_diff(&an_a, &fd_a).powi(2) + fro_diff(&an_b, &fd_b).powi(2)).sqrt() / scale;
            eprintln!("GGA-fxc FD check: eps={eps:.2e}  rel_err(fro)={rel:.3e}");
            rel
        };
        // Halving ladder in the small-ε (truncation-dominated) regime. As ε
        // shrinks the residual falls steeply until it meets the central-
        // difference round-off floor (∝ ulp/ε from the /2ε division), ~1–2e-6
        // here for a 6-31G OH grid.
        let eps_ladder = [1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3];
        let rel_errs: Vec<f64> = eps_ladder.iter().map(|&e| rel_at(e)).collect();

        // Final agreement must be tight — the analytic kernel reproduces the FD
        // limit of the validated first-derivative V_xc builder to the FD floor.
        let best = rel_errs.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            best < 5e-6,
            "GGA f_xc analytic vs FD: best relative error {best:e} (want < 5e-6). \
             ladder={rel_errs:?}"
        );
        // Clean convergence into the floor: while the ladder is still
        // TRUNCATION-DOMINATED the first halving must shrink the residual
        // super-linearly (> 4×, i.e. ≥ O(ε²)), and the sequence must be
        // monotonically decreasing.
        //
        // # Why both checks are gated on `rel_errs[0] > FD_FLOOR` (2026-09-16)
        //
        // Both were written when this test's reference density came from an
        // OH/6-31G UHF solve that landed on a SADDLE 4.22 eV above the true
        // minimum (E = −75.207997 versus PySCF's −75.363168; see
        // `ferric-scf`'s `oh_reaches_the_pyscf_reference_after_the_fix`). At
        // that wrong density the ladder started at 6.1e-4 — comfortably
        // truncation-dominated — and the O(ε²) trend was real and measurable.
        //
        // `fix/scf-unconstrained-state-selection` made `solve_uhf` honour the
        // configured guess, so this test now differentiates the CORRECT OH
        // density, and the analytic kernel agrees with the finite difference
        // about a million times better: the ladder is
        // 1.077e-9 → 3.051e-10 → 2.438e-10 → 4.809e-10, entirely AT the
        // central-difference round-off floor. At that level the residual is
        // ulp noise divided by ε, so it neither falls quadratically nor stays
        // monotone, and demanding that it do so is asking arithmetic noise to
        // obey a truncation law — a guard whose pass condition is unreachable.
        //
        // Note the ACCURACY bar above (`best < 5e-6`) is untouched and is now
        // met by four orders of magnitude rather than three-fold. The order
        // checks below still run in full whenever the ladder starts above the
        // floor, which is what they were written to police; they are skipped —
        // loudly — only when there is no truncation error left to measure.
        const FD_FLOOR: f64 = 1e-7;
        if rel_errs[0] > FD_FLOOR {
            assert!(
                rel_errs[1] < rel_errs[0] && rel_errs[0] / rel_errs[1] > 4.0,
                "GGA f_xc FD residual must fall ≥ quadratically on the first halving: \
                 {:.3e} → {:.3e} (want ratio > 4). ladder={rel_errs:?}",
                rel_errs[0],
                rel_errs[1]
            );
            for w in rel_errs.windows(2) {
                assert!(
                    w[1] <= w[0] * 1.0000001,
                    "GGA f_xc FD residual must be monotonically non-increasing: \
                     {:.3e} → {:.3e}. ladder={rel_errs:?}",
                    w[0],
                    w[1]
                );
            }
        } else {
            eprintln!(
                "GGA-fxc FD check: the whole ladder is at the round-off floor \
                 (first residual {:.3e} <= {FD_FLOOR:.0e}), so the O(eps^2) trend and \
                 monotonicity checks are SKIPPED -- there is no truncation error left \
                 to measure. The accuracy bar (best = {best:.3e} < 5e-6) still applies \
                 and is what proves the kernel here. ladder={rel_errs:?}",
                rel_errs[0]
            );
        }
    }

    /// The GGA kernel must transparently reproduce the LDA kernel on a pure-LDA
    /// functional (σ-coupling terms all vanish). Cross-checks both kernels on
    /// the same reference + perturbation.
    #[test]
    fn gga_kernel_matches_lda_kernel_on_lda_functional() {
        // Kernel construction transitively resolves FERRIC_MEM_BUDGET_GB
        // (checked AO-grid eval); hold the crate-wide env lock so budget-
        // mutating tests in ks/ao_grid can't race this read.
        let _env_guard = crate::TEST_BUDGET_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
        let bs = basis::bundled("6-31g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let ucfg = ferric_scf::uhf::UhfConfig {
            energy_conv: 1e-9,
            density_conv: 1e-7,
            ..Default::default()
        };
        let res = ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &ucfg).unwrap();
        let d_a = res.density_alpha.clone();
        let d_b = res.density_beta.clone().unwrap();

        let cfg = AtomicGridConfig {
            n_radial: 50,
            n_angular: 110,
            ..Default::default()
        };
        let lda_k = LdaFxcKernel::new(
            &mol,
            &bs,
            crate::libxc::xc_def_from_name_nspin("LDA", 2).unwrap(),
            &cfg,
        )
        .unwrap();
        let gga_k = GgaFxcKernel::new(
            &mol,
            &bs,
            crate::libxc::xc_def_from_name_nspin("LDA", 2).unwrap(),
            &cfg,
        )
        .unwrap();

        let nbf = d_a.nrows();
        let mut rng = Xorshift64(0x1357924680ABCDEF);
        let mut m = Array2::<f64>::from_shape_fn((nbf, nbf), |_| rng.next_f64());
        m = 0.5 * (&m + &m.t());
        let dd_a = m.clone();
        let mut m2 = Array2::<f64>::from_shape_fn((nbf, nbf), |_| rng.next_f64());
        m2 = 0.5 * (&m2 + &m2.t());
        let dd_b = m2;

        let (lra, lrb) = lda_k.reference_density(&d_a, &d_b);
        let (lda_va, lda_vb) = lda_k.apply_with_ref(&lra, &lrb, &dd_a, &dd_b);

        let gref = gga_k.reference_density(&d_a, &d_b);
        let (gga_va, gga_vb) = gga_k.apply_with_ref(&gref, &dd_a, &dd_b);

        let scale = max_abs(&lda_va).max(max_abs(&lda_vb)).max(1e-30);
        let da = max_abs_diff2(&lda_va, &gga_va) / scale;
        let db = max_abs_diff2(&lda_vb, &gga_vb) / scale;
        assert!(
            da < 1e-10 && db < 1e-10,
            "GGA kernel on LDA functional must match LDA kernel: rel da={da:e}, db={db:e}"
        );
    }

    // ── meta-GGA f_xc kernel ────────────────────────────────────────────────

    /// UHF reference densities (D_α, D_β) for `xyz` in 6-31G.
    fn uhf_reference(xyz: &str, mult: usize) -> (Molecule, BasisSet, Array2<f64>, Array2<f64>) {
        let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
        let bs = basis::bundled("6-31g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds =
            ferric_scf::screening::SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let ucfg = ferric_scf::uhf::UhfConfig {
            energy_conv: 1e-9,
            density_conv: 1e-7,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let res = ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &ucfg).unwrap();
        let d_b = res.density_beta.clone().unwrap();
        (mol, bs, res.density_alpha, d_b)
    }

    const OH: &str = "2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
    const WATER: &str = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n";

    fn small_grid() -> AtomicGridConfig {
        AtomicGridConfig {
            n_radial: 50,
            n_angular: 110,
            ..Default::default()
        }
    }

    /// A random symmetric matrix scaled to ‖·‖∞ = `scale`.
    fn rand_sym(rng: &mut Xorshift64, n: usize, scale: f64) -> Array2<f64> {
        let m = Array2::<f64>::from_shape_fn((n, n), |_| rng.next_f64());
        let m = 0.5 * (&m + &m.t());
        (scale / max_abs(&m).max(1e-30)) * &m
    }

    /// libxc's polarized meta-GGA first derivatives at one point, as the
    /// 7-vector (v_ρα, v_ρβ, v_σαα, v_σαβ, v_σββ, v_τα, v_τβ), summed over `xc`.
    fn mgga_first_derivs(xc: &XcDef, x: &[f64; NVAR]) -> [f64; NVAR] {
        let rho = [x[RA], x[RB]];
        let sigma = [x[SAA], x[SAB], x[SBB]];
        let tau = [x[TA], x[TB]];
        let mut out = [0.0; NVAR];
        for (i, f) in xc.funcs.iter().enumerate() {
            let w = xc.weights.as_ref().map_or(1.0, |ws| ws[i]);
            let (mut e, mut vr, mut vs, mut vt) = ([0.0], [0.0; 2], [0.0; 3], [0.0; 2]);
            f.eval_mgga_polarized(&rho, &sigma, &tau, &mut e, &mut vr, &mut vs, &mut vt);
            let v = [vr[0], vr[1], vs[0], vs[1], vs[2], vt[0], vt[1]];
            for k in 0..NVAR {
                out[k] += w * v[k];
            }
        }
        out
    }

    /// Pins libxc's τ-block PACKING (ρ×τ, σ×τ, τ×τ) and ferric's scatter of
    /// it: every column of the assembled 7×7 Hessian equals a central FD of
    /// libxc's own first derivatives, at spin-asymmetric points (ρ_α ≠ ρ_β,
    /// τ_α ≠ τ_β, σ_αα ≠ σ_ββ), so a transposed ρα·τβ / ρβ·τα block, a
    /// swapped spin or a missing term fails. Independent of any grid or AO
    /// code. Artifact hypothesis: a packing error gives an O(1) relative
    /// column error; a correct packing leaves only FD truncation (~1e-7).
    #[test]
    fn mgga_point_hessian_matches_fd_of_libxc_vxc() {
        let mut rng = Xorshift64(0x9E3779B97F4A7C15);
        for name in ["r2SCAN", "SCAN", "TPSS"] {
            let xc = crate::libxc::xc_def_from_name_nspin(name, 2).unwrap();
            let mut worst = 0.0_f64;
            for _ in 0..6 {
                let ra = 0.05 + 0.6 * rng.next_f64().abs();
                let rb = 0.03 + 0.4 * rng.next_f64().abs();
                let ga = [rng.next_f64(), rng.next_f64(), rng.next_f64()].map(|v| 0.5 * v);
                let gb = [rng.next_f64(), rng.next_f64(), rng.next_f64()].map(|v| 0.3 * v);
                let (saa, sab, sbb) = (dot3(&ga, &ga), dot3(&ga, &gb), dot3(&gb, &gb));
                // τ_σ comfortably above the von Weizsäcker bound σ_σσ/(8ρ_σ).
                let ta = saa / (8.0 * ra) * 1.7 + 0.2 * ra;
                let tb = sbb / (8.0 * rb) * 2.3 + 0.1 * rb;
                let x = [ra, rb, saa, sab, sbb, ta, tb];
                let rho = [ra, rb];
                let sigma = [saa, sab, sbb];
                let tau = [ta, tb];
                let (h, _) = point_hessians(&xc, &rho, &sigma, &tau).unwrap();
                let h = h[0];
                let hmax = h.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
                for j in 0..NVAR {
                    let step = 1e-5 * x[j].abs().max(1e-3);
                    let (mut xp, mut xm) = (x, x);
                    xp[j] += step;
                    xm[j] -= step;
                    let (gp, gm) = (mgga_first_derivs(&xc, &xp), mgga_first_derivs(&xc, &xm));
                    for i in 0..NVAR {
                        let fd = (gp[i] - gm[i]) / (2.0 * step);
                        worst = worst.max((fd - h[i][j]).abs() / hmax);
                    }
                }
            }
            eprintln!("{name}: point Hessian vs FD of libxc vxc, worst rel = {worst:.3e}");
            assert!(
                worst < 1e-6,
                "{name}: point Hessian vs FD worst rel {worst:e}"
            );
        }
    }

    /// Relative Frobenius distance of a stacked (α, β) pair.
    fn rel_pair(a: (&Array2<f64>, &Array2<f64>), b: (&Array2<f64>, &Array2<f64>)) -> f64 {
        let num = (fro_diff(a.0, b.0).powi(2) + fro_diff(a.1, b.1).powi(2)).sqrt();
        num / (fro(a.0).powi(2) + fro(a.1).powi(2)).sqrt().max(1e-30)
    }

    /// Production polarized V_xc (`vxc.rs`, WITH its density floor) at (D_α, D_β).
    fn production_vxc_polarized(
        k: &MggaFxcKernel,
        xc: &XcDef,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let dens = eval_density_uks(d_a, d_b, &k.chi, &k.dchi);
        let (ta, tb) = eval_tau_uks(d_a, d_b, &k.dchi);
        let (_, va, vb) = crate::vxc::semilocal_vxc_polarized(
            &k.grid,
            &k.chi,
            &k.dchi,
            &dens,
            Some((&ta, &tb)),
            xc,
        );
        (va, vb)
    }

    /// The meta-GGA kernel is the derivative of the production polarized
    /// V_xc: δV(δD_α, δD_β) equals a central FD of `semilocal_vxc_polarized`
    /// (floor included — the kernel applies the same per-spin mask), on the
    /// OH doublet (D_α ≠ D_β, so αα/αβ/ββ and both τ spins are exercised),
    /// for three random perturbations. r2SCAN is the tight target; SCAN and
    /// TPSS must agree too.
    ///
    /// Artifact hypothesis: a factor-of-2 τ error (e.g. ½ dropped in the t
    /// back-projection, or δτ built from the total instead of the per-spin
    /// D) makes the τ term — a sizeable fraction of the meta-GGA response —
    /// wrong by 100%, so the relative error would be O(1e-1..1), flat in ε. A
    /// correct kernel leaves an FD residual that falls with ε to the
    /// round-off floor.
    #[test]
    fn mgga_fxc_matches_finite_difference_of_production_vxc_polarized() {
        let _env_guard = crate::TEST_BUDGET_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (mol, bs, d_a, d_b) = uhf_reference(OH, 2);
        let nbf = d_a.nrows();
        for (name, bar) in [("r2SCAN", 1e-6), ("SCAN", 1e-5), ("TPSS", 1e-6)] {
            let xc = crate::libxc::xc_def_from_name_nspin(name, 2).unwrap();
            let k = MggaFxcKernel::new(
                &mol,
                &bs,
                crate::libxc::xc_def_from_name_nspin(name, 2).unwrap(),
                &small_grid(),
            )
            .unwrap();
            let r = k.reference(&d_a, &d_b).unwrap();
            let mut rng = Xorshift64(0xC0FFEE123456789);
            let mut worst = 0.0_f64;
            for _ in 0..3 {
                let dd_a = rand_sym(&mut rng, nbf, 1e-3);
                let dd_b = rand_sym(&mut rng, nbf, 1e-3);
                let (an_a, an_b) = k.apply_with_ref(&r, &dd_a, &dd_b);
                let mut ladder = Vec::new();
                for eps in [1e-2, 5e-3, 2.5e-3] {
                    let (pa, pb) = production_vxc_polarized(
                        &k,
                        &xc,
                        &(&d_a + &(eps * &dd_a)),
                        &(&d_b + &(eps * &dd_b)),
                    );
                    let (ma, mb) = production_vxc_polarized(
                        &k,
                        &xc,
                        &(&d_a - &(eps * &dd_a)),
                        &(&d_b - &(eps * &dd_b)),
                    );
                    let inv = 0.5 / eps;
                    let fd = ((&pa - &ma) * inv, (&pb - &mb) * inv);
                    ladder.push(rel_pair((&an_a, &an_b), (&fd.0, &fd.1)));
                }
                eprintln!("{name} polarized kernel vs FD(V): ladder {ladder:?}");
                worst = worst.max(ladder.iter().cloned().fold(f64::INFINITY, f64::min));
            }
            assert!(
                worst < bar,
                "{name}: polarized kernel vs FD worst {worst:e} (bar {bar:e})"
            );
        }
    }

    /// Closed shell: the polarized kernel at (D/2, D/2) along (X/2, X/2) is
    /// the derivative of the CLOSED-shell production V_xc[D] along X (the
    /// route `XcFunctional.response` and `cphf(xc=)` take), water/6-31G.
    #[test]
    fn mgga_fxc_matches_finite_difference_of_production_vxc_closed() {
        let _env_guard = crate::TEST_BUDGET_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (mol, bs, d_a, d_b) = uhf_reference(WATER, 1);
        let d = &d_a + &d_b;
        let nbf = d.nrows();
        let xc = crate::libxc::xc_def_from_name_nspin("r2SCAN", 1).unwrap();
        let k = MggaFxcKernel::new(
            &mol,
            &bs,
            crate::libxc::xc_def_from_name_nspin("r2SCAN", 2).unwrap(),
            &small_grid(),
        )
        .unwrap();
        let half = 0.5 * &d;
        let r = k.reference(&half, &half).unwrap();
        let vclosed = |dm: &Array2<f64>| {
            let dens = eval_density_closed(dm, &k.chi, &k.dchi);
            let tau = crate::density_on_grid::eval_tau_closed(dm, &k.dchi);
            crate::vxc::semilocal_vxc_closed(&k.grid, &k.chi, &k.dchi, &dens, Some(&tau), &xc).1
        };
        let mut rng = Xorshift64(0x5DEECE66D);
        let mut worst = 0.0_f64;
        for _ in 0..3 {
            let x = rand_sym(&mut rng, nbf, 2e-3);
            let xh = 0.5 * &x;
            let (va, vb) = k.apply_with_ref(&r, &xh, &xh);
            assert!(
                max_abs_diff2(&va, &vb) <= 1e-12 * max_abs(&va),
                "α ≠ β at equal spins"
            );
            let mut ladder = Vec::new();
            for eps in [1e-2, 5e-3, 2.5e-3] {
                let fd =
                    (vclosed(&(&d + &(eps * &x))) - vclosed(&(&d - &(eps * &x)))) * (0.5 / eps);
                ladder.push(fro_diff(&va, &fd) / fro(&va));
            }
            eprintln!("r2SCAN closed kernel vs FD(V): ladder {ladder:?}");
            worst = worst.max(ladder.iter().cloned().fold(f64::INFINITY, f64::min));
        }
        assert!(worst < 1e-6, "closed r2SCAN kernel vs FD worst {worst:e}");
    }

    /// Exactness anchor: on functionals with NO τ dependence the meta-GGA
    /// kernel reduces to the validated GGA kernel. This runs the meta-GGA
    /// kernel's own code end to end — its point-Hessian assembly (ρ/σ
    /// scatter), δσ/δτ construction, linearisation, density-floor mask and
    /// three-term back-projection (the τ GEMMs run on a zero t) — against the
    /// independent `GgaFxcKernel`. PBE exercises the σ blocks, a weighted
    /// LDA+GGA mix (B3LYP's semilocal part) exercises the sub-functional
    /// weights and the mixed-family path.
    ///
    /// Blind spot (CLAUDE.md "anchor blind spots"): with every τ block zero
    /// this anchor cannot see a defect in the τ blocks or the t term; the FD
    /// tests and `mgga_point_hessian_matches_fd_of_libxc_vxc` cover those.
    /// The GGA kernel has no density floor, so the comparison lifts the
    /// meta-GGA kernel's mask (the floored difference is printed: it is the
    /// whole of the residual, measured 6.1e-9 for PBE on OH/6-31G).
    #[test]
    fn mgga_kernel_on_gga_functional_equals_gga_kernel() {
        let _env_guard = crate::TEST_BUDGET_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (mol, bs, d_a, d_b) = uhf_reference(OH, 2);
        let nbf = d_a.nrows();
        let mut rng = Xorshift64(0x2545F4914F6CDD1D);
        let dd_a = rand_sym(&mut rng, nbf, 1.0);
        let dd_b = rand_sym(&mut rng, nbf, 1.0);
        for name in ["PBE", "B3LYP", "LDA"] {
            let def = || crate::libxc::xc_def_from_name_nspin(name, 2).unwrap();
            let gk = GgaFxcKernel::new(&mol, &bs, def(), &small_grid()).unwrap();
            let mk = MggaFxcKernel::new(&mol, &bs, def(), &small_grid()).unwrap();
            let (ga, gb) = gk.apply_with_ref(&gk.reference_density(&d_a, &d_b), &dd_a, &dd_b);
            let mut r = mk.reference(&d_a, &d_b).unwrap();
            let (fa, fb) = mk.apply_with_ref(&r, &dd_a, &dd_b);
            // The GGA kernel has no floor: lift the mask to compare like for like.
            let n = r.live[0].len();
            let dead = r.live.iter().flatten().filter(|l| !**l).count();
            r.live = [vec![true; n], vec![true; n]];
            let (ma, mb) = mk.apply_with_ref(&r, &dd_a, &dd_b);
            let rel = rel_pair((&ga, &gb), (&ma, &mb));
            let floor = rel_pair((&ga, &gb), (&fa, &fb));
            eprintln!(
                "{name}: meta-GGA vs GGA kernel rel = {rel:.3e} unmasked, {floor:.3e} with the \
                 floor ({dead} of {} spin-points below it)",
                2 * n
            );
            assert!(
                rel < 1e-12,
                "{name}: meta-GGA kernel vs GGA kernel rel {rel:e}"
            );
        }
    }
}
