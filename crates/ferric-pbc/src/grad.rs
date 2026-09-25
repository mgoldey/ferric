//! Analytic nuclear gradients (forces) of the Gamma-point periodic RHF, UHF,
//! RKS and UKS — the Rust port of `reference/pbc/pbc_grad.py` (FINDINGS
//! "Iteration 16") and `reference/pbc/pbc_grad_open.py` ("Iteration 17").
//!
//! Energy (Gamma, G = 0 dropped everywhere; `v_M` = Madelung shift of
//! `exxdiv = ewald`, 0 for `none`; `α` = exact-exchange fraction, 1 for
//! HF, 0.25 for PBE0, 0 for LDA/GGA; `D = D_α + D_β`; RHF/RKS are the
//! same with `D_α = D_β = D/2`):
//!
//! ```text
//! E = Σ D h + Σ Γ_μνλσ I_μνλσ − (α v_M/2) Σ_σ tr(D_σ S D_σ S) + E_xc[ρ_α, ρ_β] + E_nn,
//! Γ = ½ D_μν D_λσ − (α/2) Σ_σ D^σ_μλ D^σ_νσ          (RHF: ½ DD − ¼ DD)
//! h = T + V_SR(ω) + V_LR(ω) + c0 Z_tot S,   c0 = π/(ω²Ω)   (PeriodicHcore)
//! I = (1/Ω) Σ_{G≠0} (4π/G²) Re[P*_μν(G) P_λσ(G)]            (DenseAftEri, pure AFT)
//! F_σ = h + J[D] − α (K[D_σ] + v_M S D_σ S) + V_xc^σ
//! ```
//!
//! Moving atom `A` moves its nucleus, every basis function centred on it
//! AND (KS) every grid point homed on it, in every lattice image.
//! Differentiating at fixed (converged) `D_σ`:
//!
//! ```text
//! dE/dR_A = Σ D dT/dR_A                          lattice-summed shifted 1e derivative blocks
//!         + Σ D dV_SR/dR_A   (basis + nucleus)   erfc 3-centre derivative, Gaussian nuclei,
//!                                                nucleus = −(bra + ket) by translation invariance
//!         + Σ D dV_LR/dR_A   basis:   −(2/Ω) Σ_{G∈half} v_ω Re[ρ'_A* S(G)],  ρ'_A = 2 Σ_{μ∈A,ν} D_μν Q_μν
//!                            nucleus: −(2/Ω) Σ_{G∈half} v_ω Re[ρ* (−iG) Z_A e^{−iG·R_A}]
//!         + Σ Γ dI/dR_A    = (8/Ω) Σ_{G∈half} (4π/G²) Re Σ_{μ∈A,ν} Q*_μν Z_μν,
//!                            Z(G) = ½ D ρ(G) − (α/2) Σ_σ D_σ P(G) D_σ
//!         + Σ M dS/dR_A,     M = −W + c0 Z_tot D − α v_M Σ_σ D_σ S D_σ,   W = Σ_σ D_σ F_σ D_σ
//!         + dE_nn/dR_A       Ewald: SR erfc + LR structure factor
//!         + dE_xc/dR_A       AO term + grid response (below)
//! ```
//!
//! with `Q_μν = ∂P_μν/∂A_μ` the bra-centre pair-FT derivative
//! ([`crate::pair_ft::pair_ft_deriv_chunked`]; the ket derivative is `Q_νμ`
//! because `P_μν = P_νμ` at reciprocal-lattice G), streamed per G chunk
//! together with `P` and contracted immediately — `Q` (3 × the pair FT) is
//! never held for more than one chunk. `v_ω = 4π/G² e^{−G²/4ω²}`.
//!
//! RHF reduces exactly: `Σ_σ D_σ X D_σ = ½ D X D` for `D_σ = D/2`, so
//! `W = ½ D F D`, the Madelung term is `−(v_M/2) D S D` and
//! `Z = ½ D ρ − ¼ D P D` (the restricted path computes `½ D X D` directly;
//! the factors are powers of two, so it is the Iteration-16 arithmetic).
//!
//! # The overlap-coupled matrix `M`
//!
//! * `−W` is the usual Pulay term (orthonormality constraint, per spin).
//! * `c0 Z_tot D`: `h`'s G = 0 bookkeeping `V_G0 = c0 Z_tot S` enters ONLY
//!   through `S`. (The prototype also carried `−c0 N D + α c0 Σ_σ D_σ S D_σ`
//!   from the Ewald-split ERI's own G = 0 term; the dense-AFT ERI here is
//!   pure reciprocal space and has none.)
//! * `−α v_M Σ_σ D_σ S D_σ` is the explicit `S` dependence of
//!   `E_M = −(α v_M/2) Σ_σ tr(D_σ S D_σ S)`, PER SPIN. With it, forces for
//!   `exxdiv = ewald` and `none` agree to roundoff for any `α` and any spin
//!   state (`D_σ S D_σ = D_σ`, and the `−α v_M` shift of the occupied levels
//!   inside `W_σ` cancels it); in the RHF (total-density) form it is a
//!   1.7e-2..1.5e-1 Ha/Bohr error on open shells with `α > 0` that closed
//!   shells, `exxdiv = none` and the sum of forces CANNOT see (prototype,
//!   measured).
//!
//! # XC gradient (RKS / UKS; LDA, GGA, global hybrids)
//!
//! On the periodic SSF/Becke grid of [`crate::dft`] (construction A2, grid
//! rebuilt point-for-point with the energy's weights), `e(r) = ρ ε_xc`,
//! lattice-summed AOs `χ^Γ_μ = Σ_L χ_μ(r − R_A − L)` (so
//! `∂χ^Γ_μ/∂R_A = −∇χ^Γ_μ` for `μ` on `A`):
//!
//! ```text
//! a_{g,A} = −2 w_g Σ_σ Σ_{μ∈A} [ v_ρσ ∂_jχ_μ (D_σχ)_μ
//!                                + Σ_i c^σ_i (∂_j∂_iχ_μ (D_σχ)_μ + ∂_jχ_μ (D_σ∂_iχ)_μ) ]
//! c^α = 2 v_σαα ∇ρ_α + v_σαβ ∇ρ_β   (RKS: c = 2 v_σ ∇ρ on the total density)
//! AO term:        Σ_g a_{g,A}
//! point motion:   −Σ_{g home A} Σ_C a_{g,C}          (∇_r e = −Σ_C ∂e/∂R_C at fixed r)
//! weight term:    Σ_g e(r_g) d(w_rl P_home)/dR_A      (every image of A; the home point moves too)
//! ```
//!
//! The weight derivative is analytic over the IMAGE neighbour list
//! (`ferric_dft::becke::partition_weight_over_and_grad`: prefix/suffix
//! products because SSF has exact zeros), folded onto cell atoms, with
//! `∇_r w = −Σ_k ∂w/∂X_k` for the home point. With the full response the
//! force is translation invariant; without it `ΣF` is grid-error sized
//! (prototype 1e-4..3e-4 at (40, 50)). The response is ~2-3e-3 Ha/Bohr at
//! (40, 50) on the prototype cells: well above optimisation thresholds, so it
//! is not optional. Changes of the hard `D` neighbour mask under a
//! displacement are not differentiable (measure zero); they show up only as
//! FD noise of the ENERGY (1e-11..1e-9 Ha jumps on H₂ a = 4, FINDINGS
//! "Iteration 17"), never in the analytic force.
//!
//! The per-spin XC kernels and `V_σ` are ferric-dft's unchanged
//! `closed_kernel` / `polarized_kernel` / `semilocal_vxc_*` (the SCF's own),
//! with the same `ρ_σ > DENSITY_FLOOR` gates on the potential-like terms.
//!
//! # Consistency with the energy
//!
//! Every lattice sum uses exactly the energy's truncation: the `S`/`T`/`V_SR`
//! pair images, SR nucleus candidates and per-triplet screen come from the
//! same `hcore` helpers at `hcore_cfg.precision`; the `V_LR` G sphere is
//! `hcore::lr_gcut`; the ERI G sphere is `DenseAftEri::gcut`; the XC grid is
//! the SCF's `PeriodicGridConfig`. The G spheres do not depend on the
//! geometry, so the gradient is the derivative of the truncated energy up to
//! the primitive screens (`≤ precision`) and the AO extent threshold.
//!
//! libint2's `erf_nuclear`/`erfc_nuclear` derivative operators are NOT used
//! (the libint 2.7.2 bug of FINDINGS "Iteration 2" affects them as it does
//! the energy); the SR attraction goes through the ordinary erfc 3-centre
//! derivative engine on Gaussian nuclei.
//!
//! # RS-GDF J/K (the `*_rsgdf` entry points; FINDINGS "Iteration 18")
//!
//! With J/K from an [`RsGdf`] built by [`RsGdf::build_for_gradient`], the
//! `Σ Γ dI` line is replaced by the fitted-ERI derivative of
//! [`crate::rsgdf`]'s `deriv` module doc:
//!
//! ```text
//! Σ Γ dI^fit = Σ Y dJ3 + Σ Wm dJ2,     Y_P = D c_P − α Σ_σ D_σ C_P D_σ
//!   dJ3: SR erfc (orbital + aux centre), LR pair-FT derivative (orbital) and −iG X_P (aux)
//!   dJ2: SR erfc (d/dQ = −d/dP), LR −iG X_P;   Wm = Loewner form (kept–kept + kept–dropped)
//!   G = 0 of J3 (−c0' S q):  M_g0 = −c0' Σ_P Y_P q_P added to M,   c0' = π/(ω_gdf² Ω)
//! ```
//!
//! Everything else (1e, `V_SR`/`V_LR`, Ewald, `M` with the per-spin Madelung
//! term, `h`'s own `c0 Z_tot D`, XC) is the dense-AFT assembly unchanged.
//! The dense-AFT ERI has NO G = 0 term; the Iteration-16 Ewald-split ERI
//! M-term `c0(−N D + α Σ_σ D_σ S D_σ)` must NOT be used here: it is exact
//! only when the fitted pair charges equal the true ones, which the G = 0-
//! blind metric does not enforce (prototype: 0.7..10 × 1e-3 Ha/Bohr error
//! with a real aux basis, blind only in the exact-span limit — mutant
//! [`GradMutation::FitG0Dense`]). The aux-centre and metric derivatives are
//! each O(1e-2..1e-1) and cancel to O(fit error); both are required.
//! Aux centres that are functions of the atoms (ghost sites) fold through
//! [`RsGdfGradSource::aux_jac`].
//!
//! # ROHF / ROKS (the `gamma_ro{hf,ks}_gradient*` entry points; FINDINGS "Iteration 20")
//!
//! No new term. The ROHF energy is the UHF/UKS functional `E_U[D_α, D_β]`
//! restricted to ONE orthonormal orbital set `C = [C_c | C_o | C_v]`
//! (`D_α = C_c C_cᵀ + C_o C_oᵀ`, `D_β = C_c C_cᵀ`), so the force is the
//! unrestricted assembly above on `(D_α, D_β, F_α, F_β)` with the SPIN Focks
//! `F_σ` rebuilt from `D_σ` (`unrestricted_focks`, plus the POLARIZED
//! `V_σ` for ROKS — restricted orbitals do not make the density
//! unpolarized) and the orthonormality-Lagrangian
//!
//! ```text
//! W_RO = sym(D_α F_α D_α + D_α F_β D_β)          ([`rohf_lagrangian_w`])
//! W_RO − Σ_σ D_σ F_σ D_σ = ½ [C_o (f_β)_oc C_cᵀ + h.c.],   f_σ = Cᵀ F_σ C
//! ```
//!
//! The Lagrangian exists exactly when the three ROHF orbital-gradient blocks
//! vanish — `(f_α+f_β)_cv`, `(f_α)_ov`, `(f_β)_co` — and then `W_RO` IS the
//! UHF-form W (the difference is ½ × the closed–open β gradient; FINDINGS
//! measured ≤ 2.4e-12 in W, ≤ 1.1e-12 in F). A UHF-form-W "mutant" is
//! therefore an identity at convergence, not a defect
//! ([`GradMutation::RoWUhfForm`]). What IS wrong: `W` from the Roothaan
//! EFFECTIVE Fock's eigenpairs (`ScfResult::fock_alpha` IS `F_eff` for
//! ROHF — never use it here) and any `W` without the `(f_α)_co` block.
//! The force is first order in the orbital gradient, so an unconverged
//! result is refused on it ([`RO_ORBITAL_GRADIENT_TOL`]), not on ΔP/ΔE.
//!
//! # Out of scope (documented, not implemented)
//!
//! * **Stress** — lives in [`crate::stress`] (FINDINGS "Iteration 19"); it
//!   reuses this module's `SpinSet`, `JkSource` and input checks.
//! * ECPs (`v_ecp`), meta-GGA, range-separated hybrids, VV10, k-points:
//!   rejected or not provided.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::{DenseAftEri, ExxDiv};
use crate::dft::{
    build_response_grid, resolve_periodic_functional, GammaRksConfig, GammaUksConfig,
    LatticeAoHess, PeriodicGridConfig,
};
use crate::ewald::{
    default_ewald_omega, ewald_nuclear_gradient_parts, madelung_constant, DEFAULT_EWALD_PRECISION,
};
use crate::hcore::{
    gvector_list_bytes, half_gvectors, hcore_pair_images, lr_gcut, sr_attraction_gradient,
    PeriodicHcore, PeriodicHcoreConfig, G_CHUNK_BYTES, ONE_E_ENGINE_PRECISION,
};
use crate::lattice::Cell;
use crate::pair_ft::pair_ft_deriv_chunked;
use crate::rohf::GammaRoksConfig;
use crate::rsgdf::deriv::{check_aux_map, fit_densities, fit_derivatives, fold_aux, FitDensities};
use crate::rsgdf::{aux_ft, RsGdf, RsGdfFitDiagnostics};
use ferric_core::FerricError;
use ferric_dft::density_on_grid::{eval_density_closed, eval_density_uks};
use ferric_dft::grid::GridPoint;
use ferric_dft::libxc::{xc_def_from_name_nspin, FunctionalFamily};
use ferric_dft::vxc::{
    closed_kernel, polarized_kernel, semilocal_vxc_closed, semilocal_vxc_polarized, DENSITY_FLOOR,
};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::result::{ScfResult, Spin};
use ndarray::{s, Array2, Array3, Axis};
use num_complex::Complex64;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Deliberate defects for the mutation tests (`tests/pbc_grad.rs`,
/// `tests/pbc_grad_open.rs`): each must be caught by the finite-difference
/// anchor where the algebra says it is visible, and (measured in the
/// prototype) NONE of the first seven is caught by the sum of forces. Never
/// set in production.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradMutation {
    /// Drop the basis-centre motion in `V_ne` (SR and LR): only the nucleus
    /// moves.
    VneNoBasis,
    /// `+W` instead of `−W` in `M`.
    WSign,
    /// Drop the reciprocal-space part of the Ewald `E_nn` gradient.
    NoEwaldLr,
    /// Drop the Madelung `S` term from `M` entirely.
    NoMadelungS,
    /// `W = ½ D F̄ D`, `F̄ = (F_α + F_β)/2` (the RHF formula on the total
    /// density) instead of `Σ_σ D_σ F_σ D_σ`. Blind on closed shells.
    WSpinSum,
    /// Madelung `S` term `−(α v_M/2) D S D` (RHF form on the total density)
    /// instead of per spin. Blind on closed shells, `exxdiv = none`, `α = 0`.
    MadelungTotal,
    /// Exchange two-electron density `−(α/4) D ⊗ D` (RHF form) instead of
    /// per spin. Blind on closed shells and for `α = 0`.
    ExchTotal,
    /// Drop the XC AO-derivative term (the point-motion term still uses it).
    XcNoAo,
    /// Drop the GGA (`σ`) part of the XC AO term (AO and point motion).
    XcNoGga,
    /// Drop the grid-point motion term.
    NoPointMotion,
    /// Drop the partition-weight derivative term.
    NoWeightDeriv,
    /// RS-GDF: drop the metric derivative `Σ Wm dJ2` (SR and LR).
    FitNoMetric,
    /// RS-GDF: drop the aux-centre derivative of `J3` (SR and LR; the
    /// metric term is kept). The only fit mutant `ΣF` can see.
    FitNoAux,
    /// RS-GDF: drop the `J3` G = 0 term `M_g0`.
    FitNoG0,
    /// RS-GDF: the dense-AFT / Ewald-split ERI G = 0 M-term
    /// `c0(−N D + α Σ_σ D_σ S D_σ)` instead of `M_g0 = −c0 Σ_P Y_P q_P`.
    /// Blind in the exact-aux-span limit only.
    FitG0Dense,
    /// RS-GDF: drop the LR part of `dJ3` (orbital and aux centre).
    FitNoLr3,
    /// RS-GDF: textbook metric term (kept–kept block only, no kept–dropped
    /// Loewner block). Blind when nothing is dropped.
    FitTextbookMetric,
    /// ROHF/ROKS: `W` from the Roothaan EFFECTIVE Fock (`ScfResult::fock_alpha`)
    /// as the RHF-style `C diag(2ε_c, ε_o) Cᵀ` (`= 2 D_β F_eff D_β +
    /// P_o F_eff P_o`, `P_o = D_α − D_β`, at convergence).
    RoWFeffEps,
    /// ROHF/ROKS: `W_RO` without its closed–open `(f_α)_co` block
    /// (`− D_β F_α P_o − h.c.`). Invisible iff `(f_α)_co = 0`.
    RoWNoCo,
    /// ROHF/ROKS: the XC force pieces (AO term + grid response) through the
    /// RKS (unpolarized, total-density) path; the Focks keep the polarized
    /// `V_σ`.
    RoXcRksForm,
    /// ROHF/ROKS: the UHF-form `W = Σ_σ D_σ F_σ D_σ` instead of `W_RO`.
    /// NOT a defect: an IDENTITY at the ROHF stationary point (module doc,
    /// "ROHF / ROKS"); its force change is ~ the SCF residual. Exists only so
    /// a test can pin that identity. Never list it as a must-fail mutant.
    RoWUhfForm,
}

/// Settings for the `gamma_*_gradient_with` entry points.
#[derive(Debug, Clone, Copy, Default)]
pub struct GammaGradConfig {
    /// Memory budget (bytes); `None` = ferric's unified budget
    /// ([`crate::budget::resolve`]). The KS entries reserve the response
    /// grid, its weight derivatives and the per-chunk AO Hessians on it too
    /// (the SCF config's own grid/XC budgets are not read here).
    pub budget_bytes: Option<usize>,
    /// TEST ONLY: a deliberate defect ([`GradMutation`]).
    #[doc(hidden)]
    pub mutation: Option<GradMutation>,
    /// Gaussian-nucleus exponent used ONLY for the short-range attraction
    /// DERIVATIVE (`None` = [`GRAD_NUCLEUS_EXPONENT`], never tighter than the
    /// energy's `hcore_cfg.nucleus_exponent`). libint2's 3-centre derivative
    /// loses precision for very tight Gaussians (the error grows ~10-30x per
    /// decade above ~1e9), while below ~1e8 the mismatch with the energy's
    /// nucleus takes over. Measured max|analytic − FD| (Ha/Bohr, 2026-09-25):
    ///
    /// | exponent | H tri s+p RHF | LiH 6-31G RS-GDF | CO cc-pVDZ/jkfit |
    /// |---|---|---|---|
    /// | 1e7  | 1.05e-7 | —      | —       |
    /// | 1e8  | 7.2e-9  | 3.2e-9 | 1.35e-7 |
    /// | 1e9  | 6.4e-9  | 2.3e-8 | 1.19e-7 |
    /// | 1e10 | 1.85e-7 | 9.0e-8 | 8.85e-7 |
    /// | 1e11 | —       | 1.4e-6 | 1.28e-5 |
    ///
    /// (The triclinic cell also gave 8.3e-2 at 1e16 and 4.3e-6 at 1e12.)
    pub nucleus_exponent: Option<f64>,
}

/// Default exponent for the SR attraction derivative (see
/// [`GammaGradConfig::nucleus_exponent`]): the best single value for H and
/// within 2x of the best for Li, C and O in the scan above.
pub const GRAD_NUCLEUS_EXPONENT: f64 = 1e9;

/// Points per chunk of the XC gradient (13 AO planes + per-spin
/// intermediates live per chunk).
const XC_GRAD_CHUNK: usize = 128;

/// Per-term breakdown of the gradient (each `natoms × 3`, Hartree/Bohr).
#[derive(Debug, Clone)]
pub struct GammaGradParts {
    /// `Σ M dS` (Pulay, G = 0 bookkeeping, Madelung).
    pub overlap: Array2<f64>,
    /// `Σ D dT`.
    pub kinetic: Array2<f64>,
    /// `Σ D dV_SR`, basis-centre motion.
    pub vsr_basis: Array2<f64>,
    /// `Σ D dV_SR`, nucleus motion.
    pub vsr_nuc: Array2<f64>,
    /// `Σ D dV_LR`, basis-centre motion (pair-FT derivative).
    pub vlr_basis: Array2<f64>,
    /// `Σ D dV_LR`, nucleus motion (structure factor).
    pub vlr_nuc: Array2<f64>,
    /// `Σ Γ dI` (dense AFT J and K, pair-FT derivative).
    pub eri: Array2<f64>,
    /// Ewald `E_nn`, real-space part.
    pub nn_sr: Array2<f64>,
    /// Ewald `E_nn`, reciprocal-space part.
    pub nn_lr: Array2<f64>,
    /// XC AO-derivative term at fixed grid (zero for HF).
    pub xc_ao: Array2<f64>,
    /// XC grid response: grid points moving with their home atom.
    pub xc_point: Array2<f64>,
    /// XC grid response: partition-weight derivative.
    pub xc_weight: Array2<f64>,
    /// RS-GDF `Σ Y dJ3`, orbital-centre motion, SR (erfc lattice sums).
    /// Zero on the dense-AFT path (as are all `fit_*`).
    pub fit_orb_sr: Array2<f64>,
    /// RS-GDF `Σ Y dJ3`, orbital-centre motion, LR (pair-FT derivative).
    pub fit_orb_lr: Array2<f64>,
    /// RS-GDF `Σ Y dJ3`, aux-centre motion, SR.
    pub fit_aux_sr: Array2<f64>,
    /// RS-GDF `Σ Y dJ3`, aux-centre motion, LR (`−iG X_P`).
    pub fit_aux_lr: Array2<f64>,
    /// RS-GDF `Σ Wm dJ2`, SR.
    pub fit_metric_sr: Array2<f64>,
    /// RS-GDF `Σ Wm dJ2`, LR.
    pub fit_metric_lr: Array2<f64>,
    /// RS-GDF `J3` G = 0 term through `dS` (`Σ M_g0 dS`).
    pub fit_g0: Array2<f64>,
}

/// Output of the `gamma_*_gradient_with` entry points.
#[derive(Debug, Clone)]
pub struct GammaGradient {
    /// `dE/dR_A` per cell atom, `natoms × 3` (Hartree/Bohr, per cell).
    pub grad: Array2<f64>,
    /// Per-term breakdown (sums to `grad`).
    pub parts: GammaGradParts,
    /// `max_σ max |F_σ D_σ S − S D_σ F_σ|` of the Fock matrices rebuilt from
    /// the SCF densities (a gradient is only meaningful at a stationary `D`).
    /// ROHF/ROKS: the per-spin commutators do NOT vanish there, so this is
    /// instead [`RohfOrbitalGradient::max`] (the gated quantity).
    pub commutator: f64,
    /// `v_M` used in `K` and `M` (0 for `exxdiv = none`).
    pub madelung: f64,
    /// Exact-exchange fraction `α` (1 for HF).
    pub exact_exchange_fraction: f64,
    /// `E_xc` on the gradient's grid at the SCF density (`None` for HF).
    pub e_xc: Option<f64>,
    /// XC grid points (0 for HF).
    pub n_grid_points: usize,
    /// `max_x |Σ_A dE/dR_{A,x}|`. SANITY ONLY: translation invariance is
    /// blind to every non-XC mutation in [`GradMutation`] (prototype,
    /// measured).
    pub net_force: f64,
    /// Shifted 3-centre derivative calls in the SR attraction.
    pub n_sr_triplets: usize,
    /// Pair images summed for `dS`/`dT`.
    pub n_images: usize,
    /// Half-sphere G vectors in the `V_LR` derivative.
    pub n_g_lr: usize,
    /// Half-sphere G vectors in the ERI derivative.
    pub n_g_eri: usize,
    /// `pair_ft_deriv` chunks (`V_LR` + ERI).
    pub n_chunks: usize,
    /// Resolved memory budget (bytes).
    pub budget_bytes: usize,
    /// RS-GDF metric/cut diagnostics and derivative-walk counts (`None` on
    /// the dense-AFT path).
    pub fit: Option<RsGdfFitDiagnostics>,
}

/// The RS-GDF J/K a `*_rsgdf` gradient differentiates.
#[derive(Clone, Copy)]
pub struct RsGdfGradSource<'a> {
    /// Built with [`RsGdf::build_for_gradient`] (bitwise the B of
    /// [`RsGdf::build`]) from the SCF's lattice overlap `hc.s`, for the
    /// orbital basis the SCF used; the SCF must have run on this B.
    pub gdf: &'a RsGdf,
    /// The aux basis B was built with (spherical above l = 1).
    pub aux: &'a PreparedBasis,
    /// `dC_k/dR_A`, `(aux atoms, cell atoms)`, when the aux centres are
    /// functions of the atom positions (e.g. `SiteBasis` ghost sites; the
    /// same factor for x, y, z). `None`: the aux basis sits on exactly the
    /// cell's atoms (checked).
    pub aux_jac: Option<&'a Array2<f64>>,
}

/// Where J/K (and so `Σ Γ dI`) come from.
#[derive(Clone, Copy)]
pub(crate) enum JkSource<'a> {
    Dense(&'a DenseAftEri),
    Fit(RsGdfGradSource<'a>),
}

impl JkSource<'_> {
    pub(crate) fn build_j(
        &self,
        d: &Array2<f64>,
        out: &mut Array2<f64>,
    ) -> Result<(), FerricError> {
        match self {
            Self::Dense(e) => e.j_builder().build(d, out)?,
            Self::Fit(f) => f.gdf.j_builder().build(d, out)?,
        };
        Ok(())
    }

    pub(crate) fn build_k(
        &self,
        vm: f64,
        d: &Array2<f64>,
        out: &mut Array2<f64>,
    ) -> Result<(), FerricError> {
        match self {
            Self::Dense(e) => e.k_builder_with_madelung(vm).build(d, out)?,
            Self::Fit(f) => f.gdf.k_builder_with_madelung(vm).build(d, out)?,
        };
        Ok(())
    }
}

/// The Iteration-16 name of [`GammaGradient`].
pub type GammaRhfGradient = GammaGradient;

/// Per-spin densities and Fock matrices at the SCF solution.
pub(crate) enum SpinSet {
    /// `D` total, `F` (`D_α = D_β = D/2`, `F_α = F_β = F`).
    Restricted { d: Array2<f64>, f: Array2<f64> },
    /// `(D_α, D_β, F_α, F_β)`.
    Unrestricted {
        da: Array2<f64>,
        db: Array2<f64>,
        fa: Array2<f64>,
        fb: Array2<f64>,
    },
}

impl SpinSet {
    pub(crate) fn total(&self) -> Array2<f64> {
        match self {
            Self::Restricted { d, .. } => d.clone(),
            Self::Unrestricted { da, db, .. } => da + db,
        }
    }

    /// `Σ_σ D_σ X D_σ` (restricted: `½ D X D`).
    pub(crate) fn sandwich(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Restricted { d, .. } => d.dot(x).dot(d) * 0.5,
            Self::Unrestricted { da, db, .. } => da.dot(x).dot(da) + db.dot(x).dot(db),
        }
    }

    /// `W = Σ_σ D_σ F_σ D_σ` (restricted: `½ D F D`).
    pub(crate) fn energy_weighted(&self) -> Array2<f64> {
        match self {
            Self::Restricted { d, f } => 0.5 * d.dot(f).dot(d),
            Self::Unrestricted { da, db, fa, fb } => da.dot(fa).dot(da) + db.dot(fb).dot(db),
        }
    }

    /// `Σ_σ D_σ X D_σ` as `[(c, D)]` terms (restricted: `[(½, D)]`).
    pub(crate) fn exch_terms(&self) -> Vec<(f64, &Array2<f64>)> {
        match self {
            Self::Restricted { d, .. } => vec![(0.5, d)],
            Self::Unrestricted { da, db, .. } => vec![(1.0, da), (1.0, db)],
        }
    }

    /// MUTANT `W = ½ D F̄ D`.
    fn energy_weighted_spin_sum(&self) -> Array2<f64> {
        match self {
            Self::Restricted { d, f } => 0.5 * d.dot(f).dot(d),
            Self::Unrestricted { da, db, fa, fb } => {
                let d = da + db;
                let fbar = 0.5 * (fa + fb);
                0.5 * d.dot(&fbar).dot(&d)
            }
        }
    }

    /// `max_σ max |F_σ D_σ S − (F_σ D_σ S)ᵀ|`.
    pub(crate) fn commutator(&self, s_mat: &Array2<f64>) -> f64 {
        let one = |f: &Array2<f64>, d: &Array2<f64>| {
            let fds = f.dot(d).dot(s_mat);
            fds.iter()
                .zip(fds.t().iter())
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()))
        };
        match self {
            Self::Restricted { d, f } => one(f, d),
            Self::Unrestricted { da, db, fa, fb } => one(fa, da).max(one(fb, db)),
        }
    }
}

/// The XC pieces, computed before the Fock matrices are rebuilt (they need
/// `V_σ`).
struct XcGradient {
    e_xc: f64,
    /// `V_xc` (restricted) or `V_α`.
    v_a: Array2<f64>,
    /// `V_β` (unrestricted only).
    v_b: Option<Array2<f64>>,
    ao: Array2<f64>,
    point: Array2<f64>,
    weight: Array2<f64>,
    npts: usize,
}

/// The density handed to the XC gradient.
#[derive(Clone, Copy)]
enum XcDensity<'a> {
    Closed(&'a Array2<f64>),
    Polarized(&'a Array2<f64>, &'a Array2<f64>),
}

/// Gamma-point RHF nuclear gradient `dE/dR` (`natoms × 3`, per cell) with
/// default settings; see [`gamma_rhf_gradient_with`].
pub fn gamma_rhf_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    exxdiv: ExxDiv,
) -> Result<Array2<f64>, FerricError> {
    Ok(gamma_rhf_gradient_with(
        cell,
        prep,
        hcore_cfg,
        hc,
        eri,
        scf,
        exxdiv,
        &GammaGradConfig::default(),
    )?
    .grad)
}

/// Gamma-point RHF nuclear gradient with its per-term breakdown (module doc).
///
/// * `hcore_cfg`, `hc` — the config and output of
///   [`crate::hcore::periodic_hcore`] the SCF used (`hc.omega` must equal
///   `hcore_cfg.omega`; the SR/LR truncation is rebuilt from `hcore_cfg`).
/// * `eri` — the dense pure-AFT tensor the SCF used (its G sphere is reused;
///   its own `madelung` is ignored in favour of `exxdiv`).
/// * `scf` — the converged restricted result (`density_total`); `F` is
///   rebuilt from it, `W = ½ D F D`.
/// * `exxdiv` — the exchange-divergence treatment of the energy being
///   differentiated. At Gamma RHF it does not change `D`, and (with the
///   Madelung `S` term) not the force either.
#[allow(clippy::too_many_arguments)]
pub fn gamma_rhf_gradient_with(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    rhf_gradient(
        "gamma_rhf_gradient",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Dense(eri),
        scf,
        exxdiv,
        cfg,
    )
}

/// Gamma-point RHF nuclear gradient when J/K come from RS-GDF (module doc,
/// "RS-GDF J/K"): `scf` converged on `fit.gdf`'s J/K builders. Other
/// arguments as [`gamma_rhf_gradient_with`].
#[allow(clippy::too_many_arguments)]
pub fn gamma_rhf_gradient_rsgdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    fit: &RsGdfGradSource<'_>,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    rhf_gradient(
        "gamma_rhf_gradient_rsgdf",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Fit(*fit),
        scf,
        exxdiv,
        cfg,
    )
}

#[allow(clippy::too_many_arguments)]
fn rhf_gradient(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    jk: JkSource<'_>,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    check_inputs(who, cell, prep, hcore_cfg, hc, &jk, scf, Spin::Restricted)?;
    let mut ledger = open_ledger(cell, prep, cfg, false)?;
    let d = scf.density_total.clone();
    let vm = madelung_for(cell, exxdiv)?;
    let n = prep.nbasis();
    let mut jm = Array2::<f64>::zeros((n, n));
    let mut km = Array2::<f64>::zeros((n, n));
    jk.build_j(&d, &mut jm)?;
    jk.build_k(vm, &d, &mut km)?;
    let f = &(&hc.h + &jm) - &(0.5 * &km);
    let spins = SpinSet::Restricted { d, f };
    assemble(
        cell,
        prep,
        hcore_cfg,
        hc,
        &jk,
        &spins,
        1.0,
        vm,
        None,
        None,
        cfg,
        &mut ledger,
    )
}

/// Gamma-point UHF nuclear gradient `dE/dR` with default settings; see
/// [`gamma_uhf_gradient_with`].
pub fn gamma_uhf_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    exxdiv: ExxDiv,
) -> Result<Array2<f64>, FerricError> {
    Ok(gamma_uhf_gradient_with(
        cell,
        prep,
        hcore_cfg,
        hc,
        eri,
        scf,
        exxdiv,
        &GammaGradConfig::default(),
    )?
    .grad)
}

/// Gamma-point UHF nuclear gradient (module doc, `α = 1`, no XC) of a
/// converged unrestricted result (e.g. [`crate::uhf::gamma_uhf`]`.scf` on the
/// dense-AFT J/K). `F_σ = h + J[D] − K[D_σ] − v_M S D_σ S` is rebuilt from
/// `D_σ`; `W = Σ_σ D_σ F_σ D_σ`. `exxdiv` must be the convention of the
/// final SCF stage (the force does not depend on it; the Madelung `S` term
/// cancels per spin). Other arguments as [`gamma_rhf_gradient_with`].
#[allow(clippy::too_many_arguments)]
pub fn gamma_uhf_gradient_with(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    uhf_gradient(
        "gamma_uhf_gradient",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Dense(eri),
        scf,
        exxdiv,
        cfg,
    )
}

/// Gamma-point UHF nuclear gradient when J/K come from RS-GDF (`scf`, e.g.
/// [`crate::uhf::gamma_uhf`] with `GammaUhfIntegrals::RsGdf(fit.gdf)`).
/// Other arguments as [`gamma_uhf_gradient_with`].
#[allow(clippy::too_many_arguments)]
pub fn gamma_uhf_gradient_rsgdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    fit: &RsGdfGradSource<'_>,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    uhf_gradient(
        "gamma_uhf_gradient_rsgdf",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Fit(*fit),
        scf,
        exxdiv,
        cfg,
    )
}

#[allow(clippy::too_many_arguments)]
fn uhf_gradient(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    jk: JkSource<'_>,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    check_inputs(who, cell, prep, hcore_cfg, hc, &jk, scf, Spin::Unrestricted)?;
    let mut ledger = open_ledger(cell, prep, cfg, true)?;
    let (da, db) = spin_densities(who, scf)?;
    let vm = madelung_for(cell, exxdiv)?;
    let (fa, fb) = unrestricted_focks(hc, &jk, &da, &db, 1.0, vm, None)?;
    let spins = SpinSet::Unrestricted { da, db, fa, fb };
    assemble(
        cell,
        prep,
        hcore_cfg,
        hc,
        &jk,
        &spins,
        1.0,
        vm,
        None,
        None,
        cfg,
        &mut ledger,
    )
}

/// Gamma-point RKS nuclear gradient `dE/dR` with default settings; see
/// [`gamma_rks_gradient_with`].
#[allow(clippy::too_many_arguments)]
pub fn gamma_rks_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    dft: &GammaRksConfig,
) -> Result<Array2<f64>, FerricError> {
    Ok(gamma_rks_gradient_with(
        cell,
        prep,
        hcore_cfg,
        hc,
        eri,
        scf,
        dft,
        &GammaGradConfig::default(),
    )?
    .grad)
}

/// Gamma-point closed-shell KS-DFT nuclear gradient (LDA, GGA, global
/// hybrids) with the full periodic grid response (module doc).
///
/// * `scf` — the converged restricted result of [`crate::dft::gamma_rks`]
///   on the dense-AFT J/K `eri`.
/// * `dft` — the SAME config the SCF ran with: its `functional`, `grid`,
///   `xc.ao_threshold` and `exxdiv` define the energy being differentiated
///   (the grid is rebuilt point-for-point, with weight derivatives). Its
///   budgets and `scf` knobs are not read; the gradient's memory is
///   [`GammaGradConfig::budget_bytes`].
///
/// `F = h + J − (α/2)(K[D] + v_M S D S) + V_xc` is rebuilt from `D`.
#[allow(clippy::too_many_arguments)]
pub fn gamma_rks_gradient_with(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    dft: &GammaRksConfig,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    rks_gradient(
        "gamma_rks_gradient",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Dense(eri),
        scf,
        dft,
        cfg,
    )
}

/// Gamma-point RKS nuclear gradient when J/K come from RS-GDF (`scf` from
/// [`crate::dft::gamma_rks`] with `GammaUhfIntegrals::RsGdf(fit.gdf)`).
/// Other arguments as [`gamma_rks_gradient_with`].
#[allow(clippy::too_many_arguments)]
pub fn gamma_rks_gradient_rsgdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    fit: &RsGdfGradSource<'_>,
    scf: &ScfResult,
    dft: &GammaRksConfig,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    rks_gradient(
        "gamma_rks_gradient_rsgdf",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Fit(*fit),
        scf,
        dft,
        cfg,
    )
}

#[allow(clippy::too_many_arguments)]
fn rks_gradient(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    jk: JkSource<'_>,
    scf: &ScfResult,
    dft: &GammaRksConfig,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    check_inputs(who, cell, prep, hcore_cfg, hc, &jk, scf, Spin::Restricted)?;
    let (_, alpha) = resolve_periodic_functional(&dft.functional)?;
    let mut ledger = open_ledger(cell, prep, cfg, false)?;
    let d = scf.density_total.clone();
    let vm = madelung_for(cell, dft.exxdiv)?;
    let aoat = ao_atoms(prep);
    let xg = xc_gradient(
        cell,
        prep,
        &dft.functional,
        &dft.grid,
        dft.xc.ao_threshold,
        XcDensity::Closed(&d),
        &aoat,
        cfg.mutation,
        &mut ledger,
    )?;
    let n = prep.nbasis();
    let mut jm = Array2::<f64>::zeros((n, n));
    jk.build_j(&d, &mut jm)?;
    let mut f = &(&hc.h + &jm) + &xg.v_a;
    if alpha != 0.0 {
        let mut km = Array2::<f64>::zeros((n, n));
        jk.build_k(vm, &d, &mut km)?;
        f.scaled_add(-0.5 * alpha, &km);
    }
    let spins = SpinSet::Restricted { d, f };
    assemble(
        cell,
        prep,
        hcore_cfg,
        hc,
        &jk,
        &spins,
        alpha,
        vm,
        Some(xg),
        None,
        cfg,
        &mut ledger,
    )
}

/// Gamma-point UKS nuclear gradient `dE/dR` with default settings; see
/// [`gamma_uks_gradient_with`].
#[allow(clippy::too_many_arguments)]
pub fn gamma_uks_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    dft: &GammaUksConfig,
) -> Result<Array2<f64>, FerricError> {
    Ok(gamma_uks_gradient_with(
        cell,
        prep,
        hcore_cfg,
        hc,
        eri,
        scf,
        dft,
        &GammaGradConfig::default(),
    )?
    .grad)
}

/// Gamma-point open-shell (or closed-shell) UKS nuclear gradient with the
/// full periodic grid response (module doc). `scf` is the final stage of
/// [`crate::dft::gamma_uks`] on the dense-AFT J/K `eri`; `dft` is the SAME
/// config the SCF ran with (see [`gamma_rks_gradient_with`]).
/// `F_σ = h + J[D] − α (K[D_σ] + v_M S D_σ S) + V_σ` is rebuilt from `D_σ`.
#[allow(clippy::too_many_arguments)]
pub fn gamma_uks_gradient_with(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    dft: &GammaUksConfig,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    uks_gradient(
        "gamma_uks_gradient",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Dense(eri),
        scf,
        dft,
        cfg,
    )
}

/// Gamma-point UKS nuclear gradient when J/K come from RS-GDF (`scf` the
/// final stage of [`crate::dft::gamma_uks`] with
/// `GammaUhfIntegrals::RsGdf(fit.gdf)`). Other arguments as
/// [`gamma_uks_gradient_with`].
#[allow(clippy::too_many_arguments)]
pub fn gamma_uks_gradient_rsgdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    fit: &RsGdfGradSource<'_>,
    scf: &ScfResult,
    dft: &GammaUksConfig,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    uks_gradient(
        "gamma_uks_gradient_rsgdf",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Fit(*fit),
        scf,
        dft,
        cfg,
    )
}

#[allow(clippy::too_many_arguments)]
fn uks_gradient(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    jk: JkSource<'_>,
    scf: &ScfResult,
    dft: &GammaUksConfig,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    check_inputs(who, cell, prep, hcore_cfg, hc, &jk, scf, Spin::Unrestricted)?;
    let (_, alpha) = resolve_periodic_functional(&dft.functional)?;
    let mut ledger = open_ledger(cell, prep, cfg, true)?;
    let (da, db) = spin_densities(who, scf)?;
    let vm = madelung_for(cell, dft.exxdiv)?;
    let aoat = ao_atoms(prep);
    let mut xg = xc_gradient(
        cell,
        prep,
        &dft.functional,
        &dft.grid,
        dft.xc.ao_threshold,
        XcDensity::Polarized(&da, &db),
        &aoat,
        cfg.mutation,
        &mut ledger,
    )?;
    let v_b = xg.v_b.take().ok_or_else(|| {
        FerricError::General(format!("{who}: the polarized XC pass returned no V_beta"))
    })?;
    let (fa, fb) = unrestricted_focks(hc, &jk, &da, &db, alpha, vm, Some((&xg.v_a, &v_b)))?;
    let spins = SpinSet::Unrestricted { da, db, fa, fb };
    assemble(
        cell,
        prep,
        hcore_cfg,
        hc,
        &jk,
        &spins,
        alpha,
        vm,
        Some(xg),
        None,
        cfg,
        &mut ledger,
    )
}

/// Refusal threshold on the ROHF orbital gradient ([`RohfOrbitalGradient::max`],
/// Hartree) of the `gamma_ro*_gradient*` entry points.
///
/// Why this value: ferric's ROHF/ROKS stops on `ΔP_rms < density_conv` (and
/// `ΔP_max < 10 density_conv`), not on the orbital gradient `g`. Near the
/// solution one Roothaan step moves the orbitals by `~ g / Δε` (`Δε` the
/// relevant level gap, 0.1..1 Ha on the cells measured), so `|g| ≈ Δε · ΔP`:
/// ≲ 1e-9 at the Gamma drivers' default `density_conv = 1e-10`, and ≲ 1e-6
/// even at a loose 1e-6. The force error is first order in `g` (~ `|g|` ×
/// O(1) AO-derivative norms), so 1e-5 admits every result converged at
/// `density_conv ≤ 1e-6` and caps the resulting force error near 1e-5
/// Ha/Bohr (below geometry-optimizer thresholds, ~4.5e-4), while refusing
/// max-iter snapshots and wrong-state results (the chaotic ROKS PBE0 wander
/// of `crate::rohf` sits at `|g|` ≫ 1e-5). It is a refusal of a
/// non-stationary state, NOT a precision guarantee: for FD-grade forces
/// converge tightly (the tests run at `density_conv = 1e-10`).
pub const RO_ORBITAL_GRADIENT_TOL: f64 = 1e-5;

/// Occupation tolerance for classifying the ROHF MOs from `(D_α, D_β)`.
const RO_OCC_TOL: f64 = 1e-6;

/// The three ROHF orbital-gradient blocks (max |element|, MO basis,
/// `f_σ = Cᵀ F_σ C` with the SPIN Focks) and the orbital classes they were
/// taken over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RohfOrbitalGradient {
    /// `max |(f_β)_co|` (closed → open).
    pub closed_open: f64,
    /// `max |(f_α)_ov|` (open → virtual).
    pub open_virtual: f64,
    /// `max |(f_α + f_β)_cv|` (closed → virtual).
    pub closed_virtual: f64,
    /// `max |(f_α)_co|`: NOT a gradient block — the closed–open coupling
    /// `W_RO` carries (the size of what [`GradMutation::RoWNoCo`] drops).
    pub alpha_closed_open: f64,
    /// Closed (doubly occupied) MOs.
    pub n_closed: usize,
    /// Open (α-only) MOs.
    pub n_open: usize,
}

impl RohfOrbitalGradient {
    /// Largest of the three gradient blocks.
    pub fn max(&self) -> f64 {
        self.closed_open
            .max(self.open_virtual)
            .max(self.closed_virtual)
    }
}

/// ROHF orbital gradient of the MOs `c` (columns; all-zero columns are
/// skipped) at the spin densities / SPIN Focks `(D_σ, F_σ)` in the metric
/// `s`. Each MO is classified by its occupations `n_i^σ = c_iᵀ S D_σ S c_i`
/// (closed (1, 1), open (1, 0), virtual (0, 0)); errors when an occupation
/// is not 0/1 within 1e-6, a MO is β-only, or the class counts disagree with
/// `tr(D_σ S)` — i.e. when `c` is not the orbital set of `(D_α, D_β)`.
pub fn rohf_orbital_gradient(
    c: &Array2<f64>,
    s: &Array2<f64>,
    da: &Array2<f64>,
    db: &Array2<f64>,
    fa: &Array2<f64>,
    fb: &Array2<f64>,
) -> Result<RohfOrbitalGradient, FerricError> {
    let n = s.nrows();
    if c.nrows() != n || [da, db, fa, fb].iter().any(|m| m.dim() != (n, n)) {
        return Err(FerricError::General(format!(
            "rohf_orbital_gradient: shape mismatch (S {:?}, C {:?})",
            s.dim(),
            c.dim()
        )));
    }
    let occ = |d: &Array2<f64>| -> Vec<f64> {
        let sc = s.dot(c);
        let m = sc.t().dot(d).dot(&sc);
        (0..c.ncols()).map(|i| m[(i, i)]).collect()
    };
    let (occ_a, occ_b) = (occ(da), occ(db));
    let (mut closed, mut open, mut virt) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..c.ncols() {
        if c.column(i).iter().all(|v| *v == 0.0) {
            continue;
        }
        let (na, nb) = (occ_a[i], occ_b[i]);
        let (ra, rb) = (na.round(), nb.round());
        let ok = |x: f64, r: f64| (x - r).abs() < RO_OCC_TOL && (r == 0.0 || r == 1.0);
        if !ok(na, ra) || !ok(nb, rb) {
            return Err(FerricError::General(format!(
                "rohf_orbital_gradient: MO {i} has occupations (α {na:.3e}, β {nb:.3e}), not 0/1: \
                 the MOs are not the orbital set of (D_α, D_β)"
            )));
        }
        match (ra == 1.0, rb == 1.0) {
            (true, true) => closed.push(i),
            (true, false) => open.push(i),
            (false, false) => virt.push(i),
            (false, true) => {
                return Err(FerricError::General(format!(
                    "rohf_orbital_gradient: MO {i} is β-occupied but α-empty (not a ROHF state)"
                )))
            }
        }
    }
    let ne_a = (da * s).sum();
    let ne_b = (db * s).sum();
    if (ne_a - (closed.len() + open.len()) as f64).abs() > 1e-6
        || (ne_b - closed.len() as f64).abs() > 1e-6
    {
        return Err(FerricError::General(format!(
            "rohf_orbital_gradient: {} closed + {} open MOs do not carry tr(D_α S) = {ne_a:.8}, \
             tr(D_β S) = {ne_b:.8}",
            closed.len(),
            open.len()
        )));
    }
    let fa_mo = c.t().dot(fa).dot(c);
    let fb_mo = c.t().dot(fb).dot(c);
    let block = |rows: &[usize], cols: &[usize], f: &dyn Fn(usize, usize) -> f64| {
        // NaN-propagating max (f64::max would hide a NaN Fock).
        let mut m = 0.0_f64;
        for &p in rows {
            for &q in cols {
                let a = f(p, q).abs();
                if a.is_nan() || a > m {
                    m = a;
                }
            }
        }
        m
    };
    Ok(RohfOrbitalGradient {
        closed_open: block(&closed, &open, &|p: usize, q: usize| fb_mo[(p, q)]),
        open_virtual: block(&open, &virt, &|p: usize, q: usize| fa_mo[(p, q)]),
        closed_virtual: block(&closed, &virt, &|p: usize, q: usize| {
            fa_mo[(p, q)] + fb_mo[(p, q)]
        }),
        alpha_closed_open: block(&closed, &open, &|p: usize, q: usize| fa_mo[(p, q)]),
        n_closed: closed.len(),
        n_open: open.len(),
    })
}

/// The ROHF orthonormality-Lagrangian energy-weighted matrix
/// `W_RO = sym(D_α F_α D_α + D_α F_β D_β)` (SPIN Focks; module doc "ROHF /
/// ROKS"; ferric-scf's molecular `rohf_energy_weighted_density`). Equal to
/// `Σ_σ D_σ F_σ D_σ` up to ½ × the closed–open β orbital gradient.
pub fn rohf_lagrangian_w(
    da: &Array2<f64>,
    db: &Array2<f64>,
    fa: &Array2<f64>,
    fb: &Array2<f64>,
) -> Array2<f64> {
    let w = da.dot(fa).dot(da) + da.dot(fb).dot(db);
    0.5 * (&w + &w.t())
}

/// Gamma-point ROHF nuclear gradient `dE/dR` with default settings; see
/// [`gamma_rohf_gradient_with`].
pub fn gamma_rohf_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    exxdiv: ExxDiv,
) -> Result<Array2<f64>, FerricError> {
    Ok(gamma_rohf_gradient_with(
        cell,
        prep,
        hcore_cfg,
        hc,
        eri,
        scf,
        exxdiv,
        &GammaGradConfig::default(),
    )?
    .grad)
}

/// Gamma-point ROHF nuclear gradient (module doc, "ROHF / ROKS") of a
/// converged restricted-open result (e.g. [`crate::rohf::gamma_rohf`]`.scf`
/// on the dense-AFT J/K). `D_σ` from `density_alpha`/`density_beta`, the
/// SPIN Focks `F_σ = h + J[D] − K[D_σ] − v_M S D_σ S` rebuilt from them
/// (`scf.fock_alpha`, the Roothaan `F_eff`, is not read), `W = W_RO`.
/// Refuses a result whose ROHF orbital gradient exceeds
/// [`RO_ORBITAL_GRADIENT_TOL`]; `GammaGradient::commutator` reports it.
/// Other arguments as [`gamma_uhf_gradient_with`].
#[allow(clippy::too_many_arguments)]
pub fn gamma_rohf_gradient_with(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    ro_gradient(
        "gamma_rohf_gradient",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Dense(eri),
        scf,
        RoKind::Hf(exxdiv),
        cfg,
    )
}

/// Gamma-point ROHF nuclear gradient when J/K come from RS-GDF (`scf` from
/// [`crate::rohf::gamma_rohf`] with `GammaUhfIntegrals::RsGdf(fit.gdf)`).
/// Other arguments as [`gamma_rohf_gradient_with`].
#[allow(clippy::too_many_arguments)]
pub fn gamma_rohf_gradient_rsgdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    fit: &RsGdfGradSource<'_>,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    ro_gradient(
        "gamma_rohf_gradient_rsgdf",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Fit(*fit),
        scf,
        RoKind::Hf(exxdiv),
        cfg,
    )
}

/// Gamma-point ROKS nuclear gradient `dE/dR` with default settings; see
/// [`gamma_roks_gradient_with`].
#[allow(clippy::too_many_arguments)]
pub fn gamma_roks_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    dft: &GammaRoksConfig,
) -> Result<Array2<f64>, FerricError> {
    Ok(gamma_roks_gradient_with(
        cell,
        prep,
        hcore_cfg,
        hc,
        eri,
        scf,
        dft,
        &GammaGradConfig::default(),
    )?
    .grad)
}

/// Gamma-point ROKS nuclear gradient (LDA, GGA, global hybrids) with the
/// full periodic grid response: `scf` the final stage of
/// [`crate::rohf::gamma_roks`] on the dense-AFT J/K `eri`, `dft` the SAME
/// config it ran with (functional, grid, AO threshold, exxdiv; see
/// [`gamma_rks_gradient_with`]). The UKS assembly on the ROHF spin densities:
/// POLARIZED XC kernel and response, `F_σ = h + J[D] − α (K[D_σ] +
/// v_M S D_σ S) + V_σ`, `W = W_RO`. Refuses as [`gamma_rohf_gradient_with`].
#[allow(clippy::too_many_arguments)]
pub fn gamma_roks_gradient_with(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    dft: &GammaRoksConfig,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    ro_gradient(
        "gamma_roks_gradient",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Dense(eri),
        scf,
        RoKind::Ks(dft),
        cfg,
    )
}

/// Gamma-point ROKS nuclear gradient when J/K come from RS-GDF (`scf` from
/// [`crate::rohf::gamma_roks`] with `GammaUhfIntegrals::RsGdf(fit.gdf)`).
/// Other arguments as [`gamma_roks_gradient_with`].
#[allow(clippy::too_many_arguments)]
pub fn gamma_roks_gradient_rsgdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    fit: &RsGdfGradSource<'_>,
    scf: &ScfResult,
    dft: &GammaRoksConfig,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    ro_gradient(
        "gamma_roks_gradient_rsgdf",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Fit(*fit),
        scf,
        RoKind::Ks(dft),
        cfg,
    )
}

/// ROHF (`α = 1`, no XC) or ROKS (the config's functional).
#[derive(Clone, Copy)]
enum RoKind<'a> {
    Hf(ExxDiv),
    Ks(&'a GammaRoksConfig),
}

/// The ROHF orbital gradient of `scf.mos_alpha` at `(D_σ, F_σ)`, refused
/// above [`RO_ORBITAL_GRADIENT_TOL`] (messages prefixed by `who`).
pub(crate) fn ro_gate(
    who: &str,
    scf: &ScfResult,
    s: &Array2<f64>,
    da: &Array2<f64>,
    db: &Array2<f64>,
    fa: &Array2<f64>,
    fb: &Array2<f64>,
) -> Result<RohfOrbitalGradient, FerricError> {
    let og = rohf_orbital_gradient(&scf.mos_alpha, s, da, db, fa, fb)
        .map_err(|e| FerricError::General(format!("{who}: {e}")))?;
    if og.max().is_nan() || og.max() > RO_ORBITAL_GRADIENT_TOL {
        return Err(FerricError::General(format!(
            "{who}: ROHF orbital gradient {:.3e} (co {:.3e}, ov {:.3e}, cv {:.3e}) exceeds \
             {RO_ORBITAL_GRADIENT_TOL:.0e}: not a stationary ROHF state (the force is first \
             order in it); converge the SCF tighter",
            og.max(),
            og.closed_open,
            og.open_virtual,
            og.closed_virtual
        )));
    }
    Ok(og)
}

#[allow(clippy::too_many_arguments)]
fn ro_gradient(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    jk: JkSource<'_>,
    scf: &ScfResult,
    kind: RoKind<'_>,
    cfg: &GammaGradConfig,
) -> Result<GammaGradient, FerricError> {
    check_inputs(
        who,
        cell,
        prep,
        hcore_cfg,
        hc,
        &jk,
        scf,
        Spin::RestrictedOpen,
    )?;
    let (alpha, exxdiv) = match kind {
        RoKind::Hf(e) => (1.0, e),
        RoKind::Ks(d) => (resolve_periodic_functional(&d.functional)?.1, d.exxdiv),
    };
    let mut ledger = open_ledger(cell, prep, cfg, true)?;
    let (da, db) = spin_densities(who, scf)?;
    let vm = madelung_for(cell, exxdiv)?;
    let (xg, v_b) = match kind {
        RoKind::Hf(_) => (None, None),
        RoKind::Ks(dft) => {
            let aoat = ao_atoms(prep);
            let mut xg = xc_gradient(
                cell,
                prep,
                &dft.functional,
                &dft.grid,
                dft.xc.ao_threshold,
                XcDensity::Polarized(&da, &db),
                &aoat,
                cfg.mutation,
                &mut ledger,
            )?;
            let v_b = xg.v_b.take().ok_or_else(|| {
                FerricError::General(format!("{who}: the polarized XC pass returned no V_beta"))
            })?;
            if cfg.mutation == Some(GradMutation::RoXcRksForm) {
                let dt = &da + &db;
                let xr = xc_gradient(
                    cell,
                    prep,
                    &dft.functional,
                    &dft.grid,
                    dft.xc.ao_threshold,
                    XcDensity::Closed(&dt),
                    &aoat,
                    cfg.mutation,
                    &mut ledger,
                )?;
                xg.ao = xr.ao;
                xg.point = xr.point;
                xg.weight = xr.weight;
            }
            (Some(xg), Some(v_b))
        }
    };
    let v = match (&xg, &v_b) {
        (Some(x), Some(vb)) => Some((&x.v_a, vb)),
        _ => None,
    };
    let (fa, fb) = unrestricted_focks(hc, &jk, &da, &db, alpha, vm, v)?;
    let og = ro_gate(who, scf, &hc.s, &da, &db, &fa, &fb)?;
    let w = match cfg.mutation {
        Some(GradMutation::RoWUhfForm) => None,
        Some(GradMutation::RoWFeffEps) => {
            let feff = &scf.fock_alpha;
            if feff.dim() != da.dim() {
                return Err(FerricError::General(format!(
                    "{who}: RoWFeffEps needs the n×n Roothaan F_eff in scf.fock_alpha"
                )));
            }
            let po = &da - &db;
            Some(2.0 * db.dot(feff).dot(&db) + po.dot(feff).dot(&po))
        }
        Some(GradMutation::RoWNoCo) => {
            let po = &da - &db;
            let cpl = db.dot(&fa).dot(&po);
            Some(rohf_lagrangian_w(&da, &db, &fa, &fb) - &cpl - cpl.t())
        }
        _ => Some(rohf_lagrangian_w(&da, &db, &fa, &fb)),
    };
    let spins = SpinSet::Unrestricted { da, db, fa, fb };
    let mut out = assemble(
        cell,
        prep,
        hcore_cfg,
        hc,
        &jk,
        &spins,
        alpha,
        vm,
        xg,
        w,
        cfg,
        &mut ledger,
    )?;
    out.commutator = og.max();
    Ok(out)
}

/// Shared argument checks (messages prefixed by `who`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_inputs(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    jk: &JkSource<'_>,
    scf: &ScfResult,
    spin: Spin,
) -> Result<(), FerricError> {
    let n = prep.nbasis();
    if hc.omega != hcore_cfg.omega {
        return Err(FerricError::General(format!(
            "{who}: hcore was built at omega = {} but hcore_cfg.omega = {}",
            hc.omega, hcore_cfg.omega
        )));
    }
    if hc.v_ecp.is_some() {
        return Err(FerricError::General(format!(
            "{who}: periodic ECP gradients are not implemented"
        )));
    }
    if scf.spin != spin {
        let want = match spin {
            Spin::Restricted => "a restricted closed-shell",
            Spin::RestrictedOpen => "a restricted open-shell (ROHF/ROKS)",
            Spin::Unrestricted => "an unrestricted",
        };
        return Err(FerricError::General(format!(
            "{who}: needs {want} result, got {:?}",
            scf.spin
        )));
    }
    if !scf.converged {
        return Err(FerricError::General(format!(
            "{who}: the SCF did not converge; a gradient needs a stationary density"
        )));
    }
    let d_ok = match spin {
        Spin::Restricted => scf.density_total.dim() == (n, n),
        _ => {
            scf.density_alpha.dim() == (n, n)
                && scf.density_beta.as_ref().map(|d| d.dim()) == Some((n, n))
        }
    };
    let (jk_ok, jk_dim) = jk_shape(jk, n);
    if hc.s.dim() != (n, n) || !d_ok || !jk_ok {
        return Err(FerricError::General(format!(
            "{who}: shape mismatch (nbasis {n}: S {:?}, D {:?}, ERI/B {jk_dim:?})",
            hc.s.dim(),
            scf.density_total.dim(),
        )));
    }
    if let JkSource::Fit(f) = jk {
        check_rsgdf_inputs(who, cell, hc, f, n)?;
    }
    Ok(())
}

/// `(shape ok, shape)` of the dense ERI (`n² × n²`) or the fitted B
/// (`naux × n²`).
fn jk_shape(jk: &JkSource<'_>, n: usize) -> (bool, (usize, usize)) {
    match jk {
        JkSource::Dense(eri) => (eri.eri().dim() == (n * n, n * n), eri.eri().dim()),
        JkSource::Fit(f) => (f.gdf.b().ncols() == n * n, f.gdf.b().dim()),
    }
}

/// The RS-GDF-only argument checks of [`check_inputs`] (messages prefixed
/// by `who`).
fn check_rsgdf_inputs(
    who: &str,
    cell: &Cell,
    hc: &PeriodicHcore,
    f: &RsGdfGradSource<'_>,
    n: usize,
) -> Result<(), FerricError> {
    if !f.gdf.has_gradient_parts() {
        return Err(FerricError::General(format!(
            "{who}: the RsGdf carries no gradient parts; build it with RsGdf::build_for_gradient"
        )));
    }
    if f.aux.nbasis() != f.gdf.stats().naux {
        return Err(FerricError::General(format!(
            "{who}: aux basis has {} functions but B was built with {}",
            f.aux.nbasis(),
            f.gdf.stats().naux
        )));
    }
    // J3's G = 0 term is c0 S q with B's S: it must be the S whose
    // derivative the overlap term contracts (hc.s).
    let smax = hc.s.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    let ds = f
        .gdf
        .overlap()
        .iter()
        .zip(hc.s.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    if f.gdf.overlap().dim() != (n, n) || ds > 1e-12 * smax {
        return Err(FerricError::General(format!(
            "{who}: the RsGdf was built with a different overlap than hc.s \
             (max |ΔS| = {ds:.3e}); build it from the SCF's hcore"
        )));
    }
    check_aux_map(cell, f.aux, f.aux_jac)
}

/// Ledger with the n×n and per-term reservations of every entry point.
fn open_ledger(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &GammaGradConfig,
    unrestricted: bool,
) -> Result<Ledger, FerricError> {
    let n = prep.nbasis();
    let natoms = cell.positions().len();
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    // D, J, K, F, W, M, DSD, SD + per-G complex scratch (ρ-weighted Z,
    // D P D in re/im, P in re/im: ~8 real n²); per spin D_σ, F_σ, V_σ and
    // the per-spin sandwiches double part of it.
    let per = if unrestricted { 8 * 28 } else { 8 * 16 };
    ledger.reserve(
        &format!("gamma gradient n×n matrices + per-G scratch (n = {n})"),
        bytes_of((n * n) as u64, per),
    )?;
    ledger.reserve(
        &format!("gamma gradient per-term arrays (natoms = {natoms})"),
        bytes_of((natoms * 3) as u64, 8 * 24),
    )?;
    Ok(ledger)
}

pub(crate) fn madelung_for(cell: &Cell, exxdiv: ExxDiv) -> Result<f64, FerricError> {
    Ok(match exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => madelung_constant(cell)?,
    })
}

pub(crate) fn spin_densities(
    who: &str,
    scf: &ScfResult,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let db = scf
        .density_beta
        .clone()
        .ok_or_else(|| FerricError::General(format!("{who}: the SCF has no beta density")))?;
    Ok((scf.density_alpha.clone(), db))
}

/// `F_σ = h + J[D_α + D_β] − α (K[D_σ] + v_M S D_σ S) + V_σ`.
pub(crate) fn unrestricted_focks(
    hc: &PeriodicHcore,
    jk: &JkSource<'_>,
    da: &Array2<f64>,
    db: &Array2<f64>,
    alpha: f64,
    vm: f64,
    v: Option<(&Array2<f64>, &Array2<f64>)>,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let n = da.nrows();
    let d = da + db;
    let mut jm = Array2::<f64>::zeros((n, n));
    jk.build_j(&d, &mut jm)?;
    let base = &hc.h + &jm;
    let mut fa = base.clone();
    let mut fb = base;
    if alpha != 0.0 {
        let mut ka = Array2::<f64>::zeros((n, n));
        let mut kb = Array2::<f64>::zeros((n, n));
        jk.build_k(vm, da, &mut ka)?;
        jk.build_k(vm, db, &mut kb)?;
        fa.scaled_add(-alpha, &ka);
        fb.scaled_add(-alpha, &kb);
    }
    if let Some((va, vb)) = v {
        fa += va;
        fb += vb;
    }
    Ok((fa, fb))
}

/// AO → cell atom.
pub(crate) fn ao_atoms(prep: &PreparedBasis) -> Vec<usize> {
    let sh2at = prep.shell_to_atom();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut aoat = vec![0usize; prep.nbasis()];
    for sh in 0..prep.nshells() {
        for k in 0..dims[sh] {
            aoat[offs[sh] + k] = sh2at[sh];
        }
    }
    aoat
}

/// Everything but the XC pass: one-electron, nuclear, two-electron and the
/// overlap-coupled terms at the given per-spin `(D_σ, F_σ)` (module doc).
/// `w`: the energy-weighted matrix when it is not `spins`' own
/// `Σ_σ D_σ F_σ D_σ` (ROHF/ROKS: `W_RO`); `None` everywhere else.
#[allow(clippy::too_many_arguments)]
fn assemble(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    jk: &JkSource<'_>,
    spins: &SpinSet,
    alpha: f64,
    vm: f64,
    xc: Option<XcGradient>,
    w: Option<Array2<f64>>,
    cfg: &GammaGradConfig,
    ledger: &mut Ledger,
) -> Result<GammaGradient, FerricError> {
    let n = prep.nbasis();
    let natoms = cell.positions().len();
    let mutation = cfg.mutation;

    // --- Total density, W, M.
    let d = spins.total();
    let s_mat = &hc.s;
    let commutator = spins.commutator(s_mat);
    let w = if mutation == Some(GradMutation::WSpinSum) {
        spins.energy_weighted_spin_sum()
    } else if let Some(w) = w {
        w
    } else {
        spins.energy_weighted()
    };
    let zs = cell.nuclear_charges();
    let ztot: f64 = zs.iter().sum();
    let omega = hcore_cfg.omega;
    let vol = cell.volume();
    let c0 = PI / (omega * omega * vol);
    let mut m = if mutation == Some(GradMutation::WSign) {
        w.clone()
    } else {
        -&w
    };
    m.scaled_add(c0 * ztot, &d);
    match mutation {
        Some(GradMutation::NoMadelungS) => {}
        Some(GradMutation::MadelungTotal) => {
            let dsd = d.dot(s_mat).dot(&d);
            m.scaled_add(-0.5 * alpha * vm, &dsd);
        }
        _ => {
            if alpha != 0.0 && vm != 0.0 {
                let dsd = spins.sandwich(s_mat);
                m.scaled_add(-alpha * vm, &dsd);
            }
        }
    }

    // --- RS-GDF: the fitted densities Y, Wm first — J3's G = 0 term enters
    // only through dS, as M_g0 = −c0' Σ_P Y_P q_P (c0' at the GDF's ω).
    let fit = fit_prelude(jk, spins, &d, s_mat, alpha, vol, mutation, ledger)?;
    let m_g0 = match &fit {
        Some((_, _, Some(mg))) => Some(mg),
        _ => None,
    };

    // --- dS, dT over the energy's pair images (bra and ket blocks).
    let images = hcore_pair_images(cell, prep, hcore_cfg.precision, ledger)?;
    let sh2at = prep.shell_to_atom().to_vec();
    let dims = prep.shell_dims().to_vec();
    let offs = prep.shell_offsets().to_vec();
    let nsh = prep.nshells();
    let mut g_s = Array2::<f64>::zeros((natoms, 3));
    let mut g_t = Array2::<f64>::zeros((natoms, 3));
    let mut g_fit_g0 = Array2::<f64>::zeros((natoms, 3));
    {
        let mut eng_s = Engine::new_1e_deriv(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
        let mut eng_t = Engine::new_1e_deriv(ffi::OP_KINETIC, prep, ONE_E_ENGINE_PRECISION)?;
        for l in &images {
            for s1 in 0..nsh {
                for s2 in 0..nsh {
                    if let Some(blk) = eng_s.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        add_pair_deriv(&mut g_s, blk, &m, &dims, &offs, &sh2at, s1, s2);
                        if let Some(mg) = m_g0 {
                            add_pair_deriv(&mut g_fit_g0, blk, mg, &dims, &offs, &sh2at, s1, s2);
                        }
                    }
                    if let Some(blk) = eng_t.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        add_pair_deriv(&mut g_t, blk, &d, &dims, &offs, &sh2at, s1, s2);
                    }
                }
            }
        }
    }

    // --- V_SR: Gaussian nuclei, erfc(ω), the energy's image/screen sets.
    let sr_cfg = PeriodicHcoreConfig {
        nucleus_exponent: cfg
            .nucleus_exponent
            .unwrap_or(GRAD_NUCLEUS_EXPONENT)
            .min(hcore_cfg.nucleus_exponent),
        ..*hcore_cfg
    };
    let (mut g_vsr_basis, g_vsr_nuc, n_sr_triplets) =
        sr_attraction_gradient(cell, prep, &sr_cfg, &d, ledger)?;

    let aoat = ao_atoms(prep);
    let pos = cell.positions();

    // --- V_LR: pair-FT derivative (basis) and structure factor (nucleus).
    let gcut_lr = lr_gcut(prep, omega, hcore_cfg.precision);
    ledger.reserve(
        &format!("gamma gradient V_LR G list (|G| <= {gcut_lr:.3})"),
        gvector_list_bytes(cell, gcut_lr)?,
    )?;
    let gv_lr = half_gvectors(cell, gcut_lr)?;
    let mut g_vlr_basis = Array2::<f64>::zeros((natoms, 3));
    let mut g_vlr_nuc = Array2::<f64>::zeros((natoms, 3));
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let mut n_chunks = pair_ft_deriv_chunked(
        cell,
        prep,
        &gv_lr,
        0.1 * hcore_cfg.precision,
        chunk_budget,
        0,
        |_g0, gs, p, q| {
            let mut rq = vec![Complex64::new(0.0, 0.0); natoms * 3];
            for (g, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let v = 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp();
                let fac = -2.0 / vol * v;
                // S(G) = Σ_C Z_C e^{−iG·R_C}
                let (mut sre, mut sim) = (0.0_f64, 0.0_f64);
                for (z, r) in zs.iter().zip(&pos) {
                    let ph = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                    sre += z * ph.cos();
                    sim -= z * ph.sin();
                }
                let mut rho = Complex64::new(0.0, 0.0);
                rq.fill(Complex64::new(0.0, 0.0));
                for mu in 0..n {
                    let a = aoat[mu];
                    for nu in 0..n {
                        let dmn = d[(mu, nu)];
                        if dmn == 0.0 {
                            continue;
                        }
                        rho += p[[mu, nu, g]] * dmn;
                        for c in 0..3 {
                            rq[a * 3 + c] += q[c][[mu, nu, g]] * (2.0 * dmn);
                        }
                    }
                }
                for a in 0..natoms {
                    for c in 0..3 {
                        let z = rq[a * 3 + c];
                        // Re[ρ'* S] = ρ'.re S.re + ρ'.im S.im
                        g_vlr_basis[(a, c)] += fac * (z.re * sre + z.im * sim);
                    }
                    if zs[a] == 0.0 {
                        continue;
                    }
                    let ph = gvec[0] * pos[a][0] + gvec[1] * pos[a][1] + gvec[2] * pos[a][2];
                    // t = (−i) Z_A e^{−iφ} = Z_A (−sin φ − i cos φ)
                    let (tre, tim) = (-zs[a] * ph.sin(), -zs[a] * ph.cos());
                    let re = rho.re * tre + rho.im * tim;
                    for c in 0..3 {
                        g_vlr_nuc[(a, c)] += fac * re * gvec[c];
                    }
                }
            }
            Ok(())
        },
    )?;
    if mutation == Some(GradMutation::VneNoBasis) {
        g_vsr_basis.fill(0.0);
        g_vlr_basis.fill(0.0);
    }

    // --- Two-electron: dense pure AFT (pair-FT derivative of I) or RS-GDF
    // (Σ Y dJ3 + Σ Wm dJ2, rsgdf::deriv).
    let zeros_n = || Array2::<f64>::zeros((natoms, 3));
    let mut g_eri = zeros_n();
    let (mut f_orb_sr, mut f_orb_lr, mut f_aux_sr, mut f_aux_lr, mut f_met_sr, mut f_met_lr) = (
        zeros_n(),
        zeros_n(),
        zeros_n(),
        zeros_n(),
        zeros_n(),
        zeros_n(),
    );
    let n_g_eri: usize;
    let mut fit_diag: Option<RsGdfFitDiagnostics> = None;
    match jk {
        JkSource::Dense(eri) => {
            let (g, chunks, n_g) =
                dense_eri_gradient(cell, prep, eri, spins, &d, alpha, &aoat, mutation, ledger)?;
            g_eri = g;
            n_chunks += chunks;
            n_g_eri = n_g;
        }
        JkSource::Fit(_) => {
            let ff = fit_two_electron(cell, prep, fit, natoms, mutation, ledger)?;
            f_orb_sr = ff.orb_sr;
            f_orb_lr = ff.orb_lr;
            f_aux_sr = ff.aux_sr;
            f_aux_lr = ff.aux_lr;
            f_met_sr = ff.met_sr;
            f_met_lr = ff.met_lr;
            n_chunks += ff.n_chunks;
            n_g_eri = ff.n_g_half;
            fit_diag = Some(ff.diag);
        }
    }

    // --- Ewald E_nn (the energy's ω; the result is ω-independent).
    let (nn_sr, nn_lr) =
        ewald_nuclear_gradient_parts(cell, default_ewald_omega(cell), DEFAULT_EWALD_PRECISION)?;
    let to_arr = |v: &[[f64; 3]]| Array2::from_shape_fn((natoms, 3), |(a, c)| v[a][c]);
    let g_nn_sr = to_arr(&nn_sr);
    let mut g_nn_lr = to_arr(&nn_lr);
    if mutation == Some(GradMutation::NoEwaldLr) {
        g_nn_lr.fill(0.0);
    }

    let zeros = || Array2::<f64>::zeros((natoms, 3));
    let (xc_ao, xc_point, xc_weight, e_xc, n_grid_points) = match xc {
        Some(x) => (x.ao, x.point, x.weight, Some(x.e_xc), x.npts),
        None => (zeros(), zeros(), zeros(), None, 0),
    };

    let mut grad = &g_s
        + &g_t
        + &g_vsr_basis
        + &g_vsr_nuc
        + &g_vlr_basis
        + &g_vlr_nuc
        + &g_eri
        + &g_nn_sr
        + &g_nn_lr
        + &f_orb_sr
        + &f_orb_lr
        + &f_aux_sr
        + &f_aux_lr
        + &f_met_sr
        + &f_met_lr
        + &g_fit_g0;
    if e_xc.is_some() {
        grad = grad + &xc_ao + &xc_point + &xc_weight;
    }
    let net_force = (0..3)
        .map(|c| grad.column(c).sum().abs())
        .fold(0.0_f64, f64::max);
    ferric_core::memory::warn_if_rss_over("ferric-pbc gamma gradient", ledger.budget(), 1.1);
    Ok(GammaGradient {
        grad,
        parts: GammaGradParts {
            overlap: g_s,
            kinetic: g_t,
            vsr_basis: g_vsr_basis,
            vsr_nuc: g_vsr_nuc,
            vlr_basis: g_vlr_basis,
            vlr_nuc: g_vlr_nuc,
            eri: g_eri,
            nn_sr: g_nn_sr,
            nn_lr: g_nn_lr,
            xc_ao,
            xc_point,
            xc_weight,
            fit_orb_sr: f_orb_sr,
            fit_orb_lr: f_orb_lr,
            fit_aux_sr: f_aux_sr,
            fit_aux_lr: f_aux_lr,
            fit_metric_sr: f_met_sr,
            fit_metric_lr: f_met_lr,
            fit_g0: g_fit_g0,
        },
        commutator,
        madelung: vm,
        exact_exchange_fraction: alpha,
        e_xc,
        n_grid_points,
        net_force,
        n_sr_triplets,
        n_images: images.len(),
        n_g_lr: gv_lr.len(),
        n_g_eri,
        n_chunks,
        budget_bytes: ledger.budget(),
        fit: fit_diag,
    })
}

/// Dense pure-AFT two-electron term `Σ Γ dI` through the pair-FT derivative
/// of I. Returns `(g_eri, chunks, |G| count)`.
#[allow(clippy::too_many_arguments)]
fn dense_eri_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    eri: &DenseAftEri,
    spins: &SpinSet,
    d: &Array2<f64>,
    alpha: f64,
    aoat: &[usize],
    mutation: Option<GradMutation>,
    ledger: &mut Ledger,
) -> Result<(Array2<f64>, usize, usize), FerricError> {
    let n = prep.nbasis();
    let natoms = cell.positions().len();
    let vol = cell.volume();
    let mut g_eri = Array2::<f64>::zeros((natoms, 3));
    // Exchange part of Z: −(α/2) Σ_σ D_σ P D_σ (MUTANT ExchTotal: −(α/4) D P D).
    let exch = |x: &Array2<f64>| -> Array2<f64> {
        if mutation == Some(GradMutation::ExchTotal) {
            d.dot(x).dot(d) * 0.5
        } else {
            spins.sandwich(x)
        }
    };
    let gcut_eri = eri.gcut();
    ledger.reserve(
        &format!("gamma gradient ERI G list (|G| <= {gcut_eri:.3})"),
        gvector_list_bytes(cell, gcut_eri)?,
    )?;
    let gv_eri = half_gvectors(cell, gcut_eri)?;
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let n_chunks = pair_ft_deriv_chunked(
        cell,
        prep,
        &gv_eri,
        eri.pair_thresh(),
        chunk_budget,
        0,
        |_g0, gs, p, q| {
            for (g, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let fac = 8.0 / vol * (4.0 * PI / g2);
                let pg = p.slice(s![.., .., g]);
                let pre = pg.mapv(|z| z.re);
                let pim = pg.mapv(|z| z.im);
                let rho_re = (d * &pre).sum();
                let rho_im = (d * &pim).sum();
                // Z = ½ D ρ − (α/2) Σ_σ D_σ P D_σ   (RHF: ½ D ρ − ¼ D P D)
                let (mut zr, mut zi) = if alpha != 0.0 {
                    (exch(&pre) * (-0.5 * alpha), exch(&pim) * (-0.5 * alpha))
                } else {
                    (Array2::<f64>::zeros((n, n)), Array2::<f64>::zeros((n, n)))
                };
                zr.scaled_add(0.5 * rho_re, d);
                zi.scaled_add(0.5 * rho_im, d);
                for mu in 0..n {
                    let a = aoat[mu];
                    let mut acc = [0.0_f64; 3];
                    for nu in 0..n {
                        let (zre, zim) = (zr[(mu, nu)], zi[(mu, nu)]);
                        for (c, qc) in q.iter().enumerate() {
                            let qz = qc[[mu, nu, g]];
                            // Re[Q* Z]
                            acc[c] += qz.re * zre + qz.im * zim;
                        }
                    }
                    for c in 0..3 {
                        g_eri[(a, c)] += fac * acc[c];
                    }
                }
            }
            Ok(())
        },
    )?;
    Ok((g_eri, n_chunks, gv_eri.len()))
}

/// `assemble`'s RS-GDF state from before the dS/dT pass: the source, its
/// fitted densities and J3's G = 0 overlap weight `M_g0` (`None` under
/// [`GradMutation::FitNoG0`]).
type FitPrelude<'a> = (RsGdfGradSource<'a>, FitDensities, Option<Array2<f64>>);

/// RS-GDF: the fitted densities `Y`, `Wm` and `M_g0 = −c0' Σ_P Y_P q_P`
/// (`c0'` at the GDF's ω; MUTANT FitG0Dense: the Iteration-16 dense-ERI
/// form). `None` for dense-AFT J/K.
#[allow(clippy::too_many_arguments)]
fn fit_prelude<'a>(
    jk: &JkSource<'a>,
    spins: &SpinSet,
    d: &Array2<f64>,
    s_mat: &Array2<f64>,
    alpha: f64,
    vol: f64,
    mutation: Option<GradMutation>,
    ledger: &mut Ledger,
) -> Result<Option<FitPrelude<'a>>, FerricError> {
    let src = match jk {
        JkSource::Dense(_) => return Ok(None),
        JkSource::Fit(src) => *src,
    };
    let n = d.nrows();
    let exch = spins.exch_terms();
    let fd = fit_densities(
        src.gdf,
        d,
        &exch,
        alpha,
        mutation == Some(GradMutation::FitTextbookMetric),
        ledger,
    )?;
    let omega_f = src.gdf.stats().omega;
    let c0f = PI / (omega_f * omega_f * vol);
    let m_g0 = match mutation {
        Some(GradMutation::FitNoG0) => None,
        Some(GradMutation::FitG0Dense) => {
            let nel = (d * s_mat).sum();
            let mut mg = d * (-c0f * nel);
            if alpha != 0.0 {
                mg.scaled_add(c0f * alpha, &spins.sandwich(s_mat));
            }
            Some(mg)
        }
        _ => {
            let q: ndarray::Array1<f64> = aux_ft(src.aux, &[[0.0; 3]])?.column(0).mapv(|z| z.re);
            let v = fd.y.t().dot(&q); // (n²)
            Some(Array2::from_shape_fn((n, n), |(a, b)| -c0f * v[a * n + b]))
        }
    };
    Ok(Some((src, fd, m_g0)))
}

/// The RS-GDF two-electron force parts `assemble` adds (aux-centre and
/// metric pieces folded onto the cell atoms) and the counters it reports.
struct FitTwoElectron {
    orb_sr: Array2<f64>,
    orb_lr: Array2<f64>,
    aux_sr: Array2<f64>,
    aux_lr: Array2<f64>,
    met_sr: Array2<f64>,
    met_lr: Array2<f64>,
    n_chunks: usize,
    n_g_half: usize,
    diag: RsGdfFitDiagnostics,
}

/// RS-GDF two-electron: `Σ Y dJ3 + Σ Wm dJ2` (rsgdf::deriv) with the
/// FitNoMetric / FitNoAux / FitNoLr3 mutants applied.
fn fit_two_electron(
    cell: &Cell,
    prep: &PreparedBasis,
    fit: Option<FitPrelude<'_>>,
    natoms: usize,
    mutation: Option<GradMutation>,
    ledger: &mut Ledger,
) -> Result<FitTwoElectron, FerricError> {
    let Some((src, fd, _)) = fit else {
        return Err(FerricError::General(
            "gamma gradient: RS-GDF fitted densities missing (internal)".into(),
        ));
    };
    let der = fit_derivatives(src.gdf, cell, prep, src.aux, &fd.y, &fd.wm, ledger)?;
    let fold = |x: &Array2<f64>| fold_aux(x, src.aux, src.aux_jac, natoms);
    let orb_sr = der.orb_sr;
    let mut orb_lr = der.orb_lr;
    let mut aux_sr = fold(&der.aux3_sr);
    let mut aux_lr = fold(&der.aux3_lr);
    let mut met_sr = fold(&der.metric_sr);
    let mut met_lr = fold(&der.metric_lr);
    match mutation {
        Some(GradMutation::FitNoMetric) => {
            met_sr.fill(0.0);
            met_lr.fill(0.0);
        }
        Some(GradMutation::FitNoAux) => {
            aux_sr.fill(0.0);
            aux_lr.fill(0.0);
        }
        Some(GradMutation::FitNoLr3) => {
            orb_lr.fill(0.0);
            aux_lr.fill(0.0);
        }
        _ => {}
    }
    let mut diag = fd.diag;
    diag.n_sr3_deriv = der.n_sr3;
    diag.n_sr2_deriv = der.n_sr2;
    diag.n_g_half = der.n_g_half;
    diag.n_g_chunks = der.n_chunks;
    Ok(FitTwoElectron {
        orb_sr,
        orb_lr,
        aux_sr,
        aux_lr,
        met_sr,
        met_lr,
        n_chunks: der.n_chunks,
        n_g_half: der.n_g_half,
        diag,
    })
}

/// One chunk's XC contributions.
struct XcChunk {
    e: f64,
    v_a: Array2<f64>,
    v_b: Option<Array2<f64>>,
    ao: Array2<f64>,
    point: Array2<f64>,
    weight: Array2<f64>,
}

/// One spin channel of a chunk's XC kernel: `D_σ`, the gated `v_ρσ` and the
/// gated GGA vector `c^σ` (`3 × np`; `None` for LDA).
struct SpinTerm<'t> {
    d: &'t Array2<f64>,
    vr: Vec<f64>,
    c: Option<Array2<f64>>,
}

/// A chunk's `E_xc`, `V_σ`, `e(r_g) = ρ ε_xc` and per-spin terms.
struct ChunkKernel<'t> {
    e: f64,
    v_a: Array2<f64>,
    v_b: Option<Array2<f64>>,
    eps: Vec<f64>,
    terms: Vec<SpinTerm<'t>>,
}

/// The XC energy, potential(s) and the three XC force pieces (module doc) on
/// the rebuilt periodic grid with weight derivatives.
#[allow(clippy::too_many_arguments)]
fn xc_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    functional: &str,
    grid_cfg: &PeriodicGridConfig,
    ao_threshold: f64,
    dens_in: XcDensity<'_>,
    aoat: &[usize],
    mutation: Option<GradMutation>,
    ledger: &mut Ledger,
) -> Result<XcGradient, FerricError> {
    let (xc_def, _) = resolve_periodic_functional(functional)?;
    let xc_pol = match dens_in {
        XcDensity::Closed(_) => None,
        XcDensity::Polarized(..) => Some(xc_def_from_name_nspin(functional, 2).map_err(|e| {
            FerricError::General(format!(
                "periodic XC gradient: functional {functional:?} (spin-polarized): {e:?}"
            ))
        })?),
    };
    let gga = xc_def
        .funcs
        .iter()
        .any(|f| !matches!(f.family(), FunctionalFamily::Lda));
    let use_gga = gga && mutation != Some(GradMutation::XcNoGga);
    let natoms = cell.positions().len();
    let nbf = prep.nbasis();

    let rg = build_response_grid(cell, grid_cfg, ledger)?;
    let ev = LatticeAoHess::new(cell, prep.basis_set(), &rg.points, ao_threshold, ledger)?;
    if ev.nbf() != nbf || aoat.len() != nbf {
        return Err(FerricError::General(format!(
            "periodic XC gradient: lattice AO count {} != prepared basis {nbf}",
            ev.nbf()
        )));
    }
    let chunks = LatticeAoHess::chunks(&rg.points, XC_GRAD_CHUNK);
    let par = rayon::current_num_threads().max(1);
    // Per concurrently live chunk: χ, ∇χ, ∇∇χ (13 planes), per spin D·χ and
    // D·∇χ (≤ 8), the V_xc GEMM scratch (1); V_σ (≤ 2 n²); the per-point
    // per-atom scratch.
    let per_chunk = bytes_of((nbf * XC_GRAD_CHUNK) as u64, 22 * 8)
        .saturating_add(bytes_of((nbf * nbf) as u64, 2 * 8))
        .saturating_add(bytes_of((XC_GRAD_CHUNK * natoms * 3) as u64, 8));
    ledger.reserve(
        &format!(
            "periodic XC gradient AO/Hessian chunks ({par} x {XC_GRAD_CHUNK} points, nbf = {nbf})"
        ),
        per_chunk.saturating_mul(par),
    )?;

    let run_chunk = |idx: &Vec<usize>| -> Result<XcChunk, FerricError> {
        let pts: Vec<GridPoint> = idx.iter().map(|&i| rg.points[i]).collect();
        let xyz: Vec<[f64; 3]> = pts.iter().map(|g| g.xyz).collect();
        let (chi, dchi, ddchi) = ev.eval(&xyz)?;
        let np = pts.len();
        let ChunkKernel {
            e,
            v_a,
            v_b,
            eps,
            terms,
        } = match dens_in {
            XcDensity::Closed(d) => {
                let dens = eval_density_closed(d, &chi, &dchi);
                let k = closed_kernel(&dens, None, &xc_def);
                let (e, v) = semilocal_vxc_closed(&pts, &chi, &dchi, &dens, None, &xc_def);
                let live = |g: usize| dens.rho[g] > DENSITY_FLOOR;
                let vr: Vec<f64> = (0..np)
                    .map(|g| if live(g) { k.vrho[g] } else { 0.0 })
                    .collect();
                let c = gga.then(|| {
                    Array2::from_shape_fn((3, np), |(i, g)| {
                        if live(g) {
                            2.0 * k.vsigma[g] * dens.grad[(i, g)]
                        } else {
                            0.0
                        }
                    })
                });
                ChunkKernel {
                    e,
                    v_a: v,
                    v_b: None,
                    eps: (0..np).map(|g| dens.rho[g] * k.exc[g]).collect(),
                    terms: vec![SpinTerm { d, vr, c }],
                }
            }
            XcDensity::Polarized(da, db) => {
                let xp = xc_pol.as_ref().expect("polarized XcDef resolved above");
                let dens = eval_density_uks(da, db, &chi, &dchi);
                let k = polarized_kernel(&dens, None, xp);
                let (e, va, vb) = semilocal_vxc_polarized(&pts, &chi, &dchi, &dens, None, xp);
                let gate = |rho: &ndarray::Array1<f64>, v: &ndarray::Array1<f64>| -> Vec<f64> {
                    (0..np)
                        .map(|g| if rho[g] > DENSITY_FLOOR { v[g] } else { 0.0 })
                        .collect()
                };
                let vr_a = gate(&dens.rho_a, &k.vrho_a);
                let vr_b = gate(&dens.rho_b, &k.vrho_b);
                // c^α = 2 v_σαα ∇ρ_α + v_σαβ ∇ρ_β (gated on ρ_α), and α↔β.
                let (c_a, c_b) = if gga {
                    let ca = Array2::from_shape_fn((3, np), |(i, g)| {
                        if dens.rho_a[g] > DENSITY_FLOOR {
                            2.0 * k.vsigma_aa[g] * dens.grad_a[(i, g)]
                                + k.vsigma_ab[g] * dens.grad_b[(i, g)]
                        } else {
                            0.0
                        }
                    });
                    let cb = Array2::from_shape_fn((3, np), |(i, g)| {
                        if dens.rho_b[g] > DENSITY_FLOOR {
                            2.0 * k.vsigma_bb[g] * dens.grad_b[(i, g)]
                                + k.vsigma_ab[g] * dens.grad_a[(i, g)]
                        } else {
                            0.0
                        }
                    });
                    (Some(ca), Some(cb))
                } else {
                    (None, None)
                };
                ChunkKernel {
                    e,
                    v_a: va,
                    v_b: Some(vb),
                    eps: (0..np)
                        .map(|g| (dens.rho_a[g] + dens.rho_b[g]) * k.exc[g])
                        .collect(),
                    terms: vec![
                        SpinTerm {
                            d: da,
                            vr: vr_a,
                            c: c_a,
                        },
                        SpinTerm {
                            d: db,
                            vr: vr_b,
                            c: c_b,
                        },
                    ],
                }
            }
        };
        // a[g, A, j] = −2 w_g Σ_σ Σ_{μ∈A} t^σ[μ, j, g]
        let mut a_pt = Array3::<f64>::zeros((np, natoms, 3));
        for SpinTerm { d: dm, vr, c } in &terms {
            let m = dm.dot(&chi);
            let c = if use_gga { c.as_ref() } else { None };
            let md: Option<[Array2<f64>; 3]> =
                c.map(|_| std::array::from_fn(|i| dm.dot(&dchi.index_axis(Axis(0), i))));
            for g in 0..np {
                let wg = -2.0 * pts[g].weight;
                for mu in 0..nbf {
                    let at = aoat[mu];
                    let mm = m[(mu, g)];
                    for j in 0..3 {
                        let dj = dchi[(j, mu, g)];
                        let mut t = vr[g] * dj * mm;
                        if let (Some(c), Some(md)) = (c, md.as_ref()) {
                            for i in 0..3 {
                                t += c[(i, g)] * (ddchi[(j, i, mu, g)] * mm + dj * md[i][(mu, g)]);
                            }
                        }
                        a_pt[(g, at, j)] += wg * t;
                    }
                }
            }
        }
        let mut ao = Array2::<f64>::zeros((natoms, 3));
        let mut point = Array2::<f64>::zeros((natoms, 3));
        let mut weight = Array2::<f64>::zeros((natoms, 3));
        for (g, &gi) in idx.iter().enumerate() {
            let home = pts[g].home_atom;
            let mut tot = [0.0_f64; 3];
            for at in 0..natoms {
                for j in 0..3 {
                    let v = a_pt[(g, at, j)];
                    ao[(at, j)] += v;
                    tot[j] += v;
                    weight[(at, j)] += eps[g] * rg.dweight[gi * rg.natoms + at][j];
                }
            }
            for j in 0..3 {
                point[(home, j)] -= tot[j];
            }
        }
        Ok(XcChunk {
            e,
            v_a,
            v_b,
            ao,
            point,
            weight,
        })
    };

    // Concurrency-sized batches, reduced in chunk order: memory is bounded
    // by `par` chunks and the sums do not depend on the thread count.
    let mut e_xc = 0.0;
    let mut v_a = Array2::<f64>::zeros((nbf, nbf));
    let mut v_b =
        matches!(dens_in, XcDensity::Polarized(..)).then(|| Array2::<f64>::zeros((nbf, nbf)));
    let mut ao = Array2::<f64>::zeros((natoms, 3));
    let mut point = Array2::<f64>::zeros((natoms, 3));
    let mut weight = Array2::<f64>::zeros((natoms, 3));
    for batch in chunks.chunks(par) {
        let outs: Vec<Result<XcChunk, FerricError>> = batch.par_iter().map(&run_chunk).collect();
        for out in outs {
            let c = out?;
            e_xc += c.e;
            v_a += &c.v_a;
            if let (Some(acc), Some(vb)) = (v_b.as_mut(), c.v_b.as_ref()) {
                *acc += vb;
            }
            ao += &c.ao;
            point += &c.point;
            weight += &c.weight;
        }
    }
    if !e_xc.is_finite()
        || v_a.iter().any(|x| !x.is_finite())
        || ao
            .iter()
            .chain(point.iter())
            .chain(weight.iter())
            .any(|x| !x.is_finite())
    {
        return Err(FerricError::General(format!(
            "periodic XC gradient: non-finite E_xc/V_xc/force for {functional}"
        )));
    }
    match mutation {
        Some(GradMutation::XcNoAo) => ao.fill(0.0),
        Some(GradMutation::NoPointMotion) => point.fill(0.0),
        Some(GradMutation::NoWeightDeriv) => weight.fill(0.0),
        _ => {}
    }
    Ok(XcGradient {
        e_xc,
        v_a,
        v_b,
        ao,
        point,
        weight,
        npts: rg.points.len(),
    })
}

/// `g[atom(s1)] += Σ w_μν ∂_bra`, `g[atom(s2)] += Σ w_μν ∂_ket` for one
/// shifted 1e derivative block (`[bra xyz, ket xyz]`, each `n1 × n2`).
#[allow(clippy::too_many_arguments)]
fn add_pair_deriv(
    g: &mut Array2<f64>,
    blk: &[f64],
    w: &Array2<f64>,
    dims: &[usize],
    offs: &[usize],
    sh2at: &[usize],
    s1: usize,
    s2: usize,
) {
    let (n1, n2) = (dims[s1], dims[s2]);
    let bs = n1 * n2;
    let (a1, a2) = (sh2at[s1], sh2at[s2]);
    for i in 0..n1 {
        for j in 0..n2 {
            let wv = w[(offs[s1] + i, offs[s2] + j)];
            if wv == 0.0 {
                continue;
            }
            let idx = i * n2 + j;
            for c in 0..3 {
                g[(a1, c)] += wv * blk[c * bs + idx];
                g[(a2, c)] += wv * blk[(3 + c) * bs + idx];
            }
        }
    }
}
