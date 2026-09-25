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
//! # Out of scope (documented, not implemented)
//!
//! * **RS-GDF J/K gradient** — the scalable route. Needs the standard DF
//!   gradient `Σ Γ^P_μν d(μν|P)' − ½ Σ Γ^{PQ} d(P|Q)'` with every primed
//!   integral in the RS split (SR erfc 3c/2c derivative integrals over pair
//!   and aux images; LR pair-FT derivative `Q` and the one-centre aux FT
//!   derivative; G = 0 only through `dS`), plus a check that the lindep
//!   eigen-cut count does not change between displaced geometries
//!   (FINDINGS "Iteration 16", "RS-GDF gradient route"). Only the dense-AFT
//!   oracle is differentiated here.
//! * **Stress** — needs the lattice-vector derivative of every piece
//!   (per-image `L ⊗ Q_L` virials, `∂v(G)/∂ε`, `∂v_M/∂ε`, `∂Ω/∂ε`); FINDINGS
//!   "Iteration 16", "Stress".
//! * ECPs (`v_ecp`), ROHF/ROKS, meta-GGA, range-separated hybrids, VV10,
//!   k-points: rejected or not provided.

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
    /// loses precision for very tight Gaussians: measured on the triclinic
    /// 4H s+p cell, the FD error was 8.3e-2 at 1e16, 4.3e-6 at 1e12 and 1.9e-7
    /// at 1e10, while the smeared-vs-point potential error is ~3.6e-7 at 1e10.
    pub nucleus_exponent: Option<f64>,
}

/// Default exponent for the SR attraction derivative (see
/// [`GammaGradConfig::nucleus_exponent`]).
pub const GRAD_NUCLEUS_EXPONENT: f64 = 1e10;

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
}

/// The Iteration-16 name of [`GammaGradient`].
pub type GammaRhfGradient = GammaGradient;

/// Per-spin densities and Fock matrices at the SCF solution.
enum SpinSet {
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
    fn total(&self) -> Array2<f64> {
        match self {
            Self::Restricted { d, .. } => d.clone(),
            Self::Unrestricted { da, db, .. } => da + db,
        }
    }

    /// `Σ_σ D_σ X D_σ` (restricted: `½ D X D`).
    fn sandwich(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Restricted { d, .. } => d.dot(x).dot(d) * 0.5,
            Self::Unrestricted { da, db, .. } => da.dot(x).dot(da) + db.dot(x).dot(db),
        }
    }

    /// `W = Σ_σ D_σ F_σ D_σ` (restricted: `½ D F D`).
    fn energy_weighted(&self) -> Array2<f64> {
        match self {
            Self::Restricted { d, f } => 0.5 * d.dot(f).dot(d),
            Self::Unrestricted { da, db, fa, fb } => da.dot(fa).dot(da) + db.dot(fb).dot(db),
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
    fn commutator(&self, s_mat: &Array2<f64>) -> f64 {
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
    let who = "gamma_rhf_gradient";
    check_inputs(who, prep, hcore_cfg, hc, eri, scf, Spin::Restricted)?;
    let mut ledger = open_ledger(cell, prep, cfg, false)?;
    let d = scf.density_total.clone();
    let vm = madelung_for(cell, exxdiv)?;
    let n = prep.nbasis();
    let mut jm = Array2::<f64>::zeros((n, n));
    let mut km = Array2::<f64>::zeros((n, n));
    eri.j_builder().build(&d, &mut jm)?;
    eri.k_builder_with_madelung(vm).build(&d, &mut km)?;
    let f = &(&hc.h + &jm) - &(0.5 * &km);
    let spins = SpinSet::Restricted { d, f };
    assemble(
        cell,
        prep,
        hcore_cfg,
        hc,
        eri,
        &spins,
        1.0,
        vm,
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
    let who = "gamma_uhf_gradient";
    check_inputs(who, prep, hcore_cfg, hc, eri, scf, Spin::Unrestricted)?;
    let mut ledger = open_ledger(cell, prep, cfg, true)?;
    let (da, db) = spin_densities(who, scf)?;
    let vm = madelung_for(cell, exxdiv)?;
    let (fa, fb) = unrestricted_focks(hc, eri, &da, &db, 1.0, vm, None)?;
    let spins = SpinSet::Unrestricted { da, db, fa, fb };
    assemble(
        cell,
        prep,
        hcore_cfg,
        hc,
        eri,
        &spins,
        1.0,
        vm,
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
    let who = "gamma_rks_gradient";
    check_inputs(who, prep, hcore_cfg, hc, eri, scf, Spin::Restricted)?;
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
    eri.j_builder().build(&d, &mut jm)?;
    let mut f = &(&hc.h + &jm) + &xg.v_a;
    if alpha != 0.0 {
        let mut km = Array2::<f64>::zeros((n, n));
        eri.k_builder_with_madelung(vm).build(&d, &mut km)?;
        f.scaled_add(-0.5 * alpha, &km);
    }
    let spins = SpinSet::Restricted { d, f };
    assemble(
        cell,
        prep,
        hcore_cfg,
        hc,
        eri,
        &spins,
        alpha,
        vm,
        Some(xg),
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
    let who = "gamma_uks_gradient";
    check_inputs(who, prep, hcore_cfg, hc, eri, scf, Spin::Unrestricted)?;
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
    let (fa, fb) = unrestricted_focks(hc, eri, &da, &db, alpha, vm, Some((&xg.v_a, &v_b)))?;
    let spins = SpinSet::Unrestricted { da, db, fa, fb };
    assemble(
        cell,
        prep,
        hcore_cfg,
        hc,
        eri,
        &spins,
        alpha,
        vm,
        Some(xg),
        cfg,
        &mut ledger,
    )
}

/// Shared argument checks (messages prefixed by `who`).
fn check_inputs(
    who: &str,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
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
            _ => "an unrestricted",
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
    if hc.s.dim() != (n, n) || !d_ok || eri.eri().dim() != (n * n, n * n) {
        return Err(FerricError::General(format!(
            "{who}: shape mismatch (nbasis {n}: S {:?}, D {:?}, ERI {:?})",
            hc.s.dim(),
            scf.density_total.dim(),
            eri.eri().dim()
        )));
    }
    Ok(())
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
        bytes_of((natoms * 3) as u64, 8 * 16),
    )?;
    Ok(ledger)
}

fn madelung_for(cell: &Cell, exxdiv: ExxDiv) -> Result<f64, FerricError> {
    Ok(match exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => madelung_constant(cell)?,
    })
}

fn spin_densities(who: &str, scf: &ScfResult) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let db = scf
        .density_beta
        .clone()
        .ok_or_else(|| FerricError::General(format!("{who}: the SCF has no beta density")))?;
    Ok((scf.density_alpha.clone(), db))
}

/// `F_σ = h + J[D_α + D_β] − α (K[D_σ] + v_M S D_σ S) + V_σ`.
fn unrestricted_focks(
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    da: &Array2<f64>,
    db: &Array2<f64>,
    alpha: f64,
    vm: f64,
    v: Option<(&Array2<f64>, &Array2<f64>)>,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let n = da.nrows();
    let d = da + db;
    let mut jm = Array2::<f64>::zeros((n, n));
    eri.j_builder().build(&d, &mut jm)?;
    let base = &hc.h + &jm;
    let mut fa = base.clone();
    let mut fb = base;
    if alpha != 0.0 {
        let mut k = eri.k_builder_with_madelung(vm);
        let mut ka = Array2::<f64>::zeros((n, n));
        let mut kb = Array2::<f64>::zeros((n, n));
        k.build(da, &mut ka)?;
        k.build(db, &mut kb)?;
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
fn ao_atoms(prep: &PreparedBasis) -> Vec<usize> {
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
#[allow(clippy::too_many_arguments)]
fn assemble(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    spins: &SpinSet,
    alpha: f64,
    vm: f64,
    xc: Option<XcGradient>,
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

    // --- dS, dT over the energy's pair images (bra and ket blocks).
    let images = hcore_pair_images(cell, prep, hcore_cfg.precision, ledger)?;
    let sh2at = prep.shell_to_atom().to_vec();
    let dims = prep.shell_dims().to_vec();
    let offs = prep.shell_offsets().to_vec();
    let nsh = prep.nshells();
    let mut g_s = Array2::<f64>::zeros((natoms, 3));
    let mut g_t = Array2::<f64>::zeros((natoms, 3));
    {
        let mut eng_s = Engine::new_1e_deriv(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
        let mut eng_t = Engine::new_1e_deriv(ffi::OP_KINETIC, prep, ONE_E_ENGINE_PRECISION)?;
        for l in &images {
            for s1 in 0..nsh {
                for s2 in 0..nsh {
                    if let Some(blk) = eng_s.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        add_pair_deriv(&mut g_s, blk, &m, &dims, &offs, &sh2at, s1, s2);
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

    // --- Two-electron (dense pure AFT J and K) through the pair-FT derivative.
    // Exchange part of Z: −(α/2) Σ_σ D_σ P D_σ (MUTANT ExchTotal: −(α/4) D P D).
    let exch = |x: &Array2<f64>| -> Array2<f64> {
        if mutation == Some(GradMutation::ExchTotal) {
            d.dot(x).dot(&d) * 0.5
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
    let mut g_eri = Array2::<f64>::zeros((natoms, 3));
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    n_chunks += pair_ft_deriv_chunked(
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
                let rho_re = (&d * &pre).sum();
                let rho_im = (&d * &pim).sum();
                // Z = ½ D ρ − (α/2) Σ_σ D_σ P D_σ   (RHF: ½ D ρ − ¼ D P D)
                let (mut zr, mut zi) = if alpha != 0.0 {
                    (exch(&pre) * (-0.5 * alpha), exch(&pim) * (-0.5 * alpha))
                } else {
                    (Array2::<f64>::zeros((n, n)), Array2::<f64>::zeros((n, n)))
                };
                zr.scaled_add(0.5 * rho_re, &d);
                zi.scaled_add(0.5 * rho_im, &d);
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
        + &g_nn_lr;
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
        n_g_eri: gv_eri.len(),
        n_chunks,
        budget_bytes: ledger.budget(),
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
