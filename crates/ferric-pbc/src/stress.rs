//! Analytic stress tensor of the Gamma-point periodic RHF, UHF, RKS and UKS
//! (dense pure-AFT J/K or RS-GDF J/K) — the Rust port of
//! `reference/pbc/pbc_stress.py` (FINDINGS "Iteration 19").
//!
//! # Definition
//!
//! Homogeneous strain `r → (1 + ε) r` of the lattice rows AND the atoms
//! (fixed fractional coordinates; PySCF `rks_stress` convention, same sign):
//!
//! ```text
//! σ_ab = (1/Ω) dE/dε_ab,     a_i → (1 + ε) a_i,  R_A → (1 + ε) R_A,
//! G = Σ n_i b_i → (1 + ε)^{−T} G  (Miller index n FIXED),  Ω → Ω det(1 + ε)
//! ```
//!
//! Basis functions keep their exponents and move with their atoms in every
//! image. Every truncated set (G spheres, image lists) is FROZEN as integer
//! indices at the reference cell ([`Cell::strained`]); this module is the
//! exact derivative of THAT energy. Re-selecting `|G| <= gcut` at each strain
//! makes `E(ε)` piecewise (FINDINGS "Iteration 19" (g)); at a truncated set
//! the fixed-index stress differs from the converged one by a plane-wave-style
//! "Pulay stress" that vanishes with the cutoff ((h): 1 % on one component at
//! an unconverged gcut 12, 1e-10 by gcut 20; ferric's spheres are derived from
//! `precision` and sit far beyond that).
//!
//! # Energy (the `crate::grad` conventions)
//!
//! ```text
//! E = Σ D (T + V_SR + V_LR + c0 Z_tot S) + E_2e − (α v_M/2) Σ_σ tr(D_σ S D_σ S) + E_xc + E_nn
//! V_LR = −(2/Ω) Σ_{G∈half} v_ω Re[P̄ S(G)],   v_ω = 4π/G² e^{−G²/4ω²},   c0 = π/(ω²Ω)
//! E_2e (dense AFT) = (2/Ω) Σ_{G∈half} (4π/G²) Re Σ P̄ Z,  Z = ½ D ρ − (α/2) Σ_σ D_σ P D_σ
//! ```
//!
//! # Strain derivative `dE/dε_ab` (at the converged, fixed `D_σ`)
//!
//! Every real-space piece is "Σ over centres `∂E/∂X_{c,a} X_{c,b}`", written
//! with relative vectors (translation invariance) so it is origin-free; S(G)
//! and every `G·X` phase are strain-invariant.
//!
//! 1. Overlap-coupled (Pulay, `c0` and Madelung S-terms): `Σ M dS/dε`, with the
//!    forces' `M = −W + c0 Z_tot D − α v_M Σ_σ D_σ S D_σ` UNCHANGED, and
//!    `dS_mn/dε_ab = Σ_L ∂_{A,a}⟨m|n_L⟩ (A − B − L)_b` — the per-IMAGE
//!    derivative block of the forces weighted by its pair vector
//!    (`½(bra − ket)` of the shifted block). Kinetic: `Σ D dT`, the same.
//! 2. `V_SR` (Gaussian nuclei, erfc): per triplet `(g_{C,M}|m_0 n_L)`,
//!    `∂_A (A − X)_b + ∂_{B′} (B′ − X)_b`, `X = R_C + M`, `B′ = B + L`, from
//!    the two directly computed ket blocks (`hcore::sr_attraction_strain`).
//! 3. Pair FT (per primitive, `P = e^{−iG·P_c} R(G, A − B′)`):
//!    `dP/dε_ab = [Q_a + iG_a (α/p) P](A − B′)_b + G_a G_b/(2p) P − G_a P^g_b`
//!    ([`crate::pair_ft::pair_ft_strain_chunked`]).
//! 4. Every reciprocal energy `E_LR = (1/Ω) Σ v Φ`: `−δ_ab E_LR + (1/Ω) Σ [dv Φ + v dΦ]`,
//!    `dv/dε_ab = dv/d(G²) (−2 G_a G_b)`. `V_LR`: `Φ = −Re[ρ̄ S]`; ERI:
//!    `Φ = Re Σ P̄ Z`, `dΦ = 2 Re Σ dP̄ Z`.
//! 5. `c0 Z_tot N` volume term: `−δ_ab c0 Z_tot N` (its S part is in `M`).
//! 6. Madelung: `−(α/2) Σ_σ tr(D_σ S D_σ S) dv_M/dε`, `dv_M/dε = −2 ×` the
//!    Ewald strain of one unit charge ([`crate::ewald::madelung_strain`]).
//! 7. `E_nn`: [`crate::ewald::ewald_nuclear_strain`] (SR virial, LR volume +
//!    kernel, background volume).
//! 8. XC (atom-centred grid, points riding rigidly on their home atom with
//!    lab-fixed offsets): AO part `dχ_m(r_g)/dε_ab = Σ_L ∂_aχ_m(r_g − X_L)(O_g − X_L)_b`
//!    (`O_g` = home atom; GGA: the Hessian row for `d∇χ`), weight part
//!    `Σ_g e(r_g) dW_g/dε_ab`, `dW/dε_ab = w_rl Σ_B ∂P/∂X_{B,a} (X_B − R_home)_b`
//!    over IMAGE atoms (the forces' per-image weight derivative before its
//!    fold onto cell atoms).
//! 9. RS-GDF: `Σ Y dJ3/dε + Σ Wm dJ2/dε` with the forces' `Y`, `Wm` (Loewner
//!    form) unchanged ([`crate::rsgdf`]'s `strain` module), and J3's G = 0
//!    term through `dS` as the forces' `M_g0 = −c0′ Σ_P Y_P q_P`.
//!
//! # Rotational invariance
//!
//! HF (and a uniform grid) is exactly rotation invariant with frozen index
//! sets, so `σ − σᵀ ≈ 0` to roundoff. An atom-centred grid is NOT (the
//! Lebedev offsets are lab-fixed): its stress has an antisymmetric part at
//! grid-error size (prototype 1.8e-3..2.3e-3 at (40, 50)), which the analytic
//! value reproduces exactly (it is a property of the grid energy).
//! [`GammaStress::symmetric`] is what a cell optimiser should use;
//! [`GammaStress::antisymmetry`] is a free grid-quality diagnostic.
//!
//! # Scope
//!
//! ROHF/ROKS (`gamma_rohf_stress` / `gamma_roks_stress`) run the UHF/UKS
//! assembly on the ROHF spin densities and SPIN Focks, gated on the ROHF
//! orbital gradient (`crate::grad::RO_ORBITAL_GRADIENT_TOL`). They use the
//! UHF-form `W = Σ_σ D_σ F_σ D_σ`, which equals the ROHF Lagrangian `W_RO`
//! up to ½ × the closed–open β orbital gradient (`crate::grad` module doc,
//! "ROHF / ROKS"): exact at the stationary point, as for the forces.
//!
//! Not covered (refused or not provided, as for the forces): ECPs,
//! meta-GGA, range-separated hybrids, VV10, k-points, the uniform KS grid
//! (the energy path does not offer it), aux centres that do not strain
//! homogeneously with the cell.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::{DenseAftEri, ExxDiv};
use crate::dft::{
    build_strain_grid, resolve_periodic_functional, GammaRksConfig, GammaUksConfig, LatticeAoHess,
    PeriodicGridConfig,
};
use crate::ewald::{
    default_ewald_omega, ewald_nuclear_strain, madelung_strain, DEFAULT_EWALD_PRECISION,
};
use crate::grad::{
    check_inputs, madelung_for, ro_gate, spin_densities, unrestricted_focks, JkSource,
    RsGdfGradSource, SpinSet, GRAD_NUCLEUS_EXPONENT,
};
use crate::hcore::{
    gvector_list_bytes, half_gvectors, hcore_pair_images, lr_gcut, sr_attraction_strain,
    PeriodicHcore, PeriodicHcoreConfig, G_CHUNK_BYTES, ONE_E_ENGINE_PRECISION,
};
use crate::lattice::Cell;
use crate::pair_ft::{pair_ft_strain_chunked, PairFtStrainTerms};
use crate::rohf::GammaRoksConfig;
use crate::rsgdf::deriv::fit_densities;
use crate::rsgdf::strain::{fit_strain, FitStrainTerms};
use crate::rsgdf::{aux_ft, RsGdfFitDiagnostics};
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
use ferric_scf::result::{ScfResult, Spin};
use ndarray::{s, Array2, Axis};
use num_complex::Complex64;
use rayon::prelude::*;
use std::f64::consts::PI;

/// A 3 × 3 tensor, row-major `[a][b]`.
pub type Mat3 = [[f64; 3]; 3];

const ZERO3: Mat3 = [[0.0; 3]; 3];

/// Points per chunk of the XC stress (AO values, gradients, 9 first-moment
/// planes and, for GGA, 27 Hessian-moment planes live per chunk).
const XC_STRESS_CHUNK: usize = 64;

/// Deliberate defects for the mutation tests (`tests/pbc_stress.rs`): each
/// is the prototype's `_MUTANT` of the same name (FINDINGS "Iteration 19"
/// (c), measured misses ≥ 3e-2 except `NoAuxFt`) and must be caught by the
/// finite-difference anchor on at least one of the nine components. Never set
/// in production.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressMutation {
    /// Drop `Σ M dS` (overlap / Pulay term).
    NoPulay,
    /// Pair FT: drop the basis-centre (pair-vector) part of `dP`.
    FtNoCentres,
    /// G treated as unstrained: drop the G parts of `dP`, `dv` and `dX`.
    GUnstrained,
    /// Drop the `−δ E_LR` volume terms of `V_LR`, the ERI and the RS-GDF LR
    /// parts (diagonal only by construction).
    NoVolume,
    /// Drop the reciprocal part of the Ewald `E_nn` strain.
    NoEwaldLr,
    /// Drop the Ewald background volume term (diagonal only).
    NoEwaldBg,
    /// Drop `dv_M/dε` (exxdiv = ewald only).
    NoMadelung,
    /// Drop `−α v_M Σ_σ D_σ S D_σ` from `M` (the force-level Madelung term).
    MadelungS,
    /// Drop `−δ c0 Z_tot N` (hcore's G = 0 bookkeeping; diagonal only).
    NoC0Volume,
    /// SR / overlap / RS-GDF-SR virials with the image translations dropped
    /// from the pair vectors.
    NoSrImages,
    /// Drop the grid-weight strain derivative.
    XcNoWeight,
    /// Drop the XC AO strain term.
    XcNoAo,
    /// Atom grid anchored at the point itself (`O_g = r_g`: the uniform-grid
    /// formula on an atom-centred grid).
    XcPointFixed,
    /// RS-GDF: drop `Σ Wm dJ2` (SR, LR and its G = 0 volume term).
    NoMetric,
    /// RS-GDF: aux FT strain-free in J3 AND J2. Nearly blind on an accurate
    /// aux set (it errs by the fit error only; prototype 7.7e-6..1.2e-4) —
    /// use [`StressMutation::NoAuxFt3`] for a sharp check.
    NoAuxFt,
    /// RS-GDF: aux FT strain-free in J3 only (prototype 7.3e-2..1.6e-1).
    NoAuxFt3,
    /// RS-GDF: drop the strain-only J2 G = 0 volume term `+δ c0 qᵀ Wm q`.
    NoJ2G0Vol,
    /// RS-GDF: drop the J3 G = 0 volume term `+δ c0 Σ (Yq)·S`.
    NoJ3G0Vol,
    /// RS-GDF: drop the J3 G = 0 overlap term `Σ M_g0 dS`.
    NoG0,
}

/// Settings for the stress entry points.
#[derive(Debug, Clone, Copy, Default)]
pub struct GammaStressConfig {
    /// Memory budget (bytes); `None` = ferric's unified budget.
    pub budget_bytes: Option<usize>,
    /// TEST ONLY: a deliberate defect ([`StressMutation`]).
    #[doc(hidden)]
    pub mutation: Option<StressMutation>,
    /// Gaussian-nucleus exponent used ONLY for the SR attraction derivative
    /// (`None` = [`GRAD_NUCLEUS_EXPONENT`], never tighter than the energy's);
    /// see `GammaGradConfig::nucleus_exponent` (libint2 precision).
    pub nucleus_exponent: Option<f64>,
}

/// Per-term breakdown of `dE/dε` (each 3 × 3, Hartree; NOT divided by Ω).
#[derive(Debug, Clone, Default)]
pub struct GammaStressParts {
    /// `Σ M dS` (Pulay, `c0 Z_tot D`, Madelung S-term).
    pub overlap: Mat3,
    /// `Σ D dT`.
    pub kinetic: Mat3,
    /// `Σ D dV_SR` (basis centres and nuclei).
    pub vsr: Mat3,
    /// `Σ D dV_LR`, including its `−δ E` volume term.
    pub vlr: Mat3,
    /// `−δ c0 Z_tot N`.
    pub c0_volume: Mat3,
    /// Dense-AFT `Σ Γ dI`, including its volume term (zero on RS-GDF).
    pub eri: Mat3,
    /// `−(α/2) Σ_σ tr(D_σ S D_σ S) dv_M/dε` (zero for exxdiv = none).
    pub madelung: Mat3,
    /// Ewald `E_nn`: real-space, reciprocal (with volume), background.
    pub nn_sr: Mat3,
    pub nn_lr: Mat3,
    pub nn_background: Mat3,
    /// XC AO strain term (zero for HF).
    pub xc_ao: Mat3,
    /// XC grid-weight strain term (zero for HF).
    pub xc_weight: Mat3,
    /// RS-GDF `Σ Y dJ3`: SR, LR (with volume).
    pub fit_j3_sr: Mat3,
    pub fit_j3_lr: Mat3,
    /// RS-GDF `Σ Wm dJ2`: SR, LR (with volume).
    pub fit_j2_sr: Mat3,
    pub fit_j2_lr: Mat3,
    /// RS-GDF J3 G = 0 through `dS` (`Σ M_g0 dS`).
    pub fit_g0_overlap: Mat3,
    /// RS-GDF `+δ c0 Σ (Yq)·S` and `+δ c0 qᵀ Wm q`.
    pub fit_j3_g0_volume: Mat3,
    pub fit_j2_g0_volume: Mat3,
}

impl GammaStressParts {
    fn all(&self) -> [&Mat3; 19] {
        [
            &self.overlap,
            &self.kinetic,
            &self.vsr,
            &self.vlr,
            &self.c0_volume,
            &self.eri,
            &self.madelung,
            &self.nn_sr,
            &self.nn_lr,
            &self.nn_background,
            &self.xc_ao,
            &self.xc_weight,
            &self.fit_j3_sr,
            &self.fit_j3_lr,
            &self.fit_j2_sr,
            &self.fit_j2_lr,
            &self.fit_g0_overlap,
            &self.fit_j3_g0_volume,
            &self.fit_j2_g0_volume,
        ]
    }

    /// Sum of every part (`dE/dε`).
    pub fn total(&self) -> Mat3 {
        let mut t = ZERO3;
        for m in self.all() {
            add_into(&mut t, m, 1.0);
        }
        t
    }
}

/// Output of the stress entry points.
#[derive(Debug, Clone)]
pub struct GammaStress {
    /// `dE/dε_ab` (Hartree per cell).
    pub de_deps: Mat3,
    /// `σ_ab = (1/Ω) dE/dε_ab` (Hartree/Bohr³). Not symmetrised (module doc).
    pub sigma: Mat3,
    /// Per-term breakdown of `de_deps`.
    pub parts: GammaStressParts,
    /// Cell volume Ω (Bohr³).
    pub volume: f64,
    /// `max_σ max |F_σ D_σ S − S D_σ F_σ|` of the rebuilt Fock matrices
    /// (NOT zero for ROHF/ROKS; their gate is the ROHF orbital gradient).
    pub commutator: f64,
    /// `v_M` used in `K` and `M` (0 for exxdiv = none).
    pub madelung: f64,
    /// `dv_M/dε` of the cell (reported for every exxdiv).
    pub madelung_strain: Mat3,
    /// `N = tr(D S)`.
    pub electrons: f64,
    /// Exact-exchange fraction `α`.
    pub exact_exchange_fraction: f64,
    /// `E_xc` on the stress grid (`None` for HF).
    pub e_xc: Option<f64>,
    /// XC grid points (0 for HF).
    pub n_grid_points: usize,
    /// `Σ D V_LR` recomputed from the strain kernel's `P` (equals
    /// `Σ D ∘ hc.v_lr` to the primitive screens: a consistency check that the
    /// kernel's `P` is the energy's).
    pub e_vlr: f64,
    /// Dense-AFT `E_2e` recomputed from the strain kernel's `P` (0 on RS-GDF).
    pub e_eri: f64,
    /// SR attraction triplets, pair images, half-sphere G counts and pair-FT
    /// chunks (V_LR + ERI).
    pub n_sr_triplets: usize,
    pub n_images: usize,
    pub n_g_lr: usize,
    pub n_g_eri: usize,
    pub n_chunks: usize,
    /// Resolved memory budget (bytes).
    pub budget_bytes: usize,
    /// RS-GDF diagnostics (`None` on dense AFT).
    pub fit: Option<RsGdfFitDiagnostics>,
}

impl GammaStress {
    /// `(σ + σᵀ)/2`.
    pub fn symmetric(&self) -> Mat3 {
        let mut o = ZERO3;
        for a in 0..3 {
            for b in 0..3 {
                o[a][b] = 0.5 * (self.sigma[a][b] + self.sigma[b][a]);
            }
        }
        o
    }

    /// `max_ab |σ_ab − σ_ba|` (Hartree/Bohr³): roundoff for HF, grid-error
    /// sized for an atom-centred KS grid (module doc).
    pub fn antisymmetry(&self) -> f64 {
        let mut m = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                m = m.max((self.sigma[a][b] - self.sigma[b][a]).abs());
            }
        }
        m
    }

    /// Pressure `P = −tr(σ)/3 = −dE/dΩ` (Hartree/Bohr³).
    pub fn pressure(&self) -> f64 {
        -(self.sigma[0][0] + self.sigma[1][1] + self.sigma[2][2]) / 3.0
    }
}

fn add_into(t: &mut Mat3, m: &Mat3, f: f64) {
    for a in 0..3 {
        for b in 0..3 {
            t[a][b] += f * m[a][b];
        }
    }
}

fn diag(x: f64) -> Mat3 {
    [[x, 0.0, 0.0], [0.0, x, 0.0], [0.0, 0.0, x]]
}

fn scaled(m: &Mat3, f: f64) -> Mat3 {
    let mut o = *m;
    for row in o.iter_mut() {
        for v in row.iter_mut() {
            *v *= f;
        }
    }
    o
}

// ------------------------------------------------------------------ entries

/// Gamma-point RHF stress on the dense-AFT J/K. Arguments as
/// `crate::grad::gamma_rhf_gradient_with` (the SAME `hcore_cfg`, `hc`,
/// `eri` and converged `scf` the energy used; `exxdiv` the energy's).
#[allow(clippy::too_many_arguments)]
pub fn gamma_rhf_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaStressConfig,
) -> Result<GammaStress, FerricError> {
    hf_stress(
        "gamma_rhf_stress",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Dense(eri),
        scf,
        exxdiv,
        Spin::Restricted,
        cfg,
    )
}

/// Gamma-point RHF stress with RS-GDF J/K (`fit.gdf` from
/// `RsGdf::build_for_gradient`; the SCF ran on its J/K).
#[allow(clippy::too_many_arguments)]
pub fn gamma_rhf_stress_rsgdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    fit: &RsGdfGradSource<'_>,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaStressConfig,
) -> Result<GammaStress, FerricError> {
    hf_stress(
        "gamma_rhf_stress_rsgdf",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Fit(*fit),
        scf,
        exxdiv,
        Spin::Restricted,
        cfg,
    )
}

/// Gamma-point UHF stress on the dense-AFT J/K (`scf` the final stage of
/// `crate::uhf::gamma_uhf`; `exxdiv` its convention).
#[allow(clippy::too_many_arguments)]
pub fn gamma_uhf_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaStressConfig,
) -> Result<GammaStress, FerricError> {
    hf_stress(
        "gamma_uhf_stress",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Dense(eri),
        scf,
        exxdiv,
        Spin::Unrestricted,
        cfg,
    )
}

/// Gamma-point UHF stress with RS-GDF J/K.
#[allow(clippy::too_many_arguments)]
pub fn gamma_uhf_stress_rsgdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    fit: &RsGdfGradSource<'_>,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaStressConfig,
) -> Result<GammaStress, FerricError> {
    hf_stress(
        "gamma_uhf_stress_rsgdf",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Fit(*fit),
        scf,
        exxdiv,
        Spin::Unrestricted,
        cfg,
    )
}

/// Gamma-point RKS stress on the dense-AFT J/K. `dft` is the SAME config
/// the SCF ran with (functional, grid, AO threshold, exxdiv).
#[allow(clippy::too_many_arguments)]
pub fn gamma_rks_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    dft: &GammaRksConfig,
    cfg: &GammaStressConfig,
) -> Result<GammaStress, FerricError> {
    ks_stress(
        "gamma_rks_stress",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Dense(eri),
        scf,
        KsSpec::of_rks(dft),
        cfg,
    )
}

/// Gamma-point RKS stress with RS-GDF J/K.
#[allow(clippy::too_many_arguments)]
pub fn gamma_rks_stress_rsgdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    fit: &RsGdfGradSource<'_>,
    scf: &ScfResult,
    dft: &GammaRksConfig,
    cfg: &GammaStressConfig,
) -> Result<GammaStress, FerricError> {
    ks_stress(
        "gamma_rks_stress_rsgdf",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Fit(*fit),
        scf,
        KsSpec::of_rks(dft),
        cfg,
    )
}

/// Gamma-point UKS stress on the dense-AFT J/K (`scf` the final stage of
/// `crate::dft::gamma_uks`; `dft` its config).
#[allow(clippy::too_many_arguments)]
pub fn gamma_uks_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    dft: &GammaUksConfig,
    cfg: &GammaStressConfig,
) -> Result<GammaStress, FerricError> {
    ks_stress(
        "gamma_uks_stress",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Dense(eri),
        scf,
        KsSpec::of_uks(dft),
        cfg,
    )
}

/// Gamma-point UKS stress with RS-GDF J/K.
#[allow(clippy::too_many_arguments)]
pub fn gamma_uks_stress_rsgdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    fit: &RsGdfGradSource<'_>,
    scf: &ScfResult,
    dft: &GammaUksConfig,
    cfg: &GammaStressConfig,
) -> Result<GammaStress, FerricError> {
    ks_stress(
        "gamma_uks_stress_rsgdf",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Fit(*fit),
        scf,
        KsSpec::of_uks(dft),
        cfg,
    )
}

/// Gamma-point ROHF stress on the dense-AFT J/K (`scf` the final stage of
/// `crate::rohf::gamma_rohf`; `exxdiv` its convention). Refuses a result
/// whose ROHF orbital gradient exceeds `crate::grad::RO_ORBITAL_GRADIENT_TOL`.
#[allow(clippy::too_many_arguments)]
pub fn gamma_rohf_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaStressConfig,
) -> Result<GammaStress, FerricError> {
    hf_stress(
        "gamma_rohf_stress",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Dense(eri),
        scf,
        exxdiv,
        Spin::RestrictedOpen,
        cfg,
    )
}

/// Gamma-point ROKS stress on the dense-AFT J/K (`scf` the final stage of
/// `crate::rohf::gamma_roks`; `dft` its config). Refuses as
/// [`gamma_rohf_stress`].
#[allow(clippy::too_many_arguments)]
pub fn gamma_roks_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    dft: &GammaRoksConfig,
    cfg: &GammaStressConfig,
) -> Result<GammaStress, FerricError> {
    ks_stress(
        "gamma_roks_stress",
        cell,
        prep,
        hcore_cfg,
        hc,
        JkSource::Dense(eri),
        scf,
        KsSpec::of_roks(dft),
        cfg,
    )
}

/// What the KS entries need from the SCF config.
struct KsSpec<'a> {
    functional: &'a str,
    grid: &'a PeriodicGridConfig,
    ao_threshold: f64,
    exxdiv: ExxDiv,
    spin: Spin,
}

impl<'a> KsSpec<'a> {
    fn of_rks(d: &'a GammaRksConfig) -> Self {
        Self {
            functional: &d.functional,
            grid: &d.grid,
            ao_threshold: d.xc.ao_threshold,
            exxdiv: d.exxdiv,
            spin: Spin::Restricted,
        }
    }

    fn of_uks(d: &'a GammaUksConfig) -> Self {
        Self {
            functional: &d.functional,
            grid: &d.grid,
            ao_threshold: d.xc.ao_threshold,
            exxdiv: d.exxdiv,
            spin: Spin::Unrestricted,
        }
    }

    fn of_roks(d: &'a GammaRoksConfig) -> Self {
        Self {
            functional: &d.functional,
            grid: &d.grid,
            ao_threshold: d.xc.ao_threshold,
            exxdiv: d.exxdiv,
            spin: Spin::RestrictedOpen,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn hf_stress(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    jk: JkSource<'_>,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    spin: Spin,
    cfg: &GammaStressConfig,
) -> Result<GammaStress, FerricError> {
    check_inputs(who, cell, prep, hcore_cfg, hc, &jk, scf, spin)?;
    let unrestricted = spin != Spin::Restricted;
    let mut ledger = open_ledger(prep, cfg, unrestricted)?;
    let vm = madelung_for(cell, exxdiv)?;
    let n = prep.nbasis();
    let spins = if unrestricted {
        let (da, db) = spin_densities(who, scf)?;
        let (fa, fb) = unrestricted_focks(hc, &jk, &da, &db, 1.0, vm, None)?;
        if spin == Spin::RestrictedOpen {
            ro_gate(who, scf, &hc.s, &da, &db, &fa, &fb)?;
        }
        SpinSet::Unrestricted { da, db, fa, fb }
    } else {
        let d = scf.density_total.clone();
        let mut jm = Array2::<f64>::zeros((n, n));
        let mut km = Array2::<f64>::zeros((n, n));
        jk.build_j(&d, &mut jm)?;
        jk.build_k(vm, &d, &mut km)?;
        let f = &(&hc.h + &jm) - &(0.5 * &km);
        SpinSet::Restricted { d, f }
    };
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
        cfg,
        &mut ledger,
    )
}

#[allow(clippy::too_many_arguments)]
fn ks_stress(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    jk: JkSource<'_>,
    scf: &ScfResult,
    ks: KsSpec<'_>,
    cfg: &GammaStressConfig,
) -> Result<GammaStress, FerricError> {
    check_inputs(who, cell, prep, hcore_cfg, hc, &jk, scf, ks.spin)?;
    let (_, alpha) = resolve_periodic_functional(ks.functional)?;
    let unrestricted = ks.spin != Spin::Restricted;
    let mut ledger = open_ledger(prep, cfg, unrestricted)?;
    let vm = madelung_for(cell, ks.exxdiv)?;
    let n = prep.nbasis();
    if unrestricted {
        let (da, db) = spin_densities(who, scf)?;
        let mut xs = xc_stress(
            cell,
            prep,
            ks.functional,
            ks.grid,
            ks.ao_threshold,
            StressDensity::Polarized(&da, &db),
            cfg.mutation,
            &mut ledger,
        )?;
        let v_b = xs.v_b.take().ok_or_else(|| {
            FerricError::General(format!("{who}: the polarized XC pass returned no V_beta"))
        })?;
        let (fa, fb) = unrestricted_focks(hc, &jk, &da, &db, alpha, vm, Some((&xs.v_a, &v_b)))?;
        if ks.spin == Spin::RestrictedOpen {
            ro_gate(who, scf, &hc.s, &da, &db, &fa, &fb)?;
        }
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
            Some(xs),
            cfg,
            &mut ledger,
        )
    } else {
        let d = scf.density_total.clone();
        let xs = xc_stress(
            cell,
            prep,
            ks.functional,
            ks.grid,
            ks.ao_threshold,
            StressDensity::Closed(&d),
            cfg.mutation,
            &mut ledger,
        )?;
        let mut jm = Array2::<f64>::zeros((n, n));
        jk.build_j(&d, &mut jm)?;
        let mut f = &(&hc.h + &jm) + &xs.v_a;
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
            Some(xs),
            cfg,
            &mut ledger,
        )
    }
}

/// Ledger with the n×n and per-G scratch reservations of every entry point.
fn open_ledger(
    prep: &PreparedBasis,
    cfg: &GammaStressConfig,
    unrestricted: bool,
) -> Result<Ledger, FerricError> {
    let n = prep.nbasis();
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    // D, J, K, F, W, M, M_g0, DSD + per-G ERI scratch (P re/im, Z re/im,
    // D P D temporaries) + the nine per-G dρ; per spin D_σ, F_σ, V_σ and the
    // per-spin sandwiches.
    let per = if unrestricted { 8 * 30 } else { 8 * 18 };
    ledger.reserve(
        &format!("gamma stress n×n matrices + per-G scratch (n = {n})"),
        bytes_of((n * n) as u64, per),
    )?;
    Ok(ledger)
}

// ----------------------------------------------------------------- assembly

/// Everything but the XC pass, at the per-spin `(D_σ, F_σ)` (module doc).
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
    xc: Option<XcStress>,
    cfg: &GammaStressConfig,
    ledger: &mut Ledger,
) -> Result<GammaStress, FerricError> {
    let n = prep.nbasis();
    let mutation = cfg.mutation;
    let is = |m: StressMutation| mutation == Some(m);
    let mut parts = GammaStressParts::default();

    // --- Total density, W, M (the forces' M, per spin).
    let d = spins.total();
    let s_mat = &hc.s;
    let commutator = spins.commutator(s_mat);
    let w = spins.energy_weighted();
    let zs = cell.nuclear_charges();
    let ztot: f64 = zs.iter().sum();
    let omega = hcore_cfg.omega;
    let vol = cell.volume();
    let c0 = PI / (omega * omega * vol);
    let electrons = (&d * s_mat).sum();
    let mut m = -&w;
    m.scaled_add(c0 * ztot, &d);
    let dsd = spins.sandwich(s_mat);
    if alpha != 0.0 && vm != 0.0 && !is(StressMutation::MadelungS) {
        m.scaled_add(-alpha * vm, &dsd);
    }

    // --- RS-GDF: fitted densities and J3's G = 0 overlap weight M_g0.
    let fit = match jk {
        JkSource::Dense(_) => None,
        JkSource::Fit(src) => {
            let exch = spins.exch_terms();
            let fd = fit_densities(src.gdf, &d, &exch, alpha, false, ledger)?;
            let omega_f = src.gdf.stats().omega;
            let c0f = PI / (omega_f * omega_f * vol);
            let q: ndarray::Array1<f64> = aux_ft(src.aux, &[[0.0; 3]])?.column(0).mapv(|z| z.re);
            let yq = fd.y.t().dot(&q);
            let m_g0 = Array2::from_shape_fn((n, n), |(a, b)| -c0f * yq[a * n + b]);
            Some((*src, fd, m_g0))
        }
    };

    // --- dS, dT: per-image derivative blocks weighted by the pair vector.
    let images = hcore_pair_images(cell, prep, hcore_cfg.precision, ledger)?;
    let dims = prep.shell_dims().to_vec();
    let offs = prep.shell_offsets().to_vec();
    let centers: Vec<[f64; 3]> = prep.located_shells().iter().map(|s| s.center).collect();
    let nsh = prep.nshells();
    let drop_images = is(StressMutation::NoSrImages);
    {
        let mut eng_s = Engine::new_1e_deriv(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
        let mut eng_t = Engine::new_1e_deriv(ffi::OP_KINETIC, prep, ONE_E_ENGINE_PRECISION)?;
        for l in &images {
            for s1 in 0..nsh {
                for s2 in 0..nsh {
                    let (a, b) = (centers[s1], centers[s2]);
                    let li = if drop_images { [0.0; 3] } else { *l };
                    let rel = [
                        a[0] - b[0] - li[0],
                        a[1] - b[1] - li[1],
                        a[2] - b[2] - li[2],
                    ];
                    if let Some(blk) = eng_s.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        add_pair_virial(&mut parts.overlap, blk, &m, &dims, &offs, s1, s2, rel);
                        if let Some((_, _, mg)) = fit.as_ref() {
                            add_pair_virial(
                                &mut parts.fit_g0_overlap,
                                blk,
                                mg,
                                &dims,
                                &offs,
                                s1,
                                s2,
                                rel,
                            );
                        }
                    }
                    if let Some(blk) = eng_t.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        add_pair_virial(&mut parts.kinetic, blk, &d, &dims, &offs, s1, s2, rel);
                    }
                }
            }
        }
    }
    if is(StressMutation::NoPulay) {
        parts.overlap = ZERO3;
    }
    if is(StressMutation::NoG0) {
        parts.fit_g0_overlap = ZERO3;
    }

    // --- V_SR: Gaussian nuclei, erfc(ω), the energy's image/screen sets.
    let sr_cfg = PeriodicHcoreConfig {
        nucleus_exponent: cfg
            .nucleus_exponent
            .unwrap_or(GRAD_NUCLEUS_EXPONENT)
            .min(hcore_cfg.nucleus_exponent),
        ..*hcore_cfg
    };
    let (vsr, n_sr_triplets) = sr_attraction_strain(cell, prep, &sr_cfg, &d, drop_images, ledger)?;
    parts.vsr = vsr;

    let ft_terms = PairFtStrainTerms {
        centres: !is(StressMutation::FtNoCentres),
        g_shape: !is(StressMutation::GUnstrained),
    };
    let g_shape = ft_terms.g_shape;
    let with_volume = !is(StressMutation::NoVolume);

    // --- V_LR: −(2/Ω) Σ_half v_ω Re[ρ̄ S].
    let pos = cell.positions();
    let gcut_lr = lr_gcut(prep, omega, hcore_cfg.precision);
    ledger.reserve(
        &format!("gamma stress V_LR G list (|G| <= {gcut_lr:.3})"),
        gvector_list_bytes(cell, gcut_lr)?,
    )?;
    let gv_lr = half_gvectors(cell, gcut_lr)?;
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let mut e_vlr = 0.0_f64;
    let mut s_vlr = ZERO3;
    let mut n_chunks = pair_ft_strain_chunked(
        cell,
        prep,
        &gv_lr,
        0.1 * hcore_cfg.precision,
        chunk_budget,
        0,
        ft_terms,
        |_g0, gs, p, dp| {
            let fac = 2.0 / vol;
            for (g, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let v = 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp();
                let dvdg2 = -v * (1.0 / g2 + 1.0 / (4.0 * omega * omega));
                let (mut sre, mut sim) = (0.0_f64, 0.0_f64);
                for (z, r) in zs.iter().zip(&pos) {
                    let ph = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                    sre += z * ph.cos();
                    sim -= z * ph.sin();
                }
                let mut rho = Complex64::new(0.0, 0.0);
                let mut drho = [Complex64::new(0.0, 0.0); 9];
                for mu in 0..n {
                    for nu in 0..n {
                        let dmn = d[(mu, nu)];
                        if dmn == 0.0 {
                            continue;
                        }
                        rho += p[[mu, nu, g]] * dmn;
                        for (k, dk) in drho.iter_mut().enumerate() {
                            *dk += dp[k][[mu, nu, g]] * dmn;
                        }
                    }
                }
                let phi = -(rho.re * sre + rho.im * sim);
                e_vlr += fac * v * phi;
                for a in 0..3 {
                    for b in 0..3 {
                        let dz = drho[3 * a + b];
                        let dv = if g_shape {
                            dvdg2 * (-2.0 * gvec[a] * gvec[b])
                        } else {
                            0.0
                        };
                        let dphi = -(dz.re * sre + dz.im * sim);
                        s_vlr[a][b] += fac * (dv * phi + v * dphi);
                    }
                }
            }
            Ok(())
        },
    )?;
    if with_volume {
        add_into(&mut s_vlr, &diag(-e_vlr), 1.0);
    }
    parts.vlr = s_vlr;

    // --- h's G = 0 bookkeeping: E_c0 = c0 Z_tot N ∝ 1/Ω.
    if !is(StressMutation::NoC0Volume) {
        parts.c0_volume = diag(-c0 * ztot * electrons);
    }

    // --- Two-electron: dense AFT or RS-GDF.
    let mut e_eri = 0.0_f64;
    let n_g_eri: usize;
    let mut fit_diag: Option<RsGdfFitDiagnostics> = None;
    match (jk, fit) {
        (JkSource::Dense(eri), _) => {
            let r = dense_eri_strain(cell, prep, eri, spins, &d, alpha, ft_terms, ledger)?;
            let mut s = r.s;
            if with_volume {
                add_into(&mut s, &diag(-r.e), 1.0);
            }
            parts.eri = s;
            e_eri = r.e;
            n_g_eri = r.n_g;
            n_chunks += r.n_chunks;
        }
        (JkSource::Fit(_), Some((src, fd, _))) => {
            let terms = FitStrainTerms {
                pair: ft_terms,
                g_shape,
                volume: with_volume,
                images: !drop_images,
                aux_ft3: !is(StressMutation::NoAuxFt) && !is(StressMutation::NoAuxFt3),
                aux_ft2: !is(StressMutation::NoAuxFt),
            };
            let fs = fit_strain(src.gdf, cell, prep, src.aux, &fd.y, &fd.wm, terms, ledger)?;
            parts.fit_j3_sr = fs.j3_sr;
            parts.fit_j3_lr = fs.j3_lr;
            parts.fit_j2_sr = fs.j2_sr;
            parts.fit_j2_lr = fs.j2_lr;
            parts.fit_j3_g0_volume = fs.j3_g0_volume;
            parts.fit_j2_g0_volume = fs.j2_g0_volume;
            if is(StressMutation::NoMetric) {
                parts.fit_j2_sr = ZERO3;
                parts.fit_j2_lr = ZERO3;
                parts.fit_j2_g0_volume = ZERO3;
            }
            if is(StressMutation::NoJ2G0Vol) {
                parts.fit_j2_g0_volume = ZERO3;
            }
            if is(StressMutation::NoJ3G0Vol) {
                parts.fit_j3_g0_volume = ZERO3;
            }
            n_g_eri = fs.n_g_half;
            n_chunks += fs.n_chunks;
            let mut diag_f = fd.diag;
            diag_f.n_sr3_deriv = fs.n_sr3;
            diag_f.n_sr2_deriv = fs.n_sr2;
            diag_f.n_g_half = fs.n_g_half;
            diag_f.n_g_chunks = fs.n_chunks;
            fit_diag = Some(diag_f);
        }
        (JkSource::Fit(_), None) => {
            return Err(FerricError::General(
                "gamma stress: RS-GDF fitted densities missing (internal)".into(),
            ));
        }
    }

    // --- Madelung: −(α/2) Σ_σ tr(D_σ S D_σ S) dv_M/dε.
    let vm_strain = madelung_strain(cell)?;
    if alpha != 0.0 && vm != 0.0 && !is(StressMutation::NoMadelung) {
        let t_m = (&dsd * s_mat).sum();
        parts.madelung = scaled(&vm_strain, -0.5 * alpha * t_m);
    }

    // --- Ewald E_nn (the energy's ω; the result is ω-independent).
    let nn = ewald_nuclear_strain(cell, default_ewald_omega(cell), DEFAULT_EWALD_PRECISION)?;
    parts.nn_sr = nn.sr;
    if !is(StressMutation::NoEwaldLr) {
        parts.nn_lr = nn.lr;
    }
    if !is(StressMutation::NoEwaldBg) {
        parts.nn_background = nn.background;
    }

    // --- XC.
    let (e_xc, n_grid_points) = match xc {
        Some(x) => {
            parts.xc_ao = x.ao;
            parts.xc_weight = x.weight;
            (Some(x.e_xc), x.npts)
        }
        None => (None, 0),
    };

    let de_deps = parts.total();
    if de_deps.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FerricError::General(
            "gamma stress: non-finite strain derivative".into(),
        ));
    }
    let sigma = scaled(&de_deps, 1.0 / vol);
    ferric_core::memory::warn_if_rss_over("ferric-pbc gamma stress", ledger.budget(), 1.1);
    Ok(GammaStress {
        de_deps,
        sigma,
        parts,
        volume: vol,
        commutator,
        madelung: vm,
        madelung_strain: vm_strain,
        electrons,
        exact_exchange_fraction: alpha,
        e_xc,
        n_grid_points,
        e_vlr,
        e_eri,
        n_sr_triplets,
        n_images: images.len(),
        n_g_lr: gv_lr.len(),
        n_g_eri,
        n_chunks,
        budget_bytes: ledger.budget(),
        fit: fit_diag,
    })
}

/// `out[a][b] += Σ_mn w_mn ½(∂_bra − ∂_ket)_a ⟨m|op|n_L⟩ · rel_b` for one
/// shifted 1e derivative block (`[bra xyz, ket xyz]`, each `n1 × n2`),
/// `rel = A − B − L`. With exact translation invariance `½(bra − ket) = bra`.
#[allow(clippy::too_many_arguments)]
fn add_pair_virial(
    out: &mut Mat3,
    blk: &[f64],
    w: &Array2<f64>,
    dims: &[usize],
    offs: &[usize],
    s1: usize,
    s2: usize,
    rel: [f64; 3],
) {
    let (n1, n2) = (dims[s1], dims[s2]);
    let bs = n1 * n2;
    let mut g = [0.0_f64; 3];
    for i in 0..n1 {
        for j in 0..n2 {
            let wv = w[(offs[s1] + i, offs[s2] + j)];
            if wv == 0.0 {
                continue;
            }
            let idx = i * n2 + j;
            for (c, gc) in g.iter_mut().enumerate() {
                *gc += wv * 0.5 * (blk[c * bs + idx] - blk[(3 + c) * bs + idx]);
            }
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            out[a][b] += g[a] * rel[b];
        }
    }
}

/// Dense-AFT ERI strain: `(Σ_G [dv Φ + v dΦ] (2/Ω), E_2e, chunks, |G|)`
/// without the volume term.
struct DenseEriStrain {
    s: Mat3,
    e: f64,
    n_chunks: usize,
    n_g: usize,
}

#[allow(clippy::too_many_arguments)]
fn dense_eri_strain(
    cell: &Cell,
    prep: &PreparedBasis,
    eri: &DenseAftEri,
    spins: &SpinSet,
    d: &Array2<f64>,
    alpha: f64,
    terms: PairFtStrainTerms,
    ledger: &mut Ledger,
) -> Result<DenseEriStrain, FerricError> {
    let n = prep.nbasis();
    let vol = cell.volume();
    let gcut = eri.gcut();
    ledger.reserve(
        &format!("gamma stress ERI G list (|G| <= {gcut:.3})"),
        gvector_list_bytes(cell, gcut)?,
    )?;
    let gv = half_gvectors(cell, gcut)?;
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let mut s_acc = ZERO3;
    let mut e = 0.0_f64;
    let n_chunks = pair_ft_strain_chunked(
        cell,
        prep,
        &gv,
        eri.pair_thresh(),
        chunk_budget,
        0,
        terms,
        |_g0, gs, p, dp| {
            let fac = 2.0 / vol;
            for (g, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let v = 4.0 * PI / g2;
                let dvdg2 = -v / g2;
                let pg = p.slice(s![.., .., g]);
                let pre = pg.mapv(|z| z.re);
                let pim = pg.mapv(|z| z.im);
                let rho_re = (d * &pre).sum();
                let rho_im = (d * &pim).sum();
                // Z = ½ D ρ − (α/2) Σ_σ D_σ P D_σ
                let (mut zr, mut zi) = if alpha != 0.0 {
                    (
                        spins.sandwich(&pre) * (-0.5 * alpha),
                        spins.sandwich(&pim) * (-0.5 * alpha),
                    )
                } else {
                    (Array2::<f64>::zeros((n, n)), Array2::<f64>::zeros((n, n)))
                };
                zr.scaled_add(0.5 * rho_re, d);
                zi.scaled_add(0.5 * rho_im, d);
                let phi = (&pre * &zr).sum() + (&pim * &zi).sum();
                e += fac * v * phi;
                for a in 0..3 {
                    for b in 0..3 {
                        let dpk = &dp[3 * a + b];
                        let mut dphi = 0.0_f64;
                        for mu in 0..n {
                            for nu in 0..n {
                                let z = dpk[[mu, nu, g]];
                                dphi += z.re * zr[(mu, nu)] + z.im * zi[(mu, nu)];
                            }
                        }
                        dphi *= 2.0;
                        let dv = if terms.g_shape {
                            dvdg2 * (-2.0 * gvec[a] * gvec[b])
                        } else {
                            0.0
                        };
                        s_acc[a][b] += fac * (dv * phi + v * dphi);
                    }
                }
            }
            Ok(())
        },
    )?;
    Ok(DenseEriStrain {
        s: s_acc,
        e,
        n_chunks,
        n_g: gv.len(),
    })
}

// ----------------------------------------------------------------------- XC

/// The density handed to the XC stress.
#[derive(Clone, Copy)]
enum StressDensity<'a> {
    Closed(&'a Array2<f64>),
    Polarized(&'a Array2<f64>, &'a Array2<f64>),
}

/// The XC pieces (computed before the Fock matrices are rebuilt: they need
/// `V_σ`).
struct XcStress {
    e_xc: f64,
    v_a: Array2<f64>,
    v_b: Option<Array2<f64>>,
    ao: Mat3,
    weight: Mat3,
    npts: usize,
}

struct XcChunk {
    e: f64,
    v_a: Array2<f64>,
    v_b: Option<Array2<f64>>,
    ao: Mat3,
    weight: Mat3,
}

/// One spin channel: `D_σ`, the gated `v_ρσ` and the gated GGA vector `c^σ`
/// (`3 × np`; `None` for LDA) — the forces' kernel terms.
struct SpinTerm<'t> {
    d: &'t Array2<f64>,
    vr: Vec<f64>,
    c: Option<Array2<f64>>,
}

/// `E_xc`, `V_σ` and the XC strain pieces on the rebuilt periodic grid
/// (module doc, item 8):
///
/// ```text
/// AO:     Σ_σ Σ_g w_g [ v_ρσ dρ_σ/dε_ab + Σ_k c^σ_k d(∂_kρ_σ)/dε_ab ]
///         dρ_σ/dε_ab       = 2 Σ_μ m1[a,b]_μ (D_σ χ)_μ
///         d(∂_kρ_σ)/dε_ab  = 2 Σ_μ ( m2[k,a,b]_μ (D_σ χ)_μ + m1[a,b]_μ (D_σ ∂_kχ)_μ )
/// weight: Σ_g e(r_g) dW_g/dε_ab
/// ```
///
/// (`m1`, `m2` the image-resolved AO first moments of `LatticeAoHess::
/// eval_strain`, anchored at the home atom; `c^σ` as in the forces:
/// `2 v_σσσ ∇ρ_σ + v_σαβ ∇ρ_σ′`, closed shell `2 v_σ ∇ρ`.)
#[allow(clippy::too_many_arguments)]
fn xc_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    functional: &str,
    grid_cfg: &PeriodicGridConfig,
    ao_threshold: f64,
    dens_in: StressDensity<'_>,
    mutation: Option<StressMutation>,
    ledger: &mut Ledger,
) -> Result<XcStress, FerricError> {
    let (xc_def, _) = resolve_periodic_functional(functional)?;
    let xc_pol = match dens_in {
        StressDensity::Closed(_) => None,
        StressDensity::Polarized(..) => {
            Some(xc_def_from_name_nspin(functional, 2).map_err(|e| {
                FerricError::General(format!(
                    "periodic XC stress: functional {functional:?} (spin-polarized): {e:?}"
                ))
            })?)
        }
    };
    let gga = xc_def
        .funcs
        .iter()
        .any(|f| !matches!(f.family(), FunctionalFamily::Lda));
    let nbf = prep.nbasis();
    let point_fixed = mutation == Some(StressMutation::XcPointFixed);

    let sg = build_strain_grid(cell, grid_cfg, ledger)?;
    let ev = LatticeAoHess::new(cell, prep.basis_set(), &sg.points, ao_threshold, ledger)?;
    if ev.nbf() != nbf {
        return Err(FerricError::General(format!(
            "periodic XC stress: lattice AO count {} != prepared basis {nbf}",
            ev.nbf()
        )));
    }
    let chunks = LatticeAoHess::chunks(&sg.points, XC_STRESS_CHUNK);
    let par = rayon::current_num_threads().max(1);
    // Per concurrently live chunk: χ, ∇χ, the 9 first-moment planes, the 27
    // Hessian-moment planes (GGA), per spin D·χ and D·∇χ (≤ 8), V (≤ 2 n²).
    let planes = 13 + if gga { 27 } else { 0 } + 8;
    let per_chunk = bytes_of((nbf * XC_STRESS_CHUNK) as u64, planes * 8)
        .saturating_add(bytes_of((nbf * nbf) as u64, 2 * 8));
    ledger.reserve(
        &format!(
            "periodic XC stress AO moment chunks ({par} x {XC_STRESS_CHUNK} points, nbf = {nbf})"
        ),
        per_chunk.saturating_mul(par),
    )?;

    let run_chunk = |idx: &Vec<usize>| -> Result<XcChunk, FerricError> {
        let pts: Vec<GridPoint> = idx.iter().map(|&i| sg.points[i]).collect();
        let xyz: Vec<[f64; 3]> = pts.iter().map(|g| g.xyz).collect();
        let anchors: Vec<[f64; 3]> = if point_fixed {
            xyz.clone()
        } else {
            idx.iter().map(|&i| sg.anchors[i]).collect()
        };
        let mom = ev.eval_strain(&xyz, &anchors, gga)?;
        let (chi, dchi) = (&mom.chi, &mom.dchi);
        let np = pts.len();
        let (e, v_a, v_b, eps, terms) = match dens_in {
            StressDensity::Closed(d) => {
                let dens = eval_density_closed(d, chi, dchi);
                let k = closed_kernel(&dens, None, &xc_def);
                let (e, v) = semilocal_vxc_closed(&pts, chi, dchi, &dens, None, &xc_def);
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
                let eps: Vec<f64> = (0..np).map(|g| dens.rho[g] * k.exc[g]).collect();
                (e, v, None, eps, vec![SpinTerm { d, vr, c }])
            }
            StressDensity::Polarized(da, db) => {
                let xp = xc_pol.as_ref().expect("polarized XcDef resolved above");
                let dens = eval_density_uks(da, db, chi, dchi);
                let k = polarized_kernel(&dens, None, xp);
                let (e, va, vb) = semilocal_vxc_polarized(&pts, chi, dchi, &dens, None, xp);
                let gate = |rho: &ndarray::Array1<f64>, v: &ndarray::Array1<f64>| -> Vec<f64> {
                    (0..np)
                        .map(|g| if rho[g] > DENSITY_FLOOR { v[g] } else { 0.0 })
                        .collect()
                };
                let vr_a = gate(&dens.rho_a, &k.vrho_a);
                let vr_b = gate(&dens.rho_b, &k.vrho_b);
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
                let eps: Vec<f64> = (0..np)
                    .map(|g| (dens.rho_a[g] + dens.rho_b[g]) * k.exc[g])
                    .collect();
                (
                    e,
                    va,
                    Some(vb),
                    eps,
                    vec![
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
                )
            }
        };
        let mut ao = ZERO3;
        for SpinTerm { d: dm, vr, c } in &terms {
            let dchi_m = dm.dot(chi);
            let dchi_k: Option<[Array2<f64>; 3]> = c
                .as_ref()
                .map(|_| std::array::from_fn(|k| dm.dot(&dchi.index_axis(Axis(0), k))));
            for g in 0..np {
                let wg = pts[g].weight;
                for a in 0..3 {
                    for b in 0..3 {
                        let mut drho = 0.0_f64;
                        for mu in 0..nbf {
                            drho += mom.m1[(a, b, mu, g)] * dchi_m[(mu, g)];
                        }
                        let mut t = vr[g] * 2.0 * drho;
                        if let (Some(c), Some(dk), Some(m2)) =
                            (c.as_ref(), dchi_k.as_ref(), mom.m2.as_ref())
                        {
                            for k in 0..3 {
                                let mut dg = 0.0_f64;
                                for mu in 0..nbf {
                                    dg += m2[(3 * k + a, b, mu, g)] * dchi_m[(mu, g)]
                                        + mom.m1[(a, b, mu, g)] * dk[k][(mu, g)];
                                }
                                t += c[(k, g)] * 2.0 * dg;
                            }
                        }
                        ao[a][b] += wg * t;
                    }
                }
            }
        }
        let mut weight = ZERO3;
        for (g, &gi) in idx.iter().enumerate() {
            add_into(&mut weight, &sg.dweight[gi], eps[g]);
        }
        Ok(XcChunk {
            e,
            v_a,
            v_b,
            ao,
            weight,
        })
    };

    // Concurrency-sized batches, reduced in chunk order (thread-count
    // independent sums).
    let mut e_xc = 0.0;
    let mut v_a = Array2::<f64>::zeros((nbf, nbf));
    let mut v_b =
        matches!(dens_in, StressDensity::Polarized(..)).then(|| Array2::<f64>::zeros((nbf, nbf)));
    let mut ao = ZERO3;
    let mut weight = ZERO3;
    for batch in chunks.chunks(par) {
        let outs: Vec<Result<XcChunk, FerricError>> = batch.par_iter().map(&run_chunk).collect();
        for out in outs {
            let c = out?;
            e_xc += c.e;
            v_a += &c.v_a;
            if let (Some(acc), Some(vb)) = (v_b.as_mut(), c.v_b.as_ref()) {
                *acc += vb;
            }
            add_into(&mut ao, &c.ao, 1.0);
            add_into(&mut weight, &c.weight, 1.0);
        }
    }
    if !e_xc.is_finite()
        || v_a.iter().any(|x| !x.is_finite())
        || ao
            .iter()
            .chain(weight.iter())
            .flatten()
            .any(|x| !x.is_finite())
    {
        return Err(FerricError::General(format!(
            "periodic XC stress: non-finite E_xc/V_xc/stress for {functional}"
        )));
    }
    match mutation {
        Some(StressMutation::XcNoAo) => ao = ZERO3,
        Some(StressMutation::XcNoWeight) => weight = ZERO3,
        _ => {}
    }
    Ok(XcStress {
        e_xc,
        v_a,
        v_b,
        ao,
        weight,
        npts: sg.points.len(),
    })
}
