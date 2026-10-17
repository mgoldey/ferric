//! Analytic stress tensor of the k-point RHF and UHF (Gamma-centred or
//! shifted meshes) on the dense pure-AFT J/K or the RS-GDF J/K (unsplit or
//! range-split) — the strain twin of [`crate::kgrad`], built from the same
//! phase-folded weights.
//!
//! # Convention (the Gamma stress's, [`crate::stress`])
//!
//! ```text
//! σ_ab = (1/Ω) dE/dε_ab,   a_i → (1 + ε) a_i,  R_A → (1 + ε) R_A,
//! G = Σ n_i b_i → (1 + ε)^{−T} G  (Miller index FIXED),  k = Σ f_i b_i (f_i FIXED)
//! ```
//!
//! # Why no "k strain" term
//!
//! A k-point is held at FIXED fractional coordinates, `k(ε) = (1 + ε)^{−T} k₀`
//! (exactly what `KPointMesh::gamma_centred(&cell.strained(ε), n)` builds from
//! the same integer numerators). The Bloch phase `e^{ik·L}` depends only on
//! those numerators and on the integer lattice coordinates of `L`, so it is
//! strain-INVARIANT, and so is the residue binning of the images. Every
//! per-k weight the forces build (`Mt(L)`, `Δ(L)`, `Yt_r`, ...) is therefore
//! reused unchanged: the stress multiplies the same per-image / per-triplet
//! derivative by the PAIR VECTOR instead of by a unit displacement, exactly
//! as the Gamma stress does with the Gamma force walks.
//!
//! # Terms (per cell; `D(k)`, `F(k)` fixed)
//!
//! 1. overlap / kinetic virial: `Σ_L Σ_mn w[r(L)]_{mn} ½(∂_bra − ∂_ket)⟨m|op|n_L⟩ (A − B − L)_b`,
//!    weights `w_s` (`M(k) = −W + c0 Z_tot D − v_M Σ_s D_s S D_s`) and `w_t`
//!    of [`crate::kgrad`].
//! 2. `V_SR`: the SR attraction walk's per-triplet `∂_A (A − X)_b + ∂_B′ (B′ − X)_b`
//!    with the force's image weight `Re Δ(L)`.
//! 3. `V_LR`: the full G sphere, `E_LR = −(1/Ω) Σ_G v_ω Re[S̄ ρ]`, with the
//!    RESIDUE-resolved pair-FT strain kernel
//!    ([`crate::pair_ft::residues::pair_ft_strain_residues_chunked`]); `−δ E_LR`
//!    volume term.
//! 4. `c0` volume `−δ c0 Z_tot N`, `N = (1/N_k) Σ_k tr[D(k) S(k)]`.
//! 5. Ewald `E_nn` ([`crate::ewald::ewald_nuclear_strain`]).
//! 6. Dense-AFT J (q = 0) and K (every q, `K = G + q`): per `K` the same
//!    `(1/Ω)[dv Φ + v dΦ]` as the Gamma ERI with `dΦ` through the residue
//!    strain kernel and the force's `Yt_r`; `−δ E_2e` volume term.
//! 7. RS-GDF J/K ([`KStressJk::RsGdf`]): the SR 3-centre and
//!    metric strain walks with the force's per-`(L, T)` / per-`T` phase-folded
//!    weights (`Re Zbin[r_L, r_T]`, `Re Σ_q e^{iq·t} Wm(q)ᵀ`) times the pair
//!    vector; the LR pass per q class on the full `{G + q}` set with the
//!    complex per-residue weights `Zr`, the residue pair-FT strain kernel and
//!    the aux-FT strain; `−δ E_LR` volume terms for `J3` and `J2`; the `J3`
//!    G = 0 term through `dS` (`M_g0(k)`, in the overlap pass) and its volume
//!    term `−δ E_g0`; the strain-only `J2` G = 0 volume term `+δ c0 qᵀ Wm(0) q`
//!    (constant under atom motion, so the forces never see it). All at FIXED
//!    `Z`, `Wm` (the k force's Loewner metric weights). A RANGE-SPLIT build
//!    follows the same partition (`rsgdf::split::kstrain`): the SR walks over
//!    the kept calls / compact pairs, the moved blocks with the kernel
//!    `v_SR(|G + q|)` and its slope `dv_SR/dK² = −v_SR/K² + v_LR/4ω²`, a
//!    smooth-pair pass (`conj(v_SR X_c)` on the smooth orbital pieces), the
//!    volume terms with the smooth-pair energy and the kept `(q_c, S − S_ss)`
//!    G = 0 inputs, and the smooth-piece overlap virial `dS_ss/dε` (image
//!    resolved, residue-weighted by `M_g0`) SUBTRACTED from the G = 0 overlap
//!    term.
//! 8. Madelung (`exxdiv = ewald`): `−½ (1/N_k) Σ_s Σ_k tr[D_s S D_s S] dv_M/dε`
//!    with `dv_M/dε` the Ewald strain of the `diag(N)` SUPERCELL lattice (`v_M`
//!    is the mesh constant, [`KPointMesh::madelung`]); its `S` part is in the
//!    overlap weights.
//!
//! # Scope
//!
//! Dense pure-AFT and RS-GDF (unsplit or range-split) J/K, HF, RHF and UHF,
//! with or without a scalar periodic ECP. A range-separated dense source is
//! refused with a typed error naming the missing piece. The SR column rotation is
//! differentiated: when the energy builds ran the rotated SR walks (hcore and
//! fit) the strain walks run on the rotated shells with `Tᵀ W T` residue
//! weights, so the stress follows the walk the SCF's energy ran (the config
//! must agree with the build, [`crate::kgrad`]'s refusals apply). The SR
//! derivative walks use the Gaussian-nucleus exponent
//! `min(GRAD_NUCLEUS_EXPONENT, cfg)`, the energy walk's own 1e16 by default,
//! so the only gap between either derivative and the derivative of the
//! SCREENED energy is the screen itself (stress rotated vs unrotated 6.5e-9 Ha
//! at hcore precision 1e-9, shrinking with the screen).
//!
//! ECP ([`KStressParts::ecp`]): the Gamma ECP strain walk
//! ([`crate::ecp::periodic_ecp_strain`]) with the SR walk's image weight
//! `W_L = w_d[residue(L)]` in place of the density (module doc of
//! [`crate::ecp`], "k-points"); the phases and residues are strain-invariant,
//! so there is no k term. The ECP walk is driven by the BASIS, then
//! cross-checked against `hk.v_ecp` (kept-triple count, presence): an `hk`
//! whose `v_ecp` was cleared while the basis carries an ECP, or the reverse,
//! is refused. The Fock guard rebuilds `F` from `hk.h`, which carries `V_ECP`,
//! so an SCF run without the ECP is refused by the guard.
//!
//! Range split: the config must agree with the build about whether a split ran
//! (typed refusal otherwise) and `lambda > 1` is refused (moved blocks whose FT
//! need not converge in the sphere, and a K set filtered in the strained frame
//! makes the energy piecewise). A config whose `lambda`, `omega` or `precision`
//! differs from the build's changes the stress only at the split tolerance
//! (~1e-11 for `lambda <= 1`; `lambda = 2` is ~1e-8, at the Fock-guard bar).
//! Tested: single-primitive aux shells (a contracted aux shell split partway,
//! and a split that moves part of a contracted orbital shell, are untested at
//! k), s/p/d aux, H2 and H3 STO-3G; the pure-d aux supercell anchor carries a
//! per-part floor of ~1e-10 in the two J2 parts at the default `lindep` (the
//! fit's sensitivity along one near-null metric mode; it collapses to ~3e-12
//! when that mode is dropped, see
//! `d_aux_part_floor_collapses_when_the_near_null_mode_is_dropped`).
//!
//! # Input guards
//!
//! The stress rebuilds `F = h + J − αK` from the density and the J/K source
//! (dense kernels or the RS-GDF build) and refuses (typed [`FerricError`]) unless it reproduces the SCF's
//! stored Fock to [`FOCK_CONSISTENCY_BAR`] (defined in [`crate::kgrad`], which applies the same guard): a KS result (no `V_xc` in the
//! rebuilt `F`), an SCF run at another `exxdiv`, or an RS-GDF source on
//! another aux basis than the SCF's would otherwise give a finite but wrong
//! σ (the RS-GDF energy is ω-independent: a fit at another ω reproduces the
//! Fock to 1e-14 and the stress to 2.6e-12 on H2 1×1×3, so another ω is not
//! an inconsistency). A dense source built with a range-separated K kernel is
//! refused outright. Anchored meshes: Gamma-centred (explicit supercell and
//! FD) and Monkhorst-Pack (FD only; an even MP mesh is a twisted supercell).
//!
//! # Supercell mapping
//!
//! For a Gamma-centred mesh the k stress per cell equals the Gamma stress of
//! the explicit `diag(N)` supercell divided by `N` in `dE/dε` (`E_cell =
//! E_sc/N`) and EQUAL in `σ` (`Ω_sc = N Ω`).
//!
//! # Anchor blind spots
//!
//! As [`crate::kgrad`]: on TRIM-only meshes (every `n_i <= 2`) [`KStressMutation::ConjD`]
//! and [`KStressMutation::WrongQ`] are identities; [`KStressMutation::GammaMadelung`]
//! and [`KStressMutation::GammaMadelungStrain`] are blind for `exxdiv = none`
//! and at 1×1×1. The volume-only mutants are diagonal by construction.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::ecp::{periodic_ecp_strain_weighted, EcpStrainMutation, EcpWeights};
use crate::ewald::{
    default_ewald_omega, ewald_nuclear_strain, madelung_constant, madelung_strain,
    DEFAULT_EWALD_PRECISION,
};
use crate::grad::hcore_rotation_built;
use crate::grad::GRAD_NUCLEUS_EXPONENT;
use crate::hcore::kpoint::PeriodicHcoreK;
use crate::hcore::{
    gvector_list_bytes, hcore_pair_images, lr_gcut, sr_attraction_deriv_visit, PeriodicHcoreConfig,
    G_CHUNK_BYTES, ONE_E_ENGINE_PRECISION,
};
use crate::kdense_aft::KDenseAftEri;
pub use crate::kgrad::FOCK_CONSISTENCY_BAR;
use crate::kgrad::{
    check_fock, check_inputs, cr, ecp_presence, fold_complex, fold_real, herm, jk_build,
    residue_trace, scaled_exch_terms, KFitChecks, KGradJk, KRsGdfGradSource, KSpins,
};
use crate::kpts::{lattice_coords, KPointMesh};
use crate::kscf::KScfResult;
use crate::kuscf::KUScfResult;
use crate::lattice::Cell;
use crate::pair_ft::residues::{pair_ft_strain_residues_chunked, residue_coords, residue_index};
use crate::pair_ft::PairFtStrainTerms;
use crate::rsgdf::kpoint::kderiv::{
    kpoint_fit_strain, rotate_hcore_weights, KFitMutation, KFitStrain, KFitStrainTerms, KRotKind,
};
use crate::stress::{add_into, add_pair_virial, diag, scaled, Mat3, ZERO3};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ndarray::{Array2, Array3};
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// Remedy text of the stress Fock guard.
const STRESS_REMEDY: &str =
    "this stress is Hartree-Fock only: a KS result has no k-point stress entry point yet";

/// Deliberate defects for the mutation tests (`tests/pbc_kstress.rs`; the
/// Gamma [`crate::stress::StressMutation`] and [`crate::kgrad::KGradMutation`]
/// seams, same names where the defect is the same). Never set in production.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KStressMutation {
    /// Bloch phase dropped on every derivative weight (`Mt`, `Δ`, `Yt`; the
    /// energy-side `ρ` and `P^{k'}` keep theirs).
    NoPhase,
    /// `D_s(k)`, `F_s(k)` conjugated inside the stress. IDENTITY on
    /// TRIM-only meshes.
    ConjD,
    /// Exchange derivative pairs `k'` with `k = k' + q` instead of `k' − q`.
    /// IDENTITY on TRIM-only meshes.
    WrongQ,
    /// Missing `1/N_k` in the overlap / kinetic weights (blind at 1×1×1).
    NoInvNk,
    /// Primitive-cell Gamma `v_M` in the overlap term `−v_M Σ_s D_s S D_s`
    /// (blind for `exxdiv = none` and at 1×1×1).
    GammaMadelung,
    /// Primitive-cell `dv_M/dε` in the Madelung strain (blind for
    /// `exxdiv = none` and at 1×1×1).
    GammaMadelungStrain,
    /// Drop `Σ M dS` (overlap / Pulay term).
    NoPulay,
    /// Drop `c0 Z_tot D` from `M` (the c0 overlap part).
    NoC0Overlap,
    /// Drop `−v_M Σ_s D_s S D_s` from `M`.
    MadelungS,
    /// Drop `dv_M/dε` (exxdiv = ewald only).
    NoMadelung,
    /// Pair FT: drop the basis-centre (pair-vector) part of `dP`.
    FtNoCentres,
    /// G treated as unstrained: drop the G parts of `dP` and `dv`.
    GUnstrained,
    /// Drop the `−δ E_LR` volume terms (V_LR, J, K; diagonal only).
    NoVolume,
    /// UHF only (identity for RHF): the exchange pairs `D_α` with `D_β`
    /// (`D_α P^H D_β + D_β P^H D_α`) instead of each spin with itself.
    CrossSpinK,
    /// UHF only (identity for RHF): the Pulay weight `W` built with the
    /// opposite spin's Fock (`D_α F_β D_α + D_β F_α D_β`).
    SwapSpinFock,
    /// Drop `−δ c0 Z_tot N` (diagonal only).
    NoC0Volume,
    /// SR / overlap virials with the image translations dropped from the
    /// pair vectors.
    NoSrImages,
    /// Drop the reciprocal part of the Ewald `E_nn` strain.
    NoEwaldLr,
    /// Drop the Ewald background volume term (diagonal only).
    NoEwaldBg,
    /// RS-GDF: drop `Σ Wm dJ2` (SR, LR and its G = 0 volume term).
    NoMetric,
    /// RS-GDF: aux FT strain-free in J3 AND J2 (nearly blind on an accurate
    /// aux set: it errs by the fit error only; see [`KStressMutation::NoAuxFt3`]).
    NoAuxFt,
    /// RS-GDF: aux FT strain-free in J3 only.
    NoAuxFt3,
    /// RS-GDF: drop the strain-only J2 G = 0 volume term `+δ c0 qᵀ Wm(0) q`
    /// (diagonal only).
    NoJ2G0Vol,
    /// RS-GDF: drop the J3 G = 0 volume term `−δ E_g0` (diagonal only).
    NoJ3G0Vol,
    /// RS-GDF: drop the J3 G = 0 overlap term `Σ M_g0 dS`.
    NoG0,
    /// RS-GDF: `e^{+iq·T}` on the SR aux images of `Z` (the energy has `e^{−iq·T}`).
    FitAuxPhase,
    /// RS-GDF: the LR `K` set built as `G − q` (IDENTITY at q = 0, so blind at
    /// 1×1×1; the Z weight of class q then meets the kernel of class −q).
    FitLrWrongQ,
    /// RS-GDF: the kept-dropped Loewner entries zeroed (identity for an exact
    /// null vector, sharp for a near-dependent dropped mode).
    FitNoLoewnerKd,
    /// RS-GDF, UHF only (identity for RHF): the fitted exchange `Z` pairs
    /// `D_α` with `D_β` (`D_α J3^H D_β + D_β J3^H D_α`).
    FitCrossSpin,
    /// RS-GDF range split: drop the SR kernel-weight strain `dv_SR(|K|)/dε`
    /// of the moved G-space blocks.
    SplitNoSrKernelStrain,
    /// RS-GDF range split: drop the smooth-pair `dP̄_ss` / `dXc` strain terms.
    SplitNoSmoothPair,
    /// RS-GDF range split: the unsplit G = 0 inputs `(q, S)` in the volume
    /// terms and `M_g0`, and no `dS_ss` virial.
    SplitFullG0,
    /// RS-GDF range split: omit the smooth-piece overlap virial `dS_ss/dε`
    /// of the `J3` G = 0 term.
    SplitNoSmoothOverlap,
    /// RS-GDF range split: the sign of the SR kernel slope `dv_SR/dK²` flipped.
    SplitFlipSrSlope,
    /// RS-GDF range split: the smooth-pair LR energy left out of the
    /// `−δ E_LR` volume term (diagonal only).
    SplitNoSmoothVolume,
    /// Hybrid only (identity at `a = 1` and `a = 0`): the Madelung M-term
    /// and its strain WITHOUT the exchange fraction.
    MadelungUnscaled,
    /// Hybrid RS-GDF only: the fit's exchange pieces dropped (a Coulomb-only
    /// fit strain; identity for `a = 0`).
    FitExchangeDropped,
    /// ECP: drop the ECP strain term altogether.
    NoEcp,
    /// Column rotation: no `T` in the forward transform of the SR weights
    /// (rotated walks, unrotated weights; hcore and fit).
    RotDropT,
    /// Column rotation: `T` where `Tᵀ` belongs.
    RotUseT,
    /// Column rotation: only the first AO index transformed.
    RotOneSided,
    /// Column rotation: off-diagonal `T_ks` with the wrong sign.
    RotFlipSign,
    /// Column rotation (fit): the weight's AO pair read transposed.
    RotSwapPair,
    /// Column rotation (fit): transformed residue bins in reversed order.
    RotFitWrongResidue,
    /// Column rotation (hcore): transformed residue weights in reversed order.
    RotHcoreWrongResidue,
    /// Column rotation: the strain walks unrotated although the energy ran
    /// rotated (the behaviour before the k rotation derivative existed).
    RotOff,
    /// Column rotation: only the hcore SR strain walk unrotated.
    RotHcoreOff,
    /// ECP: drop the ECP-centre virial `Σ ∂_C X_C`.
    EcpNoCentre,
    /// ECP: the centre virial with the wrong sign.
    EcpCentreSign,
    /// ECP: the ket derivative weighted by the home-cell position (the
    /// orbital image `L` dropped from the pair vector).
    EcpNoKetImage,
    /// ECP: a unit vector `δ_ab` in place of the pair vector (a force, not a
    /// strain; diagonal only).
    EcpUnitPairVector,
    /// ECP: the image weight read at the residue of `−L` (transposed weight).
    EcpWrongResidue,
    /// ECP: the image weight's residue taken modulo the mesh counts instead
    /// of the residue moduli (identity on a Gamma-centred mesh).
    EcpMeshCountModuli,
}

/// Settings for the k-point stress entry points.
#[derive(Debug, Clone, Copy, Default)]
pub struct KStressConfig {
    /// Memory budget (bytes); `None` = ferric's unified budget.
    pub budget_bytes: Option<usize>,
    /// TEST ONLY: a deliberate defect ([`KStressMutation`]).
    #[doc(hidden)]
    pub mutation: Option<KStressMutation>,
    /// Gaussian-nucleus exponent for the SR attraction strain only (`None` =
    /// [`GRAD_NUCLEUS_EXPONENT`], never tighter than the energy's).
    pub nucleus_exponent: Option<f64>,
}

/// Bloch phases per `(k, residue)`: the energy's (`true_`) and the
/// derivative-side ones (`d`; all 1 under [`KStressMutation::NoPhase`]).
pub(crate) struct Phases {
    pub(crate) moduli: [usize; 3],
    pub(crate) true_: Vec<Vec<C64>>,
    pub(crate) d: Vec<Vec<C64>>,
}

impl Phases {
    pub(crate) fn new(mesh: &KPointMesh, mutation: Option<KStressMutation>) -> Self {
        let moduli = mesh.residue_moduli();
        let nr = moduli[0] * moduli[1] * moduli[2];
        let nk = mesh.nk();
        let true_: Vec<Vec<C64>> = (0..nk)
            .map(|k| {
                (0..nr)
                    .map(|r| mesh.phase(k, residue_coords(r, moduli)))
                    .collect()
            })
            .collect();
        let d = if mutation == Some(KStressMutation::NoPhase) {
            vec![vec![cr(1.0); nr]; nk]
        } else {
            true_.clone()
        };
        Self { moduli, true_, d }
    }
}

/// `Σ_r Σ_mn dP[r][ab]_mn(g) w[r]_nm` (the strain analogue of
/// `kgrad::residue_trace`).
pub(crate) fn residue_trace_d(
    dp: &[[Array3<C64>; 9]],
    ab: usize,
    gi: usize,
    w: &[Array2<C64>],
) -> C64 {
    let n = w[0].nrows();
    let mut acc = C64::new(0.0, 0.0);
    for (dr, wr) in dp.iter().zip(w) {
        let a = &dr[ab];
        for m in 0..n {
            for nn in 0..n {
                acc += a[[m, nn, gi]] * wr[(nn, m)];
            }
        }
    }
    acc
}

/// The one-electron and nuclear pieces of the k stress (`dE/dε`, Hartree per
/// cell, NOT divided by Ω).
#[derive(Debug, Clone, Default)]
pub struct KOneElectronStress {
    /// `Σ_k (1/N_k) tr[D dT]`.
    pub kinetic: Mat3,
    /// `Σ D dV_SR` (basis centres and nuclei).
    pub vsr: Mat3,
    /// `Σ D dV_LR`, including its `−δ E_LR` volume term.
    pub vlr: Mat3,
    /// `−δ c0 Z_tot N`.
    pub c0_volume: Mat3,
    /// `Σ_L W_L ∘ dV_L` strain of the periodic ECP ([`crate::ecp`]); zero
    /// without an ECP.
    pub ecp: Mat3,
    /// Kept ECP triples differentiated (0 without an ECP).
    pub n_ecp_triples: usize,
    /// Virial of each overlap weight set handed in (`Σ w dS`), in order.
    pub overlap: Vec<Mat3>,
    /// Ewald `E_nn`: real space, reciprocal (with volume), background.
    pub nn_sr: Mat3,
    pub nn_lr: Mat3,
    pub nn_background: Mat3,
    /// `Σ D V_LR` recomputed from the strain kernel's `P`.
    pub e_vlr: f64,
    /// `N = (1/N_k) Σ_k tr[D(k) S(k)]`.
    pub electrons: f64,
    /// Pair images, SR triplets, V_LR G count and pair-FT chunks.
    pub n_images: usize,
    pub n_sr_triplets: usize,
    /// Orbital columns the hcore SR derivative walk ran rotated.
    pub hcore_rotated_columns: usize,
    pub n_g_lr: usize,
    pub n_chunks: usize,
}

/// The periodic-ECP strain term at k: `(Σ_L W_L ∘ dV_L strain, triples)`, zeros
/// for an all-electron cell; `w_d[r]` is the SR walk's image weight. Errors
/// when hcore's `V_ECP` and the strain screen disagree.
#[allow(clippy::too_many_arguments)]
fn ecp_strain_term(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    w_d: &[Array2<f64>],
    mutation: Option<KStressMutation>,
    ledger: &mut Ledger,
) -> Result<(Mat3, usize), FerricError> {
    if mutation == Some(KStressMutation::NoEcp) && hk.v_ecp.is_some() {
        return Ok((ZERO3, 0));
    }
    let ecp_mut = match mutation {
        Some(KStressMutation::EcpNoCentre) => Some(EcpStrainMutation::NoCentre),
        Some(KStressMutation::EcpCentreSign) => Some(EcpStrainMutation::CentreSign),
        Some(KStressMutation::EcpNoKetImage) => Some(EcpStrainMutation::NoKetImage),
        Some(KStressMutation::EcpUnitPairVector) => Some(EcpStrainMutation::UnitPairVector),
        _ => None,
    };
    let weights = EcpWeights::Residue {
        w: w_d,
        recip: cell.reciprocal(),
        moduli: if mutation == Some(KStressMutation::EcpMeshCountModuli) {
            mesh.n()
        } else {
            mesh.residue_moduli()
        },
        flip_l: mutation == Some(KStressMutation::EcpWrongResidue),
    };
    let r = periodic_ecp_strain_weighted(
        cell,
        prep,
        &hcore_cfg.ecp_config(),
        &weights,
        ecp_mut,
        ledger,
    )?;
    let n = ecp_presence("k stress", hk, r.as_ref().map(|x| x.1))?;
    Ok((r.map_or(ZERO3, |x| x.0), n))
}

/// Resolved ledger for the stress passes (per-k matrices and residue weights).
pub(crate) fn open_ledger(
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    cfg: &KStressConfig,
    unrestricted: bool,
) -> Result<Ledger, FerricError> {
    let n = prep.nbasis();
    let nk = mesh.nk();
    let m = mesh.residue_moduli();
    let nr = m[0] * m[1] * m[2];
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    let per_k = if unrestricted { 16 } else { 10 };
    ledger.reserve(
        &format!("k stress per-k complex n×n matrices (n = {n}, N_k = {nk})"),
        bytes_of((nk * n * n) as u64, 16 * per_k),
    )?;
    ledger.reserve(
        &format!("k stress residue-folded weights (R = {nr}, n = {n})"),
        bytes_of((nr * n * n) as u64, 16 * 2 + 8 * 6),
    )?;
    Ok(ledger)
}

/// The one-electron + nuclear strain pass at the total density `dtot(k)`.
///
/// `overlap_sets` are real per-residue weight sets (`w[r][m, n]`, block
/// `(m_0, n_L)`, `L ≡ r`, as [`fold_real`]) whose overlap virials `Σ w dS`
/// are returned in order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn one_electron_pass(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    dtot: &[Array2<C64>],
    ph: &Phases,
    overlap_sets: &[&[Array2<f64>]],
    cfg: &KStressConfig,
    ledger: &mut Ledger,
) -> Result<KOneElectronStress, FerricError> {
    let mutation = cfg.mutation;
    let is = |m: KStressMutation| mutation == Some(m);
    let n = prep.nbasis();
    let nk = mesh.nk();
    let inv_nk = 1.0 / nk as f64;
    let moduli = ph.moduli;
    let b = cell.reciprocal();
    let nrm1 = if is(KStressMutation::NoInvNk) {
        1.0
    } else {
        inv_nk
    };
    let w_t = fold_real(dtot, &ph.d, nrm1);
    let w_d = fold_real(dtot, &ph.d, inv_nk);
    let delta = fold_complex(dtot, &ph.d, inv_nk);
    let delta_true = fold_complex(dtot, &ph.true_, inv_nk);
    let mut out = KOneElectronStress {
        overlap: vec![ZERO3; overlap_sets.len()],
        ..Default::default()
    };

    // --- N = (1/N_k) Σ_k Re tr[D(k) S(k)].
    let mut electrons = 0.0_f64;
    for (d, s) in dtot.iter().zip(&hk.s) {
        for m in 0..n {
            for nn in 0..n {
                electrons += (d[(m, nn)] * s[(nn, m)]).re;
            }
        }
    }
    electrons *= inv_nk;
    out.electrons = electrons;

    // --- dS, dT: per-image derivative blocks weighted by the pair vector.
    let images = hcore_pair_images(cell, prep, hcore_cfg.precision, ledger)?;
    let dims = prep.shell_dims().to_vec();
    let offs = prep.shell_offsets().to_vec();
    let centers: Vec<[f64; 3]> = prep.located_shells().iter().map(|s| s.center).collect();
    let nsh = prep.nshells();
    let drop_images = is(KStressMutation::NoSrImages);
    {
        let mut eng_s = Engine::new_1e_deriv(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
        let mut eng_t = Engine::new_1e_deriv(ffi::OP_KINETIC, prep, ONE_E_ENGINE_PRECISION)?;
        for l in &images {
            let r = residue_index(lattice_coords(&b, l), moduli);
            let li = if drop_images { [0.0; 3] } else { *l };
            for s1 in 0..nsh {
                for s2 in 0..nsh {
                    let (a, bc) = (centers[s1], centers[s2]);
                    let rel = [
                        a[0] - bc[0] - li[0],
                        a[1] - bc[1] - li[1],
                        a[2] - bc[2] - li[2],
                    ];
                    if let Some(blk) = eng_s.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        for (acc, set) in out.overlap.iter_mut().zip(overlap_sets) {
                            add_pair_virial(acc, blk, &set[r], &dims, &offs, s1, s2, rel);
                        }
                    }
                    if let Some(blk) = eng_t.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        add_pair_virial(&mut out.kinetic, blk, &w_t[r], &dims, &offs, s1, s2, rel);
                    }
                }
            }
        }
    }
    out.n_images = images.len();

    // --- V_SR: the force's image weight Re Δ(L), strain virial per triplet.
    let sr_cfg = PeriodicHcoreConfig {
        nucleus_exponent: cfg
            .nucleus_exponent
            .unwrap_or(GRAD_NUCLEUS_EXPONENT)
            .min(hcore_cfg.nucleus_exponent),
        ..*hcore_cfg
    };
    let mut vsr = ZERO3;
    // The energy's SR walk: rotated shells (weights `Tᵀ w T`) when `hk` ran
    // rotated.
    let rot = hcore_rotation_built("k stress", cell, prep, hcore_cfg, hk.sr_rotated_columns)?;
    let w_rot = rotate_hcore_weights(
        rot.as_ref(),
        KRotKind::defect(rot_kind(mutation)),
        is(KStressMutation::RotHcoreWrongResidue),
        &w_d,
    );
    let w_sr = w_rot.as_deref().unwrap_or(&w_d);
    let rot_walk = rot.as_ref().filter(|_| w_rot.is_some());
    out.hcore_rotated_columns = rot_walk.map_or(0, |r| r.n_rotated_columns);
    out.n_sr_triplets =
        sr_attraction_deriv_visit(cell, prep, rot_walk, &sr_cfg, ledger, false, |t| {
            let wd = &w_sr[residue_index(lattice_coords(&b, &t.l), moduli)];
            let (bb, xx) = if drop_images {
                (
                    [
                        t.b_image[0] - t.l[0],
                        t.b_image[1] - t.l[1],
                        t.b_image[2] - t.l[2],
                    ],
                    [
                        t.nucleus[0] - t.m[0],
                        t.nucleus[1] - t.m[1],
                        t.nucleus[2] - t.m[2],
                    ],
                )
            } else {
                (t.b_image, t.nucleus)
            };
            let ra = [
                t.a_center[0] - xx[0],
                t.a_center[1] - xx[1],
                t.a_center[2] - xx[2],
            ];
            let rb = [bb[0] - xx[0], bb[1] - xx[1], bb[2] - xx[2]];
            let mut ga = [0.0_f64; 3];
            let mut gb = [0.0_f64; 3];
            for i in 0..t.dim1 {
                for j in 0..t.dim2 {
                    let coeff = t.f * wd[(t.off1 + i, t.off2 + j)];
                    if coeff == 0.0 {
                        continue;
                    }
                    let idx = i * t.dim2 + j;
                    let idx_swapped = j * t.dim1 + i;
                    for c in 0..3 {
                        ga[c] += coeff * t.ket_a[c * t.bs + idx_swapped];
                        gb[c] += coeff * t.ket_b[c * t.bs + idx];
                    }
                }
            }
            for a in 0..3 {
                for bq in 0..3 {
                    vsr[a][bq] += ga[a] * ra[bq] + gb[a] * rb[bq];
                }
            }
        })?;
    out.vsr = vsr;

    // --- V_ECP: the Gamma ECP strain walk with the SR walk's image weight.
    let (e_ecp, n_ecp) = ecp_strain_term(cell, prep, mesh, hcore_cfg, hk, &w_d, mutation, ledger)?;
    out.ecp = e_ecp;
    out.n_ecp_triples = n_ecp;

    // --- V_LR: full G sphere, residue pair-FT strain kernel.
    let zs = cell.nuclear_charges();
    let ztot: f64 = zs.iter().sum();
    let omega = hcore_cfg.omega;
    let vol = cell.volume();
    let pos = cell.positions();
    let ft_terms = PairFtStrainTerms {
        centres: !is(KStressMutation::FtNoCentres),
        g_shape: !is(KStressMutation::GUnstrained),
    };
    let gcut_lr = lr_gcut(prep, omega, hcore_cfg.precision);
    ledger.reserve(
        &format!("k stress V_LR G list (|G| <= {gcut_lr:.3})"),
        gvector_list_bytes(cell, gcut_lr)?,
    )?;
    let gv_lr: Vec<[f64; 3]> = cell
        .gvectors(gcut_lr)?
        .into_iter()
        .filter(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2] > 0.0)
        .collect();
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let mut e_vlr = 0.0_f64;
    let mut s_vlr = ZERO3;
    out.n_chunks = pair_ft_strain_residues_chunked(
        cell,
        prep,
        &gv_lr,
        moduli,
        0.1 * hcore_cfg.precision,
        chunk_budget,
        0,
        ft_terms,
        |_g0, gs, p, dp| {
            let fac = 1.0 / vol;
            for (gi, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let v = 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp();
                let dvdg2 = -v * (1.0 / g2 + 1.0 / (4.0 * omega * omega));
                let (mut sre, mut sim) = (0.0_f64, 0.0_f64);
                for (z, r) in zs.iter().zip(&pos) {
                    let a = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                    sre += z * a.cos();
                    sim -= z * a.sin();
                }
                let rho = residue_trace(p, gi, &delta_true);
                let phi = -(rho.re * sre + rho.im * sim);
                e_vlr += fac * v * phi;
                for a in 0..3 {
                    for bq in 0..3 {
                        let dz = residue_trace_d(dp, 3 * a + bq, gi, &delta);
                        let dv = if ft_terms.g_shape {
                            dvdg2 * (-2.0 * gvec[a] * gvec[bq])
                        } else {
                            0.0
                        };
                        let dphi = -(dz.re * sre + dz.im * sim);
                        s_vlr[a][bq] += fac * (dv * phi + v * dphi);
                    }
                }
            }
            Ok(())
        },
    )?;
    if !is(KStressMutation::NoVolume) {
        add_into(&mut s_vlr, &diag(-e_vlr), 1.0);
    }
    out.vlr = s_vlr;
    out.e_vlr = e_vlr;
    out.n_g_lr = gv_lr.len();

    // --- h's G = 0 bookkeeping: E_c0 = c0 Z_tot N ∝ 1/Ω.
    let c0 = PI / (omega * omega * vol);
    if !is(KStressMutation::NoC0Volume) {
        out.c0_volume = diag(-c0 * ztot * electrons);
    }

    // --- Ewald E_nn (the energy's ω; the result is ω-independent).
    let nn = ewald_nuclear_strain(cell, default_ewald_omega(cell), DEFAULT_EWALD_PRECISION)?;
    out.nn_sr = nn.sr;
    if !is(KStressMutation::NoEwaldLr) {
        out.nn_lr = nn.lr;
    }
    if !is(KStressMutation::NoEwaldBg) {
        out.nn_background = nn.background;
    }
    Ok(out)
}

impl KOneElectronStress {
    /// `dE/dε` of `(1/N_k) Σ_k tr[D h(k)] + E_nn` at FIXED `D` when the
    /// single overlap set handed to the pass was `c0 Z_tot D` (the `Σ D dS`
    /// part of `h`'s G = 0 term): kinetic + V_SR + V_LR + c0 (volume and
    /// overlap) + Ewald. The Pulay `−W dS` is NOT part of it.
    pub fn hcore_total(&self) -> Mat3 {
        let mut t = ZERO3;
        for m in [
            &self.kinetic,
            &self.vsr,
            &self.ecp,
            &self.vlr,
            &self.c0_volume,
            &self.nn_sr,
            &self.nn_lr,
            &self.nn_background,
        ]
        .into_iter()
        .chain(self.overlap.iter())
        {
            add_into(&mut t, m, 1.0);
        }
        t
    }
}

/// The column-rotation test defect a mutation requests.
fn rot_kind(m: Option<KStressMutation>) -> Option<KRotKind> {
    match m {
        Some(KStressMutation::RotDropT) => Some(KRotKind::DropT),
        Some(KStressMutation::RotUseT) => Some(KRotKind::UseT),
        Some(KStressMutation::RotOneSided) => Some(KRotKind::OneSided),
        Some(KStressMutation::RotFlipSign) => Some(KRotKind::FlipSign),
        Some(KStressMutation::RotSwapPair) => Some(KRotKind::SwapPair),
        Some(KStressMutation::RotFitWrongResidue) => Some(KRotKind::FitWrongResidue),
        Some(KStressMutation::RotHcoreWrongResidue) => Some(KRotKind::HcoreWrongResidue),
        Some(KStressMutation::RotOff) => Some(KRotKind::Off),
        Some(KStressMutation::RotHcoreOff) => Some(KRotKind::HcoreOff),
        _ => None,
    }
}

/// TEST/DIAGNOSTIC: the one-electron + `E_nn` strain of the k-point energy at
/// a FIXED total density `D(k)` (`dtot`, one `n × n` matrix per mesh point),
/// i.e. `dE/dε` of `(1/N_k) Σ_k tr[D(k) h(k)] + E_nn` with `D` held fixed.
/// No SCF, no two-electron terms: the hcore gate that must pass before the
/// ERI terms are trusted. `hcore_total()` of the result is the quantity.
#[doc(hidden)]
pub fn kpoint_hcore_stress_fixed_d(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    dtot: &[Array2<C64>],
    cfg: &KStressConfig,
) -> Result<KOneElectronStress, FerricError> {
    let who = "kpoint_hcore_stress_fixed_d";
    let n = prep.nbasis();
    let nk = mesh.nk();
    if hk.omega != hcore_cfg.omega {
        return Err(FerricError::General(format!(
            "{who}: hcore was built at omega = {} but hcore_cfg.omega = {}",
            hk.omega, hcore_cfg.omega
        )));
    }
    if hk.s.len() != nk
        || hk.h.len() != nk
        || dtot.len() != nk
        || hk
            .s
            .iter()
            .chain(&hk.h)
            .chain(dtot)
            .any(|x| x.dim() != (n, n))
    {
        return Err(FerricError::General(format!(
            "{who}: shape mismatch (nbasis {n}, N_k {nk}: S(k) {}, h(k) {}, D(k) {})",
            hk.s.len(),
            hk.h.len(),
            dtot.len()
        )));
    }
    let mut ledger = open_ledger(prep, mesh, cfg, false)?;
    let ph = Phases::new(mesh, cfg.mutation);
    let omega = hcore_cfg.omega;
    let c0 = PI / (omega * omega * cell.volume());
    let ztot: f64 = cell.nuclear_charges().iter().sum();
    let w_c0 = fold_real(
        &dtot
            .iter()
            .map(|d| d.mapv(|z| z * c0 * ztot))
            .collect::<Vec<_>>(),
        &ph.d,
        if cfg.mutation == Some(KStressMutation::NoInvNk) {
            1.0
        } else {
            1.0 / mesh.nk() as f64
        },
    );
    one_electron_pass(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        dtot,
        &ph,
        &[&w_c0],
        cfg,
        &mut ledger,
    )
}

// ----------------------------------------------------------------- full stress

/// Where the k-point J/K come from.
#[derive(Clone, Copy)]
pub enum KStressJk<'a> {
    /// Dense pure-AFT oracle (bare `4π/K²` kernels).
    Dense(&'a KDenseAftEri),
    /// k-point RS-GDF (UNSPLIT builds; a range split is refused with a typed error).
    RsGdf(KRsGdfGradSource<'a>),
}

/// Per-term breakdown of `dE/dε` (each 3 × 3, Hartree per cell; NOT divided by Ω).
#[derive(Debug, Clone, Default)]
pub struct KStressParts {
    /// `Σ w_pulay dS`, `w = −W`.
    pub overlap_pulay: Mat3,
    /// `Σ w dS`, `w = c0 Z_tot D` (h's G = 0 term, overlap part).
    pub overlap_c0: Mat3,
    /// `Σ w dS`, `w = −v_M Σ_s D_s S D_s` (Madelung S-term).
    pub overlap_madelung: Mat3,
    /// `Σ D dT`.
    pub kinetic: Mat3,
    /// `Σ D dV_SR`.
    pub vsr: Mat3,
    /// `Σ_L W_L ∘ dV_L` of the periodic ECP (zero without an ECP).
    pub ecp: Mat3,
    /// `Σ D dV_LR`, including its `−δ E_LR` volume term.
    pub vlr: Mat3,
    /// `−δ c0 Z_tot N`.
    pub c0_volume: Mat3,
    /// Dense-AFT Coulomb, including its `−δ E_J` volume term (zero on RS-GDF).
    pub coulomb: Mat3,
    /// Dense-AFT exchange, including its `−δ E_K` volume term.
    pub exchange: Mat3,
    /// `−½ (1/N_k) Σ_s Σ_k tr[D_s S D_s S] dv_M/dε` (zero for exxdiv = none).
    pub madelung: Mat3,
    /// Ewald `E_nn`: real space, reciprocal (with volume), background.
    pub nn_sr: Mat3,
    pub nn_lr: Mat3,
    pub nn_background: Mat3,
    /// RS-GDF `Σ Z dJ3`: SR, LR (with volume).
    pub fit_j3_sr: Mat3,
    pub fit_j3_lr: Mat3,
    /// RS-GDF `Σ Wm dJ2`: SR, LR (with volume).
    pub fit_j2_sr: Mat3,
    pub fit_j2_lr: Mat3,
    /// RS-GDF J3 G = 0 through `dS` (`Σ M_g0 dS`).
    pub fit_g0_overlap: Mat3,
    /// RS-GDF `−δ E_g0` and `+δ c0 qᵀ Wm(0) q`.
    pub fit_j3_g0_volume: Mat3,
    pub fit_j2_g0_volume: Mat3,
}

impl KStressParts {
    fn all(&self) -> [&Mat3; 21] {
        [
            &self.overlap_pulay,
            &self.overlap_c0,
            &self.overlap_madelung,
            &self.kinetic,
            &self.vsr,
            &self.ecp,
            &self.vlr,
            &self.c0_volume,
            &self.coulomb,
            &self.exchange,
            &self.madelung,
            &self.nn_sr,
            &self.nn_lr,
            &self.nn_background,
            &self.fit_j3_sr,
            &self.fit_j3_lr,
            &self.fit_j2_sr,
            &self.fit_j2_lr,
            &self.fit_g0_overlap,
            &self.fit_j3_g0_volume,
            &self.fit_j2_g0_volume,
        ]
    }

    /// `Σ M dS` (Pulay + c0 + Madelung S-term), the Gamma `overlap` part.
    pub fn overlap(&self) -> Mat3 {
        let mut t = ZERO3;
        for m in [
            &self.overlap_pulay,
            &self.overlap_c0,
            &self.overlap_madelung,
        ] {
            add_into(&mut t, m, 1.0);
        }
        t
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

/// Output of the k-point stress entry points.
#[derive(Debug, Clone)]
pub struct KStress {
    /// `dE/dε_ab` (Hartree per cell).
    pub de_deps: Mat3,
    /// `σ_ab = (1/Ω) dE/dε_ab` (Hartree/Bohr³). Not symmetrised.
    pub sigma: Mat3,
    /// Per-term breakdown of `de_deps`.
    pub parts: KStressParts,
    /// Cell volume Ω (Bohr³).
    pub volume: f64,
    /// `max_{σ,k} max |F_σ D_σ S − S D_σ F_σ|` of the rebuilt Focks. For the
    /// restricted-open-shell entry points ([`crate::kuxcstress::kpoint_roks_stress`],
    /// `kpoint_rohf_stress`) the per-spin commutators do not vanish at a ROHF
    /// state, so this field instead holds the ROHF ORBITAL GRADIENT the gate
    /// measured (the largest spectral norm of the closed-open, open-virtual and
    /// closed-virtual blocks).
    pub commutator: f64,
    /// `v_M` used in `K` and `M` (0 for exxdiv = none).
    pub madelung: f64,
    /// `max |F_rebuilt − F_stored|` over k (and spins): the input
    /// consistency measure refused above [`FOCK_CONSISTENCY_BAR`].
    pub fock_mismatch: f64,
    /// `dv_M/dε` of the mesh (supercell) lattice (reported for every exxdiv).
    pub madelung_strain: Mat3,
    /// `N = (1/N_k) Σ_k tr[D(k) S(k)]`.
    pub electrons: f64,
    /// `Σ D V_LR` recomputed from the strain kernel's `P`.
    pub e_vlr: f64,
    /// Dense-AFT Coulomb energy `(1/2Ω) Σ_{G≠0} v |ρ|²`, per cell (`None` on
    /// RS-GDF: the fitted energy is not split into J and K pieces here).
    pub e_coulomb: Option<f64>,
    /// Dense-AFT exchange energy `−(1/(2Ω N_k²)) Σ ...` (without the Madelung
    /// term; `None` on RS-GDF).
    pub e_exchange: Option<f64>,
    /// Madelung energy `−(v_M/2)(1/N_k) Σ_s Σ_k tr[D_s S D_s S]`.
    pub e_madelung: f64,
    /// Pair images, SR triplets, V_LR G count, ERI K count, pair-FT chunks.
    pub n_images: usize,
    pub n_sr_triplets: usize,
    /// Orbital columns the hcore SR derivative walk ran rotated (equals
    /// `PeriodicHcoreK::sr_rotated_columns`; 0 = unrotated).
    pub hcore_rotated_columns: usize,
    /// ECP triples differentiated (the energy's `PeriodicHcoreK::n_ecp_triples`).
    pub n_ecp_triples: usize,
    pub n_g_lr: usize,
    pub n_k_eri: usize,
    pub n_chunks: usize,
    /// Resolved memory budget (bytes).
    pub budget_bytes: usize,
    /// RS-GDF: largest number of metric eigenvalues dropped at any q (`None` dense).
    pub fit_dropped_max: Option<usize>,
    /// RS-GDF: rebuild consistency and strain-walk counters (`None` dense).
    pub fit_checks: Option<KFitChecks>,
}

impl KStress {
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

    /// `max_ab |σ_ab − σ_ba|` (Hartree/Bohr³): roundoff for HF.
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

/// `v_M` of the energy: 0 for `ExxDiv::None`, the mesh (supercell) Madelung
/// constant for `ExxDiv::Ewald`.
fn mesh_madelung(cell: &Cell, mesh: &KPointMesh, exxdiv: ExxDiv) -> Result<f64, FerricError> {
    Ok(match exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => mesh.madelung(cell)?,
    })
}

/// The source in the force module's form (one J/K builder for both).
pub(crate) fn grad_jk<'a>(jk: &KStressJk<'a>) -> KGradJk<'a> {
    match jk {
        KStressJk::Dense(e) => KGradJk::Dense(e),
        KStressJk::RsGdf(s) => KGradJk::RsGdf(*s),
    }
}

/// The typed refusals shared by every entry point: a range-separated dense
/// kernel and an inconsistent range split (a range-split RS-GDF build is
/// supported; so is an ECP).
pub(crate) fn refuse_unsupported(who: &str, jk: &KStressJk<'_>) -> Result<(), FerricError> {
    if let KStressJk::Dense(e) = jk {
        if e.rsh().is_some() {
            return Err(FerricError::General(format!(
                "{who}: the dense source was built with a range-separated exchange kernel \
                 (KDenseAftConfig::rsh); its strain (screened kernel at K = G + q) is not \
                 implemented"
            )));
        }
    }
    if let KStressJk::RsGdf(src) = jk {
        if src.gdf.rsh().is_some() {
            return Err(FerricError::General(format!(
                "{who}: the RS-GDF source carries a range-separated exchange \
                 (KRsGdf::build_rsh); its strain is not implemented"
            )));
        }
        refuse_split_mismatch(who, src)?;
    }
    Ok(())
}

/// Range-split consistency of an RS-GDF source: the config must say "split"
/// exactly when the build ran the split path, and a split above `lambda = 1`
/// is refused for the stress. The strain differentiates the build's own
/// partition from the CONFIG's, so a config that disagrees with the build in
/// lambda, omega or precision changes the result only at the split tolerance
/// (~1e-11 measured for lambda <= 1; lambda = 2 is ~1e-8, at the Fock-guard
/// bar), but split vs unsplit would differentiate a different partition.
fn refuse_split_mismatch(who: &str, src: &KRsGdfGradSource<'_>) -> Result<(), FerricError> {
    let built = !src.gdf.stats().split_counters.is_empty();
    let cfg_split = src.cfg.gdf.range_split;
    if built != cfg_split.is_some() {
        return Err(FerricError::General(format!(
            "{who}: the RS-GDF build {} the range split but the config {}; pass the config the \
             build used",
            if built { "ran" } else { "did not run" },
            if cfg_split.is_some() {
                "requests one"
            } else {
                "does not"
            }
        )));
    }
    if let Some(rs) = cfg_split {
        if rs.lambda > 1.0 {
            return Err(FerricError::General(format!(
                "{who}: range split lambda = {} > 1 is not supported by the k stress (it moves \
                 blocks whose FT need not converge in the sphere, and the K set is filtered in \
                 the strained frame, so the energy is piecewise under strain)",
                rs.lambda
            )));
        }
    }
    Ok(())
}

/// k-point RHF stress (module doc) of a converged
/// [`crate::kscf::solve_krhf_injected`] / [`crate::kscf::solve_krhf`] result.
/// Arguments as [`crate::kgrad::kpoint_rhf_gradient`]; `exxdiv` is the
/// energy's (the Madelung strain is present only for `ewald`).
#[allow(clippy::too_many_arguments)]
pub fn kpoint_rhf_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KStressJk<'_>,
    scf: &KScfResult,
    exxdiv: ExxDiv,
    cfg: &KStressConfig,
) -> Result<KStress, FerricError> {
    let who = "kpoint_rhf_stress";
    refuse_unsupported(who, &jk)?;
    let gj = grad_jk(&jk);
    check_inputs(
        who,
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &gj,
        &[&scf.densities[..]],
        scf.converged,
        &scf.kpts,
    )?;
    let mut ledger = open_ledger(prep, mesh, cfg, false)?;
    let vm = mesh_madelung(cell, mesh, exxdiv)?;
    let d = scf.densities.clone();
    let (j, k) = jk_build(&gj, vm, &d)?;
    let f: Vec<Array2<C64>> = (0..mesh.nk())
        .map(|q| &(&hk.h[q] + &j[q]) - &k[q].mapv(|z| z * 0.5))
        .collect();
    let mismatch = check_fock(who, "fock", &f, &scf.fock, STRESS_REMEDY)?;
    let mut out = assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &gj,
        KSpins::Restricted { d, f },
        vm,
        1.0,
        cfg,
        &mut ledger,
    )?;
    out.fock_mismatch = mismatch;
    Ok(out)
}

/// k-point UHF stress (module doc) of a converged
/// [`crate::kuscf::solve_kuhf`] stage (`KUhfResult::scf`) or
/// [`crate::kuscf::solve_kuhf_injected`] result. Arguments as
/// [`kpoint_rhf_stress`].
#[allow(clippy::too_many_arguments)]
pub fn kpoint_uhf_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KStressJk<'_>,
    scf: &KUScfResult,
    exxdiv: ExxDiv,
    cfg: &KStressConfig,
) -> Result<KStress, FerricError> {
    let who = "kpoint_uhf_stress";
    refuse_unsupported(who, &jk)?;
    let gj = grad_jk(&jk);
    check_inputs(
        who,
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &gj,
        &[&scf.density_alpha[..], &scf.density_beta[..]],
        scf.converged,
        &scf.kpts,
    )?;
    let mut ledger = open_ledger(prep, mesh, cfg, true)?;
    let vm = mesh_madelung(cell, mesh, exxdiv)?;
    let da = scf.density_alpha.clone();
    let db = scf.density_beta.clone();
    let (ja, ka) = jk_build(&gj, vm, &da)?;
    let (jb, kb) = jk_build(&gj, vm, &db)?;
    let nk = mesh.nk();
    let base: Vec<Array2<C64>> = (0..nk).map(|q| &(&hk.h[q] + &ja[q]) + &jb[q]).collect();
    let fa: Vec<Array2<C64>> = (0..nk).map(|q| &base[q] - &ka[q]).collect();
    let fb: Vec<Array2<C64>> = (0..nk).map(|q| &base[q] - &kb[q]).collect();
    let mismatch = check_fock(who, "fock_alpha", &fa, &scf.fock_alpha, STRESS_REMEDY)?.max(
        check_fock(who, "fock_beta", &fb, &scf.fock_beta, STRESS_REMEDY)?,
    );
    let mut out = assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &gj,
        KSpins::Unrestricted { da, db, fa, fb },
        vm,
        1.0,
        cfg,
        &mut ledger,
    )?;
    out.fock_mismatch = mismatch;
    Ok(out)
}

/// Assemble the k-point stress from the per-spin `D(k)`/`F(k)`; `vm` is the
/// Madelung shift of the energy and `exch_scale` the exact-exchange fraction
/// `a` ([`crate::kgrad`]'s convention: `1` is Hartree-Fock, bitwise the
/// unscaled terms; `0` is no exchange at all: the K strain walks (dense and
/// fitted) are skipped and the Madelung M-term and its strain vanish; `0 < a
/// < 1` scales the dense K walks, the fit's exchange pieces and the Madelung
/// M-term and strain by `a`, validated by the global-hybrid KS stress
/// [`crate::kxcstress`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: &KGradJk<'_>,
    spins: KSpins,
    vm: f64,
    exch_scale: f64,
    cfg: &KStressConfig,
    ledger: &mut Ledger,
) -> Result<KStress, FerricError> {
    let nk = mesh.nk();
    let mutation = cfg.mutation;
    let is = |m: KStressMutation| mutation == Some(m);
    let s_k = &hk.s;
    let commutator = spins.commutator(s_k);
    let mut spins = spins;
    if is(KStressMutation::ConjD) {
        spins.conjugate();
    }
    let ph = Phases::new(mesh, mutation);
    let inv_nk = 1.0 / nk as f64;
    let nrm1 = if is(KStressMutation::NoInvNk) {
        1.0
    } else {
        inv_nk
    };
    let dtot: Vec<Array2<C64>> = (0..nk).map(|k| spins.total(k)).collect();
    let zs = cell.nuclear_charges();
    let ztot: f64 = zs.iter().sum();
    let omega = hcore_cfg.omega;
    let vol = cell.volume();
    let c0 = PI / (omega * omega * vol);
    let madelung_a = if is(KStressMutation::MadelungUnscaled) {
        1.0
    } else {
        exch_scale
    };
    let vm_m = madelung_a
        * if is(KStressMutation::GammaMadelung) && vm != 0.0 {
            madelung_constant(cell)?
        } else {
            vm
        };

    // --- The forces' overlap weights M(k), split into its three pieces.
    let w_pulay = fold_real(
        &(0..nk)
            .map(|k| match &spins {
                KSpins::Unrestricted { da, db, fa, fb } if is(KStressMutation::SwapSpinFock) => {
                    -(&da[k].dot(&fb[k]).dot(&da[k]) + &db[k].dot(&fa[k]).dot(&db[k]))
                }
                _ => spins.energy_weighted(k).mapv(|z| -z),
            })
            .collect::<Vec<_>>(),
        &ph.d,
        nrm1,
    );
    let w_c0 = fold_real(
        &dtot
            .iter()
            .map(|d| d.mapv(|z| z * (c0 * ztot)))
            .collect::<Vec<_>>(),
        &ph.d,
        nrm1,
    );
    let w_mad = fold_real(
        &(0..nk)
            .map(|k| spins.sandwich(k, &s_k[k]).mapv(|z| z * (-vm_m)))
            .collect::<Vec<_>>(),
        &ph.d,
        nrm1,
    );
    // --- RS-GDF fitted strain pieces first (M_g0 enters the dS pass).
    let fit = match jk {
        KGradJk::Dense(_) => None,
        KGradJk::RsGdf(src) => Some(fit_pass(
            src, cell, prep, mesh, s_k, &dtot, &spins, exch_scale, cfg, ledger,
        )?),
    };
    let w_g0 = fit.as_ref().map(|f| fold_real(&f.mg0, &ph.d, 1.0));
    let mut sets: Vec<&[Array2<f64>]> = vec![&w_pulay, &w_c0, &w_mad];
    if let Some(w) = &w_g0 {
        sets.push(w);
    }
    let one = one_electron_pass(
        cell, prep, mesh, hcore_cfg, hk, &dtot, &ph, &sets, cfg, ledger,
    )?;
    let mut parts = KStressParts {
        overlap_pulay: one.overlap[0],
        overlap_c0: one.overlap[1],
        overlap_madelung: one.overlap[2],
        kinetic: one.kinetic,
        vsr: one.vsr,
        ecp: one.ecp,
        vlr: one.vlr,
        c0_volume: one.c0_volume,
        nn_sr: one.nn_sr,
        nn_lr: one.nn_lr,
        nn_background: one.nn_background,
        ..Default::default()
    };
    if is(KStressMutation::NoPulay) {
        parts.overlap_pulay = ZERO3;
        parts.overlap_c0 = ZERO3;
        parts.overlap_madelung = ZERO3;
    }
    if is(KStressMutation::NoC0Overlap) {
        parts.overlap_c0 = ZERO3;
    }
    if is(KStressMutation::MadelungS) {
        parts.overlap_madelung = ZERO3;
    }

    // --- Two-electron: dense AFT J and K, or the fitted RS-GDF pieces.
    let ft_terms = PairFtStrainTerms {
        centres: !is(KStressMutation::FtNoCentres),
        g_shape: !is(KStressMutation::GUnstrained),
    };
    let mut two = DenseStrain {
        coulomb: ZERO3,
        exchange: ZERO3,
        e_j: 0.0,
        e_k: 0.0,
        n_chunks: 0,
        n_kv: 0,
    };
    let mut fit_dropped_max = None;
    let mut fit_checks = None;
    if let (KGradJk::Dense(eri), None) = (jk, &fit) {
        let delta = fold_complex(&dtot, &ph.d, inv_nk);
        let delta_true = fold_complex(&dtot, &ph.true_, inv_nk);
        two = dense_two_electron_strain(
            cell,
            prep,
            mesh,
            eri,
            &spins,
            &delta,
            &delta_true,
            &ph,
            ft_terms,
            exch_scale,
            mutation,
            ledger,
        )?;
        parts.coulomb = two.coulomb;
        parts.exchange = two.exchange;
        if !is(KStressMutation::NoVolume) {
            add_into(&mut parts.coulomb, &diag(-two.e_j), 1.0);
            add_into(&mut parts.exchange, &diag(-two.e_k), 1.0);
        }
    }
    if let Some(f) = &fit {
        let fs: &KFitStrain = f.strain.as_ref().ok_or_else(|| {
            FerricError::General("k stress: RS-GDF strain pieces missing (internal)".into())
        })?;
        parts.fit_j3_sr = fs.j3_sr;
        parts.fit_j3_lr = fs.j3_lr;
        parts.fit_j2_sr = fs.j2_sr;
        parts.fit_j2_lr = fs.j2_lr;
        parts.fit_j3_g0_volume = fs.j3_g0_volume;
        parts.fit_j2_g0_volume = fs.j2_g0_volume;
        parts.fit_g0_overlap = one.overlap[3];
        // Range split: J3 -= c0 (S - S_ss) q_c, so the smooth-piece overlap
        // virial (over the build's pair images, residue-weighted by M_g0) is
        // SUBTRACTED; none unsplit and under the two mutants.
        if let (Some(sg), Some(w)) = (f.smooth_g0.as_ref(), w_g0.as_ref()) {
            if !is(KStressMutation::SplitNoSmoothOverlap) && !is(KStressMutation::SplitFullG0) {
                let b = cell.reciprocal();
                let moduli = ph.moduli;
                let sv = sg.overlap_virial_by(
                    |l| &w[residue_index(lattice_coords(&b, &l), moduli)],
                    is(KStressMutation::NoSrImages),
                )?;
                add_into(&mut parts.fit_g0_overlap, &sv, -1.0);
            }
        }
        if is(KStressMutation::NoMetric) {
            parts.fit_j2_sr = ZERO3;
            parts.fit_j2_lr = ZERO3;
            parts.fit_j2_g0_volume = ZERO3;
        }
        if is(KStressMutation::NoJ2G0Vol) {
            parts.fit_j2_g0_volume = ZERO3;
        }
        if is(KStressMutation::NoJ3G0Vol) {
            parts.fit_j3_g0_volume = ZERO3;
        }
        if is(KStressMutation::NoG0) {
            parts.fit_g0_overlap = ZERO3;
        }
        two.n_chunks = f.n_chunks;
        two.n_kv = f.n_k_lr;
        fit_dropped_max = Some(f.n_dropped_max);
        fit_checks = Some(KFitChecks {
            b_norm_mismatch: f.b_norm_mismatch,
            n_sr3: f.n_sr3,
            n_sr2: f.n_sr2,
            rotated_columns: f.rotated_columns,
        });
    }

    // --- Madelung: −½ (1/N_k) Σ_s Σ_k tr[D_s S D_s S] dv_M/dε, the strain of
    // the diag(N) SUPERCELL lattice's Madelung constant.
    let sc_cell = Cell::new(cell.mol().clone(), mesh.supercell_lattice())?;
    let vm_strain = if is(KStressMutation::GammaMadelungStrain) {
        madelung_strain(cell)?
    } else {
        madelung_strain(&sc_cell)?
    };
    let mut t_m = 0.0_f64;
    for k in 0..nk {
        let sd = spins.sandwich(k, &s_k[k]);
        let n = sd.nrows();
        for m in 0..n {
            for q in 0..n {
                t_m += (sd[(m, q)] * s_k[k][(q, m)]).re;
            }
        }
    }
    t_m *= inv_nk;
    if vm != 0.0 && exch_scale != 0.0 && !is(KStressMutation::NoMadelung) {
        parts.madelung = scaled(&vm_strain, -0.5 * madelung_a * t_m);
    }
    let e_madelung = -0.5 * exch_scale * vm * t_m;

    let de_deps = parts.total();
    if de_deps.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FerricError::General(
            "k stress: non-finite strain derivative".into(),
        ));
    }
    let sigma = scaled(&de_deps, 1.0 / vol);
    ferric_core::memory::warn_if_rss_over("ferric-pbc k stress", ledger.budget(), 1.1);
    Ok(KStress {
        de_deps,
        sigma,
        parts,
        volume: vol,
        commutator,
        madelung: vm,
        fock_mismatch: 0.0,
        madelung_strain: vm_strain,
        electrons: one.electrons,
        e_vlr: one.e_vlr,
        e_coulomb: fit.is_none().then_some(two.e_j),
        e_exchange: fit.is_none().then_some(two.e_k),
        e_madelung,
        n_images: one.n_images,
        n_sr_triplets: one.n_sr_triplets,
        hcore_rotated_columns: one.hcore_rotated_columns,
        n_ecp_triples: one.n_ecp_triples,
        n_g_lr: one.n_g_lr,
        n_k_eri: two.n_kv,
        n_chunks: one.n_chunks + two.n_chunks,
        budget_bytes: ledger.budget(),
        fit_dropped_max,
        fit_checks,
    })
}

/// The RS-GDF strain pass of [`assemble`] (the fitted `Z`, `Wm`, phase bins).
#[allow(clippy::too_many_arguments)]
fn fit_pass(
    src: &KRsGdfGradSource<'_>,
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    s_k: &[Array2<C64>],
    dtot: &[Array2<C64>],
    spins: &KSpins,
    exch_scale: f64,
    cfg: &KStressConfig,
    ledger: &mut Ledger,
) -> Result<crate::rsgdf::kpoint::kderiv::KFitGrad, FerricError> {
    let is = |m: KStressMutation| cfg.mutation == Some(m);
    let exch = if is(KStressMutation::FitExchangeDropped) {
        Vec::new()
    } else {
        scaled_exch_terms(spins, exch_scale)
    };
    let fm = KFitMutation {
        rot: KRotKind::defect(rot_kind(cfg.mutation)),
        aux_phase: is(KStressMutation::FitAuxPhase),
        no_metric: is(KStressMutation::NoMetric),
        lr_wrong_q: is(KStressMutation::FitLrWrongQ),
        no_loewner_kd: is(KStressMutation::FitNoLoewnerKd),
        cross_spin: is(KStressMutation::FitCrossSpin),
        split_full_g0: is(KStressMutation::SplitFullG0),
        ..Default::default()
    };
    let terms = KFitStrainTerms {
        pair: PairFtStrainTerms {
            centres: !is(KStressMutation::FtNoCentres),
            g_shape: !is(KStressMutation::GUnstrained),
        },
        g_shape: !is(KStressMutation::GUnstrained),
        volume: !is(KStressMutation::NoVolume),
        images: !is(KStressMutation::NoSrImages),
        aux_ft3: !is(KStressMutation::NoAuxFt) && !is(KStressMutation::NoAuxFt3),
        aux_ft2: !is(KStressMutation::NoAuxFt),
        sr_kernel: !is(KStressMutation::SplitNoSrKernelStrain),
        smooth_pair: !is(KStressMutation::SplitNoSmoothPair),
        kept_g0: !is(KStressMutation::SplitFullG0),
        flip_slope: is(KStressMutation::SplitFlipSrSlope),
        smooth_volume: !is(KStressMutation::SplitNoSmoothVolume),
    };
    kpoint_fit_strain(
        src.gdf, src.cfg, cell, prep, src.aux, mesh, s_k, dtot, &exch, fm, terms, ledger,
    )
}

/// Dense-AFT J and K strain (before the `−δ E` volume terms) and the energies.
struct DenseStrain {
    coulomb: Mat3,
    exchange: Mat3,
    e_j: f64,
    e_k: f64,
    n_chunks: usize,
    n_kv: usize,
}

/// Dense-AFT Coulomb and exchange strain over every q class (the strain twin
/// of `kgrad::dense_two_electron`): per `K = G + q`
///
/// ```text
/// J (q = 0):  (1/Ω)[ ½ dv |ρ|² + v Re(ρ̄ dρ) ],        ρ = Σ_r tr[P_r Δtrue_r],  dρ = Σ_r tr[dP_r Δ_r]
/// K:          c [ dv Φ + v dΦ ],  c = −1/(2Ω N_k²),    Φ = Σ_k' Re tr[P^{k'} Y^{k'}],  dΦ = 2 Re Σ_r tr[dP_r Yt_r]
/// ```
///
/// with `dv = v'(K²)(−2 K_a K_b)` (the G shape; frozen Miller index) and the
/// force's `Y`, `Yt_r`.
#[allow(clippy::too_many_arguments)]
fn dense_two_electron_strain(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    eri: &KDenseAftEri,
    spins: &KSpins,
    delta: &[Array2<C64>],
    delta_true: &[Array2<C64>],
    ph: &Phases,
    terms: PairFtStrainTerms,
    exch_scale: f64,
    mutation: Option<KStressMutation>,
    ledger: &mut Ledger,
) -> Result<DenseStrain, FerricError> {
    let n = prep.nbasis();
    let nk = mesh.nk();
    let moduli = ph.moduli;
    let nr = moduli[0] * moduli[1] * moduli[2];
    let vol = cell.volume();
    let gcut = eri.gcut();
    let g2cut = gcut * gcut;
    let qmax = (0..nk)
        .map(|iq| {
            let q = mesh.q_class(iq).0;
            (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()
        })
        .fold(0.0_f64, f64::max);
    ledger.reserve(
        &format!("k stress ERI K list (|K| <= {gcut:.3}, |q| <= {qmax:.3})"),
        gvector_list_bytes(cell, gcut + qmax)?.saturating_mul(2),
    )?;
    let exch = scaled_exch_terms(spins, exch_scale);
    let c_k = -1.0 / (2.0 * vol * (nk * nk) as f64);
    let mut out = DenseStrain {
        coulomb: ZERO3,
        exchange: ZERO3,
        e_j: 0.0,
        e_k: 0.0,
        n_chunks: 0,
        n_kv: 0,
    };
    // Per K: Yt (R n²), P^{k'} and its adjoint, Y (4 n²), complex.
    let extra_per_g = bytes_of(((nr + 4) * n * n) as u64, 16);
    for iq in 0..nk {
        let (q, mq) = mesh.q_class(iq);
        // Without exchange only the q = 0 Coulomb class contributes.
        if exch.is_empty() && mq != [0, 0, 0] {
            continue;
        }
        let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
        let ks: Vec<[f64; 3]> = cell
            .gvectors(gcut + qn)?
            .into_iter()
            .map(|g| [g[0] + q[0], g[1] + q[1], g[2] + q[2]])
            .filter(|k| {
                let k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                k2 > 0.0 && k2 <= g2cut
            })
            .collect();
        out.n_kv += ks.len();
        let mq_used = if mutation == Some(KStressMutation::WrongQ) {
            [-mq[0], -mq[1], -mq[2]]
        } else {
            mq
        };
        let kof: Vec<usize> = (0..nk).map(|j| mesh.k_minus_q(j, mq_used)).collect();
        let is_q0 = mq == [0, 0, 0];
        let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
        out.n_chunks += pair_ft_strain_residues_chunked(
            cell,
            prep,
            &ks,
            moduli,
            eri.pair_thresh(),
            chunk_budget,
            extra_per_g,
            terms,
            |_k0, kv, p, dp| {
                for (gi, kvec) in kv.iter().enumerate() {
                    let k2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
                    let v = 4.0 * PI / k2;
                    let dvdk2 = -v / k2;
                    let dv = |a: usize, b: usize| {
                        if terms.g_shape {
                            dvdk2 * (-2.0 * kvec[a] * kvec[b])
                        } else {
                            0.0
                        }
                    };
                    if is_q0 {
                        let rho = residue_trace(p, gi, delta_true);
                        let rho2 = rho.norm_sqr();
                        out.e_j += 0.5 / vol * v * rho2;
                        for a in 0..3 {
                            for b in 0..3 {
                                let drho = residue_trace_d(dp, 3 * a + b, gi, delta);
                                out.coulomb[a][b] +=
                                    (0.5 * dv(a, b) * rho2 + v * (rho.conj() * drho).re) / vol;
                            }
                        }
                    }
                    if exch.is_empty() {
                        continue;
                    }
                    // Exchange: Yt_r = Σ_k' e^{ik'·t_r} Σ_s D_s(k') P^{k'H} D_s(k'−q).
                    let mut yt: Vec<Array2<C64>> = (0..nr).map(|_| Array2::zeros((n, n))).collect();
                    let mut phi = 0.0_f64;
                    for j in 0..nk {
                        let mut pj = Array2::<C64>::zeros((n, n));
                        for (pr, phs) in p.iter().zip(&ph.true_[j]) {
                            for m in 0..n {
                                for l in 0..n {
                                    pj[(m, l)] += *phs * pr[[m, l, gi]];
                                }
                            }
                        }
                        let pjh = herm(&pj);
                        let mut y = Array2::<C64>::zeros((n, n));
                        match spins {
                            KSpins::Unrestricted { da, db, .. }
                            | KSpins::RestrictedOpen { da, db, .. }
                                if mutation == Some(KStressMutation::CrossSpinK) =>
                            {
                                y += &da[j].dot(&pjh).dot(&db[kof[j]]);
                                y += &db[j].dot(&pjh).dot(&da[kof[j]]);
                            }
                            _ => {
                                for (c, ds) in &exch {
                                    y.scaled_add(cr(*c), &ds[j].dot(&pjh).dot(&ds[kof[j]]));
                                }
                            }
                        }
                        for m in 0..n {
                            for l in 0..n {
                                phi += (pj[(m, l)] * y[(l, m)]).re;
                            }
                        }
                        for (r, ytr) in yt.iter_mut().enumerate() {
                            ytr.scaled_add(ph.d[j][r], &y);
                        }
                    }
                    out.e_k += c_k * v * phi;
                    for a in 0..3 {
                        for b in 0..3 {
                            let dphi = 2.0 * residue_trace_d(dp, 3 * a + b, gi, &yt).re;
                            out.exchange[a][b] += c_k * (dv(a, b) * phi + v * dphi);
                        }
                    }
                }
                Ok(())
            },
        )?;
    }
    Ok(out)
}
