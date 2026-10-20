//! Analytic nuclear forces of the k-point RHF and UHF (Gamma-centred or
//! shifted meshes) — the Rust port of `reference/pbc/pbc_kgrad.py` and
//! `reference/pbc/pbc_kgrad_gdf.py` (FINDINGS "Iteration 21 (Python,
//! k-point RHF/UHF forces)").
//!
//! # Energy (per cell; [`crate::kscf`] / [`crate::kuscf`] conventions)
//!
//! ```text
//! E = (1/N_k) Σ_k tr[h(k) D(k)] + (1/2Ω) Σ_{G≠0} v |ρ(G)|²
//!   − (1/(2Ω N_k²)) Σ_s Σ_{k,k'} Σ_{K∈G+q, K≠0} v(K) tr[P^{k'}(K) D_s(k') P^{k'}(K)^H D_s(k)]
//!   − (v_M/2)(1/N_k) Σ_s Σ_k tr[D_s S D_s S] + E_nn,          q = k' − k
//! h(k) = T(k) + V_SR(k) + V_LR(k) + c0 Z_tot S(k)              (hcore::kpoint, c0 = π/(ω²Ω))
//! ```
//!
//! `v_M` is the MESH (diag(N) supercell) Madelung constant for
//! `exxdiv = ewald`, 0 for `none`. RHF: `D` total, `D_s = D/2`.
//!
//! # Derivative (derivation: the `pbc_kgrad.py` docstring)
//!
//! Every k enters through a Bloch phase on an IMAGE-RESOLVED molecular
//! quantity (`X(k) = Σ_L e^{ik·L} X(L)`, `P^{k'} = Σ_r e^{ik'·t_r} p_r`); the
//! phases do not depend on the atoms, so `d/dR_A` acts on `X(L)` and `p_r`
//! only, and every term is the Gamma term ([`crate::grad`]) with the image /
//! residue index kept and a phase-folded weight:
//!
//! ```text
//! 1e / overlap:  Σ_L Σ_mn dX_mn(L) Re Mt_nm(L),   Mt(L) = (1/N_k) Σ_k e^{ik·L} M(k)
//!                M = D for T;  M(k) = −W(k) + c0 Z_tot D(k) − v_M Σ_s D_s S D_s,
//!                W(k) = Σ_s D_s(k) F_s(k) D_s(k)        (RHF: ½ D F D, −(v_M/2) D S D)
//! V_SR:          the Gamma erfc 3-centre derivative walk, each image L weighted
//!                by Re Δ_nm(L), Δ(L) = (1/N_k) Σ_k e^{ik·L} D(k); nucleus by
//!                translation invariance (−(bra + ket)); nuclei carry no phase.
//! V_LR, J (q=0): dρ(G) = Σ_r Σ_mn dp_r,mn Δ_r,nm;
//!                dE = (1/Ω) Σ_G Re[(v conj ρ − v_ω conj S) dρ] − (1/Ω) Σ v_ω Re[ρ conj dS/dR]
//! K (every q):   dE = −(1/(Ω N_k²)) Σ_K v Re Σ_r tr[dp_r Yt_r],
//!                Yt_r = Σ_{k'} e^{ik'·t_r} Y^{k'},  Y^{k'} = Σ_s D_s(k') P^{k'}(K)^H D_s(k'−q)
//! pair FT:       bra Qb = ∂p_r/∂A_m (raised-bra kernel, binned by residue);
//!                ket −iK p_r − Qb (translation identity at ANY K — the Gamma
//!                `Q_mn = Q_nm` shortcut is wrong at K = G + q)
//! ```
//!
//! `c0 Z_tot D` in `M`: `c0 = π/(ω²Ω)` and `Z_tot` do not depend on the
//! atoms, so `h`'s G = 0 bookkeeping differentiates only through `S(k)`,
//! exactly as at Gamma (the prototype's pure-AFT `h` had none; this term and
//! the SR erfc part are anchored by 1×1×1 ≡ [`crate::grad`] and the supercell
//! anchor, both in `tests/pbc_kgrad.rs`).
//!
//! With RS-GDF J/K ([`KRsGdf`]) the two-electron lines are replaced by the
//! fitted derivative of `crate::rsgdf::kpoint::kderiv` (SR bins with the
//! energy's phases, complex Loewner metric weight per q, LR per q, `J3`'s
//! G = 0 term `M_g0(k)` through the same phase-weighted `dS` pass).
//!
//! # Supercell mapping (FINDINGS, measured)
//!
//! `E_cell = E_sc/N` and moving `A` moves all N copies, so the k-mesh force
//! IS the Gamma force on ONE copy of `A` in the diag(N) supercell (every copy
//! is equal by translation symmetry), NOT the sum over copies (that is N ×).
//!
//! # Anchor blind spots
//!
//! On meshes with every `n_i <= 2` (TRIM-only) `D(k)` is real and `q ≡ −q`:
//! a conjugation slip on `D(k)` ([`KGradMutation::ConjD`]) and a `k' − k`
//! sign slip in the exchange derivative ([`KGradMutation::WrongQ`]) are
//! algebraic IDENTITIES there. A mesh with some `n_i >= 3` is required to
//! test the phases. `ΣF` is blind to every mutant here except (sometimes)
//! `NoPhase`.
//!
//! A range-split RS-GDF build ([`crate::rsgdf::RsGdfConfig::range_split`])
//! is differentiated along its partition (`kderiv`'s module doc); its `J3`
//! G = 0 term enters through `d(S − S_ss)`, the smooth-piece part over the
//! build's pair images (`smooth_g0_force`).
//!
//! # Column rotation
//!
//! A k energy that ran the rotated SR walks (hcore `V_SR(k)`, RS-GDF SR
//! 3-centre sum) is differentiated on the rotated shells: the hcore image
//! weights `Re Δ(L)` and the fit's `Zbin` enter as `Tᵀ W T`
//! ([`crate::sr_rotation`], "Forces and stress"; `kderiv` module doc). The
//! configs must say what the build ran (typed errors otherwise).
//!
//! # Scope
//!
//! HF here (no ROHF at k); pure LDA/GGA RKS reuses this assembly with the
//! exchange fraction 0 plus the XC terms ([`crate::kxcgrad`]). Dense kernels
//! built with a range separation ([`crate::rsh`]) are differentiated with the
//! exchange walks weighted by the kernel factor `f(|K|)` (J bare) for the
//! closed-shell RKS force only (`check_inputs_rsh`; the closed-shell RKS
//! stress, [`crate::kstress`], admits them the same way); every other entry,
//! and RS-GDF with an attenuated fit, refuses them. RS-GDF aux on the
//! cell's atoms (no ghost sites). A scalar periodic ECP is differentiated by the Gamma ECP walk
//! ([`crate::ecp::periodic_ecp_gradient`]) with the SR walk's image weight
//! `Re Δ_nm(L)` in place of the density (`parts.ecp`; derivation in the
//! [`crate::ecp`] module doc, "k-points"); the walk is driven by the basis and
//! cross-checked against `hk.v_ecp`. The energy-weighted Pulay term already
//! carries the ECP through the rebuilt `F = hk.h + ...`.
//!
//! The full per-q Loewner metric weight is used, so an active lindep cut at any q is differentiated consistently (reported as
//! [`KGradient::fit_dropped_max`]).

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::ecp::{periodic_ecp_gradient_weighted, EcpGradMutation, EcpWeights};
use crate::ewald::{
    default_ewald_omega, ewald_nuclear_gradient_parts, madelung_constant, DEFAULT_EWALD_PRECISION,
};
use crate::grad::hcore_rotation_built;
use crate::grad::{ao_atoms, GRAD_NUCLEUS_EXPONENT};
use crate::hcore::kpoint::PeriodicHcoreK;
use crate::hcore::{
    gvector_list_bytes, hcore_pair_images, lr_gcut, sr_attraction_deriv_visit, PeriodicHcoreConfig,
    G_CHUNK_BYTES, ONE_E_ENGINE_PRECISION,
};
use crate::kdense_aft::KDenseAftEri;
use crate::kpts::{lattice_coords, KPointMesh};
use crate::kscf::KScfResult;
use crate::kuscf::KUScfResult;
use crate::lattice::Cell;
use crate::pair_ft::residues::{pair_ft_deriv_residues_chunked, residue_coords, residue_index};
use crate::rsgdf::deriv::{check_aux_map, fold_aux};
use crate::rsgdf::kpoint::kderiv::{
    kpoint_fit_gradient, rotate_hcore_weights, KFitGrad, KFitMutation, KRotKind,
};
use crate::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use crate::rsh::RshParams;
use crate::sr_rotation::RotatedBasis;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ndarray::{Array2, Array3};
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// Deliberate defects for the mutation tests (`tests/pbc_kgrad.rs`; the
/// prototype's `_MUTANT`). Never set in production.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KGradMutation {
    /// Bloch phase dropped on every derivative weight (Mt, Δ, Yt; the
    /// energy-side `ρ` and `P^{k'}` keep theirs).
    NoPhase,
    /// `D_s(k)`, `F_s(k)` conjugated inside the gradient (= `D(−k)` at k).
    /// IDENTITY on TRIM-only meshes.
    ConjD,
    /// Exchange derivative pairs `k'` with `k = k' + q` instead of `k' − q`
    /// (dense-AFT exchange only; a no-op on the RS-GDF path). IDENTITY on
    /// TRIM-only meshes.
    WrongQ,
    /// Primitive-cell Gamma `v_M` in the M-term instead of the mesh `v_M`
    /// (blind for `exxdiv = none` and at 1×1×1).
    GammaMadelung,
    /// Missing `1/N_k` in the overlap / kinetic `Mt(L)` (blind at 1×1×1).
    NoInvNk,
    /// RS-GDF: `e^{+iq·T}` on the SR aux images of `dJ3`.
    FitAuxPhase,
    /// RS-GDF: drop the metric derivative `Σ Wm dJ2`.
    FitNoMetric,
    /// RS-GDF: drop `J3`'s G = 0 term `M_g0`.
    FitNoG0,
    /// NOT a defect: run the FROZEN pre-parallel serial SR derivative walks
    /// (SR attraction, RS-GDF SR 3-centre and metric) instead of the
    /// ordered-parallel ones ([`crate::ordered`]); the force must be
    /// BIT-IDENTICAL either way (`tests/pbc_parallel_bitwise.rs`). Never list
    /// it as a must-fail mutant. (Unsplit RS-GDF only: the range-split walks
    /// have no serial oracle.)
    SerialDerivWalks,
    /// RS-GDF range split: the moved blocks' derivative weights with the
    /// Gamma kernel `v_SR(|K − q|)` (BLIND at 1×1×1; prototype 1.1e-3 at
    /// H2 1×1×3).
    SplitGammaKernel,
    /// RS-GDF range split: drop the image-resolved smooth-piece overlap
    /// derivative of `J3`'s G = 0 term (`d(S − S_ss)` → `dS`).
    SplitNoSmoothOverlap,
    /// RS-GDF range split: the unsplit G = 0 term in the derivative (every
    /// aux charge `q` and `dS`, not `(q_c, d(S − S_ss))`).
    SplitFullG0,
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
    /// Column rotation: the derivative walks unrotated although the energy
    /// ran rotated (the behaviour before the k rotation derivative existed).
    RotOff,
    /// Column rotation: only the hcore SR derivative walk unrotated.
    RotHcoreOff,
    /// ECP: drop the ECP force term altogether.
    NoEcp,
    /// ECP: the centre derivative `∂_C` dropped.
    EcpNoCentre,
    /// ECP: the centre derivative with the wrong sign.
    EcpCentreSign,
    /// ECP: the centre derivative folded onto the next cell atom.
    EcpCentreWrongAtom,
    /// ECP: the image weight read at the residue of `−L` (the transposed
    /// weight `W_{−L} = W_Lᵀ`).
    EcpWrongResidue,
    /// ECP: orbital image `L = 0` only.
    EcpL0Only,
    /// ECP: home ECP image `M = 0` only.
    EcpM0Only,
    /// ECP: the image weight's residue taken modulo the mesh counts `n`
    /// instead of the residue moduli (aliases images that differ by half a
    /// supercell on a shifted mesh; identity on a Gamma-centred mesh).
    EcpMeshCountModuli,
    /// Hybrid scale: the dense-AFT K derivative walks run with the full
    /// (unscaled, `a = 1`) exchange coefficients while everything else uses `a`.
    ExchDenseUnscaled,
    /// Hybrid scale: the RS-GDF fit's exchange pieces (`H`, `Z`, hence the
    /// metric weight and the fitted 3-centre weights) use the unscaled
    /// coefficients while everything else uses `a`.
    ExchFitUnscaled,
    /// Hybrid scale: the Madelung M-term keeps the unscaled `v_M`.
    ExchMadelungUnscaled,
    /// Hybrid scale: `a` also multiplies the dense-AFT Coulomb derivative.
    ExchJScaled,
    /// Hybrid scale: the exchange coefficients scaled by `a` twice (`a²`).
    ExchScaledTwice,
    /// Hybrid scale: a closed-shell spin-factor slip on the exchange
    /// coefficients (`D_s X D_s` weight 1 instead of ½).
    ExchSpinFactor,
    /// Range separation: the dense K derivative walks with the bare `4π/K²`
    /// (the kernel factor `c_sr + (c_lr − c_sr) e^{−K²/4ω²}` dropped).
    RshNoKernelFactor,
    /// Range separation: the kernel factor also applied to the Coulomb (J)
    /// derivative.
    RshKernelOnJ,
    /// Range separation: `c_sr` and `c_lr` swapped in the derivative's kernel.
    RshSwapFractions,
    /// Range separation: the plain full-Coulomb mesh Madelung constant in the
    /// M-term instead of the mixed one the SCF's K carried (blind for
    /// `exxdiv = none`).
    RshPlainMadelung,
    /// Range separation: `ω × 1.01` in the derivative's kernel.
    RshOmegaScaled,
    /// Range separation: the attenuated (erf) piece dropped from the
    /// derivative's kernel (factor `c_sr` at every K).
    RshNoErfPiece,
    /// Range separation: the Madelung part of the M-term dropped (blind for
    /// `exxdiv = none`).
    RshNoMadelungTerm,
}

/// The hcore SR derivative walk's rotation and forward-transformed image
/// weights (`None` weights = walk the parent shells).
fn hcore_sr_walk(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    mutation: Option<KGradMutation>,
    w_d: &[Array2<f64>],
) -> Result<(Option<RotatedBasis>, Option<Vec<Array2<f64>>>), FerricError> {
    let rot = hcore_rotation_built("k gradient", cell, prep, hcore_cfg, hk.sr_rotated_columns)?;
    let w_rot = rotate_hcore_weights(
        rot.as_ref(),
        KRotKind::defect(rot_kind(mutation)),
        mutation == Some(KGradMutation::RotHcoreWrongResidue),
        w_d,
    );
    // The rotation is walked only together with its transformed weights.
    let rot = rot.filter(|_| w_rot.is_some());
    Ok((rot, w_rot))
}

/// Rotated columns of the walk actually run (0 without a rotation).
pub(crate) fn walked_columns(r: Option<&RotatedBasis>) -> usize {
    r.map_or(0, |r| r.n_rotated_columns)
}

/// The column-rotation test defect a mutation requests.
fn rot_kind(m: Option<KGradMutation>) -> Option<KRotKind> {
    match m {
        Some(KGradMutation::RotDropT) => Some(KRotKind::DropT),
        Some(KGradMutation::RotUseT) => Some(KRotKind::UseT),
        Some(KGradMutation::RotOneSided) => Some(KRotKind::OneSided),
        Some(KGradMutation::RotFlipSign) => Some(KRotKind::FlipSign),
        Some(KGradMutation::RotSwapPair) => Some(KRotKind::SwapPair),
        Some(KGradMutation::RotFitWrongResidue) => Some(KRotKind::FitWrongResidue),
        Some(KGradMutation::RotHcoreWrongResidue) => Some(KRotKind::HcoreWrongResidue),
        Some(KGradMutation::RotOff) => Some(KRotKind::Off),
        Some(KGradMutation::RotHcoreOff) => Some(KRotKind::HcoreOff),
        _ => None,
    }
}

/// Settings for [`kpoint_rhf_gradient`] / [`kpoint_uhf_gradient`].
#[derive(Debug, Clone, Copy, Default)]
pub struct KGradConfig {
    /// Memory budget (bytes); `None` = ferric's unified budget.
    pub budget_bytes: Option<usize>,
    /// TEST ONLY: a deliberate defect ([`KGradMutation`]).
    #[doc(hidden)]
    pub mutation: Option<KGradMutation>,
    /// Gaussian-nucleus exponent for the SR attraction DERIVATIVE only
    /// (`None` = [`GRAD_NUCLEUS_EXPONENT`], never tighter than the energy's;
    /// see [`crate::grad::GammaGradConfig::nucleus_exponent`]).
    pub nucleus_exponent: Option<f64>,
}

/// The k-point RS-GDF J/K a force differentiates.
#[derive(Clone, Copy)]
pub struct KRsGdfGradSource<'a> {
    /// The build the SCF ran on.
    pub gdf: &'a KRsGdf,
    /// The config it was built with (the tensors are rebuilt per q; the
    /// rebuilt fit is checked against `gdf`).
    pub cfg: &'a KRsGdfConfig,
    /// Its aux basis (on the cell's atoms).
    pub aux: &'a PreparedBasis,
}

/// Where the k-point J/K come from.
#[derive(Clone, Copy)]
pub enum KGradJk<'a> {
    /// Dense pure-AFT oracle.
    Dense(&'a KDenseAftEri),
    /// k-point RS-GDF.
    RsGdf(KRsGdfGradSource<'a>),
}

/// Per-term breakdown (each `natoms × 3`, Hartree/Bohr per cell).
#[derive(Debug, Clone)]
pub struct KGradParts {
    /// `Σ Mt dS` (Pulay, `h`'s G = 0 bookkeeping, Madelung).
    pub overlap: Array2<f64>,
    /// `Σ Mt_D dT`.
    pub kinetic: Array2<f64>,
    /// SR attraction, basis-centre motion.
    pub vsr_basis: Array2<f64>,
    /// SR attraction, nucleus motion.
    pub vsr_nuc: Array2<f64>,
    /// LR attraction, basis-centre motion (pair-FT derivative).
    pub vlr_basis: Array2<f64>,
    /// LR attraction, nucleus motion (structure factor).
    pub vlr_nuc: Array2<f64>,
    /// Dense AFT Coulomb (zero on the RS-GDF path).
    pub coulomb: Array2<f64>,
    /// Dense AFT exchange (zero on the RS-GDF path).
    pub exchange: Array2<f64>,
    /// Ewald `E_nn`, real space.
    pub nn_sr: Array2<f64>,
    /// Ewald `E_nn`, reciprocal space.
    pub nn_lr: Array2<f64>,
    /// RS-GDF `dJ3`, orbital centres, SR / LR.
    pub fit_orb_sr: Array2<f64>,
    pub fit_orb_lr: Array2<f64>,
    /// RS-GDF `dJ3`, aux centres, SR / LR.
    pub fit_aux_sr: Array2<f64>,
    pub fit_aux_lr: Array2<f64>,
    /// RS-GDF metric `Σ Wm dJ2`, SR / LR.
    pub fit_metric_sr: Array2<f64>,
    pub fit_metric_lr: Array2<f64>,
    /// RS-GDF `J3` G = 0 term through `dS`.
    pub fit_g0: Array2<f64>,
    /// Periodic ECP (`Σ_L W_L ∘ dV_L`, [`crate::ecp`]); zeros without an ECP.
    pub ecp: Array2<f64>,
}

/// Output of the k-point force entry points.
#[derive(Debug, Clone)]
pub struct KGradient {
    /// `dE/dR_A` per cell atom, `natoms × 3` (Hartree/Bohr, per cell) — the
    /// force on ONE copy of each atom in the diag(N) supercell.
    pub grad: Array2<f64>,
    /// Per-term breakdown (sums to `grad`).
    pub parts: KGradParts,
    /// `max_{σ,k} max |F_σ D_σ S − S D_σ F_σ|` of the Focks rebuilt from the
    /// SCF densities (a gradient is only meaningful at a stationary state).
    /// For the restricted-open-shell force path (`kpoint_roks_gradient`,
    /// `kpoint_rohf_gradient`) the per-spin commutators do not vanish at a ROHF
    /// state, so this field holds the ROHF ORBITAL GRADIENT `g` the gate
    /// measured instead (largest spectral norm of the three blocks).
    pub commutator: f64,
    /// `v_M` used in K and M (0 for `exxdiv = none`).
    pub madelung: f64,
    /// `max |F_rebuilt − F_stored|` over k (and spins): the input consistency
    /// measure, refused above [`FOCK_CONSISTENCY_BAR`].
    pub fock_mismatch: f64,
    /// `max_x |Σ_A dE/dR_{A,x}|` (sanity only; blind to most mutants).
    pub net_force: f64,
    /// Pair images summed for `dS`/`dT`.
    pub n_images: usize,
    /// SR attraction derivative triplets.
    pub n_sr_triplets: usize,
    /// Orbital columns the hcore SR attraction derivative walk ran rotated
    /// (equals `PeriodicHcoreK::sr_rotated_columns`; 0 = unrotated).
    pub hcore_rotated_columns: usize,
    /// Kept ECP triples of the screen the walk differentiates (checked equal to
    /// the energy's `PeriodicHcoreK::n_ecp_triples`; 0 without an ECP).
    pub n_ecp_triples: usize,
    /// Full-sphere G vectors in the `V_LR` derivative.
    pub n_g_lr: usize,
    /// K vectors (all q classes) in the two-electron derivative.
    pub n_k_eri: usize,
    /// Residue pair-FT derivative chunks (all passes).
    pub n_chunks: usize,
    /// Resolved memory budget (bytes).
    pub budget_bytes: usize,
    /// RS-GDF: largest number of metric eigenvalues dropped at any q
    /// (`None` on the dense path).
    pub fit_dropped_max: Option<usize>,
    /// RS-GDF: rebuild consistency and derivative counters (`None` on the
    /// dense path).
    pub fit_checks: Option<KFitChecks>,
}

/// RS-GDF k-force diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct KFitChecks {
    /// Largest relative `‖B‖_F` mismatch per `(k, k')` between the force's
    /// rebuilt 3-index tensor and the energy build's (kept counts must match).
    pub b_norm_mismatch: f64,
    /// SR 3-centre derivative triplets.
    pub n_sr3: usize,
    /// SR metric derivative pairs.
    pub n_sr2: usize,
    /// Orbital columns the SR 3-centre derivative walk ran rotated (equals
    /// the build's `KRsGdfStats::sr_rotated_columns`; 0 = unrotated).
    pub rotated_columns: usize,
}

/// Per-spin `D_s(k)`, `F_s(k)`.
pub(crate) enum KSpins {
    /// `D` total (occupation 2), `F`.
    Restricted {
        d: Vec<Array2<C64>>,
        f: Vec<Array2<C64>>,
    },
    Unrestricted {
        da: Vec<Array2<C64>>,
        db: Vec<Array2<C64>>,
        fa: Vec<Array2<C64>>,
        fb: Vec<Array2<C64>>,
    },
    /// Restricted-open-shell (ROHF / ROKS): the unrestricted bookkeeping on
    /// the spin densities / spin Focks, with the orthonormality-Lagrangian
    /// energy-weighted matrix `W = sym(D_α F_α D_α + D_α F_β D_β)`.
    RestrictedOpen {
        da: Vec<Array2<C64>>,
        db: Vec<Array2<C64>>,
        fa: Vec<Array2<C64>>,
        fb: Vec<Array2<C64>>,
    },
}

/// Conjugate transpose.
pub(crate) fn herm(m: &Array2<C64>) -> Array2<C64> {
    m.t().mapv(|z| z.conj())
}

/// Real number as `Complex64`.
pub(crate) fn cr(x: f64) -> C64 {
    C64::new(x, 0.0)
}

/// Both-spin densities of an open-shell variant (`None` for restricted).
fn spin_pair(spins: &KSpins) -> Option<(&[Array2<C64>], &[Array2<C64>])> {
    match spins {
        KSpins::Restricted { .. } => None,
        KSpins::Unrestricted { da, db, .. } | KSpins::RestrictedOpen { da, db, .. } => {
            Some((da, db))
        }
    }
}

/// `D_α(k) + D_β(k)` of an open-shell variant (zero for restricted).
fn pair_total(spins: &KSpins, k: usize) -> Array2<C64> {
    match spin_pair(spins) {
        Some((da, db)) => &da[k] + &db[k],
        None => Array2::zeros((0, 0)),
    }
}

/// ROHF orthonormality Lagrangian `sym(D_α F_α D_α + D_α F_β D_β)` at k.
fn w_ro(
    da: &[Array2<C64>],
    db: &[Array2<C64>],
    fa: &[Array2<C64>],
    fb: &[Array2<C64>],
    k: usize,
) -> Array2<C64> {
    let w = da[k].dot(&fa[k]).dot(&da[k]) + da[k].dot(&fb[k]).dot(&db[k]);
    (&w + &herm(&w)).mapv(|z| z * 0.5)
}

impl KSpins {
    /// Number of k-points.
    pub(crate) fn nk(&self) -> usize {
        match self {
            Self::Restricted { d, .. } => d.len(),
            Self::Unrestricted { da, .. } | Self::RestrictedOpen { da, .. } => da.len(),
        }
    }

    /// Total density `D(k)` (restricted: `D`; unrestricted: `D_α + D_β`).
    pub(crate) fn total(&self, k: usize) -> Array2<C64> {
        match self {
            Self::Restricted { d, .. } => d[k].clone(),
            open => pair_total(open, k),
        }
    }

    /// `Σ_s D_s X D_s` at k (restricted `½ D X D`).
    pub(crate) fn sandwich(&self, k: usize, x: &Array2<C64>) -> Array2<C64> {
        match (spin_pair(self), self) {
            (Some((da, db)), _) => &da[k].dot(x).dot(&da[k]) + &db[k].dot(x).dot(&db[k]),
            (None, Self::Restricted { d, .. }) => d[k].dot(x).dot(&d[k]).mapv(|z| z * 0.5),
            _ => unreachable!("spin_pair is Some for every open-shell variant"),
        }
    }

    /// `W(k) = Σ_s D_s F_s D_s` (restricted `½ D F D`; restricted-open-shell:
    /// the ROHF orthonormality Lagrangian).
    pub(crate) fn energy_weighted(&self, k: usize) -> Array2<C64> {
        match self {
            Self::Restricted { d, f } => d[k].dot(&f[k]).dot(&d[k]).mapv(|z| z * 0.5),
            Self::Unrestricted { da, db, fa, fb } => {
                &da[k].dot(&fa[k]).dot(&da[k]) + &db[k].dot(&fb[k]).dot(&db[k])
            }
            Self::RestrictedOpen { da, db, fa, fb } => w_ro(da, db, fa, fb, k),
        }
    }

    /// `Σ_s D_s X D_s = Σ (c, D) c·D X D` terms.
    pub(crate) fn exch_terms(&self) -> Vec<(f64, &[Array2<C64>])> {
        match (spin_pair(self), self) {
            (Some((da, db)), _) => vec![(1.0, da), (1.0, db)],
            (None, Self::Restricted { d, .. }) => vec![(0.5, &d[..])],
            _ => unreachable!("spin_pair is Some for every open-shell variant"),
        }
    }

    /// `max_{s,k} |F D S − S D F|`.
    pub(crate) fn commutator(&self, s: &[Array2<C64>]) -> f64 {
        let one = |f: &Array2<C64>, d: &Array2<C64>, sk: &Array2<C64>| {
            let fds = f.dot(d).dot(sk);
            let sdf = sk.dot(d).dot(f);
            fds.iter()
                .zip(sdf.iter())
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).norm()))
        };
        let mut w = 0.0_f64;
        for k in 0..self.nk() {
            match self {
                Self::Restricted { d, f } => w = w.max(one(&f[k], &d[k], &s[k])),
                Self::Unrestricted { da, db, fa, fb } | Self::RestrictedOpen { da, db, fa, fb } => {
                    w = w
                        .max(one(&fa[k], &da[k], &s[k]))
                        .max(one(&fb[k], &db[k], &s[k]));
                }
            }
        }
        w
    }

    /// MUTANT ConjD: every `D_s(k)`, `F_s(k)` conjugated.
    pub(crate) fn conjugate(&mut self) {
        let cj = |v: &mut Vec<Array2<C64>>| {
            for m in v.iter_mut() {
                m.mapv_inplace(|z| z.conj());
            }
        };
        match self {
            Self::Restricted { d, f } => {
                cj(d);
                cj(f);
            }
            Self::Unrestricted { da, db, fa, fb } | Self::RestrictedOpen { da, db, fa, fb } => {
                cj(da);
                cj(db);
                cj(fa);
                cj(fb);
            }
        }
    }
}

/// k-point RHF nuclear gradient (module doc) of a converged
/// [`crate::kscf::solve_krhf_injected`] / [`crate::kscf::solve_krhf`]
/// result.
///
/// * `hcore_cfg`, `hk` — the config and output of
///   [`crate::hcore::kpoint::periodic_hcore_kpts`] the SCF used.
/// * `jk` — the J/K source the SCF used (its own Madelung setting is
///   ignored in favour of `exxdiv`).
/// * `scf` — the converged result; `F(k) = h + J − ½K` is rebuilt from
///   `D(k)`.
/// * `exxdiv` — the exchange-divergence treatment of the energy (the force
///   does not depend on it: the mesh-`v_M` M-term cancels the shift).
#[allow(clippy::too_many_arguments)]
pub fn kpoint_rhf_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KScfResult,
    exxdiv: ExxDiv,
    cfg: &KGradConfig,
) -> Result<KGradient, FerricError> {
    let who = "kpoint_rhf_gradient";
    check_inputs(
        who,
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        &[&scf.densities[..]],
        scf.converged,
        &scf.kpts,
    )?;
    let mut ledger = open_ledger(cell, prep, mesh, cfg, false)?;
    let vm = madelung_for(cell, mesh, exxdiv)?;
    let d = scf.densities.clone();
    let (j, k) = jk_build(&jk, vm, &d)?;
    let f: Vec<Array2<C64>> = (0..mesh.nk())
        .map(|q| &(&hk.h[q] + &j[q]) - &k[q].mapv(|z| z * 0.5))
        .collect();
    let sds: Vec<Array2<C64>> = (0..mesh.nk())
        .map(|q| hk.s[q].dot(&d[q]).dot(&hk.s[q]))
        .collect();
    let mismatch = check_fock_mod_madelung(who, &[("fock", &f, &scf.fock, sds)])?;
    let mut out = assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        KSpins::Restricted { d, f },
        vm,
        1.0,
        cfg,
        &mut ledger,
    )?;
    out.fock_mismatch = mismatch;
    Ok(out)
}

/// k-point UHF nuclear gradient (module doc) of a converged
/// [`crate::kuscf::solve_kuhf`] stage (`KUhfResult::scf`) or
/// [`crate::kuscf::solve_kuhf_injected`] result. `F_σ(k) = h + J[D_α + D_β]
/// − K[D_σ]` (K with the mesh `v_M` for ewald) is rebuilt from `D_σ(k)`.
/// Other arguments as [`kpoint_rhf_gradient`].
#[allow(clippy::too_many_arguments)]
pub fn kpoint_uhf_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KUScfResult,
    exxdiv: ExxDiv,
    cfg: &KGradConfig,
) -> Result<KGradient, FerricError> {
    let who = "kpoint_uhf_gradient";
    check_inputs(
        who,
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        &[&scf.density_alpha[..], &scf.density_beta[..]],
        scf.converged,
        &scf.kpts,
    )?;
    let mut ledger = open_ledger(cell, prep, mesh, cfg, true)?;
    let vm = madelung_for(cell, mesh, exxdiv)?;
    let da = scf.density_alpha.clone();
    let db = scf.density_beta.clone();
    let (ja, ka) = jk_build(&jk, vm, &da)?;
    let (jb, kb) = jk_build(&jk, vm, &db)?;
    let nk = mesh.nk();
    let base: Vec<Array2<C64>> = (0..nk).map(|q| &(&hk.h[q] + &ja[q]) + &jb[q]).collect();
    let fa: Vec<Array2<C64>> = (0..nk).map(|q| &base[q] - &ka[q]).collect();
    let fb: Vec<Array2<C64>> = (0..nk).map(|q| &base[q] - &kb[q]).collect();
    let mismatch = check_fock_uhf(who, hk, &da, &db, &fa, &fb, scf)?;
    let mut out = assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        KSpins::Unrestricted { da, db, fa, fb },
        vm,
        1.0,
        cfg,
        &mut ledger,
    )?;
    out.fock_mismatch = mismatch;
    Ok(out)
}

/// Shape and consistency guard: converged SCF, matching `omega`, no ECP, `S`/`h`/`D` shapes
/// `(nbasis, nbasis)` for all `N_k`, SCF k-points equal to the mesh's, and J/K source k-count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_inputs(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: &KGradJk<'_>,
    dens: &[&[Array2<C64>]],
    converged: bool,
    kpts: &[[f64; 3]],
) -> Result<(), FerricError> {
    check_inputs_rsh(
        who, cell, prep, mesh, hcore_cfg, hk, jk, dens, converged, kpts, None,
    )
}

/// [`check_inputs`] for an entry that differentiates a range-separated
/// exchange: `rsh` is the functional's range separation (`None` for every
/// entry but the closed-shell RKS force and stress). A dense source must carry exactly
/// these parameters (and none when `rsh` is `None`); an RS-GDF source must
/// carry none and `rsh` must be `None` (the attenuated fit has no derivative).
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_inputs_rsh(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: &KGradJk<'_>,
    dens: &[&[Array2<C64>]],
    converged: bool,
    kpts: &[[f64; 3]],
    rsh: Option<RshParams>,
) -> Result<(), FerricError> {
    let n = prep.nbasis();
    let nk = mesh.nk();
    if !converged {
        return Err(FerricError::General(format!(
            "{who}: the SCF did not converge; a gradient needs a stationary density"
        )));
    }
    if hk.omega != hcore_cfg.omega {
        return Err(FerricError::General(format!(
            "{who}: hcore was built at omega = {} but hcore_cfg.omega = {}",
            hk.omega, hcore_cfg.omega
        )));
    }
    // The hcore SR derivative follows the walk the hcore ran: a config that
    // disagrees with it is refused here, before any pass.
    hcore_rotation_built(who, cell, prep, hcore_cfg, hk.sr_rotated_columns)?;
    if hk.s.len() != nk
        || hk.h.len() != nk
        || hk.s.iter().chain(&hk.h).any(|x| x.dim() != (n, n))
        || dens
            .iter()
            .any(|d| d.len() != nk || d.iter().any(|x| x.dim() != (n, n)))
    {
        return Err(FerricError::General(format!(
            "{who}: shape mismatch (nbasis {n}, N_k {nk}: S(k) {}, D(k) {:?})",
            hk.s.len(),
            dens.iter().map(|d| d.len()).collect::<Vec<_>>()
        )));
    }
    let dk = kpts
        .iter()
        .zip(mesh.kpts())
        .map(|(a, b)| (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0_f64, f64::max))
        .fold(0.0_f64, f64::max);
    if kpts.len() != nk || dk > 1e-12 {
        return Err(FerricError::General(format!(
            "{who}: the SCF's k-points are not this mesh's (max |Δk| {dk:.2e})"
        )));
    }
    match jk {
        KGradJk::Dense(e) => {
            check_dense_rsh(who, e.rsh(), rsh)?;
            if e.nk() != nk {
                return Err(FerricError::General(format!(
                    "{who}: dense kernels have {} k-points, the mesh {nk}",
                    e.nk()
                )));
            }
        }
        KGradJk::RsGdf(s) => {
            if s.gdf.rsh().is_some() || rsh.is_some() {
                return Err(FerricError::General(format!(
                    "{who}: range-separated exchange on RS-GDF J/K (KRsGdf::build_rsh) has no \
                     derivative here (the attenuated fit is not differentiated, so the force \
                     would be wrong); the range-separated k-point force and stress cover \
                     closed-shell RKS (kpoint_rks_gradient, kpoint_rks_stress) on dense-AFT J/K \
                     only"
                )));
            }
            if s.gdf.nk() != nk {
                return Err(FerricError::General(format!(
                    "{who}: KRsGdf has {} k-points, the mesh {nk}",
                    s.gdf.nk()
                )));
            }
            check_aux_map(cell, s.aux, None)?;
        }
    }
    Ok(())
}

/// The range-separation consistency of a dense source: the kernels' `have`
/// against the entry's `want` (module doc of [`check_inputs_rsh`]).
fn check_dense_rsh(
    who: &str,
    have: Option<RshParams>,
    want: Option<RshParams>,
) -> Result<(), FerricError> {
    match (have, want) {
        (None, None) => Ok(()),
        (Some(h), Some(w)) if h == w => Ok(()),
        (Some(h), Some(w)) => Err(FerricError::General(format!(
            "{who}: the dense kernels carry a range-separated exchange {h:?} but the \
             functional's is {w:?}; build the J/K with the functional's own parameters"
        ))),
        (Some(_), None) => Err(FerricError::General(format!(
            "{who}: the dense kernels carry an attenuated (range-separated) exchange (rsh); \
             this entry point does not differentiate it (the range-separated k-point force \
             and stress cover closed-shell RKS, kpoint_rks_gradient / kpoint_rks_stress with \
             a range-separated functional, on dense-AFT J/K only)"
        ))),
        (None, Some(w)) => Err(FerricError::General(format!(
            "{who}: the functional is range-separated ({w:?}) but the dense kernels carry the \
             bare Coulomb exchange; build the J/K with KDenseAftConfig::rsh set (as solve_krks \
             does)"
        ))),
    }
}

/// Open the memory ledger and reserve the per-k matrices, residue-folded weights and
/// per-term arrays of the k-point gradient.
pub(crate) fn open_ledger(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    cfg: &KGradConfig,
    unrestricted: bool,
) -> Result<Ledger, FerricError> {
    let n = prep.nbasis();
    let nk = mesh.nk();
    let m = mesh.residue_moduli();
    let nr = m[0] * m[1] * m[2];
    let natoms = cell.positions().len();
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    // Per k: D_s, F_s, J_s, K_s, D, W, M, sandwiches (restricted ~10,
    // unrestricted ~16 complex n²).
    let per_k = if unrestricted { 16 } else { 10 };
    ledger.reserve(
        &format!("k gradient per-k complex n×n matrices (n = {n}, N_k = {nk})"),
        bytes_of((nk * n * n) as u64, 16 * per_k),
    )?;
    // Per residue: Δ, Δ_true (complex), and five real weight sets (the sixth
    // slot is the column-rotated copy of the SR image weight).
    ledger.reserve(
        &format!("k gradient residue-folded weights (R = {nr}, n = {n})"),
        bytes_of((nr * n * n) as u64, 16 * 2 + 8 * 6),
    )?;
    ledger.reserve(
        &format!("k gradient per-term arrays (natoms = {natoms}, nao = {n})"),
        bytes_of(((natoms + n) * 3) as u64, 8 * 24),
    )?;
    Ok(ledger)
}

/// `v_M` of the mesh: 0 for `ExxDiv::None`, the mesh Madelung constant for `ExxDiv::Ewald`.
fn madelung_for(cell: &Cell, mesh: &KPointMesh, exxdiv: ExxDiv) -> Result<f64, FerricError> {
    Ok(match exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => mesh.madelung(cell)?,
    })
}

type KJk = (Vec<Array2<C64>>, Vec<Array2<C64>>);

/// `J(k)`, `K(k)` (K with `v_M S D S`) of `dm` from the source.
pub(crate) fn jk_build(jk: &KGradJk<'_>, vm: f64, dm: &[Array2<C64>]) -> Result<KJk, FerricError> {
    let nk = dm.len();
    let n = dm[0].nrows();
    let mut j: Vec<Array2<C64>> = (0..nk).map(|_| Array2::zeros((n, n))).collect();
    let mut k: Vec<Array2<C64>> = (0..nk).map(|_| Array2::zeros((n, n))).collect();
    match jk {
        KGradJk::Dense(e) => e.contract(dm, vm, &mut j, &mut k)?,
        KGradJk::RsGdf(s) => s.gdf.contract(dm, vm, &mut j, &mut k)?,
    }
    Ok((j, k))
}

/// Per residue `r`: `w[r][m, n] = Re(scale Σ_k ph[k][r] X(k)[n, m])` — the
/// real weight of the derivative block `(m_0, n_L)`, `L ≡ r`.
pub(crate) fn fold_real(mats: &[Array2<C64>], ph: &[Vec<C64>], scale: f64) -> Vec<Array2<f64>> {
    let nr = ph[0].len();
    let n = mats[0].nrows();
    (0..nr)
        .map(|r| {
            let mut acc = Array2::<C64>::zeros((n, n));
            for (k, x) in mats.iter().enumerate() {
                acc.scaled_add(ph[k][r], x);
            }
            Array2::from_shape_fn((n, n), |(m, nn)| scale * acc[(nn, m)].re)
        })
        .collect()
}

/// Per residue `r`: `(1/N_k) Σ_k ph[k][r] X(k)` (complex, `[n, m]` as `X`).
pub(crate) fn fold_complex(mats: &[Array2<C64>], ph: &[Vec<C64>], scale: f64) -> Vec<Array2<C64>> {
    let nr = ph[0].len();
    let n = mats[0].nrows();
    (0..nr)
        .map(|r| {
            let mut acc = Array2::<C64>::zeros((n, n));
            for (k, x) in mats.iter().enumerate() {
                acc.scaled_add(ph[k][r] * scale, x);
            }
            acc
        })
        .collect()
}

/// `ρ = Σ_r Σ_mn P[r]_mn(g) w[r]_nm`.
pub(crate) fn residue_trace(p: &[Array3<C64>], gi: usize, w: &[Array2<C64>]) -> C64 {
    let n = w[0].nrows();
    let mut acc = C64::new(0.0, 0.0);
    for (pr, wr) in p.iter().zip(w) {
        for m in 0..n {
            for nn in 0..n {
                acc += pr[[m, nn, gi]] * wr[(nn, m)];
            }
        }
    }
    acc
}

/// `g_ao[m] += Re Σ_r Σ_n Qb[r]_mn (s w[r])_nm`,
/// `g_ao[n] += Re Σ_r Σ_m (−iK P[r] − Qb[r])_mn (s w[r])_nm` (bra and ket
/// motion of the residue-resolved pair FT at `K = kvec`).
fn contract_pair_deriv(
    g_ao: &mut [[f64; 3]],
    kvec: &[f64; 3],
    gi: usize,
    p: &[Array3<C64>],
    q: &[[Array3<C64>; 3]],
    w: &[Array2<C64>],
    scale: C64,
) {
    let n = g_ao.len();
    for ((pr, qr), wr) in p.iter().zip(q).zip(w) {
        for m in 0..n {
            for nn in 0..n {
                let wv = scale * wr[(nn, m)];
                if wv.re == 0.0 && wv.im == 0.0 {
                    continue;
                }
                let pv = pr[[m, nn, gi]];
                for x in 0..3 {
                    let qb = qr[x][[m, nn, gi]];
                    g_ao[m][x] += (qb * wv).re;
                    let qk = C64::new(0.0, -kvec[x]) * pv - qb;
                    g_ao[nn][x] += (qk * wv).re;
                }
            }
        }
    }
}

/// `g[atom(s1)] += Σ w_μν ∂_bra`, `g[atom(s2)] += Σ w_μν ∂_ket` (the Gamma
/// `add_pair_deriv` with a per-image weight).
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

/// Sum the per-AO Cartesian gradient `g_ao` into a `(natoms, 3)` array by owning atom `aoat[μ]`.
fn fold_ao(g_ao: &[[f64; 3]], aoat: &[usize], natoms: usize) -> Array2<f64> {
    let mut g = Array2::<f64>::zeros((natoms, 3));
    for (mu, v) in g_ao.iter().enumerate() {
        for x in 0..3 {
            g[(aoat[mu], x)] += v[x];
        }
    }
    g
}

/// The Madelung constant of the M-term: the mesh `v_M` scaled by the exchange
/// fraction (`GammaMadelung` mutant: the primitive-cell constant).
fn m_term_madelung(
    cell: &Cell,
    vm: f64,
    exch_scale: f64,
    mutation: Option<KGradMutation>,
) -> Result<f64, FerricError> {
    let base = if mutation == Some(KGradMutation::GammaMadelung) && vm != 0.0 {
        madelung_constant(cell)?
    } else {
        vm
    };
    Ok(exch_scale * base)
}

/// The range separation of a dense source (`check_inputs_rsh` admits it on
/// the closed-shell RKS force only) and the `v_M` the M-term uses: the SCF's
/// (mixed, for a range-separated K) `vm`, or the plain mesh constant under
/// the [`KGradMutation::RshPlainMadelung`] defect.
fn rsh_and_m_term_vm(
    jk: &KGradJk<'_>,
    cell: &Cell,
    mesh: &KPointMesh,
    vm: f64,
    mutation: Option<KGradMutation>,
) -> Result<(Option<RshParams>, f64), FerricError> {
    let rsh = match jk {
        KGradJk::Dense(e) => e.rsh(),
        KGradJk::RsGdf(_) => None,
    };
    let defect = |m: KGradMutation| rsh.is_some() && vm != 0.0 && mutation == Some(m);
    let vm_used = if defect(KGradMutation::RshPlainMadelung) {
        mesh.madelung(cell)?
    } else if defect(KGradMutation::RshNoMadelungTerm) {
        0.0
    } else {
        vm
    };
    Ok((rsh, vm_used))
}

/// The scale each exchange-bearing piece of [`assemble`] carries: the
/// exchange fraction `a` everywhere, except where a test defect leaves one
/// piece wrong (`KGradMutation::Exch*`).
struct ExchScales {
    /// Dense-AFT K walks (coefficient scale).
    dense: f64,
    /// RS-GDF fit exchange pieces (coefficient scale).
    fit: f64,
    /// Madelung M-term.
    madelung: f64,
    /// Dense-AFT Coulomb derivative (1 in production).
    coulomb: f64,
}

impl ExchScales {
    fn of(a: f64, mutation: Option<KGradMutation>) -> Self {
        use KGradMutation as M;
        let coeff = match mutation {
            Some(M::ExchScaledTwice) => a * a,
            Some(M::ExchSpinFactor) => 2.0 * a,
            _ => a,
        };
        // An unscaled piece is only meaningful when exchange exists at all.
        let pick = |wrong: M| {
            if mutation == Some(wrong) && a != 0.0 {
                1.0
            } else {
                coeff
            }
        };
        Self {
            dense: pick(M::ExchDenseUnscaled),
            fit: pick(M::ExchFitUnscaled),
            madelung: if mutation == Some(M::ExchMadelungUnscaled) {
                1.0
            } else {
                a
            },
            coulomb: if mutation == Some(M::ExchJScaled) {
                a
            } else {
                1.0
            },
        }
    }
}

/// The exchange terms `Σ_s c D_s X D_s` of `spins`, coefficients scaled by the
/// exchange fraction (`1` is bitwise the unscaled terms; empty for `0`).
pub(crate) fn scaled_exch_terms(spins: &KSpins, exch_scale: f64) -> Vec<(f64, &[Array2<C64>])> {
    if exch_scale == 0.0 {
        return Vec::new();
    }
    spins
        .exch_terms()
        .into_iter()
        .map(|(c, d)| (c * exch_scale, d))
        .collect()
}

/// Assemble the k-point gradient from the per-spin `D(k)`/`F(k)`, hcore and J/K derivative
/// pieces; `vm` is the Madelung shift and `exch_scale` the exact-exchange
/// fraction `a` of the energy (`1` = Hartree-Fock, bitwise the unscaled
/// terms; `0` = no exchange at all: the K-derivative walks are skipped, the
/// RS-GDF fit differentiates the Coulomb part only and the Madelung M-term
/// vanishes; `0 < a < 1` scales the exchange coefficients of the K walks and
/// of the fit's exchange pieces and the Madelung M-term by `a`, validated by
/// the global-hybrid KS force, [`crate::kxcgrad`]).
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
    cfg: &KGradConfig,
    ledger: &mut Ledger,
) -> Result<KGradient, FerricError> {
    let n = prep.nbasis();
    let nk = mesh.nk();
    let natoms = cell.positions().len();
    let mutation = cfg.mutation;
    let s_k = &hk.s;
    let commutator = spins.commutator(s_k);
    let mut spins = spins;
    if mutation == Some(KGradMutation::ConjD) {
        spins.conjugate();
    }

    // --- Phases per (k, residue): energy-side (true) and derivative-side.
    let moduli = mesh.residue_moduli();
    let nr = moduli[0] * moduli[1] * moduli[2];
    let ph_true: Vec<Vec<C64>> = (0..nk)
        .map(|k| {
            (0..nr)
                .map(|r| mesh.phase(k, residue_coords(r, moduli)))
                .collect()
        })
        .collect();
    let ph_d: Vec<Vec<C64>> = if mutation == Some(KGradMutation::NoPhase) {
        vec![vec![cr(1.0); nr]; nk]
    } else {
        ph_true.clone()
    };
    let inv_nk = 1.0 / nk as f64;

    // --- D(k), M(k).
    let dtot: Vec<Array2<C64>> = (0..nk).map(|k| spins.total(k)).collect();
    let zs = cell.nuclear_charges();
    let ztot: f64 = zs.iter().sum();
    let omega = hcore_cfg.omega;
    let vol = cell.volume();
    let c0 = PI / (omega * omega * vol);
    let sc = ExchScales::of(exch_scale, mutation);
    let (rsh, vm_used) = rsh_and_m_term_vm(jk, cell, mesh, vm, mutation)?;
    let vm_m = m_term_madelung(cell, vm_used, sc.madelung, mutation)?;
    // The exchange coefficients, scaled by the exact-exchange fraction. The
    // dense walks and the RS-GDF fit take them separately so a test defect can
    // leave exactly one of the two unscaled.
    let exch_dense = scaled_exch_terms(&spins, sc.dense);
    let exch_fit = scaled_exch_terms(&spins, sc.fit);
    let j_scale = sc.coulomb;
    let mk: Vec<Array2<C64>> = (0..nk)
        .map(|k| {
            let mut m = spins.energy_weighted(k).mapv(|z| -z);
            m.scaled_add(cr(c0 * ztot), &dtot[k]);
            if vm_m != 0.0 {
                m.scaled_add(cr(-vm_m), &spins.sandwich(k, &s_k[k]));
            }
            m
        })
        .collect();
    let nrm1 = if mutation == Some(KGradMutation::NoInvNk) {
        1.0
    } else {
        inv_nk
    };
    let w_s = fold_real(&mk, &ph_d, nrm1);
    let w_t = fold_real(&dtot, &ph_d, nrm1);
    let w_d = fold_real(&dtot, &ph_d, inv_nk);
    let delta = fold_complex(&dtot, &ph_d, inv_nk);
    let delta_true = fold_complex(&dtot, &ph_true, inv_nk);
    drop(mk);

    // --- RS-GDF fitted two-electron pieces first (M_g0 enters the dS pass).
    let fit = match jk {
        KGradJk::Dense(_) => None,
        KGradJk::RsGdf(src) => {
            let fm = KFitMutation {
                rot: KRotKind::defect(rot_kind(mutation)),
                aux_phase: mutation == Some(KGradMutation::FitAuxPhase),
                no_metric: mutation == Some(KGradMutation::FitNoMetric),
                no_g0: mutation == Some(KGradMutation::FitNoG0),
                serial_sr: mutation == Some(KGradMutation::SerialDerivWalks),
                split_gamma_kernel: mutation == Some(KGradMutation::SplitGammaKernel),
                split_full_g0: mutation == Some(KGradMutation::SplitFullG0),
                lr_wrong_q: false,
                no_loewner_kd: false,
                cross_spin: false,
                coulomb_scale: (sc.coulomb != 1.0).then_some(sc.coulomb),
            };
            Some(kpoint_fit_gradient(
                src.gdf, src.cfg, cell, prep, src.aux, mesh, s_k, &dtot, &exch_fit, fm, ledger,
            )?)
        }
    };
    let w_g0 = fit.as_ref().map(|f| fold_real(&f.mg0, &ph_d, 1.0));

    // --- dS, dT (and the fit's G = 0 dS) over the energy's pair images.
    let images = hcore_pair_images(cell, prep, hcore_cfg.precision, ledger)?;
    let b = cell.reciprocal();
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
            let r = residue_index(lattice_coords(&b, l), moduli);
            for s1 in 0..nsh {
                for s2 in 0..nsh {
                    if let Some(blk) = eng_s.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        add_pair_deriv(&mut g_s, blk, &w_s[r], &dims, &offs, &sh2at, s1, s2);
                        if let Some(wg) = &w_g0 {
                            add_pair_deriv(
                                &mut g_fit_g0,
                                blk,
                                &wg[r],
                                &dims,
                                &offs,
                                &sh2at,
                                s1,
                                s2,
                            );
                        }
                    }
                    if let Some(blk) = eng_t.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        add_pair_deriv(&mut g_t, blk, &w_t[r], &dims, &offs, &sh2at, s1, s2);
                    }
                }
            }
        }
    }
    g_fit_g0 -= &smooth_g0_force(fit.as_ref(), w_g0.as_deref(), cell, moduli, mutation)?;

    // --- V_SR: the Gamma walk, each image weighted by Re Δ(L).
    let sr_cfg = PeriodicHcoreConfig {
        nucleus_exponent: cfg
            .nucleus_exponent
            .unwrap_or(GRAD_NUCLEUS_EXPONENT)
            .min(hcore_cfg.nucleus_exponent),
        ..*hcore_cfg
    };
    let mut g_vsr_basis = Array2::<f64>::zeros((natoms, 3));
    let mut g_vsr_nuc = Array2::<f64>::zeros((natoms, 3));
    let serial = mutation == Some(KGradMutation::SerialDerivWalks);
    // The energy's SR walk: rotated shells (weights `Tᵀ w T`) when `hk` ran
    // rotated.
    let (rot, w_rot) = hcore_sr_walk(cell, prep, hcore_cfg, hk, mutation, &w_d)?;
    let w_sr = w_rot.as_deref().unwrap_or(&w_d);
    let rot_walk = rot.as_ref();
    let n_sr_triplets =
        sr_attraction_deriv_visit(cell, prep, rot_walk, &sr_cfg, ledger, serial, |t| {
            let wd = &w_sr[residue_index(lattice_coords(&b, &t.l), moduli)];
            for i in 0..t.dim1 {
                for j in 0..t.dim2 {
                    let coeff = t.f * wd[(t.off1 + i, t.off2 + j)];
                    if coeff == 0.0 {
                        continue;
                    }
                    let idx = i * t.dim2 + j;
                    let idx_swapped = j * t.dim1 + i;
                    for c in 0..3 {
                        let d1 = coeff * t.ket_a[c * t.bs + idx_swapped];
                        let d2 = coeff * t.ket_b[c * t.bs + idx];
                        g_vsr_basis[(t.at1, c)] += d1;
                        g_vsr_basis[(t.at2, c)] += d2;
                        g_vsr_nuc[(t.atc, c)] -= d1 + d2;
                    }
                }
            }
        })?;

    // --- V_ECP: the Gamma ECP walk with the SR walk's image weight W_L = w_d[r(L)].
    let (g_ecp, n_ecp_triples) =
        ecp_force_term(cell, prep, mesh, hcore_cfg, hk, &w_d, mutation, ledger)?;

    // --- V_LR: full G sphere of periodic_hcore_kpts, residue pair-FT
    // derivative at q = 0.
    let aoat = ao_atoms(prep);
    let pos = cell.positions();
    let gcut_lr = lr_gcut(prep, omega, hcore_cfg.precision);
    ledger.reserve(
        &format!("k gradient V_LR G list (|G| <= {gcut_lr:.3})"),
        gvector_list_bytes(cell, gcut_lr)?,
    )?;
    let gv_lr: Vec<[f64; 3]> = cell
        .gvectors(gcut_lr)?
        .into_iter()
        .filter(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2] > 0.0)
        .collect();
    let mut vlr_ao = vec![[0.0_f64; 3]; n];
    let mut g_vlr_nuc = Array2::<f64>::zeros((natoms, 3));
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let mut n_chunks = pair_ft_deriv_residues_chunked(
        cell,
        prep,
        &gv_lr,
        moduli,
        0.1 * hcore_cfg.precision,
        chunk_budget,
        0,
        |_g0, gs, p, q| {
            for (gi, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let v = 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp();
                let mut sg = C64::new(0.0, 0.0);
                for (z, r) in zs.iter().zip(&pos) {
                    let a = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                    sg += C64::new(z * a.cos(), -z * a.sin());
                }
                let rho = residue_trace(p, gi, &delta_true);
                // basis: −(1/Ω) v Re[conj(S) dρ]
                contract_pair_deriv(&mut vlr_ao, gvec, gi, p, q, &delta, sg.conj() * (-v / vol));
                // nucleus: −(1/Ω) v Re[ρ conj(dS/dR_A)], dS/dR_A = −iG Z_A e^{−iG·R_A}
                for (a, r) in pos.iter().enumerate() {
                    if zs[a] == 0.0 {
                        continue;
                    }
                    let ph = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                    let t = v / vol * zs[a] * (rho.re * ph.sin() + rho.im * ph.cos());
                    for c in 0..3 {
                        g_vlr_nuc[(a, c)] += t * gvec[c];
                    }
                }
            }
            Ok(())
        },
    )?;
    let g_vlr_basis = fold_ao(&vlr_ao, &aoat, natoms);
    drop(vlr_ao);

    // --- Two-electron.
    let zeros = || Array2::<f64>::zeros((natoms, 3));
    let (mut g_j, mut g_k) = (zeros(), zeros());
    let n_k_eri: usize;
    let mut fit_parts = [zeros(), zeros(), zeros(), zeros(), zeros(), zeros()];
    let mut fit_dropped_max = None;
    let mut fit_checks = None;
    match jk {
        KGradJk::Dense(eri) => {
            let (jao, kao, ch, nkv) = dense_two_electron(
                cell,
                prep,
                mesh,
                eri,
                &exch_dense,
                rsh_kernel(rsh, mutation),
                j_scale,
                &delta,
                &delta_true,
                &ph_true,
                &ph_d,
                mutation,
                ledger,
            )?;
            g_j = fold_ao(&jao, &aoat, natoms);
            g_k = fold_ao(&kao, &aoat, natoms);
            n_chunks += ch;
            n_k_eri = nkv;
        }
        KGradJk::RsGdf(src) => {
            let f = fit.ok_or_else(|| {
                FerricError::General("k gradient: RS-GDF pieces missing (internal)".into())
            })?;
            let fold = |x: &Array2<f64>| fold_aux(x, src.aux, None, natoms);
            fit_parts = [
                f.orb_sr.clone(),
                f.orb_lr.clone(),
                fold(&f.aux_sr),
                fold(&f.aux_lr),
                fold(&f.metric_sr),
                fold(&f.metric_lr),
            ];
            n_chunks += f.n_chunks;
            n_k_eri = f.n_k_lr;
            fit_dropped_max = Some(f.n_dropped_max);
            fit_checks = Some(KFitChecks {
                b_norm_mismatch: f.b_norm_mismatch,
                n_sr3: f.n_sr3,
                n_sr2: f.n_sr2,
                rotated_columns: f.rotated_columns,
            });
        }
    }

    // --- Ewald E_nn.
    let (nn_sr, nn_lr) =
        ewald_nuclear_gradient_parts(cell, default_ewald_omega(cell), DEFAULT_EWALD_PRECISION)?;
    let to_arr = |v: &[[f64; 3]]| Array2::from_shape_fn((natoms, 3), |(a, c)| v[a][c]);
    let g_nn_sr = to_arr(&nn_sr);
    let g_nn_lr = to_arr(&nn_lr);

    let [f_orb_sr, f_orb_lr, f_aux_sr, f_aux_lr, f_met_sr, f_met_lr] = fit_parts;
    let grad = &g_s
        + &g_t
        + &g_vsr_basis
        + &g_vsr_nuc
        + &g_ecp
        + &g_vlr_basis
        + &g_vlr_nuc
        + &g_j
        + &g_k
        + &g_nn_sr
        + &g_nn_lr
        + &f_orb_sr
        + &f_orb_lr
        + &f_aux_sr
        + &f_aux_lr
        + &f_met_sr
        + &f_met_lr
        + &g_fit_g0;
    if grad.iter().any(|v| !v.is_finite()) {
        return Err(FerricError::General("k gradient: non-finite force".into()));
    }
    let net_force = (0..3)
        .map(|c| grad.column(c).sum().abs())
        .fold(0.0_f64, f64::max);
    ferric_core::memory::warn_if_rss_over("ferric-pbc k gradient", ledger.budget(), 1.1);
    Ok(KGradient {
        grad,
        parts: KGradParts {
            overlap: g_s,
            kinetic: g_t,
            vsr_basis: g_vsr_basis,
            vsr_nuc: g_vsr_nuc,
            vlr_basis: g_vlr_basis,
            vlr_nuc: g_vlr_nuc,
            coulomb: g_j,
            exchange: g_k,
            nn_sr: g_nn_sr,
            nn_lr: g_nn_lr,
            fit_orb_sr: f_orb_sr,
            fit_orb_lr: f_orb_lr,
            fit_aux_sr: f_aux_sr,
            fit_aux_lr: f_aux_lr,
            fit_metric_sr: f_met_sr,
            fit_metric_lr: f_met_lr,
            fit_g0: g_fit_g0,
            ecp: g_ecp,
        },
        commutator,
        madelung: vm,
        fock_mismatch: 0.0,
        net_force,
        n_images: images.len(),
        n_sr_triplets,
        hcore_rotated_columns: walked_columns(rot_walk),
        n_ecp_triples,
        n_g_lr: gv_lr.len(),
        n_k_eri,
        n_chunks,
        budget_bytes: ledger.budget(),
        fit_dropped_max,
        fit_checks,
    })
}

/// Largest `|F_rebuilt − F_stored|` the entry points accept (Hartree). The
/// forces and the stress rebuild the Fock matrix from the density and the
/// J/K source (`h + J − αK`, plus `V_xc` in the KS force entry); the SCF's
/// stored (non-extrapolated) Fock must agree, which fails for a wrong `exxdiv`
/// (`v_M S D S` shift), an SCF run without the ECP that `hk.h` carries, a
/// different J/K source, and (for the HF entries, which rebuild no `V_xc`) a
/// KS result. Measured on H2 1x1x2: matching inputs 2e-16 (none) and 2e-16
/// (ewald); the wrong `exxdiv` 7.9e-1 either way; an LDA result given to an HF
/// entry 3.4e-1; HI 1x1x3 matching 5.7e-17, an ECP-free SCF 6.9; the KS force
/// entry with the wrong functional 5.9e-3.
pub const FOCK_CONSISTENCY_BAR: f64 = 1e-8;

pub(crate) fn check_fock(
    who: &str,
    what: &str,
    rebuilt: &[Array2<C64>],
    stored: &[Array2<C64>],
    remedy: &str,
) -> Result<f64, FerricError> {
    let mut worst = 0.0_f64;
    for (a, b) in rebuilt.iter().zip(stored) {
        if a.dim() != b.dim() {
            return Err(FerricError::General(format!(
                "{who}: stored {what} has a different shape than the rebuilt one"
            )));
        }
        for (x, y) in a.iter().zip(b.iter()) {
            worst = worst.max((x - y).norm());
        }
    }
    if rebuilt.len() != stored.len() || !(worst <= FOCK_CONSISTENCY_BAR) {
        return Err(FerricError::General(format!(
            "{who}: the Fock matrix rebuilt from the density and the J/K source differs from \
             the SCF's stored {what} by {worst:.2e} (bar {FOCK_CONSISTENCY_BAR:.0e}); the analytic \
             derivative needs the SCF's own Fock (same J/K kernels, hcore and exxdiv, not an \
             extrapolated state); {remedy}"
        )));
    }
    Ok(worst)
}

/// Force-side Fock guard. The force does not depend on `exxdiv` (the mesh
/// `v_M` shift of `K` is cancelled by the M-term), so an SCF run at another
/// `exxdiv` is legitimate: its stored Fock differs from the rebuilt one by
/// exactly `c · S D_σ S` with one scalar `c` (the `v_M` difference). The check
/// therefore removes the best-fit `c · S D_σ S` (shared by all k and spins)
/// before applying [`FOCK_CONSISTENCY_BAR`]; what survives (a missing ECP, a
/// KS potential, a different J/K source) is not of that form. Each group is
/// `(what, rebuilt F, stored F, S D_σ S)`. Returns the surviving mismatch.
pub(crate) fn check_fock_mod_madelung(
    who: &str,
    groups: &[(&str, &[Array2<C64>], &[Array2<C64>], Vec<Array2<C64>>)],
) -> Result<f64, FerricError> {
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (what, rebuilt, stored, x) in groups {
        if rebuilt.len() != stored.len() || x.len() != stored.len() {
            return Err(FerricError::General(format!(
                "{who}: stored {what} has {} k-points, the rebuilt one {}",
                stored.len(),
                rebuilt.len()
            )));
        }
        for ((a, b), xk) in rebuilt.iter().zip(stored.iter()).zip(x) {
            if a.dim() != b.dim() {
                return Err(FerricError::General(format!(
                    "{who}: stored {what} has a different shape than the rebuilt one"
                )));
            }
            for ((ra, sb), xv) in a.iter().zip(b.iter()).zip(xk.iter()) {
                num += ((sb - ra) * xv.conj()).re;
                den += xv.norm_sqr();
            }
        }
    }
    let c = if den > 0.0 { num / den } else { 0.0 };
    let mut worst = 0.0_f64;
    for (_, rebuilt, stored, x) in groups {
        for ((a, b), xk) in rebuilt.iter().zip(stored.iter()).zip(x) {
            for ((ra, sb), xv) in a.iter().zip(b.iter()).zip(xk.iter()) {
                worst = worst.max((sb - ra - xv * c).norm());
            }
        }
    }
    if !(worst <= FOCK_CONSISTENCY_BAR) {
        return Err(FerricError::General(format!(
            "{who}: the Fock matrix rebuilt from the density and the J/K source differs from \
             the SCF's stored Fock by {worst:.2e} beyond a Madelung (exxdiv) shift (bar \
             {FOCK_CONSISTENCY_BAR:.0e}); the analytic derivative covers Hartree-Fock with the \
             SAME J/K kernels and hcore the SCF used (not an SCF run without the ECP, not an \
             extrapolated state); a KS result needs `kpoint_rks_gradient`"
        )));
    }
    Ok(worst)
}

/// UHF form of the force-side Fock guard ([`check_fock_mod_madelung`]).
fn check_fock_uhf(
    who: &str,
    hk: &PeriodicHcoreK,
    da: &[Array2<C64>],
    db: &[Array2<C64>],
    fa: &[Array2<C64>],
    fb: &[Array2<C64>],
    scf: &KUScfResult,
) -> Result<f64, FerricError> {
    let sds = |dm: &[Array2<C64>]| -> Vec<Array2<C64>> {
        (0..dm.len())
            .map(|q| hk.s[q].dot(&dm[q]).dot(&hk.s[q]))
            .collect()
    };
    check_fock_mod_madelung(
        who,
        &[
            ("fock_alpha", fa, &scf.fock_alpha, sds(da)),
            ("fock_beta", fb, &scf.fock_beta, sds(db)),
        ],
    )
}

/// Largest `|h − T − V − V_ECP|` over k that `hk` may show (Hartree): `h` is
/// assembled from exactly these pieces, so anything above roundoff means the
/// ECP was stripped from (or added to) one of them after the build.
const HK_ECP_CONSISTENCY_BAR: f64 = 1e-10;

/// Presence/count cross-check of the ECP walk against `hk`: `kept` is the
/// walk's kept-triple count (`None` when the basis has no ECP centre).
/// Errors when the walk and `hk.v_ecp` disagree in presence or count, or when
/// `hk.h` is not `T + V (+ V_ECP)` (an `hk` whose `h` lost its ECP but kept
/// the marker, or the reverse). Returns the count (0 without an ECP).
pub(crate) fn ecp_presence(
    who: &str,
    hk: &PeriodicHcoreK,
    kept: Option<usize>,
) -> Result<usize, FerricError> {
    let mut worst = 0.0_f64;
    for k in 0..hk.h.len() {
        let mut r = &hk.h[k] - &hk.t[k] - &hk.v[k];
        if let Some(ve) = &hk.v_ecp {
            r -= &ve[k];
        }
        worst = worst.max(r.iter().fold(0.0_f64, |m, z| m.max(z.norm())));
    }
    match (&hk.v_ecp, kept) {
        (Some(_), Some(n)) => {
            if n != hk.n_ecp_triples {
                return Err(FerricError::General(format!(
                    "{who}: the ECP screen keeps {n} triples but hcore's V_ECP kept {}; hcore \
                     was built with a different ECP precision/config than hcore_cfg (or at \
                     another geometry)",
                    hk.n_ecp_triples
                )));
            }
            if !(worst <= HK_ECP_CONSISTENCY_BAR) {
                return Err(FerricError::General(format!(
                    "{who}: hcore's h differs from T + V + V_ECP by {worst:.2e} (bar \
                     {HK_ECP_CONSISTENCY_BAR:.0e}); pass the PeriodicHcoreK the SCF used"
                )));
            }
            Ok(n)
        }
        (None, None) => {
            if !(worst <= HK_ECP_CONSISTENCY_BAR) {
                return Err(FerricError::General(format!(
                    "{who}: hcore's h differs from T + V by {worst:.2e} (bar \
                     {HK_ECP_CONSISTENCY_BAR:.0e}); pass the PeriodicHcoreK the SCF used"
                )));
            }
            Ok(0)
        }
        (Some(_), None) => Err(FerricError::General(format!(
            "{who}: hcore carries V_ECP but the basis/cell has no ECP centre"
        ))),
        (None, Some(_)) => Err(FerricError::General(format!(
            "{who}: the basis carries an ECP for a cell atom but hcore has no V_ECP; pass the \
             PeriodicHcoreK the SCF used"
        ))),
    }
}

/// The ECP force walk at k with the image weight `w_d` (residue-resolved,
/// [`fold_real`]); `None` when the basis has no ECP centre. Shared by the
/// production term and the fixed-density entry.
fn ecp_force_walk(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    w_d: &[Array2<f64>],
    mutation: Option<KGradMutation>,
    ledger: &mut Ledger,
) -> Result<Option<crate::ecp::PeriodicEcpGradient>, FerricError> {
    let ecp_mut = match mutation {
        Some(KGradMutation::EcpNoCentre) => Some(EcpGradMutation::NoCentre),
        Some(KGradMutation::EcpCentreSign) => Some(EcpGradMutation::CentreSign),
        Some(KGradMutation::EcpCentreWrongAtom) => Some(EcpGradMutation::CentreWrongAtom),
        Some(KGradMutation::EcpL0Only) => Some(EcpGradMutation::L0Only),
        Some(KGradMutation::EcpM0Only) => Some(EcpGradMutation::M0Only),
        Some(KGradMutation::SerialDerivWalks) => Some(EcpGradMutation::SerialImages),
        _ => None,
    };
    let weights = EcpWeights::Residue {
        w: w_d,
        recip: cell.reciprocal(),
        moduli: if mutation == Some(KGradMutation::EcpMeshCountModuli) {
            mesh.n()
        } else {
            mesh.residue_moduli()
        },
        flip_l: mutation == Some(KGradMutation::EcpWrongResidue),
    };
    periodic_ecp_gradient_weighted(
        cell,
        prep,
        &hcore_cfg.ecp_config(),
        &weights,
        ecp_mut,
        ledger,
    )
}

/// TEST/DIAGNOSTIC: the periodic-ECP force of the k energy `(1/N_k) Σ_k
/// Re tr[D(k) V_ECP(k)]` at a FIXED total density `dtot` (one `n × n` matrix
/// per mesh point, Hermitian). No SCF: the hcore gate that isolates the ECP
/// term before anything else is trusted. `None` when the basis has no ECP.
#[doc(hidden)]
pub fn kpoint_ecp_gradient_fixed_d(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    dtot: &[Array2<C64>],
) -> Result<Option<crate::ecp::PeriodicEcpGradient>, FerricError> {
    kpoint_ecp_gradient_fixed_d_with(cell, prep, mesh, hcore_cfg, dtot, None)
}

/// [`kpoint_ecp_gradient_fixed_d`] with a test-only [`KGradMutation`] (the ECP
/// ones, `NoPhase` and `ConjD` act here; the rest are ignored).
#[doc(hidden)]
pub fn kpoint_ecp_gradient_fixed_d_with(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    dtot: &[Array2<C64>],
    mutation: Option<KGradMutation>,
) -> Result<Option<crate::ecp::PeriodicEcpGradient>, FerricError> {
    let n = prep.nbasis();
    if dtot.len() != mesh.nk() || dtot.iter().any(|d| d.dim() != (n, n)) {
        return Err(FerricError::General(format!(
            "kpoint_ecp_gradient_fixed_d: expected {} density matrices of {n}×{n}",
            mesh.nk()
        )));
    }
    let moduli = mesh.residue_moduli();
    let nr = moduli[0] * moduli[1] * moduli[2];
    let ph: Vec<Vec<C64>> = (0..mesh.nk())
        .map(|k| {
            (0..nr)
                .map(|r| {
                    if mutation == Some(KGradMutation::NoPhase) {
                        cr(1.0)
                    } else {
                        mesh.phase(k, residue_coords(r, moduli))
                    }
                })
                .collect()
        })
        .collect();
    let dens: Vec<Array2<C64>> = if mutation == Some(KGradMutation::ConjD) {
        dtot.iter().map(|d| d.mapv(|z| z.conj())).collect()
    } else {
        dtot.to_vec()
    };
    let w_d = fold_real(&dens, &ph, 1.0 / mesh.nk() as f64);
    let mut ledger = Ledger::new(crate::budget::resolve(None));
    ecp_force_walk(cell, prep, mesh, hcore_cfg, &w_d, mutation, &mut ledger)
}

/// The periodic-ECP force term at k: `(Σ_L W_L ∘ dV_L, kept triples)`, zeros
/// for an all-electron cell. `w_d[r]` is the SR walk's image weight
/// (`Re (1/N_k) Σ_k e^{ik·L} D(k)ᵀ`, residue `r` of `L`); errors when hcore's
/// `V_ECP` and the force screen disagree ([`ecp_presence`]).
#[allow(clippy::too_many_arguments)]
fn ecp_force_term(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    w_d: &[Array2<f64>],
    mutation: Option<KGradMutation>,
    ledger: &mut Ledger,
) -> Result<(Array2<f64>, usize), FerricError> {
    let natoms = cell.positions().len();
    let zeros = || Array2::<f64>::zeros((natoms, 3));
    if mutation == Some(KGradMutation::NoEcp) && hk.v_ecp.is_some() {
        return Ok((zeros(), 0));
    }
    let ecp = ecp_force_walk(cell, prep, mesh, hcore_cfg, w_d, mutation, ledger)?;
    let n = ecp_presence("k gradient", hk, ecp.as_ref().map(|e| e.n_triples))?;
    match ecp {
        Some(e) => Ok((e.grad, n)),
        None => Ok((zeros(), 0)),
    }
}

/// The range split's `J3` G = 0 term through `d S_ss`: the image-resolved
/// smooth-piece overlap derivative (over the build's pair images, the set
/// `S_ss` is defined on) contracted with the phase-folded `M_g0` weight of
/// each image's residue; the caller SUBTRACTS it (`J3 −= c0 (S − S_ss) q_c`).
/// Zeros unsplit and under the `SplitNoSmoothOverlap` / `SplitFullG0`
/// mutants.
fn smooth_g0_force(
    fit: Option<&KFitGrad>,
    w_g0: Option<&[Array2<f64>]>,
    cell: &Cell,
    moduli: [usize; 3],
    mutation: Option<KGradMutation>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = cell.positions().len();
    let dropped = matches!(
        mutation,
        Some(KGradMutation::SplitNoSmoothOverlap | KGradMutation::SplitFullG0)
    );
    let (Some(sg), Some(w)) = (fit.and_then(|f| f.smooth_g0.as_ref()), w_g0) else {
        return Ok(Array2::zeros((natoms, 3)));
    };
    if dropped {
        return Ok(Array2::zeros((natoms, 3)));
    }
    let b = cell.reciprocal();
    sg.overlap_force_by(
        |l| &w[residue_index(lattice_coords(&b, &l), moduli)],
        natoms,
    )
}

/// The range-separation kernel the dense derivative walks apply: the
/// exchange kernel factor ([`RshParams::kernel_factor`]) and whether J
/// carries it too (only under [`KGradMutation::RshKernelOnJ`]).
#[derive(Clone, Copy)]
struct RshKernel {
    params: Option<RshParams>,
    on_j: bool,
}

impl RshKernel {
    /// Multiplier of the exchange `4π/K²` at `|K|² = k2` (1 without range separation).
    fn exchange(&self, k2: f64) -> f64 {
        self.params.map_or(1.0, |r| r.kernel_factor(k2))
    }

    /// Multiplier of the Coulomb `4π/K²` (1 in production).
    fn coulomb(&self, k2: f64) -> f64 {
        if self.on_j {
            self.exchange(k2)
        } else {
            1.0
        }
    }
}

/// The derivative's [`RshKernel`] for the source's `rsh`, with the
/// `Rsh*` test defects applied.
fn rsh_kernel(rsh: Option<RshParams>, mutation: Option<KGradMutation>) -> RshKernel {
    use KGradMutation as M;
    let params = match (rsh, mutation) {
        (Some(_), Some(M::RshNoKernelFactor)) => None,
        (Some(r), Some(M::RshSwapFractions)) => Some(RshParams {
            c_sr: r.c_lr,
            c_lr: r.c_sr,
            ..r
        }),
        (Some(r), Some(M::RshOmegaScaled)) => Some(RshParams {
            omega: 1.01 * r.omega,
            ..r
        }),
        (Some(r), Some(M::RshNoErfPiece)) => Some(RshParams { c_lr: r.c_sr, ..r }),
        (r, _) => r,
    };
    RshKernel {
        params,
        on_j: rsh.is_some() && mutation == Some(M::RshKernelOnJ),
    }
}

/// AO-resolved Coulomb and exchange forces of the dense pure-AFT kernels,
/// every q class (module doc). The exchange kernel is `4π/K²` times the
/// range-separation factor of `kern` (the factor depends on `|K|` only, so
/// the pair-FT derivative is unchanged); J keeps the bare kernel. Returns
/// `(J per AO, K per AO, chunks, K count)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn dense_two_electron(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    eri: &KDenseAftEri,
    exch: &[(f64, &[Array2<C64>])],
    kern: RshKernel,
    j_scale: f64,
    delta: &[Array2<C64>],
    delta_true: &[Array2<C64>],
    ph_true: &[Vec<C64>],
    ph_d: &[Vec<C64>],
    mutation: Option<KGradMutation>,
    ledger: &mut Ledger,
) -> Result<(Vec<[f64; 3]>, Vec<[f64; 3]>, usize, usize), FerricError> {
    let n = prep.nbasis();
    let nk = mesh.nk();
    let moduli = mesh.residue_moduli();
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
        &format!("k gradient ERI K list (|K| <= {gcut:.3}, |q| <= {qmax:.3})"),
        gvector_list_bytes(cell, gcut + qmax)?.saturating_mul(2),
    )?;
    let cx_base = -1.0 / (vol * (nk * nk) as f64);
    let mut jao = vec![[0.0_f64; 3]; n];
    let mut kao = vec![[0.0_f64; 3]; n];
    let mut n_chunks = 0usize;
    let mut n_kv = 0usize;
    // Per K: Yt (R n²), P^{k'} and its adjoint, Y (4 n²), complex.
    let extra_per_g = bytes_of(((nr + 4) * n * n) as u64, 16);
    for iq in 0..nk {
        let (q, mq) = mesh.q_class(iq);
        // Coulomb only (a pure functional): the q != 0 classes carry exchange.
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
        n_kv += ks.len();
        let mq_used = if mutation == Some(KGradMutation::WrongQ) {
            [-mq[0], -mq[1], -mq[2]]
        } else {
            mq
        };
        let kof: Vec<usize> = (0..nk).map(|j| mesh.k_minus_q(j, mq_used)).collect();
        let is_q0 = mq == [0, 0, 0];
        let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
        n_chunks += pair_ft_deriv_residues_chunked(
            cell,
            prep,
            &ks,
            moduli,
            eri.pair_thresh(),
            chunk_budget,
            extra_per_g,
            |_k0, kv, p, qd| {
                for (gi, kvec) in kv.iter().enumerate() {
                    let k2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
                    let v = 4.0 * PI / k2;
                    if is_q0 {
                        let rho = residue_trace(p, gi, delta_true);
                        contract_pair_deriv(
                            &mut jao,
                            kvec,
                            gi,
                            p,
                            qd,
                            delta,
                            rho.conj() * (v * kern.coulomb(k2) / vol * j_scale),
                        );
                    }
                    if exch.is_empty() {
                        continue;
                    }
                    // Exchange: Yt_r = Σ_k' e^{ik'·t_r} Σ_s D_s(k') P^{k'H} D_s(k'−q).
                    let mut yt: Vec<Array2<C64>> = (0..nr).map(|_| Array2::zeros((n, n))).collect();
                    for j in 0..nk {
                        let mut pj = Array2::<C64>::zeros((n, n));
                        for (pr, ph) in p.iter().zip(&ph_true[j]) {
                            for m in 0..n {
                                for l in 0..n {
                                    pj[(m, l)] += *ph * pr[[m, l, gi]];
                                }
                            }
                        }
                        let pjh = herm(&pj);
                        let mut y = Array2::<C64>::zeros((n, n));
                        for (c, ds) in exch {
                            y.scaled_add(cr(*c), &ds[j].dot(&pjh).dot(&ds[kof[j]]));
                        }
                        for (r, ytr) in yt.iter_mut().enumerate() {
                            ytr.scaled_add(ph_d[j][r], &y);
                        }
                    }
                    let vx = v * kern.exchange(k2);
                    contract_pair_deriv(&mut kao, kvec, gi, p, qd, &yt, cr(cx_base * vx));
                }
                Ok(())
            },
        )?;
    }
    Ok((jao, kao, n_chunks, n_kv))
}
