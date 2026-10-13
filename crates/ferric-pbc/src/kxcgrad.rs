//! Analytic nuclear forces of the k-point closed-shell KS-DFT energy
//! ([`crate::kdft::solve_krks`]) for LDA, GGA, meta-GGA (SCAN, r2SCAN)
//! and GLOBAL HYBRID functionals (exact-exchange fraction `a`,
//! `0 ≤ a ≤ 1`), dense-AFT and RS-GDF J/K.
//!
//! # Energy (per cell; [`crate::kdft`] conventions)
//!
//! ```text
//! E = (1/N_k) Σ_k Re tr[(h(k) + ½ J(k) − (a/4) K(k)) D(k)] + E_xc^rem[D] + E_nn
//! E_xc = Σ_g w_g ε(ρ_g, σ_g),     ρ_g   = (1/N_k) Σ_k Σ_μν D(k)_μν φ^k_μ(r_g) φ^k_ν(r_g)*
//!                                 ∇ρ_g  = (1/N_k) Σ_k 2 Re Σ_μν D(k)_μν ∇φ^k_μ φ^k_ν*
//! φ^k_μ(r) = Σ_L e^{ik·L} χ_μ(r − R_A − L)        (μ on atom A)
//! ```
//!
//! `ε = ρ ε_xc`, `σ = |∇ρ|²`, `V_xc(k)_μν = ∂E_xc/∂D(k)_νμ`-equivalent as in
//! [`crate::kdft::KPeriodicXc`]. `E_xc^rem` is libxc's functional for the
//! hybrid's SEMILOCAL REMAINDER (PBE0: `0.75 E_x^PBE + E_c^PBE`); the force
//! resolves the same [`XcDef`] as the energy, so `V_xc` is the remainder's
//! potential and the Fock guard refuses any other functional.
//!
//! # Global hybrids: the exchange fraction `a`
//!
//! `K(k)` is the energy's injected exchange, Madelung shift included
//! (`K = K_AFT + v_M S D S`, `v_M` the mesh constant for `exxdiv = ewald`, 0
//! for `none`); [`crate::kscf::solve_krks_injected`] scales the whole of it by
//! `a`. At fixed `D` the nuclear derivative is therefore the HF derivative
//! with the exchange-bearing pieces multiplied by `a` and the Coulomb pieces
//! untouched (`assemble(.., exch_scale = a)`):
//!
//! * the dense-AFT K derivative walks: the exchange coefficients `c_s → a c_s`
//!   (`J` keeps its coefficient);
//! * the RS-GDF fit, shared by `J` and `K`: its gradient is bilinear in the
//!   fitted objects and the density terms. `J` enters through the total
//!   density (`ρ_P = Σ_k D_k·B_P`, `crate::rsgdf::kpoint::kderiv`'s
//!   `coulomb_q0`), `K` through `Σ_s c_s D_s(k') J3^H D_s(k)` in `H(q)` and
//!   `Z`. Only the `c_s` multiply: `a` scales the exchange part of `H` (so of
//!   the metric weight `Wm = U (Lo ∘ U^H H U) U^H`) and of `Z` (so of the
//!   fitted 3-centre weights, the SR bins and the LR line), never the
//!   Coulomb `ρ ρ^H` / `c` pieces, never the Löwdin metric `f(J2)` itself
//!   (it depends on the fit, not on `a`). The fit is the SAME for every `a`
//!   (it does not know `a`), so the lindep cut and the rebuild check are
//!   `a`-independent;
//! * the Madelung M-term `−v_M Σ_s D_s S D_s` in the overlap weight, from the
//!   energy's `−(a v_M/2)(1/N_k) Σ_s Σ_k tr[D_s S D_s S]`: `v_M → a v_M`;
//! * the energy-weighted Pulay term `W = Σ_s D_s F_s D_s`, `F = h + J −
//!   (a/2) K + V_xc^rem` (K with the shift).
//!
//! The force at fixed `D` is AFFINE in `a` (checked in
//! `tests/pbc_kxcgrad_hybrid.rs`, residual ≤ 1e-15 dense, ≤ 1.3e-13 RS-GDF).
//! That check catches ONLY `a`-nonlinear defects (`a` applied twice); a `J`
//! scaled by `a`, an unscaled piece, a `2a` spin factor or an unscaled
//! Madelung are all still affine. What pins the scaling instead: the
//! `a = 1` HF anchor (blind to every `a`-scaling defect except a spin
//! factor), the Gamma hybrid gradient at 1×1×1 per part (dense AND RS-GDF,
//! `gamma_rks_gradient{,_rsgdf}`, which scale `J` and `K` independently), the
//! Gamma supercell identity on the non-TRIM 1×1×3 mesh (dense), central FD of
//! the energy, and the production-wired mutants of the test file.
//!
//! ## exxdiv
//!
//! At a FIXED density the force does not depend on `exxdiv`: the shift
//! `−(a v_M/2) S D S` in `F` enters the Pulay weight `D F D` as
//! `−(a v_M/2) D S D S D = −a v_M D S D` (using `D S D = 2 D` for the
//! idempotent closed-shell density), which cancels the M-term
//! `−a v_M D S D` exactly, as for HF (`kpoint_rhf_gradient` is
//! exxdiv-independent). The SCFs run with `none` and `ewald` reached the same
//! density here and gave the same force (2e-15 dense, 3e-13 RS-GDF), each
//! equal to the FD of its OWN energy. That the DENSITIES coincide is not
//! guaranteed in general: the shift moves occupied levels down by `a v_M`
//! only, so with aufbau across k the two fixed points coincide when the gap
//! exceeds `a v_M`; a small-gap cell can converge to different densities,
//! and each force is still the exact derivative of its own energy.
//!
//! The force still needs the SCF's `exxdiv` to REBUILD `K` for the Fock guard
//! ([`KRksGradConfig::exxdiv`]; required for a hybrid, ignored for a pure
//! functional). Unlike the HF force's guard (which fits the `S D S` group out
//! and accepts any exxdiv), this guard demands an exact match, so a wrong
//! exxdiv is a refusal, not a silent pass.
//!
//! # What is new relative to the k-point HF force ([`crate::kgrad`])
//!
//! For `a = 0` the whole HF derivative is reused with the exchange absent
//! (the K walks, the fitted-exchange weights and the M-term vanish; the
//! Coulomb parts are unchanged), and the energy-weighted Pulay term is built
//! from the Fock matrix INCLUDING `V_xc(k)`. The only new derivative terms
//! are those of `E_xc` at fixed density matrix (identical for hybrids, with
//! the remainder functional):
//!
//! ```text
//! dE_xc/dR_{A,j} =  X1 (AO motion)  +  X3 (grid-point motion)  +  X4 (weight response)
//! ```
//!
//! # X1: AO derivative at fixed grid
//!
//! Bloch phases `e^{ik·L}` do not depend on the atoms, so an atom moves only
//! the shell centre inside each image: `∂φ^k_μ/∂R_{A,j} = −∂_jφ^k_μ` for
//! `μ ∈ A`, with `∂_jφ^k_μ = Σ_L e^{ik·L} ∂_jχ_μ(r − R_A − L)` (at FIXED
//! fractional `k` the phases are also strain/displacement invariant). With
//! `p^k_μ = Σ_ν D(k)_μν φ^k_ν*` and `m^{k,i}_μ = Σ_ν D(k)_μν (∂_iφ^k_ν)*`
//! (Hermitian `D`, so the `ν ∈ A` terms are the complex conjugates of the
//! `μ ∈ A` ones):
//!
//! ```text
//! dρ/dR_{A,j}     = −(2/N_k) Σ_k Re Σ_{μ∈A} ∂_jφ^k_μ p^k_μ
//! d∇_iρ/dR_{A,j}  = −(2/N_k) Σ_k Re Σ_{μ∈A} [∂_i∂_jφ^k_μ p^k_μ + ∂_jφ^k_μ m^{k,i}_μ]
//! a_{g,A,j} = w_g [ v_ρ dρ/dR_{A,j} + Σ_i c_i d∇_iρ/dR_{A,j} ],   c = 2 v_σ ∇ρ
//! ```
//!
//! It is the Gamma term of [`crate::grad`] with the Bloch AOs complex, `Re`
//! taken after the contraction and a `1/N_k`; GGA needs the AO Hessian
//! (`∂_i∂_jφ`), which is NOT cached (the energy cache holds `φ, ∇φ`), so it is
//! recomputed chunk by chunk (`LatticeAoHess::eval_bloch`) with
//! all k inside the chunk. The kernel gates (`ρ > DENSITY_FLOOR`) are the
//! energy's.
//!
//! # Meta-GGA: the tau term
//!
//! The energy's kinetic-energy density ([`crate::kdft`]) is
//!
//! ```text
//! τ_g = (1/N_k) Σ_k ½ Σ_i Re Σ_μν D(k)_μν ∂_iφ^k_μ (∂_iφ^k_ν)*,      E_xc = Σ_g w_g ε(ρ_g, σ_g, τ_g)
//! V^τ(k)_μν = Σ_g ½ w_g v_τ(g) Σ_i (∂_iφ^k_μ)* ∂_iφ^k_ν              (added to V_xc(k))
//! ```
//!
//! with `v_τ = ∂ε/∂τ` from libxc (gated like the other kernel pieces by
//! `ρ > DENSITY_FLOOR`). The force reuses everything above unchanged
//! (`ρ`/`σ` terms, X3, X4, the Fock guard with `V_xc` now including `V^τ`) and
//! adds one AO-derivative term. An atom move shifts `φ_μ → φ_μ(r − R_A)` for
//! `μ ∈ A`, hence `∂_iφ_μ` moves by `−∂_j∂_iφ_μ` (the AO Hessian, already
//! evaluated for GGA; no third derivatives). With
//! `m^{k,i}_μ = Σ_ν D(k)_μν (∂_iφ^k_ν)*` the `μ ∈ A` and `ν ∈ A` variations of
//! the double sum are complex conjugates (Hermitian `D`), so
//!
//! ```text
//! dτ/dR_{A,j} = (1/N_k) · ½ · 2 Re Σ_{μ∈A,i} (−∂_j∂_iφ^k_μ) m^{k,i}_μ
//!             = −(1/N_k) Σ_k Re Σ_{μ∈A,i} ∂_j∂_iφ^k_μ m^{k,i}_μ        (the ½ cancels the doubling)
//! a^τ_{g,A,j} = w_g v_τ(g) dτ/dR_{A,j}
//! ```
//!
//! i.e. the density-like prefactor `−2` of the `ρ` term becomes `−1`
//! (`TauDoubled` re-adds the `½` and doubles the term). `a^τ` joins `a` in
//! X1; the point-motion term X3 subtracts `Σ_A a` over ALL parts of the
//! integrand, tau included, which is the molecular `tau_spatial_grad`
//! (grid-point response of τ) by translation invariance, so no separate
//! `∂τ/∂r` evaluation exists here; X4 is `ε_g dw_g/dR` with `ε = ρ ε_xc`
//! evaluated with τ, unchanged. The GGA `σ` contraction is independent of the
//! tau term (both use `m^i`; the force keeps them as separate sums).
//!
//! The Gamma periodic path has no meta-GGA, so there is no `1×1×1 = Gamma`
//! anchor; the anchors are the fixed-`D` central difference of the
//! XC energy (no SCF noise), FD of the SCF energy, and the explicit
//! supercell identity on a non-TRIM mesh (`tests/pbc_kxcgrad_mgga.rs`).
//! SCAN and r2SCAN are validated end to end. TPSS resolves only as the raw
//! single-component names `MGGA_X_TPSS` / `MGGA_C_TPSS` (no combined TPSS
//! functional exists), so its tau kernels are checked by the fixed-`D`
//! finite difference on those components alone. No hybrid
//! meta-GGA resolves ([`ferric_dft::libxc::xc_def_from_name`] rejects
//! `HYB_MGGA_*`), so there is nothing for the exchange fraction to compose
//! with.
//!
//! Limits of the checks. The H2 fixtures have `α ≈ 0` (single-orbital
//! regions) and never reach SCAN's `α ≈ 1` switching region. The null-tau hook
//! multiplies every tau term by `v_τ = 0`, so it checks plumbing only. The k
//! SCF has no virtual level shift (the molecular code applies 0.5 to
//! meta-GGAs automatically), so SCAN on a real solid may not converge; the
//! force then refuses the unconverged result with a typed error.
//!
//! # X3, X4: the atom-centred grid
//!
//! A grid point is rigidly attached to its home atom `H`: it moves with it.
//! Holding `D` fixed, the integrand at a moving point changes by the point
//! motion, which is exactly minus the sum of the AO-motion terms over all atoms
//! (translating atoms and point together leaves `φ(r)` unchanged):
//!
//! ```text
//! X3:  dE/dR_{H(g),j} −= Σ_A a_{g,A,j}
//! X4:  dE/dR_{A,j}    += Σ_g ε_g · d w_g/dR_{A,j}        (SSF partition response,
//!                         [`crate::dft::build_response_grid`], k-INDEPENDENT)
//! ```
//!
//! The grid, its weights and `dw/dR` do not depend on `k` at all, so X4 uses
//! the Gamma `build_response_grid` verbatim (`ε_g = ρ_g ε_xc,g` from the k
//! density) and X3 is the Gamma recipe re-implemented here on complex AOs. On a
//! uniform grid ([`crate::dft::PeriodicGrid::uniform`]) the points are fixed in
//! space and the weights are constants: X3 and X4 are absent by construction
//! (and must be: an XC force that kept them would be wrong), only X1 remains.
//!
//! # Fock guard
//!
//! The Pulay term needs `F(k) = h + J − (a/2) K + V_xc`. The entry rebuilds it
//! from the density and compares with the SCF's stored (non-extrapolated)
//! Fock against [`crate::kgrad::FOCK_CONSISTENCY_BAR`], with NO Madelung-shift
//! allowance (the shift is rebuilt from [`KRksGradConfig::exxdiv`]). A result
//! from another functional (other `a` or other remainder), grid, J/K route or
//! `exxdiv` does not match (measured 3.9e-2 .. 6.2e-1 against the 1e-8 bar).
//!
//! # Memory
//!
//! The Bloch AO cache of the energy is `npts·nbf·N_k·16 B` per plane and is
//! never retained here. The derivative pass holds, per concurrently live
//! chunk, `13·nbf·N_k·chunk` complex values (`φ`, `∇φ`, `∇∇φ` for every k) plus
//! six `nbf·chunk` complex work planes (`D·conj(φ)` and its gradient
//! counterparts) and a partial `V_xc` (`2 N_k nbf²`); the chunk
//! shrinks with `N_k·nbf`, the number of concurrent chunks shrinks with the
//! remaining budget, and the pass is pre-flighted on a `crate::budget::Ledger`
//! (a typed error when not even one chunk fits). The `ResponseGrid` weight
//! derivatives (`npts·natoms·24 B`) are reserved too; the transient of the XC
//! pass is released before the HF assembly reserves its own arrays. A
//! meta-GGA adds NOTHING to that per-chunk charge: its Hessian moments
//! `m^{k,i}` are the GGA work planes already counted (the Hessian `∇∇φ` is
//! among the 13 planes) and the tau part of `a` is accumulated into the same
//! `a[g, A, j]` plane as the other terms. Only the `TauNoPoint` test mutant
//! keeps a separate `np·natoms·3` real tau plane (and charges it).
//!
//! # Scope
//!
//! LDA, GGA, meta-GGA and global hybrids (`0 ≤ a ≤ 1`: PBE0, B3LYP; dense AFT,
//! unsplit and range-split RS-GDF; the RS-GDF route is exercised for
//! meta-GGA by one SCF finite-difference component only). Range-separated hybrids (the erf-kernel K
//! walks and their Madelung are not differentiated), VV10 and double
//! hybrids are refused with typed errors ([`PeriodicDftError::Unsupported`]);
//! the open-shell k-point KS (`solve_kuks`, `solve_kroks`) forces are in
//! [`crate::kuxcgrad`].
//!
//! # Cost and budget notes
//!
//! * `jk_build` computes a full `K` and discards it for a pure functional (an
//!   `O(N_k²)` waste on the dense AFT path); only `J` is used. For a hybrid the
//!   Fock-guard matrices `D, J, K, F, V_xc` (`5 · N_k n² · 16 B`) are resident
//!   next to the assembly's arrays and are charged to the ledger after the XC
//!   pass; the K derivative walks themselves are charged inside `assemble`
//!   (`(R + 4) n²` complex per pair-FT chunk column on the dense path, the
//!   fit's `Z`/`H` blocks on RS-GDF), exactly as for HF.
//! * The weight derivatives take `npts·natoms·24 B` (about 2 GB for 64 atoms at
//!   (75, 302)); they are reserved, so an oversize system is a typed error, not
//!   an out-of-memory kill.
//! * The resident J/K objects (dense kernels or `KRsGdf`) belong to the caller
//!   and are not charged by the force's ledger; wiring KS into the retained-state
//!   optimizer path will need a `solve_krks_keep` and charging them through the
//!   retained-state derivative budget.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::dft::{
    build_response_grid, BlochAoHess, BlochPhase, LatticeAoHess, PeriodicDftError, PeriodicGrid,
    PeriodicGridConfig, DEFAULT_AO_THRESHOLD,
};
use crate::grad::ao_atoms;
use crate::hcore::kpoint::PeriodicHcoreK;
use crate::hcore::PeriodicHcoreConfig;
use crate::kgrad::{
    assemble, check_fock, check_inputs, jk_build, open_ledger, KGradConfig, KGradJk, KGradient,
    KSpins,
};
use crate::kpts::KPointMesh;
use crate::kscf::KScfResult;
use crate::lattice::Cell;
use crate::rsh::resolve_periodic_functional_rsh;
use ferric_core::FerricError;
use ferric_dft::density_on_grid::DensityGrid;
use ferric_dft::grid::GridPoint;
use ferric_dft::libxc::{FunctionalFamily, XcDef};
use ferric_dft::vxc::{closed_kernel, DENSITY_FLOOR};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array1, Array2, Array3, Axis};
use num_complex::Complex64 as C64;
use rayon::prelude::*;

/// Remedy text of the KS force's Fock guard.
///
/// Stricter than the HF force's guard on purpose: the HF guard fits the
/// `S D S` (Madelung-shift) group out of the comparison, so any `exxdiv`
/// passes; this one rebuilds `K` with [`KRksGradConfig::exxdiv`] and demands
/// an exact match, so the exxdiv given must be the one the SCF ran (a wrong
/// one is the commonest refusal of a hybrid force). A pure functional has no
/// `K`, so the setting is irrelevant there.
pub(crate) const KS_REMEDY: &str = "the functional, grid, AO threshold and J/K route must be the SCF's own (a different functional gives a different V_xc), and for a hybrid KRksGradConfig::exxdiv must be the exxdiv the SCF ran";

/// Upper bound on points per derivative chunk (the Gamma `XC_GRAD_CHUNK`).
pub(crate) const MAX_CHUNK: usize = 128;
/// Lower bound on points per derivative chunk.
pub(crate) const MIN_CHUNK: usize = 8;
/// Target bytes of AO planes per live chunk (shrinks the chunk with `N_k·nbf`).
pub(crate) const CHUNK_TARGET_BYTES: usize = 64 << 20;
/// Complex AO planes per k and point held by a chunk: `φ`, `∇φ` (3), `∇∇φ` (9).
pub(crate) const AO_PLANES: usize = 13;

/// The grid the energy integrated on, rebuilt for the derivative.
#[derive(Debug, Clone)]
pub enum KXcGrid {
    /// The atom-centred SSF grid of [`PeriodicGrid::build`] (point motion and
    /// weight response included). The SAME config the energy used.
    Atom(PeriodicGridConfig),
    /// [`PeriodicGrid::uniform`] with these mesh counts (points fixed in space,
    /// constant weights: no X3/X4 terms).
    Uniform([usize; 3]),
}

impl Default for KXcGrid {
    fn default() -> Self {
        Self::Atom(PeriodicGridConfig::default())
    }
}

/// Deliberate defects for the mutation tests (`tests/pbc_kxcgrad.rs`). Never
/// set in production.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KXcMutation {
    /// Drop the AO-motion term X1 (and with it X3, which is built from it).
    NoAo,
    /// Drop the grid-point motion X3.
    NoPointMotion,
    /// Drop the weight response X4.
    NoWeightDeriv,
    /// Drop the GGA part of the AO derivative (`c_i` terms).
    NoGga,
    /// Keep the GGA `∂_jφ m^i` term but drop the Hessian term `∂_i∂_jφ p`.
    NoHessian,
    /// Force-side AOs without the Bloch phase (the density keeps its own).
    NoPhase,
    /// Force-side AOs with `e^{−ik·L}` (identity on TRIM-only meshes).
    ConjPhase,
    /// `−1` instead of `−2` in the AO-derivative prefactor (a spin-factor slip).
    WrongFactor,
    /// Missing `1/N_k` in the AO-derivative term.
    NoInvNk,
    /// The Pulay term built from a Fock matrix WITHOUT `V_xc` (the guard is
    /// skipped for this defect: it would refuse it).
    PulayNoVxc,
    /// Hybrid: a WRONG semilocal part. The XC terms (X1/X3/X4) AND the `V_xc`
    /// in the Pulay Fock both use the pure parent (PBE for PBE0, BLYP for
    /// B3LYP; any other hybrid is an error) instead of the hybrid's
    /// remainder: the wrong-remainder defect ALONE. Only the Fock guard is
    /// skipped (it would refuse it); nothing else is dropped.
    WrongRemainder,
    /// The Fock guard skipped, nothing else changed (isolates the effect of
    /// the guard-skip used by `PulayNoVxc` / `WrongRemainder`; must be 0).
    GuardSkippedOnly,
    /// Meta-GGA: drop the tau AO-derivative term (and with it the tau part of
    /// the point-motion term X3, which is built from it).
    TauNoAo,
    /// Meta-GGA: the tau AO-derivative prefactor `−2` instead of `−1`
    /// (the `½` of `τ`'s definition re-added against the `μ↔ν` doubling).
    TauDoubled,
    /// Meta-GGA: keep the tau AO term but leave it out of the point-motion
    /// term X3 (the grid point moves with its home atom for every part of the
    /// integrand, tau included).
    TauNoPoint,
    /// Meta-GGA: `V^τ` missing from the Fock of the Pulay term, the Fock
    /// guard left ON (the guard must refuse it).
    TauNoVxc,
    /// Meta-GGA: as [`Self::TauNoVxc`] with the guard skipped, to measure what
    /// the defect does to the force.
    TauNoVxcUnguarded,
    /// NULL-TAU HOOK: take the meta-GGA code path (tau, `V^τ`, the tau force
    /// term) for a functional with no tau dependence (`v_τ ≡ 0`). Must
    /// reproduce the GGA force exactly.
    ForceTauPath,
}

/// Settings of [`kpoint_rks_gradient`].
#[derive(Debug, Clone)]
pub struct KRksGradConfig {
    /// The HF-part settings (budget, nucleus exponent, HF mutation).
    pub grad: KGradConfig,
    /// The grid of the energy.
    pub grid: KXcGrid,
    /// AO truncation of the energy ([`crate::dft::PeriodicXcConfig::ao_threshold`]).
    pub ao_threshold: f64,
    /// Cap on concurrently live derivative chunks (`None` = the thread count,
    /// further limited by the remaining budget). The result does not depend on it.
    pub max_live_chunks: Option<usize>,
    /// The exchange-divergence treatment of the SCF (`solve_krks`'s
    /// `krhf.exxdiv`). Needed only by a global hybrid (the exact exchange feels
    /// the Madelung shift, and the Fock guard rebuilds `K` with it); a pure
    /// functional ignores it. `None` with a hybrid is a typed error: the
    /// force cannot guess what the SCF ran.
    pub exxdiv: Option<ExxDiv>,
    /// TEST ONLY: a deliberate XC-force defect.
    #[doc(hidden)]
    pub mutation: Option<KXcMutation>,
}

impl Default for KRksGradConfig {
    fn default() -> Self {
        Self {
            grad: KGradConfig::default(),
            grid: KXcGrid::default(),
            ao_threshold: DEFAULT_AO_THRESHOLD,
            max_live_chunks: None,
            exxdiv: None,
            mutation: None,
        }
    }
}

/// Output of [`kpoint_rks_gradient`].
#[derive(Debug, Clone)]
pub struct KRksGradient {
    /// `dE/dR_A` per cell atom (`natoms × 3`, Hartree/Bohr per cell): HF part
    /// plus XC.
    pub grad: Array2<f64>,
    /// The k-point HF machinery's result with exchange fraction 0 (its `grad`
    /// is the non-XC part; per-term breakdown in `hf.parts`).
    pub hf: KGradient,
    /// XC AO-motion term X1 (before the point-motion X3 is subtracted).
    pub xc_ao: Array2<f64>,
    /// XC grid-point motion X3 (zero on a uniform grid).
    pub xc_point: Array2<f64>,
    /// XC weight response X4 (zero on a uniform grid).
    pub xc_weight: Array2<f64>,
    /// `E_xc` at the SCF density.
    pub e_xc: f64,
    /// `∫ρ` at the SCF density (the grid error against `N_e`).
    pub electrons_on_grid: f64,
    /// Grid points.
    pub n_grid_points: usize,
    /// Points per derivative chunk (shrinks with `N_k · nbf`).
    pub chunk_points: usize,
    /// Chunks evaluated concurrently (bounded by the remaining budget).
    pub live_chunks: usize,
}

/// What an XC response returns (internal: lets tests inject a null XC).
pub(crate) struct XcOut {
    pub(crate) e_xc: f64,
    pub(crate) electrons: f64,
    /// `V_xc(k)`, Hermitian.
    pub(crate) v: Vec<Array2<C64>>,
    pub(crate) ao: Array2<f64>,
    pub(crate) point: Array2<f64>,
    pub(crate) weight: Array2<f64>,
    pub(crate) npts: usize,
    pub(crate) chunk_points: usize,
    pub(crate) live_chunks: usize,
}

/// The XC energy functional's value, potential and fixed-`D` force terms.
pub(crate) trait KXcResponse {
    /// Everything the force needs from the XC energy at density `d`; `ledger`
    /// is the caller's budget (a transient clone is used inside).
    fn evaluate(&self, d: &[Array2<C64>], ledger: &Ledger) -> Result<XcOut, FerricError>;
}

/// The zero XC (`E_xc = 0`, `V_xc = 0`, no force): the exactness anchor
/// against the exchange-free HF machinery.
pub(crate) struct NullXc;

impl KXcResponse for NullXc {
    fn evaluate(&self, d: &[Array2<C64>], _ledger: &Ledger) -> Result<XcOut, FerricError> {
        let n = d[0].nrows();
        Ok(XcOut {
            e_xc: 0.0,
            electrons: 0.0,
            v: d.iter().map(|_| Array2::zeros((n, n))).collect(),
            ao: Array2::zeros((0, 3)),
            point: Array2::zeros((0, 3)),
            weight: Array2::zeros((0, 3)),
            npts: 0,
            chunk_points: 0,
            live_chunks: 0,
        })
    }
}

/// The real XC on the energy's grid.
struct GridXc<'a> {
    cell: &'a Cell,
    prep: &'a PreparedBasis,
    xc: XcDef,
    gga: bool,
    /// Meta-GGA: tau, `V^τ` and the tau force term are evaluated.
    mgga: bool,
    kpts: Vec<[f64; 3]>,
    aoat: Vec<usize>,
    grid: &'a KXcGrid,
    ao_threshold: f64,
    max_live_chunks: Option<usize>,
    mutation: Option<KXcMutation>,
}

/// `ρ`, `∇ρ`, `σ` of one chunk (the arithmetic of `KPeriodicXc::chunk_density`).
pub(crate) fn chunk_density(
    d: &[Array2<C64>],
    chi: &[Array2<C64>],
    dchi: &[Array3<C64>],
    gga: bool,
) -> DensityGrid {
    let nk = d.len();
    let nbf = d[0].nrows();
    let np = chi[0].ncols();
    let inv_nk = 1.0 / nk as f64;
    let mut rho = Array1::<f64>::zeros(np);
    let mut grad = Array2::<f64>::zeros((3, np));
    for k in 0..nk {
        let p = d[k].dot(&conj(&chi[k]));
        for g in 0..np {
            let mut r = C64::new(0.0, 0.0);
            for mu in 0..nbf {
                r += chi[k][(mu, g)] * p[(mu, g)];
            }
            rho[g] += inv_nk * r.re;
            if gga {
                for ax in 0..3 {
                    let mut gr = C64::new(0.0, 0.0);
                    for mu in 0..nbf {
                        gr += dchi[k][(ax, mu, g)] * p[(mu, g)];
                    }
                    grad[(ax, g)] += inv_nk * 2.0 * gr.re;
                }
            }
        }
    }
    let sigma = (0..np)
        .map(|g| grad[(0, g)].powi(2) + grad[(1, g)].powi(2) + grad[(2, g)].powi(2))
        .collect();
    DensityGrid { rho, grad, sigma }
}

/// `V_xc(k)` of one chunk from the Bloch AOs, the point weights, the gated
/// `v_ρ` (`vr`) and the gated GGA vector `c = 2 v_σ ∇ρ` (the arithmetic of
/// `KPeriodicXc::build`; shared by the force and the stress).
pub(crate) fn chunk_vxc(
    chi: &[Array2<C64>],
    dchi: &[Array3<C64>],
    w: &[f64],
    vr: &[f64],
    c: &Array2<f64>,
    gga: bool,
) -> Vec<Array2<C64>> {
    let nk = chi.len();
    let np = w.len();
    let a: Vec<f64> = (0..np).map(|g| w[g] * vr[g]).collect();
    let mut v: Vec<Array2<C64>> = Vec::with_capacity(nk);
    for k in 0..nk {
        let scale = |fac: &[f64]| {
            let mut m = conj(&chi[k]);
            for g in 0..np {
                m.column_mut(g).mapv_inplace(|z| z * fac[g]);
            }
            m
        };
        let mut vk = scale(&a).dot(&chi[k].t());
        if gga {
            for ax in 0..3 {
                let f: Vec<f64> = (0..np).map(|g| w[g] * c[(ax, g)]).collect();
                let d_ax = dchi[k].index_axis(Axis(0), ax);
                let m = scale(&f).dot(&d_ax.t());
                let mh = conj(&m).t().to_owned();
                vk += &m;
                vk += &mh;
            }
        }
        v.push(vk);
    }
    v
}

/// Kinetic-energy density of one chunk (the arithmetic of
/// `KPeriodicXc::chunk_tau`): `τ = (1/N_k) Σ_k ½ Σ_ax Re Σ_μν D(k)_μν ∂_axφ_μ (∂_axφ_ν)*`.
fn chunk_tau(d: &[Array2<C64>], dchi: &[Array3<C64>]) -> Array1<f64> {
    let nk = d.len();
    let nbf = d[0].nrows();
    let np = dchi[0].shape()[2];
    let w = 0.5 / nk as f64;
    let mut tau = Array1::<f64>::zeros(np);
    for k in 0..nk {
        for ax in 0..3 {
            let d_ax = dchi[k].index_axis(Axis(0), ax).to_owned();
            let p = d[k].dot(&conj(&d_ax));
            for g in 0..np {
                let mut t = C64::new(0.0, 0.0);
                for mu in 0..nbf {
                    t += d_ax[(mu, g)] * p[(mu, g)];
                }
                tau[g] += w * t.re;
            }
        }
    }
    tau
}

/// Adds `V^τ(k)_μν = Σ_g ½ w_g v_τ(g) Σ_ax (∂_axφ_μ)* ∂_axφ_ν` to `v(k)` (the
/// arithmetic of `KPeriodicXc::build`; `vt` is the gated `v_τ`). Separate from
/// [`chunk_vxc`], which the stress shares and which carries no tau.
fn add_vtau(v: &mut [Array2<C64>], dchi: &[Array3<C64>], w: &[f64], vt: &[f64]) {
    let np = w.len();
    for (k, vk) in v.iter_mut().enumerate() {
        for ax in 0..3 {
            let d_ax = dchi[k].index_axis(Axis(0), ax).to_owned();
            let mut m = conj(&d_ax);
            for g in 0..np {
                m.column_mut(g).mapv_inplace(|z| z * (0.5 * w[g] * vt[g]));
            }
            *vk += &m.dot(&d_ax.t());
        }
    }
}

pub(crate) fn conj(m: &Array2<C64>) -> Array2<C64> {
    m.mapv(|z| z.conj())
}

/// One chunk's contributions.
struct Chunk {
    e: f64,
    ne: f64,
    v: Vec<Array2<C64>>,
    ao: Array2<f64>,
    point: Array2<f64>,
    weight: Array2<f64>,
}

impl KXcResponse for GridXc<'_> {
    fn evaluate(&self, d: &[Array2<C64>], ledger: &Ledger) -> Result<XcOut, FerricError> {
        let mut led = ledger.clone();
        let natoms = self.cell.positions().len();
        let nbf = self.prep.nbasis();
        let nk = d.len();
        let mutation = self.mutation;
        // --- Grid (and, for the atom grid, the weight derivatives).
        let (points, dweight): (Vec<GridPoint>, Option<Vec<[f64; 3]>>) = match self.grid {
            KXcGrid::Atom(cfg) => {
                let rg = build_response_grid(self.cell, cfg, &mut led)?;
                (rg.points, Some(rg.dweight))
            }
            KXcGrid::Uniform(n) => (
                PeriodicGrid::uniform(self.cell, *n)?.points().to_vec(),
                None,
            ),
        };
        let ev = LatticeAoHess::new(
            self.cell,
            self.prep.basis_set(),
            &points,
            self.ao_threshold,
            &led,
        )?;
        if ev.nbf() != nbf || self.aoat.len() != nbf {
            return Err(FerricError::General(format!(
                "k-point XC force: lattice AO count {} != prepared basis {nbf}",
                ev.nbf()
            )));
        }
        // --- Chunking and the memory pre-flight.
        let tau_plane = self.mgga && mutation == Some(KXcMutation::TauNoPoint);
        let per_point = |np: usize| -> usize {
            bytes_of((nk * nbf * np) as u64, 16 * AO_PLANES)
                .saturating_add(bytes_of((nbf * np) as u64, 16 * 6))
                .saturating_add(bytes_of((np * natoms * 3) as u64, 8))
                // The tau part of `a[g, A, j]`, kept apart ONLY for the
                // `TauNoPoint` mutant; production meta-GGA charges nothing here.
                .saturating_add(if tau_plane {
                    bytes_of((np * natoms * 3) as u64, 8)
                } else {
                    0
                })
        };
        let fixed = bytes_of((nk * nbf * nbf) as u64, 16 * 2);
        let mut chunk_points = MAX_CHUNK;
        while chunk_points > MIN_CHUNK && per_point(chunk_points) > CHUNK_TARGET_BYTES {
            chunk_points /= 2;
        }
        let per_chunk = per_point(chunk_points).saturating_add(fixed);
        // One chunk must fit; more run concurrently while the budget allows.
        led.check(
            &format!(
                "k-point XC force AO/Hessian chunk ({chunk_points} points, nbf = {nbf}, N_k = {nk})"
            ),
            per_chunk,
        )?;
        let par = rayon::current_num_threads()
            .max(1)
            .min(self.max_live_chunks.unwrap_or(usize::MAX).max(1))
            .min((led.remaining() / per_chunk.max(1)).max(1));
        led.reserve(
            &format!("k-point XC force chunks ({par} x {chunk_points} points)"),
            per_chunk.saturating_mul(par),
        )?;
        let chunks = LatticeAoHess::chunks(&points, chunk_points);
        let inv_nk = 1.0 / nk as f64;
        let use_gga = self.gga && mutation != Some(KXcMutation::NoGga);
        let alt_phase = match mutation {
            Some(KXcMutation::NoPhase) => Some(BlochPhase::None),
            Some(KXcMutation::ConjPhase) => Some(BlochPhase::Conj),
            _ => None,
        };

        let run = |idx: &Vec<usize>| -> Result<Chunk, FerricError> {
            let pts: Vec<GridPoint> = idx.iter().map(|&i| points[i]).collect();
            let xyz: Vec<[f64; 3]> = pts.iter().map(|g| g.xyz).collect();
            let np = pts.len();
            let bl: BlochAoHess = ev.eval_bloch(&xyz, &self.kpts, BlochPhase::True, self.gga)?;
            let dens = chunk_density(d, &bl.chi, &bl.dchi, self.gga);
            let tau = self.mgga.then(|| chunk_tau(d, &bl.dchi));
            let kern = closed_kernel(&dens, tau.as_ref(), &self.xc);
            let live = |g: usize| dens.rho[g] > DENSITY_FLOOR;
            let mut e = 0.0;
            let mut ne = 0.0;
            for g in 0..np {
                e += pts[g].weight * dens.rho[g] * kern.exc[g];
                ne += pts[g].weight * dens.rho[g];
            }
            // Gated kernel vectors (the energy's gates).
            let vr: Vec<f64> = (0..np)
                .map(|g| if live(g) { kern.vrho[g] } else { 0.0 })
                .collect();
            let c = Array2::from_shape_fn((3, np), |(i, g)| {
                if self.gga && live(g) {
                    2.0 * kern.vsigma[g] * dens.grad[(i, g)]
                } else {
                    0.0
                }
            });
            // v_tau, gated like the energy's V^tau (ρ > DENSITY_FLOOR).
            let vt: Vec<f64> = (0..np)
                .map(|g| {
                    if self.mgga && live(g) {
                        kern.vtau[g]
                    } else {
                        0.0
                    }
                })
                .collect();
            let tau_in_vxc = !matches!(
                mutation,
                Some(KXcMutation::TauNoVxc) | Some(KXcMutation::TauNoVxcUnguarded)
            );
            // --- V_xc(k) (the arithmetic of `KPeriodicXc::build`).
            let wts: Vec<f64> = pts.iter().map(|p| p.weight).collect();
            let mut v = chunk_vxc(&bl.chi, &bl.dchi, &wts, &vr, &c, self.gga);
            if self.mgga && tau_in_vxc {
                add_vtau(&mut v, &bl.dchi, &wts, &vt);
            }
            // --- Force: a[g, A, j] (module doc, X1).
            let alt: Option<BlochAoHess> = alt_phase
                .map(|ph| ev.eval_bloch(&xyz, &self.kpts, ph, self.gga))
                .transpose()?;
            let fb: &BlochAoHess = alt.as_ref().unwrap_or(&bl);
            let pref = if mutation == Some(KXcMutation::WrongFactor) {
                -1.0
            } else {
                -2.0
            };
            let kscale = if mutation == Some(KXcMutation::NoInvNk) {
                1.0
            } else {
                inv_nk
            };
            let mut a_pt = Array3::<f64>::zeros((np, natoms, 3));
            // The tau part of `a`, kept apart only for the `TauNoPoint` mutant
            // (empty, and uncharged, otherwise).
            let mut a_tau = Array3::<f64>::zeros((if tau_plane { np } else { 0 }, natoms, 3));
            let tau_term = self.mgga && !matches!(mutation, Some(KXcMutation::TauNoAo));
            // dτ/dR_{A,j} = −(1/N_k) Re Σ_{μ∈A,i} ∂_i∂_jφ_μ m^i_μ: the ½ of
            // τ cancels the μ↔ν doubling, so the prefactor is `pref/2`.
            let tau_pref = if mutation == Some(KXcMutation::TauDoubled) {
                pref
            } else {
                0.5 * pref
            };
            if mutation != Some(KXcMutation::NoAo) {
                for k in 0..nk {
                    let p = d[k].dot(&conj(&fb.chi[k]));
                    let md: Option<[Array2<C64>; 3]> = (use_gga || tau_term).then(|| {
                        std::array::from_fn(|i| {
                            d[k].dot(&conj(&fb.dchi[k].index_axis(Axis(0), i).to_owned()))
                        })
                    });
                    for g in 0..np {
                        let wg = pref * pts[g].weight * kscale;
                        let wt = tau_pref * pts[g].weight * kscale * vt[g];
                        for mu in 0..nbf {
                            let at = self.aoat[mu];
                            let pm = p[(mu, g)];
                            for j in 0..3 {
                                let dj = fb.dchi[k][(j, mu, g)];
                                let mut t = dj * pm * vr[g];
                                if let (Some(md), true) = (md.as_ref(), use_gga) {
                                    for i in 0..3 {
                                        let mut inner = dj * md[i][(mu, g)];
                                        if mutation != Some(KXcMutation::NoHessian) {
                                            inner += fb.ddchi[k][(j, i, mu, g)] * pm;
                                        }
                                        t += inner * c[(i, g)];
                                    }
                                }
                                a_pt[(g, at, j)] += wg * t.re;
                                if let (Some(md), true) = (md.as_ref(), tau_term) {
                                    let mut tt = C64::new(0.0, 0.0);
                                    for i in 0..3 {
                                        tt += fb.ddchi[k][(j, i, mu, g)] * md[i][(mu, g)];
                                    }
                                    let val = wt * tt.re;
                                    a_pt[(g, at, j)] += val;
                                    if tau_plane {
                                        a_tau[(g, at, j)] += val;
                                    }
                                }
                            }
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
                        let val = a_pt[(g, at, j)];
                        ao[(at, j)] += val;
                        tot[j] += val;
                        if tau_plane {
                            tot[j] -= a_tau[(g, at, j)];
                        }
                        if let Some(dw) = dweight.as_ref() {
                            let eps = dens.rho[g] * kern.exc[g];
                            weight[(at, j)] += eps * dw[gi * natoms + at][j];
                        }
                    }
                }
                if dweight.is_some() {
                    for j in 0..3 {
                        point[(home, j)] -= tot[j];
                    }
                }
            }
            Ok(Chunk {
                e,
                ne,
                v,
                ao,
                point,
                weight,
            })
        };

        // Concurrency-sized batches reduced in chunk order: memory is bounded by
        // `par` chunks and the sums do not depend on the thread count.
        let mut e_xc = 0.0;
        let mut electrons = 0.0;
        let mut v: Vec<Array2<C64>> = (0..nk).map(|_| Array2::zeros((nbf, nbf))).collect();
        let mut ao = Array2::<f64>::zeros((natoms, 3));
        let mut point = Array2::<f64>::zeros((natoms, 3));
        let mut weight = Array2::<f64>::zeros((natoms, 3));
        for batch in chunks.chunks(par) {
            let outs: Vec<Result<Chunk, FerricError>> = batch.par_iter().map(&run).collect();
            for out in outs {
                let c = out?;
                e_xc += c.e;
                electrons += c.ne;
                for (acc, x) in v.iter_mut().zip(&c.v) {
                    *acc += x;
                }
                ao += &c.ao;
                point += &c.point;
                weight += &c.weight;
            }
        }
        if !e_xc.is_finite()
            || v.iter().any(|m| m.iter().any(|z| !z.re.is_finite()))
            || ao
                .iter()
                .chain(point.iter())
                .chain(weight.iter())
                .any(|x| !x.is_finite())
        {
            return Err(FerricError::General(
                "k-point XC force: non-finite E_xc/V_xc/force".into(),
            ));
        }
        // Hermitise (as `KPeriodicXc::build`).
        for m in v.iter_mut() {
            let h = conj(m).t().to_owned();
            *m = (&*m + &h).mapv(|z| z * 0.5);
        }
        match mutation {
            Some(KXcMutation::NoPointMotion) => point.fill(0.0),
            Some(KXcMutation::NoWeightDeriv) => weight.fill(0.0),
            _ => {}
        }
        Ok(XcOut {
            e_xc,
            electrons,
            v,
            ao,
            point,
            weight,
            npts: points.len(),
            chunk_points,
            live_chunks: par,
        })
    }
}

/// Hermitian part.
pub(crate) fn hermitize(m: &Array2<C64>) -> Array2<C64> {
    let h = conj(m).t().to_owned();
    (m + &h).mapv(|z| z * 0.5)
}

/// Resolve the functional and its global exact-exchange fraction `a`; refuse
/// what the force does not cover (range-separated hybrids, meta-GGA, VV10 and
/// double hybrids). Meta-GGA is refused here on purpose: the closed-shell
/// entry point uses [`resolve_global_mgga`]; a caller without the tau terms
/// (an unrestricted force) must keep this refusal.
pub(crate) fn resolve_global(functional: &str) -> Result<(XcDef, f64), FerricError> {
    let (xc, exx) = resolve_global_mgga(functional)?;
    if xc
        .funcs
        .iter()
        .any(|f| matches!(f.family(), FunctionalFamily::MetaGga))
    {
        return Err(PeriodicDftError::Unsupported {
            feature: "meta-GGA force at k",
            reason: format!(
                "{functional}: the tau derivative (AO Hessian terms) is not implemented"
            ),
        }
        .into());
    }
    Ok((xc, exx))
}

/// [`resolve_global`] that lets meta-GGA (SCAN, r2SCAN) through: the
/// closed-shell force supplies the tau terms. Every other refusal is shared.
fn resolve_global_mgga(functional: &str) -> Result<(XcDef, f64), FerricError> {
    let (xc, exx, rsh) = resolve_periodic_functional_rsh(functional)?;
    if rsh.is_some() {
        return Err(PeriodicDftError::Unsupported {
            feature: "range-separated hybrid force at k",
            reason: format!(
                "{functional}: the k-point XC force covers LDA, GGA and GLOBAL hybrid functionals; \
                 the attenuated-exchange derivative (erf-kernel K walks and Madelung) is not \
                 implemented"
            ),
        }
        .into());
    }
    if !(0.0..=1.0).contains(&exx) {
        return Err(FerricError::General(format!(
            "{functional}: exact-exchange fraction {exx} outside [0, 1]"
        )));
    }
    Ok((xc, exx))
}

/// k-point closed-shell RKS nuclear gradient (module doc) of a converged
/// [`crate::kdft::solve_krks`] result for an LDA, GGA, meta-GGA or global-hybrid
/// `functional` (a hybrid also needs [`KRksGradConfig::exxdiv`]).
///
/// * `hcore_cfg`, `hk`, `jk` — the config/output of the hcore and the J source
///   the SCF used (dense AFT or RS-GDF), as for [`crate::kgrad::kpoint_rhf_gradient`]
///   (including the SR column-rotation setting of the energy).
/// * `scf` — `KRksResult::scf`.
/// * `cfg.grid`/`cfg.ao_threshold` — the grid and AO truncation of the energy.
///
/// Refuses (typed errors) range-separated hybrids, VV10/double hybrids, a hybrid without `cfg.exxdiv`, and an SCF whose Fock the rebuilt
/// `h + J − (a/2) K + V_xc` does not reproduce.
#[allow(clippy::too_many_arguments)]
pub fn kpoint_rks_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KScfResult,
    functional: &str,
    cfg: &KRksGradConfig,
) -> Result<KRksGradient, FerricError> {
    let (mut xc, exch) = resolve_global_mgga(functional)?;
    if cfg.mutation == Some(KXcMutation::WrongRemainder) {
        // TEST ONLY: the pure parent in place of the hybrid's remainder.
        let parent = match functional.to_ascii_uppercase().as_str() {
            "PBE0" => "PBE",
            "B3LYP" => "BLYP",
            other => {
                return Err(FerricError::General(format!(
                    "WrongRemainder mutant: no pure parent registered for {other}"
                )))
            }
        };
        xc = ferric_dft::libxc::xc_def_from_name(parent)
            .map_err(|e| FerricError::General(format!("{functional}: pure parent: {e:?}")))?;
    }
    let gga = xc
        .funcs
        .iter()
        .any(|f| !matches!(f.family(), FunctionalFamily::Lda));
    if cfg.mutation == Some(KXcMutation::ForceTauPath) && !gga {
        return Err(FerricError::General(
            "ForceTauPath mutant: needs a GGA functional (the AO Hessian is evaluated only for GGA)"
                .into(),
        ));
    }
    let mgga = cfg.mutation == Some(KXcMutation::ForceTauPath)
        || xc
            .funcs
            .iter()
            .any(|f| matches!(f.family(), FunctionalFamily::MetaGga));
    let resp = GridXc {
        cell,
        prep,
        xc,
        gga,
        mgga,
        kpts: mesh.kpts().to_vec(),
        aoat: ao_atoms(prep),
        grid: &cfg.grid,
        ao_threshold: cfg.ao_threshold,
        max_live_chunks: cfg.max_live_chunks,
        mutation: cfg.mutation,
    };
    rks_gradient_with(
        "kpoint_rks_gradient",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        scf,
        &resp,
        exch,
        cfg,
    )
}

/// TEST/DIAGNOSTIC: the KS machinery with a zero XC (`E_xc = 0`, `V_xc = 0`)
/// and exact-exchange fraction `exch` on an SCF run by `solve_krks_injected`
/// with a zero XC builder and the same `exch`. With `exch = 0` it is the
/// Hartree-only anchor against [`kpoint_hartree_gradient`]; with `exch = 1`
/// it is k-point HF ([`crate::kgrad::kpoint_rhf_gradient`]). A hybrid
/// (`exch != 0`) needs `cfg.exxdiv`. `exch` outside `[0, 1]` (or NaN) is a
/// typed error.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn kpoint_null_xc_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KScfResult,
    exch: f64,
    cfg: &KRksGradConfig,
) -> Result<KRksGradient, FerricError> {
    rks_gradient_with(
        "kpoint_null_xc_gradient",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        scf,
        &NullXc,
        exch,
        cfg,
    )
}

/// The shared body: XC response, Fock guard, exchange-free HF assembly.
#[allow(clippy::too_many_arguments)]
fn rks_gradient_with(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KScfResult,
    xc: &dyn KXcResponse,
    exch: f64,
    cfg: &KRksGradConfig,
) -> Result<KRksGradient, FerricError> {
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
    if !(0.0..=1.0).contains(&exch) {
        return Err(FerricError::General(format!(
            "{who}: exact-exchange fraction {exch} outside [0, 1]"
        )));
    }
    let natoms = cell.positions().len();
    let nk = mesh.nk();
    // The Madelung shift of the SCF's K (0 for a pure functional: no K).
    let vm = if exch == 0.0 {
        0.0
    } else {
        let exxdiv = cfg.exxdiv.ok_or_else(|| {
            FerricError::General(format!(
                "{who}: a hybrid functional (exact exchange {exch}) needs \
                 KRksGradConfig::exxdiv, the treatment the SCF ran (its K carries the \
                 Madelung shift)"
            ))
        })?;
        match exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => mesh.madelung(cell)?,
        }
    };
    let mut ledger = open_ledger(cell, prep, mesh, &cfg.grad, false)?;
    let d = scf.densities.clone();
    let mut xo = xc.evaluate(&d, &ledger)?;
    if xo.ao.nrows() == 0 {
        xo.ao = Array2::zeros((natoms, 3));
        xo.point = Array2::zeros((natoms, 3));
        xo.weight = Array2::zeros((natoms, 3));
    }
    // Peak of the guard stage, beyond `open_ledger`'s own arrays: D (kept by
    // `KSpins` through `assemble`), J, K, F and V_xc, each N_k n² complex. J
    // and K are dropped once F is built, so the charge (held to the end of
    // `assemble`) is conservative for the assembly itself. The XC chunk
    // transient was already released with its ledger clone.
    ledger.reserve(
        &format!(
            "k KS force Fock-guard matrices D, J, K, F, V_xc (n = {}, N_k = {nk})",
            prep.nbasis()
        ),
        bytes_of((nk * prep.nbasis() * prep.nbasis()) as u64, 16 * 5),
    )?;
    // F = h + J - (a/2) K + V_xc; the stored Fock of the SCF must agree. K
    // carries the Madelung shift and is scaled as a whole (the energy's
    // convention); a = 0 contributes exactly nothing.
    let (j, kmat) = jk_build(&jk, vm, &d)?;
    let with_vxc = cfg.mutation != Some(KXcMutation::PulayNoVxc);
    let guard = with_vxc
        && cfg.mutation != Some(KXcMutation::TauNoVxcUnguarded)
        && cfg.mutation != Some(KXcMutation::WrongRemainder)
        && cfg.mutation != Some(KXcMutation::GuardSkippedOnly);
    let f: Vec<Array2<C64>> = (0..nk)
        .map(|q| {
            let mut base = &hk.h[q] + &j[q];
            if exch != 0.0 {
                base.scaled_add(C64::new(-0.5 * exch, 0.0), &kmat[q]);
            }
            let base = hermitize(&base);
            if with_vxc {
                hermitize(&(&base + &xo.v[q]))
            } else {
                base
            }
        })
        .collect();
    drop((j, kmat));
    let mismatch = if guard {
        check_fock(who, "fock", &f, &scf.fock, KS_REMEDY)?
    } else {
        0.0
    };
    let mut hf = assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        KSpins::Restricted { d, f },
        vm,
        exch,
        &cfg.grad,
        &mut ledger,
    )?;
    hf.fock_mismatch = mismatch;
    let grad = &(&(&hf.grad + &xo.ao) + &xo.point) + &xo.weight;
    if grad.iter().any(|v| !v.is_finite()) {
        return Err(FerricError::General(format!("{who}: non-finite force")));
    }
    Ok(KRksGradient {
        grad,
        hf,
        xc_ao: xo.ao,
        xc_point: xo.point,
        xc_weight: xo.weight,
        e_xc: xo.e_xc,
        electrons_on_grid: xo.electrons,
        n_grid_points: xo.npts,
        chunk_points: xo.chunk_points,
        live_chunks: xo.live_chunks,
    })
}

/// TEST/DIAGNOSTIC: the exchange-free HF machinery on a Hartree-only SCF
/// (`F = h + J`), the independent twin of [`kpoint_null_xc_gradient`] (it
/// shares `assemble` but none of the XC plumbing or the KS Fock guard).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn kpoint_hartree_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KScfResult,
    cfg: &KGradConfig,
) -> Result<KGradient, FerricError> {
    let who = "kpoint_hartree_gradient";
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
    let d = scf.densities.clone();
    let (j, _k) = jk_build(&jk, 0.0, &d)?;
    let f: Vec<Array2<C64>> = (0..mesh.nk())
        .map(|q| hermitize(&(&hk.h[q] + &j[q])))
        .collect();
    let mismatch = check_fock(who, "fock", &f, &scf.fock, KS_REMEDY)?;
    let mut out = assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        KSpins::Restricted { d, f },
        0.0,
        0.0,
        cfg,
        &mut ledger,
    )?;
    out.fock_mismatch = mismatch;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `resolve_global` (shared with force entry points that carry no tau
    /// terms) keeps refusing meta-GGA with its original feature and message,
    /// while the closed-shell resolver accepts the same names.
    #[test]
    fn resolve_global_still_refuses_meta_gga() {
        for name in ["SCAN", "R2SCAN"] {
            let e = resolve_global(name).expect_err("meta-GGA must be refused");
            let msg = e.to_string();
            assert!(msg.contains("meta-GGA force at k"), "{msg}");
            assert!(
                msg.contains(&format!(
                    "{name}: the tau derivative (AO Hessian terms) is not implemented"
                )),
                "{msg}"
            );
            let (xc, exx) = resolve_global_mgga(name).expect("closed-shell accepts");
            assert_eq!(exx, 0.0);
            assert!(xc
                .funcs
                .iter()
                .any(|f| matches!(f.family(), FunctionalFamily::MetaGga)));
        }
        assert!(resolve_global("PBE").is_ok());
    }
}
