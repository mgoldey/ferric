//! Analytic nuclear forces of the k-point open-shell KS-DFT energy
//! ([`crate::kuks::solve_kuks`]) for LDA, GGA and GLOBAL HYBRID functionals
//! (exact-exchange fraction `a`, `0 ≤ a ≤ 1`), dense-AFT and RS-GDF J/K, for
//! the pure meta-GGAs SCAN / r2SCAN (UKS, dense-AFT and RS-GDF J/K), and for
//! RANGE-SEPARATED hybrids (UKS and ROKS, dense-AFT and RS-GDF J/K).
//!
//! The closed-shell derivation ([`crate::kxcgrad`]: AO motion X1, grid-point
//! motion X3, weight response X4, the Madelung and exchange-fraction
//! bookkeeping, the Fock guard) carries over spin by spin. Only what is
//! spin-specific is written out here.
//!
//! # Energy (per cell)
//!
//! ```text
//! F_σ(k) = h(k) + J[D_α + D_β](k) − a K[D_σ](k) + V_xc,σ(k)
//! E = (1/N_k) Σ_k Σ_σ ½ Re tr[(h + F_σ^HF) D_σ] + E_xc[ρ_α, ρ_β] + E_nn
//! ρ_σ,g   = (1/N_k) Σ_k Re Σ_μν D_σ(k)_μν φ^k_μ φ^k_ν*
//! ∇ρ_σ,g  = (1/N_k) Σ_k 2 Re Σ_μν D_σ(k)_μν ∇φ^k_μ φ^k_ν*
//! E_xc = Σ_g w_g (ρ_α + ρ_β) ε(ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ),   σ_στ = ∇ρ_σ·∇ρ_τ
//! ```
//!
//! `D_σ` has unit occupation (the closed-shell `D` of [`crate::kxcgrad`] is
//! `D_α + D_β`). `K[D_σ]` carries the Madelung shift and is scaled as a whole
//! by `a` (no per-spin ½).
//!
//! # Spin-resolved XC terms
//!
//! At fixed `D_α, D_β` the AO motion of `φ^k_μ` (`μ ∈ A`) moves BOTH spin
//! densities, so with the libxc partials `v_ρσ = ∂(ρε)/∂ρ_σ`,
//! `v_σσσ = ∂(ρε)/∂σ_σσ`, `v_σαβ = ∂(ρε)/∂σ_αβ`:
//!
//! ```text
//! p^{σ,k}_μ = Σ_ν D_σ(k)_μν φ^k_ν*,    m^{σ,k,i}_μ = Σ_ν D_σ(k)_μν (∂_iφ^k_ν)*
//! c^α_i = 2 v_σαα ∇_iρ_α + v_σαβ ∇_iρ_β ,   c^β_i = 2 v_σββ ∇_iρ_β + v_σαβ ∇_iρ_α
//! a_{g,A,j} = −(2 w_g / N_k) Σ_σ Σ_k Re Σ_{μ∈A} [ v_ρσ ∂_jφ^k_μ p^{σ,k}_μ
//!             + Σ_i c^σ_i ( ∂_i∂_jφ^k_μ p^{σ,k}_μ + ∂_jφ^k_μ m^{σ,k,i}_μ ) ]
//! ```
//!
//! which is the closed-shell expression of [`crate::kxcgrad`] summed over the
//! two spins, each with its own `v_ρσ`, `c^σ` (the `v_σαβ` cross term feeds
//! the OTHER spin's gradient into `c^σ`). Gates are the energy's
//! ([`crate::kuks::KPeriodicXcPolarized`]): spin `σ`'s `v_ρσ` and `c^σ` are
//! zeroed where `ρ_σ ≤ DENSITY_FLOOR` (each spin on its own density).
//! X3 (`dE/dR_{H(g)} −= Σ_A a_{g,A}`) and X4 (`Σ_g (ρ_α+ρ_β) ε dw_g/dR_A`,
//! k-independent `crate::dft::build_response_grid`) are spin-summed
//! quantities and are unchanged. A uniform grid has neither.
//!
//! In the closed-shell limit `D_α = D_β = D/2`: `ρ_σ = ρ/2`, and the sum over
//! spins of the terms above is the closed-shell AO term to roundoff
//! (`tests/pbc_kuxcgrad.rs` asserts it per part).
//!
//! # Meta-GGA (UKS): the per-spin tau term
//!
//! The energy ([`crate::kuks`]) evaluates the polarized libxc kernel on the
//! per-spin `τ_σ = (1/N_k) Σ_k ½ Σ_i Re Σ_μν D_σ(k)_μν ∂_iφ_μ (∂_iφ_ν)*` and
//! gives each spin `V^τ_σ(k)_μν = Σ_g ½ w_g v_τσ Σ_i (∂_iφ_μ)* ∂_iφ_ν`. `τ_σ` is
//! LINEAR in `D_σ` alone, so its fixed-density derivative is the closed-shell
//! one of [`crate::kxcgrad`] ("Meta-GGA: the tau term") applied to `D_σ`, and
//! the spin sum carries `v_τσ = ∂(ρε)/∂τ_σ`:
//!
//! ```text
//! a^τ_{g,A,j} = −(w_g / N_k) Σ_σ v_τσ Σ_k Re Σ_{μ∈A,i} ∂_j∂_iφ^k_μ m^{σ,k,i}_μ
//! ```
//!
//! (the `½` of `τ` cancels the `μ↔ν` doubling: half the density-like `−2`).
//! `v_τσ` is gated on `ρ_σ > DENSITY_FLOOR` like the energy's `V^τ_σ`. `a^τ`
//! joins `a` in X1 and therefore in the point motion X3; X4 carries `ε` with
//! `(τ_α, τ_β)` unchanged; the Fock guard rebuilds `F_σ` with `V^τ_σ`. No new
//! derivative algebra: it is the closed-shell kernel per spin (which is why
//! the singlet UKS force reproduces the RKS one, and why that anchor is blind
//! to every α/β mix-up; the polarized anchors are in
//! `tests/pbc_kuxc_mgga.rs`). ROKS refuses meta-GGA (its SCF has no tau
//! term). The tau terms are independent of the J/K route (a pure meta-GGA
//! has no exact exchange), so the RS-GDF J route composes unchanged: anchored
//! (exact-aux RS-GDF == dense == PySCF `KUKS` energy, singlet UKS == RKS, FD
//! of the RS-GDF SCF energy) in `tests/pbc_kuks_mgga_rsgdf.rs`, issue #492.
//!
//! # Exact exchange and the Pulay term
//!
//! The k HF UHF force ([`crate::kgrad::kpoint_uhf_gradient`]) is
//! `assemble(KSpins::Unrestricted, exch_scale = 1)`; here `exch_scale = a`. The
//! exchange coefficients `c_σ = 1` (`E_x = −(a/2) Σ_σ tr D_σ K_σ`, the closed
//! shell has `½` for `D_total`) become `a c_σ`, the Madelung M-term
//! `−a v_M Σ_σ D_σ S D_σ` and the fitted-exchange weights are scaled the same
//! way ([`crate::kxcgrad`] "Global hybrids"). The energy-weighted Pulay term is
//! `W = Σ_σ D_σ F_σ D_σ` with `F_σ = h + J[D_α+D_β] − a K[D_σ] + V_xc,σ`.
//!
//! # Range-separated hybrids (UKS, ROKS)
//!
//! The open-shell RSH energies ([`crate::kuks`], [`crate::kroks`]) take the closed-shell
//! attenuated K ([`crate::rsh`]) per spin, `K_σ = K[D_σ]` scaled by 1: the
//! fractions sit inside the kernel `v(K) = 4π/K² f(K)`,
//! `f(K) = c_sr + (c_lr − c_sr) e^{−K²/4ω²}`, and the `exxdiv = ewald` shift
//! is the mixed constant `c_sr v_M + (c_lr − c_sr) v_M^{LR}(ω)` applied to
//! each spin's K. `f` depends on `|K|` only, so the force is the UHF assembly
//! above with `a = 1`, the exchange walks of BOTH spin terms `(1, D_α),
//! (1, D_β)` weighted by `f(K)` (`J[D_α + D_β]` keeps the bare kernel), and
//! the M-term `−v_M^{mix} Σ_σ D_σ S D_σ` (per spin, unit occupation: no ½).
//! The derivative kernel is the closed-shell one of [`crate::kxcgrad`]
//! ("Range-separated hybrids"); nothing new is differentiated, only the spin
//! bookkeeping of the exchange coefficients and of the M-term, which the
//! singlet anchor cannot see (`tests/pbc_kuks_rsh_force.rs` tests them on the
//! polarized cells). The dense kernels must carry the functional's
//! [`crate::rsh::RshParams`] (as `solve_kuks` builds them). ROKS
//! ([`kpoint_roks_gradient`]) builds the SAME per-spin Focks before the
//! Roothaan coupling, so its force is this assembly on the ROHF spin densities
//! with `W_RO` and the orbital-gradient gate of "ROHF / ROKS" below, nothing
//! else changed (`tests/pbc_kroks_rsh_force.rs`).
//!
//! On RS-GDF J/K (`KRsGdf::build_rsh`) the exchange walks are the closed-shell
//! fit passes of [`crate::kxcgrad`] (the plain fit's exchange pieces scaled by
//! `c_sr`, the attenuated fit's LR and metric passes by `c_lr − c_sr`), fed
//! the same open-shell term list `(1, D_α), (1, D_β)` (twice the
//! closed-shell `(½, D)` coefficient per spin), and the same M-term; no new
//! derivative, only the composition (`crate::kgrad` "Scope"). Anchors and FD
//! in `tests/pbc_kuks_rsh_rsgdf_force.rs` and
//! `tests/validation_kuks_rsh_rsgdf_force.rs` (singlet UKS == RKS #514 per
//! part, equal fractions == global hybrid, FD of the polarized UKS and ROKS
//! energies). The open-shell stress composes the same way
//! ([`crate::kuxcstress`], "Range-separated hybrids").
//!
//! # Fock guard
//!
//! `F_α`, `F_β` are rebuilt from the densities and compared with the SCF's
//! stored (non-extrapolated) `fock_alpha`, `fock_beta` against
//! [`crate::kgrad::FOCK_CONSISTENCY_BAR`], with no Madelung-shift allowance
//! (the shift is rebuilt from [`KRksGradConfig::exxdiv`]). Another functional,
//! grid, J/K route or `exxdiv` is a typed refusal. The routed Gamma ROKS force
//! ([`crate::gamma_rsh::gamma_roks_rsh_gradient`]) compares with the Gamma
//! result's spin Focks (`ScfResult::rohf_spin_focks`); only a result without
//! them falls back to comparing the Roothaan `F_eff` of the rebuilt spin
//! Focks with the stored `F_eff` (same bar, blind to equal-and-opposite spin
//! changes on its diagonal blocks).
//!
//! # ROHF / ROKS ([`kpoint_roks_gradient`], [`kpoint_rohf_gradient`])
//!
//! The k-point ROHF/ROKS energy ([`crate::kroks`]) is the UHF/UKS functional
//! `E_U[D_α, D_β]` restricted to ONE orthonormal complex orbital set per k,
//! `C(k) = [C_c | C_o | C_v]` (`D_α = C_c C_c† + C_o C_o†`, `D_β = C_c C_c†`),
//! so the force is the open-shell assembly of this module on the ROHF spin
//! densities and the spin Focks rebuilt from them (ROKS: the polarized `V_σ`;
//! ROHF: zero XC, `a = 1`), with the orthonormality Lagrangian
//!
//! ```text
//! W_RO(k) = sym(D_α F_α D_α + D_α F_β D_β)        (sym = Hermitian part)
//! W_RO − Σ_σ D_σ F_σ D_σ = ½ [C_o (f_β)_oc C_c† + h.c.],   f_σ = C† F_σ C
//! ```
//!
//! (`crate::kgrad::KSpins::RestrictedOpen`). `W_RO` exists exactly when the
//! three ROHF orbital-gradient blocks vanish at every k, `(f_β)_co`,
//! `(f_α)_ov`, `(f_α + f_β)_cv`; the entry points compute the largest of them
//! from the SCF's spin Focks and refuse a state above
//! [`crate::grad::RO_ORBITAL_GRADIENT_TOL`] (the force is first order in it).
//! At a converged state the unrestricted form `Σ_σ D_σ F_σ D_σ` differs from
//! `W_RO` only by ½ the closed-open β gradient (the identity above), so using it
//! ([`KUSpinMutation::RoWUhfForm`]) is an identity there, not a defect. The
//! Roothaan effective Fock must never be used for `W`.
//!
//! ## The gate is basis-independent (and was not)
//!
//! The blocks are measured from the DENSITIES: with the SCF's MOs `C` used only
//! as an `S`-orthonormal basis of the SCF's space, `D_σ^o = C† S D_σ S C`,
//! `P_c = D_β^o`, `P_o = D_α^o − D_β^o`, `P_v = 1 − D_α^o` (each asserted to be
//! a nested orthogonal projector, else a typed error), and the value is the
//! largest SPECTRAL NORM over k of `P_c F_β P_o`, `P_o F_α P_v`,
//! `P_c (F_α + F_β) P_v`; a rotation of `C` within its span changes nothing.
//! An earlier form read the three blocks in the SCF's MOs, which are the
//! eigenvectors of the Roothaan `F_eff[D]` of the FINAL density; in that basis
//! `F_eff` (whose off-diagonal blocks are exactly these gradients) is diagonal
//! by construction, so the measured gradient was partly rotated away. On a toy
//! model (fixed Focks, `S = 1`, one Givens rotation `κ` of a stationary state,
//! `κ = 1e-3 .. 1e-5`) it reported 1/2.2 of the closed-open, 1/5.4 of the
//! open-virtual and 1/26.9 of the closed-virtual gradient (1/9.2 of the maximum
//! when all three are rotated); the occupation check it relied on bounds `κ`
//! only to about `1e-3`. The current form equals the true gradient to 1e-8
//! relative and is proportional to `κ` (unit tests `gate_tests`). The spectral
//! norm is at least the largest element in any orthonormal basis, so the
//! tolerance remains an orbital-gradient refusal threshold (and the same
//! constant that bounds the Gamma largest-element measure applies here to a
//! larger number: the k gate can refuse a state the Gamma gate accepts).
//! Calibration (force error against the reported `g`, gate off; MEASURED ON
//! H3/STO-3G AT 1x1x1, ROKS PBE ONLY, where every block has rank 1 so the
//! spectral norm equals the vector norm; for larger cells the per-k maximum is
//! not the sum over orbitals or atoms, and for blocks of rank > 1 the spectral
//! norm is below the Frobenius norm by up to `sqrt(rank)`, so this is not a
//! general bound; the force was not measured on a rank > 1 system): `g` 6.9e-3
//! -> 4.4e-3, 2.8e-4 -> 3.2e-5, 9.3e-7 -> 6.9e-7, 7.5e-9 -> 4.9e-9, i.e. at most
//! about 0.74 `g` (and at least 0.11 `g`) there, so the tolerance 1e-5 caps the
//! force error near 7e-6 Ha/Bohr on that system. The tolerance is unchanged but a state that passed
//! the earlier gate may now be refused. The MOs must be one set per k-point (a
//! short set is a typed error). ROHF/ROKS take the dense-AFT or the
//! RS-GDF J/K, unsplit or range-split: the fit pass is generic over `KSpins`
//! and `W_RO` enters only as the overlap weight; anchors in
//! `tests/pbc_kuxcstress_ro_rsgdf.rs` (the range split at `lambda = 1` only;
//! like the RHF/UKS force, no `lambda` is refused here; the `lambda > 1`
//! refusal of the stress is strain-specific).
//!
//! # Stationarity of UKS
//!
//! The UKS force is first order in the departure from self-consistency. The
//! entry points refuse a state with `max_{σ,k} |F_σ D_σ S − S D_σ F_σ|` above
//! [`UKS_COMMUTATOR_TOL`].
//!
//! # Memory
//!
//! As [`crate::kxcgrad`]: the Bloch AO cache of the energy is never retained;
//! per live chunk the pass holds `13·nbf·N_k·chunk` complex AO values (shared
//! by both spins) plus eight `nbf·chunk` work planes (the two spins' density
//! products and the gradient products of the spin being processed), a partial
//! `V_xc,α` and `V_xc,β` (`4 N_k nbf²`), the chunk's `a[g,A,j]`; the whole is
//! pre-flighted on a `crate::budget::Ledger`. The Fock-guard stage holds
//! `D_α, D_β, J_α, J_β, K_α, K_β, F_α, F_β, V_α, V_β` (`10 N_k n²` complex).
//!
//! # Scope
//!
//! LDA, GGA and global hybrids; range-separated hybrids for UKS and ROKS on
//! the dense-AFT and RS-GDF J/K; for UKS also SCAN / r2SCAN on either J/K
//! (any other meta-GGA and meta-GGA ROKS are refused). VV10 and double hybrids are refused with the
//! typed errors of [`crate::kxcgrad`].

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::dft::{
    build_response_grid, BlochAoHess, BlochPhase, LatticeAoHess, PeriodicDftError, PeriodicGrid,
};
use crate::grad::ao_atoms;
use crate::grad::RO_ORBITAL_GRADIENT_TOL;
use crate::hcore::kpoint::PeriodicHcoreK;
use crate::hcore::PeriodicHcoreConfig;
use crate::kgrad::{
    assemble, check_fock, check_inputs_rsh, jk_build, open_ledger, KGradJk, KSpins,
};
use crate::kpts::KPointMesh;
use crate::kroks::KRoScfResult;
use crate::kuscf::KUScfResult;
use crate::kxcgrad::{
    add_vtau, chunk_density, chunk_tau, chunk_vxc, conj, hermitize, hybrid_madelung,
    resolve_global, resolve_global_mgga, resolve_global_or_rsh, KRksGradConfig, KRksGradient,
    KXcGrid, KXcMutation, AO_PLANES, CHUNK_TARGET_BYTES, KS_REMEDY, MAX_CHUNK, MIN_CHUNK,
};
use crate::lattice::Cell;
use crate::rsh::RshParams;
use ferric_core::FerricError;
use ferric_dft::density_on_grid::UksDensityGrid;
use ferric_dft::grid::GridPoint;
use ferric_dft::libxc::{xc_def_from_name_nspin, FunctionalFamily, XcDef};
use ferric_dft::vxc::{polarized_kernel, PolarizedKernel, DENSITY_FLOOR};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array1, Array2, Array3, Axis};
use num_complex::Complex64 as C64;
use rayon::prelude::*;

/// Result of [`kpoint_uks_gradient`] (the closed-shell result type: HF part,
/// XC parts, grid diagnostics).
pub type KUksGradient = KRksGradient;

/// Spin-specific deliberate defects for the mutation tests
/// (`tests/pbc_kuxcgrad.rs`). Never set in production.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KUSpinMutation {
    /// The XC force term uses `v_ρβ`, `c^β` for the α density and vice versa
    /// (the potentials `V_σ` keep their own spin).
    XcSpinSwap,
    /// The β spin is dropped from the XC force term.
    XcDropBeta,
    /// The `v_σαβ ∇ρ_other` cross term is dropped from `c^σ`.
    XcCrossDropped,
    /// The cross term uses the spin's OWN gradient (`v_σαβ ∇ρ_σ`).
    XcCrossOwn,
    /// The XC force term uses the TOTAL-density closed-shell form
    /// (`v_ρ = (v_ρα + v_ρβ)/2` on `D_α + D_β`) instead of the spin-resolved one.
    XcSpinAveraged,
    /// The Pulay term built from `W = D_α F_β D_α + D_β F_α D_β` (spin Focks
    /// swapped). The Fock guard is skipped for this defect.
    PulayFockSwap,
    /// `F_σ` rebuilt with the OTHER spin's exchange (`K[D_σ̄]`). The guard is
    /// skipped.
    PulayKSwap,
    /// `F_σ` rebuilt with the Coulomb matrix of `D_α` only. The guard is
    /// skipped.
    PulayJOneSpin,
    /// `F_σ` rebuilt without `V_xc,β` (α keeps its own). The guard is skipped.
    PulayNoVxcBeta,
    /// ROHF/ROKS: the energy-weighted matrix in the UNRESTRICTED form
    /// `Σ_σ D_σ F_σ D_σ`. An IDENTITY at a converged ROHF state (it differs
    /// from `W_RO` by ½ the closed-open β orbital gradient): the tests assert
    /// that it moves the converged force by < 1e-8; off stationarity the gate
    /// refuses it (measured with [`Self::RoWUhfFormGateOff`]).
    RoWUhfForm,
    /// ROHF/ROKS: `W = D_α F_α D_α` (the `D_α F_β D_β` cross term dropped).
    RoWDropCross,
    /// ROHF/ROKS: the orbital-gradient gate skipped. An identity at a
    /// converged state; on a loosely converged state it lets the entry point
    /// return a force that is wrong by an amount the tests assert exceeds the
    /// FD bar (`tests/pbc_kuxcgrad_ro.rs`).
    RoGateOff,
    /// ROHF/ROKS: [`Self::RoWUhfForm`] with the gate skipped (one mutation
    /// slot): what the unrestricted-form `W` costs off stationarity.
    RoWUhfFormGateOff,
    /// UKS: the stationarity gate (`max |F D S − S D F|`) skipped.
    StationarityGateOff,
    /// Meta-GGA: the tau force term of spin `σ` uses `v_τσ̄` (the other
    /// spin's); `V^τ_σ` and every non-tau term keep their own spin.
    TauSpinSwap,
    /// Meta-GGA: the β tau force term dropped (`V^τ_β` kept).
    TauDropBeta,
    /// Meta-GGA: the tau force term uses `v_τσ` of a libxc kernel evaluated
    /// with the TOTAL `τ_α + τ_β` in both channels (`V^τ_σ` keeps the true
    /// kernel).
    TauTotalBothChannels,
    /// Meta-GGA: `V^τ_β` missing from the β Fock of the Pulay term. The Fock
    /// guard is skipped.
    TauNoVxcBeta,
}

/// Refusal threshold on `max_{σ,k} |F_σ D_σ S − S D_σ F_σ|` (Hartree) of the
/// open-shell UKS entry points. Measured (`tests/pbc_kuxcgrad.rs`,
/// `uks_stationarity_probe`, H3 doublet PBE 1x1x2) against the force of a
/// tightly converged SCF: commutator 1.6e-3 -> force error 7.2e-4, 2.7e-5 ->
/// 1.4e-5, 2.6e-6 -> 5.0e-7 (error about half the commutator, first order).
/// 1e-5 admits the second-order-small residuals of a converged SCF and caps
/// the force error near 5e-6 Ha/Bohr, below geometry-optimizer thresholds
/// (~4.5e-4); it is the value used for the ROHF orbital gradient.
pub const UKS_COMMUTATOR_TOL: f64 = 1e-5;

/// Settings of [`kpoint_uks_gradient`].
#[derive(Debug, Clone, Default)]
pub struct KUksGradConfig {
    /// The closed-shell settings (budget, grid, AO threshold, exxdiv, chunk
    /// cap, XC mutation).
    pub base: KRksGradConfig,
    /// TEST ONLY: a deliberate spin-specific defect.
    #[doc(hidden)]
    pub spin_mutation: Option<KUSpinMutation>,
}

/// The converged spin state an open-shell force differentiates (from a
/// [`KUScfResult`] or a [`KRoScfResult`]).
pub(crate) struct SpinInput<'a> {
    da: &'a [Array2<C64>],
    db: &'a [Array2<C64>],
    fock: StoredFock<'a>,
    converged: bool,
    kpts: &'a [[f64; 3]],
    /// ROHF/ROKS: the one orthonormal MO set per k.
    mos: Option<&'a [Array2<C64>]>,
}

/// The SCF Fock(s) the force's Fock guard compares the rebuilt spin Focks
/// with (module doc "Fock guard").
#[derive(Clone, Copy)]
pub(crate) enum StoredFock<'a> {
    /// The SCF's spin Focks `F_α(k)`, `F_β(k)` (k-point UKS / ROKS / ROHF, and
    /// a Gamma ROKS result's `rohf_spin_focks`).
    Spin(&'a [Array2<C64>], &'a [Array2<C64>]),
    /// Only the Roothaan effective Fock `F_eff(k)` (FALLBACK for a Gamma ROKS
    /// result without `rohf_spin_focks`): compared with the Roothaan `F_eff`
    /// of the rebuilt spin Focks. Weaker than [`Self::Spin`]: `F_eff` carries
    /// only `½(δF_α + δF_β)` on its diagonal blocks, so an equal-and-opposite
    /// spin change there is invisible to it.
    Roothaan(&'a [Array2<C64>]),
}

impl<'a> SpinInput<'a> {
    /// A restricted-open-shell state (the Gamma ROKS result as the 1x1x1
    /// one) with the stored Fock(s) `fock` its guard compares against.
    pub(crate) fn restricted_open(
        (da, db): (&'a [Array2<C64>], &'a [Array2<C64>]),
        fock: StoredFock<'a>,
        mos: &'a [Array2<C64>],
        converged: bool,
        kpts: &'a [[f64; 3]],
    ) -> Self {
        Self {
            da,
            db,
            fock,
            converged,
            kpts,
            mos: Some(mos),
        }
    }
}

/// The Fock guard: the largest deviation of the rebuilt spin Focks (or of
/// their Roothaan `F_eff`) from the SCF's stored one(s), refused above
/// [`crate::kgrad::FOCK_CONSISTENCY_BAR`].
fn stored_fock_mismatch(
    who: &str,
    stored: StoredFock<'_>,
    (fa, fb): (&[Array2<C64>], &[Array2<C64>]),
    (da, db): (&[Array2<C64>], &[Array2<C64>]),
    s: &[Array2<C64>],
) -> Result<f64, FerricError> {
    match stored {
        StoredFock::Spin(sa, sb) => {
            let ma = check_fock(who, "fock_alpha", fa, sa, KS_REMEDY)?;
            Ok(ma.max(check_fock(who, "fock_beta", fb, sb, KS_REMEDY)?))
        }
        StoredFock::Roothaan(se) => {
            let feff: Vec<Array2<C64>> = (0..fa.len())
                .map(|k| crate::kroks::roothaan_fock_k(&fa[k], &fb[k], &da[k], &db[k], &s[k]))
                .collect();
            check_fock(who, "Roothaan fock_eff", &feff, se, KS_REMEDY)
        }
    }
}

/// Number of `n²` complex arrays the orbital-gradient gate holds at its peak
/// (one k at a time, `m <= n` MOs), counted from `ro_orbital_gradient_with`:
/// 15 named arrays live at once (`C†`, `S C`, `(S C)†`, `D_α^o`, `D_β^o`,
/// `1`, `P_c`, `P_o`, `P_v`, `F_α^o`, `F_β^o`, `F^o_α + F^o_β`, `P_c F_β P_o`,
/// `P_o F_α P_v`, `P_c F P_v`), one product temporary, and the spectral-norm
/// stage on top (`gᴴ`, `gᴴ g`, its Hermitian part, the column-major copy, the
/// eigenvectors, about two for the LAPACK workspace): 22, charged as 24.
#[doc(hidden)]
pub const RO_GATE_N2_ARRAYS: usize = 24;

/// Bytes the gate's per-k transients are charged on the ledger.
pub(crate) fn ro_gate_scratch_bytes(n: usize) -> usize {
    bytes_of((n * n) as u64, 16 * RO_GATE_N2_ARRAYS)
}

/// Deliberate defects of the ROHF orbital-gradient measurement (unit tests
/// only; production passes `None`).
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GateMutation {
    /// The closed-virtual block is not measured.
    DropClosedVirtual,
    /// The open-virtual block is not measured.
    DropOpenVirtual,
    /// The closed-open block is not measured.
    DropClosedOpen,
    /// `D` is transformed without `S` (`C† D C`): the wrong metric.
    WrongS,
    /// The Focks are transformed with `S C` (`(S C)† F (S C)`) instead of `C`:
    /// `S` wrongly placed on the covariant side.
    WrongFockS,
}

/// Largest singular value of `g` (NaN if any element is not finite).
fn spectral_norm(g: &Array2<C64>) -> f64 {
    if g.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return f64::NAN;
    }
    if g.iter().all(|z| z.norm() == 0.0) {
        return 0.0;
    }
    let m = hermitize(&g.t().mapv(|z| z.conj()).dot(g));
    match crate::kscf::eigh_herm(&m) {
        Ok((w, _)) => w.iter().cloned().fold(0.0_f64, f64::max).max(0.0).sqrt(),
        Err(_) => f64::NAN,
    }
}

/// `NaN`-propagating maximum (`f64::max` swallows NaN).
fn max_nan(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// The ROHF orbital gradient at k (max over k; module doc "ROHF / ROKS").
///
/// BASIS-INDEPENDENT form: `mos` serve only as an `S`-orthonormal basis of
/// the SCF's variational space (any rotation within it gives the same value);
/// the closed / open / virtual subspaces come from the DENSITIES, not from the
/// MOs' occupations. With `D_σ^o = C† S D_σ S C`, `F_σ^o = C† F_σ C`,
/// `P_c = D_β^o`, `P_o = D_α^o − D_β^o`, `P_v = 1 − D_α^o`, the value is the
/// largest spectral norm over k and over the three blocks `P_c F_β P_o`,
/// `P_o F_α P_v`, `P_c (F_α + F_β) P_v`. (An earlier form read the blocks in
/// the eigenvectors of the Roothaan `F_eff[D]` of the final density, in which
/// those blocks are rotated away: it under-reported the gradient by up to two
/// orders of magnitude.)
pub(crate) fn ro_orbital_gradient_k(
    who: &str,
    mos: &[Array2<C64>],
    s: &[Array2<C64>],
    da: &[Array2<C64>],
    db: &[Array2<C64>],
    fa: &[Array2<C64>],
    fb: &[Array2<C64>],
) -> Result<f64, FerricError> {
    ro_orbital_gradient_with(who, mos, s, da, db, fa, fb, None)
}

/// Tolerance on the idempotency / nesting of the density projectors.
const PROJECTOR_TOL: f64 = 1e-6;

#[allow(clippy::too_many_arguments)]
pub(crate) fn ro_orbital_gradient_with(
    who: &str,
    mos: &[Array2<C64>],
    s: &[Array2<C64>],
    da: &[Array2<C64>],
    db: &[Array2<C64>],
    fa: &[Array2<C64>],
    fb: &[Array2<C64>],
    mutation: Option<GateMutation>,
) -> Result<f64, FerricError> {
    let nk = s.len();
    if mos.len() != nk || da.len() != nk || db.len() != nk || fa.len() != nk || fb.len() != nk {
        return Err(FerricError::General(format!(
            "{who}: {} MO sets for {nk} k-points (the SCF's MOs must be the mesh's, one set per k)",
            mos.len()
        )));
    }
    let herm = |m: &Array2<C64>| m.t().mapv(|z| z.conj());
    let mut worst = 0.0_f64;
    for k in 0..nk {
        let c = &mos[k];
        let n = s[k].nrows();
        if c.nrows() != n || c.ncols() > n {
            return Err(FerricError::General(format!(
                "{who}: MO set at k-point {k} has shape {:?}, expected ({n}, m <= {n})",
                c.dim()
            )));
        }
        let ch = herm(c);
        let sc = s[k].dot(c);
        let sch = herm(&sc);
        let to_mo = |d: &Array2<C64>| -> Array2<C64> {
            if mutation == Some(GateMutation::WrongS) {
                ch.dot(d).dot(c)
            } else {
                sch.dot(d).dot(&sc)
            }
        };
        let (dao, dbo) = (to_mo(&da[k]), to_mo(&db[k]));
        let m = c.ncols();
        let eye = Array2::<C64>::eye(m);
        let pc = dbo.clone();
        let po = &dao - &dbo;
        let pv = &eye - &dao;
        // The densities must be nested orthogonal projectors in the MO space.
        let dev = |a: &Array2<C64>, b: &Array2<C64>| {
            a.iter()
                .zip(b)
                .fold(0.0_f64, |w, (x, y)| max_nan(w, (x - y).norm()))
        };
        let bad = max_nan(
            max_nan(dev(&dao.dot(&dao), &dao), dev(&dbo.dot(&dbo), &dbo)),
            dev(&dbo.dot(&dao), &dbo),
        );
        if bad.is_nan() || bad > PROJECTOR_TOL {
            return Err(FerricError::General(format!(
                "{who}: at k-point {k} (D_alpha, D_beta) are not nested projectors in the SCF's MO \
                 space (deviation {bad:.3e}): not a ROHF state of these MOs"
            )));
        }
        let (fam, fbm) = if mutation == Some(GateMutation::WrongFockS) {
            (sch.dot(&fa[k]).dot(&sc), sch.dot(&fb[k]).dot(&sc))
        } else {
            (ch.dot(&fa[k]).dot(c), ch.dot(&fb[k]).dot(c))
        };
        let fcv = &fam + &fbm;
        let g_co = pc.dot(&fbm).dot(&po);
        let g_ov = po.dot(&fam).dot(&pv);
        let g_cv = pc.dot(&fcv).dot(&pv);
        let mut w = worst;
        if mutation != Some(GateMutation::DropClosedOpen) {
            w = max_nan(w, spectral_norm(&g_co));
        }
        if mutation != Some(GateMutation::DropOpenVirtual) {
            w = max_nan(w, spectral_norm(&g_ov));
        }
        if mutation != Some(GateMutation::DropClosedVirtual) {
            w = max_nan(w, spectral_norm(&g_cv));
        }
        worst = w;
    }
    Ok(worst)
}

/// What a polarized XC response returns.
pub(crate) struct XcOutPol {
    pub(crate) e_xc: f64,
    pub(crate) electrons: f64,
    pub(crate) va: Vec<Array2<C64>>,
    pub(crate) vb: Vec<Array2<C64>>,
    pub(crate) ao: Array2<f64>,
    pub(crate) point: Array2<f64>,
    pub(crate) weight: Array2<f64>,
    pub(crate) npts: usize,
    pub(crate) chunk_points: usize,
    pub(crate) live_chunks: usize,
}

/// The polarized XC energy functional's potentials and fixed-`D` force terms.
pub(crate) trait KUXcResponse {
    fn evaluate(
        &self,
        da: &[Array2<C64>],
        db: &[Array2<C64>],
        ledger: &Ledger,
    ) -> Result<XcOutPol, FerricError>;
}

/// Zero XC (exactness anchor against the exchange-bearing HF machinery).
pub(crate) struct NullXcPol;

impl KUXcResponse for NullXcPol {
    fn evaluate(
        &self,
        da: &[Array2<C64>],
        _db: &[Array2<C64>],
        _ledger: &Ledger,
    ) -> Result<XcOutPol, FerricError> {
        let n = da[0].nrows();
        let z = || da.iter().map(|_| Array2::zeros((n, n))).collect();
        Ok(XcOutPol {
            e_xc: 0.0,
            electrons: 0.0,
            va: z(),
            vb: z(),
            ao: Array2::zeros((0, 3)),
            point: Array2::zeros((0, 3)),
            weight: Array2::zeros((0, 3)),
            npts: 0,
            chunk_points: 0,
            live_chunks: 0,
        })
    }
}

struct GridXcPol<'a> {
    cell: &'a Cell,
    prep: &'a PreparedBasis,
    xc: XcDef,
    gga: bool,
    /// Meta-GGA: per-spin tau, `V^τ_σ` and the tau force term are evaluated.
    mgga: bool,
    kpts: Vec<[f64; 3]>,
    aoat: Vec<usize>,
    grid: &'a KXcGrid,
    ao_threshold: f64,
    max_live_chunks: Option<usize>,
    mutation: Option<KXcMutation>,
    spin_mutation: Option<KUSpinMutation>,
}

struct ChunkPol {
    e: f64,
    ne: f64,
    va: Vec<Array2<C64>>,
    vb: Vec<Array2<C64>>,
    ao: Array2<f64>,
    point: Array2<f64>,
    weight: Array2<f64>,
}

/// One spin's gated kernel vectors on a chunk: `v_ρσ`, `c^σ_i` and `v_τσ`
/// (zeros for LDA/GGA). Shared with the open-shell stress.
pub(crate) struct SpinKernel {
    pub(crate) vr: Vec<f64>,
    pub(crate) c: Array2<f64>,
    pub(crate) vt: Vec<f64>,
}

/// Which gradient feeds the `v_σαβ` cross term of `c^σ`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cross {
    /// The other spin's gradient (production).
    Other,
    /// No cross term (mutant).
    Dropped,
    /// The spin's own gradient (mutant).
    Own,
}

/// Spin `s`'s gated `v_ρσ`, `c^σ` and `v_τσ` (the energy's gates: each spin
/// on its own density, [`crate::kuks::KPeriodicXcPolarized`]). `vtau` is the
/// `(v_τα, v_τβ)` pair the tau term reads (the true kernel's, or a mutant's);
/// `None` (LDA/GGA) gives zeros.
pub(crate) fn spin_kernel(
    s: usize,
    cross: Cross,
    dens: &UksDensityGrid,
    kern: &PolarizedKernel,
    gga: bool,
    vtau: Option<(&Array1<f64>, &Array1<f64>)>,
) -> SpinKernel {
    let np = dens.rho_a.len();
    let (rho, vrho, vss, gself, gother) = if s == 0 {
        (
            &dens.rho_a,
            &kern.vrho_a,
            &kern.vsigma_aa,
            &dens.grad_a,
            &dens.grad_b,
        )
    } else {
        (
            &dens.rho_b,
            &kern.vrho_b,
            &kern.vsigma_bb,
            &dens.grad_b,
            &dens.grad_a,
        )
    };
    let live = |g: usize| rho[g] > DENSITY_FLOOR;
    let vr = (0..np)
        .map(|g| if live(g) { vrho[g] } else { 0.0 })
        .collect();
    let c = Array2::from_shape_fn((3, np), |(i, g)| {
        if gga && live(g) {
            let x = match cross {
                Cross::Other => kern.vsigma_ab[g] * gother[(i, g)],
                Cross::Dropped => 0.0,
                Cross::Own => kern.vsigma_ab[g] * gself[(i, g)],
            };
            2.0 * vss[g] * gself[(i, g)] + x
        } else {
            0.0
        }
    });
    let vt = match vtau {
        Some((ta, tb)) => {
            let v = if s == 0 { ta } else { tb };
            (0..np).map(|g| if live(g) { v[g] } else { 0.0 }).collect()
        }
        None => vec![0.0; np],
    };
    SpinKernel { vr, c, vt }
}

/// The per-spin kinetic-energy densities `(τ_α, τ_β)` of a chunk (libxc's ½
/// convention, [`chunk_tau`] of each spin density), and the `(v_τα, v_τβ)`
/// the tau FORCE/STRAIN terms read: the true kernel's, or under
/// `total_tau` those of a kernel evaluated with `τ_α + τ_β` in both channels
/// (a test defect). Shared with the open-shell stress.
pub(crate) fn tau_force_vtau(
    dens: &UksDensityGrid,
    tau: &(Array1<f64>, Array1<f64>),
    kern: &PolarizedKernel,
    xc: &XcDef,
    total_tau: bool,
) -> (Array1<f64>, Array1<f64>) {
    if total_tau {
        let t = &tau.0 + &tau.1;
        let k2 = polarized_kernel(dens, Some((&t, &t)), xc);
        (k2.vtau_a, k2.vtau_b)
    } else {
        (kern.vtau_a.clone(), kern.vtau_b.clone())
    }
}

/// The (density, kernel vectors) pairs of the force passes, with the spin
/// defects of `smut` applied (production: `[(D_α, k_α), (D_β, k_β)]`).
/// `vtau` is what the tau term reads ([`tau_force_vtau`]).
#[allow(clippy::too_many_arguments)]
fn force_passes<'b>(
    da: &'b [Array2<C64>],
    db: &'b [Array2<C64>],
    dsum: Option<&'b [Array2<C64>]>,
    dens: &UksDensityGrid,
    kern: &PolarizedKernel,
    gga: bool,
    vtau: Option<(&Array1<f64>, &Array1<f64>)>,
    smut: Option<KUSpinMutation>,
) -> Vec<(&'b [Array2<C64>], SpinKernel)> {
    let cross = match smut {
        Some(KUSpinMutation::XcCrossDropped) => Cross::Dropped,
        Some(KUSpinMutation::XcCrossOwn) => Cross::Own,
        _ => Cross::Other,
    };
    let mut fka = spin_kernel(0, cross, dens, kern, gga, vtau);
    let mut fkb = spin_kernel(1, cross, dens, kern, gga, vtau);
    match smut {
        Some(KUSpinMutation::TauSpinSwap) => std::mem::swap(&mut fka.vt, &mut fkb.vt),
        Some(KUSpinMutation::TauDropBeta) => fkb.vt.iter_mut().for_each(|x| *x = 0.0),
        _ => {}
    }
    match smut {
        Some(KUSpinMutation::XcSpinSwap) => vec![(da, fkb), (db, fka)],
        Some(KUSpinMutation::XcDropBeta) => vec![(da, fka)],
        Some(KUSpinMutation::XcSpinAveraged) => {
            let np = fka.vr.len();
            let avg = |x: &[f64], y: &[f64]| -> Vec<f64> {
                x.iter().zip(y).map(|(a, b)| 0.5 * (a + b)).collect()
            };
            let mean = SpinKernel {
                vr: avg(&fka.vr, &fkb.vr),
                c: Array2::from_shape_fn((3, np), |(i, g)| 0.5 * (fka.c[(i, g)] + fkb.c[(i, g)])),
                vt: avg(&fka.vt, &fkb.vt),
            };
            vec![(dsum.expect("dsum for XcSpinAveraged"), mean)]
        }
        _ => vec![(da, fka), (db, fkb)],
    }
}

/// Switches of the force contraction (production values unless a mutant).
struct ForceOpts {
    use_gga: bool,
    tau_term: bool,
    hessian: bool,
    pref: f64,
    tau_pref: f64,
    kscale: f64,
}

/// One spin's AO-motion term `a[g, A, j]` (module doc, spin-resolved X1 plus
/// the tau term), added to `a_pt`.
fn ao_force_pass(
    dm: &[Array2<C64>],
    sk: &SpinKernel,
    fb: &BlochAoHess,
    w: &[f64],
    aoat: &[usize],
    o: &ForceOpts,
    a_pt: &mut Array3<f64>,
) {
    let nbf = dm[0].nrows();
    let np = w.len();
    for (k, dk) in dm.iter().enumerate() {
        let p = dk.dot(&conj(&fb.chi[k]));
        let md: Option<[Array2<C64>; 3]> = (o.use_gga || o.tau_term).then(|| {
            std::array::from_fn(|i| dk.dot(&conj(&fb.dchi[k].index_axis(Axis(0), i).to_owned())))
        });
        for g in 0..np {
            let wg = o.pref * w[g] * o.kscale;
            let wt = o.tau_pref * w[g] * o.kscale * sk.vt[g];
            for mu in 0..nbf {
                let at = aoat[mu];
                let pm = p[(mu, g)];
                for j in 0..3 {
                    let dj = fb.dchi[k][(j, mu, g)];
                    let mut t = dj * pm * sk.vr[g];
                    if let (Some(md), true) = (md.as_ref(), o.use_gga) {
                        for i in 0..3 {
                            let mut inner = dj * md[i][(mu, g)];
                            if o.hessian {
                                inner += fb.ddchi[k][(j, i, mu, g)] * pm;
                            }
                            t += inner * sk.c[(i, g)];
                        }
                    }
                    a_pt[(g, at, j)] += wg * t.re;
                    // dτ_σ/dR_{A,j} = −(1/N_k) Re Σ_{μ∈A,i} ∂_i∂_jφ_μ md^{σ,i}_μ:
                    // the ½ of τ cancels the μ↔ν doubling (prefactor `pref/2`).
                    if let (Some(md), true) = (md.as_ref(), o.tau_term) {
                        let mut tt = C64::new(0.0, 0.0);
                        for i in 0..3 {
                            tt += fb.ddchi[k][(j, i, mu, g)] * md[i][(mu, g)];
                        }
                        a_pt[(g, at, j)] += wt * tt.re;
                    }
                }
            }
        }
    }
}

impl KUXcResponse for GridXcPol<'_> {
    fn evaluate(
        &self,
        da: &[Array2<C64>],
        db: &[Array2<C64>],
        ledger: &Ledger,
    ) -> Result<XcOutPol, FerricError> {
        let mut led = ledger.clone();
        let natoms = self.cell.positions().len();
        let nbf = self.prep.nbasis();
        let nk = da.len();
        let mutation = self.mutation;
        let smut = self.spin_mutation;
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
                "k-point UKS XC force: lattice AO count {} != prepared basis {nbf}",
                ev.nbf()
            )));
        }
        // Chunking and the memory pre-flight (two spins' work planes, two
        // partial potentials).
        let per_point = |np: usize| -> usize {
            bytes_of((nk * nbf * np) as u64, 16 * AO_PLANES)
                .saturating_add(bytes_of((nbf * np) as u64, 16 * 8))
                .saturating_add(bytes_of((np * natoms * 3) as u64, 8))
        };
        let fixed = bytes_of((nk * nbf * nbf) as u64, 16 * 4);
        let mut chunk_points = MAX_CHUNK;
        while chunk_points > MIN_CHUNK && per_point(chunk_points) > CHUNK_TARGET_BYTES {
            chunk_points /= 2;
        }
        let per_chunk = per_point(chunk_points).saturating_add(fixed);
        led.check(
            &format!(
                "k-point UKS XC force AO/Hessian chunk ({chunk_points} points, nbf = {nbf}, N_k = {nk})"
            ),
            per_chunk,
        )?;
        let par = rayon::current_num_threads()
            .max(1)
            .min(self.max_live_chunks.unwrap_or(usize::MAX).max(1))
            .min((led.remaining() / per_chunk.max(1)).max(1));
        led.reserve(
            &format!("k-point UKS XC force chunks ({par} x {chunk_points} points)"),
            per_chunk.saturating_mul(par),
        )?;
        let chunks = LatticeAoHess::chunks(&points, chunk_points);
        let pref = if mutation == Some(KXcMutation::WrongFactor) {
            -1.0
        } else {
            -2.0
        };
        let opts = ForceOpts {
            use_gga: self.gga && mutation != Some(KXcMutation::NoGga),
            tau_term: self.mgga && mutation != Some(KXcMutation::TauNoAo),
            hessian: mutation != Some(KXcMutation::NoHessian),
            pref,
            tau_pref: if mutation == Some(KXcMutation::TauDoubled) {
                pref
            } else {
                0.5 * pref
            },
            kscale: if mutation == Some(KXcMutation::NoInvNk) {
                1.0
            } else {
                1.0 / nk as f64
            },
        };
        let alt_phase = match mutation {
            Some(KXcMutation::NoPhase) => Some(BlochPhase::None),
            Some(KXcMutation::ConjPhase) => Some(BlochPhase::Conj),
            _ => None,
        };
        let dsum: Option<Vec<Array2<C64>>> = (smut == Some(KUSpinMutation::XcSpinAveraged))
            .then(|| da.iter().zip(db).map(|(a, b)| a + b).collect());

        let run = |idx: &Vec<usize>| -> Result<ChunkPol, FerricError> {
            let pts: Vec<GridPoint> = idx.iter().map(|&i| points[i]).collect();
            let xyz: Vec<[f64; 3]> = pts.iter().map(|g| g.xyz).collect();
            let np = pts.len();
            let bl: BlochAoHess = ev.eval_bloch(&xyz, &self.kpts, BlochPhase::True, self.gga)?;
            let dena = chunk_density(da, &bl.chi, &bl.dchi, self.gga);
            let denb = chunk_density(db, &bl.chi, &bl.dchi, self.gga);
            let mut sigma = Array2::<f64>::zeros((3, np));
            for g in 0..np {
                let dot = |x: &Array2<f64>, y: &Array2<f64>| -> f64 {
                    (0..3).map(|ax| x[(ax, g)] * y[(ax, g)]).sum()
                };
                sigma[(0, g)] = dot(&dena.grad, &dena.grad);
                sigma[(1, g)] = dot(&dena.grad, &denb.grad);
                sigma[(2, g)] = dot(&denb.grad, &denb.grad);
            }
            let dens = UksDensityGrid {
                rho_a: dena.rho,
                rho_b: denb.rho,
                grad_a: dena.grad,
                grad_b: denb.grad,
                sigma,
            };
            // Per-spin tau of THIS chunk (the energy's `chunk_tau` of `D_σ`).
            let tau = self
                .mgga
                .then(|| (chunk_tau(da, &bl.dchi), chunk_tau(db, &bl.dchi)));
            let kern = polarized_kernel(&dens, tau.as_ref().map(|(a, b)| (a, b)), &self.xc);
            let mut e = 0.0;
            let mut ne = 0.0;
            for g in 0..np {
                let rt = dens.rho_a[g] + dens.rho_b[g];
                e += pts[g].weight * rt * kern.exc[g];
                ne += pts[g].weight * rt;
            }
            let true_vtau = self.mgga.then_some((&kern.vtau_a, &kern.vtau_b));
            let wts: Vec<f64> = pts.iter().map(|p| p.weight).collect();
            // --- V_xc,σ(k) with the TRUE kernel vectors (V^τ_σ included).
            let potential = |s: usize| -> Vec<Array2<C64>> {
                let sk = spin_kernel(s, Cross::Other, &dens, &kern, self.gga, true_vtau);
                let mut v = chunk_vxc(&bl.chi, &bl.dchi, &wts, &sk.vr, &sk.c, self.gga);
                let drop_tau = matches!(
                    mutation,
                    Some(KXcMutation::TauNoVxc) | Some(KXcMutation::TauNoVxcUnguarded)
                ) || (s == 1 && smut == Some(KUSpinMutation::TauNoVxcBeta));
                if self.mgga && !drop_tau {
                    add_vtau(&mut v, &bl.dchi, &wts, &sk.vt);
                }
                v
            };
            let va = potential(0);
            let vb = potential(1);
            // --- Force vectors (possibly mutated).
            let force_vtau = tau.as_ref().map(|t| {
                tau_force_vtau(
                    &dens,
                    t,
                    &kern,
                    &self.xc,
                    smut == Some(KUSpinMutation::TauTotalBothChannels),
                )
            });
            let passes = force_passes(
                da,
                db,
                dsum.as_deref(),
                &dens,
                &kern,
                self.gga,
                force_vtau.as_ref().map(|(a, b)| (a, b)),
                smut,
            );
            let alt: Option<BlochAoHess> = alt_phase
                .map(|ph| ev.eval_bloch(&xyz, &self.kpts, ph, self.gga))
                .transpose()?;
            let fb: &BlochAoHess = alt.as_ref().unwrap_or(&bl);
            let mut a_pt = Array3::<f64>::zeros((np, natoms, 3));
            if mutation != Some(KXcMutation::NoAo) {
                for (dm, sk) in &passes {
                    ao_force_pass(dm, sk, fb, &wts, &self.aoat, &opts, &mut a_pt);
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
                        if let Some(dw) = dweight.as_ref() {
                            let eps = (dens.rho_a[g] + dens.rho_b[g]) * kern.exc[g];
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
            Ok(ChunkPol {
                e,
                ne,
                va,
                vb,
                ao,
                point,
                weight,
            })
        };

        let mut e_xc = 0.0;
        let mut electrons = 0.0;
        let zeros = || -> Vec<Array2<C64>> { (0..nk).map(|_| Array2::zeros((nbf, nbf))).collect() };
        let mut va = zeros();
        let mut vb = zeros();
        let mut ao = Array2::<f64>::zeros((natoms, 3));
        let mut point = Array2::<f64>::zeros((natoms, 3));
        let mut weight = Array2::<f64>::zeros((natoms, 3));
        for batch in chunks.chunks(par) {
            let outs: Vec<Result<ChunkPol, FerricError>> = batch.par_iter().map(&run).collect();
            for out in outs {
                let c = out?;
                e_xc += c.e;
                electrons += c.ne;
                for (acc, x) in va.iter_mut().zip(&c.va) {
                    *acc += x;
                }
                for (acc, x) in vb.iter_mut().zip(&c.vb) {
                    *acc += x;
                }
                ao += &c.ao;
                point += &c.point;
                weight += &c.weight;
            }
        }
        if !e_xc.is_finite()
            || va
                .iter()
                .chain(vb.iter())
                .any(|m| m.iter().any(|z| !z.re.is_finite()))
            || ao
                .iter()
                .chain(point.iter())
                .chain(weight.iter())
                .any(|x| !x.is_finite())
        {
            return Err(FerricError::General(
                "k-point UKS XC force: non-finite E_xc/V_xc/force".into(),
            ));
        }
        for m in va.iter_mut().chain(vb.iter_mut()) {
            *m = hermitize(m);
        }
        match mutation {
            Some(KXcMutation::NoPointMotion) => point.fill(0.0),
            Some(KXcMutation::NoWeightDeriv) => weight.fill(0.0),
            _ => {}
        }
        Ok(XcOutPol {
            e_xc,
            electrons,
            va,
            vb,
            ao,
            point,
            weight,
            npts: points.len(),
            chunk_points,
            live_chunks: par,
        })
    }
}

/// Resolve an open-shell derivative's functional and exact-exchange fraction.
/// `allow_mgga` (UKS) lets the validated pure meta-GGAs SCAN / r2SCAN through
/// on either J/K (dense-AFT or RS-GDF; `tests/pbc_kuks_mgga_rsgdf.rs`);
/// ROKS/ROHF keep [`resolve_global`]'s
/// refusal (the k-point ROKS SCF has no tau term). `base` is the force's or
/// the stress's own resolver (their refusal wordings differ). Returns `(a, mgga)`;
/// shared with the open-shell stress. `who` names the entry point.
pub(crate) fn resolve_open_shell(
    who: &str,
    functional: &str,
    allow_mgga: bool,
    base: &dyn Fn(&str) -> Result<(XcDef, f64), FerricError>,
) -> Result<(f64, bool), FerricError> {
    if !allow_mgga {
        return Ok((resolve_global(functional)?.1, false));
    }
    let (xc, exch) = base(functional)?;
    let mgga = xc
        .funcs
        .iter()
        .any(|f| matches!(f.family(), FunctionalFamily::MetaGga));
    if mgga && !crate::kxcgrad::is_validated_mgga(functional) {
        return Err(PeriodicDftError::Unsupported {
            feature: "open-shell meta-GGA derivative at k",
            reason: format!(
                "{who}: {functional}: the k-point UKS force and stress cover the pure SCAN and \
                 r2SCAN meta-GGAs only (those the k-point UKS SCF accepts)"
            ),
        }
        .into());
    }
    if mgga && (xc.vv10.is_some() || xc.weights.is_some()) {
        return Err(PeriodicDftError::Unsupported {
            feature: "VV10/double-hybrid derivative at k",
            reason: format!("{functional}: VV10 and mixing weights are not covered"),
        }
        .into());
    }
    Ok((exch, mgga))
}

/// The polarized XC response of an open-shell force, the exchange scale and
/// the range separation. Both the UKS and the ROKS force take a
/// range-separated hybrid (exchange scale 1, its fractions inside the kernel;
/// [`resolve_global_or_rsh`] refuses a range-separated meta-GGA, VV10 or
/// double hybrid; `check_inputs_rsh` checks the J/K carry its parameters).
/// `allow_mgga` marks the UKS force (SCAN / r2SCAN); ROKS keeps
/// [`resolve_global`]'s meta-GGA refusal.
fn grid_response<'a>(
    cell: &'a Cell,
    prep: &'a PreparedBasis,
    mesh: &KPointMesh,
    functional: &str,
    cfg: &'a KUksGradConfig,
    (who, allow_mgga): (&str, bool),
) -> Result<(GridXcPol<'a>, f64, Option<RshParams>), FerricError> {
    let rsh = resolve_global_or_rsh(functional)?.2;
    let (exch, is_mgga) = if rsh.is_some() {
        (1.0, false)
    } else {
        resolve_open_shell(who, functional, allow_mgga, &resolve_global_mgga)?
    };
    let mut name = functional.to_string();
    if cfg.base.mutation == Some(KXcMutation::WrongRemainder) {
        name = match functional.to_ascii_uppercase().as_str() {
            "PBE0" => "PBE",
            "B3LYP" => "BLYP",
            other => {
                return Err(FerricError::General(format!(
                    "WrongRemainder mutant: no pure parent registered for {other}"
                )))
            }
        }
        .to_string();
    }
    if cfg.base.mutation == Some(KXcMutation::TauNoPoint) {
        return Err(FerricError::General(
            "TauNoPoint mutant: wired for the closed-shell force only".into(),
        ));
    }
    let xc = xc_def_from_name_nspin(&name, 2)
        .map_err(|e| FerricError::General(format!("{functional}: spin-polarized: {e:?}")))?;
    let gga = xc
        .funcs
        .iter()
        .any(|f| !matches!(f.family(), FunctionalFamily::Lda));
    let force_tau = cfg.base.mutation == Some(KXcMutation::ForceTauPath);
    if force_tau && !gga {
        return Err(FerricError::General(
            "ForceTauPath mutant: needs a GGA functional (the AO Hessian is evaluated only for GGA)"
                .into(),
        ));
    }
    Ok((
        GridXcPol {
            cell,
            prep,
            xc,
            gga,
            mgga: is_mgga || force_tau,
            kpts: mesh.kpts().to_vec(),
            aoat: ao_atoms(prep),
            grid: &cfg.base.grid,
            ao_threshold: cfg.base.ao_threshold,
            max_live_chunks: cfg.base.max_live_chunks,
            mutation: cfg.base.mutation,
            spin_mutation: cfg.spin_mutation,
        },
        exch,
        rsh,
    ))
}

/// k-point open-shell UKS nuclear gradient (module doc) of a converged
/// [`crate::kuks::solve_kuks`] result (`KUksResult::scf`) for an LDA, GGA or
/// global-hybrid `functional` (a hybrid also needs
/// [`KRksGradConfig::exxdiv`], here `cfg.base.exxdiv`). Arguments as
/// [`crate::kxcgrad::kpoint_rks_gradient`], with the SCF a [`KUScfResult`].
///
/// Also the pure meta-GGAs SCAN / r2SCAN on the dense-AFT or RS-GDF J/K
/// (module doc, "Meta-GGA (UKS)").
///
/// Also range-separated hybrids (HSE06 and CAM-B3LYP force-tested; wB97X and
/// other CAM GGA hybrids accepted, not exercised by a force test) on the
/// dense-AFT or RS-GDF J/K built with the functional's [`RshParams`] (as
/// `solve_kuks` builds them): `K_σ` with scale 1 and the mixed Madelung
/// constant (module doc, "Range-separated hybrids (UKS, ROKS)").
///
/// Refuses (typed errors) any other meta-GGA, VV10/double hybrids, a hybrid without
/// `cfg.base.exxdiv`, and an SCF whose spin Focks the rebuilt
/// `h + J − a K_σ + V_xc,σ` do not reproduce.
#[allow(clippy::too_many_arguments)]
pub fn kpoint_uks_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KUScfResult,
    functional: &str,
    cfg: &KUksGradConfig,
) -> Result<KUksGradient, FerricError> {
    let (resp, exch, rsh) = grid_response(
        cell,
        prep,
        mesh,
        functional,
        cfg,
        ("kpoint_uks_gradient", true),
    )?;
    uks_gradient_with(
        "kpoint_uks_gradient",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        &uks_input(scf),
        &resp,
        (exch, rsh),
        cfg,
    )
}

/// k-point ROKS nuclear gradient of a converged
/// [`crate::kroks::solve_kroks`] result (`KRoksResult::scf`): the open-shell
/// assembly on the ROHF spin densities `D_α, D_β` and spin Focks, with the
/// orthonormality-Lagrangian `W_RO = sym(D_α F_α D_α + D_α F_β D_β)` at each k
/// (module doc "ROHF / ROKS"). Arguments and refusals as
/// [`kpoint_uks_gradient`] (range-separated hybrids included, on the
/// dense-AFT or RS-GDF J/K built with the functional's [`RshParams`], as `solve_kroks`
/// builds them; module doc "Range-separated hybrids (UKS, ROKS)"), except
/// meta-GGA (the k-point ROKS SCF has no tau term), plus: a state whose ROHF
/// orbital gradient (closed-open `f_β`, open-virtual `f_α`, closed-virtual
/// `f_α + f_β`, any k) exceeds [`crate::grad::RO_ORBITAL_GRADIENT_TOL`] is
/// refused.
#[allow(clippy::too_many_arguments)]
pub fn kpoint_roks_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KRoScfResult,
    functional: &str,
    cfg: &KUksGradConfig,
) -> Result<KUksGradient, FerricError> {
    let parts = RoParts {
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
    };
    parts.roks_gradient("kpoint_roks_gradient", jk, &ro_input(scf), functional, cfg)
}

/// The cell, basis, mesh and hcore of a ROKS force (the arguments
/// [`kpoint_roks_gradient`] and the routed Gamma ROKS force share).
pub(crate) struct RoParts<'a> {
    pub(crate) cell: &'a Cell,
    pub(crate) prep: &'a PreparedBasis,
    pub(crate) mesh: &'a KPointMesh,
    pub(crate) hcore_cfg: &'a PeriodicHcoreConfig,
    pub(crate) hk: &'a PeriodicHcoreK,
}

impl RoParts<'_> {
    /// The ROKS force of the spin state `inp` (module doc "ROHF / ROKS").
    pub(crate) fn roks_gradient(
        &self,
        who: &str,
        jk: KGradJk<'_>,
        inp: &SpinInput<'_>,
        functional: &str,
        cfg: &KUksGradConfig,
    ) -> Result<KUksGradient, FerricError> {
        let (resp, exch, rsh) = grid_response(
            self.cell,
            self.prep,
            self.mesh,
            functional,
            cfg,
            (who, false),
        )?;
        uks_gradient_with(
            who,
            self.cell,
            self.prep,
            self.mesh,
            self.hcore_cfg,
            self.hk,
            jk,
            inp,
            &resp,
            (exch, rsh),
            cfg,
        )
    }
}

/// k-point ROHF nuclear gradient of a converged
/// [`crate::kroks::solve_krohf`] result: [`kpoint_roks_gradient`] with a zero
/// XC and `a = 1` (`exxdiv` is the SCF's, replacing `cfg.base.exxdiv`).
#[allow(clippy::too_many_arguments)]
pub fn kpoint_rohf_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KRoScfResult,
    exxdiv: ExxDiv,
    cfg: &KUksGradConfig,
) -> Result<KUksGradient, FerricError> {
    let mut c = cfg.clone();
    c.base.exxdiv = Some(exxdiv);
    uks_gradient_with(
        "kpoint_rohf_gradient",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        &ro_input(scf),
        &NullXcPol,
        (1.0, None),
        &c,
    )
}

/// TEST/DIAGNOSTIC: the open-shell KS machinery with a zero XC and
/// exact-exchange fraction `exch`, on an SCF run by `run_kuhf` with a zero
/// polarized XC builder and the same `exch`. With `exch = 1` it is the k-point
/// UHF force ([`crate::kgrad::kpoint_uhf_gradient`]). A hybrid
/// (`exch != 0`) needs `cfg.base.exxdiv`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn kpoint_null_xc_uks_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KUScfResult,
    exch: f64,
    cfg: &KUksGradConfig,
) -> Result<KUksGradient, FerricError> {
    kpoint_null_xc_uks_gradient_rsh(cell, prep, mesh, hcore_cfg, hk, jk, scf, exch, None, cfg)
}

/// TEST/DIAGNOSTIC: [`kpoint_null_xc_uks_gradient`] on dense kernels that
/// carry the range separation `rsh` (any `ω`, `c_sr`, `c_lr`), with the
/// exchange scale `exch` (1 for a range-separated K) and, under
/// `exxdiv = ewald`, the mixed Madelung constant of `rsh`. The open-shell
/// anchor entry for `c_sr = c_lr = c` (the global hybrid with fraction `c`).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn kpoint_null_xc_uks_gradient_rsh(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KUScfResult,
    exch: f64,
    rsh: Option<RshParams>,
    cfg: &KUksGradConfig,
) -> Result<KUksGradient, FerricError> {
    uks_gradient_with(
        "kpoint_null_xc_uks_gradient",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        &uks_input(scf),
        &NullXcPol,
        (exch, rsh),
        cfg,
    )
}

/// TEST/DIAGNOSTIC: the k-point ROKS force with a zero XC, exchange scale
/// `exch` and the range separation `rsh` of the dense kernels (Madelung shift
/// as [`kpoint_null_xc_uks_gradient_rsh`]), on an SCF from
/// [`crate::kroks::solve_kroks_injected`] without XC and the same `exch`. The
/// restricted-open-shell anchor entry for `c_sr = c_lr = c`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn kpoint_null_xc_roks_gradient_rsh(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KRoScfResult,
    exch: f64,
    rsh: Option<RshParams>,
    cfg: &KUksGradConfig,
) -> Result<KUksGradient, FerricError> {
    uks_gradient_with(
        "kpoint_null_xc_roks_gradient",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        &ro_input(scf),
        &NullXcPol,
        (exch, rsh),
        cfg,
    )
}

fn uks_input(scf: &KUScfResult) -> SpinInput<'_> {
    SpinInput {
        da: &scf.density_alpha,
        db: &scf.density_beta,
        fock: StoredFock::Spin(&scf.fock_alpha, &scf.fock_beta),
        converged: scf.converged,
        kpts: &scf.kpts,
        mos: None,
    }
}

fn ro_input(scf: &KRoScfResult) -> SpinInput<'_> {
    SpinInput {
        da: &scf.density_alpha,
        db: &scf.density_beta,
        fock: StoredFock::Spin(&scf.fock_alpha, &scf.fock_beta),
        converged: scf.converged,
        kpts: &scf.kpts,
        mos: Some(&scf.mos),
    }
}

#[allow(clippy::too_many_arguments)]
fn uks_gradient_with(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    inp: &SpinInput<'_>,
    xc: &dyn KUXcResponse,
    (exch, rsh): (f64, Option<RshParams>),
    cfg: &KUksGradConfig,
) -> Result<KUksGradient, FerricError> {
    let ro = inp.mos.is_some();
    check_inputs_rsh(
        who,
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        &[inp.da, inp.db],
        inp.converged,
        inp.kpts,
        rsh,
    )?;
    if !(0.0..=1.0).contains(&exch) {
        return Err(FerricError::General(format!(
            "{who}: exact-exchange fraction {exch} outside [0, 1]"
        )));
    }
    let natoms = cell.positions().len();
    let nk = mesh.nk();
    // The SCF's K shift: mesh v_M, or the mixed constant of a range-separated K.
    let vm = hybrid_madelung(who, cell, mesh, exch, cfg.base.exxdiv, rsh)?;
    let mut ledger = open_ledger(cell, prep, mesh, &cfg.base.grad, true)?;
    let da = inp.da.to_vec();
    let db = inp.db.to_vec();
    let mut xo = xc.evaluate(&da, &db, &ledger)?;
    if xo.ao.nrows() == 0 {
        xo.ao = Array2::zeros((natoms, 3));
        xo.point = Array2::zeros((natoms, 3));
        xo.weight = Array2::zeros((natoms, 3));
    }
    ledger.reserve(
        &format!(
            "k UKS force Fock-guard matrices D_a, D_b, J_a, J_b, K_a, K_b, F_a, F_b, V_a, V_b \
             (n = {}, N_k = {nk})",
            prep.nbasis()
        ),
        bytes_of((nk * prep.nbasis() * prep.nbasis()) as u64, 16 * 10),
    )?;
    let (ja, ka) = jk_build(&jk, vm, &da)?;
    let (jb, kb) = jk_build(&jk, vm, &db)?;
    let smut = cfg.spin_mutation;
    let mutation = cfg.base.mutation;
    let build = |s: usize| -> Vec<Array2<C64>> {
        (0..nk)
            .map(|q| {
                let jtot = if smut == Some(KUSpinMutation::PulayJOneSpin) {
                    ja[q].clone()
                } else {
                    &ja[q] + &jb[q]
                };
                let mut base = &hk.h[q] + &jtot;
                if exch != 0.0 {
                    let swap = smut == Some(KUSpinMutation::PulayKSwap);
                    let kk = if (s == 0) != swap { &ka[q] } else { &kb[q] };
                    base.scaled_add(C64::new(-exch, 0.0), kk);
                }
                let base = hermitize(&base);
                let vxc = if s == 0 { &xo.va[q] } else { &xo.vb[q] };
                let drop_v = mutation == Some(KXcMutation::PulayNoVxc)
                    || (s == 1 && smut == Some(KUSpinMutation::PulayNoVxcBeta));
                if drop_v {
                    base
                } else {
                    hermitize(&(&base + vxc))
                }
            })
            .collect()
    };
    let fa = build(0);
    let fb = build(1);
    drop((ja, jb, ka, kb));
    let guard = smut.map_or(true, |m| {
        !matches!(
            m,
            KUSpinMutation::PulayFockSwap
                | KUSpinMutation::PulayKSwap
                | KUSpinMutation::PulayJOneSpin
                | KUSpinMutation::PulayNoVxcBeta
                | KUSpinMutation::TauNoVxcBeta
        )
    }) && !matches!(
        mutation,
        Some(KXcMutation::PulayNoVxc)
            | Some(KXcMutation::WrongRemainder)
            | Some(KXcMutation::GuardSkippedOnly)
            | Some(KXcMutation::TauNoVxcUnguarded)
    );
    let mismatch = if guard {
        stored_fock_mismatch(who, inp.fock, (&fa, &fb), (&da, &db), &hk.s)?
    } else {
        0.0
    };
    let (fa, fb) = if smut == Some(KUSpinMutation::PulayFockSwap) {
        (fb, fa)
    } else {
        (fa, fb)
    };
    let og = match inp.mos {
        Some(mos) => {
            ledger.check(
                &format!(
                    "k RO force orbital-gradient gate scratch (n = {}, per-k transients)",
                    prep.nbasis()
                ),
                ro_gate_scratch_bytes(prep.nbasis()),
            )?;
            let g = ro_orbital_gradient_k(who, mos, &hk.s, &da, &db, &fa, &fb)?;
            let gate_on = !matches!(
                smut,
                Some(KUSpinMutation::RoGateOff) | Some(KUSpinMutation::RoWUhfFormGateOff)
            );
            if gate_on && (g.is_nan() || g > RO_ORBITAL_GRADIENT_TOL) {
                return Err(FerricError::General(format!(
                    "{who}: ROHF orbital gradient {g:.3e} exceeds {RO_ORBITAL_GRADIENT_TOL:.0e}: not \
                     a stationary ROHF state (the force is first order in it); converge the SCF \
                     tighter"
                )));
            }
            Some(g)
        }
        _ => None,
    };
    let uhf_w = matches!(
        smut,
        Some(KUSpinMutation::RoWUhfForm) | Some(KUSpinMutation::RoWUhfFormGateOff)
    );
    let spins = if !ro || uhf_w {
        KSpins::Unrestricted { da, db, fa, fb }
    } else if smut == Some(KUSpinMutation::RoWDropCross) {
        let zb = fb.iter().map(|m| Array2::zeros(m.dim())).collect();
        KSpins::RestrictedOpen { da, db, fa, fb: zb }
    } else {
        KSpins::RestrictedOpen { da, db, fa, fb }
    };
    let mut hf = assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        spins,
        vm,
        exch,
        &cfg.base.grad,
        &mut ledger,
    )?;
    hf.fock_mismatch = mismatch;
    // (a test defect that tampers with F skips the Fock guard and, with it, this gate)
    if !ro
        && guard
        && smut != Some(KUSpinMutation::StationarityGateOff)
        && (hf.commutator.is_nan() || hf.commutator > UKS_COMMUTATOR_TOL)
    {
        return Err(FerricError::General(format!(
            "{who}: the spin Fock/density commutator max|F D S - S D F| = {:.3e} exceeds \
             {UKS_COMMUTATOR_TOL:.0e}: not a stationary state (the force is first order in it); \
             converge the SCF tighter",
            hf.commutator
        )));
    }
    if let Some(g) = og {
        // The per-spin commutators do not vanish at a ROHF state; report the
        // gated quantity instead.
        hf.commutator = g;
    }
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

/// The pre-fix measurement (blocks read in the SCF's MOs), kept as a mutant.
#[cfg(test)]
#[allow(clippy::needless_range_loop)]
fn old_ro_orbital_gradient_k(
    who: &str,
    mos: &[Array2<C64>],
    s: &[Array2<C64>],
    da: &[Array2<C64>],
    db: &[Array2<C64>],
    fa: &[Array2<C64>],
    fb: &[Array2<C64>],
) -> Result<f64, FerricError> {
    const OCC_TOL: f64 = 1e-6;
    let nk = s.len();
    if mos.len() != nk || da.len() != nk || db.len() != nk || fa.len() != nk || fb.len() != nk {
        return Err(FerricError::General(format!(
            "{who}: {} MO sets for {nk} k-points (the SCF's MOs must be the mesh's, one set per k)",
            mos.len()
        )));
    }
    let mut worst = 0.0_f64;
    for k in 0..mos.len() {
        let c = &mos[k];
        let ch = c.t().mapv(|z| z.conj());
        let sc = s[k].dot(c);
        let sch = sc.t().mapv(|z| z.conj());
        let occ = |d: &Array2<C64>| -> Vec<f64> {
            let m = sch.dot(d).dot(&sc);
            (0..c.ncols()).map(|i| m[(i, i)].re).collect()
        };
        let (oa, ob) = (occ(&da[k]), occ(&db[k]));
        let (mut closed, mut open, mut virt) = (vec![], vec![], vec![]);
        for i in 0..c.ncols() {
            let (ra, rb) = (oa[i].round(), ob[i].round());
            if (oa[i] - ra).abs() > OCC_TOL
                || (ob[i] - rb).abs() > OCC_TOL
                || !(ra == 0.0 || ra == 1.0)
                || !(rb == 0.0 || rb == 1.0)
            {
                return Err(FerricError::General(format!(
                    "{who}: MO {i} at k-point {k} has occupations (alpha {:.3e}, beta {:.3e}), not 0/1: \
                     the MOs are not the orbital set of (D_alpha, D_beta)",
                    oa[i], ob[i]
                )));
            }
            match (ra == 1.0, rb == 1.0) {
                (true, true) => closed.push(i),
                (true, false) => open.push(i),
                (false, false) => virt.push(i),
                (false, true) => {
                    return Err(FerricError::General(format!(
                        "{who}: MO {i} at k-point {k} is beta-occupied but alpha-empty (not a \
                         ROHF state)"
                    )))
                }
            }
        }
        let fam = ch.dot(&fa[k]).dot(c);
        let fbm = ch.dot(&fb[k]).dot(c);
        let block = |rows: &[usize], cols: &[usize], f: &dyn Fn(usize, usize) -> C64| {
            let mut m = 0.0_f64;
            for &p in rows {
                for &q in cols {
                    let a = f(p, q).norm();
                    if a.is_nan() || a > m {
                        m = a;
                    }
                }
            }
            m
        };
        worst = worst
            .max(block(&closed, &open, &|p, q| fbm[(p, q)]))
            .max(block(&open, &virt, &|p, q| fam[(p, q)]))
            .max(block(&closed, &virt, &|p, q| fam[(p, q)] + fbm[(p, q)]));
    }
    Ok(worst)
}

/// Unit tests of the ROHF orbital-gradient measurement on a toy model: fixed
/// spin Focks with S = 1 (so the only moving part is the density), a
/// stationary ROHF state (closed {0}, open {1}, virtual {2..5}) rotated by
/// Givens angles `kappa`, and the MOs built as the SCF's `finish()` builds
/// them: the eigenvectors of the Roothaan `F_eff[D]` of the final density.
#[cfg(test)]
mod gate_tests {
    use super::*;
    use crate::kroks::roothaan_fock_k;

    const N: usize = 6;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as f64) / ((1u64 << 31) as f64) - 0.5
    }

    fn sym(state: &mut u64) -> Array2<f64> {
        let mut a = Array2::<f64>::zeros((N, N));
        for i in 0..N {
            for j in 0..=i {
                let v = lcg(state);
                a[(i, j)] = v;
                a[(j, i)] = v;
            }
            a[(i, i)] += (i as f64) * 0.7;
        }
        a
    }

    /// Spin Focks whose three gradient blocks vanish for closed {0}, open {1},
    /// virtual {2..N}: `(f_b)_01 = 0`, `(f_a)_{1v} = 0`, `(f_a + f_b)_{0v} = 0`.
    fn stationary_focks() -> (Array2<f64>, Array2<f64>) {
        let mut st = 12345u64;
        let mut fa = sym(&mut st);
        let mut fb = sym(&mut st);
        fb[(0, 1)] = 0.0;
        fb[(1, 0)] = 0.0;
        for v in 2..N {
            fa[(1, v)] = 0.0;
            fa[(v, 1)] = 0.0;
            fb[(0, v)] = -fa[(0, v)];
            fb[(v, 0)] = -fa[(v, 0)];
        }
        (fa, fb)
    }

    fn givens(p: usize, q: usize, kappa: f64) -> Array2<f64> {
        let mut g = Array2::<f64>::eye(N);
        g[(p, p)] = kappa.cos();
        g[(q, q)] = kappa.cos();
        g[(p, q)] = -kappa.sin();
        g[(q, p)] = kappa.sin();
        g
    }

    fn cplx(a: &Array2<f64>) -> Array2<C64> {
        a.mapv(|x| C64::new(x, 0.0))
    }

    struct Toy {
        da: Array2<C64>,
        db: Array2<C64>,
        fa: Array2<C64>,
        fb: Array2<C64>,
        s: Array2<C64>,
        /// Closed-open, open-virtual and closed-virtual gradient norms read
        /// directly in the rotated orbitals (the independent reference).
        truth: [f64; 3],
        /// The SCF's MOs: eigenvectors of F_eff[D].
        mos: Array2<C64>,
    }

    /// Rotations `[(p, q, kappa)]` applied to the stationary orbital set.
    fn toy(rots: &[(usize, usize, f64)]) -> Toy {
        let (fa, fb) = stationary_focks();
        let mut c = Array2::<f64>::eye(N);
        for &(p, q, k) in rots {
            c = c.dot(&givens(p, q, k));
        }
        let col = |i: usize| c.column(i).to_owned();
        let outer = |u: &ndarray::Array1<f64>| {
            let mut m = Array2::<f64>::zeros((N, N));
            for i in 0..N {
                for j in 0..N {
                    m[(i, j)] = u[i] * u[j];
                }
            }
            m
        };
        let db = outer(&col(0));
        let da = &db + &outer(&col(1));
        let fam = c.t().dot(&fa).dot(&c);
        let fbm = c.t().dot(&fb).dot(&c);
        let nrm = |v: Vec<f64>| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let truth = [
            fbm[(0, 1)].abs(),
            nrm((2..N).map(|v| fam[(1, v)]).collect()),
            nrm((2..N).map(|v| fam[(0, v)] + fbm[(0, v)]).collect()),
        ];
        let s = Array2::<C64>::eye(N);
        let (dac, dbc) = (cplx(&da), cplx(&db));
        let feff = roothaan_fock_k(&cplx(&fa), &cplx(&fb), &dac, &dbc, &s);
        let (_, mos) = crate::kscf::eigh_herm(&feff).unwrap();
        Toy {
            da: dac,
            db: dbc,
            fa: cplx(&fa),
            fb: cplx(&fb),
            s,
            truth,
            mos,
        }
    }

    fn new_gate(t: &Toy, m: Option<GateMutation>) -> Result<f64, FerricError> {
        ro_orbital_gradient_with(
            "toy",
            std::slice::from_ref(&t.mos),
            std::slice::from_ref(&t.s),
            std::slice::from_ref(&t.da),
            std::slice::from_ref(&t.db),
            std::slice::from_ref(&t.fa),
            std::slice::from_ref(&t.fb),
            m,
        )
    }

    fn old_gate(t: &Toy) -> Result<f64, FerricError> {
        old_ro_orbital_gradient_k(
            "toy",
            std::slice::from_ref(&t.mos),
            std::slice::from_ref(&t.s),
            std::slice::from_ref(&t.da),
            std::slice::from_ref(&t.db),
            std::slice::from_ref(&t.fa),
            std::slice::from_ref(&t.fb),
        )
    }

    /// An MO set of the wrong shape is a typed error, not an ndarray panic.
    #[test]
    fn wrong_shaped_mo_set_is_a_typed_error() {
        let mut t = toy(&[]);
        let n = t.s.nrows();
        let full = t.mos.clone();
        t.mos = full.slice(ndarray::s![..n - 1, ..]).to_owned();
        let e = new_gate(&t, None).unwrap_err().to_string();
        assert!(e.contains("MO set at k-point 0"), "{e}");
        t.mos = ndarray::concatenate![ndarray::Axis(1), full, full.slice(ndarray::s![.., ..1])];
        let e = new_gate(&t, None).unwrap_err().to_string();
        assert!(e.contains("MO set at k-point 0"), "{e}");
        t.mos = full;
        assert!(new_gate(&t, None).unwrap() < 1e-12);
    }

    /// The stationary state: both forms report ~0.
    #[test]
    fn stationary_state_has_zero_gradient() {
        let t = toy(&[]);
        let g = new_gate(&t, None).unwrap();
        assert!(g < 1e-12, "{g:e}");
        assert!(old_gate(&t).unwrap() < 1e-12);
    }

    /// Old vs new vs the reference, one rotation per block and all three
    /// together, kappa 1e-2 .. 1e-5.
    #[test]
    fn gate_equals_the_true_gradient_where_the_old_gate_under_reported() {
        let cases: [(&str, Vec<(usize, usize)>); 4] = [
            ("closed-open", vec![(0, 1)]),
            ("open-virtual", vec![(1, 2)]),
            ("closed-virtual", vec![(0, 2)]),
            ("all three", vec![(0, 1), (1, 2), (0, 2)]),
        ];
        let mut worst_old_ratio = 1.0_f64;
        for (name, pq) in &cases {
            let mut prev: Option<f64> = None;
            for kappa in [1e-2, 1e-3, 1e-4, 1e-5] {
                let rots: Vec<_> = pq.iter().map(|&(p, q)| (p, q, kappa)).collect();
                let t = toy(&rots);
                let truth = t.truth.iter().cloned().fold(0.0_f64, f64::max);
                let g = new_gate(&t, None).unwrap();
                let old = old_gate(&t);
                eprintln!(
                    "{name:<15} kappa {kappa:.0e}: true {truth:.4e}  new {g:.4e}  old {}",
                    match &old {
                        Ok(v) => format!("{v:.4e} (true/old {:.1})", truth / v.max(1e-300)),
                        Err(e) => format!(
                            "refused ({})",
                            e.to_string().chars().take(60).collect::<String>()
                        ),
                    }
                );
                // The new form IS the true gradient (independent reference in
                // the rotated orbitals) ...
                assert!(
                    ((g - truth) / truth).abs() < 1e-8,
                    "{name} {kappa}: {g:e} vs {truth:e}"
                );
                // ... and is proportional to kappa with a fixed coefficient.
                if let Some(p) = prev {
                    let r = (g / kappa) / p;
                    assert!((r - 1.0).abs() < 10.0 * kappa * 10.0, "{name} {kappa}: {r}");
                }
                prev = Some(g / kappa);
                if let Ok(v) = old {
                    worst_old_ratio = worst_old_ratio.max(truth / v.max(1e-300));
                }
            }
        }
        eprintln!("worst true/old ratio {worst_old_ratio:.1}");
    }

    /// Mutants: a dropped block, the wrong metric, and the OLD measurement
    /// (blocks read in the eigenvectors of F_eff of the final density) are
    /// each visible against the reference.
    #[test]
    fn gate_mutants_are_visible() {
        for (name, pq, m) in [
            ("co", (0, 1), GateMutation::DropClosedOpen),
            ("ov", (1, 2), GateMutation::DropOpenVirtual),
            ("cv", (0, 2), GateMutation::DropClosedVirtual),
        ] {
            let t = toy(&[(pq.0, pq.1, 1e-4)]);
            let g = new_gate(&t, None).unwrap();
            let gm = new_gate(&t, Some(m)).unwrap();
            eprintln!("drop {name}: {gm:.4e} vs {g:.4e}");
            assert!(gm < 0.9 * g, "{m:?}: {gm:e} vs {g:e}");
        }
        let t = toy(&[(0, 1, 1e-4), (1, 2, 1e-4), (0, 2, 1e-4)]);
        // A non-orthonormal AO overlap makes the wrong metric visible: the
        // orbitals are S-orthonormal, S = diag(1 + 0.4 i / N), mos = S^(-1/2).
        {
            let (fa, fb) = stationary_focks();
            let sd: Vec<f64> = (0..N).map(|i| 1.0 + 0.4 * i as f64 / N as f64).collect();
            let sinv12 = Array2::from_diag(&ndarray::Array1::from_vec(
                sd.iter().map(|x| 1.0 / x.sqrt()).collect(),
            ));
            let g = givens(0, 1, 1e-4)
                .dot(&givens(1, 2, 1e-4))
                .dot(&givens(0, 2, 1e-4));
            let c = sinv12.dot(&g);
            let outer = |i: usize| {
                let u = c.column(i).to_owned();
                Array2::from_shape_fn((N, N), |(a, b)| u[a] * u[b])
            };
            let db = outer(0);
            let da = &db + &outer(1);
            let t2 = Toy {
                da: cplx(&da),
                db: cplx(&db),
                fa: cplx(&fa),
                fb: cplx(&fb),
                s: cplx(&Array2::from_diag(&ndarray::Array1::from_vec(sd))),
                truth: [0.0; 3],
                mos: cplx(&sinv12),
            };
            let good = new_gate(&t2, None).unwrap();
            let bad = new_gate(&t2, Some(GateMutation::WrongS));
            eprintln!("wrong S: correct {good:.4e}, mutant {bad:?}");
            assert!(good > 1e-6);
            assert!(bad.is_err() || (bad.unwrap() - good).abs() > 0.05 * good);
        }
        // The old measurement under-reports on the same state.
        let old = old_gate(&t).unwrap();
        let truth = t.truth.iter().cloned().fold(0.0_f64, f64::max);
        eprintln!("old gate {old:.4e} vs true {truth:.4e}");
        assert!(
            old < 0.5 * truth,
            "the old gate no longer under-reports: {old:e} vs {truth:e}"
        );
    }

    /// Non-orthogonal AO basis with a COMPLEX Hermitian overlap and an
    /// INDEPENDENT reference: the orbitals are built as `c = S^(-1/2) G` (G the
    /// real rotated orthonormal set of the toy), the AO Focks as
    /// `F = S^(1/2) F_o S^(1/2)` and the densities as `c c†`, so the true blocks
    /// are those of the toy in `G` (no use of the gate's formulas); the gate is
    /// handed a DIFFERENT S-orthonormal basis `mos = S^(-1/2) V` (V a unitary
    /// with phases). The gate must reproduce the reference; a wrong metric on
    /// the density side (`WrongS`) and `S` wrongly placed on the Fock side
    /// (`(S C)† F (S C)`, `WrongFockS`) must each be caught.
    #[test]
    fn complex_overlap_gate_matches_an_independent_reference() {
        let rots = [(0, 1, 1e-4), (1, 2, 2e-4), (0, 2, 3e-4)];
        let t = toy(&rots);
        let truth = t.truth.iter().cloned().fold(0.0_f64, f64::max);
        let mut s = Array2::<C64>::eye(N);
        for i in 0..N {
            for j in i + 1..N {
                let ph = 0.3 * ((i + 1) * (j + 1)) as f64;
                let v = C64::from_polar(0.12 / (1.0 + (j - i) as f64), ph);
                s[(i, j)] = v;
                s[(j, i)] = v.conj();
            }
        }
        let (w, u) = crate::kscf::eigh_herm(&s).unwrap();
        assert!(w.iter().all(|&x| x > 0.3), "{w:?}");
        let fun = |f: &dyn Fn(f64) -> f64| -> Array2<C64> {
            let mut d = Array2::<C64>::zeros((N, N));
            for i in 0..N {
                d[(i, i)] = C64::new(f(w[i]), 0.0);
            }
            u.dot(&d).dot(&u.t().mapv(|z| z.conj()))
        };
        let sh = fun(&|x| x.sqrt());
        let sih = fun(&|x| 1.0 / x.sqrt());
        let da = sih.dot(&t.da).dot(&sih);
        let db = sih.dot(&t.db).dot(&sih);
        let fa = sh.dot(&t.fa).dot(&sh);
        let fb = sh.dot(&t.fb).dot(&sh);
        let mut v = cplx(&givens(0, 3, 0.7).dot(&givens(2, 5, 1.1)));
        for j in 0..N {
            let ph = C64::from_polar(1.0, 0.4 * j as f64);
            for i in 0..N {
                v[(i, j)] *= ph;
            }
        }
        let mos = sih.dot(&v);
        let t2 = Toy {
            da,
            db,
            fa,
            fb,
            s,
            truth: t.truth,
            mos,
        };
        let g = new_gate(&t2, None).unwrap();
        eprintln!("complex S: gate {g:.6e}, independent reference {truth:.6e}");
        assert!(((g - truth) / truth).abs() < 1e-8, "{g:e} vs {truth:e}");
        for m in [GateMutation::WrongS, GateMutation::WrongFockS] {
            let r = new_gate(&t2, Some(m));
            eprintln!("complex S, {m:?}: {r:?}");
            assert!(
                r.is_err() || (r.unwrap() - truth).abs() > 0.05 * truth,
                "{m:?} survives a non-orthogonal complex S"
            );
        }
    }

    /// A NaN Fock element is reported as NaN (not swallowed by `f64::max`).
    #[test]
    fn nan_fock_element_propagates() {
        let mut t = toy(&[(0, 1, 1e-4)]);
        t.fb[(0, 1)] = C64::new(f64::NAN, 0.0);
        let g = new_gate(&t, None).unwrap();
        assert!(g.is_nan(), "{g}");
    }

    /// The gate does not depend on which orthonormal basis of the space the
    /// MOs form (a unitary rotation of `mos` leaves it unchanged).
    #[test]
    fn gate_is_basis_independent() {
        let mut t = toy(&[(0, 1, 1e-4), (1, 2, 1e-4), (0, 2, 1e-4)]);
        let g0 = new_gate(&t, None).unwrap();
        let u = givens(0, 3, 0.7)
            .dot(&givens(1, 4, 0.4))
            .dot(&givens(2, 5, 1.1));
        t.mos = t.mos.dot(&cplx(&u));
        let g1 = new_gate(&t, None).unwrap();
        assert!(((g0 - g1) / g0).abs() < 1e-10, "{g0:e} vs {g1:e}");
    }
}
