//! Analytic nuclear forces of the k-point open-shell KS-DFT energy
//! ([`crate::kuks::solve_kuks`]) for LDA, GGA and GLOBAL HYBRID functionals
//! (exact-exchange fraction `a`, `0 ≤ a ≤ 1`), dense-AFT and RS-GDF J/K.
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
//! # Fock guard
//!
//! `F_α`, `F_β` are rebuilt from the densities and compared with the SCF's
//! stored (non-extrapolated) `fock_alpha`, `fock_beta` against
//! [`crate::kgrad::FOCK_CONSISTENCY_BAR`], with no Madelung-shift allowance
//! (the shift is rebuilt from [`KRksGradConfig::exxdiv`]). Another functional,
//! grid, J/K route or `exxdiv` is a typed refusal.
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
//! tolerance remains an orbital-gradient refusal threshold. Calibration (force
//! error against the reported `g`, H3 doublet ROKS PBE, gate off): `g` 6.9e-3 ->
//! 4.4e-3, 2.8e-4 -> 3.2e-5, 9.3e-7 -> 6.9e-7, 7.5e-9 -> 4.9e-9, i.e. at most
//! about 0.74 `g` (and at least 0.11 `g`), so the tolerance 1e-5 caps the force
//! error near 7e-6 Ha/Bohr. The tolerance is unchanged but a state that passed
//! the earlier gate may now be refused. The MOs must be one set per k-point (a
//! short set is a typed error). ROHF/ROKS are anchored on the dense-AFT J/K
//! only; an RS-GDF source is refused.
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
//! LDA, GGA and global hybrids. Range-separated hybrids, meta-GGA, VV10 and
//! double hybrids are refused with the typed errors of [`crate::kxcgrad`]
//! (`resolve_global`).

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::dft::{build_response_grid, BlochAoHess, BlochPhase, LatticeAoHess, PeriodicGrid};
use crate::grad::ao_atoms;
use crate::grad::RO_ORBITAL_GRADIENT_TOL;
use crate::hcore::kpoint::PeriodicHcoreK;
use crate::hcore::PeriodicHcoreConfig;
use crate::kgrad::{assemble, check_fock, check_inputs, jk_build, open_ledger, KGradJk, KSpins};
use crate::kpts::KPointMesh;
use crate::kroks::KRoScfResult;
use crate::kuscf::KUScfResult;
use crate::kxcgrad::{
    chunk_density, conj, hermitize, resolve_global, KRksGradConfig, KRksGradient, KXcGrid,
    KXcMutation, AO_PLANES, CHUNK_TARGET_BYTES, KS_REMEDY, MAX_CHUNK, MIN_CHUNK,
};
use crate::lattice::Cell;
use ferric_core::FerricError;
use ferric_dft::density_on_grid::UksDensityGrid;
use ferric_dft::grid::GridPoint;
use ferric_dft::libxc::{xc_def_from_name_nspin, FunctionalFamily, XcDef};
use ferric_dft::vxc::{polarized_kernel, DENSITY_FLOOR};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array2, Array3, Axis};
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
    /// that it moves the converged force by < 1e-8 and do not exercise it off
    /// stationarity.
    RoWUhfForm,
    /// ROHF/ROKS: `W = D_α F_α D_α` (the `D_α F_β D_β` cross term dropped).
    RoWDropCross,
    /// ROHF/ROKS: the orbital-gradient gate skipped. An identity at a
    /// converged state; on a loosely converged state it lets the entry point
    /// return a force that is wrong by an amount the tests assert exceeds the
    /// FD bar (`tests/pbc_kuxcgrad_ro.rs`).
    RoGateOff,
    /// UKS: the stationarity gate (`max |F D S − S D F|`) skipped.
    StationarityGateOff,
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
    fa: &'a [Array2<C64>],
    fb: &'a [Array2<C64>],
    converged: bool,
    kpts: &'a [[f64; 3]],
    /// ROHF/ROKS: the one orthonormal MO set per k.
    mos: Option<&'a [Array2<C64>]>,
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

/// One spin's gated kernel vectors on a chunk: `v_ρσ` and `c^σ_i`.
struct SpinKernel {
    vr: Vec<f64>,
    c: Array2<f64>,
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
        let inv_nk = 1.0 / nk as f64;
        let use_gga = self.gga && mutation != Some(KXcMutation::NoGga);
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
            let kern = polarized_kernel(&dens, None, &self.xc);
            let mut e = 0.0;
            let mut ne = 0.0;
            for g in 0..np {
                let rt = dens.rho_a[g] + dens.rho_b[g];
                e += pts[g].weight * rt * kern.exc[g];
                ne += pts[g].weight * rt;
            }
            // Gated kernel vectors per spin (the energy's gates: each spin on
            // its own density). `cross_other` selects which gradient feeds the
            // v_σαβ term.
            let spin_kernel = |s: usize, cross_mode: u8| -> SpinKernel {
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
                let vr = (0..np)
                    .map(|g| if rho[g] > DENSITY_FLOOR { vrho[g] } else { 0.0 })
                    .collect();
                let c = Array2::from_shape_fn((3, np), |(i, g)| {
                    if self.gga && rho[g] > DENSITY_FLOOR {
                        let cross = match cross_mode {
                            0 => kern.vsigma_ab[g] * gother[(i, g)],
                            1 => 0.0,
                            _ => kern.vsigma_ab[g] * gself[(i, g)],
                        };
                        2.0 * vss[g] * gself[(i, g)] + cross
                    } else {
                        0.0
                    }
                });
                SpinKernel { vr, c }
            };
            let ka = spin_kernel(0, 0);
            let kb = spin_kernel(1, 0);
            // --- V_xc,σ(k) with the TRUE kernel vectors.
            let mut vout: [Vec<Array2<C64>>; 2] = [Vec::with_capacity(nk), Vec::with_capacity(nk)];
            for (s, sk) in [&ka, &kb].into_iter().enumerate() {
                let a: Vec<f64> = (0..np).map(|g| pts[g].weight * sk.vr[g]).collect();
                for k in 0..nk {
                    let scale = |fac: &[f64]| {
                        let mut m = conj(&bl.chi[k]);
                        for g in 0..np {
                            m.column_mut(g).mapv_inplace(|z| z * fac[g]);
                        }
                        m
                    };
                    let mut vk = scale(&a).dot(&bl.chi[k].t());
                    if self.gga {
                        for ax in 0..3 {
                            let f: Vec<f64> =
                                (0..np).map(|g| pts[g].weight * sk.c[(ax, g)]).collect();
                            let d_ax = bl.dchi[k].index_axis(Axis(0), ax);
                            let m = scale(&f).dot(&d_ax.t());
                            let mh = conj(&m).t().to_owned();
                            vk += &m;
                            vk += &mh;
                        }
                    }
                    vout[s].push(vk);
                }
            }
            let [va, vb] = vout;
            // --- Force vectors (possibly mutated).
            let cross_mode = match smut {
                Some(KUSpinMutation::XcCrossDropped) => 1,
                Some(KUSpinMutation::XcCrossOwn) => 2,
                _ => 0,
            };
            let (fka, fkb) = if cross_mode == 0 {
                (spin_kernel(0, 0), spin_kernel(1, 0))
            } else {
                (spin_kernel(0, cross_mode), spin_kernel(1, cross_mode))
            };
            // (spin density used, kernel vectors used) per force pass.
            let passes: Vec<(&[Array2<C64>], SpinKernel)> = match smut {
                Some(KUSpinMutation::XcSpinSwap) => vec![(da, fkb), (db, fka)],
                Some(KUSpinMutation::XcDropBeta) => vec![(da, fka)],
                Some(KUSpinMutation::XcSpinAveraged) => {
                    let avg = SpinKernel {
                        vr: (0..np).map(|g| 0.5 * (fka.vr[g] + fkb.vr[g])).collect(),
                        c: Array2::from_shape_fn((3, np), |(i, g)| {
                            0.5 * (fka.c[(i, g)] + fkb.c[(i, g)])
                        }),
                    };
                    vec![(dsum.as_deref().expect("dsum for XcSpinAveraged"), avg)]
                }
                _ => vec![(da, fka), (db, fkb)],
            };
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
            if mutation != Some(KXcMutation::NoAo) {
                for (dm, sk) in &passes {
                    for k in 0..nk {
                        let p = dm[k].dot(&conj(&fb.chi[k]));
                        let md: Option<[Array2<C64>; 3]> = use_gga.then(|| {
                            std::array::from_fn(|i| {
                                dm[k].dot(&conj(&fb.dchi[k].index_axis(Axis(0), i).to_owned()))
                            })
                        });
                        for g in 0..np {
                            let wg = pref * pts[g].weight * kscale;
                            for mu in 0..nbf {
                                let at = self.aoat[mu];
                                let pm = p[(mu, g)];
                                for j in 0..3 {
                                    let dj = fb.dchi[k][(j, mu, g)];
                                    let mut t = dj * pm * sk.vr[g];
                                    if let Some(md) = md.as_ref() {
                                        for i in 0..3 {
                                            let mut inner = dj * md[i][(mu, g)];
                                            if mutation != Some(KXcMutation::NoHessian) {
                                                inner += fb.ddchi[k][(j, i, mu, g)] * pm;
                                            }
                                            t += inner * sk.c[(i, g)];
                                        }
                                    }
                                    a_pt[(g, at, j)] += wg * t.re;
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

/// The restricted-open-shell forces are anchored on the dense-AFT J/K only.
fn refuse_rsgdf(who: &str, jk: &KGradJk<'_>) -> Result<(), FerricError> {
    if matches!(jk, KGradJk::RsGdf(_)) {
        return Err(FerricError::General(format!(
            "{who}: RS-GDF J/K is not supported by the restricted-open-shell force (no anchor \
             against FD or Gamma exists for it); use the dense-AFT J/K"
        )));
    }
    Ok(())
}

fn grid_response<'a>(
    cell: &'a Cell,
    prep: &'a PreparedBasis,
    mesh: &KPointMesh,
    functional: &str,
    cfg: &'a KUksGradConfig,
) -> Result<(GridXcPol<'a>, f64), FerricError> {
    let (_xc1, exch) = resolve_global(functional)?;
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
    let xc = xc_def_from_name_nspin(&name, 2)
        .map_err(|e| FerricError::General(format!("{functional}: spin-polarized: {e:?}")))?;
    let gga = xc
        .funcs
        .iter()
        .any(|f| !matches!(f.family(), FunctionalFamily::Lda));
    Ok((
        GridXcPol {
            cell,
            prep,
            xc,
            gga,
            kpts: mesh.kpts().to_vec(),
            aoat: ao_atoms(prep),
            grid: &cfg.base.grid,
            ao_threshold: cfg.base.ao_threshold,
            max_live_chunks: cfg.base.max_live_chunks,
            mutation: cfg.base.mutation,
            spin_mutation: cfg.spin_mutation,
        },
        exch,
    ))
}

/// k-point open-shell UKS nuclear gradient (module doc) of a converged
/// [`crate::kuks::solve_kuks`] result (`KUksResult::scf`) for an LDA, GGA or
/// global-hybrid `functional` (a hybrid also needs
/// [`KRksGradConfig::exxdiv`], here `cfg.base.exxdiv`). Arguments as
/// [`crate::kxcgrad::kpoint_rks_gradient`], with the SCF a [`KUScfResult`].
///
/// Refuses (typed errors) range-separated hybrids, meta-GGA, VV10/double
/// hybrids, a hybrid without `cfg.base.exxdiv`, and an SCF whose spin Focks
/// the rebuilt `h + J − a K_σ + V_xc,σ` do not reproduce.
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
    let (resp, exch) = grid_response(cell, prep, mesh, functional, cfg)?;
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
        exch,
        cfg,
    )
}

/// k-point ROKS nuclear gradient of a converged
/// [`crate::kroks::solve_kroks`] result (`KRoksResult::scf`): the open-shell
/// assembly on the ROHF spin densities `D_α, D_β` and spin Focks, with the
/// orthonormality-Lagrangian `W_RO = sym(D_α F_α D_α + D_α F_β D_β)` at each k
/// (module doc "ROHF / ROKS"). Arguments and refusals as
/// [`kpoint_uks_gradient`], plus: a state whose ROHF orbital gradient
/// (closed-open `f_β`, open-virtual `f_α`, closed-virtual `f_α + f_β`, any k)
/// exceeds [`crate::grad::RO_ORBITAL_GRADIENT_TOL`] is refused.
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
    refuse_rsgdf("kpoint_roks_gradient", &jk)?;
    let (resp, exch) = grid_response(cell, prep, mesh, functional, cfg)?;
    uks_gradient_with(
        "kpoint_roks_gradient",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        &ro_input(scf),
        &resp,
        exch,
        cfg,
    )
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
    refuse_rsgdf("kpoint_rohf_gradient", &jk)?;
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
        1.0,
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
        exch,
        cfg,
    )
}

fn uks_input(scf: &KUScfResult) -> SpinInput<'_> {
    SpinInput {
        da: &scf.density_alpha,
        db: &scf.density_beta,
        fa: &scf.fock_alpha,
        fb: &scf.fock_beta,
        converged: scf.converged,
        kpts: &scf.kpts,
        mos: None,
    }
}

fn ro_input(scf: &KRoScfResult) -> SpinInput<'_> {
    SpinInput {
        da: &scf.density_alpha,
        db: &scf.density_beta,
        fa: &scf.fock_alpha,
        fb: &scf.fock_beta,
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
    exch: f64,
    cfg: &KUksGradConfig,
) -> Result<KUksGradient, FerricError> {
    let ro = inp.mos.is_some();
    check_inputs(
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
    )?;
    if !(0.0..=1.0).contains(&exch) {
        return Err(FerricError::General(format!(
            "{who}: exact-exchange fraction {exch} outside [0, 1]"
        )));
    }
    let natoms = cell.positions().len();
    let nk = mesh.nk();
    let vm = if exch == 0.0 {
        0.0
    } else {
        let exxdiv = cfg.base.exxdiv.ok_or_else(|| {
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
        )
    }) && !matches!(
        mutation,
        Some(KXcMutation::PulayNoVxc)
            | Some(KXcMutation::WrongRemainder)
            | Some(KXcMutation::GuardSkippedOnly)
    );
    let mismatch = if guard {
        let ma = check_fock(who, "fock_alpha", &fa, inp.fa, KS_REMEDY)?;
        let mb = check_fock(who, "fock_beta", &fb, inp.fb, KS_REMEDY)?;
        ma.max(mb)
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
            let gate_on = smut != Some(KUSpinMutation::RoGateOff);
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
    let spins = if !ro || smut == Some(KUSpinMutation::RoWUhfForm) {
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
