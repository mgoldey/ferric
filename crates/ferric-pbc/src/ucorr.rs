//! Stage 8: Gamma-point open-shell MP2 (UMP2) and direct RPA (URPA) on a
//! Gamma UHF reference ([`crate::uhf::gamma_uhf`]). Rust port of
//! `reference/pbc/pbc_ump2.py`, FINDINGS "Iteration 7 (Python, Gamma
//! UMP2/URPA)" and its "For the Rust port (stage 8)" section.
//!
//! ```text
//! E_aa = ½ Σ_{ijab∈α} (ia|jb)[(ia|jb) − (ib|ja)] / D      (E_bb likewise)
//! E_ab =   Σ_{ia∈α, JB∈β} (ia|JB)² / D                     (no exchange)
//! Π(iω) = Σ_σ 2 B_σ diag(e_ia,σ / (ω² + e_ia,σ²)) B_σᵀ      (per-spin factor 2)
//! E_c^URPA = (1/2π) ∫_0^∞ dω { ln det[1 + Π(iω)] − tr Π(iω) }
//! ```
//!
//! Real arithmetic (Gamma orbitals are real). The per-spin `B^P_{ia,σ}` are
//! the α and β MO transforms of ONE periodic `B[k, μν]` (the SCF's own
//! RS-GDF tensor, shared metric), so `(ia|JB) = Σ_k B^k_{ia,α} B^k_{JB,β}`.
//! Closed shell (`C_α = C_β`, `ε_α = ε_β`) reduces to the restricted
//! [`crate::mp2::gamma_mp2`] / [`crate::drpa::gamma_drpa`] (factor 4 = 2
//! spins × 2).
//!
//! # Integral sources (the Stage 6/7 enums, reused)
//!
//! * `RsGdf` — production. `B_σ = b_ov_from_ao_b(B, C_σ,occ, C_σ,vir)`
//!   (naux = the KEPT count, already metric-dressed, no further `V^{-1/2}`),
//!   packed as two [`RpaIntermediates`], then
//!   - UMP2: ferric-mp2's [`u_ri_mp2_from_parts`] (the molecular
//!     `same_spin_pair_energy` ×2 and `opposite_spin_pair_energy` kernels);
//!   - URPA: ferric-rpa's [`run_u_pdep_rpa_from_parts`] (the molecular
//!     `run_u_pdep_rpa` post-intermediate body: spin-summed ε̃(0), full-rank
//!     Lanczos eigensolve, the same frequency evaluator and integrator).
//! * `DenseAft` — TEST/ORACLE ONLY. Exact spin blocks `(ia|jb)_αα`, `_ββ`,
//!   `(ia|JB)_αβ` from the dense pure-AFT tensor ([`ovov_from_dense_pair`]),
//!   then an INDEPENDENT energy loop ([`ump2_from_ovov`]) and the
//!   spin-orbital direct-RPA PLASMON formula on the joint α+β ov space
//!   ([`urpa_plasmon`]) — no aux, no quadrature, no eigensolve of ε̃. A
//!   different ERI, a different MO transform and a different energy
//!   construction from the B path: that is what makes the trivial-aux anchor
//!   (`tests/pbc_ucorr.rs`) a test of construction.
//!
//! # Denominators (per spin, the SAME v_M)
//!
//! `exxdiv = ewald` (`K_σ += v_M S D_σ S`) leaves every C_σ unchanged and
//! lowers EVERY spin's occupied levels by `v_M` (coefficient 1 per spin; the
//! UHF module doc). The Stage 6 table ([`crate::mp2::occupied_shift`]) is
//! therefore applied with the same `v_M` to the α AND β occupied energies;
//! the virtuals are never shifted. The prototype established this by the
//! box limit, not by assumption: only the same `v_M` on both spins gives the
//! a⁻³ residual with the molecularly predicted coefficient, while `v_M/2` per
//! spin or an α-only shift leave a flat 1/a plateau (30-60× the shifted
//! residual at a = 40). PySCF agrees (`pbc.mp.UMP2` on an ewald UHF, pbc
//! UCCSD's `_adjust_occ` per spin). The reference exxdiv must be STATED
//! (`reference_exxdiv`; no `Default` on the configs); an `exxdiv = None`
//! reference with `Unshifted` denominators is what the caller must ask for
//! explicitly to get the 1/a convention.
//!
//! # Configs
//!
//! The Stage 6/7 configs are reused unchanged: [`GammaMp2Config`] for UMP2,
//! [`GammaDrpaConfig`] for URPA; `reference_exxdiv` describes the UHF
//! reference, `frozen_core` goes through [`active_occ`] for EACH spin (so a
//! spin with no electron refuses any frozen core).
//!
//! # References accepted
//!
//! `Spin::Unrestricted` only (what `gamma_uhf` returns). `Restricted` is
//! refused (use the Stage 6/7 drivers); `RestrictedOpen` is refused (no ROHF
//! Gamma reference exists and none was measured). An EMPTY spin channel
//! (`N_β = 0`, or a spin with no virtual) is allowed and contributes 0.
//!
//! # Not measured (do not extrapolate)
//!
//! ROHF, frozen core on a periodic cell, > 16 AOs in the dense oracle, real
//! solids (a⁻³ is the isolated-molecule law), k-points, cost, spin
//! contamination beyond ⟨S²⟩ ≈ 2.013.
//!
//! # Memory
//!
//! Every buffer ferric-pbc allocates or hands to ferric-mp2/ferric-rpa is
//! reserved on a [`crate::budget`] ledger against `budget_bytes` before the
//! first allocation; per-worker transients are counted ONCE (a gate must not
//! read the thread count).
//!
//! Units: Bohr and Hartree.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::drpa::{
    pdep_config, validate_quad_points, GammaDrpaConfig, GammaDrpaIntegrals, GAMMA_DRPA_QUAD_U0,
    MAX_GAMMA_DRPA_QUAD_POINTS,
};
use crate::ewald::madelung_constant;
use crate::lattice::Cell;
use crate::mp2::{
    b_ov_from_ao_b, occupied_shift, GammaMp2Config, GammaMp2Integrals, Mp2Denominators,
};
use crate::uhf::nocc_ab;
use ferric_core::FerricError;
use ferric_mp2::rimp2::{active_occ, RpaIntermediates};
use ferric_mp2::u_rimp2::u_ri_mp2_from_parts;
pub use ferric_mp2::u_rimp2::URiMp2Components;
use ferric_rpa::run_u_pdep_rpa_from_parts;
use ferric_scf::result::{ScfResult, Spin};
use ndarray::{s, Array2, ArrayView2};
use ndarray_linalg::{Eigh, UPLO};
use std::f64::consts::PI;

/// Gamma-point UMP2 result.
#[derive(Debug, Clone)]
#[must_use]
pub struct GammaUmp2Result {
    /// αα / ββ / αβ split (`e_total` = the correlation energy).
    pub components: URiMp2Components,
    /// UMP2 correlation energy (= `components.e_total`).
    pub mp2_corr: f64,
    /// `uhf.energy + mp2_corr` (the UHF energy is the reference's own; only
    /// the correlation part is made convention-consistent here).
    pub total_energy: f64,
    /// Gamma-point Madelung constant `v_M` of the cell.
    pub madelung: f64,
    /// Shift added to every active occupied ε of BOTH spins (0, −v_M, +v_M).
    pub occ_shift: f64,
    /// Correlated occupied counts `[α, β]`.
    pub nocc_active: [usize; 2],
    /// Virtual counts `[α, β]`.
    pub nvir: [usize; 2],
    /// Rows of the fitted B (`None` for the dense oracle).
    pub naux: Option<usize>,
}

/// Gamma-point URPA result.
#[derive(Debug, Clone)]
#[must_use]
pub struct GammaUrpaResult {
    /// Direct-RPA correlation energy.
    pub rpa_corr: f64,
    /// `uhf.energy + rpa_corr`.
    pub total_energy: f64,
    /// Gamma-point Madelung constant `v_M` of the cell.
    pub madelung: f64,
    /// Shift added to every active occupied ε of BOTH spins.
    pub occ_shift: f64,
    /// Correlated occupied counts `[α, β]`.
    pub nocc_active: [usize; 2],
    /// Virtual counts `[α, β]`.
    pub nvir: [usize; 2],
    /// Rows of the fitted B (`None` for the plasmon oracle).
    pub naux: Option<usize>,
    /// Frequency points used (`None` for the plasmon oracle).
    pub quad_points: Option<usize>,
    /// Static dielectric eigenvalues λ(0) of `I + Π_α + Π_β` (`None` for the
    /// plasmon oracle).
    pub eigenvalues_static: Option<Vec<f64>>,
}

/// Per-spin orbital spaces of a Gamma UHF reference, with the occupied
/// energies already shifted.
struct USpaces<'r> {
    nao: usize,
    /// First correlated occupied orbital (= frozen core), both spins.
    first: usize,
    /// Total occupied `[α, β]`.
    ntot: [usize; 2],
    /// Active occupied `[α, β]`.
    nact: [usize; 2],
    /// Virtuals `[α, β]`.
    nvir: [usize; 2],
    c: [&'r Array2<f64>; 2],
    /// Full orbital energies per spin, occupied part shifted by `occ_shift`.
    eps: [Vec<f64>; 2],
    madelung: f64,
    occ_shift: f64,
}

impl USpaces<'_> {
    fn c_occ(&self, s: usize) -> ArrayView2<'_, f64> {
        self.c[s].slice(s![.., self.first..self.ntot[s]])
    }
    fn c_vir(&self, s: usize) -> ArrayView2<'_, f64> {
        self.c[s].slice(s![.., self.ntot[s]..])
    }
    fn eps_occ(&self, s: usize) -> &[f64] {
        &self.eps[s][self.first..self.ntot[s]]
    }
    fn eps_vir(&self, s: usize) -> &[f64] {
        &self.eps[s][self.ntot[s]..]
    }
    fn nov(&self, s: usize) -> usize {
        self.nact[s] * self.nvir[s]
    }
    fn live(&self, s: usize) -> bool {
        self.nov(s) > 0
    }
}

fn spaces<'r>(
    who: &str,
    cell: &Cell,
    uhf: &'r ScfResult,
    frozen_core: usize,
    reference_exxdiv: ExxDiv,
    denominators: Mp2Denominators,
) -> Result<USpaces<'r>, FerricError> {
    match uhf.spin {
        Spin::Unrestricted => {}
        Spin::Restricted => {
            return Err(FerricError::General(format!(
                "{who}: restricted reference; use the closed-shell Gamma driver (gamma_mp2 / \
                 gamma_drpa)"
            )))
        }
        Spin::RestrictedOpen => {
            return Err(FerricError::General(format!(
                "{who}: ROHF reference not supported (no Gamma ROHF exists; unmeasured)"
            )))
        }
    }
    if !uhf.converged {
        return Err(FerricError::General(format!(
            "{who}: the UHF reference did not converge"
        )));
    }
    let (na, nb) = nocc_ab(cell.mol())?;
    let ca = &uhf.mos_alpha;
    let cb = uhf
        .mos_beta
        .as_ref()
        .ok_or_else(|| FerricError::General(format!("{who}: UHF result has no beta MOs")))?;
    let eb_in = uhf.eps_beta.as_deref().ok_or_else(|| {
        FerricError::General(format!("{who}: UHF result has no beta orbital energies"))
    })?;
    let ea_in = uhf.eps_alpha.as_slice();
    let (nao, nmo) = ca.dim();
    if cb.dim() != (nao, nmo) || ea_in.len() != nmo || eb_in.len() != nmo || nmo < na.max(nb) {
        return Err(FerricError::General(format!(
            "{who}: inconsistent UHF reference (C_α {:?}, C_β {:?}, {} / {} orbital energies, \
             N_α {na}, N_β {nb})",
            ca.dim(),
            cb.dim(),
            ea_in.len(),
            eb_in.len()
        )));
    }
    let nact = [active_occ(na, frozen_core)?, active_occ(nb, frozen_core)?];
    let madelung = madelung_constant(cell)?;
    let occ_shift = occupied_shift(reference_exxdiv, denominators, madelung);
    let shifted = |e: &[f64], n: usize| -> Vec<f64> {
        e.iter()
            .enumerate()
            .map(|(p, &x)| if p < n { x + occ_shift } else { x })
            .collect()
    };
    Ok(USpaces {
        nao,
        first: frozen_core,
        ntot: [na, nb],
        nact,
        nvir: [nmo - na, nmo - nb],
        c: [ca, cb],
        eps: [shifted(ea_in, na), shifted(eb_in, nb)],
        madelung,
        occ_shift,
    })
}

/// `B[k, ia]` for one spin; a zero-width `(naux, 0)` block for an empty
/// channel (no zero-size GEMM is formed).
fn b_ov_spin(b: &Array2<f64>, sp: &USpaces<'_>, s: usize) -> Result<Array2<f64>, FerricError> {
    if !sp.live(s) {
        return Ok(Array2::zeros((b.nrows(), 0)));
    }
    b_ov_from_ao_b(b, sp.nao, sp.c_occ(s), sp.c_vir(s))
}

fn intermediates(
    b_ov: Array2<f64>,
    v_inv_sqrt: Array2<f64>,
    sp: &USpaces<'_>,
    s: usize,
) -> RpaIntermediates {
    let naux = b_ov.nrows();
    RpaIntermediates {
        b_ov,
        v_inv_sqrt,
        nocc: sp.nact[s],
        nvir: sp.nvir[s],
        nocc_total: sp.ntot[s],
        first_occ: sp.first,
        naux,
    }
}

/// `e_ia` in `i·nvir + a` order; every entry must be finite and > 0.
pub fn excitation_energies(eps_occ: &[f64], eps_vir: &[f64]) -> Result<Vec<f64>, FerricError> {
    let mut e = Vec::with_capacity(eps_occ.len() * eps_vir.len());
    for &ei in eps_occ {
        for &ea in eps_vir {
            let d = ea - ei;
            if !(d.is_finite() && d > 0.0) {
                return Err(FerricError::General(format!(
                    "Gamma URPA: non-positive or non-finite e_ia = {d} (ε_i {ei}, ε_a {ea}); the \
                     frequency integral is undefined"
                )));
            }
            e.push(d);
        }
    }
    Ok(e)
}

/// Exact `(ia|jb)` between two orbital-pair spaces as an `(nocc₁·nvir₁,
/// nocc₂·nvir₂)` matrix from a dense `(nao², nao²)` ERI: `W₁ᵀ I W₂`,
/// `W[μ·nao+ν, i·nvir+a] = C_o[μ,i] C_v[ν,a]`. Same-spin blocks pass the same
/// spaces twice; the αβ block passes α then β. TEST/ORACLE scale only. An
/// empty space gives a zero-size matrix without any GEMM.
pub fn ovov_from_dense_pair(
    eri: &Array2<f64>,
    nao: usize,
    c_occ1: ArrayView2<'_, f64>,
    c_vir1: ArrayView2<'_, f64>,
    c_occ2: ArrayView2<'_, f64>,
    c_vir2: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, FerricError> {
    let n2 = nao * nao;
    if eri.dim() != (n2, n2)
        || [
            c_occ1.nrows(),
            c_vir1.nrows(),
            c_occ2.nrows(),
            c_vir2.nrows(),
        ]
        .iter()
        .any(|&r| r != nao)
    {
        return Err(FerricError::General(format!(
            "ovov_from_dense_pair: ERI {:?} or C rows inconsistent with nao = {nao}",
            eri.dim()
        )));
    }
    let nov1 = c_occ1.ncols() * c_vir1.ncols();
    let nov2 = c_occ2.ncols() * c_vir2.ncols();
    if nov1 == 0 || nov2 == 0 {
        return Ok(Array2::zeros((nov1, nov2)));
    }
    let w = |co: ArrayView2<'_, f64>, cv: ArrayView2<'_, f64>| {
        let (no, nv) = (co.ncols(), cv.ncols());
        let mut w = Array2::<f64>::zeros((n2, no * nv));
        for m in 0..nao {
            for n in 0..nao {
                for i in 0..no {
                    for a in 0..nv {
                        w[(m * nao + n, i * nv + a)] = co[(m, i)] * cv[(n, a)];
                    }
                }
            }
        }
        w
    };
    let w1 = w(c_occ1, c_vir1);
    let w2 = w(c_occ2, c_vir2);
    let iw2 = eri.dot(&w2);
    Ok(w1.t().dot(&iw2))
}

/// The three exact spin blocks `(aa, bb, ab)` of `(ia|jb)` for the active
/// spaces of `sp` from a dense ERI.
fn dense_blocks(
    eri: &Array2<f64>,
    sp: &USpaces<'_>,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), FerricError> {
    let blk = |s: usize, t: usize| {
        ovov_from_dense_pair(
            eri,
            sp.nao,
            sp.c_occ(s),
            sp.c_vir(s),
            sp.c_occ(t),
            sp.c_vir(t),
        )
    };
    Ok((blk(0, 0)?, blk(1, 1)?, blk(0, 1)?))
}

fn check_block(
    who: &str,
    x: &Array2<f64>,
    rows: usize,
    cols: usize,
    what: &str,
) -> Result<(), FerricError> {
    if x.dim() != (rows, cols) {
        return Err(FerricError::General(format!(
            "{who}: {what} block is {:?}, expected ({rows}, {cols})",
            x.dim()
        )));
    }
    Ok(())
}

/// Symmetrised element (a dense `Wᵀ I W` is symmetric only to roundoff).
#[inline]
fn sym(x: &Array2<f64>, p: usize, q: usize) -> f64 {
    0.5 * (x[(p, q)] + x[(q, p)])
}

/// TEST/ORACLE: UMP2 from exact spin blocks (rows/columns `i·nvir + a`) and
/// the ACTIVE per-spin energies — an energy loop independent of ferric-mp2's
/// kernels: `E_σσ = ½ Σ x_iajb (x_iajb − x_ibja) / D` (symmetrised x, so a
/// one-occupied spin gives exactly 0), `E_ab = Σ x_iaJB² / D`.
#[allow(clippy::too_many_arguments)]
pub fn ump2_from_ovov(
    ov_aa: &Array2<f64>,
    ov_bb: &Array2<f64>,
    ov_ab: &Array2<f64>,
    eo_a: &[f64],
    ev_a: &[f64],
    eo_b: &[f64],
    ev_b: &[f64],
) -> Result<URiMp2Components, FerricError> {
    let (nova, novb) = (eo_a.len() * ev_a.len(), eo_b.len() * ev_b.len());
    check_block("ump2_from_ovov", ov_aa, nova, nova, "aa")?;
    check_block("ump2_from_ovov", ov_bb, novb, novb, "bb")?;
    check_block("ump2_from_ovov", ov_ab, nova, novb, "ab")?;
    let same = |x: &Array2<f64>, eo: &[f64], ev: &[f64]| {
        let nv = ev.len();
        let mut e = 0.0;
        for i in 0..eo.len() {
            for j in 0..eo.len() {
                for a in 0..nv {
                    for b in 0..nv {
                        let xab = sym(x, i * nv + a, j * nv + b);
                        let xba = sym(x, i * nv + b, j * nv + a);
                        e += 0.5 * xab * (xab - xba) / (eo[i] + eo[j] - ev[a] - ev[b]);
                    }
                }
            }
        }
        e
    };
    let e_aa = same(ov_aa, eo_a, ev_a);
    let e_bb = same(ov_bb, eo_b, ev_b);
    let (nva, nvb) = (ev_a.len(), ev_b.len());
    let mut e_ab = 0.0;
    for i in 0..eo_a.len() {
        for a in 0..nva {
            for j in 0..eo_b.len() {
                for b in 0..nvb {
                    let x = ov_ab[(i * nva + a, j * nvb + b)];
                    e_ab += x * x / (eo_a[i] + eo_b[j] - ev_a[a] - ev_b[b]);
                }
            }
        }
    }
    Ok(URiMp2Components {
        e_aa,
        e_bb,
        e_ab,
        e_total: e_aa + e_bb + e_ab,
    })
}

/// TEST/ORACLE: DIRECT (Coulomb-only, ring) second order — the O(Π²) term of
/// URPA: `½ Σ_αα x²/D + ½ Σ_ββ x²/D + Σ_αβ x²/D`.
#[allow(clippy::too_many_arguments)]
pub fn direct_ump2_from_ovov(
    ov_aa: &Array2<f64>,
    ov_bb: &Array2<f64>,
    ov_ab: &Array2<f64>,
    eo_a: &[f64],
    ev_a: &[f64],
    eo_b: &[f64],
    ev_b: &[f64],
) -> Result<f64, FerricError> {
    let (nova, novb) = (eo_a.len() * ev_a.len(), eo_b.len() * ev_b.len());
    check_block("direct_ump2_from_ovov", ov_aa, nova, nova, "aa")?;
    check_block("direct_ump2_from_ovov", ov_bb, novb, novb, "bb")?;
    check_block("direct_ump2_from_ovov", ov_ab, nova, novb, "ab")?;
    let ring = |x: &Array2<f64>, eo1: &[f64], ev1: &[f64], eo2: &[f64], ev2: &[f64], sy: bool| {
        let (nv1, nv2) = (ev1.len(), ev2.len());
        let mut e = 0.0;
        for i in 0..eo1.len() {
            for a in 0..nv1 {
                for j in 0..eo2.len() {
                    for b in 0..nv2 {
                        let (p, q) = (i * nv1 + a, j * nv2 + b);
                        let v = if sy { sym(x, p, q) } else { x[(p, q)] };
                        e += v * v / (eo1[i] + eo2[j] - ev1[a] - ev2[b]);
                    }
                }
            }
        }
        e
    };
    Ok(0.5 * ring(ov_aa, eo_a, ev_a, eo_a, ev_a, true)
        + 0.5 * ring(ov_bb, eo_b, ev_b, eo_b, ev_b, true)
        + ring(ov_ab, eo_a, ev_a, eo_b, ev_b, false))
}

/// TEST/ORACLE: spin-orbital direct RPA by the plasmon formula on the joint
/// α+β ov space, `A = D + K`, `B = K`, `K = [[aa, ab], [abᵀ, bb]]`:
/// `Ω² = eig[D^{1/2} (D + 2K) D^{1/2}]`, `E_c = ½ (Σ Ω − tr A)`. No aux, no
/// quadrature. Errors on a non-positive Ω² (instability) or `e_ia <= 0`.
#[allow(clippy::too_many_arguments)]
pub fn urpa_plasmon(
    ov_aa: &Array2<f64>,
    ov_bb: &Array2<f64>,
    ov_ab: &Array2<f64>,
    eo_a: &[f64],
    ev_a: &[f64],
    eo_b: &[f64],
    ev_b: &[f64],
) -> Result<f64, FerricError> {
    let mut e = excitation_energies(eo_a, ev_a)?;
    let nova = e.len();
    e.extend(excitation_energies(eo_b, ev_b)?);
    let n = e.len();
    let novb = n - nova;
    check_block("urpa_plasmon", ov_aa, nova, nova, "aa")?;
    check_block("urpa_plasmon", ov_bb, novb, novb, "bb")?;
    check_block("urpa_plasmon", ov_ab, nova, novb, "ab")?;
    if n == 0 {
        return Err(FerricError::General(
            "urpa_plasmon: no occupied-virtual pair in either spin".into(),
        ));
    }
    let kval = |p: usize, q: usize| -> f64 {
        match (p < nova, q < nova) {
            (true, true) => sym(ov_aa, p, q),
            (false, false) => sym(ov_bb, p - nova, q - nova),
            (true, false) => ov_ab[(p, q - nova)],
            (false, true) => ov_ab[(q, p - nova)],
        }
    };
    let sd: Vec<f64> = e.iter().map(|x| x.sqrt()).collect();
    let mut m = Array2::<f64>::zeros((n, n));
    let mut tr_a = 0.0;
    for p in 0..n {
        for q in 0..n {
            m[(p, q)] = 2.0 * sd[p] * kval(p, q) * sd[q];
        }
        m[(p, p)] += e[p] * e[p];
        tr_a += e[p] + kval(p, p);
    }
    let (lam, _) = m
        .eigh(UPLO::Upper)
        .map_err(|err| FerricError::Lapack(format!("urpa_plasmon eigh: {err}")))?;
    let lmin = lam.iter().copied().fold(f64::INFINITY, f64::min);
    if !(lmin > 0.0) {
        return Err(FerricError::General(format!(
            "urpa_plasmon: URPA instability (smallest Ω² = {lmin:e})"
        )));
    }
    let sum_omega: f64 = lam.iter().map(|x| x.sqrt()).sum();
    Ok(0.5 * (sum_omega - tr_a))
}

/// TEST/ORACLE: `−(1/2π) ∫_0^∞ dω tr Π(iω)² / 2` with the per-spin factor 2
/// (`Π = Σ_σ 2 B_σ diag(e/(ω²+e²)) B_σᵀ`) on the Gauss–Legendre grid of the
/// production path (`u0 = 0.5`); analytically DIRECT UMP2
/// ([`direct_ump2_from_ovov`]). Written independently of ferric-rpa (only
/// the grid is shared); dense, test scale.
#[allow(clippy::too_many_arguments)]
pub fn urpa_second_order_from_b_ov(
    b_a: &Array2<f64>,
    b_b: &Array2<f64>,
    eo_a: &[f64],
    ev_a: &[f64],
    eo_b: &[f64],
    ev_b: &[f64],
    quad_points: usize,
) -> Result<f64, FerricError> {
    if !(1..=4 * MAX_GAMMA_DRPA_QUAD_POINTS).contains(&quad_points) {
        return Err(FerricError::General(format!(
            "urpa_second_order_from_b_ov: quad_points {quad_points} out of range"
        )));
    }
    let ea = excitation_energies(eo_a, ev_a)?;
    let eb = excitation_energies(eo_b, ev_b)?;
    let naux = b_a.nrows();
    if b_a.dim() != (naux, ea.len()) || b_b.dim() != (naux, eb.len()) {
        return Err(FerricError::General(format!(
            "urpa_second_order_from_b_ov: B_α {:?} / B_β {:?} inconsistent with {} / {} pairs \
             and one shared aux index",
            b_a.dim(),
            b_b.dim(),
            ea.len(),
            eb.len()
        )));
    }
    let nov = ea.len() + eb.len();
    let (freqs, weights) =
        ferric_rpa::quadrature::gauss_legendre_nodes(quad_points, GAMMA_DRPA_QUAD_U0);
    let mut total = 0.0_f64;
    for (&w, &wk) in freqs.iter().zip(&weights) {
        let mut bs = Array2::<f64>::zeros((naux, nov));
        for (col0, b, e) in [(0usize, b_a, &ea), (ea.len(), b_b, &eb)] {
            for (p, &ep) in e.iter().enumerate() {
                let f = (2.0 * ep / (w * w + ep * ep)).sqrt();
                for k in 0..naux {
                    bs[(k, col0 + p)] = f * b[(k, p)];
                }
            }
        }
        let g = if naux <= nov {
            bs.dot(&bs.t())
        } else {
            bs.t().dot(&bs)
        };
        let tr_pi2: f64 = g.iter().map(|x| x * x).sum();
        total += wk * (-0.5 * tr_pi2);
    }
    Ok(total / (2.0 * PI))
}

/// Gamma-point UMP2 from a converged Gamma UHF `uhf` (from
/// [`crate::uhf::gamma_uhf`]) on `cell`. `cfg.reference_exxdiv` is the
/// exxdiv the UHF was converged with. See the module doc.
pub fn gamma_ump2(
    cell: &Cell,
    uhf: &ScfResult,
    ints: GammaMp2Integrals<'_>,
    cfg: &GammaMp2Config,
) -> Result<GammaUmp2Result, FerricError> {
    let sp = spaces(
        "gamma_ump2",
        cell,
        uhf,
        cfg.frozen_core,
        cfg.reference_exxdiv,
        cfg.denominators,
    )?;
    let nao = sp.nao;
    let (nova, novb) = (sp.nov(0), sp.nov(1));
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));

    let (components, naux) = match ints {
        GammaMp2Integrals::RsGdf(gdf) => {
            let b = gdf.b();
            let naux = b.nrows();
            if b.ncols() != nao * nao {
                return Err(FerricError::General(format!(
                    "gamma_ump2: RS-GDF B has {} columns, the reference has nao = {nao}",
                    b.ncols()
                )));
            }
            let nx = naux as u64;
            let nv_max = sp.nvir[0].max(sp.nvir[1]);
            ledger.reserve(
                &format!(
                    "Gamma UMP2 half-transformed B[k,μ,a] (naux = {naux}, nao = {nao}, \
                     nvir = {nv_max})"
                ),
                bytes_of(nx.saturating_mul(nao as u64), nv_max.saturating_mul(8)),
            )?;
            ledger.reserve(
                &format!("Gamma UMP2 B[k,ia] α + β (naux = {naux}, nov = {nova} + {novb})"),
                bytes_of(nx, nova.saturating_add(novb).saturating_mul(8)),
            )?;
            // Kernel transient per worker: same-spin G_i is (nvir_σ, ≤ nov_σ),
            // the αβ one (nvir_α, nov_β).
            let gi = (sp.nvir[0].saturating_mul(nova))
                .max(sp.nvir[1].saturating_mul(novb))
                .max(sp.nvir[0].saturating_mul(novb));
            ledger.reserve(
                &format!("Gamma UMP2 kernel G_i block per worker ({gi} elements)"),
                bytes_of(gi as u64, 8),
            )?;
            let ia = intermediates(b_ov_spin(b, &sp, 0)?, Array2::zeros((0, naux)), &sp, 0);
            let ib = intermediates(b_ov_spin(b, &sp, 1)?, Array2::zeros((0, naux)), &sp, 1);
            // v_inv_sqrt is never read by the MP2 kernels (eigenpotential
            // back-transform only), hence the empty (0, naux) placeholders.
            (
                u_ri_mp2_from_parts(&ia, &ib, &sp.eps[0], &sp.eps[1])?,
                Some(naux),
            )
        }
        GammaMp2Integrals::DenseAft(eri) => {
            let n2 = nao * nao;
            let nov_max = nova.max(novb);
            ledger.reserve(
                &format!("Gamma UMP2 dense W₁, W₂ + I·W₂ (nao = {nao}, nov ≤ {nov_max})"),
                bytes_of(n2 as u64, nov_max.saturating_mul(24)),
            )?;
            ledger.reserve(
                &format!("Gamma UMP2 dense (ia|jb) αα, ββ, αβ (nov = {nova}, {novb})"),
                bytes_of(
                    (nova as u64).saturating_add(novb as u64),
                    nova.saturating_add(novb).saturating_mul(8),
                ),
            )?;
            let (aa, bb, ab) = dense_blocks(eri.eri(), &sp)?;
            (
                ump2_from_ovov(
                    &aa,
                    &bb,
                    &ab,
                    sp.eps_occ(0),
                    sp.eps_vir(0),
                    sp.eps_occ(1),
                    sp.eps_vir(1),
                )?,
                None,
            )
        }
    };
    if !components.e_total.is_finite() {
        return Err(FerricError::General(format!(
            "gamma_ump2: non-finite correlation energy {} (zero denominator? occupied/virtual \
             degeneracy after a {:+} occupied shift)",
            components.e_total, sp.occ_shift
        )));
    }
    let mp2_corr = components.e_total;
    Ok(GammaUmp2Result {
        components,
        mp2_corr,
        total_energy: uhf.energy + mp2_corr,
        madelung: sp.madelung,
        occ_shift: sp.occ_shift,
        nocc_active: sp.nact,
        nvir: sp.nvir,
        naux,
    })
}

/// Gamma-point URPA from a converged Gamma UHF `uhf` on `cell`. See the
/// module doc for the integral sources and the conventions.
pub fn gamma_urpa(
    cell: &Cell,
    uhf: &ScfResult,
    ints: GammaDrpaIntegrals<'_>,
    cfg: &GammaDrpaConfig,
) -> Result<GammaUrpaResult, FerricError> {
    validate_quad_points(cfg.quad_points)?;
    let sp = spaces(
        "gamma_urpa",
        cell,
        uhf,
        cfg.frozen_core,
        cfg.reference_exxdiv,
        cfg.denominators,
    )?;
    let _ = excitation_energies(sp.eps_occ(0), sp.eps_vir(0))?;
    let _ = excitation_energies(sp.eps_occ(1), sp.eps_vir(1))?;
    if !sp.live(0) && !sp.live(1) {
        return Err(FerricError::General(
            "gamma_urpa: no occupied-virtual pair in either spin".into(),
        ));
    }
    let nao = sp.nao;
    let (nova, novb) = (sp.nov(0), sp.nov(1));
    let nov = nova.saturating_add(novb);
    let budget = crate::budget::resolve(cfg.budget_bytes);
    let mut ledger = Ledger::new(budget);

    let (rpa_corr, naux, quad_points, eigenvalues_static) = match ints {
        GammaDrpaIntegrals::RsGdf(gdf) => {
            let b = gdf.b();
            let naux = b.nrows();
            let w = gdf.metric_inv_sqrt();
            if b.ncols() != nao * nao || w.ncols() != naux {
                return Err(FerricError::General(format!(
                    "gamma_urpa: RS-GDF B {:?} / metric map {:?} inconsistent with nao = {nao}",
                    b.dim(),
                    w.dim()
                )));
            }
            let naux_ao = w.nrows();
            let (nx, na) = (naux as u64, naux_ao as u64);
            let nv_max = sp.nvir[0].max(sp.nvir[1]);
            ledger.reserve(
                &format!(
                    "Gamma URPA half-transformed B[k,μ,a] (naux = {naux}, nao = {nao}, \
                     nvir = {nv_max})"
                ),
                bytes_of(nx.saturating_mul(nao as u64), nv_max.saturating_mul(8)),
            )?;
            ledger.reserve(
                &format!("Gamma URPA B[k,ia] α + β (naux = {naux}, nov = {nova} + {novb})"),
                bytes_of(nx, nov.saturating_mul(8)),
            )?;
            ledger.reserve(
                &format!(
                    "Gamma URPA metric map copies (α, β) + eigenpotentials (naux_ao = {naux_ao}, \
                     naux = {naux})"
                ),
                bytes_of(na.saturating_mul(nx), 24),
            )?;
            ledger.reserve(
                &format!("Gamma URPA static dielectric + eigenvectors (naux = {naux})"),
                bytes_of(nx.saturating_mul(nx), 16),
            )?;
            ledger.reserve(
                &format!("Gamma URPA per-worker quadrature scratch (naux = {naux}, nov = {nov})"),
                bytes_of(nx, nov.saturating_add(naux).saturating_mul(8)),
            )?;
            let ia = intermediates(b_ov_spin(b, &sp, 0)?, w.to_owned(), &sp, 0);
            let ib = intermediates(b_ov_spin(b, &sp, 1)?, w.to_owned(), &sp, 1);
            let r = run_u_pdep_rpa_from_parts(
                &ia,
                &ib,
                sp.eps_occ(0),
                sp.eps_vir(0),
                sp.eps_occ(1),
                sp.eps_vir(1),
                // The whole budget: ferric-rpa's own ceiling check reads it.
                &pdep_config(cfg.quad_points, budget),
            )?;
            if !r.eigensolver_converged {
                return Err(FerricError::General(
                    "gamma_urpa: static dielectric eigensolve reported non-convergence".into(),
                ));
            }
            (
                r.e_rpa,
                Some(naux),
                Some(cfg.quad_points),
                Some(r.eigenvalues_static),
            )
        }
        GammaDrpaIntegrals::DenseAft(eri) => {
            let n2 = nao * nao;
            let nov_max = nova.max(novb);
            ledger.reserve(
                &format!("Gamma URPA dense W₁, W₂ + I·W₂ (nao = {nao}, nov ≤ {nov_max})"),
                bytes_of(n2 as u64, nov_max.saturating_mul(24)),
            )?;
            ledger.reserve(
                &format!(
                    "Gamma URPA dense (ia|jb) blocks + joint plasmon matrix + eigenvectors \
                     (nov = {nova} + {novb})"
                ),
                bytes_of(nov as u64, nov.saturating_mul(24)),
            )?;
            let (aa, bb, ab) = dense_blocks(eri.eri(), &sp)?;
            (
                urpa_plasmon(
                    &aa,
                    &bb,
                    &ab,
                    sp.eps_occ(0),
                    sp.eps_vir(0),
                    sp.eps_occ(1),
                    sp.eps_vir(1),
                )?,
                None,
                None,
                None,
            )
        }
    };
    if !rpa_corr.is_finite() {
        return Err(FerricError::General(format!(
            "gamma_urpa: non-finite correlation energy {rpa_corr}"
        )));
    }
    Ok(GammaUrpaResult {
        rpa_corr,
        total_energy: uhf.energy + rpa_corr,
        madelung: sp.madelung,
        occ_shift: sp.occ_shift,
        nocc_active: sp.nact,
        nvir: sp.nvir,
        naux,
        quad_points,
        eigenvalues_static,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd_sym_psd_blocks(nova: usize, novb: usize) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        // A random low-rank PSD joint K = Lᵀ L, split into spin blocks.
        let mut seed = 987654321u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let n = nova + novb;
        let l = Array2::from_shape_fn((3, n), |_| 0.3 * rnd());
        let k = l.t().dot(&l);
        (
            k.slice(s![..nova, ..nova]).to_owned(),
            k.slice(s![nova.., nova..]).to_owned(),
            k.slice(s![..nova, nova..]).to_owned(),
        )
    }

    /// Plasmon ≡ its own weak-coupling limit: E(λK)/λ² → direct UMP2 as λ → 0
    /// (pins the joint-space factor 2 and the ½ against the ring loop).
    #[test]
    fn plasmon_weak_coupling_limit_is_direct_ump2() {
        let (eo_a, ev_a) = (vec![-0.9, -0.6], vec![0.4, 0.8]);
        let (eo_b, ev_b) = (vec![-0.7], vec![0.3, 0.5, 0.9]);
        let (aa, bb, ab) = rnd_sym_psd_blocks(4, 3);
        let d = direct_ump2_from_ovov(&aa, &bb, &ab, &eo_a, &ev_a, &eo_b, &ev_b).unwrap();
        let f = |lam: f64| {
            urpa_plasmon(
                &(lam * &aa),
                &(lam * &bb),
                &(lam * &ab),
                &eo_a,
                &ev_a,
                &eo_b,
                &ev_b,
            )
            .unwrap()
                / (lam * lam)
        };
        let h = 1e-2;
        let rich = 2.0 * f(h / 2.0) - f(h);
        assert!(((rich - d) / d).abs() < 1e-4, "{rich} vs {d}");
        assert!(d < 0.0);
    }

    /// A single-occupied spin has an identically zero same-spin UMP2.
    #[test]
    fn one_electron_same_spin_block_is_exactly_zero() {
        let (aa, bb, ab) = rnd_sym_psd_blocks(3, 0);
        let c = ump2_from_ovov(&aa, &bb, &ab, &[-0.5], &[0.2, 0.4, 0.7], &[], &[0.1]).unwrap();
        assert_eq!(c.e_total, 0.0);
    }
}
