//! SOSEX and rCCD (RPAx): two energy-level extensions of the drCCD Riccati
//! family already implemented in [`crate::drpa_amplitude`].
//!
//! Both are PROVED against a full spin-orbital antisymmetrized Riccati
//! oracle in `wiki/notebooks/15-sosex-rccd.ipynb` (exact-ERI, not RI), so a
//! disagreement here is an implementation bug by construction. The spatial
//! forms below are the notebook's §4 and §7 verified results — do not
//! re-derive them, and do not "simplify" the index placements: several
//! plausible-looking alternatives were regressed away in the notebook.
//!
//! # SOSEX (second-order screened exchange)
//!
//! An ENERGY-ONLY change on the CONVERGED drCCD amplitude `T` — the
//! amplitude equation itself is untouched:
//!
//! ```text
//! E_SOSEX = Σ_iajb [(ia|jb) − ½ (ib|ja)] T_iajb
//! ```
//!
//! Sharp factor check (notebook §3): evaluated on the FIRST-ORDER amplitude
//! `T⁽¹⁾ = −B/D`, this must reproduce full MP2 *with* exchange exactly,
//! whereas the drCCD ring energy `½Σ B T` reproduces direct-only MP2. That
//! is the exactness anchor and it is a sharp identity, not a tolerance.
//!
//! # rCCD / RPAx
//!
//! The spin-orbital antisymmetrized Riccati decouples (notebook §6–7, by
//! exact 2×2 diagonalization of the α/β spin block at fixed spatial
//! `(i,a),(j,b)`) into two INDEPENDENT channels sharing the drCCD Riccati
//! form and Fock superoperator, differing only in their kernel:
//!
//! ```text
//! B_S = 2(ia|jb) − (ib|ja)     "+" / singlet-like
//! B_T =          − (ib|ja)     "−" / triplet-like
//! E_rCCD = E_S + E_T,   E_c = ½ Tr[B_c T_c]
//! ```
//!
//! **The channel weight is 1:1, NOT `E_S + 3 E_T`.** The multiplicity-3
//! weight belongs to EXCITATION energies / transition properties; this is a
//! ground-state amplitude contraction. The notebook verifies the 1:1 form
//! to ≤1.5e-15 against the spin-orbital oracle on all three test systems.
//!
//! ## Energy definition is genuinely ambiguous here (unlike dRPA)
//!
//! We implement the AMPLITUDE-CONTRACTION energy above. The alternative
//! "plasmon" energy built from the physically standard TDHF/RPAx `A` matrix
//! differs by **26–46%** of the plasmon magnitude on every system and
//! channel tested (notebook §7b) — a real Szabo–Ostlund-type ambiguity, not
//! roundoff. For dRPA the two are provably identical; for rCCD they are
//! not, so the plasmon energy is NOT a drop-in diagnostic substitute and is
//! deliberately not offered as one.
//!
//! ## Triplet instability is EXPECTED, not exceptional
//!
//! The "−" channel diverges once the RHF reference becomes triplet-unstable
//! (notebook §8: H₂/STO-3G past ~r = 2.0 Bohr), and this is a property of
//! the reference, not a solver defect. We therefore run the cheap `ω² < 0`
//! eigenvalue diagnostic BEFORE the (potentially very slow) divergent solve
//! and report it in the result, rather than only surfacing a fixed-point
//! relres blowup after the fact.

use ndarray::{s, Array2, Array4};

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::diis::Diis;
use ferric_scf::result::ScfResult;

use crate::drpa_amplitude::AmplitudeDrpaConfig;
use crate::lmp2_amplitude::{
    assemble_localized, build_vvhv, check_vvhv, AmplitudeLmp2Config, LocalizedProblem, VvHv,
};

/// Which member of the family to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RccdChannel {
    /// `B_S = 2(ia|jb) − (ib|ja)`.
    Singlet,
    /// `B_T = −(ib|ja)`.
    Triplet,
}

/// Configuration for ring-CCD (singlet/triplet channel decomposition).
#[derive(Debug, Clone)]
pub struct RccdConfig {
    pub frozen_core: usize,
    pub fp_rtol: f64,
    pub fp_max_iter: usize,
    pub eri3_budget_bytes: Option<usize>,
    /// Damping on the Jacobi update. The notebook's rig uses 0.7 and that
    /// is what the reference numbers were produced with; rCCD's channels
    /// are stiffer than plain drCCD (the "−" channel especially).
    pub damp: f64,
    /// Pulay/DIIS acceleration; `None` keeps the plain damped iteration.
    pub diis: Option<usize>,
    /// Run the `ω² < 0` triplet-instability diagnostic before each channel
    /// solve. Costs one dense `(no·nv)` eigensolve per channel.
    pub stability_check: bool,
}

impl Default for RccdConfig {
    fn default() -> Self {
        Self {
            frozen_core: 0,
            fp_rtol: 1e-12,
            fp_max_iter: 8000,
            eri3_budget_bytes: None,
            damp: 0.7,
            diis: None,
            stability_check: true,
        }
    }
}

/// Result of a SOSEX (second-order screened exchange) calculation built on drCCD amplitudes.
#[derive(Debug, Clone)]
#[must_use]
pub struct SosexResult {
    /// `½ Σ B T` on the converged drCCD amplitude — the drCCD ring energy,
    /// reported alongside so the SOSEX correction is readable.
    pub e_drccd: f64,
    /// `Σ [(ia|jb) − ½(ib|ja)] T_iajb`.
    pub e_sosex: f64,
    pub e_total: f64,
    pub iterations: usize,
    pub relres: f64,
    pub converged: bool,
}

/// Result for a single rCCD channel (singlet or triplet).
#[derive(Debug, Clone)]
#[must_use]
pub struct RccdChannelResult {
    pub channel: RccdChannel,
    pub e_corr: f64,
    pub iterations: usize,
    pub relres: f64,
    pub converged: bool,
    /// Smallest `Ω²` eigenvalue of `(A−B)(A+B)` for this channel, when
    /// `stability_check` is on. Negative ⇒ the RHF reference is unstable in
    /// this channel and the Riccati fixed point is expected to diverge.
    pub min_omega_sq: Option<f64>,
}

/// Combined singlet + triplet rCCD result.
#[derive(Debug, Clone)]
#[must_use]
pub struct RccdResult {
    pub singlet: RccdChannelResult,
    pub triplet: RccdChannelResult,
    /// `E_S + E_T` — weight 1:1, see the module docs.
    pub e_corr: f64,
    pub e_total: f64,
}

/// Fock superoperator `F(T)` in the compound `(ia),(jb)` representation,
/// on the LOCALIZED (non-diagonal) Fock blocks:
/// `F(T)_iajb = Σ_c f_vv[a,c] T_icjb + T_iajc f_vv[c,b]
///            − Σ_k f_oo[i,k] T_kajb − T_iakb f_oo[k,j]`.
///
/// Identical in form to [`crate::drpa_amplitude`]'s private `fock_super`;
/// duplicated rather than re-exported to keep that module's internals free
/// to change without silently altering this one's numerics.
fn fock_super(t4: &Array4<f64>, f_oo: &Array2<f64>, f_vv: &Array2<f64>) -> Array4<f64> {
    let (no, nv, _, _) = t4.dim();
    let mut r = Array4::<f64>::zeros((no, nv, no, nv));
    for i in 0..no {
        for j in 0..no {
            let blk = t4.slice(s![i, .., j, ..]); // (nv, nv) over (a, b)
            let mut acc = f_vv.dot(&blk); // Σ_c f_vv[a,c] T_icjb
            acc += &blk.dot(f_vv); // + T_iajc f_vv[c,b]
            r.slice_mut(s![i, .., j, ..]).assign(&acc);
        }
    }
    for i in 0..no {
        for j in 0..no {
            for k in 0..no {
                let f_ik = f_oo[(i, k)];
                let f_kj = f_oo[(k, j)];
                if f_ik != 0.0 {
                    let src = t4.slice(s![k, .., j, ..]).to_owned();
                    let mut dst = r.slice_mut(s![i, .., j, ..]);
                    dst.scaled_add(-f_ik, &src);
                }
                if f_kj != 0.0 {
                    let src = t4.slice(s![i, .., k, ..]).to_owned();
                    let mut dst = r.slice_mut(s![i, .., j, ..]);
                    dst.scaled_add(-f_kj, &src);
                }
            }
        }
    }
    r
}

/// Expand the compound `(no·nv, no·nv)` matrix to `(no,nv,no,nv)`.
/// Normalizes layout first for the same reason as [`to2`] — a caller may
/// hand us a transposed or otherwise non-contiguous view.
fn to4(m: &Array2<f64>, no: usize, nv: usize) -> Array4<f64> {
    m.as_standard_layout()
        .into_owned()
        .into_shape_with_order((no, nv, no, nv))
        .expect("(no*nv)^2 reshape after standard-layout normalization")
}

/// Flatten `(no,nv,no,nv)` back to the compound `(no·nv, no·nv)` matrix.
///
/// `as_standard_layout()` is REQUIRED, not cosmetic: callers hand us arrays
/// produced by [`swap_virtuals`], which is a `permuted_axes` view. Its
/// `to_owned()` preserves the permuted (non-contiguous) strides, and
/// `into_shape_with_order` on such an array fails with
/// `IncompatibleLayout` rather than silently reinterpreting memory. Same
/// family of trap as the repo's `ndarray dot layout` convention — never
/// reshape or flat-index an array whose layout you did not just establish.
fn to2(t: &Array4<f64>) -> Array2<f64> {
    let (no, nv, _, _) = t.dim();
    t.as_standard_layout()
        .into_owned()
        .into_shape_with_order((no * nv, no * nv))
        .expect("(no,nv,no,nv) reshape after standard-layout normalization")
}

/// `(ib|ja)` from `(ia|jb)`: swap the two VIRTUAL labels only, i.e. axis
/// permutation `(i,a,j,b) -> (i,b,j,a)` = `(0,3,2,1)`.
///
/// This is the swap-gather the LMP2 lane's Eq-8 exchange closure already
/// uses as a MASK criterion; here it is contracted numerically instead.
fn swap_virtuals(j4: &Array4<f64>) -> Array4<f64> {
    j4.view().permuted_axes([0, 3, 2, 1]).to_owned()
}

/// Positive MP2-style denominators `D_iajb = f_aa + f_bb − f_ii − f_jj`
/// from the LOCALIZED Fock diagonals.
fn denominators(lp: &LocalizedProblem) -> Result<Array2<f64>, FerricError> {
    let (no, nv) = (lp.no, lp.nv);
    let n = no * nv;
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    d[(i * nv + a, j * nv + b)] =
                        lp.f_vv[(a, a)] + lp.f_vv[(b, b)] - lp.f_oo[(i, i)] - lp.f_oo[(j, j)];
                }
            }
        }
    }
    if d.iter().any(|&x| x <= 0.0) {
        return Err(FerricError::General(
            "rccd_family: non-positive denominator (not a gapped system?)".into(),
        ));
    }
    Ok(d)
}

/// Outcome of one damped Riccati fixed point on kernel `b`.
struct RiccatiSolve {
    t2: Array2<f64>,
    iterations: usize,
    relres: f64,
    converged: bool,
    /// Set when the iteration blew up (relres grew past 1e3× its best) —
    /// distinguishes genuine divergence from merely hitting `fp_max_iter`.
    diverged: bool,
}

/// Damped Riccati fixed point `R(T) = B + F(T) + BT + TB + TBT`, updating
/// `T <- T − damp·R/D`. Same structure as the drCCD solver; the kernel `b`
/// is a parameter so the rCCD channels reuse it verbatim.
///
/// The divergence guard is load-bearing (notebook §8): the "−" channel is
/// EXPECTED to diverge on triplet-unstable references, and a runaway
/// iteration must be reported as divergence rather than silently burning
/// `fp_max_iter` sweeps.
fn solve_riccati(
    b: &Array2<f64>,
    f_oo: &Array2<f64>,
    f_vv: &Array2<f64>,
    d2: &Array2<f64>,
    no: usize,
    nv: usize,
    cfg: &RccdConfig,
) -> RiccatiSolve {
    let bnorm = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if bnorm == 0.0 {
        // A zero kernel has T = 0 as its exact fixed point (H2's triplet
        // channel is literally this: (ia|jb) − (ib|ja) = 0 for i=j, a=b).
        return RiccatiSolve {
            t2: Array2::zeros(b.raw_dim()),
            iterations: 0,
            relres: 0.0,
            converged: true,
            diverged: false,
        };
    }
    let mut t2 = -b / d2; // first-order start, matching the notebook rig
    let mut relres = f64::INFINITY;
    let mut best = f64::INFINITY;
    let mut it = 0;
    let mut converged = false;
    let mut diverged = false;
    let mut diis = cfg.diis.map(Diis::new);
    while it < cfg.fp_max_iter {
        it += 1;
        let t4 = to4(&t2, no, nv);
        let f_t = to2(&fock_super(&t4, f_oo, f_vv));
        let bt = b.dot(&t2);
        let tb = t2.dot(b);
        let tbt = t2.dot(&bt);
        let mut r = b + &f_t;
        r += &bt;
        r += &tb;
        r += &tbt;
        relres = r.iter().map(|x| x * x).sum::<f64>().sqrt() / bnorm;
        if relres < cfg.fp_rtol {
            converged = true;
            break;
        }
        if !relres.is_finite() || relres > 1e3 * best.max(1e-30) {
            diverged = true;
            break;
        }
        best = best.min(relres);
        let mut t2_new = &t2 - &(&(&r / d2) * cfg.damp);
        if let Some(dd) = diis.as_mut() {
            let err = &t2_new - &t2;
            t2_new = dd.step(&t2_new, &err);
        }
        t2 = t2_new;
    }
    RiccatiSolve { t2, iterations: it, relres, converged, diverged }
}

/// Smallest `Ω²` eigenvalue of `(A−B)(A+B)` with `A = diag(e_ov) + B`, on
/// the channel kernel `b`. Negative ⇒ reference unstable in this channel.
///
/// This is the CHEAP pre-flight diagnostic (notebook §8): available before
/// the divergent solve starts, unlike the fixed-point relres blowup.
fn min_omega_sq(
    b: &Array2<f64>,
    f_oo: &Array2<f64>,
    f_vv: &Array2<f64>,
    no: usize,
    nv: usize,
) -> Result<f64, FerricError> {
    use ndarray_linalg::Eig;
    let mut a = b.clone();
    for i in 0..no {
        for x in 0..nv {
            let idx = i * nv + x;
            a[(idx, idx)] += f_vv[(x, x)] - f_oo[(i, i)];
        }
    }
    let m = (&a - b).dot(&(&a + b));
    let (w, _) = m
        .eig()
        .map_err(|e| FerricError::General(format!("rccd stability eig: {e}")))?;
    let mut lo = f64::INFINITY;
    for lam in w.iter() {
        lo = lo.min(lam.re);
    }
    Ok(lo)
}

/// Assemble the localized dense problem shared by every entry point here.
fn localized(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    frozen_core: usize,
    eri3_budget_bytes: Option<usize>,
) -> Result<LocalizedProblem, FerricError> {
    let vvhv = build_vvhv(mol, obs, obs_bs, rhf)?;
    let nocc_total = (mol.nelec() as usize) / 2;
    let (dev_orth, dev_span) = check_vvhv(obs, rhf, nocc_total, &vvhv.c_vloc);
    if dev_orth > 1e-8 || dev_span > 1e-8 {
        return Err(FerricError::General(format!(
            "rccd_family: VV-HV construction check failed (orth {dev_orth:.2e}, span {dev_span:.2e})"
        )));
    }
    localized_with_virtuals(mol, obs, dfbs, op, rhf, frozen_core, eri3_budget_bytes, &vvhv)
}

fn localized_with_virtuals(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    frozen_core: usize,
    eri3_budget_bytes: Option<usize>,
    vvhv: &VvHv,
) -> Result<LocalizedProblem, FerricError> {
    let lcfg = AmplitudeLmp2Config {
        eps: 0.0, // this lane is untruncated: no amplitude threshold
        frozen_core,
        eri3_budget_bytes,
        ..Default::default()
    };
    assemble_localized(mol, obs, dfbs, op, rhf, &lcfg, vvhv)
}

/// SOSEX on the converged drCCD amplitude.
///
/// Solves the ORDINARY drCCD Riccati (`B = 2(ia|jb)`, untruncated) and then
/// contracts the converged `T` against `(ia|jb) − ½(ib|ja)`. The amplitude
/// equation is unchanged — this is an energy-functional substitution only.
pub fn sosex(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &RccdConfig,
) -> Result<SosexResult, FerricError> {
    let lp = localized(
        mol,
        obs,
        obs_bs,
        dfbs,
        op,
        rhf,
        cfg.frozen_core,
        cfg.eri3_budget_bytes,
    )?;
    let (no, nv) = (lp.no, lp.nv);
    let d2 = denominators(&lp)?;
    let b2 = lp.j_dense.mapv(|x| 2.0 * x); // drCCD kernel B = 2(ia|jb)
    let sol = solve_riccati(&b2, &lp.f_oo, &lp.f_vv, &d2, no, nv, cfg);
    if !sol.converged {
        return Err(FerricError::General(format!(
            "sosex: drCCD Riccati failed to converge (relres {:.2e} after {} iters{})",
            sol.relres,
            sol.iterations,
            if sol.diverged { ", DIVERGED" } else { "" }
        )));
    }
    let e_drccd = 0.5 * b2.iter().zip(sol.t2.iter()).map(|(b, t)| b * t).sum::<f64>();
    let e_sosex = sosex_energy(&lp.j_dense, &sol.t2, no, nv);
    Ok(SosexResult {
        e_drccd,
        e_sosex,
        e_total: rhf.energy + e_sosex,
        iterations: sol.iterations,
        relres: sol.relres,
        converged: sol.converged,
    })
}

/// `Σ_iajb [(ia|jb) − ½(ib|ja)] T_iajb` — the verified spatial SOSEX
/// contraction (notebook §4), exposed separately so the exactness anchor
/// can evaluate it on the first-order amplitude without running a solve.
pub fn sosex_energy(j_dense: &Array2<f64>, t2: &Array2<f64>, no: usize, nv: usize) -> f64 {
    let j4 = to4(j_dense, no, nv);
    let j_swapped = swap_virtuals(&j4);
    let kernel = &j4 - &j_swapped.mapv(|x| 0.5 * x);
    let t4 = to4(t2, no, nv);
    kernel.iter().zip(t4.iter()).map(|(k, t)| k * t).sum()
}

/// rCCD / RPAx: both channels, `E = E_S + E_T`.
pub fn rccd(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &RccdConfig,
) -> Result<RccdResult, FerricError> {
    let lp = localized(
        mol,
        obs,
        obs_bs,
        dfbs,
        op,
        rhf,
        cfg.frozen_core,
        cfg.eri3_budget_bytes,
    )?;
    let (no, nv) = (lp.no, lp.nv);
    let d2 = denominators(&lp)?;
    let j4 = to4(&lp.j_dense, no, nv);
    let k_ex = to2(&swap_virtuals(&j4)); // (ib|ja)

    let b_s = &lp.j_dense.mapv(|x| 2.0 * x) - &k_ex;
    let b_t = -&k_ex;

    let singlet = run_channel(RccdChannel::Singlet, &b_s, &lp, &d2, cfg)?;
    let triplet = run_channel(RccdChannel::Triplet, &b_t, &lp, &d2, cfg)?;
    let e_corr = singlet.e_corr + triplet.e_corr;
    Ok(RccdResult {
        singlet,
        triplet,
        e_corr,
        e_total: rhf.energy + e_corr,
    })
}

fn run_channel(
    channel: RccdChannel,
    b: &Array2<f64>,
    lp: &LocalizedProblem,
    d2: &Array2<f64>,
    cfg: &RccdConfig,
) -> Result<RccdChannelResult, FerricError> {
    let (no, nv) = (lp.no, lp.nv);
    let w2 = if cfg.stability_check {
        Some(min_omega_sq(b, &lp.f_oo, &lp.f_vv, no, nv)?)
    } else {
        None
    };
    // Instability is an EXPECTED outcome for the triplet channel on a
    // stretched reference (notebook §8), so it is a typed error naming the
    // diagnostic, not a panic and not a silently-returned garbage energy.
    if let Some(w) = w2 {
        if w < -1e-8 {
            return Err(FerricError::General(format!(
                "rccd: {channel:?} channel is unstable (min Omega^2 = {w:.3e} < 0) — the RHF \
                 reference is {channel:?}-unstable, so the Riccati fixed point will not \
                 converge to a physical solution"
            )));
        }
    }
    let sol = solve_riccati(b, &lp.f_oo, &lp.f_vv, d2, no, nv, cfg);
    if !sol.converged {
        return Err(FerricError::General(format!(
            "rccd: {:?} channel Riccati failed to converge (relres {:.2e} after {} iters{}; \
             min Omega^2 = {})",
            channel,
            sol.relres,
            sol.iterations,
            if sol.diverged { ", DIVERGED" } else { "" },
            w2.map_or("not computed".to_string(), |w| format!("{w:.3e}"))
        )));
    }
    let e_corr = 0.5 * b.iter().zip(sol.t2.iter()).map(|(x, t)| x * t).sum::<f64>();
    Ok(RccdChannelResult {
        channel,
        e_corr,
        iterations: sol.iterations,
        relres: sol.relres,
        converged: sol.converged,
        min_omega_sq: w2,
    })
}

/// First-order (MP2-limit) amplitude `T⁽¹⁾ = −B/D` for a given kernel —
/// the exactness-anchor entry point. Public so the anchor test can build
/// it without duplicating the denominator convention.
pub fn first_order_amplitude(b: &Array2<f64>, d2: &Array2<f64>) -> Array2<f64> {
    -b / d2
}

/// Assemble the localized dense problem and its denominators, for tests and
/// callers that want to drive the pieces directly (e.g. the MP2-limit
/// exactness anchor, which never runs a solve).
pub fn localized_problem_and_denominators(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &RccdConfig,
) -> Result<(LocalizedProblem, Array2<f64>), FerricError> {
    let lp = localized(
        mol,
        obs,
        obs_bs,
        dfbs,
        op,
        rhf,
        cfg.frozen_core,
        cfg.eri3_budget_bytes,
    )?;
    let d2 = denominators(&lp)?;
    Ok((lp, d2))
}

/// Bridge for callers that already hold an [`AmplitudeDrpaConfig`] — keeps
/// `frozen_core`/budget/tolerance choices consistent between the dRPA lane
/// and this one without duplicating them at every call site.
impl From<&AmplitudeDrpaConfig> for RccdConfig {
    fn from(c: &AmplitudeDrpaConfig) -> Self {
        Self {
            frozen_core: c.frozen_core,
            fp_rtol: c.fp_rtol,
            fp_max_iter: c.fp_max_iter.max(8000),
            eri3_budget_bytes: c.eri3_budget_bytes,
            ..Default::default()
        }
    }
}
