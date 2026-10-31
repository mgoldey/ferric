//! Amplitude-space single-threshold local MP2 (Wang/Shen/Head-Gordon,
//! JCTC 19, 7577 (2023)) — Rust Phase 2 of the prototype in
//! `scripts/amplitude_lmp2_proto.py` (measurement history in
//! `wiki/amplitude-threshold-lmp2.md` §1–§18).
//!
//! Pipeline: Boys-localized active occupieds + VV-HV orthogonal localized
//! virtuals → integral-free R⁻⁶ pair gate → per-unique-pair direct RI
//! assembly of (ia|jb) in the localized basis (Schwarz candidate subsets,
//! optional domain-local fit / aux truncation) → single threshold ε gating
//! integral magnitudes (keep (i,a,j,b) iff |(ia|jb)|>ε or |(ib|ja)|>ε —
//! the symmetric Eq-8 test, closed under the exchange swap so the
//! Hylleraas energy retains full same-spin exchange) → ragged per-pair
//! domain-block preconditioned-CG solve → Hylleraas energy.
//! Derivation: `wiki/sympy_derivations.md` §14 + notebook 14 (Hylleraas
//! stationarity, swap-closure ⇒ one-sided error, gate/fit/aux bounds).
//!
//! # Exactness anchor
//!
//! ε=0 (full mask) must reproduce [`crate::rimp2::ri_mp2`] — canonical
//! orbitals + closed-form denominators vs localized orbitals + ragged CG is
//! an independent-construction pair sharing only the RI integrals, so the
//! bar is CG tolerance (1e-9), not the RI floor. Enforced in
//! `tests/lmp2_amplitude.rs` together with the construction mutation test
//! (dropping one hard virtual must break the anchor).
//!
//! # PCG tolerance
//!
//! The energy is the stationary Hylleraas form, so its solver error is second
//! order in the residual and bounded per run by
//! [`cg_energy_error_bound`](crate::lmp2_amplitude::cg_energy_error_bound). The
//! default solve therefore stops at relres 1e-6 and continues, on the same
//! Krylov sequence, only if the bound at the achieved residual exceeds 1e-10
//! Ha ([`CgRtolPolicy`](crate::lmp2_amplitude::CgRtolPolicy)). Running out of
//! iterations at either stage is an error. ferric-cc's
//! LinLCCD and ferric-pbc's Gamma LMP2 share the PCG but use the LINEAR energy,
//! whose error is first order, and keep their own 1e-11.
//!
//! # Documented deviations from the paper (match the Python rig's wiki note)
//!
//! - Hard virtuals: spread-weighted pivoted-Cholesky selection of
//!   projected-AO candidates + Löwdin (same construction as the Python
//!   rig, using the emultipole2 second-moment integrals), instead of the
//!   paper's weighted symmetric orthogonalization of the full redundant
//!   set. Anchors are invariant to the selection weighting (the selected
//!   set spans the same space); only domain compactness depends on it.
//! - The dense global-metric J (`assemble_localized`) survives only as the
//!   test/cross-check path. Production assembly is
//!   [`crate::lmp2_amplitude::assemble_ragged_direct_aux`]: integral-free R⁻⁶ pair gate before any
//!   GEMM (`pair_gate_cal`), per-unique-pair whitened Gram GEMMs over
//!   exact-Schwarz candidate subsets, optional per-pair domain-local
//!   same-kernel fits (`fit_radius_bohr`) and ε-linked aux truncation
//!   (`aux_tail_frac`) — no no·nv × no·nv object is ever formed. The
//!   3-index B tensor is still built globally (`eri3_mo_ov_blocked`), so
//!   the assembly is NOT integral-direct and no asymptotic scaling claim
//!   attaches; the measured record (C16/erfc/ε=1e-3 crossover, aux-domain
//!   saturation) is in `wiki/amplitude-threshold-lmp2.md` §§23–26.

use ndarray::{s, Array2};
use std::collections::HashMap;

use ferric_core::basis::BasisSet;
use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::{dipole, overlap, r2_moment};
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;

use crate::boys::boys_localize;
use crate::ragged::{
    check_swap_closed, lanczos_kappa, matvec_indexed, matvec_planned, pair_block_from_g_cand,
    solve_ragged_budgeted_steered, BudgetedSolve, MatvecPath, PairBlock, Ragged, SolvePlan,
};
use crate::rimp2::{
    active_occ, eri3_budget_bytes, eri3_mo_ov_blocked, metric_inverse_sqrt, ri_mp2, RiMp2Config,
};
use ferric_integrals::threeindex::coulomb_metric_2c;

/// The default PCG relative-residual target of local MP2. The Hylleraas
/// energy's solver error is second order in the residual, so this is enough
/// while the per-run bound holds ([`LMP2_CG_ENERGY_ERROR_TARGET`]).
pub const LMP2_CG_RTOL_DEFAULT: f64 = 1e-6;
/// The tightest tolerance the default policy goes to, and the one it uses when
/// the bound is undefined (no positive gap).
pub const LMP2_CG_RTOL_STRICT: f64 = 1e-11;
/// The Hylleraas solver-error bound (Ha) the default policy keeps a run under.
pub const LMP2_CG_ENERGY_ERROR_TARGET: f64 = 1e-10;
/// Safety factor on the residual when tightening: the bound lands at a quarter
/// of the target instead of on it.
const CG_TIGHTEN_SAFETY: f64 = 0.5;

/// Configuration for amplitude-threshold local MP2 (WSHG23 single-threshold scheme).
#[derive(Debug, Clone)]
pub struct AmplitudeLmp2Config {
    /// The single threshold ε on |(ia|jb)| in the localized basis (Eq. 8).
    /// ε = 0 keeps everything (the exactness-anchor limit).
    pub eps: f64,
    pub frozen_core: usize,
    /// PCG relative-residual target. `None` (the default) is the adaptive
    /// policy: [`LMP2_CG_RTOL_DEFAULT`], continued to a tighter tolerance if
    /// [`cg_energy_error_bound`] at the achieved residual exceeds
    /// [`LMP2_CG_ENERGY_ERROR_TARGET`], never tighter than
    /// [`LMP2_CG_RTOL_STRICT`]. `Some(x)` runs at exactly `x` with no
    /// tightening. Rust API only: the CLI and Python do not expose it.
    pub cg_rtol: Option<f64>,
    /// Iteration limit over the whole solve, tightening included.
    pub cg_max_iter: usize,
    /// 3-index working-memory budget; `None` → the shared rimp2 default.
    pub eri3_budget_bytes: Option<usize>,
    /// Integral-free pair gate: `Some(cal)` skips pair blocks whose
    /// London-type estimate `cal·(σᵢσⱼ)³/R⁶` falls below θ = 1e-2·eps
    /// (the paper's linked gate) BEFORE any solver work — and before any
    /// per-pair fit when `fit_radius_bohr` is set. `cal` is the
    /// conservatively calibrated constant from the Python rig (p95:
    /// ~0.7 Coulomb, ~0.02 erfc ω=1, stable C8→C10). Diagonal pairs are
    /// never gated; eps = 0 makes the gate inert (anchor limit).
    pub pair_gate_cal: Option<f64>,
    /// ε-linked per-pair aux (naux) truncation for the whitened path:
    /// `Some(f)` drops, per pair, the aux functions whose L1 tail bound
    /// Σ_dropped max_a|B̃_Pia|·max_b|B̃_Pjb| stays ≤ f·ε — an EXACT
    /// elementwise bound on the Gram perturbation, so the aux error is
    /// rigorously ≤ f·ε per element. `Some(0.0)` and `None` keep every
    /// aux function (byte-identical trivial limit). Inactive at ε = 0 and
    /// on the domain-fit path (which has its own aux domains).
    pub aux_tail_frac: Option<f64>,
    /// Also compute the canonical `ri_mp2` reference inside the driver (the
    /// honesty printout). OPT-IN, default false: the reference is a full
    /// N^5 canonical RI-MP2 that forms the global (naux, nocc·nvir) tensor —
    /// exactly what the local (and especially the integral-direct) path
    /// exists to avoid, so leaving it on by default made no run
    /// reduced-cost. Off: `e_corr_canonical_ri` is NaN, `t_reference_s` is
    /// 0 and `ri_mp2` is never called. Turn it on for validation and for
    /// the ε = 0 exactness anchors.
    pub compute_reference: bool,
    /// `Some(r)`: per-pair domain-local same-kernel RI fit — for pair
    /// (i,j) only aux functions within `r` Bohr of either Boys centroid
    /// enter the fit (J_ij = A_D V_DD⁻¹ A_D; with one shared domain and
    /// the same-kernel metric the Dunlap robust correction cancels
    /// identically). `None`: global-metric fit (byte-identical to the
    /// original path). r ≥ 1e5 must reproduce the global fit to machine
    /// precision (trivial-limit anchor, tested).
    pub fit_radius_bohr: Option<f64>,
    /// Adaptive Fvv product of the PCG matvec (perf plan #3e): a pair block
    /// whose retained-pattern density ρ = kept/(|Da|·|Db|) is BELOW this
    /// threshold ρ* multiplies Fvv by a sparse column/row scatter over its
    /// retained elements instead of two dense GEMMs. `0.0` (the default)
    /// keeps every block dense — byte-identical to the pre-switch path.
    /// The crossover ρ* is not measured yet (benchmark pending).
    pub fvv_sparse_rho: f64,
}

impl Default for AmplitudeLmp2Config {
    fn default() -> Self {
        Self {
            eps: 1e-4,
            frozen_core: 0,
            cg_rtol: None,
            cg_max_iter: 400,
            eri3_budget_bytes: None,
            pair_gate_cal: None,
            aux_tail_frac: None,
            compute_reference: false,
            fit_radius_bohr: None,
            fvv_sparse_rho: 0.0,
        }
    }
}

/// Result of an amplitude-threshold local MP2 calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct AmplitudeLmp2Result {
    pub e_corr: f64,
    pub e_total: f64,
    /// Canonical RI-MP2 on the same (mol, obs, dfbs, op, frozen_core) —
    /// the independent-construction reference; at ε=0 `e_corr` must match
    /// this to CG tolerance. NaN unless `compute_reference` was set (the
    /// default is off).
    pub e_corr_canonical_ri: f64,
    pub keep_fraction: f64,
    pub pair_fraction: f64,
    pub dom_mean: f64,
    pub dom_max: usize,
    pub cg_iterations: usize,
    /// The PCG's recursive relative residual.
    pub cg_relres: f64,
    /// The true relative residual `‖J + A t‖ / ‖PJ‖`, from the energy's matvec.
    pub cg_true_relres: f64,
    /// Always true on a returned result: a PCG that runs out of iterations,
    /// at the first tolerance or while tightening, is an `Err`
    /// ([`solve_under_rtol_policy`]). Whether the energy-error target was met
    /// is a separate question: [`CgRtolOutcome::target_met`].
    pub cg_converged: bool,
    pub n_valence_virt: usize,
    pub n_hard_virt: usize,
    /// Multiply-add count of one ragged matvec vs its dense equivalent
    /// (2·(no²·nv³ + no³·nv²)) — the counter pair the prototype's scaling
    /// table quotes.
    pub ragged_flops_per_matvec: u64,
    pub dense_flops_per_matvec: u64,
    /// Unique off-diagonal pairs removed by the integral-free gate
    /// (0 when the gate is off or eps = 0).
    pub n_pairs_gated: usize,
    /// Mean/max per-pair aux contraction length (== naux when the
    /// ε-linked aux truncation is off).
    pub aux_dom_mean: f64,
    pub aux_dom_max: usize,
    /// κ estimate of the diagonally preconditioned CG operator from the CG
    /// coefficients ([`crate::ragged::lanczos_kappa`]; a lower bound, NaN
    /// when undefined). The 1e-6 default gives it fewer iterations, so values
    /// are not comparable with runs from before that default.
    pub cg_lanczos_kappa_est: f64,
    /// In-block retained-pattern density (kept / (|Da|·|Db|)) over the pair
    /// blocks: mean and min (0 when there are no blocks).
    pub block_density_mean: f64,
    pub block_density_min: f64,
    /// Unordered pairs (i ≤ j) that carry a block.
    pub n_pairs_unique: usize,
    /// Whether the integral-free pair gate was configured
    /// (`pair_gate_cal.is_some()`; inert at ε = 0).
    pub gate_on: bool,
    /// Size of the PCG gather plan ([`crate::ragged::SolvePlan::bytes_for`]),
    /// computed before deciding whether to build it; reported either way.
    pub solve_plan_bytes: usize,
    /// Bytes resident beside the plan for the whole solve (the ragged
    /// system + the PCG vectors, [`crate::ragged::solve_resident_bytes`]);
    /// the plan is admitted against the budget minus this.
    pub solve_resident_bytes: usize,
    /// Bound (Ha) on the Hylleraas energy's CG solver error at the final TRUE
    /// residual, `3·‖PJ‖²·relres²/λ_min` with λ_min ≥ 2(ε_LUMO −
    /// ε_HOMO) ([`cg_energy_error_bound`]); NaN if the gap is not positive.
    pub cg_energy_error_bound: f64,
    /// Which tolerance the solve ran to, and why.
    pub cg_rtol_outcome: CgRtolOutcome,
    /// The matvec the PCG ran: planned when the plan fit the memory budget,
    /// otherwise the plan-free indexed one (bitwise the same result).
    pub matvec_path: MatvecPath,
    pub timings: StageTimings,
}

/// Where a solve's PCG tolerance came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CgRtolSource {
    /// The adaptive default, with its bound defined.
    AdaptiveDefault,
    /// The adaptive default with no positive gap: the bound is undefined, so
    /// the solve ran at [`LMP2_CG_RTOL_STRICT`].
    StrictNoGap,
    /// `cg_rtol: Some(x)`, taken literally.
    Explicit,
}

impl CgRtolSource {
    pub fn as_str(self) -> &'static str {
        match self {
            CgRtolSource::AdaptiveDefault => "adaptive-default",
            CgRtolSource::StrictNoGap => "strict-no-gap",
            CgRtolSource::Explicit => "explicit",
        }
    }
}

/// What the tolerance policy did on one solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CgRtolOutcome {
    pub source: CgRtolSource,
    /// The last tolerance the PCG ran to.
    pub rtol_used: f64,
    /// Whether the bound forced a continuation past the starting tolerance.
    pub tightened: bool,
    /// The bound target (Ha); `None` unless the bound governed the solve.
    pub target_ha: Option<f64>,
    /// Whether the bound at the final true residual met the target; `None`
    /// when no target applied. `Some(false)` means the PCG did converge, either
    /// at the floor or with a true residual that drifted from the recursive one.
    pub target_met: Option<bool>,
}

/// The tolerance rule [`solve_under_rtol_policy`] applies. Production builds
/// it from `AmplitudeLmp2Config::cg_rtol`; tests reach the tightening branch
/// by lowering `target_ha`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CgRtolPolicy {
    Explicit(f64),
    Adaptive {
        start: f64,
        floor: f64,
        target_ha: f64,
    },
}

impl CgRtolPolicy {
    pub fn from_config(cg_rtol: Option<f64>) -> Self {
        cg_rtol.map_or(
            CgRtolPolicy::Adaptive {
                start: LMP2_CG_RTOL_DEFAULT,
                floor: LMP2_CG_RTOL_STRICT,
                target_ha: LMP2_CG_ENERGY_ERROR_TARGET,
            },
            CgRtolPolicy::Explicit,
        )
    }
}

/// The pattern counters both LMP2 drivers report, computed from the
/// assembled ragged system.
pub(crate) struct RaggedCounters {
    pub keep_fraction: f64,
    pub pair_fraction: f64,
    pub dom_mean: f64,
    pub dom_max: usize,
    pub dense_flops: u64,
    pub block_density_mean: f64,
    pub block_density_min: f64,
    pub n_pairs_unique: usize,
}

impl RaggedCounters {
    pub(crate) fn new(rg: &Ragged, no: usize, nv: usize) -> Self {
        let total_el = (no * nv) as u64 * (no * nv) as u64;
        let mut kept = 0u64;
        let (mut dens_sum, mut dens_min) = (0.0f64, f64::INFINITY);
        let mut per_i: Vec<Vec<bool>> = vec![vec![false; nv]; no];
        for pb in &rg.pairs {
            let k = pb.pat.iter().filter(|&&x| x).count();
            kept += k as u64;
            let dens = k as f64 / pb.pat.len().max(1) as f64;
            dens_sum += dens;
            dens_min = dens_min.min(dens);
            for &a in &pb.da {
                per_i[pb.i][a] = true;
            }
        }
        let dom: Vec<usize> = per_i
            .iter()
            .map(|v| v.iter().filter(|&&x| x).count())
            .collect();
        let npb = rg.pairs.len();
        Self {
            keep_fraction: kept as f64 / total_el as f64,
            pair_fraction: npb as f64 / (no * no) as f64,
            dom_mean: if no > 0 {
                dom.iter().sum::<usize>() as f64 / no as f64
            } else {
                0.0
            },
            dom_max: dom.iter().copied().max().unwrap_or(0),
            dense_flops: 2
                * ((no * no) as u64 * (nv as u64).pow(3) + (no as u64).pow(3) * (nv * nv) as u64),
            block_density_mean: if npb > 0 { dens_sum / npb as f64 } else { 0.0 },
            block_density_min: if npb > 0 { dens_min } else { 0.0 },
            n_pairs_unique: rg.pairs.iter().filter(|pb| pb.i <= pb.j).count(),
        }
    }
}

impl std::fmt::Display for AmplitudeLmp2Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Amplitude-LMP2 total: {:.10} Ha (corr: {:.10}, keep: {:.1}%, {} CG iters)",
            self.e_total,
            self.e_corr,
            self.keep_fraction * 100.0,
            self.cg_iterations
        )
    }
}

/// Wall-clock per pipeline stage, seconds. `t_reference_s` is the canonical
/// ri_mp2 reference — NOT part of the method cost, reported separately so
/// benchmark tables can exclude it; exactly 0 when the (opt-in) reference
/// was not computed.
#[derive(Debug, Clone, Default)]
pub struct StageTimings {
    /// VV-HV virtual construction (`build_vvhv` + `check_vvhv`). Set by the
    /// drivers that build the space ([`amplitude_lmp2`],
    /// `lmp2_direct::amplitude_lmp2_direct`); 0 on the `*_with_virtuals`
    /// entry points, whose caller built it.
    pub t_spaces_s: f64,
    pub t_assembly_s: f64,
    pub t_solve_s: f64,
    pub t_reference_s: f64,
}

/// VV-HV orthogonal localized virtual space.
#[derive(Debug, Clone)]
pub struct VvHv {
    /// (nao, nvir) — valence virtuals first, then hard virtuals.
    pub c_vloc: Array2<f64>,
    pub n_valence: usize,
    pub n_hard: usize,
}

// ---------------------------------------------------------------------------
// small dense-linear-algebra helpers (nao-scale eigh via ndarray-linalg)
// ---------------------------------------------------------------------------

fn eigh(m: &Array2<f64>) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    use ndarray_linalg::Eigh;
    let (w, v) = m
        .eigh(ndarray_linalg::UPLO::Lower)
        .map_err(|e| FerricError::General(format!("lmp2_amplitude eigh failed: {e}")))?;
    Ok((w.to_vec(), v))
}

/// Symmetric (Löwdin) orthonormalization of the columns of `c` w.r.t. `s`.
///
/// `pub` (visibility only, 2026-09-24) so ferric-pbc's Gamma LMP2 builds its
/// periodic VV-HV space with the SAME orthonormalization; the body is
/// unchanged, so the molecular path is byte-identical.
pub fn lowdin(c: &Array2<f64>, s: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let o = c.t().dot(&s.dot(c));
    let (w, v) = eigh(&o)?;
    let wmin = w.iter().cloned().fold(f64::INFINITY, f64::min);
    if wmin < 1e-10 {
        return Err(FerricError::General(format!(
            "lmp2_amplitude: Löwdin set near-singular (min eig {wmin:.2e})"
        )));
    }
    let mut vs = v.clone();
    for (k, wk) in w.iter().enumerate() {
        let f = 1.0 / wk.sqrt();
        vs.column_mut(k).mapv_inplace(|x| x * f);
    }
    Ok(c.dot(&vs.dot(&v.t())))
}

/// Canonical orthonormalization keeping the `rank` largest-eigenvalue
/// directions; errors if the kept spectrum dips below the lindep floor.
/// `pub` for ferric-pbc (visibility only; see [`lowdin`]).
pub fn canonical_orth(
    c: &Array2<f64>,
    s: &Array2<f64>,
    rank: usize,
) -> Result<Array2<f64>, FerricError> {
    const LINDEP: f64 = 1e-8;
    let o = c.t().dot(&s.dot(c));
    let (w, v) = eigh(&o)?;
    // eigh returns ascending; take the top `rank`.
    let n = w.len();
    if rank > n {
        return Err(FerricError::General(format!(
            "lmp2_amplitude canonical_orth: rank {rank} > candidate dim {n}"
        )));
    }
    let kept: Vec<usize> = (n - rank..n).collect();
    let wmin = kept.iter().map(|&k| w[k]).fold(f64::INFINITY, f64::min);
    if wmin < LINDEP {
        return Err(FerricError::General(format!(
            "lmp2_amplitude canonical_orth: rank {rank} unreachable (min kept eig {wmin:.2e})"
        )));
    }
    let nao = c.nrows();
    let mut out = Array2::<f64>::zeros((nao, rank));
    for (col, &k) in kept.iter().enumerate() {
        let f = 1.0 / w[k].sqrt();
        let dir = c.dot(&v.column(k).to_owned());
        out.column_mut(col).assign(&(dir.mapv(|x| x * f)));
    }
    Ok(out)
}

/// Relative width inside which residual diagonals count as TIED in
/// [`pivoted_cholesky_order`]. Symmetry-equivalent candidates tie exactly in
/// exact arithmetic but differ in the last bits under any change of
/// summation order (e.g. the Stage-B1 Cholesky vs LU solve); without a
/// tolerance the pivot then follows rounding noise.
pub const PIVOT_TIE_RTOL: f64 = 1e-12;

/// Index of the pivot: the LOWEST index whose residual diagonal is within
/// [`PIVOT_TIE_RTOL`] (relative) of the maximum, so near-ties are broken
/// deterministically by position rather than by rounding.
fn select_pivot(d: &[f64]) -> (usize, f64) {
    assert!(
        d.iter().all(|x| !x.is_nan()),
        "NaN in pivoted Cholesky diagonal"
    );
    let dmax = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let floor = dmax - PIVOT_TIE_RTOL * dmax.abs();
    let j = d.iter().position(|&x| x >= floor).expect("empty diagonal");
    (j, d[j])
}

/// Greedy pivoted Cholesky on a PSD matrix; returns `rank` pivot indices.
/// Near-ties (within [`PIVOT_TIE_RTOL`] relative) go to the lowest index.
/// `pub` for ferric-pbc (visibility only; see [`lowdin`]).
pub fn pivoted_cholesky_order(m: &Array2<f64>, rank: usize) -> Result<Vec<usize>, FerricError> {
    let n = m.nrows();
    let mut d: Vec<f64> = (0..n).map(|i| m[(i, i)]).collect();
    let mut l = Array2::<f64>::zeros((rank, n));
    let mut piv = Vec::with_capacity(rank);
    for k in 0..rank {
        let (j, dj) = select_pivot(&d);
        if dj <= 0.0 {
            return Err(FerricError::General(format!(
                "lmp2_amplitude: pivoted Cholesky broke down at k={k}"
            )));
        }
        piv.push(j);
        let ljk_prev = l.slice(s![..k, j]).to_owned();
        let mrow = m.row(j).to_owned();
        let proj = l.slice(s![..k, ..]).t().dot(&ljk_prev);
        let scale = 1.0 / dj.sqrt();
        for c in 0..n {
            l[(k, c)] = (mrow[c] - proj[c]) * scale;
        }
        for c in 0..n {
            d[c] -= l[(k, c)] * l[(k, c)];
        }
        for &p in &piv {
            d[p] = f64::NEG_INFINITY;
        }
    }
    Ok(piv)
}

// ---------------------------------------------------------------------------
// projected-minimal cross overlap via a combined basis
// ---------------------------------------------------------------------------

/// Refuse a molecule with ECP-substituted atoms. The valence-virtual count is
/// `n_min − n_occ` with STO-3G, an ALL-ELECTRON minimal basis, while `n_occ`
/// excludes the ECP core, so the count is wrong on such atoms (HI in
/// aug-cc-pVDZ-PP: 28 STO-3G functions − 13 occupied = 15, which counts the
/// ECP core's functions as valence virtuals). STO-3G stopped at Ar until PR #596, so
/// these molecules used to fail here on a missing element; they still fail,
/// now with the reason.
fn refuse_ecp_atoms(mol: &Molecule, obs_bs: &BasisSet) -> Result<(), FerricError> {
    let ecp_atom = mol
        .atoms
        .iter()
        .find(|a| !a.ghost && (a.n_core_ecp > 0 || obs_bs.ecp_for_element(a.z).is_some()));
    match ecp_atom {
        None => Ok(()),
        Some(a) => Err(FerricError::Basis(format!(
            "lmp2_amplitude: the VV-HV localized virtuals project an all-electron \
             STO-3G minimal basis and cannot count valence virtuals on ECP atoms \
             ({} carries an ECP in basis {:?}); use an all-electron basis or the \
             exact (non-local) method",
            a.symbol, obs_bs.name
        ))),
    }
}

/// Overlap block ⟨obs AO | minimal AO⟩ computed by preparing ONE combined
/// basis (obs shells then STO-3G shells per element) and slicing the full
/// overlap — ferric has no cross-basis 1e path, and per-atom shell order
/// follows the merged `Vec<Shell>` order (basis_bridge walks
/// `bs.for_element` per atom), so membership is reconstructible exactly.
fn cross_overlap_with_minimal(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
) -> Result<(Array2<f64>, usize), FerricError> {
    refuse_ecp_atoms(mol, obs_bs)?;
    let min_bs = ferric_core::basis::bundled("sto-3g")
        .map_err(|e| FerricError::General(format!("lmp2_amplitude: bundled sto-3g: {e}")))?;
    let mut merged: HashMap<i32, Vec<ferric_core::basis::Shell>> = HashMap::new();
    let mut n_obs_shells: HashMap<i32, usize> = HashMap::new();
    for atom in &mol.atoms {
        let z = atom.z;
        if merged.contains_key(&z) {
            continue;
        }
        let o = obs_bs
            .for_element(z)
            .ok_or_else(|| FerricError::Basis(format!("no obs shells for Z={z}")))?;
        let m = min_bs
            .for_element(z)
            .ok_or_else(|| FerricError::Basis(format!("no sto-3g shells for Z={z}")))?;
        let mut v: Vec<ferric_core::basis::Shell> = o.to_vec();
        v.extend(m.iter().cloned());
        n_obs_shells.insert(z, o.len());
        merged.insert(z, v);
    }
    let merged_bs = BasisSet {
        name: "lmp2-obs+sto3g".to_string(),
        shells: merged,
        ecps: obs_bs.ecps.clone(),
    };
    let comb = PreparedBasis::new(mol, &merged_bs)?;
    let s_comb = overlap(&comb);

    // classify each combined AO as obs or minimal by walking shells per atom
    let mut obs_idx = Vec::new();
    let mut min_idx = Vec::new();
    let mut atom_shell_count: Vec<usize> = vec![0; mol.atoms.len()];
    for (sh, &ai) in comb.shell_to_atom().iter().enumerate() {
        let z = mol.atoms[ai].z;
        let k = atom_shell_count[ai];
        let dim = comb.shell_dims()[sh];
        let off = comb.shell_offsets()[sh];
        let is_obs = k < n_obs_shells[&z];
        for f in 0..dim {
            if is_obs {
                obs_idx.push(off + f);
            } else {
                min_idx.push(off + f);
            }
        }
        atom_shell_count[ai] += 1;
    }
    if obs_idx.len() != obs.nbasis() {
        return Err(FerricError::General(format!(
            "lmp2_amplitude: combined-basis bookkeeping mismatch ({} obs AOs vs {})",
            obs_idx.len(),
            obs.nbasis()
        )));
    }
    let nmin = min_idx.len();
    let mut s_x = Array2::<f64>::zeros((obs_idx.len(), nmin));
    for (r, &ir) in obs_idx.iter().enumerate() {
        for (c, &ic) in min_idx.iter().enumerate() {
            s_x[(r, c)] = s_comb[(ir, ic)];
        }
    }
    Ok((s_x, nmin))
}

/// Per-AO parent-atom map for the orbital basis.
fn ao_to_atom(prep: &PreparedBasis) -> Vec<usize> {
    let mut out = vec![0usize; prep.nbasis()];
    for (sh, &ai) in prep.shell_to_atom().iter().enumerate() {
        let off = prep.shell_offsets()[sh];
        for f in 0..prep.shell_dims()[sh] {
            out[off + f] = ai;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// VV-HV construction
// ---------------------------------------------------------------------------

/// Build the VV-HV orthogonal localized virtual space (paper §2.1; see the
/// module doc for the documented HV-selection deviation).
pub fn build_vvhv(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    rhf: &ScfResult,
) -> Result<VvHv, FerricError> {
    build_vvhv_with(mol, obs, obs_bs, rhf, solve_spd)
}

/// The SPD solve `S⁻¹·B` [`build_vvhv`] uses (a seam so tests can build the
/// space with a frozen oracle solver).
type SpdSolve = fn(&Array2<f64>, &Array2<f64>) -> Result<Array2<f64>, FerricError>;

fn build_vvhv_with(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    rhf: &ScfResult,
    solve: SpdSolve,
) -> Result<VvHv, FerricError> {
    build_vvhv_traced(mol, obs, obs_bs, rhf, solve).map(|(v, _)| v)
}

/// [`build_vvhv_with`] that also returns the AO index of every selected
/// hard-virtual candidate, in pivot order (the hard-virtual SUBSET, which
/// tests compare across solvers).
fn build_vvhv_traced(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    rhf: &ScfResult,
    solve: SpdSolve,
) -> Result<(VvHv, Vec<usize>), FerricError> {
    let s = overlap(obs);
    let nao = obs.nbasis();
    let nocc_total = (mol.nelec() as usize) / 2;
    let c_occ_all = rhf.mos_r().slice(s![.., ..nocc_total]).to_owned();

    // valence virtuals: projected STO-3G minus the occupied span
    let (s_x, nmin) = cross_overlap_with_minimal(mol, obs, obs_bs)?;
    let t = solve(&s, &s_x)?; // S⁻¹ S_x : projected minimal in obs AO space
    let q_occ = c_occ_all.dot(&c_occ_all.t().dot(&s));
    let tv = &t - &q_occ.dot(&t);
    let n_l = nmin.checked_sub(nocc_total).ok_or_else(|| {
        FerricError::General(format!(
            "lmp2_amplitude: minimal basis ({nmin}) smaller than nocc ({nocc_total})"
        ))
    })?;
    let dip = dipole(obs, [0.0, 0.0, 0.0])?;
    let c_l = if n_l > 0 {
        let cl = canonical_orth(&tv, &s, n_l)?;
        boys_localize(&cl, &dip, 200).c_loc
    } else {
        Array2::<f64>::zeros((nao, 0))
    };

    // hard virtuals: project occ+valence out of each AO, select by pivoted
    // Cholesky, Löwdin, per-atom pseudo-canonicalize
    let n_h = nao - nocc_total - n_l;
    let mut hard_aos = Vec::with_capacity(n_h);
    let c_h = if n_h > 0 {
        let mut c_e = Array2::<f64>::zeros((nao, nocc_total + n_l));
        c_e.slice_mut(s![.., ..nocc_total]).assign(&c_occ_all);
        c_e.slice_mut(s![.., nocc_total..]).assign(&c_l);
        let proj = c_e.dot(&c_e.t().dot(&s));
        // candidates: unit vectors minus their occ+valence projection
        let mut x = Array2::<f64>::eye(nao);
        x -= &proj; // columns are the projected AOs
        let sx = s.dot(&x);
        let mut keep_cols = Vec::new();
        for c in 0..nao {
            let nrm2 = x.column(c).dot(&sx.column(c));
            if nrm2 > 1e-8 {
                keep_cols.push((c, nrm2.sqrt()));
            }
        }
        let mut xn = Array2::<f64>::zeros((nao, keep_cols.len()));
        let mut parents = Vec::with_capacity(keep_cols.len());
        let a2a = ao_to_atom(obs);
        for (k, &(c, nrm)) in keep_cols.iter().enumerate() {
            xn.column_mut(k).assign(&(x.column(c).mapv(|v| v / nrm)));
            parents.push(a2a[c]);
        }
        // spread-weighted pivots (matches the Python rig): compact candidates
        // first, via w = 1/spread with spread = ⟨r²⟩ − |⟨r⟩|² per candidate
        let r2 = r2_moment(obs, [0.0; 3])?;
        let ncand = xn.ncols();
        let mut w = vec![0.0f64; ncand];
        for k in 0..ncand {
            let col = xn.column(k);
            let r2v = col.dot(&r2.dot(&col));
            let mut c2 = 0.0;
            for dm in dip.iter() {
                let d = col.dot(&dm.dot(&col));
                c2 += d * d;
            }
            w[k] = 1.0 / (r2v - c2).max(1e-6);
        }
        let wmax2 = w.iter().fold(0.0f64, |m, &x| m.max(x)).powi(2);
        let mut ov = xn.t().dot(&s.dot(&xn));
        for r in 0..ncand {
            for c in 0..ncand {
                ov[(r, c)] *= w[r] * w[c] / wmax2;
            }
        }
        let piv = pivoted_cholesky_order(&ov, n_h)?;
        let mut sel = Array2::<f64>::zeros((nao, n_h));
        let mut sel_parents = Vec::with_capacity(n_h);
        for (k, &p) in piv.iter().enumerate() {
            sel.column_mut(k).assign(&xn.column(p));
            sel_parents.push(parents[p]);
            hard_aos.push(keep_cols[p].0);
        }
        let mut ch = lowdin(&sel, &s)?;
        // per-atom pseudo-canonicalization (block-diagonal — cannot
        // delocalize, unlike full semicanonicalization)
        let f_ao = rhf.fock_r();
        let fh = ch.t().dot(&f_ao.dot(&ch));
        let mut by_atom: HashMap<usize, Vec<usize>> = HashMap::new();
        for (k, &a) in sel_parents.iter().enumerate() {
            by_atom.entry(a).or_default().push(k);
        }
        for idx in by_atom.values() {
            let m = idx.len();
            let mut blk = Array2::<f64>::zeros((m, m));
            for (r, &ir) in idx.iter().enumerate() {
                for (c, &ic) in idx.iter().enumerate() {
                    blk[(r, c)] = fh[(ir, ic)];
                }
            }
            let (_, u) = eigh(&blk)?;
            let cols: Vec<ndarray::Array1<f64>> =
                idx.iter().map(|&ic| ch.column(ic).to_owned()).collect();
            for (cnew, &ic) in idx.iter().enumerate() {
                let mut acc = ndarray::Array1::<f64>::zeros(nao);
                for (r, col) in cols.iter().enumerate() {
                    acc.scaled_add(u[(r, cnew)], col);
                }
                ch.column_mut(ic).assign(&acc);
            }
        }
        ch
    } else {
        Array2::<f64>::zeros((nao, 0))
    };

    let mut c_vloc = Array2::<f64>::zeros((nao, n_l + n_h));
    c_vloc.slice_mut(s![.., ..n_l]).assign(&c_l);
    c_vloc.slice_mut(s![.., n_l..]).assign(&c_h);
    let v = VvHv {
        c_vloc,
        n_valence: n_l,
        n_hard: n_h,
    };
    Ok((v, hard_aos))
}

/// `S⁻¹·B` for SPD `S`: ONE Cholesky factorization, then a forward/back
/// triangular solve per right-hand side (the previous version LU-factorized
/// S once per column — nmin factorizations of an nao² matrix).
fn solve_spd(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::{FactorizeC, SolveC, UPLO};
    let f = a
        .factorizec(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("lmp2_amplitude Cholesky of S: {e}")))?;
    let mut out = Array2::<f64>::zeros((a.nrows(), b.ncols()));
    for (c, mut oc) in out.columns_mut().into_iter().enumerate() {
        let col = f
            .solvec(&b.column(c))
            .map_err(|e| FerricError::General(format!("lmp2_amplitude solve: {e}")))?;
        oc.assign(&col);
    }
    Ok(out)
}

/// Orthonormality + span check against the canonical virtuals (the
/// construction half of the exactness anchor). Returns (dev_orth, dev_span).
pub fn check_vvhv(
    obs: &PreparedBasis,
    rhf: &ScfResult,
    nocc_total: usize,
    c_vloc: &Array2<f64>,
) -> (f64, f64) {
    let s = overlap(obs);
    let o = c_vloc.t().dot(&s.dot(c_vloc));
    let n = o.nrows();
    let mut dev_orth: f64 = 0.0;
    for r in 0..n {
        for c in 0..n {
            let target = if r == c { 1.0 } else { 0.0 };
            dev_orth = dev_orth.max((o[(r, c)] - target).abs());
        }
    }
    let c_vcan = rhf.mos_r().slice(s![.., nocc_total..]).to_owned();
    let u = c_vcan.t().dot(&s.dot(c_vloc));
    let uut = u.dot(&u.t());
    let utu = u.t().dot(&u);
    let mut dev_span: f64 = 0.0;
    for (m, dim) in [(&uut, uut.nrows()), (&utu, utu.nrows())] {
        for r in 0..dim {
            for c in 0..dim {
                let target = if r == c { 1.0 } else { 0.0 };
                dev_span = dev_span.max((m[(r, c)] - target).abs());
            }
        }
    }
    (dev_orth, dev_span)
}

// ---------------------------------------------------------------------------
// ragged per-pair domain-block PCG solver
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

/// Amplitude-threshold local MP2 with the VV-HV space built internally.
pub fn amplitude_lmp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLmp2Config,
) -> Result<AmplitudeLmp2Result, FerricError> {
    let (vvhv, t_spaces_s) = checked_vvhv(mol, obs, obs_bs, rhf, "lmp2_amplitude")?;
    let mut r = amplitude_lmp2_with_virtuals(mol, obs, dfbs, op, rhf, cfg, &vvhv)?;
    r.timings.t_spaces_s = t_spaces_s;
    Ok(r)
}

/// `build_vvhv` + the orthonormality/span check (both under the 1e-8 bar),
/// returning the space and the wall seconds both took. `who` names the
/// driver in the error.
pub fn checked_vvhv(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    rhf: &ScfResult,
    who: &str,
) -> Result<(VvHv, f64), FerricError> {
    let t0 = std::time::Instant::now();
    let vvhv = build_vvhv(mol, obs, obs_bs, rhf)?;
    let nocc_total = (mol.nelec() as usize) / 2;
    let (dev_orth, dev_span) = check_vvhv(obs, rhf, nocc_total, &vvhv.c_vloc);
    if dev_orth > 1e-8 || dev_span > 1e-8 {
        return Err(FerricError::General(format!(
            "{who}: VV-HV construction check failed (orth {dev_orth:.2e}, span {dev_span:.2e})"
        )));
    }
    Ok((vvhv, t0.elapsed().as_secs_f64()))
}

/// The assembled localized-basis problem: everything the solver consumes.
/// Public so tests can run an INDEPENDENT naive dense reference on the
/// exact same inputs (the ragged-vs-dense cross-check).
#[derive(Debug)]
pub struct LocalizedProblem {
    /// (no·nv, no·nv), row index i·nv+a — (ia|jb) in the localized basis.
    pub j_dense: Array2<f64>,
    pub f_oo: Array2<f64>,
    pub f_vv: Array2<f64>,
    pub no: usize,
    pub nv: usize,
    /// Boys-localized active occupied coefficients, (nao, no) — exposed so
    /// downstream methods (LinLCCD ladders) can transform additional blocks
    /// in the SAME localized basis.
    pub c_locc: Array2<f64>,
    /// Boys centroids of the active localized occupieds, (no, 3) Bohr.
    pub occ_centers: Array2<f64>,
    /// Orbital spreads σᵢ = sqrt(⟨r²⟩ − |⟨r⟩|²), Bohr.
    pub occ_spreads: Vec<f64>,
    /// RAII charge against the shared memory pool for `j_dense`, held for as
    /// long as this struct is.
    ///
    /// Before the pool migration, `lmp2_amplitude.rs` had ZERO memory guards
    /// of any kind: the config's `eri3_budget_bytes` reached exactly one
    /// consumer (`eri3_mo_ov_blocked`), and `j_dense` -- an `(no·nv)²` dense
    /// matrix, the largest object on this path and the one every downstream
    /// LinLCCD / dRPA consumer iterates against -- was allocated with no
    /// check, no warning and no log line.
    ///
    /// The guard is a struct field rather than a scoped local because
    /// `j_dense` escapes into this struct and outlives its builder; a scoped
    /// guard would credit the pool back while the matrix is resident for the
    /// whole amplitude solve.
    ///
    /// Inert when no pool is installed.
    pub(crate) _charge: ferric_core::memory::pool::Reservation,
}

/// The basis-stage products (everything BEFORE any 4-index work): localized
/// coefficients, Fock blocks, centroids/spreads (the pair gate's inputs),
/// and the raw + whitening RI pieces.
#[derive(Debug, Clone)]
pub struct LocalizedBasis {
    pub c_locc: Array2<f64>,
    pub f_oo: Array2<f64>,
    pub f_vv: Array2<f64>,
    pub occ_centers: Array2<f64>,
    pub occ_spreads: Vec<f64>,
    pub no: usize,
    pub nv: usize,
    /// UNWHITENED (P|ia), (naux, no·nv).
    pub b_flat: Array2<f64>,
    /// 2-center metric (P|Q).
    pub v2c: Array2<f64>,
}

/// Everything the correlation stage needs about the LOCALIZED spaces, with
/// no integral tensor attached — the shared front end of both the global
/// (`assemble_basis`) and integral-direct (`crate::lmp2_direct`) assemblies.
#[derive(Debug, Clone)]
pub struct LocalizedSpaces {
    pub c_locc: Array2<f64>,
    /// Copy of the caller's VV-HV virtuals, (nao, nv).
    pub c_vloc: Array2<f64>,
    pub f_oo: Array2<f64>,
    pub f_vv: Array2<f64>,
    pub occ_centers: Array2<f64>,
    pub occ_spreads: Vec<f64>,
    /// Dipole centroids ⟨a|r|a⟩ of the localized virtuals, (nv, 3) Bohr —
    /// the distance rule for the direct path's virtual domains.
    pub virt_centers: Array2<f64>,
    pub no: usize,
    pub nv: usize,
}

/// Localized-space stage shared by all assembly paths: Boys occupieds,
/// centroids/spreads, Fock blocks, virtual centroids. No RI work here.
pub fn localized_spaces(
    mol: &Molecule,
    obs: &PreparedBasis,
    rhf: &ScfResult,
    frozen_core: usize,
    vvhv: &VvHv,
) -> Result<LocalizedSpaces, FerricError> {
    let nocc_total = (mol.nelec() as usize) / 2;
    let no = active_occ(nocc_total, frozen_core)?;
    let first_occ = frozen_core;
    let c_occ_can = rhf.mos_r().slice(s![.., first_occ..nocc_total]).to_owned();
    let dip = dipole(obs, [0.0, 0.0, 0.0])?;
    let boys = boys_localize(&c_occ_can, &dip, 200);
    let c_locc = boys.c_loc;
    let occ_centers = boys.centers;
    let r2 = r2_moment(obs, [0.0; 3])?;
    let mut occ_spreads = Vec::with_capacity(no);
    for i in 0..no {
        let col = c_locc.column(i);
        let r2v = col.dot(&r2.dot(&col));
        let c2: f64 = (0..3).map(|x| occ_centers[(i, x)].powi(2)).sum();
        occ_spreads.push((r2v - c2).max(1e-10).sqrt());
    }
    let c_vloc = vvhv.c_vloc.clone();
    let nv = c_vloc.ncols();
    let mut virt_centers = Array2::<f64>::zeros((nv, 3));
    for a in 0..nv {
        let col = c_vloc.column(a);
        for (x, dm) in dip.iter().enumerate() {
            virt_centers[(a, x)] = col.dot(&dm.dot(&col));
        }
    }
    let f_ao = rhf.fock_r();
    let f_oo = c_locc.t().dot(&f_ao.dot(&c_locc));
    let f_vv = c_vloc.t().dot(&f_ao.dot(&c_vloc));
    Ok(LocalizedSpaces {
        c_locc,
        c_vloc,
        f_oo,
        f_vv,
        occ_centers,
        occ_spreads,
        virt_centers,
        no,
        nv,
    })
}

/// Basis stage of the assembly — no (no·nv)² object is ever formed here.
pub fn assemble_basis(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLmp2Config,
    vvhv: &VvHv,
) -> Result<LocalizedBasis, FerricError> {
    let sp = localized_spaces(mol, obs, rhf, cfg.frozen_core, vvhv)?;
    let LocalizedSpaces {
        c_locc,
        f_oo,
        f_vv,
        occ_centers,
        occ_spreads,
        no,
        nv,
        ..
    } = sp;
    let c_vloc = &vvhv.c_vloc;
    let budget = eri3_budget_bytes(cfg.eri3_budget_bytes);
    let b3 = eri3_mo_ov_blocked(op, obs, dfbs, &c_locc, c_vloc, budget)?;
    let naux = b3.shape()[0];
    let v2c = coulomb_metric_2c(op, dfbs)?;
    let b_flat = b3
        .into_shape_with_order((naux, no * nv))
        .map_err(|e| FerricError::General(format!("lmp2_amplitude reshape: {e}")))?;
    Ok(LocalizedBasis {
        c_locc,
        f_oo,
        f_vv,
        occ_centers,
        occ_spreads,
        no,
        nv,
        b_flat,
        v2c,
    })
}

/// Assemble the localized-basis problem (Boys occupieds, caller's virtuals,
/// RI J via the same helpers the canonical reference uses).
pub fn assemble_localized(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLmp2Config,
    vvhv: &VvHv,
) -> Result<LocalizedProblem, FerricError> {
    let lb = assemble_basis(mol, obs, dfbs, op, rhf, cfg, vvhv)?;
    let (no, nv) = (lb.no, lb.nv);
    // DEBIT the shared pool for `j_dense` BEFORE building it -- the point of a
    // gate is to answer "does this fit?" before the allocation, not after.
    //
    // Charged for `j_dense` alone and not for the `btilde` transient below:
    // `btilde` is (naux, no*nv), which is naux/(no*nv) times SMALLER, and it
    // dies at the end of the match arm. Charging a transient for the struct's
    // lifetime is the over-estimating-guard bug. `lb.b_flat` is charged by
    // `eri3_mo_ov_blocked`'s own path inside `assemble_basis`.
    //
    // HARD: `j_dense` is dense by construction on this path -- the ragged /
    // domain-blocked alternative is a DIFFERENT entry point
    // (`assemble_ragged_direct`), chosen by the caller, not a fallback this
    // function can take. A caller who cannot afford the dense matrix has a
    // real alternative, and refusing here with an occupancy breakdown naming
    // the plane is how they find out, rather than being OOM-killed.
    let _j_dense_charge = crate::rimp2::charge_mo_side(
        &format!("LMP2 j_dense (no={no}, nv={nv}; dense (no*nv)^2 localized ERI)"),
        no.saturating_mul(nv)
            .saturating_mul(no)
            .saturating_mul(nv)
            .saturating_mul(8),
    )?;
    let j_dense = match cfg.fit_radius_bohr {
        None => {
            let vis = metric_inverse_sqrt(&lb.v2c, op)?;
            let btilde = vis.dot(&lb.b_flat); // (naux, no*nv)
            btilde.t().dot(&btilde) // (no*nv, no*nv)
        }
        Some(r) => domain_fit_j(&lb.b_flat, &lb.v2c, mol, dfbs, &lb.occ_centers, no, nv, r)?,
    };
    Ok(LocalizedProblem {
        j_dense,
        f_oo: lb.f_oo,
        f_vv: lb.f_vv,
        no,
        nv,
        c_locc: lb.c_locc,
        occ_centers: lb.occ_centers,
        occ_spreads: lb.occ_spreads,
        _charge: _j_dense_charge,
    })
}

/// Per-pair domain-local same-kernel RI fit: J_ij = A_{i,D} V_DD⁻¹ A_{j,D}
/// with D = aux functions within `radius` Bohr of either pair centroid.
/// With one shared domain and the same-kernel metric the Dunlap robust
/// correction cancels identically (first-order fit error is zero by
/// construction) — the formulation the ferric domain-fitting lane validated.
#[allow(clippy::too_many_arguments)]
fn domain_fit_j(
    b_flat: &Array2<f64>, // (naux, no*nv), UNWHITENED (P|ia)
    v: &Array2<f64>,
    mol: &Molecule,
    dfbs: &PreparedBasis,
    occ_centers: &Array2<f64>,
    no: usize,
    nv: usize,
    radius: f64,
) -> Result<Array2<f64>, FerricError> {
    let naux = b_flat.nrows();
    // aux function -> parent atom position
    let aux_atom = ao_to_atom(dfbs);
    let mut aux_xyz = Array2::<f64>::zeros((naux, 3));
    for (p, &ai) in aux_atom.iter().enumerate() {
        let at = &mol.atoms[ai];
        aux_xyz[(p, 0)] = at.x;
        aux_xyz[(p, 1)] = at.y;
        aux_xyz[(p, 2)] = at.zpos;
    }
    let r2 = radius * radius;
    let mut j_dense = Array2::<f64>::zeros((no * nv, no * nv));
    for i in 0..no {
        for j in i..no {
            let blk = domain_fit_pair(b_flat, v, &aux_xyz, occ_centers, i, j, nv, r2, radius)?;
            for a in 0..nv {
                for b in 0..nv {
                    j_dense[(i * nv + a, j * nv + b)] = blk[(a, b)];
                    j_dense[(j * nv + b, i * nv + a)] = blk[(a, b)];
                }
            }
        }
    }
    Ok(j_dense)
}

/// One pair's domain-local same-kernel fit block J_ij (nv, nv) — see
/// [`domain_fit_j`] for the formulation notes.
#[allow(clippy::too_many_arguments)]
fn domain_fit_pair(
    b_flat: &Array2<f64>,
    v: &Array2<f64>,
    aux_xyz: &Array2<f64>,
    occ_centers: &Array2<f64>,
    i: usize,
    j: usize,
    nv: usize,
    r2: f64,
    radius: f64,
) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::InverseC;
    let naux = b_flat.nrows();
    let dist2 = |p: usize, k: usize| -> f64 {
        (0..3)
            .map(|x| (aux_xyz[(p, x)] - occ_centers[(k, x)]).powi(2))
            .sum()
    };
    let dom: Vec<usize> = (0..naux)
        .filter(|&p| dist2(p, i) <= r2 || dist2(p, j) <= r2)
        .collect();
    if dom.is_empty() {
        return Err(FerricError::General(format!(
            "lmp2_amplitude: empty aux domain for pair ({i},{j}) at radius {radius} Bohr"
        )));
    }
    let d = dom.len();
    let mut vdd = Array2::<f64>::zeros((d, d));
    for (r_, &pp) in dom.iter().enumerate() {
        for (c_, &qq) in dom.iter().enumerate() {
            vdd[(r_, c_)] = v[(pp, qq)];
        }
    }
    // SPD same-kernel metric: Cholesky-based inverse (faster + tighter than LU)
    let vdd_inv = vdd
        .invc()
        .map_err(|e| FerricError::General(format!("lmp2_amplitude V_DD Cholesky inverse: {e}")))?;
    let mut a_i = Array2::<f64>::zeros((d, nv));
    let mut a_j = Array2::<f64>::zeros((d, nv));
    for (r_, &pp) in dom.iter().enumerate() {
        for a in 0..nv {
            a_i[(r_, a)] = b_flat[(pp, i * nv + a)];
            a_j[(r_, a)] = b_flat[(pp, j * nv + a)];
        }
    }
    Ok(a_i.t().dot(&vdd_inv.dot(&a_j)))
}

/// The integral-free R⁻⁶ pair gate (paper's linked gate): returns the
/// (no·no) keep mask and the count of gated UNIQUE off-diagonal pairs.
/// Symmetric by construction; diagonal pairs are never gated; ε = 0 makes
/// the gate inert (θ = 0). Extracted so every assembly path applies the
/// EXACT same rule.
pub fn pair_gate_keep(
    occ_centers: &Array2<f64>,
    occ_spreads: &[f64],
    no: usize,
    eps: f64,
    cal: f64,
) -> (Vec<bool>, usize) {
    pair_gate_keep_with(
        |i, j| {
            (0..3)
                .map(|x| (occ_centers[(i, x)] - occ_centers[(j, x)]).powi(2))
                .sum()
        },
        occ_spreads,
        no,
        eps,
        cal,
    )
}

/// [`pair_gate_keep`] with a caller-supplied SQUARED centroid distance
/// `rij2(i, j)` (Bohr²). The molecular gate passes the Euclidean distance
/// (the closure above is the exact expression the gate always used, so the
/// molecular path is byte-identical — pinned by
/// `pair_gate_keep_with_euclid_is_bitwise_pair_gate_keep`); ferric-pbc's
/// Gamma LMP2 passes the MINIMUM-IMAGE distance (a raw Cartesian distance
/// across a supercell boundary is wrong there).
pub fn pair_gate_keep_with<D>(
    rij2: D,
    occ_spreads: &[f64],
    no: usize,
    eps: f64,
    cal: f64,
) -> (Vec<bool>, usize)
where
    D: Fn(usize, usize) -> f64,
{
    let theta = 1e-2 * eps;
    let mut keep = vec![true; no * no];
    let mut n_gated = 0usize;
    for i in 0..no {
        for j in 0..no {
            if i == j {
                continue;
            }
            let rij2: f64 = rij2(i, j);
            let est = cal * (occ_spreads[i] * occ_spreads[j]).powi(3) / rij2.powi(3).max(1e-12);
            if est < theta {
                keep[i * no + j] = false;
                if i < j {
                    n_gated += 1;
                }
            }
        }
    }
    (keep, n_gated)
}

/// The LINEAR energy form E = Σ (2 t_iajb − t_ibja) J_iajb over the
/// retained pattern — t_ibja is read from the SAME pair block at swapped
/// positions (zero outside the union domains). Exact at the converged
/// amplitudes but NOT stationary: its error is first order in the CG
/// residual. It was the molecular drivers' energy until perf Stage D and is
/// kept, arithmetic unchanged, as the oracle of [`hylleraas_energy`]
/// (`hylleraas_is_stationary`). ferric-pbc's Gamma LMP2 still uses it
/// (its numbers are not changed by Stage D).
pub fn linear_energy(rg: &Ragged, t: &[Array2<f64>]) -> f64 {
    swap_contract(rg, t, |p, r, c| rg.pairs[p].j_blk[(r, c)])
}

/// Σ over the retained pattern of (2 t_iajb − t_ibja)·w_iajb, with the same
/// accumulation order as the original linear energy (direct and exchange
/// sums kept apart, combined once at the end).
fn swap_contract<W>(rg: &Ragged, t: &[Array2<f64>], w: W) -> f64
where
    W: Fn(usize, usize, usize) -> f64,
{
    let mut e_dir = 0.0;
    let mut e_exx = 0.0;
    for (p, pb) in rg.pairs.iter().enumerate() {
        let nb = pb.db.len();
        for (r, &a) in pb.da.iter().enumerate() {
            for (c, &b) in pb.db.iter().enumerate() {
                if !pb.pat[r * nb + c] {
                    continue;
                }
                let jv = w(p, r, c);
                e_dir += t[p][(r, c)] * jv;
                let sr = pb.pos_da[b];
                let sc = pb.pos_db[a];
                if sr != usize::MAX && sc != usize::MAX {
                    e_exx += t[p][(sr, sc)] * jv;
                }
            }
        }
    }
    2.0 * e_dir - e_exx
}

/// The stationary Hylleraas energy
/// `E_H(t) = Σ (2 t_iajb − t_ibja)·(2 J_iajb + (A t)_iajb)` over the retained
/// pattern, where `A` is the pattern-projected Fock superoperator of the
/// solve (one extra matvec). With the residual R = J + A t this is
/// E_lin + Σ t̃·R, so it equals [`linear_energy`] at the solution; on a
/// swap-closed pattern (`crate::ragged::check_swap_closed`) its gradient is
/// 2(2 − S)R, so the error is second order in the residual and
/// E_H(t) − E* = ⟨(2 − S)δ, Aδ⟩ ≥ 0 (an upper bound).
///
/// `plan`: the [`SolvePlan`] the solve ran through, or `None` when it ran on
/// the plan-free indexed matvec (the budgeted solve's fallback) — the same
/// operator either way, bitwise when every Fvv product is dense.
///
/// Stationarity needs a swap-closed pattern, so this function CHECKS it
/// ([`check_swap_closed`], O(Σ|D|²) comparisons, small beside the matvec)
/// and errors instead of returning a non-stationary number.
pub fn hylleraas_energy(
    rg: &Ragged,
    f_oo: &Array2<f64>,
    plan: Option<&SolvePlan>,
    t: &[Array2<f64>],
) -> Result<f64, FerricError> {
    hylleraas_energy_and_residual(rg, f_oo, plan, t).map(|(e, _)| e)
}

/// [`hylleraas_energy`] plus the norm of the TRUE residual `‖J + A t‖` over the
/// pattern, read from the energy's own matvec (no extra one). The PCG's relres
/// is its recursive residual, which can drift from this one.
pub fn hylleraas_energy_and_residual(
    rg: &Ragged,
    f_oo: &Array2<f64>,
    plan: Option<&SolvePlan>,
    t: &[Array2<f64>],
) -> Result<(f64, f64), FerricError> {
    check_swap_closed(rg).map_err(|e| {
        FerricError::General(format!(
            "hylleraas_energy: amplitude pattern is not swap-closed ({e}); the Hylleraas \
             energy would not be stationary"
        ))
    })?;
    let mut flops = 0u64;
    let at = match plan {
        Some(plan) => {
            let mut at: Vec<Array2<f64>> = rg.pairs.iter().map(|_| Array2::zeros((0, 0))).collect();
            matvec_planned(rg, plan, t, &mut at, &mut flops);
            at
        }
        None => matvec_indexed(rg, f_oo, t, &mut flops),
    };
    let energy = swap_contract(rg, t, |p, r, c| {
        2.0 * rg.pairs[p].j_blk[(r, c)] + at[p][(r, c)]
    });
    let r2: f64 = rg
        .pairs
        .iter()
        .zip(&at)
        .map(|(pb, a)| {
            pb.j_blk
                .iter()
                .zip(a.iter())
                .zip(&pb.pat)
                .filter(|(_, &k)| k)
                .map(|((j, av), _)| (j + av) * (j + av))
                .sum::<f64>()
        })
        .sum();
    Ok((energy, r2.sqrt()))
}

/// `‖P J‖`: the 2-norm of the pattern-projected right-hand side over every
/// block — the `bnorm` the PCG's `relres` is relative to.
pub fn pattern_rhs_norm(rg: &Ragged) -> f64 {
    rg.pairs
        .iter()
        .map(|pb| {
            pb.j_blk
                .iter()
                .zip(&pb.pat)
                .filter(|(_, &k)| k)
                .map(|(j, _)| j * j)
                .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt()
}

/// A rigorous lower bound on λ_min of the pattern-projected PCG operator
/// `A`: `2·(ε_LUMO − ε_HOMO)` from the eigenvalues of the localized-space
/// Fock blocks (`f_vv` spans the whole virtual space, `f_oo` the active
/// occupieds). The unprojected operator's eigenvalues are ε_a + ε_b − ε_i −
/// ε_j ≥ 2(ε_LUMO − ε_HOMO), and the pattern/domain projection is a
/// compression, so by Cauchy interlacing its λ_min is no smaller. (The CG
/// Lanczos Ritz values are NOT used: the smallest Ritz value is an UPPER
/// bound on λ_min, and of the preconditioned operator.) NaN if not positive.
pub fn operator_lambda_min_lower_bound(
    f_oo: &Array2<f64>,
    f_vv: &Array2<f64>,
) -> Result<f64, FerricError> {
    let (eo, _) = eigh(f_oo)?;
    let (ev, _) = eigh(f_vv)?;
    let (Some(&homo), Some(&lumo)) = (eo.last(), ev.first()) else {
        return Ok(f64::NAN);
    };
    let lb = 2.0 * (lumo - homo);
    Ok(if lb > 0.0 { lb } else { f64::NAN })
}

/// A-priori bound on the Hylleraas energy's solver error at the achieved
/// residual: `E_H(t) − E* = ⟨(2 − S)A⁻¹R, R⟩ ≤ 3·‖R‖²/λ_min(A)`, with the
/// swap `S` (eigenvalues ±1, so ‖2 − S‖ = 3) commuting with `A`, and
/// ‖R‖ = relres·‖PJ‖. Hence `3·‖PJ‖²·relres²/λ_min`, with λ_min from
/// [`operator_lambda_min_lower_bound`]. Holds for the recursive CG residual
/// the solve reports (to rounding).
pub fn cg_energy_error_bound(rhs_norm: f64, relres: f64, lambda_min: f64) -> f64 {
    3.0 * (rhs_norm * relres).powi(2) / lambda_min
}

/// The tolerance to continue to when the bound at `relres` exceeds `target_ha`:
/// the residual that puts the bound at a quarter of the target, capped at
/// `floor`. `None` when the bound already holds, or when the residual is
/// already at the floor (the target is then unreachable; the caller reports it).
pub fn tightened_cg_rtol(
    relres: f64,
    rhs_norm: f64,
    lambda_min: f64,
    floor: f64,
    target_ha: f64,
) -> Option<f64> {
    if cg_energy_error_bound(rhs_norm, relres, lambda_min) <= target_ha {
        return None;
    }
    let needed = CG_TIGHTEN_SAFETY * (target_ha * lambda_min / 3.0).sqrt() / rhs_norm;
    let next = needed.max(floor);
    (relres >= next).then_some(next)
}

/// What [`solve_under_rtol_policy`] returns; its PCG always converged.
pub struct PolicySolve {
    pub budgeted: BudgetedSolve,
    pub outcome: CgRtolOutcome,
    /// The stationary energy of the final amplitudes.
    pub e_hylleraas: f64,
    /// `‖J + A t‖ / ‖PJ‖` from the energy's matvec (0 when `‖PJ‖` is 0).
    pub true_relres: f64,
    /// [`cg_energy_error_bound`] at `true_relres` (NaN with no positive gap).
    pub bound: f64,
}

/// The budgeted ragged solve under a tolerance policy, its energy, and the
/// bound at the final TRUE residual. Tightening is steered by the PCG's
/// recursive residual (a stop costs no matvec) and continues the same Krylov
/// sequence, so its result is bitwise a straight solve at the final tolerance.
/// The target verdict uses the true residual. If that misses where the
/// recursive one passed, the run is reported as not meeting the target and is
/// not iterated further: the two differ only once the recursion has drifted,
/// and more steps of the same recursion would not reduce the true residual.
/// The same holds at the floor: a target that needs more than the floor is
/// reported through `target_met`, not as an error.
///
/// Running out of iterations is an error whichever tolerance was being sought:
/// the first one, or a tighter one the bound asked for. The message of the
/// second names the residual, the tolerance, the iterations, the bound and the
/// remedy.
pub fn solve_under_rtol_policy(
    rg: &Ragged,
    f_oo: &Array2<f64>,
    f_vv: &Array2<f64>,
    policy: CgRtolPolicy,
    max_iter: usize,
    budget_bytes: usize,
    fvv_rho_star: f64,
) -> Result<PolicySolve, FerricError> {
    let lambda_min = operator_lambda_min_lower_bound(f_oo, f_vv)?;
    let rhs_norm = pattern_rhs_norm(rg);
    let (start, source, governing) = match policy {
        CgRtolPolicy::Explicit(x) => (x, CgRtolSource::Explicit, None),
        CgRtolPolicy::Adaptive { floor, .. } if lambda_min.is_nan() => {
            (floor, CgRtolSource::StrictNoGap, None)
        }
        CgRtolPolicy::Adaptive {
            start,
            floor,
            target_ha,
        } => (
            start,
            CgRtolSource::AdaptiveDefault,
            Some((floor, target_ha)),
        ),
    };
    let (mut rtol_used, mut tightened) = (start, false);
    let b = solve_ragged_budgeted_steered(
        rg,
        f_oo,
        start,
        max_iter,
        budget_bytes,
        fvv_rho_star,
        |relres| {
            let (floor, target_ha) = governing?;
            let next = tightened_cg_rtol(relres, rhs_norm, lambda_min, floor, target_ha)?;
            rtol_used = next;
            tightened = true;
            Some(next)
        },
    );
    let (e_hylleraas, r_true) = hylleraas_energy_and_residual(rg, f_oo, b.plan(), &b.solve.t)?;
    let true_relres = if rhs_norm > 0.0 {
        r_true / rhs_norm
    } else {
        0.0
    };
    let bound = cg_energy_error_bound(rhs_norm, true_relres, lambda_min);
    let outcome = CgRtolOutcome {
        source,
        rtol_used,
        tightened,
        target_ha: governing.map(|(_, t)| t),
        target_met: governing.map(|(_, t)| bound <= t),
    };
    if !b.solve.converged {
        return Err(FerricError::General(unconverged_message(
            b.solve.relres,
            b.solve.iterations,
            &outcome,
            true_relres,
            bound,
        )));
    }
    Ok(PolicySolve {
        budgeted: b,
        outcome,
        e_hylleraas,
        true_relres,
        bound,
    })
}

/// Why a PCG that ran out of iterations failed the solve.
fn unconverged_message(
    relres: f64,
    iterations: usize,
    o: &CgRtolOutcome,
    true_relres: f64,
    bound: f64,
) -> String {
    if !o.tightened {
        return format!(
            "ragged CG failed to converge (relres {relres:.2e} after {iterations} iters)"
        );
    }
    format!(
        "ragged CG ran out of iterations while tightening: relres {relres:.2e} (true \
         {true_relres:.2e}) after {iterations} iters, seeking cg_rtol {:.2e}; the solver-error \
         bound is still {bound:.2e} Ha against the {:.0e} Ha target. Raise \
         AmplitudeLmp2Config::cg_max_iter, or set an explicit AmplitudeLmp2Config::cg_rtol to \
         accept the looser tolerance",
        o.rtol_used,
        o.target_ha.unwrap_or(f64::NAN),
    )
}

/// `fvv_sparse_rho` must be a finite ρ* ≥ 0 (a density threshold): NaN or a
/// negative value is a configuration error, not a silent "all dense".
pub fn validate_fvv_sparse_rho(rho: f64) -> Result<(), FerricError> {
    if rho.is_nan() || rho < 0.0 {
        return Err(FerricError::General(format!(
            "fvv_sparse_rho must be >= 0 (a pattern-density threshold), got {rho}"
        )));
    }
    Ok(())
}

/// Direct (never-dense) ragged assembly shared by the amplitude-threshold
/// family: integral-free pair gate BEFORE any GEMM/fit, per-unique-pair
/// whitened Gram GEMMs over exact-Schwarz candidate subsets (or per-pair
/// domain-local fits), transposes for the mirror pairs, rayon-parallel
/// with deterministic (i, j)-ordered collection. `scale` multiplies every
/// block (dRPA uses 2.0 for B = 2(ia|jb)); the Eq-8 threshold and the
/// Schwarz bound both act on the SCALED integrals, so eps means the same
/// thing to every consumer. Returns (ragged blocks, gated unique pairs).
#[allow(clippy::too_many_arguments)]
pub fn assemble_ragged_direct(
    mol: &Molecule,
    dfbs: &PreparedBasis,
    op: Operator,
    lb: &LocalizedBasis,
    eps: f64,
    scale: f64,
    pair_gate_cal: Option<f64>,
    fit_radius_bohr: Option<f64>,
) -> Result<(Ragged, usize), FerricError> {
    assemble_ragged_direct_aux(
        mol,
        dfbs,
        op,
        lb,
        eps,
        scale,
        pair_gate_cal,
        fit_radius_bohr,
        None,
    )
    .map(|(rg, g, _, _)| (rg, g))
}

/// [`assemble_ragged_direct`] with the ε-linked per-pair AUX truncation
/// (see `AmplitudeLmp2Config::aux_tail_frac`). Returns
/// `(ragged, gated_pairs, aux_dom_mean, aux_dom_max)` — the last two are
/// (naux, naux) when truncation is off.
#[allow(clippy::too_many_arguments)]
pub fn assemble_ragged_direct_aux(
    mol: &Molecule,
    dfbs: &PreparedBasis,
    op: Operator,
    lb: &LocalizedBasis,
    eps: f64,
    scale: f64,
    pair_gate_cal: Option<f64>,
    fit_radius_bohr: Option<f64>,
    aux_tail_frac: Option<f64>,
) -> Result<(Ragged, usize, f64, usize), FerricError> {
    let (no, nv) = (lb.no, lb.nv);
    let (f_oo, f_vv) = (&lb.f_oo, &lb.f_vv);
    let mut n_pairs_gated = 0usize;
    let keep_pair: Option<Vec<bool>> = pair_gate_cal.map(|cal| {
        let (keep, gated) = pair_gate_keep(&lb.occ_centers, &lb.occ_spreads, no, eps, cal);
        n_pairs_gated = gated;
        keep
    });

    let fo: Vec<f64> = (0..no).map(|i| f_oo[(i, i)]).collect();
    let fv: Vec<f64> = (0..nv).map(|a| f_vv[(a, a)]).collect();
    let (btilde, aux_xyz) = match fit_radius_bohr {
        None => {
            let vis = metric_inverse_sqrt(&lb.v2c, op)?;
            (Some(vis.dot(&lb.b_flat)), None)
        }
        Some(_) => {
            let aux_atom = ao_to_atom(dfbs);
            let naux = lb.b_flat.nrows();
            let mut xyz = Array2::<f64>::zeros((naux, 3));
            for (pp, &ai) in aux_atom.iter().enumerate() {
                let at = &mol.atoms[ai];
                xyz[(pp, 0)] = at.x;
                xyz[(pp, 1)] = at.y;
                xyz[(pp, 2)] = at.zpos;
            }
            (None, Some(xyz))
        }
    };
    // Schwarz on the SCALED integrals: |scale·J| <= (√s·q)_ia (√s·q)_jb
    let sq = scale.sqrt();
    let q: Option<(Vec<f64>, Vec<f64>)> = btilde.as_ref().map(|bt| {
        let mut qv = vec![0.0f64; no * nv];
        for (col, qslot) in qv.iter_mut().enumerate() {
            *qslot = sq * bt.column(col).dot(&bt.column(col)).sqrt();
        }
        let qmax: Vec<f64> = (0..no)
            .map(|i| {
                qv[i * nv..(i + 1) * nv]
                    .iter()
                    .cloned()
                    .fold(0.0f64, f64::max)
            })
            .collect();
        (qv, qmax)
    });
    // per-(P, i) row weights for the aux tail bound: n[P, i] = max_a |B̃_Pia|
    let n_block: Option<Array2<f64>> = match (&btilde, aux_tail_frac, eps > 0.0) {
        (Some(bt), Some(f), true) if f > 0.0 => {
            let naux = bt.nrows();
            let mut nb = Array2::<f64>::zeros((naux, no));
            for pp in 0..naux {
                for i in 0..no {
                    let mut m = 0.0f64;
                    for a in 0..nv {
                        m = m.max(bt[(pp, i * nv + a)].abs());
                    }
                    nb[(pp, i)] = m;
                }
            }
            Some(nb)
        }
        _ => None,
    };
    let tail_budget = aux_tail_frac.map(|f| f * eps);
    let unique: Vec<(usize, usize)> = (0..no)
        .flat_map(|i| (i..no).map(move |j| (i, j)))
        .filter(|&(i, j)| keep_pair.as_ref().map_or(true, |kp| kp[i * no + j]))
        .collect();
    use rayon::prelude::*;
    type PairOut = (Vec<PairBlock>, usize);
    let per_pair: Vec<Result<PairOut, FerricError>> = unique
        .par_iter()
        .map(|&(i, j)| {
            let cand: Vec<usize> = match (&q, eps > 0.0) {
                (Some((qv, qmax)), true) => (0..nv)
                    .filter(|&a| qv[i * nv + a] * qmax[j] >= eps || qv[j * nv + a] * qmax[i] >= eps)
                    .collect(),
                _ => (0..nv).collect(),
            };
            if cand.is_empty() {
                return Ok((Vec::new(), 0));
            }
            let mut aux_used = 0usize;
            let mut g: Array2<f64> = match (&btilde, &aux_xyz, fit_radius_bohr) {
                (Some(bt), _, _) => {
                    let naux = bt.nrows();
                    // ε-linked aux selection: smallest prefix (by n_i·n_j)
                    // whose dropped tail-sum ≤ f·ε — exact elementwise bound
                    let aux_sel: Option<Vec<usize>> = match (&n_block, tail_budget) {
                        (Some(nb), Some(budget)) => {
                            let mut scored: Vec<(f64, usize)> = (0..naux)
                                .map(|pp| (nb[(pp, i)] * nb[(pp, j)], pp))
                                .collect();
                            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("NaN aux score"));
                            let total: f64 = scored.iter().map(|x| x.0).sum();
                            let mut kept = Vec::with_capacity(naux);
                            let mut acc = 0.0;
                            for &(sc, pp) in &scored {
                                if total - acc <= budget {
                                    break;
                                }
                                kept.push(pp);
                                acc += sc;
                            }
                            kept.sort_unstable();
                            Some(kept)
                        }
                        _ => None,
                    };
                    let nc = cand.len();
                    match aux_sel {
                        None => {
                            aux_used = naux;
                            if nc == nv {
                                let bi = bt.slice(s![.., i * nv..(i + 1) * nv]);
                                let bj = bt.slice(s![.., j * nv..(j + 1) * nv]);
                                bi.t().dot(&bj)
                            } else {
                                let mut bi = Array2::<f64>::zeros((naux, nc));
                                let mut bj = Array2::<f64>::zeros((naux, nc));
                                for (k, &a) in cand.iter().enumerate() {
                                    bi.column_mut(k).assign(&bt.column(i * nv + a));
                                    bj.column_mut(k).assign(&bt.column(j * nv + a));
                                }
                                bi.t().dot(&bj)
                            }
                        }
                        Some(sel) => {
                            aux_used = sel.len();
                            let nd = sel.len();
                            let mut bi = Array2::<f64>::zeros((nd, nc));
                            let mut bj = Array2::<f64>::zeros((nd, nc));
                            for (r_, &pp) in sel.iter().enumerate() {
                                for (k, &a) in cand.iter().enumerate() {
                                    bi[(r_, k)] = bt[(pp, i * nv + a)];
                                    bj[(r_, k)] = bt[(pp, j * nv + a)];
                                }
                            }
                            bi.t().dot(&bj)
                        }
                    }
                }
                (_, Some(xyz), Some(radius)) => domain_fit_pair(
                    &lb.b_flat,
                    &lb.v2c,
                    xyz,
                    &lb.occ_centers,
                    i,
                    j,
                    nv,
                    radius * radius,
                    radius,
                )?,
                _ => unreachable!("btilde/aux_xyz exactly one is Some"),
            };
            if scale != 1.0 {
                g.mapv_inplace(|x| scale * x);
            }
            let cand_used: Vec<usize> = if g.nrows() == nv {
                (0..nv).collect()
            } else {
                cand.clone()
            };
            let mut out = Vec::with_capacity(2);
            if let Some(pb) = pair_block_from_g_cand(i, j, &g, &cand_used, nv, f_vv, &fo, &fv, eps)
            {
                out.push(pb);
            }
            if i != j {
                let gt = g.t().to_owned();
                if let Some(pb) =
                    pair_block_from_g_cand(j, i, &gt, &cand_used, nv, f_vv, &fo, &fv, eps)
                {
                    out.push(pb);
                }
            }
            Ok((out, aux_used))
        })
        .collect();
    let mut pairs: Vec<PairBlock> = Vec::new();
    let mut aux_sizes: Vec<usize> = Vec::new();
    for r in per_pair {
        let (blocks, aux_used) = r?;
        if !blocks.is_empty() {
            aux_sizes.push(aux_used);
        }
        pairs.extend(blocks);
    }
    let mut by_i: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut by_j: HashMap<usize, Vec<usize>> = HashMap::new();
    for (pidx, pb) in pairs.iter().enumerate() {
        by_i.entry(pb.i).or_default().push(pidx);
        by_j.entry(pb.j).or_default().push(pidx);
    }
    let aux_dom_mean = if aux_sizes.is_empty() {
        0.0
    } else {
        aux_sizes.iter().sum::<usize>() as f64 / aux_sizes.len() as f64
    };
    let aux_dom_max = aux_sizes.iter().copied().max().unwrap_or(0);
    Ok((
        Ragged { pairs, by_i, by_j },
        n_pairs_gated,
        aux_dom_mean,
        aux_dom_max,
    ))
}

/// Same as [`amplitude_lmp2`], with a caller-supplied virtual space — the
/// mutation-test entry point (a deliberately broken space must fail the
/// ε=0 anchor).
pub fn amplitude_lmp2_with_virtuals(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLmp2Config,
    vvhv: &VvHv,
) -> Result<AmplitudeLmp2Result, FerricError> {
    // ---- basis stage: NO 4-index object is formed here ----
    let t0 = std::time::Instant::now();
    let lb = assemble_basis(mol, obs, dfbs, op, rhf, cfg, vvhv)?;
    let (no, nv) = (lb.no, lb.nv);
    let f_oo = &lb.f_oo;

    let (rg, n_pairs_gated, aux_dom_mean, aux_dom_max) = assemble_ragged_direct_aux(
        mol,
        dfbs,
        op,
        &lb,
        cfg.eps,
        1.0,
        cfg.pair_gate_cal,
        cfg.fit_radius_bohr,
        cfg.aux_tail_frac,
    )?;
    let t_assembly_s = t0.elapsed().as_secs_f64();

    // canonical reference on the same (mol, basis, aux, op, frozen core) —
    // OPT-IN (the honesty printout; off by default, see `compute_reference`)
    let t0 = std::time::Instant::now();
    let e_ref = if cfg.compute_reference {
        ri_mp2(
            mol,
            obs,
            dfbs,
            op,
            rhf,
            &RiMp2Config {
                frozen_core: cfg.frozen_core,
                memory_budget_bytes: cfg.eri3_budget_bytes,
                ..Default::default()
            },
        )?
        .mp2_corr
    } else {
        f64::NAN
    };
    let t_reference_s = if cfg.compute_reference {
        t0.elapsed().as_secs_f64()
    } else {
        0.0
    };

    // ---- ragged solve ----
    let t0 = std::time::Instant::now();
    let mut r = solve_and_count(
        &rg,
        f_oo,
        &lb.f_vv,
        rhf.energy,
        cfg,
        vvhv,
        no,
        nv,
        "lmp2_amplitude",
    )?;
    r.e_corr_canonical_ri = e_ref;
    r.n_pairs_gated = n_pairs_gated;
    r.aux_dom_mean = aux_dom_mean;
    r.aux_dom_max = aux_dom_max;
    r.timings = StageTimings {
        t_spaces_s: 0.0,
        t_assembly_s,
        t_solve_s: t0.elapsed().as_secs_f64(),
        t_reference_s,
    };
    Ok(r)
}

/// The ragged PCG solve + Hylleraas energy + pattern counters shared by
/// both LMP2 drivers. The caller fills the reference energy, the gate/aux
/// counters and the timings (left NaN/0 here).
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_and_count(
    rg: &Ragged,
    f_oo: &Array2<f64>,
    f_vv: &Array2<f64>,
    e_scf: f64,
    cfg: &AmplitudeLmp2Config,
    vvhv: &VvHv,
    no: usize,
    nv: usize,
    who: &str,
) -> Result<AmplitudeLmp2Result, FerricError> {
    validate_fvv_sparse_rho(cfg.fvv_sparse_rho)?;
    // fail fast, before the solve (hylleraas_energy checks it again)
    check_swap_closed(rg).map_err(|e| {
        FerricError::General(format!(
            "{who}: amplitude pattern is not swap-closed ({e}); the Hylleraas energy \
             would not be stationary"
        ))
    })?;
    let budget = eri3_budget_bytes(cfg.eri3_budget_bytes);
    let PolicySolve {
        budgeted: b,
        outcome,
        e_hylleraas: e_corr,
        true_relres,
        bound,
    } = solve_under_rtol_policy(
        rg,
        f_oo,
        f_vv,
        CgRtolPolicy::from_config(cfg.cg_rtol),
        cfg.cg_max_iter,
        budget,
        cfg.fvv_sparse_rho,
    )
    .map_err(|e| match e {
        FerricError::General(m) => FerricError::General(format!("{who}: {m}")),
        e => e,
    })?;
    let (s, plan) = (&b.solve, b.decision);
    if outcome.target_met == Some(false) {
        eprintln!(
            "warning: {who}: solver-error bound {bound:.2e} Ha is above the {:.0e} Ha target \
             (true relres {true_relres:.2e}, recursive {:.2e}, after {} iters, rtol {:.0e})",
            outcome.target_ha.unwrap_or(f64::NAN),
            s.relres,
            s.iterations,
            outcome.rtol_used
        );
    }
    let c = RaggedCounters::new(rg, no, nv);
    Ok(AmplitudeLmp2Result {
        e_corr,
        e_total: e_scf + e_corr,
        e_corr_canonical_ri: f64::NAN,
        keep_fraction: c.keep_fraction,
        pair_fraction: c.pair_fraction,
        dom_mean: c.dom_mean,
        dom_max: c.dom_max,
        cg_iterations: s.iterations,
        cg_relres: s.relres,
        cg_true_relres: true_relres,
        cg_converged: s.converged,
        n_valence_virt: vvhv.n_valence,
        n_hard_virt: vvhv.n_hard,
        ragged_flops_per_matvec: s.flops_per_matvec,
        dense_flops_per_matvec: c.dense_flops,
        n_pairs_gated: 0,
        aux_dom_mean: 0.0,
        aux_dom_max: 0,
        cg_lanczos_kappa_est: lanczos_kappa(&s.trace),
        block_density_mean: c.block_density_mean,
        block_density_min: c.block_density_min,
        n_pairs_unique: c.n_pairs_unique,
        gate_on: cfg.pair_gate_cal.is_some(),
        solve_plan_bytes: plan.plan_bytes,
        solve_resident_bytes: plan.resident_bytes,
        cg_energy_error_bound: bound,
        cg_rtol_outcome: outcome,
        matvec_path: plan.path,
        timings: StageTimings::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use ferric_core::parallel::ParallelContext;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    /// The pre-Stage-B `solve_spd`: one LU factorization per column. Frozen
    /// here as the oracle of the Cholesky version.
    fn solve_spd_lu_oracle(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        use ndarray_linalg::Solve;
        let mut out = Array2::<f64>::zeros((a.nrows(), b.ncols()));
        for c in 0..b.ncols() {
            let col = a
                .solve(&b.column(c).to_owned())
                .map_err(|e| FerricError::General(format!("oracle solve: {e}")))?;
            out.column_mut(c).assign(&col);
        }
        Ok(out)
    }

    fn rhf_631g(xyz: &str) -> (Molecule, PreparedBasis, BasisSet, ScfResult) {
        let mol = Molecule::load_xyz(&format!(
            "{}/../../testdata/molecules/{xyz}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let bs = ferric_core::basis::bundled("6-31g").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let cfg = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
        let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
        (mol, obs, bs, rhf)
    }

    fn c4_631g() -> (Molecule, PreparedBasis, BasisSet, ScfResult) {
        rhf_631g("alkane_4.xyz")
    }

    /// Every column of `a` equals a DISTINCT column of `b` up to sign within
    /// 1e-10 (a bijection), and the number of columns whose partner is not
    /// at the same index. Symmetry-equivalent hard-virtual candidates tie
    /// exactly in the pivoted Cholesky, so a last-bit change may swap their
    /// ORDER; a reordering of the virtual set leaves every energy invariant,
    /// while any genuinely different column fails here.
    fn match_columns_up_to_sign_and_order(a: &Array2<f64>, b: &Array2<f64>) -> (bool, usize) {
        let mut used = vec![false; b.ncols()];
        let mut permuted = 0;
        for (k, ca) in a.columns().into_iter().enumerate() {
            let hit = b.columns().into_iter().enumerate().find(|(j, cb)| {
                let sign = if ca.dot(cb) < 0.0 { -1.0 } else { 1.0 };
                !used[*j]
                    && ca
                        .iter()
                        .zip(cb.iter())
                        .all(|(x, y)| (x - sign * y).abs() <= 1e-10)
            });
            match hit {
                Some((j, _)) => {
                    used[j] = true;
                    permuted += usize::from(j != k);
                }
                None => return (false, permuted),
            }
        }
        (true, permuted)
    }

    /// B1 anchor: the one-Cholesky multi-RHS `solve_spd` reproduces the
    /// frozen per-column-LU oracle on C4/6-31G's projected-minimal system
    /// (≤ 1e-10 relative), and the VV-HV space built on it matches the
    /// oracle-built space column by column (sign-fixed, ≤ 1e-10, same
    /// hard-virtual subset in the same order — before the near-tie pivot
    /// rule, four tied columns swapped order here) and passes the
    /// orthonormality/span check.
    #[test]
    fn solve_spd_cholesky_matches_lu_oracle_and_vvhv_is_unchanged() {
        let (mol, obs, bs, rhf) = c4_631g();
        let s = overlap(&obs);
        let (s_x, _) = cross_overlap_with_minimal(&mol, &obs, &bs).unwrap();
        let new = solve_spd(&s, &s_x).unwrap();
        let old = solve_spd_lu_oracle(&s, &s_x).unwrap();
        let scale = old.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let dev = (&new - &old).iter().fold(0.0f64, |m, x| m.max(x.abs())) / scale;
        assert!(dev <= 1e-10, "solve_spd vs LU oracle: {dev:.3e}");

        assert_vvhv_matches_lu_oracle(&mol, &obs, &bs, &rhf);
    }

    /// The VV-HV space built on the Cholesky `solve_spd` against the one
    /// built on the frozen LU oracle: the SAME hard-virtual subset in the
    /// same pivot order (AO indices of the selected candidates), the same
    /// columns (sign-fixed, <= 1e-10, no reordering), and the
    /// orthonormality/span check.
    fn assert_vvhv_matches_lu_oracle(
        mol: &Molecule,
        obs: &PreparedBasis,
        bs: &BasisSet,
        rhf: &ScfResult,
    ) {
        let (v_new, h_new) = build_vvhv_traced(mol, obs, bs, rhf, solve_spd).unwrap();
        let (v_old, h_old) = build_vvhv_traced(mol, obs, bs, rhf, solve_spd_lu_oracle).unwrap();
        assert_eq!(v_new.c_vloc.dim(), v_old.c_vloc.dim());
        assert_eq!(
            (v_new.n_valence, v_new.n_hard),
            (v_old.n_valence, v_old.n_hard)
        );
        assert!(v_new.n_hard > 0, "no hard virtuals: the test is vacuous");
        assert_eq!(
            h_new, h_old,
            "hard-virtual subset/order differs from the oracle"
        );
        let (matched, permuted) = match_columns_up_to_sign_and_order(&v_new.c_vloc, &v_old.c_vloc);
        assert!(
            matched && permuted == 0,
            "c_vloc is not the oracle's columns (<= 1e-10, sign; {permuted} reordered)"
        );
        let nocc = (mol.nelec() as usize) / 2;
        let (o, sp) = check_vvhv(obs, rhf, nocc, &v_new.c_vloc);
        assert!(
            o <= 1e-8 && sp <= 1e-8,
            "VV-HV check: orth {o:.2e} span {sp:.2e}"
        );
    }

    /// B1 oracle on symmetric non-alkanes, where symmetry-equivalent hard
    /// candidates (water's two H, benzene's six CH) tie exactly: with the
    /// near-tie rule the Cholesky- and LU-built spaces select the same
    /// hard-virtual subset in the same order.
    #[test]
    fn vvhv_matches_lu_oracle_on_symmetric_molecules() {
        for xyz in ["water.xyz", "benzene.xyz"] {
            let (mol, obs, bs, rhf) = rhf_631g(xyz);
            assert_vvhv_matches_lu_oracle(&mol, &obs, &bs, &rhf);
        }
    }

    /// Near-ties (within PIVOT_TIE_RTOL relative) go to the LOWEST index,
    /// so a last-bit perturbation cannot move the pivot; a real gap still
    /// wins; eliminated (-inf) entries are never chosen.
    #[test]
    fn select_pivot_breaks_near_ties_toward_the_lowest_index() {
        let tie = [1.0, 2.0, 2.0 * (1.0 + 1e-14), 0.5];
        assert_eq!(select_pivot(&tie).0, 1);
        let mut flipped = tie;
        flipped.swap(1, 2);
        assert_eq!(select_pivot(&flipped).0, 1);
        assert_eq!(select_pivot(&[1.0, 2.0, 2.0 + 1e-6]).0, 2);
        assert_eq!(select_pivot(&[f64::NEG_INFINITY, 0.3, 0.3]).0, 1);
    }

    /// Byte-identity guard for the ferric-pbc seam: the closure form with
    /// the Euclidean distance reproduces `pair_gate_keep` exactly (mask and
    /// count), across thresholds that gate some but not all pairs.
    #[test]
    fn pair_gate_keep_with_euclid_is_bitwise_pair_gate_keep() {
        let no = 6;
        let centers = Array2::from_shape_fn((no, 3), |(i, x)| 1.7 * i as f64 + 0.3 * x as f64);
        let spreads: Vec<f64> = (0..no).map(|i| 0.8 + 0.1 * i as f64).collect();
        let mut seen_partial = false;
        for eps in [0.0, 1e-6, 1e-4, 1e-2] {
            let a = pair_gate_keep(&centers, &spreads, no, eps, 0.7);
            let b = pair_gate_keep_with(
                |i, j| {
                    (0..3)
                        .map(|x| (centers[(i, x)] - centers[(j, x)]).powi(2))
                        .sum()
                },
                &spreads,
                no,
                eps,
                0.7,
            );
            assert_eq!(a, b, "eps {eps}");
            seen_partial |= a.1 > 0 && a.1 < no * (no - 1) / 2;
        }
        assert!(
            seen_partial,
            "no threshold gated a strict subset; the test is vacuous"
        );
    }
}
