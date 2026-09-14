//! DF-corrected incremental Fock SCF ("DF increments"): after a density-fitted
//! (RI-J/RI-K) pre-stage converges to a density `D0`, do ONE exact 4-index
//! Fock build `F_ex(D0)`, then run the remaining SCF iterations against
//!
//! ```text
//! F(D) = F_ex(D0) + F_DF(D - D0)
//! ```
//!
//! where `F_DF` is built from the (cheap) density-fitted J/K builders applied
//! to the density INCREMENT `ΔD = D - D0`, not to `D` itself. A final exact
//! build (and its own convergence check) always produces the reported energy
//! and density — see "Why the final answer is still exact" below.
//!
//! This is opt-in: `RhfConfig` gains no new field for it, and the only entry
//! point is [`solve_rhf_with_df_increments`], called explicitly. It touches
//! no existing code path — `solve_rhf`'s own exact incremental-Fock mechanism
//! (`INCREMENTAL_FULL_REBUILD_EVERY`, `d_last_fock`) is completely untouched,
//! and `solve_rhf_with_df_guess` is reused only via its pre-stage helper
//! (`ladder::run_df_guess_pre_stage`), never mutated.
//!
//! # What this is (and is not) — the "second-order" framing, corrected
//!
//! `F(D) = H + J(D) - K(D)/2` is EXACTLY affine (linear + constant) in `D` —
//! there is no quadratic term anywhere in Hartree-Fock. Writing `E(D) :=
//! J(D) - K(D)/2` for the two-electron part and `δE := E_DF - E_exact` for
//! the (also linear) DF/RI fitting-error operator, this scheme's Fock is
//!
//! ```text
//! F_ex(D0) + F_DF(D - D0)
//!   = [H + E(D0)] + [E_DF(D) - E_DF(D0)]
//!   = H + E(D) + δE(D) - δE(D0)
//!   = F_ex(D) + δE(D - D0)
//! ```
//!
//! i.e. the scheme's Fock equals the TRUE exact Fock at the CURRENT density
//! plus a term `δE(ΔD)` that is LINEAR (first-order), not second-order, in
//! `ΔD = D - D0` — `δE` is a linear operator applied to `ΔD`, so its output
//! scales with `‖ΔD‖`, not `‖ΔD‖²`. The original framing ("the DF error on
//! the increment is second order, so it vanishes as SCF converges") is
//! imprecise if read as a statement about this Fock-level error: it is
//! first-order. The only place a genuine second-order effect legitimately
//! appears is in the *energy* at a variational stationary point (a
//! first-order error in the density/Fock produces a second-order error in
//! the energy there) — a different and weaker claim than "the mechanism's
//! error vanishes". See "Why the final answer is still exact" for why the
//! scheme is sound regardless of this imprecision: correctness here never
//! depends on the inner loop's error actually being small, only on a
//! subsequent exact check.
//!
//! # Literature
//!
//! This is not a novel scheme. It is structurally identical to the exact
//! incremental-Fock mechanism already in `rhf.rs`
//! (`DirectJK::build_incremental`, `F(D) = F(D_last) + ΔF(ΔD)`), with the
//! delta-Fock kernel swapped from the exact 4-index builder to a cheap DF/RI
//! builder — i.e. "incremental Fock, but the increment uses an approximate
//! kernel instead of the exact one." That "freeze an expensive reference
//! operator, correct it with a cheap approximate response to the density
//! change, periodically refresh the reference exactly" pattern shares
//! lineage with dual-basis SCF (Liang & Head-Gordon: freeze a small-basis
//! Fock and perturbatively correct) and general two-level/multiscale SCF
//! acceleration ideas. It is NOT RIJCOSX (which fits K every iteration and
//! never freezes an exact reference) and NOT genuine dual-basis SCF (which
//! changes the AO basis, not the two-electron kernel). No single canonical
//! paper describes exactly this combination (frozen exact reference + DF-only
//! increments + a mandatory final exact confirmation/cleanup), as far as this
//! implementation is aware.

use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::operator::Operator;
use ndarray::linalg::general_mat_mul;
use ndarray::Array2;

use crate::diis::DiisDriver;
use crate::direct_jk::DirectJK;
use crate::driver::{density_change, diagonalize_rect};
use crate::fock::{JBuilder, KBuilder};
use crate::ladder::{run_df_guess_pre_stage, DF_GUESS_DEFAULT_AUX};
use crate::result::{ScfExit, ScfResult, Spin};
use crate::rhf::{canonical_orthogonalizer, resolve_three_index_budget, scf_converged, ConvergenceSignals, RhfConfig};
use crate::screening::SchwarzBounds;

/// Outcome of [`solve_rhf_with_df_increments`]: the final exact-integral
/// `ScfResult` plus enough of the intermediate stages' bookkeeping to audit
/// that the mechanism actually ran and how much exact work it saved.
#[derive(Debug, Clone)]
#[must_use]
pub struct DfIncrementsResult {
    /// The final result. ALWAYS built and convergence-checked from an exact
    /// (non-fitted) Fock — see the "final answer is exact" section of
    /// [`solve_rhf_with_df_increments`]'s doc. Callers should treat this as
    /// "the" answer, exactly as with `DfGuessResult::result`.
    pub result: ScfResult,
    /// Iterations spent in the DF pre-stage (loose convergence, builds `D0`).
    pub df_guess_iterations: usize,
    /// Whether the DF pre-stage's own (loose) gate was satisfied.
    pub df_guess_converged: bool,
    /// The DF pre-stage's own fitted energy (diagnostic only).
    pub df_guess_energy: f64,
    /// Iterations spent in the DF-corrected inner loop (cheap DF builds only;
    /// zero exact quartets). This is the stage the whole scheme exists to
    /// substitute for expensive exact iterations.
    pub inner_iterations: usize,
    /// Whether the inner (DF-corrected) loop's own convergence gate fired
    /// before its iteration cap. `false` is not fatal (mirrors
    /// `DfGuessResult::df_converged`'s reasoning) — the final exact stage
    /// re-checks convergence regardless and adds cleanup iterations if
    /// needed.
    pub inner_converged: bool,
    /// Number of ADDITIONAL exact full-Fock builds performed after the first
    /// post-inner-loop exact residual check, because that check failed (i.e.
    /// the DF-corrected density was not yet within the exact convergence
    /// gate). Zero means the single exact build right after the inner loop
    /// already satisfied convergence. A large number here means the DF
    /// correction did not land close enough to the true fixed point and the
    /// scheme's savings were partially or fully eaten by cleanup — reported
    /// so a caller can judge whether the technique is paying off on a given
    /// system, not just assume it did.
    pub exact_cleanup_iterations: usize,
    /// Total number of EXACT full Fock builds performed across the whole run
    /// (the reference build after D0, plus the confirmation build, plus any
    /// cleanup builds). The scheme's entire value proposition is this number
    /// staying small (ideally 2) regardless of how many total SCF iterations
    /// (`inner_iterations` + cleanup) were needed to get there.
    pub exact_builds: usize,
}

/// Cap on the DF-corrected inner loop, independent of `base.max_iter` (which
/// governs the exact cleanup stage). Generous: on a real system this loop
/// should converge in single digits once seeded by the DF-guess pre-stage,
/// but a cap prevents runaway iteration if the correction is oscillating.
pub const DF_INCREMENTS_INNER_MAX_ITER: usize = 40;

/// Cap on exact cleanup iterations after the inner loop, independent of
/// `base.max_iter`. If cleanup needs more than this, the DF correction was
/// not sound for this system and the run reports `converged: false` (exit
/// `MaxIter`) rather than silently burning an unbounded exact-iteration
/// budget.
pub const DF_INCREMENTS_CLEANUP_MAX_ITER: usize = 60;

/// Loosened inner-loop density-convergence threshold, mirroring
/// `solve_rhf_with_df_guess`'s DF-stage-looseness reasoning: the DF-corrected
/// fixed point differs from the exact one by RI error (plus the frozen-
/// reference error `δE(ΔD)` described in the module doc), so a tight gate
/// here would spin iterations chasing precision the final exact stage
/// discards anyway.
const DF_INCREMENTS_INNER_DENSITY_CONV: f64 = 1e-6;
/// Loosened inner-loop energy-convergence threshold (sanity "not still
/// descending" bound only, per `scf_converged`'s doc).
const DF_INCREMENTS_INNER_ENERGY_CONV: f64 = 1e-5;

/// Two-stage-plus-correction SCF: (1) a loose DF-guess pre-stage builds a
/// starting density `D0` (`ladder::run_df_guess_pre_stage`, the same
/// helper `solve_rhf_with_df_guess` uses); (2) ONE exact 4-index Fock build at
/// `D0`, `F_ex(D0)`, retained as a frozen reference; (3) a DF-corrected inner
/// loop iterates `F(D) = F_ex(D0) + F_DF(D - D0)`, where `F_DF` is built from
/// the DENSITY-fitted J/K builders applied to `D - D0` (via the density path,
/// NOT `build_from_occ` — `D - D0` is not a rank-`nocc` `C·Cᵀ` product, see
/// `crate::df_k::DfK::build_from_occ`'s doc); (4) ONE exact full Fock build at
/// the inner loop's final density, checked against the CALLER's real
/// (`base.energy_conv`/`base.density_conv`) convergence gate; (5) if that
/// check fails, additional EXACT full-rebuild SCF iterations (not
/// DF-corrected) until it passes or [`DF_INCREMENTS_CLEANUP_MAX_ITER`] is hit.
///
/// # Why the final answer is still exact
///
/// Steps (4)-(5) are the entire correctness argument, and they do not rely on
/// the DF error being small: the returned `ScfResult` is always built from —
/// and its `converged` flag always decided by — a Fock assembled with the
/// exact [`DirectJK`] builder, tested against the caller's real thresholds.
/// The inner loop (steps 2-3) exists ONLY to produce a good candidate density
/// cheaply; if it lands close to the true fixed point, step (5) needs zero
/// extra iterations, and if it doesn't, step (5) simply keeps iterating
/// exactly (at full exact-iteration cost) until it does (or the cleanup cap
/// is hit, which is reported as non-convergence, never silently accepted). So
/// a caller can never receive a result that is "converged" against anything
/// other than exact integrals — the DF stages can only affect wall-clock,
/// never correctness. Contrast with converging the DIIS/energy/density gate
/// on the DF-corrected Fock itself, which the code never treats as the
/// answer — `inner_converged` uses a deliberately LOOSE gate (see the module
/// constants) precisely so it can never be mistaken for the real one.
///
/// # Why this can still fail to be FASTER even though it stays correct
///
/// Correctness above is unconditional; performance is not. If `D0` from the
/// DF-guess pre-stage is not already close to the true fixed point, `ΔD = D -
/// D0` stays large throughout the inner loop and the DF fitting-error term
/// `δE(ΔD)` (see the module doc) is not small either — the inner loop can
/// then converge (by its own loose gate) to a candidate density that is still
/// far from exact self-consistency, and step (5) may need many exact cleanup
/// iterations, potentially erasing all savings versus a plain exact SCF. This
/// is why `exact_cleanup_iterations` is reported rather than hidden: it is
/// the signal for whether the technique paid off on a given system. The
/// DF-guess pre-stage is depended on for this precondition (it is NOT
/// optional — there is no cold-start variant of this function).
///
/// # Known inefficiency: the 3-index tensor is built twice
///
/// The DF-guess pre-stage (stage 1) builds and holds its own `DfJ`/`DfK`
/// (and their underlying raw 3-index `(P|μν)` tensor) internally inside its
/// `solve_rhf` call, then drops them when that call returns — `solve_rhf`
/// does not expose them to its caller. Stage 3 then builds a SECOND,
/// independent `DfJ`/`DfK` pair (same aux basis) for the inner loop. This
/// means the raw 3-index tensor — the single largest allocation and the most
/// expensive integral pass in the whole DF path — is computed twice per run
/// rather than once. Retaining stage 1's tensor across the boundary would
/// require restructuring `solve_rhf` to return its internal builders (or
/// duplicating a `solve_rhf`-shaped loop here so stage 1 runs inline instead
/// of through the existing pre-stage helper), which was judged out of scope
/// for this change: it would either touch `solve_rhf`'s signature (a much
/// wider-blast-radius change than this opt-in feature should require) or
/// discard the DRY sharing with `solve_rhf_with_df_guess`'s pre-stage. The
/// second build is bounded by the SAME `ooc_budget`/spill logic as the first
/// (both go through `build_df_jk`/`ThreeIndexSource`), so it is not a
/// correctness or memory-ceiling issue, only a measured-but-unavoidable-here
/// constant-factor cost. On a system where the DF pre-stage's own integral
/// build is a significant fraction of total DF-increments wall time, this
/// halves the benefit of skipping exact rebuilds; it does not affect the
/// EXACT-build count this scheme is optimizing.
///
/// # What this does NOT support
///
/// Closed-shell HF only (no `xc`, no COSMO/PCM/polarizable embedding, no
/// cDFT constraints, no Fermi smearing, no MOM, no Newton/SOSCF tail, no
/// pluggable `k_builder`/COSX): `base` must be a plain-HF `RhfConfig` with
/// those fields at their defaults, mirroring `solve_rhf_with_df_guess`'s own
/// closed-shell-only scope. This function does not validate that — passing a
/// config with those features set produces a result that silently ignores
/// them (the inner loop only ever builds plain HF J/K), which is a known gap;
/// see the deliverable notes.
#[allow(clippy::too_many_arguments)]
pub fn solve_rhf_with_df_increments(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    base: &RhfConfig,
    df_aux: Option<&str>,
) -> Result<DfIncrementsResult, FerricError> {
    let aux = df_aux.unwrap_or(DF_GUESS_DEFAULT_AUX);

    // ---- Stage 1: DF-guess pre-stage -> D0 -------------------------------
    // Reuses the SAME loosened DF pre-stage `solve_rhf_with_df_guess` runs
    // (shared helper `run_df_guess_pre_stage`), but — unlike
    // `solve_rhf_with_df_guess` — we do NOT then run a full exact SCF to
    // convergence here. That exact convergence is exactly what stages 3-5
    // below replace with the cheaper DF-corrected inner loop.
    let df_guess = run_df_guess_pre_stage(ctx, mol, prep, op, bounds, base, Some(aux))?;
    let d0 = df_guess.density_total.clone();
    let df_guess_iterations = df_guess.iterations;
    let df_guess_converged = df_guess.converged;
    let df_guess_energy = df_guess.energy;

    let n = prep.nbasis();
    let nelec = mol.nelec();
    if nelec % 2 != 0 {
        return Err(FerricError::ScfConvergence { iterations: 0, last_energy: 0.0 });
    }
    let nocc = (nelec / 2) as usize;

    let ooc_budget = resolve_three_index_budget(base.three_index_budget_bytes);

    // Geometry-only environment: S, hcore(+external), V_nn. Built directly
    // (not via `driver::prepare`) because this lane never needs the
    // XC/COSMO/PCM/polarizable/RSH machinery `prepare` also assembles — see
    // "What this does NOT support" above.
    let s = ferric_integrals::oneelectron::overlap(prep);
    let h = ferric_integrals::oneelectron::hcore_ecp_with_external(
        prep, mol, prep.basis_set(), base.external_potential.as_ref(),
    )?;
    let vnn = mol.nuclear_repulsion()
        + base.external_potential.as_ref().map_or(0.0, |ext| {
            ext.charge_nuclear_energy(mol) + ext.field_nuclear_energy(mol)
        });
    let x = canonical_orthogonalizer(&s)?;

    // ---- Stage 2: ONE exact full build at D0, retained as F_ex(D0) ------
    let mut direct_jk = DirectJK::new(ctx, prep, bounds, base.integral_thresh, ooc_budget);
    let mut exact_builds = 0usize;
    let mut exact_quartets = 0usize;
    let mut j_ref = Array2::<f64>::zeros((n, n));
    let mut k_ref = Array2::<f64>::zeros((n, n));
    exact_quartets += direct_jk.build(&d0, &mut j_ref, &mut k_ref)?;
    exact_builds += 1;
    // F_ex(D0) = H + J(D0) - 1/2 K(D0). Frozen for the whole inner loop.
    let mut f_ref = h.clone();
    f_ref += &j_ref;
    f_ref.scaled_add(-0.5, &k_ref);

    // ---- Stage 3: DF-corrected inner loop --------------------------------
    // F(D) = F_ex(D0) + F_DF(D - D0). Both DF builders are used via the
    // DENSITY path only (`JBuilder::build`/`KBuilder::build`) — `D - D0` is
    // not a rank-nocc C.Cᵀ product, so `DfK::build_from_occ` cannot represent
    // it (see that method's doc); we never call it here.
    let (df_j_opt, df_k_opt) = crate::fock_assembly::build_df_jk(
        ctx, mol, op, prep, Some(aux), Some(aux), ooc_budget,
    )?;
    let mut df_j = df_j_opt.ok_or_else(|| {
        FerricError::General("solve_rhf_with_df_increments: DF-J builder failed to construct".into())
    })?;
    let mut df_k = df_k_opt.ok_or_else(|| {
        FerricError::General("solve_rhf_with_df_increments: DF-K builder failed to construct".into())
    })?;

    let mut diis = DiisDriver::new(base.diis_flavor, base.diis_size, base.diis_switch_thresh);

    let mut d = d0.clone();
    let mut f = f_ref.clone();
    let mut fd_tmp = Array2::<f64>::zeros((n, n));
    let mut fds = Array2::<f64>::zeros((n, n));
    let mut sdf = Array2::<f64>::zeros((n, n));
    let mut err = Array2::<f64>::zeros((n, n));
    let mut j_delta = Array2::<f64>::zeros((n, n));
    let mut k_delta = Array2::<f64>::zeros((n, n));

    let mut inner_converged = false;
    let mut inner_iterations = 0usize;
    let mut prev_energy = 0.0f64;

    for iter in 1..=DF_INCREMENTS_INNER_MAX_ITER {
        ctx.check_interrupted()?;
        inner_iterations = iter;

        let delta_d = &d - &d0;
        df_j.build(&delta_d, &mut j_delta)?;
        df_k.build(&delta_d, &mut k_delta)?;

        // F = F_ex(D0) + J_DF(ΔD) - 1/2 K_DF(ΔD).
        f.assign(&f_ref);
        f += &j_delta;
        f.scaled_add(-0.5, &k_delta);

        let energy = 0.5 * (&d * &(&h + &f)).sum() + vnn;

        chained_mat_mul3(&f, &d, &s, &mut fd_tmp, &mut fds);
        chained_mat_mul3(&s, &d, &f, &mut fd_tmp, &mut sdf);
        ndarray::Zip::from(&mut err).and(&fds).and(&sdf).for_each(|e, &a, &b| *e = a - b);
        let err_max = err.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let de = (energy - prev_energy).abs();

        let f_new = diis.step(&f, &err, &d, energy, err_max);
        let (_eps, c) = diagonalize_rect(&f_new, &x)?;
        let c_occ = c.slice(ndarray::s![.., ..nocc]);
        let d_new = with_blas_threads(opt_in_blas_threads(), || 2.0 * c_occ.dot(&c_occ.t()));

        let (dp_rms, dp_max) = density_change(&d_new, &d);
        prev_energy = energy;
        d = d_new;

        // Loose inner gate: exists only to stop feeding stage 4 a moving
        // target once the DF-corrected iteration has settled to ITS OWN
        // (possibly RI-error-shifted) fixed point — never treated as the
        // real answer. iter > 1 guards against `de` being spuriously small
        // on iteration 1 (prev_energy seeded at 0.0).
        if iter > 1 {
            let sig = ConvergenceSignals { de, dp_rms, dp_max };
            if scf_converged(sig, DF_INCREMENTS_INNER_ENERGY_CONV, DF_INCREMENTS_INNER_DENSITY_CONV).is_some() {
                inner_converged = true;
                break;
            }
        }
    }

    // ---- Stages 4-5: exact confirmation + (if needed) exact cleanup -----
    // Ordinary exact-Fock Roothaan/DIIS iteration from here on — no DF
    // anywhere below this line. The FIRST pass through this loop is the
    // mandatory "stage 4" confirmation build; `exact_cleanup_iterations`
    // counts only passes beyond that first one.
    let mut d_cur = d;
    let mut last_exact_energy: Option<f64> = None;
    let mut exact_cleanup_iterations = 0usize;

    loop {
        let mut j_e = Array2::<f64>::zeros((n, n));
        let mut k_e = Array2::<f64>::zeros((n, n));
        exact_quartets += direct_jk.build(&d_cur, &mut j_e, &mut k_e)?;
        exact_builds += 1;

        let mut f_e = h.clone();
        f_e += &j_e;
        f_e.scaled_add(-0.5, &k_e);

        let energy = 0.5 * (&d_cur * &(&h + &f_e)).sum() + vnn;
        // `de` is INFINITY on the very first exact check (no prior exact
        // energy to compare against) so this check can never look converged
        // on `de` alone before an exact Fock has actually been compared
        // energy-to-energy with another exact Fock.
        let de = match last_exact_energy {
            Some(prev) => (energy - prev).abs(),
            None => f64::INFINITY,
        };
        last_exact_energy = Some(energy);

        chained_mat_mul3(&f_e, &d_cur, &s, &mut fd_tmp, &mut fds);
        chained_mat_mul3(&s, &d_cur, &f_e, &mut fd_tmp, &mut sdf);
        ndarray::Zip::from(&mut err).and(&fds).and(&sdf).for_each(|e, &a, &b| *e = a - b);
        let err_max = err.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        let (eps, c) = diagonalize_rect(&f_e, &x)?;
        let c_occ = c.slice(ndarray::s![.., ..nocc]);
        let d_new = with_blas_threads(opt_in_blas_threads(), || 2.0 * c_occ.dot(&c_occ.t()));
        let (dp_rms, dp_max) = density_change(&d_new, &d_cur);

        let sig = ConvergenceSignals { de, dp_rms, dp_max };
        let conv = scf_converged(sig, base.energy_conv, base.density_conv);

        if let Some(exit) = conv {
            let density_alpha = 0.5 * &d_cur;
            let result = ScfResult {
                spin: Spin::Restricted,
                energy,
                density_total: d_cur,
                density_alpha,
                density_beta: None,
                mos_alpha: c,
                mos_beta: None,
                eps_alpha: eps,
                eps_beta: None,
                fock_alpha: f_e,
                fock_beta: None,
                converged: true,
                exit,
                iterations: inner_iterations + exact_cleanup_iterations,
                computed_quartets: exact_quartets,
                induced_dipoles: None,
            };
            return Ok(DfIncrementsResult {
                result,
                df_guess_iterations,
                df_guess_converged,
                df_guess_energy,
                inner_iterations,
                inner_converged,
                exact_cleanup_iterations,
                exact_builds,
            });
        }

        exact_cleanup_iterations += 1;
        if exact_cleanup_iterations > DF_INCREMENTS_CLEANUP_MAX_ITER {
            let density_alpha = 0.5 * &d_new;
            let result = ScfResult {
                spin: Spin::Restricted,
                energy,
                density_total: d_new,
                density_alpha,
                density_beta: None,
                mos_alpha: c,
                mos_beta: None,
                eps_alpha: eps,
                eps_beta: None,
                fock_alpha: f_e,
                fock_beta: None,
                converged: false,
                exit: ScfExit::MaxIter,
                iterations: inner_iterations + exact_cleanup_iterations,
                computed_quartets: exact_quartets,
                induced_dipoles: None,
            };
            return Ok(DfIncrementsResult {
                result,
                df_guess_iterations,
                df_guess_converged,
                df_guess_energy,
                inner_iterations,
                inner_converged,
                exact_cleanup_iterations,
                exact_builds,
            });
        }

        // Not converged against the exact residual: take a normal
        // Roothaan/DIIS step from the exact Fock and try again.
        let f_new = diis.step(&f_e, &err, &d_cur, energy, err_max);
        let (_eps2, c2) = diagonalize_rect(&f_new, &x)?;
        let c_occ2 = c2.slice(ndarray::s![.., ..nocc]);
        d_cur = with_blas_threads(opt_in_blas_threads(), || 2.0 * c_occ2.dot(&c_occ2.t()));
    }
}

/// `(a·b)·c` computed as two `general_mat_mul` calls into caller-supplied
/// scratch, mirroring the exact association `solve_rhf` uses for its DIIS
/// commutator (`(F·D)·S`, `(S·D)·F`) under the same opt-in BLAS-thread raise.
fn chained_mat_mul3(
    a: &Array2<f64>,
    b: &Array2<f64>,
    c: &Array2<f64>,
    out1: &mut Array2<f64>,
    out2: &mut Array2<f64>,
) {
    with_blas_threads(opt_in_blas_threads(), || {
        general_mat_mul(1.0, a, b, 0.0, out1);
        general_mat_mul(1.0, out1, c, 0.0, out2);
    });
}
