//! Configurable SCF convergence ladder: walk a sequence of RhfConfig "rungs",
//! carrying each rung's final density into the next, aborting a stuck rung early.

use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use crate::rhf::{solve_rhf, RhfConfig};
use crate::result::{ScfExit, ScfResult};
use crate::screening::SchwarzBounds;

/// One rung of an SCF convergence ladder.
#[derive(Debug, Clone)]
pub struct Rung {
    /// Partial RHF config for this rung.
    pub config: RhfConfig,
    /// false: inherit the previous rung's final density (default, progressive
    /// refinement). true: discard incoming density and use this rung's own guess.
    pub restart: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RungOutcome {
    pub iters: usize,
    pub exit: ScfExit,
    pub final_err_max: f64, // best-effort; set to f64::NAN if not surfaced
    pub final_energy: f64,
}

/// Result of running a multi-rung SCF convergence ladder.
#[derive(Debug, Clone)]
#[must_use]
pub struct LadderResult {
    pub result: ScfResult,
    pub converged: bool,
    pub rung_reached: usize,
    pub rung_outcomes: Vec<RungOutcome>,
}

/// Walk the ladder: run each rung, carry density forward unless `restart`, stop
/// at the first converged rung; else return the best-effort (lowest-energy)
/// non-converged result.
pub fn solve_rhf_ladder(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    ladder: &[Rung],
) -> Result<LadderResult, FerricError> {
    if ladder.is_empty() {
        return Err(FerricError::General("empty SCF ladder".into()));
    }
    let mut carried: Option<ndarray::Array2<f64>> = None;
    let mut outcomes = Vec::with_capacity(ladder.len());
    let mut best: Option<ScfResult> = None;
    let mut best_rung = 0usize;

    for (i, rung) in ladder.iter().enumerate() {
        let mut cfg = rung.config.clone();
        if !rung.restart {
            if let Some(d) = carried.as_ref() {
                cfg.init_guess_density = Some(d.clone());
            }
        }
        let r = solve_rhf(ctx, mol, prep, op, bounds, &cfg)?;
        outcomes.push(RungOutcome {
            iters: r.iterations,
            exit: r.exit,
            final_err_max: f64::NAN,
            final_energy: r.energy,
        });
        if r.converged {
            return Ok(LadderResult {
                result: r,
                converged: true,
                rung_reached: i,
                rung_outcomes: outcomes,
            });
        }
        // Carry this rung's density forward.
        carried = Some(r.density_total.clone());
        // Track best-effort (lowest energy) fallback.
        if best.as_ref().map_or(true, |b| r.energy < b.energy) {
            best = Some(r);
            best_rung = i;
        }
    }

    let result = best.expect("non-empty ladder always sets best");
    Ok(LadderResult {
        result,
        converged: false,
        rung_reached: best_rung,
        rung_outcomes: outcomes,
    })
}

/// Default auxiliary basis for the DF-guess pre-stage: a JK-fit set (NOT an
/// MP2/RI-fit set — see `RhfConfig::df_k_aux`'s doc on why K needs a
/// dedicated JKFIT-type basis). Kept as its own constant, distinct from
/// whatever `[mp2] auxbasis` the caller may have configured for a downstream
/// correlated method, per the project convention that SCF JK aux stays
/// separate from RPA/MP2 aux (see `ferric-jk-aux-convention`).
pub const DF_GUESS_DEFAULT_AUX: &str = "def2-universal-jkfit";

/// Outcome of a [`solve_rhf_with_df_guess`] run: the exact-stage `ScfResult`
/// plus enough of the DF stage's own outcome to audit that it actually ran
/// (iteration count, convergence, energy) without re-exposing a whole second
/// `ScfResult`.
#[derive(Debug, Clone)]
#[must_use]
pub struct DfGuessResult {
    /// The exact-integral stage's result. This is what callers should treat
    /// as "the" SCF result — the DF stage is purely a density generator and
    /// never contributes to the returned energy/orbitals.
    pub result: ScfResult,
    /// Iteration count of the DF (fitted J/K) pre-stage.
    pub df_iterations: usize,
    /// Whether the DF pre-stage satisfied its own (loose) convergence gate.
    /// `false` is not fatal here — even a partially-converged DF density is a
    /// far better starting point than hcore/MINAO for the exact stage, so a
    /// non-convergent DF stage is not an error, only a (silent, by design —
    /// see `solve_rhf_with_df_guess` doc) missed speedup.
    pub df_converged: bool,
    /// The DF pre-stage's own final energy (fitted, NOT physically meaningful
    /// as a final answer — reported for diagnostics/logging only).
    pub df_energy: f64,
}

/// `max_iter` cap for the DF pre-stage (see `solve_rhf_with_df_guess` doc,
/// "Loose DF-stage threshold" section): 25 is generous relative to Psi4's own
/// ~10-ish DF pre-iterations, while still much less than a typical exact-SCF
/// `max_iter` budget (default 200) — this stage exists to get INTO the right
/// basin cheaply, not to fully converge there.
pub const DF_GUESS_MAX_ITER: usize = 25;

/// `max_iter` for the DF pre-stage when `FERRIC_DF_GUESS_TIGHT` is on.
///
/// MEASURED (benzene/aug-cc-pVDZ), and the measurement CORRECTED an assumption:
///
/// | tight cap | DF iters | exact iters | USER  |
/// |-----------|----------|-------------|-------|
/// | 25        | 25 (stalled) | 6       | 122.82 s |
/// | 120       | 100 (stalled) | 6      | 136.90 s |
///
/// Raising the cap bought ZERO exact iterations and cost 75 extra DF ones. The
/// binding constraint was never the cap: the DF stage simply CANNOT reach the
/// exact stage's thresholds, because the RI-fitted Fock has a different fixed
/// point (~1e-4 Ha away) — which is exactly what `DF_GUESS_DENSITY_CONV_LOOSEN`
/// was built around in the first place. The "a DF iteration costs ~1% of an
/// exact one, so spend 100 of them to save one exact" argument is sound
/// economics but rests on the DF density continuing to IMPROVE; past the stall
/// point it does not, so the extra iterations are pure cost.
///
/// Kept at the loose policy's cap: past ~25 the DF stage is spinning at its own
/// fixed point, not converging toward the exact one.
pub const DF_GUESS_TIGHT_MAX_ITER: usize = DF_GUESS_MAX_ITER;

/// Loosening factor applied to `density_conv` for the DF pre-stage (see
/// `solve_rhf_with_df_guess` doc).
const DF_GUESS_DENSITY_CONV_LOOSEN: f64 = 1e3;
/// Floor on the DF pre-stage's `density_conv`, so a caller with an already
/// loose target doesn't get an even-looser DF stage than that floor implies.
const DF_GUESS_DENSITY_CONV_FLOOR: f64 = 1e-5;
/// Loosening factor applied to `energy_conv` for the DF pre-stage.
const DF_GUESS_ENERGY_CONV_LOOSEN: f64 = 10.0;
/// Floor on the DF pre-stage's `energy_conv`.
const DF_GUESS_ENERGY_CONV_FLOOR: f64 = 1e-2;

/// Two-stage "DF guess" SCF: converge a density-fitted J/K SCF to a LOOSE
/// threshold first, then hand its converged (or best-effort) density to a
/// fresh exact-4-index-integral SCF that runs to the caller's real
/// `energy_conv`/`density_conv`. Mirrors Psi4's "Andy trick 2.0"
/// (`psi4/driver/procrouting/scf_proc/scf_iterator.py:56-100`,
/// `scf_compute_energy`): DF pre-iterations are far cheaper per-iteration
/// than exact 4-index ones, and the fitted Fock operator's basin is close
/// enough to the exact one that most of the iteration budget can be spent
/// there instead.
///
/// `df_aux` selects the J/K fitting basis for the pre-stage only (`None` ->
/// [`DF_GUESS_DEFAULT_AUX`]); it is never applied to the exact stage. `base`
/// is the exact-stage config (its own `df_j_aux`/`df_k_aux` — normally both
/// `None` — are left untouched and used verbatim for the exact stage).
///
/// # Why this is safe to bolt on as a pure wrapper
///
/// `base` is passed through UNCHANGED as the exact stage's config except for
/// `init_guess_density` (set to the DF stage's final density) and
/// `use_sad_guess` (forced `false`, since we now have a real starting
/// density and must not let the MINAO/SAD guess override it — see
/// `solve_rhf`'s guess-precedence doc: `init_guess_density` already wins over
/// `use_sad_guess`, but setting it `false` here makes the precedence
/// explicit rather than incidental). Every other field — `df_j_aux`/
/// `df_k_aux` on `base` in particular — is whatever the CALLER wants for the
/// exact stage (normally `None`, i.e. actually exact); the DF stage builds
/// its OWN separate config with its own aux and never mutates `base`.
///
/// # DIIS reset at the DF→exact handoff
///
/// Psi4 explicitly resets its DIIS subspace at the handoff
/// (`diis_manager_.reset_subspace()`) because the DF stage's error vectors
/// are computed from a *different* (fitted) Fock operator and would poison
/// the exact stage's extrapolation. Ferric needs no equivalent call here:
/// `solve_rhf` constructs a brand-new `DiisDriver` from scratch on every
/// invocation (see the top of its iteration setup) — there is no DIIS state
/// that persists across separate `solve_rhf` calls in the first place. Two
/// independent `solve_rhf` calls (as used here, and as `solve_rhf_ladder`
/// already relies on for its rung-to-rung transitions) are DIIS-isolated by
/// construction; only the density crosses the boundary. So the "reset" Psi4
/// needs is already the default ferric behavior — there is nothing extra to
/// wire up.
///
/// # Loose DF-stage threshold
///
/// The DF stage's fitted Fock operator has a DIFFERENT fixed point than the
/// exact one (the RI approximation shifts it, typically ~1e-4 Ha territory
/// per `default_ladder_from`'s doc on the historical DF-JK-ladder removal).
/// Converging the DF stage to the caller's real (possibly 1e-8-1e-10) density
/// threshold would burn iterations chasing precision in a density that the
/// exact stage is about to perturb anyway. We instead loosen BOTH gates by a
/// fixed factor relative to whatever the caller asked the exact stage to hit:
///   - `density_conv`: `max(base.density_conv * 1e3, 1e-5)` — three orders
///     looser than the target, floored at 1e-5 so a caller who already asked
///     for something loose (e.g. 1e-4) doesn't get a DF stage LOOSER than
///     that ratio would suggest is still a meaningful gate.
///   - `energy_conv`: `max(base.energy_conv * 10.0, 1e-2)` — one order looser
///     than the exact stage's (already loose, 1e-3-default) sanity bound.
/// `max_iter` is capped at [`DF_GUESS_MAX_ITER`]. If the DF stage hits
/// `max_iter` without converging, its (still much-improved-over-cold-guess)
/// density is used anyway — see `DfGuessResult::df_converged`.
pub fn solve_rhf_with_df_guess(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    base: &RhfConfig,
    df_aux: Option<&str>,
) -> Result<DfGuessResult, FerricError> {
    let aux = df_aux.unwrap_or(DF_GUESS_DEFAULT_AUX);

    let mut df_cfg = base.clone();
    df_cfg.df_j_aux = Some(aux.to_string());
    df_cfg.df_k_aux = Some(aux.to_string());
    // How tightly to converge the DF pre-stage.
    //
    // The original reasoning was that the RI-fitted Fock has a DIFFERENT fixed
    // point than the exact one (~1e-4 Ha), so converging the DF stage tightly
    // "wastes" iterations chasing precision in a density the exact stage is
    // about to perturb. MEASURED, that argument is incomplete: what matters is
    // not how close the DF density is to the DF fixed point, but how close it
    // is to the EXACT one — and a DF iteration costs ~1% of an exact one
    // (Psi4's own timer.dat: 0.2 vs 27.7 CPU-s per call at aDZ). Trading a
    // cheap DF iteration for an expensive exact one is therefore a ~100:1 win
    // as long as the DF density keeps improving. Benzene/aug-cc-pVDZ today
    // splits 7 DF + 7 exact; Psi4 runs 10 DF + 5 direct.
    //
    // `FERRIC_DF_GUESS_TIGHT=1` converges the DF stage to the caller's OWN
    // thresholds instead of loosened ones, so the two policies can be A/B'd on
    // one binary. Default (unset) keeps the historical loosened behavior, so
    // this is byte-identical unless opted into.
    // MEASURED: running the DF pre-stage to its STALL POINT (rather than to a
    // loosened threshold) is a win at both bases, benzene, 4 threads, idle box:
    //
    // |      | DF iters | exact iters | wall     | USER      |
    // |------|----------|-------------|----------|-----------|
    // | aDZ loose | 7   | 7           |  36.73 s |  140.73 s |
    // | aDZ tight | 25  | 6           |  32.16 s |  122.82 s |  1.15x
    // | aTZ loose | 7   | 6           | 378.94 s | 1463.53 s |
    // | aTZ tight | 25  | 5           | 323.46 s | 1243.93 s |  1.17x
    //
    // Converged energies are identical (aTZ: -230.7808857506 both ways).
    //
    // Note what the win actually is: the DF stage does NOT converge — it hits
    // the iteration cap in every tight run. The saving comes from letting it
    // run until it stalls at the fitted Fock's own fixed point (~1e-4 Ha from
    // the exact one), which is a better starting density than the loosened
    // threshold stops at, and buys one fewer EXACT iteration. A DF iteration
    // costs ~1% of an exact one, so trading ~18 extra DF iterations for one
    // fewer exact build is strongly positive. Past the stall point it is not:
    // raising the cap to 120 spent 75 more DF iterations for ZERO additional
    // exact-iteration saving (aDZ USER 122.82 -> 136.90 s), which is why
    // `DF_GUESS_TIGHT_MAX_ITER` stays at the stall point.
    //
    // Default ON. `FERRIC_DF_GUESS_TIGHT=0` restores the loosened-threshold
    // policy for A/B without a rebuild.
    let tight_df = !matches!(
        std::env::var("FERRIC_DF_GUESS_TIGHT").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    );
    if tight_df {
        // Inherit base.density_conv / base.energy_conv verbatim. The DF stage
        // still stops at DF_GUESS_MAX_ITER, so a fitted Fock that cannot reach
        // the exact stage's threshold degrades to "best DF density available"
        // rather than spinning — the same non-fatal contract as before.
        df_cfg.max_iter = base.max_iter.min(DF_GUESS_TIGHT_MAX_ITER);
    } else {
        df_cfg.density_conv = (base.density_conv * DF_GUESS_DENSITY_CONV_LOOSEN).max(DF_GUESS_DENSITY_CONV_FLOOR);
        df_cfg.energy_conv = (base.energy_conv * DF_GUESS_ENERGY_CONV_LOOSEN).max(DF_GUESS_ENERGY_CONV_FLOOR);
        df_cfg.max_iter = base.max_iter.min(DF_GUESS_MAX_ITER);
    }
    // The DF stage still wants its own fast route to a starting density,
    // exactly like an un-laddered `solve_rhf` call would.
    df_cfg.init_guess_density = None;

    let df_result = solve_rhf(ctx, mol, prep, op, bounds, &df_cfg)?;

    let mut exact_cfg = base.clone();
    exact_cfg.init_guess_density = Some(df_result.density_total.clone());
    exact_cfg.use_sad_guess = false;

    let result = solve_rhf(ctx, mol, prep, op, bounds, &exact_cfg)?;

    Ok(DfGuessResult {
        result,
        df_iterations: df_result.iterations,
        df_converged: df_result.converged,
        df_energy: df_result.energy,
    })
}

/// Built-in default ladder: stall/divergence abort on every rung, level-shift
/// escalation, density carried forward. Tuned from CCuN/aTZ measurements
/// (2026-07-08): rung 1 alone banks CCuN in <1 min. J/K is inherited from the
/// base config, never substituted — see `default_ladder_from`.
pub fn default_ladder() -> Vec<Rung> {
    default_ladder_from(&RhfConfig::default())
}

/// Same efficacy-ordered escalation as `default_ladder`, but every rung
/// starts from a caller-supplied `base` (e.g. a `RhfConfig` carrying a
/// user's flat `[scf]` TOML settings) instead of `RhfConfig::default()`.
/// `max_iter`/level-shift/DIIS-flavor/newton-trigger/smearing are
/// still escalated per rung exactly as in `default_ladder`; any other field
/// already set on `base` (mom_after_iter, xc, energy_conv, ...) is carried
/// through unchanged on every rung. Keeps `ferric-cli`'s empty-`[[scf.ladder]]`
/// fallback (`ScfCfg::build_ladder`) from silently drifting out of sync with
/// this ladder's shape.
///
/// # J/K is NOT substituted
///
/// This ladder used to force `df_j_aux`/`df_k_aux` to `def2-universal-jkfit`
/// whenever the base left them unset, silently converting an exact 4-index
/// `kind="rhf"` run into an RI-JK one. That changes the METHOD, not just the
/// convergence strategy: the same TOML run through the CLI (laddered) and
/// through `solve_rhf` directly (not laddered) then disagreed by the RI
/// fitting error (~1e-4 Ha on water/STO-3G) with nothing in the output
/// explaining why. Density fitting is a user-visible accuracy trade and must
/// be opted into explicitly (`[scf] df_j_aux`/`df_k_aux`, or the `RhfConfig`
/// fields). The ladder now escalates only *convergence* knobs and inherits
/// whatever J/K the caller chose.
pub fn default_ladder_from(base: &RhfConfig) -> Vec<Rung> {
    let rung = |level_shift: f64, max_iter: usize| {
        let mut c = base.clone();
        c.level_shift = level_shift;
        c.max_iter = max_iter;
        c.stall_window = Some(15);
        c.divergence_tol = Some(0.5);
        c
    };
    // Efficacy-ordered escalation: cheapest+most-effective first, adding a
    // convergence accelerator per rung only when the previous one didn't
    // converge. The MINAO guess (now the default in `solve_rhf`) makes rung 0
    // suffice for the large majority; the later rungs are for hard cases
    // (heavy-atom / near-degenerate-d-manifold closed shells).
    //   rung 0: MINAO guess + plain DIIS            (default; most molecules)
    //   rung 1: + ADIIS-early → DIIS-late           (robust far from convergence)
    //   rung 2: + virtual-block level shift 0.5     (damps overshooting rotation)
    //   rung 3: + level shift 1.0 and SOSCF tail    (quadratic tail once err small)
    //   rung 4: + Fermi smearing σ=0.01 Ha          (near-degenerate d-manifold)
    let with = |mut c: RhfConfig,
                flavor: crate::diis::DiisFlavor,
                newton: f64,
                smear: Option<f64>| {
        c.diis_flavor = flavor;
        c.newton_trigger = newton;
        c.smearing_sigma = smear;
        c
    };
    use crate::diis::DiisFlavor::{Adiis, Pulay};
    vec![
        Rung { config: rung(0.0, 60), restart: false },
        Rung { config: with(rung(0.0, 60), Adiis, 0.0, None), restart: false },
        Rung { config: with(rung(0.5, 60), Adiis, 0.0, None), restart: false },
        Rung { config: with(rung(1.0, 80), Adiis, 1e-3, None), restart: false },
        Rung { config: with(rung(0.5, 100), Pulay, 1e-3, Some(0.01)), restart: false },
    ]
}

/// Level-shift escalation ladder for a KS-DFT (or any) base config, carrying
/// the base's `xc` / DFT-grid / DF-JK-aux settings into every rung.
///
/// `default_ladder()` does NOT set `xc`, so it cannot be reused verbatim for
/// KS-DFT: it would discard the functional. This builder starts from the
/// caller's `base` (so `xc`, `dft_grid`, `nlc_grid`, and any explicit DF-JK aux
/// survive) and layers on the same 0 → 0.5 → 1.0 level-shift escalation plus
/// stall/divergence early-abort. It fixes the DF-B3LYP DIIS limit-cycle
/// (docs/profiles-2026-07-14.md finding (2)) the exact same way the virtual-
/// block level shift fixes closed-shell heavy-atom RHF divergence: KS-DFT
/// closed-shell IS `solve_rhf` with `cfg.xc = Some(...)`, so the shift damps
/// the same overshooting orbital rotation.
///
/// DF-JK aux is auto-defaulted to `def2-universal-jkfit` only when the base
/// leaves it unset AND a functional is present (hybrids/RSH need K; pure DFT
/// still benefits from RI-J) — a bare-HF base with no aux keeps direct J/K.
pub fn ksdft_ladder(base: &RhfConfig) -> Vec<Rung> {
    let mk = |level_shift: f64, max_iter: usize| {
        let mut c = base.clone();
        c.level_shift = level_shift;
        c.max_iter = max_iter;
        c.stall_window = Some(15);
        c.divergence_tol = Some(0.5);
        if c.xc.is_some() {
            if c.df_j_aux.is_none() {
                c.df_j_aux = Some("def2-universal-jkfit".to_string());
            }
            if c.df_k_aux.is_none() {
                c.df_k_aux = Some("def2-universal-jkfit".to_string());
            }
        }
        c
    };
    // Efficacy-ordered escalation, same shape as default_ladder() but carrying
    // the KS functional/grid through every rung. Rung 0 honors the caller's own
    // level_shift/max_iter; later rungs add a convergence accelerator each and
    // only run if the previous rung stalled.
    //   0: caller's shift + plain DIIS
    //   1: + ADIIS-early → DIIS-late
    //   2: + level shift ≥0.5
    //   3: + level shift +0.5 and SOSCF tail (skipped internally for RSH/meta-GGA)
    //   4: + Fermi smearing σ=0.01 Ha (near-degenerate d-manifold last resort)
    let ls0 = base.level_shift;
    let acc = |mut c: RhfConfig,
               flavor: crate::diis::DiisFlavor,
               newton: f64,
               smear: Option<f64>| {
        c.diis_flavor = flavor;
        c.newton_trigger = newton;
        c.smearing_sigma = smear;
        c
    };
    use crate::diis::DiisFlavor::{Adiis, Pulay};
    vec![
        Rung { config: mk(ls0, base.max_iter), restart: false },
        Rung { config: acc(mk(ls0, 60), Adiis, 0.0, None), restart: false },
        Rung { config: acc(mk(ls0.max(0.5), 60), Adiis, 0.0, None), restart: false },
        Rung { config: acc(mk(ls0.max(0.5) + 0.5, 80), Adiis, 1e-3, None), restart: false },
        Rung { config: acc(mk(ls0.max(0.5), 100), Pulay, 1e-3, Some(0.01)), restart: false },
    ]
}

/// Walk a convergence ladder using ROHF/ROKS, carrying each failed rung's density
/// forward as the next rung's guess.
///
/// The open-shell counterpart of [`solve_rhf_ladder`]. ROHF/ROKS has no closed-shell
/// fallback and is materially harder to converge — radicals and open-shell
/// transition-metal complexes are exactly the systems that stall plain DIIS, and
/// exactly what a robustness-oriented functional is for.
///
/// Returns the first converged rung, or the lowest-energy non-converged result with
/// `converged: false`.
pub fn solve_rohf_ladder(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    ladder: &[Rung],
) -> Result<LadderResult, FerricError> {
    let mut outcomes: Vec<RungOutcome> = Vec::new();
    let mut best: Option<ScfResult> = None;
    let mut carry: Option<ndarray::Array2<f64>> = None;

    for (i, rung) in ladder.iter().enumerate() {
        let mut cfg = rung.config.clone();
        // Carry the previous rung's best-effort density in as this rung's guess.
        if !rung.restart {
            if let Some(d) = carry.as_ref() {
                cfg.init_guess_density = Some(d.clone());
            }
        }
        let r = crate::rohf::solve_rohf_best_effort(ctx, mol, prep, op, bounds, &cfg)?;
        outcomes.push(RungOutcome {
            iters: r.iterations,
            exit: r.exit,
            final_err_max: f64::NAN,
            final_energy: r.energy,
        });
        if r.converged {
            return Ok(LadderResult {
                result: r,
                converged: true,
                rung_reached: i,
                rung_outcomes: outcomes,
            });
        }
        carry = Some(r.density_total.clone());
        best = match best {
            Some(b) if b.energy <= r.energy => Some(b),
            _ => Some(r),
        };
    }

    let result = best.ok_or_else(|| {
        FerricError::General("solve_rohf_ladder: empty ladder".into())
    })?;
    let n = outcomes.len().saturating_sub(1);
    Ok(LadderResult { result, converged: false, rung_reached: n, rung_outcomes: outcomes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    #[test]
    fn ladder_converges_water_first_rung() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let ladder = vec![Rung { config: RhfConfig::default(), restart: false }];
        let lr = solve_rhf_ladder(&ctx, &mol, &prep, op, &bounds, &ladder).unwrap();
        assert!(lr.converged);
        assert_eq!(lr.rung_reached, 0);
        assert_eq!(lr.rung_outcomes.len(), 1);
    }

    #[test]
    fn ladder_carries_density_to_second_rung() {
        // Both rungs disable the SAD default guess (use_sad_guess:false), so
        // rung 2 can ONLY start from a good density if the ladder's
        // carry-forward wiring (`cfg.init_guess_density = Some(carried)` in
        // `solve_rhf_ladder`) actually injects rung 1's final density. If that
        // block were deleted, rung 2 would fall back to the cold hcore guess
        // instead (SAD is off), which needs 8 iterations to converge on
        // water/STO-3G (measured) -- more than rung 2's max_iter budget below.
        //
        // Rung 1 is capped at 3 iters: a real partial SCF run from hcore that
        // does NOT converge (exits MaxIter) but leaves a much-improved density.
        // Rung 2 is capped at 6 iters: measured to be exactly enough to
        // converge when seeded with rung 1's 3-iter density, but NOT enough
        // for a cold-hcore run (which needs 8). This makes the test fail if
        // carry-forward is broken: rung 2 would hit MaxIter too and the ladder
        // would never converge.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let ladder = vec![
            Rung { config: RhfConfig { use_sad_guess: false, max_iter: 3, ..Default::default() }, restart: false },
            Rung { config: RhfConfig { use_sad_guess: false, max_iter: 6, ..Default::default() }, restart: false },
        ];
        let lr = solve_rhf_ladder(&ctx, &mol, &prep, op, &bounds, &ladder).unwrap();
        assert!(lr.converged, "ladder should converge by rung 2 if carry-forward seeds it with rung 1's density");
        assert_eq!(lr.rung_reached, 1, "should have advanced to rung 2");
        assert_eq!(lr.rung_outcomes[0].exit, ScfExit::MaxIter, "rung 1 must NOT converge in only 3 iters from cold hcore");
        assert!(
            lr.rung_outcomes[1].iters <= 6,
            "rung 2 should converge within its 6-iter budget only because it inherited rung 1's density; \
             got {} iters (exit={:?}) -- carry-forward may be broken",
            lr.rung_outcomes[1].iters, lr.rung_outcomes[1].exit
        );
    }

    #[test]
    fn default_ladder_does_not_inject_dfjk_aux() {
        let l = default_ladder();
        assert!(l.len() >= 3);
        // The ladder must NOT silently switch J/K to density fitting: RI-JK is a
        // user-visible accuracy trade (~1e-4 Ha on water/STO-3G) and is opt-in
        // via [scf] df_j_aux/df_k_aux. `RhfConfig::default()` leaves both unset,
        // so every rung must too. See `default_ladder_from`'s "J/K is NOT
        // substituted" doc.
        assert!(
            l.iter().all(|r| r.config.df_j_aux.is_none()),
            "ladder must not inject DF-J aux when the base left it unset"
        );
        assert!(
            l.iter().all(|r| r.config.df_k_aux.is_none()),
            "ladder must not inject DF-K aux when the base left it unset"
        );
        assert_eq!(l[0].config.level_shift, 0.0);
        assert_eq!(l[1].config.level_shift, 0.0, "rung 1 adds ADIIS, not level shift yet");
        assert!(l[2].config.level_shift > 0.0, "rung 2 must add level shift");
        assert!(!l[2].restart, "rung 2 must inherit density");
    }

    /// KS-DFT (B3LYP) via the level-shift ladder: benzene/def2-SVP, the
    /// workload from docs/profiles-2026-07-14.md finding (2).
    ///
    /// KS-DFT closed-shell IS `solve_rhf` with `cfg.xc = Some(...)`, so the same
    /// virtual-block level-shift ladder that fixes heavy-atom RHF divergence
    /// applies unchanged — `ksdft_ladder` clones the B3LYP base into every rung
    /// (carrying `xc` / DFT grid / DF-JK aux). This guards two things:
    ///
    ///  1. `ksdft_ladder` produces a working ladder for a hybrid functional
    ///     (functional/grid/aux survive the clone into each rung — a regression
    ///     that dropped `xc` would silently run bare HF and give ~-230.7 Ha).
    ///  2. The B3LYP SCF converges via the ladder and lands the expected energy
    ///     (rung 1's level_shift=0 already suffices on this well-behaved system
    ///     under the ΔP convergence gate — the escalation rungs are the harder-
    ///     system safety net, exercised by unit tests on synthetic stalls).
    ///
    /// Ignored by default (~10 s release: production (75,110) grid × 12 atoms).
    #[test]
    #[ignore] // ~10 s release: benzene/def2-SVP B3LYP, production DFT grid
    fn benzene_dfb3lyp_ksdft_ladder_converges() {
        let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
        let bs = basis::bundled("def2-svp").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        // Base config mirrors the CLI ksdft path: B3LYP + DF-JK, max_iter=100.
        let base = RhfConfig {
            xc: Some("B3LYP".to_string()),
            max_iter: 100,
            ..Default::default()
        };
        let ladder = ksdft_ladder(&base);
        // Every rung must keep the functional (else it silently runs bare HF).
        assert!(ladder.iter().all(|r| r.config.xc.as_deref() == Some("B3LYP")),
            "ksdft_ladder must carry xc into every rung");
        assert!(ladder[0].config.df_k_aux.is_some(),
            "hybrid needs RI-K aux auto-defaulted");

        let lr = solve_rhf_ladder(&ctx, &mol, &prep, op, &bounds, &ladder).unwrap();
        assert!(lr.converged, "B3LYP ladder must converge benzene (reached rung {})", lr.rung_reached);
        // Sanity vs the profiling run's last energy (-232.0846616020); a bare-HF
        // regression would land near -230.78 instead.
        assert!(
            (lr.result.energy - (-232.0846729516)).abs() < 1e-3,
            "B3LYP/def2-SVP benzene E={:.10}, expected ~-232.0847", lr.result.energy
        );
    }

    /// `ksdft_ladder` structural guarantees (fast, no SCF): carries the base's
    /// functional/grid into every rung, escalates the level shift, and layers
    /// stall/divergence early-abort on top.
    #[test]
    fn ksdft_ladder_carries_xc_and_escalates() {
        let grid = ferric_dft::grid::AtomicGridConfig { n_radial: 99, n_angular: 302, ..Default::default() };
        let base = RhfConfig {
            xc: Some("PBE".to_string()),
            dft_grid: Some(grid.clone()),
            max_iter: 42,
            ..Default::default()
        };
        let l = ksdft_ladder(&base);
        // Efficacy-ordered: 5 rungs (guess+DIIS → +ADIIS → +shift → +shift+SOSCF
        // → +smearing).
        assert_eq!(l.len(), 5);
        // xc + custom grid survive into every rung.
        for r in &l {
            assert_eq!(r.config.xc.as_deref(), Some("PBE"));
            let g = r.config.dft_grid.as_ref().expect("grid carried");
            assert_eq!((g.n_radial, g.n_angular), (99, 302));
            assert_eq!(r.config.stall_window, Some(15));
            assert_eq!(r.config.divergence_tol, Some(0.5));
            assert!(!r.restart, "rungs inherit density");
        }
        // Rung 0 respects the caller's max_iter and no shift. The level shift is
        // introduced from rung 2 onward (rung 1 adds ADIIS, not a shift).
        assert_eq!(l[0].config.level_shift, 0.0);
        assert_eq!(l[0].config.max_iter, 42);
        assert_eq!(l[1].config.diis_flavor, crate::diis::DiisFlavor::Adiis);
        assert!(l[2].config.level_shift > 0.0, "rung 2 adds the level shift");
        assert!(l[3].config.level_shift > l[2].config.level_shift);
        assert!(l[3].config.newton_trigger > 0.0, "rung 3 adds SOSCF");
        assert!(l[4].config.smearing_sigma.is_some(), "rung 4 adds smearing");
        // Pure PBE (no exact exchange) still gets RI-J auto-defaulted for speed.
        assert!(l[0].config.df_j_aux.is_some(), "RI-J auto-defaulted when xc set");
    }

    /// A bare-HF base (no xc) must NOT get DF-JK aux auto-forced by ksdft_ladder
    /// — that's the `default_ladder` use-case; ksdft_ladder leaves HF J/K as the
    /// caller set them (here: unset → direct J/K).
    #[test]
    fn ksdft_ladder_bare_hf_keeps_direct_jk() {
        let base = RhfConfig::default(); // no xc, no aux
        let l = ksdft_ladder(&base);
        assert!(l[0].config.xc.is_none());
        assert!(l[0].config.df_j_aux.is_none(), "bare HF keeps direct J");
        assert!(l[0].config.df_k_aux.is_none(), "bare HF keeps direct K");
    }

    /// STALE REFERENCE — re-derive before trusting. The `-1731.3137` expectation
    /// and the "~1-2 min" runtime were both measured when `default_ladder_from`
    /// silently forced `def2-universal-jkfit` DF-JK. That substitution has been
    /// removed (density fitting is a method choice, not a convergence knob), so
    /// this now runs EXACT 4-index J/K on a Cu complex at aug-cc-pVTZ. Two
    /// consequences, neither verified: (a) the energy shifts by the RI fitting
    /// error the reference baked in — the 1e-2 Ha tolerance is loose enough that
    /// it may still pass by luck rather than by correctness; (b) exact 4-index on
    /// a heavy-atom diffuse-basis system is far more expensive than the budgeted
    /// 1-2 min. Kept `#[ignore]`d and flagged rather than silently left wrong.
    /// To restore the original intent, set `df_j_aux`/`df_k_aux` explicitly on
    /// the base config here and re-derive the reference energy.
    #[test]
    #[ignore] // STALE: see doc above — reference predates DF-JK removal
    fn ccun_atz_default_ladder_converges() {
        let xyz = "3\nmol\nC 0.0000 0.0000 0.0000\nN 0.0000 0.0000 1.158\nCu 0.0000 0.0000 -1.832\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("aug-cc-pvtz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let lr = solve_rhf_ladder(&ctx, &mol, &prep, op, &bounds, &default_ladder()).unwrap();
        assert!(lr.converged, "CCuN/aTZ must converge via default ladder");
        assert!((lr.result.energy - (-1731.3137)).abs() < 1e-2,
            "CCuN/aTZ E={:.6}, expected ~-1731.3137", lr.result.energy);
        assert_eq!(lr.rung_reached, 0, "DF-JK rung 1 should suffice for CCuN");
    }
}
