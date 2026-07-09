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

#[derive(Clone)]
pub struct Rung {
    /// Partial RHF config for this rung.
    pub config: RhfConfig,
    /// false: inherit the previous rung's final density (default, progressive
    /// refinement). true: discard incoming density and use this rung's own guess.
    pub restart: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct RungOutcome {
    pub iters: usize,
    pub exit: ScfExit,
    pub final_err_max: f64, // best-effort; set to f64::NAN if not surfaced
    pub final_energy: f64,
}

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

/// Built-in default ladder: DF-JK throughout, stall/divergence abort on every
/// rung, level-shift escalation, density carried forward. Tuned from CCuN/aTZ
/// measurements (2026-07-08): rung 1 alone banks CCuN in <1 min.
pub fn default_ladder() -> Vec<Rung> {
    let base = |level_shift: f64, max_iter: usize| RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        df_k_aux: Some("def2-universal-jkfit".to_string()),
        level_shift,
        max_iter,
        stall_window: Some(15),
        divergence_tol: Some(0.5),
        ..Default::default()
    };
    vec![
        Rung { config: base(0.0, 60), restart: false },
        Rung { config: base(0.5, 60), restart: false },
        Rung { config: base(1.0, 80), restart: false },
    ]
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
    fn default_ladder_has_dfjk_first_rung() {
        let l = default_ladder();
        assert!(l.len() >= 2);
        assert!(l[0].config.df_j_aux.is_some(), "rung 1 must use DF-J");
        assert!(l[0].config.df_k_aux.is_some(), "rung 1 must use DF-K");
        assert_eq!(l[0].config.level_shift, 0.0);
        assert!(l[1].config.level_shift > 0.0, "rung 2 must add level shift");
        assert!(!l[1].restart, "rung 2 must inherit density");
    }

    #[test]
    #[ignore] // ~1-2 min release: CCuN/aTZ full neutral RHF via default ladder
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
