//! COSX SCF grid schedule (`CosxConfig::schedule`, `crate::cosx_schedule`):
//! coarse grid early, production grid to convergence, final pass unchanged.
//!
//! * `schedule_off_is_inert`: the exactness anchor. Default off; with no
//!   schedule (or a schedule next to a non-COSX K) nothing is recorded and
//!   the SCF is bit-identical.
//! * `schedule_on_converges_to_the_production_energy`: the scheduled SCF lands
//!   on the unscheduled production-grid energy, with a tolerance placed
//!   between the two measured sides (see the test's doc).
//! * `schedule_uses_the_coarse_grid_early_and_converges_on_production`: the
//!   per-iteration record proves the coarse grid built the early K, the
//!   production grid built the converged one, and the converged density was
//!   itself produced by a production Fock.
//! * `schedule_switches_when_the_coarse_grid_would_converge_first`, the UHF
//!   case and the ROHF refusal cover the remaining branches.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{CosxConfig, CosxK};
use ferric_scf::cosx_schedule::{CosxGridPhase, CosxGridSchedule};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

fn water() -> Molecule {
    Molecule::parse_xyz(WATER, 0, 1).expect("water")
}

fn run_rhf(cfg: &RhfConfig) -> ScfResult {
    let mol = water();
    let bs = bundled("cc-pvdz").expect("cc-pvdz");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, cfg).expect("rhf")
}

/// `energy_conv` is the loose sanity bound and `density_conv` the tight
/// criterion (see `cosx_scf.rs::tight` for why 1e-7 / 1e-8).
fn cosx_rhf(cosx: CosxConfig) -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-7,
        density_conv: 1e-8,
        k_builder: Some("cosx".into()),
        cosx,
        ..Default::default()
    }
}

fn scheduled() -> CosxConfig {
    CosxConfig {
        schedule: Some(CosxGridSchedule::default()),
        ..CosxConfig::default()
    }
}

fn assert_same_bits(a: &ScfResult, b: &ScfResult, what: &str) {
    assert_eq!(
        a.energy.to_bits(),
        b.energy.to_bits(),
        "{what}: {:.17e} vs {:.17e}",
        a.energy,
        b.energy
    );
    assert_eq!(a.iterations, b.iterations, "{what}: iterations");
    assert!(
        a.density_total
            .iter()
            .zip(b.density_total.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "{what}: density bits differ"
    );
}

/// EXACTNESS ANCHOR (off = unchanged). The schedule is off by default, an
/// unscheduled COSX run records nothing, and a schedule carried next to a
/// non-COSX exchange builder never engages: bit-identical energy, density and
/// iteration count to the same run without it. (The off path was also
/// checked bit-for-bit against the pre-schedule commit, 255c6393, by hand:
/// the in-tree comparison cannot see the old code.)
#[test]
fn schedule_off_is_inert() {
    assert!(CosxConfig::default().schedule.is_none(), "must default OFF");
    let off = run_rhf(&cosx_rhf(CosxConfig::default()));
    assert!(off.converged);
    assert!(off.cosx_schedule.is_none());

    let base = RhfConfig {
        k_builder: Some("link".into()),
        ..cosx_rhf(CosxConfig::default())
    };
    let link = run_rhf(&base);
    let link_sched = run_rhf(&RhfConfig {
        cosx: scheduled(),
        ..base.clone()
    });
    assert!(link_sched.cosx_schedule.is_none());
    assert_same_bits(&link, &link_sched, "LinK with an inert schedule");
}

/// The scheduled SCF converges to the unscheduled (production-grid) energy.
///
/// TOLERANCE, derived from both sides (water/cc-pVDZ RHF, default sgx (35,194)
/// production grid + sgx (50,302) final pass, density_conv 1e-8; measured
/// 2026-10-08):
///
/// * defect side: an SCF converged on the COARSE grid alone has an SCF-grid
///   energy 1.89e-5 Ha off the production SCF (`d_coarse_scf`) — what "the
///   converged energy came from the coarse grid" would look like;
/// * correct side: the scheduled run differs from the unscheduled one by
///   1.1e-13 Ha (`d_sched`, both the SCF-grid and the final energy): two
///   converged SCFs stopping on the same fixed point.
///
/// `T = 1e-8` Ha is five orders above the correct side and three below the
/// defect side; the test asserts both sides so it fails if they ever stop
/// being separable. The defect is asserted on the SCF-GRID energy because the
/// final pass hides it: the coarse-only run's FINAL energy is only 3.6e-10 Ha
/// off (the final-grid energy is second order in the density error), below
/// `T`, so a final-energy-only check could not see a coarse-grid answer.
#[test]
fn schedule_on_converges_to_the_production_energy() {
    const T: f64 = 1e-8;
    let off = run_rhf(&cosx_rhf(CosxConfig::default()));
    let on = run_rhf(&cosx_rhf(scheduled()));
    let coarse_grid = CosxGridSchedule::default().coarse_grid;
    let coarse_only = run_rhf(&cosx_rhf(CosxConfig {
        grid: coarse_grid,
        ..CosxConfig::default()
    }));
    assert!(off.converged && on.converged && coarse_only.converged);
    let fe = |r: &ScfResult| r.cosx_final.expect("final pass ran");
    let d_sched = (on.energy - off.energy).abs();
    let d_sched_scf = (fe(&on).e_scf_grid - fe(&off).e_scf_grid).abs();
    let d_coarse = (coarse_only.energy - off.energy).abs();
    let d_coarse_scf = (fe(&coarse_only).e_scf_grid - fe(&off).e_scf_grid).abs();
    println!(
        "water/cc-pVDZ RHF COSX: off E={:.12} ({} it), scheduled E={:.12} ({} it, switch {:?}), \
         coarse-only E={:.12} ({} it); |on-off| final {d_sched:.3e} scf-grid {d_sched_scf:.3e}; \
         |coarse-off| final {d_coarse:.3e} scf-grid {d_coarse_scf:.3e}",
        off.energy,
        off.iterations,
        on.energy,
        on.iterations,
        on.cosx_schedule.as_ref().and_then(|s| s.switch_iter),
        coarse_only.energy,
        coarse_only.iterations,
    );
    assert!(
        d_sched < T && d_sched_scf < T,
        "scheduled run is off the production energy"
    );
    assert!(
        d_coarse_scf > 10.0 * T,
        "coarse-grid SCF-grid energy is within 10 T of production ({d_coarse_scf:e}): the \
         test can no longer tell a coarse-grid answer from a production one"
    );
}

/// The per-iteration record: coarse grid first, then production for good,
/// switch strictly before the converged iteration, and the coarse builds
/// really ran on the coarse grid — their point count equals an independently
/// constructed coarse `CosxK`, and is smaller than the production one.
///
/// Mutation record (2026-10-08): building the first COSX builder on the
/// production grid (coarse never used) fails here at iteration 1's point
/// count (10866 vs 4062). Letting the FIRST production iteration converge is
/// not reachable end to end (its ΔD/ΔE still carry the coarse step); the
/// unit test `cosx_schedule::tests::gate_refuses_coarse_and_first_production_iterations`
/// kills that mutation.
#[test]
fn schedule_uses_the_coarse_grid_early_and_converges_on_production() {
    let r = run_rhf(&cosx_rhf(scheduled()));
    assert!(r.converged);
    let rec = r.cosx_schedule.clone().expect("schedule record");
    let n = r.iterations;
    assert_eq!(rec.phases.len(), n, "one record per iteration");
    let sw = rec.switch_iter.expect("switched to production");
    println!(
        "phases {:?} npts {:?} switch {sw} of {n}",
        rec.phases, rec.npts
    );
    assert!(sw >= 2, "iteration 1 must be on the coarse grid");
    // The converged density came from a production Fock: at least one full
    // production iteration precedes the converged one.
    assert!(
        sw < n,
        "converged on the switch iteration (switch {sw}, n {n})"
    );
    for (i, p) in rec.phases.iter().enumerate() {
        let want = if i + 1 < sw {
            CosxGridPhase::Coarse
        } else {
            CosxGridPhase::Production
        };
        assert_eq!(*p, want, "iteration {}", i + 1);
    }

    let mol = water();
    let bs = bundled("cc-pvdz").expect("cc-pvdz");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let ctx = ParallelContext::default();
    let npts_of = |c: CosxConfig| CosxK::new(&ctx, &mol, &prep, c, 0).expect("cosx").npts();
    let coarse = npts_of(CosxConfig {
        grid: CosxGridSchedule::default().coarse_grid,
        ..CosxConfig::default()
    });
    let production = npts_of(CosxConfig::default());
    println!("water grid points: coarse {coarse}, production {production}");
    assert!(coarse < production);
    for (i, &np) in rec.npts.iter().enumerate() {
        let want = if i + 1 < sw { coarse } else { production };
        assert_eq!(np, want, "grid points of iteration {}", i + 1);
    }
}

/// A switch threshold BELOW the convergence gate (`switch_dp_max` 1e-12 vs
/// `dp_max < 1e-7`) means the coarse SCF reaches the gate before the ΔD
/// trigger fires. The gate must still refuse the coarse answer and switch:
/// the run ends on the production grid at the production energy.
///
/// Mutation record (2026-10-08): letting the coarse phase pass `conv`
/// through survives `schedule_uses_the_coarse_grid_early_...` (at the default
/// 1e-3 the switch always fires first, so that branch is unreachable there)
/// and is killed here.
#[test]
fn schedule_switches_when_the_coarse_grid_would_converge_first() {
    let off = run_rhf(&cosx_rhf(CosxConfig::default()));
    let r = run_rhf(&cosx_rhf(CosxConfig {
        schedule: Some(CosxGridSchedule {
            switch_dp_max: 1e-12,
            ..CosxGridSchedule::default()
        }),
        ..CosxConfig::default()
    }));
    assert!(r.converged);
    let rec = r.cosx_schedule.clone().expect("record");
    let sw = rec
        .switch_iter
        .expect("must switch, not converge on coarse");
    println!("switch {sw} of {}: {:?}", r.iterations, rec.phases);
    assert!(sw < r.iterations);
    assert_eq!(*rec.phases.last().unwrap(), CosxGridPhase::Production);
    assert!((r.energy - off.energy).abs() < 1e-8);
}

/// Open shell: the UHF loop carries the same schedule and lands on the same
/// production energy (same tolerance as the RHF test).
#[test]
fn uhf_schedule_on_converges_to_the_production_energy() {
    let mol = Molecule::parse_xyz("2\noh\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
    let bs = bundled("6-31g").expect("6-31g");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let solve = |c: CosxConfig| solve_uhf(&ctx, &mol, &prep, &bounds, &cosx_rhf(c)).unwrap();
    let off = solve(CosxConfig::default());
    let on = solve(scheduled());
    assert!(off.converged && on.converged);
    let rec = on.cosx_schedule.clone().expect("record");
    let sw = rec.switch_iter.expect("switched");
    println!(
        "OH/6-31G UHF COSX: off {:.12} ({} it), on {:.12} ({} it, switch {sw}) |d| {:.3e}",
        off.energy,
        off.iterations,
        on.energy,
        on.iterations,
        (on.energy - off.energy).abs()
    );
    assert!(sw >= 2 && sw < on.iterations);
    assert_eq!(rec.phases[0], CosxGridPhase::Coarse);
    assert_eq!(*rec.phases.last().unwrap(), CosxGridPhase::Production);
    assert!((on.energy - off.energy).abs() < 1e-8);
}

/// ROHF/ROKS has no schedule support: a configured schedule with COSX K is a
/// hard error, never silently ignored.
#[test]
fn rohf_refuses_the_schedule() {
    let mol = Molecule::parse_xyz("2\noh\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
    let bs = bundled("6-31g").expect("6-31g");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let e = ferric_scf::rohf::solve_rohf(
        &ParallelContext::default(),
        &mol,
        &prep,
        op,
        &bounds,
        &cosx_rhf(scheduled()),
    )
    .unwrap_err();
    assert!(e.to_string().contains("grid schedule"), "{e}");
}
