//! MEASUREMENT: TRAH vs the existing convergence ladder on hard systems.
//!
//! These are `#[ignore]`d benchmarks, not CI gates — they take minutes and
//! their job is to produce numbers, not to pass. Run one with:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release \
//!     --test trah_hard_systems -- --ignored --nocapture <name>
//! ```
//!
//! # What is being compared, and what is NOT
//!
//! The comparison is "the ladder as it exists today" vs "the same ladder with
//! one TRAH rung appended". The TRAH rung is appended in THIS FILE rather than
//! added to `ladder::default_ladder*` — the default ladder is deliberately left
//! alone so every existing result stays unchanged, and so this measurement says
//! what a TRAH rung WOULD buy rather than silently changing what ships.
//!
//! Iteration counts across rungs are summed, because a ladder that exhausts
//! five rungs has paid for all five. Wall time is reported alongside, since an
//! iteration of TRAH is more expensive than an iteration of DIIS (it pays a
//! Davidson eigensolve whose every matvec is a Fock build) and an
//! iteration-count win is not automatically a wall-time win.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ladder::{ksdft_ladder, solve_rhf_ladder, Rung};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::trah::TrahConfig;
use std::time::Instant;

/// The alpha-terpinyl cation: 27 atoms, closed-shell singlet, +1 charge.
/// PBE/6-31G. This is a system the current ladder is known to fail on.
const TERPINYL_XYZ: &str = include_str!("../../../testdata/molecules/terpinyl_relaxed.xyz");

fn report(label: &str, res: &ferric_scf::ladder::LadderResult, secs: f64) {
    let total: usize = res.rung_outcomes.iter().map(|o| o.iters).sum();
    eprintln!(
        "{label:<28} converged={:<5} rung={}/{} total_iters={total:<5} wall={secs:7.1}s  E={:.10}",
        res.converged,
        res.rung_reached,
        res.rung_outcomes.len().saturating_sub(1),
        res.result.energy
    );
    for (i, o) in res.rung_outcomes.iter().enumerate() {
        eprintln!(
            "    rung {i}: iters={:<4} exit={:?} E={:.10}",
            o.iters, o.exit, o.final_energy
        );
    }
}

/// Append one TRAH rung to a ladder, inheriting the last rung's settings.
///
/// Kept local to this test file on purpose: `ladder.rs` is not modified, so the
/// shipped ladder is unchanged and this measures a hypothetical, not a default.
fn with_trah_rung(mut ladder: Vec<Rung>, base: &RhfConfig, trigger: f64) -> Vec<Rung> {
    let mut cfg = ladder
        .last()
        .map(|r| r.config.clone())
        .unwrap_or_else(|| base.clone());
    cfg.trah_trigger = Some(trigger);
    cfg.trah = TrahConfig::default();
    cfg.smearing_sigma = None;
    cfg.newton_trigger = 0.0;
    cfg.max_iter = 100;
    ladder.push(Rung {
        config: cfg,
        restart: false,
    });
    ladder
}

/// alpha-terpinyl cation, PBE/6-31G — the system the current ladder FAILS on.
#[test]
#[ignore = "benchmark: minutes of wall time"]
fn terpinyl_cation_pbe_ladder_vs_trah() {
    let mol = Molecule::parse_xyz(TERPINYL_XYZ, 1, 1).expect("terpinyl geometry");
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    eprintln!(
        "alpha-terpinyl cation: {} atoms, {} basis functions, PBE/6-31G, charge +1",
        mol.atoms.len(),
        prep.nbasis()
    );

    let base = RhfConfig {
        xc: Some("PBE".into()),
        energy_conv: 1e-8,
        density_conv: 1e-7,
        ..Default::default()
    };

    let t0 = Instant::now();
    let plain = solve_rhf_ladder(&ctx, &mol, &prep, op, &bounds, &ksdft_ladder(&base)).unwrap();
    let t_plain = t0.elapsed().as_secs_f64();
    report("ladder (current)", &plain, t_plain);

    let ladder_trah = with_trah_rung(ksdft_ladder(&base), &base, 1e-2);
    let t1 = Instant::now();
    let trah = solve_rhf_ladder(&ctx, &mol, &prep, op, &bounds, &ladder_trah).unwrap();
    let t_trah = t1.elapsed().as_secs_f64();
    report("ladder + TRAH rung", &trah, t_trah);

    eprintln!(
        "TRAH steps taken = {}, rejected = {}",
        ferric_scf::trah::TRAH_STEPS_TAKEN.load(std::sync::atomic::Ordering::Relaxed),
        ferric_scf::trah::TRAH_STEPS_REJECTED.load(std::sync::atomic::Ordering::Relaxed)
    );
}

/// A second, smaller hard case so the comparison is not a single data point.
/// C2H4·Ar-style near-degenerate closed shell is not available here, so this
/// uses a stretched-geometry water cation triplet via the RHF ladder on a
/// cation, which stresses the same machinery at a size that runs quickly.
#[test]
#[ignore = "benchmark: minutes of wall time"]
fn benzene_b3lyp_ladder_vs_trah() {
    let mol = Molecule::load_xyz("testdata/molecules/benzene.xyz").expect("benzene");
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    eprintln!(
        "benzene: {} atoms, {} basis functions, B3LYP/6-31G",
        mol.atoms.len(),
        prep.nbasis()
    );

    let base = RhfConfig {
        xc: Some("B3LYP".into()),
        energy_conv: 1e-8,
        density_conv: 1e-7,
        ..Default::default()
    };

    let t0 = Instant::now();
    let plain = solve_rhf_ladder(&ctx, &mol, &prep, op, &bounds, &ksdft_ladder(&base)).unwrap();
    let t_plain = t0.elapsed().as_secs_f64();
    report("ladder (current)", &plain, t_plain);

    let ladder_trah = with_trah_rung(ksdft_ladder(&base), &base, 1e-2);
    let t1 = Instant::now();
    let trah = solve_rhf_ladder(&ctx, &mol, &prep, op, &bounds, &ladder_trah).unwrap();
    let t_trah = t1.elapsed().as_secs_f64();
    report("ladder + TRAH rung", &trah, t_trah);
}

/// Direct single-config comparison (no ladder) on water/cc-pVDZ RHF and
/// RKS/PBE, reporting iterations AND wall time.
///
/// The cheap case exists to make the cost side of the trade visible: TRAH pays
/// a Davidson eigensolve per step, so on a system DIIS already handles well it
/// can win on iterations and still lose on wall time.
#[test]
#[ignore = "benchmark"]
fn water_iteration_and_walltime_comparison() {
    let mol = Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n",
        0,
        1,
    )
    .unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    for (label, xc, ls) in [
        ("RHF", None, 0.0),
        ("RKS/PBE", Some("PBE".to_string()), 0.2),
    ] {
        let diis = RhfConfig {
            xc: xc.clone(),
            level_shift: ls,
            energy_conv: 1e-10,
            density_conv: 1e-8,
            max_iter: 300,
            ..Default::default()
        };
        let newton = RhfConfig {
            newton_trigger: 1e-2,
            ..diis.clone()
        };
        let trah = RhfConfig {
            trah_trigger: Some(1e-2),
            ..diis.clone()
        };

        for (name, cfg) in [("DIIS", &diis), ("Newton", &newton), ("TRAH", &trah)] {
            let t = Instant::now();
            let r = ferric_scf::rhf::solve_rhf(&ctx, &mol, &prep, op, &bounds, cfg).unwrap();
            eprintln!(
                "{label:<9} {name:<7} iters={:<4} wall={:7.3}s converged={} E={:.10}",
                r.iterations,
                t.elapsed().as_secs_f64(),
                r.converged,
                r.energy
            );
        }
    }
}
