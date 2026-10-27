//! End-to-end cDFT identities on LiH/def2-SVP. No external reference — all
//! checks are exact identities or self-consistency conditions.

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::cdft::{population, Constraint, SpinChannel};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cdft_driver::solve_cdft_uhf;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

fn setup() -> (
    Molecule,
    basis::BasisSet,
    PreparedBasis,
    Operator,
    SchwarzBounds,
) {
    let mol = Molecule::parse_xyz("2\nLiH\nLi 0 0 0\nH 0 0 1.60\n", 0, 1).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    (mol, bs, prep, op, bounds)
}

fn setup_hf() -> (
    Molecule,
    basis::BasisSet,
    PreparedBasis,
    Operator,
    SchwarzBounds,
) {
    let mol = Molecule::parse_xyz("2\nHF\nH 0 0 0\nF 0 0 0.917\n", 0, 1).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    (mol, bs, prep, op, bounds)
}

/// N_C(λ) responds to λ with a consistent sign on LiH near λ = 0 — the correct
/// saddle-point direction. Sample three λ values via single inner solves (Total
/// channel adds +λW to both spins) and check N_C moves monotonically across
/// them.
///
/// SCOPE — this is a LOCAL sign check on ONE system, not a global monotonicity
/// claim. c(λ) is not smooth in general: on HeNe⁺/def2-SVP the response is a
/// near-step function with two reachable branches (at R = 2.0 Å, λ = 0 gives
/// N(He) = 1.9307 and λ = +0.5 gives 1.0124). The window sampled here,
/// λ ∈ [−0.1, +0.1] on LiH, is narrow and well inside a smooth region.
/// See the `cdft_driver.rs` module doc for the full measured scan.
#[test]
fn population_responds_to_lambda_with_a_consistent_sign_on_lih() {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::cdft::build_weight_matrix;
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
    use ndarray::Array2;

    let (mol, bs, prep, _op, bounds) = setup();
    let ctx = ParallelContext::default();

    // W^Li once on the 302-pt angular grid (matches the driver).
    let grid = build_atomic_grid(
        &mol,
        &AtomicGridConfig {
            n_radial: 99,
            n_angular: 302,
            ..Default::default()
        },
    );
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(&mol, &bs, &pts).unwrap();
    let w = build_weight_matrix(&mol, &grid, &chi, &[0]);

    // N_Li at three increasing λ. Assert strict monotonicity.
    let mut pops = Vec::new();
    for &lam in &[-0.10_f64, 0.0, 0.10] {
        let wl = &w;
        let fm = move |f_a: &mut Array2<f64>, f_b: &mut Array2<f64>| {
            let lw = lam * wl;
            *f_a += &lw;
            *f_b += &lw;
        };
        let scf = ferric_scf::uhf::solve_uhf_fockmod(
            &ctx,
            &mol,
            &prep,
            &bounds,
            &RhfConfig::default(),
            None,
            Some(&fm),
        )
        .unwrap();
        let d_a = &scf.density_alpha;
        let d_b = scf.density_beta.as_ref().unwrap_or(d_a);
        pops.push(population(&w, d_a, d_b, &SpinChannel::Total));
    }
    // Strictly monotonic (one direction; sign depends on the convention above).
    let increasing = pops[0] < pops[1] && pops[1] < pops[2];
    let decreasing = pops[0] > pops[1] && pops[1] > pops[2];
    assert!(
        increasing || decreasing,
        "N_Li not monotonic in λ: {pops:?}"
    );
}

/// The constraint is actually satisfied at convergence.
#[test]
fn charge_constraint_is_satisfied() {
    let (mol, bs, prep, _op, bounds) = setup();
    let ctx = ParallelContext::default();
    let target = 2.2; // pull a bit of charge onto Li (Z=3)
    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![0],
            spin: SpinChannel::Total,
            target,
        }],
        cdft_lambda_tol: 1e-5,
        // Pinned so "this path converges" is not also an assertion about
        // how many outer iterations a particular CPU needs -- CI and this
        // box differ by more than the old hardcoded 30. See `cdft_max_outer`.
        //
        // 40, not higher: every path in this repo that converges at all does
        // so in <= 23 outer iters locally, and the cap is also the price paid
        // by paths that NEVER converge (each wasted outer iteration runs a
        // full inner SCF). At 64 this file took 462 s; the non-convergent
        // R = 2.5/3.0 sigma points burn the whole cap before failing.
        cdft_max_outer: 40,
        fractional_occ: false,
        ..Default::default()
    };
    let res = solve_cdft_uhf(&ctx, &mol, &prep, &bs, &bounds, &cfg).unwrap();
    assert!(
        (res.populations[0] - target).abs() < 1e-5,
        "Li pop {} vs target {target}",
        res.populations[0]
    );
    assert!(res.scf.converged, "inner SCF not converged");
}

/// Third system for the charge constraint (widens past LiH-family): HF
/// molecule, closed-shell singlet, def2-SVP. Different chemistry from LiH
/// (a strongly polar, closed-shell diatomic with a genuinely electronegative
/// heavy atom rather than an electropositive one) exercising the SAME
/// `SpinChannel::Total` charge-constraint code path.
#[test]
fn charge_constraint_is_satisfied_hf_molecule() {
    let (mol, bs, prep, _op, bounds) = setup_hf();
    let ctx = ParallelContext::default();
    // F is atom index 1; pull extra charge onto F (already electronegative,
    // so a modest increase over the unconstrained baseline is physical).
    let target = 9.3;
    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![1],
            spin: SpinChannel::Total,
            target,
        }],
        cdft_lambda_tol: 1e-5,
        // Pinned so "this path converges" is not also an assertion about
        // how many outer iterations a particular CPU needs -- CI and this
        // box differ by more than the old hardcoded 30. See `cdft_max_outer`.
        //
        // 40, not higher: every path in this repo that converges at all does
        // so in <= 23 outer iters locally, and the cap is also the price paid
        // by paths that NEVER converge (each wasted outer iteration runs a
        // full inner SCF). At 64 this file took 462 s; the non-convergent
        // R = 2.5/3.0 sigma points burn the whole cap before failing.
        cdft_max_outer: 40,
        fractional_occ: false,
        ..Default::default()
    };
    let res = solve_cdft_uhf(&ctx, &mol, &prep, &bs, &bounds, &cfg).unwrap();
    assert!(
        (res.populations[0] - target).abs() < 1e-5,
        "F pop {} vs target {target}",
        res.populations[0]
    );
    assert!(res.scf.converged, "inner SCF not converged");
}

/// Second constraint case: a genuine SPIN constraint (`SpinChannel::SpinDiff`),
/// not another charge constraint. Every other end-to-end cDFT test in this
/// file (and `cdft_weight.rs`) only ever exercises `SpinChannel::Total`; the
/// `SpinDiff` arm of `population`/the driver's fock_mod sign convention
/// (`+λW` on α, `−λW` on β, vs `Total`'s `+λW` on both) had never been run
/// through the outer Newton loop before this test. This is a different
/// molecule from the `charge_constraint_is_satisfied`/`population_responds_to_lambda_with_a_consistent_sign_on_lih`
/// LiH cases above: LiH⁺ (charge +1, doublet, 3 electrons — an open-shell
/// system, required because a closed-shell singlet has Nα=Nβ globally and
/// there is no unpaired spin density to redistribute onto a fragment).
///
/// Baseline (unconstrained UHF) puts N_α−N_β on Li at ≈0.274 e (the doublet's
/// one unpaired electron is mostly delocalized/H-centered). A manual λ-scan
/// (not part of the assertions here) showed a smooth, strictly monotonic,
/// cliff-free response over λ∈[-0.5,+1.0] (spin_Li: 0.558 → 0.274 → 0.135),
/// unlike He₂⁺'s plateau-then-cliff charge response — so this case does not
/// need He₂⁺'s loose tolerance/level-shift workarounds. Target 0.45 e sits
/// inside that smooth region, roughly the midpoint between baseline and the
/// λ=-0.5 sample.
#[test]
fn spin_constraint_is_satisfied_lih_plus() {
    let mol = Molecule::parse_xyz("2\nLiH+\nLi 0 0 0\nH 0 0 1.60\n", 1, 2).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let target = 0.45; // pull more of the unpaired spin density onto Li
    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![0],
            spin: SpinChannel::SpinDiff,
            target,
        }],
        cdft_lambda_tol: 1e-5,
        // Pinned so "this path converges" is not also an assertion about
        // how many outer iterations a particular CPU needs -- CI and this
        // box differ by more than the old hardcoded 30. See `cdft_max_outer`.
        //
        // 40, not higher: every path in this repo that converges at all does
        // so in <= 23 outer iters locally, and the cap is also the price paid
        // by paths that NEVER converge (each wasted outer iteration runs a
        // full inner SCF). At 64 this file took 462 s; the non-convergent
        // R = 2.5/3.0 sigma points burn the whole cap before failing.
        cdft_max_outer: 40,
        fractional_occ: false,
        ..Default::default()
    };
    let res = solve_cdft_uhf(&ctx, &mol, &prep, &bs, &bounds, &cfg).unwrap();
    assert!(
        (res.populations[0] - target).abs() < 1e-5,
        "Li spin pop {} vs target {target}",
        res.populations[0]
    );
    assert!(res.scf.converged, "inner SCF not converged");

    // Sanity: the converged λ must be genuinely nonzero (a zero λ would mean
    // the outer loop simply left the unconstrained density untouched, which
    // is not the target — baseline spin_Li is ≈0.274, target is 0.45).
    assert!(
        res.lambdas[0].abs() > 1e-3,
        "lambda suspiciously near zero: {}",
        res.lambdas[0]
    );
}

/// cDFT's per-iteration UhfFockMod closure (constraint) and the external
/// potential's hcore-level addition are architecturally independent. The
/// outer λ-Newton loop is *designed* to hit `target` regardless of what else
/// perturbs hcore, so convergence + population-at-target alone would pass
/// identically even if `external_potential` were silently dropped somewhere
/// in the cDFT path. To get genuine evidence the point charge reached the
/// inner SCF, run the same constraint TWICE — once with no external
/// potential (baseline) and once with the point charge (perturbed) — and
/// check that the converged solution actually differs: the converged λ must
/// shift (λ absorbs exactly the hcore perturbation needed to hold the
/// population at the fixed target, so a nonzero shift is direct evidence the
/// point charge's Fock contribution was present during the inner solve), and
/// the converged total energy must differ (which also picks up the classical
/// `ext.charge_nuclear_energy(&mol)` term added to `vnn`).
#[test]
fn cdft_composes_with_external_point_charge() {
    let (mol, bs, prep, _op, bounds) = setup();
    let ctx = ParallelContext::default();
    let target = 2.2; // same target as charge_constraint_is_satisfied

    let base_cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![0],
            spin: SpinChannel::Total,
            target,
        }],
        cdft_lambda_tol: 1e-5,
        // Pinned so "this path converges" is not also an assertion about
        // how many outer iterations a particular CPU needs -- CI and this
        // box differ by more than the old hardcoded 30. See `cdft_max_outer`.
        //
        // 40, not higher: every path in this repo that converges at all does
        // so in <= 23 outer iters locally, and the cap is also the price paid
        // by paths that NEVER converge (each wasted outer iteration runs a
        // full inner SCF). At 64 this file took 462 s; the non-convergent
        // R = 2.5/3.0 sigma points burn the whole cap before failing.
        cdft_max_outer: 40,
        fractional_occ: false,
        external_potential: None,
        ..Default::default()
    };
    let base = solve_cdft_uhf(&ctx, &mol, &prep, &bs, &bounds, &base_cfg).unwrap();
    assert!(base.scf.converged, "baseline inner SCF not converged");
    assert!(
        (base.populations[0] - target).abs() < 1e-5,
        "baseline Li pop {} vs target {target}",
        base.populations[0]
    );

    let ext = ExternalPotential {
        point_charges: vec![PointCharge {
            q: 1.0,
            x: 0.0,
            y: 0.0,
            z: 20.0,
        }],
        smeared_charges: Vec::new(),
        field: None,
    };
    let perturbed_cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![0],
            spin: SpinChannel::Total,
            target,
        }],
        cdft_lambda_tol: 1e-5,
        // Pinned so "this path converges" is not also an assertion about
        // how many outer iterations a particular CPU needs -- CI and this
        // box differ by more than the old hardcoded 30. See `cdft_max_outer`.
        //
        // 40, not higher: every path in this repo that converges at all does
        // so in <= 23 outer iters locally, and the cap is also the price paid
        // by paths that NEVER converge (each wasted outer iteration runs a
        // full inner SCF). At 64 this file took 462 s; the non-convergent
        // R = 2.5/3.0 sigma points burn the whole cap before failing.
        cdft_max_outer: 40,
        fractional_occ: false,
        external_potential: Some(ext),
        ..Default::default()
    };
    let perturbed = solve_cdft_uhf(&ctx, &mol, &prep, &bs, &bounds, &perturbed_cfg).unwrap();
    assert!(
        perturbed.scf.converged,
        "inner SCF not converged with cDFT + external potential"
    );
    assert!(
        (perturbed.populations[0] - target).abs() < 1e-5,
        "Li pop {} vs target {target} (cDFT constraint broken by external potential)",
        perturbed.populations[0]
    );

    // The population constraint is satisfied identically in both runs (by
    // construction of the outer loop) — the real evidence is that reaching
    // it required a different λ and produced a different converged energy.
    assert!(
        (perturbed.lambdas[0] - base.lambdas[0]).abs() > 1e-6,
        "lambda did not shift: base {} vs perturbed {} (point charge's Fock \
         contribution may not have reached the inner SCF)",
        base.lambdas[0],
        perturbed.lambdas[0]
    );
    assert!(
        (perturbed.scf.energy - base.scf.energy).abs() > 1e-6,
        "energy did not change: base {} vs perturbed {}",
        base.scf.energy,
        perturbed.scf.energy
    );
}

/// λ = 0 (no constraint added) reproduces plain UHF. Exercises the
/// solve_uhf_fockmod plumbing directly: an empty closure must not change the
/// converged energy.
///
/// Note: this is *not* a bit-for-bit comparison. Two independent `solve_uhf`
/// runs on identical inputs already differ at the ULP level (~2e-15) because
/// the J/K builds reduce float sums under rayon in a nondeterministic order.
/// The meaningful "no behavior change" criterion is therefore agreement to the
/// SCF's own energy convergence (1e-10), which the empty closure trivially
/// preserves (it is a literal no-op).
#[test]
fn lambda_zero_equals_plain_uhf() {
    let (mol, _bs, prep, _op, bounds) = setup();
    let ctx = ParallelContext::default();
    let plain = solve_uhf(&ctx, &mol, &prep, &bounds, &RhfConfig::default()).unwrap();
    let fm = |_fa: &mut ndarray::Array2<f64>, _fb: &mut ndarray::Array2<f64>| {};
    let modded = ferric_scf::uhf::solve_uhf_fockmod(
        &ctx,
        &mol,
        &prep,
        &bounds,
        &RhfConfig::default(),
        None,
        Some(&fm),
    )
    .unwrap();
    assert!(
        (plain.energy - modded.energy).abs() < 1e-10,
        "λ=0 fock_mod changed the energy: plain {} vs modded {}",
        plain.energy,
        modded.energy
    );
}

// ---------------------------------------------------------------------------
// Plateau conditioning: the driver must not report success on a solve in which
// the constraint was never applied.
//
// HeNe⁺/def2-SVP (charge +1, doublet, nbf = 19) is the reproducer. Its
// N(λ) response is a near-step function with two reachable branches
// (N(He) ≈ 1.93–1.97 and ≈ 1.00), so a target a few hundredths from the
// unconstrained population sits on a flat plateau. See the module
// doc of `cdft_driver.rs` for the measured λ scan.
// ---------------------------------------------------------------------------

/// HeNe⁺ setup at separation `r` Å, plus the (99,302) grid config the driver
/// defaults to. Returned explicitly so the tests and the driver agree.
fn setup_hene(
    r: f64,
) -> (
    Molecule,
    basis::BasisSet,
    PreparedBasis,
    SchwarzBounds,
    ferric_dft::grid::AtomicGridConfig,
) {
    let xyz = format!("2\nHeNe+\nHe 0 0 0\nNe 0 0 {r}\n");
    let mol = Molecule::parse_xyz(&xyz, 1, 2).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let gcfg = ferric_dft::grid::AtomicGridConfig {
        n_radial: 99,
        n_angular: 302,
        ..Default::default()
    };
    (mol, bs, prep, bounds, gcfg)
}

/// Unconstrained N(fragment) for HeNe⁺, on the driver's own grid.
fn hene_free_population(
    mol: &Molecule,
    bs: &basis::BasisSet,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    gcfg: &ferric_dft::grid::AtomicGridConfig,
    cfg: &RhfConfig,
    frag: usize,
) -> (f64, f64) {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::cdft::build_weight_matrix;
    use ferric_dft::grid::build_atomic_grid;

    let ctx = ParallelContext::default();
    let scf = solve_uhf(&ctx, mol, prep, bounds, cfg).unwrap();
    let grid = build_atomic_grid(mol, gcfg);
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(mol, bs, &pts).unwrap();
    let w = build_weight_matrix(mol, &grid, &chi, &[frag]);
    drop(chi);
    let d_a = &scf.density_alpha;
    let d_b = scf.density_beta.as_ref().unwrap_or(d_a);
    (population(&w, d_a, d_b, &SpinChannel::Total), scf.energy)
}

/// EXACTNESS ANCHOR for the plateau guard, written and run before it.
///
/// A constraint whose target IS the unconstrained population must still be
/// accepted: λ stays at 0, the energy is plain UHF's, and the guard below must
/// not fire. This is the trivial limit in which the constraint does nothing,
/// and it is the case the guard is most at risk of breaking — a guard keyed on
/// "λ never moved" would reject it if it did not also look at the residual.
#[test]
fn hene_vacuous_constraint_is_still_accepted() {
    let (mol, bs, prep, bounds, gcfg) = setup_hene(2.0);
    let ctx = ParallelContext::default();
    let base = RhfConfig {
        level_shift: 0.5,
        dft_grid: Some(gcfg.clone()),
        fractional_occ: false,
        ..Default::default()
    };
    let (n_free, e_free) = hene_free_population(&mol, &bs, &prep, &bounds, &gcfg, &base, 1);

    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![1],
            spin: SpinChannel::Total,
            target: n_free,
        }],
        cdft_lambda_tol: 1e-6,
        ..base
    };
    let res = solve_cdft_uhf(&ctx, &mol, &prep, &bs, &bounds, &cfg)
        .expect("vacuous constraint must not be rejected by the plateau guard");
    assert_eq!(res.lambdas[0], 0.0, "vacuous constraint moved λ");
    assert!(
        res.max_residual() <= 1e-6,
        "vacuous residual {:.3e} should be at the quadrature floor",
        res.max_residual()
    );
    assert!(
        (res.scf.energy - e_free).abs() < 1e-10,
        "vacuous constraint changed the energy: {} vs plain {e_free}",
        res.scf.energy
    );
}

/// THE GUARD FIRES. With `cdft_lambda_tol` looser than the distance from the
/// unconstrained population to the target, the outer loop used to exit on
/// iteration 1 with λ = 0 and return the UNCONSTRAINED state as if it were
/// constrained: `Ok`, with the plain-UHF energy, off target by the whole gap.
/// MEASURED on origin/main before the guard (R = 2.0 Å, tolerance 1e-1, every
/// `level_shift` below, stability descent on and off): λ = 0, outer = 1,
/// E = −130.50534054 (plain UHF), residual ±6.93e-2, all returned `Ok`.
///
/// The tolerance is DERIVED from the measured gap (twice it), not hard-coded,
/// so the test keeps exercising the λ = 0 exit if a guess change moves the
/// unconstrained state; the gap is asserted to be far above the vacuous floor,
/// so the pass condition is reachable only through the guard. Both formulations
/// are checked because they are the same state expressed on either fragment
/// (N(He) = 2 and N(Ne) = 9 on an 11-electron cation). The `level_shift` sweep
/// rules out an inner-solve stiffness artifact.
#[test]
fn plateau_side_target_with_loose_tolerance_is_rejected() {
    let (mol, bs, prep, bounds, gcfg) = setup_hene(2.0);
    let ctx = ParallelContext::default();
    for &(frag, target) in &[(1usize, 9.0_f64), (0usize, 2.0_f64)] {
        for &ls in &[0.2_f64, 0.5, 0.8] {
            let base = RhfConfig {
                dft_grid: Some(gcfg.clone()),
                level_shift: ls,
                max_iter: 400,
                fractional_occ: false,
                ..Default::default()
            };
            let (n_free, _) = hene_free_population(&mol, &bs, &prep, &bounds, &gcfg, &base, frag);
            let gap = (n_free - target).abs();
            assert!(
                gap > 1e-2,
                "frag {frag} ls {ls}: gap {gap:.3e} too small to separate the guard from \
                 the vacuous floor"
            );
            let cfg = RhfConfig {
                constraints: vec![Constraint {
                    fragment: vec![frag],
                    spin: SpinChannel::Total,
                    target,
                }],
                cdft_lambda_tol: 2.0 * gap,
                ..base
            };
            let err = solve_cdft_uhf(&ctx, &mol, &prep, &bs, &bounds, &cfg).expect_err(&format!(
                "frag {frag} target {target} level_shift {ls}: a λ = 0 return with the \
                 constraint violated by {gap:.3e} e must not be reported as success"
            ));
            let msg = format!("{err}");
            assert!(
                msg.contains("the constraint was never applied"),
                "wrong error for frag {frag} ls {ls}: {msg}"
            );
            assert!(
                msg.contains("re-express the constraint on the other fragment"),
                "error must name the remedy: {msg}"
            );
        }
    }
}

/// A LOOSE tolerance that the loop meets only AFTER moving λ is not vacuous,
/// and is accepted with its residual reported honestly.
///
/// The guard is keyed on "λ never moved", not on "the tolerance is loose", so
/// a loose solve in which the constraint term did reach the Fock must pass.
/// MEASURED: at `cdft_lambda_tol = 5e-2` the λ = 0 solve misses N(Ne) = 9 by
/// 6.93e-2, one Newton step lands at λ ≈ +0.511 with residual +4.73e-2, and the
/// loop stops there on outer iteration 2.
#[test]
fn loose_tolerance_met_after_lambda_moved_is_accepted() {
    let (mol, bs, prep, bounds, gcfg) = setup_hene(2.0);
    let ctx = ParallelContext::default();
    let target = 9.0;
    let tol = 5e-2;
    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![1],
            spin: SpinChannel::Total,
            target,
        }],
        cdft_lambda_tol: tol,
        dft_grid: Some(gcfg.clone()),
        level_shift: 0.5,
        max_iter: 400,
        fractional_occ: false,
        cdft_stability_descent: false,
        ..Default::default()
    };
    let res = solve_cdft_uhf(&ctx, &mol, &prep, &bs, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("a loose solve whose λ moved must be accepted: {e}"));
    assert!(res.lambdas[0] != 0.0, "λ did not move: {:?}", res.lambdas);
    assert!(res.outer_iters > 1, "outer = {}", res.outer_iters);
    assert!(
        res.max_residual() < tol,
        "residual {:.3e}",
        res.max_residual()
    );
    assert!(
        (res.residuals[0] - (res.populations[0] - target)).abs() < 1e-14,
        "residual {} != population {} − target {target}",
        res.residuals[0],
        res.populations[0]
    );
}

/// THE GUARD STAYS QUIET on the same two targets once the tolerance is tight
/// enough that the driver actually solves for λ.
///
/// This is the other half of the guard's pass condition: it must separate
/// "λ never moved" from "the response is flat", because a flat response is not
/// by itself an error. MEASURED: with the default stability descent both
/// formulations reach λ = ±2.4390, E = −130.42671, agreeing to 7e-6 Ha (without
/// the descent, the saddle at λ = ±2.7537, E = −130.40218); agreement is the
/// physical consistency check, since they describe the same state.
#[test]
fn cliff_side_targets_with_tight_tolerance_are_accepted() {
    let (mol, bs, prep, bounds, gcfg) = setup_hene(2.0);
    let ctx = ParallelContext::default();
    let mut energies = Vec::new();
    for &(frag, target) in &[(1usize, 9.0_f64), (0usize, 2.0_f64)] {
        let cfg = RhfConfig {
            constraints: vec![Constraint {
                fragment: vec![frag],
                spin: SpinChannel::Total,
                target,
            }],
            cdft_lambda_tol: 1e-5,
            dft_grid: Some(gcfg.clone()),
            level_shift: 0.5,
            fractional_occ: false,
            ..Default::default()
        };
        let res = solve_cdft_uhf(&ctx, &mol, &prep, &bs, &bounds, &cfg).unwrap_or_else(|e| {
            panic!("frag {frag} target {target} must still solve at tol 1e-5: {e}")
        });
        assert!(
            res.lambdas[0].abs() > 1e-6,
            "frag {frag}: λ = {} — the constraint was not applied",
            res.lambdas[0]
        );
        assert!(
            res.max_residual() < 1e-5,
            "frag {frag}: residual {:.3e} above tolerance",
            res.max_residual()
        );
        energies.push(res.scf.energy);
    }
    // The two formulations are the same physical state.
    assert!(
        (energies[0] - energies[1]).abs() < 1e-4,
        "N(Ne)=9 and N(He)=2 are the same state but gave {} and {}",
        energies[0],
        energies[1]
    );
}

/// `residuals` is populated and consistent with `populations` − target, for a
/// case that genuinely solves. Without this the new field could be silently
/// wrong (e.g. always zero) and every other assertion here would still pass.
#[test]
fn reported_residual_matches_population_minus_target() {
    let (mol, bs, prep, _op, bounds) = setup();
    let ctx = ParallelContext::default();
    let target = 2.2;
    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![0],
            spin: SpinChannel::Total,
            target,
        }],
        cdft_lambda_tol: 1e-5,
        fractional_occ: false,
        ..Default::default()
    };
    let res = solve_cdft_uhf(&ctx, &mol, &prep, &bs, &bounds, &cfg).unwrap();
    assert_eq!(res.residuals.len(), 1);
    assert!(
        (res.residuals[0] - (res.populations[0] - target)).abs() < 1e-14,
        "residual {} != population {} − target {target}",
        res.residuals[0],
        res.populations[0]
    );
    assert!((res.max_residual() - res.residuals[0].abs()).abs() < 1e-14);
}

/// The vacuous-constraint floor has SLACK, and the slack is load-bearing.
///
/// `hene_vacuous_constraint_is_still_accepted` passes the population straight
/// back as the target, so its residual is bit-identically 0.0 and it cannot
/// tell `VACUOUS_RESIDUAL_FLOOR = 1e-6` from `0.0` — verified by mutation, which
/// was INERT on that test. A real caller does not get an exact round trip: a
/// target read off a previous run, or computed on a slightly different grid,
/// lands a few times 1e-8 away. Offsetting the target by 1e-8 (well inside the
/// (99,302) grid's ~1e-8 population accuracy, and 100x below the floor) must
/// still be accepted, while the measured silent-wrong case sits four orders
/// ABOVE the floor at 6.9e-2.
#[test]
fn near_vacuous_constraint_within_the_quadrature_floor_is_accepted() {
    let (mol, bs, prep, bounds, gcfg) = setup_hene(2.0);
    let ctx = ParallelContext::default();
    let base = RhfConfig {
        level_shift: 0.5,
        dft_grid: Some(gcfg.clone()),
        fractional_occ: false,
        ..Default::default()
    };
    let (n_free, _) = hene_free_population(&mol, &bs, &prep, &bounds, &gcfg, &base, 1);

    // Offset by 1e-8: inside the grid's population accuracy, 100x below the
    // 1e-6 floor. Tolerance stays loose so the loop still exits on iteration 1
    // with λ = 0 — i.e. this takes exactly the code path the guard polices.
    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![1],
            spin: SpinChannel::Total,
            target: n_free + 1e-8,
        }],
        cdft_lambda_tol: 5e-2,
        ..base
    };
    let res = solve_cdft_uhf(&ctx, &mol, &prep, &bs, &bounds, &cfg).unwrap_or_else(|e| {
        panic!("a target 1e-8 from the unconstrained population is vacuous, not an error: {e}")
    });
    assert_eq!(res.lambdas[0], 0.0);
    assert_eq!(res.outer_iters, 1);
    assert!(
        res.max_residual() < 1e-6,
        "residual {:.3e} should be inside the quadrature floor",
        res.max_residual()
    );
}
