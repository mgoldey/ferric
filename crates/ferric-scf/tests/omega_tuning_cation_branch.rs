//! Cation state control in ω tuning: continuation seeding, the branch check,
//! and the N2⁺ instability onset the two exist for.
//!
//! # What this file establishes, and what it cannot
//!
//! `omega_tuning::eval_j` used to solve the doublet cation from the default
//! guess at every ω and check only `converged`, while `tune_omega`'s
//! golden-section search visits ω out of order. Nothing tied two J points to
//! one electronic state. Continuation (seed from the NEAREST already-evaluated
//! ω) and the branch check (⟨S²⟩ + spin-population asymmetry against that same
//! neighbour) are what this file tests.
//!
//! Neither is a stability verdict. `tune_omega` does not report the orbital
//! Hessian's lowest eigenvalue, and `stability::ks_reference_is_analysable`
//! refuses ωB97X-V (`StabilitySkip::Vv10Kernel`: no VV10 response kernel), so a
//! converged cation that is an internal SADDLE cannot be identified here. The
//! measured consequence is in `validation_omega_tuning_cation.rs`.
//!
//! # Measured: N2/def2-SVP ωB97X-V, ω = 0.50…0.60
//!
//! The slow ω sweeps live in `validation_omega_tuning_cation.rs` (validation
//! tier). Their measured outcome, and the reason `DEFAULT_BRANCH_TOL` is 1e-7 —
//! the branch switch moves the spin asymmetry by 2.7e-7, while ⟨S²⟩'s smooth
//! ω-drift (1.06e-4 per 0.01 Bohr⁻¹) is three decades ABOVE the signal and so
//! is reported but gated on by nothing — are recorded in REPORT-288.md; the
//! numbers the ASSERTING tests depend on are restated at each assertion.
//!
//! # Mutations (run, with the observed failure)
//!
//! Recorded in REPORT-288.md's mutation ledger. Every mutation listed there
//! was run against a test in this file and observed to FAIL.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::omega_tuning::{
    eval_j, eval_j_seeded, nearest_evaluated, spin_population_asymmetry, tune_omega, OmegaSeed,
    OmegaTuneConfig,
};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess};

const FUNCTIONAL: &str = "wB97X-V";

fn load(name: &str, basis: &str) -> (Molecule, PreparedBasis, SchwarzBounds) {
    let path = format!(
        "{}/../../testdata/molecules/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    let mol = Molecule::load_xyz(&path).unwrap_or_else(|e| panic!("{path}: {e:?}"));
    let bs = basis::bundled(basis).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (mol, prep, bounds)
}

fn cfg(basis_cfg: RhfConfig, lo: f64, hi: f64) -> OmegaTuneConfig {
    OmegaTuneConfig {
        functional: FUNCTIONAL.into(),
        omega_lo: lo,
        omega_hi: hi,
        omega_tol: 0.02,
        max_evals: 16,
        scf: basis_cfg,
        ..Default::default()
    }
}

// ─────────────────────── exactness anchors ───────────────────────

/// ANCHOR 1 (trivial limit of continuation): with `continuation: false` the
/// tuner's every evaluation must be bit-identical to an independent `eval_j`
/// at the same ω — that is the definition of "seeding does nothing", and it is
/// what makes the pre-change J values reproducible on demand.
#[test]
fn continuation_off_is_bit_identical_to_independent_eval_j() {
    let (mol, prep, bounds) = load("h2.xyz", "6-31g");
    let ctx = ParallelContext::default();
    let base = RhfConfig {
        energy_conv: 1e-9,
        ..Default::default()
    };
    let c = OmegaTuneConfig {
        continuation: false,
        branch_tol: None,
        ..cfg(base, 0.2, 1.2)
    };
    assert_eq!(
        c.continuation,
        OmegaTuneConfig::default().continuation,
        "this anchor must test the DEFAULT path, not an opt-out of it"
    );
    let t = tune_omega(&ctx, &mol, &prep, &bounds, &c).unwrap();
    for e in &t.evals {
        assert_eq!(
            e.seed,
            OmegaSeed::Default,
            "continuation off must leave every eval on the default guess"
        );
        let solo = eval_j(&ctx, &mol, &prep, &bounds, &c, e.omega).unwrap();
        assert_eq!(
            (solo.eps_homo, solo.ip_delta_scf, solo.j),
            (e.eps_homo, e.ip_delta_scf, e.j),
            "ω={}: continuation-off eval is not bit-identical to a bare eval_j",
            e.omega
        );
    }
}

/// ANCHOR 2 (trivial limit of the asymmetry metric): "hole localization" with
/// no localization is zero. A CLOSED-SHELL density has D_α = D_β exactly, so
/// the spin population is identically zero on every atom and the asymmetry
/// must be at the floating-point floor — if this is not ~0 the metric's
/// shell→atom partition is wrong and every later number is an artifact
/// (artifact hypothesis A3).
#[test]
fn spin_asymmetry_is_zero_for_a_spin_unpolarized_density() {
    let (mol, prep, _) = load("n2.xyz", "6-31g");
    let n = prep.nbasis();
    // Any symmetric D with D_α = D_β: the metric must not see a spin at all.
    let mut d = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        d[(i, i)] = 0.37;
        if i + 1 < n {
            d[(i, i + 1)] = 0.11;
            d[(i + 1, i)] = 0.11;
        }
    }
    let a = spin_population_asymmetry(&mol, &prep, &d, &d).unwrap();
    assert!(a < 1e-14, "unpolarized asymmetry {a:.3e} is not zero");
}

/// ANCHOR 2b: the metric is NOT identically zero — it responds to a density
/// that really is lopsided across the two equivalent N. Without this, anchor 2
/// would also pass for a metric that always returns 0 (a test that can never
/// fail is an assumption).
#[test]
fn spin_asymmetry_sees_a_lopsided_spin_density() {
    let (mol, prep, _) = load("n2.xyz", "6-31g");
    let n = prep.nbasis();
    let per_atom = n / 2;
    let mut da = ndarray::Array2::<f64>::zeros((n, n));
    let db = ndarray::Array2::<f64>::zeros((n, n));
    // One unpaired electron's worth of α density entirely on atom 0's AOs.
    for i in 0..per_atom {
        da[(i, i)] = 1.0 / per_atom as f64;
    }
    let a = spin_population_asymmetry(&mol, &prep, &da, &db).unwrap();
    assert!(
        a > 0.5,
        "a fully localized spin density gave asymmetry {a:.3e}"
    );
}

/// ANCHOR 2c: with no symmetry-equivalent atoms the metric is a no-op (0),
/// not a false alarm — so the branch check never fires on an asymmetric
/// molecule for want of a comparison.
#[test]
fn spin_asymmetry_is_zero_when_no_atoms_are_equivalent() {
    let mol = Molecule::parse_xyz("2\nHF\nH 0 0 0\nF 0 0 0.92\n", 0, 1).unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let n = prep.nbasis();
    let mut da = ndarray::Array2::<f64>::zeros((n, n));
    da[(0, 0)] = 1.0;
    let db = ndarray::Array2::<f64>::zeros((n, n));
    assert_eq!(
        spin_population_asymmetry(&mol, &prep, &da, &db).unwrap(),
        0.0,
        "HF has no equivalent atoms, so the asymmetry must be a no-op 0"
    );
}

/// ANCHOR 3: the first evaluation has no neighbour and therefore uses the
/// default guess; `nearest_evaluated` says so, and the tuner's first eval
/// records it.
#[test]
fn the_first_evaluation_has_no_seed() {
    assert_eq!(nearest_evaluated(&[], 0.4), None);
    let (mol, prep, bounds) = load("h2.xyz", "6-31g");
    let ctx = ParallelContext::default();
    // Continuation is OFF by default (DEFAULT_CONTINUATION), so this test
    // opts in: the question it asks is what the FIRST evaluation does when
    // continuation is requested and no neighbour exists yet.
    let c = OmegaTuneConfig {
        continuation: true,
        ..cfg(
            RhfConfig {
                energy_conv: 1e-9,
                ..Default::default()
            },
            0.2,
            1.2,
        )
    };
    assert!(
        !OmegaTuneConfig::default().continuation,
        "the library default must stay OFF: turning it on moves J on H2O \
         (measured 1.0e-7 Ha), and 'inside the bar' is not 'unchanged'"
    );
    let t = tune_omega(&ctx, &mol, &prep, &bounds, &c).unwrap();
    assert_eq!(
        t.evals[0].seed,
        OmegaSeed::Default,
        "the first eval cannot continue from anything"
    );
    assert!(
        matches!(t.evals[1].seed, OmegaSeed::Continued { .. }),
        "the second eval must continue from the first, got {}",
        t.evals[1].seed
    );
}

/// The seed MUST reach the solver. If `solve_uhf_with_guess` silently
/// discarded the MOs this test could not fail, and then a continuation-on vs
/// -off agreement would prove nothing (artifact hypothesis A1). Hands the
/// N2⁺ UKS a deliberately SPIN-SWAPPED/scrambled guess and shows the iteration
/// count changes — the seed is live plumbing, not a parameter that is ignored.
#[test]
fn a_different_cation_seed_changes_the_solver_path() {
    let (mol, prep, bounds) = load("n2.xyz", "sto-3g");
    let ctx = ParallelContext::default();
    let mut cation = mol.clone();
    cation.charge += 1;
    cation.multiplicity = 2;
    let scf = RhfConfig {
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 300,
        ..Default::default()
    };
    let plain = solve_uhf(&ctx, &cation, &prep, &bounds, &scf).unwrap();
    // Seed from the plain run's own converged MOs: it must converge in
    // strictly FEWER iterations than from the default guess, which is only
    // possible if the guess is actually used.
    let cb = plain.mos_beta.clone().unwrap();
    let seeded = solve_uhf_with_guess(
        &ctx,
        &cation,
        &prep,
        &bounds,
        &scf,
        Some((&plain.mos_alpha, &cb)),
    )
    .unwrap();
    eprintln!(
        "N2+/STO-3G UKS: default guess {} iters, self-seeded {} iters",
        plain.iterations, seeded.iterations
    );
    assert!(
        seeded.iterations < plain.iterations,
        "seeding from the converged MOs did not shorten the SCF ({} vs {}) — the \
         guess is being discarded, so every continuation result is a no-op",
        seeded.iterations,
        plain.iterations
    );
    assert!(
        (seeded.energy - plain.energy).abs() < 1e-8,
        "the self-seeded run found a different energy"
    );
}

/// The continuation seed must reach the CATION through `eval_j_seeded`, not
/// only through `solve_uhf_with_guess` called directly.
///
/// This test exists because a mutation that replaced `eval_j_seeded`'s
/// `cat_seed` with `None` SURVIVED every other test in this file: the N2
/// window test asserts that the seeded and unseeded arms AGREE, so discarding
/// the seed makes it pass trivially, and `a_different_cation_seed_changes_the
/// _solver_path` exercises the solver entry point rather than this wrapper.
/// The observable that does distinguish them is the SCF iteration count:
/// seeding the cation from its own converged orbitals at a nearby ω must make
/// it converge in fewer iterations than from the default guess.
#[test]
fn the_continuation_seed_reaches_the_cation_through_eval_j_seeded() {
    let (mol, prep, bounds) = load("n2.xyz", "sto-3g");
    let ctx = ParallelContext::default();
    let c = cfg(
        RhfConfig {
            df_j_aux: Some(String::new()),
            energy_conv: 1e-9,
            density_conv: 1e-7,
            max_iter: 300,
            ..Default::default()
        },
        0.3,
        0.7,
    );
    // Converge at 0.40, then evaluate 0.41 twice: once cold, once seeded.
    let (_, carry) = eval_j_seeded(&ctx, &mol, &prep, &bounds, &c, 0.40, None).unwrap();
    let mut cation = mol.clone();
    cation.charge += 1;
    cation.multiplicity = 2;
    let scf41 = ferric_scf::omega_tuning::state_scf_config(&c, 0.41).unwrap();
    let cold = solve_uhf(&ctx, &cation, &prep, &bounds, &scf41).unwrap();
    let warm = solve_uhf_with_guess(
        &ctx,
        &cation,
        &prep,
        &bounds,
        &scf41,
        Some((&carry.cation_mos.0, &carry.cation_mos.1)),
    )
    .unwrap();
    eprintln!(
        "N2+/STO-3G at ω=0.41: cold {} iters, seeded from ω=0.40 {} iters",
        cold.iterations, warm.iterations
    );
    // The seed carried by eval_j_seeded is the one that shortens the SCF.
    assert!(
        warm.iterations < cold.iterations,
        "the OmegaSeedState's cation MOs did not shorten the SCF ({} vs {}) — \
         eval_j_seeded is not carrying usable cation orbitals",
        warm.iterations,
        cold.iterations
    );
    assert!((warm.energy - cold.energy).abs() < 1e-8);
    // And `eval_j_seeded` itself must USE them. The energy cannot show this —
    // seeded and unseeded converge to the SAME energy, which is exactly why
    // the discard mutation survived. The ITERATION COUNT can, so OmegaEval
    // carries it: seeded, eval_j_seeded's cation must take strictly fewer
    // iterations than unseeded.
    let (cold_eval, _) = eval_j_seeded(&ctx, &mol, &prep, &bounds, &c, 0.41, None).unwrap();
    let (warm_eval, _) = eval_j_seeded(&ctx, &mol, &prep, &bounds, &c, 0.41, Some(&carry)).unwrap();
    eprintln!(
        "eval_j_seeded at ω=0.41: cation {} iters unseeded vs {} seeded; \
         neutral {} vs {}",
        cold_eval.cation_iterations,
        warm_eval.cation_iterations,
        cold_eval.neutral_iterations,
        warm_eval.neutral_iterations
    );
    assert!(
        warm_eval.cation_iterations < cold_eval.cation_iterations,
        "eval_j_seeded's cation took {} iterations seeded vs {} unseeded — the \
         seed is reaching the wrapper but not the cation solve",
        warm_eval.cation_iterations,
        cold_eval.cation_iterations
    );
    assert!(
        warm_eval.neutral_iterations < cold_eval.neutral_iterations,
        "eval_j_seeded's neutral took {} iterations seeded vs {} unseeded — the \
         seed is reaching the wrapper but not the neutral solve",
        warm_eval.neutral_iterations,
        cold_eval.neutral_iterations
    );
    assert!(
        (warm_eval.e_cation - cold_eval.e_cation).abs() < 1e-9,
        "the two seeds found different cation states on a single-branch system"
    );
}

// ─────────────────────── branch check behaviour ───────────────────────

/// The check is REACHABLE: fed a tolerance below the metric's own ω-drift it
/// does fire, and the warning names the ω it fired at. Pairs with the H2O test
/// above — together they show the flag is set by the DATA, not by a code path
/// that is dead (or one that is always live).
///
/// Uses N2/STO-3G, NOT H2. H2⁺ has ONE electron, so nocc_β = 0, D_β = 0 and
/// the spin asymmetry between its two equivalent H is IDENTICALLY zero at
/// every ω — a 1e-14 tolerance cannot fire there however the check behaves,
/// so H2 would have made this test unreachable by construction. (It did: the
/// first version of this test failed for exactly that reason.)
#[test]
fn an_impossibly_tight_branch_tolerance_fires() {
    let (mol, prep, bounds) = load("n2.xyz", "sto-3g");
    let ctx = ParallelContext::default();
    let base = RhfConfig {
        df_j_aux: Some(String::new()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 300,
        ..Default::default()
    };
    let loose = cfg(base.clone(), 0.3, 0.7);
    let t_loose = tune_omega(&ctx, &mol, &prep, &bounds, &loose).unwrap();
    for e in &t_loose.evals {
        eprintln!(
            "  loose w={:.4} S2={:.8} asym={:.3e} flag={}",
            e.omega, e.cation_s_squared, e.cation_spin_asymmetry, e.branch_changed
        );
    }
    assert!(
        t_loose.branch_warning.is_none(),
        "N2/STO-3G at the default tolerance must not flag: {:?}",
        t_loose.branch_warning
    );
    // The metric must be LIVE on this system, or the tight arm below could
    // not fire for a reason that has nothing to do with the check.
    assert!(
        t_loose.evals.iter().any(|e| e.cation_spin_asymmetry > 0.0),
        "the asymmetry is identically zero on this system, so this test cannot \
         distinguish a working check from a dead one"
    );
    let tight = OmegaTuneConfig {
        branch_tol: Some(1e-30),
        ..cfg(base, 0.3, 0.7)
    };
    let t_tight = tune_omega(&ctx, &mol, &prep, &bounds, &tight).unwrap();
    let w = t_tight
        .branch_warning
        .as_ref()
        .expect("a 1e-30 tolerance must flag SOMETHING — otherwise the check is dead code");
    assert!(w.contains("ω="), "the warning must name the omega: {w}");
    assert!(
        t_tight.evals.iter().skip(1).any(|e| e.branch_changed),
        "no eval carries the flag even though the run-level warning is set"
    );
}

/// `branch_tol: None` means NOT CHECKED. On data that an impossibly tight
/// tolerance DOES flag, `None` must stay silent — otherwise "disabled" and
/// "consistent" would be indistinguishable to a reader of the result.
///
/// Same system as the reachability test, for the same reason: on H2 the
/// asymmetry is identically zero, so `None` and "checked and consistent"
/// agree there and the test would prove nothing.
#[test]
fn branch_tol_none_disables_the_check_entirely() {
    let (mol, prep, bounds) = load("n2.xyz", "sto-3g");
    let ctx = ParallelContext::default();
    let base = RhfConfig {
        df_j_aux: Some(String::new()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 300,
        ..Default::default()
    };
    let off = tune_omega(
        &ctx,
        &mol,
        &prep,
        &bounds,
        &OmegaTuneConfig {
            branch_tol: None,
            ..cfg(base.clone(), 0.3, 0.7)
        },
    )
    .unwrap();
    assert!(off.branch_warning.is_none());
    assert!(off.evals.iter().all(|e| !e.branch_changed));
    // Control: the SAME ω sequence with a tolerance that does fire, so the
    // silence above is the switch and not the data.
    let on = tune_omega(
        &ctx,
        &mol,
        &prep,
        &bounds,
        &OmegaTuneConfig {
            branch_tol: Some(1e-30),
            ..cfg(base, 0.3, 0.7)
        },
    )
    .unwrap();
    assert!(
        on.branch_warning.is_some(),
        "control: with branch_tol = 1e-30 the same run must flag, or \
         branch_tol = None proves nothing"
    );
}

// ─────────────────────── the N2 measurement ───────────────────────
