//! **Live-SCF** validation of NWChem-style angular grid pruning.
//!
//! # Why this file exists, given `grid_prune.rs` already passes
//!
//! `grid_prune.rs` measures pruning's accuracy by re-evaluating `E_xc` on a
//! density that was converged on the FLAT grid. That isolates the grid's
//! contribution cleanly, and it is a useful measurement — but it is **not**
//! the measurement that licenses turning pruning on. In a real KS SCF the
//! pruned grid enters the Fock matrix at *every* iteration, so the SCF
//! converges to a slightly different density, and the per-iteration errors can
//! compound (or cancel) in a way a single fixed-density re-evaluation cannot
//! see. Under this repo's Experimental Protocol those are two different
//! experiments, and only the live one gates the change.
//!
//! So: this file runs the **same molecules, bases, functionals and PySCF
//! reference files** as the `ferric-scf` DFT reference suites (`dft_lda.rs`,
//! `dft_pbe.rs`, `dft_b3lyp.rs`), through a **full pruned-grid SCF**, and
//! asserts against **those suites' existing tolerances** — not relaxed ones.
//! A pruned run that needed a wider tolerance than the flat run would be a
//! failed validation, not a reason to widen the bar.
//!
//! The reference JSONs record `main_grid: [75, 110]`, i.e. PySCF itself was
//! run on the flat grid. That is deliberate and is what makes this a real
//! test: it asks whether ferric-with-pruning still reproduces an
//! independently-computed flat-grid answer to the same accuracy that
//! ferric-without-pruning does.
//!
//! # Artifact hypothesis (stated before the numbers were taken)
//!
//! * If pruning is sound, the pruned-SCF error vs PySCF should be
//!   *indistinguishable* from the flat-SCF error vs PySCF — both dominated by
//!   ferric's own pre-existing grid/SCF floor, and their difference far below
//!   the tolerance.
//! * If pruning is broken in a way the fixed-density probe missed (per-shell
//!   weight bug amplified through self-consistency, or a region boundary that
//!   matters only once the density can relax), the pruned error would exceed
//!   the flat error by an amount that GROWS with the tolerance-relevant scale
//!   — and, importantly, would not be a constant offset.
//!
//! These predict different observations, so the experiment can distinguish
//! them. `pruned_scf_matches_pyscf_within_the_existing_reference_tolerance`
//! records both errors side by side for exactly that reason.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::prune::PruneScheme;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ladder::{default_ladder_from, solve_rhf_ladder};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Ref {
    e_total: f64,
    #[serde(default)]
    converged: bool,
}

fn ref_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference")
        .join(name)
}

const H2: &str = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
const H2O: &str = "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n";
const CH4: &str = "5\nCH4\nC 0 0 0\n\
                   H 0.6276 0.6276 0.6276\n\
                   H -0.6276 -0.6276 0.6276\n\
                   H -0.6276 0.6276 -0.6276\n\
                   H 0.6276 -0.6276 -0.6276\n";
const NH3: &str = "4\nNH3\n\
    N 0.000000 0.000000 0.116489\n\
    H 0.000000 0.939731 -0.271808\n\
    H 0.813831 -0.469865 -0.271808\n\
    H -0.813831 -0.469865 -0.271808\n";

/// Tolerances copied VERBATIM from the `ferric-scf` reference suites this
/// file mirrors. They are the bar a pruned SCF has to clear; nothing here may
/// widen them.
///
/// * `dft_lda.rs`   — `const TOL: f64 = 1e-5`
/// * `dft_pbe.rs`   — `const TOL: f64 = 2e-5`
/// * `dft_b3lyp.rs` — `const TOL: f64 = 5e-5`
///
/// LDA's is the TIGHTEST of the three, so it is the case most able to detect a
/// pruning-induced shift — which is exactly why it must not be loosened to
/// match the others.
fn tolerance_for(xc: &str) -> f64 {
    match xc {
        "LDA" => 1e-5,
        "PBE" => 2e-5,
        "B3LYP" => 5e-5,
        other => panic!("no reference tolerance recorded for functional {other}"),
    }
}

/// Run one KS-DFT SCF to convergence with the given pruning setting and return
/// the total energy.
///
/// The SCF setup (ladder, RI-J aux, convergence thresholds) is copied from
/// `dft_pbe.rs` / `dft_lda.rs` so that the ONLY difference between the flat
/// and pruned runs compared here is `AtomicGridConfig::prune`.
fn run_scf(xyz: &str, basis_name: &str, xc: &str, prune: Option<PruneScheme>) -> f64 {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        // B3LYP is a hybrid and `dft_b3lyp.rs` fits K as well as J. Matching
        // that matters: an unfitted K would change the energy by far more than
        // pruning does and would swamp the effect under test.
        df_k_aux: if xc == "B3LYP" {
            Some("def2-universal-jkfit".into())
        } else {
            None
        },
        energy_conv: 1e-10,
        // dft_lda.rs uses 1e-7 here; PBE/B3LYP use 1e-8. Take the tighter of
        // the two everywhere so the SCF floor cannot be mistaken for a
        // pruning effect.
        density_conv: 1e-8,
        max_iter: 800,
        // The single variable under test. 75x110 is `AtomicGridConfig::
        // default()`'s shape and the shape the PySCF references were
        // generated at.
        dft_grid: Some(AtomicGridConfig { n_radial: 75, n_angular: 110, prune }),
        ..Default::default()
    };
    // The ladder, as dft_pbe.rs / dft_b3lyp.rs use. dft_lda.rs uses plain
    // solve_rhf, but the ladder's rung 0 IS that plain solve, so running the
    // ladder for every functional is a superset — it cannot make a case pass
    // that plain DIIS would fail on the flat grid, because the flat arm is run
    // through the identical path and both are compared to the same reference.
    let ladder = default_ladder_from(&cfg);
    let lr = solve_rhf_ladder(&ctx, &mol, &obs, op, &bounds, &ladder).unwrap();
    lr.result.energy
}

struct Case {
    label: &'static str,
    xyz: &'static str,
    basis: &'static str,
    xc: &'static str,
    reference: &'static str,
}

/// The closed-shell reference matrix, mirroring the cases in `dft_lda.rs`,
/// `dft_pbe.rs` and `dft_b3lyp.rs` that have a bundled reference JSON.
const CASES: &[Case] = &[
    Case { label: "H2",  xyz: H2,  basis: "cc-pvdz", xc: "LDA",   reference: "h2_cc-pvdz_lda.json" },
    Case { label: "H2O", xyz: H2O, basis: "cc-pvdz", xc: "LDA",   reference: "h2o_cc-pvdz_lda.json" },
    Case { label: "CH4", xyz: CH4, basis: "cc-pvdz", xc: "LDA",   reference: "methane_cc-pvdz_lda.json" },
    Case { label: "H2",  xyz: H2,  basis: "cc-pvdz", xc: "PBE",   reference: "h2_cc-pvdz_pbe.json" },
    Case { label: "H2O", xyz: H2O, basis: "cc-pvdz", xc: "PBE",   reference: "h2o_cc-pvdz_pbe.json" },
    Case { label: "CH4", xyz: CH4, basis: "cc-pvdz", xc: "PBE",   reference: "methane_cc-pvdz_pbe.json" },
    Case { label: "H2",  xyz: H2,  basis: "cc-pvdz", xc: "B3LYP", reference: "h2_cc-pvdz_b3lyp.json" },
    Case { label: "H2O", xyz: H2O, basis: "cc-pvdz", xc: "B3LYP", reference: "h2o_cc-pvdz_b3lyp.json" },
    Case { label: "CH4", xyz: CH4, basis: "cc-pvdz", xc: "B3LYP", reference: "methane_cc-pvdz_b3lyp.json" },
    // NH3 — the 4th molecule every reference suite carries. Worth including
    // because it is the case `dft_pbe.rs` documents as landing in a WRONG SCF
    // solution under plain DIIS (hence the ladder), i.e. the case most
    // sensitive to the Fock matrix being perturbed — which is exactly what
    // pruning does, every iteration.
    Case { label: "NH3", xyz: NH3, basis: "cc-pvdz", xc: "LDA",   reference: "nh3_cc-pvdz_lda.json" },
    Case { label: "NH3", xyz: NH3, basis: "cc-pvdz", xc: "PBE",   reference: "nh3_cc-pvdz_pbe.json" },
    Case { label: "NH3", xyz: NH3, basis: "cc-pvdz", xc: "B3LYP", reference: "nh3_cc-pvdz_b3lyp.json" },
    // Second basis, mirroring the def2-SVP arms of the same suites. def2-SVP
    // is the basis whose contractions are renormalised at load, so it
    // exercises a different AO path than cc-pVDZ.
    Case { label: "H2O", xyz: H2O, basis: "def2-svp", xc: "LDA",   reference: "h2o_def2-svp_lda.json" },
    Case { label: "H2O", xyz: H2O, basis: "def2-svp", xc: "PBE",   reference: "h2o_def2-svp_pbe.json" },
    Case { label: "H2O", xyz: H2O, basis: "def2-svp", xc: "B3LYP", reference: "h2o_def2-svp_b3lyp.json" },
    Case { label: "CH4", xyz: CH4, basis: "def2-svp", xc: "PBE",   reference: "methane_def2-svp_pbe.json" },
];

/// THE gate. A full pruned-grid SCF must reproduce the PySCF reference to the
/// same tolerance the flat-grid reference suites are held to.
///
/// Reports the flat error alongside the pruned error for every case, so a
/// failure says whether pruning caused it or whether ferric was already at
/// that error on the flat grid.
#[test]
fn pruned_scf_matches_pyscf_within_the_existing_reference_tolerance() {
    let mut failures: Vec<String> = Vec::new();
    eprintln!(
        "{:<22} {:>14} {:>12} {:>12} {:>10} {:>6}",
        "case", "E_pruned (Ha)", "err_pruned", "err_flat", "tol", "pass"
    );
    for c in CASES {
        let r: Ref =
            serde_json::from_str(&std::fs::read_to_string(ref_path(c.reference)).unwrap())
                .unwrap();
        assert!(r.converged, "PySCF reference {} not converged", c.reference);

        let e_pruned = run_scf(c.xyz, c.basis, c.xc, Some(PruneScheme::NwchemLike));
        let e_flat = run_scf(c.xyz, c.basis, c.xc, None);
        let err_pruned = (e_pruned - r.e_total).abs();
        let err_flat = (e_flat - r.e_total).abs();
        let tol = tolerance_for(c.xc);
        let pass = err_pruned < tol;

        let name = format!("{}/{}/{}", c.label, c.basis, c.xc);
        eprintln!(
            "{name:<22} {e_pruned:>14.9} {err_pruned:>12.2e} {err_flat:>12.2e} \
             {tol:>10.0e} {:>6}",
            if pass { "yes" } else { "NO" }
        );
        if !pass {
            failures.push(format!(
                "{name}: pruned-SCF err {err_pruned:.3e} Ha exceeds the reference \
                 tolerance {tol:.0e} Ha (flat-SCF err on the same case: {err_flat:.3e} Ha)"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "pruned live SCF broke {} reference case(s):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// EXACTNESS ANCHOR (Experimental Protocol: write it before measuring).
///
/// Pruning's trivial limit is `prune = None`, which must leave the SCF
/// *bit-identical*, not merely close — `build_atomic_grid_pruned(.., None)`
/// delegates to `build_atomic_grid`, so any drift here means the plumbing
/// changed something it should not have (e.g. a grid rebuilt in a different
/// order, or the config no longer reaching the constructor).
///
/// This is the assert that catches a whole class of wiring error before any
/// accuracy sweep is worth running. It is kept SCF-free on purpose: the
/// property is a property of the grid, and asserting it on the grid directly
/// is both stronger (bit-identical points, not just a bit-identical scalar)
/// and far cheaper than converging two SCFs to compare one number.
#[test]
fn prune_none_reproduces_the_flat_grid_bit_for_bit() {
    // The grid helper's own trivial limit, at the level where a wiring change
    // would show up: `build_atomic_grid_pruned(.., None)` must be bit-identical
    // to `build_atomic_grid`, point for point and weight for weight.
    use ferric_dft::grid::{build_atomic_grid, build_atomic_grid_pruned};
    let mol = Molecule::parse_xyz(H2O, 0, 1).unwrap();
    let cfg = AtomicGridConfig { n_radial: 75, n_angular: 110, prune: None };
    let flat = build_atomic_grid(&mol, &cfg);
    let none = build_atomic_grid_pruned(&mol, &cfg, None).unwrap();
    assert_eq!(flat.len(), none.len());
    for (i, (a, b)) in flat.iter().zip(none.iter()).enumerate() {
        assert_eq!(a.weight.to_bits(), b.weight.to_bits(), "weight at point {i}");
        for k in 0..3 {
            assert_eq!(a.xyz[k].to_bits(), b.xyz[k].to_bits(), "xyz[{k}] at point {i}");
        }
    }
}

/// The pruned SCF must actually be running on a smaller grid — otherwise the
/// accuracy test above would pass vacuously (it would be comparing the flat
/// grid to itself).
///
/// Checks the pass condition is REACHABLE and that the measurement is not
/// arithmetic: asserts the point count really drops, at the production grid,
/// on the molecules the accuracy table uses.
#[test]
fn the_pruned_scf_really_uses_fewer_points() {
    use ferric_dft::grid::{build_atomic_grid, build_atomic_grid_pruned};

    for (label, xyz) in [("H2", H2), ("H2O", H2O), ("CH4", CH4)] {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let cfg = AtomicGridConfig { n_radial: 75, n_angular: 110, prune: None };
        let flat = build_atomic_grid(&mol, &cfg);
        let pruned =
            build_atomic_grid_pruned(&mol, &cfg, Some(PruneScheme::NwchemLike)).unwrap();
        let saved = 100.0 * (1.0 - pruned.len() as f64 / flat.len() as f64);
        eprintln!(
            "[{label}] 75x110: flat {} -> pruned {} ({saved:.1}% fewer)",
            flat.len(),
            pruned.len()
        );
        assert!(
            saved > 15.0,
            "{label}: pruning removed only {saved:.1}% of points; the accuracy \
             test above would be near-vacuous"
        );
    }
}

/// THE VV10 / `n_angular = 50` HAZARD.
///
/// The NLC (nonlocal-correlation) grid used by VV10 functionals such as
/// wB97X-V is 50x50, and pruning has no valid table at angular order 50. This
/// pins the two halves of how that is handled:
///
/// 1. A 50x50 config with pruning ON is a hard ERROR, never a silent flat
///    grid (config honesty).
/// 2. The NLC grid's `prune` is INDEPENDENT of the main grid's, so a wB97X-V
///    run with the main grid pruned still builds its NLC grid flat and works.
///
/// Point 2 is the one that would actually break in production, and it is why
/// `AtomicGridConfig::default()` must keep `prune: None` — flipping that
/// default would enable pruning on every 50x50 NLC grid in the workspace at
/// once (they are all built with `..Default::default()`), breaking VV10
/// everywhere.
#[test]
fn vv10_nlc_grid_survives_a_pruned_main_grid() {
    use ferric_dft::grid::build_atomic_grid_pruned;

    let mol = Molecule::parse_xyz(H2O, 0, 1).unwrap();

    // (1) 50x50 + pruning must error, not silently un-prune.
    let nlc_pruned = AtomicGridConfig { n_radial: 50, n_angular: 50, prune: None };
    assert!(
        build_atomic_grid_pruned(&mol, &nlc_pruned, Some(PruneScheme::NwchemLike)).is_err(),
        "pruning at n_angular = 50 must be a hard error"
    );

    // (2) The default NLC config carries prune = None, so it builds fine even
    // while the main grid is pruned. This is the invariant the KsXc wiring
    // relies on.
    let nlc_default =
        AtomicGridConfig { n_radial: 50, n_angular: 50, ..Default::default() };
    assert!(
        nlc_default.prune.is_none(),
        "AtomicGridConfig::default() must keep prune = None or every 50x50 NLC \
         grid in the workspace breaks"
    );
    assert!(build_atomic_grid_pruned(&mol, &nlc_default, nlc_default.prune).is_ok());

    // And the live proof: a full VV10 (wB97X-V) SCF with the MAIN grid pruned
    // must still converge. This is the end-to-end version of (2) -- if the
    // main grid's prune setting ever leaked into the NLC grid, this errors.
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: Some("wB97X-V".into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-8,
        density_conv: 1e-6,
        dft_grid: Some(AtomicGridConfig {
            n_radial: 75,
            n_angular: 110,
            prune: Some(PruneScheme::NwchemLike),
        }),
        nlc_grid: Some(AtomicGridConfig { n_radial: 50, n_angular: 50, prune: None }),
        ..Default::default()
    };
    let ladder = default_ladder_from(&cfg);
    let lr = solve_rhf_ladder(&ctx, &mol, &obs, op, &bounds, &ladder)
        .expect("pruned-main-grid wB97X-V SCF must build its NLC grid unpruned and run");
    eprintln!("[H2O/cc-pVDZ/wB97X-V] pruned main grid + flat NLC: E = {:.9} Ha", lr.result.energy);
    assert!(lr.result.energy.is_finite());
}

/// The GRADIENT restriction, pinned.
///
/// `build_atomic_grid_with_response` builds the Becke weight-derivative term
/// on the unpruned grid only. Silently ignoring `prune` there would return a
/// vector that is not the gradient of the pruned energy the SCF converged —
/// an inconsistency visible only under finite difference. It must be a hard
/// error instead.
#[test]
fn gradient_grid_rejects_a_pruned_config_instead_of_ignoring_it() {
    use ferric_dft::grid::build_atomic_grid_with_response;

    let mol = Molecule::parse_xyz(H2, 0, 1).unwrap();
    let pruned = AtomicGridConfig {
        n_radial: 75,
        n_angular: 110,
        prune: Some(PruneScheme::NwchemLike),
    };
    let err = build_atomic_grid_with_response(&mol, &pruned)
        .expect_err("a pruned config must be rejected on the grid-response path");
    let msg = format!("{err}");
    assert!(
        msg.contains("prun"),
        "the rejection must say pruning is the reason; got: {msg}"
    );

    // ...and the unpruned config still works, so this is a targeted rejection
    // rather than a broken path.
    let flat = AtomicGridConfig { n_radial: 75, n_angular: 110, prune: None };
    assert!(build_atomic_grid_with_response(&mol, &flat).is_ok());
}
