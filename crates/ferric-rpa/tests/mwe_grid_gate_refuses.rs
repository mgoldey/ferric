//! MWE: does the grid path REFUSE a job that cannot fit its budget?
//!
//! The end of the chain the other MWEs build:
//!
//! 1. `mwe_budget_respected.rs`   — the band width must honor the byte cap
//! 2. `mwe_panel_width_budget.rs` — the panel width must honor it too
//! 3. `mwe_explicit_budget_reaches_mp2.rs` — the user's budget must arrive
//! 4. `mwe_estimator_sees_grid.rs` — the estimate must model the grid
//! 5. **this file** — and a job over the ceiling must be refused, not attempted
//!
//! Steps 1-4 are all inert without step 5: an accurate estimate nobody consults
//! prevents nothing. Before this work `properties.rs` and `dispersion.rs` had
//! ZERO `check_alloc`/`estimate_peak_bytes` call sites, while the CLI's
//! `compute_alpha_atomic` defaults to `true` — so a stock run walked into an
//! ungoverned path and, three times on 2026-07-13, took the box down with it.
//!
//! Run with a tiny explicit budget so the refusal is deterministic and the test
//! never actually allocates anything large. Nothing here is scale-dependent.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::pdep_polarizability_becke;
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Water/STO-3G: the smallest system that still exercises the full
/// grid + per-atom-dipole path. Deliberately tiny — the contract under test is
/// structural (is the gate consulted at all?), not scale-dependent, and this
/// box is shared.
type WaterSetup = (
    Molecule,
    PreparedBasis,
    basis::BasisSet,
    PreparedBasis,
    ferric_scf::result::ScfResult,
);

fn water_scf() -> WaterSetup {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).expect("parse water");
    let obs_bs = basis::bundled("sto-3g").expect("sto-3g");
    let dfbs_bs = basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("prep obs");
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).expect("prep dfbs");

    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("rhf");
    (mol, obs, obs_bs, dfbs, rhf)
}

/// CONTRACT 1: a starvation budget must produce a clean error, not an attempt.
///
/// One byte cannot hold `chi`, so the only correct behavior is to refuse before
/// allocating. An `Ok(...)` here means the gate is absent or bypassed — which is
/// precisely the pre-fix state.
#[test]
fn grid_path_refuses_a_starvation_budget() {
    let (mol, obs, obs_bs, dfbs, rhf) = water_scf();
    let cfg = PdepRpaConfig { memory_budget_bytes: Some(1), ..PdepRpaConfig::default() };

    let got = pdep_polarizability_becke(
        &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::coulomb(), &cfg,
    );

    let err = got.err().expect(
        "a 1-byte budget must be REFUSED before allocating chi; Ok means the \
         grid path has no pre-flight gate",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("requires") && msg.contains("budget is"),
        "the refusal must name the estimate and the budget so the user can act \
         on it; got: {msg}"
    );
}

/// CONTRACT 2: the refusal must identify the offending path and its shape.
///
/// A bare "out of memory" teaches nothing. The message must say which method
/// refused and at what shape, so the user can decide whether to raise the
/// budget or shrink the system — the two remedies `check_alloc` names.
#[test]
fn refusal_message_identifies_path_and_shape() {
    let (mol, obs, obs_bs, dfbs, rhf) = water_scf();
    let cfg = PdepRpaConfig { memory_budget_bytes: Some(1), ..PdepRpaConfig::default() };

    let msg = pdep_polarizability_becke(
        &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::coulomb(), &cfg,
    )
    .err()
    .expect("must refuse")
    .to_string();

    for needle in ["pdep_polarizability_becke", "natoms=", "npts=", "nbf="] {
        assert!(
            msg.contains(needle),
            "refusal message must contain {needle:?} to be actionable; got: {msg}"
        );
    }
}

/// CONTRACT 3: an ample budget must still run to completion.
///
/// The complement — the gate must not become a wall for jobs that genuinely
/// fit. Water/STO-3G needs only a few MB, so a 4 GiB budget must sail through
/// and return one 3x3 tensor per atom. Guards against overcorrecting contracts
/// 1-2 into a permanently-closed door.
#[test]
fn grid_path_runs_under_an_ample_budget() {
    let (mol, obs, obs_bs, dfbs, rhf) = water_scf();
    let cfg = PdepRpaConfig {
        memory_budget_bytes: Some(4 * 1024 * 1024 * 1024),
        ..PdepRpaConfig::default()
    };

    let alpha = pdep_polarizability_becke(
        &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::coulomb(), &cfg,
    )
    .expect("water/STO-3G must fit a 4 GiB budget comfortably");

    assert_eq!(alpha.len(), mol.atoms.len(), "one 3x3 tensor per atom");
    // Sanity: a polarizability must be finite and its trace positive.
    for (a, t) in alpha.iter().enumerate() {
        let trace = t[0][0] + t[1][1] + t[2][2];
        assert!(trace.is_finite(), "atom {a}: non-finite alpha trace");
        assert!(trace > 0.0, "atom {a}: non-positive alpha trace {trace}");
    }
}
