//! Unrestricted OO-RI-MP2 refuses a frozen core instead of panicking.
//!
//! The orbital-gradient assembly sizes its occupied block by the total
//! occupied count, while the response kernel returns only the active
//! orbitals. A frozen core therefore mismatched the two and panicked inside
//! `add_hf_gradient` (OH/6-31G doublet, frozen_core = 1: 5 columns against
//! 4). The CLI passed `[mp2] frozen_core` straight through, so a user input
//! could abort the process. `u_oo_ri_mp2` now returns an error.
//!
//! The test catches a panic and turns it into an assertion failure, so the
//! mutation "remove the refusal" fails on an assertion, not on a panic.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::u_oo_rimp2::{u_oo_ri_mp2, UOoRiMp2Config};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn run_oh(frozen_core: usize) -> std::thread::Result<Result<(), String>> {
    let mol = Molecule::parse_xyz("2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let uhf = solve_uhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        &bounds,
        &RhfConfig::default(),
    )
    .unwrap();
    assert!(uhf.converged);
    let config = UOoRiMp2Config {
        frozen_core,
        ..UOoRiMp2Config::default()
    };
    catch_unwind(AssertUnwindSafe(|| {
        u_oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &uhf, &config, None)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }))
}

#[test]
fn u_oo_rimp2_refuses_a_frozen_core_with_an_error() {
    let outcome = run_oh(1);
    assert!(
        outcome.is_ok(),
        "u_oo_ri_mp2 with frozen_core = 1 panicked instead of returning an error"
    );
    let err = outcome
        .unwrap()
        .expect_err("frozen_core = 1 must be refused");
    assert!(
        err.contains("frozen_core = 1"),
        "error names the setting: {err}"
    );
}

#[test]
fn u_oo_rimp2_runs_with_no_frozen_core() {
    let outcome = run_oh(0);
    assert!(outcome.is_ok(), "frozen_core = 0 panicked");
    assert!(
        outcome.unwrap().is_ok(),
        "frozen_core = 0 must not be refused"
    );
}
