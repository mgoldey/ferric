//! Weekly validation half of `pbc_rohf.rs` (every test
//! `#[ignore = "validation: ..."]`): the tests whose per-commit CI time
//! was >= 120 s (mean of runs 40957025137 and 40953725119), moved so the
//! fast tier stays balanced over 8 shards. The module docs, hypotheses and
//! bars are in `pbc_rohf.rs`; the shared helpers are `pbc_rohf_support/mod.rs`.
//!
//! Fast-tier canaries that keep these invariants per commit:
//! * `second_order_reaches_the_diis_state_on_h3_roks_pbe0_none` (the same
//!   anchor on the H3 doublet, coarse grid)
//! * `roks_with_an_hf_equivalent_builder_is_gamma_rohf`,
//!   `second_order_converges_tri_rohf_to_pyscf`

mod common;
#[path = "pbc_rohf_support/mod.rs"]
mod support;

use support::*;

/// EXACTNESS ANCHOR (tri 4H s+p triplet ROKS PBE0, `exxdiv = none`, the
/// case whose minimum repels the Roothaan map): DIIS with the hybrid shift
/// and the second-order solver (FD XC response) reach the same state, the
/// prototype pin, and a true minimum (prototype λ_min +0.0693).
#[test]
#[ignore = "validation: PBC Gamma ROKS second-order solver, tri 4H s+p PBE0 none vs DIIS + pin; CI ~243 s"]
fn second_order_reaches_the_diis_state_on_tri_roks_pbe0_none() {
    let su = setup(cell_mult(&TRI_ATOMS, TRI_A, 3), sp_basis_h());
    let gcfg = ssf_grid(75, 302, 10.0);
    let grid = PeriodicGrid::build(&su.cell, &gcfg).expect("grid");
    assert_eq!(grid.len(), TRI_SSF_NPTS);
    let mut pxc = PeriodicXc::new(
        &su.cell,
        su.prep.basis_set(),
        "PBE0",
        &grid,
        &PeriodicXcConfig::default(),
    )
    .expect("PeriodicXc PBE0");
    let base = GammaRoksConfig {
        exxdiv: ExxDiv::None,
        ..GammaRoksConfig::new("PBE0")
    };
    let d = gamma_roks_with_xc(&su.cell, &su.prep, &su.hc, ints(&su), &mut pxc, &base)
        .expect("DIIS PBE0 none");
    let so_cfg = GammaRoksConfig {
        solver: second_order(true),
        ..base.clone()
    };
    let s = gamma_roks_with_xc(&su.cell, &su.prep, &su.hc, ints(&su), &mut pxc, &so_cfg)
        .expect("second-order PBE0 none");
    let info = s.second_order.as_ref().expect("second-order diagnostics");
    eprintln!("  tri PBE0 none second order: {info:?}");
    assert!(info.converged);
    close(
        s.scf.energy,
        d.scf.energy,
        1e-10,
        "tri PBE0 none SO vs DIIS",
    );
    close(
        s.scf.energy,
        TRI_ROKS_PBE0_NONE,
        1e-8,
        "tri PBE0 none SO vs pin",
    );
    same_state(&s.scf, &d.scf, "tri PBE0 none");
    let lam = info.lowest_hessian_eigenvalue.expect("lambda_min reported");
    assert!(
        lam > 0.0,
        "tri PBE0 none: not a minimum, lambda_min {lam:e}"
    );
    close(s.s2, 2.0, 1e-12, "tri PBE0 <S2>");
}
