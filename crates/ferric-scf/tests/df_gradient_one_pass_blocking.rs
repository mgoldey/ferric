//! The DF-JK gradient's one-pass `(P|μν)` contraction does not depend on how
//! the aux range is blocked.
//!
//! `df_jk_gradient` computes the raw three-centre tensor one aux-shell block
//! at a time and never stores it (it used to hold the whole tensor in core:
//! 7.6 GB for danuglipron/def2-SVP). Every aux row is contracted on its own,
//! so a budget that forces one shell per block must give the SAME BITS as one
//! block for everything. A block boundary that dropped or repeated a row, or
//! split a shell, changes `d_P` / `L^s` and fails here.

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_gradient::{df_jk_gradient, DfKChannel};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

#[test]
fn one_shell_per_block_is_bit_identical_to_one_block() {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
    let bs = ferric_core::basis::bundled("6-31g").expect("6-31g");
    let obs = PreparedBasis::new(&mol, &bs).expect("obs");
    let aux = ferric_core::basis::bundled("def2-universal-jkfit").expect("jkfit");
    let dfbs = PreparedBasis::new(&mol, &aux).expect("dfbs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let scf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .expect("RHF converges");
    let d = scf.density_r();
    let natoms = mol.atoms.len();
    let ch = [DfKChannel {
        density: d,
        alpha: -0.25,
    }];

    // Budget 1 byte -> one aux row per block -> one shell per block.
    let whole = df_jk_gradient(natoms, &obs, &dfbs, op, Some(d), &ch, usize::MAX).unwrap();
    let shells = df_jk_gradient(natoms, &obs, &dfbs, op, Some(d), &ch, 1).unwrap();
    assert!(
        whole.iter().any(|v| v.abs() > 1e-3),
        "vacuous: the gradient is ~zero"
    );
    for (a, b) in whole.iter().zip(shells.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "blocked {b:e} vs whole {a:e}");
    }
}
