//! Regression lock for the 2-call ring algebra the dRPA fixed point uses
//! (2026-08-17): `BT + TB + TBT == BT + T·(B + BT)` on a REAL alkane_4
//! eps=1e-3 ragged pattern — exact by linearity of the contraction in its
//! second operand, so the two forms may differ only by floating-point
//! summation order (bar 1e-13 on O(1e-2) entries).
//!
//! Mutation arm: `T·(B − BT)` must NOT reproduce the 3-call form — proves
//! the identity check exercises the ring product on nonzero data rather
//! than passing on structural zeros.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{
    assemble_basis, assemble_ragged_direct, build_vvhv, AmplitudeLmp2Config,
};
use ferric_mp2::ragged::ring_product;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

#[test]
fn two_call_ring_algebra_matches_three_call_form() {
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/alkane_4.xyz"
    ))
    .unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { energy_conv: 1e-10, ..Default::default() },
    )
    .unwrap();
    let vvhv = build_vvhv(&mol, &obs, &obs_bs, &rhf).unwrap();
    let lcfg = AmplitudeLmp2Config { eps: 1e-3, frozen_core: 4, ..Default::default() };
    let lb = assemble_basis(&mol, &obs, &dfbs, op, &rhf, &lcfg, &vvhv).unwrap();
    let (rg, _) = assemble_ragged_direct(&mol, &dfbs, op, &lb, 1e-3, 2.0, None, None).unwrap();
    assert!(rg.pairs.len() > 50, "pattern too small to be a real test");

    let b: Vec<Array2<f64>> = rg.pairs.iter().map(|pb| pb.j_blk.clone()).collect();
    // an MP1-like amplitude as T so X != Y and both operand slots carry
    // structurally different data
    let t: Vec<Array2<f64>> = rg.pairs.iter().map(|pb| -&pb.j_blk / &pb.denom).collect();

    // 3-call form
    let bt = ring_product(&rg, &b, &t);
    let tb = ring_product(&rg, &t, &b);
    let tbt = ring_product(&rg, &t, &bt);
    // 2-call form
    let u: Vec<Array2<f64>> = b.iter().zip(&bt).map(|(bb, c)| bb + c).collect();
    let tu = ring_product(&rg, &t, &u);

    let max_abs = |blocks: &[Array2<f64>]| -> f64 {
        blocks.iter().flat_map(|m| m.iter()).fold(0.0f64, |a, &v| a.max(v.abs()))
    };
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for p in 0..rg.pairs.len() {
        let three = &(&bt[p] + &tb[p]) + &tbt[p];
        let two = &bt[p] + &tu[p];
        worst = worst.max((&three - &two).mapv(f64::abs).iter().cloned().fold(0.0, f64::max));
        scale = scale.max(three.mapv(f64::abs).iter().cloned().fold(0.0, f64::max));
    }
    eprintln!("2-call vs 3-call: max|diff| = {worst:.3e} (scale {scale:.3e}, tb magnitude {:.3e})", max_abs(&tb));
    assert!(scale > 1e-4, "identity tested on numerically trivial data");
    assert!(worst < 1e-13, "2-call ring algebra deviates: {worst:.3e}");

    // MUTATION ARM: wrong combination must fail loudly at the same bar
    let u_bad: Vec<Array2<f64>> = b.iter().zip(&bt).map(|(bb, c)| bb - c).collect();
    let tu_bad = ring_product(&rg, &t, &u_bad);
    let mut worst_bad = 0.0f64;
    for p in 0..rg.pairs.len() {
        let three = &(&bt[p] + &tb[p]) + &tbt[p];
        let two_bad = &bt[p] + &tu_bad[p];
        worst_bad = worst_bad
            .max((&three - &two_bad).mapv(f64::abs).iter().cloned().fold(0.0, f64::max));
    }
    eprintln!("mutation arm (T*(B-BT)): max|diff| = {worst_bad:.3e}");
    assert!(worst_bad > 1e-6, "mutation arm unreachable — identity test is vacuous");
}
