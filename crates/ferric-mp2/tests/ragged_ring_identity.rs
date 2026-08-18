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
use ferric_mp2::ragged::{ring_product, ring_product_planned, RingPlan};
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

/// Deterministic LCG (fixed seed byte pattern, no ambient randomness) —
/// used only to fill T with "random-ish" nonzero values so the oracle
/// below exercises real GEMM data, not structural zeros.
fn lcg_fill(shapes: &[(usize, usize)], seed: u64) -> Vec<Array2<f64>> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        // splitmix64
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // map to roughly [-1, 1]
        (z as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    shapes
        .iter()
        .map(|&(na, nb)| Array2::<f64>::from_shape_fn((na, nb), |_| next()))
        .collect()
}

/// Oracle: `ring_product_planned(RingPlan::new(rg, b), t)` must agree with
/// unmodified `ring_product(rg, b, t)` on a real alkane_4 pattern — the
/// plan hoists B's sub-block gathers and the intersection bookkeeping out
/// of the per-call path, but must compute the SAME sum. T is filled with
/// a deterministic pseudo-random pattern (fixed seed, not ambient
/// randomness) so every entry the plan gathers is exercised.
///
/// Mutation arm: corrupt one cached panel entry in the plan and confirm
/// the planned result diverges from the oracle — proves the test would
/// actually catch a broken cache, not just a vacuously-agreeing one.
#[test]
fn ring_product_planned_matches_ring_product_oracle() {
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
    let shapes: Vec<(usize, usize)> =
        rg.pairs.iter().map(|pb| (pb.da.len(), pb.db.len())).collect();
    let t = lcg_fill(&shapes, 0xC0FF_EE12_3456_789A);

    let oracle = ring_product(&rg, &b, &t);
    let plan = RingPlan::new(&rg, &b);
    let planned = ring_product_planned(&plan, &t);

    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for p in 0..rg.pairs.len() {
        worst = worst.max((&oracle[p] - &planned[p]).mapv(f64::abs).iter().cloned().fold(0.0, f64::max));
        scale = scale.max(oracle[p].mapv(f64::abs).iter().cloned().fold(0.0, f64::max));
    }
    eprintln!("planned vs oracle: max|diff| = {worst:.3e} (scale {scale:.3e})");
    assert!(scale > 1e-4, "identity tested on numerically trivial data");
    assert!(worst < 1e-13, "ring_product_planned deviates from the oracle: {worst:.3e}");

    // MUTATION ARM: corrupt one cached panel entry in the plan (simulated
    // by rebuilding the plan against a B with one entry perturbed) and
    // confirm the result no longer matches the true oracle.
    let mut b_corrupt = b.clone();
    // find a nonempty block to corrupt
    let corrupt_p = b_corrupt
        .iter()
        .position(|blk| blk.iter().any(|&v| v != 0.0))
        .expect("no nonzero B block found — pattern is vacuous");
    b_corrupt[corrupt_p][(0, 0)] += 1.0; // large corruption relative to typical |B| entries
    let bad_plan = RingPlan::new(&rg, &b_corrupt);
    let bad_planned = ring_product_planned(&bad_plan, &t);
    let mut worst_bad = 0.0f64;
    for p in 0..rg.pairs.len() {
        worst_bad = worst_bad
            .max((&oracle[p] - &bad_planned[p]).mapv(f64::abs).iter().cloned().fold(0.0, f64::max));
    }
    eprintln!("mutation arm (corrupted B panel): max|diff| vs oracle = {worst_bad:.3e}");
    assert!(worst_bad > 1e-6, "mutation arm unreachable — corrupted-panel cache went undetected");
}
