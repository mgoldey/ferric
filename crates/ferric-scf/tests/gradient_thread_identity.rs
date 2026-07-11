//! P1 regression guard: the KS (B3LYP) nuclear gradient must be bit-identical
//! across rayon thread counts. The 2e derivative quartet loop is fanned out
//! over a flat screened quartet list with a deterministic grouped reduction
//! (see gradient.rs::par_twoelectron_gradient); the XC and 1e pieces must be
//! equally thread-count-independent for the total to match to the bit.
//!
//! Basis restriction: AO Hessians (needed by the GGA XC gradient) are only
//! implemented for s/p shells, so this uses 6-31G (no d functions). Water in
//! 6-31G has ~1000 screened quartets — above the 2e serial-fallback threshold,
//! so the parallel grouped path is exercised in both pools.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

#[test]
fn b3lyp_gradient_bit_identical_across_thread_counts() {
    let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some("b3lyp".into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };
    // One SCF solve outside the pools; the same ScfResult feeds both gradient
    // evaluations so any bit difference is the gradient pipeline's own.
    let result = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();

    let run = |threads: usize| -> Array2<f64> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| ks_gradient_closed(&mol, &prep, &bs, op, &bounds, "b3lyp", &result, None).unwrap())
    };
    let g1 = run(1);
    let g4 = run(4);
    for (i, (v1, v4)) in g1.iter().zip(g4.iter()).enumerate() {
        assert_eq!(
            v1.to_bits(),
            v4.to_bits(),
            "B3LYP gradient element {i} differs across thread counts: {v1:e} vs {v4:e}"
        );
    }
}
