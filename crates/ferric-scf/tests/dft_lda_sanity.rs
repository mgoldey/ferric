use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

#[test]
fn lda_h2_converges_and_is_lower_than_hf() {
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();

    let rhf_only = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    eprintln!("HF energy:  {:.6}", rhf_only.energy);

    let lda_cfg = RhfConfig {
        xc: Some("LDA".to_string()),
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        ..Default::default()
    };
    let ks = solve_rhf(&ctx, &mol, &obs, op, &bounds, &lda_cfg).unwrap();
    eprintln!("LDA energy: {:.6}", ks.energy);
    eprintln!("converged: {}, iters: {}", ks.converged, ks.iterations);

    // LDA is variationally lower than HF (it includes correlation).
    // For H2/cc-pVDZ: HF ≈ -1.129, LDA ≈ -1.137.
    assert!(ks.converged, "LDA SCF did not converge");
    assert!(ks.energy < rhf_only.energy, "LDA E should be < HF E");
    assert!(ks.energy > -1.50 && ks.energy < -1.00, "H2 LDA energy out of range: {}", ks.energy);
}
