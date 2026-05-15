use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig, RhfResult};
use ferric_scf::screening::SchwarzBounds;

fn h2_sto3g_setup() -> (
    Molecule,
    PreparedBasis,
    PreparedBasis,
    Operator,
    RhfResult,
) {
    let mol = Molecule::load_xyz("../../testdata/molecules/h2.xyz").unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let dfbs_bs = basis::bundled("sto-3g").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf_cfg = RhfConfig::default();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &rhf_cfg).unwrap();
    (mol, obs, dfbs, op, rhf)
}

#[test]
fn h2_sto3g_rpa_energy_sign() {
    let (mol, obs, dfbs, op, rhf) = h2_sto3g_setup();
    let cfg = PdepRpaConfig::default();
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    assert!(result.e_rpa < 0.0,
        "RPA correlation energy should be negative, got {}", result.e_rpa);
    assert!(result.n_eigenpotentials > 0,
        "should have at least 1 eigenpotential");
    println!("H2/STO-3G RPA E_c = {:.8}", result.e_rpa);
    println!("n_eigenpotentials = {}", result.n_eigenpotentials);
    println!("quad_freqs = {:?}", result.quad_freqs);
}

#[test]
fn h2_sto3g_rpa_vs_drpa_diagnostic() {
    let (mol, obs, dfbs, op, rhf) = h2_sto3g_setup();
    let mut cfg = PdepRpaConfig::default();
    cfg.davidson_conv_thresh = 1e-8;
    let result = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    if let Some(e_diag) = result.e_rpa_dft_diag {
        let diff = (result.e_rpa - e_diag).abs();
        println!("PDEP-RPA = {:.8}, RI-dRPA = {:.8}, diff = {:.2e}", result.e_rpa, e_diag, diff);
        assert!(diff < 1e-5,
            "PDEP-RPA ({:.8}) vs RI-dRPA ({:.8}) differ by {:.2e}",
            result.e_rpa, e_diag, diff);
    }
}
