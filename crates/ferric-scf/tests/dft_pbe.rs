use ferric_core::mol::Molecule;
use ferric_core::basis;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_core::parallel::ParallelContext;
use ferric_dft::{dft, DftConfig};

#[test]
fn test_rhf_plus_pbe_dft() {
    let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    
    // 1. Converge RHF
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    
    // 2. Perform post-SCF DFT correction with PBE
    let dft_cfg = DftConfig {
        functional: "GGA_X_PBE".to_string(),
        grid_spacing: 0.2,
    };
    
    let dft_res = dft(&rhf.density, &dft_cfg).unwrap();
    
    println!("RHF Energy: {:.6}", rhf.energy);
    println!("PBE Correction: {:.6}", dft_res.total_energy);
    println!("Total DFT Energy: {:.6}", rhf.energy + dft_res.total_energy);
    
    // Verify PBE-specific dummy energy
    assert_eq!(dft_res.total_energy, -0.75);
}
