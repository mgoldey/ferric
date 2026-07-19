use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::ao_grid::eval_basis_and_grad_on_points;
use ferric_dft::density_on_grid::eval_density_closed;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_dft::libxc::xc_def_from_name;
use ferric_dft::vxc::semilocal_vxc_closed;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn build_h2o() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n", 0, 1,
    ).unwrap()
}

#[test]
fn vxc_lda_h2o_is_hermitian_and_energy_is_negative() {
    let mol = build_h2o();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let grid = build_atomic_grid(&mol, &AtomicGridConfig::default());
    let pts: Vec<[f64;3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();
    let dens = eval_density_closed(&rhf.density_total, &chi, &dchi);

    let xc = xc_def_from_name("LDA").unwrap();
    let (e_xc, vxc) = semilocal_vxc_closed(&grid, &chi, &dchi, &dens, None, &xc);

    // V_xc must be Hermitian after symmetrization
    let n = vxc.nrows();
    let mut asym: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let a = (vxc[(i, j)] - vxc[(j, i)]).abs();
            if a > asym { asym = a; }
        }
    }
    assert!(asym < 1e-12, "V_xc not Hermitian: max asym = {asym:.2e}");

    // LDA E_xc for H2O / cc-pVDZ on (75,110) grid should be ~ -8.85 Ha
    // (compare against PySCF; loose bracket for sanity check only)
    eprintln!("LDA E_xc(H2O, cc-pVDZ) = {e_xc:.6} Ha");
    assert!(e_xc < 0.0,        "E_xc should be negative; got {e_xc}");
    assert!(e_xc > -15.0,      "E_xc is too negative: {e_xc}");
    assert!(e_xc < -3.0,       "E_xc is too small in magnitude: {e_xc}");
}

#[test]
fn vxc_pbe_h2o_is_hermitian() {
    let mol = build_h2o();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let grid = build_atomic_grid(&mol, &AtomicGridConfig::default());
    let pts: Vec<[f64;3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();
    let dens = eval_density_closed(&rhf.density_total, &chi, &dchi);

    let xc = xc_def_from_name("PBE").unwrap();
    let (e_xc, vxc) = semilocal_vxc_closed(&grid, &chi, &dchi, &dens, None, &xc);

    let n = vxc.nrows();
    let mut asym: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let a = (vxc[(i, j)] - vxc[(j, i)]).abs();
            if a > asym { asym = a; }
        }
    }
    assert!(asym < 1e-12, "V_xc(PBE) not Hermitian: max asym = {asym:.2e}");

    eprintln!("PBE E_xc(H2O, cc-pVDZ) = {e_xc:.6} Ha");
    assert!(e_xc < 0.0 && e_xc > -15.0, "PBE E_xc out of range: {e_xc}");
}
