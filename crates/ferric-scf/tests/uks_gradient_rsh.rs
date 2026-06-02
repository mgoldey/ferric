//! FD validation of UKS-RSH gradient (wB97X-V on open shell).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::ks_gradient_uks;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

fn cfg() -> RhfConfig {
    RhfConfig {
        xc: Some("wB97X-V".into()),
        energy_conv: 1e-7,
        density_conv: 1e-5,
        max_iter: 400,
        ..Default::default()
    }
}

fn fd_gradient(xyz: &str, mult: usize, basis_name: &str, delta: f64) -> Array2<f64> {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let natoms = mol.atoms.len();
    let mut grad = Array2::<f64>::zeros((natoms, 3));
    let bs = basis::bundled(basis_name).unwrap();
    let cfg = cfg();
    for atom in 0..natoms {
        for coord in 0..3 {
            let mut mol_p = mol.clone();
            let mut mol_m = mol.clone();
            match coord {
                0 => { mol_p.atoms[atom].x += delta; mol_m.atoms[atom].x -= delta; }
                1 => { mol_p.atoms[atom].y += delta; mol_m.atoms[atom].y -= delta; }
                _ => { mol_p.atoms[atom].zpos += delta; mol_m.atoms[atom].zpos -= delta; }
            }
            let prep_p = PreparedBasis::new(&mol_p, &bs).unwrap();
            let bounds_p = SchwarzBounds::compute(Operator::coulomb(), &prep_p).unwrap();
            let res_p = solve_uhf(
                &ParallelContext::default(), &mol_p, &prep_p, &bounds_p, &cfg,
            ).unwrap();
            let prep_m = PreparedBasis::new(&mol_m, &bs).unwrap();
            let bounds_m = SchwarzBounds::compute(Operator::coulomb(), &prep_m).unwrap();
            let res_m = solve_uhf(
                &ParallelContext::default(), &mol_m, &prep_m, &bounds_m, &cfg,
            ).unwrap();
            grad[(atom, coord)] = (res_p.energy - res_m.energy) / (2.0 * delta);
        }
    }
    grad
}

fn run_case(label: &str, xyz: &str, mult: usize, basis_name: &str, tol: f64) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = cfg();
    let res = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();
    let g_ana = ks_gradient_uks(&mol, &prep, &bs, op, &bounds, "wB97X-V", &res).unwrap();
    let g_fd = fd_gradient(xyz, mult, basis_name, 5e-4);

    eprintln!("=== {label} wB97X-V UKS gradient analytic vs FD ===");
    let mut max_diff = 0.0_f64;
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let diff = (g_ana[(a, c)] - g_fd[(a, c)]).abs();
            if diff > max_diff { max_diff = diff; }
            eprintln!(
                "  atom={a} coord={c}: ana={:+.6e} fd={:+.6e} diff={:.2e}",
                g_ana[(a, c)], g_fd[(a, c)], diff
            );
        }
    }
    eprintln!("  max diff: {max_diff:.2e}, tol: {tol:.0e}");
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let diff = (g_ana[(a, c)] - g_fd[(a, c)]).abs();
            assert!(diff < tol, "{label}: atom={a} coord={c} diff={diff:.2e}");
        }
    }
}

#[test]
fn uks_rsh_grad_oh_ccpvdz_wb97x_v() {
    run_case("OH/cc-pVDZ", "2\nOH\nO 0 0 0\nH 0 0 1.10\n", 2, "cc-pvdz", 5e-3);
}
