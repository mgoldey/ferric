//! FD validation of the KS gradient on a basis that exercises f-shells.
//!
//! cc-pVTZ isn't bundled, but aug-cc-pVTZ carries an f-set on heavy atoms
//! (one pure-f shell on oxygen, 7 functions). This is the smallest cheap
//! probe that exercises the f-shell AO Hessian path in the GGA gradient
//! assembly. (def2-TZVP is also bundled but its contraction normalization
//! conflicts with ferric's primitive-norm convention — separate bug.)

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

fn cfg() -> RhfConfig {
    RhfConfig {
        xc: Some("PBE".into()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    }
}

fn fd_gradient(xyz: &str, basis_name: &str, delta: f64) -> Array2<f64> {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
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
            let res_p = solve_rhf(
                &ParallelContext::default(), &mol_p, &prep_p, Operator::coulomb(), &bounds_p, &cfg,
            ).unwrap();
            let prep_m = PreparedBasis::new(&mol_m, &bs).unwrap();
            let bounds_m = SchwarzBounds::compute(Operator::coulomb(), &prep_m).unwrap();
            let res_m = solve_rhf(
                &ParallelContext::default(), &mol_m, &prep_m, Operator::coulomb(), &bounds_m, &cfg,
            ).unwrap();
            grad[(atom, coord)] = (res_p.energy - res_m.energy) / (2.0 * delta);
        }
    }
    grad
}

fn run_case(label: &str, xyz: &str, basis_name: &str, tol: f64) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = cfg();
    let res = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
    let g_ana = ks_gradient_closed(&mol, &prep, &bs, op, &bounds, "PBE", &res).unwrap();
    let g_fd = fd_gradient(xyz, basis_name, 5e-4);

    eprintln!("=== {label} PBE gradient analytic vs FD ===");
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

/// H2/aug-cc-pVTZ places one (7-pure) f-set on hydrogen — smallest f-shell
/// SCF gradient smoke. Hydrogen f-functions are diffuse but mathematically
/// exercise the full Cartesian-Hessian assembly.
#[test]
fn pbe_gradient_h2_augccpvtz_vs_fd() {
    run_case(
        "H2/aug-cc-pVTZ",
        "2\nH2\nH 0 0 0\nH 0 0 0.74\n",
        "aug-cc-pvtz",
        2e-3,
    );
}
