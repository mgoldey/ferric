//! FD validation of the ROKS analytic gradient (LDA, PBE, B3LYP).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::ks_gradient_roks;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

fn cfg(xc: &str) -> RhfConfig {
    // Doublet OH at LDA/PBE benefits from a level shift to damp DIIS
    // oscillations. The shift is rational-damped by err_max in solve_rohf
    // so the converged state is the unshifted stationary point.
    // The SOMO/HOMO near-degeneracy on doublet OH at LDA/PBE makes plain
    // DIIS oscillate without converging. The Maximum-Overlap Method
    // (mom_after_iter) pins the SOMO identity by AO-overlap once DIIS has
    // descended into a basin, which converges these cases tightly — see
    // mom-fixes-oh-lda-plateau. We enable it for every functional (it is a
    // no-op once orbitals are well-separated, as for B3LYP/wB97X-V).
    RhfConfig {
        xc: Some(xc.into()),
        energy_conv: 1e-6,
        density_conv: 1e-3,
        max_iter: 400,
        level_shift: 0.2,
        mom_after_iter: 5,
        ..Default::default()
    }
}

/// Run a single ROKS calculation. The level shift plus MOM (mom_after_iter)
/// in cfg() converge every functional here tightly, including LDA/PBE on
/// doublet OH at FD-displaced geometries — MOM pins the SOMO identity and
/// breaks the DIIS oscillation that used to keep those two cases above
/// density_conv (see mom-fixes-oh-lda-plateau).
fn run_one(mol: &Molecule, prep: &PreparedBasis, bounds: &SchwarzBounds, cfg: &RhfConfig)
    -> ferric_scf::result::ScfResult
{
    solve_rohf(&ParallelContext::default(), mol, prep, Operator::coulomb(), bounds, cfg)
        .unwrap_or_else(|e| panic!("ROKS unexpected error: {e:?}"))
}

fn fd_gradient(xyz: &str, mult: usize, basis_name: &str, xc: &str, delta: f64) -> Array2<f64> {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let natoms = mol.atoms.len();
    let mut grad = Array2::<f64>::zeros((natoms, 3));
    let bs = basis::bundled(basis_name).unwrap();
    let cfg = cfg(xc);
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
            let res_p = run_one(&mol_p, &prep_p, &bounds_p, &cfg);
            let prep_m = PreparedBasis::new(&mol_m, &bs).unwrap();
            let bounds_m = SchwarzBounds::compute(Operator::coulomb(), &prep_m).unwrap();
            let res_m = run_one(&mol_m, &prep_m, &bounds_m, &cfg);
            grad[(atom, coord)] = (res_p.energy - res_m.energy) / (2.0 * delta);
        }
    }
    grad
}

fn run_case(label: &str, xc: &str, xyz: &str, mult: usize, basis_name: &str, tol: f64) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = cfg(xc);
    let res = run_one(&mol, &prep, &bounds, &cfg);
    let g_ana = ks_gradient_roks(&mol, &prep, &bs, op, &bounds, xc, &res).unwrap();
    let g_fd = fd_gradient(xyz, mult, basis_name, xc, 5e-4);

    eprintln!("=== {label} {xc} ROKS gradient analytic vs FD ===");
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

// OH stretched to 1.10 Å so the gradient is well above FD noise.
#[test]
fn roks_grad_oh_ccpvdz_b3lyp() {
    run_case("OH/cc-pVDZ", "B3LYP",
             "2\nOH\nO 0 0 0\nH 0 0 1.10\n", 2, "cc-pvdz", 5e-3);
}

// LDA / PBE ROKS on doublet OH used to plateau at err_max ~ 1e-3 and were
// #[ignore]'d. MOM (mom_after_iter=5 in cfg()) now converges them tightly;
// analytic-vs-FD agrees to ~1.1e-3 Ha/Bohr, well under the 5e-3 tol.
#[test]
fn roks_grad_oh_ccpvdz_pbe() {
    run_case("OH/cc-pVDZ", "PBE",
             "2\nOH\nO 0 0 0\nH 0 0 1.10\n", 2, "cc-pvdz", 5e-3);
}

#[test]
fn roks_grad_oh_ccpvdz_lda() {
    run_case("OH/cc-pVDZ", "LDA",
             "2\nOH\nO 0 0 0\nH 0 0 1.10\n", 2, "cc-pvdz", 5e-3);
}

#[test]
fn roks_rsh_grad_oh_ccpvdz_wb97x_v() {
    run_case("OH/cc-pVDZ", "wB97X-V",
             "2\nOH\nO 0 0 0\nH 0 0 1.10\n", 2, "cc-pvdz", 5e-3);
}
