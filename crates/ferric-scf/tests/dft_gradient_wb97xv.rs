//! FD validation of the full wB97X-V gradient (RSH + VV10 nonlocal).
//!
//! The analytic gradient includes the moving-grid VV10 term (AO, point
//! motion, pair distance and Becke weights; see `vv10_gradient_moving_grid.rs`
//! for the fixed-density checks), so it is the derivative of the SCF energy
//! and matches a central difference of re-converged energies to the
//! difference's own error. The complement (wB97X, no VV10) is exercised in
//! `dft_gradient_rsh.rs`.
//!
//! Bar: measured agreement is 5.8e-8..8.6e-8 Ha/Bohr on all four cases
//! (step 5e-4 Bohr, SCF to 1e-11 Ha, f64 RI-J), the O(h²) floor of the
//! difference; `BAR = 3e-7` is ~3.5x that. The VV10 grid terms are worth
//! 5.5e-6 on water/6-31G (`vv10_gradient_moving_grid.rs`), well above it.
//!
//! Each case runs 2·natoms·3 extra wB97X-V SCFs (~35 s in a release test
//! build), hence the medium tier.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Analytic-vs-FD bar (Ha/Bohr); see the module doc.
const BAR: f64 = 3e-7;

fn cfg() -> RhfConfig {
    RhfConfig::default()
        // f64 RI-J: the f32 default jitters the energy surface by ~1e-7 Ha,
        // above these bars (site/src/methods/scf.md, "RI-J tensor precision").
        .with_ri_j_precision(ferric_scf::df_j::RiJPrecision::F64)
        .with_xc("wB97X-V")
        .with_df_j_aux("def2-universal-jkfit")
        .with_df_k_aux("def2-universal-jkfit")
        .with_energy_conv(1e-11)
        .with_density_conv(1e-9)
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
                0 => {
                    mol_p.atoms[atom].x += delta;
                    mol_m.atoms[atom].x -= delta;
                }
                1 => {
                    mol_p.atoms[atom].y += delta;
                    mol_m.atoms[atom].y -= delta;
                }
                _ => {
                    mol_p.atoms[atom].zpos += delta;
                    mol_m.atoms[atom].zpos -= delta;
                }
            }
            let prep_p = PreparedBasis::new(&mol_p, &bs).unwrap();
            let bounds_p = SchwarzBounds::compute(Operator::coulomb(), &prep_p).unwrap();
            let res_p = solve_rhf(
                &ParallelContext::default(),
                &mol_p,
                &prep_p,
                Operator::coulomb(),
                &bounds_p,
                &cfg,
            )
            .unwrap();
            let prep_m = PreparedBasis::new(&mol_m, &bs).unwrap();
            let bounds_m = SchwarzBounds::compute(Operator::coulomb(), &prep_m).unwrap();
            let res_m = solve_rhf(
                &ParallelContext::default(),
                &mol_m,
                &prep_m,
                Operator::coulomb(),
                &bounds_m,
                &cfg,
            )
            .unwrap();
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
    let g_ana = ks_gradient_closed(&mol, &prep, &bs, op, &bounds, "wB97X-V", &res, None).unwrap();
    let g_fd = fd_gradient(xyz, basis_name, 5e-4);

    eprintln!("=== {label} wB97X-V gradient analytic vs FD ===");
    let mut max_diff = 0.0_f64;
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let diff = (g_ana[(a, c)] - g_fd[(a, c)]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            eprintln!(
                "  atom={a} coord={c}: ana={:+.6e} fd={:+.6e} diff={:.2e}",
                g_ana[(a, c)],
                g_fd[(a, c)],
                diff
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
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
fn wb97xv_gradient_h2_sto3g_vs_fd() {
    run_case("H2/sto-3g", "2\nH2\nH 0 0 0\nH 0 0 0.74\n", "sto-3g", BAR);
}

#[test]
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
fn wb97xv_gradient_h2o_sto3g_vs_fd() {
    run_case(
        "H2O/sto-3g",
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        "sto-3g",
        BAR,
    );
}

#[test]
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
fn wb97xv_gradient_h2_ccpvdz_vs_fd() {
    run_case("H2/cc-pVDZ", "2\nH2\nH 0 0 0\nH 0 0 0.74\n", "cc-pvdz", BAR);
}

#[test]
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
fn wb97xv_gradient_h2o_ccpvdz_vs_fd() {
    run_case(
        "H2O/cc-pVDZ",
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        "cc-pvdz",
        BAR,
    );
}
