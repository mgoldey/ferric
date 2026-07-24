//! FD validation of the PBE (GGA) and B3LYP (hybrid GGA) closed-shell nuclear
//! gradients. AO Hessians are only implemented for s/p shells, so these tests
//! restrict to basis sets without d functions (STO-3G, 6-31G).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use rayon::prelude::*;

fn rhf_cfg(xc: &str, hybrid: bool) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: if hybrid { Some("def2-universal-jkfit".into()) } else { None },
        // energy_conv follows rhf.rs's documented convergence philosophy
        // (RhfConfig::default doc comment): dp_rms is the real convergence
        // signal; ΔE floors with naux (DF-J/DF-K are in play here) and is
        // only a loose "not still descending" sanity bound. The previous
        // energy_conv=1e-10 demanded a gate scf_converged is designed to
        // never satisfy under DF, so affected perturbed geometries silently
        // ran to max_iter with a stale (sometimes wrong-basin) density —
        // root cause of the FD reference blowing up at isolated
        // displacements while every other component tracked the analytic
        // gradient.
        energy_conv: 1e-8,
        density_conv: 1e-8,
        max_iter: 500,
        ..Default::default()
    }
}

fn fd_gradient(xyz: &str, basis_name: &str, xc: &str, hybrid: bool, delta: f64) -> Array2<f64> {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let natoms = mol.atoms.len();
    let mut grad = Array2::<f64>::zeros((natoms, 3));
    let bs = basis::bundled(basis_name).unwrap();

    // Seed every perturbed SCF from the *equilibrium*-geometry converged
    // density rather than a fresh SAD guess at each displaced geometry.
    // With a fresh SAD guess at every one of the 3*natoms perturbed points,
    // DIIS occasionally lands B3LYP/PBE on a different (higher-energy) SCF
    // solution than its +/- twin — the two energies then differ by ~0.1 Ha
    // instead of ~delta*(true force), producing a nonphysical FD "gradient"
    // (observed: isolated components off by 40-220 Ha/Bohr while every other
    // component agreed with the analytic gradient to <1e-4). Continuation
    // from one common, well-converged starting density keeps every
    // displacement in the same electronic-state basin, which is the
    // standard way to do numerical FD gradient checks.
    let base_cfg = rhf_cfg(xc, hybrid);
    let prep0 = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds0 = SchwarzBounds::compute(Operator::coulomb(), &prep0).unwrap();
    let res0 = solve_rhf(
        &ParallelContext::default(), &mol, &prep0, Operator::coulomb(), &bounds0, &base_cfg,
    ).unwrap();
    let cfg = RhfConfig {
        init_guess_density: Some(res0.density_r().clone()),
        use_sad_guess: false,
        ..base_cfg
    };

    // Each (atom, coord) displacement pair is independent given the shared
    // `cfg` (which embeds the common equilibrium-density seed established
    // above) -- fan out over rayon. Each solve_rhf call still serializes its
    // own BLAS internally, so nesting under this outer iterator is safe (the
    // same pattern the production code uses, e.g. rimp2.rs's per-pair
    // parallelism over per-i BLAS3 GEMMs).
    let pairs: Vec<(usize, usize)> = (0..natoms).flat_map(|a| (0..3).map(move |c| (a, c))).collect();
    let results: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(atom, coord)| {
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
            assert!(res_p.converged, "FD solve did not converge at atom={atom} coord={coord} (+): exit={:?}", res_p.exit);
            let prep_m = PreparedBasis::new(&mol_m, &bs).unwrap();
            let bounds_m = SchwarzBounds::compute(Operator::coulomb(), &prep_m).unwrap();
            let res_m = solve_rhf(
                &ParallelContext::default(), &mol_m, &prep_m, Operator::coulomb(), &bounds_m, &cfg,
            ).unwrap();
            assert!(res_m.converged, "FD solve did not converge at atom={atom} coord={coord} (-): exit={:?}", res_m.exit);
            ((atom, coord), (res_p.energy - res_m.energy) / (2.0 * delta))
        })
        .collect();
    for ((atom, coord), g) in results {
        grad[(atom, coord)] = g;
    }
    grad
}

fn run_fd_test(label: &str, xyz: &str, basis_name: &str, xc: &str, hybrid: bool, tol: f64) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = rhf_cfg(xc, hybrid);
    let res = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();

    let g_ana = ks_gradient_closed(&mol, &prep, &bs, op, &bounds, xc, &res, None).unwrap();
    let g_fd = fd_gradient(xyz, basis_name, xc, hybrid, 5e-4);

    eprintln!("=== {label} {xc} gradient (analytic vs FD) ===");
    let mut max_diff = 0.0_f64;
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let diff = (g_ana[(a, c)] - g_fd[(a, c)]).abs();
            max_diff = max_diff.max(diff);
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
            assert!(diff < tol, "{label} {xc}: atom={a} coord={c} diff={diff:.2e}");
        }
    }
}

#[test]
fn pbe_gradient_h2_sto3g_vs_fd() {
    run_fd_test("H2/sto-3g", "2\nH2\nH 0 0 0\nH 0 0 0.74\n", "sto-3g", "PBE", false, 1e-3);
}

#[test]
fn pbe_gradient_h2o_sto3g_vs_fd() {
    run_fd_test("H2O/sto-3g",
                "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
                "sto-3g", "PBE", false, 2e-3);
}

#[test]
fn b3lyp_gradient_h2_sto3g_vs_fd() {
    run_fd_test("H2/sto-3g", "2\nH2\nH 0 0 0\nH 0 0 0.74\n", "sto-3g", "B3LYP", true, 1e-3);
}

#[test]
fn pbe_gradient_h2_ccpvdz_vs_fd() {
    run_fd_test("H2/cc-pVDZ", "2\nH2\nH 0 0 0\nH 0 0 0.74\n", "cc-pvdz", "PBE", false, 1e-3);
}

#[test]
fn pbe_gradient_h2o_ccpvdz_vs_fd() {
    run_fd_test("H2O/cc-pVDZ",
                "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
                "cc-pvdz", "PBE", false, 2e-3);
}

#[test]
fn b3lyp_gradient_h2o_ccpvdz_vs_fd() {
    run_fd_test("H2O/cc-pVDZ",
                "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
                "cc-pvdz", "B3LYP", true, 2e-3);
}
