//! Analytic RHF/UHF Hessian with an effective core potential AND density-fitted
//! J/K (RI-J, RI-K, RI-JK) (issue #406).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test ecp_df_hessian
//! ```
//!
//! # Why this composes
//!
//! `V_ECP` is a one-electron operator of the core Hamiltonian. The fitted J/K
//! Hessian replaces only the four-index skeleton and the `J^x`/`K^x` and CPHF
//! pieces; the ECP skeleton `Σ D ∂²V_ECP` and `∂V_ECP/∂x` in `F^x` are added by
//! the same shared code (`extra_skeleton`, `add_extra_first_order`) on the exact
//! and the fitted routes. The refusal was a precaution, not a missing term.
//!
//! # Anchors and what each can see
//!
//! * FD: analytic Hessian vs central FD of ferric's own analytic RI gradient
//!   (ECP in hcore on both sides), H-O-I RHF with RI-J, RI-K and RI-JK and
//!   H-I-H UHF with RI-JK. Independent of any Hessian formula. A defect in
//!   either ECP term appears at O(skeleton) ~1e-1; the physics floor is ~1e-7.
//! * PySCF: HI and CH3I def2-SVP, `density_fit(def2-universal-jkfit)` RHF with
//!   the same ECP (references `testdata/reference/validation/ecp_df_hessian`,
//!   generator `scripts/validation/gen_ecp_df_hessian.py`; PySCF's own FD vs
//!   analytic gap is 1.1e-7 HI / 4.4e-7 CH3I).
//! * Negative controls: dropped `dV_ECP/dx` in `F^x`, dropped ECP skeleton, and
//!   the exact-integral (unfitted) Hessian against the RI reference must each
//!   miss by `MUST_MISS` x bar.
//!
//! Artifact hypothesis, written before measuring: a broken composition (ECP
//! term lost or doubled in the fitted route) misses FD by 1e-2..1e-1; a
//! correct one sits at the FD floor (<1e-6). The 4+ orders separate them.

use std::path::PathBuf;

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{rhf_gradient, rohf_gradient, uhf_gradient};
use ferric_scf::hessian::{rhf_hessian_parts, rohf_hessian_parts, uhf_hessian_parts};
use ferric_scf::hessian_ecp::ecp_skeleton_hessian;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ndarray::Array2;

const AUX: &str = "def2-universal-jkfit";
const TOL_FD: f64 = 5e-6;
/// vs PySCF analytic (measured max 2.7e-7 or less; see the test output).
const TOL_PYSCF: f64 = 1e-5;
const MUST_MISS: f64 = 10.0;
const H_FD: f64 = 1e-3;

const HOI: &str = "3\nHOI tilted\nH 0.30 -0.40 0.20\nO 0.00 0.10 0.95\nI 0.45 0.20 2.80\n";
const HIH: &str = "3\nHIH tilted\nH 0.30 -0.40 0.20\nI 0.00 0.00 1.61\nH -1.10 0.70 2.90\n";

fn has_deriv2() -> bool {
    ferric_integrals::engine::libint_max_deriv_order() >= 2
}

#[derive(Clone, Copy)]
enum Fit {
    Exact,
    J,
    K,
    Jk,
}

fn config(fit: Fit) -> RhfConfig {
    let aux = || Some(AUX.to_string());
    let (j, k) = match fit {
        Fit::Exact => (None, None),
        Fit::J => (aux(), None),
        Fit::K => (None, aux()),
        Fit::Jk => (aux(), aux()),
    };
    RhfConfig {
        max_iter: 300,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some(j.unwrap_or_default()),
        df_k_aux: Some(k.unwrap_or_default()),
        ..Default::default()
    }
}

fn ecp_mol(xyz: &str, mult: usize, bs: &BasisSet) -> Molecule {
    let mut mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    mol.apply_ecp(bs);
    mol
}

struct Solved {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    scf: ScfResult,
}

fn solve(mol: &Molecule, bs: &BasisSet, open: bool, fit: Fit) -> Solved {
    let prep = PreparedBasis::new(mol, bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = config(fit);
    let scf = if open {
        solve_uhf(&ctx, mol, &prep, &bounds, &cfg).unwrap()
    } else {
        solve_rhf(&ctx, mol, &prep, op, &bounds, &cfg).unwrap()
    };
    assert!(scf.converged);
    Solved {
        mol: mol.clone(),
        prep,
        bounds,
        scf,
    }
}

fn density(s: &Solved, open: bool) -> Array2<f64> {
    if open {
        &s.scf.density_alpha + s.scf.density_beta.as_ref().unwrap()
    } else {
        s.scf.density_r().clone()
    }
}

fn hessian(s: &Solved, open: bool, fit: Fit) -> Array2<f64> {
    let (ctx, op, cfg) = (ParallelContext::default(), Operator::coulomb(), config(fit));
    let parts = if open {
        uhf_hessian_parts(&ctx, &s.mol, &s.prep, op, &s.bounds, &s.scf, &cfg).unwrap()
    } else {
        rhf_hessian_parts(&ctx, &s.mol, &s.prep, op, &s.bounds, &s.scf, &cfg).unwrap()
    };
    parts.total()
}

fn displaced(mol: &Molecule, atom: usize, coord: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match coord {
        0 => m.atoms[atom].x += h,
        1 => m.atoms[atom].y += h,
        _ => m.atoms[atom].zpos += h,
    }
    m
}

fn gradient(mol: &Molecule, bs: &BasisSet, open: bool, fit: Fit) -> Array2<f64> {
    let s = solve(mol, bs, open, fit);
    let op = Operator::coulomb();
    if open {
        uhf_gradient(&s.mol, &s.prep, op, &s.bounds, &s.scf, None).unwrap()
    } else {
        rhf_gradient(&s.mol, &s.prep, op, &s.bounds, &s.scf, None).unwrap()
    }
}

fn fd_columns(mol: &Molecule, bs: &BasisSet, open: bool, fit: Fit, cols: &[usize]) -> Array2<f64> {
    let n3 = 3 * mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    for &b in cols {
        let g = |h: f64| gradient(&displaced(mol, b / 3, b % 3, h), bs, open, fit);
        let (gp, gm) = (g(H_FD), g(-H_FD));
        for a in 0..n3 {
            out[(a, b)] = (gp[(a / 3, a % 3)] - gm[(a / 3, a % 3)]) / (2.0 * H_FD);
        }
    }
    out
}

fn max_diff_cols(a: &Array2<f64>, b: &Array2<f64>, cols: &[usize]) -> f64 {
    let mut m = 0.0f64;
    for &c in cols {
        for r in 0..a.nrows() {
            assert!(a[(r, c)].is_finite() && b[(r, c)].is_finite(), "non-finite");
            m = m.max((a[(r, c)] - b[(r, c)]).abs());
        }
    }
    m
}

/// Analytic vs FD of the analytic RI gradient, with the ECP-term controls.
fn check_against_fd(label: &str, xyz: &str, mult: usize, open: bool, fit: Fit, cols: &[usize]) {
    if !has_deriv2() {
        eprintln!("{label}: skipped, libint2 without second derivatives");
        return;
    }
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(xyz, mult, &bs);
    let s = solve(&mol, &bs, open, fit);
    let an = hessian(&s, open, fit);
    let fd = fd_columns(&mol, &bs, open, fit, cols);
    let d = max_diff_cols(&an, &fd, cols);
    eprintln!("{label}: max|analytic - FD| = {d:.3e} (bar {TOL_FD:.0e})");
    assert!(d < TOL_FD, "{label}: max|analytic - FD| = {d:.3e}");

    // Control 1: ECP skeleton removed.
    let skel = ecp_skeleton_hessian(&mol, &bs, &density(&s, open)).unwrap();
    let miss_skel = max_diff_cols(&(&an - &skel), &fd, cols);
    eprintln!("  {label}: no ECP skeleton misses FD by {miss_skel:.3e}");
    assert!(miss_skel > MUST_MISS * TOL_FD, "{label}: skeleton vacuous");

    // Control 2: dV_ECP/dx dropped from F^x (the fitted route's response).
    let ctx = ParallelContext::default();
    let cfg = config(fit);
    let dropped = ferric_scf::hessian::rij_support::with_ecp_first_order_dropped(|| {
        if open {
            uhf_hessian_parts(
                &ctx,
                &s.mol,
                &s.prep,
                Operator::coulomb(),
                &s.bounds,
                &s.scf,
                &cfg,
            )
        } else {
            rhf_hessian_parts(
                &ctx,
                &s.mol,
                &s.prep,
                Operator::coulomb(),
                &s.bounds,
                &s.scf,
                &cfg,
            )
        }
        .unwrap()
        .total()
    });
    let miss_fx = max_diff_cols(&dropped, &fd, cols);
    eprintln!("  {label}: no dV_ECP/dx in F^x misses FD by {miss_fx:.3e}");
    assert!(miss_fx > MUST_MISS * TOL_FD, "{label}: F^x control vacuous");
}

#[test]
fn rhf_ri_jk_ecp_matches_fd_of_the_ri_gradient() {
    check_against_fd("HOI RHF RI-JK", HOI, 1, false, Fit::Jk, &[0, 4, 8]);
}

#[test]
fn rhf_ri_j_ecp_matches_fd_of_the_ri_gradient() {
    check_against_fd("HOI RHF RI-J", HOI, 1, false, Fit::J, &[1, 8]);

    // RI-J-only is not the RI-JK Hessian (guards a Fit::J that silently fits K).
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(HOI, 1, &bs);
    let hj = hessian(&solve(&mol, &bs, false, Fit::J), false, Fit::J);
    let hjk = hessian(&solve(&mol, &bs, false, Fit::Jk), false, Fit::Jk);
    let d = max_diff_cols(&hj, &hjk, &(0..9).collect::<Vec<_>>());
    eprintln!("HOI RI-J vs RI-JK Hessian: {d:.3e}");
    assert!(d > 1e-5, "RI-J and RI-JK Hessians coincide ({d:.3e})");
}

#[test]
fn rhf_ri_k_ecp_matches_fd_of_the_ri_gradient() {
    check_against_fd("HOI RHF RI-K", HOI, 1, false, Fit::K, &[2, 7]);
}

#[test]
fn uhf_ri_jk_ecp_matches_fd_of_the_ri_gradient() {
    check_against_fd("HIH UHF RI-JK", HIH, 2, true, Fit::Jk, &[0, 5, 8]);
}

// ---------------------------------------------------------------------------
// ROHF (the ECP terms are the shared ones; the commutator-form response reads
// the same F^x). Reference: FD of ferric's analytic ROHF gradient.
// ---------------------------------------------------------------------------

fn rohf_solve(mol: &Molecule, bs: &BasisSet, fit: Fit) -> Solved {
    let prep = PreparedBasis::new(mol, bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let scf = solve_rohf(
        &ParallelContext::default(),
        mol,
        &prep,
        op,
        &bounds,
        &config(fit),
    )
    .unwrap();
    assert!(scf.converged);
    Solved {
        mol: mol.clone(),
        prep,
        bounds,
        scf,
    }
}

fn rohf_hessian_of(s: &Solved, fit: Fit) -> Array2<f64> {
    let (ctx, op, cfg) = (ParallelContext::default(), Operator::coulomb(), config(fit));
    rohf_hessian_parts(&ctx, &s.mol, &s.prep, op, &s.bounds, &s.scf, &cfg)
        .unwrap()
        .total()
}

fn check_rohf(label: &str, fit: Fit, cols: &[usize]) {
    if !has_deriv2() {
        eprintln!("{label}: skipped, libint2 without second derivatives");
        return;
    }
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(HIH, 2, &bs);
    let s = rohf_solve(&mol, &bs, fit);
    let an = rohf_hessian_of(&s, fit);
    let n3 = 3 * mol.atoms.len();
    let mut fd = Array2::<f64>::zeros((n3, n3));
    for &b in cols {
        let g = |h: f64| {
            let t = rohf_solve(&displaced(&mol, b / 3, b % 3, h), &bs, fit);
            assert!(
                (t.scf.energy - s.scf.energy).abs() < 1e-3,
                "displaced ROHF changed state"
            );
            rohf_gradient(
                &t.mol,
                &t.prep,
                Operator::coulomb(),
                &t.bounds,
                &t.scf,
                None,
            )
            .unwrap()
        };
        let (gp, gm) = (g(H_FD), g(-H_FD));
        for a in 0..n3 {
            fd[(a, b)] = (gp[(a / 3, a % 3)] - gm[(a / 3, a % 3)]) / (2.0 * H_FD);
        }
    }
    let d = max_diff_cols(&an, &fd, cols);
    eprintln!("{label}: max|analytic - FD| = {d:.3e} (bar {TOL_FD:.0e})");
    assert!(d < TOL_FD, "{label}: max|analytic - FD| = {d:.3e}");

    let d_tot = &s.scf.density_alpha + s.scf.density_beta.as_ref().unwrap();
    let skel = ecp_skeleton_hessian(&mol, &bs, &d_tot).unwrap();
    let miss_skel = max_diff_cols(&(&an - &skel), &fd, cols);
    eprintln!("  {label}: no ECP skeleton misses FD by {miss_skel:.3e}");
    assert!(miss_skel > MUST_MISS * TOL_FD, "{label}: skeleton vacuous");

    let dropped =
        ferric_scf::hessian::rij_support::with_ecp_first_order_dropped(|| rohf_hessian_of(&s, fit));
    let miss_fx = max_diff_cols(&dropped, &fd, cols);
    eprintln!("  {label}: no dV_ECP/dx in F^x misses FD by {miss_fx:.3e}");
    assert!(miss_fx > MUST_MISS * TOL_FD, "{label}: F^x control vacuous");
}

#[test]
fn rohf_exact_ecp_matches_fd_of_the_rohf_gradient() {
    check_rohf("HIH ROHF exact", Fit::Exact, &[0, 5, 8]);
}

#[test]
fn rohf_ri_j_ecp_matches_fd_of_the_rohf_gradient() {
    check_rohf("HIH ROHF RI-J", Fit::J, &[0, 5, 8]);
}

// ---------------------------------------------------------------------------
// PySCF
// ---------------------------------------------------------------------------

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf()
}

fn matrix(v: &serde_json::Value, key: &str) -> Array2<f64> {
    let rows = v[key].as_array().unwrap_or_else(|| panic!("{key} missing"));
    let n = rows.len();
    Array2::from_shape_fn((n, n), |(i, j)| rows[i][j].as_f64().unwrap())
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn check_pyscf(system: &str) {
    if !has_deriv2() {
        eprintln!("{system}: skipped, libint2 without second derivatives");
        return;
    }
    let path = root().join(format!(
        "testdata/reference/validation/ecp_df_hessian/{system}_def2-svp.json"
    ));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing reference {} ({e})", path.display()));
    let r: serde_json::Value = serde_json::from_str(&text).unwrap();
    let xyz = root().join(format!("testdata/molecules/validation/ecp/{system}.xyz"));
    let mut mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    mol.apply_ecp(&bs);
    let s = solve(&mol, &bs, false, Fit::Jk);
    let e_ref = r["energy"].as_f64().unwrap();
    assert!(
        (s.scf.energy - e_ref).abs() < 2.5e-7,
        "{system}: state/energy {} vs {e_ref}",
        s.scf.energy
    );
    let h = hessian(&s, false, Fit::Jk);
    let d_an = max_diff(&h, &matrix(&r, "hessian_analytic"));
    let d_fd = max_diff(&h, &matrix(&r, "hessian_fd"));
    eprintln!(
        "{system}: vs PySCF analytic {d_an:.2e}, vs PySCF FD {d_fd:.2e} (bar {TOL_PYSCF:.0e})"
    );
    assert!(d_an < TOL_PYSCF, "{system}: vs PySCF analytic {d_an:.3e}");
    assert!(d_fd < TOL_PYSCF, "{system}: vs PySCF FD {d_fd:.3e}");

    // Control: the unfitted Hessian of the unfitted SCF is the wrong answer
    // for this reference by (at least) the fitting error it was generated with.
    let fit_signal = r["exact_vs_ri_jk_hessian_max_abs"].as_f64().unwrap();
    let ex = solve(&mol, &bs, false, Fit::Exact);
    let h_ex = hessian(&ex, false, Fit::Exact);
    let miss = max_diff(&h_ex, &matrix(&r, "hessian_analytic"));
    eprintln!("  {system}: exact-integral Hessian misses the RI reference by {miss:.3e} (PySCF fit signal {fit_signal:.2e})");
    assert!(
        miss > 0.5 * fit_signal && miss > 3.0 * d_an,
        "{system}: fitting control vacuous"
    );
}

#[test]
fn hi_ri_jk_ecp_matches_pyscf() {
    check_pyscf("hi");
}

#[test]
fn ch3i_ri_jk_ecp_matches_pyscf() {
    check_pyscf("ch3i");
}

// ---------------------------------------------------------------------------
// Probe
// ---------------------------------------------------------------------------

#[test]
fn probe_accepts_ri_with_ecp_but_still_refuses_ri_with_environment() {
    if !has_deriv2() {
        return;
    }
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(HOI, 1, &bs);
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    // RI-J: the RI-K metric of an iodine system has cond ~1e9 with jkfit, which
    // the probe refuses for its own reason (not the ECP).
    let cfg = config(Fit::J);
    ferric_scf::hessian::analytic_hessian_available(&mol, &prep, Operator::coulomb(), &cfg)
        .expect("RI + ECP is analytic");

    let cfg = RhfConfig {
        external_potential: Some(ferric_core::external_potential::ExternalPotential {
            field: Some([0.0, 0.0, 0.004]),
            ..Default::default()
        }),
        ..cfg
    };
    let err =
        ferric_scf::hessian::analytic_hessian_available(&mol, &prep, Operator::coulomb(), &cfg)
            .expect_err("RI + environment stays refused");
    assert!(
        err.to_string()
            .contains("combined with an external potential"),
        "{err}"
    );
}
