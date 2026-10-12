//! VALIDATION tier - VALIDATION.md row "Analytic RHF Hessian, RI-J + RI-K".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-scf --test validation_df_hessian_rik \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What ferric computes (read from `crates/ferric-scf/src/hessian_df_k.rs`)
//!
//! `rhf_hessian_parts` with `df_j_aux = df_k_aux = def2-universal-jkfit`: both
//! the Coulomb energy `1/2 d^T V^-1 d` and the exchange energy
//! `-1/4 sum_ij L_ij . V^-1 L_ij` are fitted, so NO four-centre integral enters
//! (skeleton from the 3c/2c second-derivative stencil, `J^x` and `K^x` with the
//! auxiliary-fit parts, the SCF's own `DfJ` and `DfK` in the CPHF kernel).
//!
//! # References (`scripts/validation/gen_df_hessian.py --jk` -> `testdata/reference/validation/df_hessian_jk/`)
//!
//! PySCF fed ferric's own basis and auxiliary-basis JSON, Bohr geometry and
//! masses. PySCF's standard `density_fit(auxbasis)` RHF fits J and K with the
//! same auxiliary basis, so its analytic Hessian `mf.Hessian()`
//! (`pyscf.df.hessian.rhf`, full auxiliary response) is a DIRECT reference:
//!
//! * `hessian_analytic` - PySCF's analytic DF Hessian;
//! * `hessian_fd` - central FD of PySCF's analytic DF gradient;
//! * `hessian_exact` - PySCF's plain exact-integral Hessian (negative control).
//!
//! Systems: H2O, CH2O, distorted NH3 x cc-pVDZ / def2-universal-jkfit;
//! distorted H2O x def2-SVP / def2-universal-jkfit.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric is right: total and frequencies match PySCF's analytic DF
//!   Hessian at the reference's own floor.
//! * Skeleton wrong / `K^x` wrong / CPHF kernel wrong: the total misses; the
//!   production-wired mutants below miss by >= `MUST_MISS_FACTOR` x the bar.
//! * Harness wrong (an auxiliary basis that is not the same, a geometry
//!   constant): nuclear repulsion, AO count or the SCF energy fail first.
//! * Both codes share a convention error: not excluded by PySCF alone; the
//!   independent anchors are the own-gradient FD and the rank-one identity in
//!   `tests/df_hessian_rik.rs`.
//!
//! # NEGATIVE CONTROLS (asserted inside the tests)
//!
//! * the exact-integral Hessians (PySCF's and ferric's own), RI-J with exact K,
//!   and RI-K with exact J must each miss the reference by >= `MUST_MISS_FACTOR`
//!   x the Hessian bar;
//! * RI-K terms dropped (`K^x`, its metric part, the 2c second derivative, the
//!   cross terms) must each miss by the same factor.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{atom_masses, frequencies_from_cartesian_hessian};
use ferric_scf::hessian::rij_support::{rhf_hessian_parts_rik_mutated, RikMutants};
use ferric_scf::hessian::{rhf_hessian, rhf_hessian_parts};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/df_hessian_jk";
const MOL_DIR: &str = "testdata/molecules/validation";
const AUX: &str = "def2-universal-jkfit";
const SYSTEMS: [(&str, &str); 4] = [
    ("h2o", "cc-pvdz"),
    ("ch2o", "cc-pvdz"),
    ("nh3_distorted", "cc-pvdz"),
    ("h2o_distorted", "def2-svp"),
];

/// Nuclear repulsion, Ha: geometry/constant like-for-like.
const TOL_ENUC: f64 = 1e-9;
/// SCF energy vs reference, Ha (pins geometry, basis, auxiliary basis, fit).
const TOL_ENERGY: f64 = 1e-8;
/// Total vs PySCF's analytic DF Hessian, Ha/Bohr^2. Measured 2026-10-11:
/// 3.3e-8 (H2O), 8.2e-8 (CH2O), 1.8e-8 (NH3), 1.6e-7 (H2O/def2-SVP, where the
/// reference's own raw asymmetry is 1.0e-7); bar = 10x the worst. The smallest
/// negative-control margin at this bar is 17x (PySCF exact, def2-SVP).
const TOL_HESS_ANALYTIC: f64 = 1.6e-6;
/// Total vs PySCF's FD Hessian of its own DF gradient: that reference's own
/// floor (the generator measured FD-vs-analytic 3.4e-7 .. 1.6e-6; ferric vs it
/// 3.3e-7, 1.6e-6 (CH2O), 5.0e-7, 3.3e-7); bar = 6x the worst.
const TOL_HESS_FD: f64 = 1e-5;
/// Frequencies vs PySCF's analytic frequencies, cm-1. Measured <= 6.9e-4;
/// bar = 14x. Exact-integral frequencies miss by 0.28 .. 0.54 cm-1.
const TOL_FREQ: f64 = 1e-2;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 10.0;

fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("testdata").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-scf manifest dir should be <root>/crates/ferric-scf")
        .to_path_buf()
}

/// Load a reference JSON. Missing or unparsable is a HARD failure.
fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_df_hessian.py --jk - a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn num(v: &Value, ptr: &str, ctx: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a number"))
}

fn vec_f64(v: &Value, ptr: &str, ctx: &str) -> Vec<f64> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn matrix(v: &Value, ptr: &str, ctx: &str) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a matrix"));
    let n = rows.len();
    let mut m = Array2::<f64>::zeros((n, n));
    for (i, row) in rows.iter().enumerate() {
        let row = row.as_array().expect("matrix row");
        assert_eq!(row.len(), n, "{ctx}: {ptr} is not square");
        for (j, x) in row.iter().enumerate() {
            m[(i, j)] = x.as_f64().expect("number");
        }
    }
    m
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "Hessian shape");
    a.iter()
        .zip(b.iter())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn max_freq_diff(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "frequency count differs");
    let mut g = got.to_vec();
    let mut w = want.to_vec();
    g.sort_by(|a, b| a.partial_cmp(b).unwrap());
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    g.iter()
        .zip(&w)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
}

fn check_hessian(ctx: &str, what: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) {
    let d = max_abs_diff(got, want);
    eprintln!("{ctx}: vs {what:<9} max|d| {d:.2e} Ha/Bohr^2 (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: Hessian vs {what}: max|d| {d:.3e} >= {tol:.0e}"
    );
}

fn must_miss(ctx: &str, what: &str, got: &Array2<f64>, want: &Array2<f64>, bar: f64) {
    let d = max_abs_diff(got, want);
    eprintln!(
        "{ctx}: NEGATIVE CONTROL {what}: miss {d:.2e} = {:.0}x the bar {bar:.0e}",
        d / bar
    );
    assert!(
        d > MUST_MISS_FACTOR * bar,
        "{ctx}: NEGATIVE CONTROL {what} misses by only {d:.3e} (<= {MUST_MISS_FACTOR} x {bar:.0e}) \
         - the comparison cannot see this term"
    );
}

fn scf_config(df_j: &str, df_k: &str) -> RhfConfig {
    RhfConfig {
        max_iter: 500,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some(df_j.to_string()),
        df_k_aux: Some(df_k.to_string()),
        ..Default::default()
    }
}

fn check_system(system: &str, basis_name: &str) {
    if ferric_integrals::engine::libint_max_deriv_order() < 2 {
        eprintln!("SKIP: libint2 without second derivatives; the controls need them");
        return;
    }
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}/{AUX}");
    assert_eq!(r["aux_basis"].as_str(), Some(AUX), "{ctx}: aux basis");
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));

    // --- harness like-for-like ---------------------------------------------
    let enuc_ref = num(&r, "/nuclear_repulsion", &ctx);
    assert!(
        (mol.nuclear_repulsion() - enuc_ref).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion - geometry/unit mismatch"
    );
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    assert_eq!(
        prep.nbasis() as u64,
        r["nao"].as_u64().expect("nao"),
        "{ctx}: AO count"
    );
    let masses = atom_masses(&mol).unwrap();
    assert_eq!(
        masses,
        vec_f64(&r, "/masses_amu", &ctx),
        "{ctx}: masses differ"
    );

    // --- RI-J + RI-K SCF -----------------------------------------------------
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctxp = ParallelContext::default();
    let cfg = scf_config(AUX, AUX);
    let rhf = solve_rhf(&ctxp, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged, "{ctx}: SCF did not converge");
    assert!(
        rhf.df_jk
            .as_ref()
            .is_some_and(|r| r.j_aux.is_some() && r.k_aux.is_some()),
        "{ctx}: no RI-J + RI-K route"
    );
    let e_ref = num(&r, "/energy", &ctx);
    eprintln!("{ctx}: SCF energy diff {:.2e} Ha", rhf.energy - e_ref);
    assert!(
        (rhf.energy - e_ref).abs() < TOL_ENERGY,
        "{ctx}: SCF energy {:.12} vs {e_ref:.12}",
        rhf.energy
    );

    // --- Hessian ------------------------------------------------------------
    let parts = rhf_hessian_parts(&ctxp, &mol, &prep, op, &bounds, &rhf, &cfg).unwrap();
    eprintln!(
        "{ctx}: CPHF iterations {:?}, max residual {:.2e}, response asymmetry {:.2e}",
        parts.cphf_iterations, parts.cphf_max_residual, parts.response_asymmetry
    );
    let total = parts.total();
    let h_an = matrix(&r, "/hessian_analytic", &ctx);
    check_hessian(&ctx, "analytic", &total, &h_an, TOL_HESS_ANALYTIC);
    check_hessian(
        &ctx,
        "FD",
        &total,
        &matrix(&r, "/hessian_fd", &ctx),
        TOL_HESS_FD,
    );

    // --- frequencies --------------------------------------------------------
    let freqs = frequencies_from_cartesian_hessian(&mol, &total, &masses)
        .unwrap()
        .frequencies;
    let f_ref = vec_f64(&r, "/freq_analytic_cm", &ctx);
    let df = max_freq_diff(&freqs, &f_ref);
    eprintln!("{ctx}: freqs max|d| {df:.3e} cm-1 (tol {TOL_FREQ:.0e}) ferric {freqs:.2?}");
    assert!(
        df < TOL_FREQ,
        "{ctx}: frequencies max|d| {df:.4} >= {TOL_FREQ}"
    );

    // --- negative controls ----------------------------------------------------
    must_miss(
        &ctx,
        "PySCF exact",
        &matrix(&r, "/hessian_exact", &ctx),
        &h_an,
        TOL_HESS_ANALYTIC,
    );
    let hess_of = |j: &str, k: &str| {
        let c = scf_config(j, k);
        let s = solve_rhf(&ctxp, &mol, &prep, op, &bounds, &c).unwrap();
        rhf_hessian(&ctxp, &mol, &prep, op, &bounds, &s, &c).unwrap()
    };
    let h_ex = hess_of("", "");
    must_miss(&ctx, "ferric exact", &h_ex, &h_an, TOL_HESS_ANALYTIC);
    let f_ex = frequencies_from_cartesian_hessian(&mol, &h_ex, &masses)
        .unwrap()
        .frequencies;
    let df_ex = max_freq_diff(&f_ex, &f_ref);
    eprintln!("{ctx}: exact frequencies miss {df_ex:.3} cm-1 (bar {TOL_FREQ} cm-1)");
    assert!(
        df_ex > MUST_MISS_FACTOR * TOL_FREQ,
        "{ctx}: exact frequencies miss by only {df_ex:.3}"
    );
    must_miss(
        &ctx,
        "ferric RI-J only",
        &hess_of(AUX, ""),
        &h_an,
        TOL_HESS_ANALYTIC,
    );
    must_miss(
        &ctx,
        "ferric RI-K only",
        &hess_of("", AUX),
        &h_an,
        TOL_HESS_ANALYTIC,
    );
    // Production-wired defects in the RI-K terms (on the RI-J + RI-K SCF).
    let d = RikMutants::default();
    let mutants = [
        ("K^x dropped", RikMutants { drop_kx: true, ..d }),
        (
            "metric response of K^x dropped",
            RikMutants {
                drop_kx_metric: true,
                ..d
            },
        ),
        (
            "aux-centre part of K^x omitted",
            RikMutants {
                omit_aux_centre_kx: true,
                ..d
            },
        ),
        (
            "metric second derivative dropped",
            RikMutants {
                drop_metric_second: true,
                ..d
            },
        ),
        (
            "cross terms dropped",
            RikMutants {
                drop_cross: true,
                ..d
            },
        ),
    ];
    for (name, m) in mutants {
        let h = rhf_hessian_parts_rik_mutated(&ctxp, &mol, &prep, op, &bounds, &rhf, &cfg, &m)
            .unwrap()
            .total();
        must_miss(&ctx, name, &h, &h_an, TOL_HESS_ANALYTIC);
    }
}

#[test]
#[ignore = "validation: analytic DF Hessian"]
fn all_reference_files_present() {
    for (system, basis_name) in SYSTEMS {
        let r = reference(system, basis_name);
        assert_eq!(r["system"].as_str(), Some(system));
        assert_eq!(r["basis"].as_str(), Some(basis_name));
        assert_eq!(r["aux_basis"].as_str(), Some(AUX));
    }
}

#[test]
#[ignore = "validation: analytic DF Hessian"]
fn h2o_ccpvdz() {
    check_system("h2o", "cc-pvdz");
}

#[test]
#[ignore = "validation: analytic DF Hessian"]
fn ch2o_ccpvdz() {
    check_system("ch2o", "cc-pvdz");
}

#[test]
#[ignore = "validation: analytic DF Hessian"]
fn nh3_distorted_ccpvdz() {
    check_system("nh3_distorted", "cc-pvdz");
}

#[test]
#[ignore = "validation: analytic DF Hessian"]
fn h2o_distorted_def2svp() {
    check_system("h2o_distorted", "def2-svp");
}
