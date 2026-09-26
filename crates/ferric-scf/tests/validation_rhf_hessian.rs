//! VALIDATION tier — VALIDATION.md row "Analytic RHF Hessian".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-scf --test validation_rhf_hessian \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What ferric computes (read from `crates/ferric-scf/src/hessian.rs`)
//!
//! `rhf_hessian_parts` assembles the closed-shell RHF Hessian with exact
//! four-centre J/K from libint2 second-derivative integrals, term by term as
//! PySCF's `hessian/rhf.py` does: nuclear repulsion, `D·h''` (kinetic +
//! nuclear attraction including the nuclear-centre derivatives), `−W·S''`,
//! `Γ·ERI''` (the frozen-density "skeleton"), plus the CPHF orbital response
//! solved by preconditioned conjugate gradient. `analytic_frequencies` feeds
//! the total through the same mass-weighting/projection as the FD path.
//!
//! # References (`scripts/validation/gen_rhf_hessian.py` → `testdata/reference/validation/rhf_hessian/`)
//!
//! PySCF fed ferric's own basis JSON, Bohr geometry and masses:
//!
//! * `hessian_analytic` — `mf.Hessian().kernel()`, CPHF tolerance 1e-12,
//!   symmetrized (raw asymmetry recorded). The like-for-like reference.
//! * `hessian_skeleton` — `partial_hess_elec + hess_nuc`: terms 1–4 alone, so
//!   a miss can be localized to skeleton vs response.
//! * `hessian_fd` — central FD (1e-3 Bohr) of PySCF's analytic gradient:
//!   PySCF's own independent check of its analytic Hessian.
//! * Frequencies: `harmonic_analysis(mass = ferric's masses)`.
//!
//! Systems: H2O, NH3, CH2O × cc-pVDZ; off-C2v H2O × def2-SVP.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric is right: skeleton and total match PySCF's analytic Hessian at
//!   the CPHF/SCF convergence floor (~1e-8 Ha/Bohr²), frequencies to ~1e-4
//!   cm⁻¹; vs PySCF's FD Hessian at that FD's own gap (~5e-7).
//! * Skeleton wrong (integral order, missing nuclear-centre derivatives, a
//!   factor in Γ or W): `hessian_skeleton` misses; the total misses by the
//!   same amount.
//! * Response wrong: `hessian_skeleton` matches, the total misses.
//! * Harness wrong (geometry constant, basis, masses): nuclear repulsion, AO
//!   count, SCF energy or the exact mass equality fails first.
//!
//! # TOLERANCES
//!
//! Each bar sits above ferric's measured maximum, written next to it.
//! Reference-side floors (PySCF against
//! itself, measured by the generator 2026-09-25). PySCF's raw analytic
//! asymmetry is also 2.1e-9 for CH2O and 1.0e-7 for distorted H2O/def2-SVP;
//! the latter is the floor under that case's 1.6e-7 agreement:
//!
//! | quantity | h2o/cc-pVDZ | nh3/cc-pVDZ |
//! |---|---:|---:|
//! | raw analytic Hessian asymmetry | 8.5e-9 | 4.1e-9 Ha/Bohr² |
//! | max abs(H_FD − H_analytic) | 3.4e-7 | 4.5e-7 Ha/Bohr² |
//! | FD vs analytic frequencies | 2.4e-3 | 1.4e-3 cm⁻¹ |
//!
//! # NEGATIVE CONTROLS (asserted inside the tests)
//!
//! * No response: ferric's skeleton alone must miss PySCF's total Hessian by
//!   ≥ `MUST_MISS_FACTOR` × the Hessian bar (the comparison can see term 5).
//! * Response sign: skeleton − response must miss by the same factor.
//! * Frequencies without response must miss the reference frequencies by
//!   ≥ `MUST_MISS_FACTOR` × the frequency bar.
//!
//! # MUTATION
//!
//! The per-term mutations in `tests/rhf_hessian_fd.rs` apply here too; the
//! localization is: skeleton mutations fail `hessian_skeleton` AND the total,
//! the CPHF mutation (`rhs -= …G(D_oo)…` → `+=` in `solve_cphf_one`) fails
//! only the total.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{atom_masses, frequencies_from_cartesian_hessian};
use ferric_scf::hessian::rhf_hessian_parts;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/rhf_hessian";
const MOL_DIR: &str = "testdata/molecules/validation";
const SYSTEMS: [(&str, &str); 4] = [
    ("h2o", "cc-pvdz"),
    ("nh3", "cc-pvdz"),
    ("ch2o", "cc-pvdz"),
    ("h2o_distorted", "def2-svp"),
];

/// Nuclear repulsion, Ha: geometry/constant like-for-like.
const TOL_ENUC: f64 = 1e-9;
/// SCF energy vs reference, Ha (pins geometry, basis, exact J/K).
const TOL_ENERGY: f64 = 1e-8;
/// Total Hessian vs PySCF's analytic Hessian, Ha/Bohr², elementwise.
/// Measured ≤ 1.6e-7 (distorted H2O/def2-SVP), ≤ 8.2e-8 at cc-pVDZ.
const TOL_HESS_ANALYTIC: f64 = 1e-6;
/// Skeleton (terms 1–4) vs PySCF `partial_hess_elec + hess_nuc`, Ha/Bohr².
/// Measured ≤ 5.9e-10.
const TOL_HESS_SKELETON: f64 = 1e-8;
/// Total Hessian vs PySCF's FD Hessian, Ha/Bohr². Measured ≤ 1.6e-6 (CH2O), the
/// reference FD's own floor.
const TOL_HESS_FD: f64 = 5e-6;
/// Frequencies vs PySCF's analytic-Hessian frequencies, cm⁻¹. Measured ≤ 6.9e-4.
const TOL_FREQ: f64 = 1e-2;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 100.0;

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
             scripts/validation/gen_rhf_hessian.py — a missing reference is a failure, never a skip",
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

fn must_miss_hessian(ctx: &str, what: &str, got: &Array2<f64>, want: &Array2<f64>, bar: f64) {
    let d = max_abs_diff(got, want);
    assert!(
        d > MUST_MISS_FACTOR * bar,
        "{ctx}: NEGATIVE CONTROL {what} misses by only {d:.3e} (<= {MUST_MISS_FACTOR} x {bar:.0e}) \
         — the comparison cannot see this term"
    );
}

fn scf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 500,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        ..Default::default()
    }
}

fn check_system(system: &str, basis_name: &str) {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));

    // --- harness like-for-like ---------------------------------------------
    let enuc = mol.nuclear_repulsion();
    let enuc_ref = num(&r, "/nuclear_repulsion", &ctx);
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion {enuc:.12} vs {enuc_ref:.12} — geometry/unit mismatch"
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

    // --- SCF ------------------------------------------------------------------
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctxp = ParallelContext::default();
    let cfg = scf_config();
    let rhf = solve_rhf(&ctxp, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged, "{ctx}: SCF did not converge");
    let e_ref = num(&r, "/energy", &ctx);
    assert!(
        (rhf.energy - e_ref).abs() < TOL_ENERGY,
        "{ctx}: SCF energy {:.12} vs {e_ref:.12}",
        rhf.energy
    );

    // --- Hessian ----------------------------------------------------------------
    let parts = rhf_hessian_parts(&ctxp, &mol, &prep, op, &bounds, &rhf, &cfg).unwrap();
    eprintln!(
        "{ctx}: CPHF iterations {:?}, max residual {:.2e}, response asymmetry {:.2e}",
        parts.cphf_iterations, parts.cphf_max_residual, parts.response_asymmetry
    );
    let total = parts.total();
    let skeleton = parts.skeleton();
    let h_an = matrix(&r, "/hessian_analytic", &ctx);
    check_hessian(
        &ctx,
        "skeleton",
        &skeleton,
        &matrix(&r, "/hessian_skeleton", &ctx),
        TOL_HESS_SKELETON,
    );
    check_hessian(&ctx, "analytic", &total, &h_an, TOL_HESS_ANALYTIC);
    check_hessian(
        &ctx,
        "FD",
        &total,
        &matrix(&r, "/hessian_fd", &ctx),
        TOL_HESS_FD,
    );

    // --- frequencies -----------------------------------------------------------
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

    // --- negative controls -----------------------------------------------------
    must_miss_hessian(&ctx, "skeleton-only", &skeleton, &h_an, TOL_HESS_ANALYTIC);
    must_miss_hessian(
        &ctx,
        "response sign flipped",
        &(&skeleton - &parts.response),
        &h_an,
        TOL_HESS_ANALYTIC,
    );
    let f_skel = frequencies_from_cartesian_hessian(&mol, &skeleton, &masses)
        .unwrap()
        .frequencies;
    let df_skel = max_freq_diff(&f_skel, &f_ref);
    assert!(
        df_skel > MUST_MISS_FACTOR * TOL_FREQ,
        "{ctx}: NEGATIVE CONTROL skeleton-only frequencies miss by only {df_skel:.3} cm-1"
    );
}

#[test]
#[ignore = "validation: RHF analytic Hessian vs PySCF"]
fn all_reference_files_present() {
    for (system, basis_name) in SYSTEMS {
        let r = reference(system, basis_name);
        assert_eq!(r["system"].as_str(), Some(system));
        assert_eq!(r["basis"].as_str(), Some(basis_name));
    }
}

#[test]
#[ignore = "validation: RHF analytic Hessian vs PySCF"]
fn h2o_ccpvdz() {
    check_system("h2o", "cc-pvdz");
}

#[test]
#[ignore = "validation: RHF analytic Hessian vs PySCF"]
fn nh3_ccpvdz() {
    check_system("nh3", "cc-pvdz");
}

#[test]
#[ignore = "validation: RHF analytic Hessian vs PySCF"]
fn ch2o_ccpvdz() {
    check_system("ch2o", "cc-pvdz");
}

#[test]
#[ignore = "validation: RHF analytic Hessian vs PySCF"]
fn h2o_distorted_def2svp() {
    check_system("h2o_distorted", "def2-svp");
}
