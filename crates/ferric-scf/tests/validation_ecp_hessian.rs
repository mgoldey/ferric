//! VALIDATION tier — VALIDATION.md row "Analytic Hessian with ECPs".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-scf --test validation_ecp_hessian \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What ferric computes (read from `crates/ferric-scf/src/hessian_ecp.rs`)
//!
//! The analytic RHF/UHF Hessian of `hessian.rs` plus two ECP terms: the
//! skeleton `Σ D ∂²V_ECP` as a masked fourth-order stencil of the
//! first-derivative ECP block kernel (default backend: ferric's quadrature,
//! `FERRIC_ECP_BACKEND`), and `∂V_ECP/∂x` in the first-order Fock matrices of
//! the CPHF response.
//!
//! # References (`scripts/validation/gen_ecp_hessian.py` → `testdata/reference/validation/ecp_hessian/`)
//!
//! PySCF fed ferric's own basis JSON, ECP, Bohr geometry and masses, exact
//! four-index J/K:
//!
//! * `hessian_analytic` — `mf.Hessian().kernel()` (ECP integrals
//!   `ECPscalar_ipipnuc`, `ipnucip`, `ipiprinv`, `iprinvip`).
//! * `hessian_skeleton` — `partial_hess_elec + hess_nuc`.
//! * `hessian_ecp_skeleton` — the ECP-only part of that skeleton, rebuilt from
//!   the same four integral families; the generator checks it against a
//!   fixed-density 5-point FD of PySCF's own analytic ECP gradient term.
//! * `hessian_fd` — central FD of PySCF's analytic gradient.
//!
//! The comparison needs no AO-order mapping: every compared quantity is a
//! 3N × 3N matrix (the density contraction happens on each side in its own AO
//! order); ferric's AO order only enters through its own D.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric is right: the ECP skeleton and the total Hessian match PySCF's
//!   analytic Hessian at the SCF-offset floor (the ECP energy offset against
//!   PySCF is ~1e-8 Ha, `validation_ecp.rs`), frequencies to ~1e-2 cm⁻¹.
//! * Skeleton stencil wrong (wrong atom in the centre fold, a mask that drops
//!   triples, a wrong weight sign): `hessian_ecp_skeleton` misses and so does
//!   the total, by the same amount.
//! * `∂V_ECP/∂x` missing from `F^x`: the skeletons match, only the total
//!   misses (the CPHF response is the only place it enters).
//! * Different SCF state or basis: the energy check fails first.
//!
//! # TOLERANCES
//!
//! Each bar sits above ferric's measured maximum, written next to it.
//!
//! # NEGATIVE CONTROLS (asserted inside the tests)
//!
//! Dropping the ECP skeleton from the total must miss PySCF's analytic Hessian
//! by ≥ `MUST_MISS_FACTOR` × the bar, and so must the frequencies.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{atom_masses, frequencies_from_cartesian_hessian};
use ferric_scf::hessian::{rhf_hessian_parts, uhf_hessian_parts};
use ferric_scf::hessian_ecp::ecp_skeleton_hessian;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/ecp_hessian";
const MOL_DIR: &str = "testdata/molecules/validation/ecp";

/// Nuclear repulsion (effective charges), Ha.
const TOL_ENUC: f64 = 1e-9;
/// SCF energy vs reference, Ha (pins geometry, basis, ECP AND the state).
const TOL_ENERGY: f64 = 2.5e-7;
/// ECP-only skeleton vs the fixed-density FD of PySCF's own analytic ECP
/// gradient term, Ha/Bohr². Measured ≤ 5.95e-8 (CH2I; 2.6e-10 HI, 8.2e-9 CH3I).
const TOL_ECP_SKELETON_FD: f64 = 5e-7;
/// ECP-only skeleton vs PySCF's analytic second-derivative integrals,
/// Ha/Bohr². PySCF's own roughness: its analytic ECP skeleton misses the FD of
/// its own gradient term by the SAME amount (generator
/// `hessian_ecp_skeleton_fixed_density_fd_max_err`: 2.5e-7 CH3I, 3.7e-6 CH2I),
/// so no bar tighter than that is meaningful. Measured ≤ 3.66e-6 (CH2I; 2.6e-10
/// HI, 2.6e-7 CH3I).
const TOL_ECP_SKELETON: f64 = 1e-5;
/// Whole skeleton vs PySCF `partial_hess_elec + hess_nuc`, Ha/Bohr².
const TOL_SKELETON: f64 = 1e-5;
/// Total Hessian vs PySCF's analytic Hessian, Ha/Bohr² (same roughness).
const TOL_HESS: f64 = 1e-5;
// (measured vs PySCF analytic: 9.2e-9 HI, 2.6e-7 CH3I, 3.7e-6 CH2I)
/// Total Hessian vs PySCF's FD Hessian, Ha/Bohr². Measured ≤ 4.4e-7 (CH3I;
/// 1.05e-7 HI, 3.2e-7 CH2I).
const TOL_HESS_FD: f64 = 2e-6;
/// Frequencies vs PySCF's analytic-Hessian frequencies, cm⁻¹.
const TOL_FREQ: f64 = 5e-2;
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
             scripts/validation/gen_ecp_hessian.py — a missing reference is a failure, never a skip",
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
    eprintln!("{ctx}: vs {what:<12} max|d| {d:.2e} Ha/Bohr^2 (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: Hessian vs {what}: max|d| {d:.3e} >= {tol:.0e}"
    );
}

fn scf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 500,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        check_stability: true,
        scf_stability_descent: true,
        ..Default::default()
    }
}

fn check_system(system: &str, basis_name: &str) {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let open_shell = r["method"].as_str() == Some("uhf");
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mut mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let bs = basis::bundled(basis_name).unwrap();
    mol.apply_ecp(&bs);

    // --- harness like-for-like ---------------------------------------------
    let core: Vec<i64> = mol.atoms.iter().map(|a| a.n_core_ecp as i64).collect();
    let core_ref: Vec<i64> = r["ecp_core_electrons"]
        .as_array()
        .expect("ecp_core_electrons")
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(core, core_ref, "{ctx}: per-atom ECP core electrons");
    assert_eq!(mol.nelec() as i64, r["nelectron"].as_i64().unwrap());
    let enuc = mol.nuclear_repulsion();
    let enuc_ref = num(&r, "/nuclear_repulsion", &ctx);
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion {enuc:.12} vs {enuc_ref:.12}"
    );
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    assert_eq!(prep.nbasis() as u64, r["nao"].as_u64().expect("nao"));
    let masses = atom_masses(&mol).unwrap();
    assert_eq!(masses, vec_f64(&r, "/masses_amu", &ctx), "{ctx}: masses");

    // --- SCF (same state as the reference) ----------------------------------
    let (op, cfg, pc) = (
        Operator::coulomb(),
        scf_config(),
        ParallelContext::default(),
    );
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let scf = if open_shell {
        solve_uhf(&pc, &mol, &prep, &bounds, &cfg).unwrap()
    } else {
        solve_rhf(&pc, &mol, &prep, op, &bounds, &cfg).unwrap()
    };
    assert!(scf.converged, "{ctx}: SCF did not converge");
    let (e, e_ref) = (scf.energy, num(&r, "/energy", &ctx));
    eprintln!(
        "{ctx}: SCF energy offset vs PySCF {:.2e} Ha",
        (e - e_ref).abs()
    );
    assert!(
        (e - e_ref).abs() < TOL_ENERGY,
        "{ctx}: state/energy {e} vs {e_ref}"
    );

    // --- Hessian ---------------------------------------------------------------
    let parts = if open_shell {
        uhf_hessian_parts(&pc, &mol, &prep, op, &bounds, &scf, &cfg).unwrap()
    } else {
        rhf_hessian_parts(&pc, &mol, &prep, op, &bounds, &scf, &cfg).unwrap()
    };
    let d_tot = if open_shell {
        &scf.density_alpha + scf.density_beta.as_ref().unwrap()
    } else {
        scf.density_r().clone()
    };
    let ecp_skel = ecp_skeleton_hessian(&mol, &bs, &d_tot).unwrap();
    let (total, skeleton) = (parts.total(), parts.skeleton());
    let h_an = matrix(&r, "/hessian_analytic", &ctx);
    let ecp_ref = matrix(&r, "/hessian_ecp_skeleton", &ctx);
    eprintln!(
        "{ctx}: ECP skeleton max element {:.3e}; CPHF max residual {:.2e}",
        ecp_ref.iter().fold(0.0f64, |m, v| m.max(v.abs())),
        parts.cphf_max_residual
    );
    check_hessian(
        &ctx,
        "ECP skel FD",
        &ecp_skel,
        &matrix(&r, "/hessian_ecp_skeleton_fd", &ctx),
        TOL_ECP_SKELETON_FD,
    );
    check_hessian(&ctx, "ECP skeleton", &ecp_skel, &ecp_ref, TOL_ECP_SKELETON);
    check_hessian(
        &ctx,
        "skeleton",
        &skeleton,
        &matrix(&r, "/hessian_skeleton", &ctx),
        TOL_SKELETON,
    );
    check_hessian(&ctx, "analytic", &total, &h_an, TOL_HESS);
    check_hessian(
        &ctx,
        "FD",
        &total,
        &matrix(&r, "/hessian_fd", &ctx),
        TOL_HESS_FD,
    );

    // --- frequencies -------------------------------------------------------------
    let f_of = |h: &Array2<f64>| {
        frequencies_from_cartesian_hessian(&mol, h, &masses)
            .unwrap()
            .frequencies
    };
    let f_ref = vec_f64(&r, "/freq_analytic_cm", &ctx);
    let df = max_freq_diff(&f_of(&total), &f_ref);
    eprintln!("{ctx}: freqs max|d| {df:.3e} cm-1 (tol {TOL_FREQ:.0e})");
    assert!(df < TOL_FREQ, "{ctx}: frequencies max|d| {df:.4} cm-1");

    // --- negative control: no ECP skeleton -----------------------------------------
    let no_skel = &total - &ecp_skel;
    let miss = max_abs_diff(&no_skel, &h_an);
    let miss_f = max_freq_diff(&f_of(&no_skel), &f_ref);
    eprintln!(
        "{ctx}: without the ECP skeleton: Hessian misses by {miss:.3e}, freqs by {miss_f:.2} cm-1"
    );
    assert!(
        miss > MUST_MISS_FACTOR * TOL_HESS,
        "{ctx}: NEGATIVE CONTROL: no-skeleton Hessian misses by only {miss:.3e}"
    );
    assert!(
        miss_f > MUST_MISS_FACTOR * TOL_FREQ,
        "{ctx}: NEGATIVE CONTROL: no-skeleton frequencies miss by only {miss_f:.3} cm-1"
    );
}

#[test]
#[ignore = "validation: analytic ECP Hessian"]
fn all_reference_files_present() {
    for (system, basis_name) in [
        ("hi", "def2-svp"),
        ("ch3i", "def2-svp"),
        ("ag2", "aug-cc-pvdz-pp"),
        ("ch2i", "def2-svp"),
    ] {
        let r = reference(system, basis_name);
        assert_eq!(r["system"].as_str(), Some(system));
        assert_eq!(r["basis"].as_str(), Some(basis_name));
    }
}

#[test]
#[ignore = "validation: analytic ECP Hessian"]
fn hi_def2svp() {
    check_system("hi", "def2-svp");
}

#[test]
#[ignore = "validation: analytic ECP Hessian"]
fn ch3i_def2svp() {
    check_system("ch3i", "def2-svp");
}

#[test]
#[ignore = "validation: analytic ECP Hessian"]
fn ag2_augccpvdzpp() {
    check_system("ag2", "aug-cc-pvdz-pp");
}

#[test]
#[ignore = "validation: analytic ECP Hessian (UHF)"]
fn ch2i_uhf_def2svp() {
    check_system("ch2i", "def2-svp");
}
