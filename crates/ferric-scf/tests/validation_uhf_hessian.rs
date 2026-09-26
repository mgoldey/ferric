//! VALIDATION tier — VALIDATION.md row "Analytic UHF Hessian".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-scf --test validation_uhf_hessian \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What ferric computes (read from `crates/ferric-scf/src/hessian.rs`)
//!
//! `uhf_hessian_parts` assembles the UHF Hessian with exact four-centre J/K
//! from libint2 second-derivative integrals, term by term as PySCF's
//! `hessian/uhf.py` does: nuclear repulsion, `D·h''` with `D = D_α + D_β`,
//! `−W·S''` with the UHF `W`, `Γ·ERI''` with the UHF `Γ` (the frozen-density
//! "skeleton"), plus the orbital response from ONE coupled α/β CPHF per
//! nuclear coordinate, solved by preconditioned conjugate gradient.
//!
//! # References (`scripts/validation/gen_uhf_hessian.py` → `testdata/reference/validation/uhf_hessian/`)
//!
//! PySCF fed ferric's own basis JSON, Bohr geometry and masses, on the lowest
//! internally STABLE UHF state (`common.run_open_shell`):
//!
//! * `hessian_analytic` — `mf.Hessian().kernel()`, CPHF tolerance 1e-12,
//!   symmetrized (raw asymmetry recorded). The like-for-like reference.
//! * `hessian_skeleton` — `partial_hess_elec + hess_nuc`: terms 1–4 alone.
//! * `hessian_fd` — central FD (1e-3 Bohr) of PySCF's analytic UHF gradient,
//!   displaced SCFs seeded from the centre densities.
//! * Frequencies: `harmonic_analysis(mass = ferric's masses)`.
//!
//! Systems: OH ²Π, NH₂ ²B₁, CH₂ ³B₁ × cc-pVDZ.
//!
//! # OH: a degenerate state
//!
//! The singly-occupied π orbital of linear ²Π OH may point at any angle about
//! the bond (z) axis: rotating it is an exact zero mode of the UHF orbital
//! Hessian (PySCF's λ_min is 5e-12 here), so ferric and PySCF may converge to
//! the same state rotated by an arbitrary angle. The TOTAL Hessian is
//! independent of that angle (PySCF's own xx and yy blocks agree to 3e-7), but
//! the skeleton's perpendicular blocks are not (PySCF: −0.011 vs 0.052
//! Ha/Bohr²). For OH the skeleton is therefore compared through quantities
//! invariant under a rotation about z (per atom pair: `zz`, the trace, the
//! antisymmetric part and the Frobenius norm of the xy sub-block, and the norms
//! of the xz/yz and zx/zy pairs); the total and the frequencies are compared
//! element by element as for the other systems. The reference records the
//! α−β quadrupole (`beta_hole_quadrupole`) for a reviewer who wants the angle.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric is right: skeleton and total match PySCF's analytic Hessian at
//!   the CPHF/SCF convergence floor (~1e-7 Ha/Bohr², OH set by PySCF's own
//!   raw asymmetry 9.9e-8), frequencies to ~1e-3 cm⁻¹; vs PySCF's FD Hessian
//!   at that FD's own gap (≤ 1.1e-6).
//! * Skeleton wrong (Γ or W for D_α ≠ D_β): `hessian_skeleton` misses and the
//!   total misses by the same amount.
//! * Response wrong (spin coupling, occupations): the skeleton matches, the
//!   total misses.
//! * Different state (ferric on another UHF solution): the SCF energy check
//!   fails first.
//! * Harness wrong (geometry constant, basis, masses): nuclear repulsion, AO
//!   count or the exact mass equality fails first.
//!
//! # TOLERANCES
//!
//! Each bar sits above ferric's measured maximum, written next to it.
//! Reference-side floors
//! (PySCF against itself, generator run 2026-09-25):
//!
//! | quantity | oh | nh2 | ch2_triplet |
//! |---|---:|---:|---:|
//! | raw analytic Hessian asymmetry | 9.9e-8 | 1.8e-9 | 6.3e-9 Ha/Bohr² |
//! | max abs(H_FD − H_analytic) | 1.1e-6 | 1.1e-6 | 6.2e-7 Ha/Bohr² |
//! | FD vs analytic frequencies | 4e-3 | 2e-3 | 1e-3 cm⁻¹ |
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
//! The per-term mutations in `tests/uhf_hessian_fd.rs` apply here too; the
//! localization is: `gamma_uhf` → `gamma` and the doubled `W` fail
//! `hessian_skeleton` AND the total; the spin-coupling mutation (cross-spin J
//! dropped in `TwoElectronBuilder::g_uhf`) fails only the total.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{atom_masses, frequencies_from_cartesian_hessian};
use ferric_scf::hessian::uhf_hessian_parts;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/uhf_hessian";
const MOL_DIR: &str = "testdata/molecules/validation";
const SYSTEMS: [(&str, &str); 3] = [
    ("oh", "cc-pvdz"),
    ("nh2", "cc-pvdz"),
    ("ch2_triplet", "cc-pvdz"),
];

/// Nuclear repulsion, Ha: geometry/constant like-for-like.
const TOL_ENUC: f64 = 1e-9;
/// SCF energy vs reference, Ha (pins geometry, basis, exact J/K AND the state).
const TOL_ENERGY: f64 = 1e-8;
/// Total Hessian vs PySCF's analytic Hessian, Ha/Bohr², elementwise.
/// Measured ≤ 3.8e-7 (OH; PySCF's own CPHF converges to ~5.7e-7), ≤ 7.7e-8
/// for NH2 and CH2.
const TOL_HESS_ANALYTIC: f64 = 2e-6;
/// Skeleton (terms 1–4) vs PySCF `partial_hess_elec + hess_nuc`, Ha/Bohr²
/// (OH: rotation invariants). Measured ≤ 1.4e-10.
const TOL_HESS_SKELETON: f64 = 1e-8;
/// Total Hessian vs PySCF's FD Hessian, Ha/Bohr². Measured ≤ 1.1e-6; the
/// reference FD's own floor is 1.1e-6.
const TOL_HESS_FD: f64 = 5e-6;
/// Frequencies vs PySCF's analytic-Hessian frequencies, cm⁻¹. Measured ≤ 1.4e-4.
const TOL_FREQ: f64 = 2e-3;
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
             scripts/validation/gen_uhf_hessian.py — a missing reference is a failure, never a skip",
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

/// Quantities of a Hessian invariant under one rotation about the z axis
/// applied to every atom, per ordered atom pair (A, B) with 3×3 block `b`:
/// `b_zz`, `tr M`, `M_xy − M_yx`, `‖M‖_F` (M = the xy sub-block),
/// `‖(b_xz, b_yz)‖` and `‖(b_zx, b_zy)‖`, as a column (n_pairs · 6, 1).
fn z_rotation_invariants(h: &Array2<f64>) -> Array2<f64> {
    let natoms = h.nrows() / 3;
    let mut out = Vec::with_capacity(natoms * natoms * 6);
    for a in 0..natoms {
        for b in 0..natoms {
            let e = |i: usize, j: usize| h[(3 * a + i, 3 * b + j)];
            let frob =
                (e(0, 0).powi(2) + e(0, 1).powi(2) + e(1, 0).powi(2) + e(1, 1).powi(2)).sqrt();
            out.extend([
                e(2, 2),
                e(0, 0) + e(1, 1),
                e(0, 1) - e(1, 0),
                frob,
                e(0, 2).hypot(e(1, 2)),
                e(2, 0).hypot(e(2, 1)),
            ]);
        }
    }
    let n = out.len();
    Array2::from_shape_vec((n, 1), out).expect("column")
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

fn load_molecule(system: &str, mult: usize) -> Molecule {
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()))
}

fn check_system(system: &str, basis_name: &str) {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let mol = load_molecule(system, mult);

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

    // --- SCF (same stable state as the reference) ---------------------------
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctxp = ParallelContext::default();
    let cfg = scf_config();
    let uhf = solve_uhf(&ctxp, &mol, &prep, &bounds, &cfg).unwrap();
    assert!(uhf.converged, "{ctx}: SCF did not converge");
    let e_ref = num(&r, "/energy", &ctx);
    assert!(
        (uhf.energy - e_ref).abs() < TOL_ENERGY,
        "{ctx}: SCF energy {:.12} vs {e_ref:.12} — a different UHF state",
        uhf.energy
    );

    // --- Hessian ----------------------------------------------------------------
    let parts = uhf_hessian_parts(&ctxp, &mol, &prep, op, &bounds, &uhf, &cfg).unwrap();
    eprintln!(
        "{ctx}: CPHF iterations {:?}, max residual {:.2e}, response asymmetry {:.2e}",
        parts.cphf_iterations, parts.cphf_max_residual, parts.response_asymmetry
    );
    let total = parts.total();
    let skeleton = parts.skeleton();
    let h_an = matrix(&r, "/hessian_analytic", &ctx);
    let skel_ref = matrix(&r, "/hessian_skeleton", &ctx);
    if r.get("beta_hole_quadrupole").is_some() {
        // Degenerate π hole (see the module doc): the skeleton is compared
        // through its z-rotation invariants.
        check_hessian(
            &ctx,
            "skeleton (z-rotation invariants)",
            &z_rotation_invariants(&skeleton),
            &z_rotation_invariants(&skel_ref),
            TOL_HESS_SKELETON,
        );
    } else {
        check_hessian(&ctx, "skeleton", &skeleton, &skel_ref, TOL_HESS_SKELETON);
    }
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
#[ignore = "validation: UHF analytic Hessian vs PySCF"]
fn all_reference_files_present() {
    for (system, basis_name) in SYSTEMS {
        let r = reference(system, basis_name);
        assert_eq!(r["system"].as_str(), Some(system));
        assert_eq!(r["basis"].as_str(), Some(basis_name));
    }
}

#[test]
#[ignore = "validation: UHF analytic Hessian vs PySCF"]
fn oh_ccpvdz() {
    check_system("oh", "cc-pvdz");
}

#[test]
#[ignore = "validation: UHF analytic Hessian vs PySCF"]
fn nh2_ccpvdz() {
    check_system("nh2", "cc-pvdz");
}

#[test]
#[ignore = "validation: UHF analytic Hessian vs PySCF"]
fn ch2_triplet_ccpvdz() {
    check_system("ch2_triplet", "cc-pvdz");
}
