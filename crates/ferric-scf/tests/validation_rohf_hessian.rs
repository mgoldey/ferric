//! VALIDATION tier — VALIDATION.md row "Analytic ROHF Hessian".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-scf --test validation_rohf_hessian \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What ferric computes (read from `crates/ferric-scf/src/hessian/rohf.rs`)
//!
//! `rohf_hessian_parts` assembles the ROHF Hessian with exact four-centre J/K
//! from libint2 second-derivative integrals: nuclear repulsion, `D·h''`,
//! `−W·S''` with `W = sym(D_α F_α D_α + D_α F_β D_β)` (the ROHF energy-weighted
//! density, not the UHF eigenvalue form), `Γ·ERI''` with the UHF `Γ` on the
//! constrained `(D_α, D_β)` (the frozen-density skeleton), plus the orbital
//! response: one coupled closed-virtual / open-virtual / closed-open rotation
//! solve per nuclear coordinate in the commutator-form orbital Hessian.
//!
//! # Reference (`scripts/validation/gen_rohf_hessian.py` → `testdata/reference/validation/rohf_hessian/`)
//!
//! PySCF has no analytic ROHF Hessian. The reference is a central finite
//! difference (step `fd.step`, and `2·step` for the error estimate) of PySCF's
//! ANALYTIC ROHF gradient, on the lowest internally stable ROHF state, fed
//! ferric's basis JSON, Bohr geometry and masses. Frequencies are
//! `harmonic_analysis(mass = ferric's masses)` of that Hessian, cm⁻¹.
//!
//! Systems: NH₂ ²B₁, CH₂ ³B₁, HO₂ ²A″ × cc-pVDZ.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric is right: the Hessian matches PySCF's FD Hessian at that FD's
//!   own error (`|H_h − H_2h|`, 0.7e-6..2.6e-6) and the frequencies to
//!   ~3e-3 cm⁻¹.
//! * Response wrong (rotation blocks, commutator form, `W^y`): the skeleton
//!   matches in `rohf_hessian_skeleton`, the total misses here by 0.3-1.1.
//! * UHF-type response at ROHF orbitals (the wrong operator): the total misses
//!   by 4e-2..8e-2.
//! * Different state (ferric on another ROHF solution): the SCF energy check
//!   fails first. Harness wrong (geometry, basis, masses): nuclear repulsion
//!   or the AO count fails first.
//!
//! # NEGATIVE CONTROLS (asserted inside the tests)
//!
//! * No response: the skeleton alone must miss the reference Hessian by
//!   ≥ `MUST_MISS_FACTOR` × the Hessian bar.
//! * Frequencies of the skeleton alone must miss the reference frequencies by
//!   ≥ `MUST_MISS_FACTOR` × the frequency bar.
//!
//! The mutants of the response (eight, each ≥ 1e6 × the bar on all three
//! systems) are in `tests/rohf_hessian_response.rs`.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{atom_masses, frequencies_from_cartesian_hessian};
use ferric_scf::hessian::rohf_hessian_parts;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/rohf_hessian";
const MOL_DIR: &str = "testdata/molecules/validation";

/// Nuclear repulsion, Ha: geometry/constant like-for-like.
const TOL_ENUC: f64 = 1e-9;
/// SCF energy vs reference, Ha (pins geometry, basis, exact J/K AND the state).
const TOL_ENERGY: f64 = 1e-8;
/// Total Hessian vs PySCF's FD-of-analytic-gradient Hessian, Ha/Bohr².
/// Measured 5.6e-7..1.0e-6 against PySCF's own h-vs-2h gap of 0.7e-6..2.6e-6.
const TOL_HESS_FD: f64 = 3e-6;
/// Frequencies vs the reference's, cm⁻¹. Measured <= 0.0032; PySCF's own
/// h-vs-2h frequency gap is 0.0096.
const TOL_FREQ: f64 = 0.03;
/// A control that ferric must MISS is missed by at least this multiple of the bar.
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
fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_cc-pvdz.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_rohf_hessian.py — a missing reference is a failure, never a skip",
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

/// Exact J/K, tight convergence.
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

fn check_system(system: &str) {
    let order = ferric_integrals::engine::libint_max_deriv_order();
    if order < 2 {
        eprintln!("SKIP: libint2 generated with derivative order {order}; no analytic Hessian");
        return;
    }
    let ctx = format!("{system}/cc-pvdz");
    let r = reference(system);
    let mult = num(&r, "/multiplicity", &ctx) as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, mult).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    assert_eq!(
        prep.nbasis() as f64,
        num(&r, "/nao", &ctx),
        "{ctx}: AO count differs"
    );
    let enuc = (mol.nuclear_repulsion() - num(&r, "/nuclear_repulsion", &ctx)).abs();
    eprintln!("{ctx}: |E_nuc - ref| {enuc:.2e}");
    assert!(
        enuc < TOL_ENUC,
        "{ctx}: nuclear repulsion differs by {enuc:.3e}"
    );

    let (op, pctx) = (Operator::coulomb(), ParallelContext::default());
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = scf_config();
    let rohf = solve_rohf(&pctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(rohf.converged, "{ctx}: ROHF did not converge");
    let de = (rohf.energy - num(&r, "/energy", &ctx)).abs();
    eprintln!("{ctx}: |E_SCF - ref| {de:.2e} Ha");
    assert!(
        de < TOL_ENERGY,
        "{ctx}: SCF energy differs by {de:.3e} (another state?)"
    );

    let t0 = std::time::Instant::now();
    let parts = rohf_hessian_parts(&pctx, &mol, &prep, op, &bounds, &rohf, &cfg).unwrap();
    eprintln!(
        "{ctx}: analytic ROHF Hessian {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    let h = parts.total();
    let h = 0.5 * (&h + &h.t());
    let h_fd = matrix(&r, "/hessian_fd", &ctx);
    let h_fd2 = matrix(&r, "/hessian_fd_2h", &ctx);
    let own = max_abs_diff(&h_fd, &h_fd2);
    let d = max_abs_diff(&h, &h_fd);
    eprintln!(
        "{ctx}: vs PySCF FD (h) {d:.2e}, (2h) {:.2e} Ha/Bohr^2 (tol {TOL_HESS_FD:.0e}); \
         PySCF's own h-vs-2h {own:.2e}",
        max_abs_diff(&h, &h_fd2)
    );
    assert!(d < TOL_HESS_FD, "{ctx}: Hessian vs PySCF FD {d:.3e}");

    // Frequencies, cm^-1, with ferric's masses (the reference's masses are
    // asserted equal).
    let masses = atom_masses(&mol).unwrap();
    for (a, b) in masses.iter().zip(vec_f64(&r, "/masses_amu", &ctx)) {
        assert!((a - b).abs() < 1e-12, "{ctx}: mass {a} vs reference {b}");
    }
    let f = frequencies_from_cartesian_hessian(&mol, &h, &masses).unwrap();
    let f_ref = vec_f64(&r, "/freq_fd_cm", &ctx);
    let df = max_freq_diff(&f.frequencies, &f_ref);
    eprintln!(
        "{ctx}: frequencies {:?} vs {:?}: max|d| {df:.4} cm^-1 (tol {TOL_FREQ})",
        f.frequencies, f_ref
    );
    assert!(df < TOL_FREQ, "{ctx}: frequencies off by {df:.4} cm^-1");

    // Negative controls: the comparison can see the orbital response.
    let skel = parts.skeleton();
    let miss = max_abs_diff(&skel, &h_fd);
    eprintln!("{ctx}: CONTROL no response misses by {miss:.2e}");
    assert!(
        miss > MUST_MISS_FACTOR * TOL_HESS_FD,
        "{ctx}: NEGATIVE CONTROL no-response misses by only {miss:.3e}"
    );
    let f_skel = frequencies_from_cartesian_hessian(&mol, &(0.5 * (&skel + &skel.t())), &masses)
        .map(|r| max_freq_diff(&r.frequencies, &f_ref));
    if let Ok(miss_f) = f_skel {
        eprintln!("{ctx}: CONTROL no response frequencies miss by {miss_f:.2e} cm^-1");
        assert!(
            miss_f > MUST_MISS_FACTOR * TOL_FREQ,
            "{ctx}: NEGATIVE CONTROL no-response frequencies miss by only {miss_f:.3e}"
        );
    }
}

#[test]
#[ignore = "validation: analytic ROHF Hessian"]
fn rohf_hessian_nh2_matches_pyscf_fd_of_analytic_gradient() {
    check_system("nh2");
}

#[test]
#[ignore = "validation: analytic ROHF Hessian"]
fn rohf_hessian_ch2_triplet_matches_pyscf_fd_of_analytic_gradient() {
    check_system("ch2_triplet");
}

#[test]
#[ignore = "validation: analytic ROHF Hessian"]
fn rohf_hessian_ho2_matches_pyscf_fd_of_analytic_gradient() {
    check_system("ho2");
}
