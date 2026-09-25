//! VALIDATION tier — VALIDATION.md row "RPA gradient" (closed-shell
//! PDEP-RPA nuclear gradient, `ferric_rpa::gradient`).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-rpa --release \
//!     --test validation_rpa_gradient --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What gradient.rs computes (read from the code, not its doc comments)
//!
//! `total_rpa_gradient(mol, obs, aux, op, cfg, h)` returns `(E_tot, g)`:
//! `E_tot = E_RHF + E_c^RPA` at the reference geometry, and
//! `g[A,k] = (E(R + h e_Ak) − E(R − h e_Ak)) / 2h`, a 3-point central finite
//! difference of the TOTAL energy, every point a fresh exact-J/K `solve_rhf`
//! (density_conv 1e-9) plus a full `run_pdep_rpa` with the caller's config.
//! So: RHF reference orbitals of the displaced geometry, frozen core and
//! quadrature from `cfg`, no Ritz basis held fixed. It is not analytic, and
//! `rpa_correlation_gradient` returns dE_total/dR despite its name, which the
//! corr-only control below pins.
//!
//! # What is compared
//!
//! | test | anchor |
//! |---|---|
//! | `*_energy_vs_pyscf` | E_RHF, E_c (40-point Gauss–Legendre, full rank) vs PySCF exact-integral `RHF` + `gw.rpa.RPA` |
//! | `*_gradient_vs_pyscf_fd` | g vs the 3-point FD of PySCF's total energy at the SAME h (like-for-like) and vs the step-converged 5-point FD (h = 2e-3, agrees with h = 1e-3 to 3.9e-10/7.4e-10); translation invariance |
//! | `*_gradient_vs_own_fd` | g vs a 5-point FD (h = 1e-3) of ferric's own E_RHF + E_c written in this file — no external code |
//! | `h2o_frozen_core_*` | frozen_core = 1 vs PySCF `RPA(frozen=1)` FD |
//! | `h2o_quadrature_control` | 6-point grid matches PySCF nw = 6 FD and misses the 40-point one |
//!
//! Systems: distorted H2O and NH3 (no symmetry; every component ≥ 4.9e-4
//! Ha/Bohr) / cc-pVDZ (cc-pvdz-ri). References: `scripts/validation/
//! gen_rpa_gradient.py` → `testdata/reference/validation/rpa_gradient/`.
//! The quadrature grids are MATCHED, not converged: PySCF's
//! `_get_scaled_legendre_roots(nw, 0.5)` is ferric's `GaussLegendre` with
//! u0 = 0.5 (U-RPA row).
//!
//! # Physics vs artifact hypotheses
//!
//! * If the gradient is right: it equals PySCF's same-stencil FD to SCF noise
//!   and the 5-point FD to the 3-point truncation (3.3e-8 H2O, 5.6e-8 NH3,
//!   measured on PySCF's surface).
//! * If it differentiates only E_c (as its doc says) the corr-only control
//!   matches instead (misses the total by 3.1e-2 / 2.6e-2).
//! * If it is the RHF gradient (RPA dropped) the RHF-analytic control matches
//!   (misses by 2.1e-2 / 1.5e-2).
//! * If `cfg.quadrature` / `cfg.frozen_core` did not reach the displaced
//!   energies, the coarse / frozen-core runs would match the 40-point
//!   all-electron reference instead (differences 3.1e-4, 7.1e-4).
//! * HARNESS errors (geometry, basis, aux) fail E_nuc / nao / naux / E_RHF first.
//!
//! # TOLERANCES
//!
//! Each bar is 3–25x the worst |d| measured over both systems, noted beside it.
//!
//! # NEGATIVE CONTROLS (asserted)
//!
//! RHF analytic gradient, corr-only FD, all-electron FD (for the frozen-core
//! run), 40-point FD (for the 6-point run): each must be MISSED by
//! MUST_MISS_FACTOR × the bar it would otherwise pass.
//!
//! # MUTATION (run 2026-09-25)
//!
//! `crates/ferric-rpa/src/gradient.rs`, `rpa_total_energy`'s
//! `Ok(rhf.energy + r.e_rpa)` → `Ok(r.e_rpa)` (differentiate E_c only) fails
//! all six gradient tests (vs PySCF FD, vs own FD, frozen core, quadrature);
//! the two `*_energy_vs_pyscf` tests pass, as they do not go through it.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_rpa::gradient::total_rpa_gradient;
use ferric_rpa::run_pdep_rpa;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/rpa_gradient";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "cc-pvdz";
const N_QUAD: usize = 40;
const N_QUAD_COARSE: usize = 6;
const U0: f64 = 0.5;
/// Step handed to gradient.rs (3-point stencil); the reference's `fd3_h0.0005`.
const H_LIB: f64 = 5e-4;
/// Step of the in-test 5-point FD of ferric's own energy.
const H_OWN: f64 = 1e-3;
const FD5_PRIMARY: &str = "fd5_h0.002";
const FD3_LIB: &str = "fd3_h0.0005";

const TOL_ENUC: f64 = 1e-9;
// Measured worst |d| (2026-09-25) beside each bar.
// E_RHF: 1.7e-12.
const TOL_E_RHF: f64 = 1e-11;
// E_c (40- and 6-point): 1.5e-13.
const TOL_E_C: f64 = 1e-12;
// E_tot from total_rpa_gradient's own SCF: 2.3e-12.
const TOL_E_TOT_LIB: f64 = 2e-11;
// Same 3-point stencil and h on both surfaces: 1.9e-9 Ha/Bohr.
const TOL_G_SAME_STENCIL: f64 = 2e-8;
// vs PySCF 5-point: 5.7e-8 Ha/Bohr, the 3-point truncation at h = 5e-4.
const TOL_G_FD5: f64 = 2e-7;
// vs ferric's own 5-point FD: 5.6e-8 Ha/Bohr, the same truncation.
const TOL_G_OWN_FD: f64 = 2e-7;
// Sum over atoms: 4.8e-9 Ha/Bohr.
const TOL_TRANS: f64 = 5e-8;
const MUST_MISS_FACTOR: f64 = 10.0;

// ---------------------------------------------------------------------------
// Reference plumbing
// ---------------------------------------------------------------------------

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
        .expect("ferric-rpa manifest dir should be <root>/crates/ferric-rpa")
        .to_path_buf()
}

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_rpa_gradient.py — a missing reference is a failure, never a skip",
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

fn grad_at(v: &Value, ptr: &str, natm: usize, ctx: &str) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference gradient {ptr} missing"));
    assert_eq!(rows.len(), natm, "{ctx}: {ptr} atom count");
    let mut g = Array2::zeros((natm, 3));
    for (a, row) in rows.iter().enumerate() {
        let row = row.as_array().expect("gradient row");
        assert_eq!(row.len(), 3);
        for k in 0..3 {
            g[(a, k)] = row[k].as_f64().expect("gradient entry");
        }
    }
    g
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<30} ferric {got:+.12} ref {want:+.12} |d| {d:.2e}");
    assert!(
        d < tol,
        "{ctx}: {what}: ferric {got:.12} vs reference {want:.12} (|d| {d:.2e}) exceeds {tol:.1e}"
    );
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> (f64, (usize, usize)) {
    let mut worst = (0.0, (0, 0));
    for ((i, j), x) in a.indexed_iter() {
        let d = (x - b[(i, j)]).abs();
        if d > worst.0 {
            worst = (d, (i, j));
        }
    }
    worst
}

fn check_grad(ctx: &str, what: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) {
    for ((a, k), x) in got.indexed_iter() {
        eprintln!(
            "{ctx}: {what} atom {a} coord {k}: ferric {x:+.10} ref {:+.10} |d| {:.2e}",
            want[(a, k)],
            (x - want[(a, k)]).abs()
        );
    }
    let (d, (a, k)) = max_abs_diff(got, want);
    eprintln!("{ctx}: {what}: max |d| {d:.3e} Ha/Bohr (atom {a} coord {k}), bar {tol:.1e}");
    assert!(
        d < tol,
        "{ctx}: {what}: max |d| {d:.3e} Ha/Bohr at atom {a} coord {k} exceeds {tol:.1e}"
    );
}

fn assert_grad_misses(ctx: &str, what: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) {
    let (d, (a, k)) = max_abs_diff(got, want);
    eprintln!(
        "{ctx}: control {what}: max |d| {d:.3e} (atom {a} coord {k}), must exceed {:.1e}",
        MUST_MISS_FACTOR * tol
    );
    assert!(
        d > MUST_MISS_FACTOR * tol,
        "{ctx}: negative control '{what}' did not miss: max |d| {d:.3e} <= {MUST_MISS_FACTOR} x {tol:.1e}"
    );
}

// ---------------------------------------------------------------------------
// System setup
// ---------------------------------------------------------------------------

struct Sys {
    label: String,
    r: Value,
    mol: Molecule,
    obs_bs: BasisSet,
    aux_bs: BasisSet,
}

fn load_system(system: &str) -> Sys {
    let r = reference(system);
    let label = format!("{system}/{BASIS}");
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check_close(
        &label,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &label),
        TOL_ENUC,
    );
    let obs_bs = basis::bundled(BASIS).unwrap();
    let aux_name = r["aux"].as_str().expect("aux");
    let aux_bs = basis::bundled(aux_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    assert_eq!(
        obs.nbasis() as i64,
        r["nao"].as_i64().unwrap(),
        "{label}: nao"
    );
    assert_eq!(
        dfbs.nbasis() as i64,
        r["naux"].as_i64().unwrap(),
        "{label}: naux"
    );
    Sys {
        label,
        r,
        mol,
        obs_bs,
        aux_bs,
    }
}

/// Full rank (trunc 0: the Lanczos arm is one dense eigh keeping every mode),
/// Gauss–Legendre with PySCF's map.
fn rpa_cfg(n_quad: usize, frozen_core: usize) -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core,
        trunc_thresh: 0.0,
        eigensolver: Eigensolver::Lanczos,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: n_quad,
            u0: U0,
        },
        ..Default::default()
    }
}

fn tight_rhf() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    }
}

/// ferric's E_RHF and E_c at `mol`, written here independently of gradient.rs
/// (tight SCF). Asserts full rank.
fn own_energy(sys: &Sys, mol: &Molecule, cfg: &PdepRpaConfig) -> (f64, f64) {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(mol, &sys.obs_bs).unwrap();
    let dfbs = PreparedBasis::new(mol, &sys.aux_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, mol, &obs, op, &bounds, &tight_rhf())
        .unwrap_or_else(|e| panic!("{}: RHF failed: {e:?}", sys.label));
    assert!(rhf.converged, "{}: RHF did not converge", sys.label);
    let r = run_pdep_rpa(mol, &obs, &dfbs, op, &rhf, cfg)
        .unwrap_or_else(|e| panic!("{}: run_pdep_rpa failed: {e:?}", sys.label));
    // Full rank means every mode chi0 can reach: rank(chi0) = min(naux, n_ov).
    // With a frozen core n_ov can fall below naux (H2O/cc-pVDZ fc1: 4 x 19 = 76
    // < 84); the extra naux - n_ov eigenvalues are exactly 1, contribute
    // ln(1) - (1 - 1) = 0 to E_c, and may be dropped at the cut.
    let nocc_act = (mol.nelec() / 2) as usize - cfg.frozen_core;
    let n_ov = nocc_act * (rhf.mos_r().ncols() - (mol.nelec() / 2) as usize);
    let reachable = dfbs.nbasis().min(n_ov);
    assert!(
        r.n_eigenpotentials >= reachable && r.n_eigenpotentials <= dfbs.nbasis(),
        "{}: trunc_thresh 0 must keep every reachable dielectric mode: kept {} of \
         min(naux {}, n_ov {n_ov}) = {reachable}",
        sys.label,
        r.n_eigenpotentials,
        dfbs.nbasis()
    );
    assert_eq!(r.quad_freqs.len(), cfg.quadrature.n_points);
    (rhf.energy, r.e_rpa)
}

fn displaced(mol: &Molecule, atom: usize, k: usize, d: f64) -> Molecule {
    let mut m = mol.clone();
    match k {
        0 => m.atoms[atom].x += d,
        1 => m.atoms[atom].y += d,
        _ => m.atoms[atom].zpos += d,
    }
    m
}

/// 5-point central FD of ferric's own E_RHF + E_c, step `H_OWN`.
fn own_fd5(sys: &Sys, cfg: &PdepRpaConfig) -> Array2<f64> {
    let natm = sys.mol.atoms.len();
    let stencil = [
        (-2.0, 1.0 / 12.0),
        (-1.0, -8.0 / 12.0),
        (1.0, 8.0 / 12.0),
        (2.0, -1.0 / 12.0),
    ];
    let mut g = Array2::zeros((natm, 3));
    for a in 0..natm {
        for k in 0..3 {
            let mut acc = 0.0;
            for (s, w) in stencil {
                let (e_hf, e_c) = own_energy(sys, &displaced(&sys.mol, a, k, s * H_OWN), cfg);
                acc += w * (e_hf + e_c);
            }
            g[(a, k)] = acc / H_OWN;
        }
    }
    g
}

fn lib_gradient(sys: &Sys, cfg: &PdepRpaConfig) -> (f64, Array2<f64>) {
    total_rpa_gradient(
        &sys.mol,
        &sys.obs_bs,
        &sys.aux_bs,
        Operator::coulomb(),
        cfg,
        H_LIB,
    )
    .unwrap_or_else(|e| panic!("{}: total_rpa_gradient failed: {e:?}", sys.label))
}

fn translation_invariance(ctx: &str, g: &Array2<f64>) {
    for k in 0..3 {
        let s: f64 = g.column(k).sum();
        eprintln!("{ctx}: sum over atoms, coord {k}: {s:+.3e}");
        assert!(
            s.abs() < TOL_TRANS,
            "{ctx}: translation invariance, coord {k}: sum {s:.3e} exceeds {TOL_TRANS:.1e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

fn energy_case(system: &str) {
    let sys = load_system(system);
    let ctx = sys.label.clone();
    let (e_hf, e_c) = own_energy(&sys, &sys.mol, &rpa_cfg(N_QUAD, 0));
    check_close(
        &ctx,
        "E_RHF",
        e_hf,
        num(&sys.r, "/energy/e_rhf", &ctx),
        TOL_E_RHF,
    );
    check_close(
        &ctx,
        "E_c (40-pt GL)",
        e_c,
        num(&sys.r, "/energy/e_c", &ctx),
        TOL_E_C,
    );
    let (_, e_c6) = own_energy(&sys, &sys.mol, &rpa_cfg(N_QUAD_COARSE, 0));
    check_close(
        &ctx,
        "E_c (6-pt GL)",
        e_c6,
        num(&sys.r, "/energy/e_c_coarse", &ctx),
        TOL_E_C,
    );
    if let Some(nc) = sys.r["frozen_core_block"].as_u64() {
        let (_, e_fc) = own_energy(&sys, &sys.mol, &rpa_cfg(N_QUAD, nc as usize));
        check_close(
            &ctx,
            "E_c (frozen core)",
            e_fc,
            num(&sys.r, "/energy/e_c_fc", &ctx),
            TOL_E_C,
        );
    }
}

fn gradient_vs_pyscf_case(system: &str) {
    let sys = load_system(system);
    let ctx = sys.label.clone();
    let natm = sys.mol.atoms.len();
    let (e_tot, g) = lib_gradient(&sys, &rpa_cfg(N_QUAD, 0));
    check_close(
        &ctx,
        "E_tot (total_rpa_gradient)",
        e_tot,
        num(&sys.r, "/energy/e_total", &ctx),
        TOL_E_TOT_LIB,
    );
    let fd = |name: &str, block: &str| {
        grad_at(&sys.r, &format!("/gradient/fd/{name}/{block}"), natm, &ctx)
    };
    check_grad(
        &ctx,
        "vs PySCF 3-pt FD same h",
        &g,
        &fd(FD3_LIB, "total"),
        TOL_G_SAME_STENCIL,
    );
    check_grad(
        &ctx,
        "vs PySCF 5-pt FD",
        &g,
        &fd(FD5_PRIMARY, "total"),
        TOL_G_FD5,
    );
    translation_invariance(&ctx, &g);
    // Controls: the reference distinguishes the two wrong quantities a total
    // gradient could silently be.
    assert_grad_misses(
        &ctx,
        "PySCF analytic RHF gradient",
        &g,
        &grad_at(&sys.r, "/gradient/rhf_analytic", natm, &ctx),
        TOL_G_FD5,
    );
    assert_grad_misses(
        &ctx,
        "PySCF FD of E_c alone",
        &g,
        &fd(FD5_PRIMARY, "corr_only"),
        TOL_G_FD5,
    );
}

fn gradient_vs_own_fd_case(system: &str) {
    let sys = load_system(system);
    let ctx = sys.label.clone();
    let cfg = rpa_cfg(N_QUAD, 0);
    let (_, g) = lib_gradient(&sys, &cfg);
    let g_own = own_fd5(&sys, &cfg);
    check_grad(&ctx, "vs own 5-pt FD", &g, &g_own, TOL_G_OWN_FD);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "validation: RPA gradient"]
fn rpa_gradient_h2o_energy_vs_pyscf() {
    energy_case("h2o_distorted");
}

#[test]
#[ignore = "validation: RPA gradient"]
fn rpa_gradient_nh3_energy_vs_pyscf() {
    energy_case("nh3_distorted");
}

#[test]
#[ignore = "validation: RPA gradient"]
fn rpa_gradient_h2o_gradient_vs_pyscf_fd() {
    gradient_vs_pyscf_case("h2o_distorted");
}

#[test]
#[ignore = "validation: RPA gradient"]
fn rpa_gradient_nh3_gradient_vs_pyscf_fd() {
    gradient_vs_pyscf_case("nh3_distorted");
}

#[test]
#[ignore = "validation: RPA gradient"]
fn rpa_gradient_h2o_gradient_vs_own_fd() {
    gradient_vs_own_fd_case("h2o_distorted");
}

#[test]
#[ignore = "validation: RPA gradient"]
fn rpa_gradient_nh3_gradient_vs_own_fd() {
    gradient_vs_own_fd_case("nh3_distorted");
}

/// frozen_core = 1 reaches every displaced energy: matches PySCF
/// `RPA(frozen=1)` FD and misses the all-electron one.
#[test]
#[ignore = "validation: RPA gradient"]
fn rpa_gradient_h2o_frozen_core_vs_pyscf_fd() {
    let sys = load_system("h2o_distorted");
    let ctx = format!("{} fc", sys.label);
    let natm = sys.mol.atoms.len();
    let nc = sys.r["frozen_core_block"]
        .as_u64()
        .expect("frozen_core_block") as usize;
    let (_, g) = lib_gradient(&sys, &rpa_cfg(N_QUAD, nc));
    let fd = |name: &str, block: &str| {
        grad_at(&sys.r, &format!("/gradient/fd/{name}/{block}"), natm, &ctx)
    };
    check_grad(
        &ctx,
        "vs PySCF 3-pt FD same h",
        &g,
        &fd(FD3_LIB, "total_fc"),
        TOL_G_SAME_STENCIL,
    );
    check_grad(
        &ctx,
        "vs PySCF 5-pt FD",
        &g,
        &fd(FD5_PRIMARY, "total_fc"),
        TOL_G_FD5,
    );
    assert_grad_misses(
        &ctx,
        "all-electron FD",
        &g,
        &fd(FD5_PRIMARY, "total"),
        TOL_G_FD5,
    );
}

/// A 6-point grid reaches every displaced energy: matches PySCF nw = 6 FD
/// (same map) and misses the 40-point one.
#[test]
#[ignore = "validation: RPA gradient"]
fn rpa_gradient_h2o_quadrature_control() {
    let sys = load_system("h2o_distorted");
    let ctx = format!("{} nw={N_QUAD_COARSE}", sys.label);
    let natm = sys.mol.atoms.len();
    assert_eq!(sys.r["nw_coarse"].as_u64(), Some(N_QUAD_COARSE as u64));
    let (_, g) = lib_gradient(&sys, &rpa_cfg(N_QUAD_COARSE, 0));
    let fd = |name: &str, block: &str| {
        grad_at(&sys.r, &format!("/gradient/fd/{name}/{block}"), natm, &ctx)
    };
    check_grad(
        &ctx,
        "vs PySCF 3-pt FD same h",
        &g,
        &fd(FD3_LIB, "total_coarse"),
        TOL_G_SAME_STENCIL,
    );
    assert_grad_misses(
        &ctx,
        "40-point FD",
        &g,
        &fd(FD5_PRIMARY, "total"),
        TOL_G_FD5,
    );
}
