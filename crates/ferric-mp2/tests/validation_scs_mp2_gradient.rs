//! VALIDATION tier — row "SCS-MP2 analytic gradient".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_scs_mp2_gradient \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's analytic SCS-MP2 nuclear gradient (`scs_mp2_gradient_analytical`:
//! the relaxed RI-MP2 gradient with the pair weight `T̃ = (c_os + c_ss) t −
//! c_ss t̄`, closed-shell RHF reference with exact four-centre J/K, density
//! fitting in the correlation part only, Coulomb metric) against the 5-point
//! central finite difference of PySCF's own DF-MP2 energy components. The
//! energy is `E = E_RHF + c_os E_OS + c_ss E_SS`, linear in the scaling, so
//! the reference stores the FD gradients of `E_RHF`, `E_OS` and `E_SS`
//! separately (`pyscf.gradient_fd.{hf,os,ss,os_fc,ss_fc}`) and the test
//! composes `g = g_hf + c_os g_os + c_ss g_ss` for every scaling it checks:
//! Grimme (6/5, 1/3), SOS (1, 0), the unit scaling (1, 1), and Grimme with
//! the frozen core (O or N 1s).
//!
//! | system | basis | aux (/C) |
//! |---|---|---|
//! | H2O (off C2v) | cc-pVDZ | cc-pvdz-ri |
//! | NH3 (off C3v) | def2-SVP | def2-svp-rifit |
//!
//! PySCF 2.13.1 `pyscf.mp.dfmp2.DFMP2` (`e_corr_os`, `e_corr_ss` tags) on an
//! exact-integral RHF with `with_df.auxmol` built from ferric's aux JSON.
//! PySCF has NO analytic DF-MP2 gradient and its conventional `grad.mp2` has
//! neither density fitting nor SCS scaling, so the reference is an FD of the
//! energy: an independent code and an independent method (Z-vector vs FD).
//! The generator (`scripts/validation/gen_scs_mp2_gradient.py`) refuses to
//! write if halving the step moves any component by more than 5e-9. There is
//! no ORCA reference here. The conventional-integral PySCF gradient is not
//! used: the DF-versus-conventional difference is the RI error, which this
//! test does not bound.
//!
//! # Exactness anchors (asserted first, per system)
//!
//! * nuclear repulsion, AO and aux counts (geometry and basis like-for-like);
//! * E_RHF, `E_OS` and `E_SS` (all electron and frozen core) against PySCF;
//! * ferric's exact-J/K RHF gradient against the FD of PySCF's `E_RHF`;
//! * the unit scaling (1, 1) reproduces ferric's RI-MP2 gradient to 1e-12
//!   (the SCS entry point and the plain one share the intermediates code, so
//!   this pins the entry route and the frozen-core threading, not the
//!   arithmetic; the arithmetic is what the PySCF FD tests).
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * Right: every scaling agrees with the FD to the combined SCF/FD floor
//!   (~1e-8 Ha/Bohr).
//! * A wrong scaling construction (c_os and c_ss swapped, the same-spin part
//!   dropped, the RI-MP2 gradient scaled by `(c_os + c_ss)/2`): the miss is
//!   `|c_os - c_ss| |g_os - g_ss|`-sized, 1e-3 and up on these molecules
//!   (the swapped control below is asserted to exceed [`MUST_MISS`]).
//! * Harness broken: a wrong geometry, basis or aux fails the first anchors.
//!
//! # TOLERANCES
//!
//! [`TOL_G`] 2e-8 Ha/Bohr: the PySCF FD is step-stable to the generator's
//! 5e-9 gate, and ferric's SCF (`density_conv` 1e-9) and Z-vector residual
//! (< 1e-8) enter the analytic side times O(0.1) couplings. The bar is a
//! measured-floor multiple, recorded next to the constant.
//!
//! # NEGATIVE CONTROLS / MUTATIONS
//!
//! * Always on: the swapped scaling `(c_ss, c_os)`, the RI-MP2 gradient
//!   `(1, 1)` and the SOS gradient each miss the Grimme reference by more
//!   than [`MUST_MISS`], so the bar separates the scalings.
//! * Mutations (manual, on the branch the test reaches; recorded in the
//!   commit message): swap `c_os`/`c_ss` in `scs_mp2_gradient_analytical`;
//!   drop the same-spin part of the unrelaxed densities. Both fail the
//!   gradient assertions.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::gradient::{rimp2_gradient_analytical, scs_mp2_gradient_analytical};
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::scs::{scs_mp2, ScsMp2Config};
use ferric_scf::gradient::rhf_gradient_exact_jk;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_testkit::reference::num_ctx;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/scs_mp2_gradient";
const MOL_DIR: &str = "testdata/molecules/validation";

const TOL_ENUC: f64 = 1e-9;
/// E_RHF, E_OS, E_SS vs PySCF (exact J/K, same aux).
const TOL_E: f64 = 1e-9;
/// ferric exact-J/K RHF gradient vs the FD of PySCF's E_RHF.
const TOL_G_RHF: f64 = 1e-8;
/// SCS gradient vs the composed PySCF FD gradient.
const TOL_G: f64 = 2e-8;
/// The unit scaling is the RI-MP2 gradient (same code, same arithmetic).
const TOL_UNIT: f64 = 1e-12;
/// A wrong scaling must miss the reference by more than this (40x the bar).
const MUST_MISS: f64 = 1e-6;

const GRIMME: (f64, f64) = (6.0 / 5.0, 1.0 / 3.0);
const SOS: (f64, f64) = (1.0, 0.0);

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
        .expect("ferric-mp2 manifest dir should be <root>/crates/ferric-mp2")
        .to_path_buf()
}

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_scs_mp2_gradient.py — a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn grad(v: &Value, ptr: &str, natm: usize, ctx: &str) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"));
    assert_eq!(rows.len(), natm, "{ctx}: {ptr} has {} rows", rows.len());
    let mut g = Array2::zeros((natm, 3));
    for (a, row) in rows.iter().enumerate() {
        for k in 0..3 {
            g[(a, k)] = row[k]
                .as_f64()
                .unwrap_or_else(|| panic!("{ctx}: {ptr}[{a}][{k}] not a number"));
        }
    }
    g
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<24} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(d < tol, "{ctx}: {what} |d| {d:.2e} >= {tol:.0e}");
}

fn check_grad(ctx: &str, what: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) {
    let d = max_abs_diff(got, want);
    eprintln!("{ctx}: {what:<24} max|d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} max |d| {d:.2e} >= {tol:.0e}\n ferric {got:?}\n ref {want:?}"
    );
}

fn check_miss(ctx: &str, what: &str, got: &Array2<f64>, want: &Array2<f64>) {
    let d = max_abs_diff(got, want);
    eprintln!("{ctx}: {what:<24} max|d| {d:.2e} (must exceed {MUST_MISS:.0e})");
    assert!(
        d > MUST_MISS,
        "{ctx}: negative control {what}: max |d| {d:.2e} <= {MUST_MISS:.0e}"
    );
}

fn scs_cfg(s: (f64, f64), frozen_core: usize) -> ScsMp2Config {
    ScsMp2Config {
        c_os: s.0,
        c_ss: s.1,
        frozen_core,
        memory_budget_bytes: None,
    }
}

/// `g_hf + c_os g_os + c_ss g_ss` from the stored FD components.
fn composed(r: &Value, natm: usize, ctx: &str, s: (f64, f64), fc: bool) -> Array2<f64> {
    let (os, ss) = if fc { ("os_fc", "ss_fc") } else { ("os", "ss") };
    let g = |k: &str| grad(r, &format!("/pyscf/gradient_fd/{k}"), natm, ctx);
    g("hf") + s.0 * g(os) + s.1 * g(ss)
}

fn check_system(system: &str, basis_name: &str, aux_name: &str) {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    assert_eq!(r["aux_basis"].as_str(), Some(aux_name), "{ctx}: aux basis");
    assert_eq!(r["basis"].as_str(), Some(basis_name), "{ctx}: basis");

    // ---- anchors: geometry, basis, RHF ----
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz(xyz.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let natm = mol.atoms.len();
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num_ctx(&r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let obs_bs = basis::bundled(basis_name).unwrap();
    let aux_bs = basis::bundled(aux_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    assert_eq!(
        obs.nbasis() as u64,
        r["nao"].as_u64().unwrap(),
        "{ctx}: nao"
    );
    assert_eq!(
        dfbs.nbasis() as u64,
        r["naux"].as_u64().unwrap(),
        "{ctx}: naux"
    );
    let n_frozen = r["n_frozen"].as_u64().expect("n_frozen") as usize;

    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };
    assert!(
        cfg.df_j_aux.is_none() && cfg.df_k_aux.is_none(),
        "exact J/K"
    );
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(rhf.converged, "{ctx}: RHF not converged");
    check_close(
        &ctx,
        "E_RHF",
        rhf.energy,
        num_ctx(&r, "/pyscf/e_rhf", &ctx),
        TOL_E,
    );

    // E_OS and E_SS from the scaling endpoints (1, 0) and (0, 1).
    for (fc, os_key, ss_key) in [
        (0, "/pyscf/e_os", "/pyscf/e_ss"),
        (n_frozen, "/pyscf/e_os_fc", "/pyscf/e_ss_fc"),
    ] {
        let e_os = scs_mp2(&mol, &obs, &dfbs, &rhf, &scs_cfg((1.0, 0.0), fc))
            .unwrap()
            .scs_corr;
        let e_ss = scs_mp2(&mol, &obs, &dfbs, &rhf, &scs_cfg((0.0, 1.0), fc))
            .unwrap()
            .scs_corr;
        check_close(
            &ctx,
            &format!("E_OS fc={fc}"),
            e_os,
            num_ctx(&r, os_key, &ctx),
            TOL_E,
        );
        check_close(
            &ctx,
            &format!("E_SS fc={fc}"),
            e_ss,
            num_ctx(&r, ss_key, &ctx),
            TOL_E,
        );
    }

    let g_rhf = rhf_gradient_exact_jk(&mol, &obs, op, &bounds, &rhf, None).unwrap();
    check_grad(
        &ctx,
        "RHF grad vs FD(PySCF)",
        &g_rhf,
        &grad(&r, "/pyscf/gradient_fd/hf", natm, &ctx),
        TOL_G_RHF,
    );

    // ---- SCS gradients ----
    let analytic = |s: (f64, f64), fc: usize| {
        scs_mp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &scs_cfg(s, fc), None)
            .unwrap_or_else(|e| panic!("{ctx}: scs gradient {s:?} fc {fc} failed: {e:?}"))
    };
    let g_grimme = analytic(GRIMME, 0);
    let ref_grimme = composed(&r, natm, &ctx, GRIMME, false);
    check_grad(&ctx, "Grimme vs FD(PySCF)", &g_grimme, &ref_grimme, TOL_G);
    let g_sos = analytic(SOS, 0);
    check_grad(
        &ctx,
        "SOS (1,0) vs FD(PySCF)",
        &g_sos,
        &composed(&r, natm, &ctx, SOS, false),
        TOL_G,
    );
    let g_unit = analytic((1.0, 1.0), 0);
    check_grad(
        &ctx,
        "(1,1) vs FD(PySCF)",
        &g_unit,
        &composed(&r, natm, &ctx, (1.0, 1.0), false),
        TOL_G,
    );
    let g_fc = analytic(GRIMME, n_frozen);
    check_grad(
        &ctx,
        "Grimme fc vs FD(PySCF)",
        &g_fc,
        &composed(&r, natm, &ctx, GRIMME, true),
        TOL_G,
    );

    // ---- anchor: the unit scaling is the RI-MP2 gradient ----
    let mp2_cfg = RiMp2Config::default();
    let g_rimp2 =
        rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &mp2_cfg, None).unwrap();
    check_grad(&ctx, "(1,1) vs RI-MP2", &g_unit, &g_rimp2, TOL_UNIT);

    // ---- negative controls: the bar separates the scalings ----
    let g_swapped = analytic((GRIMME.1, GRIMME.0), 0);
    check_miss(&ctx, "swapped (c_ss, c_os)", &g_swapped, &ref_grimme);
    check_miss(&ctx, "RI-MP2 (1,1)", &g_unit, &ref_grimme);
    check_miss(&ctx, "SOS (1,0)", &g_sos, &ref_grimme);
    check_miss(
        &ctx,
        "Grimme all-e vs fc",
        &g_grimme,
        &composed(&r, natm, &ctx, GRIMME, true),
    );
}

#[test]
#[ignore = "validation: SCS-MP2 gradient (needs testdata/reference/validation/scs_mp2_gradient)"]
fn scs_mp2_gradient_h2o_ccpvdz_vs_pyscf_fd() {
    check_system("h2o_distorted", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: SCS-MP2 gradient (needs testdata/reference/validation/scs_mp2_gradient)"]
fn scs_mp2_gradient_nh3_def2svp_vs_pyscf_fd() {
    check_system("nh3_distorted", "def2-svp", "def2-svp-rifit");
}
