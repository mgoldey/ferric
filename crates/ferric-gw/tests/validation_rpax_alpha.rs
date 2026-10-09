//! VALIDATION tier — VALIDATION.md row "TDHF / RPAx polarizability"
//! (CLI `method.kind = "tdhf-static-polarizability"`).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-gw --release \
//!     --test validation_rpax_alpha --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! `run_rpax_static_polarizability` (the statically screened RPAx kernel,
//! not bare TDHF despite the CLI name) against an INDEPENDENT numpy assembly
//! of the same kernel (`scripts/validation/gen_rpax_alpha.py`) on PySCF
//! ingredients: libcint integrals, PySCF SCF, PySCF's DF tensor with ferric's
//! aux, W(0) = Lᵀ(I + Π(0))⁻¹L by a direct inverse.
//!
//! | case | reference | scissor |
//! |---|---|---|
//! | H2O / aug-cc-pVDZ (aug-cc-pvdz-rifit) | RHF, exact J/K | 0 |
//! | NH3 / aug-cc-pVDZ (aug-cc-pvdz-rifit) | RHF, exact J/K | 0 |
//! | H2O / cc-pVDZ (cc-pvdz-ri) | RKS/PBE, exact J, (75,110) grid | 0.36 Ha |
//!
//! References: `testdata/reference/validation/rpax_alpha/<system>_<basis>_<ref>.json`.
//!
//! # What the two paths share, and what they do not
//!
//! Shared: geometry, the orbital and aux basis JSONs, the DF Coulomb metric
//! (both fit with the same aux, so the screened comparison is DF vs DF), the
//! kernel FORMULAS (the generator was written from the bse.rs loop), and for
//! PBE the grid recipe and the 1e-10 density floor. A conceptual error common
//! to both formulas is therefore invisible to the screened comparison; the
//! bare limit is what ties the convention to an independent response code.
//! Not shared: integral engines, SCF solvers, the W construction (PDEP
//! eigen-decomposition + redressing vs a direct (I + Π)⁻¹), the A±B assembly
//! and the linear solve.
//!
//! # Exactness anchors
//!
//! 1. (generator, asserted before writing) the numpy BARE kernel (W → v) on a
//!    density-fitted RHF equals PySCF DF-CPHF α (`scf.cphf.solve` with
//!    `gen_response(hermi=1)`, same aux) to ≤ 1e-9 relative, and with exact
//!    MO ERIs equals PySCF exact CPHF α to ≤ 1e-9 relative. Measured values
//!    are in each file's `anchors` block.
//! 2. (here) ferric's bare hook (`RpaxScreening::Bare`) at the HF reference
//!    equals the numpy bare α, and the exact-ERI CPHF α to within the DF
//!    fitting gap the generator measured.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! Recorded before measuring in `HYPOTHESES-rpax-static-alpha.md`. The static
//! α depends only on A+B (the code's `(A−B)(A+B) t = (A−B) μ` gives
//! `t = (A+B)⁻¹ μ`), so an A−B defect is invisible to α: it is checked by
//! the lowest eigenvalue of A−B instead.
//!
//! # TOLERANCES (measured 2026-10-05, see the consts)
//!
//! | quantity | measured max | bar |
//! |---|---|---|
//! | screened α, worst element / max|α| | 5.3e-9 | 5e-8 |
//! | screened α_iso, relative | 2.8e-9 | 5e-8 |
//! | bare-hook α vs numpy bare | 5.4e-9 | 5e-8 |
//! | lowest eigenvalue of A+B, A−B | 7.5e-10 Ha | 1e-8 Ha |
//! | E_SCF vs PySCF | 5.2e-12 Ha | 1e-9 Ha |
//! | bare hook vs exact-ERI CPHF (DF fitting gap) | 9.0e-5 (H2O), 1.5e-5 (NH3) | 3x the generator's gap |
//!
//! Generator anchors (asserted at ≤ 1e-9): numpy bare vs PySCF DF-CPHF
//! 3.1e-15 / 7.6e-15, exact-ERI numpy bare vs exact CPHF 5.3e-15 / 1.9e-14
//! (H2O / NH3). PySCF's ITERATIVE `cphf.solve` stops 1.7e-8..5.5e-8 short of
//! the dense CPHF solution even at tol 1e-13, so the anchor uses the dense
//! CPHF matrix built from PySCF's own `gen_response` (recorded beside it).
//!
//! Tensor elements are compared as |d| / max_k |α_k| (the off-diagonal
//! elements vanish by symmetry, so a per-element relative error is undefined).
//!
//! # NEGATIVE CONTROLS (asserted; must exceed 10x the bar = 5e-7)
//!
//! * screened ferric α misses the numpy BARE α: 7.9e-2 (H2O/aug), 8.0e-2
//!   (NH3), 1.6e-1 (PBE) — screening is live;
//! * a truncated PDEP (`trunc_thresh = 1e-2`) misses the full-rank reference
//!   by 7.6e-5;
//! * an aux swap (def2-universal-jkfit) misses the same-aux reference by
//!   3.9e-5;
//! * scissor 0 at PBE: the INDEPENDENT numpy kernel's tensor has
//!   α_xx = −2.680881 (diag −2.68, +14.60, +16.49), and ferric refuses with
//!   the `check_alpha_diagonal_positive` error naming α_xx = −2.680881. The
//!   instability is the kernel's, not an assembly defect.
//!
//! # MUTATIONS (run 2026-10-05; full ledger in HYPOTHESES-rpax-static-alpha.md)
//!
//! * A−B roles swapped (`amb = w_abij − w_ibaj`): α is unchanged (A−B cancels);
//!   killed only by the A−B lowest-eigenvalue check, in all three cases.
//! * `4.0 * coul` → `2.0 * coul`: α off by 0.29 relative; scissor-0 refusal lost.
//! * `w_red` → 0 (W → v): screened α off by 0.086–0.19 relative.
//! * α prefactor 4 → 2: α off by 0.50 relative.
//!
//! # What this does NOT validate
//!
//! The dynamic response and C6 (`run_bse_c6_ks`, ~63% low) and the physical
//! accuracy of the kernel: at PBE + 0.36 Ha scissor water's α is ~46% below
//! DOSD, and the independent numpy kernel gives the same number, so that
//! deficit is the kernel's physics, not an implementation defect. The CLI's
//! defaults (DF-J/K SCF, truncated PDEP) are not the matched settings here.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::bse::{
    rpax_static_polarizability_with_screening, run_rpax_static_polarizability, RpaxScreening,
    RpaxStaticAlphaDetail,
};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/rpax_alpha";
const MOL_DIR: &str = "testdata/molecules/validation";
/// W frequency points (the BSE row's PDEP settings; ω > 0 does not enter W(0)).
const N_QUAD: usize = 100;

// Screened α, ferric vs numpy: worst element |d| / max|α| and the iso
// relative error. Measured 5.3e-9 (H2O/aug-cc-pVDZ@RHF element), 2.3e-9
// (H2O/cc-pVDZ@PBE), 2.1e-9 (NH3); iso 2.8e-9 at worst. Bar ~10x.
const TOL_ALPHA: f64 = 5e-8;
// Bare hook (W → v) vs numpy bare, |d| / max|α|. Measured 5.4e-9 (H2O),
// 2.1e-9 (NH3).
const TOL_ALPHA_BARE: f64 = 5e-8;
// Lowest eigenvalue of A+B and of A−B (Ha). Measured 7.5e-10 (PBE case).
const TOL_MIN_EIG: f64 = 1e-8;
const TOL_E_SCF: f64 = 1e-9;
const TOL_ENUC: f64 = 1e-9;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 10.0;
/// The bare hook vs exact-ERI CPHF may differ by the DF fitting gap the
/// generator measured; allowed up to this multiple of it.
const DF_GAP_FACTOR: f64 = 3.0;

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
        .expect("ferric-gw manifest dir should be <root>/crates/ferric-gw")
        .to_path_buf()
}

/// Load a reference JSON. Missing or unparsable is a HARD failure.
fn reference(system: &str, basis_name: &str, scf_ref: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}_{scf_ref}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_rpax_alpha.py — a missing reference is a failure, never a skip",
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

fn tensor(v: &Value, ptr: &str, ctx: &str) -> [[f64; 3]; 3] {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"));
    assert_eq!(rows.len(), 3, "{ctx}: {ptr} is not 3x3");
    std::array::from_fn(|i| {
        let r = rows[i].as_array().expect("row");
        assert_eq!(r.len(), 3, "{ctx}: {ptr} row {i} is not length 3");
        std::array::from_fn(|j| r[j].as_f64().expect("number"))
    })
}

fn max_abs(t: &[[f64; 3]; 3]) -> f64 {
    t.iter().flatten().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// max_ij |got − want| / max|want|, printing every element first.
fn tensor_rel_diff(ctx: &str, what: &str, got: &[[f64; 3]; 3], want: &[[f64; 3]; 3]) -> f64 {
    let scale = max_abs(want);
    assert!(scale > 1e-3, "{ctx}: {what}: reference tensor is ~0");
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let d = (got[i][j] - want[i][j]).abs() / scale;
            eprintln!(
                "{ctx}: {what}[{i}{j}] ferric {:+.12} ref {:+.12} |d|/max {d:.2e}",
                got[i][j], want[i][j]
            );
            worst = worst.max(d);
        }
    }
    eprintln!("{ctx}: {what} WORST |d|/max|alpha| {worst:.3e}");
    worst
}

fn check_tensor(ctx: &str, what: &str, got: &[[f64; 3]; 3], want: &[[f64; 3]; 3], tol: f64) {
    let worst = tensor_rel_diff(ctx, what, got, want);
    assert!(
        worst < tol,
        "{ctx}: {what}: worst element |d|/max|alpha| {worst:.3e} exceeds {tol:.1e}"
    );
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<28} ferric {got:+.12} ref {want:+.12} |d| {d:.2e}");
    assert!(
        d < tol,
        "{ctx}: {what}: ferric {got:.12} vs reference {want:.12} (|d| {d:.2e}) exceeds {tol:.1e}"
    );
}

fn check_iso(ctx: &str, what: &str, got: f64, want: f64, tol_rel: f64) {
    let d = (got - want).abs() / want.abs();
    eprintln!("{ctx}: {what:<28} ferric {got:+.12} ref {want:+.12} |d|/|ref| {d:.2e}");
    assert!(
        d < tol_rel,
        "{ctx}: {what}: ferric {got:.12} vs reference {want:.12} (rel {d:.2e}) exceeds {tol_rel:.1e}"
    );
}

/// The tensor must MISS `want` by more than MUST_MISS_FACTOR × `tol`.
fn assert_misses(ctx: &str, what: &str, got: &[[f64; 3]; 3], want: &[[f64; 3]; 3], tol: f64) {
    let worst = tensor_rel_diff(ctx, what, got, want);
    eprintln!(
        "{ctx}: control {what}: {worst:.3e}, must exceed {:.1e}",
        MUST_MISS_FACTOR * tol
    );
    assert!(
        worst > MUST_MISS_FACTOR * tol,
        "{ctx}: negative control '{what}' did not miss: {worst:.3e} <= {MUST_MISS_FACTOR} x \
         {tol:.1e} — the comparison does not respond to what the control changes"
    );
}

// ---------------------------------------------------------------------------
// System setup
// ---------------------------------------------------------------------------

struct Sys {
    label: String,
    r: Value,
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    scf: ScfResult,
    scissor: f64,
}

fn rhf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    }
}

fn pbe_config() -> RhfConfig {
    RhfConfig {
        xc: Some("pbe".into()),
        // Exact J, as the reference (an empty name is the explicit opt-out of
        // the RI-J auto-default in rhf.rs `resolve_aux`).
        df_j_aux: Some(String::new()),
        ..rhf_config()
    }
}

fn load_system(system: &str, basis_name: &str, scf_ref: &str) -> Sys {
    let r = reference(system, basis_name, scf_ref);
    let label = format!("{system}/{basis_name}/{scf_ref}");
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
    assert_eq!(
        mol.nelec() as i64,
        r["nelectron"].as_i64().unwrap(),
        "{label}: electron count"
    );
    assert_eq!(
        r["reference"].as_str(),
        Some(scf_ref),
        "{label}: file label"
    );
    let obs_bs = basis::bundled(basis_name).unwrap();
    let aux_name = r["aux"].as_str().expect("aux").to_string();
    let aux_bs = basis::bundled(&aux_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    assert_eq!(
        obs.nbasis() as i64,
        r["nao"].as_i64().unwrap(),
        "{label}: AO count"
    );
    assert_eq!(
        dfbs.nbasis() as i64,
        r["naux"].as_i64().unwrap(),
        "{label}: aux count"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let cfg = match scf_ref {
        "rhf" => rhf_config(),
        "pbe" => pbe_config(),
        other => panic!("unknown reference {other}"),
    };
    let scf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &cfg,
    )
    .unwrap_or_else(|e| panic!("{label}: SCF failed: {e:?}"));
    assert!(scf.converged, "{label}: SCF did not converge");
    // E_SCF FIRST: a harness or SCF difference fails here, not as an α miss.
    check_close(
        &label,
        "E_SCF",
        scf.energy,
        num(&r, "/scf/energy", &label),
        TOL_E_SCF,
    );
    let scissor = num(&r, "/scissor", &label);
    Sys {
        label,
        r,
        mol,
        obs,
        dfbs,
        scf,
        scissor,
    }
}

/// The BSE row's like-for-like PDEP settings: full rank, 100-point GL.
fn pdep_cfg(trunc_thresh: f64) -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh,
        eigensolver: Eigensolver::Lanczos,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: N_QUAD,
            u0: 0.5,
        },
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        ..Default::default()
    }
}

fn run(sys: &Sys, screening: RpaxScreening, trunc: f64) -> RpaxStaticAlphaDetail {
    rpax_static_polarizability_with_screening(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        &sys.scf,
        &pdep_cfg(trunc),
        0,
        sys.scissor,
        screening,
        true,
    )
    .unwrap_or_else(|e| panic!("{}: RPAx ({screening:?}) failed: {e:?}", sys.label))
}

/// Screened α (all 9 elements + iso), the A±B lowest eigenvalues, and the
/// bare-kernel control.
fn check_screened(sys: &Sys, d: &RpaxStaticAlphaDetail) {
    let ctx = format!("{} screened", sys.label);
    let r = &sys.r;
    let want = tensor(r, "/screened/tensor", &ctx);
    check_tensor(&ctx, "alpha", &d.result.tensor, &want, TOL_ALPHA);
    check_iso(
        &ctx,
        "alpha_iso",
        d.result.iso,
        num(r, "/screened/iso", &ctx),
        TOL_ALPHA,
    );
    check_close(
        &ctx,
        "min eig A+B",
        d.min_eig_apb.expect("requested"),
        num(r, "/screened/min_eig_apb", &ctx),
        TOL_MIN_EIG,
    );
    check_close(
        &ctx,
        "min eig A-B",
        d.min_eig_amb.expect("requested"),
        num(r, "/screened/min_eig_amb", &ctx),
        TOL_MIN_EIG,
    );
    // Screening is live: the bare-kernel α is MISSED.
    let bare = tensor(r, "/bare/tensor", &ctx);
    assert_misses(
        &ctx,
        "screened vs numpy BARE",
        &d.result.tensor,
        &bare,
        TOL_ALPHA,
    );
}

/// Bare hook at an HF reference: numpy bare α (same DF), and exact-ERI CPHF
/// within the measured DF fitting gap.
fn check_bare(sys: &Sys, d: &RpaxStaticAlphaDetail) {
    let ctx = format!("{} bare", sys.label);
    let r = &sys.r;
    check_tensor(
        &ctx,
        "alpha",
        &d.result.tensor,
        &tensor(r, "/bare/tensor", &ctx),
        TOL_ALPHA_BARE,
    );
    check_close(
        &ctx,
        "min eig A+B",
        d.min_eig_apb.expect("requested"),
        num(r, "/bare/min_eig_apb", &ctx),
        TOL_MIN_EIG,
    );
    check_close(
        &ctx,
        "min eig A-B",
        d.min_eig_amb.expect("requested"),
        num(r, "/bare/min_eig_amb", &ctx),
        TOL_MIN_EIG,
    );
    let gap = num(
        r,
        "/anchors/df_fitting_gap_bare_df_vs_exact_cphf_max_rel",
        &ctx,
    );
    let cphf = tensor(r, "/anchors/b_exact_cphf_tensor", &ctx);
    let vs_cphf = tensor_rel_diff(&ctx, "alpha vs exact CPHF", &d.result.tensor, &cphf);
    assert!(
        vs_cphf < DF_GAP_FACTOR * gap.max(TOL_ALPHA_BARE),
        "{ctx}: bare hook vs PySCF exact CPHF {vs_cphf:.3e} exceeds {DF_GAP_FACTOR} x the \
         generator's DF fitting gap {gap:.3e}"
    );
    // The anchors were asserted by the generator; re-read them so a file
    // written without them fails here.
    for k in [
        "/anchors/a_df_bare_numpy_vs_pyscf_df_cphf_max_rel",
        "/anchors/b_exact_bare_numpy_vs_pyscf_exact_cphf_max_rel",
    ] {
        let v = num(r, k, &ctx);
        assert!(v <= 1e-9, "{ctx}: generator anchor {k} = {v:.2e} > 1e-9");
    }
}

fn hf_case(system: &str) {
    let sys = load_system(system, "aug-cc-pvdz", "rhf");
    let bare = run(&sys, RpaxScreening::Bare, 0.0);
    check_bare(&sys, &bare);
    let screened = run(&sys, RpaxScreening::StaticPdep, 0.0);
    check_screened(&sys, &screened);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "validation: TDHF / RPAx polarizability"]
fn rpax_alpha_h2o_aug_cc_pvdz_rhf_vs_numpy_and_cphf() {
    hf_case("h2o");
}

#[test]
#[ignore = "validation: TDHF / RPAx polarizability"]
fn rpax_alpha_nh3_aug_cc_pvdz_rhf_vs_numpy_and_cphf() {
    hf_case("nh3");
}

#[test]
#[ignore = "validation: TDHF / RPAx polarizability"]
fn rpax_alpha_h2o_cc_pvdz_pbe_scissor_vs_numpy() {
    let sys = load_system("h2o", "cc-pvdz", "pbe");
    assert!(
        sys.scissor > 0.0,
        "{}: PBE case must carry a scissor",
        sys.label
    );
    let screened = run(&sys, RpaxScreening::StaticPdep, 0.0);
    check_screened(&sys, &screened);
    // The public function is the StaticPdep hook.
    let public = run_rpax_static_polarizability(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        &sys.scf,
        &pdep_cfg(0.0),
        0,
        sys.scissor,
    )
    .expect("public run");
    assert_eq!(
        public.tensor, screened.result.tensor,
        "{}: public function differs from the StaticPdep hook",
        sys.label
    );
}

/// Negative control: a truncated PDEP W misses the full-rank reference.
#[test]
#[ignore = "validation: TDHF / RPAx polarizability"]
fn rpax_alpha_truncated_pdep_misses_full_rank_reference() {
    let sys = load_system("h2o", "cc-pvdz", "pbe");
    let d = run(&sys, RpaxScreening::StaticPdep, 1e-2);
    let ctx = format!("{} trunc 1e-2", sys.label);
    let want = tensor(&sys.r, "/screened/tensor", &ctx);
    assert_misses(
        &ctx,
        "truncated vs full rank",
        &d.result.tensor,
        &want,
        TOL_ALPHA,
    );
}

/// Negative control: the wrong aux misses the same-aux reference.
#[test]
#[ignore = "validation: TDHF / RPAx polarizability"]
fn rpax_alpha_aux_swap_misses_same_aux_reference() {
    let mut sys = load_system("h2o", "aug-cc-pvdz", "rhf");
    sys.dfbs =
        PreparedBasis::new(&sys.mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let d = run(&sys, RpaxScreening::StaticPdep, 0.0);
    let ctx = format!("{} aux def2-universal-jkfit", sys.label);
    let want = tensor(&sys.r, "/screened/tensor", &ctx);
    assert_misses(&ctx, "aux swap", &d.result.tensor, &want, TOL_ALPHA);
}

/// Scissor 0 at PBE: the documented excitonic instability. The numpy kernel
/// records whether ITS α diagonal is negative there; ferric must refuse
/// exactly when the independent kernel says the tensor is unphysical, and
/// match it otherwise.
#[test]
#[ignore = "validation: TDHF / RPAx polarizability"]
fn rpax_pbe_scissor0_matches_numpy_sign() {
    let sys = load_system("h2o", "cc-pvdz", "pbe");
    let ctx = format!("{} scissor 0", sys.label);
    let want = tensor(&sys.r, "/scissor0/tensor", &ctx);
    let negative = (0..3).any(|d| !(want[d][d] > 1e-8));
    eprintln!(
        "{ctx}: numpy scissor-0 diagonal ({:+.6}, {:+.6}, {:+.6})",
        want[0][0], want[1][1], want[2][2]
    );
    let res = run_rpax_static_polarizability(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        &sys.scf,
        &pdep_cfg(0.0),
        0,
        0.0,
    );
    match (negative, res) {
        (true, Err(e)) => {
            let msg = e.to_string();
            eprintln!("{ctx}: refused as the numpy kernel predicts: {msg}");
            assert!(
                msg.contains("unphysical static polarizability"),
                "{ctx}: errored, but not via the alpha-diagonal guard: {msg}"
            );
        }
        (true, Ok(r)) => panic!(
            "{ctx}: numpy kernel has a non-positive diagonal but ferric returned {:?}",
            r.tensor
        ),
        (false, Err(e)) => panic!("{ctx}: numpy kernel is positive but ferric refused: {e}"),
        (false, Ok(r)) => check_tensor(&ctx, "alpha", &r.tensor, &want, TOL_ALPHA),
    }
}
