//! VALIDATION tier — row "ROHF-referenced U-RPA / U-G0W0 / U-MP2".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-gw --release \
//!     --test validation_rohf_reference --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's unrestricted correlated methods handed a RAW ROHF result
//! (`solve_rohf`, `Spin::RestrictedOpen`) — `run_u_pdep_rpa`, `run_u_gw` (G0W0),
//! `u_ri_mp2` — against PySCF 2.13.1 on the SEMI-CANONICALIZED ROHF orbitals:
//! per spin σ, the ROHF MOs rotated inside the σ-occupied and σ-virtual blocks to
//! diagonalize `F_σ = h + J[D] − K[D_σ]`, orbital energies = the block eigenvalues.
//! OH and CH3 doublets / cc-pVDZ (aux cc-pvdz-ri), exact-integral ROHF on both
//! sides; the test asserts E_ROHF and every semi-canonical orbital energy before
//! any correlated number.
//!
//! References: `scripts/validation/gen_rohf_semicanonical.py` →
//! `testdata/reference/validation/rohf_semicanonical/<system>_cc-pvdz.json`
//! (U-RPA with gen_urpa.py's recipe, U-G0W0 with gen_gw.py's, U-MP2 doubles in
//! numpy on URPA's Cholesky ovL, cross-checked against PySCF DFUMP2 to ≤2.8e-17).
//!
//! # U-MP2: doubles only, NOT ROMP2
//!
//! ROHF is not a UHF stationary point: each spin's `f_ia` is non-zero (max |f_ov|
//! 5.5e-2 Ha on OH, 3.2e-2 on CH3). ROMP2 adds the singles term
//! `Σ_σ Σ_ia |f_ia^σ|² / (ε_i − ε_a)` (−2.72e-3 Ha OH, −2.43e-3 CH3, recorded as
//! `romp2_singles`). `u_ri_mp2` computes the doubles-only UMP2 expression on the
//! semi-canonical orbitals, so that is what is compared; the test asserts ferric
//! MISSES the ROMP2 total, so a silently-added singles term would fail.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric semi-canonicalizes correctly: same orbitals and energies on both
//!   sides; agreement at the SCF-convergence / engine level (like the UHF rows).
//! * If the semi-canonicalization is skipped (the pre-fix code path): ferric
//!   reproduces the `legacy_effective_fock` control instead — ROHF MOs with the
//!   Roothaan EFFECTIVE Fock's eigenvalues for BOTH spins. That control is
//!   4.7e-3 (MP2) / 6.1e-3 (RPA) Ha away on OH and moves the α-HOMO QP by ~9 eV.
//! * If only one spin were semi-canonicalized, or β used α's Fock: the
//!   per-spin orbital-energy check fails before any correlated number.
//! * If the harness is wrong (geometry, basis, aux, a different ROHF state):
//!   E_nuc, AO/aux counts or E_ROHF fail first.
//!
//! # TOLERANCES
//!
//! Each bar is 3–25× the worst |d| measured over OH and CH3 (2026-09-25),
//! noted beside it.
//!
//! # NEGATIVE CONTROLS (asserted)
//!
//! * Legacy: ferric on the raw ROHF must MISS `legacy_effective_fock` (RPA E_c,
//!   MP2 E_c, QP window) by ≥10× the bar.
//! * ROMP2: ferric's U-MP2 must MISS `semicanonical/romp2_total`.
//! * Legacy reproduction (`legacy_view_reproduces_*`): ferric fed the legacy view
//!   explicitly (ROHF MOs + effective-Fock energies for both spins, labelled
//!   Unrestricted) must MATCH the legacy control — the control is the old code
//!   path, and the harness sees it.
//!
//! # MUTATION (run 2026-09-25)
//!
//! `crates/ferric-scf/src/semicanonical.rs`, `semicanonicalize_from_spin_focks`:
//! semicanonicalize β with the α spin Fock (`semicanonicalize_spin(c, f_a,
//! nocc_b)`). Both `*_vs_pyscf_semicanonical` tests fail; the two legacy tests,
//! which build the legacy view themselves, pass. Returning the ROHF result
//! unchanged (`Ok(Cow::Borrowed(scf))`) panics instead (`eps_b()`/`mos_b()` on a
//! RestrictedOpen result), because the consumers' old β fallbacks are gone.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_u_gw, GwConfig, GwMethod, UGwResult};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::u_rimp2::u_ri_mp2;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_rpa::run_u_pdep_rpa;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::semicanonical::unrestricted_reference;
use ferric_scf::{ScfResult, Spin};
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/rohf_semicanonical";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "cc-pvdz";
const HA_TO_EV: f64 = 27.211_386_245_988;
/// U-RPA frequency points (PySCF nw = 40, same GL map, x0 = u0 = 0.5).
const N_QUAD_RPA: usize = 40;
/// U-G0W0 W-integral points (PySCF nw = 100, gen_gw.py's recipe).
const N_QUAD_GW: usize = 100;

// E_ROHF: 3.6e-13.
const TOL_E_SCF: f64 = 5e-12;
// Semi-canonical orbital energies, every MO both spins: 6.4e-10.
const TOL_EPS: f64 = 5e-9;
// U-RPA E_c: 7.3e-12.
const TOL_E_RPA: f64 = 1e-10;
// U-MP2 E_c (doubles): 8.7e-12.
const TOL_E_MP2: f64 = 1e-10;
// U-G0W0 QP energies: 1.5e-5 (OH beta; OH's degenerate pi hole, as in the
// U-G0W0@UHF row); CH3 <= 1.3e-6.
const TOL_QP: f64 = 5e-5;
const TOL_ENUC: f64 = 1e-9;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
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
        .expect("ferric-gw manifest dir should be <root>/crates/ferric-gw")
        .to_path_buf()
}

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_rohf_semicanonical.py — a missing reference is a failure, \
             never a skip",
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

fn vec_usize(v: &Value, ptr: &str, ctx: &str) -> Vec<usize> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|x| x.as_u64().expect("index") as usize)
        .collect()
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<30} ferric {got:+.12} ref {want:+.12} |d| {d:.2e}");
    assert!(
        d < tol,
        "{ctx}: {what}: ferric {got:.12} vs reference {want:.12} (|d| {d:.2e} Ha) exceeds {tol:.1e} Ha"
    );
}

fn max_dev(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0_f64, f64::max)
}

fn check_vec(ctx: &str, what: &str, got: &[f64], want: &[f64], tol: f64) {
    let d = max_dev(got, want);
    eprintln!(
        "{ctx}: {what:<30} max |d| {d:.3e} Ha ({:.4} meV) over {} values",
        d * HA_TO_EV * 1e3,
        got.len()
    );
    assert!(
        d.is_finite() && d < tol,
        "{ctx}: {what}: max |d| {d:.3e} Ha exceeds {tol:.1e} Ha\n  ferric {got:?}\n  ref    {want:?}"
    );
}

fn assert_misses(ctx: &str, what: &str, got: &[f64], want: &[f64], tol: f64) {
    let d = max_dev(got, want);
    eprintln!(
        "{ctx}: control {what}: max |d| {d:.3e} Ha, must exceed {:.1e}",
        MUST_MISS_FACTOR * tol
    );
    assert!(
        d > MUST_MISS_FACTOR * tol,
        "{ctx}: negative control '{what}' did not miss: max |d| {d:.3e} Ha <= {MUST_MISS_FACTOR} \
         x {tol:.1e} — the comparison does not respond to what the control changes"
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
    rohf: ScfResult,
}

fn load(system: &str) -> Sys {
    let r = reference(system);
    let label = format!("{system}/{BASIS}");
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check_close(
        &label,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &label),
        TOL_ENUC,
    );
    let obs = PreparedBasis::new(&mol, &basis::bundled(BASIS).unwrap()).unwrap();
    let aux_name = r["aux"].as_str().expect("aux").to_string();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(&aux_name).unwrap()).unwrap();
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
    let rohf = solve_rohf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig {
            max_iter: 400,
            energy_conv: 1e-11,
            density_conv: 1e-9,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("{label}: solve_rohf failed: {e:?}"));
    assert!(rohf.converged, "{label}: ROHF did not converge");
    assert_eq!(rohf.spin, Spin::RestrictedOpen);
    check_close(
        &label,
        "E_ROHF",
        rohf.energy,
        num(&r, "/rohf/energy", &label),
        TOL_E_SCF,
    );
    Sys {
        label,
        r,
        mol,
        obs,
        dfbs,
        rohf,
    }
}

/// The pre-fix treatment of a ROHF reference: ROHF MOs and the effective-Fock
/// eigenvalues for BOTH spins, labelled Unrestricted so the methods take them as given.
fn legacy_view(rohf: &ScfResult) -> ScfResult {
    let mut v = rohf.clone();
    v.spin = Spin::Unrestricted;
    v.mos_beta = Some(rohf.mos_alpha.clone());
    v.eps_beta = Some(rohf.eps_alpha.clone());
    v
}

fn pdep_cfg(n_quad: usize) -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        eigensolver: Eigensolver::Lanczos,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: n_quad,
            u0: 0.5,
        },
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        ..Default::default()
    }
}

fn rpa(sys: &Sys, scf: &ScfResult) -> f64 {
    let res = run_u_pdep_rpa(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        scf,
        &pdep_cfg(N_QUAD_RPA),
    )
    .unwrap_or_else(|e| panic!("{}: run_u_pdep_rpa failed: {e:?}", sys.label));
    // trunc_thresh = 0 drops only modes with lambda = 1 exactly, which add
    // nothing to E_c (ln 1 - (1 - 1) = 0). OH and CH3 each have one such mode
    // (69/70 and 97/98 kept); E_c itself is compared below.
    assert!(
        res.n_eigenpotentials + 2 >= sys.dfbs.nbasis(),
        "{}: trunc_thresh = 0 kept only {} of {} dielectric modes",
        sys.label,
        res.n_eigenpotentials,
        sys.dfbs.nbasis()
    );
    res.e_rpa
}

fn mp2(sys: &Sys, scf: &ScfResult) -> f64 {
    u_ri_mp2(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        scf,
        &RiMp2Config::default(),
    )
    .unwrap_or_else(|e| panic!("{}: u_ri_mp2 failed: {e:?}", sys.label))
    .mp2_corr
}

fn gw(sys: &Sys, scf: &ScfResult, orbs: &[usize]) -> UGwResult {
    let qp = orbs[0]..orbs[orbs.len() - 1] + 1;
    assert_eq!(
        qp.len(),
        orbs.len(),
        "{}: QP window not contiguous",
        sys.label
    );
    let res = run_u_gw(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        scf,
        &pdep_cfg(N_QUAD_GW),
        &GwConfig {
            method: GwMethod::G0W0,
            qp_mos: Some(qp),
            frozen_core: 0,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("{}: run_u_gw failed: {e:?}", sys.label));
    assert_eq!(res.mo_indices, orbs, "{}: QP window", sys.label);
    assert!(
        res.qp_converged_a
            .iter()
            .chain(&res.qp_converged_b)
            .all(|&c| c),
        "{}: QP Newton not converged",
        sys.label
    );
    res
}

fn qp_both(res: &UGwResult) -> Vec<f64> {
    res.eps_qp_a
        .iter()
        .chain(res.eps_qp_b.iter())
        .copied()
        .collect()
}

fn ref_qp_both(r: &Value, block: &str, ctx: &str) -> Vec<f64> {
    let mut v = vec_f64(r, &format!("/{block}/u_g0w0/alpha/eps_qp"), ctx);
    v.extend(vec_f64(r, &format!("/{block}/u_g0w0/beta/eps_qp"), ctx));
    v
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// ferric on the RAW ROHF result vs PySCF on the semi-canonical ROHF orbitals.
fn semicanonical_case(system: &str) {
    let sys = load(system);
    let ctx = sys.label.clone();
    let r = &sys.r;

    // The semi-canonicalization itself: every orbital energy, both spins.
    let view = unrestricted_reference(&sys.mol, &sys.rohf)
        .unwrap_or_else(|e| panic!("{ctx}: unrestricted_reference failed: {e:?}"));
    check_vec(
        &ctx,
        "semi-canonical eps_alpha",
        view.eps_a(),
        &vec_f64(r, "/semicanonical/eps_alpha", &ctx),
        TOL_EPS,
    );
    check_vec(
        &ctx,
        "semi-canonical eps_beta",
        view.eps_b(),
        &vec_f64(r, "/semicanonical/eps_beta", &ctx),
        TOL_EPS,
    );

    // U-RPA.
    let e_rpa = rpa(&sys, &sys.rohf);
    check_close(
        &ctx,
        "U-RPA E_c (40-pt GL)",
        e_rpa,
        num(r, "/semicanonical/urpa/e_corr", &ctx),
        TOL_E_RPA,
    );
    assert_misses(
        &ctx,
        "U-RPA vs legacy effective-Fock",
        &[e_rpa],
        &[num(r, "/legacy_effective_fock/urpa/e_corr", &ctx)],
        TOL_E_RPA,
    );

    // U-MP2, doubles only.
    let e_mp2 = mp2(&sys, &sys.rohf);
    check_close(
        &ctx,
        "U-MP2 E_c (doubles)",
        e_mp2,
        num(r, "/semicanonical/ump2_doubles/e_corr", &ctx),
        TOL_E_MP2,
    );
    assert_misses(
        &ctx,
        "U-MP2 vs legacy effective-Fock",
        &[e_mp2],
        &[num(r, "/legacy_effective_fock/ump2_doubles/e_corr", &ctx)],
        TOL_E_MP2,
    );
    assert_misses(
        &ctx,
        "U-MP2 vs ROMP2 (doubles + singles)",
        &[e_mp2],
        &[num(r, "/semicanonical/romp2_total", &ctx)],
        TOL_E_MP2,
    );

    // U-G0W0.
    let orbs = vec_usize(r, "/semicanonical/u_g0w0/orbs", &ctx);
    let res = gw(&sys, &sys.rohf, &orbs);
    for (spin, eps_mf) in [("alpha", &res.eps_mf_a), ("beta", &res.eps_mf_b)] {
        check_vec(
            &ctx,
            &format!("U-G0W0 eps_mf {spin}"),
            eps_mf.as_slice().unwrap(),
            &vec_f64(r, &format!("/semicanonical/u_g0w0/{spin}/eps_mf"), &ctx),
            TOL_EPS,
        );
    }
    check_vec(
        &ctx,
        "U-G0W0 eps_qp alpha",
        res.eps_qp_a.as_slice().unwrap(),
        &vec_f64(r, "/semicanonical/u_g0w0/alpha/eps_qp", &ctx),
        TOL_QP,
    );
    check_vec(
        &ctx,
        "U-G0W0 eps_qp beta",
        res.eps_qp_b.as_slice().unwrap(),
        &vec_f64(r, "/semicanonical/u_g0w0/beta/eps_qp", &ctx),
        TOL_QP,
    );
    assert_misses(
        &ctx,
        "U-G0W0 QP vs legacy effective-Fock",
        &qp_both(&res),
        &ref_qp_both(r, "legacy_effective_fock", &ctx),
        TOL_QP,
    );
}

/// ferric fed the legacy view explicitly reproduces PySCF's legacy control: the
/// control IS the pre-fix code path, and the harness distinguishes it.
fn legacy_case(system: &str) {
    let sys = load(system);
    let ctx = format!("{} [legacy view]", sys.label);
    let r = &sys.r;
    let legacy = legacy_view(&sys.rohf);
    check_close(
        &ctx,
        "U-RPA E_c",
        rpa(&sys, &legacy),
        num(r, "/legacy_effective_fock/urpa/e_corr", &ctx),
        TOL_E_RPA,
    );
    check_close(
        &ctx,
        "U-MP2 E_c",
        mp2(&sys, &legacy),
        num(r, "/legacy_effective_fock/ump2_doubles/e_corr", &ctx),
        TOL_E_MP2,
    );
    // U-G0W0 is not compared on the legacy view: with ROHF Roothaan energies
    // for both spins the quasiparticle equation of several window states has
    // nearby roots (the same view puts OH's alpha HOMO 9 eV from the
    // semicanonical value), so two correct solvers can land on different ones
    // (measured 8.8e-4 Ha OH, 8.3e-5 Ha CH3). U-RPA and U-MP2 above identify
    // the legacy path to 1e-11.
}

#[test]
#[ignore = "validation: ROHF-referenced correlation"]
fn rohf_reference_oh_vs_pyscf_semicanonical() {
    semicanonical_case("oh");
}

#[test]
#[ignore = "validation: ROHF-referenced correlation"]
fn rohf_reference_ch3_vs_pyscf_semicanonical() {
    semicanonical_case("ch3");
}

#[test]
#[ignore = "validation: ROHF-referenced correlation"]
fn legacy_view_reproduces_the_legacy_control_oh() {
    legacy_case("oh");
}

#[test]
#[ignore = "validation: ROHF-referenced correlation"]
fn legacy_view_reproduces_the_legacy_control_ch3() {
    legacy_case("ch3");
}
