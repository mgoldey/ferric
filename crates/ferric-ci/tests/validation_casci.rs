//! VALIDATION tier — VALIDATION.md row "CAS-CI".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-ci --test validation_casci \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! `ferric_ci::run_cas_ci` on ferric's own exact-J/K RHF (`solve_rhf`
//! default: no RI-J, no RI-K) against PySCF 2.13 `mcscf.CASCI(mf, ncas,
//! nelecas)` on PySCF's exact-integral RHF, both fed ferric's basis JSON and
//! Bohr geometry. The orbital window is the same rule on both sides: ferric's
//! `active_start` = PySCF's default `ncore = (N − nelecas)/2`, canonical RHF
//! MOs in energy order.
//!
//! | system | basis | CAS | active_start | N_det | max \|c\| |
//! |---|---|---|---:|---:|---:|
//! | N2, r = 1.10 Å | cc-pVDZ | (6,6) | 4 | 400 | 0.97 |
//! | N2, r = 2.20 Å | cc-pVDZ | (6,6) | 4 | 400 | 0.47 |
//! | H2O | 6-31G | (4,4) | 3 | 36 | 0.9996 |
//!
//! Quantities: E_RHF, the canonical orbital energies across the window edges
//! (the SAME orbitals must be in the window), E_CASCI, the core energy e_core
//! (E_nuc + inactive electronic energy; PySCF `get_h1eff()[1]`), and the
//! active-space energy `CasCiResult::e_active` = E_CASCI − e_core (PySCF
//! `e_cas`).
//!
//! The PySCF CAS-CI is anchored in the generator: the full dense active-space
//! Hamiltonian (`fci.direct_spin1.pspace` over every determinant, numpy
//! `eigvalsh`) reproduces `e_tot` to ≤ 5.7e-14 Ha (`dense_anchor_diff`,
//! re-checked here), so the reference is the true lowest Ms = 0 root, which is
//! what ferric's Davidson targets (no spin penalty on either side; PySCF
//! ⟨S²⟩ ≈ 0 for every root compared).
//!
//! References: `scripts/validation/gen_casci.py` →
//! `testdata/reference/validation/casci/<system>_<basis>.json`.
//!
//! # The stretched-N2 reference state
//!
//! At r = 2.20 Å the symmetric RHF (D∞h aufbau, 3σg² 1πu⁴) is internally
//! UNSTABLE toward a symmetry-broken RHF 0.19 Ha lower (recorded as
//! `rhf/symmetry_broken_rhf_energy`). Both ferric and PySCF reach the symmetric
//! RHF from their default guesses, and the generator pins the reference to it
//! (it must equal PySCF's point-group-adapted RHF to 1e-9). The CAS is built
//! on that symmetric RHF — the conventional reference for a valence CAS of
//! stretched N2 — and its CAS(6,6) is strongly multireference (largest CI
//! coefficient 0.47, E_CASCI − E_RHF = −0.50 Ha).
//!
//! # Exactness anchor (asserted first, per system)
//!
//! E_nuc (geometry/units), the AO count, E_RHF, and the window-edge orbital
//! energies. A different RHF state, a different orbital order at the window
//! edge, or a degenerate set cut by the window fails here, before any CI
//! number is compared (the generator also refuses a window whose edge gap is
//! below 1e-4 Ha).
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's CAS-CI is right: E_CASCI, e_core and the active energy agree
//!   with PySCF at the SCF-convergence floor (both codes use exact integrals
//!   and the same window; E_CASCI is first order in the orbitals, so the floor
//!   is set by the RHF density convergence, ~1e-10 Ha).
//! * If the inactive-core fold is wrong (a missing exchange term in e_core or
//!   in h_eff, a factor of 2 on the core density): e_core misses by
//!   O(1) Ha and the active energy by the compensating amount — the split
//!   is compared separately so the failure names the part.
//! * If the Slater–Condon elements or the Davidson are wrong: the active energy
//!   misses while e_core passes. N2 at 2.20 Å has a large multideterminant
//!   weight, so errors in double-excitation elements are not suppressed by a
//!   dominant reference determinant, as they are at 1.10 Å and in H2O.
//! * If the window were silently off by one (the negative control below): the
//!   total misses by 9.5e-5 (H2O) to 0.13 Ha (N2 2.20 Å).
//!
//! # TOLERANCES
//!
//! | quantity | measured max \|d\| | bar |
//! |---|---:|---:|
//! | E_RHF | 2.2e-12 Ha | [`TOL_E_RHF`] 5e-11 Ha |
//! | window orbital energies | 1.2e-10 Ha | [`TOL_EPS`] 2e-9 Ha |
//! | E_CASCI, e_core, e_active, E_CASCI − E_RHF | 2.6e-10 Ha (e_core and e_active; E_CASCI 1.4e-11) | [`TOL_E_CAS`] 3e-9 Ha |
//!
//! # NEGATIVE CONTROLS (asserted)
//!
//! * Shifted window: the generator also records CAS(n+2, m) with
//!   `active_start − 1` (`casci_shifted`, same n_active, two more active
//!   electrons). ferric on that window must MATCH `casci_shifted`, and each
//!   ferric result must MISS the other window's reference by >
//!   [`MUST_MISS_FACTOR`] × the bar (measured separations: H2O 9.5e-5,
//!   N2 1.10 Å 4.8e-3, N2 2.20 Å 0.13 Ha). A comparison that ignored the
//!   window would fail.
//!
//! # MUTATION (to run once, record the outcome here)
//!
//! In `crates/ferric-ci/src/integrals.rs`, drop the inactive exchange term
//! from the core energy:
//!
//! ```text
//! e_core += 2.0 * mo_eri[midx(i, i, j, j)] - mo_eri[midx(i, j, j, i)];
//!   ->
//! e_core += 2.0 * mo_eri[midx(i, i, j, j)];
//! ```
//!
//! Expected: e_core and E_CASCI fail for every system (by Σ_ij (ij|ji), O(1)
//! Ha); the active energy still passes (h_eff is untouched).
//! Outcome (2026-09-25): all three tests fail at E_CASCI (6.6–10 Ha off); the
//! active-energy check is after it, so this run does not observe it.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_ci::{run_cas_ci, CasCiConfig, CasCiResult};
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/casci";
const MOL_DIR: &str = "testdata/molecules/validation";

const TOL_ENUC: f64 = 1e-9;
/// E_RHF vs PySCF (exact J/K both sides). Measured 2.2e-12.
const TOL_E_RHF: f64 = 5e-11;
/// Canonical orbital energies across the window edges. Measured 1.2e-10.
const TOL_EPS: f64 = 2e-9;
/// Every CAS-CI energy (total, core, active, correlation). Measured 2.6e-10
/// (e_core and e_active, which cancel in E_CASCI: 1.4e-11).
const TOL_E_CAS: f64 = 3e-9;
/// The generator's dense-Hamiltonian anchor bar.
const TOL_DENSE_ANCHOR: f64 = 1e-10;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 100.0;

/// Workspace root, found by walking up from the CWD (nextest sets the CWD to
/// the package dir); `CARGO_MANIFEST_DIR` is only a fallback because it is
/// baked in at compile time and is wrong inside a nextest archive.
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
        .expect("ferric-ci manifest dir should be <root>/crates/ferric-ci")
        .to_path_buf()
}

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_casci.py — a missing reference is a failure, never a skip",
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

fn uint(v: &Value, ptr: &str, ctx: &str) -> usize {
    v.pointer(ptr)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an integer"))
        as usize
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<22} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_miss(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    let must = MUST_MISS_FACTOR * tol;
    eprintln!("{ctx}: control {what:<34} |d| {d:.2e} (must exceed {must:.0e})");
    assert!(
        d > must,
        "{ctx}: negative control '{what}' did not miss: |d| {d:.2e} <= {must:.0e} — the \
         comparison does not respond to what the control changes"
    );
}

struct Sys {
    label: String,
    r: Value,
    mol: Molecule,
    prep: PreparedBasis,
    rhf: ScfResult,
}

/// Load the reference, build the molecule and basis, converge ferric's RHF and
/// assert the exactness anchors (E_nuc, AO count, E_RHF).
fn load(system: &str, basis_name: &str) -> Sys {
    let r = reference(system, basis_name);
    let label = format!("{system}/{basis_name}");
    assert_eq!(r["basis"].as_str(), Some(basis_name), "{label}: basis");
    assert_eq!(r["charge"].as_i64(), Some(0), "{label}: charge");
    assert_eq!(r["multiplicity"].as_u64(), Some(1), "{label}: multiplicity");
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
        mol.nelec() as usize,
        uint(&r, "/nelectron", &label),
        "{label}: electron count"
    );
    let prep = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    assert_eq!(prep.nbasis(), uint(&r, "/nao", &label), "{label}: AO count");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        max_iter: 300,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    };
    assert!(
        cfg.df_j_aux.is_none() && cfg.df_k_aux.is_none(),
        "{label}: the reference RHF uses exact J/K"
    );
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{label}: RHF failed: {e:?}"));
    assert!(rhf.converged, "{label}: RHF did not converge");
    assert!(
        rhf.df_jk.is_none(),
        "{label}: RHF used density fitting ({:?}); the reference RHF is exact",
        rhf.df_jk
    );
    check_close(
        &label,
        "E_RHF",
        rhf.energy,
        num(&r, "/rhf/energy", &label),
        TOL_E_RHF,
    );
    Sys {
        label,
        r,
        mol,
        prep,
        rhf,
    }
}

/// Run ferric's CAS-CI on the window recorded in reference block `block`,
/// after checking that the same orbitals sit in (and just outside) the window.
fn run_block(sys: &Sys, block: &str) -> CasCiResult {
    let ctx = format!("{} {block}", sys.label);
    let r = &sys.r;
    let active_start = uint(r, &format!("/{block}/active_start"), &ctx);
    let n_active = uint(r, &format!("/{block}/n_active"), &ctx);
    let na = uint(r, &format!("/{block}/n_elec_active/0"), &ctx);
    let nb = uint(r, &format!("/{block}/n_elec_active/1"), &ctx);
    assert_eq!(
        uint(r, &format!("/{block}/ncore"), &ctx),
        active_start,
        "{ctx}: PySCF ncore must equal ferric's active_start"
    );

    // Window orbitals: [active_start - 1, active_start + n_active] inclusive,
    // clipped to the MO range, exactly as the generator records them.
    let eps = sys.rhf.eps_r();
    let lo = active_start.saturating_sub(1);
    let hi = (active_start + n_active + 1).min(eps.len());
    let want = r
        .pointer(&format!("/{block}/window_mo_energies"))
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: window_mo_energies missing"));
    assert_eq!(want.len(), hi - lo, "{ctx}: window orbital count");
    let d_eps = eps[lo..hi]
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w.as_f64().expect("orbital energy")).abs())
        .fold(0.0f64, f64::max);
    eprintln!("{ctx}: window orbital energies max |d| {d_eps:.2e} (tol {TOL_EPS:.0e})");
    assert!(
        d_eps < TOL_EPS,
        "{ctx}: ferric's orbitals around the window differ from PySCF's by {d_eps:.2e} — \
         a different RHF state or orbital order; no CI number is meaningful"
    );

    let anchor = num(r, &format!("/{block}/dense_anchor_diff"), &ctx);
    assert!(
        anchor.abs() < TOL_DENSE_ANCHOR,
        "{ctx}: generator dense-H anchor {anchor:.2e} exceeds {TOL_DENSE_ANCHOR:.0e}"
    );

    let cfg = CasCiConfig {
        n_active,
        n_elec_active: (na, nb),
        active_start,
        conv_thresh: 1e-9,
        ..Default::default()
    };
    let res = run_cas_ci(&sys.mol, &sys.prep, &sys.rhf, cfg)
        .unwrap_or_else(|e| panic!("{ctx}: run_cas_ci failed: {e:?}"));
    assert!(res.converged, "{ctx}: Davidson did not converge");
    assert_eq!(
        res.n_determinants,
        uint(r, &format!("/{block}/n_determinants"), &ctx),
        "{ctx}: determinant count"
    );
    res
}

/// Compare one ferric CAS-CI result against reference block `block`.
fn check_block(sys: &Sys, block: &str, res: &CasCiResult) {
    let ctx = format!("{} {block}", sys.label);
    let r = &sys.r;
    check_close(
        &ctx,
        "E_CASCI",
        res.e_total,
        num(r, &format!("/{block}/e_total"), &ctx),
        TOL_E_CAS,
    );
    check_close(
        &ctx,
        "e_core",
        res.e_core,
        num(r, &format!("/{block}/e_core"), &ctx),
        TOL_E_CAS,
    );
    check_close(
        &ctx,
        "e_active (E_CASCI - e_core)",
        res.e_active,
        num(r, &format!("/{block}/e_active"), &ctx),
        TOL_E_CAS,
    );
    check_close(
        &ctx,
        "E_CASCI - E_RHF",
        res.e_total - sys.rhf.energy,
        num(r, &format!("/{block}/e_corr_vs_rhf"), &ctx),
        TOL_E_CAS,
    );
}

fn casci_case(system: &str, basis_name: &str) {
    let sys = load(system, basis_name);
    let main = run_block(&sys, "casci");
    check_block(&sys, "casci", &main);
    let shifted = run_block(&sys, "casci_shifted");
    check_block(&sys, "casci_shifted", &shifted);
    // Controls: the two windows are distinguishable in both directions.
    check_miss(
        &sys.label,
        "E_CASCI vs shifted-window ref",
        main.e_total,
        num(&sys.r, "/casci_shifted/e_total", &sys.label),
        TOL_E_CAS,
    );
    check_miss(
        &sys.label,
        "shifted E_CASCI vs plan-window ref",
        shifted.e_total,
        num(&sys.r, "/casci/e_total", &sys.label),
        TOL_E_CAS,
    );
}

#[test]
#[ignore = "validation: CAS-CI"]
fn casci_n2_r1p10_ccpvdz_cas66_vs_pyscf() {
    casci_case("n2_r1.10", "cc-pvdz");
}

#[test]
#[ignore = "validation: CAS-CI"]
fn casci_n2_r2p20_ccpvdz_cas66_vs_pyscf() {
    casci_case("n2_r2.20", "cc-pvdz");
}

#[test]
#[ignore = "validation: CAS-CI"]
fn casci_h2o_631g_cas44_vs_pyscf() {
    casci_case("h2o", "6-31g");
}
