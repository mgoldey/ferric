//! VALIDATION tier — VALIDATION.md rows "Meta-GGA gradient, closed" and
//! "Meta-GGA gradient, open".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_mgga_gradients \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's analytic SCAN and r2SCAN nuclear gradients
//! (`ks_gradient_closed` / `ks_gradient_uks` / `ks_gradient_roks`) against
//! PySCF 2.13 fed ferric's own basis JSON, aux JSON and Bohr geometry
//! (`scripts/validation/common.py`). References:
//! `scripts/validation/gen_mgga_gradients.py` →
//! `testdata/reference/validation/mgga_gradients/<system>_<basis>.json`.
//!
//! 1. **Closed shell (RKS)**: H2O and NH3 × 6-31G and def2-SVP × {RI-J, exact
//!    J}, against PySCF `RKS` with `grid_response = True`.
//! 2. **Open shell (UKS)**: HO2 (²A'') and NH2 (²B1) × 6-31G, RI-J, against
//!    PySCF `UKS` with `grid_response = True`, each reference state followed to
//!    an internally stable solution by a `stability()` loop.
//! 3. **Open shell (ROKS)**: NH2 × 6-31G, RI-J, against a CENTRAL FINITE
//!    DIFFERENCE of PySCF's ROKS energy (steps 2e-3, 1e-3, 5e-4 Bohr;
//!    Richardson of the two smallest). The step convergence is recorded in
//!    the JSON and asserted here before the reference is used.
//! 4. **Internal consistency, independent of PySCF**: ferric's analytic
//!    gradient against a central FD of ferric's OWN energy at two steps
//!    (H2O/6-31G RKS r2SCAN; NH2/6-31G UKS r2SCAN), the FD itself asserted
//!    step-converged first.
//!
//! # Reference quality (measured by the generator, 2026-09-24)
//!
//! * PySCF UKS analytic (grid_response) vs central FD of PySCF's own UKS
//!   energy, NH2/6-31G: 1.3e-10 (SCAN), 1.9e-10 (r2SCAN) Ha/Bohr — the
//!   analytic reference is the derivative of the energy it is paired with.
//! * PySCF ROKS FD, NH2/6-31G: max|g(2e-3)−g(1e-3)| = 3.1e-7 / 2.9e-7,
//!   max|g(1e-3)−g(5e-4)| = 7.7e-8 / 7.4e-8 (SCAN / r2SCAN; O(h²) truncation,
//!   ratio ~4), Richardson vs h=5e-4 2.5e-8. PySCF's analytic ROKS gradient
//!   agrees with that FD to 5.4e-10 / 3.3e-10.
//! * Every reference SCF is internally stable (RKS and UKS/ROKS `stability()`).
//!
//! # The like-for-like recipe (read from the code)
//!
//! * Grid: ferric's SCF and `ks_gradient_*` both use
//!   `AtomicGridConfig::default()` = (75,110) flat Becke grid with Becke-1988
//!   radii; the generator sets PySCF to `atom_grid=(75,110)`, `prune=None`,
//!   `radii_adjust=becke_atomic_radii_adjust` (the recipe the KS-energy row
//!   matches to ~1e-12 Ha for GGAs).
//! * Coulomb: for a functional, an UNSET `df_j_aux` auto-defaults to RI-J with
//!   def2-universal-jkfit in `solve_rhf` (rhf.rs `resolve_aux`), and the
//!   gradient differentiates the fitted energy (`ScfResult::df_jk` →
//!   `df_gradient`). The closed-shell RI-J case therefore leaves `df_j_aux`
//!   unset (the default path) and asserts that the SCF recorded RI-J; PySCF
//!   is `density_fit(aux)` with its default `auxbasis_response`. The exact-J
//!   control sets `df_j_aux = Some("")` (explicit four-centre J) against plain
//!   PySCF. The open-shell solvers do NOT auto-default (uhf.rs/rohf.rs read
//!   `df_j_aux` verbatim), so UKS/ROKS set `Some(AUX)` explicitly.
//! * State: ferric's stability analysis SKIPS meta-GGAs
//!   (`StabilitySkip::MetaGga`: no τ-dependent f_xc). The UKS state is pinned
//!   by the energy and ⟨S²⟩ matching PySCF's stability-followed reference.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's meta-GGA gradient is right: energies agree at the grid/fit
//!   floor, and gradients agree with PySCF (and with FD of ferric's own energy)
//!   at ~1e-7..1e-6 Ha/Bohr.
//! * If the τ term is wrong (a factor ½, a missing μ↔ν symmetrization, the
//!   grid-response τ piece dropped): the error scales with v_τ, so SCAN and
//!   r2SCAN miss by DIFFERENT amounts, and ferric's analytic gradient also
//!   disagrees with ferric's own FD (item 4) — the PySCF-independent witness.
//! * If the fitted-J gradient is wrong: the RI-J
//!   cases fail while the exact-J cases pass, and the FD control on the RI-J
//!   default path fails too.
//! * If ferric converges to a different open-shell state: the energy and
//!   ⟨S²⟩ assertions fail before any gradient is compared.
//! * If the d-shell AO-Hessian path is wrong: 6-31G passes and def2-SVP fails,
//!   for BOTH functionals and BOTH J treatments.
//! * If the HARNESS is broken (geometry constant, basis, mislabelled file, a
//!   functional or J mode silently not applied): E_nuc / AO count fail first,
//!   and the negative controls below fail rather than pass by accident.
//!
//! # Exactness anchors
//!
//! * Translational invariance: with grid response, Σ_A ∂E/∂R_A = 0 exactly,
//!   so every analytic gradient here is asserted to sum to zero (without grid
//!   response the sum is ~1e-5, so this is not vacuous).
//! * E_nuc, AO count and the SCF energy are asserted before any gradient.
//!
//! # TOLERANCES (measured max vs bar)
//!
//! | quantity | measured | bar |
//! |---|---:|---:|
//! | energy vs PySCF (RKS/UKS/ROKS) | 5.7e-13 Ha | `TOL_E` 1e-10 |
//! | gradient vs PySCF, closed | 1.3e-9 Ha/Bohr | `TOL_G_CLOSED` 1e-8 |
//! | gradient vs PySCF, UKS | 6.2e-9 | `TOL_G_OPEN` 3e-8 |
//! | gradient vs PySCF-ROKS FD | 2.2e-10 | `TOL_G_ROKS` 1e-7 |
//! | ferric analytic vs own FD | 1.0e-9 | `TOL_G_SELF_FD` 1e-8 |
//! | own-FD step convergence | 3.0e-7 | `TOL_FD_STEP` 1e-6 |
//! | Σ_A g_A | 5.4e-12 | `TOL_TRANSLATION` 1e-10 |
//! | ⟨S²⟩ | < 5e-9 | `TOL_S2` 5e-8 |
//!
//! # NEGATIVE CONTROLS (asserted inside the tests)
//!
//! * Functional swap: ferric's SCAN gradient must MISS the r2SCAN reference
//!   (and vice versa) by ≥ `MUST_MISS_FACTOR` × the bar.
//! * Basis swap (closed shell): ferric's 6-31G gradient must miss the
//!   def2-SVP reference by ≥ `MUST_MISS_FACTOR` × the bar.
//! * J swap (closed shell): ferric's RI-J gradient must miss the exact-J
//!   reference (and vice versa) by ≥ `J_MISS_FACTOR` × the bar, proving the
//!   bar can see the density-fitting term of the gradient. PySCF's own RI-J vs
//!   exact-J gradient gap is 3.1e-5 (H2O) / 1.8e-5 (NH3) at 6-31G and
//!   2.9e-6 / 7.5e-7 at def2-SVP, all above 3 × `TOL_G_CLOSED`.
//! * Method swap (ROKS): ferric's ROKS gradient must miss PySCF's UKS
//!   gradient (gap 6.4e-5 / 8.0e-5) by ≥ `MUST_MISS_FACTOR` × the bar.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::{ks_gradient_closed, ks_gradient_roks, ks_gradient_uks};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess};
use ferric_scf::ScfResult;
use ndarray::Array2;
use rayon::prelude::*;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/mgga_gradients";
const MOL_DIR: &str = "testdata/molecules/validation";
const AUX: &str = "def2-universal-jkfit";
const XCS: [&str; 2] = ["scan", "r2scan"];

/// SCF energy vs PySCF at the matched grid and fit. Measured max 5.7e-13.
const TOL_E: f64 = 1e-10;
/// Closed-shell gradient vs PySCF RKS `grid_response=True`. Measured max
/// 1.3e-9 (NH3 / 6-31G, r2SCAN); def2-SVP 5.0e-10.
const TOL_G_CLOSED: f64 = 1e-8;
/// UKS gradient vs PySCF UKS `grid_response=True`. Measured max 6.2e-9
/// (HO2, r2SCAN).
const TOL_G_OPEN: f64 = 3e-8;
/// ROKS gradient vs the PySCF ROKS central FD. Measured max 2.2e-10; the bar
/// covers the reference FD's own step uncertainty (~2.5e-8, recorded in the JSON).
const TOL_G_ROKS: f64 = 1e-7;
/// ferric analytic vs Richardson FD of ferric's own energy. Measured 1.0e-9.
const TOL_G_SELF_FD: f64 = 1e-8;
/// |g_FD(2e-3) − g_FD(1e-3)|: O(h²) truncation. Measured 3.0e-7.
const TOL_FD_STEP: f64 = 1e-6;
/// Σ_A g_A with grid response. Measured max 5.4e-12.
const TOL_TRANSLATION: f64 = 1e-10;
/// ⟨S²⟩ UKS vs PySCF. Agrees to all 8 printed digits.
const TOL_S2: f64 = 5e-8;
/// Largest |FD(h) − FD(h/2)| the REFERENCE's own FD may show before it is
/// refused as a reference (it would then not be a converged derivative).
const TOL_REF_FD_STEP: f64 = 5e-7;
const TOL_ENUC: f64 = 1e-9;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 10.0;
/// RI-J vs exact-J: the fitting term's gradient must be visible at the bar.
const J_MISS_FACTOR: f64 = 3.0;
/// ferric's own FD steps (Bohr).
const SELF_FD_STEPS: [f64; 2] = [2e-3, 1e-3];

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
             scripts/validation/gen_mgga_gradients.py — a missing reference is a failure, never a skip",
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

/// A `(natoms, 3)` gradient stored as a list of rows at JSON pointer `ptr`.
fn grad_at(v: &Value, ptr: &str, ctx: &str) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference gradient {ptr} missing"));
    let mut g = Array2::<f64>::zeros((rows.len(), 3));
    for (a, row) in rows.iter().enumerate() {
        let r = row.as_array().expect("gradient row");
        assert_eq!(r.len(), 3, "{ctx}: gradient row {a} is not length 3");
        for c in 0..3 {
            g[(a, c)] = r[c].as_f64().expect("gradient entry");
        }
    }
    g
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "gradient shapes differ");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// ferric's spelling of each functional.
fn ferric_xc(xc: &str) -> &'static str {
    match xc {
        "scan" => "SCAN",
        "r2scan" => "r2SCAN",
        other => panic!("unknown functional {other}"),
    }
}

fn other_xc(xc: &str) -> &'static str {
    match xc {
        "scan" => "r2scan",
        "r2scan" => "scan",
        other => panic!("unknown functional {other}"),
    }
}

struct System {
    mol: Molecule,
    bs: BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
    label: String,
}

fn load_system(system: &str, basis_name: &str, reference: &Value) -> System {
    let charge = reference["charge"].as_i64().expect("charge") as i32;
    let mult = reference["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let label = format!("{system}/{basis_name}");
    // Geometry like-for-like FIRST: a constant or unit slip fails here as a
    // geometry defect, not later as an unexplained gradient offset.
    let enuc_ref = num(reference, "/nuclear_repulsion", &label);
    let enuc = mol.nuclear_repulsion();
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{label}: nuclear repulsion {enuc:.12} vs reference {enuc_ref:.12} — geometry/unit mismatch"
    );
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let nao_ref = reference["nao"].as_u64().expect("nao") as usize;
    let s = ferric_integrals::oneelectron::overlap(&prep);
    assert_eq!(
        s.nrows(),
        nao_ref,
        "{label}: AO count differs from the reference's"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    System {
        mol,
        bs,
        prep,
        bounds,
        ctx: ParallelContext::default(),
        label,
    }
}

/// Closed-shell config. `j = "rij"` leaves `df_j_aux` UNSET on purpose: that
/// is ferric's default KS path (auto RI-J with def2-universal-jkfit), and the
/// caller asserts the SCF recorded it. `j = "exact"` opts out explicitly.
fn rks_config(xc: &str, j: &str) -> RhfConfig {
    let df_j_aux = match j {
        "rij" => None,
        "exact" => Some(String::new()),
        other => panic!("unknown J mode {other}"),
    };
    RhfConfig {
        xc: Some(ferric_xc(xc).into()),
        df_j_aux,
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 500,
        ..Default::default()
    }
}

/// Open-shell config: the UHF/ROHF solvers read `df_j_aux` verbatim (no
/// auto-default), so RI-J is requested explicitly to match PySCF `density_fit`.
fn open_config(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(ferric_xc(xc).into()),
        df_j_aux: Some(AUX.into()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 500,
        ..Default::default()
    }
}

fn assert_j_route(res: &ScfResult, j: &str, ctx: &str) {
    let j_aux = res.df_jk.as_ref().and_then(|r| r.j_aux.clone());
    match j {
        "rij" => assert_eq!(
            j_aux.as_deref(),
            Some(AUX),
            "{ctx}: SCF did not record RI-J with {AUX} — the like-for-like J recipe is broken"
        ),
        "exact" => assert!(
            j_aux.is_none(),
            "{ctx}: exact-J run recorded RI-J ({j_aux:?})"
        ),
        other => panic!("unknown J mode {other}"),
    }
}

fn assert_translation(g: &Array2<f64>, ctx: &str) {
    let worst = (0..3).map(|c| g.column(c).sum().abs()).fold(0.0, f64::max);
    eprintln!("{ctx}: max_c |Σ_A g_A,c| = {worst:.2e}");
    assert!(
        worst < TOL_TRANSLATION,
        "{ctx}: gradient is not translationally invariant (|Σ g| = {worst:.3e}); \
         grid response or a Pulay term is missing"
    );
}

fn report_and_assert(ctx: &str, g: &Array2<f64>, g_ref: &Array2<f64>, tol: f64) -> f64 {
    let d = max_abs_diff(g, g_ref);
    eprintln!("{ctx}: max|ferric − ref| = {d:.3e} (bar {tol:.0e})");
    for a in 0..g.nrows() {
        eprintln!(
            "    atom {a}: ferric [{:+.8e} {:+.8e} {:+.8e}]  ref [{:+.8e} {:+.8e} {:+.8e}]",
            g[(a, 0)],
            g[(a, 1)],
            g[(a, 2)],
            g_ref[(a, 0)],
            g_ref[(a, 1)],
            g_ref[(a, 2)]
        );
    }
    assert!(
        d < tol,
        "{ctx}: max|ferric − ref| = {d:.3e} exceeds {tol:.0e}"
    );
    d
}

fn assert_misses(ctx: &str, g: &Array2<f64>, g_wrong: &Array2<f64>, tol: f64, factor: f64) {
    let d = max_abs_diff(g, g_wrong);
    eprintln!(
        "{ctx}: NEGATIVE CONTROL max|ferric − wrong ref| = {d:.3e} (must be > {:.1e})",
        factor * tol
    );
    assert!(
        d > factor * tol,
        "{ctx}: negative control did not fail — ferric matches the WRONG reference to {d:.3e} \
         (< {factor}× bar {tol:.0e}); the comparison cannot discriminate"
    );
}

fn nocc_ab(mol: &Molecule) -> (usize, usize) {
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    ((nelec + two_s) / 2, (nelec - two_s) / 2)
}

/// ⟨S²⟩ of the UKS determinant (the quantity PySCF's `spin_square` reports).
fn s_squared(res: &ScfResult, s: &Array2<f64>, na: usize, nb: usize) -> f64 {
    let sz = 0.5 * (na as f64 - nb as f64);
    let ca = res.mos_alpha.slice(ndarray::s![.., ..na]);
    let cb = res.mos_beta.as_ref().unwrap().slice(ndarray::s![.., ..nb]);
    let ov = ca.t().dot(s).dot(&cb);
    sz * (sz + 1.0) + nb as f64 - ov.iter().map(|v| v * v).sum::<f64>()
}

// ── 1. Closed shell ─────────────────────────────────────────────────────────

/// Runs every closed-shell case for one system and basis, with the
/// functional-, J- and basis-swap negative controls.
fn run_closed(system: &str, basis_name: &str, other_basis: &str) {
    let r = reference(system, basis_name);
    let r_other = reference(system, other_basis);
    let sys = load_system(system, basis_name, &r);
    let op = Operator::coulomb();
    for xc in XCS {
        for j in ["rij", "exact"] {
            let ctx = format!("{} RKS {xc} {j}-J", sys.label);
            let res = solve_rhf(
                &sys.ctx,
                &sys.mol,
                &sys.prep,
                op,
                &sys.bounds,
                &rks_config(xc, j),
            )
            .unwrap_or_else(|e| panic!("{ctx}: SCF failed: {e:?}"));
            assert!(res.converged, "{ctx}: SCF did not converge");
            assert_j_route(&res, j, &ctx);

            let key = format!("/rks_{xc}_{j}");
            let e_ref = num(&r, &format!("{key}/energy"), &ctx);
            let de = (res.energy - e_ref).abs();
            eprintln!(
                "{ctx}: E ferric {:.10} ref {e_ref:.10} |d| {de:.2e}",
                res.energy
            );
            assert!(
                de < TOL_E,
                "{ctx}: energy off by {de:.3e} (bar {TOL_E:.0e})"
            );

            let g = ks_gradient_closed(
                &sys.mol,
                &sys.prep,
                &sys.bs,
                op,
                &sys.bounds,
                ferric_xc(xc),
                &res,
                None,
            )
            .unwrap_or_else(|e| panic!("{ctx}: gradient failed: {e:?}"));
            assert_translation(&g, &ctx);
            let g_ref = grad_at(&r, &format!("{key}/gradient"), &ctx);
            report_and_assert(&ctx, &g, &g_ref, TOL_G_CLOSED);

            // Negative controls.
            let g_swap = grad_at(&r, &format!("/rks_{}_{j}/gradient", other_xc(xc)), &ctx);
            assert_misses(
                &format!("{ctx} vs {} ref", other_xc(xc)),
                &g,
                &g_swap,
                TOL_G_CLOSED,
                MUST_MISS_FACTOR,
            );
            let other_j = if j == "rij" { "exact" } else { "rij" };
            let g_jswap = grad_at(&r, &format!("/rks_{xc}_{other_j}/gradient"), &ctx);
            assert_misses(
                &format!("{ctx} vs {other_j}-J ref"),
                &g,
                &g_jswap,
                TOL_G_CLOSED,
                J_MISS_FACTOR,
            );
            let g_bswap = grad_at(&r_other, &format!("{key}/gradient"), &ctx);
            assert_misses(
                &format!("{ctx} vs {other_basis} ref"),
                &g,
                &g_bswap,
                TOL_G_CLOSED,
                MUST_MISS_FACTOR,
            );
        }
    }
}

#[test]
#[ignore = "validation: Meta-GGA gradient, closed"]
fn mgga_gradient_closed_h2o_631g() {
    run_closed("h2o", "6-31g", "def2-svp");
}

#[test]
#[ignore = "validation: Meta-GGA gradient, closed"]
fn mgga_gradient_closed_h2o_def2svp() {
    run_closed("h2o", "def2-svp", "6-31g");
}

#[test]
#[ignore = "validation: Meta-GGA gradient, closed"]
fn mgga_gradient_closed_nh3_631g() {
    run_closed("nh3", "6-31g", "def2-svp");
}

#[test]
#[ignore = "validation: Meta-GGA gradient, closed"]
fn mgga_gradient_closed_nh3_def2svp() {
    run_closed("nh3", "def2-svp", "6-31g");
}

// ── 2. Open shell, UKS ──────────────────────────────────────────────────────

fn run_uks(system: &str, basis_name: &str) {
    let r = reference(system, basis_name);
    let sys = load_system(system, basis_name, &r);
    let op = Operator::coulomb();
    let (na, nb) = nocc_ab(&sys.mol);
    let s = ferric_integrals::oneelectron::overlap(&sys.prep);
    for xc in XCS {
        let ctx = format!("{} UKS {xc}", sys.label);
        let key = format!("/uks_{xc}_rij");
        assert_eq!(
            r.pointer(&format!("{key}/stability/internal_stable"))
                .and_then(Value::as_bool),
            Some(true),
            "{ctx}: reference state is not recorded as internally stable"
        );
        let res = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &open_config(xc))
            .unwrap_or_else(|e| panic!("{ctx}: SCF failed: {e:?}"));
        assert!(res.converged, "{ctx}: SCF did not converge");
        assert_j_route(&res, "rij", &ctx);

        // State first: energy and ⟨S²⟩ (ferric skips meta-GGA stability).
        let e_ref = num(&r, &format!("{key}/energy"), &ctx);
        let de = (res.energy - e_ref).abs();
        let s2 = s_squared(&res, &s, na, nb);
        let s2_ref = num(&r, &format!("{key}/s_squared"), &ctx);
        eprintln!(
            "{ctx}: E ferric {:.10} ref {e_ref:.10} |d| {de:.2e}; <S2> {s2:.8} ref {s2_ref:.8}",
            res.energy
        );
        assert!(
            de < TOL_E,
            "{ctx}: energy off by {de:.3e} — different state or recipe"
        );
        assert!(
            (s2 - s2_ref).abs() < TOL_S2,
            "{ctx}: <S2> {s2:.8} vs {s2_ref:.8} — different state"
        );

        let g = ks_gradient_uks(
            &sys.mol,
            &sys.prep,
            &sys.bs,
            op,
            &sys.bounds,
            ferric_xc(xc),
            &res,
            None,
        )
        .unwrap_or_else(|e| panic!("{ctx}: gradient failed: {e:?}"));
        assert_translation(&g, &ctx);
        let g_ref = grad_at(&r, &format!("{key}/gradient"), &ctx);
        report_and_assert(&ctx, &g, &g_ref, TOL_G_OPEN);

        let g_swap = grad_at(&r, &format!("/uks_{}_rij/gradient", other_xc(xc)), &ctx);
        assert_misses(
            &format!("{ctx} vs {} ref", other_xc(xc)),
            &g,
            &g_swap,
            TOL_G_OPEN,
            MUST_MISS_FACTOR,
        );
    }
}

#[test]
#[ignore = "validation: Meta-GGA gradient, open"]
fn mgga_gradient_uks_ho2_631g() {
    run_uks("ho2", "6-31g");
}

#[test]
#[ignore = "validation: Meta-GGA gradient, open"]
fn mgga_gradient_uks_nh2_631g() {
    run_uks("nh2", "6-31g");
}

// ── 3. Open shell, ROKS vs FD of PySCF's ROKS energy ────────────────────────

#[test]
#[ignore = "validation: Meta-GGA gradient, open"]
fn mgga_gradient_roks_nh2_631g_vs_pyscf_fd() {
    let r = reference("nh2", "6-31g");
    let sys = load_system("nh2", "6-31g", &r);
    let op = Operator::coulomb();
    for xc in XCS {
        let ctx = format!("{} ROKS {xc}", sys.label);
        let key = format!("/roks_{xc}_rij");
        // The reference is only a reference if its FD is step-converged.
        let steps = r
            .pointer(&format!("{key}/fd/step_convergence"))
            .and_then(Value::as_object)
            .unwrap_or_else(|| panic!("{ctx}: fd/step_convergence missing"));
        for (k, v) in steps {
            let d = v.as_f64().expect("step convergence value");
            eprintln!("{ctx}: reference FD {k} = {d:.2e}");
            assert!(
                d < TOL_REF_FD_STEP,
                "{ctx}: reference FD not step-converged: {k} = {d:.3e}"
            );
        }

        let res = solve_rohf(
            &sys.ctx,
            &sys.mol,
            &sys.prep,
            op,
            &sys.bounds,
            &open_config(xc),
        )
        .unwrap_or_else(|e| panic!("{ctx}: SCF failed: {e:?}"));
        assert!(res.converged, "{ctx}: SCF did not converge");
        assert_j_route(&res, "rij", &ctx);
        let e_ref = num(&r, &format!("{key}/energy"), &ctx);
        let de = (res.energy - e_ref).abs();
        eprintln!(
            "{ctx}: E ferric {:.10} ref {e_ref:.10} |d| {de:.2e}",
            res.energy
        );
        assert!(de < TOL_E, "{ctx}: energy off by {de:.3e}");

        let g = ks_gradient_roks(
            &sys.mol,
            &sys.prep,
            &sys.bs,
            op,
            &sys.bounds,
            ferric_xc(xc),
            &res,
            None,
        )
        .unwrap_or_else(|e| panic!("{ctx}: gradient failed: {e:?}"));
        assert_translation(&g, &ctx);
        let g_ref = grad_at(&r, &format!("{key}/fd/gradient"), &ctx);
        report_and_assert(&ctx, &g, &g_ref, TOL_G_ROKS);

        // Informational: PySCF's own analytic ROKS gradient (not asserted
        // against — the FD is the reference; its FD gap is in the JSON).
        let g_pyscf = grad_at(&r, &format!("{key}/pyscf_analytic_gradient"), &ctx);
        eprintln!(
            "{ctx}: (info) max|ferric − PySCF analytic ROKS| = {:.3e}",
            max_abs_diff(&g, &g_pyscf)
        );

        let g_swap = grad_at(&r, &format!("/roks_{}_rij/fd/gradient", other_xc(xc)), &ctx);
        assert_misses(
            &format!("{ctx} vs {} ref", other_xc(xc)),
            &g,
            &g_swap,
            TOL_G_ROKS,
            MUST_MISS_FACTOR,
        );
        // ROKS is not UKS: PySCF's UKS and ROKS gradients for NH2 differ by
        // 6.4e-5 (SCAN) / 8.0e-5 (r2SCAN) Ha/Bohr, so a ferric ROKS gradient
        // that silently fell through to the UKS path would fail here.
        let g_uks = grad_at(&r, &format!("/uks_{xc}_rij/gradient"), &ctx);
        assert_misses(
            &format!("{ctx} vs UKS ref"),
            &g,
            &g_uks,
            TOL_G_ROKS,
            MUST_MISS_FACTOR,
        );
    }
}

// ── 4. ferric analytic vs FD of ferric's own energy ─────────────────────────

fn displaced(mol: &Molecule, atom: usize, coord: usize, delta: f64) -> Molecule {
    let mut m = mol.clone();
    match coord {
        0 => m.atoms[atom].x += delta,
        1 => m.atoms[atom].y += delta,
        _ => m.atoms[atom].zpos += delta,
    }
    m
}

/// Central FD of `energy(mol)` at step `h` (Bohr), all 3N components.
fn fd_gradient<F>(mol: &Molecule, h: f64, energy: &F) -> Array2<f64>
where
    F: Fn(&Molecule) -> f64 + Sync,
{
    let natoms = mol.atoms.len();
    let pairs: Vec<(usize, usize)> = (0..natoms)
        .flat_map(|a| (0..3).map(move |c| (a, c)))
        .collect();
    let vals: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(a, c)| {
            let ep = energy(&displaced(mol, a, c, h));
            let em = energy(&displaced(mol, a, c, -h));
            ((a, c), (ep - em) / (2.0 * h))
        })
        .collect();
    let mut g = Array2::<f64>::zeros((natoms, 3));
    for ((a, c), v) in vals {
        g[(a, c)] = v;
    }
    g
}

/// Assert FD(h) and FD(h/2) agree, then compare the analytic gradient to the
/// Richardson combination (4 FD(h/2) − FD(h)) / 3.
fn assert_self_fd<F>(ctx: &str, mol: &Molecule, g_ana: &Array2<f64>, energy: F)
where
    F: Fn(&Molecule) -> f64 + Sync,
{
    let [h1, h2] = SELF_FD_STEPS;
    let g1 = fd_gradient(mol, h1, &energy);
    let g2 = fd_gradient(mol, h2, &energy);
    let step = max_abs_diff(&g1, &g2);
    eprintln!("{ctx}: own-FD step convergence max|g({h1:.0e}) − g({h2:.0e})| = {step:.3e}");
    assert!(
        step < TOL_FD_STEP,
        "{ctx}: ferric's own FD is not step-converged ({step:.3e}); SCF noise or a basin change"
    );
    let rich = (&g2 * 4.0 - &g1) / 3.0;
    report_and_assert(
        &format!("{ctx} analytic vs own FD"),
        g_ana,
        &rich,
        TOL_G_SELF_FD,
    );
}

#[test]
#[ignore = "validation: Meta-GGA gradient, closed"]
fn mgga_gradient_closed_h2o_631g_r2scan_vs_own_fd() {
    let r = reference("h2o", "6-31g");
    let sys = load_system("h2o", "6-31g", &r);
    let op = Operator::coulomb();
    let ctx = format!("{} RKS r2scan rij-J (default path)", sys.label);
    let cfg = rks_config("r2scan", "rij");
    let res = solve_rhf(&sys.ctx, &sys.mol, &sys.prep, op, &sys.bounds, &cfg).unwrap();
    assert!(res.converged, "{ctx}: SCF did not converge");
    assert_j_route(&res, "rij", &ctx);
    let g = ks_gradient_closed(
        &sys.mol,
        &sys.prep,
        &sys.bs,
        op,
        &sys.bounds,
        "r2SCAN",
        &res,
        None,
    )
    .unwrap();
    // Every displaced SCF starts from the reference density (a fresh guess can
    // land a ± twin in another basin and fabricate an FD "gradient").
    let seeded = RhfConfig {
        init_guess_density: Some(res.density_r().clone()),
        use_sad_guess: false,
        ..cfg
    };
    let bs = sys.bs.clone();
    let energy = |m: &Molecule| {
        let prep = PreparedBasis::new(m, &bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let r = solve_rhf(&ParallelContext::default(), m, &prep, op, &bounds, &seeded).unwrap();
        assert!(r.converged, "displaced SCF did not converge");
        r.energy
    };
    assert_self_fd(&ctx, &sys.mol, &g, energy);
}

#[test]
#[ignore = "validation: Meta-GGA gradient, open"]
fn mgga_gradient_uks_nh2_631g_r2scan_vs_own_fd() {
    let r = reference("nh2", "6-31g");
    let sys = load_system("nh2", "6-31g", &r);
    let op = Operator::coulomb();
    let ctx = format!("{} UKS r2scan rij-J", sys.label);
    let cfg = open_config("r2scan");
    let res = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg).unwrap();
    assert!(res.converged, "{ctx}: SCF did not converge");
    let g = ks_gradient_uks(
        &sys.mol,
        &sys.prep,
        &sys.bs,
        op,
        &sys.bounds,
        "r2SCAN",
        &res,
        None,
    )
    .unwrap();
    let ca = res.mos_alpha.clone();
    let cb = res.mos_beta.clone().expect("UKS beta MOs");
    let e0 = res.energy;
    let bs = sys.bs.clone();
    let energy = |m: &Molecule| {
        let prep = PreparedBasis::new(m, &bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let r = solve_uhf_with_guess(
            &ParallelContext::default(),
            m,
            &prep,
            &bounds,
            &cfg,
            Some((&ca, &cb)),
        )
        .unwrap();
        assert!(r.converged, "displaced SCF did not converge");
        assert!((r.energy - e0).abs() < 1e-3, "displaced SCF changed basin");
        r.energy
    };
    assert_self_fd(&ctx, &sys.mol, &g, energy);
}
