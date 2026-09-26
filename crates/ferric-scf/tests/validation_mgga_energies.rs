//! VALIDATION tier — VALIDATION.md row "SCAN/r2SCAN energy".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_mgga_energies \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! `dft_scan.rs` checks SCAN/r2SCAN SCF energies at ONE basis (cc-pVDZ) with a
//! 1e-4 Ha bar. This file widens the row to two bases, a second-row molecule
//! and both Coulomb routes, against PySCF 2.13 fed ferric's own basis JSON,
//! aux JSON and Bohr geometry (`scripts/validation/common.py`):
//!
//! 1. **Closed shell (RKS, `solve_rhf`)**: H2O, NH3, H2S, CH4 × SCAN, r2SCAN ×
//!    def2-SVP, def2-TZVP × {RI-J (ferric's default), exact J}. Energy, HOMO
//!    and LUMO.
//! 2. **Open shell (UKS, `solve_uhf`)**: NH2 (²B1) × SCAN, r2SCAN × def2-SVP,
//!    def2-TZVP, RI-J. Energy, ⟨S²⟩ and the α/β frontier orbitals, against a
//!    reference followed to an internally stable state by PySCF's
//!    `stability()` loop.
//!
//! References: `scripts/validation/gen_mgga_energies.py` →
//! `testdata/reference/validation/mgga_energies/<system>_<basis>.json`.
//!
//! # The like-for-like recipe (read from the code)
//!
//! * Grid: ferric's `AtomicGridConfig::default()` = (75,110) flat Becke grid
//!   with Becke-1988 radii; PySCF `atom_grid=(75,110)`, `prune=None`,
//!   `radii_adjust=becke_atomic_radii_adjust`, `small_rho_cutoff=0`.
//! * Coulomb: for a functional, an UNSET `df_j_aux` auto-defaults to RI-J with
//!   def2-universal-jkfit in `solve_rhf` (rhf.rs `resolve_aux`); the RI-J
//!   closed-shell case leaves it unset and asserts the SCF recorded RI-J, the
//!   exact-J case sets `Some("")`. `solve_uhf` reads `df_j_aux` verbatim, so
//!   UKS sets `Some(AUX)`. PySCF: `density_fit(aux)` / plain KS.
//! * XC density floor: ferric zeroes every V_xc factor where ρ ≤ 1e-10 (total
//!   ρ closed shell, ρ_σ per spin open shell) and sums E_xc unfloored
//!   (vxc.rs / xc_batch.rs `DENSITY_FLOOR`); the generator applies the same
//!   rule to PySCF's `eval_xc_eff`. Measured E(floored) − E(unfloored) on the
//!   reference side: ≤ 1.6e-12 Ha, the SCF convergence noise at these bases.
//! * ferric's default meta-GGA virtual-block level shift (0.5, ramped to 0 at
//!   convergence) changes the SCF path, not the converged energy.
//!
//! # Reference quality (measured by the generator, 2026-09-25)
//!
//! * Every RKS reference is internally stable (`stability()`); the NH2 UKS
//!   references are stable from all three guesses with no stability rounds,
//!   λ_min = 6.2e-2..8.1e-2 Ha, and a single stable minimum.
//! * Separations the negative controls need (reference side): SCAN vs r2SCAN
//!   5.9e-3..5.7e-2 Ha; RI-J vs exact J 2.5e-5..5.9e-5 Ha; def2-SVP vs
//!   def2-TZVP ≥ 4.6e-2 Ha.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's meta-GGA energy is right: energies agree at the grid/fit
//!   floor the gradient row already measured at def2-SVP (5.7e-13 Ha), at
//!   def2-TZVP too, and for the second-row H2S.
//! * If the τ term is wrong (a factor ½ in τ, V^τ dropped or doubled): the
//!   error scales with v_τ, so SCAN and r2SCAN miss by DIFFERENT, mHa-scale
//!   amounts, on every system and both bases.
//! * If the f-shell / higher-l AO-derivative path used for τ is wrong:
//!   def2-SVP passes and def2-TZVP (the first basis here with f functions on
//!   the heavy atoms) fails, for both functionals and both J routes.
//! * If the spin-polarized τ path is wrong: the RKS half passes and the NH2
//!   UKS half fails.
//! * If the HARNESS is broken (geometry constant, basis, mislabelled file, a
//!   functional or J route silently not applied): E_nuc / AO count fail first,
//!   and the negative controls below fail rather than pass by accident.
//!
//! # TOLERANCES (measured max vs bar)
//!
//! | quantity | measured | bar |
//! |---|---:|---:|
//! | RKS energy, RI-J and exact J | 4.0e-12 Ha | `TOL_E` 5e-11 Ha |
//! | RKS HOMO / LUMO | 2.7e-9 Ha | `TOL_EPS` 1e-7 Ha |
//! | UKS energy (NH2) | 2.4e-13 Ha | `TOL_E` 5e-11 Ha |
//! | UKS ⟨S²⟩ | 5.2e-10 | `TOL_S2` 5e-9 |
//! | UKS α/β HOMO / LUMO | 1.1e-8 Ha | `TOL_EPS` 1e-7 Ha |
//! | nuclear repulsion | 1.8e-15 Ha | `TOL_ENUC` 1e-9 Ha |
//!
//! # NEGATIVE CONTROLS (asserted inside the tests)
//!
//! * Functional swap: ferric's SCAN energy must MISS the r2SCAN reference
//!   (and vice versa) by ≥ `MUST_MISS_FACTOR` × `TOL_E`.
//! * Basis swap: ferric's def2-SVP energy must miss the def2-TZVP reference
//!   (and vice versa) by ≥ `MUST_MISS_FACTOR` × `TOL_E`.
//! * J swap (closed shell): ferric's RI-J energy must miss the exact-J
//!   reference (and vice versa) by ≥ `MUST_MISS_FACTOR` × `TOL_E`, so the bar
//!   can see a Coulomb-route mismatch.
//! * Functional applied at all: RKS must miss the exact RHF control, UKS the
//!   exact UHF control, by > `CONTROL_GAP`.
//!
//! # MUTATIONS (to run once, record the outcome here)
//!
//! * MUTATION A — zero the V^τ factor (`t_half`) in xc_batch.rs
//!   `integrate_closed`: every RKS energy must fail, UKS must still pass.
//! * MUTATION B — same in `integrate_polarized`: every UKS energy must fail,
//!   RKS must still pass.
//!
//! Run 2026-09-25: A fails all eight RKS tests with both UKS green; B fails
//! both UKS tests with all eight RKS green.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{ScfResult, Spin};
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/mgga_energies";
const MOL_DIR: &str = "testdata/molecules/validation";
const AUX: &str = "def2-universal-jkfit";
const BASES: [&str; 2] = ["def2-svp", "def2-tzvp"];
const J_MODES: [&str; 2] = ["rij", "exact"];

/// SCF energy vs PySCF at the matched grid and fit.
const TOL_E: f64 = 5e-11; // measured ≤ 4.0e-12 (H2S/def2-TZVP r2SCAN exact J)
/// HOMO/LUMO (and α/β frontier orbitals for UKS) vs PySCF.
const TOL_EPS: f64 = 1e-7; // measured ≤ 1.1e-8 (NH2 β HOMO)
/// ⟨S²⟩ UKS vs PySCF.
const TOL_S2: f64 = 5e-9; // measured ≤ 5.2e-10
const TOL_ENUC: f64 = 1e-9; // measured ≤ 1.8e-15
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 100.0;
/// KS vs HF: the functional must move the energy by more than this.
const CONTROL_GAP: f64 = 1e-2;

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
             scripts/validation/gen_mgga_energies.py — a missing reference is a failure, never a skip",
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

fn other_basis(basis_name: &str) -> &'static str {
    match basis_name {
        "def2-svp" => "def2-tzvp",
        "def2-tzvp" => "def2-svp",
        other => panic!("unknown basis {other}"),
    }
}

struct System {
    mol: Molecule,
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
    // geometry defect, not later as an unexplained energy offset.
    let enuc_ref = num(reference, "/nuclear_repulsion", &label);
    let enuc = mol.nuclear_repulsion();
    eprintln!(
        "{label}: E_nuc ferric {enuc:.12} ref {enuc_ref:.12} |d| {:.2e}",
        (enuc - enuc_ref).abs()
    );
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

/// Open-shell config: `solve_uhf` reads `df_j_aux` verbatim (no auto-default),
/// so RI-J is requested explicitly to match PySCF `density_fit`.
fn uks_config(xc: &str) -> RhfConfig {
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

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<10} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

/// ferric's energy must MISS a reference computed with a different
/// functional, basis or Coulomb route.
fn assert_misses(ctx: &str, what: &str, e: f64, e_wrong: f64) {
    let d = (e - e_wrong).abs();
    let must = MUST_MISS_FACTOR * TOL_E;
    eprintln!("{ctx}: NEGATIVE CONTROL vs {what}: |d| {d:.3e} (must be > {must:.0e})");
    assert!(
        d > must,
        "{ctx}: ferric energy {e:.12} is within {must:.0e} of the {what} reference \
         {e_wrong:.12} — the comparison cannot discriminate"
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

/// One system × functional: both bases × both J routes, with the functional-,
/// J-, basis-swap and RHF controls.
fn run_rks(system: &str, xc: &str) {
    for basis_name in BASES {
        let r = reference(system, basis_name);
        let r_other = reference(system, other_basis(basis_name));
        let sys = load_system(system, basis_name, &r);
        let nocc = (sys.mol.nelec() as usize) / 2;
        for j in J_MODES {
            let ctx = format!("{} RKS {xc} {j}-J", sys.label);
            let res = solve_rhf(
                &sys.ctx,
                &sys.mol,
                &sys.prep,
                Operator::coulomb(),
                &sys.bounds,
                &rks_config(xc, j),
            )
            .unwrap_or_else(|e| panic!("{ctx}: SCF failed: {e:?}"));
            assert!(res.converged, "{ctx}: SCF did not converge");
            assert_j_route(&res, j, &ctx);

            let key = format!("/rks_{xc}_{j}");
            check_close(
                &ctx,
                "energy",
                res.energy,
                num(&r, &format!("{key}/energy"), &ctx),
                TOL_E,
            );
            check_close(
                &ctx,
                "HOMO",
                res.eps_alpha[nocc - 1],
                num(&r, &format!("{key}/homo"), &ctx),
                TOL_EPS,
            );
            check_close(
                &ctx,
                "LUMO",
                res.eps_alpha[nocc],
                num(&r, &format!("{key}/lumo"), &ctx),
                TOL_EPS,
            );

            // Negative controls.
            let xc_swap = other_xc(xc);
            assert_misses(
                &ctx,
                &format!("{xc_swap} {j}-J"),
                res.energy,
                num(&r, &format!("/rks_{xc_swap}_{j}/energy"), &ctx),
            );
            let j_swap = if j == "rij" { "exact" } else { "rij" };
            assert_misses(
                &ctx,
                &format!("{xc} {j_swap}-J"),
                res.energy,
                num(&r, &format!("/rks_{xc}_{j_swap}/energy"), &ctx),
            );
            assert_misses(
                &ctx,
                other_basis(basis_name),
                res.energy,
                num(&r_other, &format!("{key}/energy"), &ctx),
            );
            let e_rhf = num(&r, "/rhf_control/energy", &ctx);
            assert!(
                (res.energy - e_rhf).abs() > CONTROL_GAP,
                "{ctx}: RKS {:.10} is within {CONTROL_GAP:.0e} of the exact RHF control \
                 {e_rhf:.10} — the functional is not being applied",
                res.energy
            );
        }
    }
}

#[test]
#[ignore = "validation: SCAN/r2SCAN energy"]
fn rks_h2o_scan_vs_pyscf() {
    run_rks("h2o", "scan");
}

#[test]
#[ignore = "validation: SCAN/r2SCAN energy"]
fn rks_h2o_r2scan_vs_pyscf() {
    run_rks("h2o", "r2scan");
}

#[test]
#[ignore = "validation: SCAN/r2SCAN energy"]
fn rks_nh3_scan_vs_pyscf() {
    run_rks("nh3", "scan");
}

#[test]
#[ignore = "validation: SCAN/r2SCAN energy"]
fn rks_nh3_r2scan_vs_pyscf() {
    run_rks("nh3", "r2scan");
}

#[test]
#[ignore = "validation: SCAN/r2SCAN energy"]
fn rks_h2s_scan_vs_pyscf() {
    run_rks("h2s", "scan");
}

#[test]
#[ignore = "validation: SCAN/r2SCAN energy"]
fn rks_h2s_r2scan_vs_pyscf() {
    run_rks("h2s", "r2scan");
}

#[test]
#[ignore = "validation: SCAN/r2SCAN energy"]
fn rks_ch4_scan_vs_pyscf() {
    run_rks("ch4", "scan");
}

#[test]
#[ignore = "validation: SCAN/r2SCAN energy"]
fn rks_ch4_r2scan_vs_pyscf() {
    run_rks("ch4", "r2scan");
}

// ── 2. Open shell, UKS ──────────────────────────────────────────────────────

fn run_uks(system: &str, xc: &str) {
    for basis_name in BASES {
        let r = reference(system, basis_name);
        let r_other = reference(system, other_basis(basis_name));
        let sys = load_system(system, basis_name, &r);
        let (na, nb) = nocc_ab(&sys.mol);
        let s = ferric_integrals::oneelectron::overlap(&sys.prep);
        let ctx = format!("{} UKS {xc}", sys.label);
        let key = format!("/uks_{xc}_rij");
        assert_eq!(
            r.pointer(&format!("{key}/stability/internal_stable"))
                .and_then(Value::as_bool),
            Some(true),
            "{ctx}: reference state is not recorded as internally stable"
        );
        assert_eq!(
            r.pointer(&format!("{key}/multiple_stable_minima"))
                .and_then(Value::as_bool),
            Some(false),
            "{ctx}: reference has several stable minima; the energy would test state selection"
        );
        let res = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &uks_config(xc))
            .unwrap_or_else(|e| panic!("{ctx}: SCF failed: {e:?}"));
        assert!(res.converged, "{ctx}: SCF did not converge");
        assert!(
            matches!(res.spin, Spin::Unrestricted),
            "{ctx}: not an unrestricted result"
        );
        assert_j_route(&res, "rij", &ctx);

        // State first: ⟨S²⟩ (ferric skips meta-GGA stability analysis,
        // StabilitySkip::MetaGga, so the state is pinned by matching the
        // stability-followed reference).
        let s2 = s_squared(&res, &s, na, nb);
        check_close(
            &ctx,
            "<S2>",
            s2,
            num(&r, &format!("{key}/s_squared"), &ctx),
            TOL_S2,
        );
        check_close(
            &ctx,
            "energy",
            res.energy,
            num(&r, &format!("{key}/energy"), &ctx),
            TOL_E,
        );
        let eb = res.eps_beta.as_ref().expect("UKS beta orbital energies");
        for (what, got, field) in [
            ("HOMO a", res.eps_alpha[na - 1], "homo_alpha"),
            ("LUMO a", res.eps_alpha[na], "lumo_alpha"),
            ("HOMO b", eb[nb - 1], "homo_beta"),
            ("LUMO b", eb[nb], "lumo_beta"),
        ] {
            check_close(
                &ctx,
                what,
                got,
                num(&r, &format!("{key}/{field}"), &ctx),
                TOL_EPS,
            );
        }

        // Negative controls.
        let xc_swap = other_xc(xc);
        assert_misses(
            &ctx,
            xc_swap,
            res.energy,
            num(&r, &format!("/uks_{xc_swap}_rij/energy"), &ctx),
        );
        assert_misses(
            &ctx,
            other_basis(basis_name),
            res.energy,
            num(&r_other, &format!("{key}/energy"), &ctx),
        );
        let e_uhf = num(&r, "/uhf_control/energy", &ctx);
        assert!(
            (res.energy - e_uhf).abs() > CONTROL_GAP,
            "{ctx}: UKS {:.10} is within {CONTROL_GAP:.0e} of the exact UHF control \
             {e_uhf:.10} — the functional is not being applied",
            res.energy
        );
    }
}

#[test]
#[ignore = "validation: SCAN/r2SCAN energy"]
fn uks_nh2_scan_vs_pyscf() {
    run_uks("nh2", "scan");
}

#[test]
#[ignore = "validation: SCAN/r2SCAN energy"]
fn uks_nh2_r2scan_vs_pyscf() {
    run_uks("nh2", "r2scan");
}
