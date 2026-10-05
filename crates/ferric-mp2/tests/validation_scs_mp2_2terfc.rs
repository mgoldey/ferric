//! VALIDATION tier — row "SCS-MP2(2terfc)", energy half.
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_scs_mp2_2terfc \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! `ferric_mp2::rimp2::ri_mp2_spin_components(.., Operator::terfc(r₀), ..)`
//! (E_OS, E_SS, E_total at r₀ = 0.75, 1.00, 1.05 Å) and
//! `ferric_mp2::scs::scs_mp2_2terfc` with its DEFAULT configuration (the
//! published r₀ = 0.75/1.05 Å, c_OS = 1.27, c_SS = 4.05) against a numpy
//! RI-MP2 on Fourier-space terfc integrals
//! (`scripts/validation/gen_terfc_integrals.py`; the integral half of this
//! row, with the construction and its anchors, is
//! `crates/ferric-integrals/tests/validation_terfc_integrals.rs`). The numpy
//! side factorizes exactly as ferric does: metric = the same terfc operator
//! (`RiMp2Config::metric_op = None`), Cholesky, B = L⁻¹(Q|ia), PySCF
//! exact-J/K RHF orbitals, canonical denominators; it is anchored at the
//! Coulomb kernel against PySCF `dfmp2.DFMP2` before any terfc energy is
//! written. H2O and NH3 / cc-pVDZ + cc-pVDZ-RI, H2O / aug-cc-pVDZ +
//! aug-cc-pVDZ-RIFIT, frozen core 0 and 1 (the heavy-atom count).
//!
//! The published parameters are fitted for aug-cc-pVTZ with frozen core and
//! no counterpoise. This row validates the OPERATOR and the ASSEMBLY of
//! Eq. 12 at these smaller bases, not the fit: S66 is not run here.
//!
//! Exactness anchor: terfc → Coulomb as r₀ → ∞, at O(1/r₀) (not
//! exponentially; `terfc_base_validation.rs`), so the anchor asserts the
//! TREND — the gap to Coulomb RI-MP2 halves when r₀ doubles — not equality.
//!
//! # Physics hypothesis vs artifact hypothesis (Experimental Protocol)
//!
//! * If ferric is right: every E_OS/E_SS and the Eq. 12 total agree with the
//!   reference at the integral floor propagated through RI-MP2.
//! * A shared MD error in ferric's integrals shows up in p/d/f classes and not
//!   in ss (see the integral half); here it moves the energy by far more than
//!   the bar. A Coulomb fitting metric moves it by ~the RI error (1e-5 Ha).
//! * A mistranslated Å→Bohr in the defaults, a dropped SS difference in
//!   Eq. 12, or a swapped r₀ moves the total by mHa.
//! * Harness: nuclear repulsion, AO/aux counts and the RHF energy fail first.
//!
//! # TOLERANCES
//!
//! Each bar is ~10× the measured maximum recorded on its const.
//!
//! # NEGATIVE CONTROLS
//!
//! Always-on: ferric terfc at r₀ = 0.75 Å must MISS the 1.05 Å reference, and
//! ferric erfc at ω = 1/(r₀√2) must MISS the terfc(r₀) reference, each by
//! ≥ [`MUST_MISS`]. The mutation ledger is at the end of this file.
//!
//! A missing JSON or a missing table directory is a HARD failure.

use std::path::{Path, PathBuf};
use std::sync::Once;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config, SpinComponents};
use ferric_mp2::scs::{scs_mp2_2terfc, ScsMp2TerfcConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/scs_mp2_2terfc";
const MOL_DIR: &str = "testdata/molecules/validation";
const CASES: &[(&str, &str, &str)] = &[
    ("h2o", "cc-pvdz", "cc-pvdz-ri"),
    ("nh3", "cc-pvdz", "cc-pvdz-ri"),
    ("h2o", "aug-cc-pvdz", "aug-cc-pvdz-rifit"),
];

/// RHF total energy vs PySCF. Measured max: TBD.
const TOL_RHF: f64 = 1e-10;
/// E_OS, E_SS, E_total per r₀ and the Eq. 12 total. Measured max: TBD.
const TOL_CORR: f64 = 1e-9;
const TOL_ENUC: f64 = 1e-9;
const MUST_MISS: f64 = 1000.0 * TOL_CORR;

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
        .expect("manifest dir should be <root>/crates/ferric-mp2")
        .to_path_buf()
}

static TABLES: Once = Once::new();

/// The table engine reads `FERRIC_TERF_TABLE_DIR` only. Point it at the
/// committed `terf-tables/` when unset (the CI validation job does not set
/// it), and FAIL — never skip — when no table set can be found.
fn ensure_tables() {
    TABLES.call_once(|| {
        let probe = "16_4_2.bin";
        if let Ok(d) = std::env::var("FERRIC_TERF_TABLE_DIR") {
            assert!(
                Path::new(&d).join(probe).is_file(),
                "FERRIC_TERF_TABLE_DIR={d} has no {probe}; a validation row never skips"
            );
            return;
        }
        let repo = workspace_root().join("terf-tables");
        assert!(
            repo.join(probe).is_file(),
            "terfc tables not found: FERRIC_TERF_TABLE_DIR is unset and {} has no {probe}",
            repo.display()
        );
        std::env::set_var("FERRIC_TERF_TABLE_DIR", &repo);
    });
}

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_terfc_integrals.py — a missing reference is a failure",
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

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) -> f64 {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<10} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
    d
}

fn check_misses(ctx: &str, what: &str, got: f64, other: f64) {
    let d = (got - other).abs();
    eprintln!("{ctx}: resolving power vs {what}: |d| {d:.2e} (must be >= {MUST_MISS:.0e})");
    assert!(
        d >= MUST_MISS,
        "{ctx}: ferric's energy is within {d:.2e} of the {what} reference"
    );
}

struct System {
    r: Value,
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn load_system(system: &str, basis_name: &str, auxbasis: &str) -> System {
    ensure_tables();
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz(xyz.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let obs = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(auxbasis).unwrap()).unwrap();
    assert_eq!(
        obs.nbasis(),
        r["nao"].as_u64().unwrap() as usize,
        "{ctx}: nao"
    );
    assert_eq!(
        dfbs.nbasis(),
        r["naux"].as_u64().unwrap() as usize,
        "{ctx}: naux"
    );
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        max_iter: 200,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{ctx}: RHF failed: {e:?}"));
    assert!(rhf.converged, "{ctx}: RHF not converged");
    check_close(
        &ctx,
        "E_RHF",
        rhf.energy,
        num(&r, "/rhf_energy", &ctx),
        TOL_RHF,
    );
    System {
        r,
        mol,
        obs,
        dfbs,
        rhf,
    }
}

fn spin(sys: &System, op: Operator, frozen_core: usize) -> SpinComponents {
    let cfg = RiMp2Config {
        frozen_core,
        ..Default::default()
    };
    ri_mp2_spin_components(&sys.mol, &sys.obs, &sys.dfbs, op, &sys.rhf, &cfg)
        .unwrap_or_else(|e| panic!("ri_mp2_spin_components failed: {e:?}"))
        .0
}

fn check_system(system: &str, basis_name: &str, auxbasis: &str) {
    let sys = load_system(system, basis_name, auxbasis);
    let blocks = sys.r["energies"].as_array().expect("energies").clone();
    let fcs: Vec<u64> = blocks
        .iter()
        .map(|b| b["frozen_core"].as_u64().unwrap())
        .collect();
    assert_eq!(
        fcs,
        vec![0, sys.r["n_heavy"].as_u64().unwrap()],
        "{system}: frozen-core set"
    );
    for b in &blocks {
        let fc = b["frozen_core"].as_u64().unwrap() as usize;
        for p in b["per_r0"].as_array().unwrap() {
            let r0 = p["r0_bohr"].as_f64().unwrap();
            let ctx = format!(
                "{system}/{basis_name}/fc={fc}/r0={}A",
                p["r0_angstrom"].as_f64().unwrap()
            );
            let sc = spin(&sys, Operator::terfc(r0), fc);
            check_close(&ctx, "E_OS", sc.e_os, num(p, "/e_os", &ctx), TOL_CORR);
            check_close(&ctx, "E_SS", sc.e_ss, num(p, "/e_ss", &ctx), TOL_CORR);
            check_close(
                &ctx,
                "E_total",
                sc.e_total,
                num(p, "/e_total", &ctx),
                TOL_CORR,
            );
        }
        // Eq. 12 through the production entry point with its DEFAULT config:
        // the published r0 (in Angstrom, converted by scs.rs), c_OS and c_SS.
        let ctx = format!("{system}/{basis_name}/fc={fc}/scs-2terfc");
        let s = &b["scs_2terfc"];
        let cfg = ScsMp2TerfcConfig {
            frozen_core: fc,
            ..Default::default()
        };
        assert_eq!(cfg.c_os, num(s, "/c_os", &ctx), "{ctx}: c_OS");
        assert_eq!(cfg.c_ss, num(s, "/c_ss", &ctx), "{ctx}: c_SS");
        let res = scs_mp2_2terfc(&sys.mol, &sys.obs, &sys.dfbs, &sys.rhf, &cfg)
            .unwrap_or_else(|e| panic!("{ctx}: scs_mp2_2terfc failed: {e:?}"));
        check_close(
            &ctx,
            "scs_corr",
            res.scs_corr,
            num(s, "/scs_corr", &ctx),
            TOL_CORR,
        );
        check_close(
            &ctx,
            "E_total",
            res.total_energy,
            num(s, "/total_energy", &ctx),
            TOL_CORR + TOL_RHF,
        );
    }
}

#[test]
#[ignore = "validation: SCS-MP2(2terfc)"]
fn scs_mp2_2terfc_h2o_cc_pvdz_vs_kspace() {
    check_system("h2o", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: SCS-MP2(2terfc)"]
fn scs_mp2_2terfc_nh3_cc_pvdz_vs_kspace() {
    check_system("nh3", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: SCS-MP2(2terfc)"]
fn scs_mp2_2terfc_h2o_aug_cc_pvdz_vs_kspace() {
    check_system("h2o", "aug-cc-pvdz", "aug-cc-pvdz-rifit");
}

/// terfc → Coulomb as r₀ → ∞ at O(1/r₀): the gap to Coulomb RI-MP2 must
/// shrink, and roughly halve, when r₀ doubles. ferric vs itself; the Coulomb
/// end is the attenuated-MP2 row's anchor.
#[test]
#[ignore = "validation: SCS-MP2(2terfc)"]
fn terfc_ri_mp2_approaches_coulomb_as_r0_grows() {
    let sys = load_system("h2o", "cc-pvdz", "cc-pvdz-ri");
    let coul = spin(&sys, Operator::coulomb(), 0).e_total;
    let gaps: Vec<f64> = [25.0, 50.0, 100.0]
        .iter()
        .map(|&r0| (spin(&sys, Operator::terfc(r0), 0).e_total - coul).abs())
        .collect();
    eprintln!(
        "h2o/cc-pvdz: |E(terfc r0) - E(Coulomb)| at r0 = 25, 50, 100 Bohr: {:?}",
        gaps.iter().map(|g| format!("{g:.3e}")).collect::<Vec<_>>()
    );
    for w in gaps.windows(2) {
        let ratio = w[0] / w[1];
        assert!(
            (1.6..2.4).contains(&ratio),
            "gap ratio on doubling r0 is {ratio:.3}, expected ~2 (O(1/r0))"
        );
    }
}

/// Resolving power: r₀ = 0.75 Å vs 1.05 Å, and terfc vs erfc at the same
/// curvature, must both be visible far above the bar.
#[test]
#[ignore = "validation: SCS-MP2(2terfc)"]
fn scs_mp2_2terfc_negative_controls() {
    for &(system, basis_name, auxbasis) in CASES {
        let sys = load_system(system, basis_name, auxbasis);
        let b = &sys.r["energies"][0];
        let per = b["per_r0"].as_array().unwrap();
        let first = &per[0];
        let last = &per[per.len() - 1];
        let r0_1 = first["r0_bohr"].as_f64().unwrap();
        let ctx = format!("{system}/{basis_name}");
        let e1 = spin(&sys, Operator::terfc(r0_1), 0).e_total;
        check_misses(&ctx, "terfc(r0_2)", e1, num(last, "/e_total", &ctx));
        let w = 1.0 / (r0_1 * std::f64::consts::SQRT_2);
        let ee = spin(&sys, Operator::erfc(w), 0).e_total;
        check_misses(&ctx, "terfc(r0_1) [erfc]", ee, num(first, "/e_total", &ctx));
    }
}

// MUTATION LEDGER: TBD
