//! VALIDATION tier — VALIDATION.md row "Attenuated RI-MP2".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_attenuated_mp2 \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! `ferric_mp2::attenuated::attenuated_ri_mp2` (erfc(ωr)/r on BOTH the
//! 3-center `(P|μν)` and the 2-center metric `(P|Q)`, standard same-kernel fit,
//! full-Coulomb RHF reference, no frozen core) against a numpy assembly on
//! PySCF integrals built under `Mole.with_range_coulomb(-ω)` on the orbital AND
//! auxiliary Moles — the identical construction. H2O and NH3, aug-cc-pVDZ with
//! aug-cc-pVDZ-RIFIT, ω ∈ {0.2, 0.222254 (= 0.42 Å⁻¹, the library default),
//! 0.42, 1.0} Bohr⁻¹. The library takes ω in Bohr⁻¹; only the CLI and Python
//! convert from Å⁻¹. References: `scripts/validation/gen_attenuated_mp2.py`,
//! `testdata/reference/validation/attenuated_mp2/<system>_aug-cc-pvdz.json`.
//!
//! The generator asserts, before writing anything, that erfc + erf == Coulomb
//! for its int3c2e/int2c2e blocks (the sign convention picked erfc on both
//! Moles) and that its ω = 0 numpy assembly equals PySCF's own `DFMP2` with the
//! same aux basis. It also records the exact 4-index erfc MP2 (no RI) as
//! information: the RI error of the attenuated fit is 1e-5–3e-5 Ha here, far
//! above this row's bar, so a reference that silently used exact integrals
//! would fail.
//!
//! Exactness anchor: erfc(0) = 1, so at ω = 1e-8 attenuated RI-MP2 must equal
//! ferric's own Coulomb `ri_mp2` ([`attenuated_matches_coulomb_ri_mp2_in_the_trivial_limit`]),
//! which is in turn checked against the reference's Coulomb block.
//!
//! # Physics hypothesis vs artifact hypothesis (Experimental Protocol)
//!
//! * If ferric is right: E_corr, E_os and E_ss agree with the reference at the
//!   RHF/integral floor (~1e-9 Ha) at every ω, and the ω-dependence (0.22 →
//!   0.07 Ha over the grid) is reproduced point by point.
//! * If ferric attenuated only the 3-center integrals and kept a Coulomb metric
//!   (the robust-fit variant): the energy moves by the fit difference, which is
//!   of the order of the RI error itself (1e-5 Ha) — 1000× the bar.
//! * If ω were interpreted in Å⁻¹ where Bohr⁻¹ is meant: the ω = 0.222 and
//!   ω = 0.42 points are swapped — a 0.03 Ha miss; asserted as an always-on
//!   negative control below.
//! * If the harness is broken (geometry, basis, aux basis): nuclear repulsion,
//!   AO/aux counts and the RHF energy fail first.
//!
//! # TOLERANCES
//!
//! Each bar is set from the measured maximum recorded on its const:
//! correlation energies 5.9e-12 Ha across ω ∈ {0.2, 0.222254, 0.42, 1.0}.
//!
//! # NEGATIVE CONTROLS / MUTATIONS
//!
//! * Always-on: ferric at ω = 0.42 Bohr⁻¹ must MISS the ω = 0.222254
//!   reference by ≥ [`MUST_MISS`] (unit-slip control), and ferric at every ω
//!   must MISS the Coulomb reference.
//! * MUTATION A — in `attenuated_ri_mp2` pass `RiMp2Config { metric_op:
//!   Some(Operator::coulomb()), .. }`: every attenuated assertion fails.
//! * MUTATION B — build the operator as `Operator::erf(config.omega)`: every
//!   attenuated assertion fails (erf picks up the long-range complement).
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::attenuated::{attenuated_ri_mp2, AttenuatedMp2Config};
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/attenuated_mp2";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "aug-cc-pvdz";
const AUXBASIS: &str = "aug-cc-pvdz-rifit";

/// RHF total energy vs PySCF. Measured max 2.3e-13.
const TOL_RHF: f64 = 1e-10;
/// Correlation energies (total, OS, SS). Measured max 5.9e-12.
const TOL_CORR: f64 = 1e-10;
/// ω = 1e-8 anchor: attenuated vs Coulomb ri_mp2 inside ferric. The erf
/// complement at ω is ~2ω/√π·S_ia·S_jb = 0 for orthogonal MOs, but the metric
/// picks up a rank-1 O(ω) shift. Measured 1.6e-15.
const TOL_ANCHOR: f64 = 1e-12;
const ANCHOR_OMEGA: f64 = 1e-8;
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
        .expect("ferric-mp2 manifest dir should be <root>/crates/ferric-mp2")
        .to_path_buf()
}

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_attenuated_mp2.py — a missing reference is a failure, never a skip",
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

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<10} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_misses(ctx: &str, what: &str, got: f64, other: f64) {
    let d = (got - other).abs();
    eprintln!("{ctx}: resolving power vs {what}: |d| {d:.2e} (must be >= {MUST_MISS:.0e})");
    assert!(
        d >= MUST_MISS,
        "{ctx}: ferric's E_corr is within {d:.2e} of the {what} reference"
    );
}

struct System {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn load_system(system: &str, r: &Value) -> System {
    let ctx = format!("{system}/{BASIS}");
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz(xyz.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let obs = PreparedBasis::new(&mol, &basis::bundled(BASIS).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUXBASIS).unwrap()).unwrap();
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
        num(r, "/rhf_energy", &ctx),
        TOL_RHF,
    );
    System {
        mol,
        obs,
        dfbs,
        rhf,
    }
}

fn att(sys: &System, omega: f64) -> ferric_mp2::attenuated::AttenuatedMp2Result {
    let cfg = AttenuatedMp2Config {
        omega,
        ..Default::default()
    };
    attenuated_ri_mp2(&sys.mol, &sys.obs, &sys.dfbs, &sys.rhf, &cfg)
        .unwrap_or_else(|e| panic!("attenuated_ri_mp2(omega={omega}) failed: {e:?}"))
}

fn check_system(system: &str) {
    let r = reference(system);
    let sys = load_system(system, &r);
    let e_coul_ref = num(&r, "/coulomb/e_corr", system);

    // Default omega is 0.420 A^-1 converted with ferric's constant; the
    // reference's second grid point is that same number.
    let default_omega = AttenuatedMp2Config::default().omega;
    let blocks = r["attenuated"].as_array().expect("attenuated blocks");
    assert!(blocks.len() >= 3, "{system}: need >= 3 omega points");
    let mut e_by_omega = Vec::new();
    for b in blocks {
        let omega = b["omega_bohr_inv"].as_f64().unwrap();
        let ctx = format!("{system}/{BASIS}/omega={omega}");
        let res = att(&sys, omega);
        check_close(
            &ctx,
            "E_corr",
            res.mp2_corr,
            num(b, "/e_corr", &ctx),
            TOL_CORR,
        );
        check_close(
            &ctx,
            "E_os",
            res.spin_components.e_os,
            num(b, "/e_os", &ctx),
            TOL_CORR,
        );
        check_close(
            &ctx,
            "E_ss",
            res.spin_components.e_ss,
            num(b, "/e_ss", &ctx),
            TOL_CORR,
        );
        eprintln!(
            "{ctx}: RI error of the reference fit (info): {:+.2e}",
            num(b, "/ri_error_info", &ctx)
        );
        check_misses(&ctx, "Coulomb", res.mp2_corr, e_coul_ref);
        e_by_omega.push((omega, res.mp2_corr));
    }

    // Unit-slip control: 0.42 Bohr^-1 (what "0.42" means if the A^-1 value
    // is passed raw) must miss the default-omega (0.42 A^-1) reference.
    let def_ref = blocks
        .iter()
        .find(|b| (b["omega_bohr_inv"].as_f64().unwrap() - default_omega).abs() < 1e-12)
        .unwrap_or_else(|| panic!("{system}: reference lacks the default omega {default_omega}"));
    let e_042 = e_by_omega
        .iter()
        .find(|(w, _)| (*w - 0.42).abs() < 1e-12)
        .map(|&(_, e)| e)
        .unwrap_or_else(|| panic!("{system}: reference lacks omega = 0.42 Bohr^-1"));
    check_misses(
        &format!("{system}/unit-slip"),
        "omega=0.42 A^-1",
        e_042,
        def_ref["e_corr"].as_f64().unwrap(),
    );
}

/// Exactness anchor: erfc(0·r) = 1, so ω → 0 must reproduce Coulomb RI-MP2 —
/// ferric vs itself, and ferric's Coulomb RI-MP2 vs the reference.
#[test]
#[ignore = "validation: Attenuated RI-MP2"]
fn attenuated_matches_coulomb_ri_mp2_in_the_trivial_limit() {
    for system in ["h2o", "nh3"] {
        let r = reference(system);
        let sys = load_system(system, &r);
        let ctx = format!("{system}/{BASIS}/anchor");
        let (coul, _) = ri_mp2_spin_components(
            &sys.mol,
            &sys.obs,
            &sys.dfbs,
            Operator::coulomb(),
            &sys.rhf,
            &RiMp2Config::default(),
        )
        .unwrap();
        check_close(
            &ctx,
            "E_coul",
            coul.e_total,
            num(&r, "/coulomb/e_corr", &ctx),
            TOL_CORR,
        );
        let a = att(&sys, ANCHOR_OMEGA);
        check_close(&ctx, "E(w->0)", a.mp2_corr, coul.e_total, TOL_ANCHOR);
        check_close(
            &ctx,
            "E_os(w->0)",
            a.spin_components.e_os,
            coul.e_os,
            TOL_ANCHOR,
        );
    }
}

#[test]
#[ignore = "validation: Attenuated RI-MP2"]
fn attenuated_ri_mp2_h2o_vs_pyscf() {
    check_system("h2o");
}

#[test]
#[ignore = "validation: Attenuated RI-MP2"]
fn attenuated_ri_mp2_nh3_vs_pyscf() {
    check_system("nh3");
}
