//! VALIDATION tier: the Anchors row "SCS-MP2 (RI, Coulomb)".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_scs_mp2 \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! `ferric_mp2::scs::scs_mp2` (the CLI `scs-mp2` kind and Python
//! `run_scs_mp2`) at its default coefficients against the Coulomb block of the
//! attenuated RI-MP2 references: a numpy RI-MP2 assembly on PySCF integrals
//! (exact-J/K RHF, same aux basis, Cholesky metric, no frozen core) whose
//! E_OS + E_SS equals PySCF's own `dfmp2.DFMP2` (`checks/
//! numpy_vs_pyscf_dfmp2_omega0_abs` in each file). H2O and NH3, aug-cc-pVDZ
//! with aug-cc-pVDZ-RIFIT. References: `scripts/validation/gen_attenuated_mp2.py`,
//! `testdata/reference/validation/attenuated_mp2/<system>_aug-cc-pvdz.json`.
//!
//! The SCS reference is formed here from the reference's components with
//! Grimme's published coefficients (J. Chem. Phys. 118, 9095 (2003)):
//! c_OS = 6/5, c_SS = 1/3, written as literals, not read from
//! `ScsMp2Config::default()`, so a changed default fails this test.
//!
//! Exactness anchor: with c_OS = c_SS = 1, SCS-MP2 is RI-MP2. The anchor test
//! checks that against ferric's own `ri_mp2_spin_components` and against the
//! reference's E_corr.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric is right: E_OS, E_SS and the SCS energy agree with the
//!   reference at the integral and RHF floor (5.9e-12 Ha on these files for the
//!   Coulomb components, measured by `validation_attenuated_mp2.rs`).
//! * If the spin split is wrong but the sum is right (exchange counted in the
//!   wrong component), RI-MP2 anchors cannot see it; E_OS, E_SS and E_SCS here
//!   miss by the size of the moved term (1e-2 Ha).
//! * If the coefficients are swapped or wrong, E_SCS misses by
//!   |Δc_OS·E_OS + Δc_SS·E_SS| (about 0.1 Ha for a swap).
//! * If the harness is broken (geometry, basis, aux basis), nuclear repulsion,
//!   the AO/aux counts and the RHF energy fail first.
//!
//! # TOLERANCES
//!
//! Components: 1e-10 Ha, the bar of `validation_attenuated_mp2.rs` on the same
//! reference files. E_SCS: |ΔE_SCS| ≤ c_OS·|ΔE_OS| + c_SS·|ΔE_SS|, so its bar
//! is (6/5 + 1/3)·1e-10 Ha.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_mp2::scs::{scs_mp2, ScsMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/attenuated_mp2";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "aug-cc-pvdz";
const AUXBASIS: &str = "aug-cc-pvdz-rifit";

/// Grimme's SCS-MP2 coefficients, J. Chem. Phys. 118, 9095 (2003).
const C_OS: f64 = 6.0 / 5.0;
const C_SS: f64 = 1.0 / 3.0;

const TOL_ENUC: f64 = 1e-9;
/// RHF total energy vs PySCF (measured 2.3e-13 on these files).
const TOL_RHF: f64 = 1e-10;
/// E_OS, E_SS and E_corr (see the module doc).
const TOL_COMPONENT: f64 = 1e-10;
/// E_SCS: the component bar propagated through the coefficients.
const TOL_SCS: f64 = (C_OS + C_SS) * TOL_COMPONENT;
/// SCS-MP2 must miss plain MP2 by far more than the bar.
const MUST_MISS: f64 = 1000.0 * TOL_SCS;

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
             scripts/validation/gen_attenuated_mp2.py",
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
    eprintln!("{ctx}: {what:<8} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.1e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.1e})"
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
    assert_eq!(r["auxbasis"].as_str(), Some(AUXBASIS), "{ctx}: aux basis");
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

fn check_system(system: &str) {
    let r = reference(system);
    let sys = load_system(system, &r);
    let ctx = format!("{system}/{BASIS}");
    let e_os_ref = num(&r, "/coulomb/e_os", &ctx);
    let e_ss_ref = num(&r, "/coulomb/e_ss", &ctx);
    let e_mp2_ref = num(&r, "/coulomb/e_corr", &ctx);
    let e_scs_ref = C_OS * e_os_ref + C_SS * e_ss_ref;

    let res = scs_mp2(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        &sys.rhf,
        &ScsMp2Config::default(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: scs_mp2 failed: {e:?}"));
    check_close(&ctx, "E_OS", res.e_os, e_os_ref, TOL_COMPONENT);
    check_close(&ctx, "E_SS", res.e_ss, e_ss_ref, TOL_COMPONENT);
    check_close(&ctx, "E_SCS", res.scs_corr, e_scs_ref, TOL_SCS);
    check_close(
        &ctx,
        "E_total",
        res.total_energy,
        sys.rhf.energy + e_scs_ref,
        TOL_SCS,
    );

    // Resolving power: SCS-MP2 is not plain MP2 on these systems.
    let d = (res.scs_corr - e_mp2_ref).abs();
    eprintln!("{ctx}: E_SCS vs plain MP2 |d| {d:.2e} (must be >= {MUST_MISS:.1e})");
    assert!(
        d >= MUST_MISS,
        "{ctx}: E_SCS is within {d:.2e} of plain MP2; the test cannot see the coefficients"
    );
}

/// Exactness anchor: c_OS = c_SS = 1 is RI-MP2, against ferric's own
/// `ri_mp2_spin_components` (same arithmetic, so bit-for-bit up to the final
/// sum) and against the reference's E_corr.
#[test]
#[ignore = "validation: SCS-MP2"]
fn scs_mp2_with_unit_coefficients_is_ri_mp2() {
    for system in ["h2o", "nh3"] {
        let r = reference(system);
        let sys = load_system(system, &r);
        let ctx = format!("{system}/{BASIS}/anchor");
        let unit = ScsMp2Config {
            c_os: 1.0,
            c_ss: 1.0,
            ..Default::default()
        };
        let res = scs_mp2(&sys.mol, &sys.obs, &sys.dfbs, &sys.rhf, &unit)
            .unwrap_or_else(|e| panic!("{ctx}: scs_mp2 failed: {e:?}"));
        let (sc, _) = ri_mp2_spin_components(
            &sys.mol,
            &sys.obs,
            &sys.dfbs,
            Operator::coulomb(),
            &sys.rhf,
            &RiMp2Config::default(),
        )
        .unwrap_or_else(|e| panic!("{ctx}: ri_mp2_spin_components failed: {e:?}"));
        check_close(&ctx, "vs RI-MP2", res.scs_corr, sc.e_total, 1e-14);
        check_close(
            &ctx,
            "vs ref",
            res.scs_corr,
            num(&r, "/coulomb/e_corr", &ctx),
            TOL_COMPONENT,
        );
    }
}

#[test]
#[ignore = "validation: SCS-MP2"]
fn scs_mp2_h2o_vs_numpy() {
    check_system("h2o");
}

#[test]
#[ignore = "validation: SCS-MP2"]
fn scs_mp2_nh3_vs_numpy() {
    check_system("nh3");
}
