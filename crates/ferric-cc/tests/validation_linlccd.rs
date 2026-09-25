//! VALIDATION tier — VALIDATION.md row "LinLCCD(hh)".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-cc --test validation_linlccd \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's LinLCCD(hh) (Carter-Fenk, JPCA 129, 7251 (2025), eq. 14: driver +
//! canonical Fock diagonal + hole–hole ladder only), closed shell through
//! `linlccd::linlccd` and open shell through `linlccd_u::u_linlccd`, all
//! electrons correlated, cc-pVDZ-RI Coulomb-metric fitting, against an
//! independent numpy reference on PySCF density-fitted integrals
//! (`scripts/validation/gen_linlccd.py` →
//! `testdata/reference/validation/linlccd/<system>_<basis>.json`):
//!
//! | system | reference SCF | bases |
//! |---|---|---|
//! | H2O | RHF | 6-31G, cc-pVDZ |
//! | NH3 | RHF | 6-31G, cc-pVDZ |
//! | OH ²Π | UHF, stability-checked (3 guesses, stable state kept) | 6-31G, cc-pVDZ |
//!
//! The reference does not iterate: the hh equation is linear and, per (a,b),
//! couples only occupied pairs, so it is solved EXACTLY in the eigenbasis of
//! the occupied-pair matrix (the paper's eq. 15 form). ferric iterates Jacobi +
//! DIIS on a spin-orbital tensor. For the closed-shell systems the generator
//! also builds a spin-adapted (spatial-orbital) solution and a spin-orbital
//! one from explicitly spin-blocked MO coefficients and refuses to write unless
//! they agree to 1e-12 (measured ≤ 2e-16); the spin-orbital construction is
//! the one used for OH.
//!
//! # Exactness anchors (asserted first, per system)
//!
//! E_nuc (geometry), nao/naux (basis), the SCF energy vs PySCF (for OH: the
//! same stable UHF state; ferric runs `check_stability` + descent and must not
//! end UNSTABLE — OH's degenerate π rotation is a zero mode, so MARGINAL is
//! accepted and the energy decides the state), then LADDER OFF:
//! `LadderVariant::DriversOnly` must equal the reference MP2, which the
//! generator proved equal to PySCF `DFMP2`/`DFUMP2` to 1e-12 (measured
//! ≤ 6e-16), and — closed shell — ferric's own `ri_mp2_spin_components` to
//! [`TOL_SELF`].
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's hh ladder is right: E_hh agrees with the reference to the
//!   RI/SCF + DIIS-convergence floor (~1e-11) on all six inputs.
//! * If the ladder has a wrong factor (1 instead of ½), sign, or index order
//!   (<ij||kl> vs <kl||ij> is harmless for real orbitals; <ki||lj> is not):
//!   the error is a fraction of E_hh − E_MP2 = +1.5e-2 .. +3.0e-2 Eh here —
//!   eight orders above the bar. The MP2 anchor cannot see any of these.
//! * If ferric's open-shell SCF lands on a different state (the 2026-09-17
//!   guess-defect class) the UHF energy anchor fails first, by mHa or more.
//! * If the HARNESS is broken: E_nuc / nao / naux / E_SCF / MP2 anchors fail
//!   before the ladder is examined.
//!
//! # TOLERANCES
//!
//! Each bar is set from the measured maximum recorded on its const.
//!
//! # NEGATIVE CONTROLS (asserted in-test, always on)
//!
//! * LADDER SWAP: ferric Hh must MISS the reference MP2, and ferric
//!   DriversOnly must MISS the reference hh, by more than [`MUST_MISS`].
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_cc::linlccd::{linlccd, LadderVariant};
use ferric_cc::linlccd_u::u_linlccd;
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{ScfResult, Spin};
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/linlccd";
const MOL_DIR: &str = "testdata/molecules/validation";
const AUX: &str = "cc-pvdz-ri";

/// Geometry check (PySCF E_nuc from ferric's Bohr geometry).
const TOL_ENUC: f64 = 1e-9;
/// SCF energy vs PySCF, exact J/K on both sides.
// Measured max 7.3e-12.
const TOL_E_SCF: f64 = 1e-9;
/// MP2 (DriversOnly) and LinLCCD(hh) correlation energy vs the reference.
// Measured max 1.2e-12 (OH UHF, cc-pVDZ); closed shell <= 4.4e-13.
const TOL_E_CORR: f64 = 1e-11;
/// ferric DriversOnly vs ferric's own RI-MP2 (same integrals, different
/// code path: spin-orbital einsum vs i-blocked spatial GEMMs).
// Measured max 1.1e-16.
const TOL_SELF: f64 = 1e-11;
/// A reference ferric must MISS (negative controls). |E_hh − E_MP2| is
/// 1.5e-2 .. 3.0e-2 Eh on these inputs (JSON `hh_minus_mp2`).
const MUST_MISS: f64 = 1e-6;

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
        .expect("ferric-cc manifest dir should be <root>/crates/ferric-cc")
        .to_path_buf()
}

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_linlccd.py — a missing reference is a failure, never a skip",
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
    eprintln!("{ctx}: {what:<24} ferric {got:+.13} ref {want:+.13} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.13} vs reference {want:.13} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_miss(ctx: &str, what: &str, got: f64, want: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<24} |d| {d:.2e} (must exceed {MUST_MISS:.0e})");
    assert!(
        d > MUST_MISS,
        "{ctx}: negative control {what}: |d| {d:.2e} <= {MUST_MISS:.0e} — the bar cannot \
         tell these apart"
    );
}

fn cc_config() -> CcConfig {
    let cfg = CcConfig {
        energy_conv: 1e-12,
        max_iter: 200,
        ..Default::default()
    };
    assert_eq!(cfg.frozen_core, 0, "all-electron expected");
    cfg
}

fn scf_config(open_shell: bool) -> RhfConfig {
    let cfg = RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-10,
        check_stability: open_shell,
        scf_stability_descent: open_shell,
        ..Default::default()
    };
    assert!(
        cfg.df_j_aux.is_none() && cfg.df_k_aux.is_none(),
        "exact J/K expected"
    );
    cfg
}

fn check_system(system: &str, basis_name: &str) {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    assert_eq!(r["basis"].as_str(), Some(basis_name), "{ctx}: basis");
    assert_eq!(r["aux_basis"].as_str(), Some(AUX), "{ctx}: aux basis");
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let open_shell = mult != 1;

    // ---- exactness anchor: geometry, basis, SCF ----
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let obs = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
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

    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let pctx = ParallelContext::default();
    let scf: ScfResult = if open_shell {
        let res = solve_uhf(&pctx, &mol, &obs, &bounds, &scf_config(true))
            .unwrap_or_else(|e| panic!("{ctx}: solve_uhf failed: {e:?}"));
        assert!(matches!(res.spin, Spin::Unrestricted), "{ctx}: not UHF");
        let st = res
            .stability
            .as_ref()
            .unwrap_or_else(|| panic!("{ctx}: check_stability set but no verdict"));
        eprintln!("{ctx}: {}", st.summary());
        assert!(
            matches!(
                st.verdict(),
                StabilityVerdict::Stable | StabilityVerdict::Marginal
            ),
            "{ctx}: ferric's UHF state is not a minimum: {}",
            st.summary()
        );
        res
    } else {
        solve_rhf(&pctx, &mol, &obs, op, &bounds, &scf_config(false))
            .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"))
    };
    assert!(scf.converged, "{ctx}: SCF not converged");
    check_close(
        &ctx,
        "E_SCF",
        scf.energy,
        num(&r, "/e_scf", &ctx),
        TOL_E_SCF,
    );

    let cfg = cc_config();
    let run = |variant: LadderVariant| -> f64 {
        let res = if open_shell {
            u_linlccd(&mol, &obs, &dfbs, op, &scf, &cfg, variant)
        } else {
            linlccd(&mol, &obs, &dfbs, op, &scf, &cfg, variant)
        };
        res.unwrap_or_else(|e| panic!("{ctx}: LinLCCD {variant:?} failed: {e:?}"))
            .correlation_energy
    };

    // ---- anchor: ladder off == MP2 ----
    let ref_mp2 = num(&r, "/mp2/e_corr", &ctx);
    let ref_hh = num(&r, "/linlccd_hh/e_corr", &ctx);
    let e_drv = run(LadderVariant::DriversOnly);
    check_close(&ctx, "DriversOnly vs ref MP2", e_drv, ref_mp2, TOL_E_CORR);
    check_close(
        &ctx,
        "DriversOnly vs PySCF",
        e_drv,
        num(&r, "/mp2/pyscf_df_mp2_e_corr", &ctx),
        TOL_E_CORR,
    );
    if !open_shell {
        let own = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &scf, &RiMp2Config::default())
            .unwrap_or_else(|e| panic!("{ctx}: ri_mp2 failed: {e:?}"))
            .0
            .e_total;
        check_close(&ctx, "DriversOnly vs ferric MP2", e_drv, own, TOL_SELF);
    }

    // ---- the row: hh ladder ----
    let e_hh = run(LadderVariant::Hh);
    check_close(&ctx, "LinLCCD(hh)", e_hh, ref_hh, TOL_E_CORR);

    // ---- negative controls ----
    check_miss(&ctx, "Hh vs ref MP2", e_hh, ref_mp2);
    check_miss(&ctx, "DriversOnly vs ref hh", e_drv, ref_hh);
}

#[test]
#[ignore = "validation: LinLCCD(hh)"]
fn linlccd_hh_h2o_631g_vs_numpy() {
    check_system("h2o", "6-31g");
}

#[test]
#[ignore = "validation: LinLCCD(hh)"]
fn linlccd_hh_h2o_ccpvdz_vs_numpy() {
    check_system("h2o", "cc-pvdz");
}

#[test]
#[ignore = "validation: LinLCCD(hh)"]
fn linlccd_hh_nh3_631g_vs_numpy() {
    check_system("nh3", "6-31g");
}

#[test]
#[ignore = "validation: LinLCCD(hh)"]
fn linlccd_hh_nh3_ccpvdz_vs_numpy() {
    check_system("nh3", "cc-pvdz");
}

#[test]
#[ignore = "validation: LinLCCD(hh)"]
fn linlccd_hh_oh_uhf_631g_vs_numpy() {
    check_system("oh", "6-31g");
}

#[test]
#[ignore = "validation: LinLCCD(hh)"]
fn linlccd_hh_oh_uhf_ccpvdz_vs_numpy() {
    check_system("oh", "cc-pvdz");
}
