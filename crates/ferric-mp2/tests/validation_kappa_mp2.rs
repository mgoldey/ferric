//! VALIDATION tier — VALIDATION.md row "κ-MP2".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_kappa_mp2 \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's κ-regularized RI-MP2 (`RiMp2Config::kappa`, Lee & Head-Gordon,
//! JCTC 14, 5203 (2018): every (i,a,j,b) term of both spin components damped
//! by `(1 − e^{−κΔ})²`, `Δ = ε_a + ε_b − ε_i − ε_j`) at κ ∈ {0.5, 1.1, 2.0}
//! Eh⁻¹ on H2O and CH4 at cc-pVDZ with cc-pVDZ-RI, all electrons correlated,
//! against an independent dense numpy evaluation of the same formula on PySCF
//! density-fitted integrals (`scripts/validation/gen_kappa_mp2.py` →
//! `testdata/reference/validation/kappa_mp2/<system>_cc-pvdz.json`). PySCF was
//! fed ferric's own orbital and aux basis JSON and ferric's geometry in Bohr;
//! its RHF is exact four-centre J/K, as is ferric's `RhfConfig::default()`.
//!
//! These are MANY-PAIR systems (9 025 and 21 025 (ia|jb) terms) and every κ is
//! INTERIOR: E(κ)/E(∞) is 0.76/0.96/0.997 (H2O) and 0.63/0.91/0.989 (CH4),
//! and the per-term damping spans 0.24–1.0 at κ = 0.5. The existing unit
//! tests pin only the two trivial limits and a one-pair analytic identity.
//!
//! # Exactness anchor (asserted first)
//!
//! E_nuc (geometry), AO/aux counts (basis), E_RHF (exact J/K on both sides),
//! then the κ → ∞ limit: ferric at κ = 1e6 must equal ferric plain RI-MP2 to
//! 1e-12 AND the plain reference, which the generator proved equal to PySCF's
//! own `DFMP2` to 1e-12 (JSON `plain.numpy_vs_pyscf_abs_diff`). Only after
//! that is any interior κ compared.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's damping is right: E_OS(κ) and E_SS(κ) agree with the numpy
//!   reference to the plain-MP2 floor (RI/SCF, ~1e-11) at every κ.
//! * If the damping is wrong (first power instead of squared, Δ taken per
//!   orbital pair instead of per quadruple, damping applied to OS only, κ in
//!   the wrong units): the error is a fraction of E_corr − E(κ), i.e. 1e-4 to
//!   5e-2 Eh here — seven or more orders above the bar. Plain MP2 cannot see
//!   any of these (they all vanish at κ → ∞), which is why the anchor alone is
//!   not the test.
//! * If the HARNESS is broken: E_nuc / nao / naux / E_RHF fail first, or the
//!   plain anchor fails before any κ is examined.
//!
//! # TOLERANCES
//!
//! Each bar is set from the measured maximum recorded on its const.
//!
//! # NEGATIVE CONTROLS (asserted in-test, always on)
//!
//! * κ SWAP: ferric at each κ must MISS the reference at every other κ by more
//!   than [`MUST_MISS`] — the bar separates neighbouring κ.
//! * PLAIN vs κ: ferric at each κ must MISS the plain-MP2 reference.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config, SpinComponents};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/kappa_mp2";
const MOL_DIR: &str = "testdata/molecules/validation";

/// Geometry check (PySCF E_nuc from ferric's Bohr geometry).
const TOL_ENUC: f64 = 1e-9;
/// E_RHF vs PySCF, exact J/K on both sides.
// Measured max 6.9e-12.
const TOL_E_RHF: f64 = 1e-9;
/// E_OS / E_SS / E_corr vs the numpy reference, plain and every κ.
// Measured max 3.3e-12 over {plain, 0.5, 1.1, 2.0} x {OS, SS, total}.
const TOL_E_CORR: f64 = 3e-11;
/// κ → ∞ (κ = 1e6) vs ferric's own plain RI-MP2: the damping underflows to
/// exactly 1.0, so only summation-order noise remains (the κ branch has the
/// same loop order as the plain one).
const TOL_KAPPA_INF: f64 = 1e-12;
/// A reference ferric must MISS (negative controls). The closest pair
/// (κ = 1.1 vs 2.0 on H2O) differs by 6.8e-3 Eh and κ = 2.0 vs plain by
/// 6.2e-4 (H2O), so 1e-6 separates them by >600x while sitting 1e4 above
/// the agreement bar.
const MUST_MISS: f64 = 1e-6;
const KAPPA_INF: f64 = 1e6;

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
             scripts/validation/gen_kappa_mp2.py — a missing reference is a failure, never a skip",
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

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn components(su: &Setup, kappa: Option<f64>) -> SpinComponents {
    ri_mp2_spin_components(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &RiMp2Config {
            kappa,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("ri_mp2_spin_components(kappa={kappa:?}) failed: {e:?}"))
    .0
}

fn check_system(system: &str, basis_name: &str, aux_name: &str) {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    assert_eq!(r["basis"].as_str(), Some(basis_name), "{ctx}: basis");
    assert_eq!(r["aux_basis"].as_str(), Some(aux_name), "{ctx}: aux basis");

    // ---- exactness anchor: geometry, basis, RHF ----
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
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
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
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };
    assert!(
        cfg.df_j_aux.is_none() && cfg.df_k_aux.is_none(),
        "exact J/K expected"
    );
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(rhf.converged, "{ctx}: RHF not converged");
    check_close(
        &ctx,
        "E_RHF",
        rhf.energy,
        num(&r, "/e_rhf", &ctx),
        TOL_E_RHF,
    );
    assert_eq!(
        RiMp2Config::default().frozen_core,
        0,
        "all-electron default"
    );

    let su = Setup {
        mol,
        obs,
        dfbs,
        rhf,
    };

    // ---- anchor: plain MP2 and the κ → ∞ limit ----
    let plain = components(&su, None);
    check_close(
        &ctx,
        "plain E_OS",
        plain.e_os,
        num(&r, "/plain/e_os", &ctx),
        TOL_E_CORR,
    );
    check_close(
        &ctx,
        "plain E_SS",
        plain.e_ss,
        num(&r, "/plain/e_ss", &ctx),
        TOL_E_CORR,
    );
    check_close(
        &ctx,
        "plain E_corr vs PySCF",
        plain.e_total,
        num(&r, "/plain/pyscf_dfmp2_e_corr", &ctx),
        TOL_E_CORR,
    );
    let inf = components(&su, Some(KAPPA_INF));
    check_close(
        &ctx,
        "kappa=1e6 vs ferric plain",
        inf.e_total,
        plain.e_total,
        TOL_KAPPA_INF,
    );

    // ---- interior κ ----
    let rows = r["kappa"].as_array().expect("kappa array");
    assert_eq!(rows.len(), 3, "{ctx}: expected three kappa rows");
    let refs: Vec<(f64, f64, f64, f64)> = rows
        .iter()
        .map(|row| {
            (
                num(row, "/kappa", &ctx),
                num(row, "/e_os", &ctx),
                num(row, "/e_ss", &ctx),
                num(row, "/e_corr", &ctx),
            )
        })
        .collect();
    let plain_ref = num(&r, "/plain/e_corr", &ctx);
    for (idx, &(kappa, e_os_ref, e_ss_ref, e_corr_ref)) in refs.iter().enumerate() {
        let kctx = format!("{ctx}/kappa={kappa}");
        let got = components(&su, Some(kappa));
        check_close(&kctx, "E_OS", got.e_os, e_os_ref, TOL_E_CORR);
        check_close(&kctx, "E_SS", got.e_ss, e_ss_ref, TOL_E_CORR);
        check_close(&kctx, "E_corr", got.e_total, e_corr_ref, TOL_E_CORR);

        // Negative controls: every other κ, and plain MP2.
        for (jdx, &(other, _, _, e_other)) in refs.iter().enumerate() {
            if jdx != idx {
                check_miss(
                    &kctx,
                    &format!("vs ref kappa={other}"),
                    got.e_total,
                    e_other,
                );
            }
        }
        check_miss(&kctx, "vs plain-MP2 ref", got.e_total, plain_ref);
    }
}

#[test]
#[ignore = "validation: κ-MP2"]
fn kappa_mp2_h2o_ccpvdz_vs_numpy_df() {
    check_system("h2o", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: κ-MP2"]
fn kappa_mp2_ch4_ccpvdz_vs_numpy_df() {
    check_system("ch4", "cc-pvdz", "cc-pvdz-ri");
}
