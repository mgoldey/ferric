//! VALIDATION tier — VALIDATION.md row "Amplitude-threshold LMP2".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_lmp2_amplitude \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's amplitude-threshold local RI-MP2 (`rimp2` with `[local] scheme =
//! "amplitude-threshold"`), BOTH assemblies — the in-core path
//! (`amplitude_lmp2`, global-metric fit and per-pair domain-local fit) and the
//! integral-direct path (`amplitude_lmp2_direct`) — against PySCF
//! `mp.dfmp2.DFMP2` (`scripts/validation/gen_lmp2_amplitude.py` →
//! `testdata/reference/validation/lmp2_amplitude/<system>_<basis>.json`) on
//! H2O and n-butane at cc-pVDZ and 6-31G, cc-pVDZ-RI aux, frozen core 0 and
//! the number of heavy atoms. PySCF was fed ferric's own orbital and aux
//! basis JSON and ferric's geometry in Bohr; its RHF is exact four-centre
//! J/K, as is ferric's `RhfConfig::default()`. The generator refuses to write
//! unless a dense numpy DF-MP2 sum equals PySCF's DFMP2 to 1e-12.
//!
//! # Exactness anchor (asserted first)
//!
//! E_nuc (geometry), AO/aux counts (basis) and E_RHF (exact J/K on both
//! sides), then ε = 0 with every locality knob at its trivial limit: the
//! mask keeps every amplitude, so the local energy is canonical DF-MP2 of the
//! same fitted integrals in a rotated (Boys / VV-HV) orbital basis. The
//! reference shares NOTHING with ferric's local path but the definition —
//! canonical orbitals, closed-form denominators, PySCF's integrals and
//! Cholesky-factorized metric vs localized orbitals, a ragged PCG solve and
//! ferric's V^{-1/2}- or domain-fit integrals — so agreement tests the whole
//! construction, not one code path against itself.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * Construction right: every ε = 0 energy equals DF-MP2 to the CG/SCF
//!   floor (~1e-10), on both bases and both frozen-core settings.
//! * Construction wrong — a virtual space that does not span the canonical
//!   one, a frozen-core offset on either side, a missing same-spin exchange
//!   term, a domain fit that is not trivial at the "trivial" radius, a
//!   direct-path strip missing rows: the energy moves by ≥1e-6 Eh (the
//!   negative controls below MEASURE how far each one moves it), at least
//!   three orders above the bar.
//! * Harness wrong: E_nuc / nao / naux / E_RHF fail before any MP2 runs.
//!
//! # Finite ε (measurement, then a structural assertion)
//!
//! On n-butane/cc-pVDZ at ε = 1e-5, 1e-4 (the value the examples and the
//! measured error maps use) and 1e-3, both frozen-core settings, both
//! paths at their production settings: the error against PySCF is recorded
//! with the keep / pair fractions and domain sizes, and the assertion is
//! only the structure the method guarantees — under-correlation (one-sided)
//! that grows monotonically with ε. No finite-ε tolerance is a bar.
//!
//! # ORCA DLPNO-MP2 ballpark (not a bar)
//!
//! `orca_<system>_cc-pvdz.json` (n-butane, n-octane; all electrons) carry
//! ORCA 6.1.1 RI-MP2 and DLPNO-MP2 at NormalPNO and TightPNO, same basis and
//! aux. DLPNO truncates by PNO occupation and PAO domains on Foster-Boys
//! LMOs, a different local scheme from ferric's amplitude threshold, so the
//! two % recoveries are printed side by side and NEVER compared. The only
//! assertion is like-for-like: ORCA RI-MP2 equals PySCF DFMP2.
//!
//! # TOLERANCES
//!
//! Each bar is set from the measured maximum recorded on its const.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{
    amplitude_lmp2, amplitude_lmp2_with_virtuals, build_vvhv, AmplitudeLmp2Config,
    AmplitudeLmp2Result, VvHv,
};
use ferric_mp2::lmp2_direct::{
    amplitude_lmp2_direct, amplitude_lmp2_direct_with_virtuals, DirectConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::s;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/lmp2_amplitude";
const MOL_DIR: &str = "testdata/molecules/validation";
const AUX: &str = "cc-pvdz-ri";

/// Geometry check (PySCF E_nuc from ferric's Bohr geometry).
const TOL_ENUC: f64 = 1e-9;
/// E_RHF vs PySCF, exact J/K on both sides.
// Measured max 6.5e-11 (c4h10/6-31G; 1.8e-11 at cc-pVDZ, <=5.4e-13 H2O).
const TOL_E_RHF: f64 = 1e-9;
/// ε = 0 E_corr vs PySCF DFMP2, all three local assemblies, both bases,
/// both frozen-core settings.
// Measured max 1.07e-11 (c4h10/cc-pVDZ integral-direct; in-core global fit
// 9.8e-12, domain fit 8.7e-12; H2O <=3.8e-12) over 24 comparisons.
const TOL_ANCHOR: f64 = 1e-10;
/// A reference ferric must MISS (negative controls). Smallest measured miss
/// 1.85e-5 (H2O/6-31G fc=1 at eps = 1e-3); wrong frozen core >= 1.0e-3;
/// one hard virtual dropped >= 2.9e-3. 1e-6 sits 18x below the smallest
/// miss and 1e4 above the agreement bar.
const MUST_MISS: f64 = 1e-6;
/// A radius that makes the per-pair domain fit (in-core) and the aux fit
/// domain (direct) contain every aux function: the trivial limit.
const TRIVIAL_RADIUS: f64 = 1e6;
/// The production ε (examples and measured error maps) and its neighbours.
const EPS_SWEEP: [f64; 3] = [1e-5, 1e-4, 1e-3];
/// ORCA RI-MP2 vs PySCF DFMP2 (same basis, aux, frozen core 0).
// Measured max 3.96e-8 (c4h10; c8h18 2.9e-8): ORCA's own integral and
// RI-MP2 thresholds, with E_RHF agreeing to 2.2e-10.
const TOL_ORCA_RIMP2: f64 = 4e-7;

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

fn load_json(name: &str) -> Value {
    let path = workspace_root().join(ROW_DIR).join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_lmp2_amplitude.py — a missing reference is a failure, never a skip",
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
    eprintln!("{ctx}: {what:<34} ferric {got:+.13} ref {want:+.13} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.13} vs reference {want:.13} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_miss(ctx: &str, what: &str, got: f64, want: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: NEG {what:<30} |d| {d:.2e} (must exceed {MUST_MISS:.0e})");
    assert!(
        d > MUST_MISS,
        "{ctx}: negative control {what}: |d| {d:.2e} <= {MUST_MISS:.0e} — the bar cannot \
         tell these apart"
    );
}

struct Setup {
    ctx: String,
    mol: Molecule,
    obs_bs: BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

/// Load geometry and bases, assert E_nuc / nao / naux, solve and assert RHF.
fn setup(system: &str, basis_name: &str, r: &Value) -> Setup {
    let ctx = format!("{system}/{basis_name}");
    assert_eq!(r["basis"].as_str(), Some(basis_name), "{ctx}: basis");
    assert_eq!(r["aux_basis"].as_str(), Some(AUX), "{ctx}: aux basis");
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
    let obs_bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
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
    check_close(&ctx, "E_RHF", rhf.energy, num(r, "/e_rhf", &ctx), TOL_E_RHF);
    Setup {
        ctx,
        mol,
        obs_bs,
        obs,
        dfbs,
        rhf,
    }
}

fn lcfg(eps: f64, frozen_core: usize, fit_radius_bohr: Option<f64>) -> AmplitudeLmp2Config {
    AmplitudeLmp2Config {
        eps,
        frozen_core,
        fit_radius_bohr,
        ..Default::default()
    }
}

/// Every integral-direct locality map at its no-op limit.
fn trivial_maps() -> DirectConfig {
    DirectConfig {
        aux_radius_bohr: TRIVIAL_RADIUS,
        virt_radius_bohr: None,
        ao_tail: 0.0,
        schwarz_skip: 0.0,
        batch_merge: 1,
        virt_schwarz_kappa: None,
        ..Default::default()
    }
}

/// The integral-direct production maps (`examples/alkane8-rimp2-local-direct.toml`).
fn production_maps() -> DirectConfig {
    DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5,
        batch_merge: 4,
        virt_schwarz_kappa: None,
        ..Default::default()
    }
}
const PRODUCTION_GATE_CAL: f64 = 0.7;

fn in_core(su: &Setup, cfg: &AmplitudeLmp2Config) -> AmplitudeLmp2Result {
    let r = amplitude_lmp2(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        cfg,
    )
    .unwrap_or_else(|e| panic!("{}: amplitude_lmp2 failed: {e:?}", su.ctx));
    assert!(r.cg_converged, "{}: CG not converged", su.ctx);
    r
}

fn direct(su: &Setup, cfg: &AmplitudeLmp2Config, dcfg: &DirectConfig) -> AmplitudeLmp2Result {
    let (r, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        cfg,
        dcfg,
    )
    .unwrap_or_else(|e| panic!("{}: amplitude_lmp2_direct failed: {e:?}", su.ctx));
    assert!(r.cg_converged, "{}: CG not converged", su.ctx);
    r
}

/// The VV-HV space with its LAST hard virtual removed: still orthonormal,
/// no longer spanning the canonical virtual space.
fn drop_one_hard_virtual(su: &Setup) -> VvHv {
    let v = build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
    let nvir = v.c_vloc.ncols();
    assert!(v.n_hard > 0, "{}: no hard virtual to drop", su.ctx);
    VvHv {
        c_vloc: v.c_vloc.slice(s![.., ..nvir - 1]).to_owned(),
        n_valence: v.n_valence,
        n_hard: v.n_hard - 1,
    }
}

/// ε = 0 anchor for one (system, basis): every frozen-core row of the JSON,
/// three local assemblies each, plus the negative controls.
fn anchor(system: &str, basis_name: &str) {
    let r = load_json(&format!("{system}_{basis_name}.json"));
    let su = setup(system, basis_name, &r);
    let rows = r["dfmp2"].as_array().expect("dfmp2 rows");
    assert_eq!(rows.len(), 2, "{}: expected fc = 0 and fc > 0", su.ctx);
    let refs: Vec<(usize, f64)> = rows
        .iter()
        .map(|row| {
            (
                row["frozen_core"].as_u64().unwrap() as usize,
                num(row, "/e_corr", &su.ctx),
            )
        })
        .collect();
    assert_eq!(refs[0].0, 0, "{}: first row must be all-electron", su.ctx);
    assert!(refs[1].0 > 0, "{}: second row must freeze a core", su.ctx);

    for (idx, &(fc, e_ref)) in refs.iter().enumerate() {
        let ctx = format!("{}/fc={fc}", su.ctx);
        let glob = in_core(&su, &lcfg(0.0, fc, None));
        let dfit = in_core(&su, &lcfg(0.0, fc, Some(TRIVIAL_RADIUS)));
        let dir = direct(&su, &lcfg(0.0, fc, None), &trivial_maps());
        for (name, x) in [
            ("in-core global fit", &glob),
            ("in-core domain fit (trivial)", &dfit),
            ("integral-direct (trivial)", &dir),
        ] {
            assert!(
                x.keep_fraction == 1.0 && x.pair_fraction == 1.0,
                "{ctx}: {name}: eps = 0 must keep everything (keep {} pairs {})",
                x.keep_fraction,
                x.pair_fraction
            );
            check_close(&ctx, name, x.e_corr, e_ref, TOL_ANCHOR);
        }

        // Negative control: the other frozen-core count's reference.
        let (fc_other, e_other) = refs[1 - idx];
        check_miss(
            &ctx,
            &format!("vs fc={fc_other} reference"),
            glob.e_corr,
            e_other,
        );

        // Negative control: finite ε must miss the ε = 0 reference.
        let e3 = in_core(&su, &lcfg(1e-3, fc, None));
        check_miss(&ctx, "in-core eps=1e-3", e3.e_corr, e_ref);
        let d3 = direct(&su, &lcfg(1e-3, fc, None), &trivial_maps());
        check_miss(&ctx, "direct eps=1e-3", d3.e_corr, e_ref);
    }

    // Negative control: a virtual space that is orthonormal but one hard
    // virtual short must miss, on both assemblies (all-electron row).
    let (fc, e_ref) = refs[0];
    let broken = drop_one_hard_virtual(&su);
    let ctx = format!("{}/fc={fc}", su.ctx);
    let bi = amplitude_lmp2_with_virtuals(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &lcfg(0.0, fc, None),
        &broken,
    )
    .unwrap();
    check_miss(&ctx, "in-core, one hard virtual dropped", bi.e_corr, e_ref);
    let (bd, _) = amplitude_lmp2_direct_with_virtuals(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &lcfg(0.0, fc, None),
        &trivial_maps(),
        &broken,
    )
    .unwrap();
    check_miss(&ctx, "direct, one hard virtual dropped", bd.e_corr, e_ref);
}

#[test]
#[ignore = "validation: Amplitude-threshold LMP2"]
fn eps_zero_h2o_ccpvdz_matches_pyscf_dfmp2() {
    anchor("h2o", "cc-pvdz");
}

#[test]
#[ignore = "validation: Amplitude-threshold LMP2"]
fn eps_zero_h2o_631g_matches_pyscf_dfmp2() {
    anchor("h2o", "6-31g");
}

#[test]
#[ignore = "validation: Amplitude-threshold LMP2"]
fn eps_zero_c4h10_ccpvdz_matches_pyscf_dfmp2() {
    anchor("c4h10", "cc-pvdz");
}

#[test]
#[ignore = "validation: Amplitude-threshold LMP2"]
fn eps_zero_c4h10_631g_matches_pyscf_dfmp2() {
    anchor("c4h10", "6-31g");
}

/// Finite ε on n-butane/cc-pVDZ: the error against PySCF DFMP2 is recorded,
/// and must be one-sided (less correlation) and grow monotonically with ε,
/// on both paths at their production settings and both frozen-core rows.
#[test]
#[ignore = "validation: Amplitude-threshold LMP2"]
fn finite_eps_c4h10_ccpvdz_error_is_one_sided_and_monotone() {
    let r = load_json("c4h10_cc-pvdz.json");
    let su = setup("c4h10", "cc-pvdz", &r);
    for row in r["dfmp2"].as_array().unwrap() {
        let fc = row["frozen_core"].as_u64().unwrap() as usize;
        let e_ref = num(row, "/e_corr", &su.ctx);
        for path in ["in-core", "direct"] {
            let mut errs = Vec::new();
            for &eps in &EPS_SWEEP {
                let x = if path == "in-core" {
                    in_core(&su, &lcfg(eps, fc, None))
                } else {
                    direct(
                        &su,
                        &AmplitudeLmp2Config {
                            pair_gate_cal: Some(PRODUCTION_GATE_CAL),
                            ..lcfg(eps, fc, None)
                        },
                        &production_maps(),
                    )
                };
                let err = x.e_corr - e_ref;
                eprintln!(
                    "MEASURE c4h10/cc-pvdz fc={fc} {path:<7} eps={eps:.0e}: E_corr {:+.10} \
                     err {err:+.3e} ({:.4}% of DF-MP2) keep {:.4} pairs {:.3} \
                     dom(mean/max) {:.1}/{} cg {}",
                    x.e_corr,
                    100.0 * x.e_corr / e_ref,
                    x.keep_fraction,
                    x.pair_fraction,
                    x.dom_mean,
                    x.dom_max,
                    x.cg_iterations
                );
                errs.push(err);
            }
            assert!(
                errs.iter().all(|&e| e > 0.0),
                "c4h10 fc={fc} {path}: error not one-sided: {errs:?}"
            );
            assert!(
                errs.windows(2).all(|w| w[1] > w[0]),
                "c4h10 fc={fc} {path}: error not monotone in eps {EPS_SWEEP:?}: {errs:?}"
            );
        }
    }
}

/// ORCA DLPNO-MP2 ballpark, all electrons: prints ferric's in-core local
/// % recovery next to ORCA's DLPNO % (NOT compared: different schemes).
/// Asserts only ORCA RI-MP2 == PySCF DFMP2 and ferric's own anchors.
fn orca_ballpark(system: &str) {
    let r = load_json(&format!("orca_{system}_cc-pvdz.json"));
    let ctx = format!("{system}/cc-pvdz (ORCA ballpark)");
    let e_df = num(&r, "/pyscf_dfmp2_e_corr", &ctx);
    let e_orca_ri = num(&r, "/orca/ri_mp2/e_corr", &ctx);
    check_close(
        &ctx,
        "ORCA RI-MP2 vs PySCF DFMP2",
        e_orca_ri,
        e_df,
        TOL_ORCA_RIMP2,
    );
    let su = setup(system, "cc-pvdz", &r);
    for tag in ["dlpno_normalpno", "dlpno_tightpno"] {
        eprintln!(
            "BALLPARK {system}/cc-pvdz fc=0 ORCA {tag:<16} {:.4}% of ORCA RI-MP2 \
             (TCutPNO {:.0e}, avg PNOs/pair {})",
            num(&r, &format!("/orca/{tag}/pct_of_orca_ri_mp2"), &ctx),
            num(&r, &format!("/orca/{tag}/tcut_pno"), &ctx),
            num(&r, &format!("/orca/{tag}/avg_pnos_per_pair"), &ctx),
        );
    }
    let mut prev = f64::NEG_INFINITY;
    for &eps in &EPS_SWEEP {
        let x = in_core(&su, &lcfg(eps, 0, None));
        let pct = 100.0 * x.e_corr / e_df;
        eprintln!(
            "BALLPARK {system}/cc-pvdz fc=0 ferric eps={eps:.0e}   {pct:.4}% of PySCF DF-MP2 \
             (keep {:.4}, pairs {:.3})",
            x.keep_fraction, x.pair_fraction
        );
        let err = x.e_corr - e_df;
        assert!(err > 0.0 && err > prev, "{ctx}: eps={eps}: err {err:+.3e}");
        prev = err;
    }
}

#[test]
#[ignore = "validation: Amplitude-threshold LMP2"]
fn orca_dlpno_ballpark_c4h10_ccpvdz() {
    orca_ballpark("c4h10");
}

#[test]
#[ignore = "validation: Amplitude-threshold LMP2"]
fn orca_dlpno_ballpark_c8h18_ccpvdz() {
    orca_ballpark("c8h18");
}
