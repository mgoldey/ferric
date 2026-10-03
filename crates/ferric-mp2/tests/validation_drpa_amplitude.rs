//! VALIDATION tier — VALIDATION.md row "Amplitude-threshold dRPA".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_drpa_amplitude \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's dRPA@HF by the drCCD Riccati solve in the Boys-localized basis
//! (`ferric_mp2::drpa_amplitude`) against an independent numpy plasmon dRPA
//! on PySCF density-fitted integrals (`scripts/validation/gen_drpa_amplitude.py`
//! → `testdata/reference/validation/drpa_amplitude/<system>_<basis>.json`).
//! PySCF was fed ferric's orbital and aux basis JSON and ferric's geometry in
//! Bohr; its RHF is exact four-centre J/K, as is ferric's here. The numpy
//! side semicanonicalizes the converged Fock, builds A = D + 2(ia|jb),
//! B = 2(ia|jb) over the active occupieds, and evaluates
//! E = ½(Σ√eig[(A−B)(A+B)] − Tr A) by a SYMMETRIC eigensolve (ferric's
//! in-crate reference uses a non-symmetric one). The generator proved that
//! formula, with these spin factors and frozen cores, equal to PySCF's own
//! frequency-integrated `pyscf.gw.rpa.RPA` on the same DF integrals (≤ 3.6e-11
//! at 160 points, every system and frozen-core setting) and the H2/STO-3G
//! value equal to the proof notebook's −0.0126072623 (3.7e-11).
//!
//! At ε = 0, on H2/STO-3G (STO-3G aux), water and n-butane at 6-31G and
//! cc-pVDZ (cc-pVDZ-RI aux), every frozen-core setting the reference carries
//! (0/1 for water, 0/1/4 for butane), three ferric routes must land on the
//! reference: the exact method as the CLI and `run_drpa` run it
//! (`amplitude_drpa`, ε = 0, DIIS 8, ε-linked rtol 0.1), the integral-direct
//! assembly with trivial locality maps (`amplitude_drpa_direct`), and the
//! in-crate `canonical_plasmon_drpa` that the unit tests anchor against
//! (until now compared to no external code).
//!
//! # Exactness anchor (asserted first)
//!
//! E_nuc (geometry), AO/aux counts (basis), E_RHF (exact J/K on both sides),
//! then ε = 0. Only after that is any finite ε examined.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric is right: the three ε = 0 routes agree with numpy to the
//!   RI/SCF floor (≤ 3.0e-11 measured), every system and frozen core.
//! * If a spin factor is wrong (B = (ia|jb) or 4(ia|jb)), the ring coupling
//!   is dropped, the frozen core is off by one, or the virtual space is
//!   incomplete: the miss is 1e-3 to 1e-1 Eh (the frozen-core gaps here are
//!   ≥ 1.4e-3 Eh, the ring-dropped gap ≥ 2.7e-3 Eh) — orders above the bar.
//! * If the HARNESS is broken: E_nuc / nao / naux / E_RHF fail first.
//!
//! # n-octane ε sweep — CHARACTERIZATION, not validation
//!
//! [`drpa_amplitude_eps_sweep_alkane_8_631g`] runs n-octane / 6-31G /
//! cc-pVDZ-RI with the 8 carbon cores frozen at ε ∈ {0, 1e-6, 1e-5, 1e-4,
//! 1e-3} on one SCF and one localized assembly, and compares to the numpy
//! ε = 0 reference. There is NO external finite-ε reference; this is
//! ferric's truncation error against an external exact value. Asserted:
//! ε = 0 on the reference; for ε > 0 the error is positive (less
//! correlation) and grows monotonically with ε, and the kept fraction falls.
//!
//! MEASUREMENT (2026-10-02, this file's sweep; err = E(ε) − E_ref, Eh):
//!
//! ```text
//! eps     err (Eh)       keep      pairs   iters
//! 0       -3.04e-11      1.000000  1.0000  29
//! 1e-6    +6.679e-7      0.824492  1.0000  28
//! 1e-5    +3.873e-5      0.502643  1.0000  28
//! 1e-4    +5.464e-4      0.176051  1.0000  26
//! 1e-3    +1.0106e-2     0.028275  0.8016  24
//! ```
//!
//! Interpretation (provisional, one molecule, one basis): the error is
//! positive at every ε > 0 and grows FASTER than linearly over this range
//! (local log-log slopes 1.76, 1.15, 1.27 per decade).
//!
//! # TOLERANCES
//!
//! Each bar is set from the measured maximum recorded on its const.
//!
//! # NEGATIVE CONTROLS (asserted in-test, always on)
//!
//! * ε = 0 ferric must MISS the ring-dropped value −½ Σ B²/(D_ia + D_jb)
//!   (the Riccati root with BT + TB + TBT removed: second-order dRPA).
//! * ε = 0 ferric at each frozen-core setting must MISS the reference at
//!   every OTHER frozen-core setting.
//! * ε = 1e-3 must MISS the ε = 0 reference by more than the bar (water, and
//!   the octane sweep).
//!
//! A missing reference JSON is a HARD failure (panic naming the path).
//!
//! # MUTATIONS (2026-10-02; each applied to `src/drpa_amplitude.rs`, the
//! water/6-31G test run, the file restored)
//!
//! * drop one hard virtual on the exact route → FAILS, |d| 4.13e-3;
//! * B → 0 in the Riccati residual (BT + TB + TBT removed, solve and DIIS
//!   check) → FAILS, |d| 5.90e-2, landing on the numpy ring-dropped value
//!   (−0.1974214130703 vs −0.1974214131); with the DIIS post-hoc check left
//!   intact the solver itself refuses (relres 4.4e-1);
//! * frozen core off by one on the Riccati route → FAILS, |d| 1.35e-3;
//! * B = 1·(ia|jb) instead of 2·(ia|jb) → FAILS, |d| 9.82e-2.

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::drpa_amplitude::{
    amplitude_drpa, amplitude_drpa_direct, amplitude_drpa_scan, canonical_plasmon_drpa,
    AmplitudeDrpaConfig,
};
use ferric_mp2::lmp2_direct::DirectConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/drpa_amplitude";
const MOL_DIR: &str = "testdata/molecules/validation";

/// Geometry check (PySCF E_nuc from ferric's Bohr geometry).
const TOL_ENUC: f64 = 1e-9;
/// E_RHF vs PySCF, exact J/K on both sides.
// Measured max 5.6e-10 (n-butane/cc-pVDZ); water 4.8e-12, n-octane 5.0e-11.
const TOL_E_RHF: f64 = 5e-9;
/// dRPA E_c at ε = 0 vs the numpy plasmon reference, every route, system and
/// frozen-core setting.
// Measured max 3.0e-11 (n-octane/6-31G, exact route); n-butane 2.0e-11 to
// 2.9e-11 on all three routes, water ≤ 2.0e-13, H2 ≤ 6.9e-16.
const TOL_E_CORR: f64 = 3e-10;
/// A reference ferric must MISS (negative controls). The closest measured
/// control is water/6-31G frozen core 0 at ε = 1e-3 against the ε = 0
/// reference (1.54e-5 Eh); the closest frozen-core pair is water/6-31G 0 vs 1
/// (1.35e-3 Eh). 1e-6 sits 15x below the first and >3000x above the
/// agreement bar.
const MUST_MISS: f64 = 1e-6;
/// The DIIS subspace the CLI and `run_drpa` use for the exact method.
const DIIS: usize = 8;

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
             scripts/validation/gen_drpa_amplitude.py — a missing reference is a failure, never a skip",
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
    eprintln!("{ctx}: {what:<28} ferric {got:+.13} ref {want:+.13} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.13} vs reference {want:.13} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_miss(ctx: &str, what: &str, got: f64, want: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<28} |d| {d:.2e} (must exceed {MUST_MISS:.0e})");
    assert!(
        d > MUST_MISS,
        "{ctx}: negative control {what}: |d| {d:.2e} <= {MUST_MISS:.0e} — the bar cannot \
         tell these apart"
    );
}

struct Setup {
    ctx: String,
    r: Value,
    mol: Molecule,
    obs_bs: BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

/// Geometry, basis and RHF anchors; returns everything the dRPA runs need.
fn setup(system: &str, basis_name: &str, aux_name: &str) -> Setup {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    assert_eq!(r["basis"].as_str(), Some(basis_name), "{ctx}: basis");
    assert_eq!(r["aux_basis"].as_str(), Some(aux_name), "{ctx}: aux basis");

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
    let obs_bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
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
    // dRPA is FIRST order in a Fock/orbital inconsistency (proof notebook
    // §3), so the density is converged one decade past the κ-MP2 harness.
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-10,
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
    Setup {
        ctx,
        r,
        mol,
        obs_bs,
        obs,
        dfbs,
        rhf,
    }
}

/// The exact method as the CLI (`method.kind = "drpa"`) and `run_drpa` run
/// it: ε = 0, DIIS 8, ε-linked stopping tolerance 0.1 (a no-op at ε = 0).
fn exact_cfg(frozen_core: usize) -> AmplitudeDrpaConfig {
    AmplitudeDrpaConfig {
        eps: 0.0,
        frozen_core,
        diis: Some(DIIS),
        eps_rtol_factor: Some(0.1),
        ..Default::default()
    }
}

/// Every locality map at its no-op limit (as `tests/drpa_direct.rs`).
fn trivial_maps() -> DirectConfig {
    DirectConfig {
        aux_radius_bohr: 1e6,
        virt_radius_bohr: None,
        ao_tail: 0.0,
        ..Default::default()
    }
}

/// (frozen_core, e_corr, e_corr_ring_dropped) for every setting the
/// reference carries, in ascending frozen-core order.
fn frozen_settings(su: &Setup) -> Vec<(usize, f64, f64)> {
    let blocks = su.r["frozen_core"]
        .as_object()
        .unwrap_or_else(|| panic!("{}: reference has no frozen_core block", su.ctx));
    let mut out: Vec<(usize, f64, f64)> = blocks
        .iter()
        .map(|(k, b)| {
            let fc: usize = k.parse().expect("frozen-core key");
            assert_eq!(b["frozen_core"].as_u64(), Some(fc as u64), "{}", su.ctx);
            (
                fc,
                num(b, "/e_corr", &su.ctx),
                num(b, "/e_corr_ring_dropped", &su.ctx),
            )
        })
        .collect();
    out.sort_by_key(|t| t.0);
    out
}

fn check_eps_zero(system: &str, basis_name: &str, aux_name: &str) -> Setup {
    let su = setup(system, basis_name, aux_name);
    let op = Operator::coulomb();
    let settings = frozen_settings(&su);
    for &(fc, e_ref, e_ring) in &settings {
        let ctx = format!("{}/fc={fc}", su.ctx);
        let cfg = exact_cfg(fc);

        let exact = amplitude_drpa(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg)
            .unwrap_or_else(|e| panic!("{ctx}: amplitude_drpa failed: {e:?}"));
        assert!(exact.converged, "{ctx}: exact Riccati not converged");
        assert!(
            exact.keep_fraction == 1.0 && exact.pair_fraction == 1.0,
            "{ctx}: eps = 0 truncated something (keep {}, pairs {})",
            exact.keep_fraction,
            exact.pair_fraction
        );
        check_close(
            &ctx,
            "exact Riccati (CLI config)",
            exact.e_corr,
            e_ref,
            TOL_E_CORR,
        );

        let (direct, _) = amplitude_drpa_direct(
            &su.mol,
            &su.obs,
            &su.obs_bs,
            &su.dfbs,
            op,
            &su.rhf,
            &cfg,
            &trivial_maps(),
        )
        .unwrap_or_else(|e| panic!("{ctx}: amplitude_drpa_direct failed: {e:?}"));
        check_close(
            &ctx,
            "direct, trivial maps",
            direct.e_corr,
            e_ref,
            TOL_E_CORR,
        );

        let plasmon = canonical_plasmon_drpa(&su.mol, &su.obs, &su.dfbs, op, &su.rhf, &cfg)
            .unwrap_or_else(|e| panic!("{ctx}: canonical_plasmon_drpa failed: {e:?}"));
        check_close(
            &ctx,
            "in-crate canonical plasmon",
            plasmon,
            e_ref,
            TOL_E_CORR,
        );

        // Negative controls: ring coupling dropped, and every other frozen core.
        check_miss(&ctx, "vs ring-dropped ref", exact.e_corr, e_ring);
        for &(other, e_other, _) in &settings {
            if other != fc {
                check_miss(&ctx, &format!("vs ref fc={other}"), exact.e_corr, e_other);
            }
        }
    }
    su
}

#[test]
#[ignore = "validation: Amplitude-threshold dRPA"]
fn drpa_amplitude_h2_sto3g_vs_numpy_plasmon() {
    let su = check_eps_zero("h2", "sto-3g", "sto-3g");
    // The proof notebook's value, recorded by the generator next to its own.
    let nb = num(&su.r, "/frozen_core/0/proof_notebook_value", &su.ctx);
    let e = amplitude_drpa(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &exact_cfg(0),
    )
    .unwrap()
    .e_corr;
    check_close(&su.ctx, "vs proof notebook", e, nb, TOL_E_CORR);
}

/// Water also carries the finite-ε negative control: ε = 1e-3 must MISS the
/// ε = 0 reference by more than the bar.
fn water_with_finite_eps_control(basis_name: &str) {
    let su = check_eps_zero("h2o", basis_name, "cc-pvdz-ri");
    for (fc, e_ref, _) in frozen_settings(&su) {
        let ctx = format!("{}/fc={fc}", su.ctx);
        let r = amplitude_drpa(
            &su.mol,
            &su.obs,
            &su.obs_bs,
            &su.dfbs,
            Operator::coulomb(),
            &su.rhf,
            &AmplitudeDrpaConfig {
                eps: 1e-3,
                ..exact_cfg(fc)
            },
        )
        .unwrap_or_else(|e| panic!("{ctx}: eps=1e-3 failed: {e:?}"));
        assert!(r.keep_fraction < 1.0, "{ctx}: eps = 1e-3 did not truncate");
        check_miss(&ctx, "eps=1e-3 vs eps=0 ref", r.e_corr, e_ref);
    }
}

#[test]
#[ignore = "validation: Amplitude-threshold dRPA"]
fn drpa_amplitude_h2o_631g_vs_numpy_plasmon() {
    water_with_finite_eps_control("6-31g");
}

#[test]
#[ignore = "validation: Amplitude-threshold dRPA"]
fn drpa_amplitude_h2o_ccpvdz_vs_numpy_plasmon() {
    water_with_finite_eps_control("cc-pvdz");
}

#[test]
#[ignore = "validation: Amplitude-threshold dRPA"]
fn drpa_amplitude_alkane_4_631g_vs_numpy_plasmon() {
    check_eps_zero("alkane_4", "6-31g", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: Amplitude-threshold dRPA"]
fn drpa_amplitude_alkane_4_ccpvdz_vs_numpy_plasmon() {
    check_eps_zero("alkane_4", "cc-pvdz", "cc-pvdz-ri");
}

/// CHARACTERIZATION (see the module doc): n-octane ε sweep against the numpy
/// ε = 0 reference. Plain fixed-point stopping (fp_rtol 1e-12, no ε link) so
/// the solver error stays far below the truncation error being measured.
#[test]
#[ignore = "validation: Amplitude-threshold dRPA"]
fn drpa_amplitude_eps_sweep_alkane_8_631g() {
    let su = setup("alkane_8", "6-31g", "cc-pvdz-ri");
    let settings = frozen_settings(&su);
    assert_eq!(settings.len(), 1, "{}: one frozen-core setting", su.ctx);
    let (fc, e_ref, _) = settings[0];
    assert_eq!(fc, 8, "{}: the 8 carbon cores are frozen", su.ctx);
    let eps_list = [0.0, 1e-6, 1e-5, 1e-4, 1e-3];
    let base = AmplitudeDrpaConfig {
        eps: 0.0,
        frozen_core: fc,
        diis: Some(DIIS),
        eps_rtol_factor: None,
        ..Default::default()
    };
    let rs = amplitude_drpa_scan(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &base,
        &eps_list,
    )
    .unwrap_or_else(|e| panic!("{}: amplitude_drpa_scan failed: {e:?}", su.ctx));
    eprintln!(
        "{}: eps sweep, err = E(eps) - E_ref(numpy, eps = 0) = {e_ref:+.12}",
        su.ctx
    );
    eprintln!("    eps        err (Eh)        keep      pairs   iters");
    for (eps, r) in eps_list.iter().zip(&rs) {
        eprintln!(
            "    {eps:<8.0e} {:+.6e}  {:.6}  {:.4}  {}",
            r.e_corr - e_ref,
            r.keep_fraction,
            r.pair_fraction,
            r.iterations
        );
    }
    check_close(
        &su.ctx,
        "eps=0 (exact) vs numpy",
        rs[0].e_corr,
        e_ref,
        TOL_E_CORR,
    );
    assert_eq!(rs[0].keep_fraction, 1.0, "{}: eps = 0 truncated", su.ctx);
    let mut prev_err = rs[0].e_corr - e_ref;
    let mut prev_keep = rs[0].keep_fraction;
    for (eps, r) in eps_list.iter().zip(&rs).skip(1) {
        let err = r.e_corr - e_ref;
        assert!(
            err > TOL_E_CORR,
            "{}: eps={eps:e}: error {err:+.3e} is not under-correlation above the bar",
            su.ctx
        );
        assert!(
            err > prev_err,
            "{}: eps={eps:e}: error {err:+.3e} not larger than at the previous eps ({prev_err:+.3e})",
            su.ctx
        );
        assert!(
            r.keep_fraction < prev_keep,
            "{}: eps={eps:e}: keep fraction {} did not fall (previous {prev_keep})",
            su.ctx,
            r.keep_fraction
        );
        prev_err = err;
        prev_keep = r.keep_fraction;
    }
    // Negative control: the loosest point misses the exact reference.
    check_miss(&su.ctx, "eps=1e-3 vs eps=0 ref", rs[4].e_corr, e_ref);
}
