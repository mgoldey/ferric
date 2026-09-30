//! VALIDATION tier — VALIDATION.md row "RSH ω tuning".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_rsh_omega \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What ferric computes (read from `omega_tuning.rs`, not a doc)
//!
//! ```text
//! J(ω) = ε_HOMO(N; ω) + E(N−1; ω) − E(N; ω)          (signed, Hartree)
//! ```
//!
//! No square and no second (anion / EA) term. `eval_j` runs a closed-shell
//! RKS neutral (`solve_rhf`) and a doublet UKS cation (`solve_uhf`) with the
//! same `RhfConfig`, differing from the base config only in `xc` and
//! `xc_omega` (ω in Bohr⁻¹). `tune_omega` minimizes |J| by golden-section
//! search and returns the evaluation with the smallest |J|, so ω* is the root
//! of J.
//!
//! References: `scripts/validation/gen_rsh_omega.py` →
//! `testdata/reference/validation/rsh_omega/<system>_def2-svp.json`: PySCF
//! ωB97X-V with `mf.omega` overridden, def2-SVP:
//!
//! * H2O, NH3: J at ω = 0.2, 0.3, 0.4, 0.5, 0.6 and ω* from `brentq` on the
//!   signed J (xtol 1e-7).
//! * N2: J at ω = 0.2, 0.3, 0.4, 0.45, 0.5 only. The D∞h ²Σg⁺ N2⁺ cation
//!   becomes UKS-unstable (hole localization) between ω = 0.53 and 0.56, and
//!   J's root on that branch is ~0.56, past the onset; the stable state there
//!   is symmetry-broken and PySCF's second-order solver does not converge it,
//!   so there is no stable reference at ω*. The generator re-measures the
//!   onset every run (`symmetric_cation_probe`).
//!
//! At every point the cation is converged from three guesses and followed to
//! an internally stable UKS state.
//!
//! # The like-for-like recipe
//!
//! | piece | ferric (this file's config) | PySCF (generator) |
//! |---|---|---|
//! | J, cation | exact (uhf.rs forces `j_aux_eff = None` for ω > 0) | exact 4-index |
//! | J, neutral | exact: `df_j_aux = Some("")` (the RKS default would be RI-J) | exact 4-index |
//! | K, both | c_SR K^DF[erfc(ω)] + c_LR K^DF[erf(ω)], jkfit aux | same, attenuated metric |
//! | grid | (75,110) Becke + Becke radii; VV10 (50,50) | same |
//!
//! `tune_omega` resolves an unset `df_j_aux` to exact J for both states
//! (`omega_tuning::state_scf_config`), which is this file's configuration.
//! The generator also records, per ω, how far an RI-J neutral would move the
//! IP (`neutral_rij_shift`); that is not compared.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric's objective is right: ε_HOMO, IP and J agree at every ω at
//!   the ωB97X-V DF-K / VV10 floor, and ω* agrees to |ΔJ| / |dJ/dω|.
//! * If the ω override does not reach the functional or the SR/LR split: J
//!   is flat in ω; ferric's J at ω_k then matches (or fails to miss) the
//!   reference at ω_{k±1} — the wrong-ω control below fails.
//! * If the cation lands on another state (ferric's stability analysis skips
//!   RSH functionals, and `tune_omega` has no state control on the cation):
//!   IP and J miss while ε_HOMO (a neutral-only quantity) still agrees — the
//!   per-quantity checks separate the two.
//! * If the IP is assembled with the wrong sign or the LUMO is used: J misses
//!   by ~0.5 Ha or more.
//!
//! # TOLERANCES (measured 2026-09-30, worst over every system and ω)
//!
//! | quantity | measured max |d| | bar |
//! |---|---:|---:|
//! | ε_HOMO(N; ω) | 6.7e-6 Ha (NH3; N2 4.8e-7) | 2e-5 Ha |
//! | IP(ω) | 8.8e-7 Ha (H2O; N2 3.4e-9) | 5e-6 Ha |
//! | J(ω) | 6.3e-6 Ha (NH3) | 2e-5 Ha |
//! | ω* | 2.8e-5 Bohr⁻¹ (NH3; H2O 2.4e-5) | 1e-4 Bohr⁻¹ |
//! | nuclear repulsion | 1.8e-15 Ha | 1e-9 Ha |
//!
//! ε_HOMO sets the J bar: J = ε_HOMO + IP, and the IP agrees to <1e-6. The
//! ~6e-6 Ha ε_HOMO floor on H2O/NH3 is the ωB97X-V construction difference
//! documented in `validation_ks_energies.rs` (VV10 pairwise sum, range-separated
//! DF-K metrics); the same functional's UKS energies sit at 6.0e-6 Ha there,
//! with the same 2e-5 bar.
//!
//! # NEGATIVE CONTROLS (always on)
//!
//! * Wrong ω: ferric's J at ω_k must MISS the reference J at ω_{k−1} and
//!   ω_{k+1} by ≥ 100× the J bar.
//! * Shifted ω*: ferric's ω* must MISS the reference ω* + `OMEGA_SHIFT`
//!   (10× the ω* bar) — the ω* comparison can see an error of that size.
//! * The tuner must converge strictly inside its bracket, and its J(ω*) must
//!   be smaller in magnitude than the J at every reference grid point.
//!
//! # MUTATIONS (to run once, record the outcome here)
//!
//! * A — in `eval_j`, drop `xc_omega: Some(omega)` from the per-evaluation
//!   config: every J point except ω = 0.3 (the published value) must fail,
//!   and the wrong-ω control must fail.
//! * B — in `eval_j`, set `j: eps_homo - ip`: every J point must fail by
//!   ~2·IP.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::omega_tuning::{eval_j, tune_omega, OmegaTuneConfig};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/rsh_omega";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "def2-svp";
const AUX: &str = "def2-universal-jkfit";
const FUNCTIONAL: &str = "wB97X-V";

// Bars — measured maxima in the module doc's TOLERANCES table.
const TOL_EPS: f64 = 2e-5;
const TOL_IP: f64 = 5e-6;
const TOL_J: f64 = 2e-5;
const TOL_OMEGA_STAR: f64 = 1e-4;
const TOL_ENUC: f64 = 1e-9;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 100.0;
/// Deliberate shift of the reference ω* for the shifted-ω* control.
const OMEGA_SHIFT: f64 = 10.0 * TOL_OMEGA_STAR;
/// Tuner bracket and resolution (Bohr⁻¹). Fixed, NOT centred on the reference.
const TUNE_LO: f64 = 0.2;
const TUNE_HI: f64 = 0.8;
const TUNE_TOL: f64 = 1e-5;
const TUNE_MAX_EVALS: usize = 40;

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
fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_rsh_omega.py — a missing reference is a failure, never a skip",
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

struct System {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn load_system(system: &str, r: &Value) -> System {
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    assert_eq!(
        r["functional"].as_str(),
        Some(FUNCTIONAL),
        "{system}: reference functional is not {FUNCTIONAL}"
    );
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    // Geometry like-for-like FIRST: a constant or unit slip fails here as a
    // geometry defect, not later as an unexplained J offset.
    let enuc_ref = num(r, "/nuclear_repulsion", system);
    let enuc = mol.nuclear_repulsion();
    eprintln!(
        "{system}: E_nuc ferric {enuc:.12} ref {enuc_ref:.12} |d| {:.2e}",
        (enuc - enuc_ref).abs()
    );
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{system}: nuclear repulsion {enuc:.12} vs reference {enuc_ref:.12} — geometry/unit mismatch"
    );
    let bs = basis::bundled(BASIS).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let nao_ref = r["nao"].as_u64().expect("nao") as usize;
    assert_eq!(
        prep.nbasis(),
        nao_ref,
        "{system}: AO count differs from the reference's"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    System {
        mol,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

/// Exact J on BOTH states (see the module-doc recipe table); RSH K is fitted
/// with the jkfit aux in the attenuated metric.
fn tune_config(omega_lo: f64, omega_hi: f64) -> OmegaTuneConfig {
    OmegaTuneConfig {
        functional: FUNCTIONAL.into(),
        omega_lo,
        omega_hi,
        omega_tol: TUNE_TOL,
        max_evals: TUNE_MAX_EVALS,
        scf: RhfConfig {
            df_j_aux: Some(String::new()),
            df_k_aux: Some(AUX.into()),
            max_iter: 500,
            energy_conv: 1e-10,
            density_conv: 1e-7,
            ..Default::default()
        },
    }
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<9} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

/// J(ω), ε_HOMO(ω) and IP(ω) at every reference grid point, plus the wrong-ω
/// control against the neighbouring points.
fn check_points(system: &str) {
    let r = reference(system);
    let sys = load_system(system, &r);
    let pts = r["points"].as_array().expect("points");
    assert!(pts.len() >= 5, "{system}: expected >= 5 reference points");
    let cfg = tune_config(TUNE_LO, TUNE_HI);
    let mut j_ferric = Vec::with_capacity(pts.len());
    for (k, p) in pts.iter().enumerate() {
        let w = num(p, "/omega", system);
        let ctx = format!("{system}/{BASIS}/w={w:.3}");
        let e = eval_j(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg, w)
            .unwrap_or_else(|err| panic!("{ctx}: eval_j failed: {err:?}"));
        assert_eq!(e.omega, w, "{ctx}: eval_j echoed a different omega");
        check_close(
            &ctx,
            "eps_HOMO",
            e.eps_homo,
            num(p, "/eps_homo", &ctx),
            TOL_EPS,
        );
        check_close(&ctx, "IP", e.ip_delta_scf, num(p, "/ip", &ctx), TOL_IP);
        check_close(&ctx, "J", e.j, num(p, "/j", &ctx), TOL_J);
        // J is exactly the documented combination of the other two.
        assert_eq!(e.j, e.eps_homo + e.ip_delta_scf, "{ctx}: J != eps + IP");
        j_ferric.push((k, e.j));
    }
    // Wrong-ω control: ferric J at ω_k must MISS the reference at ω_{k±1}.
    let must_miss = MUST_MISS_FACTOR * TOL_J;
    for (k, j) in j_ferric {
        for n in [k.wrapping_sub(1), k + 1] {
            let Some(q) = pts.get(n) else { continue };
            let j_other = num(q, "/j", system);
            assert!(
                (j - j_other).abs() > must_miss,
                "{system}: ferric J at point {k} ({j:+.3e}) is within {must_miss:.0e} of the \
                 reference J at point {n} ({j_other:+.3e}) — J does not respond to omega"
            );
        }
    }
}

/// ω* from ferric's golden-section tuner against the reference root, plus the
/// shifted-reference control.
fn check_omega_star(system: &str) {
    let r = reference(system);
    let sys = load_system(system, &r);
    let ctx = format!("{system}/{BASIS}/omega*");
    let w_ref = num(&r, "/omega_star", &ctx);
    assert!(
        w_ref > TUNE_LO && w_ref < TUNE_HI,
        "{ctx}: reference omega* {w_ref} outside the tuner bracket [{TUNE_LO}, {TUNE_HI}]"
    );
    let cfg = tune_config(TUNE_LO, TUNE_HI);
    let t = tune_omega(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg)
        .unwrap_or_else(|e| panic!("{ctx}: tune_omega failed: {e:?}"));
    eprintln!("{ctx}: {t}");
    assert!(
        t.converged,
        "{ctx}: tuner did not converge in {TUNE_MAX_EVALS} evals"
    );
    assert!(
        t.omega > TUNE_LO + TUNE_TOL && t.omega < TUNE_HI - TUNE_TOL,
        "{ctx}: omega* {} pinned at the bracket edge",
        t.omega
    );
    for p in r["points"].as_array().expect("points") {
        let jp = num(p, "/j", &ctx);
        assert!(
            t.j.abs() < jp.abs(),
            "{ctx}: |J(omega*)| {:.3e} not below |J| {:.3e} at grid omega {}",
            t.j.abs(),
            jp.abs(),
            num(p, "/omega", &ctx)
        );
    }
    check_close(&ctx, "omega*", t.omega, w_ref, TOL_OMEGA_STAR);
    // Shifted-reference control: the comparison can see an error of
    // OMEGA_SHIFT, i.e. it is not passing by construction.
    let shifted = w_ref + OMEGA_SHIFT;
    assert!(
        (t.omega - shifted).abs() > TOL_OMEGA_STAR,
        "{ctx}: ferric omega* {:.8} also matches the SHIFTED reference {shifted:.8} — \
         the omega* bar cannot resolve a {OMEGA_SHIFT:.0e} error",
        t.omega
    );
}

#[test]
#[ignore = "validation: RSH omega tuning"]
fn j_of_omega_h2o_wb97xv_vs_pyscf() {
    check_points("h2o");
}

#[test]
#[ignore = "validation: RSH omega tuning"]
fn j_of_omega_n2_wb97xv_vs_pyscf() {
    check_points("n2");
}

#[test]
#[ignore = "validation: RSH omega tuning"]
fn j_of_omega_nh3_wb97xv_vs_pyscf() {
    check_points("nh3");
}

#[test]
#[ignore = "validation: RSH omega tuning"]
fn omega_star_h2o_wb97xv_vs_pyscf() {
    check_omega_star("h2o");
}

#[test]
#[ignore = "validation: RSH omega tuning"]
fn omega_star_nh3_wb97xv_vs_pyscf() {
    check_omega_star("nh3");
}

// No omega_star for N2: the reference cation is unstable at ω* (module doc).
