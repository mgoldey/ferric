//! VALIDATION tier — VALIDATION.md row "RSH ω tuning" (cation stability).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_rsh_cation_stability \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is checked
//!
//! `tune_omega` runs the UKS internal-stability analysis on every cation
//! (`OmegaTuneConfig::check_cation_stability`, default on) and records the
//! orbital Hessian's lowest eigenvalue λ_min per `OmegaEval`. The onset of the
//! N2⁺ instability is a CURVATURE change that J, ⟨S²⟩ and the spin populations
//! do not see, so λ_min is the only observable.
//!
//! Reference: `scripts/validation/gen_rsh_cation_stability.py` →
//! `testdata/reference/validation/rsh_cation_stability/<system>_def2-svp.json`:
//! PySCF UKS/ωB97X (NOT ωB97X-V: no VV10, so ferric's Hessian omits nothing)
//! doublet cation, `mf.omega` overridden, default guess, not stability-followed,
//! λ_min by dense `gen_g_hop_uhf`. N2⁺ λ_min changes sign between ω = 0.50
//! (+2.2e-3) and 0.55 (−6.9e-3); H2O⁺ and NH3⁺ are positive (≥ 7.8e-2) at
//! every ω.
//!
//! # Artifact hypothesis
//!
//! If the check is real, ferric's λ_min tracks PySCF's at every ω on both sides
//! of the sign change and the verdict flips with it. If the Hessian were
//! evaluated at a wrong operator or the verdict read from a wrong field, λ_min
//! would miss by O(1e-2) (cf. `validation_rsh_stability.rs`: a plain-Coulomb
//! response misses by ≥ 0.29) or the sign would not follow ω.
//!
//! # MEASURED (2026-10-06): max |Δλ_min| 2.3e-8 (H2O), 7.2e-9 (NH3), 1.8e-9
//! (N2) over 13 points; verdicts follow the PySCF sign at every point; see
//! `TOL_LAMBDA`.
//!
//! # MUTATIONS (ledger in the PR body)
//!
//! * M1 — `eval_j_seeded` never sets `check_stability` on the cation: every
//!   λ_min reads `NotAnalysed(AnalysisFailed)`; the λ_min and verdict tests fail.
//! * M2 — `CationStability::is_saddle` matches `Stable` instead of `Unstable`:
//!   the tuner no longer refuses the saddle ω*; the refusal test fails.
//! * M3 — `stability_report` ignores `NotAnalysed`: the VV10 test fails.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::omega_tuning::{eval_j, tune_omega, CationStability, OmegaTuneConfig};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilitySkip;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/rsh_cation_stability";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "def2-svp";
const AUX: &str = "def2-universal-jkfit";
const FUNCTIONAL: &str = "HYB_GGA_XC_WB97X";

/// Bar on |λ_min(ferric) − λ_min(PySCF)| in Ha/rad²: ~13x the measured max
/// (2.3e-8, H2O ω = 1.0; N2 1.8e-9, NH3 7.2e-9), and 4 decades below the
/// smallest |λ_min| the sign test must resolve (2.2e-3, N2 ω = 0.50).
const TOL_LAMBDA: f64 = 3e-7;

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

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_rsh_cation_stability.py — a missing reference is a failure",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

struct System {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

/// The NEUTRAL molecule (tune_omega adds the charge itself).
fn load(system: &str) -> System {
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let bs = basis::bundled(BASIS).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    System {
        mol,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

fn config(functional: &str, lo: f64, hi: f64, check: bool) -> OmegaTuneConfig {
    OmegaTuneConfig {
        functional: functional.into(),
        omega_lo: lo,
        omega_hi: hi,
        omega_tol: 2e-2,
        max_evals: 4,
        scf: RhfConfig {
            df_j_aux: Some(String::new()),
            df_k_aux: Some(AUX.into()),
            max_iter: 500,
            energy_conv: 1e-10,
            density_conv: 1e-7,
            ..Default::default()
        },
        check_cation_stability: check,
        ..Default::default()
    }
}

fn points(system: &str) -> Vec<(f64, f64, f64)> {
    reference(system)["points"]
        .as_array()
        .expect("points")
        .iter()
        .map(|p| {
            (
                p["omega"].as_f64().unwrap(),
                p["cation_lambda_min"].as_f64().unwrap(),
                p["e_cation"].as_f64().unwrap(),
            )
        })
        .collect()
}

/// ferric λ_min vs PySCF at every reference ω; sign (verdict) must follow.
fn check_lambda(system: &str, expect_sign_change: bool) {
    let sys = load(system);
    let cfg = config(FUNCTIONAL, 0.1, 1.0, true);
    let mut signs = Vec::new();
    for (w, lam_ref, e_ref) in points(system) {
        let e = eval_j(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg, w)
            .unwrap_or_else(|err| panic!("{system} w={w}: {err:?}"));
        let CationStability::Analysed {
            lambda_min,
            noise_floor,
            verdict,
        } = e.cation_stability
        else {
            panic!(
                "{system} w={w}: stability not analysed: {:?}",
                e.cation_stability
            );
        };
        let d = (lambda_min - lam_ref).abs();
        eprintln!(
            "{system} w={w:.3} lam ferric {lambda_min:+.6e} ref {lam_ref:+.6e} |d| {d:.2e} \
             |dE| {:.2e} verdict {verdict:?} floor {noise_floor:.1e}",
            (e.e_cation - e_ref).abs()
        );
        assert!(d < TOL_LAMBDA, "{system} w={w}: λ_min off by {d:.3e}");
        assert_eq!(
            e.cation_stability.is_saddle(),
            lam_ref < 0.0,
            "{system} w={w}: verdict disagrees with the reference sign"
        );
        assert_eq!(
            e.cation_stability.is_proven_stable(),
            lam_ref > 0.0,
            "{system} w={w}: stable verdict disagrees with the reference sign"
        );
        signs.push(lam_ref < 0.0);
    }
    let changes = signs.windows(2).any(|p| p[0] != p[1]);
    assert_eq!(
        changes, expect_sign_change,
        "{system}: sign-change expectation"
    );
}

#[test]
#[ignore = "validation: RSH omega tuning"]
fn n2_cation_lambda_min_tracks_pyscf_across_the_sign_change() {
    check_lambda("n2", true);
}

#[test]
#[ignore = "validation: RSH omega tuning"]
fn h2o_cation_is_stable_and_not_flagged() {
    check_lambda("h2o", false);
}

#[test]
#[ignore = "validation: RSH omega tuning"]
fn nh3_cation_is_stable_and_not_flagged() {
    check_lambda("nh3", false);
}

/// A tuned ω* on a saddle cation is an error naming ω and λ_min; the same
/// system below the onset tunes cleanly with no stability warning.
#[test]
#[ignore = "validation: RSH omega tuning"]
fn tune_omega_refuses_a_saddle_cation_and_accepts_a_stable_one() {
    let sys = load("n2");
    let above = tune_omega(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        &config(FUNCTIONAL, 0.60, 0.90, true),
    );
    let msg = match above {
        Err(e) => format!("{e}"),
        Ok(r) => panic!(
            "N2+ saddle bracket returned omega* = {} without error",
            r.omega
        ),
    };
    eprintln!("refusal: {msg}");
    assert!(msg.contains("SADDLE") && msg.contains("λ_min = -"), "{msg}");

    let below = tune_omega(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        &config(FUNCTIONAL, 0.20, 0.45, true),
    )
    .expect("stable bracket must tune");
    assert!(
        below.stability_warning.is_none(),
        "{:?}",
        below.stability_warning
    );
    assert!(below
        .evals
        .iter()
        .all(|e| e.cation_stability.is_proven_stable()));

    // Mixed bracket straddling the onset: ω* may land on either side, but it
    // is never returned on a saddle.
    if let Ok(r) = tune_omega(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        &config(FUNCTIONAL, 0.30, 0.80, true),
    ) {
        let best = r
            .evals
            .iter()
            .find(|e| e.omega == r.omega)
            .expect("omega* is an evaluated point");
        assert!(!best.cation_stability.is_saddle());
    }
}

/// Exactness anchor: with the check off nothing changes, bit for bit, and
/// every eval says it was not checked.
#[test]
#[ignore = "validation: RSH omega tuning"]
fn check_off_is_bit_identical_and_reports_not_checked() {
    let sys = load("h2o");
    let on = config(FUNCTIONAL, 0.1, 1.0, true);
    let off = config(FUNCTIONAL, 0.1, 1.0, false);
    let a = eval_j(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &on, 0.4).unwrap();
    let b = eval_j(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &off, 0.4).unwrap();
    assert_eq!(a.j.to_bits(), b.j.to_bits());
    assert_eq!(a.e_cation.to_bits(), b.e_cation.to_bits());
    assert_eq!(a.eps_homo.to_bits(), b.eps_homo.to_bits());
    assert_eq!(b.cation_stability, CationStability::NotChecked);
    assert!(a.cation_stability.is_proven_stable());
}

/// VV10 (ωB97X-V) cannot be analysed: every eval says so, the result warns,
/// and nothing is ever read as stable.
#[test]
#[ignore = "validation: RSH omega tuning"]
fn vv10_functional_is_reported_not_analysed_never_stable() {
    let sys = load("h2o");
    let r = tune_omega(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        &OmegaTuneConfig {
            max_evals: 2,
            ..config("wB97X-V", 0.3, 0.6, true)
        },
    )
    .expect("VV10 tuning runs; it just cannot certify stability");
    assert!(r.evals.iter().all(|e| matches!(
        e.cation_stability,
        CationStability::NotAnalysed(StabilitySkip::Vv10Kernel)
    )));
    assert!(r
        .evals
        .iter()
        .all(|e| !e.cation_stability.is_proven_stable()));
    let w = r.stability_warning.expect("must warn");
    assert!(w.contains("NOT ANALYSED") && w.contains("VV10"), "{w}");
}
