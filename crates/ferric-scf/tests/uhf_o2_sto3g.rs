//! **O₂ triplet / STO-3G UHF: three stationary points, and which one ferric
//! returns.**
//!
//! # The report this file answers
//!
//! "UHF on O₂ (M = 3) / STO-3G reports `converged = true` at an energy 0.255 Ha
//! ABOVE PySCF's UHF and above ferric's own ROHF; at cc-pVDZ it agrees."
//!
//! Reproduced (2026-09-23) with the extension built from the main checkout's
//! `feat/esp-on-surface` branch, whose `uhf.rs` PREDATES #83 (`396e0d61`, "open-shell
//! SCF discarded its initial guess"): that build gives −147.37890121 Ha. On
//! `origin/main` (which carries #83) the default path gives −147.63396964 and
//! the 0.255 Ha state is reachable only through `use_sad_guess = false`.
//!
//! # The three states (PySCF 2.13.0, `scripts/gen_pyscf_o2_uhf_ref.py`)
//!
//! ```text
//!   hcore/1e/huckel guess  -147.3789011760   lambda_min -2.63e-1  SADDLE (the report)
//!   minao/atom guess       -147.6339696085   lambda_min -3.06e-2  SADDLE (doubly degenerate)
//!   stability-followed     -147.6352963519   lambda_min ~ -2e-9   minimum (+ a zero mode)
//!   ROHF                   -147.6321910013
//! ```
//!
//! The hcore state is aufbau-consistent in both spins (α HOMO −0.342 / LUMO
//! +0.703, β HOMO −0.287 / LUMO +0.032) — it is NOT a non-aufbau occupation,
//! it is a genuine higher stationary point with the π/π* shells split
//! unevenly. PySCF lands on it from three of its own five guesses.
//!
//! The minimum breaks the x/y equivalence of the SPIN density on each O
//! (2px/2py spin populations 0.561/0.439, mirrored on the other atom) while the
//! total density stays centrosymmetric. Rotating that polarization about the
//! bond axis costs nothing, hence the ~0 Hessian eigenvalue there: ferric's
//! verdict on it is expected to be STABLE or MARGINAL, never UNSTABLE.
//!
//! At 6-31G and cc-pVDZ the default-guess state is already the minimum (0
//! stability-following rounds in PySCF), which is why the report saw agreement
//! at cc-pVDZ.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;

/// Same tolerance as the `*_uhf.json` checks in `tests/uhf.rs`. The states
/// this file separates are 1.3e-3 and 2.6e-1 Ha apart, three and five orders
/// of magnitude above it.
const TOL: f64 = 1e-6;

/// O₂ at R = 1.208 Å — the geometry of `testdata/molecules/o2.xyz` and of both
/// `o2_sto-3g_{uhf,rohf}.json`.
const O2_XYZ: &str = "2\nO2\nO 0.0 0.0 0.0\nO 0.0 0.0 1.208\n";

fn reference() -> serde_json::Value {
    let path = "../../testdata/reference/o2_sto-3g_uhf.json";
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path} missing ({e}); run scripts/gen_pyscf_o2_uhf_ref.py"));
    serde_json::from_str(&text).unwrap()
}

fn ref_energy(key: &str) -> f64 {
    reference()[key]
        .as_f64()
        .unwrap_or_else(|| panic!("o2_sto-3g_uhf.json has no numeric \"{key}\""))
}

fn rohf_json_energy(slug: &str) -> f64 {
    let path = format!("../../testdata/reference/{slug}");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    v["energy"].as_f64().unwrap()
}

struct Sys {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn sys(xyz: &str, charge: i32, mult: usize, basis_name: &str) -> Sys {
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Sys {
        mol,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

fn o2(basis_name: &str) -> Sys {
    sys(O2_XYZ, 0, 3, basis_name)
}

/// Tight, plain config: default guess (MINAO), no descent, no stability check.
fn tight_cfg() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        density_conv: 1e-10,
        energy_conv: 1e-11,
        ..Default::default()
    }
}

fn uhf(s: &Sys, cfg: &RhfConfig) -> ScfResult {
    solve_uhf(&s.ctx, &s.mol, &s.prep, &s.bounds, cfg).unwrap()
}

fn rohf(s: &Sys, cfg: &RhfConfig) -> ScfResult {
    solve_rohf(
        &s.ctx,
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        cfg,
    )
    .unwrap()
}

fn verdict(r: &ScfResult) -> StabilityVerdict {
    r.stability
        .as_ref()
        .expect("check_stability was set, so a verdict must be present")
        .verdict()
}

fn lambda_min(r: &ScfResult) -> f64 {
    r.stability
        .as_ref()
        .map(|s| s.lowest_eigenvalue)
        .unwrap_or(f64::NAN)
}

/// **The default path does not return the reported 0.255 Ha state.**
///
/// It lands on PySCF's default-guess state (itself a saddle — see
/// [`the_default_state_is_a_saddle_and_the_descent_reaches_the_uhf_minimum`]),
/// not on the hcore saddle the report saw.
///
/// IF REVERTED (the #83 guess fix undone, i.e. `uhf_guess_mos` returning
/// `None` so the SCF starts from bare hcore): E = −147.3789012, and the first
/// assert fails with |dE| = 0.255 Ha.
#[test]
fn default_uhf_does_not_land_on_the_hcore_saddle() {
    let e_default = ref_energy("energy_default_guess");
    let e_hcore = ref_energy("energy_hcore_guess");
    let r = uhf(&o2("sto-3g"), &tight_cfg());
    println!(
        "O2/STO-3G default UHF: E = {:.10} (PySCF default {e_default:.10}, hcore saddle \
         {e_hcore:.10}), {} iters",
        r.energy, r.iterations
    );
    assert!(r.converged);
    assert!(
        (r.energy - e_default).abs() < TOL,
        "O2/STO-3G default UHF E = {:.10}, expected PySCF's default-guess state {e_default:.10} \
         (|dE| = {:.3e} Ha)",
        r.energy,
        (r.energy - e_default).abs()
    );
    assert!(
        r.energy < e_hcore - 0.2,
        "O2/STO-3G default UHF E = {:.10} is at/near the hcore saddle {e_hcore:.10}",
        r.energy
    );
}

/// **NEGATIVE CONTROL: the trap is real on this input.** From the bare hcore
/// guess ferric converges — `converged = true` — onto exactly the state the
/// report described, its own stability analysis calls it UNSTABLE, and it lies
/// ABOVE ROHF, so the variational guard below is REACHABLE (it would fire on
/// this state). Without this, the two tests above/below could pass on a
/// system that simply has no second basin.
///
/// FAILS IF: `use_sad_guess = false` stopped reaching the solver (E would be
/// the default state, first assert); the stability analysis lost the
/// instability (verdict assert); or the ROHF/UHF comparison were somehow
/// inverted (last assert).
#[test]
fn hcore_guess_reproduces_the_reported_saddle_and_it_is_flagged() {
    let e_hcore = ref_energy("energy_hcore_guess");
    let s = o2("sto-3g");
    let cfg = RhfConfig {
        use_sad_guess: false,
        check_stability: true,
        ..tight_cfg()
    };
    let r = uhf(&s, &cfg);
    let e_rohf = rohf(&s, &tight_cfg()).energy;
    println!(
        "O2/STO-3G hcore-guess UHF: E = {:.10} (PySCF hcore {e_hcore:.10}), converged = {}, \
         lambda_min = {:+.4e}, {}; ROHF = {e_rohf:.10}",
        r.energy,
        r.converged,
        lambda_min(&r),
        verdict(&r).label()
    );
    assert!(r.converged, "the reported symptom includes converged = true");
    assert!(
        (r.energy - e_hcore).abs() < TOL,
        "hcore guess gave E = {:.10}, not the reported saddle {e_hcore:.10}; this file's \
         premise (a second basin at +0.255 Ha) is gone",
        r.energy
    );
    assert_eq!(
        verdict(&r),
        StabilityVerdict::Unstable,
        "the 0.255 Ha state must be flagged UNSTABLE (lambda_min = {:+.4e})",
        lambda_min(&r)
    );
    assert!(
        r.energy > e_rohf + 0.2,
        "the hcore saddle ({:.10}) should lie above ROHF ({e_rohf:.10}); otherwise the \
         UHF <= ROHF guard could never fire on it",
        r.energy
    );
}

/// **The default state is itself a saddle; the opt-in descent reaches the UHF
/// minimum.** This is the same split as N₂⁺/6-31G in `scf_state_selection.rs`:
/// the guess fix alone gets PySCF's default answer, and — exactly like PySCF —
/// following the internal instability is what reaches the bottom.
///
/// IF REVERTED (descent disabled, or `stability_descent` returning its input):
/// the descended E stays at −147.6339696, 1.33e-3 Ha above the reference, and
/// the third assert fails. If the stability analysis stopped seeing the
/// instability, the second assert fails (and the descent would never run).
#[test]
fn the_default_state_is_a_saddle_and_the_descent_reaches_the_uhf_minimum() {
    let e_min = ref_energy("energy");
    let e_default = ref_energy("energy_default_guess");
    let s = o2("sto-3g");

    let checked = uhf(
        &s,
        &RhfConfig {
            check_stability: true,
            ..tight_cfg()
        },
    );
    let descended = uhf(
        &s,
        &RhfConfig {
            check_stability: true,
            scf_stability_descent: true,
            ..tight_cfg()
        },
    );
    println!(
        "O2/STO-3G: default E = {:.10} ({}, lambda_min {:+.4e}); descent E = {:.10} ({}, \
         lambda_min {:+.4e}); PySCF minimum {e_min:.10}",
        checked.energy,
        verdict(&checked).label(),
        lambda_min(&checked),
        descended.energy,
        verdict(&descended).label(),
        lambda_min(&descended)
    );

    assert!((checked.energy - e_default).abs() < TOL);
    assert_eq!(
        verdict(&checked),
        StabilityVerdict::Unstable,
        "PySCF finds lambda_min = -3.06e-2 at this state; ferric reported {:+.4e}",
        lambda_min(&checked)
    );
    assert!(descended.converged);
    assert!(
        (descended.energy - e_min).abs() < TOL,
        "the descent did not reach the O2/STO-3G UHF minimum: E = {:.10} vs {e_min:.10} \
         (|dE| = {:.3e} Ha)",
        descended.energy,
        (descended.energy - e_min).abs()
    );
    // The minimum has a zero mode (rotation of the spin polarization about the
    // bond axis), so MARGINAL is a correct verdict there; UNSTABLE is not.
    assert_ne!(
        verdict(&descended),
        StabilityVerdict::Unstable,
        "the descended state is still reported UNSTABLE (lambda_min = {:+.4e})",
        lambda_min(&descended)
    );
}

/// **The descent also escapes the reported 0.255 Ha saddle**, not just the
/// small one: from the bare hcore guess, check + descent reaches the minimum.
/// PySCF needs two stability-following rounds from here (hcore saddle →
/// default saddle → minimum); `MAX_DESCENT_ROUNDS` is 3.
///
/// IF REVERTED (descent off): E stays at −147.3789012, |dE| = 0.256 Ha.
/// If the descent were capped at ONE round, E would stop at the default-guess
/// saddle (−147.6339696), |dE| = 1.33e-3 Ha — so this test also pins that the
/// multi-round loop is live.
#[test]
fn the_descent_also_escapes_the_hcore_saddle() {
    let e_min = ref_energy("energy");
    let s = o2("sto-3g");
    let r = uhf(
        &s,
        &RhfConfig {
            use_sad_guess: false,
            check_stability: true,
            scf_stability_descent: true,
            ..tight_cfg()
        },
    );
    println!(
        "O2/STO-3G hcore + descent: E = {:.10} ({}, lambda_min {:+.4e}); minimum {e_min:.10}",
        r.energy,
        verdict(&r).label(),
        lambda_min(&r)
    );
    assert!(r.converged);
    assert!(
        (r.energy - e_min).abs() < TOL,
        "hcore + descent ended at E = {:.10}, not the UHF minimum {e_min:.10} \
         (|dE| = {:.3e} Ha)",
        r.energy,
        (r.energy - e_min).abs()
    );
}

/// **Variational sanity guard: UHF <= ROHF**, on the default path, for the
/// same system and basis. UHF's variational space contains ROHF's, so a UHF
/// energy above ROHF can only be a non-minimum UHF stationary point.
///
/// Every ROHF number is ANCHORED to an external PySCF reference first, so a
/// ROHF that drifted UP onto a wrong state cannot make this guard vacuous by
/// raising the bar.
///
/// IF REVERTED (UHF back on the hcore guess): O2/STO-3G UHF = −147.3789 vs
/// ROHF −147.6322 and OH/6-31G UHF = −75.2080 vs ROHF −75.3618 both fail the
/// `<=` assert (reachability shown directly in
/// [`hcore_guess_reproduces_the_reported_saddle_and_it_is_flagged`]).
#[test]
fn uhf_is_never_above_rohf_for_the_same_system() {
    // OH/6-31G ROHF, R = 0.97 Å: PySCF 2.13.0, from rohf_state_selection.rs.
    const OH_631G_ROHF: f64 = -75.361_846_292_5;
    let ch3_xyz = "4\nCH3\n\
                   C 0.0 0.0 0.0\n\
                   H 1.079 0.0 0.0\n\
                   H -0.5395 0.934441 0.0\n\
                   H -0.5395 -0.934441 0.0\n";
    let oh_xyz = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
    let cases: Vec<(&str, Sys, Option<f64>)> = vec![
        (
            "O2/STO-3G",
            o2("sto-3g"),
            Some(rohf_json_energy("o2_sto-3g_rohf.json")),
        ),
        ("O2/6-31G", o2("6-31g"), None),
        ("OH/6-31G", sys(oh_xyz, 0, 2, "6-31g"), Some(OH_631G_ROHF)),
        (
            "OH/cc-pVDZ",
            sys(oh_xyz, 0, 2, "cc-pvdz"),
            Some(rohf_json_energy("oh_cc-pvdz_rohf.json")),
        ),
        (
            "CH3/cc-pVDZ",
            sys(ch3_xyz, 0, 2, "cc-pvdz"),
            Some(rohf_json_energy("ch3_cc-pvdz_rohf.json")),
        ),
    ];
    let mut failures = Vec::new();
    for (name, s, rohf_ref) in &cases {
        let u = uhf(s, &tight_cfg());
        let r = rohf(s, &tight_cfg());
        println!(
            "{name:12} UHF = {:.10}  ROHF = {:.10}  UHF-ROHF = {:+.3e} Ha{}",
            u.energy,
            r.energy,
            u.energy - r.energy,
            rohf_ref
                .map(|e| format!("  (ROHF ref {e:.10})"))
                .unwrap_or_default()
        );
        assert!(u.converged && r.converged, "{name}: SCF did not converge");
        if let Some(e_ref) = rohf_ref {
            if (r.energy - e_ref).abs() > TOL {
                failures.push(format!(
                    "{name}: ROHF anchor broken, {:.10} vs PySCF {e_ref:.10}",
                    r.energy
                ));
            }
        }
        // 1e-8: both SCFs are converged to dP 1e-10, so their energies are
        // good to well below this; a genuine violation is >= 1e-3 Ha here.
        if u.energy > r.energy + 1e-8 {
            failures.push(format!(
                "{name}: UHF {:.10} is ABOVE ROHF {:.10} by {:.3e} Ha",
                u.energy,
                r.energy,
                u.energy - r.energy
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "variational guard failed:\n  {}",
        failures.join("\n  ")
    );
}
