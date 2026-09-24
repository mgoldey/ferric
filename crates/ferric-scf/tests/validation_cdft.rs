//! VALIDATION tier — VALIDATION.md row "cDFT".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_cdft \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's constrained UKS (`solve_cdft_uhf`, Becke fragment populations,
//! PBE, 99 radial x 302 Lebedev grid) against NWChem 7.2.2 `cdft ... pop becke`
//! on three systems x two bases (6-31G and def2-SVP, both fed to NWChem from
//! ferric's bundled JSON):
//!
//! | system | fragment | constraints (targets) |
//! |---|---|---|
//! | LiH | Li | charge 2.60, 2.75, 2.95 (natural 2.83) |
//! | HF | F | charge 8.80, 8.90, 9.10 (natural 9.00-9.03) |
//! | H2O+ (doublet) | O | charge 7.30, 7.40, 7.60 (natural 7.49-7.51); spin N_a - N_b 0.90 (natural 0.97-0.98) |
//!
//! References: `scripts/validation/gen_cdft.py` ->
//! `testdata/reference/validation/cdft/<system>_<basis>.json`, NWChem inputs in
//! `scripts/validation/nwchem/cdft/`.
//!
//! # Like-for-like (read from the NWChem source; details in the generator)
//!
//! * Same Becke partition: Becke 1988 cell functions with the Bragg-Slater
//!   size adjustment, identical radii for H/Li/O/F, `grid ... becke`.
//! * Same sign convention: both add `+lambda W` to the Fock and drive
//!   `Tr[W D] - target` to zero, so `lambda` compares directly and
//!   `dE/dN_target = -lambda` in both. NWChem's `charge q` input is converted
//!   by the generator to the electron target `N = Z_frag - q`.
//! * Different W QUADRATURE. NWChem integrates `w_C chi chi` over the grid
//!   points whose home atom is in the fragment; ferric integrates
//!   `w_C(r) chi chi` over all points. Both are quadratures of the same
//!   integral. On LiH (diffuse Li functions) this gives the largest residual:
//!   ferric's own lambda moves by 4.4e-6 between 99 and 250 radial points at
//!   302 angular, and ferric's Lebedev table stops at 302. The test therefore
//!   compares against NWChem at its GRID LIMIT (300 x 974, "converged"), and
//!   compares `E(N) - E_unconstrained` rather than `E(N)`, which cancels the XC
//!   grid offset. The matched-grid (99 x 302) NWChem numbers are printed.
//! * ferric's W itself was reproduced by an independent numpy build (Treutler
//!   M4 x Lebedev 302 grid, Becke with size adjustment, PySCF AO values) to
//!   7e-16 on LiH/def2-SVP, so the residual above is quadrature, not W's
//!   construction.
//! * Pure HF cannot be the functional: NWChem builds the Becke W only on its
//!   XC grid, which it skips for `xc hfexch`, and the cdft solve then aborts.
//!
//! # Exactness anchor and internal identity (NWChem-independent)
//!
//! 1. TRIVIAL LIMIT. With the target set to the unconstrained fragment
//!    population `Tr[W D_unc]`, the constraint does nothing: `lambda = 0` and
//!    `E = E_unc`. Checked first on every system/basis.
//! 2. `dE/dN = -lambda`. The constrained energy is a Legendre-type function of
//!    the target, so `E(T+h) - E(T-h) = -int_{T-h}^{T+h} lambda dN`. Evaluated
//!    with Simpson's rule on the three solves the central difference needs
//!    (`-(h/3)(lambda_+ + 4 lambda_0 + lambda_-)`, error O(h^5)). This IS the
//!    central-difference identity `(E_+ - E_-)/(2h) = -lambda_0` with its
//!    O(h^2) truncation term `-(lambda_+ - 2 lambda_0 + lambda_-)/6` included
//!    exactly; the bare central difference is also asserted, at a bar set by
//!    that truncation (measured 7.3e-7 on LiH at h = 1e-3).
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric's cDFT is right: the fragment population equals the target to
//!   1e-9; `lambda` and `E(N) - E_unc` agree with NWChem's grid-limit values to
//!   the W-quadrature floor (<= 1.2e-6 in lambda, <= 2e-7 Ha in energy); the
//!   Simpson residual is at SCF noise (~1e-12 Ha).
//! * If the W operator, partition or sign is wrong: the fragment population
//!   of a given density moves by 0.09-0.86 e (measured on the def2-SVP PBE
//!   densities with an independent numpy Becke build: dropping the size
//!   adjustment -0.39..-0.86 e, Mulliken instead of Becke -0.12..+0.21 e), so
//!   lambda moves by ~0.1 (dlambda/dN ~ -1.3 on LiH) — four orders of
//!   magnitude above the bar. A flipped lambda sign or a `lambda N` term left
//!   in the reported energy fails the Simpson identity by ~1e-3 Ha.
//! * If the HARNESS is broken (units, basis, wrong file): the nuclear-repulsion
//!   and AO-count checks fail first; a basis swap moves lambda by >= 4e-3 (LiH
//!   2.60: 0.2177 vs 0.2221), far above the bar.
//!
//! # TOLERANCES (set from the measured floor recorded on each const)
//!
//! Measured 2026-09-24 by this test and, independently, through the `ferric`
//! Python binding (`run_cdft`, same grid and thresholds); the two agree on
//! every worst case below. All 6 system/basis pairs:
//!
//! | quantity | measured max |d| | worst case |
//! |---|---:|---|
//! | unconstrained E vs NWChem 99x302 | 1.04e-8 Ha | LiH / 6-31G |
//! | E(N) - E_unc vs NWChem 300x974 | 1.96e-7 Ha | LiH / 6-31G, 2.60 |
//! | lambda vs NWChem 300x974 | 1.14e-6 | LiH / 6-31G, 2.60 |
//! | anchor lambda at natural target | 4.7e-9 | LiH / def2-SVP |
//! | Simpson residual (abs), def2-SVP | 2.3e-12 Ha | LiH / def2-SVP, 2.60 (this test; the binding also measured 4.5e-12 on H2O+ / 6-31G spin, which this test does not run) |
//! | population - target | 4.7e-10 | H2O+ / def2-SVP spin |
//!
//! The binding's `run_dft` is closed-shell only, so for H2O+ the unconstrained
//! energy in that measurement was the cDFT energy at the natural target found
//! by secant on lambda (|lambda| < 1e-8, i.e. E_unc to O(lambda^2)); the anchor
//! row above is therefore measured on LiH and HF only. This test computes
//! E_unc for H2O+ directly with `solve_uhf`.
//!
//! # NEGATIVE CONTROLS (asserted inside the test)
//!
//! * The lambda (and `E - E_unc`) ferric reports for one target must MISS
//!   NWChem's value for every OTHER target on the same system/basis by far more
//!   than the bar — so the lambda bar distinguishes neighbouring constraints
//!   (nearest pair: 0.068 apart vs a 5e-6 bar).
//! * The Simpson identity with the sign of lambda flipped (`dE/dN = +lambda`)
//!   must fail by >> its bar.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::ao_grid::eval_basis_on_points;
use ferric_dft::cdft::{build_weight_matrix, population, Constraint, SpinChannel};
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cdft_driver::{solve_cdft_uhf, CdftResult};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/cdft";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASES: [&str; 2] = ["6-31g", "def2-svp"];
/// Basis on which the dE/dN identity is run (one basis keeps the runtime down;
/// the identity is basis-independent and is still exercised on 3 systems).
const FD_BASIS: &str = "def2-svp";
/// Target step for the dE/dN identity.
const FD_H: f64 = 1e-3;

const TOL_ENUC: f64 = 1e-9;
// Bar 5e-8 Ha = 5x the measured 1.04e-8 (LiH/6-31G); others
// <= 3.4e-9. Same functional, same (unpruned 99x302 Becke) grid on both sides.
const TOL_E_UNC: f64 = 5e-8;
// Bar 1e-6 Ha = 5x the measured 1.96e-7 (LiH/6-31G 2.60 vs
// NWChem 300x974); every other entry <= 9.5e-8.
const TOL_DE: f64 = 1e-6;
// Bar 5e-6 = 4.4x the measured 1.14e-6 (LiH/6-31G 2.60);
// every other entry <= 5.9e-7.
const TOL_LAMBDA: f64 = 5e-6;
// Bar 1e-9 electrons; driver lambda_tol is 1e-10 here, measured
// max |N - target| 4.7e-10 at lambda_tol 1e-9.
const TOL_POP: f64 = 1e-9;
// Bar 1e-7; this test measures lambda = 0 exactly at the natural target (one
// outer iteration) for all three systems, H2O+ included; the binding's secant
// search gave <= 4.7e-9.
const TOL_ANCHOR_LAMBDA: f64 = 1e-7;
// Bar 1e-9 Ha; measured |E - E_unc| at the natural target
// <= 7e-14 (the lambda = 5e-9 residual contributes ~lambda*dN ~ 1e-17).
const TOL_ANCHOR_E: f64 = 1e-9;
// Bar 5e-11 Ha = 21x this test's measured 2.3e-12 (LiH/def2-SVP 2.60);
// all four cases 1.6e-13..2.3e-12. A 1e-7 relative error in lambda already
// exceeds this bar (|E_+ - E_-| ~ 4e-4 Ha for charge targets).
const TOL_SIMPSON: f64 = 5e-11;
// Bare central difference |dE/dN + lambda_0|, dominated by its
// O(h^2) truncation (lambda'')/6: measured 7.3e-7 on LiH/def2-SVP 2.60.
const TOL_FD_CENTRAL: f64 = 5e-6;
/// A reference ferric must MISS (different target / flipped sign).
const MUST_MISS_LAMBDA: f64 = 1000.0 * TOL_LAMBDA;
const MUST_MISS_DE: f64 = 100.0 * TOL_DE;
const MUST_MISS_SIMPSON: f64 = 1e6 * TOL_SIMPSON;

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
fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_cdft.py — a missing reference is a failure, never a skip",
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

/// The grid both the XC quadrature and the cDFT weight operator use (the
/// driver's default, set explicitly so the test's own W is the driver's W).
fn grid_cfg() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 99,
        n_angular: 302,
        ..Default::default()
    }
}

/// Unconstrained UKS/PBE config; exact J (no `df_j_aux`), as NWChem `direct`.
fn base_config() -> RhfConfig {
    RhfConfig {
        xc: Some("PBE".into()),
        dft_grid: Some(grid_cfg()),
        max_iter: 500,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    }
}

struct System {
    tag: String,
    mol: Molecule,
    bs: basis::BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
    fragment: Vec<usize>,
}

fn load_system(system: &str, basis_name: &str, r: &Value) -> System {
    let tag = format!("{system}/{basis_name}");
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    // Geometry like-for-like FIRST.
    let enuc_ref = num(r, "/nuclear_repulsion", &tag);
    let enuc = mol.nuclear_repulsion();
    eprintln!(
        "{tag}: E_nuc ferric {enuc:.12} NWChem {enuc_ref:.12} |d| {:.2e}",
        (enuc - enuc_ref).abs()
    );
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{tag}: nuclear repulsion {enuc:.12} vs NWChem {enuc_ref:.12} — geometry/unit mismatch"
    );
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let nao_ref = r["nao"].as_u64().expect("nao") as usize;
    assert_eq!(
        prep.nbasis(),
        nao_ref,
        "{tag}: AO count differs from the reference's"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let fragment: Vec<usize> = r["fragment"]
        .as_array()
        .expect("fragment")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    System {
        tag,
        mol,
        bs,
        prep,
        bounds,
        ctx: ParallelContext::default(),
        fragment,
    }
}

fn spin_of(kind: &str) -> SpinChannel {
    match kind {
        "charge" => SpinChannel::Total,
        "spin" => SpinChannel::SpinDiff,
        other => panic!("unknown constraint kind {other:?} in reference"),
    }
}

fn solve_constrained(s: &System, spin: SpinChannel, target: f64) -> CdftResult {
    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: s.fragment.clone(),
            spin,
            target,
        }],
        cdft_lambda_tol: 1e-10,
        cdft_max_outer: 60,
        ..base_config()
    };
    let r = solve_cdft_uhf(&s.ctx, &s.mol, &s.prep, &s.bs, &s.bounds, &cfg)
        .unwrap_or_else(|e| panic!("{}: cDFT {spin:?} target {target}: {e:?}", s.tag));
    assert!(
        r.scf.converged,
        "{}: inner SCF not converged at target {target}",
        s.tag
    );
    let pop = r.populations[0];
    assert!(
        (pop - target).abs() < TOL_POP,
        "{}: {spin:?} population {pop:.12} misses target {target} by {:.2e}",
        s.tag,
        (pop - target).abs()
    );
    r
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<22} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} ferric {got:.12} vs reference {want:.12}: |d| {d:.2e} >= {tol:.0e}"
    );
}

/// Run the whole row for one system (both bases).
fn run_system(system: &str) {
    for basis_name in BASES {
        let r = reference(system, basis_name);
        let s = load_system(system, basis_name, &r);
        let tag = s.tag.clone();

        // ── unconstrained UKS, and the same-grid floor vs NWChem ──
        let unc = solve_uhf(&s.ctx, &s.mol, &s.prep, &s.bounds, &base_config())
            .unwrap_or_else(|e| panic!("{tag}: unconstrained UKS: {e:?}"));
        assert!(unc.converged, "{tag}: unconstrained UKS not converged");
        let e_unc = unc.energy;
        check_close(
            &tag,
            "E_unc vs NWChem 99x302",
            e_unc,
            num(&r, "/unconstrained/matched/energy", &tag),
            TOL_E_UNC,
        );
        eprintln!(
            "{tag}: E_unc - NWChem 300x974 = {:+.2e} (XC grid offset; cancels in E(N)-E_unc)",
            e_unc - num(&r, "/unconstrained/converged/energy", &tag)
        );

        // ── EXACTNESS ANCHOR: target = natural population ⇒ λ = 0, E = E_unc ──
        let grid = build_atomic_grid(&s.mol, &grid_cfg());
        let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let chi = eval_basis_on_points(&s.mol, &s.bs, &pts).unwrap();
        let w: Array2<f64> = build_weight_matrix(&s.mol, &grid, &chi, &s.fragment);
        let d_a = &unc.density_alpha;
        let d_b = unc.density_beta.as_ref().unwrap_or(d_a);
        let n_nat = population(&w, d_a, d_b, &SpinChannel::Total);
        let anchor = solve_constrained(&s, SpinChannel::Total, n_nat);
        let dw = (&anchor.weight_matrices[0] - &w)
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()));
        assert!(
            dw < 1e-12,
            "{tag}: the driver's W differs from the test's W by {dw:.2e} — not the same grid"
        );
        eprintln!(
            "{tag}: ANCHOR N_nat {n_nat:.9} lambda {:+.2e} E-E_unc {:+.2e} outer {}",
            anchor.lambdas[0],
            anchor.scf.energy - e_unc,
            anchor.outer_iters
        );
        assert!(
            anchor.lambdas[0].abs() < TOL_ANCHOR_LAMBDA,
            "{tag}: lambda at the natural population is {:.3e}, not 0",
            anchor.lambdas[0]
        );
        assert!(
            (anchor.scf.energy - e_unc).abs() < TOL_ANCHOR_E,
            "{tag}: E at the natural population differs from E_unc by {:.3e}",
            anchor.scf.energy - e_unc
        );

        // ── constrained energies and multipliers vs NWChem ──
        let cons = r["constrained"].as_array().expect("constrained");
        let mut got: Vec<(f64, f64)> = Vec::new(); // (λ, E - E_unc) per entry
        let mut first_of_kind: Vec<(String, f64, CdftResult)> = Vec::new();
        for c in cons {
            let kind = c["kind"].as_str().expect("kind");
            let target = num(c, "/target", &tag);
            let ctx = format!("{tag} {kind} {target:.2}");
            let res = solve_constrained(&s, spin_of(kind), target);
            let lam = res.lambdas[0];
            let de = res.scf.energy - e_unc;
            eprintln!(
                "{ctx}: vs NWChem 99x302 (info): dlambda {:+.2e} d(E-E_unc) {:+.2e}; outer {}",
                lam - num(c, "/matched/lambda", &ctx),
                de - num(c, "/matched/e_minus_unconstrained", &ctx),
                res.outer_iters
            );
            check_close(
                &ctx,
                "lambda",
                lam,
                num(c, "/converged/lambda", &ctx),
                TOL_LAMBDA,
            );
            check_close(
                &ctx,
                "E(N) - E_unc",
                de,
                num(c, "/converged/e_minus_unconstrained", &ctx),
                TOL_DE,
            );
            got.push((lam, de));
            if !first_of_kind.iter().any(|(k, _, _)| k == kind) {
                first_of_kind.push((kind.to_string(), target, res));
            }
        }

        // ── NEGATIVE CONTROL: another target's reference must be missed ──
        for (i, (lam, de)) in got.iter().enumerate() {
            for (j, c) in cons.iter().enumerate() {
                if i == j {
                    continue;
                }
                let lam_j = num(c, "/converged/lambda", &tag);
                let de_j = num(c, "/converged/e_minus_unconstrained", &tag);
                assert!(
                    (lam - lam_j).abs() > MUST_MISS_LAMBDA,
                    "{tag}: lambda of entry {i} ({lam:.8}) is within {MUST_MISS_LAMBDA:.0e} of \
                     entry {j}'s reference ({lam_j:.8}) — the lambda bar cannot tell targets apart"
                );
                assert!(
                    (de - de_j).abs() > MUST_MISS_DE,
                    "{tag}: E-E_unc of entry {i} ({de:.10}) is within {MUST_MISS_DE:.0e} of \
                     entry {j}'s reference ({de_j:.10})"
                );
            }
        }

        // ── INTERNAL IDENTITY dE/dN = −λ (one basis, first target of each kind) ──
        if basis_name == FD_BASIS {
            for (kind, t0, r0) in &first_of_kind {
                let spin = spin_of(kind);
                let rp = solve_constrained(&s, spin, t0 + FD_H);
                let rm = solve_constrained(&s, spin, t0 - FD_H);
                let (lp, l0, lm) = (rp.lambdas[0], r0.lambdas[0], rm.lambdas[0]);
                let de = rp.scf.energy - rm.scf.energy;
                let simpson = -(FD_H / 3.0) * (lp + 4.0 * l0 + lm);
                let central = de / (2.0 * FD_H);
                let ctx = format!("{tag} {kind} {t0:.2} dE/dN");
                eprintln!(
                    "{ctx}: E+-E- {de:+.9e} -int(lambda) {simpson:+.9e} resid {:+.2e}; \
                     central dE/dN {central:+.9} vs -lambda {:+.9} (|d| {:.2e}, predicted \
                     O(h^2) term {:+.2e})",
                    de - simpson,
                    -l0,
                    (central + l0).abs(),
                    -(lp - 2.0 * l0 + lm) / 6.0
                );
                assert!(
                    (de - simpson).abs() < TOL_SIMPSON,
                    "{ctx}: E(T+h)-E(T-h) = {de:.12e} but -int lambda dN = {simpson:.12e} \
                     (|d| {:.2e})",
                    (de - simpson).abs()
                );
                assert!(
                    (central + l0).abs() < TOL_FD_CENTRAL,
                    "{ctx}: central dE/dN {central:.10} vs -lambda {:.10}",
                    -l0
                );
                // NEGATIVE CONTROL: the opposite sign convention must fail.
                assert!(
                    (de + simpson).abs() > MUST_MISS_SIMPSON,
                    "{ctx}: the identity also holds with dE/dN = +lambda ({:.2e}) — it has \
                     no resolving power here",
                    (de + simpson).abs()
                );
            }
        }
    }
}

#[test]
#[ignore = "validation: cDFT"]
fn cdft_lih_matches_nwchem_becke() {
    run_system("lih");
}

#[test]
#[ignore = "validation: cDFT"]
fn cdft_hf_matches_nwchem_becke() {
    run_system("hf");
}

#[test]
#[ignore = "validation: cDFT"]
fn cdft_h2o_cation_charge_and_spin_match_nwchem_becke() {
    run_system("h2o");
}
