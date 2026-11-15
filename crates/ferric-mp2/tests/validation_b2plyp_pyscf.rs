//! VALIDATION tier: validation.md Anchors row "B2PLYP double hybrid"
//! (`mp2_double_hybrid` with `DoubleHybridKind::B2plyp`, CLI kind `b2plyp`).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_b2plyp_pyscf \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's whole double hybrid (its own KS SCF, then the scaled RI-MP2 on the
//! KS orbitals) against PySCF built from the DEFINITION of B2PLYP (Grimme,
//! JCP 124, 034108 (2006)): `dft.RKS` with `0.53*HF + 0.47*B88, 0.73*LYP`
//! plus `0.27 (E_OS + E_SS)` from `mp.dfmp2.DFMP2` on those KS orbitals
//! (`scripts/validation/gen_b2plyp_pyscf.py`). H2O, NH3 and CH4 at cc-pVDZ
//! (aux cc-pVDZ-RI) and def2-SVP (aux def2-SVP-RIFIT), frozen core 0 and 1.
//! Compared per case: E_KS, the KS HOMO and LUMO, E_OS and E_SS raw, and the
//! total.
//!
//! # Settings, and the ones that do not match
//!
//! Matched: ferric's basis JSON and Bohr geometry; the (75, 110) Becke grid
//! with Becke radii (PySCF `prune=None`, `becke_atomic_radii_adjust`); RI-J
//! and RI-K over def2-universal-jkfit (J tensor stored in f64); the PT2 aux
//! basis and Coulomb metric; frozen core 0 and 1.
//!
//! NOT matched, because the codes have no common setting:
//!
//! * the CLI's KS default grid is the pruned
//!   `AtomicGridConfig::production()` and its RI-J tensor storage is f32. This
//!   test pins the flat grid and f64 storage (the certificate records both, and
//!   the anchor is attached only to a reference that ran them). The effect of
//!   the defaults on the total energy is measured below.
//!
//! # Bar derivation (every term measured, none guessed)
//!
//! The comparison has no free parameter: same grid, same fits, same orbitals up
//! to SCF convergence. What separates the codes is (a) each code's own
//! run-to-run spread, (b) libcint vs libint integral rounding and grid-weight
//! summation order, and (c) the SCF convergence ferric stops at (energy
//! 1e-10, density 1e-8). Each bar is `SAFETY (4) x` the sum of its terms:
//!
//! | term | measured max over the six cases | where it comes from |
//! |---|---:|---|
//! | (a) PySCF minao vs atom guess, conv 1e-12: E_KS / E_OS / E_SS | 1.6e-13 / 2.5e-12 / 9e-14 | each reference's `reproduction` block (read per case) |
//! | (c) ferric loose vs tight (1e-12 / 1e-10) SCF: E_KS / E_OS / E_SS / total / HOMO,LUMO | 1e-13 / 4.4e-11 / 1.4e-11 / 1.3e-11 / 1.8e-9 | `ferric_scf_convergence_term`, which asserts them |
//! | (b) E_KS left when both SCFs are converged | 7.1e-13 | CH4 / cc-pVDZ |
//!
//! Measured ferric-vs-PySCF maxima against those bars: E_KS 7.1e-13, HOMO
//! 1.9e-9, LUMO 3.6e-10, E_OS 4.1e-11, E_SS 1.5e-11, total 1.3e-11. The
//! E_OS, E_SS and total differences equal the ferric SCF-convergence terms
//! (4.1e-11 against 4.4e-11, 1.5e-11 against 1.4e-11): the PT2 term has no
//! other source of disagreement at this level.
//!
//! The CLI's pruned grid and f32 RI-J storage shift the total by up to 9.9e-7
//! Ha against the pinned settings (`default_grid_and_ri_j_storage_shift_the_total`).
//!
//! # Artifact hypothesis vs physics hypothesis
//!
//! If B2PLYP is right: every field agrees at the integral/SCF-noise level above.
//! If the PT2 fraction is wrong (0.27 mis-set), the total misses by
//! (c - 0.27) (E_OS + E_SS) ~ 1e-3 Ha for a 0.01 slip: five orders over the bar.
//! If the SCF half is wrong (functional weights, HF fraction), E_KS misses by
//! mHa. The two halves are separated because E_KS, E_OS and E_SS are compared
//! as well as the total.
//!
//! # Mutation
//!
//! `testdata/mutations/b2plyp.patch` sets the B2PLYP PT2 fraction in
//! `double_hybrid.rs` from 0.27 to 0.26: the total fails on every case, E_KS,
//! E_OS and E_SS still pass.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::double_hybrid::{mp2_double_hybrid, DoubleHybridKind, Mp2DoubleHybridResult};
use ferric_scf::df_j::RiJPrecision;
use ferric_scf::ladder::{ksdft_ladder, solve_rhf_ladder};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/b2plyp_pyscf";
const MOL_DIR: &str = "testdata/molecules/validation";
const JKFIT: &str = "def2-universal-jkfit";
const TOL_ENUC: f64 = 1e-9;
/// The B2PLYP PT2 fraction of the definition, as the reference stores it.
const C_PT2: f64 = 0.27;

const CASES: [(&str, &str, &str); 6] = [
    ("h2o", "cc-pvdz", "cc-pvdz-ri"),
    ("nh3", "cc-pvdz", "cc-pvdz-ri"),
    ("ch4", "cc-pvdz", "cc-pvdz-ri"),
    ("h2o", "def2-svp", "def2-svp-rifit"),
    ("nh3", "def2-svp", "def2-svp-rifit"),
    ("ch4", "def2-svp", "def2-svp-rifit"),
];

// Bars: SAFETY x (the measured terms of the header table), per case. The
// ferric SCF-convergence terms are the maxima of `ferric_scf_convergence_term`
// over the six cases (rounded up); the PySCF terms are each reference's own
// `reproduction` block; FLOOR is the largest E_KS difference that remains when
// both SCFs are converged (7.1e-13, CH4 / cc-pVDZ), which is libint vs libcint
// rounding and the grid-weight summation order.
const SAFETY: f64 = 4.0;
const CONV_E_KS: f64 = 1.0e-13;
const CONV_E_OS: f64 = 4.5e-11;
const CONV_E_SS: f64 = 1.5e-11;
const CONV_TOTAL: f64 = 1.4e-11;
const CONV_EPS: f64 = 2.0e-9;
const FLOOR: f64 = 1.0e-12;
/// Orbital energies: the SCF-convergence term plus its floor; PySCF's own
/// guess-to-guess spread of an orbital energy is below its energy spread.
const FLOOR_EPS: f64 = 1.0e-10;
/// A control is missed by at least this multiple of the total bar.
const MUST_MISS_FACTOR: f64 = 100.0;

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
             scripts/validation/gen_b2plyp_pyscf.py: a missing reference is a failure, never a skip",
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

/// The KS config of `run_mp2_double_hybrid_arm`, pinned to the settings the
/// reference runs (flat grid, f64 RI-J storage). `tight` tightens the SCF
/// stop by two orders to measure ferric's own convergence term.
fn ks_config(tight: bool) -> RhfConfig {
    RhfConfig {
        xc: Some(DoubleHybridKind::B2plyp.xc_name().into()),
        df_j_aux: Some(JKFIT.to_string()),
        df_k_aux: Some(JKFIT.to_string()),
        ri_j_precision: Some(RiJPrecision::F64),
        energy_conv: if tight { 1e-12 } else { 1e-10 },
        density_conv: if tight { 1e-10 } else { 1e-8 },
        max_iter: 200,
        dft_grid: Some(AtomicGridConfig::default()),
        ..Default::default()
    }
}

struct Run {
    ks: ScfResult,
    mol: Molecule,
    prep: PreparedBasis,
    dfbs: PreparedBasis,
}

fn solve(system: &str, basis_name: &str, aux: &str, tight: bool) -> Run {
    solve_with(system, basis_name, aux, ks_config(tight))
}

fn solve_with(system: &str, basis_name: &str, aux: &str, cfg: RhfConfig) -> Run {
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ladder = ksdft_ladder(&cfg);
    let lr = solve_rhf_ladder(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &ladder,
    )
    .unwrap_or_else(|e| panic!("{system}/{basis_name}: {e}"));
    assert!(
        lr.result.converged,
        "{system}/{basis_name}: KS not converged"
    );
    Run {
        ks: lr.result,
        mol,
        prep,
        dfbs,
    }
}

fn dh(run: &Run, frozen_core: usize) -> Mp2DoubleHybridResult {
    let mut cfg = DoubleHybridKind::B2plyp.mp2_config();
    cfg.frozen_core = frozen_core;
    mp2_double_hybrid(&run.mol, &run.prep, &run.dfbs, &run.ks, &cfg).expect("mp2_double_hybrid")
}

struct Bars {
    e_ks: f64,
    eps: f64,
    e_os: f64,
    e_ss: f64,
    total: f64,
}

fn bars(r: &Value, fc: usize, ctx: &str) -> Bars {
    let rep = |k: &str| num(r, &format!("/reproduction/{k}"), ctx);
    let (ks, os, ss) = (
        rep("e_ks"),
        rep(&format!("fc{fc}_e_os")),
        rep(&format!("fc{fc}_e_ss")),
    );
    Bars {
        e_ks: SAFETY * (CONV_E_KS + ks + FLOOR),
        eps: SAFETY * (CONV_EPS + FLOOR_EPS),
        e_os: SAFETY * (CONV_E_OS + os + FLOOR),
        e_ss: SAFETY * (CONV_E_SS + ss + FLOOR),
        total: SAFETY * (CONV_TOTAL + ks + FLOOR + C_PT2 * (os + ss)),
    }
}

fn check(ctx: &str, what: &str, got: f64, want: f64, tol: f64) -> f64 {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<10} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
    d
}

fn run_case(system: &str, basis_name: &str, aux: &str) {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    let run = solve(system, basis_name, aux, false);
    let enuc_ref = num(&r, "/nuclear_repulsion", &ctx);
    assert!(
        (run.mol.nuclear_repulsion() - enuc_ref).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion differs: geometry/unit mismatch"
    );
    let nocc = run.mol.nelec() as usize / 2;
    let b0 = bars(&r, 0, &ctx);
    check(
        &ctx,
        "E_KS",
        run.ks.energy,
        num(&r, "/ks/energy", &ctx),
        b0.e_ks,
    );
    check(
        &ctx,
        "HOMO",
        run.ks.eps_alpha[nocc - 1],
        num(&r, "/ks/homo", &ctx),
        b0.eps,
    );
    check(
        &ctx,
        "LUMO",
        run.ks.eps_alpha[nocc],
        num(&r, "/ks/lumo", &ctx),
        b0.eps,
    );
    // The functional is applied: B3LYP on the same grid and fit is far away.
    let b3 = num(&r, "/b3lyp_control/energy", &ctx);
    assert!(
        (run.ks.energy - b3).abs() > 1e-3,
        "{ctx}: B2PLYP KS energy is within 1e-3 of the B3LYP control"
    );
    for fc in [0usize, 1] {
        let tag = format!("{ctx} fc{fc}");
        let out = dh(&run, fc);
        let k = format!("/fc{fc}");
        let b = bars(&r, fc, &ctx);
        check(
            &tag,
            "E_OS",
            out.spin_components.e_os,
            num(&r, &format!("{k}/e_os"), &tag),
            b.e_os,
        );
        check(
            &tag,
            "E_SS",
            out.spin_components.e_ss,
            num(&r, &format!("{k}/e_ss"), &tag),
            b.e_ss,
        );
        check(
            &tag,
            "total",
            out.total_energy,
            num(&r, &format!("{k}/total"), &tag),
            b.total,
        );
        // The PT2 term is not negligible, and the total is the definition.
        assert!(out.e_corr_scaled.abs() > 1e-3, "{tag}: PT2 term vanished");
        let must_miss = MUST_MISS_FACTOR * b.total;
        let hf_like = run.ks.energy; // without PT2
        assert!(
            (out.total_energy - hf_like).abs() > must_miss,
            "{tag}: total equals the bare KS energy"
        );
    }
}

/// ferric's own SCF-convergence term: the loose SCF (what the comparison
/// runs) against a two-orders-tighter one. Each difference must stay under the
/// `CONV_*` constants the bars are built from, so the header table is checked
/// by this test and not only quoted.
#[test]
#[ignore = "validation: B2PLYP vs PySCF"]
fn ferric_scf_convergence_term() {
    for (system, basis_name, aux) in CASES {
        let loose = solve(system, basis_name, aux, false);
        let tight = solve(system, basis_name, aux, true);
        let (a, b) = (dh(&loose, 0), dh(&tight, 0));
        let nocc = loose.mol.nelec() as usize / 2;
        let d_eps = (loose.ks.eps_alpha[nocc - 1] - tight.ks.eps_alpha[nocc - 1])
            .abs()
            .max((loose.ks.eps_alpha[nocc] - tight.ks.eps_alpha[nocc]).abs());
        let d = [
            ("E_KS", (loose.ks.energy - tight.ks.energy).abs(), CONV_E_KS),
            (
                "E_OS",
                (a.spin_components.e_os - b.spin_components.e_os).abs(),
                CONV_E_OS,
            ),
            (
                "E_SS",
                (a.spin_components.e_ss - b.spin_components.e_ss).abs(),
                CONV_E_SS,
            ),
            ("total", (a.total_energy - b.total_energy).abs(), CONV_TOTAL),
            ("eps", d_eps, CONV_EPS),
        ];
        for (what, got, term) in d {
            eprintln!(
                "{system}/{basis_name}: loose-vs-tight {what:<6} {got:.2e} (term {term:.1e})"
            );
            assert!(
                got <= term,
                "{system}/{basis_name}: {what} convergence term {got:.2e} exceeds {term:.1e}"
            );
        }
    }
}

/// The CLI's defaults (pruned production grid, f32 RI-J storage) against the
/// pinned settings the evidence runs: the shift of the total energy, which the
/// CLI notice quotes. No PySCF involved; ferric against itself.
#[test]
#[ignore = "validation: B2PLYP vs PySCF"]
fn default_grid_and_ri_j_storage_shift_the_total() {
    let mut worst = 0.0f64;
    for (system, basis_name, aux) in CASES {
        let pinned = solve(system, basis_name, aux, false);
        let default_cfg = RhfConfig {
            dft_grid: None,
            ri_j_precision: None,
            ..ks_config(false)
        };
        let dflt = solve_with(system, basis_name, aux, default_cfg);
        let d = (dh(&pinned, 0).total_energy - dh(&dflt, 0).total_energy).abs();
        eprintln!("{system}/{basis_name}: default-vs-pinned total {d:.2e}");
        worst = worst.max(d);
    }
    eprintln!("default-vs-pinned total, max over cases {worst:.2e}");
    assert!(worst < 1e-3, "defaults move the total by {worst:.2e} Ha");
}

#[test]
#[ignore = "validation: B2PLYP vs PySCF"]
fn b2plyp_h2o_cc_pvdz_vs_pyscf() {
    run_case("h2o", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: B2PLYP vs PySCF"]
fn b2plyp_nh3_cc_pvdz_vs_pyscf() {
    run_case("nh3", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: B2PLYP vs PySCF"]
fn b2plyp_ch4_cc_pvdz_vs_pyscf() {
    run_case("ch4", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: B2PLYP vs PySCF"]
fn b2plyp_h2o_def2_svp_vs_pyscf() {
    run_case("h2o", "def2-svp", "def2-svp-rifit");
}

#[test]
#[ignore = "validation: B2PLYP vs PySCF"]
fn b2plyp_nh3_def2_svp_vs_pyscf() {
    run_case("nh3", "def2-svp", "def2-svp-rifit");
}

#[test]
#[ignore = "validation: B2PLYP vs PySCF"]
fn b2plyp_ch4_def2_svp_vs_pyscf() {
    run_case("ch4", "def2-svp", "def2-svp-rifit");
}
