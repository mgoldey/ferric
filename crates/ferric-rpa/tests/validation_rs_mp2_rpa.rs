//! VALIDATION tier — validation.md row "RS-MP2-RPA B and T" (range-separated
//! MP2 + long-range RPA, `rs_mp2_lr_rpa`, formulations `DeltaLr` (B) and
//! `CoupledRings` (T), erf/erfc attenuator).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-rpa --release \
//!     --test validation_rs_mp2_rpa --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! Every `RsMp2RpaResult` field, and the B and T totals, against an
//! independent numpy assembly on PySCF 2.13.1 density-fitting integrals,
//! both on the same full-Coulomb RHF state:
//!
//! | systems / basis (aux) | ω (Bohr⁻¹) |
//! |---|---|
//! | H2O, NH3 / cc-pVDZ (cc-pvdz-ri); H2O / aug-cc-pVDZ (aug-cc-pvdz-rifit) | 0.222254 (= 0.420 Å⁻¹, `RsMp2RpaConfig::default().omega`), 0.42 |
//!
//! References: `scripts/validation/gen_rs_mp2_rpa.py` →
//! `testdata/reference/validation/rs_mp2_rpa/<system>_<basis>.json`.
//!
//! # The pieces (read from `crates/ferric-rpa/src/rs_mp2_rpa.rs`)
//!
//! One `compute_rpa_intermediates(.., op, ..)` per operator op ∈ {Coulomb,
//! erf(ω), erfc(ω)} (op in the 3-centre integrals AND the metric; eigh with a
//! 1e-10 drop for erf, Cholesky otherwise), fed to both
//! `spin_components_from_b_ov` (SC[op] = E_OS, E_SS, E_MP2) and
//! `run_pdep_rpa_from_intermediates` (E_dRPA[op]):
//!
//! | field | definition | line |
//! |---|---|---|
//! | `e_mp2_full` | SC[Coulomb].E_MP2 | 464 |
//! | `e_sr_mp2` | SC[erfc].E_MP2 | 465 |
//! | `e_lr_mp2` | SC[erf].E_MP2 | 466 |
//! | `e_dmp2_lr` | 2·SC[erf].E_OS | 461 |
//! | B `e_drpa_lr` | E_dRPA[erf] | 408–413 |
//! | B `e_corr_naive` | SC[erfc].E_MP2 + E_dRPA[erf] | 414 |
//! | B `e_corr` | SC[C].E_MP2 + E_dRPA[erf] − 2·SC[erf].E_OS | 404, 410 |
//! | T `e_delta_drpa_full` | E_dRPA[C] − 2·SC[C].E_OS | 442 |
//! | T `e_delta_drpa_sr` | E_dRPA[erfc] − 2·SC[erfc].E_OS | 443 |
//! | T `e_corr` | SC[C].E_MP2 + Δfull − Δsr | 445 |
//! | `total_energy` | E_RHF + e_corr | 473 |
//!
//! The generator reproduces each piece (numpy RI-MP2 spin decomposition and
//! numpy dRPA from the same B), anchors the Coulomb pieces to PySCF `DFMP2`
//! (E_OS/E_SS/E_MP2, ~3e-16 Ha) and `gw.rpa.RPA` (~2e-14 Ha), and pins the
//! "2" in 2·E_OS by an independent route: the second-order term of its dRPA
//! integrated over frequency (−½ tr Π², converged 160-point grid) equals
//! 2·E_OS to ≤ 3.2e-15 relative for every kernel.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric is right: every piece is already validated in isolation
//!   (attenuated RI-MP2 and attenuated RPA rows); here the assembly is new, so
//!   each field agrees at the ~1e-10 Ha level of those rows.
//! * A coefficient slip in the ring subtraction (1·E_OS for 2·E_OS): the total
//!   lands on the `dmp2_coeff_1` control (≥ 9.5e-5 Ha away for B, ≥ 3.7e-3
//!   for T) while `e_dmp2_lr` (recomputed at line 461) still passes.
//! * A sign slip: the total lands on `wrong_sign` (≥ 3.8e-4 Ha for B, ≥ 9.8e-2
//!   Ha for T).
//! * Formulations swapped: B and T differ by 2.2e-3…1.7e-2 Ha here.
//! * ω passed in Å⁻¹ where Bohr⁻¹ is expected: the default's 0.420 digits land
//!   on the ω = 0.42 Bohr⁻¹ reference (B moves ≥ 1.7e-4 Ha, T ≥ 1.1e-2 Ha).
//!
//! # Exactness anchors (asserted, both sides)
//!
//! The formulas share both endpoints (erf + erfc = Coulomb):
//! B: ω → 0 ⇒ E_MP2[C]; ω → ∞ ⇒ E_MP2[C] + E_dRPA[C] − 2·E_OS[C].
//! T: ω → 0 ⇒ the two Δ terms cancel ⇒ E_MP2[C]; ω → ∞ ⇒ the same
//! E_MP2[C] + ΔdRPA[C]. Anchors run at B ω = 1e-2 / 1e5 and T ω = 1e-5 / 1e2
//! Bohr⁻¹, where the numpy side's residual to the exact target is ≤ 2.5e-13,
//! ≤ 2.3e-11, ≤ 8.7e-12 and ≤ 6.5e-13 Ha. ferric must equal the numpy anchor
//! value AND the exact target; B at ω → ∞ must also equal ferric's OWN
//! `e_mp2_full + e_delta_drpa_full` from the T run (an internal cross-route).
//!
//! ANCHOR BLIND SPOT: at B ω → 0 every erf piece vanishes, so this anchor
//! cannot see any defect in the erf ring subtraction (the mutation below
//! passes it). The ω → ∞ anchors and the finite-ω comparisons carry that.
//!
//! # TOLERANCES
//!
//! Each bar is 3–25× the worst |d| measured over all systems and ω
//! (2026-09-25), recorded beside it.
//!
//! # NEGATIVE CONTROLS (asserted)
//!
//! * B vs T: ferric B must MISS ferric T and the numpy T reference (and vice
//!   versa) at every ω.
//! * Unit slip: ferric at ω = 0.42 (the default's Å⁻¹ digits read as Bohr⁻¹)
//!   must MISS the 0.222254 reference, both formulations (and it MATCHES the
//!   0.42 reference in the main loop, so the miss is the unit, not noise).
//! * Wrong sign and 1·E_OS ring coefficient: ferric must MISS both controls.
//!
//! # MUTATION (run 2026-09-25)
//!
//! `crates/ferric-rpa/src/rs_mp2_rpa.rs`, `DeltaLr` arm:
//! `let e_dmp2_lr = 2.0 * sc_lr.e_os;` → `let e_dmp2_lr = 1.0 * sc_lr.e_os;`
//! fails all three `*_vs_numpy` tests at the B `e_corr` check (shift =
//! E_OS[erf], 9.5e-5 Ha for H2O/cc-pVDZ at the default ω). The B ω → 0 anchor
//! cannot see it (blind spot above).
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_rpa::{rs_mp2_lr_rpa, RsMp2RpaConfig, RsMp2RpaFormulation, RsMp2RpaResult};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/rs_mp2_rpa";
const MOL_DIR: &str = "testdata/molecules/validation";
/// Frequency points on both sides (generator `NW`).
const N_QUAD: usize = 40;
/// Gauss–Legendre map scale; PySCF `x0`.
const U0: f64 = 0.5;
/// ω grid in Bohr⁻¹; `OMEGAS[I_DEFAULT]` is asserted equal to the production
/// default, `OMEGAS[I_UNIT_SLIP]` is its Å⁻¹ digits read as Bohr⁻¹.
const OMEGAS: [f64; 2] = [0.420 / 1.8897259886, 0.42];
const I_DEFAULT: usize = 0;
const I_UNIT_SLIP: usize = 1;
/// Limit anchors: (reference key, formulation, ω in Bohr⁻¹) — generator
/// `ANCHORS`.
const ANCHORS: [(&str, RsMp2RpaFormulation, f64); 4] = [
    ("B_omega_to_0", RsMp2RpaFormulation::DeltaLr, 1e-2),
    ("B_omega_to_inf", RsMp2RpaFormulation::DeltaLr, 1e5),
    ("T_omega_to_0", RsMp2RpaFormulation::CoupledRings, 1e-5),
    ("T_omega_to_inf", RsMp2RpaFormulation::CoupledRings, 1e2),
];

// Measured worst |d| (2026-09-25) beside each bar.
// MP2 spin-component fields (e_mp2_full, e_sr_mp2, e_lr_mp2, e_dmp2_lr): 4.2e-11.
const TOL_MP2: f64 = 5e-10;
// dRPA-bearing fields (e_drpa_lr, e_corr_naive, e_delta_drpa_full,
// e_delta_drpa_sr): 2.9e-11.
const TOL_DRPA: f64 = 5e-10;
// B and T e_corr and total_energy: 4.7e-11.
const TOL_E_CORR: f64 = 5e-10;
// ferric's ω → 0 / ω → ∞ anchors vs the exact targets: 4.3e-11.
const TOL_LIMIT: f64 = 5e-10;
// MP2 fields reported by both a B run and a T run: identical (0).
const TOL_B_T_SHARED: f64 = 1e-12;
// RHF vs PySCF: 4.8e-12.
const TOL_E_SCF: f64 = 5e-11;
const TOL_ENUC: f64 = 1e-9;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 10.0;

// ---------------------------------------------------------------------------
// Reference plumbing (same shape as validation_attenuated_rpa.rs)
// ---------------------------------------------------------------------------

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
        .expect("ferric-rpa manifest dir should be <root>/crates/ferric-rpa")
        .to_path_buf()
}

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_rs_mp2_rpa.py — a missing reference is a failure, never a skip",
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

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) -> f64 {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<44} ferric {got:+.12} ref {want:+.12} |d| {d:.2e}");
    assert!(
        d < tol,
        "{ctx}: {what}: ferric {got:.12} vs reference {want:.12} (|d| {d:.2e} Ha) exceeds {tol:.1e} Ha"
    );
    d
}

fn assert_misses(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!(
        "{ctx}: control {what}: |d| {d:.3e} Ha, must exceed {:.1e}",
        MUST_MISS_FACTOR * tol
    );
    assert!(
        d > MUST_MISS_FACTOR * tol,
        "{ctx}: negative control '{what}' did not miss: |d| {d:.3e} Ha <= {MUST_MISS_FACTOR} x \
         {tol:.1e} — the comparison does not respond to what the control changes"
    );
}

/// The reference's `omegas[k]` block for `omega` (matched to 1e-12).
fn omega_block<'a>(r: &'a Value, omega: f64, ctx: &str) -> &'a Value {
    r["omegas"]
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: reference has no omegas array"))
        .iter()
        .find(|b| (b["omega_bohr_inv"].as_f64().unwrap() - omega).abs() < 1e-12)
        .unwrap_or_else(|| panic!("{ctx}: reference has no block for omega {omega}"))
}

// ---------------------------------------------------------------------------
// System setup
// ---------------------------------------------------------------------------

struct Sys {
    label: String,
    r: Value,
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn load_system(system: &str, basis_name: &str) -> Sys {
    let r = reference(system, basis_name);
    let label = format!("{system}/{basis_name}");
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check_close(
        &label,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &label),
        TOL_ENUC,
    );
    assert_eq!(
        mol.nelec() as i64,
        r["nelectron"].as_i64().unwrap(),
        "{label}: electron count"
    );
    assert_eq!(r["frozen_core"].as_i64(), Some(0), "{label}: frozen core");
    let obs_bs = basis::bundled(basis_name).unwrap();
    let aux_name = r["aux"].as_str().expect("aux").to_string();
    let aux_bs = basis::bundled(&aux_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    assert_eq!(
        obs.nbasis() as i64,
        r["nao"].as_i64().unwrap(),
        "{label}: AO count differs from the reference's"
    );
    assert_eq!(
        dfbs.nbasis() as i64,
        r["naux"].as_i64().unwrap(),
        "{label}: aux count differs from the reference's"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig {
            max_iter: 400,
            energy_conv: 1e-11,
            density_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("{label}: RHF failed: {e:?}"));
    assert!(rhf.converged, "{label}: RHF did not converge");
    check_close(
        &label,
        "E_RHF",
        rhf.energy,
        num(&r, "/rhf/energy", &label),
        TOL_E_SCF,
    );
    Sys {
        label,
        r,
        mol,
        obs,
        dfbs,
        rhf,
    }
}

/// Full rank, Gauss–Legendre with PySCF's map (same as the attenuated-RPA row).
fn rpa_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        eigensolver: Eigensolver::Lanczos,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: N_QUAD,
            u0: U0,
        },
        ..Default::default()
    }
}

fn run(sys: &Sys, omega: f64, form: RsMp2RpaFormulation) -> RsMp2RpaResult {
    let cfg = RsMp2RpaConfig {
        omega,
        formulation: form,
        frozen_core: 0,
        drpa: rpa_cfg(),
        ..Default::default()
    };
    let r = rs_mp2_lr_rpa(&sys.mol, &sys.obs, &sys.dfbs, &sys.rhf, &cfg).unwrap_or_else(|e| {
        panic!(
            "{}: rs_mp2_lr_rpa(ω={omega}, {form:?}) failed: {e:?}",
            sys.label
        )
    });
    assert!(
        (r.total_energy - (sys.rhf.energy + r.e_corr)).abs() < 1e-12,
        "{}: total_energy != E_RHF + e_corr",
        sys.label
    );
    r
}

fn form_key(form: RsMp2RpaFormulation) -> &'static str {
    match form {
        RsMp2RpaFormulation::DeltaLr => "B",
        RsMp2RpaFormulation::CoupledRings => "T",
    }
}

/// Every field of one run vs the `omegas[k]` block `b`, plus the sign and
/// coefficient controls. Field order = failure localisation: the first
/// failing line names the piece.
fn check_fields(ctx: &str, sys: &Sys, res: &RsMp2RpaResult, b: &Value, form: RsMp2RpaFormulation) {
    let f = form_key(form);
    for (name, got, tol) in [
        ("e_mp2_full", res.e_mp2_full, TOL_MP2),
        ("e_sr_mp2", res.e_sr_mp2, TOL_MP2),
        ("e_lr_mp2", res.e_lr_mp2, TOL_MP2),
        ("e_dmp2_lr", res.e_dmp2_lr, TOL_MP2),
    ] {
        check_close(
            ctx,
            &format!("{f} {name}"),
            got,
            num(b, &format!("/{name}"), ctx),
            tol,
        );
    }
    let opt = |v: Option<f64>, name: &str| {
        v.unwrap_or_else(|| panic!("{ctx}: {f} run must report {name}"))
    };
    match form {
        RsMp2RpaFormulation::DeltaLr => {
            assert!(res.e_delta_drpa_full.is_none() && res.e_delta_drpa_sr.is_none());
            for (name, got) in [
                ("e_drpa_lr", opt(res.e_drpa_lr, "e_drpa_lr")),
                ("e_corr_naive", opt(res.e_corr_naive, "e_corr_naive")),
            ] {
                check_close(
                    ctx,
                    &format!("B {name}"),
                    got,
                    num(b, &format!("/B/{name}"), ctx),
                    TOL_DRPA,
                );
            }
        }
        RsMp2RpaFormulation::CoupledRings => {
            assert!(res.e_drpa_lr.is_none() && res.e_corr_naive.is_none());
            for (name, got) in [
                (
                    "e_delta_drpa_full",
                    opt(res.e_delta_drpa_full, "e_delta_drpa_full"),
                ),
                (
                    "e_delta_drpa_sr",
                    opt(res.e_delta_drpa_sr, "e_delta_drpa_sr"),
                ),
            ] {
                check_close(
                    ctx,
                    &format!("T {name}"),
                    got,
                    num(b, &format!("/T/{name}"), ctx),
                    TOL_DRPA,
                );
            }
        }
    }
    check_close(
        ctx,
        &format!("{f} e_corr"),
        res.e_corr,
        num(b, &format!("/{f}/e_corr"), ctx),
        TOL_E_CORR,
    );
    check_close(
        ctx,
        &format!("{f} total_energy"),
        res.total_energy,
        num(&sys.r, "/rhf/energy", ctx) + num(b, &format!("/{f}/e_corr"), ctx),
        TOL_E_CORR + TOL_E_SCF,
    );
    for control in ["wrong_sign", "dmp2_coeff_1"] {
        assert_misses(
            ctx,
            &format!("{f} e_corr vs {f}_{control}"),
            res.e_corr,
            num(b, &format!("/controls/{f}_{control}"), ctx),
            TOL_E_CORR,
        );
    }
}

/// The whole row for one system.
fn run_case(system: &str, basis_name: &str) {
    assert!(
        (OMEGAS[I_DEFAULT] - RsMp2RpaConfig::default().omega).abs() < 1e-15,
        "OMEGAS[{I_DEFAULT}] must be the production default RsMp2RpaConfig::omega"
    );
    let sys = load_system(system, basis_name);
    let ctx = sys.label.clone();
    let r = &sys.r;

    // Finite ω: every field, both formulations, plus B-vs-T controls.
    let mut at_slip: Vec<(RsMp2RpaFormulation, f64)> = Vec::new();
    let mut coulomb_limit_ferric = None;
    for (i, &w) in OMEGAS.iter().enumerate() {
        let b = omega_block(r, w, &ctx);
        let here = format!("{ctx} ω={w:.6}");
        let rb = run(&sys, w, RsMp2RpaFormulation::DeltaLr);
        let rt = run(&sys, w, RsMp2RpaFormulation::CoupledRings);
        check_fields(&here, &sys, &rb, b, RsMp2RpaFormulation::DeltaLr);
        check_fields(&here, &sys, &rt, b, RsMp2RpaFormulation::CoupledRings);
        for (name, x, y) in [
            ("e_mp2_full", rb.e_mp2_full, rt.e_mp2_full),
            ("e_sr_mp2", rb.e_sr_mp2, rt.e_sr_mp2),
            ("e_lr_mp2", rb.e_lr_mp2, rt.e_lr_mp2),
            ("e_dmp2_lr", rb.e_dmp2_lr, rt.e_dmp2_lr),
        ] {
            check_close(
                &here,
                &format!("B vs T shared {name}"),
                x,
                y,
                TOL_B_T_SHARED,
            );
        }
        assert_misses(
            &here,
            "B e_corr vs ferric T e_corr",
            rb.e_corr,
            rt.e_corr,
            TOL_E_CORR,
        );
        assert_misses(
            &here,
            "B e_corr vs the T reference",
            rb.e_corr,
            num(b, "/T/e_corr", &ctx),
            TOL_E_CORR,
        );
        assert_misses(
            &here,
            "T e_corr vs the B reference",
            rt.e_corr,
            num(b, "/B/e_corr", &ctx),
            TOL_E_CORR,
        );
        if i == I_DEFAULT {
            coulomb_limit_ferric = Some(rt.e_mp2_full + rt.e_delta_drpa_full.unwrap());
        }
        if i == I_UNIT_SLIP {
            at_slip.push((RsMp2RpaFormulation::DeltaLr, rb.e_corr));
            at_slip.push((RsMp2RpaFormulation::CoupledRings, rt.e_corr));
        }
    }

    // Unit slip: ω = 0.42 (the default's Å⁻¹ digits read as Bohr⁻¹) already
    // MATCHED the 0.42 reference above; it must MISS the true default's.
    let b_def = omega_block(r, OMEGAS[I_DEFAULT], &ctx);
    for (form, e) in at_slip {
        let f = form_key(form);
        assert_misses(
            &ctx,
            &format!("unit slip: {f}(0.42) vs the {f}(0.222254) reference"),
            e,
            num(b_def, &format!("/{f}/e_corr"), &ctx),
            TOL_E_CORR,
        );
    }

    // Exactness anchors.
    for (name, form, w) in ANCHORS {
        let a = &r["anchors"][name];
        assert_eq!(
            num(a, "/omega", &ctx),
            w,
            "{ctx}: anchor {name} ω differs from the generator's"
        );
        let res = run(&sys, w, form);
        check_close(
            &ctx,
            &format!("{name}(ω={w:e}) e_corr vs numpy"),
            res.e_corr,
            num(a, "/e_corr", &ctx),
            TOL_E_CORR,
        );
        check_close(
            &ctx,
            &format!("LIMIT {name}(ω={w:e}) vs exact target"),
            res.e_corr,
            num(a, "/target", &ctx),
            TOL_LIMIT,
        );
        if name == "B_omega_to_inf" {
            let own = coulomb_limit_ferric.expect("T run at the default ω");
            check_close(
                &ctx,
                "LIMIT B(ω→∞) vs ferric T's e_mp2_full + e_delta_drpa_full",
                res.e_corr,
                own,
                TOL_LIMIT,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "validation: RS-MP2-RPA"]
fn rs_mp2_rpa_h2o_cc_pvdz_vs_numpy() {
    run_case("h2o", "cc-pvdz");
}

#[test]
#[ignore = "validation: RS-MP2-RPA"]
fn rs_mp2_rpa_nh3_cc_pvdz_vs_numpy() {
    run_case("nh3", "cc-pvdz");
}

#[test]
#[ignore = "validation: RS-MP2-RPA"]
fn rs_mp2_rpa_h2o_aug_cc_pvdz_vs_numpy() {
    run_case("h2o", "aug-cc-pvdz");
}
