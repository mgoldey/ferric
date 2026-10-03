//! VALIDATION tier — VALIDATION.md row "SCF ladder".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_scf_ladder \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is validated
//!
//! ferric's DEFAULT convergence ladder (`ferric_scf::ladder::default_ladder`,
//! walked by `solve_rhf_ladder`: MINAO+DIIS → +ADIIS → +level shift 0.5 →
//! +level shift 1.0 & SOSCF → +Fermi smearing), with the opt-in RHF state
//! selection turned on (`check_stability` + `scf_stability_descent`, both
//! default OFF — see "STATE SELECTION" below), must land on the same,
//! internally STABLE RHF minimum that PySCF reaches with a second-order solver
//! (`scf.RHF(mol).newton()`) followed by a `stability()` loop over four initial
//! guesses (`scripts/validation/gen_scf_ladder.py`). For each system:
//!
//! (a) the ladder converges, on a rung that does NOT smear (a smeared rung
//!     returns fractional occupations, which is not an RHF energy);
//! (b) its energy equals PySCF's lowest stable RHF minimum;
//! (c) ferric's own RHF-internal stability check (`check_stability`,
//!     `stability::rhf_internal_stability`) reports the state STABLE — or
//!     MARGINAL where the reference minimum carries an exact zero mode — with
//!     λ_min matching the reference spectrum.
//!
//! The RHF→UHF (external) instability is INFORMATION in this row; it is
//! measured with ferric's UHF Hessian at the RHF point (see
//! validation_scf_stability.rs for why that is the external operator) and
//! compared with PySCF's `hop_rhf2uhf`, but the row is about the RHF minimum.
//!
//! # Systems (all-electron def2-SVP, exact 4-index J/K, no fitting)
//!
//! | system | AOs | reference state |
//! |---|---:|---|
//! | N2, r = 1.60 Å | 28 | stable RHF min −108.5016165203; ONE exact zero mode (+1e-8), then +0.1109. RHF→UHF unstable (triplet −0.148). Plain DIIS lands on a SADDLE 19.06 mHa higher (singlet −0.0654, doubly degenerate). |
//! | CuCN (Cu–C≡N) | 59 | −1730.9327350219, STABLE (singlet +0.1699), externally stable (triplet +0.0269). All four guesses and plain DIIS agree. |
//! | Cr(CO)6, Oh | 199 | −1718.9155652441, STABLE (singlet +0.0771620, triply degenerate). RHF→UHF UNSTABLE (triplet −0.0463; broken-symmetry UHF 11.41 mHa lower, ⟨S²⟩ 0.915). All four guesses and plain DIIS agree. |
//!
//! Every reference spectrum is the FULL dense eigvalsh of the Hessian built
//! explicitly from MO integrals (dim 7830 for Cr(CO)6), cross-checked against
//! PySCF's own matvecs on random vectors (max deviation 3.4e-11). The first
//! Cr(CO)6 reference used a PySCF Davidson seeded on the 8 smallest diagonal
//! entries; it stayed out of the triply degenerate block and reported singlet
//! +0.18006 and triplet +0.1037. ferric's block Davidson found +0.0771620179
//! (residual 6.5e-10), which the dense spectrum confirms to 2e-10.
//!
//! Basis: N2 uses ferric's bundled def2-SVP. ferric's bundled `def2-svp.json`
//! has NO Z = 21–34, so CuCN and Cr(CO)6 load
//! `testdata/reference/validation/scf_ladder/basis/def2-svp_bse_c-n-o-cr-cu.json`
//! (a C/N/O/Cr/Cu subset of the BSE def2-SVP JSON, C/N/O byte-identical to the
//! bundled file) through `basis::load_bse_json`; PySCF is fed the same file.
//! Cu and Cr are all-electron in def2-SVP (def2 ECPs start at Rb); the file
//! carries no ECP and the generator asserts that.
//!
//! # Physics hypothesis vs artifact hypothesis (Experimental Protocol)
//!
//! * If the ladder works: (a)–(c) hold on all three systems, the energy at the
//!   SCF-convergence floor (quadratic in the density error) and λ_min at the
//!   first-order floor.
//! * If the ladder lands on the wrong STATE (a saddle or a higher minimum —
//!   the failure this row exists to catch): the energy misses by mHa and
//!   ferric's own verdict is UNSTABLE. N2 is the system where this is
//!   EXPECTED to be discriminating: plain DIIS (both codes) converges to a
//!   saddle 19.06 mHa above the minimum.
//! * If the HARNESS is broken (geometry, basis file, unit constant): nuclear
//!   repulsion and the AO count fail FIRST, so a harness error cannot pose as
//!   a state error.
//! * If the comparison is VACUOUS (the stable minimum is what every SCF finds
//!   anyway): the non-vacuity control below would fail to separate two states.
//!   It does separate them on N2. On CuCN and Cr(CO)6 it CANNOT — all four
//!   PySCF guesses and plain DIIS reach the same state — so their (b)/(c) are
//!   a regression anchor on heavy-atom systems, not a state-selection test.
//!
//! # NON-VACUITY CONTROL
//!
//! `n2_plain_diis_rung_alone_lands_on_the_saddle`: rung 0 of the default
//! ladder ALONE (MINAO + plain DIIS) on N2 must converge to PySCF's plain-DIIS
//! state, ≥ 10 mHa above the stable minimum, with ferric's verdict UNSTABLE and
//! λ_min matching PySCF's negative singlet eigenvalue. That proves the
//! (b)/(c) assertions in the main N2 test can fail and that ferric's check
//! can tell the two states apart.
//!
//! # STATE SELECTION
//!
//! MEASURED (this file, before `solve_rhf` had a descent): the default ladder
//! on N2 converged on rung 0 to −108.4825600107, the plain-DIIS saddle 19.06
//! mHa above the minimum, with ferric's own verdict UNSTABLE (λ_min −0.0654).
//! The ladder escalates only on NON-convergence, so a converged saddle is
//! accepted. The fix is `solve_rhf`'s opt-in internal stability descent
//! (`scf_stability_descent`, the knob `solve_uhf` already honoured; see
//! `rhf_stability_descent` in src/rhf.rs and tests/rhf_stability_descent.rs).
//! It runs inside whichever rung converges. It stays OFF in `default_ladder`
//! because it costs a Davidson eigensolve on every converged solve (64
//! iterations on Cr(CO)6, more J/K builds than its 35-iteration SCF), so the
//! row tests the default ladder WITH the two opt-in knobs; whether to flip
//! them on in `default_ladder_from` is a separate decision.
//!
//! With the descent on (2026-10-01, release, 53 min for the four tests) the
//! default ladder reaches PySCF's stable minimum on all three systems.
//!
//! # TOLERANCES (measured 2026-10-01)
//!
//! | quantity | measured max | bar |
//! |---|---:|---:|
//! | energy vs PySCF stable minimum | 8.8e-10 Ha (Cr(CO)6) | 1e-8 |
//! | stability eigenvalues (Davidson, dense, external) | 1.3e-9 Ha (CuCN) | 1e-8 |
//!
//! The saddle the descent must leave sits 1.906e-2 Ha above the minimum, so the
//! energy bar is 2e6 below the defect it catches.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::engine_pool::EnginePool;
use ferric_scf::ladder::{
    default_ladder, default_ladder_from, solve_rhf_ladder, LadderResult, Rung,
};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rhf_newton::RhfNewtonInputs;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::{
    uhf_internal_stability, StabilityConfig, StabilityKind, StabilityResult, StabilityVerdict,
};
use ferric_scf::uhf_newton::UhfNewtonInputs;
use ferric_scf::ScfResult;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/scf_ladder";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS_LABEL: &str = "def2-svp";
const BUNDLED_PREFIX: &str = "crates/ferric-core/src/basis/bundled/";

/// Geometry / unit-constant check (same bar as the other validation rows).
const TOL_ENUC: f64 = 1e-9;
/// Ladder energy vs PySCF's stable RHF minimum. Measured ≤ 8.8e-10 Ha.
const TOL_E: f64 = 1e-8;
/// Stability eigenvalues (ferric Davidson λ_min, ferric dense spectrum,
/// external) vs PySCF's dense Hessians. Measured ≤ 1.3e-9 Ha. The references are converged to conv_tol_grad 1e-7
/// (the AH solver stalls near |g| 2e-7 on N2), and eigenvalues are FIRST order
/// in the orbital error, so this is looser than the stability row's 1e-8.
const TOL_LAMBDA: f64 = 1e-8;
/// The non-vacuity control requires the saddle to sit at least this far above
/// the stable minimum (PySCF measured 1.9057e-2 Ha on N2).
const MIN_STATE_SEPARATION: f64 = 1e-2;
/// Number of lowest dense-Hessian eigenvalues compared where a dense spectrum
/// is built (N2, CuCN).
const N_COMPARE: usize = 4;
/// Convergence target for every rung. Tighter than RhfConfig::default()'s
/// 1e-6 because (c) compares FIRST-order quantities; only this,
/// `check_stability` and `scf_stability_descent` differ from the production
/// default ladder (asserted in `ladder_under_test`).
const DENSITY_CONV: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Harness
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
        .expect("ferric-scf manifest dir should be <root>/crates/ferric-scf")
        .to_path_buf()
}

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS_LABEL}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_scf_ladder.py — a missing reference is a failure, never a skip",
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

fn nums(v: &Value, ptr: &str, ctx: &str) -> Vec<f64> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn boolean(v: &Value, ptr: &str, ctx: &str) -> bool {
    v.pointer(ptr)
        .and_then(Value::as_bool)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a bool"))
}

struct System {
    name: String,
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

/// The basis the reference was generated with: ferric's bundled set when the
/// reference names a bundled file, else the row's own BSE JSON via
/// `load_bse_json` (the generator records the path in `basis_file`).
fn load_basis(r: &Value, ctx: &str) -> BasisSet {
    let file = r["basis_file"]
        .as_str()
        .unwrap_or_else(|| panic!("{ctx}: reference has no basis_file"));
    if file.starts_with(BUNDLED_PREFIX) {
        basis::bundled(BASIS_LABEL).unwrap()
    } else {
        let p = workspace_root().join(file);
        basis::load_bse_json(p.to_str().unwrap())
            .unwrap_or_else(|e| panic!("{ctx}: cannot load {}: {e:?}", p.display()))
    }
}

fn load_system(system: &str, r: &Value) -> System {
    let ctx = format!("{system}/{BASIS_LABEL}");
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let enuc_ref = num(r, "/nuclear_repulsion", &ctx);
    let enuc = mol.nuclear_repulsion();
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion {enuc:.12} vs reference {enuc_ref:.12} — geometry/unit mismatch"
    );
    assert_eq!(
        mol.nelec() as u64,
        r["nelectron"].as_u64().expect("nelectron"),
        "{ctx}: electron count differs (an ECP crept in?)"
    );
    let bs = load_basis(r, &ctx);
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    assert_eq!(
        prep.nbasis(),
        r["nao"].as_u64().expect("nao") as usize,
        "{ctx}: AO count differs from the reference's"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    System {
        name: ctx,
        mol,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<26} ferric {got:+.12e} ref {want:+.12e} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12e} vs reference {want:.12e} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn stability<'a>(res: &'a ScfResult, ctx: &str) -> &'a StabilityResult {
    let st = res
        .stability
        .as_ref()
        .unwrap_or_else(|| panic!("{ctx}: check_stability was set but no verdict came back"));
    eprintln!("{ctx}: {}", st.summary());
    assert!(
        st.converged,
        "{ctx}: Davidson not converged: {}",
        st.summary()
    );
    assert_eq!(st.kind, StabilityKind::RhfInternal);
    st
}

/// The production default ladder with `check_stability` on, the RHF state
/// selection (`scf_stability_descent`) on iff `descent`, and `density_conv`
/// tightened. Asserts that every convergence knob the ladder ESCALATES is
/// identical to `default_ladder()`'s, so this row tests the shipped ladder
/// shape and not a private variant.
fn ladder_under_test(descent: bool) -> Vec<Rung> {
    let base = RhfConfig {
        density_conv: DENSITY_CONV,
        check_stability: true,
        scf_stability_descent: descent,
        ..Default::default()
    };
    let ours = default_ladder_from(&base);
    let prod = default_ladder();
    assert_eq!(ours.len(), prod.len());
    for (i, (a, b)) in ours.iter().zip(prod.iter()).enumerate() {
        let (a, b) = (&a.config, &b.config);
        assert_eq!(a.max_iter, b.max_iter, "rung {i} max_iter");
        assert_eq!(a.level_shift, b.level_shift, "rung {i} level_shift");
        assert_eq!(a.diis_flavor, b.diis_flavor, "rung {i} diis_flavor");
        assert_eq!(
            a.newton_trigger, b.newton_trigger,
            "rung {i} newton_trigger"
        );
        assert_eq!(
            a.smearing_sigma, b.smearing_sigma,
            "rung {i} smearing_sigma"
        );
        assert_eq!(a.use_sad_guess, b.use_sad_guess, "rung {i} guess");
        assert_eq!(a.stall_window, b.stall_window, "rung {i} stall_window");
        assert_eq!(
            a.divergence_tol, b.divergence_tol,
            "rung {i} divergence_tol"
        );
        assert!(
            a.df_j_aux.is_none() && a.df_k_aux.is_none(),
            "rung {i}: exact J/K"
        );
    }
    ours
}

fn run_ladder(sys: &System, ladder: &[Rung]) -> LadderResult {
    let lr = solve_rhf_ladder(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        Operator::coulomb(),
        &sys.bounds,
        ladder,
    )
    .unwrap_or_else(|e| panic!("{}: solve_rhf_ladder failed: {e:?}", sys.name));
    for (i, o) in lr.rung_outcomes.iter().enumerate() {
        eprintln!(
            "{}: rung {i}: {:?} after {} iters, E = {:.10}",
            sys.name, o.exit, o.iters, o.final_energy
        );
    }
    eprintln!(
        "{}: ladder converged={} rung_reached={} E={:.10}",
        sys.name, lr.converged, lr.rung_reached, lr.result.energy
    );
    lr
}

fn sym_eigh(h: &Array2<f64>, ctx: &str) -> Vec<f64> {
    let asym = (h - &h.t()).iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    eprintln!(
        "{ctx}: dense Hessian dim {} max asymmetry {asym:.2e}",
        h.nrows()
    );
    assert!(
        asym < 1e-8,
        "{ctx}: orbital Hessian not symmetric ({asym:.2e})"
    );
    let (ev, _) = (0.5 * (h + &h.t())).eigh(UPLO::Lower).unwrap();
    ev.to_vec()
}

/// Dense spectrum of ferric's RHF (singlet) Hessian at a converged RHF result.
fn dense_rhf_spectrum(sys: &System, res: &ScfResult) -> Vec<f64> {
    let nocc = sys.mol.nelec() as usize / 2;
    let c = res.mos_alpha.clone();
    let f_mo = c.t().dot(&res.fock_alpha).dot(&c);
    let inp = RhfNewtonInputs {
        prep: &sys.prep,
        bounds: &sys.bounds,
        c: &c,
        f_mo: &f_mo,
        nocc,
        k_mix_sr: 1.0,
        fxc: None,
        thresh: RhfConfig::default().integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let n = c.nrows();
    let (no, nv) = (nocc, n - nocc);
    let pool = EnginePool::new(sys.bounds.op, &sys.prep, 1e-14).unwrap();
    let mut h = Array2::<f64>::zeros((nv * no, nv * no));
    for col in 0..nv * no {
        let mut v = vec![0.0; nv * no];
        v[col] = 1.0;
        let k = Array2::from_shape_vec((nv, no), v).unwrap();
        let hv = ferric_scf::rhf_newton::hessian_matvec(&sys.ctx, &inp, &k, &pool).unwrap();
        for (r, x) in hv.iter().enumerate() {
            h[(r, col)] = *x;
        }
    }
    sym_eigh(&h, &format!("{} RHF", sys.name))
}

/// INFORMATION: the RHF→UHF external check, by BOTH constructions — ferric's
/// UHF Hessian at the RHF point (λ_min = min(singlet, triplet)) and the
/// dedicated `rhf_external_stability` triplet operator (λ_min = triplet alone,
/// which is what `RhfConfig::check_stability` reports). See
/// validation_scf_stability.rs for the normalization and the channel split.
fn external_check(sys: &System, res: &ScfResult, r: &Value) {
    let ctx = sys.name.as_str();
    let nocc = sys.mol.nelec() as usize / 2;
    let c = res.mos_alpha.clone();
    let f_mo = c.t().dot(&res.fock_alpha).dot(&c);
    let inp = UhfNewtonInputs {
        prep: &sys.prep,
        bounds: &sys.bounds,
        c_a: &c,
        c_b: &c,
        f_a_mo: &f_mo,
        f_b_mo: &f_mo,
        nocc_a: nocc,
        nocc_b: nocc,
        k_mix_sr: 1.0,
        fxc: None,
        thresh: RhfConfig::default().integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let st = uhf_internal_stability(&sys.ctx, &inp, &StabilityConfig::default())
        .unwrap_or_else(|e| panic!("{ctx}: external analysis failed: {e:?}"));
    eprintln!(
        "{ctx}: external (UHF Hessian at the RHF point): {}",
        st.summary()
    );
    assert!(st.converged, "{ctx}: {}", st.summary());
    let sing = nums(r, "/rhf/singlet_rhf_internal/lowest", ctx)[0];
    let trip = nums(r, "/rhf/triplet_rhf_to_uhf/lowest", ctx)[0];
    check_close(
        ctx,
        "external lambda_min",
        st.lowest_eigenvalue,
        sing.min(trip),
        TOL_LAMBDA,
    );

    // The DEDICATED triplet operator, which is what `check_stability` runs:
    // it isolates the RHF→UHF channel instead of reporting min(singlet,
    // triplet), so it is compared against the TRIPLET reference alone.
    let rinp = RhfNewtonInputs {
        prep: &sys.prep,
        bounds: &sys.bounds,
        c: &c,
        f_mo: &f_mo,
        nocc,
        k_mix_sr: 1.0,
        fxc: None,
        thresh: RhfConfig::default().integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let st_trip = ferric_scf::stability::rhf_external_stability(
        &sys.ctx,
        &rinp,
        &StabilityConfig::default(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: dedicated triplet analysis failed: {e:?}"));
    eprintln!(
        "{ctx}: external (dedicated triplet operator): {}",
        st_trip.summary()
    );
    assert!(st_trip.converged, "{ctx}: {}", st_trip.summary());
    check_close(
        ctx,
        "triplet lambda_min",
        st_trip.lowest_eigenvalue,
        trip,
        TOL_LAMBDA,
    );
    let ext_stable = boolean(r, "/rhf/stability/external_stable", ctx);
    let ext_verdict = if ext_stable {
        StabilityVerdict::Stable
    } else {
        StabilityVerdict::Unstable
    };
    assert_eq!(
        st.verdict(),
        ext_verdict,
        "{ctx}: RHF->UHF verdict: {}",
        st.summary()
    );
    if let Some(bs) = r.get("broken_symmetry_uhf") {
        eprintln!(
            "{ctx}: (info) PySCF broken-symmetry UHF {:.10} <S^2> {:.4}, {:.3e} Ha below RHF",
            bs["energy"].as_f64().unwrap_or(f64::NAN),
            bs["s_squared"].as_f64().unwrap_or(f64::NAN),
            bs["below_rhf_by"].as_f64().unwrap_or(f64::NAN),
        );
    }
}

/// (a)+(b)+(c) for one system. `dense`: also compare ferric's dense singlet
/// spectrum (affordable for N2 and CuCN, not for Cr(CO)6).
fn ladder_row(system: &str, dense: bool) {
    let r = reference(system);
    let sys = load_system(system, &r);
    let ctx = sys.name.clone();
    let ctx = ctx.as_str();
    let ladder = ladder_under_test(true);
    let lr = run_ladder(&sys, &ladder);

    // (a) converged, on a rung that does not smear.
    assert!(
        lr.converged,
        "{ctx}: default ladder did not converge (best rung {})",
        lr.rung_reached
    );
    assert!(
        ladder[lr.rung_reached].config.smearing_sigma.is_none(),
        "{ctx}: converged only on the Fermi-smearing rung {} — fractional occupations, \
         not an RHF energy",
        lr.rung_reached
    );
    let res = &lr.result;

    // (b) the energy of PySCF's lowest stable RHF minimum. The reference also
    // lists every distinct stable minimum PySCF found, for diagnosis.
    let e_ref = num(&r, "/rhf/energy", ctx);
    let minima = nums(&r, "/stable_minima_found", ctx);
    if (res.energy - e_ref).abs() >= TOL_E {
        let near = minima.iter().map(|e| (e, (res.energy - e).abs())).fold(
            (f64::NAN, f64::INFINITY),
            |b, (e, d)| if d < b.1 { (*e, d) } else { b },
        );
        eprintln!(
            "{ctx}: ladder E {:.10} is {:+.3e} from the reference minimum; nearest PySCF \
             stable minimum {:.10} (|d| {:.2e}); PySCF plain-DIIS state {:.10}",
            res.energy,
            res.energy - e_ref,
            near.0,
            near.1,
            num(&r, "/plain_diis_pyscf/energy", ctx)
        );
    }
    check_close(ctx, "energy", res.energy, e_ref, TOL_E);

    // (c) ferric's own internal verdict.
    let st = stability(res, ctx);
    let want = nums(&r, "/rhf/singlet_rhf_internal/lowest", ctx);
    let zero_modes = r
        .pointer("/rhf/stability/internal_zero_modes")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("{ctx}: reference lacks internal_zero_modes"));
    let expected = if zero_modes > 0 {
        // An exact zero mode is not curvature: MARGINAL, never STABLE.
        StabilityVerdict::Marginal
    } else {
        StabilityVerdict::Stable
    };
    assert_eq!(st.verdict(), expected, "{ctx}: {}", st.summary());
    check_close(
        ctx,
        "Davidson lambda_min",
        st.lowest_eigenvalue,
        want[0],
        TOL_LAMBDA,
    );
    if dense {
        let spec = dense_rhf_spectrum(&sys, res);
        for k in 0..N_COMPARE.min(want.len()) {
            check_close(
                ctx,
                &format!("dense singlet[{k}]"),
                spec[k],
                want[k],
                TOL_LAMBDA,
            );
        }
    }

    // INFORMATION: RHF -> UHF.
    external_check(&sys, res, &r);
}

// ---------------------------------------------------------------------------
// The row
// ---------------------------------------------------------------------------

/// N2 at 1.60 Å: rung 0 converges to the plain-DIIS saddle; the RHF state
/// selection must carry it to the stable minimum (see module doc, "STATE
/// SELECTION").
#[test]
#[ignore = "validation: SCF ladder"]
fn n2_default_ladder_reaches_stable_rhf_minimum() {
    ladder_row("n2_r1.60", true);
}

/// CuCN (linear Cu–C≡N). Regression anchor: every PySCF route reaches the
/// same state, so this cannot show state SELECTION (see module doc).
#[test]
#[ignore = "validation: SCF ladder"]
fn cucn_default_ladder_reaches_stable_rhf_minimum() {
    ladder_row("cucn", true);
}

/// Cr(CO)6, Oh. 199 AOs: Davidson λ_min only, no dense spectrum. Same
/// regression-anchor caveat as CuCN.
#[test]
#[ignore = "validation: SCF ladder"]
fn cr_co6_default_ladder_reaches_stable_rhf_minimum() {
    ladder_row("cr_co6", false);
}

/// NON-VACUITY CONTROL: rung 0 of the default ladder ALONE (MINAO + plain
/// DIIS) on N2 reaches PySCF's plain-DIIS state — a saddle ≥ 10 mHa above the
/// stable minimum — and ferric's own check calls it UNSTABLE with PySCF's
/// negative eigenvalue. This is what makes (b)/(c) above discriminating.
#[test]
#[ignore = "validation: SCF ladder"]
fn n2_plain_diis_rung_alone_lands_on_the_saddle() {
    let system = "n2_r1.60";
    let r = reference(system);
    let sys = load_system(system, &r);
    let ctx = format!("{} rung 0 alone", sys.name);
    let ctx = ctx.as_str();
    // Descent OFF: this control is about what plain DIIS converges to.
    let rung0 = vec![ladder_under_test(false)[0].clone()];
    let lr = run_ladder(&sys, &rung0);
    assert!(lr.converged, "{ctx}: plain DIIS rung did not converge");

    assert!(
        !boolean(&r, "/plain_diis_pyscf/same_state_as_reference", ctx),
        "{ctx}: the reference says plain DIIS reaches the stable minimum — this control is vacuous"
    );
    let e_saddle = num(&r, "/plain_diis_pyscf/energy", ctx);
    let e_min = num(&r, "/rhf/energy", ctx);
    assert!(
        e_saddle - e_min > MIN_STATE_SEPARATION,
        "{ctx}: reference states only {:.3e} Ha apart",
        e_saddle - e_min
    );
    check_close(ctx, "energy (saddle)", lr.result.energy, e_saddle, TOL_E);

    let st = stability(&lr.result, ctx);
    assert_eq!(
        st.verdict(),
        StabilityVerdict::Unstable,
        "{ctx}: ferric must flag the plain-DIIS state: {}",
        st.summary()
    );
    let want = nums(&r, "/plain_diis_pyscf/singlet_rhf_internal/lowest", ctx);
    assert!(
        want[0] < 0.0,
        "{ctx}: reference saddle must have a negative eigenvalue"
    );
    check_close(
        ctx,
        "Davidson lambda_min",
        st.lowest_eigenvalue,
        want[0],
        TOL_LAMBDA,
    );
    let spec = dense_rhf_spectrum(&sys, &lr.result);
    for k in 0..N_COMPARE.min(want.len()) {
        check_close(
            ctx,
            &format!("dense singlet[{k}]"),
            spec[k],
            want[k],
            TOL_LAMBDA,
        );
    }
}
