//! VALIDATION tier — validation.md row "Attenuated PDEP-RPA" (closed-shell
//! dRPA correlation energy with an erf / erfc range-separated kernel).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-rpa --release \
//!     --test validation_attenuated_rpa --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric `run_pdep_rpa(.., Operator::erf(ω) | Operator::erfc(ω), ..)` (full
//! rank, 40-point Gauss–Legendre, u0 = 0.5, no frozen core) against an
//! independent numpy dRPA on PySCF 2.13.1 density-fitting integrals, both on
//! the same full-Coulomb RHF state:
//!
//! | systems / basis (aux) | ω (Bohr⁻¹) | kernels |
//! |---|---|---|
//! | H2O, NH3 / cc-pVDZ (cc-pvdz-ri); H2O / aug-cc-pVDZ (aug-cc-pvdz-rifit) | 0.2, 0.222254 (= 0.420 Å⁻¹, the RS-MP2+RPA default), 0.42, 1.0 | erf and erfc |
//!
//! References: `scripts/validation/gen_attenuated_rpa.py` →
//! `testdata/reference/validation/attenuated_rpa/<system>_<basis>.json`.
//!
//! # How `op` reaches the energy (read from the code)
//!
//! `run_pdep_rpa` → `ferric_mp2::rimp2::compute_rpa_intermediates(.., op, ..)`,
//! which threads the one operator into all three RI pieces:
//! the metric `threeindex::coulomb_metric_2c(op, dfbs)`, its factor
//! `metric_inverse_sqrt(&v2c, op)` (regularized symmetric eigh, drop
//! eigenvalues < 1e-10, for erf; Cholesky for erfc) and the 3-centre source
//! `ThreeIndexSource::build(op, ..)`. Past that point the energy depends only on
//! `inter.b_ov` (see `run_pdep_rpa_from_intermediates`'s doc). The generator
//! reproduces exactly this: PySCF `int3c2e` and `int2c2e` both under
//! `Mole.with_range_coulomb` (positive ω = erf, negative = erfc — proved by
//! the erf + erfc = Coulomb identity it checks on both blocks), the same
//! eigh-with-1e-10-drop / Cholesky factor, and its numpy RPA equals PySCF's
//! `gw.rpa.RPA` to ~1e-14 Ha at the Coulomb kernel.
//!
//! Which attenuated energies are production: `rs_mp2_lr_rpa`'s default
//! `DeltaLr` formulation uses the erf dRPA, `CoupledRings` the erfc dRPA, both
//! through `run_pdep_rpa_from_intermediates` on `compute_rpa_intermediates`
//! built with that operator — the same object tested here.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric is right: grid, rank, aux, RHF and the metric factor are all
//!   matched, so only the integral engines (libint2 vs libcint) differ, and E_c
//!   agrees at the level the Coulomb row does (~1e-10 Ha).
//! * If `op` does not reach the metric (the most likely wiring slip): ferric
//!   returns the `mixed_coulomb_metric` value, 1e-4…4e-2 Ha away.
//! * If erf and erfc are swapped: the value lands on the other kernel's
//!   reference (≥ 6.1e-3 Ha away; the closest pair is NH3 at ω = 1.0).
//! * If ω is passed in Å⁻¹ where Bohr⁻¹ is expected: the default 0.420
//!   lands on the ω = 0.42 Bohr⁻¹ reference, not the 0.222254 one (≥ 3.3e-3 Ha).
//! * erf's regularized metric is ill-conditioned: the generator records how
//!   far E_c(erf) moves if the lindep cut is 1e-12 or 1e-8 instead of 1e-10
//!   (≤ 6e-13 Ha at ω ≤ 0.2223, up to 1.4e-8 Ha at ω = 1.0) and how far the
//!   nearest metric eigenvalue sits from the cut (≥ 0.028 decade, H2O/aug at
//!   the default ω: smallest kept eigenvalue 1.07e-10, ~500x integral noise).
//!   That is the part of an erf disagreement the regularization could own.
//!
//! # Exactness anchors (asserted)
//!
//! erfc(ω → 0) and erf(ω → ∞) are the Coulomb kernel. ferric at erfc(1e-5)
//! and erf(1e5) must equal the Coulomb reference E_c (numpy's own limit
//! residuals: erfc(1e-5) 3.7e-12…1.3e-11, erf(1e5) 4.3e-11…5.9e-11 Ha). erf(ω)
//! and erfc(ω) are NOT additive in RPA (nonlinear in the kernel: E(erf) +
//! E(erfc) − E(Coulomb) is 3.8e-3…8.8e-2 Ha here), so no additivity is
//! asserted.
//!
//! # TOLERANCES
//!
//! Each bar is 3–25× the worst |d| measured over all systems and ω
//! (2026-09-25), recorded beside it.
//!
//! # NEGATIVE CONTROLS (asserted)
//!
//! * Metric operator: ferric must MISS `mixed_coulomb_metric` (attenuated
//!   3-centre, Coulomb metric) at every ω and kernel. The control value is
//!   also REPRODUCED by driving ferric that way (`..._mixed_metric_control`,
//!   public `ThreeIndexSource` + `stream_dressed_mo_band` with the Coulomb
//!   metric), so the missed number is what the wiring slip would really give.
//! * erf/erfc swap: ferric erf(ω) must MISS the erfc(ω) reference and vice
//!   versa.
//! * Unit slip: ferric at ω = 0.42 (the default's Å⁻¹ digits read as Bohr⁻¹)
//!   must MISS the 0.222254 reference, for both kernels (and it MATCHES the
//!   0.42 reference in the main loop, so the miss is the unit, not noise).
//!
//! # MUTATION (run 2026-09-25)
//!
//! `crates/ferric-mp2/src/rimp2.rs`, `compute_rpa_intermediates`:
//! `let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;` →
//! `let v2c = threeindex::coulomb_metric_2c(Operator::coulomb(), dfbs)?;`
//! (operator not threaded into the metric) fails all three `*_vs_numpy` tests;
//! the mixed-metric control test, which builds its own intermediates, passes.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex;
use ferric_mp2::rimp2::{
    eri3_budget_bytes, metric_inverse_sqrt, stream_dressed_mo_band, RpaIntermediates,
};
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, run_pdep_rpa_from_intermediates, PdepRpaResult, RsMp2RpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/attenuated_rpa";
const MOL_DIR: &str = "testdata/molecules/validation";
/// Frequency points on both sides (generator `NW`, shared with the U-RPA row).
const N_QUAD: usize = 40;
/// Gauss–Legendre map scale; PySCF `x0`.
const U0: f64 = 0.5;
/// ω grid in Bohr⁻¹; the second entry is asserted equal to the production
/// default `RsMp2RpaConfig::default().omega`.
const OMEGAS: [f64; 4] = [0.2, 0.420 / 1.8897259886, 0.42, 1.0];
/// Index of the production default in `OMEGAS`, and of its Å⁻¹-digits slip.
const I_DEFAULT: usize = 1;
const I_UNIT_SLIP: usize = 2;
/// Limit anchors (generator `ERFC_ANCHOR` / `ERF_ANCHOR`).
const ERFC_ANCHOR: f64 = 1e-5;
const ERF_ANCHOR: f64 = 1e5;

// erfc (Cholesky metric): 3.5e-11.
const TOL_E_C_ERFC: f64 = 5e-10;
// erf (regularized, ill-conditioned metric): 5.8e-11. The generator measured
// up to 1.4e-8 Ha sensitivity to the metric eigenvalue cut at ω = 1.0, so the
// bar keeps extra margin over this run's value.
const TOL_E_C_ERF: f64 = 1e-9;
// Coulomb baseline: 4.5e-11.
const TOL_E_C_COULOMB: f64 = 5e-10;
// ferric at the ω extremes vs the COULOMB reference: 4.9e-11.
const TOL_LIMIT: f64 = 5e-10;
// RHF vs PySCF: 4.9e-12.
const TOL_E_SCF: f64 = 5e-11;
const TOL_ENUC: f64 = 1e-9;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 10.0;

// ---------------------------------------------------------------------------
// Reference plumbing (same shape as validation_urpa.rs)
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
             scripts/validation/gen_attenuated_rpa.py — a missing reference is a failure, never a skip",
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
    eprintln!("{ctx}: {what:<40} ferric {got:+.12} ref {want:+.12} |d| {d:.2e}");
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

/// The reference's `attenuated[k]` block for `omega` (matched to 1e-12).
fn att_block<'a>(r: &'a Value, omega: f64, ctx: &str) -> &'a Value {
    r["attenuated"]
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: reference has no attenuated array"))
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

/// Full rank (every dielectric mode kept), Gauss–Legendre with PySCF's map.
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

fn rpa(sys: &Sys, op: Operator) -> PdepRpaResult {
    let res = run_pdep_rpa(&sys.mol, &sys.obs, &sys.dfbs, op, &sys.rhf, &rpa_cfg())
        .unwrap_or_else(|e| panic!("{}: run_pdep_rpa({op:?}) failed: {e:?}", sys.label));
    assert_eq!(
        res.quad_freqs.len(),
        N_QUAD,
        "{}: quadrature size",
        sys.label
    );
    eprintln!(
        "{}: {op:?}: {} of {} dielectric modes kept",
        sys.label,
        res.n_eigenpotentials,
        sys.dfbs.nbasis()
    );
    res
}

#[derive(Clone, Copy)]
enum Kind {
    Erf,
    Erfc,
}

impl Kind {
    fn op(self, omega: f64) -> Operator {
        match self {
            Kind::Erf => Operator::erf(omega),
            Kind::Erfc => Operator::erfc(omega),
        }
    }
    fn key(self) -> &'static str {
        match self {
            Kind::Erf => "erf",
            Kind::Erfc => "erfc",
        }
    }
    fn other(self) -> Self {
        match self {
            Kind::Erf => Kind::Erfc,
            Kind::Erfc => Kind::Erf,
        }
    }
    fn tol(self) -> f64 {
        match self {
            Kind::Erf => TOL_E_C_ERF,
            Kind::Erfc => TOL_E_C_ERFC,
        }
    }
}

/// The whole row for one system: Coulomb baseline, both limit anchors, every
/// (ω, kernel) vs numpy with the metric-operator and erf/erfc-swap controls,
/// and the unit-slip control.
fn run_case(system: &str, basis_name: &str) {
    assert!(
        (OMEGAS[I_DEFAULT] - RsMp2RpaConfig::default().omega).abs() < 1e-15,
        "OMEGAS[{I_DEFAULT}] must be the production default RsMp2RpaConfig::omega"
    );
    let sys = load_system(system, basis_name);
    let ctx = sys.label.clone();
    let r = &sys.r;

    // Coulomb baseline (the object the U-RPA row already validates).
    let e_coul_ref = num(r, "/coulomb/e_corr", &ctx);
    let coul = rpa(&sys, Operator::coulomb());
    assert_eq!(
        coul.n_eigenpotentials,
        sys.dfbs.nbasis(),
        "{ctx}: Coulomb trunc_thresh = 0 must keep every mode"
    );
    check_close(&ctx, "E_c Coulomb", coul.e_rpa, e_coul_ref, TOL_E_C_COULOMB);

    // Exactness anchors: the ω extremes are the Coulomb kernel.
    assert_eq!(num(r, "/anchors/erfc_omega", &ctx), ERFC_ANCHOR);
    assert_eq!(num(r, "/anchors/erf_omega", &ctx), ERF_ANCHOR);
    for (kind, w, ptr) in [
        (Kind::Erfc, ERFC_ANCHOR, "/anchors/erfc_e_corr"),
        (Kind::Erf, ERF_ANCHOR, "/anchors/erf_e_corr"),
    ] {
        let a = rpa(&sys, kind.op(w));
        check_close(
            &ctx,
            &format!("E_c {}({w:e}) vs numpy", kind.key()),
            a.e_rpa,
            num(r, ptr, &ctx),
            kind.tol(),
        );
        check_close(
            &ctx,
            &format!("LIMIT {}({w:e}) vs Coulomb ref", kind.key()),
            a.e_rpa,
            e_coul_ref,
            TOL_LIMIT,
        );
    }

    // Every (ω, kernel) vs the numpy reference, plus the controls.
    let mut at_slip = [0.0_f64; 2];
    for (i, &w) in OMEGAS.iter().enumerate() {
        let b = att_block(r, w, &ctx);
        for (k, kind) in [Kind::Erf, Kind::Erfc].into_iter().enumerate() {
            let res = rpa(&sys, kind.op(w));
            let key = kind.key();
            let here = format!("{key}({w:.6})");
            check_close(
                &ctx,
                &format!("E_c {here}"),
                res.e_rpa,
                num(b, &format!("/{key}/e_corr"), &ctx),
                kind.tol(),
            );
            assert_misses(
                &ctx,
                &format!("{here} vs Coulomb-metric/attenuated-3c"),
                res.e_rpa,
                num(b, &format!("/{key}/mixed_coulomb_metric"), &ctx),
                kind.tol(),
            );
            assert_misses(
                &ctx,
                &format!("{here} vs the {}({w:.6}) reference", kind.other().key()),
                res.e_rpa,
                num(b, &format!("/{}/e_corr", kind.other().key()), &ctx),
                kind.tol(),
            );
            if i == I_UNIT_SLIP {
                at_slip[k] = res.e_rpa;
            }
        }
    }

    // Unit slip: the default's Å⁻¹ digits read as Bohr⁻¹ (ω = 0.42) must miss
    // the true default's (ω = 0.222254 Bohr⁻¹) reference. at_slip already
    // matched the 0.42 reference above.
    let b_def = att_block(r, OMEGAS[I_DEFAULT], &ctx);
    for (k, kind) in [Kind::Erf, Kind::Erfc].into_iter().enumerate() {
        assert_misses(
            &ctx,
            &format!(
                "unit slip: {}(0.42) vs the {}(0.222254) reference",
                kind.key(),
                kind.key()
            ),
            at_slip[k],
            num(b_def, &format!("/{}/e_corr", kind.key()), &ctx),
            kind.tol(),
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "validation: attenuated RPA"]
fn attenuated_rpa_h2o_cc_pvdz_vs_numpy() {
    run_case("h2o", "cc-pvdz");
}

#[test]
#[ignore = "validation: attenuated RPA"]
fn attenuated_rpa_nh3_cc_pvdz_vs_numpy() {
    run_case("nh3", "cc-pvdz");
}

#[test]
#[ignore = "validation: attenuated RPA"]
fn attenuated_rpa_h2o_aug_cc_pvdz_vs_numpy() {
    run_case("h2o", "aug-cc-pvdz");
}

/// Metric-operator control, driven through ferric itself: attenuated
/// (Q|op|ia) whitened with the COULOMB metric (what `run_pdep_rpa` would do if
/// `op` stopped reaching `coulomb_metric_2c`) must MATCH the generator's
/// `mixed_coulomb_metric` value and MISS the properly attenuated reference.
/// Proves the control number the main tests miss is the real failure value.
#[test]
#[ignore = "validation: attenuated RPA"]
fn attenuated_rpa_h2o_cc_pvdz_mixed_metric_control() {
    let sys = load_system("h2o", "cc-pvdz");
    let ctx = sys.label.clone();
    let w = OMEGAS[I_DEFAULT];
    let b = att_block(&sys.r, w, &ctx);

    let nocc_total = sys.mol.nelec() as usize / 2;
    let nbas = sys.obs.nbasis();
    let naux = sys.dfbs.nbasis();
    let c = sys.rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., 0..nocc_total]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();
    let v2c = threeindex::coulomb_metric_2c(Operator::coulomb(), &sys.dfbs).unwrap();
    let v_inv_sqrt = metric_inverse_sqrt(&v2c, Operator::coulomb()).unwrap();

    for kind in [Kind::Erf, Kind::Erfc] {
        let op = kind.op(w);
        let mut src = ThreeIndexSource::build(op, &sys.obs, &sys.dfbs, eri3_budget_bytes(None))
            .unwrap_or_else(|e| panic!("{ctx}: ThreeIndexSource({op:?}): {e:?}"));
        let b_ov = stream_dressed_mo_band(&mut src, &v_inv_sqrt, &c_occ, &c_vir, None).unwrap();
        let inter = RpaIntermediates {
            b_ov,
            v_inv_sqrt: v_inv_sqrt.clone(),
            nocc: nocc_total,
            nvir: nbas - nocc_total,
            nocc_total,
            first_occ: 0,
            naux,
        };
        let res = run_pdep_rpa_from_intermediates(
            &inter,
            &sys.mol,
            &sys.obs,
            &sys.dfbs,
            op,
            &sys.rhf,
            &rpa_cfg(),
        )
        .unwrap_or_else(|e| panic!("{ctx}: mixed-metric RPA failed: {e:?}"));
        let key = kind.key();
        check_close(
            &ctx,
            &format!("mixed-metric {key}(default) vs numpy"),
            res.e_rpa,
            num(b, &format!("/{key}/mixed_coulomb_metric"), &ctx),
            kind.tol(),
        );
        assert_misses(
            &ctx,
            &format!("mixed-metric {key}(default) vs the attenuated reference"),
            res.e_rpa,
            num(b, &format!("/{key}/e_corr"), &ctx),
            kind.tol(),
        );
    }
}
