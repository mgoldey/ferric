//! VALIDATION tier — row "attMP2 + VV10 (MP2-V)" (issue #273).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_mp2_v \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! References: `scripts/validation/gen_mp2_v.py` →
//! `testdata/reference/validation/mp2_v/<system>_<basis>.json` for H2O /
//! cc-pVDZ and aug-cc-pVDZ, NH3 / cc-pVDZ and the S22 water dimer /
//! aug-cc-pVDZ. Each file carries PySCF's exact-J/K RHF density in ferric AO
//! order, so the VV10 half is compared on the SAME density (no SCF error), on
//! ferric's NLC grid (50 × 50, unpruned) rebuilt in numpy from PySCF
//! primitives.
//!
//! * Undamped E_nl vs PySCF `dft.numint._vv10nlc` (b = 11.0 and 8.0,
//!   C = 0.0089). ferric's kernel is a port of `_vv10nlc`, so this checks the
//!   density on the grid, the grid, the 1e-8 threshold and the summation, not
//!   the VV10 formula.
//! * The formula is checked through the generator's numpy kernel written from
//!   Vydrov & Van Voorhis, JCP 133, 244103 (2010); the generator asserts it
//!   equals `_vv10nlc` undamped (measured 2.1e-17) before writing.
//! * Damped E_nl (paper Eq. 11, `1 − terfc(R, r₀)²` on the pair kernel only)
//!   vs that numpy kernel at the published r₀ = 1.00 Å (b 11.0) and the
//!   Table 1 point r₀ = 0.85 Å (b 8.0), both through the
//!   `AttVv10Config` constructors so the Å→Bohr conversion and the
//!   damping/MP2 r₀ lockstep are on the tested path.
//! * The erfc-attenuator MP2 half (the table-free control the module ships,
//!   ω = 1/(r₀√2) Bohr⁻¹, frozen core 1 per first-row heavy atom) vs numpy
//!   RI-MP2 on PySCF `with_range_coulomb(-ω)` integrals.
//! * The full `att_mp2_vv10` chain on ferric's own RHF: E_HF first, then E_c,
//!   E_nl and the additivity of `total`.
//! * `u_att_mp2_vv10` on a UHF singlet that collapsed onto RHF reproduces the
//!   same references (a construction anchor, not a parameter validation).
//!
//! NOT compared: the terfc-attenuated E_c. Its independent reference is the
//! row-119 terfc generator, which this row reuses instead of building a
//! second one; until it lands the terfc MP2 half has no external number.
//!
//! # Exactness anchors (run first in each system)
//!
//! * Undamped (`Vv10Damping::None`) equals `_vv10nlc`.
//! * r₀ → 0: only the R = 0 self pair stays damped (f(0) = 0 for every r₀), so
//!   damped(r₀ ≪ min pair distance) = undamped − S, with S the self-pair term
//!   the generator records. Checked here against ferric's damped kernel.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! Recorded before measuring in `HYPOTHESES-mp2-v-validation.md`. In short:
//! real ⇒ every VV10 comparison at summation noise (≲1e-12 Ha), erfc E_c at
//! the attenuated-row floor (≲1e-11). Wrong AO order ⇒ the overlap check fails
//! first; wrong grid ⇒ the point count / ∫ρ check fails first; r₀ unit slip or
//! inverted damping ⇒ the damped comparison misses by ≥ 1e-4.
//!
//! # TOLERANCES
//!
//! Each bar is about 10× the measured maximum written beside its const. The
//! VV10 comparisons on the reference density agree to 3.1e-16 Ha (a few ulps
//! of E_nl ≈ 0.02–0.06 Ha), the erfc E_c to 7.0e-12, E_HF to 8.1e-12.
//!
//! # NEGATIVE CONTROLS (always on)
//!
//! * Damped reference minus undamped reference ≥ [`MUST_MISS`] on every system
//!   (the damping is live; measured 4.6e-4 to 1.8e-3 Ha).
//! * ferric's E_nl on a (75, 110) grid with the same density misses the
//!   (50, 50) reference by ≥ [`MUST_MISS`] (measured 6.8e-7 to 3.7e-6 Ha, i.e.
//!   ≥ 1e8 × [`TOL_VV10`]): the VV10 bar measures the grid match.
//! * The erfc E_c misses the Coulomb RI-MP2 reference by ≥ [`MUST_MISS`]
//!   (measured ≥ 2.3e-2 Ha).
//!
//! # MUTATION LEDGER
//!
//! See the end of this file.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_dft::libxc::Vv10Params;
use ferric_dft::vv10::Vv10Damping;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::att_vv10::{
    att_mp2_vv10, u_att_mp2_vv10, vv10_energy_on_density, AttVv10Config, AttVv10Result,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::{ScfResult, Spin};
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/mp2_v";

/// E_nuc vs the reference. Measured 0.
const TOL_ENUC: f64 = 1e-9;
/// PySCF overlap in ferric AO order vs ferric's overlap. Measured max 1.3e-15.
const TOL_S: f64 = 1e-12;
/// ∫ρ on ferric's NLC grid vs the numpy replica. Measured max 1.2e-13.
const TOL_GRID: f64 = 1e-12;
/// Σw on the grid, RELATIVE. Measured max 1.8e-14 relative (3.6e-10 abs on
/// Σw = 19641, water dimer).
const TOL_GRID_REL: f64 = 2e-13;
/// E_nl on the reference density: undamped vs `_vv10nlc`, r₀ → 0 vs
/// undamped − S, damped vs numpy. Measured max 3.1e-16 over 4 systems × 2
/// (r₀, b) × 3 quantities. The margin is ~16×: both sides are dense O(N²)
/// sums of ~1e4 terms, so a different summation order alone could move the
/// last few ulps; a defect moves it by ≥ 1e-7 (see the grid control).
const TOL_VV10: f64 = 5e-15;
/// RHF total energy vs PySCF. Measured max 8.1e-12 (water dimer).
const TOL_RHF: f64 = 1e-10;
/// erfc E_c vs numpy RI-MP2. Measured max 7.0e-12 (water dimer).
const TOL_CORR: f64 = 1e-10;
/// E_nl on ferric's OWN RHF density vs the reference (carries the SCF density
/// error). Measured max 1.1e-13.
const TOL_VV10_CHAIN: f64 = 1e-12;
/// UHF-singlet path vs the references (E_c and E_nl; the SCF collapse floor).
/// Measured max 7.5e-12 (E_c, H2O) and 1.3e-13 (E_nl).
const TOL_UHF: f64 = 1e-10;
/// A control must miss by at least this much: 100× the loosest bar it is
/// set against (TOL_CORR) and 2e6× TOL_VV10. Smallest measured miss: the
/// (75, 110) grid control, 6.8e-7 (water dimer, b = 11).
const MUST_MISS: f64 = 1e-8;

struct Case {
    system: &'static str,
    xyz: &'static str,
    basis: &'static str,
    aux: &'static str,
}

const CASES: [Case; 4] = [
    Case {
        system: "h2o",
        xyz: "testdata/molecules/validation/h2o.xyz",
        basis: "cc-pvdz",
        aux: "cc-pvdz-ri",
    },
    Case {
        system: "h2o",
        xyz: "testdata/molecules/validation/h2o.xyz",
        basis: "aug-cc-pvdz",
        aux: "aug-cc-pvdz-rifit",
    },
    Case {
        system: "nh3",
        xyz: "testdata/molecules/validation/nh3.xyz",
        basis: "cc-pvdz",
        aux: "cc-pvdz-ri",
    },
    Case {
        system: "water_dimer",
        xyz: "testdata/molecules/s22/water_dimer.xyz",
        basis: "aug-cc-pvdz",
        aux: "aug-cc-pvdz-rifit",
    },
];

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

fn reference(c: &Case) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{}_{}.json", c.system, c.basis));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_mp2_v.py — a missing reference is a failure, never a skip",
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

fn mat(v: &Value, ptr: &str, ctx: &str) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"));
    let n = rows.len();
    let m = rows
        .first()
        .and_then(Value::as_array)
        .map_or(0, |r| r.len());
    Array2::from_shape_fn((n, m), |(i, j)| {
        rows[i][j]
            .as_f64()
            .unwrap_or_else(|| panic!("{ctx}: {ptr}[{i}][{j}] is not a number"))
    })
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<24} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_misses(ctx: &str, what: &str, a: f64, b: f64) {
    let d = (a - b).abs();
    eprintln!("{ctx}: control {what:<30} |d| {d:.2e} (must be >= {MUST_MISS:.0e})");
    assert!(
        d >= MUST_MISS,
        "{ctx}: control {what} did not miss (|d| {d:.2e} < {MUST_MISS:.0e})"
    );
}

struct Loaded {
    ctx: String,
    r: Value,
    mol: Molecule,
    bs: BasisSet,
    obs: PreparedBasis,
    d_ref: Array2<f64>,
}

/// Geometry, basis, AO-order and grid checks; returns the reference density.
fn load(c: &Case) -> Loaded {
    let ctx = format!("{}/{}", c.system, c.basis);
    let r = reference(c);
    let xyz = workspace_root().join(c.xyz);
    let mol = Molecule::load_xyz(xyz.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let bs = basis::bundled(c.basis).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    assert_eq!(
        obs.nbasis(),
        r["nao"].as_u64().unwrap() as usize,
        "{ctx}: nao"
    );
    let s = ferric_integrals::oneelectron::overlap(&obs);
    let s_ref = mat(&r, "/overlap_ferric_order", &ctx);
    let ds = (&s - &s_ref).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    eprintln!("{ctx}: overlap max|d| {ds:.2e} (tol {TOL_S:.0e})");
    assert!(
        ds < TOL_S,
        "{ctx}: PySCF overlap in ferric AO order differs by {ds:.2e}; the AO permutation is \
         wrong and no density-level comparison below would mean anything"
    );
    let d_ref = mat(&r, "/density_ferric_order", &ctx);

    // Grid: point count, Σw and ∫ρ on ferric's NLC grid vs the numpy replica.
    let grid_cfg = AttVv10Config::default().nlc_grid;
    let grid = build_atomic_grid(&mol, &grid_cfg);
    assert_eq!(
        grid.len(),
        r["nlc_grid"]["npts"].as_u64().unwrap() as usize,
        "{ctx}: NLC grid point count"
    );
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = ferric_dft::ao_grid::eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();
    let dens = ferric_dft::density_on_grid::eval_density_closed(&d_ref, &chi, &dchi);
    let sum_w: f64 = grid.iter().map(|g| g.weight).sum();
    let n_el: f64 = grid
        .iter()
        .zip(dens.rho.iter())
        .map(|(g, rho)| g.weight * rho)
        .sum();
    check_close(
        &ctx,
        "grid sum w",
        sum_w,
        num(&r, "/nlc_grid/sum_weights", &ctx),
        TOL_GRID_REL * sum_w,
    );
    check_close(
        &ctx,
        "grid int rho",
        n_el,
        num(&r, "/nlc_grid/n_electrons_on_grid", &ctx),
        TOL_GRID,
    );
    Loaded {
        ctx,
        r,
        mol,
        bs,
        obs,
        d_ref,
    }
}

fn e_nl(l: &Loaded, b: f64, damping: Vv10Damping, grid: &AtomicGridConfig) -> f64 {
    let params = Vv10Params { c: 0.0089, b };
    vv10_energy_on_density(&l.mol, &l.bs, &l.d_ref, &params, damping, grid)
        .unwrap_or_else(|e| panic!("{}: vv10_energy_on_density: {e:?}", l.ctx))
        .0
}

/// The two damped configurations, built through the public constructors so
/// the Å→Bohr conversion, the Table 1 b and the damping lockstep are on the
/// tested path: (reference key, config).
fn damped_configs() -> [(&'static str, AttVv10Config); 2] {
    [
        ("b=11.0", AttVv10Config::mp2_v_terfc_atz()),
        (
            "b=8.0",
            AttVv10Config::mp2_v_terfc_atz_at_r0(0.85).expect("Table 1 has r0 = 0.85 A"),
        ),
    ]
}

/// VV10 half on the reference density: anchors, damped values, controls.
fn check_vv10(c: &Case) {
    let l = load(c);
    let grid = AttVv10Config::default().nlc_grid;
    for (key, cfg) in damped_configs() {
        let ctx = format!("{}/{key}", l.ctx);
        let blk = &l.r["vv10"][key];
        let b = cfg.vv10.b;

        // Anchor 1: undamped vs PySCF _vv10nlc.
        let undamped = e_nl(&l, b, Vv10Damping::None, &grid);
        let undamped_ref = num(blk, "/undamped_pyscf_vv10nlc", &ctx);
        check_close(&ctx, "E_nl undamped", undamped, undamped_ref, TOL_VV10);

        // Anchor 2: r0 -> 0 leaves only the self pair damped.
        let r0_tiny = num(blk, "/r0_to_zero/r0_bohr", &ctx);
        let tiny = e_nl(
            &l,
            b,
            Vv10Damping::Terfc {
                r0_bohr: r0_tiny,
                omega_bohr_inv: None,
            },
            &grid,
        );
        check_close(
            &ctx,
            "E_nl r0->0",
            tiny,
            num(blk, "/r0_to_zero/undamped_minus_self_pair", &ctx),
            TOL_VV10,
        );

        // Damped, through the config the method actually uses.
        let damping = cfg.effective_vv10_damping().unwrap();
        let damped = e_nl(&l, b, damping, &grid);
        let damped_ref = num(blk, "/damped/e_nl_numpy", &ctx);
        assert!(
            (num(blk, "/damped/r0_bohr", &ctx) - cfg.r0_bohr).abs() < 1e-12,
            "{ctx}: config r0 differs from the reference's"
        );
        check_close(&ctx, "E_nl damped", damped, damped_ref, TOL_VV10);

        // Controls: the damping is live; the grid match is what the bar sees.
        check_misses(&ctx, "damped ref vs undamped ref", damped_ref, undamped_ref);
        let other_grid = AtomicGridConfig {
            n_radial: 75,
            n_angular: 110,
            prune: None,
        };
        let off_grid = e_nl(&l, b, damping, &other_grid);
        check_misses(&ctx, "(75,110) grid vs (50,50) ref", off_grid, damped_ref);
    }
}

#[test]
#[ignore = "validation: attMP2 + VV10 (MP2-V)"]
fn vv10_half_h2o_ccpvdz() {
    check_vv10(&CASES[0]);
}

#[test]
#[ignore = "validation: attMP2 + VV10 (MP2-V)"]
fn vv10_half_h2o_aug_ccpvdz() {
    check_vv10(&CASES[1]);
}

#[test]
#[ignore = "validation: attMP2 + VV10 (MP2-V)"]
fn vv10_half_nh3_ccpvdz() {
    check_vv10(&CASES[2]);
}

#[test]
#[ignore = "validation: attMP2 + VV10 (MP2-V)"]
fn vv10_half_water_dimer_aug_ccpvdz() {
    check_vv10(&CASES[3]);
}

fn rhf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 200,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    }
}

/// The erfc-control MP2-V config at the published (r0, b, C) with the
/// paper's frozen core.
fn erfc_config(r: &Value, ctx: &str) -> AttVv10Config {
    let mut cfg = AttVv10Config::erfc_control_at_atz_params();
    cfg.frozen_core = r["frozen_core"].as_u64().unwrap() as usize;
    let omega = 1.0 / (cfg.r0_bohr * std::f64::consts::SQRT_2);
    assert!(
        (omega - num(r, "/mp2_erfc_control/omega_bohr_inv", ctx)).abs() < 1e-12,
        "{ctx}: erfc control omega differs from the reference's"
    );
    cfg
}

/// The reference-side checks shared by the RHF and the UHF-singlet chains.
fn check_result(ctx: &str, r: &Value, res: &AttVv10Result, tol_corr: f64, tol_nl: f64) {
    check_close(
        ctx,
        "E_c(erfc)",
        res.e_c_att_mp2,
        num(r, "/mp2_erfc_control/e_corr", ctx),
        tol_corr,
    );
    check_close(
        ctx,
        "E_nl damped (own density)",
        res.e_nl_vv10,
        num(r, "/vv10/b=11.0/damped/e_nl_numpy", ctx),
        tol_nl,
    );
    let resid = res.components_sum_to_total();
    eprintln!("{ctx}: total - (E_HF + E_c + E_nl) = {resid:.2e}");
    assert!(
        resid.abs() <= 4.0 * f64::EPSILON * res.total.abs(),
        "{ctx}: total is not E_HF + E_c + E_nl ({resid:.2e})"
    );
    check_misses(
        ctx,
        "E_c(erfc) vs Coulomb ref",
        res.e_c_att_mp2,
        num(r, "/mp2_erfc_control/coulomb_e_corr", ctx),
    );
}

fn check_chain(c: &Case, with_uhf: bool) {
    let l = load(c);
    let ctx = l.ctx.clone();
    let r = &l.r;
    let dfbs = PreparedBasis::new(&l.mol, &basis::bundled(c.aux).unwrap()).unwrap();
    assert_eq!(
        dfbs.nbasis(),
        r["naux"].as_u64().unwrap() as usize,
        "{ctx}: naux"
    );
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &l.obs).unwrap();
    let rhf: ScfResult = solve_rhf(
        &ParallelContext::default(),
        &l.mol,
        &l.obs,
        op,
        &bounds,
        &rhf_config(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: RHF failed: {e:?}"));
    assert!(rhf.converged, "{ctx}: RHF not converged");
    check_close(
        &ctx,
        "E_RHF",
        rhf.energy,
        num(r, "/rhf_energy", &ctx),
        TOL_RHF,
    );

    let cfg = erfc_config(r, &ctx);
    let res = att_mp2_vv10(&l.mol, &l.obs, &l.bs, &dfbs, &rhf, &cfg)
        .unwrap_or_else(|e| panic!("{ctx}: att_mp2_vv10: {e:?}"));
    check_close(
        &ctx,
        "E_HF (result)",
        res.e_hf,
        num(r, "/rhf_energy", &ctx),
        TOL_RHF,
    );
    check_result(&ctx, r, &res, TOL_CORR, TOL_VV10_CHAIN);

    if !with_uhf {
        return;
    }
    let uctx = format!("{ctx}/UHF-singlet");
    let uhf = ferric_scf::uhf::solve_uhf(
        &ParallelContext::default(),
        &l.mol,
        &l.obs,
        &bounds,
        &rhf_config(),
    )
    .unwrap_or_else(|e| panic!("{uctx}: UHF failed: {e:?}"));
    assert_eq!(uhf.spin, Spin::Unrestricted, "{uctx}: spin");
    // The collapse premise: the comparison means nothing on a broken-symmetry state.
    check_close(
        &uctx,
        "E_UHF",
        uhf.energy,
        num(r, "/rhf_energy", &uctx),
        TOL_RHF,
    );
    let ures = u_att_mp2_vv10(&l.mol, &l.obs, &l.bs, &dfbs, &uhf, &cfg)
        .unwrap_or_else(|e| panic!("{uctx}: u_att_mp2_vv10: {e:?}"));
    assert!(ures.is_open_shell_extrapolation(), "{uctx}: flag");
    check_result(&uctx, r, &ures, TOL_UHF, TOL_UHF);
}

#[test]
#[ignore = "validation: attMP2 + VV10 (MP2-V)"]
fn mp2_v_chain_h2o_ccpvdz() {
    check_chain(&CASES[0], true);
}

#[test]
#[ignore = "validation: attMP2 + VV10 (MP2-V)"]
fn mp2_v_chain_h2o_aug_ccpvdz() {
    check_chain(&CASES[1], false);
}

#[test]
#[ignore = "validation: attMP2 + VV10 (MP2-V)"]
fn mp2_v_chain_nh3_ccpvdz() {
    check_chain(&CASES[2], true);
}

#[test]
#[ignore = "validation: attMP2 + VV10 (MP2-V)"]
fn mp2_v_chain_water_dimer_aug_ccpvdz() {
    check_chain(&CASES[3], false);
}

// MUTATION LEDGER (2026-10-05). Each mutant was applied to the source,
// compiled, and run against `vv10_half_h2o_ccpvdz` + `mp2_v_chain_h2o_ccpvdz`
// with `--ignored` (counts read from `test result:`); every one FAILED a named
// assertion and the source was restored and checked clean afterwards.
//
// | mutant | where | first failing assertion | miss |
// |---|---|---|---|
// | invert `1 − terfc²` → `terfc²` | vv10.rs `factor_from_r2` | `E_nl r0->0`; chain `E_nl damped (own density)` | 1.4e-3; 4.1e-4 |
// | wrong b (11.0 → 10.0) | `mp2_v_terfc_atz` | `E_nl undamped` (b read from the config) | 2.6e-3 |
// | damping r₀ in Å, MP2 r₀ in Bohr | `effective_vv10_damping` | `E_nl damped` | 2.8e-4 |
// | drop grid weight in the E_nl sum | vv10.rs finalisation | `E_nl undamped` | 2.8e2 |
// | erfc ω from r₀ in Å | `mp2_operator` | `E_c(erfc)` | 6.7e-2 |
// | frozen core ignored | `ri_mp2_config` | `E_c(erfc)` | 2.3e-3 |
// | drop VV10 from `total` | `assemble` | additivity of `total` | 1.9e-2 |
// | damping disabled in `assemble` | `assemble` | chain `E_nl damped (own density)` | 4.7e-4 |
// | Table 1 b at 0.85 Å (8.0 → 9.0) | `TABLE1_R0_B_PAIRS` | `E_nl undamped` (b = 8 block) | 4.4e-3 |
//
// Of the 8 mutations seeded for the earlier Smoke grade, 7 are reached here
// (drop VV10, Å→Bohr, disable damping, drop weights, invert the factor, wrong
// b, desync damping r₀). "Hold b fixed across the valley" is the Table 1 row
// above. Not reached: the terfc MP2 operator (no reference for it here).
