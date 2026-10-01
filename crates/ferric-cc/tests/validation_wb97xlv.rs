//! VALIDATION tier — VALIDATION.md row "ωB97X-L-V components".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-cc --test validation_wb97xlv \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ωB97X-L-V (Ransford & Carter-Fenk, PCCP 28, 14428 (2026), eqn 27):
//!
//! ```text
//!   E = E_KS[ωB97X-L + VV10(b = 10, C = 0.01)] + λ·E_c,LinLCCD(hh)^{sr,ω,λ}
//! ```
//!
//! No absolute ωB97X-L-V energy is published anywhere
//! (`testdata/reference/wb97x_l_v_params.json`), so the two COMPONENTS are each
//! checked against an independent construction, then their sum:
//!
//! * E_KS vs PySCF 2.13 KS with libxc `HYB_GGA_XC_WB97X_V` customised with
//!   ferric's 18 ext params (parsed from `WB97X_L_V_EXT_PARAMS` and refused
//!   unless equal to the paper transcription and in libxc's own name order),
//!   VV10 overridden to b = 10, C = 0.01. Water RKS, OH ²Π ROKS.
//! * λ·E_c vs numpy LinLCCD(hh) (exact eigen-solve, `gen_linlccd.py`) on
//!   integrals of λ·erfc(ωr)/r fitted in the same operator's metric with
//!   ferric's def2-SVP-RIFIT. OH: ROKS → one unrestricted KS Fock build →
//!   semicanonical per-spin orbitals → unrestricted LinLCCD(hh).
//!
//! Generator: `scripts/validation/gen_wb97xlv.py` →
//! `testdata/reference/validation/wb97xlv/<system>_def2-svp.json`.
//!
//! # Like-for-like recipe (read from ferric's code)
//!
//! * Grid: ferric's defaults, (75,110) unpruned Becke + VV10 on (50,50)
//!   unpruned (`rhf.rs`/`rohf.rs`/`XcSpec::build` all default to these); the
//!   reference uses Becke-1988 radii adjust on BOTH grids. Pre-measured on
//!   water RI-J: Treutler adjust on PySCF's NLC grid moves E_KS by 3.3e-7.
//! * Exchange: ferric builds range-separated K ONLY by DF (K[erfc] and K[erf]
//!   each in its own attenuated metric, def2-universal-jkfit); the reference
//!   serves every K request that way (see the generator's `ks_class`).
//! * Coulomb: EXACT J in the `ks` blocks (water: `df_j_aux = Some("")`; OH:
//!   ROKS ignores `df_j_aux` for an RSH functional). Water's `ks_rij` block
//!   is the production `run_wb97x_l_v` recipe (RI-J, jkfit).
//! * Correlation: all electrons (`DoubleHybridConfig` default frozen_core 0).
//!
//! # Two kinds of E_c comparison
//!
//! E_c is NOT variational in the orbitals, so a ~1e-6 Ha E_KS difference
//! between two independent SCFs moves E_c at first order (RI-J vs exact-J
//! orbitals alone move water's λ·E_c by 8.9e-7). A 1e-8 bar is therefore
//! applied only on IDENTICAL orbitals: the reference orbitals (written in
//! ferric's AO order) are INJECTED into a clone of ferric's own ScfResult.
//! The AO mapping is proved first by comparing ferric's AO overlap with the
//! reference's (a permutation or sign error in p/d fails there, not as a
//! mystery energy). The end-to-end E_c on ferric's own orbitals, and the
//! total, get the separate orbital-limited bars.
//!
//! # Exactness anchors (asserted first)
//!
//! E_nuc, nao, naux, AO overlap; λ = 0 returns exactly the KS energy;
//! LADDER OFF: `DriversOnly` under λ·erfc(ω) on injected orbitals equals the
//! reference, which the generator proved equal to λ² × PySCF DF-(U)MP2 run with
//! `range_coulomb(-ω)` to 1e-12 (measured ≤ 1.4e-17); closed shell also
//! against ferric's own `ri_mp2_spin_components(erfc(ω)) × λ²`.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric is right: E_KS agrees to the XC-construction floor
//!   (pre-measured water RI-J with the mainline Python build: 6.5e-10 Ha);
//!   injected E_c to the RI/DIIS floor (~1e-11).
//! * λ outside the amplitude equations (the pre-#239 defect, solve at λ = 1
//!   and multiply by λ): λ·E_c misses by 4.3e-2 Ha (water) — the JSON's
//!   `negative_control_linear_lambda` must be MISSED.
//! * Stock VV10 b = 6 instead of 10: E_KS misses by 2.2e-2 Ha
//!   (`control_vv10_b6_energy`, must be MISSED).
//! * A mistranscribed ext param: not caught here (both sides read the same
//!   Rust constant) — that is `ferric-dft/tests/wb97x_l_v_published_params.rs`.
//! * Harness broken (geometry, basis, AO order): E_nuc / nao / overlap fail
//!   before any energy is examined.
//! * OH: the reference ROKS spread over six starts is 5.3e-7 Ha (π-hole
//!   orientation on the Lebedev grid; selected start max|g| 3.2e-6), so the
//!   OH E_KS bar cannot go below ~1e-6.
//! * OH in a different state (σ hole): ≥ 0.1 Ha, far outside every bar; the
//!   generator refuses a non-π SOMO.
//!
//! # TOLERANCES (measured 2026-10-01, release, water and OH / def2-SVP)
//!
//! | quantity | measured max | bar |
//! |---|---:|---:|
//! | AO overlap | 6.7e-16 | 1e-10 |
//! | E_KS (exact J, RI-J, ROKS) | 1.05e-9 | 1e-8 |
//! | λ·E_c and DriversOnly on injected orbitals | 7.2e-14 | 1e-11 |
//! | DriversOnly vs λ² × ferric SR-MP2 | 1.4e-17 | 1e-12 |
//! | λ·E_c on ferric's own orbitals | 1.06e-7 | 5e-7 |
//! | total, own orbitals | 1.07e-7 | 5e-7 |
//! | OH semicanonical orbital energies | 3.3e-5 | 1e-4 |
//!
//! Every bar sits below the smallest defect it must catch: RI-J vs exact-J
//! orbitals move λ·E_c by 8.9e-7 and E_KS by 2.9e-5; the controls below must
//! each be missed by MUST_MISS = 1e-5, which is 20x the loosest energy bar.
//!
//! # Negative controls (always on)
//!
//! Linear-λ value, DriversOnly vs Hh, VV10 b = 6 control, exact RHF/ROHF
//! controls, and the RI-J vs exact-J reference split.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_cc::double_hybrid::{
    run_wb97x_l_v, solve_wb97x_l_v, u_solve_wb97x_l_v, DoubleHybridConfig, WB97X_L_V_NAME,
};
use ferric_cc::linlccd::{linlccd, LadderVariant};
use ferric_cc::linlccd_u::u_linlccd;
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::{Operator, OperatorKind};
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::semicanonical::{semicanonicalize, XcSpec};
use ferric_scf::{ScfResult, Spin};
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/wb97xlv";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "def2-svp";
const CORR_AUX: &str = "def2-svp-rifit";
const LAMBDA: f64 = 0.6;
const OMEGA: f64 = 0.1;

/// Geometry check (PySCF E_nuc from ferric's Bohr geometry).
const TOL_ENUC: f64 = 1e-9;
/// ferric AO overlap vs the reference's (ferric AO order). Construction check.
const TOL_OVERLAP: f64 = 1e-10;
/// E_KS vs PySCF (plan bar 5e-5; kept far below |RI-J − exact J| = 2.9e-5).
const TOL_E_KS: f64 = 1e-8;
/// λ·E_c and DriversOnly on INJECTED reference orbitals (plan bar 1e-8).
const TOL_E_C_INJECTED: f64 = 1e-11;
/// ferric DriversOnly(λ·erfc) vs λ² × ferric RI-MP2(erfc): same code family,
/// different path (composite-operator scaling).
const TOL_SELF: f64 = 1e-12;
/// λ·E_c on ferric's OWN orbitals vs the reference's (orbital-limited).
/// RI-J vs exact-J orbitals alone move water's λ·E_c by 8.9e-7; the bar is below that.
const TOL_E_C_OWN: f64 = 5e-7;
/// Total ωB97X-L-V energy (E_KS + λ·E_c, both own) vs the reference sum.
const TOL_TOTAL: f64 = 5e-7;
/// Semicanonical per-spin KS orbital energies, OH (ferric's own vs reference).
const TOL_EPS: f64 = 1e-4;
/// A reference ferric must MISS (negative controls). Smallest planted gap:
/// |hh − drivers| = 7.5e-3 (water); linear-λ 4.3e-2; VV10 b=6 2.2e-2.
/// Smallest control in use: E_KS RI-J vs exact J, 2.9e-5.
const MUST_MISS: f64 = 1e-5;

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
        .expect("ferric-cc manifest dir should be <root>/crates/ferric-cc")
        .to_path_buf()
}

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_wb97xlv.py — a missing reference is a failure, never a skip",
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

fn vec_at(v: &Value, ptr: &str, ctx: &str) -> Vec<f64> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn mat_at(v: &Value, ptr: &str, ctx: &str) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a matrix"));
    let n = rows.len();
    let m = rows[0].as_array().expect("row").len();
    let mut out = Array2::<f64>::zeros((n, m));
    for (i, r) in rows.iter().enumerate() {
        for (j, x) in r.as_array().expect("row").iter().enumerate() {
            out[[i, j]] = x.as_f64().expect("number");
        }
    }
    out
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<34} ferric {got:+.13} ref {want:+.13} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.13} vs reference {want:.13} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_miss(ctx: &str, what: &str, got: f64, want: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<34} |d| {d:.2e} (must exceed {MUST_MISS:.0e})");
    assert!(
        d > MUST_MISS,
        "{ctx}: negative control {what}: |d| {d:.2e} <= {MUST_MISS:.0e} — the bar cannot \
         tell these apart"
    );
}

/// λ·erfc(ω)/r — the operator `double_hybrid` builds (eqn 22).
fn lambda_erfc() -> Operator {
    Operator::composite(&[(LAMBDA, OperatorKind::ErfcCoulomb, OMEGA)])
}

fn cc_tight() -> CcConfig {
    let cfg = CcConfig {
        energy_conv: 1e-12,
        max_iter: 200,
        ..Default::default()
    };
    assert_eq!(cfg.frozen_core, 0, "all-electron expected");
    cfg
}

fn dh_tight() -> DoubleHybridConfig {
    let cfg = DoubleHybridConfig {
        cc: cc_tight(),
        ..Default::default()
    };
    assert_eq!(cfg.lambda, LAMBDA);
    assert_eq!(cfg.omega, OMEGA);
    assert_eq!(cfg.variant, LadderVariant::Hh);
    cfg
}

/// Tight KS config with ferric's default grids ((75,110) / VV10 (50,50)).
/// `exact_j`: explicit conventional J (`Some("")`); K stays RI in the
/// attenuated metrics with def2-universal-jkfit (the only RSH-K path).
fn ks_config(exact_j: bool) -> RhfConfig {
    let mut cfg = RhfConfig {
        max_iter: 300,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    };
    cfg.xc = Some(WB97X_L_V_NAME.to_string());
    assert!(
        cfg.dft_grid.is_none() && cfg.nlc_grid.is_none(),
        "default grids expected"
    );
    if exact_j {
        cfg.df_j_aux = Some(String::new());
    }
    assert!(
        cfg.df_k_aux.is_none(),
        "RSH K must use the default jkfit aux"
    );
    cfg
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
    r: Value,
    ctx: String,
}

/// Geometry / basis / AO-order anchors.
fn setup(system: &str) -> Sys {
    let r = reference(system);
    let ctx = format!("{system}/{BASIS}");
    assert_eq!(r["basis"].as_str(), Some(BASIS), "{ctx}: basis");
    assert_eq!(r["corr_aux_basis"].as_str(), Some(CORR_AUX), "{ctx}: aux");
    assert_eq!(num(&r, "/lambda", &ctx), LAMBDA, "{ctx}: lambda");
    assert_eq!(num(&r, "/omega", &ctx), OMEGA, "{ctx}: omega");
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let obs = PreparedBasis::new(&mol, &basis::bundled(BASIS).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(CORR_AUX).unwrap()).unwrap();
    assert_eq!(
        obs.nbasis() as u64,
        r["nao"].as_u64().unwrap(),
        "{ctx}: nao"
    );
    assert_eq!(
        dfbs.nbasis() as u64,
        r["naux_corr"].as_u64().unwrap(),
        "{ctx}: naux"
    );

    // AO-order proof: the reference matrices are in ferric's AO order iff the
    // two overlaps agree element-wise (a p/d permutation or sign flip changes
    // off-diagonal overlaps between centres).
    let s_ref = mat_at(&r, "/ao_overlap_ferric_order", &ctx);
    let s = ferric_integrals::oneelectron::overlap(&obs);
    let d = (&s - &s_ref).iter().fold(0.0f64, |m, x| m.max(x.abs()));
    eprintln!("{ctx}: AO overlap max |d| {d:.2e} (tol {TOL_OVERLAP:.0e})");
    assert!(
        d < TOL_OVERLAP,
        "{ctx}: AO order/sign mismatch, overlap |d| {d:.2e}"
    );

    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    Sys {
        mol,
        obs,
        dfbs,
        bounds,
        r,
        ctx,
    }
}

/// max |CᵀSC − I| for injected orbitals (a second, orbital-level AO-order check).
fn orthonormality_defect(c: &Array2<f64>, obs: &PreparedBasis) -> f64 {
    let s = ferric_integrals::oneelectron::overlap(obs);
    let m = c.t().dot(&s).dot(c);
    let mut d = 0.0f64;
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            let want = if i == j { 1.0 } else { 0.0 };
            d = d.max((m[[i, j]] - want).abs());
        }
    }
    d
}

/// Closed shell (water): E_KS (exact J), injected-orbital E_c, own-orbital E_c, total.
#[test]
#[ignore = "validation: ωB97X-L-V components"]
fn wb97xlv_h2o_def2svp_components_vs_pyscf() {
    let sys = setup("h2o");
    let (r, ctx) = (&sys.r, sys.ctx.as_str());
    let pctx = ParallelContext::default();

    // ---- E_KS ----
    let ks = solve_rhf(
        &pctx,
        &sys.mol,
        &sys.obs,
        Operator::coulomb(),
        &sys.bounds,
        &ks_config(true),
    )
    .unwrap_or_else(|e| panic!("{ctx}: RKS failed: {e:?}"));
    assert!(ks.converged, "{ctx}: RKS not converged");
    let e_ks_ref = num(r, "/ks/energy", ctx);
    check_close(ctx, "E_KS (exact J)", ks.energy, e_ks_ref, TOL_E_KS);
    let rij_split = num(r, "/rij_minus_exact_j_ks", ctx).abs();
    assert!(
        TOL_E_KS < rij_split,
        "{ctx}: TOL_E_KS {TOL_E_KS:.0e} cannot tell RI-J from exact J (split {rij_split:.2e})"
    );
    check_miss(
        ctx,
        "E_KS vs exact RHF",
        ks.energy,
        num(r, "/rhf_control/energy", ctx),
    );
    check_miss(
        ctx,
        "E_KS vs VV10 b=6 control",
        ks.energy,
        num(r, "/ks/control_vv10_b6_energy", ctx),
    );

    // ---- trivial limit: λ = 0 adds nothing ----
    let zero = solve_wb97x_l_v(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        &ks,
        &DoubleHybridConfig {
            lambda: 0.0,
            ..dh_tight()
        },
    )
    .unwrap();
    assert_eq!(
        zero.e_c_scaled, 0.0,
        "{ctx}: lambda = 0 must add no correlation"
    );
    assert_eq!(
        zero.total_energy, ks.energy,
        "{ctx}: lambda = 0 total must be E_KS"
    );

    // ---- E_c on INJECTED reference orbitals ----
    let mut inj: ScfResult = ks.clone();
    inj.mos_alpha = mat_at(r, "/ks/mo_coeff", ctx);
    inj.eps_alpha = vec_at(r, "/ks/mo_energy", ctx);
    assert_eq!(inj.mos_alpha.dim(), ks.mos_alpha.dim(), "{ctx}: MO shape");
    let orth = orthonormality_defect(&inj.mos_alpha, &sys.obs);
    eprintln!("{ctx}: injected CᵀSC − I max {orth:.2e}");
    assert!(
        orth < 1e-8,
        "{ctx}: injected orbitals not orthonormal in ferric's AO basis"
    );

    let cc = cc_tight();
    let run = |v: LadderVariant| {
        linlccd(&sys.mol, &sys.obs, &sys.dfbs, lambda_erfc(), &inj, &cc, v)
            .unwrap_or_else(|e| panic!("{ctx}: linlccd {v:?}: {e:?}"))
            .correlation_energy
    };
    let ref_drv = num(r, "/corr/drivers_only_scaled", ctx);
    let ref_hh = num(r, "/corr/e_c_scaled", ctx);
    let e_drv = run(LadderVariant::DriversOnly);
    check_close(
        ctx,
        "DriversOnly(λ·erfc) injected",
        e_drv,
        ref_drv,
        TOL_E_C_INJECTED,
    );
    let own_mp2 = ri_mp2_spin_components(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::erfc(OMEGA),
        &inj,
        &RiMp2Config::default(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: ri_mp2: {e:?}"))
    .0
    .e_total;
    check_close(
        ctx,
        "DriversOnly vs λ²·ferric SR-MP2",
        e_drv,
        LAMBDA * LAMBDA * own_mp2,
        TOL_SELF,
    );
    let e_hh = run(LadderVariant::Hh);
    check_close(
        ctx,
        "λ·E_c LinLCCD(hh) injected",
        e_hh,
        ref_hh,
        TOL_E_C_INJECTED,
    );
    // The public double-hybrid path builds the same operator.
    let dh_inj = solve_wb97x_l_v(&sys.mol, &sys.obs, &sys.dfbs, &inj, &dh_tight()).unwrap();
    check_close(
        ctx,
        "solve_wb97x_l_v e_c_scaled inj",
        dh_inj.e_c_scaled,
        ref_hh,
        TOL_E_C_INJECTED,
    );
    check_close(
        ctx,
        "e_c_wft = e_c_scaled / λ",
        dh_inj.e_c_wft,
        num(r, "/corr/e_c_wft", ctx),
        TOL_E_C_INJECTED / LAMBDA,
    );
    check_miss(ctx, "Hh vs DriversOnly ref", e_hh, ref_drv);
    check_miss(ctx, "DriversOnly vs hh ref", e_drv, ref_hh);
    check_miss(
        ctx,
        "λ·E_c vs linear-λ (pre-#239)",
        dh_inj.e_c_scaled,
        num(r, "/corr/negative_control_linear_lambda", ctx),
    );

    // ---- E_c and total on ferric's OWN orbitals ----
    let dh = solve_wb97x_l_v(&sys.mol, &sys.obs, &sys.dfbs, &ks, &dh_tight()).unwrap();
    check_close(
        ctx,
        "λ·E_c own orbitals",
        dh.e_c_scaled,
        ref_hh,
        TOL_E_C_OWN,
    );
    check_close(
        ctx,
        "total",
        dh.total_energy,
        num(r, "/total", ctx),
        TOL_TOTAL,
    );
    assert!((dh.total_energy - (dh.e_ks + dh.e_c_scaled)).abs() < 1e-12);
}

/// Closed shell, production path: `run_wb97x_l_v` defaults (RI-J + RSH DF-K,
/// def2-universal-jkfit) vs the reference's RI-J block.
#[test]
#[ignore = "validation: ωB97X-L-V components"]
fn wb97xlv_h2o_def2svp_production_rij_vs_pyscf() {
    let sys = setup("h2o");
    let (r, ctx) = (&sys.r, sys.ctx.as_str());
    let (dh, ks) = run_wb97x_l_v(
        &ParallelContext::default(),
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        &sys.bounds,
        &ks_config(false),
        &dh_tight(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: run_wb97x_l_v: {e:?}"));
    assert!(ks.converged);
    let ctx = &format!("{ctx} (RI-J)");
    check_close(
        ctx,
        "E_KS (RI-J)",
        dh.e_ks,
        num(r, "/ks_rij/energy", ctx),
        TOL_E_KS,
    );
    // A J-recipe swap (exact J against this RI-J reference) must be visible.
    check_miss(
        ctx,
        "E_KS(RI-J) vs exact-J ref",
        dh.e_ks,
        num(r, "/ks/energy", ctx),
    );
    check_close(
        ctx,
        "λ·E_c own orbitals",
        dh.e_c_scaled,
        num(r, "/corr_rij/e_c_scaled", ctx),
        TOL_E_C_OWN,
    );
    check_close(
        ctx,
        "total",
        dh.total_energy,
        num(r, "/total_rij", ctx),
        TOL_TOTAL,
    );
}

/// Open shell (OH ²Π): ROKS → semicanonical (KS Fock) → unrestricted LinLCCD(hh).
#[test]
#[ignore = "validation: ωB97X-L-V components"]
fn wb97xlv_oh_def2svp_components_vs_pyscf() {
    let sys = setup("oh");
    let (r, ctx) = (&sys.r, sys.ctx.as_str());
    let pctx = ParallelContext::default();

    // ---- E_KS (ROKS; J exact by construction for an RSH functional) ----
    let roks = solve_rohf(
        &pctx,
        &sys.mol,
        &sys.obs,
        Operator::coulomb(),
        &sys.bounds,
        &ks_config(false),
    )
    .unwrap_or_else(|e| panic!("{ctx}: ROKS failed: {e:?}"));
    assert!(roks.converged, "{ctx}: ROKS not converged");
    assert!(matches!(roks.spin, Spin::RestrictedOpen), "{ctx}: not ROKS");
    let spread = num(r, "/ks/accepted_spread", ctx);
    eprintln!("{ctx}: reference ROKS accepted-start spread {spread:.2e} (pi-hole orientation)");
    check_close(
        ctx,
        "E_KS (ROKS)",
        roks.energy,
        num(r, "/ks/energy", ctx),
        TOL_E_KS,
    );
    check_miss(
        ctx,
        "E_KS vs exact ROHF",
        roks.energy,
        num(r, "/rohf_control/energy", ctx),
    );

    // ---- semicanonicalisation against the KS Fock (default grids) ----
    let spec = XcSpec::new(WB97X_L_V_NAME);
    let sc = semicanonicalize(
        &pctx,
        &sys.mol,
        &sys.obs,
        &sys.bounds,
        &roks,
        1e-12,
        Some(&spec),
    )
    .unwrap_or_else(|e| panic!("{ctx}: semicanonicalize: {e:?}"));
    let semi = sc.to_unrestricted_result(&roks);
    let eps_a_ref = vec_at(r, "/ks/semicanonical/mo_energy_alpha", ctx);
    let eps_b_ref = vec_at(r, "/ks/semicanonical/mo_energy_beta", ctx);
    for (spin, got, want) in [
        ("a", &sc.eps_alpha, &eps_a_ref),
        ("b", &sc.eps_beta, &eps_b_ref),
    ] {
        assert_eq!(got.len(), want.len(), "{ctx}: eps_{spin} length");
        let d = got
            .iter()
            .zip(want.iter())
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        eprintln!("{ctx}: semicanonical eps_{spin} max |d| {d:.2e} (tol {TOL_EPS:.0e})");
        assert!(d < TOL_EPS, "{ctx}: semicanonical eps_{spin} |d| {d:.2e}");
    }

    // ---- E_c on INJECTED reference semicanonical orbitals ----
    let mut inj: ScfResult = semi.clone();
    assert!(matches!(inj.spin, Spin::Unrestricted));
    inj.mos_alpha = mat_at(r, "/ks/semicanonical/mo_coeff_alpha", ctx);
    inj.mos_beta = Some(mat_at(r, "/ks/semicanonical/mo_coeff_beta", ctx));
    inj.eps_alpha = eps_a_ref.clone();
    inj.eps_beta = Some(eps_b_ref.clone());
    for c in [&inj.mos_alpha, inj.mos_beta.as_ref().unwrap()] {
        let orth = orthonormality_defect(c, &sys.obs);
        eprintln!("{ctx}: injected CᵀSC − I max {orth:.2e}");
        assert!(orth < 1e-8, "{ctx}: injected orbitals not orthonormal");
    }
    let cc = cc_tight();
    let run = |v: LadderVariant| {
        u_linlccd(&sys.mol, &sys.obs, &sys.dfbs, lambda_erfc(), &inj, &cc, v)
            .unwrap_or_else(|e| panic!("{ctx}: u_linlccd {v:?}: {e:?}"))
            .correlation_energy
    };
    let ref_drv = num(r, "/corr/drivers_only_scaled", ctx);
    let ref_hh = num(r, "/corr/e_c_scaled", ctx);
    let e_drv = run(LadderVariant::DriversOnly);
    check_close(
        ctx,
        "DriversOnly(λ·erfc) injected",
        e_drv,
        ref_drv,
        TOL_E_C_INJECTED,
    );
    let e_hh = run(LadderVariant::Hh);
    check_close(
        ctx,
        "λ·E_c LinLCCD(hh) injected",
        e_hh,
        ref_hh,
        TOL_E_C_INJECTED,
    );
    let dh_inj = u_solve_wb97x_l_v(&sys.mol, &sys.obs, &sys.dfbs, &inj, &dh_tight()).unwrap();
    check_close(
        ctx,
        "u_solve e_c_scaled injected",
        dh_inj.e_c_scaled,
        ref_hh,
        TOL_E_C_INJECTED,
    );
    check_miss(ctx, "Hh vs DriversOnly ref", e_hh, ref_drv);
    check_miss(ctx, "DriversOnly vs hh ref", e_drv, ref_hh);
    check_miss(
        ctx,
        "λ·E_c vs linear-λ (pre-#239)",
        dh_inj.e_c_scaled,
        num(r, "/corr/negative_control_linear_lambda", ctx),
    );

    // ---- E_c and total on ferric's OWN semicanonical orbitals ----
    let dh = u_solve_wb97x_l_v(&sys.mol, &sys.obs, &sys.dfbs, &semi, &dh_tight()).unwrap();
    check_close(
        ctx,
        "λ·E_c own orbitals",
        dh.e_c_scaled,
        ref_hh,
        TOL_E_C_OWN,
    );
    check_close(
        ctx,
        "total",
        dh.total_energy,
        num(r, "/total", ctx),
        TOL_TOTAL,
    );
}
