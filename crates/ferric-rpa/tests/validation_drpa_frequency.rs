//! VALIDATION tier — the exact dRPA route of `method.kind = "drpa"`.
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-rpa --test validation_drpa_frequency \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! The CLI's exact `drpa` runs `ferric_rpa::exact_drpa_config` through
//! `ferric_rpa::run_pdep_rpa_energy`: the frequency-integrated
//! `ln det(1 + Π) − tr Π` in the dressed aux basis, nothing truncated, no
//! static eigensolve (the aux-basis log-det route). It is compared to the
//! independent numpy plasmon dRPA on PySCF density-fitted integrals that
//! validates the Riccati route (`ferric-mp2/tests/validation_drpa_amplitude.rs`,
//! `scripts/validation/gen_drpa_amplitude.py`, references in
//! `testdata/reference/validation/drpa_amplitude/`): H2/STO-3G, water and
//! n-butane at 6-31G and cc-pVDZ, n-octane at 6-31G, every frozen-core
//! setting each reference carries. The RHF is exact four-centre J/K on both
//! sides, converged as in that file.
//!
//! # Exactness anchor (asserted first)
//!
//! E_nuc, the AO/aux counts and E_RHF against the reference, then the route
//! itself: the run must report `AuxBasisLogDet { trunc_thresh: 0 }`.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If the route is right: the energy lands on the plasmon reference within
//!   [`TOL_E_CORR`] (the SCF and integral floor between the codes) plus the
//!   route's own [`rounding_bar`], once the quadrature has converged.
//! * If `tr Π` were dropped, a spin factor were wrong, or the frozen core were
//!   off by one: misses of 1e-3 to 1e-1 Eh (frozen-core gaps ≥ 1.4e-3,
//!   ring-dropped gaps ≥ 2.7e-3 Eh), checked as negative controls.
//! * If the quadrature were too coarse: the convergence table shows it.
//!
//! # Quadrature order (convergence table)
//!
//! `drpa_frequency_quadrature_convergence_table` prints the error against the
//! reference at n = 24 … 200 Gauss-Legendre points (u0 = 0.5) for every system
//! and frozen core, with each order's rounding bar, and asserts the chosen
//! order. MEASURED (2026-11-14): with every orbital correlated 32 points miss
//! by up to 1.3e-8 Eh (water/cc-pVDZ) and 4.7e-9 (n-butane/cc-pVDZ), 40 by up
//! to 9.4e-10; at 48 every row is within 2.7e-10 (n-octane/6-31G, bar
//! 4.1e-10; every other row ≤ 9.8e-11), and from 64 points on the rounding
//! grows past the quadrature error (up to 4.0e-9 at 200). Hence
//! `EXACT_DRPA_QUAD_POINTS = 48`.
//!
//! # Riccati, plasmon and frequency routes
//!
//! `three_routes_agree_*` runs the drCCD Riccati solve (ε = 0), the in-crate
//! canonical plasmon formula and the frequency route on one SCF, for water
//! and n-hexane at cc-pVDZ, and holds each pair to the Riccati residual bar
//! plus the quadrature bar (derivations on [`riccati_residual_bar`] and
//! [`quadrature_bar`]). MEASURED (2026-11-14): water (fc 1) Riccati−plasmon
//! 4.5e-14 (bar 1.4e-12), frequency−plasmon 3.4e-12 (bar 9.0e-11); n-hexane
//! (fc 6) 6.4e-13 (bar 2.0e-12) and 4.2e-11 (bar 6.5e-10).
//!
//! # Mutations (applied to `src/`, run, restored; `git diff` clean)
//!
//! * `energy.rs::dielectric_trace_log_summand` drops `tr Π` (returns
//!   `log_det`) → every system FAILS against the reference by > 1e-1 Eh.
//! * `energy_route.rs::exact_drpa_config` with `frozen_core + 1` → FAILS by
//!   the frozen-core gap.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::drpa_amplitude::{amplitude_drpa, canonical_plasmon_drpa, AmplitudeDrpaConfig};
use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};
use ferric_rpa::{
    exact_drpa_config, run_pdep_rpa, run_pdep_rpa_energy, PdepRpaConfig, RpaEnergyRoute,
    EXACT_DRPA_QUAD_POINTS,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/drpa_amplitude";
const MOL_DIR: &str = "testdata/molecules/validation";

/// Geometry check (PySCF E_nuc from ferric's Bohr geometry).
const TOL_ENUC: f64 = 1e-9;
/// E_RHF vs PySCF, exact J/K on both sides (the Riccati file's bar).
const TOL_E_RHF: f64 = 5e-9;
/// dRPA E_c vs the numpy plasmon reference, before the frequency route's own
/// rounding bar ([`rounding_bar`]) is added: the Riccati file's bar, which
/// covers the SCF and integral differences between the two codes (that route
/// measured ≤ 3.0e-11 against these references).
const TOL_E_CORR: f64 = 3e-10;
/// A reference the route must MISS (the Riccati file's negative-control bar).
const MUST_MISS: f64 = 1e-6;
/// The quadrature orders the convergence table walks.
const TABLE_POINTS: [usize; 10] = [24, 32, 40, 48, 56, 64, 80, 96, 128, 200];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing reference {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn num(v: &Value, ptr: &str, ctx: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{ctx}: reference lacks {ptr}"))
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    assert!(
        d <= tol,
        "{ctx}: {what}: ferric {got:.13} vs reference {want:.13} (|d| {d:.2e}) exceeds {tol:.1e}"
    );
}

fn check_miss(ctx: &str, what: &str, got: f64, want: f64) {
    let d = (got - want).abs();
    assert!(
        d > MUST_MISS,
        "{ctx}: {what}: ferric {got:.13} sits within {MUST_MISS:.0e} of a value it must miss \
         ({want:.13})"
    );
}

struct Setup {
    ctx: String,
    mol: Molecule,
    obs_bs: basis::BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

/// Exact-J/K RHF, converged one decade past the κ-MP2 harness (dRPA is first
/// order in a Fock/orbital inconsistency), as the Riccati validation does.
fn exact_rhf(mol: &Molecule, obs: &PreparedBasis, ctx: &str) -> ScfResult {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, obs).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-10,
        max_iter: 200,
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), mol, obs, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(rhf.converged, "{ctx}: RHF not converged");
    rhf
}

fn load(mol_path: &Path, basis_name: &str, aux_name: &str, ctx: &str) -> Setup {
    let mol = Molecule::load_xyz(mol_path.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: {e:?}", mol_path.display()));
    let obs_bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let rhf = exact_rhf(&mol, &obs, ctx);
    Setup {
        ctx: ctx.to_string(),
        mol,
        obs_bs,
        obs,
        dfbs,
        rhf,
    }
}

/// A reference system: geometry, basis and RHF anchors first.
fn reference_setup(system: &str, basis_name: &str, aux_name: &str) -> (Setup, Value) {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    assert_eq!(r["basis"].as_str(), Some(basis_name), "{ctx}: basis");
    assert_eq!(r["aux_basis"].as_str(), Some(aux_name), "{ctx}: aux basis");
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let su = load(&xyz, basis_name, aux_name, &ctx);
    check_close(
        &ctx,
        "E_nuc",
        su.mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    assert_eq!(su.obs.nbasis() as u64, r["nao"].as_u64().unwrap(), "{ctx}");
    assert_eq!(
        su.dfbs.nbasis() as u64,
        r["naux"].as_u64().unwrap(),
        "{ctx}"
    );
    check_close(
        &ctx,
        "E_RHF",
        su.rhf.energy,
        num(&r, "/e_rhf", &ctx),
        TOL_E_RHF,
    );
    (su, r)
}

/// (frozen_core, e_corr, e_corr_ring_dropped), ascending frozen core.
fn frozen_settings(r: &Value, ctx: &str) -> Vec<(usize, f64, f64)> {
    let blocks = r["frozen_core"]
        .as_object()
        .unwrap_or_else(|| panic!("{ctx}: reference has no frozen_core block"));
    let mut out: Vec<(usize, f64, f64)> = blocks
        .iter()
        .map(|(k, b)| {
            (
                k.parse().expect("frozen-core key"),
                num(b, "/e_corr", ctx),
                num(b, "/e_corr_ring_dropped", ctx),
            )
        })
        .collect();
    out.sort_by_key(|t| t.0);
    out
}

fn exact_cfg_at(frozen_core: usize, n_points: usize) -> PdepRpaConfig {
    let mut cfg = exact_drpa_config(frozen_core, None);
    cfg.quadrature.n_points = n_points;
    cfg
}

/// E_c on the CLI's exact route at `n_points` (the route asserted).
fn frequency_route(su: &Setup, frozen_core: usize, n_points: usize) -> f64 {
    let cfg = exact_cfg_at(frozen_core, n_points);
    let r = run_pdep_rpa_energy(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
    )
    .unwrap_or_else(|e| panic!("{}: frequency route failed: {e:?}", su.ctx));
    assert_eq!(r.route, RpaEnergyRoute::AuxBasisLogDet, "{}", su.ctx);
    r.e_rpa
}

/// The rounding bar of the frequency route at `n_points`, by the random-sign
/// model of `tests/rpa_energy_route.rs` for ONE route: per frequency
/// `σ_k = (2·M + √(M·N))·u·λ_max(iω_k)` (the trace, the LU log-det and the
/// assembly of `ε̃` from `N = nocc·nvir` pairs), summed in absolute value with
/// the weights, `(1/2π) Σ_k w_k σ_k`. `λ_max(iω_k)` comes from the full-rank
/// static route with the per-frequency eigenvalues kept. The weights of the
/// mapped Gauss-Legendre rule grow without bound towards `ω → ∞`, where every
/// summand still carries `O(M·u)` absolute rounding, so this bar grows with
/// the number of points: the reason more points stop helping.
fn rounding_bar(su: &Setup, frozen_core: usize, n_points: usize) -> f64 {
    const U: f64 = f64::EPSILON / 2.0;
    let cfg = PdepRpaConfig {
        need_eigenvalues_freq: true,
        ..exact_cfg_at(frozen_core, n_points)
    };
    let r = run_pdep_rpa(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
    )
    .unwrap_or_else(|e| panic!("{}: static route failed: {e:?}", su.ctx));
    let m = r.eigenvalues_freq.ncols() as f64;
    let nocc_total = (su.mol.nelec() as usize) / 2;
    let n_pairs = ((nocc_total - frozen_core) * (su.rhf.eps_r().len() - nocc_total)) as f64;
    let width = 2.0 * m + (m * n_pairs).sqrt();
    let sum: f64 = r
        .eigenvalues_freq
        .outer_iter()
        .zip(&r.quad_weights)
        .map(|(row, w)| w * width * U * row.iter().copied().fold(1.0, f64::max))
        .sum();
    sum / (2.0 * std::f64::consts::PI)
}

/// Every frozen-core setting on the reference, plus the negative controls.
fn check_against_reference(system: &str, basis_name: &str, aux_name: &str) {
    let (su, r) = reference_setup(system, basis_name, aux_name);
    let settings = frozen_settings(&r, &su.ctx);
    for &(fc, e_ref, e_ring) in &settings {
        let ctx = format!("{}/fc={fc}", su.ctx);
        let e = frequency_route(&su, fc, EXACT_DRPA_QUAD_POINTS);
        let bar = TOL_E_CORR + rounding_bar(&su, fc, EXACT_DRPA_QUAD_POINTS);
        println!(
            "{ctx}: E_c {e:.13} vs reference {e_ref:.13}: |d| {:.2e} (bar {bar:.2e})",
            (e - e_ref).abs()
        );
        check_close(&ctx, "frequency route", e, e_ref, bar);
        check_miss(&ctx, "ring-dropped value", e, e_ring);
        for &(other, e_other, _) in &settings {
            if other != fc {
                check_miss(&ctx, &format!("fc={other} reference"), e, e_other);
            }
        }
    }
}

#[test]
#[ignore = "validation: Exact dRPA (frequency route)"]
fn drpa_frequency_h2_sto3g_vs_numpy_plasmon() {
    check_against_reference("h2", "sto-3g", "sto-3g");
}

#[test]
#[ignore = "validation: Exact dRPA (frequency route)"]
fn drpa_frequency_h2o_631g_vs_numpy_plasmon() {
    check_against_reference("h2o", "6-31g", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: Exact dRPA (frequency route)"]
fn drpa_frequency_h2o_ccpvdz_vs_numpy_plasmon() {
    check_against_reference("h2o", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: Exact dRPA (frequency route)"]
fn drpa_frequency_alkane_4_631g_vs_numpy_plasmon() {
    check_against_reference("alkane_4", "6-31g", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: Exact dRPA (frequency route)"]
fn drpa_frequency_alkane_4_ccpvdz_vs_numpy_plasmon() {
    check_against_reference("alkane_4", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: Exact dRPA (frequency route)"]
fn drpa_frequency_alkane_8_631g_vs_numpy_plasmon() {
    check_against_reference("alkane_8", "6-31g", "cc-pvdz-ri");
}

/// The quadrature order: error against the reference at each order in
/// [`TABLE_POINTS`], every system and frozen core, with the rounding bar of
/// each order on the next line. Asserted: at `EXACT_DRPA_QUAD_POINTS` the
/// error is within [`TOL_E_CORR`] plus that order's rounding bar.
#[test]
#[ignore = "validation: Exact dRPA (frequency route)"]
fn drpa_frequency_quadrature_convergence_table() {
    let systems = [
        ("h2", "sto-3g", "sto-3g"),
        ("h2o", "6-31g", "cc-pvdz-ri"),
        ("h2o", "cc-pvdz", "cc-pvdz-ri"),
        ("alkane_4", "6-31g", "cc-pvdz-ri"),
        ("alkane_4", "cc-pvdz", "cc-pvdz-ri"),
        ("alkane_8", "6-31g", "cc-pvdz-ri"),
    ];
    let header: Vec<String> = TABLE_POINTS.iter().map(|n| format!("{n:>9}")).collect();
    println!("{:<26}{}", "system/fc  error at n =", header.join(""));
    let mut misses = Vec::new();
    for (system, basis_name, aux_name) in systems {
        let (su, r) = reference_setup(system, basis_name, aux_name);
        for (fc, e_ref, _) in frozen_settings(&r, &su.ctx) {
            let errs: Vec<f64> = TABLE_POINTS
                .iter()
                .map(|&n| frequency_route(&su, fc, n) - e_ref)
                .collect();
            let bars: Vec<f64> = TABLE_POINTS
                .iter()
                .map(|&n| rounding_bar(&su, fc, n))
                .collect();
            let cells = |v: &[f64]| -> String {
                v.iter()
                    .map(|e| format!("{e:>9.1e}"))
                    .collect::<Vec<_>>()
                    .join("")
            };
            println!("{:<26}{}", format!("{}/fc={fc}", su.ctx), cells(&errs));
            println!("{:<26}{}", "  rounding bar", cells(&bars));
            for ((&n, &e), &b) in TABLE_POINTS.iter().zip(&errs).zip(&bars) {
                if n == EXACT_DRPA_QUAD_POINTS && e.abs() > TOL_E_CORR + b {
                    misses.push(format!("{}/fc={fc}: n = {n}: error {e:.2e}", su.ctx));
                }
            }
        }
    }
    assert!(misses.is_empty(), "above {TOL_E_CORR:.1e}: {misses:#?}");
}

/// Bound on the drCCD Riccati energy's distance from the exact root, from its
/// final relative residual `relres = ‖R‖_F / ‖B‖_F`.
///
/// Linearizing `R(T) = 0` at the returned `T̃`, `δT` solves a Sylvester
/// equation whose operator is `(A + BT̃) ⊗ I + I ⊗ (A + T̃B)`, with smallest
/// eigenvalue `≥ 2 Ω_min ≥ 2 Δ_min` (Ω are the RPA excitation energies, all
/// `≥ Δ_min`, the smallest orbital gap, since `(A − B)(A + B) ⪰ D²`). So
/// `‖δT‖_F ≤ ‖R‖_F / (2 Δ_min)` and `|δE| = ½|⟨B, δT⟩| ≤ relres·‖B‖_F² /
/// (4 Δ_min)`, with `B = 2(ia|jb)`: `‖B‖_F² = 4 ‖G‖_F²`, `G = b_ovᵀ b_ov`,
/// `‖G‖_F = ‖b_ov b_ovᵀ‖_F` (invariant under the localizing rotations the
/// Riccati route works in).
fn riccati_residual_bar(su: &Setup, frozen_core: usize, relres: f64) -> f64 {
    let op = Operator::coulomb();
    let cfg = RiMp2Config {
        frozen_core,
        ..Default::default()
    };
    let inter = compute_rpa_intermediates(&su.mol, &su.obs, &su.dfbs, op, &su.rhf, &cfg).unwrap();
    let gram = inter.b_ov.dot(&inter.b_ov.t());
    let g_f2: f64 = gram.iter().map(|x| x * x).sum();
    let eps = su.rhf.eps_r();
    let homo = eps[inter.nocc_total - 1];
    let lumo = eps[inter.nocc_total];
    relres * g_f2 / (lumo - homo)
}

/// The frequency route's error bar at `EXACT_DRPA_QUAD_POINTS`: the
/// quadrature error, estimated as the distance to the same route at twice the
/// points, plus the [`rounding_bar`] of both orders (the estimate carries the
/// rounding of each, and the value under test the rounding of the first).
fn quadrature_bar(su: &Setup, frozen_core: usize, e_n: f64) -> f64 {
    let n = EXACT_DRPA_QUAD_POINTS;
    let e_2n = frequency_route(su, frozen_core, 2 * n);
    (e_n - e_2n).abs()
        + 2.0 * rounding_bar(su, frozen_core, n)
        + rounding_bar(su, frozen_core, 2 * n)
}

fn three_routes_agree(su: &Setup, frozen_core: usize) {
    let op = Operator::coulomb();
    let ctx = format!("{}/fc={frozen_core}", su.ctx);
    let ric_cfg = AmplitudeDrpaConfig {
        eps: 0.0,
        frozen_core,
        diis: Some(8),
        eps_rtol_factor: Some(0.1),
        ..Default::default()
    };
    let ric = amplitude_drpa(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &ric_cfg,
    )
    .unwrap_or_else(|e| panic!("{ctx}: Riccati failed: {e:?}"));
    assert!(ric.converged, "{ctx}");
    let plasmon = canonical_plasmon_drpa(&su.mol, &su.obs, &su.dfbs, op, &su.rhf, &ric_cfg)
        .unwrap_or_else(|e| panic!("{ctx}: plasmon failed: {e:?}"));
    let freq = frequency_route(su, frozen_core, EXACT_DRPA_QUAD_POINTS);
    let b_ric = riccati_residual_bar(su, frozen_core, ric.relres);
    let b_quad = quadrature_bar(su, frozen_core, freq);
    let pairs = [
        ("Riccati vs plasmon", ric.e_corr, plasmon, b_ric + 1e-12),
        ("frequency vs plasmon", freq, plasmon, b_quad),
        ("frequency vs Riccati", freq, ric.e_corr, b_quad + b_ric),
    ];
    println!(
        "{ctx}: Riccati {:.13} (relres {:.1e}, bar {b_ric:.1e}), plasmon {plasmon:.13}, \
         frequency {freq:.13} (quadrature bar {b_quad:.1e})",
        ric.e_corr, ric.relres
    );
    for (what, a, b, bar) in pairs {
        let d = (a - b).abs();
        println!("  {what}: |d| {d:.2e} (bar {bar:.2e})");
        assert!(d <= bar, "{ctx}: {what}: |d| {d:.2e} > bar {bar:.2e}");
    }
}

#[test]
#[ignore = "validation: Exact dRPA (frequency route)"]
fn three_routes_agree_water_ccpvdz() {
    let xyz = workspace_root().join(MOL_DIR).join("h2o.xyz");
    let su = load(&xyz, "cc-pvdz", "cc-pvdz-ri", "h2o/cc-pvdz");
    three_routes_agree(&su, 1);
}

#[test]
#[ignore = "validation: Exact dRPA (frequency route)"]
fn three_routes_agree_hexane_ccpvdz() {
    let xyz = workspace_root().join("testdata/molecules/alkane_6.xyz");
    let su = load(&xyz, "cc-pvdz", "cc-pvdz-ri", "n-hexane/cc-pvdz");
    three_routes_agree(&su, 6);
}
