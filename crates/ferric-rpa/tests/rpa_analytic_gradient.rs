//! Analytic closed-shell RI-dRPA nuclear gradient (`ferric_rpa::gradient_analytic`)
//! against finite differences of the same n-point energy and against the
//! SCS-MP2 gradient in the second-order limit.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --release --test rpa_analytic_gradient
//! ```
//!
//! # What is compared
//!
//! | test | anchor |
//! |---|---|
//! | `*_matches_fd5_of_the_same_quadrature` | g vs the step-converged 5-point FD of E_RHF + E_c (PySCF surface, `testdata/reference/validation/rpa_gradient`) at n = 40 and n = 6 all-electron and n = 40 frozen core |
//! | `second_order_term_equals_the_scs_mp2_gradient` | the second-order summand `-tr Pi^2/2` at n = 200 vs `scs_mp2_gradient_analytical` at (c_os, c_ss) = (2, 0) |
//! | `h2o_matches_fd5_of_ferrics_own_energy` (ignored, ~1 min) | g vs a 5-point FD (h = 1e-3) of ferric's own `run_pdep_rpa` total energy |
//! | `gradient_is_translation_and_rotation_invariant` | sum of forces and of torques |
//! | `truncated_pdep_is_refused` | trunc_thresh > 0 is an error |
//!
//! Systems: distorted H2O and NH3 / cc-pVDZ (cc-pvdz-ri), every Cartesian
//! component nonzero, so no symmetry zero can hide a missing
//! orbital-relaxation term.
//!
//! # Physics vs artifact hypotheses
//!
//! * Right: g equals the FD5 of the SAME n-point energy to the FD truncation
//!   plus SCF noise (bars below), at n = 6 as well as n = 40.
//! * Differentiating the converged integral instead of the n-point sum: the
//!   n = 6 case misses by the 3e-4 Ha/Bohr n = 40 vs n = 6 difference.
//! * Orbital-energy densities P_oo/P_vv dropped (no Fock dependence): misses
//!   by 1e-2 (MUTATION below).
//! * A quadrature weight applied twice: misses (MUTATION below).
//! * A wrong Z-vector/Lagrangian normalization: the second-order anchor
//!   against the independent SCS-MP2 code fails.
//!
//! # TOLERANCES
//!
//! FD reference: the 5-point FD at h = 2e-3 and h = 1e-3 agree to 3.9e-10
//! (H2O) and 7.4e-10 (NH3) Ha/Bohr, so its truncation is below 1e-9. The
//! PySCF and ferric energies agree to 1.7e-12 (E_RHF) and 9.2e-13 (E_c)
//! (validation_rpa_gradient.rs). Both SCFs stop at a density residual near
//! 1e-10; the gradient is first order in that residual through the
//! non-variational E_c, measured in the prototype at <= 2e-9. Bar
//! `TOL_FD = 2e-8`. Measured (release): H2O 2.40e-9 (n = 40), 2.34e-9
//! (n = 6), 2.44e-9 (n = 40, fc = 1); NH3 8.31e-10 (n = 40), 7.54e-10
//! (n = 6); H2O vs ferric's own FD5 2.37e-9. The n = 6 gradient misses the
//! n = 40 FD5 by 3.05e-4, far outside the bar.
//!
//! Second-order anchor: the n = 200 Gauss-Legendre sum of the second-order
//! integrand differs from 1/(e_ia + e_jb) by a quadrature error measured
//! (prototype) at ~1e-10 in the gradient; the two Z-vectors stop at
//! |r| < 1e-8, contributing <= |L| 1e-8 / lambda_min. Bar `TOL_RING = 1e-7`;
//! measured 2.6e-15.
//!
//! Invariance: |sum F| is rounding (`TOL_TRANS = 1e-9`, measured 2.9e-14
//! H2O, 1.6e-14 NH3); |sum R x F| is first order in the SCF residual at
//! density_conv 1e-9 (`TOL_ROT = 1e-8`, measured 1.6e-9 H2O, 1.9e-9 NH3).
//!
//! # MUTATIONS
//!
//! Each applied to `rpa_ri_densities` in `src/gradient_analytic.rs`, the
//! file restored afterwards (sha256 identical):
//!
//! | mutation | failing assertions (miss, Ha/Bohr) |
//! |---|---|
//! | P_oo, P_vv set to zero (orbital-energy term dropped) | H2O FD5 3.49e-2, NH3 FD5 2.62e-2, SCS-MP2 6.68e-2, torque 6.9e-5 |
//! | quadrature weight applied twice in Γ, P | H2O FD5 5.65e-5, NH3 FD5 4.77e-5, SCS-MP2 4.40e-6 |
//!
//! The invariance test passes under the doubled weight, as it must: the
//! doubled weight is the exact gradient of a different, still invariant, energy.

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::gradient::scs_mp2_gradient_analytical;
use ferric_mp2::scs::ScsMp2Config;
use ferric_rpa::config::{PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_rpa::gradient_analytic::{rpa_analytic_gradient, rpa_analytic_gradient_on, RingOrder};
use ferric_rpa::run_pdep_rpa;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use serde_json::Value;

const BASIS: &str = "cc-pvdz";
const AUX: &str = "cc-pvdz-ri";
const ROW_DIR: &str = "testdata/reference/validation/rpa_gradient";
const MOL_DIR: &str = "testdata/molecules/validation";
const TOL_FD: f64 = 2e-8;
const TOL_RING: f64 = 1e-7;
const TOL_TRANS: f64 = 1e-9;
const TOL_ROT: f64 = 1e-8;

fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| p.join("Cargo.toml").is_file() && p.join("testdata").is_dir();
    let mut here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while !looks_like_root(&here) {
        assert!(here.pop(), "workspace root not found");
    }
    here
}

fn molecule(system: &str) -> Molecule {
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1).unwrap()
}

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing reference {} ({e})", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn fd5(r: &Value, key: &str, natm: usize) -> Array2<f64> {
    let rows = r
        .pointer(&format!("/gradient/fd/fd5_h0.002/{key}"))
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("reference gradient {key} missing"));
    assert_eq!(rows.len(), natm);
    let mut g = Array2::zeros((natm, 3));
    for (a, row) in rows.iter().enumerate() {
        for k in 0..3 {
            g[(a, k)] = row[k].as_f64().unwrap();
        }
    }
    g
}

fn rpa_cfg(n_quad: usize, frozen_core: usize) -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core,
        trunc_thresh: 0.0,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: n_quad,
            u0: 0.5,
        },
        ..Default::default()
    }
}

fn bases() -> (BasisSet, BasisSet) {
    (basis::bundled(BASIS).unwrap(), basis::bundled(AUX).unwrap())
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn analytic(mol: &Molecule, cfg: &PdepRpaConfig) -> Array2<f64> {
    let (obs, aux) = bases();
    rpa_analytic_gradient(mol, &obs, &aux, Operator::coulomb(), cfg)
        .unwrap()
        .gradient
}

fn fd_case(system: &str) {
    let mol = molecule(system);
    let r = reference(system);
    let natm = mol.atoms.len();
    let g40 = analytic(&mol, &rpa_cfg(40, 0));
    let g6 = analytic(&mol, &rpa_cfg(6, 0));
    let mut cases = vec![("n=40", g40, "total"), ("n=6", g6.clone(), "total_coarse")];
    // Only the H2O reference carries a frozen-core block.
    if r.pointer("/gradient/fd/fd5_h0.002/total_fc").is_some() {
        cases.push(("n=40 fc=1", analytic(&mol, &rpa_cfg(40, 1)), "total_fc"));
    }
    for (what, g, key) in &cases {
        let d = max_abs_diff(g, &fd5(&r, key, natm));
        eprintln!("{system} {what}: |g - FD5|_max {d:.2e} (bar {TOL_FD:.0e})");
        assert!(d < TOL_FD, "{system} {what}: |g - FD5| {d:e} > {TOL_FD:e}");
    }
    // The 6-point gradient is a different function's derivative: it must
    // miss the 40-point FD (3e-4 apart) by far more than the bar.
    let miss = max_abs_diff(&g6, &fd5(&r, "total", natm));
    assert!(
        miss > 1e3 * TOL_FD,
        "n=6 gradient does not see its own grid: {miss:e}"
    );
}

#[test]
fn h2o_matches_fd5_of_the_same_quadrature() {
    fd_case("h2o_distorted");
}

#[test]
fn nh3_matches_fd5_of_the_same_quadrature() {
    fd_case("nh3_distorted");
}

/// The second-order summand at n = 200 against the SCS-MP2 gradient at
/// (c_os, c_ss) = (2, 0): the direct-ring MP2 energy 2 E_OS.
#[test]
fn second_order_term_equals_the_scs_mp2_gradient() {
    let mol = molecule("h2o_distorted");
    let (obs_bs, aux_bs) = bases();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf_cfg = RhfConfig::default().with_density_conv(1e-10);
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &rhf_cfg).unwrap();
    let ring = rpa_analytic_gradient_on(
        &mol,
        &obs,
        &dfbs,
        op,
        &bounds,
        &rhf,
        &rpa_cfg(200, 0),
        RingOrder::SecondOrder,
    )
    .unwrap();
    let scs = ScsMp2Config {
        c_os: 2.0,
        c_ss: 0.0,
        ..Default::default()
    };
    let g_scs =
        scs_mp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &scs, None).unwrap();
    let d = max_abs_diff(&ring.gradient, &g_scs);
    eprintln!("second order (n=200) vs SCS-MP2(2, 0): |dg|_max {d:.2e} (bar {TOL_RING:.0e})");
    assert!(
        d < TOL_RING,
        "second-order RPA vs SCS-MP2(2,0) gradient: {d:e}"
    );
    // The full dRPA gradient is a different energy: it must miss.
    let full = rpa_analytic_gradient_on(
        &mol,
        &obs,
        &dfbs,
        op,
        &bounds,
        &rhf,
        &rpa_cfg(40, 0),
        RingOrder::Full,
    )
    .unwrap();
    let miss = max_abs_diff(&full.gradient, &g_scs);
    assert!(
        miss > 1e2 * TOL_RING,
        "full dRPA equals the second-order term: {miss:e}"
    );
}

#[test]
fn gradient_is_translation_and_rotation_invariant() {
    for system in ["h2o_distorted", "nh3_distorted"] {
        let mol = molecule(system);
        let g = analytic(&mol, &rpa_cfg(20, 0));
        let mut force = [0.0f64; 3];
        let mut torque = [0.0f64; 3];
        for (a, atom) in mol.atoms.iter().enumerate() {
            let (r, f) = (
                [atom.x, atom.y, atom.zpos],
                [g[(a, 0)], g[(a, 1)], g[(a, 2)]],
            );
            for k in 0..3 {
                force[k] += f[k];
                torque[k] += r[(k + 1) % 3] * f[(k + 2) % 3] - r[(k + 2) % 3] * f[(k + 1) % 3];
            }
        }
        let (ft, tq) = (
            force.iter().fold(0.0f64, |m, v| m.max(v.abs())),
            torque.iter().fold(0.0f64, |m, v| m.max(v.abs())),
        );
        eprintln!("{system}: |sum F| {ft:.2e}, |sum R x F| {tq:.2e}");
        assert!(ft < TOL_TRANS, "{system}: translation {ft:e}");
        assert!(tq < TOL_ROT, "{system}: rotation {tq:e}");
    }
}

#[test]
fn truncated_pdep_is_refused() {
    let mol = molecule("h2o_distorted");
    let (obs, aux) = bases();
    let cfg = PdepRpaConfig {
        trunc_thresh: 1e-4,
        ..rpa_cfg(20, 0)
    };
    let err = rpa_analytic_gradient(&mol, &obs, &aux, Operator::coulomb(), &cfg).unwrap_err();
    assert!(format!("{err}").contains("trunc_thresh"), "{err}");
}

/// 5-point FD (h = 1e-3 Bohr) of ferric's own E_RHF + E_c (run_pdep_rpa,
/// full rank, n = 40), no external code.
#[test]
#[ignore = "36 RHF + RPA energies, ~1 min in release"]
fn h2o_matches_fd5_of_ferrics_own_energy() {
    let mol = molecule("h2o_distorted");
    let cfg = rpa_cfg(40, 0);
    let (obs_bs, aux_bs) = bases();
    let op = Operator::coulomb();
    let energy = |m: &Molecule| {
        let ctx = ParallelContext::default();
        let obs = PreparedBasis::new(m, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(m, &aux_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf_cfg = RhfConfig::default().with_density_conv(1e-10);
        let rhf = solve_rhf(&ctx, m, &obs, op, &bounds, &rhf_cfg).unwrap();
        rhf.energy + run_pdep_rpa(m, &obs, &dfbs, op, &rhf, &cfg).unwrap().e_rpa
    };
    let h = 1e-3;
    let mut fd = Array2::<f64>::zeros((mol.atoms.len(), 3));
    for a in 0..mol.atoms.len() {
        for k in 0..3 {
            let at = |s: f64| {
                let mut m = mol.clone();
                match k {
                    0 => m.atoms[a].x += s * h,
                    1 => m.atoms[a].y += s * h,
                    _ => m.atoms[a].zpos += s * h,
                }
                energy(&m)
            };
            fd[(a, k)] = (at(-2.0) - 8.0 * at(-1.0) + 8.0 * at(1.0) - at(2.0)) / (12.0 * h);
        }
    }
    let d = max_abs_diff(&analytic(&mol, &cfg), &fd);
    eprintln!("H2O n=40: |g - FD5(own)|_max {d:.2e} (bar {TOL_FD:.0e})");
    assert!(d < TOL_FD, "own FD5: {d:e}");
}

/// One timing pair on n-hexane / cc-pVDZ (all-anti, testdata/molecules/alkane_6.xyz):
/// the analytic gradient against the 3-point FD route of `[rpa] gradient =
/// "fd"` (h = 5e-4, 120 RHF + RPA energies), both at n = 20, full rank. The
/// gradients must agree within the FD route's truncation (bar 1e-6 Ha/Bohr,
/// 20x its 5.6e-8 on H2O/NH3 to allow for the larger molecule).
/// Measured once (release, OPENBLAS_NUM_THREADS=1, box load ~4-5): analytic
/// 38.4 s, FD 1556.6 s, |g_a - g_fd|_max 2.25e-8.
#[test]
#[ignore = "timing: ~120 RHF + RPA energies on hexane / cc-pVDZ"]
fn hexane_analytic_vs_fd_timing() {
    let xyz = workspace_root().join("testdata/molecules/alkane_6.xyz");
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1).unwrap();
    let (obs, aux) = bases();
    let op = Operator::coulomb();
    let cfg = rpa_cfg(20, 0);
    let t0 = std::time::Instant::now();
    let a = rpa_analytic_gradient(&mol, &obs, &aux, op, &cfg).unwrap();
    let t_analytic = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let (e_fd, g_fd) =
        ferric_rpa::gradient::total_rpa_gradient(&mol, &obs, &aux, op, &cfg, 5e-4).unwrap();
    let t_fd = t1.elapsed().as_secs_f64();
    let d = max_abs_diff(&a.gradient, &g_fd);
    eprintln!(
        "hexane/cc-pVDZ n=20: analytic {t_analytic:.1} s, FD {t_fd:.1} s ({:.1}x); \
         |dE| {:.2e}, |g_a - g_fd|_max {d:.2e}",
        t_fd / t_analytic,
        (a.energy - e_fd).abs()
    );
    assert!(d < 1e-6, "hexane analytic vs FD: {d:e}");
}
