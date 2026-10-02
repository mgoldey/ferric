//! Harmonic frequencies on a dispersion-corrected surface
//! (`ferric_scf::frequencies::harmonic_frequencies_with_scf_correction`).
//!
//! The driver central-differences the CORRECTED analytic gradient
//! `g_KS + g_disp`, each from the SCF converged at that displaced geometry, so
//! the dispersion part of the Hessian is
//!
//! ```text
//! H_disp = H(KS + disp) - H(KS)
//! ```
//!
//! exactly (the KS SCFs at each displaced geometry are the same in both runs).
//! These tests check that difference against constructions that never touch
//! the dispersion GRADIENT code:
//!
//! * `driver_adds_the_correction_hessian_exactly` (fast) -- a synthetic
//!   harmonic-spring correction with a closed-form Hessian, on H2/STO-3G HF.
//!   Tests the driver's plumbing (sign, factor, which geometry the correction
//!   is evaluated at) to near machine precision.
//! * `correction_free_closure_is_bit_identical_to_plain_fd` (fast) -- the
//!   exactness anchor: with `(0.0, None)` the result IS `harmonic_frequencies`.
//! * `d3_hessian_matches_second_differences_of_the_d3_energy` (fast) --
//!   H_disp(D3) on PBE/STO-3G water against 4-point second differences of the
//!   D3(BJ) ENERGY (geometry only, no gradient code).
//! * `mbd_hessian_matches_second_differences_of_the_full_pipeline_energy`
//!   (ignored, slow) -- H_disp(MBD@rsSCS) along fixed directions against
//!   `(E(+hv) - 2E(0) + E(-hv))/h^2` of the MBD energy with the SCF re-solved
//!   at every displaced geometry (density dependence included).
//! * `measure_water_frequency_shifts` (ignored) -- the D3 / MBD shifts in
//!   cm^-1, symmetry and trans/rot diagnostics.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli --release --test dispersion_frequencies
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli --release --test dispersion_frequencies \
//!   -- --ignored --nocapture
//! ```

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{
    mbd_rsscs_for_density, mbd_rsscs_for_scf, MbdFreeAtomCache, MbdRsscsConfig,
};
use ferric_scf::frequencies::{
    harmonic_frequencies, harmonic_frequencies_with_scf_correction, FrequencyConfig,
    FrequencyResult, HessianMethod,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::Array2;

/// Near-equilibrium water (Angstrom).
const H2O_XYZ: &str = "3\nwater\nO 0.000000 0.000000 0.117300\nH 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n";
const H2_XYZ: &str = "2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n";

fn rks_pbe() -> RhfConfig {
    RhfConfig {
        xc: Some("PBE".to_string()),
        max_iter: 300,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    }
}

fn tight_hf() -> RhfConfig {
    RhfConfig {
        max_iter: 200,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    }
}

fn fd_config() -> FrequencyConfig {
    FrequencyConfig {
        hessian: HessianMethod::FiniteDifference,
        ..Default::default()
    }
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn to_array(rows: &[[f64; 3]]) -> Array2<f64> {
    let mut g = Array2::<f64>::zeros((rows.len(), 3));
    for (k, row) in rows.iter().enumerate() {
        for a in 0..3 {
            g[[k, a]] = row[a];
        }
    }
    g
}

fn shifted(mol: &Molecule, dir: &[f64], s: f64) -> Molecule {
    let mut m = mol.clone();
    for (i, at) in m.atoms.iter_mut().enumerate() {
        at.x += s * dir[3 * i];
        at.y += s * dir[3 * i + 1];
        at.zpos += s * dir[3 * i + 2];
    }
    m
}

fn unit(n: usize, i: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    v[i] = 1.0;
    v
}

/// 4-point central second differences of a pure function of geometry,
/// `H_ab = [E(+a+b) - E(+a-b) - E(-a+b) + E(-a-b)] / (4h^2)` off the diagonal
/// and `[E(+a) - 2E(0) + E(-a)]/h^2` on it.
fn energy_second_differences(mol: &Molecule, h: f64, e: impl Fn(&Molecule) -> f64) -> Array2<f64> {
    let n = 3 * mol.atoms.len();
    let e0 = e(mol);
    let mut hess = Array2::<f64>::zeros((n, n));
    for a in 0..n {
        let ua = unit(n, a);
        hess[(a, a)] = (e(&shifted(mol, &ua, h)) - 2.0 * e0 + e(&shifted(mol, &ua, -h))) / (h * h);
        for b in (a + 1)..n {
            let ub = unit(n, b);
            let at = |sa: f64, sb: f64| e(&shifted(&shifted(mol, &ua, sa * h), &ub, sb * h));
            let v = (at(1.0, 1.0) - at(1.0, -1.0) - at(-1.0, 1.0) + at(-1.0, -1.0)) / (4.0 * h * h);
            hess[(a, b)] = v;
            hess[(b, a)] = v;
        }
    }
    hess
}

fn scf(mol: &Molecule, bs: &BasisSet, cfg: &RhfConfig) -> ScfResult {
    let prep = ferric_integrals::basis_bridge::PreparedBasis::new(mol, bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let r = solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, cfg).expect("scf");
    assert!(r.converged, "SCF did not converge");
    r
}

/// D3(BJ)/PBE as a frequency-driver correction closure.
fn d3_correction(
) -> impl FnMut(&Molecule, &ScfResult) -> Result<(f64, Option<Array2<f64>>), FerricError> {
    let params = ferric_d3::d3bj_params_for_functional("pbe").expect("d3 params");
    move |m, _scf| {
        let e = ferric_d3::d3bj_energy_for_molecule(m, &params)?;
        let g = to_array(&ferric_d3::d3bj_gradient_for_molecule(m, &params)?);
        Ok((e, Some(g)))
    }
}

/// Synthetic correction: a harmonic spring `E = k/2 (r - r0)^2` between atoms
/// 0 and 1, whose Hessian is closed-form.
const SPRING_K: f64 = 0.3;
const SPRING_R0: f64 = 1.2;

fn spring(m: &Molecule) -> (f64, Array2<f64>) {
    let (a, b) = (&m.atoms[0], &m.atoms[1]);
    let d = [b.x - a.x, b.y - a.y, b.zpos - a.zpos];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let e = 0.5 * SPRING_K * (r - SPRING_R0).powi(2);
    let mut g = Array2::<f64>::zeros((m.atoms.len(), 3));
    for k in 0..3 {
        let f = SPRING_K * (r - SPRING_R0) * d[k] / r;
        g[(1, k)] = f;
        g[(0, k)] = -f;
    }
    (e, g)
}

/// Closed-form Cartesian Hessian of [`spring`].
fn spring_hessian(m: &Molecule) -> Array2<f64> {
    let (a, b) = (&m.atoms[0], &m.atoms[1]);
    let d = [b.x - a.x, b.y - a.y, b.zpos - a.zpos];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let n = 3 * m.atoms.len();
    let mut h = Array2::<f64>::zeros((n, n));
    for i in 0..3 {
        for j in 0..3 {
            let delta = if i == j { 1.0 } else { 0.0 };
            let u = d[i] * d[j] / (r * r);
            let k_ij = SPRING_K * (u + (1.0 - SPRING_R0 / r) * (delta - u));
            h[(3 + i, 3 + j)] = k_ij;
            h[(i, j)] = k_ij;
            h[(i, 3 + j)] = -k_ij;
            h[(3 + i, j)] = -k_ij;
        }
    }
    h
}

/// Bar on the plumbing test, at delta = 1e-3 Bohr. The spring Hessian has
/// 1/r terms, so central FD carries an O(delta^2) truncation: measured
/// max|H_driver - H_spring| = 1.646e-6 at the default delta = 5e-3 and
/// 6.58e-8 at 1e-3 (ratio 25.0 = (5e-3/1e-3)^2, pure truncation; the SCF
/// parts cancel exactly because both runs solve the same SCFs).
const SPRING_DELTA: f64 = 1e-3;
const SPRING_TOL: f64 = 2e-7;

/// The driver's correction Hessian is the closed-form spring Hessian.
/// `H(HF + spring) - H(HF)` isolates it because the SCF gradients at each
/// displaced geometry are the same in both runs. Dropping the correction
/// gradient, flipping its sign, or evaluating it at the undisplaced geometry
/// all miss by O(SPRING_K) = 0.3, far above the bar.
#[test]
fn driver_adds_the_correction_hessian_exactly() {
    let mol = Molecule::parse_xyz(H2_XYZ, 0, 1).expect("h2");
    let cfg = tight_hf();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let fc = FrequencyConfig {
        delta: SPRING_DELTA,
        ..fd_config()
    };
    let plain = harmonic_frequencies(&ctx, &mol, "sto-3g", op, &cfg, &fc).expect("plain");
    let corr =
        harmonic_frequencies_with_scf_correction(&ctx, &mol, "sto-3g", op, &cfg, &fc, |m, _| {
            let (e, g) = spring(m);
            Ok((e, Some(g)))
        })
        .expect("corrected");
    let h_corr = &corr.cartesian_hessian - &plain.cartesian_hessian;
    let exact = spring_hessian(&mol);
    let diff = max_abs(&(&h_corr - &exact));
    println!(
        "spring: max|H_spring| = {:.3e}, max|driver - exact| = {diff:.3e}",
        max_abs(&exact)
    );
    assert!(
        diff < SPRING_TOL,
        "driver correction Hessian off by {diff:.3e}"
    );
    // The energy is the corrected total, and the correction is reported.
    let (e_spring, _) = spring(&mol);
    assert_eq!(corr.correction_energy, e_spring);
    assert_eq!(corr.energy, plain.energy + e_spring);
    // Reachability: the correction is large enough that losing it would fail.
    assert!(max_abs(&exact) > 100.0 * SPRING_TOL);
}

/// THE exactness anchor: a correction that does nothing reproduces
/// `harmonic_frequencies` with `hessian = "fd"` bit for bit.
#[test]
fn correction_free_closure_is_bit_identical_to_plain_fd() {
    let mol = Molecule::parse_xyz(H2_XYZ, 0, 1).expect("h2");
    let cfg = tight_hf();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let plain = harmonic_frequencies(&ctx, &mol, "sto-3g", op, &cfg, &fd_config()).expect("plain");
    let none = harmonic_frequencies_with_scf_correction(
        &ctx,
        &mol,
        "sto-3g",
        op,
        &cfg,
        &FrequencyConfig::default(),
        |_, _| Ok((0.0, None)),
    )
    .expect("no-op correction");
    assert_eq!(plain.cartesian_hessian, none.cartesian_hessian);
    assert_eq!(plain.frequencies, none.frequencies);
    assert_eq!(plain.energy, none.energy);
    assert_eq!(none.correction_energy, 0.0);
    assert_eq!(none.n_gradient_evaluations, plain.n_gradient_evaluations);
}

/// The refusals: an analytic Hessian request, an open-shell reference, and an
/// energy correction without a gradient.
#[test]
fn unsupported_combinations_are_refused() {
    let mol = Molecule::parse_xyz(H2_XYZ, 0, 1).expect("h2");
    let cfg = tight_hf();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let analytic = FrequencyConfig {
        hessian: HessianMethod::Analytic,
        ..Default::default()
    };
    let e = harmonic_frequencies_with_scf_correction(
        &ctx,
        &mol,
        "sto-3g",
        op,
        &cfg,
        &analytic,
        |_, _| Ok((0.0, None)),
    )
    .expect_err("analytic must be refused");
    assert!(e.to_string().contains("analytic"), "{e}");
    let uhf = FrequencyConfig {
        reference: ferric_scf::frequencies::FrequencyReference::Uhf,
        ..Default::default()
    };
    let e =
        harmonic_frequencies_with_scf_correction(&ctx, &mol, "sto-3g", op, &cfg, &uhf, |_, _| {
            Ok((0.0, None))
        })
        .expect_err("open shell must be refused");
    assert!(e.to_string().contains("closed-shell"), "{e}");
    let e = harmonic_frequencies_with_scf_correction(
        &ctx,
        &mol,
        "sto-3g",
        op,
        &cfg,
        &FrequencyConfig::default(),
        |_, _| Ok((1e-3, None)),
    )
    .expect_err("energy without gradient must be refused");
    assert!(e.to_string().contains("no gradient"), "{e}");
}

/// Bar for the D3 Hessian against second differences of the D3 energy.
/// Both sides are finite differences (driver: gradient, delta = 5e-3; energy:
/// h = 2e-3), so this is an O(h^2) truncation bar; measured 5.28e-9 against
/// max|H_disp| = 1.86e-5 (PBE/STO-3G water).
const D3_TOL: f64 = 2e-8;
/// Bar on the frequencies: the driver's against those of
/// `H_KS + E''_D3` (the plain FD Hessian plus the energy second differences),
/// in cm^-1. Measured 9.8e-6 cm^-1; the D3 shifts themselves are 1.4e-2,
/// -3.2e-2 and -4.3e-2 cm^-1 (PBE/STO-3G water at this fixed geometry).
const D3_FREQ_TOL: f64 = 5e-5;
/// The negative control: D3 must move at least one frequency by more than
/// this (cm^-1; measured max shift 4.3e-2), or the test cannot tell a dropped
/// correction from a kept one.
const D3_MIN_SHIFT: f64 = 1e-2;

/// H_disp(D3(BJ)) from the driver (PBE/STO-3G water) against 4-point second
/// differences of the D3(BJ) energy. Also checks the KS part cancels: the
/// same SCFs run in both driver calls.
#[test]
fn d3_hessian_matches_second_differences_of_the_d3_energy() {
    let mol = Molecule::parse_xyz(H2O_XYZ, 0, 1).expect("h2o");
    let cfg = rks_pbe();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let plain = harmonic_frequencies(&ctx, &mol, "sto-3g", op, &cfg, &fd_config()).expect("plain");
    let with = harmonic_frequencies_with_scf_correction(
        &ctx,
        &mol,
        "sto-3g",
        op,
        &cfg,
        &FrequencyConfig::default(),
        d3_correction(),
    )
    .expect("d3");
    let h_disp = &with.cartesian_hessian - &plain.cartesian_hessian;
    let params = ferric_d3::d3bj_params_for_functional("pbe").expect("params");
    let reference = energy_second_differences(&mol, 2e-3, |m| {
        ferric_d3::d3bj_energy_for_molecule(m, &params).expect("d3 energy")
    });
    let diff = max_abs(&(&h_disp - &reference));
    println!(
        "D3 water: max|H_disp| = {:.3e}, max|driver - E''| = {diff:.3e}, asym(with) = {:.3e}",
        max_abs(&reference),
        with.asymmetry
    );
    assert!(
        diff < D3_TOL,
        "D3 Hessian vs energy second differences: {diff:.3e}"
    );
    assert!(
        max_abs(&reference) > 100.0 * D3_TOL,
        "the D3 Hessian must be resolvable at the bar"
    );
    // The same comparison in the observable: frequencies of the driver vs
    // frequencies of H_KS + E''_D3.
    let masses = ferric_scf::frequencies::atom_masses(&mol).expect("masses");
    let independent = ferric_scf::frequencies::frequencies_from_cartesian_hessian(
        &mol,
        &(&plain.cartesian_hessian + &reference),
        &masses,
    )
    .expect("independent frequencies");
    let mut freq_diff: f64 = 0.0;
    let mut shift: f64 = 0.0;
    for ((w, wi), w0) in with
        .frequencies
        .iter()
        .zip(&independent.frequencies)
        .zip(&plain.frequencies)
    {
        freq_diff = freq_diff.max((w - wi).abs());
        shift = shift.max((w - w0).abs());
    }
    println!(
        "D3 water: freqs {:?} vs PBE {:?}; max|driver - independent| = {freq_diff:.3e} cm^-1, \
         max|shift| = {shift:.3e} cm^-1",
        with.frequencies, plain.frequencies
    );
    assert!(
        freq_diff < D3_FREQ_TOL,
        "D3 frequencies off by {freq_diff:.3e} cm^-1"
    );
    assert!(
        shift > D3_MIN_SHIFT,
        "D3 must shift a frequency measurably (negative control): {shift:.3e} cm^-1"
    );
}

/// Energy second-difference step for the MBD check. The MBD energy of the
/// full pipeline carries noise of ~3e-10 Ha (SCF convergence and the
/// piecewise-linear proatom tables), which second differences amplify as
/// 1/h^2: measured along probe direction 2, |E'' - (v.g)'| = 4.4e-5, 1.2e-5,
/// 2.4e-6, 1.0e-6, 6.6e-7 at h = 0.0025, 0.005, 0.01, 0.02, 0.04 Bohr
/// (`measure_mbd_hessian_step_scan`), while the gradient-based slope stays
/// within 2.5e-5..2.8e-5. h = 0.02 balances that noise against truncation.
const MBD_ENERGY_H: f64 = 0.02;
/// Bar for the MBD Hessian along fixed directions, against second differences
/// of the full-pipeline MBD energy at [`MBD_ENERGY_H`]. NOISE-LIMITED, not a
/// construction bar: measured worst 1.55e-6 over three directions (others
/// 4.8e-7, 1.54e-6) with max|v^T H v| = 1.97e-4, i.e. 0.8%. Dropping the
/// Z-vector orbital relaxation from the gradient misses by 3.2e-5 (worst
/// direction; 1.9e-5 and 4.4e-6 on the others), asserted to fail the bar.
const MBD_TOL: f64 = 3e-6;

/// Fixed, non-symmetric probe directions (normalized below).
fn probe_directions(n: usize) -> Vec<Vec<f64>> {
    let raw: [[f64; 9]; 3] = [
        [0.3, -0.1, 0.7, -0.2, 0.5, 0.1, 0.4, -0.6, -0.3],
        [0.0, 0.0, 1.0, 0.0, 0.6, -0.4, 0.0, -0.6, -0.4],
        [0.9, 0.2, -0.1, -0.5, 0.3, 0.8, 0.1, -0.7, 0.2],
    ];
    raw.iter()
        .map(|r| {
            let v: Vec<f64> = r.iter().take(n).copied().collect();
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.iter().map(|x| x / norm).collect()
        })
        .collect()
}

fn quad_form(h: &Array2<f64>, v: &[f64]) -> f64 {
    let n = v.len();
    let mut s = 0.0;
    for i in 0..n {
        for j in 0..n {
            s += v[i] * h[(i, j)] * v[j];
        }
    }
    s
}

fn mbd_frequencies(
    mol: &Molecule,
    bs: &BasisSet,
    cfg: &RhfConfig,
    cache: &MbdFreeAtomCache,
    mcfg: &MbdRsscsConfig,
    relaxed: bool,
) -> FrequencyResult {
    let ctx = ParallelContext::default();
    harmonic_frequencies_with_scf_correction(
        &ctx,
        mol,
        &bs.name,
        Operator::coulomb(),
        cfg,
        &FrequencyConfig::default(),
        |m, scf| {
            let r = mbd_rsscs_for_scf(&ctx, cache, m, bs, Operator::coulomb(), cfg, scf, mcfg)?;
            // `relaxed = false` is a deliberate defect (no Z-vector term),
            // used to show the energy check can see the orbital relaxation.
            let g = if relaxed {
                r.gradient
            } else {
                r.gradient_unrelaxed
            };
            Ok((r.energy, g))
        },
    )
    .expect("mbd frequencies")
}

/// H_disp(MBD@rsSCS) (PBE/STO-3G water) along three fixed directions v
/// against `(E(+hv) - 2E(0) + E(-hv))/h^2` of the MBD energy with the SCF
/// re-solved at every displaced geometry. The energy side never calls the
/// gradient code, so it is an independent construction of the same second
/// derivative, density response included.
#[test]
#[ignore = "slow: 19 SCF + MBD gradients plus 7 SCF energies on PBE/STO-3G water"]
fn mbd_hessian_matches_second_differences_of_the_full_pipeline_energy() {
    let mol = Molecule::parse_xyz(H2O_XYZ, 0, 1).expect("h2o");
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let cfg = rks_pbe();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let mcfg = MbdRsscsConfig::for_functional("PBE").expect("beta");
    let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
    let plain = harmonic_frequencies(&ctx, &mol, "sto-3g", op, &cfg, &fd_config()).expect("plain");
    let with = mbd_frequencies(&mol, &bs, &cfg, &cache, &mcfg, true);
    let h_disp = &with.cartesian_hessian - &plain.cartesian_hessian;
    let unrelaxed = mbd_frequencies(&mol, &bs, &cfg, &cache, &mcfg, false);
    let h_unrelaxed = &unrelaxed.cartesian_hessian - &plain.cartesian_hessian;
    let e_mbd = |m: &Molecule| {
        let d = scf(m, &bs, &cfg);
        mbd_rsscs_for_density(&cache, m, &bs, d.density_total(), &mcfg, false)
            .expect("mbd")
            .energy
    };
    let h: f64 = std::env::var("MBD_HESS_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MBD_ENERGY_H);
    let e0 = e_mbd(&mol);
    let mut worst: f64 = 0.0;
    let mut worst_unrelaxed: f64 = 0.0;
    let mut scale: f64 = 0.0;
    for v in probe_directions(9) {
        let fd =
            (e_mbd(&shifted(&mol, &v, h)) - 2.0 * e0 + e_mbd(&shifted(&mol, &v, -h))) / (h * h);
        let an = quad_form(&h_disp, &v);
        println!(
            "MBD water v^T H v: driver = {an:+.6e}, E'' (h={h}) = {fd:+.6e}, diff = {:.3e}",
            an - fd
        );
        let un = quad_form(&h_unrelaxed, &v);
        println!(
            "    unrelaxed (no Z-vector) driver = {un:+.6e}, diff = {:.3e}",
            un - fd
        );
        worst = worst.max((an - fd).abs());
        worst_unrelaxed = worst_unrelaxed.max((un - fd).abs());
        scale = scale.max(fd.abs());
    }
    println!("MBD water: worst diff without the Z-vector term = {worst_unrelaxed:.3e}");
    println!(
        "MBD water: max|v^T H v| = {scale:.3e}, worst diff = {worst:.3e}, asym(with) = {:.3e}, \
         E_MBD = {:.6e}, correction_energy = {:.6e}",
        with.asymmetry, e0, with.correction_energy
    );
    assert!(
        worst < MBD_TOL,
        "MBD Hessian vs energy second differences: {worst:.3e}"
    );
    assert!(
        scale > 10.0 * MBD_TOL,
        "the MBD Hessian must be resolvable at the bar"
    );
    assert!(
        worst_unrelaxed > MBD_TOL,
        "the check must SEE a missing orbital relaxation: {worst_unrelaxed:.3e}"
    );
}

/// Measurement: water frequencies at PBE/6-31G without dispersion, with
/// D3(BJ) and with MBD@rsSCS, at ONE fixed geometry (the uncorrected PBE
/// minimum) so the shift is the Hessian's alone, plus symmetry and trans/rot
/// diagnostics.
#[test]
#[ignore = "measurement: three PBE/6-31G water frequency runs plus an optimization"]
fn measure_water_frequency_shifts() {
    let basis_name = std::env::var("DISP_FREQ_BASIS").unwrap_or_else(|_| "6-31g".to_string());
    let bs = basis::bundled(&basis_name).expect("basis");
    let cfg = rks_pbe();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let start = Molecule::parse_xyz(H2O_XYZ, 0, 1).expect("h2o");
    let opt = ferric_scf::optimize::optimize_geometry(
        &ctx,
        &start,
        &basis_name,
        op,
        &cfg,
        &ferric_scf::optimize::OptimizeConfig {
            g_max_thresh: 1e-6,
            g_rms_thresh: 1e-6,
            ..Default::default()
        },
    )
    .expect("opt");
    assert!(opt.converged);
    let mol = opt.mol;
    let plain =
        harmonic_frequencies(&ctx, &mol, &basis_name, op, &cfg, &fd_config()).expect("plain");
    let d3 = harmonic_frequencies_with_scf_correction(
        &ctx,
        &mol,
        &basis_name,
        op,
        &cfg,
        &FrequencyConfig::default(),
        d3_correction(),
    )
    .expect("d3");
    let mcfg = MbdRsscsConfig::for_functional("PBE").expect("beta");
    let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
    let mbd = mbd_frequencies(&mol, &bs, &cfg, &cache, &mcfg, true);
    for (label, r) in [
        ("PBE", &plain),
        ("PBE-D3(BJ)", &d3),
        ("PBE+MBD@rsSCS", &mbd),
    ] {
        println!(
            "{label:>14} /{basis_name}: freqs = {:?} cm^-1, asym = {:.3e}, max|trans/rot| = {:.3e} cm^-1, E = {:.10}, E_corr = {:+.3e}",
            r.frequencies,
            r.asymmetry,
            r.trans_rot_frequencies.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
            r.energy,
            r.correction_energy
        );
    }
    for (label, r) in [("D3(BJ)", &d3), ("MBD@rsSCS", &mbd)] {
        let shifts: Vec<f64> = r
            .frequencies
            .iter()
            .zip(&plain.frequencies)
            .map(|(a, b)| a - b)
            .collect();
        println!("{label} shift vs PBE (cm^-1): {shifts:?}");
    }
}

/// Diagnostic: step-size scan of the MBD second derivative along one
/// direction, from the energy (second differences) and from the exact
/// gradient (first differences of v.g), plus the first-derivative
/// consistency v.g(0) vs (E(+hv) - E(-hv))/(2h).
#[test]
#[ignore = "diagnostic: MBD Hessian step-size scan, slow"]
fn measure_mbd_hessian_step_scan() {
    let mol = Molecule::parse_xyz(H2O_XYZ, 0, 1).expect("h2o");
    let bs = basis::bundled("sto-3g").expect("sto-3g");
    let cfg = rks_pbe();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let mcfg = MbdRsscsConfig::for_functional("PBE").expect("beta");
    let cache = MbdFreeAtomCache::build(&ctx, &mol, &bs, op, &cfg).expect("cache");
    let eval = |m: &Molecule| {
        let d = scf(m, &bs, &cfg);
        let r = mbd_rsscs_for_scf(&ctx, &cache, m, &bs, op, &cfg, &d, &mcfg).expect("mbd");
        let g = r.gradient.expect("gradient");
        (r.energy, g)
    };
    let dot = |g: &Array2<f64>, v: &[f64]| -> f64 { g.iter().zip(v).map(|(a, b)| a * b).sum() };
    let which: usize = std::env::var("MBD_SCAN_DIR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let v = probe_directions(9)[which].clone();
    let (e0, g0) = eval(&mol);
    println!("direction {which}: v.g(0) = {:+.10e}", dot(&g0, &v));
    for h in [0.0025, 0.005, 0.01, 0.02, 0.04] {
        let (ep, gp) = eval(&shifted(&mol, &v, h));
        let (em, gm) = eval(&shifted(&mol, &v, -h));
        let s_e = (ep - 2.0 * e0 + em) / (h * h);
        let s_g = (dot(&gp, &v) - dot(&gm, &v)) / (2.0 * h);
        let d1 = (ep - em) / (2.0 * h);
        println!(
            "h = {h:.4}: E'' = {s_e:+.6e}, (v.g)' = {s_g:+.6e}, diff = {:+.3e}; \
             (E+ - E-)/2h = {d1:+.10e}, v.g(0) - that = {:+.3e}",
            s_e - s_g,
            dot(&g0, &v) - d1
        );
    }
}
