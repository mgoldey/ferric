//! Harmonic frequencies on a dispersion-corrected surface for OPEN-SHELL
//! references: `harmonic_frequencies_with_scf_correction` with
//! `FrequencyReference::Uhf` (UKS) and `FrequencyReference::Rohf` (ROKS).
//!
//! The driver solves the reference's own SCF at every displaced geometry and
//! hands that [`ScfResult`] to the correction, so
//!
//! ```text
//! H_disp = H(KS + disp) - H(KS)
//! ```
//!
//! exactly (the UKS/ROKS SCFs at each displaced geometry are the same in both
//! runs). These are the per-commit tests; the MBD@rsSCS Hessian (density
//! dependent, through the UKS / ROKS Z-vector) is checked against second
//! differences of the full SCF + MBD energy in
//! `validation_dispersion_frequencies_open_shell.rs`.
//!
//! Per reference (NH2 doublet for UKS; HCO doublet for ROKS, whose SOMO
//! shares symmetry with closed orbitals; both PBE/STO-3G, bent so no gradient
//! component vanishes by symmetry):
//!
//! * the EXACTNESS ANCHOR -- a correction returning `(0.0, None)` reproduces
//!   `harmonic_frequencies` with `hessian = "fd"` on the same reference bit
//!   for bit, and the closure sees an SCF of that reference's spin;
//! * the D3(BJ) Hessian against 4-point second differences of the D3(BJ)
//!   ENERGY (no gradient code), and the frequencies against those of the
//!   plain Hessian plus that independent D3 Hessian;
//! * rigid-body sum rules of the dispersion Hessian: translations
//!   (`sum_A H[(A,a), b] = 0`) and rotations
//!   (`sum_A eps_kij R_Ai H[(A,j), b] + eps_k(beta)j g_Bj = 0`), plus the
//!   translational sum of the full corrected Hessian.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli --release \
//!   --test dispersion_frequencies_open_shell -- --nocapture
//! ```

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    atom_masses, frequencies_from_cartesian_hessian, harmonic_frequencies,
    harmonic_frequencies_with_scf_correction, FrequencyConfig, FrequencyReference, HessianMethod,
};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::{ScfResult, Spin};
use ndarray::Array2;

// Bent off the symmetric geometry (as in ferric-rpa's mbd_scf_gradient_roks.rs)
// so that no Cartesian is inert by symmetry.
const NH2_XYZ: &str =
    "3\nNH2\nN 0.000000 0.020000 0.142000\nH 0.050000 0.802000 -0.497000\nH 0.000000 -0.782000 -0.517000\n";
// Formyl radical: in-plane (a') SOMO, so the ROKS closed-open coupling is
// nonzero.
const HCO_XYZ: &str =
    "3\nHCO\nC 0.000000 0.000000 0.000000\nO 1.180000 0.000000 0.000000\nH -0.627000 0.916000 0.000000\n";

fn pbe() -> RhfConfig {
    RhfConfig {
        xc: Some("PBE".to_string()),
        max_iter: 300,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    }
}

fn fd(reference: FrequencyReference) -> FrequencyConfig {
    FrequencyConfig {
        reference,
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

/// 4-point central second differences of a pure function of geometry.
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

/// Largest translational sum-rule residual `|sum_A H[(A,a), b]|` over a, b.
fn translation_residual(h: &Array2<f64>) -> f64 {
    let n = h.nrows();
    let mut worst: f64 = 0.0;
    for a in 0..3 {
        for b in 0..n {
            let s: f64 = (0..n / 3).map(|at| h[(3 * at + a, b)]).sum();
            worst = worst.max(s.abs());
        }
    }
    worst
}

/// Largest rotational sum-rule residual of a Hessian `h` whose gradient at
/// `mol` is `g` (the derivative of the zero-torque identity
/// `sum_A R_A x g_A = 0`), and the size of the gradient term in it (so a
/// residual far below that term shows the identity is actually exercised).
fn rotation_residual(mol: &Molecule, h: &Array2<f64>, g: &Array2<f64>) -> (f64, f64) {
    let eps = |k: usize, i: usize, j: usize| -> f64 {
        match (k, i, j) {
            (0, 1, 2) | (1, 2, 0) | (2, 0, 1) => 1.0,
            (0, 2, 1) | (1, 0, 2) | (2, 1, 0) => -1.0,
            _ => 0.0,
        }
    };
    let pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let n = h.nrows();
    let (mut worst, mut scale): (f64, f64) = (0.0, 0.0);
    for k in 0..3 {
        for b in 0..n {
            let (bb, beta) = (b / 3, b % 3);
            let mut hess_term = 0.0;
            for (at, r) in pos.iter().enumerate() {
                for i in 0..3 {
                    for j in 0..3 {
                        hess_term += eps(k, i, j) * r[i] * h[(3 * at + j, b)];
                    }
                }
            }
            let grad_term: f64 = (0..3).map(|j| eps(k, beta, j) * g[(bb, j)]).sum();
            worst = worst.max((hess_term + grad_term).abs());
            scale = scale.max(grad_term.abs());
        }
    }
    (worst, scale)
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

/// Bar for the D3 Hessian against second differences of the D3 energy. Both
/// sides are finite differences (driver: gradient, delta = 5e-3; energy:
/// h = 2e-3), so this is an O(h^2) truncation bar, the closed-shell water
/// test's. Measured max|driver - E''| = 3.7e-9 (UKS NH2, max|H_disp| 1.7e-5)
/// and 6.5e-9 (ROKS HCO, max|H_disp| 2.9e-5).
const D3_TOL: f64 = 2e-8;
/// Bar on the frequencies: driver vs `H_KS + E''_D3`, cm^-1. Measured 9.4e-6
/// (UKS NH2) and 3.0e-5 (ROKS HCO).
const D3_FREQ_TOL: f64 = 5e-5;
/// Negative control: D3 must move at least one frequency by more than this
/// (cm^-1; measured max shift 4.2e-2 UKS NH2, 1.4e-1 ROKS HCO), or the test
/// cannot tell a dropped correction from a kept one.
const D3_MIN_SHIFT: f64 = 1e-2;
/// Translational sum rule of the D3 part of the Hessian. The D3 gradient sums
/// to zero at every geometry, so only round-off of the central difference
/// remains: measured 1.1e-11 (UKS NH2), 9.5e-12 (ROKS HCO).
const D3_TRANS_TOL: f64 = 1e-10;
/// Rotational sum rule of the D3 Hessian with the D3 gradient at the input
/// geometry. The analytic Hessian satisfies it exactly; the driver's central
/// difference carries its O(delta^2) truncation, the same size as the
/// D3-vs-E'' difference above: measured 1.6e-9 (UKS NH2), 3.0e-9 (ROKS HCO),
/// against a gradient term of 1.9e-6 / 1.1e-5 that a missing or mis-signed
/// piece would leave behind.
const D3_ROT_TOL: f64 = 1e-8;
/// Translational sum of the FULL corrected Hessian, Hartree/Bohr^2: the KS
/// gradient includes the grid response, so this is limited by SCF and
/// quadrature noise over the 5e-3 Bohr step. Measured 1.1e-6 (UKS NH2),
/// 1.3e-6 (ROKS HCO).
const FULL_TRANS_TOL: f64 = 1e-5;

/// The anchor and the D3 checks for one open-shell reference.
fn check_reference(
    label: &str,
    xyz: &str,
    multiplicity: usize,
    reference: FrequencyReference,
    spin: Spin,
) {
    let mol = Molecule::parse_xyz(xyz, 0, multiplicity).expect("molecule");
    let cfg = pbe();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let plain =
        harmonic_frequencies(&ctx, &mol, "sto-3g", op, &cfg, &fd(reference)).expect("plain");

    // --- Exactness anchor: (0.0, None) is bit-identical to the plain FD path,
    // and the closure sees this reference's SCF at every geometry.
    let mut calls = 0usize;
    let none = harmonic_frequencies_with_scf_correction(
        &ctx,
        &mol,
        "sto-3g",
        op,
        &cfg,
        &FrequencyConfig {
            reference,
            ..Default::default()
        },
        |_, scf| {
            assert_eq!(scf.spin, spin, "{label}: the correction saw the wrong SCF");
            calls += 1;
            Ok((0.0, None))
        },
    )
    .expect("no-op correction");
    assert_eq!(
        plain.cartesian_hessian, none.cartesian_hessian,
        "{label}: Hessian"
    );
    assert_eq!(plain.frequencies, none.frequencies, "{label}: frequencies");
    assert_eq!(plain.energy, none.energy, "{label}: energy");
    assert_eq!(none.correction_energy, 0.0);
    assert_eq!(none.n_gradient_evaluations, plain.n_gradient_evaluations);
    assert_eq!(
        calls,
        1 + 6 * mol.atoms.len(),
        "{label}: one call per geometry"
    );

    // --- D3(BJ): Hessian vs second differences of the D3 energy.
    let with = harmonic_frequencies_with_scf_correction(
        &ctx,
        &mol,
        "sto-3g",
        op,
        &cfg,
        &fd(reference),
        d3_correction(),
    )
    .expect("d3");
    let h_disp = &with.cartesian_hessian - &plain.cartesian_hessian;
    let params = ferric_d3::d3bj_params_for_functional("pbe").expect("params");
    let e_d3 = ferric_d3::d3bj_energy_for_molecule(&mol, &params).expect("d3 energy");
    assert_eq!(with.correction_energy, e_d3, "{label}: reported correction");
    assert_eq!(
        with.energy,
        plain.energy + e_d3,
        "{label}: corrected energy"
    );
    let reference_h = energy_second_differences(&mol, 2e-3, |m| {
        ferric_d3::d3bj_energy_for_molecule(m, &params).expect("d3 energy")
    });
    let diff = max_abs(&(&h_disp - &reference_h));
    println!(
        "{label} D3: max|H_disp| = {:.3e}, max|driver - E''| = {diff:.3e}, asym(with) = {:.3e}",
        max_abs(&reference_h),
        with.asymmetry
    );
    assert!(diff < D3_TOL, "{label}: D3 Hessian vs E'': {diff:.3e}");
    assert!(
        max_abs(&reference_h) > 100.0 * D3_TOL,
        "{label}: the D3 Hessian must be resolvable at the bar"
    );

    let masses = atom_masses(&mol).expect("masses");
    let independent = frequencies_from_cartesian_hessian(
        &mol,
        &(&plain.cartesian_hessian + &reference_h),
        &masses,
    )
    .expect("independent frequencies");
    let (mut freq_diff, mut shift): (f64, f64) = (0.0, 0.0);
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
        "{label} D3: freqs {:?} vs PBE {:?}; max|driver - independent| = {freq_diff:.3e} cm^-1, \
         max|shift| = {shift:.3e} cm^-1",
        with.frequencies, plain.frequencies
    );
    assert!(
        freq_diff < D3_FREQ_TOL,
        "{label}: D3 frequencies off by {freq_diff:.3e}"
    );
    assert!(
        shift > D3_MIN_SHIFT,
        "{label}: D3 shift {shift:.3e} cm^-1 too small to see"
    );

    // --- Rigid-body modes: sum rules of the dispersion Hessian (unprojected).
    let g_d3 = to_array(&ferric_d3::d3bj_gradient_for_molecule(&mol, &params).expect("g"));
    let t_disp = translation_residual(&h_disp);
    let (r_disp, r_scale) = rotation_residual(&mol, &h_disp, &g_d3);
    let t_full = translation_residual(&with.cartesian_hessian);
    let tr_max = with
        .trans_rot_frequencies
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    println!(
        "{label} rigid body: D3 translation {t_disp:.3e}, rotation {r_disp:.3e} (gradient term \
         {r_scale:.3e}); full-Hessian translation {t_full:.3e}; projected trans/rot max \
         {tr_max:.3e} cm^-1"
    );
    assert!(
        t_disp < D3_TRANS_TOL,
        "{label}: D3 translation sum {t_disp:.3e}"
    );
    assert!(r_disp < D3_ROT_TOL, "{label}: D3 rotation sum {r_disp:.3e}");
    assert!(
        r_scale > 100.0 * D3_ROT_TOL,
        "{label}: rotation rule not exercised"
    );
    assert!(
        t_full < FULL_TRANS_TOL,
        "{label}: full translation sum {t_full:.3e}"
    );
}

#[test]
fn uks_nh2_dispersion_frequencies() {
    check_reference(
        "UKS NH2",
        NH2_XYZ,
        2,
        FrequencyReference::Uhf,
        Spin::Unrestricted,
    );
}

#[test]
fn roks_hco_dispersion_frequencies() {
    check_reference(
        "ROKS HCO",
        HCO_XYZ,
        2,
        FrequencyReference::Rohf,
        Spin::RestrictedOpen,
    );
}
