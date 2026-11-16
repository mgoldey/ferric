//! IR intensities: the field route (production) against the dipole route.
//!
//! The atomic polar tensor `P[a, i] = d mu_a / d R_i` is computed two ways
//! that share nothing but the SCF solver:
//!
//! 1. `dipole_derivatives_field`: `-(g_i(+h e_a) - g_i(-h e_a)) / 2h`, the
//!    analytic gradient of SCFs solved in a uniform field `+-h`;
//! 2. `dipole_derivatives_dipole_fd`: `(mu_a(R + d e_i) - mu_a(R - d e_i)) /
//!    2d`, SCF dipoles at displaced geometries.
//!
//! Physics hypothesis: both are the same derivative of one surface, so they
//! agree to their truncation plus SCF noise. Artifact hypothesis: a sign
//! error in the field convention flips every element of route 1 (it then
//! differs from route 2 by `2|P|`, far outside the bar); a field applied to
//! the wrong axis permutes rows; a gradient that ignores the field gives
//! `P = 0`. Each of those is orders of magnitude outside the bar, which the
//! test also checks is far below `max |P|` (it is not vacuous).
//!
//! The bar is derived, not guessed. For a central difference with step `s`,
//! `A(s) = A + c s^2 + O(s^4)`, so `|A(s) - A| ~ |A(2s) - A(s)| / 3`
//! (Richardson); each route is run at `s` and `2s` and its truncation
//! estimate is `t = max |A(2s) - A(s)| / 3`. SCF exits add noise: the
//! gradient moves by at most `K_g |dD|_F` and the dipole by at most
//! `K_mu |dD|_F`, with `|dD|_F <= n density_conv` and `K` measured from a
//! loose and a tight SCF, so route 1 carries at most `K_g n dc / h` and
//! route 2 `K_mu n dc / d`. The bar is
//! `2 (t_1 + t_2) + 2 (e_1 + e_2)`; the factor 2 covers the `O(s^4)`
//! remainder and the noise inside the Richardson estimate itself.
//!
//! Intensities from both tensors along the same modes are compared to the
//! bound that the tensor bar implies: `|A_1 - A_2| <= C (2 |v| + b) b` with
//! `v = d mu / d Q` of route 2 and `b = bar * sqrt(3) * |x_k|_1`.

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    harmonic_frequencies, FrequencyConfig, FrequencyReference, HessianMethod,
};
use ferric_scf::infrared::{
    dipole_derivatives_dipole_fd, dipole_derivatives_field, ir_intensities, sum_rule_residual,
    AU_TO_KM_PER_MOL,
};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const DC: f64 = 1.0e-10;

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .unwrap()
}

fn hydroxyl() -> Molecule {
    Molecule::parse_xyz("2\n\nO 0 0 0\nH 0 0.15 0.95\n", 0, 2).unwrap()
}

fn scf(xc: Option<&str>) -> RhfConfig {
    RhfConfig {
        xc: xc.map(str::to_string),
        density_conv: DC,
        energy_conv: DC,
        max_iter: 300,
        ..Default::default()
    }
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// `(K_g, K_mu)`: max change of the gradient and of the dipole per unit
/// Frobenius change of the density, from an SCF at density_conv 1e-5 against
/// one at `DC` (zero field).
fn sensitivities(
    mol: &Molecule,
    basis: &str,
    cfg: &RhfConfig,
    reference: FrequencyReference,
) -> (f64, f64) {
    let ctx = ParallelContext::default();
    let bs = ferric_core::basis::bundled(basis).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let at = |dc: f64| {
        let c = RhfConfig {
            density_conv: dc,
            energy_conv: 1e-3,
            ..cfg.clone()
        };
        let r = match reference {
            FrequencyReference::Rhf => {
                ferric_scf::rhf::solve_rhf(&ctx, mol, &prep, op, &bounds, &c).unwrap()
            }
            FrequencyReference::Uhf => {
                ferric_scf::uhf::solve_uhf(&ctx, mol, &prep, &bounds, &c).unwrap()
            }
            FrequencyReference::Rohf => {
                ferric_scf::rohf::solve_rohf(&ctx, mol, &prep, op, &bounds, &c).unwrap()
            }
        };
        let g = match reference {
            FrequencyReference::Rhf => {
                ferric_scf::gradient::restricted_scf_gradient(mol, &prep, &bs, op, &bounds, &c, &r)
            }
            FrequencyReference::Uhf => ferric_scf::gradient::unrestricted_scf_gradient(
                mol, &prep, &bs, op, &bounds, &c, &r,
            ),
            FrequencyReference::Rohf => ferric_scf::gradient::restricted_open_scf_gradient(
                mol, &prep, &bs, op, &bounds, &c, &r,
            ),
        }
        .unwrap();
        let mu = ferric_scf::properties::dipole_moment(mol, &prep, r.density_total()).unwrap();
        (g, mu, r.density_total().clone())
    };
    let (g1, mu1, d1) = at(1e-5);
    let (g2, mu2, d2) = at(DC);
    let dd = (&d1 - &d2).mapv(|v| v * v).sum().sqrt();
    assert!(dd > 0.0, "loose and tight SCFs landed on the same density");
    let dg = max_abs(&(&g1 - &g2));
    let dmu = (0..3).map(|a| (mu1[a] - mu2[a]).abs()).fold(0.0, f64::max);
    (dg / dd, dmu / dd)
}

fn check_routes(label: &str, mol: &Molecule, basis: &str, xc: Option<&str>, r: FrequencyReference) {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let cfg = scf(xc);
    let (h, d) = (1.0e-3, 5.0e-3);
    let f = |s| dipole_derivatives_field(&ctx, mol, basis, op, &cfg, r, s).unwrap();
    let m = |s| dipole_derivatives_dipole_fd(&ctx, mol, basis, op, &cfg, r, s).unwrap();
    let (p1, p1_2) = (f(h), f(2.0 * h));
    let (p2, p2_2) = (m(d), m(2.0 * d));
    let t1 = max_abs(&(&p1_2 - &p1)) / 3.0;
    let t2 = max_abs(&(&p2_2 - &p2)) / 3.0;
    let n = {
        let bs = ferric_core::basis::bundled(basis).unwrap();
        PreparedBasis::new(mol, &bs).unwrap().nbasis() as f64
    };
    let (kg, kmu) = sensitivities(mol, basis, &cfg, r);
    let (e1, e2) = (kg * n * DC / h, kmu * n * DC / d);
    let bar = 2.0 * (t1 + t2) + 2.0 * (e1 + e2);
    let diff = max_abs(&(&p1 - &p2));
    let scale = max_abs(&p2);
    let q = f64::from(mol.charge);
    let (s1, s2) = (sum_rule_residual(&p1, q), sum_rule_residual(&p2, q));
    eprintln!(
        "{label}: max|P_field - P_dipole| {diff:.2e} (bar {bar:.2e}: t_field {t1:.1e}, \
         t_dipole {t2:.1e}, e_field {e1:.1e}, e_dipole {e2:.1e}; K_g {kg:.3}, K_mu {kmu:.3}, \
         n {n}); max|P| {scale:.3}; sum rule field {s1:.1e} dipole {s2:.1e}"
    );
    assert!(
        diff <= bar,
        "{label}: routes differ by {diff:e} > bar {bar:e}"
    );
    assert!(
        bar < 1e-3 * scale,
        "{label}: bar {bar:e} is not small against |P| {scale}"
    );
    assert!(
        s1 <= bar && s2 <= bar,
        "{label}: sum rule {s1:e} / {s2:e} > {bar:e}"
    );

    // Intensities along this system's normal modes.
    let fc = FrequencyConfig {
        reference: r,
        hessian: HessianMethod::Auto,
        ..Default::default()
    };
    let freq = harmonic_frequencies(&ctx, mol, basis, op, &cfg, &fc).unwrap();
    let a1 = ir_intensities(&p1, &freq.normal_modes).unwrap();
    let a2 = ir_intensities(&p2, &freq.normal_modes).unwrap();
    for (k, (x, y)) in a1.iter().zip(&a2).enumerate() {
        let xk = freq.normal_modes.row(k);
        let l1: f64 = xk.iter().map(|v| v.abs()).sum();
        let b = bar * 3f64.sqrt() * l1;
        let v = (y / AU_TO_KM_PER_MOL).sqrt();
        let bound = AU_TO_KM_PER_MOL * (2.0 * v + b) * b;
        eprintln!(
            "{label}: mode {k} {:.1} cm-1: {x:.4} vs {y:.4} km/mol (bound {bound:.1e})",
            freq.frequencies[k]
        );
        assert!(
            (x - y).abs() <= bound,
            "{label}: mode {k}: {x} vs {y} > {bound}"
        );
    }
}

#[test]
fn rhf_field_and_dipole_routes_agree() {
    check_routes(
        "RHF water/6-31G",
        &water(),
        "6-31g",
        None,
        FrequencyReference::Rhf,
    );
}

#[test]
fn rks_field_and_dipole_routes_agree() {
    check_routes(
        "RKS(PBE) water/6-31G",
        &water(),
        "6-31g",
        Some("PBE"),
        FrequencyReference::Rhf,
    );
}

#[test]
fn uhf_field_and_dipole_routes_agree() {
    check_routes(
        "UHF OH/6-31G",
        &hydroxyl(),
        "6-31g",
        None,
        FrequencyReference::Uhf,
    );
}

#[test]
fn uks_field_and_dipole_routes_agree() {
    check_routes(
        "UKS(PBE) OH/6-31G",
        &hydroxyl(),
        "6-31g",
        Some("PBE"),
        FrequencyReference::Uhf,
    );
}

#[test]
fn rohf_field_and_dipole_routes_agree() {
    check_routes(
        "ROHF OH/6-31G",
        &hydroxyl(),
        "6-31g",
        None,
        FrequencyReference::Rohf,
    );
}
