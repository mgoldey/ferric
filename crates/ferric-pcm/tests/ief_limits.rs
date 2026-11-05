//! Analytic limits of the IEF-PCM operators (`crates/ferric-pcm/theory.md`
//! §5-§6; continuum identities proven in `scripts/pcm_ief_sympy.py`).
//!
//! * A point charge inside a single sphere: the discrete
//!   `build_s_d` → `build_k_r` → `solve_pcm_charges` chain against the
//!   analytic Born (central) and Kirkwood (off-centre) reaction-field
//!   energies, and against the Gauss-law total charge `−(ε−1)/ε·Q`.
//! * The conductor limit `ε → ∞`: the IEF charges approach the C-PCM/COSMO
//!   conductor charges `q = −S⁻¹ v`, linearly in `1 − f(ε)`.
//!
//! Tolerances are a-posteriori discretization estimates (the change between
//! the two finest Lebedev orders), not fixed numbers.

use ferric_core::mol::Molecule;
use ferric_pcm::cavity::{build_cavity, CavityConfig, Tessera};
use ferric_pcm::matrices::{build_k_r, build_s_d};
use ferric_pcm::solver::solve_pcm_charges;
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

/// One-atom cavity (no neighbour, so every tessera has `switch_fun = 1`):
/// an exact Lebedev sphere of radius `vdw_scale · r_Bondi(Ne)`.
fn ne_sphere(lebedev_order: usize) -> Vec<Tessera> {
    let mol = Molecule::parse_xyz("1\nNe\nNe 0.0 0.0 0.0\n", 0, 1).unwrap();
    let cfg = CavityConfig {
        lebedev_order,
        ..CavityConfig::default()
    };
    let tess = build_cavity(&mol, &cfg).unwrap();
    assert_eq!(tess.len(), lebedev_order);
    tess
}

/// Kirkwood reaction-field energy of a unit point charge at distance `d`
/// from the centre of a sphere of radius `r` (vacuum inside, dielectric `ε`
/// outside): `E = −(1/2r) Σ_l (l+1)(ε−1)/((l+1)ε+l) · (d/r)^{2l}`.
/// `d = 0` is the Born energy `−(ε−1)/(2εr)`.
fn kirkwood_energy(eps: f64, r: f64, d: f64) -> f64 {
    let x = (d / r).powi(2);
    let mut sum = 0.0;
    let mut xl = 1.0;
    for l in 0..400 {
        let lf = l as f64;
        sum += (lf + 1.0) * (eps - 1.0) / ((lf + 1.0) * eps + lf) * xl;
        xl *= x;
        if xl < 1e-20 {
            break;
        }
    }
    -0.5 * sum / r
}

/// `(E_pcm, Σ q)` for a unit charge at `centre` inside the sphere.
fn sphere_response(tess: &[Tessera], eps: f64, centre: [f64; 3]) -> (f64, f64) {
    let v = Array1::from_iter(tess.iter().map(|t| {
        let dx = t.position[0] - centre[0];
        let dy = t.position[1] - centre[1];
        let dz = t.position[2] - centre[2];
        1.0 / (dx * dx + dy * dy + dz * dz).sqrt()
    }));
    let (s, d) = build_s_d(tess);
    let (k, r, _) = build_k_r(&s, &d, tess, eps).unwrap();
    let res = solve_pcm_charges(&k, &r, &v).unwrap();
    (res.e_pcm, res.q.sum())
}

/// Born ion (`d = 0`) and two off-centre charges (`d = R/4`, `R/2`, which
/// put weight on the `l > 0` Kirkwood channels and so on `D`'s off-diagonal
/// and on the order of `Rᵀ` and `(Kᵀ)⁻¹` in the symmetrized solve), at a
/// high and a low dielectric.
///
/// For each case the Lebedev orders 50 → 110 → 302 must (i) converge
/// monotonically to the analytic energy, and (ii) at 302 sit within the
/// a-posteriori discretization estimate `|E₃₀₂ − E₁₁₀|` of it; the total
/// charge must sit within `|Q₃₀₂ − Q₁₁₀|` of the Gauss-law `−(ε−1)/ε`.
#[test]
fn point_charge_in_sphere_matches_born_kirkwood_and_gauss_law() {
    let orders = [50usize, 110, 302];
    let spheres: Vec<Vec<Tessera>> = orders.iter().map(|&n| ne_sphere(n)).collect();
    let r_sph = spheres[0][0].sphere_radius;
    // Off-axis direction so no Lebedev symmetry element maps the charge to itself.
    let dir = {
        let u = [0.3_f64, -0.5, 0.81];
        let nrm = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        [u[0] / nrm, u[1] / nrm, u[2] / nrm]
    };
    for &eps in &[78.4_f64, 4.0] {
        let q_gauss = -(eps - 1.0) / eps;
        for &frac in &[0.0_f64, 0.25, 0.5] {
            let dist = frac * r_sph;
            let centre = [dist * dir[0], dist * dir[1], dist * dir[2]];
            let e_exact = kirkwood_energy(eps, r_sph, dist);
            let res: Vec<(f64, f64)> = spheres
                .iter()
                .map(|t| sphere_response(t, eps, centre))
                .collect();
            let err: Vec<f64> = res.iter().map(|(e, _)| (e - e_exact).abs()).collect();
            let est_e = (res[2].0 - res[1].0).abs();
            let est_q = (res[2].1 - res[1].1).abs();
            let label = format!("eps={eps} d/R={frac}");
            eprintln!(
                "{label}: E_exact={e_exact:.10} E(50,110,302)=({:.10},{:.10},{:.10}) \
                 rel err=({:.2e},{:.2e},{:.2e}) est={:.2e}; Q302={:.10} (Gauss {q_gauss:.10}) est_q={est_q:.2e}",
                res[0].0,
                res[1].0,
                res[2].0,
                err[0] / e_exact.abs(),
                err[1] / e_exact.abs(),
                err[2] / e_exact.abs(),
                est_e / e_exact.abs(),
                res[2].1,
            );
            assert!(
                err[2] < err[1] && err[1] < err[0],
                "{label}: energy error not converging with Lebedev order: {err:?}"
            );
            assert!(
                err[2] <= est_e,
                "{label}: |E302 - E_exact| = {:.3e} exceeds discretization estimate {est_e:.3e}",
                err[2]
            );
            assert!(
                (res[2].1 - q_gauss).abs() <= est_q,
                "{label}: total charge {} vs Gauss law {q_gauss}, estimate {est_q:.3e}",
                res[2].1
            );
        }
    }
}

/// Relative distance of the IEF charges at `eps` from the conductor charges.
fn conductor_gap(
    s: &Array2<f64>,
    d: &Array2<f64>,
    tess: &[Tessera],
    v: &Array1<f64>,
    q_cond: &Array1<f64>,
    eps: f64,
) -> f64 {
    let (k, r, _) = build_k_r(s, d, tess, eps).unwrap();
    let q = solve_pcm_charges(&k, &r, v).unwrap().q;
    let diff = &q - q_cond;
    diff.dot(&diff).sqrt() / q_cond.dot(q_cond).sqrt()
}

/// At `f = 1` the IEF operators factor as `K = (I − DA/2π) S`,
/// `R = −(I − DA/2π)`, so `K⁻¹R = −S⁻¹` exactly on ANY discretization:
/// the conductor-like (C-PCM / COSMO ε → ∞) equation `S q = −v`. Since
/// `K⁻¹R` is smooth in `f`, the gap is first order in `1 − f = 2/(ε+1)`;
/// the gap ratio between two large `ε` must equal the ratio of `1 − f`
/// (bracketed by a factor 2 for the second-order term, which at
/// `1 − f ≤ 2e-6` is ~1e-6 relative). Water's molecular cavity, where
/// `D` is not symmetric and the IEF and conductor solutions differ at
/// finite ε.
#[test]
fn ief_reduces_to_conductor_equation_as_eps_goes_to_infinity() {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
    let (s, d) = build_s_d(&tess);
    let v = Array1::from_iter(
        tess.iter()
            .map(|t| 0.1 + 0.05 * t.position[2] + 0.02 * t.position[1] * t.position[1]),
    );
    let q_cond = -s.solve(&v).unwrap();

    let (eps1, eps2) = (1e6_f64, 1e8_f64);
    let g1 = conductor_gap(&s, &d, &tess, &v, &q_cond, eps1);
    let g2 = conductor_gap(&s, &d, &tess, &v, &q_cond, eps2);
    let g_water = conductor_gap(&s, &d, &tess, &v, &q_cond, 78.4);
    let one_minus_f = |e: f64| 2.0 / (e + 1.0);
    let expected_ratio = one_minus_f(eps1) / one_minus_f(eps2);
    eprintln!(
        "gap(1e6)={g1:.3e} gap(1e8)={g2:.3e} ratio={:.3} expected {expected_ratio:.3}; gap(78.4)={g_water:.3e}",
        g1 / g2
    );
    assert!(
        g_water > 1e3 * g1,
        "IEF at eps=78.4 must differ from the conductor limit (gap {g_water:.3e})"
    );
    let ratio = g1 / g2;
    assert!(
        ratio > 0.5 * expected_ratio && ratio < 2.0 * expected_ratio,
        "conductor gap not first order in 1-f: ratio {ratio:.3}, expected {expected_ratio:.3}"
    );
}
