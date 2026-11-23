//! The OO-RI-MP2 analytic nuclear gradient (`oo_ri_mp2_gradient`, closed
//! shell, frozen core 0, zero orbital response) against a Richardson finite
//! difference of ferric's own re-converged OO-RI-MP2 energy, on BOTH J/K
//! routes of the reference SCF.
//!
//! # What is compared
//!
//! | case | basis | aux (/C) | J/K route |
//! |---|---|---|---|
//! | H2O (off C2v) | cc-pVDZ | cc-pvdz-ri | RI-JK (`def2-universal-jkfit`) |
//! | NH3 (distorted) | 6-31G | cc-pvdz-ri | RI-JK (`def2-universal-jkfit`) |
//! | NH3 (distorted) | 6-31G | cc-pvdz-ri | exact four-centre |
//!
//! The exact-J/K route is also covered, on H2O and NH3 at cc-pVDZ against
//! ORCA and a 5-point self-FD, by the validation tier
//! (`validation_oo_rimp2.rs`); the RI-JK route had only an H2O/STO-3G case at
//! a loose bar (`oo_rimp2_gradient.rs`). Every displaced geometry re-runs the
//! RHF and the full orbital optimization; nothing is probed at frozen
//! orbitals (that would validate only the integral-response terms, not the
//! absent-by-construction orbital response).
//!
//! The FD is the Richardson combination `(4 D(h) - D(2h)) / 3` of central
//! differences, `h = 2e-3` Bohr, the same stencil as `u_rimp2_gradient_fd.rs`.
//!
//! # Bar: 2e-8 Ha/Bohr, derived
//!
//! * truncation: `D(h) = g + h² E'''/6 + h⁴ E⁽⁵⁾/120 + ...`; Richardson
//!   removes the `h²` term and leaves `-h⁴ E⁽⁵⁾/30` = `5e-13 |E⁽⁵⁾|`, at most
//!   5e-10 for `|E⁽⁵⁾| <= 1e3` Ha/Bohr⁵ (the quartic coefficient of a
//!   molecular PES is far below that);
//! * energy noise: each displaced energy carries `δE <= 1e-11` Ha. The OO
//!   energy is quadratic in the orbital residual (`g² / 2λ`, `g <= 1e-8`,
//!   `λ ~ 0.5`: 1e-16), so `δE` is the integral and SCF-threshold floor
//!   (RI-JK has no screening; the exact route's Schwarz floor is the larger
//!   one, ~1e-11). Worst-case Richardson amplification is
//!   `(4/3)(δE/h) + (1/3)(δE/2h) = 1.5 δE/h`: 7.5e-9;
//! * the analytic side: the converged orbital-gradient norm
//!   (`grad_norm <= 1e-8`, asserted at EVERY point, the OO `grad_conv` is
//!   1e-9) times an O(1) coupling to the density: <= 1e-8.
//!
//! Sum <= 1.8e-8, rounded to 2e-8. A looser bar would hide a missing term;
//! the smallest term this test must see (see Mutations) is 1e-4 or larger.
//!
//! # Exactness anchors (independent of the FD)
//!
//! * translational invariance, `Σ_A g_A = 0`, and rotational invariance,
//!   `Σ_A r_A × g_A = 0`, hold for the exact derivative of any functional
//!   whose basis and aux functions follow the atoms. Both hold only at a
//!   point stationary to `grad_norm`, so they are bounded by [`TOL_INV`] =
//!   1e-8;
//! * the energy is invariant to occ-occ and vir-vir rotations (the premise
//!   of the zero-response formula): checked by the gradient not changing
//!   when the converged orbitals are rotated within the occupied block
//!   ([`rotate_occupied`], bar 1e-10).
//!
//! The "zero rotation" limit is not an anchor for this code: the OO gradient
//! formula is valid only at the OO stationary point, and the plain RI-MP2
//! gradient at the OO orbitals is not defined (its Z-vector assumes
//! canonical HF orbitals).
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * Right: the analytic gradient equals the FD to the bar on every case.
//! * An omitted orbital-dependent term (the energy-weighted density's
//!   `vhf_s1occ` piece), or a route ignored (four-centre J/K or two-electron
//!   derivative under a fitted functional): misses by >= 1e-4 (the
//!   `fit_effect` control asserts the two routes differ by more than ten
//!   bars, so a route-blind gradient cannot pass).
//! * Unconverged orbitals: the FD and the analytic gradient both move at
//!   first order in the residual; the `grad_norm` assertion at every point
//!   stops that being read as agreement.
//!
//! # Mutations (each fails an assertion; source restored)
//!
//! Recorded in the commit message.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config, OoRiMp2Result};
use ferric_mp2::oo_rimp2_gradient::oo_ri_mp2_gradient;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_testkit::compare::max_abs_diff;
use ndarray::Array2;

const H: f64 = 2e-3;
const BAR: f64 = 2e-8;
/// Translation / rotation invariance of the analytic gradient.
const TOL_INV: f64 = 1e-8;
/// Largest converged orbital-gradient norm accepted at any point.
const MAX_GRAD_NORM: f64 = 1e-8;
const AUX: &str = "cc-pvdz-ri";
const JKFIT: &str = "def2-universal-jkfit";

const WATER: &str = "3\nwater, off C2v\nO 0.02 -0.01 0.117790\nH 0.0 0.785453 -0.451161\nH 0.03 -0.735453 -0.491161\n";
const AMMONIA: &str = "4\nammonia, distorted\nN 0.02 -0.01 0.11\nH 0.0 0.95 -0.26\nH 0.84 -0.46 -0.30\nH -0.80 -0.50 -0.24\n";

fn oo_config() -> OoRiMp2Config {
    OoRiMp2Config {
        grad_conv: 1e-9,
        energy_conv: 1e-12,
        max_iter: 300,
        ..Default::default()
    }
}

struct Solved {
    prep: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
    oo: OoRiMp2Result,
}

/// RHF (tight, on `jk_aux` or exact) and the orbital optimization at `mol`.
fn solve(mol: &Molecule, obs: &BasisSet, aux: &BasisSet, jk_aux: Option<&str>) -> Solved {
    let op = Operator::coulomb();
    let prep = PreparedBasis::new(mol, obs).unwrap();
    let dfbs = PreparedBasis::new(mol, aux).unwrap();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = {
        let mut c = RhfConfig::default()
            .with_energy_conv(1e-12)
            .with_density_conv(1e-11)
            .with_max_iter(300);
        c.df_j_aux = jk_aux.map(str::to_string);
        c.df_k_aux = jk_aux.map(str::to_string);
        c
    }
    .for_geometry_task();
    let rhf = solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged, "reference RHF did not converge");
    assert_eq!(rhf.df_jk.is_some(), jk_aux.is_some(), "J/K route");
    let oo = oo_ri_mp2(mol, &prep, &dfbs, op, &bounds, &rhf, &oo_config(), None).unwrap();
    assert!(
        oo.converged,
        "OO did not converge: |g| {:.2e}",
        oo.grad_norm
    );
    assert!(
        oo.grad_norm <= MAX_GRAD_NORM,
        "OO residual |g| {:.2e} above {MAX_GRAD_NORM:.0e}",
        oo.grad_norm
    );
    assert_eq!(oo.jk_route.is_some(), jk_aux.is_some(), "OO J/K route");
    Solved {
        prep,
        dfbs,
        bounds,
        oo,
    }
}

fn analytic(mol: &Molecule, s: &Solved) -> Array2<f64> {
    analytic_with(mol, s, &s.oo)
}

fn analytic_with(mol: &Molecule, s: &Solved, oo: &OoRiMp2Result) -> Array2<f64> {
    oo_ri_mp2_gradient(
        mol,
        &s.prep,
        &s.dfbs,
        Operator::coulomb(),
        &s.bounds,
        oo,
        0,
        None,
    )
    .unwrap()
}

fn displaced(mol: &Molecule, atom: usize, coord: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match coord {
        0 => m.atoms[atom].x += h,
        1 => m.atoms[atom].y += h,
        _ => m.atoms[atom].zpos += h,
    }
    m
}

/// Richardson FD of the re-converged OO energy, every component.
fn richardson_fd(
    mol: &Molecule,
    obs: &BasisSet,
    aux: &BasisSet,
    jk_aux: Option<&str>,
) -> Array2<f64> {
    let n = mol.atoms.len();
    let mut g = Array2::<f64>::zeros((n, 3));
    for atom in 0..n {
        for coord in 0..3 {
            let e = |h: f64| {
                let m = displaced(mol, atom, coord, h);
                solve(&m, obs, aux, jk_aux).oo.total_energy
            };
            let d = |h: f64| (e(h) - e(-h)) / (2.0 * h);
            g[(atom, coord)] = (4.0 * d(H) - d(2.0 * H)) / 3.0;
        }
    }
    g
}

/// `max |Σ_A g_A|` and `max |Σ_A r_A × g_A|` (about the origin, Bohr).
fn invariance_residuals(mol: &Molecule, g: &Array2<f64>) -> (f64, f64) {
    let mut f = [0.0f64; 3];
    let mut t = [0.0f64; 3];
    for (a, at) in mol.atoms.iter().enumerate() {
        let r = [at.x, at.y, at.zpos];
        let ga = [g[(a, 0)], g[(a, 1)], g[(a, 2)]];
        for k in 0..3 {
            f[k] += ga[k];
        }
        t[0] += r[1] * ga[2] - r[2] * ga[1];
        t[1] += r[2] * ga[0] - r[0] * ga[2];
        t[2] += r[0] * ga[1] - r[1] * ga[0];
    }
    let m = |v: [f64; 3]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    (m(f), m(t))
}

/// The gradient of the same point with the converged orbitals rotated among
/// the occupied block by a fixed non-trivial orthogonal matrix: the energy,
/// and so the gradient, is invariant to it.
fn rotate_occupied(mol: &Molecule, s: &Solved) -> Array2<f64> {
    let nocc = (mol.nelec() / 2) as usize;
    let mut rot = s.oo.clone();
    let c = &s.oo.mos;
    let mut c2 = c.clone();
    // Givens rotation by 0.7 rad in the (0, 1) occupied plane and 0.3 rad in
    // the (1, 2) plane (needs nocc >= 3).
    assert!(nocc >= 3);
    for (i, j, th) in [(0usize, 1usize, 0.7f64), (1, 2, 0.3)] {
        let (cs, sn) = (th.cos(), th.sin());
        for mu in 0..c2.nrows() {
            let (a, b) = (c2[(mu, i)], c2[(mu, j)]);
            c2[(mu, i)] = cs * a - sn * b;
            c2[(mu, j)] = sn * a + cs * b;
        }
    }
    rot.mos = c2;
    analytic_with(mol, s, &rot)
}

fn check_case(label: &str, xyz: &str, basis_name: &str, jk_aux: Option<&str>) -> Array2<f64> {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = basis::bundled(basis_name).unwrap();
    let aux = basis::bundled(AUX).unwrap();
    let s = solve(&mol, &obs, &aux, jk_aux);
    let g = analytic(&mol, &s);
    let fd = richardson_fd(&mol, &obs, &aux, jk_aux);
    let worst = max_abs_diff(&g, &fd);
    let (f, t) = invariance_residuals(&mol, &g);
    eprintln!(
        "{label}: OO |g| {:.2e}; max |analytic - FD| = {worst:.3e} (bar {BAR:.0e}); \
         max|sum g| {f:.2e}, max|sum r x g| {t:.2e}; max|g| {:.3e}",
        s.oo.grad_norm,
        g.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    );
    assert!(worst < BAR, "{label}: max |analytic - FD| = {worst:.3e}");
    assert!(f < TOL_INV, "{label}: translational residual {f:.2e}");
    assert!(t < TOL_INV, "{label}: rotational residual {t:.2e}");
    g
}

#[test]
fn oo_gradient_matches_fd_water_ccpvdz_rijk() {
    check_case("H2O/cc-pVDZ RI-JK", WATER, "cc-pvdz", Some(JKFIT));
}

#[test]
fn oo_gradient_matches_fd_ammonia_631g_rijk() {
    check_case("NH3/6-31G RI-JK", AMMONIA, "6-31g", Some(JKFIT));
}

#[test]
fn oo_gradient_matches_fd_ammonia_631g_exact_jk() {
    check_case("NH3/6-31G exact J/K", AMMONIA, "6-31g", None);
}

/// TEETH for the route: the fitted and four-centre gradients at one geometry
/// differ by more than ten bars (a route-blind gradient cannot pass the RI-JK
/// FD), and the gradient is invariant to occupied-occupied rotation of the
/// converged orbitals.
#[test]
fn oo_gradient_route_and_rotation_controls_ammonia_631g() {
    let mol = Molecule::parse_xyz(AMMONIA, 0, 1).unwrap();
    let obs = basis::bundled("6-31g").unwrap();
    let aux = basis::bundled(AUX).unwrap();
    let fitted = solve(&mol, &obs, &aux, Some(JKFIT));
    let exact = solve(&mol, &obs, &aux, None);
    let (gf, ge) = (analytic(&mol, &fitted), analytic(&mol, &exact));
    let fit_effect = max_abs_diff(&gf, &ge);
    let rot = max_abs_diff(&rotate_occupied(&mol, &fitted), &gf);
    eprintln!("NH3/6-31G: max|fitted - four-centre| = {fit_effect:.3e}; occ-rotation = {rot:.2e}");
    assert!(fit_effect > 10.0 * BAR, "fit effect {fit_effect:.3e}");
    assert!(rot < 1e-10, "occupied-rotation change {rot:.2e}");
}
