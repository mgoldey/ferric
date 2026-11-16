//! The two-body D3(BJ) analytic Hessian against central differences of the
//! analytic gradient (`d3bj_gradient`, itself checked against FD of the energy
//! in `gradient_vs_fd.rs` and against simple-dftd3 in `vs_reference_dftd3*.rs`).
//!
//! ## Artifact hypothesis, written before running
//!
//! The Hessian has four parts: the direct pair block at fixed `C6`, the mixed
//! `r`/`CN` term, the `CN`-`CN` term (`∂²C6/∂CN²` and `∂²C6/∂CN_i∂CN_j`) and
//! the `e_a ∂²CN_a` term (`theory.md`, "Hessian").
//!
//! * All four right: the FD error falls as `h²` (ratio ~4 per halving) down to
//!   the derived truncation bar, on every system.
//! * A CN part missing or mis-weighted: the error stalls at the size of the
//!   missing part, so the `h²` ratio collapses to ~1 and the bar fails. The
//!   CN parts vanish for a dimer of atoms with CN ~ 0 (Ar2), so the systems
//!   run from Ar2 up to mixed-element, mixed-coordination fragments (the
//!   lesson of the gradient's chain rule, `gradient_vs_fd.rs`).
//!
//! ## Derived bar
//!
//! `H_fd(h) = H + (h²/6) T + O(h⁴)` with `T` the third derivatives of the
//! energy. With `Δ = H_fd(h) - H_fd(h/2) = (h²/8) T + O(h⁴)`, the truncation
//! error of `H_fd(h/2)` is `Δ/3` to leading order. Round-off: every gradient
//! element carries an absolute error `ε_g ≤ 64 ε max|g|`-sized terms; the FD
//! column at step `h/2` carries `ε_g/(h/2)`. The bar on `max|H - H_fd(h/2)|`
//! is twice their sum. It is computed per system from the data, not guessed.

use ferric_d3::{d3bj_gradient, d3bj_hessian, d3bj_hessian_for_molecule, D3Params};

/// B3LYP's D3(BJ) parameters (simple-dftd3 1.6.0).
fn params() -> D3Params {
    D3Params {
        s6: 1.0,
        s8: 1.9889,
        a1: 0.3981,
        a2: 4.4211,
        s9: 0.0,
    }
}

type System = (Vec<u8>, Vec<[f64; 3]>);

fn ar2() -> System {
    (vec![18, 18], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 7.1]])
}

fn water() -> System {
    (
        vec![8, 1, 1],
        vec![
            [0.0, 0.0, 0.2217],
            [0.0, 1.4309, -0.8867],
            [0.0, -1.4309, -0.8867],
        ],
    )
}

/// Coordination numbers ~2 (central C) to ~0.5 (H); see `gradient_vs_fd.rs`.
fn branched() -> System {
    (
        vec![6, 6, 6, 6, 1, 1, 1, 1],
        vec![
            [0.0, 0.0, 0.0],
            [2.9, 0.0, 0.0],
            [-1.45, 2.51, 0.0],
            [-1.45, -2.51, 0.0],
            [0.0, 0.0, 2.07],
            [3.9, 1.7, 0.0],
            [-2.45, 2.9, 1.7],
            [-2.45, -2.9, -1.7],
        ],
    )
}

/// Mixed elements and mixed coordination (C/F/O/H/N).
fn mixed() -> System {
    (
        vec![6, 9, 9, 8, 1, 7],
        vec![
            [0.0, 0.0, 0.0],
            [2.5, 0.3, 0.1],
            [-1.2, 2.2, -0.3],
            [-1.0, -1.1, 2.3],
            [-0.5, -1.9, -1.6],
            [1.1, -2.6, 3.4],
        ],
    )
}

fn systems() -> Vec<(&'static str, System)> {
    vec![
        ("Ar2", ar2()),
        ("water", water()),
        ("branched", branched()),
        ("mixed", mixed()),
    ]
}

/// Column `b` of the Hessian is the central difference of the gradient along
/// coordinate `b`. Returns the FD Hessian and `max|g|` over the displaced
/// gradients (for the round-off term).
fn fd_hessian(numbers: &[u8], coords: &[[f64; 3]], h: f64) -> (Vec<Vec<f64>>, f64) {
    let p = params();
    let n3 = 3 * numbers.len();
    let mut out = vec![vec![0.0; n3]; n3];
    let mut gmax = 0.0f64;
    for b in 0..n3 {
        let mut plus = coords.to_vec();
        let mut minus = coords.to_vec();
        plus[b / 3][b % 3] += h;
        minus[b / 3][b % 3] -= h;
        let gp = d3bj_gradient(numbers, &plus, &p).unwrap();
        let gm = d3bj_gradient(numbers, &minus, &p).unwrap();
        for a in 0..n3 {
            let (x, y) = (gp[a / 3][a % 3], gm[a / 3][a % 3]);
            gmax = gmax.max(x.abs()).max(y.abs());
            out[a][b] = (x - y) / (2.0 * h);
        }
    }
    (out, gmax)
}

fn max_diff(h: &ndarray::Array2<f64>, fd: &[Vec<f64>]) -> f64 {
    let mut m = 0.0f64;
    for (a, row) in fd.iter().enumerate() {
        for (b, v) in row.iter().enumerate() {
            m = m.max((h[(a, b)] - v).abs());
        }
    }
    m
}

/// FD step (Bohr) of the comparison; the second step is `H_STEP / 2`.
const H_STEP: f64 = 2e-3;

#[test]
fn analytic_hessian_matches_fd_of_the_analytic_gradient_within_the_derived_bar() {
    let p = params();
    for (name, (z, x)) in systems() {
        let h = d3bj_hessian(&z, &x, &p).unwrap();
        let (fd1, _) = fd_hessian(&z, &x, H_STEP);
        let (fd2, gmax) = fd_hessian(&z, &x, 0.5 * H_STEP);
        let delta = fd1
            .iter()
            .zip(&fd2)
            .flat_map(|(r1, r2)| r1.iter().zip(r2).map(|(a, b)| (a - b).abs()))
            .fold(0.0f64, f64::max);
        let roundoff = 64.0 * f64::EPSILON * gmax / (0.5 * H_STEP);
        let bar = 2.0 * (delta / 3.0 + roundoff);
        let (e1, e2) = (max_diff(&h, &fd1), max_diff(&h, &fd2));
        let hmax = h.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        eprintln!(
            "{name}: max|H| {hmax:.3e}; |H - FD(h)| {e1:.3e}, |H - FD(h/2)| {e2:.3e} \
             (ratio {:.2}); bar {bar:.3e} (truncation {:.3e}, round-off {roundoff:.1e})",
            e1 / e2,
            delta / 3.0
        );
        assert!(e2 < bar, "{name}: |H - FD(h/2)| = {e2:.3e} > bar {bar:.3e}");
        // O(h²): halving the step cuts the error ~4x while truncation dominates.
        if e2 > 10.0 * roundoff {
            assert!(
                (3.0..5.0).contains(&(e1 / e2)),
                "{name}: error ratio {:.2} is not the O(h²) 4",
                e1 / e2
            );
        }
    }
}

/// Symmetry and `Σ_b H[(a, 3b + y)] = 0` (moving every atom by the same
/// vector changes nothing) hold term by term, so a wrong sign or a dropped
/// transpose in one part shows here without any FD.
#[test]
fn the_hessian_is_symmetric_and_translationally_invariant() {
    let p = params();
    for (name, (z, x)) in systems() {
        let h = d3bj_hessian(&z, &x, &p).unwrap();
        let n3 = h.nrows();
        let hmax = h.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut asym = 0.0f64;
        let mut trans = 0.0f64;
        for a in 0..n3 {
            for b in 0..n3 {
                asym = asym.max((h[(a, b)] - h[(b, a)]).abs());
            }
            for y in 0..3 {
                let s: f64 = (0..n3 / 3).map(|bat| h[(a, 3 * bat + y)]).sum();
                trans = trans.max(s.abs());
            }
        }
        eprintln!("{name}: asymmetry {asym:.2e}, translation sum {trans:.2e}, max|H| {hmax:.2e}");
        // Both are identities; what is left is the round-off of O(N²) sums.
        assert!(
            asym < 1e-14 * hmax.max(1e-3),
            "{name}: asymmetry {asym:.3e}"
        );
        assert!(
            trans < 1e-13 * hmax.max(1e-3),
            "{name}: translation {trans:.3e}"
        );
    }
}

#[test]
fn fewer_than_two_atoms_has_an_exactly_zero_hessian() {
    let p = params();
    assert_eq!(d3bj_hessian(&[], &[], &p).unwrap().dim(), (0, 0));
    let h = d3bj_hessian(&[6], &[[0.1, 0.2, 0.3]], &p).unwrap();
    assert!(h.iter().all(|v| *v == 0.0));
}

#[test]
fn the_three_body_term_is_refused_not_dropped() {
    let p = D3Params {
        s9: 1.0,
        ..params()
    };
    let (z, x) = water();
    let e = d3bj_hessian(&z, &x, &p).unwrap_err().to_string();
    assert!(e.contains("three-body"), "{e}");
}

#[test]
fn ghost_rows_and_columns_are_zero_and_real_blocks_are_not_shifted() {
    use ferric_core::mol::Molecule;
    let p = params();
    // The ghost FIRST, so a filtered index scattered back unshifted is off by one.
    let with_ghost = Molecule::parse_xyz(
        "4\nghost-first\n@He 2.0 2.0 2.0\nO 0.0 0.0 0.1173\n\
         H 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .unwrap();
    let plain = Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .unwrap();
    let full = d3bj_hessian_for_molecule(&with_ghost, &p).unwrap();
    let bare = d3bj_hessian_for_molecule(&plain, &p).unwrap();
    assert_eq!(full.dim(), (12, 12));
    for a in 0..12 {
        for b in 0..12 {
            let v = full[(a, b)];
            if a < 3 || b < 3 {
                assert_eq!(v, 0.0, "ghost row/column ({a}, {b})");
            } else {
                assert!((v - bare[(a - 3, b - 3)]).abs() < 1e-15, "({a}, {b})");
            }
        }
    }
    assert!(
        bare.iter().any(|v| v.abs() > 1e-6),
        "water's D3 Hessian is ~0"
    );
}
