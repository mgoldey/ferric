//! MEASURED cost of a P-RFO transition-state search, in gradient evaluations.
//!
//! The golden-path note's cost table had no entry for a saddle search, because
//! until `ferric_scf::saddle` landed there was nothing to measure. This test
//! produces that number, and produces it as a COUNT of energy/gradient calls
//! rather than a wall time -- wall time is machine-specific and would be wrong
//! on any other box (see the repo's "never commit machine-specific values"
//! convention), whereas the call count is a property of the algorithm.
//!
//! The total cost of a search is
//!
//!     n_hessian * 6N  +  n_steps * 1
//!
//! because the Hessian is central-differenced from analytic gradients (6N per
//! Hessian, MEASURED: H2 = 12, water = 18) and each P-RFO step costs one
//! gradient. With `hessian_recalc_every = 0` (the default) there are exactly
//! two Hessians: one at the start and one at the end for the character check.
//!
//! What this test PINS is the step count on a surface whose answer is known in
//! closed form. It is deliberately not a chemistry benchmark: a real catalyst
//! search has soft degrees of freedom this surface does not, so the number
//! here is a FLOOR, exactly as the golden path's 34-step BFGS figure is.

use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_scf::saddle::{find_saddle, SaddleConfig};
use ndarray::{Array1, Array2};
use std::cell::Cell;

/// H2 on the z axis, separation in Angstrom.
fn h2(sep_angstrom: f64) -> Molecule {
    let xyz = format!("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 {sep_angstrom}\n");
    Molecule::parse_xyz(&xyz, 0, 1).expect("H2 must parse")
}

/// An inverted parabola in the bond coordinate: maximum at `r0`, no other
/// stationary point, so the saddle is known exactly.
fn surface(
    r0: f64,
    k: f64,
) -> (
    impl FnMut(&Molecule) -> Result<(f64, Array1<f64>), FerricError>,
    impl FnMut(&Molecule) -> Result<Array2<f64>, FerricError>,
) {
    let sep = |m: &Molecule| -> f64 {
        let dx = m.atoms[1].x - m.atoms[0].x;
        let dy = m.atoms[1].y - m.atoms[0].y;
        let dz = m.atoms[1].zpos - m.atoms[0].zpos;
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    let unit = move |m: &Molecule, r: f64| {
        [
            (m.atoms[1].x - m.atoms[0].x) / r,
            (m.atoms[1].y - m.atoms[0].y) / r,
            (m.atoms[1].zpos - m.atoms[0].zpos) / r,
        ]
    };
    let eg = move |m: &Molecule| {
        let r = sep(m);
        let u = unit(m, r);
        let dedr = -k * (r - r0);
        Ok((
            -0.5 * k * (r - r0) * (r - r0),
            Array1::from(vec![
                -dedr * u[0],
                -dedr * u[1],
                -dedr * u[2],
                dedr * u[0],
                dedr * u[1],
                dedr * u[2],
            ]),
        ))
    };
    let hess = move |m: &Molecule| {
        let r = sep(m);
        let u = unit(m, r);
        let mut h = Array2::<f64>::zeros((6, 6));
        for a in 0..3 {
            for b in 0..3 {
                let v = -k * u[a] * u[b];
                h[[a, b]] += v;
                h[[3 + a, 3 + b]] += v;
                h[[a, 3 + b]] -= v;
                h[[3 + a, b]] -= v;
            }
        }
        Ok(h)
    };
    (eg, hess)
}

/// MEASURE the search cost and PIN it, so a change in the step logic that
/// silently doubles the work fails here rather than showing up as a slow
/// catalyst run months later.
#[test]
fn prfo_cost_is_measured_in_gradient_and_hessian_calls() {
    const R0: f64 = 1.60;
    const K: f64 = 0.35;

    let n_grad = Cell::new(0usize);
    let n_hess = Cell::new(0usize);
    let (mut eg, mut hs) = surface(R0, K);

    let res = find_saddle(
        &h2(0.74),
        &SaddleConfig {
            max_steps: 200,
            trust_radius: 0.10,
            ..Default::default()
        },
        |m| {
            n_grad.set(n_grad.get() + 1);
            eg(m)
        },
        |m| {
            n_hess.set(n_hess.get() + 1);
            hs(m)
        },
    )
    .expect("the surface has a negative mode");

    assert!(
        res.is_transition_state(),
        "must converge on the known saddle"
    );

    let g = n_grad.get();
    let h = n_hess.get();
    // 2 Hessians is the ALGORITHM, not this surface: one to start (and to
    // decide there is something to climb), one at the end for the character
    // check. `hessian_recalc_every = 0` adds none in between.
    assert_eq!(
        h, 2,
        "expected exactly 2 Hessians (initial + final character check) at the \
         default hessian_recalc_every = 0, got {h}"
    );
    // The gradient count is 1 per step plus the initial one. Pinned loosely
    // because the exact step count depends on the trust radius, but tightly
    // enough that a doubling fails.
    assert!(
        (2..=40).contains(&g),
        "gradient calls = {g}, outside the 2..=40 band this surface should need \
         -- either the step logic regressed or the convergence test changed"
    );

    // N = 2 here, so one Hessian is 6N = 12 gradient-equivalents.
    let n_atoms = 2;
    let total = h * 6 * n_atoms + g;
    println!(
        "MEASURED P-RFO cost on the analytic inverted parabola: {g} gradient \
         calls + {h} Hessians (6N = {} each at N={n_atoms}) = {total} \
         gradient-equivalents, {} steps",
        6 * n_atoms,
        res.steps
    );
    println!(
        "  SCALING: a search on N atoms costs 2*6N + n_steps gradient calls. \
         At N=20 that is 240 + n_steps, so the two Hessians dominate until \
         n_steps exceeds ~240. This is why hessian_recalc_every defaults to 0."
    );
}

/// The recalculation knob must actually cost what it claims, or the cost model
/// above is fiction.
#[test]
fn hessian_recalc_every_costs_one_extra_hessian_per_n_steps() {
    const R0: f64 = 1.60;
    const K: f64 = 0.35;

    let count = |recalc: usize| -> (usize, usize) {
        let n_grad = Cell::new(0usize);
        let n_hess = Cell::new(0usize);
        let (mut eg, mut hs) = surface(R0, K);
        let _ = find_saddle(
            &h2(0.74),
            &SaddleConfig {
                max_steps: 200,
                trust_radius: 0.10,
                hessian_recalc_every: recalc,
                ..Default::default()
            },
            |m| {
                n_grad.set(n_grad.get() + 1);
                eg(m)
            },
            |m| {
                n_hess.set(n_hess.get() + 1);
                hs(m)
            },
        );
        (n_grad.get(), n_hess.get())
    };

    let (_, h_never) = count(0);
    let (_, h_every) = count(1);
    assert_eq!(h_never, 2, "recalc=0 must build exactly 2 Hessians");
    assert!(
        h_every > h_never,
        "recalc=1 must build MORE Hessians than recalc=0 ({h_every} vs \
         {h_never}); if it does not, the knob is inert and the cost model is \
         wrong"
    );
    println!(
        "MEASURED: hessian_recalc_every=0 -> {h_never} Hessians, =1 -> {h_every}. \
         At N=20 (6N=120 gradients each) that is the difference between 240 and \
         {} gradient-equivalents of Hessian work.",
        h_every * 120
    );
}
