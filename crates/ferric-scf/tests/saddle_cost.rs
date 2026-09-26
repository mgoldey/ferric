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
//!     n_hessian * (6N + 1)  +  (n_steps + 1)
//!
//! because the Hessian is central-differenced from analytic gradients: 6N
//! DISPLACED evaluations (MEASURED via `n_gradient_evaluations`: H2 = 12,
//! water = 18) plus ONE at the undisplaced geometry, which
//! `harmonic_frequencies` takes before the loop and which
//! `n_gradient_evaluations` does not count. `find_saddle` likewise takes one
//! gradient before its first step.
//!
//! The `+1`s are small but they are not rounding: this test exists to be the
//! cost model other code quotes, and a model that is wrong by a constant is
//! the kind of thing that gets multiplied by a candidate count later. An
//! earlier version of this comment said `2 * 6N + n_steps` and undercounted a
//! default search by 3 gradient evaluations.
//!
//! With `hessian_recalc_every = 0` (the default) there are exactly two
//! Hessians: one at the start and one at the end for the character check.
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

    // N = 2 here, so one Hessian is 6N + 1 = 13 gradient-equivalents: 6N
    // displaced, plus the undisplaced one `harmonic_frequencies` takes before
    // its loop (which `n_gradient_evaluations` does NOT report).
    let n_atoms = 2;
    let per_hessian = 6 * n_atoms + 1;
    let total = h * per_hessian + g;
    println!(
        "MEASURED P-RFO cost on the analytic inverted parabola: {g} gradient \
         calls + {h} Hessians ({per_hessian} = 6N+1 each at N={n_atoms}) = \
         {total} gradient-equivalents, {} steps",
        res.steps
    );
    println!(
        "  SCALING: a search on N atoms costs 2*(6N+1) + (n_steps+1) gradient \
         calls. At N=20 that is 242 + n_steps + 1, so the two Hessians dominate \
         until n_steps exceeds ~240. This is why hessian_recalc_every \
         defaults to 0."
    );
}

/// The `6N + 1` in the cost model, measured against the REAL finite-difference
/// Hessian rather than the analytic closure the test above uses.
///
/// This is the gap that let the `+1` go unnoticed for a whole PR.
/// `prfo_cost_is_measured_...` supplies its own analytic Hessian, so it never
/// touches `harmonic_frequencies` and cannot see what a real Hessian costs.
/// The documented cost model describes the FD path; nothing measured it.
///
/// What is asserted, and why it is two separate things:
///
/// 1. `n_gradient_evaluations == 6N`. That is the DISPLACED count -- the
///    counter is initialised to zero AFTER the undisplaced
///    `energy_and_gradient` at the top of `harmonic_frequencies`.
/// 2. The undisplaced call HAPPENS, evidenced by `energy` being a real
///    converged number. `FrequencyResult::energy` is the energy AT THE
///    UNDISPLACED GEOMETRY and can only come from that call, so a finite value
///    witnesses a gradient evaluation the counter does not report.
///
/// Together: reported 6N, true cost 6N + 1. If someone later folds the
/// undisplaced call into the counter, (1) fails and the cost model gets
/// revisited -- which is the point.
#[test]
fn a_finite_difference_hessian_costs_6n_plus_one_gradients() {
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::operator::Operator;
    use ferric_scf::frequencies::{harmonic_frequencies, FrequencyConfig, HessianMethod};
    use ferric_scf::rhf::RhfConfig;

    let mol = h2(0.74);
    let n_atoms = mol.atoms.len();
    let ctx = ParallelContext::new();

    let r = harmonic_frequencies(
        &ctx,
        &mol,
        "sto-3g",
        Operator::coulomb(),
        &RhfConfig::default(),
        // The 6N count is a property of the finite-difference Hessian.
        &FrequencyConfig {
            hessian: HessianMethod::FiniteDifference,
            ..Default::default()
        },
    )
    .expect("H2/STO-3G harmonic frequencies");

    assert_eq!(
        r.n_gradient_evaluations,
        6 * n_atoms,
        "n_gradient_evaluations reports the DISPLACED count, 6N = {}",
        6 * n_atoms
    );
    assert!(
        r.energy.is_finite() && r.energy < 0.0,
        "the undisplaced energy must be a real converged number -- it is the \
         witness that the uncounted 6N+1'th gradient evaluation happened"
    );
    println!(
        "MEASURED: N={n_atoms}, reported {} displaced gradients, true Hessian \
         cost {} (6N+1) -- the undisplaced call is not counted",
        r.n_gradient_evaluations,
        r.n_gradient_evaluations + 1
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

/// The trans/rot shift must stay FAR above every physical eigenvalue.
///
/// `projected_eigen` lifts the 5-6 translation/rotation modes to
/// `TR_SHIFT = 1.0e3`, and `count_negative` then treats the top `n_tr` slots as
/// shifted. That is only safe while no PHYSICAL mode can reach 1000, and
/// nothing in the code enforces it -- it is an assumption about chemistry
/// wearing the appearance of an index guard.
///
/// It is also convention-dependent. `saddle.rs` mass-weights with `atom_masses`
/// output directly, which is in AMU; `frequencies.rs` converts to electron
/// masses first (`AMU_TO_ELECTRON_MASS`, a factor of 1822.888). A uniform mass
/// rescale cannot change eigenvalue SIGNS, ORDER or mode directions -- so it
/// changes no decision `saddle.rs` makes -- but it does move every eigenvalue
/// relative to the absolute `TR_SHIFT`, and the ELECTRON-mass convention would
/// push them 1823x SMALLER, i.e. further from the shift. The AMU convention in
/// use is therefore the less safe of the two, which is the one worth measuring.
///
/// MEASURED (amu convention, the one `saddle.rs` uses), max eigenvalue of the
/// mass-weighted Hessian at STO-3G:
///
/// ```text
///   H2    0.9613      H2O   0.8706      HF    0.9310
/// ```
///
/// Three orders of magnitude of headroom. This test pins that the headroom
/// EXISTS rather than pinning the numbers, so a future stiffer system or a
/// changed mass convention fails here with an explanation instead of silently
/// miscounting imaginary modes.
#[test]
fn physical_eigenvalues_stay_far_below_the_trans_rot_shift() {
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::operator::Operator;
    use ferric_scf::frequencies::{
        atom_masses, harmonic_frequencies, FrequencyConfig, FrequencyReference,
    };
    use ferric_scf::rhf::RhfConfig;
    use ndarray_linalg::{Eigh, UPLO};

    /// Mirrors `saddle.rs`'s private `TR_SHIFT`. If that constant changes this
    /// must change with it -- which is the point of stating it here.
    const TR_SHIFT: f64 = 1.0e3;
    /// Demand two orders of magnitude, not a hair's breadth.
    const REQUIRED_MARGIN: f64 = 100.0;

    let op = Operator::coulomb();
    for (tag, xyz) in [
        ("H2", "2\nh2\nH 0 0 0\nH 0 0 0.74\n"),
        (
            "H2O",
            "3\nw\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n",
        ),
        // HF: the stiffest bond available at this scale, so the worst case for
        // the margin among small closed-shell molecules.
        ("HF", "2\nhf\nH 0 0 0\nF 0 0 0.92\n"),
    ] {
        let mol = Molecule::parse_xyz(xyz, 0, 1).expect("parse");
        let mut scf = RhfConfig::default();
        scf.max_iter = 300;
        let fc = FrequencyConfig {
            reference: FrequencyReference::Rhf,
            hessian: ferric_scf::frequencies::HessianMethod::FiniteDifference,
            ..Default::default()
        };
        let fr = harmonic_frequencies(&ParallelContext::default(), &mol, "sto-3g", op, &scf, &fc)
            .expect("frequencies");
        let m = atom_masses(&mol).expect("masses");
        let n = 3 * mol.atoms.len();
        let mut hw = fr.cartesian_hessian.clone();
        for i in 0..n {
            for j in 0..n {
                hw[[i, j]] /= (m[i / 3] * m[j / 3]).sqrt();
            }
        }
        let (evals, _) = hw.eigh(UPLO::Lower).expect("eigh");
        let max = evals.iter().cloned().fold(f64::MIN, f64::max);
        println!("{tag:4} max mass-weighted eigenvalue = {max:.4} (TR_SHIFT = {TR_SHIFT:.0})");
        assert!(
            max * REQUIRED_MARGIN < TR_SHIFT,
            "{tag}: the largest physical eigenvalue is {max:.4}, within {REQUIRED_MARGIN}x \
             of TR_SHIFT = {TR_SHIFT:.0}. count_negative assumes shifted modes occupy the \
             TOP n_tr slots; if a physical mode can reach the shift, that assumption \
             breaks and imaginary modes are miscounted."
        );
    }
}
