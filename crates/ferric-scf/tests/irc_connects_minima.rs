//! The IRC must land in the two minima the saddle actually separates.
//!
//! `find_saddle` + `n_imaginary == 1` proves a geometry IS a first-order
//! saddle. It does NOT prove it is YOUR saddle: a molecule has many, the one
//! found is whichever was uphill from the start, and a methyl rotor gives one
//! imaginary mode exactly as a bond-breaking coordinate does. Until the IRC
//! landed, `harmonic_frequencies` could confirm a TS and nothing could say
//! what reaction it was the TS OF.
//!
//! ## The test system, and why it can tell right from wrong
//!
//! NH3 umbrella inversion. The planar D3h form is the saddle; the two minima
//! are the pyramidal structures with nitrogen above and below the H3 plane.
//! That gives an unambiguous, SIGNED observable -- the nitrogen's displacement
//! from the plane -- which must come out with OPPOSITE SIGNS on the two
//! branches.
//!
//! A test that only checked "both endpoints are lower than the saddle" would
//! pass for a walk that went the same way twice, which is the most likely way
//! to get the direction handling wrong.
//!
//! ## What is NOT asserted
//!
//! That the endpoints are stationary points. The walk stops on a gradient
//! threshold, and confirming a minimum needs a Hessian there (another 6N+1
//! gradients per side). `IrcBranch::converged` reports which stopping
//! condition fired, and the tests below assert it rather than assuming it.

use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::irc::{follow_irc, IrcConfig};
use ferric_scf::qmmm::{QmSelection, QmmmAtom, QmmmSystem};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array1;

const R_NH: f64 = 1.006;

/// Planar NH3: the umbrella-inversion transition state.
fn planar_nh3() -> Molecule {
    let a: f64 = 120.0_f64.to_radians();
    Molecule::parse_xyz(
        &format!(
            "4\nplanar NH3\nN 0.0 0.0 0.0\nH {:.6} 0.0 0.0\nH {:.6} {:.6} 0.0\n\
             H {:.6} {:.6} 0.0\n",
            R_NH,
            R_NH * a.cos(),
            R_NH * a.sin(),
            R_NH * (2.0 * a).cos(),
            R_NH * (2.0 * a).sin()
        ),
        0,
        1,
    )
    .expect("parse planar NH3")
}

/// The umbrella mode: N moves along +z, the three H along -z so the centre of
/// mass stays put. Normalisation does not matter -- `follow_irc` renormalises.
fn umbrella_mode() -> Array1<f64> {
    Array1::from_vec(vec![
        0.0, 0.0, 1.0, // N
        0.0, 0.0, -0.2, // H
        0.0, 0.0, -0.2, // H
        0.0, 0.0, -0.2, // H
    ])
}

/// Nitrogen's signed height above the plane of the three hydrogens.
fn pyramidalisation(mol: &Molecule) -> f64 {
    let h_mean: f64 = mol.atoms[1..].iter().map(|a| a.zpos).sum::<f64>() / 3.0;
    mol.atoms[0].zpos - h_mean
}

fn gas_phase_eg() -> impl FnMut(&Molecule) -> Result<(f64, Array1<f64>), FerricError> {
    let op = Operator::coulomb();
    move |m: &Molecule| {
        let bs = ferric_core::basis::bundled("sto-3g")?;
        let prep = PreparedBasis::new(m, &bs)?;
        let b = SchwarzBounds::compute(op, &prep)?;
        let mut scf = RhfConfig::default();
        scf.max_iter = 300;
        let res = solve_rhf(&ParallelContext::default(), m, &prep, op, &b, &scf)?;
        let g = rhf_gradient(m, &prep, op, &b, &res, None)?;
        Ok((res.energy, Array1::from_iter(g.iter().copied())))
    }
}

#[test]
fn the_two_branches_land_on_opposite_sides_of_the_plane() {
    let saddle = planar_nh3();
    assert!(
        pyramidalisation(&saddle).abs() < 1e-9,
        "the starting geometry must be planar, or this test measures nothing"
    );

    let res = follow_irc(
        &saddle,
        &umbrella_mode(),
        &IrcConfig {
            step: 0.15,
            max_steps: 120,
            ..Default::default()
        },
        gas_phase_eg(),
    )
    .expect("IRC from the planar saddle");

    let pf = pyramidalisation(&res.forward.mol);
    let pr = pyramidalisation(&res.reverse.mol);
    println!(
        "forward: pyramidalisation {pf:+.4} Bohr, E = {:.8}, {} steps, converged={}\n\
         reverse: pyramidalisation {pr:+.4} Bohr, E = {:.8}, {} steps, converged={}\n\
         saddle E = {:.8}",
        res.forward.energy,
        res.forward.steps,
        res.forward.converged,
        res.reverse.energy,
        res.reverse.steps,
        res.reverse.converged,
        res.saddle_energy
    );

    // THE assertion: opposite signs. A walk that went the same way twice --
    // the most likely direction bug -- fails here and passes every
    // energy-based check.
    assert!(
        pf * pr < 0.0,
        "the two IRC branches must end on OPPOSITE sides of the H3 plane; got \
         {pf:+.4} and {pr:+.4} Bohr. Same sign means both walks followed the \
         same direction."
    );
    assert!(
        pf.abs() > 0.05 && pr.abs() > 0.05,
        "both branches must actually pyramidalise; got {pf:+.4} and {pr:+.4}"
    );

    // ...and by symmetry the two endpoints are mirror images, so their
    // energies must agree. This catches a branch that wandered off the
    // umbrella coordinate entirely.
    assert!(
        (res.forward.energy - res.reverse.energy).abs() < 1e-6,
        "NH3's two pyramidal minima are mirror images and must be degenerate; \
         got {} and {}",
        res.forward.energy,
        res.reverse.energy
    );
}

#[test]
fn both_endpoints_are_downhill_from_the_saddle() {
    let res = follow_irc(
        &planar_nh3(),
        &umbrella_mode(),
        &IrcConfig {
            step: 0.15,
            max_steps: 120,
            ..Default::default()
        },
        gas_phase_eg(),
    )
    .expect("IRC");

    // A barrier is saddle - endpoint, so it must be POSITIVE in both
    // directions. Negative would mean the walk climbed, i.e. the gradient sign
    // is inverted somewhere.
    assert!(
        res.forward_barrier() > 0.0,
        "forward barrier is {:.3e} Ha -- the endpoint is ABOVE the saddle",
        res.forward_barrier()
    );
    assert!(
        res.reverse_barrier() > 0.0,
        "reverse barrier is {:.3e} Ha -- the endpoint is ABOVE the saddle",
        res.reverse_barrier()
    );
    println!(
        "barrier: forward {:.6} Ha, reverse {:.6} Ha",
        res.forward_barrier(),
        res.reverse_barrier()
    );
}

/// The IRC must run under QM/MM embedding, on the same surface the saddle
/// search used.
///
/// Wiring it separately would let the search and the path disagree about the
/// field, which is the one way to get two endpoints that do not belong to the
/// saddle you found. Nothing new is needed: `follow_irc` takes the same
/// closure as `find_saddle`, so the embedding enters through `RhfConfig`
/// exactly as it does there.
#[test]
fn the_irc_runs_under_qmmm_embedding() {
    let a: f64 = 120.0_f64.to_radians();
    // Symmetric charges about the H3 plane. An ANTISYMMETRIC pair applies a
    // constant force along z and destroys the reflection symmetry the saddle
    // is defined by -- see qmmm_saddle_converges.rs.
    let atoms = vec![
        QmmmAtom::new("N", 7, 0.0, 0.0, 0.0, 0.0),
        QmmmAtom::new("H", 1, R_NH, 0.0, 0.0, 0.0),
        QmmmAtom::new("H", 1, R_NH * a.cos(), R_NH * a.sin(), 0.0, 0.0),
        QmmmAtom::new(
            "H",
            1,
            R_NH * (2.0 * a).cos(),
            R_NH * (2.0 * a).sin(),
            0.0,
            0.0,
        ),
        QmmmAtom::new("X", 0, 0.0, 0.0, 6.0, -0.2),
        QmmmAtom::new("X", 0, 0.0, 0.0, -6.0, -0.2),
    ];
    let system =
        QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2, 3]), 0, 1).expect("qmmm system");
    let mut scf = RhfConfig {
        external_potential: system.to_external_potential(),
        ..Default::default()
    };
    scf.max_iter = 300;
    let op = Operator::coulomb();

    let eg = move |m: &Molecule| -> Result<(f64, Array1<f64>), FerricError> {
        let bs = ferric_core::basis::bundled("sto-3g")?;
        let prep = PreparedBasis::new(m, &bs)?;
        let b = SchwarzBounds::compute(op, &prep)?;
        let res = solve_rhf(&ParallelContext::default(), m, &prep, op, &b, &scf)?;
        let g = rhf_gradient(m, &prep, op, &b, &res, scf.external_potential.as_ref())?;
        Ok((res.energy, Array1::from_iter(g.iter().copied())))
    };

    let res = follow_irc(
        &planar_nh3(),
        &umbrella_mode(),
        &IrcConfig {
            step: 0.15,
            max_steps: 120,
            ..Default::default()
        },
        eg,
    )
    .expect("embedded IRC");

    let pf = pyramidalisation(&res.forward.mol);
    let pr = pyramidalisation(&res.reverse.mol);
    println!("embedded IRC: {pf:+.4} / {pr:+.4} Bohr");
    assert!(
        pf * pr < 0.0,
        "embedded IRC branches must still separate: {pf:+.4} and {pr:+.4}"
    );
    assert!(res.forward_barrier() > 0.0 && res.reverse_barrier() > 0.0);
}

/// The endpoint must not depend on the step size, and the walk must TERMINATE.
///
/// Both halves earned their place by failing. With a FIXED-length step the
/// walk could not settle into a basin -- near the minimum it overshoots and
/// oscillates across it -- so at `g_max_thresh = 1e-4` step 0.15 and step 0.05
/// both ran to the step budget (400 and 800) without converging, while 0.02
/// converged in 63. A test that only checked the endpoints would have passed
/// throughout, because the ENDPOINTS were right the whole time; only
/// `converged` was false.
///
/// Damping the step to `min(step, 2*|g|)` fixed it: all three now converge, in
/// 71/71/89 steps. That near-invariance in the step COUNT is the real evidence
/// the integrator is following one path rather than three.
#[test]
fn the_endpoints_do_not_depend_on_the_step_size() {
    let mut results = Vec::new();
    for step in [0.15_f64, 0.05, 0.02] {
        let res = follow_irc(
            &planar_nh3(),
            &umbrella_mode(),
            &IrcConfig {
                step,
                max_steps: 1500,
                ..Default::default()
            },
            gas_phase_eg(),
        )
        .expect("IRC");
        assert!(
            res.both_converged(),
            "step {step}: the walk must TERMINATE, not exhaust its budget \
             (fwd {} steps converged={}, rev {} steps converged={}). An \
             undamped fixed-length step oscillates across the minimum forever.",
            res.forward.steps,
            res.forward.converged,
            res.reverse.steps,
            res.reverse.converged
        );
        println!(
            "step {step:.2}: fwd {:+.4} / rev {:+.4} Bohr, {} / {} steps",
            pyramidalisation(&res.forward.mol),
            pyramidalisation(&res.reverse.mol),
            res.forward.steps,
            res.reverse.steps
        );
        results.push((
            pyramidalisation(&res.forward.mol),
            pyramidalisation(&res.reverse.mol),
        ));
    }
    // The endpoint is the ANSWER, so it must be step-independent. A 0.02 Bohr
    // band is far tighter than the ~0.8 Bohr pyramidalisation being measured
    // and far looser than the integrator's own noise.
    let (f0, r0) = results[0];
    for (f, r) in &results[1..] {
        assert!(
            (f - f0).abs() < 0.02 && (r - r0).abs() < 0.02,
            "the endpoint moved with the step size: {f0:+.4}/{r0:+.4} vs \
             {f:+.4}/{r:+.4}. The walk is resolving the step, not the path."
        );
    }
}

/// The path must be MASS-WEIGHTED, which is what makes it an IRC.
///
/// THIS TEST EXISTS BECAUSE A MUTATION SURVIVED. Deleting the mass weighting
/// entirely -- turning `follow_irc` into plain Cartesian steepest descent --
/// left all six other tests passing. On NH3 the two paths reach the SAME
/// endpoints, so every endpoint-based assertion is blind to the difference,
/// and the property that makes this an IRC rather than a generic downhill
/// walk was untested.
///
/// The discriminating system needs a heavy/light asymmetry, because mass
/// weighting is exactly what slows heavy atoms relative to light ones. HCN
/// with the hydrogen displaced does it: MEASURED over 24 steps from the same
/// start, the CARBON moves
///
/// ```text
///   mass-weighted   +0.0078 Bohr
///   unweighted      +0.3936 Bohr      50x further
/// ```
///
/// while the hydrogen moves comparably in both. That is the physics: an IRC is
/// the path of a classical trajectory with infinitesimal kinetic energy, so
/// heavy nuclei barely move. Unweighted descent lets carbon and hydrogen
/// respond equally to the same force, which is not a reaction path.
///
/// Asserted on the CARBON's displacement rather than on the endpoints, because
/// the endpoints are precisely what cannot tell the two apart.
///
/// A NOTE ON MUTATING THIS. `sm` enters in TWO coupled places -- the gradient
/// transform `g/sm` and the coordinate transform `q = x*sm`. Removing it from
/// only one leaves the walk mass-weighted, because the two cancel. My first
/// attempt did exactly that and "survived", which meant the MUTATION was
/// invalid rather than the test weak. To mutate this honestly, set `sm` to all
/// ones at its definition: MEASURED, that gives carbon 0.3461 Bohr and fails
/// here.
#[test]
fn the_path_is_mass_weighted_not_plain_steepest_descent() {
    // H displaced off a C-N axis: light atom on a heavy frame.
    let mol = Molecule::parse_xyz(
        "3\nHCN bent\nH 0.0 1.0 0.0\nC 0.0 0.0 0.0\nN 1.15 0.0 0.0\n",
        0,
        1,
    )
    .expect("parse HCN");
    let mode = Array1::from_vec(vec![0.3, -0.3, 0.0, 0.0, 0.02, 0.0, -0.02, 0.0, 0.0]);

    let res = follow_irc(
        &mol,
        &mode,
        &IrcConfig {
            step: 0.05,
            max_steps: 24,
            // Deliberately unreachable: this test is about the PATH taken in a
            // fixed number of steps, not about where it ends.
            g_max_thresh: 1e-12,
            ..Default::default()
        },
        gas_phase_eg(),
    )
    .expect("IRC on HCN");

    let c_moved = (res.forward.mol.atoms[1].x - mol.atoms[1].x).abs();
    let h_moved = (res.forward.mol.atoms[0].y - mol.atoms[0].y).abs();
    println!("carbon moved {c_moved:.4} Bohr, hydrogen moved {h_moved:.4} Bohr");

    // Unweighted descent gives ~0.39 Bohr for carbon here; mass-weighted gives
    // ~0.008. The bar sits an order of magnitude from each, so it separates
    // them without pinning either number.
    assert!(
        c_moved < 0.05,
        "carbon moved {c_moved:.4} Bohr in 24 steps. Mass-weighted descent \
         gives ~0.008 and UNWEIGHTED gives ~0.39 -- this is the signature of a \
         missing sqrt(m), which makes the walk plain steepest descent rather \
         than an IRC."
    );
    // ...and the light atom must actually be moving, or the walk is simply
    // stuck and the assertion above is satisfied by nothing happening.
    assert!(
        h_moved > 0.05,
        "hydrogen moved only {h_moved:.4} Bohr -- the walk is not progressing, \
         so the carbon test above proves nothing"
    );
}

/// A zero initial displacement must be REFUSED, not silently return the saddle.
///
/// The gradient at a saddle is zero by definition, so a walk started exactly
/// there never moves and would report the saddle itself as both endpoints --
/// a confident, symmetric, entirely wrong answer.
#[test]
fn a_zero_initial_displacement_is_refused() {
    let err = follow_irc(
        &planar_nh3(),
        &umbrella_mode(),
        &IrcConfig {
            initial_displacement: 0.0,
            ..Default::default()
        },
        gas_phase_eg(),
    )
    .expect_err("a zero displacement must be refused");
    assert!(
        err.to_string().contains("displacement"),
        "the error must name the cause, got: {err}"
    );
}

#[test]
fn a_mis_sized_mode_is_refused() {
    let err = follow_irc(
        &planar_nh3(),
        &Array1::from_vec(vec![1.0, 0.0, 0.0]), // 3 components for 12 coordinates
        &IrcConfig::default(),
        gas_phase_eg(),
    )
    .expect_err("a mis-sized mode must be refused");
    assert!(err.to_string().contains("imaginary mode"));
}

/// `IrcBranch.mol` and `IrcBranch.energy` must describe the SAME geometry.
///
/// The walk sets the geometry, evaluates there, then advances. A trailing
/// `set_cart_from_mass_weighted` after the loop moved the geometry ONE STEP
/// PAST the point the energy came from. On the `break` paths they happened to
/// agree; on BUDGET EXHAUSTION they did not -- and that is the case a caller
/// is most likely to inspect by hand, because it is the one that went wrong.
///
/// Checked by re-evaluating the returned geometry: the energy must match what
/// the branch reports, to SCF precision.
#[test]
fn the_branch_energy_belongs_to_the_branch_geometry() {
    // A tiny budget forces the exhaustion path, which is where they diverged.
    let res = follow_irc(
        &planar_nh3(),
        &umbrella_mode(),
        &IrcConfig {
            step: 0.15,
            max_steps: 4,
            ..Default::default()
        },
        gas_phase_eg(),
    )
    .expect("IRC");
    assert!(
        !res.forward.converged,
        "this test needs the BUDGET-EXHAUSTION path; raise the step count if \
         the walk now converges in 4 steps"
    );

    let mut eg = gas_phase_eg();
    for (tag, branch) in [("forward", &res.forward), ("reverse", &res.reverse)] {
        let (e_recomputed, _) = eg(&branch.mol).expect("re-evaluate");
        assert!(
            (e_recomputed - branch.energy).abs() < 1e-9,
            "{tag}: the branch reports E = {} but its geometry evaluates to \
             {e_recomputed}. The two describe different structures.",
            branch.energy
        );
    }
}

/// Every scalar knob must be FINITE and positive, not merely positive.
///
/// Each infinity fails differently and none of them fails loudly:
/// `step = inf` was masked only by the `min(step, 2|g|)` damping cap, so it
/// depended on an implementation detail; `initial_displacement = inf` hands
/// NON-FINITE coordinates to the caller's SCF, which fails somewhere far from
/// here; `g_max_thresh = inf` makes every gradient look converged, so the walk
/// stops at step 1 and confidently reports a basin.
#[test]
fn non_finite_settings_are_refused() {
    let inf = f64::INFINITY;
    let cases = [
        (
            "step",
            IrcConfig {
                step: inf,
                ..Default::default()
            },
        ),
        (
            "initial_displacement",
            IrcConfig {
                initial_displacement: inf,
                ..Default::default()
            },
        ),
        (
            "g_max_thresh",
            IrcConfig {
                g_max_thresh: inf,
                ..Default::default()
            },
        ),
        (
            "g_max_thresh<=0",
            IrcConfig {
                g_max_thresh: 0.0,
                ..Default::default()
            },
        ),
        (
            "nan step",
            IrcConfig {
                step: f64::NAN,
                ..Default::default()
            },
        ),
    ];
    for (tag, cfg) in cases {
        let err = follow_irc(&planar_nh3(), &umbrella_mode(), &cfg, gas_phase_eg())
            .expect_err(&format!("{tag} must be refused"));
        assert!(
            err.to_string().contains("finite"),
            "{tag}: the error must say the value is not finite/positive, got: {err}"
        );
    }

    // The anchor: the default config still runs, so the guard has not been
    // over-corrected into rejecting everything.
    assert!(follow_irc(
        &planar_nh3(),
        &umbrella_mode(),
        &IrcConfig {
            max_steps: 3,
            ..Default::default()
        },
        gas_phase_eg()
    )
    .is_ok());
}

/// A callback returning the wrong gradient length must ERROR, not panic.
///
/// The callback is caller-supplied, so its length is an INPUT and not an
/// invariant. Indexing past the end would panic inside a library; a short
/// gradient would silently walk on truncated data.
#[test]
fn a_wrong_length_callback_gradient_is_refused() {
    let err = follow_irc(
        &planar_nh3(),
        &umbrella_mode(),
        &IrcConfig::default(),
        |_m: &Molecule| Ok((0.0, Array1::zeros(3))), // 3 for a 12-coordinate molecule
    )
    .expect_err("a short gradient must be refused");
    assert!(
        err.to_string().contains("components"),
        "the error must name the length mismatch, got: {err}"
    );
}
