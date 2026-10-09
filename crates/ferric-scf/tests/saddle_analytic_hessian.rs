//! `find_saddle` driven by the ANALYTIC RHF Hessian (issue #293).
//!
//! `find_saddle` takes its Hessian as a callback, so the analytic Hessian
//! needs no change to the driver: the callback just asks
//! `harmonic_frequencies` for `HessianMethod::Analytic`. These tests pin, on
//! NH3 umbrella inversion (RHF/STO-3G, the system the Python saddle tests
//! use):
//!
//! 1. the analytic-Hessian search lands on the SAME saddle as the
//!    finite-difference one (geometry, energy, one imaginary mode);
//! 2. it costs fewer gradient evaluations -- COUNTED, not timed, as in
//!    `saddle_cost.rs`;
//! 3. the FD callback still reports `HessianSource::FiniteDifference` at the
//!    6N + 1 cost (the default path is unchanged);
//! 4. an unsupported system is a typed refusal under `Analytic`, and `Auto`
//!    falls back to FD and says so.
//!
//! Run with `OPENBLAS_NUM_THREADS=1`.

use std::cell::{Cell, RefCell};

use ferric_core::error::FerricError;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::libint_max_deriv_order;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    harmonic_frequencies, FrequencyConfig, FrequencyReference, HessianMethod, HessianSource,
};
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::saddle::{find_saddle, SaddleConfig, SaddleResult};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array1;

fn near_planar_nh3() -> Molecule {
    Molecule::parse_xyz(
        "4\nnh3 near planar\n\
         N  0.0000  0.0000  0.0000\n\
         H  0.0000  1.0100  0.1000\n\
         H  0.8747 -0.5050  0.1000\n\
         H -0.8747 -0.5050  0.1000\n",
        0,
        1,
    )
    .expect("parse NH3")
}

fn has_deriv2() -> bool {
    let order = libint_max_deriv_order();
    if order < 2 {
        eprintln!("SKIP: libint2 generated with derivative order {order}; no analytic Hessian");
    }
    order >= 2
}

struct Run {
    result: SaddleResult,
    /// Gradient evaluations made by the SEARCH itself (not inside Hessians).
    n_search_grad: usize,
    /// Gradient evaluations inside Hessians: `6N + 1` per FD Hessian, 0 per
    /// analytic one (one SCF + CPHF instead).
    n_hessian_grad: usize,
    n_hessians: usize,
    sources: Vec<HessianSource>,
}

impl Run {
    fn total_grad(&self) -> usize {
        self.n_search_grad + self.n_hessian_grad
    }
}

fn search(method: HessianMethod, recalc: usize, scf: &RhfConfig) -> Result<Run, FerricError> {
    let op = Operator::coulomb();
    let n_search_grad = Cell::new(0usize);
    let n_hessian_grad = Cell::new(0usize);
    let n_hessians = Cell::new(0usize);
    let sources = RefCell::new(Vec::new());

    let eg = |m: &Molecule| -> Result<(f64, Array1<f64>), FerricError> {
        n_search_grad.set(n_search_grad.get() + 1);
        let bs = ferric_core::basis::bundled("sto-3g")?;
        let prep = PreparedBasis::new(m, &bs)?;
        let b = SchwarzBounds::compute(op, &prep)?;
        let res = solve_rhf(&ParallelContext::default(), m, &prep, op, &b, scf)?;
        let g = rhf_gradient(m, &prep, op, &b, &res, scf.external_potential.as_ref())?;
        Ok((res.energy, Array1::from_iter(g.iter().copied())))
    };
    let hs = |m: &Molecule| {
        let fc = FrequencyConfig {
            reference: FrequencyReference::Rhf,
            hessian: method,
            ..Default::default()
        };
        let fr = harmonic_frequencies(&ParallelContext::default(), m, "sto-3g", op, scf, &fc)?;
        n_hessians.set(n_hessians.get() + 1);
        // The undisplaced gradient is not in the reported counter (see
        // saddle_cost.rs): an FD Hessian truly costs reported + 1.
        let cost = match fr.hessian_source {
            HessianSource::FiniteDifference => fr.n_gradient_evaluations + 1,
            HessianSource::Analytic => fr.n_gradient_evaluations,
        };
        n_hessian_grad.set(n_hessian_grad.get() + cost);
        sources.borrow_mut().push(fr.hessian_source);
        Ok(fr.cartesian_hessian)
    };

    let result = find_saddle(
        &near_planar_nh3(),
        &SaddleConfig {
            max_steps: 60,
            hessian_recalc_every: recalc,
            ..Default::default()
        },
        eg,
        hs,
    )?;
    Ok(Run {
        result,
        n_search_grad: n_search_grad.get(),
        n_hessian_grad: n_hessian_grad.get(),
        n_hessians: n_hessians.get(),
        sources: sources.into_inner(),
    })
}

fn max_coord_diff(a: &Molecule, b: &Molecule) -> f64 {
    a.atoms
        .iter()
        .zip(&b.atoms)
        .map(|(p, q)| {
            (p.x - q.x)
                .abs()
                .max((p.y - q.y).abs())
                .max((p.zpos - q.zpos).abs())
        })
        .fold(0.0, f64::max)
}

#[test]
fn analytic_hessian_search_finds_the_same_saddle_for_fewer_gradients() {
    if !has_deriv2() {
        return;
    }
    let scf = RhfConfig::default();
    let fd = search(HessianMethod::FiniteDifference, 0, &scf).expect("FD search");
    let an = search(HessianMethod::Analytic, 0, &scf).expect("analytic search");

    assert!(fd.result.is_transition_state() && an.result.is_transition_state());
    assert_eq!(fd.result.n_imaginary, 1);
    assert_eq!(an.result.n_imaginary, 1);
    assert!(fd
        .sources
        .iter()
        .all(|s| *s == HessianSource::FiniteDifference));
    assert!(an.sources.iter().all(|s| *s == HessianSource::Analytic));

    let dx = max_coord_diff(&fd.result.mol, &an.result.mol);
    let de = (fd.result.energy - an.result.energy).abs();
    println!(
        "MEASURED NH3/STO-3G recalc=0: FD {} Hessians, {} search + {} Hessian = {} \
         gradients, {} steps; analytic {} Hessians, {} search + {} Hessian = {} \
         gradients, {} steps; max|dx| = {dx:.2e} Bohr, |dE| = {de:.2e} Ha, \
         lowest eig FD {:.6e} analytic {:.6e}",
        fd.n_hessians,
        fd.n_search_grad,
        fd.n_hessian_grad,
        fd.total_grad(),
        fd.result.steps,
        an.n_hessians,
        an.n_search_grad,
        an.n_hessian_grad,
        an.total_grad(),
        an.result.steps,
        fd.result.lowest_eigenvalue,
        an.result.lowest_eigenvalue,
    );
    // Both searches stop on a gradient threshold, so the geometries agree to
    // the convergence limit, not to machine precision.
    assert!(dx < 5e-3, "saddle geometries differ by {dx:.3e} Bohr");
    assert!(de < 1e-6, "saddle energies differ by {de:.3e} Ha");
    assert!(
        (fd.result.lowest_eigenvalue - an.result.lowest_eigenvalue).abs()
            < 1e-2 * fd.result.lowest_eigenvalue.abs(),
        "imaginary-mode curvature: FD {} vs analytic {}",
        fd.result.lowest_eigenvalue,
        an.result.lowest_eigenvalue
    );

    // Cost: the FD Hessians are 2 * (6N + 1) = 50 gradients; the analytic ones
    // are zero displaced gradients.
    assert_eq!(fd.n_hessians, 2);
    assert_eq!(an.n_hessians, 2);
    assert_eq!(fd.n_hessian_grad, 2 * (6 * 4 + 1));
    assert_eq!(an.n_hessian_grad, 0);
    assert!(an.total_grad() < fd.total_grad());
}

#[test]
fn recalc_every_with_analytic_hessian_costs_no_extra_gradients() {
    if !has_deriv2() {
        return;
    }
    let scf = RhfConfig::default();
    for recalc in [1usize, 2, 3] {
        let fd = search(HessianMethod::FiniteDifference, recalc, &scf).expect("FD");
        let an = search(HessianMethod::Analytic, recalc, &scf).expect("analytic");
        println!(
            "MEASURED NH3/STO-3G recalc={recalc}: FD steps={} hessians={} gradients={}; \
             analytic steps={} hessians={} gradients={}",
            fd.result.steps,
            fd.n_hessians,
            fd.total_grad(),
            an.result.steps,
            an.n_hessians,
            an.total_grad(),
        );
        assert!(an.result.is_transition_state());
        assert!(an.n_hessians >= 2);
        // An analytic Hessian adds no gradient evaluations, so extra
        // recalculation can only change the step count.
        assert_eq!(an.n_hessian_grad, 0);
        assert_eq!(an.total_grad(), an.n_search_grad);
    }
}

fn charged_scf() -> RhfConfig {
    RhfConfig {
        external_potential: Some(ExternalPotential {
            point_charges: vec![
                PointCharge {
                    q: -0.4,
                    x: 0.0,
                    y: 0.0,
                    z: 6.0,
                },
                PointCharge {
                    q: -0.4,
                    x: 0.0,
                    y: 0.0,
                    z: -6.0,
                },
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn analytic_is_refused_with_a_point_charge_field_and_auto_falls_back() {
    if !has_deriv2() {
        return;
    }
    let scf = charged_scf();
    let err = match search(HessianMethod::Analytic, 0, &scf) {
        Ok(_) => panic!("Analytic must refuse an external potential, not fall back"),
        Err(e) => e.to_string(),
    };
    println!("REFUSAL: {err}");
    assert!(
        err.to_lowercase().contains("hessian") && err.to_lowercase().contains("external"),
        "the refusal must name the analytic Hessian and the external potential: {err}"
    );

    let auto = search(HessianMethod::Auto, 0, &scf).expect("Auto must fall back to FD");
    assert!(
        auto.sources
            .iter()
            .all(|s| *s == HessianSource::FiniteDifference),
        "Auto fell back, so every Hessian must report finite-difference"
    );
    assert_eq!(auto.result.n_imaginary, 1);
}
