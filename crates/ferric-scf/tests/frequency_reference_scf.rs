//! `FrequencyResult::reference_scf` is the SCF at the UNDISPLACED geometry,
//! solved with the config the frequency task runs (`for_geometry_task`), for
//! every route of `harmonic_frequencies*`, so a caller that certifies the
//! reference point reads it instead of solving it a second time.
//!
//! Exactness anchor: the carried SCF is bit-identical to an SCF a caller
//! solves itself at the input geometry with the carried config (energy,
//! total density, MO coefficients). The solver is deterministic on one
//! thread-pool configuration, and the two solves run the same inputs, so
//! the expected difference is exactly 0.
//!
//! Artifact hypothesis. If the carried SCF came from a displaced geometry
//! (the first displaced point of a finite-difference stencil is 5e-3 Bohr
//! away), its energy would differ from the reference energy by about
//! |g| * delta ~ 1e-7 Ha or more, many orders above the 0 expected here.

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    harmonic_frequencies, harmonic_frequencies_with_scf_correction, FrequencyConfig,
    FrequencyReference, FrequencyResult, HessianMethod,
};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

const BASIS: &str = "sto-3g";

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nwater\nO 0.000000 0.000000 0.117300\n\
         H 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n",
        0,
        1,
    )
    .unwrap()
}

fn hydroxyl() -> Molecule {
    Molecule::parse_xyz("2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap()
}

fn freq_config(reference: FrequencyReference, hessian: HessianMethod) -> FrequencyConfig {
    FrequencyConfig {
        reference,
        hessian,
        ..Default::default()
    }
}

fn frequencies(
    mol: &Molecule,
    scf: &RhfConfig,
    fc: &FrequencyConfig,
) -> Result<FrequencyResult, ferric_core::FerricError> {
    harmonic_frequencies(
        &ParallelContext::default(),
        mol,
        BASIS,
        Operator::coulomb(),
        scf,
        fc,
    )
}

/// The SCF a caller solves itself at `mol` with the config the result carries.
fn independent_scf(mol: &Molecule, res: &FrequencyResult, restricted: bool) -> ScfResult {
    let reference = res.reference_scf.as_ref().expect("a reference SCF");
    let bs = ferric_core::basis::bundled(BASIS).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    if restricted {
        solve_rhf(&ctx, mol, &prep, op, &bounds, &reference.config).unwrap()
    } else {
        solve_uhf(&ctx, mol, &prep, &bounds, &reference.config).unwrap()
    }
}

fn bits(a: &Array2<f64>) -> Vec<u64> {
    a.iter().map(|x| x.to_bits()).collect()
}

/// The carried SCF is the reference one: converged, the result's energy, and
/// bit-identical to an independent solve at `mol` (energy, density, MOs).
fn assert_is_the_reference_scf(mol: &Molecule, res: &FrequencyResult, restricted: bool) {
    let reference = res.reference_scf.as_ref().expect("a reference SCF");
    assert!(
        reference.scf.converged,
        "the reference SCF did not converge"
    );
    assert_eq!(
        reference.scf.energy.to_bits(),
        (res.energy - res.correction_energy).to_bits(),
        "the carried SCF energy is not the result's reference energy"
    );
    let mine = independent_scf(mol, res, restricted);
    assert_eq!(
        reference.scf.energy.to_bits(),
        mine.energy.to_bits(),
        "carried {} vs independent {} Ha",
        reference.scf.energy,
        mine.energy
    );
    assert_eq!(
        bits(&reference.scf.density_total),
        bits(&mine.density_total)
    );
    assert_eq!(bits(&reference.scf.mos_alpha), bits(&mine.mos_alpha));
    // The frequency task's config: an unset RI-J precision resolves to f64.
    assert_eq!(
        reference.config.ri_j_task,
        ferric_scf::df_j::RiJTask::Geometry
    );
}

#[test]
fn analytic_rhf_route_carries_the_reference_scf() {
    let mol = water();
    let res = frequencies(
        &mol,
        &RhfConfig::default(),
        &freq_config(FrequencyReference::Rhf, HessianMethod::Auto),
    )
    .unwrap();
    assert_eq!(res.n_gradient_evaluations, 0, "expected the analytic route");
    assert_is_the_reference_scf(&mol, &res, true);
}

#[test]
fn analytic_uhf_route_carries_the_reference_scf() {
    let mol = hydroxyl();
    let res = frequencies(
        &mol,
        &RhfConfig::default(),
        &freq_config(FrequencyReference::Uhf, HessianMethod::Analytic),
    )
    .unwrap();
    assert_eq!(res.n_gradient_evaluations, 0, "expected the analytic route");
    assert_is_the_reference_scf(&mol, &res, false);
}

#[test]
fn moving_grid_ks_route_carries_the_reference_scf() {
    let mol = water();
    let scf = RhfConfig::default().with_xc("PBE");
    let res = frequencies(
        &mol,
        &scf,
        &freq_config(FrequencyReference::Rhf, HessianMethod::AnalyticMovingGrid),
    )
    .unwrap();
    assert_eq!(res.n_gradient_evaluations, 0, "expected the analytic route");
    assert_is_the_reference_scf(&mol, &res, true);
}

#[test]
fn finite_difference_route_carries_the_reference_scf() {
    let mol = water();
    let res = frequencies(
        &mol,
        &RhfConfig::default(),
        &freq_config(FrequencyReference::Rhf, HessianMethod::FiniteDifference),
    )
    .unwrap();
    assert_eq!(res.n_gradient_evaluations, 18);
    assert_is_the_reference_scf(&mol, &res, true);
}

#[test]
fn corrected_route_carries_the_reference_scf() {
    let mol = water();
    let res = harmonic_frequencies_with_scf_correction(
        &ParallelContext::default(),
        &mol,
        BASIS,
        Operator::coulomb(),
        &RhfConfig::default(),
        &freq_config(FrequencyReference::Rhf, HessianMethod::FiniteDifference),
        |_, _| Ok((0.0, None)),
    )
    .unwrap();
    assert_is_the_reference_scf(&mol, &res, true);
}

#[test]
fn a_result_not_built_from_an_scf_carries_none() {
    // A harmonic surface E = k/2 |R - R0|^2 (k = 0.5 Ha/Bohr^2), so the
    // Hessian is well defined without any SCF.
    let mol = water();
    let at_reference = mol.clone();
    let res = ferric_scf::frequencies::harmonic_frequencies_from_gradient(&mol, 5e-3, 0.0, |m| {
        let mut g = Array2::<f64>::zeros((m.atoms.len(), 3));
        for (i, (a, r)) in m.atoms.iter().zip(&at_reference.atoms).enumerate() {
            g[(i, 0)] = 0.5 * (a.x - r.x);
            g[(i, 1)] = 0.5 * (a.y - r.y);
            g[(i, 2)] = 0.5 * (a.zpos - r.zpos);
        }
        Ok(g)
    })
    .unwrap();
    assert!(res.reference_scf.is_none());
}
