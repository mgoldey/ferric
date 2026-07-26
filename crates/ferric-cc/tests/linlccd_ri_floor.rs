//! Quantify the RI (density-fitting) error in LinLCCD(hh) against exact integrals.
//!
//! The RI error floor is invisible from inside the RI path -- it looks like a converged
//! answer. These tests solve the SAME amplitude equations from exact 4-center
//! integrals and difference the two, so the floor is measured rather than assumed.
//!
//! Everything downstream of the integral source is shared between the two paths (same
//! antisymmetrizers, same residual, same DIIS), so a difference here is attributable to
//! density fitting alone.

use ferric_cc::linlccd::{linlccd, LadderVariant};
use ferric_cc::linlccd_exact::linlccd_exact;
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::canonical::canonical_mp2;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn mol_path(name: &str) -> String {
    format!("{}/../../testdata/molecules/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn converged_rhf(
    mol: &Molecule,
    obs: &PreparedBasis,
) -> ferric_scf::result::ScfResult {
    let bounds = SchwarzBounds::compute(Operator::coulomb(), obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        mol,
        obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig { density_conv: 1e-10, max_iter: 200, ..Default::default() },
    )
    .unwrap();
    assert!(rhf.converged, "reference SCF must converge");
    rhf
}

/// The exact path must reproduce canonical MP2 when only driver terms are kept.
///
/// This validates the exact path itself before it is used to judge the RI path --
/// otherwise a bug in the reference would be misread as RI error. `canonical_mp2` is
/// independently validated against PySCF.
#[test]
fn exact_drivers_only_reproduces_canonical_mp2() {
    let mol = Molecule::load_xyz(&mol_path("water.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let rhf = converged_rhf(&mol, &obs);
    let op = Operator::coulomb();

    let e_canon = canonical_mp2(&mol, &obs, op, &rhf, 0).unwrap();
    let e_exact = linlccd_exact(
        &mol,
        &obs,
        op,
        &rhf,
        &CcConfig { energy_conv: 1e-12, max_iter: 100, ..Default::default() },
        LadderVariant::DriversOnly,
    )
    .unwrap()
    .correlation_energy;

    eprintln!("canonical MP2          = {e_canon:.12}");
    eprintln!("exact LinLCCD drivers  = {e_exact:.12}");
    eprintln!("difference             = {:+.3e}", e_exact - e_canon);
    assert!(
        (e_exact - e_canon).abs() < 1e-10,
        "the exact reference path is itself wrong: {e_exact:.12} vs canonical \
         {e_canon:.12}"
    );
}

/// MEASURE the RI error in LinLCCD(hh) on water/STO-3G and cc-pVDZ.
///
/// Reports rather than tightly bounds: the point is to know the number. The assertion
/// is loose enough to catch a real defect (a wrong aux basis, a broken metric) but not
/// so tight that ordinary DF error trips it.
#[test]
fn measure_ri_error_vs_exact() {
    let op = Operator::coulomb();
    let cfg = CcConfig { energy_conv: 1e-12, max_iter: 200, ..Default::default() };

    for (obs_name, aux_name) in [("sto-3g", "cc-pvdz-ri"), ("cc-pvdz", "cc-pvdz-ri")] {
        let mol = Molecule::load_xyz(&mol_path("water.xyz")).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
        let rhf = converged_rhf(&mol, &obs);

        for variant in [LadderVariant::DriversOnly, LadderVariant::Hh] {
            let e_ri = linlccd(&mol, &obs, &dfbs, op, &rhf, &cfg, variant)
                .unwrap()
                .correlation_energy;
            let e_ex = linlccd_exact(&mol, &obs, op, &rhf, &cfg, variant)
                .unwrap()
                .correlation_energy;
            let err = e_ri - e_ex;
            eprintln!(
                "{obs_name:8} {variant:12?}  RI = {e_ri:14.10}  exact = {e_ex:14.10}  \
                 RI error = {err:+.3e}  ({:.4}%)",
                100.0 * err / e_ex
            );
            assert!(
                err.abs() < 5e-3,
                "{obs_name}/{variant:?}: RI error {err:+.3e} is far larger than density \
                 fitting should produce -- suspect the aux basis or the metric"
            );
        }
    }
}

/// The hh ladder must be reproduced by RI, not just the drivers.
///
/// If the hh contribution (which uses the oooo block, a different RI product than the
/// ovov block MP2 needs) were badly fit, the RI error on `Hh` would be much larger than
/// on `DriversOnly`. This compares the two directly.
#[test]
fn ri_reproduces_the_hh_contribution() {
    let mol = Molecule::load_xyz(&mol_path("water.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let rhf = converged_rhf(&mol, &obs);
    let op = Operator::coulomb();
    let cfg = CcConfig { energy_conv: 1e-12, max_iter: 200, ..Default::default() };

    let d_ri = linlccd(&mol, &obs, &dfbs, op, &rhf, &cfg, LadderVariant::DriversOnly)
        .unwrap()
        .correlation_energy;
    let h_ri =
        linlccd(&mol, &obs, &dfbs, op, &rhf, &cfg, LadderVariant::Hh).unwrap().correlation_energy;
    let d_ex = linlccd_exact(&mol, &obs, op, &rhf, &cfg, LadderVariant::DriversOnly)
        .unwrap()
        .correlation_energy;
    let h_ex =
        linlccd_exact(&mol, &obs, op, &rhf, &cfg, LadderVariant::Hh).unwrap().correlation_energy;

    // The hh contribution itself, isolated in each scheme.
    let hh_ri = h_ri - d_ri;
    let hh_ex = h_ex - d_ex;
    eprintln!("hh contribution: RI = {hh_ri:+.10}  exact = {hh_ex:+.10}  \
               diff = {:+.3e} ({:.3}%)", hh_ri - hh_ex, 100.0 * (hh_ri - hh_ex) / hh_ex);

    assert!(
        (hh_ri - hh_ex).abs() < 1e-3,
        "RI reproduces the hh ladder poorly: {hh_ri:+.10} vs exact {hh_ex:+.10}"
    );
    assert!(
        hh_ri.signum() == hh_ex.signum(),
        "RI and exact disagree on the SIGN of the hh contribution"
    );
}

/// The RI error must not swamp the short-range attenuated correlation the double
/// hybrid actually uses.
///
/// ωB97X-L-V evaluates correlation with erfc(ω = 0.1). The erf/erfc RI metric is
/// known to be more delicate than the Coulomb one (ferric regularizes it separately),
/// so this checks the operator the functional actually runs with.
#[test]
fn ri_error_under_short_range_attenuation() {
    let mol = Molecule::load_xyz(&mol_path("water.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let rhf = converged_rhf(&mol, &obs);
    let cfg = CcConfig { energy_conv: 1e-12, max_iter: 200, ..Default::default() };

    for omega in [0.1_f64, 0.5] {
        let op = Operator::erfc(omega);
        let e_ri = linlccd(&mol, &obs, &dfbs, op, &rhf, &cfg, LadderVariant::Hh)
            .unwrap()
            .correlation_energy;
        let e_ex =
            linlccd_exact(&mol, &obs, op, &rhf, &cfg, LadderVariant::Hh).unwrap().correlation_energy;
        let err = e_ri - e_ex;
        eprintln!(
            "erfc(omega={omega})  RI = {e_ri:14.10}  exact = {e_ex:14.10}  \
             RI error = {err:+.3e} ({:.4}%)",
            100.0 * err / e_ex
        );
        assert!(
            err.abs() < 5e-3,
            "RI error {err:+.3e} under erfc({omega}) is too large -- the attenuated \
             metric may be mis-handled"
        );
    }
}
