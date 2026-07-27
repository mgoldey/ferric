//! Open-shell LinLCCD(hh).
//!
//! The load-bearing test is `u_linlccd_matches_restricted_on_closed_shell`: on a
//! closed-shell system a UHF reference collapses to the RHF one, so the unrestricted
//! path must reproduce the already-validated restricted path exactly. That pins the
//! per-spin integral builders, the interleaved spin-orbital layout, and the
//! denominators all at once.

use ferric_cc::linlccd::{linlccd, LadderVariant};
use ferric_cc::linlccd_u::u_linlccd;
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::semicanonical::semicanonicalize;
use ferric_scf::uhf::solve_uhf;

fn mol_path(name: &str) -> String {
    format!("{}/../../testdata/molecules/{}", env!("CARGO_MANIFEST_DIR"), name)
}

/// THE LOAD-BEARING TEST — closed-shell UHF must reproduce RHF.
///
/// For a closed-shell singlet the UHF solution is the RHF one with alpha == beta, so
/// `u_linlccd` and `linlccd` must agree to numerical noise. Any error in the per-spin
/// builders, the interleaved layout, or the denominators breaks this.
#[test]
fn u_linlccd_matches_restricted_on_closed_shell() {
    let mol = Molecule::load_xyz(&mol_path("water.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let ctx = ParallelContext::default();
    let scf_cfg = RhfConfig { density_conv: 1e-10, max_iter: 200, ..Default::default() };
    let cfg = CcConfig { energy_conv: 1e-11, max_iter: 200, ..Default::default() };
    let op = Operator::coulomb();

    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();
    assert!(rhf.converged);
    let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &scf_cfg).unwrap();
    assert!(uhf.converged);
    assert!(
        (rhf.energy - uhf.energy).abs() < 1e-8,
        "test premise: closed-shell UHF must collapse to RHF ({:.10} vs {:.10})",
        uhf.energy,
        rhf.energy
    );

    for variant in [LadderVariant::DriversOnly, LadderVariant::Hh] {
        let r = linlccd(&mol, &obs, &dfbs, op, &rhf, &cfg, variant).unwrap().correlation_energy;
        let u = u_linlccd(&mol, &obs, &dfbs, op, &uhf, &cfg, variant).unwrap().correlation_energy;
        eprintln!("{variant:12?}  restricted = {r:.12}  unrestricted = {u:.12}  diff = {:+.3e}", u - r);
        assert!(
            (u - r).abs() < 1e-8,
            "{variant:?}: unrestricted path disagrees with the validated restricted one \
             ({u:.12} vs {r:.12})"
        );
    }
}

/// An open-shell system must run and give a physically sensible correlation energy.
#[test]
fn runs_on_an_open_shell_doublet() {
    let mol = Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let ctx = ParallelContext::default();
    let scf_cfg = RhfConfig { density_conv: 1e-10, max_iter: 200, ..Default::default() };
    let cfg = CcConfig { energy_conv: 1e-11, max_iter: 200, ..Default::default() };

    let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &scf_cfg).unwrap();
    assert!(uhf.converged);

    let drivers =
        u_linlccd(&mol, &obs, &dfbs, Operator::coulomb(), &uhf, &cfg, LadderVariant::DriversOnly)
            .unwrap()
            .correlation_energy;
    let hh = u_linlccd(&mol, &obs, &dfbs, Operator::coulomb(), &uhf, &cfg, LadderVariant::Hh)
        .unwrap()
        .correlation_energy;

    eprintln!("OH doublet: drivers = {drivers:.10}   LinLCCD(hh) = {hh:.10}");
    assert!(drivers < 0.0 && hh < 0.0, "correlation energies must be negative");
    assert!(drivers.is_finite() && hh.is_finite());
    // Same signature as the closed-shell case: the hh dressing widens the gap, so it
    // reduces the magnitude of the correlation energy.
    assert!(
        hh.abs() < drivers.abs(),
        "hh dressing must reduce |E_corr|: {hh:.10} vs {drivers:.10}"
    );
}

/// A raw ROHF reference must be REFUSED, and the semi-canonicalized one accepted.
///
/// ROHF carries no per-spin orbital energies; its `eps` belong to the effective
/// Roothaan Fock, i.e. to neither spin. Silently using them is the approximation
/// `u_rimp2.rs:97` still makes, and this path deliberately does not repeat it.
#[test]
fn raw_rohf_is_refused_but_semicanonical_works() {
    let mol = Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let ctx = ParallelContext::default();
    let scf_cfg = RhfConfig { density_conv: 1e-10, max_iter: 200, ..Default::default() };
    let cfg = CcConfig { energy_conv: 1e-11, max_iter: 200, ..Default::default() };
    let op = Operator::coulomb();

    let rohf = solve_rohf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();
    assert!(rohf.converged);

    assert!(
        u_linlccd(&mol, &obs, &dfbs, op, &rohf, &cfg, LadderVariant::Hh).is_err(),
        "a raw ROHF reference must be refused -- its eps belong to neither spin"
    );

    let sc = semicanonicalize(&ctx, &mol, &obs, &bounds, &rohf, 1e-12, None).unwrap();
    let semi = sc.to_unrestricted_result(&rohf);
    let e = u_linlccd(&mol, &obs, &dfbs, op, &semi, &cfg, LadderVariant::Hh)
        .expect("semi-canonicalized ROHF must be accepted")
        .correlation_energy;

    eprintln!("ROHF -> semi-canonical -> LinLCCD(hh): E_corr = {e:.10}");
    assert!(e < 0.0 && e.is_finite(), "correlation energy must be negative and finite");
}

/// The short-range attenuated operator must work open-shell too — this is the path
/// open-shell wB97X-L-V needs.
#[test]
fn short_range_attenuation_works_open_shell() {
    let mol = Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let ctx = ParallelContext::default();
    let scf_cfg = RhfConfig { density_conv: 1e-10, max_iter: 200, ..Default::default() };
    let cfg = CcConfig { energy_conv: 1e-11, max_iter: 200, ..Default::default() };

    let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &scf_cfg).unwrap();
    let coul = u_linlccd(&mol, &obs, &dfbs, Operator::coulomb(), &uhf, &cfg, LadderVariant::Hh)
        .unwrap()
        .correlation_energy;
    let sr = u_linlccd(&mol, &obs, &dfbs, Operator::erfc(0.1), &uhf, &cfg, LadderVariant::Hh)
        .unwrap()
        .correlation_energy;
    let strong = u_linlccd(&mol, &obs, &dfbs, Operator::erfc(1.0), &uhf, &cfg, LadderVariant::Hh)
        .unwrap()
        .correlation_energy;

    eprintln!("Coulomb = {coul:.10}   erfc(0.1) = {sr:.10}   erfc(1.0) = {strong:.10}");
    assert!(sr.is_finite() && strong.is_finite());
    assert!(
        strong.abs() < 0.7 * coul.abs(),
        "strong attenuation should strip much of the correlation: {strong:.10} vs {coul:.10}"
    );
}
