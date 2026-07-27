//! DLPNO retention on real molecules — measuring what the screening actually buys.
//!
//! `pair_domains.rs` and `pno.rs` are exact in the limit and report their own
//! retention. This test drives both on real Boys-localized orbitals so the
//! accuracy/cost curve is a measurement rather than an assumption — which matters
//! because ferric has a MEASURED negative result for virtual truncation at small
//! sizes (the OSV sweep: 100% retention at accurate thresholds, or 48–76 mHa error
//! at loose ones).
//!
//! These tests assert *structural* properties that must hold regardless of how the
//! numbers come out, and print the retentions so the curve is visible.

use ferric_cc::pair_domains::{build_pair_domains, complete_pair_domains};
use ferric_cc::pno::build_pno_transforms;
use ferric_core::{basis, mol::Molecule, parallel::ParallelContext};
use ferric_integrals::{basis_bridge::PreparedBasis, operator::Operator};
use ferric_mp2::boys::boys_localize;
use ferric_scf::{
    rhf::{solve_rhf, RhfConfig},
    screening::SchwarzBounds,
};
use ndarray::Array2;

/// Converge RHF and Boys-localize the occupied space, returning the Boys centers.
fn boys_centers(xyz: &str, bas: &str) -> (Array2<f64>, usize, usize) {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(xyz).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(bas).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let cfg = RhfConfig { density_conv: 1e-9, max_iter: 100, ..Default::default() };
    let rhf = solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &cfg).unwrap();
    assert!(rhf.converged, "SCF must converge for the localization to mean anything");

    let nocc = (mol.nelec() / 2) as usize;
    let nbas = obs.nbasis();
    let nvir = nbas - nocc;
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., ..nocc]).to_owned();

    // Dipole integrals for the Boys functional.
    let dip = ferric_integrals::oneelectron::dipole(&obs, [0.0, 0.0, 0.0]).unwrap();
    let boys = boys_localize(&c_occ, &dip, 200);
    (boys.centers, nocc, nvir)
}

/// Both screens must be inert at infinite cutoff / zero threshold.
///
/// This is the composed exactness statement: with screening disabled the layer keeps
/// every pair, every coupling, and every virtual, so switching it on cannot change
/// any energy. Everything else here is only meaningful because this holds.
#[test]
fn disabled_screening_retains_everything() {
    let (centers, _nocc, nvir) = boys_centers("../../testdata/molecules/water.xyz", "cc-pvdz");

    let domains = complete_pair_domains(&centers).unwrap();
    assert!(domains.is_complete());
    assert_eq!(domains.pair_retention(), 1.0);
    assert_eq!(domains.coupling_retention(), 1.0);

    // Identity-like amplitudes are enough: the assertion is about counts, not values.
    let pnos = build_pno_transforms(&domains, nvir, 0.0, |_, _| {
        Array2::<f64>::from_shape_fn((nvir, nvir), |(a, b)| if a == b { 1.0 } else { 0.1 })
    })
    .unwrap();

    assert!(pnos.is_complete(), "t_cut_pno = 0 must keep every virtual");
    assert_eq!(pnos.virtual_retention(), 1.0);
    assert_eq!(pnos.max_discarded_weight(), 0.0);
}

/// Retention must fall monotonically as the pair cutoff tightens.
///
/// Printed so the curve is visible: this is the number that decides whether the
/// occupied-side screening is worth anything on a system of this size.
#[test]
fn pair_retention_falls_with_cutoff() {
    let (centers, nocc, _nvir) = boys_centers("../../testdata/molecules/water.xyz", "cc-pvdz");
    eprintln!("water/cc-pVDZ: nocc = {nocc}");

    let mut last = f64::INFINITY;
    for cutoff in [f64::INFINITY, 8.0, 4.0, 2.0, 1.0] {
        let d = build_pair_domains(&centers, cutoff, cutoff).unwrap();
        eprintln!(
            "  cutoff = {cutoff:>5} Bohr:  pairs {:>3}/{:<3} ({:.3})   coupling {:.3}",
            d.pairs.len(),
            nocc * (nocc + 1) / 2,
            d.pair_retention(),
            d.coupling_retention()
        );
        assert!(
            d.pair_retention() <= last + 1e-12,
            "retention rose when the cutoff tightened: {} > {last}",
            d.pair_retention()
        );
        last = d.pair_retention();
        // The diagonal is never screened, so retention has a hard floor.
        assert!(d.pair_retention() >= nocc as f64 / (nocc * (nocc + 1) / 2) as f64 - 1e-12);
    }
}

/// A larger molecule must be *more* screenable than a small one at fixed cutoff.
///
/// This is the locality claim in its testable form: correlation is local, so as the
/// system grows the fraction of pairs within a fixed radius must fall. If this failed,
/// the whole premise of domain screening would be wrong for this code path.
#[test]
fn locality_improves_with_system_size() {
    let cutoff = 4.0;
    let (c_small, no_small, _) = boys_centers("../../testdata/molecules/water.xyz", "cc-pvdz");
    let (c_large, no_large, _) = boys_centers("../../testdata/molecules/benzene.xyz", "sto-3g");

    let d_small = build_pair_domains(&c_small, cutoff, cutoff).unwrap();
    let d_large = build_pair_domains(&c_large, cutoff, cutoff).unwrap();

    eprintln!(
        "at cutoff {cutoff} Bohr:  water (nocc={no_small}) pair retention {:.3}, \
         coupling {:.3}",
        d_small.pair_retention(),
        d_small.coupling_retention()
    );
    eprintln!(
        "                        benzene (nocc={no_large}) pair retention {:.3}, \
         coupling {:.3}",
        d_large.pair_retention(),
        d_large.coupling_retention()
    );

    assert!(
        d_large.pair_retention() < d_small.pair_retention(),
        "benzene ({:.3}) should be more screenable than water ({:.3}) -- if not, \
         correlation locality is not being captured by these domains",
        d_large.pair_retention(),
        d_small.pair_retention()
    );
    // The quartic factor is the one that matters for the hh ladder.
    assert!(
        d_large.coupling_retention() < d_small.coupling_retention(),
        "the pair x pair (n_o^4) block should screen harder on the larger system"
    );
}

/// The two layers compose: domain screening reduces how many PNO blocks get built.
#[test]
fn domains_and_pnos_compose() {
    let (centers, _nocc, nvir) = boys_centers("../../testdata/molecules/benzene.xyz", "sto-3g");

    let all = complete_pair_domains(&centers).unwrap();
    let screened = build_pair_domains(&centers, 4.0, f64::INFINITY).unwrap();
    assert!(screened.pairs.len() < all.pairs.len(), "test premise: screening drops pairs");

    let amp = |nvir: usize| {
        Array2::<f64>::from_shape_fn((nvir, nvir), |(a, b)| {
            1.0 / (1.0 + (a as f64 - b as f64).abs())
        })
    };
    let p_all = build_pno_transforms(&all, nvir, 0.0, |_, _| amp(nvir)).unwrap();
    let p_scr = build_pno_transforms(&screened, nvir, 0.0, |_, _| amp(nvir)).unwrap();

    eprintln!(
        "benzene/STO-3G: PNO blocks {} (dense) -> {} (screened), a {:.1}% cut",
        p_all.pairs.len(),
        p_scr.pairs.len(),
        100.0 * (1.0 - p_scr.pairs.len() as f64 / p_all.pairs.len() as f64)
    );
    assert!(p_scr.pairs.len() < p_all.pairs.len());
}
