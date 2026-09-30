//! ωB97X-L-V Be₂ bond dissociation energy vs the published value.
//!
//! Ransford & Carter-Fenk, PCCP 2026, 28, 14428, Table 4 (p. 14437): the Be···Be BDE
//! computed as E(r = 10 000 Å) − E(r_eq) in def2-QZVPPD has signed error −0.4 kcal/mol
//! against the 2.7 kcal/mol experimental well, so ωB97X-L-V gives **2.3 kcal/mol**.
//! Both geometries are closed-shell singlets with only fully occupied pairs, so the
//! restricted path is well posed even at 10 000 Å.
//!
//! H₂ (the other closed-shell entry in Table 4) is NOT usable: at 10 000 Å σg and σu
//! are degenerate to machine precision, an unsymmetrized RKS solver occupies an
//! arbitrary mix of them (a localized, ionic H⁻/H⁺ determinant), and the paper's
//! value needs the symmetry-adapted σg² state. Q-Chem enforces point-group symmetry;
//! ferric does not.

use ferric_cc::double_hybrid::{run_wb97x_l_v, DoubleHybridConfig, DoubleHybridResult};
use ferric_core::basis::load_bse_json;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;

const HARTREE_TO_KCAL: f64 = 627.509_474;

/// Table 4: 2.7 experimental well − 0.4 signed error, printed to 0.1 kcal/mol.
const PAPER_BDE_KCAL: f64 = 2.3;

fn bundled_file(name: &str) -> ferric_core::basis::BasisSet {
    load_bse_json(&format!(
        "{}/../ferric-core/src/basis/bundled/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

fn be2(r_angstrom: f64, cfg: &DoubleHybridConfig) -> DoubleHybridResult {
    let mol = Molecule::parse_xyz(
        &format!("2\nBe2\nBe 0 0 0\nBe 0 0 {r_angstrom}\n"),
        0,
        1,
    )
    .unwrap();
    let obs = PreparedBasis::new(&mol, &bundled_file("def2-qzvppd")).unwrap();
    let dfbs = PreparedBasis::new(&mol, &bundled_file("def2-qzvppd-rifit")).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let (dh, _) = run_wb97x_l_v(
        &ParallelContext::default(),
        &mol,
        &obs,
        &dfbs,
        &bounds,
        &RhfConfig::default(),
        cfg,
    )
    .expect("wB97X-L-V must run on Be2");
    dh
}

/// Experimental Be₂ r_e (Merritt, Bondybey & Heaven, Science 2009), the standard
/// benchmark geometry; the paper does not print its r_eq.
const R_EQ_ANGSTROM: f64 = 2.4536;

/// Measured 2026-09-30 at r = r_e (release, ~3 s):
///
/// ```text
///                         BDE (kcal/mol)   error vs 2.3
///   eqn (22), this code       2.299          -0.001
///   linear λ (solve at λ=1,   4.199          +1.90   <- fails this test
///     then multiply by λ)
///   KS only (no WFT)         -0.79           -3.1
/// ```
///
/// The bar sits between the good and bad measurements, 6x below the linear-λ error. It also has to absorb the paper's unprinted r_eq: moving r by
/// ±0.1 Å moves the BDE by −0.55/+0.20 kcal/mol, which is why r is pinned to r_e.
const TOL_KCAL: f64 = 0.3;

#[test]
fn be2_bond_dissociation_energy_matches_the_paper() {
    let cfg = DoubleHybridConfig::default();
    let far = be2(10_000.0, &cfg);
    let eq = be2(R_EQ_ANGSTROM, &cfg);
    let bde = (far.total_energy - eq.total_energy) * HARTREE_TO_KCAL;
    eprintln!(
        "Be2 BDE {bde:.3} kcal/mol (paper {PAPER_BDE_KCAL}); lambda*E_c eq {:.8} far {:.8}",
        eq.e_c_scaled, far.e_c_scaled
    );
    assert!(
        (bde - PAPER_BDE_KCAL).abs() < TOL_KCAL,
        "Be2 BDE {bde:.3} kcal/mol vs published {PAPER_BDE_KCAL} (tol {TOL_KCAL})"
    );
}
