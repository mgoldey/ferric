//! Tests for Becke-Lebedev per-atom static polarizability.
//!
//! The Becke partition is exact at every grid point (Σ_A w^A(r) = 1), so
//! the sum rule Σ_A α^A = α_molecular must hold to grid-quadrature accuracy.
//!
//! Unlike the Slater-Hirshfeld scheme (which had ~50% magnitude error from
//! the bad proatom), Becke is geometry-only and accurate at production
//! grid sizes (75 radial × 110 angular).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::{pdep_polarizability_becke, pdep_polarizability_static};
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn setup_h2o() -> (Molecule, PreparedBasis, ferric_core::basis::BasisSet, PreparedBasis,
                   Operator, ferric_scf::ScfResult) {
    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    (mol, obs, obs_bs, dfbs, op, rhf)
}

#[test]
fn becke_h2o_sum_rule_matches_molecular_alpha() {
    // With analytical-renormalization of the per-atom AO dipoles, Σ_A α^A
    // = molecular α exactly (the partition is just a fractional split of
    // the same analytical dipole integral).
    let (mol, obs, obs_bs, dfbs, op, rhf) = setup_h2o();
    let cfg = PdepRpaConfig::default();
    let alphas_per_atom = pdep_polarizability_becke(
        &mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg,
    ).unwrap();
    let alpha_mol = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, op, &cfg).unwrap();

    let mut sum = [[0.0f64; 3]; 3];
    for tens in &alphas_per_atom {
        for i in 0..3 {
            for j in 0..3 {
                sum[i][j] += tens[i][j];
            }
        }
    }
    let sum_iso = (sum[0][0] + sum[1][1] + sum[2][2]) / 3.0;
    eprintln!("Becke sum α_iso = {:.6}, molecular α_iso = {:.6}", sum_iso, alpha_mol.iso);
    eprintln!("per-atom α_iso: {:?}",
        alphas_per_atom.iter().map(|t| (t[0][0]+t[1][1]+t[2][2])/3.0).collect::<Vec<_>>());

    let dev_iso = (sum_iso - alpha_mol.iso).abs();
    assert!(dev_iso < 1e-4,
        "Becke sum-rule on H2O: |Σ α^A − α_mol|_iso = {dev_iso:.2e}, want < 1e-4");
}

#[test]
fn becke_h2o_oxygen_dominant() {
    // Physically reasonable: oxygen carries more of the polarizability
    // than the hydrogens. (O is bigger, more valence electrons, more
    // polarizable.) This is a *qualitative* test — the actual ratio
    // depends on partition scheme.
    let (mol, obs, obs_bs, dfbs, op, rhf) = setup_h2o();
    let cfg = PdepRpaConfig::default();
    let alphas = pdep_polarizability_becke(&mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg).unwrap();
    let isos: Vec<f64> = alphas.iter()
        .map(|t| (t[0][0] + t[1][1] + t[2][2]) / 3.0).collect();
    eprintln!("H2O Becke per-atom α_iso: O={:.4}, H={:.4}, H={:.4}", isos[0], isos[1], isos[2]);

    // O is index 0 (per setup_h2o ordering).
    assert!(isos[0] > isos[1], "O α should exceed H α: O={}, H1={}", isos[0], isos[1]);
    assert!(isos[0] > isos[2], "O α should exceed H α: O={}, H2={}", isos[0], isos[2]);
    // H atoms equivalent (within grid noise).
    assert!((isos[1] - isos[2]).abs() < 0.05,
        "H atoms should have equivalent α: H1={}, H2={}", isos[1], isos[2]);
}
