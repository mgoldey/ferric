//! Per-atom Hirshfeld decomposition of the static polarizability tensor.
//!
//! Validates:
//!   * Sum rule: Σ_A α^A reproduces the molecular α to <1e-3 a.u.
//!   * Per-atom symmetry: max|α^A − (α^A)^T| < 1e-5 (consumer schema bound).
//!   * Chemical sanity: oxygen dominates over hydrogen in H2O.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::{pdep_polarizability_hirshfeld, pdep_polarizability_static};
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn setup_h2o_ccpvdz() -> (
    Molecule,
    PreparedBasis,
    ferric_core::basis::BasisSet,
    PreparedBasis,
    Operator,
    ferric_scf::ScfResult,
) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
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
fn h2o_hirshfeld_sum_rule_and_symmetry() {
    let (mol, obs, obs_bs, dfbs, op, rhf) = setup_h2o_ccpvdz();

    let mut cfg = PdepRpaConfig::default();
    cfg.frozen_core = 0;
    cfg.trunc_thresh = 0.0;
    cfg.davidson_conv_thresh = 1e-10;

    let alpha_atomic =
        pdep_polarizability_hirshfeld(&mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg).unwrap();
    let alpha_mol = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, op, &cfg)
        .unwrap()
        .tensor;

    assert_eq!(alpha_atomic.len(), mol.atoms.len());

    // Per-atom symmetry < 1e-5 (consumer schema tolerance).
    for (a, alpha_a) in alpha_atomic.iter().enumerate() {
        let mut asym = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                asym = asym.max((alpha_a[i][j] - alpha_a[j][i]).abs());
            }
        }
        assert!(
            asym < 1e-5,
            "atom {a} symmetry violation {asym:.3e} > 1e-5"
        );
    }

    // Sum rule < 1e-3 a.u. per component.
    let mut sum = [[0.0_f64; 3]; 3];
    for alpha_a in &alpha_atomic {
        for i in 0..3 {
            for j in 0..3 {
                sum[i][j] += alpha_a[i][j];
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            let d = (sum[i][j] - alpha_mol[i][j]).abs();
            assert!(
                d < 1e-3,
                "sum rule fails at ({i},{j}): Σ={:.6} vs α_mol={:.6} (Δ={:.2e})",
                sum[i][j], alpha_mol[i][j], d
            );
        }
    }

    // Chemical sanity: oxygen (Z=8) should dominate over hydrogens (Z=1).
    // Locate O atom and the two H atoms.
    let mut o_idx = usize::MAX;
    let mut h_idxs = Vec::new();
    for (a, atom) in mol.atoms.iter().enumerate() {
        match atom.z {
            8 => o_idx = a,
            1 => h_idxs.push(a),
            _ => {}
        }
    }
    assert_ne!(o_idx, usize::MAX, "no oxygen found in water.xyz");
    let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
    let iso_o = iso(&alpha_atomic[o_idx]);
    for &h in &h_idxs {
        let iso_h = iso(&alpha_atomic[h]);
        assert!(
            iso_o > iso_h,
            "expected α_iso(O)={iso_o:.3} > α_iso(H{h})={iso_h:.3}"
        );
    }

    eprintln!(
        "H2O Hirshfeld α_iso: O={:.3}, H={:.3},{:.3} (mol={:.3})",
        iso_o,
        iso(&alpha_atomic[h_idxs[0]]),
        iso(&alpha_atomic[h_idxs[1]]),
        (alpha_mol[0][0] + alpha_mol[1][1] + alpha_mol[2][2]) / 3.0,
    );
}
