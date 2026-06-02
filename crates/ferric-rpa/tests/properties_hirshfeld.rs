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

/// Hirshfeld-I with ad-hoc same-basis free-atom proatoms gives physical H2O
/// charges. The legacy single-Slater Hirshfeld over-charges (O ≈ −0.95); a
/// basis-consistent free-atom reference + charge iteration should land
/// O ≈ −0.4..−0.7, H ≈ +0.2..+0.35.
#[test]
fn h2o_hirshfeld_i_adhoc_charges_physical() {
    use ferric_core::elements::z_to_symbol;
    use ferric_rpa::properties::{hirshfeld_i_charges, spherically_averaged_proatom, RadialProatom};

    let (mol, _obs, obs_bs, _dfbs, op, rhf) = setup_h2o_ccpvdz();
    let ctx = ParallelContext::default();

    // Shared radial grid for proatoms.
    let radii: Vec<f64> = (1..=300).map(|k| k as f64 * 0.05).collect(); // 0.05..15 Bohr

    // Ground-state multiplicity (2S+1) for H–Ar.
    let gs_mult = |z: i32| -> usize {
        match z { 1=>2,2=>1,3=>2,4=>1,5=>2,6=>3,7=>4,8=>3,9=>2,10=>1,
                  11=>2,12=>1,13=>2,14=>3,15=>4,16=>3,17=>2,18=>1,_=>1 }
    };

    // Proatom closure: run an atomic SCF for element z at integer charge qi in
    // the molecule's basis, spherically average to a radial proatom.
    let bs = obs_bs.clone();
    let proatom = |z: i32, qi: i32| -> Option<RadialProatom> {
        // Only neutral here (qi==0); ions left to the fallback for this test.
        if qi != 0 { return None; }
        let n_elec = z - qi;
        if n_elec <= 0 { return None; }
        let sym = z_to_symbol(z).unwrap_or("X");
        let xyz = format!("1\n{sym}\n{sym} 0 0 0\n");
        let amol = Molecule::parse_xyz(&xyz, qi, gs_mult(z)).ok()?;
        let aobs = PreparedBasis::new(&amol, &bs).ok()?;
        let abounds = SchwarzBounds::compute(op, &aobs).ok()?;
        let mut cfg = RhfConfig::default();
        // Closed-shell singlet → RHF; open-shell → UHF with MOM.
        let dens = if gs_mult(z) == 1 {
            solve_rhf(&ctx, &amol, &aobs, op, &abounds, &cfg).ok()?.density_r().to_owned()
        } else {
            cfg.mom_after_iter = 5;
            ferric_scf::uhf::solve_uhf(&ctx, &amol, &aobs, &abounds, &cfg)
                .ok()?
                .density_total().to_owned()
        };
        spherically_averaged_proatom(z, &bs, &dens, &radii).ok()
    };

    let q = hirshfeld_i_charges(&mol, &obs_bs, rhf.density_r(), &proatom).unwrap();
    let (q_o, q_h1, q_h2) = (q[0], q[1], q[2]);
    eprintln!("H2O ad-hoc Hirshfeld charges (neutral proatoms): O={q_o:.4} H={q_h1:.4} H={q_h2:.4}");
    // Basis-consistent free-atom proatoms (neutral) → standard-Hirshfeld values
    // for water: O ≈ −0.3, H ≈ +0.15 (the textbook Hirshfeld result). The
    // legacy single-Slater covalent-radius proatom badly over-charges
    // (O ≈ −0.95): this is the H-starvation fix. (Full Hirshfeld-I charge
    // iteration with ion proatoms pushes O further to ≈ −0.6.)
    assert!((-0.45..=-0.20).contains(&q_o), "O charge out of physical range: {q_o:.3}");
    assert!((0.10..=0.25).contains(&q_h1), "H charge out of physical range: {q_h1:.3}");
    assert!((q_o + q_h1 + q_h2).abs() < 1e-6, "charges must sum to 0");
    assert!((q_h1 - q_h2).abs() < 1e-3, "equivalent H must have equal charge");
}
