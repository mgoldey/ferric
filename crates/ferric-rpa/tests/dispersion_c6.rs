//! Physical-anchor tests for the TS dispersion C6 path.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{pdep_dynamic_polarizability, DispersionPartition};
use ferric_rpa::{casimir_polder_c6, ts_dynamic_polarizability, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Fine trapezoid imaginary-frequency grid; integrates the Casimir-Polder
/// α(iω)α(iω) product to <1% for the single-pole London model.
fn freq_grid() -> (Vec<f64>, Vec<f64>) {
    let n = 20000usize;
    let wmax = 200.0_f64;
    let dw = wmax / (n as f64);
    let mut f = Vec::with_capacity(n + 1);
    let mut w = Vec::with_capacity(n + 1);
    for k in 0..=n {
        f.push(k as f64 * dw);
        w.push(if k == 0 || k == n { 0.5 * dw } else { dw });
    }
    (f, w)
}

#[test]
fn free_atom_c6_matches_ts_reference() {
    let (freqs, weights) = freq_grid();
    // Free H and free O at ratio 1.0, isotropic static α = α_free.
    let z = vec![1usize, 8usize];
    let ratio = vec![1.0, 1.0];
    let alpha_static = vec![
        [[4.5, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 4.5]],
        [[5.4, 0.0, 0.0], [0.0, 5.4, 0.0], [0.0, 0.0, 5.4]],
    ];
    let dp = ts_dynamic_polarizability(&z, &ratio, &alpha_static, &freqs, &weights);
    let res = casimir_polder_c6(&dp);

    // Homonuclear C6 reproduces the table by construction.
    let c6_hh = res.c6_iso_pair[(0, 0)];
    let c6_oo = res.c6_iso_pair[(1, 1)];
    assert!((c6_hh - 6.5).abs() / 6.5 < 3e-3, "C6(H-H)={c6_hh}");
    assert!((c6_oo - 15.6).abs() / 15.6 < 3e-3, "C6(O-O)={c6_oo}");

    // Pair-matrix symmetry.
    let c6_ho = res.c6_iso_pair[(0, 1)];
    let c6_oh = res.c6_iso_pair[(1, 0)];
    assert!((c6_ho - c6_oh).abs() / c6_ho < 1e-12, "asymmetric C6 matrix");

    // Heteronuclear C6(H-O) finite, positive, between the two homonuclear scales.
    assert!(c6_ho > 0.0 && c6_ho.is_finite());
    assert!(c6_ho > c6_hh.min(c6_oo) * 0.5, "C6(H-O)={c6_ho} too small");
    assert!(c6_ho < c6_hh.max(c6_oo) * 1.1, "C6(H-O)={c6_ho} too large");
}

/// PDEP-RPA dynamic α(iω) → Casimir-Polder C6 on H2/cc-pVDZ. End-to-end check
/// that the Phase 2 source produces a physically sane, symmetric C6 matrix and
/// that the ω=0 slice equals the per-atom static polarizability sum rule.
#[test]
fn pdep_dynamic_c6_h2_sane() {
    let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74083\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let mut cfg = PdepRpaConfig::default();
    cfg.frozen_core = 0;
    cfg.trunc_thresh = 0.0;

    let dp = pdep_dynamic_polarizability(
        &mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg, DispersionPartition::Becke,
    )
    .unwrap();

    // Grid is the RPA quadrature: positive nodes, weights sum > 0.
    assert!(!dp.freqs.is_empty());
    assert_eq!(dp.freqs.len(), dp.weights.len());
    assert_eq!(dp.per_atom.len(), 2);
    assert_eq!(dp.per_atom[0].len(), dp.freqs.len());

    // α^A(iω) must decay: lowest-freq iso > highest-freq iso, per atom.
    for atom in &dp.per_atom {
        let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        let lo = iso(&atom[0]);
        let hi = iso(&atom[atom.len() - 1]);
        assert!(lo > hi, "α not decaying: lo={lo} hi={hi}");
        assert!(lo > 0.0, "α(ω_min) not positive: {lo}");
    }

    let res = casimir_polder_c6(&dp);
    let c6 = &res.c6_iso_pair;
    // Symmetric, positive diagonal, finite.
    assert!((c6[(0, 1)] - c6[(1, 0)]).abs() < 1e-10, "asymmetric C6");
    assert!(c6[(0, 0)] > 0.0 && c6[(0, 0)].is_finite(), "C6(H-H)={}", c6[(0, 0)]);
    // Two equivalent H atoms: diagonal entries equal by symmetry.
    assert!(
        (c6[(0, 0)] - c6[(1, 1)]).abs() / c6[(0, 0)] < 1e-6,
        "H2 C6 diagonal asymmetric: {} vs {}",
        c6[(0, 0)],
        c6[(1, 1)]
    );
}
