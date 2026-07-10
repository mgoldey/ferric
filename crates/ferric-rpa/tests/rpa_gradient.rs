//! Validation tests for the closed-shell RI-RPA correlation gradient.
//!
//! These tests are expensive (multiple PDEP-RPA runs per displacement);
//! the H2O/cc-pVDZ test takes ~30-60 s.  The cc-pVTZ test is `#[ignore]`d
//! by default — run with `--ignored` to include it.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::gradient::rpa_correlation_gradient;
use ferric_rpa::PdepRpaConfig;

fn h2o() -> Molecule {
    let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
    Molecule::parse_xyz(xyz, 0, 1).unwrap()
}

fn small_rpa_cfg(n_quad: usize) -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 1e-4,
        eigensolver_conv_thresh: 1e-10,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: n_quad,
            u0: 0.5,
        },
        ..Default::default()
    }
}

#[test]
fn rpa_gradient_h2o_sto3g_translational_invariance() {
    let mol = h2o();
    let obs_bs = basis::bundled("sto-3g").unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let cfg = small_rpa_cfg(12);

    let grad = rpa_correlation_gradient(&mol, &obs_bs, &aux_bs, op, &cfg, 5e-4).unwrap();

    eprintln!("H2O/STO-3G RPA correlation gradient (Ha/Bohr):");
    for (a, row) in grad.outer_iter().enumerate() {
        eprintln!(
            "  atom {a}: [{:+.8}, {:+.8}, {:+.8}]",
            row[0], row[1], row[2]
        );
    }

    // Translational invariance: sum over atoms must be zero per coordinate.
    for c in 0..3 {
        let s: f64 = (0..3).map(|a| grad[(a, c)]).sum();
        assert!(
            s.abs() < 1e-6,
            "translational invariance violated: coord {c}, sum {s:.3e}"
        );
    }
}

#[test]
fn rpa_gradient_h2_ccpvdz_symmetry() {
    // H2 along z: x,y components must vanish; z components equal/opposite.
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let cfg = small_rpa_cfg(16);

    let grad = rpa_correlation_gradient(&mol, &obs_bs, &aux_bs, op, &cfg, 5e-4).unwrap();
    eprintln!("H2/cc-pVDZ RPA correlation gradient (Ha/Bohr):");
    for (a, row) in grad.outer_iter().enumerate() {
        eprintln!(
            "  atom {a}: [{:+.8}, {:+.8}, {:+.8}]",
            row[0], row[1], row[2]
        );
    }

    for a in 0..2 {
        for c in 0..2 {
            assert!(
                grad[(a, c)].abs() < 1e-6,
                "H2: x,y must vanish (atom {a} coord {c} = {:.3e})",
                grad[(a, c)]
            );
        }
    }
    assert!(
        (grad[(0, 2)] + grad[(1, 2)]).abs() < 1e-6,
        "H2: z must be equal/opposite ({:.3e} vs {:.3e})",
        grad[(0, 2)],
        grad[(1, 2)]
    );
}

#[test]
#[ignore] // ~2 min runtime
fn rpa_gradient_h2o_ccpvdz_fd_self_consistent() {
    // Compare gradient at h=5e-4 vs h=2.5e-4.  Central FD has O(h²)
    // error so the two should agree to ≤ a few e-5 Ha/Bohr.
    let mol = h2o();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let cfg = small_rpa_cfg(16);

    let g1 = rpa_correlation_gradient(&mol, &obs_bs, &aux_bs, op, &cfg, 5e-4).unwrap();
    let g2 = rpa_correlation_gradient(&mol, &obs_bs, &aux_bs, op, &cfg, 2.5e-4).unwrap();

    eprintln!("=== H2O/cc-pVDZ RPA gradient: FD step consistency ===");
    let mut max = 0.0f64;
    for a in 0..3 {
        for c in 0..3 {
            let d = (g1[(a, c)] - g2[(a, c)]).abs();
            max = max.max(d);
            eprintln!(
                "  atom={a} coord={c}: h=5e-4 {:+.8} h=2.5e-4 {:+.8} diff {:.2e}",
                g1[(a, c)],
                g2[(a, c)],
                d
            );
        }
    }
    eprintln!("  max diff = {:.2e}", max);
    assert!(max < 1e-4, "FD consistency failed: max diff {max:.2e}");
}

#[test]
#[ignore] // ~5-10 min runtime: small geometry-optimization smoke test
fn rpa_optimize_h2_ccpvdz() {
    use ferric_rpa::optimize::optimize_geometry_rpa;
    use ferric_scf::optimize::OptimizeConfig;

    // Start from a stretched H2: 0.8 Å (RPA/cc-pVDZ equilibrium is ~0.74 Å).
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.80\n", 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let cfg = small_rpa_cfg(12);

    let opt_cfg = OptimizeConfig {
        max_steps: 15,
        g_max_thresh: 1e-3,
        g_rms_thresh: 1e-3,
        e_conv: 1e-7,
        trust_radius: 0.1,
    };

    let res = optimize_geometry_rpa(&mol, &obs_bs, &aux_bs, op, &cfg, &opt_cfg, 5e-4).unwrap();
    let dz = (res.mol.atoms[0].zpos - res.mol.atoms[1].zpos).abs();
    eprintln!(
        "H2/cc-pVDZ RPA opt: steps={} converged={} bond={:.6} Bohr final E={:.10}",
        res.steps, res.converged, dz, res.energy
    );
    // RPA/cc-pVDZ H2 bond length is ~1.40 Bohr (close to HF/CCSD value)
    assert!(
        res.converged,
        "RPA H2 optimization did not converge in {} steps",
        res.steps
    );
    assert!(
        (dz - 1.40).abs() < 0.05,
        "RPA H2 bond {dz:.4} Bohr, expected near 1.40"
    );
}

#[test]
#[ignore] // ~3 min runtime
fn rpa_gradient_h2o_ccpvdz_vs_pyscf() {
    // Compare ferric projection-fixed RPA gradient vs PySCF central-FD
    // reference (testdata/reference/h2o_cc-pvdz_rpa_grad.json).
    let mol = h2o();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let cfg = small_rpa_cfg(16);

    let grad = rpa_correlation_gradient(&mol, &obs_bs, &aux_bs, op, &cfg, 5e-4).unwrap();

    let ref_json = std::fs::read_to_string("../../testdata/reference/h2o_cc-pvdz_rpa_grad.json")
        .or_else(|_| std::fs::read_to_string("testdata/reference/h2o_cc-pvdz_rpa_grad.json"))
        .expect("read PySCF reference");
    let v: serde_json::Value = serde_json::from_str(&ref_json).unwrap();
    let g_ref = v["gradient_ha_per_bohr"].as_array().unwrap();

    eprintln!("=== H2O/cc-pVDZ ferric vs PySCF reference ===");
    let mut max = 0.0f64;
    for a in 0..3 {
        let row = g_ref[a].as_array().unwrap();
        for c in 0..3 {
            let r = row[c].as_f64().unwrap();
            let d = (grad[(a, c)] - r).abs();
            max = max.max(d);
            eprintln!(
                "  atom={a} coord={c}: ferric {:+.8} pyscf {:+.8} diff {:.2e}",
                grad[(a, c)],
                r,
                d
            );
        }
    }
    eprintln!("  max diff = {:.2e}", max);
    assert!(max < 1e-4, "ferric vs PySCF max diff {max:.2e} > 1e-4");
}

#[test]
#[ignore] // ~10+ min runtime
fn rpa_gradient_h2o_ccpvtz_fd_self_consistent() {
    let mol = h2o();
    let obs_bs = basis::bundled("cc-pvtz").unwrap();
    let aux_bs = basis::bundled("cc-pvtz-ri").or_else(|_| basis::bundled("cc-pvdz-ri")).unwrap();
    let op = Operator::coulomb();
    let cfg = small_rpa_cfg(20);

    let g1 = rpa_correlation_gradient(&mol, &obs_bs, &aux_bs, op, &cfg, 5e-4).unwrap();
    let g2 = rpa_correlation_gradient(&mol, &obs_bs, &aux_bs, op, &cfg, 2.5e-4).unwrap();

    eprintln!("=== H2O/cc-pVTZ RPA gradient: FD step consistency ===");
    let mut max = 0.0f64;
    for a in 0..3 {
        for c in 0..3 {
            let d = (g1[(a, c)] - g2[(a, c)]).abs();
            max = max.max(d);
            eprintln!(
                "  atom={a} coord={c}: h=5e-4 {:+.8} h=2.5e-4 {:+.8} diff {:.2e}",
                g1[(a, c)],
                g2[(a, c)],
                d
            );
        }
    }
    eprintln!("  max diff = {:.2e}", max);
    assert!(max < 1e-4, "FD consistency failed: max diff {max:.2e}");
}
