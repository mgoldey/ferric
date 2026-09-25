//! Stage-1 PBC step 4: `PeriodicHcore` (reference/pbc/stage1-design.md §3,
//! option (b) — Gaussian nuclei through the shifted 3-centre engine, because
//! libint2 2.7.2's erf/erfc_nuclear are wrong; FINDINGS "Iteration 2").
//!
//! Anchors, cheapest first:
//! 1. `gaussian_nucleus_erf_plus_erfc_is_point_nuclear` — V_erf + V_erfc of
//!    Gaussian nuclei ≡ libint2's point-charge OP_NUCLEAR (molecular, water
//!    cc-pVDZ) plus the ζ-sweep that measures the finite-nucleus model and
//!    checks libint2 survives ζ = 1e16. BLIND to an erf↔erfc mix-up or a
//!    libint2 kernel bug that preserves the sum (that is how the
//!    erf_nuclear bug hid); test 2 covers that.
//! 2. `single_s_primitive_matches_closed_form` — the test that caught the
//!    libint2 bug: one normalised s primitive (α = 1.3) on its own nucleus,
//!    `V_erf(ω) = −(2/√π)·(1/ω² + 1/p + 1/ζ)^{−1/2}`, p = 2α.
//! 3. `periodic_hcore_matches_prototype_h2_sto3g_a4` — S, T, V, h, E_nn vs
//!    `pbc_gamma.py` (pure-AFT, ω-free) for H2/STO-3G in the a = 4 cube.
//! 4. `periodic_hcore_is_omega_independent_triclinic_sp` — V(ω) for three ω
//!    vs the pure reciprocal-space `pure_aft_nuclear` (independent: no
//!    libint2, no SR sum, no G = 0 bookkeeping), s+p triclinic 4H; plus the
//!    negative control that the G = 0 term is visible at this tolerance.
//!
//! Artifact hypotheses: a G = 0 bookkeeping error is `c·Z_tot·S/(ω²Ω)`-shaped
//! and ω-DEPENDENT, so it fails 3 and 4 at ω ≠ 1 (tests avoid ω = 1, where
//! `1/ω` vs `1/ω²` coincide); SR truncation would show as a residual that
//! shrinks with `precision` (printed); the sign of the L/M shifts is
//! invisible to any Gamma lattice sum (image sets are closed under negation)
//! — the direction anchors for that live in
//! `ferric-integrals/tests/eri3_shifted.rs`.
//!
//! Mutation plan (each should turn one of these red):
//! * swap `Operator::erfc` → `erf` in periodic_hcore's SR engine: 4 (and 3).
//! * drop `v_g0` or use `π/(ωΩ)`: 3, 4.
//! * `+iG·R` in the structure factor: 3 (atoms are not inversion-symmetric).
//! * drop the factor 2 of the half G sphere: 3, 4.
//! * `norm_int` wrong / ζ ignored: 1, 2.
//! * truncate the SR nucleus images to M = 0: 3, 4.

mod common;

use common::*;
use ferric_core::basis::{bundled, BasisSet, Shell};
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_pbc::hcore::{
    gaussian_nucleus_attraction, periodic_hcore, pure_aft_nuclear, PeriodicHcoreConfig,
    DEFAULT_HCORE_PRECISION, GAUSSIAN_NUCLEUS_EXPONENT,
};
use std::collections::HashMap;
use std::f64::consts::PI;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

fn nuclei_of(prep: &PreparedBasis) -> Vec<(f64, [f64; 3])> {
    prep.atoms()
        .iter()
        .map(|a| (a.atomic_number, [a.x, a.y, a.z]))
        .collect()
}

#[test]
fn gaussian_nucleus_erf_plus_erfc_is_point_nuclear() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &bundled("cc-pvdz").unwrap()).unwrap();
    let nuc = nuclei_of(&prep);
    let v_point = oneelectron::nuclear(&prep);

    // ζ sweep with the plain Coulomb kernel: error model π Z ρ(R_C)/ζ.
    let mut errs = Vec::new();
    for zeta in [1e8, 1e10, 1e12, 1e14, 1e16] {
        let v = gaussian_nucleus_attraction(&prep, &nuc, Operator::coulomb(), zeta).unwrap();
        let e = max_abs_diff(&v, &v_point);
        eprintln!("zeta = {zeta:.0e}: max|V_gauss - V_point| = {e:.3e}");
        errs.push(e);
    }
    // Finite-nucleus regime behaves as the 1/ζ model predicts.
    let ratio = errs[0] / errs[1];
    assert!(
        (30.0..300.0).contains(&ratio),
        "err(1e8)/err(1e10) = {ratio:.1} (1/ζ model predicts ~100)"
    );
    assert!(errs[1] > errs[2], "not monotone 1e10 -> 1e12: {errs:?}");
    // libint2 survives the production exponent.
    assert_eq!(GAUSSIAN_NUCLEUS_EXPONENT, 1e16);
    assert!(errs[4] < 1e-10, "zeta = 1e16 error {:.3e}", errs[4]);

    for omega in [0.3, 1.3, 3.0] {
        let ve = gaussian_nucleus_attraction(
            &prep,
            &nuc,
            Operator::erf(omega),
            GAUSSIAN_NUCLEUS_EXPONENT,
        )
        .unwrap();
        let vc = gaussian_nucleus_attraction(
            &prep,
            &nuc,
            Operator::erfc(omega),
            GAUSSIAN_NUCLEUS_EXPONENT,
        )
        .unwrap();
        let e = max_abs_diff(&(&ve + &vc), &v_point);
        eprintln!("omega = {omega}: max|V_erf + V_erfc - V_point| = {e:.3e}");
        assert!(e < 1e-10, "omega {omega}: {e:.3e}");
        // Non-vacuity: both halves carry weight.
        assert!(ve.iter().any(|x| x.abs() > 1e-2) && vc.iter().any(|x| x.abs() > 1e-2));
    }
}

#[test]
fn single_s_primitive_matches_closed_form() {
    let alpha = 1.3;
    let p = 2.0 * alpha;
    let zeta = GAUSSIAN_NUCLEUS_EXPONENT;
    let r = [0.4, -0.2, 0.7];
    let mol = Molecule {
        atoms: vec![Atom {
            symbol: "H".into(),
            z: 1,
            x: r[0],
            y: r[1],
            zpos: r[2],
            ghost: false,
            n_core_ecp: 0,
        }],
        charge: 0,
        multiplicity: 2,
    };
    let mut shells = HashMap::new();
    shells.insert(
        1,
        vec![Shell {
            l: 0,
            pure: false,
            exponents: vec![alpha],
            coefficients: vec![1.0],
        }],
    );
    let bs = BasisSet {
        name: "single-s".into(),
        shells,
        ecps: HashMap::new(),
    };
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let nuc = [(1.0, r)];
    // Unit-charge density |φ|² (exponent p) against a Gaussian nucleus (ζ)
    // through erf(ωr)/r: the three Gaussian widths add in quadrature.
    let closed_erf = |w: f64| -2.0 / PI.sqrt() / (1.0 / (w * w) + 1.0 / p + 1.0 / zeta).sqrt();
    let closed_coul = -2.0 / PI.sqrt() / (1.0 / p + 1.0 / zeta).sqrt();
    let vc = gaussian_nucleus_attraction(&prep, &nuc, Operator::coulomb(), zeta).unwrap()[(0, 0)];
    assert!(
        (vc - closed_coul).abs() < 1e-12,
        "coulomb {vc} vs {closed_coul}"
    );
    for w in [0.5, 1.5, 4.0] {
        let ve = gaussian_nucleus_attraction(&prep, &nuc, Operator::erf(w), zeta).unwrap()[(0, 0)];
        let vs = gaussian_nucleus_attraction(&prep, &nuc, Operator::erfc(w), zeta).unwrap()[(0, 0)];
        let (ce, cs) = (closed_erf(w), closed_coul - closed_erf(w));
        eprintln!("omega {w}: V_erf {ve:.15} (closed {ce:.15}), V_erfc {vs:.15} (closed {cs:.15})");
        assert!((ve - ce).abs() < 1e-12, "erf omega {w}: {ve} vs {ce}");
        assert!((vs - cs).abs() < 1e-12, "erfc omega {w}: {vs} vs {cs}");
        // The libint2 erf_nuclear defect evaluates the closed form at 2ω;
        // this test can tell the two apart by orders of magnitude.
        assert!((ce - closed_erf(2.0 * w)).abs() > 1e-2);
    }
}

// pbc_gamma.build_integrals(Cell(eye*4, H2, 'sto-3g'), None) (pure AFT,
// ω-free), PySCF sto-3g, measured 2026-09-23 (scratch gen script; the same
// run gives E_ewald -1.6583270610487082, E_none -0.9490026911785538).
const REF_S: [[f64; 2]; 2] = [
    [1.8564558917189407, 1.6425187685488463],
    [1.6425187685488465, 1.8564558917189407],
];
const REF_T: [[f64; 2]; 2] = [
    [0.583166233491849, 0.0909816836297924],
    [0.09098168362979235, 0.583166233491849],
];
const REF_V: [[f64; 2]; 2] = [
    [-0.7787037660791539, -0.5350093475143196],
    [-0.5350093475143196, -0.7787037660791539],
];
const REF_ENN: f64 = -0.6281049380287603;

#[test]
fn periodic_hcore_matches_prototype_h2_sto3g_a4() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let s_ref = array2(&[&REF_S[0], &REF_S[1]]);
    let t_ref = array2(&[&REF_T[0], &REF_T[1]]);
    let v_ref = array2(&[&REF_V[0], &REF_V[1]]);
    let h_ref = &t_ref + &v_ref;
    for omega in [0.8, 1.7] {
        let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(omega)).unwrap();
        let (ds, dt, dv, dh) = (
            max_abs_diff(&hc.s, &s_ref),
            max_abs_diff(&hc.t, &t_ref),
            max_abs_diff(&hc.v, &v_ref),
            max_abs_diff(&hc.h, &h_ref),
        );
        eprintln!(
            "omega {omega}: |dS| {ds:.2e} |dT| {dt:.2e} |dV| {dv:.2e} |dh| {dh:.2e} dEnn {:.2e} \
             (images {}, SR triplets {}, half-G {}, SR asym {:.1e}, |V_G0| {:.3e})",
            hc.enn - REF_ENN,
            hc.n_images,
            hc.n_sr_triplets,
            hc.n_g_half,
            hc.sr_asymmetry,
            hc.v_g0[(0, 0)].abs()
        );
        assert!(ds < 1e-12 && dt < 1e-12, "S/T");
        assert!(dv < 1e-10 && dh < 1e-10, "V/h at omega {omega}");
        assert!((hc.enn - REF_ENN).abs() < 1e-12, "E_nn");
        assert!(
            hc.sr_asymmetry < 1e-12,
            "SR asymmetry {:.3e}",
            hc.sr_asymmetry
        );
        // Non-vacuity: every piece of the split carries weight.
        assert!(hc.v_sr[(0, 0)].abs() > 1e-3 && hc.v_lr[(0, 0)].abs() > 1e-3);
        assert!(hc.v_g0[(0, 0)].abs() > 1e-3);
    }
}

#[test]
fn periodic_hcore_is_omega_independent_triclinic_sp() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    let (v_aft, ng) = pure_aft_nuclear(&cell, &prep, 1e-14).unwrap();
    eprintln!("pure AFT: {ng} half-sphere G");
    for omega in [0.6, 1.3, 1.9] {
        let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(omega)).unwrap();
        let d = max_abs_diff(&hc.v, &v_aft);
        eprintln!(
            "omega {omega}: max|V(omega) - V_AFT| = {d:.2e} at precision {DEFAULT_HCORE_PRECISION:.0e} \
             (SR triplets {}, half-G {}, SR asym {:.1e})",
            hc.n_sr_triplets, hc.n_g_half, hc.sr_asymmetry
        );
        if omega == 1.3 {
            // Truncation diagnostic (printed, not asserted: the bounds are
            // conservative, so both may sit at roundoff): a truncation
            // residual shrinks with precision, a construction error does not.
            let loose = PeriodicHcoreConfig {
                precision: 1e-8,
                ..PeriodicHcoreConfig::with_omega(omega)
            };
            let d_loose = max_abs_diff(&periodic_hcore(&cell, &prep, &loose).unwrap().v, &v_aft);
            eprintln!("omega {omega}: same at precision 1e-8: {d_loose:.2e}");
        }
        assert!(d < 1e-9, "omega {omega}: {d:.3e}");
        // Negative control: the G = 0 correction is load-bearing at this
        // tolerance (a test that cannot see it could not catch its defects).
        let without_g0 = &hc.v - &hc.v_g0;
        assert!(max_abs_diff(&without_g0, &v_aft) > 1e-3);
    }
}

#[test]
fn periodic_hcore_rejects_bad_config_and_foreign_basis() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    for bad in [0.0, -1.0, f64::NAN] {
        assert!(periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(bad)).is_err());
    }
    let cfg = PeriodicHcoreConfig {
        precision: 2.0,
        ..PeriodicHcoreConfig::with_omega(1.0)
    };
    assert!(periodic_hcore(&cell, &prep, &cfg).is_err());
    // A basis built on a different geometry is refused, not silently used.
    let other = h2_cell(5.0);
    let moved = Molecule {
        atoms: other
            .mol()
            .atoms
            .iter()
            .map(|a| {
                let mut b = a.clone();
                b.x += 0.5;
                b
            })
            .collect(),
        ..other.mol().clone()
    };
    let prep_moved = PreparedBasis::new(&moved, &pyscf_sto3g_h()).unwrap();
    assert!(periodic_hcore(&cell, &prep_moved, &PeriodicHcoreConfig::with_omega(1.0)).is_err());
}

/// A shell above md3c1e::MAX_L (an h shell, l = 5) is a typed error, not a
/// DFACT index panic inside prim_norm (CodeRabbit, PR #150).
#[test]
fn periodic_hcore_rejects_l_above_max_l() {
    let cell = h2_cell(4.0);
    let h_shell = Shell {
        l: 5,
        pure: true,
        exponents: vec![0.8],
        coefficients: vec![1.0],
    };
    let mut bs = pyscf_sto3g_h();
    for shells in bs.shells.values_mut() {
        shells.push(h_shell.clone());
    }
    let prep = PreparedBasis::new(cell.mol(), &bs).expect("libint accepts l=5");
    let err = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(1.0))
        .expect_err("l=5 must be rejected");
    assert!(err.to_string().contains("l=5"), "{err}");
}
