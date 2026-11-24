//! Tests of `ferric_scf::x2c`, the spin-free X2C-1e core Hamiltonian.
//!
//! * the contraction matrix maps the uncontracted S, T, V onto the
//!   contracted ones (checks the AO order of the two bases);
//! * the nonrelativistic limit: `h(c) - (T + V)` falls as `c^-2`, and at
//!   large `c` it reaches `T + V` within the leading term plus rounding;
//! * generalized eigenvalues of `(h, S)` against PySCF `sfx2c1e`
//!   (`testdata/reference/x2c_sfx2c1e_cc-pvdz.json`,
//!   `scripts/gen_pyscf_x2c_refs.py`) on Ne, Ar, Kr and HBr / cc-pVDZ;
//! * refusals: ECP atoms, a non-positive speed of light.
//!
//! The c-limit is blind to the sign of W (the leading term still scales as
//! `c^-2`); only the PySCF comparison detects it
//! (`scripts/prototypes/x2c_derivation.md`).

use ferric_core::basis::{bundled, BasisSet};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::libint_max_deriv_order;
use ferric_integrals::{oneelectron, pvp};
use ferric_scf::x2c::{
    contraction_matrix, uncontracted_basis, x2c1e_hcore, x2c1e_transform, LIGHT_SPEED,
};
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

/// libint2's engine precision (1e-14 per primitive integral) times 10 and
/// the three terms of W: the per-element rounding bar of
/// `crates/ferric-integrals/tests/pvp.rs`, also the prototype's `ROUND`.
const ROUND: f64 = 3e-13;

fn skip() -> bool {
    if libint_max_deriv_order() < 2 {
        eprintln!("SKIP: libint2 generated without second derivatives (no pVp)");
        return true;
    }
    false
}

fn mol(xyz_angstrom: &str) -> Molecule {
    let lines: Vec<&str> = xyz_angstrom.lines().collect();
    Molecule::parse_xyz(&format!("{}\n\n{}\n", lines.len(), xyz_angstrom), 0, 1).unwrap()
}

fn system(label: &str) -> Molecule {
    mol(match label {
        "Ne" => "Ne 0 0 0",
        "Ar" => "Ar 0 0 0",
        "Kr" => "Kr 0 0 0",
        "HBr" => "H 0 0 0\nBr 0 0 1.414",
        _ => unreachable!(),
    })
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0, |m, x| m.max(x.abs()))
}

fn fro(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

struct Unc {
    s: Array2<f64>,
    t: Array2<f64>,
    v: Array2<f64>,
    w: Array2<f64>,
    c: Array2<f64>,
}

fn uncontracted(m: &Molecule, bs: &BasisSet) -> Unc {
    let unc = uncontracted_basis(bs).unwrap();
    let prep = PreparedBasis::new(m, &unc).unwrap();
    Unc {
        s: oneelectron::overlap(&prep),
        t: oneelectron::kinetic(&prep),
        v: oneelectron::nuclear(&prep),
        w: pvp::pvp_nuclear(&prep).unwrap(),
        c: contraction_matrix(m, bs).unwrap(),
    }
}

/// `C^T (h(c) - T - V) C`, the X2C correction in the contracted basis.
fn delta_h(u: &Unc, c: f64) -> Array2<f64> {
    let h = x2c1e_transform(&u.s, &u.t, &u.v, &u.w, c).unwrap();
    let d = u.c.t().dot(&(h - &u.t - &u.v)).dot(&u.c);
    0.5 * (&d + &d.t())
}

/// Generalized eigenvalues of `(a, s)` and their first-order rounding bound
/// `n ROUND (max|a| + |lambda| max|s|) / lambda_min(s)` (the prototype's
/// `gen_eig_bar`; the same bound is stored for the PySCF side).
fn gen_eig_bar(a: &Array2<f64>, s: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let n = s.nrows() as f64;
    let (es, _) = s.eigh(UPLO::Lower).unwrap();
    let lam_min = es[0];
    let (lam, _) = (a.clone(), s.clone()).eigh(UPLO::Lower).unwrap();
    let (amax, smax) = (max_abs(a), max_abs(s));
    let bar = lam.mapv(|l| n * ROUND * (amax + l.abs() * smax) / lam_min);
    (lam, bar)
}

#[test]
fn contraction_matrix_maps_uncontracted_s_t_v_onto_contracted() {
    if skip() {
        return;
    }
    let bs = bundled("cc-pvdz").unwrap();
    for label in ["Ne", "HBr"] {
        let m = system(label);
        let u = uncontracted(&m, &bs);
        let prep = PreparedBasis::new(&m, &bs).unwrap();
        for (name, unc, con) in [
            ("S", &u.s, oneelectron::overlap(&prep)),
            ("T", &u.t, oneelectron::kinetic(&prep)),
            ("V", &u.v, oneelectron::nuclear(&prep)),
        ] {
            let dev = max_abs(&(u.c.t().dot(unc).dot(&u.c) - &con));
            // Each contracted element is a sum over at most n_unc^2 products
            // of ROUND-accurate primitive integrals.
            let bar = unc.nrows() as f64 * ROUND * max_abs(&con);
            eprintln!("{label} {name}: max dev {dev:.2e} (bar {bar:.2e})");
            assert!(
                dev <= bar,
                "{label} {name}: C^T X_unc C - X_con = {dev:e} > {bar:e}"
            );
        }
    }
}

/// `||h(c f) - (T+V)|| / ||h(10 c f) - (T+V)||` for f = 1, 10, 100 is
/// `100 (1 + O(c^-2))`: the deviation from 100 shrinks about 100x per decade.
/// Measured (Ne) 92.9, 99.84, 99.998; (Kr) 76.5, 98.15, 99.970.
#[test]
fn x2c_correction_falls_as_c_minus_two() {
    if skip() {
        return;
    }
    let bs = bundled("cc-pvdz").unwrap();
    for label in ["Ne", "Kr"] {
        let u = uncontracted(&system(label), &bs);
        let norms: Vec<f64> = [1.0, 10.0, 100.0, 1000.0]
            .iter()
            .map(|f| fro(&delta_h(&u, LIGHT_SPEED * f)))
            .collect();
        let ratios: Vec<f64> = (0..3).map(|i| norms[i] / norms[i + 1]).collect();
        let dev: Vec<f64> = ratios.iter().map(|r| (r / 100.0 - 1.0).abs()).collect();
        let shown: Vec<String> = norms.iter().map(|x| format!("{x:.4e}")).collect();
        eprintln!("{label}: ||dH|| {shown:?}, ratios {ratios:.4?}");
        // The prototype's criterion (anchor_cscan).
        assert!(dev[1] < 0.05, "{label}: ratio at 10c is {}", ratios[1]);
        assert!(dev[2] < dev[1] / 10.0 + 1e-6, "{label}: ratios {ratios:?}");
    }
}

/// At c = 1e6 x LIGHT_SPEED, `x2c1e_hcore` is ferric's `T + V` within
/// (a) the leading term, `max|dH(100 c0)|` scaled by `(100 c0 / c)^2`, with
///     1% for the O(c^-2) relative correction (measured at most 3e-4 at
///     100 c0, Kr);
/// (b) the contraction rounding, the bar of the contraction test;
/// (c) the rounding of the transformation, `n_unc eps cond(S_u) max|T_u + V_u|`
///     (the transformation applies S_u^(+-1/2) to matrices of that size).
#[test]
fn nonrelativistic_limit_reproduces_hcore() {
    if skip() {
        return;
    }
    let bs = bundled("cc-pvdz").unwrap();
    for label in ["Ne", "Kr", "HBr"] {
        let m = system(label);
        let u = uncontracted(&m, &bs);
        let prep = PreparedBasis::new(&m, &bs).unwrap();
        let h0 = oneelectron::hcore(&prep);
        let f = 1e6;
        let h = x2c1e_hcore(&m, &bs, LIGHT_SPEED * f).unwrap();
        let dev = max_abs(&(&h - &h0));
        let n = u.s.nrows() as f64;
        let (es, _) = u.s.eigh(UPLO::Lower).unwrap();
        let cond = es[es.len() - 1] / es[0];
        let leading = 1.01 * max_abs(&delta_h(&u, LIGHT_SPEED * 100.0)) * (100.0 / f).powi(2);
        let contract = n * ROUND * max_abs(&h0);
        let transform = n * f64::EPSILON * cond * max_abs(&(&u.t + &u.v));
        let bar = leading + contract + transform;
        eprintln!(
            "{label}: max|h(c) - hcore| {dev:.3e}, bar {bar:.3e} \
             (leading {leading:.2e}, contraction {contract:.2e}, transform {transform:.2e})"
        );
        assert!(dev <= bar, "{label}: {dev:e} > {bar:e}");
    }
}

#[derive(serde::Deserialize)]
struct RefSystem {
    nao: usize,
    hcore_eigenvalues: Vec<f64>,
    hcore_eig_bar: Vec<f64>,
}

#[derive(serde::Deserialize)]
struct RefFile {
    light_speed: f64,
    systems: std::collections::HashMap<String, RefSystem>,
}

fn reference() -> RefFile {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/x2c_sfx2c1e_cc-pvdz.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Generalized eigenvalues of (h_X2C, S) against PySCF `sfx2c1e` at the same
/// c, uncontracted basis and cc-pVDZ data: |d lambda| at most the sum of the
/// two codes' first-order rounding bounds (`gen_eig_bar`). Measured by the
/// prototype: 2.2e-11 to 4.4e-11 Ha against bars of 4.6e-9 to 5.8e-7.
#[test]
fn hcore_eigenvalues_match_pyscf_sfx2c1e() {
    if skip() {
        return;
    }
    let refs = reference();
    assert_eq!(refs.light_speed, LIGHT_SPEED);
    let bs = bundled("cc-pvdz").unwrap();
    for label in ["Ne", "Ar", "Kr", "HBr"] {
        let r = &refs.systems[label];
        let m = system(label);
        let prep = PreparedBasis::new(&m, &bs).unwrap();
        let s = oneelectron::overlap(&prep);
        let h = x2c1e_hcore(&m, &bs, LIGHT_SPEED).unwrap();
        assert_eq!(h.nrows(), r.nao, "{label}: nao");
        let (lam, bar) = gen_eig_bar(&h, &s);
        let mut worst = 0.0f64;
        let mut max_dev = 0.0f64;
        for i in 0..r.nao {
            let dev = (lam[i] - r.hcore_eigenvalues[i]).abs();
            max_dev = max_dev.max(dev);
            worst = worst.max(dev / (bar[i] + r.hcore_eig_bar[i]));
        }
        eprintln!(
            "{label}: lowest eig {:.10} (PySCF {:.10}), max |d eig| {max_dev:.2e}, worst dev/bar {worst:.2e}",
            lam[0], r.hcore_eigenvalues[0]
        );
        assert!(
            worst <= 1.0,
            "{label}: eigenvalue deviation {worst:e} x its bar"
        );
    }
}

#[test]
fn ecp_atom_is_refused() {
    let bs = bundled("cc-pvdz-pp").unwrap();
    let m = mol("Xe 0 0 0");
    assert!(
        bs.ecp_for_element(54).is_some(),
        "fixture: Xe must carry an ECP"
    );
    let err = x2c1e_hcore(&m, &bs, LIGHT_SPEED).unwrap_err().to_string();
    assert!(err.contains("ECP"), "{err}");
}

#[test]
fn nonpositive_or_nonfinite_light_speed_is_refused() {
    let bs = bundled("sto-3g").unwrap();
    let m = system("Ne");
    for c in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(x2c1e_hcore(&m, &bs, c).is_err(), "c = {c} accepted");
    }
}
