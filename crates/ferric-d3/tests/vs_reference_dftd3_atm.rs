//! D3(BJ) energy AND gradient, with the Axilrod-Teller-Muto three-body term
//! on (`s9 = 1`) and off (`s9 = 0`), against simple-dftd3 (the `dftd3` Python
//! package), an independent Fortran implementation.
//!
//! `testdata/d3bj_atm_reference.json` is written by
//! `scripts/gen_d3_atm_reference.py`, which passes `(s6, s8, a1, a2, s9)`
//! explicitly to `RationalDampingParam` (alp = 14, so the three-body damping
//! exponent is alp + 2 = 16).
//!
//! The three-body part is compared on its own as well as inside the total:
//! it is 1e-10..1e-5 Ha, so a total-energy tolerance alone would not see a
//! wrong three-body term on the small systems.

use ferric_d3::{d3bj_energy, d3bj_gradient, D3Params};
use serde_json::Value;

struct Case {
    system: String,
    functional: String,
    numbers: Vec<u8>,
    coords: Vec<[f64; 3]>,
    params: D3Params,
    /// (energy, gradient) at s9 = 0 and s9 = 1.
    off: (f64, Vec<[f64; 3]>),
    on: (f64, Vec<[f64; 3]>),
}

fn rows(v: &Value) -> Vec<[f64; 3]> {
    v.as_array()
        .expect("rows")
        .iter()
        .map(|r| {
            let r = r.as_array().expect("row");
            [
                r[0].as_f64().unwrap(),
                r[1].as_f64().unwrap(),
                r[2].as_f64().unwrap(),
            ]
        })
        .collect()
}

fn cases() -> Vec<Case> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/d3bj_atm_reference.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let root: Value = serde_json::from_str(&text).expect("reference JSON");
    let mut out = Vec::new();
    for (system, sys) in root["systems"].as_object().expect("systems") {
        let numbers: Vec<u8> = sys["numbers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|z| z.as_u64().unwrap() as u8)
            .collect();
        let coords = rows(&sys["coords_bohr"]);
        for (functional, f) in sys["functionals"].as_object().unwrap() {
            let p = &f["params"];
            let params = D3Params {
                s6: p["s6"].as_f64().unwrap(),
                s8: p["s8"].as_f64().unwrap(),
                a1: p["a1"].as_f64().unwrap(),
                a2: p["a2"].as_f64().unwrap(),
                s9: 0.0,
            };
            let eg = |k: &str| (f[k]["energy"].as_f64().unwrap(), rows(&f[k]["gradient"]));
            out.push(Case {
                system: system.clone(),
                functional: functional.clone(),
                numbers: numbers.clone(),
                coords: coords.clone(),
                params,
                off: eg("s9_0"),
                on: eg("s9_1"),
            });
        }
    }
    out
}

fn max_abs_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs()))
        .fold(0.0f64, f64::max)
}

fn sub(a: &[[f64; 3]], b: &[[f64; 3]]) -> Vec<[f64; 3]> {
    a.iter()
        .zip(b)
        .map(|(x, y)| [x[0] - y[0], x[1] - y[1], x[2] - y[2]])
        .collect()
}

fn with_s9(p: D3Params, s9: f64) -> D3Params {
    D3Params { s9, ..p }
}

/// Total energy and gradient at s9 = 0 and s9 = 1 match simple-dftd3, and so
/// does the isolated three-body part (s9=1 minus s9=0) of each.
///
/// Tolerances are rounding-level: 1e-12 Ha / 1e-11 Ha/Bohr on totals, and on
/// the three-body part 1e-14 Ha + 1e-9 relative (energy) / 1e-12 Ha/Bohr
/// (gradient). The smallest nonzero three-body energy in the set (water,
/// 2.3e-10 Ha) sits 4 orders above the energy bar.
#[test]
fn energy_and_gradient_match_simple_dftd3_with_and_without_atm() {
    let cases = cases();
    assert_eq!(cases.len(), 27, "9 systems x 3 functionals");
    let (mut w_e, mut w_g, mut w_e3, mut w_g3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut n_nonzero = 0;
    for c in &cases {
        let tag = format!("{}/{}", c.system, c.functional);
        let p0 = with_s9(c.params, 0.0);
        let p1 = with_s9(c.params, 1.0);
        let e0 = d3bj_energy(&c.numbers, &c.coords, &p0).unwrap();
        let e1 = d3bj_energy(&c.numbers, &c.coords, &p1).unwrap();
        let g0 = d3bj_gradient(&c.numbers, &c.coords, &p0).unwrap();
        let g1 = d3bj_gradient(&c.numbers, &c.coords, &p1).unwrap();

        for (label, e, eref, g, gref) in [
            ("s9=0", e0, c.off.0, &g0, &c.off.1),
            ("s9=1", e1, c.on.0, &g1, &c.on.1),
        ] {
            let de = (e - eref).abs();
            let dg = max_abs_diff(g, gref);
            w_e = w_e.max(de);
            w_g = w_g.max(dg);
            assert!(de < 1e-12, "{tag} {label}: E {e:.15e} vs ref {eref:.15e}");
            assert!(dg < 1e-11, "{tag} {label}: gradient max|diff| {dg:.3e}");
        }

        let e3 = e1 - e0;
        let e3_ref = c.on.0 - c.off.0;
        let de3 = (e3 - e3_ref).abs();
        w_e3 = w_e3.max(de3);
        assert!(
            de3 < 1e-14 + 1e-9 * e3_ref.abs(),
            "{tag}: three-body E {e3:.6e} vs ref {e3_ref:.6e}"
        );
        let dg3 = max_abs_diff(&sub(&g1, &g0), &sub(&c.on.1, &c.off.1));
        w_g3 = w_g3.max(dg3);
        assert!(
            dg3 < 1e-12,
            "{tag}: three-body gradient max|diff| {dg3:.3e}"
        );
        if e3_ref.abs() > 1e-12 {
            n_nonzero += 1;
        }
    }
    // Reachability: the three-body comparison must have been made on systems
    // where the term is actually there (7 of 9 systems have >= 3 atoms).
    assert_eq!(n_nonzero, 21, "three-body term nonzero on 7 systems x 3");
    println!("MEASURED worst |diff|: E {w_e:.2e}, grad {w_g:.2e}, E3 {w_e3:.2e}, grad3 {w_g3:.2e}");
}

/// Exactness anchor: the three-body term is exactly linear in s9 and vanishes
/// for fewer than three atoms.
///
/// `s9 = 0` skips the three-body sum, so the two-body result is untouched; a
/// dimer has no triples, so `s9` must not change its energy or gradient at
/// all (bit level).
#[test]
fn three_body_term_is_linear_in_s9_and_absent_for_dimers() {
    for c in cases() {
        let e = |s9| d3bj_energy(&c.numbers, &c.coords, &with_s9(c.params, s9)).unwrap();
        let (e0, e1, e2) = (e(0.0), e(1.0), e(2.0));
        if c.numbers.len() < 3 {
            assert_eq!(e0.to_bits(), e1.to_bits(), "{}: dimer energy", c.system);
            let g = |s9| d3bj_gradient(&c.numbers, &c.coords, &with_s9(c.params, s9)).unwrap();
            assert_eq!(g(0.0), g(1.0), "{}: dimer gradient", c.system);
        } else {
            let lin = ((e2 - e0) - 2.0 * (e1 - e0)).abs();
            assert!(
                lin < 1e-15,
                "{}: E(s9) not linear, residual {lin:e}",
                c.system
            );
        }
    }
}
