//! The two-body D3(BJ) analytic Hessian against central differences of the
//! simple-dftd3 analytic gradient (the `dftd3` Python package, an independent
//! Fortran implementation), B3LYP parameters, s9 = 0.
//!
//! `testdata/d3bj_hessian_reference.json` is written by
//! `scripts/gen_d3_hessian_reference.py`: per system the FD Hessian at step
//! `h` and `h/2`. The bar is derived from the data: `H_fd(h/2)` misses the
//! exact Hessian by `|H_fd(h) − H_fd(h/2)|/3` to leading order, plus the
//! round-off of a gradient difference (`1e-14/(h/2)`, the dftd3 gradient's
//! last digits). The bar is twice their sum.
//!
//! This sees what `hessian_vs_fd.rs` cannot: a construction error shared by
//! ferric's gradient and Hessian (both use the same tables and CN).

use ferric_d3::{d3bj_hessian, D3Params};
use serde_json::Value;

fn matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect()
}

#[test]
fn analytic_hessian_matches_fd_of_the_dftd3_gradient_within_the_derived_bar() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/d3bj_hessian_reference.json"
    );
    let root: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let p = &root["params"];
    let params = D3Params {
        s6: p["s6"].as_f64().unwrap(),
        s8: p["s8"].as_f64().unwrap(),
        a1: p["a1"].as_f64().unwrap(),
        a2: p["a2"].as_f64().unwrap(),
        s9: 0.0,
    };
    let h_step = root["step_bohr"].as_f64().unwrap();
    let systems = root["systems"].as_object().unwrap();
    assert_eq!(systems.len(), 4);
    for (name, s) in systems {
        let numbers: Vec<u8> = s["numbers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|z| z.as_u64().unwrap() as u8)
            .collect();
        let coords: Vec<[f64; 3]> = s["coords_bohr"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                let r = r.as_array().unwrap();
                [0, 1, 2].map(|k| r[k].as_f64().unwrap())
            })
            .collect();
        let (fd1, fd2) = (matrix(&s["hessian_fd_h"]), matrix(&s["hessian_fd_h2"]));
        let h = d3bj_hessian(&numbers, &coords, &params).unwrap();
        let mut delta = 0.0f64;
        let mut miss = 0.0f64;
        for a in 0..h.nrows() {
            for b in 0..h.ncols() {
                delta = delta.max((fd1[a][b] - fd2[a][b]).abs());
                miss = miss.max((h[(a, b)] - fd2[a][b]).abs());
            }
        }
        let bar = 2.0 * (delta / 3.0 + 1e-14 / (0.5 * h_step));
        let hmax = h.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        eprintln!(
            "{name}: max|H| {hmax:.3e}; |H - FD_dftd3(h/2)| {miss:.3e}, bar {bar:.3e} \
             (truncation {:.3e})",
            delta / 3.0
        );
        assert!(miss < bar, "{name}: {miss:.3e} >= {bar:.3e}");
    }
}
