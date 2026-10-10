//! Validation: analytic RHF/UHF Hessian with external point charges
//! (QM–QM block, MM charges fixed) against PySCF.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release \
//!     --test validation_qmmm_hessian -- --ignored --nocapture
//! ```
//!
//! References: `testdata/reference/validation/qmmm_hessian/*.json`, generated
//! by `scripts/validation/gen_qmmm_hessian.py`. PySCF has no QM/MM Hessian,
//! so the reference is the central FD (step 1e-3 Bohr) of PySCF's analytic
//! `qmmm.mm_charge(...).nuc_grad_method()` gradient, QM atoms only. The
//! negative control is the same FD without the charges: the embedded analytic
//! Hessian must miss it by at least `MISS_FACTOR` times the bar.
//!
//! # MUTATION (each must fail a named test; see `src/hessian.rs`)
//! * `charge_nuclear_hessian` dropped from `nuclear_term`;
//! * `&[]` for `extra` in the nuclear `contract_1e_deriv2` or in
//!   `deriv1_1e_matrices` (∂V_ext/∂x missing from F^x);
//! * the external-centre blocks kept in `scatter_unique_pairs`;
//! * `n_charges` sized with `prep.atoms().len()` only;
//! * a flipped charge sign;
//! * (smeared rows, #358) `smeared_skeleton_hessian` or
//!   `smeared_first_derivative_matrices` dropped, the smeared term dropped from
//!   `charge_nuclear_hessian`, a single-shell stencil displacement.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::elements;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::hessian::{rhf_hessian, uhf_hessian};
use ferric_scf::qmmm::{QmSelection, QmmmAtom, QmmmSystem};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/qmmm_hessian";

/// Analytic vs PySCF-gradient FD, Ha/Bohr². Measured max 8.0e-7 (methanol);
/// the bar is ~10x that
/// (the reference FD's own floor is ~1e-6).
const TOL_HESSIAN: f64 = 1e-5;
/// The gas-phase FD Hessian must miss the embedded analytic one by this
/// multiple of `TOL_HESSIAN` (smallest gap, methanol: 7.7e-3 = 770x).
const MISS_FACTOR: f64 = 10.0;
const TOL_ENUC: f64 = 1e-9;

fn workspace_root() -> PathBuf {
    let ok = |p: &Path| p.join("Cargo.toml").is_file() && p.join("testdata").is_dir();
    let mut here = std::env::current_dir().ok();
    while let Some(p) = here {
        if ok(&p) {
            return p;
        }
        here = p.parent().map(Path::to_path_buf);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn reference(stem: &str) -> Value {
    let path = workspace_root().join(ROW_DIR).join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_qmmm_hessian.py — a missing reference is a failure",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap()
}

fn xyz3(v: &Value) -> [f64; 3] {
    let a = v.as_array().unwrap();
    [
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    ]
}

fn matrix(r: &Value, key: &str) -> Array2<f64> {
    let rows = r[key].as_array().unwrap_or_else(|| panic!("{key} missing"));
    let n = rows.len();
    let mut m = Array2::zeros((n, n));
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.as_array().unwrap().iter().enumerate() {
            m[(i, j)] = v.as_f64().unwrap();
        }
    }
    m
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| {
        // f64::max would silently ignore a NaN entry.
        assert!(x.is_finite() && y.is_finite(), "non-finite Hessian entry");
        m.max((x - y).abs())
    })
}

fn system(r: &Value) -> QmmmSystem {
    let mut atoms: Vec<QmmmAtom> = r["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            let sym = a["symbol"].as_str().unwrap();
            let [x, y, z] = xyz3(&a["xyz_bohr"]);
            QmmmAtom::new(sym, elements::symbol_to_z(sym).unwrap(), x, y, z, 99.0)
        })
        .collect();
    let nqm = atoms.len();
    for c in r["mm_charges"].as_array().unwrap() {
        let [x, y, z] = xyz3(&c["xyz_bohr"]);
        let q = c["q"].as_f64().unwrap();
        atoms.push(match c["width"].as_f64() {
            // Gaussian-smeared site: width in Bohr == PySCF mm_charge radii (unit="Bohr").
            Some(w) => QmmmAtom::new_smeared("X", 0, x, y, z, q, w),
            None => QmmmAtom::new("X", 0, x, y, z, q),
        });
    }
    QmmmSystem::new(
        &atoms,
        QmSelection::Indices((0..nqm).collect()),
        r["charge"].as_i64().unwrap() as i32,
        r["multiplicity"].as_u64().unwrap() as usize,
    )
    .unwrap()
}

fn check(stem: &str, uhf: bool) {
    let r = reference(stem);
    let sys = system(&r);
    let mol = sys.to_qm_molecule();
    let enuc = r["nuclear_repulsion"].as_f64().unwrap();
    assert!(
        (mol.nuclear_repulsion() - enuc).abs() < TOL_ENUC,
        "{stem}: geometry/unit mismatch"
    );
    let ext = sys.to_external_potential().expect("MM region");
    let bs = basis::bundled(r["basis"].as_str().unwrap()).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        external_potential: Some(ext),
        energy_conv: 1e-12,
        density_conv: 1e-10,
        max_iter: 300,
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let (res, h) = if uhf {
        let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
        let h = uhf_hessian(&ctx, &mol, &prep, op, &bounds, &res, &cfg).unwrap();
        (res, h)
    } else {
        let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        let h = rhf_hessian(&ctx, &mol, &prep, op, &bounds, &res, &cfg).unwrap();
        (res, h)
    };
    assert!(res.converged);
    let e_ref = r["energy"].as_f64().unwrap();
    assert!(
        (res.energy - e_ref).abs() < 1e-8,
        "{stem}: E {} vs {e_ref}",
        res.energy
    );

    let d = max_diff(&h, &matrix(&r, "hessian_fd"));
    let miss = max_diff(&h, &matrix(&r, "hessian_gas_fd"));
    eprintln!("{stem}: max|analytic - PySCF FD| = {d:.3e} (bar {TOL_HESSIAN:.0e}); gas-phase control miss {miss:.3e}");
    assert!(
        d < TOL_HESSIAN,
        "{stem}: max|analytic - PySCF FD| = {d:.3e}"
    );
    assert!(
        miss > MISS_FACTOR * TOL_HESSIAN,
        "{stem}: gas-phase control misses by only {miss:.3e}: the charges may not be applied"
    );
}

#[test]
#[ignore = "validation: analytic QM/MM Hessian"]
fn h2o_q10_rhf_ccpvdz() {
    check("h2o_q10_cc-pvdz", false);
}

#[test]
#[ignore = "validation: analytic QM/MM Hessian"]
fn ch3oh_q20_rhf_def2svp() {
    check("ch3oh_q20_def2-svp", false);
}

#[test]
#[ignore = "validation: analytic QM/MM Hessian"]
fn h2o_q10_cation_uhf_ccpvdz() {
    check("h2o_q10_cation_cc-pvdz", true);
}

#[test]
#[ignore = "validation: analytic QM/MM Hessian"]
fn h2o_q10_smeared_rhf_ccpvdz() {
    check("h2o_q10_smeared_cc-pvdz", false);
}

#[test]
#[ignore = "validation: analytic QM/MM Hessian"]
fn h2o_q10_cation_smeared_uhf_ccpvdz() {
    check("h2o_q10_cation_smeared_cc-pvdz", true);
}
