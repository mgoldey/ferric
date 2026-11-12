//! Shared set-up of the CASSCF integration tests.

#![allow(dead_code)]

use ferric_ci::casscf::CasScfProblem;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
use serde_json::Value;
use std::path::PathBuf;

pub const EPS: f64 = f64::EPSILON;

pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Molecule from xyz text (charge 0, singlet: the RHF start), exact-J/K RHF.
pub fn setup_text(xyz: &str, basis_name: &str) -> (Molecule, PreparedBasis, ScfResult) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        max_iter: 300,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{basis_name}: RHF failed: {e:?}"));
    assert!(rhf.converged, "{basis_name}: RHF did not converge");
    assert!(rhf.df_jk.is_none(), "RHF must use exact J/K");
    (mol, prep, rhf)
}

pub fn molecule_text(file: &str) -> String {
    let path = workspace_root().join("testdata/molecules").join(file);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The PySCF reference written by `scripts/gen_casscf_refs.py`.
pub fn pyscf_reference(name: &str) -> Value {
    let path = workspace_root()
        .join("testdata/reference")
        .join(format!("casscf_{name}_pyscf.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

pub fn f64s(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("missing array {key}"))
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

pub fn num(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing number {key}"))
}

pub fn maxabs<'a>(it: impl IntoIterator<Item = &'a f64>) -> f64 {
    it.into_iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// Lowest eigenvalue of the CI-relaxed orbital Hessian by central
/// differences of the analytic gradient.
pub fn hessian_min_eig(problem: &CasScfProblem, c: &Array2<f64>, h: f64) -> f64 {
    let n = problem.pairs().len();
    let mut hes = Array2::<f64>::zeros((n, n));
    for k in 0..n {
        let mut kv = vec![0.0; n];
        kv[k] = h;
        let gp = problem
            .evaluate(&problem.rotate(c, &kv).unwrap())
            .unwrap()
            .gradient;
        kv[k] = -h;
        let gm = problem
            .evaluate(&problem.rotate(c, &kv).unwrap())
            .unwrap()
            .gradient;
        for r in 0..n {
            hes[(r, k)] = (gp[r] - gm[r]) / (2.0 * h);
        }
    }
    let sym = 0.5 * (&hes + &hes.t());
    let (ev, _) = sym.eigh(UPLO::Upper).unwrap();
    ev[0]
}
