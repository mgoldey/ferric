//! Shared harness of `ecp_df_hessian.rs` (per-commit) and
//! `validation_ecp_df_hessian.rs` (validation tier): ECP + density-fitted
//! analytic Hessians against central FD of ferric's analytic gradient.
#![allow(dead_code)]

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{rhf_gradient, rohf_gradient, uhf_gradient};
use ferric_scf::hessian::{rhf_hessian_parts, rohf_hessian_parts, uhf_hessian_parts};
use ferric_scf::hessian_ecp::ecp_skeleton_hessian;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ndarray::Array2;

/// Well-conditioned fitting basis on iodine systems (RI-K metric cond < 1e8;
/// measured through the probe: HI, HOI, CH3I all pass).
pub const AUX_OK: &str = "def2-svp-rifit";
/// Ordinary J/K fit. On iodine systems its exchange metric has cond 4.9e8
/// (HI) to 1.4e9 (CH3I), above `RIK_COND_WARN` = 1e8, where the code states
/// accuracy is no longer established; the probe refuses it for `Auto`.
pub const AUX_HIGH_COND: &str = "def2-universal-jkfit";

/// Analytic vs FD of the analytic gradient, Ha/Bohr².
///
/// Derived from both sides. Physics floor: measured 1.0e-7..3.6e-7 (central
/// FD, step 1e-3, displaced SCFs converged to 1e-12). Defect size: the
/// smallest effect these tests must see is the fitting error of the Hessian
/// itself, 1.0e-5 (HI) to 7.6e-5 (CH3I), and any dropped ECP term is 1e-2 or
/// more. 1e-6 is 3x above the floor and 10x below the smallest defect.
pub const TOL_FD: f64 = 1e-6;
/// Analytic vs PySCF's analytic Hessian. Measured 3.7e-8 (HI) and 2.8e-7
/// (CH3I); PySCF's own analytic-vs-FD gap is 1.1e-7 / 4.4e-7 (its ECP
/// second-derivative roughness), so nothing tighter is meaningful; the
/// smallest defect to catch is the fitting signal 1.0e-5. 1e-6 sits 3.6x above
/// the floor and 10x below the defect.
pub const TOL_PYSCF: f64 = 1e-6;
pub const MUST_MISS: f64 = 10.0;
pub const H_FD: f64 = 1e-3;

pub const HOI: &str = "3\nHOI tilted\nH 0.30 -0.40 0.20\nO 0.00 0.10 0.95\nI 0.45 0.20 2.80\n";
pub const HIH: &str = "3\nHIH tilted\nH 0.30 -0.40 0.20\nI 0.00 0.00 1.61\nH -1.10 0.70 2.90\n";

pub fn has_deriv2() -> bool {
    ferric_integrals::engine::libint_max_deriv_order() >= 2
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reference {
    Rhf,
    Uhf,
    Rohf,
}

impl Reference {
    pub fn open(self) -> bool {
        self != Reference::Rhf
    }
}

#[derive(Clone, Copy)]
pub enum Fit {
    Exact,
    J(&'static str),
    K(&'static str),
    Jk(&'static str),
}

pub fn config(fit: Fit) -> RhfConfig {
    let aux = |a: &str| Some(a.to_string());
    let (j, k) = match fit {
        Fit::Exact => (None, None),
        Fit::J(a) => (aux(a), None),
        Fit::K(a) => (None, aux(a)),
        Fit::Jk(a) => (aux(a), aux(a)),
    };
    RhfConfig {
        max_iter: 300,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some(j.unwrap_or_default()),
        df_k_aux: Some(k.unwrap_or_default()),
        ..Default::default()
    }
}

pub fn ecp_mol(xyz: &str, mult: usize, bs: &BasisSet) -> Molecule {
    let mut mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    mol.apply_ecp(bs);
    mol
}

pub struct Solved {
    pub mol: Molecule,
    pub prep: PreparedBasis,
    pub bounds: SchwarzBounds,
    pub scf: ScfResult,
}

pub fn solve(mol: &Molecule, bs: &BasisSet, r: Reference, fit: Fit) -> Solved {
    let prep = PreparedBasis::new(mol, bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = config(fit);
    let scf = match r {
        Reference::Rhf => solve_rhf(&ctx, mol, &prep, op, &bounds, &cfg),
        Reference::Uhf => solve_uhf(&ctx, mol, &prep, &bounds, &cfg),
        Reference::Rohf => solve_rohf(&ctx, mol, &prep, op, &bounds, &cfg),
    }
    .unwrap();
    assert!(scf.converged);
    Solved {
        mol: mol.clone(),
        prep,
        bounds,
        scf,
    }
}

pub fn density(s: &Solved, r: Reference) -> Array2<f64> {
    if r.open() {
        &s.scf.density_alpha + s.scf.density_beta.as_ref().unwrap()
    } else {
        s.scf.density_r().clone()
    }
}

pub fn hessian(s: &Solved, r: Reference, fit: Fit) -> Array2<f64> {
    let (ctx, op, cfg) = (ParallelContext::default(), Operator::coulomb(), config(fit));
    let a = (&ctx, &s.mol, &s.prep, op, &s.bounds, &s.scf, &cfg);
    match r {
        Reference::Rhf => rhf_hessian_parts(a.0, a.1, a.2, a.3, a.4, a.5, a.6),
        Reference::Uhf => uhf_hessian_parts(a.0, a.1, a.2, a.3, a.4, a.5, a.6),
        Reference::Rohf => rohf_hessian_parts(a.0, a.1, a.2, a.3, a.4, a.5, a.6),
    }
    .unwrap()
    .total()
}

pub fn displaced(mol: &Molecule, atom: usize, coord: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match coord {
        0 => m.atoms[atom].x += h,
        1 => m.atoms[atom].y += h,
        _ => m.atoms[atom].zpos += h,
    }
    m
}

pub fn gradient_of(t: &Solved, r: Reference) -> Array2<f64> {
    let op = Operator::coulomb();
    match r {
        Reference::Rhf => rhf_gradient(&t.mol, &t.prep, op, &t.bounds, &t.scf, None),
        Reference::Uhf => uhf_gradient(&t.mol, &t.prep, op, &t.bounds, &t.scf, None),
        Reference::Rohf => rohf_gradient(&t.mol, &t.prep, op, &t.bounds, &t.scf, None),
    }
    .unwrap()
}

/// Columns `cols` of the central FD of the analytic gradient (others zero).
pub fn fd_columns(
    s: &Solved,
    bs: &BasisSet,
    r: Reference,
    fit: Fit,
    cols: &[usize],
) -> Array2<f64> {
    let n3 = 3 * s.mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    for &b in cols {
        let g = |h: f64| {
            let t = solve(&displaced(&s.mol, b / 3, b % 3, h), bs, r, fit);
            assert!(
                (t.scf.energy - s.scf.energy).abs() < 1e-3,
                "displaced SCF changed state"
            );
            gradient_of(&t, r)
        };
        let (gp, gm) = (g(H_FD), g(-H_FD));
        for a in 0..n3 {
            out[(a, b)] = (gp[(a / 3, a % 3)] - gm[(a / 3, a % 3)]) / (2.0 * H_FD);
        }
    }
    out
}

pub fn max_diff_cols(a: &Array2<f64>, b: &Array2<f64>, cols: &[usize]) -> f64 {
    let mut m = 0.0f64;
    for &c in cols {
        for r in 0..a.nrows() {
            assert!(a[(r, c)].is_finite() && b[(r, c)].is_finite(), "non-finite");
            m = m.max((a[(r, c)] - b[(r, c)]).abs());
        }
    }
    m
}

pub fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

/// Analytic vs FD of the analytic gradient (ECP + fitted J/K), with the
/// ECP-term negative controls.
pub fn check_against_fd(
    label: &str,
    xyz: &str,
    mult: usize,
    r: Reference,
    fit: Fit,
    cols: &[usize],
) {
    if !has_deriv2() {
        eprintln!("{label}: skipped, libint2 without second derivatives");
        return;
    }
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(xyz, mult, &bs);
    let s = solve(&mol, &bs, r, fit);
    let an = hessian(&s, r, fit);
    let fd = fd_columns(&s, &bs, r, fit, cols);
    let d = max_diff_cols(&an, &fd, cols);
    eprintln!("{label}: max|analytic - FD| = {d:.3e} (bar {TOL_FD:.0e})");
    assert!(d < TOL_FD, "{label}: max|analytic - FD| = {d:.3e}");

    let skel = ecp_skeleton_hessian(&mol, &bs, &density(&s, r)).unwrap();
    let miss_skel = max_diff_cols(&(&an - &skel), &fd, cols);
    eprintln!("  {label}: no ECP skeleton misses FD by {miss_skel:.3e}");
    assert!(miss_skel > MUST_MISS * TOL_FD, "{label}: skeleton vacuous");

    let dropped =
        ferric_scf::hessian::rij_support::with_ecp_first_order_dropped(|| hessian(&s, r, fit));
    let miss_fx = max_diff_cols(&dropped, &fd, cols);
    eprintln!("  {label}: no dV_ECP/dx in F^x misses FD by {miss_fx:.3e}");
    assert!(miss_fx > MUST_MISS * TOL_FD, "{label}: F^x control vacuous");
}
