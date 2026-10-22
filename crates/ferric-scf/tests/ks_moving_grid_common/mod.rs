//! Shared harness of `ks_moving_grid_hessian.rs` (per-commit) and
//! `validation_ks_moving_grid_hessian.rs` (validation tier): the moving-grid
//! analytic KS Hessian and its pieces against finite differences of the
//! production (moving-grid) gradient and `V_xc`.
#![allow(dead_code)]

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::hessian::{rks_fixed_grid_hessian, rks_moving_grid_hessian};
use ferric_scf::ks_fixed_grid::{FixedGridKs, XcGeometry, XcMutants};
use ferric_scf::ks_gradient::{ks_gradient_closed_with_exchange_on_grids, KsGradGrids};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array2, Axis};

pub const WATER: &str = "3\nH2O distorted off C2v\n\
O      0.010000    -0.020000     0.120000\n\
H      0.030000     0.770000    -0.460000\n\
H     -0.020000    -0.740000    -0.490000\n";

pub fn grid_cfg() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 50,
        n_angular: 110,
        prune: None,
    }
}

pub fn has_deriv2() -> bool {
    let order = ferric_integrals::engine::libint_max_deriv_order();
    if order < 2 {
        eprintln!("SKIP: libint2 generated with derivative order {order}; no analytic Hessian");
    }
    order >= 2
}

pub fn water() -> Molecule {
    Molecule::parse_xyz(WATER, 0, 1).unwrap()
}

/// Production SCF / Hessian config: functional `xc`, exact J and K, the
/// test grid.
pub fn config(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        dft_grid: Some(grid_cfg()),
        max_iter: 200,
        energy_conv: 1e-13,
        density_conv: 1e-11,
        ..Default::default()
    }
}

pub struct Pt {
    pub mol: Molecule,
    pub bs: basis::BasisSet,
    pub prep: PreparedBasis,
    pub bounds: SchwarzBounds,
    pub res: ScfResult,
    /// The XC grid the SCF ran on.
    pub grid: AtomicGridConfig,
}

/// Production SCF (grid built on the atoms of `mol`).
pub fn scf(mol: &Molecule, basis_name: &str, cfg: &RhfConfig) -> Pt {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let res = solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, cfg).unwrap();
    assert!(res.converged, "{basis_name}: SCF did not converge");
    Pt {
        mol: mol.clone(),
        bs,
        prep,
        bounds,
        res,
        grid: cfg.dft_grid.clone().unwrap_or_default(),
    }
}

pub fn displaced(mol: &Molecule, coord: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match coord % 3 {
        0 => m.atoms[coord / 3].x += h,
        1 => m.atoms[coord / 3].y += h,
        _ => m.atoms[coord / 3].zpos += h,
    }
    m
}

/// Richardson-extrapolated central difference from `h` and `h/2`.
pub fn rich(h: f64, mut f: impl FnMut(f64) -> Array2<f64>) -> Array2<f64> {
    let mut d = |h: f64| (f(h) - f(-h)) / (2.0 * h);
    let coarse = d(h);
    let fine = d(0.5 * h);
    (4.0 * fine - coarse) / 3.0
}

pub fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

pub fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    max_abs(&(a - b))
}

pub fn flat(g: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_vec((g.len(), 1), g.iter().copied().collect()).unwrap()
}

/// `max_α |Σ_B H[(B,α), :]|`: the translation sum rule residual of a Hessian.
pub fn sum_rule(h: &Array2<f64>) -> f64 {
    (0..3)
        .map(|a| {
            let rows: Vec<usize> = (a..h.nrows()).step_by(3).collect();
            max_abs(
                &h.select(Axis(0), &rows)
                    .sum_axis(Axis(0))
                    .insert_axis(Axis(0)),
            )
        })
        .fold(0.0, f64::max)
}

/// The moving-grid XC gradient at a FROZEN density from `ferric_dft`
/// (riding `weight1` grid response), `(natoms, 3)`.
pub fn dft_moving_gradient(
    mol: &Molecule,
    bs: &basis::BasisSet,
    d: &Array2<f64>,
    xc: &str,
    grid: &AtomicGridConfig,
) -> Array2<f64> {
    let prep = PreparedBasis::new(mol, bs).unwrap();
    let args = (
        prep.shell_to_atom(),
        prep.shell_offsets(),
        prep.shell_dims(),
    );
    let g = if xc == "LDA" {
        ferric_dft::gradient::xc_gradient_closed_lda_from_density(
            mol, bs, d, xc, grid, args.0, args.1, args.2,
        )
    } else {
        ferric_dft::gradient::xc_gradient_closed_gga_from_density(
            mol, bs, d, xc, grid, args.0, args.1, args.2,
        )
    };
    g.unwrap()
}

/// `V_xc` at the frozen density `d` on the grid built at `mol`.
pub fn vxc_rebuilt(
    mol: &Molecule,
    bs: &basis::BasisSet,
    d: &Array2<f64>,
    xc: &str,
    grid: &AtomicGridConfig,
) -> Array2<f64> {
    let prep = PreparedBasis::new(mol, bs).unwrap();
    let grid = FixedGridKs::new(mol, xc, grid).unwrap();
    grid.energy_vxc(&XcGeometry::new(mol, &prep).unwrap(), d)
        .unwrap()
        .1
}

/// Frozen-density finite-difference references: the skeleton (FD of the
/// `ferric_dft` moving gradient) and every `F^x` (FD of rebuilt `V_xc`).
pub struct FrozenRef {
    pub skel: Array2<f64>,
    pub fock: Vec<Array2<f64>>,
}

/// Step of the frozen-density differences, Bohr.
pub const H_FROZEN: f64 = 5e-4;

pub fn frozen_reference(p: &Pt, xc: &str) -> FrozenRef {
    let d = p.res.density_r().clone();
    let n3 = 3 * p.mol.atoms.len();
    let mut skel = Array2::<f64>::zeros((n3, n3));
    let mut fock = Vec::with_capacity(n3);
    for b in 0..n3 {
        let col = rich(H_FROZEN, |s| {
            flat(&dft_moving_gradient(
                &displaced(&p.mol, b, s),
                &p.bs,
                &d,
                xc,
                &p.grid,
            ))
        });
        skel.column_mut(b).assign(&col.column(0));
        fock.push(rich(H_FROZEN, |s| {
            vxc_rebuilt(&displaced(&p.mol, b, s), &p.bs, &d, xc, &p.grid)
        }));
    }
    FrozenRef { skel, fock }
}

/// Worst skeleton and `F^x` miss of the moving pieces under `m`.
pub fn frozen_misses(p: &Pt, xc: &str, r: &FrozenRef, m: &XcMutants) -> (f64, f64) {
    let grid = FixedGridKs::new(&p.mol, xc, &p.grid).unwrap();
    let geo = XcGeometry::new(&p.mol, &p.prep).unwrap();
    let d = p.res.density_r();
    let skel = grid.skeleton_hessian_moving(&geo, d, m).unwrap();
    let fock = grid.first_order_fock_moving(&geo, d, m).unwrap();
    let fmiss = fock
        .iter()
        .zip(&r.fock)
        .map(|(a, b)| max_abs_diff(a, b))
        .fold(0.0, f64::max);
    (max_abs_diff(&skel, &r.skel), fmiss)
}

/// Skeleton vs FD of the moving gradient, Ha/Bohr² (measured max 4.9e-11 over
/// LDA/PBE x STO-3G/6-31G; the fixed-grid skeleton misses the same reference by
/// 2.4e-3 to 9.6e-3).
pub const TOL_SKELETON: f64 = 1e-9;
/// `F^x` vs FD of rebuilt `V_xc`, Ha/Bohr (measured max 7.3e-12).
pub const TOL_FOCK: f64 = 1e-10;
/// A mutant must miss by at least this many bars.
pub const MUTANT_FACTOR: f64 = 10.0;

/// Central difference of the production KS gradient (production SCF at every
/// displaced geometry on the same grid).
pub fn fd_of_production_gradient(
    p: &Pt,
    basis_name: &str,
    cfg: &RhfConfig,
    xc: &str,
) -> Array2<f64> {
    let n3 = 3 * p.mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    for b in 0..n3 {
        let col = rich(H_FROZEN, |s| {
            let q = scf(&displaced(&p.mol, b, s), basis_name, cfg);
            flat(
                &ks_gradient_closed_with_exchange_on_grids(
                    &q.mol,
                    &q.prep,
                    &q.bs,
                    Operator::coulomb(),
                    &q.bounds,
                    xc,
                    &q.res,
                    None,
                    None,
                    &KsGradGrids::from_config(cfg),
                )
                .unwrap(),
            )
        });
        out.column_mut(b).assign(&col.column(0));
    }
    out
}

/// Full Hessian vs FD of the production gradient (Richardson, h = 5e-4 and
/// 2.5e-4), Ha/Bohr²: measured 7.9e-11 (LDA) and 4.8e-11 (PBE) at STO-3G on the
/// 50x110 grid and 1.7e-10 (LDA, 6-31G, default 75x110 grid); the fixed-grid
/// Hessian misses the same references by 2.4e-3, 9.7e-3 and 1.1e-3.
pub const TOL_HESS_FD: f64 = 2e-9;

/// Moving and fixed-grid analytic Hessians vs FD of the production gradient
/// for the SCF of `cfg`; returns `(moving miss, fixed miss)` after asserting
/// the moving one is below `tol` and the fixed one above `10 tol`.
pub fn full_hessian_case(cfg: &RhfConfig, xc: &str, basis_name: &str, tol: f64) -> (f64, f64) {
    use ferric_scf::frequencies::{atom_masses, frequencies_from_cartesian_hessian};
    let p = scf(&water(), basis_name, cfg);
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let moving =
        rks_moving_grid_hessian(&ctx, &p.mol, &p.prep, op, &p.bounds, &p.res, cfg).unwrap();
    let fixed = rks_fixed_grid_hessian(&ctx, &p.mol, &p.prep, op, &p.bounds, &p.res, cfg).unwrap();
    let fd = fd_of_production_gradient(&p, basis_name, cfg, xc);
    let miss = max_abs_diff(&moving, &fd);
    let fixed_miss = max_abs_diff(&fixed, &fd);
    let masses = atom_masses(&p.mol).unwrap();
    let nu = |h: &Array2<f64>| {
        frequencies_from_cartesian_hessian(&p.mol, h, &masses)
            .unwrap()
            .frequencies
    };
    let fd_sym = 0.5 * (&fd + &fd.t());
    let dnu = |h: &Array2<f64>| {
        nu(h)
            .iter()
            .zip(&nu(&fd_sym))
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
    };
    eprintln!(
        "{xc} {basis_name}: moving-grid Hessian vs FD of the production gradient {miss:.3e} \
         (max |H| {:.3e}, sum rule {:.1e}, max |d nu| {:.2e} cm^-1); fixed-grid Hessian misses \
         by {fixed_miss:.3e} (max |d nu| {:.2e} cm^-1)",
        max_abs(&moving),
        sum_rule(&moving),
        dnu(&moving),
        dnu(&fixed)
    );
    assert!(miss < tol, "{xc} {basis_name}: {miss:.3e}");
    assert!(fixed_miss > MUTANT_FACTOR * tol, "{fixed_miss:.3e}");
    (miss, fixed_miss)
}
