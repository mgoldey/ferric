//! Shared harness of the analytic UKS Hessian tests (`uks_xc_pieces.rs`,
//! `uks_moving_grid_hessian.rs`): open-shell radicals (unequal α/β densities),
//! production UKS SCFs, and finite differences of the production UKS gradient
//! and of the polarized `V_xc`.
#![allow(dead_code)]

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_fixed_grid::{FixedGridKs, XcGeometry};
use ferric_scf::ks_gradient::{ks_gradient_uks_with_exchange_on_grids, KsGradGrids};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::{Array2, Axis};

/// NH2 doublet, distorted off C2v (Angstrom).
pub const NH2: &str = "3\nNH2 doublet\n\
N      0.000000     0.010000     0.140000\n\
H      0.020000     0.800000    -0.500000\n\
H     -0.030000    -0.780000    -0.470000\n";

/// CH3 doublet, non-planar and distorted (Angstrom).
pub const CH3: &str = "4\nCH3 doublet\n\
C      0.010000    -0.020000     0.060000\n\
H      1.080000     0.010000    -0.020000\n\
H     -0.530000     0.940000     0.030000\n\
H     -0.550000    -0.930000    -0.060000\n";

/// O2 triplet, slightly off-axis (Angstrom).
pub const O2: &str = "2\nO2 triplet\n\
O      0.000000     0.000000     0.000000\n\
O      0.010000     0.020000     1.210000\n";

/// Water, distorted (closed shell; the singlet anchors).
pub const WATER: &str = "3\nH2O distorted off C2v\n\
O      0.010000    -0.020000     0.120000\n\
H      0.030000     0.770000    -0.460000\n\
H     -0.020000    -0.740000    -0.490000\n";

pub fn nh2() -> Molecule {
    Molecule::parse_xyz(NH2, 0, 2).unwrap()
}
pub fn ch3() -> Molecule {
    Molecule::parse_xyz(CH3, 0, 2).unwrap()
}
pub fn o2() -> Molecule {
    Molecule::parse_xyz(O2, 0, 3).unwrap()
}
pub fn water() -> Molecule {
    Molecule::parse_xyz(WATER, 0, 1).unwrap()
}

pub fn grid_cfg() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 50,
        n_angular: 110,
        prune: None,
        weight_cut: None,
    }
}

pub fn has_deriv2() -> bool {
    let order = ferric_integrals::engine::libint_max_deriv_order();
    if order < 2 {
        eprintln!("SKIP: libint2 generated with derivative order {order}; no analytic Hessian");
    }
    order >= 2
}

/// UKS config: functional `xc`, exact J and K, the test grid.
pub fn config(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        dft_grid: Some(grid_cfg()),
        max_iter: 300,
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
    pub grid: AtomicGridConfig,
}

/// Production UKS SCF.
pub fn scf(mol: &Molecule, basis_name: &str, cfg: &RhfConfig) -> Pt {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let res = solve_uhf(&ParallelContext::default(), mol, &prep, &bounds, cfg).unwrap();
    assert!(res.converged, "{basis_name}: UKS SCF did not converge");
    Pt {
        mol: mol.clone(),
        bs,
        prep,
        bounds,
        res,
        grid: cfg
            .dft_grid
            .clone()
            .unwrap_or_else(AtomicGridConfig::production),
    }
}

pub fn spin_densities(p: &Pt) -> (Array2<f64>, Array2<f64>) {
    (
        p.res.density_alpha.clone(),
        p.res.density_beta.clone().expect("UKS beta density"),
    )
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

/// `max_α |Σ_B H[(B,α), :]|`.
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

/// The moving-grid polarized XC gradient at FROZEN spin densities from
/// `ferric_dft` (an independent code: riding `weight1`), `(natoms, 3)`.
pub fn dft_uks_gradient(
    mol: &Molecule,
    bs: &basis::BasisSet,
    d: (&Array2<f64>, &Array2<f64>),
    xc: &str,
    grid: &AtomicGridConfig,
) -> Array2<f64> {
    let prep = PreparedBasis::new(mol, bs).unwrap();
    ferric_dft::gradient::xc_gradient_uks_from_density(
        mol,
        bs,
        d.0,
        d.1,
        xc,
        grid,
        prep.shell_to_atom(),
        prep.shell_offsets(),
        prep.shell_dims(),
    )
    .unwrap()
}

/// Polarized `[V^α, V^β]` at frozen spin densities on the grid built at `mol`.
pub fn vxc_rebuilt(
    mol: &Molecule,
    bs: &basis::BasisSet,
    d: (&Array2<f64>, &Array2<f64>),
    xc: &str,
    grid: &AtomicGridConfig,
) -> [Array2<f64>; 2] {
    let prep = PreparedBasis::new(mol, bs).unwrap();
    let grid = FixedGridKs::new(mol, xc, grid).unwrap();
    grid.polarized()
        .unwrap()
        .energy_vxc(&XcGeometry::new(mol, &prep).unwrap(), [d.0, d.1])
        .unwrap()
        .1
}

/// Step of every difference, Bohr.
pub const H_FD: f64 = 5e-4;

/// Frozen-density references: the moving skeleton (FD of the `ferric_dft`
/// UKS XC gradient) and every `F^{s,x}` (FD of the rebuilt polarized `V_xc`).
pub struct FrozenRef {
    pub skel: Array2<f64>,
    pub fock: [Vec<Array2<f64>>; 2],
}

pub fn frozen_reference(p: &Pt, xc: &str) -> FrozenRef {
    let (da, db) = spin_densities(p);
    let n3 = 3 * p.mol.atoms.len();
    let mut skel = Array2::<f64>::zeros((n3, n3));
    let mut fock: [Vec<Array2<f64>>; 2] = [Vec::new(), Vec::new()];
    for b in 0..n3 {
        let col = rich(H_FD, |s| {
            flat(&dft_uks_gradient(
                &displaced(&p.mol, b, s),
                &p.bs,
                (&da, &db),
                xc,
                &p.grid,
            ))
        });
        skel.column_mut(b).assign(&col.column(0));
        for (sp, f) in fock.iter_mut().enumerate() {
            f.push(rich(H_FD, |s| {
                vxc_rebuilt(&displaced(&p.mol, b, s), &p.bs, (&da, &db), xc, &p.grid)[sp].clone()
            }));
        }
    }
    FrozenRef { skel, fock }
}

/// SCF convergence of every SCF behind a gradient difference (see the RKS
/// harness `ks_moving_grid_common::fd_reference_config`): a displaced SCF that
/// stops one iteration later than its neighbours breaks the cancellation of
/// the shared residual.
pub fn fd_reference_config(cfg: &RhfConfig) -> RhfConfig {
    RhfConfig {
        density_conv: cfg.density_conv.min(1e-13),
        energy_conv: cfg.energy_conv.min(1e-14),
        ..cfg.clone()
    }
}

/// Largest DIIS `err_max` an SCF behind a gradient difference may stop at
/// (the RKS harness's bar: `4/(3h)` amplification, ~10x under the Hessian bar).
pub const FD_SCF_ERR_MAX: f64 = 1e-13;

/// The production UKS gradient of `q`, flattened.
pub fn production_gradient(q: &Pt, cfg: &RhfConfig, xc: &str) -> Array2<f64> {
    flat(
        &ks_gradient_uks_with_exchange_on_grids(
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
}

/// Central difference of the production UKS gradient (production UKS SCF at
/// every displaced geometry). Every displaced SCF must stop with DIIS
/// `err_max` below [`FD_SCF_ERR_MAX`]. Returns the FD Hessian and the worst
/// `err_max`.
pub fn fd_of_production_gradient(
    p: &Pt,
    basis_name: &str,
    cfg: &RhfConfig,
    xc: &str,
) -> (Array2<f64>, f64) {
    let n3 = 3 * p.mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    let mut worst = 0.0f64;
    for b in 0..n3 {
        let col = rich(H_FD, |s| {
            let q = scf(&displaced(&p.mol, b, s), basis_name, cfg);
            let err_max = q
                .res
                .convergence
                .as_ref()
                .expect("a converged SCF reports its convergence signals")
                .err_max;
            worst = worst.max(err_max);
            assert!(
                err_max < FD_SCF_ERR_MAX,
                "{basis_name} coordinate {b} step {s:+e}: SCF stopped at err_max {err_max:.2e}"
            );
            production_gradient(&q, cfg, xc)
        });
        out.column_mut(b).assign(&col.column(0));
    }
    (out, worst)
}
