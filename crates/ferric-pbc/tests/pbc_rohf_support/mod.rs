//! Shared setup of `pbc_rohf.rs` (per commit) and `validation_pbc_rohf.rs`
//! (weekly): every non-test item of the original file, made `pub`.
#![allow(dead_code, unused_imports)]

pub use crate::common::*;
pub use ferric_core::basis::BasisSet;
pub use ferric_core::mol::Molecule;
pub use ferric_core::FerricError;
pub use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
pub use ferric_pbc::dft::{
    gamma_rks, GammaRksConfig, GammaUksConfig, PeriodicGrid, PeriodicGridConfig, PeriodicXc,
    PeriodicXcConfig,
};
pub use ferric_pbc::ewald::madelung_constant;
pub use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::rohf::{
    gamma_rohf, gamma_roks, gamma_roks_with_xc, rohf_occupation_gaps, GammaRohfConfig,
    GammaRohfResult, GammaRoksConfig, GammaRoksResult, RohfSolver, RohfTrahConfig,
    ROKS_HYBRID_LEVEL_SHIFT, ROKS_HYBRID_MAX_ITER,
};
pub use ferric_pbc::uhf::{gamma_uhf, EwaldStart, GammaUhfConfig, GammaUhfIntegrals};
pub use ferric_scf::rhf::{RhfConfig, XcBuilder};
pub use ndarray::Array2;
pub use ndarray_linalg::{Eigh, UPLO};

pub const HCORE_OMEGA: f64 = 0.8;

/// `test_prototype.py` H_ATOM (Bohr).
pub const H_ATOM: [[f64; 3]; 1] = [[0.3, 0.2, 0.1]];

pub const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];

// --- (2) FINDINGS It. 6 PySCF pbc.scf.UHF pins (N_β = 0: ROHF == UHF).
pub const H_UHF_E: [f64; 2] = [-0.402177788224, -0.756839973159];

pub const H2T_UHF_E: [f64; 2] = [0.322229103842, -0.387095266028];

// --- (2) tri 4H s+p triplet: PySCF 2.13 pbc.scf.ROHF (AFTDF 61^3), [none,
// ewald], same value from PySCF's default guess and from the UHF density;
// run_roks_ssf_pins.py hf. The prototype's roks() reproduces the ewald value
// to 1e-12 from the UHF density; from the core guess it lands in a trapped
// state at -1.810787564982 (+1.53e-2), the ROHF analogue of Iteration 6.
pub const TRI_ROHF_EWALD_TRAPPED: f64 = -1.810787564982;

pub const TRI_ROHF_PYSCF_E: [f64; 2] = [-0.581222768976, -1.826096526949];

// --- (3) Prototype own construction (pbc_uks.roks on the SSF 75x302 D=10
// grid, dense pure-AFT, exxdiv ewald, conv 1e-12), [LDA, PBE, PBE0];
// run_roks_ssf_pins.py tri. The prototype reaches the same values from the
// UKS density and from the core guess (1e-12, none and ewald alike). That is
// TWO starts, NOT start-independence: for PBE0 none the prototype's own
// [F, D] DIIS converges from only 5 of 19 perturbed core starts, and a
// replica of ferric's unshifted loop from 2 of 19 (FINDINGS "ROKS PBE0 CI
// non-convergence (Python diagnosis) — 2026-09-25"); the shifted default of
// GammaRoksConfig::new is what makes it robust (37/37 in the replica).
// LDA/PBE were not surveyed beyond the core guess + 3 rotations. PBE0 none =
// TRI_ROKS_PBE0_NONE, so ewald - none = -a v_M N/2 holds to 1e-12 in the
// prototype.
pub const TRI_ROKS_SSF_E: [f64; 3] = [-1.694206925329, -1.731150286924, -1.776676719941];

/// The exxdiv = none stage of TRI_ROKS_SSF_E[2] (PBE0), same construction.
pub const TRI_ROKS_PBE0_NONE: f64 = -1.465458280448;

/// pbc_uks.rs TRI_SSF_E: UKS on the same grid (ROKS must lie above).
pub const TRI_UKS_SSF_E: [f64; 3] = [-1.694518592905, -1.731614505391, -1.777304384785];

pub const TRI_SSF_NPTS: usize = 70784;

/// FINDINGS It. 10 PySCF pbc.dft.ROKS (A1 (50,146) grid, a DIFFERENT grid
/// construction; asserted to the construction gap, as in pbc_uks.rs).
pub const TRI_ROKS_PYSCF_E: [f64; 3] = [-1.694361887534, -1.731308266751, -1.776801906487];

pub const GRID_CONSTRUCTION_GAP: f64 = 1.1e-3;

/// `run_grad_ro_anchor.py` H3 doublet (Bohr), cubic a = 4.5, STO-3G:
/// (nd, no) = (1, 1), the prototype's P1 anchor system.
pub const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];

pub const H3_A: f64 = 4.5;

/// Iteration 24 lowest exact orbital-Hessian eigenvalue at PySCF's tri ROHF
/// minimum (+0.1231, none and ewald alike) and at the second, trapped
/// minimum (+0.0297); the window below separates them.
pub const TRI_ROHF_LAMBDA_MIN_WINDOW: (f64, f64) = (0.08, 0.2);

pub struct Setup {
    pub cell: Cell,
    pub prep: ferric_integrals::basis_bridge::PreparedBasis,
    pub hc: PeriodicHcore,
    pub eri: DenseAftEri,
}

/// A neutral hydrogen cell with atoms at `atoms` (Bohr), lattice rows
/// `lattice` (Bohr) and spin multiplicity `mult`.
pub fn cell_mult(atoms: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(atoms);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

pub fn setup(cell: Cell, bs: BasisSet) -> Setup {
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(HCORE_OMEGA)).unwrap();
    // exxdiv is taken from the gamma_* configs by the builders, not the tensor.
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    Setup {
        cell,
        prep,
        hc,
        eri,
    }
}

pub fn ints(su: &Setup) -> GammaUhfIntegrals<'_> {
    GammaUhfIntegrals::DenseAft(&su.eri)
}

pub fn rohf(su: &Setup, exx: ExxDiv, start: EwaldStart) -> GammaRohfResult {
    let cfg = GammaRohfConfig {
        exxdiv: exx,
        ewald_start: start,
        ..Default::default()
    };
    let r = gamma_rohf(&su.cell, &su.prep, &su.hc, ints(su), &cfg).expect("gamma_rohf");
    assert!(r.scf.converged);
    assert!(r.scf.mos_beta.is_none(), "ROHF has one MO set");
    r
}

/// [`RohfSolver::SecondOrder`] with the default controls, optionally
/// reporting the lowest exact orbital-Hessian eigenvalue.
pub fn second_order(check_minimum: bool) -> RohfSolver {
    RohfSolver::SecondOrder(RohfTrahConfig {
        check_minimum,
        ..Default::default()
    })
}

/// [`rohf`] on the second-order solver (λ_min reported).
pub fn rohf_so(su: &Setup, exx: ExxDiv, start: EwaldStart) -> GammaRohfResult {
    let cfg = GammaRohfConfig {
        exxdiv: exx,
        ewald_start: start,
        solver: second_order(true),
        ..Default::default()
    };
    let r = gamma_rohf(&su.cell, &su.prep, &su.hc, ints(su), &cfg)
        .unwrap_or_else(|e| panic!("gamma_rohf second order {exx:?} {start:?}: {e}"));
    assert!(r.scf.converged);
    assert!(r.scf.mos_beta.is_none(), "ROHF has one MO set");
    let info = r.second_order.as_ref().expect("second-order diagnostics");
    assert!(info.converged, "{info:?}");
    eprintln!("  second order {exx:?} {start:?}: {info:?}");
    r
}

/// Same state: per-spin densities agree.
pub fn same_state(
    a: &ferric_scf::result::ScfResult,
    b: &ferric_scf::result::ScfResult,
    what: &str,
) {
    let da = max_abs_diff(&a.density_alpha, &b.density_alpha);
    let db = max_abs_diff(
        a.density_beta.as_ref().expect("beta density"),
        b.density_beta.as_ref().expect("beta density"),
    );
    eprintln!("  {what}: max|dD_a| {da:.2e} max|dD_b| {db:.2e}");
    assert!(da < 1e-6 && db < 1e-6, "{what}: different state");
}

/// Periodic grid of `n_rad` radial and `n_ang` angular points with neighbour
/// cutoff `d` Bohr; the other fields are the defaults of `with_size`.
pub fn ssf_grid(n_rad: usize, n_ang: usize, d: f64) -> PeriodicGridConfig {
    PeriodicGridConfig {
        neighbour_cutoff: Some(d),
        ..PeriodicGridConfig::with_size(n_rad, n_ang)
    }
}

pub fn roks(su: &Setup, xc: &str, grid: PeriodicGridConfig, exx: ExxDiv) -> GammaRoksResult {
    let cfg = GammaRoksConfig {
        grid,
        exxdiv: exx,
        ..GammaRoksConfig::new(xc)
    };
    let r = gamma_roks(&su.cell, &su.prep, &su.hc, ints(su), &cfg)
        .unwrap_or_else(|e| panic!("gamma_roks {xc}: {e}"));
    assert!(r.scf.converged);
    r
}

pub fn close(got: f64, want: f64, tol: f64, what: &str) {
    eprintln!(
        "  {what}: {got:.12} (want {want:.12}, diff {:.2e})",
        got - want
    );
    assert!(got.is_finite(), "{what}: non-finite");
    assert!(
        (got - want).abs() < tol,
        "{what}: {got:.12e} vs {want:.12e} (tol {tol:.1e})"
    );
}

pub fn s_s1(na: usize, nb: usize) -> f64 {
    let sz = 0.5 * (na as f64 - nb as f64);
    sz * (sz + 1.0)
}

/// `a = frac`, zero semilocal XC: with `frac = 1` the ROKS loop IS the ROHF.
pub struct ZeroXc {
    pub frac: f64,
}

impl XcBuilder for ZeroXc {
    fn build(&mut self, d: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        Ok((0.0, Array2::zeros(d.dim())))
    }
    fn exact_exchange_fraction(&self) -> f64 {
        self.frac
    }
    fn build_polarized(
        &mut self,
        d_a: &Array2<f64>,
        _d_b: &Array2<f64>,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
        Ok((0.0, Array2::zeros(d_a.dim()), Array2::zeros(d_a.dim())))
    }
    fn supports_polarized(&self) -> bool {
        true
    }
}

/// Closed-shell-only builder (default `supports_polarized() = false`).
pub struct ClosedOnlyXc;

impl XcBuilder for ClosedOnlyXc {
    fn build(&mut self, d: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        Ok((0.0, Array2::zeros(d.dim())))
    }
    fn exact_exchange_fraction(&self) -> f64 {
        0.0
    }
}

/// Deterministic uniform [-1, 1) stream (Knuth MMIX LCG, top 53 bits).
pub struct Lcg(pub u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        let mut g = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03);
        for _ in 0..8 {
            g.uniform();
        }
        g
    }
    pub fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// The injected core guess (symmetric `S^{-1/2}`, eigenvectors of
/// `S^{-1/2} h S^{-1/2}`), as `solve_rohf_impl` builds it.
pub fn core_mos(s: &Array2<f64>, h: &Array2<f64>) -> Array2<f64> {
    let (se, sv) = s.eigh(UPLO::Upper).expect("S eigh");
    let mut xs = sv.clone();
    for (j, e) in se.iter().enumerate() {
        xs.column_mut(j).mapv_inplace(|v| v / e.sqrt());
    }
    let x = xs.dot(&sv.t());
    let (_, cp) = x.dot(h).dot(&x).eigh(UPLO::Upper).expect("h' eigh");
    x.dot(&cp)
}

/// `C exp(K)`, `K` antisymmetric with entries uniform in `[-t, t)` from
/// `seed` (Taylor series; `‖K‖ ≤ n t` is tiny here). Keeps `Cᵀ S C = 1`.
pub fn rotated(c: &Array2<f64>, t: f64, seed: u64) -> Array2<f64> {
    let n = c.ncols();
    let mut g = Lcg::new(seed);
    let mut k = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..i {
            let v = t * g.uniform();
            k[(i, j)] = v;
            k[(j, i)] = -v;
        }
    }
    let mut r = Array2::<f64>::eye(n);
    let mut term = Array2::<f64>::eye(n);
    for m in 1..=40 {
        term = term.dot(&k) / m as f64;
        r += &term;
        if term.iter().all(|v| v.abs() < 1e-22) {
            break;
        }
    }
    c.dot(&r)
}

/// Perturbation amplitudes x seeds of the PBE0-none start survey.
pub const PERTURB_T: [f64; 3] = [1e-6, 1e-5, 1e-4];

pub const PERTURB_SEEDS: [u64; 2] = [1, 2];

/// Tri 4H s+p triplet PBE0 `exxdiv = none` (the first stage of the staged
/// ewald run, the one CI failed) from the core guess rotated by each
/// `(t, seed)`, with `edit` applied to `GammaRoksConfig::new("PBE0").scf`.
/// Returns `(t, seed, Ok((E, iterations)) | Err(reason))`.
pub fn pbe0_none_from_perturbed_starts(
    edit: impl Fn(&mut RhfConfig),
) -> Vec<(f64, u64, Result<(f64, usize), String>)> {
    let su = setup(cell_mult(&TRI_ATOMS, TRI_A, 3), sp_basis_h());
    let gcfg = ssf_grid(75, 302, 10.0);
    let grid = PeriodicGrid::build(&su.cell, &gcfg).expect("grid");
    assert_eq!(grid.len(), TRI_SSF_NPTS);
    let mut pxc = PeriodicXc::new(
        &su.cell,
        su.prep.basis_set(),
        "PBE0",
        &grid,
        &PeriodicXcConfig::default(),
    )
    .expect("PeriodicXc PBE0");
    let c0 = core_mos(&su.hc.s, &su.hc.h);
    let mut out = Vec::new();
    for t in PERTURB_T {
        for seed in PERTURB_SEEDS {
            let c = rotated(&c0, t, seed);
            let ortho = c.t().dot(&su.hc.s).dot(&c);
            assert!(
                max_abs_diff(&ortho, &Array2::eye(c.ncols())) < 1e-10,
                "rotated start is not S-orthonormal"
            );
            let mut cfg = GammaRoksConfig {
                exxdiv: ExxDiv::None,
                initial_mos: Some(c),
                ..GammaRoksConfig::new("PBE0")
            };
            edit(&mut cfg.scf);
            let r = gamma_roks_with_xc(&su.cell, &su.prep, &su.hc, ints(&su), &mut pxc, &cfg);
            let res = match r {
                Ok(o) if o.scf.converged => Ok((o.scf.energy, o.scf.iterations)),
                Ok(o) => Err(format!("not converged after {}", o.scf.iterations)),
                Err(e) => Err(e.to_string()),
            };
            eprintln!(
                "  PBE0 none t {t:.0e} seed {seed} (ls {}, max_iter {}): {res:?}",
                cfg.scf.level_shift, cfg.scf.max_iter
            );
            out.push((t, seed, res));
        }
    }
    out
}
