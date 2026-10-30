//! Shared setup of `pbc_grad_open.rs` (per commit) and
//! `validation_pbc_grad_open.rs` (weekly): every non-test item of the original
//! file, made `pub`.
#![allow(dead_code, unused_imports)]

pub use crate::common::*;
pub use ferric_core::basis::BasisSet;
pub use ferric_core::mol::Molecule;
pub use ferric_integrals::basis_bridge::PreparedBasis;
pub use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
pub use ferric_pbc::dft::{
    gamma_rks, gamma_uks, GammaRksConfig, GammaUksConfig, GammaUksResult, PeriodicGridConfig,
};
pub use ferric_pbc::grad::{
    gamma_rhf_gradient_with, gamma_rks_gradient_with, gamma_uhf_gradient_with,
    gamma_uks_gradient_with, GammaGradConfig, GammaGradient, GradMutation,
};
pub use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::uhf::{gamma_uhf, GammaUhfConfig, GammaUhfIntegrals, GammaUhfResult};
pub use ferric_scf::result::{ScfResult, Spin};
pub use ndarray::Array2;
pub use std::sync::OnceLock;

/// ω of the nuclear-attraction split (as `pbc_grad.rs`).
pub const OMEGA: f64 = 0.8;

pub const HCORE_PRECISION: f64 = 1e-14;

pub const FD_H: f64 = 1e-4;

/// H2 a = 4 KS step (module doc: neighbour-mask jumps at h ≥ 1e-4).
pub const FD_H_H2_KS: f64 = 5e-5;

pub const FD_BAR: f64 = 1e-7;

pub const MUTANT_BAR: f64 = 1e-3;

pub const NET_FORCE_BAR: f64 = 1e-10;

pub const IDENTITY_BAR: f64 = 1e-12;

/// UKS(D/2, D/2) vs RKS(D): polarized vs unpolarized libxc kernels, and the
/// `ρ_σ > 1e-10` vs `ρ > 1e-10` potential gates differ in a thin tail shell
/// (the same asymmetry the energy path has; `pbc_uks.rs` asserts UKS ≡ RKS
/// energies at 1e-11).
pub const XC_IDENTITY_BAR: f64 = 1e-11;

pub const EWALD_NONE_BAR: f64 = 1e-11;

/// `run_grad_open_anchor.py` H2 (Bohr), a = 4, STO-3G.
pub const GRAD_H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

pub const H2_COMPS: [(usize, usize); 3] = [(0, 0), (0, 2), (1, 1)];

/// `run_grad_open_anchor.py` H3 (Bohr), a = 4.5, STO-3G, doublet (2, 1).
pub const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];

pub const H3_A: f64 = 4.5;

pub const H3_COMPS: [(usize, usize); 3] = [(0, 0), (2, 1), (1, 2)];

/// `run_grad_oracle.py` TRI_MOVED (Bohr), s+p basis, triplet (3, 1).
pub const TRI_MOVED: [[f64; 3]; 4] = [
    [0.13, 0.25, 0.31],
    [0.02, 0.27, 1.66],
    [2.47, 2.41, 2.25],
    [3.52, 2.98, 2.71],
];

pub const TRI_COMPS: [(usize, usize); 2] = [(2, 1), (0, 2)];

/// Hcore config at the file's `OMEGA` with `precision = HCORE_PRECISION`; the
/// other fields are the defaults of `with_omega`.
pub fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
}

/// The prototype's anchor grid: (40, 50), SSF, D = 8.
pub fn grid_cfg() -> PeriodicGridConfig {
    PeriodicGridConfig {
        neighbour_cutoff: Some(8.0),
        ..PeriodicGridConfig::with_size(40, 50)
    }
}

/// A neutral hydrogen cell with atoms at `pos` (Bohr), lattice rows `lattice`
/// (Bohr) and spin multiplicity `mult`.
pub fn cell_at(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

/// `pos` (Bohr) with Cartesian component `x` of atom `a` displaced by `h` Bohr.
pub fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

pub struct Setup {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub hc: PeriodicHcore,
    pub eri: DenseAftEri,
}

/// hcore + the dense tensor (its own exxdiv is irrelevant: the drivers and
/// the gradients take exxdiv from their configs).
pub fn setup(cell: Cell, bs: &BasisSet) -> Setup {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).expect("hcore");
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("dense AFT");
    Setup {
        cell,
        prep,
        hc,
        eri,
    }
}

pub type Mos = Option<(Array2<f64>, Array2<f64>)>;

/// The `(α, β)` MO coefficients of `r` as a starting guess; panics without
/// beta MOs.
pub fn mos_of(r: &ScfResult) -> Mos {
    Some((r.mos_alpha.clone(), r.mos_beta.clone().expect("beta MOs")))
}

/// UHF config with exchange-divergence treatment `exx` and starting MOs
/// `init`; the other fields are the defaults.
pub fn uhf_cfg(exx: ExxDiv, init: Mos) -> GammaUhfConfig {
    GammaUhfConfig {
        exxdiv: exx,
        initial_mos: init,
        ..GammaUhfConfig::default()
    }
}

/// UKS config for functional `xc` on `grid_cfg()` with exchange
/// divergence `exx` and starting MOs `init`.
pub fn uks_cfg(xc: &str, exx: ExxDiv, init: Mos) -> GammaUksConfig {
    GammaUksConfig {
        grid: grid_cfg(),
        exxdiv: exx,
        initial_mos: init,
        ..GammaUksConfig::new(xc)
    }
}

/// RKS config for functional `xc` on `grid_cfg()` with exchange
/// divergence `exx`.
pub fn rks_cfg(xc: &str, exx: ExxDiv) -> GammaRksConfig {
    GammaRksConfig {
        grid: grid_cfg(),
        exxdiv: exx,
        ..GammaRksConfig::new(xc)
    }
}

/// Gamma UHF on the setup's dense-AFT integrals; panics on error or if the
/// SCF does not converge.
pub fn uhf(su: &Setup, cfg: &GammaUhfConfig) -> GammaUhfResult {
    let r = gamma_uhf(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        cfg,
    )
    .expect("gamma_uhf");
    assert!(r.scf.converged);
    r
}

/// Gamma UKS on the setup's dense-AFT integrals; panics on error or if the
/// SCF does not converge.
pub fn uks(su: &Setup, cfg: &GammaUksConfig) -> GammaUksResult {
    let r = gamma_uks(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        cfg,
    )
    .unwrap_or_else(|e| panic!("gamma_uks {}: {e}", cfg.functional));
    assert!(r.scf.converged);
    r
}

/// Gamma RKS on the setup's dense-AFT integrals; returns the SCF result and
/// panics on error.
pub fn rks(su: &Setup, cfg: &GammaRksConfig) -> ScfResult {
    gamma_rks(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        cfg,
    )
    .unwrap_or_else(|e| panic!("gamma_rks {}: {e}", cfg.functional))
    .scf
}

/// Default Gamma gradient config with the test-only `mutation`.
pub fn gcfg(mutation: Option<GradMutation>) -> GammaGradConfig {
    GammaGradConfig {
        mutation,
        ..Default::default()
    }
}

/// The dense-AFT Gamma UHF gradient of `scf` at exchange divergence `exx`
/// with test-only mutation `m`; panics on error.
pub fn uhf_grad(
    su: &Setup,
    scf: &ScfResult,
    exx: ExxDiv,
    m: Option<GradMutation>,
) -> GammaGradient {
    gamma_uhf_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        scf,
        exx,
        &gcfg(m),
    )
    .expect("gamma_uhf_gradient")
}

pub fn uks_grad(
    su: &Setup,
    scf: &ScfResult,
    cfg: &GammaUksConfig,
    m: Option<GradMutation>,
) -> GammaGradient {
    gamma_uks_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        scf,
        cfg,
        &gcfg(m),
    )
    .expect("gamma_uks_gradient")
}

pub fn rks_grad(
    su: &Setup,
    scf: &ScfResult,
    cfg: &GammaRksConfig,
    m: Option<GradMutation>,
) -> GammaGradient {
    gamma_rks_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        scf,
        cfg,
        &gcfg(m),
    )
    .expect("gamma_rks_gradient")
}

/// The same closed-shell density as an unrestricted result `(D/2, D/2)`.
pub fn as_unrestricted(r: &ScfResult) -> ScfResult {
    let mut u = r.clone();
    let half = &r.density_total * 0.5;
    u.spin = Spin::Unrestricted;
    u.density_alpha = half.clone();
    u.density_beta = Some(half);
    u.mos_beta = Some(r.mos_alpha.clone());
    u.eps_beta = Some(r.eps_alpha.clone());
    u.fock_beta = Some(r.fock_alpha.clone());
    u
}

/// Largest element-wise `|a − b|`. `zip` truncates silently if the shapes
/// differ.
pub fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Central FD `(E(+h) − E(−h))/2h` of every energy `energies` returns, per
/// component.
pub fn fd<F>(
    pos: &[[f64; 3]],
    lattice: [[f64; 3]; 3],
    mult: usize,
    comps: &[(usize, usize)],
    h: f64,
    energies: F,
) -> Vec<((usize, usize), Vec<f64>)>
where
    F: Fn(&Cell) -> Vec<f64>,
{
    comps
        .iter()
        .map(|&(a, x)| {
            let ep = energies(&cell_at(&moved(pos, a, x, h), lattice, mult));
            let em = energies(&cell_at(&moved(pos, a, x, -h), lattice, mult));
            let d = ep
                .iter()
                .zip(&em)
                .map(|(p, m)| (p - m) / (2.0 * h))
                .collect();
            ((a, x), d)
        })
        .collect()
}

/// Largest `|g[a, x] − fd[(a, x)][k]|` over the finite-difference entries:
/// `g` is the analytic `(natoms, 3)` gradient and `k` selects the energy
/// column.
pub fn max_fd_err(g: &Array2<f64>, fdv: &[((usize, usize), Vec<f64>)], k: usize) -> f64 {
    fdv.iter()
        .map(|((a, x), v)| (g[(*a, *x)] - v[k]).abs())
        .fold(0.0_f64, f64::max)
}

/// The H3 doublet cell: `H3_ATOMS` in a cubic lattice of edge `H3_A` Bohr,
/// multiplicity 2.
pub fn h3_cell() -> Cell {
    cell_at(&H3_ATOMS, cubic(H3_A), 2)
}

// ------------------------------------------------------------------ H3 UHF

/// Reference H3 doublet UHF: the staged ewald run and its none stage.
pub fn h3_uhf_ref(su: &Setup) -> (ScfResult, ScfResult) {
    let r = uhf(su, &uhf_cfg(ExxDiv::Ewald, None));
    let none = r.none_stage.clone().expect("staged none stage");
    (none, r.scf)
}

pub static H3_UHF_FD: OnceLock<Vec<((usize, usize), Vec<f64>)>> = OnceLock::new();

/// FD `[none, ewald]` of the H3 UHF energy (one staged run per geometry,
/// seeded with the reference none-stage MOs so every geometry lands on the
/// same state).
pub fn h3_uhf_fd() -> &'static Vec<((usize, usize), Vec<f64>)> {
    H3_UHF_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let (none, _) = h3_uhf_ref(&setup(h3_cell(), &bs));
        let init = mos_of(&none);
        fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, FD_H, |c| {
            let r = uhf(
                &setup(c.clone(), &bs),
                &uhf_cfg(ExxDiv::Ewald, init.clone()),
            );
            vec![r.none_stage.expect("none stage").energy, r.scf.energy]
        })
    })
}

// ------------------------------------------------------------------ H3 UKS

/// Reference H3 doublet UKS results: LDA and PBE (`α = 0`, so exxdiv does
/// not enter their energy), PBE0 none and PBE0 ewald (the none entry is the
/// none stage of the staged PBE0 ewald run).
pub fn h3_uks_refs(su: &Setup) -> Vec<(GammaUksConfig, ScfResult)> {
    let lda = uks(su, &uks_cfg("LDA", ExxDiv::Ewald, None)).scf;
    let pbe = uks(su, &uks_cfg("PBE", ExxDiv::Ewald, None)).scf;
    let pbe0 = uks(su, &uks_cfg("PBE0", ExxDiv::Ewald, None));
    let pbe0_none = pbe0.none_stage.clone().expect("staged PBE0 none stage");
    vec![
        (uks_cfg("LDA", ExxDiv::Ewald, None), lda),
        (uks_cfg("PBE", ExxDiv::Ewald, None), pbe),
        (uks_cfg("PBE0", ExxDiv::None, None), pbe0_none),
        (uks_cfg("PBE0", ExxDiv::Ewald, None), pbe0.scf),
    ]
}

pub static H3_UKS_FD: OnceLock<Vec<((usize, usize), Vec<f64>)>> = OnceLock::new();

/// FD of the four H3 UKS energies (grid rebuilt per geometry; seeded with
/// the reference MOs).
pub fn h3_uks_fd() -> &'static Vec<((usize, usize), Vec<f64>)> {
    H3_UKS_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let refs = h3_uks_refs(&setup(h3_cell(), &bs));
        let seeds: Vec<Mos> = refs.iter().map(|(_, r)| mos_of(r)).collect();
        fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, FD_H, |c| {
            let su = setup(c.clone(), &bs);
            let lda = uks(&su, &uks_cfg("LDA", ExxDiv::Ewald, seeds[0].clone()));
            let pbe = uks(&su, &uks_cfg("PBE", ExxDiv::Ewald, seeds[1].clone()));
            // The staged PBE0 run starts from the none-stage MOs.
            let pbe0 = uks(&su, &uks_cfg("PBE0", ExxDiv::Ewald, seeds[2].clone()));
            vec![
                lda.scf.energy,
                pbe.scf.energy,
                pbe0.none_stage.expect("none stage").energy,
                pbe0.scf.energy,
            ]
        })
    })
}

// ------------------------------------------------------------ H2 RKS anchor

pub static H2_RKS_FD: OnceLock<Vec<((usize, usize), Vec<f64>)>> = OnceLock::new();

pub const H2_RKS_XCS: [&str; 2] = ["LDA", "PBE0"];

/// The cached finite-difference RKS energies for the H2 cell
/// (`GRAD_H2_ATOMS`, cubic edge 4 Bohr) over `H2_COMPS` and every
/// functional of `H2_RKS_XCS`, computed on first use.
pub fn h2_rks_fd() -> &'static Vec<((usize, usize), Vec<f64>)> {
    H2_RKS_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        fd(&GRAD_H2_ATOMS, cubic(4.0), 1, &H2_COMPS, FD_H_H2_KS, |c| {
            let su = setup(c.clone(), &bs);
            H2_RKS_XCS
                .iter()
                .map(|xc| rks(&su, &rks_cfg(xc, ExxDiv::Ewald)).energy)
                .collect()
        })
    })
}

// ---------------------------------------------------- triclinic s+p triplet

/// Richardson-extrapolated central FD from steps `h` and `h/2`: removes the
/// `h² f'''/6` truncation, which is large on this state (prototype frozen-
/// grid h-scan, PBE (0,z): −5.7e-8 / −2.27e-7 at h = 5e-5 / 1e-4).
pub fn richardson(
    fd_h: &[((usize, usize), Vec<f64>)],
    fd_h2: &[((usize, usize), Vec<f64>)],
) -> Vec<((usize, usize), Vec<f64>)> {
    fd_h.iter()
        .zip(fd_h2)
        .map(|((c, a), (_, b))| {
            (
                *c,
                a.iter().zip(b).map(|(x, y)| (4.0 * y - x) / 3.0).collect(),
            )
        })
        .collect()
}
