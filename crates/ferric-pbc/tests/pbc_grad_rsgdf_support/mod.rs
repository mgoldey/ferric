//! Shared setup of `pbc_grad_rsgdf.rs` (per commit) and
//! `validation_pbc_grad_rsgdf.rs` (weekly): every non-test item of the original
//! file, made `pub`.
#![allow(dead_code, unused_imports)]

pub use crate::common::*;
pub use ferric_core::basis::{self, BasisSet, Shell};
pub use ferric_core::mol::Molecule;
pub use ferric_integrals::basis_bridge::PreparedBasis;
pub use ferric_integrals::site_basis::SiteBasis;
pub use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
pub use ferric_pbc::dft::{
    gamma_rks, gamma_uks, GammaRksConfig, GammaUksConfig, GammaUksResult, PeriodicGridConfig,
};
pub use ferric_pbc::grad::{
    gamma_rhf_gradient_rsgdf, gamma_rhf_gradient_with, gamma_rks_gradient_rsgdf,
    gamma_uhf_gradient_rsgdf, gamma_uks_gradient_rsgdf, GammaGradConfig, GammaGradient,
    GradMutation, RsGdfGradSource,
};
pub use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig, DEFAULT_RSGDF_LINDEP};
pub use ferric_pbc::uhf::{gamma_uhf, GammaUhfConfig, GammaUhfIntegrals, GammaUhfResult};
pub use ferric_scf::result::{ScfResult, Spin};
pub use ndarray::Array2;
pub use std::collections::HashMap;
pub use std::sync::OnceLock;

/// ω of the nuclear-attraction split (as `pbc_grad.rs`).
pub const OMEGA: f64 = 0.8;

pub const HCORE_PRECISION: f64 = 1e-14;

/// RS-GDF split (the prototype's w = 1; the span anchor uses 1.2 so a slip
/// between h's c0 and the fit's c0' is visible).
pub const GDF_OMEGA: f64 = 1.0;

pub const AMPLE: usize = 1 << 31;

pub const FD_H: f64 = 1e-4;

pub const FD_BAR: f64 = 1e-7;

pub const MUTANT_BAR: f64 = 1e-4;

pub const NET_FORCE_BAR: f64 = 1e-10;

pub const EWALD_NONE_BAR: f64 = 1e-11;

pub const IDENTITY_BAR: f64 = 1e-12;

pub const XC_IDENTITY_BAR: f64 = 1e-11;

pub const SPAN_BAR: f64 = 1e-10;

/// Active-cut bars (module doc): prototype dk − FD 2.42e-9, std − FD
/// 1.48e-7, |F_dk − F_std| 2.9e-7.
pub const CUT_LINDEP: f64 = 3e-3;

pub const CUT_DK_BAR: f64 = 3e-8;

pub const CUT_STD_MIN: f64 = 7e-8;

pub const CUT_CROSS_MIN: f64 = 1e-7;

/// `run_grad_gdf_anchor.py` H2 (Bohr), a = 4.
pub const H2_ATOMS_G: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

pub const H2_COMPS: [(usize, usize); 3] = [(0, 0), (0, 2), (1, 1)];

/// H3 (Bohr), a = 4.5, doublet.
pub const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];

pub const H3_A: f64 = 4.5;

pub const H3_COMPS: [(usize, usize); 3] = [(0, 0), (2, 1), (1, 2)];

/// `run_grad_oracle.py` TRI_MOVED (Bohr), s+p basis.
pub const TRI_MOVED: [[f64; 3]; 4] = [
    [0.13, 0.25, 0.31],
    [0.02, 0.27, 1.66],
    [2.47, 2.41, 2.25],
    [3.52, 2.98, 2.71],
];

pub const TRI_COMPS: [(usize, usize); 2] = [(2, 1), (0, 2)];

pub const FIT_MUTANTS: [GradMutation; 5] = [
    GradMutation::FitNoMetric,
    GradMutation::FitNoAux,
    GradMutation::FitNoG0,
    GradMutation::FitG0Dense,
    GradMutation::FitNoLr3,
];

/// Hcore config at the file's `OMEGA` with `precision = HCORE_PRECISION`; the
/// other fields are the defaults of `with_omega`.
pub fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
}

/// RS-GDF config at splitting `omega` and `lindep` with `ExxDiv::None` and
/// budget `AMPLE`.
pub fn gdf_cfg(omega: f64, lindep: f64) -> RsGdfConfig {
    RsGdfConfig {
        omega,
        lindep,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

/// The prototype's anchor grid: (40, 50), SSF, D = 8.
pub fn grid_cfg() -> PeriodicGridConfig {
    PeriodicGridConfig {
        neighbour_cutoff: Some(8.0),
        ..PeriodicGridConfig::with_size(40, 50)
    }
}

/// Even-tempered s + Cartesian p aux on H: `a0 β^k`, k < n (the prototype's
/// "ET l≤1" sets; one unit-normalised primitive per shell).
pub fn et_aux(a0: f64, beta: f64, n: usize) -> BasisSet {
    let mut shells = Vec::new();
    for k in 0..n {
        let a = a0 * beta.powi(k as i32);
        for l in [0, 1] {
            shells.push(Shell {
                l,
                pure: false,
                exponents: vec![a],
                coefficients: vec![1.0],
            });
        }
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

/// H2 ET l≤1 a0 0.3, β 2.5, 5 exponents: 40 aux (prototype s_min 3.9e-5).
pub fn et40() -> BasisSet {
    et_aux(0.3, 2.5, 5)
}

/// The bundled `cc-pvdz-ri` auxiliary basis; panics if it is not bundled.
pub fn cc_pvdz_ri() -> BasisSet {
    basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri")
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
    pub aux: PreparedBasis,
    /// `build_for_gradient`, exxdiv none (the drivers / K builders take the
    /// Madelung shift from their own config).
    pub gdf: RsGdf,
}

/// Builds the setup: the periodic hcore of `cell` in `bs`, the aux basis
/// `aux_bs` on the same atoms, and the gradient-capable `RsGdf` at
/// `GDF_OMEGA` and `lindep`.
pub fn setup(cell: Cell, bs: &BasisSet, aux_bs: &BasisSet, lindep: f64) -> Setup {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).expect("hcore");
    let aux = PreparedBasis::new(cell.mol(), aux_bs).expect("aux prep");
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &gdf_cfg(GDF_OMEGA, lindep))
        .expect("RsGdf::build_for_gradient");
    Setup {
        cell,
        prep,
        hc,
        aux,
        gdf,
    }
}

/// Energy-only build (what the FD differentiates): `RsGdf::build`.
pub fn energy_gdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    aux_bs: &BasisSet,
    lindep: f64,
) -> (PreparedBasis, RsGdf) {
    let aux = PreparedBasis::new(cell.mol(), aux_bs).expect("aux prep");
    let gdf = RsGdf::build(cell, prep, &aux, &hc.s, &gdf_cfg(GDF_OMEGA, lindep)).expect("RsGdf");
    (aux, gdf)
}

pub fn rhf_on(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    gdf: &RsGdf,
    exx: ExxDiv,
) -> ScfResult {
    let g = gdf.clone().with_exxdiv(cell, exx).expect("with_exxdiv");
    gamma_rhf_jk(
        cell,
        prep,
        hc,
        Box::new(g.j_builder()),
        Box::new(g.k_builder()),
    )
}

/// The RS-GDF gradient source of the setup's `gdf` and `aux`, with no aux
/// Jacobian (aux centres on the cell's atoms).
pub fn src(su: &Setup) -> RsGdfGradSource<'_> {
    RsGdfGradSource {
        gdf: &su.gdf,
        aux: &su.aux,
        aux_jac: None,
    }
}

/// Default Gamma gradient config with budget `AMPLE` and the test-only
/// `mutation`.
pub fn gcfg(mutation: Option<GradMutation>) -> GammaGradConfig {
    GammaGradConfig {
        mutation,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

/// The RS-GDF Gamma RHF gradient of `scf` at exchange divergence `exx` with
/// test-only mutation `m`; panics on error.
pub fn rhf_grad(
    su: &Setup,
    scf: &ScfResult,
    exx: ExxDiv,
    m: Option<GradMutation>,
) -> GammaGradient {
    gamma_rhf_gradient_rsgdf(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &src(su),
        scf,
        exx,
        &gcfg(m),
    )
    .expect("gamma_rhf_gradient_rsgdf")
}

/// The RS-GDF Gamma UHF gradient of `scf` at exchange divergence `exx` with
/// test-only mutation `m`; panics on error.
pub fn uhf_grad(
    su: &Setup,
    scf: &ScfResult,
    exx: ExxDiv,
    m: Option<GradMutation>,
) -> GammaGradient {
    gamma_uhf_gradient_rsgdf(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &src(su),
        scf,
        exx,
        &gcfg(m),
    )
    .expect("gamma_uhf_gradient_rsgdf")
}

pub fn uks_grad(
    su: &Setup,
    scf: &ScfResult,
    cfg: &GammaUksConfig,
    m: Option<GradMutation>,
) -> GammaGradient {
    gamma_uks_gradient_rsgdf(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &src(su),
        scf,
        cfg,
        &gcfg(m),
    )
    .expect("gamma_uks_gradient_rsgdf")
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

/// Gamma UHF on the RS-GDF integrals `gdf`; panics on error or if the SCF
/// does not converge.
pub fn uhf_on(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    gdf: &RsGdf,
    cfg: &GammaUhfConfig,
) -> GammaUhfResult {
    let r = gamma_uhf(cell, prep, hc, GammaUhfIntegrals::RsGdf(gdf), cfg).expect("gamma_uhf");
    assert!(r.scf.converged);
    r
}

/// Gamma UKS on the RS-GDF integrals `gdf`; panics on error or if the SCF
/// does not converge.
pub fn uks_on(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    gdf: &RsGdf,
    cfg: &GammaUksConfig,
) -> GammaUksResult {
    let r = gamma_uks(cell, prep, hc, GammaUhfIntegrals::RsGdf(gdf), cfg)
        .unwrap_or_else(|e| panic!("gamma_uks {}: {e}", cfg.functional));
    assert!(r.scf.converged);
    r
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

pub type Fd = Vec<((usize, usize), Vec<f64>)>;

/// Central FD `(E(+h) − E(−h))/2h` of every energy `energies` returns.
pub fn fd<F>(
    pos: &[[f64; 3]],
    lattice: [[f64; 3]; 3],
    mult: usize,
    comps: &[(usize, usize)],
    h: f64,
    energies: F,
) -> Fd
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
pub fn max_fd_err(g: &Array2<f64>, fdv: &Fd, k: usize) -> f64 {
    fdv.iter()
        .map(|((a, x), v)| (g[(*a, *x)] - v[k]).abs())
        .fold(0.0_f64, f64::max)
}

pub fn report(tag: &str, g: &GammaGradient, err: f64) {
    let f = g.fit.as_ref().expect("RS-GDF diagnostics");
    eprintln!(
        "{tag}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, comm {:.1e}; naux {} dropped {} \
         (s_kept_min {:.2e}, s_drop_max {:?}, cross {:.1e}); SR3 {} SR2 {} half-G {}\n\
         parts: orb_sr {:.3e} orb_lr {:.3e} aux_sr {:.3e} aux_lr {:.3e} met_sr {:.3e} met_lr {:.3e} g0 {:.3e}\n{:.10}",
        g.net_force,
        g.commutator,
        f.naux,
        f.n_dropped,
        f.s_kept_min,
        f.s_dropped_max,
        f.cross_norm,
        f.n_sr3_deriv,
        f.n_sr2_deriv,
        f.n_g_half,
        amax(&g.parts.fit_orb_sr),
        amax(&g.parts.fit_orb_lr),
        amax(&g.parts.fit_aux_sr),
        amax(&g.parts.fit_aux_lr),
        amax(&g.parts.fit_metric_sr),
        amax(&g.parts.fit_metric_lr),
        amax(&g.parts.fit_g0),
        g.grad
    );
}

/// Largest element-wise `|a|` (0 for an empty matrix).
pub fn amax(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

// ------------------------------------------------------------------ H2 RHF

pub static H2_FD: OnceLock<Fd> = OnceLock::new();

/// FD `[none, ewald]` of the H2 ET-40 RS-GDF RHF energy.
pub fn h2_fd() -> &'static Fd {
    H2_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let aux_bs = et40();
        fd(&H2_ATOMS_G, cubic(4.0), 1, &H2_COMPS, FD_H, |c| {
            let prep = prep_for(c, &bs);
            let hc = periodic_hcore(c, &prep, &hcore_cfg()).unwrap();
            let (_, gdf) = energy_gdf(c, &prep, &hc, &aux_bs, DEFAULT_RSGDF_LINDEP);
            [ExxDiv::None, ExxDiv::Ewald]
                .iter()
                .map(|&e| rhf_on(c, &prep, &hc, &gdf, e).energy)
                .collect()
        })
    })
}

/// The H2 setup: `H2_ATOMS_G` in a cubic cell of edge 4 Bohr, singlet,
/// PySCF STO-3G with the `et40` aux basis.
pub fn h2_setup() -> Setup {
    setup(
        cell_at(&H2_ATOMS_G, cubic(4.0), 1),
        &pyscf_sto3g_h(),
        &et40(),
        DEFAULT_RSGDF_LINDEP,
    )
}

// ------------------------------------------------------------------ H3 UHF

/// The H3 doublet setup: `H3_ATOMS` in a cubic cell of edge `H3_A` Bohr,
/// PySCF STO-3G with `cc-pvdz-ri`.
pub fn h3_setup() -> Setup {
    setup(
        cell_at(&H3_ATOMS, cubic(H3_A), 2),
        &pyscf_sto3g_h(),
        &cc_pvdz_ri(),
        DEFAULT_RSGDF_LINDEP,
    )
}

/// Reference H3 doublet UHF: `(none stage, ewald)`.
pub fn h3_uhf_ref(su: &Setup) -> (ScfResult, ScfResult) {
    let r = uhf_on(
        &su.cell,
        &su.prep,
        &su.hc,
        &su.gdf,
        &uhf_cfg(ExxDiv::Ewald, None),
    );
    (r.none_stage.clone().expect("none stage"), r.scf)
}

pub static H3_UHF_FD: OnceLock<Fd> = OnceLock::new();

pub fn h3_uhf_fd() -> &'static Fd {
    H3_UHF_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let aux_bs = cc_pvdz_ri();
        let (none, _) = h3_uhf_ref(&h3_setup());
        let init = mos_of(&none);
        fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, FD_H, |c| {
            let prep = prep_for(c, &bs);
            let hc = periodic_hcore(c, &prep, &hcore_cfg()).unwrap();
            let (_, gdf) = energy_gdf(c, &prep, &hc, &aux_bs, DEFAULT_RSGDF_LINDEP);
            let r = uhf_on(c, &prep, &hc, &gdf, &uhf_cfg(ExxDiv::Ewald, init.clone()));
            vec![r.none_stage.expect("none stage").energy, r.scf.energy]
        })
    })
}

// ------------------------------------------------------------ H3 UKS PBE0

pub static H3_UKS_FD: OnceLock<Fd> = OnceLock::new();

/// The Gamma UKS PBE0 reference of the setup at `ExxDiv::Ewald` on its
/// RS-GDF integrals.
pub fn h3_pbe0_ref(su: &Setup) -> GammaUksResult {
    uks_on(
        &su.cell,
        &su.prep,
        &su.hc,
        &su.gdf,
        &uks_cfg("PBE0", ExxDiv::Ewald, None),
    )
}

/// FD `[none stage, ewald]` of the H3 UKS PBE0 energy (grid rebuilt per
/// geometry, seeded with the reference none-stage MOs).
pub fn h3_uks_fd() -> &'static Fd {
    H3_UKS_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let aux_bs = cc_pvdz_ri();
        let r = h3_pbe0_ref(&h3_setup());
        let seed = mos_of(r.none_stage.as_ref().expect("none stage"));
        fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, FD_H, |c| {
            let prep = prep_for(c, &bs);
            let hc = periodic_hcore(c, &prep, &hcore_cfg()).unwrap();
            let (_, gdf) = energy_gdf(c, &prep, &hc, &aux_bs, DEFAULT_RSGDF_LINDEP);
            let r = uks_on(
                c,
                &prep,
                &hc,
                &gdf,
                &uks_cfg("PBE0", ExxDiv::Ewald, seed.clone()),
            );
            vec![r.none_stage.expect("none stage").energy, r.scf.energy]
        })
    })
}

// ------------------------------------------------- exact aux span vs dense

/// The 24 exact-span aux sites (`pbc_rsgdf.rs::anchor_sites`, all 8
/// classes) and their Jacobian `dC_k/dR_A`: (0,0) classes ride on atom 0,
/// (1,1) on atom 1, (0,1) midpoints half on each.
pub fn span_sites(cell: &Cell) -> (Vec<[f64; 4]>, Array2<f64>) {
    let r = cell.positions();
    let a = cell.lattice();
    let mut out = Vec::new();
    let mut jac = Array2::<f64>::zeros((24, 2));
    let mut row = 0;
    for (i, j) in [(0usize, 0usize), (1, 1), (0, 1)] {
        for k in 0..8usize {
            let h = [(k >> 2) & 1, (k >> 1) & 1, k & 1];
            let mut c = [0.0; 3];
            for d in 0..3 {
                let ha: f64 = (0..3).map(|x| h[x] as f64 * a[x][d]).sum();
                c[d] = 0.5 * (r[i][d] + r[j][d] + ha);
            }
            out.push([c[0], c[1], c[2], 1.0]);
            jac[(row, i)] += 0.5;
            jac[(row, j)] += 0.5;
            row += 1;
        }
    }
    (out, jac)
}
