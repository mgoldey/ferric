//! Shared setup of `pbc_grad_ro.rs` (per commit) and
//! `validation_pbc_grad_ro.rs` (weekly): every non-test item of the original
//! file, made `pub`.
#![allow(dead_code, unused_imports)]

pub use crate::common::*;
pub use ferric_core::basis::{self, BasisSet};
pub use ferric_core::mol::Molecule;
pub use ferric_integrals::basis_bridge::PreparedBasis;
pub use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
pub use ferric_pbc::dft::{gamma_rks, GammaRksConfig, PeriodicGridConfig};
pub use ferric_pbc::ewald::madelung_constant;
pub use ferric_pbc::grad::{
    gamma_rhf_gradient_with, gamma_rks_gradient_with, gamma_rohf_gradient_rsgdf,
    gamma_rohf_gradient_with, gamma_roks_gradient_with, gamma_uhf_gradient_with, rohf_lagrangian_w,
    rohf_orbital_gradient, GammaGradConfig, GammaGradient, GradMutation, RsGdfGradSource,
};
pub use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::rohf::{
    gamma_rohf, gamma_roks, GammaRohfConfig, GammaRohfResult, GammaRoksConfig, GammaRoksResult,
};
pub use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig, DEFAULT_RSGDF_LINDEP};
pub use ferric_pbc::stress::{gamma_rohf_stress, GammaStressConfig};
pub use ferric_pbc::uhf::{gamma_uhf, GammaUhfConfig, GammaUhfIntegrals};
pub use ferric_scf::fock::{JBuilder, KBuilder};
pub use ferric_scf::result::{ScfResult, Spin};
pub use ndarray::{s, Array2};
pub use std::sync::OnceLock;

/// ω of the nuclear-attraction split (as `pbc_grad.rs`).
pub const OMEGA: f64 = 0.8;

pub const HCORE_PRECISION: f64 = 1e-14;

pub const FD_H: f64 = 1e-4;

/// ROKS step (module doc: hard-mask energy jumps at h ≥ 1e-4).
pub const FD_H_KS: f64 = 5e-5;

pub const FD_BAR: f64 = 1e-7;

pub const MUTANT_BAR: f64 = 1e-4;

pub const NET_FORCE_BAR: f64 = 1e-10;

pub const IDENTITY_BAR: f64 = 1e-12;

/// ROKS(D/2, D/2) vs RKS(D): polarized vs unpolarized libxc kernel and the
/// per-spin density gates (as `pbc_grad_open.rs`).
pub const XC_IDENTITY_BAR: f64 = 1e-11;

pub const EWALD_NONE_BAR: f64 = 1e-11;

/// `|F(W_UHF-form) − F(W_RO)|` (prototype ≤ 1.1e-12 at SCF stop 1e-11;
/// ferric stops on ΔP_rms < 1e-10, so the SCF residual sets the scale).
pub const W_IDENTITY_FORCE_BAR: f64 = 1e-10;

/// `|W_RO − W_UHF-form|` elementwise (prototype ≤ 2.4e-12).
pub const W_IDENTITY_BAR: f64 = 1e-8;

/// Occupation-subspace continuity of a displaced SCF (prototype 4.2e-5).
pub const CONT_BAR: f64 = 1e-3;

/// ROHF ≠ UHF force: asserted only above the FD floor.
pub const RO_VS_U_FLOOR: f64 = 1e-5;

/// Stress FD bar for an open shell (`pbc_stress.rs` FD_BAR_OPEN).
pub const STRESS_FD_BAR_OPEN: f64 = 2e-7;

pub const STRESS_ANTISYM_BAR: f64 = 1e-10;

/// RS-GDF split and ample budget (as `pbc_grad_rsgdf.rs`).
pub const GDF_OMEGA: f64 = 1.0;

pub const AMPLE: usize = 1 << 31;

/// `run_grad_ro_anchor.py` H3 (Bohr), a = 4.5, STO-3G, doublet (nd, no) = (1, 1).
pub const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];

pub const H3_A: f64 = 4.5;

pub const H3_COMPS: [(usize, usize); 3] = [(0, 0), (2, 1), (1, 2)];

/// `run_grad_ro_anchor.py` H4 (Bohr), a = 5, STO-3G, triplet (1, 2): two
/// stretched H2 units.
pub const H4_ATOMS: [[f64; 3]; 4] = [
    [0.3, 0.2, 0.1],
    [0.35, 0.12, 1.5],
    [2.6, 2.4, 2.2],
    [2.7, 2.35, 3.65],
];

pub const H4_A: f64 = 5.0;

pub const H4_COMPS: [(usize, usize); 3] = [(0, 0), (2, 1), (3, 2)];

/// Closed-shell H2 (Bohr), a = 4.
pub const GRAD_H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

pub const ALL9: [(usize, usize); 9] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
];

pub type Fd = Vec<((usize, usize), Vec<f64>)>;

pub type Mat3 = [[f64; 3]; 3];

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

/// The H3 doublet cell: `H3_ATOMS` in a cubic lattice of edge `H3_A` Bohr,
/// multiplicity 2.
pub fn h3_cell() -> Cell {
    cell_at(&H3_ATOMS, cubic(H3_A), 2)
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

/// ROHF config with exchange divergence `exx` and starting MOs `init`; the
/// other fields are the defaults.
pub fn rohf_cfg(exx: ExxDiv, init: Option<Array2<f64>>) -> GammaRohfConfig {
    GammaRohfConfig {
        exxdiv: exx,
        initial_mos: init,
        ..GammaRohfConfig::default()
    }
}

/// `GammaRoksConfig::new` (hybrids: level shift 0.05, 600 iterations) on
/// the anchor grid.
pub fn roks_cfg(xc: &str, exx: ExxDiv, init: Option<Array2<f64>>) -> GammaRoksConfig {
    GammaRoksConfig {
        grid: grid_cfg(),
        exxdiv: exx,
        initial_mos: init,
        ..GammaRoksConfig::new(xc)
    }
}

/// Gamma ROHF on the setup's dense-AFT integrals; panics on error, on an
/// unconverged SCF, or if the result is not restricted open-shell.
pub fn rohf_on(su: &Setup, cfg: &GammaRohfConfig) -> GammaRohfResult {
    let r = gamma_rohf(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        cfg,
    )
    .expect("gamma_rohf");
    assert!(r.scf.converged);
    assert_eq!(r.scf.spin, Spin::RestrictedOpen);
    r
}

/// Gamma ROKS on the setup's dense-AFT integrals; panics on error or if the
/// SCF does not converge.
pub fn roks_on(su: &Setup, cfg: &GammaRoksConfig) -> GammaRoksResult {
    let r = gamma_roks(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        cfg,
    )
    .unwrap_or_else(|e| panic!("gamma_roks {}: {e}", cfg.functional));
    assert!(r.scf.converged);
    r
}

/// Default Gamma gradient config with the test-only `mutation`.
pub fn gcfg(mutation: Option<GradMutation>) -> GammaGradConfig {
    GammaGradConfig {
        mutation,
        ..Default::default()
    }
}

/// The dense-AFT Gamma ROHF gradient of `scf` at exchange divergence `exx`
/// with test-only mutation `m`; panics on error.
pub fn rohf_grad(
    su: &Setup,
    scf: &ScfResult,
    exx: ExxDiv,
    m: Option<GradMutation>,
) -> GammaGradient {
    gamma_rohf_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        scf,
        exx,
        &gcfg(m),
    )
    .expect("gamma_rohf_gradient")
}

pub fn roks_grad(
    su: &Setup,
    scf: &ScfResult,
    cfg: &GammaRoksConfig,
    m: Option<GradMutation>,
) -> GammaGradient {
    gamma_roks_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        scf,
        cfg,
        &gcfg(m),
    )
    .expect("gamma_roks_gradient")
}

/// Largest element-wise `|a − b|`. `zip` truncates silently if the shapes
/// differ.
pub fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

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

/// `max(|‖C_cᵀ S C_c^ref‖² − nd|, |‖C_oᵀ S C_o^ref‖² − no|)`: 0 (up to the
/// O(h) metric change) when a displaced SCF continues the reference
/// occupation pattern. ROHF MO columns are `[closed | open | virtual]`.
pub fn continuity(
    r: &ScfResult,
    nocc: (usize, usize),
    c_ref: &Array2<f64>,
    s_mat: &Array2<f64>,
) -> f64 {
    let (na, nb) = nocc;
    let c = &r.mos_alpha;
    let blk = |lo: usize, hi: usize| -> f64 {
        if hi == lo {
            return 0.0;
        }
        let ov = c
            .slice(s![.., lo..hi])
            .t()
            .dot(s_mat)
            .dot(&c_ref.slice(s![.., lo..hi]));
        (ov.iter().map(|v| v * v).sum::<f64>() - (hi - lo) as f64).abs()
    };
    blk(0, nb).max(blk(nb, na))
}

/// ROHF spin Focks `F_σ = h + J[D] − K[D_σ] − v_M S D_σ S` rebuilt in the
/// test from the dense tensor (independent of the gradient's own rebuild).
pub fn rohf_spin_focks(su: &Setup, scf: &ScfResult, exx: ExxDiv) -> (Array2<f64>, Array2<f64>) {
    let n = su.prep.nbasis();
    let vm = match exx {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => madelung_constant(&su.cell).unwrap(),
    };
    let da = &scf.density_alpha;
    let db = scf.density_beta.as_ref().unwrap();
    let mut j = Array2::<f64>::zeros((n, n));
    su.eri.j_builder().build(&(da + db), &mut j).unwrap();
    let mut ka = Array2::<f64>::zeros((n, n));
    let mut kb = Array2::<f64>::zeros((n, n));
    su.eri
        .k_builder_with_madelung(vm)
        .build(da, &mut ka)
        .unwrap();
    su.eri
        .k_builder_with_madelung(vm)
        .build(db, &mut kb)
        .unwrap();
    let base = &su.hc.h + &j;
    (&base - &ka, &base - &kb)
}

// ------------------------------------------------------------------ H3 ROHF

/// Reference H3 doublet ROHF: `(none stage, ewald)` of the staged run.
pub fn h3_rohf_ref(su: &Setup) -> (GammaRohfResult, ScfResult, ScfResult) {
    let r = rohf_on(su, &rohf_cfg(ExxDiv::Ewald, None));
    let none = r.none_stage.clone().expect("staged none stage");
    let ewald = r.scf.clone();
    (r, none, ewald)
}

pub static H3_ROHF_FD: OnceLock<Fd> = OnceLock::new();

/// FD `[none, ewald]` of the H3 ROHF energy (one staged run per geometry,
/// seeded from the reference none-stage MOs; each stage checked to continue
/// the reference occupation state).
pub fn h3_rohf_fd() -> &'static Fd {
    H3_ROHF_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let (r0, none, ewald) = h3_rohf_ref(&setup(h3_cell(), &bs));
        let init = Some(none.mos_alpha.clone());
        fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, FD_H, |c| {
            let su = setup(c.clone(), &bs);
            let r = rohf_on(&su, &rohf_cfg(ExxDiv::Ewald, init.clone()));
            let rn = r.none_stage.expect("none stage");
            let cn = continuity(&rn, r0.nocc, &none.mos_alpha, &su.hc.s);
            let ce = continuity(&r.scf, r0.nocc, &ewald.mos_alpha, &su.hc.s);
            assert!(
                cn < CONT_BAR && ce < CONT_BAR,
                "displaced H3 ROHF left the reference state: {cn:e} / {ce:e}"
            );
            vec![rn.energy, r.scf.energy]
        })
    })
}

// ------------------------------------------------------------------ H3 ROKS

/// Reference H3 doublet ROKS: LDA, PBE (α = 0: exxdiv does not enter),
/// PBE0 none (the none stage of the staged run) and PBE0 ewald.
pub fn h3_roks_refs(su: &Setup) -> Vec<(GammaRoksConfig, ScfResult)> {
    let lda = roks_on(su, &roks_cfg("LDA", ExxDiv::Ewald, None)).scf;
    let pbe = roks_on(su, &roks_cfg("PBE", ExxDiv::Ewald, None)).scf;
    let pbe0 = roks_on(su, &roks_cfg("PBE0", ExxDiv::Ewald, None));
    let pbe0_none = pbe0.none_stage.clone().expect("staged PBE0 none stage");
    vec![
        (roks_cfg("LDA", ExxDiv::Ewald, None), lda),
        (roks_cfg("PBE", ExxDiv::Ewald, None), pbe),
        (roks_cfg("PBE0", ExxDiv::None, None), pbe0_none),
        (roks_cfg("PBE0", ExxDiv::Ewald, None), pbe0.scf),
    ]
}

pub static H3_ROKS_FD: OnceLock<Fd> = OnceLock::new();

/// FD of the four H3 ROKS energies at h = 5e-5 (grid rebuilt per geometry;
/// seeded with the reference MOs; state continuity checked).
pub fn h3_roks_fd() -> &'static Fd {
    H3_ROKS_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let refs = h3_roks_refs(&setup(h3_cell(), &bs));
        let nocc = (2, 1);
        fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, FD_H_KS, |c| {
            let su = setup(c.clone(), &bs);
            let seed = |k: usize| Some(refs[k].1.mos_alpha.clone());
            let lda = roks_on(&su, &roks_cfg("LDA", ExxDiv::Ewald, seed(0)));
            let pbe = roks_on(&su, &roks_cfg("PBE", ExxDiv::Ewald, seed(1)));
            // The staged PBE0 run starts from the none-stage MOs.
            let pbe0 = roks_on(&su, &roks_cfg("PBE0", ExxDiv::Ewald, seed(2)));
            let pbe0_none = pbe0.none_stage.expect("none stage");
            let states = [&lda.scf, &pbe.scf, &pbe0_none, &pbe0.scf];
            for (k, r) in states.iter().enumerate() {
                let cont = continuity(r, nocc, &refs[k].1.mos_alpha, &su.hc.s);
                assert!(
                    cont < CONT_BAR,
                    "ROKS state {k} left the reference: {cont:e}"
                );
            }
            states.iter().map(|r| r.energy).collect()
        })
    })
}

// ---------------------------------------------------- closed-shell identities

/// A closed-shell restricted result as a restricted-open one `(D/2, D/2)`
/// on the same MOs (`nd = N/2`, `no = 0`).
pub fn as_restricted_open(r: &ScfResult) -> ScfResult {
    let mut o = r.clone();
    let half = &r.density_total * 0.5;
    o.spin = Spin::RestrictedOpen;
    o.density_alpha = half.clone();
    o.density_beta = Some(half);
    o.mos_beta = None;
    o
}

// ------------------------------------------------------------ H3 RS-GDF ROHF

/// RS-GDF config at `GDF_OMEGA` with `DEFAULT_RSGDF_LINDEP`,
/// `ExxDiv::None` and budget `AMPLE`.
pub fn gdf_cfg() -> RsGdfConfig {
    RsGdfConfig {
        omega: GDF_OMEGA,
        lindep: DEFAULT_RSGDF_LINDEP,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

/// The bundled `cc-pvdz-ri` auxiliary basis; panics if it is not bundled.
pub fn cc_pvdz_ri() -> BasisSet {
    basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri")
}

pub fn rohf_gdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    gdf: &RsGdf,
    init: Option<Array2<f64>>,
) -> GammaRohfResult {
    let r = gamma_rohf(
        cell,
        prep,
        hc,
        GammaUhfIntegrals::RsGdf(gdf),
        &rohf_cfg(ExxDiv::Ewald, init),
    )
    .expect("gamma_rohf RS-GDF");
    assert!(r.scf.converged);
    r
}

// ------------------------------------------------------ H3 ROHF stress (slow)

/// A strain matrix that is zero except `ε[i][j] = h`.
pub fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

/// Largest element-wise `|a − b|` of two 3×3 matrices; panics on a
/// non-finite difference.
pub fn max_err3(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let d = (a[i][j] - b[i][j]).abs();
            assert!(d.is_finite(), "non-finite stress difference");
            m = m.max(d);
        }
    }
    m
}

/// `max_ij |a_ij − a_ji|` of a 3×3 matrix.
pub fn antisym3(a: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - a[j][i]).abs());
        }
    }
    m
}
