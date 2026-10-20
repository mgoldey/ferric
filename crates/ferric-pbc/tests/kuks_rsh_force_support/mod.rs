//! Shared fixtures of the open-shell range-separated k-point force tests
//! (`pbc_kuks_rsh_force.rs` per commit, `validation_kuks_rsh_force.rs`
//! weekly): the k-UKS SCF on dense-AFT kernels with the functional's range
//! separation, its force, central FD of the converged energy, the explicit
//! supercell and the zero-XC equal-fraction SCF.
#![allow(dead_code)]

use crate::common::*;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kgrad::{KGradConfig, KGradJk, KGradMutation};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::KPointInjection;
use ferric_pbc::kuks::{solve_kuks, KUksConfig, KUksResult};
use ferric_pbc::kuscf::{solve_kuks_injected, KPointXcPolarized, KUScfResult};
use ferric_pbc::kuxcgrad::{
    kpoint_null_xc_uks_gradient_rsh, kpoint_uks_gradient, KUSpinMutation, KUksGradConfig,
    KUksGradient,
};
use ferric_pbc::kxcgrad::{KRksGradConfig, KXcGrid};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsh::{resolve_periodic_functional_rsh, RshParams};
use ndarray::Array2;
use num_complex::Complex64;

/// libxc name of CAM-B3LYP (the friendly spelling does not resolve).
pub const CAM: &str = "HYB_GGA_XC_CAM_B3LYP";
pub const FUNCTIONALS: [&str; 2] = ["HSE06", CAM];

pub const OMEGA: f64 = 0.8;
pub const AMPLE: usize = 1 << 31;
/// Explicit neighbour cutoff of every grid here (a default that depends on
/// the geometry would put steps into the FD energies).
pub const CUTOFF: f64 = 14.0;
pub const PRECISION: f64 = 1e-12;
/// Anchor bar (singlet, equal fractions, supercell, Gamma): the measured
/// values are <= 2.3e-13 (singlet, per part), 1.7e-16 (equal fractions),
/// 7.2e-15 (supercell), 8.5e-14 (Gamma); >= 1e4 x below the smallest
/// asserted mutant (1e-5).
pub const ANCHOR_TOL: f64 = 1e-9;

/// FD bar (as the closed-shell RSH force, `krsh_force_support::FD_BAR`).
pub const FD_BAR: f64 = 1e-7;

/// The H3 doublet / H4 triplet cells of the open-shell RSH energy (#480,
/// `oshell_rsh_support`), triclinic lattice `TRI_A`.
pub const H3D: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.5], [0.3, 0.2, 2.9]];
pub const H4T: [[f64; 3]; 4] = [
    [0.3, 0.2, 0.1],
    [0.3, 0.2, 1.5],
    [1.6, 1.1, 2.9],
    [1.6, 1.1, 4.1],
];
/// H2 singlet in the a = 4 cube (the closed-shell RSH force fixture, #486).
pub const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

/// Which geometry family a fixture belongs to (how a displaced cell is built).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Triclinic `TRI_A` with the given multiplicity.
    Tri(usize),
    /// The a = 4 cube, singlet.
    Cube,
}

pub fn cell_of(kind: Kind, pos: &[[f64; 3]]) -> Cell {
    let mut mol = hydrogens(pos);
    match kind {
        Kind::Tri(m) => {
            mol.multiplicity = m;
            Cell::new(mol, TRI_A).unwrap()
        }
        Kind::Cube => Cell::new(mol, cubic(4.0)).unwrap(),
    }
}

pub fn rsh_of(f: &str) -> RshParams {
    resolve_periodic_functional_rsh(f).unwrap().2.unwrap()
}

pub fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

pub fn dense_cfg(rsh: Option<RshParams>) -> KDenseAftConfig {
    KDenseAftConfig {
        precision: PRECISION,
        budget_bytes: Some(AMPLE),
        rsh,
        ..Default::default()
    }
}

pub fn kuks_cfg(cell: &Cell, f: &str, exxdiv: ExxDiv) -> KUksConfig {
    let mut cfg = KUksConfig::new(cell, exxdiv, f);
    cfg.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.kuhf.scf.energy_conv = 1e-13;
    cfg.kuhf.scf.grad_conv = 1e-10;
    cfg.kuhf.scf.max_iter = 300;
    cfg.kuhf.dense = dense_cfg(None);
    cfg.grid = grid_cfg();
    cfg
}

pub fn uks_scf(cell: &Cell, n: [usize; 3], f: &str, exxdiv: ExxDiv) -> KUksResult {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let r = solve_kuks(cell, &prep, None, &mesh, &kuks_cfg(cell, f, exxdiv))
        .unwrap_or_else(|e| panic!("solve_kuks {f} {n:?} {exxdiv:?}: {e}"));
    assert!(r.scf.converged, "{f} {n:?} {exxdiv:?} not converged");
    r
}

/// A converged k-UKS RSH state with the kernels its force needs.
pub struct Fx {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub eri: KDenseAftEri,
    pub res: KUksResult,
    pub f: String,
    pub exxdiv: ExxDiv,
}

pub fn fixture_cell(cell: Cell, n: [usize; 3], f: &str, exxdiv: ExxDiv) -> Fx {
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    let res = uks_scf(&cell, n, f, exxdiv);
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    // The kernels exactly as solve_kuks built them (the functional's rsh).
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::None,
        &dense_cfg(Some(rsh_of(f))),
    )
    .unwrap();
    Fx {
        cell,
        prep,
        mesh,
        hk,
        eri,
        res,
        f: f.into(),
        exxdiv,
    }
}

pub fn fixture(kind: Kind, pos: &[[f64; 3]], n: [usize; 3], f: &str, exxdiv: ExxDiv) -> Fx {
    fixture_cell(cell_of(kind, pos), n, f, exxdiv)
}

pub fn gcfg(
    exxdiv: ExxDiv,
    m: Option<KGradMutation>,
    sm: Option<KUSpinMutation>,
) -> KUksGradConfig {
    KUksGradConfig {
        base: KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                mutation: m,
                ..Default::default()
            },
            grid: KXcGrid::Atom(grid_cfg()),
            exxdiv: Some(exxdiv),
            ..Default::default()
        },
        spin_mutation: sm,
    }
}

pub fn try_force_on(
    fx: &Fx,
    jk: KGradJk<'_>,
    f: &str,
    cfg: &KUksGradConfig,
) -> Result<KUksGradient, FerricError> {
    kpoint_uks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        jk,
        &fx.res.scf,
        f,
        cfg,
    )
}

pub fn force_m(
    fx: &Fx,
    m: Option<KGradMutation>,
    sm: Option<KUSpinMutation>,
) -> Result<KUksGradient, FerricError> {
    try_force_on(fx, KGradJk::Dense(&fx.eri), &fx.f, &gcfg(fx.exxdiv, m, sm))
}

pub fn force(fx: &Fx) -> KUksGradient {
    force_m(fx, None, None).unwrap_or_else(|e| panic!("kpoint_uks_gradient {}: {e}", fx.f))
}

pub fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

/// Central FD of the converged k-UKS energy (fresh SCF per displacement).
#[allow(clippy::too_many_arguments)]
pub fn fd(
    kind: Kind,
    pos: &[[f64; 3]],
    a: usize,
    x: usize,
    h: f64,
    n: [usize; 3],
    f: &str,
    e: ExxDiv,
) -> f64 {
    let en = |p: &[[f64; 3]]| uks_scf(&cell_of(kind, p), n, f, e).scf.energy;
    (en(&moved(pos, a, x, h)) - en(&moved(pos, a, x, -h))) / (2.0 * h)
}

pub fn diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

pub fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// `Σ_k Σ |D_α − D_β|`: proof that a fixture is polarized.
pub fn spin_asymmetry(scf: &KUScfResult) -> f64 {
    scf.density_alpha
        .iter()
        .zip(&scf.density_beta)
        .map(|(a, b)| (a - b).iter().map(|z| z.norm()).sum::<f64>())
        .sum()
}

/// The explicit `n`-cell supercell of a triclinic cell `pos` (multiplicity
/// `mult` of the giant determinant).
pub fn supercell(pos: &[[f64; 3]], n: [usize; 3], mult: usize) -> Cell {
    let a = TRI_A;
    let mut atoms = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                let t: Vec<f64> = (0..3)
                    .map(|d| m0 as f64 * a[0][d] + m1 as f64 * a[1][d] + m2 as f64 * a[2][d])
                    .collect();
                for p in pos {
                    atoms.push([p[0] + t[0], p[1] + t[1], p[2] + t[2]]);
                }
            }
        }
    }
    let mut lat = a;
    for (i, row) in lat.iter_mut().enumerate() {
        row.iter_mut().for_each(|v| *v *= n[i] as f64);
    }
    let mut mol = hydrogens(&atoms);
    mol.multiplicity = mult;
    Cell::new(mol, lat).unwrap()
}

/// Every (atom, axis) component of an `n`-atom cell.
pub fn all_comps(n: usize) -> Vec<(usize, usize)> {
    (0..n).flat_map(|a| (0..3).map(move |x| (a, x))).collect()
}

// ------------------------------------------------ zero-XC (equal fractions)

pub struct ZeroXcPol(pub usize);
impl KPointXcPolarized for ZeroXcPol {
    #[allow(clippy::type_complexity)]
    fn build_polarized(
        &mut self,
        da: &[Array2<Complex64>],
        _db: &[Array2<Complex64>],
    ) -> Result<(f64, Vec<Array2<Complex64>>, Vec<Array2<Complex64>>), FerricError> {
        let z = || da.iter().map(|_| Array2::zeros((self.0, self.0))).collect();
        Ok((0.0, z(), z()))
    }
}

/// A zero-XC k-UKS SCF on dense kernels with range separation `rsh` (or bare)
/// and exchange scale `a`.
pub struct Zx {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub eri: KDenseAftEri,
    pub scf: KUScfResult,
    pub exxdiv: ExxDiv,
    pub rsh: Option<RshParams>,
    pub a: f64,
}

pub fn zero_xc(
    pos: &[[f64; 3]],
    mult: usize,
    n: [usize; 3],
    exxdiv: ExxDiv,
    rsh: Option<RshParams>,
    a: f64,
) -> Zx {
    let cell = cell_of(Kind::Tri(mult), pos);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let eri =
        KDenseAftEri::build(&cell, &prep, &mesh, &hk.s, ExxDiv::None, &dense_cfg(rsh)).unwrap();
    // The K shift the SCF applies: the mixed constant of a range-separated
    // kernel (as solve_kuks), the mesh constant of a bare one.
    let vm = match (exxdiv, rsh) {
        (ExxDiv::None, _) => 0.0,
        (ExxDiv::Ewald, Some(_)) => eri.madelung_ewald(),
        (ExxDiv::Ewald, None) => mesh.madelung(&cell).unwrap(),
    };
    let mut sc = ferric_pbc::kscf::KScfConfig::default();
    sc.energy_conv = 1e-13;
    sc.grad_conv = 1e-10;
    sc.max_iter = 300;
    let scf = {
        let inj = KPointInjection {
            s: hk.s.clone(),
            h: hk.h.clone(),
            vnn: hk.enn,
            jk: Box::new(eri.jk_builder_with_madelung(vm)),
        };
        let ne = cell.mol().nelec() as usize;
        let nb = (ne + 1 - mult) / 2;
        let na = ne - nb;
        let mut z = ZeroXcPol(prep.nbasis());
        solve_kuks_injected(&cell, &mesh, &sc, inj, na, nb, &mut z, a).unwrap()
    };
    assert!(scf.converged);
    Zx {
        cell,
        prep,
        mesh,
        hk,
        eri,
        scf,
        exxdiv,
        rsh,
        a,
    }
}

pub fn null_force(z: &Zx, m: Option<KGradMutation>) -> KUksGradient {
    kpoint_null_xc_uks_gradient_rsh(
        &z.cell,
        &z.prep,
        &z.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &z.hk,
        KGradJk::Dense(&z.eri),
        &z.scf,
        z.a,
        z.rsh,
        &gcfg(z.exxdiv, m, None),
    )
    .unwrap()
}

/// The production-wired defects the ledger exercises, with names.
pub fn ledger_mutants() -> Vec<(&'static str, Option<KGradMutation>, Option<KUSpinMutation>)> {
    use KGradMutation as M;
    vec![
        ("ExchBetaUsesAlpha", Some(M::ExchBetaUsesAlpha), None),
        ("ExchDropBeta", Some(M::ExchDropBeta), None),
        ("MTermTotalDensity", Some(M::MTermTotalDensity), None),
        ("PulayKSwap", None, Some(KUSpinMutation::PulayKSwap)),
        ("RshKernelOnJ", Some(M::RshKernelOnJ), None),
        ("RshPlainMadelung", Some(M::RshPlainMadelung), None),
        ("RshSwapFractions", Some(M::RshSwapFractions), None),
        ("RshNoKernelFactor", Some(M::RshNoKernelFactor), None),
        ("RshNoErfPiece", Some(M::RshNoErfPiece), None),
        ("RshOmegaScaled", Some(M::RshOmegaScaled), None),
    ]
}

/// Largest `|c_lr - c_sr| e^{-K^2/4w^2}` over the exchange K set of a mesh
/// (`K = G + q != 0`): how much of the attenuated (erf) piece the walks see.
pub fn max_erf_weight(cell: &Cell, mesh: &KPointMesh, r: RshParams) -> f64 {
    let mut w = 0.0_f64;
    for iq in 0..mesh.nk() {
        let q = mesh.q_class(iq).0;
        for g in cell.gvectors(3.0).unwrap() {
            let k = [g[0] + q[0], g[1] + q[1], g[2] + q[2]];
            let k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
            if k2 > 1e-20 {
                w = w.max((r.c_lr - r.c_sr).abs() * (-k2 / (4.0 * r.omega * r.omega)).exp());
            }
        }
    }
    w
}

// ------------------------------------------------ closed-shell twin, Gamma

/// The closed-shell k-RKS RSH force (#486) on the same cell, mesh, grid and
/// kernels as the singlet fixture `fx`, from an independent closed-shell SCF.
pub fn rks_twin(fx: &Fx) -> ferric_pbc::kxcgrad::KRksGradient {
    let mut cfg = ferric_pbc::kdft::KRksConfig::new(&fx.cell, fx.exxdiv, &fx.f);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 300;
    cfg.krhf.dense = dense_cfg(None);
    cfg.grid = grid_cfg();
    let r = ferric_pbc::kdft::solve_krks(&fx.cell, &fx.prep, None, &fx.mesh, &cfg).unwrap();
    assert!(r.scf.converged);
    ferric_pbc::kxcgrad::kpoint_rks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KGradJk::Dense(&fx.eri),
        &r.scf,
        &fx.f,
        &KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            grid: KXcGrid::Atom(grid_cfg()),
            exxdiv: Some(fx.exxdiv),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Singlet k-UKS vs k-RKS per part on the H2 singlet (1x1x3): `(label, |UKS -
/// RKS|)` rows, plus the UKS fixture and force for the blindness checks.
pub fn singlet_rows(f: &str) -> (Vec<(&'static str, f64)>, Fx, KUksGradient) {
    let fx = fixture(Kind::Cube, &H2, [1, 1, 3], f, ExxDiv::Ewald);
    assert!(spin_asymmetry(&fx.res.scf) < 1e-8);
    let u = force(&fx);
    let r = rks_twin(&fx);
    let (p, q) = (&u.hf.parts, &r.hf.parts);
    assert!(max_abs(&p.exchange) > 1e-3, "exchange exercised");
    let rows = vec![
        ("exchange", diff(&p.exchange, &q.exchange)),
        ("coulomb", diff(&p.coulomb, &q.coulomb)),
        ("overlap", diff(&p.overlap, &q.overlap)),
        ("xc_ao", diff(&u.xc_ao, &r.xc_ao)),
        ("total", diff(&u.grad, &r.grad)),
    ];
    (rows, fx, u)
}

/// The defects that only reassign spins: invisible on a singlet.
pub const SPIN_ONLY: [&str; 3] = ["ExchBetaUsesAlpha", "MTermTotalDensity", "PulayKSwap"];

/// The SCF of the Gamma UKS RSH force on the H3 doublet: the Gamma SCF
/// (`gamma_uks_rsh`, dense J and exchange) and everything its routed force
/// needs.
pub struct GammaFx {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub hcfg: PeriodicHcoreConfig,
    pub ex: ferric_pbc::gamma_rsh::GammaRshExchange,
    pub gc: ferric_pbc::dft::GammaUksConfig,
    pub scf: ferric_scf::result::ScfResult,
}

pub fn gamma_fixture(f: &str, exxdiv: ExxDiv) -> GammaFx {
    use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
    let cell = cell_of(Kind::Tri(2), &H3D);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hc = ferric_pbc::hcore::periodic_hcore(&cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        exxdiv,
        PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let ex = ferric_pbc::gamma_rsh::GammaRshExchange::build_dense(
        &cell,
        &prep,
        &hc.s,
        exxdiv,
        &dense_cfg(None),
        rsh_of(f),
    )
    .unwrap();
    let mut gc = ferric_pbc::dft::GammaUksConfig::new(f);
    gc.grid = grid_cfg();
    gc.exxdiv = exxdiv;
    gc.scf.energy_conv = 1e-13;
    gc.scf.density_conv = 1e-10;
    gc.scf.max_iter = 300;
    let r = ferric_pbc::dft::gamma_uks_rsh(
        &cell,
        &prep,
        &hc,
        ferric_pbc::uhf::GammaUhfIntegrals::DenseAft(&eri),
        &ex,
        &gc,
    )
    .unwrap();
    assert!(r.scf.converged);
    GammaFx {
        cell,
        prep,
        hcfg,
        ex,
        gc,
        scf: r.scf,
    }
}

pub fn kgrad_cfg() -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

pub fn gamma_force(g: &GammaFx) -> Result<KUksGradient, FerricError> {
    ferric_pbc::gamma_rsh::gamma_uks_rsh_gradient(
        &g.cell,
        &g.prep,
        &g.hcfg,
        &g.scf,
        &g.gc,
        &g.ex,
        &kgrad_cfg(),
    )
}

/// Gamma vs the 1x1x1 k-UKS force on the H3 doublet: `(|diff|, Fock guard,
/// max |F|)`.
pub fn gamma_vs_one_point(f: &str, exxdiv: ExxDiv) -> (f64, f64, f64) {
    let g = gamma_force(&gamma_fixture(f, exxdiv)).unwrap();
    let k = force(&fixture(Kind::Tri(2), &H3D, [1, 1, 1], f, exxdiv));
    (diff(&g.grad, &k.grad), g.hf.fock_mismatch, max_abs(&g.grad))
}

/// 1x1x2 mesh force of the H3 doublet vs one copy in the explicit 2-cell
/// supercell (multiplicity 3, one k-point).
pub fn mesh_vs_supercell(f: &str) -> f64 {
    let k = force(&fixture(Kind::Tri(2), &H3D, [1, 1, 2], f, ExxDiv::Ewald));
    let sc = force(&fixture_cell(
        supercell(&H3D, [1, 1, 2], 3),
        [1, 1, 1],
        f,
        ExxDiv::Ewald,
    ));
    diff(&k.grad, &sc.grad.slice(ndarray::s![0..3, ..]).to_owned())
}

/// The equal-fraction anchor (`pbc_kuks_rsh_force.rs` (3)) under `exxdiv`.
pub fn equal_fraction_case(exxdiv: ExxDiv) {
    let c = 0.4;
    let eq = RshParams {
        omega: 0.33,
        c_sr: c,
        c_lr: c,
    };
    let zr = zero_xc(&H3D, 2, [1, 1, 2], exxdiv, Some(eq), 1.0);
    let zg = zero_xc(&H3D, 2, [1, 1, 2], exxdiv, None, c);
    let de = (zr.scf.energy - zg.scf.energy).abs();
    let gr = null_force(&zr, None);
    let gg = null_force(&zg, None);
    let d = diff(&gr.grad, &gg.grad);
    eprintln!(
        "{exxdiv:?}: equal fractions vs global c={c}: |dE| {de:.1e} force {d:.2e} \
         K part {:.2e}",
        diff(&gr.hf.parts.exchange, &gg.hf.parts.exchange)
    );
    assert!(spin_asymmetry(&zr.scf) > 0.1, "polarized");
    assert!(de < 1e-11, "{de:e}");
    assert!(d < ANCHOR_TOL, "{exxdiv:?}: {d:e}");
    assert!(max_abs(&gg.hf.parts.exchange) > 1e-4);
    for m in [
        KGradMutation::RshNoKernelFactor,
        KGradMutation::RshKernelOnJ,
        KGradMutation::RshPlainMadelung,
        KGradMutation::MTermTotalDensity,
        KGradMutation::ExchBetaUsesAlpha,
    ] {
        let miss = diff(&null_force(&zr, Some(m)).grad, &gg.grad);
        eprintln!("  {exxdiv:?} equal-fraction anchor vs {m:?}: miss {miss:.2e}");
        let blind = exxdiv == ExxDiv::None
            && matches!(
                m,
                KGradMutation::RshPlainMadelung | KGradMutation::MTermTotalDensity
            );
        if blind {
            assert!(miss < ANCHOR_TOL, "blind by construction under exxdiv none");
        } else {
            assert!(miss > 1e-3, "{m:?} {exxdiv:?}: {miss:e}");
        }
    }
}
