//! Shared fixtures of the OPEN-SHELL (UKS, ROKS) range-separated k-point
//! STRESS (`pbc_kuks_rsh_stress.rs` per commit, `validation_kuks_rsh_stress.rs`
//! weekly), on the dense-AFT and the RS-GDF J/K.
//!
//! One system builder for both routes and all three spin treatments: the
//! RS-GDF settings of the closed-shell RS-GDF RSH file (`rs`: STO-3G, the
//! ET-sp aux, Ewald split `ω_0 = 1`, 1x1x2 mesh, hcore `ω = 0.8`), with the
//! dense-AFT kernels built on the SAME orbital basis, mesh and hcore, so a
//! dense row and an RS-GDF row are the same physical problem and a UKS row
//! and its RKS / ROKS twin share every integral.
#![allow(dead_code)]

use crate::common::*;
use crate::rs::{gdf_cfg, sys, Sys, AMPLE, OMEGA_HCORE};
use ferric_core::FerricError;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kdft::{solve_krks_keep, KRksConfig};
use ferric_pbc::kgrad::KRsGdfGradSource;
use ferric_pbc::kroks::{solve_kroks_injected, solve_kroks_keep, KRoScfResult, KRoksConfig};
use ferric_pbc::kscf::{
    solve_krks_injected, KJkKind, KPointInjection, KPointXc, KRetained, KRetainedJk, KScfConfig,
    KScfResult,
};
use ferric_pbc::kstress::{KStressConfig, KStressJk, KStressMutation};
use ferric_pbc::kuks::{solve_kuks_keep, KUksConfig};
use ferric_pbc::kuscf::{solve_kuks_injected, KPointXcPolarized, KUScfResult};
use ferric_pbc::kuxcstress::{
    kpoint_null_xc_roks_stress_rsh, kpoint_null_xc_uks_stress_rsh, kpoint_roks_stress,
    kpoint_uks_stress, KUStressMutation, KUksStress, KUksStressConfig,
};
use ferric_pbc::kxcgrad::KXcGrid;
use ferric_pbc::kxcstress::{
    kpoint_null_xc_stress_rsh, kpoint_rks_stress, KRksStress, KRksStressConfig,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsh::{resolve_periodic_functional_rsh, RshParams};
use ndarray::Array2;
use num_complex::Complex64 as C64;

pub type Mat3 = [[f64; 3]; 3];

/// libxc name of CAM-B3LYP (the friendly spelling does not resolve).
pub const CAM: &str = "HYB_GGA_XC_CAM_B3LYP";
pub const FUNCTIONALS: [&str; 2] = ["HSE06", CAM];

/// The H3 doublet of the open-shell RSH files (`TRI_A`).
pub const H3D: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.5], [0.3, 0.2, 2.9]];
/// The H2 singlet of the closed-shell RS-GDF RSH file (`rs::TRI2_ATOMS`).
pub const H2S: [[f64; 3]; 2] = [[0.13, 0.25, 0.31], [0.02, 0.27, 1.66]];

/// Dense-AFT precision of every kernel here.
pub const PRECISION: f64 = 1e-12;
/// The RS-GDF metric cut that makes the `ω = 0.33` attenuated fit FD-clean
/// (`rs::LINDEP_FD`).
pub const LINDEP: f64 = 1e-6;
/// Explicit neighbour cutoff of the KS grids (no geometry-dependent default).
pub const CUTOFF: f64 = 14.0;
/// Exactness-anchor bar (the task's `<= 1e-10`).
pub const ANCHOR_TOL: f64 = 1e-10;

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

/// A CAM-type separation whose attenuated piece is live on `TRI_A`.
pub fn cam_type() -> RshParams {
    RshParams {
        omega: 0.33,
        c_sr: 0.19,
        c_lr: 0.65,
    }
}

/// An HSE-type separation (`c_lr = 0`, `ω = 0.11`).
pub fn hse_type() -> RshParams {
    RshParams {
        omega: 0.11,
        c_sr: 0.25,
        c_lr: 0.0,
    }
}

pub fn rsh_of(f: &str) -> RshParams {
    resolve_periodic_functional_rsh(f).unwrap().2.unwrap()
}

/// Hydrogens at `pos` with multiplicity `mult` in `lattice`.
pub fn cell_in(lattice: [[f64; 3]; 3], mult: usize, pos: &[[f64; 3]]) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).unwrap()
}

/// The polarized H3 doublet in `TRI_A`.
pub fn h3d() -> Cell {
    cell_in(TRI_A, 2, &H3D)
}

/// The H2 singlet in `TRI_A`.
pub fn h2s() -> Cell {
    cell_in(TRI_A, 1, &H2S)
}

pub fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

pub fn mdiff(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

pub fn mmax(a: &Mat3) -> f64 {
    a.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// `Σ_k Σ |D_α − D_β|`.
pub fn asym(da: &[Array2<C64>], db: &[Array2<C64>]) -> f64 {
    da.iter()
        .zip(db)
        .map(|(a, b)| (a - b).iter().map(|z| z.norm()).sum::<f64>())
        .sum()
}

/// `(N_α, N_β)` of a cell.
pub fn nocc(cell: &Cell) -> (usize, usize) {
    let ne = cell.mol().nelec() as usize;
    let nb = (ne + 1 - cell.mol().multiplicity) / 2;
    (ne - nb, nb)
}

/// Which J/K.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Dense,
    RsGdf,
}

/// Which spin treatment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spin {
    Rks,
    Uks,
    Roks,
}

/// The J/K a fixture ran on.
#[allow(clippy::large_enum_variant)]
pub enum Jk {
    Dense(KDenseAftEri),
    RsGdf(KRsGdf, KRsGdfConfig),
}

impl Jk {
    pub fn stress_jk<'a>(
        &'a self,
        aux: &'a ferric_integrals::basis_bridge::PreparedBasis,
    ) -> KStressJk<'a> {
        match self {
            Jk::Dense(e) => KStressJk::Dense(e),
            Jk::RsGdf(gdf, cfg) => KStressJk::RsGdf(KRsGdfGradSource { gdf, cfg, aux }),
        }
    }
}

/// A converged SCF of any spin treatment.
pub enum Scf {
    R(KScfResult),
    U(KUScfResult),
    Ro(KRoScfResult),
}

impl Scf {
    pub fn energy(&self) -> f64 {
        match self {
            Scf::R(r) => r.energy,
            Scf::U(r) => r.energy,
            Scf::Ro(r) => r.energy,
        }
    }
    pub fn asym(&self) -> f64 {
        match self {
            Scf::R(_) => 0.0,
            Scf::U(r) => asym(&r.density_alpha, &r.density_beta),
            Scf::Ro(r) => asym(&r.density_alpha, &r.density_beta),
        }
    }
    fn converged(&self) -> bool {
        match self {
            Scf::R(r) => r.converged,
            Scf::U(r) => r.converged,
            Scf::Ro(r) => r.converged,
        }
    }
}

// ------------------------------------------------ zero-XC SCF + stress

struct ZeroXc(usize);
impl KPointXc for ZeroXc {
    fn build(&mut self, dm: &[Array2<C64>]) -> Result<(f64, Vec<Array2<C64>>), FerricError> {
        Ok((
            0.0,
            dm.iter().map(|_| Array2::zeros((self.0, self.0))).collect(),
        ))
    }
}

struct ZeroXcPol(usize);
impl KPointXcPolarized for ZeroXcPol {
    #[allow(clippy::type_complexity)]
    fn build_polarized(
        &mut self,
        da: &[Array2<C64>],
        _db: &[Array2<C64>],
    ) -> Result<(f64, Vec<Array2<C64>>, Vec<Array2<C64>>), FerricError> {
        let z = || da.iter().map(|_| Array2::zeros((self.0, self.0))).collect();
        Ok((0.0, z(), z()))
    }
}

/// A converged zero-XC k SCF: `rsh = Some` is the range-separated J/K
/// (exchange scale `a = 1`, the mixed Madelung constant under ewald), `None`
/// the bare / plain-fit J/K with exchange fraction `a`.
pub struct Zx {
    pub sys: Sys,
    pub jk: Jk,
    pub scf: Scf,
    pub rsh: Option<RshParams>,
    pub a: f64,
    pub exxdiv: ExxDiv,
}

pub fn dense_cfg(rsh: Option<RshParams>) -> KDenseAftConfig {
    KDenseAftConfig {
        precision: PRECISION,
        budget_bytes: Some(AMPLE),
        rsh,
        ..Default::default()
    }
}

pub fn zx(
    cell: Cell,
    (route, spin): (Route, Spin),
    rsh: Option<RshParams>,
    a: f64,
    exxdiv: ExxDiv,
) -> Zx {
    let s = sys(cell);
    let (jk, vm) = match route {
        Route::Dense => {
            let eri = KDenseAftEri::build(
                &s.cell,
                &s.prep,
                &s.mesh,
                &s.hk.s,
                ExxDiv::None,
                &dense_cfg(rsh),
            )
            .unwrap();
            // The K shift of the SCF: the mixed constant of a range-separated
            // kernel (as solve_kuks), the mesh constant of a bare one.
            let vm = match (exxdiv, rsh) {
                (ExxDiv::None, _) => 0.0,
                (ExxDiv::Ewald, Some(_)) => eri.madelung_ewald(),
                (ExxDiv::Ewald, None) => s.mesh.madelung(&s.cell).unwrap(),
            };
            (Jk::Dense(eri), vm)
        }
        Route::RsGdf => {
            let cfg = gdf_cfg(LINDEP, exxdiv);
            let gdf = match rsh {
                Some(r) => KRsGdf::build_rsh(&s.cell, &s.prep, &s.aux, &s.mesh, &s.hk.s, &cfg, r),
                None => KRsGdf::build(&s.cell, &s.prep, &s.aux, &s.mesh, &s.hk.s, &cfg),
            }
            .unwrap();
            (Jk::RsGdf(gdf, cfg), 0.0)
        }
    };
    let builder: Box<dyn ferric_pbc::kscf::KPointJk + '_> = match &jk {
        Jk::Dense(eri) => Box::new(eri.jk_builder_with_madelung(vm)),
        Jk::RsGdf(gdf, _) => Box::new(gdf.jk_builder()),
    };
    let sc = KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 600,
        ..Default::default()
    };
    let inj = KPointInjection {
        s: s.hk.s.clone(),
        h: s.hk.h.clone(),
        vnn: s.hk.enn,
        jk: builder,
    };
    let (na, nb) = nocc(&s.cell);
    let n = s.prep.nbasis();
    let scf = match spin {
        Spin::Rks => {
            Scf::R(solve_krks_injected(&s.cell, &s.mesh, &sc, inj, &mut ZeroXc(n), a).unwrap())
        }
        Spin::Uks => Scf::U(
            solve_kuks_injected(&s.cell, &s.mesh, &sc, inj, na, nb, &mut ZeroXcPol(n), a).unwrap(),
        ),
        Spin::Roks => {
            Scf::Ro(solve_kroks_injected(&s.cell, &s.mesh, &sc, inj, na, nb, None, a).unwrap())
        }
    };
    assert!(
        scf.converged(),
        "zero-XC {route:?} {spin:?} {rsh:?} not converged"
    );
    Zx {
        sys: s,
        jk,
        scf,
        rsh,
        a,
        exxdiv,
    }
}

pub fn hcore() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig::with_omega(OMEGA_HCORE)
}

pub fn base_cfg(exxdiv: ExxDiv, m: Option<KStressMutation>) -> KRksStressConfig {
    KRksStressConfig {
        stress: KStressConfig {
            budget_bytes: Some(AMPLE),
            mutation: m,
            ..Default::default()
        },
        exxdiv: Some(exxdiv),
        ..Default::default()
    }
}

/// The zero-XC stress of a [`Zx`] (any spin) under the HF-part defect `m`
/// and the spin defect `sm` (open shell only).
pub fn zx_stress_m(
    z: &Zx,
    m: Option<KStressMutation>,
    sm: Option<KUStressMutation>,
) -> Result<KRksStress, FerricError> {
    let s = &z.sys;
    let jk = z.jk.stress_jk(&s.aux);
    let hc = hcore();
    let ucfg = KUksStressConfig {
        base: base_cfg(z.exxdiv, m),
        spin_mutation: sm,
    };
    match &z.scf {
        Scf::R(r) => kpoint_null_xc_stress_rsh(
            &s.cell,
            &s.prep,
            &s.mesh,
            &hc,
            &s.hk,
            jk,
            r,
            z.a,
            z.rsh,
            &base_cfg(z.exxdiv, m),
        ),
        Scf::U(r) => kpoint_null_xc_uks_stress_rsh(
            &s.cell, &s.prep, &s.mesh, &hc, &s.hk, jk, r, z.a, z.rsh, &ucfg,
        ),
        Scf::Ro(r) => kpoint_null_xc_roks_stress_rsh(
            &s.cell, &s.prep, &s.mesh, &hc, &s.hk, jk, r, z.a, z.rsh, &ucfg,
        ),
    }
}

pub fn zx_stress(z: &Zx) -> KRksStress {
    zx_stress_m(z, None, None).unwrap()
}

/// Central FD of the zero-XC energy under strain component `(i, j)` (fresh
/// build and SCF per point).
pub fn zx_fd(
    cell: &Cell,
    (i, j, h): (usize, usize, f64),
    (route, spin): (Route, Spin),
    rsh: RshParams,
    exxdiv: ExxDiv,
) -> f64 {
    let e = |t: f64| {
        zx(
            cell.strained(&eps_at(i, j, t)).unwrap(),
            (route, spin),
            Some(rsh),
            1.0,
            exxdiv,
        )
        .scf
        .energy()
    };
    (e(h) - e(-h)) / (2.0 * h)
}

/// The production-wired defects of the open-shell RSH stress ledger: the
/// task's three (β dropped, closed-shell factor, the attenuated piece's
/// strain dropped: the dense `f'` slope, the RS-GDF attenuated fit pass, the
/// `v_M^LR` strain) plus the M-term spin form, the per-spin Pulay Fock and
/// the kernel / fraction defects of the closed-shell ledger. `Route` limits a
/// defect to the J/K where it is wired.
#[allow(clippy::type_complexity)]
pub fn ledger_mutants() -> Vec<(
    &'static str,
    Option<KStressMutation>,
    Option<KUStressMutation>,
    Option<Route>,
)> {
    use KStressMutation as M;
    vec![
        ("ExchDropBeta", Some(M::ExchDropBeta), None, None),
        (
            "ExchClosedShellFactor",
            Some(M::ExchClosedShellFactor),
            None,
            None,
        ),
        ("MTermTotalDensity", Some(M::MTermTotalDensity), None, None),
        ("PulayKSwap", None, Some(KUStressMutation::PulayKSwap), None),
        (
            "RshNoKernelSlope",
            Some(M::RshNoKernelSlope),
            None,
            Some(Route::Dense),
        ),
        (
            "RshKernelOnJ",
            Some(M::RshKernelOnJ),
            None,
            Some(Route::Dense),
        ),
        ("RshSwapFractions", Some(M::RshSwapFractions), None, None),
        (
            "RshPlainMadelungStrain",
            Some(M::RshPlainMadelungStrain),
            None,
            None,
        ),
        (
            "RshNoLrMadelungStrain",
            Some(M::RshNoLrMadelungStrain),
            None,
            None,
        ),
        (
            "RshFitNoAttenuated",
            Some(M::RshFitNoAttenuated),
            None,
            Some(Route::RsGdf),
        ),
        (
            "RshFitNoScale",
            Some(M::RshFitNoScale),
            None,
            Some(Route::RsGdf),
        ),
    ]
}

/// The defects that only reassign spins: invisible on a singlet.
pub const SPIN_ONLY: [&str; 2] = ["MTermTotalDensity", "PulayKSwap"];

// ------------------------------------------------ KS SCF + stress

pub fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

/// A converged KS run (any spin) on either J/K with the functional `f`,
/// kept for its stress.
pub struct Ks {
    pub sys: Sys,
    pub rcfg: KRsGdfConfig,
    pub scf: Scf,
    pub kept: KRetained,
    pub f: String,
    pub exxdiv: ExxDiv,
}

fn tight(sc: &mut KScfConfig) {
    sc.energy_conv = 1e-13;
    sc.grad_conv = 1e-10;
    sc.max_iter = 300;
}

pub fn ks(cell: Cell, (route, spin): (Route, Spin), f: &str, exxdiv: ExxDiv) -> Ks {
    let s = sys(cell);
    let rcfg = gdf_cfg(LINDEP, exxdiv);
    let dense = dense_cfg(None);
    let (jk, aux) = match route {
        Route::Dense => (KJkKind::Dense, None),
        Route::RsGdf => (KJkKind::RsGdf, Some(&s.aux)),
    };
    let (scf, kept) = match spin {
        Spin::Rks => {
            let mut c = KRksConfig::new(&s.cell, exxdiv, f);
            c.krhf.hcore = hcore();
            tight(&mut c.krhf.scf);
            c.krhf.dense = dense;
            c.krhf.jk = jk;
            c.krhf.rsgdf = rcfg;
            c.grid = grid_cfg();
            let (r, k) = solve_krks_keep(&s.cell, &s.prep, aux, &s.mesh, &c)
                .unwrap_or_else(|e| panic!("solve_krks {f}: {e}"));
            (Scf::R(r.scf), k)
        }
        Spin::Uks => {
            let mut c = KUksConfig::new(&s.cell, exxdiv, f);
            c.kuhf.hcore = hcore();
            tight(&mut c.kuhf.scf);
            c.kuhf.dense = dense;
            c.kuhf.jk = jk;
            c.kuhf.rsgdf = rcfg;
            c.grid = grid_cfg();
            let (r, k) = solve_kuks_keep(&s.cell, &s.prep, aux, &s.mesh, &c)
                .unwrap_or_else(|e| panic!("solve_kuks {f}: {e}"));
            (Scf::U(r.scf), k)
        }
        Spin::Roks => {
            let mut c = KRoksConfig::new(&s.cell, exxdiv, f);
            c.rohf.hcore = hcore();
            c.rohf.scf.energy_conv = 1e-13;
            c.rohf.scf.grad_conv = 1e-10;
            c.rohf.dense = dense;
            c.rohf.jk = jk;
            c.rohf.rsgdf = rcfg;
            c.grid = grid_cfg();
            let (r, k) = solve_kroks_keep(&s.cell, &s.prep, aux, &s.mesh, &c)
                .unwrap_or_else(|e| panic!("solve_kroks {f}: {e}"));
            (Scf::Ro(r.scf), k)
        }
    };
    assert!(scf.converged(), "{f} {route:?} {spin:?} not converged");
    Ks {
        sys: s,
        rcfg,
        scf,
        kept,
        f: f.into(),
        exxdiv,
    }
}

pub fn ks_stress_m(
    k: &Ks,
    m: Option<KStressMutation>,
    sm: Option<KUStressMutation>,
) -> Result<KUksStress, FerricError> {
    let s = &k.sys;
    let jk = match &k.kept.jk {
        KRetainedJk::Dense(e) => KStressJk::Dense(e),
        KRetainedJk::RsGdf(g) => KStressJk::RsGdf(KRsGdfGradSource {
            gdf: g,
            cfg: &k.rcfg,
            aux: &s.aux,
        }),
    };
    let mut base = base_cfg(k.exxdiv, m);
    base.grid = KXcGrid::Atom(grid_cfg());
    let hc = hcore();
    match &k.scf {
        Scf::R(r) => kpoint_rks_stress(
            &s.cell, &s.prep, &s.mesh, &hc, &k.kept.hk, jk, r, &k.f, &base,
        ),
        Scf::U(r) => kpoint_uks_stress(
            &s.cell,
            &s.prep,
            &s.mesh,
            &hc,
            &k.kept.hk,
            jk,
            r,
            &k.f,
            &KUksStressConfig {
                base,
                spin_mutation: sm,
            },
        ),
        Scf::Ro(r) => kpoint_roks_stress(
            &s.cell,
            &s.prep,
            &s.mesh,
            &hc,
            &k.kept.hk,
            jk,
            r,
            &k.f,
            &KUksStressConfig {
                base,
                spin_mutation: sm,
            },
        ),
    }
}

pub fn ks_stress(k: &Ks) -> KUksStress {
    ks_stress_m(k, None, None).unwrap_or_else(|e| panic!("{} stress: {e}", k.f))
}

/// Central FD of the KS energy under strain component `(i, j)` (fresh SCF
/// per point).
pub fn ks_fd(
    cell: &Cell,
    (i, j, h): (usize, usize, f64),
    rs: (Route, Spin),
    f: &str,
    exxdiv: ExxDiv,
) -> f64 {
    let e = |t: f64| {
        ks(cell.strained(&eps_at(i, j, t)).unwrap(), rs, f, exxdiv)
            .scf
            .energy()
    };
    (e(h) - e(-h)) / (2.0 * h)
}
