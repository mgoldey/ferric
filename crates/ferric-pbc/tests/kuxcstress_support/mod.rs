//! Shared fixtures and helpers of `pbc_kuxcstress.rs` (per commit) and
//! `validation_pbc_kuxcstress.rs` (weekly): the H2 / H3 k-UKS fixtures, the
//! stress drivers, the zero-XC diagnostic entry, the Gamma UKS reference and
//! the anchor bodies.
#![allow(dead_code, unused_imports)]

use crate::common::*;
pub use ferric_core::basis::{BasisSet, Shell};
pub use ferric_pbc::dense_aft::ExxDiv;
pub use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
pub use ferric_pbc::dft::{gamma_uks, GammaUksConfig, PeriodicGridConfig};
pub use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
pub use ferric_pbc::hcore::periodic_hcore;
pub use ferric_pbc::hcore::PeriodicHcoreConfig;
pub use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
pub use ferric_pbc::kdft::{solve_krks, KRksConfig};
pub use ferric_pbc::kgrad::KRsGdfGradSource;
pub use ferric_pbc::kpts::KPointMesh;
pub use ferric_pbc::kscf::{KJkKind, KPointInjection, KPointJk};
pub use ferric_pbc::kstress::{
    kpoint_uhf_stress, KStressConfig, KStressJk, KStressMutation, KStressParts,
};
pub use ferric_pbc::kuks::{solve_kuks, KUksConfig, KUksResult};
pub use ferric_pbc::kuscf::{solve_kuks_injected, KPointXcPolarized, KUScfResult};
pub use ferric_pbc::kuxcstress::{
    kpoint_null_xc_uks_stress, kpoint_uks_stress, KUStressMutation, KUksStress, KUksStressConfig,
    UKS_COMMUTATOR_TOL,
};
pub use ferric_pbc::kxcgrad::KXcGrid;
pub use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStressConfig, KXcStressMutation};
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
pub use ferric_pbc::rsgdf::RsGdfConfig;
pub use ferric_pbc::stress::{gamma_uks_stress, GammaStress, GammaStressConfig, Mat3};
pub use ferric_pbc::uhf::GammaUhfIntegrals;
pub use ndarray::Array2;
pub use num_complex::Complex64;
pub use std::collections::HashMap;

pub const OMEGA: f64 = 0.8;
pub const AMPLE: usize = 1 << 31;
pub const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
/// H3 doublet (na = 2, nb = 1): a genuinely spin-polarised cell, Da != Db.
pub const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];
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
/// Explicit neighbour cutoff shared by every energy and stress here (the default
/// depends on the cell and puts a one-step discontinuity in the energy).
pub const CUTOFF: f64 = 14.0;
pub const ANCHOR_PRECISION: f64 = 1e-12;
/// (energy_conv, grad_conv) of the deliberately loose UKS SCF of the gate test.
pub const LOOSE_UKS: (f64, f64) = (1e-5, 1e-2);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Jk {
    Dense,
    Gdf,
}

#[derive(Clone, Copy)]
pub struct Mesh {
    pub n: [usize; 3],
    pub mp: bool,
}

pub const G111: Mesh = Mesh {
    n: [1, 1, 1],
    mp: false,
};
pub const G113: Mesh = Mesh {
    n: [1, 1, 3],
    mp: false,
};

pub fn mesh_of(cell: &Cell, m: Mesh) -> KPointMesh {
    if m.mp {
        KPointMesh::monkhorst_pack(cell, m.n).unwrap()
    } else {
        KPointMesh::gamma_centred(cell, m.n).unwrap()
    }
}

pub fn et_sp_aux() -> BasisSet {
    let mut shells = Vec::new();
    for a in [4.7, 1.9, 0.75, 0.3] {
        shells.push(Shell {
            l: 0,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    for a in [1.25, 0.5] {
        shells.push(Shell {
            l: 1,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-sp-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

pub fn kgdf_cfg() -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    }
}

thread_local! {
    /// (energy_conv, grad_conv) of the k SCF on THIS test thread; only the
    /// stationarity-gate test changes the defaults.
    pub static SETTINGS: std::cell::Cell<(f64, f64)> = const { std::cell::Cell::new((1e-14, 1e-12)) };
}

pub fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

pub fn cell_of(pos: &[[f64; 3]]) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = if pos.len() % 2 == 1 { 2 } else { 1 };
    Cell::new(mol, cubic(4.0)).expect("cell")
}

pub fn kuks_cfg(cell: &Cell, functional: &str, jk: Jk, exxdiv: ExxDiv) -> KUksConfig {
    let mut cfg = KUksConfig::new(cell, exxdiv, functional);
    cfg.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    let (econv, gconv) = SETTINGS.with(|s| s.get());
    cfg.kuhf.scf.energy_conv = econv;
    cfg.kuhf.scf.grad_conv = gconv;
    cfg.kuhf.scf.max_iter = 300;
    if jk == Jk::Gdf {
        cfg.kuhf.jk = KJkKind::RsGdf;
        cfg.kuhf.rsgdf = kgdf_cfg();
    }
    cfg.grid = grid_cfg();
    cfg
}

/// Fresh UKS SCF on `cell`.
pub fn scf_on(cell: &Cell, m: Mesh, functional: &str, jk: Jk, exxdiv: ExxDiv) -> KUksResult {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = mesh_of(cell, m);
    let cfg = kuks_cfg(cell, functional, jk, exxdiv);
    let aux = (jk == Jk::Gdf).then(|| prep_for(cell, &et_sp_aux()));
    let r = solve_kuks(cell, &prep, aux.as_ref(), &mesh, &cfg)
        .unwrap_or_else(|e| panic!("solve_kuks {functional}: {e}"));
    assert!(r.scf.converged);
    r
}

pub fn energy_on(cell: &Cell, m: Mesh, f: &str, jk: Jk, exxdiv: ExxDiv) -> f64 {
    scf_on(cell, m, f, jk, exxdiv).scf.energy
}

pub struct Fx {
    pub cell: Cell,
    pub prep: ferric_integrals::basis_bridge::PreparedBasis,
    pub aux: ferric_integrals::basis_bridge::PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub res: KUksResult,
    pub jk: Jk,
    pub exxdiv: ExxDiv,
    pub functional: String,
}

pub fn fixture_cell(cell: Cell, m: Mesh, functional: &str, jk: Jk, exxdiv: ExxDiv) -> Fx {
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = mesh_of(&cell, m);
    let res = scf_on(&cell, m, functional, jk, exxdiv);
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let aux = prep_for(&cell, &et_sp_aux());
    Fx {
        cell,
        prep,
        aux,
        mesh,
        hk,
        res,
        jk,
        exxdiv,
        functional: functional.to_string(),
    }
}

pub fn fixture(pos: &[[f64; 3]], m: Mesh, functional: &str, jk: Jk, exxdiv: ExxDiv) -> Fx {
    fixture_cell(cell_of(pos), m, functional, jk, exxdiv)
}

pub fn scfg(
    exxdiv: Option<ExxDiv>,
    xm: Option<KXcStressMutation>,
    sm: Option<KUStressMutation>,
    hm: Option<KStressMutation>,
) -> KUksStressConfig {
    KUksStressConfig {
        base: KRksStressConfig {
            stress: KStressConfig {
                budget_bytes: Some(AMPLE),
                mutation: hm,
                ..Default::default()
            },
            grid: KXcGrid::Atom(grid_cfg()),
            exxdiv,
            mutation: xm,
            ..Default::default()
        },
        spin_mutation: sm,
    }
}

pub fn stress_with(
    fx: &Fx,
    functional: &str,
    scf: &KUScfResult,
    cfg: &KUksStressConfig,
) -> Result<KUksStress, ferric_core::FerricError> {
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    match fx.jk {
        Jk::Dense => {
            let eri = KDenseAftEri::build(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &fx.hk.s,
                fx.exxdiv,
                &Default::default(),
            )
            .unwrap();
            kpoint_uks_stress(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &hcfg,
                &fx.hk,
                KStressJk::Dense(&eri),
                scf,
                functional,
                cfg,
            )
        }
        Jk::Gdf => {
            let kc = kgdf_cfg();
            let gdf = KRsGdf::build(&fx.cell, &fx.prep, &fx.aux, &fx.mesh, &fx.hk.s, &kc)
                .unwrap()
                .with_exxdiv(fx.exxdiv);
            kpoint_uks_stress(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &hcfg,
                &fx.hk,
                KStressJk::RsGdf(KRsGdfGradSource {
                    gdf: &gdf,
                    cfg: &kc,
                    aux: &fx.aux,
                }),
                scf,
                functional,
                cfg,
            )
        }
    }
}

pub fn stress(fx: &Fx) -> KUksStress {
    stress_with(
        fx,
        &fx.functional,
        &fx.res.scf,
        &scfg(Some(fx.exxdiv), None, None, None),
    )
    .expect("kpoint_uks_stress")
}

pub fn stress_mut(
    fx: &Fx,
    xm: Option<KXcStressMutation>,
    sm: Option<KUStressMutation>,
    hm: Option<KStressMutation>,
) -> KUksStress {
    stress_with(
        fx,
        &fx.functional,
        &fx.res.scf,
        &scfg(Some(fx.exxdiv), xm, sm, hm),
    )
    .expect("kpoint_uks_stress")
}

// ------------------------------------------------------------ small helpers

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

pub fn mscale(a: &Mat3, f: f64) -> Mat3 {
    let mut o = *a;
    for r in o.iter_mut() {
        for x in r.iter_mut() {
            *x *= f;
        }
    }
    o
}

pub fn madd(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = *a;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] += b[i][j];
        }
    }
    o
}

pub fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

pub fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    cell.strained(&eps_at(i, j, h)).expect("strained cell")
}

/// Central FD of the UKS SCF energy for the listed components at step `h`.
pub fn fd_components(
    cell: &Cell,
    m: Mesh,
    f: &str,
    jk: Jk,
    exxdiv: ExxDiv,
    h: f64,
    comps: &[(usize, usize)],
) -> Vec<f64> {
    comps
        .iter()
        .map(|&(i, j)| {
            let ep = energy_on(&strained(cell, i, j, h), m, f, jk, exxdiv);
            let em = energy_on(&strained(cell, i, j, -h), m, f, jk, exxdiv);
            (ep - em) / (2.0 * h)
        })
        .collect()
}

/// Every labelled HF part of a `KStressParts`.
pub fn hf_parts(p: &KStressParts) -> Vec<(&'static str, Mat3)> {
    vec![
        ("overlap_pulay", p.overlap_pulay),
        ("overlap_c0", p.overlap_c0),
        ("overlap_madelung", p.overlap_madelung),
        ("kinetic", p.kinetic),
        ("vsr", p.vsr),
        ("ecp", p.ecp),
        ("vlr", p.vlr),
        ("c0_volume", p.c0_volume),
        ("coulomb", p.coulomb),
        ("exchange", p.exchange),
        ("madelung", p.madelung),
        ("nn_sr", p.nn_sr),
        ("nn_lr", p.nn_lr),
        ("nn_background", p.nn_background),
        ("fit_j3_sr", p.fit_j3_sr),
        ("fit_j3_lr", p.fit_j3_lr),
        ("fit_j2_sr", p.fit_j2_sr),
        ("fit_j2_lr", p.fit_j2_lr),
        ("fit_g0_overlap", p.fit_g0_overlap),
        ("fit_j3_g0_volume", p.fit_j3_g0_volume),
        ("fit_j2_g0_volume", p.fit_j2_g0_volume),
    ]
}

pub fn kcfg() -> KStressConfig {
    KStressConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

// ----------------------------------------------- zero-XC fixed-density SCF

pub struct ZeroXcPol(usize);
impl KPointXcPolarized for ZeroXcPol {
    #[allow(clippy::type_complexity)]
    fn build_polarized(
        &mut self,
        da: &[Array2<Complex64>],
        _db: &[Array2<Complex64>],
    ) -> Result<(f64, Vec<Array2<Complex64>>, Vec<Array2<Complex64>>), ferric_core::FerricError>
    {
        let z = || da.iter().map(|_| Array2::zeros((self.0, self.0))).collect();
        Ok((0.0, z(), z()))
    }
}

pub struct Zx {
    pub cell: Cell,
    pub prep: ferric_integrals::basis_bridge::PreparedBasis,
    pub aux: ferric_integrals::basis_bridge::PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub scf: KUScfResult,
    pub jk: Jk,
    pub exxdiv: ExxDiv,
}

pub fn dense_cfg() -> KDenseAftConfig {
    KDenseAftConfig {
        precision: ANCHOR_PRECISION,
        ..Default::default()
    }
}

/// Hartree + `a` exact-exchange k UKS with a zero XC (a = 1 is k-point UHF).
pub fn zero_xc(pos: &[[f64; 3]], m: Mesh, jk: Jk, exxdiv: ExxDiv, a: f64) -> Zx {
    let cell = cell_of(pos);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = prep_for(&cell, &et_sp_aux());
    let mesh = mesh_of(&cell, m);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).unwrap();
    let mut sc = ferric_pbc::kscf::KScfConfig::default();
    sc.energy_conv = 1e-13;
    sc.grad_conv = 1e-10;
    sc.max_iter = 300;
    let vm = match exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => mesh.madelung(&cell).unwrap(),
    };
    let eri;
    let gdf;
    let jkb: Box<dyn KPointJk + '_> = match jk {
        Jk::Dense => {
            eri = KDenseAftEri::build(&cell, &prep, &mesh, &hk.s, ExxDiv::None, &dense_cfg())
                .unwrap();
            Box::new(eri.jk_builder_with_madelung(vm))
        }
        Jk::Gdf => {
            gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kgdf_cfg()).unwrap();
            Box::new(gdf.jk_builder_with_madelung(vm))
        }
    };
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: jkb,
    };
    let ne = cell.mol().nelec() as usize;
    let (na, nb) = (ne / 2 + ne % 2, ne / 2);
    let mut z = ZeroXcPol(prep.nbasis());
    let scf = solve_kuks_injected(&cell, &mesh, &sc, inj, na, nb, &mut z, a).unwrap();
    assert!(scf.converged);
    Zx {
        cell,
        prep,
        aux,
        mesh,
        hk,
        scf,
        jk,
        exxdiv,
    }
}

pub fn with_jk<R>(z: &Zx, f: impl FnOnce(KStressJk<'_>) -> R) -> R {
    match z.jk {
        Jk::Dense => {
            let eri =
                KDenseAftEri::build(&z.cell, &z.prep, &z.mesh, &z.hk.s, z.exxdiv, &dense_cfg())
                    .unwrap();
            f(KStressJk::Dense(&eri))
        }
        Jk::Gdf => {
            let kc = kgdf_cfg();
            let gdf = KRsGdf::build(&z.cell, &z.prep, &z.aux, &z.mesh, &z.hk.s, &kc)
                .unwrap()
                .with_exxdiv(z.exxdiv);
            f(KStressJk::RsGdf(KRsGdfGradSource {
                gdf: &gdf,
                cfg: &kc,
                aux: &z.aux,
            }))
        }
    }
}

pub fn null_stress(
    z: &Zx,
    a: f64,
    cfg: &KUksStressConfig,
) -> Result<KUksStress, ferric_core::FerricError> {
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    with_jk(z, |j| {
        kpoint_null_xc_uks_stress(&z.cell, &z.prep, &z.mesh, &hcfg, &z.hk, j, &z.scf, a, cfg)
    })
}

pub fn a_equal_one_case(jk: Jk, exxdiv: ExxDiv) {
    a_equal_one_case_on(jk, exxdiv, G113);
}

/// The a = 1 wiring anchor on `mesh` (the weekly rows run 1x1x3; the
/// per-commit canary 1x1x1).
pub fn a_equal_one_case_on(jk: Jk, exxdiv: ExxDiv, mesh: Mesh) {
    let z = zero_xc(&H3, mesh, jk, exxdiv, 1.0);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let ks = null_stress(&z, 1.0, &scfg(Some(exxdiv), None, None, None)).unwrap();
    let hf = with_jk(&z, |j| {
        kpoint_uhf_stress(
            &z.cell,
            &z.prep,
            &z.mesh,
            &hcfg,
            &z.hk,
            j,
            &z.scf,
            exxdiv,
            &kcfg(),
        )
        .unwrap()
    });
    let mut worst = 0.0_f64;
    for ((n, a), (_, b)) in hf_parts(&ks.hf.parts).into_iter().zip(hf_parts(&hf.parts)) {
        let d = mdiff(&a, &b);
        eprintln!("{jk:?} {exxdiv:?}: a=1 null-XC vs UHF stress, {n:<18} {d:.1e}");
        worst = worst.max(d);
    }
    let d_tot = mdiff(&ks.de_deps, &hf.de_deps);
    eprintln!("{jk:?} {exxdiv:?}: total {d_tot:.1e}");
    assert!(
        worst < 1e-13 && d_tot < 1e-13,
        "{jk:?} {exxdiv:?}: {worst:e}"
    );
    // Not vacuous: open shell and the exchange walks matter.
    let nrm = |m: &[Array2<Complex64>]| -> f64 {
        m.iter()
            .map(|x| x.iter().map(|y| y.norm()).sum::<f64>())
            .sum()
    };
    assert!((nrm(&z.scf.density_alpha) - nrm(&z.scf.density_beta)).abs() > 0.1);
    match jk {
        Jk::Dense => assert!(mmax(&hf.parts.exchange) > 1e-4),
        Jk::Gdf => assert!(mmax(&hf.parts.fit_j3_sr) > 1e-4),
    }
    // The a-scaling mutants are identities here.
    for xm in [
        KXcStressMutation::ExchUnscaled,
        KXcStressMutation::ExchScaledTwice,
    ] {
        let m = null_stress(&z, 1.0, &scfg(Some(exxdiv), Some(xm), None, None)).unwrap();
        assert_eq!(mdiff(&m.de_deps, &ks.de_deps), 0.0, "{xm:?} at a = 1");
    }
}

// ------------------------------------------------------------------ (2)

pub struct RksTwin {
    pub stress: ferric_pbc::kxcstress::KRksStress,
}

pub fn rks_twin(fx: &Fx, f: &str) -> RksTwin {
    let cell = &fx.cell;
    let mut rcfg = KRksConfig::new(cell, ExxDiv::Ewald, f);
    rcfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    rcfg.krhf.scf.energy_conv = 1e-14;
    rcfg.krhf.scf.grad_conv = 1e-12;
    rcfg.krhf.scf.max_iter = 200;
    rcfg.grid = grid_cfg();
    let r = solve_krks(cell, &fx.prep, None, &fx.mesh, &rcfg).unwrap();
    let eri = KDenseAftEri::build(
        cell,
        &fx.prep,
        &fx.mesh,
        &fx.hk.s,
        ExxDiv::Ewald,
        &Default::default(),
    )
    .unwrap();
    let s = kpoint_rks_stress(
        cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KStressJk::Dense(&eri),
        &r.scf,
        f,
        &KRksStressConfig {
            stress: kcfg(),
            grid: KXcGrid::Atom(grid_cfg()),
            exxdiv: Some(ExxDiv::Ewald),
            ..Default::default()
        },
    )
    .unwrap();
    RksTwin { stress: s }
}

pub fn closed_shell_vs_rks(functionals: &[&str]) {
    for &f in functionals {
        let fx = fixture(&H2, G113, f, Jk::Dense, ExxDiv::Ewald);
        let u = stress(&fx);
        let r = rks_twin(&fx, f).stress;
        let mut rows = vec![
            ("xc_ao", mdiff(&u.xc_ao, &r.xc_ao)),
            ("xc_weight", mdiff(&u.xc_weight, &r.xc_weight)),
            ("de_deps", mdiff(&u.de_deps, &r.de_deps)),
            ("sigma", mdiff(&u.sigma, &r.sigma)),
        ];
        for ((n, a), (_, b)) in hf_parts(&u.hf.parts).into_iter().zip(hf_parts(&r.hf.parts)) {
            rows.push((n, mdiff(&a, &b)));
        }
        for (n, d) in rows {
            eprintln!("{f} closed-shell UKS vs RKS: {n:<18} {d:.2e}");
            assert!(d < BAR_CLOSED, "{f} {n}: {d:e}");
        }
        assert!(mmax(&u.xc_ao) > 1e-4 && mmax(&u.xc_weight) > 1e-4);
    }
}

/// Worst correct closed-shell residual 4.6e-15 (overlap_pulay); the smallest
/// visible mutant miss is 1.8e-2 (XcCrossDropped). Bar 1e-11.
pub const BAR_CLOSED: f64 = 1e-11;

// ------------------------------------------------------------------ (3)

pub fn gamma_side(cell: &Cell, f: &str) -> GammaStress {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hc = periodic_hcore(cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        cell,
        &prep,
        &hc.s,
        ExxDiv::Ewald,
        ANCHOR_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let mut gc = GammaUksConfig::new(f);
    gc.grid = grid_cfg();
    gc.exxdiv = ExxDiv::Ewald;
    gc.scf.energy_conv = 1e-13;
    gc.scf.density_conv = 1e-10;
    gc.scf.max_iter = 300;
    let r = gamma_uks(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &gc).unwrap();
    assert!(r.scf.converged);
    gamma_uks_stress(
        cell,
        &prep,
        &hcfg,
        &hc,
        &eri,
        &r.scf,
        &gc,
        &GammaStressConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Rows of (name, |k part - Gamma part|) for `n_cells` cells against the
/// (super)cell Gamma stress (`dE/de` per cell = supercell value / n_cells).
pub fn gamma_rows(k: &KUksStress, g: &GammaStress, n_cells: f64) -> Vec<(&'static str, f64)> {
    let p = &g.parts;
    let kp = &k.hf.parts;
    let s = 1.0 / n_cells;
    vec![
        ("xc_ao", mdiff(&k.xc_ao, &mscale(&p.xc_ao, s))),
        ("xc_weight", mdiff(&k.xc_weight, &mscale(&p.xc_weight, s))),
        ("overlap", mdiff(&kp.overlap(), &mscale(&p.overlap, s))),
        ("kinetic", mdiff(&kp.kinetic, &mscale(&p.kinetic, s))),
        ("vsr", mdiff(&kp.vsr, &mscale(&p.vsr, s))),
        ("vlr", mdiff(&kp.vlr, &mscale(&p.vlr, s))),
        ("c0_volume", mdiff(&kp.c0_volume, &mscale(&p.c0_volume, s))),
        (
            "eri=J+K",
            mdiff(&madd(&kp.coulomb, &kp.exchange), &mscale(&p.eri, s)),
        ),
        ("madelung", mdiff(&kp.madelung, &mscale(&p.madelung, s))),
        ("sigma", mdiff(&k.sigma, &g.sigma)),
    ]
}

/// Worst correct per-part residual against the Gamma anchors 9.8e-12 (J+K,
/// 1x1x1 PBE), 2.1e-12 (1x1x3 supercell); mutants miss by >= 4.7e-3. Bar 1e-9:
/// 100x above the floor, 5e6 below the smallest mutant.
pub const BAR_GAMMA: f64 = 1e-9;

pub fn one_point_case(f: &str) {
    let gam = gamma_side(&cell_of(&H3), f);
    let fx = fixture(&H3, G111, f, Jk::Dense, ExxDiv::Ewald);
    let k = stress(&fx);
    for (n, d) in gamma_rows(&k, &gam, 1.0) {
        eprintln!("{f} H3 1x1x1 vs Gamma UKS: {n:<10} {d:.2e}");
        assert!(d < BAR_GAMMA, "{f} {n}: {d:e}");
    }
    assert!(mmax(&k.xc_ao) > 1e-4 && mmax(&k.xc_weight) > 1e-4);
    assert!(mmax(&k.hf.parts.exchange) > 1e-4 || f != "PBE0");
}

/// Explicit diag(n) supercell of `pos` (copy-major, copy 0 first).
pub fn supercell_pos(pos: &[[f64; 3]], a: f64, n: [usize; 3], mult: usize) -> Cell {
    let mut atoms = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                for p in pos {
                    atoms.push([
                        p[0] + m0 as f64 * a,
                        p[1] + m1 as f64 * a,
                        p[2] + m2 as f64 * a,
                    ]);
                }
            }
        }
    }
    let lat = [
        [a * n[0] as f64, 0.0, 0.0],
        [0.0, a * n[1] as f64, 0.0],
        [0.0, 0.0, a * n[2] as f64],
    ];
    let mut mol = hydrogens(&atoms);
    mol.multiplicity = mult;
    Cell::new(mol, lat).unwrap()
}

// ------------------------------------------------------------------ (4)(5)

/// FD step and bar, from `fd_window_probe` (H3 doublet, PBE0, dense Ewald,
/// 1x1x3, shared cutoff 14, all nine components): worst |FD - analytic| 5.4e-9
/// at h = 2.5e-5, 1.5e-8 at 5e-5, 5.4e-8 at 1e-4 (h^2 truncation; the h = 2.5e-5
/// residual is the SCF-energy noise / 2h floor). All-nine at 2.5e-5: PBE0 Ewald
/// 5.4e-9, PBE none 5.3e-9; RS-GDF 2.0e-9, Monkhorst-Pack 2.2e-10, uniform grid
/// 2.0e-9. Bar 3e-8 = 5.6x the worst correct residual and ~1.6e5 below the
/// smallest mutant miss (NoHessian, 4.7e-3 against the Gamma anchor).
pub const FD_H: f64 = 2.5e-5;
pub const FD_BAR: f64 = 3e-8;

pub fn fd_case(f: &str, jk: Jk, exxdiv: ExxDiv, m: Mesh, comps: &[(usize, usize)]) {
    let fx = fixture(&H3, m, f, jk, exxdiv);
    let s = stress(&fx);
    let fd = fd_components(&fx.cell, m, f, jk, exxdiv, FD_H, comps);
    let mut worst = 0.0_f64;
    for (&(i, j), d) in comps.iter().zip(&fd) {
        let miss = (s.de_deps[i][j] - d).abs();
        eprintln!(
            "{f} {jk:?} {exxdiv:?} eps({i},{j}): analytic {:.9e} FD {d:.9e} miss {miss:.2e}",
            s.de_deps[i][j]
        );
        worst = worst.max(miss);
    }
    assert!(worst < FD_BAR, "{f} {jk:?} {exxdiv:?}: {worst:e}");
}

// ------------------------------------------------------------------ memory

/// The byte count of a typed ledger error (`... (N bytes; ...`).
pub fn charge_of(m: &str) -> usize {
    let head = &m[..m.find(" bytes;").expect("byte count in the error")];
    head[head.rfind('(').unwrap() + 1..].parse().unwrap()
}

/// J(k), K(k) (K with the build's Madelung shift) of density `d`.
pub fn jk_mats(
    z: &Zx,
    d: &[Array2<Complex64>],
) -> (Vec<Array2<Complex64>>, Vec<Array2<Complex64>>) {
    let n = z.prep.nbasis();
    let nk = z.mesh.nk();
    let mut j: Vec<Array2<Complex64>> = (0..nk).map(|_| Array2::zeros((n, n))).collect();
    let mut k = j.clone();
    let eri =
        KDenseAftEri::build(&z.cell, &z.prep, &z.mesh, &z.hk.s, z.exxdiv, &dense_cfg()).unwrap();
    eri.jk_builder().build(d, &mut j, &mut k).unwrap();
    (j, k)
}

// ------------------------------------------------------------------ mutants

pub const MUTANT_BAR: f64 = 1e-5;

pub fn all_spin_mutants() -> [KUStressMutation; 9] {
    use KUStressMutation as U;
    [
        U::XcSpinSwap,
        U::XcDropBeta,
        U::XcCrossDropped,
        U::XcCrossOwn,
        U::XcSpinAveraged,
        U::PulayFockSwap,
        U::PulayKSwap,
        U::PulayJOneSpin,
        U::PulayNoVxcBeta,
    ]
}
