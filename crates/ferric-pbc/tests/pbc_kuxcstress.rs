//! Analytic stress of the k-point open-shell UKS energy (`ferric_pbc::kuxcstress`,
//! module doc) for LDA, GGA and global hybrids. Predictions are written next to
//! each test before measuring; measured numbers are recorded after.
//!
//! WHICH ANCHOR COVERS WHICH DEFECT (the closed-shell limit is blind to every
//! alpha/beta asymmetry; a = 1 with a zero XC is blind to every a-scaling):
//!
//! * (1) a = 1 + zero XC == `kpoint_uhf_stress` per part: the KS entry's F
//!   rebuild, exxdiv routing, spin plumbing of the HF part. Blind to a-scaling
//!   (ExchUnscaled / ExchScaledTwice are asserted identities there).
//! * (2) closed-shell UKS (Da == Db) == the merged RKS stress per part: spin
//!   FACTORS (2 vs 1, a dropped spin that is not doubled), the shared XC terms.
//!   BLIND to XcSpinSwap, XcCrossOwn, XcSpinAveraged, PulayFockSwap, PulayKSwap
//!   (asserted blind below, so the claim is measured, not assumed).
//! * (3) 1x1x1 vs the Gamma UKS stress on H3 (Da != Db) and the 1x1x3 vs the
//!   explicit 3-copy supercell: every spin defect and every a-scaling defect at
//!   nk = 1; the supercell adds the Bloch phases and mesh Madelung constant.
//! * (4) FD of the energy under symmetric strain: the stress of the energy as
//!   computed, any wrong factor.
//! * (5) the trace against isotropic scaling.
//!
//! NOT covered: p/d shells and heavy atoms / ECP (every fixture is H/STO-3G),
//! ROKS/ROHF (refused, typed), range-separated hybrids / meta-GGA / VV10 (refused),
//! shifted meshes except one ignored FD row, the production (75, 302) grid,
//! no external (PySCF) anchor (its k stress refuses open-shell hybrids).

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::{gamma_uks, GammaUksConfig, PeriodicGridConfig};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::periodic_hcore;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kdft::{solve_krks, KRksConfig};
use ferric_pbc::kgrad::KRsGdfGradSource;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{KJkKind, KPointInjection, KPointJk};
use ferric_pbc::kstress::{
    kpoint_uhf_stress, KStressConfig, KStressJk, KStressMutation, KStressParts,
};
use ferric_pbc::kuks::{solve_kuks, KUksConfig, KUksResult};
use ferric_pbc::kuscf::{solve_kuks_injected, KPointXcPolarized, KUScfResult};
use ferric_pbc::kuxcstress::{
    kpoint_null_xc_uks_stress, kpoint_rohf_stress, kpoint_roks_stress, kpoint_uks_stress,
    KUStressMutation, KUksStress, KUksStressConfig, UKS_COMMUTATOR_TOL,
};
use ferric_pbc::kxcgrad::KXcGrid;
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStressConfig, KXcStressMutation};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::stress::{gamma_uks_stress, GammaStress, GammaStressConfig, Mat3};
use ferric_pbc::uhf::GammaUhfIntegrals;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
/// H3 doublet (na = 2, nb = 1): a genuinely spin-polarised cell, Da != Db.
const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];
const ALL9: [(usize, usize); 9] = [
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
const CUTOFF: f64 = 14.0;
const ANCHOR_PRECISION: f64 = 1e-12;
/// (energy_conv, grad_conv) of the deliberately loose UKS SCF of the gate test.
const LOOSE_UKS: (f64, f64) = (1e-5, 1e-2);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Jk {
    Dense,
    Gdf,
}

#[derive(Clone, Copy)]
struct Mesh {
    n: [usize; 3],
    mp: bool,
}

const G111: Mesh = Mesh {
    n: [1, 1, 1],
    mp: false,
};
const G113: Mesh = Mesh {
    n: [1, 1, 3],
    mp: false,
};

fn mesh_of(cell: &Cell, m: Mesh) -> KPointMesh {
    if m.mp {
        KPointMesh::monkhorst_pack(cell, m.n).unwrap()
    } else {
        KPointMesh::gamma_centred(cell, m.n).unwrap()
    }
}

fn et_sp_aux() -> BasisSet {
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

fn kgdf_cfg() -> KRsGdfConfig {
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
    static SETTINGS: std::cell::Cell<(f64, f64)> = const { std::cell::Cell::new((1e-14, 1e-12)) };
}

fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

fn cell_of(pos: &[[f64; 3]]) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = if pos.len() % 2 == 1 { 2 } else { 1 };
    Cell::new(mol, cubic(4.0)).expect("cell")
}

fn kuks_cfg(cell: &Cell, functional: &str, jk: Jk, exxdiv: ExxDiv) -> KUksConfig {
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
fn scf_on(cell: &Cell, m: Mesh, functional: &str, jk: Jk, exxdiv: ExxDiv) -> KUksResult {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = mesh_of(cell, m);
    let cfg = kuks_cfg(cell, functional, jk, exxdiv);
    let aux = (jk == Jk::Gdf).then(|| prep_for(cell, &et_sp_aux()));
    let r = solve_kuks(cell, &prep, aux.as_ref(), &mesh, &cfg)
        .unwrap_or_else(|e| panic!("solve_kuks {functional}: {e}"));
    assert!(r.scf.converged);
    r
}

fn energy_on(cell: &Cell, m: Mesh, f: &str, jk: Jk, exxdiv: ExxDiv) -> f64 {
    scf_on(cell, m, f, jk, exxdiv).scf.energy
}

struct Fx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    aux: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    res: KUksResult,
    jk: Jk,
    exxdiv: ExxDiv,
    functional: String,
}

fn fixture_cell(cell: Cell, m: Mesh, functional: &str, jk: Jk, exxdiv: ExxDiv) -> Fx {
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

fn fixture(pos: &[[f64; 3]], m: Mesh, functional: &str, jk: Jk, exxdiv: ExxDiv) -> Fx {
    fixture_cell(cell_of(pos), m, functional, jk, exxdiv)
}

fn scfg(
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

fn stress_with(
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

fn stress(fx: &Fx) -> KUksStress {
    stress_with(
        fx,
        &fx.functional,
        &fx.res.scf,
        &scfg(Some(fx.exxdiv), None, None, None),
    )
    .expect("kpoint_uks_stress")
}

fn stress_mut(
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

fn mdiff(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

fn mmax(a: &Mat3) -> f64 {
    a.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn mscale(a: &Mat3, f: f64) -> Mat3 {
    let mut o = *a;
    for r in o.iter_mut() {
        for x in r.iter_mut() {
            *x *= f;
        }
    }
    o
}

fn madd(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = *a;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] += b[i][j];
        }
    }
    o
}

fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    cell.strained(&eps_at(i, j, h)).expect("strained cell")
}

/// Central FD of the UKS SCF energy for the listed components at step `h`.
fn fd_components(
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
fn hf_parts(p: &KStressParts) -> Vec<(&'static str, Mat3)> {
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

fn kcfg() -> KStressConfig {
    KStressConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

// ----------------------------------------------- zero-XC fixed-density SCF

struct ZeroXcPol(usize);
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

struct Zx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    aux: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    scf: KUScfResult,
    jk: Jk,
    exxdiv: ExxDiv,
}

fn dense_cfg() -> KDenseAftConfig {
    KDenseAftConfig {
        precision: ANCHOR_PRECISION,
        ..Default::default()
    }
}

/// Hartree + `a` exact-exchange k UKS with a zero XC (a = 1 is k-point UHF).
fn zero_xc(pos: &[[f64; 3]], m: Mesh, jk: Jk, exxdiv: ExxDiv, a: f64) -> Zx {
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

fn with_jk<R>(z: &Zx, f: impl FnOnce(KStressJk<'_>) -> R) -> R {
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

fn null_stress(
    z: &Zx,
    a: f64,
    cfg: &KUksStressConfig,
) -> Result<KUksStress, ferric_core::FerricError> {
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    with_jk(z, |j| {
        kpoint_null_xc_uks_stress(&z.cell, &z.prep, &z.mesh, &hcfg, &z.hk, j, &z.scf, a, cfg)
    })
}

// ------------------------------------------------------------------ (1)

/// (1) WIRING ANCHOR, per part. a = 1 with a zero XC IS the k-point UHF stress:
/// same `assemble(Unrestricted)`, but the KS entry rebuilds F_sigma itself
/// (guarded) and takes its exxdiv from the config. Artifact hypothesis: a leaked
/// XC term, a K scaled by anything but 1, a swapped spin Fock would show as
/// nonzero; physics predicts 0 to roundoff. BLIND to every a-scaling defect
/// (ExchUnscaled/ExchScaledTwice are identities at a = 1: asserted EXACTLY 0).
#[test]
fn a_equal_one_with_zero_xc_is_the_uhf_stress_per_part() {
    a_equal_one_case(Jk::Dense, ExxDiv::Ewald);
}

#[test]
#[ignore = "the other three rows (dense none, RS-GDF none/ewald) of the a = 1 anchor; ~90 s"]
fn a_equal_one_with_zero_xc_is_the_uhf_stress_per_part_other_rows() {
    for (jk, exxdiv) in [
        (Jk::Dense, ExxDiv::None),
        (Jk::Gdf, ExxDiv::None),
        (Jk::Gdf, ExxDiv::Ewald),
    ] {
        a_equal_one_case(jk, exxdiv);
    }
}

fn a_equal_one_case(jk: Jk, exxdiv: ExxDiv) {
    let z = zero_xc(&H3, G113, jk, exxdiv, 1.0);
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

struct RksTwin {
    stress: ferric_pbc::kxcstress::KRksStress,
}

fn rks_twin(fx: &Fx, f: &str) -> RksTwin {
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

/// (2) CLOSED-SHELL LIMIT == the merged RKS stress, per part. Independent SCF
/// loops (k-UHF skeleton with a polarized kernel vs the closed-shell loop).
/// Artifact hypothesis: a spin-factor slip (2 vs 1) or a mishandled gate breaks
/// equality; physics predicts agreement at the SCF-density floor. BLIND to the
/// alpha/beta asymmetric defects (see `closed_shell_limit_is_blind_to_...`).
#[test]
fn closed_shell_limit_equals_the_rks_stress_per_part() {
    closed_shell_vs_rks(&["PBE0"]);
}

#[test]
#[ignore = "closed-shell UKS vs RKS for LDA and PBE; ~40 s"]
fn closed_shell_limit_equals_the_rks_stress_per_part_lda_pbe() {
    closed_shell_vs_rks(&["LDA", "PBE"]);
}

fn closed_shell_vs_rks(functionals: &[&str]) {
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

const BAR_CLOSED: f64 = 1e-11;

/// The closed-shell limit is BLIND to the alpha/beta-asymmetric defects (so
/// the Gamma/FD anchors, not this one, must carry them): asserted, not assumed.
/// A dropped beta, J of one spin, and the other exact-exchange/Fock defects are
/// NOT blind and must be visible there.
#[test]
fn closed_shell_limit_is_blind_to_spin_swaps_and_sees_dropped_spins() {
    use KUStressMutation as U;
    let fx = fixture(&H2, G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let base = stress(&fx);
    for sm in [
        U::XcSpinSwap,
        U::XcCrossOwn,
        U::XcSpinAveraged,
        U::PulayFockSwap,
        U::PulayKSwap,
    ] {
        let d = mdiff(
            &stress_mut(&fx, None, Some(sm), None).de_deps,
            &base.de_deps,
        );
        eprintln!("closed-shell blindness {sm:?}: {d:.2e}");
        assert!(d < BAR_CLOSED, "{sm:?} is visible closed-shell: {d:e}");
    }
    for sm in [
        U::XcDropBeta,
        U::XcCrossDropped,
        U::PulayJOneSpin,
        U::PulayNoVxcBeta,
    ] {
        let d = mdiff(
            &stress_mut(&fx, None, Some(sm), None).de_deps,
            &base.de_deps,
        );
        eprintln!("closed-shell visibility {sm:?}: {d:.2e}");
        assert!(d > 1e-6, "{sm:?} is invisible closed-shell: {d:e}");
    }
}

// ------------------------------------------------------------------ (3)

fn gamma_side(cell: &Cell, f: &str) -> GammaStress {
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
fn gamma_rows(k: &KUksStress, g: &GammaStress, n_cells: f64) -> Vec<(&'static str, f64)> {
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

const BAR_GAMMA: f64 = 1e-9;

/// (3a) 1x1x1 == the Gamma UKS stress, PER PART, on the open-shell H3 doublet
/// (real lattice sums vs complex Bloch AOs; separate SCFs). Sees the spin
/// channels (Da != Db) that the closed-shell limit cannot. Prediction: every
/// part <= 1e-9 (both SCFs converge to ~1e-10 in the density).
#[test]
fn one_point_mesh_matches_the_gamma_uks_stress_per_part() {
    one_point_case("PBE0");
}

#[test]
#[ignore = "PBE and LDA rows of the 1x1x1 Gamma UKS anchor; ~40 s"]
fn one_point_mesh_matches_the_gamma_uks_stress_per_part_pbe_lda() {
    one_point_case("PBE");
    one_point_case("LDA");
}

fn one_point_case(f: &str) {
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
fn supercell_pos(pos: &[[f64; 3]], a: f64, n: [usize; 3], mult: usize) -> Cell {
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

/// (3b) SUPERCELL ANCHOR on the non-TRIM 1x1x3 mesh, open shell: the k stress
/// per cell is the Gamma UKS stress of the explicit 3-copy supercell
/// (multiplicity (Na - Nb) N_k + 1 = 4; sigma equal, dE/de / 3), per part. With
/// Ewald the supercell Gamma Madelung constant IS the mesh constant. Same
/// explicit cutoff on both sides. Prediction <= 1e-9 per part.
#[test]
#[ignore = "9-atom Gamma UKS supercell; ~60 s"]
fn k_mesh_uks_stress_is_the_gamma_supercell_stress_per_part() {
    let f = "PBE0";
    let gam = gamma_side(&supercell_pos(&H3, 4.0, [1, 1, 3], 4), f);
    let fx = fixture(&H3, G113, f, Jk::Dense, ExxDiv::Ewald);
    let k = stress(&fx);
    for (n, d) in gamma_rows(&k, &gam, 3.0) {
        eprintln!("{f} H3 1x1x3 vs supercell: {n:<10} {d:.2e}");
        assert!(d < BAR_GAMMA, "{f} {n}: {d:e}");
    }
    assert!(mmax(&k.hf.parts.madelung) > 1e-6);
}

// ------------------------------------------------------------------ (4)(5)

/// FD step and bar: see `fd_window_probe` (printout is the floor and window).
const FD_H: f64 = 2.5e-5;
const FD_BAR: f64 = 3e-8;

/// (4) FD of ferric's own UKS energy under symmetric strain at fixed
/// fractional coordinates (fresh SCF per displaced cell, mesh rebuilt on the
/// strained cell with the same integer numerators, shared cutoff 14). Three
/// components (diagonal along the mesh axis, a shear, the transverse diagonal).
#[test]
fn pbe0_uks_stress_matches_fd_dense_ewald_1x1x3() {
    fd_case(
        "PBE0",
        Jk::Dense,
        ExxDiv::Ewald,
        G113,
        &[(2, 2), (0, 2), (0, 0)],
    );
}

#[test]
#[ignore = "PBE row of the FD anchor (6 SCFs); ~60 s"]
fn pbe_uks_stress_matches_fd_dense_none_1x1x3() {
    fd_case(
        "PBE",
        Jk::Dense,
        ExxDiv::None,
        G113,
        &[(2, 2), (0, 2), (1, 0)],
    );
}

#[test]
#[ignore = "RS-GDF row of the FD anchor (6 SCFs); ~120 s"]
fn pbe0_uks_stress_matches_fd_rsgdf_ewald_1x1x3() {
    fd_case("PBE0", Jk::Gdf, ExxDiv::Ewald, G113, &[(2, 2), (0, 1)]);
}

#[test]
#[ignore = "8-point shifted mesh, 2 components x 2 SCFs; needs a quiet box"]
fn pbe0_uks_stress_matches_fd_on_the_monkhorst_pack_mesh() {
    let m = Mesh {
        n: [2, 2, 2],
        mp: true,
    };
    fd_case("PBE0", Jk::Dense, ExxDiv::Ewald, m, &[(2, 2), (0, 1)]);
}

fn fd_case(f: &str, jk: Jk, exxdiv: ExxDiv, m: Mesh, comps: &[(usize, usize)]) {
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

/// (5) Pressure from isotropic scaling: sum_a dE/de_aa equals the FD of E under
/// eps = h*I (PBE, dense, none: the pure-GGA twin of the PBE0 FD above).
#[test]
fn trace_matches_isotropic_scaling_fd() {
    let fx = fixture(&H3, G113, "PBE", Jk::Dense, ExxDiv::None);
    let s = stress(&fx);
    let tr = s.de_deps[0][0] + s.de_deps[1][1] + s.de_deps[2][2];
    let iso = |h: f64| {
        let e = [[h, 0.0, 0.0], [0.0, h, 0.0], [0.0, 0.0, h]];
        let c = fx.cell.strained(&e).unwrap();
        energy_on(&c, G113, "PBE", Jk::Dense, ExxDiv::None)
    };
    let fd = (iso(FD_H) - iso(-FD_H)) / (2.0 * FD_H);
    eprintln!(
        "PBE trace: analytic {tr:.9e} FD(isotropic) {fd:.9e} miss {:.2e}",
        (tr - fd).abs()
    );
    assert!((tr - fd).abs() < 3.0 * FD_BAR);
}

/// PROBE: choose the FD step. All nine components of PBE0 (dense, Ewald,
/// 1x1x3 H3) at four steps; the printout is the floor and the window.
#[test]
#[ignore = "probe: 4 steps x 18 SCFs, needs a quiet box; run with --nocapture"]
fn fd_window_probe() {
    let fx = fixture(&H3, G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let s = stress(&fx);
    for h in [1e-5, 2.5e-5, 5e-5, 1e-4] {
        let fd = fd_components(&fx.cell, G113, "PBE0", Jk::Dense, ExxDiv::Ewald, h, &ALL9);
        let mut row = String::new();
        let mut worst = 0.0_f64;
        for (&(i, j), d) in ALL9.iter().zip(&fd) {
            let m = (d - s.de_deps[i][j]).abs();
            worst = worst.max(m);
            row += &format!(" {m:.1e}");
        }
        eprintln!("h={h:.1e} worst {worst:.2e}; per component (00 01 02 10 11 12 20 21 22):{row}");
    }
}

#[test]
#[ignore = "all-nine FD for PBE0 dense Ewald and PBE dense none (36 SCFs); needs a quiet box"]
fn uks_stress_matches_fd_all_nine() {
    for (f, exxdiv) in [("PBE0", ExxDiv::Ewald), ("PBE", ExxDiv::None)] {
        let fx = fixture(&H3, G113, f, Jk::Dense, exxdiv);
        let s = stress(&fx);
        let fd = fd_components(&fx.cell, G113, f, Jk::Dense, exxdiv, FD_H, &ALL9);
        let mut worst = 0.0_f64;
        for (&(i, j), d) in ALL9.iter().zip(&fd) {
            worst = worst.max((d - s.de_deps[i][j]).abs());
        }
        eprintln!("{f} {exxdiv:?}: all-nine FD miss {worst:.2e}");
        assert!(worst < FD_BAR, "{f} {exxdiv:?}: {worst:e}");
    }
}

// ------------------------------------------------------ bitwise / guards

#[test]
#[ignore = "two full PBE0 stresses at 1 and 4 threads; thread bit-identity of the shared assembly is also pinned by pbc_parallel_bitwise"]
fn uks_stress_is_bit_identical_across_threads() {
    let fx = fixture(&H3, G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| stress(&fx))
    };
    let (a, b) = (run(1), run(4));
    assert_eq!(a.de_deps, b.de_deps);
    assert_eq!(a.xc_ao, b.xc_ao);
    assert_eq!(a.e_xc, b.e_xc);
}

/// Typed refusals (exact wording), the per-spin Fock guard from every side, and
/// the ROKS/ROHF refusal.
#[test]
fn refusals_and_the_per_spin_fock_guard() {
    let fx = fixture(&H3, G111, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let ok_cfg = scfg(Some(ExxDiv::Ewald), None, None, None);
    let run = |f: &str, s: &KUScfResult, cfg: &KUksStressConfig| stress_with(&fx, f, s, cfg);
    for (name, key) in [
        (
            "HSE06",
            "range-separated hybrid stress at k: HSE06: the k-point XC stress covers LDA, GGA and GLOBAL hybrid functionals; the screened-exchange strain is not implemented",
        ),
        (
            "SCAN",
            "meta-GGA stress at k: SCAN: the tau strain (Hessian-moment terms) is not implemented",
        ),
        ("wB97X-V", "VV10"),
        ("B2PLYP", "double hybrid"),
    ] {
        let e = run(name, &fx.res.scf, &ok_cfg)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        eprintln!("{name}: {e}");
        assert!(e.contains(key), "{name}: {e}");
    }
    let e = run("PBE0", &fx.res.scf, &scfg(None, None, None, None))
        .expect_err("no exxdiv")
        .to_string();
    assert!(
        e.contains("kpoint_uks_stress: a hybrid functional (exact exchange 0.25) needs KRksStressConfig::exxdiv"),
        "{e}"
    );
    let ok = run("PBE0", &fx.res.scf, &ok_cfg).unwrap();
    assert!(ok.hf.fock_mismatch < 1e-12, "{:e}", ok.hf.fock_mismatch);
    assert!(ok.hf.commutator < UKS_COMMUTATOR_TOL / 100.0);
    let pbe = fixture(&H3, G111, "PBE", Jk::Dense, ExxDiv::Ewald);
    let guard_msg = "kpoint_uks_stress: the Fock matrix rebuilt";
    let cases: Vec<(&str, Result<KUksStress, ferric_core::FerricError>)> = vec![
        (
            "exxdiv None on an Ewald SCF",
            run(
                "PBE0",
                &fx.res.scf,
                &scfg(Some(ExxDiv::None), None, None, None),
            ),
        ),
        ("B3LYP on a PBE0 SCF", run("B3LYP", &fx.res.scf, &ok_cfg)),
        ("PBE stress on a PBE0 SCF", run("PBE", &fx.res.scf, &ok_cfg)),
        (
            "PBE0 stress on a PBE SCF",
            run("PBE0", &pbe.res.scf, &ok_cfg),
        ),
    ];
    for (n, r) in cases {
        let e = r
            .err()
            .unwrap_or_else(|| panic!("{n} accepted"))
            .to_string();
        eprintln!("{n}: {e}");
        assert!(e.contains(guard_msg), "{n}: {e}");
    }
    // Tampers of each spin's Fock are refused, naming the spin.
    let mut t = fx.res.scf.clone();
    t.fock_alpha[0][(0, 1)] += Complex64::new(1e-6, 0.0);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("alpha tamper")
        .to_string();
    assert!(e.contains(guard_msg) && e.contains("fock_alpha"), "{e}");
    let mut t = fx.res.scf.clone();
    t.fock_beta[0][(0, 1)] += Complex64::new(1e-6, 0.0);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("beta tamper")
        .to_string();
    assert!(e.contains(guard_msg) && e.contains("fock_beta"), "{e}");
    let mut t = fx.res.scf.clone();
    std::mem::swap(&mut t.fock_alpha, &mut t.fock_beta);
    let e = run("PBE0", &t, &ok_cfg).expect_err("swapped").to_string();
    assert!(e.contains(guard_msg) && e.contains("fock_alpha"), "{e}");
    // Unconverged SCF.
    let mut t = fx.res.scf.clone();
    t.converged = false;
    assert!(run("PBE0", &t, &ok_cfg).is_err());
    // A pure functional needs no exxdiv and has no exchange part.
    let a = stress_with(&pbe, "PBE", &pbe.res.scf, &scfg(None, None, None, None)).unwrap();
    assert_eq!(mmax(&a.hf.parts.exchange), 0.0);
    // The exchange fraction of the diagnostic entry is validated.
    let z = zero_xc(&H3, G111, Jk::Dense, ExxDiv::None, 0.0);
    for bad in [-0.1, 1.5, f64::NAN] {
        let e = null_stress(&z, bad, &scfg(None, None, None, None))
            .err()
            .unwrap_or_else(|| panic!("a = {bad} accepted"))
            .to_string();
        assert!(e.contains("outside [0, 1]"), "{e}");
    }
}

/// ROKS/ROHF stress is refused with a typed, named error.
#[test]
fn roks_and_rohf_stress_are_refused_typed() {
    use ferric_pbc::kroks::{solve_kroks, KRoksConfig};
    let cell = cell_of(&H3);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let mut cfg = KRoksConfig::new(&cell, ExxDiv::Ewald, "PBE0");
    cfg.rohf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.grid = grid_cfg();
    let r = solve_kroks(&cell, &prep, None, &mesh, &cfg).unwrap();
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::Ewald,
        &Default::default(),
    )
    .unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let c = scfg(Some(ExxDiv::Ewald), None, None, None);
    let e = kpoint_roks_stress(
        &cell,
        &prep,
        &mesh,
        &hcfg,
        &hk,
        KStressJk::Dense(&eri),
        &r.scf,
        "PBE0",
        &c,
    )
    .expect_err("ROKS stress accepted")
    .to_string();
    assert!(
        e.contains("restricted-open-shell stress at k: kpoint_roks_stress: the ROKS stress is not implemented"),
        "{e}"
    );
    let e = kpoint_rohf_stress(
        &cell,
        &prep,
        &mesh,
        &hcfg,
        &hk,
        KStressJk::Dense(&eri),
        &r.scf,
        ExxDiv::Ewald,
        &c,
    )
    .expect_err("ROHF stress accepted")
    .to_string();
    assert!(
        e.contains("kpoint_rohf_stress: the ROHF stress is not implemented"),
        "{e}"
    );
}

/// A loosely converged UKS state is refused with the commutator message; the
/// same state with the gate off returns a stress that is wrong by far more than
/// the FD bar (so the gate matters), and a tight state passes the gate.
#[test]
fn loose_uks_scf_is_refused_by_the_stationarity_gate() {
    let tight = fixture(&H3, G111, "PBE", Jk::Dense, ExxDiv::None);
    let st = stress(&tight);
    assert!(
        st.hf.commutator < UKS_COMMUTATOR_TOL / 100.0,
        "{:e}",
        st.hf.commutator
    );
    SETTINGS.with(|s| s.set(LOOSE_UKS));
    let loose = fixture(&H3, G111, "PBE", Jk::Dense, ExxDiv::None);
    let e = stress_with(&loose, "PBE", &loose.res.scf, &scfg(None, None, None, None))
        .expect_err("loose SCF accepted")
        .to_string();
    eprintln!("loose UKS: {e}");
    assert!(
        e.contains("kpoint_uks_stress: the spin Fock/density commutator")
            && e.contains("exceeds 1e-5"),
        "{e}"
    );
    let off = stress_mut(
        &loose,
        None,
        Some(KUStressMutation::StationarityGateOff),
        None,
    );
    let miss = mdiff(&off.de_deps, &st.de_deps);
    eprintln!(
        "loose UKS: commutator {:.3e}, stress error with the gate off {miss:.3e}",
        off.hf.commutator
    );
    assert!(off.hf.commutator > UKS_COMMUTATOR_TOL);
    assert!(
        miss > 10.0 * FD_BAR,
        "gate off is not wrong enough: {miss:e}"
    );
}

// ------------------------------------------------------------------ memory

/// The byte count of a typed ledger error (`... (N bytes; ...`).
fn charge_of(m: &str) -> usize {
    let head = &m[..m.find(" bytes;").expect("byte count in the error")];
    head[head.rfind('(').unwrap() + 1..].parse().unwrap()
}

/// Budget gates of the UKS stress. (i) A 4 MB budget fails typed and a 16 MB
/// budget reproduces the AMPLE-budget stress BITWISE (null XC, a = 0.5, no grid
/// pass). (ii) The Fock-guard reservation is read from the typed error the
/// PRODUCTION path raises when the budget holds the per-k arrays but not the
/// guard matrices: it must be ten N_k n^2 complex arrays
/// (D_a, D_b, J_a, J_b, K_a, K_b, F_a, F_b, V_a, V_b). A charge reverted to the
/// closed-shell five fails the equality.
#[test]
fn budgets_either_reproduce_the_uks_stress_or_fail_typed() {
    let z = zero_xc(&H3, G113, Jk::Dense, ExxDiv::Ewald, 0.5);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let mut scf = z.scf.clone();
    let (ja, ka) = jk_mats(&z, &z.scf.density_alpha);
    let (jb, kb) = jk_mats(&z, &z.scf.density_beta);
    for q in 0..scf.fock_alpha.len() {
        for (f, k) in [(&mut scf.fock_alpha, &ka), (&mut scf.fock_beta, &kb)] {
            let mut m = &z.hk.h[q] + &ja[q] + &jb[q];
            m.scaled_add(Complex64::new(-0.5, 0.0), &k[q]);
            f[q] = m;
        }
    }
    let z = Zx { scf, ..z };
    let at = |b: usize| {
        let mut cfg = scfg(Some(ExxDiv::Ewald), None, None, None);
        cfg.base.stress.budget_bytes = Some(b);
        null_stress(&z, 0.5, &cfg)
    };
    let _ = hcfg;
    let full = at(AMPLE).unwrap();
    let e = at(4 << 20).expect_err("4 MB must not hold the strain K list");
    eprintln!("4 MB: {e}");
    assert_eq!(at(16 << 20).unwrap().de_deps, full.de_deps);
    let (mut lo, mut hi) = (0usize, 1usize << 20);
    let mut guard_msg = None;
    for _ in 0..40 {
        let mid = lo + (hi - lo) / 2;
        match at(mid) {
            Err(e) => {
                let m = e.to_string();
                if m.contains("Fock-guard matrices") {
                    guard_msg = Some(m);
                    break;
                } else if m.contains("k stress per-k complex") || m.contains("residue-folded") {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            Ok(_) => hi = mid,
        }
    }
    let m = guard_msg.expect("the bisection must land in the guard band");
    eprintln!("guard gate: {m}");
    assert!(
        m.contains("D_a, D_b, J_a, J_b, K_a, K_b, F_a, F_b, V_a, V_b"),
        "{m}"
    );
    let n = z.prep.nbasis();
    assert_eq!(charge_of(&m), 10 * z.mesh.nk() * n * n * 16, "{m}");
}

/// J(k), K(k) (K with the build's Madelung shift) of density `d`.
fn jk_mats(z: &Zx, d: &[Array2<Complex64>]) -> (Vec<Array2<Complex64>>, Vec<Array2<Complex64>>) {
    let n = z.prep.nbasis();
    let nk = z.mesh.nk();
    let mut j: Vec<Array2<Complex64>> = (0..nk).map(|_| Array2::zeros((n, n))).collect();
    let mut k = j.clone();
    let eri =
        KDenseAftEri::build(&z.cell, &z.prep, &z.mesh, &z.hk.s, z.exxdiv, &dense_cfg()).unwrap();
    eri.jk_builder().build(d, &mut j, &mut k).unwrap();
    (j, k)
}

/// The XC chunk charge, read from the typed error of the PRODUCTION PBE0 path:
/// bisect the budget until the AO-moment chunk check fails and compare its byte
/// count with the arithmetic of the module doc (40 AO planes per k and point,
/// eight work planes, four N_k nbf^2 partial-V/accumulator arrays).
#[test]
fn xc_chunk_charge_is_read_from_the_production_error() {
    let fx = fixture(&H3, G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let at = |b: usize| {
        let mut cfg = scfg(Some(ExxDiv::Ewald), None, None, None);
        cfg.base.stress.budget_bytes = Some(b);
        stress_with(&fx, &fx.functional, &fx.res.scf, &cfg)
    };
    let (mut lo, mut hi) = (0usize, 64usize << 20);
    let mut found = None;
    for _ in 0..60 {
        let mid = lo + (hi - lo) / 2;
        match at(mid) {
            Err(e) => {
                let m = e.to_string();
                if m.contains("UKS XC stress AO moment chunk") {
                    found = Some(m);
                    break;
                }
                lo = mid;
            }
            Ok(_) => hi = mid,
        }
    }
    let m = found.expect("the bisection must land on the XC chunk gate");
    eprintln!("xc chunk gate: {m}");
    let pts: usize = m
        .split('(')
        .nth(1)
        .unwrap()
        .split(' ')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let (nk, n) = (fx.mesh.nk(), fx.prep.nbasis());
    let want = nk * n * pts * 16 * 40 + n * pts * 16 * 8 + nk * n * n * 16 * 4;
    assert_eq!(charge_of(&m), want, "{m}");
}

// ------------------------------------------------------------------ mutants

const MUTANT_BAR: f64 = 1e-5;

fn all_spin_mutants() -> [KUStressMutation; 9] {
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

/// Production-wired spin-specific, XC and exchange-scale defects against the
/// Gamma UKS stress at 1x1x1 (H3, PBE0, Ewald): the anchor that sees BOTH the
/// spin defects (Da != Db) and the a-scaling defects (a = 0.25, Madelung on).
/// Each must miss the Gamma reference by far more than the Gamma bar (the
/// correct residual is <= BAR_GAMMA = 1e-9; the mutant bar is 1e-5).
#[test]
fn mutants_are_caught_by_the_gamma_anchor() {
    use KStressMutation as S;
    use KXcStressMutation as X;
    let f = "PBE0";
    let gam = gamma_side(&cell_of(&H3), f);
    let fx = fixture(&H3, G111, f, Jk::Dense, ExxDiv::Ewald);
    let sigma_gam = gam.de_deps;
    let check = |label: String, s: KUksStress| {
        let d = mdiff(&s.de_deps, &sigma_gam);
        eprintln!("1x1x1 vs Gamma under {label}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{label} survives the Gamma anchor: {d:e}");
    };
    // Control: the unmutated stress passes the same comparison.
    let d0 = mdiff(&stress(&fx).de_deps, &sigma_gam);
    assert!(d0 < BAR_GAMMA, "{d0:e}");
    for sm in all_spin_mutants() {
        check(format!("{sm:?}"), stress_mut(&fx, None, Some(sm), None));
    }
    for xm in [
        X::NoAo,
        X::NoPointMotion,
        X::NoWeightDeriv,
        X::NoGga,
        X::NoHessian,
        X::NoStrainMoment,
        X::WrongFactor,
        X::NoInvNk,
        X::PulayNoVxc,
        X::ExchUnscaled,
        X::ExchScaledTwice,
    ] {
        // NoInvNk is an identity at one k-point (1/N_k = 1): excluded.
        if xm == X::NoInvNk {
            assert_eq!(
                mdiff(
                    &stress_mut(&fx, Some(xm), None, None).de_deps,
                    &stress(&fx).de_deps
                ),
                0.0
            );
            continue;
        }
        check(format!("{xm:?}"), stress_mut(&fx, Some(xm), None, None));
    }
    for hm in [
        S::MadelungUnscaled,
        S::CrossSpinK,
        S::SwapSpinFock,
        S::NoPulay,
    ] {
        check(format!("{hm:?}"), stress_mut(&fx, None, None, Some(hm)));
    }
}

/// The k-dependent defects need nk > 1: against the unmutated 1x1x3 stress
/// (itself anchored by the supercell and the FD). The correct residuals are
/// <= 1e-9; every mutant below misses by far more.
#[test]
#[ignore = "7 PBE0 UKS stresses on 1x1x3; ~120 s"]
fn k_dependent_mutants_are_caught_on_the_mesh() {
    use KXcStressMutation as X;
    let fx = fixture(&H3, G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let base = stress(&fx);
    for (xm, sm, hm) in [
        (Some(X::NoInvNk), None, None),
        (Some(X::NoPhase), None, None),
        (Some(X::ConjPhase), None, None),
        (Some(X::KWrongDensity), None, None),
        (None, None, Some(KStressMutation::GammaMadelung)),
        (None, None, Some(KStressMutation::CrossSpinK)),
        (None, Some(KUStressMutation::XcSpinSwap), None),
    ] {
        let d = mdiff(&stress_mut(&fx, xm, sm, hm).de_deps, &base.de_deps);
        eprintln!("1x1x3 {xm:?}/{sm:?}/{hm:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{xm:?}/{sm:?}/{hm:?} survives: {d:e}");
    }
}

/// Uniform grid: no partition (symmetric stress to roundoff) and the volume
/// weight term; FD of the uniform-grid energy.
#[test]
#[ignore = "uniform-grid FD (2 components x 2 SCFs); needs a quiet box"]
fn uniform_grid_stress_is_symmetric_and_matches_fd() {
    use ferric_pbc::kuks::solve_kuks_on_grid;
    let n = [24, 24, 24];
    let run = |cell: &Cell| {
        let prep = prep_for(cell, &pyscf_sto3g_h());
        let mesh = KPointMesh::gamma_centred(cell, [1, 1, 3]).unwrap();
        let cfg = kuks_cfg(cell, "PBE", Jk::Dense, ExxDiv::None);
        let grid = ferric_pbc::PeriodicGrid::uniform(cell, n).unwrap();
        solve_kuks_on_grid(cell, &prep, None, &mesh, &grid, &cfg).unwrap()
    };
    let cell = cell_of(&H3);
    let r = run(&cell);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::None,
        &Default::default(),
    )
    .unwrap();
    let mut cfg = scfg(None, None, None, None);
    cfg.base.grid = KXcGrid::Uniform(n);
    let s = kpoint_uks_stress(
        &cell,
        &prep,
        &mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &hk,
        KStressJk::Dense(&eri),
        &r.scf,
        "PBE",
        &cfg,
    )
    .unwrap();
    eprintln!("uniform-grid antisymmetry {:.2e}", s.antisymmetry());
    assert!(s.antisymmetry() < 1e-9);
    let mut worst = 0.0_f64;
    for (i, j) in [(2, 2), (0, 2)] {
        let e = |h: f64| run(&strained(&cell, i, j, h)).scf.energy;
        let d = (e(FD_H) - e(-FD_H)) / (2.0 * FD_H);
        worst = worst.max((s.de_deps[i][j] - d).abs());
    }
    eprintln!("uniform-grid UKS stress vs FD worst {worst:.2e}");
    assert!(worst < FD_BAR, "{worst:e}");
}
