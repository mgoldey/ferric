//! Analytic stress of the k-point closed-shell RKS energy for GLOBAL HYBRID
//! functionals (`ferric_pbc::kxcstress`, module doc "HF part with exchange
//! fraction a"). Written before measuring; predictions are stated at each test.
//!
//! Anchors: a = 1 with a zero XC == the k-point HF stress PER PART (dense and
//! RS-GDF, exxdiv none and ewald; checks the KS entry's F rebuild and exxdiv
//! routing ONLY, it is blind to a-scaling); a = 0 PBE reproduces values
//! recorded from the pure-GGA entry point before this change (origin/main
//! a12a526f) and ignores the configured exxdiv; the wrong-a Fock guard; the
//! stress at a fixed density is affine in a; 1x1x1 and the non-TRIM
//! 1x1x3 supercell identities against the Gamma hybrid stress per part; central
//! FD of the PBE0 / B3LYP energy under symmetric strain at fixed fractional
//! coordinates with a shared explicit neighbour cutoff (14 Bohr), on 1x1x3 and a
//! Monkhorst-Pack mesh; the trace against isotropic scaling; thread
//! bit-identity; the Fock guard from every side, typed refusals, the budget
//! gate, and production-wired mutants.
//!
//! NOT covered: p/d shells on the hybrid K walks (every fixture is H2/STO-3G),
//! range-separated hybrids / meta-GGA / VV10 (refused), heavy atoms / ECP with a
//! hybrid, open shell, range-split RS-GDF above lambda = 1, no external (PySCF)
//! anchor (its k stress refuses hybrids).

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::dft::{gamma_rks, GammaRksConfig};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::periodic_hcore;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kdft::{solve_krks, KRksConfig, KRksResult};
use ferric_pbc::kgrad::KRsGdfGradSource;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{
    solve_krks_injected, KJkKind, KPointInjection, KPointJk, KPointXc, KScfResult,
};
use ferric_pbc::kstress::{
    kpoint_rhf_stress, KStress, KStressConfig, KStressJk, KStressMutation, KStressParts,
};
use ferric_pbc::kxcgrad::KXcGrid;
use ferric_pbc::kxcstress::{
    kpoint_null_xc_stress, kpoint_rks_stress, KRksStress, KRksStressConfig, KXcStressMutation,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::stress::{gamma_rks_stress, GammaStress, GammaStressConfig, Mat3};
use ferric_pbc::uhf::GammaUhfIntegrals;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
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
/// Neighbour cutoff shared by every energy and stress here (the default depends
/// on the cell and puts a one-step discontinuity in the energy).
const CUTOFF: f64 = 14.0;
const ANCHOR_PRECISION: f64 = 1e-12;
const MUTANT_BAR: f64 = 1e-5;
/// Pure-PBE stress values recorded from `kpoint_rks_stress` at origin/main
/// a12a526f (before the hybrid change), same fixture (H2/STO-3G, 1x1x3, atom
/// grid (20, 50), cutoff 14, SCF conv 1e-14/1e-12), row-major dE/de or part.
const RECORDED_PBE: [(&str, [f64; 9]); 20] = [
    (
        "Dense.de_deps",
        [
            4.010711307399635e-1,
            4.4902724466232616e-4,
            -2.9462120889361433e-4,
            4.636342200861118e-4,
            4.003980158558805e-1,
            6.720738539802175e-4,
            -4.968238943049473e-3,
            7.962698114706809e-3,
            -1.5803056463435128e-1,
        ],
    ),
    (
        "Dense.xc_ao",
        [
            8.04684166652521e-1,
            -7.476772833604858e-4,
            1.1961739969847167e-2,
            -7.501605143873572e-4,
            8.059203235374971e-1,
            -1.916186130662964e-2,
            1.2255646230276224e-2,
            -1.9613738497529464e-2,
            7.986759840774635e-1,
        ],
    ),
    (
        "Dense.xc_weight",
        [
            -1.7564110753833792e-1,
            4.7000714180372766e-4,
            -9.18281686466741e-4,
            4.8709734825438917e-4,
            -1.767230642171948e-1,
            1.6947543153214204e-3,
            -5.885805681051678e-3,
            9.437255766947835e-3,
            -4.913210629865279e-1,
        ],
    ),
    (
        "Dense.overlap",
        [
            -7.759164780641192e-1,
            2.0531535281992797e-4,
            -3.8603137288415286e-3,
            2.0531535281993865e-4,
            -7.764212190956046e-1,
            6.172989350592111e-3,
            -3.8603137288415116e-3,
            6.1729893505921025e-3,
            -2.445969275780899e-1,
        ],
    ),
    (
        "Dense.coulomb",
        [
            5.05200760927883e-2,
            2.7411530885752757e-4,
            -4.49188564551051e-3,
            2.74115308857527e-4,
            4.986078388300305e-2,
            7.184481461108251e-3,
            -4.491885645510508e-3,
            7.184481461108253e-3,
            -8.054010397693509e-2,
        ],
    ),
    (
        "Dense.kinetic",
        [
            7.9184348569861e-2,
            6.158489592536247e-4,
            -1.0609369104188708e-2,
            6.158489592536112e-4,
            7.84112743752226e-2,
            1.6972496030279657e-2,
            -1.0609369104188722e-2,
            1.6972496030279647e-2,
            -4.123147923730064e-1,
        ],
    ),
    (
        "Dense.vsr",
        [
            4.8875263830674864e-1,
            -1.2101954834874264e-3,
            2.0412972954399266e-2,
            -1.2101954834874281e-3,
            4.90403326258911e-1,
            -3.2654987367694586e-2,
            2.0412972954399287e-2,
            -3.265498736769458e-2,
            8.223528041955592e-1,
        ],
    ),
    (
        "Dense.vlr",
        [
            -2.57106946277577e-1,
            -3.9979517292758717e-4,
            5.348820313203268e-3,
            -3.997951729275865e-4,
            -2.558853366163767e-1,
            -8.553784087081734e-3,
            5.348820313203266e-3,
            -8.553784087081734e-3,
            1.1191804280077841e-1,
        ],
    ),
    (
        "Dense.fit_j3_sr",
        [0e0, 0e0, 0e0, 0e0, 0e0, 0e0, 0e0, 0e0, 0e0],
    ),
    (
        "Dense.fit_j2_lr",
        [0e0, 0e0, 0e0, 0e0, 0e0, 0e0, 0e0, 0e0, 0e0],
    ),
    (
        "Gdf.de_deps",
        [
            4.0100722809308875e-1,
            4.490366677818646e-4,
            -2.890504864758955e-4,
            4.636436029263199e-4,
            4.0032724730760794e-1,
            6.646061002407852e-4,
            -4.962591082831434e-3,
            7.955104562427418e-3,
            -1.5805840994317638e-1,
        ],
    ),
    (
        "Gdf.xc_ao",
        [
            8.046833901048287e-1,
            -7.476918415124859e-4,
            1.1962377362562334e-2,
            -7.501751062744654e-4,
            8.059195582520082e-1,
            -1.9162880836277704e-2,
            1.2256287817177688e-2,
            -1.9614764944347832e-2,
            7.986629658569074e-1,
        ],
    ),
    (
        "Gdf.xc_weight",
        [
            -1.7564092676734377e-1,
            4.7000662750726417e-4,
            -9.186363828326085e-4,
            4.870968274136814e-4,
            -1.7672288175399575e-1,
            1.6953229178502732e-3,
            -5.886087433804483e-3,
            9.437705488107587e-3,
            -4.913201754297358e-1,
        ],
    ),
    (
        "Gdf.overlap",
        [
            -7.759564525815044e-1,
            2.0533144340468954e-4,
            -3.8608770011617456e-3,
            2.0533144340468347e-4,
            -7.76461227343407e-1,
            6.173889972001285e-3,
            -3.8608770011617456e-3,
            6.173889972001287e-3,
            -2.4458976706846527e-1,
        ],
    ),
    ("Gdf.coulomb", [0e0, 0e0, 0e0, 0e0, 0e0, 0e0, 0e0, 0e0, 0e0]),
    (
        "Gdf.kinetic",
        [
            7.918249558119686e-2,
            6.158698825676971e-4,
            -1.0610034034480029e-2,
            6.158698825677015e-4,
            7.8409400390637e-2,
            1.6973559674735662e-2,
            -1.0610034034480027e-2,
            1.697355967473565e-2,
            -4.1231419915640705e-1,
        ],
    ),
    (
        "Gdf.vsr",
        [
            4.8875369483799697e-1,
            -1.2102201689441466e-3,
            2.041381005283408e-2,
            -1.2102201689441455e-3,
            4.9040440912489913e-1,
            -3.265632627504214e-2,
            2.0413810052834082e-2,
            -3.265632627504213e-2,
            8.22346865751531e-1,
        ],
    ),
    (
        "Gdf.vlr",
        [
            -2.571065071084841e-1,
            -3.997999903773091e-4,
            5.349056048868368e-3,
            -3.997999903773091e-4,
            -2.558848900479165e-1,
            -8.554161062578867e-3,
            5.349056048868368e-3,
            -8.554161062578867e-3,
            1.119120498030114e-1,
        ],
    ),
    (
        "Gdf.fit_j3_sr",
        [
            -2.677105748066188e-1,
            3.052943501516035e-4,
            -5.24666742084596e-3,
            3.0370437203314975e-4,
            -2.683094830360932e-1,
            8.374514601348026e-3,
            -4.669721472892906e-3,
            7.4797905364478385e-3,
            -2.888502071594871e-1,
        ],
    ),
    (
        "Gdf.fit_j2_lr",
        [
            -1.0235526491247343e-1,
            -6.224648607673893e-5,
            1.2516243120903898e-3,
            -6.0431329694932086e-5,
            -1.0190225394866059e-1,
            -1.9914860512751536e-3,
            6.081631264778192e-4,
            -9.92533119557923e-4,
            -1.1292466467981759e-2,
        ],
    ),
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Jk {
    Dense,
    Gdf,
    /// RS-GDF with the range split (lambda = 1).
    GdfSplit,
}

/// A k mesh: counts and Monkhorst-Pack (shifted for even counts) or
/// Gamma-centred.
#[derive(Clone, Copy, Debug)]
struct Mesh {
    n: [usize; 3],
    mp: bool,
}

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

fn kgdf_cfg_of(jk: Jk) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            range_split: (jk == Jk::GdfSplit).then(|| ferric_pbc::rsgdf::RangeSplit::new(1.0)),
            ..Default::default()
        },
        mutation: None,
    }
}

fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

fn cell_of(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(4.0)).expect("cell")
}

fn krks_cfg(cell: &Cell, functional: &str, jk: Jk, exxdiv: ExxDiv) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, exxdiv, functional);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-14;
    cfg.krhf.scf.grad_conv = 1e-12;
    cfg.krhf.scf.max_iter = 200;
    if jk != Jk::Dense {
        cfg.krhf.jk = KJkKind::RsGdf;
        cfg.krhf.rsgdf = kgdf_cfg_of(jk);
    }
    cfg.grid = grid_cfg();
    cfg
}

/// Fresh SCF (energy) on `cell`.
fn scf_on(cell: &Cell, m: Mesh, functional: &str, jk: Jk, exxdiv: ExxDiv) -> KRksResult {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = mesh_of(cell, m);
    let cfg = krks_cfg(cell, functional, jk, exxdiv);
    let aux = (jk != Jk::Dense).then(|| prep_for(cell, &et_sp_aux()));
    solve_krks(cell, &prep, aux.as_ref(), &mesh, &cfg)
        .unwrap_or_else(|e| panic!("solve_krks {functional}: {e}"))
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
    res: KRksResult,
    jk: Jk,
    exxdiv: ExxDiv,
    functional: String,
}

fn fixture_cell(cell: Cell, m: Mesh, functional: &str, jk: Jk, exxdiv: ExxDiv) -> Fx {
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = mesh_of(&cell, m);
    let res = scf_on(&cell, m, functional, jk, exxdiv);
    let cfg = krks_cfg(&cell, functional, jk, exxdiv);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &cfg.krhf.hcore).unwrap();
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

fn fixture(m: Mesh, functional: &str, jk: Jk, exxdiv: ExxDiv) -> Fx {
    fixture_cell(cell_of(&H2), m, functional, jk, exxdiv)
}

fn scfg(
    exxdiv: Option<ExxDiv>,
    xm: Option<KXcStressMutation>,
    sm: Option<KStressMutation>,
) -> KRksStressConfig {
    KRksStressConfig {
        stress: KStressConfig {
            budget_bytes: Some(AMPLE),
            mutation: sm,
            ..Default::default()
        },
        grid: KXcGrid::Atom(grid_cfg()),
        exxdiv,
        mutation: xm,
        ..Default::default()
    }
}

fn stress_with(
    fx: &Fx,
    functional: &str,
    scf: &KScfResult,
    cfg: &KRksStressConfig,
) -> Result<KRksStress, ferric_core::FerricError> {
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
            kpoint_rks_stress(
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
        Jk::Gdf | Jk::GdfSplit => {
            let kc = kgdf_cfg_of(fx.jk);
            let gdf = KRsGdf::build(&fx.cell, &fx.prep, &fx.aux, &fx.mesh, &fx.hk.s, &kc)
                .unwrap()
                .with_exxdiv(fx.exxdiv);
            kpoint_rks_stress(
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

fn stress(fx: &Fx) -> KRksStress {
    stress_with(
        fx,
        &fx.functional,
        &fx.res.scf,
        &scfg(Some(fx.exxdiv), None, None),
    )
    .expect("kpoint_rks_stress")
}

fn stress_mut(fx: &Fx, xm: Option<KXcStressMutation>, sm: Option<KStressMutation>) -> KRksStress {
    stress_with(
        fx,
        &fx.functional,
        &fx.res.scf,
        &scfg(Some(fx.exxdiv), xm, sm),
    )
    .expect("kpoint_rks_stress")
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

/// Central FD of the SCF energy for the listed components at step `h`.
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

struct ZeroXc(usize);
impl KPointXc for ZeroXc {
    fn build(
        &mut self,
        dm: &[Array2<Complex64>],
    ) -> Result<(f64, Vec<Array2<Complex64>>), ferric_core::FerricError> {
        Ok((
            0.0,
            dm.iter().map(|_| Array2::zeros((self.0, self.0))).collect(),
        ))
    }
}

struct Zx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    aux: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    scf: KScfResult,
    jk: Jk,
    exxdiv: ExxDiv,
}

fn dense_cfg() -> KDenseAftConfig {
    KDenseAftConfig {
        precision: ANCHOR_PRECISION,
        ..Default::default()
    }
}

/// Hartree + `a`/2 exact-exchange k SCF with a zero XC (a = 1 is k-point HF).
fn zero_xc(m: Mesh, jk: Jk, exxdiv: ExxDiv, a: f64) -> Zx {
    let cell = cell_of(&H2);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = prep_for(&cell, &et_sp_aux());
    let mesh = mesh_of(&cell, m);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).unwrap();
    let mut sc = ferric_pbc::kscf::KScfConfig::default();
    sc.energy_conv = 1e-13;
    sc.grad_conv = 1e-10;
    sc.max_iter = 200;
    let eri;
    let gdf;
    let jkb: Box<dyn KPointJk + '_> = match jk {
        Jk::Dense => {
            eri = KDenseAftEri::build(&cell, &prep, &mesh, &hk.s, exxdiv, &dense_cfg()).unwrap();
            Box::new(eri.jk_builder())
        }
        Jk::Gdf | Jk::GdfSplit => {
            gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kgdf_cfg_of(jk))
                .unwrap()
                .with_exxdiv(exxdiv);
            Box::new(gdf.jk_builder())
        }
    };
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: jkb,
    };
    let mut z = ZeroXc(prep.nbasis());
    let scf = solve_krks_injected(&cell, &mesh, &sc, inj, &mut z, a).unwrap();
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
        Jk::Gdf | Jk::GdfSplit => {
            let kc = kgdf_cfg_of(z.jk);
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

/// J(k), K(k) (K with the build's Madelung shift) of the SCF density.
fn jk_mats(z: &Zx) -> (Vec<Array2<Complex64>>, Vec<Array2<Complex64>>) {
    let n = z.prep.nbasis();
    let nk = z.mesh.nk();
    let mut j: Vec<Array2<Complex64>> = (0..nk).map(|_| Array2::zeros((n, n))).collect();
    let mut k = j.clone();
    match z.jk {
        Jk::Dense => {
            let eri =
                KDenseAftEri::build(&z.cell, &z.prep, &z.mesh, &z.hk.s, z.exxdiv, &dense_cfg())
                    .unwrap();
            eri.jk_builder()
                .build(&z.scf.densities, &mut j, &mut k)
                .unwrap();
        }
        Jk::Gdf | Jk::GdfSplit => {
            let gdf = KRsGdf::build(
                &z.cell,
                &z.prep,
                &z.aux,
                &z.mesh,
                &z.hk.s,
                &kgdf_cfg_of(z.jk),
            )
            .unwrap()
            .with_exxdiv(z.exxdiv);
            gdf.jk_builder()
                .build(&z.scf.densities, &mut j, &mut k)
                .unwrap();
        }
    }
    (j, k)
}

fn null_stress(z: &Zx, a: f64, scf: &KScfResult, cfg: &KRksStressConfig) -> KRksStress {
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    with_jk(z, |j| {
        kpoint_null_xc_stress(&z.cell, &z.prep, &z.mesh, &hcfg, &z.hk, j, scf, a, cfg).unwrap()
    })
}

/// (1) WIRING ANCHOR, per part. a = 1 with a zero XC remainder IS the k-point
/// HF stress: same `assemble`, but the KS entry rebuilds F itself (h + J - K/2
/// with the Madelung shift, exact guard) and takes its exxdiv from the config.
/// Prediction: every part equal to roundoff (F is hermitised on one side only).
/// WHAT THIS CHECKS: the KS entry's F rebuild, the guard and the exxdiv routing
/// (a leaked XC term, a dropped Madelung shift in F, a wrong exxdiv). WHAT IT IS
/// BLIND TO: any a-scaling defect, since every scale is 1 here
/// (`MadelungUnscaled` and `ExchScaledTwice` are identities at a = 1). The
/// a-scaling is covered by the affine-in-a test, the 1x1x1 and supercell Gamma
/// anchors, the FD tests and the mutants.
#[test]
fn a_equal_one_with_zero_xc_is_the_hf_stress_per_part() {
    a_equal_one_case(Jk::Dense);
}

#[test]
#[ignore = "RS-GDF rows of the a = 1 anchor (2 fitted zero-XC SCFs + 4 stresses); measured ~90 s"]
fn a_equal_one_with_zero_xc_is_the_hf_stress_per_part_rsgdf() {
    a_equal_one_case(Jk::Gdf);
}

fn a_equal_one_case(jk: Jk) {
    {
        for exxdiv in [ExxDiv::None, ExxDiv::Ewald] {
            let z = zero_xc(G113, jk, exxdiv, 1.0);
            let ks = null_stress(&z, 1.0, &z.scf, &scfg(Some(exxdiv), None, None));
            let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
            let hf: KStress = with_jk(&z, |j| {
                kpoint_rhf_stress(
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
            for ((n, pa), (_, pg)) in hf_parts(&ks.hf.parts).into_iter().zip(hf_parts(&hf.parts)) {
                let d = mdiff(&pa, &pg);
                worst = worst.max(d);
                assert!(d < 1e-13, "{jk:?} {exxdiv:?} part {n}: {d:e}");
            }
            let dt = mdiff(&ks.de_deps, &hf.de_deps);
            eprintln!(
                "{jk:?} {exxdiv:?}: a=1 null-XC vs HF stress: worst part {worst:.1e}, total {dt:.1e}"
            );
            assert!(dt < 1e-13);
            // Not vacuous.
            assert!(mmax(&hf.de_deps) > 1e-3);
            match jk {
                Jk::Dense => assert!(mmax(&hf.parts.exchange) > 1e-4),
                _ => assert!(mmax(&hf.parts.fit_j3_sr) > 1e-4),
            }
            if exxdiv == ExxDiv::Ewald {
                assert!(mmax(&hf.parts.madelung) > 1e-6, "Madelung is exercised");
            }
        }
    }
}

/// (2) a = 0 REPRODUCES THE RECORDED PURE-GGA STRESS. The PBE stress of the
/// pre-change `kpoint_rks_stress` (origin/main a12a526f, same fixture) was
/// recorded to full precision (`RECORDED_PBE`: total dE/de, xc_ao, xc_weight and
/// the overlap, coulomb, kinetic, vsr, vlr and (RS-GDF) fit parts); the current
/// code, with the hybrid machinery armed (`exxdiv` configured), must reproduce
/// each. Prediction 1e-12 or better (same arithmetic, same SCF); bar 1e-10: the
/// two builds differ in no a = 0 arithmetic, and any exchange leak at a = 0
/// (a K walk run, a Madelung term left in, a shifted F) moves these by
/// >= 1e-2 (the a = 1 exchange parts are of that size). The exxdiv is also
/// asserted to be IGNORED at a = 0 (bitwise across none/Ewald), and the
/// exchange and Madelung parts to be exactly 0. The merged suite
/// `pbc_kxcstress.rs` (FD, Gamma, supercell, PySCF anchors) pins the pure value
/// itself.
#[test]
fn a_zero_reproduces_the_recorded_pure_gga_stress() {
    a_zero_case(Jk::Dense);
}

#[test]
#[ignore = "RS-GDF row of the a = 0 recorded-value anchor; measured ~25 s"]
fn a_zero_reproduces_the_recorded_pure_gga_stress_rsgdf() {
    a_zero_case(Jk::Gdf);
}

fn a_zero_case(jk: Jk) {
    let fx = fixture(G113, "PBE", jk, ExxDiv::None);
    let run =
        |e: Option<ExxDiv>| stress_with(&fx, "PBE", &fx.res.scf, &scfg(e, None, None)).unwrap();
    let base = run(None);
    let tag = format!("{jk:?}");
    let part = |name: &str, s: &KRksStress| -> Mat3 {
        let p = &s.hf.parts;
        match name {
            "de_deps" => s.de_deps,
            "xc_ao" => s.xc_ao,
            "xc_weight" => s.xc_weight,
            "overlap" => p.overlap(),
            "coulomb" => p.coulomb,
            "kinetic" => p.kinetic,
            "vsr" => p.vsr,
            "vlr" => p.vlr,
            "fit_j3_sr" => p.fit_j3_sr,
            "fit_j2_lr" => p.fit_j2_lr,
            other => panic!("unknown recorded part {other}"),
        }
    };
    let mut worst = 0.0_f64;
    let mut n = 0;
    for e in [None, Some(ExxDiv::None), Some(ExxDiv::Ewald)] {
        let s = run(e);
        for (key, vals) in RECORDED_PBE.iter().filter(|(k, _)| k.starts_with(&tag)) {
            let name = &key[tag.len() + 1..];
            let want = [
                [vals[0], vals[1], vals[2]],
                [vals[3], vals[4], vals[5]],
                [vals[6], vals[7], vals[8]],
            ];
            let d = mdiff(&part(name, &s), &want);
            worst = worst.max(d);
            n += 1;
            assert!(d < 1e-10, "{jk:?} {e:?} {name}: {d:e}");
        }
        assert_eq!(
            s.de_deps, base.de_deps,
            "{jk:?} {e:?}: exxdiv must be ignored"
        );
    }
    assert!(n >= 24, "recorded parts were compared ({n})");
    assert_eq!(mmax(&base.hf.parts.exchange), 0.0);
    assert_eq!(mmax(&base.hf.parts.madelung), 0.0);
    eprintln!("{jk:?}: a = 0 vs recorded pure-GGA values, worst {worst:.2e} over {n} comparisons");
}

/// (3) AFFINITY IN a at a FIXED density (null XC, a = 1 density, the stored
/// Fock replaced by the rebuilt F(a) so the guard passes). The two-electron
/// energy is J + a K and F = h + J - (a/2) K, so dE/de is affine in a.
/// WHAT THIS CATCHES: a-NONLINEAR defects (a applied twice). WHAT IT DOES NOT:
/// an unscaled Madelung or an unscaled piece are still affine; the Gamma, FD
/// and mutant tests pin those.
#[test]
#[ignore = "5 null-XC stresses (~32 s); the a^2 defect it targets is also caught in the default set by the ExchScaledTwice mutant"]
fn stress_is_affine_in_the_exchange_fraction_at_fixed_density() {
    affine_case(Jk::Dense);
}

#[test]
#[ignore = "RS-GDF row of the affinity anchor (6 null-XC stresses); measured ~60 s"]
fn stress_is_affine_in_the_exchange_fraction_at_fixed_density_rsgdf() {
    affine_case(Jk::Gdf);
}

fn affine_case(jk: Jk) {
    {
        let z = zero_xc(G113, jk, ExxDiv::Ewald, 1.0);
        let (j, k) = jk_mats(&z);
        let at = |a: f64| -> Mat3 {
            let mut scf = z.scf.clone();
            for q in 0..scf.fock.len() {
                let mut f = &z.hk.h[q] + &j[q];
                f.scaled_add(Complex64::new(-0.5 * a, 0.0), &k[q]);
                scf.fock[q] = f;
            }
            null_stress(&z, a, &scf, &scfg(Some(ExxDiv::Ewald), None, None)).de_deps
        };
        let (s0, s1) = (at(0.0), at(1.0));
        let span = mdiff(&s1, &s0);
        let mut worst = 0.0_f64;
        for a in [0.25, 0.5, 0.75] {
            let want = madd(&mscale(&s0, 1.0 - a), &mscale(&s1, a));
            worst = worst.max(mdiff(&at(a), &want));
        }
        eprintln!("{jk:?}: |S(1)-S(0)| {span:.3e}; affine residual {worst:.2e}");
        assert!(span > 1e-3);
        // Bar from both sides: the correct residual is 4.9e-15 (dense) and
        // 1.8e-12 (RS-GDF, fit roundoff); the a^2 defect misses by
        // (a - a^2) |S(1) - S(0)| = 0.19 * 0.33 ~ 6e-2 at a = 0.25 (by
        // arithmetic from the measured span; the production mutant itself is
        // exercised in `smoke_mutants_are_caught`, miss 1.6e-1 on PBE0).
        assert!(worst < 1e-10, "{jk:?}: {worst:e}");
    }
}

/// (3b) The Fock guard sees a wrong `a` ALONE. Null-XC SCF at a = 0.5 (Ewald),
/// stress asked at a different a with the guard on. The rebuilt F differs by
/// (da/2) K, |K| ~ 1, so the 1e-8 guard bar separates a correct a (mismatch at
/// roundoff) from da = 1e-6 (mismatch ~ 5e-7 > bar) by orders of magnitude on
/// both sides. Prediction: a = 0.5 passes with mismatch < 1e-12; a = 0.5 + 1e-6
/// and a = 0.25 are refused with the Fock-guard message.
#[test]
fn the_guard_refuses_a_wrong_exchange_fraction_alone() {
    let z = zero_xc(G113, Jk::Dense, ExxDiv::Ewald, 0.5);
    let cfg = scfg(Some(ExxDiv::Ewald), None, None);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let run = |a: f64| {
        with_jk(&z, |jk| {
            kpoint_null_xc_stress(&z.cell, &z.prep, &z.mesh, &hcfg, &z.hk, jk, &z.scf, a, &cfg)
        })
    };
    let ok = run(0.5).expect("the correct a passes");
    eprintln!(
        "a = 0.5 (correct): fock mismatch {:.2e} (bar 1e-8)",
        ok.hf.fock_mismatch
    );
    assert!(ok.hf.fock_mismatch < 1e-12, "{:e}", ok.hf.fock_mismatch);
    for bad in [0.5 + 1e-6, 0.25] {
        let m = run(bad)
            .err()
            .unwrap_or_else(|| panic!("a = {bad} accepted"))
            .to_string();
        eprintln!("a = {bad}: {m}");
        assert!(m.contains("differs"), "{m}");
    }
}

// ------------------------------------------------------------------- Gamma

fn gamma_side(cell: &Cell, functional: &str, exxdiv: ExxDiv) -> GammaStress {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hc = periodic_hcore(cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        cell,
        &prep,
        &hc.s,
        exxdiv,
        ANCHOR_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let mut gc = GammaRksConfig::new(functional);
    gc.grid = grid_cfg();
    gc.exxdiv = exxdiv;
    gc.scf.energy_conv = 1e-13;
    gc.scf.density_conv = 1e-10;
    gc.scf.max_iter = 200;
    let r = gamma_rks(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &gc).unwrap();
    gamma_rks_stress(
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
fn gamma_rows(k: &KRksStress, g: &GammaStress, n_cells: f64) -> Vec<(&'static str, f64)> {
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

/// (5a) 1x1x1 == the Gamma hybrid KS stress, PER PART (the total can hide a
/// defect of a part's size), exxdiv none and ewald. Independent construction:
/// real lattice sums vs complex Bloch AOs, the Gamma code's own scaling of
/// J and K, its own SCF. Floor: both SCFs converge to ~1e-10 in the density.
/// Prediction: <= 1e-10 per part; the madelung row is exactly 0 for none.
/// MEASURED (PBE0 none/Ewald, B3LYP Ewald): xc_ao 2.2e-16, xc_weight 5.6e-17,
/// overlap 4.1e-13..4.3e-13, kinetic <= 5.6e-17, vsr <= 1.3e-15, vlr 1.8e-15,
/// c0_volume 0, J+K 1.85e-12..1.90e-12, madelung <= 6.9e-18, sigma 2.4e-14.
#[test]
fn one_point_mesh_matches_the_gamma_hybrid_stress_per_part() {
    one_point_case(&[("PBE0", ExxDiv::Ewald)]);
}

#[test]
#[ignore = "the other rows (PBE0 none, B3LYP ewald) of the 1x1x1 Gamma anchor; measured ~20 s"]
fn one_point_mesh_matches_the_gamma_hybrid_stress_per_part_other_rows() {
    one_point_case(&[("PBE0", ExxDiv::None), ("B3LYP", ExxDiv::Ewald)]);
}

fn one_point_case(rows: &[(&str, ExxDiv)]) {
    for &(f, exxdiv) in rows {
        let gam = gamma_side(&cell_of(&H2), f, exxdiv);
        let fx = fixture(
            Mesh {
                n: [1, 1, 1],
                mp: false,
            },
            f,
            Jk::Dense,
            exxdiv,
        );
        let k = stress(&fx);
        for (n, d) in gamma_rows(&k, &gam, 1.0) {
            eprintln!("{f} {exxdiv:?} 1x1x1 vs Gamma: {n:<10} {d:.2e}");
            assert!(d < 1e-10, "{f} {exxdiv:?} {n}: {d:e}");
        }
        assert!(mmax(&k.hf.parts.exchange) > 1e-4, "K walks exercised");
        if exxdiv == ExxDiv::Ewald {
            assert!(mmax(&k.hf.parts.madelung) > 1e-6);
        }
    }
}

/// Explicit diag(n) supercell of `pos` (copy-major, copy 0 first).
fn supercell_pos(pos: &[[f64; 3]], a: f64, n: [usize; 3]) -> Cell {
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
    Cell::new(hydrogens(&atoms), lat).unwrap()
}

/// (5b) SUPERCELL ANCHOR on the non-TRIM 1x1x3 mesh: the k stress per cell is
/// the Gamma stress of the explicit 3-copy supercell (sigma equal, dE/de / 3),
/// per part, exxdiv none and ewald. With Ewald the supercell Gamma Madelung
/// constant IS the mesh constant, so a primitive-cell v_M in the M-term or its
/// strain misses here (the `GammaMadelung*` mutants). Same explicit cutoff on
/// both sides. Prediction <= 1e-10 per part. MEASURED (none / Ewald): xc_ao
/// 6.3e-14 / 6.7e-14, xc_weight 5.1e-15, overlap 1.9e-13, kinetic 1.3e-14, vsr
/// 2.0e-13, vlr 3.6e-14 / 3.8e-14, J+K 1.25e-12, madelung 0 / 2.8e-17, sigma
/// 2.1e-14.
#[test]
fn k_mesh_hybrid_stress_is_the_gamma_supercell_stress_per_part() {
    supercell_case(ExxDiv::Ewald);
}

#[test]
#[ignore = "second exxdiv row of the supercell anchor (6-atom Gamma SCF); measured ~30 s"]
fn k_mesh_hybrid_stress_is_the_gamma_supercell_stress_per_part_exxdiv_none() {
    supercell_case(ExxDiv::None);
}

fn supercell_case(exxdiv: ExxDiv) {
    let f = "PBE0";
    let gam = gamma_side(&supercell_pos(&H2, 4.0, [1, 1, 3]), f, exxdiv);
    let fx = fixture(G113, f, Jk::Dense, exxdiv);
    let k = stress(&fx);
    for (n, d) in gamma_rows(&k, &gam, 3.0) {
        eprintln!("{f} {exxdiv:?} 1x1x3 vs supercell: {n:<10} {d:.2e}");
        assert!(d < 1e-10, "{f} {exxdiv:?} {n}: {d:e}");
    }
    if exxdiv == ExxDiv::Ewald {
        assert!(mmax(&k.hf.parts.madelung) > 1e-6);
    }
}

// ----------------------------------------------------------------------- FD

/// FD step and bar of the atom-grid hybrid anchors. The step and the floor come
/// from `fd_window_probe_pbe0` (shared cutoff 14, PBE0, 1x1x3, dense Ewald,
/// all nine components): worst |FD - analytic| 1.5e-9 at h = 1e-5 (energy
/// roundoff / 2h), 1.0e-9 at 2.5e-5, 4.6e-9 at 5e-5, 2.0e-8 at 1e-4 (h^2
/// truncation); no isolated jump at cutoff 14. The bar is 30x the floor at the
/// chosen step; the smallest production-wired mutant misses by 1.1e-2.
const FD_H: f64 = 2.5e-5;
const FD_BAR: f64 = 3e-8;

/// (3) FD of ferric's own PBE0 energy under symmetric strain at fixed
/// fractional coordinates (fresh SCF per displaced cell, the mesh rebuilt on the
/// strained cell with the same integer numerators). Three components covering
/// the diagonal, a shear, and the axis of the mesh. Prediction: analytic and FD
/// agree at the floor; artifact hypothesis: a stress that is the right stress
/// of a different energy (an unscaled exchange or Madelung, a stale K) misses
/// at 1e-3..1e-1, not at the floor.
#[test]
fn pbe0_stress_matches_fd_dense_ewald_1x1x3() {
    let fx = fixture(G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let s = stress(&fx);
    let comps = [(2, 2), (0, 2), (0, 0)];
    let fd = fd_components(
        &fx.cell,
        G113,
        "PBE0",
        Jk::Dense,
        ExxDiv::Ewald,
        FD_H,
        &comps,
    );
    let mut worst = 0.0_f64;
    for (&(i, j), d) in comps.iter().zip(&fd) {
        let miss = (s.de_deps[i][j] - d).abs();
        eprintln!(
            "PBE0 dense ewald eps({i},{j}): analytic {:.9e} FD {d:.9e} miss {miss:.2e}",
            s.de_deps[i][j]
        );
        worst = worst.max(miss);
    }
    assert!(worst < FD_BAR, "{worst:e}");
}

/// (4) Pressure from isotropic scaling: sum_a dE/de_aa equals the FD of E under
/// eps = h*I (dense, Ewald, B3LYP; the other functional of the pair). Bar
/// 3 * FD_BAR: the trace error is the sum of three diagonal-component errors,
/// each allowed FD_BAR; measured 1.6e-9 (three components at the all-nine floor
/// of ~1e-9 would give 3e-9).
#[test]
fn trace_matches_isotropic_scaling_fd() {
    let fx = fixture(G113, "B3LYP", Jk::Dense, ExxDiv::Ewald);
    let s = stress(&fx);
    let tr = s.de_deps[0][0] + s.de_deps[1][1] + s.de_deps[2][2];
    let iso = |h: f64| {
        let e = [[h, 0.0, 0.0], [0.0, h, 0.0], [0.0, 0.0, h]];
        let c = fx.cell.strained(&e).unwrap();
        energy_on(&c, G113, "B3LYP", Jk::Dense, ExxDiv::Ewald)
    };
    let fd = (iso(FD_H) - iso(-FD_H)) / (2.0 * FD_H);
    eprintln!(
        "B3LYP trace: analytic {tr:.9e} FD(isotropic) {fd:.9e} miss {:.2e}",
        (tr - fd).abs()
    );
    assert!((tr - fd).abs() < 3.0 * FD_BAR);
}

/// PROBE: choose the FD step. All nine components of PBE0 (dense, Ewald,
/// 1x1x3) at four steps; the printout is the floor and the window.
#[test]
#[ignore = "probe: 4 steps x 18 SCFs, needs a quiet box; run with --nocapture"]
fn fd_window_probe_pbe0() {
    let fx = fixture(G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
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

/// All nine components of PBE0 and B3LYP, every route and exxdiv.
#[test]
#[ignore = "all-nine FD for 6 configurations x 18 SCFs; needs a quiet box"]
fn hybrid_stress_matches_fd_all_nine_every_route() {
    for (f, jk, exxdiv) in [
        ("PBE0", Jk::Dense, ExxDiv::None),
        ("PBE0", Jk::Dense, ExxDiv::Ewald),
        ("B3LYP", Jk::Dense, ExxDiv::Ewald),
        ("PBE0", Jk::Gdf, ExxDiv::None),
        ("PBE0", Jk::Gdf, ExxDiv::Ewald),
        ("PBE0", Jk::GdfSplit, ExxDiv::Ewald),
    ] {
        let fx = fixture(G113, f, jk, exxdiv);
        let s = stress(&fx);
        let fd = fd_components(&fx.cell, G113, f, jk, exxdiv, FD_H, &ALL9);
        let mut worst = 0.0_f64;
        for (&(i, j), d) in ALL9.iter().zip(&fd) {
            worst = worst.max((d - s.de_deps[i][j]).abs());
        }
        eprintln!("{f} {jk:?} {exxdiv:?}: all-nine FD miss {worst:.2e}");
        assert!(worst < FD_BAR, "{f} {jk:?} {exxdiv:?}: {worst:e}");
    }
}

/// Monkhorst-Pack 2x2x2 (shifted, even: no supercell anchor exists, FD is the
/// only one). Three components, dense Ewald and RS-GDF none.
#[test]
#[ignore = "8-point shifted mesh, 3 components x 2 routes x 2 SCFs; needs a quiet box"]
fn pbe0_stress_matches_fd_on_the_monkhorst_pack_mesh() {
    let m = Mesh {
        n: [2, 2, 2],
        mp: true,
    };
    for (jk, exxdiv) in [(Jk::Dense, ExxDiv::Ewald), (Jk::Gdf, ExxDiv::None)] {
        let fx = fixture(m, "PBE0", jk, exxdiv);
        let s = stress(&fx);
        let comps = [(2, 2), (0, 1), (1, 1)];
        let fd = fd_components(&fx.cell, m, "PBE0", jk, exxdiv, FD_H, &comps);
        let mut worst = 0.0_f64;
        for (&(i, j), d) in comps.iter().zip(&fd) {
            worst = worst.max((d - s.de_deps[i][j]).abs());
        }
        eprintln!("MP 2x2x2 {jk:?} {exxdiv:?}: FD miss {worst:.2e}");
        assert!(worst < FD_BAR, "{jk:?} {exxdiv:?}: {worst:e}");
    }
}

/// RS-GDF (unsplit) PBE0, Ewald: FD of the energy (two components) and the
/// RS-GDF a-scaling mutant in the default set. `ExchUnscaled` runs the fit's
/// exchange pieces and the Madelung at a = 1: the stress is the right stress of
/// a different energy, so it must miss the FD-anchored base by far more than
/// the bar 1e-5 (measured 6.6e-1; the correct residual is ~2e-10).
#[test]
fn pbe0_rsgdf_stress_matches_fd_and_catches_the_unscaled_fit_exchange() {
    let fx = fixture(G113, "PBE0", Jk::Gdf, ExxDiv::Ewald);
    let s = stress(&fx);
    let comps = [(2, 2), (0, 2)];
    let fd = fd_components(&fx.cell, G113, "PBE0", Jk::Gdf, ExxDiv::Ewald, FD_H, &comps);
    let mut worst = 0.0_f64;
    for (&(i, j), d) in comps.iter().zip(&fd) {
        worst = worst.max((d - s.de_deps[i][j]).abs());
    }
    eprintln!("PBE0 RS-GDF ewald: FD miss {worst:.2e}");
    assert!(worst < FD_BAR, "{worst:e}");
    let d = miss(&fx, Some(KXcStressMutation::ExchUnscaled), None, &s);
    eprintln!("RS-GDF ExchUnscaled: miss {d:.2e}");
    assert!(d > MUTANT_BAR, "ExchUnscaled survives on RS-GDF: {d:e}");
}

// ----------------------------------------------------- threads and refusals

#[test]
#[ignore = "two full PBE0 stresses at 1 and 4 threads (~28 s); the default set is held to ~300 s, and thread bit-identity of the shared assembly is also pinned by pbc_parallel_bitwise"]
fn hybrid_stress_is_bit_identical_across_threads() {
    let fx = fixture(G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| stress(&fx))
    };
    let (a, b) = (run(1), run(4));
    assert_eq!(a.de_deps, b.de_deps);
    assert_eq!(a.e_xc, b.e_xc);
}

/// Typed refusals and the Fock guard from every side.
#[test]
fn refusals_and_the_hybrid_fock_guard() {
    let fx = fixture(G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let run = |f: &str, e: Option<ExxDiv>, scf: &KScfResult| {
        stress_with(&fx, f, scf, &scfg(e, None, None))
    };
    for (name, key) in [
        ("HSE06", "range-separated"),
        ("SCAN", "meta-GGA"),
        ("wB97X-V", "VV10"),
        ("B2PLYP", "double hybrid"),
    ] {
        let e = run(name, Some(ExxDiv::Ewald), &fx.res.scf)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        eprintln!("{name}: {e}");
        assert!(e.contains(key), "{name}: {e}");
    }
    // A hybrid without the exxdiv.
    let e = run("PBE0", None, &fx.res.scf)
        .expect_err("needs exxdiv")
        .to_string();
    eprintln!("no exxdiv: {e}");
    assert!(e.contains("KRksStressConfig::exxdiv"), "{e}");
    // Matching: the exact guard passes.
    let ok = run("PBE0", Some(ExxDiv::Ewald), &fx.res.scf).unwrap();
    assert!(ok.hf.fock_mismatch < 1e-12, "{:e}", ok.hf.fock_mismatch);
    // Wrong exxdiv: the SCF ran Ewald, the config says None.
    let e = run("PBE0", Some(ExxDiv::None), &fx.res.scf)
        .expect_err("wrong exxdiv")
        .to_string();
    eprintln!("wrong exxdiv: {e}");
    assert!(e.contains("differs"), "{e}");
    // Wrong a: B3LYP stress of a PBE0 SCF; and the pure PBE (a = 0).
    for f in ["B3LYP", "PBE"] {
        let e = run(f, Some(ExxDiv::Ewald), &fx.res.scf)
            .expect_err("wrong functional")
            .to_string();
        assert!(e.contains("differs"), "{f}: {e}");
    }
    // Tamper.
    let mut t = fx.res.scf.clone();
    t.fock[1][(0, 1)] += Complex64::new(1e-6, 0.0);
    assert!(run("PBE0", Some(ExxDiv::Ewald), &t).is_err());
    let mut t = fx.res.scf.clone();
    t.fock[1][(0, 1)] += Complex64::new(1e-10, 0.0);
    assert!(run("PBE0", Some(ExxDiv::Ewald), &t).is_ok());
    // The exchange fraction of the diagnostic entry is validated.
    let z = zero_xc(G113, Jk::Dense, ExxDiv::None, 0.0);
    for bad in [-0.1, 1.5, f64::NAN] {
        let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
        let e = with_jk(&z, |j| {
            kpoint_null_xc_stress(
                &z.cell,
                &z.prep,
                &z.mesh,
                &hcfg,
                &z.hk,
                j,
                &z.scf,
                bad,
                &scfg(None, None, None),
            )
        })
        .err()
        .unwrap_or_else(|| panic!("a = {bad} accepted"))
        .to_string();
        assert!(e.contains("outside [0, 1]"), "{e}");
    }
    // An unconverged SCF.
    let mut t = fx.res.scf.clone();
    t.converged = false;
    assert!(run("PBE0", Some(ExxDiv::Ewald), &t).is_err());
}

// ------------------------------------------------------------------ memory

/// Budget gates of the hybrid stress (null XC, a = 0.5 density: no grid pass).
/// (i) A 4 MB budget fails typed on the strain ERI K list and a 16 MB budget
/// reproduces the AMPLE-budget stress BITWISE. (ii) The Fock-guard reservation
/// is read from the typed error the PRODUCTION path raises when the budget
/// holds the per-k arrays but not the guard matrices: bisect the budget until
/// the error names "Fock-guard matrices" and parse its byte count, which must be
/// five N_k n^2 complex arrays (D, J, K, F, V_xc) = 5 * 3 * 4 * 16 = 960 B. A
/// guard that reverted to the four-array charge (768 B) fails the equality.
#[test]
fn budgets_either_reproduce_the_hybrid_stress_or_fail_typed() {
    let z = zero_xc(G113, Jk::Dense, ExxDiv::Ewald, 0.5);
    let (j, k) = jk_mats(&z);
    let mut scf = z.scf.clone();
    for q in 0..scf.fock.len() {
        let mut f = &z.hk.h[q] + &j[q];
        f.scaled_add(Complex64::new(-0.25, 0.0), &k[q]);
        scf.fock[q] = f;
    }
    with_jk(&z, |jk| {
        let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
        let at = |b: usize| {
            let mut cfg = scfg(Some(ExxDiv::Ewald), None, None);
            cfg.stress.budget_bytes = Some(b);
            kpoint_null_xc_stress(&z.cell, &z.prep, &z.mesh, &hcfg, &z.hk, jk, &scf, 0.5, &cfg)
        };
        let full = at(AMPLE).unwrap();
        let e = at(4 << 20).expect_err("4 MB must not hold the strain K list");
        let m = e.to_string();
        eprintln!("4 MB: {m}");
        assert!(m.contains("k stress ERI K list"), "{m}");
        assert_eq!(at(16 << 20).unwrap().de_deps, full.de_deps);

        // (ii) Bisect to the guard gate. Below it the open-ledger arrays fail;
        // in the 960 B band above it the guard fails; above that the run goes on.
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
        assert!(m.contains("D, J, K, F, V_xc"), "{m}");
        let head = &m[..m.find(" bytes;").expect("byte count in the error")];
        let charged: usize = head[head.rfind('(').unwrap() + 1..].parse().unwrap();
        let n = z.prep.nbasis();
        assert_eq!(charged, 5 * z.mesh.nk() * n * n * 16, "{m}");
    });
}

/// The fine walk under the smallest passing budget (bisection, then a 16-byte
/// walk over the 1 KiB below it): every budget fails typed, none passes.
#[test]
#[ignore = "about 90 null-XC stresses at ~12 s each; measured 1167 s"]
fn budget_walk_below_the_smallest_passing_budget_is_typed_errors_only() {
    let z = zero_xc(G113, Jk::Dense, ExxDiv::Ewald, 0.5);
    let (j, k) = jk_mats(&z);
    let mut scf = z.scf.clone();
    for q in 0..scf.fock.len() {
        let mut f = &z.hk.h[q] + &j[q];
        f.scaled_add(Complex64::new(-0.25, 0.0), &k[q]);
        scf.fock[q] = f;
    }
    with_jk(&z, |jk| {
        let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
        let at = |b: usize| {
            let mut cfg = scfg(Some(ExxDiv::Ewald), None, None);
            cfg.stress.budget_bytes = Some(b);
            kpoint_null_xc_stress(&z.cell, &z.prep, &z.mesh, &hcfg, &z.hk, jk, &scf, 0.5, &cfg)
        };
        let full = at(AMPLE).unwrap();
        let (mut lo, mut hi) = (0usize, 1usize << 28);
        while hi - lo > 16 {
            let mid = lo + (hi - lo) / 2;
            if at(mid).is_ok() {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        assert_eq!(at(hi).unwrap().de_deps, full.de_deps);
        let mut b = hi.saturating_sub(1024);
        while b < hi {
            assert!(at(b).is_err(), "budget {b} < {hi} passes");
            b += 16;
        }
        eprintln!("smallest passing budget {hi} B");
    });
}

// ------------------------------------------------------------------ mutants

fn miss(
    fx: &Fx,
    xm: Option<KXcStressMutation>,
    sm: Option<KStressMutation>,
    base: &KRksStress,
) -> f64 {
    mdiff(&stress_mut(fx, xm, sm).de_deps, &base.de_deps)
}

/// Production-wired defects of the exchange scale, each against the FD-,
/// Gamma- and supercell-anchored stress (dense AFT, Ewald, PBE0, 1x1x3). The
/// largest correct residual of those anchors is 1.0e-9 (all-nine FD at the
/// chosen step, over every route; the Gamma anchors are <= 1.9e-12) and the
/// smallest mutant miss below is the wrong-density K at 1.1e-2, so the bar 1e-5
/// sits between with margins of 1e4 (above the correct) and 1e3 (below the
/// smallest mutant).
#[test]
fn smoke_mutants_are_caught() {
    use KStressMutation as S;
    use KXcStressMutation as X;
    let fx = fixture(G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let base = stress(&fx);
    for (xm, sm) in [
        (Some(X::ExchUnscaled), None),
        (Some(X::ExchScaledTwice), None),
        (Some(X::KWrongDensity), None),
        (None, Some(S::MadelungUnscaled)),
    ] {
        let d = miss(&fx, xm, sm, &base);
        eprintln!("dense Ewald {xm:?}/{sm:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{xm:?}/{sm:?} survives: {d:e}");
    }
}

/// The remaining mutants: the pre-existing Madelung seams under a hybrid, and
/// the RS-GDF fit pieces (a missing exchange piece of the fit, the scaling
/// mutants on the fit route). Identities must show EXACTLY 0.
#[test]
#[ignore = "9 PBE0 stresses on dense and RS-GDF; measured ~120 s"]
fn remaining_mutants_are_caught() {
    use KStressMutation as S;
    use KXcStressMutation as X;
    let fx = fixture(G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let base = stress(&fx);
    for sm in [S::NoMadelung, S::GammaMadelung] {
        let d = miss(&fx, None, Some(sm), &base);
        eprintln!("dense Ewald {sm:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{sm:?} survives: {d:e}");
    }
    // Identity: no Madelung without Ewald.
    let fn0 = fixture(G113, "PBE0", Jk::Dense, ExxDiv::None);
    let b0 = stress(&fn0);
    assert_eq!(miss(&fn0, None, Some(S::MadelungUnscaled), &b0), 0.0);
    // RS-GDF: the fit pieces.
    let fg = fixture(G113, "PBE0", Jk::Gdf, ExxDiv::Ewald);
    let bg = stress(&fg);
    for (xm, sm) in [
        (None, Some(S::FitExchangeDropped)),
        (Some(X::ExchUnscaled), None),
        (Some(X::ExchScaledTwice), None),
        (None, Some(S::MadelungUnscaled)),
    ] {
        let d = miss(&fg, xm, sm, &bg);
        eprintln!("RS-GDF Ewald {xm:?}/{sm:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{xm:?}/{sm:?} survives: {d:e}");
    }
}

/// The scaling mutants are identities where they should be: a pure functional
/// (a = 0) is untouched by ExchUnscaled / ExchScaledTwice / MadelungUnscaled /
/// FitExchangeDropped, bitwise; and ExchUnscaled / ExchScaledTwice are
/// identities at a = 1 (null-XC).
#[test]
#[ignore = "PBE RS-GDF SCF + 5 stresses + a = 1 null-XC; measured 59 s"]
fn scaling_mutants_are_identities_at_a_zero_and_a_one() {
    use KStressMutation as S;
    use KXcStressMutation as X;
    let fx = fixture(G113, "PBE", Jk::Gdf, ExxDiv::None);
    let base = stress(&fx);
    for (xm, sm) in [
        (Some(X::ExchUnscaled), None),
        (Some(X::ExchScaledTwice), None),
        (None, Some(S::MadelungUnscaled)),
        (None, Some(S::FitExchangeDropped)),
    ] {
        assert_eq!(miss(&fx, xm, sm, &base), 0.0, "{xm:?}/{sm:?} at a = 0");
    }
    let z = zero_xc(G113, Jk::Dense, ExxDiv::Ewald, 1.0);
    let b = null_stress(&z, 1.0, &z.scf, &scfg(Some(ExxDiv::Ewald), None, None));
    for xm in [X::ExchUnscaled, X::ExchScaledTwice] {
        let m = null_stress(&z, 1.0, &z.scf, &scfg(Some(ExxDiv::Ewald), Some(xm), None));
        assert_eq!(mdiff(&m.de_deps, &b.de_deps), 0.0, "{xm:?} at a = 1");
    }
}
