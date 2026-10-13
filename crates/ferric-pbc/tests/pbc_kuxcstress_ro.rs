//! Analytic stress of the k-point ROHF / ROKS energy
//! (`ferric_pbc::kuxcstress::{kpoint_rohf_stress, kpoint_roks_stress}`, module
//! doc "ROHF / ROKS"): the open-shell UKS stress assembly on the ROHF spin
//! densities with the orthonormality Lagrangian `W_RO` as the overlap weight
//! and a per-k ROHF orbital-gradient gate.
//!
//! WHICH ANCHOR COVERS WHICH DEFECT (measured, not assumed; the numbers are in
//! the doc comment of each test):
//!
//! * (1) closed-shell limit (no open orbital): ROHF == the RHF stress and ROKS
//!   == the RKS stress, per part. Sees the W_RO factor (`RoWDropCross` halves
//!   `W`), a wrong overall spin factor; BLIND to `RoWUhfForm` (identity),
//!   to every alpha/beta asymmetry.
//! * (2) 1x1x1 vs the Gamma ROHF / ROKS stress per part (H3 doublet, H4
//!   triplet): `RoWDropCross` (W_RO cross term), `XcSpinSwap`, exchange scaled
//!   twice / unscaled, wrong factor, every spin defect at nk = 1. BLIND to
//!   `RoWUhfForm` (identity at convergence).
//! * (3) 1x1x3 vs the 3-copy Gamma supercell: the Bloch phases / mesh Madelung.
//! * (4) FD of the energy under symmetric strain: the stress of the energy as
//!   computed (all nine components; Monkhorst-Pack mesh; the trace).
//! * (5) a = 1 null-XC through the ROKS SCF driver == the ROHF-driver stress.
//! * the orbital-gradient gate: a loose state is refused (exact message) and
//!   is wrong by far more than the FD bar with the gate OFF; with the gate off
//!   the unrestricted-form `W` is NO better than `W_RO` (so the gate, not the
//!   form of `W`, is what protects off stationarity).
//!
//! NOT covered: RS-GDF (refused), p/d shells and heavy atoms / ECP (every
//! fixture is H/STO-3G), range-separated hybrids / meta-GGA / VV10 (refused),
//! the production (75, 302) grid, no external (PySCF) anchor.

#![allow(unused_imports, dead_code)]
mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::periodic_hcore;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kdft::{solve_krks, KRksConfig};
use ferric_pbc::kgrad::KRsGdfGradSource;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::{
    solve_krohf, solve_kroks, solve_kroks_injected, KRoScfResult, KRohfConfig, KRoksConfig,
    KRoksResult,
};
use ferric_pbc::kscf::{
    solve_krhf, KPointInjection, KPointJk, KRhfConfig,
};
use ferric_pbc::kstress::{kpoint_rhf_stress, KStressConfig, KStressJk, KStressMutation};
use ferric_pbc::kuxcstress::{
    kpoint_null_xc_roks_stress, kpoint_rohf_stress, kpoint_roks_stress, KUStressMutation,
    KUksStress, KUksStressConfig,
};
use ferric_pbc::kxcgrad::KXcGrid;
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStressConfig, KXcStressMutation};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rohf::{gamma_rohf, gamma_roks, GammaRohfConfig, GammaRoksConfig};
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::stress::{
    gamma_rohf_stress, gamma_roks_stress, GammaStress, GammaStressConfig, Mat3,
};
use ferric_pbc::uhf::GammaUhfIntegrals;
use num_complex::Complex64;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
/// H3 doublet (na = 2, nb = 1): closed 1, open 1, virtual 1 (n = 3).
const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];
/// H4 triplet (na = 3, nb = 1): closed 1, open 2, virtual 1 (n = 4).
const H4: [[f64; 3]; 4] = [
    [0.3, 0.2, 0.1],
    [0.35, 0.12, 1.5],
    [0.3, 1.9, 0.4],
    [1.8, 0.5, 2.2],
];
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
/// Shared explicit neighbour cutoff (the default depends on the cell and puts a
/// one-step discontinuity in the energy; see `pbc_kuxcstress.rs`).
const CUTOFF: f64 = 14.0;
const ANCHOR_PRECISION: f64 = 1e-12;
/// FD step and bar: those of the UKS stress (`pbc_kuxcstress.rs`,
/// `fd_window_probe`); the RO measurements are recorded in `ro_fd_window_probe`.
const FD_H: f64 = 2.5e-5;
const FD_BAR: f64 = 3e-8;
const MUTANT_BAR: f64 = 1e-5;

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

thread_local! {
    /// (energy_conv, grad_conv) of the k-ROHF/ROKS SCF on THIS test thread.
    static SETTINGS: std::cell::Cell<(f64, f64)> = const { std::cell::Cell::new((1e-13, 1e-10)) };
}

fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

fn cell_of(pos: &[[f64; 3]]) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = match pos.len() {
        3 => 2,
        4 => 3,
        _ => 1,
    };
    Cell::new(mol, cubic(4.0)).expect("cell")
}

fn rohf_cfg(cell: &Cell, exxdiv: ExxDiv) -> KRohfConfig {
    let mut c = KRohfConfig::for_cell(cell, exxdiv);
    c.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    let (e, g) = SETTINGS.with(|s| s.get());
    c.scf.energy_conv = e;
    c.scf.grad_conv = g;
    c.scf.max_iter = 600;
    c
}

/// A converged (unless the thread settings are loosened) ROHF/ROKS SCF.
fn scf_on(cell: &Cell, m: Mesh, functional: Option<&str>, exxdiv: ExxDiv) -> KRoksResult {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = mesh_of(cell, m);
    let rc = rohf_cfg(cell, exxdiv);
    let r = match functional {
        None => solve_krohf(cell, &prep, None, &mesh, &rc),
        Some(f) => {
            let mut c = KRoksConfig::new(cell, exxdiv, f);
            c.rohf = rc;
            if f.contains("PBE0") {
                c.rohf.level_shift = 0.05;
            }
            c.grid = grid_cfg();
            solve_kroks(cell, &prep, None, &mesh, &c)
        }
    }
    .unwrap_or_else(|e| panic!("k-ROHF/ROKS {functional:?}: {e}"));
    assert!(r.scf.converged, "{functional:?} not converged");
    r
}

fn energy_on(cell: &Cell, m: Mesh, f: Option<&str>, exxdiv: ExxDiv) -> f64 {
    scf_on(cell, m, f, exxdiv).scf.energy
}

struct Fx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    aux: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    res: KRoksResult,
    exxdiv: ExxDiv,
    functional: Option<String>,
}

fn fixture_cell(cell: Cell, m: Mesh, functional: Option<&str>, exxdiv: ExxDiv) -> Fx {
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = mesh_of(&cell, m);
    let res = scf_on(&cell, m, functional, exxdiv);
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
        exxdiv,
        functional: functional.map(|s| s.to_string()),
    }
}

fn fixture(pos: &[[f64; 3]], m: Mesh, functional: Option<&str>, exxdiv: ExxDiv) -> Fx {
    fixture_cell(cell_of(pos), m, functional, exxdiv)
}

fn scfg(
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
            exxdiv: Some(ExxDiv::Ewald),
            mutation: xm,
            ..Default::default()
        },
        spin_mutation: sm,
    }
}

fn dense_eri(fx: &Fx) -> KDenseAftEri {
    KDenseAftEri::build(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hk.s,
        fx.exxdiv,
        &Default::default(),
    )
    .unwrap()
}

/// The RO stress entry matching the fixture (ROHF if `functional` is `None`).
fn stress_with(
    fx: &Fx,
    scf: &KRoScfResult,
    cfg: &KUksStressConfig,
) -> Result<KUksStress, ferric_core::FerricError> {
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let eri = dense_eri(fx);
    let mut c = cfg.clone();
    c.base.exxdiv = Some(fx.exxdiv);
    match &fx.functional {
        None => kpoint_rohf_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &hcfg,
            &fx.hk,
            KStressJk::Dense(&eri),
            scf,
            fx.exxdiv,
            &c,
        ),
        Some(f) => kpoint_roks_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &hcfg,
            &fx.hk,
            KStressJk::Dense(&eri),
            scf,
            f,
            &c,
        ),
    }
}

fn stress(fx: &Fx) -> KUksStress {
    stress_with(fx, &fx.res.scf, &scfg(None, None, None)).expect("k RO stress")
}

fn stress_mut(
    fx: &Fx,
    xm: Option<KXcStressMutation>,
    sm: Option<KUStressMutation>,
    hm: Option<KStressMutation>,
) -> KUksStress {
    stress_with(fx, &fx.res.scf, &scfg(xm, sm, hm)).expect("k RO stress (mutant)")
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

/// Central FD of the ROHF/ROKS SCF energy for the listed components.
fn fd_components(
    cell: &Cell,
    m: Mesh,
    f: Option<&str>,
    exxdiv: ExxDiv,
    h: f64,
    comps: &[(usize, usize)],
) -> Vec<f64> {
    comps
        .iter()
        .map(|&(i, j)| {
            let ep = energy_on(&strained(cell, i, j, h), m, f, exxdiv);
            let em = energy_on(&strained(cell, i, j, -h), m, f, exxdiv);
            (ep - em) / (2.0 * h)
        })
        .collect()
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

// ------------------------------------------------------------------ (2)

fn gamma_side(cell: &Cell, f: Option<&str>) -> GammaStress {
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
    let gc = GammaStressConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    match f {
        None => {
            let mut c = GammaRohfConfig::default();
            c.exxdiv = ExxDiv::Ewald;
            c.scf.energy_conv = 1e-13;
            c.scf.density_conv = 1e-10;
            c.scf.max_iter = 600;
            let r = gamma_rohf(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &c).unwrap();
            gamma_rohf_stress(cell, &prep, &hcfg, &hc, &eri, &r.scf, ExxDiv::Ewald, &gc).unwrap()
        }
        Some(f) => {
            let mut c = GammaRoksConfig::new(f);
            c.grid = grid_cfg();
            c.exxdiv = ExxDiv::Ewald;
            c.scf.energy_conv = 1e-13;
            c.scf.density_conv = 1e-10;
            c.scf.max_iter = 600;
            let r = gamma_roks(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &c).unwrap();
            gamma_roks_stress(cell, &prep, &hcfg, &hc, &eri, &r.scf, &c, &gc).unwrap()
        }
    }
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

/// Per-part bar against the Gamma anchors (derived in the test docs).
const BAR_GAMMA: f64 = 1e-9;

/// Per-part comparison with the Gamma stress; with `w_mutant`, the same
/// reference must reject the `W_RO` cross term dropped and accept the
/// unrestricted-form `W` (an identity at convergence).
fn one_point_vs_gamma(pos: &[[f64; 3]], f: Option<&str>, w_mutant: bool) {
    let gam = gamma_side(&cell_of(pos), f);
    let fx = fixture(pos, G111, f, ExxDiv::Ewald);
    let k = stress(&fx);
    for (n, d) in gamma_rows(&k, &gam, 1.0) {
        eprintln!("{f:?} H{} 1x1x1 vs Gamma RO: {n:<10} {d:.2e}", pos.len());
        assert!(d < BAR_GAMMA, "{f:?} {n}: {d:e}");
    }
    assert!(mmax(&k.hf.parts.exchange) > 1e-4);
    // The reported gate value is the orbital gradient, tiny at convergence.
    assert!(k.hf.commutator < 1e-8, "{:e}", k.hf.commutator);
    if w_mutant {
        let drop = stress_mut(&fx, None, Some(KUStressMutation::RoWDropCross), None);
        let d = mdiff(&drop.de_deps, &gam.de_deps);
        eprintln!("{f:?} vs Gamma RO under RoWDropCross: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "RoWDropCross survives the Gamma anchor: {d:e}");
        let uhf = stress_mut(&fx, None, Some(KUStressMutation::RoWUhfForm), None);
        let d = mdiff(&uhf.de_deps, &k.de_deps);
        eprintln!("{f:?} RoWUhfForm at convergence: {d:.2e}");
        assert!(d < 1e-8, "identity at convergence: {d:e}");
    }
}

/// (2a) 1x1x1 == the Gamma ROHF stress, per part, on the H3 doublet.
#[test]
fn one_point_mesh_matches_the_gamma_rohf_stress_per_part() {
    one_point_vs_gamma(&H3, None, true);
}

#[test]
#[ignore = "H4 triplet ROHF vs Gamma ROHF; measured below"]
fn one_point_mesh_matches_the_gamma_rohf_stress_per_part_triplet() {
    one_point_vs_gamma(&H4, None, true);
}

#[test]
#[ignore = "Gamma ROKS 1x1x1 anchors (PBE, PBE0); measured below"]
fn one_point_mesh_matches_the_gamma_roks_stress_per_part() {
    one_point_vs_gamma(&H3, Some("PBE"), false);
    one_point_vs_gamma(&H3, Some("PBE0"), false);
}
