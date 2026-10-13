//! Analytic forces of the k-point ROHF / ROKS energy
//! (`ferric_pbc::kuxcgrad::{kpoint_rohf_gradient, kpoint_roks_gradient}`,
//! module doc "ROHF / ROKS"): the open-shell assembly on the ROHF spin
//! densities with the orthonormality Lagrangian `W_RO` and a per-k ROHF
//! orbital-gradient gate.
//!
//! Anchors: 1x1x1 against the Gamma ROHF / ROKS gradients per part on the H3
//! doublet (ROHF in the default set, with the `W_RO` mutant rejected by the
//! same reference); central FD of the ROHF and ROKS energies (fresh SCF per
//! displacement, explicit shared neighbour cutoff); the orbital-gradient gate
//! from both sides (a loose state is refused with the gate message and is
//! wrong by far more than the FD bar with the gate off); a short MO set and an
//! RS-GDF source are typed refusals; the unrestricted-form `W` is an IDENTITY
//! at convergence and asserted as such.
//!
//! NOT covered: the supercell identity (the Gamma ROHF high-spin supercell is
//! not converged here), RS-GDF (refused), shifted meshes, p/d shells, stress.

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::grad::{
    gamma_rohf_gradient_with, gamma_roks_gradient_with, GammaGradConfig, GammaGradient,
};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::periodic_hcore;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kgrad::{KGradConfig, KGradJk, KRsGdfGradSource};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::{solve_krohf, solve_kroks, KRohfConfig, KRoksConfig, KRoksResult};
use ferric_pbc::kuxcgrad::{
    kpoint_rohf_gradient, kpoint_roks_gradient, KUSpinMutation, KUksGradConfig, KUksGradient,
};
use ferric_pbc::kxcgrad::{KRksGradConfig, KXcGrid, KXcMutation};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rohf::{gamma_rohf, gamma_roks, GammaRohfConfig, GammaRoksConfig};
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::uhf::GammaUhfIntegrals;
use ndarray::Array2;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
// Shared neighbour cutoff, FD step and bar: see `pbc_kuxcgrad.rs`
// (`fd_step_probe`). FD bar 1e-8: the measured ROHF/ROKS residuals at h = 5e-5
// are 2.4e-10 (ROHF), 3.2e-10 (ROKS PBE) and 3.1e-10 (ROKS PBE0), so the bar
// is ~30x the worst correct residual and ~1e6 below the smallest mutant
// (`RoWDropCross`, 1.8e-2).
//
// `RO_ORBITAL_GRADIENT_TOL` = 1e-5 (grad.rs) against the k-point entry points,
// from `ro_gate_probe` (H3 doublet ROKS PBE 1x1x1; orbital gradient g of an SCF
// stopped early -> force error with the gate off, against the tightly
// converged force):
//   g 3.5e-5 -> 3.2e-5     g 2.5e-7 -> 6.9e-7     g 2.0e-9 -> 4.9e-9
// i.e. the error is of the order of g (0.9 g to 3 g), so 1e-5 caps it near
// 1e-5 Ha/Bohr. Looser states (g >~ 1e-4) fail the MO occupation check
// (the density is no longer a projector) before the gate.
const CUTOFF: f64 = 14.0;
const FD_BAR: f64 = 1e-8;
const FD_H: f64 = 5e-5;
const ANCHOR_PRECISION: f64 = 1e-12;
const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];

fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

fn cell_of(pos: &[[f64; 3]]) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = 2;
    Cell::new(mol, cubic(4.0)).expect("cell")
}

fn rohf_cfg(cell: &Cell, exxdiv: ExxDiv) -> KRohfConfig {
    let mut c = KRohfConfig::for_cell(cell, exxdiv);
    c.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    c.scf.energy_conv = 1e-13;
    c.scf.grad_conv = 1e-10;
    c.scf.max_iter = 600;
    c
}

struct Fx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    res: KRoksResult,
    exxdiv: ExxDiv,
    functional: Option<String>,
}

fn run_scf(
    pos: &[[f64; 3]],
    mesh_n: [usize; 3],
    functional: Option<&str>,
    exxdiv: ExxDiv,
) -> KRoksResult {
    run_scf_conv(pos, mesh_n, functional, exxdiv, 1e-13, 1e-10)
}

fn run_scf_conv(
    pos: &[[f64; 3]],
    mesh_n: [usize; 3],
    functional: Option<&str>,
    exxdiv: ExxDiv,
    energy_conv: f64,
    grad_conv: f64,
) -> KRoksResult {
    let cell = cell_of(pos);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let mut rc = rohf_cfg(&cell, exxdiv);
    rc.scf.energy_conv = energy_conv;
    rc.scf.grad_conv = grad_conv;
    let r = match functional {
        None => solve_krohf(&cell, &prep, None, &mesh, &rc),
        Some(f) => {
            let mut c = KRoksConfig::new(&cell, exxdiv, f);
            c.rohf = rc;
            if f.contains("PBE0") {
                c.rohf.level_shift = 0.05;
            }
            c.grid = grid_cfg();
            solve_kroks(&cell, &prep, None, &mesh, &c)
        }
    }
    .unwrap_or_else(|e| panic!("k-ROHF/ROKS {functional:?}: {e}"));
    assert!(r.scf.converged);
    r
}

fn fixture(pos: &[[f64; 3]], mesh_n: [usize; 3], functional: Option<&str>, exxdiv: ExxDiv) -> Fx {
    let cell = cell_of(pos);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let res = run_scf(pos, mesh_n, functional, exxdiv);
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    Fx {
        cell,
        prep,
        mesh,
        hk,
        res,
        exxdiv,
        functional: functional.map(|s| s.to_string()),
    }
}

fn gcfg(xm: Option<KXcMutation>, sm: Option<KUSpinMutation>) -> KUksGradConfig {
    KUksGradConfig {
        base: KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
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

fn force_scf(
    fx: &Fx,
    scf: &ferric_pbc::kroks::KRoScfResult,
    cfg: &KUksGradConfig,
) -> Result<KUksGradient, ferric_core::FerricError> {
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let eri = KDenseAftEri::build(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hk.s,
        fx.exxdiv,
        &Default::default(),
    )
    .unwrap();
    let mut c = cfg.clone();
    c.base.exxdiv = Some(fx.exxdiv);
    match &fx.functional {
        None => kpoint_rohf_gradient(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &hcfg,
            &fx.hk,
            KGradJk::Dense(&eri),
            scf,
            fx.exxdiv,
            &c,
        ),
        Some(f) => kpoint_roks_gradient(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &hcfg,
            &fx.hk,
            KGradJk::Dense(&eri),
            scf,
            f,
            &c,
        ),
    }
}

fn force(fx: &Fx) -> KUksGradient {
    force_scf(fx, &fx.res.scf, &gcfg(None, None)).unwrap()
}

fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

fn parts_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn gamma_side(cell: &Cell, f: Option<&str>) -> GammaGradient {
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
    let gc = GammaGradConfig {
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
            gamma_rohf_gradient_with(cell, &prep, &hcfg, &hc, &eri, &r.scf, ExxDiv::Ewald, &gc)
                .unwrap()
        }
        Some(f) => {
            let mut c = GammaRoksConfig::new(f);
            c.grid = grid_cfg();
            c.exxdiv = ExxDiv::Ewald;
            c.scf.energy_conv = 1e-13;
            c.scf.density_conv = 1e-10;
            c.scf.max_iter = 600;
            let r = gamma_roks(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &c).unwrap();
            gamma_roks_gradient_with(cell, &prep, &hcfg, &hc, &eri, &r.scf, &c, &gc).unwrap()
        }
    }
}

/// 1x1x1 == the Gamma ROHF / ROKS gradient per part (H3 doublet; independent
/// real-AO construction and SCF). With `w_mutant`, the same reference must
/// reject the `W_RO` cross term dropped (`RoWDropCross`).
fn one_point_vs_gamma_ro(f: Option<&str>, w_mutant: bool) {
    let gam = gamma_side(&cell_of(&H3), f);
    let fx = fixture(&H3, [1, 1, 1], f, ExxDiv::Ewald);
    let k = force(&fx);
    let (p, kp) = (&gam.parts, &k.hf.parts);
    let mut rows = vec![
        ("overlap", parts_diff(&kp.overlap, &p.overlap)),
        ("kinetic", parts_diff(&kp.kinetic, &p.kinetic)),
        ("eri=J+K", parts_diff(&(&kp.coulomb + &kp.exchange), &p.eri)),
        ("total", parts_diff(&k.grad, &gam.grad)),
    ];
    if f.is_some() {
        rows.push(("xc_ao", parts_diff(&k.xc_ao, &p.xc_ao)));
        rows.push(("xc_weight", parts_diff(&k.xc_weight, &p.xc_weight)));
    }
    for (n, d) in rows {
        eprintln!("{f:?} H3 1x1x1 vs Gamma RO: {n:<10} {d:.2e}");
        // Worst measured 1.6e-12.
        assert!(d < 1e-9, "{f:?} {n}: {d:e}");
    }
    // The gate value reported is the orbital gradient, tiny at convergence.
    assert!(k.hf.commutator < 1e-8, "{:e}", k.hf.commutator);
    if w_mutant {
        let m = force_scf(
            &fx,
            &fx.res.scf,
            &gcfg(None, Some(KUSpinMutation::RoWDropCross)),
        )
        .unwrap();
        let d = parts_diff(&m.grad, &gam.grad);
        eprintln!("{f:?} H3 1x1x1 vs Gamma RO under RoWDropCross: miss {d:.2e}");
        assert!(d > 1e-6, "RoWDropCross survives the Gamma anchor: {d:e}");
    }
}

#[test]
fn one_point_mesh_matches_the_gamma_rohf_gradient_per_part() {
    one_point_vs_gamma_ro(None, true);
}

#[test]
#[ignore = "Gamma ROKS 1x1x1 anchors (PBE, PBE0); measured ~40 s (box load 15-60: inflated)"]
fn one_point_mesh_matches_the_gamma_roks_gradient_per_part() {
    one_point_vs_gamma_ro(Some("PBE"), false);
    one_point_vs_gamma_ro(Some("PBE0"), false);
}

fn fd(pos: &[[f64; 3]], a: usize, x: usize, h: f64, f: Option<&str>, exxdiv: ExxDiv) -> f64 {
    let e = |p: &[[f64; 3]]| run_scf(p, [1, 1, 3], f, exxdiv).scf.energy;
    (e(&moved(pos, a, x, h)) - e(&moved(pos, a, x, -h))) / (2.0 * h)
}

#[test]
#[ignore = "FD of ROHF/ROKS energies; measured ~45 s (box load 15-60: inflated)"]
fn ro_force_matches_fd() {
    for (f, exxdiv) in [
        (None, ExxDiv::Ewald),
        (Some("PBE"), ExxDiv::None),
        (Some("PBE0"), ExxDiv::Ewald),
    ] {
        let fx = fixture(&H3, [1, 1, 3], f, exxdiv);
        let g = force(&fx);
        let mut worst = 0.0_f64;
        for (a, x) in [(0, 2), (2, 1)] {
            let d = fd(&H3, a, x, FD_H, f, exxdiv);
            worst = worst.max((g.grad[(a, x)] - d).abs());
        }
        eprintln!("{f:?} {exxdiv:?}: RO analytic vs FD worst {worst:.2e}");
        assert!(worst < FD_BAR, "{f:?} {exxdiv:?}: {worst:e}");
    }
}

/// (energy_conv, grad_conv) of the deliberately loose ROKS SCF of the gate
/// tests, and the orbital gradient `g` / gate-off force error they give (see
/// `ro_gate_probe`).
const LOOSE_RO: (f64, f64) = (1e-4, 1e-2);

/// The unrestricted-form `W` is an identity at convergence; dropping the
/// cross term misses; a tampered Fock is refused naming the spin.
#[test]
#[ignore = "RoWUhfForm identity, RoWDropCross and the beta Fock tamper; measured 40 s (box load 5-8)"]
fn w_ro_mutants_and_the_fock_guard() {
    let fx = fixture(&H3, [1, 1, 1], Some("PBE"), ExxDiv::None);
    let base = force(&fx);
    let uhf_w = force_scf(
        &fx,
        &fx.res.scf,
        &gcfg(None, Some(KUSpinMutation::RoWUhfForm)),
    )
    .unwrap();
    let d_uhf = parts_diff(&uhf_w.grad, &base.grad);
    eprintln!("RoWUhfForm at convergence: miss {d_uhf:.2e}");
    assert!(d_uhf < 1e-8, "identity at convergence: {d_uhf:e}");
    let drop = force_scf(
        &fx,
        &fx.res.scf,
        &gcfg(None, Some(KUSpinMutation::RoWDropCross)),
    )
    .unwrap();
    let d_drop = parts_diff(&drop.grad, &base.grad);
    eprintln!("RoWDropCross: miss {d_drop:.2e}");
    assert!(d_drop > 1e-5, "{d_drop:e}");
    let mut t = fx.res.scf.clone();
    t.fock_beta[0][(0, 1)] += num_complex::Complex64::new(1e-6, 0.0);
    let e = force_scf(&fx, &t, &gcfg(None, None))
        .expect_err("tamper")
        .to_string();
    assert!(
        e.contains("kpoint_roks_gradient: the Fock matrix rebuilt") && e.contains("fock_beta"),
        "{e}"
    );
}

/// The ROHF orbital-gradient gate fires. A loosely converged ROKS state with
/// `g > RO_ORBITAL_GRADIENT_TOL` (g recorded below) is refused with the gate
/// message, NOT the Fock-guard message (the stored Fock is the density's own);
/// the same state with the gate off returns a force wrong by far more than
/// the FD bar, so the gate matters.
#[test]
fn the_orbital_gradient_gate_fires_and_matters() {
    let tight = fixture(&H3, [1, 1, 1], Some("PBE"), ExxDiv::None);
    let gt = force(&tight);
    let loose = Fx {
        res: run_scf_conv(
            &H3,
            [1, 1, 1],
            Some("PBE"),
            ExxDiv::None,
            LOOSE_RO.0,
            LOOSE_RO.1,
        ),
        cell: tight.cell.clone(),
        prep: prep_for(&tight.cell, &pyscf_sto3g_h()),
        mesh: KPointMesh::gamma_centred(&tight.cell, [1, 1, 1]).unwrap(),
        hk: tight.hk.clone(),
        exxdiv: tight.exxdiv,
        functional: tight.functional.clone(),
    };
    let e = force_scf(&loose, &loose.res.scf, &gcfg(None, None))
        .expect_err("loose ROKS accepted")
        .to_string();
    eprintln!("loose ROKS: {e}");
    assert!(
        e.contains("kpoint_roks_gradient: ROHF orbital gradient") && e.contains("exceeds 1e-5"),
        "{e}"
    );
    assert!(!e.contains("Fock matrix rebuilt"), "{e}");
    let off = force_scf(
        &loose,
        &loose.res.scf,
        &gcfg(None, Some(KUSpinMutation::RoGateOff)),
    )
    .unwrap();
    let g = off.hf.commutator;
    let miss = parts_diff(&off.grad, &gt.grad);
    eprintln!("loose ROKS: g = {g:.3e}, force error with the gate off {miss:.3e}");
    assert!(g > 1e-5, "state is not past the gate: g = {g:e}");
    assert!(
        miss > 10.0 * FD_BAR,
        "gate off is not wrong enough: {miss:e}"
    );
}

/// Probe behind `RO_ORBITAL_GRADIENT_TOL` and `LOOSE_RO`: the orbital gradient
/// g of ROKS SCFs stopped early against the force error with the gate off.
#[test]
#[ignore = "probe: 7 open-shell SCFs"]
fn ro_gate_probe() {
    let gt = force(&fixture(&H3, [1, 1, 1], Some("PBE"), ExxDiv::None));
    for (ec, gc) in [
        (1e-2, 1e-1),
        (1e-3, 3e-2),
        (1e-4, 1e-2),
        (1e-5, 3e-3),
        (1e-6, 1e-3),
        (1e-8, 1e-4),
        (1e-10, 1e-6),
    ] {
        let mut fx = fixture(&H3, [1, 1, 1], Some("PBE"), ExxDiv::None);
        fx.res = run_scf_conv(&H3, [1, 1, 1], Some("PBE"), ExxDiv::None, ec, gc);
        match force_scf(
            &fx,
            &fx.res.scf,
            &gcfg(None, Some(KUSpinMutation::RoGateOff)),
        ) {
            Ok(off) => eprintln!(
                "conv ({ec:.0e},{gc:.0e}): g {:.3e}, force error (gate off) {:.3e}",
                off.hf.commutator,
                parts_diff(&off.grad, &gt.grad)
            ),
            Err(e) => eprintln!("conv ({ec:.0e},{gc:.0e}): refused: {e}"),
        }
    }
}

/// A short MO set and an RS-GDF J/K source are typed refusals.
#[test]
fn short_mos_and_rsgdf_are_refused() {
    let fx = fixture(&H3, [1, 1, 1], Some("PBE"), ExxDiv::None);
    let mut t = fx.res.scf.clone();
    t.mos.pop();
    t.mos.pop();
    let e = force_scf(&fx, &t, &gcfg(None, None))
        .expect_err("short mos")
        .to_string();
    eprintln!("short mos: {e}");
    assert!(
        e.contains("kpoint_roks_gradient:") && e.contains("MO sets for 1 k-points"),
        "{e}"
    );
    // RS-GDF source (refused before any use of the SCF result).
    let aux = prep_for(&fx.cell, &et_sp_aux());
    let kc = KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    };
    let gdf = KRsGdf::build(&fx.cell, &fx.prep, &aux, &fx.mesh, &fx.hk.s, &kc).unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let src = || {
        KGradJk::RsGdf(KRsGdfGradSource {
            gdf: &gdf,
            cfg: &kc,
            aux: &aux,
        })
    };
    let cfg = gcfg(None, None);
    let e = kpoint_roks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        src(),
        &fx.res.scf,
        "PBE",
        &cfg,
    )
    .expect_err("roks rsgdf")
    .to_string();
    assert!(
        e.contains("kpoint_roks_gradient: RS-GDF J/K is not supported"),
        "{e}"
    );
    let e = kpoint_rohf_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        src(),
        &fx.res.scf,
        ExxDiv::None,
        &cfg,
    )
    .expect_err("rohf rsgdf")
    .to_string();
    assert!(
        e.contains("kpoint_rohf_gradient: RS-GDF J/K is not supported"),
        "{e}"
    );
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
