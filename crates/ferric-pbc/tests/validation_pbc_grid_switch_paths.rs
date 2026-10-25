//! The smooth neighbour switch (`PeriodicGridConfig::neighbour_switch`,
//! issue #495) on the derivative paths `validation_pbc_grid_switch.rs` (#504)
//! leaves out: open shell (UKS, ROKS), meta-GGA (closed and open shell) and
//! range-separated hybrids (RSH). #504 FD-checked the closed-shell PBE force
//! and stress only.
//!
//! The switch lives in ONE place, the partition kernel inside
//! `dft::generate_points`, which the energy grid, the response (force) grid
//! and the strain (stress) grid all go through. What differs per path is the
//! XC derivative driver that consumes the switched weights and their
//! derivatives:
//!
//! | path | SCF | force / stress driver |
//! |---|---|---|
//! | Gamma UKS PBE | `dft::gamma_uks` | native `grad::uks_gradient`, `stress::ks_stress` (polarized) |
//! | Gamma ROKS PBE | `rohf::gamma_roks` | native `grad::ro_gradient`, `stress::ks_stress` (RO) |
//! | k UKS PBE, 1x1x2 | `kuks::solve_kuks` | `kuxcgrad` / `kuxcstress` (Bloch phases) |
//! | k ROKS PBE, 1x1x2 | `kroks::solve_kroks` | `kuxcgrad` / `kuxcstress` ROKS assembly |
//! | Gamma RKS SCAN | `dft::gamma_rks` | `gamma_mgga` → 1x1x1 `kxcgrad` / `kxcstress` tau terms |
//! | k UKS SCAN, 1x1x1 | `kuks::solve_kuks` | `kuxcgrad` / `kuxcstress` polarized tau terms |
//! | Gamma RKS CAM-B3LYP | `dft::gamma_rks_rsh` | `gamma_rsh` → 1x1x1 `kxcgrad` / `kxcstress` |
//! | Gamma UKS CAM-B3LYP | `dft::gamma_uks_rsh` | `gamma_rsh` → 1x1x1 `kuxcgrad` (force only) |
//! | Gamma ROKS CAM-B3LYP | `rohf::gamma_roks_rsh` | `gamma_rsh` → 1x1x1 `kuxcgrad` ROKS (force only) |
//!
//! Not a path: the open-shell RSH STRESS (UKS and ROKS) does not exist with
//! or without the switch (the UKS stress has no range-separated term); it is
//! refused by name, pinned per commit by `pbc_grid_switch_refusals.rs`. ROKS
//! meta-GGA has no SCF (refused by `kroks` / `rohf`), so it has no derivative
//! to check. Open-shell meta-GGA at Gamma routes through the k code on a
//! 1x1x1 mesh (the k UKS SCAN row).
//!
//! Physics hypothesis: with `W > 0` each path's analytic force and stress
//! match the central FD of ferric's OWN switched SCF energy at `h = 1e-4` to
//! the `h²` truncation (halving `h` cuts the miss ~4x, and the Richardson
//! combination falls to the SCF floor). Artifact hypothesis:
//! a driver that drops or mis-weights the presence derivative of the taper
//! (the part #504 added) misses FD by the taper-derivative size while the
//! energy stays smooth, so the miss does NOT shrink with `h` (a constant
//! offset, which the Richardson combination below exposes); a wrong switched
//! cell factor or presence product fails the `W → 0` anchor, which compares
//! against the HARD kernel (an independent code path), not against itself.
//! The anchor is blind to the presence DERIVATIVE (it vanishes when every
//! presence is 0 or 1), so the FD rows carry that mutant.
//!
//! Exactness anchor (per path, before the FD): at `W = 1e-9` Bohr no image
//! lies inside the taper zone, every presence is exactly 1 or 0, and the
//! switched kernel reduces to the hard one up to rounding (`1 − (1 − s)` vs
//! `s`), so energy, force and stress must equal the `W = 0` (hard) run to SCF
//! noise.
//!
//! Fixture: H2 (closed shell) / H3 doublet (`N_α = 2, N_β = 1`) / STO-3G in
//! the 4 Bohr cube, atom grid (20, 50) SSF, explicit `D = 10`, `W = 2`,
//! exxdiv none, dense AFT — #504's cell and grid, so the bars transfer.
//!
//! # Measured (2026-10-25, release, 3 threads; misses in Ha/Bohr and Ha)
//!
//! | path | anchor max | F h=1e-4 / 5e-5 / Rich. | dE/dε h=1e-4 / 5e-5 / Rich. | E(W=2) − E(hard) |
//! |---|---|---|---|---|
//! | Gamma UKS PBE | 5.8e-15 | 1.7e-9 / 1.1e-9 / 9.8e-10 | 1.5e-7 / 4.0e-8 / 2.5e-9 | +5.12e-6 |
//! | Gamma ROKS PBE | 7.9e-15 | 1.8e-9 / 1.1e-9 / 9.9e-10 | 1.6e-7 / 4.3e-8 / 2.5e-9 | +5.13e-6 |
//! | k UKS PBE 1x1x2 | 5.6e-16 | 1.5e-9 / 8.6e-10 / 6.5e-10 | 8.8e-8 / 2.3e-8 / 1.3e-9 | +3.54e-6 |
//! | k ROKS PBE 1x1x2 | 1.7e-15 | 1.5e-9 / 7.8e-10 / 5.5e-10 | 9.0e-8 / 2.4e-8 / 1.4e-9 | +3.54e-6 |
//! | Gamma RKS SCAN | 8.9e-16 | 2.3e-9 / 7.3e-10 / 1.9e-10 | 3.1e-8 / 7.5e-9 / 3.1e-10 | +7.58e-6 |
//! | k UKS SCAN 1x1x1 | 7.2e-15 | 4.8e-9 / 1.5e-9 / 9.5e-10 | 3.3e-7 / 8.4e-8 / 2.5e-9 | +5.38e-6 |
//! | Gamma RKS CAM-B3LYP | 2.8e-17 | 2.4e-9 / 7.4e-10 / 2.0e-10 | 1.5e-8 / 3.4e-9 / 3.1e-10 | +3.73e-6 |
//! | Gamma UKS CAM-B3LYP (force) | 6.1e-16 | 1.8e-9 / 1.1e-9 / 9.6e-10 | — | +2.65e-6 |
//! | Gamma ROKS CAM-B3LYP (force) | 1.6e-15 | 1.8e-9 / 1.1e-9 / 9.5e-10 | — | +2.65e-6 |
//!
//! # Mutation ledger (source mutants in `ferric_dft::becke::switched_weight`,
//! applied by hand, all nine tests run, then reverted; 2026-10-25)
//!
//! * M1, the image presence derivative dropped (`cf = 0`): ALL NINE fail.
//!   Force miss 1.73e-6 .. 3.97e-6, stress 2.72e-5 .. 5.54e-5, identical at
//!   h = 1e-4, 5e-5 and Richardson (a constant offset). The anchors pass
//!   (blind by construction, see above).
//! * M4, the cell factor `1 − f_C s` instead of `1 − f_C (1 − s)`: ALL NINE
//!   fail at the anchor (|ΔE| 0.37 .. 1.22 Ha, |ΔF| 2.7e-2 .. 4.6e-2).
//!
//! Every SCF test is `#[ignore]`d; run with
//! `cargo test --release -p ferric-pbc --test validation_pbc_grid_switch_paths -- --ignored --nocapture --test-threads=1`.

mod common;

use common::*;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{
    gamma_rks, gamma_rks_rsh, gamma_uks, gamma_uks_rsh, GammaRksConfig, GammaUksConfig,
    PeriodicGridConfig,
};
use ferric_pbc::gamma_mgga::{
    gamma_rks_mgga_gradient, gamma_rks_mgga_stress, GammaMggaDerivConfig,
};
use ferric_pbc::gamma_rsh::{
    gamma_rks_rsh_gradient, gamma_rks_rsh_stress, gamma_roks_rsh_gradient, gamma_uks_rsh_gradient,
    GammaRshExchange,
};
use ferric_pbc::grad::{gamma_roks_gradient_with, gamma_uks_gradient_with, GammaGradConfig};
use ferric_pbc::hcore::kpoint::periodic_hcore_kpts;
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kgrad::{KGradConfig, KGradJk};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::{solve_kroks, KRoksConfig};
use ferric_pbc::kstress::{KStressConfig, KStressJk};
use ferric_pbc::kuks::{solve_kuks, KUksConfig};
use ferric_pbc::kuxcgrad::{kpoint_roks_gradient, kpoint_uks_gradient, KUksGradConfig};
use ferric_pbc::kuxcstress::{kpoint_roks_stress, kpoint_uks_stress, KUksStressConfig};
use ferric_pbc::kxcgrad::{KRksGradConfig, KXcGrid};
use ferric_pbc::kxcstress::KRksStressConfig;
use ferric_pbc::lattice::Cell;
use ferric_pbc::rohf::{gamma_roks, gamma_roks_rsh, GammaRoksConfig};
use ferric_pbc::rsh::resolve_periodic_functional_rsh;
use ferric_pbc::stress::{gamma_roks_stress, gamma_uks_stress, GammaStressConfig, Mat3};
use ferric_pbc::uhf::GammaUhfIntegrals;
use ndarray::Array2;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
/// H3 doublet (N_alpha = 2, N_beta = 1): spin-polarized, Da != Db.
const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];
/// Explicit neighbour cutoff D (Bohr), #504's.
const CUT: f64 = 10.0;
/// Taper width W of the switched runs (Bohr), #504's.
const WIDTH: f64 = 2.0;
/// The `W → 0` anchor's width: no image within 1e-9 Bohr of D here.
const TINY: f64 = 1e-9;
const H: f64 = 1e-4;
const CAM: &str = "HYB_GGA_XC_CAM_B3LYP";

// Bars, derived as #504's from the measured table in the module doc (worst
// switched floor over the nine paths vs the smallest mutant miss):
/// Force vs FD at h = 1e-4: #504's bar. Worst floor 4.8e-9 (k UKS SCAN);
/// the dropped presence derivative misses by >= 1.73e-6 (Gamma RKS CAM-B3LYP).
const FD_BAR: f64 = 1e-7;
/// Stress vs FD at h = 1e-4. The H3 cells carry a larger h² truncation than
/// #504's H2 cell: worst 3.27e-7 (k UKS SCAN), falling ≈3.7-4.4x per halving
/// of h on every row (truncation, not a defect), so #504's 1e-7 would fail on
/// arithmetic; the bar is 3x the worst floor. Dropped presence derivative:
/// >= 2.72e-5.
const STRESS_FD_BAR: f64 = 1e-6;
/// The force misses sit at the SCF noise floor (~1e-9; h → h/2 ratios
/// 1.5-3, not h²-limited), so for the force this bar tests noise against
/// 2e-8, not truncation; for the stress it removes a real h² term.
/// Richardson `(4 FD(h/2) − FD(h)) / 3` vs analytic, force and stress: the
/// h² term cancels, so this is the sharp check. Worst floor 2.53e-9 (k UKS
/// SCAN stress); dropped presence derivative >= 1.73e-6 (its miss is the
/// same at h, h/2 and Richardson: a constant offset).
const RICHARDSON_BAR: f64 = 2e-8;
/// `W = 1e-9` vs `W = 0`: worst 7.9e-15 (energy, force, stress; two
/// separately converged SCFs land on the same digits); the `t = 1 − f_C s`
/// cell-factor mutant misses by >= 2.7e-2 (force) and >= 0.37 Ha (energy).
const ANCHOR_BAR: f64 = 1e-12;

fn grid(width: f64) -> PeriodicGridConfig {
    PeriodicGridConfig {
        neighbour_cutoff: Some(CUT),
        neighbour_switch: width,
        ..PeriodicGridConfig::with_size(20, 50)
    }
}

fn hcfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig::with_omega(OMEGA)
}

fn cell_of(pos: &[[f64; 3]]) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = if pos.len() % 2 == 1 { 2 } else { 1 };
    Cell::new(mol, cubic(4.0)).expect("cell")
}

fn moved(cell: &Cell, a: usize, x: usize, h: f64) -> Cell {
    let mut pos = cell.positions();
    pos[a][x] += h;
    let mut mol = hydrogens(&pos);
    mol.multiplicity = cell.mol().multiplicity;
    Cell::new(mol, *cell.lattice()).expect("cell")
}

fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    cell.strained(&e).expect("strained")
}

fn mdiff(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

fn adiff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn prep_of(cell: &Cell) -> ferric_integrals::basis_bridge::PreparedBasis {
    prep_for(cell, &pyscf_sto3g_h())
}

fn dense_gamma(
    cell: &Cell,
    prep: &ferric_integrals::basis_bridge::PreparedBasis,
    s: &Array2<f64>,
) -> DenseAftEri {
    DenseAftEri::build(
        cell,
        prep,
        s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap()
}

fn ggrad() -> GammaGradConfig {
    GammaGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

fn gstress() -> GammaStressConfig {
    GammaStressConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

fn kgrad() -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

fn kstress() -> KStressConfig {
    KStressConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

fn kugrad(width: f64) -> KUksGradConfig {
    KUksGradConfig {
        base: KRksGradConfig {
            grad: kgrad(),
            grid: KXcGrid::Atom(grid(width)),
            ..Default::default()
        },
        spin_mutation: None,
    }
}

fn kustress(width: f64) -> KUksStressConfig {
    KUksStressConfig {
        base: KRksStressConfig {
            stress: kstress(),
            grid: KXcGrid::Atom(grid(width)),
            ..Default::default()
        },
        spin_mutation: None,
    }
}

fn rsh_exchange(
    cell: &Cell,
    prep: &ferric_integrals::basis_bridge::PreparedBasis,
    s: &Array2<f64>,
) -> GammaRshExchange {
    let rsh = resolve_periodic_functional_rsh(CAM).unwrap().2.unwrap();
    GammaRshExchange::build_dense(
        cell,
        prep,
        s,
        ExxDiv::None,
        &KDenseAftConfig::default(),
        rsh,
    )
    .unwrap()
}

// ------------------------------------------------------------------ runners

/// One path at one geometry: the SCF energy and, when asked, `dE/dR` and
/// `dE/dε` (`None` where the path has no stress).
struct Out {
    e: f64,
    grad: Option<Array2<f64>>,
    de: Option<Mat3>,
}

impl Out {
    fn energy(e: f64) -> Self {
        Self {
            e,
            grad: None,
            de: None,
        }
    }
}

fn gamma_uks_cfg(f: &str, width: f64) -> GammaUksConfig {
    let mut c = GammaUksConfig::new(f);
    c.grid = grid(width);
    c.exxdiv = ExxDiv::None;
    c.scf.energy_conv = 1e-13;
    c.scf.density_conv = 1e-10;
    c.scf.max_iter = 300;
    c
}

fn gamma_roks_cfg(f: &str, width: f64) -> GammaRoksConfig {
    let mut c = GammaRoksConfig::new(f);
    c.grid = grid(width);
    c.exxdiv = ExxDiv::None;
    c.scf.energy_conv = 1e-13;
    c.scf.density_conv = 1e-10;
    c
}

fn gamma_rks_cfg(f: &str, width: f64) -> GammaRksConfig {
    let mut c = GammaRksConfig::new(f);
    c.grid = grid(width);
    c.exxdiv = ExxDiv::None;
    c.scf.energy_conv = 1e-13;
    c.scf.density_conv = 1e-10;
    c.scf.max_iter = 300;
    c
}

/// Native Gamma UKS (PBE): `grad.rs` / `stress.rs`, spin-polarized XC.
fn run_gamma_uks(cell: &Cell, width: f64, derivs: bool) -> Out {
    let prep = prep_of(cell);
    let hc = periodic_hcore(cell, &prep, &hcfg()).unwrap();
    let eri = dense_gamma(cell, &prep, &hc.s);
    let c = gamma_uks_cfg("PBE", width);
    let r = gamma_uks(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &c).unwrap();
    assert!(r.scf.converged);
    if !derivs {
        return Out::energy(r.scf.energy);
    }
    let g = gamma_uks_gradient_with(cell, &prep, &hcfg(), &hc, &eri, &r.scf, &c, &ggrad()).unwrap();
    let s = gamma_uks_stress(cell, &prep, &hcfg(), &hc, &eri, &r.scf, &c, &gstress()).unwrap();
    Out {
        e: r.scf.energy,
        grad: Some(g.grad),
        de: Some(s.de_deps),
    }
}

/// Native Gamma ROKS (PBE): the UKS assembly on the ROHF spin densities.
fn run_gamma_roks(cell: &Cell, width: f64, derivs: bool) -> Out {
    let prep = prep_of(cell);
    let hc = periodic_hcore(cell, &prep, &hcfg()).unwrap();
    let eri = dense_gamma(cell, &prep, &hc.s);
    let c = gamma_roks_cfg("PBE", width);
    let r = gamma_roks(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &c).unwrap();
    assert!(r.scf.converged);
    if !derivs {
        return Out::energy(r.scf.energy);
    }
    let g =
        gamma_roks_gradient_with(cell, &prep, &hcfg(), &hc, &eri, &r.scf, &c, &ggrad()).unwrap();
    let s = gamma_roks_stress(cell, &prep, &hcfg(), &hc, &eri, &r.scf, &c, &gstress()).unwrap();
    Out {
        e: r.scf.energy,
        grad: Some(g.grad),
        de: Some(s.de_deps),
    }
}

/// k-point UKS (`f` = PBE or SCAN) on the Gamma-centred mesh `n`.
fn run_k_uks(f: &str, n: [usize; 3], cell: &Cell, width: f64, derivs: bool) -> Out {
    let prep = prep_of(cell);
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let mut c = KUksConfig::new(cell, ExxDiv::None, f);
    c.kuhf.hcore = hcfg();
    c.kuhf.scf.energy_conv = 1e-13;
    c.kuhf.scf.grad_conv = 1e-10;
    c.kuhf.scf.max_iter = 300;
    c.grid = grid(width);
    let r = solve_kuks(cell, &prep, None, &mesh, &c).unwrap();
    assert!(r.scf.converged);
    if !derivs {
        return Out::energy(r.scf.energy);
    }
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &hcfg()).unwrap();
    let eri =
        KDenseAftEri::build(cell, &prep, &mesh, &hk.s, ExxDiv::None, &Default::default()).unwrap();
    let g = kpoint_uks_gradient(
        cell,
        &prep,
        &mesh,
        &hcfg(),
        &hk,
        KGradJk::Dense(&eri),
        &r.scf,
        f,
        &kugrad(width),
    )
    .unwrap();
    let s = kpoint_uks_stress(
        cell,
        &prep,
        &mesh,
        &hcfg(),
        &hk,
        KStressJk::Dense(&eri),
        &r.scf,
        f,
        &kustress(width),
    )
    .unwrap();
    Out {
        e: r.scf.energy,
        grad: Some(g.grad),
        de: Some(s.de_deps),
    }
}

/// k-point ROKS (PBE) on the Gamma-centred mesh `n`.
fn run_k_roks(n: [usize; 3], cell: &Cell, width: f64, derivs: bool) -> Out {
    let prep = prep_of(cell);
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let mut c = KRoksConfig::new(cell, ExxDiv::None, "PBE");
    c.rohf.hcore = hcfg();
    c.rohf.scf.energy_conv = 1e-13;
    c.rohf.scf.grad_conv = 1e-10;
    c.rohf.scf.max_iter = 300;
    c.grid = grid(width);
    let r = solve_kroks(cell, &prep, None, &mesh, &c).unwrap();
    assert!(r.scf.converged);
    if !derivs {
        return Out::energy(r.scf.energy);
    }
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &hcfg()).unwrap();
    let eri =
        KDenseAftEri::build(cell, &prep, &mesh, &hk.s, ExxDiv::None, &Default::default()).unwrap();
    let g = kpoint_roks_gradient(
        cell,
        &prep,
        &mesh,
        &hcfg(),
        &hk,
        KGradJk::Dense(&eri),
        &r.scf,
        "PBE",
        &kugrad(width),
    )
    .unwrap();
    let s = kpoint_roks_stress(
        cell,
        &prep,
        &mesh,
        &hcfg(),
        &hk,
        KStressJk::Dense(&eri),
        &r.scf,
        "PBE",
        &kustress(width),
    )
    .unwrap();
    Out {
        e: r.scf.energy,
        grad: Some(g.grad),
        de: Some(s.de_deps),
    }
}

/// Closed-shell Gamma SCAN, derivatives routed through the 1x1x1 k code.
fn run_gamma_scan(cell: &Cell, width: f64, derivs: bool) -> Out {
    let prep = prep_of(cell);
    let hc = periodic_hcore(cell, &prep, &hcfg()).unwrap();
    let eri = dense_gamma(cell, &prep, &hc.s);
    let c = gamma_rks_cfg("SCAN", width);
    let r = gamma_rks(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &c).unwrap();
    assert!(r.scf.converged);
    if !derivs {
        return Out::energy(r.scf.energy);
    }
    let d = GammaMggaDerivConfig {
        kdense: KDenseAftConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        grad: kgrad(),
        stress: kstress(),
        ..Default::default()
    };
    let g = gamma_rks_mgga_gradient(cell, &prep, &hcfg(), &r.scf, &c, &d).unwrap();
    let s = gamma_rks_mgga_stress(cell, &prep, &hcfg(), &r.scf, &c, &d).unwrap();
    Out {
        e: r.scf.energy,
        grad: Some(g.grad),
        de: Some(s.de_deps),
    }
}

/// Closed-shell Gamma CAM-B3LYP force and stress (1x1x1 k route).
fn run_gamma_rks_rsh(cell: &Cell, width: f64, derivs: bool) -> Out {
    let prep = prep_of(cell);
    let hc = periodic_hcore(cell, &prep, &hcfg()).unwrap();
    let eri = dense_gamma(cell, &prep, &hc.s);
    let ex = rsh_exchange(cell, &prep, &hc.s);
    let c = gamma_rks_cfg(CAM, width);
    let r = gamma_rks_rsh(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &ex, &c).unwrap();
    assert!(r.scf.converged);
    if !derivs {
        return Out::energy(r.scf.energy);
    }
    let g = gamma_rks_rsh_gradient(cell, &prep, &hcfg(), &r.scf, &c, &ex, &kgrad()).unwrap();
    let s = gamma_rks_rsh_stress(cell, &prep, &hcfg(), &r.scf, &c, &ex, &kstress()).unwrap();
    Out {
        e: r.scf.energy,
        grad: Some(g.grad),
        de: Some(s.de_deps),
    }
}

/// Open-shell Gamma UKS CAM-B3LYP force (1x1x1 k UKS route; no stress).
fn run_gamma_uks_rsh(cell: &Cell, width: f64, derivs: bool) -> Out {
    let prep = prep_of(cell);
    let hc = periodic_hcore(cell, &prep, &hcfg()).unwrap();
    let eri = dense_gamma(cell, &prep, &hc.s);
    let ex = rsh_exchange(cell, &prep, &hc.s);
    let c = gamma_uks_cfg(CAM, width);
    let r = gamma_uks_rsh(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &ex, &c).unwrap();
    assert!(r.scf.converged);
    if !derivs {
        return Out::energy(r.scf.energy);
    }
    let g = gamma_uks_rsh_gradient(cell, &prep, &hcfg(), &r.scf, &c, &ex, &kgrad()).unwrap();
    Out {
        e: r.scf.energy,
        grad: Some(g.grad),
        de: None,
    }
}

/// Open-shell Gamma ROKS CAM-B3LYP force (1x1x1 k ROKS route; no stress).
fn run_gamma_roks_rsh(cell: &Cell, width: f64, derivs: bool) -> Out {
    let prep = prep_of(cell);
    let hc = periodic_hcore(cell, &prep, &hcfg()).unwrap();
    let eri = dense_gamma(cell, &prep, &hc.s);
    let ex = rsh_exchange(cell, &prep, &hc.s);
    let c = gamma_roks_cfg(CAM, width);
    let r = gamma_roks_rsh(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &ex, &c).unwrap();
    assert!(r.scf.converged);
    if !derivs {
        return Out::energy(r.scf.energy);
    }
    let g = gamma_roks_rsh_gradient(cell, &prep, &hcfg(), &r.scf, &c, &ex, &kgrad()).unwrap();
    Out {
        e: r.scf.energy,
        grad: Some(g.grad),
        de: None,
    }
}

// ------------------------------------------------------------- the check

/// Central FD of the energy at step `h`: every force component (atom-major,
/// `3 natoms`) and, when `stress`, the nine strain components (row-major).
fn fd_vector(cell: &Cell, h: f64, stress: bool, energy: &dyn Fn(&Cell) -> f64) -> Vec<f64> {
    let mut v = Vec::new();
    for a in 0..cell.positions().len() {
        for x in 0..3 {
            v.push((energy(&moved(cell, a, x, h)) - energy(&moved(cell, a, x, -h))) / (2.0 * h));
        }
    }
    if stress {
        for i in 0..3 {
            for j in 0..3 {
                v.push(
                    (energy(&strained(cell, i, j, h)) - energy(&strained(cell, i, j, -h)))
                        / (2.0 * h),
                );
            }
        }
    }
    v
}

/// The analytic counterpart of [`fd_vector`].
fn analytic_vector(out: &Out) -> Vec<f64> {
    let mut v: Vec<f64> = out.grad.as_ref().expect("force").iter().copied().collect();
    if let Some(de) = &out.de {
        v.extend(de.iter().flatten());
    }
    v
}

/// `(max force miss, max stress miss)` of `fd` against `an` (`n_force`
/// leading force entries).
fn misses(fd: &[f64], an: &[f64], n_force: usize) -> (f64, Option<f64>) {
    let worst = |r: std::ops::Range<usize>| r.map(|i| (fd[i] - an[i]).abs()).fold(0.0, f64::max);
    let f = worst(0..n_force);
    let s = (an.len() > n_force).then(|| worst(n_force..an.len()));
    (f, s)
}

/// The per-path check: the `W → 0` anchor against the hard run, then the
/// switched force (and stress) against the central FD of the switched energy
/// at `h = 1e-4` (#504's bar), at `h / 2`, and their Richardson combination
/// `(4 FD(h/2) − FD(h)) / 3`, which cancels the `h²` truncation and leaves
/// any constant (derivative) defect.
fn check(label: &str, cell: &Cell, run: &dyn Fn(&Cell, f64, bool) -> Out) {
    let hard = run(cell, 0.0, true);
    let tiny = run(cell, TINY, true);
    let ae = (hard.e - tiny.e).abs();
    let ag = adiff(hard.grad.as_ref().unwrap(), tiny.grad.as_ref().unwrap());
    let as_ = match (&hard.de, &tiny.de) {
        (Some(a), Some(b)) => mdiff(a, b),
        _ => 0.0,
    };
    eprintln!("{label}: anchor W=1e-9 vs hard: |dE| {ae:.2e} |dF| {ag:.2e} |d(dE/de)| {as_:.2e}");
    assert!(
        ae.max(ag).max(as_) < ANCHOR_BAR,
        "{label}: W -> 0 does not reproduce the hard cutoff: {ae:e} {ag:e} {as_:e}"
    );
    let sw = run(cell, WIDTH, true);
    let en = |c: &Cell| run(c, WIDTH, false).e;
    let an = analytic_vector(&sw);
    let n_force = 3 * cell.positions().len();
    let fd1 = fd_vector(cell, H, sw.de.is_some(), &en);
    let fd2 = fd_vector(cell, H / 2.0, sw.de.is_some(), &en);
    let rich: Vec<f64> = fd1
        .iter()
        .zip(&fd2)
        .map(|(a, b)| (4.0 * b - a) / 3.0)
        .collect();
    let (f1, s1) = misses(&fd1, &an, n_force);
    let (f2, s2) = misses(&fd2, &an, n_force);
    let (fr, sr) = misses(&rich, &an, n_force);
    let fmt = |s: Option<f64>| s.map_or("n/a".into(), |s| format!("{s:.2e}"));
    eprintln!(
        "{label}: W={WIDTH} E {:.12} (switched - hard {:+.3e}); |F-FD| h=1e-4 {f1:.2e} \
         h=5e-5 {f2:.2e} Richardson {fr:.2e}; |dE/de-FD| h=1e-4 {} h=5e-5 {} Richardson {}",
        sw.e,
        sw.e - hard.e,
        fmt(s1),
        fmt(s2),
        fmt(sr)
    );
    assert!(
        f1 < FD_BAR,
        "{label}: switched force misses FD(h=1e-4) by {f1:e}"
    );
    assert!(
        fr < RICHARDSON_BAR,
        "{label}: switched force misses Richardson FD by {fr:e}"
    );
    if let (Some(s), Some(r)) = (s1, sr) {
        assert!(
            s < STRESS_FD_BAR,
            "{label}: switched stress misses FD(h=1e-4) by {s:e}"
        );
        assert!(
            r < RICHARDSON_BAR,
            "{label}: switched stress misses Richardson FD by {r:e}"
        );
    }
}

#[test]
#[ignore = "validation: Gamma UKS PBE H3, 75 SCFs; measured 78 s release"]
fn gamma_uks_force_and_stress_match_fd_with_the_switch() {
    check("Gamma UKS PBE", &cell_of(&H3), &run_gamma_uks);
}

#[test]
#[ignore = "validation: Gamma ROKS PBE H3, 75 SCFs; measured ~80 s release"]
fn gamma_roks_force_and_stress_match_fd_with_the_switch() {
    check("Gamma ROKS PBE", &cell_of(&H3), &run_gamma_roks);
}

#[test]
#[ignore = "validation: k UKS PBE H3 1x1x2, 75 k SCFs; ~2 min release"]
fn kmesh_uks_force_and_stress_match_fd_with_the_switch() {
    check("k UKS PBE 1x1x2", &cell_of(&H3), &|c, w, d| {
        run_k_uks("PBE", [1, 1, 2], c, w, d)
    });
}

#[test]
#[ignore = "validation: k ROKS PBE H3 1x1x2, 75 k SCFs; ~2 min release"]
fn kmesh_roks_force_and_stress_match_fd_with_the_switch() {
    check("k ROKS PBE 1x1x2", &cell_of(&H3), &|c, w, d| {
        run_k_roks([1, 1, 2], c, w, d)
    });
}

#[test]
#[ignore = "validation: Gamma RKS SCAN H2, 63 SCFs; ~1 min release"]
fn gamma_scan_force_and_stress_match_fd_with_the_switch() {
    check("Gamma RKS SCAN", &cell_of(&H2), &run_gamma_scan);
}

#[test]
#[ignore = "validation: k UKS SCAN H3 1x1x1, 75 k SCFs; ~2 min release"]
fn kpoint_uks_scan_force_and_stress_match_fd_with_the_switch() {
    check("k UKS SCAN 1x1x1", &cell_of(&H3), &|c, w, d| {
        run_k_uks("SCAN", [1, 1, 1], c, w, d)
    });
}

#[test]
#[ignore = "validation: Gamma RKS CAM-B3LYP H2, 63 SCFs; ~1 min release"]
fn gamma_rsh_force_and_stress_match_fd_with_the_switch() {
    check("Gamma RKS CAM-B3LYP", &cell_of(&H2), &run_gamma_rks_rsh);
}

#[test]
#[ignore = "validation: Gamma UKS CAM-B3LYP H3 force, 39 SCFs; ~1 min release"]
fn gamma_uks_rsh_force_matches_fd_with_the_switch() {
    check("Gamma UKS CAM-B3LYP", &cell_of(&H3), &run_gamma_uks_rsh);
}

#[test]
#[ignore = "validation: Gamma ROKS CAM-B3LYP H3 force, 39 SCFs; ~1 min release"]
fn gamma_roks_rsh_force_matches_fd_with_the_switch() {
    check("Gamma ROKS CAM-B3LYP", &cell_of(&H3), &run_gamma_roks_rsh);
}
