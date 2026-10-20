//! Validation of the smooth neighbour switch on the periodic atom-centred XC
//! grid (`PeriodicGridConfig::neighbour_switch`, issue #495).
//!
//! The hard neighbour mask (image atoms within `D` of a grid point enter the
//! partition) makes `E(R, ε)` jump wherever an image crosses `D` (measured
//! here up to 2.2e-7 Ha on the (20, 50) grid at D = 10, and a 3.6e-5 Ha jump
//! on an s+p triclinic cell at D = 10 in `pbc_kxcstress_mgga.rs`; the size
//! is grid- and cell-dependent); the analytic force and stress cannot see the jumps, so a
//! central FD whose step straddles one misses by `J / 2h`. The switch tapers
//! each image's presence over `[D − W, D]` and differentiates the taper.
//!
//! Physics hypothesis: with `W > 0`, `E` is smooth, so (i) a dense strain or
//! displacement scan has third differences that follow the smooth `δ³` law
//! with no isolated jump, and (ii) the analytic force/stress match the FD at
//! `h = 1e-4` to the `h²` truncation without step tuning. Artifact
//! hypothesis: if the switch derivative were missing or mis-signed, (ii)
//! fails at the taper-derivative size while (i) still passes (the energy is
//! smooth either way) — the two checks are independent; if the switch were
//! not wired into the SCF grid at all, (i) would show the same jumps as the
//! hard mask.
//!
//! Fixture (the one where `pbc_kxcstress.rs`'s window probe found a mask
//! jump at `h = 1e-4`): H2/STO-3G in a 4 Bohr cubic cell, PBE, (20, 50),
//! explicit `D = 10`, dense AFT, Gamma and a 1x1x3 mesh.
//!
//! Every test here is `#[ignore]`d (SCF-heavy); run with
//! `cargo test -p ferric-pbc --test validation_pbc_grid_switch -- --ignored --nocapture --test-threads=1`.
//! The grid-level (SCF-free) anchors and the mutation ledger of the switch
//! derivative run per commit: `ferric_pbc::dft::switch_tests` and
//! `ferric_dft::becke::partition_over_tests`.

mod common;

use common::*;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::{gamma_rks, GammaRksConfig, PeriodicGridConfig};
use ferric_pbc::grad::{gamma_rks_gradient_with, GammaGradConfig};
use ferric_pbc::hcore::kpoint::periodic_hcore_kpts;
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kdft::{solve_krks, KRksConfig};
use ferric_pbc::kgrad::{KGradConfig, KGradJk};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::KScfResult;
use ferric_pbc::kstress::{KStressConfig, KStressJk};
use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KXcGrid};
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStressConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::stress::{gamma_rks_stress, GammaStressConfig, Mat3};
use ferric_pbc::uhf::GammaUhfIntegrals;
use ndarray::Array2;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const CUT: f64 = 10.0;
/// Taper width of the switched runs (Bohr).
const WIDTH: f64 = 2.0;
const MESH: [usize; 3] = [1, 1, 3];
const H: f64 = 1e-4;

fn grid(width: f64) -> PeriodicGridConfig {
    PeriodicGridConfig {
        neighbour_cutoff: Some(CUT),
        neighbour_switch: width,
        ..PeriodicGridConfig::with_size(20, 50)
    }
}

fn cell_of(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(4.0)).expect("cell")
}

fn moved(cell: &Cell, a: usize, x: usize, h: f64) -> Cell {
    let mut pos = cell.positions();
    pos[a][x] += h;
    Cell::new(hydrogens(&pos), *cell.lattice()).expect("cell")
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

// --------------------------------------------------------------- k-mesh side

fn kcfg(cell: &Cell, width: f64) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, ExxDiv::None, "PBE");
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 200;
    cfg.grid = grid(width);
    cfg
}

fn k_scf(cell: &Cell, width: f64) -> KScfResult {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(cell, MESH).unwrap();
    let r = solve_krks(cell, &prep, None, &mesh, &kcfg(cell, width)).expect("solve_krks");
    assert!(r.scf.converged);
    r.scf
}

fn k_energy(cell: &Cell, width: f64) -> f64 {
    k_scf(cell, width).energy
}

/// `(energy, dE/dR, dE/dε)` of the k-mesh RKS at `cell`.
fn k_derivs(cell: &Cell, width: f64) -> (f64, Array2<f64>, Mat3) {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(cell, MESH).unwrap();
    let cfg = kcfg(cell, width);
    let scf = k_scf(cell, width);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &hcfg).unwrap();
    let eri =
        KDenseAftEri::build(cell, &prep, &mesh, &hk.s, ExxDiv::None, &Default::default()).unwrap();
    let g = kpoint_rks_gradient(
        cell,
        &prep,
        &mesh,
        &hcfg,
        &hk,
        KGradJk::Dense(&eri),
        &scf,
        "PBE",
        &KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            grid: KXcGrid::Atom(cfg.grid.clone()),
            ..Default::default()
        },
    )
    .expect("k gradient");
    let s = kpoint_rks_stress(
        cell,
        &prep,
        &mesh,
        &hcfg,
        &hk,
        KStressJk::Dense(&eri),
        &scf,
        "PBE",
        &KRksStressConfig {
            stress: KStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            grid: KXcGrid::Atom(cfg.grid.clone()),
            ..Default::default()
        },
    )
    .expect("k stress");
    (scf.energy, g.grad, s.de_deps)
}

// ---------------------------------------------------------------- Gamma side

fn gamma_cfg(width: f64) -> GammaRksConfig {
    let mut gc = GammaRksConfig::new("PBE");
    gc.grid = grid(width);
    gc.exxdiv = ExxDiv::None;
    gc.scf.energy_conv = 1e-13;
    gc.scf.density_conv = 1e-10;
    gc.scf.max_iter = 200;
    gc
}

/// `(energy, dE/dR, dE/dε)` of the Gamma RKS at `cell`; `derivs = false`
/// returns zeros for the derivatives.
fn gamma_run(cell: &Cell, width: f64, derivs: bool) -> (f64, Array2<f64>, Mat3) {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hc = periodic_hcore(cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        1e-12,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let gc = gamma_cfg(width);
    let r = gamma_rks(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &gc).unwrap();
    assert!(r.scf.converged);
    if !derivs {
        return (r.scf.energy, Array2::zeros((2, 3)), [[0.0; 3]; 3]);
    }
    let g = gamma_rks_gradient_with(
        cell,
        &prep,
        &hcfg,
        &hc,
        &eri,
        &r.scf,
        &gc,
        &GammaGradConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
    )
    .unwrap();
    let s = gamma_rks_stress(
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
    .unwrap();
    (r.scf.energy, g.grad, s.de_deps)
}

// ------------------------------------------------------------ FD comparison

/// Worst `|analytic − central FD(h)|` over all six force components and all
/// nine strain components.
fn fd_misses(
    cell: &Cell,
    grad: &Array2<f64>,
    de: &Mat3,
    h: f64,
    energy: &dyn Fn(&Cell) -> f64,
) -> (f64, f64) {
    let mut fmiss = 0.0_f64;
    for a in 0..2 {
        for x in 0..3 {
            let fd = (energy(&moved(cell, a, x, h)) - energy(&moved(cell, a, x, -h))) / (2.0 * h);
            fmiss = fmiss.max((fd - grad[(a, x)]).abs());
        }
    }
    let mut fd9 = [[0.0; 3]; 3];
    for (i, row) in fd9.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (energy(&strained(cell, i, j, h)) - energy(&strained(cell, i, j, -h))) / (2.0 * h);
        }
    }
    (fmiss, mdiff(&fd9, de))
}

/// FD bar at h = 1e-4 (force and stress). MEASURED (W = 2, D = 10):
/// switched force 2.7e-9 (k) / 2.3e-9 (Gamma), stress 2.5e-8 / 2.9e-8, the
/// h² truncation (5.9e-9 / 7.1e-9 at h = 5e-5, ÷4 per halving). The hard mask
/// on the same cells: stress 2.0e-6 (k) / 2.9e-6 (Gamma) at h = 1e-4 (a jump
/// straddled; 5.9e-9 / 7.0e-9 at h = 5e-5, where none is). The bar sits 3-4x
/// above the switched stress floor and 20x below the hard-mask miss; the
/// switch-derivative mutants are caught per commit (grid-level ledger in
/// `ferric_pbc::dft::switch_tests`: ≥ 2.4e-4 force, ≥ 2.6e-3 stress).
const FD_BAR: f64 = 1e-7;

/// (ii) k-mesh: analytic force and stress vs central FD at h = 1e-4, hard
/// mask and switch side by side, plus the energy shift the switch costs.
#[test]
#[ignore = "validation: 120 k-mesh SCFs (H2 1x1x3 PBE); measured ~3 min release"]
fn kmesh_force_and_stress_match_fd_at_h_1e4_with_the_switch() {
    let cell = cell_of(&H2);
    let mut rows = Vec::new();
    for width in [0.0, WIDTH] {
        let (e, g, s) = k_derivs(&cell, width);
        let en = |c: &Cell| k_energy(c, width);
        let (f1, s1) = fd_misses(&cell, &g, &s, H, &en);
        let (f2, s2) = fd_misses(&cell, &g, &s, H / 2.0, &en);
        eprintln!(
            "k 1x1x3 W={width}: E {e:.12}  |F-FD| h=1e-4 {f1:.2e} h=5e-5 {f2:.2e}  |dE/de-FD| h=1e-4 {s1:.2e} h=5e-5 {s2:.2e}"
        );
        rows.push((e, f1, s1));
    }
    eprintln!(
        "k energy shift switched - hard: {:.3e}",
        rows[1].0 - rows[0].0
    );
    assert!(rows[1].1 < FD_BAR, "switched force {:e}", rows[1].1);
    assert!(rows[1].2 < FD_BAR, "switched stress {:e}", rows[1].2);
}

/// (ii) Gamma: the same, Gamma RKS force and stress.
#[test]
#[ignore = "validation: 122 Gamma SCFs (H2 PBE); measured 57 s release"]
fn gamma_force_and_stress_match_fd_at_h_1e4_with_the_switch() {
    let cell = cell_of(&H2);
    let mut rows = Vec::new();
    for width in [0.0, WIDTH] {
        let (e, g, s) = gamma_run(&cell, width, true);
        let en = |c: &Cell| gamma_run(c, width, false).0;
        let (f1, s1) = fd_misses(&cell, &g, &s, H, &en);
        let (f2, s2) = fd_misses(&cell, &g, &s, H / 2.0, &en);
        eprintln!(
            "Gamma W={width}: E {e:.12}  |F-FD| h=1e-4 {f1:.2e} h=5e-5 {f2:.2e}  |dE/de-FD| h=1e-4 {s1:.2e} h=5e-5 {s2:.2e}"
        );
        rows.push((e, f1, s1));
    }
    eprintln!(
        "Gamma energy shift switched - hard: {:.3e}",
        rows[1].0 - rows[0].0
    );
    assert!(rows[1].1 < FD_BAR, "switched force {:e}", rows[1].1);
    assert!(rows[1].2 < FD_BAR, "switched stress {:e}", rows[1].2);
}

/// `(max |Δ³E|, max |Δ⁴E|, count of |Δ⁴E| > thresh)` over a scan.
///
/// A one-step jump `J` puts `|Δ⁴E| ≥ J` on the windows that straddle it
/// (coefficients 1, −4, 6, −4, 1), while a smooth `E` gives `E⁗ δ⁴`.
/// `Δ³` is reported too: on a smooth curve it is the `E‴ δ³` background
/// (under strain ~1.5e-11 at δ = 1e-4 here), which is why the jump test is
/// on `Δ⁴`.
fn differences(e: &[f64], thresh: f64) -> (f64, f64, usize) {
    let d3 = e
        .windows(4)
        .map(|w| (w[3] - 3.0 * w[2] + 3.0 * w[1] - w[0]).abs())
        .fold(0.0_f64, f64::max);
    let d4: Vec<f64> = e
        .windows(5)
        .map(|w| (w[4] - 4.0 * w[3] + 6.0 * w[2] - 4.0 * w[1] + w[0]).abs())
        .collect();
    (
        d3,
        d4.iter().fold(0.0_f64, |m, &x| m.max(x)),
        d4.iter().filter(|&&x| x > thresh).count(),
    )
}

/// Jump threshold on `|Δ⁴E|` at δ = 1e-4: the smooth part `E⁗ δ⁴` is far
/// below roundoff and the SCF noise (energy_conv 1e-13) gives `|Δ⁴E|` of a
/// few 1e-13; the hard-mask jumps on this cell are up to 2.2e-7 (measured
/// below), and the smallest worth catching are ~1e-11.
const JUMP: f64 = 1e-11;

/// (i) Dense scans on the 1x1x3 mesh: strain ε_11 over ±3e-3 and the
/// second H's x over ±3e-3 Bohr, step 1e-4, hard vs switched. MEASURED:
/// hard max|Δ⁴E| 6.7e-7 (ε_11, 55 of 57 windows above 1e-11) and 4.8e-8
/// (x, 16 above); the largest single-step jump, estimated from the first
/// differences, is 2.2e-7 Ha (ε_11) and 1.6e-8 Ha (x). Switched: max|Δ⁴E|
/// 2.4e-13 and 7.6e-14, none above 1e-11.
#[test]
#[ignore = "validation: 244 k-mesh SCFs; measured 194 s release"]
fn kmesh_dense_scans_have_no_jumps_with_the_switch() {
    let cell = cell_of(&H2);
    let n = 30_i32;
    let delta = 1e-4;
    for width in [0.0, WIDTH] {
        let strain: Vec<f64> = (-n..=n)
            .map(|i| k_energy(&strained(&cell, 1, 1, i as f64 * delta), width))
            .collect();
        let disp: Vec<f64> = (-n..=n)
            .map(|i| k_energy(&moved(&cell, 1, 0, i as f64 * delta), width))
            .collect();
        let (s3, s4, sc) = differences(&strain, JUMP);
        let (d3, d4, dc) = differences(&disp, JUMP);
        eprintln!(
            "k scan W={width}: eps_11 max|d3E| {s3:.2e} max|d4E| {s4:.2e} ({sc} > {JUMP:e}); \
             x(H1) max|d3E| {d3:.2e} max|d4E| {d4:.2e} ({dc} > {JUMP:e})"
        );
        eprintln!("k scan W={width} eps_11 energies: {strain:.15?}");
        eprintln!("k scan W={width} x(H1) energies: {disp:.15?}");
        if width > 0.0 {
            assert_eq!(sc + dc, 0, "switched scan has a jump");
        } else {
            // The scan is where the hard mask really jumps (artifact check).
            assert!(sc + dc > 0, "hard-mask scan shows no jump: probe misplaced");
        }
    }
}

/// The energy cost of the taper, against the cutoff convergence it rides on:
/// Gamma energies for the hard mask at D = 8, 10, 12, 14, 20, 22 and the
/// switch (W = 2) at D = 10, 12, 14, 22. The switch at (D, W) keeps full
/// presence to D − W and partial presence to D, so its energy is ASSERTED to
/// lie between the hard ones at D − W and D (within SCF noise). MEASURED (E − E(hard, D = 20)): hard D = 8 / 10 / 12 / 14:
/// +4.1e-5 / +3.9e-6 / −4.8e-8 / +6.6e-12; switched D = 10 / 12 / 14 / 22:
/// +1.1e-5 / +1.7e-7 / −3.8e-9 / 0. On this (20, 50) H2 cell the cutoff
/// itself converges slowly (D = 10 is 3.9e-6 off), and each switched energy
/// lands between the hard ones at D − W and D, as constructed.
#[test]
#[ignore = "validation: 10 Gamma SCFs (H2 PBE); measured 3.5 s"]
fn energy_shift_of_the_switch_against_cutoff_convergence() {
    let cell = cell_of(&H2);
    let e = |d: f64, w: f64| {
        let mut gc = gamma_cfg(w);
        gc.grid.neighbour_cutoff = Some(d);
        let prep = prep_for(&cell, &pyscf_sto3g_h());
        let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
        let hc = periodic_hcore(&cell, &prep, &hcfg).unwrap();
        let eri = DenseAftEri::build(
            &cell,
            &prep,
            &hc.s,
            ExxDiv::None,
            1e-12,
            DEFAULT_DENSE_AFT_MAX_BYTES,
        )
        .unwrap();
        gamma_rks(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &gc)
            .unwrap()
            .scf
            .energy
    };
    // Hard energies at every D the switched cases bracket, relative to D = 20.
    let reference = e(20.0, 0.0);
    let mut hard = std::collections::BTreeMap::new();
    for d in [8, 10, 12, 14, 20, 22] {
        let v = if d == 20 {
            0.0
        } else {
            e(d as f64, 0.0) - reference
        };
        eprintln!("Gamma D={d} W=0: E - E(hard, D=20) = {v:+.3e}");
        hard.insert(d, v);
    }
    // SCF noise allowance (energy_conv 1e-13 on both sides of each difference).
    let tol = 1e-11;
    for d in [10, 12, 14, 22] {
        let v = e(d as f64, WIDTH) - reference;
        let (lo, hi) = (hard[&(d - 2)], hard[&d]);
        eprintln!(
            "Gamma D={d} W={WIDTH}: E - E(hard, D=20) = {v:+.3e} (hard D-W {lo:+.3e}, D {hi:+.3e})"
        );
        assert!(
            v >= lo.min(hi) - tol && v <= lo.max(hi) + tol,
            "switched D={d} outside [hard D-W, hard D]: {v:e} vs [{lo:e}, {hi:e}]"
        );
    }
}
