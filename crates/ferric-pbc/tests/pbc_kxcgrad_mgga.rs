//! Analytic forces of the k-point closed-shell RKS energy for meta-GGA
//! functionals (SCAN, r2SCAN; `ferric_pbc::kxcgrad`, section "Meta-GGA: the
//! tau term").
//!
//! The Gamma periodic path has no meta-GGA, so there is no 1x1x1 = Gamma
//! anchor. The anchors are:
//! * `null_tau_path_reproduces_the_gga_force_exactly` — the meta-GGA code path
//!   (tau, V^tau, the tau force term) taken for PBE (`v_tau = 0`) is the PBE
//!   force bit for bit;
//! * `fixed_density_xc_force_matches_fd_of_exc` — the XC part of the force
//!   (X1 + X3 + X4) against the central difference of `E_xc[D]` at FIXED `D`
//!   (no SCF, so no convergence noise) with the atom grid rebuilt at each
//!   displacement, SCAN and r2SCAN, non-TRIM 1x1x3 mesh;
//! * `meta_gga_force_matches_fd_of_the_scf_energy` — the total force against
//!   FD of the converged SCF energy;
//! * `supercell_identity_per_part` — the 1x1x3 force per part equals the
//!   force on one copy of each atom of the explicit 3-cell supercell run at
//!   1x1x1 by the same code (this checks the k sum and Bloch phases, not the
//!   tau derivative itself);
//! * the uniform grid (X3/X4 absent), thread/chunk bit-identity, the budget
//!   gates, refusals and the Fock guard, and production-wired mutants.
//!
//! Also covered: TPSS through its raw single-component names
//! (`tpss_components_xc_force_matches_fd_of_exc`, fixed-D XC force; there is no
//! combined TPSS functional and no TPSS SCF), p/d shells on a triclinic cell
//! (`spd_triclinic_xc_force_matches_fd_of_exc`, ignored, fixed-D XC force) and
//! the RS-GDF J route by ONE ignored test, one force component of a SCAN SCF
//! (`rsgdf_scan_force_component_matches_fd`).
//!
//! NOT covered: hybrid meta-GGAs (none resolve), range-separated and VV10
//! meta-GGAs (refused), a TPSS SCF, RS-GDF beyond that one component and the
//! range-split fit, heavy atoms/ECPs, shifted meshes, metals, open shell,
//! stress, SCAN's alpha ~ 1 region (the H2 fixtures reach alpha ~ 0 only), and
//! the default (75, 302) grid.
//!
//! Bars. `FD_BAR = 1e-7`: the largest correct residual measured here is
//! 2.2e-9 (SCF-energy FD, r2SCAN), 2.3e-9 (uniform grid), 4.5e-10 (fixed-D,
//! p/d), 4.1e-11 (RS-GDF); 1e-7 is ~45x above the worst of them and ~5e4x
//! below the smallest mutant miss. `MUTANT_BAR = 1e-5`: the smallest
//! production-wired mutant miss is 4.73e-3 (`TauNoPoint`), ~470x above the
//! bar, and every correct comparison above is >= 3 orders below it. The FD
//! caveat: the energy has hard one-step discontinuities from the neighbour
//! cutoff, so FD of the SCF energy uses h = 1e-4 as in `pbc_kxcgrad.rs`.

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::{PeriodicGrid, PeriodicGridConfig, PeriodicXcConfig};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kdft::{solve_krks, solve_krks_on_grid, KPeriodicXc, KRksConfig, KRksResult};
use ferric_pbc::kgrad::{KGradConfig, KGradJk, KRsGdfGradSource};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{KJkKind, KPointXc, KScfResult};
use ferric_pbc::kxcgrad::{
    kpoint_rks_gradient, KRksGradConfig, KRksGradient, KXcGrid, KXcMutation,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const FD_BAR: f64 = 1e-7;
const MUTANT_BAR: f64 = 1e-5;

fn atom_grid(nr: usize, na: usize) -> KXcGrid {
    KXcGrid::Atom(PeriodicGridConfig::with_size(nr, na))
}

fn cell_of(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(4.0)).expect("cell")
}

fn krks_cfg(cell: &Cell, functional: &str, grid: &KXcGrid) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, ExxDiv::None, functional);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 300;
    if let KXcGrid::Atom(g) = grid {
        cfg.grid = g.clone();
    }
    cfg
}

fn scf_cell(cell: &Cell, mesh_n: [usize; 3], functional: &str, grid: &KXcGrid) -> KRksResult {
    let bs = pyscf_sto3g_h();
    let prep = prep_for(cell, &bs);
    let mesh = KPointMesh::gamma_centred(cell, mesh_n).unwrap();
    let cfg = krks_cfg(cell, functional, grid);
    match grid {
        KXcGrid::Atom(_) => solve_krks(cell, &prep, None, &mesh, &cfg),
        KXcGrid::Uniform(n) => {
            let g = PeriodicGrid::uniform(cell, *n).unwrap();
            solve_krks_on_grid(cell, &prep, None, &mesh, &g, &cfg)
        }
    }
    .unwrap_or_else(|e| panic!("solve_krks {functional}: {e}"))
}

struct Fx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    eri: KDenseAftEri,
    res: KRksResult,
    grid: KXcGrid,
    functional: String,
}

fn fixture_cell(cell: Cell, mesh_n: [usize; 3], functional: &str, grid: KXcGrid) -> Fx {
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let res = scf_cell(&cell, mesh_n, functional, &grid);
    let cfg = krks_cfg(&cell, functional, &grid);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &cfg.krhf.hcore).unwrap();
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::None,
        &Default::default(),
    )
    .unwrap();
    Fx {
        cell,
        prep,
        mesh,
        hk,
        eri,
        res,
        grid,
        functional: functional.to_string(),
    }
}

fn fixture(pos: &[[f64; 3]], mesh_n: [usize; 3], functional: &str, grid: KXcGrid) -> Fx {
    fixture_cell(cell_of(pos), mesh_n, functional, grid)
}

fn gcfg(grid: &KXcGrid, mutation: Option<KXcMutation>) -> KRksGradConfig {
    KRksGradConfig {
        grad: KGradConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        grid: grid.clone(),
        mutation,
        ..Default::default()
    }
}

fn force_with(
    fx: &Fx,
    functional: &str,
    scf: &KScfResult,
    cfg: &KRksGradConfig,
) -> Result<KRksGradient, ferric_core::FerricError> {
    kpoint_rks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KGradJk::Dense(&fx.eri),
        scf,
        functional,
        cfg,
    )
}

fn force(fx: &Fx, mutation: Option<KXcMutation>) -> KRksGradient {
    force_with(fx, &fx.functional, &fx.res.scf, &gcfg(&fx.grid, mutation)).unwrap()
}

fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

fn diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn scf_energy(pos: &[[f64; 3]], mesh_n: [usize; 3], f: &str, grid: &KXcGrid) -> f64 {
    scf_cell(&cell_of(pos), mesh_n, f, grid).scf.energy
}

/// `E_xc[D]` at fixed AO density matrices `d` on the atom grid of `cell`.
fn exc_fixed_d_on(
    cell: &Cell,
    bs: &BasisSet,
    mesh_n: [usize; 3],
    f: &str,
    cfg: &PeriodicGridConfig,
    d: &[Array2<Complex64>],
) -> f64 {
    let mesh = KPointMesh::gamma_centred(cell, mesh_n).unwrap();
    let grid = PeriodicGrid::build(cell, cfg).unwrap();
    let mut xc = KPeriodicXc::new(cell, bs, f, &grid, &mesh, &PeriodicXcConfig::default()).unwrap();
    xc.build(d).unwrap().0
}

fn exc_fixed_d(
    pos: &[[f64; 3]],
    mesh_n: [usize; 3],
    f: &str,
    cfg: &PeriodicGridConfig,
    d: &[Array2<Complex64>],
) -> f64 {
    exc_fixed_d_on(&cell_of(pos), &pyscf_sto3g_h(), mesh_n, f, cfg, d)
}

/// XC part (X1 + X3 + X4) of the `meta` force on the density of a converged
/// PBE SCF, against the central FD of `E_xc^meta[D]` at that fixed `D`. The
/// XC parts do not depend on the Fock guard, which `GuardSkippedOnly` skips
/// (the PBE Fock is not the meta functional's), so no meta-GGA SCF is needed.
/// Returns `(worst |analytic - FD|, largest |XC force|)` over `comps`.
#[allow(clippy::too_many_arguments)]
fn pbe_density_meta_xc_fd(
    cell_of: &dyn Fn(&[[f64; 3]]) -> Cell,
    bs: &BasisSet,
    pos: &[[f64; 3]],
    mesh_n: [usize; 3],
    meta: &str,
    grid_cfg: &PeriodicGridConfig,
    comps: &[(usize, usize)],
    h: f64,
) -> (f64, f64) {
    let cell = cell_of(pos);
    let prep = prep_for(&cell, bs);
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let grid = KXcGrid::Atom(grid_cfg.clone());
    let cfg = krks_cfg(&cell, "PBE", &grid);
    let res = solve_krks(&cell, &prep, None, &mesh, &cfg).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &cfg.krhf.hcore).unwrap();
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::None,
        &Default::default(),
    )
    .unwrap();
    let g = kpoint_rks_gradient(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &hk,
        KGradJk::Dense(&eri),
        &res.scf,
        meta,
        &gcfg(&grid, Some(KXcMutation::GuardSkippedOnly)),
    )
    .unwrap();
    let xc = &(&g.xc_ao + &g.xc_point) + &g.xc_weight;
    let d0 = res.scf.densities.clone();
    let mut worst = 0.0_f64;
    for &(a, x) in comps {
        let ep = exc_fixed_d_on(
            &cell_of(&moved(pos, a, x, h)),
            bs,
            mesh_n,
            meta,
            grid_cfg,
            &d0,
        );
        let em = exc_fixed_d_on(
            &cell_of(&moved(pos, a, x, -h)),
            bs,
            mesh_n,
            meta,
            grid_cfg,
            &d0,
        );
        worst = worst.max((xc[(a, x)] - (ep - em) / (2.0 * h)).abs());
    }
    (worst, xc.iter().fold(0.0_f64, |m, v| m.max(v.abs())))
}

// ------------------------------------------------------------------ anchors

/// Null-tau limit. Artifact hypothesis: a tau path that perturbed the
/// non-tau terms (a mis-sized plane, a stray accumulation) would shift the PBE
/// force; physics predicts exactly 0 because `v_tau = 0`.
#[test]
fn null_tau_path_reproduces_the_gga_force_exactly() {
    let fx = fixture(&H2, [1, 1, 3], "PBE", atom_grid(20, 50));
    let base = force(&fx, None);
    let hooked = force(&fx, Some(KXcMutation::ForceTauPath));
    let d = diff(&hooked.grad, &base.grad);
    eprintln!("PBE through the tau path: max diff {d:.2e}");
    assert_eq!(d, 0.0);
}

/// XC force (X1 + X3 + X4) against FD of `E_xc[D_fixed]`.
#[test]
fn fixed_density_xc_force_matches_fd_of_exc() {
    let grid_cfg = PeriodicGridConfig::with_size(20, 50);
    for f in ["SCAN", "R2SCAN", "PBE"] {
        let fx = fixture(&H2, [1, 1, 3], f, KXcGrid::Atom(grid_cfg.clone()));
        let g = force(&fx, None);
        let xc = &(&g.xc_ao + &g.xc_point) + &g.xc_weight;
        let d0 = fx.res.scf.densities.clone();
        let h = 1e-4;
        let mut worst = 0.0_f64;
        for (a, x) in [(0, 0), (0, 2), (1, 1), (1, 2)] {
            let fd = (exc_fixed_d(&moved(&H2, a, x, h), [1, 1, 3], f, &grid_cfg, &d0)
                - exc_fixed_d(&moved(&H2, a, x, -h), [1, 1, 3], f, &grid_cfg, &d0))
                / (2.0 * h);
            worst = worst.max((xc[(a, x)] - fd).abs());
        }
        eprintln!(
            "{f}: XC force vs FD of fixed-D E_xc worst {worst:.2e} (|xc| max {:.2e})",
            xc.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
        );
        assert!(worst < FD_BAR, "{f}: {worst:e}");
    }
}

/// Total force against FD of the converged SCF energy.
#[test]
fn meta_gga_force_matches_fd_of_the_scf_energy() {
    let grid = atom_grid(20, 50);
    for f in ["SCAN", "R2SCAN"] {
        let fx = fixture(&H2, [1, 1, 3], f, grid.clone());
        let g = force(&fx, None);
        let h = 1e-4;
        let mut worst = 0.0_f64;
        for (a, x) in [(0, 0), (0, 2), (1, 1)] {
            let fd = (scf_energy(&moved(&H2, a, x, h), [1, 1, 3], f, &grid)
                - scf_energy(&moved(&H2, a, x, -h), [1, 1, 3], f, &grid))
                / (2.0 * h);
            worst = worst.max((g.grad[(a, x)] - fd).abs());
        }
        eprintln!("{f}: analytic vs FD of the SCF energy worst {worst:.2e}");
        assert!(worst < FD_BAR, "{f}: {worst:e}");
    }
}

fn supercell(pos: &[[f64; 3]], a: f64, n: usize) -> Cell {
    let mut atoms = Vec::new();
    for m in 0..n {
        for p in pos {
            atoms.push([p[0], p[1], p[2] + m as f64 * a]);
        }
    }
    let mut lat = cubic(a);
    lat[2][2] = a * n as f64;
    Cell::new(hydrogens(&atoms), lat).unwrap()
}

/// k = 1x1x3 versus the 3-cell supercell at 1x1x1, same code, same explicit
/// neighbour cutoff, per part, on one copy of each atom.
#[test]
#[ignore = "two SCFs (1x1x3 and the 3-cell supercell) x 2 functionals; measured 73 s alone in the test profile on a shared box"]
fn supercell_identity_per_part() {
    let mut grid = PeriodicGridConfig::with_size(20, 50);
    grid.neighbour_cutoff = Some(10.0);
    let kgrid = KXcGrid::Atom(grid);
    for f in ["SCAN", "R2SCAN"] {
        let k = force(&fixture(&H2, [1, 1, 3], f, kgrid.clone()), None);
        let sc = force(
            &fixture_cell(supercell(&H2, 4.0, 3), [1, 1, 1], f, kgrid.clone()),
            None,
        );
        let first = |a: &Array2<f64>| a.slice(ndarray::s![0..2, ..]).to_owned();
        for (n, kk, ss) in [
            ("xc_ao", &k.xc_ao, &sc.xc_ao),
            ("xc_point", &k.xc_point, &sc.xc_point),
            ("xc_weight", &k.xc_weight, &sc.xc_weight),
            ("total", &k.grad, &sc.grad),
        ] {
            let d = diff(kk, &first(ss));
            eprintln!("{f} 1x1x3 vs supercell: {n:<9} {d:.2e}");
            assert!(d < 1e-9, "{f} {n}: {d:e}");
        }
    }
}

/// Uniform grid: X3 and X4 absent exactly; the AO term (tau included) alone
/// matches FD.
#[test]
fn uniform_grid_force_matches_fd() {
    let grid = KXcGrid::Uniform([24, 24, 24]);
    let f = "R2SCAN";
    let fx = fixture(&H2, [1, 1, 3], f, grid.clone());
    let g = force(&fx, None);
    let grid_terms = g
        .xc_point
        .iter()
        .chain(g.xc_weight.iter())
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    assert_eq!(grid_terms, 0.0);
    let h = 1e-4;
    let mut worst = 0.0_f64;
    for (a, x) in [(0, 0), (0, 2), (1, 1)] {
        let fd = (scf_energy(&moved(&H2, a, x, h), [1, 1, 3], f, &grid)
            - scf_energy(&moved(&H2, a, x, -h), [1, 1, 3], f, &grid))
            / (2.0 * h);
        worst = worst.max((g.grad[(a, x)] - fd).abs());
    }
    eprintln!("uniform 24^3 r2SCAN: analytic vs FD worst {worst:.2e}");
    assert!(worst < FD_BAR, "{worst:e}");
}

// ----------------------------------------------------- bitwise and budgets

#[test]
fn chunk_cap_is_bit_identical_and_the_chunk_gate_trips() {
    let fx = fixture(&H2, [1, 1, 3], "R2SCAN", atom_grid(20, 50));
    let full = force(&fx, None);
    let mut cfg = gcfg(&fx.grid, None);
    cfg.max_live_chunks = Some(1);
    let one = force_with(&fx, "R2SCAN", &fx.res.scf, &cfg).unwrap();
    assert_eq!(one.live_chunks, 1);
    assert_eq!(one.grad, full.grad);
    let mut saw = false;
    let mut b = 300_000usize;
    while b < 500_000 {
        let mut c = gcfg(&fx.grid, None);
        c.grad.budget_bytes = Some(b);
        if let Err(e) = force_with(&fx, "R2SCAN", &fx.res.scf, &c) {
            saw |= e.to_string().contains("k-point XC force AO/Hessian chunk");
        }
        b += b / 8;
    }
    assert!(saw, "some budget must fail on the XC chunk gate");
}

/// The XC chunk gate's byte charge, read from the typed error of a budget too
/// small for one chunk.
fn chunk_gate_bytes(fx: &Fx, functional: &str, mutation: Option<KXcMutation>) -> usize {
    let mut b = 300_000usize;
    while b < 600_000 {
        let mut c = gcfg(&fx.grid, mutation);
        c.grad.budget_bytes = Some(b);
        if let Err(e) = force_with(fx, functional, &fx.res.scf, &c) {
            let m = e.to_string();
            if let Some(i) = m.find("k-point XC force AO/Hessian chunk (") {
                let rest = &m[i..];
                let j = rest.find(") (").expect("charge in the message") + 3;
                let digits: String = rest[j..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                return digits.parse().expect("bytes");
            }
        }
        b += b / 8;
    }
    panic!("no budget in range tripped the XC chunk gate");
}

/// Memory ledger: a production meta-GGA charges exactly what a GGA charges
/// (the Hessian planes are shared, the tau term adds no plane), while the
/// `TauNoPoint` mutant, the only owner of a separate tau plane, charges that
/// plane (`chunk_points * natoms * 3 * 8` bytes) on top. A ledger term added
/// for production meta-GGA breaks the first assertion.
#[test]
fn meta_gga_chunk_charge_is_the_gga_charge_and_the_tau_plane_is_mutant_only() {
    let fx = fixture(&H2, [1, 1, 3], "R2SCAN", atom_grid(20, 50));
    let ok = force(&fx, None);
    assert_eq!(ok.chunk_points, 128);
    let pbe = chunk_gate_bytes(&fx, "PBE", None);
    let meta = chunk_gate_bytes(&fx, "R2SCAN", None);
    let mutant = chunk_gate_bytes(&fx, "R2SCAN", Some(KXcMutation::TauNoPoint));
    let natoms = 2;
    eprintln!("chunk charge: PBE {pbe}, R2SCAN {meta}, TauNoPoint {mutant}");
    assert_eq!(
        meta, pbe,
        "meta-GGA must charge the GGA plane budget and nothing more"
    );
    assert_eq!(mutant - meta, 128 * natoms * 3 * 8);
}

// ------------------------------------------------- refusals and Fock guard

#[test]
fn refusals_and_the_fock_guard() {
    let fx = fixture(&H2, [1, 1, 3], "SCAN", atom_grid(20, 50));
    let cfg = gcfg(&fx.grid, None);
    for (name, key) in [("HSE06", "range-separated"), ("wB97X-V", "VV10")] {
        let e = force_with(&fx, name, &fx.res.scf, &cfg)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        eprintln!("{name}: {e}");
        assert!(e.contains(key), "{name}: {e}");
    }
    let ok = force_with(&fx, "SCAN", &fx.res.scf, &cfg).unwrap();
    eprintln!("SCAN guard mismatch {:.1e}", ok.hf.fock_mismatch);
    assert!(ok.hf.fock_mismatch < 1e-12);
    for other in ["R2SCAN", "PBE"] {
        let e = force_with(&fx, other, &fx.res.scf, &cfg)
            .err()
            .unwrap_or_else(|| panic!("{other} on a SCAN result accepted"))
            .to_string();
        assert!(e.contains("differs"), "{other}: {e}");
    }
    let tamper = gcfg(&fx.grid, Some(KXcMutation::TauNoVxc));
    let e = force_with(&fx, "SCAN", &fx.res.scf, &tamper)
        .expect_err("guard must refuse a Fock without V^tau")
        .to_string();
    eprintln!("no V^tau: {e}");
    assert!(e.contains("differs"), "{e}");
}

// ------------------------------------------------------------------ mutants

#[test]
fn mutants_are_caught() {
    let fx = fixture(&H2, [1, 1, 3], "R2SCAN", atom_grid(20, 50));
    let base = force(&fx, None);
    for m in [
        KXcMutation::TauNoAo,
        KXcMutation::TauDoubled,
        KXcMutation::TauNoPoint,
        KXcMutation::TauNoVxcUnguarded,
        KXcMutation::NoGga,
        KXcMutation::NoHessian,
        KXcMutation::NoPhase,
        KXcMutation::ConjPhase,
        KXcMutation::NoPointMotion,
        KXcMutation::NoWeightDeriv,
        KXcMutation::NoInvNk,
        KXcMutation::WrongFactor,
    ] {
        let d = diff(&force(&fx, Some(m)).grad, &base.grad);
        eprintln!("R2SCAN 1x1x3 {m:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{m:?} survives: {d:e}");
    }
    // Identity: on the TRIM-only mesh 1x1x2 D(k) is real and the conjugate
    // phase changes nothing.
    let trim = fixture(&H2, [1, 1, 2], "R2SCAN", atom_grid(20, 50));
    let tb = force(&trim, None);
    let d = diff(&force(&trim, Some(KXcMutation::ConjPhase)).grad, &tb.grad);
    eprintln!("R2SCAN 1x1x2 ConjPhase (TRIM identity): {d:.2e}");
    assert!(d < 1e-12);
}

// ------------------------------------------------- TPSS, p/d shells, RS-GDF

/// TPSS exists only as the raw single-component names (no combined
/// functional): each is checked as its own `E_xc` (exchange-only,
/// correlation-only), which exercises TPSS's tau kernels without a TPSS SCF.
#[test]
fn tpss_components_xc_force_matches_fd_of_exc() {
    let bs = pyscf_sto3g_h();
    let g = PeriodicGridConfig::with_size(20, 50);
    for f in ["MGGA_X_TPSS", "MGGA_C_TPSS"] {
        let (worst, scale) = pbe_density_meta_xc_fd(
            &cell_of,
            &bs,
            &H2,
            [1, 1, 3],
            f,
            &g,
            &[(0, 0), (0, 2), (1, 1), (1, 2)],
            1e-4,
        );
        eprintln!("{f}: XC force vs fixed-D FD worst {worst:.2e} (|xc| max {scale:.2e})");
        assert!(worst < FD_BAR, "{f}: {worst:e}");
    }
}

fn spd_basis_h() -> BasisSet {
    let mut bs = sp_basis_h();
    bs.shells.get_mut(&1).unwrap().push(Shell {
        l: 2,
        pure: false,
        exponents: vec![0.9],
        coefficients: vec![1.0],
    });
    bs
}

/// s+p+d shells on a triclinic cell: the p and d AO Hessians under tau,
/// SCAN and r2SCAN, fixed-D.
#[test]
#[ignore = "triclinic s+p+d PBE SCF + fixed-D XC builds; measured at most 271 s wall (loaded shared box, run with two other tests)"]
fn spd_triclinic_xc_force_matches_fd_of_exc() {
    const TRI2: [[f64; 3]; 2] = [[0.13, 0.25, 0.31], [0.02, 0.27, 1.66]];
    let tri = |pos: &[[f64; 3]]| Cell::new(hydrogens(pos), TRI_A).expect("triclinic cell");
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(14.0);
    for f in ["SCAN", "R2SCAN"] {
        let (worst, scale) = pbe_density_meta_xc_fd(
            &tri,
            &spd_basis_h(),
            &TRI2,
            [1, 1, 2],
            f,
            &g,
            &[(0, 0), (0, 2), (1, 1)],
            1e-4,
        );
        eprintln!(
            "spd triclinic {f}: XC force vs fixed-D FD worst {worst:.2e} (|xc| max {scale:.2e})"
        );
        assert!(worst < FD_BAR, "{f}: {worst:e}");
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

/// SCAN energy (and the force) on the unsplit RS-GDF J route.
fn gdf_scan_case(pos: &[[f64; 3]], want_force: bool) -> (f64, Option<KRksGradient>) {
    let cell = cell_of(pos);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let aux = prep_for(&cell, &et_sp_aux());
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let grid = atom_grid(20, 50);
    let mut cfg = krks_cfg(&cell, "SCAN", &grid);
    cfg.krhf.jk = KJkKind::RsGdf;
    cfg.krhf.rsgdf = kgdf_cfg();
    let res = solve_krks(&cell, &prep, Some(&aux), &mesh, &cfg).unwrap();
    if !want_force {
        return (res.scf.energy, None);
    }
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &cfg.krhf.hcore).unwrap();
    let kc = kgdf_cfg();
    let gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kc)
        .unwrap()
        .with_exxdiv(ExxDiv::None);
    let g = kpoint_rks_gradient(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &hk,
        KGradJk::RsGdf(KRsGdfGradSource {
            gdf: &gdf,
            cfg: &kc,
            aux: &aux,
        }),
        &res.scf,
        "SCAN",
        &gcfg(&grid, None),
    )
    .unwrap();
    (res.scf.energy, Some(g))
}

#[test]
#[ignore = "SCAN on RS-GDF, 3 SCFs; measured within the same 271 s wall run, under 271 s"]
fn rsgdf_scan_force_component_matches_fd() {
    let g = gdf_scan_case(&H2, true).1.unwrap();
    let h = 1e-4;
    let (a, x) = (1, 1);
    let fd = (gdf_scan_case(&moved(&H2, a, x, h), false).0
        - gdf_scan_case(&moved(&H2, a, x, -h), false).0)
        / (2.0 * h);
    let d = (g.grad[(a, x)] - fd).abs();
    eprintln!(
        "SCAN RS-GDF: analytic {:.10} FD {fd:.10} diff {d:.2e}",
        g.grad[(a, x)]
    );
    assert!(d < FD_BAR, "{d:e}");
}
