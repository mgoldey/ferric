//! Range-separated hybrids in the k-point KS-DFT (`ferric_pbc::rsh`, dense
//! AFT J/K). There is no Gamma-point periodic RSH to mirror (`gamma_rks`
//! refuses CAM functionals), so the anchors are:
//!
//! * `madelung_lr_matches_pinned_pyscf` — the mixed-kernel G = 0 constants
//!   (`tools.pbc.madelung(cell, kpts, omega=±ω)`), supercell lattices.
//! * `rsh_with_equal_fractions_is_the_global_hybrid` — `c_sr = c_lr = c`
//!   must reproduce `c ×` the plain k-point kernel and `c · v_M` (the
//!   range-separation algebra collapses exactly).
//! * `rsh_k_mesh_is_the_supercell` — E/cell of an N-point mesh against the
//!   1-point mesh of the explicit N-cell supercell / N (same RSH code at the
//!   supercell's Gamma; the mesh/q structure and supercell Madelung constant
//!   are what differ), for HSE06 (`c_lr = 0`) and CAM-B3LYP (mixed).
//! * `rsh_krks_matches_pinned_pyscf` — PySCF 2.13 `pbc.dft.KRKS`
//!   (AFTDF, UniformGrids 61³, exxdiv ewald) for HSE06, CAM-B3LYP, wB97X.
//! * `rsh_mutations_break_the_anchors` — the wrong (primitive-cell)
//!   Madelung constant and a dropped range-separation factor each break the
//!   supercell anchor.

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::{PeriodicGrid, PeriodicGridConfig};
use ferric_pbc::hcore::kpoint::periodic_hcore_kpts;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri, KMutation};
use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig, KRksResult};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsh::{madelung_lr, RshParams};

const OMEGA: f64 = 0.8;

fn krks_cfg(cell: &Cell, functional: &str) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, ExxDiv::Ewald, functional);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-12;
    cfg.krhf.scf.grad_conv = 1e-9;
    cfg
}

fn run(
    cell: &Cell,
    bs: &BasisSet,
    mesh: &KPointMesh,
    grid: &PeriodicGrid,
    cfg: &KRksConfig,
) -> KRksResult {
    let prep = prep_for(cell, bs);
    solve_krks_on_grid(cell, &prep, None, mesh, grid, cfg)
        .unwrap_or_else(|e| panic!("solve_krks {}: {e}", cfg.functional))
}

fn sc_cell(cell: &Cell, n: [usize; 3]) -> (Cell, Vec<[f64; 3]>) {
    let a = *cell.lattice();
    let mut atoms = Vec::new();
    let mut shifts = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                let t: Vec<f64> = (0..3)
                    .map(|d| m0 as f64 * a[0][d] + m1 as f64 * a[1][d] + m2 as f64 * a[2][d])
                    .collect();
                shifts.push([t[0], t[1], t[2]]);
                for p in cell.positions() {
                    atoms.push([p[0] + t[0], p[1] + t[1], p[2] + t[2]]);
                }
            }
        }
    }
    let mut lat = a;
    for (i, row) in lat.iter_mut().enumerate() {
        for v in row.iter_mut() {
            *v *= n[i] as f64;
        }
    }
    (
        Cell::new(hydrogens(&atoms), lat).expect("supercell"),
        shifts,
    )
}

// PySCF 2.13 tools.pbc.madelung(cell, kpts, omega), H2 a = 4 Bohr cubic,
// gamma-centred meshes (scratchpad `ref/mad.py`, 2026-10-09):
// (mesh, full, [(omega, LR (omega > 0), SR (omega < 0))]).
#[allow(clippy::type_complexity)]
const PYSCF_MADELUNG: [([usize; 3], f64, [(f64, f64, f64); 2]); 3] = [
    (
        [1, 1, 1],
        0.709324369870,
        [
            (0.11, 0.124121708381, 0.585202661490),
            (0.4, 0.441029230735, 0.268295139136),
        ],
    ),
    (
        [1, 1, 2],
        0.451460452613,
        [
            (0.11, 0.124120779764, 0.327339672849),
            (0.4, 0.322691045503, 0.128769407110),
        ],
    ),
    (
        [2, 2, 2],
        0.354662184935,
        [
            (0.11, 0.124121011916, 0.230541173019),
            (0.4, 0.316317184724, 0.038345000211),
        ],
    ),
];

#[test]
fn madelung_lr_matches_pinned_pyscf() {
    use ferric_pbc::ewald::madelung_constant;
    let cell = h2_cell(4.0);
    for (n, full, rows) in PYSCF_MADELUNG {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        let sc = Cell::new(cell.mol().clone(), mesh.supercell_lattice()).unwrap();
        let vf = madelung_constant(&sc).unwrap();
        assert!((vf - full).abs() < 1e-8, "{n:?} full {vf} vs {full}");
        for (w, lr, sr) in rows {
            let vlr = madelung_lr(&sc, w).unwrap();
            // SR = full - LR; the mixed constant for c_sr = 1, c_lr = 0.
            let vsr = RshParams {
                omega: w,
                c_sr: 1.0,
                c_lr: 0.0,
            }
            .madelung(&sc)
            .unwrap();
            eprintln!("{n:?} w={w}: LR {vlr:.12} (pyscf {lr:.12}) SR {vsr:.12} (pyscf {sr:.12})");
            assert!((vlr - lr).abs() < 1e-8, "{n:?} w={w} LR");
            assert!((vsr - sr).abs() < 1e-8, "{n:?} w={w} SR");
        }
    }
}

#[test]
fn rsh_with_equal_fractions_is_the_global_hybrid() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let plain = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::Ewald,
        &KDenseAftConfig::default(),
    )
    .unwrap();
    let c = 0.37;
    let rsh = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::Ewald,
        &KDenseAftConfig {
            rsh: Some(RshParams {
                omega: 0.23,
                c_sr: c,
                c_lr: c,
            }),
            ..Default::default()
        },
    )
    .unwrap();
    let mut worst = 0.0_f64;
    for k in 0..mesh.nk() {
        for kp in 0..mesh.nk() {
            let d = (&rsh.kker(k, kp).mapv(|z| z / c) - plain.kker(k, kp))
                .iter()
                .fold(0.0_f64, |m, z| m.max(z.norm()));
            worst = worst.max(d);
            // J is untouched.
            let dj = (rsh.jker(k, kp) - plain.jker(k, kp))
                .iter()
                .fold(0.0_f64, |m, z| m.max(z.norm()));
            assert!(dj == 0.0, "J kernel changed by rsh: {dj}");
        }
    }
    let dm = (rsh.madelung_ewald() - c * plain.madelung_ewald()).abs();
    eprintln!("equal-fraction RSH: max|K/c - K| {worst:.2e}, |v_M - c v_M| {dm:.2e}");
    assert!(worst < 1e-12, "{worst}");
    assert!(dm < 1e-10, "{dm}");
}

fn supercell_anchor(functional: &str, n: [usize; 3], cell_mutation: Option<KMutation>) -> f64 {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let nk = (n[0] * n[1] * n[2]) as f64;
    let cell_grid = PeriodicGrid::build(&cell, &PeriodicGridConfig::with_size(40, 110)).unwrap();
    let (sc, shifts) = sc_cell(&cell, n);
    let sc_grid = cell_grid.replicated(&shifts);
    let sc_mesh = KPointMesh::gamma_centred(&sc, [1, 1, 1]).unwrap();
    let e_sc = run(&sc, &bs, &sc_mesh, &sc_grid, &krks_cfg(&sc, functional))
        .scf
        .energy
        / nk;
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    let mut cfg = krks_cfg(&cell, functional);
    cfg.krhf.dense.mutation = cell_mutation;
    let r = run(&cell, &bs, &mesh, &cell_grid, &cfg);
    eprintln!(
        "{functional} {n:?}: E_k {:.12} E_super/N {e_sc:.12} dE {:.2e}",
        r.scf.energy,
        r.scf.energy - e_sc
    );
    r.scf.energy - e_sc
}

#[test]
fn rsh_k_mesh_is_the_supercell() {
    for functional in ["HSE06", "HYB_GGA_XC_CAM_B3LYP"] {
        let d = supercell_anchor(functional, [1, 1, 2], None);
        assert!(d.abs() < 1e-8, "{functional}: {d}");
    }
}

#[test]
fn rsh_mutations_break_the_anchors() {
    let d = supercell_anchor("HSE06", [1, 1, 2], Some(KMutation::PrimitiveMadelung));
    assert!(
        d.abs() > 1e-4,
        "PrimitiveMadelung left HSE06 anchor green: {d}"
    );
    let d = supercell_anchor("HSE06", [1, 1, 2], Some(KMutation::KernelAtG));
    assert!(d.abs() > 1e-4, "KernelAtG left HSE06 anchor green: {d}");
}

// PySCF 2.13 pbc.dft.KRKS, gamma-centred mesh, AFTDF mesh 61^3, exxdiv
// 'ewald', UniformGrids(61^3), conv_tol 1e-12, STO-3G H2 a = 4
// (scratchpad `ref/krsh2.py`, 2026-10-09; 2x2x2 meshes omitted: PySCF's
// attenuated AFTDF K at mesh 61 did not finish within 10 minutes).
const PYSCF: [(&str, &str, [usize; 3], f64); 6] = [
    ("HSE06", "HSE06", [1, 1, 1], -1.575428343697),
    ("HSE06", "HSE06", [1, 1, 2], -1.347311737653),
    (
        "CAMB3LYP",
        "HYB_GGA_XC_CAM_B3LYP",
        [1, 1, 1],
        -1.591829863303,
    ),
    (
        "CAMB3LYP",
        "HYB_GGA_XC_CAM_B3LYP",
        [1, 1, 2],
        -1.354469613012,
    ),
    ("WB97X", "HYB_GGA_XC_WB97X", [1, 1, 1], -1.605512318088),
    ("WB97X", "HYB_GGA_XC_WB97X", [1, 1, 2], -1.363565753524),
];

#[test]
fn rsh_krks_matches_pinned_pyscf() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::uniform(&cell, [61; 3]).unwrap();
    for (label, name, n, e_ref) in PYSCF {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        let r = run(&cell, &bs, &mesh, &grid, &krks_cfg(&cell, name));
        eprintln!(
            "{label} {n:?}: E {:.12} pyscf {e_ref:.12} dE {:.2e}",
            r.scf.energy,
            r.scf.energy - e_ref
        );
        assert!((r.scf.energy - e_ref).abs() < 1e-9, "{label} {n:?}");
    }
}

#[test]
fn rsh_with_rsgdf_is_refused_not_silently_wrong() {
    use ferric_pbc::kscf::KJkKind;
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let grid = PeriodicGrid::uniform(&cell, [24; 3]).unwrap();
    let mut cfg = krks_cfg(&cell, "HSE06");
    cfg.krhf.jk = KJkKind::RsGdf;
    let err = solve_krks_on_grid(&cell, &prep, Some(&prep), &mesh, &grid, &cfg)
        .expect_err("RSH + RS-GDF must be refused");
    assert!(err.to_string().contains("range-separated"), "{err}");
}

/// `KPeriodicXc` accepts range-separated hybrids for the closed-shell solver,
/// but the spin-resolved k-point solvers have no range-separated exchange:
/// they must refuse HSE06 on every entry point, not run it as a global hybrid.
#[test]
fn open_shell_solvers_refuse_rsh() {
    use ferric_pbc::kroks::{solve_kroks_on_grid, KRoksConfig};
    use ferric_pbc::kuks::{solve_kuks_on_grid, KUksConfig};
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let grid = PeriodicGrid::uniform(&cell, [24; 3]).unwrap();
    let roks = KRoksConfig::new(&cell, ExxDiv::Ewald, "HSE06");
    assert!(
        solve_kroks_on_grid(&cell, &prep, None, &mesh, &grid, &roks).is_err(),
        "solve_kroks_on_grid accepted HSE06"
    );
    let uks = KUksConfig::new(&cell, ExxDiv::Ewald, "HSE06");
    assert!(
        solve_kuks_on_grid(&cell, &prep, None, &mesh, &grid, &uks).is_err(),
        "solve_kuks_on_grid accepted HSE06"
    );
}
