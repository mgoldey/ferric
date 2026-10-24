// LANL2DZ exponents such as 0.318 are basis data, not approximations of 1/pi.
#![allow(clippy::approx_constant)]
//! k-point open-shell KS-DFT (UKS, ROKS, ROHF) on a SHIFTED mesh of a
//! TRICLINIC cell, with dense AFT and RS-GDF, plus k-RKS on a cell with
//! CORE ELECTRONS (HI, I LANL2DZ ECP) — issue #356 validation breadth.
//!
//! Pins: `testdata/reference/pbc_kks_open_ecp_sto3g.json`
//! (generator `scripts/gen_pyscf_kks_open_ecp_refs.py`). PySCF and ferric use
//! the SAME uniform grid (`UniformGrids` N^3 vs `PeriodicGrid::uniform`), so
//! a small N is a valid pin: the grid need not converge the XC integral, it
//! only has to be the same discretisation (the open-shell 61^3 of `pbc_kuks.rs`
//! was a choice, not a necessity; here N = 31).
//!
//! # Physics / artifact hypothesis (written before measuring)
//!
//! Physics: dense-AFT energies reproduce PySCF to SCF/grid noise (~1e-8 Ha,
//! as for the closed-shell shifted pins). The RS-GDF energy differs from the
//! dense one by the aux-fit error (measured 7e-6 Ha UKS/ROKS, bar `FIT_BAR`).
//! Artifact: a dropped shift, a q-phase sign error or a wrong G = 0 treatment
//! moves the open-shell RS-GDF energy by >= `MUTANT_MIN`; `NoTimeReversal`
//! (build every q class explicitly) is NOT a defect and must be a no-op.
//! Core electrons: a bare-Z (no `apply_ecp`) cell is refused; a PySCF pin
//! at PySCF's converged ECP precision (cell.precision 1e-12) differs from
//! ferric by ~1.1e-6 Ha (reason not separated), so the ECP bar is `ECP_BAR`.
//!
//! LIMIT: the shifted-mesh Madelung/ewald constant is validated only against
//! PySCF's convention (no independent twisted-boundary anchor exists).

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet, Shell};
use ferric_core::ecp::{EcpDef, EcpShell, EcpTerm};
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGrid;
use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::{solve_krohf, solve_kroks_on_grid, KRoksConfig};
use ferric_pbc::kscf::KJkKind;
use ferric_pbc::kuks::{solve_kuks_on_grid, KUksConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdfConfig, KRsGdfMutation};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::uhf::EwaldStart;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const AUX: &str = "cc-pvdz-ri";
const REF_JSON: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../testdata/reference/pbc_kks_open_ecp_sto3g.json"
);
const HI_A: [[f64; 3]; 3] = [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]];
const HI_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.4], [0.3, 0.2, 3.44]];

// ------------------------------------------------------------ fixtures

/// Contraction over unit-normalised primitives scaled to unit self-overlap
/// (what PySCF's `make_bas_env` and ferric's BSE loader both do).
fn shell(l: i32, exps: &[f64], coefs: &[f64]) -> Shell {
    let lf = l as f64;
    let mut s = 0.0;
    for (a, ca) in exps.iter().zip(coefs) {
        for (b, cb) in exps.iter().zip(coefs) {
            s += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(lf + 1.5);
        }
    }
    Shell {
        l,
        pure: false,
        exponents: exps.to_vec(),
        coefficients: coefs.iter().map(|c| c / s.sqrt()).collect(),
    }
}

/// LANL2DZ ECP for I (PySCF / libecpint `lanl2dz.ecp`, identical digits):
/// local channel l = 3 (ul), projectors s, p, d; (n, ζ, d) with the
/// `r^{n−2}` convention both libecpint and PySCF use.
fn lanl2dz_i_ecp() -> EcpDef {
    let ch = |l: i32, t: &[(i32, f64, f64)]| EcpShell {
        angular_momentum: l,
        terms: t
            .iter()
            .map(|&(n, z, d)| EcpTerm {
                coef: d,
                r_exp: n,
                gexp: z,
            })
            .collect(),
    };
    EcpDef {
        n_core: 46,
        shells: vec![
            ch(
                3,
                &[
                    (0, 1.0715702, -0.0747621),
                    (1, 44.1936028, -30.0811224),
                    (2, 12.9367609, -75.3722721),
                    (2, 3.1956412, -22.0563758),
                    (2, 0.8589806, -1.6979585),
                ],
            ),
            ch(
                0,
                &[
                    (0, 127.9202670, 2.9380036),
                    (1, 78.6211465, 41.2471267),
                    (2, 36.5146237, 287.8680095),
                    (2, 9.9065681, 114.3758506),
                    (2, 1.9420086, 37.6547714),
                ],
            ),
            ch(
                1,
                &[
                    (0, 13.0035304, 2.2222630),
                    (1, 76.0331404, 39.4090831),
                    (2, 24.1961684, 177.4075002),
                    (2, 6.4053433, 77.9889462),
                    (2, 1.5851786, 25.7547641),
                ],
            ),
            ch(
                2,
                &[
                    (0, 40.4278108, 7.0524360),
                    (1, 28.9084375, 33.3041635),
                    (2, 15.6268936, 186.9453875),
                    (2, 4.1442856, 71.9688361),
                    (2, 0.9377235, 9.3630657),
                ],
            ),
        ],
    }
}

/// H STO-3G (PySCF digits) + I LANL2DZ (`full`) or the compact s+p I basis,
/// with the LANL2DZ ECP on I.
fn hi_basis(full: bool) -> BasisSet {
    let mut shells = HashMap::new();
    shells.insert(
        1,
        vec![shell(
            0,
            &[3.42525091, 0.62391373, 0.1688554],
            &[0.15432897, 0.53532814, 0.44463454],
        )],
    );
    let i_shells = if full {
        vec![
            shell(0, &[0.7242, 0.4653], &[-2.9731048, 3.4827643]),
            shell(0, &[0.1336], &[1.0]),
            shell(1, &[1.29, 0.318], &[-0.2092377, 1.1035347]),
            shell(1, &[0.1053], &[1.0]),
        ]
    } else {
        vec![shell(0, &[0.4653], &[1.0]), shell(1, &[0.318], &[1.0])]
    };
    shells.insert(53, i_shells);
    let mut ecps = HashMap::new();
    ecps.insert(53, lanl2dz_i_ecp());
    BasisSet {
        name: if full { "HI-lanl2dz" } else { "HI-compact" }.into(),
        shells,
        ecps,
    }
}

/// An atom of `symbol` and atomic number `z` at Cartesian `r` Bohr (not a
/// ghost, no ECP core).
fn atom(symbol: &str, z: i32, r: [f64; 3]) -> Atom {
    Atom {
        symbol: symbol.into(),
        z,
        x: r[0],
        y: r[1],
        zpos: r[2],
        ghost: false,
        n_core_ecp: 0,
    }
}

/// H/I molecule at `pos` (repeated H, I pairs), optionally through
/// `apply_ecp(bs)`.
fn hi_molecule(pos: &[[f64; 3]], bs: &BasisSet, apply: bool) -> Molecule {
    let atoms = pos
        .iter()
        .enumerate()
        .map(|(i, r)| {
            if i % 2 == 0 {
                atom("H", 1, *r)
            } else {
                atom("I", 53, *r)
            }
        })
        .collect();
    let mut mol = Molecule {
        atoms,
        charge: 0,
        multiplicity: 1,
    };
    if apply {
        mol.apply_ecp(bs);
    }
    mol
}

/// The HI test cell: `hi_molecule(&HI_ATOMS, bs, true)` in the lattice
/// `HI_A`.
fn hi_cell(bs: &BasisSet) -> Cell {
    Cell::new(hi_molecule(&HI_ATOMS, bs, true), HI_A).expect("HI cell")
}

// ------------------------------------------------------------ open shell

const GRID_N: usize = 31;
/// Dense-AFT energies reproduce the PySCF pin to 2.3e-10 (measured, all 10
/// open-shell rows). The offset (-2.23e-10..-2.25e-10 on the triclinic cell,
/// +3.1e-11 on the cubic H3 pins) is identical for UKS, ROKS, ROHF (no XC) and
/// every functional, so it is an energy-independent per-cell constant (nuclear
/// / Madelung bookkeeping), not an XC or SCF effect. It is NOT PySCF's
/// cell.precision (regenerating at 1e-12 left it unchanged); its source on the
/// ferric or PySCF side is not separated.
const PIN_BAR: f64 = 1e-8;
/// ECP oracle: PySCF `cell.precision = 1e-12`. At its default 1e-8 the PySCF
/// ECP integral is 2.25e-6 off (max matrix element) and the KRKS PBE energy
/// 1.9e-6 off; 1e-12 vs 1e-14 agree to 3.0e-9 (PBE energy), so 1e-12 is
/// converged. Measured ferric - PySCF: -1.151e-6 (PBE), -1.096e-6 (PBE0): a
/// real, reproducible ECP-integral difference, NOT ferric's truncation (see
/// `ecp_gap_is_not_ferric_truncation`) nor PySCF's (converged). The remaining
/// candidates are ferric's ECP quadrature vs PySCF's, or the ECP basis/Z_eff
/// bookkeeping; not separated here. Bar derivation: below, 1.7x the larger
/// measured gap; above, ~2e-6 is the smallest defect size this file claims
/// to see. LIMIT: this pin cannot catch an ECP defect smaller than ~2e-6 Ha.
/// No ECP mutant is expressible through the production config (image caps /
/// `EcpMutation` sit below `PeriodicHcoreConfig`), so none was run here;
/// ECP-internal mutants live in `pbc_ecp.rs`, and only the bare-Z refusal is
/// exercised in this file.
const ECP_BAR: f64 = 2e-6;
/// RS-GDF-vs-dense gap (the cc-pVDZ-RI fit error) bar, and the smallest
/// movement a q-phase / G = 0 mutant must make (measured values in the
/// per-test comments).
const FIT_BAR: f64 = 1e-4;
const MUTANT_MIN: f64 = 5e-3;

/// First three atoms of the triclinic cell: a doublet.
fn h3_triclinic() -> Cell {
    let mut mol: Molecule = hydrogens(&TRI_ATOMS[..3]);
    mol.multiplicity = 2;
    Cell::new(mol, TRI_A).expect("cell")
}

fn rsgdf_cfg(mutation: Option<KRsGdfMutation>) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 0.8,
            exxdiv: ExxDiv::Ewald,
            budget_bytes: Some(1 << 30),
            ..Default::default()
        },
        mutation,
    }
}

fn aux_for(cell: &Cell) -> PreparedBasis {
    PreparedBasis::new(cell.mol(), &basis::bundled(AUX).unwrap()).unwrap()
}

/// `Some(mutation_option)` selects RS-GDF; `None` is the dense-AFT build.
type Rs = Option<Option<KRsGdfMutation>>;

fn kuks(cell: &Cell, mesh: &KPointMesh, f: &str, rs: Rs) -> (f64, bool, usize) {
    let grid = PeriodicGrid::uniform(cell, [GRID_N; 3]).unwrap();
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mut cfg = KUksConfig::new(cell, ExxDiv::Ewald, f);
    cfg.kuhf.hcore.omega = OMEGA;
    cfg.kuhf.scf.energy_conv = 1e-12;
    cfg.kuhf.scf.grad_conv = 1e-9;
    let aux = rs.map(|m| {
        cfg.kuhf.jk = KJkKind::RsGdf;
        cfg.kuhf.rsgdf = rsgdf_cfg(m);
        aux_for(cell)
    });
    let r = solve_kuks_on_grid(cell, &prep, aux.as_ref(), mesh, &grid, &cfg)
        .unwrap_or_else(|e| panic!("solve_kuks {f}: {e}"));
    // Total iterations: the staged exxdiv = none start plus the final ewald
    // stage (which alone can read "1 it" when the start already converged it).
    let it = r.scf.iterations + r.none_stage.as_ref().map_or(0, |s| s.iterations);
    (r.scf.energy, r.scf.converged, it)
}

fn kroks(
    cell: &Cell,
    mesh: &KPointMesh,
    f: &str,
    rs: Rs,
    level_shift: f64,
    start: EwaldStart,
) -> (f64, bool, usize) {
    let grid = PeriodicGrid::uniform(cell, [GRID_N; 3]).unwrap();
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mut cfg = KRoksConfig::new(cell, ExxDiv::Ewald, f);
    cfg.rohf.hcore.omega = OMEGA;
    cfg.rohf.scf.energy_conv = 1e-12;
    cfg.rohf.scf.grad_conv = 1e-9;
    cfg.rohf.level_shift = Some(level_shift);
    cfg.rohf.ewald_start = start;
    let aux = rs.map(|m| {
        cfg.rohf.jk = KJkKind::RsGdf;
        cfg.rohf.rsgdf = rsgdf_cfg(m);
        aux_for(cell)
    });
    let r = solve_kroks_on_grid(cell, &prep, aux.as_ref(), mesh, &grid, &cfg)
        .unwrap_or_else(|e| panic!("solve_kroks {f}: {e}"));
    let it = r.scf.iterations + r.none_stage.as_ref().map_or(0, |s| s.iterations);
    (r.scf.energy, r.scf.converged, it)
}

fn load_rows(cell_name: &str) -> (serde_json::Value, Vec<serde_json::Value>) {
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(REF_JSON).unwrap()).unwrap();
    let rows = j["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["cell"].as_str().unwrap() == cell_name)
        .cloned()
        .collect();
    (j, rows)
}

fn row_mesh(row: &serde_json::Value) -> [usize; 3] {
    let n: Vec<usize> = row["mesh"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    [n[0], n[1], n[2]]
}

/// The cell and grid of the file are the ones this test builds.
fn check_cell(j: &serde_json::Value, name: &str, cell: &Cell, n: usize) {
    let spec = &j["cells"][name];
    assert_eq!(spec["mesh"].as_u64().unwrap() as usize, n, "grid");
    for (i, a) in cell.lattice().iter().enumerate() {
        for (d, x) in a.iter().enumerate() {
            assert!((spec["a"][i][d].as_f64().unwrap() - x).abs() < 1e-12);
        }
    }
    for (i, p) in cell.positions().iter().enumerate() {
        for (d, x) in p.iter().enumerate() {
            assert!((spec["atoms"][i][1][d].as_f64().unwrap() - x).abs() < 1e-12);
        }
    }
}

#[test]
fn shifted_triclinic_kuks_kroks_match_pinned_pyscf() {
    let cell = h3_triclinic();
    let (j, rows) = load_rows("h3_triclinic");
    check_cell(&j, "h3_triclinic", &cell, GRID_N);
    let mut seen = 0;
    for row in rows {
        let method = row["method"].as_str().unwrap();
        if method == "KROHF" {
            continue;
        }
        let f = row["xc"].as_str().unwrap();
        let n = row_mesh(&row);
        let e_ref = row["energy"].as_f64().unwrap();
        let mesh = KPointMesh::monkhorst_pack(&cell, n).unwrap();
        // k-ROKS hybrids: NO level shift and the default iteration cap
        // (neither the automatic hybrid shift nor a 600 cap is needed here).
        let (e, conv, it) = if method == "KUKS" {
            kuks(&cell, &mesh, f, None)
        } else {
            kroks(&cell, &mesh, f, None, 0.0, EwaldStart::Staged)
        };
        eprintln!(
            "{method} {n:?} {f}: {e:.12} PySCF {e_ref:.12} dE {:.2e} ({it} it)",
            e - e_ref
        );
        assert!(conv, "{method} {n:?} {f} did not converge");
        assert!((e - e_ref).abs() < PIN_BAR, "{method} {n:?} {f}");
        seen += 1;
    }
    assert_eq!(seen, 8, "reference rows");
}

/// Both open-shell RS-GDF builds on the shifted 1x1x4 mesh (non-TRIM q =
/// +-1/4; at a shifted N = 2 mesh the q-phase mutant is invisible).
fn open_shell_rsgdf(method: &str) {
    let cell = h3_triclinic();
    let mesh = KPointMesh::monkhorst_pack(&cell, [1, 1, 4]).unwrap();
    let f = "PBE0";
    let run = |rs: Rs| {
        if method == "KUKS" {
            kuks(&cell, &mesh, f, rs).0
        } else {
            kroks(&cell, &mesh, f, rs, 0.0, EwaldStart::Staged).0
        }
    };
    let dense = run(None);
    let gap = |m: Option<KRsGdfMutation>| run(Some(m)) - dense;
    let g = gap(None);
    eprintln!("{method} shifted 1x1x4 {f}: fit gap {g:.3e}");
    assert!(g.abs() > 1e-7 && g.abs() < FIT_BAR, "fit gap {g:e}");
    let ntr = gap(Some(KRsGdfMutation::NoTimeReversal));
    eprintln!("NoTimeReversal - default {:.2e}", ntr - g);
    assert!((ntr - g).abs() < 1e-9, "time-reversal fill: {:e}", ntr - g);
    for m in [KRsGdfMutation::QPhaseSign, KRsGdfMutation::G0AllQ] {
        let d = gap(Some(m)) - g;
        eprintln!("{method} {m:?}: {d:.2e}");
        assert!(d.abs() > MUTANT_MIN, "{m:?} left the anchor green: {d:e}");
    }
}

#[test]
fn rsgdf_shifted_triclinic_kuks_is_dense_to_the_fit_error() {
    open_shell_rsgdf("KUKS");
}

#[test]
fn rsgdf_shifted_triclinic_kroks_is_dense_to_the_fit_error() {
    open_shell_rsgdf("KROKS");
}

/// ROKS level-shift measurement: the converged hybrid energy must not depend
/// on the shift (0 vs the Gamma default 0.05) nor on the start (staged vs one
/// direct ewald SCF), and the unshifted runs converge within the default
/// iteration cap (200; no 600 cap is needed).
#[test]
fn kroks_pbe0_needs_no_level_shift() {
    let cell = h3_triclinic();
    let mesh = KPointMesh::monkhorst_pack(&cell, [1, 1, 2]).unwrap();
    let runs = [
        ("staged ls=0", 0.0, EwaldStart::Staged),
        ("staged ls=0.05", 0.05, EwaldStart::Staged),
        ("direct ls=0", 0.0, EwaldStart::Direct),
        ("direct ls=0.05", 0.05, EwaldStart::Direct),
    ];
    let mut e0 = None;
    // Artifact guard: the knob is live (a huge shift slows the same SCF).
    let (_, _, it_ref) = kroks(&cell, &mesh, "PBE0", None, 0.0, EwaldStart::Direct);
    let (_, _, it_big) = kroks(&cell, &mesh, "PBE0", None, 3.0, EwaldStart::Direct);
    eprintln!("direct: ls=0 {it_ref} it, ls=3 {it_big} it");
    assert_ne!(it_ref, it_big, "level_shift has no effect on the SCF path");
    for (name, ls, start) in runs {
        let (e, conv, it) = kroks(&cell, &mesh, "PBE0", None, ls, start);
        eprintln!("PBE0 1x1x2 {name}: {e:.12} ({it} it total)");
        assert!(conv, "{name}");
        let e0 = *e0.get_or_insert(e);
        assert!((e - e0).abs() < 1e-8, "{name}: {:e}", e - e0);
    }
}

/// The ORIGINAL case that got a hand-set level shift 0.05 / 600 iterations in
/// `pbc_kroks.rs` (both removed in #356): H3 doublet, cubic a = 4.5, 1x1x2, PBE0, 45^3 uniform grid,
/// against the same PySCF `KROKS` pins, with NO level shift and the default
/// iteration cap.
#[test]
fn original_kroks_hybrid_pins_converge_without_shift_or_long_cap() {
    let atoms = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
    let mut mol: Molecule = hydrogens(&atoms);
    mol.multiplicity = 2;
    let cell = Cell::new(mol, cubic(4.5)).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let grid = PeriodicGrid::uniform(&cell, [45; 3]).unwrap();
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    for (exx, e_ref) in [
        (ExxDiv::Ewald, -1.483324363436),
        (ExxDiv::None, -1.332837545898),
    ] {
        let mut cfg = KRoksConfig::new(&cell, exx, "PBE0");
        cfg.rohf.hcore.omega = OMEGA;
        cfg.rohf.scf.energy_conv = 1e-12;
        cfg.rohf.scf.grad_conv = 1e-9;
        assert_eq!(cfg.rohf.level_shift, None, "automatic by default");
        cfg.rohf.level_shift = Some(0.0);
        let r = solve_kroks_on_grid(&cell, &prep, None, &mesh, &grid, &cfg).unwrap();
        let it = r.scf.iterations + r.none_stage.as_ref().map_or(0, |s| s.iterations);
        eprintln!(
            "H3 cubic PBE0 {exx:?}: {:.12} dE {:.2e} ({it} it)",
            r.scf.energy,
            r.scf.energy - e_ref
        );
        assert!(r.scf.converged, "{exx:?}");
        assert!((r.scf.energy - e_ref).abs() < 1e-8, "{exx:?}");
    }
}

/// PySCF `KROHF` rows (shifted 1x1x2 and 1x1x4): both converged in PySCF, so
/// both are pinned.
#[test]
fn krohf_matches_pyscf_on_both_shifted_meshes() {
    let cell = h3_triclinic();
    let (_, rows) = load_rows("h3_triclinic");
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mut pinned = 0;
    for row in rows {
        if row["method"].as_str().unwrap() != "KROHF" {
            continue;
        }
        let n = row_mesh(&row);
        let mesh = KPointMesh::monkhorst_pack(&cell, n).unwrap();
        let mut cfg = ferric_pbc::kroks::KRohfConfig::for_cell(&cell, ExxDiv::Ewald);
        cfg.hcore.omega = OMEGA;
        cfg.scf.energy_conv = 1e-12;
        cfg.scf.grad_conv = 1e-9;
        let r = solve_krohf(&cell, &prep, None, &mesh, &cfg).unwrap();
        let e_ref = row["energy"].as_f64().unwrap();
        eprintln!(
            "KROHF {n:?}: {:.12} PySCF {e_ref:.12} (PySCF converged {}) dE {:.2e}",
            r.scf.energy,
            row["converged"],
            r.scf.energy - e_ref
        );
        assert!(r.scf.converged);
        assert!(row["converged"].as_bool().unwrap(), "PySCF KROHF {n:?}");
        assert!((r.scf.energy - e_ref).abs() < PIN_BAR, "KROHF {n:?}");
        pinned += 1;
    }
    assert_eq!(pinned, 2, "KROHF reference rows");
}

// ------------------------------------------------------------------- ECP

fn hi_energy(cell: &Cell, bs: &BasisSet, mesh: &KPointMesh, f: &str, prec: Option<f64>) -> f64 {
    let grid = PeriodicGrid::uniform(cell, [45; 3]).unwrap();
    let prep = PreparedBasis::new(cell.mol(), bs).unwrap();
    let mut cfg = KRksConfig::new(cell, ExxDiv::Ewald, f);
    cfg.krhf.hcore.omega = OMEGA;
    if let Some(p) = prec {
        cfg.krhf.hcore.precision = p;
    }
    cfg.krhf.scf.energy_conv = 1e-12;
    cfg.krhf.scf.grad_conv = 1e-9;
    let r = solve_krks_on_grid(cell, &prep, None, mesh, &grid, &cfg)
        .unwrap_or_else(|e| panic!("solve_krks {f}: {e}"));
    assert!(r.scf.converged);
    r.scf.energy
}

/// HI with the I LANL2DZ ECP (46 core electrons): shifted 1x1x2 k-RKS PBE and
/// PBE0 against PySCF `KRKS` (own ECP integral at cell.precision 1e-12, hence `ECP_BAR`).
#[test]
fn ecp_cell_krks_matches_pinned_pyscf() {
    let bs = hi_basis(true);
    let cell = hi_cell(&bs);
    let (j, rows) = load_rows("hi_ecp");
    check_cell(&j, "hi_ecp", &cell, 45);
    let mut seen = 0;
    for row in rows {
        let f = row["xc"].as_str().unwrap();
        let n = row_mesh(&row);
        let mesh = KPointMesh::monkhorst_pack(&cell, n).unwrap();
        let e = hi_energy(&cell, &bs, &mesh, f, None);
        let e_ref = row["energy"].as_f64().unwrap();
        eprintln!(
            "HI {n:?} {f}: {e:.12} PySCF {e_ref:.12} dE {:.2e}",
            e - e_ref
        );
        assert!((e - e_ref).abs() < ECP_BAR, "HI {n:?} {f}");
        seen += 1;
    }
    assert_eq!(seen, 2);
}

/// A cell whose basis carries an ECP but whose atoms never went through
/// `apply_ecp` would charge Z = 53: it is refused, not run.
#[test]
fn ecp_cell_without_apply_ecp_is_refused() {
    let bs = hi_basis(true);
    let cell = Cell::new(hi_molecule(&HI_ATOMS, &bs, false), HI_A).unwrap();
    let mesh = KPointMesh::monkhorst_pack(&cell, [1, 1, 2]).unwrap();
    let grid = PeriodicGrid::uniform(&cell, [15; 3]).unwrap();
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap();
    let mut cfg = KRksConfig::new(&cell, ExxDiv::Ewald, "PBE");
    cfg.krhf.hcore.omega = OMEGA;
    let e = solve_krks_on_grid(&cell, &prep, None, &mesh, &grid, &cfg)
        .expect_err("bare-Z cell must be refused");
    assert!(format!("{e}").to_lowercase().contains("ecp"), "{e}");
}

/// Source of the ~1.1e-6 ferric-PySCF ECP gap: NOT ferric's lattice-sum
/// truncation. Tightening ferric's hcore precision (which sets the ECP image
/// and orbital-pair truncation) to 1e-12 leaves the energy where it was.
#[test]
fn ecp_gap_is_not_ferric_truncation() {
    let bs = hi_basis(true);
    let cell = hi_cell(&bs);
    let mesh = KPointMesh::monkhorst_pack(&cell, [1, 1, 2]).unwrap();
    let e0 = hi_energy(&cell, &bs, &mesh, "PBE", None);
    let e1 = hi_energy(&cell, &bs, &mesh, "PBE", Some(1e-12));
    eprintln!(
        "HI PBE: default {e0:.12} precision 1e-12 {e1:.12} d {:.2e}",
        e1 - e0
    );
    assert!((e1 - e0).abs() < 2e-8, "{:e}", e1 - e0);
}
