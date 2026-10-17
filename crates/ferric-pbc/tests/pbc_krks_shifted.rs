//! k-point KS-DFT on SHIFTED (Monkhorst-Pack, `make_kpts(n,
//! with_gamma_point=False)`) meshes and a triclinic cell, dense AFT and RS-GDF
//! (issue #356 validation breadth). The existing KS anchors (`pbc_krks.rs`,
//! `pbc_krsh*.rs`) are all Gamma-centred cubic H2.
//!
//! # Physics hypothesis / artifact hypothesis (written before measuring)
//!
//! Physics: a shifted even mesh samples the BZ at `(m + 1/2)/N - 1/2`, so its
//! E/cell differs from the Gamma-centred one at O(1e-2..1e-1) Ha for H2 at
//! a = 4 Bohr, and every PySCF `KRKS` pin on it is reproduced to the grid /
//! SCF noise (~1e-8 Ha, as on Gamma-centred meshes). An ODD shifted mesh
//! contains the same points as the Gamma-centred one (`(2m+1-N)/2N` with
//! `2m+1-N` even), so there the two must agree to round-off.
//! Artifact: if the shift were silently dropped, the shifted energies would
//! equal the Gamma-centred ones (the `shift_is_not_dropped` guard asserts a gap
//! of >= 1e-3 Ha, observed ~1e-2); if the time-reversal fill of the q-classes
//! were wrong at non-TRIM points, `NoTimeReversal` (build every class
//! explicitly, NOT a defect) would differ from the default build.
//!
//! The RS-GDF build is compared with the dense-AFT build on the same grid;
//! their gap is the aux fit error, bounded from both sides by measurement
//! (see `FIT_BAR`): the genuine gap is <= 2e-4 Ha while the q-phase and
//! split-kernel mutants move E by >= 1e-3 Ha.

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGrid;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig, KRksResult};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::KJkKind;
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdfConfig, KRsGdfMutation};
use ferric_pbc::rsgdf::RsGdfConfig;

const OMEGA: f64 = 0.8;
const REF_JSON: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../testdata/reference/pbc_kks_shifted_sto3g.json"
);

fn cell_of(name: &str) -> (Cell, usize) {
    match name {
        "h2_cubic_a4" => (h2_cell(4.0), 61),
        "h4_triclinic" => (triclinic_cell(), 45),
        _ => panic!("unknown cell {name}"),
    }
}

fn run(
    cell: &Cell,
    bs: &BasisSet,
    aux: Option<&str>,
    mesh: &KPointMesh,
    grid: &PeriodicGrid,
    f: &str,
    mutation: Option<KRsGdfMutation>,
) -> KRksResult {
    let prep = prep_for(cell, bs);
    let aux = aux.map(|n| PreparedBasis::new(cell.mol(), &basis::bundled(n).unwrap()).unwrap());
    let mut cfg = KRksConfig::new(cell, ExxDiv::Ewald, f);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-12;
    cfg.krhf.scf.grad_conv = 1e-9;
    if aux.is_some() {
        cfg.krhf.jk = KJkKind::RsGdf;
        cfg.krhf.rsgdf = KRsGdfConfig {
            gdf: RsGdfConfig {
                omega: 0.8,
                exxdiv: ExxDiv::Ewald,
                budget_bytes: Some(1 << 30),
                ..Default::default()
            },
            mutation,
        };
    }
    solve_krks_on_grid(cell, &prep, aux.as_ref(), mesh, grid, &cfg)
        .unwrap_or_else(|e| panic!("solve_krks {f}: {e}"))
}

fn energy(cell: &Cell, mesh: &KPointMesh, n: usize, f: &str) -> f64 {
    let grid = PeriodicGrid::uniform(cell, [n; 3]).unwrap();
    run(cell, &pyscf_sto3g_h(), None, mesh, &grid, f, None)
        .scf
        .energy
}

/// Odd-N Monkhorst-Pack is the Gamma-centred mesh (same points, so the same
/// E to round-off) for the full dense KS path.
#[test]
fn odd_monkhorst_pack_is_gamma_centred() {
    let cell = h2_cell(4.0);
    for f in ["PBE", "PBE0"] {
        let g = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
        let m = KPointMesh::monkhorst_pack(&cell, [1, 1, 3]).unwrap();
        let (eg, em) = (energy(&cell, &g, 31, f), energy(&cell, &m, 31, f));
        eprintln!("{f} 1x1x3: gamma {eg:.12} mp {em:.12} d {:.2e}", em - eg);
        assert!((eg - em).abs() < 1e-9, "{f}: {eg} vs {em}");
    }
}

/// The artifact guard of the identity above: for an EVEN mesh the shift is
/// physical, so dropping it would be visible.
#[test]
fn shift_is_not_dropped() {
    let cell = h2_cell(4.0);
    let g = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let m = KPointMesh::monkhorst_pack(&cell, [1, 1, 2]).unwrap();
    let d = energy(&cell, &m, 31, "PBE") - energy(&cell, &g, 31, "PBE");
    eprintln!("PBE 1x1x2 shifted - gamma-centred: {d:.3e}");
    assert!(d.abs() > 1e-3, "shifted == gamma-centred: {d:e}");
}

/// Rows of `testdata/reference/pbc_kks_shifted_sto3g.json` (generator
/// `scripts/gen_pyscf_kks_shifted_refs.py`) for `cell_name`: PySCF `KRKS`
/// (AFTDF, UniformGrids, ewald) on shifted meshes.
fn check_pins(cell_name: &str) {
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(REF_JSON).unwrap()).unwrap();
    let (cell, ngrid) = cell_of(cell_name);
    let mut seen = 0;
    for row in j["rows"].as_array().unwrap() {
        if row["cell"].as_str().unwrap() != cell_name {
            continue;
        }
        let n: Vec<usize> = row["mesh"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let f = row["xc"].as_str().unwrap();
        let e_ref = row["energy"].as_f64().unwrap();
        let mesh = KPointMesh::monkhorst_pack(&cell, [n[0], n[1], n[2]]).unwrap();
        let e = energy(&cell, &mesh, ngrid, f);
        eprintln!(
            "{cell_name} {n:?} {f}: {e:.12} PySCF {e_ref:.12} dE {:.2e}",
            e - e_ref
        );
        assert!((e - e_ref).abs() < 1e-8, "{cell_name} {n:?} {f}");
        seen += 1;
    }
    assert!(seen >= 4, "{cell_name}: only {seen} reference rows");
}

#[test]
fn shifted_mesh_triclinic_krks_matches_pinned_pyscf() {
    check_pins("h4_triclinic");
}

#[test]
#[ignore = "heavy: 61^3 dense-AFT KS on 2 and 4 shifted k-points; run serially on a quiet box (minutes)"]
fn shifted_mesh_cubic_krks_matches_pinned_pyscf() {
    check_pins("h2_cubic_a4");
}

// ------------------------------------------------------------------ RS-GDF

const AUX: &str = "cc-pvdz-ri";
const GRID: usize = 31;

/// A genuine RS-GDF-vs-dense gap (the cc-pVDZ-RI fit error) is <= 1.2e-4 Ha
/// here (measured shifted 1x1x4 PBE0: H2 -9.7e-5, triclinic -3.5e-5; the
/// Gamma-centred meshes of the same cells give -1.2e-4 / -3.5e-5). The
/// q-phase and G = 0 mutants move E by 1.1e-1 / 1.4e-2 (H2) and 2.0e-1 /
/// 1.9e-2 (triclinic). `FIT_BAR` sits 8x above the first and 14x below the
/// second.
const FIT_BAR: f64 = 1e-3;
const MUTANT_MIN: f64 = 5e-3;

/// Shifted 1x1x4 has non-TRIM q = +-1/4 (a shifted N = 2 mesh has only TRIM
/// q, where `e^{+iq.T} == e^{-iq.T}` and `QPhaseSign` is invisible: measured
/// 0.0 on every N = 2 mesh).
fn shifted_rsgdf(cell: &Cell) {
    let mesh = KPointMesh::monkhorst_pack(cell, [1, 1, 4]).unwrap();
    let f = "PBE0";
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::uniform(cell, [GRID; 3]).unwrap();
    let dense = run(cell, &bs, None, &mesh, &grid, f, None).scf.energy;
    let gap = |m: Option<KRsGdfMutation>| {
        run(cell, &bs, Some(AUX), &mesh, &grid, f, m).scf.energy - dense
    };
    let g = gap(None);
    eprintln!("shifted 1x1x4 {f}: fit gap {g:.3e}");
    assert!(g.abs() > 1e-7 && g.abs() < FIT_BAR, "fit gap {g:e}");
    // NOT a defect: every q class built explicitly instead of the
    // time-reversal fill; must reproduce the default to round-off.
    let ntr = gap(Some(KRsGdfMutation::NoTimeReversal));
    eprintln!("NoTimeReversal - default {:.2e}", ntr - g);
    assert!((ntr - g).abs() < 1e-9, "time-reversal fill: {:e}", ntr - g);
    for m in [KRsGdfMutation::QPhaseSign, KRsGdfMutation::G0AllQ] {
        let d = gap(Some(m)) - g;
        eprintln!("{m:?}: {d:.2e}");
        assert!(d.abs() > MUTANT_MIN, "{m:?} left the anchor green: {d:e}");
    }
}

#[test]
fn rsgdf_shifted_mesh_is_dense_to_the_fit_error_cubic() {
    shifted_rsgdf(&h2_cell(4.0));
}

#[test]
fn rsgdf_shifted_mesh_is_dense_to_the_fit_error_triclinic() {
    shifted_rsgdf(&triclinic_cell());
}
