//! k-RKS on a 2x2x1 mesh (the existing PySCF KS pins are all 1x1xN),
//! Gamma-centred and shifted, vs PySCF `KRKS` (issue #356 validation).
//!
//! # Physics / artifact hypothesis (written before measuring)
//!
//! Physics: a 2x2x1 mesh has k-vectors with components along two axes, so
//! `q = k - k'` mixes the a and b reciprocal directions; ferric matches
//! PySCF to SCF noise (~1e-8 Ha) as on 1x1xN. Artifact: an axis-ordering or
//! mesh-index error (e.g. a transposed (i, j) fold of the q class) is
//! invisible on 1x1xN, since only one axis varies, and would move the energy
//! by >> 1e-6; the shifted row also guards a dropped shift.
//! LIMIT: H2 in a cube is symmetric in a and b, so an a<->b swap is NOT
//! visible here; the triclinic cell is not pinned on 2x2x1.

mod common;

use common::*;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGrid;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig};
use ferric_pbc::kpts::KPointMesh;

const REF_JSON: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../testdata/reference/pbc_kks_h2_31_sto3g.json"
);

fn energy(mesh: &KPointMesh, f: &str, exxdiv: ExxDiv) -> f64 {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let grid = PeriodicGrid::uniform(&cell, [31; 3]).unwrap();
    let mut cfg = KRksConfig::new(&cell, exxdiv, f);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(0.8);
    cfg.krhf.scf.energy_conv = 1e-12;
    cfg.krhf.scf.grad_conv = 1e-9;
    let r = solve_krks_on_grid(&cell, &prep, None, mesh, &grid, &cfg).unwrap();
    assert!(r.scf.converged);
    r.scf.energy
}

#[test]
fn krks_2x2x1_matches_pinned_pyscf() {
    let cell = h2_cell(4.0);
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(REF_JSON).unwrap()).unwrap();
    let mut seen = 0;
    for row in j["rows"].as_array().unwrap() {
        if row["method"] != "KRKS" {
            continue;
        }
        let f = row["xc"].as_str().unwrap();
        let gamma = row["gamma_centred"].as_bool().unwrap();
        let mesh = if gamma {
            KPointMesh::gamma_centred(&cell, [2, 2, 1]).unwrap()
        } else {
            KPointMesh::monkhorst_pack(&cell, [2, 2, 1]).unwrap()
        };
        let e = energy(&mesh, f, ExxDiv::Ewald);
        let e_ref = row["energy"].as_f64().unwrap();
        eprintln!(
            "2x2x1 gamma={gamma} {f}: {e:.12} PySCF {e_ref:.12} dE {:.2e}",
            e - e_ref
        );
        assert!((e - e_ref).abs() < 1e-7, "2x2x1 gamma={gamma} {f}");
        seen += 1;
    }
    assert_eq!(seen, 4);
}

/// Mutant guard: PBE0 without the exxdiv term misses the pin by >> bar.
#[test]
fn dropped_madelung_misses_the_2x2x1_pin() {
    let cell = h2_cell(4.0);
    let mesh = KPointMesh::gamma_centred(&cell, [2, 2, 1]).unwrap();
    let d = energy(&mesh, "PBE0", ExxDiv::None) - energy(&mesh, "PBE0", ExxDiv::Ewald);
    eprintln!("ExxDiv::None - Ewald {d:.3e}");
    assert!(d.abs() > 1e-3);
}
