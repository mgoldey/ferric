//! Cross-check of `ferric_rpa::properties::mulliken_charges` against an
//! independent PySCF reference (`mol.mulliken_pop()`), generated two
//! independent ways in `scripts/gen_pyscf_mulliken_ref.py` -- hand `D @ S`
//! diagonal vs `pyscf.scf.hf.mulliken_pop`, which agree with each other to
//! ~1e-15, so the PySCF numbers below are trusted ground truth.
//!
//! Unlike Löwdin (`properties_lowdin.rs`), Mulliken has no meta/
//! orthogonalization variant to disambiguate -- `mf.mulliken_pop()` is the
//! one unambiguous textbook definition, so there is no F1-style history to
//! recount here. The one lesson that DOES carry over from the Löwdin work:
//! the basis must be loaded from ferric's OWN bundled JSON and fed to PySCF
//! as an explicit basis dict (not the string `basis='cc-pvdz'`, which makes
//! PySCF load its own internal *segmented* re-expression of Dunning's
//! general contraction -- different individual basis functions, so a
//! population reference built from PySCF's string-loaded basis would not be
//! comparable to ferric atom-by-atom, even though both span the same
//! variational space). See docs/basis-data-corrections.md and
//! `properties_lowdin.rs`'s doc comment for the full provenance chain.
//!
//! Reference JSONs: testdata/reference/{water,methane,h2}_cc-pvdz_mulliken.json.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::mulliken_charges;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn run_mulliken(xyz: &str, basis_name: &str) -> (f64, Vec<f64>) {
    let ctx = ParallelContext::new();
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ctx,
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { energy_conv: 1e-12, density_conv: 1e-11, ..Default::default() },
    )
    .unwrap();
    let charges = mulliken_charges(&mol, &obs, rhf.density_r()).unwrap();
    (rhf.energy, charges)
}

/// Tolerance for the ferric-vs-PySCF Mulliken cross-check. Both compute the
/// same well-defined linear-algebra quantity (diag(D @ S)) from a converged
/// density over identical basis functions, so they agree to SCF-convergence
/// precision. 1e-7 leaves margin for the independent SCF convergence paths
/// (ferric density_conv 1e-11 vs PySCF conv_tol 1e-12).
const MULLIKEN_TOL: f64 = 1e-7;

/// H2/cc-pVDZ: homonuclear, both charges ~0 by symmetry. Sanity check that
/// the plumbing works and the reference matches.
#[test]
fn h2_mulliken_matches_pyscf() {
    // PySCF reference (testdata/reference/h2_cc-pvdz_mulliken.json): [~0, ~0].
    let (e, q) = run_mulliken("2\nh2\nH 0 0 0\nH 0 0 0.740830\n", "cc-pvdz");
    assert!((e - (-1.128709260618651)).abs() < 1e-8, "SCF energy mismatch: {e}");
    assert_eq!(q.len(), 2);
    for &qi in &q {
        assert!(qi.abs() < MULLIKEN_TOL, "expected ~0 by symmetry, got {qi:.3e}");
    }
    assert!((q[0] + q[1]).abs() < 1e-10, "charges must sum to 0");
}

/// Water/cc-pVDZ: ferric matches PySCF (fed ferric's own authentic Dunning
/// general-contraction basis) to <1e-7.
/// PySCF reference (testdata/reference/water_cc-pvdz_mulliken.json):
///   q_O = -0.30538662807075667, q_H = +0.15269331403537867 (x2)
#[test]
fn h2o_mulliken_matches_pyscf() {
    let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
    let (e, q) = run_mulliken(xyz, "cc-pvdz");
    assert!((e - (-76.02676799737671)).abs() < 1e-8, "SCF energy mismatch: {e}");
    assert_eq!(q.len(), 3);
    assert!((q.iter().sum::<f64>()).abs() < 1e-8, "charges must sum to 0");

    let pyscf_o = -0.30538662807075667_f64;
    let pyscf_h = 0.15269331403537867_f64;
    assert!(
        (q[0] - pyscf_o).abs() < MULLIKEN_TOL,
        "O Mulliken charge {:.10} != PySCF {pyscf_o:.10} (diff {:.2e})",
        q[0],
        (q[0] - pyscf_o).abs()
    );
    assert!(
        (q[1] - pyscf_h).abs() < MULLIKEN_TOL,
        "H[1] Mulliken charge {:.10} != PySCF {pyscf_h:.10} (diff {:.2e})",
        q[1],
        (q[1] - pyscf_h).abs()
    );
    assert!(
        (q[2] - pyscf_h).abs() < MULLIKEN_TOL,
        "H[2] Mulliken charge {:.10} != PySCF {pyscf_h:.10} (diff {:.2e})",
        q[2],
        (q[2] - pyscf_h).abs()
    );
}

/// Methane/cc-pVDZ: ferric matches PySCF to <1e-7.
/// PySCF reference (testdata/reference/methane_cc-pvdz_mulliken.json):
///   q_C = -0.142067780409584, q_H = +0.03551694510239434 (x4)
#[test]
fn ch4_mulliken_matches_pyscf() {
    let xyz = "5\nmethane\nC 0.000000 0.000000 0.000000\nH 0.629118 0.629118 0.629118\nH -0.629118 -0.629118 0.629118\nH -0.629118 0.629118 -0.629118\nH 0.629118 -0.629118 -0.629118\n";
    let (e, q) = run_mulliken(xyz, "cc-pvdz");
    assert!((e - (-40.19870854248165)).abs() < 1e-8, "SCF energy mismatch: {e}");
    assert_eq!(q.len(), 5);
    assert!((q.iter().sum::<f64>()).abs() < 1e-8, "charges must sum to 0");

    let pyscf_c = -0.142067780409584_f64;
    let pyscf_h = 0.03551694510239434_f64;
    assert!(
        (q[0] - pyscf_c).abs() < MULLIKEN_TOL,
        "C Mulliken charge {:.10} != PySCF {pyscf_c:.10} (diff {:.2e})",
        q[0],
        (q[0] - pyscf_c).abs()
    );
    for (i, &qh) in q.iter().enumerate().skip(1) {
        assert!(
            (qh - pyscf_h).abs() < MULLIKEN_TOL,
            "H[{i}] Mulliken charge {qh:.10} != PySCF {pyscf_h:.10} (diff {:.2e})",
            (qh - pyscf_h).abs()
        );
    }
}
