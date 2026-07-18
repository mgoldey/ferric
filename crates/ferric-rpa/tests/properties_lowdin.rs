//! Cross-check of `ferric_rpa::properties::lowdin_charges` against an
//! independent PySCF reference (S^1/2 D S^1/2 symmetric Löwdin population,
//! generated two independent ways in `scripts/gen_pyscf_lowdin_ref.py` --
//! hand `eigh`-based S^1/2 vs `pyscf.lo.orth_ao(method='lowdin')` -- which
//! agree with each other to ~1e-14, so the PySCF numbers below are trusted
//! ground truth).
//!
//! RESOLUTION (task F1, 2026-07-17): ferric's `lowdin_charges` is CORRECT.
//! It matches PySCF to ~1e-9 when both use the SAME basis functions.
//!
//! History of the earlier apparent discrepancy (S7 spike, item #7): the S7
//! spike compared ferric against a PySCF reference generated with the STRING
//! `basis='cc-pvdz'`, which makes PySCF load its OWN internal `cc-pvdz.dat`.
//! PySCF's internal table is a *segmented* re-expression of Dunning's
//! cc-pVDZ: for oxygen it stores an 8-primitive tight/medium s-shell plus a
//! separate 1-primitive diffuse s-shell, DROPPING the most-diffuse primitive
//! (exp 0.3023) from the tight/medium contraction columns entirely.
//!
//! Ferric's bundled `cc-pvdz.json` (and aug-cc-pvdz / aug-cc-pvdz-pp /
//! aug-cc-pvtz) instead carries the AUTHENTIC Dunning *general contraction*:
//! that same 0.3023 primitive appears in ALL three s columns with its
//! genuine published coefficients (tight `-2.585e-3`, medium `0.572759`,
//! diffuse `1.0`). This was verified against the Basis Set Exchange
//! (`https://www.basissetexchange.org/api/basis/cc-pvdz/format/{json,nwchem,
//! gaussian94}`) across TWO independent BSE revisions (v0 "Original Basis Set
//! Exchange", v1 "ccRepo/Grant Hill"), both citing Dunning 1989 -- see
//! `docs/basis-data-corrections.md` for the full provenance chain.
//!
//! The two representations span the IDENTICAL variational space (rank-3
//! s-block either way), so SCF energies agree to ~2e-13 Ha. But Löwdin
//! charges are AO-identity-dependent: PySCF's segmented tight/medium s
//! functions are DIFFERENT individual basis functions from ferric's/Dunning's
//! general-contraction ones, so PySCF's *string-loaded* Löwdin charges are
//! simply not comparable to ferric's atom-by-atom. That was the entire "5-6x
//! discrepancy" -- a basis-representation mismatch in the REFERENCE, not a
//! ferric bug.
//!
//! The fix (task F1): NO basis JSON was changed (ferric's data is authentic).
//! `scripts/gen_pyscf_lowdin_ref.py` now feeds ferric's OWN bundled basis
//! JSON to PySCF as an explicit basis dict, so the reference is for the same
//! basis functions. Ferric then matches it to ~1e-9 (tight tolerance below).
//!
//! Reference JSONs: testdata/reference/{water,methane,h2}_cc-pvdz_lowdin.json
//! (regenerated from ferric's authentic general contraction).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::lowdin_charges;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn run_lowdin(xyz: &str, basis_name: &str) -> (f64, Vec<f64>) {
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
    let charges = lowdin_charges(&mol, &obs, rhf.density_r()).unwrap();
    (rhf.energy, charges)
}

/// Tolerance for the ferric-vs-PySCF Löwdin cross-check. Both compute the
/// same well-defined linear-algebra quantity (S^1/2 D S^1/2) from a
/// converged density over identical basis functions, so they agree to SCF-
/// convergence precision. 1e-7 leaves margin for the independent SCF
/// convergence paths (ferric density_conv 1e-11 vs PySCF conv_tol 1e-12).
const LOWDIN_TOL: f64 = 1e-7;

/// H2/cc-pVDZ: homonuclear, both charges ~0 by symmetry. Sanity check that
/// the plumbing works and the reference matches.
#[test]
fn h2_lowdin_matches_pyscf() {
    // PySCF reference (testdata/reference/h2_cc-pvdz_lowdin.json): [0, ~0].
    let (e, q) = run_lowdin("2\nh2\nH 0 0 0\nH 0 0 0.740830\n", "cc-pvdz");
    assert!((e - (-1.128709260618651)).abs() < 1e-8, "SCF energy mismatch: {e}");
    assert_eq!(q.len(), 2);
    for &qi in &q {
        assert!(qi.abs() < LOWDIN_TOL, "expected ~0 by symmetry, got {qi:.3e}");
    }
    assert!((q[0] + q[1]).abs() < 1e-10, "charges must sum to 0");
}

/// Water/cc-pVDZ: ferric matches PySCF (fed ferric's own authentic Dunning
/// general-contraction basis) to <1e-7.
/// PySCF reference (testdata/reference/water_cc-pvdz_lowdin.json):
///   q_O = -0.48084935437182175, q_H = +0.24042467718591243 (x2)
#[test]
fn h2o_lowdin_matches_pyscf() {
    let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
    let (e, q) = run_lowdin(xyz, "cc-pvdz");
    assert!((e - (-76.02676799737671)).abs() < 1e-8, "SCF energy mismatch: {e}");
    assert_eq!(q.len(), 3);
    assert!((q.iter().sum::<f64>()).abs() < 1e-8, "charges must sum to 0");

    let pyscf_o = -0.48084935437182175_f64;
    let pyscf_h = 0.24042467718591243_f64;
    assert!(
        (q[0] - pyscf_o).abs() < LOWDIN_TOL,
        "O Löwdin charge {:.10} != PySCF {pyscf_o:.10} (diff {:.2e})",
        q[0],
        (q[0] - pyscf_o).abs()
    );
    assert!(
        (q[1] - pyscf_h).abs() < LOWDIN_TOL,
        "H[1] Löwdin charge {:.10} != PySCF {pyscf_h:.10} (diff {:.2e})",
        q[1],
        (q[1] - pyscf_h).abs()
    );
    assert!(
        (q[2] - pyscf_h).abs() < LOWDIN_TOL,
        "H[2] Löwdin charge {:.10} != PySCF {pyscf_h:.10} (diff {:.2e})",
        q[2],
        (q[2] - pyscf_h).abs()
    );
}

/// Methane/cc-pVDZ: ferric matches PySCF to <1e-7.
/// PySCF reference (testdata/reference/methane_cc-pvdz_lowdin.json):
///   q_C = -0.5659965387523984, q_H = +0.14149913468809794 (x4)
#[test]
fn ch4_lowdin_matches_pyscf() {
    let xyz = "5\nmethane\nC 0.000000 0.000000 0.000000\nH 0.629118 0.629118 0.629118\nH -0.629118 -0.629118 0.629118\nH -0.629118 0.629118 -0.629118\nH 0.629118 -0.629118 -0.629118\n";
    let (e, q) = run_lowdin(xyz, "cc-pvdz");
    assert!((e - (-40.19870854248165)).abs() < 1e-8, "SCF energy mismatch: {e}");
    assert_eq!(q.len(), 5);
    assert!((q.iter().sum::<f64>()).abs() < 1e-8, "charges must sum to 0");

    let pyscf_c = -0.5659965387523984_f64;
    let pyscf_h = 0.14149913468809794_f64;
    assert!(
        (q[0] - pyscf_c).abs() < LOWDIN_TOL,
        "C Löwdin charge {:.10} != PySCF {pyscf_c:.10} (diff {:.2e})",
        q[0],
        (q[0] - pyscf_c).abs()
    );
    for (i, &qh) in q.iter().enumerate().skip(1) {
        assert!(
            (qh - pyscf_h).abs() < LOWDIN_TOL,
            "H[{i}] Löwdin charge {qh:.10} != PySCF {pyscf_h:.10} (diff {:.2e})",
            (qh - pyscf_h).abs()
        );
    }
}
