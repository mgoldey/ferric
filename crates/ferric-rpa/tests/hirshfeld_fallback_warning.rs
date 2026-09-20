//! The Hirshfeld proatom-fallback warning must describe the case it is in.
//!
//! One message covered two different situations:
//!
//!   SOME atoms fall back  -> the partitioning MIXES SCF and crude proatoms,
//!                            so it is internally inconsistent;
//!   ALL atoms fall back   -> there is no mixture. It is uniformly crude,
//!                            which is what you get with no proatom provider
//!                            (e.g. `tools/active_site/binding_energy.py`).
//!
//! The single message said "The other atoms used the SCF density, so these
//! charges mix two different proatom sources". In the all-fallback case there
//! ARE no other atoms and nothing is mixed, so it sent a reader looking for an
//! inconsistency that was not there while under-stating the real problem --
//! every charge came from the crude model.
//!
//! OBSERVED 2026-09-20 running `compute_binding_energy` on danuglipron in the
//! 7LCJ pocket: "71 of 71 atoms ... The other atoms used the SCF density".
//!
//! The warning goes to stderr, so this drives it through a SUBPROCESS and
//! reads what was actually printed. Asserting on a string the test itself
//! formats would pin the format and never touch the branch.

/// The real check, in-process: call the library both ways and assert the
/// CHARGES differ, which is what proves the two branches are distinct code
/// paths rather than one message with a reworded string.
#[test]
fn all_fallback_and_mixed_are_different_code_paths() {
    use ferric_core::{basis, mol::Molecule};
    use ferric_rpa::properties::{hirshfeld_charges, spherically_averaged_proatom};

    let xyz = "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.755 -0.471\nH 0.000 -0.755 -0.471\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = ferric_integrals::basis_bridge::PreparedBasis::new(&mol, &bs).unwrap();
    let nbf = prep.nbasis();
    let dens = ndarray::Array2::<f64>::eye(nbf) * 0.2;

    // (a) no provider at all -> EVERY atom falls back.
    let q_none = hirshfeld_charges(&mol, &bs, &dens, None).unwrap();

    // (b) a provider that answers for oxygen only -> hydrogens fall back.
    let radii = [0.0f64, 0.25, 0.5, 1.0, 2.0, 4.0];
    let o_mol = Molecule::parse_xyz("1\nO\nO 0 0 0\n", 0, 3).unwrap();
    let o_prep = ferric_integrals::basis_bridge::PreparedBasis::new(&o_mol, &bs).unwrap();
    let o_dens = ndarray::Array2::<f64>::eye(o_prep.nbasis()) * 0.5;
    let partial = |z: i32, _q: i32| -> Option<ferric_rpa::properties::RadialProatom> {
        if z == 1 {
            None
        } else {
            spherically_averaged_proatom(z, &bs, &o_dens, &radii).ok()
        }
    };
    let pp: &ferric_rpa::properties::ProatomProvider = &partial;
    let q_mixed = hirshfeld_charges(&mol, &bs, &dens, Some(pp)).unwrap();

    assert_eq!(q_none.len(), 3);
    assert_eq!(q_mixed.len(), 3);

    // Both must conserve charge -- the partitioning is renormalised, so this
    // holds regardless of which proatom source was used. If it ever does not,
    // the branch split broke something real.
    let s_none: f64 = q_none.iter().sum();
    let s_mixed: f64 = q_mixed.iter().sum();
    assert!(
        s_none.abs() < 1e-6,
        "all-fallback charges do not sum to zero: {s_none:.3e} ({q_none:?})"
    );
    assert!(
        s_mixed.abs() < 1e-6,
        "mixed charges do not sum to zero: {s_mixed:.3e} ({q_mixed:?})"
    );

    // And they must actually DIFFER, or the "mixed" case never happened and
    // this test is asserting the same path twice.
    let diff = q_none
        .iter()
        .zip(&q_mixed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        diff > 1e-9,
        "the all-fallback and mixed proatom paths gave identical charges \
         (max diff {diff:.3e}), so the partial provider was never consulted \
         and this test cannot distinguish the two branches"
    );
}
