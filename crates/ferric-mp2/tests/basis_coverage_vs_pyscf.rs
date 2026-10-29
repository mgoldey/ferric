//! Bundled basis sets against PySCF: the PER-COMMIT part.
//!
//! The full sweep (every reference case in
//! `testdata/reference/basis_coverage_pyscf.json`, some at QZ) is the
//! validation-tier `validation_basis_coverage.rs`, which also says what is
//! compared, how the references were made, and records the mutation checks.
//! It is too slow for a shard, so this file keeps:
//!
//! * a small subset of each kind through the same harness
//!   (`common/basis_coverage.rs`), chosen to cover the code paths that differ
//!   between sets: a Pople Cartesian-d set, a spherical set, an ECP/PP set
//!   (RHF and RI-MP2), an RI aux set, and an aug-JKFIT set in its J/K role;
//! * the data-only guards (no SCF): every registered set has a reference
//!   case, the Pople references use the Gaussian d convention, and an ECP atom
//!   with h shells is a clean setup error.

#[path = "common/basis_coverage.rs"]
mod common;

use common::{molecule, refs};
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;

/// RHF: 6-31G* (Cartesian 6D), cc-pVDZ (spherical), cc-pVDZ-PP on HBr (ECP).
const FAST_RHF: &[&str] = &["6-31g*/h2o", "cc-pvdz/h2o", "cc-pvdz-pp/hbr"];
/// RI-MP2: an all-electron RI aux, and the PP RI aux on Pd (ECP).
const FAST_RIMP2: &[&str] = &[
    "def2-svp+def2-svp-rifit/h2o",
    "cc-pvdz-pp+cc-pvdz-pp-rifit/pd",
];
/// RI-J + RI-K RHF: aug-JKFIT (PySCF's copy) and def2-universal-JKFIT.
const FAST_DFHF: &[&str] = &[
    "aug-cc-pvdz+aug-cc-pvdz-jkfit/h2o",
    "def2-svp+def2-universal-jkfit/h2o",
];

#[test]
fn fast_orbital_sets_match_pyscf_rhf() {
    common::run_cases("rhf", Some(FAST_RHF));
}

#[test]
fn fast_auxiliary_sets_match_pyscf_rimp2() {
    common::run_cases("rimp2", Some(FAST_RIMP2));
}

#[test]
fn fast_jk_sets_match_pyscf_density_fitted_rhf() {
    common::run_cases("dfhf", Some(FAST_DFHF));
}

/// cc-pVQZ-PP / aug-cc-pVQZ-PP (and def2-QZVPP on the lanthanides) put h
/// (l = 5) shells on ECP atoms. The ECP integrals stop at l = 4, so that is a
/// clean error at basis setup, naming the shell and the ECP atom, instead of
/// the panic it used to be in the first hcore build.
#[test]
fn ecp_atom_with_h_shells_is_a_clean_setup_error() {
    for name in ["cc-pvqz-pp", "aug-cc-pvqz-pp"] {
        let bs = basis::bundled(name).unwrap();
        let mut mol = Molecule::parse_xyz("1\nPd\nPd 0 0 0\n", 0, 1).unwrap();
        mol.apply_ecp(&bs);
        let msg = match PreparedBasis::new(&mol, &bs) {
            Ok(_) => panic!("{name}/Pd must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("l=5") && msg.contains("ECP") && msg.contains("Pd"),
            "{msg}"
        );
        // The same family at TZ (max l = 4) is accepted.
    }
    let tz = basis::bundled("cc-pvtz-pp").unwrap();
    let mut mol = Molecule::parse_xyz("1\nPd\nPd 0 0 0\n", 0, 1).unwrap();
    mol.apply_ecp(&tz);
    assert!(
        PreparedBasis::new(&mol, &tz).is_ok(),
        "cc-pvtz-pp/Pd (max l = 4) must be accepted"
    );
}

/// Exhaustiveness: every registered set (by primary name) is exercised by at
/// least one reference case, as the orbital or the auxiliary basis.
#[test]
fn every_registered_set_has_a_case() {
    let refs = refs();
    let covered: std::collections::HashSet<&str> = refs
        .cases
        .iter()
        .flat_map(|c| std::iter::once(c.basis.as_str()).chain(c.auxbasis.as_deref()))
        .collect();
    let missing: Vec<&str> = basis::bundled_names()
        .into_iter()
        .filter(|n| !covered.contains(n))
        .collect();
    assert!(
        missing.is_empty(),
        "bundled sets with no PySCF reference case: {missing:?}"
    );
}

/// The Pople references were computed under ferric's (Gaussian's) d-shell
/// convention: every 6-31G-family case that has d shells with PySCF
/// `cart=True`, every 6-311G-family case with PySCF's default `cart=False`
/// (spherical on every element, Na-Ar included). A mixed C/Cl molecule at
/// 6-311G** is among them, where BSE's own typing would mix 5D and 6D.
#[test]
fn pople_references_use_the_gaussian_d_convention() {
    let refs = refs();
    let (mut n_6d, mut n_5d) = (0, 0);
    for c in refs.cases.iter().filter(|c| c.basis.starts_with("6-31")) {
        let bs = basis::bundled(&c.basis).unwrap();
        let has_d = molecule(c)
            .atoms
            .iter()
            .any(|a| bs.for_element(a.z).unwrap().iter().any(|s| s.l == 2));
        if c.basis.starts_with("6-311") {
            assert!(!c.cart, "{}: 6-311G family must be cart=False", c.label());
            n_5d += usize::from(has_d);
        } else {
            assert_eq!(
                c.cart,
                has_d,
                "{}: 6-31G family with d must be cart=True",
                c.label()
            );
            n_6d += usize::from(has_d);
        }
    }
    assert!(
        n_6d >= 6 && n_5d >= 5,
        "only {n_6d} 6D and {n_5d} 5D d-carrying cases"
    );
    assert!(
        refs.cases
            .iter()
            .any(|c| c.basis == "6-311g**" && c.molecule == "ch3cl"),
        "the mixed C/Cl 6-311G** case is missing"
    );
}
