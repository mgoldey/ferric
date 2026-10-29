//! VALIDATION tier — VALIDATION.md row "Bundled basis sets".
//!
//! Every bundled basis set against PySCF fed the same basis from the Basis
//! Set Exchange: the exactness anchor that a set must pass before it counts.
//! All 77 reference cases (some at QZ) take longer than a per-commit shard
//! allows, so the sweep runs only in the weekly `validation` CI job (and on
//! demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-mp2 --test validation_basis_coverage \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! `basis_coverage_vs_pyscf.rs` runs a small subset of each kind per commit,
//! plus the data-only coverage guards.
//!
//! References: `testdata/reference/basis_coverage_pyscf.json`, written by
//! `scripts/gen_pyscf_basis_coverage_refs.py`. That script fetches each set
//! from BSE BY NAME and lets PySCF parse and normalize it, so it shares
//! neither ferric's bundled JSON nor its loader: agreement checks the shipped
//! digits, the BSE-JSON parse, the contraction renormalization, the
//! spherical/Cartesian choice and (for the -PP sets) the inline ECP.
//!
//! * orbital sets: RHF energy, exact integrals, `TOL_E` Ha;
//! * RI / JK auxiliary sets: all-electron RI-MP2 correlation energy with the
//!   matching orbital set, `TOL_E` Ha (and the RHF energy again);
//! * JK sets also in their own role: RHF with J AND K density-fitted
//!   (`df_j_aux` = `df_k_aux` = the set) vs PySCF `density_fit`, `TOL_E` Ha.
//!   aug-cc-pVXZ-JKFIT is not in BSE: its PySCF side is PySCF's own copy,
//!   proved bit-identical to ferric's by `scripts/import_psi4_jkfit.py`, so
//!   for those three sets this file checks parsing/normalization and the
//!   import script checks the digits;
//! * every case: basis-function count equal to PySCF's (catches a 5D/6D or
//!   7F/10F mix-up independently of the energy), and the libint overlap
//!   diagonal equal to 1 (Cartesian l >= 2 sets: the axial components only,
//!   since libint and PySCF normalize Cartesian shells on x^l).
//!
//! `basis_coverage_vs_pyscf::every_registered_set_has_a_case` makes the parametrization exhaustive: a
//! set added to `basis::REGISTRY` without a reference case fails here.
//!
//! MUTATIONS (run 2026-10-29 with BASIS_COVERAGE_ONLY; each failed as shown):
//! * `6-31g_d.json`, first O coefficient x1.01 -> `orbital_sets_match_pyscf_rhf`
//!   6-31g*/h2o off by 4.3e-6 Ha;
//! * `def2-svp-rifit.json`, a mid O p-shell exponent x1.01 ->
//!   `auxiliary_sets_match_pyscf_rimp2` off by 1.6e-6 Ha;
//! * `aug-cc-pvtz-jkfit.json`, a mid O p-shell exponent x1.01 ->
//!   `jk_sets_match_pyscf_density_fitted_rhf` off by 1.1e-7 Ha;
//! * the same x1.01 on the TIGHTEST O s exponent of either aux set PASSED
//!   (fitting error < 1e-8 Ha): this bar catches errors in the functions that
//!   carry weight, not every digit. Digit-level fidelity is the job of
//!   `scripts/fetch_bse_bases.py --check` (numerical equality with BSE) and,
//!   for the aug-JKFIT sets, `scripts/import_psi4_jkfit.py`;
//! * `renormalize_contraction` made a no-op -> ferric-core's
//!   `every_bundled_contraction_has_unit_self_overlap` (libint renormalizes
//!   internally, so the energies here cannot see it).

#[path = "common/basis_coverage.rs"]
mod common;

#[test]
#[ignore = "validation: Bundled basis sets"]
fn orbital_sets_match_pyscf_rhf() {
    common::run_cases("rhf", None);
}

#[test]
#[ignore = "validation: Bundled basis sets"]
fn auxiliary_sets_match_pyscf_rimp2() {
    common::run_cases("rimp2", None);
}

/// J/K fitting sets in their own role: RI-J + RI-K RHF.
#[test]
#[ignore = "validation: Bundled basis sets"]
fn jk_sets_match_pyscf_density_fitted_rhf() {
    common::run_cases("dfhf", None);
}
