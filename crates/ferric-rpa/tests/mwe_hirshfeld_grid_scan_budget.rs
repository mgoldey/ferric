//! MWE: `atomic_effective_volumes_hirshfeld` / `hirshfeld_i_charges` must be
//! gated on `chi` AND `d_chi` AND `rho_free`, not on `chi` alone.
//!
//! # The defect
//!
//! Both functions call `ferric_integrals::ao_grid::eval_basis_on_grid` to
//! build `chi`, a `(nbf, npts)` AO-on-grid matrix. That function's own
//! internal gate charges exactly `nbf*npts*8` bytes (`chi` alone) against
//! `resolve_budget_bytes(None)` — it has no way to see what its CALLER does
//! next. Both callers immediately compute `d_chi = density.dot(&chi)` (a
//! second full `(nbf, npts)` block, read in the same loop as `chi`, so
//! genuinely co-resident, not a transient) and then build `rho_free`
//! (`(natoms, npts)`) plus several `O(npts)` side vectors on top. None of
//! that was charged anywhere.
//!
//! `npts` here comes from `GridSpec::bounding_box` — a regular real-space
//! lattice that grows with molecular VOLUME (not atom count, unlike the
//! Becke-Lebedev grid other property paths use), so this is not a small-nbf
//! corner: at nbf=200/npts≈5.6e6/natoms=20 the callee's own gate approves
//! `chi` alone (9.0 GB) while the true peak is ~18.9-19.2 GB, roughly 2x.
//!
//! # What this file tests
//!
//! `estimate_hirshfeld_grid_scan_bytes` (the pure byte estimate) and
//! `preflight_hirshfeld_grid_scan` (the gate `properties.rs` now calls BEFORE
//! `eval_basis_on_grid`, at both call sites) — both are `pub` specifically so
//! this MWE can pin the real production arithmetic without standing up an
//! SCF, a density matrix, or a proatom provider. `nbf_for_basis` (also `pub`)
//! is checked separately against a real molecule + basis to confirm it agrees
//! with what `eval_basis_on_grid`'s `chi.nrows()` would report — the two MUST
//! agree, since the preflight runs before `chi` exists and cannot read its
//! row count directly (that IS the whole point of the fix: gate before the
//! expensive allocation, not after).
//!
//! # What is NOT covered end-to-end
//!
//! There is no affordable fixture that drives `atomic_effective_volumes_hirshfeld`
//! itself up to the multi-GB shape above (that would require actually
//! allocating the multi-GB tensors this fix exists to avoid allocating on a
//! starved budget). The arithmetic is pinned here; the wiring (that the two
//! public functions call this gate before `eval_basis_on_grid`) is pinned by
//! reading the source, not by a test, the same disclosed gap
//! `mwe_open_shell_dynamic_scratch.rs` documents for its own call sites.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_rpa::properties::{estimate_hirshfeld_grid_scan_bytes, nbf_for_basis, preflight_hirshfeld_grid_scan};

/// The audit's incident-adjacent shape: nbf=200, a ~24 Bohr molecule under the
/// default 6-Bohr-margin/0.20-spacing box (npts ≈ 178³), natoms=20.
const NBF: usize = 200;
const NPTS: usize = 178 * 178 * 178;
const NATOMS: usize = 20;
const F64_BYTES: usize = 8;

/// CONTRACT 1 (REACHABILITY): the full estimate is materially larger than
/// `chi` alone — the callee's old (and still-present, for other callers)
/// single-plane charge.
///
/// If this ratio came out at ~1.0x, the fix would be charging nothing new and
/// the test would be measuring a rounding artifact, not the defect.
#[test]
fn full_estimate_is_materially_larger_than_chi_alone() {
    let chi_alone = NBF * NPTS * F64_BYTES;
    let full = estimate_hirshfeld_grid_scan_bytes(NBF, NPTS, NATOMS);
    let ratio = full as f64 / chi_alone as f64;
    assert!(
        ratio > 1.9,
        "expected the full co-resident estimate to be ~2x the chi-alone charge at \
         nbf={NBF}, npts={NPTS}, natoms={NATOMS}; got {:.3}x ({:.2} GB vs {:.2} GB)",
        ratio,
        full as f64 / 1e9,
        chi_alone as f64 / 1e9
    );
}

/// CONTRACT 2: the estimate matches the hand-derived formula exactly
/// (2 planes + rho_free + 5 side vectors, all f64).
#[test]
fn estimate_matches_hand_derived_formula() {
    let expected =
        (2 * NBF * NPTS + NATOMS * NPTS + 5 * NPTS) * F64_BYTES;
    let got = estimate_hirshfeld_grid_scan_bytes(NBF, NPTS, NATOMS);
    assert_eq!(got, expected, "estimate_hirshfeld_grid_scan_bytes({NBF},{NPTS},{NATOMS}) = {got}, expected {expected}");
}

/// CONTRACT 3 (OVER-REJECTION): a trivial shape (water-scale nbf/npts) must
/// not be refused — this preflight must not become a wall for small jobs.
/// `preflight_hirshfeld_grid_scan` resolves the ambient budget internally
/// (matching the pre-existing callee behavior it replaces), so this only
/// asserts the property that must hold on ANY machine capable of running the
/// test suite at all: a kilobyte-scale request must fit.
#[test]
fn a_trivial_shape_is_not_refused() {
    let tiny_nbf = 7; // water/STO-3G
    let tiny_npts = 1000;
    let tiny_natoms = 3;
    let res = preflight_hirshfeld_grid_scan(
        "mwe trivial shape",
        tiny_nbf,
        tiny_npts,
        tiny_natoms,
    );
    assert!(
        res.is_ok(),
        "a kilobyte-scale request must not be refused by any budget this test \
         environment could plausibly have: {res:?}"
    );
}

/// CONTRACT 4: `nbf_for_basis` agrees with the canonical shell-sum
/// (`ferric_integrals::ao_grid::nbasis`, the same function `chi.nrows()`
/// would report) on a real molecule + bundled basis. This is the correctness
/// precondition for gating BEFORE `chi` exists: if `nbf_for_basis`
/// under-counts, the preflight under-charges by exactly the same defect
/// pattern this fix closes.
#[test]
fn nbf_for_basis_matches_the_canonical_shell_sum() {
    let xyz = "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();

    let expected = ferric_integrals::ao_grid::nbasis(&mol, &bs).unwrap();
    let got = nbf_for_basis(&mol, &bs).unwrap();
    assert_eq!(got, expected, "nbf_for_basis disagreed with ao_grid::nbasis");
    // STO-3G water: O gets 1s/2s/2p (5 functions), each H gets 1s (1 each) = 7.
    assert_eq!(got, 7, "STO-3G water should have 7 basis functions");
}

/// CONTRACT 5 (OVER-REJECTION direction, correctness half): a molecule with
/// an element missing from the basis must ERROR, not silently report a
/// smaller-than-true `nbf`. A silent undercount here would defeat the whole
/// point of gating before `chi` exists — it would look like coverage that
/// is not there. (This was a real risk in an earlier draft of this helper,
/// which used `bs.for_element(..).filter_map(..)` and would have dropped
/// such atoms instead of erroring; delegating to `ao_grid::nbasis` avoids it.)
#[test]
fn missing_element_errors_instead_of_undercounting() {
    let xyz = "1\nargon\nAr 0.0 0.0 0.0\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    // sto-3g has no argon shells bundled in this basis file set used elsewhere
    // in this crate's tests for exactly this property; if that ever changes,
    // this test's premise (an element absent from the basis) no longer holds
    // and should be swapped for a basis/element pair that is still missing.
    let bs = basis::bundled("sto-3g").unwrap();
    if bs.for_element(18).is_some() {
        // Premise no longer holds on this basis file; nothing to test.
        return;
    }
    let res = nbf_for_basis(&mol, &bs);
    assert!(
        res.is_err(),
        "an element missing from the basis must error, not silently under-count nbf"
    );
}
