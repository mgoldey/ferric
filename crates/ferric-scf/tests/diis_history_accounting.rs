//! The DIIS-history memory projection must count the buffers that are actually
//! pushed, per SCF variant.
//!
//! Regression cover for a 2026-08-30 fix. `driver::warn_if_diis_history_large`
//! hardcoded `diis_size × 2 × n² × 8` for every variant, on the argument that
//! "UHF's coupled driver keeps separate α/β histories but the *combined* error
//! vector is still one logical DIIS slot per iteration". That counts logical
//! slots; memory is consumed by `n × n` buffers. `Diis::step_pair` pushes
//! `fock_hist_b`/`err_hist_b` on top of `fock_hist`/`err_hist`, so UHF holds
//! four matrices per entry, and an ADIIS/EDIIS run holds an `EnergyDiis`
//! (`fock_hist` + `dens_hist`) alongside the Pulay history for two more.
//!
//! The old figure was thus 2× low for UHF and up to 2× low for RHF with
//! ADIIS/EDIIS — and it was invoked from RHF only, so UHF, the variant it
//! under-counted most, projected nothing at all.

use ferric_scf::diis::{diis_history_bytes, DiisHistoryShape};

/// UHF's `step_pair` fills the β histories too, so it holds twice what the
/// single-spin `step` path does.
#[test]
fn pair_spin_holds_twice_the_single_spin_history() {
    let single = diis_history_bytes(100, 8, DiisHistoryShape::SingleSpin, false);
    let pair = diis_history_bytes(100, 8, DiisHistoryShape::PairSpin, false);
    assert_eq!(pair, 2 * single);
    assert_eq!(single, 8 * 2 * 100 * 100 * 8);
}

/// ADIIS/EDIIS keeps an `EnergyDiis` live ALONGSIDE the Pulay history rather
/// than instead of it — two further matrices per entry, a term the old
/// projection omitted entirely.
#[test]
fn energy_diis_adds_two_matrices_per_entry() {
    let pulay = diis_history_bytes(64, 4, DiisHistoryShape::SingleSpin, false);
    let with_energy = diis_history_bytes(64, 4, DiisHistoryShape::SingleSpin, true);
    assert_eq!(pulay, 4 * 2 * 64 * 64 * 8);
    assert_eq!(with_energy, 4 * 4 * 64 * 64 * 8);
    assert_eq!(with_energy, 2 * pulay);
}

/// The matrix count must match what the step functions actually push, not what
/// `Diis::new` allocates: `new` always creates four `RingHistory`s, but a
/// `RingHistory` is empty until pushed into.
#[test]
fn matrices_per_entry_matches_the_step_functions() {
    assert_eq!(DiisHistoryShape::SingleSpin.matrices_per_entry(), 2);
    assert_eq!(DiisHistoryShape::PairSpin.matrices_per_entry(), 4);
}

/// The worst case — UHF with an energy-DIIS flavor — is three times the old
/// hardcoded projection.
#[test]
fn worst_case_is_three_times_the_old_hardcoded_figure() {
    let n = 2000;
    let old_hardcoded = 8 * 2 * n * n * 8;
    let worst = diis_history_bytes(n, 8, DiisHistoryShape::PairSpin, true);
    assert_eq!(worst, 3 * old_hardcoded);
}

/// A zero subspace must still charge one entry rather than reporting that DIIS
/// is free.
#[test]
fn zero_subspace_charges_one_entry() {
    assert_eq!(
        diis_history_bytes(10, 0, DiisHistoryShape::SingleSpin, false),
        2 * 10 * 10 * 8
    );
}

/// An absurd shape must saturate rather than wrap to a small number that would
/// make an over-budget job look free.
#[test]
fn absurd_shapes_saturate_instead_of_wrapping() {
    assert_eq!(
        diis_history_bytes(usize::MAX, 8, DiisHistoryShape::PairSpin, true),
        usize::MAX
    );
}
