//! The pair-parallel SR walks (FINDINGS "Performance plan (research) —
//! 2026-09-25" §3 Class A: RS-GDF SR 3-centre `J3` and the hcore SR nuclear
//! attraction `V_SR`) are BIT-IDENTICAL across rayon thread counts AND to the
//! serial loops they replaced.
//!
//! Construction under test: the shell-pair loop is outermost and runs in
//! parallel; each pair accumulates its own block from zero in exactly the
//! serial per-element order ("L ascending, then T / nucleus candidate in
//! walk order") and COPIES it into the output. What each test guards:
//!
//! * `rsgdf_sr_sums_are_bitwise_across_threads_and_vs_serial` — J3 (and J2)
//!   at 1/2/6 threads agree bit for bit, AND the parallel Gamma J3 equals the
//!   SERIAL `L → i1 → i2 → P → T` walk (the k-point residue-binned walk at one
//!   bin, which still runs the old serial nest over ORDERED pairs) bit for
//!   bit on its `μ ≤ ν` rows, mirrored (the Gamma walk is s2: `rsgdf` module
//!   doc "Orbital-pair symmetry"; J2 is compared as is). A pair-outer nest
//!   that reordered an element's addends (e.g. P-outer over L, or a per-thread
//!   partial sum folded at the end) fails the serial comparison even at one
//!   thread; a shared-accumulation race or a thread-count-dependent split
//!   fails the cross-thread one.
//! * `hcore_sr_attraction_is_bitwise_across_threads_and_vs_serial` — the same
//!   for `V_SR` (s2, mirrored) against a FROZEN copy of the pre-parallel
//!   serial ordered loop, both production-screened and with the
//!   predicted-skip tracking of the SR screening study.
//! * `rsgdf_rhf_energy_is_bitwise_across_threads` — `h`, B and the RS-GDF RHF
//!   energy at 1/2/6 threads agree bit for bit, and the energy keeps the
//!   pinned RS-GDF − dense-AFT fitting error of `pbc_rsgdf.rs`
//!   (−1.974228505452e-06 Ha, cc-pvdz-ri, exxdiv none).
//!
//! * `krsgdf_sr_bins_are_bitwise_across_threads_and_vs_serial` — the
//!   k-point residue-binned SR metric and 3-centre bins (the
//!   `(pair, r_L)`-parallel metric walk and the s2 3-centre walk,
//!   `rsgdf::kpoint` module doc "Orbital-pair symmetry at k") at 1/2/6
//!   threads agree bit for bit, and equal the FROZEN serial ordered binned
//!   walk bit for bit (J2 as is; J3 under the s2 prescription
//!   `common::kpair_s2_expected`, i.e. with the `−r_L` residue map), on a
//!   Monkhorst-Pack mesh whose pair-image moduli (`2N`) differ from the
//!   aux-image moduli (`N`), so a bin mix-up in either dimension fails.
//! * `hcore_kpts_sr_attraction_is_bitwise_across_threads_and_vs_serial` —
//!   `V_SR(k)` (s2: the serial loop's `μ ≤ ν` elements and their conjugates)
//!   against a FROZEN copy of the serial ordered k-point loop, and the full
//!   `h(k)` across thread counts.
//! * `krsgdf_blocks_and_krhf_energy_are_bitwise_across_threads` — every
//!   `B(k, k')` of `KRsGdf`, `h(k)` and the k-point RS-GDF RHF energy at
//!   1/2/6 threads agree bit for bit (1×1×2, all TRIM, and 1×1×3, with a
//!   mirrored q pair); the 1×1×2 energy keeps its PySCF-backed pin of
//!   `pbc_krsgdf.rs` (`TRI_112_JKFIT`, exxdiv none).
//!
//! Reachability: the J3/V systems have ≥ 64 ordered shell pairs (≥ 10 per
//! thread at 6 threads, asserted), so the parallel split really binds.
//!
//! The force / stress DERIVATIVE walks add every triplet into SHARED rows
//! (atom rows, aux rows, the 3×3 stress), so no element belongs to one pair
//! task; they are ordered-parallel instead (`ferric_pbc::ordered`: the
//! integrals and per-triplet from-zero subtotals in parallel over a window
//! of units, every `+=` into a shared row replayed serially in the serial
//! walk order). That is BIT-IDENTICAL to the serial walks, and these tests
//! assert it against the FROZEN pre-parallel walks (the `SerialDerivWalks`
//! / `SerialImages` test mutations), at 1/2/6 threads, with the serial
//! oracle itself run at 1 and 6 threads:
//!
//! * `gamma_rhf_rsgdf_force_…` — SR attraction force visitor, RS-GDF SR
//!   3-centre and SR metric force walks (triclinic 4 H s+p, cc-pvdz-ri).
//! * `gamma_uks_rsgdf_force_…` — the same on an open-shell hybrid (H3 UKS
//!   PBE0).
//! * `gamma_rhf_rsgdf_stress_…` — SR attraction strain visitor, RS-GDF SR
//!   3-centre and metric strain walks.
//! * `kpoint_rhf_rsgdf_force_…` — the L-weighted SR attraction visitor and
//!   the phase-folded k-point SR 3-centre / metric walks (H2 1×1×3).
//! * `kpoint_rhf_rsgdf_stress_…` — the k-point RS-GDF STRESS (phase-folded
//!   SR 3-centre / metric strain walks, per-q LR strain, G = 0 terms, the
//!   one-electron pass), bitwise across threads (H2 1×1×3).
//! * `ecp_force_term_…` — the periodic-ECP image loop (compact H–I,
//!   LANL2DZ), fixed density.
//!
//! A test of this kind only proves anything if the oracle and the parallel
//! path are different code: the oracles are verbatim copies of the pre-
//! parallel loops (`*_serial` functions), selected by the mutation.
//!
//! # Per-commit canaries vs the weekly anchors
//!
//! The anchors above that cost minutes on CI (every test ≥ 15 s, which
//! includes every test ≥ 120 s) run weekly in
//! `validation_pbc_parallel_bitwise.rs`; the bodies are shared through
//! `parallel_bitwise_support/mod.rs`, so a canary and its anchor run the
//! same code. Kept here unchanged (each ≤ ~12 s on CI): the RS-GDF RHF
//! energy, the four pair-FT kernel comparisons, Gamma and k-point S/T, the
//! exchange-grouping tests and the overlap/kinetic force walk. Every moved
//! family keeps a SHRUNK canary (1/2 threads instead of 1/2/6; the serial
//! oracle at 1 and 2 threads; smaller fixtures as named on each test):
//! Gamma SR sums, k SR bins, Gamma and k hcore SR attraction, k blocks +
//! kRHF energy, Gamma RS-GDF force and stress walks, k RS-GDF force and
//! split stress walks, the ECP force term, the LR stage vs the frozen serial
//! stage, the LR G-chunking invariance (issue #507) and the several-LR-chunks
//! B + energy. The Gamma UKS force walk and the overlap/kinetic end-to-end
//! force are covered by the RHF force canary and the overlap/kinetic walk
//! test (the same ordered walks).

mod common;
#[path = "parallel_bitwise_support/mod.rs"]
mod support;

use common::*;
use ferric_pbc::rsgdf::RangeSplit;
use support::*;

/// Canary threads: the cross-thread and the serial comparison at the
/// smallest split that binds.
const CANARY_THREADS: [usize; 2] = [1, 2];

/// Off-axis H2 (Bohr): the canaries' smallest force / stress system.
const H2_OFF_AXIS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

// ------------------------------------------------- kept unchanged (cheap)

#[test]
fn rsgdf_rhf_energy_is_bitwise_across_threads() {
    rsgdf_rhf_energy_is_bitwise_across_threads_body();
}

#[test]
fn pair_ft_kernels_match_frozen_serial_kernels_across_threads() {
    pair_ft_kernels_match_frozen_serial_kernels_across_threads_body();
}

#[test]
fn pair_ft_kernels_match_frozen_serial_kernels_on_the_direct_phase() {
    pair_ft_kernels_match_frozen_serial_kernels_on_the_direct_phase_body();
}

#[test]
fn pair_ft_kernels_match_frozen_serial_kernels_in_partial_windows() {
    pair_ft_kernels_match_frozen_serial_kernels_in_partial_windows_body();
}

#[test]
fn pair_ft_kernels_match_frozen_serial_kernels_with_general_contraction() {
    pair_ft_kernels_match_frozen_serial_kernels_with_general_contraction_body();
}

#[test]
fn hcore_overlap_kinetic_is_bitwise_across_threads_and_vs_serial() {
    hcore_overlap_kinetic_is_bitwise_across_threads_and_vs_serial_body();
}

#[test]
fn hcore_kpts_overlap_kinetic_is_bitwise_across_threads_and_vs_serial() {
    hcore_kpts_overlap_kinetic_is_bitwise_across_threads_and_vs_serial_body();
}

#[test]
fn exchange_aux_groups_are_fixed_contiguous_and_cover_every_row() {
    exchange_aux_groups_are_fixed_contiguous_and_cover_every_row_body();
}

#[test]
fn rsgdf_exchange_is_bitwise_across_threads_and_within_rounding_of_serial() {
    rsgdf_exchange_is_bitwise_across_threads_and_within_rounding_of_serial_body();
}

#[test]
fn krsgdf_exchange_is_bitwise_across_threads_and_within_rounding_of_serial() {
    krsgdf_exchange_is_bitwise_across_threads_and_within_rounding_of_serial_body();
}

#[test]
fn rsgdf_k_builder_is_the_grouped_exchange_on_a_real_b() {
    rsgdf_k_builder_is_the_grouped_exchange_on_a_real_b_body();
}

#[test]
fn overlap_kinetic_force_walk_is_bitwise_across_threads_and_vs_serial_loop() {
    overlap_kinetic_force_walk_is_bitwise_across_threads_and_vs_serial_loop_body();
}

// ------------------------------------- canaries of the weekly anchors

/// Canary of `rsgdf_sr_sums_are_bitwise_across_threads_and_vs_serial`:
/// triclinic 4 H in STO-3G (4 shells) with the ET s+p aux, 1/2 threads.
#[test]
fn rsgdf_sr_sums_bitwise_canary() {
    rsgdf_sr_sums_case(&pyscf_sto3g_h(), &et_sp_aux(), &CANARY_THREADS);
}

/// Canary of `hcore_sr_attraction_is_bitwise_across_threads_and_vs_serial`:
/// triclinic 4 H in STO-3G, both screening cases, 1/2 threads.
#[test]
fn hcore_sr_attraction_bitwise_canary() {
    hcore_sr_attraction_case(&pyscf_sto3g_h(), &CANARY_THREADS);
}

/// Canary of `hcore_kpts_sr_attraction_is_bitwise_across_threads_and_vs_serial`:
/// triclinic 4 H in STO-3G on MP 1×2×2, 1/2 threads.
#[test]
fn hcore_kpts_sr_attraction_bitwise_canary() {
    hcore_kpts_sr_attraction_case(&pyscf_sto3g_h(), &CANARY_THREADS);
}

/// Canary of `krsgdf_sr_bins_are_bitwise_across_threads_and_vs_serial`:
/// triclinic 4 H in STO-3G, ET s+p aux, MP 1×2×2, 1/2 threads.
#[test]
fn krsgdf_sr_bins_bitwise_canary() {
    krsgdf_sr_bins_case(&pyscf_sto3g_h(), &et_sp_aux(), &CANARY_THREADS);
}

/// Canary of `krsgdf_blocks_and_krhf_energy_are_bitwise_across_threads`:
/// H2 STO-3G (a = 4), ET s+p aux, 1×1×3 only (complex q phases), 1/2
/// threads, no pin (the Gamma pin is `rsgdf_rhf_energy_…` above).
#[test]
fn krsgdf_blocks_and_krhf_energy_bitwise_canary() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = prep_for(&cell, &et_sp_aux());
    krsgdf_blocks_and_krhf_energy_case(&cell, &prep, &aux, &[[1, 1, 3]], &CANARY_THREADS, None);
}

/// Canary of the Gamma RS-GDF FORCE walks (`gamma_rhf_rsgdf_force_…`,
/// `gamma_uks_rsgdf_force_…`): off-axis H2 STO-3G in a = 4, cc-pvdz-ri aux,
/// 1/2 threads, serial oracle at 1 and 2.
#[test]
fn gamma_rhf_rsgdf_force_bitwise_canary() {
    let sys = gdf_sys(cell_of(&H2_OFF_AXIS, cubic(4.0), 1), &pyscf_sto3g_h());
    gamma_rhf_rsgdf_force_case(&sys, &CANARY_THREADS);
}

/// Canary of `gamma_rhf_rsgdf_stress_is_bitwise_across_threads_and_vs_serial_walks`:
/// the same H2 system.
#[test]
fn gamma_rhf_rsgdf_stress_bitwise_canary() {
    let sys = gdf_sys(cell_of(&H2_OFF_AXIS, cubic(4.0), 1), &pyscf_sto3g_h());
    gamma_rhf_rsgdf_stress_case(&sys, &CANARY_THREADS);
}

/// Canary of `kpoint_rhf_rsgdf_force_is_bitwise_across_threads_and_vs_serial_walks`:
/// 1×1×2 instead of 1×1×3, 1/2 threads.
#[test]
fn kpoint_rhf_rsgdf_force_bitwise_canary() {
    kpoint_rhf_rsgdf_force_case([1, 1, 2], &CANARY_THREADS);
}

/// Canary of the k-point RS-GDF STRESS (`kpoint_rhf_rsgdf_stress_…` and
/// `kpoint_rhf_split_rsgdf_stress_…`): the range-split one (it runs the
/// unsplit kept-call walks too), 1/2 threads.
#[test]
fn kpoint_rhf_split_rsgdf_stress_bitwise_canary() {
    kpoint_rhf_rsgdf_stress_bitwise(Some(RangeSplit::new(1.0)), &CANARY_THREADS);
}

/// Canary of `ecp_force_term_is_bitwise_across_threads_and_vs_serial_loop`:
/// 1/2 threads.
#[test]
fn ecp_force_term_bitwise_canary() {
    ecp_force_term_case(&CANARY_THREADS);
}

/// Canary of `rsgdf_lr_sums_match_frozen_serial_across_threads`: STO-3G s +
/// p instead of s + p + pure d (so the row-blocked J3 GEMM does not split),
/// unsplit and split, 1/2 threads.
#[test]
fn rsgdf_lr_sums_match_frozen_serial_canary() {
    rsgdf_lr_sums_case(
        &sp_basis_h(),
        1,
        &[None, Some(RangeSplit::default())],
        &CANARY_THREADS,
        false,
    );
}

/// Canary of `rsgdf_lr_sums_are_invariant_to_the_g_chunking` (issue #507):
/// STO-3G s + p instead of s + p + pure d.
#[test]
fn rsgdf_lr_g_chunking_canary() {
    rsgdf_lr_g_chunking_case(&sp_basis_h(), 1);
}

/// Canary of `rsgdf_b_and_energy_are_bitwise_across_threads_with_several_lr_chunks`:
/// STO-3G s + p, unsplit, 1/2 threads.
#[test]
fn rsgdf_b_and_energy_several_lr_chunks_canary() {
    rsgdf_b_and_energy_case(&sp_basis_h(), 1, &[None], &CANARY_THREADS);
}
