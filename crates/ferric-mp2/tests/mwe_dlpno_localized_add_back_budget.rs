//! MWE: the localized DLPNO-MP2 add-back must stay inside `memory_budget_bytes`
//! and its result must not depend on the budget-derived chunk width or on the
//! ambient thread count.
//!
//! # The defect (CodeRabbit PRRT_kwDOUAC5Oc6v09wY, PR #530)
//!
//! With `add_back_truncation`, `discarded_spin_components` ran
//! `discarded.par_iter()`, each closure building one `(ia|jb)` block
//! (`nvir^2` doubles), so up to `min(W, n_disc)` blocks were live, while the
//! retained `K` and `T` (`2 * n_pairs` blocks) were still in scope. The budget
//! check counted `3 * n_pairs` blocks only, so a tight budget was overrun by
//! however many workers rayon happened to have.
//!
//! # The fix this pins
//!
//! `K`/`T` are dropped before the add-back, and the add-back runs in chunks of
//! `cap / block` pairs (budget-derived, never thread-derived). Each chunk's
//! per-pair terms are collected and folded SERIALLY in pair order, so the
//! result is bit-identical to the old single ordered left fold at every width.
//!
//! Note: the screen always keeps the diagonal pairs, so `n_pairs >= nocc` and
//! the retained check (`(3 n_pairs + 2)` blocks) already admits >= 1 block for
//! the add-back. The below-one-block refusal is therefore not reachable from
//! the public API; it is pinned by the unit test in `dlpno_localized.rs`.
//!
//! # Hypotheses
//!
//! * REAL: chunking changes only how many blocks are in flight, so every
//!   admissible width gives the same bits.
//! * ARTIFACT: a fold that sums per-chunk subtotals (a reassociation) or a
//!   width read from `current_num_threads()` would change bits between widths /
//!   pools — this is what the tests below are built to see.
//!
//! # Mutation ledger (each applied to `dlpno_localized.rs`, run, reverted)
//!
//! * per-chunk subtotal fold (reassociation) -> `..._across_budget_widths` fails
//! * admission charges `current_num_threads()` blocks -> all three add-back
//!   tests here fail
//! * chunk width capped at `current_num_threads()` -> bits unchanged (ordered
//!   fold, by design); caught by the src unit test `add_back_chunk_is_budget_derived_..`
//! * below-one-block guard removed -> caught by that src unit test only
//! * energy temporaries not counted -> `working_bytes_counts_the_energy_temporaries`
//! * `jacobi_update` divides by reciprocal multiply, or inherits `k[p]`'s
//!   layout -> src `jacobi_update_in_place_is_bit_identical_to_the_reference`

#[path = "common/dlpno_loc.rs"]
mod dlpno_loc;
use dlpno_loc::{load, LocFixture};

use ferric_mp2::dlpno_localized::{
    dlpno_mp2_localized_from_b_ov, localized_working_bytes, LocalizedMp2Config,
    LocalizedMp2Diagnostics,
};
use ferric_mp2::rimp2::SpinComponents;

/// Ethane: nocc = 9, so 36 off-diagonal pairs can be screened against a
/// retained set of 9 diagonals — `n_disc` exceeds the retained-phase block
/// count, which is what makes more than one add-back chunk reachable.
const ETHANE: &str = "../../testdata/molecules/c2h6.xyz";

fn fixture() -> LocFixture {
    load(ETHANE, "6-31g", "cc-pvdz-ri", false)
}

fn block(fx: &LocFixture) -> usize {
    fx.nvir * fx.nvir * std::mem::size_of::<f64>()
}

/// Everything off-diagonal is screened.
fn cfg(budget: Option<usize>, add_back: bool) -> LocalizedMp2Config {
    LocalizedMp2Config {
        t_cut_pairs: 1e3,
        add_back_truncation: add_back,
        memory_budget_bytes: budget,
        ..LocalizedMp2Config::default()
    }
}

type Run = Result<(SpinComponents, LocalizedMp2Diagnostics), ferric_core::FerricError>;

fn run(fx: &LocFixture, c: &LocalizedMp2Config) -> Run {
    dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, c)
}

fn bits(r: &(SpinComponents, LocalizedMp2Diagnostics)) -> [u64; 4] {
    [
        r.0.e_os.to_bits(),
        r.0.e_ss.to_bits(),
        r.0.e_total.to_bits(),
        r.1.truncation_correction.to_bits(),
    ]
}

fn in_pool<T: Send>(n: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
        .install(f)
}

#[test]
fn add_back_is_bit_identical_across_budget_widths() {
    let fx = fixture();
    let blk = block(&fx);
    let reference = run(&fx, &cfg(None, true)).unwrap();
    let np = reference.1.n_pairs;
    let n_disc = fx.nocc * (fx.nocc + 1) / 2 - np;
    assert_eq!(np, fx.nocc, "premise: only the diagonals are retained");
    assert!(reference.1.truncation_correction != 0.0);
    let need = localized_working_bytes(np, fx.nvir);
    let min_chunk = need / blk;
    eprintln!(
        "ethane: nocc={} nvir={} np={np} n_disc={n_disc} block={blk} B need={need} B \
         (smallest chunk {min_chunk}); bits {:x?}",
        fx.nocc,
        fx.nvir,
        bits(&reference)
    );
    assert!(
        min_chunk < n_disc,
        "premise: the tightest admissible budget must split the add-back into >1 chunk"
    );
    // Smallest admissible budget (several chunks, ragged tail), one block
    // more, and the single-chunk width.
    for cap in [
        need,
        need + blk,
        need + 3 * blk,
        blk * n_disc.max(min_chunk),
    ] {
        let got = run(&fx, &cfg(Some(cap), true)).unwrap();
        assert_eq!(
            bits(&got),
            bits(&reference),
            "add-back bits changed at budget {cap} B (chunk {})",
            cap / blk
        );
    }
}

/// Raising the budget never changes admission of the add-back relative to the
/// plain run: the retained phase is the binding constraint, and the add-back
/// fits in whatever it leaves.
#[test]
fn add_back_admission_matches_the_retained_check() {
    let fx = fixture();
    let blk = block(&fx);
    let np = fx.nocc;
    let need = localized_working_bytes(np, fx.nvir);
    for cap in [need - 1, need, need + blk - 1] {
        let off = run(&fx, &cfg(Some(cap), false)).is_ok();
        let on = run(&fx, &cfg(Some(cap), true)).is_ok();
        assert_eq!(off, cap >= need, "retained check at {cap} B");
        assert_eq!(on, off, "add-back changed admission at {cap} B");
    }
}

#[test]
fn add_back_admission_ignores_thread_count() {
    let fx = fixture();
    let blk = block(&fx);
    let need = localized_working_bytes(fx.nocc, fx.nvir);
    let caps: Vec<Option<usize>> = std::iter::once(None)
        .chain((0..8).map(|k| Some(need - blk + k * blk / 2)))
        .collect();
    let sweep = |n: usize| {
        in_pool(n, || {
            caps.iter()
                .map(|&c| run(&fx, &cfg(c, true)).ok().map(|r| bits(&r)))
                .collect::<Vec<_>>()
        })
    };
    let one = sweep(1);
    let four = sweep(4);
    assert!(one.iter().any(Option::is_none), "premise: some cap refuses");
    assert!(one.iter().any(Option::is_some), "premise: some cap admits");
    assert_eq!(one, four, "admission or bits depend on the thread count");
}

/// `K`, `T`, the Jacobi copy (3 blocks per retained pair) plus the two serial
/// energy-loop temporaries, which exist only when there is a retained pair.
#[test]
fn working_bytes_counts_the_energy_temporaries() {
    let nvir = 21;
    let blk = nvir * nvir * 8;
    assert_eq!(localized_working_bytes(9, nvir), (3 * 9 + 2) * blk);
    assert_eq!(localized_working_bytes(1, nvir), 5 * blk);
    assert_eq!(localized_working_bytes(0, nvir), 0);
}
