//! MWE: the MO-stream chunk width is budget-derived, deterministic, and even.
//!
//! `MO_STREAM_CHUNK` was a hardcoded 256. It now derives from the resolved
//! memory budget — with three properties that make that safe, because the width
//! is NOT numerically inert:
//!
//! The dressing step is
//! `general_mat_mul(1.0, &msub, &mo_blk, 1.0, &mut b_flat)` — **beta = 1** — so
//! the chunk width k-blocks a partial-sum accumulation. Changing it perturbs
//! the last digits (`DRESS_ROW_BLOCK` in `three_index_source.rs` measured ~7e-15
//! from an odd-vs-even split).
//!
//! The three properties, each pinned by a contract below:
//!
//! 1. **Capped at 256**, so an ample budget reproduces the historical numerics
//!    EXACTLY. Only memory-constrained runs narrow the chunk, and only those
//!    see any change at all.
//! 2. **Deterministic** — a pure function of `(width, budget)`, never of the
//!    ambient thread count or free memory. Pin `[memory] budget_gb` and the
//!    numerics are pinned with it.
//! 3. **Even**, because odd splits perturb the GEMM via a 2-wide SIMD
//!    accumulation effect.
//!
//! Arithmetic only — no SCF, no allocation.

use ferric_mp2::rimp2::mo_stream_chunk_for_test as chunk_for;

/// The historical fixed width, and the cap.
const HISTORICAL: usize = 256;
/// The floor: never so narrow that BLAS3 efficiency collapses.
const MIN: usize = 32;

/// A representative MO width (`nleft · nright`) at production scale.
const WIDTH: usize = 150 * 650;

/// CONTRACT 1: an ample budget reproduces the historical width exactly.
///
/// The property that keeps this change safe to land: a user who is not memory
/// constrained sees byte-identical results to before. If this fails, every
/// existing reference value in the repo is in play.
#[test]
fn an_ample_budget_reproduces_the_historical_width() {
    for budget_gb in [8usize, 16, 64, 256] {
        let budget = budget_gb * 1024 * 1024 * 1024;
        assert_eq!(
            chunk_for(WIDTH, budget),
            HISTORICAL,
            "at a {budget_gb} GiB budget the chunk must stay at the historical \
             {HISTORICAL}, so an unconstrained run is bit-identical to before"
        );
    }
}

/// CONTRACT 2: a tight budget narrows the chunk.
///
/// The point of the change. If the width never responds, the budget is
/// decorative — the same defect this whole effort exists to fix.
#[test]
fn a_tight_budget_narrows_the_chunk() {
    // 1 MB against a width of ~97500 doubles/row: one row alone is ~1.5 MB, so
    // this must clamp to the floor.
    let tight = chunk_for(WIDTH, 1024 * 1024);
    assert!(
        tight < HISTORICAL,
        "a 1 MB budget must narrow the chunk below {HISTORICAL}, got {tight}"
    );
    assert!(tight >= MIN, "but never below the {MIN} floor, got {tight}");
}

/// CONTRACT 3: the width is always EVEN.
///
/// Odd block sizes perturb the dressing GEMM (~7e-15) via a 2-wide SIMD
/// accumulation effect. This is the invariant that survives from
/// `DRESS_ROW_BLOCK`, and it must hold at every budget, not just typical ones.
#[test]
fn the_width_is_always_even() {
    for budget in [
        0usize,
        1,
        1024,
        1024 * 1024,
        7_777_777,
        13 * 1024 * 1024,
        1024 * 1024 * 1024,
        99 * 1024 * 1024 * 1024,
    ] {
        let w = chunk_for(WIDTH, budget);
        assert_eq!(
            w % 2,
            0,
            "chunk {w} at budget {budget} must be even — odd splits perturb the \
             dressing GEMM at ~7e-15"
        );
    }
}

/// CONTRACT 4: the width never collapses to zero.
///
/// Bounding memory must degrade to *slow*, never to *stuck*. A zero chunk would
/// make the streaming loop spin forever.
#[test]
fn the_width_never_reaches_zero() {
    for budget in [0usize, 1, 8, 64] {
        let w = chunk_for(WIDTH, budget);
        assert!(
            w >= MIN,
            "a starvation budget must still yield >= {MIN} (slow, not stuck), got {w}"
        );
    }
}

/// CONTRACT 5: the width is deterministic — a pure function of its inputs.
///
/// The property that makes budget-dependent numerics acceptable: same
/// `(width, budget)`, same answer, every time and on every machine. Nothing
/// ambient (thread count, free memory, wall clock) may leak in. Without this,
/// a run would not be reproducible from its config alone.
#[test]
fn the_width_is_deterministic() {
    for budget in [1024usize * 1024, 64 * 1024 * 1024, 4 * 1024 * 1024 * 1024] {
        let first = chunk_for(WIDTH, budget);
        for _ in 0..16 {
            assert_eq!(
                chunk_for(WIDTH, budget),
                first,
                "the chunk width must be a pure function of (width, budget)"
            );
        }
    }
}

/// CONTRACT 6: a wider MO block gets a narrower chunk at a fixed budget.
///
/// The transient is `chunk · width · 8`, so holding the budget fixed and
/// growing `width` must shrink `chunk` — otherwise the bound is not actually
/// tracking bytes.
#[test]
fn a_wider_mo_block_narrows_the_chunk() {
    let budget = 4 * 1024 * 1024; // small enough that neither case hits the cap
    let narrow = chunk_for(1_000, budget);
    let wide = chunk_for(1_000_000, budget);
    assert!(
        wide <= narrow,
        "a 1000x wider MO block must not get a wider chunk: {wide} vs {narrow}"
    );
}
