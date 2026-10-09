//! Ordered-window parallelism for the force / stress derivative walks.
//!
//! # Why the energy trick does not apply
//!
//! The energy SR sums are parallel over shell pairs because every output
//! element belongs to ONE pair (`J3[μν, P]`, `V[μν]`), so a pair task can sum
//! its block from zero in the serial order and copy it. A derivative walk
//! adds every triplet into SHARED rows (`natoms × 3` forces, `naux × 3` aux
//! forces, the 3×3 stress): row `A` receives addends from every pair touching
//! an atom on `A`, interleaved in the serial `L → i1 → i2 → …` order. A
//! per-pair (or per-thread) subtotal re-associates that sum, so it is
//! deterministic at best, never equal to the serial bits.
//!
//! # What is done instead: parallel evaluate, serial apply
//!
//! Every walk is a fixed sequence of units (e.g. `(L, i1, i2)` ascending) and
//! each unit's work splits into
//!
//! * an EXPENSIVE, PURE part — the screen and the derivative integrals, plus
//!   any per-triplet subtotal the serial code already forms from zero (e.g.
//!   `ga`, `gb`, `gpx` of the RS-GDF contraction, the stress addend
//!   `ga·ra + gb·rb`) — a function of the unit and read-only inputs only;
//! * a CHEAP accumulation into the shared outputs.
//!
//! [`ordered_units`] evaluates the pure part of a contiguous WINDOW of units
//! in parallel (`collect` keeps unit order) and then runs the accumulation of
//! every unit of the window serially, in unit order, with EXACTLY the serial
//! code's scalar sequence. Every output element therefore receives the same
//! addends in the same order as the serial walk: the result is
//! BIT-IDENTICAL to the serial loop at any thread count and any window size
//! (the window only bounds memory; it never changes a bit, so it may adapt).
//! The only assumption is that the pure part really is pure: every libint
//! `compute_*_shifted` call copies and moves its shells per call (stateless),
//! and a per-thread engine ([`EnginePool`](ferric_integrals::engine_pool::EnginePool))
//! is built identically to the serial one.
//!
//! What is NOT allowed in the pure part: anything that turns several of the
//! serial code's `+=` into one (a per-unit subtotal of a quantity the serial
//! code adds to a shared element per triplet or per basis function). Where
//! the serial accumulation is per element (the SR attraction force visitor,
//! the metric force `metric[p] += v; metric[q] −= v`), the raw blocks are
//! stored and the serial body is replayed unchanged.
//!
//! Memory: one window of stored unit results, bounded by `budget` bytes
//! (sized from the previous window's measured bytes per unit; the first
//! window from the caller's per-unit hint, or [`FIRST_WINDOW`] units). Amdahl: the serial part is the accumulation alone (a few flops per
//! stored scalar), not the integrals or the screen.

use ferric_core::FerricError;
use rayon::prelude::*;

/// Units in the first window when the caller has no size hint (before any
/// result size has been measured).
pub(crate) const FIRST_WINDOW: usize = 64;
/// Largest window, in units.
const MAX_WINDOW: usize = 1 << 16;
/// Default byte budget of one window of stored results.
pub(crate) const WINDOW_BYTES: usize = 64 << 20;

/// Heap + inline bytes a stored unit result occupies (the window budget).
pub(crate) trait Stored {
    /// Heap plus inline bytes of this stored value.
    fn stored_bytes(&self) -> usize;
}

impl Stored for Vec<f64> {
    /// Inline size plus 8 bytes per element.
    fn stored_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + 8 * self.len()
    }
}

impl<const N: usize> Stored for [[f64; 3]; N] {
    /// Fixed inline size (no heap).
    fn stored_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// The window budget of a walk: `min(remaining, WINDOW_BYTES)`. Transient
/// (freed per window), so it is not reserved; a window always holds at
/// least one unit.
pub(crate) fn window_budget(remaining: usize) -> usize {
    remaining.min(WINDOW_BYTES)
}

/// Units `0..n_units`: `compute(u)` (pure, parallel, per window) then
/// `apply(u, value)` (serial, in unit order). `hint`: expected stored bytes
/// per unit (0 = unknown) for the first window. Returns the first error in
/// unit order (units before it are applied, as in the serial loop; the
/// caller discards its outputs on error either way). See the module doc for
/// why this is bit-identical to `for u in 0..n { apply(u, compute(u)?)? }`.
pub(crate) fn ordered_units<V, C, A>(
    n_units: usize,
    budget: usize,
    hint: usize,
    compute: C,
    mut apply: A,
) -> Result<(), FerricError>
where
    V: Send + Stored,
    C: Fn(usize) -> Result<V, FerricError> + Sync,
    A: FnMut(usize, V) -> Result<(), FerricError>,
{
    let mut start = 0usize;
    let mut width = budget
        .checked_div(hint)
        .map_or(FIRST_WINDOW, |w| w.clamp(1, MAX_WINDOW));
    while start < n_units {
        let end = start.saturating_add(width).min(n_units);
        let vals: Vec<Result<V, FerricError>> =
            (start..end).into_par_iter().map(&compute).collect();
        let mut bytes = 0usize;
        for (u, v) in (start..end).zip(vals) {
            let v = v?;
            bytes = bytes.saturating_add(v.stored_bytes());
            apply(u, v)?;
        }
        width = next_width(budget, bytes, end - start);
        start = end;
    }
    Ok(())
}

/// The next window from the last one's measured bytes per unit.
fn next_width(budget: usize, bytes: usize, units: usize) -> usize {
    let per_unit = (bytes / units.max(1)).max(1);
    (budget / per_unit).clamp(1, MAX_WINDOW)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A running sum whose bits depend on the addend order: the ordered
    /// replay must give the serial fold at any thread count and window.
    #[test]
    fn ordered_units_reproduce_the_serial_fold() {
        let n = 5000usize;
        let term =
            |u: usize| -> f64 { ((u as f64) * 0.7311).sin() * 10f64.powi((u % 17) as i32 - 8) };
        let mut serial = 0.0_f64;
        for u in 0..n {
            serial += term(u);
        }
        for threads in [1usize, 2, 6] {
            for budget in [1usize, 1 << 10, 1 << 30] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let got = pool.install(|| {
                    let mut acc = 0.0_f64;
                    let mut order = Vec::new();
                    ordered_units(
                        n,
                        budget,
                        0,
                        |u| Ok(vec![term(u)]),
                        |u, v: Vec<f64>| {
                            order.push(u);
                            acc += v[0];
                            Ok(())
                        },
                    )
                    .unwrap();
                    assert!(order.iter().copied().eq(0..n), "apply order");
                    acc
                });
                assert_eq!(
                    got.to_bits(),
                    serial.to_bits(),
                    "{threads} threads, budget {budget}"
                );
            }
        }
    }

    /// The first error in unit order is returned, and nothing after it is
    /// applied.
    #[test]
    fn ordered_units_return_the_first_error_in_order() {
        let mut applied = Vec::new();
        let r = ordered_units(
            200,
            1 << 20,
            0,
            |u| {
                if u == 70 || u == 150 {
                    Err(FerricError::General(format!("unit {u}")))
                } else {
                    Ok(vec![u as f64])
                }
            },
            |u, _v: Vec<f64>| {
                applied.push(u);
                Ok(())
            },
        );
        match r {
            Err(FerricError::General(m)) => assert_eq!(m, "unit 70"),
            other => panic!("expected the unit-70 error, got {other:?}"),
        }
        assert!(applied.iter().copied().eq(0..70));
    }
}
