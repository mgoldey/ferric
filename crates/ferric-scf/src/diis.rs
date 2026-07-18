//! DIIS (Direct Inversion in the Iterative Subspace) convergence accelerator.
//!
//! Implements Pulay's DIIS extrapolation for accelerating SCF convergence.
//! The error vectors (FDS - SDF) are used to construct a least-squares
//! extrapolation of the Fock matrix.
//!
//! ## Incremental B-matrix bookkeeping
//!
//! The B-matrix entry `B[i][j] = <err_i, err_j>` only changes for the row/column
//! of a newly-pushed vector; all other entries are unchanged from the previous
//! iteration. History is kept in a ring buffer addressed by *physical slot*
//! (`0..max_subspace`); when a new vector overwrites the oldest logical entry,
//! only that physical slot's cached inner products against all other live slots
//! need to be recomputed, and the rest of the cache is reused verbatim.
//! `remove(0)` array shifting is replaced by index arithmetic.

use ndarray::Array2;

/// Fixed-capacity ring buffer of history vectors, addressed by physical slot.
///
/// `push` overwrites the oldest logical entry once at capacity. `logical_order()`
/// yields physical slot indices in oldest-to-newest order, matching the order the
/// old `Vec` + `remove(0)` bookkeeping produced.
struct RingHistory<T> {
    capacity: usize,
    slots: Vec<Option<T>>,
    /// Physical slot the *next* push will write to.
    next_slot: usize,
    /// Total number of pushes ever made (monotonic).
    total_pushed: usize,
}

impl<T> RingHistory<T> {
    fn new(capacity: usize) -> Self {
        RingHistory {
            capacity,
            slots: (0..capacity).map(|_| None).collect(),
            next_slot: 0,
            total_pushed: 0,
        }
    }

    fn clear(&mut self) {
        for s in self.slots.iter_mut() {
            *s = None;
        }
        self.next_slot = 0;
        self.total_pushed = 0;
    }

    fn len(&self) -> usize {
        self.total_pushed.min(self.capacity)
    }

    /// Push a new value, evicting the oldest if at capacity.
    /// Returns the physical slot index the value was written to.
    fn push(&mut self, value: T) -> usize {
        let slot = self.next_slot;
        self.slots[slot] = Some(value);
        self.next_slot = (self.next_slot + 1) % self.capacity;
        self.total_pushed += 1;
        slot
    }

    /// Physical slot indices of currently-live entries, oldest first.
    fn logical_order(&self) -> Vec<usize> {
        let m = self.len();
        if self.total_pushed <= self.capacity {
            // No wraparound yet: slots 0..m were filled in order.
            (0..m).collect()
        } else {
            // Wrapped: oldest is next_slot, then wrap around to next_slot-1.
            (0..m).map(|k| (self.next_slot + k) % self.capacity).collect()
        }
    }

    fn get(&self, slot: usize) -> &T {
        self.slots[slot].as_ref().expect("slot must be occupied")
    }

    fn last(&self) -> &T {
        let last_slot = (self.next_slot + self.capacity - 1) % self.capacity;
        self.get(last_slot)
    }
}

/// Cache of pairwise inner products among live history entries, addressed by
/// physical slot index (stable across evictions — only an overwritten slot's
/// row/column goes stale).
struct GramCache {
    capacity: usize,
    /// `values[i * capacity + j]` = <vec_i, vec_j> for physical slots i, j.
    /// Only entries among currently-live slots are meaningful.
    values: Vec<f64>,
}

impl GramCache {
    fn new(capacity: usize) -> Self {
        GramCache {
            capacity,
            values: vec![0.0; capacity * capacity],
        }
    }

    fn clear(&mut self) {
        self.values.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Recompute the row/column for `new_slot` against all currently-live slots
    /// (including itself), using `vecs` to look up stored vectors by slot.
    fn update_slot<T>(&mut self, new_slot: usize, live_slots: &[usize], vecs: &RingHistory<T>, dot_fn: impl Fn(&T, &T) -> f64) {
        let cap = self.capacity;
        let new_vec = vecs.get(new_slot);
        for &other in live_slots {
            let other_vec = vecs.get(other);
            let d = dot_fn(new_vec, other_vec);
            self.values[new_slot * cap + other] = d;
            self.values[other * cap + new_slot] = d;
        }
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        self.values[i * self.capacity + j]
    }
}

/// DIIS extrapolator with a fixed-size rolling subspace.
pub struct Diis {
    fock_hist: RingHistory<Array2<f64>>,
    err_hist: RingHistory<Array2<f64>>,
    /// Optional β-spin history used by `step_pair` for coupled UHF DIIS.
    /// Kept parallel to `fock_hist`/`err_hist` when in pair mode.
    fock_hist_b: RingHistory<Array2<f64>>,
    err_hist_b: RingHistory<Array2<f64>>,
    /// Cached <err_i, err_j> Gram matrix (α-only for `step`; α-contribution
    /// for `step_pair`), addressed by physical slot.
    gram: GramCache,
    /// Cached <err_b_i, err_b_j> Gram matrix, used only by `step_pair`.
    gram_b: GramCache,
}

impl Diis {
    /// Create a DIIS accelerator with the given maximum subspace size.
    pub fn new(max_subspace: usize) -> Self {
        let cap = max_subspace.max(1);
        Diis {
            fock_hist: RingHistory::new(cap),
            err_hist: RingHistory::new(cap),
            fock_hist_b: RingHistory::new(cap),
            err_hist_b: RingHistory::new(cap),
            gram: GramCache::new(cap),
            gram_b: GramCache::new(cap),
        }
    }

    /// Clear the DIIS history.
    pub fn reset(&mut self) {
        self.fock_hist.clear();
        self.err_hist.clear();
        self.fock_hist_b.clear();
        self.err_hist_b.clear();
        self.gram.clear();
        self.gram_b.clear();
    }

    /// Add a Fock matrix and error vector to the history, return the extrapolated Fock matrix.
    pub fn step(&mut self, f: &Array2<f64>, err: &Array2<f64>) -> Array2<f64> {
        self.fock_hist.push(f.clone());
        let new_slot = self.err_hist.push(err.clone());
        let live = self.err_hist.logical_order();
        self.gram.update_slot(new_slot, &live, &self.err_hist, dot);

        let m = live.len();
        if m < 2 {
            return f.clone();
        }
        let dim = m + 1;
        let mut a = vec![0.0f64; dim * dim];
        let mut rhs = vec![0.0f64; dim];
        for (i, &si) in live.iter().enumerate() {
            for (j, &sj) in live.iter().enumerate() {
                a[i * dim + j] = self.gram.get(si, sj);
            }
            a[i * dim + m] = 1.0;
            a[m * dim + i] = 1.0;
        }
        rhs[m] = 1.0;
        let c = match solve_linear(a, rhs, dim) {
            Some(c) => c,
            None => return self.fock_hist.last().clone(),
        };
        let shape = f.dim();
        let mut out = Array2::zeros(shape);
        for (i, &si) in live.iter().enumerate() {
            out.scaled_add(c[i], self.fock_hist.get(si));
        }
        out
    }

    /// Coupled UHF DIIS step. Stores (F_α, F_β, err_α, err_β) pairs and
    /// computes a single coefficient vector by minimizing the *joint* error
    /// norm ‖Σ c_i err_α^i‖² + ‖Σ c_i err_β^i‖². Same coefficients applied
    /// to both spin blocks so α and β stay synchronized.
    ///
    /// On the first call (no history), returns the inputs unchanged. The
    /// inner B-matrix is the sum of per-spin err inner products, equivalent
    /// to block-diagonal err vectors.
    pub fn step_pair(
        &mut self,
        f_a: &Array2<f64>, f_b: &Array2<f64>,
        err_a: &Array2<f64>, err_b: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        // Reuse fock_hist/err_hist for α; keep β in parallel ring buffers.
        self.fock_hist.push(f_a.clone());
        let new_slot = self.err_hist.push(err_a.clone());
        self.fock_hist_b.push(f_b.clone());
        let new_slot_b = self.err_hist_b.push(err_b.clone());
        debug_assert_eq!(new_slot, new_slot_b, "α/β ring buffers must stay in lockstep");

        let live = self.err_hist.logical_order();
        self.gram.update_slot(new_slot, &live, &self.err_hist, dot);
        self.gram_b.update_slot(new_slot_b, &live, &self.err_hist_b, dot);

        let m = live.len();
        if m < 2 {
            return (f_a.clone(), f_b.clone());
        }
        let dim = m + 1;
        let mut a = vec![0.0f64; dim * dim];
        let mut rhs = vec![0.0f64; dim];
        for (i, &si) in live.iter().enumerate() {
            for (j, &sj) in live.iter().enumerate() {
                // Joint inner product = α-block + β-block (block-diagonal err).
                a[i * dim + j] = self.gram.get(si, sj) + self.gram_b.get(si, sj);
            }
            a[i * dim + m] = 1.0;
            a[m * dim + i] = 1.0;
        }
        rhs[m] = 1.0;
        let c = match solve_linear(a, rhs, dim) {
            Some(c) => c,
            None => return (
                self.fock_hist.last().clone(),
                self.fock_hist_b.last().clone(),
            ),
        };
        let mut out_a = Array2::zeros(f_a.dim());
        let mut out_b = Array2::zeros(f_b.dim());
        for (i, &si) in live.iter().enumerate() {
            out_a.scaled_add(c[i], self.fock_hist.get(si));
            out_b.scaled_add(c[i], self.fock_hist_b.get(si));
        }
        (out_a, out_b)
    }
}  // impl Diis

fn dot(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a * b).sum()
}

fn solve_linear(mut a: Vec<f64>, mut x: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut pivot = col;
        let mut max_val = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > max_val {
                max_val = v;
                pivot = r;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            x.swap(col, pivot);
        }
        for r in (col + 1)..n {
            let factor = a[r * n + col] / a[col * n + col];
            for k in col..n {
                a[r * n + k] -= factor * a[col * n + k];
            }
            x[r] -= factor * x[col];
        }
    }
    for r in (0..n).rev() {
        let mut s = x[r];
        for k in (r + 1)..n {
            s -= a[r * n + k] * x[k];
        }
        x[r] = s / a[r * n + r];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diis_one_iteration() {
        let mut d = Diis::new(8);
        let f = Array2::eye(3);
        let err = Array2::zeros((3, 3));
        let out = d.step(&f, &err);
        assert_eq!(out, f);
    }

    #[test]
    fn test_diis_two_iterations() {
        let mut d = Diis::new(8);
        let f1 = Array2::eye(3);
        let err1 = Array2::zeros((3, 3));
        d.step(&f1, &err1);
        let f2 = Array2::eye(3) * 2.0;
        let err2 = Array2::zeros((3, 3));
        let out = d.step(&f2, &err2);
        assert_eq!(out.dim(), (3, 3));
    }

    // --- Reference (from-scratch) bookkeeping, kept only for cross-checking
    // the incremental ring-buffer implementation above. Mirrors the
    // pre-optimization `step`/`step_pair` logic exactly (full B recompute
    // every iteration, `remove(0)` eviction).
    struct RefDiis {
        max_subspace: usize,
        fock_hist: Vec<Array2<f64>>,
        err_hist: Vec<Array2<f64>>,
        fock_hist_b: Vec<Array2<f64>>,
        err_hist_b: Vec<Array2<f64>>,
    }

    impl RefDiis {
        fn new(max_subspace: usize) -> Self {
            RefDiis {
                max_subspace,
                fock_hist: Vec::new(),
                err_hist: Vec::new(),
                fock_hist_b: Vec::new(),
                err_hist_b: Vec::new(),
            }
        }

        fn step(&mut self, f: &Array2<f64>, err: &Array2<f64>) -> (Array2<f64>, Vec<f64>) {
            self.fock_hist.push(f.clone());
            self.err_hist.push(err.clone());
            if self.fock_hist.len() > self.max_subspace {
                self.fock_hist.remove(0);
                self.err_hist.remove(0);
            }
            let m = self.fock_hist.len();
            if m < 2 {
                return (f.clone(), vec![]);
            }
            let dim = m + 1;
            let mut a = vec![0.0f64; dim * dim];
            let mut rhs = vec![0.0f64; dim];
            for i in 0..m {
                for j in 0..m {
                    a[i * dim + j] = dot(&self.err_hist[i], &self.err_hist[j]);
                }
                a[i * dim + m] = 1.0;
                a[m * dim + i] = 1.0;
            }
            rhs[m] = 1.0;
            let b_flat = a.clone();
            let c = match solve_linear(a, rhs, dim) {
                Some(c) => c,
                None => return (self.fock_hist.last().unwrap().clone(), b_flat),
            };
            let shape = f.dim();
            let mut out = Array2::zeros(shape);
            for i in 0..m {
                out.scaled_add(c[i], &self.fock_hist[i]);
            }
            (out, b_flat)
        }

        fn step_pair(
            &mut self,
            f_a: &Array2<f64>, f_b: &Array2<f64>,
            err_a: &Array2<f64>, err_b: &Array2<f64>,
        ) -> ((Array2<f64>, Array2<f64>), Vec<f64>) {
            self.fock_hist.push(f_a.clone());
            self.err_hist.push(err_a.clone());
            self.fock_hist_b.push(f_b.clone());
            self.err_hist_b.push(err_b.clone());
            if self.fock_hist.len() > self.max_subspace {
                self.fock_hist.remove(0);
                self.err_hist.remove(0);
                self.fock_hist_b.remove(0);
                self.err_hist_b.remove(0);
            }
            let m = self.fock_hist.len();
            if m < 2 {
                return ((f_a.clone(), f_b.clone()), vec![]);
            }
            let dim = m + 1;
            let mut a = vec![0.0f64; dim * dim];
            let mut rhs = vec![0.0f64; dim];
            for i in 0..m {
                for j in 0..m {
                    a[i * dim + j] =
                        dot(&self.err_hist[i], &self.err_hist[j])
                        + dot(&self.err_hist_b[i], &self.err_hist_b[j]);
                }
                a[i * dim + m] = 1.0;
                a[m * dim + i] = 1.0;
            }
            rhs[m] = 1.0;
            let b_flat = a.clone();
            let c = match solve_linear(a, rhs, dim) {
                Some(c) => c,
                None => return (
                    (self.fock_hist.last().unwrap().clone(), self.fock_hist_b.last().unwrap().clone()),
                    b_flat,
                ),
            };
            let mut out_a = Array2::zeros(f_a.dim());
            let mut out_b = Array2::zeros(f_b.dim());
            for i in 0..m {
                out_a.scaled_add(c[i], &self.fock_hist[i]);
                out_b.scaled_add(c[i], &self.fock_hist_b[i]);
            }
            ((out_a, out_b), b_flat)
        }
    }

    /// Recover the assembled B-matrix (dim = m+1 upper-left m x m block) from
    /// the new incremental implementation, by re-deriving it the same way
    /// `step` does internally: read back gram entries in logical order.
    fn assembled_b_from_new(d: &Diis) -> Vec<f64> {
        let live = d.err_hist.logical_order();
        let m = live.len();
        if m < 2 {
            return vec![];
        }
        let dim = m + 1;
        let mut a = vec![0.0f64; dim * dim];
        for (i, &si) in live.iter().enumerate() {
            for (j, &sj) in live.iter().enumerate() {
                a[i * dim + j] = d.gram.get(si, sj);
            }
            a[i * dim + m] = 1.0;
            a[m * dim + i] = 1.0;
        }
        a
    }

    fn assembled_b_pair_from_new(d: &Diis) -> Vec<f64> {
        let live = d.err_hist.logical_order();
        let m = live.len();
        if m < 2 {
            return vec![];
        }
        let dim = m + 1;
        let mut a = vec![0.0f64; dim * dim];
        for (i, &si) in live.iter().enumerate() {
            for (j, &sj) in live.iter().enumerate() {
                a[i * dim + j] = d.gram.get(si, sj) + d.gram_b.get(si, sj);
            }
            a[i * dim + m] = 1.0;
            a[m * dim + i] = 1.0;
        }
        a
    }

    /// Deterministic pseudo-random f64 in [-1, 1) from a simple LCG, so the
    /// test has no external RNG dependency.
    fn synthetic(seed: &mut u64) -> f64 {
        // xorshift64
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn synthetic_matrix(seed: &mut u64, n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |_| synthetic(seed))
    }

    #[test]
    fn incremental_b_matches_from_scratch_step() {
        let max_subspace = 5;
        let n_iters = 3 * max_subspace + 2; // exercise multiple full eviction cycles
        let dim = 4;
        let mut seed = 0xDEADBEEFu64;

        let mut d_ref = RefDiis::new(max_subspace);
        let mut d_new = Diis::new(max_subspace);

        for iter in 0..n_iters {
            let f = synthetic_matrix(&mut seed, dim);
            let err = synthetic_matrix(&mut seed, dim);

            let (out_ref, b_ref) = d_ref.step(&f, &err);
            let out_new = d_new.step(&f, &err);
            let b_new = assembled_b_from_new(&d_new);

            assert_eq!(
                b_ref.len(), b_new.len(),
                "B dimension mismatch at iter {iter}"
            );
            for (k, (br, bn)) in b_ref.iter().zip(b_new.iter()).enumerate() {
                assert!(
                    (br - bn).abs() < 1e-15,
                    "B mismatch at iter {iter}, entry {k}: ref={br} new={bn}"
                );
            }
            assert_eq!(out_ref.dim(), out_new.dim());
            for (a, b) in out_ref.iter().zip(out_new.iter()) {
                assert!(
                    (a - b).abs() < 1e-15,
                    "extrapolated F mismatch at iter {iter}: ref={a} new={b}"
                );
            }
        }
    }

    #[test]
    fn incremental_b_matches_from_scratch_step_pair() {
        let max_subspace = 4;
        let n_iters = 3 * max_subspace + 2;
        let dim = 3;
        let mut seed = 0xC0FFEE_u64;

        let mut d_ref = RefDiis::new(max_subspace);
        let mut d_new = Diis::new(max_subspace);

        for iter in 0..n_iters {
            let f_a = synthetic_matrix(&mut seed, dim);
            let f_b = synthetic_matrix(&mut seed, dim);
            let err_a = synthetic_matrix(&mut seed, dim);
            let err_b = synthetic_matrix(&mut seed, dim);

            let ((out_a_ref, out_b_ref), b_ref) = d_ref.step_pair(&f_a, &f_b, &err_a, &err_b);
            let (out_a_new, out_b_new) = d_new.step_pair(&f_a, &f_b, &err_a, &err_b);
            let b_new = assembled_b_pair_from_new(&d_new);

            assert_eq!(b_ref.len(), b_new.len(), "B dimension mismatch at iter {iter}");
            for (k, (br, bn)) in b_ref.iter().zip(b_new.iter()).enumerate() {
                assert!(
                    (br - bn).abs() < 1e-15,
                    "B mismatch at iter {iter}, entry {k}: ref={br} new={bn}"
                );
            }
            for (a, b) in out_a_ref.iter().zip(out_a_new.iter()) {
                assert!((a - b).abs() < 1e-15, "F_a mismatch at iter {iter}");
            }
            for (a, b) in out_b_ref.iter().zip(out_b_new.iter()) {
                assert!((a - b).abs() < 1e-15, "F_b mismatch at iter {iter}");
            }
        }
    }

    #[test]
    fn ring_history_logical_order_matches_vec_remove0() {
        // Sanity-check the ring buffer's oldest-to-newest ordering against a
        // naive Vec + remove(0) model across pushes that both fill and wrap.
        let cap = 3;
        let mut ring = RingHistory::<i32>::new(cap);
        let mut naive: Vec<i32> = Vec::new();

        for v in 0..10 {
            ring.push(v);
            naive.push(v);
            if naive.len() > cap {
                naive.remove(0);
            }
            let order = ring.logical_order();
            let ring_vals: Vec<i32> = order.iter().map(|&s| *ring.get(s)).collect();
            assert_eq!(ring_vals, naive, "mismatch after pushing {v}");
        }
    }

    #[test]
    fn reset_clears_ring_and_gram_cache() {
        let mut d = Diis::new(3);
        let f1 = Array2::eye(2);
        let err1 = Array2::from_elem((2, 2), 0.1);
        d.step(&f1, &err1);
        let f2 = Array2::eye(2) * 2.0;
        let err2 = Array2::from_elem((2, 2), 0.2);
        d.step(&f2, &err2);

        d.reset();
        assert_eq!(d.err_hist.len(), 0);

        // After reset, behaves like a fresh Diis: first step returns f unchanged.
        let f3 = Array2::eye(2) * 3.0;
        let err3 = Array2::zeros((2, 2));
        let out = d.step(&f3, &err3);
        assert_eq!(out, f3);
    }
}
