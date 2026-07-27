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

        if live.len() < 2 {
            return f.clone();
        }
        let Some((c, kept)) = solve_diis_coeffs(&live, |si, sj| self.gram.get(si, sj)) else {
            return self.fock_hist.last().clone();
        };
        let shape = f.dim();
        let mut out = Array2::zeros(shape);
        for (i, &si) in kept.iter().enumerate() {
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

        if live.len() < 2 {
            return (f_a.clone(), f_b.clone());
        }
        // Joint inner product = α-block + β-block (block-diagonal err vector).
        // Same oldest-first shrink-and-retry as `step` — see `solve_diis_coeffs`.
        let solved = solve_diis_coeffs(&live, |si, sj| {
            self.gram.get(si, sj) + self.gram_b.get(si, sj)
        });
        let Some((c, kept)) = solved else {
            return (self.fock_hist.last().clone(), self.fock_hist_b.last().clone());
        };
        let mut out_a = Array2::zeros(f_a.dim());
        let mut out_b = Array2::zeros(f_b.dim());
        for (i, &si) in kept.iter().enumerate() {
            out_a.scaled_add(c[i], self.fock_hist.get(si));
            out_b.scaled_add(c[i], self.fock_hist_b.get(si));
        }
        (out_a, out_b)
    }
}  // impl Diis

fn dot(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a * b).sum()
}

// ===========================================================================
// Energy-based DIIS variants: ADIIS (Hu & Yang, JCP 132, 054109 (2010)) and
// EDIIS (Kudin, Scuseria & Cancès, JCP 116, 8255 (2002)).
//
// Plain Pulay DIIS extrapolates the Fock matrix by minimizing the commutator
// error norm ‖Σ c_i err_i‖² with the SINGLE linear constraint Σ c_i = 1 and no
// sign constraint on c_i. That is cheap and quadratically convergent NEAR the
// solution, but far from convergence (hard cases: transition-metal dimers with
// near-degenerate d-manifolds) the unconstrained extrapolation can produce
// coefficients of either sign that overshoot into a worse point, stalling or
// limit-cycling.
//
// ADIIS/EDIIS instead minimize an *energy*-based functional over the density
// history under the FULL simplex constraint (c_i ≥ 0 AND Σ c_i = 1). The
// extrapolated density Σ c_i D_i is therefore a genuine convex combination of
// previously-seen densities — it can never leave their hull — which is what
// makes these variants robust in the early/far regime. The mature-code recipe
// (PySCF `scf.ADIIS`/`EDIIS`, ORCA, Psi4) is: run ADIIS or EDIIS while the DIIS
// error is large, then switch to (or blend with) plain DIIS once the error is
// small enough that the quadratic model is trustworthy. `DiisDriver` below
// implements exactly that switch.
//
// Both functionals are of the form  f(c) = gᵀc + ½ cᵀ H c  restricted to the
// simplex, differing only in how g (linear) and H (quadratic) are assembled:
//
//   EDIIS:  g_i = E_i
//           H_ij = − Tr[(D_i − D_j)·(F_i − F_j)]        (symmetric, H_ii = 0)
//
//   ADIIS:  reference = the newest pair (D_n, F_n).  With ΔD_i = D_i − D_n,
//           ΔF_j = F_j − F_n:
//           g_i = 2·Tr[ΔD_i · F_n]
//           H_ij = 2·Tr[ΔD_i · ΔF_j]                    (NOT symmetric in general)
//
// Neither H is guaranteed positive-definite, so we do not solve a KKT linear
// system (which can hand back an exterior point). Instead we minimize on the
// simplex directly by the standard PySCF trick: parametrize c_i = t_i² / Σ t_k²,
// which automatically enforces c_i ≥ 0 and Σ c_i = 1 for any real t, turning a
// constrained problem into an unconstrained one in t. We then run a short
// projected/backtracking gradient descent in t. The objective is smooth and the
// simplex is tiny (subspace ≤ diis_size, typically ≤ 8), so a few dozen steps
// converge to machine precision at negligible cost relative to a Fock build.
// ===========================================================================

/// Which family of DIIS coefficients to compute for a given step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiisFlavor {
    /// Plain Pulay commutator-DIIS (least-squares error-norm, unconstrained sign).
    Pulay,
    /// ADIIS energy functional (Hu & Yang) referenced to the newest density.
    Adiis,
    /// EDIIS energy functional (Kudin–Scuseria–Cancès) over the SCF energies.
    Ediis,
}

/// Minimize `f(c) = g·c + ½ cᵀ H c` over the probability simplex
/// (`c_i ≥ 0`, `Σ c_i = 1`) via the `c_i = t_i²/Σt_k²` reparametrization plus
/// backtracking gradient descent in `t`. `h` is row-major `m×m` (need not be
/// symmetric — the ADIIS quadratic isn't). Returns the minimizing convex
/// coefficients. `m` is assumed ≥ 1.
fn minimize_on_simplex(g: &[f64], h: &[f64], m: usize) -> Vec<f64> {
    debug_assert_eq!(g.len(), m);
    debug_assert_eq!(h.len(), m * m);

    // c_i = t_i² / S with S = Σ t_k². Start from the uniform point (all t equal),
    // i.e. c_i = 1/m — an interior point so every coordinate can move.
    let mut t = vec![1.0f64; m];

    // Objective and gradient (w.r.t. t) evaluated at a given t.
    // c = t∘t / S.  f(c) = g·c + ½ cᵀ H c.
    // df/dc_k = g_k + ½ Σ_j (H_kj + H_jk) c_j   (symmetrized H acts on the quad).
    // dc_i/dt_l = (2 t_i/S)(δ_il − c_i)   ⇒   df/dt_l = 2 t_l/S (df/dc_l − Σ_i c_i df/dc_i).
    let eval = |t: &[f64]| -> (f64, Vec<f64>, Vec<f64>) {
        let s: f64 = t.iter().map(|x| x * x).sum::<f64>().max(1e-300);
        let c: Vec<f64> = t.iter().map(|x| x * x / s).collect();
        // f value
        let mut fval = 0.0;
        for i in 0..m {
            fval += g[i] * c[i];
            for j in 0..m {
                fval += 0.5 * h[i * m + j] * c[i] * c[j];
            }
        }
        // df/dc
        let mut dfdc = vec![0.0f64; m];
        for k in 0..m {
            let mut acc = g[k];
            for j in 0..m {
                // ½(H_kj + H_jk) c_j
                acc += 0.5 * (h[k * m + j] + h[j * m + k]) * c[j];
            }
            dfdc[k] = acc;
        }
        // chain rule to t
        let mean_grad: f64 = (0..m).map(|i| c[i] * dfdc[i]).sum();
        let mut dfdt = vec![0.0f64; m];
        for l in 0..m {
            dfdt[l] = 2.0 * t[l] / s * (dfdc[l] - mean_grad);
        }
        (fval, dfdt, c)
    };

    let (mut fval, mut grad, mut c) = eval(&t);
    for _ in 0..500 {
        let gnorm: f64 = grad.iter().map(|x| x * x).sum::<f64>().sqrt();
        if gnorm < 1e-12 {
            break;
        }
        // Backtracking line search along −grad in t-space.
        let mut step = 1.0;
        let mut improved = false;
        for _ in 0..40 {
            let t_try: Vec<f64> = (0..m).map(|i| t[i] - step * grad[i]).collect();
            let (f_try, grad_try, c_try) = eval(&t_try);
            // Armijo-lite: accept any strict decrease (objective is smooth, cheap).
            if f_try < fval - 1e-14 * step * gnorm * gnorm {
                t = t_try;
                fval = f_try;
                grad = grad_try;
                c = c_try;
                improved = true;
                break;
            }
            step *= 0.5;
        }
        if !improved {
            break;
        }
    }
    c
}

/// Energy-based DIIS accelerator holding the density/Fock (and, for EDIIS, the
/// energy) history needed to assemble the ADIIS/EDIIS simplex objective.
///
/// Kept deliberately separate from [`Diis`] (which owns the commutator/Fock
/// history and its incremental Gram-matrix bookkeeping): the energy-based
/// variants need the *density* matrices, which plain DIIS never stores. A step
/// returns the extrapolated **Fock** matrix `Σ c_i F_i` — a drop-in replacement
/// for `Diis::step`'s return value, so the SCF loop diagonalizes it identically.
pub struct EnergyDiis {
    flavor: DiisFlavor,
    fock_hist: RingHistory<Array2<f64>>,
    dens_hist: RingHistory<Array2<f64>>,
    energy_hist: RingHistory<f64>,
}

impl EnergyDiis {
    /// Create an ADIIS or EDIIS accelerator with the given subspace capacity.
    /// Passing `DiisFlavor::Pulay` is a programming error (energy-based history
    /// is meaningless for plain DIIS) and panics — use [`Diis`] for Pulay.
    pub fn new(flavor: DiisFlavor, max_subspace: usize) -> Self {
        assert!(
            flavor != DiisFlavor::Pulay,
            "EnergyDiis is ADIIS/EDIIS only; use Diis for DiisFlavor::Pulay"
        );
        let cap = max_subspace.max(1);
        EnergyDiis {
            flavor,
            fock_hist: RingHistory::new(cap),
            dens_hist: RingHistory::new(cap),
            energy_hist: RingHistory::new(cap),
        }
    }

    /// Clear the history.
    pub fn reset(&mut self) {
        self.fock_hist.clear();
        self.dens_hist.clear();
        self.energy_hist.clear();
    }

    /// Number of live history entries.
    pub fn len(&self) -> usize {
        self.fock_hist.len()
    }

    /// True when no history has been pushed yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push the current `(F, D, E)` and return the energy-DIIS-extrapolated
    /// Fock matrix `Σ c_i F_i`, along with the convex coefficients `c` (for the
    /// caller to blend with a Pulay step if desired). On the first call
    /// (single entry) the coefficients are trivially `[1.0]` and the Fock is
    /// returned unchanged.
    pub fn step(
        &mut self,
        f: &Array2<f64>,
        d: &Array2<f64>,
        energy: f64,
    ) -> (Array2<f64>, Vec<f64>) {
        self.fock_hist.push(f.clone());
        self.dens_hist.push(d.clone());
        self.energy_hist.push(energy);

        let live = self.fock_hist.logical_order();
        let m = live.len();
        if m < 2 {
            return (f.clone(), vec![1.0]);
        }

        let c = self.coefficients(&live);

        let mut out = Array2::zeros(f.dim());
        for (i, &si) in live.iter().enumerate() {
            out.scaled_add(c[i], self.fock_hist.get(si));
        }
        (out, c)
    }

    /// Assemble the ADIIS/EDIIS `(g, H)` objective over the live history slots
    /// and minimize it on the simplex. Public-in-crate for direct unit testing.
    fn coefficients(&self, live: &[usize]) -> Vec<f64> {
        let m = live.len();
        let mut g = vec![0.0f64; m];
        let mut h = vec![0.0f64; m * m];

        match self.flavor {
            DiisFlavor::Ediis => {
                // g_i = E_i ;  H_ij = −Tr[(D_i−D_j)(F_i−F_j)]
                for (i, &si) in live.iter().enumerate() {
                    g[i] = *self.energy_hist.get(si);
                }
                for (i, &si) in live.iter().enumerate() {
                    let di = self.dens_hist.get(si);
                    let fi = self.fock_hist.get(si);
                    for (j, &sj) in live.iter().enumerate() {
                        let dj = self.dens_hist.get(sj);
                        let fj = self.fock_hist.get(sj);
                        // Tr[(D_i−D_j)(F_i−F_j)] = <D_i−D_j, F_i−F_j> (both symmetric).
                        let dd = di - dj;
                        let df = fi - fj;
                        h[i * m + j] = -dot(&dd, &df);
                    }
                }
            }
            DiisFlavor::Adiis => {
                // Reference = newest entry (last in oldest→newest order).
                let &s_n = live.last().unwrap();
                let dn = self.dens_hist.get(s_n);
                let fn_ = self.fock_hist.get(s_n);
                // g_i = 2 Tr[(D_i−D_n) F_n] ;  H_ij = 2 Tr[(D_i−D_n)(F_j−F_n)]
                for (i, &si) in live.iter().enumerate() {
                    let ddi = self.dens_hist.get(si) - dn;
                    g[i] = 2.0 * dot(&ddi, fn_);
                    for (j, &sj) in live.iter().enumerate() {
                        let dfj = self.fock_hist.get(sj) - fn_;
                        h[i * m + j] = 2.0 * dot(&ddi, &dfj);
                    }
                }
            }
            DiisFlavor::Pulay => unreachable!("EnergyDiis never holds Pulay"),
        }

        minimize_on_simplex(&g, &h, m)
    }
}

/// Combined DIIS driver implementing the mature-code recipe: run an energy-based
/// variant (ADIIS or EDIIS) while the DIIS error is large, then switch to plain
/// Pulay DIIS once the commutator error drops below `switch_thresh`.
///
/// The driver owns BOTH a [`Diis`] (for the late/Pulay regime) and an
/// [`EnergyDiis`] (for the early regime), feeding history to both every step so
/// neither has a cold start at the switch point. Each `step` returns the
/// extrapolated Fock matrix the SCF loop should diagonalize — a drop-in for
/// `Diis::step`'s return.
///
/// Set `switch_thresh = 0.0` to disable the energy-based phase entirely (pure
/// Pulay — behaves exactly like calling `Diis::step` directly, so the default
/// SCF path is unperturbed). Set it to `f64::INFINITY` for energy-DIIS-only.
/// PySCF's default crossover is ~1e-1; ORCA switches around the same scale.
pub struct DiisDriver {
    pulay: Diis,
    energy: EnergyDiis,
    /// Commutator err_max below which the driver uses plain Pulay DIIS.
    switch_thresh: f64,
}

impl DiisDriver {
    /// Build a combined driver. `flavor` selects the early-regime variant
    /// (ADIIS or EDIIS); `switch_thresh` is the err_max crossover to Pulay.
    ///
    /// `flavor == Pulay` (or `switch_thresh == 0.0`) yields a driver that is
    /// pure Pulay DIIS — identical to using [`Diis`] directly.
    pub fn new(flavor: DiisFlavor, max_subspace: usize, switch_thresh: f64) -> Self {
        let energy_flavor = if flavor == DiisFlavor::Pulay {
            // Placeholder history holder; never consulted when switch_thresh==0.
            DiisFlavor::Adiis
        } else {
            flavor
        };
        DiisDriver {
            pulay: Diis::new(max_subspace),
            energy: EnergyDiis::new(energy_flavor, max_subspace),
            switch_thresh: if flavor == DiisFlavor::Pulay { 0.0 } else { switch_thresh },
        }
    }

    /// Clear all history.
    pub fn reset(&mut self) {
        self.pulay.reset();
        self.energy.reset();
    }

    /// One combined step. `err_max` is the max-abs commutator error (the SCF
    /// loop already computes it). When `err_max >= switch_thresh` the energy
    /// variant drives; otherwise plain Pulay drives. History is pushed to BOTH
    /// every call so the inactive one is warm at the crossover. Returns the
    /// extrapolated Fock matrix to diagonalize.
    ///
    /// `switch_thresh == f64::INFINITY` selects the energy variant
    /// unconditionally ("energy-DIIS-only", as the struct doc promises). This
    /// is special-cased because the natural `err_max >= switch_thresh` test is
    /// vacuously FALSE against an infinite threshold — a finite `err_max` is
    /// never `>= INFINITY` — which silently made the documented
    /// energy-DIIS-only sentinel run pure Pulay instead.
    pub fn step(
        &mut self,
        f: &Array2<f64>,
        err: &Array2<f64>,
        d: &Array2<f64>,
        energy: f64,
        err_max: f64,
    ) -> Array2<f64> {
        // Always feed Pulay so its Gram history is current at the crossover.
        let f_pulay = self.pulay.step(f, err);

        if self.switch_thresh <= 0.0 {
            // Pure Pulay: don't even touch the energy history (keeps it cheap).
            return f_pulay;
        }

        let (f_energy, _c) = self.energy.step(f, d, energy);

        if self.switch_thresh.is_infinite() || err_max >= self.switch_thresh {
            f_energy
        } else {
            f_pulay
        }
    }
}

/// Solve the bordered DIIS equations for `live` (oldest→newest slots), shrinking
/// the subspace from the OLDEST end until the system is non-singular.
///
/// Returns the coefficients together with the slots they correspond to (a
/// suffix of `live`), or `None` if even the two most recent vectors are
/// linearly dependent — the one case where refusing to extrapolate is right.
///
/// # Why retry instead of giving up
///
/// As an SCF converges its error vectors become genuinely linearly dependent:
/// each is nearly a scalar multiple of the last, so the Gram matrix goes
/// singular no matter how the pivot test is written. The previous code treated
/// that as terminal — one failed solve and `step` returned the un-extrapolated
/// Fock, permanently, because every later iteration pushed another dependent
/// vector onto an already-degenerate history that could never recover.
///
/// MEASURED on water / STO-3G / PBE: the solve first failed at subspace size 7
/// and then failed on all 73 remaining iterations, leaving the SCF running bare
/// Roothaan steps with DIIS dead. `dp_rms` bottomed at 1.8e-8 and drifted
/// *upward* 10× instead of draining past the 1e-8 gate.
///
/// Dropping the oldest vectors is the standard remedy (and is exactly why a
/// small `diis_size` accidentally "fixed" the same run — eviction kept the
/// stale directions out). The newest vectors carry the current search
/// direction, so a suffix is the right thing to keep.
fn solve_diis_coeffs<F>(live: &[usize], gram: F) -> Option<(Vec<f64>, Vec<usize>)>
where
    F: Fn(usize, usize) -> f64,
{
    // Shrink from the oldest end: keep the newest `m` vectors, m = len … 2.
    for m in (2..=live.len()).rev() {
        let kept = &live[live.len() - m..];
        let dim = m + 1;
        let mut a = vec![0.0f64; dim * dim];
        let mut rhs = vec![0.0f64; dim];
        for (i, &si) in kept.iter().enumerate() {
            for (j, &sj) in kept.iter().enumerate() {
                a[i * dim + j] = gram(si, sj);
            }
            a[i * dim + m] = 1.0;
            a[m * dim + i] = 1.0;
        }
        // Constraint row: Σ c_i = 1.
        rhs[m] = 1.0;
        // Normalize the Gram BLOCK ONLY (not the border of 1.0s) to unit scale.
        //
        // Scaling the whole assembled matrix would let the border set the scale:
        // the Gram entries shrink as ‖err‖² on a converging SCF, so against a
        // border fixed at 1.0 they get crushed to relative zero and EVERY
        // subspace looks singular — the same failure this fix exists to remove,
        // just relocated. Normalizing the block alone keeps the two parts of the
        // bordered system commensurate.
        //
        // Dividing the Gram by `g_scale` measures the error vectors in units of
        // the largest one, which leaves the DIIS coefficients unchanged: they
        // are invariant to a common rescaling of all error vectors (the
        // minimizer of ‖Σ c_i err_i‖² under Σ c_i = 1 does not depend on a
        // uniform scale factor).
        let mut g_scale = 0.0f64;
        for i in 0..m {
            for j in 0..m {
                g_scale = g_scale.max(a[i * dim + j].abs());
            }
        }
        if !(g_scale > 0.0) || !g_scale.is_finite() {
            continue;
        }
        let inv = 1.0 / g_scale;
        for i in 0..m {
            for j in 0..m {
                a[i * dim + j] *= inv;
            }
        }
        if let Some(c) = solve_linear(a, rhs, dim) {
            return Some((c, kept.to_vec()));
        }
    }
    None
}

/// Relative pivot tolerance for the DIIS linear solve.
///
/// Applied to a B-matrix pre-scaled so its largest entry is 1 (see
/// [`solve_linear`]), so this is a genuine conditioning test: it rejects a
/// system whose pivots have collapsed *relative to the matrix's own scale*,
/// which is the only scale-free statement of "singular". Sized well above
/// f64 epsilon (2.2e-16) to leave headroom for the elimination's own
/// round-off growth.
const DIIS_PIVOT_RTOL: f64 = 1e-13;

/// Solve `A x = b` by Gaussian elimination with partial pivoting, returning
/// `None` when `A` is numerically singular.
///
/// # Why the pivot test is relative, not absolute
///
/// `A` here is the bordered DIIS B-matrix, whose interior entries are
/// `⟨err_i, err_j⟩` — they scale as ‖err‖². A *converging* SCF therefore drives
/// the whole matrix toward zero: at ‖err‖ ~ 1e-8 the entries are ~1e-16.
///
/// An ABSOLUTE pivot floor (this function previously used `< 1e-14`) turns that
/// benign rescaling into a spurious singularity report. MEASURED on water /
/// STO-3G / PBE: from iteration 7 onward every solve tripped the absolute
/// floor, `step` silently fell back to the un-extrapolated Fock, and the SCF
/// ran 74 further iterations with DIIS effectively dead — `dp_rms` bottomed at
/// 1.8e-8, then drifted *upward* by 10× and never met the 1e-8 gate. Plain HF
/// was unaffected only because its errors stay large enough to clear 1e-14.
///
/// Scaling `A` by its largest-magnitude entry before elimination makes the test
/// scale-free, so it fires on genuine linear dependence among the error vectors
/// (which SHOULD stop extrapolation) and not on mere convergence. Scaling both
/// sides leaves the solution exactly invariant — `(A/s)x = b/s` has the same
/// `x` as `Ax = b` for any `s > 0` — so this changes only which systems are
/// *rejected*, never the coefficients of an accepted one.
///
/// Note that [`solve_diis_coeffs`] additionally normalizes the Gram block on its
/// own — excluding the border of 1.0s, which would otherwise set the scale and
/// crush the Gram entries to relative zero. The normalization here is a
/// general-purpose safeguard for any other caller, and is a no-op on a system
/// that already arrives scaled.
fn solve_linear(mut a: Vec<f64>, mut x: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    // Scale-free conditioning: normalize by the largest entry so the pivot
    // test below measures conditioning rather than absolute magnitude.
    let scale = a.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if !(scale > 0.0) || !scale.is_finite() {
        return None;
    }
    let inv = 1.0 / scale;
    for v in a.iter_mut() {
        *v *= inv;
    }
    for v in x.iter_mut() {
        *v *= inv;
    }

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
        if max_val < DIIS_PIVOT_RTOL {
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

    /// `solve_linear` must be invariant to a uniform rescaling of the system.
    ///
    /// This is the property the old absolute pivot floor violated: `A x = b`
    /// and `(sA) x = (sb)` have the same solution for any `s > 0`, so the
    /// solver must accept or reject them together. Scaling by 1e-10 puts a
    /// well-conditioned system's entries under the old `1e-14` floor, which is
    /// exactly how a converging SCF's B-matrix was mistaken for singular.
    #[test]
    fn solve_linear_is_scale_invariant() {
        let n = 3;
        // Well-conditioned, symmetric-positive-definite-ish system.
        let a = vec![
            4.0, 1.0, 0.5,
            1.0, 3.0, 0.25,
            0.5, 0.25, 2.0,
        ];
        let b = vec![1.0, 2.0, 3.0];

        let base = solve_linear(a.clone(), b.clone(), n).expect("well-conditioned solve");

        for scale in [1e-4, 1e-10, 1e-16, 1e6] {
            let a_s: Vec<f64> = a.iter().map(|v| v * scale).collect();
            let b_s: Vec<f64> = b.iter().map(|v| v * scale).collect();
            let got = solve_linear(a_s, b_s, n)
                .unwrap_or_else(|| panic!("scale {scale:e} spuriously reported singular"));
            for (i, (g, r)) in got.iter().zip(&base).enumerate() {
                assert!(
                    (g - r).abs() <= 1e-9 * r.abs().max(1.0),
                    "scale {scale:e}: x[{i}] = {g} vs unscaled {r}"
                );
            }
        }
    }

    /// A genuinely singular system must still be rejected — the relative test
    /// must not be so permissive that it accepts a rank-deficient B-matrix.
    #[test]
    fn solve_linear_still_rejects_singular() {
        // Row 2 is exactly 2× row 0: rank-deficient at any scale.
        let n = 3;
        let a = vec![
            1.0, 2.0, 3.0,
            0.0, 1.0, 1.0,
            2.0, 4.0, 6.0,
        ];
        let b = vec![1.0, 1.0, 2.0];
        assert!(solve_linear(a.clone(), b.clone(), n).is_none(), "singular system accepted");

        // ...and the same holds after rescaling, in both directions.
        for scale in [1e-12, 1e8] {
            let a_s: Vec<f64> = a.iter().map(|v| v * scale).collect();
            let b_s: Vec<f64> = b.iter().map(|v| v * scale).collect();
            assert!(
                solve_linear(a_s, b_s, n).is_none(),
                "singular system accepted at scale {scale:e}"
            );
        }
    }

    /// When the full subspace is degenerate, `solve_diis_coeffs` must fall back
    /// to a smaller (newest-first) subspace rather than refusing outright.
    ///
    /// Models the converged-SCF situation directly: three "error vectors" whose
    /// Gram matrix is rank-1 in the two oldest directions but informative in the
    /// newest. The old code returned nothing here and killed DIIS permanently.
    #[test]
    fn degenerate_subspace_shrinks_instead_of_failing() {
        // Slots 0,1 carry identical directions (perfectly dependent); slot 2 is
        // independent. Gram entries: <i,j> = v_i · v_j with v_0 = v_1 = (1,0),
        // v_2 = (0,1) — scaled tiny to mimic a converged SCF.
        let s = 1e-9;
        let v = [[s, 0.0], [s, 0.0], [0.0, s]];
        let gram = |i: usize, j: usize| v[i][0] * v[j][0] + v[i][1] * v[j][1];

        let (c, kept) = solve_diis_coeffs(&[0, 1, 2], gram)
            .expect("must fall back to a smaller subspace, not give up");

        assert!(kept.len() >= 2, "kept subspace too small: {kept:?}");
        assert!(kept.len() < 3, "full degenerate subspace should not have solved: {kept:?}");
        // DIIS coefficients over the kept vectors always sum to 1.
        let sum: f64 = c[..kept.len()].iter().sum();
        assert!((sum - 1.0).abs() < 1e-8, "coefficients sum to {sum}, expected 1");
        // The retained slots must be the NEWEST ones (a suffix of the input).
        assert_eq!(kept, vec![1, 2], "must keep the newest vectors");
    }

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

    // ----- ADIIS / EDIIS ----------------------------------------------------

    /// The simplex minimizer must always return a valid convex combination:
    /// all coefficients ≥ 0 and summing to 1, for arbitrary (g, H).
    #[test]
    fn simplex_min_returns_valid_convex_combination() {
        let m = 4;
        // Non-symmetric H with mixed signs, arbitrary g.
        let g = vec![0.3, -1.2, 0.7, 0.1];
        let h = vec![
            0.0, -0.5, 0.2, 0.1,
            0.4, 0.0, -0.3, 0.6,
            -0.1, 0.2, 0.0, -0.4,
            0.3, -0.2, 0.5, 0.0,
        ];
        let c = minimize_on_simplex(&g, &h, m);
        assert_eq!(c.len(), m);
        let sum: f64 = c.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "coeffs must sum to 1, got {sum}");
        for (i, &ci) in c.iter().enumerate() {
            assert!(ci >= -1e-12, "coeff {i} must be ≥ 0, got {ci}");
        }
    }

    /// On a purely-linear objective (H = 0) over the simplex, the minimum is the
    /// vertex at the smallest g_i — the minimizer should put ~all weight there.
    #[test]
    fn simplex_min_linear_picks_smallest_g_vertex() {
        let m = 3;
        let g = vec![1.0, -2.0, 0.5]; // smallest is index 1
        let h = vec![0.0; m * m];
        let c = minimize_on_simplex(&g, &h, m);
        let sum: f64 = c.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(c[1] > 0.99, "weight should collapse onto min-g vertex; c={c:?}");
        assert!(c[0] < 1e-2 && c[2] < 1e-2, "other vertices ~0; c={c:?}");
    }

    /// EDIIS coefficients: build a synthetic history where one point has a much
    /// lower energy and near-zero energy-difference coupling to itself; the
    /// convex EDIIS solution must be a valid simplex point favoring the low-E
    /// point.
    #[test]
    fn ediis_coefficients_favor_low_energy_and_are_convex() {
        let n = 2;
        let mut ed = EnergyDiis::new(DiisFlavor::Ediis, 8);
        // Two distinct (F, D, E) with the SECOND much lower in energy.
        let f0 = Array2::from_shape_fn((n, n), |(i, j)| (i + j) as f64 * 0.1);
        let d0 = Array2::eye(n);
        let f1 = &f0 + &Array2::from_elem((n, n), 0.05);
        let d1 = &d0 * 1.2;
        ed.step(&f0, &d0, -1.0);
        let (_f, c) = ed.step(&f1, &d1, -5.0); // second is much lower
        assert_eq!(c.len(), 2);
        let sum: f64 = c.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "EDIIS coeffs sum to 1, got {sum}");
        assert!(c[0] >= -1e-12 && c[1] >= -1e-12, "EDIIS coeffs ≥ 0: {c:?}");
        assert!(c[1] > c[0], "lower-energy point should carry more weight: {c:?}");
    }

    /// ADIIS coefficients over a synthetic history must also form a valid convex
    /// combination, with the reference (newest) point receiving finite weight.
    #[test]
    fn adiis_coefficients_are_convex() {
        let n = 3;
        let mut ad = EnergyDiis::new(DiisFlavor::Adiis, 8);
        let mut seed = 0x1234_5678u64;
        for _ in 0..4 {
            let f = synthetic_matrix(&mut seed, n);
            let d = synthetic_matrix(&mut seed, n);
            let e = synthetic(&mut seed);
            let (_f, c) = ad.step(&f, &d, e);
            let sum: f64 = c.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "ADIIS coeffs sum to 1, got {sum}");
            for &ci in &c {
                assert!(ci >= -1e-12, "ADIIS coeff ≥ 0: {c:?}");
            }
        }
    }

    /// The extrapolated Fock returned by EnergyDiis::step must equal Σ c_i F_i.
    #[test]
    fn energy_diis_fock_is_convex_combo_of_history() {
        let n = 2;
        let mut ad = EnergyDiis::new(DiisFlavor::Adiis, 8);
        let f0 = Array2::from_elem((n, n), 1.0);
        let f1 = Array2::from_elem((n, n), 3.0);
        let d0 = Array2::eye(n);
        let d1 = &d0 * 2.0;
        ad.step(&f0, &d0, -1.0);
        let (f_out, c) = ad.step(&f1, &d1, -1.5);
        // Reconstruct expected Σ c_i F_i (history order: f0 then f1).
        let expected = c[0] * &f0 + c[1] * &f1;
        for (a, b) in f_out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "Fock ≠ Σ c_i F_i: {a} vs {b}");
        }
    }

    /// DiisDriver with switch_thresh = 0 must be byte-identical to plain Diis —
    /// the default SCF path must be unperturbed by the new driver.
    #[test]
    fn driver_pure_pulay_matches_plain_diis() {
        let max_subspace = 5;
        let n_iters = 3 * max_subspace + 2;
        let dim = 4;
        let mut seed = 0xABCDEFu64;

        let mut plain = Diis::new(max_subspace);
        // flavor=Pulay forces switch_thresh internally to 0.
        let mut driver = DiisDriver::new(DiisFlavor::Pulay, max_subspace, 1e-1);

        for _ in 0..n_iters {
            let f = synthetic_matrix(&mut seed, dim);
            let err = synthetic_matrix(&mut seed, dim);
            let d = synthetic_matrix(&mut seed, dim);
            let energy = synthetic(&mut seed);
            let err_max = err.iter().map(|v| v.abs()).fold(0.0, f64::max);

            let out_plain = plain.step(&f, &err);
            let out_driver = driver.step(&f, &err, &d, energy, err_max);
            for (a, b) in out_plain.iter().zip(out_driver.iter()) {
                assert!((a - b).abs() < 1e-15, "driver(Pulay) ≠ plain DIIS: {a} vs {b}");
            }
        }
    }

    /// Above the switch threshold the driver returns the energy-DIIS Fock; below
    /// it, the plain-Pulay Fock. Verify the crossover picks the right branch.
    #[test]
    fn driver_switches_at_threshold() {
        let dim = 3;
        let mut seed = 0x55AA_55AAu64;
        let mut driver = DiisDriver::new(DiisFlavor::Ediis, 6, 1e-2);

        // Independent references to compare each branch.
        let mut ref_pulay = Diis::new(6);
        let mut ref_energy = EnergyDiis::new(DiisFlavor::Ediis, 6);

        for iter in 0..5 {
            let f = synthetic_matrix(&mut seed, dim);
            let err = synthetic_matrix(&mut seed, dim);
            let d = synthetic_matrix(&mut seed, dim);
            let energy = synthetic(&mut seed);
            // Force above-threshold for first 3 iters, below for the rest.
            let err_max = if iter < 3 { 1.0 } else { 1e-6 };

            let out = driver.step(&f, &err, &d, energy, err_max);
            let ref_p = ref_pulay.step(&f, &err);
            let (ref_e, _c) = ref_energy.step(&f, &d, energy);

            if iter < 2 {
                // First iter both return f unchanged (single history entry);
                // from iter 1 the energy branch is active above threshold.
                continue;
            }
            let expect = if err_max >= 1e-2 { &ref_e } else { &ref_p };
            for (a, b) in out.iter().zip(expect.iter()) {
                assert!((a - b).abs() < 1e-12, "wrong branch at iter {iter}: {a} vs {b}");
            }
        }
    }

    /// `switch_thresh = INFINITY` is documented on `DiisDriver` as selecting the
    /// energy variant only. It previously ran pure Pulay instead: the dispatch
    /// test was `err_max >= switch_thresh`, and no finite `err_max` is ever
    /// `>= INFINITY`, so the energy branch was unreachable. The bug was
    /// invisible to a same-flavor A/B (both arms returned the Pulay matrix and
    /// so agreed bit-for-bit), which is exactly what made it survive.
    #[test]
    fn infinite_switch_thresh_selects_energy_variant_not_pulay() {
        let n = 4;
        let mk = |seed: f64| {
            Array2::from_shape_fn((n, n), |(i, j)| ((i * n + j) as f64 * 0.37 + seed).sin())
        };
        let drive = |flavor: DiisFlavor, thresh: f64| -> Array2<f64> {
            let mut drv = DiisDriver::new(flavor, 8, thresh);
            let mut last = Array2::<f64>::zeros((n, n));
            for it in 0..5 {
                last = drv.step(
                    &mk(it as f64),
                    &mk(it as f64 + 10.0),
                    &mk(it as f64 + 20.0),
                    -100.0 - it as f64,
                    // Well below any finite threshold used here, so only the
                    // INFINITY special-case can route to the energy variant.
                    1.0,
                );
            }
            last
        };

        let pulay = drive(DiisFlavor::Pulay, 1e-1);
        let adiis_inf = drive(DiisFlavor::Adiis, f64::INFINITY);
        let ediis_inf = drive(DiisFlavor::Ediis, f64::INFINITY);

        let maxdiff = |a: &Array2<f64>, b: &Array2<f64>| {
            (a - b).iter().map(|v| v.abs()).fold(0.0f64, f64::max)
        };
        assert!(
            maxdiff(&adiis_inf, &pulay) > 1e-8,
            "ADIIS with switch_thresh=INFINITY must not fall back to Pulay"
        );
        assert!(
            maxdiff(&ediis_inf, &pulay) > 1e-8,
            "EDIIS with switch_thresh=INFINITY must not fall back to Pulay"
        );
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
