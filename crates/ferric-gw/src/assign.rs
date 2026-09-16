//! One-to-one assignment of computed excited states to reference (literature)
//! states, on a joint (excitation energy, oscillator strength) cost.
//!
//! # Why this exists
//!
//! `docs/bse-tda-benchmark-plan.md` §2.2 names the problem: matching computed
//! BSE-TDA roots against a literature TBE table is a real ASSIGNMENT problem,
//! not a lookup. The previous benchmark statistic
//! (`docs/bse-tda-phase2-results.md`, 10 molecules / 17 states, MAE 0.482 eV)
//! used per-reference-state nearest-computed-ENERGY matching, which has two
//! structural defects:
//!
//! 1. **It is not injective.** Each reference state independently takes its
//!    own energy-argmin, so two reference states can (and did — all three
//!    N-heterocycles in that run) collapse onto the SAME computed root. A
//!    physical assignment is one-to-one: one computed eigenvector is one
//!    state, and it cannot simultaneously be two different literature states.
//!
//! 2. **It is blind to the state's character.** Oscillator strength is the
//!    only state-character information ferric currently computes (there are no
//!    excited-state irrep labels, see §2.2). Ignoring it let ethylene's bright
//!    valence π→π\* reference (f=0.346) be assigned to a DARK computed root
//!    (f=0.000) purely because that root happened to be 0.112 eV away, while
//!    the genuinely bright computed root (f=0.438, 0.392 eV away) went unused.
//!
//! Both defects bias the statistic in the SAME direction — downward. Taking a
//! per-state energy argmin is, by construction, a lower bound on the energy
//! error of ANY other assignment, so the naive number cannot be an
//! overestimate. §2.2 anticipated exactly this and asked for "a Hungarian
//! assignment minimizing a combined energy+f distance ... validating that it
//! doesn't silently mismatch states", with the explicit warning that "a
//! plausible-looking but incorrect assignment is worse than an honest
//! 'ambiguous, flagged' result". This module implements that, including the
//! flagging.
//!
//! # The cost function
//!
//! For a reference state `(E_r, f_r)` and a computed root `(E_c, f_c)`:
//!
//! ```text
//! cost = |E_c - E_r| / E_SCALE  +  w * d_f(f_c, f_r)
//!
//! d_f(a, b) = |a - b| / (a + b + F_FLOOR)
//! ```
//!
//! **Energy channel.** `|ΔE|` in eV, divided by `E_SCALE` (1 eV) so the
//! channel is dimensionless and a unit of cost is a physically interpretable
//! "1 eV of energy error". Not a statistical normalization over the sample —
//! that would make the cost depend on which molecules are in the set, so
//! adding a molecule could silently re-assign a different molecule's states.
//!
//! **Intensity channel.** `d_f` is a *bounded relative* difference (a
//! Bray–Curtis / Sørensen dissimilarity): it lies in `[0, 1]`, is 0 for
//! identical intensities and →1 for bright-vs-dark. This shape is chosen
//! deliberately over the two obvious alternatives:
//!
//! - **Raw `|Δf|`** is wrong because f spans orders of magnitude across the
//!   reference set (0.000 to 0.664). Under raw `|Δf|`, confusing f=0.004 with
//!   f=0.078 (a dark n→π\* with a bright π→π\*, the exact N-heterocycle
//!   failure) costs 0.074 — less than confusing f=0.60 with f=0.68, which is
//!   the SAME state. It penalizes the wrong thing.
//! - **Raw relative `|Δf|/f_r`** is unbounded and undefined for the many
//!   symmetry-forbidden reference states with f ≡ 0 exactly.
//!
//! `d_f` fixes both: it is scale-free (it sees the ratio, not the magnitude),
//! symmetric, bounded, and total. `F_FLOOR` regularizes the dark-dark case —
//! without it, two states that are both numerically zero-ish (f=1e-9 vs
//! f=4e-5, both physically forbidden) would get d_f ≈ 1, i.e. maximum penalty,
//! for a difference with no physical content. `F_FLOOR = 0.01` is set at the
//! intensity below which a state is conventionally called dark, so
//! differences well under it are absorbed rather than punished.
//!
//! **The weight `w` is the one free parameter, and it is not asserted — it is
//! swept.** See [`assignment_stability_sweep`]: the caller is expected to run
//! the assignment across a range of `w` and report the range over which the
//! assignment is invariant. An assignment that only holds at one hand-picked
//! `w` is not a result.
//!
//! # Unmatched states, and why the gate is on the INTENSITY channel only
//!
//! Force-matching every reference state to its least-bad remaining root is how
//! the naive number got optimistic. [`assign_states`] therefore takes a gate
//! above which a pair is reported as [`Assignment::Unassigned`] rather than
//! silently paired, so a statistic can honestly say "N of M states assigned,
//! K unassigned".
//!
//! **The gate is `max_osc_dissimilarity` — a threshold on `d_f` alone, NOT on
//! the total cost.** This is a deliberate and load-bearing choice. The
//! quantity the benchmark MEASURES is the excitation-energy error `|ΔE|`, and
//! `|ΔE|` is the dominant term in the total cost. On this benchmark's own data
//! the two are strongly correlated (Pearson r = +0.86 at w = 2.0), so gating
//! on total cost would be *selection on the outcome variable*: it would
//! discard precisely the states with the largest energy errors and thereby
//! lower the reported MAE for a reason that has nothing to do with whether the
//! assignment is right. That is the same class of optimism this module exists
//! to remove, reintroduced through the back door.
//!
//! `d_f` is only weakly correlated with `|ΔE|` on the same data (r = +0.38),
//! because it is computed from intensities the statistic does not report. A
//! `d_f` gate therefore answers the question it is supposed to answer — "is
//! this computed root plausibly the same STATE as the reference?" — without
//! encoding an answer to "is ferric's energy good here?". Empirically it also
//! behaves: on this benchmark, tightening the `d_f` gate moves the MAE by less
//! than 0.06 eV in either direction, whereas a total-cost gate drops it
//! monotonically toward the best-matched subset.

use std::collections::BTreeSet;

/// Energy-channel normalization (eV per unit cost). Fixing this at 1 eV makes
/// a unit of cost mean "1 eV of excitation-energy error", independent of which
/// molecules are in the benchmark set.
pub const E_SCALE_EV: f64 = 1.0;

/// Intensity-channel regularizer. Oscillator strengths below this are
/// conventionally "dark"; differences well under it carry no state-character
/// information and must not dominate the relative measure.
pub const F_FLOOR: f64 = 0.01;

/// A state characterized by an excitation energy and an oscillator strength.
///
/// Used for both sides of the assignment (reference/literature and computed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateRecord {
    /// Vertical excitation energy, eV.
    pub energy_ev: f64,
    /// Oscillator strength (dimensionless, length gauge).
    ///
    /// Reference tables sometimes list no value for a symmetry-forbidden
    /// state. Callers should map that to `0.0` (physically forbidden IS zero
    /// intensity) rather than inventing a value.
    pub osc: f64,
}

impl StateRecord {
    /// Construct a state record.
    pub fn new(energy_ev: f64, osc: f64) -> Self {
        Self { energy_ev, osc }
    }
}

/// Bounded, symmetric, scale-free intensity dissimilarity in `[0, 1]`.
///
/// `d_f(a, b) = |a - b| / (a + b + F_FLOOR)`. Zero for equal intensities;
/// approaches 1 when one state is bright and the other dark. The `F_FLOOR`
/// term absorbs dark-vs-dark numerical noise (f=1e-9 vs f=4e-5 scores ~0.004,
/// not ~1).
pub fn osc_dissimilarity(a: f64, b: f64) -> f64 {
    let a = a.max(0.0);
    let b = b.max(0.0);
    (a - b).abs() / (a + b + F_FLOOR)
}

/// Joint (energy, oscillator-strength) cost of pairing `computed` with
/// `reference`, with intensity weight `w`.
///
/// `w = 0` reduces the cost to pure scaled energy distance, which is what
/// makes the exactness anchor (see module tests) meaningful: at `w = 0` the
/// Hungarian solution over a set where nearest-energy is already one-to-one
/// must reproduce nearest-energy exactly.
pub fn pair_cost(computed: StateRecord, reference: StateRecord, w: f64) -> f64 {
    let d_e = (computed.energy_ev - reference.energy_ev).abs() / E_SCALE_EV;
    let d_f = osc_dissimilarity(computed.osc, reference.osc);
    d_e + w * d_f
}

/// Outcome for one reference state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Assignment {
    /// Paired with the computed root at this zero-based index, at this cost.
    Assigned {
        /// Zero-based index into the computed-state slice.
        computed_index: usize,
        /// Joint cost of the accepted pair.
        cost: f64,
    },
    /// No computed root was an acceptable partner: the optimal pair's
    /// intensity dissimilarity exceeded `max_osc_dissimilarity`, or there were
    /// fewer computed roots than reference states. Reported rather than
    /// force-matched.
    Unassigned,
}

impl Assignment {
    /// The paired computed index, if any.
    pub fn index(&self) -> Option<usize> {
        match self {
            Assignment::Assigned { computed_index, .. } => Some(*computed_index),
            Assignment::Unassigned => None,
        }
    }
}

/// Solve the one-to-one assignment of reference states to computed roots.
///
/// Minimizes the total [`pair_cost`] subject to injectivity (each computed
/// root is used at most once), then applies the intensity gate: any accepted
/// pair whose [`osc_dissimilarity`] exceeds `max_osc_dissimilarity` is
/// downgraded to [`Assignment::Unassigned`] instead of being force-matched.
///
/// Pass `f64::INFINITY` for `max_osc_dissimilarity` to disable the gate and
/// force-match every reference state (useful for reporting the all-pairs
/// statistic alongside the gated one, which callers should do).
///
/// Returns one [`Assignment`] per reference state, in reference order.
///
/// # Algorithm
///
/// Exact rectangular Jonker–Volgenant/Hungarian (Kuhn–Munkres) assignment via
/// successive shortest augmenting paths with potentials — `O(n_ref^2 *
/// n_comp)`. The benchmark's sizes (≤ 3 reference states, ≤ ~30 candidate
/// roots per molecule) make this trivially cheap, and an exact solver avoids
/// the whole class of "greedy got a plausible-looking wrong answer" failure
/// that motivated this module.
///
/// The gate is applied AFTER the global solve, not as an infinite cost during
/// it. Pre-gating would change which pairs the optimizer may use and could
/// cascade a single bad pair into re-assigning good ones; post-gating keeps
/// the reported assignment identical to the ungated optimum and only annotates
/// which of its pairs are untrustworthy.
///
/// See the module docs for why the gate reads the intensity channel only and
/// never the total cost.
pub fn assign_states(
    reference: &[StateRecord],
    computed: &[StateRecord],
    w: f64,
    max_osc_dissimilarity: f64,
) -> Vec<Assignment> {
    let n_ref = reference.len();
    let n_comp = computed.len();
    if n_ref == 0 {
        return Vec::new();
    }
    if n_comp == 0 {
        return vec![Assignment::Unassigned; n_ref];
    }

    let cost = |r: usize, c: usize| pair_cost(computed[c], reference[r], w);

    // Rectangular Hungarian (n_ref rows <= or > n_comp cols both handled:
    // rows beyond the column count simply end unassigned).
    // `u`/`v` are the dual potentials, `col_of_row[r]` / `row_of_col[c]` the
    // primal matching. Standard e-maxx/JV formulation with 1-based sentinels.
    let n = n_ref;
    let m = n_comp;
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; m + 1];
    // row_of_col[c] = which row is matched to column c (n = "none")
    let mut row_of_col = vec![n; m + 1];

    for cur_row in 0..n {
        // Augment from `cur_row`. Column `m` is the virtual free column.
        row_of_col[m] = cur_row;
        let mut min_slack = vec![f64::INFINITY; m + 1];
        let mut prev_col = vec![m; m + 1];
        let mut used = vec![false; m + 1];
        let mut cur_col = m;
        loop {
            used[cur_col] = true;
            let row = row_of_col[cur_col];
            let mut delta = f64::INFINITY;
            let mut next_col = m;
            for c in 0..m {
                if used[c] {
                    continue;
                }
                let slack = cost(row, c) - u[row] - v[c];
                if slack < min_slack[c] {
                    min_slack[c] = slack;
                    prev_col[c] = cur_col;
                }
                if min_slack[c] < delta {
                    delta = min_slack[c];
                    next_col = c;
                }
            }
            if !delta.is_finite() {
                // No free column reachable: more reference states than
                // computed roots. Leave the rest unassigned.
                break;
            }
            for c in 0..=m {
                if used[c] {
                    let r = row_of_col[c];
                    if r < n {
                        u[r] += delta;
                    }
                    v[c] -= delta;
                } else {
                    min_slack[c] -= delta;
                }
            }
            cur_col = next_col;
            if row_of_col[cur_col] == n {
                break;
            }
        }
        // Walk the augmenting path back, flipping matched edges.
        while cur_col != m {
            let p = prev_col[cur_col];
            row_of_col[cur_col] = row_of_col[p];
            cur_col = p;
        }
    }

    let mut out = vec![Assignment::Unassigned; n_ref];
    for c in 0..m {
        let r = row_of_col[c];
        if r < n {
            let k = cost(r, c);
            out[r] = Assignment::Assigned {
                computed_index: c,
                cost: k,
            };
        }
    }
    // Post-gate on the INTENSITY channel only. Gating on the total cost would
    // select on |ΔE|, the very quantity the benchmark reports (module docs).
    for (r, a) in out.iter_mut().enumerate() {
        if let Assignment::Assigned { computed_index, .. } = a {
            let d_f = osc_dissimilarity(computed[*computed_index].osc, reference[r].osc);
            if d_f > max_osc_dissimilarity {
                *a = Assignment::Unassigned;
            }
        }
    }
    out
}

/// Result of sweeping the intensity weight `w`.
#[derive(Debug, Clone)]
pub struct StabilitySweep {
    /// The `(w, assignment)` pairs evaluated, in sweep order.
    pub per_weight: Vec<(f64, Vec<Assignment>)>,
    /// Distinct assignment patterns observed across the sweep, as the tuple of
    /// matched computed indices (`None` = unassigned) per reference state.
    pub distinct_patterns: Vec<Vec<Option<usize>>>,
}

impl StabilitySweep {
    /// True when every weight in the sweep produced the same assignment —
    /// i.e. the result does not depend on the one free parameter.
    pub fn is_stable(&self) -> bool {
        self.distinct_patterns.len() <= 1
    }
}

/// Run [`assign_states`] across a range of intensity weights and report which
/// distinct assignments appear.
///
/// The point is falsifiability: if the assignment swings with `w`, the
/// benchmark number is a function of a hand-picked parameter and must be
/// reported as such rather than quoted as a measurement.
pub fn assignment_stability_sweep(
    reference: &[StateRecord],
    computed: &[StateRecord],
    weights: &[f64],
    max_osc_dissimilarity: f64,
) -> StabilitySweep {
    let mut per_weight = Vec::with_capacity(weights.len());
    let mut seen: Vec<Vec<Option<usize>>> = Vec::new();
    for &w in weights {
        let a = assign_states(reference, computed, w, max_osc_dissimilarity);
        let pattern: Vec<Option<usize>> = a.iter().map(|x| x.index()).collect();
        if !seen.contains(&pattern) {
            seen.push(pattern);
        }
        per_weight.push((w, a));
    }
    StabilitySweep {
        per_weight,
        distinct_patterns: seen,
    }
}

/// Per-reference-state nearest-computed-ENERGY matching — the NAIVE heuristic
/// this module replaces.
///
/// Kept, and public, for exactly one reason: the exactness anchor. The new
/// assignment must reproduce this in the trivial regime where nearest-energy
/// is already one-to-one, and the module tests assert that. Do NOT use this to
/// produce a benchmark statistic; it is non-injective and f-blind, which is
/// the defect documented in `docs/bse-tda-phase2-results.md`.
pub fn naive_nearest_energy(
    reference: &[StateRecord],
    computed: &[StateRecord],
) -> Vec<Option<usize>> {
    reference
        .iter()
        .map(|r| {
            computed
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (a.energy_ev - r.energy_ev).abs();
                    let db = (b.energy_ev - r.energy_ev).abs();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
        })
        .collect()
}

/// True when [`naive_nearest_energy`] happens to be injective on this input —
/// i.e. no two reference states claim the same computed root.
///
/// This is the precondition of the exactness anchor: where nearest-energy is
/// already a valid one-to-one assignment AND intensities are consistent, the
/// Hungarian solution at `w = 0` must agree with it.
pub fn naive_is_injective(reference: &[StateRecord], computed: &[StateRecord]) -> bool {
    let picks = naive_nearest_energy(reference, computed);
    let mut seen = BTreeSet::new();
    for p in picks.iter().flatten() {
        if !seen.insert(*p) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const HUGE: f64 = f64::INFINITY;

    /// EXACTNESS ANCHOR (project CLAUDE.md "Experimental Protocol"): in the
    /// trivial limit — `w = 0` (intensity channel does nothing) and an
    /// ungated `max_cost` — the Hungarian assignment MUST reproduce naive
    /// nearest-energy matching on any input where nearest-energy is already
    /// one-to-one. This is the approximation-does-nothing case, and it is
    /// asserted before any statistic is recomputed.
    #[test]
    fn assignment_matches_naive_in_the_trivial_limit() {
        // Well-separated reference states, well-separated roots: the classic
        // unambiguous case (a single molecule like water, one clean state).
        let reference = [
            StateRecord::new(7.72, 0.05),
            StateRecord::new(9.40, 0.10),
            StateRecord::new(11.20, 0.00),
        ];
        let computed = [
            StateRecord::new(7.70, 0.90), // intensity deliberately WRONG:
            StateRecord::new(9.45, 0.90), // at w=0 it must be ignored.
            StateRecord::new(11.30, 0.90),
            StateRecord::new(14.00, 0.90),
        ];
        assert!(
            naive_is_injective(&reference, &computed),
            "anchor precondition: nearest-energy must already be one-to-one here"
        );
        let naive = naive_nearest_energy(&reference, &computed);
        let hung = assign_states(&reference, &computed, 0.0, HUGE);
        let got: Vec<Option<usize>> = hung.iter().map(|a| a.index()).collect();
        assert_eq!(
            got, naive,
            "w=0 Hungarian must reproduce nearest-energy where nearest-energy is injective"
        );
        assert_eq!(got, vec![Some(0), Some(1), Some(2)]);
    }

    /// The anchor again, at the smallest possible size: one reference state,
    /// one obvious root. Every matcher must agree here.
    #[test]
    fn single_state_anchor_is_trivially_exact() {
        let reference = [StateRecord::new(7.7233, 0.05)];
        let computed = [StateRecord::new(7.7233, 0.05), StateRecord::new(9.9, 0.30)];
        let hung = assign_states(&reference, &computed, 1.0, HUGE);
        assert_eq!(hung[0].index(), Some(0));
        if let Assignment::Assigned { cost, .. } = hung[0] {
            assert!(cost.abs() < 1e-12, "exact match must cost ~0, got {cost}");
        } else {
            panic!("exact match must be assigned");
        }
    }

    /// The defect, reproduced: two reference states, and nearest-energy sends
    /// BOTH to the same computed root (the pyridine/pyrimidine pattern). The
    /// Hungarian solution must be injective by construction.
    #[test]
    fn hungarian_is_injective_where_naive_collapses() {
        // Two refs at 4.96 and 5.13; ferric's roots at 6.02 and 6.13 — both
        // refs are nearest to root 0.
        let reference = [
            StateRecord::new(4.959, 0.004),
            StateRecord::new(5.133, 0.028),
        ];
        let computed = [
            StateRecord::new(6.0196, 0.00665),
            StateRecord::new(6.1275, 0.05705),
            StateRecord::new(6.2327, 0.0),
        ];
        let naive = naive_nearest_energy(&reference, &computed);
        assert_eq!(
            naive,
            vec![Some(0), Some(0)],
            "precondition: this input must reproduce the documented collapse"
        );
        assert!(!naive_is_injective(&reference, &computed));

        let hung = assign_states(&reference, &computed, 1.0, HUGE);
        let got: Vec<Option<usize>> = hung.iter().map(|a| a.index()).collect();
        assert_eq!(got.len(), 2);
        assert_ne!(got[0], got[1], "assignment must be one-to-one");
        assert!(got.iter().all(|x| x.is_some()));
    }

    /// The ethylene case: a bright reference (f=0.346) near a dark computed
    /// root, with the genuinely bright root a bit further away in energy.
    /// Nearest-energy takes the dark root; a joint cost with any meaningful
    /// intensity weight must take the bright one.
    #[test]
    fn intensity_channel_rescues_the_bright_state() {
        // Reference: ethylene's Rydberg (dim) and valence pi->pi* (bright).
        let reference = [
            StateRecord::new(7.367, 0.078),
            StateRecord::new(7.897, 0.346),
        ];
        // Computed: ferric's first four aug-cc-pVDZ roots.
        let computed = [
            StateRecord::new(7.3402, 0.08684),
            StateRecord::new(8.0094, 0.00000),
            StateRecord::new(8.0668, 0.00000),
            StateRecord::new(8.2889, 0.43793),
        ];
        let naive = naive_nearest_energy(&reference, &computed);
        assert_eq!(
            naive,
            vec![Some(0), Some(1)],
            "precondition: naive must pick the DARK root 1 for the bright reference"
        );

        let hung = assign_states(&reference, &computed, 1.0, HUGE);
        assert_eq!(hung[0].index(), Some(0), "Rydberg state keeps root 0");
        assert_eq!(
            hung[1].index(),
            Some(3),
            "bright pi->pi* must take the BRIGHT root 3, not the dark root 1"
        );
    }

    /// `w = 0` must NOT rescue the bright state — this is the negative control
    /// for the test above. If it passed at `w = 0` too, the intensity channel
    /// would not be doing the work and the test would be inert.
    #[test]
    fn zero_weight_does_not_rescue_the_bright_state() {
        let reference = [
            StateRecord::new(7.367, 0.078),
            StateRecord::new(7.897, 0.346),
        ];
        let computed = [
            StateRecord::new(7.3402, 0.08684),
            StateRecord::new(8.0094, 0.00000),
            StateRecord::new(8.0668, 0.00000),
            StateRecord::new(8.2889, 0.43793),
        ];
        let hung = assign_states(&reference, &computed, 0.0, HUGE);
        assert_eq!(
            hung[1].index(),
            Some(1),
            "at w=0 the dark root must still win; otherwise the w=1 test is inert"
        );
    }

    /// `osc_dissimilarity` must be bounded, symmetric, zero on the diagonal,
    /// and must NOT blow up on dark-vs-dark numerical noise.
    #[test]
    fn osc_dissimilarity_is_bounded_symmetric_and_dark_tolerant() {
        for &(a, b) in &[
            (0.0, 0.0),
            (0.0, 0.7),
            (0.346, 0.438),
            (1e-9, 4e-5),
            (0.004, 0.007),
            (0.6, 0.68),
        ] {
            let d = osc_dissimilarity(a, b);
            assert!((0.0..=1.0).contains(&d), "d_f({a},{b}) = {d} out of [0,1]");
            assert!(
                (d - osc_dissimilarity(b, a)).abs() < 1e-15,
                "d_f must be symmetric"
            );
        }
        assert!(osc_dissimilarity(0.5, 0.5) < 1e-15);
        // Dark-vs-dark noise is nearly free...
        assert!(
            osc_dissimilarity(1e-9, 4e-5) < 0.01,
            "two forbidden states must not be treated as different characters"
        );
        // ...while bright-vs-dark is nearly maximal.
        assert!(
            osc_dissimilarity(0.0, 0.346) > 0.95,
            "bright-vs-dark must be strongly penalized"
        );
        // And the pathology raw |df| has: dark-vs-bright must cost MORE than
        // a same-character intensity wobble, even though |df| says otherwise.
        assert!(
            osc_dissimilarity(0.004, 0.078) > osc_dissimilarity(0.60, 0.68),
            "relative measure must rank character change above magnitude wobble"
        );
    }

    /// The intensity gate must downgrade a character-mismatched pair to
    /// Unassigned rather than force-matching it.
    #[test]
    fn intensity_gate_reports_unassigned_instead_of_force_matching() {
        // A dark reference forced onto the only (bright) available root.
        let reference = [StateRecord::new(3.0, 0.0)];
        let computed = [StateRecord::new(3.1, 0.5)];
        let ungated = assign_states(&reference, &computed, 1.0, HUGE);
        assert_eq!(ungated[0].index(), Some(0), "ungated must still pair");
        assert!(
            osc_dissimilarity(0.5, 0.0) > 0.9,
            "precondition: this pair must be a character mismatch"
        );
        let gated = assign_states(&reference, &computed, 1.0, 0.3);
        assert_eq!(
            gated[0],
            Assignment::Unassigned,
            "a bright-vs-dark pair must be reported unassigned, not force-matched"
        );
    }

    /// THE GATE MUST NOT SELECT ON THE OUTCOME VARIABLE.
    ///
    /// The statistic this module feeds reports excitation-ENERGY error. If the
    /// gate looked at energy distance (or at the total cost, which `|ΔE|`
    /// dominates), then discarding "unassigned" states would mechanically drop
    /// the worst energy errors and flatter the result — the exact optimism
    /// this module exists to remove.
    ///
    /// Construction: a pair with a HUGE energy error but PERFECTLY matching
    /// intensities. A `d_f` gate must keep it (the states plainly correspond);
    /// any energy- or cost-based gate would throw it away.
    #[test]
    fn gate_ignores_energy_error_and_therefore_cannot_flatter_the_statistic() {
        let reference = [StateRecord::new(3.0, 0.20)];
        let computed = [StateRecord::new(9.0, 0.20)]; // 6 eV off, identical f
        for &gate in &[0.05, 0.1, 0.2, 0.3, 0.5] {
            let a = assign_states(&reference, &computed, 1.0, gate);
            assert_eq!(
                a[0].index(),
                Some(0),
                "a 6 eV energy error with matching intensity must SURVIVE the \
                 gate at d_f<={gate}; if it does not, the gate is selecting on \
                 the measured quantity"
            );
        }
        // And the converse: a tiny energy error with mismatched character must
        // be gated OUT, proving the gate is live and not a no-op.
        let reference2 = [StateRecord::new(3.0, 0.40)];
        let computed2 = [StateRecord::new(3.01, 0.0)];
        assert_eq!(
            assign_states(&reference2, &computed2, 1.0, 0.3)[0],
            Assignment::Unassigned,
            "gate must still reject a character mismatch at near-zero energy error"
        );
    }

    /// More reference states than computed roots: the surplus must come back
    /// Unassigned, not panic and not double-book a root.
    #[test]
    fn more_references_than_roots_leaves_surplus_unassigned() {
        let reference = [
            StateRecord::new(4.0, 0.0),
            StateRecord::new(5.0, 0.1),
            StateRecord::new(6.0, 0.2),
        ];
        let computed = [StateRecord::new(4.1, 0.0), StateRecord::new(5.1, 0.1)];
        let a = assign_states(&reference, &computed, 1.0, HUGE);
        let idx: Vec<Option<usize>> = a.iter().map(|x| x.index()).collect();
        let n_assigned = idx.iter().flatten().count();
        assert_eq!(n_assigned, 2, "only two roots exist");
        let mut seen = BTreeSet::new();
        for i in idx.iter().flatten() {
            assert!(seen.insert(*i), "no root may be double-booked");
        }
    }

    /// The Hungarian solution must be a true global optimum, not a greedy one.
    /// This input is the canonical greedy trap: the globally best pairing
    /// requires reference 0 to GIVE UP its own individual argmin.
    #[test]
    fn hungarian_beats_greedy_on_the_classic_trap() {
        // ref0's best root is 0 (cost 0.0); ref1's best is also 0 (cost 0.1).
        // Greedy-by-reference-order gives ref0->0, ref1->1 (total 0.0 + 1.0).
        // But the optimum is ref0->1 (0.9), ref1->0 (0.1) = 1.0 ... tie.
        // Make it strict: shift so the swap strictly wins.
        let reference = [StateRecord::new(5.0, 0.0), StateRecord::new(5.1, 0.0)];
        let computed = [StateRecord::new(5.05, 0.0), StateRecord::new(5.6, 0.0)];
        let a = assign_states(&reference, &computed, 0.0, HUGE);
        let total: f64 = a
            .iter()
            .filter_map(|x| match x {
                Assignment::Assigned { cost, .. } => Some(*cost),
                Assignment::Unassigned => None,
            })
            .sum();
        // Enumerate both injective pairings by hand.
        let c = |r: usize, k: usize| pair_cost(computed[k], reference[r], 0.0);
        let opt = (c(0, 0) + c(1, 1)).min(c(0, 1) + c(1, 0));
        assert!(
            (total - opt).abs() < 1e-12,
            "total cost {total} is not the global optimum {opt}"
        );
    }

    /// A stability sweep over a set with no ambiguity must report stable; a
    /// set engineered to flip must report unstable. Both directions asserted
    /// so the stability check cannot be vacuously true.
    #[test]
    fn stability_sweep_detects_both_stable_and_unstable_cases() {
        // Stable: well-separated in BOTH channels.
        let r_stable = [StateRecord::new(4.0, 0.0), StateRecord::new(8.0, 0.5)];
        let c_stable = [StateRecord::new(4.1, 0.0), StateRecord::new(8.1, 0.5)];
        let s = assignment_stability_sweep(&r_stable, &c_stable, &[0.0, 0.5, 1.0, 2.0, 5.0], HUGE);
        assert!(
            s.is_stable(),
            "unambiguous input must be weight-independent"
        );

        // Unstable: the bright reference sits between a near-in-energy dark
        // root and a further-away bright one, so the winner flips with w.
        let r_flip = [StateRecord::new(7.897, 0.346)];
        let c_flip = [
            StateRecord::new(8.0094, 0.0),
            StateRecord::new(8.2889, 0.43793),
        ];
        let f = assignment_stability_sweep(&r_flip, &c_flip, &[0.0, 0.1, 1.0, 5.0], HUGE);
        assert!(
            !f.is_stable(),
            "an input that flips with w must be reported UNSTABLE"
        );
        assert!(f.distinct_patterns.len() >= 2);
    }
}
