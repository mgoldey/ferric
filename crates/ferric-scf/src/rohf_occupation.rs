//! Occupation bookkeeping for the ROHF/ROKS DIIS loop (defect F6).
//!
//! # The defect this exists for
//!
//! `solve_rohf` assigns the closed / open / virtual labels by the INDEX of the
//! eigenvalues of the Guest-Saunders effective Fock: the lowest `nocc_double`
//! eigenvectors are closed, the next `nocc_open` are open. Guest-Saunders sets
//! both the closed-closed and the open-open diagonal blocks to
//! `F_c = (F_α + F_β)/2`, and nothing in that choice orders an open orbital
//! ABOVE a closed one. MEASURED (OH ²Π, ROKS/PBE/STO-3G, a numpy replica of
//! this loop on PySCF integrals): in the state with π_y doubly and π_x singly
//! occupied, the effective-Fock eigenvalues are π_x −0.1103 and π_y −0.0764 —
//! the OPEN π lies BELOW the closed one — so index aufbau re-occupies the other
//! π on the next iteration, and the iteration after that swaps back. The two
//! determinants are symmetry-equivalent (same energy, same orbital gradient),
//! so the energy and the gradient both converge while the density swaps the
//! hole between π_x and π_y every iteration: `dp_rms` is pinned at 0.236 and
//! the ΔP gate can never fire. Pulay DIIS on that history is fed near-identical
//! error vectors, its coefficients blow up (measured +62.9 / −62.5), and the
//! extrapolated Fock lands on a different configuration 0.58 Ha higher
//! (σ* singly occupied, one π emptied). The loop can stagnate there, and the
//! ΔP-only gate then reports it as converged: on the CI runner at the
//! unperturbed geometry (−73.9916949), and reproduced locally with exact J
//! and H moved 1e-7 Å along the bond (−73.9911207, 65 iterations, accepted
//! with `err_max` = 0.12).
//!
//! The same flip has a second form: in NO and CH (²Π, one electron in a
//! degenerate pair) it is the single ELECTRON that hops between the two π
//! orbitals, an open↔virtual swap rather than a closed↔open one. MEASURED in
//! the replica for NO/6-31G/PBE: the effective-Fock eigenvalues of the two π*
//! (−0.1465 / −0.1269) cross every iteration, while `F_α` keeps the occupied
//! one lower (−0.1591 vs −0.1538). ROKS/PBE on OH (STO-3G, 6-31G), NO and CH
//! (6-31G) all failed to converge in 200 iterations before this module.
//!
//! Plain ROHF (no functional) on OH is unaffected: there the effective-Fock
//! ordering of the π pair is the physical one and no swap ever occurs (7 and
//! 10 iterations at every perturbation, STO-3G and 6-31G).
//!
//! # The three remedies, and why each is gated the way it is
//!
//! 1. **Hole-swap lock** ([`open_space_changed`], [`select_by_continuity`]).
//!    Once the open space has changed identity [`HOLE_SWAP_LOCK_STREAK`]
//!    iterations in a row, the labels are chosen by CONTINUITY (maximum
//!    overlap with the previous iteration's closed and open blocks) instead of
//!    by eigenvalue index, and the DIIS history is cleared. A first version
//!    chose them by spin-Fock (Koopmans) aufbau instead; that failed on
//!    SVWN-LDA — see [`select_by_continuity`].
//!    The lock is NOT engaged from iteration 1, because early-iteration
//!    orbitals are too poor to hold on to. MEASURED on the 28 ROHF
//!    state-selection rows (both guesses, 56 runs), with the spin-Fock
//!    version: engaging from iteration 1 locked four hcore-guess runs
//!    (HeNe⁺, O₂, NO, CH, all 6-31G) into states 0.12-0.30 Ha high, and a
//!    two-swap streak still locked HeNe⁺/6-31G from MINAO into a state
//!    4.4 mHa high. A three-swap streak engages on ONE of the 56 runs
//!    (HeNe⁺/6-31G from hcore) and still catches the OH, NO and CH flips,
//!    which swap every iteration. With the continuity lock that one run locks
//!    into a higher state, from which the swap witness finds a one-electron
//!    move 61 mHa lower: it still ends on the reference, but in 135 iterations
//!    (the unlocked loop: 67-127 across replica runs).
//! 2. **Gradient guard** ([`ACCEPT_MAX_ORBITAL_GRADIENT`]). The shared
//!    convergence gate reads ΔP and ΔE only. A DIIS extrapolation that
//!    stagnates reproduces the same density and energy without being a
//!    stationary point, which is how the 0.58 Ha state was accepted on CI. A
//!    state is accepted only if its orbital gradient is also below this bound,
//!    which sits three orders above the largest RI gradient floor on record in
//!    this repo (~1.3e-6, toluene/aTZ) and two orders below the stagnated
//!    state's 0.12.
//! 3. **Swap witness** ([`aufbau_swap_candidates`], [`AUFBAU_WITNESS_TOL`]).
//!    At convergence the energy of the best single-swap neighbour is evaluated
//!    (one closed↔open β swap and one open↔virtual α swap, each chosen by the
//!    smallest Koopmans estimate). The swapped determinant is a valid ROHF
//!    determinant, so its UNRELAXED energy is an upper bound on what the SCF
//!    can reach from it; if it is lower than the converged energy, the
//!    converged state is provably not the lowest state reachable by moving one
//!    electron, and the SCF continues from the swapped determinant. The
//!    Koopmans estimate alone cannot be used as the test: on OH/6-31G from the
//!    hcore guess (σ hole, ²Σ⁺, 4.3 eV high) the best β swap has Koopmans
//!    +0.462 Ha — "stable" — while its unrelaxed energy is 0.154 Ha LOWER,
//!    because the hole-electron Coulomb term the estimate omits is ~0.6 Ha.
//!
//! All three are disabled when the caller asked for MOM (`mom_after_iter > 0`)
//! — MOM is the documented tool for deliberately holding a non-aufbau
//! (ΔSCF excited) state, and the guard must not undo it — and all three are
//! switched off by `RhfConfig::rohf_occupation_guard = false`, which restores
//! the pre-F6 loop exactly.

use crate::diis::Diis;
use crate::driver::ScfMonitor;
use crate::result::{ScfExit, ScfResult};
use ndarray::{s, Array2};

/// Consecutive iterations in which the open space changed identity before the
/// occupation labels are locked by continuity. Measured, not guessed
/// (see the module doc): 2 locks HeNe⁺/6-31G (MINAO) into a state 4.4 mHa
/// high; 3 engages on one of 56 state-selection runs (same state) and still
/// catches the OH/NO/CH flips, which swap EVERY iteration; 6 was also tried
/// and delays the OH lock enough to cost 13 iterations.
pub(crate) const HOLE_SWAP_LOCK_STREAK: usize = 3;

/// How far the open-space weight `‖C_open_prevᵀ S C_open_new‖²_F` (which is
/// `nocc_open` when nothing moved) must drop to count as a swap. A clean swap
/// drops it by ~1 per exchanged orbital, a continuous iteration by ~0.
pub(crate) const HOLE_SWAP_OVERLAP: f64 = 0.5;

/// A ΔP/ΔE-converged ROHF state whose `err_max` (max |AO orbital-gradient|)
/// exceeds this is not accepted. See the module doc for where it sits.
pub(crate) const ACCEPT_MAX_ORBITAL_GRADIENT: f64 = 1e-3;

/// A swapped determinant must be lower than the converged energy by more than
/// this (Ha) to count as a witness. Sized above the ≤ 3.4e-6 Ha by which a
/// ROKS energy was measured to move when a degenerate hole is rotated relative
/// to the (not axially symmetric) Becke-Lebedev grid, and far below every
/// state gap the witness has found (≥ 2.4e-3 Ha, B₂/6-31G).
pub(crate) const AUFBAU_WITNESS_TOL: f64 = 1e-5;

/// Witness-triggered restarts allowed before the solve gives up and reports
/// non-convergence instead of returning a state it could not certify.
pub(crate) const MAX_AUFBAU_RESTARTS: usize = 3;

/// [`MAX_AUFBAU_RESTARTS`], unless a test lowered it through
/// `crate::rohf::ROHF_MAX_AUFBAU_RESTARTS_OVERRIDE` (`usize::MAX` = no
/// override) to reach the give-up exit on a small system.
fn max_restarts() -> usize {
    match crate::rohf::ROHF_MAX_AUFBAU_RESTARTS_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed)
    {
        usize::MAX => MAX_AUFBAU_RESTARTS,
        n => n,
    }
}

/// Did the OPEN space change identity between `c_prev` and `c_new`?
///
/// The weight `w = ‖C_open_prevᵀ S C_open_new‖²_F ∈ [0, nocc_open]` is
/// `nocc_open` when the new open orbitals span the old open space (any
/// rotation inside it included, e.g. the two π* of triplet O₂) and drops by
/// ~1 for every open orbital that was exchanged with a closed one (the OH
/// ²Π flip: the hole moves between π_x and π_y) or with a virtual one (the
/// NO / CH ²Π flip: the single electron moves between the two π*). Counted as
/// a swap when `w < nocc_open − HOLE_SWAP_OVERLAP`.
pub(crate) fn open_space_changed(
    c_prev: &Array2<f64>,
    c_new: &Array2<f64>,
    s: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
) -> bool {
    if nocc_open == 0 {
        return false;
    }
    let lo = nocc_double;
    let hi = nocc_double + nocc_open;
    let open_prev = c_prev.slice(s![.., lo..hi]);
    let open_new = c_new.slice(s![.., lo..hi]);
    let ov = open_prev.t().dot(&s.dot(&open_new));
    let w: f64 = ov.iter().map(|x| x * x).sum();
    w < nocc_open as f64 - HOLE_SWAP_OVERLAP
}

/// Diagonal `⟨c_k|F|c_k⟩` for the columns `lo..hi` of `c`.
fn mo_diagonal(c: &Array2<f64>, f: &Array2<f64>, lo: usize, hi: usize) -> Vec<f64> {
    let block = c.slice(s![.., lo..hi]);
    let fc = f.dot(&block);
    (0..hi - lo)
        .map(|k| block.column(k).dot(&fc.column(k)))
        .collect()
}

/// For every column `p` of `c_new`: `Σ_r (c_ref_rᵀ S c_new_p)²` over the
/// columns `lo..hi` of `c_ref` — the weight of `c_new_p` inside that block.
fn block_weights(c_ref: &Array2<f64>, lo: usize, hi: usize, s_c_new: &Array2<f64>) -> Vec<f64> {
    let ov = c_ref.slice(s![.., lo..hi]).t().dot(s_c_new);
    (0..s_c_new.ncols())
        .map(|p| ov.column(p).iter().map(|x| x * x).sum())
        .collect()
}

/// The `k` entries of `candidates` with the largest `weight` (ties to the lower
/// index), returned in ascending index order.
fn top_k_by_weight(candidates: &[usize], weight: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = candidates.to_vec();
    idx.sort_by(|&a, &b| {
        weight[b]
            .partial_cmp(&weight[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let mut chosen: Vec<usize> = idx.into_iter().take(k).collect();
    chosen.sort_unstable();
    chosen
}

/// Re-label the eigenvectors `c_new` by CONTINUITY with `c_ref` (maximum
/// overlap, Gilbert-Besley-Gill): the `nocc_double` eigenvectors with the
/// largest weight in `c_ref`'s closed block become closed, then the
/// `nocc_open` of the rest with the largest weight in its open block become
/// open. Used only once the hole-swap lock is engaged.
///
/// Why continuity and not a spin-Fock (Koopmans) choice: the first version of
/// this lock picked the open orbitals by `F_α` and the closed ones by `F_β`.
/// That cured OH, NO and CH under PBE, but MEASURED on OH/STO-3G under
/// SVWN-LDA (R = 0.98 Å) the β-Fock order of the two π inverts with the
/// occupation — LDA self-interaction lets an occupied orbital feel its own
/// field — so the locked loop kept swapping the hole every iteration
/// (dp_rms 0.236 for 400 iterations at a converged energy and gradient
/// 5e-11). Continuity cannot oscillate by construction, and a lock that holds
/// the wrong state is what the swap witness exists to catch.
///
/// Within each block the eigenvalue order is kept, and `eps` is permuted with
/// the columns. Returns the inputs UNCHANGED (no permutation, same
/// allocation) when the choice is already the index split.
pub(crate) fn select_by_continuity(
    eps: Vec<f64>,
    c_new: Array2<f64>,
    c_ref: &Array2<f64>,
    s: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
) -> (Vec<f64>, Array2<f64>) {
    let n = c_new.ncols();
    let nocc_a = nocc_double + nocc_open;
    if nocc_open == 0 || nocc_a > n {
        return (eps, c_new);
    }
    let s_c_new = s.dot(&c_new);
    let all: Vec<usize> = (0..n).collect();
    let closed_w = block_weights(c_ref, 0, nocc_double, &s_c_new);
    let closed = top_k_by_weight(&all, &closed_w, nocc_double);
    let rest: Vec<usize> = all.into_iter().filter(|p| !closed.contains(p)).collect();
    let open_w = block_weights(c_ref, nocc_double, nocc_a, &s_c_new);
    let open = top_k_by_weight(&rest, &open_w, nocc_open);
    let virt: Vec<usize> = rest.into_iter().filter(|p| !open.contains(p)).collect();
    let order: Vec<usize> = closed.into_iter().chain(open).chain(virt).collect();
    if order.iter().enumerate().all(|(k, &p)| k == p) {
        return (eps, c_new);
    }
    (permute_eps(&eps, &order), permute_columns(&c_new, &order))
}

fn permute_columns(c: &Array2<f64>, order: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros(c.raw_dim());
    for (dst, &src) in order.iter().enumerate() {
        out.column_mut(dst).assign(&c.column(src));
    }
    out
}

fn permute_eps(eps: &[f64], order: &[usize]) -> Vec<f64> {
    if eps.len() != order.len() {
        // Defensive: eps from a diagonalization always has ncols entries.
        return eps.to_vec();
    }
    order.iter().map(|&k| eps[k]).collect()
}

/// `c` with columns `i` and `j` exchanged.
pub(crate) fn swap_columns(c: &Array2<f64>, i: usize, j: usize) -> Array2<f64> {
    let mut out = c.clone();
    if i != j {
        out.column_mut(i).assign(&c.column(j));
        out.column_mut(j).assign(&c.column(i));
    }
    out
}

/// Which single-electron move a witness candidate represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SwapKind {
    /// A β electron moves from closed orbital `from` into open orbital `to`
    /// (the hole moves from `to` to `from`).
    BetaClosedToOpen,
    /// The α electron in open orbital `from` moves into virtual `to`.
    AlphaOpenToVirtual,
}

/// One witness candidate: the swap, the two column indices, and its
/// first-order (Koopmans) energy estimate — recorded for the log only; the
/// decision is made on the evaluated energy.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SwapCandidate {
    pub kind: SwapKind,
    pub from: usize,
    pub to: usize,
    pub koopmans: f64,
}

/// The single-swap neighbours worth evaluating at convergence: the closed↔open
/// β swap and the open↔virtual α swap with the smallest Koopmans estimate
/// (`F_β[t,t] − F_β[i,i]` and `F_α[a,a] − F_α[t,t]`). At most two.
///
/// Closed↔virtual (pair) moves are not generated: at convergence the
/// effective-Fock eigenvalues ARE the `F_c` diagonals and the occupied set is
/// their aufbau set, so that ordering already holds by construction.
pub(crate) fn aufbau_swap_candidates(
    c: &Array2<f64>,
    f_a: &Array2<f64>,
    f_b: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
) -> Vec<SwapCandidate> {
    let n = c.ncols();
    let nocc_a = nocc_double + nocc_open;
    let mut out = Vec::new();
    if nocc_open == 0 || nocc_a > n {
        return out;
    }
    if nocc_double > 0 {
        let fbd = mo_diagonal(c, f_b, 0, nocc_a);
        let mut best: Option<SwapCandidate> = None;
        for i in 0..nocc_double {
            for t in nocc_double..nocc_a {
                let k = fbd[t] - fbd[i];
                if best.map_or(true, |b| k < b.koopmans) {
                    best = Some(SwapCandidate {
                        kind: SwapKind::BetaClosedToOpen,
                        from: i,
                        to: t,
                        koopmans: k,
                    });
                }
            }
        }
        out.extend(best);
    }
    if nocc_a < n {
        let fad = mo_diagonal(c, f_a, nocc_double, n);
        let mut best: Option<SwapCandidate> = None;
        for t in nocc_double..nocc_a {
            for a in nocc_a..n {
                let k = fad[a - nocc_double] - fad[t - nocc_double];
                if best.map_or(true, |b| k < b.koopmans) {
                    best = Some(SwapCandidate {
                        kind: SwapKind::AlphaOpenToVirtual,
                        from: t,
                        to: a,
                        koopmans: k,
                    });
                }
            }
        }
        out.extend(best);
    }
    out
}

/// A converged state held while its single-swap neighbours are evaluated.
struct AufbauProbe {
    /// The converged result, returned unchanged if no neighbour is lower.
    stash: Box<ScfResult>,
    /// Its energy.
    energy: f64,
    /// Neighbours not yet evaluated (popped from the back).
    pending: Vec<(SwapCandidate, Array2<f64>)>,
    /// The neighbour whose Fock the current pass of the loop builds.
    current: SwapCandidate,
}

/// What `solve_rohf`'s loop does after [`OccupationGuard::on_energy`].
pub(crate) enum GuardStep {
    /// Carry on with the ordinary iteration.
    Proceed,
    /// Start the next pass now (a witness probe's orbitals are queued; see
    /// [`OccupationGuard::take_pending_orbitals`]).
    Continue,
    /// No neighbour was lower: the held converged state stands.
    Return(Box<ScfResult>),
    /// The witness forced more than [`MAX_AUFBAU_RESTARTS`] restarts. Carries
    /// the HELD converged state — the one the last witness showed a lower
    /// neighbour of — marked `converged = false`, `exit = NotCertified`:
    /// energy, MOs and densities all describe that one state. (The loop's own
    /// variables at this point hold the probe determinant, the previous
    /// iteration's Fock and a stale energy, so the generic non-converged exit
    /// must NOT be used here.)
    Stop(Box<ScfResult>),
}

/// All of the F6 state `solve_rohf` carries, behind the four calls the loop
/// makes (see the module doc for what each remedy is and why).
///
/// Disabled (`enabled = false`) every method is the identity / pass-through,
/// so the loop is bit-identical to the pre-F6 one.
pub(crate) struct OccupationGuard {
    enabled: bool,
    nocc_double: usize,
    nocc_open: usize,
    /// Labels chosen by continuity instead of eigenvalue index.
    locked: bool,
    /// Whether the continuity lock may engage at all (false on the injected
    /// periodic path, see [`OccupationGuard::with_continuity_lock`]).
    lock_enabled: bool,
    swap_streak: usize,
    grad_warned: bool,
    probe: Option<AufbauProbe>,
    /// Orbitals the next pass must build its Fock from (a witness probe).
    pending_orbitals: Option<Array2<f64>>,
    restarts: usize,
    /// Extra passes used by probes; they do not consume `max_iter`.
    probe_iters: usize,
}

impl OccupationGuard {
    pub(crate) fn new(enabled: bool, nocc_double: usize, nocc_open: usize) -> Self {
        Self {
            enabled,
            nocc_double,
            nocc_open,
            locked: false,
            lock_enabled: true,
            swap_streak: 0,
            grad_warned: false,
            probe: None,
            pending_orbitals: None,
            restarts: 0,
            probe_iters: 0,
        }
    }

    /// Disable the continuity lock (the gradient guard and the aufbau swap
    /// witness stay active). Used by the injected periodic ROHF/ROKS path:
    /// measured on the triclinic 4H s+p triplet (Gamma, periodic SSF LDA), the
    /// lock held a hole state 2.4e-2 Ha above the aufbau ground state
    /// (gap_alpha -0.04) that the unrelaxed single-swap witness could not
    /// refute; with the lock off the state matches PySCF/the prototype to 1e-12.
    pub(crate) fn with_continuity_lock(mut self, enabled: bool) -> Self {
        self.lock_enabled = enabled;
        self
    }

    /// The loop's pass limit: `max_iter` plus the passes probes have used.
    pub(crate) fn iteration_cap(&self, max_iter: usize) -> usize {
        max_iter + self.probe_iters
    }

    /// Orbitals queued for a witness probe, to be installed (with their
    /// densities) BEFORE this pass builds its Fock.
    pub(crate) fn take_pending_orbitals(&mut self) -> Option<Array2<f64>> {
        self.pending_orbitals.take()
    }

    /// Re-label a diagonalization's eigenpairs by continuity with `c_ref`
    /// when the lock is engaged; otherwise return them untouched.
    pub(crate) fn relabel(
        &self,
        c_ref: &Array2<f64>,
        eps: Vec<f64>,
        c_new: Array2<f64>,
        s: &Array2<f64>,
    ) -> (Vec<f64>, Array2<f64>) {
        if !self.locked {
            return (eps, c_new);
        }
        select_by_continuity(eps, c_new, c_ref, s, self.nocc_double, self.nocc_open)
    }

    /// The DIIS step's new orbitals: count open-space swaps against `c_prev`,
    /// engage the lock after [`HOLE_SWAP_LOCK_STREAK`] in a row (clearing the
    /// DIIS history built on the swapping states), and apply it.
    pub(crate) fn select(
        &mut self,
        c_prev: &Array2<f64>,
        eps: Vec<f64>,
        c_new: Array2<f64>,
        s: &Array2<f64>,
        diis: &mut Diis,
        announce: bool,
    ) -> Array2<f64> {
        if !self.enabled || !self.lock_enabled {
            return c_new;
        }
        if !self.locked {
            let swapped = open_space_changed(c_prev, &c_new, s, self.nocc_double, self.nocc_open);
            self.swap_streak = if swapped { self.swap_streak + 1 } else { 0 };
            if self.swap_streak < HOLE_SWAP_LOCK_STREAK {
                return c_new;
            }
            self.locked = true;
            diis.reset();
            if announce {
                eprintln!(
                    "ROHF: open shell swapped {} iterations running (degenerate open shell) \
                     — holding the occupation by orbital continuity from here on",
                    self.swap_streak
                );
            }
        }
        self.relabel(c_prev, eps, c_new, s).1
    }

    /// The gradient guard: a ΔP/ΔE-converged state is accepted only if its
    /// orbital gradient is below [`ACCEPT_MAX_ORBITAL_GRADIENT`]; otherwise
    /// DIIS is cleared and the loop continues.
    pub(crate) fn accept(
        &mut self,
        converged: bool,
        err_max: f64,
        iter: usize,
        diis: &mut Diis,
        root: bool,
    ) -> bool {
        if !converged || !self.enabled || err_max <= ACCEPT_MAX_ORBITAL_GRADIENT {
            return converged;
        }
        if !self.grad_warned && root {
            eprintln!(
                "ROHF: density and energy have stopped changing at iteration {iter} but the \
                 orbital gradient is {err_max:.3e} (> {ACCEPT_MAX_ORBITAL_GRADIENT:.0e}) — not a \
                 stationary point (stagnated extrapolation). Clearing DIIS and continuing."
            );
            self.grad_warned = true;
        }
        diis.reset();
        false
    }

    /// A converged `result` (whose `mos_alpha` are the converged orbitals):
    /// `Some(result)` to return it now, or `None` after queueing its first
    /// single-swap neighbour for evaluation (the caller `continue`s).
    pub(crate) fn on_converged(
        &mut self,
        result: ScfResult,
        f_a: &Array2<f64>,
        f_b: &Array2<f64>,
    ) -> Option<ScfResult> {
        if !self.enabled {
            return Some(result);
        }
        let c_f = &result.mos_alpha;
        let mut pending: Vec<(SwapCandidate, Array2<f64>)> =
            aufbau_swap_candidates(c_f, f_a, f_b, self.nocc_double, self.nocc_open)
                .into_iter()
                .map(|cand| (cand, swap_columns(c_f, cand.from, cand.to)))
                .collect();
        // `pop()` then yields them in generation order (β swap first).
        pending.reverse();
        let Some((current, c_probe)) = pending.pop() else {
            return Some(result);
        };
        self.pending_orbitals = Some(c_probe);
        self.probe_iters += 1;
        self.probe = Some(AufbauProbe {
            energy: result.energy,
            stash: Box::new(result),
            pending,
            current,
        });
        None
    }

    /// Called with each pass's energy. While a probe is outstanding this
    /// energy is the probe determinant's UNRELAXED energy: lower than the held
    /// state by [`AUFBAU_WITNESS_TOL`] ⇒ restart from it (lock on, DIIS and
    /// the convergence monitor cleared); otherwise evaluate the next
    /// neighbour, or return the held state when none is left.
    pub(crate) fn on_energy(
        &mut self,
        energy: f64,
        iter: usize,
        diis: &mut Diis,
        mon: &mut ScfMonitor,
        root: bool,
    ) -> GuardStep {
        let Some(mut probe) = self.probe.take() else {
            return GuardStep::Proceed;
        };
        if energy < probe.energy - AUFBAU_WITNESS_TOL {
            return self.restart(probe, energy, iter, diis, mon, root);
        }
        match probe.pending.pop() {
            Some((next, c_next)) => {
                probe.current = next;
                self.pending_orbitals = Some(c_next);
                self.probe_iters += 1;
                self.probe = Some(probe);
                GuardStep::Continue
            }
            None => GuardStep::Return(probe.stash),
        }
    }

    fn restart(
        &mut self,
        probe: AufbauProbe,
        energy: f64,
        iter: usize,
        diis: &mut Diis,
        mon: &mut ScfMonitor,
        root: bool,
    ) -> GuardStep {
        self.restarts += 1;
        let cand = probe.current;
        if root {
            eprintln!(
                "ROHF aufbau check: the state converged at E = {:.10} is NOT the lowest \
                 single-swap state — moving one electron ({:?}, MO {} -> {}, Koopmans estimate \
                 {:+.4e}) gives E = {energy:.10} unrelaxed ({:+.4e} Ha). Continuing the SCF \
                 from it (restart {}/{}).",
                probe.energy,
                cand.kind,
                cand.from,
                cand.to,
                cand.koopmans,
                energy - probe.energy,
                self.restarts,
                max_restarts(),
            );
        }
        if self.restarts > max_restarts() {
            if root {
                eprintln!(
                    "ROHF aufbau check: giving up after {} restarts — returning the last \
                     converged state (E = {:.10}) as NOT converged (exit NotCertified).",
                    max_restarts(),
                    probe.energy
                );
            }
            let mut held = probe.stash;
            held.converged = false;
            held.exit = ScfExit::NotCertified;
            held.iterations = iter;
            return GuardStep::Stop(held);
        }
        self.locked = self.lock_enabled;
        self.swap_streak = 0;
        diis.reset();
        *mon = ScfMonitor::new();
        GuardStep::Proceed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Four orthonormal MOs in an orthonormal AO basis (S = I).
    fn eye4() -> Array2<f64> {
        Array2::eye(4)
    }

    #[test]
    fn continuity_is_the_identity_when_nothing_moved() {
        let eps = vec![-1.1, -0.6, 0.1, 0.8];
        let (e2, c2) = select_by_continuity(eps.clone(), eye4(), &eye4(), &eye4(), 1, 1);
        assert_eq!(e2, eps);
        assert_eq!(c2, eye4());
    }

    #[test]
    fn continuity_undoes_a_closed_open_swap() {
        // The OH ²Π pattern: index aufbau put the previous OPEN orbital in the
        // closed slot and vice versa. Continuity restores the previous labels,
        // and each eigenvalue follows its column.
        let eps = vec![-0.1103, -0.0764, 0.335, 0.8];
        let c_new = swap_columns(&eye4(), 0, 1);
        let (e2, c2) = select_by_continuity(eps, c_new, &eye4(), &eye4(), 1, 1);
        assert_eq!(c2, eye4());
        assert_eq!(e2, vec![-0.0764, -0.1103, 0.335, 0.8]);
    }

    #[test]
    fn continuity_undoes_an_open_virtual_swap() {
        // The NO / CH ²Π pattern: the single electron hopped to the other π*.
        let c_new = swap_columns(&eye4(), 1, 2);
        let (_, c2) = select_by_continuity(vec![0.0; 4], c_new, &eye4(), &eye4(), 1, 1);
        assert_eq!(c2, eye4());
    }

    #[test]
    fn continuity_follows_a_rotated_reference_in_a_nonorthogonal_basis() {
        // S != I: overlaps must go through S. c_ref's closed orbital is the
        // S-normalized first AO; c_new lists the same orbitals in reverse.
        let s = array![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let x = 1.0 / (1.0f64 - 0.04).sqrt();
        // Löwdin-free orthonormal set: e1, (e2 - 0.2 e1)/sqrt(0.96), e3.
        let c_ref = array![[1.0, -0.2 * x, 0.0], [0.0, x, 0.0], [0.0, 0.0, 1.0]];
        let order = [2usize, 1, 0];
        let c_new = permute_columns(&c_ref, &order);
        let (_, c2) = select_by_continuity(vec![0.0; 3], c_new, &c_ref, &s, 1, 1);
        assert_eq!(c2, c_ref);
    }

    #[test]
    fn open_space_change_detects_both_swap_kinds_and_ignores_continuity() {
        let s = eye4();
        let c_prev = eye4();
        // nd = 1, no = 1: open is col 1.
        assert!(!open_space_changed(&c_prev, &eye4(), &s, 1, 1));
        // closed <-> open (OH hole flip).
        assert!(open_space_changed(
            &c_prev,
            &swap_columns(&eye4(), 0, 1),
            &s,
            1,
            1
        ));
        // open <-> virtual (NO / CH electron flip).
        assert!(open_space_changed(
            &c_prev,
            &swap_columns(&eye4(), 1, 2),
            &s,
            1,
            1
        ));
        // closed <-> virtual leaves the open space alone: not this defect.
        assert!(!open_space_changed(
            &c_prev,
            &swap_columns(&eye4(), 0, 2),
            &s,
            1,
            1
        ));
        // A small rotation between closed and open is NOT a swap.
        let (co, si) = (0.99f64, (1.0f64 - 0.99 * 0.99).sqrt());
        let rot = array![
            [co, -si, 0.0, 0.0],
            [si, co, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        assert!(!open_space_changed(&c_prev, &rot, &s, 1, 1));
        // Two open orbitals rotated INTO each other (triplet O2's π*) keep the
        // open space: not a swap.
        let r45 = std::f64::consts::FRAC_1_SQRT_2;
        let mix = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, r45, -r45, 0.0],
            [0.0, r45, r45, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        assert!(!open_space_changed(&c_prev, &mix, &s, 1, 2));
    }

    #[test]
    fn witness_candidates_pick_the_smallest_koopmans_estimate_per_class() {
        // nd = 2, no = 1: closed {0,1}, open {2}, virtual {3}.
        let f_b = Array2::from_diag(&array![-1.0, -0.3, -0.1, 0.5]);
        let f_a = Array2::from_diag(&array![-1.1, -0.4, -0.2, 0.6]);
        let cands = aufbau_swap_candidates(&eye4(), &f_a, &f_b, 2, 1);
        assert_eq!(cands.len(), 2);
        assert_eq!(cands[0].kind, SwapKind::BetaClosedToOpen);
        assert_eq!((cands[0].from, cands[0].to), (1, 2));
        assert!((cands[0].koopmans - 0.2).abs() < 1e-12);
        assert_eq!(cands[1].kind, SwapKind::AlphaOpenToVirtual);
        assert_eq!((cands[1].from, cands[1].to), (2, 3));
        assert!((cands[1].koopmans - 0.8).abs() < 1e-12);
    }

    #[test]
    fn no_witness_candidates_without_an_open_shell() {
        let f = Array2::from_diag(&array![-1.0, -0.3, 0.2, 0.5]);
        assert!(aufbau_swap_candidates(&eye4(), &f, &f, 2, 0).is_empty());
    }
}
