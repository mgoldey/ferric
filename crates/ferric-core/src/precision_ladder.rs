//! The SCF precision ladder shared by every mixed-precision Fock-build kernel.
//!
//! A mixed-precision Fock contribution (the device `xc-vxc` V_xc, the `dfj-pack`
//! RI-J, the `dfk-occ` DF-K) is allowed only while the SCF density still moves:
//! a build is mixed when mixed precision was requested, a previous build of the
//! same SCF exists, and some spin density moved by more than a threshold τ
//! (RMS over the AO matrix) since that build. Every other build is f64: the
//! first build of every SCF, every one-shot build, and every build once the
//! density has settled. The ladder alone does not make a converged exit f64
//! (the SCF's exit gate is a different test), so each user also keeps a record
//! of its last mixed build ([`PrecisionLadder::keep_mixed`](crate::precision_ladder::PrecisionLadder::keep_mixed)) and, at a
//! converged exit, replaces it by an f64 rebuild ([`PrecisionLadder::take_mixed`](crate::precision_ladder::PrecisionLadder::take_mixed)):
//! the XC evaluators swap the mixed V_xc for an f64 one before the final
//! diagonalisation, the SCF J/K ladder (`ferric_scf::jk_ladder`) refuses the
//! exit and runs the remaining iterations in f64.
//!
//! This is the ladder `ferric-dft`'s `ks.rs` introduced for `xc-vxc`, extracted
//! without a change of arithmetic: [`PrecisionLadder::step_with`](crate::precision_ladder::PrecisionLadder::step_with) is the former
//! `MixedLadder::step_with` and [`rms_diff`](crate::precision_ladder::rms_diff) the former `ks.rs` helper, so the
//! XC path is bit-identical.
//!
//! Counters ([`counters`](crate::precision_ladder::counters)) are process-wide and monotone, per [`LadderSite`](crate::precision_ladder::LadderSite):
//! the builds the ladder was consulted for while mixed was requested, the
//! builds that kept a mixed contribution, and the converged exits whose last
//! mixed contribution was rebuilt in f64.
//!
//! The per-kernel provenance counters are separate
//! ([`gpu::stats`](mod@crate::gpu::stats)): each ladder's user reports to them, as
//! `LadderSite::Jk` here covers two kernels.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use ndarray::Array2;

/// Which ladder a counter belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LadderSite {
    /// The device `xc-vxc` V_xc (`ferric_dft::ks`).
    XcVxc = 0,
    /// The SCF J/K build: `dfj-pack` RI-J and `dfk-occ` DF-K (`ferric_scf::jk_ladder`).
    Jk = 1,
}

const NSITE: usize = 2;
static BUILDS_TOTAL: [AtomicU64; NSITE] = [const { AtomicU64::new(0) }; NSITE];
static BUILDS_MIXED: [AtomicU64; NSITE] = [const { AtomicU64::new(0) }; NSITE];
static EXIT_REBUILDS: [AtomicU64; NSITE] = [const { AtomicU64::new(0) }; NSITE];

/// The ladder counters of one site at one instant.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LadderCounters {
    /// Builds the ladder was consulted for while mixed precision was requested.
    pub builds_total: u64,
    /// Builds that ran (at least partly) mixed and were kept for the exit rebuild.
    pub builds_mixed: u64,
    /// Converged exits whose last mixed contribution was rebuilt in f64.
    pub exit_rebuilt_f64: u64,
}

impl LadderCounters {
    /// The counts accumulated since `before` (saturating).
    pub fn since(&self, before: &LadderCounters) -> LadderCounters {
        LadderCounters {
            builds_total: self.builds_total.saturating_sub(before.builds_total),
            builds_mixed: self.builds_mixed.saturating_sub(before.builds_mixed),
            exit_rebuilt_f64: self
                .exit_rebuilt_f64
                .saturating_sub(before.exit_rebuilt_f64),
        }
    }
}

/// The process-wide counters of `site`.
pub fn counters(site: LadderSite) -> LadderCounters {
    let i = site as usize;
    LadderCounters {
        builds_total: BUILDS_TOTAL[i].load(Ordering::Relaxed),
        builds_mixed: BUILDS_MIXED[i].load(Ordering::Relaxed),
        exit_rebuilt_f64: EXIT_REBUILDS[i].load(Ordering::Relaxed),
    }
}

/// Mutation seams for the ladder tests (`test-seams` feature). Never set by a
/// shipped build.
#[cfg(feature = "test-seams")]
#[doc(hidden)]
pub mod seams {
    use std::sync::atomic::AtomicBool;
    /// Mutant: the exit criterion never fires; every build with a previous
    /// build of the same shape is mixed whatever the density did.
    pub static NEVER_EXIT: AtomicBool = AtomicBool::new(false);
    /// Mutant: no exit rebuild; [`super::PrecisionLadder::take_mixed`] always
    /// answers `None`, so a converged exit keeps its last mixed contribution.
    pub static NO_EXIT_REBUILD: AtomicBool = AtomicBool::new(false);
}

fn never_exit() -> bool {
    #[cfg(feature = "test-seams")]
    {
        seams::NEVER_EXIT.load(Ordering::Relaxed)
    }
    #[cfg(not(feature = "test-seams"))]
    {
        false
    }
}

fn no_exit_rebuild() -> bool {
    #[cfg(feature = "test-seams")]
    {
        seams::NO_EXIT_REBUILD.load(Ordering::Relaxed)
    }
    #[cfg(not(feature = "test-seams"))]
    {
        false
    }
}

/// Per-SCF memory of the last Fock build, deciding whether the next build may
/// run mixed and holding the last mixed build (payload `T`) for the exit
/// rebuild. Interior mutability (`Mutex`) because the XC evaluators are reached
/// through `&self`; every lock is uncontended (one SCF thread).
#[derive(Debug)]
pub struct PrecisionLadder<T> {
    site: LadderSite,
    prev: Mutex<Vec<Array2<f64>>>,
    last_mixed: Mutex<Option<T>>,
}

impl<T> PrecisionLadder<T> {
    /// An empty ladder (the next build is a first build) counting under `site`.
    pub const fn new(site: LadderSite) -> Self {
        Self {
            site,
            prev: Mutex::new(Vec::new()),
            last_mixed: Mutex::new(None),
        }
    }

    /// Record the spin densities `ds` of this build and say whether it may run
    /// mixed: `requested`, a previous build of the same shape (same number of
    /// spins, same `nbf`), and some spin with [`rms_diff`] `> thresh` (exactly
    /// τ is not "more than τ"; a NaN never selects mixed). Nothing is recorded
    /// when mixed precision is not requested (no nbf² copies). Every step
    /// forgets the previous build's mixed record.
    pub fn step_with(&self, requested: bool, thresh: f64, ds: &[&Array2<f64>]) -> bool {
        *self.last_mixed.lock().unwrap_or_else(|e| e.into_inner()) = None;
        if !requested {
            return false;
        }
        BUILDS_TOTAL[self.site as usize].fetch_add(1, Ordering::Relaxed);
        let mut prev = self.prev.lock().unwrap_or_else(|e| e.into_inner());
        let same_shape =
            prev.len() == ds.len() && prev.iter().zip(ds).all(|(p, d)| p.dim() == d.dim());
        let moving = same_shape
            && (never_exit() || prev.iter().zip(ds).any(|(p, d)| rms_diff(p, d) > thresh));
        if same_shape {
            for (p, d) in prev.iter_mut().zip(ds) {
                p.assign(d);
            }
        } else {
            *prev = ds.iter().map(|d| (*d).clone()).collect();
        }
        moving
    }

    /// Forget every earlier build: the next one has no previous build.
    pub fn reset(&self) {
        self.prev.lock().unwrap_or_else(|e| e.into_inner()).clear();
        *self.last_mixed.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// Keep the record of the mixed build just made.
    pub fn keep_mixed(&self, payload: T) {
        BUILDS_MIXED[self.site as usize].fetch_add(1, Ordering::Relaxed);
        *self.last_mixed.lock().unwrap_or_else(|e| e.into_inner()) = Some(payload);
    }

    /// At a converged exit: the last build's mixed record, if it was mixed
    /// (and forget it). `Some` obliges the caller to rebuild that contribution
    /// in f64 before returning, and counts one exit rebuild.
    pub fn take_mixed(&self) -> Option<T> {
        let taken = self
            .last_mixed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if no_exit_rebuild() {
            return None;
        }
        if taken.is_some() {
            EXIT_REBUILDS[self.site as usize].fetch_add(1, Ordering::Relaxed);
        }
        taken
    }
}

/// `sqrt(Σ (a − b)² / len)`; NaN propagates (and then compares false, so a
/// NaN density never selects the mixed kernel).
pub fn rms_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let n = a.len().max(1) as f64;
    let ss = a
        .iter()
        .zip(b.iter())
        .fold(0.0_f64, |acc, (x, y)| acc + (x - y) * (x - y));
    (ss / n).sqrt()
}

#[cfg(test)]
mod ladder_tests {
    // Moved verbatim from `ferric-dft`'s `ks.rs` (`MixedLadder` is now this type).
    use super::*;

    type MixedLadder = PrecisionLadder<()>;
    impl Default for MixedLadder {
        fn default() -> Self {
            Self::new(LadderSite::XcVxc)
        }
    }

    fn m(v: f64) -> Array2<f64> {
        Array2::from_elem((3, 3), v)
    }

    #[test]
    fn first_build_is_f64_and_only_a_moving_density_is_mixed() {
        let l = MixedLadder::default();
        let t = 1e-5;
        assert!(!l.step_with(true, t, &[&m(0.0)]), "no previous build: f64");
        assert!(l.step_with(true, t, &[&m(1e-3)]), "moved by 1e-3: mixed");
        assert!(
            !l.step_with(true, t, &[&m(1e-3 + 1e-6)]),
            "moved by 1e-6: f64"
        );
        assert!(!l.step_with(true, t, &[&m(1e-3 + 1e-6)]), "unchanged: f64");
        // Exactly τ is not "more than τ".
        let l = MixedLadder::default();
        l.step_with(true, 0.25, &[&m(0.0)]);
        assert!(!l.step_with(true, 0.25, &[&m(0.25)]));
    }

    #[test]
    fn any_moving_spin_counts_and_a_shape_change_restarts() {
        let l = MixedLadder::default();
        let t = 1e-5;
        l.step_with(true, t, &[&m(0.0), &m(0.0)]);
        assert!(l.step_with(true, t, &[&m(0.0), &m(1.0)]), "beta moved");
        // A different number of spins, or a different nbf, has no previous build.
        assert!(!l.step_with(true, t, &[&m(5.0)]));
        assert!(!l.step_with(true, t, &[&Array2::from_elem((4, 4), 9.0)]));
        assert!(l.step_with(true, t, &[&Array2::from_elem((4, 4), 0.0)]));
    }

    #[test]
    fn not_requested_records_nothing_and_nan_is_never_mixed() {
        let l = MixedLadder::default();
        assert!(!l.step_with(false, 1e-5, &[&m(0.0)]));
        assert!(
            l.prev.lock().unwrap().is_empty(),
            "no copy when not requested"
        );
        assert!(!l.step_with(false, 1e-5, &[&m(1.0)]));
        let l = MixedLadder::default();
        l.step_with(true, 1e-5, &[&m(0.0)]);
        assert!(!l.step_with(true, 1e-5, &[&m(f64::NAN)]));
    }

    /// Catches: a step that keeps the previous build's mixed record (an exit
    /// would then rebuild a contribution the last build never made), a
    /// `take_mixed` that does not forget, and a `reset` that keeps it.
    #[test]
    fn the_mixed_record_lives_for_exactly_one_step() {
        let l = PrecisionLadder::<u8>::new(LadderSite::Jk);
        l.step_with(true, 1e-5, &[&m(0.0)]);
        l.keep_mixed(7);
        assert_eq!(l.take_mixed(), Some(7));
        assert_eq!(l.take_mixed(), None, "taken twice");
        l.keep_mixed(8);
        l.step_with(true, 1e-5, &[&m(0.0)]);
        assert_eq!(l.take_mixed(), None, "survived a step");
        l.keep_mixed(9);
        l.reset();
        assert_eq!(l.take_mixed(), None, "survived a reset");
        // A step that is not requested also forgets it.
        l.keep_mixed(10);
        l.step_with(false, 1e-5, &[&m(0.0)]);
        assert_eq!(l.take_mixed(), None);
    }
}
