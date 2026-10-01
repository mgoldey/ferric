//! Per-thread 2e integral-engine pool.
//!
//! Lives in `ferric-integrals` (next to [`crate::engine::Engine`]) rather than `ferric-scf` so
//! that lower-level integral code — Schwarz bounds, 3-index drivers — can use it
//! too. `ferric_scf::engine_pool` re-exports it, so existing callers are
//! unaffected. It was originally written for the direct (exact-ERI) Fock
//! builders; `schwarz.rs` could not reuse it while it lived in the higher crate,
//! and hand-rolled a per-chunk `map_init` that hit exactly the pathology
//! described below (123 ms at RAYON=1 vs 3211 ms at RAYON=12).
//!
//! ## Why this exists
//!
//! libint2 engine construction is **expensive** (it allocates scratch sized by
//! the basis's max angular momentum and max primitive count, and builds
//! recurrence tables) and is **serialized behind a global mutex** in the shim
//! (libint2 `Engine` ctors are not thread-safe). The direct builders parallelize
//! with `into_par_iter().fold(|| Engine::new_2e(...), ...)`. Rayon calls a
//! `fold` init closure **once per work-chunk, not once per thread** — for a small
//! molecule the shell-pair list splits into dozens of chunks, so the engine was
//! constructed dozens of times per Fock build, every one of them queueing on the
//! global ctor mutex. With N threads all stuck on that mutex, wall time
//! *exploded* (PH3/aug-cc-pVDZ: 9.6 s at RAYON=1 vs >120 s at RAYON=8 — the
//! threads spent all their time contending, not computing). The effect is worst
//! for high-angular-momentum heavy-element bases (Si, P, S, Cl) where each
//! construction is slowest.
//!
//! ## The fix
//!
//! Build **exactly one engine per rayon worker thread** up front (so the mutex is
//! hit at most `num_threads` times total), store them in a pool indexed by
//! `rayon::current_thread_index()`, and have each parallel task borrow its
//! thread's engine via a `Mutex`. Construction count drops from O(chunks) to
//! O(threads), killing the contention while keeping full parallelism.

use crate::basis_bridge::PreparedBasis;
use crate::engine::Engine;
use crate::operator::Operator;
use ferric_core::FerricError;
use std::sync::Mutex;

/// libint2 engine precision for the two-electron integrals that build SCF
/// energies and Fock matrices (J, K, LinK, CFMM, Newton/stability response).
///
/// libint2 drops primitive products whose estimated contribution falls below
/// this value, and the dropped mass accumulates over contracted core shells.
/// Measured on one full J/K build at a fixed converged density (cc-pVDZ),
/// against precision 0 (no primitive screening):
///
/// | precision | worst E_J error | worst Σ D·K error | build cost vs 1e-14 |
/// |---|---:|---:|---|
/// | 1e-14 | 7.0e-10 (CCl4) | 1.5e-9 (CCl4) | 1 |
/// | 1e-16 | 4.6e-13 | 1.4e-10 (CCl4) | 0.99–1.16 |
/// | 1e-18 | 2.3e-13 | 1.7e-13 | 1.12–1.29 |
/// | 1e-20 | 0 | 0 | 1.09–1.40 |
/// | 0 | – | – | 1.8–5.7 |
///
/// (benzene, CS2, CCl4; the high end of each cost range is CCl4.) On free atoms
/// in aug-cc-pVDZ, 1e-14 put E_J off by up to 6.0e-9 Ha (Al; Na 5.3e-9) against
/// PySCF's unscreened integrals, and 1e-18 still left Na at 3e-11; 1e-20 matches
/// to ≤1e-13. 1e-20 is the loosest value that reaches double precision on every
/// system measured.
pub const ERI_PRECISION: f64 = 1e-20;

/// The precision the SCF J/K engines actually use: [`ERI_PRECISION`] unless
/// overridden.
///
/// Precedence, as for every [`ferric_core::config::ConfigVar`]: an explicit
/// value set with [`set_eri_precision`] (the CLI's `[scf] eri_precision`) beats
/// the `FERRIC_ERI_PRECISION` environment variable, which beats the default.
/// Allowed values are `0 ≤ p ≤ 1e-8`; 0 turns primitive screening off (exact,
/// 1.8–5.7x slower per J/K build). A malformed or out-of-range environment
/// value warns and falls back to the default, because this is read where no
/// `Result` can propagate; an explicit value is validated when it is set.
pub fn eri_precision() -> f64 {
    let bits = ERI_PRECISION_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    let explicit = (bits != UNSET).then(|| f64::from_bits(bits));
    ERI_PRECISION_VAR
        .resolve(explicit, ferric_core::config::env_lookup)
        .map(|r| r.value)
        .unwrap_or_else(|e| {
            eprintln!("[config] FERRIC_ERI_PRECISION: {e}; using default {ERI_PRECISION:e}");
            ERI_PRECISION
        })
}

/// Set (or with `None`, clear) the process-wide explicit precision that
/// [`eri_precision`] returns ahead of the environment. Errors on a value
/// outside `0 ≤ p ≤ 1e-8`.
pub fn set_eri_precision(value: Option<f64>) -> Result<(), String> {
    let bits = match value {
        Some(v) => {
            (ERI_PRECISION_VAR.validate)(&v).map_err(|e| format!("eri_precision {v:e}: {e}"))?;
            v.to_bits()
        }
        None => UNSET,
    };
    ERI_PRECISION_OVERRIDE.store(bits, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// Sentinel for "no explicit precision" (a NaN payload no caller can set,
/// because validation rejects NaN).
const UNSET: u64 = u64::MAX;
static ERI_PRECISION_OVERRIDE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(UNSET);

/// The descriptor behind [`eri_precision`].
pub static ERI_PRECISION_VAR: ferric_core::config::ConfigVar<f64> =
    ferric_core::config::ConfigVar {
        env_name: "FERRIC_ERI_PRECISION",
        default: ERI_PRECISION,
        parse: |s| s.parse::<f64>().map_err(|e| e.to_string()),
        validate: |v| {
            (v.is_finite() && (0.0..=1e-8).contains(v))
                .then_some(())
                .ok_or_else(|| "must be finite with 0 <= p <= 1e-8".to_string())
        },
    };

/// A pool of 2e engines, one slot per rayon worker thread (plus one spare for
/// the calling thread / non-rayon contexts at index `len-1`).
pub struct EnginePool {
    engines: Vec<Mutex<Engine>>,
}

impl std::fmt::Debug for EnginePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnginePool")
            .field("n_engines", &self.engines.len())
            .finish_non_exhaustive()
    }
}

impl EnginePool {
    /// Construct `num_threads + 1` engines (serialized, but bounded). The `+1`
    /// covers `current_thread_index() == None` (work run on a non-pool thread).
    pub fn new(op: Operator, prep: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let n = rayon::current_num_threads().max(1) + 1;
        let mut engines = Vec::with_capacity(n);
        for _ in 0..n {
            engines.push(Mutex::new(Engine::new_2e(op, prep, precision)?));
        }
        Ok(EnginePool { engines })
    }

    /// Run `f` with this thread's engine. Indexed by `current_thread_index()`;
    /// falls back to the spare slot for non-rayon threads. The per-slot `Mutex`
    /// is uncontended in practice (one thread maps to one slot), so the lock is
    /// effectively free — it exists only to satisfy `&mut Engine` borrowing.
    #[inline]
    pub fn with<R>(&self, f: impl FnOnce(&mut Engine) -> R) -> R {
        let idx = rayon::current_thread_index().unwrap_or(self.engines.len() - 1);
        // Guard against an index beyond the pool (shouldn't happen, but be safe).
        let slot = idx.min(self.engines.len() - 1);
        let mut eng = self.engines[slot].lock().unwrap();
        f(&mut eng)
    }
}

#[cfg(test)]
mod eri_precision_tests {
    use super::{ERI_PRECISION, ERI_PRECISION_VAR};

    #[test]
    fn explicit_beats_env_beats_default() {
        let env = |v: Option<&'static str>| {
            move |k: &str| {
                (k == "FERRIC_ERI_PRECISION")
                    .then_some(v)
                    .flatten()
                    .map(str::to_string)
            }
        };
        assert_eq!(
            ERI_PRECISION_VAR.resolve(None, env(None)).unwrap().value,
            ERI_PRECISION
        );
        assert_eq!(
            ERI_PRECISION_VAR
                .resolve(None, env(Some("1e-16")))
                .unwrap()
                .value,
            1e-16
        );
        assert_eq!(
            ERI_PRECISION_VAR
                .resolve(Some(0.0), env(Some("1e-16")))
                .unwrap()
                .value,
            0.0
        );
    }

    #[test]
    fn out_of_range_and_malformed_are_errors() {
        let env = |v: &'static str| move |_: &str| Some(v.to_string());
        for bad in ["1e-6", "-1e-20", "nan", "inf", "abc"] {
            assert!(
                ERI_PRECISION_VAR.resolve(None, env(bad)).is_err(),
                "{bad} accepted"
            );
        }
        assert!(ERI_PRECISION_VAR.resolve(Some(1e-7), env("1e-16")).is_err());
    }
}
