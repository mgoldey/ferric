//! Shared test helpers for the ferric workspace.
//!
//! This crate is `publish = false` and a **dev-dependency only**: it must
//! appear under `[dev-dependencies]` in other crates' manifests, never under
//! `[dependencies]` (`scripts/check_testkit_dev_only.py` fails CI otherwise).
//!
//! # Dependency direction
//!
//! `ferric-testkit` depends on `ferric-core` only. So `ferric-core` itself,
//! and any crate `ferric-core` depends on, cannot use it: a dev-dependency
//! back onto a crate's own dependency would give the unit tests two copies of
//! `ferric-core` (the one under test and the one the testkit links). Every
//! crate above `ferric-core` can. Helpers that need a higher crate
//! (`ferric-integrals`, `ferric-scf`) do not belong here for that reason;
//! keep those next to the tests that use them.
//!
//! # What lives here
//!
//! Only helpers that were copied byte for byte into three or more test files.
//! Tolerances are never baked in: every comparison helper returns a number
//! and the call site asserts against its own bound.
//!
//! * [`pool`]: the process-global memory-pool test slot (`CleanSlot`).
//! * [`compare`]: max-abs norms and differences of `Array2<f64>`.
//! * [`mod@reference`]: typed accessors for the JSON reference files under
//!   `testdata/reference`.
//! * [`paths`]: the workspace root and `testdata` locations.
//! * [`molecules`]: the standard molecules loaded from `testdata/molecules`.

pub mod compare;
pub mod molecules;
pub mod paths;
pub mod pool;
pub mod reference;
