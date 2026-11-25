//! `finalize_mpi` really finalizes, once, on the initializing thread only, and
//! MPI cannot come back afterwards; MPI runs at `MPI_THREAD_SERIALIZED`.
//!
//! Its own test binary: finalizing MPI ends MPI for the whole process, so no
//! other test may share it. Runs as a singleton (no `mpirun` needed):
//! `cargo test -p ferric-core --features mpi --test mpi_finalize`.
#![cfg(feature = "mpi")]

use ferric_core::parallel::{finalize_mpi, ParallelContext};

/// `(MPI_Initialized, MPI_Finalized)`; both are callable at any time.
fn mpi_state() -> (bool, bool) {
    let (mut init, mut fin) = (0, 0);
    // SAFETY: the two query functions take an out-pointer and are valid
    // before MPI_Init and after MPI_Finalize.
    unsafe {
        mpi::ffi::MPI_Initialized(&mut init);
        mpi::ffi::MPI_Finalized(&mut fin);
    }
    (init != 0, fin != 0)
}

#[test]
fn finalize_runs_once_and_mpi_cannot_be_reinitialized() {
    // Before any initialization it is a no-op.
    finalize_mpi();
    let ctx = ParallelContext::new();
    assert_eq!((ctx.rank, ctx.size), (0, 1), "a singleton world");
    assert_eq!(mpi_state(), (true, false));
    // The collectives run from the rank's rayon pool, so MPI must be up at
    // MPI_THREAD_SERIALIZED or above, not rsmpi's default Single.
    let level = mpi::environment::threading_support();
    assert!(
        matches!(level, mpi::Threading::Serialized | mpi::Threading::Multiple),
        "MPI initialized at {level:?}"
    );
    // MPI_Finalize belongs to the initializing thread: from another thread
    // finalize_mpi refuses and MPI stays up for the real call below.
    let elsewhere = std::thread::spawn(|| std::panic::catch_unwind(finalize_mpi)).join();
    assert!(
        matches!(elsewhere, Ok(Err(_))),
        "finalize_mpi from another thread must panic"
    );
    assert_eq!(mpi_state(), (true, false), "a foreign thread finalized MPI");
    finalize_mpi();
    assert_eq!(mpi_state(), (true, true), "finalize_mpi left MPI running");
    // A second call is a no-op, not a double MPI_Finalize (which would abort).
    finalize_mpi();
    let again = std::panic::catch_unwind(ParallelContext::new);
    assert!(
        again.is_err(),
        "ParallelContext::new after finalize_mpi must panic"
    );
}
