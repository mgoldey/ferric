//! `finalize_mpi` really finalizes, once, and MPI cannot come back afterwards.
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
