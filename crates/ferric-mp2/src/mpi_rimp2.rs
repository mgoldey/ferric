//! MPI-distributed RI-MP2 (requires `mpi` feature).
//!
//! Stub — will be implemented when MPI integration is complete.

#[cfg(feature = "mpi")]
mod inner {
    use ferric_core::FerricError;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::ScfResult;
    use mpi::traits::*;

    pub struct MpiMp2Result {
        pub mp2_corr: f64,
        pub total_energy: f64,
    }

    // NOTE: mpi 0.7 renamed the world communicator type to `SimpleCommunicator`
    // (`SystemCommunicator` was dropped). This stub is still T9 (unimplemented);
    // only the type name is updated so the workspace compiles under `--features mpi`.
    pub fn run_mpi_ri_mp2(
        _world: &mpi::topology::SimpleCommunicator,
        _obs: &PreparedBasis,
        _dfbs: &PreparedBasis,
        _op: Operator,
        _rhf: &ScfResult,
        _frozen_core: usize,
    ) -> Result<MpiMp2Result, FerricError> {
        Err(FerricError::Libint("MPI RI-MP2 not yet implemented".into()))
    }
}

#[cfg(feature = "mpi")]
pub use inner::*;
