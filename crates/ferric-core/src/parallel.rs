//! Global parallelization context (Rayon + MPI).

#[cfg(feature = "mpi")]
use mpi::topology::SystemCommunicator;

/// A context representing the parallel execution environment.
///
/// Handles both single-node (Rayon) and multi-node (MPI) parallelization.
pub struct ParallelContext {
    #[cfg(feature = "mpi")]
    pub world: Option<SystemCommunicator>,
    pub rank: usize,
    pub size: usize,
}

impl ParallelContext {
    /// Create a new context. If MPI is enabled, initializes the world.
    pub fn new() -> Self {
        #[cfg(feature = "mpi")]
        {
            // Initialize MPI if possible
            // Note: In some environments, MPI is initialized by the caller.
            // For now, we assume MPI is already active if the feature is on.
            let world = mpi::initialize().map(|init| init.world());
            if let Some(w) = world {
                return Self {
                    rank: w.rank() as usize,
                    size: w.size() as usize,
                    world: Some(w),
                };
            }
        }
        
        Self {
            rank: 0,
            size: 1,
            #[cfg(feature = "mpi")]
            world: None,
        }
    }

    pub fn is_root(&self) -> bool {
        self.rank == 0
    }

    /// Run a task only on the root process.
    pub fn root_only<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce() -> R,
    {
        if self.is_root() {
            Some(f())
        } else {
            None
        }
    }
}

impl Default for ParallelContext {
    fn default() -> Self {
        Self::new()
    }
}
