//! Workspace planning helpers for cuEST (CPU-side, no device).
//!
//! cuEST allocates no GPU memory itself; the caller queries required sizes via
//! `WorkspaceQuery` and allocates persistent/temporary buffers. This module
//! captures those sizes so they can be reasoned about and tested without a
//! device.

/// A plan for cuEST workspace buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspacePlan {
    /// Bytes required for the persistent workspace (lives as long as the handle).
    pub persistent_bytes: usize,
    /// Bytes required for the temporary workspace (per call).
    pub temporary_bytes: usize,
}

impl WorkspacePlan {
    pub fn new(persistent_bytes: usize, temporary_bytes: usize) -> Self {
        Self {
            persistent_bytes,
            temporary_bytes,
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.persistent_bytes + self.temporary_bytes
    }

    pub fn fits_budget(&self, budget_bytes: usize) -> bool {
        self.total_bytes() <= budget_bytes
    }
}
