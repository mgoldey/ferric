//! Grid conversion helpers for cuEST (CPU-side, no device).
//!
//! cuEST owns its own molecular grid (`cuestMolecularGrid_t`). These helpers
//! capture the ferric-side grid parameters so they can be validated and
//! round-tripped without a device; the actual cuEST grid handle is created in
//! a later phase (device-required).

/// A device-independent description of a molecular integration grid.
#[derive(Debug, Clone, PartialEq)]
pub struct GridPlan {
    pub n_radial: usize,
    pub n_angular: usize,
    pub prune: Option<String>,
    pub weight_cut: Option<f64>,
}

impl GridPlan {
    pub fn new(n_radial: usize, n_angular: usize) -> Self {
        Self {
            n_radial,
            n_angular,
            prune: None,
            weight_cut: None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_radial == 0 {
            return Err("n_radial must be > 0".into());
        }
        if self.n_angular == 0 {
            return Err("n_angular must be > 0".into());
        }
        if let Some(c) = self.weight_cut {
            if !c.is_finite() || c < 0.0 {
                return Err("weight_cut must be finite and >= 0".into());
            }
        }
        Ok(())
    }

    /// Total number of (radial, angular) point slots (upper bound; pruning
    /// lowers the actual count on-device).
    pub fn slot_count(&self, n_atoms: usize) -> usize {
        n_atoms * self.n_radial * self.n_angular
    }
}
