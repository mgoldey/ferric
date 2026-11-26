//! Pair-list conversion helpers for cuEST (CPU-side, no device).

/// A device-independent description of an AO pair list screening threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct PairListPlan {
    pub threshold: f64,
}

impl PairListPlan {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.threshold.is_finite() || self.threshold < 0.0 {
            return Err("threshold must be finite and >= 0".into());
        }
        Ok(())
    }
}
