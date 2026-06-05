//! Coupled-Cluster methods for ferric.
//!
//! Spin-orbital RI-CCD, RI-CCSD, and the CCSD(T) perturbative-triples
//! correction, all validated against exact-integral / PySCF references.
//! The infrastructure reuses RI integrals from ferric-mp2.

use ndarray::{Array2, Array4};

pub mod ccd;
pub mod ccsd;
pub mod ccsd_t;
pub mod helpers;

pub struct CcConfig {
    pub frozen_core: usize,
    pub max_iter: usize,
    pub energy_conv: f64,
    pub diis_start: usize,
    pub diis_subspace: usize,
}

impl Default for CcConfig {
    fn default() -> Self {
        Self {
            frozen_core: 0,
            max_iter: 50,
            energy_conv: 1e-8,
            diis_start: 5,
            diis_subspace: 6,
        }
    }
}

pub struct CcResult {
    pub correlation_energy: f64,
    pub t1: Option<Array2<f64>>,
    pub t2: Array4<f64>,
}
