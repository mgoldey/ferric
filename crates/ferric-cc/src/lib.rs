//! Coupled-Cluster methods for ferric.
//!
//! Spin-orbital RI-CCD, RI-CCSD, and the CCSD(T) perturbative-triples
//! correction, all validated against exact-integral / PySCF references.
//! The infrastructure reuses RI integrals from ferric-mp2.

use ndarray::{Array2, Array4};

/// Spin-orbital coupled-cluster doubles (CCD).
pub mod ccd;
/// Spin-orbital CCSD (coupled-cluster singles and doubles).
pub mod ccsd;
/// Spin-adapted closed-shell CCSD (spatial orbitals, ~8-10x faster).
pub mod ccsd_closed_shell;
/// Spin-orbital CCSD(T) perturbative triples (streaming per-triple-block).
pub mod ccsd_t;
/// Spin-adapted closed-shell CCSD(T) (spatial orbitals, ~10-40x faster).
pub mod ccsd_t_closed_shell;
/// DLPNO-CCSD (domain-based local pair natural orbital CCSD).
pub mod dlpno_ccsd;
/// DLPNO-CCSD amplitude update kernel.
pub mod dlpno_ccsd_kernel;
/// DLPNO-CCSD(T) perturbative triples with local domains.
pub mod dlpno_ccsd_t;
/// DLPNO-CCSD(T) triples kernel.
pub mod dlpno_ccsd_t_kernel;
/// DLPNO-CCSD(T) virtual-domain machinery.
pub mod dlpno_ccsd_t_virtual;
/// DLPNO-CCSD virtual-domain construction.
pub mod dlpno_ccsd_virtual;
/// DLPNO-LinLCCD (linearized local coupled-cluster doubles).
pub mod dlpno_linlccd;
/// Double-hybrid DFT: wB97X-L-V and similar post-KS correlation.
pub mod double_hybrid;
/// Shared CC helpers: MO integrals, amplitude utilities.
pub mod helpers;
/// Linearized LCCD (LinLCCD) from RI intermediates.
pub mod linlccd;
/// LinLCCD amplitude update kernel.
pub mod linlccd_amplitude;
/// LinLCCD via exact (non-RI) integrals (validation only).
pub mod linlccd_exact;
/// Unrestricted LinLCCD (α/β spin channels).
pub mod linlccd_u;

/// Local-correlation machinery, re-exported from `ferric-mp2`.
///
/// These live in `ferric-mp2` rather than here because MP2 and RPA need them too
/// and `ferric-mp2` is UPSTREAM of this crate — a module here would be unreachable
/// from either. Re-exported so `ferric_cc::pair_domains::…` keeps resolving.
pub use ferric_mp2::pair_domains;
pub use ferric_mp2::pair_energy_screen;
pub use ferric_mp2::local_pno as pno;

/// Configuration for coupled-cluster iterations (CCD, CCSD).
#[derive(Debug, Clone, PartialEq)]
pub struct CcConfig {
    pub frozen_core: usize,
    pub max_iter: usize,
    pub energy_conv: f64,
    pub diis_start: usize,
    pub diis_subspace: usize,
    /// Optional resident-bytes ceiling for RI integral transforms feeding the
    /// CC amplitude equations. `None` → resolved via
    /// [`ferric_core::memory::resolve_budget_bytes`]. Currently wiring-only:
    /// the dense CC contractions dominate memory, but the field lets callers
    /// propagate the unified budget uniformly.
    pub memory_budget_bytes: Option<usize>,
}

impl Default for CcConfig {
    fn default() -> Self {
        Self {
            frozen_core: 0,
            max_iter: 50,
            energy_conv: 1e-8,
            diis_start: 5,
            diis_subspace: 6,
            memory_budget_bytes: None,
        }
    }
}

impl CcConfig {
    /// Set the number of frozen core orbitals.
    pub fn with_frozen_core(mut self, n: usize) -> Self {
        self.frozen_core = n;
        self
    }
    /// Set the maximum number of CC iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    /// Set the energy convergence threshold (Ha).
    pub fn with_energy_conv(mut self, thresh: f64) -> Self {
        self.energy_conv = thresh;
        self
    }
    /// Set the memory budget in bytes.
    pub fn with_memory_budget_bytes(mut self, bytes: usize) -> Self {
        self.memory_budget_bytes = Some(bytes);
        self
    }
}

/// Converged coupled-cluster result: correlation energy and amplitudes.
#[derive(Debug, Clone)]
#[must_use]
pub struct CcResult {
    pub correlation_energy: f64,
    pub t1: Option<Array2<f64>>,
    pub t2: Array4<f64>,
}

impl std::fmt::Display for CcResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let method = if self.t1.is_some() { "CCSD" } else { "CCD" };
        write!(f, "{} correlation: {:.10} Ha", method, self.correlation_energy)
    }
}
