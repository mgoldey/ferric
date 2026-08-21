//! Generic MP2-based double-hybrid DFT driver.
//!
//! Supports B2PLYP (Grimme, JCP 124, 034108, 2006) and DSD-PBEP86
//! (Kozuch & Martin, PCCP 13, 20104, 2011). The pattern:
//!
//! ```text
//! E_DH = E_KS + c_os * E_OS + c_ss * E_SS
//! ```
//!
//! where `E_OS`/`E_SS` are the opposite-spin and same-spin RI-MP2
//! correlation energies from `ri_mp2_spin_components`.

use crate::rimp2::{ri_mp2_spin_components, RiMp2Config, SpinComponents};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;

/// Configuration for an MP2-based double hybrid.
#[derive(Debug, Clone)]
pub struct Mp2DoubleHybridConfig {
    /// Opposite-spin MP2 correlation scaling.
    pub c_os: f64,
    /// Same-spin MP2 correlation scaling.
    pub c_ss: f64,
    /// Frozen-core orbitals for the MP2 step.
    pub frozen_core: usize,
    /// Memory budget for the 3-index MO transform (bytes). `None` → auto.
    pub memory_budget_bytes: Option<usize>,
}

/// Predefined double-hybrid parameter sets.
#[derive(Debug, Clone, Copy)]
pub enum DoubleHybridKind {
    B2plyp,
    DsdPbep86,
}

impl DoubleHybridKind {
    pub fn mp2_config(&self) -> Mp2DoubleHybridConfig {
        match self {
            Self::B2plyp => Mp2DoubleHybridConfig { c_os: 0.27, c_ss: 0.27, frozen_core: 0, memory_budget_bytes: None },
            Self::DsdPbep86 => Mp2DoubleHybridConfig { c_os: 0.56, c_ss: 0.29, frozen_core: 0, memory_budget_bytes: None },
        }
    }

    pub fn xc_name(&self) -> &'static str {
        match self {
            Self::B2plyp => "B2PLYP",
            Self::DsdPbep86 => "DSD-PBEP86",
        }
    }
}

/// Result of an MP2-based double-hybrid calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct Mp2DoubleHybridResult {
    pub total_energy: f64,
    pub e_ks: f64,
    pub spin_components: SpinComponents,
    pub e_corr_scaled: f64,
    pub c_os: f64,
    pub c_ss: f64,
}

impl std::fmt::Display for Mp2DoubleHybridResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Double hybrid total: {:.10} Ha (KS: {:.10}, MP2 corr: {:.10})",
            self.total_energy, self.e_ks, self.e_corr_scaled)
    }
}

/// Compute an MP2-based double-hybrid energy on a converged KS reference.
pub fn mp2_double_hybrid(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    ks: &ScfResult,
    cfg: &Mp2DoubleHybridConfig,
) -> Result<Mp2DoubleHybridResult, FerricError> {
    if !ks.converged {
        return Err(FerricError::ScfConvergence {
            iterations: ks.iterations,
            last_energy: ks.energy,
        });
    }

    let op = Operator::coulomb();
    let ri_cfg = RiMp2Config {
        frozen_core: cfg.frozen_core,
        memory_budget_bytes: cfg.memory_budget_bytes,
        ..Default::default()
    };

    let (sc, _b_ov) = ri_mp2_spin_components(mol, obs, dfbs, op, ks, &ri_cfg)?;

    let e_corr_scaled = cfg.c_os * sc.e_os + cfg.c_ss * sc.e_ss;

    Ok(Mp2DoubleHybridResult {
        total_energy: ks.energy + e_corr_scaled,
        e_ks: ks.energy,
        spin_components: sc,
        e_corr_scaled,
        c_os: cfg.c_os,
        c_ss: cfg.c_ss,
    })
}
