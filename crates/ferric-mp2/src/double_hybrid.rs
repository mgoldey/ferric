//! Generic MP2-based double-hybrid DFT driver.
//!
//! Supports B2PLYP (Grimme, JCP 124, 034108, 2006) and DSD-PBEP86 without a
//! dispersion term (Kozuch & Martin, J. Comput. Chem. 34, 2327, 2013: 0.72 HF,
//! 0.28 PBE exchange, 0.44 P86 correlation, `c_os = 0.51`, `c_ss = 0.36`),
//! plus their D3(BJ) sets ([`DoubleHybridKind`](crate::double_hybrid::DoubleHybridKind)). A closed-shell Kohn-Sham
//! reference takes [`mp2_double_hybrid`](crate::double_hybrid::mp2_double_hybrid); an unrestricted one takes
//! [`u_mp2_double_hybrid`](crate::double_hybrid::u_mp2_double_hybrid). The pattern:
//!
//! ```text
//! E_DH = E_KS + c_os * E_OS + c_ss * E_SS
//! ```
//!
//! where `E_OS`/`E_SS` are the opposite-spin and same-spin RI-MP2
//! correlation energies from `ri_mp2_spin_components` (closed shell) or
//! `u_ri_mp2` (unrestricted: `E_OS = E_alpha-beta`, `E_SS = E_alpha-alpha +
//! E_beta-beta`).

use crate::rimp2::{ri_mp2_spin_components, RiMp2Config, SpinComponents};
use crate::u_rimp2::u_ri_mp2;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::PreparedAux;
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
///
/// The `D3bj` variants name the published set that pairs with a D3(BJ) term:
/// `B2plypD3bj` has B2PLYP's coefficients (Goerigk and Grimme, PCCP 13, 6670
/// (2011), fit D3(BJ) to the unchanged B2PLYP), while `DsdPbep86D3bj` is the
/// refit DSD-PBEP86 set of Kozuch and Martin (2013), whose HF fraction and PT2
/// scalings differ from the no-dispersion set. The D3(BJ) energy itself is a
/// geometry term the caller adds (`ferric_d3`); see [`Self::d3bj_functional`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoubleHybridKind {
    /// B2PLYP: `c_os = c_ss = 0.27`.
    B2plyp,
    /// B2PLYP with its D3(BJ) term; the same coefficients as `B2plyp`.
    B2plypD3bj,
    /// DSD-PBEP86 without dispersion: `c_os = 0.51`, `c_ss = 0.36`.
    DsdPbep86,
    /// DSD-PBEP86-D3(BJ), the refit set: 0.69 HF, 0.31 PBE exchange, 0.44 P86
    /// correlation, `c_os = 0.52`, `c_ss = 0.22`.
    DsdPbep86D3bj,
}

impl DoubleHybridKind {
    /// MP2 scaling for this parameter set, with no frozen core and an
    /// auto-sized memory budget.
    pub fn mp2_config(&self) -> Mp2DoubleHybridConfig {
        let (c_os, c_ss) = match self {
            Self::B2plyp | Self::B2plypD3bj => (0.27, 0.27),
            Self::DsdPbep86 => (0.51, 0.36),
            Self::DsdPbep86D3bj => (0.52, 0.22),
        };
        Mp2DoubleHybridConfig {
            c_os,
            c_ss,
            frozen_core: 0,
            memory_budget_bytes: None,
        }
    }

    /// Functional name for the KS reference (`"B2PLYP"`, `"DSD-PBEP86"` or
    /// `"DSD-PBEP86-D3BJ"`); the Python binding passes it as the KS config's
    /// `xc`.
    pub fn xc_name(&self) -> &'static str {
        match self {
            Self::B2plyp | Self::B2plypD3bj => "B2PLYP",
            Self::DsdPbep86 => "DSD-PBEP86",
            Self::DsdPbep86D3bj => "DSD-PBEP86-D3BJ",
        }
    }

    /// Name of this set's anchor in
    /// `ferric_core::certificate::anchors::ENERGY_ANCHORS`. The D3(BJ)
    /// variants share their base set's name; the certificate scope checks the
    /// coefficients and the dispersion before attaching it.
    pub fn anchor_name(&self) -> &'static str {
        match self {
            Self::B2plyp | Self::B2plypD3bj => "B2PLYP",
            Self::DsdPbep86 | Self::DsdPbep86D3bj => "DSD-PBEP86",
        }
    }

    /// The `method.kind` name of the base set, without the dispersion suffix.
    pub fn base_kind(&self) -> &'static str {
        match self {
            Self::B2plyp | Self::B2plypD3bj => "b2plyp",
            Self::DsdPbep86 | Self::DsdPbep86D3bj => "dsd-pbep86",
        }
    }

    /// The same functional with its D3(BJ) term: B2PLYP keeps its
    /// coefficients, DSD-PBEP86 switches to the refit set.
    pub fn with_d3bj(&self) -> Self {
        match self {
            Self::B2plyp | Self::B2plypD3bj => Self::B2plypD3bj,
            Self::DsdPbep86 | Self::DsdPbep86D3bj => Self::DsdPbep86D3bj,
        }
    }

    /// The `ferric_d3::d3bj_params_for_functional` name whose damping row
    /// pairs with this set, or `None` for a set published without D3(BJ).
    pub fn d3bj_functional(&self) -> Option<&'static str> {
        match self {
            Self::B2plypD3bj => Some("b2plyp"),
            Self::DsdPbep86D3bj => Some("dsdpbep86"),
            Self::B2plyp | Self::DsdPbep86 => None,
        }
    }
}

/// Result of an MP2-based double-hybrid calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct Mp2DoubleHybridResult {
    /// `e_ks + e_corr_scaled` (Hartree).
    pub total_energy: f64,
    /// Energy of the converged KS reference (Hartree).
    pub e_ks: f64,
    /// Unscaled RI-MP2 opposite-/same-spin components on the KS orbitals.
    pub spin_components: SpinComponents,
    /// `c_os × e_os + c_ss × e_ss` (Hartree).
    pub e_corr_scaled: f64,
    /// Opposite-spin scaling used, copied from the config.
    pub c_os: f64,
    /// Same-spin scaling used, copied from the config.
    pub c_ss: f64,
}

impl std::fmt::Display for Mp2DoubleHybridResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Double hybrid total: {:.10} Ha (KS: {:.10}, MP2 corr: {:.10})",
            self.total_energy, self.e_ks, self.e_corr_scaled
        )
    }
}

/// [`mp2_double_hybrid`] on the system `aux` was prepared on (see [`PreparedAux`]):
/// `ks` is checked against that system first
/// ([`PreparedAux::check_reference`]).
/// The reference must be a closed-shell restricted (RHF / RKS) result
/// ([`PreparedAux::check_restricted_reference`]).
pub fn mp2_double_hybrid_on(
    aux: &PreparedAux<'_>,
    ks: &ScfResult,
    cfg: &Mp2DoubleHybridConfig,
) -> Result<Mp2DoubleHybridResult, FerricError> {
    let sys = aux.system();
    aux.check_restricted_reference(ks)?;
    mp2_double_hybrid(sys.mol(), sys.basis(), aux.basis(), ks, cfg)
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

/// [`u_mp2_double_hybrid`] on the system `aux` was prepared on (see [`PreparedAux`]):
/// `ks` is checked against that system first
/// ([`PreparedAux::check_reference`]).
pub fn u_mp2_double_hybrid_on(
    aux: &PreparedAux<'_>,
    ks: &ScfResult,
    cfg: &Mp2DoubleHybridConfig,
) -> Result<Mp2DoubleHybridResult, FerricError> {
    let sys = aux.system();
    aux.check_reference(ks)?;
    u_mp2_double_hybrid(sys.mol(), sys.basis(), aux.basis(), ks, cfg)
}

/// Compute an MP2-based double-hybrid energy on a converged UNRESTRICTED
/// Kohn-Sham reference (UKS).
///
/// `E_OS` is the alpha-beta component of `u_ri_mp2` and `E_SS` the sum of its
/// alpha-alpha and beta-beta components, on the UKS orbitals and orbital
/// energies. A restricted `ks` is an error: use [`mp2_double_hybrid`].
pub fn u_mp2_double_hybrid(
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
    let ri_cfg = RiMp2Config {
        frozen_core: cfg.frozen_core,
        memory_budget_bytes: cfg.memory_budget_bytes,
        ..Default::default()
    };
    let u = u_ri_mp2(mol, obs, dfbs, Operator::coulomb(), ks, &ri_cfg)?;
    let sc = SpinComponents {
        e_os: u.components.e_ab,
        e_ss: u.components.e_aa + u.components.e_bb,
        e_total: u.components.e_total,
    };
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
