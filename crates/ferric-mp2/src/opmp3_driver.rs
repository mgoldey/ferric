//! One OP-MP3 run, shared by the CLI (`method.kind = "opmp3"`) and Python
//! (`ferric.run_opmp3`): the restricted reference SCF (RHF/ROHF, or RKS/ROKS
//! for a functional), the HF-Fock semicanonical reference with `E_HF[ρ]` and
//! `E_NBS` ([`crate::opmp3_reference`]), the spin-blocked MP3 amplitudes
//! ([`crate::u_mp3`]), `W^(2)`, `W^(3)` and every model ([`crate::opmp3`]).
//!
//! Every total is `E_HF[ρ] + E_NBS + E_c(model)` (the paper's definition).
//!
//! The drivers take the [`PreparedSystem`](ferric_scf::PreparedSystem)
//! (molecule with the basis set's ECP applied, prepared basis, Coulomb
//! Schwarz bounds, context) and a fitting basis prepared on it
//! ([`PreparedAux`](ferric_scf::PreparedAux)), so the reference, the HF Fock
//! build and the amplitudes all see one molecule and one basis.

use crate::opmp3::{paper_x_ss, w_matrices, Opmp3Models, WMatrices, PAPER_X_SS};
use crate::opmp3_reference::{opmp3_reference, Opmp3Reference};
use crate::u_mp3::{
    u_mp3_from_orbitals, UMp3Channels, UMp3Config, UMp3Dims, UMp3Eri, UMp3SpinReference,
};
use ferric_core::FerricError;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::{PreparedAux, PreparedSystem, ScfResult};

/// SCF density-change threshold of the reference when none is given. The
/// correlation energy moves to first order with the orbitals, so the run's
/// floor is about `|E_c| · density_conv` (≈ 1e-9 Ha here).
pub const OPMP3_DENSITY_CONV: f64 = 1e-8;
/// SCF energy-change threshold of the reference when none is given.
pub const OPMP3_ENERGY_CONV: f64 = 1e-10;
/// Schwarz threshold of the exact J/K build of the HF Fock matrices.
pub const OPMP3_FOCK_THRESH: f64 = 1e-12;

/// Which reference determinant to start from.
#[derive(Debug, Clone, PartialEq)]
pub enum Opmp3RefKind {
    /// RHF (singlet) or ROHF (open shell).
    Hf,
    /// RKS (singlet) or ROKS (open shell) with this libxc functional name.
    Dft(String),
}

impl Opmp3RefKind {
    /// `"hf"` → [`Self::Hf`], anything else a functional name. Empty is an error.
    pub fn parse_config_str(s: &str) -> Result<Self, String> {
        let t = s.trim();
        if t.is_empty() {
            return Err(
                "reference must be \"hf\" or a functional name, got an empty string".into(),
            );
        }
        Ok(if t.eq_ignore_ascii_case("hf") {
            Self::Hf
        } else {
            Self::Dft(t.to_string())
        })
    }

    /// The name used to look up the paper's `x_ss` (`"hf"` or the functional).
    pub fn name(&self) -> &str {
        match self {
            Self::Hf => "hf",
            Self::Dft(f) => f,
        }
    }
}

/// Integrals of the amplitude and W stage (the reference Fock is always exact).
#[derive(Debug, Clone, Copy)]
pub enum Opmp3Integrals<'a> {
    /// Exact four-centre (the paper's choice).
    Exact,
    /// Coulomb-metric RI with this auxiliary basis, prepared on the run's
    /// system ([`PreparedAux::basis_for`] refuses one of another system).
    Ri(&'a PreparedAux<'a>),
}

/// Settings of [`run_opmp3`].
#[derive(Debug, Clone)]
pub struct Opmp3Config {
    /// Reference determinant.
    pub reference: Opmp3RefKind,
    /// Same-spin damping of SSD-OPMP3. `None` = the paper's printed value for
    /// the reference ([`paper_x_ss`]); a reference without one is an error.
    pub x_ss: Option<f64>,
    /// Lowest occupied orbitals of each spin left out of the correlation and
    /// of `E_NBS` (the paper correlates every electron).
    pub frozen_core: usize,
    /// SCF iteration cap (`None` = the library default).
    pub scf_max_iter: Option<usize>,
    /// SCF energy-change threshold (`None` = [`OPMP3_ENERGY_CONV`]).
    pub scf_energy_conv: Option<f64>,
    /// SCF density-change threshold (`None` = [`OPMP3_DENSITY_CONV`]).
    pub scf_density_conv: Option<f64>,
    /// Memory budget in bytes (`None` = the resolved process budget).
    pub memory_budget_bytes: Option<usize>,
}

impl Opmp3Config {
    /// The paper's SSD-OPMP3 run on `reference`: its printed `x_ss`,
    /// all electrons, default SCF thresholds.
    pub fn new(reference: Opmp3RefKind) -> Self {
        Opmp3Config {
            reference,
            x_ss: None,
            frozen_core: 0,
            scf_max_iter: None,
            scf_energy_conv: None,
            scf_density_conv: None,
            memory_budget_bytes: None,
        }
    }

    /// The `x_ss` of this run, or why there is none.
    pub fn resolved_x_ss(&self) -> Result<f64, FerricError> {
        let x = self
            .x_ss
            .or_else(|| paper_x_ss(self.reference.name()))
            .ok_or_else(|| {
                let printed = PAPER_X_SS
                    .iter()
                    .map(|(k, x)| format!("{k} {x}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                FerricError::General(format!(
                    "opmp3: the paper prints no SSD-OPMP3 x_ss for reference {:?}; give x_ss \
                     explicitly (printed: {printed})",
                    self.reference.name()
                ))
            })?;
        if !x.is_finite() {
            return Err(FerricError::General(format!(
                "opmp3: x_ss must be finite, got {x}"
            )));
        }
        Ok(x)
    }
}

/// Result of [`run_opmp3`]. Every energy in Hartree.
#[derive(Debug, Clone)]
pub struct Opmp3Result {
    /// The reference SCF energy as converged (the KS energy for DFT).
    pub e_scf: f64,
    /// HF energy of the reference density.
    pub e_hf: f64,
    /// Non-Brillouin singles energy (active occupied, both spins).
    pub e_nbs: f64,
    /// Second-order doubles energy by spin channel.
    pub e2: UMp3Channels,
    /// Third-order energy by spin channel.
    pub e3: UMp3Channels,
    /// `W^(2)` and `W^(3)` per spin and channel.
    pub w: WMatrices,
    /// Model correlation energies.
    pub models: Opmp3Models,
    /// `max |F_ia|` per spin in the semicanonical basis.
    pub max_ov: [f64; 2],
    /// Orbital counts of the amplitude stage.
    pub dims: UMp3Dims,
    /// Bytes the amplitude stage pre-flighted at its peak.
    pub peak_bytes: usize,
    /// SCF iterations of the reference.
    pub scf_iterations: usize,
}

impl Opmp3Result {
    /// `E_HF + E_NBS + e_c`, the total of a model correlation energy.
    pub fn total(&self, e_c: f64) -> f64 {
        self.e_hf + self.e_nbs + e_c
    }
}

/// The SCF settings of [`opmp3_scf`]: exact J/K, the thresholds of `config`
/// (or [`OPMP3_ENERGY_CONV`] / [`OPMP3_DENSITY_CONV`]), the functional of a
/// DFT reference.
pub fn opmp3_scf_config(config: &Opmp3Config) -> RhfConfig {
    let mut cfg = RhfConfig::default()
        .with_energy_conv(config.scf_energy_conv.unwrap_or(OPMP3_ENERGY_CONV))
        .with_density_conv(config.scf_density_conv.unwrap_or(OPMP3_DENSITY_CONV));
    cfg.three_index_budget_bytes = config.memory_budget_bytes.unwrap_or(0);
    cfg.max_iter = config.scf_max_iter.unwrap_or(cfg.max_iter);
    if let Opmp3RefKind::Dft(f) = &config.reference {
        cfg.xc = Some(f.clone());
    }
    cfg
}

/// The reference SCF of `config` on `sys` (restricted: RHF/RKS for a
/// singlet, ROHF/ROKS otherwise), exact J/K, screening with the system's
/// kind ([`PreparedSystem::screening`]). Unconverged is an error.
pub fn opmp3_scf(sys: &PreparedSystem, config: &Opmp3Config) -> Result<ScfResult, FerricError> {
    let mut cfg = opmp3_scf_config(config);
    cfg.screening = sys.screening();
    let scf = if sys.mol().multiplicity == 1 {
        sys.solve_rhf(&cfg)?
    } else {
        sys.solve_rohf(&cfg)?
    };
    if !scf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: scf.iterations,
            last_energy: scf.energy,
        });
    }
    Ok(scf)
}

/// OP-MP3 on an existing converged reference `scf` (any spin type).
pub fn opmp3_on_reference(
    sys: &PreparedSystem,
    scf: &ScfResult,
    integrals: Opmp3Integrals<'_>,
    config: &Opmp3Config,
) -> Result<Opmp3Result, FerricError> {
    let x_ss = config.resolved_x_ss()?;
    let eri = match integrals {
        Opmp3Integrals::Exact => UMp3Eri::Exact,
        Opmp3Integrals::Ri(aux) => UMp3Eri::Ri {
            aux: aux.basis_for(sys)?,
        },
    };
    let prep = sys.basis();
    let r: Opmp3Reference = opmp3_reference(
        sys.ctx(),
        sys.mol(),
        prep,
        sys.bounds(),
        OPMP3_FOCK_THRESH,
        scf,
    )?;
    let e_nbs = r.e_nbs(config.frozen_core)?;
    let spin = |s: usize| UMp3SpinReference {
        mos: &r.spins[s].mos,
        eps: &r.spins[s].eps,
        nocc: r.spins[s].nocc,
    };
    let ump3 = u_mp3_from_orbitals(
        prep,
        Operator::coulomb(),
        [spin(0), spin(1)],
        r.e_hf,
        eri,
        &UMp3Config {
            frozen_core: config.frozen_core,
            memory_budget_bytes: config.memory_budget_bytes,
        },
    )?;
    let w = w_matrices(&ump3);
    let models = Opmp3Models::evaluate(&w, x_ss)?;
    Ok(Opmp3Result {
        e_scf: r.e_scf,
        e_hf: r.e_hf,
        e_nbs,
        e2: ump3.e2,
        e3: ump3.e3,
        w,
        models,
        max_ov: [r.spins[0].max_ov(), r.spins[1].max_ov()],
        dims: ump3.dims,
        peak_bytes: ump3.peak_bytes,
        scf_iterations: scf.iterations,
    })
}

/// The whole run: [`opmp3_scf`], then [`opmp3_on_reference`].
pub fn run_opmp3(
    sys: &PreparedSystem,
    integrals: Opmp3Integrals<'_>,
    config: &Opmp3Config,
) -> Result<Opmp3Result, FerricError> {
    config.resolved_x_ss()?;
    let scf = opmp3_scf(sys, config)?;
    opmp3_on_reference(sys, &scf, integrals, config)
}
