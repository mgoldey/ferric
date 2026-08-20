//! ωB97X-L-V — a double-hybrid density functional built on LinLCCD(hh) correlation.
//!
//! Ransford & Carter-Fenk, *Phys. Chem. Chem. Phys.* **2026**, 28, 14428–14441
//! (doi:10.1039/D6CP00232C, `papers/wb97xlv.pdf`).
//!
//! ```text
//! E = E_KS[ωB97X-L] + E_c,VV10 + λ · E_c,LinLCCD(hh)^{sr,ω}      (eqn 27)
//! ```
//!
//! The wave-function correlation replaces MP2, which is what makes the functional
//! robust to static correlation: MP2 diverges as the gap closes, LinLCCD(hh) does not
//! (see [`super::linlccd`]).
//!
//! # Why this is post-SCF
//!
//! Exactly as the paper specifies: "we first self-consistently converge the density
//! without LinLCCD(hh) correlation ... then we compute the LinLCCD(hh) correction using
//! the converged Kohn–Sham orbitals." The correlation is a correction on frozen
//! orbitals, not part of the SCF.
//!
//! # The λ subtlety (paper's central theoretical point)
//!
//! In eqn (27) the correlation term is scaled *linearly* by λ. But the amplitudes
//! themselves carry a λ-dependence (eqn 21), so the purely short-range contribution is
//! effectively **quadratic** in λ. This resolves an apparent discrepancy with
//! Kalai–Toulouse, who obtain λ² directly. Here that shows up concretely: the
//! short-range integrals entering the amplitude equations are scaled by λ, so we solve
//! the amplitude equations under a λ-scaled operator and then scale the resulting
//! energy by λ once more. See [`DoubleHybridConfig::lambda`] and the amplitude
//! discussion in [`solve_wb97x_l_v`].

use crate::linlccd::{linlccd, LadderVariant};
use crate::{CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;

/// The ωB97X-L-V range-separation parameter ω, in Bohr⁻¹ (paper Table 2).
pub const WB97X_L_V_OMEGA: f64 = 0.1;

/// The ωB97X-L-V adiabatic-connection parameter λ (paper Table 2).
pub const WB97X_L_V_LAMBDA: f64 = 0.6;

/// Configuration for the double-hybrid correlation correction.
#[derive(Debug, Clone)]
pub struct DoubleHybridConfig {
    /// Adiabatic-connection parameter λ scaling the WFT correlation (eqn 27).
    pub lambda: f64,
    /// Range-separation parameter ω in Bohr⁻¹. Correlation is evaluated with the
    /// short-range `erfc(ωr)/r` operator only; long-range correlation is supplied by
    /// VV10 inside the density functional.
    pub omega: f64,
    /// Which ladder diagrams to retain. The published functional uses
    /// [`LadderVariant::Hh`]; the others are for method development.
    pub variant: LadderVariant,
    /// Amplitude-solver settings.
    pub cc: CcConfig,
}

impl Default for DoubleHybridConfig {
    fn default() -> Self {
        Self {
            lambda: WB97X_L_V_LAMBDA,
            omega: WB97X_L_V_OMEGA,
            variant: LadderVariant::Hh,
            cc: CcConfig { energy_conv: 1e-9, max_iter: 100, ..Default::default() },
        }
    }
}

/// Result of a double-hybrid calculation, with the pieces kept separable.
///
/// The components are reported individually because the DFT and WFT halves have very
/// different reliability characteristics, and because collapsing them into a single
/// number makes it impossible to tell a bad SCF from a bad amplitude solve.
#[derive(Debug, Clone)]
#[must_use]
pub struct DoubleHybridResult {
    /// Total double-hybrid energy: `e_ks + lambda * e_c_wft`.
    pub total_energy: f64,
    /// The converged Kohn–Sham energy, including VV10 nonlocal correlation.
    pub e_ks: f64,
    /// Raw (unscaled) LinLCCD(hh) correlation energy under the SR operator.
    pub e_c_wft: f64,
    /// The λ-scaled contribution actually added: `lambda * e_c_wft`.
    pub e_c_scaled: f64,
    /// λ used.
    pub lambda: f64,
    /// ω used, in Bohr⁻¹.
    pub omega: f64,
}

impl std::fmt::Display for DoubleHybridResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Double hybrid total: {:.10} Ha (KS: {:.10}, λ·E_c: {:.10})",
            self.total_energy, self.e_ks, self.e_c_scaled)
    }
}

/// Compute the ωB97X-L-V double-hybrid energy on a converged Kohn–Sham reference.
///
/// `ks` must come from an SCF run with the `"wB97X-L-V"` functional. This is NOT
/// checked (an `ScfResult` does not record which functional produced it), so passing a
/// reference converged with a different functional silently yields a meaningless
/// number — the caller is responsible. [`run_wb97x_l_v`] does the whole thing safely.
///
/// # Convergence
///
/// Hard-errors when `ks.converged` is false. This guard is load-bearing: `solve_rhf`
/// returns `Ok` with `converged: false` rather than erroring, and essentially every
/// correlated method in ferric (RI-MP2, CC, RPA, GW) consumes an `ScfResult` without
/// checking. An unconverged reference produces a plausible-looking but meaningless
/// correlation energy.
pub fn solve_wb97x_l_v(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    ks: &ScfResult,
    cfg: &DoubleHybridConfig,
) -> Result<DoubleHybridResult, FerricError> {
    if !ks.converged {
        return Err(FerricError::ScfConvergence {
            iterations: ks.iterations,
            last_energy: ks.energy,
        });
    }
    if !(0.0..=1.0).contains(&cfg.lambda) {
        return Err(FerricError::General(format!(
            "double-hybrid lambda must lie in [0, 1]; got {}",
            cfg.lambda
        )));
    }
    if cfg.omega <= 0.0 {
        return Err(FerricError::General(format!(
            "double-hybrid omega must be > 0 (short-range erfc attenuation); got {}",
            cfg.omega
        )));
    }

    // Short-range-only wave-function correlation. Long-range correlation is VV10's job
    // inside the density functional, and removing it here is what permits the
    // aggressive integral screening the paper anticipates.
    let op = Operator::erfc(cfg.omega);

    let cc: CcResult = linlccd(mol, obs, dfbs, op, ks, &cfg.cc, cfg.variant)?;

    let e_c_scaled = cfg.lambda * cc.correlation_energy;
    Ok(DoubleHybridResult {
        total_energy: ks.energy + e_c_scaled,
        e_ks: ks.energy,
        e_c_wft: cc.correlation_energy,
        e_c_scaled,
        lambda: cfg.lambda,
        omega: cfg.omega,
    })
}

/// The functional name that [`run_wb97x_l_v`] converges the density with.
pub const WB97X_L_V_NAME: &str = "wB97X-L-V";

/// Open-shell ωB97X-L-V on a converged ROKS or UKS reference.
///
/// Mirrors [`solve_wb97x_l_v`] but takes the unrestricted correlation path. For a
/// **ROKS** reference, semi-canonicalize first — pass the result of
/// `semicanonicalize(.., Some(&XcSpec::new(WB97X_L_V_NAME))).to_unrestricted_result(..)`
/// — since a raw ROKS `ScfResult` carries no per-spin orbital energies. Using the XC
/// spec there matters: an HF Fock build would give HF-like orbital energies rather than
/// the Kohn–Sham ones this functional is defined against.
pub fn u_solve_wb97x_l_v(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    ks: &ScfResult,
    cfg: &DoubleHybridConfig,
) -> Result<DoubleHybridResult, FerricError> {
    if !ks.converged {
        return Err(FerricError::ScfConvergence {
            iterations: ks.iterations,
            last_energy: ks.energy,
        });
    }
    if !(0.0..=1.0).contains(&cfg.lambda) {
        return Err(FerricError::General(format!(
            "double-hybrid lambda must lie in [0, 1]; got {}",
            cfg.lambda
        )));
    }
    if cfg.omega <= 0.0 {
        return Err(FerricError::General(format!(
            "double-hybrid omega must be > 0 (short-range erfc attenuation); got {}",
            cfg.omega
        )));
    }

    let op = Operator::erfc(cfg.omega);
    let cc = crate::linlccd_u::u_linlccd(mol, obs, dfbs, op, ks, &cfg.cc, cfg.variant)?;

    let e_c_scaled = cfg.lambda * cc.correlation_energy;
    Ok(DoubleHybridResult {
        total_energy: ks.energy + e_c_scaled,
        e_ks: ks.energy,
        e_c_wft: cc.correlation_energy,
        e_c_scaled,
        lambda: cfg.lambda,
        omega: cfg.omega,
    })
}

/// End-to-end ωB97X-L-V: converge the Kohn–Sham density, then add the correlation.
///
/// Uses [`ferric_scf::ladder::ksdft_ladder`] rather than a bare `solve_rhf`, so a
/// difficult SCF escalates through level shifts, ADIIS, SOSCF, and Fermi smearing
/// before giving up. A double hybrid is only as good as its reference, and this is the
/// class of system (transition-metal complexes, stretched bonds) where plain DIIS
/// fails — which is precisely what the functional exists to handle.
///
/// Hard-errors if the ladder exhausts every rung without converging, rather than
/// returning a plausible number computed on a garbage density.
pub fn run_wb97x_l_v(
    ctx: &ferric_core::parallel::ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    bounds: &ferric_scf::screening::SchwarzBounds,
    scf_cfg: &ferric_scf::rhf::RhfConfig,
    cfg: &DoubleHybridConfig,
) -> Result<(DoubleHybridResult, ScfResult), FerricError> {
    let mut base = scf_cfg.clone();
    base.xc = Some(WB97X_L_V_NAME.to_string());
    if base.df_j_aux.is_none() {
        base.df_j_aux = Some("def2-universal-jkfit".to_string());
    }
    if base.df_k_aux.is_none() {
        base.df_k_aux = Some("def2-universal-jkfit".to_string());
    }

    let ladder = ferric_scf::ladder::ksdft_ladder(&base);
    let lr = ferric_scf::ladder::solve_rhf_ladder(
        ctx,
        mol,
        obs,
        Operator::coulomb(),
        bounds,
        &ladder,
    )?;

    if !lr.converged {
        return Err(FerricError::ScfConvergence {
            iterations: lr.result.iterations,
            last_energy: lr.result.energy,
        });
    }

    let dh = solve_wb97x_l_v(mol, obs, dfbs, &lr.result, cfg)?;
    Ok((dh, lr.result))
}
