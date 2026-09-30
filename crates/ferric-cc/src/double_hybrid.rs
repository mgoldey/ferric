//! ωB97X-L-V — a double-hybrid density functional built on LinLCCD(hh) correlation.
//!
//! Ransford & Carter-Fenk, *Phys. Chem. Chem. Phys.* **2026**, 28, 14428–14441
//! (doi:10.1039/D6CP00232C, `papers/wb97xlv.pdf`).
//!
//! ```text
//! E = E_KS[ωB97X-L] + E_c,VV10 + λ · E_c,LinLCCD(hh)^{sr,ω,λ}    (eqn 27)
//! ```
//!
//! The wave-function correlation replaces MP2, which is what makes the functional
//! robust to static correlation: MP2 diverges as the gap closes, LinLCCD(hh) does not
//! (see [`crate::linlccd`]).
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
//! Eqn (22) puts λ on BOTH the driver and the hole–hole ladder of the amplitude
//! equations, while the Fock terms stay unscaled:
//!
//! ```text
//!   (Fock terms) t_kl^cd = −λ ( <ij||ab>^sr + ½ <ik||jl>^sr t_kl^ab )
//! ```
//!
//! so the amplitudes are the LinLCCD(hh) amplitudes of the λ-scaled operator
//! λ·erfc(ωr)/r, and the eqn (27) term λ·E_c^{sr,ω,λ} = ¼ Σ (λ<ij||ab>^sr) t is
//! "actually quadratic in λ" (paper, below eqn 23). In the MP2 limit it is exactly
//! λ²·E_c,MP2^{sr,ω} (eqns 19–20). We therefore run LinLCCD(hh) once under the
//! single-component composite operator `λ·erfc(ω)`: both the 3-centre integrals and
//! the RI metric scale by λ, so the RI factor scales by √λ and every 4-index integral
//! by λ, and the correlation energy that comes back is already λ·E_c^{sr,ω,λ}.
//!
//! Solving at λ = 1 and multiplying by λ afterwards is linear in λ and overstates the
//! correlation by ~1/λ (1.67× at λ = 0.6) in the MP2 limit.

use crate::linlccd::{linlccd, LadderVariant};
use crate::{CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::{Operator, OperatorKind};
use ferric_scf::ScfResult;

/// The ωB97X-L-V range-separation parameter ω, in Bohr⁻¹ (paper Table 2).
pub const WB97X_L_V_OMEGA: f64 = 0.1;

/// The ωB97X-L-V adiabatic-connection parameter λ (paper Table 2).
pub const WB97X_L_V_LAMBDA: f64 = 0.6;

/// Configuration for the double-hybrid correlation correction.
#[derive(Debug, Clone)]
pub struct DoubleHybridConfig {
    /// Adiabatic-connection parameter λ. It scales the short-range operator inside
    /// the amplitude equations (eqn 22) AND the energy (eqn 27), so the WFT
    /// correlation is quadratic in λ at leading order.
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
            cc: CcConfig {
                energy_conv: 1e-9,
                max_iter: 100,
                ..Default::default()
            },
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
    /// E_c^{sr,ω,λ}: the LinLCCD(hh) correlation energy ¼ Σ <ij||ab>^sr t(λ), with
    /// amplitudes solved under the λ-scaled operator (eqn 22). Zero at λ = 0.
    pub e_c_wft: f64,
    /// The contribution actually added: `lambda * e_c_wft` (eqn 27).
    pub e_c_scaled: f64,
    /// λ used.
    pub lambda: f64,
    /// ω used, in Bohr⁻¹.
    pub omega: f64,
}

impl std::fmt::Display for DoubleHybridResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Double hybrid total: {:.10} Ha (KS: {:.10}, λ·E_c: {:.10})",
            self.total_energy, self.e_ks, self.e_c_scaled
        )
    }
}

/// The operator λ·erfc(ωr)/r of eqn (22), or `None` at λ = 0.
///
/// Short-range only: long-range correlation is VV10's job inside the density
/// functional, and removing it here is what permits the aggressive integral
/// screening the paper anticipates. At λ = 0 the operator vanishes, the RI metric
/// is singular, and there is no correlation to compute.
fn lambda_scaled_sr_operator(cfg: &DoubleHybridConfig) -> Option<Operator> {
    (cfg.lambda > 0.0)
        .then(|| Operator::composite(&[(cfg.lambda, OperatorKind::ErfcCoulomb, cfg.omega)]))
}

/// `e_c_scaled` is λ·E_c^{sr,ω,λ} as returned by the λ-scaled solve.
fn assemble(e_ks: f64, e_c_scaled: f64, cfg: &DoubleHybridConfig) -> DoubleHybridResult {
    let e_c_wft = if cfg.lambda > 0.0 {
        e_c_scaled / cfg.lambda
    } else {
        0.0
    };
    DoubleHybridResult {
        total_energy: e_ks + e_c_scaled,
        e_ks,
        e_c_wft,
        e_c_scaled,
        lambda: cfg.lambda,
        omega: cfg.omega,
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

    let e_c_scaled = match lambda_scaled_sr_operator(cfg) {
        Some(op) => {
            let cc: CcResult = linlccd(mol, obs, dfbs, op, ks, &cfg.cc, cfg.variant)?;
            cc.correlation_energy
        }
        None => 0.0,
    };
    Ok(assemble(ks.energy, e_c_scaled, cfg))
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

    let e_c_scaled = match lambda_scaled_sr_operator(cfg) {
        Some(op) => {
            crate::linlccd_u::u_linlccd(mol, obs, dfbs, op, ks, &cfg.cc, cfg.variant)?
                .correlation_energy
        }
        None => 0.0,
    };
    Ok(assemble(ks.energy, e_c_scaled, cfg))
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
    let lr =
        ferric_scf::ladder::solve_rhf_ladder(ctx, mol, obs, Operator::coulomb(), bounds, &ladder)?;

    if !lr.converged {
        return Err(FerricError::ScfConvergence {
            iterations: lr.result.iterations,
            last_energy: lr.result.energy,
        });
    }

    let dh = solve_wb97x_l_v(mol, obs, dfbs, &lr.result, cfg)?;
    Ok((dh, lr.result))
}
