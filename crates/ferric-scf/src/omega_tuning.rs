//! Optimal (IP-based, "gap") tuning of the range-separation parameter ω for
//! range-separated hybrid functionals — the standard Baer/Kronik ΔSCF
//! condition: choose ω so that Koopmans' theorem holds for the HOMO,
//!
//! ```text
//! J(ω) = ε_HOMO(N; ω) + IP_ΔSCF(ω),   IP = E(N−1; ω) − E(N; ω),
//! ```
//!
//! minimized as |J| by golden-section search. Each evaluation runs one
//! closed-shell RKS (neutral) and one UKS (cation, doublet) with the SAME
//! geometry, basis, grids and functional, differing only in ω via
//! `RhfConfig::xc_omega` (libxc `_omega` override; hard error for
//! functionals without one, so a tuning run can never silently fall back to
//! a fixed-ω functional).
//!
//! Scope: closed-shell neutral references (even-electron RKS) with a
//! doublet cation, the textbook case. Anions/EA-tuning and open-shell
//! references are not implemented — extend, don't approximate.

use crate::rhf::{solve_rhf, RhfConfig};
use crate::screening::SchwarzBounds;
use crate::uhf::solve_uhf;
use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;

/// Configuration for IP-based optimal tuning of the range-separation parameter ω.
#[derive(Debug, Clone)]
pub struct OmegaTuneConfig {
    /// RSH functional name (must carry a libxc `_omega` parameter).
    pub functional: String,
    /// Search bracket, Bohr⁻¹.
    pub omega_lo: f64,
    pub omega_hi: f64,
    /// Convergence width on ω (Bohr⁻¹).
    pub omega_tol: f64,
    /// Hard cap on J evaluations (each = 2 SCF solves).
    pub max_evals: usize,
    /// Base SCF settings applied to BOTH states (grids, convergence, ...).
    /// `xc`/`xc_omega` in it are overwritten per evaluation.
    pub scf: RhfConfig,
}

impl Default for OmegaTuneConfig {
    fn default() -> Self {
        Self {
            functional: String::new(),
            omega_lo: 0.1,
            omega_hi: 1.0,
            omega_tol: 5e-3,
            max_evals: 24,
            scf: RhfConfig::default(),
        }
    }
}

/// A single ω evaluation: HOMO eigenvalue, ΔSCF ionization potential, and the Koopmans residual J.
#[derive(Debug, Clone, Copy)]
pub struct OmegaEval {
    pub omega: f64,
    pub eps_homo: f64,
    pub ip_delta_scf: f64,
    pub j: f64,
}

/// Result of an ω-tuning run: optimal ω, residual J, and the full evaluation trace.
#[derive(Debug, Clone)]
#[must_use]
pub struct OmegaTuneResult {
    pub omega: f64,
    /// J(ω*) — the residual Koopmans violation at the tuned ω.
    pub j: f64,
    pub evals: Vec<OmegaEval>,
    pub converged: bool,
}

impl std::fmt::Display for OmegaTuneResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ω-tuning: ω* = {:.6} Bohr⁻¹ (J = {:.6}, {} evals, converged: {})",
            self.omega, self.j, self.evals.len(), self.converged)
    }
}

fn eval_j(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    cfg: &OmegaTuneConfig,
    omega: f64,
) -> Result<OmegaEval, FerricError> {
    let scf_cfg = RhfConfig {
        xc: Some(cfg.functional.clone()),
        xc_omega: Some(omega),
        ..cfg.scf.clone()
    };
    let op = Operator::coulomb();
    let neutral = solve_rhf(ctx, mol, prep, op, bounds, &scf_cfg)?;
    if !neutral.converged {
        return Err(FerricError::ScfConvergence {
            iterations: neutral.iterations,
            last_energy: neutral.energy,
        });
    }
    let nocc = (mol.nelec() as usize) / 2;
    let eps_homo = neutral.eps_r()[nocc - 1];

    let mut cation = mol.clone();
    cation.charge += 1;
    cation.multiplicity = 2;
    let cat = solve_uhf(ctx, &cation, prep, bounds, &scf_cfg)?;
    if !cat.converged {
        return Err(FerricError::ScfConvergence {
            iterations: cat.iterations,
            last_energy: cat.energy,
        });
    }
    let ip = cat.energy - neutral.energy;
    Ok(OmegaEval { omega, eps_homo, ip_delta_scf: ip, j: eps_homo + ip })
}

/// Golden-section minimization of |J(ω)| over the bracket.
pub fn tune_omega(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    cfg: &OmegaTuneConfig,
) -> Result<OmegaTuneResult, FerricError> {
    if mol.nelec() % 2 != 0 {
        return Err(FerricError::General(
            "tune_omega: closed-shell (even-electron) neutral references only".into(),
        ));
    }
    if !(cfg.omega_lo > 0.0 && cfg.omega_hi > cfg.omega_lo) {
        return Err(FerricError::General(format!(
            "tune_omega: invalid bracket [{}, {}]",
            cfg.omega_lo, cfg.omega_hi
        )));
    }
    let mut evals: Vec<OmegaEval> = Vec::new();
    let f = |w: f64, evals: &mut Vec<OmegaEval>| -> Result<f64, FerricError> {
        let e = eval_j(ctx, mol, prep, bounds, cfg, w)?;
        evals.push(e);
        Ok(e.j.abs())
    };
    const INVPHI: f64 = 0.618_033_988_749_894_9;
    let (mut a, mut b) = (cfg.omega_lo, cfg.omega_hi);
    let mut c = b - (b - a) * INVPHI;
    let mut d = a + (b - a) * INVPHI;
    let mut fc = f(c, &mut evals)?;
    let mut fd = f(d, &mut evals)?;
    let mut converged = false;
    while evals.len() < cfg.max_evals {
        if (b - a) < cfg.omega_tol {
            converged = true;
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * INVPHI;
            fc = f(c, &mut evals)?;
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * INVPHI;
            fd = f(d, &mut evals)?;
        }
    }
    let best = evals
        .iter()
        .cloned()
        .min_by(|x, y| x.j.abs().partial_cmp(&y.j.abs()).expect("NaN J"))
        .expect("at least two evaluations");
    Ok(OmegaTuneResult { omega: best.omega, j: best.j, evals, converged })
}
