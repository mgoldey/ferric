//! Which construction evaluates a closed-shell RPA correlation energy.
//!
//! Both routes integrate the same quantity,
//!
//! ```text
//! E_c = (1/2π) Σ_k w_k [ ln det ε̃(iω_k) + tr(I − ε̃(iω_k)) ],   ε̃(iω) = I + Π(iω),
//! ```
//!
//! with `Π(iω) = B̃ S(iω)² B̃ᵀ` in the dressed auxiliary basis
//! (`S² = 4 e_ia / (ω² + e_ia²)`).
//!
//! * [`RpaEnergyRoute::StaticEigenbasis`]: diagonalize `ε̃(0)` (the PDEP
//!   eigensolve), keep the modes with `|λ − 1| > trunc_thresh`, and assemble
//!   `ε̃(iω)` in the kept eigenbasis. The route for truncation, eigenpotentials,
//!   NPZ/cube export and GW.
//! * [`RpaEnergyRoute::AuxBasisLogDet`]: assemble `ε̃(iω)` in the auxiliary
//!   basis itself. With `trunc_thresh <= 0` every mode is kept, the eigenbasis
//!   is a full orthogonal rotation and both traces are invariant under it, so
//!   the energy is the same up to rounding, without the static eigensolve.
//!
//! [`run_pdep_rpa_energy`] takes the second route when the configuration asks
//! for the energy only with `trunc_thresh <= 0`, and the first otherwise.
//!
//! A positive threshold keeps the static route even when it would drop nothing:
//! `Π(0)` is close to singular in practice (`λ_min(0) − 1` measured at
//! −3e-16 on water/cc-pVDZ with and without frozen core, 5.7e-7 on n-octane
//! and 6.1e-7 on n-hexadecane at cc-pVDZ/cc-pVDZ-RI, 2026-11-14), so a test
//! that every mode clears the default `1e-4` would fail on every system and
//! cost one more `ε̃(0)` build.

use crate::config::{Chi0Backend, Chi0Sparsity, PdepRpaConfig};
use crate::{energy, quadrature, PdepRpaResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config, RpaIntermediates};
use ferric_scf::PreparedAux;
use ferric_scf::ScfResult;
use ndarray::Array2;

/// Gauss-Legendre points of the exact dRPA ([`exact_drpa_config`]).
///
/// From the convergence table in `tests/validation_drpa_frequency.rs`: the
/// smallest order at which the error against the plasmon formula (which has
/// no quadrature) reaches the rounding floor on every system there, frozen
/// core 0 included. All-electron runs need it: the core excitations push the
/// integrand out to large `ω`, and 32 points still miss by 1.4e-8 Eh on
/// water/cc-pVDZ. Past the floor the mapped rule's growing weights amplify
/// rounding, so more points move the energy by about 1e-9 Eh at 200 points.
pub const EXACT_DRPA_QUAD_POINTS: usize = 48;

/// The configuration under which [`run_pdep_rpa_energy`] returns the exact
/// closed-shell dRPA energy: nothing truncated, energy only, dense χ₀,
/// Gauss-Legendre with [`EXACT_DRPA_QUAD_POINTS`] points at `u0 = 0.5`. It
/// always takes the aux-basis log-det route. The CLI's `method.kind = "drpa"`
/// without `[local]` runs exactly this.
pub fn exact_drpa_config(frozen_core: usize, memory_budget_bytes: Option<usize>) -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core,
        trunc_thresh: 0.0,
        quadrature: crate::config::QuadratureConfig {
            scheme: crate::config::QuadratureScheme::GaussLegendre,
            n_points: EXACT_DRPA_QUAD_POINTS,
            u0: 0.5,
        },
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        memory_budget_bytes,
        need_inv_dielectric_freq: false,
        need_eigenvalues_freq: false,
        ..Default::default()
    }
}

/// [`aux_route_charge_bytes`] on the system `aux` was prepared on (see [`PreparedAux`]).
pub fn aux_route_charge_bytes_on(
    aux: &PreparedAux<'_>,
    config: &PdepRpaConfig,
) -> Result<usize, FerricError> {
    let sys = aux.system();
    aux_route_charge_bytes(sys.mol(), sys.basis(), aux.basis(), config)
}

/// The bytes [`run_pdep_rpa_energy`]'s preflight charges on the aux-basis
/// route, so a caller can refuse a run that cannot fit before its SCF.
pub fn aux_route_charge_bytes(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    config: &PdepRpaConfig,
) -> Result<usize, FerricError> {
    let planes = crate::budget::EnergyRoutePlanes::AuxBasisLogDet;
    Ok(crate::closed_shell_charge(mol, obs, dfbs, config, planes)?
        .1
        .total)
}

/// The construction that produced a [`PdepRpaResult::e_rpa`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RpaEnergyRoute {
    /// The PDEP eigensolve at ω = 0, then the frequency loop in the kept modes.
    StaticEigenbasis,
    /// `ln det(I + Π) − tr Π` per frequency in the auxiliary basis, with no
    /// static eigensolve and no mode dropped (`trunc_thresh <= 0`).
    AuxBasisLogDet,
}

impl RpaEnergyRoute {
    /// One line for the log and the certificate.
    pub fn describe(&self) -> String {
        match *self {
            Self::StaticEigenbasis => {
                "static eigenbasis (PDEP eigensolve of eps(0), frequency loop in the kept modes)"
                    .to_string()
            }
            Self::AuxBasisLogDet => "aux-basis log-det (ln det(1 + Pi) - tr Pi per frequency, \
                 no static eigensolve, every mode kept)"
                .to_string(),
        }
    }
}

/// Why `config` must keep the static-eigenbasis route, decided from the
/// configuration alone; `None` when the aux-basis log-det route may run.
fn static_route_required(config: &PdepRpaConfig, n_atoms: usize) -> Option<&'static str> {
    if config.need_eigenvalues_freq {
        Some("the caller asked for the per-frequency eigenvalues")
    } else if config.need_inv_dielectric_freq {
        Some("the caller asked for the inverse-dielectric stack")
    } else if !matches!(config.chi0_backend, Chi0Backend::Dense) {
        Some("the Laplace chi0 backend has no aux-basis evaluator")
    } else if !matches!(config.chi0_sparsity.resolve(n_atoms), Chi0Sparsity::Dense) {
        Some("chi0_sparsity selects the Boys-screened eigensolve")
    } else if config.trunc_thresh > 0.0 {
        Some("trunc_thresh > 0 truncates the static modes")
    } else {
        None
    }
}

/// [`run_pdep_rpa_energy`] on the system `aux` was prepared on (see [`PreparedAux`]):
/// `rhf` is checked against that system first
/// ([`PreparedAux::check_reference`]).
pub fn run_pdep_rpa_energy_on(
    aux: &PreparedAux<'_>,
    rhf: &ScfResult,
    config: &PdepRpaConfig,
) -> Result<PdepRpaResult, FerricError> {
    let sys = aux.system();
    aux.check_reference(rhf)?;
    run_pdep_rpa_energy(sys.mol(), sys.basis(), aux.basis(), sys.op(), rhf, config)
}

/// Energy-only closed-shell PDEP-RPA: [`crate::run_pdep_rpa`]'s energy, by the
/// aux-basis log-det route when the configuration asks for the energy only
/// with `trunc_thresh <= 0`, the dense χ₀ backend and dense sparsity, else by
/// [`crate::run_pdep_rpa`] itself (the reason goes to stderr).
///
/// For a caller that reads `e_rpa` (and `e_rpa_dft_diag`) only: on the
/// aux-basis route the eigenpotential fields are empty (see
/// [`PdepRpaResult::route`]).
pub fn run_pdep_rpa_energy(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &PdepRpaConfig,
) -> Result<PdepRpaResult, FerricError> {
    if let Some(reason) = static_route_required(config, mol.atoms.len()) {
        note_static_route(reason);
        return crate::run_pdep_rpa(mol, obs, dfbs, op, rhf, config);
    }
    let _pool_charge = crate::preflight_check_closed_shell(
        mol,
        obs,
        dfbs,
        config,
        "PDEP-RPA",
        crate::budget::EnergyRoutePlanes::AuxBasisLogDet,
    )?;
    let mp2_cfg = RiMp2Config {
        frozen_core: config.frozen_core,
        memory_budget_bytes: config.memory_budget_bytes,
        ..Default::default()
    };
    let t_setup = crate::timing::Stage::start("pdep:rpa_intermediates(ERI3+metric+MOtransform)");
    let inter = compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    t_setup.end();
    run_pdep_rpa_energy_from_intermediates(&inter, mol, obs, dfbs, op, rhf, config)
}

/// [`run_pdep_rpa_energy`] from pre-built intermediates (the same contract as
/// [`crate::run_pdep_rpa_from_intermediates`], whose route it falls back to).
#[allow(clippy::too_many_arguments)]
pub fn run_pdep_rpa_energy_from_intermediates(
    inter: &RpaIntermediates,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &PdepRpaConfig,
) -> Result<PdepRpaResult, FerricError> {
    let eps_occ = &rhf.eps_r()[inter.first_occ..inter.first_occ + inter.nocc];
    let eps_vir = &rhf.eps_r()[inter.nocc_total..inter.nocc_total + inter.nvir];
    if let Some(reason) = static_route_required(config, mol.atoms.len()) {
        note_static_route(reason);
        return crate::run_pdep_rpa_from_intermediates(inter, mol, obs, dfbs, op, rhf, config);
    }
    aux_basis_log_det_energy(inter, eps_occ, eps_vir, config)
}

/// The aux-basis log-det route itself, once it is known to apply.
fn aux_basis_log_det_energy(
    inter: &RpaIntermediates,
    eps_occ: &[f64],
    eps_vir: &[f64],
    config: &PdepRpaConfig,
) -> Result<PdepRpaResult, FerricError> {
    let (quad_freqs, quad_weights) = quadrature::build_quadrature(&config.quadrature);
    let t_quad = crate::timing::Stage::start("pdep:freq_quad(aux-basis logdet)");
    let summands = energy::eval_trace_log_summands_aux_budgeted(
        &inter.b_ov,
        eps_occ,
        eps_vir,
        &quad_freqs,
        config.memory_budget_bytes,
    )?;
    t_quad.end();
    let e_rpa = energy::rpa_correlation_energy_from_summands(&quad_weights, &summands);
    let e_rpa_dft_diag = if config.run_diagnostics {
        Some(crate::diagnostics::ri_drpa_energy(
            &inter.b_ov,
            eps_occ,
            eps_vir,
            &quad_freqs,
            &quad_weights,
        )?)
    } else {
        None
    };
    let naux = inter.naux;
    Ok(PdepRpaResult {
        e_rpa,
        n_eigenpotentials: naux,
        eigenvalues_static: Vec::new(),
        eigenpotentials: Array2::zeros((naux, 0)),
        dressed_eigenvectors: Array2::zeros((naux, 0)),
        quad_freqs,
        quad_weights,
        eigenvalues_freq: Array2::zeros((0, 0)),
        inv_dielectric_freq: None,
        e_rpa_dft_diag,
        eigensolver_converged: true,
        route: RpaEnergyRoute::AuxBasisLogDet,
    })
}

/// One stderr line when an energy-only call keeps the static route.
fn note_static_route(reason: &str) {
    eprintln!("[ferric] RPA energy route: static eigenbasis ({reason})");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Water in `basis_name` (STO-3G: 7 functions, 6-31G: 13), with its
    /// converged RHF, for the reference-check tests of the `*_on` entries.
    fn reference_check_water(basis_name: &str) -> (ferric_scf::PreparedSystem, ScfResult) {
        let mol = Molecule::parse_xyz(
            "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n",
            0,
            1,
        )
        .unwrap();
        let bs = ferric_core::basis::bundled(basis_name).unwrap();
        let sys = ferric_scf::PreparedSystem::new(mol, &bs).unwrap();
        let rhf = sys
            .solve_rhf(&ferric_scf::rhf::RhfConfig::default())
            .unwrap();
        (sys, rhf)
    }

    fn assert_reference_refused<T>(who: &str, r: Result<T, FerricError>) {
        let msg = r.map(|_| ()).unwrap_err().to_string();
        assert!(
            msg.contains("does not belong to this system"),
            "{who}: {msg}"
        );
    }

    #[test]
    fn on_entries_refuse_a_reference_of_another_system() {
        let (small, _) = reference_check_water("sto-3g");
        let (_large, rhf_large) = reference_check_water("6-31g");
        let aux = small
            .prepare_aux(&ferric_core::basis::bundled("cc-pvdz-ri").unwrap())
            .unwrap();
        let cfg = PdepRpaConfig::default();
        assert_reference_refused(
            "run_pdep_rpa_energy_on",
            run_pdep_rpa_energy_on(&aux, &rhf_large, &cfg),
        );
        assert_reference_refused(
            "run_pdep_rpa_on",
            crate::run_pdep_rpa_on(&aux, &rhf_large, &cfg),
        );
        assert_reference_refused(
            "run_pdep_rpa_fitted_on",
            crate::run_pdep_rpa_fitted_on(
                &aux,
                &rhf_large,
                &cfg,
                &ferric_mp2::omega_ri::RiFit::Coulomb,
            ),
        );
        assert_reference_refused(
            "rpa_analytic_gradient_on",
            crate::gradient_analytic::rpa_analytic_gradient_on(
                &aux,
                &rhf_large,
                &cfg,
                crate::gradient_analytic::RingOrder::Full,
            ),
        );
        assert_reference_refused(
            "rs_mp2_lr_rpa_on",
            crate::rs_mp2_rpa::rs_mp2_lr_rpa_on(&aux, &rhf_large, &Default::default()),
        );
    }
}
