//! GW and BSE bindings: moved verbatim from `lib.rs`.

use super::*;

// ── GW (closed-shell: G0W0 / COHSEX / evGW0 / evGW) ──
//
// `run_u_gw` is wired below; `bse.rs`'s `run_bse_tda` is wired further down
// (see `run_cis_tda`/`run_bse_c6`/`run_bse_c6_ks`'s still-open note there) —
// see docs/open-work-triage-2026-07-14-open.md #54.

#[pyclass]
#[pyo3(name = "GwResult")]
pub(crate) struct PyGwResult {
    #[pyo3(get)]
    pub(crate) ref_energy: f64,
    /// MO indices (absolute) for which QP energies were computed.
    #[pyo3(get)]
    pub(crate) mo_indices: Vec<usize>,
    /// Mean-field (input) orbital energies for those MOs, Ha.
    pub(crate) eps_mf: Vec<f64>,
    /// QP energies (final), Ha.
    pub(crate) eps_qp: Vec<f64>,
    /// Exchange self-energy Σ_x (diagonal MO), Ha.
    pub(crate) sigma_x: Vec<f64>,
    /// Correlation self-energy Σ_c at the converged QP energy, Ha.
    pub(crate) sigma_c: Vec<f64>,
    /// Z-factor (renormalization), dimensionless.
    pub(crate) z_factor: Vec<f64>,
    /// Per-state QP Newton-solve convergence flag, aligned with `mo_indices`.
    /// `false` ⇒ that MO's `eps_qp`/`sigma_c`/`z_factor` is the Newton
    /// solver's last iterate, not a converged root. Always all-`true` for
    /// COHSEX (closed-form).
    #[pyo3(get)]
    pub(crate) qp_converged: Vec<bool>,
    /// evGW/evGW0 outer eigenvalue self-consistency iteration count (0 for
    /// G0W0/COHSEX).
    #[pyo3(get)]
    pub(crate) n_ev_iter: usize,
    /// Whether the evGW/evGW0 outer loop met `ev_conv_thresh` within
    /// `max_ev_iter`. Always `true` for G0W0/COHSEX.
    #[pyo3(get)]
    pub(crate) outer_converged: bool,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PyGwResult {
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    #[getter]
    fn eps_mf<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_mf)
    }
    #[getter]
    fn eps_qp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_qp)
    }
    #[getter]
    fn sigma_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.sigma_x)
    }
    #[getter]
    fn sigma_c<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.sigma_c)
    }
    #[getter]
    fn z_factor<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.z_factor)
    }
    fn __repr__(&self) -> String {
        format!(
            "GwResult(ref_energy={:.10}, n_qp={}, outer_converged={})",
            self.ref_energy,
            self.mo_indices.len(),
            self.outer_converged,
        )
    }
    fn __str__(&self) -> String {
        let n_unconverged = self.qp_converged.iter().filter(|&&c| !c).count();
        format!(
            "GW: ref_energy={:.10} Ha, {} QP states, {} unconverged QP, outer_converged={}, {} ev iterations",
            self.ref_energy, self.mo_indices.len(), n_unconverged, self.outer_converged, self.n_ev_iter,
        )
    }
}

/// Closed-shell G0W0/COHSEX/evGW0/evGW on an RHF or RKS reference.
///
/// `method`: "g0w0" (default) | "cohsex" | "evgw0" | "evgw" (case-insensitive;
/// unknown values are a hard `ValueError`, matching this repo's strict-config
/// convention elsewhere). `qp_mos`: optional `(lo, hi)` absolute-MO range;
/// unset uses the library default `{HOMO-2..LUMO+2}`. `xc`: `None` (default)
/// runs an HF reference; setting it (e.g. `"pbe"`) runs the closed-shell
/// KS-DFT solver first and folds Σx−vxc into the QP self-consistency (see
/// `ferric_gw::vxc_mo::vxc_diagonal_mo`).
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis,
    method=None, xc=None, qp_mos=None,
    max_ev_iter=None, ev_conv_thresh=None, pade_npts=None, qp_newton_damp=None,
    frozen_core=None, n_quad=None, quadrature=None, u0=None,
    trunc_thresh=None, eigensolver_conv_thresh=None,
    k_builder=None, chi0_sparsity=None, memory_budget_gb=None,
    scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, scf_ri_j_precision=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_gw(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    method: Option<&str>,
    xc: Option<&str>,
    qp_mos: Option<(usize, usize)>,
    max_ev_iter: Option<usize>,
    ev_conv_thresh: Option<f64>,
    pade_npts: Option<usize>,
    qp_newton_damp: Option<f64>,
    frozen_core: Option<usize>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    u0: Option<f64>,
    trunc_thresh: Option<f64>,
    eigensolver_conv_thresh: Option<f64>,
    k_builder: Option<&str>,
    chi0_sparsity: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    scf_ri_j_precision: Option<&str>,
) -> PyResult<PyGwResult> {
    use ferric_gw::{run_gw as run_gw_inner, vxc_mo::vxc_diagonal_mo_on_grids, GwConfig, GwMethod};
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
    use ferric_rpa::PdepRpaConfig;

    let gw_method = match method.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("g0w0") => GwMethod::G0W0,
        Some("cohsex") => GwMethod::Cohsex,
        Some("evgw0") => GwMethod::EvGw0,
        Some("evgw") => GwMethod::EvGw,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "method: unknown value \"{other}\"; expected \"g0w0\", \"cohsex\", \"evgw0\", or \"evgw\""
            )));
        }
    };

    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();

    let mut cfg = rhf_config_conv(
        k_builder,
        budget_bytes_from_gb(memory_budget_gb),
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    let vxc_diag = if let Some(xc_name) = xc {
        // KS reference: RI-J/RI-K default (matches run_dft/run_ksdft), run
        // through the level-shift ladder for the same DIIS-oscillation
        // fallback KS-DFT gets elsewhere.
        cfg.xc = Some(xc_name.to_string());
        cfg.df_j_aux = Some("def2-universal-jkfit".to_string());
        cfg.ri_j_precision = parse_ri_j_precision(scf_ri_j_precision)?;
        cfg.df_k_aux = Some("def2-universal-jkfit".to_string());
        let ladder = ferric_scf::ladder::ksdft_ladder(&cfg);
        let lr =
            ferric_scf::ladder::solve_rhf_ladder(&ctx, &mol.inner, &prep, op, &bounds, &ladder)
                .map_err(make_err)?;
        if let Some(r) = ladder.get(lr.rung_reached) {
            cfg = r.config.clone();
        }
        let scf = lr.result;
        if !scf.converged {
            return Err(make_err(ferric_core::FerricError::ScfConvergence {
                iterations: scf.iterations,
                last_energy: scf.energy,
            }));
        }
        let (diag, _beta) = {
            // The grids this reference SCF ran on.
            let grids = ferric_scf::ks_gradient::KsGradGrids::from_config(&cfg);
            vxc_diagonal_mo_on_grids(
                &mol.inner,
                &basis_set.inner,
                xc_name,
                &scf,
                &grids.main,
                &grids.nlc,
            )
            .map_err(make_err)?
        };
        (scf, Some(diag))
    } else {
        let scf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &cfg).map_err(make_err)?;
        if !scf.converged {
            return Err(make_err(ferric_core::FerricError::ScfConvergence {
                iterations: scf.iterations,
                last_energy: scf.energy,
            }));
        }
        (scf, None)
    };
    let (scf, vxc_diag) = vxc_diag;

    let scheme = QuadratureScheme::parse_config_str(quadrature)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("quadrature: {e}")))?;
    let fc = frozen_core.unwrap_or(0);
    let pdep_cfg = PdepRpaConfig {
        frozen_core: fc,
        trunc_thresh: trunc_thresh.unwrap_or(1e-4),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: lanczos_conv_thresh_or(eigensolver_conv_thresh, 1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: n_quad.unwrap_or(ferric_gw::DEFAULT_N_QUAD),
            u0: u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: false,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: ferric_rpa::config::Chi0Sparsity::parse_config_str(chi0_sparsity)
            .map_err(make_err)?,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        // run_gw forces this on internally; set explicitly for clarity too.
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        verbose: false,
    };
    let gw_cfg = GwConfig {
        method: gw_method,
        qp_mos: qp_mos.map(|(lo, hi)| lo..hi),
        max_ev_iter: max_ev_iter.unwrap_or(20),
        ev_conv_thresh: ev_conv_thresh.unwrap_or(1e-4),
        pade_npts: pade_npts.unwrap_or(0),
        qp_newton_damp: qp_newton_damp.unwrap_or(1.0),
        frozen_core: fc,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        verbose: false,
    };

    // All closure arguments are owned Rust values (an owned Molecule clone
    // plus locals built above) so nothing Python-borrowed crosses the
    // boundary; safe to release the GIL for the expensive GW compute.
    let emol_gw = mol.inner.clone();
    let r = py
        .allow_threads(|| {
            run_gw_inner(
                &emol_gw,
                &prep,
                &dfbs,
                op,
                &scf,
                &pdep_cfg,
                &gw_cfg,
                vxc_diag.as_ref(),
            )
        })
        .map_err(make_err)?;
    if !r.outer_converged {
        eprintln!(
            "warning: {:?} eigenvalue self-consistency did NOT converge in {} \
             iterations (thresh {:.1e}); QP energies are the last sweep",
            gw_cfg.method, r.n_ev_iter, gw_cfg.ev_conv_thresh
        );
    }
    let bad: Vec<usize> = r
        .qp_converged
        .iter()
        .enumerate()
        .filter(|(_, &c)| !c)
        .map(|(i, _)| r.mo_indices[i])
        .collect();
    if !bad.is_empty() {
        eprintln!(
            "warning: QP Newton solve did not converge for MO(s) {bad:?}; \
             those QP energies are best-effort"
        );
    }
    let certificate_data = ferric_gw::certificate::gw_certificate(
        &reference_certificate(&mol.inner, &scf, &cfg, &basis_set.inner),
        &auxbasis.inner.name,
        &pdep_cfg,
        &gw_cfg,
        &r,
    );
    Ok(PyGwResult {
        certificate_data,
        ref_energy: scf.energy,
        mo_indices: r.mo_indices,
        eps_mf: r.eps_mf.to_vec(),
        eps_qp: r.eps_qp.to_vec(),
        sigma_x: r.sigma_x.to_vec(),
        sigma_c: r.sigma_c.to_vec(),
        z_factor: r.z_factor.to_vec(),
        qp_converged: r.qp_converged,
        n_ev_iter: r.n_ev_iter,
        outer_converged: r.outer_converged,
    })
}

// ── U-GW (open-shell: U-G0W0 / U-COHSEX / U-evGW0 / U-evGW) ──
//
// BSE-TDA (`run_bse_tda`) is closed-shell only and wired further down
// (see the "── BSE-TDA ──" section) — see docs/open-work-triage-2026-07-14-open.md #54.

#[pyclass]
#[pyo3(name = "UGwResult")]
pub(crate) struct PyUGwResult {
    #[pyo3(get)]
    pub(crate) ref_energy: f64,
    /// MO indices (absolute) for which QP energies were computed, shared by
    /// both spin channels.
    #[pyo3(get)]
    pub(crate) mo_indices: Vec<usize>,
    pub(crate) eps_mf_a: Vec<f64>,
    pub(crate) eps_qp_a: Vec<f64>,
    pub(crate) sigma_x_a: Vec<f64>,
    pub(crate) sigma_c_a: Vec<f64>,
    pub(crate) z_factor_a: Vec<f64>,
    pub(crate) eps_mf_b: Vec<f64>,
    pub(crate) eps_qp_b: Vec<f64>,
    pub(crate) sigma_x_b: Vec<f64>,
    pub(crate) sigma_c_b: Vec<f64>,
    pub(crate) z_factor_b: Vec<f64>,
    /// Per-state QP Newton-solve convergence flags, aligned with `mo_indices`
    /// (see `PyGwResult::qp_converged` for the per-flag meaning). Always
    /// all-`true` for COHSEX.
    #[pyo3(get)]
    pub(crate) qp_converged_a: Vec<bool>,
    #[pyo3(get)]
    pub(crate) qp_converged_b: Vec<bool>,
    /// evGW/evGW0 outer eigenvalue self-consistency iteration count (0 for
    /// G0W0/COHSEX).
    #[pyo3(get)]
    pub(crate) n_ev_iter: usize,
    /// Whether the U-evGW/U-evGW0 outer loop met `ev_conv_thresh` within
    /// `max_ev_iter`. Always `true` for U-G0W0/U-COHSEX.
    #[pyo3(get)]
    pub(crate) outer_converged: bool,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PyUGwResult {
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    #[getter]
    fn eps_mf_a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_mf_a)
    }
    #[getter]
    fn eps_qp_a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_qp_a)
    }
    #[getter]
    fn sigma_x_a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.sigma_x_a)
    }
    #[getter]
    fn sigma_c_a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.sigma_c_a)
    }
    #[getter]
    fn z_factor_a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.z_factor_a)
    }
    #[getter]
    fn eps_mf_b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_mf_b)
    }
    #[getter]
    fn eps_qp_b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_qp_b)
    }
    #[getter]
    fn sigma_x_b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.sigma_x_b)
    }
    #[getter]
    fn sigma_c_b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.sigma_c_b)
    }
    #[getter]
    fn z_factor_b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.z_factor_b)
    }
    fn __repr__(&self) -> String {
        format!(
            "UGwResult(ref_energy={:.10}, n_qp={}, outer_converged={})",
            self.ref_energy,
            self.mo_indices.len(),
            self.outer_converged,
        )
    }
    fn __str__(&self) -> String {
        let n_unconverged_a = self.qp_converged_a.iter().filter(|&&c| !c).count();
        let n_unconverged_b = self.qp_converged_b.iter().filter(|&&c| !c).count();
        format!(
            "U-GW: ref_energy={:.10} Ha, {} QP states, unconverged QP: {} alpha/{} beta, outer_converged={}, {} ev iterations",
            self.ref_energy, self.mo_indices.len(), n_unconverged_a, n_unconverged_b, self.outer_converged, self.n_ev_iter,
        )
    }
}

/// Open-shell U-G0W0/U-COHSEX/U-evGW0/U-evGW on a UHF/UKS or ROHF reference.
///
/// `reference`: "uhf" (default) | "rohf" (case-insensitive; unknown values
/// are a hard `ValueError`). `method`/`xc`/`qp_mos`/... mirror `run_gw`'s
/// kwarg shape exactly; `xc` set runs the open-shell KS-DFT ladder (UKS) and
/// passes the per-spin v_xc diagonals to `ferric_gw::run_u_gw`, which puts
/// Σx−vxc inside each spin channel's QP equation (same contract as `run_gw`).
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis,
    reference=None, method=None, xc=None, qp_mos=None,
    max_ev_iter=None, ev_conv_thresh=None, pade_npts=None, qp_newton_damp=None,
    frozen_core=None, n_quad=None, quadrature=None, u0=None,
    trunc_thresh=None, eigensolver_conv_thresh=None,
    k_builder=None, chi0_sparsity=None, memory_budget_gb=None,
    scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, scf_ri_j_precision=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_u_gw(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    reference: Option<&str>,
    method: Option<&str>,
    xc: Option<&str>,
    qp_mos: Option<(usize, usize)>,
    max_ev_iter: Option<usize>,
    ev_conv_thresh: Option<f64>,
    pade_npts: Option<usize>,
    qp_newton_damp: Option<f64>,
    frozen_core: Option<usize>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    u0: Option<f64>,
    trunc_thresh: Option<f64>,
    eigensolver_conv_thresh: Option<f64>,
    k_builder: Option<&str>,
    chi0_sparsity: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    scf_ri_j_precision: Option<&str>,
) -> PyResult<PyUGwResult> {
    use ferric_gw::{
        run_u_gw as run_u_gw_inner, vxc_mo::vxc_diagonal_mo_on_grids, GwConfig, GwMethod,
    };
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
    use ferric_rpa::PdepRpaConfig;
    use ferric_scf::rohf::solve_rohf;

    let gw_method = match method.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("g0w0") => GwMethod::G0W0,
        Some("cohsex") => GwMethod::Cohsex,
        Some("evgw0") => GwMethod::EvGw0,
        Some("evgw") => GwMethod::EvGw,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "method: unknown value \"{other}\"; expected \"g0w0\", \"cohsex\", \"evgw0\", or \"evgw\""
            )));
        }
    };
    let reference = match reference.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("uhf") => "uhf",
        Some("rohf") => "rohf",
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "reference: unknown value \"{other}\"; expected \"uhf\" or \"rohf\""
            )));
        }
    };

    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();

    let mut cfg = rhf_config_conv(
        k_builder,
        budget_bytes_from_gb(memory_budget_gb),
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    // MOM after 5 DIIS iters prevents orbital reordering on open-shell atoms
    // (same precedent as the CLI's "pdep-rpa"/"gw" open-shell dispatch).
    cfg.mom_after_iter = 5;
    let vxc_diag = if let Some(xc_name) = xc {
        cfg.xc = Some(xc_name.to_string());
        cfg.df_j_aux = Some("def2-universal-jkfit".to_string());
        cfg.ri_j_precision = parse_ri_j_precision(scf_ri_j_precision)?;
        cfg.df_k_aux = Some("def2-universal-jkfit".to_string());
        let scf = if reference == "rohf" {
            solve_rohf(&ctx, &mol.inner, &prep, op, &bounds, &cfg).map_err(make_err)?
        } else {
            solve_uhf(&ctx, &mol.inner, &prep, &bounds, &cfg).map_err(make_err)?
        };
        if !scf.converged {
            return Err(make_err(ferric_core::FerricError::ScfConvergence {
                iterations: scf.iterations,
                last_energy: scf.energy,
            }));
        }
        // The certificate's reference is the SCF as solved (a ROKS reference
        // is graded `rohf`), before the semi-canonical conversion below.
        let ref_cert = reference_certificate(&mol.inner, &scf, &cfg, &basis_set.inner);
        // ROKS -> semi-canonical per-spin orbitals (what run_u_gw uses
        // internally), so v_xc is diagonalized in the QP equation's own basis;
        // UKS is unchanged.
        let scf = ferric_scf::semicanonical::unrestricted_reference(&mol.inner, &scf)
            .map_err(make_err)?
            .into_owned();
        let (diag_a, diag_b) = {
            // The grids this reference SCF ran on.
            let grids = ferric_scf::ks_gradient::KsGradGrids::from_config(&cfg);
            vxc_diagonal_mo_on_grids(
                &mol.inner,
                &basis_set.inner,
                xc_name,
                &scf,
                &grids.main,
                &grids.nlc,
            )
            .map_err(make_err)?
        };
        (scf, Some((diag_a, diag_b)), ref_cert)
    } else {
        let scf = if reference == "rohf" {
            solve_rohf(&ctx, &mol.inner, &prep, op, &bounds, &cfg).map_err(make_err)?
        } else {
            solve_uhf(&ctx, &mol.inner, &prep, &bounds, &cfg).map_err(make_err)?
        };
        if !scf.converged {
            return Err(make_err(ferric_core::FerricError::ScfConvergence {
                iterations: scf.iterations,
                last_energy: scf.energy,
            }));
        }
        let ref_cert = reference_certificate(&mol.inner, &scf, &cfg, &basis_set.inner);
        (scf, None, ref_cert)
    };
    let (scf, vxc_diag, reference_cert) = vxc_diag;

    let scheme = QuadratureScheme::parse_config_str(quadrature)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("quadrature: {e}")))?;
    let fc = frozen_core.unwrap_or(0);
    let pdep_cfg = PdepRpaConfig {
        frozen_core: fc,
        trunc_thresh: trunc_thresh.unwrap_or(1e-4),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: lanczos_conv_thresh_or(eigensolver_conv_thresh, 1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: n_quad.unwrap_or(ferric_gw::DEFAULT_N_QUAD),
            u0: u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: false,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: ferric_rpa::config::Chi0Sparsity::parse_config_str(chi0_sparsity)
            .map_err(make_err)?,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        // run_u_gw forces this on internally; set explicitly for clarity too.
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        verbose: false,
    };
    let gw_cfg = GwConfig {
        method: gw_method,
        qp_mos: qp_mos.map(|(lo, hi)| lo..hi),
        max_ev_iter: max_ev_iter.unwrap_or(20),
        ev_conv_thresh: ev_conv_thresh.unwrap_or(1e-4),
        pade_npts: pade_npts.unwrap_or(0),
        qp_newton_damp: qp_newton_damp.unwrap_or(1.0),
        frozen_core: fc,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        verbose: false,
    };

    let r = run_u_gw_inner(
        &mol.inner,
        &prep,
        &dfbs,
        op,
        &scf,
        &pdep_cfg,
        &gw_cfg,
        vxc_diag.as_ref().map(|(a, b)| (a, b)),
    )
    .map_err(make_err)?;
    if !r.outer_converged {
        eprintln!(
            "warning: U-{:?} eigenvalue self-consistency did NOT converge in {} \
             iterations (thresh {:.1e}); QP energies are the last sweep",
            gw_cfg.method, r.n_ev_iter, gw_cfg.ev_conv_thresh
        );
    }
    for (spin_label, flags) in [("alpha", &r.qp_converged_a), ("beta", &r.qp_converged_b)] {
        let bad: Vec<usize> = flags
            .iter()
            .enumerate()
            .filter(|(_, &c)| !c)
            .map(|(i, _)| r.mo_indices[i])
            .collect();
        if !bad.is_empty() {
            eprintln!(
                "warning: QP Newton solve did not converge for {spin_label} MO(s) {bad:?}; \
                 those QP energies are best-effort"
            );
        }
    }
    let certificate_data = ferric_gw::certificate::u_gw_certificate(
        &reference_cert,
        &auxbasis.inner.name,
        &pdep_cfg,
        &gw_cfg,
        &r,
        vxc_diag.is_some(),
    );
    Ok(PyUGwResult {
        certificate_data,
        ref_energy: scf.energy,
        mo_indices: r.mo_indices,
        eps_mf_a: r.eps_mf_a.to_vec(),
        eps_qp_a: r.eps_qp_a.to_vec(),
        sigma_x_a: r.sigma_x_a.to_vec(),
        sigma_c_a: r.sigma_c_a.to_vec(),
        z_factor_a: r.z_factor_a.to_vec(),
        eps_mf_b: r.eps_mf_b.to_vec(),
        eps_qp_b: r.eps_qp_b.to_vec(),
        sigma_x_b: r.sigma_x_b.to_vec(),
        sigma_c_b: r.sigma_c_b.to_vec(),
        z_factor_b: r.z_factor_b.to_vec(),
        qp_converged_a: r.qp_converged_a,
        qp_converged_b: r.qp_converged_b,
        n_ev_iter: r.n_ev_iter,
        outer_converged: r.outer_converged,
    })
}

// ── BSE-TDA (closed-shell singlet excitation energies on a G0W0@HF reference) ──
//
// `run_cis_tda`/`run_bse_c6`/`run_bse_c6_ks` are NOT wired here — see
// docs/open-work-triage-2026-07-14-open.md #54. `run_cis_tda` is a
// diagnostic-only assembly cross-check (not a production entry point, per
// its own doc comment in `ferric_gw::bse`); the C6/dispersion variants are
// lower-priority per the G9 task brief and still open.

#[pyclass]
#[pyo3(name = "BseResult")]
pub(crate) struct PyBseResult {
    /// Number of occupied / virtual orbitals in the BSE (ia) window
    /// (frozen-core aware).
    #[pyo3(get)]
    pub(crate) nocc: usize,
    #[pyo3(get)]
    pub(crate) nvir: usize,
    /// Singlet excitation energies Ω_n (Hartree), ascending.
    pub(crate) omega: Vec<f64>,
    /// GW quasiparticle energies used for the diagonal (active block, Ha).
    pub(crate) eps_qp: Vec<f64>,
    /// Length-gauge oscillator strengths f_n (dimensionless), same ordering
    /// as `omega`. See `ferric_gw::bse::tda_oscillator_strengths` for the
    /// convention (PySCF-cross-checked, see
    /// `crates/ferric-gw/tests/bse_oscillator_strength.rs`).
    pub(crate) oscillator_strength: Vec<f64>,
    /// Absolute MO indices whose G0W0 quasiparticle solve is poorly determined
    /// (Newton did not converge, or the Z renormalization was clamped) and
    /// which still entered the TDA diagonal. Observability only: `omega` is
    /// unaffected, and the measured effect of these MOs on the lowest
    /// excitations is ~4 orders of magnitude below their own uncertainty.
    pub(crate) qp_suspect_mos: Vec<usize>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PyBseResult {
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    #[getter]
    fn omega<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.omega)
    }
    #[getter]
    fn eps_qp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_qp)
    }
    #[getter]
    fn oscillator_strength<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.oscillator_strength)
    }
    /// Absolute MO indices with a poorly-determined quasiparticle energy on
    /// the TDA diagonal (empty in the healthy case).
    #[getter]
    fn qp_suspect_mos(&self) -> Vec<usize> {
        self.qp_suspect_mos.clone()
    }
    /// Lowest singlet excitation energy in eV.
    fn lowest_ev(&self) -> f64 {
        self.omega[0] * 27.211_386_245_988
    }
    /// Oscillator strength of the lowest singlet.
    fn lowest_oscillator_strength(&self) -> f64 {
        self.oscillator_strength[0]
    }
    fn __repr__(&self) -> String {
        format!(
            "BseResult(nocc={}, nvir={}, n_excitations={})",
            self.nocc,
            self.nvir,
            self.omega.len(),
        )
    }
    fn __str__(&self) -> String {
        let lowest_ev = if self.omega.is_empty() {
            "N/A".to_string()
        } else {
            format!("{:.4}", self.omega[0] * 27.211_386_245_988)
        };
        format!(
            "BSE-TDA: {} excitations, lowest={} eV, nocc={}, nvir={}",
            self.omega.len(),
            lowest_ev,
            self.nocc,
            self.nvir,
        )
    }
}

/// BSE-TDA singlet excitation energies on a closed-shell (RHF) reference.
///
/// Computes G0W0@HF quasiparticle energies internally for every MO (so every
/// particle–hole pair in the active window has a real QP energy), builds the
/// static-screened TDA matrix, and returns its eigenvalues. Closed-shell
/// only — `ferric_gw::bse::run_bse_tda` hard-errors on a non-restricted
/// reference; there is no `reference`/UHF kwarg (mirrors `run_gw`'s HF-only
/// default path, not `run_u_gw`). Kwargs mirror `run_gw`'s `[rpa]`-equivalent
/// shape (n_quad/quadrature/u0/trunc_thresh/eigensolver_conv_thresh/
/// k_builder/chi0_sparsity/memory_budget_gb) plus `frozen_core`, which is
/// threaded to both the PDEP (W) build and the internal GW self-energy build
/// for consistency, exactly like the CLI's `"bse-tda"` dispatch arm.
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis,
    frozen_core=None, n_quad=None, quadrature=None, u0=None,
    trunc_thresh=None, eigensolver_conv_thresh=None,
    k_builder=None, chi0_sparsity=None, memory_budget_gb=None,
    scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_bse_tda(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    u0: Option<f64>,
    trunc_thresh: Option<f64>,
    eigensolver_conv_thresh: Option<f64>,
    k_builder: Option<&str>,
    chi0_sparsity: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyBseResult> {
    use ferric_gw::bse::run_bse_tda as run_bse_tda_inner;
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
    use ferric_rpa::PdepRpaConfig;

    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();

    let cfg = rhf_config_conv(
        k_builder,
        budget_bytes_from_gb(memory_budget_gb),
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    let scf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &cfg).map_err(make_err)?;
    if !scf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: scf.iterations,
            last_energy: scf.energy,
        }));
    }

    let scheme = QuadratureScheme::parse_config_str(quadrature)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("quadrature: {e}")))?;
    let fc = frozen_core.unwrap_or(0);
    let pdep_cfg = PdepRpaConfig {
        frozen_core: fc,
        trunc_thresh: trunc_thresh.unwrap_or(1e-4),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: lanczos_conv_thresh_or(eigensolver_conv_thresh, 1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: n_quad.unwrap_or(20),
            u0: u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: false,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: ferric_rpa::config::Chi0Sparsity::parse_config_str(chi0_sparsity)
            .map_err(make_err)?,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        // run_bse_tda's internal GW build forces this on regardless; set it
        // explicitly for clarity at the call site too (matches run_gw).
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        verbose: false,
    };

    let r =
        run_bse_tda_inner(&mol.inner, &prep, &dfbs, op, &scf, &pdep_cfg, fc).map_err(make_err)?;
    let certificate_data = ferric_gw::certificate::bse_tda_certificate(
        &reference_certificate(&mol.inner, &scf, &cfg, &basis_set.inner),
        &auxbasis.inner.name,
        &pdep_cfg,
        &r,
    );
    Ok(PyBseResult {
        certificate_data,
        nocc: r.nocc,
        nvir: r.nvir,
        omega: r.omega,
        eps_qp: r.eps_qp,
        oscillator_strength: r.oscillator_strength,
        qp_suspect_mos: r.qp_suspect_mos,
    })
}

// ── TDHF/RPAx static polarizability (closed-shell, KS reference) ──
//
// `run_bse_c6`/`run_bse_c6_ks` (the dynamic alpha(iw)/C6 variants of this same
// kernel) are deliberately NOT wired here. docs/VALIDATION.md's "Correlation /
// response (RPA, GW, BSE)" table records a validated negative result: C6 from
// alpha(iw) on this kernel stays ~63% low regardless of the HOMO-LUMO gap,
// worse than ferric's production dRPA/PDEP C6 pipeline. Only the STATIC
// (omega=0) polarizability is exposed as a production entry point.

#[pyclass]
#[pyo3(name = "TdhfStaticPolarizabilityResult")]
pub(crate) struct PyTdhfStaticPolarizabilityResult {
    #[pyo3(get)]
    pub(crate) nocc: usize,
    #[pyo3(get)]
    pub(crate) nvir: usize,
    /// Isotropic average (1/3) Tr(alpha), a.u.
    #[pyo3(get)]
    pub(crate) iso: f64,
    pub(crate) tensor: [[f64; 3]; 3],
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PyTdhfStaticPolarizabilityResult {
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    /// Cartesian alpha_ij(0) tensor (3x3, a.u.), i,j in {x,y,z}.
    #[getter]
    fn tensor<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let flat: Vec<f64> = self
            .tensor
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let arr = Array2::from_shape_vec((3, 3), flat).expect("3x3 tensor is always well-shaped");
        PyArray2::from_array(py, &arr)
    }
    fn __repr__(&self) -> String {
        format!(
            "TdhfStaticPolarizabilityResult(iso={:.6}, nocc={}, nvir={})",
            self.iso, self.nocc, self.nvir,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "TDHF Static Polarizability: iso={:.6} a.u., nocc={}, nvir={}",
            self.iso, self.nocc, self.nvir,
        )
    }
}

/// RPAx@KS **static** polarizability (omega=0) on a closed-shell KS reference.
///
/// **Scope: static polarizability only.** Do not use this method, or its
/// output, for C6/dispersion — see the module doc above and
/// `ferric_gw::bse::run_rpax_static_polarizability`'s doc comment for the
/// full negative-result caveat (site/src/reference/validation.md, "Known
/// limits and negatives": C6 built
/// from this same kernel's dynamic alpha(iw) stays ~63% low regardless of
/// the HOMO-LUMO gap, worse than ferric's production dRPA/PDEP C6 pipeline).
///
/// `xc` is REQUIRED (e.g. `"pbe"`): the path is wired and checked for a KS
/// reference only, so there is no HF-default fallback here (unlike
/// `run_gw`'s `xc=None` HF default). Static alpha is NOT validated: on
/// water/cc-pVDZ/PBE with `scissor=0.36` it is 5.20 a.u. against the DOSD
/// reference 9.64 (46% low). The earlier "9.24 vs 9.64" figure came from
/// `scissor=0.0`, whose tensor had a negative diagonal element, and is
/// retracted. `scissor` (Hartree, default 0.0) is added to every virtual
/// orbital energy before assembling the diagonal — a cheap proxy for
/// widening the KS gap toward a GW-level gap (0.36 Ha matches water's GW gap
/// at PBE/cc-pVDZ). At 0.0 the kernel is often excitonically unstable and
/// the call raises on the resulting negative alpha diagonal; ~0.3-0.4 Ha is
/// the remedy.
/// Other kwargs mirror `run_bse_tda`'s `[rpa]`-equivalent shape (n_quad/
/// quadrature/u0/trunc_thresh/eigensolver_conv_thresh/k_builder/
/// chi0_sparsity/memory_budget_gb) plus `frozen_core`.
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis, xc,
    scissor=None, frozen_core=None, n_quad=None, quadrature=None, u0=None,
    trunc_thresh=None, eigensolver_conv_thresh=None,
    k_builder=None, chi0_sparsity=None, memory_budget_gb=None,
    scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, scf_ri_j_precision=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_tdhf_static_polarizability(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    xc: &str,
    scissor: Option<f64>,
    frozen_core: Option<usize>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    u0: Option<f64>,
    trunc_thresh: Option<f64>,
    eigensolver_conv_thresh: Option<f64>,
    k_builder: Option<&str>,
    chi0_sparsity: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    scf_ri_j_precision: Option<&str>,
) -> PyResult<PyTdhfStaticPolarizabilityResult> {
    use ferric_gw::bse::run_rpax_static_polarizability;
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
    use ferric_rpa::PdepRpaConfig;

    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();

    // KS reference (required — see doc comment above), same ladder path
    // run_gw's xc branch uses.
    let mut cfg = rhf_config_conv(
        k_builder,
        budget_bytes_from_gb(memory_budget_gb),
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    cfg.xc = Some(xc.to_string());
    cfg.df_j_aux = Some("def2-universal-jkfit".to_string());
    cfg.ri_j_precision = parse_ri_j_precision(scf_ri_j_precision)?;
    cfg.df_k_aux = Some("def2-universal-jkfit".to_string());
    let ladder = ferric_scf::ladder::ksdft_ladder(&cfg);
    let lr = ferric_scf::ladder::solve_rhf_ladder(&ctx, &mol.inner, &prep, op, &bounds, &ladder)
        .map_err(make_err)?;
    if let Some(r) = ladder.get(lr.rung_reached) {
        cfg = r.config.clone();
    }
    let scf = lr.result;
    if !scf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: scf.iterations,
            last_energy: scf.energy,
        }));
    }

    let scheme = QuadratureScheme::parse_config_str(quadrature)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("quadrature: {e}")))?;
    let fc = frozen_core.unwrap_or(0);
    let pdep_cfg = PdepRpaConfig {
        frozen_core: fc,
        trunc_thresh: trunc_thresh.unwrap_or(1e-4),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: lanczos_conv_thresh_or(eigensolver_conv_thresh, 1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: n_quad.unwrap_or(20),
            u0: u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: false,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: ferric_rpa::config::Chi0Sparsity::parse_config_str(chi0_sparsity)
            .map_err(make_err)?,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        // No GW self-energy build in this path (static screening modes from
        // run_pdep_rpa only) — does not need the inverse-dielectric
        // frequency stack that run_bse_tda/run_gw force on.
        need_inv_dielectric_freq: false,
        need_eigenvalues_freq: true,
        verbose: false,
    };

    let r = run_rpax_static_polarizability(
        &mol.inner,
        &prep,
        &dfbs,
        op,
        &scf,
        &pdep_cfg,
        fc,
        scissor.unwrap_or(0.0),
    )
    .map_err(make_err)?;
    let certificate_data = ferric_gw::certificate::tdhf_static_polarizability_certificate(
        &reference_certificate(&mol.inner, &scf, &cfg, &basis_set.inner),
        &auxbasis.inner.name,
        &pdep_cfg,
        scissor.unwrap_or(0.0),
    );
    Ok(PyTdhfStaticPolarizabilityResult {
        certificate_data,
        nocc: r.nocc,
        nvir: r.nvir,
        iso: r.iso,
        tensor: r.tensor,
    })
}
