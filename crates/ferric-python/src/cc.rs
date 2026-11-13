//! Coupled-cluster bindings: moved verbatim from `lib.rs`.

use super::*;

// ── CC (stub) ──

#[pyclass]
#[pyo3(name = "CcResult")]
pub(crate) struct PyCcResult {
    #[pyo3(get)]
    pub(crate) correlation_energy: f64,
    #[pyo3(get)]
    pub(crate) t_correction: Option<f64>,
    /// Closed-shell CCSD's particle-particle ladder route: `"dense"` or
    /// `"ri"` (`None` for CCD). With `ladder="auto"` this says which ran.
    #[pyo3(get)]
    pub(crate) ladder_route: Option<String>,
    /// The RI ladder's `a`-block width (`None` unless `ladder_route == "ri"`).
    #[pyo3(get)]
    pub(crate) ladder_block: Option<usize>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

/// `(ladder_route, ladder_block)` for the Python result.
pub(crate) fn py_ladder_route(r: &ferric_cc::CcResult) -> (Option<String>, Option<usize>) {
    use ferric_cc::ccsd_closed_shell::LadderRoute;
    match r.ladder_route {
        Some(LadderRoute::Dense) => (Some("dense".into()), None),
        Some(LadderRoute::Ri { block }) => (Some("ri".into()), Some(block)),
        None => (None, None),
    }
}

#[pymethods]
impl PyCcResult {
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    fn __repr__(&self) -> String {
        match self.t_correction {
            Some(t) => format!(
                "CcResult(correlation_energy={:.10}, t_correction={:.10})",
                self.correlation_energy, t,
            ),
            None => format!(
                "CcResult(correlation_energy={:.10})",
                self.correlation_energy,
            ),
        }
    }
    fn __str__(&self) -> String {
        match self.t_correction {
            Some(t) => format!(
                "CC Correlation Energy: {:.10} Ha, (T) correction: {:.10} Ha, total corr+T: {:.10} Ha",
                self.correlation_energy, t, self.correlation_energy + t,
            ),
            None => format!(
                "CC Correlation Energy: {:.10} Ha",
                self.correlation_energy,
            ),
        }
    }
}

/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None, memory_budget_gb=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_ccd(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyCcResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf_config = rhf_config_conv(
        k_builder,
        budget_bytes_from_gb(memory_budget_gb),
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        }));
    }
    let cfg = CcConfig {
        frozen_core: frozen_core.unwrap_or(0),
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        ..Default::default()
    };
    let r = run_ccd_inner(&mol.inner, &prep, &dfbs, op, &rhf, &cfg).map_err(make_err)?;
    let certificate_data = ferric_cc::certificate::cc_certificate(
        "ccd",
        &reference_certificate(&mol.inner, &rhf, &rhf_config, &basis_set.inner),
        &auxbasis.inner.name,
        &cfg,
        &r,
    );
    Ok(PyCcResult {
        correlation_energy: r.correlation_energy,
        t_correction: None,
        ladder_route: None,
        ladder_block: None,
        certificate_data,
    })
}

/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None, memory_budget_gb=None, ladder="auto", scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_ccsd(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    ladder: &str,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyCcResult> {
    let emol = mol.inner.clone();
    let ebasis = basis_set.inner.clone();
    let eaux = auxbasis.inner.clone();
    let rhf_config = rhf_config_conv(
        k_builder,
        budget_bytes_from_gb(memory_budget_gb),
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    let cfg = CcConfig {
        frozen_core: frozen_core.unwrap_or(0),
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        // Strict: an unknown route is a ValueError before any SCF runs.
        ladder: ferric_cc::CcsdLadder::parse_config_str(ladder)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        ..Default::default()
    };
    // Release the GIL for the SCF + CCSD compute. All captured values are
    // owned clones/locals built above, so nothing Python-borrowed crosses
    // the closure boundary.
    //
    // `solve_rhf` always yields a restricted reference, so this is the
    // spin-adapted path unconditionally: same CCSD energy, but over spatial
    // orbitals (no/nv) rather than spin orbitals (2no/2nv), so the O(N^6) VVVV
    // block is 16x smaller. Measured on water/aug-cc-pVDZ: 24.6 s -> 1.10 s
    // (22x) and peak RSS 1.82 GB -> 0.48 GB.
    //
    // NOTE `run_ccsd_t` below deliberately does NOT switch — see its comment.
    let r = py
        .allow_threads(|| -> Result<_, ferric_core::FerricError> {
            let prep = PreparedBasis::new(&emol, &ebasis)?;
            let dfbs = PreparedBasis::new(&emol, &eaux)?;
            let op = Operator::coulomb();
            let bounds = SchwarzBounds::compute(op, &prep)?;
            let ctx = ParallelContext::default();
            let rhf = solve_rhf(&ctx, &emol, &prep, op, &bounds, &rhf_config)?;
            if !rhf.converged {
                return Err(ferric_core::FerricError::ScfConvergence {
                    iterations: rhf.iterations,
                    last_energy: rhf.energy,
                });
            }
            let reference = reference_certificate(&emol, &rhf, &rhf_config, &basis_set.inner);
            Ok((
                run_ccsd_cs_inner(&emol, &prep, &dfbs, op, &rhf, &cfg)?,
                reference,
            ))
        })
        .map_err(make_err)?;
    let (r, reference) = r;
    let (ladder_route, ladder_block) = py_ladder_route(&r);
    Ok(PyCcResult {
        correlation_energy: r.correlation_energy,
        t_correction: None,
        ladder_route,
        ladder_block,
        certificate_data: ferric_cc::certificate::cc_certificate(
            "ccsd", &reference, &eaux.name, &cfg, &r,
        ),
    })
}

/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None, memory_budget_gb=None, ladder="auto", scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_ccsd_t(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    ladder: &str,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyCcResult> {
    let emol = mol.inner.clone();
    let ebasis = basis_set.inner.clone();
    let eaux = auxbasis.inner.clone();
    let rhf_config = rhf_config_conv(
        k_builder,
        budget_bytes_from_gb(memory_budget_gb),
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    let cfg = CcConfig {
        frozen_core: frozen_core.unwrap_or(0),
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        // Strict: an unknown route is a ValueError before any SCF runs.
        ladder: ferric_cc::CcsdLadder::parse_config_str(ladder)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        ..Default::default()
    };
    // Release the GIL for the SCF + CCSD + (T) compute. All captured values
    // are owned clones/locals built above, so nothing Python-borrowed
    // crosses the closure boundary.
    //
    // `solve_rhf` always yields a restricted reference, so both halves take the
    // spin-adapted path: `ccsd_closed_shell` for the amplitudes and
    // `ccsd_t_closed_shell` for the triples. Both work over spatial orbitals
    // (no/nv) instead of spin orbitals (2no/2nv).
    //
    // The amplitudes flow straight through — no `expand_amplitudes_to_spin_orbital`
    // round-trip, since the spin-adapted (T) consumes the spatial convention
    // natively (and hard-rejects spin-orbital amplitudes with a typed error).
    //
    // Measured on water/aug-cc-pVDZ: CCSD 24.8 -> 0.70 s and (T) 1.87 -> 0.14 s,
    // i.e. ~26.6 s -> under a second for the pair. The two (T) implementations
    // agree to 1.9e-16 when fed identical amplitudes, and to 7.8e-10..6.9e-7
    // against PySCF's `ccsd_t()` — inside the ~5e-6 Ha RI floor.
    let (r_cs, e_t, reference) = py
        .allow_threads(|| -> Result<_, ferric_core::FerricError> {
            let prep = PreparedBasis::new(&emol, &ebasis)?;
            let dfbs = PreparedBasis::new(&emol, &eaux)?;
            let op = Operator::coulomb();
            let bounds = SchwarzBounds::compute(op, &prep)?;
            let ctx = ParallelContext::default();
            let rhf = solve_rhf(&ctx, &emol, &prep, op, &bounds, &rhf_config)?;
            if !rhf.converged {
                return Err(ferric_core::FerricError::ScfConvergence {
                    iterations: rhf.iterations,
                    last_energy: rhf.energy,
                });
            }
            let r_cs = run_ccsd_cs_inner(&emol, &prep, &dfbs, op, &rhf, &cfg)?;
            let e_t = run_ccsd_t_cs_inner(&emol, &prep, &dfbs, op, &rhf, &r_cs, &cfg)?;
            let reference = reference_certificate(&emol, &rhf, &rhf_config, &basis_set.inner);
            Ok((r_cs, e_t, reference))
        })
        .map_err(make_err)?;
    let (ladder_route, ladder_block) = py_ladder_route(&r_cs);
    Ok(PyCcResult {
        correlation_energy: r_cs.correlation_energy,
        t_correction: Some(e_t),
        ladder_route,
        ladder_block,
        certificate_data: ferric_cc::certificate::cc_certificate(
            "ccsd(t)", &reference, &eaux.name, &cfg, &r_cs,
        ),
    })
}
