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
    /// The reference the CC ran on: `"rhf"`, or `"uhf"` for an open-shell
    /// molecule (spin-blocked UCCSD / UCCSD(T)).
    #[pyo3(get)]
    pub(crate) reference: String,
    /// Spin components of a UHF-reference run, in Hartree: `e_aa`, `e_ab`,
    /// `e_bb` (CCSD) and, for `run_ccsd_t`, `t_aaa`, `t_aab`, `t_abb`,
    /// `t_bbb` (the (T) classes by occupied spins). `None` on an RHF
    /// reference.
    #[pyo3(get)]
    pub(crate) spin_components: Option<std::collections::HashMap<String, f64>>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

/// Which coupled-cluster energy `run_ccsd` / `run_ccsd_t` compute.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CcLevel {
    Ccsd,
    CcsdT,
}

impl CcLevel {
    /// The certificate's method kind.
    fn kind(self) -> &'static str {
        match self {
            CcLevel::Ccsd => "ccsd",
            CcLevel::CcsdT => "ccsd(t)",
        }
    }
}

/// Spin-blocked UCCSD and, at `CcLevel::CcsdT`, UCCSD(T) on a UHF reference
/// of the open-shell `mol` (the open-shell branch of `run_ccsd` /
/// `run_ccsd_t`). Returns the result for Python, built without the GIL.
fn run_u_ccsd_inner(
    mol: &Molecule,
    basis: &ferric_core::basis::BasisSet,
    aux: &ferric_core::basis::BasisSet,
    scf_config: &RhfConfig,
    cfg: &CcConfig,
    level: CcLevel,
) -> Result<PyCcResult, ferric_core::FerricError> {
    let prep = PreparedBasis::new(mol, basis)?;
    let dfbs = PreparedBasis::new(mol, aux)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep)?;
    let ctx = ParallelContext::default();
    let uhf = require_converged(solve_uhf(&ctx, mol, &prep, &bounds, scf_config)?)?;
    let cc = ferric_cc::u_ccsd::u_ccsd(mol, &prep, &dfbs, op, &uhf, cfg)?;
    let mut comps = std::collections::HashMap::from([
        ("e_aa".to_string(), cc.e_aa),
        ("e_ab".to_string(), cc.e_ab),
        ("e_bb".to_string(), cc.e_bb),
    ]);
    let t_correction = if level == CcLevel::CcsdT {
        let t = ferric_cc::u_ccsd_t::u_ccsd_t(mol, &prep, &dfbs, op, &uhf, &cc, cfg)?;
        comps.extend([
            ("t_aaa".to_string(), t.aaa),
            ("t_aab".to_string(), t.aab),
            ("t_abb".to_string(), t.abb),
            ("t_bbb".to_string(), t.bbb),
        ]);
        Some(t.e_t)
    } else {
        None
    };
    let reference = reference_certificate(mol, &uhf, scf_config, basis);
    Ok(PyCcResult {
        correlation_energy: cc.correlation_energy,
        t_correction,
        ladder_route: None,
        ladder_block: None,
        reference: "uhf".into(),
        spin_components: Some(comps),
        certificate_data: ferric_cc::certificate::u_cc_certificate(
            level.kind(),
            &reference,
            &aux.name,
            cfg,
        ),
    })
}

/// The `ladder` argument is read by closed-shell CCSD only; an explicit
/// route on an open-shell molecule is refused rather than ignored.
fn refuse_open_shell_ladder(mol: &Molecule, ladder: &str) -> PyResult<()> {
    if mol.multiplicity > 1 && ladder != "auto" {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "ladder={ladder:?} is read by closed-shell CCSD only; the UHF route for an \
             open-shell molecule builds its VVVV blocks densely (use ladder=\"auto\")"
        )));
    }
    Ok(())
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
    let op = Operator::coulomb();
    let rhf_config = rhf_config_conv(
        k_builder,
        budget_bytes_from_gb(memory_budget_gb),
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    let (prep, dfbs, rhf) =
        converged_rhf_with_aux(&mol.inner, &basis_set.inner, &auxbasis.inner, &rhf_config)?;
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
        reference: "rhf".into(),
        spin_components: None,
        certificate_data,
    })
}

/// The CC settings of the closed-shell entry points; an unknown `ladder` route
/// is a ValueError before any SCF runs.
fn cc_config(
    frozen_core: Option<usize>,
    memory_budget_gb: Option<f64>,
    ladder: &str,
) -> PyResult<CcConfig> {
    Ok(CcConfig {
        frozen_core: frozen_core.unwrap_or(0),
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        ladder: ferric_cc::CcsdLadder::parse_config_str(ladder)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        ..Default::default()
    })
}

/// The Python arguments `run_ccsd` and `run_ccsd_t` share.
struct CcsdArgs<'a> {
    mol: &'a PyMolecule,
    basis_set: &'a PyBasisSet,
    auxbasis: &'a PyBasisSet,
    frozen_core: Option<usize>,
    k_builder: Option<&'a str>,
    memory_budget_gb: Option<f64>,
    ladder: &'a str,
    conv: ScfConv,
}

/// `run_ccsd` / `run_ccsd_t`: UCCSD(/T) for an open-shell molecule, else the
/// spin-adapted closed-shell CCSD and, at `CcLevel::CcsdT`, its (T).
fn run_ccsd_level(py: Python<'_>, a: CcsdArgs<'_>, level: CcLevel) -> PyResult<PyCcResult> {
    let emol = a.mol.inner.clone();
    let ebasis = a.basis_set.inner.clone();
    let eaux = a.auxbasis.inner.clone();
    let rhf_config = rhf_config_conv(
        a.k_builder,
        budget_bytes_from_gb(a.memory_budget_gb),
        a.conv,
    );
    let cfg = cc_config(a.frozen_core, a.memory_budget_gb, a.ladder)?;
    refuse_open_shell_ladder(&emol, a.ladder)?;
    if emol.multiplicity > 1 {
        return py
            .allow_threads(|| run_u_ccsd_inner(&emol, &ebasis, &eaux, &rhf_config, &cfg, level))
            .map_err(make_err);
    }
    // Release the GIL for the SCF + CCSD (+ (T)) compute. All captured values
    // are owned clones/locals built above, so nothing Python-borrowed crosses
    // the closure boundary.
    //
    // A closed-shell molecule reaches here, and `solve_rhf` yields a
    // restricted reference, so both halves take the spin-adapted path
    // unconditionally: `ccsd_closed_shell` for the amplitudes and
    // `ccsd_t_closed_shell` for the triples, over spatial orbitals (no/nv)
    // rather than spin orbitals (2no/2nv), so the O(N^6) VVVV block is 16x
    // smaller. The amplitudes flow straight through — no
    // `expand_amplitudes_to_spin_orbital` round-trip, since the spin-adapted
    // (T) consumes the spatial convention natively (and hard-rejects
    // spin-orbital amplitudes with a typed error).
    //
    // Measured on water/aug-cc-pVDZ: CCSD 24.6 s -> 1.10 s (22x), peak RSS
    // 1.82 GB -> 0.48 GB; (T) 1.87 -> 0.14 s. The two (T) implementations
    // agree to 1.9e-16 when fed identical amplitudes, and to 7.8e-10..6.9e-7
    // against PySCF's `ccsd_t()` — inside the ~5e-6 Ha RI floor.
    let (r, t_correction, reference) = py
        .allow_threads(|| -> Result<_, ferric_core::FerricError> {
            let (prep, aux, rhf) = rhf_with_aux(&emol, &ebasis, &eaux, &rhf_config)?;
            let op = Operator::coulomb();
            let r = run_ccsd_cs_inner(&emol, &prep, &aux, op, &rhf, &cfg)?;
            let e_t = match level {
                CcLevel::Ccsd => None,
                CcLevel::CcsdT => {
                    Some(run_ccsd_t_cs_inner(&emol, &prep, &aux, op, &rhf, &r, &cfg)?)
                }
            };
            let reference = reference_certificate(&emol, &rhf, &rhf_config, &ebasis);
            Ok((r, e_t, reference))
        })
        .map_err(make_err)?;
    let (ladder_route, ladder_block) = py_ladder_route(&r);
    let certificate =
        ferric_cc::certificate::cc_certificate(level.kind(), &reference, &eaux.name, &cfg, &r);
    let certificate_data = match level {
        CcLevel::Ccsd => ferric_mp2::certificate::with_anchor_envelope(
            certificate,
            "CCSD",
            &emol,
            &ebasis,
            Some(&eaux),
        ),
        // `run_ccsd_t` has never carried the anchor envelope.
        CcLevel::CcsdT => certificate,
    };
    Ok(PyCcResult {
        correlation_energy: r.correlation_energy,
        t_correction,
        ladder_route,
        ladder_block,
        reference: "rhf".into(),
        spin_components: None,
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
    let conv = ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv);
    let args = CcsdArgs {
        mol,
        basis_set,
        auxbasis,
        frozen_core,
        k_builder,
        memory_budget_gb,
        ladder,
        conv,
    };
    run_ccsd_level(py, args, CcLevel::Ccsd)
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
    let conv = ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv);
    let args = CcsdArgs {
        mol,
        basis_set,
        auxbasis,
        frozen_core,
        k_builder,
        memory_budget_gb,
        ladder,
        conv,
    };
    run_ccsd_level(py, args, CcLevel::CcsdT)
}

#[pyclass]
#[pyo3(name = "EomCcsdResult")]
pub(crate) struct PyEomCcsdResult {
    #[pyo3(get)]
    pub(crate) rhf_energy: f64,
    #[pyo3(get)]
    pub(crate) ccsd_correlation_energy: f64,
    #[pyo3(get)]
    pub(crate) iterations: usize,
    #[pyo3(get)]
    pub(crate) products: usize,
    #[pyo3(get)]
    pub(crate) ladder_route: String,
    pub(crate) inner: ferric_cc::eom_ccsd::EomCcsdResult,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PyEomCcsdResult {
    /// Validation certificate (see `RhfResult.certificate`), capability
    /// `"eom-ccsd"`.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    /// Singlet excitation energies ω (Hartree), ascending.
    #[getter]
    fn excitation_energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.excitation_energies)
    }
    /// Davidson residual norm of each root.
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.residuals)
    }
    /// Singles weight `2|r1|^2 / |R|^2` of each root (spin-orbital norm).
    #[getter]
    fn singles_weight<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.singles_weight)
    }
    /// Singles part `r1[i, a]` of root `k`'s right eigenvector.
    fn r1<'py>(&self, py: Python<'py>, k: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let r = self.inner.r1.get(k).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!("root {k} out of range"))
        })?;
        Ok(PyArray2::from_array(py, r))
    }
    /// Doubles part `r2[i, j, a, b]` (alpha-beta amplitude) of root `k`.
    fn r2<'py>(&self, py: Python<'py>, k: usize) -> PyResult<Bound<'py, numpy::PyArray4<f64>>> {
        let r = self.inner.r2.get(k).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!("root {k} out of range"))
        })?;
        Ok(numpy::PyArray4::from_array(py, r))
    }
    fn __repr__(&self) -> String {
        let lowest = self
            .inner
            .excitation_energies
            .first()
            .map_or(0.0, |w| w * 27.211_386_245_988);
        format!(
            "EomCcsdResult(n_roots={}, lowest={lowest:.4} eV)",
            self.inner.excitation_energies.len()
        )
    }
}

/// EE-EOM-CCSD singlet excitation energies on an RHF reference: SCF, then
/// spin-adapted CCSD (amplitudes converged to |dE| < 1e-10), then the
/// non-symmetric Davidson. `scf_*` control the reference SCF as in
/// `run_rhf`; `n_roots`, `conv_tol`, `max_iter`, `max_space` the Davidson.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, n_roots=5, conv_tol=1e-6, max_iter=100, max_space=None, frozen_core=None, k_builder=None, memory_budget_gb=None, ladder="auto", scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_eom_ccsd(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    n_roots: usize,
    conv_tol: f64,
    max_iter: usize,
    max_space: Option<usize>,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    ladder: &str,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyEomCcsdResult> {
    if n_roots == 0 || max_iter == 0 || !(conv_tol.is_finite() && conv_tol > 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "run_eom_ccsd: n_roots and max_iter must be at least 1 and conv_tol positive",
        ));
    }
    let emol = mol.inner.clone();
    let ebasis = basis_set.inner.clone();
    let eaux = auxbasis.inner.clone();
    let budget = budget_bytes_from_gb(memory_budget_gb);
    let rhf_config = rhf_config_conv(
        k_builder,
        budget,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    let cfg = CcConfig {
        energy_conv: ferric_cc::eom_ccsd::EOM_CCSD_ENERGY_CONV,
        ..cc_config(frozen_core, memory_budget_gb, ladder)?
    };
    let eom_cfg = ferric_cc::eom_ccsd::EomCcsdConfig {
        n_roots,
        conv_tol,
        max_iter,
        max_space,
        memory_budget_bytes: budget,
    };
    let (rhf_energy, cc, eom, reference) = py
        .allow_threads(|| -> Result<_, ferric_core::FerricError> {
            let (prep, dfbs, rhf) = rhf_with_aux(&emol, &ebasis, &eaux, &rhf_config)?;
            let op = Operator::coulomb();
            let cc = run_ccsd_cs_inner(&emol, &prep, &dfbs, op, &rhf, &cfg)?;
            let eom = ferric_cc::eom_ccsd::eom_ccsd_singlets(
                &emol,
                &prep,
                &dfbs,
                op,
                &rhf,
                &cc,
                cfg.frozen_core,
                &eom_cfg,
            )?;
            let reference = reference_certificate(&emol, &rhf, &rhf_config, &ebasis);
            Ok((rhf.energy, cc, eom, reference))
        })
        .map_err(make_err)?;
    let ladder_route = match eom.ladder_route {
        ferric_cc::ccsd_closed_shell::LadderRoute::Dense => "dense".to_string(),
        ferric_cc::ccsd_closed_shell::LadderRoute::Ri { .. } => "ri".to_string(),
    };
    let certificate_data =
        ferric_cc::eom_ccsd::eom_ccsd_certificate(&reference, &eaux.name, cfg.frozen_core, &eom);
    Ok(PyEomCcsdResult {
        rhf_energy,
        ccsd_correlation_energy: cc.correlation_energy,
        iterations: eom.iterations,
        products: eom.products,
        ladder_route,
        inner: eom,
        certificate_data,
    })
}

/// Register the closed-shell CC entry points (`run_ccd`, `run_ccsd`,
/// `run_ccsd_t`, `run_eom_ccsd`) and their result classes.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCcResult>()?;
    m.add_class::<PyEomCcsdResult>()?;
    m.add_function(wrap_pyfunction!(run_ccd, m)?)?;
    m.add_function(wrap_pyfunction!(run_ccsd, m)?)?;
    m.add_function(wrap_pyfunction!(run_ccsd_t, m)?)?;
    m.add_function(wrap_pyfunction!(run_eom_ccsd, m)?)
}
