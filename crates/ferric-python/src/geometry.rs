//! Geometry optimisation bindings: moved verbatim from `lib.rs`.

use super::*;

// ── Optimize ──

#[pyclass]
#[pyo3(name = "OptimizeResult")]
pub(crate) struct PyOptimizeResult {
    #[pyo3(get)]
    pub(crate) energy: f64,
    #[pyo3(get)]
    pub(crate) converged: bool,
    #[pyo3(get)]
    pub(crate) steps: usize,
    /// Energy at every point the optimizer EVALUATED, in order.
    ///
    /// `energy`/`steps`/`converged` cannot tell "ran out of steps near a
    /// minimum" from "walked uphill and oscillated". Feed this to
    /// `tools.viz.energy_plots.optimization_trace`, which flags a climb.
    #[pyo3(get)]
    pub(crate) energy_trace: Vec<f64>,
    pub(crate) mol_data: Molecule,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PyOptimizeResult {
    /// Validation certificate of this result (dict form of
    /// `ferric_core::certificate::Certificate`; see `RhfResult.certificate`).
    /// `capability` is `"optimize:<kind>"`; the final-point SCF (re-solved at
    /// the returned geometry) is the reference whose grade, approximations and
    /// warnings it inherits, `gradient` and `optimizer` are recorded as
    /// approximations, and a run with `converged = False` is `not graded`.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }

    /// The optimized geometry as a new `Molecule`.
    fn mol(&self) -> PyMolecule {
        PyMolecule {
            inner: self.mol_data.clone(),
        }
    }
    fn __repr__(&self) -> String {
        format!(
            "OptimizeResult(energy={:.10}, converged={}, steps={})",
            self.energy, self.converged, self.steps
        )
    }
    fn __str__(&self) -> String {
        format!(
            "Optimized Energy: {:.10} Ha (converged: {}, {} steps)",
            self.energy, self.converged, self.steps
        )
    }
}

/// Certificate of a finished `run_optimize*`: the SCF at the returned
/// geometry is the reference; a failure of that SCF gives a `not graded`
/// certificate carrying the reason.
pub(crate) fn optimize_result_certificate(
    r: &ferric_scf::optimize::OptimizeResult,
    opt_config: &OptimizeConfig,
    basis_name: &str,
    config: &RhfConfig,
    spin: ferric_scf::Spin,
) -> ferric_core::certificate::Certificate {
    let (kind, reference) = task_reference_certificate(&r.mol, basis_name, config, spin);
    match reference {
        Ok(reference) => ferric_scf::certificate::optimize_certificate(
            kind,
            &reference,
            r.converged,
            r.steps,
            r.energy,
            opt_config,
            "analytic",
        ),
        Err(why) => {
            ferric_core::certificate::Certificate::ungraded(format!("optimize:{kind}")).warn(why)
        }
    }
}

#[pyfunction]
#[pyo3(signature = (mol, basis_name, max_steps=None, e_conv=None, point_charges=None, external_field=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, solvent=None, pcm_lebedev_order=None, scf_warm_start=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_optimize(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_name: &str,
    max_steps: Option<usize>,
    e_conv: Option<f64>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    solvent: Option<&Bound<'_, PyAny>>,
    pcm_lebedev_order: Option<usize>,
    scf_warm_start: Option<bool>,
) -> PyResult<PyOptimizeResult> {
    let ctx = ParallelContext::default();
    let mut rhf_config = RhfConfig {
        external_potential: build_external_potential(point_charges, external_field),
        pcm: build_pcm_config(solvent, pcm_lebedev_order)?,
        ..Default::default()
    };
    ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv).apply(&mut rhf_config);
    // Exact J/K here (no `xc`, no aux kwargs), so RI-J never runs; marked
    // anyway so the certificate's reference SCF matches the driver's.
    geometry_task_scf(&mut rhf_config, None)?;
    // Apply the ECP before optimizing: each SCF derives its occupations from
    // `Molecule::nelec()`, which counts ALL electrons until `apply_ecp` has
    // run. Without this an ECP basis (aug-cc-pvdz-pp, Z >= 37) would optimize
    // with the all-electron count against ECP integrals.
    let bs = basis::bundled(basis_name).map_err(make_err)?;
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&bs);
    let opt_config = OptimizeConfig {
        max_steps: max_steps.unwrap_or(100),
        e_conv: e_conv.unwrap_or(1e-6),
        scf_warm_start: scf_warm_start.unwrap_or(true),
        ..Default::default()
    };
    let r = py
        .allow_threads(|| {
            optimize_geometry(
                &ctx,
                &emol,
                basis_name,
                Operator::coulomb(),
                &rhf_config,
                &opt_config,
            )
        })
        .map_err(make_err)?;
    let certificate_data = optimize_result_certificate(
        &r,
        &opt_config,
        basis_name,
        &rhf_config,
        ferric_scf::Spin::Restricted,
    );
    Ok(PyOptimizeResult {
        energy: r.energy,
        converged: r.converged,
        steps: r.steps,
        energy_trace: r.energy_trace,
        mol_data: r.mol,
        certificate_data,
    })
}

/// Optimize an OPEN-SHELL geometry with a UHF reference.
///
/// The spin state comes from the `Molecule` (its multiplicity), exactly as in
/// `run_uhf` -- there is no `multiplicity` kwarg here, and passing a
/// closed-shell molecule simply runs UHF on a singlet.
///
/// `run_optimize` uses an RHF reference and cannot relax a radical: the
/// doublet would be forced into a closed-shell density. Use this for any
/// molecule with unpaired electrons.
#[pyfunction]
#[pyo3(signature = (mol, basis_name, max_steps=None, e_conv=None, point_charges=None, external_field=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, solvent=None, pcm_lebedev_order=None, scf_warm_start=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_optimize_uhf(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_name: &str,
    max_steps: Option<usize>,
    e_conv: Option<f64>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    solvent: Option<&Bound<'_, PyAny>>,
    pcm_lebedev_order: Option<usize>,
    scf_warm_start: Option<bool>,
) -> PyResult<PyOptimizeResult> {
    let ctx = ParallelContext::default();
    let mut uhf_config = RhfConfig {
        external_potential: build_external_potential(point_charges, external_field),
        pcm: build_pcm_config(solvent, pcm_lebedev_order)?,
        ..Default::default()
    };
    ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv).apply(&mut uhf_config);
    // Exact J/K here (no `xc`, no aux kwargs), so RI-J never runs; marked
    // anyway so the certificate's reference SCF matches the driver's.
    geometry_task_scf(&mut uhf_config, None)?;
    // Apply the ECP before optimizing: each SCF derives its occupations from
    // `Molecule::nelec()`, which counts ALL electrons until `apply_ecp` has
    // run. Without this an ECP basis (aug-cc-pvdz-pp, Z >= 37) would optimize
    // with the all-electron count against ECP integrals.
    let bs = basis::bundled(basis_name).map_err(make_err)?;
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&bs);
    let opt_config = OptimizeConfig {
        max_steps: max_steps.unwrap_or(100),
        e_conv: e_conv.unwrap_or(1e-6),
        scf_warm_start: scf_warm_start.unwrap_or(true),
        ..Default::default()
    };
    let r = py
        .allow_threads(|| {
            optimize_geometry_uhf(
                &ctx,
                &emol,
                basis_name,
                Operator::coulomb(),
                &uhf_config,
                &opt_config,
            )
        })
        .map_err(make_err)?;
    let certificate_data = optimize_result_certificate(
        &r,
        &opt_config,
        basis_name,
        &uhf_config,
        ferric_scf::Spin::Unrestricted,
    );
    Ok(PyOptimizeResult {
        energy: r.energy,
        converged: r.converged,
        steps: r.steps,
        energy_trace: r.energy_trace,
        mol_data: r.mol,
        certificate_data,
    })
}

#[pyclass]
#[pyo3(name = "FrequencyResult")]
pub(crate) struct PyFrequencyResult {
    /// Vibrational wavenumbers in cm^-1, ascending. `3N-6` entries (`3N-5` if
    /// linear). A NEGATIVE entry is an imaginary frequency, by the usual
    /// convention -- it means a mode with a negative force constant, i.e. the
    /// geometry is not a minimum.
    #[pyo3(get)]
    pub(crate) frequencies: Vec<f64>,
    /// The projected-out translation/rotation modes, cm^-1. Should be ~0;
    /// retained as a diagnostic.
    #[pyo3(get)]
    pub(crate) trans_rot_frequencies: Vec<f64>,
    #[pyo3(get)]
    pub(crate) is_linear: bool,
    /// Largest |H_ij - H_ji| in the raw Cartesian Hessian, Hartree/Bohr^2.
    /// Zero in exact arithmetic, so this is a direct read on whether `delta`
    /// and the SCF thresholds suit the system. A large value invalidates the
    /// frequencies.
    #[pyo3(get)]
    pub(crate) asymmetry: f64,
    #[pyo3(get)]
    pub(crate) n_gradient_evaluations: usize,
    /// `"analytic"`, `"analytic-moving-grid"` or `"finite-difference"`: which
    /// construction produced the Hessian.
    #[pyo3(get)]
    pub(crate) hessian_source: String,
    /// The Cartesian Hessian before mass-weighting, Hartree/Bohr^2,
    /// `3N x 3N`, rows and columns ordered `[x0,y0,z0,x1,...]` like
    /// `normal_modes`.
    #[pyo3(get)]
    pub(crate) cartesian_hessian: Vec<Vec<f64>>,
    /// Electronic energy at the undisplaced geometry. With `dispersion=` it is
    /// the CORRECTED total, KS + dispersion.
    #[pyo3(get)]
    pub(crate) energy: f64,
    /// The dispersion energy at the undisplaced geometry (Hartree), already
    /// included in `energy`. `None` when `dispersion=` was not given (not
    /// evaluated), never 0.0.
    #[pyo3(get)]
    pub(crate) e_dispersion: Option<f64>,
    /// Normal-mode displacement vectors in CARTESIAN coordinates: one row per
    /// entry of `frequencies`, `3N` values each, ordered `[x0,y0,z0,x1,...]`
    /// to match `Molecule.symbols()`. These are the mass-weighted eigenvectors
    /// divided back through by `sqrt(m)`, so light atoms visibly move further
    /// than heavy ones; they are not normalized to any convention.
    ///
    /// Needed because `n_imaginary() == 1` is NECESSARY BUT NOT SUFFICIENT for
    /// a transition state -- a methyl rotor gives one imaginary frequency too.
    /// Confirming a saddle means checking the imaginary mode displaces the
    /// atoms along the reaction coordinate, which requires the vector.
    #[pyo3(get)]
    pub(crate) normal_modes: Vec<Vec<f64>>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PyFrequencyResult {
    /// Validation certificate of this result (dict form of
    /// `ferric_core::certificate::Certificate`; see `RhfResult.certificate`).
    /// `capability` is `"frequencies:<kind>"`; the SCF at the input geometry
    /// is the reference whose grade, approximations and warnings it inherits,
    /// the Hessian construction is recorded as the `hessian` approximation,
    /// and each imaginary frequency count is a warning.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }

    fn __repr__(&self) -> String {
        format!(
            "FrequencyResult(energy={:.10}, n_modes={}, is_linear={})",
            self.energy,
            self.frequencies.len(),
            self.is_linear,
        )
    }
    fn __str__(&self) -> String {
        let n_imag = self.frequencies.iter().filter(|&&f| f < 0.0).count();
        format!(
            "Vibrational Frequencies: {} modes, {} imaginary, energy={:.10} Ha, asymmetry={:.2e}",
            self.frequencies.len(),
            n_imag,
            self.energy,
            self.asymmetry,
        )
    }
}

/// Harmonic vibrational frequencies, mass-weighted with translations and
/// rotations projected out.
///
/// `df_j_aux` / `df_k_aux` (default: exact J/K) density-fit the Coulomb and
/// exchange of every SCF the run does, exactly as in `run_rhf`; the analytic
/// RHF, UHF and ROHF Hessians then differentiate the fitted energy (RI-J,
/// RI-K or both; ferric_scf::hessian_df / hessian_df_k), and the
/// finite-difference path differences the fitted analytic gradient. RI with
/// `point_charges` / `external_field` takes the finite-difference path under
/// "auto" (`"analytic"` raises).
///
/// `hessian`: "auto" (default) uses the analytic Hessian for RHF (closed
/// shell) or UHF (any multiplicity) with exact J/K, with or without an ECP,
/// or ROHF with exact J/K and no external potential (an ECP is analytic for all three, also with RI-J / RI-K),
/// and a basis up to f functions and fixed external point charges or a uniform field (QM/MM, RHF and UHF only,
/// QM-QM block; this binding takes `point_charges=` / `external_field=` only,
/// the Gaussian-smeared charges the Hessian also covers are reachable from the
/// CLI and Rust API) (one SCF plus CPHF); for closed-shell Kohn-Sham (`xc=`
/// with `reference="rhf"`) the "analytic-moving-grid" Hessian below wherever
/// it applies; and a finite difference of the ANALYTIC gradient (6N gradient
/// evaluations) otherwise (UKS, ROKS, meta-GGA, VV10, range-separated
/// hybrids, ECPs or external potentials with `xc=`, `dispersion=`, and large
/// systems whose response AO table -- basis functions x grid points x 4 for
/// a GGA, 8 bytes each -- exceeds 2 GiB: on the default grid a GGA above
/// about 8,100 basis functions x atoms). The choice is made before the SCF;
/// an error an analytic path raises after it (a non-converged SCF; for ROHF a
/// Fock-consistency failure from a loose convergence threshold; a
/// non-converging response, non-positive curvature) is raised, not retried
/// with finite differences. "analytic" raises if the Hartree-Fock analytic Hessian does not
/// apply (it never covers KS); "fd" always differences gradients.
/// `.hessian_source` says which ran.
/// "analytic-moving-grid" is the analytic
/// closed-shell Kohn-Sham Hessian on the production (moving) grid, Becke
/// weight response included: the Hessian of the surface the KS gradient
/// differentiates, so it agrees with "fd" to the finite-difference error.
/// It needs `xc=` (LDA/GGA, pure or global hybrid) and `reference="rhf"`, and
/// raises before the SCF for Hartree-Fock, "uhf"/"rohf", meta-GGA, VV10,
/// range-separated hybrids, ECPs, `point_charges=`/`external_field=` and
/// `dispersion=`.
///
/// `reference` selects the SCF: "rhf" (default), "uhf", or "rohf". Setting
/// `xc` promotes it to the matching KS variant (RKS/UKS/ROKS).
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: SCF controls of every
/// SCF the run does (defaults 200, 1e-3, 1e-6, as in `run_rhf`'s `max_iter`,
/// `energy_conv`, `density_conv`; those bare names are deprecated aliases). A finite-difference
/// Hessian differentiates SCF gradients, so its noise floor is the SCF's.
///
/// `delta` is the central-difference displacement in Bohr (default 5e-3). It is
/// a real accuracy knob that degrades silently -- too large adds truncation
/// error, too small amplifies SCF noise. Check `.asymmetry` rather than
/// assuming.
///
/// `scf_warm_start` (default False, opt-in): a finite-difference Hessian
/// starts every displaced SCF from the undisplaced geometry's converged
/// orbitals (anchored at the reference, never chained from the previous
/// displacement; `[frequencies] scf_warm_start`). At equal `scf_density_conv`
/// the warm Hessian is noisier (warm SCFs stop nearer the threshold; propane
/// def2-SVP: 9.4 cm^-1 on a soft mode at 1e-6 against 0.03 cold), so pair it
/// with a ~100x tighter `scf_density_conv`. No effect on an analytic Hessian.
///
/// `dispersion`: the same strict spellings as `run_dft` ("d3bj", "d3(bj)",
/// "d3bj(<functional>)", "mbd", "mbd(<functional>)"). Requires `xc`; any
/// reference ("rhf", "uhf", "rohf", i.e. RKS, UKS or ROKS). The Hessian is
/// then the central difference of the corrected analytic gradient
/// (KS + dispersion), each from the SCF converged at that displaced geometry,
/// so `hessian="analytic"` and "analytic-moving-grid" raise and "auto" runs
/// finite differences. MBD@rsSCS
/// enters through its exact gradient, including the orbital relaxation of its
/// Hirshfeld volumes (the reference's own Z-vector: RKS, UKS or ROKS). `.energy` is then the corrected total and
/// `.e_dispersion` the dispersion part.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_name, reference=None, xc=None, delta=None, multiplicity=None,
    point_charges=None, external_field=None, hessian="auto", dispersion=None,
    max_iter=None, energy_conv=None, density_conv=None,
    df_j_aux=None, df_k_aux=None, scf_ri_k_precision=None,
    scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None,
    scf_warm_start=None, scf_ri_j_precision=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_frequencies(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_name: &str,
    reference: Option<&str>,
    xc: Option<&str>,
    delta: Option<f64>,
    multiplicity: Option<u32>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    hessian: &str,
    dispersion: Option<&str>,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    scf_ri_k_precision: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    scf_warm_start: Option<bool>,
    scf_ri_j_precision: Option<&str>,
) -> PyResult<PyFrequencyResult> {
    use ferric_scf::frequencies::{FrequencyConfig, FrequencyReference, HessianMethod};
    let max_iter = alias_kwarg(
        py,
        "run_frequencies",
        "max_iter",
        "scf_max_iter",
        max_iter,
        scf_max_iter,
    )?;
    let energy_conv = alias_kwarg(
        py,
        "run_frequencies",
        "energy_conv",
        "scf_energy_conv",
        energy_conv,
        scf_energy_conv,
    )?;
    let density_conv = alias_kwarg(
        py,
        "run_frequencies",
        "density_conv",
        "scf_density_conv",
        density_conv,
        scf_density_conv,
    )?;

    // Strict, like the CLI: an unrecognized reference must ERROR rather than
    // silently running RHF and handing back frequencies for the wrong system.
    let refr = match reference {
        None | Some("rhf") => FrequencyReference::Rhf,
        Some("uhf") => FrequencyReference::Uhf,
        Some("rohf") => FrequencyReference::Rohf,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown reference \"{other}\"; expected \"rhf\", \"uhf\" or \"rohf\""
            )))
        }
    };
    if let Some(d) = delta {
        if !(d.is_finite() && d > 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "delta must be finite and > 0 (got {d})"
            )));
        }
    }

    let mut m = mol.inner.clone();
    if let Some(mult) = multiplicity {
        m.multiplicity = mult as usize;
    }
    // The MM field. `harmonic_frequencies` already threads
    // `config.external_potential` into the same gradient calls, so an embedded
    // Hessian needs no new machinery -- only a way to ask for one. Without
    // this, CONFIRMING an embedded saddle (exactly one imaginary frequency)
    // had to be done in vacuum, against a Hessian from a different surface.
    let mut scf_cfg = RhfConfig {
        xc: xc.map(|s| s.to_string()),
        external_potential: build_external_potential(point_charges, external_field),
        ..Default::default()
    };
    // Unset keeps the RhfConfig defaults (200, 1e-3, 1e-6), as before.
    scf_cfg.max_iter = max_iter.unwrap_or(scf_cfg.max_iter);
    scf_cfg.energy_conv = energy_conv.unwrap_or(scf_cfg.energy_conv);
    scf_cfg.density_conv = density_conv.unwrap_or(scf_cfg.density_conv);
    // Unset = exact J/K (unchanged); same opt-out spellings as run_rhf.
    scf_cfg.df_j_aux = df_j_aux.map(ferric_scf::rhf::normalize_df_aux);
    scf_cfg.df_k_aux = df_k_aux.map(ferric_scf::rhf::normalize_df_aux);
    geometry_task_scf(&mut scf_cfg, scf_ri_j_precision)?;
    scf_cfg.ri_k_precision = parse_ri_k_precision(scf_ri_k_precision)?;
    let mut fcfg = FrequencyConfig {
        reference: refr,
        hessian: HessianMethod::parse_config_str(hessian)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        ..Default::default()
    };
    if let Some(d) = delta {
        fcfg.delta = d;
    }
    fcfg.scf_warm_start = scf_warm_start.unwrap_or(false);

    let (r, e_dispersion) =
        frequencies_with_dispersion(&m, basis_name, &scf_cfg, &fcfg, dispersion, xc)?;
    let spin = match refr {
        FrequencyReference::Rhf => ferric_scf::Spin::Restricted,
        FrequencyReference::Uhf => ferric_scf::Spin::Unrestricted,
        FrequencyReference::Rohf => ferric_scf::Spin::RestrictedOpen,
    };
    let mut m_ecp = m.clone();
    m_ecp.apply_ecp(&basis::bundled(basis_name).map_err(make_err)?);
    let (kind, reference) = task_reference_certificate(&m_ecp, basis_name, &scf_cfg, spin);
    let certificate_data = match reference {
        Ok(reference) => {
            let c = ferric_scf::certificate::frequencies_certificate(kind, &reference, &r, &fcfg);
            match (dispersion, e_dispersion) {
                (Some(spec), Some(_)) => c.with_approximation(
                    "dispersion",
                    format!("{spec} in the Hessian (central differences of the KS + dispersion gradient)"),
                ),
                _ => c,
            }
        }
        Err(why) => {
            ferric_core::certificate::Certificate::ungraded(format!("frequencies:{kind}")).warn(why)
        }
    };
    Ok(PyFrequencyResult {
        frequencies: r.frequencies,
        trans_rot_frequencies: r.trans_rot_frequencies,
        is_linear: r.is_linear,
        asymmetry: r.asymmetry,
        n_gradient_evaluations: r.n_gradient_evaluations,
        hessian_source: r.hessian_source.label().to_string(),
        cartesian_hessian: r
            .cartesian_hessian
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        energy: r.energy,
        e_dispersion,
        // Array2 -> Vec<Vec<f64>>, one row per mode. `.rows()` preserves the
        // (mode, 3N) layout the Rust field documents; collecting from the flat
        // slice would silently transpose whenever the array is not square.
        normal_modes: r
            .normal_modes
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        certificate_data,
    })
}

/// Parse the xTB knobs shared by `run_xtb_*`. Method names go through
/// `XtbMethod::parse_config_str` (strict: unknown names error).
#[cfg(feature = "xtb")]
pub(crate) fn xtb_config(
    method: &str,
    accuracy: Option<f64>,
    scc_max_iter: Option<u32>,
    electronic_temp: Option<f64>,
) -> PyResult<ferric_xtb::XtbConfig> {
    let mut cfg = ferric_xtb::XtbConfig {
        method: ferric_xtb::XtbMethod::parse_config_str(method)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        ..Default::default()
    };
    if let Some(a) = accuracy {
        cfg.accuracy = a;
    }
    if let Some(n) = scc_max_iter {
        cfg.scc_max_iter = n;
    }
    if let Some(t) = electronic_temp {
        cfg.electronic_temp = t;
    }
    Ok(cfg)
}

#[cfg(not(feature = "xtb"))]
pub(crate) fn xtb_unavailable() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(
        "this ferric build was compiled without the `xtb` feature (libxtb); \
         rebuild with `cargo build --release -p ferric-python --features xtb`",
    )
}

/// GFN-xTB energy (Hartree) and gradient (Hartree/Bohr, one `[x, y, z]` row
/// per atom) at the molecule's geometry. `method`: "gfn1", "gfn2" (default) or
/// "gfnff". Charge and multiplicity come from the `Molecule`.
///
/// `accuracy` (xtb scale, 1.0 default), `scc_max_iter` (SCC iterations,
/// default 250; ignored by GFN-FF) and `electronic_temp` (K, default 300)
/// are xtb's own controls. An SCC that does not converge raises.
#[pyfunction]
#[pyo3(signature = (mol, method="gfn2", accuracy=None, scc_max_iter=None, electronic_temp=None))]
#[cfg_attr(not(feature = "xtb"), allow(unused_variables))]
pub(crate) fn run_xtb_energy_gradient(
    py: Python<'_>,
    mol: &PyMolecule,
    method: &str,
    accuracy: Option<f64>,
    scc_max_iter: Option<u32>,
    electronic_temp: Option<f64>,
) -> PyResult<(f64, Vec<Vec<f64>>)> {
    #[cfg(feature = "xtb")]
    {
        let cfg = xtb_config(method, accuracy, scc_max_iter, electronic_temp)?;
        let m = mol.inner.clone();
        let (e, g) = py
            .allow_threads(|| ferric_xtb::xtb_energy_gradient(&m, cfg))
            .map_err(make_err)?;
        Ok((e, g.rows().into_iter().map(|r| r.to_vec()).collect()))
    }
    #[cfg(not(feature = "xtb"))]
    Err(xtb_unavailable())
}

/// Geometry optimization on a GFN-xTB surface, using the same BFGS driver as
/// `run_optimize` (`max_steps` geometry steps, `e_conv` energy criterion;
/// `g_max_thresh` / `g_rms_thresh` gradient criteria in Hartree/Bohr).
/// The SCC at every geometry is controlled by `scc_max_iter`; an unconverged
/// SCC raises instead of continuing the search. Returns an `OptimizeResult`
/// whose certificate is `not graded` (no validation grade exists for xTB).
#[pyfunction]
#[pyo3(signature = (
    mol, method="gfn2", max_steps=None, e_conv=None, g_max_thresh=None, g_rms_thresh=None,
    accuracy=None, scc_max_iter=None, electronic_temp=None,
))]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "xtb"), allow(unused_variables))]
pub(crate) fn run_xtb_optimize(
    py: Python<'_>,
    mol: &PyMolecule,
    method: &str,
    max_steps: Option<usize>,
    e_conv: Option<f64>,
    g_max_thresh: Option<f64>,
    g_rms_thresh: Option<f64>,
    accuracy: Option<f64>,
    scc_max_iter: Option<u32>,
    electronic_temp: Option<f64>,
) -> PyResult<PyOptimizeResult> {
    #[cfg(feature = "xtb")]
    {
        let cfg = xtb_config(method, accuracy, scc_max_iter, electronic_temp)?;
        let d = OptimizeConfig::default();
        let opt_config = OptimizeConfig {
            max_steps: max_steps.unwrap_or(d.max_steps),
            e_conv: e_conv.unwrap_or(d.e_conv),
            g_max_thresh: g_max_thresh.unwrap_or(d.g_max_thresh),
            g_rms_thresh: g_rms_thresh.unwrap_or(d.g_rms_thresh),
            ..d
        };
        let m = mol.inner.clone();
        let r = py
            .allow_threads(|| ferric_xtb::optimize_geometry_xtb(&m, cfg, &opt_config))
            .map_err(make_err)?;
        let certificate_data = ferric_core::certificate::Certificate::ungraded("optimize:xtb")
            .warn("GFN-xTB is a semiempirical method with no validation grade in ferric");
        Ok(PyOptimizeResult {
            energy: r.energy,
            converged: r.converged,
            steps: r.steps,
            energy_trace: r.energy_trace,
            mol_data: r.mol,
            certificate_data,
        })
    }
    #[cfg(not(feature = "xtb"))]
    Err(xtb_unavailable())
}

/// Harmonic frequencies on a GFN-xTB surface: central differences (`delta`,
/// Bohr, default 5e-3) of the analytic xTB gradient (`6N` evaluations), then
/// the mass-weighting and translation/rotation projection `run_frequencies`
/// uses. Input should be an xTB minimum (e.g. from `run_xtb_optimize`);
/// check `.asymmetry`. An unconverged SCC raises.
#[pyfunction]
#[pyo3(signature = (mol, method="gfn2", delta=None, accuracy=None, scc_max_iter=None, electronic_temp=None))]
#[cfg_attr(not(feature = "xtb"), allow(unused_variables))]
pub(crate) fn run_xtb_frequencies(
    py: Python<'_>,
    mol: &PyMolecule,
    method: &str,
    delta: Option<f64>,
    accuracy: Option<f64>,
    scc_max_iter: Option<u32>,
    electronic_temp: Option<f64>,
) -> PyResult<PyFrequencyResult> {
    #[cfg(feature = "xtb")]
    {
        let cfg = xtb_config(method, accuracy, scc_max_iter, electronic_temp)?;
        let delta = delta.unwrap_or(ferric_xtb::DEFAULT_HESSIAN_DELTA);
        if !(delta.is_finite() && delta > 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "delta must be finite and > 0 (got {delta})"
            )));
        }
        let m = mol.inner.clone();
        let r = py
            .allow_threads(|| ferric_xtb::harmonic_frequencies_xtb(&m, cfg, delta))
            .map_err(make_err)?;
        let certificate_data = ferric_core::certificate::Certificate::ungraded("frequencies:xtb")
            .warn("GFN-xTB is a semiempirical method with no validation grade in ferric");
        Ok(PyFrequencyResult {
            frequencies: r.frequencies,
            trans_rot_frequencies: r.trans_rot_frequencies,
            is_linear: r.is_linear,
            asymmetry: r.asymmetry,
            n_gradient_evaluations: r.n_gradient_evaluations,
            hessian_source: r.hessian_source.label().to_string(),
            cartesian_hessian: r
                .cartesian_hessian
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
            energy: r.energy,
            e_dispersion: None,
            normal_modes: r
                .normal_modes
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
            certificate_data,
        })
    }
    #[cfg(not(feature = "xtb"))]
    Err(xtb_unavailable())
}

/// `run_frequencies(dispersion=...)`, resolved BEFORE any SCF with
/// `run_dft`'s strict parser. The parameters are fitted per functional, so
/// there is no dispersion without `xc`; MBD@rsSCS needs its exact (Z-vector)
/// gradient for the chosen reference.
pub(crate) fn resolve_frequency_dispersion(
    dispersion: Option<&str>,
    xc: Option<&str>,
    reference: ferric_scf::frequencies::FrequencyReference,
    scf_cfg: &RhfConfig,
) -> PyResult<Option<DispersionSpec>> {
    let Some(spec) = dispersion else {
        return Ok(None);
    };
    let xc_name = xc.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "dispersion= requires xc=: D3(BJ) and MBD@rsSCS parameters are fitted per \
             functional, so there is no dispersion correction for Hartree-Fock",
        )
    })?;
    let spec = resolve_dispersion(spec, xc_name)?;
    if matches!(spec, DispersionSpec::Mbd(_)) {
        use ferric_scf::frequencies::FrequencyReference;
        let unsupported = match reference {
            FrequencyReference::Rhf => ferric_scf::zvector_ks::unsupported_reason(scf_cfg),
            FrequencyReference::Uhf => {
                ferric_scf::zvector_ks::unsupported_reason_unrestricted(scf_cfg)
            }
            FrequencyReference::Rohf => ferric_scf::zvector_ks::unsupported_reason_roks(scf_cfg),
        };
        if let Some(why) = unsupported {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dispersion=\"mbd\": the exact MBD@rsSCS gradient is not available: {why}"
            )));
        }
    }
    Ok(Some(spec))
}

/// Harmonic frequencies, plain or (with `dispersion`) on the KS + dispersion
/// surface, with the dispersion energy at the input geometry (`None` without
/// `dispersion`). See [`resolve_frequency_dispersion`] for what is refused.
pub(crate) fn frequencies_with_dispersion(
    mol: &ferric_core::mol::Molecule,
    basis_name: &str,
    scf_cfg: &RhfConfig,
    fcfg: &ferric_scf::frequencies::FrequencyConfig,
    dispersion: Option<&str>,
    xc: Option<&str>,
) -> PyResult<(ferric_scf::frequencies::FrequencyResult, Option<f64>)> {
    let spec = resolve_frequency_dispersion(dispersion, xc, fcfg.reference, scf_cfg)?;
    let ctx = ParallelContext::default();
    let r = match &spec {
        None => ferric_scf::frequencies::harmonic_frequencies(
            &ctx,
            mol,
            basis_name,
            Operator::coulomb(),
            scf_cfg,
            fcfg,
        ),
        Some(spec) => dispersion_frequencies(&ctx, mol, basis_name, scf_cfg, fcfg, spec),
    }
    .map_err(make_err)?;
    let e_dispersion = spec.map(|_| r.correction_energy);
    Ok((r, e_dispersion))
}

/// Harmonic frequencies on the KS + `spec` surface: central differences of
/// the corrected analytic gradient, each from the SCF converged at that
/// displaced geometry.
pub(crate) fn dispersion_frequencies(
    ctx: &ParallelContext,
    mol: &ferric_core::mol::Molecule,
    basis_name: &str,
    scf_cfg: &RhfConfig,
    fcfg: &ferric_scf::frequencies::FrequencyConfig,
    spec: &DispersionSpec,
) -> Result<ferric_scf::frequencies::FrequencyResult, ferric_core::FerricError> {
    // Refuse an unsupported configuration (e.g. hessian = "analytic") before
    // the MBD@rsSCS free-atom SCFs below, not after them.
    ferric_scf::frequencies::check_scf_correction_config(fcfg)?;
    let op = Operator::coulomb();
    let bs = ferric_core::basis::bundled(basis_name)?;
    // MBD@rsSCS free-atom references are per element: built once, not at each
    // of the 6N + 1 geometries.
    let cache = match spec {
        DispersionSpec::Mbd(_) => Some(ferric_rpa::dispersion::mbd_scf::MbdFreeAtomCache::build(
            ctx, mol, &bs, op, scf_cfg,
        )?),
        DispersionSpec::D3Bj(_) => None,
    };
    ferric_scf::frequencies::harmonic_frequencies_with_scf_correction(
        ctx,
        mol,
        basis_name,
        op,
        scf_cfg,
        fcfg,
        |m, scf| {
            let d = evaluate_dispersion(spec, ctx, m, &bs, op, scf_cfg, scf, true, cache.as_ref())?;
            Ok((d.energy, d.gradient))
        },
    )
}
