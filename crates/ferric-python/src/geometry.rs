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
            r.search,
            "analytic",
        ),
        Err(why) => {
            ferric_core::certificate::Certificate::ungraded(format!("optimize:{kind}")).warn(why)
        }
    }
}

/// Geometry optimization on any surface ferric has an analytic gradient for.
///
/// `method` picks the surface:
///
/// - `"scf"` (default): Hartree-Fock, or Kohn-Sham with `xc=`. `reference`
///   is `"rhf"`, `"uhf"` or `"rohf"` (with `xc`: RKS, UKS or ROKS); unset, it
///   is `"rhf"` for a singlet and `"uhf"` for an open-shell molecule.
///   `dispersion=` (needs `xc`; the spellings of `run_dft`) optimizes the
///   KS + D3(BJ) or KS + MBD@rsSCS surface, energy and gradient from the same
///   parameters. `solvent=` adds IEF-PCM, `point_charges=` /
///   `external_field=` an external potential. `df_j_aux` / `df_k_aux`
///   density-fit J / K as in `run_frequencies` (unset: exact four-centre J/K
///   for HF; with `xc`, RI-J and RI-K on def2-universal-jkfit for RKS, UKS and
///   ROKS, the CLI's default; `"none"` for exact); `scf_ri_j_precision` /
///   `scf_ri_k_precision` as there.
/// - `"rimp2"`: RI-MP2 (analytic Z-vector gradient): RHF + RI-MP2 on a
///   singlet, UHF + unrestricted RI-MP2 on an open-shell molecule or with
///   `reference="uhf"` (`reference="rohf"` raises: the gradient needs a UHF
///   reference). Needs
///   `auxbasis` (the RI basis, e.g. `"cc-pvdz-ri"`). `scf_jk` is `"rijk"`
///   (default, on `jk_auxbasis`, unset the JK-fit set matched to the orbital
///   basis: cc-pvtz-jkfit for cc-pVDZ/cc-pVTZ, def2-universal-jkfit for a
///   basis without a matched set) or `"exact"`, resolved as the CLI resolves
///   an omitted `[scf] jk`. The fitted gradient checks its peak memory
///   against `memory_budget_gb` (unset: the auto-detected budget) before
///   it allocates and raises when it does not fit. The
///   reference RHF is solved by the gradient driver at a fixed tight density
///   threshold (1e-9), so the `scf_max_iter` / `scf_energy_conv` /
///   `scf_density_conv` kwargs raise here. `frozen_core` and
///   `memory_budget_gb` as in `run_rimp2`.
/// - `"oo-rimp2"`: closed-shell orbital-optimized RI-MP2 (zero-response
///   gradient at the converged orbitals). Needs `auxbasis`; `scf_jk` is
///   `"rijk"` or `"exact"` with the default of `"rimp2"` (four-centre when
///   `scf_eri` is set) on `jk_auxbasis`, whose fitted J/K the
///   orbital steps and the gradient reuse (RI-J stored in f64); `frozen_core`
///   must be 0; `oo_max_iter` / `oo_grad_conv` control the
///   orbital loop (defaults of `run_oo_rimp2`), the `scf_*` kwargs its
///   starting RHF. An orbital optimization that does not converge at some
///   geometry raises.
///
/// - `"scs-mp2"`: closed-shell SCS-MP2, `E_HF + c_os E_os + c_ss E_ss`
///   (analytic Z-vector gradient of that energy). `c_os` / `c_ss` default to
///   Grimme's 6/5 and 1/3 as in `run_scs_mp2`; every other kwarg as for
///   `"rimp2"`.
///
/// - `"att-rimp2"`: attenuated RI-MP2, `E_HF + E_c[erfc(ωr)/r]` on a
///   Coulomb RHF, or on a UHF for an open shell (`reference="uhf"`, the
///   default at multiplicity > 1; ROHF raises as for `"rimp2"`), with the
///   analytic Z-vector gradient of that energy (the energy of
///   `run_attenuated_rimp2`). `omega` in Å⁻¹ (default 0.420); every other
///   kwarg as for `"rimp2"`. The terfc operator has no gradient.
///
/// - `"pdep-rpa"`: closed-shell RHF + RI-dRPA, the CLI's `[rpa]` optimize
///   route (`ferric_rpa::optimize::optimize_geometry_rpa`). Needs `auxbasis`;
///   `frozen_core`, `memory_budget_gb`, `n_quad` (16), `quadrature`
///   (gauss-legendre). The reference RHF is exact four-centre at density_conv
///   1e-9, so the `scf_*`, `scf_jk`, `point_charges` and `external_field`
///   kwargs raise. `rpa_gradient`: `"analytic"` (Z-vector, full rank, dense
///   chi0, Coulomb; raises with a nonzero `trunc_thresh`) or `"fd"` (h = 5e-4
///   Bohr); unset runs analytic where it applies and finite differences
///   otherwise, the reason recorded in the certificate. `trunc_thresh` unset
///   is 0 for analytic and 1e-4 for an explicit `"fd"`.
///
/// The correlated methods are closed-shell only, take no
/// `xc`, `dispersion` or `solvent`, and every kwarg a method does not read
/// raises `ValueError` rather than being ignored.
///
/// The SCF of every step starts from the previous step's orbitals
/// (`scf_warm_start`, default True). Each route calls the library driver the
/// CLI's `task = "optimize"` calls for the same `method.kind`.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_name, max_steps=None, e_conv=None, point_charges=None, external_field=None,
    scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, solvent=None,
    pcm_lebedev_order=None, scf_warm_start=None,
    method="scf", reference=None, xc=None, dispersion=None,
    df_j_aux=None, df_k_aux=None, scf_ri_j_precision=None, scf_ri_k_precision=None,
    auxbasis=None, frozen_core=None, scf_jk=None, jk_auxbasis=None, memory_budget_gb=None,
    oo_max_iter=None, oo_grad_conv=None, c_os=None, c_ss=None, scf_eri=None,
    scf_relativistic=None, fock_hook=None, fock_hook_grad=None, coordinates=None,
    rpa_gradient=None, n_quad=None, quadrature=None, trunc_thresh=None, omega=None,
    initial_hessian=None, trust_update=None,
))]
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
    method: &str,
    reference: Option<&str>,
    xc: Option<&str>,
    dispersion: Option<&str>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    scf_ri_j_precision: Option<&str>,
    scf_ri_k_precision: Option<&str>,
    auxbasis: Option<&str>,
    frozen_core: Option<usize>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    memory_budget_gb: Option<f64>,
    oo_max_iter: Option<usize>,
    oo_grad_conv: Option<f64>,
    c_os: Option<f64>,
    c_ss: Option<f64>,
    scf_eri: Option<&str>,
    scf_relativistic: Option<&str>,
    fock_hook: Option<&Bound<'_, PyAny>>,
    fock_hook_grad: Option<&Bound<'_, PyAny>>,
    coordinates: Option<&str>,
    rpa_gradient: Option<&str>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    trunc_thresh: Option<f64>,
    omega: Option<f64>,
    initial_hessian: Option<&str>,
    trust_update: Option<&str>,
) -> PyResult<PyOptimizeResult> {
    // Before any setup: no method here has an X2C-1e gradient.
    refuse_relativistic_kwarg(scf_relativistic, "analytic nuclear gradient (run_optimize)")?;
    let k = OptimizeKwargs {
        xc,
        dispersion,
        df_j_aux,
        df_k_aux,
        scf_ri_j_precision,
        scf_ri_k_precision,
        auxbasis,
        frozen_core,
        scf_jk,
        jk_auxbasis,
        memory_budget_gb,
        oo_max_iter,
        oo_grad_conv,
        c_os,
        c_ss,
        solvent: solvent.is_some(),
        scf_conv_set: scf_max_iter.is_some()
            || scf_energy_conv.is_some()
            || scf_density_conv.is_some(),
        scf_eri: scf_eri.is_some(),
        external: point_charges.is_some() || external_field.is_some(),
        rpa_gradient,
        n_quad,
        quadrature,
        trunc_thresh,
        omega,
    };
    let method = parse_optimize_method(method)?;
    let spin = parse_optimize_reference(reference, &mol.inner)?;
    refuse_unread_optimize_kwargs(method, spin, &k)?;
    let mut cfg = RhfConfig {
        external_potential: build_external_potential(point_charges, external_field),
        pcm: build_pcm_config(solvent, pcm_lebedev_order)?,
        ..Default::default()
    };
    ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv).apply(&mut cfg);
    cfg.xc = xc.map(str::to_string);
    cfg.df_j_aux = crate::geometry_df_aux(xc, df_j_aux);
    cfg.df_k_aux = crate::geometry_df_aux(xc, df_k_aux);
    geometry_task_scf(&mut cfg, scf_ri_j_precision)?;
    cfg.ri_k_precision = parse_ri_k_precision(scf_ri_k_precision)?;
    cfg.eri = crate::scf::parse_eri(scf_eri)?;
    // Apply the ECP before optimizing: each SCF derives its occupations from
    // `Molecule::nelec()`, which counts ALL electrons until `apply_ecp` has
    // run. Without this an ECP basis (aug-cc-pvdz-pp, Z >= 37) would optimize
    // with the all-electron count against ECP integrals.
    let bs = basis::bundled(basis_name).map_err(make_err)?;
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&bs);
    let mut opt_config = optimize_config(max_steps, e_conv, scf_warm_start, coordinates)?;
    if let Some(s) = initial_hessian {
        opt_config.initial_hessian = Some(
            ferric_scf::optimize::InitialHessian::parse_config_str(s)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        );
    }
    if let Some(s) = trust_update {
        opt_config.trust_update = ferric_scf::optimize::TrustUpdate::parse_config_str(s)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
    }
    if fock_hook.is_some() || fock_hook_grad.is_some() {
        let hooks = check_optimize_hooks(fock_hook, fock_hook_grad, method, &k, scf_warm_start)?;
        let opt_config = OptimizeConfig {
            scf_warm_start: false,
            ..opt_config
        };
        return run_optimize_hooked(py, &emol, basis_name, &cfg, &opt_config, spin, hooks);
    }
    match method {
        OptimizeMethod::Scf => run_optimize_scf(py, &emol, basis_name, &cfg, &opt_config, spin, &k),
        OptimizeMethod::RiMp2 => {
            run_optimize_rimp2(py, &emol, basis_name, &cfg, &opt_config, &k, spin)
        }
        OptimizeMethod::OoRiMp2 => {
            run_optimize_oo_rimp2(py, &emol, basis_name, &cfg, &opt_config, &k)
        }
        OptimizeMethod::ScsMp2 => {
            run_optimize_scs_mp2(py, &emol, basis_name, &cfg, &opt_config, &k)
        }
        OptimizeMethod::AttRiMp2 => {
            run_optimize_att_rimp2(py, &emol, basis_name, &cfg, &opt_config, &k, spin)
        }
        OptimizeMethod::PdepRpa => run_optimize_pdep_rpa(py, &emol, basis_name, &opt_config, &k),
    }
}

/// Optimize an OPEN-SHELL geometry with a UHF reference:
/// `run_optimize(..., reference="uhf")` with the same kwargs.
///
/// The spin state comes from the `Molecule` (its multiplicity), exactly as in
/// `run_uhf` -- there is no `multiplicity` kwarg here, and passing a
/// closed-shell molecule simply runs UHF on a singlet.
#[pyfunction]
#[pyo3(signature = (mol, basis_name, max_steps=None, e_conv=None, point_charges=None, external_field=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, solvent=None, pcm_lebedev_order=None, scf_warm_start=None, coordinates=None))]
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
    coordinates: Option<&str>,
) -> PyResult<PyOptimizeResult> {
    run_optimize(
        py,
        mol,
        basis_name,
        max_steps,
        e_conv,
        point_charges,
        external_field,
        scf_max_iter,
        scf_energy_conv,
        scf_density_conv,
        solvent,
        pcm_lebedev_order,
        scf_warm_start,
        "scf",
        Some("uhf"),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        coordinates,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

/// ValueError for a `scf_relativistic=` value on a task that differentiates
/// the SCF: the X2C-1e core Hamiltonian has no nuclear derivative.
fn refuse_relativistic_kwarg(kwarg: Option<&str>, what: &str) -> PyResult<()> {
    ferric_scf::x2c::refuse_relativistic(crate::scf::parse_relativistic(kwarg)?, what)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// The surface `run_optimize` differentiates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OptimizeMethod {
    Scf,
    RiMp2,
    OoRiMp2,
    ScsMp2,
    AttRiMp2,
    PdepRpa,
}

/// `run_optimize`'s method- and reference-specific kwargs.
pub(crate) struct OptimizeKwargs<'a> {
    xc: Option<&'a str>,
    dispersion: Option<&'a str>,
    df_j_aux: Option<&'a str>,
    df_k_aux: Option<&'a str>,
    scf_ri_j_precision: Option<&'a str>,
    scf_ri_k_precision: Option<&'a str>,
    auxbasis: Option<&'a str>,
    frozen_core: Option<usize>,
    scf_jk: Option<&'a str>,
    jk_auxbasis: Option<&'a str>,
    memory_budget_gb: Option<f64>,
    oo_max_iter: Option<usize>,
    oo_grad_conv: Option<f64>,
    c_os: Option<f64>,
    c_ss: Option<f64>,
    /// `solvent=` was given.
    solvent: bool,
    /// Any of `scf_max_iter` / `scf_energy_conv` / `scf_density_conv` was given.
    scf_conv_set: bool,
    /// `scf_eri=` was given.
    scf_eri: bool,
    /// `point_charges=` or `external_field=` was given.
    external: bool,
    rpa_gradient: Option<&'a str>,
    n_quad: Option<usize>,
    quadrature: Option<&'a str>,
    trunc_thresh: Option<f64>,
    /// `omega=` (Å⁻¹), read by `method="att-rimp2"` only.
    omega: Option<f64>,
}

fn value_error(msg: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg)
}

/// Strict `method=`.
pub(crate) fn parse_optimize_method(method: &str) -> PyResult<OptimizeMethod> {
    match method {
        "scf" => Ok(OptimizeMethod::Scf),
        "rimp2" => Ok(OptimizeMethod::RiMp2),
        "oo-rimp2" => Ok(OptimizeMethod::OoRiMp2),
        "scs-mp2" => Ok(OptimizeMethod::ScsMp2),
        "att-rimp2" => Ok(OptimizeMethod::AttRiMp2),
        "pdep-rpa" => Ok(OptimizeMethod::PdepRpa),
        other => Err(value_error(format!(
            "run_optimize: unknown method {other:?}; expected \"scf\", \"rimp2\", \
             \"oo-rimp2\", \"scs-mp2\", \"att-rimp2\" or \"pdep-rpa\""
        ))),
    }
}

/// Strict `reference=`; unset is RHF for a singlet and UHF otherwise.
pub(crate) fn parse_optimize_reference(
    reference: Option<&str>,
    mol: &Molecule,
) -> PyResult<ferric_scf::Spin> {
    match reference {
        None if mol.multiplicity > 1 => Ok(ferric_scf::Spin::Unrestricted),
        None | Some("rhf") => Ok(ferric_scf::Spin::Restricted),
        Some("uhf") => Ok(ferric_scf::Spin::Unrestricted),
        Some("rohf") => Ok(ferric_scf::Spin::RestrictedOpen),
        Some(other) => Err(value_error(format!(
            "run_optimize: unknown reference {other:?}; expected \"rhf\", \"uhf\" or \"rohf\""
        ))),
    }
}

/// Refuse every kwarg `method` does not read, and a reference it cannot use.
fn refuse_unread_optimize_kwargs(
    method: OptimizeMethod,
    spin: ferric_scf::Spin,
    k: &OptimizeKwargs,
) -> PyResult<()> {
    let unread = |names: &[(&str, bool)], who: &str| -> PyResult<()> {
        match names.iter().find(|(_, set)| *set) {
            Some((name, _)) => Err(value_error(format!(
                "run_optimize: {name} is not read by method=\"{who}\""
            ))),
            None => Ok(()),
        }
    };
    let correlated_only = [
        ("auxbasis", k.auxbasis.is_some()),
        ("frozen_core", k.frozen_core.is_some()),
        ("scf_jk", k.scf_jk.is_some()),
        ("jk_auxbasis", k.jk_auxbasis.is_some()),
        ("memory_budget_gb", k.memory_budget_gb.is_some()),
    ];
    let oo_only = [
        ("oo_max_iter", k.oo_max_iter.is_some()),
        ("oo_grad_conv", k.oo_grad_conv.is_some()),
    ];
    let scs_only = [("c_os", k.c_os.is_some()), ("c_ss", k.c_ss.is_some())];
    let att_only = [("omega", k.omega.is_some())];
    let scf_only = [
        ("xc", k.xc.is_some()),
        ("dispersion", k.dispersion.is_some()),
        ("solvent", k.solvent),
        ("df_j_aux", k.df_j_aux.is_some()),
        ("df_k_aux", k.df_k_aux.is_some()),
        ("scf_ri_k_precision", k.scf_ri_k_precision.is_some()),
    ];
    let rpa_only = [
        ("rpa_gradient", k.rpa_gradient.is_some()),
        ("n_quad", k.n_quad.is_some()),
        ("quadrature", k.quadrature.is_some()),
        ("trunc_thresh", k.trunc_thresh.is_some()),
    ];
    let who = match method {
        OptimizeMethod::Scf => {
            unread(&correlated_only, "scf")?;
            unread(&scs_only, "scf")?;
            unread(&att_only, "scf")?;
            unread(&rpa_only, "scf")?;
            return unread(&oo_only, "scf");
        }
        OptimizeMethod::RiMp2 => "rimp2",
        OptimizeMethod::OoRiMp2 => "oo-rimp2",
        OptimizeMethod::ScsMp2 => "scs-mp2",
        OptimizeMethod::AttRiMp2 => "att-rimp2",
        OptimizeMethod::PdepRpa => "pdep-rpa",
    };
    if method != OptimizeMethod::AttRiMp2 {
        unread(&att_only, who)?;
    }
    if method != OptimizeMethod::PdepRpa {
        unread(&rpa_only, who)?;
    }
    if method != OptimizeMethod::ScsMp2 {
        unread(&scs_only, who)?;
    }
    let has_u_gradient = matches!(method, OptimizeMethod::RiMp2 | OptimizeMethod::AttRiMp2);
    if has_u_gradient && spin == ferric_scf::Spin::RestrictedOpen {
        return Err(value_error(format!(
            "run_optimize: method=\"{who}\": {}",
            ferric_mp2::u_rimp2_gradient::ROHF_REFUSAL
        )));
    }
    let open_ok = has_u_gradient && spin == ferric_scf::Spin::Unrestricted;
    if spin != ferric_scf::Spin::Restricted && !open_ok {
        return Err(value_error(format!(
            "run_optimize: method=\"{who}\" is closed-shell only (the unrestricted MP2 \
             nuclear gradients are method=\"rimp2\" and \"att-rimp2\"); use \
             reference=\"rhf\" on a singlet"
        )));
    }
    unread(&scf_only, who)?;
    if method == OptimizeMethod::PdepRpa {
        // The RPA gradient drivers solve their own exact four-centre RHF at a
        // fixed tight density threshold and carry no environment terms.
        unread(&oo_only, who)?;
        unread(
            &[
                ("scf_jk", k.scf_jk.is_some()),
                ("jk_auxbasis", k.jk_auxbasis.is_some()),
                (
                    "scf_max_iter / scf_energy_conv / scf_density_conv",
                    k.scf_conv_set,
                ),
                ("scf_ri_j_precision", k.scf_ri_j_precision.is_some()),
                ("scf_eri", k.scf_eri),
                ("point_charges / external_field", k.external),
            ],
            who,
        )?;
    }
    if k.auxbasis.is_none() {
        return Err(value_error(format!(
            "run_optimize: method=\"{who}\" needs auxbasis= (the RI basis, e.g. \"cc-pvdz-ri\")"
        )));
    }
    if matches!(
        method,
        OptimizeMethod::RiMp2 | OptimizeMethod::ScsMp2 | OptimizeMethod::AttRiMp2
    ) {
        unread(&oo_only, who)?;
        let fixed_scf = [
            (
                "scf_max_iter / scf_energy_conv / scf_density_conv",
                k.scf_conv_set,
            ),
            ("scf_ri_j_precision", k.scf_ri_j_precision.is_some()),
            ("scf_eri", k.scf_eri),
        ];
        return unread(&fixed_scf, who);
    }
    unread(
        &[("scf_ri_j_precision", k.scf_ri_j_precision.is_some())],
        who,
    )
}

/// The optimizer settings `run_optimize` exposes; the rest stay at the
/// library defaults. `coordinates` takes the CLI's `[optimize] coordinates`
/// values ("cartesian", the default, or "internal") and is just as strict.
fn optimize_config(
    max_steps: Option<usize>,
    e_conv: Option<f64>,
    scf_warm_start: Option<bool>,
    coordinates: Option<&str>,
) -> PyResult<OptimizeConfig> {
    let coord_system = match coordinates {
        None => ferric_scf::optimize::CoordSystem::Cartesian,
        Some(s) => ferric_scf::optimize::CoordSystem::parse_config_str(s)
            .map_err(pyo3::exceptions::PyValueError::new_err)?,
    };
    Ok(OptimizeConfig {
        max_steps: max_steps.unwrap_or(100),
        e_conv: e_conv.unwrap_or(1e-6),
        scf_warm_start: scf_warm_start.unwrap_or(true),
        coord_system,
        ..Default::default()
    })
}

/// `method="scf"`: `optimize_geometry{,_uhf,_rohf}_with_scf_correction`, the
/// drivers of the CLI's rhf/ksdft, uhf and rohf `optimize`, with the
/// dispersion correction (if any) as the additive term.
fn run_optimize_scf(
    py: Python<'_>,
    emol: &Molecule,
    basis_name: &str,
    cfg: &RhfConfig,
    opt_config: &OptimizeConfig,
    spin: ferric_scf::Spin,
    k: &OptimizeKwargs,
) -> PyResult<PyOptimizeResult> {
    use ferric_scf::frequencies::FrequencyReference;
    let fref = match spin {
        ferric_scf::Spin::Restricted => FrequencyReference::Rhf,
        ferric_scf::Spin::Unrestricted => FrequencyReference::Uhf,
        ferric_scf::Spin::RestrictedOpen => FrequencyReference::Rohf,
    };
    let spec = resolve_frequency_dispersion(k.dispersion, k.xc, fref, cfg)?;
    let r = py
        .allow_threads(|| scf_optimize(emol, basis_name, cfg, opt_config, spin, spec.as_ref()))
        .map_err(make_err)?;
    let mut certificate_data = optimize_result_certificate(&r, opt_config, basis_name, cfg, spin);
    if let Some(d) = k.dispersion {
        certificate_data = certificate_data.with_approximation(
            "dispersion",
            format!("{d} in the energy and the gradient (KS + dispersion surface)"),
        );
    }
    Ok(PyOptimizeResult {
        energy: r.energy,
        converged: r.converged,
        steps: r.steps,
        energy_trace: r.energy_trace,
        mol_data: r.mol,
        certificate_data,
    })
}

/// `run_optimize`'s `fock_hook` / `fock_hook_grad` pair, checked: both or
/// neither, `method="scf"` only, no `dispersion` (its terms would not see the
/// hook), and no `scf_warm_start=True` (each hooked step starts from the
/// default guess).
fn check_optimize_hooks<'a, 'py>(
    fock_hook: Option<&'a Bound<'py, PyAny>>,
    fock_hook_grad: Option<&'a Bound<'py, PyAny>>,
    method: OptimizeMethod,
    k: &OptimizeKwargs,
    scf_warm_start: Option<bool>,
) -> PyResult<(&'a Bound<'py, PyAny>, blocks::PyFockHookGrad)> {
    let refuse = |why: &str| Err(value_error(format!("run_optimize: {why}")));
    let (Some(hook), Some(grad)) = (fock_hook, fock_hook_grad) else {
        return refuse(
            "fock_hook and fock_hook_grad go together: the optimizer needs the hook's explicit \
             nuclear derivative dE_hook/dR at fixed density (zeros when the hook depends on \
             the density only)",
        );
    };
    if method != OptimizeMethod::Scf {
        return refuse("fock_hook is read only with method=\"scf\"");
    }
    if k.dispersion.is_some() {
        return refuse("fock_hook cannot be combined with dispersion");
    }
    if scf_warm_start == Some(true) {
        return refuse(
            "scf_warm_start=True is not applied with a fock_hook (each step starts \
             from the default guess)",
        );
    }
    Ok((hook, blocks::PyFockHookGrad::new(grad)?))
}

/// `run_optimize(method="scf", fock_hook=, fock_hook_grad=)`: the BFGS
/// driver of every SCF optimization
/// (`ferric_scf::optimize::optimize_geometry_with_energy_gradient`) on the
/// hooked surface. At each geometry: the hooked SCF of `spin` from the default
/// guess (both hooks are called with the step's `Molecule` as their last
/// argument), its analytic gradient, plus the hook's explicit term. The
/// certificate's reference is the hooked SCF at the returned geometry.
fn run_optimize_hooked(
    py: Python<'_>,
    emol: &Molecule,
    basis_name: &str,
    cfg: &RhfConfig,
    opt_config: &OptimizeConfig,
    spin: ferric_scf::Spin,
    (hook, grad): (&Bound<'_, PyAny>, blocks::PyFockHookGrad),
) -> PyResult<PyOptimizeResult> {
    let cfg = ferric_scf::gradient::gradient_task_config(&cfg.for_geometry_task()).into_owned();
    ferric_scf::rhf::validate_fock_hook(&cfg).map_err(|e| value_error(e.to_string()))?;
    let preflight = match spin {
        ferric_scf::Spin::Restricted => ferric_scf::gradient::preflight_cosx_restricted(&cfg),
        ferric_scf::Spin::Unrestricted => ferric_scf::gradient::preflight_cosx_unrestricted(&cfg),
        ferric_scf::Spin::RestrictedOpen => {
            ferric_scf::gradient::preflight_cosx_restricted_open(&cfg)
        }
    };
    preflight.map_err(|e| value_error(format!("run_optimize: {e}")))?;
    let bs = basis::bundled(basis_name).map_err(make_err)?;
    let nao = PreparedBasis::new(emol, &bs).map_err(make_err)?.nbasis();
    let mut h = blocks::PyFockHook::new(hook, nao)?;
    let mut step = HookedStep {
        bs: &bs,
        cfg: &cfg,
        spin,
        grad: &grad,
        py_err: None,
        lowest: None,
        last: None,
    };
    let r = py.allow_threads(|| {
        ferric_scf::optimize::optimize_geometry_with_energy_gradient(emol, opt_config, |m| {
            step.evaluate(m, &mut h)
        })
    });
    if let Some(e) = step.py_err.take() {
        return Err(e);
    }
    let r = h.finish(r)?;
    let candidates = [step.last, step.lowest];
    let certificate_data = hooked_optimize_certificate(&r, candidates, opt_config, &cfg, spin);
    Ok(PyOptimizeResult {
        energy: r.energy,
        converged: r.converged,
        steps: r.steps,
        energy_trace: r.energy_trace,
        mol_data: r.mol,
        certificate_data,
    })
}

/// One geometry of [`run_optimize_hooked`], with the state the steps share:
/// a Python error raised by `fock_hook_grad` (or by building the geometry
/// argument), re-raised once the optimizer has unwound, and the last and the
/// lowest-energy evaluations, one of which is at the returned geometry (the
/// certificate's reference).
struct HookedStep<'a> {
    bs: &'a ferric_core::basis::BasisSet,
    cfg: &'a RhfConfig,
    spin: ferric_scf::Spin,
    grad: &'a blocks::PyFockHookGrad,
    py_err: Option<PyErr>,
    lowest: Option<(Molecule, ScfResult)>,
    last: Option<(Molecule, ScfResult)>,
}

impl HookedStep<'_> {
    /// Keep `e` for the caller, stashing a Python error.
    fn stash<T>(&mut self, r: PyResult<T>) -> Result<T, ferric_core::FerricError> {
        r.map_err(|e| {
            self.py_err = Some(e);
            ferric_core::FerricError::General("fock_hook_grad failed".into())
        })
    }

    /// The hooked SCF energy and full gradient at `m`.
    fn evaluate(
        &mut self,
        m: &Molecule,
        h: &mut blocks::PyFockHook,
    ) -> Result<(f64, Array2<f64>), ferric_core::FerricError> {
        use ferric_scf::Spin;
        let prep = PreparedBasis::new(m, self.bs)?;
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute_for_screening(op, &prep, self.cfg.screening)?;
        let cfg = ferric_scf::raw_handoff::with_fresh_handoff(self.cfg);
        let geometry = Python::with_gil(|py| h.set_geometry(py, m));
        self.stash(geometry)?;
        let r = hooked_solve(m, &prep, &bounds, &cfg, self.spin, h)?;
        if !r.converged {
            return Err(ferric_core::FerricError::ScfConvergence {
                iterations: r.iterations,
                last_energy: r.energy,
            });
        }
        let bs = self.bs;
        let g = match self.spin {
            Spin::Restricted => ferric_scf::gradient::restricted_scf_gradient,
            Spin::Unrestricted => ferric_scf::gradient::unrestricted_scf_gradient,
            Spin::RestrictedOpen => ferric_scf::gradient::restricted_open_scf_gradient,
        }(m, &prep, bs, op, &bounds, &cfg, &r)?;
        let open = self.spin != Spin::Restricted;
        let g = Python::with_gil(|py| self.grad.add_to(py, g, &r, open, m));
        let g = self.stash(g)?;
        let e = r.energy;
        self.record(m, r);
        Ok((e, g))
    }

    /// Make `(m, r)` the last evaluation; `lowest` keeps the lowest-energy
    /// earlier one when it is below every later one (`None` when the last
    /// evaluation is the lowest).
    fn record(&mut self, m: &Molecule, r: ScfResult) {
        let below = |a: &Option<(Molecule, ScfResult)>, e: f64| {
            a.as_ref().is_some_and(|(_, b)| b.energy < e)
        };
        if let Some(prev) = self.last.take() {
            if !below(&self.lowest, prev.1.energy) {
                self.lowest = Some(prev);
            }
        }
        if !below(&self.lowest, r.energy) {
            self.lowest = None;
        }
        self.last = Some((m.clone(), r));
    }
}

/// The hooked SCF of `spin` at one geometry.
fn hooked_solve(
    m: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    cfg: &RhfConfig,
    spin: ferric_scf::Spin,
    h: &mut blocks::PyFockHook,
) -> Result<ScfResult, ferric_core::FerricError> {
    use ferric_scf::{rhf, rohf, uhf, Spin};
    let ctx = ParallelContext::default();
    match spin {
        Spin::Restricted => {
            let mut f = |d: &Array2<f64>| h.call(d);
            rhf::solve_rhf_with_hook(&ctx, m, prep, Operator::coulomb(), bounds, cfg, &mut f)
        }
        Spin::Unrestricted => {
            let mut f = |a: &Array2<f64>, b: &Array2<f64>| h.call_spin(a, b);
            uhf::solve_uhf_with_hook(&ctx, m, prep, bounds, cfg, &mut f)
        }
        Spin::RestrictedOpen => {
            let mut f = |a: &Array2<f64>, b: &Array2<f64>| h.call_spin(a, b);
            rohf::solve_rohf_with_hook(&ctx, m, prep, bounds, cfg, &mut f)
        }
    }
}

/// Certificate of a hooked optimization: the hooked SCF of the candidate
/// evaluation (last, lowest) at the returned geometry is the reference,
/// recorded with the `fock-hook` approximation. If neither is at the returned
/// geometry the certificate is `not graded` and says so.
fn hooked_optimize_certificate(
    r: &ferric_scf::optimize::OptimizeResult,
    candidates: [Option<(Molecule, ScfResult)>; 2],
    opt_config: &OptimizeConfig,
    cfg: &RhfConfig,
    spin: ferric_scf::Spin,
) -> ferric_core::certificate::Certificate {
    let kind = match (spin, cfg.xc.is_some()) {
        (ferric_scf::Spin::Restricted, false) => "rhf",
        (ferric_scf::Spin::Restricted, true) => "ksdft",
        (ferric_scf::Spin::Unrestricted, _) => "uhf",
        (ferric_scf::Spin::RestrictedOpen, _) => "rohf",
    };
    let same_geometry = |m: &Molecule| {
        m.atoms.len() == r.mol.atoms.len()
            && m.atoms.iter().zip(&r.mol.atoms).all(|(a, b)| {
                a.x.to_bits() == b.x.to_bits()
                    && a.y.to_bits() == b.y.to_bits()
                    && a.zpos.to_bits() == b.zpos.to_bits()
            })
    };
    match candidates
        .into_iter()
        .flatten()
        .find(|(m, _)| same_geometry(m))
    {
        Some((m, scf)) => {
            let reference =
                ferric_scf::certificate::with_fock_hook(stamped_scf_certificate(&m, &scf, cfg));
            ferric_scf::certificate::optimize_certificate(
                kind,
                &reference,
                r.converged,
                r.steps,
                r.energy,
                opt_config,
                r.search,
                "analytic",
            )
        }
        _ => ferric_core::certificate::Certificate::ungraded(format!("optimize:{kind}")).warn(
            "no hooked SCF was kept at the returned geometry, so there is no final-point \
             reference",
        ),
    }
}

/// The SCF optimizer of `spin`, plus `spec`'s energy and analytic gradient at
/// every geometry (none: the correction is `(0.0, None)`, which adds no
/// floating-point operation).
fn scf_optimize(
    mol: &Molecule,
    basis_name: &str,
    cfg: &RhfConfig,
    opt_config: &OptimizeConfig,
    spin: ferric_scf::Spin,
    spec: Option<&DispersionSpec>,
) -> Result<ferric_scf::optimize::OptimizeResult, ferric_core::FerricError> {
    use ferric_scf::optimize::{
        optimize_geometry_rohf_with_scf_correction, optimize_geometry_uhf_with_scf_correction,
        optimize_geometry_with_scf_correction,
    };
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bs = basis::bundled(basis_name)?;
    // MBD@rsSCS free-atom references are per element: built once.
    let cache = match spec {
        Some(DispersionSpec::Mbd(_)) => Some(
            ferric_rpa::dispersion::mbd_scf::MbdFreeAtomCache::build(&ctx, mol, &bs, op, cfg)?,
        ),
        _ => None,
    };
    let correction = |m: &Molecule, scf: &ScfResult| match spec {
        None => Ok((0.0, None)),
        Some(spec) => {
            let d = evaluate_dispersion(spec, &ctx, m, &bs, op, cfg, scf, true, cache.as_ref())?;
            Ok((d.energy, d.gradient))
        }
    };
    match spin {
        ferric_scf::Spin::Restricted => optimize_geometry_with_scf_correction(
            &ctx, mol, basis_name, op, cfg, opt_config, correction,
        ),
        ferric_scf::Spin::Unrestricted => optimize_geometry_uhf_with_scf_correction(
            &ctx, mol, basis_name, op, cfg, opt_config, correction,
        ),
        ferric_scf::Spin::RestrictedOpen => optimize_geometry_rohf_with_scf_correction(
            &ctx, mol, basis_name, op, cfg, opt_config, correction,
        ),
    }
}

/// The certificate of a correlated (`kind` = "rimp2" / "scs-mp2" / "oo-rimp2")
/// optimization: the RHF on `ref_cfg` at the returned geometry is the
/// reference.
#[allow(clippy::too_many_arguments)]
fn correlated_optimize_certificate(
    kind: &str,
    final_mol: &Molecule,
    basis_name: &str,
    ref_cfg: &RhfConfig,
    (converged, steps, energy): (bool, usize, f64),
    opt_config: &OptimizeConfig,
    search: ferric_scf::optimize::SearchSettings,
    gradient: &str,
    spin: ferric_scf::Spin,
) -> ferric_core::certificate::Certificate {
    let (_, reference) = task_reference_certificate(final_mol, basis_name, ref_cfg, spin);
    match reference {
        Ok(reference) => ferric_scf::certificate::optimize_certificate(
            kind, &reference, converged, steps, energy, opt_config, search, gradient,
        ),
        Err(why) => {
            ferric_core::certificate::Certificate::ungraded(format!("optimize:{kind}")).warn(why)
        }
    }
}

/// `method="rimp2"`: `ferric_mp2::optimize::optimize_geometry_rimp2_on`, the
/// driver of the CLI's rimp2 `optimize`, on a four-centre or RI-JK reference.
fn run_optimize_rimp2(
    py: Python<'_>,
    emol: &Molecule,
    basis_name: &str,
    cfg: &RhfConfig,
    opt_config: &OptimizeConfig,
    k: &OptimizeKwargs,
    spin: ferric_scf::Spin,
) -> PyResult<PyOptimizeResult> {
    let scf_jk = mp2_gradient_scf_jk(k, basis_name, "rimp2")?;
    let mp2_config = RiMp2Config {
        frozen_core: k.frozen_core.unwrap_or(0),
        memory_budget_bytes: budget_bytes_from_gb(k.memory_budget_gb),
        ..Default::default()
    };
    let obs = basis::bundled(basis_name).map_err(make_err)?;
    let aux = basis::bundled(k.auxbasis.unwrap_or_default()).map_err(make_err)?;
    let ext = cfg.external_potential.clone();
    // UHF reference (`refuse_unread_optimize_kwargs` admitted only RHF/UHF):
    // the unrestricted gradient.
    let driver = if spin == ferric_scf::Spin::Unrestricted {
        ferric_mp2::optimize::optimize_geometry_u_rimp2_on
    } else {
        ferric_mp2::optimize::optimize_geometry_rimp2_on
    };
    let r = py
        .allow_threads(|| {
            driver(
                emol,
                &obs,
                &aux,
                Operator::coulomb(),
                &mp2_config,
                opt_config,
                ext.as_ref(),
                &scf_jk,
            )
        })
        .map_err(make_err)?;
    Ok(mp2_gradient_optimize_result(
        "rimp2", r, basis_name, ext, &scf_jk, opt_config, spin,
    ))
}

/// The RPA settings of `run_optimize` and `run_frequencies` with
/// `method="pdep-rpa"`, the CLI's optimize settings (`n_quad` 16,
/// Gauss-Legendre) with the strict parsers, plus the requested gradient
/// (`None`: resolve the default). An unset `trunc_thresh` is 0 for the
/// analytic gradient (the default) and the energy runs' 1e-4 for an explicit
/// `rpa_gradient="fd"`.
fn rpa_gradient_config(
    frozen_core: Option<usize>,
    rpa_gradient: Option<&str>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    trunc_thresh: Option<f64>,
    memory_budget_bytes: Option<usize>,
) -> PyResult<(
    ferric_rpa::PdepRpaConfig,
    Option<ferric_rpa::gradient::RpaGradientMethod>,
)> {
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
    use ferric_rpa::gradient::RpaGradientMethod;
    let requested = rpa_gradient
        .map(RpaGradientMethod::parse_config_str)
        .transpose()
        .map_err(|e| value_error(format!("rpa_gradient: {e}")))?;
    let scheme = QuadratureScheme::parse_config_str(quadrature)
        .map_err(|e| value_error(format!("quadrature: {e}")))?;
    let trunc_thresh = trunc_thresh.unwrap_or(match requested {
        Some(RpaGradientMethod::FiniteDifference { .. }) => 1e-4,
        _ => 0.0,
    });
    let cfg = ferric_rpa::PdepRpaConfig {
        frozen_core: frozen_core.unwrap_or(0),
        trunc_thresh,
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: 1e-8,
        quadrature: QuadratureConfig {
            scheme,
            n_points: n_quad.unwrap_or(16),
            u0: 0.5,
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: false,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: ferric_rpa::config::Chi0Sparsity::Dense,
        memory_budget_bytes,
        need_inv_dielectric_freq: false,
        need_eigenvalues_freq: false,
        verbose: false,
    };
    Ok((cfg, requested))
}

/// `method="pdep-rpa"`: `ferric_rpa::optimize::optimize_geometry_rpa`, the
/// driver of the CLI's pdep-rpa `optimize`, with the CLI's settings
/// (`n_quad` 16, Gauss-Legendre). `rpa_gradient` unset resolves through
/// `RpaGradientMethod::resolve_default`: analytic when the configuration is
/// full-rank, dense and Coulomb (unset `trunc_thresh` is then 0), else finite
/// differences, the reason recorded in the certificate's `gradient` entry.
/// `rpa_gradient="analytic"` with a truncating `trunc_thresh` raises.
fn run_optimize_pdep_rpa(
    py: Python<'_>,
    emol: &Molecule,
    basis_name: &str,
    opt_config: &OptimizeConfig,
    k: &OptimizeKwargs,
) -> PyResult<PyOptimizeResult> {
    use ferric_rpa::gradient::RpaGradientMethod;
    let op = Operator::coulomb();
    let (rpa_cfg, requested) = rpa_gradient_config(
        k.frozen_core,
        k.rpa_gradient,
        k.n_quad,
        k.quadrature,
        k.trunc_thresh,
        budget_bytes_from_gb(k.memory_budget_gb),
    )?;
    let (gradient, fallback) = match requested {
        Some(g) => (g, None),
        None => RpaGradientMethod::resolve_default(&rpa_cfg, op),
    };
    let obs = basis::bundled(basis_name).map_err(make_err)?;
    let aux = basis::bundled(k.auxbasis.unwrap_or_default()).map_err(make_err)?;
    let r = py
        .allow_threads(|| {
            ferric_rpa::optimize::optimize_geometry_rpa(
                emol, &obs, &aux, op, &rpa_cfg, opt_config, gradient,
            )
        })
        .map_err(make_err)?;
    let ref_cfg = RhfConfig {
        density_conv: 1e-9,
        ..Default::default()
    }
    .for_geometry_task();
    let certificate_data = correlated_optimize_certificate(
        "pdep-rpa",
        &r.mol,
        basis_name,
        &ref_cfg,
        (r.converged, r.steps, r.energy),
        opt_config,
        r.search,
        &gradient.certificate_label_with(fallback.as_deref()),
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

/// `method="scs-mp2"`: `ferric_mp2::optimize::optimize_geometry_scs_mp2_on`,
/// the driver of the CLI's scs-mp2 `optimize`, on a four-centre or RI-JK
/// reference, `c_os` / `c_ss` defaulting as in `run_scs_mp2`.
fn run_optimize_scs_mp2(
    py: Python<'_>,
    emol: &Molecule,
    basis_name: &str,
    cfg: &RhfConfig,
    opt_config: &OptimizeConfig,
    k: &OptimizeKwargs,
) -> PyResult<PyOptimizeResult> {
    let scf_jk = mp2_gradient_scf_jk(k, basis_name, "scs-mp2")?;
    let d = ferric_mp2::scs::ScsMp2Config::default();
    let scs_config = ferric_mp2::scs::ScsMp2Config {
        c_os: k.c_os.unwrap_or(d.c_os),
        c_ss: k.c_ss.unwrap_or(d.c_ss),
        frozen_core: k.frozen_core.unwrap_or(0),
        memory_budget_bytes: budget_bytes_from_gb(k.memory_budget_gb),
    };
    let obs = basis::bundled(basis_name).map_err(make_err)?;
    let aux = basis::bundled(k.auxbasis.unwrap_or_default()).map_err(make_err)?;
    let ext = cfg.external_potential.clone();
    let r = py
        .allow_threads(|| {
            ferric_mp2::optimize::optimize_geometry_scs_mp2_on(
                emol,
                &obs,
                &aux,
                &scs_config,
                opt_config,
                ext.as_ref(),
                &scf_jk,
            )
        })
        .map_err(make_err)?;
    Ok(mp2_gradient_optimize_result(
        "scs-mp2",
        r,
        basis_name,
        ext,
        &scf_jk,
        opt_config,
        ferric_scf::Spin::Restricted,
    ))
}

/// `method="att-rimp2"`:
/// `ferric_mp2::optimize::optimize_geometry_att_rimp2_on` (RHF) or
/// `optimize_geometry_u_att_rimp2_on` (UHF), the drivers of the CLI's
/// att-rimp2 `optimize` (erfc(ωr)/r correlation on a Coulomb reference,
/// four-centre or RI-JK), `omega` in Å⁻¹ defaulting to 0.420 as in
/// `run_attenuated_rimp2`.
fn run_optimize_att_rimp2(
    py: Python<'_>,
    emol: &Molecule,
    basis_name: &str,
    cfg: &RhfConfig,
    opt_config: &OptimizeConfig,
    k: &OptimizeKwargs,
    spin: ferric_scf::Spin,
) -> PyResult<PyOptimizeResult> {
    let omega = k.omega.unwrap_or(0.420);
    if !(omega.is_finite() && omega > 0.0) {
        return Err(value_error(format!(
            "run_optimize: method=\"att-rimp2\": omega must be finite and > 0 (Å⁻¹), got {omega}"
        )));
    }
    let scf_jk = mp2_gradient_scf_jk(k, basis_name, "att-rimp2")?;
    let att_config = ferric_mp2::attenuated::AttenuatedMp2Config {
        omega: omega * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
        scaling: 1.0,
        frozen_core: k.frozen_core.unwrap_or(0),
        screen_thresh: None,
        memory_budget_bytes: budget_bytes_from_gb(k.memory_budget_gb),
    };
    let obs = basis::bundled(basis_name).map_err(make_err)?;
    let aux = basis::bundled(k.auxbasis.unwrap_or_default()).map_err(make_err)?;
    let ext = cfg.external_potential.clone();
    // UHF reference (`refuse_unread_optimize_kwargs` admitted only RHF/UHF):
    // the unrestricted gradient.
    let driver = if spin == ferric_scf::Spin::Unrestricted {
        ferric_mp2::optimize::optimize_geometry_u_att_rimp2_on
    } else {
        ferric_mp2::optimize::optimize_geometry_att_rimp2_on
    };
    let r = py
        .allow_threads(|| {
            driver(
                emol,
                &obs,
                &aux,
                &att_config,
                opt_config,
                ext.as_ref(),
                &scf_jk,
            )
        })
        .map_err(make_err)?;
    Ok(mp2_gradient_optimize_result(
        "att-rimp2",
        r,
        basis_name,
        ext,
        &scf_jk,
        opt_config,
        spin,
    ))
}

/// `scf_jk` / `jk_auxbasis` of an MP2-family optimize on `basis_name`, by
/// the CLI's rule for an omitted `[scf] jk` (`ferric-cli`
/// `Config::omitted_scf_jk`): unset, RI-JK on the JK-fit set matched to the
/// orbital basis (`ferric_scf::scf_jk::matched_jk_aux`), or four-centre when
/// `scf_eri` (read only by the four-centre route) is set; `jk_auxbasis`
/// without `scf_jk` raises, as `[scf] jk_aux` without `jk` does.
fn optimize_scf_jk(k: &OptimizeKwargs, basis_name: &str) -> PyResult<ferric_scf::scf_jk::ScfJk> {
    use ferric_scf::scf_jk::{matched_jk_aux, ScfJk, ScfJkKeys, ScfJkMode};
    if k.scf_jk.is_none() {
        if let Some(a) = k.jk_auxbasis {
            return Err(value_error(format!(
                "run_optimize: jk_auxbasis = \"{a}\" is only read with scf_jk = \"rijk\""
            )));
        }
    }
    let default = if k.scf_eri {
        ScfJkMode::Exact
    } else {
        ScfJkMode::RiJk
    };
    let mut jk = ScfJk::resolve(k.scf_jk, k.jk_auxbasis, None, default, ScfJkKeys::PYTHON)
        .map_err(|e| value_error(format!("run_optimize: {e}")))?;
    if k.scf_jk.is_none() && jk.mode == ScfJkMode::RiJk {
        jk.aux = Some(matched_jk_aux(basis_name).to_string());
    }
    Ok(jk)
}

/// `scf_jk` / `jk_auxbasis` of a relaxed-MP2-gradient optimize (`who` =
/// "rimp2" / "scs-mp2"): [`optimize_scf_jk`], four-centre or RI-JK; anything
/// else raises.
fn mp2_gradient_scf_jk(
    k: &OptimizeKwargs,
    basis_name: &str,
    who: &str,
) -> PyResult<ferric_scf::scf_jk::ScfJk> {
    use ferric_scf::scf_jk::{ScfJk, ScfJkMode};
    let jk = optimize_scf_jk(k, basis_name)?;
    match jk.mode {
        ScfJkMode::Exact => Ok(ferric_mp2::gradient::exact_scf_jk()),
        ScfJkMode::RiJk => Ok(ScfJk {
            mode: ScfJkMode::RiJk,
            aux: jk.aux.clone(),
            k_builder: None,
        }),
        _ => Err(value_error(format!(
            "run_optimize: method=\"{who}\" differentiates a four-centre or an RI-JK \
             reference (scf_jk=\"exact\" | \"rijk\"); got scf_jk={:?}",
            jk.mode.as_str()
        ))),
    }
}

/// The result of a relaxed-MP2-gradient optimize of `kind`, certified on the
/// reference the gradient driver solves (`total_rimp2_gradient_on` /
/// `total_scs_mp2_gradient_on`).
fn mp2_gradient_optimize_result(
    kind: &str,
    r: ferric_mp2::optimize::RiMp2OptimizeResult,
    basis_name: &str,
    ext: Option<ferric_core::external_potential::ExternalPotential>,
    scf_jk: &ferric_scf::scf_jk::ScfJk,
    opt_config: &OptimizeConfig,
    spin: ferric_scf::Spin,
) -> PyOptimizeResult {
    let mut ref_cfg = RhfConfig {
        density_conv: 1e-9,
        external_potential: ext,
        ..Default::default()
    }
    .for_geometry_task();
    scf_jk.apply(&mut ref_cfg);
    let certificate_data = correlated_optimize_certificate(
        kind,
        &r.mol,
        basis_name,
        &ref_cfg,
        (r.converged, r.steps, r.energy),
        opt_config,
        r.search,
        &ferric_mp2::optimize::rimp2_gradient_label(scf_jk),
        spin,
    );
    PyOptimizeResult {
        energy: r.energy,
        converged: r.converged,
        steps: r.steps,
        energy_trace: r.energy_trace,
        mol_data: r.mol,
        certificate_data,
    }
}

/// `method="oo-rimp2"`: `ferric_mp2::optimize::optimize_geometry_oo_rimp2`,
/// the driver of the CLI's oo-rimp2 `optimize`.
fn run_optimize_oo_rimp2(
    py: Python<'_>,
    emol: &Molecule,
    basis_name: &str,
    cfg: &RhfConfig,
    opt_config: &OptimizeConfig,
    k: &OptimizeKwargs,
) -> PyResult<PyOptimizeResult> {
    use ferric_scf::scf_jk::{ScfJkKeys, OO_BUILDS_EXACT_OR_RIJK};
    if let Some(n) = k.frozen_core.filter(|&n| n > 0) {
        return Err(value_error(format!(
            "run_optimize: method=\"oo-rimp2\" needs frozen_core=0 (got {n}): the core-active \
             and core-virtual rotations are not optimised, so the zero-response gradient is not \
             the derivative of the frozen-core energy"
        )));
    }
    let jk = optimize_scf_jk(k, basis_name)?;
    jk.require_exact_or_rijk(ScfJkKeys::PYTHON, OO_BUILDS_EXACT_OR_RIJK)
        .map_err(|e| value_error(format!("run_optimize: {e}")))?;
    // The reference route is the surface's route: the driver's orbital steps
    // and gradient reuse an RI-JK reference's fits (`df_j_aux`/`df_k_aux`
    // are refused for this method, so `cfg` carries no other fit).
    let mut cfg = cfg.clone();
    jk.apply(&mut cfg);
    let cfg = &cfg;
    let d = OoRiMp2Config::default();
    let oo_config = OoRiMp2Config {
        max_iter: k.oo_max_iter.unwrap_or(d.max_iter),
        grad_conv: k.oo_grad_conv.unwrap_or(d.grad_conv),
        frozen_core: 0,
        memory_budget_bytes: budget_bytes_from_gb(k.memory_budget_gb),
        ..d
    };
    let obs = basis::bundled(basis_name).map_err(make_err)?;
    let aux = basis::bundled(k.auxbasis.unwrap_or_default()).map_err(make_err)?;
    let r = py
        .allow_threads(|| {
            ferric_mp2::optimize::optimize_geometry_oo_rimp2(
                emol,
                &obs,
                &aux,
                Operator::coulomb(),
                cfg,
                &oo_config,
                opt_config,
            )
        })
        .map_err(make_err)?;
    let certificate_data = correlated_optimize_certificate(
        "oo-rimp2",
        &r.mol,
        basis_name,
        &cfg.for_geometry_task(),
        (r.converged, r.steps, r.energy),
        opt_config,
        r.search,
        &ferric_mp2::optimize::oo_rimp2_gradient_label(cfg),
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
    /// to match `Molecule.symbols()`. These are the unit mass-weighted
    /// eigenvectors (masses in electron masses) divided back through by
    /// `sqrt(m)`, so light atoms visibly move further than heavy ones and each
    /// row is `dR/dQ` for the mass-weighted normal coordinate `Q`.
    ///
    /// Needed because `n_imaginary() == 1` is NECESSARY BUT NOT SUFFICIENT for
    /// a transition state -- a methyl rotor gives one imaginary frequency too.
    /// Confirming a saddle means checking the imaginary mode displaces the
    /// atoms along the reaction coordinate, which requires the vector.
    #[pyo3(get)]
    pub(crate) normal_modes: Vec<Vec<f64>>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
    /// The geometry the frequencies were computed at (`thermo`).
    pub(crate) molecule: ferric_core::mol::Molecule,
    /// What `ir_intensities` differentiates, or why it cannot.
    pub(crate) ir_source: Result<IrSource, String>,
    /// `ir_intensities` after its first call.
    pub(crate) ir_cache: Option<Vec<f64>>,
}

/// The SCF of a frequency run, kept so `FrequencyResult.ir_intensities` can
/// differentiate it on demand.
pub(crate) struct IrSource {
    pub(crate) basis_name: String,
    pub(crate) scf_cfg: RhfConfig,
    pub(crate) reference: ferric_scf::frequencies::FrequencyReference,
    pub(crate) normal_modes: ndarray::Array2<f64>,
}

#[pymethods]
impl PyFrequencyResult {
    /// RRHO thermochemistry (dict; see the .pyi). Imaginary modes are left
    /// out with a `RuntimeWarning`.
    #[pyo3(signature = (temperature=298.15, pressure=101325.0, symmetry_number=1))]
    fn thermo(
        &self,
        py: Python<'_>,
        temperature: f64,
        pressure: f64,
        symmetry_number: u32,
    ) -> PyResult<PyObject> {
        let cfg = ferric_scf::thermo::ThermoConfig {
            temperature,
            pressure,
            symmetry_number,
        };
        cfg.validate().map_err(|e| value_error(e.to_string()))?;
        let th = ferric_scf::thermo::rrho_thermochemistry(
            &self.molecule,
            &self.frequencies,
            self.energy,
            &cfg,
        )
        .map_err(make_err)?;
        for w in &th.warnings {
            crate::scf::emit_runtime_warning(py, w.clone());
        }
        let json = py.import("json")?;
        Ok(json
            .call_method1("loads", (th.to_json().to_string(),))?
            .unbind())
    }

    /// IR intensity of each mode in km/mol, in the order of `frequencies`
    /// (the field route: six SCF + gradient evaluations in a field of
    /// +-1e-3 a.u., computed on the first call and cached).
    fn ir_intensities(&mut self, py: Python<'_>) -> PyResult<Vec<f64>> {
        if let Some(a) = &self.ir_cache {
            return Ok(a.clone());
        }
        let src = self
            .ir_source
            .as_ref()
            .map_err(|why| value_error(why.clone()))?;
        let ctx = ParallelContext::default();
        let mol = &self.molecule;
        let apt = py
            .allow_threads(|| {
                ferric_scf::infrared::dipole_derivatives_field(
                    &ctx,
                    mol,
                    &src.basis_name,
                    Operator::coulomb(),
                    &src.scf_cfg,
                    src.reference,
                    ferric_scf::infrared::DEFAULT_IR_FIELD,
                )
            })
            .map_err(make_err)?;
        let a = ferric_scf::infrared::ir_intensities(&apt, &src.normal_modes).map_err(make_err)?;
        self.ir_cache = Some(a.clone());
        Ok(a)
    }

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
/// `df_j_aux` / `df_k_aux` density-fit the Coulomb and exchange of every SCF
/// the run does, with `run_rhf`'s spellings. Unset is exact J/K for HF and,
/// with `xc`, RI-J and RI-K on def2-universal-jkfit for every reference (the
/// CLI's default; `"none"` for exact). The analytic
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
/// CLI and Rust API) (one SCF plus CPHF); for Kohn-Sham (`xc=` with
/// `reference="rhf"` or `"uhf"`) the "analytic-moving-grid" Hessian below
/// wherever it applies; and a finite difference of the ANALYTIC gradient (6N
/// gradient evaluations) otherwise (ROKS, meta-GGA, VV10, range-separated
/// hybrids, external potentials with `xc=`, MBD@rsSCS `dispersion=`, and large
/// systems whose response AO table -- basis functions x grid points x 4 for
/// a GGA, 8 bytes each -- exceeds 2 GiB: on the default grid a GGA above
/// about 8,100 basis functions x atoms). The choice is made before the SCF;
/// an error an analytic path raises after it (a non-converged SCF; for ROHF a
/// Fock-consistency failure from a loose convergence threshold; a
/// non-converging response, non-positive curvature) is raised, not retried
/// with finite differences. "analytic" raises if the Hartree-Fock analytic Hessian does not
/// apply (it never covers KS); "fd" always differences gradients.
/// `.hessian_source` says which ran.
/// "analytic-moving-grid" is the analytic RKS or UKS Hessian on the
/// production (moving) grid, Becke weight response included: the Hessian of
/// the surface the KS gradient differentiates, so it agrees with "fd" to the
/// finite-difference error. It needs `xc=` (LDA/GGA, pure or global hybrid)
/// and `reference="rhf"` or `"uhf"`, and raises before the SCF for
/// Hartree-Fock, "rohf" (ROKS), meta-GGA, VV10, range-separated hybrids,
/// `point_charges=`/`external_field=` and MBD@rsSCS `dispersion=`. An ECP
/// (shells up to f) and D3(BJ) `dispersion=` are supported.
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
/// reference ("rhf", "uhf", "rohf", i.e. RKS, UKS or ROKS). D3(BJ) depends
/// on the geometry only: wherever the run without dispersion would use an
/// analytic Hessian, the Hessian is that one plus the analytic D3(BJ) Hessian;
/// otherwise the central difference of the corrected analytic gradient. With
/// MBD@rsSCS the Hessian is always the central difference of the corrected
/// analytic gradient (KS + dispersion), each from the SCF converged at that
/// displaced geometry, so `hessian="analytic"` and "analytic-moving-grid"
/// raise and "auto" runs finite differences. MBD@rsSCS
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
    scf_warm_start=None, scf_ri_j_precision=None, scf_eri=None, scf_relativistic=None,
    method="scf", auxbasis=None, frozen_core=None, rpa_gradient=None, n_quad=None,
    quadrature=None, trunc_thresh=None, omega=None, scf_jk=None, jk_auxbasis=None,
    memory_budget_gb=None,
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
    scf_eri: Option<&str>,
    scf_relativistic: Option<&str>,
    method: &str,
    auxbasis: Option<&str>,
    frozen_core: Option<usize>,
    rpa_gradient: Option<&str>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    trunc_thresh: Option<f64>,
    omega: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    memory_budget_gb: Option<f64>,
) -> PyResult<PyFrequencyResult> {
    use ferric_scf::frequencies::{FrequencyConfig, FrequencyReference, HessianMethod};
    // Kwargs only `method="att-rimp2"` reads.
    let att_only = [
        ("omega", omega.is_some()),
        ("scf_jk", scf_jk.is_some()),
        ("jk_auxbasis", jk_auxbasis.is_some()),
        ("memory_budget_gb", memory_budget_gb.is_some()),
    ];
    match method {
        "scf" => {
            let rpa_only = [
                ("auxbasis", auxbasis.is_some()),
                ("frozen_core", frozen_core.is_some()),
                ("rpa_gradient", rpa_gradient.is_some()),
                ("n_quad", n_quad.is_some()),
                ("quadrature", quadrature.is_some()),
                ("trunc_thresh", trunc_thresh.is_some()),
            ];
            if let Some((name, _)) = rpa_only.iter().chain(att_only.iter()).find(|(_, set)| *set) {
                return Err(value_error(format!(
                    "run_frequencies: {name} is not read by method=\"scf\""
                )));
            }
        }
        "att-rimp2" => {
            let unread = [
                ("reference", reference.is_some_and(|r| r != "rhf")),
                ("xc", xc.is_some()),
                ("multiplicity", multiplicity.is_some()),
                (
                    "point_charges / external_field",
                    point_charges.is_some() || external_field.is_some(),
                ),
                ("hessian", hessian != "auto"),
                ("dispersion", dispersion.is_some()),
                (
                    "scf_max_iter / scf_energy_conv / scf_density_conv / max_iter / \
                     energy_conv / density_conv",
                    max_iter.is_some()
                        || energy_conv.is_some()
                        || density_conv.is_some()
                        || scf_max_iter.is_some()
                        || scf_energy_conv.is_some()
                        || scf_density_conv.is_some(),
                ),
                (
                    "df_j_aux / df_k_aux / scf_ri_j_precision / scf_ri_k_precision \
                     (use scf_jk / jk_auxbasis)",
                    df_j_aux.is_some()
                        || df_k_aux.is_some()
                        || scf_ri_j_precision.is_some()
                        || scf_ri_k_precision.is_some(),
                ),
                ("scf_relativistic", scf_relativistic.is_some()),
                ("rpa_gradient", rpa_gradient.is_some()),
                ("n_quad", n_quad.is_some()),
                ("quadrature", quadrature.is_some()),
                ("trunc_thresh", trunc_thresh.is_some()),
            ];
            if let Some((name, _)) = unread.iter().find(|(_, set)| *set) {
                return Err(value_error(format!(
                    "run_frequencies: {name} is not read by method=\"att-rimp2\" (closed-shell \
                     RHF + attenuated RI-MP2, central differences of its nuclear gradient)"
                )));
            }
            return run_frequencies_att_rimp2(
                mol,
                basis_name,
                delta,
                scf_warm_start,
                auxbasis,
                frozen_core,
                omega,
                scf_jk,
                jk_auxbasis,
                scf_eri.is_some(),
                memory_budget_gb,
                py,
            );
        }
        "pdep-rpa" => {
            let unread = [
                ("reference", reference.is_some_and(|r| r != "rhf")),
                ("xc", xc.is_some()),
                ("multiplicity", multiplicity.is_some()),
                (
                    "point_charges / external_field",
                    point_charges.is_some() || external_field.is_some(),
                ),
                ("hessian", hessian != "auto"),
                ("dispersion", dispersion.is_some()),
                (
                    "scf_max_iter / scf_energy_conv / scf_density_conv / max_iter / \
                     energy_conv / density_conv",
                    max_iter.is_some()
                        || energy_conv.is_some()
                        || density_conv.is_some()
                        || scf_max_iter.is_some()
                        || scf_energy_conv.is_some()
                        || scf_density_conv.is_some(),
                ),
                (
                    "df_j_aux / df_k_aux / scf_ri_j_precision / scf_ri_k_precision / scf_eri",
                    df_j_aux.is_some()
                        || df_k_aux.is_some()
                        || scf_ri_j_precision.is_some()
                        || scf_ri_k_precision.is_some()
                        || scf_eri.is_some(),
                ),
                ("scf_relativistic", scf_relativistic.is_some()),
            ];
            if let Some((name, _)) = unread.iter().chain(att_only.iter()).find(|(_, set)| *set) {
                return Err(value_error(format!(
                    "run_frequencies: {name} is not read by method=\"pdep-rpa\" (closed-shell \
                     RHF + RI-dRPA, central differences of its nuclear gradient)"
                )));
            }
            return run_frequencies_pdep_rpa(
                mol,
                basis_name,
                delta,
                scf_warm_start,
                auxbasis,
                frozen_core,
                rpa_gradient,
                n_quad,
                quadrature,
                trunc_thresh,
                py,
            );
        }
        other => {
            return Err(value_error(format!(
                "run_frequencies: unknown method {other:?}; expected \"scf\", \"pdep-rpa\" or \"att-rimp2\""
            )))
        }
    }
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
    // Unset = exact J/K for HF, def2-universal-jkfit with `xc` (the CLI's
    // rule, `geometry_df_aux`); same opt-out spellings as run_rhf.
    scf_cfg.df_j_aux = crate::geometry_df_aux(xc, df_j_aux);
    scf_cfg.df_k_aux = crate::geometry_df_aux(xc, df_k_aux);
    geometry_task_scf(&mut scf_cfg, scf_ri_j_precision)?;
    scf_cfg.ri_k_precision = parse_ri_k_precision(scf_ri_k_precision)?;
    scf_cfg.eri = crate::scf::parse_eri(scf_eri)?;
    // Before any SCF (`harmonic_frequencies` refuses it too).
    refuse_relativistic_kwarg(scf_relativistic, "Hessian or gradient (run_frequencies)")?;
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
    let ir_source = Ok(IrSource {
        basis_name: basis_name.to_string(),
        scf_cfg: scf_cfg.clone(),
        reference: refr,
        normal_modes: r.normal_modes.clone(),
    });
    Ok(frequency_result_to_py(
        r,
        e_dispersion,
        certificate_data,
        m,
        ir_source,
    ))
}

/// `run_frequencies(method="pdep-rpa")`: central differences (`delta`, Bohr)
/// of the RHF + RI-dRPA nuclear gradient, `6 N_atoms` gradients, through the
/// same shared mass-weighting and projection as the SCF frequencies. The
/// gradient follows `rpa_gradient` exactly as in `run_optimize`
/// (`resolve_default` when unset).
#[allow(clippy::too_many_arguments)]
fn run_frequencies_pdep_rpa(
    mol: &PyMolecule,
    basis_name: &str,
    delta: Option<f64>,
    scf_warm_start: Option<bool>,
    auxbasis: Option<&str>,
    frozen_core: Option<usize>,
    rpa_gradient: Option<&str>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    trunc_thresh: Option<f64>,
    py: Python<'_>,
) -> PyResult<PyFrequencyResult> {
    use ferric_scf::frequencies::FrequencyConfig;
    let aux_name = auxbasis.ok_or_else(|| {
        value_error(
            "run_frequencies: method=\"pdep-rpa\" needs auxbasis= (the RI basis, e.g. \
             \"cc-pvdz-ri\")"
                .to_string(),
        )
    })?;
    let mut fcfg = FrequencyConfig::default();
    if let Some(d) = delta {
        if !(d.is_finite() && d > 0.0) {
            return Err(value_error(format!(
                "delta must be finite and > 0 (got {d})"
            )));
        }
        fcfg.delta = d;
    }
    fcfg.scf_warm_start = scf_warm_start.unwrap_or(false);
    let (rpa_cfg, requested) = rpa_gradient_config(
        frozen_core,
        rpa_gradient,
        n_quad,
        quadrature,
        trunc_thresh,
        None,
    )?;
    let op = Operator::coulomb();
    let (gradient, fallback) = match requested {
        Some(g) => (g, None),
        None => ferric_rpa::gradient::RpaGradientMethod::resolve_default(&rpa_cfg, op),
    };
    let obs = basis::bundled(basis_name).map_err(make_err)?;
    let aux = basis::bundled(aux_name).map_err(make_err)?;
    let mut m = mol.inner.clone();
    m.apply_ecp(&obs);
    let delta_used = fcfg.delta;
    let r = py
        .allow_threads(|| {
            let mut warm = ferric_scf::warm_start::ScfWarmStart::anchored(fcfg.scf_warm_start);
            let (energy, _) = ferric_rpa::gradient::total_rpa_gradient_by(
                &m, &obs, &aux, op, &rpa_cfg, gradient, &mut warm,
            )?;
            ferric_scf::frequencies::harmonic_frequencies_from_gradient(
                &m,
                delta_used,
                energy,
                |g| {
                    ferric_rpa::gradient::total_rpa_gradient_by(
                        g, &obs, &aux, op, &rpa_cfg, gradient, &mut warm,
                    )
                    .map(|(_, grad)| grad)
                },
            )
        })
        .map_err(make_err)?;
    let ref_cfg = RhfConfig {
        density_conv: 1e-9,
        ..Default::default()
    }
    .for_geometry_task();
    let (kind, reference) =
        task_reference_certificate(&m, basis_name, &ref_cfg, ferric_scf::Spin::Restricted);
    let _ = kind;
    let certificate_data = match reference {
        Ok(reference) => {
            ferric_scf::certificate::frequencies_certificate("pdep-rpa", &reference, &r, &fcfg)
                .with_approximation(
                    "gradient",
                    gradient.certificate_label_with(fallback.as_deref()),
                )
        }
        Err(why) => {
            ferric_core::certificate::Certificate::ungraded("frequencies:pdep-rpa").warn(why)
        }
    };
    let no_ir = Err(
        "ir_intensities: method=\"pdep-rpa\" has no dipole derivative; IR intensities are \
         computed for the SCF references only"
            .to_string(),
    );
    Ok(frequency_result_to_py(r, None, certificate_data, m, no_ir))
}

/// `run_frequencies(method="att-rimp2")`: central differences (`delta`, Bohr)
/// of the closed-shell attenuated RI-MP2 nuclear gradient
/// (`total_att_rimp2_gradient_on`, the surface of the CLI's att-rimp2
/// `frequencies`), `6 N_atoms` gradients, through the same shared
/// mass-weighting and projection as the SCF frequencies. `omega` (Å⁻¹) and
/// the reference (`scf_jk`, `jk_auxbasis`) follow `run_optimize`.
#[allow(clippy::too_many_arguments)]
fn run_frequencies_att_rimp2(
    mol: &PyMolecule,
    basis_name: &str,
    delta: Option<f64>,
    scf_warm_start: Option<bool>,
    auxbasis: Option<&str>,
    frozen_core: Option<usize>,
    omega: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    scf_eri_set: bool,
    memory_budget_gb: Option<f64>,
    py: Python<'_>,
) -> PyResult<PyFrequencyResult> {
    use ferric_scf::frequencies::FrequencyConfig;
    let aux_name = auxbasis.ok_or_else(|| {
        value_error(
            "run_frequencies: method=\"att-rimp2\" needs auxbasis= (the RI basis, e.g. \
             \"cc-pvdz-ri\")"
                .to_string(),
        )
    })?;
    if mol.inner.multiplicity != 1 {
        return Err(value_error(
            "run_frequencies: method=\"att-rimp2\" is closed-shell (singlet) only".to_string(),
        ));
    }
    let omega = omega.unwrap_or(0.420);
    if !(omega.is_finite() && omega > 0.0) {
        return Err(value_error(format!(
            "run_frequencies: method=\"att-rimp2\": omega must be finite and > 0 (Å⁻¹), got {omega}"
        )));
    }
    let mut fcfg = FrequencyConfig::default();
    if let Some(d) = delta {
        if !(d.is_finite() && d > 0.0) {
            return Err(value_error(format!(
                "delta must be finite and > 0 (got {d})"
            )));
        }
        fcfg.delta = d;
    }
    fcfg.scf_warm_start = scf_warm_start.unwrap_or(false);
    let k = OptimizeKwargs {
        xc: None,
        dispersion: None,
        df_j_aux: None,
        df_k_aux: None,
        scf_ri_j_precision: None,
        scf_ri_k_precision: None,
        auxbasis,
        frozen_core,
        scf_jk,
        jk_auxbasis,
        memory_budget_gb,
        oo_max_iter: None,
        oo_grad_conv: None,
        c_os: None,
        c_ss: None,
        solvent: false,
        scf_conv_set: false,
        scf_eri: scf_eri_set,
        external: false,
        rpa_gradient: None,
        n_quad: None,
        quadrature: None,
        trunc_thresh: None,
        omega: Some(omega),
    };
    let scf_jk = mp2_gradient_scf_jk(&k, basis_name, "att-rimp2")?;
    let att_config = ferric_mp2::attenuated::AttenuatedMp2Config {
        omega: omega * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
        scaling: 1.0,
        frozen_core: frozen_core.unwrap_or(0),
        screen_thresh: None,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
    };
    let obs = basis::bundled(basis_name).map_err(make_err)?;
    let aux = basis::bundled(aux_name).map_err(make_err)?;
    let mut m = mol.inner.clone();
    m.apply_ecp(&obs);
    let delta_used = fcfg.delta;
    let r = py
        .allow_threads(|| {
            let mut warm = ferric_scf::warm_start::ScfWarmStart::anchored(fcfg.scf_warm_start);
            let (energy, _) = ferric_mp2::att_rimp2_gradient::total_att_rimp2_gradient_on(
                &m,
                &obs,
                &aux,
                &att_config,
                None,
                &scf_jk,
                &mut warm,
            )?;
            ferric_scf::frequencies::harmonic_frequencies_from_gradient(
                &m,
                delta_used,
                energy,
                |g| {
                    ferric_mp2::att_rimp2_gradient::total_att_rimp2_gradient_on(
                        g,
                        &obs,
                        &aux,
                        &att_config,
                        None,
                        &scf_jk,
                        &mut warm,
                    )
                    .map(|(_, grad)| grad)
                },
            )
        })
        .map_err(make_err)?;
    let mut ref_cfg = RhfConfig {
        density_conv: 1e-9,
        ..Default::default()
    }
    .for_geometry_task();
    scf_jk.apply(&mut ref_cfg);
    let (_, reference) =
        task_reference_certificate(&m, basis_name, &ref_cfg, ferric_scf::Spin::Restricted);
    let certificate_data = match reference {
        Ok(reference) => {
            ferric_scf::certificate::frequencies_certificate("att-rimp2", &reference, &r, &fcfg)
                .with_approximation(
                    "gradient",
                    ferric_mp2::optimize::rimp2_gradient_label(&scf_jk),
                )
        }
        Err(why) => {
            ferric_core::certificate::Certificate::ungraded("frequencies:att-rimp2").warn(why)
        }
    };
    let no_ir = Err(
        "ir_intensities: method=\"att-rimp2\" has no dipole derivative; IR intensities are \
         computed for the SCF references only"
            .to_string(),
    );
    Ok(frequency_result_to_py(r, None, certificate_data, m, no_ir))
}

/// The `PyFrequencyResult` of a library `FrequencyResult`.
fn frequency_result_to_py(
    r: ferric_scf::frequencies::FrequencyResult,
    e_dispersion: Option<f64>,
    certificate_data: ferric_core::certificate::Certificate,
    molecule: ferric_core::mol::Molecule,
    ir_source: Result<IrSource, String>,
) -> PyFrequencyResult {
    PyFrequencyResult {
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
        normal_modes: r
            .normal_modes
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        certificate_data,
        molecule,
        ir_source,
        ir_cache: None,
    }
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
            molecule: m,
            ir_source: Err(
                "ir_intensities: GFN-xTB frequencies have no dipole derivative \
                            in ferric"
                    .to_string(),
            ),
            ir_cache: None,
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

/// Harmonic frequencies on the KS + `spec` surface. D3(BJ) depends on the
/// geometry only: the analytic KS Hessian plus the analytic D3(BJ) Hessian
/// wherever the run without dispersion would use an analytic Hessian, central
/// differences of the corrected gradient otherwise
/// (`harmonic_frequencies_with_geometric_correction`). MBD@rsSCS follows the
/// density: always central differences of the corrected analytic gradient,
/// each from the SCF converged at that displaced geometry.
pub(crate) fn dispersion_frequencies(
    ctx: &ParallelContext,
    mol: &ferric_core::mol::Molecule,
    basis_name: &str,
    scf_cfg: &RhfConfig,
    fcfg: &ferric_scf::frequencies::FrequencyConfig,
    spec: &DispersionSpec,
) -> Result<ferric_scf::frequencies::FrequencyResult, ferric_core::FerricError> {
    let op = Operator::coulomb();
    if let DispersionSpec::D3Bj(functional) = spec {
        let params = ferric_d3::d3bj_params_for_functional(functional)?;
        return ferric_scf::frequencies::harmonic_frequencies_with_geometric_correction(
            ctx,
            mol,
            basis_name,
            op,
            scf_cfg,
            fcfg,
            |m| d3_geometric_correction(m, &params),
        );
    }
    // Refuse an unsupported configuration (e.g. hessian = "analytic") before
    // the MBD@rsSCS free-atom SCFs below, not after them.
    ferric_scf::frequencies::check_scf_correction_config(fcfg)?;
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

/// D3(BJ) energy, gradient and Hessian at `mol`.
fn d3_geometric_correction(
    mol: &ferric_core::mol::Molecule,
    params: &ferric_d3::D3Params,
) -> Result<ferric_scf::frequencies::GeometricCorrection, ferric_core::FerricError> {
    let energy = ferric_d3::d3bj_energy_for_molecule(mol, params)?;
    let rows = ferric_d3::d3bj_gradient_for_molecule(mol, params)?;
    let gradient = ndarray::Array2::from_shape_fn((rows.len(), 3), |(k, a)| rows[k][a]);
    let hessian = ferric_d3::d3bj_hessian_for_molecule(mol, params)?;
    Ok(ferric_scf::frequencies::GeometricCorrection {
        energy,
        gradient,
        hessian,
    })
}
