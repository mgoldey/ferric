//! MP2-family bindings: moved verbatim from `lib.rs`.

use super::*;

// ── RI-MP2 ──

/// Result of a `run_rimp2` calculation.
#[pyclass]
#[pyo3(name = "RiMp2Result")]
pub(crate) struct PyRiMp2Result {
    /// Reference + MP2 correlation energy, Hartree.
    #[pyo3(get)]
    pub(crate) total_energy: f64,
    /// The converged reference SCF energy alone, Hartree: RHF for a
    /// closed-shell molecule, UHF for an open-shell one (see `reference`).
    /// The name is kept for compatibility.
    #[pyo3(get)]
    pub(crate) rhf_energy: f64,
    /// MP2 correlation energy alone (always negative), Hartree.
    #[pyo3(get)]
    pub(crate) mp2_corr: f64,
    /// The SCF reference the correlation was built on: "RHF" (closed-shell
    /// RI-MP2) or "UHF" (unrestricted RI-MP2 for multiplicity > 1).
    #[pyo3(get)]
    pub(crate) reference: String,
    /// The local model: `None` for the exact RI-MP2, else a dict with
    /// `scheme`, `eps`, `keep_fraction`, `pair_fraction`, `integral_direct`,
    /// `e_corr_canonical_ri` (the exact reference, or `None` unless
    /// `compute_reference=True`), the solver counters, and the model
    /// certificate (`ferric_mp2::lmp2_certificate`: maps, precision, CG
    /// state, validation grade, eps trust line, warnings, build stamp).
    /// Read through the `local` getter.
    pub(crate) local: Option<Py<pyo3::types::PyDict>>,
    /// How the reference SCF built J/K: "exact", "rijk" or "rijcosx".
    #[pyo3(get)]
    pub(crate) scf_jk: String,
    /// The printout's SCF J/K echo (`ScfJk::label`: names a non-default
    /// exchange builder on the exact path).
    pub(crate) jk_label: String,
    /// The SCF's JK-fitting basis (`None` for "exact").
    #[pyo3(get)]
    pub(crate) jk_auxbasis: Option<String>,
    /// Precision provenance of the correlation energy and of the reference
    /// SCF; read through the `precision` / `scf_precision` getters.
    pub(crate) precision_rec: ferric_core::gpu::provenance::PrecisionRecord,
    pub(crate) scf_precision_rec: ferric_core::gpu::provenance::PrecisionRecord,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
    /// `dE_total/dR` (natoms x 3, Hartree/Bohr) when solved with `with_gradient=True`.
    pub(crate) gradient_data: Option<Array2<f64>>,
}

#[pymethods]
impl PyRiMp2Result {
    /// Analytic nuclear gradient of `total_energy` as an (natoms, 3) array in
    /// Hartree/Bohr, rows in the Molecule's atom order; `None` unless the
    /// result came from `run_rimp2(..., with_gradient=True)`.
    fn gradient<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.gradient_data
            .as_ref()
            .map(|g| PyArray2::from_array(py, g))
    }
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// capability `"rimp2"` (`"att-rimp2"` for `run_terfc_rimp2`), the grade
    /// capped by the reference SCF's, the auxiliary basis, frozen core and
    /// correlation operator, the reference's approximations as `scf <name>`,
    /// and for a local run the local-scheme cap and warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    /// Precision provenance of the MP2 correlation energy (the
    /// `rimp2-energy` kernel): `{"precision": "f64", "counter_scope": "process", "kernels": []}` for an
    /// all-f64 run. See `RhfResult.precision` for the schema.
    #[getter]
    fn precision(&self, py: Python<'_>) -> PyResult<PyObject> {
        precision_dict(py, &self.precision_rec)
    }
    /// Precision provenance of the reference SCF (see `RhfResult.precision`).
    #[getter]
    fn scf_precision(&self, py: Python<'_>) -> PyResult<PyObject> {
        precision_dict(py, &self.scf_precision_rec)
    }
    /// The local model: `None` for the exact RI-MP2, else a dict with
    /// `scheme`, `eps`, `keep_fraction`, `pair_fraction`, `integral_direct`,
    /// `e_corr_canonical_ri` (`None` unless `compute_reference=True`), the
    /// solver counters and the model certificate (maps, precision, CG
    /// state, `validation_grade`, `eps_trust`, `warnings`, `build_commit`,
    /// `build_dirty`).
    #[getter]
    fn local(&self, py: Python<'_>) -> Option<Py<pyo3::types::PyDict>> {
        self.local.as_ref().map(|d| d.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        format!(
            "RiMp2Result(total_energy={:.10}, mp2_corr={:.10})",
            self.total_energy, self.mp2_corr
        )
    }
    fn __str__(&self) -> String {
        format!(
            "RI-MP2 {} Total Energy: {:.10} Ha ({}: {:.10}, corr: {:.10}, SCF J/K: {})",
            if self.local.is_some() {
                "(local)"
            } else {
                "(exact)"
            },
            self.total_energy,
            self.reference,
            self.rhf_energy,
            self.mp2_corr,
            self.jk_label,
        )
    }
}

/// Resolve the `local=` / `eps=` / `compute_reference=` / `integral_direct=`
/// (+ integral-direct knobs) kwargs of `run_rimp2` / `run_drpa` /
/// `run_linlccd` with the SAME rules as the CLI's `[local]` section
/// (`ferric_cli::LocalCfg::model`): `local` is `None`/`"none"` (exact) or
/// `"amplitude-threshold"`; `eps` is required with the threshold and refused
/// without it; `compute_reference` and every integral-direct knob are refused
/// on the exact method. `Ok(None)` is the exact method. Errors are
/// `ValueError`, raised before any SCF.
pub(crate) fn local_model_from_kwargs(
    fname: &str,
    cfg: ferric_cli::LocalCfg,
) -> PyResult<Option<ferric_cli::LocalModel>> {
    cfg.model().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{fname}: {} (Python kwargs mirror the CLI [local] keys: local= is scheme, \
             compute_reference= is reference)",
            e.trim_start_matches("[local] ")
        ))
    })
}

/// The `local` dict every local result carries: scheme, threshold, kept
/// fraction, whether the integral-direct path ran.
pub(crate) fn local_dict<'py>(
    py: Python<'py>,
    eps: f64,
    keep_fraction: f64,
    integral_direct: bool,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item(
        "scheme",
        ferric_cli::LocalScheme::AmplitudeThreshold.as_str(),
    )?;
    d.set_item("eps", eps)?;
    d.set_item("keep_fraction", keep_fraction)?;
    d.set_item("integral_direct", integral_direct)?;
    Ok(d)
}

/// The solver/pattern counters and stage timings of an LMP2 result, into
/// its `local` dict.
pub(crate) fn set_lmp2_counters(
    d: &Bound<'_, pyo3::types::PyDict>,
    r: &ferric_mp2::lmp2_amplitude::AmplitudeLmp2Result,
    reference: bool,
) -> PyResult<()> {
    d.set_item("pair_fraction", r.pair_fraction)?;
    d.set_item(
        "e_corr_canonical_ri",
        reference.then_some(r.e_corr_canonical_ri),
    )?;
    d.set_item("dom_mean", r.dom_mean)?;
    d.set_item("dom_max", r.dom_max)?;
    d.set_item("cg_iterations", r.cg_iterations)?;
    d.set_item("cg_lanczos_kappa_lower", r.cg_lanczos_kappa_est)?;
    d.set_item("block_density_mean", r.block_density_mean)?;
    d.set_item("block_density_min", r.block_density_min)?;
    d.set_item("n_pairs_unique", r.n_pairs_unique)?;
    d.set_item("gate_on", r.gate_on)?;
    d.set_item("t_spaces_s", r.timings.t_spaces_s)?;
    d.set_item("t_assembly_s", r.timings.t_assembly_s)?;
    d.set_item("t_solve_s", r.timings.t_solve_s)
}

/// The integral-direct assembly counters and stage timers, into an LMP2
/// result's `local` dict.
pub(crate) fn set_direct_stats(
    d: &Bound<'_, pyo3::types::PyDict>,
    st: &ferric_mp2::lmp2_direct::DirectStats,
) -> PyResult<()> {
    d.set_item("strip_rows_mean", st.strip_rows_mean)?;
    d.set_item("strip_rows_max", st.strip_rows_max)?;
    d.set_item("strip_cols_mean", st.strip_cols_mean)?;
    d.set_item("strip_cols_max", st.strip_cols_max)?;
    d.set_item("n_eri3_shell_triples", st.n_eri3_shell_triples)?;
    d.set_item("n_eri3_skipped", st.n_eri3_skipped)?;
    d.set_item("virt_cand_mean", st.virt_cand_mean)?;
    d.set_item("virt_cand_max", st.virt_cand_max)?;
    d.set_item("t_maps_s", st.t_maps_s)?;
    d.set_item("t_eri3_s", st.t_eri3_s)?;
    d.set_item("t_eri3_fill_thread_s", st.t_eri3_fill_thread_s)?;
    d.set_item("t_eri3_eval_thread_s", st.t_eri3_eval_thread_s)?;
    d.set_item("t_eri3_scatter_thread_s", st.t_eri3_scatter_thread_s)?;
    d.set_item("t_eri3_transform_thread_s", st.t_eri3_transform_thread_s)?;
    d.set_item("n_batches", st.n_batches)?;
    d.set_item("batch_work_max_s", st.batch_work_max_s)?;
    d.set_item("batch_work_mean_s", st.batch_work_mean_s)?;
    d.set_item("t_metric_s", st.t_metric_s)?;
    d.set_item("t_pairs_s", st.t_pairs_s)?;
    d.set_item("n_pair_groups", st.n_pair_groups)?;
    d.set_item("n_distinct_occ_per_group", st.n_distinct_occ_per_group)?;
    d.set_item("n_trsm", st.n_trsm)?;
    d.set_item("n_w_reused", st.n_w_reused)?;
    d.set_item("t_loc_s", st.t_loc_s)
}

/// Set one certificate entry on `d` (`None`, bool, int, float, str, list of
/// str, or a nested dict).
pub(crate) fn set_cert_item(
    py: Python<'_>,
    d: &Bound<'_, pyo3::types::PyDict>,
    k: &str,
    v: &ferric_mp2::lmp2_certificate::CertValue,
) -> PyResult<()> {
    use ferric_mp2::lmp2_certificate::CertValue as V;
    match v {
        V::Null => d.set_item(k, py.None()),
        V::Bool(b) => d.set_item(k, *b),
        V::Usize(n) => d.set_item(k, *n),
        V::F64(x) => d.set_item(k, *x),
        V::Str(s) => d.set_item(k, s),
        V::StrList(l) => d.set_item(k, l),
        V::Map(m) => {
            let sub = pyo3::types::PyDict::new(py);
            for (sk, x) in m {
                set_cert_item(py, &sub, sk, x)?;
            }
            d.set_item(k, sub)
        }
    }
}

/// Put every LMP2 certificate field
/// ([`ferric_mp2::lmp2_certificate::Lmp2Certificate::fields`]) and the
/// build stamp (`build_commit`, `build_dirty`) into a result's `local`
/// dict — the same entries the CLI writes to its run-log JSON.
pub(crate) fn set_certificate(
    py: Python<'_>,
    d: &Bound<'_, pyo3::types::PyDict>,
    cert: &ferric_mp2::lmp2_certificate::Lmp2Certificate,
) -> PyResult<()> {
    for (k, v) in cert.fields() {
        set_cert_item(py, d, k, &v)?;
    }
    d.set_item("build_commit", ferric_build_info::COMMIT)?;
    d.set_item("build_dirty", ferric_build_info::DIRTY)?;
    Ok(())
}

/// `run_rimp2`'s compute: `(reference SCF energy, MP2 correlation, total)`.
///
/// multiplicity > 1: UHF + unrestricted RI-MP2 (`solve_rhf` refuses the
/// molecule; see `ferric_scf::rhf::require_closed_shell`). A non-converged
/// SCF is an error on both routes.
pub(crate) fn rimp2_energies(
    mol: &Molecule,
    basis_set: &basis::BasisSet,
    auxbasis: &basis::BasisSet,
    rhf_config: &RhfConfig,
    mp2_config: &RiMp2Config,
    with_gradient: bool,
) -> Result<Rimp2Energies, ferric_core::FerricError> {
    let prep = PreparedBasis::new(mol, basis_set)?;
    let dfbs = PreparedBasis::new(mol, auxbasis)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep)?;
    let ctx = ParallelContext::default();
    let open_shell = mol.multiplicity > 1;
    let scf = if open_shell {
        solve_uhf(&ctx, mol, &prep, &bounds, rhf_config)?
    } else {
        solve_rhf(&ctx, mol, &prep, op, &bounds, rhf_config)?
    };
    if !scf.converged {
        return Err(ferric_core::FerricError::ScfConvergence {
            iterations: scf.iterations,
            last_energy: scf.energy,
        });
    }
    let (mp2_corr, total_energy, precision) = if open_shell {
        let u = ferric_mp2::u_rimp2::u_ri_mp2(mol, &prep, &dfbs, op, &scf, mp2_config)?;
        (u.mp2_corr, u.total_energy, u.precision)
    } else {
        let mp2 = ri_mp2(mol, &prep, &dfbs, op, &scf, mp2_config)?;
        (mp2.mp2_corr, mp2.total_energy, mp2.precision)
    };
    // Closed shell only: `run_rimp2` refuses an open-shell gradient first.
    let gradient = with_gradient
        .then(|| {
            ferric_mp2::gradient::rimp2_gradient_analytical(
                mol,
                &prep,
                &dfbs,
                op,
                &bounds,
                &scf,
                mp2_config,
                rhf_config.external_potential.as_ref(),
            )
        })
        .transpose()?;
    let reference = reference_certificate(mol, &scf, rhf_config, basis_set);
    let certificate = ferric_mp2::certificate::with_rimp2_envelope(
        ferric_mp2::certificate::rimp2_certificate(
            &reference,
            &auxbasis.name,
            mp2_config,
            &op,
            &precision,
        ),
        &reference,
        mol,
        basis_set,
        auxbasis,
    );
    Ok(Rimp2Energies {
        certificate,
        gradient,
        scf_energy: scf.energy,
        mp2_corr,
        total_energy,
        precision,
        scf_precision: scf.precision,
    })
}

/// What [`rimp2_energies`] returns: the energies and both precision records.
pub(crate) struct Rimp2Energies {
    pub(crate) scf_energy: f64,
    pub(crate) mp2_corr: f64,
    pub(crate) total_energy: f64,
    pub(crate) precision: ferric_core::gpu::provenance::PrecisionRecord,
    pub(crate) scf_precision: ferric_core::gpu::provenance::PrecisionRecord,
    pub(crate) certificate: ferric_core::certificate::Certificate,
    /// `dE_total/dR` (natoms x 3, Hartree/Bohr) when requested.
    pub(crate) gradient: Option<Array2<f64>>,
}

/// The attenuated-correlation kwargs of `run_rimp2` as the CLI's `[mp2]`
/// keys, plus whether any was set.
pub(crate) fn att_keys_from_kwargs(
    att_operator: Option<String>,
    omega: Option<f64>,
    att_r0: Option<f64>,
    att_omega: Option<f64>,
) -> (ferric_cli::Mp2Cfg, bool) {
    let keys = ferric_cli::Mp2Cfg {
        att_operator,
        omega,
        att_r0,
        att_omega,
        ..Default::default()
    };
    let sets = keys.att_operator.is_some()
        || keys.omega.is_some()
        || keys.att_r0.is_some()
        || keys.att_omega.is_some();
    (keys, sets)
}

pub(crate) fn att_keys_on_exact_error() -> PyErr {
    pyo3::exceptions::PyValueError::new_err(
        "run_rimp2: att_operator / omega / att_r0 / att_omega select the attenuated \
         correlation operator of the LOCAL (amplitude-threshold) MP2 and are not read \
         by the exact method; use run_terfc_rimp2 / run_attenuated_rimp2 for exact \
         attenuated RI-MP2, or set local=\"amplitude-threshold\"",
    )
}

/// Correlation operator of a local `run_rimp2`: Coulomb unless an attenuator
/// was selected (the SCF is always full Coulomb). Strict, shared with the
/// CLI's `Mp2Cfg::att_rimp2_operator`.
pub(crate) fn local_correlation_operator(
    att_keys: Option<&ferric_cli::Mp2Cfg>,
    model: &ferric_cli::LocalModel,
) -> PyResult<Operator> {
    let value_err = pyo3::exceptions::PyValueError::new_err;
    let Some(k) = att_keys else {
        return Ok(Operator::coulomb());
    };
    if k.att_operator.is_none() {
        return Err(value_err(
            "run_rimp2: omega / att_r0 / att_omega parameterize an attenuated operator and \
             would be silently ignored without att_operator; set att_operator=\"erfc\" \
             (omega) or \"terfc\" (att_r0, att_omega), or remove them"
                .to_string(),
        ));
    }
    let op = k
        .att_rimp2_operator()
        .map_err(|e| value_err(format!("run_rimp2: {e}")))?;
    if matches!(op.kind, ferric_integrals::operator::OperatorKind::Terfc)
        && model.direct.as_ref().is_some_and(|d| d.schwarz_skip > 0.0)
    {
        ferric_integrals::schwarz::check_terfc_parameters_validated(op)
            .map_err(|e| value_err(format!("run_rimp2: integral_direct: {e}")))?;
    }
    Ok(op)
}

/// Resolution-of-identity (density-fitted) MP2. Runs its own internal SCF
/// first (via `k_builder`, same convention as `run_rhf`'s `k_builder`
/// kwarg), then the RI-MP2 correlation energy using `auxbasis` as the
/// fitting basis.
///
/// EXACT by default. `local="amplitude-threshold"` with `eps=` (required: the
/// threshold is part of the model and has no default) runs the
/// amplitude-threshold local MP2 instead (WSHG23 single threshold,
/// closed-shell, `task` energy only); `eps=0` reproduces the exact RI-MP2.
/// `compute_reference=True` also computes the exact RI-MP2 and puts it in
/// `result.local["e_corr_canonical_ri"]`. `integral_direct=True` selects the
/// integral-direct local MP2, which never forms the global 3-index tensor;
/// its locality maps are `aux_radius` (Bohr, default 10.0), `virt_radius`
/// (Bohr, 12.0), `ao_tail` (1e-3), `schwarz_skip` (1e-5; 0 disables the cut), `batch_merge` (4), `gate_cal` (pair gate, off) and
/// `virt_schwarz_kappa` (off), the same names and defaults as the CLI's
/// `[local]` keys. The rules are the CLI's: every local kwarg is a
/// `ValueError` on the exact method, and `kappa` is a `ValueError` on the
/// local one (it reads only `frozen_core` and `auxbasis`).
///
/// Local runs only: `att_operator="erfc"|"terfc"` runs the local MP2 of the
/// ATTENUATED correlation (SCF stays full Coulomb) with the CLI's
/// `[mp2]` keys of the same names: `omega` (A^-1, erfc, default 0.420),
/// `att_r0` (A, terfc, default 1.05) and `att_omega` (A^-1, terfc only: the
/// decoupled seam sharpness, `Operator::terfc_with_omega`; omitted = the
/// linked 1/(r0*sqrt2)). Omitted `att_operator` is the Coulomb operator.
/// terfc needs `FERRIC_TERF_TABLE_DIR`; with `integral_direct=True` it takes
/// the same `schwarz_skip` cut as the other operators (Schwarz factors from
/// the terfc engine's own diagonals; `schwarz_skip=0.0` evaluates every
/// triple). These four are a
/// `ValueError` on the exact method (use `run_terfc_rimp2` /
/// `run_attenuated_rimp2` there).
///
/// The reference follows the molecule's multiplicity: RHF + closed-shell
/// RI-MP2 for a singlet, UHF + unrestricted RI-MP2 (UMP2, as PySCF's
/// `mp.MP2(uhf)`) for multiplicity > 1. `result.reference` says which.
/// `kappa` (kappa-regularized MP2) exists for the closed-shell kernel only
/// and raises ValueError on an open-shell molecule rather than being
/// silently dropped.
///
/// `auxbasis` is the RI auxiliary basis (e.g. a bundled `*-ri`/`*-rifit` set
/// such as `"cc-pvdz-ri"` for orbital basis `"cc-pvdz"`, or `"def2-svp-rifit"`
/// for `"def2-svp"`) — NOT the SCF's own `df_j_aux`/`df_k_aux`. See
/// the quickstart (site/src/using/quickstart.md) for a worked pairing.
///
/// `frozen_core` (default 0) excludes that many lowest-energy occupied
/// orbitals from the correlation treatment.
///
/// `scf_jk` sets how the reference SCF builds J/K: "exact" (default,
/// four-centre), "rijk" (RI-J + RI-K) or "rijcosx" (RI-J + COSX K), fitted
/// on `jk_auxbasis` (a bundled JK-fit name, default "def2-universal-jkfit";
/// never `auxbasis`). Strict: an unknown mode, `jk_auxbasis` with "exact",
/// or a `k_builder` the mode overrides raise ValueError. Echoed as
/// `result.scf_jk` / `result.jk_auxbasis`.
///
/// Raises if the internal SCF does not converge.
///
/// Returns a [`RiMp2Result`](PyRiMp2Result) with `total_energy` (reference +
/// MP2 correlation), `rhf_energy` (the reference SCF energy, RHF or UHF),
/// `mp2_corr` (the correlation energy alone, always negative), `reference`
/// ("RHF" or "UHF") and `local` (`None` for the exact method, else the
/// local model dict).
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None, memory_budget_gb=None, kappa=None, local=None, eps=None, compute_reference=None, integral_direct=None, aux_radius=None, virt_radius=None, ao_tail=None, schwarz_skip=None, batch_merge=None, gate_cal=None, virt_schwarz_kappa=None, att_operator=None, omega=None, att_r0=None, att_omega=None, scf_jk=None, jk_auxbasis=None, scf_ri_j_precision=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, virtual_space=None, virtual_tau=None, virtual_tau_loose=None, virtual_solve_radius=None, virtual_boys_tau=None, with_gradient=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_rimp2(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    kappa: Option<f64>,
    local: Option<String>,
    eps: Option<f64>,
    compute_reference: Option<bool>,
    integral_direct: Option<bool>,
    aux_radius: Option<f64>,
    virt_radius: Option<f64>,
    ao_tail: Option<f64>,
    schwarz_skip: Option<f64>,
    batch_merge: Option<usize>,
    gate_cal: Option<f64>,
    virt_schwarz_kappa: Option<f64>,
    att_operator: Option<String>,
    omega: Option<f64>,
    att_r0: Option<f64>,
    att_omega: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    scf_ri_j_precision: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    virtual_space: Option<String>,
    virtual_tau: Option<f64>,
    virtual_tau_loose: Option<f64>,
    virtual_solve_radius: Option<f64>,
    virtual_boys_tau: Option<f64>,
    with_gradient: bool,
) -> PyResult<PyRiMp2Result> {
    let (att_keys, sets_att) = att_keys_from_kwargs(att_operator, omega, att_r0, att_omega);
    let (rhf_config, jk) = mp2_exact_default_config(
        "run_rimp2",
        k_builder,
        memory_budget_gb,
        scf_jk,
        jk_auxbasis,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
        scf_ri_j_precision,
    )?;
    let model = local_model_from_kwargs(
        "run_rimp2",
        ferric_cli::LocalCfg {
            scheme: local,
            eps,
            eps_sweep: None,
            reference: compute_reference,
            integral_direct,
            aux_radius,
            virt_radius,
            ao_tail,
            schwarz_skip,
            batch_merge,
            gate_cal,
            virt_schwarz_kappa,
            virtual_space,
            virtual_tau,
            virtual_tau_loose,
            virtual_solve_radius,
            virtual_boys_tau,
        },
    )?;
    if with_gradient {
        refuse_rimp2_gradient(
            &mol.inner,
            model.is_some() || sets_att,
            kappa,
            &rhf_config,
            &jk,
        )?;
    }
    if let Some(model) = model {
        return run_rimp2_local(
            py,
            mol,
            basis_set,
            auxbasis,
            frozen_core,
            (&rhf_config, &jk),
            memory_budget_gb,
            kappa,
            &model,
            if sets_att { Some(&att_keys) } else { None },
        );
    }
    if sets_att {
        return Err(att_keys_on_exact_error());
    }
    run_rimp2_exact(
        py,
        mol,
        basis_set,
        auxbasis,
        frozen_core,
        memory_budget_gb,
        kappa,
        (rhf_config, jk),
        with_gradient,
    )
}

/// `run_rimp2(with_gradient=True)` covers what the analytic RI-MP2 gradient
/// (`ferric_mp2::gradient::rimp2_gradient_analytical`) differentiates: the
/// exact closed-shell RI-MP2 on an RHF reference with exact four-centre J/K.
/// Everything else is a `ValueError` before any integral is computed.
pub(crate) fn refuse_rimp2_gradient(
    mol: &Molecule,
    local_or_attenuated: bool,
    kappa: Option<f64>,
    rhf_config: &RhfConfig,
    jk: &ferric_scf::scf_jk::ScfJk,
) -> PyResult<()> {
    let refuse = |what: String| {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "run_rimp2: with_gradient=True {what}"
        )))
    };
    if mol.multiplicity > 1 {
        return refuse(format!(
            "is closed-shell only (there is no unrestricted RI-MP2 gradient), but the \
             molecule has multiplicity {}",
            mol.multiplicity
        ));
    }
    if local_or_attenuated {
        return refuse(
            "differentiates the exact Coulomb RI-MP2 only: remove local= and the \
             att_operator keys"
                .into(),
        );
    }
    if kappa.is_some() {
        return refuse("does not differentiate kappa-regularized MP2: remove kappa".into());
    }
    if jk.is_fitted() || rhf_config.k_builder.as_deref() == Some("cosx") {
        return refuse(format!(
            "needs the exact four-centre J/K reference its gradient differentiates; got the \
             SCF J/K \"{}\". Use scf_jk=\"exact\" and an exact k_builder",
            jk.label()
        ));
    }
    Ok(())
}

/// `run_rimp2`'s exact (canonical RI-MP2) path: RHF + RI-MP2 for a singlet,
/// UHF + unrestricted RI-MP2 for multiplicity > 1.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_rimp2_exact(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
    memory_budget_gb: Option<f64>,
    kappa: Option<f64>,
    (rhf_config, jk): (RhfConfig, ferric_scf::scf_jk::ScfJk),
    with_gradient: bool,
) -> PyResult<PyRiMp2Result> {
    let emol = mol.inner.clone();
    let ebasis = basis_set.inner.clone();
    let eaux = auxbasis.inner.clone();
    let mp2_budget = budget_bytes_from_gb(memory_budget_gb);
    let open_shell = emol.multiplicity > 1;
    if open_shell && kappa.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "run_rimp2: kappa is not supported for an open-shell molecule (multiplicity {}): \
             kappa-regularization exists for the closed-shell kernel only. Omit kappa for \
             plain unrestricted RI-MP2.",
            emol.multiplicity
        )));
    }
    let mp2_config = RiMp2Config {
        frozen_core: frozen_core.unwrap_or(0),
        memory_budget_bytes: mp2_budget,
        kappa,
        ..Default::default()
    };
    // Release the GIL for the SCF + RI-MP2 compute. All captured values are
    // owned clones/locals built above, so nothing Python-borrowed crosses
    // the closure boundary.
    let r = py
        .allow_threads(|| {
            rimp2_energies(
                &emol,
                &ebasis,
                &eaux,
                &rhf_config,
                &mp2_config,
                with_gradient,
            )
        })
        .map_err(make_err)?;
    Ok(PyRiMp2Result {
        total_energy: r.total_energy,
        rhf_energy: r.scf_energy,
        mp2_corr: r.mp2_corr,
        reference: if open_shell { "UHF" } else { "RHF" }.to_string(),
        local: None,
        scf_jk: jk.mode.as_str().to_string(),
        jk_label: jk.label(),
        jk_auxbasis: jk.aux,
        precision_rec: r.precision,
        scf_precision_rec: r.scf_precision,
        certificate_data: r.certificate,
        gradient_data: r.gradient,
    })
}

/// The reference-SCF config of every MP2-family entry point: `k_builder`, the
/// memory budget, and `scf_jk` / `jk_auxbasis` resolved by the ONE shared
/// rule (`ferric_scf::scf_jk::ScfJk`), so "rijk" means the same `RhfConfig`
/// in every function. `default` is the entry point's historical J/K mode,
/// which keeps an unset `scf_jk` bit-identical to before. Errors are
/// `ValueError` naming `who`, raised before any integral is computed.
pub(crate) fn mp2_reference_config(
    who: &str,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    default: ferric_scf::scf_jk::ScfJkMode,
    conv: ScfConv,
    ri_j_precision: Option<&str>,
) -> PyResult<(RhfConfig, ferric_scf::scf_jk::ScfJk)> {
    let jk = ferric_scf::scf_jk::ScfJk::resolve(
        scf_jk,
        jk_auxbasis,
        k_builder,
        default,
        ferric_scf::scf_jk::ScfJkKeys::PYTHON,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{who}: {e}")))?;
    if let Some(w) = jk.aux_warning(ferric_scf::scf_jk::ScfJkKeys::PYTHON) {
        eprintln!("[ferric] warning: {who}: {w}");
    }
    let mut cfg = rhf_config_budgeted(k_builder, budget_bytes_from_gb(memory_budget_gb));
    jk.apply(&mut cfg);
    conv.apply(&mut cfg);
    cfg.ri_j_precision = parse_ri_j_precision(ri_j_precision)?;
    Ok((cfg, jk))
}

/// [`mp2_reference_config`] with the exact four-centre default every
/// MP2-family entry point except `run_rs_mp2_rpa` has always used.
pub(crate) fn mp2_exact_default_config(
    who: &str,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    conv: ScfConv,
    ri_j_precision: Option<&str>,
) -> PyResult<(RhfConfig, ferric_scf::scf_jk::ScfJk)> {
    use ferric_scf::scf_jk::ScfJkMode;
    mp2_reference_config(
        who,
        k_builder,
        memory_budget_gb,
        scf_jk,
        jk_auxbasis,
        ScfJkMode::Exact,
        conv,
        ri_j_precision,
    )
}

/// Solve the closed-shell Coulomb RHF reference, building its Schwarz table.
pub(crate) fn converged_rhf(
    mol: &Molecule,
    prep: &PreparedBasis,
    config: &RhfConfig,
) -> PyResult<ferric_scf::result::ScfResult> {
    let bounds = SchwarzBounds::compute(Operator::coulomb(), prep).map_err(make_err)?;
    converged_rhf_with_bounds(mol, prep, &bounds, config)
}

/// Solve the closed-shell Coulomb RHF reference every MP2-family entry point
/// shares, erroring on non-convergence (same rule as every `run_*` driver).
pub(crate) fn converged_rhf_with_bounds(
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
) -> PyResult<ferric_scf::result::ScfResult> {
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, mol, prep, Operator::coulomb(), bounds, config).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        }));
    }
    Ok(rhf)
}

/// The exact-J/K RHF reference of the local dRPA / LinLCCD paths.
pub(crate) fn local_rhf_reference(
    mol: &Molecule,
    prep: &PreparedBasis,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    conv: ScfConv,
) -> PyResult<ferric_scf::result::ScfResult> {
    let config = rhf_config_conv(k_builder, budget_bytes_from_gb(memory_budget_gb), conv);
    converged_rhf(mol, prep, &config)
}

/// `run_rimp2(local="amplitude-threshold", ...)`: the amplitude-threshold
/// local MP2 (`ferric_mp2::lmp2_amplitude`), or the integral-direct one
/// (`ferric_mp2::lmp2_direct`) with `integral_direct=True`. Closed-shell.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_rimp2_local(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
    (rhf_config, jk): (&RhfConfig, &ferric_scf::scf_jk::ScfJk),
    memory_budget_gb: Option<f64>,
    kappa: Option<f64>,
    model: &ferric_cli::LocalModel,
    att_keys: Option<&ferric_cli::Mp2Cfg>,
) -> PyResult<PyRiMp2Result> {
    use ferric_mp2::lmp2_amplitude::{amplitude_lmp2, AmplitudeLmp2Config};
    use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig};
    if mol.inner.multiplicity > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "run_rimp2: local=\"amplitude-threshold\" is closed-shell only (it localizes a \
             single restricted occupied space), but the molecule has multiplicity {}. Omit \
             local= for the exact unrestricted RI-MP2.",
            mol.inner.multiplicity
        )));
    }
    if let Some(k) = kappa {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "run_rimp2: kappa = {k} is not read by the local (amplitude-threshold) MP2; \
             kappa-regularized MP2 exists for the exact rimp2 only. Omit kappa or local=."
        )));
    }
    let op = local_correlation_operator(att_keys, model)?;
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let rhf = converged_rhf(&mol.inner, &prep, rhf_config)?;
    let eps = model.eps[0];
    let lcfg = AmplitudeLmp2Config {
        eps,
        frozen_core: frozen_core.unwrap_or(0),
        eri3_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        pair_gate_cal: model.direct.as_ref().and_then(|d| d.gate_cal),
        compute_reference: model.reference,
        virtual_space: model.virtual_space.clone(),
        ..Default::default()
    };
    let dcfg = model.direct.as_ref().map(|d| DirectConfig {
        aux_radius_bohr: d.aux_radius,
        virt_radius_bohr: Some(d.virt_radius),
        ao_tail: d.ao_tail,
        schwarz_skip: d.schwarz_skip,
        batch_merge: d.batch_merge,
        virt_schwarz_kappa: d.virt_schwarz_kappa,
        ..Default::default()
    });
    let (r, st) = match &dcfg {
        None => (
            amplitude_lmp2(&mol.inner, &prep, &basis_set.inner, &dfbs, op, &rhf, &lcfg)
                .map_err(make_err)?,
            None,
        ),
        Some(dc) => {
            let (r, st) = amplitude_lmp2_direct(
                &mol.inner,
                &prep,
                &basis_set.inner,
                &dfbs,
                op,
                &rhf,
                &lcfg,
                dc,
            )
            .map_err(make_err)?;
            (r, Some(st))
        }
    };
    let cert = ferric_mp2::lmp2_certificate::Lmp2Certificate::new(
        &lcfg,
        dcfg.as_ref(),
        &r,
        &ferric_mp2::lmp2_certificate::SystemShape::new(&mol.inner, &basis_set.inner),
        &op,
    )
    .with_direct_stats(st.as_ref());
    let mut certificate_data = ferric_mp2::certificate::local_rimp2_certificate(
        &cert,
        &reference_certificate(&mol.inner, &rhf, rhf_config, &basis_set.inner),
    );
    if op.kind != ferric_integrals::operator::OperatorKind::Coulomb {
        certificate_data = certificate_data.with_approximation(
            "operator",
            ferric_mp2::lmp2_certificate::operator_label(&op),
        );
    }
    let d = local_dict(py, eps, r.keep_fraction, model.direct.is_some())?;
    set_lmp2_counters(&d, &r, model.reference)?;
    set_certificate(py, &d, &cert)?;
    for (k, v) in r.virtual_space.fields(&r.timings) {
        set_cert_item(py, &d, k, &v)?;
    }
    if let Some(st) = st {
        d.set_item("n_pairs_gated", r.n_pairs_gated)?;
        set_direct_stats(&d, &st)?;
    }
    Ok(PyRiMp2Result {
        total_energy: r.e_total,
        rhf_energy: rhf.energy,
        mp2_corr: r.e_corr,
        reference: "RHF".to_string(),
        local: Some(d.unbind()),
        scf_jk: jk.mode.as_str().to_string(),
        jk_label: jk.label(),
        jk_auxbasis: jk.aux.clone(),
        // The local model's energy is its own CPU f64 kernel, not
        // `rimp2-energy`: nothing in it can run mixed.
        precision_rec: ferric_core::gpu::provenance::PrecisionRecord::f64(),
        scf_precision_rec: rhf.precision.clone(),
        certificate_data,
        gradient_data: None,
    })
}

/// Build an `AmplitudeDrpaConfig` from the shared `run_drpa`/`run_drpa_scan`
/// kwargs. `diis`/`eps_rtol_factor` default ON at the BINDING level
/// (diis=8, eps_rtol_factor=0.1) — the Rust library default stays
/// None/off; these are the calibrated values from
/// wiki/amplitude-threshold-drpa.md's 2026-08-17 DIIS+eps-link session
/// (70-79 -> 8-11 iterations, energies within the measured subdominance
/// bound, ~10% of truncation error). `diis=0` maps to `None` (disables
/// DIIS); `eps_rtol_factor=0.0` maps to `None` (disables the eps-link,
/// which is also the eps=0 behavior regardless — see the Rust doc).
#[allow(clippy::too_many_arguments)]
pub(crate) fn drpa_config(
    eps: Option<f64>,
    frozen_core: Option<usize>,
    memory_budget_gb: Option<f64>,
    compute_reference: bool,
    diis: Option<usize>,
    eps_rtol_factor: Option<f64>,
) -> ferric_mp2::drpa_amplitude::AmplitudeDrpaConfig {
    use ferric_mp2::drpa_amplitude::AmplitudeDrpaConfig;
    let diis_subspace = diis.unwrap_or(8);
    let eps_rtol = eps_rtol_factor.unwrap_or(0.1);
    AmplitudeDrpaConfig {
        // The caller always resolves eps (0.0 for the exact method): there is
        // no default threshold.
        eps: eps.unwrap_or(0.0),
        frozen_core: frozen_core.unwrap_or(0),
        eri3_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        compute_reference,
        diis: if diis_subspace == 0 {
            None
        } else {
            Some(diis_subspace)
        },
        eps_rtol_factor: if eps_rtol == 0.0 {
            None
        } else {
            Some(eps_rtol)
        },
        ..Default::default()
    }
}

/// `e_corr_plasmon_canonical` is `None` unless the (opt-in) reference was
/// computed — the same Optional-float convention as the result classes'
/// `Option<f64>` getters, never a NaN.
pub(crate) fn drpa_result_to_dict(
    py: Python<'_>,
    r: &ferric_mp2::drpa_amplitude::AmplitudeDrpaResult,
    rhf_energy: f64,
    compute_reference: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("e_corr", r.e_corr)?;
    d.set_item(
        "e_corr_plasmon_canonical",
        compute_reference.then_some(r.e_corr_plasmon_canonical),
    )?;
    d.set_item("total_energy", r.e_total)?;
    d.set_item("rhf_energy", rhf_energy)?;
    d.set_item("keep_fraction", r.keep_fraction)?;
    d.set_item("pair_fraction", r.pair_fraction)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("relres", r.relres)?;
    d.set_item("converged", r.converged)?;
    Ok(d.into())
}

/// Direct RPA correlation (dRPA@HF) by the drCCD Riccati solve on localized
/// orbitals (proof notebook 12). Closed-shell.
///
/// EXACT by default: the Riccati solve with nothing truncated (eps = 0),
/// anchored to the canonical semicanonicalized plasmon formula. Riccati,
/// plasmon and full-rank PDEP (`run_pdep_rpa(..., trunc_thresh=0)`) are
/// algorithms for the same exact dRPA. The exact path holds an `no^3 nv^2`
/// ring-product plan, so a run that cannot fit the memory budget raises
/// `MemoryError` BEFORE the SCF and points at `run_pdep_rpa`.
///
/// `local="amplitude-threshold"` with `eps=` (required; no default) runs the
/// amplitude-threshold dRPA: a ~linear, one-sided threshold error (dRPA is
/// non-variational). `compute_reference=True` (local only) adds the canonical
/// plasmon reference (a dense eigensolve over a global B) as
/// `e_corr_plasmon_canonical`; otherwise that key is None.
///
/// `diis` (default 8): Pulay/DIIS subspace size accelerating the Riccati
/// fixed point; `diis=0` disables it. `eps_rtol_factor` (local only, default
/// 0.1) links the stopping tolerance to eps (effective rtol =
/// max(1e-12, eps_rtol_factor*eps)); `0.0` disables it. It is a no-op at
/// eps = 0, so it is a ValueError on the exact method. Both knobs change
/// which point on the same truncated-equation solution manifold is returned,
/// calibrated to stay within ~10% of eps's own truncation error
/// (wiki/amplitude-threshold-drpa.md, "Subdominance calibration").
///
/// Returns a dict: e_corr, e_corr_plasmon_canonical, total_energy,
/// rhf_energy, keep_fraction, pair_fraction, iterations, relres, converged,
/// and `local` (None for the exact method, else scheme/eps/keep_fraction/
/// integral_direct).
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None, memory_budget_gb=None, local=None, eps=None, compute_reference=None, diis=None, eps_rtol_factor=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_drpa(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    local: Option<String>,
    eps: Option<f64>,
    compute_reference: Option<bool>,
    diis: Option<usize>,
    eps_rtol_factor: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    use ferric_mp2::drpa_amplitude::{amplitude_drpa, exact_drpa_peak_bytes};
    let model = local_model_from_kwargs(
        "run_drpa",
        ferric_cli::LocalCfg {
            scheme: local,
            eps,
            reference: compute_reference,
            ..Default::default()
        },
    )?;
    if model.is_none() && eps_rtol_factor.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "run_drpa: eps_rtol_factor links the stopping tolerance to eps and is a no-op at \
             eps = 0, i.e. on the exact method; pass it only with local=\"amplitude-threshold\"",
        ));
    }
    let (eps, compute_reference) = match &model {
        None => (0.0, false),
        Some(m) => (m.eps[0], m.reference),
    };
    let fc = frozen_core.unwrap_or(0);
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    if model.is_none() {
        // Refuse an exact run that cannot fit, BEFORE the SCF.
        let nocc = (mol.inner.nelec() as usize) / 2;
        let no = nocc.saturating_sub(fc);
        let nv = prep.nbasis().saturating_sub(nocc);
        let diis_sub = match diis.unwrap_or(8) {
            0 => None,
            n => Some(n),
        };
        let need = exact_drpa_peak_bytes(no, nv, diis_sub);
        let have = ferric_core::memory::resolve_budget(budget_bytes_from_gb(memory_budget_gb));
        if need > have.bytes {
            let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "run_drpa: exact dRPA needs ~{:.2} GiB for the Riccati solve (no = {no}, \
                 nv = {nv}; the no^3*nv^2 ring-product plan dominates), over the {:.2} GiB \
                 memory budget [source: {}]. run_pdep_rpa(..., trunc_thresh=0) computes the \
                 same exact dRPA energy (to its frequency-quadrature error) at far lower \
                 memory; local=\"amplitude-threshold\" with eps= is the local approximation.",
                gib(need),
                gib(have.bytes),
                have.source.label()
            )));
        }
    }
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let rhf = local_rhf_reference(
        &mol.inner,
        &prep,
        k_builder,
        memory_budget_gb,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    )?;
    let cfg = drpa_config(
        Some(eps),
        frozen_core,
        memory_budget_gb,
        compute_reference,
        diis,
        eps_rtol_factor,
    );
    let r = amplitude_drpa(
        &mol.inner,
        &prep,
        &basis_set.inner,
        &dfbs,
        Operator::coulomb(),
        &rhf,
        &cfg,
    )
    .map_err(make_err)?;
    let d = drpa_result_to_dict(py, &r, rhf.energy, compute_reference)?;
    let local_v = match &model {
        None => None,
        Some(_) => Some(local_dict(py, eps, r.keep_fraction, false)?),
    };
    let certificate = ferric_mp2::certificate::drpa_certificate(
        &reference_certificate(
            &mol.inner,
            &rhf,
            &rhf_config_conv(
                k_builder,
                budget_bytes_from_gb(memory_budget_gb),
                ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
            ),
            &basis_set.inner,
        ),
        &auxbasis.inner.name,
        fc,
        &Operator::coulomb(),
        model.as_ref().map(|_| (eps, r.keep_fraction)),
        r.converged,
        r.iterations,
        r.relres,
    );
    d.bind(py).set_item("local", local_v)?;
    d.bind(py)
        .set_item("certificate", certificate_dict(py, &certificate)?)?;
    Ok(d)
}

/// The amplitude-threshold (LOCAL) dRPA over a LIST of eps values — an
/// eps scan of the local scheme by definition, so it takes no `local=`
/// kwarg; each point is `run_drpa(..., local="amplitude-threshold", eps=e)`.
/// Reuses ONE
/// RHF solve + ONE localized-basis assembly (RHF, Boys localization,
/// VV-HV, Fock blocks, unwhitened RI B — everything eps-INDEPENDENT)
/// across every point; only the eps-dependent ragged assembly + Riccati
/// solve reruns per eps. Returns a list of dicts, each with the SAME keys
/// as `run_drpa`, plus `wall_s` (that point's ragged-assembly+solve wall
/// time) and `prefix_wall_s` (the ONE shared RHF+localization+assembly
/// wall time, repeated on every dict for convenience — it is not
/// re-paid per point). Per-point energies are byte-identical (<1e-12) to
/// calling `run_drpa` separately at each eps with the same kwargs.
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, eps_list, frozen_core=None, k_builder=None, memory_budget_gb=None, compute_reference=None, diis=None, eps_rtol_factor=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_drpa_scan(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    eps_list: Vec<f64>,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    compute_reference: Option<bool>,
    diis: Option<usize>,
    eps_rtol_factor: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<Py<pyo3::types::PyList>> {
    use ferric_mp2::drpa_amplitude::amplitude_drpa_scan_timed;
    // The same eps_sweep rules as the CLI's [local] eps_sweep (non-empty,
    // finite, >= 0); the points are evaluated in the order given.
    local_model_from_kwargs(
        "run_drpa_scan",
        ferric_cli::LocalCfg {
            scheme: Some("amplitude-threshold".to_string()),
            eps_sweep: Some(eps_list.clone()),
            ..Default::default()
        },
    )?;
    // Opt-in: `None` (the default) and `False` both skip the N^5 reference.
    let compute_reference = compute_reference.unwrap_or(false);
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(
        &ctx,
        &mol.inner,
        &prep,
        op,
        &bounds,
        &rhf_config_conv(
            k_builder,
            budget_bytes_from_gb(memory_budget_gb),
            ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
        ),
    )
    .map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        }));
    }
    let base_cfg = drpa_config(
        None,
        frozen_core,
        memory_budget_gb,
        compute_reference,
        diis,
        eps_rtol_factor,
    );
    let (results, prefix_wall_s, per_point_walls) = amplitude_drpa_scan_timed(
        &mol.inner,
        &prep,
        &basis_set.inner,
        &dfbs,
        op,
        &rhf,
        &base_cfg,
        &eps_list,
    )
    .map_err(make_err)?;
    let out = pyo3::types::PyList::empty(py);
    let scf_certificate = reference_certificate(
        &mol.inner,
        &rhf,
        &rhf_config_budgeted(k_builder, budget_bytes_from_gb(memory_budget_gb)),
        &basis_set.inner,
    );
    for ((r, eps), wall_s) in results.iter().zip(&eps_list).zip(&per_point_walls) {
        let d = drpa_result_to_dict(py, r, rhf.energy, compute_reference)?;
        let d_ref = d.bind(py);
        d_ref.set_item("eps", *eps)?;
        d_ref.set_item("local", local_dict(py, *eps, r.keep_fraction, false)?)?;
        // Same builder as `run_drpa(local=..., eps=...)` at this point.
        let certificate = ferric_mp2::certificate::drpa_certificate(
            &scf_certificate,
            &auxbasis.inner.name,
            frozen_core.unwrap_or(0),
            &Operator::coulomb(),
            Some((*eps, r.keep_fraction)),
            r.converged,
            r.iterations,
            r.relres,
        );
        d_ref.set_item("certificate", certificate_dict(py, &certificate)?)?;
        d_ref.set_item("wall_s", *wall_s)?;
        d_ref.set_item("prefix_wall_s", prefix_wall_s)?;
        out.append(d_ref)?;
    }
    Ok(out.into())
}

/// Linearized ladder CCD on an RHF reference (closed-shell). `variant`:
/// `"hh"` (default, LinLCCD(hh)), `"drivers-only"` (no ladder; reproduces
/// RI-MP2) or `"full"` (hh + pp ladders, CCD-like VVVV memory) — the CLI's
/// `[mp2] linlccd_variant` spellings, parsed by the same strict parser.
///
/// EXACT by default (`ferric_cc::linlccd::linlccd`, the CLI `linlccd`).
/// `local="amplitude-threshold"` with `eps=` (required; no default) runs the
/// amplitude-threshold LinLCCD in the localized basis; `eps=0` reproduces the
/// exact method of the same variant. `compute_reference=True` (local only)
/// also runs the exact LinLCCD and reports it as `local["e_corr_exact"]`.
///
/// Returns a dict: e_corr, total_energy, rhf_energy, variant, and `local`
/// (None for the exact method, else scheme/eps/keep_fraction/integral_direct
/// plus cg_iterations, converged and e_corr_exact).
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, variant=None, local=None, eps=None, compute_reference=None, frozen_core=None, k_builder=None, memory_budget_gb=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_linlccd(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    variant: Option<&str>,
    local: Option<String>,
    eps: Option<f64>,
    compute_reference: Option<bool>,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    use ferric_cc::linlccd::{linlccd, LadderVariant};
    use ferric_cc::linlccd_amplitude::{amplitude_linlccd, AmplitudeLinLccdConfig};
    let var = match variant {
        None => LadderVariant::Hh,
        Some(v) => LadderVariant::parse_config_str(v).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("run_linlccd: variant: {e}"))
        })?,
    };
    let model = local_model_from_kwargs(
        "run_linlccd",
        ferric_cli::LocalCfg {
            scheme: local,
            eps,
            reference: compute_reference,
            ..Default::default()
        },
    )?;
    if mol.inner.multiplicity > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "run_linlccd: closed-shell (RHF) only, but the molecule has multiplicity {}; \
             open-shell LinLCCD(hh) is library-only (ferric_cc::linlccd_u)",
            mol.inner.multiplicity
        )));
    }
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let rhf = local_rhf_reference(
        &mol.inner,
        &prep,
        k_builder,
        memory_budget_gb,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    )?;
    let op = Operator::coulomb();
    let fc = frozen_core.unwrap_or(0);
    let budget = budget_bytes_from_gb(memory_budget_gb);
    let exact = |var: LadderVariant| -> PyResult<f64> {
        let cc = linlccd(
            &mol.inner,
            &prep,
            &dfbs,
            op,
            &rhf,
            &ferric_cc::CcConfig {
                frozen_core: fc,
                memory_budget_bytes: budget,
                ..Default::default()
            },
            var,
        )
        .map_err(make_err)?;
        Ok(cc.correlation_energy)
    };
    let d = pyo3::types::PyDict::new(py);
    let mut local_cert = None;
    let mut cg_converged = true;
    let e_corr = match &model {
        None => {
            let e = exact(var)?;
            d.set_item("local", py.None())?;
            e
        }
        Some(m) => {
            let eps = m.eps[0];
            let r = amplitude_linlccd(
                &mol.inner,
                &prep,
                &basis_set.inner,
                &dfbs,
                op,
                &rhf,
                &AmplitudeLinLccdConfig {
                    eps,
                    frozen_core: fc,
                    eri3_budget_bytes: budget,
                    ..Default::default()
                },
                var,
            )
            .map_err(make_err)?;
            let l = local_dict(py, eps, r.keep_fraction, false)?;
            l.set_item("cg_iterations", r.cg_iterations)?;
            l.set_item("converged", r.cg_converged)?;
            local_cert = Some((eps, r.keep_fraction));
            cg_converged = r.cg_converged;
            let e_ref = if m.reference { Some(exact(var)?) } else { None };
            l.set_item("e_corr_exact", e_ref)?;
            d.set_item("local", l)?;
            r.e_corr
        }
    };
    d.set_item("e_corr", e_corr)?;
    d.set_item("total_energy", rhf.energy + e_corr)?;
    d.set_item("rhf_energy", rhf.energy)?;
    d.set_item("variant", var.as_str())?;
    let certificate = ferric_cc::certificate::linlccd_certificate(
        &reference_certificate(
            &mol.inner,
            &rhf,
            &rhf_config_conv(
                k_builder,
                budget,
                ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
            ),
            &basis_set.inner,
        ),
        &auxbasis.inner.name,
        fc,
        var,
        local_cert,
        cg_converged,
    );
    d.set_item("certificate", certificate_dict(py, &certificate)?)?;
    Ok(d.into())
}

/// Optimal (Baer/Kronik IP-based) tuning of the range-separation omega for
/// an RSH functional. Closed-shell neutral + doublet cation.
///
/// Returns a dict:
///
/// * `omega`, `j`, `converged` — the tuned omega (Bohr^-1), its Koopmans
///   residual J, and whether the golden-section search met `omega_tol`.
/// * `branch_warning` — `None`, or a string naming every omega whose cation
///   changed electronic state relative to its nearest evaluated neighbour.
///   When this is set the J curve was assembled from more than one state and
///   `omega` is not the tuned omega of either.
/// * `evals` — one dict per evaluation: `omega`, `eps_homo`, `ip`, `j`,
///   `e_neutral`, `e_cation`, `cation_iterations`, `neutral_iterations`,
///   `cation_s_squared`, `cation_spin_asymmetry`
///   (largest spin-population difference over symmetry-equivalent atoms; ~0
///   for a hole shared over equivalent centres, O(1) for a localized hole),
///   `seed` ("default" or "continued"), `seed_from_omega` (the omega continued
///   from, or `None`), and `branch_changed`.
///
/// `continuation` (default `False`) seeds each evaluation's neutral and cation
/// SCF from the converged orbitals of the NEAREST already-evaluated omega, so
/// a golden-section search — which visits omega out of order — keeps the
/// cation on one branch instead of re-solving it from the default guess at
/// every point. `False` reproduces independent per-omega solves exactly.
///
/// `branch_tol` (default 1e-7) is the largest change in cation spin
/// asymmetry still attributed to omega moving rather than to the cation
/// changing state; `0` or a negative value disables the check, in which case
/// `branch_changed` is `False` everywhere because nothing was CHECKED, not
/// because the states agreed.
///
/// `check_cation_stability` (default `True`) runs the UKS internal-stability
/// analysis on every cation. Each eval dict then carries `cation_lambda_min`
/// (lowest orbital-Hessian eigenvalue, Ha/rad^2, or `None`), `cation_stability`
/// (`"stable"`, `"unstable"`, `"marginal"`, `"indeterminate"`, `"not_analysed"`
/// or `"not_checked"`) and `cation_stability_skip` (the reason when not
/// analysed). A tuned omega whose cation is an internal SADDLE raises
/// `RuntimeError` naming omega and lambda_min: it is not a valid result. The
/// onset is a curvature change that no energy, <S^2> or population observable
/// can see. Functionals whose Hessian cannot be built (VV10, e.g. wB97X-V;
/// meta-GGA) are NOT analysed: every eval is `"not_analysed"`, a warning is
/// printed, `stability_warning` is set, and the returned omega is not
/// certified stable. `False` skips the analysis and reproduces the previous
/// results bit for bit.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, functional, omega_lo=None, omega_hi=None, omega_tol=None, max_evals=None, continuation=None, branch_tol=None, check_cation_stability=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tune_omega(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    functional: &str,
    omega_lo: Option<f64>,
    omega_hi: Option<f64>,
    omega_tol: Option<f64>,
    max_evals: Option<usize>,
    continuation: Option<bool>,
    branch_tol: Option<f64>,
    check_cation_stability: Option<bool>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    use ferric_scf::omega_tuning::{tune_omega as tune, OmegaTuneConfig};
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let defaults = OmegaTuneConfig::default();
    let cfg = OmegaTuneConfig {
        functional: functional.to_string(),
        omega_lo: omega_lo.unwrap_or(0.1),
        omega_hi: omega_hi.unwrap_or(1.0),
        omega_tol: omega_tol.unwrap_or(5e-3),
        max_evals: max_evals.unwrap_or(24),
        continuation: continuation.unwrap_or(defaults.continuation),
        // A non-positive tolerance is the documented "off" switch; anything
        // positive is taken literally rather than clamped, so a caller that
        // asks for an absurdly tight bar gets flags, not a silent default.
        branch_tol: match branch_tol {
            None => defaults.branch_tol,
            Some(t) if t > 0.0 => Some(t),
            Some(_) => None,
        },
        check_cation_stability: check_cation_stability.unwrap_or(defaults.check_cation_stability),
        ..defaults
    };
    let r = tune(&ctx, &mol.inner, &prep, &bounds, &cfg).map_err(make_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("omega", r.omega)?;
    d.set_item("j", r.j)?;
    d.set_item("converged", r.converged)?;
    d.set_item("branch_warning", r.branch_warning.clone())?;
    d.set_item("stability_warning", r.stability_warning.clone())?;
    let evals = pyo3::types::PyList::empty(py);
    for e in &r.evals {
        evals.append(omega_eval_dict(py, e)?)?;
    }
    d.set_item("evals", evals)?;
    Ok(d.into())
}

/// The cation-stability keys of an `OmegaEval` dict.
pub(crate) fn set_stability_items(
    item: &pyo3::Bound<'_, pyo3::types::PyDict>,
    cs: &ferric_scf::omega_tuning::CationStability,
) -> PyResult<()> {
    item.set_item("cation_stability", cs.label())?;
    item.set_item("cation_lambda_min", cs.lambda_min())?;
    item.set_item(
        "cation_stability_skip",
        match *cs {
            ferric_scf::omega_tuning::CationStability::NotAnalysed(sk) => Some(sk.reason()),
            _ => None,
        },
    )?;
    Ok(())
}

/// One `OmegaEval` as the dict `tune_omega` returns in its `evals` list.
pub(crate) fn omega_eval_dict<'py>(
    py: Python<'py>,
    e: &ferric_scf::omega_tuning::OmegaEval,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    use ferric_scf::omega_tuning::OmegaSeed;
    let item = pyo3::types::PyDict::new(py);
    item.set_item("omega", e.omega)?;
    item.set_item("eps_homo", e.eps_homo)?;
    item.set_item("ip", e.ip_delta_scf)?;
    item.set_item("j", e.j)?;
    item.set_item("e_neutral", e.e_neutral)?;
    item.set_item("e_cation", e.e_cation)?;
    item.set_item("cation_s_squared", e.cation_s_squared)?;
    item.set_item("cation_spin_asymmetry", e.cation_spin_asymmetry)?;
    item.set_item("cation_iterations", e.cation_iterations)?;
    item.set_item("neutral_iterations", e.neutral_iterations)?;
    item.set_item("branch_changed", e.branch_changed)?;
    let (seed, from) = match e.seed {
        OmegaSeed::Default => ("default", None),
        OmegaSeed::Continued { from_omega } => ("continued", Some(from_omega)),
    };
    set_stability_items(&item, &e.cation_stability)?;
    item.set_item("seed", seed)?;
    item.set_item("seed_from_omega", from)?;
    Ok(item)
}

// ── OO-RI-MP2 (orbital-optimized) ──

#[pyclass]
#[pyo3(name = "OoRiMp2Result")]
pub(crate) struct PyOoRiMp2Result {
    #[pyo3(get)]
    pub(crate) total_energy: f64,
    #[pyo3(get)]
    pub(crate) hf_energy: f64,
    #[pyo3(get)]
    pub(crate) mp2_corr: f64,
    #[pyo3(get)]
    pub(crate) converged: bool,
    #[pyo3(get)]
    pub(crate) iterations: usize,
    #[pyo3(get)]
    pub(crate) grad_norm: f64,
    /// How the reference SCF built J/K: "exact", "rijk" or "rijcosx".
    #[pyo3(get)]
    pub(crate) scf_jk: String,
    /// The printout's SCF J/K echo (`ScfJk::label`: names a non-default
    /// exchange builder on the exact path).
    pub(crate) jk_label: String,
    /// The SCF's JK-fitting basis (`None` for "exact").
    #[pyo3(get)]
    pub(crate) jk_auxbasis: Option<String>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
    /// `dE_total/dR` (natoms x 3, Hartree/Bohr) when solved with `with_gradient=True`.
    pub(crate) gradient_data: Option<Array2<f64>>,
}

#[pymethods]
impl PyOoRiMp2Result {
    /// Analytic nuclear gradient of `total_energy` as an (natoms, 3) array in
    /// Hartree/Bohr, rows in the Molecule's atom order; `None` unless the
    /// result came from `run_oo_rimp2(..., with_gradient=True)`.
    fn gradient<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.gradient_data
            .as_ref()
            .map(|g| PyArray2::from_array(py, g))
    }
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "OoRiMp2Result(total_energy={:.10}, converged={}, iterations={})",
            self.total_energy, self.converged, self.iterations,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "OO-RI-MP2 Total Energy: {:.10} Ha (HF: {:.10}, corr: {:.10}, converged: {}, {} iterations, grad_norm: {:.2e}) [SCF J/K: {}]",
            self.total_energy, self.hf_energy, self.mp2_corr, self.converged, self.iterations, self.grad_norm,
            self.jk_label,
        )
    }
}

/// Orbital-optimized RI-MP2: jointly minimizes E_HF + E_MP2 over orbital
/// rotations (level-shifted approximate Newton + DIIS + Cayley rotation).
/// Starts from a converged RHF reference (same convention as `run_rimp2`).
/// `max_iter`/`grad_conv`/`level_shift`/`diis_size` control the orbital
/// rotation loop; the rest of `OoRiMp2Config` (energy_conv,
/// use_diis) stays at its library default, matching the CLI's `oo-rimp2`
/// arm which only threads `frozen_core` + `memory_budget_bytes` through.
/// `oo_max_iter` is the orbital-optimization iteration cap; the bare
/// `max_iter` is a deprecated alias of it. `scf_max_iter`, `scf_energy_conv`,
/// `scf_density_conv` control the internal reference RHF (names and defaults of
/// `run_rhf`'s `max_iter`, `energy_conv`, `density_conv`; 200, 1e-3, 1e-6).
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis, frozen_core=None, k_builder=None,
    max_iter=None, grad_conv=None, level_shift=None, diis_size=None,
    memory_budget_gb=None, scf_jk=None, jk_auxbasis=None, scf_ri_j_precision=None,
    scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None,
    oo_max_iter=None, with_gradient=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_oo_rimp2(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    max_iter: Option<usize>,
    grad_conv: Option<f64>,
    level_shift: Option<f64>,
    diis_size: Option<usize>,
    memory_budget_gb: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    scf_ri_j_precision: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    oo_max_iter: Option<usize>,
    with_gradient: bool,
) -> PyResult<PyOoRiMp2Result> {
    let max_iter = alias_kwarg(
        py,
        "run_oo_rimp2",
        "max_iter",
        "oo_max_iter",
        max_iter,
        oo_max_iter,
    )?;
    let (rhf_config, jk) = mp2_exact_default_config(
        "run_oo_rimp2",
        k_builder,
        memory_budget_gb,
        scf_jk,
        jk_auxbasis,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
        scf_ri_j_precision,
    )?;
    // OO-MP2 rebuilds exact four-centre J/K at every orbital step (and takes
    // its HF energy from those builds), so a fitted reference would only
    // change the starting orbitals: refused, never echoed as if honoured.
    jk.require_exact(
        ferric_scf::scf_jk::ScfJkKeys::PYTHON,
        ferric_scf::scf_jk::OO_REBUILDS_EXACT_JK,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("run_oo_rimp2: {e}")))?;
    if with_gradient && frozen_core.unwrap_or(0) > 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "run_oo_rimp2: with_gradient=True needs frozen_core=0: the core-active and \
             core-virtual rotations are not optimised, so the zero-response OO-MP2 gradient is \
             not the derivative of the frozen-core energy",
        ));
    }
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let rhf = converged_rhf_with_bounds(&mol.inner, &prep, &bounds, &rhf_config)?;
    // Orbital-rotation loop knobs default to OoRiMp2Config::default() (same
    // library default the CLI's oo-rimp2 arm uses); frozen_core/memory_budget
    // follow the run_rimp2 convention.
    let default_cfg = OoRiMp2Config::default();
    let cfg = OoRiMp2Config {
        max_iter: max_iter.unwrap_or(default_cfg.max_iter),
        grad_conv: grad_conv.unwrap_or(default_cfg.grad_conv),
        level_shift: level_shift.unwrap_or(default_cfg.level_shift),
        diis_size: diis_size.unwrap_or(default_cfg.diis_size),
        frozen_core: frozen_core.unwrap_or(0),
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        ..default_cfg
    };
    // `run_oo_rimp2` has no point_charges/external_field kwargs today (unlike
    // run_rhf/run_uhf/run_optimize) — this Python entry point never builds an
    // ExternalPotential, so `ext` is always None here. Adding QM/MM support to
    // this binding is out of Lane F1's scope; this just keeps behavior
    // unchanged after `oo_ri_mp2`'s hcore-bug fix.
    let r = oo_ri_mp2(&mol.inner, &prep, &dfbs, op, &bounds, &rhf, &cfg, None).map_err(make_err)?;
    let gradient_data = with_gradient
        .then(|| oo_rimp2_gradient(py, &mol.inner, &prep, &dfbs, &bounds, &r))
        .transpose()?;
    let certificate_data = ferric_mp2::certificate::oo_rimp2_certificate(
        &reference_certificate(&mol.inner, &rhf, &rhf_config, &basis_set.inner),
        &auxbasis.inner.name,
        cfg.frozen_core,
        &op,
        r.converged,
        r.iterations,
        r.grad_norm,
    );
    Ok(PyOoRiMp2Result {
        total_energy: r.total_energy,
        hf_energy: r.hf_energy,
        mp2_corr: r.mp2_corr,
        converged: r.converged,
        iterations: r.iterations,
        grad_norm: r.grad_norm,
        scf_jk: jk.mode.as_str().to_string(),
        jk_label: jk.label(),
        jk_auxbasis: jk.aux,
        certificate_data,
        gradient_data,
    })
}

/// The analytic OO-RI-MP2 gradient of a converged orbital optimisation
/// (`ferric_mp2::oo_rimp2_gradient::oo_ri_mp2_gradient`, all electrons
/// correlated). It assumes a stationary point of the OO-MP2 energy, so an
/// unconverged loop raises; its accuracy follows `grad_conv`.
pub(crate) fn oo_rimp2_gradient(
    py: Python<'_>,
    mol: &Molecule,
    prep: &PreparedBasis,
    dfbs: &PreparedBasis,
    bounds: &SchwarzBounds,
    r: &ferric_mp2::oo_rimp2::OoRiMp2Result,
) -> PyResult<Array2<f64>> {
    if !r.converged {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "run_oo_rimp2: with_gradient=True: the orbital optimisation did not converge \
             ({} iterations, orbital gradient norm {:.2e}); the zero-response gradient holds only \
             at a stationary point. Raise oo_max_iter or loosen grad_conv",
            r.iterations, r.grad_norm
        )));
    }
    let op = Operator::coulomb();
    py.allow_threads(|| {
        ferric_mp2::oo_rimp2_gradient::oo_ri_mp2_gradient(mol, prep, dfbs, op, bounds, r, 0, None)
    })
    .map_err(make_err)
}

// ── MP3 (spin-orbital, via einsum!) ──

#[pyclass]
#[pyo3(name = "Mp3Result")]
pub(crate) struct PyMp3Result {
    #[pyo3(get)]
    pub(crate) e_hf: f64,
    #[pyo3(get)]
    pub(crate) e_mp2: f64,
    #[pyo3(get)]
    pub(crate) e_mp3: f64,
    #[pyo3(get)]
    pub(crate) e_corr: f64,
    #[pyo3(get)]
    pub(crate) e_total: f64,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PyMp3Result {
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "Mp3Result(e_total={:.10}, e_corr={:.10})",
            self.e_total, self.e_corr
        )
    }
    fn __str__(&self) -> String {
        format!(
            "MP3 Total Energy: {:.10} Ha (HF: {:.10}, MP2: {:.10}, MP3: {:.10})",
            self.e_total, self.e_hf, self.e_mp2, self.e_mp3,
        )
    }
}

/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None, memory_budget_gb=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_mp3(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyMp3Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let budget = budget_bytes_from_gb(memory_budget_gb);
    let rhf_config = rhf_config_conv(
        k_builder,
        budget,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        }));
    }
    // The VVVV size guard inside mp3_energy used to resolve the budget itself
    // with a hardcoded `None` (env/auto-detect only), so a caller's budget
    // never reached it; it is now a parameter.
    let mp3 = mp3_energy(
        &mol.inner,
        &prep,
        &dfbs,
        op,
        &rhf,
        frozen_core.unwrap_or(0),
        budget,
    )
    .map_err(make_err)?;
    Ok(PyMp3Result {
        e_hf: mp3.e_hf,
        e_mp2: mp3.e_mp2,
        e_mp3: mp3.e_mp3,
        e_corr: mp3.e_corr,
        e_total: mp3.e_total,
        certificate_data: ferric_mp2::certificate::mp3_certificate(
            &reference_certificate(&mol.inner, &rhf, &rhf_config, &basis_set.inner),
            &auxbasis.inner.name,
            frozen_core.unwrap_or(0),
            &op,
        ),
    })
}

// ── Laplace RI-MP2 ──

#[pyclass]
#[pyo3(name = "LaplaceMp2Result")]
pub(crate) struct PyLaplaceMp2Result {
    #[pyo3(get)]
    pub(crate) total_energy: f64,
    #[pyo3(get)]
    pub(crate) mp2_corr: f64,
    #[pyo3(get)]
    pub(crate) e_os: f64,
    #[pyo3(get)]
    pub(crate) e_ss: f64,
    /// How the reference SCF built J/K: "exact", "rijk" or "rijcosx".
    #[pyo3(get)]
    pub(crate) scf_jk: String,
    /// The printout's SCF J/K echo (`ScfJk::label`: names a non-default
    /// exchange builder on the exact path).
    pub(crate) jk_label: String,
    /// The SCF's JK-fitting basis (`None` for "exact").
    #[pyo3(get)]
    pub(crate) jk_auxbasis: Option<String>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PyLaplaceMp2Result {
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "LaplaceMp2Result(total_energy={:.10}, mp2_corr={:.10})",
            self.total_energy, self.mp2_corr
        )
    }
    fn __str__(&self) -> String {
        format!(
            "Laplace RI-MP2 Total Energy: {:.10} Ha (corr: {:.10}, OS: {:.10}, SS: {:.10}) [SCF J/K: {}]",
            self.total_energy, self.mp2_corr, self.e_os, self.e_ss,
            self.jk_label,
        )
    }
}

/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, n_quad=None, frozen_core=None, k_builder=None, memory_budget_gb=None, scf_jk=None, jk_auxbasis=None, scf_ri_j_precision=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_laplace_mp2(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    n_quad: Option<usize>,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    scf_ri_j_precision: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyLaplaceMp2Result> {
    let (rhf_config, jk) = mp2_exact_default_config(
        "run_laplace_mp2",
        k_builder,
        memory_budget_gb,
        scf_jk,
        jk_auxbasis,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
        scf_ri_j_precision,
    )?;
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let budget = budget_bytes_from_gb(memory_budget_gb);
    let rhf = converged_rhf(&mol.inner, &prep, &rhf_config)?;
    let r = laplace_ri_mp2(
        &mol.inner,
        &prep,
        &dfbs,
        op,
        &rhf,
        n_quad.unwrap_or(7),
        frozen_core.unwrap_or(0),
        budget,
    )
    .map_err(make_err)?;
    let certificate_data = ferric_mp2::certificate::laplace_mp2_certificate(
        &reference_certificate(&mol.inner, &rhf, &rhf_config, &basis_set.inner),
        &auxbasis.inner.name,
        frozen_core.unwrap_or(0),
        &op,
        n_quad.unwrap_or(7),
    );
    Ok(PyLaplaceMp2Result {
        total_energy: r.total_energy,
        mp2_corr: r.mp2_corr,
        e_os: r.e_os,
        e_ss: r.e_ss,
        scf_jk: jk.mode.as_str().to_string(),
        jk_label: jk.label(),
        jk_auxbasis: jk.aux,
        certificate_data,
    })
}

// ── Laplace SOS-MP2 ──

#[pyclass]
#[pyo3(name = "SosMp2Result")]
pub(crate) struct PySosMp2Result {
    #[pyo3(get)]
    pub(crate) total_energy: f64,
    #[pyo3(get)]
    pub(crate) rhf_energy: f64,
    /// The SCALED correlation energy, `c_os * e_os`.
    #[pyo3(get)]
    pub(crate) sos_corr: f64,
    /// The UNSCALED opposite-spin correlation energy. Directly comparable
    /// against `run_rimp2(..)`'s opposite-spin component.
    #[pyo3(get)]
    pub(crate) e_os: f64,
    /// The `c_os` actually applied, echoed for provenance.
    #[pyo3(get)]
    pub(crate) c_os: f64,
    /// Quadrature points actually used.
    #[pyo3(get)]
    pub(crate) n_quad: usize,
    /// `"mo"` or `"ao"`, echoed so a caller can record which algebra ran.
    #[pyo3(get)]
    pub(crate) formulation: String,
    /// How the reference SCF built J/K: "exact", "rijk" or "rijcosx".
    #[pyo3(get)]
    pub(crate) scf_jk: String,
    /// The printout's SCF J/K echo (`ScfJk::label`: names a non-default
    /// exchange builder on the exact path).
    pub(crate) jk_label: String,
    /// The SCF's JK-fitting basis (`None` for "exact").
    #[pyo3(get)]
    pub(crate) jk_auxbasis: Option<String>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PySosMp2Result {
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "SosMp2Result(total_energy={:.10}, sos_corr={:.10}, formulation=\"{}\")",
            self.total_energy, self.sos_corr, self.formulation,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "SOS-MP2 Total Energy: {:.10} Ha (RHF: {:.10}, SOS corr: {:.10}, c_os={}, n_quad={}, formulation={}) [SCF J/K: {}]",
            self.total_energy, self.rhf_energy, self.sos_corr, self.c_os, self.n_quad, self.formulation,
            self.jk_label,
        )
    }
}

/// Laplace-transform SOS-MP2: `E = c_os * E_OS`.
///
/// `formulation` selects the algebra:
///   - `"mo"` (default) — τ-weighted `(P|ia)` amplitudes.
///   - `"ao"` — occupied/virtual pseudo-densities. EXACT: same quantity as
///     `"mo"`, agreeing to round-off. Dense here, so not a scaling win.
///   - `"ao-sparse"` — Boys-localized, domain-restricted AO. The one
///     APPROXIMATE variant: it discards AO pairs outside every orbital domain,
///     converging to `"ao"` as `domain_cutoff_bohr` grows. Requires that
///     argument; passing it with `"mo"`/`"ao"` raises rather than being ignored.
///     The cutoff changes the energy only, not the cost.
///   - `"local"` — Boys LMOs x VV-HV localized virtuals; each LMO keeps the
///     virtuals within `domain_cutoff_bohr` (required). APPROXIMATE; the radius
///     sets the size of the dense Gram GEMM, so it changes cost and memory.
///
/// An unrecognized value raises rather than silently running the default.
///
/// There is deliberately no `c_ss`: SOS-MP2 *is* the `c_ss = 0` limit, which is
/// exactly what lets the Laplace denominator factorize. `c_os = 1.0` recovers
/// the bare opposite-spin MP2 energy (the tests' hard internal reference).
///
/// `memory_budget_gb` caps the resident 3-index tensors, same meaning as
/// elsewhere in this module; the AO path fails fast with a clear error rather
/// than overshooting it.
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, c_os=None, n_quad=None, frozen_core=None,
                    formulation=None, domain_cutoff_bohr=None, k_builder=None,
                    memory_budget_gb=None, scf_jk=None, jk_auxbasis=None, scf_ri_j_precision=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_laplace_sos_mp2(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    c_os: Option<f64>,
    n_quad: Option<usize>,
    frozen_core: Option<usize>,
    formulation: Option<&str>,
    domain_cutoff_bohr: Option<f64>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    scf_ri_j_precision: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PySosMp2Result> {
    let (rhf_config, jk) = mp2_exact_default_config(
        "run_laplace_sos_mp2",
        k_builder,
        memory_budget_gb,
        scf_jk,
        jk_auxbasis,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
        scf_ri_j_precision,
    )?;
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let rhf = converged_rhf(&mol.inner, &prep, &rhf_config)?;
    let form =
        SosFormulation::parse_config_str(formulation, domain_cutoff_bohr).map_err(make_err)?;
    let cfg = SosMp2Config {
        c_os: c_os.unwrap_or(1.3),
        frozen_core: frozen_core.unwrap_or(0),
        n_quad: n_quad.unwrap_or(7),
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        domain_cutoff_bohr,
    };
    let r = laplace_sos_mp2_with_basis(
        &mol.inner,
        &prep,
        &basis_set.inner,
        &dfbs,
        op,
        &rhf,
        &cfg,
        form,
    )
    .map_err(make_err)?;
    let certificate_data = ferric_mp2::certificate::sos_mp2_certificate(
        &reference_certificate(&mol.inner, &rhf, &rhf_config, &basis_set.inner),
        &auxbasis.inner.name,
        &cfg,
        form,
        &op,
    );
    Ok(PySosMp2Result {
        total_energy: r.total_energy,
        rhf_energy: rhf.energy,
        sos_corr: r.sos_corr,
        e_os: r.e_os,
        c_os: r.c_os,
        n_quad: r.n_quad,
        formulation: match form {
            SosFormulation::Mo => "mo".to_string(),
            SosFormulation::Ao => "ao".to_string(),
            SosFormulation::AoSparse(_) => "ao-sparse".to_string(),
            SosFormulation::Local(_) => "local".to_string(),
        },
        scf_jk: jk.mode.as_str().to_string(),
        jk_label: jk.label(),
        jk_auxbasis: jk.aux,
        certificate_data,
    })
}

// ── Attenuated RI-MP2 ──

#[pyclass]
#[pyo3(name = "AttenuatedMp2Result")]
pub(crate) struct PyAttenuatedMp2Result {
    #[pyo3(get)]
    pub(crate) total_energy: f64,
    #[pyo3(get)]
    pub(crate) rhf_energy: f64,
    #[pyo3(get)]
    pub(crate) mp2_corr: f64,
    #[pyo3(get)]
    pub(crate) e_os: f64,
    #[pyo3(get)]
    pub(crate) e_ss: f64,
    /// How the reference SCF built J/K: "exact", "rijk" or "rijcosx".
    #[pyo3(get)]
    pub(crate) scf_jk: String,
    /// The printout's SCF J/K echo (`ScfJk::label`: names a non-default
    /// exchange builder on the exact path).
    pub(crate) jk_label: String,
    /// The SCF's JK-fitting basis (`None` for "exact").
    #[pyo3(get)]
    pub(crate) jk_auxbasis: Option<String>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PyAttenuatedMp2Result {
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "AttenuatedMp2Result(total_energy={:.10}, mp2_corr={:.10})",
            self.total_energy, self.mp2_corr
        )
    }
    fn __str__(&self) -> String {
        format!(
            "Attenuated RI-MP2 Total Energy: {:.10} Ha (RHF: {:.10}, corr: {:.10}, OS: {:.10}, SS: {:.10}) [SCF J/K: {}]",
            self.total_energy, self.rhf_energy, self.mp2_corr, self.e_os, self.e_ss,
            self.jk_label,
        )
    }
}

/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, omega=None, frozen_core=None, k_builder=None, memory_budget_gb=None, scf_jk=None, jk_auxbasis=None, scf_ri_j_precision=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_attenuated_rimp2(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    omega: Option<f64>,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    scf_ri_j_precision: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyAttenuatedMp2Result> {
    let (rhf_config, jk) = mp2_exact_default_config(
        "run_attenuated_rimp2",
        k_builder,
        memory_budget_gb,
        scf_jk,
        jk_auxbasis,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
        scf_ri_j_precision,
    )?;
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let rhf = converged_rhf(&mol.inner, &prep, &rhf_config)?;
    // omega is supplied in Å⁻¹; convert to Bohr⁻¹ for the operator.
    let cfg = AttenuatedMp2Config {
        omega: omega.unwrap_or(0.420) * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
        scaling: 1.0,
        frozen_core: frozen_core.unwrap_or(0),
        screen_thresh: None,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
    };
    let r = attenuated_ri_mp2(&mol.inner, &prep, &dfbs, &rhf, &cfg).map_err(make_err)?;
    let certificate_data = ferric_mp2::certificate::att_rimp2_certificate(
        &reference_certificate(&mol.inner, &rhf, &rhf_config, &basis_set.inner),
        &auxbasis.inner.name,
        &cfg,
    );
    Ok(PyAttenuatedMp2Result {
        total_energy: r.total_energy,
        rhf_energy: rhf.energy,
        mp2_corr: r.mp2_corr,
        e_os: r.spin_components.e_os,
        e_ss: r.spin_components.e_ss,
        scf_jk: jk.mode.as_str().to_string(),
        jk_label: jk.label(),
        jk_auxbasis: jk.aux,
        certificate_data,
    })
}

// ── terfc-attenuated RI-MP2 (exact terfc via interpolation tables) ──

/// MP2(terfc): RI-MP2 with the EXACT tempered-erfc operator (Dutoi/Goldey
/// interpolation tables), a single cutoff `r0` (Å). SCF stays full-Coulomb; only
/// the MP2 correlation is attenuated. Requires the terfc tables on disk
/// (FERRIC_TERF_TABLE_DIR). Paper aDZ-optimal r0 = 1.05 Å.
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, r0=None, frozen_core=None, k_builder=None, memory_budget_gb=None, scf_jk=None, jk_auxbasis=None, scf_ri_j_precision=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_terfc_rimp2(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    r0: Option<f64>,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    scf_ri_j_precision: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyRiMp2Result> {
    let (rhf_config, jk) = mp2_exact_default_config(
        "run_terfc_rimp2",
        k_builder,
        memory_budget_gb,
        scf_jk,
        jk_auxbasis,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
        scf_ri_j_precision,
    )?;
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let rhf = converged_rhf(&mol.inner, &prep, &rhf_config)?;
    // r0 supplied in Å; convert to Bohr for the operator.
    let r0_bohr = r0.unwrap_or(1.05) * ANGSTROM_TO_BOHR;
    let terfc_op = Operator::terfc(r0_bohr);
    let mp2 = ri_mp2(
        &mol.inner,
        &prep,
        &dfbs,
        terfc_op,
        &rhf,
        &RiMp2Config {
            frozen_core: frozen_core.unwrap_or(0),
            memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
            ..Default::default()
        },
    )
    .map_err(make_err)?;
    let certificate_data = ferric_mp2::certificate::ri_correlation_certificate(
        "att-rimp2",
        &reference_certificate(&mol.inner, &rhf, &rhf_config, &basis_set.inner),
        &auxbasis.inner.name,
        frozen_core.unwrap_or(0),
        &terfc_op,
    )
    .with_precision(&mp2.precision);
    Ok(PyRiMp2Result {
        total_energy: mp2.total_energy,
        rhf_energy: rhf.energy,
        mp2_corr: mp2.mp2_corr,
        reference: "RHF".to_string(),
        local: None,
        scf_jk: jk.mode.as_str().to_string(),
        jk_label: jk.label(),
        jk_auxbasis: jk.aux,
        precision_rec: mp2.precision,
        scf_precision_rec: rhf.precision.clone(),
        certificate_data,
        gradient_data: None,
    })
}

// ── SCS-MP2 ──

#[pyclass]
#[pyo3(name = "ScsMp2Result")]
pub(crate) struct PyScsMp2Result {
    #[pyo3(get)]
    pub(crate) total_energy: f64,
    #[pyo3(get)]
    pub(crate) rhf_energy: f64,
    #[pyo3(get)]
    pub(crate) scs_corr: f64,
    #[pyo3(get)]
    pub(crate) e_os: f64,
    #[pyo3(get)]
    pub(crate) e_ss: f64,
    /// How the reference SCF built J/K: "exact", "rijk" or "rijcosx".
    #[pyo3(get)]
    pub(crate) scf_jk: String,
    /// The printout's SCF J/K echo (`ScfJk::label`: names a non-default
    /// exchange builder on the exact path).
    pub(crate) jk_label: String,
    /// The SCF's JK-fitting basis (`None` for "exact").
    #[pyo3(get)]
    pub(crate) jk_auxbasis: Option<String>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PyScsMp2Result {
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "ScsMp2Result(total_energy={:.10}, scs_corr={:.10})",
            self.total_energy, self.scs_corr
        )
    }
    fn __str__(&self) -> String {
        format!(
            "SCS-MP2 Total Energy: {:.10} Ha (RHF: {:.10}, SCS corr: {:.10}, OS: {:.10}, SS: {:.10}) [SCF J/K: {}]",
            self.total_energy, self.rhf_energy, self.scs_corr, self.e_os, self.e_ss,
            self.jk_label,
        )
    }
}

/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, c_os=None, c_ss=None, frozen_core=None, k_builder=None, memory_budget_gb=None, scf_jk=None, jk_auxbasis=None, scf_ri_j_precision=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_scs_mp2(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    c_os: Option<f64>,
    c_ss: Option<f64>,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    scf_ri_j_precision: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyScsMp2Result> {
    let (rhf_config, jk) = mp2_exact_default_config(
        "run_scs_mp2",
        k_builder,
        memory_budget_gb,
        scf_jk,
        jk_auxbasis,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
        scf_ri_j_precision,
    )?;
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let rhf = converged_rhf(&mol.inner, &prep, &rhf_config)?;
    let cfg = ScsMp2Config {
        c_os: c_os.unwrap_or(6.0 / 5.0),
        c_ss: c_ss.unwrap_or(1.0 / 3.0),
        frozen_core: frozen_core.unwrap_or(0),
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
    };
    let r = scs_mp2(&mol.inner, &prep, &dfbs, &rhf, &cfg).map_err(make_err)?;
    let certificate_data = ferric_mp2::certificate::scs_mp2_certificate(
        &reference_certificate(&mol.inner, &rhf, &rhf_config, &basis_set.inner),
        &auxbasis.inner.name,
        &cfg,
    );
    Ok(PyScsMp2Result {
        total_energy: r.total_energy,
        rhf_energy: rhf.energy,
        scs_corr: r.scs_corr,
        e_os: r.e_os,
        e_ss: r.e_ss,
        scf_jk: jk.mode.as_str().to_string(),
        jk_label: jk.label(),
        jk_auxbasis: jk.aux,
        certificate_data,
    })
}

/// SCS-MP2(2terfc): dual-attenuated SCS-MP2 (Goldey & Head-Gordon, J. Phys.
/// Chem. B 118, 6519 (2014)) using the EXACT terfc operator at two cutoffs `r0_bonded` < `r0_nonbonded`
/// (Å). E = c_OS·E_OS(r0_1) + c_SS·[E_SS(r0_2) − E_SS(r0_1)]. Requires the terfc
/// tables (FERRIC_TERF_TABLE_DIR). Paper defaults: r0=0.75/1.05 Å, c_OS=1.27, c_SS=4.05.
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, r0_bonded=None, r0_nonbonded=None, c_os=None, c_ss=None, frozen_core=None, k_builder=None, memory_budget_gb=None, scf_jk=None, jk_auxbasis=None, scf_ri_j_precision=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_scs_mp2_2terfc(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    r0_bonded: Option<f64>,
    r0_nonbonded: Option<f64>,
    c_os: Option<f64>,
    c_ss: Option<f64>,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    scf_ri_j_precision: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyScsMp2Result> {
    let (rhf_config, jk) = mp2_exact_default_config(
        "run_scs_mp2_2terfc",
        k_builder,
        memory_budget_gb,
        scf_jk,
        jk_auxbasis,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
        scf_ri_j_precision,
    )?;
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let rhf = converged_rhf(&mol.inner, &prep, &rhf_config)?;
    const ANG2BOHR: f64 = ANGSTROM_TO_BOHR;
    let cfg = ScsMp2TerfcConfig {
        r0_bonded: r0_bonded.unwrap_or(0.75) * ANG2BOHR,
        r0_nonbonded: r0_nonbonded.unwrap_or(1.05) * ANG2BOHR,
        c_os: c_os.unwrap_or(1.27),
        c_ss: c_ss.unwrap_or(4.05),
        frozen_core: frozen_core.unwrap_or(0),
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
    };
    if cfg.r0_nonbonded <= cfg.r0_bonded {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r0_nonbonded must be > r0_bonded",
        ));
    }
    let r = scs_mp2_2terfc(&mol.inner, &prep, &dfbs, &rhf, &cfg).map_err(make_err)?;
    let certificate_data = ferric_mp2::certificate::scs_mp2_2terfc_certificate(
        &reference_certificate(&mol.inner, &rhf, &rhf_config, &basis_set.inner),
        &auxbasis.inner.name,
        &cfg,
    );
    Ok(PyScsMp2Result {
        total_energy: r.total_energy,
        rhf_energy: rhf.energy,
        scs_corr: r.scs_corr,
        e_os: r.e_os,
        e_ss: r.e_ss,
        scf_jk: jk.mode.as_str().to_string(),
        jk_label: jk.label(),
        jk_auxbasis: jk.aux,
        certificate_data,
    })
}

// ── MP2-V (attenuated MP2 + Eq.-11-damped VV10, JCTC 11, 4159 (2015)) ──

#[pyclass]
#[pyo3(name = "Mp2VResult")]
pub(crate) struct PyMp2VResult {
    /// E_HF + E_c^attMP2 + E_nl^VV10.
    #[pyo3(get)]
    pub(crate) total_energy: f64,
    #[pyo3(get)]
    pub(crate) rhf_energy: f64,
    /// Attenuated MP2 correlation energy (terfc/erfc at r0, optionally
    /// decoupled omega).
    #[pyo3(get)]
    pub(crate) att_mp2_corr: f64,
    /// VV10 nonlocal correlation, damped per `vv10_damping` (Eq. 11 by
    /// default). NOTE: the damping makes this LESS negative, not more — the
    /// dispersion it supplies is a difference effect (dimer minus monomers).
    #[pyo3(get)]
    pub(crate) vv10_e_nl: f64,
    #[pyo3(get)]
    pub(crate) e_os: f64,
    #[pyo3(get)]
    pub(crate) e_ss: f64,
    /// Grid points in the VV10 nonlocal integration (tells a suspiciously
    /// small E_nl from a suspiciously small grid).
    #[pyo3(get)]
    pub(crate) n_nlc_points: usize,
    /// How the reference SCF built J/K: "exact", "rijk" or "rijcosx".
    #[pyo3(get)]
    pub(crate) scf_jk: String,
    /// The printout's SCF J/K echo (`ScfJk::label`: names a non-default
    /// exchange builder on the exact path).
    pub(crate) jk_label: String,
    /// The SCF's JK-fitting basis (`None` for "exact").
    #[pyo3(get)]
    pub(crate) jk_auxbasis: Option<String>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PyMp2VResult {
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "Mp2VResult(total_energy={:.10}, att_mp2_corr={:.10}, vv10_e_nl={:.10})",
            self.total_energy, self.att_mp2_corr, self.vv10_e_nl,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "MP2-V Total Energy: {:.10} Ha (RHF: {:.10}, att-MP2 corr: {:.10}, VV10 E_nl: {:.10}) [SCF J/K: {}]",
            self.total_energy, self.rhf_energy, self.att_mp2_corr, self.vv10_e_nl,
            self.jk_label,
        )
    }
}

/// MP2-V (Goldey/Belzunces/Head-Gordon, JCTC 11, 4159 (2015)):
/// E = E_HF + E_c^attMP2(r0[, omega]) + E_nl^VV10 damped by 1 − terfc(R; r0[, omega])².
///
/// Closed-shell RHF references only (the CLI `kind = "mp2-v"` also handles
/// open shell). Units at this boundary: `r0` in Å (default 1.00, the published
/// value), `omega` in Å⁻¹ (default None = the Dutoi curvature link
/// ω = 1/(r0√2), the published method; setting it decouples the terfc seam
/// sharpness and reaches BOTH halves of Eq. 11 in lockstep — unparameterized
/// until b is refit). `attenuator`: "terfc" (published; needs
/// FERRIC_TERF_TABLE_DIR) or "erfc" (table-free control; rejects `omega`).
/// `vv10_damping`: "terfc" (published Eq. 11) or "none" (bare VV10 —
/// double-counts short range, measurement-only). b defaults 11.0, c 0.0089.
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, r0=None, b=None, c=None, omega=None, attenuator=None, vv10_damping=None, frozen_core=None, k_builder=None, memory_budget_gb=None, scf_jk=None, jk_auxbasis=None, scf_ri_j_precision=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_mp2_v(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    r0: Option<f64>,
    b: Option<f64>,
    c: Option<f64>,
    omega: Option<f64>,
    attenuator: Option<&str>,
    vv10_damping: Option<&str>,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    scf_ri_j_precision: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyMp2VResult> {
    let (rhf_config, jk) = mp2_exact_default_config(
        "run_mp2_v",
        k_builder,
        memory_budget_gb,
        scf_jk,
        jk_auxbasis,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
        scf_ri_j_precision,
    )?;
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let rhf = converged_rhf(&mol.inner, &prep, &rhf_config)?;

    // Start from the published MP2-V(terfc, aTZ) parameterization and override
    // only what the caller set (same construction order as the CLI's
    // build_att_vv10_config: damping BEFORE r0, so from_r0_angstrom sees the
    // final damping variant and keeps Eq. 11's two halves in lockstep).
    let mut cfg = AttVv10Config::mp2_v_terfc_atz();
    cfg.attenuator = match attenuator.unwrap_or("terfc") {
        "terfc" => AttVv10Attenuator::Terfc,
        "erfc" => AttVv10Attenuator::Erfc,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown attenuator \"{other}\"; expected \"terfc\" (published) or \"erfc\" (control)"
        )))
        }
    };
    cfg.vv10_damping = match vv10_damping.unwrap_or("terfc") {
        "terfc" => ferric_dft::vv10::Vv10Damping::Terfc {
            r0_bohr: cfg.r0_bohr,
            // Seam sharpness derives from `cfg.omega` at evaluation time
            // (effective_vv10_damping) — never duplicated here.
            omega_bohr_inv: None,
        },
        "none" => ferric_dft::vv10::Vv10Damping::None,
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown vv10_damping \"{other}\"; expected \"terfc\" (published, Eq. 11) or \"none\" (bare VV10, double-counts short range)"
        ))),
    };
    if let Some(r0_ang) = r0 {
        if !r0_ang.is_finite() || r0_ang <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "r0 must be finite and > 0 (got {r0_ang} A)"
            )));
        }
        cfg = cfg.from_r0_angstrom(r0_ang);
    }
    if let Some(b) = b {
        cfg.vv10.b = b;
    }
    if let Some(c) = c {
        cfg.vv10.c = c;
    }
    // omega is supplied in Å⁻¹ (same boundary convention as run_rs_mp2_rpa's
    // terf_omega); Bohr⁻¹ internally. Validation (finite/positive, terfc-only)
    // is the library's, so the error text cannot drift from the CLI's.
    cfg.omega = omega.map(|w| w * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV);
    cfg.frozen_core = frozen_core.unwrap_or(0);
    cfg.memory_budget_bytes = budget_bytes_from_gb(memory_budget_gb);

    let r =
        att_mp2_vv10(&mol.inner, &prep, &basis_set.inner, &dfbs, &rhf, &cfg).map_err(make_err)?;
    let (e_os, e_ss) = match &r.spin_components {
        AttVv10SpinComponents::Restricted(s) => (s.e_os, s.e_ss),
        AttVv10SpinComponents::Unrestricted(_) => unreachable!("closed-shell entry point"),
    };
    let certificate_data = ferric_mp2::certificate::mp2_v_certificate(
        &reference_certificate(&mol.inner, &rhf, &rhf_config, &basis_set.inner),
        &auxbasis.inner.name,
        &cfg,
    );
    Ok(PyMp2VResult {
        total_energy: r.total,
        rhf_energy: r.e_hf,
        att_mp2_corr: r.e_c_att_mp2,
        vv10_e_nl: r.e_nl_vv10,
        e_os,
        e_ss,
        n_nlc_points: r.n_nlc_points,
        scf_jk: jk.mode.as_str().to_string(),
        jk_label: jk.label(),
        jk_auxbasis: jk.aux,
        certificate_data,
    })
}

/// MP2-V scan on ONE converged RHF reference: the attenuated-MP2 half for
/// every `(r0, omega)` arm and the damped VV10 half for every arm x `b`.
///
/// `arms` is a list of `(r0_angstrom, omega_inverse_angstrom_or_None)`; `None`
/// is the published linked width. `bs` is the list of VV10 `b` values. SCF is
/// run once. Returns a dict with `rhf_energy`, `n_nlc_points`, and `arms`: a
/// list (same order) of dicts `{r0, omega, att_mp2_corr, e_os, e_ss,
/// vv10_e_nl: [per b]}`. `total(arm, k) = rhf_energy + att_mp2_corr +
/// vv10_e_nl[k]`. Same units/semantics as `run_mp2_v`; closed shell, terfc
/// only. `nlc_grid=(n_radial, n_angular)` (default (50, 50), the ferric NLC
/// default). `df_j_aux`/`df_k_aux` default to def2-universal-jkfit (the A24
/// grid convention); "exact" selects four-centre J/K. `include_coulomb_mp2=True`
/// adds `mp2_coulomb_corr` (plain RI-MP2 correlation, same SCF/aux/frozen core).
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, arms, bs, c=None, frozen_core=None, nlc_grid=None, df_j_aux=None, df_k_aux=None, k_builder=None, memory_budget_gb=None, include_coulomb_mp2=false, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, scf_ri_j_precision=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_mp2_v_scan(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    arms: Vec<(f64, Option<f64>)>,
    bs: Vec<f64>,
    c: Option<f64>,
    frozen_core: Option<usize>,
    nlc_grid: Option<(usize, usize)>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    include_coulomb_mp2: bool,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    scf_ri_j_precision: Option<&str>,
) -> PyResult<Py<pyo3::types::PyDict>> {
    use pyo3::types::{PyDict, PyList};
    for (r0, _) in &arms {
        if !r0.is_finite() || *r0 <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "r0 must be finite and > 0 (got {r0} A)"
            )));
        }
    }
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let coul = Operator::coulomb();
    let bounds = SchwarzBounds::compute(coul, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let mut rcfg = rhf_config_conv(
        k_builder,
        budget_bytes_from_gb(memory_budget_gb),
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    rcfg.df_j_aux = resolve_df_aux(df_j_aux, "def2-universal-jkfit");
    rcfg.ri_j_precision = parse_ri_j_precision(scf_ri_j_precision)?;
    rcfg.df_k_aux = resolve_df_aux(df_k_aux, "def2-universal-jkfit");
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, coul, &bounds, &rcfg).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        }));
    }
    let mut base = AttVv10Config::mp2_v_terfc_atz();
    if let Some(c) = c {
        base.vv10.c = c;
    }
    base.frozen_core = frozen_core.unwrap_or(0);
    base.memory_budget_bytes = budget_bytes_from_gb(memory_budget_gb);
    if let Some((nr, na)) = nlc_grid {
        base.nlc_grid.n_radial = nr;
        base.nlc_grid.n_angular = na;
    }
    let scan_arms: Vec<AttVv10ScanArm> = arms
        .iter()
        .map(|(r0, w)| AttVv10ScanArm {
            r0_bohr: r0 * ferric_mp2::att_vv10::BOHR_PER_ANG,
            omega: w.map(|w| w * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV),
        })
        .collect();
    let r = att_mp2_vv10_scan(
        &mol.inner,
        &prep,
        &basis_set.inner,
        &dfbs,
        &rhf,
        &base,
        &scan_arms,
        &bs,
        include_coulomb_mp2,
    )
    .map_err(make_err)?;
    let out = PyDict::new(py);
    out.set_item("rhf_energy", r.e_hf)?;
    out.set_item("n_nlc_points", r.n_nlc_points)?;
    out.set_item("mp2_coulomb_corr", r.e_c_mp2_coulomb)?;
    let lst = PyList::empty(py);
    let scf_certificate = reference_certificate(&mol.inner, &rhf, &rcfg, &basis_set.inner);
    for ((arm, res), scan_arm) in arms.iter().zip(r.arms.iter()).zip(&scan_arms) {
        let d = PyDict::new(py);
        // The config this arm ran with, as `att_mp2_vv10_scan` builds it; the
        // same builder as `run_mp2_v`. The scanned `b` values change only the
        // VV10 energies of the arm, so one certificate covers its `b` list.
        let mut arm_cfg = base.clone();
        arm_cfg.r0_bohr = scan_arm.r0_bohr;
        arm_cfg.omega = scan_arm.omega;
        arm_cfg.vv10_damping = ferric_dft::vv10::Vv10Damping::Terfc {
            r0_bohr: scan_arm.r0_bohr,
            omega_bohr_inv: None,
        };
        let certificate = ferric_mp2::certificate::mp2_v_certificate(
            &scf_certificate,
            &auxbasis.inner.name,
            &arm_cfg,
        )
        .with_approximation("vv10-b-scan", format!("b in {bs:?}, c = {}", base.vv10.c));
        d.set_item("certificate", certificate_dict(py, &certificate)?)?;
        d.set_item("r0", arm.0)?;
        d.set_item("omega", arm.1)?;
        d.set_item("att_mp2_corr", res.e_c_att_mp2)?;
        d.set_item("e_os", res.e_os)?;
        d.set_item("e_ss", res.e_ss)?;
        d.set_item("vv10_e_nl", res.e_nl_vv10.clone())?;
        lst.append(d)?;
    }
    out.set_item("arms", lst)?;
    Ok(out.unbind())
}

/// RESEARCH: VV10 E_nl on the plain-HF density of `mol` for alternative pair-kernel
/// weights (benchmarks/a24-mp2v-refit Addendum A). `specs` = list of
/// `(kind, r0_angstrom, r0_times_omega_or_None, power)` with kind "none" (bare VV10),
/// "eq11" (1 - terfc^2, the production Eq. 11 factor) or "terf" (terf^power,
/// terf = 1 - terfc). `r0_times_omega=None` is the linked width. Returns one E_nl
/// (Ha) per spec. Closed shell, post-HF; no MP2 is run.
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, specs, b=11.0, c=0.0089, nlc_grid=None, df_j_aux=None, df_k_aux=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, scf_ri_j_precision=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_vv10_variants(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    specs: Vec<(String, f64, Option<f64>, i32)>,
    b: f64,
    c: f64,
    nlc_grid: Option<(usize, usize)>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    scf_ri_j_precision: Option<&str>,
) -> PyResult<Vec<f64>> {
    use ferric_dft::vv10::{Vv10Damping, Vv10Weight};
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let coul = Operator::coulomb();
    let bounds = SchwarzBounds::compute(coul, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let mut rcfg = rhf_config_conv(
        None,
        None,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    rcfg.df_j_aux = resolve_df_aux(df_j_aux, "def2-universal-jkfit");
    rcfg.ri_j_precision = parse_ri_j_precision(scf_ri_j_precision)?;
    rcfg.df_k_aux = resolve_df_aux(df_k_aux, "def2-universal-jkfit");
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, coul, &bounds, &rcfg).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        }));
    }
    let mut weights = Vec::new();
    for (kind, r0, rw, power) in &specs {
        let r0_bohr = r0 * ferric_mp2::att_vv10::BOHR_PER_ANG;
        let omega = rw.map(|x| x / r0_bohr);
        weights.push(match kind.as_str() {
            "none" => Vv10Weight::Damping(Vv10Damping::None),
            "eq11" => Vv10Weight::Damping(Vv10Damping::Terfc {
                r0_bohr,
                omega_bohr_inv: omega,
            }),
            "terf" => Vv10Weight::TerfPower {
                r0_bohr,
                omega_bohr_inv: omega,
                power: *power,
            },
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown kind \"{other}\"; expected none|eq11|terf"
                )))
            }
        });
    }
    let mut grid = ferric_dft::grid::AtomicGridConfig {
        n_radial: 50,
        n_angular: 50,
        prune: None,
        weight_cut: None,
    };
    if let Some((nr, na)) = nlc_grid {
        grid.n_radial = nr;
        grid.n_angular = na;
    }
    ferric_mp2::att_vv10::vv10_weighted_energies_on_density(
        &mol.inner,
        &basis_set.inner,
        rhf.density_total(),
        &ferric_dft::libxc::Vv10Params { c, b },
        &weights,
        &grid,
    )
    .map_err(make_err)
}
