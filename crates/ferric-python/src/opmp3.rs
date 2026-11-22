//! `run_opmp3`: orbital-partitioned Padé MP3 (`ferric_mp2::opmp3_driver`).

use super::*;
use ferric_mp2::local_opmp3::{local_opmp3_on_reference, LocalOpmp3Result};
use ferric_mp2::opmp3::WChannels;
use ferric_mp2::opmp3_driver::{
    opmp3_on_reference, opmp3_scf, opmp3_scf_config, Opmp3Config, Opmp3Integrals, Opmp3RefKind,
    Opmp3Result,
};

#[pyclass]
#[pyo3(name = "Opmp3Result")]
pub(crate) struct PyOpmp3Result {
    inner: Opmp3Result,
    certificate_data: ferric_core::certificate::Certificate,
    /// The local run's pattern and solves, and the canonical OP-MP3 when
    /// `compute_reference=True`; `None` for the exact method.
    local: Option<(LocalOpmp3Result, Option<Opmp3Result>)>,
}

/// `(model name, correlation energy)` of every model.
fn model_energies(r: &Opmp3Result) -> [(&'static str, f64); 6] {
    let m = &r.models;
    [
        ("mp2", m.mp2),
        ("mp3", m.mp3),
        ("p-mp3", m.p_mp3),
        ("op-mp3", m.op_mp3),
        ("sop-mp3", m.sop_mp3),
        ("ssd-opmp3", m.ssd_opmp3),
    ]
}

fn channels_dict<'py>(py: Python<'py>, w: &WChannels) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    for (s, name) in ["alpha", "beta"].iter().enumerate() {
        d.set_item(format!("{name}_ss"), PyArray2::from_array(py, &w.ss[s]))?;
        d.set_item(format!("{name}_os"), PyArray2::from_array(py, &w.os[s]))?;
    }
    Ok(d)
}

#[pymethods]
impl PyOpmp3Result {
    /// The reference SCF energy as converged (the KS energy for DFT).
    #[getter]
    fn e_scf(&self) -> f64 {
        self.inner.e_scf
    }
    /// HF energy of the reference density.
    #[getter]
    fn e_hf(&self) -> f64 {
        self.inner.e_hf
    }
    /// Non-Brillouin singles energy.
    #[getter]
    fn e_nbs(&self) -> f64 {
        self.inner.e_nbs
    }
    /// SSD-OPMP3 same-spin damping used.
    #[getter]
    fn x_ss(&self) -> f64 {
        self.inner.models.x_ss
    }
    /// Correlation energy of every model, keyed `mp2`, `mp3`, `p-mp3`,
    /// `op-mp3`, `sop-mp3`, `ssd-opmp3` (Hartree, without `E_HF` or `E_NBS`).
    #[getter]
    fn correlation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        for (k, e) in model_energies(&self.inner) {
            d.set_item(k, e)?;
        }
        Ok(d)
    }
    /// `E_HF + E_NBS + E_c(model)` for a model name of [`Self::correlation`].
    fn total(&self, model: &str) -> PyResult<f64> {
        model_energies(&self.inner)
            .iter()
            .find(|(k, _)| *k == model)
            .map(|&(_, e)| self.inner.total(e))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown OP-MP3 model {model:?}; one of mp2, mp3, p-mp3, op-mp3, sop-mp3, \
                     ssd-opmp3"
                ))
            })
    }
    /// The W matrices of `order` 2 or 3, keyed `alpha_ss`, `alpha_os`,
    /// `beta_ss`, `beta_os` (occupied × occupied of that spin, active only).
    fn w<'py>(&self, py: Python<'py>, order: u8) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        match order {
            2 => channels_dict(py, &self.inner.w.w2),
            3 => channels_dict(py, &self.inner.w.w3),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "order must be 2 or 3, got {order}"
            ))),
        }
    }
    /// `max |F_ia|` per spin `(alpha, beta)` in the semicanonical basis.
    #[getter]
    fn max_ov(&self) -> (f64, f64) {
        (self.inner.max_ov[0], self.inner.max_ov[1])
    }
    /// Validation certificate (see `RhfResult.certificate`).
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    /// `None` for the exact method; for `local="amplitude-threshold"` a dict
    /// with `scheme`, `eps`, `keep_fraction`, `integral_direct` (False),
    /// `n_pairs`, `cg_iterations`, `cg_relres`, `lambda_boys`, the E3 values
    /// of the two routes (`e3`, `e3_2n1`) and, with `compute_reference=True`,
    /// `canonical` (the canonical correlation energies, keyed like
    /// [`Self::correlation`]).
    #[getter]
    fn local<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        let Some((lr, reference)) = &self.local else {
            return Ok(None);
        };
        let d = crate::mp2::local_dict(py, lr.eps, lr.keep_fraction, false)?;
        d.set_item("n_pairs", lr.n_pairs)?;
        d.set_item("cg_iterations", lr.cg_iterations.to_vec())?;
        d.set_item("cg_relres", lr.cg_relres.to_vec())?;
        d.set_item("lambda_boys", lr.lambda_boys.lambda)?;
        d.set_item("e3", lr.opmp3.e3.total())?;
        d.set_item("e3_2n1", lr.e3_2n1)?;
        if let Some(c) = reference {
            let can = pyo3::types::PyDict::new(py);
            for (k, e) in model_energies(c) {
                can.set_item(k, e)?;
            }
            d.set_item("canonical", can)?;
        }
        Ok(Some(d))
    }
    fn __repr__(&self) -> String {
        format!(
            "Opmp3Result(e_hf={:.10}, e_nbs={:.10}, ssd_opmp3={:.10})",
            self.inner.e_hf, self.inner.e_nbs, self.inner.models.ssd_opmp3
        )
    }
}

/// OP-MP3, SOP-MP3, SSD-OPMP3 (and MP2, MP3, scalar Padé) on an RHF/ROHF
/// (`reference="hf"`) or RKS/ROKS (`reference=<functional>`) reference, built
/// on the HF Fock of the reference density with non-Brillouin singles.
/// `x_ss=None` uses the paper's printed same-spin damping for the reference;
/// `auxbasis=None` runs exact four-centre integrals (the paper's choice).
/// `local="amplitude-threshold"` with `eps=` runs the local OP-MP3
/// (`ferric_mp2::local_opmp3`; RI, closed shell) with the CLI `[local]`
/// rules and defaults (`fock_screen`, `lambda_boys`, `virtual_space`); it
/// refuses an ECP basis set, as `run_rimp2(local=...)` does.
///
/// With an ECP basis set the ECP is applied to the molecule, so the reference SCF is the one
/// `run_rhf` / `run_rohf` (`run_dft` / `run_roks` for a functional reference) solves.
/// `frozen_core` counts orbitals of the valence occupied space, after the ECP has removed its core.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, reference="hf", x_ss=None, auxbasis=None, frozen_core=None, memory_budget_gb=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, local=None, eps=None, compute_reference=None, fock_screen=None, lambda_boys=None, virtual_space=None, virtual_tau=None, virtual_tau_loose=None, virtual_solve_radius=None, virtual_boys_tau=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_opmp3(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    reference: &str,
    x_ss: Option<f64>,
    auxbasis: Option<&PyBasisSet>,
    frozen_core: Option<usize>,
    memory_budget_gb: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    local: Option<String>,
    eps: Option<f64>,
    compute_reference: Option<bool>,
    fock_screen: Option<bool>,
    lambda_boys: Option<f64>,
    virtual_space: Option<String>,
    virtual_tau: Option<f64>,
    virtual_tau_loose: Option<f64>,
    virtual_solve_radius: Option<f64>,
    virtual_boys_tau: Option<f64>,
) -> PyResult<PyOpmp3Result> {
    let kind = Opmp3RefKind::parse_config_str(reference)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let model = crate::mp2::local_model_from_kwargs(
        "run_opmp3",
        ferric_cli::LocalCfg {
            scheme: local,
            eps,
            reference: compute_reference,
            fock_screen,
            virtual_space,
            virtual_tau,
            virtual_tau_loose,
            virtual_solve_radius,
            virtual_boys_tau,
            lambda_boys,
            ..Default::default()
        },
    )?;
    if model.is_some() && (auxbasis.is_none() || mol.inner.multiplicity != 1) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "run_opmp3: local=\"amplitude-threshold\" runs on RI integrals of a closed-shell \
             molecule: pass auxbasis= (and a singlet)",
        ));
    }
    let config = Opmp3Config {
        x_ss,
        frozen_core: frozen_core.unwrap_or(0),
        scf_max_iter,
        scf_energy_conv,
        scf_density_conv,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        ..Opmp3Config::new(kind)
    };
    let x = config.resolved_x_ss().map_err(make_err)?;
    let sys = prepare(mol, basis_set)?;
    let mol = sys.mol();
    let aux = auxbasis
        .map(|a| sys.prepare_aux(&a.inner))
        .transpose()
        .map_err(make_err)?;
    let integrals = aux
        .as_ref()
        .map_or(Opmp3Integrals::Exact, Opmp3Integrals::Ri);
    // the same builder as the CLI's [local] run (`LocalModel::amplitude_config`)
    let lcfg = model.as_ref().map(|m| {
        (
            m.amplitude_config(config.frozen_core, config.memory_budget_bytes),
            m.reference,
        )
    });
    let (scf, inner, local_run) = py
        .allow_threads(|| {
            let scf = opmp3_scf(&sys, &config)?;
            let canonical = || opmp3_on_reference(&sys, &scf, integrals, &config);
            match (&lcfg, aux.as_ref()) {
                (Some((lc, with_reference)), Some(aux)) => {
                    let lr = local_opmp3_on_reference(&basis_set.inner, &scf, aux, &config, lc)?;
                    let reference = with_reference.then(canonical).transpose()?;
                    let r = lr.opmp3.clone();
                    Ok::<_, ferric_core::FerricError>((scf, r, Some((lr, reference))))
                }
                _ => {
                    let r = canonical()?;
                    Ok((scf, r, None))
                }
            }
        })
        .map_err(make_err)?;
    let reference_cert =
        reference_certificate(mol, &scf, &opmp3_scf_config(&config), &basis_set.inner);
    let aux_name = auxbasis.map(|a| a.inner.name.as_str());
    let certificate_data = match (&local_run, aux_name) {
        (Some((lr, _)), Some(a)) => ferric_mp2::certificate::local_opmp3_certificate(
            &reference_cert,
            a,
            config.frozen_core,
            x,
            lr.eps,
            lr.keep_fraction,
        ),
        _ => ferric_mp2::certificate::opmp3_certificate(
            &reference_cert,
            aux_name,
            config.frozen_core,
            x,
        ),
    };
    Ok(PyOpmp3Result {
        inner,
        certificate_data,
        local: local_run,
    })
}
