//! `run_omega_ri`: the omega-RI (erfc-attenuated metric) fit for canonical
//! RI-MP2, omega-RI-CDD-(SOS-)MP2 and omega-RI RPA. One entry point: every
//! method differs only in what consumes the shared dressed `B` tensor
//! (`ferric_mp2::omega_ri::fit_dressing`).

use super::*;
use ferric_mp2::omega_ri::RiFit;
use ferric_mp2::omega_ri_cdd::{omega_ri_cdd_mp2_on, CddEnergy, OmegaRiCddConfig, PseudoOrbitals};

#[pyclass(frozen)]
#[pyo3(name = "OmegaRiResult")]
pub(crate) struct PyOmegaRiResult {
    #[pyo3(get)]
    pub(crate) total_energy: f64,
    #[pyo3(get)]
    pub(crate) rhf_energy: f64,
    /// Correlation energy of `method` (SOS-scaled for "cdd-sos-mp2").
    #[pyo3(get)]
    pub(crate) e_corr: f64,
    /// Unscaled opposite-spin MP2 component (`None` for "rpa").
    #[pyo3(get)]
    pub(crate) e_os: Option<f64>,
    /// Same-spin MP2 component (`None` for "rpa" and "cdd-sos-mp2").
    #[pyo3(get)]
    pub(crate) e_ss: Option<f64>,
    #[pyo3(get)]
    pub(crate) method: String,
    /// The fit, e.g. "erfc(omega=0.1 Bohr^-1)" or "coulomb".
    #[pyo3(get)]
    pub(crate) fit: String,
    /// Mean pseudo-MO ranks over the Laplace points (CDD methods only).
    #[pyo3(get)]
    pub(crate) mean_rank_occ: Option<f64>,
    #[pyo3(get)]
    pub(crate) mean_rank_vir: Option<f64>,
    /// Fractions of occupied pairs and of virtuals the exchange screens kept
    /// ("cdd-mp2" only; 1.0 when unscreened).
    #[pyo3(get)]
    pub(crate) pair_fraction: Option<f64>,
    #[pyo3(get)]
    pub(crate) virtual_fraction: Option<f64>,
}

#[pymethods]
impl PyOmegaRiResult {
    fn __repr__(&self) -> String {
        format!(
            "OmegaRiResult(method={:?}, fit={:?}, total_energy={:.10}, e_corr={:.10})",
            self.method, self.fit, self.total_energy, self.e_corr
        )
    }
}

fn refuse(cond: bool, msg: &str) -> PyResult<()> {
    if cond {
        Err(crate::errors::input_err(format!("run_omega_ri: {msg}")))
    } else {
        Ok(())
    }
}

/// Correlation energy with the omega-RI fitting metric.
///
///     The three-index integrals are erfc(omega r)/r-attenuated and the fit uses
///     the metric V~^-1 V V~^-1 (Glasbrenner, Graf, Ochsenfeld, JCTC 16, 6856
///     (2020); Luenser, Schurkus, Ochsenfeld, JCTC 13, 1647 (2017)).
///     ``omega_bohr`` is in Bohr^-1 (the papers recommend 0.1); 0 is the
///     Coulomb fit. The algebra is dense: the fit changes the energy, not the
///     cost.
///
///     ``method``: ``"rimp2"`` (canonical RI-MP2), ``"cdd-mp2"`` /
///     ``"cdd-sos-mp2"`` (Laplace MP2 / SOS-MP2 in pivoted-Cholesky pseudo-MOs;
///     ``n_quad`` in {3, 5, 7}, default 7; ``cdd_tol`` default 0 keeps every
///     pseudo-MO; ``cdd_schwarz_thresh`` / ``cdd_pair_thresh`` are the paper's
///     exchange screens, default 0 = none; ``c_os`` default 1.3), or ``"rpa"``
///     (full-rank PDEP-RPA, closed shell or UHF). A key the chosen method does
///     not read raises ``ValueError``. The MP2 methods are closed-shell only.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, omega_bohr, method="rimp2", frozen_core=None, n_quad=None, cdd_tol=None, cdd_schwarz_thresh=None, cdd_pair_thresh=None, c_os=None, memory_budget_gb=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_omega_ri(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    omega_bohr: f64,
    method: &str,
    frozen_core: Option<usize>,
    n_quad: Option<usize>,
    cdd_tol: Option<f64>,
    cdd_schwarz_thresh: Option<f64>,
    cdd_pair_thresh: Option<f64>,
    c_os: Option<f64>,
    memory_budget_gb: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyOmegaRiResult> {
    let fit = RiFit::omega_ri(omega_bohr).map_err(make_err)?;
    let cdd = matches!(method, "cdd-mp2" | "cdd-sos-mp2");
    refuse(
        !matches!(method, "rimp2" | "cdd-mp2" | "cdd-sos-mp2" | "rpa"),
        &format!(
            "method = {method:?}; expected \"rimp2\", \"cdd-mp2\", \"cdd-sos-mp2\" or \"rpa\""
        ),
    )?;
    refuse(
        !cdd && (n_quad.is_some() || cdd_tol.is_some()),
        "n_quad / cdd_tol are read by the cdd methods only",
    )?;
    refuse(
        method != "cdd-mp2" && (cdd_schwarz_thresh.is_some() || cdd_pair_thresh.is_some()),
        "cdd_schwarz_thresh / cdd_pair_thresh screen the exchange term of \"cdd-mp2\" only",
    )?;
    refuse(
        method != "cdd-sos-mp2" && c_os.is_some(),
        "c_os is read by \"cdd-sos-mp2\" only",
    )?;
    refuse(
        method != "rpa" && mol.inner.multiplicity != 1,
        "the MP2 methods are closed-shell only (method = \"rpa\" accepts UHF)",
    )?;
    let (rhf_config, _jk) = mp2_exact_default_config(
        "run_omega_ri",
        None,
        memory_budget_gb,
        None,
        None,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
        None,
    )?;
    let budget = budget_bytes_from_gb(memory_budget_gb);
    let fc = frozen_core.unwrap_or(0);
    let sys = prepare(py, mol, basis_set)?;
    let dfbs = prepare_aux(py, &sys, auxbasis)?;
    // RHF for a singlet, UHF otherwise; non-convergence is an error.
    let rhf = sys.solve_reference(&rhf_config).map_err(make_err)?;
    let emol = sys.mol();
    let method_s = method.to_string();
    let mut out = PyOmegaRiResult {
        total_energy: 0.0,
        rhf_energy: rhf.energy,
        e_corr: 0.0,
        e_os: None,
        e_ss: None,
        method: method_s.clone(),
        fit: fit.label(),
        mean_rank_occ: None,
        mean_rank_vir: None,
        pair_fraction: None,
        virtual_fraction: None,
    };
    py.allow_threads(|| -> Result<(), ferric_core::error::FerricError> {
        match method_s.as_str() {
            "rimp2" => {
                let cfg = RiMp2Config {
                    frozen_core: fc,
                    memory_budget_bytes: budget,
                    ..Default::default()
                };
                let (sc, _) =
                    ferric_mp2::rimp2::ri_mp2_spin_components_fitted_on(&dfbs, &rhf, &cfg, &fit)?;
                out.e_corr = sc.e_total;
                out.e_os = Some(sc.e_os);
                out.e_ss = Some(sc.e_ss);
            }
            "rpa" => {
                let cfg = ferric_rpa::config::PdepRpaConfig {
                    frozen_core: fc,
                    memory_budget_bytes: budget,
                    ..Default::default()
                };
                let r = if emol.multiplicity == 1 {
                    ferric_rpa::run_pdep_rpa_fitted_on(&dfbs, &rhf, &cfg, &fit)?
                } else {
                    ferric_rpa::run_u_pdep_rpa_fitted_on(&dfbs, &rhf, &cfg, &fit)?
                };
                out.e_corr = r.e_rpa;
            }
            _ => {
                let cfg = OmegaRiCddConfig {
                    fit,
                    frozen_core: fc,
                    n_quad: n_quad.unwrap_or(7),
                    orbitals: PseudoOrbitals::Cholesky {
                        tol: cdd_tol.unwrap_or(0.0),
                    },
                    energy: if method_s == "cdd-sos-mp2" {
                        CddEnergy::Sos {
                            c_os: c_os.unwrap_or(1.3),
                        }
                    } else {
                        CddEnergy::Mp2
                    },
                    schwarz_thresh: cdd_schwarz_thresh.unwrap_or(0.0),
                    pair_thresh: cdd_pair_thresh.unwrap_or(0.0),
                    memory_budget_bytes: budget,
                };
                let r = omega_ri_cdd_mp2_on(&dfbs, &rhf, &cfg)?;
                out.e_corr = r.e_corr;
                out.e_os = Some(r.e_os);
                out.e_ss = r.e_ss;
                out.mean_rank_occ = Some(r.mean_rank_occ);
                out.mean_rank_vir = Some(r.mean_rank_vir);
                if method_s == "cdd-mp2" {
                    out.pair_fraction = Some(r.pair_fraction);
                    out.virtual_fraction = Some(r.virtual_fraction);
                }
            }
        }
        Ok(())
    })
    .map_err(make_err)?;
    out.total_energy = out.rhf_energy + out.e_corr;
    Ok(out)
}
