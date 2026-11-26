//! KS-DFT and double-hybrid bindings: moved verbatim from `lib.rs`.

use super::*;
use crate::method_result::MethodResult;
use crate::scf::{scf_options_from_kwargs, FockHook};
use ferric_core::certificate::Certificate;
use ferric_scf::options::entry::{ScfEntryPoint, ScfKwargs};
use std::borrow::Cow;

// ── KS-DFT ──

#[pyclass(frozen)]
#[pyo3(name = "DftResult")]
pub(crate) struct PyDftResult {
    /// SCF energy PLUS the dispersion correction, when one was requested.
    /// Equals `e_scf` exactly when `dispersion=None`.
    #[pyo3(get)]
    pub(crate) total_energy: f64,
    /// The Kohn-Sham SCF energy alone, with no dispersion correction.
    #[pyo3(get)]
    pub(crate) e_scf: f64,
    /// The dispersion correction (D3(BJ) or MBD@rsSCS) in Hartree, or `None`
    /// when dispersion was not requested.
    ///
    /// `None` means UNEVALUATED, not zero. A DFT energy with no dispersion and
    /// one whose dispersion happens to be small are different claims, and a
    /// 0.0 here would assert the second while meaning the first.
    #[pyo3(get)]
    pub(crate) e_dispersion: Option<f64>,
    /// Which dispersion model `e_dispersion` came from: `"D3(BJ)"`,
    /// `"MBD@rsSCS"`, or `None` when dispersion was not requested.
    #[pyo3(get)]
    pub(crate) dispersion_model: Option<String>,
    /// MBD@rsSCS only: the per-atom Hirshfeld volume ratios v_A / v_A^free of
    /// the converged density that scaled the TS inputs. `None` otherwise.
    #[pyo3(get)]
    pub(crate) volume_ratios: Option<Vec<f64>>,
    #[pyo3(get)]
    pub(crate) converged: bool,
    /// The molecule and orbital basis the SCF ran on, kept so `vxc()` /
    /// `e_xc` can evaluate the XC potential of the converged density on
    /// demand (with `scf_config`'s functional and grids).
    pub(crate) xc_mol: Molecule,
    pub(crate) xc_basis: basis::BasisSet,
    /// `(E_xc, V_xc)` of the converged density, computed on the first
    /// `vxc()` / `e_xc` call and cached. Lazy so a caller who never asks for
    /// it does not pay a grid build + XC evaluation per `run_dft`.
    /// Concurrent first calls build the grid once.
    pub(crate) xc_cache: crate::compute_once::ComputeOnce<(f64, Array2<f64>)>,
    /// Analytic nuclear gradient (natoms × 3) in Ha/Bohr, when computed
    /// (i.e. `with_gradient=True` was passed to `run_ksdft`). Closed-shell
    /// LDA / GGA / hybrid / RSH only; VV10 nonlocal piece is excluded.
    pub(crate) gradient_data: Option<Array2<f64>>,
    /// The solve, shared (`Arc`) as in `PyRhfResult::scf_data`; `density()`
    /// reads its `density_total`.
    pub(crate) scf_data: Arc<ScfResult>,
    /// The base SCF settings (functional, grid, density fitting, thresholds)
    /// this result was solved with; see `PyRhfResult::scf_config`.
    pub(crate) scf_config: Arc<RhfConfig>,
    /// The SCF ran with a `fock_hook` (recorded on the certificate).
    #[pyo3(get)]
    pub(crate) fock_hooked: bool,
}

impl PyDftResult {
    /// `(E_xc, V_xc)` of the converged density, evaluated once and cached.
    fn xc_pair(&self, py: Python<'_>) -> PyResult<&(f64, Array2<f64>)> {
        if let Some(p) = self.xc_cache.get() {
            return Ok(p);
        }
        let (mol, bs, cfg, d) = (
            &self.xc_mol,
            &self.xc_basis,
            &self.scf_config,
            &self.scf_data.density_total,
        );
        let cache = &self.xc_cache;
        // The cache's lock is taken INSIDE allow_threads, so a waiting thread
        // has released the GIL and cannot deadlock the thread computing.
        py.allow_threads(|| {
            cache.get_or_try_init(|| ferric_scf::rhf::ks_vxc_closed(mol, bs, cfg, d))
        })
        .map_err(make_err)
    }
}

impl MethodResult for PyDftResult {
    fn total_energy(&self) -> f64 {
        self.total_energy
    }
    fn certificate(&self) -> Cow<'_, Certificate> {
        let c = scf_kind_certificate(
            &self.xc_mol,
            &self.scf_data,
            &self.scf_config,
            &self.xc_basis,
            self.fock_hooked,
        );
        Cow::Owned(match self.dispersion_model.as_deref() {
            Some(m) => c.with_approximation("dispersion", m),
            None => c,
        })
    }
}

#[pymethods]
impl PyDftResult {
    #[doc = alias_doc!(total_energy)]
    #[getter]
    fn energy(&self) -> f64 {
        crate::method_result::total_energy_getter(self)
    }
    /// Validation certificate of this result (see `RhfResult.certificate`),
    /// with a `dispersion` approximation when a correction was added to
    /// `total_energy`.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        crate::method_result::certificate_getter(py, self)
    }
    /// Precision provenance of this KS-DFT SCF (see `RhfResult.precision`;
    /// the `xc-vxc` kernel carries a `ladder` record).
    #[getter]
    fn precision(&self, py: Python<'_>) -> PyResult<PyObject> {
        precision_dict(py, &self.scf_data.precision)
    }
    /// Storage precision of the RI-J tensor the SCF actually used: `"f32"`
    /// (the default) or `"f64"`, or `None` when J was not density-fitted.
    #[getter]
    fn ri_j_precision(&self) -> Option<&'static str> {
        self.scf_data.ri_j_precision().map(|p| p.as_str())
    }
    /// Storage precision of the RI-K dressed tensor the SCF actually used:
    /// `"f64"` (the default) or `"f32"`, or `None` when K was not
    /// density-fitted.
    #[getter]
    fn ri_k_precision(&self) -> Option<&'static str> {
        self.scf_data.ri_k_precision().map(|p| p.as_str())
    }
    /// Exchange-correlation potential V_xc[D] of the converged density `D`
    /// (`density()`), as an AO matrix (n_bf x n_bf), on the SCF's own XC grid.
    ///
    /// Semilocal part (incl. the meta-GGA τ term) plus VV10 V_nl when the
    /// functional has it; exact exchange is NOT included (for a hybrid this
    /// is only the DFT exchange-correlation part; for a range-separated
    /// hybrid the semilocal exchange is the short-range-attenuated one).
    /// Computed on first call.
    fn vxc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (_, v) = self.xc_pair(py)?;
        Ok(PyArray2::from_array(py, v))
    }

    /// E_xc[D] of the converged density on the SCF's XC grid, the energy
    /// that `vxc()` is the derivative of (semilocal + VV10, no exact
    /// exchange). Computed on first access, shared with `vxc()`.
    #[getter]
    fn e_xc(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.xc_pair(py)?.0)
    }

    /// The COSX final-grid pass record (see `RhfResult.cosx_final_pass`).
    #[getter]
    fn cosx_final_pass(&self, py: Python<'_>) -> PyResult<Option<Py<pyo3::types::PyDict>>> {
        cosx_final_dict(py, self.scf_data.cosx_final)
    }
    /// The secondary-basis guess (`guess_basis=`): `{"basis", "projection",
    /// "iterations", "energy", "nbasis_guess", "nbasis_target", "converged",
    /// "unconverged_reason", "used_as_guess"}` describing the secondary SCF,
    /// or `None` when no secondary-basis guess ran.
    #[getter]
    fn guess_basis(&self, py: Python<'_>) -> PyResult<Option<Py<pyo3::types::PyDict>>> {
        guess_basis_dict(py, self.scf_data.guess_basis.as_ref())
    }

    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.scf_data.density_total)
    }

    /// Kohn-Sham orbital energies (Hartree), ascending: the eigenvalues of
    /// the converged Fock matrix, one per column of `mo_coefficients()`.
    fn orbital_energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.scf_data.eps_alpha.clone())
    }

    /// Kohn-Sham MO coefficients C (n_bf x n_mo), column k = MO k in
    /// ascending energy order (the matrix `eval_mo_on_points` indexes).
    fn mo_coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.scf_data.mos_alpha)
    }

    /// Number of SCF iterations actually performed.
    ///
    /// Exposed because wall time alone cannot distinguish "each iteration is
    /// expensive" from "the SCF took many iterations to converge" — a
    /// distinction with opposite fixes (cheaper grid/basis vs. a better guess,
    /// level shift, or damping). Note `run_dft` walks a convergence *ladder*,
    /// so this is the iteration count of the rung that finally succeeded, not
    /// the total across all rungs — see `ferric_scf::ladder`.
    #[getter]
    fn iterations(&self) -> usize {
        self.scf_data.iterations
    }

    /// First-order energy change (Hartree) of one plain Roothaan step from the
    /// converged density, `Tr[F[D] (D' - D)]` with `D'` the aufbau density of
    /// the final `F[D]`. It is `<= 0` and zero (to rounding) only at a
    /// stationary aufbau state; a value below `-stationarity_tol` marks a
    /// converged exit that is not a stationary point (the solver also warns).
    /// `None` when the exit is not an aufbau convergence (not converged, MOM,
    /// Fermi smearing, `fractional_occ`, `irrep_nelec`, or no record).
    #[getter]
    fn one_step_energy_drop(&self) -> Option<f64> {
        self.scf_data
            .convergence
            .and_then(|c| c.one_step_energy_drop)
    }
    /// The tolerance `one_step_energy_drop` is judged against:
    /// `energy_conv + span * (nbasis * density_conv)^2`, with `span` the
    /// orbital-energy range of the final Fock matrix. `None` without a
    /// convergence record.
    #[getter]
    fn stationarity_tol(&self) -> Option<f64> {
        scf_stationarity_tol(&self.scf_data)
    }
    /// SCF exit reason: one of `"Converged"`, `"Stalled"`, `"Diverged"`,
    /// `"MaxIter"`, `"NotCertified"` (ROHF/ROKS only) — the `ScfExit` variant
    /// name. Strictly more informative than the `converged` bool, which
    /// collapses every failure mode into `false` — a `MaxIter` and a
    /// `Diverged` need different fixes.
    #[getter]
    fn exit_reason(&self) -> String {
        format!("{:?}", self.scf_data.exit)
    }

    /// Return the cached analytic nuclear gradient as an (natoms, 3) array.
    /// Returns `None` if the result was produced without `with_gradient=True`.
    fn gradient<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.gradient_data
            .as_ref()
            .map(|g| PyArray2::from_array(py, g))
    }
    fn __repr__(&self) -> String {
        format!(
            "DftResult(total_energy={:.10}, converged={})",
            self.total_energy, self.converged
        )
    }
    fn __str__(&self) -> String {
        format!(
            "KS-DFT Energy: {:.10} Ha (converged: {})",
            self.total_energy, self.converged
        )
    }
}

// ---------------------------------------------------------------------------
// MP2-based double hybrids (B2PLYP, DSD-PBEP86)
// ---------------------------------------------------------------------------

#[pyclass(frozen)]
#[pyo3(name = "DoubleHybridResult")]
pub(crate) struct PyDoubleHybridResult {
    #[pyo3(get)]
    pub(crate) total_energy: f64,
    #[pyo3(get)]
    pub(crate) e_ks: f64,
    #[pyo3(get)]
    pub(crate) e_corr_scaled: f64,
    #[pyo3(get)]
    pub(crate) e_os: f64,
    #[pyo3(get)]
    pub(crate) e_ss: f64,
    /// D3(BJ) energy included in `total_energy`; `None` without `dispersion=`.
    #[pyo3(get)]
    pub(crate) e_dispersion: Option<f64>,
    /// `"restricted"` (RKS reference) or `"unrestricted"` (UKS reference).
    #[pyo3(get)]
    pub(crate) reference: &'static str,
    #[pyo3(get)]
    pub(crate) c_os: f64,
    #[pyo3(get)]
    pub(crate) c_ss: f64,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

stored_method_result!(PyDoubleHybridResult, total_energy: total_energy, certificate: certificate_data);

#[pymethods]
impl PyDoubleHybridResult {
    /// Validation certificate (see `RhfResult.certificate`): capability
    /// `"b2plyp"` or `"dsd-pbep86"`, `grade` `"proven"` (both are compared with
    /// PySCF built from the definition) capped by the Kohn-Sham reference's,
    /// `measured_error` the kind's anchor only inside the evidence's settings
    /// (flat (75, 110) grid, f64 RI-J), `approximations`
    /// `scf ...` (the reference's), `ri`, `frozen-core` and
    /// `spin-component-scaling`.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        crate::method_result::certificate_getter(py, self)
    }
    fn __repr__(&self) -> String {
        format!(
            "DoubleHybridResult(total_energy={:.10}, e_ks={:.10}, e_corr={:.10})",
            self.total_energy, self.e_ks, self.e_corr_scaled
        )
    }
    fn __str__(&self) -> String {
        format!(
            "Double Hybrid Energy: {:.10} Ha (KS: {:.10}, scaled MP2: {:.10})",
            self.total_energy, self.e_ks, self.e_corr_scaled
        )
    }
}

/// MP2-based double hybrid: kind="b2plyp" (Grimme 2006) or "dsd-pbep86"
///     (Kozuch and Martin 2013, no dispersion: 0.72 HF, 0.28 PBE, 0.44 P86,
///     c_os 0.51, c_ss 0.36).
///
///     ``dispersion="d3bj"`` (or ``"d3(bj)"``) selects the kind's published
///     D3(BJ) set and adds the D3(BJ) energy: B2PLYP-D3(BJ) keeps B2PLYP's
///     coefficients; DSD-PBEP86-D3(BJ) is the refit set of Kozuch and Martin
///     (2013): 0.69 HF, 0.31 PBE, 0.44 P86, c_os 0.52, c_ss 0.22, D3(BJ) s6 0.48,
///     a1 0, s8 0, a2 5.6. Other values raise ValueError.
///
///     The reference is a UKS when ``mol.multiplicity > 1`` or ``unrestricted``
///     is True (a singlet UKS equals the RKS route); the PT2 term then uses the
///     unrestricted RI-MP2 spin components: E_OS = alpha-beta, E_SS =
///     alpha-alpha + beta-beta.
///
///     With an ECP basis set the ECP is applied to the molecule, so the reference SCF is
///     the one `run_dft` (`run_uks` for the unrestricted route) solves.
///     `frozen_core` counts orbitals of the valence occupied space, after the ECP has
///     removed its core.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, kind="b2plyp", frozen_core=None, k_builder=None, memory_budget_gb=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, scf_ri_j_precision=None, dispersion=None, unrestricted=false, lambda_=None, omega=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_double_hybrid(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    kind: &str,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    memory_budget_gb: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    scf_ri_j_precision: Option<&str>,
    dispersion: Option<&str>,
    unrestricted: bool,
    lambda_: Option<f64>,
    omega: Option<f64>,
) -> PyResult<PyDoubleHybridResult> {
    let wb97x = kind.to_lowercase().replace(['-', '_'], "") == "wb97xlv";
    let base_kind = match kind.to_lowercase().replace(['-', '_'], "").as_str() {
        "b2plyp" => DoubleHybridKind::B2plyp,
        "dsdpbep86" => DoubleHybridKind::DsdPbep86,
        "wb97xlv" => DoubleHybridKind::B2plyp, // placeholder, handled separately
        _ => {
            return Err(crate::errors::input_err(format!(
                "unknown double hybrid kind '{kind}'; expected 'b2plyp', 'dsd-pbep86' or 'wb97x-l-v'"
            )))
        }
    };
    if wb97x && (lambda_.is_some() || omega.is_some()) {
        // validated below
    }
    if !wb97x && (lambda_.is_some() || omega.is_some()) {
        return Err(crate::errors::input_err(
            "run_double_hybrid: lambda_/omega are only used by kind='wb97x-l-v'",
        ));
    }
    if wb97x && unrestricted {
        return Err(crate::errors::input_err(
            "run_double_hybrid: kind='wb97x-l-v' is closed-shell only",
        ));
    }
    let dh_kind = match dispersion.map(|d| d.trim().to_ascii_lowercase()) {
        None => base_kind,
        Some(d) if d == "d3bj" || d == "d3(bj)" => base_kind.with_d3bj(),
        Some(d) => {
            return Err(crate::errors::input_err(format!(
                "dispersion={d:?} is not available for a double hybrid: pass \"d3bj\" (or \
                 \"d3(bj)\") for its own published D3(BJ) set, or omit it"
            )))
        }
    };
    let open_shell = unrestricted || mol.inner.multiplicity > 1;

    let sys = prepare(py, mol, basis_set)?;
    let dfbs = prepare_aux(py, &sys, auxbasis)?;
    let mol = sys.mol();

    let ri_j_precision = parse_ri_j_precision(scf_ri_j_precision)?;
    let cfg = rhf_config_conv(
        k_builder,
        budget_bytes_from_gb(memory_budget_gb),
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    )?;
    let cfg = ks_jkfit_config(cfg, dh_kind.xc_name(), ri_j_precision);

    if wb97x {
        if open_shell {
            return Err(crate::errors::input_err(
                "run_double_hybrid: kind='wb97x-l-v' is closed-shell only",
            ));
        }
        let mut dh_cfg = ferric_cc::double_hybrid::DoubleHybridConfig {
            lambda: lambda_.unwrap_or(ferric_cc::double_hybrid::WB97X_L_V_LAMBDA),
            omega: omega.unwrap_or(ferric_cc::double_hybrid::WB97X_L_V_OMEGA),
            ..Default::default()
        };
        if let Some(fc) = frozen_core {
            dh_cfg.cc.frozen_core = fc;
        }
        dh_cfg.cc.memory_budget_bytes = budget_bytes_from_gb(memory_budget_gb);
        let (dh, ks) = ferric_cc::double_hybrid::run_wb97x_l_v_on(&dfbs, &cfg, &dh_cfg)
            .map_err(make_err)?;
        let reference = reference_certificate(mol, &ks, &cfg, &basis_set.inner);
        let certificate_data = ferric_cc::certificate::wb97x_l_v_certificate(
            &reference,
            &auxbasis.inner.name,
            &dh_cfg,
        );
        return Ok(PyDoubleHybridResult {
            certificate_data,
            total_energy: dh.total_energy,
            e_ks: dh.e_ks,
            e_corr_scaled: dh.e_c_scaled,
            e_os: dh.e_c_wft,
            e_ss: 0.0,
            e_dispersion: None,
            c_os: dh.lambda,
            c_ss: 0.0,
            reference: "rhf".into(),
        });
    }

    let (ks, cfg) = if open_shell {
        let ks = sys.solve_uhf(&cfg).and_then(ScfResult::into_converged);
        (ks.map_err(make_err)?, cfg)
    } else {
        ks_ladder_reference(&sys, cfg)?
    };

    let mut mp2_cfg = dh_kind.mp2_config();
    if let Some(fc) = frozen_core {
        mp2_cfg.frozen_core = fc;
    }
    mp2_cfg.memory_budget_bytes = budget_bytes_from_gb(memory_budget_gb);

    let r = if ks.spin == ferric_scf::Spin::Unrestricted {
        u_mp2_double_hybrid_on(&dfbs, &ks, &mp2_cfg)
    } else {
        mp2_double_hybrid_on(&dfbs, &ks, &mp2_cfg)
    }
    .map_err(make_err)?;
    let e_dispersion = match dh_kind.d3bj_functional() {
        Some(f) => {
            let params = ferric_d3::d3bj_params_for_functional(f).map_err(make_err)?;
            Some(ferric_d3::d3bj_energy_for_molecule(mol, &params).map_err(make_err)?)
        }
        None => None,
    };
    let mut reference = reference_certificate(mol, &ks, &cfg, &basis_set.inner);
    if let Some(f) = dh_kind.d3bj_functional() {
        reference =
            reference.with_approximation("dispersion", format!("D3(BJ) with {f} parameters"));
    }
    let certificate_data = ferric_mp2::certificate::mp2_double_hybrid_certificate(
        dh_kind.base_kind(),
        &reference,
        &auxbasis.inner.name,
        &mp2_cfg,
    );

    Ok(PyDoubleHybridResult {
        certificate_data,
        total_energy: r.total_energy + e_dispersion.unwrap_or(0.0),
        e_ks: r.e_ks,
        e_corr_scaled: r.e_corr_scaled,
        e_os: r.spin_components.e_os,
        e_ss: r.spin_components.e_ss,
        e_dispersion,
        reference: if open_shell {
            "unrestricted"
        } else {
            "restricted"
        },
        c_os: r.c_os,
        c_ss: r.c_ss,
    })
}

/// Kohn-Sham DFT (closed-shell). fock_hook: as for run_rhf; with a
///     hook every convergence-ladder rung runs with the hook applied and SOSCF
///     off on the rungs that would switch it on (its orbital Hessian does not
///     include the hook); dispersion and symmetry raise ValueError, and so does
///     with_gradient=True without fock_hook_grad (as for run_rhf; the KS
///     gradient of the hooked SCF plus grad(D, mol)). dm0: as for run_rhf; ladder rung 0 starts from it. Note: the final_* residuals, unmet_criteria and the
///     non-convergence RuntimeWarning are RHF-only today (this result has exit_reason and iterations).
///
///     diis_size, integral_thresh, eri, guess, smeared_charges, solvent /
///     pcm_lebedev_order, stability_descent, aurora, trah_trigger: as for run_rhf
///     (solve_rhf honours them with a functional too, on every ladder rung).
///     aurora takes effect only for a hybrid with at least the validated
///     exact-exchange fraction (the solver keeps other functionals on DIIS);
///     trah_trigger is skipped for meta-GGAs and range-separated hybrids. dm0
///     with guess= raises ValueError.
///     Not offered: diis, smearing_sigma and soscf (the convergence ladder sets
///     them per rung) and mom_reference ("initial" is UHF/UKS only).
///
///
///     cosx_grid / cosx_final_pass / cosx_final_grid / cosx_overlap_fit: COSX
///     knobs, read only with ``k_builder="cosx"`` (any of them with another
///     ``k_builder`` raises ValueError). A grid is ``(radial, angular)`` or
///     ``(radial, angular, prune)`` with prune ``"none"`` / ``"sgx"`` /
///     ``"nwchem"``. The default is the pruned sgx (35, 194) SCF grid plus one
///     final pass on the sgx (50, 302) grid.
///     `dispersion` adds a dispersion correction to the SCF energy (and, with
///     `with_gradient=True`, its analytic gradient to the gradient):
///     `"d3bj"` is D3(BJ) with the damping parameters published for
///     `functional`, `"d3bj(<name>)"` uses `<name>`'s instead; `"mbd"` is
///     MBD@rsSCS on Hirshfeld volume ratios of the converged density with the
///     beta published for `functional` (PBE, PBE0, HSE06), `"mbd(<name>)"` uses
///     `<name>`'s beta. Matching is case-insensitive. The MBD@rsSCS gradient
///     is exact, including the orbital relaxation of the volumes (Z-vector). `None` (the default) applies no correction and leaves the energy
///     exactly as it was. Any other value, or a functional with no published
///     parameters, raises ValueError -- there is no spelling that means "compute
///     a zero correction".
///
///     `grid_radial` / `grid_angular` / `grid_prune` set the main XC grid. All
///     `None` (the default) is the production grid: 75x110 pruned with
///     `"nwchem-radial"` (50 radial shells for H/He, 66 for Li-Ne) and a 1e-14
///     weight cutoff. A size given alone keeps that scheme at that size. `grid_angular`
///     must be a supported Lebedev order: 6, 14, 26, 38, 50, 74, 86, 110, 146,
///     170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030,
///     2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294 or 5810 (every
///     Lebedev-Laikov rule PySCF has).
///     `grid_prune="nwchem"` applies NWChem-style radial-region angular pruning
///     (~23% fewer points at 75x110, ~1e-10 Ha), `"nwchem-radial"` (the
///     default) also thins light atoms' radial grids; `"none"` is the flat grid;
///     `"defgrid1"` / `"defgrid2"` / `"defgrid3"` are ORCA's DEFGRIDs, which fix
///     both grid sizes themselves, so combining one with `grid_radial` or
///     `grid_angular` raises ValueError. Any other value raises ValueError.
///     Pruning has no table at `grid_angular=50`. With `with_gradient=True` the analytic gradient
///     differentiates the grid the SCF ran on (including a pruned one).
///     `grid_weight_cut` (ORCA's WeightCut, e.g. 1e-14) drops every main-grid
///     point whose final weight (radial x angular x Becke partition) has
///     absolute value below it; `None` (the default) is 1e-14, and 0.0 is no
///     cutoff (the grid is then the uncut one bit for bit), and a negative, NaN or infinite value
///     raises ValueError. The gradient drops the same points, so it is the
///     gradient of the cut energy (which jumps where a weight crosses the cutoff
///     under displacement, by about cutoff x integrand).
///     With `k_builder="cosx"` and `with_gradient=True` the gradient
///     differentiates the COSX energy; the overlap fit with a functional is not
///     differentiable (its response needs the XC Fock nuclear derivative), so
///     `cosx_overlap_fit=False` is required there and the fitted case raises
///     ValueError before the SCF.
///
///     symmetry=True (default False; False is byte-identical to before): blocked
///     closed-shell RKS. Detects the highest Abelian point group, symmetrizes and
///     reorients the geometry, builds the KS grid in that frame (where it is
///     exactly symmetric, so no Fock or density symmetrisation is applied) and
///     blocks the Fock diagonalization by irrep. Returns a SymmetricRhfResult
///     (NOT a DftResult): the energy belongs to the SYMMETRIZED geometry and no
///     density/MO matrices are offered. irrep_nelec={"A1": 4, "B2": 2} pins
///     electrons per irrep (even counts; requires symmetry=True). A partially
///     filled degenerate shell (stretched H2, singlet O2) raises ValueError unless
///     irrep_nelec is given: the grid can split such a shell. with_gradient,
///     dispersion, k_builder="cosx", point_charges/external_field raise
///     ValueError with symmetry=True. The blocked SCF is one plain solve (no
///     convergence ladder); non-convergence returns converged=False with a
///     RuntimeWarning.
///
///     Density fitting is on by default. ``df_j_aux`` and ``df_k_aux`` are set
///     independently, each defaulting to ``def2-universal-jkfit``, so RI-J and
///     RI-K can use different auxiliary bases with different sizes (``naux_J`` and
///     ``naux_K`` below):
///
///     - Coulomb: RI-J for every functional. ``df_j_aux`` names another auxiliary
///       basis; ``df_j_aux=""`` selects conventional four-centre J. RI-J keeps the
///       raw packed ``(P|mn)`` (f32 when it fits, ``ri_j_precision``) of at most
///       ``naux_J * nbf * (nbf + 1) / 2`` elements (Schwarz pair screening can
///       drop AO pairs).
///     - Exchange, by functional:
///
///       - pure (LDA/GGA/meta-GGA): no exact exchange, so no RI-K tensor is
///         built and ``df_k_aux`` has no effect.
///       - global hybrid (B3LYP, PBE0, ...): RI-K, a SECOND three-index tensor
///         beside RI-J's: the V^-1/2-dressed packed ``(P|mn)`` (f64,
///         ``ri_k_precision``) of ``naux_K * nbf * (nbf + 1) / 2`` elements
///         (``naux_K * nbf**2`` in a ``[gpu]`` mode that may build K on the
///         device). ``df_k_aux=""`` selects exact four-centre K and builds no
///         RI-K tensor.
///       - range-separated hybrid (wB97X-V, CAM-B3LYP, ...): exchange only via
///         two RI-K fitters (short-range erfc and long-range erf), each its own
///         dressed tensor; ``df_k_aux=""`` raises RuntimeError.
///       - ``k_builder="cosx"``: COSX replaces RI-K (RIJCOSX), RI-J stays on;
///         a named ``df_k_aux`` next to it raises.
///
///     - ``k_builder`` (``"direct"``, ``"link"``, ``"cosx"``) picks the
///       exact-exchange builder and is honoured only where exchange is not
///       density-fitted. While RI-K is active (the hybrid default) ``"direct"``
///       and ``"link"`` are IGNORED with a warning, as is any ``k_builder`` for a
///       pure or range-separated functional. For a hybrid, ``"direct"`` needs
///       ``df_k_aux=""``; ``"link"`` needs ``df_j_aux=""`` as well, because LinK
///       builds J and K in one four-centre pass, which RI-J replaces.
///     - ``BasisSet.n_functions(mol)`` gives ``nbf`` (orbital basis) and the
///       ``naux`` of an auxiliary basis, for a memory estimate.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, functional=None, k_builder=None, with_gradient=false,
    max_iter=None, energy_conv=None, density_conv=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None, memory_budget_gb=None,
    dispersion=None, df_j_aux=None, df_k_aux=None, ri_k_precision=None,
    grid_radial=None, grid_angular=None, grid_prune=None,
    cosx_grid=None, cosx_final_pass=None, cosx_final_grid=None, cosx_overlap_fit=None,
    symmetry=false, irrep_nelec=None, fock_hook=None,
    guess_basis=None, guess_projection=None, grid_weight_cut=None, ri_j_precision=None,
    dm0=None, relativistic=None, fock_hook_grad=None,
    diis_size=None, integral_thresh=None, eri=None, guess=None,
    smeared_charges=None, solvent=None, pcm_lebedev_order=None,
    stability_descent=None, aurora=None, trah_trigger=None,
    gc_rotation=None, gc_kernel=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_dft(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    functional: Option<&str>,
    k_builder: Option<&str>,
    with_gradient: bool,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    level_shift: Option<f64>,
    mom_after_iter: Option<usize>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    memory_budget_gb: Option<f64>,
    dispersion: Option<&str>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    ri_k_precision: Option<&str>,
    grid_radial: Option<usize>,
    grid_angular: Option<usize>,
    grid_prune: Option<&str>,
    cosx_grid: Option<&Bound<'_, PyAny>>,
    cosx_final_pass: Option<bool>,
    cosx_final_grid: Option<&Bound<'_, PyAny>>,
    cosx_overlap_fit: Option<bool>,
    symmetry: bool,
    irrep_nelec: Option<std::collections::BTreeMap<String, usize>>,
    fock_hook: Option<&Bound<'_, PyAny>>,
    guess_basis: Option<&str>,
    guess_projection: Option<&str>,
    grid_weight_cut: Option<f64>,
    ri_j_precision: Option<&str>,
    dm0: Option<numpy::PyReadonlyArray2<'_, f64>>,
    relativistic: Option<&str>,
    fock_hook_grad: Option<&Bound<'_, PyAny>>,
    diis_size: Option<usize>,
    integral_thresh: Option<f64>,
    eri: Option<&str>,
    guess: Option<&str>,
    smeared_charges: Option<Vec<(f64, f64, f64, f64, f64)>>,
    solvent: Option<&Bound<'_, PyAny>>,
    pcm_lebedev_order: Option<usize>,
    stability_descent: Option<bool>,
    aurora: Option<bool>,
    trah_trigger: Option<f64>,
    gc_rotation: Option<&str>,
    gc_kernel: Option<&str>,
) -> PyResult<PyObject> {
    refuse_dm0_with_guess("run_dft", dm0.is_some(), guess, guess_basis)?;
    let hook = crate::scf::parse_fock_hook("run_dft", fock_hook, fock_hook_grad, with_gradient)?;
    let fock_hook = hook.as_ref().map(FockHook::callable);
    let hook_grad = hook.as_ref().and_then(FockHook::grad);
    let guess_basis = parse_guess_basis(guess_basis, guess_projection)?;
    // Refuse an (E, grad-E) pair that does not belong to the same surface.
    //
    // `total_energy` below is `e_scf + e_dispersion`, but `gradient_data` is
    // The system owns a clone of the molecule, so the compute closures below
    // never borrow the PyMolecule pyclass field across the allow_threads
    // boundary. The basis set's ECP core counts are applied (no-op without an
    // ECP basis), so `nelec()` is the valence count the SCF occupies and
    // `dm0` is checked against it, as in `run_rhf`; every later step uses
    // this molecule.
    let sys = prepare(py, mol, basis_set)?;
    let (ctx, emol, prep, op) = (sys.ctx(), sys.mol(), sys.basis(), sys.op());
    // Parse the string/PyO3 kwargs in the order the pre-`ScfOptions`
    // construction evaluated them, so a call with several bad kwargs raises
    // the same (first) error.
    let relativistic = crate::scf::parse_relativistic(relativistic)?;
    let xc_name = ScfEntryPoint::Dft
        .functional(functional)
        .map_err(crate::errors::input_err)?
        .expect("run_dft always resolves a functional")
        .to_string();
    // Rung 0 starts from dm0; later rungs carry the previous rung's density.
    let init_guess_density = blocks::checked_dm0(dm0, prep, emol.nelec() as f64, "run_dft")?;
    // Resolve `dispersion=` BEFORE the SCF: an unknown spelling or a
    // functional with no published parameters must not cost an SCF first.
    let dispersion = dispersion
        .map(|spec| resolve_dispersion(spec, &xc_name))
        .transpose()?;
    let ri_j_precision = parse_ri_j_precision(ri_j_precision)?;
    let ri_k_precision = parse_ri_k_precision(ri_k_precision)?;
    let cosx = resolve_cosx_kwargs(
        k_builder,
        cosx_grid,
        cosx_final_pass,
        cosx_final_grid,
        cosx_overlap_fit,
    )?;
    // Main KS grid. `None` (no grid kwarg) keeps the production grid;
    // `ksdft_ladder` clones `cfg`, so every rung uses it. `grid_weight_cut=`
    // (off by default) rides the same config, so the gradient's grid response
    // drops the same points as the SCF grid.
    let dft_grid =
        resolve_dft_grid_with_cut(grid_radial, grid_angular, grid_prune, grid_weight_cut)?;
    let eri = crate::scf::parse_eri(eri)?;
    let use_sad_guess = parse_guess(guess)?;
    let pcm = build_pcm_config(solvent, pcm_lebedev_order)?;
    let trah_trigger = crate::scf::check_trah_trigger(trah_trigger)?;
    // The binding's own refusals (messages naming run_dft's kwargs) come
    // before the policy, which also refuses irrep_nelec without symmetry.
    if symmetry || irrep_nelec.is_some() {
        blocks::refuse_symmetric_dft(
            symmetry,
            with_gradient,
            dispersion.is_some(),
            fock_hook.is_some(),
        )?;
    }
    // run_dft's policy (RI-J / RI-K on by default, the RIJCOSX rule, the
    // "LDA" default) is `ScfEntryPoint::Dft`'s (ferric_scf::options::entry).
    // Parsed last of the keywords: an unknown spelling was the solver's
    // refusal before the binding parsed it (same class and text).
    let k_builder = crate::parse_k_builder(k_builder)?;
    let kw = ScfKwargs {
        max_iter,
        energy_conv,
        density_conv,
        diis_size,
        integral_thresh,
        k_builder,
        cosx: Some(cosx),
        df_j_aux,
        df_k_aux,
        ri_j_precision,
        ri_k_precision,
        // An explicit level_shift (0 included) turns off the meta-GGA
        // automatic shift.
        level_shift,
        mom_after_iter,
        guess: Some(crate::scf::guess_from_use_sad(use_sad_guess)),
        guess_basis,
        init_guess_density,
        external_potential: build_external_potential_with_smeared(
            point_charges,
            smeared_charges,
            external_field,
        ),
        pcm,
        three_index_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        stability_descent,
        symmetry,
        irrep_nelec: irrep_nelec.map(|m| m.into_iter().collect()),
        eri: Some(eri),
        relativistic,
        aurora,
        trah_trigger,
        functional,
        dft_grid,
        dispersion: dispersion.is_some(),
        ..ScfKwargs::default()
    };
    let options = scf_options_from_kwargs(ScfEntryPoint::Dft, kw)?;
    let mut cfg = options
        .into_rhf_config(ferric_scf::df_j::RiJTask::Energy)
        .py()?;
    cfg.gc_rotation = crate::scf::parse_gc_rotation(gc_rotation)?;
    cfg.gc_kernel = crate::scf::parse_gc_kernel(gc_kernel)?;
    if cfg.symmetry {
        return solve_symmetric_outcome(py, "run_dft", &sys, &cfg);
    }
    // The gradient below goes through `restricted_scf_gradient`, which
    // differentiates `cfg.dft_grid` (pruned or not), the grid the SCF runs on.
    // COSX exchange (RIJCOSX by default here): the gradient below goes
    // through `restricted_scf_gradient`, which differentiates the COSX (and
    // RI-J) energy the SCF built; setups it cannot differentiate (overlap fit
    // with a functional) are refused BEFORE the SCF. A COSX final-grid pass
    // is energy-only, so a gradient run drops it (`gradient_task_config`) and
    // `total_energy` stays on the surface the gradient describes.
    if with_gradient {
        ferric_scf::gradient::preflight_cosx_restricted(&cfg).map_err(crate::errors::input_err)?;
        cfg = ferric_scf::gradient::gradient_task_config(&cfg).into_owned();
    }
    // MBD@rsSCS's exact gradient needs the Z-vector relaxation term: refuse an
    // unsupported setup before the SCF and the free-atom solves, not after.
    if with_gradient && matches!(dispersion, Some(DispersionSpec::Mbd(_))) {
        if let Some(r) = ferric_scf::zvector_ks::unsupported_reason(&cfg) {
            return Err(crate::errors::input_err(format!(
                "dispersion=\"mbd\" with with_gradient=True: the exact MBD@rsSCS gradient \
                 is not available: {r}"
            )));
        }
    }
    // Run through the level-shift ladder (KS-DFT = solve_rhf with cfg.xc set),
    // so a hybrid on a hard system that DIIS-limit-cycles at level_shift=0
    // escalates the virtual-block shift instead of erroring out at max_iter.
    // The base cfg carries xc / grid / DF-JK aux into every rung. Only surface
    // a convergence error if the whole ladder fails to converge.
    let ladder = ferric_scf::ladder::ksdft_ladder(&cfg);
    // Release the GIL for the duration of the SCF ladder. Every argument
    // below is an owned Rust value (`sys` holds a clone of the molecule;
    // `ladder` is built above), so nothing Python-borrowed crosses the
    // closure boundary.
    let fock_hooked = fock_hook.is_some();
    let (rhf, rung_cfg) = blocks::solve_ks_ladder(
        py,
        &sys,
        &ladder,
        fock_hook.map(|h| {
            (
                h,
                with_gradient && hook_grad.is_none(),
                dispersion.is_some(),
            )
        }),
    )?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        }));
    }
    // Dispersion, opt-in: energy, and its gradient when one is requested,
    // evaluated ONCE from the same resolved parameters (and, for MBD@rsSCS,
    // the same converged density), so `total_energy` and `gradient()` cannot
    // come from different surfaces.
    //
    // `dispersion=None` leaves the energy BYTE-IDENTICAL to before this
    // feature existed -- `e_dispersion` is then None (UNEVALUATED), never 0.0,
    // so a caller cannot mistake "not asked for" for "computed and found to be
    // zero". Any failure (unknown functional, unparameterised element) is
    // raised, never swallowed into a neutral-looking zero.
    let disp = match &dispersion {
        None => None,
        Some(spec) => {
            let bs_owned = basis_set.inner.clone();
            let scf_ref = &rhf;
            Some(
                py.allow_threads(|| {
                    evaluate_dispersion(
                        spec,
                        ctx,
                        emol,
                        &bs_owned,
                        op,
                        &cfg,
                        scf_ref,
                        with_gradient,
                        None,
                    )
                })
                .map_err(make_err)?,
            )
        }
    };
    let gradient_data = if with_gradient {
        // Exactly `ks_gradient_closed` unless the SCF's exchange was COSX.
        let mut g = ferric_scf::gradient::restricted_scf_gradient(
            emol,
            prep,
            &basis_set.inner,
            op,
            sys.bounds(),
            &cfg,
            &rhf,
        )
        .map_err(make_err)?;
        // The hook's explicit term (its Pulay term is already in `g`).
        if let Some(h) = hook_grad {
            g = h.add_to(py, g, &rhf, false, emol)?;
        }
        // Dispersion is additive in the ENERGY, so it is additive in the
        // gradient. Adding it here rather than returning the bare KS gradient
        // is what makes `total_energy` and `gradient()` describe the SAME
        // surface -- an optimizer handed a mismatched pair converges to the
        // wrong geometry with nothing to indicate it.
        if let Some(d) = &disp {
            let dg = d.gradient.as_ref().ok_or_else(|| {
                make_err(ferric_core::FerricError::General(format!(
                    "{} returned no gradient although one was requested",
                    d.model
                )))
            })?;
            if dg.nrows() != g.nrows() {
                return Err(make_err(ferric_core::FerricError::General(format!(
                    "{} gradient has {} rows but the KS gradient has {}",
                    d.model,
                    dg.nrows(),
                    g.nrows()
                ))));
            }
            for k in 0..dg.nrows() {
                for a in 0..3 {
                    g[[k, a]] += dg[[k, a]];
                }
            }
        }
        Some(g)
    } else {
        None
    };
    let e_dispersion = disp.as_ref().map(|d| d.energy);
    let (dispersion_model, volume_ratios) = match disp {
        None => (None, None),
        Some(d) => (Some(d.model.to_string()), d.volume_ratios),
    };
    let result = PyDftResult {
        total_energy: rhf.energy + e_dispersion.unwrap_or(0.0),
        e_scf: rhf.energy,
        e_dispersion,
        dispersion_model,
        volume_ratios,
        converged: rhf.converged,
        xc_mol: sys.into_mol(),
        xc_basis: basis_set.inner.clone(),
        xc_cache: crate::compute_once::ComputeOnce::new(),
        gradient_data,
        scf_data: Arc::new(rhf),
        scf_config: Arc::new(rung_cfg),
        fock_hooked,
    };
    Ok(Py::new(py, result)?.into_any())
}

/// `run_dft(symmetry=True)` scoping: `irrep_nelec` needs `symmetry`, and the
/// result lives in the symmetry frame, so no gradient (input-frame rows) and
/// no dispersion (its energy would be silently dropped from the blocked
/// result) are offered.
pub(crate) fn refuse_dft_symmetry_extras(
    symmetry: bool,
    with_gradient: bool,
    dispersion: bool,
) -> PyResult<()> {
    let refuse = |m: &str| Err(crate::errors::input_err(m.to_string()));
    if !symmetry {
        return refuse("run_dft: irrep_nelec requires symmetry=True");
    }
    if with_gradient {
        return refuse(
            "run_dft(symmetry=True): with_gradient is not available; the blocked SCF runs in \
             the reoriented, symmetrized frame. Use symmetry=False for gradients.",
        );
    }
    if dispersion {
        return refuse(
            "run_dft(symmetry=True): dispersion is not available with the blocked solve; \
             use symmetry=False or add the correction separately (d3bj_energy).",
        );
    }
    Ok(())
}

/// Number of points in the main XC grid `run_dft` builds with these kwargs.
///
///     With the default 1e-14 weight cutoff every point's Becke weight is
///     evaluated (about the cost of building the grid, without its memory);
///     with `grid_weight_cut=0.0` it is a closed-form count (no grid is built).
#[pyfunction]
#[pyo3(signature = (mol, grid_radial=None, grid_angular=None, grid_prune=None, grid_weight_cut=None))]
pub(crate) fn dft_grid_point_count(
    mol: &PyMolecule,
    grid_radial: Option<usize>,
    grid_angular: Option<usize>,
    grid_prune: Option<&str>,
    grid_weight_cut: Option<f64>,
) -> PyResult<usize> {
    let cfg = resolve_dft_grid_with_cut(grid_radial, grid_angular, grid_prune, grid_weight_cut)?
        .unwrap_or_else(ferric_dft::grid::AtomicGridConfig::production);
    ferric_dft::grid::atomic_grid_point_count(&mol.inner, &cfg, cfg.prune).map_err(make_err)
}

/// Number of points in the COSX exchange grid `cosx_grid` describes for
///     `mol` (None = the default COSX SCF grid), without building it.
#[pyfunction]
#[pyo3(signature = (mol, cosx_grid=None))]
pub(crate) fn cosx_grid_point_count(
    mol: &PyMolecule,
    cosx_grid: Option<&Bound<'_, PyAny>>,
) -> PyResult<usize> {
    let g = match cosx_grid {
        Some(v) => parse_cosx_grid("cosx_grid", v)?,
        None => ferric_scf::cosx_k::CosxConfig::default().grid,
    };
    ferric_dft::grid::atomic_grid_point_count(&mol.inner, &g, g.prune).map_err(make_err)
}

/// A resolved `dispersion=` request: the model and the functional whose
/// published parameters it uses.
pub(crate) enum DispersionSpec {
    /// Grimme D3(BJ) with `<functional>`'s damping parameters.
    D3Bj(String),
    /// MBD@rsSCS with `<functional>`'s published range-separation β.
    Mbd(String),
}

/// Resolve a `dispersion=` spec.
///
/// Accepts exactly what the CLI's `[dft] dispersion` accepts, so the two
/// surfaces cannot drift apart (case-insensitive):
///   * `"d3bj"` / `"d3(bj)"` -- D3(BJ) with the running functional's parameters
///   * `"d3bj(<name>)"` -- D3(BJ) with `<name>`'s parameters instead, for when
///     ferric's XC name and the D3 fit's name differ
///   * `"mbd"` -- MBD@rsSCS with the running functional's published β
///   * `"mbd(<name>)"` -- MBD@rsSCS with `<name>`'s published β
///
/// Anything else is an error, never a silently-skipped correction, and so is
/// a functional with no published D3(BJ) fit or MBD β (checked here, before
/// any SCF).
///
/// SHARED between the energy and the gradient on purpose: one resolution
/// feeds both, so they cannot come from different surfaces.
pub(crate) fn resolve_dispersion(spec: &str, xc_name: &str) -> PyResult<DispersionSpec> {
    let lower = spec.trim().to_ascii_lowercase();
    let named = |prefix: &str| -> PyResult<Option<String>> {
        match lower.strip_prefix(prefix).and_then(|r| r.strip_suffix(')')) {
            None => Ok(None),
            Some(inner) if inner.trim().is_empty() => Err(crate::errors::input_err(format!(
                "dispersion=\"{prefix})\" names no functional"
            ))),
            Some(inner) => Ok(Some(inner.trim().to_string())),
        }
    };
    let value_err = |e: ferric_core::FerricError| crate::errors::input_err(e);
    let d3 = |f: String| -> PyResult<DispersionSpec> {
        ferric_d3::d3bj_params_for_functional(&f).map_err(value_err)?;
        Ok(DispersionSpec::D3Bj(f))
    };
    let mbd = |f: String| -> PyResult<DispersionSpec> {
        ferric_rpa::dispersion::mbd_rsscs_beta_for_functional(&f).map_err(value_err)?;
        Ok(DispersionSpec::Mbd(f))
    };
    if lower == "d3bj" || lower == "d3(bj)" {
        return d3(xc_name.to_string());
    }
    if lower == "mbd" {
        return mbd(xc_name.to_string());
    }
    if let Some(f) = named("d3bj(")? {
        return d3(f);
    }
    if let Some(f) = named("mbd(")? {
        return mbd(f);
    }
    Err(crate::errors::input_err(format!(
        "unknown dispersion scheme {spec:?}; expected \"d3bj\", \"d3(bj)\", \
         \"d3bj(<functional>)\", \"mbd\" or \"mbd(<functional>)\". Pass \
         dispersion=None for no correction -- there is no value meaning \"compute zero\"."
    )))
}

/// A dispersion correction evaluated at one geometry.
pub(crate) struct DispersionEval {
    /// `"D3(BJ)"` or `"MBD@rsSCS"`.
    pub(crate) model: &'static str,
    /// Hartree.
    pub(crate) energy: f64,
    /// (natoms, 3) Hartree/Bohr, when requested.
    pub(crate) gradient: Option<Array2<f64>>,
    /// MBD@rsSCS only: Hirshfeld volume ratios v_A / v_A^free.
    pub(crate) volume_ratios: Option<Vec<f64>>,
}

/// Evaluate `spec` at `mol`, given the converged closed-shell SCF (MBD@rsSCS
/// takes its Hirshfeld volumes from its density; D3(BJ) does not use it).
/// With `want_gradient` the analytic gradient is returned too; for MBD@rsSCS
/// it is exact, including the orbital relaxation of the volumes (Z-vector).
/// `mbd_cache` reuses MBD@rsSCS free-atom references built once by a caller
/// that evaluates many geometries; `None` builds them here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_dispersion(
    spec: &DispersionSpec,
    ctx: &ParallelContext,
    mol: &ferric_core::mol::Molecule,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    cfg: &RhfConfig,
    scf: &ScfResult,
    want_gradient: bool,
    mbd_cache: Option<&ferric_rpa::dispersion::mbd_scf::MbdFreeAtomCache>,
) -> Result<DispersionEval, ferric_core::FerricError> {
    match spec {
        DispersionSpec::D3Bj(functional) => {
            let params = ferric_d3::d3bj_params_for_functional(functional)?;
            let energy = ferric_d3::d3bj_energy_for_molecule(mol, &params)?;
            let gradient = if want_gradient {
                let rows = ferric_d3::d3bj_gradient_for_molecule(mol, &params)?;
                let mut g = Array2::<f64>::zeros((rows.len(), 3));
                for (k, row) in rows.iter().enumerate() {
                    for a in 0..3 {
                        g[[k, a]] = row[a];
                    }
                }
                Some(g)
            } else {
                None
            };
            Ok(DispersionEval {
                model: "D3(BJ)",
                energy,
                gradient,
                volume_ratios: None,
            })
        }
        DispersionSpec::Mbd(functional) => {
            use ferric_rpa::dispersion::mbd_rsscs::MbdRsscsConfig;
            use ferric_rpa::dispersion::mbd_scf::{
                mbd_rsscs_for_density, mbd_rsscs_for_scf, MbdFreeAtomCache,
            };
            let mcfg = MbdRsscsConfig::for_functional(functional)?;
            let built;
            let cache = match mbd_cache {
                Some(c) => c,
                None => {
                    built = MbdFreeAtomCache::build(ctx, mol, bs, op, cfg)?;
                    &built
                }
            };
            let r = if want_gradient {
                mbd_rsscs_for_scf(ctx, cache, mol, bs, op, cfg, scf, &mcfg)?
            } else {
                mbd_rsscs_for_density(cache, mol, bs, scf.density_total(), &mcfg, false)?
            };
            Ok(DispersionEval {
                model: "MBD@rsSCS",
                energy: r.energy,
                gradient: r.gradient,
                volume_ratios: Some(r.volume_ratios),
            })
        }
    }
}

/// Kohn-Sham DFT (closed-shell). Alias of run_dft (same grid_* kwargs,
///     grid_weight_cut included).
///
///     `dispersion` adds a dispersion correction to the SCF energy (and, with
///     `with_gradient=True`, its analytic gradient to the gradient):
///     `"d3bj"` is D3(BJ) with the damping parameters published for
///     `functional`, `"d3bj(<name>)"` uses `<name>`'s instead; `"mbd"` is
///     MBD@rsSCS on Hirshfeld volume ratios of the converged density with the
///     beta published for `functional` (PBE, PBE0, HSE06), `"mbd(<name>)"` uses
///     `<name>`'s beta. Matching is case-insensitive. The MBD@rsSCS gradient
///     is exact, including the orbital relaxation of the volumes (Z-vector). `None` (the default) applies no correction and leaves the energy
///     exactly as it was. Any other value, or a functional with no published
///     parameters, raises ValueError -- there is no spelling that means "compute
///     a zero correction".
///     symmetry / irrep_nelec: as for run_dft (blocked RKS, SymmetricRhfResult).
///     dm0: as for run_dft (ladder rung 0 starts from it).
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, functional=None, k_builder=None, with_gradient=false,
    max_iter=None, energy_conv=None, density_conv=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None, memory_budget_gb=None,
    dispersion=None, df_j_aux=None, df_k_aux=None, ri_k_precision=None,
    grid_radial=None, grid_angular=None, grid_prune=None,
    symmetry=false, irrep_nelec=None, guess_basis=None, guess_projection=None,
    grid_weight_cut=None, ri_j_precision=None, dm0=None, relativistic=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_ksdft(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    functional: Option<&str>,
    k_builder: Option<&str>,
    with_gradient: bool,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    level_shift: Option<f64>,
    mom_after_iter: Option<usize>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    memory_budget_gb: Option<f64>,
    dispersion: Option<&str>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    ri_k_precision: Option<&str>,
    grid_radial: Option<usize>,
    grid_angular: Option<usize>,
    grid_prune: Option<&str>,
    symmetry: bool,
    irrep_nelec: Option<std::collections::BTreeMap<String, usize>>,
    guess_basis: Option<&str>,
    guess_projection: Option<&str>,
    grid_weight_cut: Option<f64>,
    ri_j_precision: Option<&str>,
    dm0: Option<numpy::PyReadonlyArray2<'_, f64>>,
    relativistic: Option<&str>,
) -> PyResult<PyObject> {
    run_dft(
        py,
        mol,
        basis_set,
        functional,
        k_builder,
        with_gradient,
        max_iter,
        energy_conv,
        density_conv,
        level_shift,
        mom_after_iter,
        point_charges,
        external_field,
        memory_budget_gb,
        dispersion,
        df_j_aux,
        df_k_aux,
        ri_k_precision,
        grid_radial,
        grid_angular,
        grid_prune,
        None,
        None,
        None,
        None,
        symmetry,
        irrep_nelec,
        None,
        guess_basis,
        guess_projection,
        grid_weight_cut,
        ri_j_precision,
        dm0,
        relativistic,
        None,
        // run_ksdft is the frozen pre-6b surface: none of run_dft's later
        // keywords.
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
    )
}

// ---------------------------------------------------------------------------
// Open-shell Kohn-Sham: UKS and ROKS
// ---------------------------------------------------------------------------

/// Which open-shell Kohn-Sham reference [`run_open_ks`] solves.
#[derive(Clone, Copy)]
enum OpenKs {
    Uks,
    Roks,
}

impl OpenKs {
    fn who(self) -> &'static str {
        match self {
            OpenKs::Uks => "run_uks",
            OpenKs::Roks => "run_roks",
        }
    }
}

/// Every keyword of [`run_uks`] / [`run_roks`] after `py`, `mol` and
/// `basis_set`, for [`run_open_ks`]. A keyword `run_roks` does not take is
/// `None` there.
struct OpenKsKwargs<'a, 'py> {
    functional: &'a str,
    k_builder: Option<&'a str>,
    with_gradient: bool,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    level_shift: Option<f64>,
    mom_after_iter: Option<usize>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    memory_budget_gb: Option<f64>,
    df_j_aux: Option<&'a str>,
    df_k_aux: Option<&'a str>,
    ri_k_precision: Option<&'a str>,
    ri_j_precision: Option<&'a str>,
    grid_radial: Option<usize>,
    grid_angular: Option<usize>,
    grid_prune: Option<&'a str>,
    grid_weight_cut: Option<f64>,
    guess: Option<&'a str>,
    dm0: Option<&'a Bound<'py, PyAny>>,
    relativistic: Option<&'a str>,
    diis_size: Option<usize>,
    integral_thresh: Option<f64>,
    eri: Option<&'a str>,
    soscf: Option<bool>,
    smeared_charges: Option<Vec<(f64, f64, f64, f64, f64)>>,
    solvent: Option<&'a Bound<'py, PyAny>>,
    pcm_lebedev_order: Option<usize>,
    stability_descent: Option<bool>,
    cosx_grid: Option<&'a Bound<'py, PyAny>>,
    cosx_final_pass: Option<bool>,
    cosx_final_grid: Option<&'a Bound<'py, PyAny>>,
    cosx_overlap_fit: Option<bool>,
    trah_trigger: Option<f64>,
    guess_basis: Option<&'a str>,
    guess_projection: Option<&'a str>,
    mom_reference: Option<&'a str>,
    fock_hook: Option<&'a Bound<'py, PyAny>>,
    fock_hook_grad: Option<&'a Bound<'py, PyAny>>,
    gc_rotation: Option<&'a str>,
    gc_kernel: Option<&'a str>,
}

/// Shared body of [`run_uks`] / [`run_roks`]: the KS-DFT SCF config of
/// `run_dft` (RI-J/RI-K default `def2-universal-jkfit`, the production grid)
/// handed to `solve_uhf` / `solve_rohf` with `RhfConfig::xc` set, which is
/// the route the CLI takes for `kind = "ksdft"` on an open-shell molecule
/// (UKS) and `kind = "rohf"` with `[dft] functional` (ROKS). Result, gradient
/// and certificate are the `UhfResult` ones.
fn run_open_ks(
    py: Python<'_>,
    kind: OpenKs,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    kw: OpenKsKwargs<'_, '_>,
) -> PyResult<PyUhfResult> {
    let who = kind.who();
    let OpenKsKwargs {
        functional,
        k_builder,
        with_gradient,
        max_iter,
        energy_conv,
        density_conv,
        level_shift,
        mom_after_iter,
        point_charges,
        external_field,
        memory_budget_gb,
        df_j_aux,
        df_k_aux,
        ri_k_precision,
        ri_j_precision,
        grid_radial,
        grid_angular,
        grid_prune,
        grid_weight_cut,
        guess,
        dm0,
        relativistic,
        diis_size,
        integral_thresh,
        eri,
        soscf,
        smeared_charges,
        solvent,
        pcm_lebedev_order,
        stability_descent,
        cosx_grid,
        cosx_final_pass,
        cosx_final_grid,
        cosx_overlap_fit,
        trah_trigger,
        guess_basis,
        guess_projection,
        mom_reference,
        fock_hook,
        fock_hook_grad,
        gc_rotation,
        gc_kernel,
    } = kw;
    refuse_dm0_with_guess(who, dm0.is_some(), guess, guess_basis)?;
    let use_sad_guess = parse_guess(guess)?;
    let ecp = EcpMolecule::new(mol.inner.clone(), &basis_set.inner);
    // Refuse a molecule whose electron count cannot carry the multiplicity
    // before any integral work (an odd count with multiplicity 1, ...).
    let n_spin = spin_counts(ecp.mol(), who)?;
    let sys = prepare_ecp(py, ecp)?;
    let (emol, prep) = (sys.mol(), sys.basis());
    let dm0 = blocks::checked_dm0_spin(dm0, prep, n_spin, who)?;
    // Parsed in the order the pre-`ScfOptions` construction evaluated them;
    // the keywords added later (6b) after them.
    let relativistic = crate::scf::parse_relativistic(relativistic)?;
    let ri_j_precision = parse_ri_j_precision(ri_j_precision)?;
    let ri_k_precision = parse_ri_k_precision(ri_k_precision)?;
    let dft_grid =
        resolve_dft_grid_with_cut(grid_radial, grid_angular, grid_prune, grid_weight_cut)?;
    let hook = crate::scf::parse_fock_hook(who, fock_hook, fock_hook_grad, with_gradient)?;
    let guess_basis = parse_guess_basis(guess_basis, guess_projection)?;
    let eri = crate::scf::parse_eri(eri)?;
    let cosx = resolve_cosx_kwargs(
        k_builder,
        cosx_grid,
        cosx_final_pass,
        cosx_final_grid,
        cosx_overlap_fit,
    )?;
    let pcm = build_pcm_config(solvent, pcm_lebedev_order)?;
    let trah_trigger = crate::scf::check_trah_trigger(trah_trigger)?;
    // run_roks has no mom_reference kwarg (None); its policy refuses one.
    let mom_reference = parse_mom_reference(who, mom_reference, mom_after_iter, stability_descent)?;
    let entry = match kind {
        OpenKs::Uks => ScfEntryPoint::Uks,
        OpenKs::Roks => ScfEntryPoint::Roks,
    };
    // run_uks / run_roks policy (run_dft's KS defaults, including the
    // RIJCOSX rule) is `ScfEntryPoint::Uks` / `Roks`'s
    // (ferric_scf::options::entry).
    // Parsed last of the keywords: an unknown spelling was the solver's
    // refusal before the binding parsed it (same class and text).
    let k_builder = crate::parse_k_builder(k_builder)?;
    let kw = ScfKwargs {
        max_iter,
        energy_conv,
        density_conv,
        diis_size,
        integral_thresh,
        k_builder,
        cosx: Some(cosx),
        df_j_aux,
        df_k_aux,
        ri_j_precision,
        ri_k_precision,
        level_shift,
        mom_after_iter,
        mom_reference,
        soscf,
        guess: Some(crate::scf::guess_from_use_sad(use_sad_guess)),
        guess_basis,
        external_potential: build_external_potential_with_smeared(
            point_charges,
            smeared_charges,
            external_field,
        ),
        pcm,
        three_index_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        stability_descent,
        eri: Some(eri),
        relativistic,
        trah_trigger,
        functional: Some(functional),
        dft_grid,
        ..ScfKwargs::default()
    };
    let options = scf_options_from_kwargs(entry, kw)?;
    let mut config = options
        .into_rhf_config(ferric_scf::df_j::RiJTask::Energy)
        .py()?;
    config.gc_rotation = crate::scf::parse_gc_rotation(gc_rotation)?;
    config.gc_kernel = crate::scf::parse_gc_kernel(gc_kernel)?;
    let (solver, gradient_kind) = match kind {
        OpenKs::Uks => (blocks::OpenShell::Uhf, ScfGradientKind::Uhf),
        OpenKs::Roks => (blocks::OpenShell::Rohf, ScfGradientKind::Rohf),
    };
    let fock_hook = hook.as_ref().map(FockHook::callable);
    if with_gradient {
        config = scf_gradient_config(who, gradient_kind, &config, fock_hook)?;
        if hook.as_ref().is_some_and(FockHook::lacks_grad) {
            return crate::scf::refuse_hook_without_grad(who);
        }
    }
    let (r, hook_term) =
        blocks::solve_open_shell(py, solver, &sys, &config, fock_hook, dm0.as_ref())?;
    // As run_uhf / run_rohf: the hook's noise-floor term can put a
    // user-set energy_conv below the floor (a zero term never warns).
    crate::scf::warn_hook_noise_floor(
        py,
        who,
        SolverFamily::OpenShell,
        energy_conv.is_some(),
        emol,
        prep,
        &config,
        hook_term,
    );
    let gradient = with_gradient
        .then(|| {
            let g = scf_gradient(py, gradient_kind, &sys, &config, &r)?;
            // The hook's explicit term (its Pulay term is already in `g`).
            match hook.as_ref().and_then(FockHook::grad) {
                Some(h) => h.add_to(py, g, &r, true, emol),
                None => Ok(g),
            }
        })
        .transpose()?;
    let mut result = PyUhfResult::from_scf(r, emol, &config, prep, fock_hook.is_some());
    result.gradient_data = gradient;
    Ok(result)
}

/// Unrestricted Kohn-Sham (open-shell DFT): solve_uhf with RhfConfig.xc set,
///     the route the CLI takes for kind="ksdft" on an open-shell molecule.
///
///     `functional` is required (no silent LDA default). alpha/beta electron counts
///     come from the molecule's charge and multiplicity; a singlet is allowed and
///     equals RKS. Returns run_uhf's UhfResult (per-spin densities, orbital
///     energies, MO coefficients, occupations, certificate). with_gradient=True:
///     the analytic UKS gradient in result.gradient() (ks_gradient_uks), at the
///     RI-J precision f64 unless ri_j_precision says otherwise, as for run_dft.
///
///     The SCF settings are run_dft's: RI-J and RI-K through def2-universal-jkfit
///     unless df_j_aux / df_k_aux say "" or "none" (exact), the production
///     Becke-Lebedev grid unless grid_* say otherwise, run_rhf's convergence
///     defaults. Range-separated hybrids and meta-GGAs run (energy and gradient); the
///     gradient, VV10 (wB97X-V) included, agrees with finite differences of the
///     energy to ~2e-7 Ha/Bohr (NH2/6-31G).
///     dm0 is a pair (D_alpha, D_beta) with run_uhf's checks; guess is
///     "minao" (default) or "hcore". run_dft raises for multiplicity > 1.
///
///     The keywords after relativistic are run_uhf's, with its meaning (solve_uhf
///     honours them with a functional too): diis_size, integral_thresh, eri,
///     soscf (range-separated hybrids stay on DIIS), smeared_charges, solvent /
///     pcm_lebedev_order (IEF-PCM, with its gradient), stability_descent (the UKS
///     stability analysis; a functional it cannot analyse is skipped with a
///     printed reason), the cosx_* knobs (with k_builder="cosx"), trah_trigger
///     (not for meta-GGAs or range-separated hybrids), guess_basis /
///     guess_projection, mom_reference and fock_hook / fock_hook_grad. Not
///     offered: diis, smearing_sigma, aurora (closed-shell solver only), symmetry
///     (the blocked open-shell solve has no Python result type yet) and
///     dispersion (UhfResult has no dispersion fields).
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, functional, k_builder=None, with_gradient=false,
    max_iter=None, energy_conv=None, density_conv=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None, memory_budget_gb=None,
    df_j_aux=None, df_k_aux=None, ri_k_precision=None, ri_j_precision=None,
    grid_radial=None, grid_angular=None, grid_prune=None, grid_weight_cut=None,
    guess=None, dm0=None, relativistic=None,
    diis_size=None, integral_thresh=None, eri=None, soscf=None,
    smeared_charges=None, solvent=None, pcm_lebedev_order=None, stability_descent=None,
    cosx_grid=None, cosx_final_pass=None, cosx_final_grid=None, cosx_overlap_fit=None,
    trah_trigger=None, guess_basis=None, guess_projection=None, mom_reference=None,
    fock_hook=None, fock_hook_grad=None,
    gc_rotation=None, gc_kernel=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_uks<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    functional: &str,
    k_builder: Option<&str>,
    with_gradient: bool,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    level_shift: Option<f64>,
    mom_after_iter: Option<usize>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    memory_budget_gb: Option<f64>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    ri_k_precision: Option<&str>,
    ri_j_precision: Option<&str>,
    grid_radial: Option<usize>,
    grid_angular: Option<usize>,
    grid_prune: Option<&str>,
    grid_weight_cut: Option<f64>,
    guess: Option<&str>,
    dm0: Option<&Bound<'py, PyAny>>,
    relativistic: Option<&str>,
    diis_size: Option<usize>,
    integral_thresh: Option<f64>,
    eri: Option<&str>,
    soscf: Option<bool>,
    smeared_charges: Option<Vec<(f64, f64, f64, f64, f64)>>,
    solvent: Option<&Bound<'py, PyAny>>,
    pcm_lebedev_order: Option<usize>,
    stability_descent: Option<bool>,
    cosx_grid: Option<&Bound<'py, PyAny>>,
    cosx_final_pass: Option<bool>,
    cosx_final_grid: Option<&Bound<'py, PyAny>>,
    cosx_overlap_fit: Option<bool>,
    trah_trigger: Option<f64>,
    guess_basis: Option<&str>,
    guess_projection: Option<&str>,
    mom_reference: Option<&str>,
    fock_hook: Option<&Bound<'py, PyAny>>,
    fock_hook_grad: Option<&Bound<'py, PyAny>>,
    gc_rotation: Option<&str>,
    gc_kernel: Option<&str>,
) -> PyResult<PyUhfResult> {
    let kw = OpenKsKwargs {
        functional,
        k_builder,
        with_gradient,
        max_iter,
        energy_conv,
        density_conv,
        level_shift,
        mom_after_iter,
        point_charges,
        external_field,
        memory_budget_gb,
        df_j_aux,
        df_k_aux,
        ri_k_precision,
        ri_j_precision,
        grid_radial,
        grid_angular,
        grid_prune,
        grid_weight_cut,
        guess,
        dm0,
        relativistic,
        diis_size,
        integral_thresh,
        eri,
        soscf,
        smeared_charges,
        solvent,
        pcm_lebedev_order,
        stability_descent,
        cosx_grid,
        cosx_final_pass,
        cosx_final_grid,
        cosx_overlap_fit,
        trah_trigger,
        guess_basis,
        guess_projection,
        mom_reference,
        fock_hook,
        fock_hook_grad,
        gc_rotation,
        gc_kernel,
    };
    run_open_ks(py, OpenKs::Uks, mol, basis_set, kw)
}

/// Restricted open-shell Kohn-Sham: solve_rohf with RhfConfig.xc set, the
///     route the CLI takes for kind="rohf" with [dft] functional.
///
///     One set of spin-pure MOs: multiplicity - 1 singly occupied orbitals, the
///     rest doubly occupied (run_rohf's occupations and Guest-Saunders coupling).
///     Settings, result, gradient (ks_gradient_roks) and exclusions as for
///     run_uks; guess= ("minao" default, or "hcore") comes after relativistic
///     here. dm0 is run_rohf's pair. As in run_rohf, an explicit
///     cosx_final_pass=True or cosx_final_grid raises RuntimeError at the solve
///     (no ROHF/ROKS COSX final-grid pass), and solve_rohf has no TRAH step,
///     no stability analysis and no mom_reference="initial", so trah_trigger,
///     stability_descent and mom_reference are not offered.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, functional, k_builder=None, with_gradient=false,
    max_iter=None, energy_conv=None, density_conv=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None, memory_budget_gb=None,
    df_j_aux=None, df_k_aux=None, ri_k_precision=None, ri_j_precision=None,
    grid_radial=None, grid_angular=None, grid_prune=None, grid_weight_cut=None,
    dm0=None, relativistic=None,
    diis_size=None, integral_thresh=None, eri=None, guess=None, soscf=None,
    smeared_charges=None, solvent=None, pcm_lebedev_order=None,
    cosx_grid=None, cosx_final_pass=None, cosx_final_grid=None, cosx_overlap_fit=None,
    guess_basis=None, guess_projection=None, fock_hook=None, fock_hook_grad=None,
    gc_rotation=None, gc_kernel=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_roks<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    functional: &str,
    k_builder: Option<&str>,
    with_gradient: bool,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    level_shift: Option<f64>,
    mom_after_iter: Option<usize>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    memory_budget_gb: Option<f64>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    ri_k_precision: Option<&str>,
    ri_j_precision: Option<&str>,
    grid_radial: Option<usize>,
    grid_angular: Option<usize>,
    grid_prune: Option<&str>,
    grid_weight_cut: Option<f64>,
    dm0: Option<&Bound<'py, PyAny>>,
    relativistic: Option<&str>,
    diis_size: Option<usize>,
    integral_thresh: Option<f64>,
    eri: Option<&str>,
    guess: Option<&str>,
    soscf: Option<bool>,
    smeared_charges: Option<Vec<(f64, f64, f64, f64, f64)>>,
    solvent: Option<&Bound<'py, PyAny>>,
    pcm_lebedev_order: Option<usize>,
    cosx_grid: Option<&Bound<'py, PyAny>>,
    cosx_final_pass: Option<bool>,
    cosx_final_grid: Option<&Bound<'py, PyAny>>,
    cosx_overlap_fit: Option<bool>,
    guess_basis: Option<&str>,
    guess_projection: Option<&str>,
    fock_hook: Option<&Bound<'py, PyAny>>,
    fock_hook_grad: Option<&Bound<'py, PyAny>>,
    gc_rotation: Option<&str>,
    gc_kernel: Option<&str>,
) -> PyResult<PyUhfResult> {
    let kw = OpenKsKwargs {
        functional,
        k_builder,
        with_gradient,
        max_iter,
        energy_conv,
        density_conv,
        level_shift,
        mom_after_iter,
        point_charges,
        external_field,
        memory_budget_gb,
        df_j_aux,
        df_k_aux,
        ri_k_precision,
        ri_j_precision,
        grid_radial,
        grid_angular,
        grid_prune,
        grid_weight_cut,
        guess,
        dm0,
        relativistic,
        diis_size,
        integral_thresh,
        eri,
        soscf,
        smeared_charges,
        solvent,
        pcm_lebedev_order,
        stability_descent: None,
        cosx_grid,
        cosx_final_pass,
        cosx_final_grid,
        cosx_overlap_fit,
        trah_trigger: None,
        guess_basis,
        guess_projection,
        mom_reference: None,
        fock_hook,
        fock_hook_grad,
        gc_rotation,
        gc_kernel,
    };
    run_open_ks(py, OpenKs::Roks, mol, basis_set, kw)
}

/// Register the restricted-open-shell and open-shell Kohn-Sham entry points
/// (`run_rohf`, `run_uks`, `run_roks`).
pub(crate) fn register_open_shell_scf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_rohf, m)?)?;
    m.add_function(wrap_pyfunction!(run_uks, m)?)?;
    m.add_function(wrap_pyfunction!(run_roks, m)?)?;
    Ok(())
}

/// MBD@rsSCS dispersion energy (Ambrosetti et al. 2014), standalone.
///
///     `volume_ratios` are per-atom Hirshfeld volume ratios V_A/V_free, one per
///     atom (`run_dft(dispersion="mbd")` reports those of its converged density
///     as `DftResult.volume_ratios`). Pass exactly one of `beta` or `functional` (PBE 0.83, PBE0 0.85,
///     HSE06 0.85); an unlisted functional raises ValueError. Ghost atoms, Z > 54
///     and a polarization catastrophe raise.
#[pyfunction]
#[pyo3(signature = (mol, functional, s9 = 0.0))]
pub(crate) fn d3bj_energy(mol: &PyMolecule, functional: &str, s9: f64) -> PyResult<f64> {
    if !s9.is_finite() {
        return Err(crate::errors::input_err(format!(
            "d3bj_energy: s9 must be finite, got {s9}"
        )));
    }
    let params = ferric_d3::D3Params {
        s9,
        ..ferric_d3::d3bj_params_for_functional(functional).map_err(make_err)?
    };
    ferric_d3::d3bj_energy_for_molecule(&mol.inner, &params).map_err(make_err)
}

/// Result of `mbd_rsscs_energy`. Atomic units; per-atom lists in atom order.
#[pyclass(frozen)]
#[pyo3(name = "MbdRsscsResult")]
pub(crate) struct PyMbdRsscsResult {
    /// MBD@rsSCS dispersion energy (Hartree).
    #[pyo3(get)]
    pub(crate) energy: f64,
    /// β used.
    #[pyo3(get)]
    pub(crate) beta: f64,
    /// TS inputs (volume-ratio scaled): α₀, C6, R_vdW (Bohr).
    #[pyo3(get)]
    pub(crate) alpha_0_ts: Vec<f64>,
    #[pyo3(get)]
    pub(crate) c6_ts: Vec<f64>,
    #[pyo3(get)]
    pub(crate) r_vdw_ts: Vec<f64>,
    /// Range-separated-screened α₀, C6, R_vdW (Bohr) and ω.
    #[pyo3(get)]
    pub(crate) alpha_0_rsscs: Vec<f64>,
    #[pyo3(get)]
    pub(crate) c6_rsscs: Vec<f64>,
    #[pyo3(get)]
    pub(crate) r_vdw_rsscs: Vec<f64>,
    #[pyo3(get)]
    pub(crate) omega_rsscs: Vec<f64>,
}

#[pymethods]
impl PyMbdRsscsResult {
    fn __repr__(&self) -> String {
        format!(
            "MbdRsscsResult(energy={:.10e}, beta={}, natoms={})",
            self.energy,
            self.beta,
            self.alpha_0_rsscs.len()
        )
    }
}

/// MBD@rsSCS many-body dispersion energy (Ambrosetti et al., JCP 140,
/// 18A508 (2014)) for a molecule, in Hartree. Standalone: no SCF.
///
/// `volume_ratios` are the per-atom Hirshfeld volume ratios V_A/V_free (one
/// per atom, from the density functional the correction is paired with).
/// Exactly one of `beta` (explicit range-separation parameter) or
/// `functional` (published β: PBE 0.83, PBE0 0.85, HSE06 0.85) is required;
/// an unlisted functional raises. Ghost atoms are refused. Elements Z > 54
/// raise (no free-atom reference). A polarization catastrophe (coupled
/// oscillator matrix not positive definite) raises.
#[pyfunction]
#[pyo3(signature = (mol, volume_ratios, beta=None, functional=None))]
pub(crate) fn mbd_rsscs_energy(
    mol: &PyMolecule,
    volume_ratios: Vec<f64>,
    beta: Option<f64>,
    functional: Option<&str>,
) -> PyResult<PyMbdRsscsResult> {
    use ferric_rpa::dispersion::mbd_rsscs::MbdRsscsConfig;
    let cfg = match (beta, functional) {
        (Some(b), None) => MbdRsscsConfig::with_beta(b),
        (None, Some(f)) => MbdRsscsConfig::for_functional(f).map_err(crate::errors::input_err)?,
        (Some(_), Some(_)) => {
            return Err(crate::errors::input_err(
                "mbd_rsscs_energy: pass either beta or functional, not both",
            ))
        }
        (None, None) => {
            return Err(crate::errors::input_err(
                "mbd_rsscs_energy: beta is functional dependent — pass beta=... or \
                 functional=... (PBE, PBE0, HSE06)",
            ))
        }
    };
    let atoms = &mol.inner.atoms;
    if let Some(i) = atoms.iter().position(|a| a.ghost) {
        return Err(crate::errors::input_err(format!(
            "mbd_rsscs_energy: atom {i} is a ghost; MBD needs a real atom (and a volume \
             ratio) at every center — remove ghost atoms first"
        )));
    }
    if volume_ratios.len() != atoms.len() {
        return Err(crate::errors::input_err(format!(
            "mbd_rsscs_energy: {} volume ratios for {} atoms",
            volume_ratios.len(),
            atoms.len()
        )));
    }
    let z: Vec<usize> = atoms.iter().map(|a| a.z.max(0) as usize).collect();
    let pos: Vec<[f64; 3]> = atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let res = ferric_rpa::dispersion::mbd_rsscs::mbd_rsscs_energy(&z, &pos, &volume_ratios, &cfg)
        .map_err(make_err)?;
    Ok(PyMbdRsscsResult {
        energy: res.energy,
        beta: res.config.beta,
        alpha_0_ts: res.ts.alpha_0,
        c6_ts: res.ts.c6,
        r_vdw_ts: res.ts.r_vdw,
        alpha_0_rsscs: res.alpha_0_rsscs,
        c6_rsscs: res.c6_rsscs,
        r_vdw_rsscs: res.r_vdw_rsscs,
        omega_rsscs: res.omega_rsscs,
    })
}
