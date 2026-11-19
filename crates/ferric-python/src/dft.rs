//! KS-DFT and double-hybrid bindings: moved verbatim from `lib.rs`.

use super::*;
use ferric_scf::options::{df_k_aux_or_default, AuxBasis, ScfOptions};

/// A Kohn-Sham `df_j_aux` / `df_k_aux` kwarg: the parsed spelling (opt-outs
/// are [`AuxBasis::Off`]), or `def2-universal-jkfit` when unset.
fn ks_aux(requested: Option<&str>) -> AuxBasis {
    requested
        .map(AuxBasis::parse)
        .unwrap_or_else(AuxBasis::default_jk)
}

// ── KS-DFT ──

#[pyclass]
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
    pub(crate) xc_cache: std::sync::OnceLock<(f64, Array2<f64>)>,
    /// Serializes the first evaluation so concurrent first calls build the
    /// grid once (`OnceLock::get_or_try_init` is not stable).
    pub(crate) xc_lock: std::sync::Mutex<()>,
    pub(crate) density_data: Array2<f64>,
    /// Analytic nuclear gradient (natoms × 3) in Ha/Bohr, when computed
    /// (i.e. `with_gradient=True` was passed to `run_ksdft`). Closed-shell
    /// LDA / GGA / hybrid / RSH only; VV10 nonlocal piece is excluded.
    pub(crate) gradient_data: Option<Array2<f64>>,
    pub(crate) scf_data: ScfResult,
    /// The base SCF settings (functional, grid, density fitting, thresholds)
    /// this result was solved with; see `PyRhfResult::scf_config`.
    pub(crate) scf_config: RhfConfig,
    /// The SCF ran with a `fock_hook` (recorded on the certificate).
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
            &self.density_data,
        );
        let (cache, lock) = (&self.xc_cache, &self.xc_lock);
        // One evaluation even under concurrent first calls: the guard is taken
        // INSIDE allow_threads, so a waiting thread has released the GIL and
        // cannot deadlock the thread computing (which needs the GIL back).
        py.allow_threads(|| {
            let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            if cache.get().is_none() {
                let pair = ferric_scf::rhf::ks_vxc_closed(mol, bs, cfg, d)?;
                let _ = cache.set(pair);
            }
            Ok::<(), ferric_core::FerricError>(())
        })
        .map_err(make_err)?;
        Ok(cache.get().expect("xc_cache is set under xc_lock above"))
    }
}

#[pymethods]
impl PyDftResult {
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
    /// Validation certificate of this result (see `RhfResult.certificate`),
    /// with a `dispersion` approximation when a correction was added to
    /// `total_energy`.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        let c = scf_kind_certificate(
            &self.xc_mol,
            &self.scf_data,
            &self.scf_config,
            &self.xc_basis,
            self.fock_hooked,
        );
        let c = match self.dispersion_model.as_deref() {
            Some(m) => c.with_approximation("dispersion", m),
            None => c,
        };
        certificate_dict(py, &c)
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
        PyArray2::from_array(py, &self.density_data)
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
    /// SCF exit reason: one of `"Converged"`, `"Plateau"`, `"Stalled"`,
    /// `"Diverged"`, `"MaxIter"`, `"NotCertified"` (ROHF/ROKS only) — the
    /// `ScfExit` variant name. Strictly more
    /// informative than the `converged` bool, which collapses every failure
    /// mode into `false` — a `Plateau` and a `Diverged` need different fixes.
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

#[pyclass]
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
        certificate_dict(py, &self.certificate_data)
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

/// Run an MP2-based double hybrid (B2PLYP or DSD-PBEP86).
///
/// `kind`: "b2plyp" (Grimme 2006) or "dsd-pbep86" (the no-dispersion set of
/// Kozuch and Martin, J. Comput. Chem. 34, 2327 (2013)).
/// Converges the appropriate DFT reference, then adds scaled MP2 correlation.
///
/// `dispersion="d3bj"` (or `"d3(bj)"`) selects the double hybrid's published
/// D3(BJ) set and adds its D3(BJ) energy: B2PLYP-D3(BJ) keeps B2PLYP's
/// coefficients (D3(BJ) row of Goerigk and Grimme, PCCP 13, 6670 (2011));
/// DSD-PBEP86-D3(BJ) is the refit set of Kozuch and Martin (2013): 0.69 HF,
/// 0.31 PBE, 0.44 P86, c_os 0.52, c_ss 0.22, D3(BJ) s6 0.48, a1 0, s8 0,
/// a2 5.6. Any other spelling is an error.
///
/// The reference is a UKS when `mol.multiplicity > 1` or `unrestricted=True`
/// (a singlet UKS equals the RKS route); the PT2 term is then
/// `c_os * E_alpha-beta + c_ss * (E_alpha-alpha + E_beta-beta)` of the
/// unrestricted RI-MP2 on the UKS orbitals.
///
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, kind="b2plyp", frozen_core=None, k_builder=None, memory_budget_gb=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, scf_ri_j_precision=None, dispersion=None, unrestricted=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_double_hybrid(
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
) -> PyResult<PyDoubleHybridResult> {
    let base_kind = match kind.to_lowercase().replace(['-', '_'], "").as_str() {
        "b2plyp" => DoubleHybridKind::B2plyp,
        "dsdpbep86" => DoubleHybridKind::DsdPbep86,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown double hybrid kind '{kind}'; expected 'b2plyp' or 'dsd-pbep86'"
            )))
        }
    };
    let dh_kind = match dispersion.map(|d| d.trim().to_ascii_lowercase()) {
        None => base_kind,
        Some(d) if d == "d3bj" || d == "d3(bj)" => base_kind.with_d3bj(),
        Some(d) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dispersion={d:?} is not available for a double hybrid: pass \"d3bj\" (or \
                 \"d3(bj)\") for its own published D3(BJ) set, or omit it"
            )))
        }
    };
    let open_shell = unrestricted || mol.inner.multiplicity > 1;

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
    let cfg = ks_jkfit_config(cfg, dh_kind.xc_name(), scf_ri_j_precision)?;

    let (ks, cfg) = if open_shell {
        let ks = solve_uhf(&ctx, &mol.inner, &prep, &bounds, &cfg).map_err(make_err)?;
        (converged(ks)?, cfg)
    } else {
        ks_ladder_reference(&prep, &bounds, &mol.inner, cfg)?
    };

    let mut mp2_cfg = dh_kind.mp2_config();
    if let Some(fc) = frozen_core {
        mp2_cfg.frozen_core = fc;
    }
    mp2_cfg.memory_budget_bytes = budget_bytes_from_gb(memory_budget_gb);

    let r = if open_shell {
        u_mp2_double_hybrid(&mol.inner, &prep, &dfbs, &ks, &mp2_cfg)
    } else {
        mp2_double_hybrid(&mol.inner, &prep, &dfbs, &ks, &mp2_cfg)
    }
    .map_err(make_err)?;
    let e_dispersion = match dh_kind.d3bj_functional() {
        Some(f) => {
            let params = ferric_d3::d3bj_params_for_functional(f).map_err(make_err)?;
            Some(ferric_d3::d3bj_energy_for_molecule(&mol.inner, &params).map_err(make_err)?)
        }
        None => None,
    };
    let mut reference = reference_certificate(&mol.inner, &ks, &cfg, &basis_set.inner);
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

/// Kohn-Sham DFT (closed-shell). `functional` is an XC name (LDA/PBE/B3LYP/
/// wB97X-V/…). Convergence aids match the CLI `[scf]` section: `level_shift`
/// (Ha) and `mom_after_iter` help difficult DFT SCFs converge.
///
/// Note: the final_* residuals, unmet_criteria and non-convergence RuntimeWarning
/// are RHF-only today (the result has `exit_reason` and `iterations`).
///
/// `symmetry=True`: point-group-blocked closed-shell RKS
/// (`ferric_scf::symmetric::solve_rhf_symmetric`), returning a
/// `SymmetricRhfResult` for the symmetrized geometry; `irrep_nelec` pins
/// electrons per irrep. Refused with `with_gradient` or `dispersion`.
///
/// Density fitting is on by default. `df_j_aux` and `df_k_aux` are set
/// independently, each defaulting to `def2-universal-jkfit`, so RI-J and
/// RI-K can use different auxiliary bases with different sizes (`naux_J` and
/// `naux_K` below):
///
/// - Coulomb: RI-J for every functional. `df_j_aux` names another auxiliary
///   basis; `df_j_aux=""` selects conventional four-centre J. RI-J keeps the
///   raw packed `(P|mn)` (f32 when it fits, `ri_j_precision`) of at most
///   `naux_J * nbf * (nbf + 1) / 2` elements (Schwarz pair screening can
///   drop AO pairs).
/// - Exchange, by functional:
///
///   - pure (LDA/GGA/meta-GGA): no exact exchange, so no RI-K tensor is
///     built and `df_k_aux` has no effect.
///   - global hybrid (B3LYP, PBE0, ...): RI-K, a second three-index tensor
///     beside RI-J's: the V^-1/2-dressed packed `(P|mn)` (f64,
///     `ri_k_precision`) of `naux_K * nbf * (nbf + 1) / 2` elements
///     (`naux_K * nbf**2` in a `[gpu]` mode that may build K on the
///     device). `df_k_aux=""` selects exact four-centre K and builds no
///     RI-K tensor.
///   - range-separated hybrid (wB97X-V, CAM-B3LYP, ...): exchange only via
///     two RI-K fitters (short-range erfc and long-range erf), each its own
///     dressed tensor; `df_k_aux=""` raises RuntimeError.
///   - `k_builder="cosx"`: COSX replaces RI-K (RIJCOSX), RI-J stays on;
///     a named `df_k_aux` next to it raises.
///
/// - `k_builder` (`"direct"`, `"link"`, `"cosx"`) picks the
///   exact-exchange builder and is honoured only where exchange is not
///   density-fitted. While RI-K is active (the hybrid default) `"direct"`
///   and `"link"` are IGNORED with a warning, as is any `k_builder` for a
///   pure or range-separated functional. For a hybrid, `"direct"` needs
///   `df_k_aux=""`; `"link"` needs `df_j_aux=""` as well, because LinK
///   builds J and K in one four-centre pass, which RI-J replaces.
/// - `BasisSet.n_functions(mol)` gives `nbf` (orbital basis) and the
///   `naux` of an auxiliary basis, for a memory estimate.
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
) -> PyResult<PyObject> {
    refuse_dm0_with_guess("run_dft", dm0.is_some(), None, guess_basis)?;
    let hook_grad =
        crate::scf::parse_fock_hook_grad("run_dft", fock_hook, fock_hook_grad, with_gradient)?;
    let guess_basis = parse_guess_basis(guess_basis, guess_projection)?;
    // Refuse an (E, grad-E) pair that does not belong to the same surface.
    //
    // `total_energy` below is `e_scf + e_dispersion`, but `gradient_data` is
    // Owned clone so the compute closure below never borrows the PyMolecule
    // pyclass field across the allow_threads boundary. The basis set's ECP
    // core counts are applied (no-op without an ECP basis), so `nelec()` is
    // the valence count the SCF occupies and `dm0` is checked against it, as
    // in `run_rhf`; every later step uses this molecule.
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let prep = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    // Parse the string/PyO3 kwargs in the order the pre-`ScfOptions`
    // construction evaluated them, so a call with several bad kwargs raises
    // the same (first) error.
    let relativistic = crate::scf::parse_relativistic(relativistic)?;
    let xc_name = functional.unwrap_or("LDA").to_string();
    // Rung 0 starts from dm0; later rungs carry the previous rung's density.
    let init_guess_density = blocks::checked_dm0(dm0, &prep, emol.nelec() as f64, "run_dft")?;
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
    let symmetric = symmetry || irrep_nelec.is_some();
    if symmetric {
        blocks::refuse_symmetric_dft(
            symmetry,
            with_gradient,
            dispersion.is_some(),
            fock_hook.is_some(),
        )?;
    }
    // run_dft's policy is the set of fields below; every `None` keeps the
    // `RhfConfig` default (ferric_scf::options).
    let options = ScfOptions {
        max_iter,
        energy_conv,
        density_conv,
        diis_size: None,
        diis_flavor: None,
        integral_thresh: None,
        k_builder: k_builder.map(str::to_string),
        cosx: Some(cosx),
        // RI-J on by DEFAULT (PySCF's density_fit convention), overridable:
        // a caller comparing against an exact-Coulomb code would otherwise
        // measure the RI-J FITTING ERROR as a ferric defect (PBE/STO-3G vs
        // conventional J: water 0.28, benzene 1.16, 71-atom drug 9.5
        // kcal/mol). `df_j_aux=""` selects conventional J.
        df_j_aux: Some(ks_aux(df_j_aux)),
        // RI-K only matters for ω = 0 hybrids and HF (`solve_rhf` skips the
        // DfK for a pure functional or an RSH). RIJCOSX: under
        // k_builder="cosx" COSX replaces RI-K, so no RI-K default applies (an
        // explicit df_k_aux next to COSX is a conflict the SCF refuses).
        df_k_aux: df_k_aux_or_default(
            k_builder,
            df_k_aux.map(AuxBasis::parse),
            Some(AuxBasis::default_jk()),
        ),
        ri_j_precision,
        ri_k_precision,
        // An explicit level_shift (0 included) turns off the meta-GGA
        // automatic shift.
        level_shift,
        mom_after_iter,
        mom_reference: None,
        smearing_sigma: None,
        soscf: None,
        guess: None,
        guess_basis,
        init_guess_density,
        external_potential: build_external_potential(point_charges, external_field),
        pcm: None,
        three_index_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        stability: None,
        symmetry: symmetric.then_some(true),
        irrep_nelec: irrep_nelec.map(|m| m.into_iter().collect()),
        eri: None,
        relativistic,
        aurora: None,
        trah_trigger: None,
        xc: Some(xc_name),
        dft_grid,
    };
    let mut cfg = options.into_rhf_config().py()?;
    if symmetric {
        return solve_symmetric_outcome(py, "run_dft", &ctx, &emol, &prep, op, &bounds, &cfg);
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
        ferric_scf::gradient::preflight_cosx_restricted(&cfg)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        cfg = ferric_scf::gradient::gradient_task_config(&cfg).into_owned();
    }
    // MBD@rsSCS's exact gradient needs the Z-vector relaxation term: refuse an
    // unsupported setup before the SCF and the free-atom solves, not after.
    if with_gradient && matches!(dispersion, Some(DispersionSpec::Mbd(_))) {
        if let Some(r) = ferric_scf::zvector_ks::unsupported_reason(&cfg) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
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
    // below is an owned Rust value (`emol` is a clone; `ctx`/`prep`/`bounds`/
    // `ladder` are built above), so nothing Python-borrowed crosses the
    // closure boundary.
    let fock_hooked = fock_hook.is_some();
    let (rhf, rung_cfg) = blocks::solve_ks_ladder(
        py,
        &ctx,
        &emol,
        &prep,
        op,
        &bounds,
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
                        &ctx,
                        &emol,
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
            &emol,
            &prep,
            &basis_set.inner,
            op,
            &bounds,
            &cfg,
            &rhf,
        )
        .map_err(make_err)?;
        // The hook's explicit term (its Pulay term is already in `g`).
        if let Some(h) = &hook_grad {
            g = h.add_to(py, g, &rhf, false, &emol)?;
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
        xc_mol: emol,
        xc_basis: basis_set.inner.clone(),
        xc_cache: std::sync::OnceLock::new(),
        xc_lock: std::sync::Mutex::new(()),
        density_data: rhf.density_total.clone(),
        gradient_data,
        scf_data: rhf,
        scf_config: rung_cfg,
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
    let refuse = |m: &str| Err(pyo3::exceptions::PyValueError::new_err(m.to_string()));
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

/// Number of points in the main KS integration grid `run_dft` would build for
/// `mol` with the same `grid_*` kwargs (same resolver, same builder). Lets a
/// caller see what pruning or a weight cutoff saves on their own molecule
/// without running an SCF. Counts via
/// `ferric_dft::grid::atomic_grid_point_count`: without `grid_weight_cut` it
/// resolves the same per-shell angular orders as the builder but never
/// generates points or Becke weights; WITH a cutoff it must evaluate every
/// point's Becke weight (streamed, no grid allocated), so it costs about as
/// much as building the grid.
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

/// Number of points in the COSX exchange grid `cosx_grid=` describes for
/// `mol` (`None` = the default COSX SCF grid), without building it.
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
            Some(inner) if inner.trim().is_empty() => Err(pyo3::exceptions::PyValueError::new_err(
                format!("dispersion=\"{prefix})\" names no functional"),
            )),
            Some(inner) => Ok(Some(inner.trim().to_string())),
        }
    };
    let value_err =
        |e: ferric_core::FerricError| pyo3::exceptions::PyValueError::new_err(e.to_string());
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
    Err(pyo3::exceptions::PyValueError::new_err(format!(
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

/// Alias under the spec's canonical name. Same surface as `run_dft`.
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

/// Shared body of [`run_uks`] / [`run_roks`]: the KS-DFT SCF config of
/// `run_dft` (RI-J/RI-K default `def2-universal-jkfit`, the production grid)
/// handed to `solve_uhf` / `solve_rohf` with `RhfConfig::xc` set, which is
/// the route the CLI takes for `kind = "ksdft"` on an open-shell molecule
/// (UKS) and `kind = "rohf"` with `[dft] functional` (ROKS). Result, gradient
/// and certificate are the `UhfResult` ones.
#[allow(clippy::too_many_arguments)]
fn run_open_ks(
    py: Python<'_>,
    kind: OpenKs,
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
    dm0: Option<&Bound<'_, PyAny>>,
    relativistic: Option<&str>,
) -> PyResult<PyUhfResult> {
    let who = kind.who();
    refuse_dm0_with_guess(who, dm0.is_some(), guess, None)?;
    let use_sad_guess = parse_guess(guess)?;
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    // Refuse a molecule whose electron count cannot carry the multiplicity
    // before any integral work (an odd count with multiplicity 1, ...).
    let n_spin = spin_counts(&emol, who)?;
    let prep = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let dm0 = blocks::checked_dm0_spin(dm0, &prep, n_spin, who)?;
    // Parsed in the order the pre-`ScfOptions` construction evaluated them.
    let relativistic = crate::scf::parse_relativistic(relativistic)?;
    let ri_j_precision = parse_ri_j_precision(ri_j_precision)?;
    let ri_k_precision = parse_ri_k_precision(ri_k_precision)?;
    let dft_grid =
        resolve_dft_grid_with_cut(grid_radial, grid_angular, grid_prune, grid_weight_cut)?;
    // run_uks / run_roks policy: run_dft's KS defaults minus what these entry
    // points do not offer (COSX knobs, symmetry, guess_basis). `run_roks`
    // passes `guess = None`, i.e. MINAO, the default.
    let options = ScfOptions {
        max_iter,
        energy_conv,
        density_conv,
        diis_size: None,
        diis_flavor: None,
        integral_thresh: None,
        k_builder: k_builder.map(str::to_string),
        cosx: None,
        // RI-J / RI-K on by default for a KS SCF, as in run_dft and the CLI.
        // No RIJCOSX rule here: these entry points take no COSX knobs, and
        // the RI-K default has always applied whatever `k_builder` says.
        df_j_aux: Some(ks_aux(df_j_aux)),
        df_k_aux: Some(ks_aux(df_k_aux)),
        ri_j_precision,
        ri_k_precision,
        level_shift,
        mom_after_iter,
        mom_reference: None,
        smearing_sigma: None,
        soscf: None,
        guess: Some(crate::scf::guess_from_use_sad(use_sad_guess)),
        guess_basis: None,
        init_guess_density: None,
        external_potential: build_external_potential(point_charges, external_field),
        pcm: None,
        three_index_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        stability: None,
        symmetry: None,
        irrep_nelec: None,
        eri: None,
        relativistic,
        aurora: None,
        trah_trigger: None,
        xc: Some(functional.to_string()),
        dft_grid,
    };
    let mut config = options.into_rhf_config().py()?;
    let (solver, gradient_kind) = match kind {
        OpenKs::Uks => (blocks::OpenShell::Uhf, ScfGradientKind::Uhf),
        OpenKs::Roks => (blocks::OpenShell::Rohf, ScfGradientKind::Rohf),
    };
    if with_gradient {
        config = scf_gradient_config(who, gradient_kind, &config, None)?;
    }
    let ctx = ParallelContext::default();
    let (r, _) = blocks::solve_open_shell(
        py,
        solver,
        &ctx,
        &emol,
        &prep,
        op,
        &bounds,
        &config,
        None,
        dm0.as_ref(),
    )?;
    let gradient = with_gradient
        .then(|| scf_gradient(py, gradient_kind, &emol, &prep, &bounds, &config, &r))
        .transpose()?;
    let mut result = PyUhfResult::from_scf(r, &emol, &config, &prep, false);
    result.gradient_data = gradient;
    Ok(result)
}

/// Unrestricted Kohn-Sham (open-shell DFT): `solve_uhf` with the functional
/// `functional` (required; no silent LDA default). α/β electron counts come
/// from the molecule's `charge` and `multiplicity`; a singlet is allowed (the
/// unrestricted solution of a closed shell). Returns the `UhfResult` of
/// `run_uhf`: `energy`, per-spin densities, orbital energies, MO coefficients
/// and occupations, `certificate`, and `gradient()` when
/// `with_gradient=True` (the analytic UKS gradient, `ks_gradient_uks`, at the
/// RI-J precision `f64` unless `ri_j_precision` says otherwise).
///
/// The SCF settings are `run_dft`'s: RI-J and RI-K through
/// `def2-universal-jkfit` unless `df_j_aux` / `df_k_aux` say `""` (exact),
/// the production Becke-Lebedev grid unless `grid_*` say otherwise, the
/// default `RhfConfig` convergence gates. Range-separated hybrids and
/// meta-GGAs run (energy and gradient); for a VV10 functional (`wB97X-V`) the
/// gradient agrees with finite differences of the energy to 6e-6 Ha/Bohr
/// (NH2/6-31G), against ~1e-7 for the others. `run_dft` stays the
/// closed-shell entry point and refuses `multiplicity > 1`. Not offered here: dispersion,
/// COSX, symmetry, `fock_hook`.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, functional, k_builder=None, with_gradient=false,
    max_iter=None, energy_conv=None, density_conv=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None, memory_budget_gb=None,
    df_j_aux=None, df_k_aux=None, ri_k_precision=None, ri_j_precision=None,
    grid_radial=None, grid_angular=None, grid_prune=None, grid_weight_cut=None,
    guess=None, dm0=None, relativistic=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_uks(
    py: Python<'_>,
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
    dm0: Option<&Bound<'_, PyAny>>,
    relativistic: Option<&str>,
) -> PyResult<PyUhfResult> {
    run_open_ks(
        py,
        OpenKs::Uks,
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
    )
}

/// Restricted open-shell Kohn-Sham: `solve_rohf` with the functional
/// `functional` (required). One set of spin-pure MOs: `nocc_open =
/// multiplicity - 1` singly occupied orbitals, the rest doubly occupied
/// (Guest-Saunders coupling as in `run_rohf`). Same result, gradient
/// (`ks_gradient_roks`), settings and exclusions as `run_uks`, without the
/// `guess=` knob (ROKS starts from the default MINAO guess).
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, functional, k_builder=None, with_gradient=false,
    max_iter=None, energy_conv=None, density_conv=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None, memory_budget_gb=None,
    df_j_aux=None, df_k_aux=None, ri_k_precision=None, ri_j_precision=None,
    grid_radial=None, grid_angular=None, grid_prune=None, grid_weight_cut=None,
    dm0=None, relativistic=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_roks(
    py: Python<'_>,
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
    dm0: Option<&Bound<'_, PyAny>>,
    relativistic: Option<&str>,
) -> PyResult<PyUhfResult> {
    run_open_ks(
        py,
        OpenKs::Roks,
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
        df_j_aux,
        df_k_aux,
        ri_k_precision,
        ri_j_precision,
        grid_radial,
        grid_angular,
        grid_prune,
        grid_weight_cut,
        None,
        dm0,
        relativistic,
    )
}

/// Register the restricted-open-shell and open-shell Kohn-Sham entry points
/// (`run_rohf`, `run_uks`, `run_roks`).
pub(crate) fn register_open_shell_scf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_rohf, m)?)?;
    m.add_function(wrap_pyfunction!(run_uks, m)?)?;
    m.add_function(wrap_pyfunction!(run_roks, m)?)?;
    Ok(())
}

/// Grimme D3(BJ) dispersion energy for a molecule, in Hartree.
///
/// Standalone: this does no SCF, so it is the cheap way to add dispersion to an
/// energy computed elsewhere, or to inspect the correction on its own.
///
/// `functional` names the XC functional whose published D3(BJ) damping
/// parameters to use -- the correction is FITTED per functional, so this is
/// required, not a convenience. An unknown functional raises rather than
/// falling back to a default, and an element outside the D3 parameterisation
/// raises rather than being silently skipped.
///
/// `s9` scales the Axilrod-Teller-Muto three-body term. The default 0.0 (off)
/// is the reference default for a named functional (simple-dftd3 without
/// `--atm`); `s9=1.0` gives D3(BJ)-ATM. A non-finite `s9` raises.
#[pyfunction]
#[pyo3(signature = (mol, functional, s9 = 0.0))]
pub(crate) fn d3bj_energy(mol: &PyMolecule, functional: &str, s9: f64) -> PyResult<f64> {
    if !s9.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
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
#[pyclass]
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
        (None, Some(f)) => MbdRsscsConfig::for_functional(f)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        (Some(_), Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mbd_rsscs_energy: pass either beta or functional, not both",
            ))
        }
        (None, None) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mbd_rsscs_energy: beta is functional dependent — pass beta=... or \
                 functional=... (PBE, PBE0, HSE06)",
            ))
        }
    };
    let atoms = &mol.inner.atoms;
    if let Some(i) = atoms.iter().position(|a| a.ghost) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "mbd_rsscs_energy: atom {i} is a ghost; MBD needs a real atom (and a volume \
             ratio) at every center — remove ghost atoms first"
        )));
    }
    if volume_ratios.len() != atoms.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
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
