//! RHF, UHF, ROHF drivers: moved verbatim from `lib.rs`.

use super::*;

// ── RHF ──

/// Default SCF iteration cap of every Python `run_*` that takes `max_iter`:
/// the Rust `RhfConfig` default itself, so the two cannot drift apart
/// (they did: Python was 100, Rust 200, until 2026-10-13).
pub(crate) fn default_scf_max_iter() -> usize {
    RhfConfig::default().max_iter
}

/// Emit a `RuntimeWarning`. If warnings are turned into errors (`-W error`)
/// the warn call returns Err; that is IGNORED on purpose so a diagnostic can
/// never change control flow (the result is never lost, the SCF still runs).
pub(crate) fn emit_runtime_warning(py: Python<'_>, msg: String) {
    let msg = std::ffi::CString::new(msg.replace('\0', " ")).expect("NUL removed");
    let _ = PyErr::warn(
        py,
        &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
        &msg,
        1,
    );
}

/// Resolve a renamed keyword that is kept as a deprecated alias: the old name
/// (`old_name`, a released spelling) still works but warns; passing both is a
/// `ValueError`. Returns the effective value.
pub(crate) fn alias_kwarg<T>(
    py: Python<'_>,
    driver: &str,
    old_name: &str,
    new_name: &str,
    old: Option<T>,
    new: Option<T>,
) -> PyResult<Option<T>> {
    match (old, new) {
        (Some(_), Some(_)) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{driver}: `{old_name}` is a deprecated alias of `{new_name}`; pass only `{new_name}`"
        ))),
        (Some(v), None) => {
            let msg = std::ffi::CString::new(format!(
                "{driver}: `{old_name}` is deprecated, use `{new_name}`"
            ))
            .expect("no NUL in a formatted name");
            // Not swallowed: under `-W error` the deprecation must raise.
            PyErr::warn(
                py,
                &py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
                &msg,
                2,
            )?;
            Ok(Some(v))
        }
        (None, v) => Ok(v),
    }
}

/// Warn, once and before the SCF starts, when a USER-SUPPLIED `energy_conv` is
/// below the estimated noise floor of `dE` for this system (issue #410). Purely
/// advisory: the threshold is not clamped and the SCF is unchanged. The default
/// (`energy_conv=None`) never warns.
pub(crate) fn warn_energy_conv_noise_floor(
    py: Python<'_>,
    fname: &str,
    family: ferric_scf::noise_floor::SolverFamily,
    user_set: bool,
    mol: &ferric_core::mol::Molecule,
    prep: &PreparedBasis,
    config: &RhfConfig,
) {
    if !user_set {
        return;
    }
    if let Some(text) =
        ferric_scf::noise_floor::energy_conv_warning(mol, prep.nbasis(), config, family)
    {
        emit_runtime_warning(py, format!("{fname}: {text}"));
    }
}

/// After a hooked SCF: warn when the hook's noise-floor term (`hook_term`,
/// [`ferric_scf::noise_floor::hook_floor_term`]) puts a USER-SUPPLIED
/// `energy_conv` below the floor that [`warn_energy_conv_noise_floor`]
/// judged it against before the SCF. A zero term never warns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn warn_hook_noise_floor(
    py: Python<'_>,
    fname: &str,
    family: ferric_scf::noise_floor::SolverFamily,
    user_set: bool,
    mol: &ferric_core::mol::Molecule,
    prep: &PreparedBasis,
    config: &RhfConfig,
    hook_term: f64,
) {
    if !user_set {
        return;
    }
    if let Some(text) = ferric_scf::noise_floor::hook_energy_conv_warning(
        mol,
        prep.nbasis(),
        config,
        family,
        hook_term,
    ) {
        emit_runtime_warning(py, format!("{fname}: {text}"));
    }
}

/// Human-readable SCF exit line: exit reason, iterations, final residuals vs
/// thresholds and the unmet gate criteria. Shared by `RhfResult.convergence_summary`
/// and the non-convergence `RuntimeWarning` of `run_rhf`.
pub(crate) fn scf_exit_summary(r: &ScfResult) -> String {
    let head = format!("SCF exit {:?} after {} iterations", r.exit, r.iterations);
    match r.convergence {
        Some(c) => format!("{head}; last {}", c.describe()),
        None => format!("{head}; no per-iteration convergence record"),
    }
}

/// The stationarity tolerance the solver judged `one_step_energy_drop`
/// against ([`ferric_scf::result::stationarity_tol`]): the energy and density
/// thresholds the SCF ran with, the basis size and the widest orbital-energy
/// span of either spin. `None` without a convergence record.
pub(crate) fn scf_stationarity_tol(r: &ScfResult) -> Option<f64> {
    let c = r.convergence?;
    let span = |eps: &[f64]| {
        if eps.is_empty() {
            0.0
        } else {
            let (lo, hi) = eps
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &e| {
                    (l.min(e), h.max(e))
                });
            hi - lo
        }
    };
    let span = span(&r.eps_alpha).max(r.eps_beta.as_deref().map_or(0.0, span));
    Some(ferric_scf::result::stationarity_tol(
        c.energy_conv,
        c.density_conv,
        r.mos_alpha.nrows(),
        span,
    ))
}

/// Result of a closed-shell RHF (or `run_ksdft` KS-DFT) calculation.
#[pyclass]
#[pyo3(name = "RhfResult")]
pub(crate) struct PyRhfResult {
    /// Total SCF energy in Hartree (electronic + nuclear repulsion).
    #[pyo3(get)]
    pub(crate) energy: f64,
    /// Whether the SCF met both the energy and density convergence thresholds.
    #[pyo3(get)]
    pub(crate) converged: bool,
    /// Number of SCF iterations run.
    #[pyo3(get)]
    pub(crate) iterations: usize,
    /// Number of unique two-electron integral quartets actually computed
    /// (reflects Schwarz/QQR screening; lower than the naive N^4 count).
    #[pyo3(get)]
    pub(crate) computed_quartets: usize,
    pub(crate) density_data: Array2<f64>,
    pub(crate) orbital_energies_data: Vec<f64>,
    pub(crate) scf_data: ScfResult,
    /// The SCF settings this result was solved with. `hirshfeld_charges`
    /// solves its free-atom proatoms with the same settings, as ferric-cli does.
    pub(crate) scf_config: RhfConfig,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
    /// `dE/dR` (natoms x 3, Hartree/Bohr) when solved with `with_gradient=True`.
    pub(crate) gradient_data: Option<Array2<f64>>,
}

#[pymethods]
impl PyRhfResult {
    /// Analytic nuclear gradient `dE/dR` as an (natoms, 3) array in
    /// Hartree/Bohr, rows in the Molecule's atom order; `None` unless the
    /// result came from `run_rhf(..., with_gradient=True)`.
    fn gradient<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.gradient_data
            .as_ref()
            .map(|g| PyArray2::from_array(py, g))
    }
    /// The converged AO-basis density matrix as a 2D numpy array (n_bf × n_bf).
    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.density_data)
    }
    /// Molecular orbital energies (Hartree) as a 1D numpy array, ascending order.
    fn orbital_energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.orbital_energies_data.clone())
    }
    /// MO coefficient matrix C (n_bf × n_mo), column k = MO k in ascending
    /// energy order — the same matrix the Rust post-HF drivers consume.
    fn mo_coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.scf_data.mos_alpha)
    }
    /// The COSX final-grid pass (`cosx_final_pass=` / `cosx_final_grid=`):
    /// `{"e_scf_grid", "e_final", "npts_scf_grid", "npts_final_grid",
    /// "gradient_differentiates"}`, or `None` when no pass ran. `energy` is
    /// `e_final` when it ran.
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
    /// Precision provenance of this SCF: `{"precision": "f64", "counter_scope": "process", "kernels": []}`
    /// for an all-f64 run; otherwise one entry per mixed-precision kernel that
    /// was allowed or ran (`kernel`, `requested`, `mode`, `device` (`None`
    /// unless `calls_mixed > 0`), `calls_mixed`, `calls_f64_fallback`,
    /// `ladder`, `error_bound`). `counter_scope = "process"`: the counts are
    /// process-wide, so a concurrent calculation in this process is counted
    /// too (sequential runs are exact). See `configure_precision`.
    #[getter]
    fn precision(&self, py: Python<'_>) -> PyResult<PyObject> {
        precision_dict(py, &self.scf_data.precision)
    }
    /// Validation certificate of this result (the dict form of
    /// `ferric_core::certificate::Certificate`): `capability`, `build`,
    /// `grade` (`"proven"`, `"proven_narrow"`, `"smoke"`, `"spike"`, `"stub"`
    /// or `"ungraded"`, from the `method.kind` matrix of the validation page;
    /// an unconverged SCF is `"ungraded"`), `grade_source`, `measured_error`,
    /// `approximations` (`[{"name", "detail"}]`: RI-J/RI-K, LinK/COSX, XC
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
    /// Validation certificate of this result (the dict form of
    /// `ferric_core::certificate::Certificate`): `capability`, `build`,
    /// `grade` (`"proven"`, `"proven_narrow"`, `"smoke"`, `"spike"`, `"stub"`
    /// or `"ungraded"`, from the `method.kind` matrix of the validation page;
    /// an unconverged SCF is `"ungraded"`), `grade_source`, `measured_error`,
    /// `approximations` (`[{"name", "detail"}]`: RI-J/RI-K, LinK/COSX, XC
    /// grid, level shift, ECP, embedding, solvation), `precision` and
    /// `warnings` (one per element, basis or functional outside the solver's
    /// energy anchor among them; those do not change the grade).
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
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
    /// SCF exit reason: the `ScfExit` variant name (`"Converged"`, `"Plateau"`,
    /// `"Stalled"`, `"Diverged"`, `"MaxIter"`). `converged=False` with
    /// `"MaxIter"` means the iteration cap (`max_iter`) was reached; the final
    /// residuals below say how far from the thresholds the run was.
    #[getter]
    fn exit_reason(&self) -> String {
        format!("{:?}", self.scf_data.exit)
    }
    /// |E - E_prev| (Hartree) as judged by the convergence gate at the final
    /// iteration (lags the returned density by one step at MaxIter), or `None` if this
    /// result carries no convergence record.
    #[getter]
    fn final_de(&self) -> Option<f64> {
        self.scf_data.convergence.map(|c| c.de)
    }
    /// RMS density change the gate judged at the final iteration (the primary gate,
    /// compared against `density_conv`), or `None`.
    #[getter]
    fn final_dp_rms(&self) -> Option<f64> {
        self.scf_data.convergence.map(|c| c.dp_rms)
    }
    /// Max-element density change the gate judged at the final iteration (gated at
    /// `10 * density_conv`), or `None`.
    #[getter]
    fn final_dp_max(&self) -> Option<f64> {
        self.scf_data.convergence.map(|c| c.dp_max)
    }
    /// Max |FDS - SDF| (DIIS commutator) at the last iteration. Diagnostic
    /// only: it is never a convergence gate. `None` if not recorded.
    #[getter]
    fn final_diis_err_max(&self) -> Option<f64> {
        self.scf_data.convergence.map(|c| c.err_max)
    }
    /// `(energy_conv, density_conv)` this SCF was judged against, or `None`.
    #[getter]
    fn conv_thresholds(&self) -> Option<(f64, f64)> {
        self.scf_data
            .convergence
            .map(|c| (c.energy_conv, c.density_conv))
    }
    /// Names of the gate criteria the final iteration did NOT meet, a subset of
    /// `["dp_rms", "dp_max", "dE", "err_max"]` (empty when all were met), or
    /// `None` if no convergence record. `"err_max"` appears only when
    /// ADIIS/EDIIS stalled: the density settled while the DIIS error was still
    /// above `diis_switch_thresh`, the exit was refused and the run restarted on
    /// Pulay DIIS (that restart is also named in `convergence_summary`). A run
    /// can have `converged=False` with an empty list only when it stopped for
    /// another reason (`exit_reason`).
    #[getter]
    fn unmet_criteria(&self) -> Option<Vec<&'static str>> {
        self.scf_data.convergence.map(|c| c.unmet())
    }
    /// One-line human summary of the SCF exit: reason, iteration count and the
    /// final residuals against their thresholds. This is the text of the
    /// `RuntimeWarning` `run_rhf` emits when the SCF does not converge.
    #[getter]
    fn convergence_summary(&self) -> String {
        scf_exit_summary(&self.scf_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "RhfResult(energy={:.10}, converged={})",
            self.energy, self.converged
        )
    }
    fn __str__(&self) -> String {
        format!(
            "RHF Energy: {:.10} Ha (converged: {}, {} iterations)",
            self.energy, self.converged, self.iterations
        )
    }
}

/// `trah_trigger` must be a positive finite DIIS-error threshold.
pub(crate) fn check_trah_trigger(t: Option<f64>) -> PyResult<Option<f64>> {
    match t {
        Some(v) if !(v.is_finite() && v > 0.0) => Err(pyo3::exceptions::PyValueError::new_err(
            format!("trah_trigger must be a positive number, got {v}"),
        )),
        other => Ok(other),
    }
}

/// Apply the opt-in `aurora=` / `trah_trigger=` kwargs (both off when omitted).
pub(crate) fn apply_accelerators(
    config: &mut RhfConfig,
    aurora: Option<bool>,
    trah_trigger: Option<f64>,
) -> PyResult<()> {
    config.aurora.enabled = aurora.unwrap_or(false);
    config.trah_trigger = check_trah_trigger(trah_trigger)?;
    Ok(())
}

/// The non-convergence RuntimeWarning of `run_rhf` (no control-flow change).
pub(crate) fn warn_rhf_not_converged(
    py: Python<'_>,
    r: &ferric_scf::result::ScfResult,
    config: &RhfConfig,
) {
    if !r.converged {
        // No control-flow change: the caller still gets the best-effort
        // `RhfResult` with `converged=False`. The warning only names the exit,
        // iteration count and unmet criteria. If warnings are turned into errors
        // (`-W error`) the warn call returns Err; that is IGNORED on purpose so
        // the result is never lost.
        let msg = format!(
            "run_rhf: SCF did not converge: {}. Raise max_iter (this call used {}; default {}), or \
             pass level_shift / soscf / a different guess; see RhfResult.unmet_criteria.",
            scf_exit_summary(r),
            config.max_iter,
            default_scf_max_iter()
        );
        emit_runtime_warning(py, msg);
    }
}

/// Closed-shell RHF (also UHF/ROHF convergence aids apply to the open-shell
/// drivers). Exposes the full SCF knob set, matching the CLI `[scf]` TOML section
/// for parity — same names, same defaults, same units.
///
/// SCF controls and defaults (run_rhf / run_uhf / run_rohf):
///   max_iter        iteration cap, default 200 (the Rust `RhfConfig`
///                   default, read from it; was 100 before 2026-10-13). Reaching it returns `converged=False`
///                   (RHF; a RuntimeWarning names the unmet criteria and never raises, even under -W error), so a
///                   slow-but-draining SCF fails on the cap alone.
///   density_conv    RMS density change gate, default 1e-6. The max-element
///                   change is gated at 10x this. THE primary criterion.
///   energy_conv     |dE| per iteration, default 1e-3: a loose "not still
///                   descending" sanity bound, not a target. A value below
///                   the estimated noise floor of dE for the system (about
///                   0.0125*integral_thresh*nbf^2.25 Ha for closed-shell exact J/K,
///                   a few eps*|E| otherwise) is legal but is met only by chance;
///                   run_rhf / run_uhf / run_rohf then emit one RuntimeWarning
///                   before the SCF starts (never raised, nothing is clamped).
///   diis_size       DIIS subspace size, default 8.
/// After an RHF run, `exit_reason`, `final_dp_rms`, `final_dp_max`, `final_de`,
/// `final_diis_err_max`, `conv_thresholds`, `unmet_criteria` and
/// `convergence_summary` report where the SCF stopped.
///
/// Convergence aids:
///   level_shift     virtual-virtual block shift in Hartree (open-shell; 0.2 is a
///                   useful default for OH-like doublets at LDA/PBE). 0 = off.
///   mom_after_iter  Maximum-Overlap Method: pin the occupied set by AO-overlap
///                   after this many DIIS iters. 0 = aufbau throughout. Fixes
///                   occupied-set flip-flop non-convergence.
///   aurora          False (default). True turns on the AURORA auxiliary-curvature
///                   accelerator (closed-shell, RHF only): same fixed point as
///                   DIIS, measured as a speed-up (fewer J/K builds), not a
///                   convergence rescue. Same as the CLI `[scf] aurora`.
///   trah_trigger    None (default) = off. A positive DIIS-error threshold turns
///                   on the trust-region augmented-Hessian (TRAH) solver once the
///                   DIIS error drops below it (e.g. 1e-2). A convergence RESCUE:
///                   it wins iterations but costs wall time. Same as the CLI
///                   `[scf] trah_trigger`.
/// Fock builders:
///   k_builder       "direct" (default), "link" (LinK exchange) or "cosx"
///                   (seminumerical exchange, Coulomb operator only; CLI
///                   defaults for grid/fit/kernel — not exposed here).
///                   Honoured by RHF, UHF and ROHF alike.
///   df_j_aux        auxiliary basis name for density-fitted Coulomb (RI-J).
///   df_k_aux        auxiliary basis name for density-fitted exchange (RI-K);
///                   should be a JK-fit basis, not an MP2-fit basis.
///                   For either, "" / "exact" / "none" / "off" /
///                   "conventional" select conventional four-centre integrals
///                   (the same spellings run_dft accepts). Omitted = no DF.
///   ri_j_precision  storage of the RI-J three-index tensor: "f32" (default;
///                   f64 accumulation, half the memory) or "f64" (bit-identical
///                   to the default before 2026-11-07). Anything else raises
///                   ValueError. Ignored without RI-J. The result's
///                   `ri_j_precision` says what was used. Same as the CLI
///                   `[scf] ri_j_precision`; drivers with more than one
///                   iterative stage spell it `scf_ri_j_precision`. The
///                   geometry/frequency drivers default to "f64".
///   ri_k_precision  storage of the RI-K dressed three-index tensor: "f64" or
///                   "f32" (strict; None = "f64"). Both keep the
///                   packed triangle and contract K in f64; "f64" is bit-for-bit
///                   the former square f64 tensor, "f32" rounds each stored
///                   element once. Same as the CLI `[scf] ri_k_precision`.
/// External perturbation:
///   point_charges   list of (q, x, y, z) classical point charges (Hartree
///                   atomic units; coordinates in Bohr) added to the one-electron
///                   Hamiltonian and nuclear repulsion. None/empty = no charges.
///                   The ORDER of the list changes the one-electron potential
///                   only by summation roundoff (3e-15..1e-14 relative on 6458
///                   charges), and ferric does not reorder it. The converged
///                   energy moves by <= ~6e-12 Ha. A different SCF iteration
///                   count (64 vs 111) was seen on rc7 for a 6458-charge field
///                   at energy_conv=1e-10; on main it appears only when the
///                   closed-shell incremental Fock update is forced on
///                   (FERRIC_SCF_INCREMENTAL=1) below its noise floor, where
///                   the order also moves the energy by a few 1e-9 Ha (up to
///                   8e-9 measured). Main turns
///                   that update off when energy_conv < 1e4*integral_thresh or
///                   density_conv < 1e3*integral_thresh (1e-8 / 1e-9 at the
///                   default integral_thresh), and the energy_conv warning
///                   names the floor. The iteration-count dependence was
///                   observed and explained only for this function's closed-shell
///                   direct-J/K path; run_uhf and run_rohf use a separate
///                   open-shell incremental path and run_dft's default RI-J does
///                   not use the direct incremental path, so for those the
///                   order changes the potential by roundoff (same as above) and
///                   iteration counts were not measured.
///   smeared_charges list of (q, x, y, z, width) Gaussian-smeared classical
///                   charges (Hartree atomic units; coordinates and width in
///                   Bohr; width = 0 is a point charge, but use
///                   `point_charges=` for those instead). None/empty = no
///                   smeared charges.
///   external_field  uniform (Ex, Ey, Ez) electric field in Hartree atomic units.
///                   None = no field.
///   solvent         IEF-PCM implicit solvation: a dielectric constant
///                   (`solvent=78.4`) or a name (`solvent="water"`). None =
///                   vacuum. Without it, a ligand and its pocket each leave
///                   water for free, so burying an ionized group costs
///                   nothing -- on danuglipron/GLP-1R that overshoots the
///                   experimental binding free energy by 4.5x. An
///                   unrecognised name ERRORS rather than falling back to
///                   vacuum.
///   pcm_lebedev_order  tesserae per atomic sphere (a positive-weight
///                   Lebedev order: any grid_angular order except 74, 230,
///                   266; default 110). Lower is faster and coarser.
/// State selection:
///   stability_descent False (default). When True, check the converged RHF
///                   solution's stability and, if the INTERNAL (singlet)
///                   channel is a SADDLE of the orbital Hessian, follow the
///                   downhill eigenvector and re-converge, keeping the lowest
///                   state (`RhfConfig::check_stability` +
///                   `scf_stability_descent`).
///                   Costs one Davidson eigensolve per solve plus one SCF per
///                   descent taken. Needed where the default guess lands on a
///                   saddle: N2 at 1.6 Å/def2-SVP (19 mHa above the minimum).
///                   Same as the CLI `[scf] stability_descent`.
///                   Setting it also enables the EXTERNAL (RHF→UHF, triplet)
///                   check, whose verdict is PRINTED to stderr but is NOT
///                   descended along: that instability breaks spin symmetry,
///                   so no restricted re-convergence can reach the lower
///                   state — the remedy it names is to run `run_uhf` seeded
///                   from the triplet eigenvector. Stretched geometries are
///                   routinely internally stable and externally unstable
///                   (water at r(OH) = 2.0 Å / 6-31G: +1.97e-2 vs −3.07e-1).
/// Parse the `ri_k_precision` kwarg (strict: `"f32"` or `"f64"`; `None` =
/// the library default, `ferric_scf::df_k::RiKPrecision::DEFAULT`).
pub(crate) fn parse_ri_k_precision(
    s: Option<&str>,
) -> PyResult<Option<ferric_scf::df_k::RiKPrecision>> {
    s.map(ferric_scf::df_k::RiKPrecision::parse_config_str)
        .transpose()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (
    mol, basis_set,
    max_iter=None, energy_conv=None, density_conv=None, diis_size=None,
    integral_thresh=None, k_builder=None, df_j_aux=None, df_k_aux=None, ri_k_precision=None,
    level_shift=None, mom_after_iter=None,
    guess=None, diis=None, smearing_sigma=None, soscf=None,
    point_charges=None, external_field=None, smeared_charges=None,
    memory_budget_gb=None, solvent=None, pcm_lebedev_order=None,
    stability_descent=None,
    cosx_grid=None, cosx_final_pass=None, cosx_final_grid=None, cosx_overlap_fit=None,
    aurora=None, trah_trigger=None, symmetry=false, irrep_nelec=None,
    fock_hook=None,
    guess_basis=None, guess_projection=None, ri_j_precision=None, dm0=None,
    with_gradient=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_rhf(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    diis_size: Option<usize>,
    integral_thresh: Option<f64>,
    k_builder: Option<&str>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    ri_k_precision: Option<&str>,
    level_shift: Option<f64>,
    mom_after_iter: Option<usize>,
    guess: Option<&str>,
    diis: Option<&str>,
    smearing_sigma: Option<f64>,
    soscf: Option<bool>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    smeared_charges: Option<Vec<(f64, f64, f64, f64, f64)>>,
    memory_budget_gb: Option<f64>,
    solvent: Option<&Bound<'_, PyAny>>,
    pcm_lebedev_order: Option<usize>,
    stability_descent: Option<bool>,
    cosx_grid: Option<&Bound<'_, PyAny>>,
    cosx_final_pass: Option<bool>,
    cosx_final_grid: Option<&Bound<'_, PyAny>>,
    cosx_overlap_fit: Option<bool>,
    aurora: Option<bool>,
    trah_trigger: Option<f64>,
    symmetry: bool,
    irrep_nelec: Option<std::collections::BTreeMap<String, usize>>,
    fock_hook: Option<&Bound<'_, PyAny>>,
    guess_basis: Option<&str>,
    guess_projection: Option<&str>,
    ri_j_precision: Option<&str>,
    dm0: Option<numpy::PyReadonlyArray2<'_, f64>>,
    with_gradient: bool,
) -> PyResult<PyObject> {
    refuse_dm0_with_guess("run_rhf", dm0.is_some(), guess, guess_basis)?;
    let guess_basis = parse_guess_basis(guess_basis, guess_projection)?;
    // The descent needs a stability verdict to act on, so the kwarg turns on
    // both halves (as run_uhf does).
    let descent = stability_descent.unwrap_or(false);
    // Apply ECP core-electron counts (no-op without an ECP basis) so nelec()
    // gives the valence count; the effective nuclear charge is set inside
    // PreparedBasis::new from basis_set.ecps.
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let prep = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let init_guess_density = blocks::checked_dm0(dm0, &prep, emol.nelec() as f64, "run_rhf")?;
    // Defaults mirror the CLI `[scf]` section (config.rs) so CLI and Python agree.
    let mut config = RhfConfig {
        max_iter: max_iter.unwrap_or_else(default_scf_max_iter),
        energy_conv: energy_conv.unwrap_or(1e-3),
        density_conv: density_conv.unwrap_or(1e-6),
        diis_size: diis_size.unwrap_or(8),
        integral_thresh: integral_thresh.unwrap_or(1e-12),
        k_builder: k_builder.map(|s| s.to_string()),
        cosx: resolve_cosx_kwargs(
            k_builder,
            cosx_grid,
            cosx_final_pass,
            cosx_final_grid,
            cosx_overlap_fit,
        )?,
        // Same opt-out spellings as run_dft ("exact"/"none"/"off"/...).
        df_j_aux: df_j_aux.map(ferric_scf::rhf::normalize_df_aux),
        df_k_aux: df_k_aux.map(ferric_scf::rhf::normalize_df_aux),
        ri_j_precision: parse_ri_j_precision(ri_j_precision)?,
        ri_k_precision: parse_ri_k_precision(ri_k_precision)?,
        level_shift: level_shift.unwrap_or(0.0),
        mom_after_iter: mom_after_iter.unwrap_or(0),
        diis_flavor: parse_diis_flavor(diis)?,
        smearing_sigma,
        newton_trigger: if soscf.unwrap_or(false) { 1e-3 } else { 0.0 },
        use_sad_guess: parse_guess(guess)?,
        guess_basis,
        init_guess_density,
        external_potential: build_external_potential_with_smeared(
            point_charges,
            smeared_charges,
            external_field,
        ),
        pcm: build_pcm_config(solvent, pcm_lebedev_order)?,
        // 0 means "unset -> auto" (ferric_scf::rhf::resolve_three_index_budget),
        // so an omitted kwarg preserves the previous auto-detect behaviour.
        three_index_budget_bytes: budget_bytes_from_gb(memory_budget_gb).unwrap_or(0),
        check_stability: descent,
        scf_stability_descent: descent,
        symmetry,
        irrep_nelec: irrep_nelec.map(|m| m.into_iter().collect()),
        ..Default::default()
    };
    apply_accelerators(&mut config, aurora, trah_trigger)?;
    if with_gradient {
        config = scf_gradient_config("run_rhf", ScfGradientKind::Rhf, &config, fock_hook)?;
    }
    let ctx = ParallelContext::default();
    // Release the GIL for the duration of the SCF. Every argument below is an
    // owned Rust value (`emol` is a clone; `prep`/`bounds`/`config` are built
    // above), so nothing Python-borrowed crosses the closure boundary.
    //
    // Without this, a second Python thread cannot even ENTER this function
    // until the first call returns, so independent jobs submitted from a
    // ThreadPoolExecutor serialize at the FFI boundary no matter how many
    // cores rayon could use. The compute itself never touches the interpreter.
    warn_energy_conv_noise_floor(
        py,
        "run_rhf",
        SolverFamily::ClosedShell,
        energy_conv.is_some(),
        &emol,
        &prep,
        &config,
    );
    let user_set = energy_conv.is_some();
    if with_gradient {
        return solve_rhf_with_gradient(py, &ctx, &emol, &prep, &basis_set.inner, &bounds, config);
    }
    solve_rhf_outcome(
        py, &ctx, &emol, &prep, op, &bounds, config, fock_hook, user_set,
    )
}

/// Which SCF a `with_gradient=True` run differentiates.
#[derive(Clone, Copy)]
pub(crate) enum ScfGradientKind {
    Rhf,
    Uhf,
    Rohf,
}

/// The SCF config of a `with_gradient=True` run of `who`, or the
/// `ValueError` for a setup whose analytic gradient would not be the
/// derivative of the reported energy. Refused before the SCF: a `fock_hook`
/// (no gradient sees the hook's term); for RHF, `symmetry=True` /
/// `irrep_nelec` (the result lives in the symmetry frame), `smearing_sigma`
/// and `mom_after_iter` (the closed-shell energy-weighted density takes the
/// lowest `nocc` orbitals fully occupied); a COSX setup the gradient cannot
/// differentiate (`preflight_cosx_*`). A COSX final-grid pass is dropped
/// (`gradient_task_config`), so the energy and gradient share the SCF grid.
pub(crate) fn scf_gradient_config(
    who: &str,
    kind: ScfGradientKind,
    config: &RhfConfig,
    fock_hook: Option<&Bound<'_, PyAny>>,
) -> PyResult<RhfConfig> {
    let refuse = |what: &str| {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{who}: with_gradient=True cannot be combined with {what}"
        )))
    };
    if fock_hook.is_some() {
        return refuse("fock_hook (the analytic gradient does not include the hook's term)");
    }
    if matches!(kind, ScfGradientKind::Rhf) {
        if config.symmetry || config.irrep_nelec.is_some() {
            return refuse("symmetry=True / irrep_nelec (the result lives in the symmetry frame)");
        }
        if config.smearing_sigma.is_some() {
            return refuse(
                "smearing_sigma (the closed-shell gradient assumes integer aufbau occupations)",
            );
        }
        if config.mom_after_iter > 0 {
            return refuse(
                "mom_after_iter (the closed-shell gradient assumes the aufbau occupied set)",
            );
        }
    }
    let preflight = match kind {
        ScfGradientKind::Rhf => ferric_scf::gradient::preflight_cosx_restricted(config),
        ScfGradientKind::Uhf => ferric_scf::gradient::preflight_cosx_unrestricted(config),
        ScfGradientKind::Rohf => ferric_scf::gradient::preflight_cosx_restricted_open(config),
    };
    preflight.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{who}: {e}")))?;
    // A gradient is a geometry task: an unset RI-J precision resolves to f64
    // (the f32 RI-J energy surface is jagged between nearby geometries), an
    // explicit one is kept.
    Ok(ferric_scf::gradient::gradient_task_config(config)
        .into_owned()
        .for_geometry_task())
}

/// The analytic gradient (natoms x 3, Hartree/Bohr) of a converged SCF of
/// `kind`, by the dispatcher that differentiates the J/K, exchange and
/// solvation the SCF used (`restricted_scf_gradient` and its open-shell
/// siblings). An unconverged SCF raises: its gradient is not the derivative
/// of its energy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scf_gradient(
    py: Python<'_>,
    kind: ScfGradientKind,
    mol: &Molecule,
    prep: &PreparedBasis,
    basis: &ferric_core::basis::BasisSet,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    r: &ScfResult,
) -> PyResult<Array2<f64>> {
    use ferric_scf::gradient::{
        restricted_open_scf_gradient, restricted_scf_gradient, unrestricted_scf_gradient,
    };
    if !r.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: r.iterations,
            last_energy: r.energy,
        }));
    }
    let op = Operator::coulomb();
    py.allow_threads(|| match kind {
        ScfGradientKind::Rhf => restricted_scf_gradient(mol, prep, basis, op, bounds, config, r),
        ScfGradientKind::Uhf => unrestricted_scf_gradient(mol, prep, basis, op, bounds, config, r),
        ScfGradientKind::Rohf => {
            restricted_open_scf_gradient(mol, prep, basis, op, bounds, config, r)
        }
    })
    .map_err(make_err)
}

/// `run_rhf(with_gradient=True)`: the plain closed-shell solve (the refusals
/// of [`scf_gradient_config`] already ran) and its analytic gradient.
pub(crate) fn solve_rhf_with_gradient(
    py: Python<'_>,
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    basis: &ferric_core::basis::BasisSet,
    bounds: &SchwarzBounds,
    config: RhfConfig,
) -> PyResult<PyObject> {
    let op = Operator::coulomb();
    let r = py
        .allow_threads(|| solve_rhf(ctx, mol, prep, op, bounds, &config))
        .map_err(make_err)?;
    let g = scf_gradient(
        py,
        ScfGradientKind::Rhf,
        mol,
        prep,
        basis,
        bounds,
        &config,
        &r,
    )?;
    let mut result = PyRhfResult::from_scf(mol, basis, r, config, false);
    result.gradient_data = Some(g);
    Ok(Py::new(py, result)?.into_any())
}

/// A refusal the library classifies as a bad symmetry request
/// (`ferric_scf::symmetric::is_symmetry_refusal`) is a `ValueError` naming
/// `<who>(symmetry=True)`; anything else is a solver failure (`RuntimeError`).
pub(crate) fn symmetry_err(who: &'static str) -> impl Fn(ferric_core::error::FerricError) -> PyErr {
    move |e| {
        if ferric_scf::symmetric::is_symmetry_refusal(&e) {
            pyo3::exceptions::PyValueError::new_err(format!("{who}(symmetry=True): {e}"))
        } else {
            make_err(e)
        }
    }
}

/// Run the point-group-blocked closed-shell solve (`config.symmetry` set) and
/// wrap it as a `SymmetricRhfResult`. `who` names the Python entry point in
/// refusals.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_symmetric_outcome(
    py: Python<'_>,
    who: &'static str,
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
) -> PyResult<PyObject> {
    let sym = py
        .allow_threads(|| {
            ferric_scf::symmetric::solve_rhf_symmetric(ctx, mol, prep, op, bounds, config)
        })
        .map_err(symmetry_err(who))?;
    warn_rhf_not_converged(py, sym.scf_in_frame(), config);
    Ok(Py::new(py, PySymmetricRhfResult::from_solution(sym))?.into_any())
}

/// Solve and wrap: the plain `RhfResult` path (unchanged), or with
/// `config.symmetry` the blocked solve returned as the distinct frame-aware
/// `SymmetricRhfResult`. A `fock_hook` takes the hooked solve
/// (`blocks::solve_rhf_hooked`), which refuses symmetry itself, and is
/// followed by the hook's noise-floor advisory ([`warn_hook_noise_floor`];
/// `user_set_energy_conv` says whether `energy_conv` was passed).
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_rhf_outcome(
    py: Python<'_>,
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: RhfConfig,
    fock_hook: Option<&Bound<'_, PyAny>>,
    user_set_energy_conv: bool,
) -> PyResult<PyObject> {
    if let Some(hook) = fock_hook {
        let (r, term) = blocks::solve_rhf_hooked(py, ctx, mol, prep, op, bounds, &config, hook)?;
        warn_hook_noise_floor(
            py,
            "run_rhf",
            SolverFamily::ClosedShell,
            user_set_energy_conv,
            mol,
            prep,
            &config,
            term,
        );
        return wrap_rhf_result(py, mol, prep.basis_set(), r, config, true);
    }
    if config.irrep_nelec.is_some() && !config.symmetry {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "run_rhf: irrep_nelec requires symmetry=True",
        ));
    }
    if config.symmetry {
        return solve_symmetric_outcome(py, "run_rhf", ctx, mol, prep, op, bounds, &config);
    }
    let r = py
        .allow_threads(|| solve_rhf(ctx, mol, prep, op, bounds, &config))
        .map_err(make_err)?;
    wrap_rhf_result(py, mol, prep.basis_set(), r, config, false)
}

/// Warn if unconverged and wrap a closed-shell solve on the orbital basis
/// `basis` as an `RhfResult`.
pub(crate) fn wrap_rhf_result(
    py: Python<'_>,
    mol: &Molecule,
    basis: &ferric_core::basis::BasisSet,
    r: ScfResult,
    config: RhfConfig,
    fock_hooked: bool,
) -> PyResult<PyObject> {
    warn_rhf_not_converged(py, &r, &config);
    Ok(Py::new(
        py,
        PyRhfResult::from_scf(mol, basis, r, config, fock_hooked),
    )?
    .into_any())
}

impl PyRhfResult {
    /// Wrap a closed-shell solve on the orbital basis `basis`, with its
    /// certificate.
    fn from_scf(
        mol: &Molecule,
        basis: &ferric_core::basis::BasisSet,
        r: ScfResult,
        config: RhfConfig,
        fock_hooked: bool,
    ) -> Self {
        let certificate_data = scf_kind_certificate(mol, &r, &config, basis, fock_hooked);
        PyRhfResult {
            energy: r.energy,
            converged: r.converged,
            iterations: r.iterations,
            computed_quartets: r.computed_quartets,
            density_data: r.density_total.clone(),
            orbital_energies_data: r.eps_alpha.clone(),
            scf_data: r,
            scf_config: config,
            certificate_data,
            gradient_data: None,
        }
    }
}

/// Register the two `run_rhf` result types.
pub(crate) fn add_rhf_result_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRhfResult>()?;
    m.add_class::<PySymmetricRhfResult>()
}

/// Result of `run_rhf(..., symmetry=True)` (point-group-blocked closed-shell
/// RHF) and of `run_dft` / `run_ksdft(..., symmetry=True)` (blocked RKS).
///
/// A distinct type from `RhfResult` on purpose. The SCF ran on the
/// reoriented, SYMMETRIZED geometry, so `energy` belongs to that geometry
/// (it differs from the input geometry's energy by the effect of
/// `max_symmetrization_shift`, at most 1e-4 Bohr) and every AO-basis matrix
/// (density, MO coefficients) would live in a rotated frame. Those accessors
/// raise instead of returning matrices that look like input-frame data, and
/// no function that takes an `RhfResult` (gradients, post-HF, Hirshfeld,
/// properties) accepts this object.
#[pyclass]
#[pyo3(name = "SymmetricRhfResult")]
pub(crate) struct PySymmetricRhfResult {
    /// Total SCF energy (Hartree) of the symmetrized geometry.
    #[pyo3(get)]
    pub(crate) energy: f64,
    /// Whether the SCF met both convergence thresholds.
    #[pyo3(get)]
    pub(crate) converged: bool,
    /// Number of SCF iterations run.
    #[pyo3(get)]
    pub(crate) iterations: usize,
    /// Schoenflies symbol of the Abelian group used for blocking ("C1" if none).
    #[pyo3(get)]
    pub(crate) point_group: String,
    /// Irrep label of each MO column ("-" for padded linearly dependent columns).
    #[pyo3(get)]
    pub(crate) orbital_irreps: Vec<String>,
    /// Largest atom displacement (Bohr) made by symmetrizing the input geometry.
    #[pyo3(get)]
    pub(crate) max_symmetrization_shift: f64,
    pub(crate) orbital_energies_data: Vec<f64>,
    pub(crate) irrep_occupations: Vec<(String, usize)>,
    pub(crate) symbols: Vec<String>,
    pub(crate) coords_angstrom: Vec<(f64, f64, f64)>,
    pub(crate) rotation: [[f64; 3]; 3],
    pub(crate) origin_angstrom: [f64; 3],
}

impl PySymmetricRhfResult {
    fn from_solution(sym: ferric_scf::symmetric::SymmetricRhf) -> Self {
        let mol = sym.molecule();
        let (rotation, origin_bohr) = sym.frame_transform();
        let scf = sym.scf_in_frame();
        PySymmetricRhfResult {
            energy: sym.energy(),
            converged: scf.converged,
            iterations: scf.iterations,
            point_group: sym.group().to_string(),
            orbital_irreps: sym.orbital_irreps().to_vec(),
            max_symmetrization_shift: sym.max_symmetrization_shift(),
            orbital_energies_data: scf.eps_alpha.clone(),
            irrep_occupations: sym.irrep_occupations(),
            symbols: mol.atoms.iter().map(|a| a.symbol.clone()).collect(),
            coords_angstrom: mol
                .atoms
                .iter()
                .map(|a| {
                    (
                        a.x / ANGSTROM_TO_BOHR,
                        a.y / ANGSTROM_TO_BOHR,
                        a.zpos / ANGSTROM_TO_BOHR,
                    )
                })
                .collect(),
            rotation,
            origin_angstrom: origin_bohr.map(|x| x / ANGSTROM_TO_BOHR),
        }
    }
}

pub(crate) const SYMMETRIC_FRAME_MSG: &str =
    "not available for a symmetry=True result: the AO matrices live in the \
reoriented, symmetrized frame (see frame_transform()), not the input frame. Run with \
symmetry=False for density/MO coefficients/gradients.";

#[pymethods]
impl PySymmetricRhfResult {
    /// Orbital energies (Hartree) in MO-column order, aligned with
    /// `orbital_irreps`. Occupied columns come first (energy-ordered), then
    /// the virtuals, so with `irrep_nelec` the list need not be globally
    /// ascending. Padded linearly dependent columns carry 1e6.
    fn orbital_energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.orbital_energies_data.clone())
    }
    /// Electrons per irrep of the converged state, `{label: n}` (virtual-only
    /// irreps are listed with 0).
    fn irrep_occupations(&self) -> std::collections::BTreeMap<String, usize> {
        self.irrep_occupations.iter().cloned().collect()
    }
    /// Element symbols of the symmetrized geometry, in input atom order.
    fn symbols(&self) -> Vec<String> {
        self.symbols.clone()
    }
    /// The symmetrized, reoriented geometry the energy belongs to (Angstrom,
    /// the symmetry frame), in input atom order.
    fn symmetrized_coords(&self) -> Vec<(f64, f64, f64)> {
        self.coords_angstrom.clone()
    }
    /// `(R, origin)` with `r_sym = R @ (r_in - origin)`; `origin` is an
    /// input-frame point in Angstrom (same units as `Molecule.coords()` and
    /// `symmetrized_coords()`). Identity and zero for C1. The relation holds
    /// to the symmetrization shift (`max_symmetrization_shift`).
    fn frame_transform(&self) -> ([[f64; 3]; 3], [f64; 3]) {
        (self.rotation, self.origin_angstrom)
    }
    /// Unavailable: the density is in the symmetry frame.
    fn density(&self) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "density(): {SYMMETRIC_FRAME_MSG}"
        )))
    }
    /// Unavailable: the MO coefficients are in the symmetry frame.
    fn mo_coefficients(&self) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "mo_coefficients(): {SYMMETRIC_FRAME_MSG}"
        )))
    }
    /// Unavailable: no gradient is defined in the symmetry frame.
    fn gradient(&self) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "gradient(): {SYMMETRIC_FRAME_MSG}"
        )))
    }
}

// ── UHF (open-shell) ──

#[pyclass]
#[pyo3(name = "UhfResult")]
pub(crate) struct PyUhfResult {
    #[pyo3(get)]
    pub(crate) energy: f64,
    #[pyo3(get)]
    pub(crate) converged: bool,
    #[pyo3(get)]
    pub(crate) iterations: usize,
    #[pyo3(get)]
    pub(crate) computed_quartets: usize,
    pub(crate) density_alpha_data: Array2<f64>,
    pub(crate) density_beta_data: Array2<f64>,
    pub(crate) eps_alpha_data: Vec<f64>,
    pub(crate) eps_beta_data: Vec<f64>,
    pub(crate) mos_alpha_data: Array2<f64>,
    pub(crate) mos_beta_data: Array2<f64>,
    pub(crate) occ_alpha_data: Vec<f64>,
    pub(crate) occ_beta_data: Vec<f64>,
    pub(crate) cosx_final: Option<ferric_scf::cosx_k::CosxFinalPass>,
    pub(crate) guess_basis_report: Option<ferric_scf::basis2::GuessBasisReport>,
    pub(crate) precision: ferric_core::gpu::provenance::PrecisionRecord,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
    pub(crate) one_step_energy_drop_data: Option<f64>,
    pub(crate) stationarity_tol_data: Option<f64>,
    /// `dE/dR` (natoms x 3, Hartree/Bohr) when solved with `with_gradient=True`.
    pub(crate) gradient_data: Option<Array2<f64>>,
}

/// Occupation number of each MO column of `c` in the spin density `d`:
/// `diag(Cᵀ S D S C)`. With `Cᵀ S C = 1` this recovers `n` from
/// `D = C diag(n) Cᵀ` whatever produced the occupation (aufbau, MOM).
pub(crate) fn mo_occupations(s: &Array2<f64>, c: &Array2<f64>, d: &Array2<f64>) -> Vec<f64> {
    let sc = s.dot(c);
    let dsc = d.dot(&sc);
    (0..c.ncols())
        .map(|k| sc.column(k).dot(&dsc.column(k)))
        .collect()
}

/// The β MO coefficients, orbital energies and density of an open-shell
/// solve. ROHF has one MO set, so its β orbitals and energies are the α ones;
/// the density falls back to the total only when there is no β channel.
pub(crate) fn beta_channel(r: &ScfResult) -> (Array2<f64>, Vec<f64>, Array2<f64>) {
    (
        r.mos_beta.clone().unwrap_or_else(|| r.mos_alpha.clone()),
        r.eps_beta.clone().unwrap_or_else(|| r.eps_alpha.clone()),
        r.density_beta
            .clone()
            .unwrap_or_else(|| r.density_total.clone()),
    )
}

/// The open-shell result both `run_uhf` and `run_rohf` return, with the β
/// channel from [`beta_channel`] and per-spin occupations from
/// [`mo_occupations`].
impl PyUhfResult {
    fn from_scf(
        r: ScfResult,
        mol: &Molecule,
        config: &RhfConfig,
        prep: &PreparedBasis,
        fock_hooked: bool,
    ) -> Self {
        let certificate_data = scf_kind_certificate(mol, &r, config, prep.basis_set(), fock_hooked);
        let s = ferric_integrals::oneelectron::overlap(prep);
        let (mos_beta_data, eps_beta_data, density_beta_data) = beta_channel(&r);
        PyUhfResult {
            occ_alpha_data: mo_occupations(&s, &r.mos_alpha, &r.density_alpha),
            occ_beta_data: mo_occupations(&s, &mos_beta_data, &density_beta_data),
            mos_alpha_data: r.mos_alpha.clone(),
            mos_beta_data,
            one_step_energy_drop_data: r.convergence.and_then(|c| c.one_step_energy_drop),
            stationarity_tol_data: scf_stationarity_tol(&r),
            certificate_data,
            energy: r.energy,
            converged: r.converged,
            iterations: r.iterations,
            computed_quartets: r.computed_quartets,
            density_beta_data,
            density_alpha_data: r.density_alpha,
            cosx_final: r.cosx_final,
            guess_basis_report: r.guess_basis,
            precision: r.precision,
            eps_alpha_data: r.eps_alpha,
            eps_beta_data,
            gradient_data: None,
        }
    }
}

#[pymethods]
impl PyUhfResult {
    /// The COSX final-grid pass record (see `RhfResult.cosx_final_pass`).
    #[getter]
    fn cosx_final_pass(&self, py: Python<'_>) -> PyResult<Option<Py<pyo3::types::PyDict>>> {
        cosx_final_dict(py, self.cosx_final)
    }
    /// The secondary-basis guess (`guess_basis=`): `{"basis", "projection",
    /// "iterations", "energy", "nbasis_guess", "nbasis_target", "converged",
    /// "unconverged_reason", "used_as_guess"}` describing the secondary SCF,
    /// or `None` when no secondary-basis guess ran.
    #[getter]
    fn guess_basis(&self, py: Python<'_>) -> PyResult<Option<Py<pyo3::types::PyDict>>> {
        guess_basis_dict(py, self.guess_basis_report.as_ref())
    }
    /// Precision provenance of this SCF (see `RhfResult.precision`).
    #[getter]
    fn precision(&self, py: Python<'_>) -> PyResult<PyObject> {
        precision_dict(py, &self.precision)
    }
    /// Validation certificate of this result (see `RhfResult.certificate`).
    /// A UHF/UKS result solved without `stability_descent=True` carries the
    /// "internal stability not checked" warning.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    /// First-order energy change of one plain Roothaan step from the converged
    /// density, summed over spins (see `RhfResult.one_step_energy_drop`), or
    /// `None` when the exit is not an aufbau convergence.
    #[getter]
    fn one_step_energy_drop(&self) -> Option<f64> {
        self.one_step_energy_drop_data
    }
    /// The tolerance `one_step_energy_drop` is judged against (see
    /// `RhfResult.stationarity_tol`; the span is the wider of the two spins).
    #[getter]
    fn stationarity_tol(&self) -> Option<f64> {
        self.stationarity_tol_data
    }
    /// Analytic nuclear gradient `dE/dR` as an (natoms, 3) array in
    /// Hartree/Bohr, rows in the Molecule's atom order; `None` unless the
    /// result came from `run_uhf` / `run_rohf(..., with_gradient=True)`.
    fn gradient<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.gradient_data
            .as_ref()
            .map(|g| PyArray2::from_array(py, g))
    }
    fn density_alpha<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.density_alpha_data)
    }
    fn density_beta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.density_beta_data)
    }
    fn orbital_energies_alpha<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.eps_alpha_data.clone())
    }
    fn orbital_energies_beta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.eps_beta_data.clone())
    }
    /// α MO coefficients (n_bf x n_mo), column k = α MO k, ascending energy.
    fn mo_coefficients_alpha<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.mos_alpha_data)
    }
    /// β MO coefficients (n_bf x n_mo); the α matrix for ROHF (one MO set).
    fn mo_coefficients_beta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.mos_beta_data)
    }
    /// α occupation of each α MO column, `diag(Cᵀ S D_α S C)`.
    fn occupations_alpha<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.occ_alpha_data.clone())
    }
    /// β occupation of each β MO column, `diag(Cᵀ S D_β S C)`.
    fn occupations_beta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.occ_beta_data.clone())
    }
    fn __repr__(&self) -> String {
        format!(
            "UhfResult(energy={:.10}, converged={})",
            self.energy, self.converged
        )
    }
    fn __str__(&self) -> String {
        format!(
            "UHF Energy: {:.10} Ha (converged: {}, {} iterations)",
            self.energy, self.converged, self.iterations
        )
    }
}

/// Unrestricted Hartree-Fock (open-shell). α/β electron counts come from the
/// molecule's `charge` and `multiplicity` (set via `Molecule.from_xyz_string`).
/// Exposes the same SCF knob set as `run_rhf`; the convergence aids `level_shift`
/// and `mom_after_iter` are especially useful for open-shell doublets / radicals
/// where DIIS plateaus or the occupied set flip-flops.
///
///   mom_reference     "previous" (default): MOM, when `mom_after_iter` > 0
///                     arms it, compares against the previous iteration's
///                     occupied set. "initial": compare against the STARTING
///                     occupied orbitals (those of `dm0`, or of the guess),
///                     fixed, from the first iteration: PySCF's
///                     `scf.addons.mom_occ`. Holds a non-aufbau start such as
///                     a core hole. Refused with `mom_after_iter` > 0 or
///                     `stability_descent`.
///   guess             "minao" (default; "sad" is an alias) or "hcore". The
///                     bare hcore guess is an explicit opt-in: on some open
///                     shells (e.g. O2/STO-3G) it converges to a different
///                     stationary point. An unrecognised value is an error,
///                     never a silent fallback.
///   stability_descent False (default). When True, check the converged UHF
///                     solution's internal stability and, if it is a SADDLE of
///                     the orbital Hessian, follow the downhill eigenvector and
///                     re-converge, keeping the lowest state
///                     (`RhfConfig::check_stability` + `scf_stability_descent`).
///                     Costs one Davidson eigensolve per solve plus one SCF per
///                     descent taken. Needed where the default guess lands on a
///                     saddle: O2 triplet/STO-3G (1.33 mHa above the UHF
///                     minimum, exactly as PySCF's own default guess does) and
///                     N2+/6-31G. Before this kwarg the repair was unreachable
///                     from Python.
///   trah_trigger    None (default) = off. A positive DIIS-error threshold turns
///                   on the trust-region augmented-Hessian (TRAH) solver once the
///                   DIIS error drops below it. A convergence RESCUE: it wins
///                   iterations but costs wall time. Same as the CLI
///                   `[scf] trah_trigger`.
///
/// Note: the exit report (final_* residuals, unmet_criteria, non-convergence RuntimeWarning) is RHF-only today; non-convergence raises.
/// A user-set energy_conv below the estimated (rounding-level) noise floor of dE emits one RuntimeWarning before the SCF starts.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set,
    max_iter=None, energy_conv=None, density_conv=None, diis_size=None,
    integral_thresh=None, k_builder=None, df_j_aux=None, df_k_aux=None, ri_k_precision=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None, memory_budget_gb=None,
    guess=None, stability_descent=None,
    cosx_grid=None, cosx_final_pass=None, cosx_final_grid=None, cosx_overlap_fit=None,
    trah_trigger=None, fock_hook=None, guess_basis=None, guess_projection=None, ri_j_precision=None,
    dm0=None, with_gradient=false, mom_reference=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_uhf(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    diis_size: Option<usize>,
    integral_thresh: Option<f64>,
    k_builder: Option<&str>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    ri_k_precision: Option<&str>,
    level_shift: Option<f64>,
    mom_after_iter: Option<usize>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    memory_budget_gb: Option<f64>,
    guess: Option<&str>,
    stability_descent: Option<bool>,
    cosx_grid: Option<&Bound<'_, PyAny>>,
    cosx_final_pass: Option<bool>,
    cosx_final_grid: Option<&Bound<'_, PyAny>>,
    cosx_overlap_fit: Option<bool>,
    trah_trigger: Option<f64>,
    fock_hook: Option<&Bound<'_, PyAny>>,
    guess_basis: Option<&str>,
    guess_projection: Option<&str>,
    ri_j_precision: Option<&str>,
    dm0: Option<&Bound<'_, PyAny>>,
    with_gradient: bool,
    mom_reference: Option<&str>,
) -> PyResult<PyUhfResult> {
    refuse_dm0_with_guess("run_uhf", dm0.is_some(), guess, guess_basis)?;
    let mom_reference = parse_mom_reference(mom_reference, mom_after_iter, stability_descent)?;
    let trah_trigger = check_trah_trigger(trah_trigger)?;
    let guess_basis = parse_guess_basis(guess_basis, guess_projection)?;
    // The shared strict parser (same as run_rhf and the CLI): a typo must not
    // silently select a different SCF state (on O2/STO-3G the two guesses
    // land on states 0.255 Ha apart).
    let use_sad_guess = parse_guess(guess)?;
    // The descent needs a stability verdict to act on, so the kwarg turns on
    // both halves; `scf_stability_descent` alone only prints a SKIPPED notice.
    let descent = stability_descent.unwrap_or(false);
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let prep = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let dm0 = match dm0 {
        Some(d) => {
            let n_spin = spin_counts(&emol, "run_uhf")?;
            blocks::checked_dm0_spin(Some(d), &prep, n_spin, "run_uhf")?
        }
        None => None,
    };
    // Same defaults as run_rhf / the CLI [scf] section.
    let mut config = UhfConfig {
        max_iter: max_iter.unwrap_or_else(default_scf_max_iter),
        energy_conv: energy_conv.unwrap_or(1e-3),
        density_conv: density_conv.unwrap_or(1e-6),
        diis_size: diis_size.unwrap_or(8),
        integral_thresh: integral_thresh.unwrap_or(1e-12),
        k_builder: k_builder.map(|s| s.to_string()),
        cosx: resolve_cosx_kwargs(
            k_builder,
            cosx_grid,
            cosx_final_pass,
            cosx_final_grid,
            cosx_overlap_fit,
        )?,
        // Same opt-out spellings as run_dft ("exact"/"none"/"off"/...).
        df_j_aux: df_j_aux.map(ferric_scf::rhf::normalize_df_aux),
        df_k_aux: df_k_aux.map(ferric_scf::rhf::normalize_df_aux),
        ri_j_precision: parse_ri_j_precision(ri_j_precision)?,
        ri_k_precision: parse_ri_k_precision(ri_k_precision)?,
        level_shift: level_shift.unwrap_or(0.0),
        mom_after_iter: mom_after_iter.unwrap_or(0),
        mom_reference,
        external_potential: build_external_potential(point_charges, external_field),
        // 0 means "unset -> auto" (resolve_three_index_budget), so an
        // omitted kwarg preserves the previous auto-detect behaviour.
        three_index_budget_bytes: budget_bytes_from_gb(memory_budget_gb).unwrap_or(0),
        use_sad_guess,
        guess_basis,
        trah_trigger,
        check_stability: descent,
        scf_stability_descent: descent,
        ..Default::default()
    };
    if with_gradient {
        config = scf_gradient_config("run_uhf", ScfGradientKind::Uhf, &config, fock_hook)?;
    }
    let ctx = ParallelContext::default();
    // Release the GIL for the duration of the SCF. Every argument below is an
    // owned Rust value (`emol` is a clone; `prep`/`bounds`/`config` are built
    // above), so nothing Python-borrowed crosses the closure boundary.
    warn_energy_conv_noise_floor(
        py,
        "run_uhf",
        SolverFamily::OpenShell,
        energy_conv.is_some(),
        &emol,
        &prep,
        &config,
    );
    let fock_hooked = fock_hook.is_some();
    let (r, term) = blocks::solve_open_shell(
        py,
        blocks::OpenShell::Uhf,
        &ctx,
        &emol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &config,
        fock_hook,
        dm0.as_ref(),
    )?;
    warn_hook_noise_floor(
        py,
        "run_uhf",
        SolverFamily::OpenShell,
        energy_conv.is_some(),
        &emol,
        &prep,
        &config,
        term,
    );
    let gradient = with_gradient
        .then(|| {
            let k = ScfGradientKind::Uhf;
            scf_gradient(py, k, &emol, &prep, &basis_set.inner, &bounds, &config, &r)
        })
        .transpose()?;
    let mut result = PyUhfResult::from_scf(r, &emol, &config, &prep, fock_hooked);
    result.gradient_data = gradient;
    Ok(result)
}

// ── ROHF (open-shell, spin-pure) ──

/// Restricted Open-Shell Hartree-Fock. α/β occupations come from the
/// molecule's `charge` and `multiplicity` (set via `Molecule.from_xyz_string`):
/// `nocc_open = multiplicity - 1` singly-occupied (α) orbitals, the remainder
/// doubly occupied. Coupling is Guest-Saunders (PySCF default; not a knob).
/// Returns the same `UhfResult` shape as `run_uhf` — ROHF's `alpha`/`beta`
/// density and orbital-energy accessors carry the doubly + singly occupied
/// α set and the doubly-occupied-only β set respectively (single set of
/// spin-pure MOs, so `mos_alpha`/`mos_beta` coincide internally but the
/// occupied-orbital semantics differ from UHF). Exposes the same SCF knob
/// set as `run_rhf`/`run_uhf`.
///
/// Note: the exit report (final_* residuals, unmet_criteria, non-convergence RuntimeWarning) is RHF-only today.
/// A user-set energy_conv below the estimated (rounding-level) noise floor of dE emits one RuntimeWarning before the SCF starts.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set,
    max_iter=None, energy_conv=None, density_conv=None, diis_size=None,
    integral_thresh=None, k_builder=None, df_j_aux=None, df_k_aux=None, ri_k_precision=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None, memory_budget_gb=None, fock_hook=None,
    guess_basis=None, guess_projection=None, ri_j_precision=None, dm0=None,
    with_gradient=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_rohf(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    diis_size: Option<usize>,
    integral_thresh: Option<f64>,
    k_builder: Option<&str>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    ri_k_precision: Option<&str>,
    level_shift: Option<f64>,
    mom_after_iter: Option<usize>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    memory_budget_gb: Option<f64>,
    fock_hook: Option<&Bound<'_, PyAny>>,
    guess_basis: Option<&str>,
    guess_projection: Option<&str>,
    ri_j_precision: Option<&str>,
    dm0: Option<&Bound<'_, PyAny>>,
    with_gradient: bool,
) -> PyResult<PyUhfResult> {
    refuse_dm0_with_guess("run_rohf", dm0.is_some(), None, guess_basis)?;
    let guess_basis = parse_guess_basis(guess_basis, guess_projection)?;
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let prep = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let dm0 = match dm0 {
        Some(d) => {
            let n_spin = spin_counts(&emol, "run_rohf")?;
            blocks::checked_dm0_spin(Some(d), &prep, n_spin, "run_rohf")?
        }
        None => None,
    };
    // Same defaults as run_rhf / run_uhf / the CLI [scf] section.
    let mut config: RohfConfig = RhfConfig {
        max_iter: max_iter.unwrap_or_else(default_scf_max_iter),
        energy_conv: energy_conv.unwrap_or(1e-3),
        density_conv: density_conv.unwrap_or(1e-6),
        diis_size: diis_size.unwrap_or(8),
        integral_thresh: integral_thresh.unwrap_or(1e-12),
        k_builder: k_builder.map(|s| s.to_string()),
        // Same opt-out spellings as run_dft ("exact"/"none"/"off"/...).
        df_j_aux: df_j_aux.map(ferric_scf::rhf::normalize_df_aux),
        df_k_aux: df_k_aux.map(ferric_scf::rhf::normalize_df_aux),
        ri_j_precision: parse_ri_j_precision(ri_j_precision)?,
        ri_k_precision: parse_ri_k_precision(ri_k_precision)?,
        level_shift: level_shift.unwrap_or(0.0),
        mom_after_iter: mom_after_iter.unwrap_or(0),
        external_potential: build_external_potential(point_charges, external_field),
        // 0 means "unset -> auto" (resolve_three_index_budget), so an
        // omitted kwarg preserves the previous auto-detect behaviour.
        three_index_budget_bytes: budget_bytes_from_gb(memory_budget_gb).unwrap_or(0),
        guess_basis,
        ..Default::default()
    };
    if with_gradient {
        config = scf_gradient_config("run_rohf", ScfGradientKind::Rohf, &config, fock_hook)?;
    }
    let ctx = ParallelContext::default();
    // Release the GIL for the duration of the SCF. Every argument below is an
    // owned Rust value (`emol` is a clone; `prep`/`bounds`/`config` are built
    // above), so nothing Python-borrowed crosses the closure boundary.
    warn_energy_conv_noise_floor(
        py,
        "run_rohf",
        SolverFamily::OpenShell,
        energy_conv.is_some(),
        &emol,
        &prep,
        &config,
    );
    let fock_hooked = fock_hook.is_some();
    let (r, term) = blocks::solve_open_shell(
        py,
        blocks::OpenShell::Rohf,
        &ctx,
        &emol,
        &prep,
        op,
        &bounds,
        &config,
        fock_hook,
        dm0.as_ref(),
    )?;
    warn_hook_noise_floor(
        py,
        "run_rohf",
        SolverFamily::OpenShell,
        energy_conv.is_some(),
        &emol,
        &prep,
        &config,
        term,
    );
    let gradient = with_gradient
        .then(|| {
            let k = ScfGradientKind::Rohf;
            scf_gradient(py, k, &emol, &prep, &basis_set.inner, &bounds, &config, &r)
        })
        .transpose()?;
    let mut result = PyUhfResult::from_scf(r, &emol, &config, &prep, fock_hooked);
    result.gradient_data = gradient;
    Ok(result)
}
