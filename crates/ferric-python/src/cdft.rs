//! Constrained DFT bindings: moved verbatim from `lib.rs`.

use super::*;

// ── Constrained DFT (cDFT) ──

/// Strict parser for a cDFT constraint `kind`. Unknown values are an error,
/// never a silent default: "charge" and "spin" select different Lagrangians,
/// and a typo that quietly picked one would return a converged answer to a
/// different question.
pub(crate) fn parse_cdft_kind(kind: &str) -> PyResult<ferric_dft::cdft::SpinChannel> {
    use ferric_dft::cdft::SpinChannel;
    match kind.to_ascii_lowercase().as_str() {
        "charge" => Ok(SpinChannel::Total),
        "spin" => Ok(SpinChannel::SpinDiff),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown cDFT constraint kind '{kind}': valid options are 'charge' \
             (N_alpha + N_beta on the fragment) and 'spin' (N_alpha - N_beta)"
        ))),
    }
}

pub(crate) fn cdft_kind_name(spin: ferric_dft::cdft::SpinChannel) -> &'static str {
    match spin {
        ferric_dft::cdft::SpinChannel::Total => "charge",
        ferric_dft::cdft::SpinChannel::SpinDiff => "spin",
    }
}

/// One constrained-DFT fragment constraint.
///
/// `atoms` are 0-based atom indices forming the fragment (non-empty, no
/// duplicates). `target` is a Becke fragment POPULATION in electrons, NOT a
/// net charge:
///
///   kind="charge"  target = N_alpha + N_beta on the fragment
///                  (e.g. 2.0 on a He atom = neutral He; 1.0 = He+).
///   kind="spin"    target = N_alpha - N_beta on the fragment.
///
/// The population is `Tr[W^C D]` with `W^C` the Becke fuzzy-cell weight
/// operator of the fragment (`ferric_dft::cdft`). With an ECP basis only the
/// explicitly-treated (valence) electrons are counted. Atom indices are
/// range-checked against the molecule when the constraint is used.
#[pyclass]
#[pyo3(name = "CdftConstraint")]
#[derive(Clone)]
pub(crate) struct PyCdftConstraint {
    pub(crate) atoms: Vec<usize>,
    pub(crate) target: f64,
    pub(crate) spin: ferric_dft::cdft::SpinChannel,
}

#[pymethods]
impl PyCdftConstraint {
    #[new]
    #[pyo3(signature = (atoms, target, kind="charge"))]
    fn new(atoms: Vec<i64>, target: f64, kind: &str) -> PyResult<Self> {
        let spin = parse_cdft_kind(kind)?;
        if atoms.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "CdftConstraint: atoms must be a non-empty list of atom indices",
            ));
        }
        let mut idx = Vec::with_capacity(atoms.len());
        for &a in &atoms {
            if a < 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "CdftConstraint: atom index {a} is negative (indices are 0-based)"
                )));
            }
            idx.push(a as usize);
        }
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != idx.len() {
            // The fragment weight is a SUM of per-atom Becke weights, so a
            // repeated index would count that atom twice.
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "CdftConstraint: duplicate atom index in {atoms:?}; a repeated atom \
                 would be weighted twice in the fragment population"
            )));
        }
        if !target.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "CdftConstraint: target must be finite, got {target}"
            )));
        }
        Ok(PyCdftConstraint {
            atoms: idx,
            target,
            spin,
        })
    }

    /// 0-based atom indices of the fragment.
    #[getter]
    fn atoms(&self) -> Vec<usize> {
        self.atoms.clone()
    }

    /// Target fragment population in electrons (see the class docstring).
    #[getter]
    fn target(&self) -> f64 {
        self.target
    }

    /// "charge" or "spin".
    #[getter]
    fn kind(&self) -> &'static str {
        cdft_kind_name(self.spin)
    }

    fn __repr__(&self) -> String {
        format!(
            "CdftConstraint(atoms={:?}, target={}, kind='{}')",
            self.atoms,
            self.target,
            cdft_kind_name(self.spin)
        )
    }
}

/// Result of a constrained UHF/UKS solve (`run_cdft`).
///
/// A returned result always has a CONVERGED outer (lambda) loop: when the
/// lambda-Newton loop exhausts `max_outer` the Rust driver errors and
/// `run_cdft` raises `RuntimeError` instead of returning. The inner SCF at the
/// final lambda can still be unconverged, which `scf_converged` reports, and
/// `converged` is True only when BOTH hold.
#[pyclass]
#[pyo3(name = "CdftResult")]
pub(crate) struct PyCdftResult {
    /// Total energy (Ha) at the constrained density, WITHOUT the constraint
    /// term: the ordinary UHF/UKS energy functional evaluated at rho_lambda.
    #[pyo3(get)]
    pub(crate) energy: f64,
    /// `scf_converged` AND every |population - target| < `lambda_tol`.
    #[pyo3(get)]
    pub(crate) converged: bool,
    /// Whether the inner SCF at the final lambda met its thresholds.
    #[pyo3(get)]
    pub(crate) scf_converged: bool,
    /// Inner SCF iterations of the final solve.
    #[pyo3(get)]
    pub(crate) iterations: usize,
    /// Outer lambda-Newton iterations taken.
    #[pyo3(get)]
    pub(crate) outer_iterations: usize,
    /// Lagrange multipliers (Ha per electron), one per constraint, in input order.
    #[pyo3(get)]
    pub(crate) lambdas: Vec<f64>,
    /// Achieved fragment populations Tr[W^C D] (electrons), one per constraint.
    #[pyo3(get)]
    pub(crate) populations: Vec<f64>,
    /// The requested targets, one per constraint.
    #[pyo3(get)]
    pub(crate) targets: Vec<f64>,
    /// Constraint residuals population_C - target_C (electrons), one per
    /// constraint, in input order.
    #[pyo3(get)]
    pub(crate) residuals: Vec<f64>,
    /// max_C |population_C - target_C| (electrons); NaN if any residual is NaN.
    #[pyo3(get)]
    pub(crate) max_constraint_error: f64,
    /// The outer-loop tolerance the solve used.
    #[pyo3(get)]
    pub(crate) lambda_tol: f64,
    pub(crate) density_alpha_data: Array2<f64>,
    pub(crate) density_beta_data: Array2<f64>,
    pub(crate) eps_alpha_data: Vec<f64>,
    pub(crate) eps_beta_data: Vec<f64>,
    // Held for `cdft_coupling`; not exposed as attributes.
    pub(crate) mos_alpha: Array2<f64>,
    pub(crate) mos_beta: Array2<f64>,
    pub(crate) nocc_alpha: usize,
    pub(crate) nocc_beta: usize,
    pub(crate) spins: Vec<ferric_dft::cdft::SpinChannel>,
    pub(crate) weight_matrices: Vec<Array2<f64>>,
    pub(crate) overlap: Array2<f64>,
    // Every run_cdft input that changes the electronic Hamiltonian (functional,
    // J/K fitting, exchange builder, XC grid, external potential), rendered
    // exactly (Debug prints f64 in shortest round-trip form). cdft_coupling
    // refuses two states whose keys differ: the Wu-Van Voorhis formula assumes
    // one shared H, and the overlap/occupation checks cannot see these inputs.
    pub(crate) hamiltonian_key: String,
}

#[pymethods]
impl PyCdftResult {
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
    /// Alpha MO coefficients (n_bf x n_mo, columns = MOs in ascending
    /// lambda-augmented orbital energy; the first N_alpha are occupied).
    fn mo_coeff_alpha<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.mos_alpha)
    }
    /// Beta MO coefficients (n_bf x n_mo; the first N_beta are occupied).
    fn mo_coeff_beta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.mos_beta)
    }
    /// Occupied (alpha, beta) orbital counts.
    #[getter]
    fn nocc(&self) -> (usize, usize) {
        (self.nocc_alpha, self.nocc_beta)
    }
    /// The AO-basis Becke weight operator W^C of constraint `index`, exactly as
    /// the solve used it. The fragment population of any UHF density pair is
    /// `trace(W @ (Da + Db))` (charge) or `trace(W @ (Da - Db))` (spin).
    fn weight_matrix<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let w = self.weight_matrices.get(index).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "constraint index {index} out of range ({} constraints)",
                self.weight_matrices.len()
            ))
        })?;
        Ok(PyArray2::from_array(py, w))
    }
    /// The constraint kinds ("charge"/"spin"), one per constraint.
    #[getter]
    fn kinds(&self) -> Vec<&'static str> {
        self.spins.iter().map(|s| cdft_kind_name(*s)).collect()
    }
    fn __repr__(&self) -> String {
        format!(
            "CdftResult(energy={:.10}, converged={}, lambdas={:?}, populations={:?})",
            self.energy, self.converged, self.lambdas, self.populations
        )
    }
    fn __str__(&self) -> String {
        format!(
            "cDFT Energy: {:.10} Ha (converged: {}, {} outer iterations, lambdas {:?}, \
             populations {:?})",
            self.energy, self.converged, self.outer_iterations, self.lambdas, self.populations
        )
    }
}

/// The key `cdft_coupling` compares to decide whether two cDFT states share
/// one Hamiltonian.
///
/// Keyed on the EFFECTIVE fitting choice: solve_uhf has no auto-default and
/// build_df_jk treats Some("") as "do not fit", so unset and "" are the same
/// exact-J/K Hamiltonian and must not look different here. The nuclear
/// Hamiltonian too: a ghost centre (`@He`) keeps the basis and AO overlap
/// identical while removing a nucleus, so the overlap check alone cannot see
/// it. Key the PHYSICAL fields only (Z, ghost, ECP core count, exact
/// coordinates via to_bits, charge, multiplicity) from the post-ECP molecule --
/// not its Debug form, which also carries the symbol as typed ("He" vs "he")
/// and would refuse two identical Hamiltonians.
pub(crate) fn cdft_hamiltonian_key(emol: &Molecule, config: &RhfConfig) -> String {
    let effective_aux = |a: &Option<String>| a.clone().filter(|s| !s.is_empty());
    let nuclei: Vec<(i32, bool, i32, [u64; 3])> = emol
        .atoms
        .iter()
        .map(|a| {
            (
                a.z,
                a.ghost,
                a.n_core_ecp,
                [a.x.to_bits(), a.y.to_bits(), a.zpos.to_bits()],
            )
        })
        .collect();
    format!(
        "nuclei={:?} charge={} mult={} xc={:?} df_j_aux={:?} df_k_aux={:?} k_builder={:?} dft_grid={:?} external={:?}",
        nuclei,
        emol.charge,
        emol.multiplicity,
        config.xc,
        effective_aux(&config.df_j_aux),
        effective_aux(&config.df_k_aux),
        config.k_builder,
        config.dft_grid,
        config.external_potential,
    )
}

/// Constrained UHF / UKS (Wu–Van Voorhis cDFT): minimize the energy subject to
/// fragment population constraints, via a nested lambda-Newton loop around an
/// ordinary UHF/UKS solve with `sum_C lambda_C W^C` added to the Fock matrix.
///
///   constraints       non-empty list of `CdftConstraint`. Each `target` is a
///                     Becke fragment POPULATION in electrons (N_a + N_b for
///                     kind="charge", N_a - N_b for kind="spin"), not a net
///                     charge. Validated here: atom indices in range, a charge
///                     target in [0, N_elec], |spin target| <= N_elec, and no
///                     two constraints on the same fragment and kind (their
///                     Jacobian would be singular). Only single-constraint
///                     solves (k = 1) are exercised by the Rust suite and
///                     carry the bracket safeguard; k > 1 runs the plain
///                     k x k Newton.
///   functional        None or "HF" (default) = UHF. Any other name = UKS with
///                     that libxc functional (validated up front). The Rust
///                     cDFT tests validate the UHF path only; UKS-cDFT is
///                     smoke-level, and the stability descent is SKIPPED on a
///                     KS reference (it would need the f_xc kernel).
///   lambda_tol        stop when max |N_C - target_C| < lambda_tol (electrons).
///                     Default 1e-5. A flat stretch of N(lambda) (e.g. He2+ at
///                     the localized-hole plateau, dN/dlambda ~ 7e-5) costs
///                     outer iterations, not accuracy: the six validated He2+
///                     diabats converge there at 1e-10 from lambda = 0. Loosen
///                     it only to buy iterations, and read what is returned --
///                     a looser tolerance accepts a lambda further from the
///                     root, and on a plateau that is a large distance in
///                     lambda for a small one in N.
///   max_outer         outer lambda-Newton iteration cap. Default 30. Exceeding
///                     it raises RuntimeError; nothing unconverged is returned.
///                     The default covers a lambda = 0 start on the validated
///                     He2+ points (deepest 21 iterations); because the Newton
///                     step is clamped to 1 per iteration, a start far from the
///                     root needs at least |lambda_root| iterations to arrive.
///   stability_descent True (default, as in the Rust driver): after the outer
///                     loop converges, check the lambda-augmented orbital
///                     Hessian and, if the constrained solution is a saddle,
///                     descend and re-converge, keeping the lower state that
///                     still meets the constraint. NOTE the default differs
///                     from run_uhf's stability_descent (False), which is the
///                     UNCONSTRAINED descent; see RhfConfig for why.
///   grid_radial /     Becke/Lebedev grid for the weight operator. Both unset
///   grid_angular      (default) = 99 x 302 for W (the XC grid, if any, keeps
///                     its 75 x 110 default). Setting either uses the given
///                     grid for BOTH W and XC, the unset one defaulting to
///                     99 / 302. The weight quadrature must resolve
///                     populations below lambda_tol (75 x 110 floors at ~1e-4).
///
/// The SCF knobs (`scf_max_iter`, `scf_energy_conv`, `scf_density_conv`,
/// `diis_size` .. `guess`) mean what they do in `run_uhf`; unset ones take the
/// Rust `RhfConfig` defaults the cDFT suite is validated with (so
/// `scf_max_iter` defaults to 200 here, as in `run_uhf`). `max_iter`,
/// `energy_conv`, `density_conv` are deprecated aliases (they warn) of the
/// `scf_` spellings; `max_outer` is the multiplier loop. `df_j_aux` /
/// `df_k_aux` unset = exact J/K, as in `run_uhf`.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, constraints,
    functional=None, lambda_tol=None, max_outer=None, stability_descent=None,
    max_iter=None, energy_conv=None, density_conv=None, diis_size=None,
    integral_thresh=None, k_builder=None, df_j_aux=None, df_k_aux=None, scf_ri_k_precision=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None, memory_budget_gb=None,
    guess=None, grid_radial=None, grid_angular=None,
    scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, scf_ri_j_precision=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_cdft(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    constraints: Vec<PyCdftConstraint>,
    functional: Option<&str>,
    lambda_tol: Option<f64>,
    max_outer: Option<usize>,
    stability_descent: Option<bool>,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    diis_size: Option<usize>,
    integral_thresh: Option<f64>,
    k_builder: Option<&str>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    scf_ri_k_precision: Option<&str>,
    level_shift: Option<f64>,
    mom_after_iter: Option<usize>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    memory_budget_gb: Option<f64>,
    guess: Option<&str>,
    grid_radial: Option<usize>,
    grid_angular: Option<usize>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    scf_ri_j_precision: Option<&str>,
) -> PyResult<PyCdftResult> {
    use ferric_dft::cdft::{Constraint, SpinChannel};
    let value_err = |msg: String| pyo3::exceptions::PyValueError::new_err(msg);
    // `max_outer` is this driver's own iteration; the bare SCF spellings are
    // deprecated aliases of the `scf_`-prefixed ones.
    let max_iter = alias_kwarg(
        py,
        "run_cdft",
        "max_iter",
        "scf_max_iter",
        max_iter,
        scf_max_iter,
    )?;
    let energy_conv = alias_kwarg(
        py,
        "run_cdft",
        "energy_conv",
        "scf_energy_conv",
        energy_conv,
        scf_energy_conv,
    )?;
    let density_conv = alias_kwarg(
        py,
        "run_cdft",
        "density_conv",
        "scf_density_conv",
        density_conv,
        scf_density_conv,
    )?;

    let use_sad_guess = parse_guess(guess)?;
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let natoms = emol.atoms.len();
    let nelec = emol.nelec();
    if nelec <= 0 {
        return Err(value_err(format!(
            "run_cdft: the molecule has {nelec} electrons; nothing to constrain"
        )));
    }
    let nelec_f = nelec as f64;

    // ── constraints ──
    if constraints.is_empty() {
        return Err(value_err(
            "run_cdft: constraints must be a non-empty list of CdftConstraint".into(),
        ));
    }
    let mut seen: Vec<(Vec<usize>, SpinChannel)> = Vec::with_capacity(constraints.len());
    let mut cons: Vec<Constraint> = Vec::with_capacity(constraints.len());
    for (ci, c) in constraints.iter().enumerate() {
        if let Some(&bad) = c.atoms.iter().find(|&&a| a >= natoms) {
            return Err(value_err(format!(
                "run_cdft: constraint {ci} names atom {bad}, but the molecule has \
                 {natoms} atoms (indices are 0-based)"
            )));
        }
        match c.spin {
            SpinChannel::Total => {
                if c.target < 0.0 || c.target > nelec_f {
                    return Err(value_err(format!(
                        "run_cdft: constraint {ci} (charge) target {} is outside \
                         [0, {nelec}]: the target is a fragment electron POPULATION, \
                         not a net charge",
                        c.target
                    )));
                }
            }
            SpinChannel::SpinDiff => {
                if c.target.abs() > nelec_f {
                    return Err(value_err(format!(
                        "run_cdft: constraint {ci} (spin) target {} exceeds the \
                         {nelec} electrons in the molecule in magnitude",
                        c.target
                    )));
                }
            }
        }
        let mut key_atoms = c.atoms.clone();
        key_atoms.sort_unstable();
        if seen.iter().any(|(a, s)| *a == key_atoms && *s == c.spin) {
            return Err(value_err(format!(
                "run_cdft: constraint {ci} repeats the fragment {:?} with kind '{}'; \
                 two constraints on the same population make the lambda Jacobian \
                 singular",
                c.atoms,
                cdft_kind_name(c.spin)
            )));
        }
        seen.push((key_atoms, c.spin));
        cons.push(Constraint {
            fragment: c.atoms.clone(),
            spin: c.spin,
            target: c.target,
        });
    }

    // ── outer-loop knobs ──
    let defaults = RhfConfig::default();
    let lambda_tol = lambda_tol.unwrap_or(defaults.cdft_lambda_tol);
    if !(lambda_tol.is_finite() && lambda_tol > 0.0) {
        return Err(value_err(format!(
            "run_cdft: lambda_tol must be finite and > 0, got {lambda_tol}"
        )));
    }
    let max_outer = max_outer.unwrap_or(defaults.cdft_max_outer);
    if max_outer == 0 {
        return Err(value_err("run_cdft: max_outer must be >= 1".into()));
    }

    // ── functional ──
    let xc = match functional {
        None => None,
        Some(name) if name.eq_ignore_ascii_case("hf") => None,
        Some(name) => {
            // Validate the name here, as a ValueError, instead of after the GIL
            // is released and the weight grid is already built.
            ferric_dft::libxc::xc_def_from_name_nspin(name, 2)
                .map_err(|e| value_err(format!("run_cdft: functional: {e}")))?;
            Some(name.to_string())
        }
    };

    // ── grid ──
    let dft_grid = if grid_radial.is_none() && grid_angular.is_none() {
        None
    } else {
        use ferric_dft::prune::SUPPORTED_LEBEDEV_ORDERS;
        let n_radial = grid_radial.unwrap_or(99);
        let n_angular = grid_angular.unwrap_or(302);
        if n_radial == 0 {
            return Err(value_err("run_cdft: grid_radial must be > 0".into()));
        }
        if !SUPPORTED_LEBEDEV_ORDERS.contains(&n_angular) {
            return Err(value_err(format!(
                "run_cdft: grid_angular = {n_angular} is not a supported Lebedev order \
                 (supported: {SUPPORTED_LEBEDEV_ORDERS:?})"
            )));
        }
        Some(ferric_dft::grid::AtomicGridConfig {
            n_radial,
            n_angular,
            ..Default::default()
        })
    };

    let config = RhfConfig {
        constraints: cons,
        cdft_lambda_tol: lambda_tol,
        cdft_max_outer: max_outer,
        cdft_stability_descent: stability_descent.unwrap_or(defaults.cdft_stability_descent),
        xc,
        dft_grid,
        max_iter: max_iter.unwrap_or(defaults.max_iter),
        energy_conv: energy_conv.unwrap_or(defaults.energy_conv),
        density_conv: density_conv.unwrap_or(defaults.density_conv),
        diis_size: diis_size.unwrap_or(defaults.diis_size),
        integral_thresh: integral_thresh.unwrap_or(defaults.integral_thresh),
        k_builder: k_builder.map(|s| s.to_string()),
        // Same shared parser as run_rhf/run_uhf/run_dft, so "exact"/"off" mean
        // exact J/K here too and cdft_coupling's Hamiltonian key sees one
        // spelling for one Hamiltonian.
        df_j_aux: df_j_aux.map(ferric_scf::rhf::normalize_df_aux),
        df_k_aux: df_k_aux.map(ferric_scf::rhf::normalize_df_aux),
        ri_j_precision: parse_ri_j_precision(scf_ri_j_precision)?,
        ri_k_precision: parse_ri_k_precision(scf_ri_k_precision)?,
        level_shift: level_shift.unwrap_or(defaults.level_shift),
        mom_after_iter: mom_after_iter.unwrap_or(defaults.mom_after_iter),
        external_potential: build_external_potential(point_charges, external_field),
        // 0 means "unset -> auto" (resolve_three_index_budget).
        three_index_budget_bytes: budget_bytes_from_gb(memory_budget_gb).unwrap_or(0),
        use_sad_guess,
        ..defaults
    };

    let hamiltonian_key = cdft_hamiltonian_key(&emol, &config);
    let bs = basis_set.inner.clone();
    let prep = PreparedBasis::new(&emol, &bs).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let overlap = ferric_integrals::oneelectron::overlap(&prep);
    let ctx = ParallelContext::default();
    // Release the GIL for the whole constrained solve. Everything the closure
    // touches is an owned Rust value built above.
    let r = py
        .allow_threads(|| {
            ferric_scf::cdft_driver::solve_cdft_uhf(&ctx, &emol, &prep, &bs, &bounds, &config)
        })
        .map_err(make_err)?;

    let targets: Vec<f64> = config.constraints.iter().map(|c| c.target).collect();
    let spins: Vec<SpinChannel> = config.constraints.iter().map(|c| c.spin).collect();
    let max_constraint_error = r.max_residual();
    // Same occupation rule as the driver (cdft_driver::occupied_alpha/_beta).
    let nel = nelec as usize;
    let nopen = emol.multiplicity.saturating_sub(1);
    let nocc_alpha = (nel + nopen) / 2;
    let nocc_beta = (nel - nopen.min(nel)) / 2;
    let scf = r.scf;
    let mos_beta = scf
        .mos_beta
        .ok_or_else(|| make_err("run_cdft: the UHF solve returned no beta MOs"))?;
    let density_beta = scf
        .density_beta
        .ok_or_else(|| make_err("run_cdft: the UHF solve returned no beta density"))?;
    Ok(PyCdftResult {
        energy: scf.energy,
        converged: scf.converged && max_constraint_error < lambda_tol,
        scf_converged: scf.converged,
        iterations: scf.iterations,
        outer_iterations: r.outer_iters,
        lambdas: r.lambdas,
        populations: r.populations,
        targets,
        residuals: r.residuals,
        max_constraint_error,
        lambda_tol,
        density_alpha_data: scf.density_alpha,
        density_beta_data: density_beta,
        eps_alpha_data: scf.eps_alpha,
        eps_beta_data: scf.eps_beta.unwrap_or_default(),
        mos_alpha: scf.mos_alpha,
        mos_beta,
        nocc_alpha,
        nocc_beta,
        spins,
        weight_matrices: r.weight_matrices,
        overlap,
        hamiltonian_key,
    })
}

/// Wu–Van Voorhis electronic coupling between two cDFT diabats.
#[pyclass]
#[pyo3(name = "CdftCouplingResult")]
pub(crate) struct PyCdftCouplingResult {
    /// Orthogonalized coupling H_ab (Ha). Its SIGN is a determinant-phase
    /// convention, not physics: compare |h_ab|.
    #[pyo3(get)]
    pub(crate) h_ab: f64,
    /// Determinant overlap <Psi_a|Psi_b> (product of the Lowdin-paired
    /// singular values over both spins).
    #[pyo3(get)]
    pub(crate) s_ab: f64,
    /// Diabat energies (Ha), each state's `CdftResult.energy`.
    #[pyo3(get)]
    pub(crate) e_a: f64,
    #[pyo3(get)]
    pub(crate) e_b: f64,
}

#[pymethods]
impl PyCdftCouplingResult {
    fn __repr__(&self) -> String {
        format!(
            "CdftCouplingResult(h_ab={:.10}, s_ab={:.6e}, e_a={:.10}, e_b={:.10})",
            self.h_ab, self.s_ab, self.e_a, self.e_b
        )
    }
}

/// Wu–Van Voorhis electron-transfer coupling H_ab between two `run_cdft`
/// diabats (J. Chem. Phys. 125, 164105 (2006)):
///
///   H_ab = [H_raw - (E_a + E_b) S_ab / 2] / (1 - S_ab^2),
///   H_raw = ½[(F_b S_ab - λ_b <a|W_b|b>) + (F_a S_ab - λ_a <a|W_a|b>)],
///   F_x   = E_x + λ_x N_x,   N_x = <x|W_x|x> (the state's own population),
///
/// with each state's energy, λ and W taken from its own result. The F form is
/// invariant under the empty redefinition W -> W - c·N̂ (target N -> N - c·N_e);
/// the form with E in place of F is not. No two-electron <a|H|b> is built.
///
/// Requirements (ValueError otherwise): each state has EXACTLY ONE constraint
/// and it is kind="charge" (the Rust kernel applies one operator to both spins
/// and takes a single λ), both states are `converged`, and both come from the
/// same molecule, geometry, basis, charge and multiplicity (checked through the
/// AO overlap matrix and the occupations) AND the same Hamiltonian: nuclei
/// (element, ghost flag, ECP, exact coordinates, charge), functional,
/// df_j_aux/df_k_aux, k_builder, XC grid, point_charges and external_field must
/// all match. Two (near-)identical states
/// (|S_ab| -> 1) make the coupling undefined and also raise.
#[pyfunction]
pub(crate) fn cdft_coupling(
    state_a: &PyCdftResult,
    state_b: &PyCdftResult,
) -> PyResult<PyCdftCouplingResult> {
    use ferric_dft::cdft::SpinChannel;
    use ferric_scf::cdft_coupling::{coupling_hab, DiabaticState};
    let value_err = |msg: String| pyo3::exceptions::PyValueError::new_err(msg);
    for (name, st) in [("state_a", state_a), ("state_b", state_b)] {
        if st.spins.len() != 1 || st.weight_matrices.len() != 1 || st.lambdas.len() != 1 {
            return Err(value_err(format!(
                "cdft_coupling: {name} has {} constraints; the Wu-Van Voorhis kernel \
                 takes exactly one charge constraint per state",
                st.spins.len()
            )));
        }
        if st.spins[0] != SpinChannel::Total {
            return Err(value_err(format!(
                "cdft_coupling: {name}'s constraint is kind='spin'; the coupling kernel \
                 supports kind='charge' only (a spin constraint acts with opposite sign \
                 on the two spins)"
            )));
        }
        if !st.converged {
            return Err(value_err(format!(
                "cdft_coupling: {name} is not converged (scf_converged={}, \
                 max_constraint_error={:.3e}, lambda_tol={:.1e}); a coupling between \
                 unconverged diabats is not meaningful",
                st.scf_converged, st.max_constraint_error, st.lambda_tol
            )));
        }
    }
    if state_a.overlap.dim() != state_b.overlap.dim() {
        return Err(value_err(format!(
            "cdft_coupling: the states have different basis sizes ({:?} vs {:?})",
            state_a.overlap.dim(),
            state_b.overlap.dim()
        )));
    }
    let s_diff = state_a
        .overlap
        .iter()
        .zip(state_b.overlap.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    if s_diff > 1e-10 {
        return Err(value_err(format!(
            "cdft_coupling: the states' AO overlap matrices differ by {s_diff:.3e}; \
             they were not computed for the same molecule, geometry and basis"
        )));
    }
    if state_a.nocc_alpha != state_b.nocc_alpha || state_a.nocc_beta != state_b.nocc_beta {
        return Err(value_err(format!(
            "cdft_coupling: occupations differ (alpha {}/{}, beta {}/{}); the states \
             must share charge and multiplicity",
            state_a.nocc_alpha, state_b.nocc_alpha, state_a.nocc_beta, state_b.nocc_beta
        )));
    }
    if state_a.hamiltonian_key != state_b.hamiltonian_key {
        return Err(value_err(format!(
            "cdft_coupling: the two states were solved with different Hamiltonians \
             (nuclei, functional, density fitting, exchange builder, XC grid or \
             external potential differ); the Wu-Van Voorhis coupling requires one shared H.\n  \
             state_a: {}\n  state_b: {}",
            state_a.hamiltonian_key, state_b.hamiltonian_key
        )));
    }
    let da = DiabaticState {
        c_a: &state_a.mos_alpha,
        c_b: &state_a.mos_beta,
        nocc_a: state_a.nocc_alpha,
        nocc_b: state_a.nocc_beta,
        energy: state_a.energy,
        lambda: state_a.lambdas[0],
        w: &state_a.weight_matrices[0],
    };
    let db = DiabaticState {
        c_a: &state_b.mos_alpha,
        c_b: &state_b.mos_beta,
        nocc_a: state_b.nocc_alpha,
        nocc_b: state_b.nocc_beta,
        energy: state_b.energy,
        lambda: state_b.lambdas[0],
        w: &state_b.weight_matrices[0],
    };
    let r = coupling_hab(&da, &db, &state_a.overlap);
    // The Rust kernel returns E_a as "H_ab" when 1 - S_ab^2 vanishes (the two
    // states coincide). That is a sentinel, not a coupling; refuse it here.
    if (1.0 - r.s_ab * r.s_ab).abs() < 1e-10 {
        return Err(value_err(format!(
            "cdft_coupling: |S_ab| = {:.12} is 1 to within 1e-10; the two states are \
             the same determinant and their coupling is undefined",
            r.s_ab.abs()
        )));
    }
    Ok(PyCdftCouplingResult {
        h_ab: r.h_ab,
        s_ab: r.s_ab,
        e_a: r.e_a,
        e_b: r.e_b,
    })
}
