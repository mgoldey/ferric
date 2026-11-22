//! TDDFT binding: moved verbatim from `lib.rs`.

use super::*;

// ── TDDFT (closed-shell linear response) ──

#[pyclass]
#[pyo3(name = "TddftResult")]
pub(crate) struct PyTddftResult {
    #[pyo3(get)]
    pub(crate) n_roots: usize,
    #[pyo3(get)]
    pub(crate) method: String,
    /// Exact-exchange fraction used on the `-c_HF` terms; for a
    /// range-separated hybrid the short-range fraction c_sr.
    #[pyo3(get)]
    pub(crate) c_hf: f64,
    /// Long-range exact-exchange fraction (equal to `c_hf` unless the
    /// functional is range-separated).
    #[pyo3(get)]
    pub(crate) c_lr: f64,
    /// Range-separation parameter in bohr^-1 (0 for a global hybrid).
    #[pyo3(get)]
    pub(crate) omega: f64,
    /// Spin of the excited states of a restricted reference: `"singlet"` or
    /// `"triplet"`; `None` on an unrestricted reference (read `s2`).
    #[pyo3(get)]
    pub(crate) spin: Option<String>,
    /// The reference the roots were computed on: `"restricted"` or
    /// `"unrestricted"`.
    #[pyo3(get)]
    pub(crate) reference: String,
    /// `<S^2>` of the reference determinant (0 for a restricted reference).
    #[pyo3(get)]
    pub(crate) s2_reference: f64,
    pub(crate) s2: Vec<f64>,
    /// Whether the (ia|f_xc|jb) kernel block was included (true for every KS
    /// reference, false only for an HF reference).
    #[pyo3(get)]
    pub(crate) fxc_included: bool,
    /// The main grid the KS reference and the f_xc kernel shared, as
    /// `(n_radial, n_angular, prune, weight_cut)` (`prune` is `None` for the
    /// flat grid); `None` for an HF reference.
    #[pyo3(get)]
    pub(crate) grid: Option<(usize, usize, Option<String>, Option<f64>)>,
    /// The eigensolver that ran: `"dense"` or `"davidson"`.
    #[pyo3(get)]
    pub(crate) solver: String,
    /// Davidson iterations (0 for the dense solver).
    #[pyo3(get)]
    pub(crate) iterations: usize,
    pub(crate) residuals: Vec<f64>,
    pub(crate) excitation_energies: Vec<f64>,
    pub(crate) oscillator_strengths: Vec<f64>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

#[pymethods]
impl PyTddftResult {
    /// Validation certificate (see `RhfResult.certificate`): capability
    /// `"tda"` or `"tddft"`, `grade` `"proven_narrow"` capped by the SCF
    /// reference's. A KS run records `response-grid` (the grid the reference
    /// SCF and the f_xc kernel shared, equal to `grid`, and whether it was the
    /// meta-GGA response default) and a meta-GGA run warns that its roots are
    /// grid-sensitive. Every run records `eigensolver` (solver, root count,
    /// largest residual).
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    #[getter]
    fn excitation_energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.excitation_energies)
    }
    /// `<S^2>` of each root, same order as the energies.
    #[getter]
    fn s2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.s2)
    }
    /// Residual norm of each root, same order as the energies.
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.residuals)
    }
    #[getter]
    fn oscillator_strengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.oscillator_strengths)
    }
    fn lowest_ev(&self) -> f64 {
        if self.excitation_energies.is_empty() {
            0.0
        } else {
            self.excitation_energies[0] * 27.211_386_245_988
        }
    }
    fn __repr__(&self) -> String {
        format!(
            "TddftResult(method={}, spin={}, n_roots={}, lowest={:.4} eV)",
            self.method,
            self.spin_label(),
            self.n_roots,
            self.lowest_ev(),
        )
    }
    fn __str__(&self) -> String {
        let ha_to_ev = 27.211_386_245_988;
        let mut s = format!(
            "TDDFT {} — {} {} roots:\n",
            self.method,
            self.n_roots,
            self.spin_label()
        );
        for (i, ((&e, &f), &s2)) in self
            .excitation_energies
            .iter()
            .zip(&self.oscillator_strengths)
            .zip(&self.s2)
            .enumerate()
        {
            s += &format!(
                "  {}: {:.6} Ha ({:.4} eV)  f = {:.6}  <S^2> = {:.4}\n",
                i + 1,
                e,
                e * ha_to_ev,
                f,
                s2
            );
        }
        s
    }
}

impl PyTddftResult {
    fn spin_label(&self) -> &str {
        self.spin.as_deref().unwrap_or("unrestricted")
    }
}

/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
///
/// With an ECP basis set the ECP is applied to the molecule, so the reference SCF is the one
/// `run_rhf` / `run_uhf` (`run_dft` / `run_uks` with `functional`) solves.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis, functional=None, n_roots=3, method="tda",
    scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None,
    grid_radial=None, grid_angular=None, grid_prune=None, grid_weight_cut=None, scf_ri_j_precision=None,
    solver="auto", spin="singlet", reference="auto",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_tddft(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    functional: Option<&str>,
    n_roots: usize,
    method: &str,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    grid_radial: Option<usize>,
    grid_angular: Option<usize>,
    grid_prune: Option<&str>,
    grid_weight_cut: Option<f64>,
    scf_ri_j_precision: Option<&str>,
    solver: &str,
    spin: &str,
    reference: &str,
) -> PyResult<PyTddftResult> {
    use ferric_tddft::{ResponseSpin, TddftConfig, TddftMethod, TddftSolver};
    let solver = TddftSolver::parse_config_str(solver)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("run_tddft: {e}")))?;
    let spin = ResponseSpin::parse_config_str(spin)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("run_tddft: {e}")))?;

    let unrestricted = match reference.to_ascii_lowercase().as_str() {
        "auto" => mol.inner.multiplicity > 1,
        "unrestricted" | "uhf" | "uks" => true,
        "restricted" | "rhf" | "rks" if mol.inner.multiplicity == 1 => false,
        "restricted" | "rhf" | "rks" => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "run_tddft: reference=\"{reference}\" needs a closed-shell molecule; this one has \
                 multiplicity {}",
                mol.inner.multiplicity
            )))
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "run_tddft: reference '{other}' is not one of \"auto\", \"restricted\", \
                 \"unrestricted\""
            )))
        }
    };

    if unrestricted && spin == ResponseSpin::Triplet {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "run_tddft: spin=\"triplet\" on an unrestricted reference; its spin-conserving \
             roots are not spin-adapted. Run the default and read TddftResult.s2.",
        ));
    }
    let sys = prepare(py, mol, basis_set)?;
    let dfbs = prepare_aux(py, &sys, auxbasis)?;
    let mol = sys.mol();

    let tddft_method = match method.to_lowercase().as_str() {
        "tda" | "cis" => TddftMethod::Tda,
        "casida" | "rpa" | "tddft" | "tdhf" => TddftMethod::Casida,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown TDDFT method '{method}'; expected 'tda' or 'casida'"
            )))
        }
    };

    // The f_xc kernel is built inside `run_tddft` from `functional` on the
    // SAME grid the KS reference is converged on (the ladder keeps
    // `dft_grid`), and the exact-exchange fraction comes from the same
    // functional. A functional without a complete kernel (VV10, a meta-GGA
    // without one) is refused before the SCF is spent on it.
    if let Some(xc_name) = functional {
        ferric_dft::lr_kernel::resolve_response_xc(xc_name, "run_tddft")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    }
    if functional.is_none()
        && (grid_radial.is_some()
            || grid_angular.is_some()
            || grid_prune.is_some()
            || grid_weight_cut.is_some())
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "run_tddft: grid_radial / grid_angular / grid_prune / grid_weight_cut size the \
             Kohn-Sham grid, and functional=None (an HF reference) builds none",
        ));
    }
    let mut scf_cfg = rhf_config();
    let (scf, grid) = if let Some(xc_name) = functional {
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv).apply(&mut scf_cfg);
        scf_cfg = ks_jkfit_config(scf_cfg, xc_name, scf_ri_j_precision)?;
        let cfg = &mut scf_cfg;
        // Explicit grid kwargs win; with none, a meta-GGA gets the finer
        // response grid, everything else the production grid.
        cfg.dft_grid =
            resolve_dft_grid_with_cut(grid_radial, grid_angular, grid_prune, grid_weight_cut)?
                .or_else(|| ferric_dft::grid::response_default_grid(Some(xc_name)));
        let grid = cfg
            .dft_grid
            .clone()
            .unwrap_or_else(ferric_dft::grid::AtomicGridConfig::production);
        if unrestricted {
            (sys.solve_uhf(cfg).map_err(make_err)?, grid)
        } else {
            let ladder = ferric_scf::ladder::ksdft_ladder(cfg);
            let lr = sys.solve_rhf_ladder(&ladder).map_err(make_err)?;
            (lr.result, grid)
        }
    } else {
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv).apply(&mut scf_cfg);
        let scf = if unrestricted {
            sys.solve_uhf(&scf_cfg)
        } else {
            sys.solve_rhf(&scf_cfg)
        }
        .map_err(make_err)?;
        (scf, ferric_dft::grid::AtomicGridConfig::default())
    };
    let scf = scf.into_converged().map_err(make_err)?;

    let grid_used = functional.map(|_| {
        (
            grid.n_radial,
            grid.n_angular,
            grid.prune.map(|p| {
                use ferric_dft::prune::PruneScheme as P;
                match p {
                    P::NwchemLike => "nwchem".to_string(),
                    P::NwchemRadial => "nwchem-radial".to_string(),
                    P::Sgx => "sgx".to_string(),
                    P::DefGrid(l) => {
                        use ferric_dft::prune::DefGridLevel as L;
                        let n = match l {
                            L::One => 1,
                            L::Two => 2,
                            L::Three => 3,
                        };
                        format!("defgrid{n}")
                    }
                }
            }),
            grid.weight_cut,
        )
    });
    let config = TddftConfig {
        n_roots,
        method: tddft_method,
        xc: functional.map(str::to_string),
        grid,
        solver,
        spin,
        ..Default::default()
    };
    // The closure borrows only Rust values (the prepared system and locals
    // built above), so nothing Python-borrowed crosses the boundary; safe to
    // release the GIL for the TDDFT compute.
    let r = py
        .allow_threads(|| ferric_tddft::run_tddft_on(&dfbs, &scf, &config))
        .map_err(make_err)?;
    let certificate_data = ferric_tddft::certificate::tddft_certificate(
        &reference_certificate(mol, &scf, &scf_cfg, &basis_set.inner),
        &auxbasis.inner.name,
        &config,
        &r,
        grid_radial.is_some()
            || grid_angular.is_some()
            || grid_prune.is_some()
            || grid_weight_cut.is_some(),
    );

    Ok(PyTddftResult {
        n_roots: r.excitation_energies.len(),
        method: format!("{:?}", r.method),
        c_hf: r.c_hf,
        c_lr: r.c_lr,
        omega: r.omega,
        spin: r.spin.map(|s| s.name().to_string()),
        reference: match r.reference {
            ferric_scf::Spin::Unrestricted => "unrestricted",
            _ => "restricted",
        }
        .to_string(),
        s2_reference: r.s2_reference,
        s2: r.s2,
        fxc_included: r.fxc_included,
        grid: grid_used,
        solver: r.solver.name().to_string(),
        iterations: r.iterations,
        residuals: r.residuals,
        certificate_data,
        excitation_energies: r.excitation_energies,
        oscillator_strengths: r.oscillator_strengths,
    })
}
