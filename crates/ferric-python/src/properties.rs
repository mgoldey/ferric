//! Electronic property bindings: moved verbatim from `lib.rs`.

use super::*;

// ── Electronic properties (ESP / Hirshfeld / Löwdin charges) ──

/// Accepts either an `RhfResult` or a `DftResult` — both carry a converged
/// closed-shell density, which is all these property functions need. Lets
/// `esp_at_atoms`/`hirshfeld_charges`/`lowdin_charges` work on any energy
/// result rather than being RHF-only.
pub(crate) enum DensitySource<'py> {
    Rhf(PyRef<'py, PyRhfResult>),
    Dft(PyRef<'py, PyDftResult>),
}

impl<'py> DensitySource<'py> {
    fn density(&self) -> &Array2<f64> {
        match self {
            DensitySource::Rhf(r) => &r.density_data,
            DensitySource::Dft(d) => &d.density_data,
        }
    }

    pub(crate) fn scf(&self) -> &ScfResult {
        match self {
            DensitySource::Rhf(r) => &r.scf_data,
            DensitySource::Dft(d) => &d.scf_data,
        }
    }

    fn scf_config(&self) -> &RhfConfig {
        match self {
            DensitySource::Rhf(r) => &r.scf_config,
            DensitySource::Dft(d) => &d.scf_config,
        }
    }
}

impl<'py> FromPyObject<'py> for DensitySource<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(r) = ob.extract::<PyRef<'py, PyRhfResult>>() {
            return Ok(DensitySource::Rhf(r));
        }
        if let Ok(d) = ob.extract::<PyRef<'py, PyDftResult>>() {
            return Ok(DensitySource::Dft(d));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected an RhfResult or DftResult",
        ))
    }
}

/// Electrostatic potential at each nucleus, in Hartree atomic units (e/Bohr).
/// `result` is an `RhfResult` or `DftResult` from a converged SCF.
#[pyfunction]
pub(crate) fn esp_at_atoms(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
) -> PyResult<Vec<f64>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    ferric_rpa::properties::esp_at_atoms(&mol.inner, &prep, result.density()).map_err(make_err)
}

/// Electrostatic potential at ARBITRARY points, in Hartree atomic units
/// (e/Bohr). `points` is an (N, 3) array of Cartesian positions in **Bohr**.
///
/// This is the primitive behind CHELPG/RESP and is what you want for an ESP on
/// a molecular surface (vdW shell, solvent-accessible grid, ...). Prefer it
/// over reconstructing the surface potential from partial charges plus induced
/// dipoles: that is a point-multipole approximation to a quantity computed
/// here exactly from the density.
///
/// `result` is an `RhfResult` or `DftResult` from a converged SCF.
#[pyfunction]
pub(crate) fn esp_at_points(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
    points: numpy::PyReadonlyArray2<f64>,
) -> PyResult<Vec<f64>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let arr = points.as_array();
    if arr.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "points must have shape (N, 3); got (_, {})",
            arr.ncols()
        )));
    }
    let pts: Vec<[f64; 3]> = arr.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    ferric_scf::properties::esp_at_points(&mol.inner, &prep, result.density(), &pts)
        .map_err(make_err)
}

/// Electrostatic potential on a molecular van der Waals surface.
///
/// A Lebedev sphere of `n_angular` points is placed at `vdw_scale` x the Bondi
/// radius of every atom; points inside another atom's scaled radius are
/// dropped. Returns `(points, esp, n_buried)`: `points` is an (M, 3) array in
/// **Bohr**, `esp` the potential there in Hartree/e (same sign convention as
/// `esp_at_points`), and `n_buried` the number of sample points dropped as
/// buried. `result` is an `RhfResult` or `DftResult` from a converged SCF.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, result, vdw_scale=1.4, n_angular=110))]
pub(crate) fn esp_on_surface<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
    vdw_scale: f64,
    n_angular: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<f64>, usize)> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let (pts, esp, n_buried) = ferric_scf::properties::esp_on_surface_counted(
        &mol.inner,
        &prep,
        result.density(),
        vdw_scale,
        n_angular,
    )
    .map_err(make_err)?;
    let mut arr = ndarray::Array2::<f64>::zeros((pts.len(), 3));
    for (i, p) in pts.iter().enumerate() {
        for (k, v) in p.iter().enumerate() {
            arr[(i, k)] = *v;
        }
    }
    Ok((PyArray2::from_array(py, &arr), esp, n_buried))
}

/// Hirshfeld partial charges (units of e). `result` is an `RhfResult` or
/// `DftResult` from a converged SCF.
///
/// `proatom` selects the free-atom reference density:
///   "scf"    (default) free-atom SCF densities in the molecule's own basis,
///            solved with the same SCF settings as `result` (method, functional,
///            density fitting, thresholds) — the proatom ferric-cli's
///            `[rpa] compute_hirshfeld_charges` uses
///            (`ferric_scf::properties::scf_proatom_provider`). An atom whose
///            free-atom SCF fails falls back to the Slater proatom, with a warning.
///   "slater" a single-exponential Slater proatom (xi = 1 / Bragg-Slater
///            radius). Qualitative: 0.23-0.72 e from "scf" on H2O, CO and
///            CH3OH. Prints a warning.
/// Any other value raises ValueError.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, result, proatom="scf"))]
pub(crate) fn hirshfeld_charges(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
    proatom: &str,
) -> PyResult<Vec<f64>> {
    let job = HirshfeldJob::new(mol, basis_set, &result, proatom)?;
    // Owned values only cross the GIL release: the free-atom SCFs can take
    // seconds per element.
    py.allow_threads(|| job.run()).map_err(make_err)
}

/// Owned inputs of one `hirshfeld_charges` call.
pub(crate) struct HirshfeldJob {
    pub(crate) mol: ferric_core::mol::Molecule,
    pub(crate) bs: ferric_core::basis::BasisSet,
    pub(crate) density: ndarray::Array2<f64>,
    pub(crate) config: RhfConfig,
    pub(crate) use_scf: bool,
}

impl HirshfeldJob {
    fn new(
        mol: &PyMolecule,
        basis_set: &PyBasisSet,
        result: &DensitySource,
        proatom: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            use_scf: parse_proatom(proatom)?,
            mol: mol.inner.clone(),
            bs: basis_set.inner.clone(),
            density: result.density().clone(),
            config: result.scf_config().clone(),
        })
    }

    fn run(&self) -> Result<Vec<f64>, ferric_core::FerricError> {
        hirshfeld_with_proatoms(
            &self.mol,
            &self.bs,
            &self.density,
            &self.config,
            self.use_scf,
        )
    }
}

/// `"scf"` → true (free-atom SCF proatoms, the CLI's), `"slater"` → false.
pub(crate) fn parse_proatom(proatom: &str) -> PyResult<bool> {
    match proatom {
        "scf" => Ok(true),
        "slater" => Ok(false),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown proatom {other:?}: expected \"scf\" or \"slater\""
        ))),
    }
}

pub(crate) fn hirshfeld_with_proatoms(
    mol: &ferric_core::mol::Molecule,
    bs: &ferric_core::basis::BasisSet,
    density: &ndarray::Array2<f64>,
    config: &RhfConfig,
    use_scf: bool,
) -> Result<Vec<f64>, ferric_core::FerricError> {
    if !use_scf {
        return ferric_rpa::properties::hirshfeld_charges(mol, bs, density, None);
    }
    let ctx = ParallelContext::default();
    let provider =
        ferric_scf::properties::scf_proatom_provider(&ctx, bs, Operator::coulomb(), config);
    let provider: &ferric_rpa::properties::ProatomProvider = &provider;
    ferric_rpa::properties::hirshfeld_charges(mol, bs, density, Some(provider))
}

/// Per-orbital centroids <p|r|p> (Bohr, list of [x,y,z]) and spatial spreads
/// sigma_p = sqrt(<r^2> - |<r>|^2) (Bohr) for the converged restricted MOs.
/// Mirrors Q-Chem 7's "second moments of orbitals" property; the spreads are
/// the pair-gate inputs of the amplitude-threshold local-correlation family.
#[pyfunction]
pub(crate) fn orbital_moments(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
) -> PyResult<(Vec<[f64; 3]>, Vec<f64>)> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let scf = result.scf();
    if scf.spin != ferric_scf::result::Spin::Restricted {
        return Err(make_err(ferric_core::FerricError::General(
            "orbital_moments: restricted (RHF/RKS) results only".into(),
        )));
    }
    let (centers, spreads) =
        ferric_integrals::oneelectron::orbital_moments(&prep, scf.mos_r()).map_err(make_err)?;
    let c: Vec<[f64; 3]> = (0..centers.nrows())
        .map(|p| [centers[(p, 0)], centers[(p, 1)], centers[(p, 2)]])
        .collect();
    Ok((c, spreads))
}

/// Electronic-density second-moment tensor (3x3 nested list, Bohr^2) about
/// the origin: M_xy = sum_uv D_uv <u|x y|v>; trace = <r^2> of the density.
#[pyfunction]
pub(crate) fn density_second_moment(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
) -> PyResult<[[f64; 3]; 3]> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    ferric_integrals::oneelectron::density_second_moment(&prep, result.density(), [0.0; 3])
        .map_err(make_err)
}

/// Validate an (N, 3) points array (Bohr) into a point list.
pub(crate) fn points_to_vec(points: &numpy::PyReadonlyArray2<f64>) -> PyResult<Vec<[f64; 3]>> {
    let arr = points.as_array();
    if arr.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "points must have shape (N, 3); got (_, {})",
            arr.ncols()
        )));
    }
    Ok(arr.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

/// AO values chi[mu, g] on `pts`, with the memory budget checked first.
pub(crate) fn ao_values_checked(
    mol: &ferric_core::mol::Molecule,
    bs: &ferric_core::basis::BasisSet,
    pts: &[[f64; 3]],
) -> PyResult<Array2<f64>> {
    let nbf = ferric_integrals::ao_grid::nbasis(mol, bs).map_err(make_err)?;
    ferric_core::memory::check_alloc(
        &format!("AO-on-points (nbf={nbf}, npts={})", pts.len()),
        nbf.saturating_mul(pts.len()).saturating_mul(8),
        ferric_core::memory::resolve_budget_bytes(None),
    )
    .map_err(make_err)?;
    ferric_integrals::ao_grid::eval_basis_on_points(mol, bs, pts).map_err(make_err)
}

/// rho[g] = sum_{mu nu} D[mu, nu] chi[mu, g] chi[nu, g] from AO values `chi` (nbf, npts).
pub(crate) fn density_from_ao_values(d: &Array2<f64>, chi: &Array2<f64>) -> Vec<f64> {
    let dchi = d.dot(chi);
    (0..chi.ncols())
        .map(|g| chi.column(g).dot(&dchi.column(g)))
        .collect()
}

/// Atomic orbital values chi_mu(r) at arbitrary points, as an (N, nbasis)
/// array (PySCF `eval_gto(...)` layout, transposed from ferric's internal
/// (nbasis, N)). `points` is an (N, 3) array of Cartesian positions in
/// **Bohr**; values are in Bohr^(-3/2). Column order is the basis-function
/// order of the SCF (`result.mo_coefficients` rows).
#[pyfunction]
pub(crate) fn eval_ao_on_points<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    points: numpy::PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pts = points_to_vec(&points)?;
    let chi = ao_values_checked(&mol.inner, &basis_set.inner, &pts)?;
    Ok(PyArray2::from_array(py, &chi.t().to_owned()))
}

/// Electron density rho(r) = sum_{mu nu} D_{mu nu} chi_mu(r) chi_nu(r)
/// (total, alpha + beta) at arbitrary points, in e/Bohr^3. `points` is an
/// (N, 3) array in **Bohr**. `result` is an `RhfResult` or `DftResult`.
#[pyfunction]
pub(crate) fn eval_density_on_points(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
    points: numpy::PyReadonlyArray2<f64>,
) -> PyResult<Vec<f64>> {
    let pts = points_to_vec(&points)?;
    let chi = ao_values_checked(&mol.inner, &basis_set.inner, &pts)?;
    Ok(density_from_ao_values(result.density(), &chi))
}

/// Restricted molecular orbital psi_i(r) = sum_mu C_{mu i} chi_mu(r) at
/// arbitrary points: returns an (N, len(mo_indices)) array in Bohr^(-3/2).
/// `mo_indices` are 0-based columns of `result.mo_coefficients` (energy
/// order). `points` is an (N, 3) array in **Bohr**. Restricted (RHF/RKS)
/// results only; orbital signs are the SCF's arbitrary phase.
#[pyfunction]
pub(crate) fn eval_mo_on_points<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
    mo_indices: Vec<usize>,
    points: numpy::PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let scf = result.scf();
    if scf.spin != ferric_scf::result::Spin::Restricted {
        return Err(make_err(
            "eval_mo_on_points: restricted (RHF/RKS) results only",
        ));
    }
    let c = scf.mos_r();
    if let Some(&bad) = mo_indices.iter().find(|&&i| i >= c.ncols()) {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "MO index {bad} out of range (result has {} MOs)",
            c.ncols()
        )));
    }
    let pts = points_to_vec(&points)?;
    let chi = ao_values_checked(&mol.inner, &basis_set.inner, &pts)?;
    let mut out = Array2::<f64>::zeros((pts.len(), mo_indices.len()));
    for (k, &i) in mo_indices.iter().enumerate() {
        let psi = c.column(i).dot(&chi);
        out.column_mut(k).assign(&psi);
    }
    Ok(PyArray2::from_array(py, &out))
}

/// Axis-aligned cube grid with isotropic `spacing` (Bohr). An omitted
/// `origin`/`shape` comes from the atoms' bounding box plus `padding`.
pub(crate) fn cube_grid_spec(
    mol: &PyMolecule,
    spacing: f64,
    padding: f64,
    origin: Option<[f64; 3]>,
    shape: Option<[usize; 3]>,
) -> PyResult<ferric_integrals::ao_grid::GridSpec> {
    if !(spacing.is_finite() && spacing > 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "spacing must be a positive number of Bohr",
        ));
    }
    let atoms = &mol.inner.atoms;
    if atoms.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "molecule has no atoms",
        ));
    }
    let origin = origin.unwrap_or_else(|| {
        let mut lo = [f64::INFINITY; 3];
        for a in atoms {
            for (k, v) in [a.x, a.y, a.zpos].iter().enumerate() {
                lo[k] = lo[k].min(*v);
            }
        }
        [lo[0] - padding, lo[1] - padding, lo[2] - padding]
    });
    let shape = shape.unwrap_or_else(|| {
        let mut hi = [f64::NEG_INFINITY; 3];
        for a in atoms {
            for (k, v) in [a.x, a.y, a.zpos].iter().enumerate() {
                hi[k] = hi[k].max(*v);
            }
        }
        let n = |k: usize| (((hi[k] + padding) - origin[k]) / spacing).ceil() as usize + 1;
        [n(0), n(1), n(2)]
    });
    if shape.contains(&0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "shape entries must be >= 1",
        ));
    }
    Ok(ferric_integrals::ao_grid::GridSpec {
        origin,
        n_x: shape[0],
        n_y: shape[1],
        n_z: shape[2],
        step_x: [spacing, 0.0, 0.0],
        step_y: [0.0, spacing, 0.0],
        step_z: [0.0, 0.0, spacing],
    })
}

/// Write a Gaussian cube file of the total electron density or one
/// restricted MO on an axis-aligned regular grid.
///
/// `what` is `"density"` (e/Bohr^3) or `"mo"` (psi_i, Bohr^(-3/2); needs
/// `mo_index`, 0-based, restricted results only). All lengths are **Bohr**,
/// matching the cube format (positive voxel counts mean Bohr). The grid is
/// `shape = (nx, ny, nz)` points starting at `origin` with isotropic
/// `spacing`; when `origin`/`shape` are omitted they are derived from the
/// atoms' bounding box plus `padding`. Returns `(origin, shape)` actually
/// used. `result` is an `RhfResult` or `DftResult`.
#[pyfunction]
#[pyo3(signature = (path, mol, basis_set, result, what="density", mo_index=None, spacing=0.3, padding=4.0, origin=None, shape=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_cube(
    path: &str,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
    what: &str,
    mo_index: Option<usize>,
    spacing: f64,
    padding: f64,
    origin: Option<[f64; 3]>,
    shape: Option<[usize; 3]>,
) -> PyResult<([f64; 3], [usize; 3])> {
    let grid = cube_grid_spec(mol, spacing, padding, origin, shape)?;
    let (origin, shape) = (grid.origin, [grid.n_x, grid.n_y, grid.n_z]);
    let (vals, comment): (Vec<f64>, String) = match what {
        "density" => {
            let chi = ferric_export::eval_basis_on_grid(&mol.inner, &basis_set.inner, &grid)
                .map_err(make_err)?;
            (
                density_from_ao_values(result.density(), &chi),
                "total electron density (e/Bohr^3), lengths in Bohr".to_string(),
            )
        }
        "mo" => {
            let i = mo_index.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("what=\"mo\" requires mo_index")
            })?;
            let scf = result.scf();
            if scf.spin != ferric_scf::result::Spin::Restricted {
                return Err(make_err(
                    "write_cube: MO cubes need a restricted (RHF/RKS) result",
                ));
            }
            let c = scf.mos_r();
            if i >= c.ncols() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "MO index {i} out of range (result has {} MOs)",
                    c.ncols()
                )));
            }
            let chi = ferric_export::eval_basis_on_grid(&mol.inner, &basis_set.inner, &grid)
                .map_err(make_err)?;
            (
                c.column(i).dot(&chi).to_vec(),
                format!("MO {i} (psi, Bohr^-3/2), lengths in Bohr"),
            )
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown what {other:?}: expected \"density\" or \"mo\""
            )))
        }
    };
    let data =
        ndarray::Array3::from_shape_vec((shape[0], shape[1], shape[2]), vals).map_err(make_err)?;
    ferric_export::export_cube(path, &mol.inner, &grid, &data, &comment).map_err(make_err)?;
    Ok((origin, shape))
}

/// Löwdin (symmetric-orthogonalization) partial charges (units of e).
/// Closed-shell only. `result` is an `RhfResult` or `DftResult` from a
/// converged SCF.
#[pyfunction]
pub(crate) fn lowdin_charges(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
) -> PyResult<Vec<f64>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    ferric_rpa::properties::lowdin_charges(&mol.inner, &prep, result.density()).map_err(make_err)
}

/// Mulliken partial charges (units of e), the standard textbook population
/// analysis. Known to be more basis-set-sensitive than Löwdin (prefer
/// `lowdin_charges` for a more basis-stable partition); included as the
/// standard baseline every QC package provides. Closed-shell only. `result`
/// is an `RhfResult` or `DftResult` from a converged SCF.
#[pyfunction]
pub(crate) fn mulliken_charges(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
) -> PyResult<Vec<f64>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    ferric_rpa::properties::mulliken_charges(&mol.inner, &prep, result.density()).map_err(make_err)
}

/// CHELPG (CHarges from Electrostatic Potentials, Grid-based) partial
/// charges (units of e). Structurally different from
/// `hirshfeld_charges`/`lowdin_charges`/`mulliken_charges`: those split the
/// electron density directly among atoms (population partition); CHELPG
/// instead fits atom-centered point charges to best reproduce the molecular
/// electrostatic potential on a grid around the molecule (Breneman & Wiberg,
/// J. Comput. Chem. 11, 361 (1990)). Closed-shell only. `result` is an
/// `RhfResult` or `DftResult` from a converged SCF.
#[pyfunction]
pub(crate) fn chelpg_charges(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
) -> PyResult<Vec<f64>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    ferric_rpa::properties::chelpg_charges(&mol.inner, &prep, result.density()).map_err(make_err)
}

/// RESP (Restrained ElectroStatic Potential) partial charges (units of e).
/// Same ESP grid-fitting as `chelpg_charges`, plus a hyperbolic restraint
/// damping non-hydrogen atomic charges toward zero (Bayly, Cieplak,
/// Cornell, Kollman, J. Phys. Chem. 97, 10269 (1993)). This is a
/// single-stage restrained fit with the standard literature weight/
/// tightness parameters — NOT the full multi-stage/multi-conformer RESP
/// averaging procedure. Closed-shell only. `result` is an `RhfResult` or
/// `DftResult` from a converged SCF.
#[pyfunction]
pub(crate) fn resp_charges(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
) -> PyResult<Vec<f64>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    ferric_rpa::properties::resp_charges(&mol.inner, &prep, result.density()).map_err(make_err)
}

/// Per-atom Hirshfeld-partitioned static dipole polarizability tensors
/// (Bohr^3), via true PDEP-RPA α(ω=0) on a real-space Becke/Hirshfeld grid.
/// Closed-shell (Restricted) only — `result` must come from `run_rhf` or
/// `run_dft` on a closed-shell molecule. `auxbasis` is the RI auxiliary basis
/// (e.g. a `*-rifit`/`*-ri` bundled set, NOT the SCF's own `df_j_aux`).
///
/// This is materially more expensive than `esp_at_atoms`/`hirshfeld_charges`/
/// `lowdin_charges`: it builds RI intermediates plus an AO-value grid over
/// the whole molecule (`FERRIC_HIRSHFELD_SPACING`/`FERRIC_HIRSHFELD_MARGIN`
/// env vars control resolution vs. memory; the grid is budget-checked
/// against `memory_budget_gb`, auto-resolved to 80% of available RAM when
/// omitted — see `ferric_core::memory::resolve_budget_bytes`).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, result, memory_budget_gb=None))]
pub(crate) fn hirshfeld_polarizability<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    result: DensitySource,
    memory_budget_gb: Option<f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let cfg = ferric_rpa::PdepRpaConfig {
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        ..Default::default()
    };
    let alpha = ferric_rpa::properties::pdep_polarizability_hirshfeld(
        &mol.inner,
        &prep,
        &basis_set.inner,
        &dfbs,
        result.scf(),
        op,
        &cfg,
        None,
    )
    .map_err(make_err)?;
    let natoms = alpha.len();
    let mut arr = Array3::<f64>::zeros((natoms, 3, 3));
    for (a, tensor) in alpha.iter().enumerate() {
        for i in 0..3 {
            for j in 0..3 {
                arr[(a, i, j)] = tensor[i][j];
            }
        }
    }
    Ok(PyArray3::from_owned_array(py, arr))
}
