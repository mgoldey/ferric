//! Python bindings for molecular dynamics (`run_aimd`, `run_ffmd`).
//!
//! Both wrap the `ferric-md` integrators: `run_aimd` through
//! `ferric_scf::run_aimd` (Born-Oppenheimer MD on the SCF/KS surface, the
//! driver the CLI's `task = "md"` calls), `run_ffmd` through
//! `ferric_mm::run_ffmd` (the AMBER-form force field of an `MmTopology`).
//! User units: fs, K, ps^-1, Angstrom; energies in Hartree.

use super::*;
use ferric_core::certificate::Certificate;
use ferric_md::{
    EnergyCsvWriter, MdConfig, MdObserver, MdSystem, Thermostat, XyzTrajectoryWriter,
    ANGSTROM_TO_BOHR, AU_TIME_FS, BOHR_TO_ANGSTROM,
};

/// Result of `run_aimd` / `run_ffmd`: one entry per recorded frame (step 0
/// and every `stride`-th step).
#[pyclass]
#[pyo3(name = "MdResult")]
pub(crate) struct PyMdResult {
    md: ferric_md::MdResult,
    mol_data: Option<Molecule>,
    scf_iterations_data: Option<Vec<usize>>,
    certificate_data: Certificate,
}

impl PyMdResult {
    fn column(&self, f: impl Fn(&ferric_md::MdFrame) -> f64) -> Vec<f64> {
        self.md.frames.iter().map(f).collect()
    }

    /// `(n_frames, n_atoms, 3)` stack of `frames`, each scaled by `scale`.
    fn stack(frames: &[Array2<f64>], scale: f64) -> Array3<f64> {
        let n_atoms = frames.first().map_or(0, |a| a.nrows());
        let mut out = Array3::<f64>::zeros((frames.len(), n_atoms, 3));
        for (mut slab, a) in out.outer_iter_mut().zip(frames) {
            slab.assign(&a.mapv(|x| x * scale));
        }
        out
    }
}

#[pymethods]
impl PyMdResult {
    /// Validation certificate (dict form of
    /// `ferric_core::certificate::Certificate`). Always `not graded`, with
    /// capability `"md:<label>"`; the integrator, thermostat, time step and
    /// seed are recorded as approximations. For `run_aimd` the SCF at the
    /// final geometry is folded in as the reference.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        crate::certificate_dict(py, &self.certificate_data)
    }
    /// Step index of each recorded frame.
    #[getter]
    fn steps(&self) -> Vec<usize> {
        self.md.frames.iter().map(|f| f.step).collect()
    }
    /// Simulation time of each frame, fs.
    #[getter]
    fn times_fs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.column(|f| f.time_fs))
    }
    /// Potential energy of each frame, Hartree.
    #[getter]
    fn potential_energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.column(|f| f.potential))
    }
    /// Kinetic energy of each frame, Hartree.
    #[getter]
    fn kinetic_energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.column(|f| f.kinetic))
    }
    /// Potential plus kinetic energy of each frame, Hartree.
    #[getter]
    fn total_energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.column(|f| f.total))
    }
    /// Conserved quantity of each frame, Hartree: the total energy minus the
    /// kinetic energy the thermostat put in (equal to the total under NVE).
    #[getter]
    fn conserved<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.column(|f| f.conserved))
    }
    /// Instantaneous temperature of each frame, K (over `n_dof`).
    #[getter]
    fn temperatures<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.column(|f| f.temperature_k))
    }
    /// Positions of each frame, `(n_frames, n_atoms, 3)`, Angstrom.
    #[getter]
    fn positions_angstrom<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        PyArray3::from_owned_array(py, Self::stack(&self.md.positions, BOHR_TO_ANGSTROM))
    }
    /// Velocities of each frame, `(n_frames, n_atoms, 3)`, Angstrom / fs.
    #[getter]
    fn velocities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let scale = BOHR_TO_ANGSTROM / AU_TIME_FS;
        PyArray3::from_owned_array(py, Self::stack(&self.md.velocities, scale))
    }
    /// Degrees of freedom the temperatures are computed over.
    #[getter]
    fn n_dof(&self) -> usize {
        self.md.n_dof
    }
    /// Number of energy/gradient evaluations (`n_steps + 1`).
    #[getter]
    fn n_force_evaluations(&self) -> usize {
        self.md.n_force_evaluations
    }
    /// Largest `|conserved - conserved[0]|` over the frames, Hartree.
    #[getter]
    fn max_conserved_drift(&self) -> f64 {
        self.md.max_conserved_drift()
    }
    /// The force provider's label (`"rhf/sto-3g"`, `"mm"`, ...).
    #[getter]
    fn label(&self) -> String {
        self.md.label.clone()
    }
    /// SCF iterations per force evaluation (`run_aimd`), else `None`.
    #[getter]
    fn scf_iterations(&self) -> Option<Vec<usize>> {
        self.scf_iterations_data.clone()
    }
    /// The final geometry as a new `Molecule` (`run_aimd`); `None` for
    /// `run_ffmd`.
    fn mol(&self) -> Option<PyMolecule> {
        self.mol_data.clone().map(|inner| PyMolecule { inner })
    }
    fn __repr__(&self) -> String {
        format!(
            "MdResult(label={:?}, n_frames={}, n_dof={}, max_conserved_drift={:.3e})",
            self.md.label,
            self.md.frames.len(),
            self.md.n_dof,
            self.md.max_conserved_drift()
        )
    }
}

fn value_error(msg: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.into())
}

/// The MD kwargs shared by `run_aimd` and `run_ffmd`.
struct MdKwargs<'a> {
    n_steps: usize,
    dt_fs: f64,
    thermostat: &'a str,
    temperature_k: Option<f64>,
    init_temperature_k: Option<f64>,
    tau_fs: Option<f64>,
    friction_per_ps: Option<f64>,
    seed: u64,
    remove_com_motion: bool,
    stride: usize,
}

impl MdKwargs<'_> {
    /// The validated `MdConfig` for a system of `n_atoms` atoms; every
    /// inconsistency is a `ValueError` raised before any force evaluation.
    fn config(&self, who: &str, n_atoms: usize) -> PyResult<MdConfig> {
        let thermostat =
            Thermostat::parse_config_str(self.thermostat, self.tau_fs, self.friction_per_ps)
                .map_err(|e| value_error(format!("{who}: {e}")))?;
        let config = MdConfig {
            n_steps: self.n_steps,
            dt_fs: self.dt_fs,
            thermostat,
            temperature_k: self.temperature_k,
            init_temperature_k: self.init_temperature_k,
            initial_velocities: None,
            seed: self.seed,
            remove_com_motion: self.remove_com_motion,
            stride: self.stride,
            keep_trajectory: true,
        };
        config
            .validate(n_atoms)
            .map_err(|e| value_error(format!("{who}: {e}")))?;
        Ok(config)
    }
}

/// The optional trajectory (XYZ, Angstrom) and energy (CSV) writers, created
/// (truncated) before the run so a bad path fails first.
struct Writers {
    xyz: Option<XyzTrajectoryWriter>,
    csv: Option<EnergyCsvWriter>,
}

impl Writers {
    fn open(trajectory_path: Option<&str>, energies_path: Option<&str>) -> PyResult<Self> {
        let io = |e: ferric_core::FerricError| pyo3::exceptions::PyOSError::new_err(e.to_string());
        Ok(Self {
            xyz: trajectory_path
                .map(|p| XyzTrajectoryWriter::create(std::path::Path::new(p)))
                .transpose()
                .map_err(io)?,
            csv: energies_path
                .map(|p| EnergyCsvWriter::create(std::path::Path::new(p)))
                .transpose()
                .map_err(io)?,
        })
    }

    fn observers(&mut self) -> Vec<&mut dyn MdObserver> {
        let mut out: Vec<&mut dyn MdObserver> = Vec::new();
        if let Some(w) = self.xyz.as_mut() {
            out.push(w);
        }
        if let Some(w) = self.csv.as_mut() {
            out.push(w);
        }
        out
    }
}

/// The ungraded MD certificate: capability `md:<label>`, with the
/// integration settings as approximations.
fn md_certificate(md: &ferric_md::MdResult) -> Certificate {
    let c = &md.config;
    let target = c.temperature_k.unwrap_or(f64::NAN);
    let (integrator, thermostat) = match c.thermostat {
        Thermostat::Nve => ("velocity Verlet", "none (NVE)".to_string()),
        Thermostat::Berendsen { tau_fs } => (
            "velocity Verlet",
            format!(
                "Berendsen, tau = {tau_fs} fs, T = {target} K (equilibration only; not canonical)"
            ),
        ),
        Thermostat::Langevin { friction_per_ps } => (
            "Langevin BAOAB",
            format!("Langevin, friction = {friction_per_ps} ps^-1, T = {target} K"),
        ),
    };
    Certificate::ungraded(format!("md:{}", md.label))
        .with_build(ferric_core::certificate::BuildStamp::new(
            ferric_build_info::VERSION,
            ferric_build_info::COMMIT,
            ferric_build_info::DIRTY,
        ))
        .with_approximation("integrator", integrator)
        .with_approximation("thermostat", thermostat)
        .with_approximation(
            "time-step",
            format!("dt = {} fs, {} steps", c.dt_fs, c.n_steps),
        )
        .with_approximation("seed", c.seed.to_string())
        .warn(format!(
            "molecular dynamics has no validation grade in ferric; max conserved-quantity \
             drift {:.3e} Hartree over the run",
            md.max_conserved_drift()
        ))
}

/// Strict `reference=` of `run_aimd`; unset is RHF for a singlet and UHF
/// otherwise.
fn parse_aimd_reference(reference: Option<&str>, mol: &Molecule) -> PyResult<ferric_scf::Spin> {
    match reference {
        None if mol.multiplicity > 1 => Ok(ferric_scf::Spin::Unrestricted),
        None | Some("rhf") => Ok(ferric_scf::Spin::Restricted),
        Some("uhf") => Ok(ferric_scf::Spin::Unrestricted),
        Some("rohf") => Ok(ferric_scf::Spin::RestrictedOpen),
        Some(other) => Err(value_error(format!(
            "run_aimd: unknown reference {other:?}; expected \"rhf\", \"uhf\" or \"rohf\""
        ))),
    }
}

/// Born-Oppenheimer molecular dynamics on the SCF surface (Hartree-Fock, or
/// Kohn-Sham with `xc=`), the driver of the CLI's `task = "md"`.
///
/// `thermostat` is `"nve"` (velocity Verlet), `"berendsen"` (needs `tau_fs`
/// and `temperature_k`; equilibration only, it does not sample the canonical
/// ensemble) or `"langevin"` (BAOAB; needs `friction_per_ps` and
/// `temperature_k`). A parameter of a thermostat that was not chosen raises.
/// Initial velocities are Maxwell-Boltzmann at `init_temperature_k` (unset:
/// `temperature_k`; neither: zero) from `seed`, with the centre-of-mass
/// motion removed (`remove_com_motion`) and rescaled to that temperature
/// exactly. Every `stride`-th step is recorded. `reference`, `xc`,
/// `dispersion`, `df_j_aux` / `df_k_aux` and the `scf_*` kwargs as in
/// `run_optimize`; each step's SCF starts from the previous step's orbitals
/// unless `scf_warm_start=False`. `masses_amu` overrides the standard masses
/// (needed beyond Z = 54). `trajectory_path` / `energies_path` write a
/// multi-frame XYZ (Angstrom) and a CSV of the energies as the run goes.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_name, n_steps, dt_fs=0.5, thermostat="nve", temperature_k=None,
    init_temperature_k=None, tau_fs=None, friction_per_ps=None, seed=0,
    remove_com_motion=true, stride=1, reference=None, xc=None, dispersion=None,
    df_j_aux=None, df_k_aux=None, scf_max_iter=None, scf_energy_conv=None,
    scf_density_conv=None, scf_warm_start=None, masses_amu=None,
    trajectory_path=None, energies_path=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_aimd(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_name: &str,
    n_steps: usize,
    dt_fs: f64,
    thermostat: &str,
    temperature_k: Option<f64>,
    init_temperature_k: Option<f64>,
    tau_fs: Option<f64>,
    friction_per_ps: Option<f64>,
    seed: u64,
    remove_com_motion: bool,
    stride: usize,
    reference: Option<&str>,
    xc: Option<&str>,
    dispersion: Option<&str>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    scf_warm_start: Option<bool>,
    masses_amu: Option<Vec<f64>>,
    trajectory_path: Option<&str>,
    energies_path: Option<&str>,
) -> PyResult<PyMdResult> {
    let spin = parse_aimd_reference(reference, &mol.inner)?;
    let mut cfg = RhfConfig::default();
    ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv).apply(&mut cfg);
    cfg.xc = xc.map(str::to_string);
    cfg.df_j_aux = crate::geometry_df_aux(xc, df_j_aux);
    cfg.df_k_aux = crate::geometry_df_aux(xc, df_k_aux);
    geometry_task_scf(&mut cfg, None)?;
    let bs = basis::bundled(basis_name).map_err(make_err)?;
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&bs);
    let system = MdSystem::from_molecule(&emol, masses_amu.as_deref())
        .map_err(|e| value_error(format!("run_aimd: {e}")))?;
    let kw = MdKwargs {
        n_steps,
        dt_fs,
        thermostat,
        temperature_k,
        init_temperature_k,
        tau_fs,
        friction_per_ps,
        seed,
        remove_com_motion,
        stride,
    };
    let md = kw.config("run_aimd", system.n_atoms())?;
    let fref = match spin {
        ferric_scf::Spin::Restricted => ferric_scf::frequencies::FrequencyReference::Rhf,
        ferric_scf::Spin::Unrestricted => ferric_scf::frequencies::FrequencyReference::Uhf,
        ferric_scf::Spin::RestrictedOpen => ferric_scf::frequencies::FrequencyReference::Rohf,
    };
    let spec = resolve_frequency_dispersion(dispersion, xc, fref, &cfg)?;
    let config = ferric_scf::AimdConfig {
        md,
        scf_warm_start: scf_warm_start.unwrap_or(true),
        masses_amu,
    };
    let mut writers = Writers::open(trajectory_path, energies_path)?;
    let r = py
        .allow_threads(|| {
            aimd(
                &emol,
                &bs,
                &cfg,
                spin,
                &config,
                spec.as_ref(),
                &mut writers.observers(),
            )
        })
        .map_err(make_ferric_err)?;
    let reference = stamped_scf_certificate(&r.mol, &r.last_scf, &cfg.for_geometry_task());
    let mut certificate_data = md_certificate(&r.md).with_reference(&reference);
    if let Some(d) = dispersion {
        certificate_data = certificate_data.with_approximation(
            "dispersion",
            format!("{d} in the energy and the gradient (KS + dispersion surface)"),
        );
    }
    Ok(PyMdResult {
        md: r.md,
        mol_data: Some(r.mol),
        scf_iterations_data: Some(r.scf_iterations),
        certificate_data,
    })
}

/// `ferric_scf::run_aimd` with `spec`'s dispersion energy and analytic
/// gradient added at every step (none: no correction).
fn aimd(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    cfg: &RhfConfig,
    spin: ferric_scf::Spin,
    config: &ferric_scf::AimdConfig,
    spec: Option<&DispersionSpec>,
    observers: &mut [&mut dyn MdObserver],
) -> Result<ferric_scf::AimdResult, ferric_core::FerricError> {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let Some(spec) = spec else {
        return ferric_scf::run_aimd(&ctx, mol, &bs.name, op, cfg, spin, config, None, observers);
    };
    // MBD@rsSCS free-atom references are per element: built once.
    let cache = match spec {
        DispersionSpec::Mbd(_) => Some(ferric_rpa::dispersion::mbd_scf::MbdFreeAtomCache::build(
            &ctx, mol, bs, op, cfg,
        )?),
        DispersionSpec::D3Bj(_) => None,
    };
    let mut correction = |m: &Molecule, scf: &ScfResult| {
        let d = evaluate_dispersion(spec, &ctx, m, bs, op, cfg, scf, true, cache.as_ref())?;
        Ok((d.energy, d.gradient))
    };
    ferric_scf::run_aimd(
        &ctx,
        mol,
        &bs.name,
        op,
        cfg,
        spin,
        config,
        Some(&mut correction as &mut ferric_scf::ScfCorrection<'_>),
        observers,
    )
}

/// Molecular dynamics on the AMBER-form force field of `topology` (an
/// `MmTopology`, e.g. from `MmTopology.from_amber_units`).
///
/// `symbols` and `coords_angstrom` (`(n_atoms, 3)`, Angstrom) give the atoms
/// in topology order; the topology carries no masses, so they come from the
/// symbols' standard atomic masses unless `masses_amu` is given.
/// `cutoff_angstrom` switches the nonbonded sum to OpenMM's
/// `CutoffNonPeriodic` (reaction-field Coulomb, truncated LJ); unset sums
/// every pair. The MD kwargs and the output files as in `run_aimd`.
#[pyfunction]
#[pyo3(signature = (
    topology, symbols, coords_angstrom, n_steps, dt_fs=1.0, thermostat="nve",
    temperature_k=None, init_temperature_k=None, tau_fs=None, friction_per_ps=None,
    seed=0, remove_com_motion=true, stride=1, masses_amu=None, cutoff_angstrom=None,
    trajectory_path=None, energies_path=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_ffmd(
    py: Python<'_>,
    topology: &PyMmTopology,
    symbols: Vec<String>,
    coords_angstrom: Vec<Vec<f64>>,
    n_steps: usize,
    dt_fs: f64,
    thermostat: &str,
    temperature_k: Option<f64>,
    init_temperature_k: Option<f64>,
    tau_fs: Option<f64>,
    friction_per_ps: Option<f64>,
    seed: u64,
    remove_com_motion: bool,
    stride: usize,
    masses_amu: Option<Vec<f64>>,
    cutoff_angstrom: Option<f64>,
    trajectory_path: Option<&str>,
    energies_path: Option<&str>,
) -> PyResult<PyMdResult> {
    let system = ffmd_system(&topology.inner, symbols, &coords_angstrom, masses_amu)?;
    let method = ffmd_method(cutoff_angstrom)?;
    let kw = MdKwargs {
        n_steps,
        dt_fs,
        thermostat,
        temperature_k,
        init_temperature_k,
        tau_fs,
        friction_per_ps,
        seed,
        remove_com_motion,
        stride,
    };
    let config = kw.config("run_ffmd", system.n_atoms())?;
    let mut writers = Writers::open(trajectory_path, energies_path)?;
    let top = &topology.inner;
    let md = py
        .allow_threads(|| {
            ferric_mm::run_ffmd(top, &system, &method, &config, &mut writers.observers())
        })
        .map_err(make_ferric_err)?;
    let certificate_data = md_certificate(&md);
    Ok(PyMdResult {
        md,
        mol_data: None,
        scf_iterations_data: None,
        certificate_data,
    })
}

/// `cutoff_angstrom` of `run_ffmd`: unset sums every pair, a positive value
/// is OpenMM's `CutoffNonPeriodic` at that distance.
fn ffmd_method(cutoff_angstrom: Option<f64>) -> PyResult<ferric_mm::NonbondedMethod> {
    match cutoff_angstrom {
        None => Ok(ferric_mm::NonbondedMethod::NoCutoff),
        Some(c) if c.is_finite() && c > 0.0 => Ok(ferric_mm::NonbondedMethod::CutoffNonPeriodic(
            ferric_mm::CutoffParams::new(c * ANGSTROM_TO_BOHR),
        )),
        Some(c) => Err(value_error(format!(
            "run_ffmd: cutoff_angstrom must be finite and positive, got {c}"
        ))),
    }
}

/// The `MdSystem` of `run_ffmd`: symbols and coordinates checked against the
/// topology, masses from the symbols unless given.
fn ffmd_system(
    top: &ferric_mm::MmTopology,
    symbols: Vec<String>,
    coords_angstrom: &[Vec<f64>],
    masses_amu: Option<Vec<f64>>,
) -> PyResult<MdSystem> {
    let n = top.n_atoms();
    if symbols.len() != n || coords_angstrom.len() != n {
        return Err(value_error(format!(
            "run_ffmd: the topology has {n} atoms but {} symbols and {} coordinate rows were \
             given",
            symbols.len(),
            coords_angstrom.len()
        )));
    }
    let mut coords = Array2::<f64>::zeros((n, 3));
    for (i, row) in coords_angstrom.iter().enumerate() {
        if row.len() != 3 {
            return Err(value_error(format!(
                "run_ffmd: coordinate row {i} has {} entries, expected 3",
                row.len()
            )));
        }
        for k in 0..3 {
            coords[(i, k)] = row[k] * ANGSTROM_TO_BOHR;
        }
    }
    let masses = match masses_amu {
        Some(m) => m,
        None => symbols
            .iter()
            .enumerate()
            .map(|(i, s)| {
                ferric_core::elements::symbol_to_z(s)
                    .and_then(ferric_core::elements::atomic_mass)
                    .ok_or_else(|| {
                        value_error(format!(
                            "run_ffmd: no standard atomic mass for atom {i} ({s:?}); pass \
                             masses_amu"
                        ))
                    })
            })
            .collect::<PyResult<Vec<f64>>>()?,
    };
    MdSystem::new(symbols, masses, coords).map_err(|e| value_error(format!("run_ffmd: {e}")))
}
