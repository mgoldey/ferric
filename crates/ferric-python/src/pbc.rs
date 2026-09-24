//! Python bindings for the periodic (`ferric-pbc`) drivers beyond
//! `run_rhf_gamma`: Gamma-point UHF / RKS / UKS / ROHF / ROKS, Gamma-point
//! MP2 / dRPA (SCF + correlation in one call), and k-point RHF / UHF / MP2 /
//! dRPA.
//!
//! Conventions (shared with `run_rhf_gamma`, whose helpers this module
//! reuses): `Molecule` coordinates and `lattice` rows in Ångström, every
//! `omega` in Å⁻¹, energies in Hartree per cell. Strict parsing: an unknown
//! string knob, or a knob the chosen path would ignore, is a `ValueError`.
//! A refusal from the Rust side (`FerricError::General`, which is how every
//! `ferric-pbc` config/shape/budget check reports) is a `ValueError` carrying
//! its message; numerical failures (LAPACK, libint, SCF non-convergence) are
//! `RuntimeError`s. Charged cells and ECP bases are refused for every driver:
//! no `ferric-pbc` path implements the electronic neutralising background.

use super::*;
use ferric_core::FerricError;
use ferric_pbc::dense_aft::{DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION};
use ferric_pbc::drpa::DEFAULT_GAMMA_DRPA_QUAD_POINTS;
use ferric_pbc::kcorr::DEFAULT_KDRPA_QUAD_POINTS;
use ferric_pbc::{
    gamma_drpa, gamma_mp2, gamma_rks, gamma_rohf, gamma_roks, gamma_uhf, gamma_uks, kpoint_drpa,
    kpoint_mp2, periodic_hcore, periodic_hcore_kpts, solve_krhf, solve_krhf_injected, solve_kuhf,
    Cell, DenseAftEri, EwaldStart, ExxDiv, GammaDrpaConfig, GammaDrpaIntegrals, GammaMp2Config,
    GammaMp2Integrals, GammaRksConfig, GammaRohfConfig, GammaRoksConfig, GammaUhfConfig,
    GammaUhfIntegrals, GammaUksConfig, GammaUksGridInfo, KCorrIntegrals, KDenseAftConfig,
    KDenseAftEri, KDenseAftPairs, KDrpaConfig, KDrpaEnergy, KJkKind, KMp2Config, KPointInjection,
    KPointMesh, KRhfConfig, KRsGdf, KRsGdfConfig, KScfConfig, KScfResult, KUhfConfig, LindepReport,
    MeshCentring, Mp2Denominators, PeriodicGridConfig, PeriodicHcore, PeriodicHcoreConfig, RsGdf,
    RsGdfConfig, SpinGapReport,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};

// ─────────────────────────────────────────────────────────────── helpers ──

fn val_err(m: String) -> PyErr {
    PyValueError::new_err(m)
}

/// Map a `ferric-pbc` error: refusals (`General`) are `ValueError`s with the
/// Rust message, anything numerical is a `RuntimeError`.
fn pbc_err(fname: &'static str) -> impl Fn(FerricError) -> PyErr {
    move |e| match e {
        FerricError::General(m) => PyValueError::new_err(format!("{fname}: {m}")),
        other => PyRuntimeError::new_err(format!("{fname}: {other}")),
    }
}

fn exx_name(exx: ExxDiv) -> String {
    match exx {
        ExxDiv::Ewald => "ewald".into(),
        ExxDiv::None => "none".into(),
    }
}

/// `ewald_start` is only read for `exxdiv="ewald"`: passing it with
/// `"none"` is a knob the run would ignore, hence an error.
fn parse_ewald_start(
    fname: &str,
    exx: ExxDiv,
    start: Option<&str>,
) -> PyResult<(EwaldStart, Option<String>)> {
    match (start, exx) {
        (Some(v), ExxDiv::None) => Err(val_err(format!(
            "{fname}: ewald_start={v:?} only applies to exxdiv=\"ewald\"; \
             exxdiv=\"none\" would ignore it"
        ))),
        (None, ExxDiv::None) => Ok((EwaldStart::Staged, None)),
        (None, ExxDiv::Ewald) => Ok((EwaldStart::Staged, Some("staged".into()))),
        (Some(v), ExxDiv::Ewald) => {
            let s =
                EwaldStart::parse_config_str(v).map_err(|e| val_err(format!("{fname}: {e}")))?;
            Ok((s, Some(v.to_ascii_lowercase())))
        }
    }
}

fn parse_denominators(fname: &str, s: &str) -> PyResult<Mp2Denominators> {
    Mp2Denominators::parse_config_str(s).map_err(|e| val_err(format!("{fname}: {e}")))
}

fn parse_centring(fname: &str, s: &str) -> PyResult<MeshCentring> {
    match s.to_ascii_lowercase().as_str() {
        "gamma" => Ok(MeshCentring::Gamma),
        "mp" => Ok(MeshCentring::MonkhorstPack),
        other => Err(val_err(format!(
            "{fname}: centring must be \"gamma\" or \"mp\", got {other:?}"
        ))),
    }
}

fn parse_kdrpa_energy(fname: &str, s: &str) -> PyResult<KDrpaEnergy> {
    match s.to_ascii_lowercase().as_str() {
        "quadrature" => Ok(KDrpaEnergy::Quadrature),
        "plasmon" => Ok(KDrpaEnergy::Plasmon),
        "second-order" => Ok(KDrpaEnergy::SecondOrder),
        other => Err(val_err(format!(
            "{fname}: energy must be \"quadrature\", \"plasmon\" or \"second-order\", \
             got {other:?}"
        ))),
    }
}

fn kdrpa_energy_name(e: KDrpaEnergy) -> String {
    match e {
        KDrpaEnergy::Quadrature => "quadrature".into(),
        KDrpaEnergy::Plasmon => "plasmon".into(),
        KDrpaEnergy::SecondOrder => "second-order".into(),
    }
}

/// A finite, strictly positive float knob.
fn positive(fname: &str, name: &str, v: f64) -> PyResult<f64> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(val_err(format!(
            "{fname}: {name} must be finite and > 0, got {v}"
        )))
    }
}

fn gamma_scf_config(max_iter: usize, density_conv: f64) -> RhfConfig {
    RhfConfig {
        use_sad_guess: false,
        density_conv,
        max_iter,
        ..Default::default()
    }
}

fn kscf_config(
    fname: &str,
    max_iter: usize,
    energy_conv: f64,
    grad_conv: f64,
) -> PyResult<KScfConfig> {
    Ok(KScfConfig {
        max_iter,
        energy_conv: positive(fname, "energy_conv", energy_conv)?,
        grad_conv: positive(fname, "grad_conv", grad_conv)?,
        ..Default::default()
    })
}

/// Periodic XC grid from the Python knobs (`neighbour_cutoff` in Ångström;
/// `None` = `max(10 Bohr, covering-radius bound)`). `n_radial`/`n_angular`
/// are validated by the Rust grid builder (a refusal → `ValueError`).
fn periodic_grid(
    fname: &str,
    n_radial: usize,
    n_angular: usize,
    neighbour_cutoff: Option<f64>,
) -> PyResult<PeriodicGridConfig> {
    let cutoff = match neighbour_cutoff {
        None => None,
        Some(d) => Some(positive(fname, "neighbour_cutoff", d)? * ANGSTROM_TO_BOHR),
    };
    Ok(PeriodicGridConfig {
        neighbour_cutoff: cutoff,
        ..PeriodicGridConfig::with_size(n_radial, n_angular)
    })
}

// ───────────────────────────────────────────────────────────────── setup ──

/// The raw periodic kwargs every binding shares.
struct PbcArgs<'a, 'py> {
    fname: &'static str,
    mol: &'a PyMolecule,
    lattice: &'a [Vec<f64>],
    basis: &'a PyBasisSet,
    exxdiv: &'a str,
    omega: Option<f64>,
    jk: &'a str,
    auxbasis: Option<&'a Bound<'py, PyAny>>,
    max_eri_gb: Option<f64>,
    memory_budget_gb: Option<f64>,
    /// RHF/RKS/MP2/dRPA: multiplicity 1 and an even electron count.
    closed_shell: bool,
}

/// A validated cell, orbital basis and J/K choice.
struct PbcSetup {
    fname: &'static str,
    cell: Cell,
    prep: PreparedBasis,
    exx: ExxDiv,
    /// Nuclear-attraction Ewald split (Bohr⁻¹), resolved.
    omega_bohr: f64,
    /// RS-GDF aux basis on the cell's atoms; `None` = dense AFT.
    aux: Option<PreparedBasis>,
    aux_name: Option<String>,
    /// Dense-AFT hard cap (bytes).
    max_eri_bytes: usize,
    /// RS-GDF / correlation budget (`None` = ferric's unified budget).
    budget_bytes: Option<usize>,
}

impl PbcSetup {
    fn jk_name(&self) -> String {
        if self.aux.is_some() { "rsgdf" } else { "dense" }.into()
    }
}

fn pbc_setup(a: &PbcArgs<'_, '_>) -> PyResult<PbcSetup> {
    let GammaOptions {
        exx,
        lattice_bohr,
        omega_bohr,
    } = parse_gamma_options(
        a.fname,
        a.exxdiv,
        a.jk,
        a.auxbasis.is_some(),
        a.max_eri_gb,
        a.memory_budget_gb,
        a.lattice,
        a.omega,
    )?;
    if a.closed_shell {
        validate_gamma_cell(a.fname, &a.mol.inner, &a.basis.inner)?;
    } else {
        validate_periodic_cell(a.fname, &a.mol.inner, &a.basis.inner)?;
    }
    let cell = Cell::new(a.mol.inner.clone(), lattice_bohr).map_err(pbc_err(a.fname))?;
    let prep = PreparedBasis::new(cell.mol(), &a.basis.inner).map_err(make_err)?;
    let (aux, aux_name) = match a.auxbasis {
        None => (None, None),
        Some(obj) => {
            let bs = gamma_auxbasis(a.fname, obj)?;
            let p = PreparedBasis::new(cell.mol(), &bs).map_err(|e| {
                val_err(format!(
                    "{}: auxbasis '{}' cannot be placed on this cell: {e}",
                    a.fname, bs.name
                ))
            })?;
            (Some(p), Some(bs.name))
        }
    };
    let omega_bohr = omega_bohr.unwrap_or_else(|| ferric_pbc::ewald::default_ewald_omega(&cell));
    Ok(PbcSetup {
        fname: a.fname,
        cell,
        prep,
        exx,
        omega_bohr,
        aux,
        aux_name,
        max_eri_bytes: a
            .max_eri_gb
            .map(ferric_core::memory::gib_to_bytes)
            .unwrap_or(DEFAULT_DENSE_AFT_MAX_BYTES),
        budget_bytes: budget_bytes_from_gb(a.memory_budget_gb),
    })
}

// ──────────────────────────────────────────────────── Gamma integrals ──

/// The Gamma J/K (and ov-integral) source. Built with the run's `exxdiv`;
/// the open-shell drivers and the correlation methods ignore the object's
/// own Madelung setting (they take it from their configs), so one object
/// serves the SCF and the correlation step.
enum GammaInts {
    Dense(Box<DenseAftEri>),
    RsGdf(Box<RsGdf>),
}

impl GammaInts {
    fn scf(&self) -> GammaUhfIntegrals<'_> {
        match self {
            GammaInts::Dense(e) => GammaUhfIntegrals::DenseAft(e),
            GammaInts::RsGdf(g) => GammaUhfIntegrals::RsGdf(g),
        }
    }
    fn mp2(&self) -> GammaMp2Integrals<'_> {
        match self {
            GammaInts::Dense(e) => GammaMp2Integrals::DenseAft(e),
            GammaInts::RsGdf(g) => GammaMp2Integrals::RsGdf(g),
        }
    }
    fn drpa(&self) -> GammaDrpaIntegrals<'_> {
        match self {
            GammaInts::Dense(e) => GammaDrpaIntegrals::DenseAft(e),
            GammaInts::RsGdf(g) => GammaDrpaIntegrals::RsGdf(g),
        }
    }
}

struct GammaSystem {
    hc: PeriodicHcore,
    ints: GammaInts,
}

/// Refuse an oversize dense cell before any lattice sum (the Rust builder
/// re-checks the same bound after hcore).
fn gamma_dense_preflight(s: &PbcSetup) -> PyResult<()> {
    if s.aux.is_some() {
        return Ok(());
    }
    let nao = s.prep.nbasis();
    let need = 8u128 * (nao as u128).pow(4);
    if need > s.max_eri_bytes as u128 {
        return Err(val_err(format!(
            "{}: the dense AFT ERI tensor needs {need} bytes (nao = {nao}) > max_eri_gb \
             cap {} bytes. This path is a toy-scale oracle; use jk=\"rsgdf\" with an \
             auxbasis for larger cells.",
            s.fname, s.max_eri_bytes
        )));
    }
    Ok(())
}

fn gamma_system(s: &PbcSetup) -> Result<GammaSystem, FerricError> {
    let hc = periodic_hcore(
        &s.cell,
        &s.prep,
        &PeriodicHcoreConfig::with_omega(s.omega_bohr),
    )?;
    let ints = match &s.aux {
        None => GammaInts::Dense(Box::new(DenseAftEri::build(
            &s.cell,
            &s.prep,
            &hc.s,
            s.exx,
            DEFAULT_DENSE_AFT_PRECISION,
            s.max_eri_bytes,
        )?)),
        Some(aux) => {
            let cfg = RsGdfConfig {
                exxdiv: s.exx,
                budget_bytes: s.budget_bytes,
                ..Default::default()
            };
            GammaInts::RsGdf(Box::new(RsGdf::build(&s.cell, &s.prep, aux, &hc.s, &cfg)?))
        }
    };
    Ok(GammaSystem { hc, ints })
}

/// Closed-shell Gamma RHF on injected J/K (the `run_rhf_gamma` assembly).
fn inject_rhf<'a>(
    s: &PbcSetup,
    hc: &PeriodicHcore,
    cfg: &RhfConfig,
    j: Box<dyn ferric_scf::fock::JBuilder + 'a>,
    k: Box<dyn ferric_scf::fock::KBuilder + 'a>,
) -> Result<ScfResult, FerricError> {
    let inj = ferric_scf::rhf::PeriodicInjection {
        s: hc.s.clone(),
        h: hc.h.clone(),
        vnn: hc.enn,
        j,
        k,
        xc: None,
    };
    let op = Operator::coulomb();
    // Never read on the injected path; required by the signature.
    let bounds = SchwarzBounds::compute(op, &s.prep)?;
    let ctx = ParallelContext::default();
    ferric_scf::rhf::solve_rhf_injected(&ctx, s.cell.mol(), &s.prep, op, &bounds, cfg, inj)
}

/// Closed-shell Gamma RHF reference on `sys`, required to converge.
fn gamma_rhf_reference(
    s: &PbcSetup,
    sys: &GammaSystem,
    cfg: &RhfConfig,
) -> Result<ScfResult, FerricError> {
    let r = match &sys.ints {
        GammaInts::Dense(e) => inject_rhf(
            s,
            &sys.hc,
            cfg,
            Box::new(e.j_builder()),
            Box::new(e.k_builder()),
        )?,
        GammaInts::RsGdf(g) => inject_rhf(
            s,
            &sys.hc,
            cfg,
            Box::new(g.j_builder()),
            Box::new(g.k_builder()),
        )?,
    };
    if !r.converged {
        return Err(FerricError::Convergence(format!(
            "the Gamma RHF reference did not converge in {} iterations (last E = {})",
            r.iterations, r.energy
        )));
    }
    Ok(r)
}

// ─────────────────────────────────────────── Gamma open-shell results ──

/// Result of `run_uhf_gamma` / `run_rohf_gamma` / `run_uks_gamma` /
/// `run_roks_gamma`. Energies per cell, Hartree.
#[pyclass]
#[pyo3(name = "GammaOpenShellResult")]
struct PyGammaOpenShellResult {
    /// `"uhf"`, `"rohf"`, `"uks"` or `"roks"`.
    #[pyo3(get)]
    method: String,
    /// XC functional (KS methods); `None` for UHF/ROHF.
    #[pyo3(get)]
    functional: Option<String>,
    /// Total energy per cell in the requested exxdiv convention.
    #[pyo3(get)]
    energy: f64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    /// Ewald nuclear repulsion per cell.
    #[pyo3(get)]
    e_nuc: f64,
    /// Gamma-point Madelung constant `v_M` (applied iff exxdiv="ewald").
    #[pyo3(get)]
    madelung: f64,
    #[pyo3(get)]
    exxdiv: String,
    /// `"staged"` / `"direct"` for exxdiv="ewald"; `None` for "none".
    #[pyo3(get)]
    ewald_start: Option<String>,
    /// Energy of the exxdiv="none" first stage of a staged ewald run.
    #[pyo3(get)]
    none_stage_energy: Option<f64>,
    #[pyo3(get)]
    nalpha: usize,
    #[pyo3(get)]
    nbeta: usize,
    /// `<S^2>` with the lattice overlap.
    #[pyo3(get)]
    s2: f64,
    /// Per-spin HOMO-LUMO gaps from the actual occupations (Hartree).
    #[pyo3(get)]
    gap_alpha: Option<f64>,
    #[pyo3(get)]
    gap_beta: Option<f64>,
    /// Whether every per-spin gap is at least the applied Madelung shift
    /// (the Ewald-trap diagnostic).
    #[pyo3(get)]
    gaps_satisfied: bool,
    /// XC energy (KS methods only).
    #[pyo3(get)]
    e_xc: Option<f64>,
    /// Exact-exchange fraction (KS methods only).
    #[pyo3(get)]
    exact_exchange_fraction: Option<f64>,
    /// Periodic XC grid points (KS methods only).
    #[pyo3(get)]
    n_grid_points: Option<usize>,
    /// Electrons integrated on the grid (KS methods only).
    #[pyo3(get)]
    electrons_on_grid: Option<f64>,
    #[pyo3(get)]
    nao: usize,
    #[pyo3(get)]
    jk: String,
    #[pyo3(get)]
    auxbasis: Option<String>,
    scf_data: ScfResult,
}

#[pymethods]
impl PyGammaOpenShellResult {
    /// Alpha MO energies (Hartree), ascending.
    fn mo_energy_alpha<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.scf_data.eps_alpha.clone())
    }
    /// Beta MO energies, or `None` when the SCF has one MO set (ROHF/ROKS).
    fn mo_energy_beta<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.scf_data
            .eps_beta
            .as_ref()
            .map(|e| PyArray1::from_vec(py, e.clone()))
    }
    /// Total AO density (nao x nao); `tr(D S) = N_e`.
    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.scf_data.density_total)
    }
    fn __repr__(&self) -> String {
        format!(
            "GammaOpenShellResult(method={:?}, energy={:.10}, s2={:.6}, exxdiv={:?}, \
             converged={})",
            self.method, self.energy, self.s2, self.exxdiv, self.converged
        )
    }
}

/// The common part of every Gamma open-shell driver's result.
struct OpenParts {
    scf: ScfResult,
    none_stage: Option<ScfResult>,
    madelung: f64,
    nocc: (usize, usize),
    s2: f64,
    gaps: SpinGapReport,
    e_xc: Option<f64>,
    a_x: Option<f64>,
    grid: Option<GammaUksGridInfo>,
}

enum OpenMethod {
    Uhf,
    Rohf,
    Uks(String),
    Roks(String),
}

impl OpenMethod {
    fn name(&self) -> &'static str {
        match self {
            OpenMethod::Uhf => "uhf",
            OpenMethod::Rohf => "rohf",
            OpenMethod::Uks(_) => "uks",
            OpenMethod::Roks(_) => "roks",
        }
    }
    fn functional(&self) -> Option<String> {
        match self {
            OpenMethod::Uks(f) | OpenMethod::Roks(f) => Some(f.clone()),
            _ => None,
        }
    }
}

struct OpenOpts {
    method: OpenMethod,
    start: EwaldStart,
    scf: RhfConfig,
    grid: PeriodicGridConfig,
}

fn run_open_ks(s: &PbcSetup, sys: &GammaSystem, o: &OpenOpts) -> Result<OpenParts, FerricError> {
    let ints = sys.ints.scf();
    match &o.method {
        OpenMethod::Uks(f) => {
            let mut cfg = GammaUksConfig::new(f);
            (cfg.grid, cfg.exxdiv, cfg.ewald_start) = (o.grid.clone(), s.exx, o.start);
            cfg.scf = o.scf.clone();
            let r = gamma_uks(&s.cell, &s.prep, &sys.hc, ints, &cfg)?;
            Ok(OpenParts {
                e_xc: Some(r.e_xc),
                a_x: Some(r.exact_exchange_fraction),
                grid: r.grid,
                scf: r.scf,
                none_stage: r.none_stage,
                madelung: r.madelung,
                nocc: r.nocc,
                s2: r.s2,
                gaps: r.gaps,
            })
        }
        _ => {
            let f = o.method.functional().unwrap_or_default();
            let mut cfg = GammaRoksConfig::new(&f);
            (cfg.grid, cfg.exxdiv, cfg.ewald_start) = (o.grid.clone(), s.exx, o.start);
            cfg.scf = o.scf.clone();
            let r = gamma_roks(&s.cell, &s.prep, &sys.hc, ints, &cfg)?;
            Ok(OpenParts {
                e_xc: Some(r.e_xc),
                a_x: Some(r.exact_exchange_fraction),
                grid: r.grid,
                scf: r.scf,
                none_stage: r.none_stage,
                madelung: r.madelung,
                nocc: r.nocc,
                s2: r.s2,
                gaps: r.gaps,
            })
        }
    }
}

fn run_open_hf(s: &PbcSetup, sys: &GammaSystem, o: &OpenOpts) -> Result<OpenParts, FerricError> {
    let ints = sys.ints.scf();
    let (scf, none_stage, madelung, nocc, s2, gaps) = match o.method {
        OpenMethod::Uhf => {
            let cfg = GammaUhfConfig {
                exxdiv: s.exx,
                ewald_start: o.start,
                scf: o.scf.clone(),
                initial_mos: None,
            };
            let r = gamma_uhf(&s.cell, &s.prep, &sys.hc, ints, &cfg)?;
            (r.scf, r.none_stage, r.madelung, r.nocc, r.s2, r.gaps)
        }
        _ => {
            let cfg = GammaRohfConfig {
                exxdiv: s.exx,
                ewald_start: o.start,
                scf: o.scf.clone(),
                initial_mos: None,
            };
            let r = gamma_rohf(&s.cell, &s.prep, &sys.hc, ints, &cfg)?;
            (r.scf, r.none_stage, r.madelung, r.nocc, r.s2, r.gaps)
        }
    };
    Ok(OpenParts {
        scf,
        none_stage,
        madelung,
        nocc,
        s2,
        gaps,
        e_xc: None,
        a_x: None,
        grid: None,
    })
}

fn open_shell_driver(s: &PbcSetup, o: &OpenOpts) -> Result<(OpenParts, f64), FerricError> {
    let sys = gamma_system(s)?;
    let parts = match o.method {
        OpenMethod::Uhf | OpenMethod::Rohf => run_open_hf(s, &sys, o)?,
        OpenMethod::Uks(_) | OpenMethod::Roks(_) => run_open_ks(s, &sys, o)?,
    };
    Ok((parts, sys.hc.enn))
}

/// Shared body of the four Gamma open-shell bindings.
fn run_open_shell(
    py: Python<'_>,
    s: PbcSetup,
    o: OpenOpts,
    ewald_start: Option<String>,
) -> PyResult<PyGammaOpenShellResult> {
    gamma_dense_preflight(&s)?;
    let (p, e_nuc) = py
        .allow_threads(|| open_shell_driver(&s, &o))
        .map_err(pbc_err(s.fname))?;
    Ok(PyGammaOpenShellResult {
        method: o.method.name().into(),
        functional: o.method.functional(),
        energy: p.scf.energy,
        converged: p.scf.converged,
        iterations: p.scf.iterations,
        e_nuc,
        madelung: p.madelung,
        exxdiv: exx_name(s.exx),
        ewald_start,
        none_stage_energy: p.none_stage.as_ref().map(|r| r.energy),
        nalpha: p.nocc.0,
        nbeta: p.nocc.1,
        s2: p.s2,
        gap_alpha: p.gaps.gap_alpha,
        gap_beta: p.gaps.gap_beta,
        gaps_satisfied: p.gaps.satisfied(),
        e_xc: p.e_xc,
        exact_exchange_fraction: p.a_x,
        n_grid_points: p.grid.as_ref().map(|g| g.n_grid_points),
        electrons_on_grid: p.grid.as_ref().map(|g| g.electrons_on_grid),
        nao: s.prep.nbasis(),
        jk: s.jk_name(),
        auxbasis: s.aux_name.clone(),
        scf_data: p.scf,
    })
}

/// Gamma-point periodic **UHF** (spin state from the Molecule's
/// multiplicity). Same cell/J-K/units/strictness contract as
/// `run_rhf_gamma` (lattice rows in Å, omega in Å⁻¹, jk "dense"|"rsgdf",
/// auxbasis/memory_budget_gb only with rsgdf, max_eri_gb only with dense).
///
/// `ewald_start` ("staged" | "direct"; exxdiv="ewald" only — passing it with
/// "none" is a ValueError). Default "staged": converge with exxdiv="none",
/// then continue with ewald from those MOs (avoids the Gamma Ewald trap,
/// where a direct ewald SCF from the core guess lands in a hole state).
/// The result reports the per-spin gaps against the applied Madelung shift
/// (`gaps_satisfied`).
///
/// Hard errors (ValueError): charged cell, ECP basis, incompatible electron
/// count / multiplicity, and every refusal of the Rust driver.
///
/// Validated (tests/test_pbc_bindings.py vs crates/ferric-pbc/tests/pbc_uhf.rs):
/// H atom / STO-3G (PySCF digits), a = 4 Bohr cube, PySCF 2.13 pbc.scf.UHF
/// AFTDF: E = -0.402177788224 (none) / -0.756839973159 (ewald), <S^2> 0.75.
#[pyfunction]
#[pyo3(signature = (
    mol, lattice, basis_set, exxdiv="ewald", ewald_start=None, omega=None,
    max_eri_gb=None, max_iter=200, density_conv=1e-10, jk="dense",
    auxbasis=None, memory_budget_gb=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_uhf_gamma(
    py: Python<'_>,
    mol: &PyMolecule,
    lattice: Vec<Vec<f64>>,
    basis_set: &PyBasisSet,
    exxdiv: &str,
    ewald_start: Option<&str>,
    omega: Option<f64>,
    max_eri_gb: Option<f64>,
    max_iter: usize,
    density_conv: f64,
    jk: &str,
    auxbasis: Option<&Bound<'_, PyAny>>,
    memory_budget_gb: Option<f64>,
) -> PyResult<PyGammaOpenShellResult> {
    let fname = "run_uhf_gamma";
    let s = pbc_setup(&PbcArgs {
        fname,
        mol,
        lattice: &lattice,
        basis: basis_set,
        exxdiv,
        omega,
        jk,
        auxbasis,
        max_eri_gb,
        memory_budget_gb,
        closed_shell: false,
    })?;
    let (start, start_name) = parse_ewald_start(fname, s.exx, ewald_start)?;
    let o = OpenOpts {
        method: OpenMethod::Uhf,
        start,
        scf: gamma_scf_config(max_iter, density_conv),
        grid: PeriodicGridConfig::default(),
    };
    run_open_shell(py, s, o, start_name)
}

/// Gamma-point periodic **ROHF** (Guest-Saunders Roothaan coupling on one
/// MO set). Same contract as `run_uhf_gamma` (ewald_start, jk, units,
/// strictness).
///
/// KNOWN LIMITATION: ferric's DIIS ROHF does not converge on the periodic
/// triclinic 4H s+p triplet (the Rust suite's `pbc_rohf.rs` case is
/// `#[ignore]`d; PySCF converges there to -0.581222768976). A non-converged
/// SCF stage is an error, never a silently returned energy. ROKS through the
/// same injected path converges and matches its pins.
///
/// Validated: for one electron (H atom, a = 4 Bohr, STO-3G PySCF digits)
/// ROHF == UHF == the PySCF pins -0.402177788224 / -0.756839973159.
#[pyfunction]
#[pyo3(signature = (
    mol, lattice, basis_set, exxdiv="ewald", ewald_start=None, omega=None,
    max_eri_gb=None, max_iter=200, density_conv=1e-10, jk="dense",
    auxbasis=None, memory_budget_gb=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_rohf_gamma(
    py: Python<'_>,
    mol: &PyMolecule,
    lattice: Vec<Vec<f64>>,
    basis_set: &PyBasisSet,
    exxdiv: &str,
    ewald_start: Option<&str>,
    omega: Option<f64>,
    max_eri_gb: Option<f64>,
    max_iter: usize,
    density_conv: f64,
    jk: &str,
    auxbasis: Option<&Bound<'_, PyAny>>,
    memory_budget_gb: Option<f64>,
) -> PyResult<PyGammaOpenShellResult> {
    let fname = "run_rohf_gamma";
    let s = pbc_setup(&PbcArgs {
        fname,
        mol,
        lattice: &lattice,
        basis: basis_set,
        exxdiv,
        omega,
        jk,
        auxbasis,
        max_eri_gb,
        memory_budget_gb,
        closed_shell: false,
    })?;
    let (start, start_name) = parse_ewald_start(fname, s.exx, ewald_start)?;
    let o = OpenOpts {
        method: OpenMethod::Rohf,
        start,
        scf: gamma_scf_config(max_iter, density_conv),
        grid: PeriodicGridConfig::default(),
    };
    run_open_shell(py, s, o, start_name)
}

/// Gamma-point periodic **UKS**. `functional`: LDA/GGA/global hybrids
/// (e.g. "LDA", "PBE", "PBE0"); range-separated, meta-GGA, VV10 and double
/// hybrids are refused (ValueError). Grid: Treutler-Ahlrichs `n_radial` x
/// Lebedev `n_angular` per atom with SSF partitioning over periodic images
/// within `neighbour_cutoff` (Å; None = max(10 Bohr, covering-radius
/// bound)). Otherwise the `run_uhf_gamma` contract (ewald_start, jk, units).
///
/// Validated vs crates/ferric-pbc/tests/pbc_uks.rs: H atom, a = 4 Bohr,
/// STO-3G PySCF digits, SSF 75x302 D = 10 Bohr, ewald: LDA -0.667583328116.
#[pyfunction]
#[pyo3(signature = (
    mol, lattice, basis_set, functional, exxdiv="ewald", ewald_start=None,
    omega=None, max_eri_gb=None, max_iter=200, density_conv=1e-10, jk="dense",
    auxbasis=None, memory_budget_gb=None, n_radial=75, n_angular=302,
    neighbour_cutoff=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_uks_gamma(
    py: Python<'_>,
    mol: &PyMolecule,
    lattice: Vec<Vec<f64>>,
    basis_set: &PyBasisSet,
    functional: &str,
    exxdiv: &str,
    ewald_start: Option<&str>,
    omega: Option<f64>,
    max_eri_gb: Option<f64>,
    max_iter: usize,
    density_conv: f64,
    jk: &str,
    auxbasis: Option<&Bound<'_, PyAny>>,
    memory_budget_gb: Option<f64>,
    n_radial: usize,
    n_angular: usize,
    neighbour_cutoff: Option<f64>,
) -> PyResult<PyGammaOpenShellResult> {
    let fname = "run_uks_gamma";
    let s = pbc_setup(&PbcArgs {
        fname,
        mol,
        lattice: &lattice,
        basis: basis_set,
        exxdiv,
        omega,
        jk,
        auxbasis,
        max_eri_gb,
        memory_budget_gb,
        closed_shell: false,
    })?;
    let (start, start_name) = parse_ewald_start(fname, s.exx, ewald_start)?;
    let o = OpenOpts {
        method: OpenMethod::Uks(functional.to_string()),
        start,
        scf: gamma_scf_config(max_iter, density_conv),
        grid: periodic_grid(fname, n_radial, n_angular, neighbour_cutoff)?,
    };
    run_open_shell(py, s, o, start_name)
}

/// Gamma-point periodic **ROKS** (spin-polarized XC, Roothaan coupling on
/// one MO set). Same functional/grid contract as `run_uks_gamma`.
///
/// KNOWN LIMITATION (shared with `run_rohf_gamma`): the zero-XC ROHF limit
/// does not converge on the triclinic 4H s+p triplet; ROKS with a real
/// functional converges there (pinned in crates/ferric-pbc/tests/pbc_rohf.rs).
///
/// Validated: for one electron ROKS == UKS (H atom, a = 4 Bohr, SSF 75x302,
/// LDA -0.667583328116).
#[pyfunction]
#[pyo3(signature = (
    mol, lattice, basis_set, functional, exxdiv="ewald", ewald_start=None,
    omega=None, max_eri_gb=None, max_iter=200, density_conv=1e-10, jk="dense",
    auxbasis=None, memory_budget_gb=None, n_radial=75, n_angular=302,
    neighbour_cutoff=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_roks_gamma(
    py: Python<'_>,
    mol: &PyMolecule,
    lattice: Vec<Vec<f64>>,
    basis_set: &PyBasisSet,
    functional: &str,
    exxdiv: &str,
    ewald_start: Option<&str>,
    omega: Option<f64>,
    max_eri_gb: Option<f64>,
    max_iter: usize,
    density_conv: f64,
    jk: &str,
    auxbasis: Option<&Bound<'_, PyAny>>,
    memory_budget_gb: Option<f64>,
    n_radial: usize,
    n_angular: usize,
    neighbour_cutoff: Option<f64>,
) -> PyResult<PyGammaOpenShellResult> {
    let fname = "run_roks_gamma";
    let s = pbc_setup(&PbcArgs {
        fname,
        mol,
        lattice: &lattice,
        basis: basis_set,
        exxdiv,
        omega,
        jk,
        auxbasis,
        max_eri_gb,
        memory_budget_gb,
        closed_shell: false,
    })?;
    let (start, start_name) = parse_ewald_start(fname, s.exx, ewald_start)?;
    let o = OpenOpts {
        method: OpenMethod::Roks(functional.to_string()),
        start,
        scf: gamma_scf_config(max_iter, density_conv),
        grid: periodic_grid(fname, n_radial, n_angular, neighbour_cutoff)?,
    };
    run_open_shell(py, s, o, start_name)
}

// ─────────────────────────────────────────────────────────── Gamma RKS ──

/// Result of `run_rks_gamma`. Energies per cell, Hartree.
#[pyclass]
#[pyo3(name = "GammaRksResult")]
struct PyGammaRksResult {
    #[pyo3(get)]
    functional: String,
    #[pyo3(get)]
    energy: f64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    e_nuc: f64,
    #[pyo3(get)]
    madelung: f64,
    #[pyo3(get)]
    exxdiv: String,
    #[pyo3(get)]
    e_xc: f64,
    #[pyo3(get)]
    exact_exchange_fraction: f64,
    #[pyo3(get)]
    n_grid_points: usize,
    /// Neighbour cutoff actually used (Å).
    #[pyo3(get)]
    neighbour_cutoff: f64,
    #[pyo3(get)]
    electrons_on_grid: f64,
    #[pyo3(get)]
    nao: usize,
    #[pyo3(get)]
    jk: String,
    #[pyo3(get)]
    auxbasis: Option<String>,
    scf_data: ScfResult,
}

#[pymethods]
impl PyGammaRksResult {
    /// MO energies (Hartree), ascending.
    fn mo_energy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.scf_data.eps_alpha.clone())
    }
    /// Total AO density (nao x nao).
    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.scf_data.density_total)
    }
    fn __repr__(&self) -> String {
        format!(
            "GammaRksResult(functional={:?}, energy={:.10}, exxdiv={:?}, converged={})",
            self.functional, self.energy, self.exxdiv, self.converged
        )
    }
}

fn rks_driver(
    s: &PbcSetup,
    functional: &str,
    grid: PeriodicGridConfig,
    scf: RhfConfig,
) -> Result<(ferric_pbc::GammaRksResult, f64), FerricError> {
    let sys = gamma_system(s)?;
    let mut cfg = GammaRksConfig::new(functional);
    (cfg.grid, cfg.exxdiv, cfg.scf) = (grid, s.exx, scf);
    let r = gamma_rks(&s.cell, &s.prep, &sys.hc, sys.ints.scf(), &cfg)?;
    Ok((r, sys.hc.enn))
}

/// Closed-shell Gamma-point periodic **RKS**. Functional and grid contract
/// as `run_uks_gamma` (no ewald_start: RKS runs one SCF in the requested
/// exxdiv); cell/J-K/units/strictness as `run_rhf_gamma`. An open shell or
/// odd electron count is a ValueError (use `run_uks_gamma`).
///
/// Validated vs crates/ferric-pbc/tests/pbc_rks.rs: H2 / STO-3G (PySCF
/// digits), a = 4 Bohr, SSF 75x302 D = 10 Bohr, ewald: LDA -1.521871150768,
/// 36234 grid points.
#[pyfunction]
#[pyo3(signature = (
    mol, lattice, basis_set, functional, exxdiv="ewald", omega=None,
    max_eri_gb=None, max_iter=200, density_conv=1e-10, jk="dense",
    auxbasis=None, memory_budget_gb=None, n_radial=75, n_angular=302,
    neighbour_cutoff=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_rks_gamma(
    py: Python<'_>,
    mol: &PyMolecule,
    lattice: Vec<Vec<f64>>,
    basis_set: &PyBasisSet,
    functional: &str,
    exxdiv: &str,
    omega: Option<f64>,
    max_eri_gb: Option<f64>,
    max_iter: usize,
    density_conv: f64,
    jk: &str,
    auxbasis: Option<&Bound<'_, PyAny>>,
    memory_budget_gb: Option<f64>,
    n_radial: usize,
    n_angular: usize,
    neighbour_cutoff: Option<f64>,
) -> PyResult<PyGammaRksResult> {
    let fname = "run_rks_gamma";
    let s = pbc_setup(&PbcArgs {
        fname,
        mol,
        lattice: &lattice,
        basis: basis_set,
        exxdiv,
        omega,
        jk,
        auxbasis,
        max_eri_gb,
        memory_budget_gb,
        closed_shell: true,
    })?;
    let grid = periodic_grid(fname, n_radial, n_angular, neighbour_cutoff)?;
    gamma_dense_preflight(&s)?;
    let scf = gamma_scf_config(max_iter, density_conv);
    let (r, e_nuc) = py
        .allow_threads(|| rks_driver(&s, functional, grid, scf))
        .map_err(pbc_err(fname))?;
    Ok(PyGammaRksResult {
        functional: functional.to_string(),
        energy: r.scf.energy,
        converged: r.scf.converged,
        iterations: r.scf.iterations,
        e_nuc,
        madelung: r.madelung,
        exxdiv: exx_name(s.exx),
        e_xc: r.e_xc,
        exact_exchange_fraction: r.exact_exchange_fraction,
        n_grid_points: r.n_grid_points,
        neighbour_cutoff: r.neighbour_cutoff / ANGSTROM_TO_BOHR,
        electrons_on_grid: r.electrons_on_grid,
        nao: s.prep.nbasis(),
        jk: s.jk_name(),
        auxbasis: s.aux_name.clone(),
        scf_data: r.scf,
    })
}

// ─────────────────────────────────────────────── Gamma MP2 / dRPA ──

/// Result of `run_mp2_gamma` / `run_drpa_gamma` (SCF + correlation).
#[pyclass]
#[pyo3(name = "GammaCorrelationResult")]
struct PyGammaCorrelationResult {
    /// `"mp2"` or `"drpa"`.
    #[pyo3(get)]
    method: String,
    /// `e_scf + correlation_energy` (the RHF energy is the reference's own:
    /// it carries `-v_M N_e / 2` under exxdiv="ewald").
    #[pyo3(get)]
    energy: f64,
    /// The RHF reference energy per cell.
    #[pyo3(get)]
    e_scf: f64,
    #[pyo3(get)]
    correlation_energy: f64,
    /// MP2 opposite-/same-spin split (`None` for dRPA).
    #[pyo3(get)]
    e_os: Option<f64>,
    #[pyo3(get)]
    e_ss: Option<f64>,
    /// Whether the RHF reference converged (always true: a non-converged
    /// reference is an error).
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    madelung: f64,
    /// Shift added to every active occupied energy (0, -v_M or +v_M).
    #[pyo3(get)]
    occ_shift: f64,
    #[pyo3(get)]
    nocc_active: usize,
    #[pyo3(get)]
    nvir: usize,
    /// Fitted aux rows (`None` for the dense oracle).
    #[pyo3(get)]
    naux: Option<usize>,
    /// dRPA frequency points (`None` for MP2 and the dense plasmon oracle).
    #[pyo3(get)]
    quad_points: Option<usize>,
    /// The reference's exxdiv ("ewald" | "none").
    #[pyo3(get)]
    exxdiv: String,
    /// "shifted" | "unshifted".
    #[pyo3(get)]
    denominators: String,
    #[pyo3(get)]
    jk: String,
    #[pyo3(get)]
    auxbasis: Option<String>,
}

#[pymethods]
impl PyGammaCorrelationResult {
    fn __repr__(&self) -> String {
        format!(
            "GammaCorrelationResult(method={:?}, correlation_energy={:.10}, energy={:.10}, \
             exxdiv={:?}, denominators={:?})",
            self.method, self.correlation_energy, self.energy, self.exxdiv, self.denominators
        )
    }
}

/// The numbers both Gamma correlation drivers return.
struct CorrParts {
    corr: f64,
    total: f64,
    e_os: Option<f64>,
    e_ss: Option<f64>,
    madelung: f64,
    occ_shift: f64,
    nocc_active: usize,
    nvir: usize,
    naux: Option<usize>,
    quad_points: Option<usize>,
}

struct CorrOpts {
    /// `None` = MP2, `Some(q)` = dRPA with `q` frequency points.
    drpa_quad: Option<usize>,
    den: Mp2Denominators,
    frozen_core: usize,
    scf: RhfConfig,
}

fn gamma_corr_step(
    s: &PbcSetup,
    sys: &GammaSystem,
    rhf: &ScfResult,
    o: &CorrOpts,
) -> Result<CorrParts, FerricError> {
    match o.drpa_quad {
        None => {
            let cfg = GammaMp2Config {
                frozen_core: o.frozen_core,
                reference_exxdiv: s.exx,
                denominators: o.den,
                budget_bytes: s.budget_bytes,
            };
            let r = gamma_mp2(&s.cell, rhf, sys.ints.mp2(), &cfg)?;
            Ok(CorrParts {
                corr: r.mp2_corr,
                total: r.total_energy,
                e_os: Some(r.components.e_os),
                e_ss: Some(r.components.e_ss),
                madelung: r.madelung,
                occ_shift: r.occ_shift,
                nocc_active: r.nocc_active,
                nvir: r.nvir,
                naux: r.naux,
                quad_points: None,
            })
        }
        Some(q) => {
            let cfg = GammaDrpaConfig {
                frozen_core: o.frozen_core,
                reference_exxdiv: s.exx,
                denominators: o.den,
                quad_points: q,
                budget_bytes: s.budget_bytes,
            };
            let r = gamma_drpa(&s.cell, rhf, sys.ints.drpa(), &cfg)?;
            Ok(CorrParts {
                corr: r.drpa_corr,
                total: r.total_energy,
                e_os: None,
                e_ss: None,
                madelung: r.madelung,
                occ_shift: r.occ_shift,
                nocc_active: r.nocc_active,
                nvir: r.nvir,
                naux: r.naux,
                quad_points: r.quad_points,
            })
        }
    }
}

fn gamma_corr_driver(s: &PbcSetup, o: &CorrOpts) -> Result<(ScfResult, CorrParts), FerricError> {
    let sys = gamma_system(s)?;
    let rhf = gamma_rhf_reference(s, &sys, &o.scf)?;
    let c = gamma_corr_step(s, &sys, &rhf, o)?;
    Ok((rhf, c))
}

fn run_gamma_corr(
    py: Python<'_>,
    s: PbcSetup,
    o: CorrOpts,
    denominators: &str,
) -> PyResult<PyGammaCorrelationResult> {
    gamma_dense_preflight(&s)?;
    let (rhf, c) = py
        .allow_threads(|| gamma_corr_driver(&s, &o))
        .map_err(pbc_err(s.fname))?;
    Ok(PyGammaCorrelationResult {
        method: if o.drpa_quad.is_some() { "drpa" } else { "mp2" }.into(),
        energy: c.total,
        e_scf: rhf.energy,
        correlation_energy: c.corr,
        e_os: c.e_os,
        e_ss: c.e_ss,
        converged: rhf.converged,
        iterations: rhf.iterations,
        madelung: c.madelung,
        occ_shift: c.occ_shift,
        nocc_active: c.nocc_active,
        nvir: c.nvir,
        naux: c.naux,
        quad_points: c.quad_points,
        exxdiv: exx_name(s.exx),
        denominators: denominators.to_ascii_lowercase(),
        jk: s.jk_name(),
        auxbasis: s.aux_name.clone(),
    })
}

/// Gamma-point closed-shell **MP2**: Gamma RHF (the `run_rhf_gamma`
/// assembly) + MP2 in one call.
///
/// `exxdiv` (the REFERENCE's exxdiv) and `denominators` are REQUIRED — there
/// is no default convention. `denominators`: "shifted" (occupied energies
/// Madelung-shifted, the physical convention; residual vs the molecule
/// O(a^-3)) or "unshifted" (exxdiv="none" occupied energies; O(1/a)).
/// Every exxdiv/denominators pair is valid; the shift is applied
/// consistently. jk="dense" uses exact dense (ia|jb); jk="rsgdf" the fitted
/// B of the SCF. `memory_budget_gb` (rsgdf only) also bounds the MP2 step.
///
/// Validated vs crates/ferric-pbc/tests/pbc_mp2.rs (PySCF 2.13 pbc.mp.RMP2,
/// AFTDF): H2 / STO-3G (PySCF digits), a = 4 Bohr: -5.891222456423e-3
/// (unshifted), -3.881428851328e-3 (shifted).
#[pyfunction]
#[pyo3(signature = (
    mol, lattice, basis_set, exxdiv, denominators, omega=None, max_eri_gb=None,
    max_iter=200, density_conv=1e-10, jk="dense", auxbasis=None,
    memory_budget_gb=None, frozen_core=0,
))]
#[allow(clippy::too_many_arguments)]
fn run_mp2_gamma(
    py: Python<'_>,
    mol: &PyMolecule,
    lattice: Vec<Vec<f64>>,
    basis_set: &PyBasisSet,
    exxdiv: &str,
    denominators: &str,
    omega: Option<f64>,
    max_eri_gb: Option<f64>,
    max_iter: usize,
    density_conv: f64,
    jk: &str,
    auxbasis: Option<&Bound<'_, PyAny>>,
    memory_budget_gb: Option<f64>,
    frozen_core: usize,
) -> PyResult<PyGammaCorrelationResult> {
    let fname = "run_mp2_gamma";
    let s = pbc_setup(&PbcArgs {
        fname,
        mol,
        lattice: &lattice,
        basis: basis_set,
        exxdiv,
        omega,
        jk,
        auxbasis,
        max_eri_gb,
        memory_budget_gb,
        closed_shell: true,
    })?;
    let o = CorrOpts {
        drpa_quad: None,
        den: parse_denominators(fname, denominators)?,
        frozen_core,
        scf: gamma_scf_config(max_iter, density_conv),
    };
    run_gamma_corr(py, s, o, denominators)
}

/// Gamma-point closed-shell **dRPA**: Gamma RHF + dRPA in one call. The
/// `exxdiv`/`denominators` contract is `run_mp2_gamma`'s.
///
/// jk="rsgdf": frequency quadrature on the fitted B, `quad_points` points
/// (None = 40). jk="dense": the exact plasmon formula on the dense (ia|jb),
/// which has no quadrature — passing `quad_points` there is a ValueError.
///
/// Validated vs crates/ferric-pbc/tests/pbc_drpa.rs (PySCF 2.13 AFTDF pins):
/// H2 / STO-3G (PySCF digits), a = 4 Bohr: -1.000052013959e-2 (unshifted),
/// -6.938130614440e-3 (shifted).
#[pyfunction]
#[pyo3(signature = (
    mol, lattice, basis_set, exxdiv, denominators, omega=None, max_eri_gb=None,
    max_iter=200, density_conv=1e-10, jk="dense", auxbasis=None,
    memory_budget_gb=None, frozen_core=0, quad_points=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_drpa_gamma(
    py: Python<'_>,
    mol: &PyMolecule,
    lattice: Vec<Vec<f64>>,
    basis_set: &PyBasisSet,
    exxdiv: &str,
    denominators: &str,
    omega: Option<f64>,
    max_eri_gb: Option<f64>,
    max_iter: usize,
    density_conv: f64,
    jk: &str,
    auxbasis: Option<&Bound<'_, PyAny>>,
    memory_budget_gb: Option<f64>,
    frozen_core: usize,
    quad_points: Option<usize>,
) -> PyResult<PyGammaCorrelationResult> {
    let fname = "run_drpa_gamma";
    let s = pbc_setup(&PbcArgs {
        fname,
        mol,
        lattice: &lattice,
        basis: basis_set,
        exxdiv,
        omega,
        jk,
        auxbasis,
        max_eri_gb,
        memory_budget_gb,
        closed_shell: true,
    })?;
    if let (Some(q), None) = (quad_points, &s.aux) {
        return Err(val_err(format!(
            "{fname}: quad_points={q} is ignored by jk=\"dense\" (the exact plasmon \
             formula has no frequency quadrature); drop it or use jk=\"rsgdf\""
        )));
    }
    let o = CorrOpts {
        drpa_quad: Some(quad_points.unwrap_or(DEFAULT_GAMMA_DRPA_QUAD_POINTS)),
        den: parse_denominators(fname, denominators)?,
        frozen_core,
        scf: gamma_scf_config(max_iter, density_conv),
    };
    run_gamma_corr(py, s, o, denominators)
}

// ──────────────────────────────────────────────────────── k-point SCF ──

/// Lindep-report totals copied onto the k-point results.
struct LindepTotals {
    threshold: f64,
    min_kept: usize,
    max_kept: usize,
    total_kept: usize,
    near_noise_floor: bool,
}

impl From<&LindepReport> for LindepTotals {
    fn from(l: &LindepReport) -> Self {
        Self {
            threshold: l.threshold,
            min_kept: l.min_kept,
            max_kept: l.max_kept,
            total_kept: l.total_kept,
            near_noise_floor: l.near_noise_floor,
        }
    }
}

/// Result of `run_rhf_kpts` / `run_uhf_kpts`. Energies per cell, Hartree.
#[pyclass]
#[pyo3(name = "KpointScfResult")]
struct PyKpointScfResult {
    /// `"rhf"` or `"uhf"`.
    #[pyo3(get)]
    method: String,
    #[pyo3(get)]
    energy: f64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    e_nuc: f64,
    /// Mesh (supercell) Madelung constant (applied iff exxdiv="ewald").
    #[pyo3(get)]
    madelung: f64,
    #[pyo3(get)]
    exxdiv: String,
    /// UHF + exxdiv="ewald" only.
    #[pyo3(get)]
    ewald_start: Option<String>,
    #[pyo3(get)]
    none_stage_energy: Option<f64>,
    #[pyo3(get)]
    mesh: (usize, usize, usize),
    #[pyo3(get)]
    centring: String,
    #[pyo3(get)]
    nk: usize,
    /// Cartesian k-points in Å⁻¹.
    #[pyo3(get)]
    kpts: Vec<[f64; 3]>,
    /// Per-k MO energies (alpha for UHF).
    #[pyo3(get)]
    mo_energy: Vec<Vec<f64>>,
    /// Per-k beta MO energies (UHF only).
    #[pyo3(get)]
    mo_energy_beta: Option<Vec<Vec<f64>>>,
    /// RHF only: global HOMO / LUMO over the mesh.
    #[pyo3(get)]
    homo: Option<f64>,
    #[pyo3(get)]
    lumo: Option<f64>,
    /// UHF only: electrons per cell by spin, <S^2> of the GIANT (supercell)
    /// determinant (not per cell), per-spin gaps over all k.
    #[pyo3(get)]
    nalpha: Option<usize>,
    #[pyo3(get)]
    nbeta: Option<usize>,
    #[pyo3(get)]
    s2: Option<f64>,
    #[pyo3(get)]
    gap_alpha: Option<f64>,
    #[pyo3(get)]
    gap_beta: Option<f64>,
    /// Canonical-orthogonaliser report on S(k): threshold, min/max/total
    /// vectors kept over the mesh, and whether the threshold is near the
    /// overlap's rounding-noise floor.
    #[pyo3(get)]
    lindep_threshold: f64,
    #[pyo3(get)]
    lindep_min_kept: usize,
    #[pyo3(get)]
    lindep_max_kept: usize,
    #[pyo3(get)]
    lindep_total_kept: usize,
    #[pyo3(get)]
    lindep_near_noise_floor: bool,
    #[pyo3(get)]
    nao: usize,
    #[pyo3(get)]
    jk: String,
    #[pyo3(get)]
    auxbasis: Option<String>,
}

#[pymethods]
impl PyKpointScfResult {
    fn __repr__(&self) -> String {
        format!(
            "KpointScfResult(method={:?}, energy={:.10}, mesh={:?}, exxdiv={:?}, \
             converged={})",
            self.method, self.energy, self.mesh, self.exxdiv, self.converged
        )
    }
}

/// Validated mesh on the setup's cell.
fn k_mesh(s: &PbcSetup, mesh: (usize, usize, usize), centring: &str) -> PyResult<KPointMesh> {
    let c = parse_centring(s.fname, centring)?;
    KPointMesh::new(&s.cell, [mesh.0, mesh.1, mesh.2], c).map_err(pbc_err(s.fname))
}

fn centring_name(c: MeshCentring) -> String {
    match c {
        MeshCentring::Gamma => "gamma".into(),
        MeshCentring::MonkhorstPack => "mp".into(),
    }
}

fn kdense_config(s: &PbcSetup) -> KDenseAftConfig {
    KDenseAftConfig {
        max_bytes: s.max_eri_bytes,
        ..Default::default()
    }
}

fn krsgdf_config(s: &PbcSetup) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            budget_bytes: s.budget_bytes,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn kjk_kind(s: &PbcSetup) -> KJkKind {
    if s.aux.is_some() {
        KJkKind::RsGdf
    } else {
        KJkKind::Dense
    }
}

/// The k-point result skeleton every k binding fills in.
fn kpoint_result_base(
    s: &PbcSetup,
    mesh: &KPointMesh,
    method: &str,
    scf: &KScfLike,
) -> PyKpointScfResult {
    let n = mesh.n();
    PyKpointScfResult {
        method: method.into(),
        energy: scf.energy,
        converged: scf.converged,
        iterations: scf.iterations,
        e_nuc: scf.e_nuc,
        madelung: scf.madelung,
        exxdiv: exx_name(s.exx),
        ewald_start: None,
        none_stage_energy: None,
        mesh: (n[0], n[1], n[2]),
        centring: centring_name(mesh.centring()),
        nk: mesh.nk(),
        kpts: scf
            .kpts
            .iter()
            .map(|k| {
                [
                    k[0] * ANGSTROM_TO_BOHR,
                    k[1] * ANGSTROM_TO_BOHR,
                    k[2] * ANGSTROM_TO_BOHR,
                ]
            })
            .collect(),
        mo_energy: Vec::new(),
        mo_energy_beta: None,
        homo: None,
        lumo: None,
        nalpha: None,
        nbeta: None,
        s2: None,
        gap_alpha: None,
        gap_beta: None,
        lindep_threshold: scf.lindep.threshold,
        lindep_min_kept: scf.lindep.min_kept,
        lindep_max_kept: scf.lindep.max_kept,
        lindep_total_kept: scf.lindep.total_kept,
        lindep_near_noise_floor: scf.lindep.near_noise_floor,
        nao: s.prep.nbasis(),
        jk: s.jk_name(),
        auxbasis: s.aux_name.clone(),
    }
}

/// The fields KRHF and KUHF results share (KScfResult / KUScfResult have
/// no common trait).
struct KScfLike {
    energy: f64,
    converged: bool,
    iterations: usize,
    e_nuc: f64,
    madelung: f64,
    kpts: Vec<[f64; 3]>,
    lindep: LindepTotals,
}

fn krhf_driver(
    s: &PbcSetup,
    mesh: &KPointMesh,
    scf: KScfConfig,
) -> Result<(KScfResult, f64), FerricError> {
    let mut cfg = KRhfConfig::for_cell(&s.cell, s.exx);
    cfg.scf = scf;
    cfg.hcore = PeriodicHcoreConfig::with_omega(s.omega_bohr);
    (cfg.jk, cfg.dense, cfg.rsgdf) = (kjk_kind(s), kdense_config(s), krsgdf_config(s));
    let r = solve_krhf(&s.cell, &s.prep, s.aux.as_ref(), mesh, &cfg)?;
    let madelung = mesh.madelung(&s.cell)?;
    Ok((r, madelung))
}

/// Closed-shell **k-point RHF** on a `mesh = (n1, n2, n3)` k-mesh.
///
/// `centring`: "gamma" (contains Gamma; PySCF `make_kpts(n)`) or "mp"
/// (Monkhorst-Pack, PySCF `with_gamma_point=False`). Strict. jk="dense":
/// the toy-scale dense k-point AFT kernels (`2 N_k^2 nao^4 x 16` bytes,
/// capped by `max_eri_gb`, default 0.5 GiB). jk="rsgdf": per-q RS-GDF,
/// REQUIRES `auxbasis`, bounded by `memory_budget_gb`. The same
/// cell/units/strictness contract as `run_rhf_gamma`; SCF convergence by
/// `energy_conv` (|dE| per cell) AND `grad_conv` (max orbital gradient).
///
/// Validated vs crates/ferric-pbc/tests/pbc_krhf.rs (PySCF 2.13 KRHF,
/// AFTDF): H2 / STO-3G (PySCF digits), a = 4 Bohr, Gamma-centred 1x1x2:
/// -0.902683427348 (none) / -1.354143879961 (ewald).
#[pyfunction]
#[pyo3(signature = (
    mol, lattice, basis_set, mesh, exxdiv="ewald", centring="gamma", omega=None,
    max_eri_gb=None, max_iter=200, energy_conv=1e-12, grad_conv=1e-9,
    jk="dense", auxbasis=None, memory_budget_gb=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_rhf_kpts(
    py: Python<'_>,
    mol: &PyMolecule,
    lattice: Vec<Vec<f64>>,
    basis_set: &PyBasisSet,
    mesh: (usize, usize, usize),
    exxdiv: &str,
    centring: &str,
    omega: Option<f64>,
    max_eri_gb: Option<f64>,
    max_iter: usize,
    energy_conv: f64,
    grad_conv: f64,
    jk: &str,
    auxbasis: Option<&Bound<'_, PyAny>>,
    memory_budget_gb: Option<f64>,
) -> PyResult<PyKpointScfResult> {
    let fname = "run_rhf_kpts";
    let s = pbc_setup(&PbcArgs {
        fname,
        mol,
        lattice: &lattice,
        basis: basis_set,
        exxdiv,
        omega,
        jk,
        auxbasis,
        max_eri_gb,
        memory_budget_gb,
        closed_shell: true,
    })?;
    let m = k_mesh(&s, mesh, centring)?;
    let scf = kscf_config(fname, max_iter, energy_conv, grad_conv)?;
    let (r, madelung) = py
        .allow_threads(|| krhf_driver(&s, &m, scf))
        .map_err(pbc_err(fname))?;
    let like = KScfLike {
        energy: r.energy,
        converged: r.converged,
        iterations: r.iterations,
        e_nuc: r.e_nuc,
        madelung,
        kpts: r.kpts.clone(),
        lindep: (&r.lindep).into(),
    };
    let mut out = kpoint_result_base(&s, &m, "rhf", &like);
    out.mo_energy = r.eps;
    (out.homo, out.lumo) = (Some(r.homo), Some(r.lumo));
    Ok(out)
}

fn kuhf_driver(
    s: &PbcSetup,
    mesh: &KPointMesh,
    scf: KScfConfig,
    start: EwaldStart,
) -> Result<ferric_pbc::KUhfResult, FerricError> {
    let mut cfg = KUhfConfig::for_cell(&s.cell, s.exx);
    (cfg.scf, cfg.ewald_start) = (scf, start);
    cfg.hcore = PeriodicHcoreConfig::with_omega(s.omega_bohr);
    (cfg.jk, cfg.dense, cfg.rsgdf) = (kjk_kind(s), kdense_config(s), krsgdf_config(s));
    cfg.budget_bytes = s.budget_bytes;
    solve_kuhf(&s.cell, &s.prep, s.aux.as_ref(), mesh, &cfg)
}

/// **k-point UHF** (spin state from the Molecule's multiplicity; electrons
/// per cell). Mesh/centring/jk contract as `run_rhf_kpts`; `ewald_start` as
/// `run_uhf_gamma` (exxdiv="ewald" only; default "staged"). `s2` is
/// `<S^2>` of the giant (supercell) determinant, NOT per cell.
///
/// Validated vs crates/ferric-pbc/tests/pbc_kuhf.rs (PySCF 2.13 KUHF,
/// AFTDF): H atom / STO-3G (PySCF digits), a = 4 Bohr, Gamma-centred 1x1x2:
/// -0.399399818915 (none) / -0.625130045222 (ewald), giant <S^2> = 2.
#[pyfunction]
#[pyo3(signature = (
    mol, lattice, basis_set, mesh, exxdiv="ewald", centring="gamma",
    ewald_start=None, omega=None, max_eri_gb=None, max_iter=200,
    energy_conv=1e-12, grad_conv=1e-9, jk="dense", auxbasis=None,
    memory_budget_gb=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_uhf_kpts(
    py: Python<'_>,
    mol: &PyMolecule,
    lattice: Vec<Vec<f64>>,
    basis_set: &PyBasisSet,
    mesh: (usize, usize, usize),
    exxdiv: &str,
    centring: &str,
    ewald_start: Option<&str>,
    omega: Option<f64>,
    max_eri_gb: Option<f64>,
    max_iter: usize,
    energy_conv: f64,
    grad_conv: f64,
    jk: &str,
    auxbasis: Option<&Bound<'_, PyAny>>,
    memory_budget_gb: Option<f64>,
) -> PyResult<PyKpointScfResult> {
    let fname = "run_uhf_kpts";
    let s = pbc_setup(&PbcArgs {
        fname,
        mol,
        lattice: &lattice,
        basis: basis_set,
        exxdiv,
        omega,
        jk,
        auxbasis,
        max_eri_gb,
        memory_budget_gb,
        closed_shell: false,
    })?;
    let m = k_mesh(&s, mesh, centring)?;
    let (start, start_name) = parse_ewald_start(fname, s.exx, ewald_start)?;
    let scf = kscf_config(fname, max_iter, energy_conv, grad_conv)?;
    let r = py
        .allow_threads(|| kuhf_driver(&s, &m, scf, start))
        .map_err(pbc_err(fname))?;
    let u = &r.scf;
    let like = KScfLike {
        energy: u.energy,
        converged: u.converged,
        iterations: u.iterations,
        e_nuc: u.e_nuc,
        madelung: r.madelung,
        kpts: u.kpts.clone(),
        lindep: (&u.lindep).into(),
    };
    let mut out = kpoint_result_base(&s, &m, "uhf", &like);
    out.ewald_start = start_name;
    out.none_stage_energy = r.none_stage.as_ref().map(|n| n.energy);
    (out.mo_energy, out.mo_energy_beta) = (u.eps_alpha.clone(), Some(u.eps_beta.clone()));
    (out.nalpha, out.nbeta, out.s2) = (Some(r.nocc.0), Some(r.nocc.1), Some(u.s2));
    (out.gap_alpha, out.gap_beta) = (r.gaps.gap_alpha, r.gaps.gap_beta);
    Ok(out)
}

// ─────────────────────────────────────────────── k-point MP2 / dRPA ──

/// Result of `run_mp2_kpts` / `run_drpa_kpts` (k-point RHF + correlation).
#[pyclass]
#[pyo3(name = "KpointCorrelationResult")]
struct PyKpointCorrelationResult {
    /// `"mp2"` or `"drpa"`.
    #[pyo3(get)]
    method: String,
    /// `e_scf + correlation_energy` per cell.
    #[pyo3(get)]
    energy: f64,
    #[pyo3(get)]
    e_scf: f64,
    #[pyo3(get)]
    correlation_energy: f64,
    /// MP2 only: opposite-/same-spin parts and the direct (Coulomb-only)
    /// MP2, i.e. the O(Pi^2) term of dRPA.
    #[pyo3(get)]
    e_os: Option<f64>,
    #[pyo3(get)]
    e_ss: Option<f64>,
    #[pyo3(get)]
    e_direct: Option<f64>,
    /// dRPA only: per-q-class contributions and the energy construction
    /// ("quadrature" | "plasmon" | "second-order").
    #[pyo3(get)]
    per_q: Option<Vec<f64>>,
    #[pyo3(get)]
    drpa_energy: Option<String>,
    #[pyo3(get)]
    quad_points: Option<usize>,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    madelung: f64,
    #[pyo3(get)]
    occ_shift: f64,
    #[pyo3(get)]
    nocc_active: usize,
    /// Virtuals per k.
    #[pyo3(get)]
    nvir: Vec<usize>,
    /// dRPA only: pair-tensor rows per q class.
    #[pyo3(get)]
    naux: Option<Vec<usize>>,
    #[pyo3(get)]
    mesh: (usize, usize, usize),
    #[pyo3(get)]
    centring: String,
    #[pyo3(get)]
    nk: usize,
    #[pyo3(get)]
    exxdiv: String,
    #[pyo3(get)]
    denominators: String,
    #[pyo3(get)]
    lindep_total_kept: usize,
    #[pyo3(get)]
    lindep_min_kept: usize,
    #[pyo3(get)]
    lindep_near_noise_floor: bool,
    #[pyo3(get)]
    jk: String,
    #[pyo3(get)]
    auxbasis: Option<String>,
}

#[pymethods]
impl PyKpointCorrelationResult {
    fn __repr__(&self) -> String {
        format!(
            "KpointCorrelationResult(method={:?}, correlation_energy={:.10}, mesh={:?}, \
             exxdiv={:?}, denominators={:?})",
            self.method, self.correlation_energy, self.mesh, self.exxdiv, self.denominators
        )
    }
}

/// k-point integrals kept alive past the SCF for the correlation step.
enum KInts {
    Dense(Box<KDenseAftPairs>),
    RsGdf(Box<KRsGdf>),
}

impl KInts {
    fn corr(&self) -> KCorrIntegrals<'_> {
        match self {
            KInts::Dense(p) => KCorrIntegrals::DenseAft(p),
            KInts::RsGdf(g) => KCorrIntegrals::RsGdf(g),
        }
    }
}

fn require_kscf_converged(r: &KScfResult) -> Result<(), FerricError> {
    if r.converged {
        return Ok(());
    }
    Err(FerricError::Convergence(format!(
        "the k-point RHF reference did not converge in {} iterations (last E = {})",
        r.iterations, r.energy
    )))
}

/// k-point RHF whose integrals outlive the SCF (dense: the SCF kernels are
/// dropped and the correlation pair tensors built; rsgdf: one object).
fn krhf_with_ints(
    s: &PbcSetup,
    mesh: &KPointMesh,
    scf: &KScfConfig,
) -> Result<(KScfResult, KInts), FerricError> {
    let hk = periodic_hcore_kpts(
        &s.cell,
        &s.prep,
        mesh,
        &PeriodicHcoreConfig::with_omega(s.omega_bohr),
    )?;
    match &s.aux {
        None => {
            let kc = kdense_config(s);
            let eri = KDenseAftEri::build(&s.cell, &s.prep, mesh, &hk.s, s.exx, &kc)?;
            let inj = KPointInjection {
                s: hk.s,
                h: hk.h,
                vnn: hk.enn,
                jk: Box::new(eri.jk_builder()),
            };
            let r = solve_krhf_injected(&s.cell, mesh, scf, inj)?;
            drop(eri);
            require_kscf_converged(&r)?;
            let pairs = KDenseAftPairs::build(&s.cell, &s.prep, mesh, &kc)?;
            Ok((r, KInts::Dense(Box::new(pairs))))
        }
        Some(aux) => {
            let gdf = KRsGdf::build(&s.cell, &s.prep, aux, mesh, &hk.s, &krsgdf_config(s))?
                .with_exxdiv(s.exx);
            let inj = KPointInjection {
                s: hk.s,
                h: hk.h,
                vnn: hk.enn,
                jk: Box::new(gdf.jk_builder()),
            };
            let r = solve_krhf_injected(&s.cell, mesh, scf, inj)?;
            require_kscf_converged(&r)?;
            Ok((r, KInts::RsGdf(Box::new(gdf))))
        }
    }
}

struct KCorrOpts {
    /// `None` = MP2; `Some((energy, quad_points))` = dRPA.
    drpa: Option<(KDrpaEnergy, usize)>,
    den: Mp2Denominators,
    frozen_core: usize,
    scf: KScfConfig,
}

fn kcorr_driver(
    s: &PbcSetup,
    mesh: &KPointMesh,
    o: &KCorrOpts,
) -> Result<(KScfResult, PyKpointCorrelationResult), FerricError> {
    let (r, ints) = krhf_with_ints(s, mesh, &o.scf)?;
    let mut out = kcorr_result_base(s, mesh, &r, o);
    match o.drpa {
        None => {
            let cfg = KMp2Config {
                frozen_core: o.frozen_core,
                reference_exxdiv: s.exx,
                denominators: o.den,
                budget_bytes: s.budget_bytes,
                mutation: None,
            };
            let m = kpoint_mp2(&s.cell, mesh, &r, ints.corr(), &cfg)?;
            (out.energy, out.correlation_energy) = (m.total_energy, m.mp2_corr);
            (out.e_os, out.e_ss, out.e_direct) = (Some(m.e_os), Some(m.e_ss), Some(m.e_direct));
            (out.madelung, out.occ_shift) = (m.madelung, m.occ_shift);
            (out.nocc_active, out.nvir) = (m.nocc_active, m.nvir);
        }
        Some((energy, q)) => {
            let cfg = KDrpaConfig {
                frozen_core: o.frozen_core,
                reference_exxdiv: s.exx,
                denominators: o.den,
                quad_points: q,
                energy,
                budget_bytes: s.budget_bytes,
                mutation: None,
            };
            let d = kpoint_drpa(&s.cell, mesh, &r, ints.corr(), &cfg)?;
            (out.energy, out.correlation_energy) = (d.total_energy, d.drpa_corr);
            (out.per_q, out.naux) = (Some(d.per_q), Some(d.naux));
            (out.madelung, out.occ_shift) = (d.madelung, d.occ_shift);
            (out.nocc_active, out.nvir) = (d.nocc_active, d.nvir);
        }
    }
    Ok((r, out))
}

fn kcorr_result_base(
    s: &PbcSetup,
    mesh: &KPointMesh,
    r: &KScfResult,
    o: &KCorrOpts,
) -> PyKpointCorrelationResult {
    let n = mesh.n();
    PyKpointCorrelationResult {
        method: if o.drpa.is_some() { "drpa" } else { "mp2" }.into(),
        energy: f64::NAN,
        e_scf: r.energy,
        correlation_energy: f64::NAN,
        e_os: None,
        e_ss: None,
        e_direct: None,
        per_q: None,
        drpa_energy: o.drpa.map(|(e, _)| kdrpa_energy_name(e)),
        quad_points: match o.drpa {
            Some((KDrpaEnergy::Quadrature, q)) => Some(q),
            _ => None,
        },
        converged: r.converged,
        iterations: r.iterations,
        madelung: f64::NAN,
        occ_shift: f64::NAN,
        nocc_active: 0,
        nvir: Vec::new(),
        naux: None,
        mesh: (n[0], n[1], n[2]),
        centring: centring_name(mesh.centring()),
        nk: mesh.nk(),
        exxdiv: exx_name(s.exx),
        denominators: match o.den {
            Mp2Denominators::MadelungShifted => "shifted".into(),
            Mp2Denominators::Unshifted => "unshifted".into(),
        },
        lindep_total_kept: r.lindep.total_kept,
        lindep_min_kept: r.lindep.min_kept,
        lindep_near_noise_floor: r.lindep.near_noise_floor,
        jk: s.jk_name(),
        auxbasis: s.aux_name.clone(),
    }
}

fn run_kcorr(
    py: Python<'_>,
    s: PbcSetup,
    mesh: KPointMesh,
    o: KCorrOpts,
) -> PyResult<PyKpointCorrelationResult> {
    let (_, out) = py
        .allow_threads(|| kcorr_driver(&s, &mesh, &o))
        .map_err(pbc_err(s.fname))?;
    Ok(out)
}

/// **k-point MP2**: k-point RHF + closed-shell KMP2 in one call. Mesh /
/// centring / jk contract as `run_rhf_kpts`; `exxdiv` (the reference's) and
/// `denominators` ("shifted" | "unshifted") are REQUIRED, as in
/// `run_mp2_gamma`. jk="dense" uses exact dense pair tensors (toy scale);
/// jk="rsgdf" the SCF's own per-q RS-GDF blocks.
///
/// Validated vs crates/ferric-pbc/tests/pbc_kcorr.rs (PySCF 2.13 KMP2 on
/// KRHF, AFTDF): H2 / STO-3G (PySCF digits), a = 4 Bohr, Gamma-centred
/// 1x1x2: -2.8883196728367e-2 (none/unshifted), -1.8228154905146e-2
/// (ewald/shifted).
#[pyfunction]
#[pyo3(signature = (
    mol, lattice, basis_set, mesh, exxdiv, denominators, centring="gamma",
    omega=None, max_eri_gb=None, max_iter=200, energy_conv=1e-12,
    grad_conv=1e-9, jk="dense", auxbasis=None, memory_budget_gb=None,
    frozen_core=0,
))]
#[allow(clippy::too_many_arguments)]
fn run_mp2_kpts(
    py: Python<'_>,
    mol: &PyMolecule,
    lattice: Vec<Vec<f64>>,
    basis_set: &PyBasisSet,
    mesh: (usize, usize, usize),
    exxdiv: &str,
    denominators: &str,
    centring: &str,
    omega: Option<f64>,
    max_eri_gb: Option<f64>,
    max_iter: usize,
    energy_conv: f64,
    grad_conv: f64,
    jk: &str,
    auxbasis: Option<&Bound<'_, PyAny>>,
    memory_budget_gb: Option<f64>,
    frozen_core: usize,
) -> PyResult<PyKpointCorrelationResult> {
    let fname = "run_mp2_kpts";
    let s = pbc_setup(&PbcArgs {
        fname,
        mol,
        lattice: &lattice,
        basis: basis_set,
        exxdiv,
        omega,
        jk,
        auxbasis,
        max_eri_gb,
        memory_budget_gb,
        closed_shell: true,
    })?;
    let m = k_mesh(&s, mesh, centring)?;
    let o = KCorrOpts {
        drpa: None,
        den: parse_denominators(fname, denominators)?,
        frozen_core,
        scf: kscf_config(fname, max_iter, energy_conv, grad_conv)?,
    };
    run_kcorr(py, s, m, o)
}

/// **k-point dRPA**: k-point RHF + closed-shell k-dRPA in one call. Contract
/// as `run_mp2_kpts`, plus `energy` ("quadrature" (default): frequency
/// quadrature with `quad_points` points, None = 40; "plasmon": the exact
/// plasmon formula; "second-order": the O(Pi^2) term, == direct KMP2).
/// `quad_points` with "plasmon"/"second-order" is a ValueError (ignored).
///
/// Validated: a 1x1x1 mesh reproduces the Gamma dRPA pins of
/// crates/ferric-pbc/tests/pbc_drpa.rs (H2 / STO-3G, a = 4 Bohr:
/// -1.000052013959e-2 unshifted, -6.938130614440e-3 shifted), the identity
/// pinned in pbc_kcorr.rs::one_point_mesh_is_the_gamma_mp2_and_drpa.
#[pyfunction]
#[pyo3(signature = (
    mol, lattice, basis_set, mesh, exxdiv, denominators, centring="gamma",
    omega=None, max_eri_gb=None, max_iter=200, energy_conv=1e-12,
    grad_conv=1e-9, jk="dense", auxbasis=None, memory_budget_gb=None,
    frozen_core=0, energy="quadrature", quad_points=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_drpa_kpts(
    py: Python<'_>,
    mol: &PyMolecule,
    lattice: Vec<Vec<f64>>,
    basis_set: &PyBasisSet,
    mesh: (usize, usize, usize),
    exxdiv: &str,
    denominators: &str,
    centring: &str,
    omega: Option<f64>,
    max_eri_gb: Option<f64>,
    max_iter: usize,
    energy_conv: f64,
    grad_conv: f64,
    jk: &str,
    auxbasis: Option<&Bound<'_, PyAny>>,
    memory_budget_gb: Option<f64>,
    frozen_core: usize,
    energy: &str,
    quad_points: Option<usize>,
) -> PyResult<PyKpointCorrelationResult> {
    let fname = "run_drpa_kpts";
    let s = pbc_setup(&PbcArgs {
        fname,
        mol,
        lattice: &lattice,
        basis: basis_set,
        exxdiv,
        omega,
        jk,
        auxbasis,
        max_eri_gb,
        memory_budget_gb,
        closed_shell: true,
    })?;
    let m = k_mesh(&s, mesh, centring)?;
    let e = parse_kdrpa_energy(fname, energy)?;
    if let (Some(q), false) = (quad_points, e == KDrpaEnergy::Quadrature) {
        return Err(val_err(format!(
            "{fname}: quad_points={q} is only used by energy=\"quadrature\"; \
             energy={energy:?} would ignore it"
        )));
    }
    let o = KCorrOpts {
        drpa: Some((e, quad_points.unwrap_or(DEFAULT_KDRPA_QUAD_POINTS))),
        den: parse_denominators(fname, denominators)?,
        frozen_core,
        scf: kscf_config(fname, max_iter, energy_conv, grad_conv)?,
    };
    run_kcorr(py, s, m, o)
}

// ─────────────────────────────────────────────────────────── register ──

/// Register the periodic bindings on the `ferric` module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_uhf_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(run_rohf_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(run_uks_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(run_roks_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(run_rks_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(run_mp2_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(run_drpa_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(run_rhf_kpts, m)?)?;
    m.add_function(wrap_pyfunction!(run_uhf_kpts, m)?)?;
    m.add_function(wrap_pyfunction!(run_mp2_kpts, m)?)?;
    m.add_function(wrap_pyfunction!(run_drpa_kpts, m)?)?;
    m.add_class::<PyGammaOpenShellResult>()?;
    m.add_class::<PyGammaRksResult>()?;
    m.add_class::<PyGammaCorrelationResult>()?;
    m.add_class::<PyKpointScfResult>()?;
    m.add_class::<PyKpointCorrelationResult>()?;
    Ok(())
}
