//! Gamma-point periodic RHF binding: moved verbatim from `lib.rs`.

use super::*;

// ── Periodic (Gamma-point) RHF ──

/// Result of `run_rhf_gamma`: closed-shell Gamma-point periodic RHF.
///
/// Energies are per unit cell, Hartree. `energy` includes `e_nuc` (Ewald,
/// neutralising-background convention = PySCF `Cell.energy_nuc()`) and, for
/// `exxdiv="ewald"`, the Madelung exchange shift `-madelung * N_e / 2`.
#[pyclass]
#[pyo3(name = "GammaRhfResult")]
pub(crate) struct PyGammaRhfResult {
    /// Total energy per cell (Hartree).
    #[pyo3(get)]
    pub(crate) energy: f64,
    /// Whether the SCF converged.
    #[pyo3(get)]
    pub(crate) converged: bool,
    /// SCF iterations run.
    #[pyo3(get)]
    pub(crate) iterations: usize,
    /// Ewald nuclear repulsion per cell (Hartree).
    #[pyo3(get)]
    pub(crate) e_nuc: f64,
    /// Gamma-point Madelung constant `v_M` of the lattice (Hartree, a.u.);
    /// matches `pyscf.pbc.tools.madelung`. Reported for BOTH `exxdiv`
    /// settings; it is only APPLIED (as `K += v_M S D S`) for `"ewald"`.
    #[pyo3(get)]
    pub(crate) madelung: f64,
    /// The exxdiv treatment used, lower-case (`"ewald"` or `"none"`).
    #[pyo3(get)]
    pub(crate) exxdiv: String,
    /// Nuclear-attraction Ewald splitting parameter used, in **Å⁻¹** (the
    /// same unit the `omega` kwarg takes).
    #[pyo3(get)]
    pub(crate) omega: f64,
    /// Number of AO basis functions in the cell.
    #[pyo3(get)]
    pub(crate) nao: usize,
    /// Half-sphere G vectors in the dense AFT ERI build (`jk="dense"`);
    /// `None` for `jk="rsgdf"`, whose long-range G list is a different
    /// quantity and is not reported under this name.
    #[pyo3(get)]
    pub(crate) n_g_half: Option<usize>,
    /// J/K builder used: `"dense"` (pure-AFT nao^4 oracle) or `"rsgdf"`
    /// (range-separated Gaussian density fitting).
    #[pyo3(get)]
    pub(crate) jk: String,
    /// Aux basis name (`BasisSet.name`) for `jk="rsgdf"`; `None` for dense.
    #[pyo3(get)]
    pub(crate) auxbasis: Option<String>,
    /// Aux functions in the cell (`jk="rsgdf"`); `None` for dense.
    #[pyo3(get)]
    pub(crate) naux: Option<usize>,
    /// Aux-metric eigenvectors kept (`naux - n_dropped`); `None` for dense.
    #[pyo3(get)]
    pub(crate) naux_kept: Option<usize>,
    /// Aux-metric eigenvalues `<= lindep` (1e-10) dropped as linearly
    /// dependent; `None` for dense.
    #[pyo3(get)]
    pub(crate) n_dropped: Option<usize>,
    pub(crate) overlap_data: Array2<f64>,
    pub(crate) scf_data: ScfResult,
    pub(crate) timings_data: ferric_pbc::PbcTimings,
    pub(crate) gradient_data: Option<Array2<f64>>,
    pub(crate) stress_data: Option<Array2<f64>>,
}

#[pymethods]
impl PyGammaRhfResult {
    /// Validation certificate (see `GammaOpenShellResult.certificate`):
    /// capability `"rhf-gamma"`, `grade` `"ungraded"`.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        pbc::pbc_certificate_dict(
            py,
            "rhf-gamma",
            &self.jk,
            self.auxbasis.as_deref(),
            &self.exxdiv,
            None,
            self.converged,
            self.iterations,
        )
    }
    /// Analytic nuclear gradient `dE/dR` (natoms x 3, Hartree/Bohr per cell,
    /// rows in the Molecule's atom order); `None` unless `with_gradient=True`.
    fn gradient<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.gradient_data
            .as_ref()
            .map(|g| PyArray2::from_array(py, g))
    }
    /// Analytic stress `σ_ab = (1/Ω) dE/dε_ab` (3 x 3, Hartree/Bohr³; strain
    /// of lattice rows AND atoms; pressure = -tr(σ)/3); `None` unless
    /// `with_stress=True`.
    fn stress<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.stress_data
            .as_ref()
            .map(|g| PyArray2::from_array(py, g))
    }
    /// Stage timings and counters: hcore (setup, S/T, SR/LR attraction,
    /// Ewald), the J/K build (RS-GDF: SR metric, SR 3-centre, LR pair FT,
    /// LR GEMM, G = 0, metric eigh, B; dense: pair FT, GEMM), the SCF's
    /// J/K calls, and counters (triplets, G vectors, chunks, aux dropped).
    /// `{"wall_s", "cpu_s", "unattributed_wall_s", "stages": {name: {"wall_s",
    /// "cpu_s", "calls"}}, "counters": {name: int}}`.
    #[getter]
    fn timings(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        pbc::timings_dict(py, &self.timings_data)
    }
    /// MO energies (Hartree), ascending, 1D numpy array.
    fn mo_energy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.scf_data.eps_alpha.clone())
    }
    /// MO coefficients C (nao x nmo), column k = MO k (ascending energy),
    /// orthonormal in the LATTICE-SUMMED overlap (`overlap()`).
    fn mo_coeff<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.scf_data.mos_alpha)
    }
    /// Total AO density matrix (nao x nao); `tr(D S) = N_e`.
    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.scf_data.density_total)
    }
    /// Lattice-summed Gamma-point AO overlap `S` (nao x nao).
    fn overlap<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.overlap_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "GammaRhfResult(energy={:.10}, exxdiv={:?}, jk={:?}, converged={})",
            self.energy, self.exxdiv, self.jk, self.converged
        )
    }
}

/// Reject what any periodic driver cannot represent (a charged cell), as a
/// `ValueError` naming the offending input, and return the cell's molecule
/// with the basis set's ECPs applied (`Molecule::apply_ecp`, exactly as the
/// molecular bindings do), so `n_core_ecp` / `effective_z` are set before
/// the `Cell` is built. Every periodic driver builds its one-electron
/// Hamiltonian through `periodic_hcore` / `periodic_hcore_kpts`, which add
/// the lattice-summed V_ECP and whose `check_ecp_applied` guard stays the
/// backstop against a bare-Z cell. Without an ECP basis the returned
/// molecule is a plain clone (`apply_ecp` is a no-op).
pub(crate) fn validate_periodic_cell(
    fname: &str,
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
) -> PyResult<Molecule> {
    if mol.charge != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{fname}: charged cell (charge = {}) is not supported: a periodic \
             charged cell needs a neutralising-background correction for the \
             electrons that is not implemented. Use a neutral cell.",
            mol.charge
        )));
    }
    let mut emol = mol.clone();
    emol.apply_ecp(bs);
    Ok(emol)
}

/// Reject what a closed-shell periodic driver (RHF/RKS and the correlation
/// methods on top) cannot represent: [`validate_periodic_cell`] plus an
/// open shell or an odd electron count. The count is the VALENCE count of
/// the ECP-applied molecule (what the SCF actually occupies). Returns that
/// ECP-applied molecule. Kept out of the bindings so their bodies stay thin
/// dispatch layers.
pub(crate) fn validate_gamma_cell(
    fname: &str,
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
) -> PyResult<Molecule> {
    let bad = |m: String| -> PyResult<Molecule> { Err(pyo3::exceptions::PyValueError::new_err(m)) };
    let emol = validate_periodic_cell(fname, mol, bs)?;
    if emol.multiplicity != 1 {
        return bad(format!(
            "{fname}: multiplicity {} requested, but this driver is closed-shell \
             (multiplicity 1) only; use the UHF/ROHF/UKS/ROKS periodic drivers \
             for an open shell",
            emol.multiplicity
        ));
    }
    let nelec = emol.nelec();
    if nelec <= 0 || nelec % 2 != 0 {
        return bad(format!(
            "{fname}: {nelec} electrons per cell; a closed-shell driver needs a \
             positive even count"
        ));
    }
    Ok(emol)
}

/// What the Gamma-point driver hands back across the GIL boundary.
pub(crate) struct GammaRun {
    pub(crate) scf: ScfResult,
    pub(crate) s: Array2<f64>,
    pub(crate) enn: f64,
    pub(crate) omega_bohr: f64,
    pub(crate) madelung: f64,
    /// Dense path only.
    pub(crate) n_g_half: Option<usize>,
    /// RS-GDF path only: `(naux, naux_kept, n_dropped)`.
    pub(crate) rsgdf_counts: Option<(usize, usize, usize)>,
    /// hcore + J/K build + SCF J/K stage timings.
    pub(crate) timings: ferric_pbc::PbcTimings,
    /// Analytic forces / stress (when requested).
    pub(crate) derivs: pbc::Derivs,
}

/// Which periodic J/K builder `run_rhf_gamma` injects.
pub(crate) enum GammaJk {
    /// Dense pure-AFT ERI, hard-capped at `max_bytes`.
    Dense { max_bytes: usize },
    /// RS-GDF with `aux` (prepared on the cell's atoms), an optional
    /// explicit budget (`None` = ferric's unified budget chain) and the
    /// opt-in range split (`None` = off; energy-only, see
    /// `pbc::parse_range_split`). `gdf_omega_bohr`: the RS-GDF Ewald split
    /// (Bohr⁻¹; `None` = `DEFAULT_RSGDF_OMEGA`, bit for bit).
    /// `sr_column_rotation`: the RESOLVED column rotation of the hcore and
    /// RS-GDF SR walks (`On` / `Off`, never `Auto`; on by default, forces
    /// and stress differentiate the same walks, see
    /// `pbc::parse_sr_column_rotation`).
    RsGdf {
        aux: Box<PreparedBasis>,
        budget_bytes: Option<usize>,
        range_split: Option<ferric_pbc::RangeSplit>,
        gdf_omega_bohr: Option<f64>,
        sr_column_rotation: ferric_pbc::SrColumnRotation,
    },
}

impl GammaJk {
    /// The explicit RS-GDF Ewald split (Bohr⁻¹; `None` = the default, or
    /// dense AFT).
    fn gdf_omega_bohr(&self) -> Option<f64> {
        match self {
            GammaJk::Dense { .. } => None,
            GammaJk::RsGdf { gdf_omega_bohr, .. } => *gdf_omega_bohr,
        }
    }

    /// The resolved SR column rotation of the hcore and RS-GDF builds
    /// (`Off` with dense AFT, which never rotates).
    fn sr_column_rotation(&self) -> ferric_pbc::SrColumnRotation {
        match self {
            GammaJk::Dense { .. } => ferric_pbc::SrColumnRotation::Off,
            GammaJk::RsGdf {
                sr_column_rotation, ..
            } => *sr_column_rotation,
        }
    }
}

/// Periodic hcore -> injected J/K (dense AFT or RS-GDF) ->
/// `solve_rhf_injected`. The same assembly as
/// `crates/ferric-pbc/tests/common/mod.rs::{gamma_rhf, gamma_rhf_jk}`.
pub(crate) fn gamma_rhf_driver(
    cell: &ferric_pbc::Cell,
    prep: &PreparedBasis,
    exx: ferric_pbc::ExxDiv,
    omega_bohr: Option<f64>,
    jk: &GammaJk,
    config: &RhfConfig,
    want: pbc::DerivRequest,
) -> Result<GammaRun, ferric_core::FerricError> {
    use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_PRECISION};
    use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
    use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
    let total = ferric_pbc::StageClock::start();
    // The default hcore split is capped by the RS-GDF LR sphere, so it
    // follows an explicit gdf_omega (absent: `for_cell`, bit for bit).
    let w = match omega_bohr {
        Some(w) => w,
        None => PeriodicHcoreConfig::for_cell_and_gdf_omega(cell, jk.gdf_omega_bohr()).omega,
    };
    let hcfg = PeriodicHcoreConfig {
        sr_column_rotation: jk.sr_column_rotation(),
        ..PeriodicHcoreConfig::with_omega(w)
    };
    let hc = periodic_hcore(cell, prep, &hcfg)?;
    let mut timings = ferric_pbc::PbcTimings::default();
    timings.absorb(&hc.timings);
    let op = Operator::coulomb();
    // Never read on the injected path; required by the signature.
    let bounds = SchwarzBounds::compute(op, prep)?;
    let ctx = ParallelContext::default();
    #[allow(clippy::too_many_arguments)]
    fn inject<'a>(
        ctx: &ParallelContext,
        cell: &ferric_pbc::Cell,
        prep: &PreparedBasis,
        bounds: &SchwarzBounds,
        config: &RhfConfig,
        hc: &ferric_pbc::PeriodicHcore,
        j: Box<dyn ferric_scf::fock::JBuilder + 'a>,
        k: Box<dyn ferric_scf::fock::KBuilder + 'a>,
    ) -> Result<ScfResult, ferric_core::FerricError> {
        let inj = ferric_scf::rhf::PeriodicInjection {
            s: hc.s.clone(),
            h: hc.h.clone(),
            vnn: hc.enn,
            j,
            k,
            xc: None,
        };
        let op = Operator::coulomb();
        ferric_scf::rhf::solve_rhf_injected(ctx, cell.mol(), prep, op, bounds, config, inj)
    }
    let derivs_of = |scf: &ScfResult,
                     jk: pbc::DerivJk<'_>,
                     budget_bytes: Option<usize>|
     -> Result<pbc::Derivs, ferric_core::FerricError> {
        if !want.any() {
            return Ok(pbc::Derivs::default());
        }
        let ctx = pbc::DerivCtx {
            cell,
            prep,
            hcfg: &hcfg,
            hc: &hc,
            jk,
            scf,
            budget_bytes,
            // Read only by a meta-GGA RKS derivative; this is RHF.
            max_eri_bytes: ferric_pbc::dense_aft::DEFAULT_DENSE_AFT_MAX_BYTES,
        };
        pbc::gamma_derivatives(&ctx, pbc::DerivMethod::Rhf(exx), want)
    };
    let (scf, n_g_half, rsgdf_counts, derivs) = match jk {
        GammaJk::Dense { max_bytes } => {
            let eri = DenseAftEri::build(
                cell,
                prep,
                &hc.s,
                exx,
                DEFAULT_DENSE_AFT_PRECISION,
                *max_bytes,
            )?;
            let (j, k) = (Box::new(eri.j_builder()), Box::new(eri.k_builder()));
            let scf = inject(&ctx, cell, prep, &bounds, config, &hc, j, k)?;
            timings.absorb(&eri.timings());
            let d = derivs_of(&scf, pbc::DerivJk::Dense(&eri), None)?;
            (scf, Some(eri.n_g_half()), None, d)
        }
        GammaJk::RsGdf {
            aux,
            budget_bytes,
            range_split,
            gdf_omega_bohr,
            sr_column_rotation,
        } => {
            let cfg = RsGdfConfig {
                omega: gdf_omega_bohr.unwrap_or(ferric_pbc::rsgdf::DEFAULT_RSGDF_OMEGA),
                exxdiv: exx,
                budget_bytes: *budget_bytes,
                range_split: *range_split,
                sr_column_rotation: *sr_column_rotation,
                ..Default::default()
            };
            // The gradient build is bitwise the energy's B plus the metric
            // pieces the RS-GDF forces/stress need.
            let gdf = if want.any() {
                RsGdf::build_for_gradient(cell, prep, aux, &hc.s, &cfg)?
            } else {
                RsGdf::build(cell, prep, aux, &hc.s, &cfg)?
            };
            let st = gdf.stats();
            let counts = (st.naux, st.naux_kept, st.n_dropped);
            let (j, k) = (Box::new(gdf.j_builder()), Box::new(gdf.k_builder()));
            let scf = inject(&ctx, cell, prep, &bounds, config, &hc, j, k)?;
            timings.absorb(&gdf.timings());
            let fit = ferric_pbc::RsGdfGradSource {
                gdf: &gdf,
                aux,
                aux_jac: None,
            };
            let d = derivs_of(&scf, pbc::DerivJk::Fit(fit), *budget_bytes)?;
            (scf, None, Some(counts), d)
        }
    };
    // Reported for both exxdiv settings (the builders only store it for Ewald).
    let madelung = ferric_pbc::ewald::madelung_constant(cell)?;
    timings.finish(&total);
    Ok(GammaRun {
        scf,
        s: hc.s,
        enn: hc.enn,
        omega_bohr: hc.omega,
        madelung,
        n_g_half,
        rsgdf_counts,
        timings,
        derivs,
    })
}

/// Resolve `run_rhf_gamma`'s `auxbasis` kwarg: a `BasisSet` object or a
/// bundled basis name. An unknown name is a `ValueError`.
pub(crate) fn gamma_auxbasis(
    fname: &str,
    obj: &Bound<'_, PyAny>,
) -> PyResult<ferric_core::basis::BasisSet> {
    if let Ok(bs) = obj.extract::<PyRef<'_, PyBasisSet>>() {
        return Ok(bs.inner.clone());
    }
    if let Ok(name) = obj.extract::<String>() {
        return basis::bundled(&name).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{fname}: auxbasis {name:?} is not a bundled basis: {e}"
            ))
        });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{fname}: auxbasis must be a BasisSet or a bundled basis name (str)"
    )))
}

/// Parsed and validated `run_rhf_gamma` options (strict: every knob the chosen
/// J/K path would ignore is an error, never a silent no-op).
pub(crate) struct GammaOptions {
    pub(crate) exx: ferric_pbc::ExxDiv,
    pub(crate) lattice_bohr: [[f64; 3]; 3],
    pub(crate) omega_bohr: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_gamma_options(
    fname: &str,
    exxdiv: &str,
    jk: &str,
    auxbasis_given: bool,
    max_eri_gb: Option<f64>,
    memory_budget_gb: Option<f64>,
    lattice: &[Vec<f64>],
    omega: Option<f64>,
) -> PyResult<GammaOptions> {
    let auxbasis: Option<()> = if auxbasis_given { Some(()) } else { None };
    let val_err = |m: String| pyo3::exceptions::PyValueError::new_err(m);
    let exx = ferric_pbc::ExxDiv::parse_config_str(exxdiv).map_err(|e| val_err(format!("{e}")))?;
    let use_rsgdf = match jk.to_ascii_lowercase().as_str() {
        "dense" => false,
        "rsgdf" => true,
        other => {
            return Err(val_err(format!(
                "{fname}: jk must be \"dense\" or \"rsgdf\", got {other:?}"
            )))
        }
    };
    // Knobs the chosen path would ignore are errors, not silent no-ops.
    if use_rsgdf {
        if auxbasis.is_none() {
            return Err(val_err(format!(
                "{fname}: jk=\"rsgdf\" requires auxbasis (a BasisSet or a bundled \
                 name such as \"cc-pvdz-ri\" or \"def2-universal-jkfit\"); there is no \
                 default aux basis"
            )));
        }
        if let Some(g) = max_eri_gb {
            return Err(val_err(format!(
                "{fname}: max_eri_gb={g} caps the dense AFT tensor and is ignored \
                 by jk=\"rsgdf\"; bound RS-GDF with memory_budget_gb instead"
            )));
        }
    } else {
        if auxbasis.is_some() {
            return Err(val_err(format!(
                "{fname}: auxbasis is only used by jk=\"rsgdf\"; jk=\"dense\" would \
                 ignore it. Pass jk=\"rsgdf\" or drop auxbasis."
            )));
        }
        if let Some(g) = memory_budget_gb {
            return Err(val_err(format!(
                "{fname}: memory_budget_gb={g} bounds RS-GDF and is ignored by \
                 jk=\"dense\"; bound the dense tensor with max_eri_gb instead"
            )));
        }
    }
    if lattice.len() != 3 || lattice.iter().any(|r| r.len() != 3) {
        return Err(val_err(format!(
            "{fname}: lattice must be 3x3 (three row vectors, Ångström), got {} rows",
            lattice.len()
        )));
    }
    let mut a = [[0.0_f64; 3]; 3];
    for (i, row) in lattice.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            a[i][j] = v * ANGSTROM_TO_BOHR;
        }
    }
    let omega_bohr = match omega {
        None => None,
        Some(w) if w.is_finite() && w > 0.0 => Some(w / ANGSTROM_TO_BOHR),
        Some(w) => {
            return Err(val_err(format!(
                "{fname}: omega must be finite and > 0 (Å⁻¹), got {w}"
            )))
        }
    };
    for (name, v) in [
        ("max_eri_gb", max_eri_gb),
        ("memory_budget_gb", memory_budget_gb),
    ] {
        if let Some(g) = v {
            if !(g.is_finite() && g > 0.0) {
                return Err(val_err(format!(
                    "{fname}: {name} must be finite and > 0, got {g}"
                )));
            }
        }
    }
    Ok(GammaOptions {
        exx,
        lattice_bohr: a,
        omega_bohr,
    })
}

/// Closed-shell **Gamma-point periodic RHF** (3-D periodic, one k-point).
///
/// Two J/K builders, chosen by `jk` (strict; default `"dense"`):
///
///   "dense"  TOY-SCALE ORACLE. The dense pure-AFT ERI (`ferric_pbc::dense_aft`):
///            an `8 * nao^4`-byte tensor, HARD-capped by `max_eri_gb` (default
///            0.5 GiB, i.e. nao <= ~90). An oversize cell is a `ValueError`
///            raised before any integral work. Exact to the AFT precision.
///   "rsgdf"  Range-separated Gaussian density fitting (`ferric_pbc::rsgdf`,
///            PySCF RSGDF convention, omega_gdf = 1 Bohr⁻¹ unless
///            `gdf_omega` is given, lindep 1e-10).
///            REQUIRES `auxbasis`: a `BasisSet` or a bundled name (e.g.
///            `"cc-pvdz-ri"`, `"def2-universal-jkfit"`). There is no default
///            aux basis: the fitting error depends on the aux/orbital pairing
///            (H2/STO-3G: -2.0e-6 Ha with cc-pvdz-ri, -4.5e-6 with
///            def2-universal-jkfit), so an implicit choice would hide it.
///            Memory is gated by ferric's unified budget: `memory_budget_gb`
///            (None = `FERRIC_MEM_BUDGET_GB` / 0.8 x available RAM).
///
/// `auxbasis`/`memory_budget_gb` with `jk="dense"`, or `max_eri_gb` with
/// `jk="rsgdf"`, is a `ValueError` (a knob the chosen path ignores would be a
/// silent no-op).
///
/// Units (the ferric Python convention):
///   mol       `Molecule` whose atoms are the reference cell; its coordinates
///             are Ångström on input (`from_xyz_string`), Bohr internally.
///   lattice   3x3 lattice vectors as ROWS, in **Ångström**
///             (`[[ax, ay, az], [bx, by, bz], [cx, cy, cz]]`), converted with
///             the same factor the XYZ parser uses.
///   omega     nuclear-attraction Ewald split in **Å⁻¹** (numerical knob;
///             any value > 0 gives the same energy to ~1e-9 Ha). None =
///             `min(2.5 sqrt(pi) / volume^(1/3), 0.9636 Bohr^-1)`
///             (`ferric_pbc::hcore::default_hcore_omega`, converted to
///             Å⁻¹ in `result.omega`; with `gdf_omega` given, the cap
///             follows it: `ferric_pbc::hcore::default_hcore_omega_for_gdf`).
///             Not the RS-GDF split, which is `gdf_omega`.
///   energies  Hartree per cell.
///
/// Arguments:
///   exxdiv       "ewald" (default; Madelung-corrected exchange, PySCF
///                `exxdiv='ewald'`, converges as L^-3) or "none" (drop G=0,
///                converges as 1/L). Strict: anything else is a ValueError.
///   jk           "dense" (default) or "rsgdf". Strict.
///   auxbasis     RS-GDF aux basis (BasisSet or bundled name); rsgdf only.
///   max_eri_gb   cap on the dense ERI tensor in GiB (> 0; default 0.5);
///                dense only.
///   memory_budget_gb  RS-GDF memory budget in GiB (> 0); rsgdf only.
///   max_iter, density_conv   SCF controls (defaults 200, 1e-10). The guess
///                is the core-Hamiltonian guess on the lattice S/h (the
///                molecular SAD guess is not periodic).
///   with_gradient  (default False) also compute the analytic nuclear
///                gradient: `result.gradient()`, natoms x 3, dE/dR in
///                Hartree/Bohr per cell (the molecular `gradient()`
///                convention). `None` when not requested.
///   with_stress  (default False) also compute the analytic stress:
///                `result.stress()`, 3 x 3, σ = (1/Ω) dE/dε in Hartree/Bohr³
///                (strain of lattice rows AND atoms). Both run on the SCF's
///                own J/K (rsgdf: B built with its gradient pieces, bitwise
///                the energy's) and need a converged SCF.
///   range_split  (default None) opt-in RS-GDF range split
///                (`ferric_pbc::rsgdf::RangeSplit`): None/False = off (the
///                unsplit build bit for bit), True = lambda 1 (the rigorous
///                default), a number > 0 = lambda. Moves the SR blocks whose
///                FT converges in the LR sphere (orbital primitive exponent
///                <= lambda*omega_gdf^2/2, aux <= lambda*omega_gdf^2) to G
///                space; lambda <= 1 adds no G vectors. The split's
///                primitive counters appear in `timings["counters"]`
///                ("rsgdf split ..."). rsgdf ENERGY only: with jk="dense" or
///                with_gradient/with_stress it is a ValueError.
///   gdf_omega    (default None) the RS-GDF Ewald split omega_gdf in **Å⁻¹**
///                (like `omega`; finite, > 0). None = 1 Bohr⁻¹
///                (`ferric_pbc::rsgdf::DEFAULT_RSGDF_OMEGA`), the default
///                build bit for bit. Any value gives the same energy up to the
///                fit's truncation; it moves work between the SR lattice sums
///                (radii ~ 1/omega_gdf) and the LR G sphere (|G| <=
///                2 omega_gdf sqrt(ln(1/1e-13)); `timings["counters"]["rsgdf
///                LR half-G"]`). The range split's thresholds and the
///                gradient/stress follow it. rsgdf only: with jk="dense" it is
///                a ValueError. Every periodic binding takes the same kwarg.
///   sr_column_rotation (default None = auto) column rotation of generally
///                contracted shells (cc-pVXZ-style s/p columns) in the Gamma
///                hcore SR attraction and RS-GDF SR 3-centre walks
///                (`ferric_pbc::SrColumnRotation`), back-transformed exactly.
///                Same energy to the screening precision, fewer SR integral
///                calls; with_gradient / with_stress differentiate the same
///                rotated walks (forces/stress equal the unrotated ones to the
///                screening precision). None: ON for a jk="rsgdf" run, OFF
///                with jk="dense". False: off (the unrotated build bit for
///                bit, as is any setting on a basis with nothing to rotate).
///                True: on, a ValueError with jk="dense". Counters
///                `timings["counters"]["hcore SR rotated columns"]` and
///                `["rsgdf SR3 rotated columns"]` (present whenever the
///                rotation was on, 0 = nothing in the basis rotates; absent
///                when it was off). The Gamma bindings take it; the k-point
///                bindings do not take the kwarg, and their RS-GDF energy
///                builds run the library default (rotated, counter
///                `["k rsgdf SR3 rotated columns"]`).
///
/// Hard errors (ValueError): charged cell (charge != 0; no neutralising-
/// background correction for electrons), multiplicity != 1 or an odd
/// electron count (RHF only; counted after the ECP), a non-3x3 or singular lattice,
/// a bad `exxdiv`/`jk`, an unknown bundled `auxbasis`, a knob given to the
/// path that ignores it, non-positive `omega`/`max_eri_gb`/`memory_budget_gb`,
/// an oversize dense cell.
///
/// Validated (crates/ferric-python/tests/test_pbc_gamma.py, and the Rust
/// `pbc_dense_aft_scf` / `pbc_rsgdf` suites): H2/STO-3G in a 4 Bohr cube vs
/// PySCF 2.13 AFTDF, E = -1.658327061049 (ewald) / -0.949002691179 (none),
/// to 1e-8 (dense); RS-GDF fitting error E_rsgdf - E_dense vs the Rust suite.
#[pyfunction]
#[pyo3(signature = (
    mol, lattice, basis_set, exxdiv="ewald", omega=None, max_eri_gb=None,
    max_iter=200, density_conv=1e-10, jk="dense", auxbasis=None,
    memory_budget_gb=None, with_gradient=false, with_stress=false,
    range_split=None, gdf_omega=None, sr_column_rotation=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_rhf_gamma(
    py: Python<'_>,
    mol: &PyMolecule,
    lattice: Vec<Vec<f64>>,
    basis_set: &PyBasisSet,
    exxdiv: &str,
    omega: Option<f64>,
    max_eri_gb: Option<f64>,
    max_iter: usize,
    density_conv: f64,
    jk: &str,
    auxbasis: Option<&Bound<'_, PyAny>>,
    memory_budget_gb: Option<f64>,
    with_gradient: bool,
    with_stress: bool,
    range_split: Option<&Bound<'_, PyAny>>,
    gdf_omega: Option<f64>,
    sr_column_rotation: Option<bool>,
) -> PyResult<PyGammaRhfResult> {
    let val_err = |m: String| pyo3::exceptions::PyValueError::new_err(m);
    let GammaOptions {
        exx,
        lattice_bohr: a,
        omega_bohr,
    } = parse_gamma_options(
        "run_rhf_gamma",
        exxdiv,
        jk,
        auxbasis.is_some(),
        max_eri_gb,
        memory_budget_gb,
        &lattice,
        omega,
    )?;
    let want = pbc::DerivRequest {
        gradient: with_gradient,
        stress: with_stress,
    };
    let range_split =
        pbc::parse_range_split("run_rhf_gamma", range_split, auxbasis.is_some(), want)?;
    let gdf_omega_bohr = pbc::parse_gdf_omega("run_rhf_gamma", gdf_omega, auxbasis.is_some())?;
    let sr_column_rotation =
        pbc::parse_sr_column_rotation("run_rhf_gamma", sr_column_rotation, auxbasis.is_some())?;
    let emol = validate_gamma_cell("run_rhf_gamma", &mol.inner, &basis_set.inner)?;
    let cell = ferric_pbc::Cell::new(emol, a).map_err(|e| val_err(format!("{e}")))?;
    let prep = PreparedBasis::new(cell.mol(), &basis_set.inner).map_err(make_err)?;
    let nao = prep.nbasis();
    let (jk_choice, aux_name) = match auxbasis {
        Some(obj) => {
            let aux_bs = gamma_auxbasis("run_rhf_gamma", obj)?;
            let aux = PreparedBasis::new(cell.mol(), &aux_bs).map_err(|e| {
                val_err(format!(
                    "run_rhf_gamma: auxbasis '{}' cannot be placed on this cell: {e}",
                    aux_bs.name
                ))
            })?;
            let choice = GammaJk::RsGdf {
                aux: Box::new(aux),
                budget_bytes: budget_bytes_from_gb(memory_budget_gb),
                range_split,
                gdf_omega_bohr,
                sr_column_rotation,
            };
            (choice, Some(aux_bs.name))
        }
        None => {
            let max_bytes = ferric_core::memory::gib_to_bytes(max_eri_gb.unwrap_or(0.5));
            // Pre-flight the dense tensor BEFORE the hcore lattice sums
            // (DenseAftEri re-checks the same bound, so this cannot drift into
            // a silent pass).
            let need = 8u128 * (nao as u128).pow(4);
            if need > max_bytes as u128 {
                return Err(val_err(format!(
                    "run_rhf_gamma: the dense AFT ERI tensor needs {need} bytes (nao = {nao}) \
                     > max_eri_gb cap {max_bytes} bytes. This path is a toy-scale oracle; \
                     use jk=\"rsgdf\" with an auxbasis for larger cells."
                )));
            }
            (GammaJk::Dense { max_bytes }, None)
        }
    };
    let config = RhfConfig {
        use_sad_guess: false,
        density_conv,
        max_iter,
        ..Default::default()
    };
    let run = py
        .allow_threads(|| {
            gamma_rhf_driver(&cell, &prep, exx, omega_bohr, &jk_choice, &config, want)
        })
        .map_err(make_err)?;
    let (naux, naux_kept, n_dropped) = match run.rsgdf_counts {
        Some((n, k, d)) => (Some(n), Some(k), Some(d)),
        None => (None, None, None),
    };
    Ok(PyGammaRhfResult {
        energy: run.scf.energy,
        converged: run.scf.converged,
        iterations: run.scf.iterations,
        e_nuc: run.enn,
        madelung: run.madelung,
        exxdiv: match exx {
            ferric_pbc::ExxDiv::Ewald => "ewald".into(),
            ferric_pbc::ExxDiv::None => "none".into(),
        },
        omega: run.omega_bohr * ANGSTROM_TO_BOHR,
        nao,
        n_g_half: run.n_g_half,
        jk: if aux_name.is_some() { "rsgdf" } else { "dense" }.into(),
        auxbasis: aux_name,
        naux,
        naux_kept,
        n_dropped,
        overlap_data: run.s,
        scf_data: run.scf,
        timings_data: run.timings,
        gradient_data: run.derivs.gradient,
        stress_data: run.derivs.stress,
    })
}
