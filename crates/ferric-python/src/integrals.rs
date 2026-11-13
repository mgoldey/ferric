//! Raw integral access: moved verbatim from `lib.rs`.

use super::*;

// ── Raw integral access (for Python-side method prototyping) ──

/// Raw 3-center Coulomb integrals (P|μν) as a numpy array of shape
/// (naux, n_bf, n_bf). Undressed — pair with `compute_metric_2c` and a
/// V^{-1/2} (or Cholesky L^{-1}) factor on the Python side to form RI
/// B-tensors. ECPs are applied to the molecule first, matching `run_rhf`.
#[pyfunction]
pub(crate) fn compute_eri3<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    aux_basis_set: &PyBasisSet,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let obs = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&emol, &aux_basis_set.inner).map_err(make_err)?;
    let eri3 = ferric_integrals::threeindex::eri3_tensor(Operator::coulomb(), &obs, &dfbs)
        .map_err(make_err)?;
    Ok(PyArray3::from_array(py, &eri3))
}

/// Blocked MO-basis 3-center Coulomb integrals (P|pq), shape
/// (naux, n_left, n_right), for arbitrary coefficient matrices `c_left` /
/// `c_right` of shape (n_bf, n_*) — e.g. localized occupieds × canonical
/// virtuals. Unlike `compute_eri3`, the naux·n_bf² AO tensor is NEVER
/// materialized: raw aux-row blocks are generated under `memory_budget_gb`
/// and transformed immediately (`eri3_mo_ov_blocked`, bit-identical to the
/// dense path). This is what makes C32+/cc-pVDZ domain prototyping possible
/// from Python — the dense AO tensor is ~13 GB there. ECPs applied first,
/// matching `run_rhf`.
/// Resolve a low-level operator spec: name + RAW Bohr-native parameters
/// (unlike the run_* wrappers, which take Angstrom units). omega in Bohr^-1,
/// r0 in Bohr. "terfc"/"terf" with omega=None use the Dutoi-linked
/// omega = 1/(r0*sqrt2); an explicit omega decouples them (tables cover
/// (omega*r0)^2 <= 80; needs FERRIC_TERF_TABLE_DIR).
pub(crate) fn resolve_operator(
    name: &str,
    omega: Option<f64>,
    r0: Option<f64>,
) -> PyResult<Operator> {
    let need = |o: Option<f64>, what: &str| {
        o.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("operator '{name}' requires {what}"))
        })
    };
    Ok(match name {
        "coulomb" => Operator::coulomb(),
        "erf" => Operator::erf(need(omega, "omega (Bohr^-1)")?),
        "erfc" => Operator::erfc(need(omega, "omega (Bohr^-1)")?),
        "terfc" => match omega {
            Some(w) => Operator::terfc_with_omega(need(r0, "r0 (Bohr)")?, w),
            None => Operator::terfc(need(r0, "r0 (Bohr)")?),
        },
        "terf" => match omega {
            Some(w) => Operator::terf_with_omega(need(r0, "r0 (Bohr)")?, w),
            None => Operator::terf(need(r0, "r0 (Bohr)")?),
        },
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown operator '{other}'; expected coulomb|erf|erfc|terf|terfc"
            )))
        }
    })
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, aux_basis_set, c_left, c_right, memory_budget_gb=2.0, operator="coulomb", omega=None, r0=None))]
pub(crate) fn compute_eri3_mo<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    aux_basis_set: &PyBasisSet,
    c_left: numpy::PyReadonlyArray2<f64>,
    c_right: numpy::PyReadonlyArray2<f64>,
    memory_budget_gb: f64,
    operator: &str,
    omega: Option<f64>,
    r0: Option<f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let obs = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&emol, &aux_basis_set.inner).map_err(make_err)?;
    let n_bf = obs.nbasis();
    let cl = c_left.as_array().to_owned();
    let cr = c_right.as_array().to_owned();
    if cl.nrows() != n_bf || cr.nrows() != n_bf {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "coefficient rows must equal n_bf={n_bf} (got c_left {}x{}, c_right {}x{})",
            cl.nrows(),
            cl.ncols(),
            cr.nrows(),
            cr.ncols()
        )));
    }
    let budget = (memory_budget_gb.max(0.1) * (1u64 << 30) as f64) as usize;
    let op = resolve_operator(operator, omega, r0)?;
    let mo = ferric_mp2::rimp2::eri3_mo_ov_blocked(op, &obs, &dfbs, &cl, &cr, budget)
        .map_err(make_err)?;
    Ok(PyArray3::from_owned_array(py, mo))
}

/// Result of Boys localization: localized coefficients, orbital centers,
/// and convergence info.
#[pyclass(name = "BoysResult")]
pub(crate) struct PyBoysResult {
    pub(crate) c_loc_data: ndarray::Array2<f64>,
    pub(crate) centers_data: ndarray::Array2<f64>,
    #[pyo3(get)]
    pub(crate) converged: bool,
    #[pyo3(get)]
    pub(crate) iterations: usize,
}

#[pymethods]
impl PyBoysResult {
    /// Localized MO coefficients, shape (n_bf, n_orb). Orthonormal (a unitary
    /// rotation of the input orbitals).
    fn c_loc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.c_loc_data)
    }
    /// Boys centers <i|r|i>, shape (n_orb, 3), in Bohr.
    fn centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.centers_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "BoysResult(converged={}, iterations={})",
            self.converged, self.iterations,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "Boys Localization: {} orbitals, converged={}, {} iterations",
            self.c_loc_data.ncols(),
            self.converged,
            self.iterations,
        )
    }
}

/// Boys (Foster-Boys) localization of the given orbitals (columns of
/// `c_orbs`, shape (n_bf, n_orb) — typically the occupied block of
/// `mo_coefficients()`). Same Jacobi-sweep implementation the Rust probes
/// use. Returns BoysResult{c_loc(), centers(), converged, iterations}.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, c_orbs, max_iter=200))]
pub(crate) fn boys_localize(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    c_orbs: numpy::PyReadonlyArray2<f64>,
    max_iter: usize,
) -> PyResult<PyBoysResult> {
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let obs = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
    let c = c_orbs.as_array().to_owned();
    if c.nrows() != obs.nbasis() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "c_orbs rows must equal n_bf={} (got {})",
            obs.nbasis(),
            c.nrows()
        )));
    }
    let dip = ferric_integrals::oneelectron::dipole(&obs, [0.0, 0.0, 0.0]).map_err(make_err)?;
    let res = ferric_mp2::boys::boys_localize(&c, &dip, max_iter);
    Ok(PyBoysResult {
        c_loc_data: res.c_loc,
        centers_data: res.centers,
        converged: res.converged,
        iterations: res.iterations,
    })
}

/// Report the CUDA backend's state without touching a device unless
/// `FERRIC_GPU` asks for one (mode defaults to `off`).
#[pyfunction]
pub(crate) fn gpu_status(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    use ferric_core::gpu::{gpu_compiled, settings, status, GpuStatus};
    // The first call may probe the device (FERRIC_GPU=auto|on); release the GIL.
    let (mode, st) = py.allow_threads(|| (settings().mode.to_string(), status()));
    let d = pyo3::types::PyDict::new(py);
    d.set_item("compiled", gpu_compiled())?;
    d.set_item("mode", mode)?;
    let s = ferric_core::gpu::settings();
    d.set_item("precision", s.precision.to_string())?;
    let kernels: Vec<&str> = s.mixed_kernels.iter().map(|k| k.name()).collect();
    d.set_item("mixed_kernels", kernels)?;
    match st {
        GpuStatus::NotCompiled => {
            d.set_item("status", "not_compiled")?;
            d.set_item("reason", py.None())?;
            d.set_item("device", py.None())?;
        }
        GpuStatus::Unavailable { reason } => {
            d.set_item("status", "unavailable")?;
            d.set_item("reason", reason)?;
            d.set_item("device", py.None())?;
        }
        GpuStatus::Ready(info) => {
            d.set_item("status", "ready")?;
            d.set_item("reason", py.None())?;
            let dev = pyo3::types::PyDict::new(py);
            dev.set_item("ordinal", info.ordinal)?;
            dev.set_item("name", &info.name)?;
            dev.set_item("cc", format!("{}.{}", info.cc_major, info.cc_minor))?;
            dev.set_item("free_bytes", info.free_bytes)?;
            dev.set_item("total_bytes", info.total_bytes)?;
            d.set_item("device", dev)?;
        }
    }
    Ok(d)
}

/// A non-negative integer GPU key (`device`, `min_flops`): a negative value is
/// a `ValueError` naming the key rather than pyo3's bare `OverflowError`.
pub(crate) fn non_negative_gpu_key(key: &str, v: Option<i64>) -> PyResult<Option<usize>> {
    v.map(|n| {
        usize::try_from(n).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "[gpu] {key}: must be a non-negative integer, got {n}"
            ))
        })
    })
    .transpose()
}

/// A `mixed_kernels` kwarg as a kernel set: each list item is one name (a
/// comma inside an item is refused), unknown names and an empty list are
/// refused with the valid list. `section` is the error prefix.
pub(crate) fn parse_kernel_list(
    section: &str,
    ks: Vec<String>,
) -> PyResult<ferric_core::gpu::MixedKernelSet> {
    let bad = |e: String| {
        pyo3::exceptions::PyValueError::new_err(format!("{section} mixed_kernels: {e}"))
    };
    if let Some(item) = ks.iter().find(|k| k.contains(',')) {
        return Err(bad(format!(
            "kernel names are separate list items, got {item:?}"
        )));
    }
    ks.join(",")
        .parse::<ferric_core::gpu::MixedKernelSet>()
        .map_err(bad)
}

/// Choose the mixed-precision kernels for this process (the Python side of
/// `[precision]`), independently of the device: `mode` is `"f64"` (the
/// default) or `"mixed"`, `mixed_kernels` a list of kernel names (default: the
/// kernels this build ships). Strict: an unknown value is a `ValueError`.
///
/// The request is staged, not installed: only the next `configure_gpu(...)`
/// installs it, together with the device settings (call this first; a
/// `configure_gpu` `precision` / `mixed_kernels` alias that disagrees with it
/// is a `ValueError`). Until then every calculation raises `RuntimeError`
/// ("configure_precision staged a request; call configure_gpu() to install
/// it"). Call `configure_gpu()` with no
/// arguments to install and check the combination now: `mode = "mixed"` with
/// a device kernel needs a GPU mode, and with the GPU off only kernels with a
/// CPU mixed path are accepted. After the settings are installed or read this
/// is a `RuntimeError`. Every result then reports what ran in its `precision`
/// record. Returns the staged request, `{"mode", "mixed_kernels"}` (`None`
/// for a key not given).
#[pyfunction]
#[pyo3(signature = (mode=None, mixed_kernels=None))]
pub(crate) fn configure_precision<'py>(
    py: Python<'py>,
    mode: Option<&str>,
    mixed_kernels: Option<Vec<String>>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use ferric_core::gpu::{Precision, PrecisionExplicit};
    let request = PrecisionExplicit {
        mode: mode
            .map(|s| {
                s.parse::<Precision>().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("[precision] mode: {e}"))
                })
            })
            .transpose()?,
        mixed_kernels: mixed_kernels
            .map(|ks| parse_kernel_list("[precision]", ks))
            .transpose()?,
    };
    ferric_core::gpu::stage_precision(request).map_err(|e| {
        if e.starts_with(ferric_core::gpu::SETTINGS_ALREADY_INSTALLED) {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{e}; call configure_precision before configure_gpu and before any calculation"
            ))
        } else {
            pyo3::exceptions::PyValueError::new_err(e)
        }
    })?;
    let staged = ferric_core::gpu::staged_precision().unwrap_or_default();
    let d = pyo3::types::PyDict::new(py);
    d.set_item("mode", staged.mode.map(|p| p.to_string()))?;
    d.set_item(
        "mixed_kernels",
        staged
            .mixed_kernels
            .map(|ks| ks.iter().map(|k| k.name()).collect::<Vec<_>>()),
    )?;
    Ok(d)
}

/// Install the CUDA backend for this process from keyword arguments (the
/// Python side of `[gpu]`): `preset` is one word, the rest are the fine-grained
/// keys. Resolved through the same machinery as the CLI (explicit > env >
/// default); a refusal is a `ValueError` naming the keys. Settings are
/// process-global: the call must precede the first read of the GPU settings
/// (`gpu_status()` or any GPU path), and a second call must repeat identical
/// settings, otherwise `RuntimeError`. A request staged by
/// `configure_precision` is installed with these settings. `precision` and
/// `mixed_kernels` are deprecated aliases of `configure_precision(mode=...,
/// mixed_kernels=...)`. Returns `gpu_status()`.
#[pyfunction]
#[pyo3(signature = (preset=None, mode=None, precision=None, mixed_kernels=None, device=None, memory_gb=None, min_flops=None))]
pub(crate) fn configure_gpu<'py>(
    py: Python<'py>,
    preset: Option<&str>,
    mode: Option<&str>,
    precision: Option<&str>,
    mixed_kernels: Option<Vec<String>>,
    device: Option<i64>,
    memory_gb: Option<f64>,
    min_flops: Option<i64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use ferric_core::gpu::{GpuMode, GpuPreset, GpuSettingsExplicit, Precision};
    let bad =
        |k: &str, e: String| pyo3::exceptions::PyValueError::new_err(format!("[gpu] {k}: {e}"));
    let explicit = GpuSettingsExplicit {
        // Python has no command line: the `--gpu` flag slot stays empty.
        cli_preset: None,
        preset: preset
            .map(|s| s.parse::<GpuPreset>().map_err(|e| bad("preset", e)))
            .transpose()?,
        mode: mode
            .map(|s| s.parse::<GpuMode>().map_err(|e| bad("mode", e)))
            .transpose()?,
        device: non_negative_gpu_key("device", device)?,
        memory_gb,
        min_flops: non_negative_gpu_key("min_flops", min_flops)?,
        precision: precision
            .map(|s| s.parse::<Precision>().map_err(|e| bad("precision", e)))
            .transpose()?,
        mixed_kernels: mixed_kernels
            .map(|ks| parse_kernel_list("[gpu]", ks))
            .transpose()?,
        precision_section: Default::default(),
    };
    // The device probe may take a while; release the GIL.
    py.allow_threads(|| ferric_core::gpu::install(explicit).map(|_| ()))
        .map_err(|e| {
            if e.starts_with(ferric_core::gpu::SETTINGS_ALREADY_INSTALLED) {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "{e}; GPU settings are process-global: call configure_gpu once, before \
                     gpu_status() or any GPU work, and repeat identical settings if called again"
                ))
            } else {
                pyo3::exceptions::PyValueError::new_err(e)
            }
        })?;
    gpu_status(py)
}

/// Shell geometry of `basis_set` on `mol` (works for orbital AND auxiliary
/// sets): returns (centers, first_function_offsets, n_functions) with shapes
/// ((n_shells, 3) in Bohr, (n_shells,), (n_shells,)). Enough to build
/// geometric fitting domains from Python (aux functions within r_cut of an
/// orbital center).
#[pyfunction]
pub(crate) fn shell_info<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let centers = prep.shell_centers();
    let mut c = ndarray::Array2::<f64>::zeros((centers.len(), 3));
    for (i, xyz) in centers.iter().enumerate() {
        c[(i, 0)] = xyz[0];
        c[(i, 1)] = xyz[1];
        c[(i, 2)] = xyz[2];
    }
    let offs: Vec<i64> = prep.shell_offsets().iter().map(|&x| x as i64).collect();
    let dims: Vec<i64> = prep.shell_dims().iter().map(|&x| x as i64).collect();
    Ok((
        PyArray2::from_owned_array(py, c),
        PyArray1::from_vec(py, offs),
        PyArray1::from_vec(py, dims),
    ))
}

/// 2-center metric (P|w|Q) over the auxiliary basis, shape (naux, naux).
/// Default w = Coulomb; `operator`/`omega`/`r0` follow the same RAW
/// Bohr-native conventions as `compute_eri3_mo`.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, aux_basis_set, operator="coulomb", omega=None, r0=None))]
pub(crate) fn compute_metric_2c<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    aux_basis_set: &PyBasisSet,
    operator: &str,
    omega: Option<f64>,
    r0: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let dfbs = PreparedBasis::new(&emol, &aux_basis_set.inner).map_err(make_err)?;
    let op = resolve_operator(operator, omega, r0)?;
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, &dfbs).map_err(make_err)?;
    Ok(PyArray2::from_array(py, &v2c))
}

/// Entry point for the `ferric` console script (pyproject.toml
/// `[project.scripts]`) -- calls the SAME CLI logic the standalone
/// `ferric`/`ferric-cli` Rust binaries call (`ferric_cli::run`), compiled
/// into THIS `.so` rather than a second binary shipped in the wheel (see the
/// maturin project's own documented "both binary and library?" guidance:
/// shipping a second compiled binary would roughly double the wheel, since
/// ferric-cli would link OpenBLAS/libxc/libint2 independently rather than
/// reuse what this extension module already links).
///
/// Deliberately reconstructs argv from Python's `sys.argv`, NOT
/// `ferric_cli::main()` (which reads `std::env::args()` itself) -- measured,
/// not assumed, that those differ here: `std::env::args()` inside this
/// pyfunction returns THREE elements for a two-argument invocation
/// (`[python_interpreter_path, script_path, "--help"]`), not the two-element
/// `[script_path, "--help"]` a native binary's argv has. See
/// `ferric_cli::run`'s doc comment for the full story (an inline debug
/// print plus a real `strace -f -e trace=execve` on the wrapper script,
/// confirming the extra element comes from Python's own runtime, not from
/// the shebang or the wrapper script maturin/pip generates) and why
/// `sys.argv` -- not a hand-patched `std::env::args()` -- is the fix:
/// `sys.argv` is the shape Python itself commits to being stable.
///
/// `sys.executable`/`sys.argv[0]` (the wrapper script's own path) is used as
/// element 0, matching what a native binary's argv[0] would be -- ferric_cli
/// never actually reads element 0's VALUE (only `args.len()` and
/// `args[1..]`), so this only needs to be non-empty and present, not any
/// particular string.
///
/// Calls `std::process::exit()` on every error path and on `--help` --
/// deliberately NOT caught or converted to a Python exception here, because
/// that IS this command's correct CLI behavior (a shell script or
/// `subprocess` caller reads $?/returncode, not a Python traceback), and
/// pyo3 has no supported way to make `std::process::exit` merely raise
/// instead of terminating the process regardless.
///
/// `init_threading()`/the ctrlc handler already ran at `import ferric` (this
/// module's own init, above) -- `ferric_cli::run` calls `init_threading()`
/// again internally, which is idempotent (same reasoning as its own doc
/// comment: it is safe-by-default, not order-sensitive).
#[pyfunction]
pub(crate) fn _cli_main(py: Python<'_>) -> PyResult<()> {
    let argv: Vec<String> = py.import("sys")?.getattr("argv")?.extract()?;
    ferric_cli::run(argv);
    Ok(())
}

/// Which build of ferric is this?
///
/// Returns a dict with:
///
/// - ``version`` (str): the PEP 440 package version, same as
///   ``ferric.__version__``. A checkout carries a ``.devN`` version; only a
///   release wheel built from a ``v*`` tag has the plain version.
/// - ``commit`` (str): full git hash of the source tree the extension was
///   compiled from, or ``"unknown"`` when that tree was not a git checkout.
/// - ``dirty`` (bool or None): whether any tracked file differed from
///   ``commit`` (untracked files never count); ``None`` when unknown.
/// - ``profile`` (str): cargo build profile, ``"release"`` or ``"debug"``.
/// - ``libint_version`` (str): version of the libint2 headers the integral
///   shim was compiled against, or ``"unknown"``.
/// - ``libint_max_deriv_order`` (int): ``LIBINT2_MAX_DERIV_ORDER`` of that
///   libint2 (1 = gradients only, 2 = analytic Hessians too).
/// - ``libint_max_am`` (dict[str, list[int]]): max angular momentum per
///   integral class (``"eri"``, ``"eri3"``, ``"eri2"``, ``"onebody"``),
///   entry ``d`` for derivative order ``d``. Tells the PyPI wheel (small
///   libint) from a conda-forge 2.13.1 build, which share version and commit.
/// - ``libint_g12_max_am`` (int or None): ``LIBINT2_MAX_AM_R12kG12``.
/// - ``libint_linkage`` (str): ``"static"`` or ``"shared"`` libint2.
///
/// All values are fixed when the extension is compiled, so they describe the
/// loaded ``.so`` even when it is a symlink into some other checkout.
#[pyfunction]
pub(crate) fn build_info(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("version", ferric_build_info::VERSION)?;
    d.set_item("commit", ferric_build_info::COMMIT)?;
    d.set_item("dirty", ferric_build_info::DIRTY)?;
    d.set_item("profile", ferric_build_info::PROFILE)?;
    add_libint_info(&d)?;
    Ok(d.unbind())
}

/// Test hook (not public API): override the per-class libint2 capability table
/// ferric checks before building derivative engines
/// (`ferric_integrals::capability`), so a test can stand in for a libint2 that
/// lacks a class without swapping libraries. `table` is six rows (4-centre,
/// 3-centre bra, 2-centre bra, one-body, 3-centre ket, 2-centre ket) of three
/// derivative orders, `-1` = not generated; `None` restores the linked table.
/// Process-wide. Returns the linked table.
#[pyfunction]
#[pyo3(signature = (table=None))]
pub(crate) fn _set_libint_max_am_for_test(table: Option<Vec<Vec<i32>>>) -> PyResult<Vec<Vec<i32>>> {
    use ferric_integrals::capability::{linked_libint_am_table, set_libint_max_am_for_test};
    let parsed = match table {
        None => None,
        Some(rows) => {
            let mut t = [[-1; 3]; 6];
            if rows.len() != t.len() || rows.iter().any(|r| r.len() != 3) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "table must be 6 rows of 3 derivative orders",
                ));
            }
            for (dst, src) in t.iter_mut().zip(&rows) {
                dst.copy_from_slice(src);
            }
            Some(t)
        }
    };
    set_libint_max_am_for_test(parsed);
    Ok(linked_libint_am_table()
        .iter()
        .map(|r| r.to_vec())
        .collect())
}

/// The `libint_*` entries of [`build_info`].
pub(crate) fn add_libint_info(d: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
    d.set_item("libint_version", ferric_integrals::libint_version())?;
    d.set_item(
        "libint_max_deriv_order",
        ferric_integrals::libint_max_deriv_order(),
    )?;
    d.set_item("libint_max_am", ferric_integrals::libint_max_am())?;
    d.set_item("libint_g12_max_am", ferric_integrals::libint_g12_max_am())?;
    d.set_item("libint_linkage", ferric_integrals::libint_linkage())
}

/// `ferric.__build__`: `{"git_sha": str, "dirty": bool}`, or `None` when the
/// build had no git metadata (an undeterminable dirty flag also gives `None`;
/// never guess a bool). smeltery reads exactly this shape.
pub(crate) fn add_build_stamp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    if let (true, Some(dirty)) = (ferric_build_info::commit_known(), ferric_build_info::DIRTY) {
        let b = pyo3::types::PyDict::new(m.py());
        b.set_item("git_sha", ferric_build_info::COMMIT)?;
        b.set_item("dirty", dirty)?;
        m.add("__build__", b)
    } else {
        m.add("__build__", m.py().None())
    }
}

/// Register `__version__`, `__build__` and `build_info`. `m.add` also appends to
/// the module's `__all__`, which carries `__version__` through maturin's
/// generated `from .ferric import *`.
pub(crate) fn add_build_identity(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", ferric_build_info::VERSION)?;
    add_build_stamp(m)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(_set_libint_max_am_for_test, m)?)?;
    m.add_function(wrap_pyfunction!(set_blas_threads, m)?)?;
    m.add_function(wrap_pyfunction!(blas_threads, m)?)
}

/// Set the OpenBLAS thread count for serial sections (the per-iteration Fock
/// diagonalization, DIIS commutator and the other dense linear algebra that
/// runs outside ferric's parallel regions) for the rest of the process.
///
/// ``None`` returns to the default: ``FERRIC_BLAS_THREADS`` if set, else the
/// rayon thread count (``RAYON_NUM_THREADS``, default every logical CPU).
/// ``1`` makes results bit-identical across thread counts; any other value
/// lets OpenBLAS split reductions by thread count, which moves results at the
/// last bit (about 1e-14 relative). Parallel regions always run OpenBLAS at 1
/// thread unless ``OPENBLAS_NUM_THREADS`` was set before ``import ferric``.
#[pyfunction]
#[pyo3(signature = (n=None))]
pub(crate) fn set_blas_threads(n: Option<usize>) -> PyResult<()> {
    if n == Some(0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "set_blas_threads(0): use a positive thread count, or None for the default",
        ));
    }
    ferric_integrals::blas_threads::set_serial_blas_threads(n);
    Ok(())
}

/// The OpenBLAS thread count serial sections use now (see
/// :func:`set_blas_threads`): the request clamped to the OpenBLAS build's
/// ``MAX_THREADS``, i.e. what OpenBLAS actually runs.
#[pyfunction]
pub(crate) fn blas_threads() -> usize {
    ferric_integrals::blas_threads::opt_in_blas_threads()
}
