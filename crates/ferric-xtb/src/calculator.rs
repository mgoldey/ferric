//! Safe RAII wrapper around xtb's C API.
//!
//! Compiled only under the `xtb` feature.
//!
//! # Unit convention (VERIFIED, not assumed)
//!
//! Both sides of this boundary use atomic units, so the coordinate transfer is a
//! straight copy with **no conversion factor**:
//!
//! - `ferric_core::mol::Molecule` stores `Atom::{x, y, zpos}` in **Bohr**
//!   (`mol.rs` converts from Angstrom with `ANGSTROM_TO_BOHR` at XYZ-parse time;
//!   every downstream consumer -- integrals, nuclear repulsion -- reads Bohr).
//! - `xtb_newMolecule` documents "quantities in Bohr"; `xtb_getEnergy` returns
//!   Hartree and `xtb_getGradient` returns Hartree/Bohr (`xtb.h`).
//!
//! The gradient is returned as an `[natoms, 3]` array of dE/dR in Hartree/Bohr,
//! matching the sign and layout of `ferric_scf::gradient::rhf_gradient`.
//!
//! # Error handling
//!
//! xtb reports failures by queueing them on the environment object rather than
//! returning a status code. Every FFI call here is followed by
//! [`XtbEnv::check`], which drains the queue into a typed [`FerricError`]. No
//! call site panics on a library error.
//!
//! # Threading -- libxtb is NOT thread-safe (MEASURED)
//!
//! [`XtbCalculator`] is not `Send`/`Sync` (it holds raw pointers), but that is
//! not sufficient: **libxtb carries process-global mutable state, so two
//! calculators running concurrently in the same process corrupt each other even
//! though each owns private handles.**
//!
//! This was measured, not assumed: this crate's own test suite passes 8/8 under
//! `--test-threads=1` and fails 3/8 under the default parallel harness, on
//! energies and gradients that are exactly correct when run serially. That is
//! why [`tests/xtb_singlepoint.rs`] must be run serially.
//!
//! Consequences for callers:
//!
//! - Do **not** drive xTB from a `rayon` parallel iterator inside one process.
//! - To screen conformers in parallel, use process-level parallelism (one
//!   molecule per process), which is also how the repo's other throughput work
//!   is organised (see the "single-job threading no-op" convention).
//! - Set `OMP_NUM_THREADS=1` to keep per-process threading predictable,
//!   matching the repo's `OPENBLAS_NUM_THREADS=1` convention.

use crate::ffi;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ndarray::Array2;
use std::os::raw::{c_char, c_int};

/// Which GFN Hamiltonian to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum XtbMethod {
    /// GFN1-xTB (Grimme et al., JCTC 13, 1989 (2017)).
    Gfn1,
    /// GFN2-xTB (Bannwarth/Ehlert/Grimme, JCTC 15, 1652 (2019)). Default.
    #[default]
    Gfn2,
    /// GFN-FF force field (Spicher/Grimme, Angew. Chem. 59, 15665 (2020)).
    /// No self-consistent charges; cheapest, least accurate.
    ///
    /// NOTE: loading GFN-FF makes libxtb write a `gfnff_topo` topology file into
    /// the **current working directory** as a side effect. There is no C-API knob
    /// to suppress it; run from a scratch directory if that matters.
    GfnFf,
}

impl XtbMethod {
    /// Parse a config string. Unknown values are a hard error, never a silent
    /// default (repo config-honesty convention).
    pub fn parse_config_str(s: &str) -> Result<Self, FerricError> {
        match s.trim().to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
            "gfn1" | "gfn1-xtb" => Ok(Self::Gfn1),
            "gfn2" | "gfn2-xtb" => Ok(Self::Gfn2),
            "gfnff" | "gfn-ff" => Ok(Self::GfnFf),
            other => Err(FerricError::General(format!(
                "unknown xtb method '{other}' (expected one of: gfn1, gfn2, gfn-ff)"
            ))),
        }
    }

    /// Whether this method solves self-consistent charges (GFN1/GFN2) as
    /// opposed to being a plain force field (GFN-FF).
    ///
    /// GFN-FF has no SCC loop, so `xtb_setMaxIter` / `xtb_setElectronicTemp`
    /// are errors for it, and it produces no partial charges.
    pub fn is_self_consistent(self) -> bool {
        matches!(self, Self::Gfn1 | Self::Gfn2)
    }

    fn name(self) -> &'static str {
        match self {
            Self::Gfn1 => "GFN1-xTB",
            Self::Gfn2 => "GFN2-xTB",
            Self::GfnFf => "GFN-FF",
        }
    }
}

/// Tunable knobs for a single-point evaluation.
#[derive(Debug, Clone, Copy)]
pub struct XtbConfig {
    pub method: XtbMethod,
    /// Numerical accuracy, xtb's range is 1000.0 (loose) to 0.0001 (tight).
    /// xtb's own default is 1.0.
    pub accuracy: f64,
    /// Max SCC iterations (ignored by GFN-FF, which is not self-consistent).
    pub max_iter: u32,
    /// Electronic (Fermi-smearing) temperature in Kelvin. xtb's default is 300.
    pub electronic_temp: f64,
    /// xtb console verbosity: 0 = muted (default here), 1 = minimal, 2 = full.
    pub verbosity: i32,
}

impl Default for XtbConfig {
    fn default() -> Self {
        Self {
            method: XtbMethod::default(),
            accuracy: 1.0,
            max_iter: 250,
            electronic_temp: 300.0,
            verbosity: 0,
        }
    }
}

/// Result of an xTB single point.
#[derive(Debug, Clone)]
pub struct XtbResult {
    /// Total energy in Hartree.
    pub energy: f64,
    /// Nuclear gradient dE/dR in Hartree/Bohr, shape `[natoms, 3]`.
    pub gradient: Array2<f64>,
    /// Partial charges in e, length `natoms`. `None` for GFN-FF, which is a
    /// force field and computes no self-consistent charges.
    pub charges: Option<Vec<f64>>,
}

/// RAII handle for `xtb_TEnvironment`.
struct XtbEnv(ffi::XtbEnvironment);

/// Ensure `XTBPATH` is set before the first environment is constructed.
///
/// libxtb reads `XTBPATH` in `xtb_newEnvironment` to locate its GFN parameter
/// files (`param_gfn2-xtb.txt` etc.); without it, loading a Hamiltonian fails
/// with "Parameter file ... not found". We default it to the prefix libxtb was
/// linked from (baked in by `build.rs`), so callers get a working calculator
/// without exporting anything.
///
/// A user-supplied `XTBPATH` is never overwritten. Runs exactly once, before
/// any environment exists, which keeps the `set_var` off the path of any
/// concurrent reader.
fn ensure_param_path() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if std::env::var_os("XTBPATH").is_some() {
            return; // caller knows better; respect it.
        }
        let dir = env!("FERRIC_XTB_PARAM_DIR");
        if std::path::Path::new(dir).is_dir() {
            // SAFETY: `set_var` is unsafe from Rust 2024 because it races with
            // concurrent getenv in other threads. This runs inside a `Once`
            // before this crate has created any xtb environment, and only when
            // the variable is currently unset.
            unsafe { std::env::set_var("XTBPATH", dir) };
        }
    });
}

impl XtbEnv {
    fn new() -> Result<Self, FerricError> {
        ensure_param_path();
        // SAFETY: no arguments; xtb returns NULL on allocation failure.
        let raw = unsafe { ffi::xtb_newEnvironment() };
        if raw.is_null() {
            return Err(FerricError::General(
                "xtb_newEnvironment returned NULL".to_string(),
            ));
        }
        Ok(Self(raw))
    }

    fn set_verbosity(&self, verbosity: i32) {
        // SAFETY: self.0 is a live environment handle.
        unsafe { ffi::xtb_setVerbosity(self.0, verbosity as c_int) };
    }

    /// Drain the environment's error queue. Returns `Ok(())` when no error is
    /// pending, otherwise a [`FerricError`] carrying xtb's own message.
    ///
    /// Must be called after EVERY xtb call that takes an environment: the API
    /// returns `void` and reports failures only here.
    fn check(&self, context: &str) -> Result<(), FerricError> {
        // SAFETY: self.0 is a live environment handle.
        let nerr = unsafe { ffi::xtb_checkEnvironment(self.0) };
        if nerr == 0 {
            return Ok(());
        }
        const BUFSIZE: usize = 4096;
        let mut buf = vec![0u8; BUFSIZE];
        let size = BUFSIZE as c_int;
        // SAFETY: buf is BUFSIZE bytes and `size` reports exactly that, so xtb
        // cannot overrun it. This also empties the error stack.
        unsafe {
            ffi::xtb_getError(self.0, buf.as_mut_ptr() as *mut c_char, &size);
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(BUFSIZE);
        let msg = String::from_utf8_lossy(&buf[..end]).trim().to_string();
        Err(FerricError::General(format!(
            "xtb error during {context}: {}",
            if msg.is_empty() {
                format!("<{nerr} error(s), no message>")
            } else {
                msg
            }
        )))
    }
}

impl Drop for XtbEnv {
    fn drop(&mut self) {
        // SAFETY: takes a pointer to the handle and NULLs it; idempotent in xtb.
        unsafe { ffi::xtb_delEnvironment(&mut self.0) };
    }
}

/// RAII handle for `xtb_TMolecule`.
struct XtbMol(ffi::XtbMolecule);

impl Drop for XtbMol {
    fn drop(&mut self) {
        // SAFETY: live molecule handle; xtb NULLs it.
        unsafe { ffi::xtb_delMolecule(&mut self.0) };
    }
}

/// RAII handle for `xtb_TCalculator`.
struct XtbCalc(ffi::XtbCalculator);

impl Drop for XtbCalc {
    fn drop(&mut self) {
        // SAFETY: live calculator handle; xtb NULLs it.
        unsafe { ffi::xtb_delCalculator(&mut self.0) };
    }
}

/// RAII handle for `xtb_TResults`.
struct XtbRes(ffi::XtbResults);

impl Drop for XtbRes {
    fn drop(&mut self) {
        // SAFETY: live results handle; xtb NULLs it.
        unsafe { ffi::xtb_delResults(&mut self.0) };
    }
}

/// A GFN-xTB calculator bound to one molecular geometry.
///
/// Holds the xtb environment, molecule, calculator and results handles, each
/// freed in `Drop` (declaration order below is also drop order: results,
/// calculator, molecule, then environment last, since the others were created
/// against it).
///
/// Not `Send`/`Sync` by construction -- use one calculator per thread.
pub struct XtbCalculator {
    // Field order matters: Rust drops fields in declaration order.
    res: XtbRes,
    calc: XtbCalc,
    mol: XtbMol,
    env: XtbEnv,
    natoms: usize,
    config: XtbConfig,
}

impl XtbCalculator {
    /// Build a calculator for `mol` using `config`.
    ///
    /// Coordinates are passed through in **Bohr** with no conversion (see the
    /// module doc). Total charge comes from `mol.charge`, and the number of
    /// unpaired electrons from `mol.multiplicity - 1`.
    ///
    /// # Errors
    ///
    /// - Empty molecule, or ghost/ECP atoms (xTB is an all-electron valence
    ///   method with its own minimal basis; neither concept transfers, so they
    ///   are rejected rather than silently mis-modelled).
    /// - Any error queued on the xtb environment (unsupported element, SCC
    ///   failure, bad parameters).
    pub fn new(mol: &Molecule, config: XtbConfig) -> Result<Self, FerricError> {
        let natoms = mol.atoms.len();
        if natoms == 0 {
            return Err(FerricError::General(
                "xtb: molecule has no atoms".to_string(),
            ));
        }

        // xTB carries its own minimal valence basis and its own effective core
        // treatment. A ferric ghost atom (basis-only, zero nuclear charge) and a
        // ferric ECP atom (reduced z) have no xTB counterpart; passing either
        // would silently compute a different system than the caller asked for.
        if let Some(a) = mol.atoms.iter().find(|a| a.ghost) {
            return Err(FerricError::General(format!(
                "xtb: ghost atom '{}' is not supported (xTB has no basis-only centers); \
                 strip ghosts before screening",
                a.symbol
            )));
        }
        if let Some(a) = mol.atoms.iter().find(|a| a.n_core_ecp != 0) {
            return Err(FerricError::General(format!(
                "xtb: atom '{}' carries an ECP (n_core_ecp = {}); xTB uses its own \
                 effective-core parameterization and must be given all-electron atomic numbers",
                a.symbol, a.n_core_ecp
            )));
        }

        let numbers: Vec<c_int> = mol.atoms.iter().map(|a| a.z as c_int).collect();
        // Row-major [natoms][3] in Bohr -- ferric's native storage unit.
        let mut positions = Vec::with_capacity(3 * natoms);
        for a in &mol.atoms {
            positions.push(a.x);
            positions.push(a.y);
            positions.push(a.zpos);
        }

        let charge = mol.charge as f64;
        // xtb's `uhf` is the number of unpaired electrons (2S), i.e. multiplicity - 1.
        let uhf = (mol.multiplicity as c_int) - 1;
        if uhf < 0 {
            return Err(FerricError::General(format!(
                "xtb: multiplicity {} is invalid (must be >= 1)",
                mol.multiplicity
            )));
        }

        let env = XtbEnv::new()?;
        env.set_verbosity(config.verbosity);

        let n_c = natoms as c_int;
        // SAFETY: `numbers` and `positions` are natoms and 3*natoms long
        // respectively, matching `n_c`; lattice/periodic are NULL for a
        // non-periodic system, which xtb accepts. All pointers outlive the call.
        let mol_raw = unsafe {
            ffi::xtb_newMolecule(
                env.0,
                &n_c,
                numbers.as_ptr(),
                positions.as_ptr(),
                &charge,
                &uhf,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        env.check("xtb_newMolecule")?;
        if mol_raw.is_null() {
            return Err(FerricError::General(
                "xtb_newMolecule returned NULL".to_string(),
            ));
        }
        let mol_h = XtbMol(mol_raw);

        // SAFETY: no arguments.
        let calc_raw = unsafe { ffi::xtb_newCalculator() };
        if calc_raw.is_null() {
            return Err(FerricError::General(
                "xtb_newCalculator returned NULL".to_string(),
            ));
        }
        let calc = XtbCalc(calc_raw);

        // Load the requested Hamiltonian. The trailing `filename` argument is an
        // optional parameter-file override; NULL selects the built-in parameters.
        // SAFETY: env/mol/calc are live handles; NULL filename is documented.
        unsafe {
            match config.method {
                XtbMethod::Gfn1 => ffi::xtb_loadGFN1xTB(env.0, mol_h.0, calc.0, std::ptr::null_mut()),
                XtbMethod::Gfn2 => ffi::xtb_loadGFN2xTB(env.0, mol_h.0, calc.0, std::ptr::null_mut()),
                XtbMethod::GfnFf => ffi::xtb_loadGFNFF(env.0, mol_h.0, calc.0, std::ptr::null_mut()),
            }
        }
        env.check(config.method.name())?;

        // SAFETY: live env/calc handles; scalars passed by value.
        unsafe { ffi::xtb_setAccuracy(env.0, calc.0, config.accuracy) };
        env.check("xtb_setAccuracy")?;

        // GFN-FF is a force field: not self-consistent, so it has no SCC
        // iteration count and no electronic temperature. Calling these on it is
        // an error inside xtb ("Cannot set iterations for non-iterative
        // method"), so skip them rather than queue a spurious failure.
        if config.method.is_self_consistent() {
            // SAFETY: live env/calc handles; scalars passed by value.
            unsafe {
                ffi::xtb_setMaxIter(env.0, calc.0, config.max_iter as c_int);
                ffi::xtb_setElectronicTemp(env.0, calc.0, config.electronic_temp);
            }
            env.check("calculator configuration")?;
        }

        // SAFETY: no arguments.
        let res_raw = unsafe { ffi::xtb_newResults() };
        if res_raw.is_null() {
            return Err(FerricError::General(
                "xtb_newResults returned NULL".to_string(),
            ));
        }

        Ok(Self {
            res: XtbRes(res_raw),
            calc,
            mol: mol_h,
            env,
            natoms,
            config,
        })
    }

    /// Convenience constructor using [`XtbConfig::default`] (GFN2-xTB).
    pub fn new_gfn2(mol: &Molecule) -> Result<Self, FerricError> {
        Self::new(mol, XtbConfig::default())
    }

    /// The method this calculator was built with.
    pub fn method(&self) -> XtbMethod {
        self.config.method
    }

    /// Run a single point and return energy, gradient and partial charges.
    ///
    /// Energy is in Hartree; gradient is dE/dR in Hartree/Bohr with shape
    /// `[natoms, 3]`; charges are in e.
    pub fn singlepoint(&mut self) -> Result<XtbResult, FerricError> {
        // SAFETY: all four handles are live and were constructed against `env`.
        unsafe { ffi::xtb_singlepoint(self.env.0, self.mol.0, self.calc.0, self.res.0) };
        self.env.check("xtb_singlepoint")?;

        let mut energy = 0.0f64;
        // SAFETY: writes one f64.
        unsafe { ffi::xtb_getEnergy(self.env.0, self.res.0, &mut energy) };
        self.env.check("xtb_getEnergy")?;

        let mut grad = vec![0.0f64; 3 * self.natoms];
        // SAFETY: buffer is exactly 3*natoms, the documented [natoms][3] size.
        unsafe { ffi::xtb_getGradient(self.env.0, self.res.0, grad.as_mut_ptr()) };
        self.env.check("xtb_getGradient")?;

        // Partial charges exist only for the self-consistent Hamiltonians;
        // asking GFN-FF for them is an error inside xtb.
        let charges = if self.config.method.is_self_consistent() {
            let mut q = vec![0.0f64; self.natoms];
            // SAFETY: buffer is exactly natoms, the documented size.
            unsafe { ffi::xtb_getCharges(self.env.0, self.res.0, q.as_mut_ptr()) };
            self.env.check("xtb_getCharges")?;
            Some(q)
        } else {
            None
        };

        let gradient = Array2::from_shape_vec((self.natoms, 3), grad).map_err(|e| {
            FerricError::General(format!("xtb gradient shape error: {e}"))
        })?;

        Ok(XtbResult {
            energy,
            gradient,
            charges,
        })
    }

    /// Energy only, in Hartree -- the conformer-screening entry point.
    pub fn energy(&mut self) -> Result<f64, FerricError> {
        Ok(self.singlepoint()?.energy)
    }
}

/// One-shot single point on `mol`. Builds a calculator, runs it, drops it.
pub fn xtb_singlepoint(mol: &Molecule, config: XtbConfig) -> Result<XtbResult, FerricError> {
    XtbCalculator::new(mol, config)?.singlepoint()
}

/// One-shot GFN2-xTB energy in Hartree.
pub fn gfn2_energy(mol: &Molecule) -> Result<f64, FerricError> {
    XtbCalculator::new_gfn2(mol)?.energy()
}
