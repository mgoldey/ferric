//! Optional GFN-xTB semiempirical binding for ferric.
//!
//! Binds the [xtb](https://github.com/grimme-lab/xtb) library through its stable
//! C ABI (`xtb.h`) so ferric can screen many conformers with a cheap
//! semiempirical Hamiltonian and refine the survivors with DFT, rather than only
//! ever being the refinement step.
//!
//! # Feature gate
//!
//! This crate is inert unless the **`xtb`** cargo feature is enabled (it is OFF
//! by default). With the feature off there is no native dependency at all, so
//! the workspace still builds on a machine without libxtb; the API surface below
//! is simply absent, and [`xtb_available`] returns `false`.
//!
//! ```text
//! cargo build -p ferric-xtb --features xtb
//! ```
//!
//! `build.rs` looks for libxtb under `$HOME/.local` (the prefix ferric's other
//! native dependencies use), overridable with `XTB_PREFIX`. Because libxtb is
//! linked dynamically, the same prefix must be on the runtime loader path
//! (meson installs into a multiarch subdir on Ubuntu):
//!
//! ```text
//! LD_LIBRARY_PATH=$HOME/.local/lib/x86_64-linux-gnu \
//!   cargo test -p ferric-xtb --features xtb
//! ```
//!
//! The GFN parameter files are located via `XTBPATH`; this crate defaults that
//! to the build-time prefix's `share/xtb` if the caller has not set it, so no
//! environment setup is normally needed.
//!
//! ## Building libxtb
//!
//! xtb has no Ubuntu package and its CMake path fails on a `tblite::tblite`
//! alias-target collision, so build it with meson (its primary build system).
//! From an xtb source tree (v6.7.1 used here), with `gfortran` available:
//!
//! ```text
//! pip install --user meson
//! meson setup build --prefix=$HOME/.local --buildtype=release -Doptimization=2 \
//!   -Dlapack=openblas -Dtblite=disabled -Dcpcmx=disabled -Ddefault_library=shared
//! meson compile -C build
//! meson install -C build
//! ```
//!
//! `tblite`/`cpcmx` are optional alternate-backend/solvation features that the
//! GFN1/GFN2/GFN-FF C API does not need.
//!
//! ## `-Doptimization=2` is REQUIRED, not a preference
//!
//! Plain `--buildtype=release` compiles at `-O3`, and **gfortran 13.3
//! miscompiles xtb 6.7.1's GFN1/GFN2 SCF gradient at `-O3`**: the analytic
//! gradient comes out ~20x too large and points the wrong way, while the
//! energy stays correct. `-Doptimization=2` overrides that back to `-O2`,
//! which is verified correct. See the "Validation status" section below.
//!
//! # Units
//!
//! ferric and xtb agree on atomic units, so nothing is rescaled at the boundary:
//! coordinates in **Bohr** in, energy in **Hartree** out, gradient in
//! **Hartree/Bohr**. See [`calculator`] for the verification of that claim.
//!
//! # Validation status (2026-07-27, xtb 6.7.1 commit a59bca3)
//!
//! **Energies: validated.** GFN1, GFN2 and GFN-FF single-point energies for
//! water reproduce the xtb CLI binary's own printed values to <1e-8 Ha. This is
//! the conformer-screening use case, and it works.
//!
//! **Gradients: validated** (against a libxtb built at `-O2`; see the build
//! note above). The analytic gradient matches a central finite difference of
//! xtb's own energy on every component to <1e-6 Ha/Bohr
//! (`gfn2_gradient_matches_finite_difference`), and this binding reproduces the
//! CLI's own `gradient` file to <1e-10, so both the value and the FFI transfer
//! are correct. Forces and xTB geometry optimisation are supported.
//!
//! ## Previously-documented gradient defect: RESOLVED (was a `-O3` miscompile)
//!
//! Earlier revisions of this crate warned that gradients must not be trusted:
//! xtb returned an analytic gradient ~20x too large that disagreed with FD,
//! `xtb --opt` on H2 drove 0.75 -> 2.92 Ang while *raising* the energy, and
//! xtb's own `meson test` failed `unit - xtb:gfn1`, `gfn2` and `hessian`.
//!
//! That was **a gfortran 13.3 miscompilation of xtb at `-O3`**, not an xtb
//! source bug and not a binding bug. Rebuilding the same source at `-O2` fixes
//! all of it: xtb's own gfn1/gfn2/hessian unit tests pass, FD agrees with the
//! analytic gradient to ~7 significant figures, and H2 optimises to 0.776 Ang.
//! Only GFN1/GFN2 were affected -- GFN0 uses a separate gradient path and was
//! always correct. Explicitly ruled out by experiment: the bundled dftd4 and
//! multicharge subprojects (correctly pinned to v3.5.0/v0.2.0, not `HEAD`),
//! OpenBLAS threading, and the BLAS backend itself.
//!
//! Energies were never affected -- they were byte-identical between the `-O3`
//! and `-O2` builds.
//!
//! # Threading
//!
//! libxtb is **not thread-safe** (process-global state -- measured, see
//! [`calculator`]). Parallelise across processes, not threads.
//!
//! [`XtbCalculator`] is `!Send`/`!Sync` on purpose, and a `#[cfg(test)]`
//! compile-time guard in `calculator.rs` (`send_sync_guard`) fails the build
//! if that ever stops being true. That property is load-bearing, not
//! decorative:
//!
//! - **Never** add `unsafe impl Send`/`unsafe impl Sync` for
//!   [`XtbCalculator`] or its handle newtypes -- the process-global state
//!   inside libxtb means two calculators driven concurrently corrupt each
//!   other even though each owns private handles.
//! - **Never** drive an [`XtbCalculator`] from inside a `rayon` parallel
//!   iterator (`par_iter`, `join`, a thread pool, ...) in the same process.
//!   Screen conformers with process-level parallelism instead -- one
//!   molecule per process, `OMP_NUM_THREADS=1` per process -- matching how
//!   the rest of the repo's throughput work is organised.
//! - Drive libxtb from **one thread at a time**, for the lifetime of the
//!   process. A single thread creating and dropping many `XtbCalculator`s in
//!   sequence is fine; handing one to another thread, or holding two live at
//!   once on different threads, is not.
//!
//! # Example
//!
//! ```ignore
//! use ferric_core::mol::Molecule;
//! use ferric_xtb::{XtbCalculator, XtbConfig, XtbMethod};
//!
//! let mol = Molecule::load_xyz("testdata/molecules/water.xyz")?;
//! let mut calc = XtbCalculator::new(&mol, XtbConfig { method: XtbMethod::Gfn2, ..Default::default() })?;
//! let out = calc.singlepoint()?;
//! println!("E(GFN2-xTB) = {:.10} Ha", out.energy);
//! ```

#[cfg(feature = "xtb")]
pub mod calculator;
#[cfg(feature = "xtb")]
pub mod ffi;

#[cfg(feature = "xtb")]
pub use calculator::{
    gfn2_energy, xtb_singlepoint, XtbCalculator, XtbConfig, XtbMethod, XtbResult,
};

/// Whether this build was compiled with the `xtb` feature (and therefore linked
/// against libxtb).
///
/// Callers that offer xTB screening as an option should branch on this and emit
/// a clear "rebuild with `--features xtb`" message instead of silently skipping
/// the screen.
pub const fn xtb_available() -> bool {
    cfg!(feature = "xtb")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_flag_matches_feature() {
        assert_eq!(xtb_available(), cfg!(feature = "xtb"));
    }
}
