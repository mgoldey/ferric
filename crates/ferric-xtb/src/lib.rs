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
//! meson setup build --prefix=$HOME/.local --buildtype=release \
//!   -Dlapack=openblas -Dtblite=disabled -Dcpcmx=disabled -Ddefault_library=shared
//! meson compile -C build
//! meson install -C build
//! ```
//!
//! `tblite`/`cpcmx` are optional alternate-backend/solvation features that the
//! GFN1/GFN2/GFN-FF C API does not need.
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
//! **Gradients: transferred faithfully, but DO NOT TRUST THE VALUES.** This
//! binding reproduces the CLI's own `gradient` file to <1e-10, so the FFI
//! transfer is correct -- but xtb 6.7.1 itself returns an analytic gradient that
//! disagrees with a finite difference of its own energy (~20x too large on
//! water), and **xtb's own `meson test` suite fails the same way**
//! (`unit - xtb:gfn2`, `unit - xtb:gfn1`, `unit - xtb:hessian`). Reproduced in
//! builds both with and without OpenMP. See the doc comment on
//! `gfn2_gradient_disagrees_with_finite_difference` in
//! `tests/xtb_singlepoint.rs` for the full evidence chain.
//!
//! Consequently: use this crate for **energy-ranked conformer screening**, not
//! for xTB geometry optimisation or forces, until the upstream defect is
//! resolved.
//!
//! # Threading
//!
//! libxtb is **not thread-safe** (process-global state -- measured, see
//! [`calculator`]). Parallelise across processes, not threads.
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
