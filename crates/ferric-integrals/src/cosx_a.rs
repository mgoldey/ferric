//! Per-grid-point three-center-one-electron AO blocks for seminumerical
//! exchange (COSX / sn-LinK).
//!
//! The quantity built here is
//!
//! ```text
//!     A^g_{mu,nu} = \int chi_mu(r) chi_nu(r) / |r - r_g| dr
//! ```
//!
//! i.e. the Coulomb potential of the AO pair density evaluated at grid point
//! `r_g`, **retained as a matrix** rather than contracted against a density.
//! This is the piece a COSX exchange build needs (Neese, Chem. Phys. 356, 98
//! (2009)); contracting it against `D` and summing over `g` instead recovers
//! the electrostatic potential, which is what `ferric_scf::properties::esp_at_points`
//! already does and which serves as this module's independent-construction
//! anchor.
//!
//! # Sign convention (load-bearing)
//!
//! libint2's nuclear-attraction operator with a unit probe charge returns the
//! **attractive** kernel `<mu| -1/|r-r_g| |nu>`. COSX needs the **repulsive**
//! `+1/|r-r_g|`, so the raw libint2 block is **negated** here. This is the
//! opposite of PySCF's `int1e_grids`, which already returns the positive
//! kernel; the Stage 0 python prototype (`scripts/cosx_proto.py`) got this
//! backwards on its first run and the error showed up as `max|dK|` plateauing
//! at `2*||K||_max`.
//!
//! Because the anchor test compares against `esp_at_points` (which consumes
//! the *un*-negated libint2 convention), the anchor must flip the sign back.
//! That asymmetry is deliberate and is documented at the anchor call site.
//!
//! # Cost structure
//!
//! libint2 offers no per-charge output for the nuclear operator: its engine
//! (`engine.impl.h`, the `for (pset ...)` loop) `std::plus`-accumulates every
//! point charge into a single scratch buffer. Batching many grid points into
//! one `set_params` call therefore returns the *sum over grid points*, which
//! is meaningless here. Each grid point consequently costs one
//! `set_point_charges` call plus a full shell-pair sweep, and
//! `set_point_charges` re-runs `compute_primdata` over every primitive pair.
//! This is the dominant cost of the whole approach and the reason Stage 2
//! exists as a kill gate.

use ferric_core::error::FerricError;
use ndarray::Array2;
use std::os::raw::c_int;

use crate::basis_bridge::PreparedBasis;
use crate::engine::Engine;
use crate::ffi::{self, CAtom};

// Screening policy and shell-pair bounds live in `cosx_screen` (shared with
// `md3c1e`); re-exported here so existing `cosx_a::{CosxScreen, PairBounds}`
// paths keep working.
pub use crate::cosx_screen::{CosxScreen, PairBounds};

/// Result of a single-point A-matrix build.
pub struct CosxPoint {
    /// The `(nbf, nbf)` matrix `A^g_{mu,nu}` with the `+1/|r-r_g|` sign.
    pub a: Array2<f64>,
    /// Number of shell pairs actually evaluated (after screening).
    pub pairs_kept: usize,
    /// Number of shell pairs considered (the unscreened total).
    pub pairs_total: usize,
}

/// Build the A-matrix at one grid point, using a pre-created engine.
///
/// The engine must have been created with [`ffi::OP_NUCLEAR`]. Reusing one
/// engine across many points is the whole point: engine creation is expensive
/// and `map_init` in the caller hands one engine per rayon worker.
pub fn a_matrix_at_point_with(
    eng: &mut Engine,
    prep: &PreparedBasis,
    r: &[f64; 3],
    bounds: Option<&PairBounds>,
    screen: CosxScreen,
) -> Result<CosxPoint, FerricError> {
    let nbf = prep.nbasis();
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();

    // Install a unit probe charge at r_g. libint2 then evaluates the
    // nuclear-attraction operator -1/|r - r_g|.
    let probe = [CAtom { atomic_number: 1.0, x: r[0], y: r[1], z: r[2] }];
    // SAFETY: `probe` is a stack-local CAtom slice alive for the call;
    // `handle_mut()` is the live engine pointer; length fits in c_int. The
    // shim catches C++ exceptions and returns a negative status.
    let rc = unsafe {
        ffi::scf_engine_set_point_charges(eng.handle_mut(), probe.as_ptr(), probe.len() as c_int)
    };
    if rc < 0 {
        return Err(FerricError::General(format!(
            "cosx a_matrix_at_point: set_point_charges failed (rc={rc})"
        )));
    }

    let mut a = Array2::<f64>::zeros((nbf, nbf));
    let mut pairs_kept = 0usize;
    let mut pairs_total = 0usize;

    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            pairs_total += 1;
            if !screen.is_vacuous() {
                if let Some(b) = bounds {
                    if b.estimate(s1, s2, r) < screen.threshold {
                        continue;
                    }
                }
            }
            pairs_kept += 1;

            let block = eng.compute_1e_block(prep, s1, s2);
            let n1 = dims[s1];
            let n2 = dims[s2];
            let o1 = offs[s1];
            let o2 = offs[s2];
            for i in 0..n1 {
                for j in 0..n2 {
                    // Negate: libint2 gives -1/|r-r_g|, COSX wants +1/|r-r_g|.
                    let v = -block[i * n2 + j];
                    a[(o1 + i, o2 + j)] = v;
                    a[(o2 + j, o1 + i)] = v;
                }
            }
        }
    }

    Ok(CosxPoint { a, pairs_kept, pairs_total })
}

/// Build the A-matrix at one grid point, creating a throwaway engine.
///
/// Convenience wrapper for tests and one-shot use. Production callers that
/// sweep many points must use [`a_matrix_at_point_with`] with a reused engine.
pub fn a_matrix_at_point(
    prep: &PreparedBasis,
    r: &[f64; 3],
    bounds: Option<&PairBounds>,
    screen: CosxScreen,
) -> Result<CosxPoint, FerricError> {
    let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14)?;
    a_matrix_at_point_with(&mut eng, prep, r, bounds, screen)
}
