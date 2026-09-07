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

/// Screening policy for the per-grid-point shell-pair sweep.
///
/// A shell pair `(s1, s2)` is retained at grid point `r_g` when
///
/// ```text
///     sqrt(max|(s1 s2)|) * (1 / d) >= threshold
/// ```
///
/// where `d` is the distance from `r_g` to the shell-pair centre of charge and
/// the first factor is the pair's overlap-magnitude bound. This is deliberately
/// crude: the point of Stage 2 is to measure whether *any* honest screen makes
/// the surviving-pair fraction fall with system size, not to build the best
/// possible screen.
///
/// `threshold <= 0.0` disables screening entirely, which is the exactness
/// anchor's trivial limit (see `cosx_a_zero_threshold_matches_unscreened`).
#[derive(Debug, Clone, Copy)]
pub struct CosxScreen {
    /// Screening threshold; `<= 0.0` means "keep every pair".
    pub threshold: f64,
}

impl CosxScreen {
    /// The trivial limit: no pair is ever dropped.
    pub fn none() -> Self {
        Self { threshold: 0.0 }
    }

    /// A screen at the given threshold.
    pub fn at(threshold: f64) -> Self {
        Self { threshold }
    }

    /// True when this screen drops nothing by construction.
    pub fn is_vacuous(&self) -> bool {
        self.threshold <= 0.0
    }
}

/// Per-shell-pair data needed by the screen, precomputed once per geometry.
///
/// Holds, for each shell pair `(s1, s2)` with `s1 >= s2`, the pair's
/// overlap-magnitude bound and its centre of charge (the Gaussian-product
/// centre of the most diffuse primitive pair, which is where the pair density
/// actually sits).
pub struct PairBounds {
    nsh: usize,
    /// `sqrt` of the max absolute overlap-type magnitude, indexed by `tri(s1, s2)`.
    mag: Vec<f64>,
    /// Pair centre of charge, indexed by `tri(s1, s2)`.
    centre: Vec<[f64; 3]>,
}

#[inline]
fn tri(s1: usize, s2: usize) -> usize {
    // s1 >= s2 assumed
    s1 * (s1 + 1) / 2 + s2
}

impl PairBounds {
    /// Build the pair bounds for `prep`.
    ///
    /// The magnitude bound comes from the overlap matrix block, which is the
    /// natural measure of how much AO pair density the shell pair carries.
    pub fn build(prep: &PreparedBasis) -> Result<Self, FerricError> {
        let nsh = prep.nshells();
        let dims = prep.shell_dims();
        // Owned Vec: hoisted out of the loop deliberately.
        let centres: Vec<[f64; 3]> = prep.shell_centers();

        let mut eng = Engine::new_1e(ffi::OP_OVERLAP, prep, 1e-14)?;
        let npair = nsh * (nsh + 1) / 2;
        let mut mag = vec![0.0_f64; npair];
        let mut centre = vec![[0.0_f64; 3]; npair];

        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                let block = eng.compute_1e_block(prep, s1, s2);
                let n = dims[s1] * dims[s2];
                let mut m = 0.0_f64;
                for v in &block[..n.min(block.len())] {
                    m = m.max(v.abs());
                }
                let idx = tri(s1, s2);
                mag[idx] = m.sqrt();
                // Midpoint of the two shell centres: for the screen's purposes
                // (a 1/d falloff) this is adequate and needs no primitive data.
                let c1 = centres[s1];
                let c2 = centres[s2];
                centre[idx] = [
                    0.5 * (c1[0] + c2[0]),
                    0.5 * (c1[1] + c2[1]),
                    0.5 * (c1[2] + c2[2]),
                ];
            }
        }

        Ok(Self { nsh, mag, centre })
    }

    /// Number of shells this was built for.
    pub fn nshells(&self) -> usize {
        self.nsh
    }

    /// Screening estimate for pair `(s1, s2)` at point `r`.
    #[inline]
    pub fn estimate(&self, s1: usize, s2: usize, r: &[f64; 3]) -> f64 {
        let idx = tri(s1, s2);
        let c = self.centre[idx];
        let dx = r[0] - c[0];
        let dy = r[1] - c[1];
        let dz = r[2] - c[2];
        let d2 = dx * dx + dy * dy + dz * dz;
        // Guard the singularity: at d -> 0 the pair is never screened away.
        let d = d2.sqrt().max(1e-8);
        self.mag[idx] / d
    }
}

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
