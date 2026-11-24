//! The pVp integral `W_mn = sum_a <d_a m | V | d_a n>` with
//! `V(r) = -sum_C q_C / |r - R_C|`.
//!
//! `W / (4 c^2)` is the small-component potential of the modified Dirac
//! equation used by the spin-free X2C-1e and DKH Hamiltonians (PySCF's
//! `int1e_pnucp`).
//!
//! The block comes from libint2's nuclear-attraction second derivatives.
//! A basis function centred at `A` satisfies `d m / d A_a = -d_a m`, so the
//! mixed bra-centre/ket-centre second derivative is
//! `d^2 <m|V|n> / dA_a dB_a = <d_a m | V | d_a n>`, and `W` is the sum of
//! the three blocks `(bra a, ket a)`, coordinates `a` and `3 + a` of
//! [`crate::engine::Engine::compute_1e_deriv2_block`]. libint2 reports these two centres
//! separately even when bra and ket sit on the same atom, and the charges'
//! own coordinates (`6..`) do not enter this block.
//!
//! The charges are worked through in chunks of [`crate::pvp::CHARGE_CHUNK`]: the result
//! is linear in the charges, and a chunk keeps the second-derivative buffer
//! at `O(chunk^2)` blocks instead of `O(natoms^2)`.
//!
//! Equations and closed forms: `scripts/pvp_ss_sympy.py`; tests:
//! `crates/ferric-integrals/tests/pvp.rs`.

use crate::basis_bridge::PreparedBasis;
use crate::engine::{deriv2_pair_index, Engine};
use crate::ffi;
use ferric_core::external_potential::PointCharge;
use ferric_core::FerricError;
use ndarray::Array2;

/// Number of point charges handed to the second-derivative engine at once.
pub const CHARGE_CHUNK: usize = 8;

/// `W_mn = sum_a <d_a m | -sum_C q_C/|r - R_C| | d_a n>` over the charges
/// `charges` (Bohr), shape `(nbasis, nbasis)`, symmetric.
///
/// Errors when the linked libint2 has no second-derivative one-body
/// integrals for the basis' angular momentum.
pub fn pvp(prep: &PreparedBasis, charges: &[PointCharge]) -> Result<Array2<f64>, FerricError> {
    let n = prep.nbasis();
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut out = Array2::<f64>::zeros((n, n));
    if charges.is_empty() {
        return Ok(out);
    }
    let mut eng = Engine::new_1e_deriv2(ffi::OP_NUCLEAR, prep, 1e-14)?;
    for chunk in charges.chunks(CHARGE_CHUNK) {
        eng.set_point_charges_only(chunk)?;
        let ncoord = 3 * (2 + chunk.len());
        let blocks = [0usize, 1, 2].map(|a| deriv2_pair_index(ncoord, a, 3 + a));
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                let n1 = dims[s1];
                let n2 = dims[s2];
                let nn = n1 * n2;
                let d2 = eng.compute_1e_deriv2_block(prep, s1, s2, chunk.len());
                for i in 0..n1 {
                    for j in 0..n2 {
                        let k = i * n2 + j;
                        let v: f64 = blocks.iter().map(|&blk| d2[blk * nn + k]).sum();
                        out[(offs[s1] + i, offs[s2] + j)] += v;
                        if s1 != s2 {
                            out[(offs[s2] + j, offs[s1] + i)] += v;
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// [`pvp`] over the nuclei of `prep` (the same charges as
/// [`crate::oneelectron::nuclear`]: zero for ghosts, `Z - n_core` for ECP
/// atoms).
pub fn pvp_nuclear(prep: &PreparedBasis) -> Result<Array2<f64>, FerricError> {
    let charges: Vec<PointCharge> = prep
        .atoms()
        .iter()
        .map(|a| PointCharge {
            q: a.atomic_number,
            x: a.x,
            y: a.y,
            z: a.z,
        })
        .collect();
    pvp(prep, &charges)
}
