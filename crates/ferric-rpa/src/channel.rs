//! Per-spin RPA input channel.
//!
//! Every RPA response kernel (dielectric build, χ₀ apply, dRPA energy) consumes
//! the same three quantities for a given spin: the RI three-index block
//! `B^P_{ia}`, the occupied orbital energies, and the virtual orbital energies.
//! In the unrestricted path these arrive as an α/β pair, so the loose form is
//! six positional arguments that must stay correctly ordered. [`crate::channel::RpaChannel`]
//! bundles the triple so the two spins travel as two values instead of six.

use ndarray::Array2;

/// The RI/orbital-energy inputs for one spin channel of an RPA calculation.
///
/// Borrows its data; construct one per spin and pass `&RpaChannel` into the
/// response kernels.
#[derive(Debug, Clone, Copy)]
pub struct RpaChannel<'a> {
    /// RI three-index occupied-virtual block `B^P_{ia}`, shape `(naux, nocc*nvir)`.
    pub b_ov: &'a Array2<f64>,
    /// Occupied orbital energies for this spin.
    pub eps_occ: &'a [f64],
    /// Virtual orbital energies for this spin.
    pub eps_vir: &'a [f64],
}

impl<'a> RpaChannel<'a> {
    /// Bundle the RI block and orbital energies for one spin channel.
    pub fn new(b_ov: &'a Array2<f64>, eps_occ: &'a [f64], eps_vir: &'a [f64]) -> Self {
        Self { b_ov, eps_occ, eps_vir }
    }
}
