//! Fock matrix builder traits and composite builder.
//!
//! The [`crate::fock::JBuilder`] and [`crate::fock::KBuilder`] traits abstract the Coulomb and exchange
//! matrix construction, allowing future pluggable implementations (e.g., LinK, CFMM).

use ferric_core::FerricError;
use ndarray::Array2;

/// Trait for building the Coulomb matrix J from a density matrix.
pub trait JBuilder {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError>;
    fn reset(&mut self);
}

/// Trait for building the exchange matrix K from a density matrix.
pub trait KBuilder {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError>;

    /// Build K from occupied MO coefficients `c_occ` (an `(n, nocc)` matrix)
    /// instead of the full `(n, n)` density. This computes exactly
    ///
    ///   K_{μν} = Σ_i Σ_{λσ} (μλ|νσ) C_{λi} C_{σi}
    ///
    /// i.e. the exchange matrix for the density `D = C_occ · C_occᵀ`
    /// (occupation weight 1 per column). K is linear in the density, so a caller
    /// whose physical density carries a scalar occupation factor scales the
    /// RETURNED K rather than pre-scaling `c_occ`: RHF's `D = 2·C_occ·C_occᵀ` is
    /// reproduced by passing the bare `C_occ` and doubling K (`K(D) = 2·K(C·Cᵀ)`
    /// — an exact power-of-2 scaling, no rounding, so the closed-shell SCF
    /// trajectory is perturbed only by the B-vs-D contraction reassociation, not
    /// by a √2 pre-scaling of C). UHF's per-spin `D_σ = C_occ,σ·C_occ,σᵀ` maps
    /// directly (factor 1). Fractional / smeared occupations are NOT expressible
    /// as a plain `C·Cᵀ`, so those paths must keep using [`KBuilder::build`] on
    /// the explicit density.
    ///
    /// The default implementation reconstructs `D = c_occ·c_occᵀ` and delegates
    /// to [`KBuilder::build`] — correct but no speedup (used by `DirectK`/`LinkK`,
    /// which have no density-fit factorization to exploit). `DfK` overrides this
    /// with an O(naux·n²·nocc) half-transform (contract the fitted 3-index tensor
    /// against `c_occ` FIRST), replacing the O(naux·n³) density contraction — the
    /// dominant kernel win at large basis / few occupied orbitals.
    fn build_from_occ(
        &mut self,
        c_occ: &Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        let d = c_occ.dot(&c_occ.t());
        self.build(&d, k)
    }

    fn update_density(&mut self, d: &Array2<f64>);
    fn reset(&mut self);

    /// Exchange-grid point count of a grid-based (COSX) builder; `None` for
    /// the integral-driven builders. Read only by the COSX grid schedule's
    /// per-iteration record.
    fn exchange_grid_npts(&self) -> Option<usize> {
        None
    }
}

/// A density-dependent open-shell Fock term for
/// [`crate::uhf::solve_uhf_with_hook`] and
/// [`crate::rohf::solve_rohf_with_hook`]: called with the per-spin densities
/// `(D_α, D_β)` every iteration, it returns `(E, V_α, V_β)`, the term's energy
/// and its potentials `V_σ = ∂E/∂D_σ` (each `(nbasis, nbasis)`, symmetric).
/// The closed-shell counterpart is [`crate::rhf::RhfFockHook`].
pub type SpinFockHook<'a> =
    &'a mut dyn FnMut(
        &Array2<f64>,
        &Array2<f64>,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError>;

/// Add `hook(d_a, d_b)`'s potentials to `f_a`/`f_b` and return its energy
/// (0 without a hook). `who` names the solver in the shape error.
pub(crate) fn apply_spin_fock_hook(
    hook: Option<&mut SpinFockHook<'_>>,
    who: &str,
    (d_a, d_b): (&Array2<f64>, &Array2<f64>),
    f_a: &mut Array2<f64>,
    f_b: &mut Array2<f64>,
) -> Result<f64, FerricError> {
    let Some(h) = hook else {
        return Ok(0.0);
    };
    let (e, v_a, v_b) = h(d_a, d_b)?;
    if v_a.dim() != f_a.dim() || v_b.dim() != f_b.dim() || !e.is_finite() {
        return Err(FerricError::General(format!(
            "{who}: hook returned V_a/V_b of shapes {:?}/{:?} (expected {:?}) and E = {e}",
            v_a.dim(),
            v_b.dim(),
            f_a.dim()
        )));
    }
    *f_a += &v_a;
    *f_b += &v_b;
    Ok(e)
}
