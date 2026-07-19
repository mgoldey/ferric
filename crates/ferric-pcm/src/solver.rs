//! IEF-PCM linear solve for the apparent surface charge `q`, and the
//! resulting solvation ("reaction field") energy.

use ferric_core::FerricError;
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

/// Result of one IEF-PCM charge solve.
#[derive(Debug, Clone)]
pub struct PcmChargeResult {
    /// Apparent surface charge at each tessera, in a.u. (symmetrized — see
    /// below).
    pub q: Array1<f64>,
    /// Reaction-field ("solvation") energy contribution: `E_pcm = ½ q·v`,
    /// where `v` is the solute potential at the tesserae. This is the
    /// energy term to be ADDED to the total electronic energy (it is
    /// already negative/stabilizing for a polar solute in a solvent, by
    /// construction of the sign convention below).
    pub e_pcm: f64,
}

/// Solve the IEF-PCM working equation `K q = R v` for the apparent surface
/// charge `q`, given the total (nuclear + electronic) electrostatic
/// potential `v` at the cavity tesserae.
///
/// Following Cancès-Mennucci-Tomasi (and as implemented, non-anisotropic
/// case, by every mainstream PCM code): the *symmetric* combination
///
/// ```text
///     q = ½ [ K⁻¹ R v + (Kᵀ)⁻¹ Rᵀ v ]
/// ```
///
/// is used rather than the bare (non-symmetric) `K⁻¹ R v`, because the raw
/// IEF-PCM operator `K` is not symmetric (`D` is not symmetric) and using it
/// directly would make the reaction-field energy path-dependent / not
/// variational. Symmetrizing this way reproduces the same fixed point as
/// the un-symmetrized solve to the extent `K` is close to symmetric, and is
/// the standard prescription (Cancès 1997; matches the "symmetrized"
/// solve pattern cross-checked against PySCF's `pcm.py::_get_vind`, which
/// computes the analogous `q_sym = (q + qt)/2`; equation-level cross-check
/// only, no code copied).
///
/// `energy_conv` is not iterative here: for a single frozen density this is
/// an exact (LAPACK-precision) linear solve, not a fixed-point iteration —
/// the *self-consistency* with the QM density is handled by the SCF outer
/// loop re-calling this once per iteration with the updated `v`.
///
/// Returns `Err(FerricError::Lapack(..))` if either solve is singular
/// (degenerate/ill-conditioned cavity — e.g. two tesserae coincide) rather
/// than silently returning NaN/garbage charges.
pub fn solve_pcm_charges(
    k: &Array2<f64>,
    r: &Array2<f64>,
    v: &Array1<f64>,
) -> Result<PcmChargeResult, FerricError> {
    let n = v.len();
    if k.shape() != [n, n] || r.shape() != [n, n] {
        return Err(FerricError::General(format!(
            "solve_pcm_charges: shape mismatch — k={:?}, r={:?}, v.len()={n}",
            k.shape(),
            r.shape()
        )));
    }

    let rv = r.dot(v);
    let q1 = k.solve(&rv).map_err(|e| {
        FerricError::Lapack(format!(
            "IEF-PCM charge solve (K q = R v) failed — singular/ill-conditioned cavity: {e}"
        ))
    })?;

    let k_t = k.t().to_owned();
    let r_t = r.t().to_owned();
    let rtv = r_t.dot(v);
    let q2 = k_t.solve(&rtv).map_err(|e| {
        FerricError::Lapack(format!(
            "IEF-PCM transpose charge solve (Kᵀ q = Rᵀ v) failed — singular/ill-conditioned cavity: {e}"
        ))
    })?;

    let q = 0.5 * (&q1 + &q2);
    let e_pcm = 0.5 * q.dot(v);

    Ok(PcmChargeResult { q, e_pcm })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cavity::{build_cavity, CavityConfig};
    use crate::matrices::{build_k_r, build_s_d};
    use ferric_core::mol::Molecule;

    #[test]
    fn shape_mismatch_is_an_error() {
        let k = Array2::<f64>::eye(3);
        let r = Array2::<f64>::eye(3);
        let v = Array1::<f64>::zeros(4);
        assert!(solve_pcm_charges(&k, &r, &v).is_err());
    }

    #[test]
    fn zero_potential_gives_zero_charge_and_energy() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let (s, d) = build_s_d(&tess);
        let (k, r, _f) = build_k_r(&s, &d, &tess, 78.4).unwrap();
        let v = Array1::<f64>::zeros(tess.len());
        let res = solve_pcm_charges(&k, &r, &v).unwrap();
        assert!(res.q.iter().all(|&x| x == 0.0));
        assert_eq!(res.e_pcm, 0.0);
    }

    #[test]
    fn uniform_positive_potential_gives_stabilizing_negative_energy() {
        // A uniform positive potential (as if from a positive point charge
        // far away, roughly constant across the cavity) should induce a net
        // negative surface charge (screening) and a net-negative (stabilizing)
        // PCM energy contribution -- the qualitative sign check.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let (s, d) = build_s_d(&tess);
        let (k, r, _f) = build_k_r(&s, &d, &tess, 78.4).unwrap();
        let v = Array1::<f64>::from_elem(tess.len(), 0.1);
        let res = solve_pcm_charges(&k, &r, &v).unwrap();
        assert!(res.e_pcm < 0.0, "expected stabilizing (negative) E_pcm, got {}", res.e_pcm);
    }
}
