//! Boundary-element S (Coulomb) and D (dielectric normal-derivative)
//! matrices on the cavity surface, and the isotropic IEF-PCM operators built
//! from them.
//!
//! # Definitions (point-tessera / GEPOL convention)
//!
//! For tesserae `i, j` with positions `r_i`, areas `a_i`, outward normals
//! `n_i`, and parent-sphere radius `R_i`:
//!
//! ```text
//! S_ij = 1 / |r_i − r_j|                          (i ≠ j)
//! S_ii = ξ · sqrt(4π / a_i)                        (self-term)
//!
//! D_ij = (r_i − r_j)·n_j / |r_i − r_j|³            (i ≠ j)
//! D_ii = − S_ii / (2 R_i)                          (self-term)
//! ```
//!
//! with `ξ = 1.0694` the standard GEPOL point-charge self-energy constant
//! (Cancès, Mennucci, Tomasi, J. Chem. Phys. 107, 3032 (1997); the same
//! constant appears in the original GEPOL papers of Pomelli & Tomasi and in
//! the Tomasi/Mennucci/Cammi Chem. Rev. 2005, 105, 2999 review, §2.3/§3 for
//! the "point charge" discretization of the apparent surface charge). This
//! is the classical closed-form diagonal used before the smoother
//! switching/Gaussian (SWIG) discretization of Lange & Herbert (2010)
//! superseded it in some production codes (e.g. PySCF's `pcm.py`, which
//! instead uses Gaussian-smeared surface charges with an
//! analytically-regularized diagonal) — see the module doc in `cavity.rs`
//! for why this simpler point-tessera scheme was chosen here.
//!
//! `D_ij` as defined above uses the normal at the *source* point `j` — this
//! is the convention that makes `D` (not `Dᵀ`) the correct operator for
//! "potential due to a dipole layer on the surface", which is what the
//! IEF-PCM working equation needs (see `solver.rs`).
//!
//! # IEF-PCM operators
//!
//! With `A = diag(area_i)` and dielectric factor `f(ε) = (ε − 1)/(ε + 1)`:
//!
//! ```text
//! DA  = D · A
//! K   = S − [f(ε) / 2π] · DA · S
//! R   = −f(ε) · (I − DA / 2π)
//! ```
//!
//! solving `K q = R v` for the apparent surface charge `q` given the
//! solute's electrostatic potential `v` at the tesserae (see `solver.rs`).
//! This exact `K`/`R` construction (and the `f(ε) = (ε−1)/(ε+1)` isotropic
//! reduction of the general anisotropic IEF-PCM operator) was cross-checked
//! against the open-source PySCF `pcm.py` IEF-PCM branch (build() method) —
//! math/equation verification only, no code was copied; ferric's
//! tessellation, S/D diagonal formulas, and linear-algebra plumbing below
//! are an independent implementation.

use ferric_core::FerricError;
use ndarray::Array2;

use crate::cavity::Tessera;

/// GEPOL point-charge self-energy constant (dimensionless).
const XI_SELF: f64 = 1.0694;

/// Build the S (Coulomb) and D (normal-derivative Coulomb) matrices for a
/// tessellated cavity. Both are `(n, n)` where `n = tess.len()`.
pub fn build_s_d(tess: &[Tessera]) -> (Array2<f64>, Array2<f64>) {
    let n = tess.len();
    let mut s = Array2::<f64>::zeros((n, n));
    let mut d = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        let ti = &tess[i];
        // Diagonal self-terms.
        let s_ii = XI_SELF * (4.0 * std::f64::consts::PI / ti.area).sqrt();
        s[(i, i)] = s_ii;
        d[(i, i)] = -s_ii / (2.0 * ti.sphere_radius);

        for j in (i + 1)..n {
            let tj = &tess[j];
            let dx = ti.position[0] - tj.position[0];
            let dy = ti.position[1] - tj.position[1];
            let dz = ti.position[2] - tj.position[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            let s_ij = 1.0 / r;
            s[(i, j)] = s_ij;
            s[(j, i)] = s_ij; // S is symmetric.

            // D_ij = (r_i - r_j)·n_j / |r_i-r_j|^3  (normal at source j)
            let r3 = r2 * r;
            let dot_j = dx * tj.normal[0] + dy * tj.normal[1] + dz * tj.normal[2];
            d[(i, j)] = dot_j / r3;

            // D_ji = (r_j - r_i)·n_i / |r_j-r_i|^3  (normal at source i)
            let dot_i = (-dx) * ti.normal[0] + (-dy) * ti.normal[1] + (-dz) * ti.normal[2];
            d[(j, i)] = dot_i / r3;
        }
    }

    (s, d)
}

/// Build the isotropic IEF-PCM operators `K` and `R` (see module docs) from
/// the S/D matrices, the tessera areas, and the solvent dielectric constant
/// `eps`.
///
/// Returns `Err` if `eps` is not a valid dielectric constant (`eps <= 1.0`
/// is unphysical — vacuum is `eps = 1`, no screening at all — or
/// non-finite).
pub fn build_k_r(
    s: &Array2<f64>,
    d: &Array2<f64>,
    tess: &[Tessera],
    eps: f64,
) -> Result<(Array2<f64>, Array2<f64>, f64), FerricError> {
    if !(eps.is_finite() && eps > 1.0) {
        return Err(FerricError::General(format!(
            "build_k_r: invalid dielectric constant eps={eps} (must be finite and > 1.0)"
        )));
    }
    let n = tess.len();
    let f_eps = (eps - 1.0) / (eps + 1.0);

    // DA = D * A, where A = diag(area). Column-scale D by area_j.
    let mut da = d.clone();
    for j in 0..n {
        let a_j = tess[j].area;
        for i in 0..n {
            da[(i, j)] *= a_j;
        }
    }

    let das = da.dot(s);
    let two_pi = 2.0 * std::f64::consts::PI;

    // K = S - f(eps)/(2*pi) * DA * S
    let k = s - &(f_eps / two_pi * &das);

    // R = -f(eps) * (I - DA/(2*pi))
    let eye = Array2::<f64>::eye(n);
    let r = -f_eps * (&eye - &(&da / two_pi));

    Ok((k, r, f_eps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cavity::{build_cavity, CavityConfig};
    use ferric_core::mol::Molecule;

    #[test]
    fn s_matrix_is_symmetric() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let (s, _d) = build_s_d(&tess);
        let n = s.nrows();
        for i in 0..n {
            for j in 0..n {
                assert!((s[(i, j)] - s[(j, i)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn s_diagonal_positive_and_off_diagonal_matches_coulomb() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let (s, _d) = build_s_d(&tess);
        for i in 0..s.nrows() {
            assert!(s[(i, i)] > 0.0);
        }
        // Off-diagonal spot check against direct 1/r.
        if s.nrows() > 1 {
            let dx = tess[0].position[0] - tess[1].position[0];
            let dy = tess[0].position[1] - tess[1].position[1];
            let dz = tess[0].position[2] - tess[1].position[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!((s[(0, 1)] - 1.0 / r).abs() < 1e-12);
        }
    }

    #[test]
    fn build_k_r_rejects_vacuum_and_subunity_eps() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let (s, d) = build_s_d(&tess);
        assert!(build_k_r(&s, &d, &tess, 1.0).is_err());
        assert!(build_k_r(&s, &d, &tess, 0.5).is_err());
        assert!(build_k_r(&s, &d, &tess, f64::NAN).is_err());
        assert!(build_k_r(&s, &d, &tess, 78.4).is_ok());
    }

    #[test]
    fn f_eps_matches_ief_pcm_formula() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let (s, d) = build_s_d(&tess);
        let (_k, _r, f_eps) = build_k_r(&s, &d, &tess, 78.4).unwrap();
        let expected = (78.4 - 1.0) / (78.4 + 1.0);
        assert!((f_eps - expected).abs() < 1e-12);
    }
}
