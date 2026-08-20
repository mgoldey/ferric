//! Boundary-element S (Coulomb) and D (dielectric normal-derivative)
//! matrices on the cavity surface, and the isotropic IEF-PCM operators built
//! from them.
//!
//! # Two S/D formulations
//!
//! [`SdKind::PointCharge`] (the module's original formulation) models each
//! tessera as a bare point charge:
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
//! (Cancès, Mennucci, Tomasi, J. Chem. Phys. 107, 3032 (1997)).
//!
//! [`SdKind::GaussianSmeared`] (PySCF `pcm.py::get_D_S`'s formulation, Li &
//! Frisch/Scalmani, "Continuous Surface Charge Polarizable Continuum
//! Models", J. Chem. Phys. 122, 194110 (2005)) instead treats each tessera as
//! a Gaussian charge distribution of width `xi_k` set by the LOCAL Lebedev
//! grid density (`xi_k = XI[ng] / (r_vdw_k * sqrt(w_k))`, `w_k` the tessera's
//! unnormalized Lebedev weight, see [`crate::cavity::Tessera::charge_exp`]):
//!
//! ```text
//! xi_ij = xi_i * xi_j / sqrt(xi_i^2 + xi_j^2)      (harmonic-combined width)
//! S_ij  = erf(xi_ij * r_ij) / r_ij                 (i ≠ j)
//! S_ii  = xi_i * sqrt(2/pi) / switch_fun_i          (self-term)
//!
//! D_ij  = S_ij * (nrij / r_ij^2)
//!         − (2/sqrt(pi)) * xi_ij * r_ij * exp(−(xi_ij*r_ij)^2) * nrij / r_ij^3
//! D_ii  = − xi_i * sqrt(2/pi) / (2 * R_vdw_i)       (self-term)
//! ```
//!
//! where `nrij = (r_i − r_j)·n_j` (normal at the source point `j`, same
//! convention as `PointCharge`'s `D`). This is ported directly from PySCF's
//! `get_D_S` (equation-level cross-check only, no code copied) and mirrors
//! the [`ferric_scf::cosmo`] crate's `SMatrixKind::GaussianSmeared`, which
//! closed most of that sibling (conductor-limit) model's gap to a PySCF
//! reference — see that crate's module doc. This is the default here (see
//! [`SdKind::default`]); `PointCharge` is kept only for the regression test
//! that measures the isolated S/D formulation effect and to reproduce
//! pre-2026-07-19 numbers.
//!
//! In both cases `D_ij` uses the normal at the *source* point `j` — the
//! convention that makes `D` (not `Dᵀ`) the correct operator for "potential
//! due to a dipole layer on the surface", which is what the IEF-PCM working
//! equation needs (see `solver.rs`).
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
//! tessellation, S/D formulas, and linear-algebra plumbing below are an
//! independent implementation.

use ferric_core::FerricError;
use ndarray::Array2;

use crate::cavity::Tessera;

/// GEPOL point-charge self-energy constant (dimensionless), used only by
/// [`SdKind::PointCharge`].
const XI_SELF: f64 = 1.0694;

/// Per-Lebedev-order Gaussian-charge-width prefactor `XI[ng]` (Table II, Li &
/// Frisch/Scalmani, J. Chem. Phys. 122, 194110 (2005), as used by PySCF's
/// `pcm.py`). Only the six Lebedev orders `ferric_dft::lebedev` (and hence
/// this crate's cavity construction) supports are tabulated — matches
/// `ferric_scf::cosmo::gaussian_xi_table` exactly. Returns `Err` for any
/// other order rather than silently guessing a value.
pub(crate) fn gaussian_xi_table(lebedev_order: usize) -> Result<f64, FerricError> {
    match lebedev_order {
        6 => Ok(4.84566077868),
        14 => Ok(4.86458714334),
        26 => Ok(4.85478226219),
        50 => Ok(4.89250673295),
        110 => Ok(4.90101060987),
        302 => Ok(4.90498088169),
        _ => Err(FerricError::General(format!(
            "gaussian_xi_table: lebedev_order {lebedev_order} not in the supported set \
             {{6,14,26,50,110,302}}"
        ))),
    }
}

/// Which S/D (boundary-element) formulation to use. See the module doc for
/// the exact formulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SdKind {
    PointCharge,
    GaussianSmeared,
}

impl Default for SdKind {
    fn default() -> Self {
        SdKind::GaussianSmeared
    }
}

// libm's `erf` (C99) is already transitively linked into every ferric binary
// (libint2/OpenBLAS both pull in libm) — bind it directly rather than adding
// a crate dependency, same convention as `ferric_scf::cosmo`.
extern "C" {
    fn erf(x: f64) -> f64;
}

/// Build the S (Coulomb) and D (normal-derivative Coulomb) matrices for a
/// tessellated cavity, using the default [`SdKind::GaussianSmeared`]
/// formulation. Both are `(n, n)` where `n = tess.len()`.
pub fn build_s_d(tess: &[Tessera]) -> (Array2<f64>, Array2<f64>) {
    build_s_d_kind(tess, SdKind::default())
}

/// Build the S/D matrices per `kind` — see the module doc for the exact
/// [`SdKind::PointCharge`] vs [`SdKind::GaussianSmeared`] formulas.
pub fn build_s_d_kind(tess: &[Tessera], kind: SdKind) -> (Array2<f64>, Array2<f64>) {
    let n = tess.len();
    let mut s = Array2::<f64>::zeros((n, n));
    let mut d = Array2::<f64>::zeros((n, n));

    match kind {
        SdKind::PointCharge => {
            for i in 0..n {
                let ti = &tess[i];
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
        }
        SdKind::GaussianSmeared => {
            let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
            for i in 0..n {
                let ti = &tess[i];
                let xi_i = ti.charge_exp;
                // Diagonal self-terms (PySCF get_D_S: fill_diagonal after the
                // off-diagonal loop; same values, computed here up front).
                s[(i, i)] = xi_i * sqrt_2_over_pi / ti.switch_fun;
                d[(i, i)] = -xi_i * sqrt_2_over_pi / (2.0 * ti.sphere_radius);

                for j in (i + 1)..n {
                    let tj = &tess[j];
                    let xi_j = tj.charge_exp;
                    let dx = ti.position[0] - tj.position[0];
                    let dy = ti.position[1] - tj.position[1];
                    let dz = ti.position[2] - tj.position[2];
                    let r2 = dx * dx + dy * dy + dz * dz;
                    let r = r2.sqrt();
                    let xi_ij = xi_i * xi_j / (xi_i * xi_i + xi_j * xi_j).sqrt();
                    let xi_r = xi_ij * r;
                    // SAFETY: erf is the C standard math library function; xi_r is a finite f64.
                    let s_ij = unsafe { erf(xi_r) } / r;
                    s[(i, j)] = s_ij;
                    s[(j, i)] = s_ij; // S is symmetric.

                    // D_ij = S_ij * nrij_j/r^2 - (2/sqrt(pi))*xi_ij*r*exp(-xi_r^2)*nrij_j/r^3
                    // (normal at source j), and D_ji analogously with normal
                    // at source i — exactly PySCF get_D_S's
                    // `D = S*nrij/rij**2 - 2*xi_r_ij/sqrt(pi)*exp(-xi_r_ij**2)*nrij/rij**3`
                    // evaluated once per (i,j) with nrij = (r_i-r_j)·n_j, and
                    // once per (j,i) with nrij = (r_j-r_i)·n_i.
                    let gauss_term = 2.0 / std::f64::consts::PI.sqrt()
                        * xi_r
                        * (-xi_r * xi_r).exp()
                        / (r2 * r);

                    let dot_j = dx * tj.normal[0] + dy * tj.normal[1] + dz * tj.normal[2];
                    d[(i, j)] = s_ij * dot_j / r2 - gauss_term * dot_j;

                    let dot_i = (-dx) * ti.normal[0] + (-dy) * ti.normal[1] + (-dz) * ti.normal[2];
                    d[(j, i)] = s_ij * dot_i / r2 - gauss_term * dot_i;
                }
            }
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
    fn s_diagonal_positive_point_charge_matches_coulomb() {
        // PointCharge kind: off-diagonal is exactly bare 1/r by construction.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let (s, _d) = build_s_d_kind(&tess, SdKind::PointCharge);
        for i in 0..s.nrows() {
            assert!(s[(i, i)] > 0.0);
        }
        if s.nrows() > 1 {
            let dx = tess[0].position[0] - tess[1].position[0];
            let dy = tess[0].position[1] - tess[1].position[1];
            let dz = tess[0].position[2] - tess[1].position[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!((s[(0, 1)] - 1.0 / r).abs() < 1e-12);
        }
    }

    #[test]
    fn s_diagonal_positive_gaussian_smeared_matches_formula() {
        // GaussianSmeared kind: diagonal must be positive, and the
        // off-diagonal must match the erf(xi_ij*r)/r formula directly (NOT
        // bare 1/r -- erf saturates to ~1 for well-separated tesserae, which
        // would make a bare-1/r comparison pass by numerical coincidence
        // rather than actually exercising the smearing).
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
        let (s, _d) = build_s_d_kind(&tess, SdKind::GaussianSmeared);
        for i in 0..s.nrows() {
            assert!(s[(i, i)] > 0.0);
        }
        if s.nrows() > 1 {
            let ti = &tess[0];
            let tj = &tess[1];
            let dx = ti.position[0] - tj.position[0];
            let dy = ti.position[1] - tj.position[1];
            let dz = ti.position[2] - tj.position[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let xi_i = ti.charge_exp;
            let xi_j = tj.charge_exp;
            let xi_ij = xi_i * xi_j / (xi_i * xi_i + xi_j * xi_j).sqrt();
            // SAFETY: erf is the C standard math library function; argument is finite.
            let expected = unsafe { erf(xi_ij * r) } / r;
            assert!((s[(0, 1)] - expected).abs() < 1e-12);
            // NOTE: for THIS specific well-separated tessera pair,
            // xi_ij*r ~ 31 so erf(xi_ij*r) saturates to exactly 1.0 in double
            // precision -- the smearing is only distinguishable from bare
            // Coulomb for close/same-sphere-neighbor tesserae or small xi.
            // The formula-match assertion above is the meaningful check;
            // see `s_diagonal_positive_point_charge_matches_coulomb` for the
            // bare-Coulomb comparison on the SAME kind.
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
