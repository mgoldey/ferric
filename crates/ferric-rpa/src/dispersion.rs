//! Many-body-dispersion C6 coefficients (data-product track).
//!
//! Two sources of the per-atom dynamic polarizability α^A(iω):
//!   * [`ts_dynamic_polarizability`] — Tkatchenko-Scheffler single-pole London
//!     model (Phase 1).
//!   * `pdep_dynamic_polarizability` — PDEP-RPA imaginary-frequency SMW
//!     (Phase 2, added later).
//!
//! Both feed [`casimir_polder_c6`], which contracts α^A(iω):α^B(iω) over the
//! imaginary-frequency quadrature to give isotropic and anisotropic C6^{AB}.

pub mod free_atom_ref;

use ndarray::Array2;

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;

use crate::config::PdepRpaConfig;
use crate::dispersion::free_atom_ref::ts_free_atom;

/// Per-atom dynamic polarizability on an imaginary-frequency quadrature grid.
#[derive(Debug, Clone)]
pub struct DynamicPolarizability {
    /// Imaginary-frequency nodes ω_k (a.u.).
    pub freqs: Vec<f64>,
    /// Casimir-Polder quadrature weights w_k (a.u.).
    pub weights: Vec<f64>,
    /// `per_atom[a][k]` = 3×3 tensor α^A_{ij}(iω_k), a.u.
    pub per_atom: Vec<Vec<[[f64; 3]; 3]>>,
}

/// C6 result: the fundamental per-atom α(iω) plus derived pair coefficients.
#[derive(Debug, Clone)]
pub struct C6Result {
    pub per_atom_dynamic: DynamicPolarizability,
    /// Isotropic C6^{AB}, shape (N, N), a.u.
    pub c6_iso_pair: Array2<f64>,
    /// Anisotropic C6^{AB}_{ij}: `c6_aniso_pair[a][b]` = 3×3 tensor.
    pub c6_aniso_pair: Vec<Vec<[[f64; 3]; 3]>>,
}

/// Partition scheme for the per-atom decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispersionPartition {
    Becke,
    Hirshfeld,
}

impl Default for DispersionPartition {
    fn default() -> Self {
        DispersionPartition::Becke
    }
}

/// Casimir-Polder contraction. SHARED SEAM between TS and PDEP-RPA sources.
///
/// ```text
///   C6^{AB}_{ij} = (3/π) Σ_k w_k α^A_{ij}(iω_k) α^B_{ij}(iω_k)
///   C6^{AB}_iso  = (3/π) Σ_k w_k α^A_iso(iω_k) α^B_iso(iω_k)
/// ```
/// where α_iso(iω_k) = (1/3) Tr α(iω_k).
pub fn casimir_polder_c6(dyn_pol: &DynamicPolarizability) -> C6Result {
    use std::f64::consts::PI;
    let natoms = dyn_pol.per_atom.len();
    let nfreq = dyn_pol.freqs.len();
    let pref = 3.0 / PI;

    // Precompute per-atom isotropic profiles α_iso^A(iω_k).
    let mut iso: Vec<Vec<f64>> = vec![vec![0.0; nfreq]; natoms];
    for a in 0..natoms {
        for k in 0..nfreq {
            let t = dyn_pol.per_atom[a][k];
            iso[a][k] = (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        }
    }

    let mut c6_iso_pair = Array2::<f64>::zeros((natoms, natoms));
    let mut c6_aniso_pair: Vec<Vec<[[f64; 3]; 3]>> =
        vec![vec![[[0.0; 3]; 3]; natoms]; natoms];

    for a in 0..natoms {
        for b in 0..natoms {
            // Isotropic.
            let mut s_iso = 0.0;
            for k in 0..nfreq {
                s_iso += dyn_pol.weights[k] * iso[a][k] * iso[b][k];
            }
            c6_iso_pair[(a, b)] = pref * s_iso;

            // Anisotropic, element-wise.
            for i in 0..3 {
                for j in 0..3 {
                    let mut s = 0.0;
                    for k in 0..nfreq {
                        s += dyn_pol.weights[k]
                            * dyn_pol.per_atom[a][k][i][j]
                            * dyn_pol.per_atom[b][k][i][j];
                    }
                    c6_aniso_pair[a][b][i][j] = pref * s;
                }
            }
        }
    }

    C6Result {
        per_atom_dynamic: dyn_pol.clone(),
        c6_iso_pair,
        c6_aniso_pair,
    }
}

/// Tkatchenko-Scheffler dynamic per-atom polarizability via a single London
/// pole, with directional shape inherited from the static α^A tensor.
///
/// Inputs:
///   * `z`            — atomic numbers, length N.
///   * `vol_ratio`    — effective-volume ratio v_A / v_free[Z_A], length N.
///   * `alpha_static` — static per-atom α^A_{ij} tensors (a.u.), length N.
///   * `freqs`,`weights` — imaginary-frequency quadrature (a.u.).
///
/// For each atom:
/// ```text
///   α_iso_eff = ratio · alpha_free[Z]
///   C6_eff    = ratio² · c6_free[Z]
///   ω_A       = (4/3) C6_eff / α_iso_eff²        (single-pole identity)
///   α_iso(iω) = α_iso_eff / (1 + (ω/ω_A)²)
///   α_{ij}(iω) = α_iso(iω) · (α^static_{ij} / α^static_iso)   (shape)
/// ```
///
/// Atoms with Z outside the reference table fall back to using the static
/// tensor's own isotropic average for α_iso_eff with the H London frequency;
/// the result is still finite.
pub fn ts_dynamic_polarizability(
    z: &[usize],
    vol_ratio: &[f64],
    alpha_static: &[[[f64; 3]; 3]],
    freqs: &[f64],
    weights: &[f64],
) -> DynamicPolarizability {
    let natoms = z.len();
    let nfreq = freqs.len();
    let mut per_atom: Vec<Vec<[[f64; 3]; 3]>> =
        vec![vec![[[0.0; 3]; 3]; nfreq]; natoms];

    for a in 0..natoms {
        let st = alpha_static[a];
        let st_iso = (st[0][0] + st[1][1] + st[2][2]) / 3.0;

        let (alpha_iso_eff, c6_eff) = match ts_free_atom(z[a]) {
            Some((alpha_free, c6_free, _vol_free)) => {
                let r = vol_ratio[a];
                (r * alpha_free, r * r * c6_free)
            }
            None => {
                // Fallback: computed static isotropic α with the H London ω.
                let (af_h, c6_h, _) = ts_free_atom(1).unwrap();
                let omega_h = (4.0 / 3.0) * c6_h / (af_h * af_h);
                let a_iso = st_iso.max(1e-6);
                (a_iso, 0.75 * a_iso * a_iso * omega_h)
            }
        };

        let alpha_iso_eff = alpha_iso_eff.max(1e-8);
        let omega_a = (4.0 / 3.0) * c6_eff / (alpha_iso_eff * alpha_iso_eff);

        // Shape factor: static tensor normalized so its iso average is 1.
        let inv_st_iso = if st_iso.abs() > 1e-12 { 1.0 / st_iso } else { 0.0 };
        let mut shape = [[0.0_f64; 3]; 3];
        if inv_st_iso != 0.0 {
            for i in 0..3 {
                for j in 0..3 {
                    shape[i][j] = st[i][j] * inv_st_iso;
                }
            }
        } else {
            // Degenerate static tensor → isotropic shape.
            shape[0][0] = 1.0;
            shape[1][1] = 1.0;
            shape[2][2] = 1.0;
        }

        for (k, &w) in freqs.iter().enumerate() {
            let a_iso = alpha_iso_eff / (1.0 + (w / omega_a).powi(2));
            for i in 0..3 {
                for j in 0..3 {
                    per_atom[a][k][i][j] = a_iso * shape[i][j];
                }
            }
        }
    }

    DynamicPolarizability {
        freqs: freqs.to_vec(),
        weights: weights.to_vec(),
        per_atom,
    }
}

/// PDEP-RPA per-atom dynamic polarizability α^A(iω) (Phase 2 source).
///
/// Evaluates the per-atom polarizability tensors at the imaginary-frequency
/// quadrature nodes drawn from `cfg.quadrature` (the same Gauss-Legendre grid
/// the RPA correlation energy uses), so the resulting `weights` are the exact
/// Casimir-Polder weights for [`casimir_polder_c6`].
///
/// Unlike [`ts_dynamic_polarizability`], this is a genuine frequency-dependent
/// response: at ω=0 it reproduces the static per-atom α exactly, and it carries
/// the true RPA frequency dependence rather than a single-pole London model.
///
/// `partition` selects the atomic decomposition. Becke is the default and the
/// only fully frequency-dependent path today; `Hirshfeld` currently falls back
/// to Becke (a dedicated Hirshfeld-dynamic grid path is future work) and emits
/// no error so callers get a usable result.
#[allow(clippy::too_many_arguments)]
pub fn pdep_dynamic_polarizability(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
    partition: DispersionPartition,
) -> Result<DynamicPolarizability, FerricError> {
    let (freqs, weights) = crate::quadrature::build_quadrature(&cfg.quadrature);

    if partition == DispersionPartition::Hirshfeld {
        // No dedicated Hirshfeld-dynamic path yet; use Becke (geometry-only,
        // robust) for the frequency dependence. Documented, non-fatal.
        eprintln!(
            "note: pdep_dynamic_polarizability: Hirshfeld partition not yet \
             implemented for the dynamic path; using Becke"
        );
    }

    let per_atom = crate::properties::pdep_polarizability_becke_dynamic(
        mol, obs, obs_bs, dfbs, rhf, op, cfg, &freqs,
    )?;

    Ok(DynamicPolarizability {
        freqs,
        weights,
        per_atom,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fine trapezoid grid on [0, ωmax] for analytic Casimir-Polder checks.
    fn trapezoid_grid(n: usize, wmax: f64) -> (Vec<f64>, Vec<f64>) {
        let dw = wmax / (n as f64);
        let mut freqs = Vec::with_capacity(n + 1);
        let mut weights = Vec::with_capacity(n + 1);
        for k in 0..=n {
            freqs.push(k as f64 * dw);
            weights.push(if k == 0 || k == n { 0.5 * dw } else { dw });
        }
        (freqs, weights)
    }

    /// Single-pole isotropic α(iω) = α0/(1+(ω/ω0)²): Casimir-Polder must
    /// reproduce analytic C6 = (3/4) α0² ω0.
    #[test]
    fn casimir_polder_single_pole_analytic() {
        let alpha0 = 4.5_f64;
        let omega0 = 0.5_f64;
        let (freqs, weights) = trapezoid_grid(20000, 200.0);
        let per_atom: Vec<Vec<[[f64; 3]; 3]>> = vec![freqs
            .iter()
            .map(|&w| {
                let a = alpha0 / (1.0 + (w / omega0).powi(2));
                [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
            })
            .collect()];
        let dp = DynamicPolarizability { freqs, weights, per_atom };
        let res = casimir_polder_c6(&dp);
        let c6 = res.c6_iso_pair[(0, 0)];
        let analytic = 0.75 * alpha0 * alpha0 * omega0;
        assert!(
            (c6 - analytic).abs() / analytic < 2e-3,
            "C-P C6={c6} vs analytic={analytic}"
        );
        let tr = (res.c6_aniso_pair[0][0][0][0]
            + res.c6_aniso_pair[0][0][1][1]
            + res.c6_aniso_pair[0][0][2][2])
            / 3.0;
        assert!((tr - c6).abs() / c6 < 1e-12, "aniso trace {tr} != iso {c6}");
    }

    /// TS dynamic α round-trip: C6 from C-P equals the closed-form c6_eff.
    #[test]
    fn ts_dynamic_round_trip_c6() {
        let z = vec![1usize];
        let vol_ratio = vec![1.0_f64];
        let alpha_static = vec![[[4.5, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 4.5]]];
        let (freqs, weights) = trapezoid_grid(20000, 200.0);
        let dp = ts_dynamic_polarizability(&z, &vol_ratio, &alpha_static, &freqs, &weights);
        let res = casimir_polder_c6(&dp);
        let c6 = res.c6_iso_pair[(0, 0)];
        assert!(
            (c6 - 6.5).abs() / 6.5 < 3e-3,
            "TS round-trip C6={c6} vs c6_eff=6.5"
        );
    }

    /// Anisotropy inherited from the static tensor: prolate static → prolate C6.
    #[test]
    fn ts_dynamic_inherits_static_anisotropy() {
        let z = vec![6usize];
        let vol_ratio = vec![1.0_f64];
        let alpha_static = vec![[[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 18.0]]];
        let (freqs, weights) = trapezoid_grid(4000, 100.0);
        let dp = ts_dynamic_polarizability(&z, &vol_ratio, &alpha_static, &freqs, &weights);
        let res = casimir_polder_c6(&dp);
        let czz = res.c6_aniso_pair[0][0][2][2];
        let cxx = res.c6_aniso_pair[0][0][0][0];
        assert!(czz > cxx, "expected prolate C6: zz={czz} xx={cxx}");
    }
}
