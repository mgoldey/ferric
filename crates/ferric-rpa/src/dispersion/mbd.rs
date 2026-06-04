//! Many-Body Dispersion (MBD@TS): coupled-dipole screening of the TS per-atom
//! polarizabilities. See docs/superpowers/specs/2026-06-04-mbd-screened-c6-design.md.

use crate::dispersion::free_atom_ref::ts_free_atom;
use ndarray::Array2;

/// erf via a rational approximation (Abramowitz & Stegun 7.1.26, |err| < 1.5e-7).
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t
            - 0.284_496_736)
            * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

/// 3N×3N damped dipole–dipole coupling tensor T (MBD@TS).
///
/// Block (A,B), A≠B: T_AB^{ij} = damping · (3 n_i n_j − δ_ij)/r³, n = r̂_AB, with
/// the standard Gaussian-overlap (error-function) damping of two QHO dipoles of
/// combined width σ_AB = √(σ_A²+σ_B²) (Tkatchenko et al., PRL 108, 236402, 2012):
///   ζ = erf(u) − (2u/√π) e^{−u²},   η = (4u³/(3√π)) e^{−u²},   u = r/σ_AB
///   T_ij = ζ·(3 n_i n_j − δ_ij)/r³ − η·(3 n_i n_j)/r³.
/// On-site blocks (A=A) are zero. Symmetric.
pub fn dipole_coupling_tensor(positions: &[[f64; 3]], sigma: &[f64]) -> Array2<f64> {
    let n = positions.len();
    let mut t = Array2::<f64>::zeros((3 * n, 3 * n));
    const SQRT_PI: f64 = 1.772_453_850_905_516;
    for a in 0..n {
        for b in (a + 1)..n {
            let d = [
                positions[b][0] - positions[a][0],
                positions[b][1] - positions[a][1],
                positions[b][2] - positions[a][2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let r = r2.sqrt();
            if r < 1e-8 {
                continue;
            }
            let r3 = r2 * r;
            let nvec = [d[0] / r, d[1] / r, d[2] / r];
            let sab = (sigma[a] * sigma[a] + sigma[b] * sigma[b]).sqrt();
            let u = r / sab;
            let e = (-u * u).exp();
            let zeta = erf(u) - (2.0 * u / SQRT_PI) * e;
            let eta = (4.0 * u * u * u / (3.0 * SQRT_PI)) * e;
            for i in 0..3 {
                for j in 0..3 {
                    let kron = if i == j { 1.0 } else { 0.0 };
                    let bare = (3.0 * nvec[i] * nvec[j] - kron) / r3;
                    let extra = 3.0 * nvec[i] * nvec[j] / r3;
                    let val = zeta * bare - eta * extra;
                    t[(3 * a + i, 3 * b + j)] = val;
                    t[(3 * b + j, 3 * a + i)] = val; // symmetric
                }
            }
        }
    }
    t
}

/// Per-atom TS parameters: (α_eff, ω_A) for each atom.
///
/// α_eff = (volume ratio) · α_free; C6_eff = ratio²·C6_free; ω_A = (4/3)C6_eff/α_eff².
/// For Z outside the table, falls back to the static isotropic α with the H
/// London frequency (matches `ts_dynamic_polarizability`'s fallback).
pub fn ts_atom_params(
    z: &[usize],
    vol_ratio: &[f64],
    alpha_static: &[[[f64; 3]; 3]],
) -> Vec<(f64, f64)> {
    z.iter()
        .enumerate()
        .map(|(a, &za)| {
            let st = alpha_static[a];
            let st_iso = (st[0][0] + st[1][1] + st[2][2]) / 3.0;
            let (alpha_eff, c6_eff) = match ts_free_atom(za) {
                Some((alpha_free, c6_free, _)) => {
                    let r = vol_ratio[a];
                    (r * alpha_free, r * r * c6_free)
                }
                None => {
                    let (af_h, c6_h, _) = ts_free_atom(1).unwrap();
                    let omega_h = (4.0 / 3.0) * c6_h / (af_h * af_h);
                    let a_iso = st_iso.max(1e-6);
                    (a_iso, 0.75 * a_iso * a_iso * omega_h)
                }
            };
            let alpha_eff = alpha_eff.max(1e-8);
            let omega_a = (4.0 / 3.0) * c6_eff / (alpha_eff * alpha_eff);
            (alpha_eff, omega_a)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ts_atom_params_free_atom_reproduces_table() {
        // Carbon at ratio=1: α_eff = α_free = 12.0, ω_A = (4/3)·46.6/12² = 0.4315.
        let st = [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]];
        let p = ts_atom_params(&[6], &[1.0], &[st]);
        assert!((p[0].0 - 12.0).abs() < 1e-9, "α_eff = {}", p[0].0);
        let omega_expected = (4.0 / 3.0) * 46.6 / (12.0 * 12.0);
        assert!((p[0].1 - omega_expected).abs() < 1e-9, "ω_A = {}", p[0].1);
    }

    #[test]
    fn dipole_tensor_offsite_decays_and_onsite_zero() {
        // Two atoms on z-axis at R=10 Bohr, widths σ=1.5 each.
        let pos = [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]];
        let sigma = [1.5, 1.5];
        let t = dipole_coupling_tensor(&pos, &sigma);
        assert_eq!(t.dim(), (6, 6));
        // On-site 3×3 blocks are zero.
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(t[(i, j)], 0.0, "on-site block A nonzero at ({i},{j})");
                assert_eq!(t[(3 + i, 3 + j)], 0.0);
            }
        }
        // zz component of the off-site block is O(1/R³) and nonzero (bond axis).
        let tzz = t[(2, 5)].abs();
        assert!(tzz > 1e-4 && tzz < 1e-2, "off-site T_zz out of range: {tzz}");
    }
}
