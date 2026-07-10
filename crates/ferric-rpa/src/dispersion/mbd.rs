//! Many-Body Dispersion (MBD@TS): coupled-dipole screening of the TS per-atom
//! polarizabilities. See docs/superpowers/specs/2026-06-04-mbd-screened-c6-design.md.

use crate::dispersion::free_atom_ref::ts_free_atom;
use crate::dispersion::DynamicPolarizability;
use ferric_core::FerricError;
use ndarray::Array2;
use ndarray_linalg::Inverse;

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
/// Block (A,B), A≠B: T_AB^{ij} = damping · (δ_ij − 3 n_i n_j)/r³, n = r̂_AB, with
/// the sign convention that pairs with the screening equation A_scs⁻¹ = α⁻¹ + T
/// (parallel dipoles along the bond couple attractively → enhanced bond-axis α).
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
                    let bare = (kron - 3.0 * nvec[i] * nvec[j]) / r3;
                    let extra = 3.0 * nvec[i] * nvec[j] / r3;
                    let val = zeta * bare + eta * extra;
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
///
/// # Errors
///
/// Returns [`FerricError::General`] naming the atom index and Z when no
/// free-atom TS reference exists for that element (the table covers Z=1..=18).
/// This is a HARD error, not a silent fallback: the previous behaviour
/// substituted hydrogen's London frequency for Z>18, which silently produced
/// H-like (wrong) C6 for every 4th-row / heavy-atom system. TS is not
/// parameterised beyond Z=18; there is no honest value to return, so the caller
/// must be told rather than handed plausible-shaped garbage.
pub fn ts_atom_params(
    z: &[usize],
    vol_ratio: &[f64],
    alpha_static: &[[[f64; 3]; 3]],
) -> Result<Vec<(f64, f64)>, FerricError> {
    z.iter()
        .enumerate()
        .map(|(a, &za)| {
            let (alpha_free, c6_free, _) = ts_free_atom(za).ok_or_else(|| {
                FerricError::General(format!(
                    "TS dispersion: no free-atom reference for atom {a} (Z={za}); \
                     the Tkatchenko-Scheffler table covers Z=1..=18 only. Heavy-atom \
                     TS/MBD C6 is not parameterised — refusing to substitute hydrogen's \
                     London frequency (which would silently yield H-like C6). Use the \
                     PDEP-RPA C6 source (c6_source=\"pdep\") for Z>18 instead."
                ))
            })?;
            let _ = alpha_static; // static tensor drives shape, not (α_eff, ω_A).
            let r = vol_ratio[a];
            let alpha_eff = (r * alpha_free).max(1e-8);
            let c6_eff = r * r * c6_free;
            let omega_a = (4.0 / 3.0) * c6_eff / (alpha_eff * alpha_eff);
            Ok((alpha_eff, omega_a))
        })
        .collect()
}

/// MBD@TS screened per-atom α(iω). For each frequency, builds the coupled matrix
/// C = A⁻¹ + T (A = block-diagonal per-atom α(iω), T = damped dipole tensor with
/// per-atom Gaussian widths σ_A = (√(2/π)·α_A(iω)/3)^{1/3}), inverts it, and
/// contracts each atom's row of blocks back to a per-atom 3×3 tensor:
///   α_A^scs = Σ_B (C⁻¹)_{AB}.
/// Returns the same shape as the input (`[atom][freq][3][3]`).
pub fn mbd_screen(
    per_atom_alpha: &[Vec<[[f64; 3]; 3]>],
    positions: &[[f64; 3]],
    _alpha_eff: &[f64],
    freqs: &[f64],
) -> Vec<Vec<[[f64; 3]; 3]>> {
    let n = positions.len();
    let nfreq = freqs.len();
    let mut out: Vec<Vec<[[f64; 3]; 3]>> = vec![vec![[[0.0; 3]; 3]; nfreq]; n];
    use std::f64::consts::FRAC_2_PI;
    for k in 0..nfreq {
        let mut a_inv = Array2::<f64>::zeros((3 * n, 3 * n));
        let mut sigma = vec![0.0_f64; n];
        for a in 0..n {
            let t = per_atom_alpha[a][k];
            let iso = ((t[0][0] + t[1][1] + t[2][2]) / 3.0).max(1e-10);
            sigma[a] = (FRAC_2_PI.sqrt() * iso / 3.0).powf(1.0 / 3.0);
            // MBD@rsSCS uses a single ISOTROPIC QHO per atom (Ambrosetti 2014
            // Eq. 6: α0_lm = δ_lm α^TS). Inverting the full anisotropic Becke
            // per-atom tensor (some near-singular) distorts the screening; use
            // the isotropic TS magnitude as the reference prescribes.
            let inv_iso = 1.0 / iso;
            for i in 0..3 {
                a_inv[(3 * a + i, 3 * a + i)] = inv_iso;
            }
        }
        let tmat = dipole_coupling_tensor(positions, &sigma);
        let c = &a_inv + &tmat;
        let cinv = c.inv().unwrap_or_else(|_| Array2::eye(3 * n));
        for a in 0..n {
            for b in 0..n {
                for i in 0..3 {
                    for j in 0..3 {
                        out[a][k][i][j] += cinv[(3 * a + i, 3 * b + j)];
                    }
                }
            }
        }
    }
    out
}

/// Build a `DynamicPolarizability` from MBD-screened per-atom α(iω). Drop-in
/// alternative to `ts_dynamic_polarizability` for `casimir_polder_c6`.
pub fn mbd_dynamic_polarizability(
    positions: &[[f64; 3]],
    z: &[usize],
    vol_ratio: &[f64],
    alpha_static: &[[[f64; 3]; 3]],
    freqs: &[f64],
    weights: &[f64],
) -> Result<DynamicPolarizability, FerricError> {
    let ts = crate::dispersion::ts_dynamic_polarizability(
        z,
        vol_ratio,
        alpha_static,
        freqs,
        weights,
    )?;
    let params = ts_atom_params(z, vol_ratio, alpha_static)?;
    let alpha_eff: Vec<f64> = params.iter().map(|p| p.0).collect();
    let screened = mbd_screen(&ts.per_atom, positions, &alpha_eff, freqs);
    let nfreq = freqs.len();
    let molecular: Vec<[[f64; 3]; 3]> = (0..nfreq)
        .map(|k| {
            let mut m = [[0.0; 3]; 3];
            for at in &screened {
                for i in 0..3 {
                    for j in 0..3 {
                        m[i][j] += at[k][i][j];
                    }
                }
            }
            m
        })
        .collect();
    Ok(DynamicPolarizability {
        freqs: freqs.to_vec(),
        weights: weights.to_vec(),
        per_atom: screened,
        molecular,
    })
}

/// MBD@TS coupled-plasmon dispersion energy (validation path).
///
/// E_MBD = ½ Σ_p √λ_p − (3/2) Σ_A ω_A, where λ_p are eigenvalues of the
/// 3N×3N coupled QHO matrix
///   H_{Ai,Bj} = ω_A² δ_{AB} δ_{ij} + ω_A ω_B √(α_A α_B) · T^{damp}_{Ai,Bj},
/// with static α used for both the widths and the prefactor (standard MBD@TS).
pub fn mbd_energy(positions: &[[f64; 3]], alpha_eff: &[f64], omega_a: &[f64]) -> f64 {
    use ndarray_linalg::Eigh;
    use std::f64::consts::FRAC_2_PI;
    let n = positions.len();
    let sigma: Vec<f64> = alpha_eff
        .iter()
        .map(|&a| (FRAC_2_PI.sqrt() * a / 3.0).powf(1.0 / 3.0))
        .collect();
    let t = dipole_coupling_tensor(positions, &sigma);
    let mut h = Array2::<f64>::zeros((3 * n, 3 * n));
    for a in 0..n {
        for i in 0..3 {
            h[(3 * a + i, 3 * a + i)] += omega_a[a] * omega_a[a];
        }
    }
    for a in 0..n {
        for b in 0..n {
            let pref = omega_a[a] * omega_a[b] * (alpha_eff[a] * alpha_eff[b]).sqrt();
            for i in 0..3 {
                for j in 0..3 {
                    h[(3 * a + i, 3 * b + j)] += pref * t[(3 * a + i, 3 * b + j)];
                }
            }
        }
    }
    let (evals, _) = h.eigh(ndarray_linalg::UPLO::Upper).unwrap();
    let coupled: f64 = evals.iter().map(|&l| l.max(0.0).sqrt()).sum::<f64>() * 0.5;
    let uncoupled: f64 = omega_a.iter().sum::<f64>() * 1.5;
    coupled - uncoupled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ts_atom_params_free_atom_reproduces_table() {
        // Carbon at ratio=1: α_eff = α_free = 12.0, ω_A = (4/3)·46.6/12² = 0.4315.
        let st = [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]];
        let p = ts_atom_params(&[6], &[1.0], &[st]).unwrap();
        assert!((p[0].0 - 12.0).abs() < 1e-9, "α_eff = {}", p[0].0);
        let omega_expected = (4.0 / 3.0) * 46.6 / (12.0 * 12.0);
        assert!((p[0].1 - omega_expected).abs() < 1e-9, "ω_A = {}", p[0].1);
    }

    /// Z outside the TS table (Z=1..=18) must HARD-ERROR from `ts_atom_params`,
    /// not fall back to hydrogen's ω. Names the atom index and Z.
    #[test]
    fn ts_atom_params_heavy_atom_errors() {
        let st = [[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]];
        // Two atoms: C (ok) then Br (Z=35, out of table) — error must point at idx 1.
        let err = ts_atom_params(&[6, 35], &[1.0, 1.0], &[st, st])
            .expect_err("Z=35 must error");
        let msg = format!("{err}");
        assert!(msg.contains("Z=35"), "error must name Z: {msg}");
        assert!(msg.contains("atom 1"), "error must name the atom index: {msg}");
    }

    /// Z > 18 regression: the old code silently substituted the molecular α
    /// with HYDROGEN's characteristic frequency, fabricating a TS C6 for Br,
    /// Kr, and every heavier element. It must now be a hard error through
    /// every public entry point, including the MBD-screened path (which has
    /// its own `ts_atom_params` call independent of `ts_dynamic_polarizability`).
    #[test]
    fn ts_unparameterized_element_errors_instead_of_hydrogen_omega() {
        let err = ts_atom_params(&[1, 35], &[1.0, 1.0], &[[[0.0; 3]; 3]; 2])
            .unwrap_err()
            .to_string();
        assert!(err.contains("Br") || err.contains("35"), "error should name Br: {err}");

        let st = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]];
        let freqs = [0.0, 0.5];
        let weights = [0.5, 0.5];
        assert!(crate::dispersion::ts_dynamic_polarizability(
            &[35], &[1.0], &[st], &freqs, &weights
        ).is_err());
        assert!(mbd_dynamic_polarizability(
            &[[0.0, 0.0, 0.0]], &[35], &[1.0], &[st], &freqs, &weights
        ).is_err());
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

    #[test]
    fn mbd_screen_large_separation_recovers_unscreened() {
        // Two He-like atoms at R=50 Bohr (T→0): screened α ≈ input α.
        let pos = [[0.0, 0.0, 0.0], [0.0, 0.0, 50.0]];
        let alpha_eff = [1.38_f64, 1.38];
        let freqs = [0.0_f64, 0.5, 2.0];
        let omega = (4.0_f64 / 3.0) * 1.46 / (1.38 * 1.38);
        let mk = |k: usize| {
            let w = freqs[k];
            let a = 1.38 / (1.0 + (w / omega).powi(2));
            [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
        };
        let input: Vec<Vec<[[f64; 3]; 3]>> =
            (0..2).map(|_| (0..3).map(mk).collect()).collect();
        let scr = mbd_screen(&input, &pos, &alpha_eff, &freqs);
        for a in 0..2 {
            for k in 0..3 {
                let in_iso =
                    (input[a][k][0][0] + input[a][k][1][1] + input[a][k][2][2]) / 3.0;
                let sc_iso = (scr[a][k][0][0] + scr[a][k][1][1] + scr[a][k][2][2]) / 3.0;
                assert!(
                    (in_iso - sc_iso).abs() < 1e-3 * in_iso.max(1e-3),
                    "screened≠unscreened at large R: in={in_iso} sc={sc_iso}"
                );
            }
        }
    }

    #[test]
    fn mbd_screening_changes_c6_and_is_finite() {
        use crate::dispersion::{casimir_polder_c6, ts_dynamic_polarizability};
        // Two carbons at 4 Bohr: screening must change the molecular C6 (coupling
        // is active) and the result must be finite & positive. The SIGN of the
        // change is direction-dependent (parallel-bond dipoles enhance,
        // perpendicular screen); the per-direction behaviour is checked separately.
        let z = [6usize, 6];
        let pos = [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]];
        let vr = [1.0, 1.0];
        let st = [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]];
        let alpha_static = [st, st];
        let freqs: Vec<f64> = (0..12).map(|i| 0.1 * i as f64).collect();
        let weights = vec![1.0; freqs.len()];
        let ts = ts_dynamic_polarizability(&z, &vr, &alpha_static, &freqs, &weights).unwrap();
        let mbd =
            mbd_dynamic_polarizability(&pos, &z, &vr, &alpha_static, &freqs, &weights).unwrap();
        let c6_ts = casimir_polder_c6(&ts).c6_molecular_iso;
        let c6_mbd = casimir_polder_c6(&mbd).c6_molecular_iso;
        assert!(c6_mbd.is_finite() && c6_mbd > 0.0, "MBD C6 not finite/positive: {c6_mbd}");
        assert!((c6_mbd - c6_ts).abs() > 0.01 * c6_ts, "screening had no effect: {c6_ts} vs {c6_mbd}");
    }

    #[test]
    fn mbd_screening_direction_resolved_signs() {
        // Two atoms on the z-axis. The screened per-atom α should be ENHANCED
        // along the bond (zz, attractive parallel dipoles) and SCREENED
        // perpendicular (xx, repulsive antiparallel) relative to the bare α.
        // This is the textbook directional signature of dipole screening and
        // pins the sign convention of T.
        let pos = [[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]];
        let alpha_eff = [8.0_f64, 8.0];
        let freqs = [0.0_f64];
        let a0 = 8.0_f64;
        let iso = [[a0, 0.0, 0.0], [0.0, a0, 0.0], [0.0, 0.0, a0]];
        let input: Vec<Vec<[[f64; 3]; 3]>> = vec![vec![iso], vec![iso]];
        let scr = mbd_screen(&input, &pos, &alpha_eff, &freqs);
        let xx = scr[0][0][0][0];
        let zz = scr[0][0][2][2];
        assert!(zz > a0, "bond-parallel αzz should be enhanced: {zz} vs {a0}");
        assert!(xx < a0, "perpendicular αxx should be screened: {xx} vs {a0}");
    }

    #[test]
    fn mbd_energy_two_atoms_negative_and_decays() {
        // Two Ar-like oscillators: E_disp < 0, and |E| shrinks with R.
        let ae = [11.1_f64, 11.1];
        let wa = [(4.0_f64 / 3.0) * 64.3 / (11.1 * 11.1); 2];
        let e_near = mbd_energy(&[[0.0, 0.0, 0.0], [0.0, 0.0, 7.0]], &ae, &wa);
        let e_far = mbd_energy(&[[0.0, 0.0, 0.0], [0.0, 0.0, 14.0]], &ae, &wa);
        assert!(e_near < 0.0, "E_disp should be negative: {e_near}");
        assert!(
            e_far.abs() < e_near.abs(),
            "E should decay with R: {e_near} {e_far}"
        );
    }

    #[test]
    fn mbd_energy_recovers_london_c6_at_large_r() {
        // The coupled-plasmon energy must reduce to the pairwise London −C6/R⁶ at
        // large separation (the code-correctness anchor for the energy path).
        let alpha = 11.1_f64;
        let omega = (4.0_f64 / 3.0) * 64.3 / (alpha * alpha);
        let r = 20.0_f64;
        let e_mbd = mbd_energy(&[[0.0, 0.0, 0.0], [0.0, 0.0, r]], &[alpha, alpha], &[omega, omega]);
        let c6 = 0.75 * alpha * alpha * omega; // single-pole London C6
        let e_pair = -c6 / r.powi(6);
        let rel = (e_mbd - e_pair).abs() / e_pair.abs();
        assert!(
            rel < 0.10,
            "MBD vs London C6/R⁶ at R=20: mbd={e_mbd} pair={e_pair} rel={rel}"
        );
    }
}
