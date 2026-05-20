//! Becke fuzzy atomic weights for atomic partition of space.
//!
//! Becke 1988 (J. Chem. Phys. 88, 2547): each atom A gets a smooth weight
//! function `w^A(r) ∈ [0, 1]` of position, with `Σ_A w^A(r) = 1` everywhere.
//! The weight depends only on **geometry** (atom positions and atomic-size
//! radii) — no electron density required, unlike Hirshfeld.
//!
//! Construction:
//!
//! 1. For each pair (A, B): hyperbolic coordinate
//!    `μ_AB(r) = (r_A − r_B) / R_AB` where `r_X = |r − R_X|`.
//! 2. Bragg-Slater size correction: rescale μ by `a_AB` to bias the
//!    boundary toward the smaller atom.
//! 3. Apply Becke smoothing polynomial three times: `f(f(f(μ)))`,
//!    `f(x) = (3 x − x³) / 2`.
//! 4. Cell function `s_AB = (1 − f(f(f(μ_AB)))) / 2`.
//! 5. Cell function `P_A = Π_{B ≠ A} s_AB`.
//! 6. Normalized weight: `w^A = P_A / Σ_B P_B`.

use ferric_core::mol::Molecule;

/// Bragg-Slater atomic radii in Bohr (Z=1..18). Becke 1988 specifically
/// recommends a slight modification (Becke radii) but Bragg-Slater is the
/// common default and equivalent at the chemical-accuracy level needed
/// for atomic partitioning.
fn bragg_slater_bohr(z: i32) -> f64 {
    let r_a: f64 = match z {
        1 => 0.35,  2 => 0.30,
        3 => 1.45,  4 => 1.05, 5 => 0.85,  6 => 0.70,  7 => 0.65,  8 => 0.60,
        9 => 0.50, 10 => 0.45,
        11 => 1.80, 12 => 1.50, 13 => 1.25, 14 => 1.10, 15 => 1.00, 16 => 1.00,
        17 => 1.00, 18 => 0.71,
        _ => 1.00,
    };
    r_a * 1.8897259886
}

/// Becke smoothing polynomial f(x) = (3x - x³) / 2, applied n times.
/// n = 3 is the standard choice (Becke 1988).
fn becke_smoothing(mu: f64, n_iter: usize) -> f64 {
    let mut x = mu;
    for _ in 0..n_iter {
        x = 0.5 * x * (3.0 - x * x);
    }
    x
}

/// Becke fuzzy weight `w^A(r)` for atom `a_idx` evaluated at position `r`.
///
/// Linear in atom count (Σ_B P_B normalization), quadratic in atom count
/// per atom (Π_{B≠A} s_AB).
pub fn becke_weight(mol: &Molecule, a_idx: usize, r: [f64; 3]) -> f64 {
    let natoms = mol.atoms.len();
    if natoms == 0 {
        return 0.0;
    }
    if natoms == 1 {
        return 1.0;
    }
    // Distances r_X = |r - R_X|.
    let mut r_dists = vec![0.0_f64; natoms];
    for x in 0..natoms {
        let dx = r[0] - mol.atoms[x].x;
        let dy = r[1] - mol.atoms[x].y;
        let dz = r[2] - mol.atoms[x].zpos;
        r_dists[x] = (dx * dx + dy * dy + dz * dz).sqrt();
    }

    // Cell functions P_A = Π_{B ≠ A} s_AB(μ_AB).
    let mut p_cell = vec![1.0_f64; natoms];
    for a in 0..natoms {
        let ra_z = mol.atoms[a].z;
        let r_a_bs = bragg_slater_bohr(ra_z);
        for b in 0..natoms {
            if a == b {
                continue;
            }
            let rb_z = mol.atoms[b].z;
            let r_b_bs = bragg_slater_bohr(rb_z);
            let dx = mol.atoms[a].x - mol.atoms[b].x;
            let dy = mol.atoms[a].y - mol.atoms[b].y;
            let dz = mol.atoms[a].zpos - mol.atoms[b].zpos;
            let r_ab = (dx * dx + dy * dy + dz * dz).sqrt();
            if r_ab < 1e-12 {
                continue; // degenerate; skip
            }
            // Hyperbolic coordinate.
            let mu = (r_dists[a] - r_dists[b]) / r_ab;
            // Bragg-Slater size correction (Becke Eq. A4):
            //   χ = R_A / R_B
            //   u = (χ - 1) / (χ + 1)
            //   a = u / (u² - 1)
            //   |a| ≤ 0.5 clipped
            //   ν = μ + a (1 - μ²)
            let chi = r_a_bs / r_b_bs;
            let u = (chi - 1.0) / (chi + 1.0);
            let mut a_corr = u / (u * u - 1.0);
            if a_corr > 0.5 { a_corr = 0.5; }
            if a_corr < -0.5 { a_corr = -0.5; }
            let nu = mu + a_corr * (1.0 - mu * mu);
            // Apply Becke smoothing 3 times.
            let smoothed = becke_smoothing(nu, 3);
            // Cell function s_AB.
            let s_ab = 0.5 * (1.0 - smoothed);
            p_cell[a] *= s_ab;
        }
    }
    let total: f64 = p_cell.iter().sum();
    if total < 1e-30 {
        return 0.0;
    }
    p_cell[a_idx] / total
}

/// All atom weights at one point (more efficient than calling becke_weight
/// N times, since the cell-function loop is shared).
pub fn becke_weights_all(mol: &Molecule, r: [f64; 3]) -> Vec<f64> {
    let natoms = mol.atoms.len();
    if natoms == 0 {
        return vec![];
    }
    if natoms == 1 {
        return vec![1.0];
    }
    let mut r_dists = vec![0.0_f64; natoms];
    for x in 0..natoms {
        let dx = r[0] - mol.atoms[x].x;
        let dy = r[1] - mol.atoms[x].y;
        let dz = r[2] - mol.atoms[x].zpos;
        r_dists[x] = (dx * dx + dy * dy + dz * dz).sqrt();
    }
    let mut p_cell = vec![1.0_f64; natoms];
    for a in 0..natoms {
        let r_a_bs = bragg_slater_bohr(mol.atoms[a].z);
        for b in 0..natoms {
            if a == b { continue; }
            let r_b_bs = bragg_slater_bohr(mol.atoms[b].z);
            let dx = mol.atoms[a].x - mol.atoms[b].x;
            let dy = mol.atoms[a].y - mol.atoms[b].y;
            let dz = mol.atoms[a].zpos - mol.atoms[b].zpos;
            let r_ab = (dx * dx + dy * dy + dz * dz).sqrt();
            if r_ab < 1e-12 { continue; }
            let mu = (r_dists[a] - r_dists[b]) / r_ab;
            let chi = r_a_bs / r_b_bs;
            let u = (chi - 1.0) / (chi + 1.0);
            let mut a_corr = u / (u * u - 1.0);
            if a_corr > 0.5 { a_corr = 0.5; }
            if a_corr < -0.5 { a_corr = -0.5; }
            let nu = mu + a_corr * (1.0 - mu * mu);
            let smoothed = becke_smoothing(nu, 3);
            let s_ab = 0.5 * (1.0 - smoothed);
            p_cell[a] *= s_ab;
        }
    }
    let total: f64 = p_cell.iter().sum();
    if total < 1e-30 {
        return vec![0.0; natoms];
    }
    p_cell.iter().map(|p| p / total).collect()
}

/// Becke fuzzy weights and their **lab-fixed-r** nuclear gradients.
///
/// Returns `(weights, dw)` where:
/// - `weights[a]` is `w^a(r)` (matches `becke_weights_all`)
/// - `dw[a][b][α]` is `∂w^a(r)/∂R_b^α` with `r` held fixed in the lab frame.
///
/// Used by the XC nuclear-gradient grid-response correction (P2.1). The
/// full PySCF "weight1" convention (which includes the home-translation
/// `∇_r w` chain-rule piece) is built on top in
/// `crate::grid::build_atomic_grid_with_response` via the
/// translational-invariance identity `Σ_c ∂w/∂R_c|_{r fixed} + ∇_r w = 0`.
///
/// Cost is O(natoms² + natoms³) per call (the pair caches + the response
/// accumulation); dominated by the AO derivative work in practice.
///
/// Derivation. With `μ_AB = (r_A − r_B) / R_AB`, `r_X = |r − R_X|`,
/// `R_AB = |R_A − R_B|`, the size-corrected coordinate is
/// `ν_AB = μ_AB + a_AB (1 − μ_AB²)` and the smoothed step is
/// `s_AB = (1 − f³(ν_AB)) / 2`. The cell function `P_A = Π_{B≠A} s_AB`,
/// and `w^A = P_A / T` with `T = Σ_C P_C`. Then
///
/// ```text
///   ∂μ_AB/∂R_C^α =
///     +δ_{CA} · [ −(r−R_A)^α / (r_A · R_AB)
///                 − (μ_AB / R_AB²) · (R_A − R_B)^α ]
///     +δ_{CB} · [ +(r−R_B)^α / (r_B · R_AB)
///                 + (μ_AB / R_AB²) · (R_A − R_B)^α ]
///   ∂ν/∂μ = 1 − 2 a_AB μ_AB
///   ∂f³/∂ν = f₂'·f₁'·f₀'  with fₖ' = 1.5·(1 − fₖ²) at the kth iterate.
///   ∂s_AB/∂R_C = −0.5 · (∂f³/∂ν) · (∂ν/∂μ) · ∂μ/∂R_C
/// ```
pub fn becke_weights_and_grad(
    mol: &Molecule,
    r: [f64; 3],
) -> (Vec<f64>, Vec<Vec<[f64; 3]>>) {
    let natoms = mol.atoms.len();
    if natoms <= 1 {
        let w = if natoms == 1 { vec![1.0] } else { vec![] };
        let dw = vec![vec![[0.0; 3]; natoms]; natoms];
        return (w, dw);
    }

    let mut r_dists = vec![0.0_f64; natoms];
    let mut r_unit = vec![[0.0_f64; 3]; natoms];
    for x in 0..natoms {
        let dx = r[0] - mol.atoms[x].x;
        let dy = r[1] - mol.atoms[x].y;
        let dz = r[2] - mol.atoms[x].zpos;
        let rx = (dx * dx + dy * dy + dz * dz).sqrt();
        r_dists[x] = rx;
        let inv = if rx > 1e-30 { 1.0 / rx } else { 0.0 };
        r_unit[x] = [dx * inv, dy * inv, dz * inv];
    }

    let mut s_pair = vec![1.0_f64; natoms * natoms];
    let mut ds_dra = vec![[0.0_f64; 3]; natoms * natoms];
    let mut ds_drb = vec![[0.0_f64; 3]; natoms * natoms];
    let idx = |a: usize, b: usize| -> usize { a * natoms + b };

    for a in 0..natoms {
        let r_a_bs = bragg_slater_bohr(mol.atoms[a].z);
        for b in 0..natoms {
            if a == b {
                continue;
            }
            let r_b_bs = bragg_slater_bohr(mol.atoms[b].z);
            let dxab = mol.atoms[a].x - mol.atoms[b].x;
            let dyab = mol.atoms[a].y - mol.atoms[b].y;
            let dzab = mol.atoms[a].zpos - mol.atoms[b].zpos;
            let r_ab = (dxab * dxab + dyab * dyab + dzab * dzab).sqrt();
            if r_ab < 1e-12 {
                continue;
            }
            let inv_r_ab = 1.0 / r_ab;
            let r_ab_vec = [dxab, dyab, dzab];

            let mu = (r_dists[a] - r_dists[b]) * inv_r_ab;

            let chi = r_a_bs / r_b_bs;
            let u = (chi - 1.0) / (chi + 1.0);
            let mut a_corr = u / (u * u - 1.0);
            if a_corr > 0.5 {
                a_corr = 0.5;
            }
            if a_corr < -0.5 {
                a_corr = -0.5;
            }
            let nu = mu + a_corr * (1.0 - mu * mu);
            let dnu_dmu = 1.0 - 2.0 * a_corr * mu;

            let f0 = nu;
            let f0p = 1.5 * (1.0 - f0 * f0);
            let f1 = 0.5 * f0 * (3.0 - f0 * f0);
            let f1p = 1.5 * (1.0 - f1 * f1);
            let f2 = 0.5 * f1 * (3.0 - f1 * f1);
            let f2p = 1.5 * (1.0 - f2 * f2);
            let f3 = 0.5 * f2 * (3.0 - f2 * f2);
            let df3_dnu = f2p * f1p * f0p;

            let s_ab = 0.5 * (1.0 - f3);
            let ds_dmu = -0.5 * df3_dnu * dnu_dmu;

            s_pair[idx(a, b)] = s_ab;

            let mut dmu_dra = [0.0_f64; 3];
            let mut dmu_drb = [0.0_f64; 3];
            for k in 0..3 {
                let ab_term = mu * inv_r_ab * inv_r_ab * r_ab_vec[k];
                dmu_dra[k] = -r_unit[a][k] * inv_r_ab - ab_term;
                dmu_drb[k] = r_unit[b][k] * inv_r_ab + ab_term;
            }
            for k in 0..3 {
                ds_dra[idx(a, b)][k] = ds_dmu * dmu_dra[k];
                ds_drb[idx(a, b)][k] = ds_dmu * dmu_drb[k];
            }
        }
    }

    let mut p_cell = vec![1.0_f64; natoms];
    for a in 0..natoms {
        for b in 0..natoms {
            if a == b {
                continue;
            }
            p_cell[a] *= s_pair[idx(a, b)];
        }
    }
    let t: f64 = p_cell.iter().sum();
    if t < 1e-30 {
        return (
            vec![0.0; natoms],
            vec![vec![[0.0; 3]; natoms]; natoms],
        );
    }
    let inv_t = 1.0 / t;
    let weights: Vec<f64> = p_cell.iter().map(|p| p * inv_t).collect();

    let mut dp = vec![vec![[0.0_f64; 3]; natoms]; natoms];
    for a in 0..natoms {
        for b in 0..natoms {
            if a == b {
                continue;
            }
            let s = s_pair[idx(a, b)];
            if s.abs() < 1e-30 {
                continue;
            }
            let ratio = p_cell[a] / s;
            for k in 0..3 {
                dp[a][a][k] += ratio * ds_dra[idx(a, b)][k];
                dp[a][b][k] += ratio * ds_drb[idx(a, b)][k];
            }
        }
    }

    let mut dt = vec![[0.0_f64; 3]; natoms];
    for a in 0..natoms {
        for c in 0..natoms {
            for k in 0..3 {
                dt[c][k] += dp[a][c][k];
            }
        }
    }

    let mut dw = vec![vec![[0.0_f64; 3]; natoms]; natoms];
    for a in 0..natoms {
        for c in 0..natoms {
            for k in 0..3 {
                dw[a][c][k] = inv_t * dp[a][c][k] - p_cell[a] * inv_t * inv_t * dt[c][k];
            }
        }
    }

    (weights, dw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::mol::{Atom, Molecule};

    fn h2_at(d: f64) -> Molecule {
        Molecule {
            atoms: vec![
                Atom { symbol: "H".into(), z: 1, x: -d/2.0, y: 0.0, zpos: 0.0 },
                Atom { symbol: "H".into(), z: 1, x:  d/2.0, y: 0.0, zpos: 0.0 },
            ],
            charge: 0,
            multiplicity: 1,
        }
    }

    #[test]
    fn becke_weights_sum_to_one_h2() {
        let mol = h2_at(1.4);
        // Sample at several positions; weights must sum to 1.
        for r in &[
            [0.0, 0.0, 0.0],
            [-0.7, 0.0, 0.0],
            [0.7, 0.0, 0.0],
            [0.0, 1.0, 0.5],
            [2.0, 2.0, 2.0],
        ] {
            let w = becke_weights_all(&mol, *r);
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12,
                "Becke weights at {:?}: sum {sum}, weights {:?}", r, w);
        }
    }

    #[test]
    fn becke_weight_centered_at_atom_is_one() {
        let mol = h2_at(1.4);
        // At atom A, w_A → 1 (smoothing function saturates at boundary).
        let w0 = becke_weights_all(&mol, [-0.7, 0.0, 0.0]);
        assert!((w0[0] - 1.0).abs() < 1e-6, "Becke w_A at R_A = {} (expect 1)", w0[0]);
        let w1 = becke_weights_all(&mol, [0.7, 0.0, 0.0]);
        assert!((w1[1] - 1.0).abs() < 1e-6, "Becke w_B at R_B = {} (expect 1)", w1[1]);
    }

    #[test]
    fn becke_weight_midpoint_h2_is_half() {
        let mol = h2_at(1.4);
        // At midpoint of homonuclear H2, both weights are 1/2.
        let w = becke_weights_all(&mol, [0.0, 0.0, 0.0]);
        assert!((w[0] - 0.5).abs() < 1e-12, "Becke w midpoint H2: {:?}", w);
        assert!((w[1] - 0.5).abs() < 1e-12, "Becke w midpoint H2: {:?}", w);
    }

    #[test]
    fn becke_weights_and_grad_match_value() {
        let mol = h2_at(1.4);
        let r = [0.3, 0.4, 0.5];
        let w_val = becke_weights_all(&mol, r);
        let (w_g, _) = becke_weights_and_grad(&mol, r);
        for a in 0..2 {
            assert!((w_val[a] - w_g[a]).abs() < 1e-14);
        }
    }

    #[test]
    fn becke_weights_grad_sums_to_zero_h2() {
        let mol = h2_at(1.4);
        for r in &[
            [0.3, 0.4, 0.5_f64],
            [-0.2, 0.6, 0.0],
            [0.7, -0.1, 0.4],
        ] {
            let (_, dw) = becke_weights_and_grad(&mol, *r);
            for c in 0..2 {
                for k in 0..3 {
                    let s: f64 = (0..2).map(|a| dw[a][c][k]).sum();
                    assert!(
                        s.abs() < 1e-12,
                        "Σ_A dw^A/dR_{c}^{k} = {s:.3e} at r={r:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn becke_weights_grad_finite_difference_ch() {
        let r = [0.4_f64, 0.3, 0.2];
        let h = 1e-5;

        let build = |atoms: Vec<Atom>| Molecule {
            atoms,
            charge: 0,
            multiplicity: 1,
        };

        let base = vec![
            Atom { symbol: "C".into(), z: 6, x: 0.0, y: 0.0, zpos: 0.0 },
            Atom { symbol: "H".into(), z: 1, x: 2.0, y: 0.0, zpos: 0.0 },
        ];
        let mol = build(base.clone());
        let (_, dw_ana) = becke_weights_and_grad(&mol, r);

        let mut max_err: f64 = 0.0;
        for c in 0..2 {
            for k in 0..3 {
                let mut atoms_plus = base.clone();
                let mut atoms_minus = base.clone();
                match k {
                    0 => { atoms_plus[c].x += h; atoms_minus[c].x -= h; }
                    1 => { atoms_plus[c].y += h; atoms_minus[c].y -= h; }
                    _ => { atoms_plus[c].zpos += h; atoms_minus[c].zpos -= h; }
                }
                let w_plus = becke_weights_all(&build(atoms_plus), r);
                let w_minus = becke_weights_all(&build(atoms_minus), r);
                for a in 0..2 {
                    let fd = (w_plus[a] - w_minus[a]) / (2.0 * h);
                    let ana = dw_ana[a][c][k];
                    let diff = (fd - ana).abs();
                    if diff > max_err { max_err = diff; }
                }
            }
        }
        eprintln!("Becke grad max |ana − FD| = {max_err:.3e}");
        assert!(max_err < 1e-7, "Becke grad FD mismatch = {max_err:.3e}");
    }

    #[test]
    fn becke_size_correction_biases_toward_smaller_atom() {
        // CH₄-like atom pair: C and H. The boundary should be shifted
        // toward the smaller H (R_BS: C=0.70 Å, H=0.35 Å).
        let mol = Molecule {
            atoms: vec![
                Atom { symbol: "C".into(), z: 6, x: 0.0, y: 0.0, zpos: 0.0 },
                Atom { symbol: "H".into(), z: 1, x: 2.0, y: 0.0, zpos: 0.0 },
            ],
            charge: 0,
            multiplicity: 1,
        };
        // Midpoint: x=1.0. Without size correction this would give w_C = w_H = 0.5.
        // With size correction toward smaller H, w_C should exceed 0.5.
        let w = becke_weights_all(&mol, [1.0, 0.0, 0.0]);
        assert!(w[0] > 0.5,
            "C-H midpoint Becke: w_C should exceed 0.5 from size correction, got w_C={}", w[0]);
    }
}
