use ndarray::{Array3, Array4};

/// Compute the Particle-Particle ladder term: L_abij = sum_P B^P_ab * (sum_cd B^P_cd * t_cdij)
/// RI complexity: O(N^5)
pub fn contract_pp_ladder(
    b_ab: &Array3<f64>,
    t2: &Array4<f64>,
) -> Array4<f64> {
    let (naux, nvir, _) = b_ab.dim();
    let (nocc, _, _, _) = t2.dim();
    let mut res = Array4::zeros((nocc, nvir, nocc, nvir));

    // Intermediate: X^P_ij = sum_cd B^P_cd * t_cdij
    // Shape: (naux, nocc, nocc)
    let mut x = Array3::zeros((naux, nocc, nocc));
    for p in 0..naux {
        let b_p = b_ab.slice(ndarray::s![p, .., ..]);
        for i in 0..nocc {
            for j in 0..nocc {
                let t_ij = t2.slice(ndarray::s![i, .., j, ..]);
                // Tr(B^P * T_ij)
                let mut sum = 0.0;
                for c in 0..nvir {
                    for d in 0..nvir {
                        sum += b_p[(c, d)] * t_ij[(c, d)];
                    }
                }
                x[(p, i, j)] = sum;
            }
        }
    }

    // Final contraction: R_iajb = sum_P B^P_ab * X^P_ij
    for i in 0..nocc {
        for j in 0..nocc {
            for p in 0..naux {
                let x_pij = x[(p, i, j)];
                let b_p = b_ab.slice(ndarray::s![p, .., ..]);
                for a in 0..nvir {
                    for b in 0..nvir {
                        res[(i, a, j, b)] += x_pij * b_p[(a, b)];
                    }
                }
            }
        }
    }
    res
}

/// Compute the Hole-Hole ladder term: H_abij = sum_kl (kl|ij) * t_abkl
/// RI complexity: O(N^5)
pub fn contract_hh_ladder(
    b_ij: &Array3<f64>,
    t2: &Array4<f64>,
) -> Array4<f64> {
    let (naux, nocc, _) = b_ij.dim();
    let (_, nvir, _, _) = t2.dim();
    let mut res = Array4::zeros((nocc, nvir, nocc, nvir));

    // Intermediate: Y^P_ab = sum_kl B^P_kl * t_abkl
    // Shape: (naux, nvir, nvir)
    let mut y = Array3::zeros((naux, nvir, nvir));
    for p in 0..naux {
        let b_p = b_ij.slice(ndarray::s![p, .., ..]);
        for a in 0..nvir {
            for b in 0..nvir {
                let mut sum = 0.0;
                for k in 0..nocc {
                    for l in 0..nocc {
                        sum += b_p[(k, l)] * t2[(k, a, l, b)];
                    }
                }
                y[(p, a, b)] = sum;
            }
        }
    }

    // Final contraction: R_iajb = sum_P B^P_ij * Y^P_ab
    for i in 0..nocc {
        for j in 0..nocc {
            for p in 0..naux {
                let b_pij = b_ij[(p, i, j)];
                let y_p = y.slice(ndarray::s![p, .., ..]);
                for a in 0..nvir {
                    for b in 0..nvir {
                        res[(i, a, j, b)] += b_pij * y_p[(a, b)];
                    }
                }
            }
        }
    }
    res
}
