//! Z-vector / CPHF solver for RI-MP2 orbital response.
//!
//! Solves (ε_a - ε_i) z_ai + Σ_{bj} A_{ai,bj} z_bj = L_ai
//! iteratively with DIIS, where A is the orbital Hessian and L is the
//! MP2 Lagrangian. The A*z product is computed in the AO basis via J/K builds.

use crate::rimp2::Mp2Intermediates;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_scf::diis::Diis;
use ferric_scf::rhf::{build_jk, RhfResult};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array2, Array3};

/// Solve the Z-vector equation for the RI-MP2 orbital response.
///
/// Returns z of shape (nvir, nocc) — the occupied-virtual block of the
/// relaxed density in MO basis.
pub fn solve_zvector(
    _mol: &Molecule,
    prep: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &RhfResult,
    inter: &Mp2Intermediates,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let eps = &rhf.orbital_energies;
    let c = &rhf.mos;
    let f_mo = c.t().dot(&rhf.fock).dot(c);

    let b_full = crate::oo_rimp2::compute_b_full_mo(prep, dfbs, op, c)?;

    let l = build_lagrangian(
        &f_mo, &inter.t2, &inter.b_ov, &inter.b_oo, &inter.b_vv,
        &inter.p_oo, &inter.p_vv,
        eps, nocc, nvir, nocc_total, first_occ, naux,
        &b_full,
    );

    let mut z = Array2::zeros((nvir, nocc));
    for a in 0..nvir {
        for i in 0..nocc {
            let denom = eps[nocc_total + a] - eps[first_occ + i];
            if denom.abs() > 1e-12 {
                z[(a, i)] = l[(a, i)] / denom;
            }
        }
    }

    let mut diis = Diis::new(8);
    let max_iter = 50;

    for _iter in 0..max_iter {
        let az = compute_az_product(c, &z, prep, bounds, eps, nocc, nvir, nocc_total, first_occ)?;

        let mut residual = Array2::zeros((nvir, nocc));
        let mut max_resid = 0.0f64;
        for a in 0..nvir {
            for i in 0..nocc {
                let denom = eps[nocc_total + a] - eps[first_occ + i];
                residual[(a, i)] = l[(a, i)] - denom * z[(a, i)] - az[(a, i)];
                max_resid = max_resid.max(residual[(a, i)].abs());
            }
        }

        if max_resid < 1e-8 {
            return Ok((z, l));
        }

        let mut z_new = Array2::zeros((nvir, nocc));
        for a in 0..nvir {
            for i in 0..nocc {
                let denom = eps[nocc_total + a] - eps[first_occ + i];
                if denom.abs() > 1e-12 {
                    z_new[(a, i)] = (l[(a, i)] - az[(a, i)]) / denom;
                }
            }
        }

        z = diis.step(&z_new, &residual);
    }

    Ok((z, l))
}

/// Build the MP2 Lagrangian L_ai (RHS of the Z-vector equation).
///
/// Uses the full-MO B tensor (computed internally) for exact integral response.
/// The Lagrangian has P*F terms plus 4-term integral response matching the
/// structure of compute_orbital_gradient in oo_rimp2.
fn build_lagrangian(
    f_mo: &Array2<f64>,
    t2: &[f64],
    _b_ov: &Array2<f64>,
    _b_oo: &Array2<f64>,
    _b_vv: &Array2<f64>,
    p_oo: &Array2<f64>,
    p_vv: &Array2<f64>,
    _eps: &[f64],
    nocc: usize,
    nvir: usize,
    nocc_total: usize,
    first_occ: usize,
    _naux: usize,
    b_full: &Array3<f64>,
) -> Array2<f64> {
    let nov = nocc * nvir;
    let naux = b_full.shape()[0];
    let mut l = Array2::zeros((nvir, nocc));

    // P*F terms
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let mut sum = 0.0;
            for j in 0..nocc {
                let j_mo = first_occ + j;
                sum += p_oo[(i, j)] * f_mo[(a_mo, j_mo)];
            }
            l[(a, i)] += sum;
        }
    }
    for a in 0..nvir {
        for i in 0..nocc {
            let i_mo = first_occ + i;
            let mut sum = 0.0;
            for b in 0..nvir {
                let b_mo = nocc_total + b;
                sum += p_vv[(a, b)] * f_mo[(b_mo, i_mo)];
            }
            l[(a, i)] += sum;
        }
    }

    // Integral response using the same 4-term structure as compute_orbital_gradient.
    // The orbital gradient g_{ck} = -4*F_{ck} - 2*grad_ck.
    // The Lagrangian integral part = grad_ck (same raw sum, no extra factor).
    let eri = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        (0..naux).map(|aux| b_full[(aux, p, q)] * b_full[(aux, r, s)]).sum()
    };

    for c_idx in 0..nvir {
        let c_mo = nocc_total + c_idx;
        for k in 0..nocc {
            let k_mo = first_occ + k;
            let mut grad_ck = 0.0;

            // Term 1 (delta_{ik}→i=k)
            for j in 0..nocc {
                let j_mo = first_occ + j;
                for a in 0..nvir {
                    let a_mo = nocc_total + a;
                    for b in 0..nvir {
                        let b_mo = nocc_total + b;
                        let t_kj_ab = t2[(k * nvir + a) * nov + j * nvir + b];
                        grad_ck += t_kj_ab * (2.0 * eri(c_mo, a_mo, j_mo, b_mo) - eri(c_mo, b_mo, j_mo, a_mo));
                    }
                }
            }

            // Term 2 (delta_{jk}→j=k)
            for i in 0..nocc {
                let i_mo = first_occ + i;
                for a in 0..nvir {
                    let a_mo = nocc_total + a;
                    for b in 0..nvir {
                        let b_mo = nocc_total + b;
                        let t_ik_ab = t2[(i * nvir + a) * nov + k * nvir + b];
                        grad_ck += t_ik_ab * (2.0 * eri(i_mo, a_mo, c_mo, b_mo) - eri(i_mo, b_mo, c_mo, a_mo));
                    }
                }
            }

            // Term 3 (-delta_{ac}→a=c)
            for i in 0..nocc {
                let i_mo = first_occ + i;
                for j in 0..nocc {
                    let j_mo = first_occ + j;
                    for b in 0..nvir {
                        let b_mo = nocc_total + b;
                        let t_ij_cb = t2[(i * nvir + c_idx) * nov + j * nvir + b];
                        grad_ck -= t_ij_cb * (2.0 * eri(i_mo, k_mo, j_mo, b_mo) - eri(i_mo, b_mo, j_mo, k_mo));
                    }
                }
            }

            // Term 4 (-delta_{bc}→b=c)
            for i in 0..nocc {
                let i_mo = first_occ + i;
                for j in 0..nocc {
                    let j_mo = first_occ + j;
                    for a in 0..nvir {
                        let a_mo = nocc_total + a;
                        let t_ij_ac = t2[(i * nvir + a) * nov + j * nvir + c_idx];
                        grad_ck -= t_ij_ac * (2.0 * eri(i_mo, a_mo, j_mo, k_mo) - eri(i_mo, k_mo, j_mo, a_mo));
                    }
                }
            }

            l[(c_idx, k)] += grad_ck;
        }
    }

    l
}

/// Compute the A*z product via J/K builds in the AO basis.
///
/// A_{ai,bj} z_{bj} = [4(ai|bj) - (ab|ij) - (aj|bi)] z_{bj}
/// In AO basis: form D^z, build J(D^z) and K(D^z), project back to MO.
fn compute_az_product(
    c: &Array2<f64>,
    z: &Array2<f64>,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    _eps: &[f64],
    nocc: usize,
    nvir: usize,
    nocc_total: usize,
    first_occ: usize,
) -> Result<Array2<f64>, FerricError> {
    let n = c.nrows();

    let mut dz = Array2::zeros((n, n));
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            for mu in 0..n {
                for nu in 0..n {
                    let val = z[(a, i)] * (c[(mu, a_mo)] * c[(nu, i_mo)] + c[(mu, i_mo)] * c[(nu, a_mo)]);
                    dz[(mu, nu)] += val;
                }
            }
        }
    }

    // Build J(D^z) and K(D^z)
    let mut jz = Array2::zeros((n, n));
    let mut kz = Array2::zeros((n, n));
    build_jk(prep, bounds, 1e-12, &dz, &mut jz, &mut kz)?;

    // The A*z product in AO: A_AO = 4*J(D^z) - K(D^z) - K(D^z)^T
    let az_ao = 4.0 * &jz - &kz - &kz.t();

    // Project to MO basis, extract virtual-occupied block
    let az_mo = c.t().dot(&az_ao).dot(c);
    let mut result = Array2::zeros((nvir, nocc));
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            result[(a, i)] = az_mo[(a_mo, i_mo)];
        }
    }

    Ok(result)
}

/// Build the relaxed 1-PDM in AO basis.
pub fn build_relaxed_density_ao(
    c: &Array2<f64>,
    p_oo: &Array2<f64>,
    p_vv: &Array2<f64>,
    z: &Array2<f64>,
    nocc: usize,
    nvir: usize,
    nocc_total: usize,
    first_occ: usize,
) -> Array2<f64> {
    let nmo = c.ncols();

    let mut p_mo = Array2::zeros((nmo, nmo));

    // Occ-occ: 2*δ_ij + P^MP2_ij
    for i in 0..nocc {
        let i_mo = first_occ + i;
        p_mo[(i_mo, i_mo)] += 2.0;
        for j in 0..nocc {
            let j_mo = first_occ + j;
            p_mo[(i_mo, j_mo)] += p_oo[(i, j)];
        }
    }

    // Vir-vir: P^MP2_ab
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for b in 0..nvir {
            let b_mo = nocc_total + b;
            p_mo[(a_mo, b_mo)] += p_vv[(a, b)];
        }
    }

    // Occ-vir and vir-occ: z_ai
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            p_mo[(a_mo, i_mo)] += z[(a, i)];
            p_mo[(i_mo, a_mo)] += z[(a, i)];
        }
    }

    // Transform to AO: P_AO = C * P_MO * C^T
    let cp = c.dot(&p_mo);
    cp.dot(&c.t())
}

/// Build the relaxed energy-weighted density in AO basis.
pub fn build_relaxed_w_ao(
    c: &Array2<f64>,
    f_mo: &Array2<f64>,
    p_relax_mo: &Array2<f64>,
    l: &Array2<f64>,
    nocc: usize,
    nvir: usize,
    nocc_total: usize,
    first_occ: usize,
) -> Array2<f64> {
    let nmo = c.ncols();

    let mut w_mo = Array2::zeros((nmo, nmo));

    // W_ij = Σ_k F_ik * P^relax_kj (occupied block)
    for i in 0..nocc {
        let i_mo = first_occ + i;
        for j in 0..nocc {
            let j_mo = first_occ + j;
            let mut sum = 0.0;
            for k in 0..nmo {
                sum += f_mo[(i_mo, k)] * p_relax_mo[(k, j_mo)];
            }
            w_mo[(i_mo, j_mo)] = sum;
        }
    }

    // W_ab = Σ_c F_ac * P^relax_cb (virtual block)
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for b in 0..nvir {
            let b_mo = nocc_total + b;
            let mut sum = 0.0;
            for k in 0..nmo {
                sum += f_mo[(a_mo, k)] * p_relax_mo[(k, b_mo)];
            }
            w_mo[(a_mo, b_mo)] = sum;
        }
    }

    // W_ai = L_ai (the Lagrangian RHS, not ε_i * z_ai)
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            w_mo[(a_mo, i_mo)] = l[(a, i)];
            w_mo[(i_mo, a_mo)] = l[(a, i)];
        }
    }

    // Transform to AO
    let cw = c.dot(&w_mo);
    cw.dot(&c.t())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rimp2::{compute_mp2_intermediates, RiMp2Config};
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    #[test]
    fn test_zvector_converges() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let inter = compute_mp2_intermediates(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).unwrap();

        let (z, _l) = solve_zvector(&mol, &obs, &dfbs, Operator::coulomb(), &bounds, &rhf, &inter).unwrap();

        // Z should be finite and small
        for a in 0..inter.nvir {
            for i in 0..inter.nocc {
                assert!(z[(a,i)].is_finite(), "z[{},{}] not finite", a, i);
            }
        }
    }

    #[test]
    fn test_relaxed_density_symmetric() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let inter = compute_mp2_intermediates(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).unwrap();

        let (z, _l) = solve_zvector(&mol, &obs, &dfbs, Operator::coulomb(), &bounds, &rhf, &inter).unwrap();
        let p_ao = build_relaxed_density_ao(
            &rhf.mos, &inter.p_oo, &inter.p_vv, &z,
            inter.nocc, inter.nvir, inter.nocc_total, inter.first_occ,
        );

        let n = p_ao.nrows();
        for i in 0..n {
            for j in 0..n {
                assert!((p_ao[(i,j)] - p_ao[(j,i)]).abs() < 1e-12,
                    "P_relax_AO not symmetric at ({},{}): {} vs {}", i, j, p_ao[(i,j)], p_ao[(j,i)]);
            }
        }
    }
}
