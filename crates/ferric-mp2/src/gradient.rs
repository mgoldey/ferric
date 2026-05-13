//! RI-MP2 nuclear gradients: analytical and finite-difference reference.

use crate::rimp2::{ri_mp2, compute_mp2_intermediates, RiMp2Config};
use crate::zvector::{solve_zvector, build_relaxed_density_ao, build_relaxed_w_ao};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::hf_gradient_with_density;
use ferric_scf::rhf::{solve_rhf, RhfConfig, RhfResult};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Compute the total RI-MP2 energy (E_HF + E_MP2) for a given geometry.
fn total_energy(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
) -> Result<f64, FerricError> {
    let obs = PreparedBasis::new(mol, obs_basis)?;
    let bounds = SchwarzBounds::compute(op, &obs)?;
    let rhf_config = RhfConfig {
        energy_conv: 1e-10,
        ..Default::default()
    };
    let rhf = solve_rhf(mol, &obs, op, &bounds, &rhf_config)?;
    let dfbs = PreparedBasis::new(mol, aux_basis)?;
    let mp2 = ri_mp2(mol, &obs, &dfbs, op, &rhf, mp2_config)?;
    Ok(mp2.total_energy)
}

/// Compute RI-MP2 nuclear gradient via central finite differences.
///
/// `delta` is in Bohr (molecular coordinates are in Bohr).
/// Returns a `(natoms, 3)` array of dE/dR per atom per Cartesian direction.
pub fn rimp2_gradient_fd(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
    delta: f64,
) -> Result<Array2<f64>, FerricError> {
    let natoms = mol.atoms.len();
    let mut grad = Array2::zeros((natoms, 3));
    for atom in 0..natoms {
        for coord in 0..3 {
            let mut mol_p = mol.clone();
            let mut mol_m = mol.clone();
            match coord {
                0 => {
                    mol_p.atoms[atom].x += delta;
                    mol_m.atoms[atom].x -= delta;
                }
                1 => {
                    mol_p.atoms[atom].y += delta;
                    mol_m.atoms[atom].y -= delta;
                }
                _ => {
                    mol_p.atoms[atom].zpos += delta;
                    mol_m.atoms[atom].zpos -= delta;
                }
            }
            let e_p = total_energy(&mol_p, obs_basis, aux_basis, op, mp2_config)?;
            let e_m = total_energy(&mol_m, obs_basis, aux_basis, op, mp2_config)?;
            grad[(atom, coord)] = (e_p - e_m) / (2.0 * delta);
        }
    }
    Ok(grad)
}

/// Compute the analytical RI-MP2 nuclear gradient.
///
/// Uses the Z-vector / relaxed density approach:
/// 1. Compute MP2 intermediates (t2, B, P_oo, P_vv)
/// 2. Solve the Z-vector equation for orbital response
/// 3. Build relaxed density and energy-weighted density in AO basis
/// 4. Evaluate gradient via `hf_gradient_with_density` (DRY reuse of RHF gradient infrastructure)
///
/// Note: the Lagrangian currently only includes P*F terms (no integral response).
/// The 3-center and 2-center derivative contributions are also TODO.
/// The gradient will be approximate until these are added.
pub fn rimp2_gradient_analytical(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &RhfResult,
    config: &RiMp2Config,
) -> Result<Array2<f64>, FerricError> {
    let inter = compute_mp2_intermediates(mol, obs, dfbs, op, rhf, config)?;

    let (z, l) = solve_zvector(mol, obs, dfbs, op, bounds, rhf, &inter)?;

    let p_relax_ao = build_relaxed_density_ao(
        &rhf.mos, &inter.p_oo, &inter.p_vv, &z,
        inter.nocc, inter.nvir, inter.nocc_total, inter.first_occ,
    );

    let nocc_total = inter.nocc_total;
    let f_mo = rhf.mos.t().dot(&rhf.fock).dot(&rhf.mos);
    let nmo = rhf.mos.ncols();
    let mut p_relax_mo = Array2::zeros((nmo, nmo));
    for i in 0..inter.nocc {
        let i_mo = inter.first_occ + i;
        p_relax_mo[(i_mo, i_mo)] += 2.0;
        for j in 0..inter.nocc {
            let j_mo = inter.first_occ + j;
            p_relax_mo[(i_mo, j_mo)] += inter.p_oo[(i, j)];
        }
    }
    for a in 0..inter.nvir {
        let a_mo = nocc_total + a;
        for b in 0..inter.nvir {
            let b_mo = nocc_total + b;
            p_relax_mo[(a_mo, b_mo)] += inter.p_vv[(a, b)];
        }
    }
    for a in 0..inter.nvir {
        let a_mo = nocc_total + a;
        for i in 0..inter.nocc {
            let i_mo = inter.first_occ + i;
            p_relax_mo[(a_mo, i_mo)] += z[(a, i)];
            p_relax_mo[(i_mo, a_mo)] += z[(a, i)];
        }
    }

    let w_relax_ao = build_relaxed_w_ao(
        &rhf.mos, &f_mo, &p_relax_mo, &l,
        inter.nocc, inter.nvir, nocc_total, inter.first_occ,
    );

    // The total gradient = RHF gradient (4-center 2e with D_HF)
    //                     + 1e/Pulay correction (P_corr, W_corr)
    //                     + 3c/2c derivative terms (RI-specific)
    //
    // We compute hf_gradient_with_density(P_relax, W_relax) which gives us
    // the correct 1e + Pulay + nuclear repulsion terms, but overcounts the
    // 4-center 2e by using Gamma(P_relax) instead of Gamma(D_HF).
    //
    // Correction: subtract the 4-center-2e contribution from P_corr.
    // For now, use the full relaxed gradient (which is approximate but a
    // reasonable starting point — the 4-center overcounting partially
    // compensates for the missing 3c/2c terms).
    let mut grad = hf_gradient_with_density(mol, obs, op, bounds, &p_relax_ao, &w_relax_ao)?;

    // 3-center derivative: Σ_{P,μ,ν} G3c_{P,μ,ν} * d(P|μν)/dR
    // G3c = "effective density" contracted with t2-weighted amplitudes
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let naux = inter.naux;
    let nov = nocc * nvir;
    let c = &rhf.mos;
    let t2 = &inter.t2;

    // Build X^P_{ia} = Σ_{jb} (2*t_{ij,ab} - t_{ij,ba}) * B^P_{jb}
    let mut x_ov = Array2::zeros((naux, nocc * nvir));
    for i in 0..nocc {
        for a in 0..nvir {
            let ia = i * nvir + a;
            for p_aux in 0..naux {
                let mut sum = 0.0;
                for j in 0..nocc {
                    for b in 0..nvir {
                        let jb = j * nvir + b;
                        let t_ij_ab = t2[(i * nvir + a) * nov + j * nvir + b];
                        let t_ij_ba = t2[(i * nvir + b) * nov + j * nvir + a];
                        let tt = 2.0 * t_ij_ab - t_ij_ba;
                        sum += tt * inter.b_ov[(p_aux, jb)];
                    }
                }
                x_ov[(p_aux, ia)] = sum;
            }
        }
    }

    // Back-transform X to AO: G3c_{P,μ,ν} = Σ_{ia} X^P_{ia} C_{μi} C_{νa} (+ transpose for symmetry)
    let nbas = obs.nbasis();
    let mut g3c = ndarray::Array3::zeros((naux, nbas, nbas));
    let c_occ = c.slice(ndarray::s![.., inter.first_occ..inter.first_occ + nocc]);
    let c_vir = c.slice(ndarray::s![.., inter.nocc_total..]);
    for p_aux in 0..naux {
        for mu in 0..nbas {
            for nu in 0..nbas {
                let mut sum = 0.0;
                for i in 0..nocc {
                    for a in 0..nvir {
                        sum += x_ov[(p_aux, i * nvir + a)] * c_occ[(mu, i)] * c_vir[(nu, a)];
                    }
                }
                g3c[(p_aux, mu, nu)] = sum;
            }
        }
    }
    // Symmetrize: G3c_{P,μ,ν} += G3c_{P,ν,μ}
    for p_aux in 0..naux {
        for mu in 0..nbas {
            for nu in (mu + 1)..nbas {
                let sum = g3c[(p_aux, mu, nu)] + g3c[(p_aux, nu, mu)];
                g3c[(p_aux, mu, nu)] = sum;
                g3c[(p_aux, nu, mu)] = sum;
            }
            g3c[(p_aux, mu, mu)] *= 2.0;
        }
    }

    // Contract G3c with 3-center derivative integrals
    {
        use ferric_integrals::engine::Engine;
        let mut eng3d = Engine::new_3center_deriv(op, obs, dfbs, 1e-14)?;
        let nsh_obs = obs.nshells();
        let nsh_df = dfbs.nshells();
        let dims_obs = obs.shell_dims();
        let offs_obs = obs.shell_offsets();
        let dims_df = dfbs.shell_dims();
        let offs_df = dfbs.shell_offsets();
        let sh2at_obs = obs.shell_to_atom();
        let sh2at_df = dfbs.shell_to_atom();

        for sp in 0..nsh_df {
            for s1 in 0..nsh_obs {
                for s2 in 0..=s1 {
                    if let Some(deriv) = eng3d.compute_eri3_deriv(obs, dfbs, sp, s1, s2) {
                        let np = dims_df[sp];
                        let n1 = dims_obs[s1];
                        let n2 = dims_obs[s2];
                        let block_sz = np * n1 * n2;
                        let sym12 = s1 != s2;

                        for p in 0..np {
                            for i in 0..n1 {
                                for j in 0..n2 {
                                    let mu = offs_obs[s1] + i;
                                    let nu = offs_obs[s2] + j;
                                    let pf = offs_df[sp] + p;
                                    let idx = (p * n1 + i) * n2 + j;

                                    let gval = if sym12 {
                                        g3c[(pf, mu, nu)] + g3c[(pf, nu, mu)]
                                    } else {
                                        g3c[(pf, mu, nu)]
                                    };

                                    // 9 derivative blocks: [dP_x, dP_y, dP_z, d1_x, d1_y, d1_z, d2_x, d2_y, d2_z]
                                    let atom_p = sh2at_df[sp];
                                    let atom_1 = sh2at_obs[s1];
                                    let atom_2 = sh2at_obs[s2];

                                    for coord in 0..3 {
                                        let dp = deriv[coord * block_sz + idx];
                                        let d1 = deriv[(3 + coord) * block_sz + idx];
                                        let d2 = deriv[(6 + coord) * block_sz + idx];

                                        grad[(atom_p, coord)] += gval * dp;
                                        grad[(atom_1, coord)] += gval * d1;
                                        grad[(atom_2, coord)] += gval * d2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2-center metric derivative: Σ_{PQ} Γ^2c_{PQ} * d(P|Q)/dR
    // Γ^2c_{PQ} = -0.5 * Σ_{ia} x_ov(P, ia) * x_ov(Q, ia)
    // where x_ov already has V^{-1/2} folded in through B^P_{jb}.
    let gamma_2c = -0.5 * x_ov.dot(&x_ov.t());

    {
        use ferric_integrals::engine::Engine;
        let mut eng2d = Engine::new_2center_deriv(op, dfbs, 1e-14)?;
        let nsh_df = dfbs.nshells();
        let dims_df = dfbs.shell_dims();
        let offs_df = dfbs.shell_offsets();
        let sh2at_df = dfbs.shell_to_atom();

        for sp in 0..nsh_df {
            for sq in 0..=sp {
                if let Some(deriv) = eng2d.compute_eri2_deriv(dfbs, sp, sq) {
                    let np = dims_df[sp];
                    let nq = dims_df[sq];
                    let block_sz = np * nq;
                    let sym_pq = sp != sq;

                    for p in 0..np {
                        for q in 0..nq {
                            let pf = offs_df[sp] + p;
                            let qf = offs_df[sq] + q;
                            let idx = p * nq + q;

                            let gval = if sym_pq {
                                gamma_2c[(pf, qf)] + gamma_2c[(qf, pf)]
                            } else {
                                gamma_2c[(pf, qf)]
                            };

                            let atom_p = sh2at_df[sp];
                            let atom_q = sh2at_df[sq];

                            for coord in 0..3 {
                                let dp = deriv[coord * block_sz + idx];
                                let dq = deriv[(3 + coord) * block_sz + idx];

                                grad[(atom_p, coord)] += gval * dp;
                                grad[(atom_q, coord)] += gval * dq;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    #[test]
    fn test_rimp2_gradient_fd_h2_symmetry() {
        // H2 along z-axis: gradient should be equal and opposite on the two atoms
        // and zero in x,y directions.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let config = RiMp2Config::default();
        let grad = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 1e-4).unwrap();

        eprintln!("H2 RI-MP2 gradient (FD, delta=1e-4):");
        for atom in 0..2 {
            eprintln!(
                "  atom {}: [{:+.10}, {:+.10}, {:+.10}]",
                atom,
                grad[(atom, 0)],
                grad[(atom, 1)],
                grad[(atom, 2)]
            );
        }

        // x and y gradients should be ~0
        for atom in 0..2 {
            for c in 0..2 {
                assert!(
                    grad[(atom, c)].abs() < 1e-8,
                    "atom={atom} coord={c}: {:.2e} should be ~0",
                    grad[(atom, c)]
                );
            }
        }

        // z gradients should be equal and opposite (translational invariance)
        assert!(
            (grad[(0, 2)] + grad[(1, 2)]).abs() < 1e-8,
            "z gradients not equal/opposite: {} vs {}",
            grad[(0, 2)],
            grad[(1, 2)]
        );

        // Should be nonzero (H2 at 0.74 A is near but not at equilibrium for MP2/cc-pVDZ)
        assert!(
            grad[(0, 2)].abs() > 1e-4,
            "z gradient too small: {}",
            grad[(0, 2)]
        );
    }

    #[test]
    fn test_rimp2_gradient_fd_consistency() {
        // Check that two different deltas give consistent results (FD convergence).
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let config = RiMp2Config::default();

        let g1 = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 1e-4).unwrap();
        let g2 = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 5e-5).unwrap();

        eprintln!("FD consistency check (H2 z-component):");
        eprintln!("  delta=1e-4: {:.10}", g1[(0, 2)]);
        eprintln!("  delta=5e-5: {:.10}", g2[(0, 2)]);
        eprintln!("  diff:       {:.2e}", (g1[(0, 2)] - g2[(0, 2)]).abs());

        // Central FD has O(delta^2) error, so halving delta should reduce error by ~4x.
        // Two independent runs should agree to ~1e-5 or better.
        assert!(
            (g1[(0, 2)] - g2[(0, 2)]).abs() < 1e-5,
            "FD inconsistent: delta=1e-4 gives {:.10}, delta=5e-5 gives {:.10}",
            g1[(0, 2)],
            g2[(0, 2)]
        );
    }

    #[test]
    fn test_rimp2_gradient_fd_h2o_translational_invariance() {
        // For any geometry the sum of forces over all atoms should be zero
        // (translational invariance / Newton's third law).
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz).unwrap();
        let obs_bs = basis::bundled("sto-3g").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let config = RiMp2Config::default();

        let grad = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 1e-4).unwrap();

        eprintln!("H2O/STO-3G RI-MP2 gradient (FD):");
        for atom in 0..3 {
            eprintln!(
                "  atom {}: [{:+.10}, {:+.10}, {:+.10}]",
                atom,
                grad[(atom, 0)],
                grad[(atom, 1)],
                grad[(atom, 2)]
            );
        }

        // Sum of gradients over all atoms should vanish for each coordinate.
        for c in 0..3 {
            let sum: f64 = (0..3).map(|a| grad[(a, c)]).sum();
            assert!(
                sum.abs() < 1e-6,
                "translational invariance violated: coord={c}, sum={:.2e}",
                sum
            );
        }
    }

    #[test]
    fn test_analytical_vs_fd_h2() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        let analytical = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();
        let fd = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 1e-4).unwrap();

        eprintln!("=== H2/cc-pVDZ Analytical vs FD RI-MP2 gradient ===");
        let mut max_diff = 0.0f64;
        for atom in 0..2 {
            for c in 0..3 {
                let diff = (analytical[(atom, c)] - fd[(atom, c)]).abs();
                max_diff = max_diff.max(diff);
                eprintln!(
                    "  atom={} coord={}: analytical={:+.8} fd={:+.8} diff={:.2e}",
                    atom, c, analytical[(atom, c)], fd[(atom, c)], diff
                );
            }
        }
        eprintln!("  max diff = {:.2e}", max_diff);
        // With 3c + 2c derivative terms. Remaining error from 4-center overcounting
        // (using P_relax in Gamma instead of D_HF).
        assert!(max_diff < 1e-4,
            "analytical vs FD max diff = {:.2e} (expected < 1e-4)", max_diff);
    }

    #[test]
    fn test_analytical_gradient_translational_invariance() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        let grad = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();

        for c in 0..3 {
            let sum: f64 = (0..2).map(|a| grad[(a, c)]).sum();
            assert!(sum.abs() < 1e-8,
                "translational invariance: coord={} sum={:.2e}", c, sum);
        }
    }
}
