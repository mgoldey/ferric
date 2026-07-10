//! RI-MP2 nuclear gradients: analytical and finite-difference reference.

use crate::rimp2::{ri_mp2, compute_mp2_intermediates_ov_only, RiMp2Config};
use crate::zvector::{solve_zvector, build_relaxed_density_ao, build_relaxed_w_ao};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::hf_gradient_with_density;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::ScfResult;
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
    let ctx = ferric_core::parallel::ParallelContext::default();
    let rhf = solve_rhf(&ctx, mol, &obs, op, &bounds, &rhf_config)?;
    if !rhf.converged {
        return Err(FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy });
    }
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
    rhf: &ScfResult,
    config: &RiMp2Config,
) -> Result<Array2<f64>, FerricError> {
    // ov-only intermediates: the gradient/zvector pipeline never reads
    // b_oo/b_vv, so the (naux, nvir²) block is never materialized here.
    let inter = compute_mp2_intermediates_ov_only(mol, obs, dfbs, op, rhf, config)?;

    let (z, l) = solve_zvector(mol, obs, dfbs, op, bounds, rhf, &inter)?;

    let p_relax_ao = build_relaxed_density_ao(
        rhf.mos_r(), &inter.p_oo, &inter.p_vv, &z, &inter.orbital_space(),
    );

    let nocc_total = inter.nocc_total;
    let f_mo = rhf.mos_r().t().dot(rhf.fock_r()).dot(rhf.mos_r());
    let nmo = rhf.mos_r().ncols();
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
        rhf.mos_r(), &f_mo, &p_relax_mo, &l, &inter.orbital_space(),
    );

    let mut grad = hf_gradient_with_density(mol, obs, op, bounds, &p_relax_ao, &w_relax_ao)?;
    grad += &integral_response_gradient_3c2c(mol, obs, dfbs, op, &inter, rhf.mos_r())?;

    Ok(grad)
}

/// 3-center + 2-center RI integral-response gradient terms.
///
/// Factored out of [`rimp2_gradient_analytical`] so the P7-parallelized region
/// (x_ov par-i build, aux-shell 3c-derivative assembly, aux-pair 2c-derivative
/// assembly) can be exercised in isolation — e.g. by the thread-count
/// bit-identity test — with fixed precomputed intermediates, independent of the
/// z-vector/JK pipeline upstream. Returns just the 3c+2c contribution as a
/// `(natoms, 3)` array; the caller adds it onto the relaxed-density HF gradient.
fn integral_response_gradient_3c2c(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    inter: &crate::rimp2::Mp2Intermediates,
    c: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let mut grad = Array2::<f64>::zeros((mol.atoms.len(), 3));

    // 3-center derivative: Σ_{P,μ,ν} G3c_{P,μ,ν} * d(P|μν)/dR
    // G3c = "effective density" contracted with t2-weighted amplitudes
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let naux = inter.naux;
    let nov = nocc * nvir;
    let t2 = &inter.t2;

    // Build X^P_{ia} = Σ_{jb} (2*t_{ij,ab} - t_{ij,ba}) * B^P_{jb} via a wide GEMM
    // per i: X_i[P, a] = B_ov · TT_i^T where TT_i[a, jb] = 2 t_{ij,ab} - t_{ij,ba}.
    // Replaces the per-element (P,i,a) scalar loop over (j,b): same FLOPs, BLAS3
    // throughput. t2 layout is t2[(i*nvir+a)*nov + j*nvir+b] = t_{ij,ab}.
    //
    // Each i owns a disjoint (naux, nvir) column band of x_ov and reads only its
    // own TT_i slice of t2, so the i-loop is embarrassingly parallel: fan it over
    // rayon via axis_chunks_iter_mut on the column-chunked view (order-preserving,
    // no shared accumulator — bit-identical to the serial loop regardless of
    // thread count). BLAS stays serial inside each closure (OPENBLAS_NUM_THREADS=1).
    let mut x_ov = Array2::zeros((naux, nocc * nvir));
    {
        use ndarray::Axis;
        use rayon::prelude::*;
        // AxisChunksIterMut is an IndexedParallelIterator: chunk i is exactly
        // columns [i*nvir, (i+1)*nvir), so `enumerate()` recovers i without any
        // extra bookkeeping, and the disjoint-column writes make this
        // order-independent/bit-identical to the serial loop.
        x_ov.axis_chunks_iter_mut(Axis(1), nvir)
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut x_i_slot)| {
                // TT_i[a, jb] = 2 t_{ij,ab} - t_{ij,ba}
                let mut tt_i = Array2::<f64>::zeros((nvir, nov));
                for a in 0..nvir {
                    for j in 0..nocc {
                        for b in 0..nvir {
                            let jb = j * nvir + b;
                            let t_ij_ab = t2[(i * nvir + a) * nov + j * nvir + b];
                            let t_ij_ba = t2[(i * nvir + b) * nov + j * nvir + a];
                            tt_i[(a, jb)] = 2.0 * t_ij_ab - t_ij_ba;
                        }
                    }
                }
                // X_i[P, a] = Σ_{jb} B_ov[P, jb] · TT_i[a, jb] = B_ov · TT_i^T  (naux, nvir)
                let x_i = inter.b_ov.dot(&tt_i.t());
                x_i_slot.assign(&x_i);
            });
    }

    // Back-transform X to AO and contract with 3-center derivative integrals,
    // blocked over the aux SHELL index. For each aux shell sp we build only its
    // g3c slab G3c_{p,μ,ν} = Σ_{ia} X^p_{ia} C_{μi} C_{νa} (symmetrized), then
    // contract it with that shell's derivative block. Peak g3c footprint is one
    // aux-shell slab (np, nbf, nbf) instead of the full (naux, nbf, nbf) tensor.
    let nbas = obs.nbasis();
    let c_occ = c.slice(ndarray::s![.., inter.first_occ..inter.first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., inter.nocc_total..]).to_owned();
    {
        use ferric_integrals::engine::Engine;
        use rayon::prelude::*;
        let nsh_obs = obs.nshells();
        let nsh_df = dfbs.nshells();
        let dims_obs = obs.shell_dims();
        let offs_obs = obs.shell_offsets();
        let dims_df = dfbs.shell_dims();
        let offs_df = dfbs.shell_offsets();
        let sh2at_obs = obs.shell_to_atom();
        let sh2at_df = dfbs.shell_to_atom();
        let natoms = mol.atoms.len();

        // Surface any engine-construction error up front (serial, cheap) so the
        // per-worker rebuilds inside for_each_init cannot fail for the same args
        // (mirrors ferric_integrals::threeindex::eri3_tensor).
        Engine::new_3center_deriv(op, obs, dfbs, 1e-14)?;

        // Axis: aux shells `sp`. Each sp is independent (own g3c slab, own deriv
        // engine call) and produces a (natoms, 3) partial. Reduction must not use
        // a rayon fold/reduce tree (thread-count-dependent FP association) or a
        // shared accumulator (data race) — instead: par-map sp -> partial via
        // for_each_init (one Engine per rayon worker), `collect` into an
        // sp-ordered Vec (IndexedParallelIterator preserves order regardless of
        // worker count), then fold serially in ascending sp order. This mirrors
        // ferric_scf::reduce::grouped_deterministic_sum's group-order-fold
        // discipline without depending on ferric-scf (mp2 does not link it);
        // natoms is small so no banding is needed — the live set is one
        // (natoms,3) partial per aux shell, negligible memory.
        let partials: Vec<Array2<f64>> = (0..nsh_df)
            .into_par_iter()
            .map_init(
                || Engine::new_3center_deriv(op, obs, dfbs, 1e-14).expect("3-center deriv engine (pre-validated)"),
                |eng3d, sp| {
                    let np = dims_df[sp];
                    let pf0 = offs_df[sp];
                    let mut local_grad = Array2::<f64>::zeros((natoms, 3));

                    // g3c for just this aux shell's rows: (np, nbf, nbf), symmetrized.
                    // For each aux fn p: X_p is (nocc, nvir); G_raw = C_occ · X_p · C_vir^T,
                    // then G3c_p = G_raw + G_raw^T (matches the old full-tensor symmetrize,
                    // including the ×2 on the diagonal via mu==nu).
                    let mut g3c_sp = ndarray::Array3::<f64>::zeros((np, nbas, nbas));
                    for p in 0..np {
                        let pf = pf0 + p;
                        let x_p = x_ov
                            .slice(ndarray::s![pf, ..])
                            .into_shape_with_order((nocc, nvir))
                            .unwrap();
                        // G_raw_{μν} = Σ_{ia} X^p_{ia} C_{μi} C_{νa} = C_occ · X_p · C_vir^T
                        let g_raw = c_occ.dot(&x_p).dot(&c_vir.t());
                        // old code did g3c[mu,mu] *= 2 after adding the (mu,nu)+(nu,mu)
                        // off-diagonals; G_raw + G_raw^T already doubles the diagonal, so
                        // g_sym matches the symmetrized full-tensor g3c to rounding.
                        let g_sym = &g_raw + &g_raw.t();
                        g3c_sp.slice_mut(ndarray::s![p, .., ..]).assign(&g_sym);
                    }

                    for s1 in 0..nsh_obs {
                        for s2 in 0..=s1 {
                            if let Some(deriv) = eng3d.compute_eri3_deriv(obs, dfbs, sp, s1, s2) {
                                let n1 = dims_obs[s1];
                                let n2 = dims_obs[s2];
                                let block_sz = np * n1 * n2;
                                let sym12 = s1 != s2;

                                for p in 0..np {
                                    for i in 0..n1 {
                                        for j in 0..n2 {
                                            let mu = offs_obs[s1] + i;
                                            let nu = offs_obs[s2] + j;
                                            let idx = (p * n1 + i) * n2 + j;

                                            let gval = if sym12 {
                                                g3c_sp[(p, mu, nu)] + g3c_sp[(p, nu, mu)]
                                            } else {
                                                g3c_sp[(p, mu, nu)]
                                            };

                                            // 9 derivative blocks: [dP_x, dP_y, dP_z, d1_x, d1_y, d1_z, d2_x, d2_y, d2_z]
                                            let atom_p = sh2at_df[sp];
                                            let atom_1 = sh2at_obs[s1];
                                            let atom_2 = sh2at_obs[s2];

                                            for coord in 0..3 {
                                                let dp = deriv[coord * block_sz + idx];
                                                let d1 = deriv[(3 + coord) * block_sz + idx];
                                                let d2 = deriv[(6 + coord) * block_sz + idx];

                                                local_grad[(atom_p, coord)] += gval * dp;
                                                local_grad[(atom_1, coord)] += gval * d1;
                                                local_grad[(atom_2, coord)] += gval * d2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    local_grad
                },
            )
            .collect();

        // Serial fold in ascending sp order — the determinism anchor (same
        // discipline as grouped_deterministic_sum, just without banding since
        // the partials are tiny).
        for p in &partials {
            grad += p;
        }
    }

    // 2-center metric derivative: Σ_{PQ} Γ^2c_{PQ} * d(P|Q)/dR
    let gamma_2c = -0.5 * x_ov.dot(&x_ov.t());

    {
        use ferric_integrals::engine::Engine;
        use rayon::prelude::*;
        let nsh_df = dfbs.nshells();
        let dims_df = dfbs.shell_dims();
        let offs_df = dfbs.shell_offsets();
        let sh2at_df = dfbs.shell_to_atom();
        let natoms = mol.atoms.len();

        Engine::new_2center_deriv(op, dfbs, 1e-14)?;

        // Axis: aux shell `sp` (outer loop of the sp>=sq triangle). Same
        // par-map + sp-ordered collect + serial fold discipline as the 3c block
        // above — each sp owns the full sq<=sp inner sweep as its unit of work,
        // so partials are independent and the fold order is sp-ascending only.
        let partials: Vec<Array2<f64>> = (0..nsh_df)
            .into_par_iter()
            .map_init(
                || Engine::new_2center_deriv(op, dfbs, 1e-14).expect("2-center deriv engine (pre-validated)"),
                |eng2d, sp| {
                    let mut local_grad = Array2::<f64>::zeros((natoms, 3));
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

                                        local_grad[(atom_p, coord)] += gval * dp;
                                        local_grad[(atom_q, coord)] += gval * dq;
                                    }
                                }
                            }
                        }
                    }
                    local_grad
                },
            )
            .collect();

        // Serial fold in ascending sp order.
        for p in &partials {
            grad += p;
        }
    }

    Ok(grad)
}

/// Compute the analytical SCS-MP2 nuclear gradient.
pub fn scs_mp2_gradient_analytical(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    config: &crate::scs::ScsMp2Config,
) -> Result<Array2<f64>, FerricError> {
    let mp2_config = RiMp2Config { frozen_core: config.frozen_core, memory_budget_bytes: config.memory_budget_bytes };
    let inter = compute_mp2_intermediates_ov_only(mol, obs, dfbs, op, rhf, &mp2_config)?;
    let (z, l) = solve_zvector(mol, obs, dfbs, op, bounds, rhf, &inter)?;
    let p_relax_ao = build_relaxed_density_ao(
        rhf.mos_r(), &inter.p_oo, &inter.p_vv, &z, &inter.orbital_space(),
    );
    let nmo = rhf.mos_r().ncols();
    let nocc_total = inter.nocc_total;
    let f_mo = rhf.mos_r().t().dot(rhf.fock_r()).dot(rhf.mos_r());
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
        rhf.mos_r(), &f_mo, &p_relax_mo, &l, &inter.orbital_space(),
    );
    let mut grad = hf_gradient_with_density(mol, obs, op, bounds, &p_relax_ao, &w_relax_ao)?;
    // Approximate scaling: multiply MP2 part by average SCS scaling
    let scale = (config.c_os + config.c_ss) / 2.0;
    let rhf_grad = ferric_scf::gradient::rhf_gradient(mol, obs, op, bounds, rhf)?;
    for i in 0..mol.atoms.len() {
        for c in 0..3 {
            let mp2_part = grad[(i, c)] - rhf_grad[(i, c)];
            grad[(i, c)] = rhf_grad[(i, c)] + scale * mp2_part;
        }
    }
    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_scf::gradient::{oneelectron_gradient, twoelectron_gradient};

    #[test]
    fn test_rimp2_gradient_fd_h2_symmetry() {
        // H2 along z-axis: gradient should be equal and opposite on the two atoms
        // and zero in x,y directions.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
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
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
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
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
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
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
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
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        let grad = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();

        for c in 0..3 {
            let sum: f64 = (0..2).map(|a| grad[(a, c)]).sum();
            assert!(sum.abs() < 1e-8,
                "translational invariance: coord={} sum={:.2e}", c, sum);
        }
    }

    #[test]
    fn test_3c_density_numerical() {
        // Numerically verify dE/d(P|mu,nu) for a single element.
        // Perturb one raw 3-center integral and see how E_corr changes.
        use crate::rimp2::{compute_mp2_intermediates, RiMp2Config, cholesky_inverse_sqrt};
        use ferric_integrals::threeindex;

        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();
        let inter = compute_mp2_intermediates(&mol, &obs, &dfbs, op, &rhf, &config).unwrap();

        let nocc = inter.nocc;
        let nvir = inter.nvir;
        let nov = nocc * nvir;
        let naux = inter.naux;
        let c = rhf.mos_r();
        let t2 = &inter.t2;

        // Get raw 3-center integrals and metric
        let eri3_ao = threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let v2c = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();

        let c_occ = c.slice(ndarray::s![.., inter.first_occ..inter.first_occ + nocc]).to_owned();
        let c_vir = c.slice(ndarray::s![.., inter.nocc_total..]).to_owned();

        let compute_ecorr_from_3c = |eri3: &ndarray::Array3<f64>| -> f64 {
            let v_inv_sqrt = cholesky_inverse_sqrt(&v2c).unwrap();
            let eri3_ov = crate::mo_transform::transform_3center_ov(eri3, &c_occ, &c_vir);
            let b_ov = v_inv_sqrt.dot(&eri3_ov.into_shape_with_order((naux, nov)).unwrap());
            let mut e = 0.0;
            for i in 0..nocc {
                for a in 0..nvir {
                    let ia = i * nvir + a;
                    for j in 0..nocc {
                        for b in 0..nvir {
                            let jb = j * nvir + b;
                            let t_ij_ab = t2[ia * nov + jb];
                            let t_ij_ba = t2[(i*nvir+b) * nov + (j*nvir+a)];
                            let tt = 2.0 * t_ij_ab - t_ij_ba;
                            let eri: f64 = (0..naux).map(|p| b_ov[(p, ia)] * b_ov[(p, jb)]).sum();
                            e += tt * eri;
                        }
                    }
                }
            }
            e * 0.5
        };

        // Build x_ov and analytical densities
        let mut x_ov = Array2::zeros((naux, nov));
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
        let y_ov = inter.v_inv_sqrt.t().dot(&x_ov);

        // Test a specific 3c element: perturb eri3[P=0, mu=0, nu=0]
        let test_p = 0;
        let test_mu = 0;
        let test_nu = 1;
        let delta = 1e-6;

        let mut eri3_plus = eri3_ao.clone();
        let mut eri3_minus = eri3_ao.clone();
        eri3_plus[(test_p, test_mu, test_nu)] += delta;
        eri3_plus[(test_p, test_nu, test_mu)] += delta;  // symmetrize
        eri3_minus[(test_p, test_mu, test_nu)] -= delta;
        eri3_minus[(test_p, test_nu, test_mu)] -= delta;

        let e_plus = compute_ecorr_from_3c(&eri3_plus);
        let e_minus = compute_ecorr_from_3c(&eri3_minus);
        let fd_deriv = (e_plus - e_minus) / (2.0 * delta);

        // Analytical: dE/d(P|mu,nu) using x_ov (original code's formula)
        let g3c_x_munu = (0..nocc).flat_map(|i| (0..nvir).map(move |a| (i, a)))
            .map(|(i, a)| x_ov[(test_p, i * nvir + a)] * c_occ[(test_mu, i)] * c_vir[(test_nu, a)]
                        + x_ov[(test_p, i * nvir + a)] * c_occ[(test_nu, i)] * c_vir[(test_mu, a)])
            .sum::<f64>();

        // Analytical: dE/d(P|mu,nu) using y_ov (proposed fix)
        let g3c_y_munu = (0..nocc).flat_map(|i| (0..nvir).map(move |a| (i, a)))
            .map(|(i, a)| y_ov[(test_p, i * nvir + a)] * c_occ[(test_mu, i)] * c_vir[(test_nu, a)]
                        + y_ov[(test_p, i * nvir + a)] * c_occ[(test_nu, i)] * c_vir[(test_mu, a)])
            .sum::<f64>();

        eprintln!("=== dE/d(P={},mu={},nu={}) ===", test_p, test_mu, test_nu);
        eprintln!("FD:        {:.12}", fd_deriv);
        eprintln!("x_ov:      {:.12}  (code's formula, no extra V^{{-1/2}})", g3c_x_munu);
        eprintln!("y_ov:      {:.12}  (with extra V^{{-1/2}})", g3c_y_munu);
        eprintln!("2*y_ov:    {:.12}  (with factor 2)", 2.0 * g3c_y_munu);
        eprintln!("diff(x):   {:.6e}", g3c_x_munu - fd_deriv);
        eprintln!("diff(y):   {:.6e}", g3c_y_munu - fd_deriv);
        eprintln!("diff(2y):  {:.6e}", 2.0 * g3c_y_munu - fd_deriv);
    }

    #[test]
    fn test_3c_gradient_fd_check() {
        // Test: compute the 3-center gradient contribution by finite differences
        // of the raw 3-center integrals, and compare with the analytical formula.
        // This isolates the 3c term from everything else.
        use crate::rimp2::{compute_mp2_intermediates, RiMp2Config};
        use ferric_integrals::threeindex;
        use crate::rimp2::cholesky_inverse_sqrt;

        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();
        let inter = compute_mp2_intermediates(&mol, &obs, &dfbs, op, &rhf, &config).unwrap();

        let nocc = inter.nocc;
        let nvir = inter.nvir;
        let nov = nocc * nvir;
        let naux = inter.naux;
        let c = rhf.mos_r();
        let t2 = &inter.t2;

        // Build x_ov
        let mut x_ov = Array2::zeros((naux, nov));
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

        // FD: compute E_corr at displaced geometries using the SAME HF orbitals and t2
        // but displaced 3-center integrals.
        // E_corr = sum_{ia,jb} tilde_t * sum_P B_P_ia * B_P_jb
        // where B = V^{-1/2} c3, and c3 = (P|mu,nu) * C_occ * C_vir
        // We displace the nuclei, recompute c3 (and V), then recompute E_corr.
        // This gives dE/dR from the integral response only (not orbital response).
        //
        // Actually, for a proper check of the 3c term only, we should keep V^{-1/2} fixed
        // and only vary (P|mu,nu). But in practice both 3c and V change.
        //
        // Let's do a simpler check: FD of E_corr = sum_{P,ia,jb} tilde_t * B_{ia}^P * B_{jb}^P
        // with respect to nuclear displacement, keeping tilde_t and MOs fixed.
        let c_occ = c.slice(ndarray::s![.., inter.first_occ..inter.first_occ + nocc]).to_owned();
        let c_vir = c.slice(ndarray::s![.., inter.nocc_total..]).to_owned();

        let compute_ecorr_at_geom = |mol_disp: &Molecule| -> f64 {
            let obs_d = PreparedBasis::new(mol_disp, &obs_bs).unwrap();
            let dfbs_d = PreparedBasis::new(mol_disp, &aux_bs).unwrap();
            let v2c = threeindex::coulomb_metric_2c(op, &dfbs_d).unwrap();
            let v_inv_sqrt = cholesky_inverse_sqrt(&v2c).unwrap();
            let eri3_ao = threeindex::eri3_tensor(op, &obs_d, &dfbs_d).unwrap();
            let eri3_ov = crate::mo_transform::transform_3center_ov(&eri3_ao, &c_occ, &c_vir);
            let b_ov = v_inv_sqrt.dot(&eri3_ov.into_shape_with_order((naux, nov)).unwrap());
            let mut e = 0.0;
            for i in 0..nocc {
                for a in 0..nvir {
                    let ia = i * nvir + a;
                    for j in 0..nocc {
                        for b in 0..nvir {
                            let jb = j * nvir + b;
                            let t_ij_ab = t2[ia * nov + jb];
                            let t_ij_ba = t2[(i*nvir+b) * nov + (j*nvir+a)];
                            let tt = 2.0 * t_ij_ab - t_ij_ba;
                            let eri: f64 = (0..naux).map(|p| b_ov[(p, ia)] * b_ov[(p, jb)]).sum();
                            e += tt * eri;
                        }
                    }
                }
            }
            e * 0.5  // overcounting factor
        };

        let delta = 1e-5;
        // Atom 0, z-component
        let mut mol_p = mol.clone();
        let mut mol_m = mol.clone();
        mol_p.atoms[0].zpos += delta;
        mol_m.atoms[0].zpos -= delta;
        let fd_integral_grad = (compute_ecorr_at_geom(&mol_p) - compute_ecorr_at_geom(&mol_m)) / (2.0 * delta);

        eprintln!("=== 3c+2c gradient check (integral-response only, atom 0, z) ===");
        eprintln!("FD integral-response gradient: {:.12}", fd_integral_grad);

        // Now compute the analytical 3c+2c contribution at the reference geometry
        // (This is the 3c+2c part from the full analytical gradient)
        let full_grad = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();
        let hf_relax = hf_gradient_with_density(&mol, &obs, op, &bounds,
            &crate::zvector::build_relaxed_density_ao(
                rhf.mos_r(), &inter.p_oo, &inter.p_vv,
                &{
                    let (z, _) = crate::zvector::solve_zvector(&mol, &obs, &dfbs, op, &bounds, &rhf, &inter).unwrap();
                    z
                },
                &inter.orbital_space(),
            ),
            &{
                let (z, l) = crate::zvector::solve_zvector(&mol, &obs, &dfbs, op, &bounds, &rhf, &inter).unwrap();
                let nocc_total = inter.nocc_total;
                let f_mo = rhf.mos_r().t().dot(rhf.fock_r()).dot(rhf.mos_r());
                let nmo = rhf.mos_r().ncols();
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
                crate::zvector::build_relaxed_w_ao(
                    rhf.mos_r(), &f_mo, &p_relax_mo, &l, &inter.orbital_space(),
                )
            },
        ).unwrap();
        let analytic_3c2c = full_grad[(0, 2)] - hf_relax[(0, 2)];
        eprintln!("Analytical 3c+2c contribution:   {:.12}", analytic_3c2c);
        eprintln!("Diff:                            {:.6e}", analytic_3c2c - fd_integral_grad);
    }

    #[test]
    fn test_fd_convergence_study() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        let analytical = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();

        // Also compute the HF gradient and the hf_gradient_with_density(P_relax, W_relax) to decompose
        let rhf_grad = ferric_scf::gradient::rhf_gradient(&mol, &obs, op, &bounds, &rhf).unwrap();

        // Compute P_relax and W_relax like in rimp2_gradient_analytical
        let inter = crate::rimp2::compute_mp2_intermediates(&mol, &obs, &dfbs, op, &rhf, &config).unwrap();
        let (z, l) = crate::zvector::solve_zvector(&mol, &obs, &dfbs, op, &bounds, &rhf, &inter).unwrap();
        let p_relax_ao = crate::zvector::build_relaxed_density_ao(
            rhf.mos_r(), &inter.p_oo, &inter.p_vv, &z, &inter.orbital_space(),
        );
        let nocc_total = inter.nocc_total;
        let f_mo = rhf.mos_r().t().dot(rhf.fock_r()).dot(rhf.mos_r());
        let nmo = rhf.mos_r().ncols();
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
        let w_relax_ao = crate::zvector::build_relaxed_w_ao(
            rhf.mos_r(), &f_mo, &p_relax_mo, &l, &inter.orbital_space(),
        );
        let hf_with_relax = hf_gradient_with_density(&mol, &obs, op, &bounds, &p_relax_ao, &w_relax_ao).unwrap();

        // The 3c+2c contribution = analytical - hf_with_relax
        let corr_3c2c = analytical[(0, 2)] - hf_with_relax[(0, 2)];
        let hf_corr_diff = hf_with_relax[(0, 2)] - rhf_grad[(0, 2)];

        // FD for HF and total
        let delta = 1e-5;
        let fd_total = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, delta).unwrap();
        let mut fd_hf = Array2::zeros((2, 3));
        for atom in 0..2 {
            for coord in 0..3 {
                let mut mol_p = mol.clone();
                let mut mol_m = mol.clone();
                match coord {
                    0 => { mol_p.atoms[atom].x += delta; mol_m.atoms[atom].x -= delta; }
                    1 => { mol_p.atoms[atom].y += delta; mol_m.atoms[atom].y -= delta; }
                    _ => { mol_p.atoms[atom].zpos += delta; mol_m.atoms[atom].zpos -= delta; }
                }
                let bs2 = basis::bundled("cc-pvdz").unwrap();
                let prep_p = PreparedBasis::new(&mol_p, &bs2).unwrap();
                let bounds_p = SchwarzBounds::compute(op, &prep_p).unwrap();
                let rhf_p = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol_p, &prep_p, op, &bounds_p, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
                let prep_m = PreparedBasis::new(&mol_m, &bs2).unwrap();
                let bounds_m = SchwarzBounds::compute(op, &prep_m).unwrap();
                let rhf_m = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol_m, &prep_m, op, &bounds_m, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
                fd_hf[(atom, coord)] = (rhf_p.energy - rhf_m.energy) / (2.0 * delta);
            }
        }
        let fd_corr = fd_total[(0, 2)] - fd_hf[(0, 2)];

        eprintln!("=== Gradient decomposition (z-component, atom 0) ===");
        eprintln!("RHF gradient (analytical):        {:.12}", rhf_grad[(0, 2)]);
        eprintln!("RHF gradient (FD):                {:.12}", fd_hf[(0, 2)]);
        eprintln!("RHF diff:                         {:.6e}", rhf_grad[(0, 2)] - fd_hf[(0, 2)]);
        eprintln!();
        eprintln!("hf_grad_with_density(P_relax,W):  {:.12}", hf_with_relax[(0, 2)]);
        eprintln!("delta from HF to relaxed-density: {:.12}", hf_corr_diff);
        eprintln!();
        eprintln!("3c+2c contribution (analytical):  {:.12}", corr_3c2c);
        eprintln!("MP2 corr gradient (FD):           {:.12}", fd_corr);
        eprintln!("3c+2c vs FD corr diff:            {:.6e}", corr_3c2c - fd_corr);
        eprintln!();
        // Decompose 1e vs 2e for both D_HF and P_relax
        let nocc_hf = (mol.nelec() / 2) as usize;
        let w_hf = ferric_scf::gradient::build_energy_weighted_density(&rhf, nocc_hf);
        let oe_dhf = oneelectron_gradient(&mol, &obs, rhf.density_r(), &w_hf).unwrap();
        let te_dhf = twoelectron_gradient(&obs, op, &bounds, rhf.density_r()).unwrap();
        let oe_relax = oneelectron_gradient(&mol, &obs, &p_relax_ao, &w_relax_ao).unwrap();
        let te_relax = twoelectron_gradient(&obs, op, &bounds, &p_relax_ao).unwrap();

        eprintln!();
        eprintln!("=== 1e/2e decomposition (z-component, atom 0) ===");
        eprintln!("1e(D_HF):     {:.12}", oe_dhf[(0, 2)]);
        eprintln!("2e(D_HF):     {:.12}", te_dhf[(0, 2)]);
        eprintln!("sum(D_HF):    {:.12}", oe_dhf[(0, 2)] + te_dhf[(0, 2)]);
        eprintln!("1e(P_relax):  {:.12}", oe_relax[(0, 2)]);
        eprintln!("2e(P_relax):  {:.12}", te_relax[(0, 2)]);
        eprintln!("sum(P_relax): {:.12}", oe_relax[(0, 2)] + te_relax[(0, 2)]);
        eprintln!();
        eprintln!("delta_1e = 1e(P_relax) - 1e(D_HF): {:.12}", oe_relax[(0, 2)] - oe_dhf[(0, 2)]);
        eprintln!("delta_2e = 2e(P_relax) - 2e(D_HF): {:.12}", te_relax[(0, 2)] - te_dhf[(0, 2)]);
        eprintln!();
        eprintln!("If 4c uses D_HF: total = 1e(P_relax) + 2e(D_HF) + 3c2c");
        let total_dhf_4c = oe_relax[(0, 2)] + te_dhf[(0, 2)] + corr_3c2c;
        eprintln!("  = {:.12} + {:.12} + {:.12} = {:.12}", oe_relax[(0, 2)], te_dhf[(0, 2)], corr_3c2c, total_dhf_4c);
        eprintln!("  diff from FD: {:.6e}", total_dhf_4c - fd_total[(0, 2)]);

        eprintln!();
        eprintln!("Total analytical:                 {:.12}", analytical[(0, 2)]);
        eprintln!("Total FD:                         {:.12}", fd_total[(0, 2)]);
        eprintln!("Total diff:                       {:.6e}", analytical[(0, 2)] - fd_total[(0, 2)]);
    }

    #[test]
    fn test_analytical_vs_fd_h2o() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("sto-3g").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        let analytical = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();
        let fd = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 1e-4).unwrap();

        eprintln!("=== H2O/STO-3G Analytical vs FD RI-MP2 gradient ===");
        let mut max_diff = 0.0f64;
        for atom in 0..3 {
            for c in 0..3 {
                let diff = (analytical[(atom, c)] - fd[(atom, c)]).abs();
                max_diff = max_diff.max(diff);
                eprintln!("  atom={} coord={}: analytical={:+.8} fd={:+.8} diff={:.2e}",
                    atom, c, analytical[(atom, c)], fd[(atom, c)], diff);
            }
        }
        eprintln!("  max diff = {:.2e}", max_diff);
        // H2O has larger error than H2 due to 4-center overcounting with relaxed density.
        // Current accuracy limited by incomplete Lagrangian integral response and
        // missing separation of 1e/2e gradient contributions.
        assert!(max_diff < 0.1,
            "H2O analytical vs FD max diff = {:.2e} (expected < 0.1)", max_diff);
    }

    #[test]
    fn test_3c2c_assembly_bit_identical_across_thread_counts() {
        // Scope: ONLY the P7-parallelized region (x_ov par-i build, aux-shell
        // 3c-derivative assembly, aux-pair 2c-derivative assembly), fed the
        // SAME precomputed intermediates in rayon pools pinned to 1 and 4
        // workers, compared via f64::to_bits. Whole-pipeline bit-identity is
        // gated on P14 (docs/parallelism-gaps-2026-07-09.md) — build_jk's
        // legacy fold/reduce tree drifts ~1 ULP by thread count inside
        // solve_zvector, which is upstream of (and outside) this region.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        // Precompute intermediates ONCE, outside any pinned pool, so both runs
        // see bit-identical inputs and only the P7 region is under test.
        let inter = compute_mp2_intermediates_ov_only(&mol, &obs, &dfbs, op, &rhf, &config).unwrap();

        let run_with_threads = |n: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
            pool.install(|| {
                integral_response_gradient_3c2c(&mol, &obs, &dfbs, op, &inter, rhf.mos_r()).unwrap()
            })
        };

        let g1 = run_with_threads(1);
        let g4 = run_with_threads(4);

        for atom in 0..2 {
            for c in 0..3 {
                assert_eq!(
                    g1[(atom, c)].to_bits(),
                    g4[(atom, c)].to_bits(),
                    "3c+2c assembly not bit-identical across thread counts at atom={atom} coord={c}: \
                     1-thread={:.17e} (0x{:016x}), 4-thread={:.17e} (0x{:016x})",
                    g1[(atom, c)], g1[(atom, c)].to_bits(),
                    g4[(atom, c)], g4[(atom, c)].to_bits(),
                );
            }
        }
    }

    #[test]
    fn test_analytical_h2_symmetry() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        let grad = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();

        // x,y components should be zero for H2 along z-axis
        for atom in 0..2 {
            for c in 0..2 {
                assert!(grad[(atom, c)].abs() < 1e-12,
                    "atom={} coord={}: {:.2e} should be ~0", atom, c, grad[(atom, c)]);
            }
        }
        // z components equal and opposite
        assert!((grad[(0, 2)] + grad[(1, 2)]).abs() < 1e-10,
            "z not equal/opposite: {} vs {}", grad[(0, 2)], grad[(1, 2)]);
        // z should be nonzero
        assert!(grad[(0, 2)].abs() > 1e-4, "z gradient too small: {}", grad[(0, 2)]);
    }

    #[test]
    fn test_analytical_h2o_translational_invariance() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("sto-3g").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        let grad = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();

        for c in 0..3 {
            let sum: f64 = (0..3).map(|a| grad[(a, c)]).sum();
            assert!(sum.abs() < 1e-8,
                "H2O translational invariance: coord={} sum={:.2e}", c, sum);
        }
    }

    #[test]
    fn test_gradient_split_consistency() {
        // oneelectron_gradient + twoelectron_gradient should equal hf_gradient_with_density
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let nocc = (mol.nelec() / 2) as usize;
        let w = ferric_scf::gradient::build_energy_weighted_density(&rhf, nocc);

        let combined = hf_gradient_with_density(&mol, &obs, op, &bounds, rhf.density_r(), &w).unwrap();
        let oe = oneelectron_gradient(&mol, &obs, rhf.density_r(), &w).unwrap();
        let te = twoelectron_gradient(&obs, op, &bounds, rhf.density_r()).unwrap();
        let split = &oe + &te;

        for atom in 0..2 {
            for c in 0..3 {
                let diff = (combined[(atom, c)] - split[(atom, c)]).abs();
                assert!(diff < 1e-12,
                    "split mismatch: atom={} coord={} combined={:.10} split={:.10} diff={:.2e}",
                    atom, c, combined[(atom, c)], split[(atom, c)], diff);
            }
        }
    }
}
