//! Analytical nuclear gradients.
//!
//! Provides both the RHF gradient and a density-parameterized core
//! ([`hf_gradient_with_density`]) that correlated methods reuse with relaxed densities.

use crate::result::{ScfResult, Spin};
use crate::rhf::RhfResult;
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use ndarray::Array2;

/// Build the HF energy-weighted density: W_μν = 2 Σ_i^occ ε_i C_μi C_νi.
pub fn build_energy_weighted_density(result: &RhfResult, nocc: usize) -> Array2<f64> {
    let n = result.mos_r().nrows();
    let c = result.mos_r();
    let eps = result.eps_r();
    let mut w = Array2::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for i in 0..nocc {
                sum += eps[i] * c[(mu, i)] * c[(nu, i)];
            }
            w[(mu, nu)] = 2.0 * sum;
        }
    }
    w
}

/// Compute the RHF analytical nuclear gradient.
/// Returns a (natoms, 3) array of dE/dR_Ax, dE/dR_Ay, dE/dR_Az per atom.
pub fn rhf_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    result: &RhfResult,
) -> Result<Array2<f64>, FerricError> {
    let nocc = (mol.nelec() / 2) as usize;
    let w = build_energy_weighted_density(result, nocc);
    hf_gradient_with_density(mol, prep, op, bounds, &result.density_r(), &w)
}

/// Compute nuclear gradient using provided density and energy-weighted density.
///
/// Shared core of both the RHF gradient and correlated gradients (which pass
/// relaxed densities). Includes nuclear repulsion, 1e (dS, dT, dV), and
/// 4-center 2e contributions.
pub fn hf_gradient_with_density(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d: &Array2<f64>,
    w: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let mut grad = oneelectron_gradient(mol, prep, d, w)?;
    grad += &twoelectron_gradient(prep, op, bounds, d)?;
    Ok(grad)
}

/// Compute one-electron gradient contributions: nuclear repulsion + dS, dT, dV.
///
/// Takes the density `d` (for kinetic + nuclear attraction derivatives) and the
/// energy-weighted density `w` (for overlap / Pulay force). Returns a `(natoms, 3)`
/// gradient array.
pub fn oneelectron_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    d: &Array2<f64>,
    w: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = mol.atoms.len();
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();

    let mut grad = Array2::zeros((natoms, 3));

    // 1. Nuclear repulsion gradient
    for i in 0..natoms {
        for j in (i + 1)..natoms {
            let a = &mol.atoms[i];
            let b = &mol.atoms[j];
            let dx = a.x - b.x;
            let dy = a.y - b.y;
            let dz = a.zpos - b.zpos;
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            let za = a.z as f64;
            let zb = b.z as f64;
            let f_over_r3 = za * zb / (r * r2);
            grad[(i, 0)] -= f_over_r3 * dx;
            grad[(i, 1)] -= f_over_r3 * dy;
            grad[(i, 2)] -= f_over_r3 * dz;
            grad[(j, 0)] += f_over_r3 * dx;
            grad[(j, 1)] += f_over_r3 * dy;
            grad[(j, 2)] += f_over_r3 * dz;
        }
    }

    // 2. One-electron gradient: Σ_μν D_μν dH_μν/dR - Σ_μν W_μν dS_μν/dR
    // H = T + V, so we need dT/dR, dV/dR, dS/dR

    // 2a. Overlap derivative (Pulay force): -W_μν dS_μν/dR
    {
        let mut eng = Engine::new_1e_deriv(ffi::OP_OVERLAP, prep, 1e-14)?;
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                if let Some(deriv) = eng.compute_1e_deriv_block(prep, s1, s2) {
                    let n1 = dims[s1];
                    let n2 = dims[s2];
                    let block_sz = n1 * n2;
                    let a1 = sh2at[s1];
                    let a2 = sh2at[s2];
                    for i in 0..n1 {
                        for j in 0..n2 {
                            let mu = offs[s1] + i;
                            let nu = offs[s2] + j;
                            let idx = i * n2 + j;
                            let wval = if s1 == s2 { w[(mu, nu)] } else { 2.0 * w[(mu, nu)] };
                            for c in 0..3 {
                                // deriv layout: [dx1, dy1, dz1, dx2, dy2, dz2]
                                let d1 = deriv[c * block_sz + idx];
                                let d2 = deriv[(3 + c) * block_sz + idx];
                                grad[(a1, c)] -= wval * d1;
                                grad[(a2, c)] -= wval * d2;
                            }
                        }
                    }
                }
            }
        }
    }

    // 2b. Kinetic derivative: D_μν dT_μν/dR
    {
        let mut eng = Engine::new_1e_deriv(ffi::OP_KINETIC, prep, 1e-14)?;
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                if let Some(deriv) = eng.compute_1e_deriv_block(prep, s1, s2) {
                    let n1 = dims[s1];
                    let n2 = dims[s2];
                    let block_sz = n1 * n2;
                    let a1 = sh2at[s1];
                    let a2 = sh2at[s2];
                    for i in 0..n1 {
                        for j in 0..n2 {
                            let mu = offs[s1] + i;
                            let nu = offs[s2] + j;
                            let idx = i * n2 + j;
                            let dval = if s1 == s2 { d[(mu, nu)] } else { 2.0 * d[(mu, nu)] };
                            for c in 0..3 {
                                let d1 = deriv[c * block_sz + idx];
                                let d2 = deriv[(3 + c) * block_sz + idx];
                                grad[(a1, c)] += dval * d1;
                                grad[(a2, c)] += dval * d2;
                            }
                        }
                    }
                }
            }
        }
    }

    // 2c. Nuclear attraction derivative: D_μν dV_μν/dR
    // Nuclear has 2 shell centers + natoms nuclear centers = 2+natoms centers for derivatives
    // But libint2 only returns derivatives w.r.t. the 2 shell centers for nuclear operator.
    // The nuclear center derivatives need separate handling.
    // Actually for nuclear integrals, libint2 returns derivatives w.r.t. all centers:
    // For 2 shell centers, results has 3*(2+natoms) entries when using the nuclear operator.
    // This is complex — let me use a simpler approach: finite difference on V for now,
    // and use the Hellmann-Feynman term for the nuclear derivative.
    //
    // The proper approach for dV/dR:
    // dV/dR_A = Σ_μν D_μν [dV_μν/dR_A(shell) + dV_μν/dR_A(nuclear)]
    // The shell derivative part is the Pulay-like term from shells on atom A.
    // The nuclear (Hellmann-Feynman) part is: Σ_μν D_μν d/dR_A [Σ_C Z_C/|r-R_C|]
    // which only contributes when A is one of the nuclear centers.
    //
    // For the Hellmann-Feynman nuclear attraction gradient, we need:
    // dV/dR_A = -Z_A Σ_μν D_μν <μ| (r-R_A)/|r-R_A|^3 |ν>
    // This is equivalent to the electric field integral at nucleus A, contracted with D.
    //
    // However, libint2's nuclear operator derivative engine handles all of this in one shot:
    // for deriv_order=1 with nuclear operator, it returns 3*(2 + N_charges) derivative blocks.
    // Blocks 0-5 are d/d(shell center 1) and d/d(shell center 2) as usual.
    // Blocks 6, 7, 8 are d/d(charge center 0) (x, y, z).
    // Blocks 9, 10, 11 are d/d(charge center 1), etc.
    //
    // So we need to handle this specially.
    {
        let mut eng = Engine::new_1e_deriv(ffi::OP_NUCLEAR, prep, 1e-14)?;
        eng.set_point_charges(prep);
        // For nuclear derivative, the buffer needs to hold (6 + 3*natoms) blocks
        // Let's compute it and handle the extended result.
        let max_fn = prep.shell_dims().iter().copied().max().unwrap_or(1);
        let nderiv_nuclear = 6 + 3 * natoms;
        let max_block = max_fn * max_fn;
        let mut nbuf = vec![0.0f64; nderiv_nuclear * max_block];

        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                let n1 = dims[s1];
                let n2 = dims[s2];
                let block_sz = n1 * n2;
                let total = nderiv_nuclear * block_sz;
                if nbuf.len() < total { nbuf.resize(total, 0.0); }
                let written = unsafe {
                    ffi::goscf_compute_1e_deriv_block(
                        eng.handle_mut(), prep.handle(),
                        s1 as std::os::raw::c_int, s2 as std::os::raw::c_int,
                        nbuf.as_mut_ptr(),
                    )
                };
                if written == 0 { continue; }
                let a1 = sh2at[s1];
                let a2 = sh2at[s2];
                for i in 0..n1 {
                    for j in 0..n2 {
                        let mu = offs[s1] + i;
                        let nu = offs[s2] + j;
                        let idx = i * n2 + j;
                        let dval = if s1 == s2 { d[(mu, nu)] } else { 2.0 * d[(mu, nu)] };
                        // Shell center derivatives (first 6 blocks)
                        for c in 0..3 {
                            grad[(a1, c)] += dval * nbuf[c * block_sz + idx];
                            grad[(a2, c)] += dval * nbuf[(3 + c) * block_sz + idx];
                        }
                        // Nuclear center derivatives (blocks 6 onwards)
                        for atom_c in 0..natoms {
                            for c in 0..3 {
                                let blk = 6 + atom_c * 3 + c;
                                grad[(atom_c, c)] += dval * nbuf[blk * block_sz + idx];
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(grad)
}

/// Compute the 4-center two-electron gradient contribution: Σ Γ_μνλσ d(μν|λσ)/dR.
///
/// Uses the two-particle density Γ built from the provided one-particle density `d`:
///   Γ_μνλσ = 0.5 D_μν D_λσ - 0.25 D_μλ D_νσ
///
/// Returns a `(natoms, 3)` gradient array.
pub fn twoelectron_gradient(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = prep.shell_to_atom().iter().copied().max().unwrap_or(0) + 1;
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();

    let mut grad = Array2::zeros((natoms, 3));

    // Two-electron gradient
    // We use a canonical shell loop with explicit permutation handling.
    // For each canonical quartet (s1>=s2, s3>=s4, (s1,s2)>=(s3,s4)), we enumerate
    // all equivalent permutations and accumulate with the correct two-particle
    // density for each permutation.
    //
    // The 2e gradient contribution from each integral (μν|λσ) is:
    //   Γ_μνλσ * d(μν|λσ)/dR
    // where Γ_μνλσ = 0.5*D_μν*D_λσ - 0.25*D_μλ*D_νσ
    //
    // The derivative d(s1,s2|s3,s4)/dR gives 12 blocks for centers {s1,s2,s3,s4}.
    // For permuted quartets, the derivative blocks are remapped by swapping centers.
    {
        let mut eng = Engine::new_2e_deriv(op, prep, 1e-14)?;
        let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                let b12 = bounds.q[(s1, s2)];
                for s3 in 0..=s1 {
                    let s4max = if s3 == s1 { s2 } else { s3 };
                    for s4 in 0..=s4max {
                        let b34 = bounds.q[(s3, s4)];
                        if b12 * b34 * max_d < 1e-12 { continue; }
                        let deriv = eng.compute_eri_deriv_quartet(prep, s1, s2, s3, s4);
                        if let Some(dq) = deriv {
                            let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                            let block_sz = n1 * n2 * n3 * n4;
                            let atoms = [sh2at[s1], sh2at[s2], sh2at[s3], sh2at[s4]];
                            let sym12 = s1 != s2;
                            let sym34 = s3 != s4;
                            let sym1234 = (s1, s2) != (s3, s4);
                            accum_2e_grad(
                                &mut grad, d, dq, block_sz,
                                n1, n2, n3, n4,
                                offs[s1], offs[s2], offs[s3], offs[s4],
                                &atoms, sym12, sym34, sym1234,
                            );
                        }
                    }
                }
            }
        }
    }

    Ok(grad)
}

/// Accumulate the 2e gradient for one shell quartet, handling all permutational symmetry.
///
/// dq layout: 12 blocks of block_sz each. Block ordering:
///   [dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3, dx4,dy4,dz4]
/// where centers 1,2,3,4 correspond to shells s1,s2,s3,s4.
///
/// The derivative blocks always correspond to the physical shell centers s1..s4.
/// For each permutation of the integral indices, the Γ prefactor changes but the
/// derivative blocks remain the same (they are derivatives w.r.t. the shell centers).
///
/// The total Γ prefactor for each derivative block is the sum of Γ over all
/// equivalent permutations.
fn accum_2e_grad(
    grad: &mut Array2<f64>, d: &Array2<f64>, dq: &[f64], block_sz: usize,
    n1: usize, n2: usize, n3: usize, n4: usize,
    o1: usize, o2: usize, o3: usize, o4: usize,
    atoms: &[usize; 4], sym12: bool, sym34: bool, sym1234: bool,
) {
    for a in 0..n1 {
        for b in 0..n2 {
            for c in 0..n3 {
                for dd in 0..n4 {
                    let idx = ((a * n2 + b) * n3 + c) * n4 + dd;
                    let mu = o1 + a;
                    let nu = o2 + b;
                    let la = o3 + c;
                    let sg = o4 + dd;

                    // Sum Γ over all equivalent permutations of (μ,ν,λ,σ).
                    // Γ_pqrs = 0.5*D_pq*D_rs - 0.25*D_pr*D_qs
                    let mut g = gamma(d, mu, nu, la, sg);

                    if sym12 {
                        g += gamma(d, nu, mu, la, sg);
                    }
                    if sym34 {
                        g += gamma(d, mu, nu, sg, la);
                    }
                    if sym12 && sym34 {
                        g += gamma(d, nu, mu, sg, la);
                    }
                    if sym1234 {
                        g += gamma(d, la, sg, mu, nu);
                        if sym12 {
                            g += gamma(d, la, sg, nu, mu);
                        }
                        if sym34 {
                            g += gamma(d, sg, la, mu, nu);
                        }
                        if sym12 && sym34 {
                            g += gamma(d, sg, la, nu, mu);
                        }
                    }

                    // Accumulate into gradient for each center
                    for center in 0..4 {
                        let atom = atoms[center];
                        for coord in 0..3 {
                            let dv = dq[(center * 3 + coord) * block_sz + idx];
                            grad[(atom, coord)] += g * dv;
                        }
                    }
                }
            }
        }
    }
}

/// Two-particle density matrix element: Γ_μνλσ = 0.5*D_μν*D_λσ - 0.25*D_μλ*D_νσ
#[inline]
fn gamma(d: &Array2<f64>, mu: usize, nu: usize, la: usize, sg: usize) -> f64 {
    0.5 * d[(mu, nu)] * d[(la, sg)] - 0.25 * d[(mu, la)] * d[(nu, sg)]
}

/// UHF two-particle density: Γ_μνλσ = 0.5*D_μν D_λσ − 0.5*(D_α,μλ D_α,νσ + D_β,μλ D_β,νσ)
/// where D = D_α + D_β.
#[inline]
fn gamma_uhf(
    d: &Array2<f64>,
    da: &Array2<f64>,
    db: &Array2<f64>,
    mu: usize,
    nu: usize,
    la: usize,
    sg: usize,
) -> f64 {
    0.5 * d[(mu, nu)] * d[(la, sg)]
        - 0.5 * (da[(mu, la)] * da[(nu, sg)] + db[(mu, la)] * db[(nu, sg)])
}

/// Build the UHF energy-weighted density: W = W_α + W_β with
/// W_σ = Σ_i^occ_σ ε_σi C_σi C_σi^T.
pub fn build_energy_weighted_density_uhf(
    result: &ScfResult,
    nocc_a: usize,
    nocc_b: usize,
) -> Array2<f64> {
    let n = result.mos_alpha.nrows();
    let mut w = Array2::<f64>::zeros((n, n));
    {
        let c = &result.mos_alpha;
        let eps = &result.eps_alpha;
        for mu in 0..n {
            for nu in 0..n {
                let mut sum = 0.0;
                for i in 0..nocc_a {
                    sum += eps[i] * c[(mu, i)] * c[(nu, i)];
                }
                w[(mu, nu)] += sum;
            }
        }
    }
    if let (Some(cb), Some(epsb)) = (result.mos_beta.as_ref(), result.eps_beta.as_ref()) {
        for mu in 0..n {
            for nu in 0..n {
                let mut sum = 0.0;
                for i in 0..nocc_b {
                    sum += epsb[i] * cb[(mu, i)] * cb[(nu, i)];
                }
                w[(mu, nu)] += sum;
            }
        }
    }
    w
}

/// Compute the UHF analytical nuclear gradient.
pub fn uhf_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    result: &ScfResult,
) -> Result<Array2<f64>, FerricError> {
    assert!(matches!(result.spin, Spin::Unrestricted), "uhf_gradient: ScfResult.spin must be Unrestricted");
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;

    let d_total = &result.density_alpha
        + result
            .density_beta
            .as_ref()
            .expect("uhf_gradient: missing density_beta");
    let w = build_energy_weighted_density_uhf(result, nocc_a, nocc_b);
    let mut grad = oneelectron_gradient(mol, prep, &d_total, &w)?;
    grad += &twoelectron_gradient_uhf(
        prep,
        op,
        bounds,
        &d_total,
        &result.density_alpha,
        result.density_beta.as_ref().unwrap(),
    )?;
    Ok(grad)
}

/// Compute the ROHF analytical nuclear gradient.
///
/// ROHF has a single set of MOs partitioned into doubly-occupied (closed),
/// singly-α-occupied (open), and virtual blocks. The total/α/β densities are:
///   D_β = Σ_i^closed C_i C_i^T,   D_α = D_β + Σ_j^open C_j C_j^T,
///   D_total = D_α + D_β = 2 D_β + D_open.
///
/// Energy-weighted density:
///   W = 2 Σ_i^closed ε_i C_i C_i^T + Σ_j^open ε_j C_j C_j^T
/// where ε are the eigenvalues of the Roothaan effective Fock.
///
/// The two-electron piece uses the UHF Γ form with D_α/D_β as above.
pub fn rohf_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    result: &ScfResult,
) -> Result<Array2<f64>, FerricError> {
    assert!(
        matches!(result.spin, Spin::RestrictedOpen),
        "rohf_gradient: ScfResult.spin must be RestrictedOpen"
    );
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let nocc_open = two_s as usize;
    let nocc_double = ((nelec - two_s) / 2) as usize;

    let d_alpha = &result.density_alpha;
    let d_beta = result
        .density_beta
        .as_ref()
        .expect("rohf_gradient: missing density_beta");
    let d_total = d_alpha + d_beta;

    // Energy-weighted density: closed orbitals weighted 2 ε_i, open weighted ε_j.
    let n = result.mos_alpha.nrows();
    let c = &result.mos_alpha;
    let eps = &result.eps_alpha;
    let mut w = Array2::<f64>::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for i in 0..nocc_double {
                sum += 2.0 * eps[i] * c[(mu, i)] * c[(nu, i)];
            }
            for j in nocc_double..nocc_double + nocc_open {
                sum += eps[j] * c[(mu, j)] * c[(nu, j)];
            }
            w[(mu, nu)] = sum;
        }
    }

    let mut grad = oneelectron_gradient(mol, prep, &d_total, &w)?;
    grad += &twoelectron_gradient_uhf(prep, op, bounds, &d_total, d_alpha, d_beta)?;
    Ok(grad)
}

/// UHF four-center two-electron gradient: Σ Γ_uhf_μνλσ d(μν|λσ)/dR.
pub fn twoelectron_gradient_uhf(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d_total: &Array2<f64>,
    d_alpha: &Array2<f64>,
    d_beta: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = prep.shell_to_atom().iter().copied().max().unwrap_or(0) + 1;
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();

    let mut grad = Array2::zeros((natoms, 3));
    let mut eng = Engine::new_2e_deriv(op, prep, 1e-14)?;
    let max_d = d_total.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let b12 = bounds.q[(s1, s2)];
            for s3 in 0..=s1 {
                let s4max = if s3 == s1 { s2 } else { s3 };
                for s4 in 0..=s4max {
                    let b34 = bounds.q[(s3, s4)];
                    if b12 * b34 * max_d < 1e-12 {
                        continue;
                    }
                    let deriv = eng.compute_eri_deriv_quartet(prep, s1, s2, s3, s4);
                    if let Some(dq) = deriv {
                        let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                        let block_sz = n1 * n2 * n3 * n4;
                        let atoms = [sh2at[s1], sh2at[s2], sh2at[s3], sh2at[s4]];
                        let sym12 = s1 != s2;
                        let sym34 = s3 != s4;
                        let sym1234 = (s1, s2) != (s3, s4);
                        accum_2e_grad_uhf(
                            &mut grad,
                            d_total,
                            d_alpha,
                            d_beta,
                            dq,
                            block_sz,
                            n1,
                            n2,
                            n3,
                            n4,
                            offs[s1],
                            offs[s2],
                            offs[s3],
                            offs[s4],
                            &atoms,
                            sym12,
                            sym34,
                            sym1234,
                        );
                    }
                }
            }
        }
    }
    Ok(grad)
}

#[allow(clippy::too_many_arguments)]
fn accum_2e_grad_uhf(
    grad: &mut Array2<f64>,
    d: &Array2<f64>,
    da: &Array2<f64>,
    db: &Array2<f64>,
    dq: &[f64],
    block_sz: usize,
    n1: usize,
    n2: usize,
    n3: usize,
    n4: usize,
    o1: usize,
    o2: usize,
    o3: usize,
    o4: usize,
    atoms: &[usize; 4],
    sym12: bool,
    sym34: bool,
    sym1234: bool,
) {
    for a in 0..n1 {
        for b in 0..n2 {
            for c in 0..n3 {
                for dd in 0..n4 {
                    let idx = ((a * n2 + b) * n3 + c) * n4 + dd;
                    let mu = o1 + a;
                    let nu = o2 + b;
                    let la = o3 + c;
                    let sg = o4 + dd;

                    let mut g = gamma_uhf(d, da, db, mu, nu, la, sg);
                    if sym12 {
                        g += gamma_uhf(d, da, db, nu, mu, la, sg);
                    }
                    if sym34 {
                        g += gamma_uhf(d, da, db, mu, nu, sg, la);
                    }
                    if sym12 && sym34 {
                        g += gamma_uhf(d, da, db, nu, mu, sg, la);
                    }
                    if sym1234 {
                        g += gamma_uhf(d, da, db, la, sg, mu, nu);
                        if sym12 {
                            g += gamma_uhf(d, da, db, la, sg, nu, mu);
                        }
                        if sym34 {
                            g += gamma_uhf(d, da, db, sg, la, mu, nu);
                        }
                        if sym12 && sym34 {
                            g += gamma_uhf(d, da, db, sg, la, nu, mu);
                        }
                    }

                    for center in 0..4 {
                        let atom = atoms[center];
                        for coord in 0..3 {
                            let dv = dq[(center * 3 + coord) * block_sz + idx];
                            grad[(atom, coord)] += g * dv;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhf::{solve_rhf, RhfConfig};
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;

    /// Compute individual gradient components for debugging.
    /// Returns (vnn_grad, overlap_grad, kinetic_grad, nuclear_grad, twoelec_grad, total_grad)
    fn gradient_components(
        mol: &Molecule,
        prep: &PreparedBasis,
        op: Operator,
        bounds: &SchwarzBounds,
        result: &RhfResult,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let natoms = mol.atoms.len();
        let n = prep.nbasis();
        let nsh = prep.nshells();
        let dims = prep.shell_dims();
        let offs = prep.shell_offsets();
        let sh2at = prep.shell_to_atom();
        let d = result.density_r();
        let nocc = (mol.nelec() / 2) as usize;
        let c = result.mos_r();
        let eps = result.eps_r();
        let mut w = Array2::zeros((n, n));
        for mu in 0..n {
            for nu in 0..n {
                let mut sum = 0.0;
                for i in 0..nocc {
                    sum += eps[i] * c[(mu, i)] * c[(nu, i)];
                }
                w[(mu, nu)] = 2.0 * sum;
            }
        }

        let mut vnn_grad = Array2::zeros((natoms, 3));
        let mut overlap_grad = Array2::zeros((natoms, 3));
        let mut kinetic_grad = Array2::zeros((natoms, 3));
        let mut nuclear_grad = Array2::zeros((natoms, 3));
        let mut twoelec_grad = Array2::zeros((natoms, 3));

        // Vnn gradient
        for i in 0..natoms {
            for j in (i + 1)..natoms {
                let a = &mol.atoms[i];
                let b = &mol.atoms[j];
                let dx = a.x - b.x;
                let dy = a.y - b.y;
                let dz = a.zpos - b.zpos;
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = r2.sqrt();
                let za = a.z as f64;
                let zb = b.z as f64;
                let f_over_r3 = za * zb / (r * r2);
                vnn_grad[(i, 0)] -= f_over_r3 * dx;
                vnn_grad[(i, 1)] -= f_over_r3 * dy;
                vnn_grad[(i, 2)] -= f_over_r3 * dz;
                vnn_grad[(j, 0)] += f_over_r3 * dx;
                vnn_grad[(j, 1)] += f_over_r3 * dy;
                vnn_grad[(j, 2)] += f_over_r3 * dz;
            }
        }

        // Overlap (Pulay) gradient
        {
            let mut eng = Engine::new_1e_deriv(ffi::OP_OVERLAP, prep, 1e-14).unwrap();
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    if let Some(deriv) = eng.compute_1e_deriv_block(prep, s1, s2) {
                        let n1 = dims[s1];
                        let n2 = dims[s2];
                        let block_sz = n1 * n2;
                        let a1 = sh2at[s1];
                        let a2 = sh2at[s2];
                        for i in 0..n1 {
                            for j in 0..n2 {
                                let mu = offs[s1] + i;
                                let nu = offs[s2] + j;
                                let idx = i * n2 + j;
                                let wval = if s1 == s2 { w[(mu, nu)] } else { 2.0 * w[(mu, nu)] };
                                for cc in 0..3 {
                                    let d1 = deriv[cc * block_sz + idx];
                                    let d2 = deriv[(3 + cc) * block_sz + idx];
                                    overlap_grad[(a1, cc)] -= wval * d1;
                                    overlap_grad[(a2, cc)] -= wval * d2;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Kinetic gradient
        {
            let mut eng = Engine::new_1e_deriv(ffi::OP_KINETIC, prep, 1e-14).unwrap();
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    if let Some(deriv) = eng.compute_1e_deriv_block(prep, s1, s2) {
                        let n1 = dims[s1];
                        let n2 = dims[s2];
                        let block_sz = n1 * n2;
                        let a1 = sh2at[s1];
                        let a2 = sh2at[s2];
                        for i in 0..n1 {
                            for j in 0..n2 {
                                let mu = offs[s1] + i;
                                let nu = offs[s2] + j;
                                let idx = i * n2 + j;
                                let dval = if s1 == s2 { d[(mu, nu)] } else { 2.0 * d[(mu, nu)] };
                                for cc in 0..3 {
                                    let d1 = deriv[cc * block_sz + idx];
                                    let d2 = deriv[(3 + cc) * block_sz + idx];
                                    kinetic_grad[(a1, cc)] += dval * d1;
                                    kinetic_grad[(a2, cc)] += dval * d2;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Nuclear attraction gradient
        {
            let mut eng = Engine::new_1e_deriv(ffi::OP_NUCLEAR, prep, 1e-14).unwrap();
            eng.set_point_charges(prep);
            let max_fn = prep.shell_dims().iter().copied().max().unwrap_or(1);
            let nderiv_nuclear = 6 + 3 * natoms;
            let max_block = max_fn * max_fn;
            let mut nbuf = vec![0.0f64; nderiv_nuclear * max_block];
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    let n1 = dims[s1];
                    let n2 = dims[s2];
                    let block_sz = n1 * n2;
                    let total = nderiv_nuclear * block_sz;
                    if nbuf.len() < total { nbuf.resize(total, 0.0); }
                    let written = unsafe {
                        ffi::goscf_compute_1e_deriv_block(
                            eng.handle_mut(), prep.handle(),
                            s1 as std::os::raw::c_int, s2 as std::os::raw::c_int,
                            nbuf.as_mut_ptr(),
                        )
                    };
                    if written == 0 { continue; }
                    let a1 = sh2at[s1];
                    let a2 = sh2at[s2];
                    for i in 0..n1 {
                        for j in 0..n2 {
                            let mu = offs[s1] + i;
                            let nu = offs[s2] + j;
                            let idx = i * n2 + j;
                            let dval = if s1 == s2 { d[(mu, nu)] } else { 2.0 * d[(mu, nu)] };
                            for cc in 0..3 {
                                nuclear_grad[(a1, cc)] += dval * nbuf[cc * block_sz + idx];
                                nuclear_grad[(a2, cc)] += dval * nbuf[(3 + cc) * block_sz + idx];
                            }
                            for atom_c in 0..natoms {
                                for cc in 0..3 {
                                    let blk = 6 + atom_c * 3 + cc;
                                    nuclear_grad[(atom_c, cc)] += dval * nbuf[blk * block_sz + idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2e gradient
        {
            let mut eng = Engine::new_2e_deriv(op, prep, 1e-14).unwrap();
            let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    let b12 = bounds.q[(s1, s2)];
                    for s3 in 0..=s1 {
                        let s4max = if s3 == s1 { s2 } else { s3 };
                        for s4 in 0..=s4max {
                            let b34 = bounds.q[(s3, s4)];
                            if b12 * b34 * max_d < 1e-12 { continue; }
                            let deriv = eng.compute_eri_deriv_quartet(prep, s1, s2, s3, s4);
                            if let Some(dq) = deriv {
                                let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                                let block_sz = n1 * n2 * n3 * n4;
                                let atoms = [sh2at[s1], sh2at[s2], sh2at[s3], sh2at[s4]];
                                let sym12 = s1 != s2;
                                let sym34 = s3 != s4;
                                let sym1234 = (s1, s2) != (s3, s4);
                                accum_2e_grad(
                                    &mut twoelec_grad, d, dq, block_sz,
                                    n1, n2, n3, n4,
                                    offs[s1], offs[s2], offs[s3], offs[s4],
                                    &atoms, sym12, sym34, sym1234,
                                );
                            }
                        }
                    }
                }
            }
        }

        let mut total = Array2::zeros((natoms, 3));
        for i in 0..natoms {
            for c in 0..3 {
                total[(i, c)] = vnn_grad[(i, c)] + overlap_grad[(i, c)]
                    + kinetic_grad[(i, c)] + nuclear_grad[(i, c)] + twoelec_grad[(i, c)];
            }
        }

        (vnn_grad, overlap_grad, kinetic_grad, nuclear_grad, twoelec_grad, total)
    }

    #[test]
    fn test_gradient_components_h2() {
        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig { energy_conv: 1e-10, ..Default::default() };
        let result = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &prep, op, &bounds, &config).unwrap();

        let (vnn, overlap, kinetic, nuclear, twoelec, total) =
            gradient_components(&mol, &prep, op, &bounds, &result);

        // Also compute FD of individual energy components
        let delta = 1e-5;
        let natoms = mol.atoms.len();
        let mut fd_vnn = Array2::<f64>::zeros((natoms, 3));
        let mut fd_total = Array2::<f64>::zeros((natoms, 3));
        let config2 = RhfConfig { energy_conv: 1e-10, ..Default::default() };
        for atom in 0..natoms {
            for coord in 0..3 {
                let mut mol_p = mol.clone();
                let mut mol_m = mol.clone();
                match coord {
                    0 => { mol_p.atoms[atom].x += delta; mol_m.atoms[atom].x -= delta; }
                    1 => { mol_p.atoms[atom].y += delta; mol_m.atoms[atom].y -= delta; }
                    _ => { mol_p.atoms[atom].zpos += delta; mol_m.atoms[atom].zpos -= delta; }
                }
                fd_vnn[(atom, coord)] = (mol_p.nuclear_repulsion() - mol_m.nuclear_repulsion()) / (2.0 * delta);

                let bs2 = basis::bundled("sto-3g").unwrap();
                let prep_p = PreparedBasis::new(&mol_p, &bs2).unwrap();
                let bounds_p = SchwarzBounds::compute(Operator::coulomb(), &prep_p).unwrap();
                let res_p = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol_p, &prep_p, Operator::coulomb(), &bounds_p, &config2).unwrap();

                let prep_m = PreparedBasis::new(&mol_m, &bs2).unwrap();
                let bounds_m = SchwarzBounds::compute(Operator::coulomb(), &prep_m).unwrap();
                let res_m = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol_m, &prep_m, Operator::coulomb(), &bounds_m, &config2).unwrap();

                fd_total[(atom, coord)] = (res_p.energy - res_m.energy) / (2.0 * delta);
            }
        }

        eprintln!("=== H2 Gradient Component Breakdown ===");
        for atom in 0..natoms {
            for c in 0..3 {
                eprintln!(
                    "atom={} coord={}: vnn={:12.8} overlap={:12.8} kinetic={:12.8} nuclear={:12.8} 2e={:12.8} total={:12.8} fd_vnn={:12.8} fd_total={:12.8}",
                    atom, c, vnn[(atom, c)], overlap[(atom, c)], kinetic[(atom, c)],
                    nuclear[(atom, c)], twoelec[(atom, c)], total[(atom, c)],
                    fd_vnn[(atom, c)], fd_total[(atom, c)]
                );
            }
        }
        // Verify total matches fd_total
        for atom in 0..natoms {
            for c in 0..3 {
                let diff = (total[(atom, c)] - fd_total[(atom, c)]).abs();
                assert!(diff < 1e-5,
                    "atom={atom} coord={c}: total={:.8} fd={:.8} diff={:.2e}",
                    total[(atom, c)], fd_total[(atom, c)], diff);
            }
        }
    }

    fn finite_difference_gradient(xyz: &str, basis_name: &str, delta: f64) -> Array2<f64> {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let natoms = mol.atoms.len();
        let mut grad = Array2::zeros((natoms, 3));
        let config = RhfConfig { energy_conv: 1e-10, ..Default::default() };

        for atom in 0..natoms {
            for coord in 0..3 {
                let mut mol_plus = mol.clone();
                let mut mol_minus = mol.clone();
                match coord {
                    0 => { mol_plus.atoms[atom].x += delta; mol_minus.atoms[atom].x -= delta; }
                    1 => { mol_plus.atoms[atom].y += delta; mol_minus.atoms[atom].y -= delta; }
                    _ => { mol_plus.atoms[atom].zpos += delta; mol_minus.atoms[atom].zpos -= delta; }
                }
                let bs = basis::bundled(basis_name).unwrap();
                let prep_p = PreparedBasis::new(&mol_plus, &bs).unwrap();
                let bounds_p = SchwarzBounds::compute(Operator::coulomb(), &prep_p).unwrap();
                let res_p = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol_plus, &prep_p, Operator::coulomb(), &bounds_p, &config).unwrap();

                let prep_m = PreparedBasis::new(&mol_minus, &bs).unwrap();
                let bounds_m = SchwarzBounds::compute(Operator::coulomb(), &prep_m).unwrap();
                let res_m = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol_minus, &prep_m, Operator::coulomb(), &bounds_m, &config).unwrap();

                grad[(atom, coord)] = (res_p.energy - res_m.energy) / (2.0 * delta);
            }
        }
        grad
    }

    #[test]
    fn test_gradient_h2_sto3g_vs_finite_diff() {
        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig { energy_conv: 1e-10, ..Default::default() };
        let result = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &prep, op, &bounds, &config).unwrap();

        let analytic = match rhf_gradient(&mol, &prep, op, &bounds, &result) {
            Ok(g) => g,
            Err(FerricError::Libint(msg)) if msg.contains("derivative engine not available") => {
                eprintln!("SKIPPED: {msg}");
                return;
            }
            Err(e) => panic!("unexpected error: {e}"),
        };
        let fd = finite_difference_gradient(xyz, "sto-3g", 1e-5);

        eprintln!("=== H2/STO-3G Gradient ===");
        for atom in 0..2 {
            for c in 0..3 {
                let diff = (analytic[(atom, c)] - fd[(atom, c)]).abs();
                eprintln!("atom={atom} coord={c}: analytic={:.8} fd={:.8} diff={:.2e}",
                    analytic[(atom, c)], fd[(atom, c)], diff);
                assert!(
                    diff < 1e-5,
                    "atom={atom} coord={c}: analytic={:.8} fd={:.8} diff={:.2e}",
                    analytic[(atom, c)], fd[(atom, c)], diff
                );
            }
        }
    }

    #[test]
    fn test_gradient_h2o_sto3g_vs_finite_diff() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig { energy_conv: 1e-10, ..Default::default() };
        let result = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &prep, op, &bounds, &config).unwrap();

        let analytic = match rhf_gradient(&mol, &prep, op, &bounds, &result) {
            Ok(g) => g,
            Err(FerricError::Libint(msg)) if msg.contains("derivative engine not available") => {
                eprintln!("SKIPPED: {msg}");
                return;
            }
            Err(e) => panic!("unexpected error: {e}"),
        };
        let fd = finite_difference_gradient(xyz, "sto-3g", 1e-5);

        eprintln!("=== H2O/STO-3G Gradient ===");
        for atom in 0..3 {
            for c in 0..3 {
                let diff = (analytic[(atom, c)] - fd[(atom, c)]).abs();
                eprintln!("atom={atom} coord={c}: analytic={:.8} fd={:.8} diff={:.2e}",
                    analytic[(atom, c)], fd[(atom, c)], diff);
                assert!(
                    diff < 1e-5,
                    "atom={atom} coord={c}: analytic={:.8} fd={:.8} diff={:.2e}",
                    analytic[(atom, c)], fd[(atom, c)], diff
                );
            }
        }
    }
}
