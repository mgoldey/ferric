//! Analytic coupled-perturbed (CPKS) relaxed MP2 static polarizability.
//!
//! Replaces the finite-field `ff_polar` path, which fails on symmetric molecules
//! (the orbital-relaxation/CPHF response is near-singular under a symmetry-axis
//! field → 1/F divergence in the differenced dipole; see
//! attenuated-mp2-alpha-experiment memory). Analytic CPKS solves the response
//! ONCE per Cartesian direction against a smooth RHS — no finite difference, so
//! the 1/F cancellation catastrophe is gone by construction.
//!
//! Three layers (2n+1 rule → first-order responses only):
//!   1. CPHF orbital response U^x: (Δε+A) U^x = −μ^x_ov   (CG, reuse compute_az_product)
//!   2. perturbed MP2 amplitudes ∂t2/∂F^x
//!   3. perturbed relaxed density: perturbed Lagrangian L^x + second CG solve z^x
//! Then α_xy = −2 Σ (∂P_relax/∂F^x)_pq μ^y_pq.
//!
//! Closed-shell (Restricted), static (ω=0). Attenuated operator via `op`.

use ferric_core::mol::Molecule;
use ferric_core::orbitals::OrbitalSpace;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::result::{ScfResult, Spin};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

use crate::ff_polar::{eig3_sym_pub as eig3_sym, Mp2Polarizability};
use crate::rimp2::RiMp2Config;

/// Analytic relaxed MP2 static polarizability (closed-shell). Stub until Stage 2.
#[allow(clippy::too_many_arguments)]
pub fn mp2_polarizability_analytic(
    _ctx: &ParallelContext,
    _mol: &Molecule,
    _obs: &PreparedBasis,
    _dfbs: &PreparedBasis,
    _op: Operator,
    _bounds: &SchwarzBounds,
    rhf: &ScfResult,
    _mp2_config: &RiMp2Config,
) -> Result<Mp2Polarizability, FerricError> {
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "mp2_polarizability_analytic: closed-shell (Restricted) only".into(),
        ));
    }
    Err(FerricError::General(
        "mp2_polarizability_analytic: not yet implemented (Stage 2)".into(),
    ))
}

/// Solve (Δε + A) X = rhs by Jacobi-preconditioned CG. Reuses the production
/// orbital-Hessian matvec `compute_az_product`. `rhs` and the return are shape
/// (nvir, nocc). Returns (X, final_residual, iters, converged).
pub(crate) fn solve_cphf_cg(
    c: &Array2<f64>,
    rhs: &Array2<f64>,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    orb: &OrbitalSpace,
    eps: &[f64],
) -> Result<(Array2<f64>, f64, usize, bool), FerricError> {
    use crate::zvector::compute_az_product;
    let OrbitalSpace {
        nocc,
        nvir,
        nocc_total,
        first_occ,
    } = *orb;
    let de = |a: usize, i: usize| eps[nocc_total + a] - eps[first_occ + i];

    // A-coupling scale 0.5: `compute_az_product` builds the 2e product with a
    // SYMMETRIC trial density dz = z(C_μa C_νi + C_μi C_νa) — both orderings —
    // which double-counts relative to the CPHF-α orbital Hessian (the Z-vector
    // path it was written for absorbs this elsewhere). Empirically pinned against
    // the finite-field HF α oracle: ASCALE=0.5 + contraction −4 reproduces the
    // FF-HF tensor exactly (water/cc-pVDZ [3.04, 6.91, 5.11]).
    const ASCALE: f64 = 0.5;
    let apply = |z: &Array2<f64>| -> Result<Array2<f64>, FerricError> {
        let mut mz = compute_az_product(c, z, prep, bounds, orb)?;
        for a in 0..nvir {
            for i in 0..nocc {
                mz[(a, i)] = ASCALE * mz[(a, i)] + de(a, i) * z[(a, i)];
            }
        }
        Ok(mz)
    };
    let dot = |x: &Array2<f64>, y: &Array2<f64>| -> f64 {
        let mut s = 0.0;
        for a in 0..nvir {
            for i in 0..nocc {
                s += x[(a, i)] * y[(a, i)];
            }
        }
        s
    };
    let precond = |r: &Array2<f64>| -> Array2<f64> {
        let mut z = Array2::<f64>::zeros((nvir, nocc));
        for a in 0..nvir {
            for i in 0..nocc {
                let d = de(a, i);
                if d.abs() > 1e-12 {
                    z[(a, i)] = r[(a, i)] / d;
                }
            }
        }
        z
    };

    let mut x = precond(rhs);
    let mut r = rhs - &apply(&x)?;
    let mut z_pc = precond(&r);
    let mut p = z_pc.clone();
    let mut rz_old = dot(&r, &z_pc);

    let max_iter = 200;
    let tol = 1e-10;
    let trace = std::env::var("FERRIC_CPKS_TRACE").ok().as_deref() == Some("1");
    let mut resid_max = r.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let mut converged = false;
    let mut it_done = 0;
    for it in 0..max_iter {
        it_done = it;
        resid_max = r.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if trace {
            eprintln!("  [cpks-cg] iter={it:3} max_resid={resid_max:.3e}");
        }
        if resid_max < tol {
            converged = true;
            break;
        }
        let mp = apply(&p)?;
        let denom = dot(&p, &mp);
        if denom.abs() < 1e-30 {
            break;
        }
        let alpha = rz_old / denom;
        for a in 0..nvir {
            for i in 0..nocc {
                x[(a, i)] += alpha * p[(a, i)];
                r[(a, i)] -= alpha * mp[(a, i)];
            }
        }
        z_pc = precond(&r);
        let rz_new = dot(&r, &z_pc);
        let beta = rz_new / rz_old;
        for a in 0..nvir {
            for i in 0..nocc {
                p[(a, i)] = z_pc[(a, i)] + beta * p[(a, i)];
            }
        }
        rz_old = rz_new;
    }
    Ok((x, resid_max, it_done + 1, converged))
}

/// MO-basis dipole ov blocks μ^d_{ia} = (C_occ^T D^d_AO C_vir), d∈{x,y,z},
/// each shape (nocc, nvir).
pub(crate) fn dipole_ov_mo(
    obs: &PreparedBasis,
    c: &Array2<f64>,
    orb: &OrbitalSpace,
) -> [Array2<f64>; 3] {
    let OrbitalSpace {
        nocc,
        nvir,
        nocc_total,
        first_occ,
    } = *orb;
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c
        .slice(ndarray::s![.., nocc_total..nocc_total + nvir])
        .to_owned();
    std::array::from_fn(|d| c_occ.t().dot(&dip_ao[d]).dot(&c_vir))
}

/// HF-level analytic (CPHF) polarizability — Layer 1 only. Validation rung and
/// the orbital-response core reused by the full MP2 path.
pub fn mp2_polarizability_analytic_hf(
    _ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    _op: &Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
) -> Result<Mp2Polarizability, FerricError> {
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General("cpks HF α: Restricted only".into()));
    }
    let c = rhf.mos_r();
    let eps = rhf.eps_r();
    let nbas = obs.nbasis();
    let nocc = (mol.nelec() / 2) as usize;
    let nvir = nbas - nocc;
    let orb = OrbitalSpace::new(nocc, nvir, nocc, 0);

    // μ^d_ov as (nvir, nocc) (CG convention).
    let mu_oc = dipole_ov_mo(obs, c, &orb); // [d] = (nocc, nvir)
    let mu: [Array2<f64>; 3] = std::array::from_fn(|d| mu_oc[d].t().to_owned()); // (nvir,nocc)

    // U^x: (Δε+A) U^x = −μ^x
    let mut u: Vec<Array2<f64>> = Vec::with_capacity(3);
    for x in 0..3 {
        let rhs = -&mu[x];
        let (ux, resid, iters, conv) = solve_cphf_cg(c, &rhs, obs, bounds, &orb, eps)?;
        if !conv {
            return Err(FerricError::General(format!(
                "cpks HF α: CPHF U^{x} did not converge (resid={resid:.2e}, iters={iters})"
            )));
        }
        u.push(ux);
    }

    // α_xy = −4 Σ_ia μ^y_ia U^x_ia. Closed-shell factor 4 = 2 (spin) × 2
    // (response symmetry); sign from α = −∂μ/∂F. Pinned with ASCALE=0.5 against
    // the finite-field HF α oracle.
    const CONTRACT: f64 = -4.0;
    let mut tensor = [[0.0_f64; 3]; 3];
    for x in 0..3 {
        for y in 0..3 {
            let mut s = 0.0;
            for a in 0..nvir {
                for i in 0..nocc {
                    s += mu[y][(a, i)] * u[x][(a, i)];
                }
            }
            tensor[x][y] = CONTRACT * s;
        }
    }
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg = 0.5 * (tensor[i][j] + tensor[j][i]);
            tensor[i][j] = avg;
            tensor[j][i] = avg;
        }
    }
    let iso = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;
    let principal = eig3_sym(tensor);
    Ok(Mp2Polarizability {
        tensor,
        iso,
        principal,
    })
}

// ===========================================================================
// Layer 2: first-order perturbed MP2 amplitudes ∂t2/∂F^x.
//
// t2_iajb = (ia|jb)/Δε,  Δε = εi+εj−εa−εb,  (ia|jb)=Σ_P B^P_ia B^P_jb.
// Field along axis x rotates orbitals by the CPHF U^x (vir,occ):
//   ∂C_·i = Σ_c U_ci C_·c ;  ∂C_·a = −Σ_k U_ak C_·k
// ⇒ ∂B^P_ia = Σ_c U_ci B^P_ca − Σ_k U_ak B^P_ik         (uses dressed b_vv, b_oo)
//   ∂(ia|jb) = Σ_P [∂B^P_ia B^P_jb + B^P_ia ∂B^P_jb]
//   ∂ε_p     = h^x_pp + Σ_ck U_ck [2(pp|ck) − (pc|pk)]   (h^x = −μ_x; perturbed-Fock diag)
//   ∂t2      = [∂(ia|jb) − t2·(∂εi+∂εj−∂εa−∂εb)] / Δε
//
// All B blocks come pre-dressed from compute_mp2_intermediates — no AO rebuild.
// Validated against the central difference of t2 from a field-perturbed RHF.
// ===========================================================================

use crate::rimp2::compute_mp2_intermediates;

/// Reshape a dressed B block (naux, n*m) → indexable [(P,r,c)] via closure.
#[inline]
fn bget(b: &Array2<f64>, p: usize, r: usize, c: usize, m: usize) -> f64 {
    b[(p, r * m + c)]
}

/// Analytic ∂t2/∂F along `axis`. Returns t2-shaped Vec (nov*nov), index ia*nov+jb,
/// alongside the U^x used (for downstream layers).
pub fn analytic_dt2_along(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    mp2_config: &RiMp2Config,
    axis: usize,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    let _ = ctx;
    let inter = compute_mp2_intermediates(mol, obs, dfbs, op, rhf, mp2_config)?;
    let (nocc, nvir, naux) = (inter.nocc, inter.nvir, inter.naux);
    let (first_occ, nocc_total) = (inter.first_occ, inter.nocc_total);
    let nov = nocc * nvir;
    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);

    // --- CPHF U^x: (Δε + 0.5 A) U = −μ^x_ov  (same operator/conventions as HF α) ---
    let mu_oc = dipole_ov_mo(obs, c, &orb); // (nocc,nvir)
    let mu_vo = mu_oc[axis].t().to_owned(); // (nvir,nocc)
    let (u, resid, iters, conv) = solve_cphf_cg(c, &(-&mu_vo), obs, bounds, &orb, eps)?;
    if !conv {
        return Err(FerricError::General(format!(
            "analytic_dt2: CPHF U^{axis} not converged (resid={resid:.2e}, iters={iters})"
        )));
    }

    // --- ∂B^P_ia = Σ_c U_ci b_vv[P,c,a] − Σ_k U_ak b_oo[P,i,k] ---
    let mut db_ov = Array2::<f64>::zeros((naux, nov));
    for p in 0..naux {
        for i in 0..nocc {
            for a in 0..nvir {
                let mut s = 0.0;
                for cc in 0..nvir {
                    s += u[(cc, i)] * bget(&inter.b_vv, p, cc, a, nvir);
                }
                for k in 0..nocc {
                    s -= u[(a, k)] * bget(&inter.b_oo, p, i, k, nocc);
                }
                db_ov[(p, i * nvir + a)] = s;
            }
        }
    }

    // --- ∂ε_p = −μ_pp + Σ_ck U_ck [2(pp|ck) − (pc|pk)] ---
    // Need full-MO dipole diagonal and the coupling integrals via dressed B.
    // μ in MO: μ_pq = Cᵀ D^x_AO C.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    let mu_mo = c.t().dot(&dip_ao[axis]).dot(c); // (nmo,nmo)
    // (pp|ck): for occ p=i → Σ_P b_oo[P,i,i] b_ov[P,c? ...]; ck is occ-vir (k occ, c vir).
    // Use B_ov for (ck): (ck)≡(k c) with k occ, c vir → b_ov[P, k*nvir + c].
    // (pp|ck) = Σ_P B^P_pp B^P_kc ;  (pc|pk): p occ-or-vir mixed — handle per block.
    let de = |i: usize, j: usize, a: usize, b: usize| {
        eps[first_occ + i] + eps[first_occ + j] - eps[nocc_total + a] - eps[nocc_total + b]
    };

    // ∂ε_p = ∂F_pp = (Cᵀ ∂F_AO C)_pp, with the FIRST-ORDER Fock
    //   ∂F_AO = ∂h_AO + G[∂D_AO],   ∂h_AO = −r_axis (uniform field),
    //   ∂D_AO = 2 Σ_ai U_ai (C_·a C_·iᵀ + C_·i C_·aᵀ)   (U-driven density response),
    //   G[∂D] = 2 J[∂D] − K[∂D]   (closed-shell two-electron response).
    // Textbook perturbed-Fock-diagonal form — replaces the ad-hoc (pp|ck)
    // contractions. (Note: −r_axis sign convention is pinned by the FD oracle.)
    let nbas = obs.nbasis();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();
    // ∂D_AO from U (factor 2 = closed-shell occupancy).
    let mut dd_ao = Array2::<f64>::zeros((nbas, nbas));
    for a in 0..nvir {
        for i in 0..nocc {
            let uai = u[(a, i)];
            if uai == 0.0 {
                continue;
            }
            for mu in 0..nbas {
                let cma = c_vir[(mu, a)];
                let cmi = c_occ[(mu, i)];
                for nu in 0..nbas {
                    dd_ao[(mu, nu)] += 2.0 * uai * (cma * c_occ[(nu, i)] + cmi * c_vir[(nu, a)]);
                }
            }
        }
    }
    let mut jdd = Array2::<f64>::zeros((nbas, nbas));
    let mut kdd = Array2::<f64>::zeros((nbas, nbas));
    ferric_scf::rhf::build_jk(ctx, obs, bounds, 1e-12, &dd_ao, &mut jdd, &mut kdd)?;
    let gscale: f64 = std::env::var("CPKS_GSCALE").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let g_ao = gscale * (2.0 * &jdd - &kdd);
    // ∂F_AO = −r_axis + G[∂D];  ∂F_MO = Cᵀ ∂F_AO C
    let df_ao = &(-&dip_ao[axis]) + &g_ao;
    let df_mo = c.t().dot(&df_ao).dot(c);
    let _ = &mu_mo; // (kept for reference; diagonal now from df_mo directly)
    let mut deps_o = vec![0.0f64; nocc];
    let mut deps_v = vec![0.0f64; nvir];
    for i in 0..nocc {
        deps_o[i] = df_mo[(first_occ + i, first_occ + i)];
    }
    for a in 0..nvir {
        deps_v[a] = df_mo[(nocc_total + a, nocc_total + a)];
    }

    // --- ∂t2_iajb = [∂(ia|jb) − t2·∂Δε] / Δε ---
    let mut dt2 = vec![0.0f64; nov * nov];
    for i in 0..nocc {
        for a in 0..nvir {
            let ia = i * nvir + a;
            for j in 0..nocc {
                for b in 0..nvir {
                    let jb = j * nvir + b;
                    let d_iajb: f64 = (0..naux)
                        .map(|p| {
                            db_ov[(p, ia)] * inter.b_ov[(p, jb)]
                                + inter.b_ov[(p, ia)] * db_ov[(p, jb)]
                        })
                        .sum();
                    let ddenom = deps_o[i] + deps_o[j] - deps_v[a] - deps_v[b];
                    let denom = de(i, j, a, b);
                    let t = inter.t2[ia * nov + jb];
                    dt2[ia * nov + jb] = (d_iajb - t * ddenom) / denom;
                }
            }
        }
    }

    Ok((dt2, u))
}

/// FD ORACLE: ∂t2/∂F along `axis` via central difference of t2 rebuilt from a
/// field-perturbed RHF. Returns the t2-shaped Vec. Used only to validate
/// `analytic_dt2_along`.
#[allow(clippy::too_many_arguments)]
pub fn fd_dt2_along(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    scf_config: &ferric_scf::rhf::RhfConfig,
    mp2_config: &RiMp2Config,
    axis: usize,
    h: f64,
) -> Result<Vec<f64>, FerricError> {
    let n = obs.nbasis();
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    let t2_at = |field: f64| -> Result<Vec<f64>, FerricError> {
        let mut v = Array2::<f64>::zeros((n, n));
        for mu in 0..n {
            for nu in 0..n {
                v[(mu, nu)] = -field * dip_ao[axis][(mu, nu)];
            }
        }
        let rhf_f = crate::ff_polar::solve_rhf_with_external(ctx, mol, obs, bounds, scf_config, &v)?;
        let inter = compute_mp2_intermediates(mol, obs, dfbs, op, &rhf_f, mp2_config)?;
        Ok(inter.t2)
    };
    let tp = t2_at(h)?;
    let tm = t2_at(-h)?;
    Ok(tp.iter().zip(tm.iter()).map(|(p, m)| (p - m) / (2.0 * h)).collect())
}

/// Analytic ∂E_MP2/∂F along `axis` — a GAUGE-INVARIANT scalar (immune to the
/// occ/vir orbital-rotation phase ambiguity that contaminates element-wise FD of
/// t2). E_MP2 = Σ_iajb t2_iajb [2(ia|jb) − (ib|ja)]; differentiate via ∂t2 and
/// ∂(ia|jb). The clean Layer-2 validation gate.
#[allow(clippy::too_many_arguments)]
pub fn analytic_de_mp2_along(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    mp2_config: &RiMp2Config,
    axis: usize,
) -> Result<f64, FerricError> {
    let inter = compute_mp2_intermediates(mol, obs, dfbs, op, rhf, mp2_config)?;
    let (nocc, nvir, naux) = (inter.nocc, inter.nvir, inter.naux);
    let nov = nocc * nvir;
    let (dt2, u) = analytic_dt2_along(ctx, mol, obs, dfbs, op, bounds, rhf, mp2_config, axis)?;

    // ∂B_ov (rebuild here from U for ∂(ia|jb); mirrors analytic_dt2_along).
    let mut db_ov = Array2::<f64>::zeros((naux, nov));
    for p in 0..naux {
        for i in 0..nocc {
            for a in 0..nvir {
                let mut s = 0.0;
                for cc in 0..nvir {
                    s += u[(cc, i)] * bget(&inter.b_vv, p, cc, a, nvir);
                }
                for k in 0..nocc {
                    s -= u[(a, k)] * bget(&inter.b_oo, p, i, k, nocc);
                }
                db_ov[(p, i * nvir + a)] = s;
            }
        }
    }
    let eri = |ia: usize, jb: usize| (0..naux).map(|p| inter.b_ov[(p, ia)] * inter.b_ov[(p, jb)]).sum::<f64>();
    let deri = |ia: usize, jb: usize| {
        (0..naux).map(|p| db_ov[(p, ia)] * inter.b_ov[(p, jb)] + inter.b_ov[(p, ia)] * db_ov[(p, jb)]).sum::<f64>()
    };
    let mut de = 0.0;
    for i in 0..nocc {
        for a in 0..nvir {
            let ia = i * nvir + a;
            for j in 0..nocc {
                for b in 0..nvir {
                    let jb = j * nvir + b;
                    let ib = i * nvir + b;
                    let ja = j * nvir + a;
                    let k = 2.0 * eri(ia, jb) - eri(ib, ja);
                    let dk = 2.0 * deri(ia, jb) - deri(ib, ja);
                    let w_t = std::env::var("CPKS_W_DT2").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0_f64);
                    let w_k = std::env::var("CPKS_W_DK").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0_f64);
                    de += w_t * dt2[ia * nov + jb] * k + w_k * inter.t2[ia * nov + jb] * dk;
                }
            }
        }
    }
    Ok(de)
}

/// FD ORACLE for ∂E_MP2/∂F: central difference of the MP2 correlation energy
/// (a scalar → gauge-stable, unlike element-wise t2). Validates analytic_de.
#[allow(clippy::too_many_arguments)]
pub fn fd_de_mp2_along(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    scf_config: &ferric_scf::rhf::RhfConfig,
    mp2_config: &RiMp2Config,
    axis: usize,
    h: f64,
) -> Result<f64, FerricError> {
    let n = obs.nbasis();
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    let e_at = |field: f64| -> Result<f64, FerricError> {
        let mut v = Array2::<f64>::zeros((n, n));
        for mu in 0..n {
            for nu in 0..n {
                v[(mu, nu)] = -field * dip_ao[axis][(mu, nu)];
            }
        }
        let rhf_f = crate::ff_polar::solve_rhf_with_external(ctx, mol, obs, bounds, scf_config, &v)?;
        let inter = compute_mp2_intermediates(mol, obs, dfbs, op, &rhf_f, mp2_config)?;
        Ok(inter.e_mp2)
    };
    Ok((e_at(h)? - e_at(-h)?) / (2.0 * h))
}
