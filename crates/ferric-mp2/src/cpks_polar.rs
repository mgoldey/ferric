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
    // Default operator (Δε + 0.5 A) — the dipole-CPHF (HF α) convention.
    solve_cphf_cg_scaled(c, rhs, prep, bounds, orb, eps, 0.5)
}

/// As `solve_cphf_cg` but with an explicit A-coupling scale. Use 0.5 for the
/// dipole-CPHF (HF α) path; use 1.0 to MATCH `solve_zvector`'s full-A operator
/// when solving the PERTURBED Z-vector ∂z (so ∂z and the un-perturbed z0 from
/// solve_zvector share the same operator — required for the relaxed-α response).
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cphf_cg_scaled(
    c: &Array2<f64>,
    rhs: &Array2<f64>,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    orb: &OrbitalSpace,
    eps: &[f64],
    ascale: f64,
) -> Result<(Array2<f64>, f64, usize, bool), FerricError> {
    use crate::zvector::compute_az_product;
    let OrbitalSpace {
        nocc,
        nvir,
        nocc_total,
        first_occ,
    } = *orb;
    let de = |a: usize, i: usize| eps[nocc_total + a] - eps[first_occ + i];

    // A-coupling scale: 0.5 for dipole-CPHF (compute_az_product's symmetric dz
    // double-counts vs the CPHF-α Hessian — pinned vs FF-HF), 1.0 to match
    // solve_zvector's full-A Z-vector operator.
    let apply = |z: &Array2<f64>| -> Result<Array2<f64>, FerricError> {
        let mut mz = compute_az_product(c, z, prep, bounds, orb)?;
        for a in 0..nvir {
            for i in 0..nocc {
                mz[(a, i)] = ascale * mz[(a, i)] + de(a, i) * z[(a, i)];
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

/// Analytic ∂t2/∂F along `axis`. Returns (dt2 [nov*nov, ia*nov+jb], U^x, ∂f_mo)
/// — the full set of first-order responses downstream layers reuse.
#[allow(clippy::too_many_arguments)]
pub fn analytic_dt2_full(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    mp2_config: &RiMp2Config,
    axis: usize,
) -> Result<(Vec<f64>, Array2<f64>, Array2<f64>), FerricError> {
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

    Ok((dt2, u, df_mo))
}

/// 2-tuple wrapper (dt2, u) for callers that don't need ∂f_mo.
#[allow(clippy::too_many_arguments)]
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
    let (dt2, u, _df) =
        analytic_dt2_full(ctx, mol, obs, dfbs, op, bounds, rhf, mp2_config, axis)?;
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
    // NOTE: tested the orbital-relaxation term Σ_ai L_ai U^x_ai (MP2 Lagrangian
    // contracted with the CPHF response) as the candidate missing ~9% — it has the
    // WRONG sign/magnitude (W_LAG=+1 → 0.51× FD, worse), so it is NOT the residual.
    // Ruled out empirically; left out. The remaining 9% (analytic 0.906× FD) is an
    // unidentified term in the ∂E_MP2 assembly — flagged for review.
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

/// DIAGNOSTIC: analytic U-driven SCF density response ∂D vs FD of the converged
/// SCF density. Returns (‖analytic‖, ‖fd‖, max|Δ|). Validates U normalization.
#[allow(clippy::too_many_arguments)]
pub fn debug_dd_norms(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    _op: &Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    scf_config: &ferric_scf::rhf::RhfConfig,
    axis: usize,
    h: f64,
) -> Result<(f64, f64, f64), FerricError> {
    let nbas = obs.nbasis();
    let nocc = (mol.nelec() / 2) as usize;
    let nvir = nbas - nocc;
    let c = rhf.mos_r();
    let eps = rhf.eps_r();
    let orb = OrbitalSpace::new(nocc, nvir, nocc, 0);
    // analytic ∂D from U
    let mu_oc = dipole_ov_mo(obs, c, &orb);
    let mu_vo = mu_oc[axis].t().to_owned();
    let (u, _r, _it, _cv) = solve_cphf_cg(c, &(-&mu_vo), obs, bounds, &orb, eps)?;
    let c_occ = c.slice(ndarray::s![.., 0..nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc..]).to_owned();
    let mut dd = Array2::<f64>::zeros((nbas, nbas));
    for a in 0..nvir {
        for i in 0..nocc {
            let uai = u[(a, i)];
            for mu in 0..nbas {
                for nu in 0..nbas {
                    dd[(mu, nu)] += 2.0 * uai * (c_vir[(mu, a)] * c_occ[(nu, i)] + c_occ[(mu, i)] * c_vir[(nu, a)]);
                }
            }
        }
    }
    // FD of SCF total density
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    let dens = |field: f64| -> Result<Array2<f64>, FerricError> {
        let mut v = Array2::<f64>::zeros((nbas, nbas));
        for m in 0..nbas { for n in 0..nbas { v[(m, n)] = -field * dip_ao[axis][(m, n)]; } }
        let r = crate::ff_polar::solve_rhf_with_external(ctx, mol, obs, bounds, scf_config, &v)?;
        Ok(r.density_total().to_owned())
    };
    let dp = dens(h)?;
    let dm = dens(-h)?;
    let fd = (&dp - &dm).mapv(|x| x / (2.0 * h));
    let an_n = dd.iter().map(|x| x * x).sum::<f64>().sqrt();
    let fd_n = fd.iter().map(|x| x * x).sum::<f64>().sqrt();
    let maxd = dd.iter().zip(fd.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    Ok((an_n, fd_n, maxd))
}

/// DIAGNOSTIC: contract analytic ∂D and FD ∂D with each dipole component →
/// Tr[∂D·r_y]. For field along `axis`, −Tr[∂D·r_y] = α^HF_{y,axis}. Tells which
/// ∂D (analytic vs FD) is the trustworthy one (matches the validated HF α).
#[allow(clippy::too_many_arguments)]
pub fn debug_dd_traces(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    scf_config: &ferric_scf::rhf::RhfConfig,
    axis: usize,
    h: f64,
) -> Result<([f64; 3], [f64; 3]), FerricError> {
    let nbas = obs.nbasis();
    let nocc = (mol.nelec() / 2) as usize;
    let nvir = nbas - nocc;
    let c = rhf.mos_r();
    let eps = rhf.eps_r();
    let orb = OrbitalSpace::new(nocc, nvir, nocc, 0);
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
    let mu_oc = dipole_ov_mo(obs, c, &orb);
    let mu_vo = mu_oc[axis].t().to_owned();
    let (u, _r, _it, _cv) = solve_cphf_cg(c, &(-&mu_vo), obs, bounds, &orb, eps)?;
    let c_occ = c.slice(ndarray::s![.., 0..nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc..]).to_owned();
    let mut dd = Array2::<f64>::zeros((nbas, nbas));
    for a in 0..nvir { for i in 0..nocc {
        let uai = u[(a, i)];
        for mu in 0..nbas { for nu in 0..nbas {
            dd[(mu, nu)] += 2.0 * uai * (c_vir[(mu, a)] * c_occ[(nu, i)] + c_occ[(mu, i)] * c_vir[(nu, a)]);
        }}
    }}
    let dens = |field: f64| -> Result<Array2<f64>, FerricError> {
        let mut v = Array2::<f64>::zeros((nbas, nbas));
        for m in 0..nbas { for n in 0..nbas { v[(m, n)] = -field * dip_ao[axis][(m, n)]; } }
        let r = crate::ff_polar::solve_rhf_with_external(ctx, mol, obs, bounds, scf_config, &v)?;
        Ok(r.density_total().to_owned())
    };
    let fd = (&dens(h)? - &dens(-h)?).mapv(|x| x / (2.0 * h));
    let tr = |d: &Array2<f64>| -> [f64; 3] {
        std::array::from_fn(|y| -(d * &dip_ao[y]).sum())
    };
    Ok((tr(&dd), tr(&fd)))
}

/// DIAGNOSTIC: E_MP2 at a single field along `axis` (for parabola/sign checks).
pub fn debug_emp2_at_field(
    ctx: &ParallelContext, mol: &Molecule, obs: &PreparedBasis, dfbs: &PreparedBasis,
    op: Operator, bounds: &SchwarzBounds, scf_config: &ferric_scf::rhf::RhfConfig,
    mp2_config: &RiMp2Config, axis: usize, field: f64,
) -> Result<f64, FerricError> {
    let n = obs.nbasis();
    let dip = oneelectron::dipole(obs, [0.0,0.0,0.0]);
    let mut v = Array2::<f64>::zeros((n,n));
    for m in 0..n { for k in 0..n { v[(m,k)] = -field*dip[axis][(m,k)]; } }
    let r = crate::ff_polar::solve_rhf_with_external(ctx, mol, obs, bounds, scf_config, &v)?;
    Ok(compute_mp2_intermediates(mol, obs, dfbs, op, &r, mp2_config)?.e_mp2)
}

// ===========================================================================
// Layer 3 (PySCF-mirrored): analytic ∂(relaxed dm1)/∂F^x, then
//   α_xy = −Tr[(∂dm1_relaxed/∂F^x)_AO · r_y_AO].
//
// Mirrors pyscf/grad/mp2.py relaxed-dm1 assembly (~/qc/pyscf):
//   dm1mo = [2δ+P_oo]_oo + [P_vv]_vv + z_vo  (z = _response_dm1 Z-vector)
//   = exactly ferric build_relaxed_density_ao.
// Its field derivative:
//   ∂P_oo, ∂P_vv  : product rule on ∂t2 (build_mp2_density is bilinear in t2)
//   ∂z            : perturbed Z-vector  (Δε+0.5A) ∂z = ∂L − ∂(Δε+0.5A)·z
// The 2δ core is field-independent. Contract with the dipole for α.
// Inner oracle: ff_polar finite-field RELAXED α on water/ch4 (FF-stable there);
// outer gate: PySCF relaxed MP2 α.
// ===========================================================================

/// ∂P_oo, ∂P_vv from ∂t2 via the product rule (build_mp2_density is bilinear:
/// P = f(t2,t2), so ∂P = f(∂t2,t2) + f(t2,∂t2)).
fn dmp2_density_response(
    t2: &[f64],
    dt2: &[f64],
    nocc: usize,
    nvir: usize,
) -> (Array2<f64>, Array2<f64>) {
    let nov = nocc * nvir;
    let mut dp_oo = Array2::<f64>::zeros((nocc, nocc));
    for i in 0..nocc {
        for j in 0..nocc {
            let mut s = 0.0;
            for k in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let ik = (i * nvir + a) * nov + k * nvir + b;
                        let jk = (j * nvir + a) * nov + k * nvir + b;
                        let jkb = (j * nvir + b) * nov + k * nvir + a;
                        // ∂[ t_ik(2t_jk − t_jk') ]
                        s += dt2[ik] * (2.0 * t2[jk] - t2[jkb])
                            + t2[ik] * (2.0 * dt2[jk] - dt2[jkb]);
                    }
                }
            }
            dp_oo[(i, j)] = -s;
        }
    }
    let mut dp_vv = Array2::<f64>::zeros((nvir, nvir));
    for a in 0..nvir {
        for b in 0..nvir {
            let mut s = 0.0;
            for i in 0..nocc {
                for j in 0..nocc {
                    for cc in 0..nvir {
                        let ac = (i * nvir + a) * nov + j * nvir + cc;
                        let bc = (i * nvir + b) * nov + j * nvir + cc;
                        let cb = (i * nvir + cc) * nov + j * nvir + b;
                        s += dt2[ac] * (2.0 * t2[bc] - t2[cb])
                            + t2[ac] * (2.0 * dt2[bc] - dt2[cb]);
                    }
                }
            }
            dp_vv[(a, b)] = s;
        }
    }
    (dp_oo, dp_vv)
}

/// Layer 3 (INCREMENTAL): α from the amplitude part of ∂dm1_relaxed only
/// (∂P_oo + ∂P_vv from ∂t2), WITHOUT the perturbed Z-vector ∂z yet. Returns the
/// partial α tensor. Used to measure how much of the relaxed α the amplitude
/// response captures vs the FF-relaxed oracle, before adding ∂z. The 2δ core is
/// field-independent (drops); the ov/vo ∂z block is the remaining piece.
#[allow(clippy::too_many_arguments)]
pub fn analytic_alpha_amplitude_only(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    mp2_config: &RiMp2Config,
) -> Result<Mp2Polarizability, FerricError> {
    let inter = compute_mp2_intermediates(mol, obs, dfbs, op, rhf, mp2_config)?;
    let (nocc, nvir) = (inter.nocc, inter.nvir);
    let (first_occ, nocc_total) = (inter.first_occ, inter.nocc_total);
    let c = rhf.mos_r();
    let nmo = c.ncols();
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);

    let mut tensor = [[0.0f64; 3]; 3];
    for x in 0..3 {
        let (dt2, _u) = analytic_dt2_along(ctx, mol, obs, dfbs, op, bounds, rhf, mp2_config, x)?;
        let (dp_oo, dp_vv) = dmp2_density_response(&inter.t2, &dt2, nocc, nvir);
        // ∂dm1_mo (amplitude part only): occ-occ ∂P_oo (sym), vir-vir ∂P_vv (sym).
        let mut ddm = Array2::<f64>::zeros((nmo, nmo));
        for i in 0..nocc {
            for j in 0..nocc {
                ddm[(first_occ + i, first_occ + j)] = dp_oo[(i, j)] + dp_oo[(j, i)];
            }
        }
        for a in 0..nvir {
            for b in 0..nvir {
                ddm[(nocc_total + a, nocc_total + b)] = dp_vv[(a, b)] + dp_vv[(b, a)];
            }
        }
        let ddm_ao = c.dot(&ddm).dot(&c.t());
        for y in 0..3 {
            tensor[x][y] = -(&ddm_ao * &dip_ao[y]).sum();
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
    Ok(Mp2Polarizability { tensor, iso, principal })
}

/// Full-MO B response ∂B^P_pq from the CPHF U^x (occ↔vir rotation):
///   ∂C_·i = Σ_c U_ci C_·c ;  ∂C_·a = −Σ_k U_ak C_·k.
/// For a full-MO index p, ∂(MO p) mixes the complementary space via U. Returns
/// (naux, nmo, nmo). Mirrors the half-index rotation of compute_b_full_mo's B.
fn db_full_from_u(
    b_full: &ndarray::Array3<f64>,
    u: &Array2<f64>,
    orb: &OrbitalSpace,
) -> ndarray::Array3<f64> {
    let OrbitalSpace { nocc, nvir, nocc_total, first_occ } = *orb;
    let naux = b_full.shape()[0];
    let nmo = b_full.shape()[1];
    // U as a full-MO antisymmetric-ish generator Θ_pq: occ i ← vir c with +U_ci,
    // vir a ← occ k with −U_ak. Θ_{c,i}=U_ci (vir,occ), Θ_{i,c}=? The orbital
    // response C^(1)_p = Σ_q Θ_qp C_q with Θ_{vir,occ}=U, Θ_{occ,vir}=−Uᵀ.
    let mut theta = Array2::<f64>::zeros((nmo, nmo)); // Θ_{q,p}: q mixes into p
    for a in 0..nvir {
        for i in 0..nocc {
            theta[(nocc_total + a, first_occ + i)] = u[(a, i)];   // occ i gains vir a
            theta[(first_occ + i, nocc_total + a)] = -u[(a, i)];  // vir a gains −occ i
        }
    }
    // ∂B^P_pq = Σ_r Θ_rp B^P_rq + Σ_r Θ_rq B^P_pr
    let mut db = ndarray::Array3::<f64>::zeros((naux, nmo, nmo));
    for p_aux in 0..naux {
        let bslice = b_full.index_axis(ndarray::Axis(0), p_aux);
        // term1[p,q] = Σ_r Θ_rp B_rq = (Θᵀ B)[p,q]; term2 = (B Θ)[p,q]
        let t1 = theta.t().dot(&bslice);
        let t2 = bslice.dot(&theta);
        let mut dbp = db.index_axis_mut(ndarray::Axis(0), p_aux);
        dbp.assign(&(&t1 + &t2));
    }
    db
}

/// Full analytic relaxed MP2 α (Layer 3, PySCF-mirrored). For each field axis x:
///   ∂dm1_relaxed = ∂P_oo + ∂P_vv (amplitude, from ∂t2) + ∂z (perturbed Z-vector),
///   ∂z solves (Δε+0.5A) ∂z = ∂L − ∂Δε⊙z, with ∂L = directional derivative of
///   build_lagrangian along (∂f_mo, ∂t2, ∂P_oo, ∂P_vv, ∂b_full) [ε central-diff,
///   gauge-stable: inputs are smooth analytic responses, no orbital re-diag].
///   α_xy = −Tr[∂dm1_relaxed_AO · r_y].
#[allow(clippy::too_many_arguments)]
pub fn analytic_alpha_relaxed(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    mp2_config: &RiMp2Config,
) -> Result<Mp2Polarizability, FerricError> {
    let inter = compute_mp2_intermediates(mol, obs, dfbs, op, rhf, mp2_config)?;
    let (nocc, nvir) = (inter.nocc, inter.nvir);
    let (first_occ, nocc_total) = (inter.first_occ, inter.nocc_total);
    let c = rhf.mos_r();
    let eps = rhf.eps_r();
    let nmo = c.ncols();
    let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0]);

    // Un-perturbed pieces (field-independent): base Lagrangian inputs + z.
    let f_mo0 = c.t().dot(rhf.fock_r()).dot(c);
    let b_full = crate::oo_rimp2::compute_b_full_mo(obs, dfbs, op, c)?;
    let (z0, _l0) = crate::zvector::solve_zvector(mol, obs, dfbs, op, bounds, rhf, &inter)?;
    let de = |a: usize, i: usize| eps[nocc_total + a] - eps[first_occ + i];

    let mut tensor = [[0.0f64; 3]; 3];
    for x in 0..3 {
        let (dt2, u, df_mo) =
            analytic_dt2_full(ctx, mol, obs, dfbs, op, bounds, rhf, mp2_config, x)?;
        let (dp_oo, dp_vv) = dmp2_density_response(&inter.t2, &dt2, nocc, nvir);
        let db_full = db_full_from_u(&b_full, &u, &orb);

        // ∂L via directional central difference of build_lagrangian in ε along the
        // analytic input derivatives (smooth → exact at small ε, no gauge issue).
        let eps_step = 1e-4;
        let lag_at = |s: f64| -> Array2<f64> {
            let f = &f_mo0 + &(s * &df_mo);
            let t: Vec<f64> = inter.t2.iter().zip(dt2.iter()).map(|(a, b)| a + s * b).collect();
            let poo = &inter.p_oo + &(s * &dp_oo);
            let pvv = &inter.p_vv + &(s * &dp_vv);
            let bf = &b_full + &(s * &db_full);
            crate::zvector::build_lagrangian(&f, &t, &poo, &pvv, &orb, &bf)
        };
        let dl = (&lag_at(eps_step) - &lag_at(-eps_step)).mapv(|v| v / (2.0 * eps_step));

        // ∂Δε⊙z : ddenom_ai = (deps_v[a] − deps_o[i]); but df_mo gives ∂ε directly.
        let mut rhs = dl.clone();
        for a in 0..nvir {
            for i in 0..nocc {
                let ddenom = df_mo[(nocc_total + a, nocc_total + a)]
                    - df_mo[(first_occ + i, first_occ + i)];
                rhs[(a, i)] -= ddenom * z0[(a, i)];
            }
        }
        // Perturbed Z-vector: (Δε+0.5A) ∂z = rhs.
        let (dz, dresid, _it, dconv) = solve_cphf_cg_scaled(c, &rhs, obs, bounds, &orb, eps, 1.0)?;
        if !dconv {
            return Err(FerricError::General(format!(
                "analytic_alpha_relaxed: ∂z axis {x} not converged (resid={dresid:.2e})"
            )));
        }
        let _ = de;

        // ∂dm1_mo: oo (∂P_oo sym) + vv (∂P_vv sym) + vo/ov (∂z).
        let mut ddm = Array2::<f64>::zeros((nmo, nmo));
        for i in 0..nocc {
            for j in 0..nocc {
                ddm[(first_occ + i, first_occ + j)] = dp_oo[(i, j)] + dp_oo[(j, i)];
            }
        }
        for a in 0..nvir {
            for b in 0..nvir {
                ddm[(nocc_total + a, nocc_total + b)] = dp_vv[(a, b)] + dp_vv[(b, a)];
            }
        }
        // vo/ov block: SCF reference orbital response (2·U, the ∂ of the 2δ core
        // — the occupied orbitals rotate by U under the field; factor 2 =
        // closed-shell occupancy; this is the dominant HF-α contribution) PLUS the
        // MP2 orbital-relaxation ∂z.
        for a in 0..nvir {
            for i in 0..nocc {
                let vo = 2.0 * u[(a, i)] + dz[(a, i)];
                ddm[(nocc_total + a, first_occ + i)] += vo;
                ddm[(first_occ + i, nocc_total + a)] += vo;
            }
        }
        let ddm_ao = c.dot(&ddm).dot(&c.t());
        for y in 0..3 {
            tensor[x][y] = -(&ddm_ao * &dip_ao[y]).sum();
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
    Ok(Mp2Polarizability { tensor, iso, principal })
}
