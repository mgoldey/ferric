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

/// `FERRIC_CPKS_TRACE` descriptor: CPKS residual trace (env-only debug toggle).
static CPKS_TRACE: ferric_core::config::ConfigVar<bool> = ferric_core::config::ConfigVar {
    env_name: "FERRIC_CPKS_TRACE",
    default: false,
    parse: ferric_core::config::parse_toggle,
    validate: ferric_core::config::accept_any,
};
fn cpks_trace() -> bool {
    CPKS_TRACE.toggle()
}

/// Resolve one of the CPKS term-scaling tuning knobs (`CPKS_*`, all default
/// `1.0` = neutral). These are dev-only probes for isolating individual response
/// terms, so any *finite* value is legal — including `0.0` (disable a term) or a
/// negative weight — hence `accept_any`, not `positive_f64`'s `> 0` rule. A
/// malformed value warns and falls back to the default rather than aborting an
/// experimental run. Centralizes the previously copy-pasted
/// `env::var(..).and_then(parse).unwrap_or(default)` idiom (notably `CPKS_WP`,
/// read at four sites) onto the shared descriptor for one parse path.
fn cpks_weight(env_name: &'static str, default: f64) -> f64 {
    let var = ferric_core::config::ConfigVar::<f64> {
        env_name,
        default,
        parse: |s| s.parse::<f64>().map_err(|e| e.to_string()),
        validate: |v| {
            v.is_finite()
                .then_some(())
                .ok_or_else(|| "must be finite".to_string())
        },
    };
    var.get().map(|r| r.value).unwrap_or_else(|e| {
        eprintln!("[config] {env_name}: {e}; using default {default}");
        default
    })
}
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

use crate::ff_polar::{eig3_sym_pub as eig3_sym, Mp2Polarizability};
use crate::rimp2::RiMp2Config;

/// Analytic relaxed MP2 static polarizability (closed-shell). Delegates to the
/// clean-room-validated full-MO driver (`analytic_alpha_full`): matches the
/// energy-Hessian / PySCF oracle to ~0.5% on water/STO-3G. Replaces the
/// finite-field path, which is 1/F-unstable on symmetric molecules.
#[allow(clippy::too_many_arguments)]
pub fn mp2_polarizability_analytic(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    mp2_config: &RiMp2Config,
) -> Result<Mp2Polarizability, FerricError> {
    analytic_alpha_full(ctx, mol, obs, dfbs, op, bounds, rhf, mp2_config)
}

/// Solve (Δε + A) X = rhs by Jacobi-preconditioned CG. Reuses the production
/// orbital-Hessian matvec `compute_az_product`. `rhs` and the return are shape
/// (nvir, nocc). Returns (X, final_residual, iters, converged).
///
/// `budget_bytes` is the caller-resolved memory ceiling threaded into every
/// `compute_az_product` call in the CG loop (up to `max_iter` calls) — see
/// `solve_cphf_cg_scaled`'s doc.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cphf_cg(
    c: &Array2<f64>,
    rhs: &Array2<f64>,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    orb: &OrbitalSpace,
    eps: &[f64],
    budget_bytes: usize,
) -> Result<(Array2<f64>, f64, usize, bool), FerricError> {
    // Default operator (Δε + 0.5 A) — the dipole-CPHF (HF α) convention.
    solve_cphf_cg_scaled(c, rhs, prep, bounds, orb, eps, 0.5, budget_bytes)
}

/// As `solve_cphf_cg` but with an explicit A-coupling scale. Use 0.5 for the
/// dipole-CPHF (HF α) path; use 1.0 to MATCH `solve_zvector`'s full-A operator
/// when solving the PERTURBED Z-vector ∂z (so ∂z and the un-perturbed z0 from
/// solve_zvector share the same operator — required for the relaxed-α response).
///
/// `budget_bytes` is the caller-resolved memory ceiling threaded into every
/// `compute_az_product` call in the CG loop below (up to `max_iter=200`
/// calls) — callers with a `RiMp2Config` (or similar `memory_budget_bytes`
/// field) in scope should pass `resolve_budget_bytes(config.memory_budget_bytes)`,
/// resolved once at their own top; callers with no config in scope pass
/// `resolve_budget_bytes(None)`, likewise resolved once, not per CG iteration.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cphf_cg_scaled(
    c: &Array2<f64>,
    rhs: &Array2<f64>,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    orb: &OrbitalSpace,
    eps: &[f64],
    ascale: f64,
    budget_bytes: usize,
) -> Result<(Array2<f64>, f64, usize, bool), FerricError> {
    use crate::zvector::compute_az_product;
    let OrbitalSpace {
        nocc,
        nvir,
        nocc_total,
        first_occ,
    } = *orb;
    let de = |a: usize, i: usize| eps[nocc_total + a] - eps[first_occ + i];

    // EnginePool is geometry/basis-only (density-independent) — build ONCE
    // here and reuse across every compute_az_product call in the CG loop
    // below (up to max_iter=200 calls), instead of build_jk constructing a
    // fresh pool per call. Reduction order is unchanged, so results stay
    // bit-identical across thread counts.
    let pool = ferric_scf::engine_pool::EnginePool::new(bounds.op, prep, 1e-14)?;

    // A-coupling scale: 0.5 for dipole-CPHF (compute_az_product's symmetric dz
    // double-counts vs the CPHF-α Hessian — pinned vs FF-HF), 1.0 to match
    // solve_zvector's full-A Z-vector operator.
    let apply = |z: &Array2<f64>| -> Result<Array2<f64>, FerricError> {
        let mut mz = compute_az_product(c, z, prep, bounds, orb, &pool, budget_bytes)?;
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
    let trace = cpks_trace();
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
) -> Result<[Array2<f64>; 3], FerricError> {
    let OrbitalSpace {
        nocc,
        nvir,
        nocc_total,
        first_occ,
    } = *orb;
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c
        .slice(ndarray::s![.., nocc_total..nocc_total + nvir])
        .to_owned();
    Ok(std::array::from_fn(|d| c_occ.t().dot(&dip_ao[d]).dot(&c_vir)))
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
    // No RiMp2Config reachable on this pure-HF-α path — hoist ONE resolve
    // here (not per CG iteration / per axis) rather than re-resolving inside
    // solve_cphf_cg's CG loop.
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(None);

    // μ^d_ov as (nvir, nocc) (CG convention).
    let mu_oc = dipole_ov_mo(obs, c, &orb)?; // [d] = (nocc, nvir)
    let mu: [Array2<f64>; 3] = std::array::from_fn(|d| mu_oc[d].t().to_owned()); // (nvir,nocc)

    // U^x: (Δε+A) U^x = −μ^x
    let mut u: Vec<Array2<f64>> = Vec::with_capacity(3);
    for x in 0..3 {
        let rhs = -&mu[x];
        let (ux, resid, iters, conv) = solve_cphf_cg(c, &rhs, obs, bounds, &orb, eps, budget_bytes)?;
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

/// The CPKS response needs the occ-occ and vir-vir dressed B blocks, which the
/// gradient-path builder (`compute_mp2_intermediates_ov_only`) skips. All CPKS
/// entry points call the full `compute_mp2_intermediates`, so this errors only
/// on a programming mistake — but it must be a clean Err, not a panic.
fn require_oo_vv(
    inter: &crate::rimp2::Mp2Intermediates,
) -> Result<(&Array2<f64>, &Array2<f64>), FerricError> {
    match (inter.b_oo.as_ref(), inter.b_vv.as_ref()) {
        (Some(oo), Some(vv)) => Ok((oo, vv)),
        _ => Err(FerricError::General(
            "CPKS needs b_oo/b_vv: intermediates were built ov-only (use compute_mp2_intermediates, not _ov_only)".into(),
        )),
    }
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
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(mp2_config.memory_budget_bytes);

    // --- CPHF U^x: (Δε + 0.5 A) U = −μ^x_ov  (same operator/conventions as HF α) ---
    let mu_oc = dipole_ov_mo(obs, c, &orb)?; // (nocc,nvir)
    let mu_vo = mu_oc[axis].t().to_owned(); // (nvir,nocc)
    let (u, resid, iters, conv) = solve_cphf_cg(c, &(-&mu_vo), obs, bounds, &orb, eps, budget_bytes)?;
    if !conv {
        return Err(FerricError::General(format!(
            "analytic_dt2: CPHF U^{axis} not converged (resid={resid:.2e}, iters={iters})"
        )));
    }

    // --- ∂B^P_ia = Σ_c U_ci b_vv[P,c,a] − Σ_k U_ak b_oo[P,i,k] ---
    let (b_oo, b_vv) = require_oo_vv(&inter)?;
    let mut db_ov = Array2::<f64>::zeros((naux, nov));
    for p in 0..naux {
        for i in 0..nocc {
            for a in 0..nvir {
                let mut s = 0.0;
                for cc in 0..nvir {
                    s += u[(cc, i)] * bget(b_vv, p, cc, a, nvir);
                }
                for k in 0..nocc {
                    s -= u[(a, k)] * bget(b_oo, p, i, k, nocc);
                }
                db_ov[(p, i * nvir + a)] = s;
            }
        }
    }

    // --- ∂ε_p = −μ_pp + Σ_ck U_ck [2(pp|ck) − (pc|pk)] ---
    // Need full-MO dipole diagonal and the coupling integrals via dressed B.
    // μ in MO: μ_pq = Cᵀ D^x_AO C.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
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
    let gscale: f64 = cpks_weight("CPKS_GSCALE", 1.0);
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
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
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
    let (b_oo, b_vv) = require_oo_vv(&inter)?;
    let mut db_ov = Array2::<f64>::zeros((naux, nov));
    for p in 0..naux {
        for i in 0..nocc {
            for a in 0..nvir {
                let mut s = 0.0;
                for cc in 0..nvir {
                    s += u[(cc, i)] * bget(b_vv, p, cc, a, nvir);
                }
                for k in 0..nocc {
                    s -= u[(a, k)] * bget(b_oo, p, i, k, nocc);
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
                    let w_t = cpks_weight("CPKS_W_DT2", 1.0);
                    let w_k = cpks_weight("CPKS_W_DK", 1.0);
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
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
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
    // No RiMp2Config reachable on this diagnostic path — hoist ONE resolve
    // (not per CG iteration) rather than re-resolving inside solve_cphf_cg's
    // CG loop.
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(None);
    // analytic ∂D from U
    let mu_oc = dipole_ov_mo(obs, c, &orb)?;
    let mu_vo = mu_oc[axis].t().to_owned();
    let (u, _r, _it, _cv) = solve_cphf_cg(c, &(-&mu_vo), obs, bounds, &orb, eps, budget_bytes)?;
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
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
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
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    // No RiMp2Config reachable on this diagnostic path — hoist ONE resolve
    // (not per CG iteration) rather than re-resolving inside solve_cphf_cg's
    // CG loop.
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(None);
    let mu_oc = dipole_ov_mo(obs, c, &orb)?;
    let mu_vo = mu_oc[axis].t().to_owned();
    let (u, _r, _it, _cv) = solve_cphf_cg(c, &(-&mu_vo), obs, bounds, &orb, eps, budget_bytes)?;
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
    let dip = oneelectron::dipole(obs, [0.0,0.0,0.0])?;
    let mut v = Array2::<f64>::zeros((n,n));
    for m in 0..n { for k in 0..n { v[(m,k)] = -field*dip[axis][(m,k)]; } }
    let r = crate::ff_polar::solve_rhf_with_external(ctx, mol, obs, bounds, scf_config, &v)?;
    Ok(compute_mp2_intermediates(mol, obs, dfbs, op, &r, mp2_config)?.e_mp2)
}

/// Build (2J−K)[D]_vo in MO from an MO-basis density `dm_mo`, via build_jk on the
/// AO density. Returns the (nvir,nocc) block. (= PySCF get_veff(dm)*2 → vo.)
fn veff_vo_mo(
    ctx: &ParallelContext,
    obs: &PreparedBasis,
    bounds: &SchwarzBounds,
    c: &Array2<f64>,
    dm_mo: &Array2<f64>,
    orb: &OrbitalSpace,
) -> Result<Array2<f64>, FerricError> {
    let OrbitalSpace { nocc, nvir, nocc_total, first_occ } = *orb;
    let nbas = obs.nbasis();
    let dm_ao = c.dot(dm_mo).dot(&c.t());
    let mut j = Array2::<f64>::zeros((nbas, nbas));
    let mut k = Array2::<f64>::zeros((nbas, nbas));
    ferric_scf::rhf::build_jk(ctx, obs, bounds, 1e-12, &dm_ao, &mut j, &mut k)?;
    let g_ao = 2.0 * &j - &k;
    let g_mo = c.t().dot(&g_ao).dot(c);
    let mut out = Array2::<f64>::zeros((nvir, nocc));
    for a in 0..nvir {
        for i in 0..nocc {
            out[(a, i)] = g_mo[(nocc_total + a, first_occ + i)];
        }
    }
    Ok(out)
}

/// VALIDATED static relaxed MP2 1-PDM in AO (matches PySCF to machine precision —
/// see scripts/cpks/mp2_alpha_pyscf.py). The recipe, pinned vs PySCF on water/STO-3G:
///   • P_oo, P_vv from `build_mp2_density`, assembled as P + Pᵀ  (the ×2).
///   • `Xvo = L + (2J−K)[dm_P]_vo`   (L = build_lagrangian; dm_P = the P+Pᵀ density).
///   • (Δε + A) z = −Xvo           (the sign! A = full orbital Hessian).
///   • D = 2δ_core + (P_oo+P_ooᵀ) + (P_vv+P_vvᵀ) + z_vo/ov.
/// NOTE: ferric's shared `build_relaxed_density_ao`/`solve_zvector` are BUGGY for this
/// (single P, no veff, +l sign → water μ_z=−0.708 vs PySCF −0.653); this is the
/// corrected path, kept local to cpks_polar (gradient code untouched).
pub fn static_relaxed_density_ao(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    mp2_config: &RiMp2Config,
) -> Result<Array2<f64>, FerricError> {
    let inter = compute_mp2_intermediates(mol, obs, dfbs, op, rhf, mp2_config)?;
    let (nocc, nvir, first_occ, nocc_total) =
        (inter.nocc, inter.nvir, inter.first_occ, inter.nocc_total);
    let c = rhf.mos_r();
    let eps = rhf.eps_r();
    let nmo = c.ncols();
    let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(mp2_config.memory_budget_bytes);

    // P-blocks (one-sided, = PySCF doo/dvv up to sign): inter.p_oo/p_vv.
    let p_oo = &inter.p_oo;
    let p_vv = &inter.p_vv;

    // dm_P (MO): (P_oo+P_ooᵀ) ⊕ (P_vv+P_vvᵀ) — the ×2 symmetrized MP2 density.
    let mut dm_p = Array2::<f64>::zeros((nmo, nmo));
    for i in 0..nocc {
        for j in 0..nocc {
            dm_p[(first_occ + i, first_occ + j)] = p_oo[(i, j)] + p_oo[(j, i)];
        }
    }
    for a in 0..nvir {
        for b in 0..nvir {
            dm_p[(nocc_total + a, nocc_total + b)] = p_vv[(a, b)] + p_vv[(b, a)];
        }
    }

    // L = ferric Lagrangian (integral part; == PySCF Imat_vo to 4e-17 for canonical).
    let f_mo = c.t().dot(rhf.fock_r()).dot(c);
    let b_full = crate::oo_rimp2::compute_b_full_mo(obs, dfbs, op, c)?;
    let l = crate::zvector::build_lagrangian(&f_mo, &inter.t2, p_oo, p_vv, &orb, &b_full);

    // Xvo = L + (2J−K)[dm_P]_vo.
    let veff = veff_vo_mo(ctx, obs, bounds, c, &dm_p, &orb)?;
    let xvo = &l + &veff;

    // (Δε + A) z = −Xvo  (full-A operator, ascale=1.0).
    let (z, resid, iters, conv) = solve_cphf_cg_scaled(c, &(-&xvo), obs, bounds, &orb, eps, 1.0, budget_bytes)?;
    if !conv {
        return Err(FerricError::General(format!(
            "static_relaxed_density: z not converged (resid={resid:.2e}, iters={iters})"
        )));
    }

    // D (MO): 2δ core + P+Pᵀ + z.
    let mut d_mo = dm_p; // already has P+Pᵀ blocks
    for i in 0..nocc {
        d_mo[(first_occ + i, first_occ + i)] += 2.0;
    }
    for a in 0..nvir {
        for i in 0..nocc {
            d_mo[(nocc_total + a, first_occ + i)] += z[(a, i)];
            d_mo[(first_occ + i, nocc_total + a)] += z[(a, i)];
        }
    }
    Ok(c.dot(&d_mo).dot(&c.t()))
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
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;

    let mut tensor = [[0.0f64; 3]; 3];
    for x in 0..3 {
        let (dt2, _u) = analytic_dt2_along(ctx, mol, obs, dfbs, op, bounds, rhf, mp2_config, x)?;
        let (dp_oo, dp_vv) = dmp2_density_response(&inter.t2, &dt2, nocc, nvir);
        // ∂dm1_mo (amplitude part only): occ-occ ∂P_oo (sym), vir-vir ∂P_vv (sym).
        let mut ddm = Array2::<f64>::zeros((nmo, nmo));
        for i in 0..nocc {
            for j in 0..nocc {
                ddm[(first_occ + i, first_occ + j)] = (dp_oo[(i, j)] + dp_oo[(j, i)]) * cpks_weight("CPKS_WP", 1.0);
            }
        }
        for a in 0..nvir {
            for b in 0..nvir {
                ddm[(nocc_total + a, nocc_total + b)] = (dp_vv[(a, b)] + dp_vv[(b, a)]) * cpks_weight("CPKS_WP", 1.0);
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
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    // Resolved once (not per solve_zvector/solve_cphf_cg_scaled call below) —
    // this function's config-reachable memory ceiling for both CPHF solves.
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(mp2_config.memory_budget_bytes);

    // Un-perturbed pieces (field-independent): base Lagrangian inputs + z.
    let f_mo0 = c.t().dot(rhf.fock_r()).dot(c);
    let b_full = crate::oo_rimp2::compute_b_full_mo(obs, dfbs, op, c)?;
    let (z0, _l0) = crate::zvector::solve_zvector(mol, obs, dfbs, op, bounds, rhf, &inter, budget_bytes)?;
    let de = |a: usize, i: usize| eps[nocc_total + a] - eps[first_occ + i];

    let mut tensor = [[0.0f64; 3]; 3];
    for x in 0..3 {
        let (dt2, u, df_mo) =
            analytic_dt2_full(ctx, mol, obs, dfbs, op, bounds, rhf, mp2_config, x)?;
        let (dp_oo, dp_vv) = dmp2_density_response(&inter.t2, &dt2, nocc, nvir);
        let db_full = db_full_from_u(&b_full, &u, &orb);

        // ∂L via directional central difference of build_lagrangian in ε along the
        // analytic input derivatives (smooth → exact at small ε, no gauge issue).
        let eps_step: f64 = cpks_weight("CPKS_EPS", 1e-4);
        let lag_at = |s: f64| -> Array2<f64> {
            let f = &f_mo0 + &(s * &df_mo);
            let t: Vec<f64> = inter.t2.iter().zip(dt2.iter()).map(|(a, b)| a + s * b).collect();
            let poo = &inter.p_oo + &(s * &dp_oo);
            let pvv = &inter.p_vv + &(s * &dp_vv);
            let bf = &b_full + &(s * &db_full);
            crate::zvector::build_lagrangian(&f, &t, &poo, &pvv, &orb, &bf)
        };
        let dl = (&lag_at(eps_step) - &lag_at(-eps_step)).mapv(|v| v / (2.0 * eps_step));

        // Differentiating the Z-vector equation (Δε+A) z₀ = L:
        //   (Δε+A) ∂z = ∂L − ∂Δε·z₀ − ∂A·z₀
        // RHS = ∂L (above) − ∂Δε·z₀ − ∂A·z₀.
        let mut rhs = dl.clone();
        // − ∂Δε·z₀  (∂ε from df_mo diagonal).
        for a in 0..nvir {
            for i in 0..nocc {
                let ddenom = df_mo[(nocc_total + a, nocc_total + a)]
                    - df_mo[(first_occ + i, first_occ + i)];
                rhs[(a, i)] -= ddenom * z0[(a, i)];
            }
        }
        // − ∂A·z₀ : first-order response of A·z₀ = Cᵀ G(D^{z₀}) C to the MO
        // rotation Θ (Θ_{vir,occ}=U, Θ_{occ,vir}=−Uᵀ). Two contributions:
        //   (i)  rotation of the z₀-density:  Cᵀ G(∂D^{z₀}) C
        //   (ii) rotation of the outer projection: Σ_p Θ_pa (Az₀)_pi + Θ_pi (Az₀)_ap
        // where Az₀_full = Cᵀ G(D^{z₀}) C in FULL MO. (Az₀ in vo only = A·z₀.)
        let waz: f64 = cpks_weight("CPKS_WAZ", 1.0);
        if waz != 0.0 {
            // Full-MO Θ generator.
            let mut theta = Array2::<f64>::zeros((nmo, nmo)); // Θ_{q,p}
            for a in 0..nvir {
                for i in 0..nocc {
                    theta[(nocc_total + a, first_occ + i)] = u[(a, i)];
                    theta[(first_occ + i, nocc_total + a)] = -u[(a, i)];
                }
            }
            // D^{z₀}_AO = Σ_ai z0_ai (C_a C_iᵀ + C_i C_aᵀ).
            let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
            let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();
            let nbas = obs.nbasis();
            // (i) ∂D^{z₀}_AO from rotating C: ∂C_·p = Σ_q Θ_qp C_·q.
            // ∂C_occ_i = Σ_a U_ai C_vir_a ; ∂C_vir_a = −Σ_i U_ai C_occ_i.
            let dc_occ = c_vir.dot(&u); // (nbas,nocc): Σ_a C_vir_a U_ai
            let dc_vir = c_occ.dot(&u.t()).mapv(|v| -v); // (nbas,nvir): −Σ_i C_occ_i U_ai
            let mut dd_z = Array2::<f64>::zeros((nbas, nbas));
            for a in 0..nvir {
                for i in 0..nocc {
                    let zai = z0[(a, i)];
                    if zai == 0.0 { continue; }
                    for mu in 0..nbas {
                        for nu in 0..nbas {
                            // ∂ of (C_a C_iᵀ + C_i C_aᵀ)
                            dd_z[(mu, nu)] += zai
                                * (dc_vir[(mu, a)] * c_occ[(nu, i)]
                                    + c_vir[(mu, a)] * dc_occ[(nu, i)]
                                    + dc_occ[(mu, i)] * c_vir[(nu, a)]
                                    + c_occ[(mu, i)] * dc_vir[(nu, a)]);
                        }
                    }
                }
            }
            let mut jz = Array2::<f64>::zeros((nbas, nbas));
            let mut kz = Array2::<f64>::zeros((nbas, nbas));
            ferric_scf::rhf::build_jk(ctx, obs, bounds, 1e-12, &dd_z, &mut jz, &mut kz)?;
            let g_ddz = 4.0 * &jz - &kz - &kz.t();
            let part_i = c.t().dot(&g_ddz).dot(c); // full MO; take vo block below
            // (ii) Az₀ in full MO = Cᵀ G(D^{z₀}) C.
            let mut dz0_ao = Array2::<f64>::zeros((nbas, nbas));
            for a in 0..nvir {
                for i in 0..nocc {
                    let zai = z0[(a, i)];
                    if zai == 0.0 { continue; }
                    for mu in 0..nbas {
                        for nu in 0..nbas {
                            dz0_ao[(mu, nu)] += zai
                                * (c_vir[(mu, a)] * c_occ[(nu, i)] + c_occ[(mu, i)] * c_vir[(nu, a)]);
                        }
                    }
                }
            }
            let mut jz0 = Array2::<f64>::zeros((nbas, nbas));
            let mut kz0 = Array2::<f64>::zeros((nbas, nbas));
            ferric_scf::rhf::build_jk(ctx, obs, bounds, 1e-12, &dz0_ao, &mut jz0, &mut kz0)?;
            let g_z0 = 4.0 * &jz0 - &kz0 - &kz0.t();
            let az0_full = c.t().dot(&g_z0).dot(c); // (nmo,nmo)
            // ∂(Az₀)_ai = part_i[a,i] + Σ_p Θ_pa az0_full[p,i] + Σ_p Θ_pi az0_full[a,p]
            let theta_az = theta.t().dot(&az0_full); // Σ_p Θ_pa az0[p,i] → [a-row? careful]
            let az_theta = az0_full.dot(&theta); // Σ_p az0[a,p] Θ_pi
            for a in 0..nvir {
                let a_mo = nocc_total + a;
                for i in 0..nocc {
                    let i_mo = first_occ + i;
                    let daz0 = part_i[(a_mo, i_mo)] + theta_az[(a_mo, i_mo)] + az_theta[(a_mo, i_mo)];
                    rhs[(a, i)] -= waz * daz0;
                }
            }
        }
        // Perturbed Z-vector: (Δε+A) ∂z = rhs  (full A to match z₀'s operator).
        let (dz, dresid, _it, dconv) = solve_cphf_cg_scaled(c, &rhs, obs, bounds, &orb, eps, 1.0, budget_bytes)?;
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
                ddm[(first_occ + i, first_occ + j)] = (dp_oo[(i, j)] + dp_oo[(j, i)]) * cpks_weight("CPKS_WP", 1.0);
            }
        }
        for a in 0..nvir {
            for b in 0..nvir {
                ddm[(nocc_total + a, nocc_total + b)] = (dp_vv[(a, b)] + dp_vv[(b, a)]) * cpks_weight("CPKS_WP", 1.0);
            }
        }
        // vo/ov block: SCF reference orbital response (2·U, the ∂ of the 2δ core
        // — the occupied orbitals rotate by U under the field; factor 2 =
        // closed-shell occupancy; this is the dominant HF-α contribution) PLUS the
        // MP2 orbital-relaxation ∂z.
        for a in 0..nvir {
            for i in 0..nocc {
                let w2u: f64 = cpks_weight("CPKS_W2U", 1.0);
                let wz: f64 = cpks_weight("CPKS_WZ", 1.0); let vo = w2u * 2.0 * u[(a, i)] + wz * dz[(a, i)];
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

// ===========================================================================
// PORTED analytic relaxed-MP2 α (clean-room-validated, scripts/cpks/mp2_alpha_clean.py).
// Operates on the FULL-MO ERI tensor (pq|rs)=Σ_P B[P,p,q]B[P,r,s] so the field
// perturbation (Θ-rotation of integrals) is direct. Non-RI in the response
// (small systems). Recipe per axis q:
//   U^q : (Δε+A) U = −r^q_vo                          (CPHF)
//   Θ_{vir,occ}=U, Θ_{occ,vir}=−U
//   ∂Imo = Σ_idx Θ-rotate each of the 4 MO indices
//   ∂D   = central-diff (ε=1e-5) of relaxed_dm_full(Imo±ε∂Imo, eps)   [deps=0]
//   ∂D  += 2U in vo/ov (SCF core response)
//   α_pq = −Σ ∂D · r_mo[p]
// ===========================================================================
use ndarray::Array4;

/// Fail-fast pre-flight guard for every `full_mo_eri` caller. The dense (pq|rs)
/// tensor is nmo⁴, built co-resident with its (nmo²)²=nmo⁴ Gram matrix (:1146),
/// and the analytic-α path additionally holds central-diff ∂Imo copies —
/// budget for ~3 live nmo⁴ f64 buffers. Placed next to `full_mo_eri` so an M3/M5
/// restructure that shrinks the peak updates the formula in the same diff.
fn check_full_mo_eri_alloc(
    label: &str,
    nmo: usize,
    explicit_budget: Option<usize>,
) -> Result<(), FerricError> {
    let peak = nmo.saturating_pow(4).saturating_mul(3).saturating_mul(8); // ~3× nmo⁴ f64
    ferric_core::memory::check_alloc(
        &format!("{label}: dense nmo⁴ MO-ERI (nmo={nmo})"),
        peak,
        ferric_core::memory::resolve_budget_bytes(explicit_budget),
    )
}

/// Full-MO ERI (pq|rs) from the dressed B tensor. (nmo^4 — small systems only.)
fn full_mo_eri(b_full: &ndarray::Array3<f64>) -> Array4<f64> {
    let naux = b_full.shape()[0];
    let nmo = b_full.shape()[1];
    let mut imo = Array4::<f64>::zeros((nmo, nmo, nmo, nmo));
    // (pq|rs) = Σ_P B[P,p,q] B[P,r,s]
    let bmat = b_full.view().into_shape_with_order((naux, nmo * nmo)).unwrap();
    let g = bmat.t().dot(&bmat); // (nmo*nmo, nmo*nmo)
    for p in 0..nmo { for q in 0..nmo { for rr in 0..nmo { for s in 0..nmo {
        imo[(p, q, rr, s)] = g[(p * nmo + q, rr * nmo + s)];
    }}}}
    imo
}

#[inline]
fn eri4(imo: &Array4<f64>, p: usize, q: usize, r: usize, s: usize) -> f64 { imo[(p, q, r, s)] }

/// t2[i,a,j,b] = (ia|jb)/Δε  from a full-MO ERI tensor.
fn t2_full(imo: &Array4<f64>, eps: &[f64], nocc: usize, nvir: usize) -> Vec<f64> {
    let nov = nocc * nvir;
    let mut t = vec![0.0; nov * nov];
    for i in 0..nocc { for a in 0..nvir { for j in 0..nocc { for b in 0..nvir {
        let d = eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b];
        t[(i * nvir + a) * nov + j * nvir + b] = eri4(imo, i, nocc + a, j, nocc + b) / d;
    }}}}
    t
}

/// Static relaxed MP2 1-PDM (MO) from a full-MO ERI tensor — mirrors the
/// clean-room `relaxed_dm` exactly (P+Pᵀ, Xvo=L+(2J−K)[dmP]_vo, (Δε+A)z=−Xvo).
/// `eps` indexed by MO (0..nmo); occ=0..nocc, vir=nocc..nmo.
fn relaxed_dm_full(imo: &Array4<f64>, eps: &[f64], nocc: usize, nvir: usize) -> Array2<f64> {
    let nmo = nocc + nvir;
    let nov = nocc * nvir;
    let t = t2_full(imo, eps, nocc, nvir);
    let tt = |i: usize, a: usize, j: usize, b: usize| t[(i * nvir + a) * nov + j * nvir + b];

    // P_oo, P_vv (one-sided).
    let mut p_oo = Array2::<f64>::zeros((nocc, nocc));
    for i in 0..nocc { for j in 0..nocc {
        let mut s = 0.0;
        for k in 0..nocc { for a in 0..nvir { for b in 0..nvir {
            s += tt(i, a, k, b) * (2.0 * tt(j, a, k, b) - tt(j, b, k, a));
        }}}
        p_oo[(i, j)] = -s;
    }}
    let mut p_vv = Array2::<f64>::zeros((nvir, nvir));
    for a in 0..nvir { for b in 0..nvir {
        let mut s = 0.0;
        for i in 0..nocc { for j in 0..nocc { for cc in 0..nvir {
            s += tt(i, a, j, cc) * (2.0 * tt(i, b, j, cc) - tt(i, cc, j, b));
        }}}
        p_vv[(a, b)] = s;
    }}

    // dm_P = P+Pᵀ in oo/vv (MO).
    let mut dm_p = Array2::<f64>::zeros((nmo, nmo));
    for i in 0..nocc { for j in 0..nocc { dm_p[(i, j)] = p_oo[(i, j)] + p_oo[(j, i)]; } }
    for a in 0..nvir { for b in 0..nvir { dm_p[(nocc + a, nocc + b)] = p_vv[(a, b)] + p_vv[(b, a)]; } }

    // L (4-term integral Lagrangian, full-MO indices). Mirrors build_lagrangian integral part.
    let mut l = Array2::<f64>::zeros((nvir, nocc));
    for c in 0..nvir { for k in 0..nocc {
        let mut g = 0.0;
        for j in 0..nocc { for a in 0..nvir { for b in 0..nvir {
            g += tt(k, a, j, b) * (2.0 * eri4(imo, nocc + c, nocc + a, j, nocc + b) - eri4(imo, nocc + c, nocc + b, j, nocc + a));
        }}}
        for i in 0..nocc { for a in 0..nvir { for b in 0..nvir {
            g += tt(i, a, k, b) * (2.0 * eri4(imo, i, nocc + a, nocc + c, nocc + b) - eri4(imo, i, nocc + b, nocc + c, nocc + a));
        }}}
        for i in 0..nocc { for j in 0..nocc { for b in 0..nvir {
            g -= tt(i, c, j, b) * (2.0 * eri4(imo, i, k, j, nocc + b) - eri4(imo, i, nocc + b, j, k));
        }}}
        for i in 0..nocc { for j in 0..nocc { for a in 0..nvir {
            g -= tt(i, a, j, c) * (2.0 * eri4(imo, i, nocc + a, j, k) - eri4(imo, i, k, j, nocc + a));
        }}}
        l[(c, k)] = g;
    }}

    // (2J−K)[dm_P]_vo in MO: G_pq = Σ_rs [2(pq|rs) − (pr|qs)] dm_P[r,s].
    let mut xvo = l.clone();
    for c in 0..nvir { for k in 0..nocc {
        let mut g = 0.0;
        for r in 0..nmo { for s in 0..nmo {
            g += (2.0 * eri4(imo, nocc + c, k, r, s) - eri4(imo, nocc + c, r, k, s)) * dm_p[(r, s)];
        }}
        xvo[(c, k)] += g;
    }}

    // (Δε + A) z = −Xvo, with full A (dense solve mirroring clean-room).
    let z = solve_zvec_dense(imo, eps, &(-&xvo), nocc, nvir);

    // D = 2δ_core + dm_P + z.
    let mut d = dm_p;
    for i in 0..nocc { d[(i, i)] += 2.0; }
    for a in 0..nvir { for i in 0..nocc { d[(nocc + a, i)] += z[(a, i)]; d[(i, nocc + a)] += z[(a, i)]; } }
    d
}

/// Dense solve of (Δε + A) x = rhs with the full orbital Hessian
/// A_{ai,bj} = 4(ai|bj) − (ab|ij) − (aj|bi), from a full-MO ERI tensor.
fn solve_zvec_dense(imo: &Array4<f64>, eps: &[f64], rhs: &Array2<f64>, nocc: usize, nvir: usize) -> Array2<f64> {
    use ndarray_linalg::Solve;
    let n = nvir * nocc;
    let mut m = Array2::<f64>::zeros((n, n));
    for a in 0..nvir { for i in 0..nocc {
        let ai = a * nocc + i;
        for b in 0..nvir { for j in 0..nocc {
            let bj = b * nocc + j;
            m[(ai, bj)] = 4.0 * eri4(imo, nocc + a, i, nocc + b, j)
                - eri4(imo, nocc + a, nocc + b, i, j)
                - eri4(imo, nocc + a, j, nocc + b, i);
        }}
        m[(ai, ai)] += eps[nocc + a] - eps[i];
    }}
    let x = m.solve(&rhs.view().into_shape_with_order(n).unwrap().to_owned()).unwrap();
    x.into_shape_with_order((nvir, nocc)).unwrap()
}

/// Analytic relaxed MP2 static polarizability via the full-MO recipe (clean-room
/// validated to ~0.5% vs energy-Hessian). Closed-shell. The proper public entry
/// `mp2_polarizability_analytic` delegates here.
pub fn analytic_alpha_full(
    _ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    _bounds: &SchwarzBounds,
    rhf: &ScfResult,
    mp2_config: &RiMp2Config,
) -> Result<Mp2Polarizability, FerricError> {
    // `_bounds` unused: the full-MO recipe uses the dense ERI tensor (no screened
    // JK). Kept in the signature for API parity with the finite-field path.
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General("analytic_alpha_full: Restricted only".into()));
    }
    let c = rhf.mos_r();
    let nmo = c.ncols();
    let nocc = (mol.nelec() / 2) as usize;
    let nvir = nmo - nocc;
    let eps_full: Vec<f64> = rhf.eps_r().to_vec();
    let orb = OrbitalSpace::new(nocc, nvir, nocc, 0);

    // Fail-fast size guard: the full-MO recipe holds the dense (pq|rs) ERI tensor
    // imo0 (nmo⁴, full_mo_eri :1143) co-resident with its Gram matrix g
    // ((nmo²)²=nmo⁴, :1146) and the central-diff ∂Imo copies in the axis loop.
    // The solve_zvec_dense (Δε+A) Hessian ((nocc·nvir)², :1243) is subsumed by
    // this larger nmo⁴ peak.
    check_full_mo_eri_alloc(
        &format!("relaxed-MP2 α (analytic; nmo={nmo}, nocc={nocc}, nvir={nvir})"),
        nmo,
        mp2_config.memory_budget_bytes,
    )?;

    // Full-MO ERI tensor (unperturbed).
    let b_full = crate::oo_rimp2::compute_b_full_mo(obs, dfbs, op, c)?;
    let imo0 = full_mo_eri(&b_full);

    // MO dipole r_pq per axis (full MO).
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let r_mo: [Array2<f64>; 3] = std::array::from_fn(|d| c.t().dot(&dip_ao[d]).dot(c));

    let ed = 1e-5;
    let mut tensor = [[0.0f64; 3]; 3];
    for q in 0..3 {
        // U^q: (Δε+A) U = −r^q_vo (CPHF, dense, same operator as the z-solve).
        let mut rvo = Array2::<f64>::zeros((nvir, nocc));
        for a in 0..nvir { for i in 0..nocc { rvo[(a, i)] = r_mo[q][(nocc + a, i)]; } }
        let u = solve_zvec_dense(&imo0, &eps_full, &(-&rvo), nocc, nvir);

        // Θ generator: Θ_{vir,occ}=U, Θ_{occ,vir}=−U.
        let mut theta = Array2::<f64>::zeros((nmo, nmo));
        for a in 0..nvir { for i in 0..nocc {
            theta[(nocc + a, i)] = u[(a, i)];
            theta[(i, nocc + a)] = -u[(a, i)];
        }}

        // ∂Imo = Σ_idx Θ-rotate each of the 4 MO indices.
        // For ±ε we build imo(±) = imo0 ± ε·∂Imo directly via rotated contraction.
        let dimo = rotate_eri(&imo0, &theta);
        let imo_p = &imo0 + &(ed * &dimo);
        let imo_m = &imo0 - &(ed * &dimo);

        // ∂D = central diff of relaxed_dm_full (deps=0: eps unchanged).
        let dp = relaxed_dm_full(&imo_p, &eps_full, nocc, nvir);
        let dm = relaxed_dm_full(&imo_m, &eps_full, nocc, nvir);
        let mut ddm = (&dp - &dm).mapv(|x| x / (2.0 * ed));

        // + 2U SCF core response in vo/ov.
        for a in 0..nvir { for i in 0..nocc {
            ddm[(nocc + a, i)] += 2.0 * u[(a, i)];
            ddm[(i, nocc + a)] += 2.0 * u[(a, i)];
        }}

        // α_pq = −Σ ∂D · r_mo[p]  (MO basis).
        for p in 0..3 {
            tensor[p][q] = -(&ddm * &r_mo[p]).sum();
        }
    }
    // Symmetrize.
    for i in 0..3 { for j in (i + 1)..3 {
        let avg = 0.5 * (tensor[i][j] + tensor[j][i]); tensor[i][j] = avg; tensor[j][i] = avg;
    }}
    let _ = orb;
    let iso = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;
    let principal = eig3_sym(tensor);
    Ok(Mp2Polarizability { tensor, iso, principal })
}

/// ∂Imo via Θ-rotation of all 4 MO indices: ∂(pq|rs) = Σ_x [Θ_xp(xq|rs)+Θ_xq(px|rs)
/// +Θ_xr(pq|xs)+Θ_xs(pq|rx)]. (Matches clean-room dImo.)
fn rotate_eri(imo: &Array4<f64>, theta: &Array2<f64>) -> Array4<f64> {
    let nmo = imo.shape()[0];
    let mut d = Array4::<f64>::zeros((nmo, nmo, nmo, nmo));
    for p in 0..nmo { for q in 0..nmo { for r in 0..nmo { for s in 0..nmo {
        let mut acc = 0.0;
        for x in 0..nmo {
            acc += theta[(x, p)] * imo[(x, q, r, s)]
                 + theta[(x, q)] * imo[(p, x, r, s)]
                 + theta[(x, r)] * imo[(p, q, x, s)]
                 + theta[(x, s)] * imo[(p, q, r, x)];
        }
        d[(p, q, r, s)] = acc;
    }}}}
    d
}

// ===========================================================================
// Dynamic CPHF (TDHF) polarizability α(iω) and Casimir-Polder C6.
//
// The static-α sweep showed attenuation shrinks α uniformly (hurts vs CRC).
// C6 = (3/π)∫₀^∞ α(iω)² dω weights the imaginary-frequency response, which
// could behave differently — this is the one lane where attenuation might
// still help dispersion. This is HF-level (CPHF/TDHF) dynamic α: it captures
// the orbital response and operator dependence; the MP2 correlation
// correction to α(iω) is a separate (larger) build deferred until the C6
// trend justifies it.
//
// Singlet closed-shell linear-response matrices from the full-MO ERIs:
//   A_{ai,bj} = (ε_a−ε_i)δ_{ai,bj} + 2(ai|bj) − (ab|ij)
//   B_{ai,bj} =                       2(ai|bj) − (aj|bi)
// (A+B) = Δε + 4(ai|bj) − (ab|ij) − (aj|bi)  ≡ the validated static Hessian
//         `solve_zvec_dense` builds. So the static driver's M is (A+B), and
//         the static α solves (A+B)X=−μ.  ⟹ α_static = 4 μᵀ(A+B)⁻¹μ.
// Imaginary-frequency response (Casida, real symmetric reduced form):
//   α(iω) = 4 μᵀ (A−B) [ (A−B)(A+B) + ω² ]⁻¹ μ
//   ω=0:  α = 4 μᵀ (A−B)[(A−B)(A+B)]⁻¹ μ = 4 μᵀ(A+B)⁻¹μ  ✓ static limit.
// Validated: cpks_dynamic_alpha_w0_matches_static (ω=0 == static HF α).
// ===========================================================================

/// Build singlet (A+B) and (A−B) in (ai)-space (n=nvir·nocc) from full-MO ERIs.
fn build_apb_amb(
    imo: &Array4<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
) -> (Array2<f64>, Array2<f64>) {
    let n = nvir * nocc;
    let mut apb = Array2::<f64>::zeros((n, n));
    let mut amb = Array2::<f64>::zeros((n, n));
    for a in 0..nvir {
        for i in 0..nocc {
            let ai = a * nocc + i;
            for b in 0..nvir {
                for j in 0..nocc {
                    let bj = b * nocc + j;
                    let coul = eri4(imo, nocc + a, i, nocc + b, j); // (ai|bj)
                    let exch_abij = eri4(imo, nocc + a, nocc + b, i, j); // (ab|ij)
                    let exch_ajbi = eri4(imo, nocc + a, j, nocc + b, i); // (aj|bi)
                    // A = Δε δ + 2(ai|bj) − (ab|ij);  B = 2(ai|bj) − (aj|bi)
                    let a_el = 2.0 * coul - exch_abij;
                    let b_el = 2.0 * coul - exch_ajbi;
                    apb[(ai, bj)] = a_el + b_el; // 4(ai|bj) − (ab|ij) − (aj|bi)
                    amb[(ai, bj)] = a_el - b_el; // (aj|bi) − (ab|ij)
                }
            }
            apb[(ai, ai)] += eps[nocc + a] - eps[i];
            amb[(ai, ai)] += eps[nocc + a] - eps[i];
        }
    }
    (apb, amb)
}

/// W-screened singlet (A±B) for BSE-flavoured correlation/response.
///
/// Generalises `build_apb_amb`: the **Hartree/coupling** term `4(ai|bj)` keeps
/// the BARE Coulomb interaction, while the two **exchange** integrals are replaced
/// by their statically SCREENED counterparts `(··|W|··)` built from PDEP modes:
///
/// ```text
///   (pq|W|rs) = Σ_α  w_α · g^α_{pq} · g^α_{rs}      g^α_{pq} = Σ_P (Ṽ_α)_P b^P_{pq}
///
///   (A+B)_W = Δε_qp δ + 4(ai|bj)_v − (ab|ij)_W − (aj|bi)_W
///   (A−B)_W = Δε_qp δ            + (aj|bi)_W − (ab|ij)_W
/// ```
///
/// matching the convention in `build_apb_amb` term-for-term (only the two
/// exchange integrals carry the W; the 4(ai|bj) Hartree term stays bare).
///
/// # Arguments
/// * `imo` — full-MO bare ERIs (pq|rs) for the Hartree term, as in `build_apb_amb`.
/// * `g_modes` — mode-projected MO tensor g[α, p, q] = Σ_P (Ṽ_α)_P b^P_{pq},
///   shape (M, nmo, nmo). For the BARE-v limit pass the raw RI tensor b^P_{pq}
///   (M = naux) so that `(pq|W|rs)` collapses to `Σ_P b^P_{pq} b^P_{rs} = (pq|rs)`.
/// * `weights` — per-mode screening weight w_α (length M). For the bare-v limit
///   pass all-ones; for true BSE pass `w_α = 1/λ_α − 1` from the PDEP spectrum.
/// * `eps_qp` — quasiparticle (or HF/KS) orbital energies for the Δε diagonal.
///
/// GATE-0 invariant (no physics): with `g_modes = b_full` and `weights = 1`,
/// this reproduces `build_apb_amb` (with the same `eps`) bit-for-bit. That
/// pins every sign/factor of the screened-exchange contraction before any W or
/// GW energy is introduced. See `bse_gate0_bare_v_collapses_to_tdhf`.
pub fn build_apb_amb_screened(
    imo: &Array4<f64>,
    g_modes: &ndarray::Array3<f64>,
    weights: &[f64],
    eps_qp: &[f64],
    nocc: usize,
    nvir: usize,
) -> (Array2<f64>, Array2<f64>) {
    let nmodes = g_modes.shape()[0];
    assert_eq!(weights.len(), nmodes, "weights length must equal #modes");
    let n = nvir * nocc;

    // Screened integral (pq|W|rs) = Σ_α w_α g^α_{pq} g^α_{rs}, evaluated on the
    // specific orbital-pair blocks the kernel needs: (ab|ij)_W and (aj|bi)_W.
    // We build them as explicit 4-index sub-tensors over the (a,b,i,j) ranges.
    let sw = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = 0.0;
        for alpha in 0..nmodes {
            acc += weights[alpha] * g_modes[(alpha, p, q)] * g_modes[(alpha, r, s)];
        }
        acc
    };

    let mut apb = Array2::<f64>::zeros((n, n));
    let mut amb = Array2::<f64>::zeros((n, n));
    for a in 0..nvir {
        for i in 0..nocc {
            let ai = a * nocc + i;
            for b in 0..nvir {
                for j in 0..nocc {
                    let bj = b * nocc + j;
                    let coul = eri4(imo, nocc + a, i, nocc + b, j); // bare (ai|bj)
                    let exch_abij = sw(nocc + a, nocc + b, i, j); // (ab|W|ij)
                    let exch_ajbi = sw(nocc + a, j, nocc + b, i); // (aj|W|bi)
                    // A = Δε δ + 2(ai|bj)_v − (ab|ij)_W;  B = 2(ai|bj)_v − (aj|bi)_W
                    let a_el = 2.0 * coul - exch_abij;
                    let b_el = 2.0 * coul - exch_ajbi;
                    apb[(ai, bj)] = a_el + b_el; // 4(ai|bj)_v − (ab|ij)_W − (aj|bi)_W
                    amb[(ai, bj)] = a_el - b_el; // (aj|bi)_W − (ab|ij)_W
                }
            }
            apb[(ai, ai)] += eps_qp[nocc + a] - eps_qp[i];
            amb[(ai, ai)] += eps_qp[nocc + a] - eps_qp[i];
        }
    }
    (apb, amb)
}

/// GATE-0 driver (test-only): build BOTH the bare TDHF (A±B) and the screened
/// (A±B) fed with the bare-v limit (raw RI modes, unit weights), and return the
/// max element-wise differences `(‖ΔAPB‖∞, ‖ΔAMB‖∞)`. The screened builder is
/// correct iff both are ~0 (machine precision). No GW, no external reference —
/// purely pins the contraction conventions of `build_apb_amb_screened`.
pub fn bse_gate0_residuals(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
) -> Result<(f64, f64), FerricError> {
    let _ = ctx;
    let c = rhf.mos_r();
    let nmo = c.ncols();
    let nocc = (mol.nelec() / 2) as usize;
    let nvir = nmo - nocc;
    let eps: Vec<f64> = rhf.eps_r().to_vec();

    // Fail-fast size guard: full_mo_eri holds the nmo⁴ ERI + nmo⁴ Gram (:1143-1146).
    check_full_mo_eri_alloc("BSE gate-0 residuals", nmo, None)?;
    let b_full = crate::oo_rimp2::compute_b_full_mo(obs, dfbs, op, c)?; // (naux, nmo, nmo)
    let imo = full_mo_eri(&b_full);

    // Reference: existing validated TDHF blocks.
    let (apb_ref, amb_ref) = build_apb_amb(&imo, &eps, nocc, nvir);

    // Bare-v limit of the screened builder: modes = raw RI tensor, weights = 1.
    let naux = b_full.shape()[0];
    let weights = vec![1.0_f64; naux];
    let (apb_w, amb_w) =
        build_apb_amb_screened(&imo, &b_full, &weights, &eps, nocc, nvir);

    let dmax = |x: &Array2<f64>, y: &Array2<f64>| -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    };
    Ok((dmax(&apb_ref, &apb_w), dmax(&amb_ref, &amb_w)))
}

/// Dynamic CPHF/TDHF polarizability tensor at imaginary frequency iω (a.u.).
/// ω=0 reproduces the static CPHF α. HF-level (no MP2 correlation in α(iω)).
pub fn dynamic_cphf_alpha_iw(
    _ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    omega: f64,
) -> Result<[[f64; 3]; 3], FerricError> {
    use ndarray_linalg::Solve;
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General("dynamic_cphf_alpha_iw: Restricted only".into()));
    }
    let c = rhf.mos_r();
    let nmo = c.ncols();
    let nocc = (mol.nelec() / 2) as usize;
    let nvir = nmo - nocc;
    let n = nvir * nocc;
    let eps: Vec<f64> = rhf.eps_r().to_vec();

    // Fail-fast size guard: full_mo_eri holds the nmo⁴ ERI + nmo⁴ Gram (:1143-1146).
    check_full_mo_eri_alloc("dynamic CPHF/TDHF α(iω)", nmo, None)?;
    let b_full = crate::oo_rimp2::compute_b_full_mo(obs, dfbs, op, c)?;
    let imo = full_mo_eri(&b_full);
    let (apb, amb) = build_apb_amb(&imo, &eps, nocc, nvir);

    // System matrix S = (A−B)(A+B) + ω²I  (n×n, same for all axes).
    let mut sysm = amb.dot(&apb);
    for k in 0..n {
        sysm[(k, k)] += omega * omega;
    }

    // dipole μ in (ai)-space per axis (MO), bare Coulomb operator.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let r_mo: [Array2<f64>; 3] = std::array::from_fn(|d| c.t().dot(&dip_ao[d]).dot(c));
    let mut mu: [ndarray::Array1<f64>; 3] = std::array::from_fn(|_| ndarray::Array1::zeros(n));
    for (d, m) in mu.iter_mut().enumerate() {
        for a in 0..nvir {
            for i in 0..nocc {
                m[a * nocc + i] = r_mo[d][(nocc + a, i)];
            }
        }
    }

    // For each axis y: solve S·t_y = (A−B) μ_y, then α_xy = 4 μ_xᵀ t_y.
    let mut tensor = [[0.0f64; 3]; 3];
    let mut t: [ndarray::Array1<f64>; 3] = std::array::from_fn(|_| ndarray::Array1::zeros(n));
    for (y, ty) in t.iter_mut().enumerate() {
        let rhs = amb.dot(&mu[y]);
        *ty = sysm.solve(&rhs).map_err(|e| {
            FerricError::General(format!("dynamic_cphf_alpha_iw: solve failed: {e}"))
        })?;
    }
    for x in 0..3 {
        for y in 0..3 {
            tensor[x][y] = 4.0 * mu[x].dot(&t[y]);
        }
    }
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg = 0.5 * (tensor[i][j] + tensor[j][i]);
            tensor[i][j] = avg;
            tensor[j][i] = avg;
        }
    }
    Ok(tensor)
}

/// Molecular isotropic C6 from dynamic CPHF α(iω) via Casimir-Polder:
///   C6 = (3/π) Σ_k w_k α_iso(iω_k)²   on a Gauss-Legendre [0,∞) grid.
/// Returns (c6, α_iso at each freq) for the given operator.
pub fn cphf_c6_molecular(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    freqs: &[f64],
    weights: &[f64],
) -> Result<(f64, Vec<f64>), FerricError> {
    use std::f64::consts::PI;
    let mut iso_prof = Vec::with_capacity(freqs.len());
    for &w in freqs {
        let t = dynamic_cphf_alpha_iw(ctx, mol, obs, dfbs, op, rhf, w)?;
        iso_prof.push((t[0][0] + t[1][1] + t[2][2]) / 3.0);
    }
    let mut s = 0.0;
    for k in 0..freqs.len() {
        s += weights[k] * iso_prof[k] * iso_prof[k];
    }
    Ok((3.0 / PI * s, iso_prof))
}

// ===========================================================================
// Frozen-amplitude MP2 dynamic α(iω) — the CHEAP MP2 spike.
//
// Full MP2 dynamic response is a research-grade derivation (~10 terms,
// frequency-dependent ∂t2). The frozen-amplitude approximation: take the
// dynamic CPHF frequency *shape* α_HF(iω) and rescale it to the validated
// static *MP2* magnitude — i.e. the MP2 correlation enters α's size (which
// the static sweep showed attenuation shrinks) but the frequency dependence
// is inherited from HF. Standard "scaled-α" dispersion trick.
//
//   shape:  s_HF(iω) = α_HF_iso(iω) / α_HF_iso(0)        (1 at ω=0)
//   α_MP2_iso(iω) = α_MP2_iso(static) · s_HF(iω)
//
// Reduces to the validated static MP2 α at ω=0 BY CONSTRUCTION. Tests whether
// MP2's magnitude correction flips the attenuation→C6 verdict vs HF-level.
// It does NOT capture an MP2-specific change to the frequency *shape* — that
// is exactly what the full (expensive) response would add, and the gap
// between this and dRPA tells us whether that's worth deriving.
// ===========================================================================

/// Frozen-amplitude MP2 dynamic molecular C6: HF-shape α(iω) rescaled to the
/// static MP2 isotropic magnitude. Returns (c6, α_MP2_iso profile, static MP2
/// iso, static HF iso). Closed-shell.
#[allow(clippy::too_many_arguments)]
pub fn frozen_mp2_c6_molecular(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    mp2_config: &RiMp2Config,
    freqs: &[f64],
    weights: &[f64],
) -> Result<(f64, Vec<f64>, f64, f64), FerricError> {
    use std::f64::consts::PI;
    // Static MP2 α (validated full-MO recipe) and static HF α (ω=0 dynamic).
    let mp2_stat = analytic_alpha_full(ctx, mol, obs, dfbs, op, bounds, rhf, mp2_config)?;
    let hf0 = dynamic_cphf_alpha_iw(ctx, mol, obs, dfbs, op, rhf, 0.0)?;
    let hf0_iso = (hf0[0][0] + hf0[1][1] + hf0[2][2]) / 3.0;
    let mp2_iso = mp2_stat.iso;
    let scale = if hf0_iso.abs() > 1e-12 { mp2_iso / hf0_iso } else { 1.0 };

    // HF frequency shape, rescaled to MP2 magnitude.
    let mut iso_prof = Vec::with_capacity(freqs.len());
    for &w in freqs {
        let t = dynamic_cphf_alpha_iw(ctx, mol, obs, dfbs, op, rhf, w)?;
        let hf_iso = (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        iso_prof.push(scale * hf_iso);
    }
    let mut s = 0.0;
    for k in 0..freqs.len() {
        s += weights[k] * iso_prof[k] * iso_prof[k];
    }
    Ok((3.0 / PI * s, iso_prof, mp2_iso, hf0_iso))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};

    #[test]
    fn analytic_alpha_fails_fast_under_tiny_budget() {
        // M2 size guard: an explicit ~1 KB budget must ERROR before the dense
        // nmo⁴ full-MO ERI is built. Explicit config budget → no env var touched.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cfg = RiMp2Config {
            frozen_core: 0,
            memory_budget_bytes: Some(ferric_core::memory::gib_to_bytes(1e-6)),
            ..Default::default()
        };
        let err = mp2_polarizability_analytic(&ctx, &mol, &obs, &dfbs, op, &bounds, &rhf, &cfg)
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("MO-ERI") && msg.contains("budget is"),
            "unexpected: {msg}"
        );
    }
}
