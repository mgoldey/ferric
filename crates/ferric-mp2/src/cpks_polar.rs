//! Coupled-perturbed relaxed RI-MP2 static polarizability (closed shell).
//!
//! `analytic_alpha_relaxed` (production, via `mp2_polarizability_analytic`):
//! dipole CPHF `U^x` → first-order Fock response → the first-order field-`±h`
//! reference (canonical orbitals of `F ± h∂F`) → central difference of
//! `static_relaxed_density_ao`. No SCF is re-converged, so the finite-field
//! path's SCF noise is absent. `α_xy = −Tr[∂D/∂F_x · r_y]`.
//!
//! Also here: HF-level CPHF α, dynamic CPHF α(iω), Casimir–Polder C6, the
//! frozen-amplitude MP2 C6 built on `analytic_alpha_relaxed`, and SCF-response
//! diagnostics.

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

use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

use crate::ff_polar::{eig3_sym_pub as eig3_sym, Mp2Polarizability};
use crate::rimp2::RiMp2Config;

/// Analytic relaxed RI-MP2 static polarizability (closed shell, no frozen
/// core). Delegates to [`analytic_alpha_relaxed`].
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
    analytic_alpha_relaxed(ctx, mol, obs, dfbs, op, bounds, rhf, mp2_config)
}

/// Solve `(Δε + ½·compute_az_product) X = rhs` by Jacobi-preconditioned CG.
/// The operator is the (A+B) orbital Hessian, the one `solve_zvector`
/// (`0.5 * compute_az_product`) uses. `rhs` and the
/// return are shape (nvir, nocc). Returns (X, final_residual, iters, converged).
///
/// `budget_bytes` is the caller-resolved memory ceiling threaded into every
/// `compute_az_product` call in the CG loop below (up to `max_iter=200`
/// calls) — callers with a `RiMp2Config` (or similar `memory_budget_bytes`
/// field) in scope should pass `resolve_budget_bytes(config.memory_budget_bytes)`,
/// resolved once at their own top; callers with no config in scope pass
/// `resolve_budget_bytes(None)`, likewise resolved once, not per CG iteration.
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
    solve_cphf_cg_to(c, rhs, prep, bounds, orb, eps, budget_bytes, CPHF_CG_TOL)
}

/// Max-abs residual at which [`solve_cphf_cg`] stops.
const CPHF_CG_TOL: f64 = 1e-10;

/// [`solve_cphf_cg`] at an explicit max-abs residual tolerance `tol`.
#[allow(clippy::too_many_arguments)]
fn solve_cphf_cg_to(
    c: &Array2<f64>,
    rhs: &Array2<f64>,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    orb: &OrbitalSpace,
    eps: &[f64],
    budget_bytes: usize,
    tol: f64,
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

    // compute_az_product returns twice the (A+B − Δε) coupling (its D^z is the
    // symmetric z density), hence the ½.
    let apply = |z: &Array2<f64>| -> Result<Array2<f64>, FerricError> {
        let mut mz = compute_az_product(c, z, prep, bounds, orb, &pool, budget_bytes)?;
        for a in 0..nvir {
            for i in 0..nocc {
                mz[(a, i)] = 0.5 * mz[(a, i)] + de(a, i) * z[(a, i)];
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
    let c_occ = c
        .slice(ndarray::s![.., first_occ..first_occ + nocc])
        .to_owned();
    let c_vir = c
        .slice(ndarray::s![.., nocc_total..nocc_total + nvir])
        .to_owned();
    Ok(std::array::from_fn(|d| {
        c_occ.t().dot(&dip_ao[d]).dot(&c_vir)
    }))
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
        let (ux, resid, iters, conv) =
            solve_cphf_cg(c, &rhs, obs, bounds, &orb, eps, budget_bytes)?;
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

use crate::rimp2::compute_mp2_intermediates;

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
                    dd[(mu, nu)] += 2.0
                        * uai
                        * (c_vir[(mu, a)] * c_occ[(nu, i)] + c_occ[(mu, i)] * c_vir[(nu, a)]);
                }
            }
        }
    }
    // FD of SCF total density
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let dens = |field: f64| -> Result<Array2<f64>, FerricError> {
        let mut v = Array2::<f64>::zeros((nbas, nbas));
        for m in 0..nbas {
            for n in 0..nbas {
                v[(m, n)] = -field * dip_ao[axis][(m, n)];
            }
        }
        let r = crate::ff_polar::solve_rhf_with_external(ctx, mol, obs, bounds, scf_config, &v)?;
        Ok(r.density_total().to_owned())
    };
    let dp = dens(h)?;
    let dm = dens(-h)?;
    let fd = (&dp - &dm).mapv(|x| x / (2.0 * h));
    let an_n = dd.iter().map(|x| x * x).sum::<f64>().sqrt();
    let fd_n = fd.iter().map(|x| x * x).sum::<f64>().sqrt();
    let maxd = dd
        .iter()
        .zip(fd.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
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
    for a in 0..nvir {
        for i in 0..nocc {
            let uai = u[(a, i)];
            for mu in 0..nbas {
                for nu in 0..nbas {
                    dd[(mu, nu)] += 2.0
                        * uai
                        * (c_vir[(mu, a)] * c_occ[(nu, i)] + c_occ[(mu, i)] * c_vir[(nu, a)]);
                }
            }
        }
    }
    let dens = |field: f64| -> Result<Array2<f64>, FerricError> {
        let mut v = Array2::<f64>::zeros((nbas, nbas));
        for m in 0..nbas {
            for n in 0..nbas {
                v[(m, n)] = -field * dip_ao[axis][(m, n)];
            }
        }
        let r = crate::ff_polar::solve_rhf_with_external(ctx, mol, obs, bounds, scf_config, &v)?;
        Ok(r.density_total().to_owned())
    };
    let fd = (&dens(h)? - &dens(-h)?).mapv(|x| x / (2.0 * h));
    let tr = |d: &Array2<f64>| -> [f64; 3] { std::array::from_fn(|y| -(d * &dip_ao[y]).sum()) };
    Ok((tr(&dd), tr(&fd)))
}

/// DIAGNOSTIC: E_MP2 at a single field along `axis` (for parabola/sign checks).
pub fn debug_emp2_at_field(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    scf_config: &ferric_scf::rhf::RhfConfig,
    mp2_config: &RiMp2Config,
    axis: usize,
    field: f64,
) -> Result<f64, FerricError> {
    let n = obs.nbasis();
    let dip = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let mut v = Array2::<f64>::zeros((n, n));
    for m in 0..n {
        for k in 0..n {
            v[(m, k)] = -field * dip[axis][(m, k)];
        }
    }
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
    let OrbitalSpace {
        nocc,
        nvir,
        nocc_total,
        first_occ,
    } = *orb;
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

/// Static relaxed RI-MP2 1-PDM in AO, `D = 2δ_core + (P+Pᵀ) + z`:
///   • P_oo, P_vv from `build_mp2_density`, assembled as P + Pᵀ.
///   • `Xvo = L + (2J−K)[dm_P]_vo`   (L = build_lagrangian; dm_P = the P+Pᵀ density).
///   • `(Δε + ½·compute_az_product) z = −Xvo`, the (A+B) Hessian of `solve_zvector`
///     (the two give the same z to the CG tolerance).
///
/// Its dipole equals `−∂E_MP2/∂F` from a finite-field difference of the RI-MP2
/// energy (`cpks_static_relaxed_dipole_vs_pyscf`). `build_relaxed_density_ao`
/// takes P one-sided, so feeding it `solve_zvector`'s z does not give this
/// density; pass it `P + Pᵀ` instead.
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
    if mp2_config.frozen_core > 0 {
        return Err(FerricError::General(format!(
            "{}: frozen_core = {} is not supported (no core–active orbital-relaxation \
             block); use frozen_core = 0",
            "static_relaxed_density_ao", mp2_config.frozen_core
        )));
    }
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

    // (Δε + (A+B−Δε)) z = −Xvo, the solve_zvector operator.
    let (z, resid, iters, conv) = solve_cphf_cg(c, &(-&xvo), obs, bounds, &orb, eps, budget_bytes)?;
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

/// Field step of the [`analytic_alpha_relaxed`] central difference. The
/// truncation error is `O(h²)`: 2.6e-6 at `h = 1e-3`, 2.6e-8 at `1e-4` on
/// water/STO-3G α_yy; the CPHF/Z-vector CG tolerance (1e-10) enters as
/// `≲ 1e-10 / h`.
const LR_FIELD_STEP: f64 = 1e-4;

/// First-order Fock response to a uniform field along one axis,
/// `∂F = r + J[∂D] − ½K[∂D]` (AO), with `∂D = 2(C_v U C_oᵀ + C_o Uᵀ C_vᵀ)` and
/// `U` the dipole CPHF solution `(A+B) U = −r_vo`. The field enters the core
/// Hamiltonian as `+F·r` (electron charge −1); the SCF Fock is
/// `h + J[D] − ½K[D]` with `D` the total density.
#[allow(clippy::too_many_arguments)]
fn first_order_fock_response(
    ctx: &ParallelContext,
    obs: &PreparedBasis,
    bounds: &SchwarzBounds,
    c: &Array2<f64>,
    eps: &[f64],
    orb: &OrbitalSpace,
    r_ao: &Array2<f64>,
    budget_bytes: usize,
) -> Result<Array2<f64>, FerricError> {
    let OrbitalSpace {
        nocc,
        nvir,
        nocc_total,
        first_occ,
    } = *orb;
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]);
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]);
    let r_vo = c_vir.t().dot(r_ao).dot(&c_occ);
    let (u, resid, iters, conv) = solve_cphf_cg(c, &(-&r_vo), obs, bounds, orb, eps, budget_bytes)?;
    if !conv {
        return Err(FerricError::General(format!(
            "first_order_fock_response: CPHF not converged (resid={resid:.2e}, iters={iters})"
        )));
    }
    let half = c_vir.dot(&u).dot(&c_occ.t());
    let dd = 2.0 * (&half + &half.t());
    let nbas = obs.nbasis();
    let mut j = Array2::<f64>::zeros((nbas, nbas));
    let mut k = Array2::<f64>::zeros((nbas, nbas));
    ferric_scf::rhf::build_jk(ctx, obs, bounds, 1e-12, &dd, &mut j, &mut k)?;
    Ok(r_ao + &j - &(0.5 * &k))
}

/// The RHF reference at field `s`, to first order: the canonical orbitals of
/// `F(s) = F + s·∂F`, diagonalised in the unperturbed MO basis. Occupied space,
/// orbital energies and Fock matrix equal those of the SCF converged at field
/// `s` to `O(s²)`, including the off-diagonal oo/vv Fock response, which the
/// diagonalisation re-canonicalises away.
fn reference_at_field(
    rhf: &ScfResult,
    dfock_ao: &Array2<f64>,
    s: f64,
    nocc: usize,
) -> Result<ScfResult, FerricError> {
    use ndarray_linalg::{Eigh, UPLO};
    let c = rhf.mos_r();
    let fock = rhf.fock_r() + &(s * dfock_ao);
    let f_mo = c.t().dot(&fock).dot(c);
    let f_mo = 0.5 * (&f_mo + &f_mo.t());
    let (w, v) = f_mo
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("reference_at_field: {e}")))?;
    let c_s = c.dot(&v);
    let d_alpha = {
        let c_occ = c_s.slice(ndarray::s![.., ..nocc]);
        c_occ.dot(&c_occ.t())
    };
    let mut out = rhf.clone();
    out.mos_alpha = c_s;
    out.eps_alpha = w.to_vec();
    out.fock_alpha = fock;
    out.density_total = 2.0 * &d_alpha;
    out.density_alpha = d_alpha;
    Ok(out)
}

/// `(α_HF, α_MP2)`, unsymmetrised, `[x][y] = −Tr[∂D/∂F_x · r_y]`, from the
/// central difference in `h` of the SCF density and of
/// [`static_relaxed_density_ao`] on [`reference_at_field`].
#[allow(clippy::too_many_arguments)]
fn relaxed_alpha_tensors(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    mp2_config: &RiMp2Config,
    h: f64,
) -> Result<([[f64; 3]; 3], [[f64; 3]; 3]), FerricError> {
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "analytic_alpha_relaxed: Restricted only".into(),
        ));
    }
    if mp2_config.frozen_core > 0 {
        return Err(FerricError::General(format!(
            "analytic_alpha_relaxed: frozen_core = {} is not supported (no core–active \
             orbital-relaxation block); use frozen_core = 0",
            mp2_config.frozen_core
        )));
    }
    let c = rhf.mos_r();
    let eps = rhf.eps_r();
    let nocc = (mol.nelec() / 2) as usize;
    let orb = OrbitalSpace::new(nocc, c.ncols() - nocc, nocc, 0);
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(mp2_config.memory_budget_bytes);
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let mut a_hf = [[0.0f64; 3]; 3];
    let mut a_mp2 = [[0.0f64; 3]; 3];
    for x in 0..3 {
        let df =
            first_order_fock_response(ctx, obs, bounds, c, eps, &orb, &dip_ao[x], budget_bytes)?;
        let mut d_hf = Vec::with_capacity(2);
        let mut d_rel = Vec::with_capacity(2);
        for s in [h, -h] {
            let r = reference_at_field(rhf, &df, s, nocc)?;
            d_rel.push(static_relaxed_density_ao(
                ctx, mol, obs, dfbs, op, bounds, &r, mp2_config,
            )?);
            d_hf.push(r.density_total);
        }
        let dd_hf = (&d_hf[0] - &d_hf[1]) / (2.0 * h);
        let dd_rel = (&d_rel[0] - &d_rel[1]) / (2.0 * h);
        for y in 0..3 {
            a_hf[x][y] = -(&dd_hf * &dip_ao[y]).sum();
            a_mp2[x][y] = -(&dd_rel * &dip_ao[y]).sum();
        }
    }
    Ok((a_hf, a_mp2))
}

/// Analytic relaxed RI-MP2 static polarizability (closed shell, no frozen
/// core). Per field axis `x`:
///   1. the dipole CPHF `U^x` and the first-order Fock response `∂F`
///      (`first_order_fock_response`);
///   2. the first-order field-`±h` reference: canonical orbitals of `F ± h∂F`
///      (`reference_at_field`);
///   3. `α_xy = −Tr[(D(+h) − D(−h))/(2h) · r_y]` with `D` =
///      `static_relaxed_density_ao`, whose dipole is `−∂E/∂F`.
///
/// No SCF is re-converged; the central difference is along the analytic
/// first-order reference, so it carries none of the finite-field path's SCF
/// noise. The SCF response terms use `bounds.op`, the MP2 terms `op`.
/// Matches PySCF finite-field DF-MP2 (same aux basis) to ≤ 2.2e-6 relative
/// (`relaxed_alpha_matches_pyscf_energy_hessian`).
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
    let (_, mut tensor) = relaxed_alpha_tensors(
        ctx,
        mol,
        obs,
        dfbs,
        op,
        bounds,
        rhf,
        mp2_config,
        LR_FIELD_STEP,
    )?;
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
// Dense full-MO RI ERI tensor (pq|rs) = Σ_P B[P,p,q] B[P,r,s] (nmo⁴, small
// systems only), shared by the TDHF/BSE response builders below.
// ===========================================================================
use ndarray::Array4;

/// Bytes a `full_mo_eri` caller holds at its nmo⁴-phase peak:
/// `pool_nmo4_buffers · nmo⁴ + naux · nmo²` f64.
///
/// The second term is the `b_full` (naux, nmo, nmo) tensor every caller keeps
/// alive to return. `full_mo_eri` alone holds two nmo⁴ buffers (the Gram
/// matrix `g` and the tensor it is copied into), so the TDHF/BSE callers pass
/// 2.
fn full_mo_eri_peak_bytes(nmo: usize, naux: usize, pool_nmo4_buffers: usize) -> usize {
    let nmo2 = nmo.saturating_mul(nmo);
    nmo2.saturating_mul(nmo2)
        .saturating_mul(pool_nmo4_buffers)
        .saturating_add(naux.saturating_mul(nmo2))
        .saturating_mul(8)
}

/// Fail-fast pre-flight guard for every `full_mo_eri` caller: ONE peak
/// ([`full_mo_eri_peak_bytes`]) is both checked against the resolved budget
/// ceiling and charged to the shared memory pool.
///
/// The charge is taken BEFORE `compute_b_full_mo` and held to function end,
/// so it is also outstanding while that call's AO 3-index tensor (charged by
/// its own source) exists. The two do not actually coexist with the nmo⁴
/// buffers, so this is deliberately conservative by the AO peak (a sum, not a
/// max): the refusal happens before any integrals are built. Inert when no
/// pool is installed. Placed next to `full_mo_eri` so an M3/M5 restructure
/// that shrinks the peak updates the formula in the same diff.
fn guard_full_mo_eri(
    label: &str,
    nmo: usize,
    naux: usize,
    pool_nmo4_buffers: usize,
    explicit_budget: Option<usize>,
) -> Result<ferric_core::memory::pool::Reservation, FerricError> {
    let label = format!("{label}: dense nmo⁴ MO-ERI (nmo={nmo})");
    let peak = full_mo_eri_peak_bytes(nmo, naux, pool_nmo4_buffers);
    ferric_core::memory::check_alloc(
        &label,
        peak,
        ferric_core::memory::resolve_budget_bytes(explicit_budget),
    )?;
    crate::rimp2::charge_mo_side(&label, peak)
}

/// Full-MO ERI (pq|rs) from the dressed B tensor. (nmo^4 — small systems only.)
fn full_mo_eri(b_full: &ndarray::Array3<f64>) -> Array4<f64> {
    let naux = b_full.shape()[0];
    let nmo = b_full.shape()[1];
    let mut imo = Array4::<f64>::zeros((nmo, nmo, nmo, nmo));
    // (pq|rs) = Σ_P B[P,p,q] B[P,r,s]
    let bmat = b_full
        .view()
        .into_shape_with_order((naux, nmo * nmo))
        .unwrap();
    let g = bmat.t().dot(&bmat); // (nmo*nmo, nmo*nmo)
    for p in 0..nmo {
        for q in 0..nmo {
            for rr in 0..nmo {
                for s in 0..nmo {
                    imo[(p, q, rr, s)] = g[(p * nmo + q, rr * nmo + s)];
                }
            }
        }
    }
    imo
}

#[inline]
fn eri4(imo: &Array4<f64>, p: usize, q: usize, r: usize, s: usize) -> f64 {
    imo[(p, q, r, s)]
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
// (A+B) = Δε + 4(ai|bj) − (ab|ij) − (aj|bi)  ≡ the static Hessian
//         `solve_cphf_cg` applies. So the static driver's M is (A+B), and
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

    // Fail-fast size guard: full_mo_eri holds the nmo⁴ ERI + nmo⁴ Gram. The
    // pool charge is held to function end, where imo and b_full die.
    let _imo_charge = guard_full_mo_eri("BSE gate-0 residuals", nmo, dfbs.nbasis(), 2, None)?;
    let b_full = crate::oo_rimp2::compute_b_full_mo(obs, dfbs, op, c)?; // (naux, nmo, nmo)
    let imo = full_mo_eri(&b_full);

    // Reference: existing validated TDHF blocks.
    let (apb_ref, amb_ref) = build_apb_amb(&imo, &eps, nocc, nvir);

    // Bare-v limit of the screened builder: modes = raw RI tensor, weights = 1.
    let naux = b_full.shape()[0];
    let weights = vec![1.0_f64; naux];
    let (apb_w, amb_w) = build_apb_amb_screened(&imo, &b_full, &weights, &eps, nocc, nvir);

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
        return Err(FerricError::General(
            "dynamic_cphf_alpha_iw: Restricted only".into(),
        ));
    }
    let c = rhf.mos_r();
    let nmo = c.ncols();
    let nocc = (mol.nelec() / 2) as usize;
    let nvir = nmo - nocc;
    let n = nvir * nocc;
    let eps: Vec<f64> = rhf.eps_r().to_vec();

    // Fail-fast size guard: full_mo_eri holds the nmo⁴ ERI + nmo⁴ Gram. The
    // pool charge is held to function end, where imo and b_full die.
    let _imo_charge = guard_full_mo_eri("dynamic CPHF/TDHF α(iω)", nmo, dfbs.nbasis(), 2, None)?;
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
// With an attenuated `op` (att-MP2: Coulomb RHF reference, erfc(ωr) on the
// correlation only) the HF shape and the static HF response use `bounds.op`,
// the operator the reference was converged with; `op` enters only the MP2
// correlation of the static magnitude, as in `analytic_alpha_relaxed`.
//
// Reduces to the validated static MP2 α at ω=0 BY CONSTRUCTION. Tests whether
// MP2's magnitude correction flips the attenuation→C6 verdict vs HF-level.
// It does NOT capture an MP2-specific change to the frequency *shape* — that
// is exactly what the full (expensive) response would add, and the gap
// between this and dRPA tells us whether that's worth deriving.
// ===========================================================================

/// Frozen-amplitude MP2 dynamic molecular C6: HF-shape α(iω) rescaled to the
/// static MP2 isotropic magnitude. Returns (c6, α_MP2_iso profile, static MP2
/// iso, static HF iso). Closed shell, no frozen core. `bounds` must be built for
/// the operator `rhf` was converged with (it sets the HF response); `op` is the
/// MP2 correlation operator.
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
    // Static relaxed MP2 α and static HF α (ω=0 dynamic) on the reference operator.
    let hf_op = bounds.op;
    let mp2_stat = analytic_alpha_relaxed(ctx, mol, obs, dfbs, op, bounds, rhf, mp2_config)?;
    let hf0 = dynamic_cphf_alpha_iw(ctx, mol, obs, dfbs, hf_op, rhf, 0.0)?;
    let hf0_iso = (hf0[0][0] + hf0[1][1] + hf0[2][2]) / 3.0;
    let mp2_iso = mp2_stat.iso;
    let scale = if hf0_iso.abs() > 1e-12 {
        mp2_iso / hf0_iso
    } else {
        1.0
    };

    // HF frequency shape, rescaled to MP2 magnitude.
    let mut iso_prof = Vec::with_capacity(freqs.len());
    for &w in freqs {
        let t = dynamic_cphf_alpha_iw(ctx, mol, obs, dfbs, hf_op, rhf, w)?;
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
        // Size guard: an explicit ~1 KB budget must ERROR before the RI-MP2
        // intermediates are built. Explicit config budget → no env var touched.
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
        let err =
            analytic_alpha_relaxed(&ctx, &mol, &obs, &dfbs, op, &bounds, &rhf, &cfg).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("RI-MP2 intermediates") && msg.contains("budget is"),
            "unexpected: {msg}"
        );
    }

    fn water_sto3g() -> (
        Molecule,
        PreparedBasis,
        PreparedBasis,
        Operator,
        SchwarzBounds,
        ParallelContext,
        ScfResult,
    ) {
        let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let ctx = ParallelContext::default();
        let cfg = RhfConfig {
            energy_conv: 1e-12,
            density_conv: 1e-10,
            ..Default::default()
        };
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();
        (mol, obs, dfbs, op, bounds, ctx, rhf)
    }

    /// Exactness anchor of the field-response construction: with the MP2
    /// density switched off (the SCF density of `reference_at_field`), the
    /// central difference must reproduce the CPHF HF α. Measured 4e-9 rel at
    /// h = 1e-4. A doubled two-electron response (2J − K) or a flipped field
    /// sign in `first_order_fock_response` misses by > 1e-2.
    #[test]
    fn field_response_reference_reproduces_cphf_hf_alpha() {
        let (mol, obs, dfbs, op, bounds, ctx, rhf) = water_sto3g();
        let cfg = RiMp2Config {
            frozen_core: 0,
            memory_budget_bytes: None,
            ..Default::default()
        };
        let hf = mp2_polarizability_analytic_hf(&ctx, &mol, &obs, &op, &bounds, &rhf).unwrap();
        let (a_hf, _) = relaxed_alpha_tensors(
            &ctx,
            &mol,
            &obs,
            &dfbs,
            op,
            &bounds,
            &rhf,
            &cfg,
            LR_FIELD_STEP,
        )
        .unwrap();
        for x in 0..3 {
            for y in 0..3 {
                let want = hf.tensor[x][y];
                assert!(
                    (a_hf[x][y] - want).abs() < 1e-7 * (1.0 + want.abs()),
                    "α_HF[{x}][{y}] = {} vs CPHF {want}",
                    a_hf[x][y]
                );
            }
        }
    }

    /// CH4/aug-cc-pVDZ (Coulomb RHF): the system whose MP2 Z-vector CG stalled
    /// at a 1.98e-9 residual (erfc ω = 0.42 attenuation sweep).
    fn ch4_aug() -> (PreparedBasis, SchwarzBounds, ScfResult) {
        let xyz = "5\nch4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\n\
             H -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvdz").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let cfg = RhfConfig {
            energy_conv: 1e-12,
            density_conv: 1e-10,
            ..Default::default()
        };
        let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
        (obs, bounds, rhf)
    }

    /// −μ^y_vo, the dipole CPHF right-hand side, shape (nvir, nocc).
    fn dipole_rhs(obs: &PreparedBasis, rhf: &ScfResult, nocc: usize) -> Array2<f64> {
        let c = rhf.mos_r();
        let dip = oneelectron::dipole(obs, [0.0; 3]).unwrap();
        let cv = c.slice(ndarray::s![.., nocc..]);
        let co = c.slice(ndarray::s![.., ..nocc]);
        -cv.t().dot(&dip[1]).dot(&co)
    }

    /// The Hessian matvec is linear, so `A(2⁻³⁰·z) = 2⁻³⁰·A(z)` bit for bit
    /// once the J/K screening sees a scale-free operand. With the absolute
    /// `bound · max|D| < 1e-12` screen applied to the raw operand, the
    /// 2⁻³⁰-scaled product misses by 1.1e-1 against `max|A(z)|` = 2.9e-1
    /// (measured).
    #[test]
    fn az_product_is_homogeneous() {
        let (obs, bounds, rhf) = ch4_aug();
        let c = rhf.mos_r();
        let nocc = 5;
        let orb = OrbitalSpace::new(nocc, c.ncols() - nocc, nocc, 0);
        let budget = ferric_core::memory::resolve_budget_bytes(None);
        let pool = ferric_scf::engine_pool::EnginePool::new(bounds.op, &obs, 1e-14).unwrap();
        let az = |z: &Array2<f64>| {
            crate::zvector::compute_az_product(c, z, &obs, &bounds, &orb, &pool, budget).unwrap()
        };
        let z = dipole_rhs(&obs, &rhf, nocc);
        let s = 2f64.powi(-30);
        let full = az(&z);
        let small = az(&(s * &z));
        let amax = full.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let miss = (&small / s - &full)
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            miss == 0.0,
            "A(s·z)/s − A(z): max {miss:.3e} (max|A(z)| = {amax:.3e})"
        );
    }

    /// Dipole CPHF on CH4/aug-cc-pVDZ to a 1e-12 max-abs residual, 100× below
    /// the production tolerance, so the last CG operands are ~1e-13. The
    /// preconditioned Hessian has κ = 2.36 (eigenvalues 0.593..1.400; the bare
    /// Hessian's smallest is 0.411, dense build), so PCG converges
    /// geometrically (rate (√κ−1)/(√κ+1) = 0.21, ~16 iterations from the
    /// 1.3e-1 start; measured 12), so the bound is 40. With
    /// the matvec's J/K screen on the raw operand, `A·p` loses its coupling as
    /// `p` shrinks and the residual stalls at ~1e-10 (measured).
    #[test]
    fn cphf_cg_converges_past_the_production_tolerance() {
        let (obs, bounds, rhf) = ch4_aug();
        let c = rhf.mos_r();
        let eps = rhf.eps_r();
        let nocc = 5;
        let orb = OrbitalSpace::new(nocc, c.ncols() - nocc, nocc, 0);
        let budget = ferric_core::memory::resolve_budget_bytes(None);
        let rhs = dipole_rhs(&obs, &rhf, nocc);
        let (_, resid, iters, conv) =
            solve_cphf_cg_to(c, &rhs, &obs, &bounds, &orb, eps, budget, 1e-12).unwrap();
        assert!(
            conv && iters <= 40,
            "CPHF CG to 1e-12: converged {conv}, resid {resid:.3e}, iters {iters}"
        );
    }
}
