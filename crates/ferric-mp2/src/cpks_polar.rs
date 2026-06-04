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
