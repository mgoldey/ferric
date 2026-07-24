//! cDFT outer driver: nested optimization. For fixed λ the inner UHF/UKS solve
//! adds Σ_C λ_C W^C to the Fock; the outer Newton drives the residual
//! c_C(λ) = N_C[ρ_λ] − target_C to zero. c(λ) is monotonic in λ, so a few
//! outer iterations suffice. Written k-dimensional (k×k Jacobian) but exercised
//! at k=1; the Jacobian is finite-difference.

use crate::rhf::RhfConfig;
use crate::result::ScfResult;
use crate::screening::SchwarzBounds;
use crate::uhf::solve_uhf_fockmod;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_dft::ao_grid::eval_basis_on_points;
use ferric_dft::cdft::{build_weight_matrix, population, SpinChannel};
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;

/// Result of a constrained SCF.
pub struct CdftResult {
    /// Inner SCF result at the converged λ (energy is the ordinary KS energy at
    /// the constrained density — the constraint term is already excluded).
    pub scf: ScfResult,
    /// Converged Lagrange multipliers, one per constraint.
    pub lambdas: Vec<f64>,
    /// Final fragment populations N_C[ρ_λ].
    pub populations: Vec<f64>,
    /// Outer-loop iterations taken.
    pub outer_iters: usize,
}

/// Solve constrained UHF/UKS. Reads `config.constraints` and
/// `config.cdft_lambda_tol`. Requires at least one constraint.
pub fn solve_cdft_uhf(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &BasisSet,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
) -> Result<CdftResult, FerricError> {
    let cons = &config.constraints;
    if cons.is_empty() {
        return Err(FerricError::General(
            "solve_cdft_uhf: no constraints".into(),
        ));
    }
    let k = cons.len();

    // Build the DFT grid + AO values once, then W^C per constraint once.
    // The weight quadrature must be converged tighter than cdft_lambda_tol or
    // the constraint can never be satisfied; the default (75,110) grid plateaus
    // at ~1e-4 on a population, so use the 302-pt angular grid (the Lebedev
    // table max), which recovers populations to ~1e-8. Honor an explicit
    // config.dft_grid if the caller set one.
    let grid_cfg = config.dft_grid.clone().unwrap_or(AtomicGridConfig {
        n_radial: 99,
        n_angular: 302,
        ..Default::default()
    });
    let grid = build_atomic_grid(mol, &grid_cfg);
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(mol, bs, &pts)
        .map_err(|e| FerricError::General(format!("cDFT AO grid eval: {e:?}")))?;
    let w_mats: Vec<Array2<f64>> = cons
        .iter()
        .map(|c| build_weight_matrix(mol, &grid, &chi, &c.fragment))
        .collect();

    // Helper: run inner UHF for a given λ, return (scf, residual c(λ), pops).
    let run_inner = |lam: &[f64]| -> Result<(ScfResult, Vec<f64>, Vec<f64>), FerricError> {
        let fm = |f_a: &mut Array2<f64>, f_b: &mut Array2<f64>| {
            for (ci, c) in cons.iter().enumerate() {
                let l = lam[ci];
                match c.spin {
                    SpinChannel::Total => {
                        // same potential to both spins
                        let lw = l * &w_mats[ci];
                        *f_a += &lw;
                        *f_b += &lw;
                    }
                    SpinChannel::SpinDiff => {
                        let lw = l * &w_mats[ci];
                        *f_a += &lw;
                        *f_b -= &lw;
                    }
                }
            }
        };
        let scf = solve_uhf_fockmod(ctx, mol, prep, bounds, config, None, Some(&fm))?;
        let d_a = &scf.density_alpha;
        let d_b = scf.density_beta.as_ref().unwrap_or(d_a);
        let mut pops = vec![0.0; k];
        let mut resid = vec![0.0; k];
        for (ci, c) in cons.iter().enumerate() {
            let n_c = population(&w_mats[ci], d_a, d_b, &c.spin);
            pops[ci] = n_c;
            resid[ci] = n_c - c.target;
        }
        Ok((scf, resid, pops))
    };

    // Outer Newton on λ (start at 0).
    let mut lam = vec![0.0_f64; k];
    let max_outer = 30usize;
    let fd = 1e-3_f64; // λ finite-difference step for the Jacobian

    for outer in 1..=max_outer {
        let (scf, resid, pops) = run_inner(&lam)?;
        let max_resid = resid.iter().fold(0.0_f64, |m, &r| m.max(r.abs()));
        if max_resid < config.cdft_lambda_tol {
            return Ok(CdftResult {
                scf,
                lambdas: lam,
                populations: pops,
                outer_iters: outer,
            });
        }

        // Finite-difference Jacobian J_{ij} = ∂c_i/∂λ_j.
        let mut jac = Array2::<f64>::zeros((k, k));
        for j in 0..k {
            let mut lam_p = lam.clone();
            lam_p[j] += fd;
            let (_, resid_p, _) = run_inner(&lam_p)?;
            for i in 0..k {
                jac[(i, j)] = (resid_p[i] - resid[i]) / fd;
            }
        }

        // Solve J · Δλ = c, then λ ← λ − Δλ.
        let mut delta = solve_linear(&jac, &resid)?;
        // Damp/clamp the Newton step to keep the outer loop from overshooting
        // into a basin where the inner SCF stalls.
        for d in delta.iter_mut() {
            *d = d.clamp(-1.0, 1.0);
        }
        for j in 0..k {
            lam[j] -= delta[j]; // λ ← λ − J⁻¹ c
        }
    }

    Err(FerricError::Convergence(format!(
        "cDFT outer loop did not converge in {max_outer} iters"
    )))
}

/// Solve J x = b for small k via Gaussian elimination with partial pivoting.
/// Avoids a hard ndarray-linalg dependency for the k=1/k=2 case.
fn solve_linear(j: &Array2<f64>, b: &[f64]) -> Result<Vec<f64>, FerricError> {
    let n = b.len();
    let mut a = j.clone();
    let mut x = b.to_vec();
    for col in 0..n {
        // pivot
        let mut piv = col;
        for r in (col + 1)..n {
            if a[(r, col)].abs() > a[(piv, col)].abs() {
                piv = r;
            }
        }
        if a[(piv, col)].abs() < 1e-14 {
            return Err(FerricError::Lapack("cDFT Jacobian singular".into()));
        }
        if piv != col {
            for c in 0..n {
                let t = a[(col, c)];
                a[(col, c)] = a[(piv, c)];
                a[(piv, c)] = t;
            }
            x.swap(col, piv);
        }
        for r in (col + 1)..n {
            let f = a[(r, col)] / a[(col, col)];
            for c in col..n {
                a[(r, c)] -= f * a[(col, c)];
            }
            x[r] -= f * x[col];
        }
    }
    // back-substitute
    for col in (0..n).rev() {
        let mut s = x[col];
        for c in (col + 1)..n {
            s -= a[(col, c)] * x[c];
        }
        x[col] = s / a[(col, col)];
    }
    Ok(x)
}
