//! cDFT outer driver: nested optimization. For fixed λ the inner UHF/UKS solve
//! adds Σ_C λ_C W^C to the Fock; the outer Newton drives the residual
//! c_C(λ) = N_C\[ρ_λ\] − target_C to zero. c(λ) is monotonic in λ, so a few
//! outer iterations suffice. Written k-dimensional (k×k Jacobian) but exercised
//! at k=1; the Jacobian is finite-difference.

use crate::result::ScfResult;
use crate::rhf::RhfConfig;
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
#[derive(Debug, Clone)]
#[must_use]
pub struct CdftResult {
    /// Inner SCF result at the converged λ (energy is the ordinary KS energy at
    /// the constrained density — the constraint term is already excluded).
    pub scf: ScfResult,
    /// Converged Lagrange multipliers, one per constraint.
    pub lambdas: Vec<f64>,
    /// Final fragment populations `N_C[ρ_λ]`.
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
    // Pre-flight the AO-grid buffer against the memory budget.
    //
    // This path is the largest ungated grid allocation in ferric-scf: the cDFT
    // default grid is 99x302 = 29,898 points/atom, roughly 3.6x the 75x110
    // production default, so `chi` alone is 12.2 GB at danuglipron/def2-SVP
    // scale (73 atoms, nbf ~ 700) and 27.9 GB at def2-TZVP —
    // and `build_weight_matrix` clones it, doubling that. Without this check the
    // job walked straight into the allocation and was OOM-killed by the kernel;
    // now it fails fast with a message naming the knobs.
    //
    // `ValueOnly`: only chi is built here (no gradients). The clone inside
    // `build_weight_matrix` is a second plane, so this under-charges by 1x --
    // stated rather than silently absorbed, because charging it here would
    // reject grids that the pre-clone code path handles fine. The honest fix is
    // to remove the clone; see the note there.
    ferric_dft::ao_grid::check_ao_grid_budget(
        ferric_dft::ao_grid::AoGridKind::ValueOnly,
        prep.nbasis(),
        pts.len(),
    )
    .map_err(|e| FerricError::General(format!("cDFT AO grid: {e}")))?;

    let w_mats: Vec<Array2<f64>> = {
        let chi = eval_basis_on_points(mol, bs, &pts)
            .map_err(|e| FerricError::General(format!("cDFT AO grid eval: {e:?}")))?;
        let w = cons
            .iter()
            .map(|c| build_weight_matrix(mol, &grid, &chi, &c.fragment))
            .collect();
        // Scope chi so it is FREED here rather than at function end. It is dead
        // after this point (nothing below reads it), but Rust would otherwise
        // keep the full (nbf, npts) buffer resident through the entire
        // lambda-Newton loop below -- which runs a full inner UHF per iteration
        // and is exactly where the peak matters.
        w
    };

    // The λ-Newton loop, as a closure over an OPTIONAL starting orbital guess.
    //
    // Factored out (it used to be inline) so the stability-descent block below
    // can re-run the WHOLE outer loop from a rotated guess. The guess is
    // applied at EVERY inner solve, not only the first: re-seeding each λ is
    // what keeps a descent inside the basin it was aimed at, whereas seeding
    // only λ⁰ lets the subsequent inner solves drift back into the saddle.
    let lambda_newton =
        |guess: Option<(&Array2<f64>, &Array2<f64>)>| -> Result<CdftResult, FerricError> {
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
                let scf = solve_uhf_fockmod(ctx, mol, prep, bounds, config, guess, Some(&fm))?;
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
        };

    let first = lambda_newton(None)?;
    if !config.cdft_stability_descent {
        return Ok(first);
    }
    Ok(stability_descent(
        ctx,
        mol,
        prep,
        bounds,
        config,
        &w_mats,
        first,
        &lambda_newton,
    ))
}

/// Step sizes tried when following a downhill eigenvector, in radians.
///
/// NOT a guess: measured on `test/cdft-constrained-stability`, where steps
/// ε ≤ 0.5 rad fall straight back into the saddle's own DIIS basin and only
/// ~1 rad escapes it. The list keeps the smaller ones so a system where a
/// gentler step suffices is not over-rotated past its minimum, and takes the
/// LOWEST constraint-satisfying result over the whole sweep rather than the
/// first success.
const DESCENT_STEPS: [f64; 3] = [0.4, 0.8, 1.2];

/// Maximum descent rounds. Each round is one eigensolve plus up to
/// `DESCENT_STEPS.len()` full λ-Newton solves, so this bounds the worst-case
/// cost at a small multiple of the unfixed solve.
const MAX_DESCENT_ROUNDS: usize = 3;

/// **cDFT state selection.** Given a converged constrained solution, check
/// whether it is a SADDLE of its own λ-augmented functional and, if so, follow
/// the downhill direction and re-converge the whole λ-Newton loop from there.
///
/// # Why the plain UHF Hessian is the right operator here
///
/// The cDFT Lagrangian `W[ρ,λ] = E[ρ] + λ(N_C[ρ] − N_target)` is a saddle in
/// the combined (ρ, λ) space BY CONSTRUCTION — minimized over ρ, maximized over
/// λ — so an ordinary `E[ρ]` stability analysis at a constrained solution would
/// report spurious instabilities on perfectly good diabats. The correct
/// question is internal stability of the λ-AUGMENTED problem AT FIXED λ.
///
/// That happens to need no new derivation. The constraint enters as `λW` with
/// `W` built ONCE from geometry and the Becke grid, outside the λ loop and
/// never rebuilt from the density (see `build_weight_matrix` above). A
/// density-independent one-electron term contributes to the Fock but NOT to the
/// Fock response `δF/δD`, so the λ-augmented orbital Hessian EQUALS the
/// ordinary UHF orbital Hessian evaluated at the constrained orbitals with the
/// CONSTRAINED (λ-augmented) orbital energies — which is exactly what
/// `uhf_newton::hessian_matvec` computes when handed this solution's `C` and
/// its λ-augmented MO Fock.
///
/// This is not assumed: it is finite-differenced against the analytic matvec by
/// `augmented_hessian_equals_plain_hessian_at_fixed_lambda` in
/// `tests/cdft_constrained_stability.rs`, and the λ-dependence of the gradient
/// it rests on is separately FD-checked at λ = −2.75 (with a doubled-λW
/// mutation) by `tests/cdft_state_selection.rs`.
///
/// `solve_uhf_fockmod` applies the Fock modifier AFTER forming the energy, so
/// `ScfResult::energy` is the BARE energy at the constrained density while
/// `fock_alpha`/`fock_beta` ARE λ-augmented — the combination this needs.
///
/// # Failure policy
///
/// Every failure mode returns the INPUT solution unchanged, after printing why.
/// A descent that cannot be computed, does not re-converge, drifts off the
/// constraint, or lands HIGHER must never make the answer worse than not
/// having tried — this function can only ever improve on `first` or leave it
/// alone.
#[allow(clippy::too_many_arguments)]
fn stability_descent(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    w_mats: &[Array2<f64>],
    first: CdftResult,
    lambda_newton: &dyn Fn(Option<(&Array2<f64>, &Array2<f64>)>) -> Result<CdftResult, FerricError>,
) -> CdftResult {
    // A KS reference needs the f_xc response kernel or the analysis is of the
    // WRONG OPERATOR (the HF Hessian at a KS density), which is exactly the
    // trap `uhf::stability_uhf` documents. Rather than reproduce that kernel
    // plumbing here, the descent SKIPS on any XC reference and says so. Pure
    // UHF — which is what the cDFT lane is validated on — is unaffected.
    if config.xc.is_some() {
        eprintln!(
            "cDFT stability descent: SKIPPED on a KS reference (xc = {:?}). The \
             λ-augmented Hessian would need the f_xc response kernel, and analysing \
             the HF Hessian at a KS density instead would be a wrong-operator \
             verdict. The constrained solution is returned as converged, which does \
             NOT mean it is the lowest state at this constraint.",
            config.xc
        );
        return first;
    }

    let mut best = first;
    for round in 1..=MAX_DESCENT_ROUNDS {
        let Some((va, vb, lmin, verdict)) =
            augmented_instability(ctx, mol, prep, bounds, config, w_mats, &best)
        else {
            return best;
        };
        if verdict != crate::stability::StabilityVerdict::Unstable {
            if round == 1 {
                eprintln!(
                    "cDFT stability descent: the constrained solution is {} \
                     (λ_min = {lmin:+.4e}); no descent taken.",
                    verdict_label(verdict)
                );
            }
            return best;
        }
        eprintln!(
            "cDFT stability descent (round {round}): the constrained solution at \
             E = {:.8} is a SADDLE of its λ-augmented functional (λ_min = \
             {lmin:+.4e}); following the downhill eigenvector.",
            best.scf.energy
        );

        // Sweep the step sizes and keep the LOWEST solution that still
        // satisfies the constraint. A restart that fails to converge or that
        // drifts off the target is skipped, not fatal.
        let mut improved: Option<CdftResult> = None;
        for &step in &DESCENT_STEPS {
            let Some(cb) = best.scf.mos_beta.as_ref() else {
                return best;
            };
            let nocc_a = occupied_alpha(mol);
            let nocc_b = occupied_beta(mol);
            let g_a = rotate_mos(&best.scf.mos_alpha, &va, nocc_a, step);
            let g_b = rotate_mos(cb, &vb, nocc_b, step);
            let cand = match lambda_newton(Some((&g_a, &g_b))) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("cDFT stability descent: step {step} did not re-converge ({e:?})");
                    continue;
                }
            };
            // The candidate must satisfy the SAME constraint, else it is a
            // different problem's answer and not a better B.
            let off = constraint_offset(&cand.populations, &config.constraints);
            if off >= config.cdft_lambda_tol {
                eprintln!(
                    "cDFT stability descent: step {step} re-converged at E = {:.8} but \
                     OFF the constraint by {off:.2e} (tol {:.1e}); discarded.",
                    cand.scf.energy, config.cdft_lambda_tol
                );
                continue;
            }
            if accepts_candidate(
                cand.scf.energy,
                best.scf.energy,
                improved.as_ref().map(|b| b.scf.energy),
            ) {
                improved = Some(cand);
            }
        }

        match improved {
            Some(c) => {
                eprintln!(
                    "cDFT stability descent (round {round}): reached a LOWER constrained \
                     solution, E = {:.8} (was {:.8}, ΔE = {:.8} Ha = {:.4} eV), \
                     λ = {:?}, N_C = {:?}",
                    c.scf.energy,
                    best.scf.energy,
                    best.scf.energy - c.scf.energy,
                    (best.scf.energy - c.scf.energy) * 27.211_386_245_988,
                    c.lambdas,
                    c.populations
                );
                best = c;
            }
            None => {
                eprintln!(
                    "cDFT stability descent (round {round}): the solution is a saddle \
                     (λ_min = {lmin:+.4e}) but NO step reached a lower \
                     constraint-satisfying solution. Returning the saddle, which is \
                     therefore NOT established as the lowest state at this constraint."
                );
                return best;
            }
        }
    }
    eprintln!(
        "cDFT stability descent: still descending after {MAX_DESCENT_ROUNDS} rounds; \
         returning the lowest found (E = {:.8}). It is NOT established as the bottom.",
        best.scf.energy
    );
    best
}

/// Should a descended candidate replace the best-so-far?
///
/// Only if it is BELOW both the incumbent (`best_e`) and any better candidate
/// already found this round (`improved_e`). Strict `<` throughout, so an exactly
/// equal energy does not churn the answer.
///
/// Extracted for the same reason as [`constraint_offset`]: inline it was
/// UNREACHABLE. Replacing `cand < best` with `true` — i.e. accepting a strictly
/// HIGHER state — left the whole cDFT suite GREEN, because on HeNe⁺ every
/// descended candidate happens to be lower. That guard is the entire reason the
/// descent cannot make an answer worse than not having tried, so it must be
/// tested rather than assumed. See `descent_never_accepts_a_higher_state`.
fn accepts_candidate(cand_e: f64, best_e: f64, improved_e: Option<f64>) -> bool {
    cand_e < best_e && improved_e.is_none_or(|b| cand_e < b)
}

/// How far a candidate solution sits from its constraint targets, as the max
/// over constraints of `|N_C − target_C|`.
///
/// Extracted from the descent loop so it can be tested DIRECTLY. Inline, it was
/// unreachable: on HeNe⁺ every descended candidate satisfies the constraint, so
/// deleting the check entirely (`if false && off >= tol`) left the whole cDFT
/// suite GREEN. A guard no test has ever been seen to exercise is an
/// assumption, and this one decides whether a lower-energy state is accepted —
/// exactly the decision the fix exists to make. See
/// `constraint_offset_rejects_a_candidate_that_missed_the_target`.
///
/// A candidate with FEWER populations than constraints (which cannot happen —
/// the λ-Newton loop fills one per constraint — but is representable) would
/// have its missing constraints silently ignored by a plain `zip`, so the
/// length mismatch is treated as infinitely far off rather than as agreement.
fn constraint_offset(pops: &[f64], cons: &[ferric_dft::cdft::Constraint]) -> f64 {
    if pops.len() != cons.len() {
        return f64::INFINITY;
    }
    pops.iter()
        .zip(cons.iter())
        .fold(0.0_f64, |m, (&n, c)| m.max((n - c.target).abs()))
}

fn verdict_label(v: crate::stability::StabilityVerdict) -> &'static str {
    use crate::stability::StabilityVerdict as V;
    match v {
        V::Stable => "STABLE",
        V::Unstable => "UNSTABLE",
        V::Marginal => "MARGINAL (|λ_min| at the noise floor — neither proven)",
        V::Indeterminate => "INDETERMINATE (the eigensolve did not converge)",
    }
}

fn occupied_alpha(mol: &Molecule) -> usize {
    let nelec = mol.nelec() as usize;
    (nelec + (mol.multiplicity - 1)) / 2
}
fn occupied_beta(mol: &Molecule) -> usize {
    let nelec = mol.nelec() as usize;
    (nelec - (mol.multiplicity - 1)) / 2
}

/// λ_min and its eigenvector for the λ-augmented orbital Hessian at a converged
/// constrained solution. `None` on any analysis failure, always after saying so.
#[allow(clippy::type_complexity)]
fn augmented_instability(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    w_mats: &[Array2<f64>],
    sol: &CdftResult,
) -> Option<(
    Array2<f64>,
    Array2<f64>,
    f64,
    crate::stability::StabilityVerdict,
)> {
    let c_a = &sol.scf.mos_alpha;
    let c_b = sol.scf.mos_beta.as_ref()?;
    // `fock_alpha`/`fock_beta` are ALREADY λ-augmented (the Fock modifier is
    // applied inside the inner SCF), so no λW is re-added here. Re-adding it
    // would double-count the constraint — the epcdft-class defect the λ = 0
    // anchor cannot see. `w_mats` is taken only to assert that.
    debug_assert_eq!(w_mats.len(), config.constraints.len());
    let nocc_a = occupied_alpha(mol);
    let nocc_b = occupied_beta(mol);
    let f_a = &sol.scf.fock_alpha;
    let f_b = sol.scf.fock_beta.as_ref()?;
    let f_a_mo = c_a.t().dot(f_a).dot(c_a);
    let f_b_mo = c_b.t().dot(f_b).dot(c_b);
    let inputs = crate::uhf_newton::UhfNewtonInputs {
        prep,
        bounds,
        c_a,
        c_b,
        f_a_mo: &f_a_mo,
        f_b_mo: &f_b_mo,
        nocc_a,
        nocc_b,
        k_mix_sr: 1.0,
        fxc: None,
        thresh: config.integral_thresh,
        ooc_budget: 0,
    };
    match crate::stability::uhf_internal_stability(
        ctx,
        &inputs,
        &crate::stability::StabilityConfig::default(),
    ) {
        Ok(res) => {
            let vb = res.eigenvector_beta.clone()?;
            let verdict = res.verdict();
            Some((
                res.eigenvector_alpha.clone(),
                vb,
                res.lowest_eigenvalue,
                verdict,
            ))
        }
        Err(e) => {
            eprintln!(
                "cDFT stability descent: the λ-augmented stability analysis FAILED ({e}). \
                 The constrained solution is returned unchanged, and is NOT established \
                 as the lowest state at this constraint."
            );
            None
        }
    }
}

/// Cayley rotation of `c` by `eps · κ_ov`, exactly orthonormality-preserving.
///
/// `(I − κ/2)⁻¹(I + κ/2)` is orthogonal for antisymmetric κ to machine
/// precision, unlike a truncated `exp(κ)` — which matters because the rotated
/// orbitals are fed straight back in as an SCF guess.
fn rotate_mos(c: &Array2<f64>, k_ov: &Array2<f64>, nocc: usize, eps: f64) -> Array2<f64> {
    use ndarray_linalg::Solve;
    let n = c.nrows();
    let mut kappa = Array2::<f64>::zeros((n, n));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            if ir >= k_ov.nrows() || i >= k_ov.ncols() {
                continue;
            }
            let v = eps * k_ov[(ir, i)];
            kappa[(a, i)] = v;
            kappa[(i, a)] = -v;
        }
    }
    let half = 0.5 * &kappa;
    let eye = Array2::<f64>::eye(n);
    let am = &eye - &half;
    let bm = &eye + &half;
    let mut u = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        match am.solve(&bm.column(col).to_owned()) {
            Ok(sol) => {
                for row in 0..n {
                    u[(row, col)] = sol[row];
                }
            }
            // A singular (I − κ/2) cannot happen for antisymmetric κ (its
            // eigenvalues are 1 ± i·imag), but the solve is fallible, so fall
            // back to the identity column rather than panicking inside a
            // best-effort descent.
            Err(_) => u[(col, col)] = 1.0,
        }
    }
    c.dot(&u)
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

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_dft::cdft::{Constraint, SpinChannel};

    fn con(target: f64) -> Constraint {
        Constraint {
            fragment: vec![0],
            target,
            spin: SpinChannel::Total,
        }
    }

    /// The descent's constraint-satisfaction guard, tested directly because it
    /// is UNREACHABLE through the driver on the systems in the suite.
    ///
    /// MEASURED 2026-09-16: deleting the guard inline (`if false && off >=
    /// tol`) left every cDFT test GREEN, because on HeNe⁺ every descended
    /// candidate happens to land on the target. That makes the inline check an
    /// assumption; this makes it a test. The guard matters because without it
    /// the descent would accept a LOWER-energy state that solves a DIFFERENT
    /// constraint — which is not a better diabat, just a different one.
    #[test]
    fn constraint_offset_rejects_a_candidate_that_missed_the_target() {
        // On target: offset is the residual, below any sane tolerance.
        assert!(constraint_offset(&[2.000_000_02], &[con(2.0)]) < 1e-5);
        // Off target by 0.05 e: far above the 1e-5 default tolerance, so the
        // descent discards it however low its energy is.
        let off = constraint_offset(&[1.95], &[con(2.0)]);
        assert!(
            off > 1e-5,
            "a candidate 0.05 e off the target must be rejected; offset = {off:.3e}"
        );
        assert!((off - 0.05).abs() < 1e-12, "offset = {off}");
        // MAX over constraints, not sum or first: one satisfied constraint must
        // not mask a violated one.
        let both = constraint_offset(&[2.0, 1.0], &[con(2.0), con(1.5)]);
        assert!(
            (both - 0.5).abs() < 1e-12,
            "the offset must be the MAX over constraints, so a satisfied one \
             cannot hide a violated one; got {both}"
        );
        // A length mismatch is infinitely far off, not silently truncated.
        assert!(constraint_offset(&[2.0], &[con(2.0), con(1.0)]).is_infinite());
        assert!(constraint_offset(&[], &[con(2.0)]).is_infinite());
    }

    /// The descent's "never make it worse" guarantee, tested directly because
    /// it is UNREACHABLE through the driver on the systems in the suite.
    ///
    /// MEASURED 2026-09-16: replacing the `cand < best` test with `true` — so
    /// the descent would accept a strictly HIGHER state — left every cDFT test
    /// GREEN, because on HeNe⁺ every descended candidate is lower anyway. This
    /// is the guard that makes the fix safe to default ON: without it, a
    /// descent that overshoots into a worse basin would silently replace a good
    /// answer with a bad one.
    #[test]
    fn descent_never_accepts_a_higher_state() {
        // Lower than the incumbent, nothing better found yet: ACCEPT.
        assert!(accepts_candidate(-130.42, -130.40, None));
        // HIGHER than the incumbent: REJECT, however tempting.
        assert!(
            !accepts_candidate(-130.38, -130.40, None),
            "a descended state ABOVE the incumbent must never be accepted — that \
             would make the fix capable of making an answer worse"
        );
        // Equal to the incumbent: REJECT (strict <), so an equal-energy result
        // does not churn which solution is returned.
        assert!(!accepts_candidate(-130.40, -130.40, None));
        // Lower than the incumbent but ABOVE a better candidate already found
        // this round: REJECT, so the sweep keeps the LOWEST step, not the last.
        assert!(!accepts_candidate(-130.41, -130.40, Some(-130.43)));
        // Lower than both: ACCEPT.
        assert!(accepts_candidate(-130.45, -130.40, Some(-130.43)));
    }
}
