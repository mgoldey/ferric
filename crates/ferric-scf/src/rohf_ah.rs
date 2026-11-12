//! Augmented-Hessian Newton step for ROHF / ROKS.
//!
//! Reformulates H κ = −g as a (n+1)×(n+1) generalized eigenvalue problem:
//!
//!   ⎡ 0    g^T ⎤ ⎡ 1 ⎤        ⎡ 1 ⎤
//!   ⎢          ⎥ ⎢   ⎥  =  λ  ⎢   ⎥
//!   ⎣ g    H   ⎦ ⎣ κ ⎦        ⎣ κ ⎦
//!
//! The lowest eigenvalue λ gives the optimal step length implicitly; the
//! eigenvector's lower block — after rescaling so the leading entry is 1
//! — gives κ. AH handles vanishing or negative Hessian eigenvalues
//! gracefully (the augmented matrix's lowest eigenvalue can be ≤ 0),
//! avoiding PCG divergence on near-degenerate open-shell ROHF/ROKS
//! problems (e.g., doublet OH at LDA where the SOMO/HOMO α-α gap vanishes).

use crate::engine_pool::EnginePool;
use crate::rohf_newton::{gradient_blocks, hessian_matvec, RohfNewtonInputs};
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ndarray::{Array1, Array2};

thread_local! {
    /// AH steps on this thread that took the PCG Newton step instead (test
    /// seam; see `ah_kappa_or_fallback`).
    static AH_FALLBACKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Lowest-root residual below which an unconverged AH Davidson is the
/// expected tight-tolerance stall and the PCG Newton fallback is silent (see
/// `rohf_ah_step`). Chosen, not derived: it is the AH tolerance this solver
/// used before it followed `density_conv` (a residual below it was accepted
/// as converged), and 14× the stall measured at density_conv 1e-10.
const AH_ENDGAME_RESIDUAL: f64 = 1e-7;

/// Number of AH steps on the calling thread that fell back to the PCG Newton
/// step: the augmented-Hessian Davidson did not converge, or its lowest
/// eigenvector has a vanishing leading entry (test seam).
#[doc(hidden)]
pub fn ah_fallbacks_on_this_thread() -> usize {
    AH_FALLBACKS.with(std::cell::Cell::get)
}

/// Inputs to one AH Newton step. Wraps `RohfNewtonInputs` directly — the AH
/// driver consumes exactly the same inputs as the PCG driver because the
/// orbital Hessian matvec is shared.
pub struct RohfAhInputs<'a, 'b> {
    /// The inputs of the PCG Newton step, used unchanged.
    pub base: &'a RohfNewtonInputs<'b>,
    /// Preconditioner level shift of the PCG Newton fallback step (the
    /// solver passes its Newton branch's, `max(level_shift, 1e-6)`).
    pub newton_level_shift: f64,
    /// Trust radius of the PCG Newton fallback step (the solver passes its
    /// Newton branch's).
    pub newton_max_step: f64,
}

/// One augmented-Hessian Newton step on ROHF/ROKS MO coefficients.
///
/// Returns updated C and the κ ∞-norm (step-size diagnostic). Below
/// `|g|max < davidson_conv` the step is zero. If the Davidson solve does not
/// converge, the PCG Newton step ([`crate::rohf_newton::rohf_newton_step`],
/// with `inp.newton_level_shift` / `inp.newton_max_step`) is taken instead:
/// silently when the lowest-root residual is below 1e-7 (the tight-tolerance
/// stall), with a warning otherwise. The same step, with a warning, is taken
/// when the lowest eigenvector's leading entry vanishes (a Hessian mode below
/// the gradient-coupled root and orthogonal to g), where κ is undefined.
///
/// # Step-length control: clipping, and what it is not
///
/// The step length is bounded by **componentwise clipping** after Davidson:
/// the AH eigenproblem is solved once at α = 1 and the resulting κ is scaled
/// down if its ∞-norm exceeds `max_step`. This is NOT the Bacskay-Hendrickson
/// / Helmich-Paris "shift μ until ‖κ‖ ≤ Δ" scheme — a clipped step is a
/// *shortened* AH step, not the solution of the level-shifted Newton equations
/// at the boundary, and the two differ in direction, not just length.
///
/// The genuine level-shifted trust region lives in [`crate::trah`], which
/// searches the AH scale factor α so the step lands ON the radius, forms the
/// predicted-vs-actual ratio ρ, and can REJECT a step. It is wired into
/// RHF/RKS and UHF/UKS.
///
/// This molecular AH branch is not on that path. The reason once given here —
/// that the three-block ROHF packing (`vc` α+β, `vo` α-only, `oc` β-only; see
/// `crate::rohf_newton::gradient_blocks`) admits no single scalar between the
/// packed and the true orbital gradient — is FALSE: the true gradient is
/// exactly **2 ×** `gradient_blocks` in every block (FD-measured in
/// `reference/pbc/pbc_rohf_newton.py`, FINDINGS "Iteration 24"; pinned by
/// `crate::rohf_trah`'s tests). The Hessian used here,
/// `rohf_newton::hessian_matvec`, is the exact orbital Hessian (half
/// scale; the per-spin diagonal Fock gap survives only as the Davidson
/// preconditioner), but this is a clipped AH step with no ρ test.
/// [`crate::rohf_trah`] is the ROHF/ROKS trust region on the exact gradient
/// and Hessian; it currently serves the injected (periodic) path. This
/// function remains the clipped AH step of the molecular solver, taken from
/// the current orbitals.
pub fn rohf_ah_step(
    ctx: &ParallelContext,
    inp: &RohfAhInputs,
    max_step: f64,
    davidson_conv: f64,
    davidson_max_vecs: usize,
) -> Result<(Array2<f64>, f64), FerricError> {
    let base = inp.base;
    let n = base.c.nrows();
    let nc = base.nocc_double;
    let no = base.nocc_open;
    let nocc_a = nc + no;
    let nvirt = n - nocc_a;

    // 1. Build gradient blocks.
    let (g_vc, g_vo, g_oc) = gradient_blocks(base);
    let nvc = g_vc.len();
    let nvo = g_vo.len();
    let noc = g_oc.len();
    let n_kappa = nvc + nvo + noc;
    let n_aug = 1 + n_kappa;

    // Pack g into a flat (n_kappa,) vector for the augmented matvec.
    let mut g_flat = Array1::<f64>::zeros(n_kappa);
    {
        let mut idx = 0;
        for &v in g_vc.iter() {
            g_flat[idx] = v;
            idx += 1;
        }
        for &v in g_vo.iter() {
            g_flat[idx] = v;
            idx += 1;
        }
        for &v in g_oc.iter() {
            g_flat[idx] = v;
            idx += 1;
        }
    }

    // Early exit if already at a stationary point — the augmented matrix
    // becomes singular and the lowest eigenvector is not [1, 0, …]^T.
    let gnorm = g_flat.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if gnorm < davidson_conv {
        return Ok((base.c.clone(), 0.0));
    }

    // EnginePool is geometry/basis-only (density-independent) — build ONCE
    // here and reuse across every hessian_matvec call inside the Davidson
    // closure below (called repeatedly per column, per Davidson iteration),
    // instead of each call constructing its own pool.
    let pool = EnginePool::new(
        base.bounds.op,
        base.prep,
        ferric_integrals::engine_pool::eri_precision(),
    )?;

    // 2. The augmented matvec on full vectors `[v₀, κ] ↦ [g·κ, g v₀ + H κ]`.
    //
    // `stability::davidson_lowest` forms the true residual from Hv. A
    // projected-matrix Davidson (closure returning only VᵀAV) would not
    // work here: its residual V(VᵀAV)y − λVy is zero inside the subspace by
    // construction, so it "converges" on the seed [1, 0, …] and every AH
    // step is κ = 0.
    let aug_matvec = |v: &[f64]| -> Result<Vec<f64>, FerricError> {
        let v_first = v[0];
        let mut k_vc_local = Array2::<f64>::zeros((nvirt, nc));
        let mut k_vo_local = Array2::<f64>::zeros((nvirt, no));
        let mut k_oc_local = Array2::<f64>::zeros((no, nc));
        unpack_three(
            ndarray::ArrayView1::from(&v[1..]),
            &mut k_vc_local,
            &mut k_vo_local,
            &mut k_oc_local,
        );
        let (h_vc, h_vo, h_oc) =
            hessian_matvec(ctx, base, &k_vc_local, &k_vo_local, &k_oc_local, &pool)?;
        let mut out = Vec::with_capacity(n_aug);
        out.push(g_flat.iter().zip(&v[1..]).map(|(g, k)| g * k).sum());
        for (idx, hk) in h_vc
            .iter()
            .chain(h_vo.iter())
            .chain(h_oc.iter())
            .enumerate()
        {
            out.push(g_flat[idx] * v_first + hk);
        }
        Ok(out)
    };

    // 3. Diagonal for the Davidson preconditioner and block seed: 0 for the
    // augmented coordinate, the per-spin Fock gaps (as the PCG Newton step's
    // preconditioner) for κ.
    // Half scale (this module's), same packing order as κ.
    let diag: Vec<f64> = std::iter::once(0.0)
        .chain(
            crate::rohf_trah::rohf_hessian_diagonal(base.f_a_mo, base.f_b_mo, nc, no)
                .into_iter()
                .map(|d| 0.5 * d),
        )
        .collect();
    // One root is wanted. n_block = 2 is the minimum `davidson_lowest`'s
    // correctness rests on (`StabilityConfig::n_block`); n_seed = 2. The
    // residual tolerance follows the step's needs, as in `rohf_trah`:
    // max(davidson_conv, 1e-2 · min(1, ‖g‖) · ‖g‖) (‖g‖ the 2-norm of the
    // half gradient), so a far-from-converged step is not solved to 1e-7.
    let g2 = g_flat.dot(&g_flat).sqrt();
    let cfg = crate::stability::StabilityConfig {
        conv_thresh: davidson_conv.max(1e-2 * g2.min(1.0) * g2),
        max_subspace: davidson_max_vecs,
        n_seed: 2,
        n_block: 2,
        ..Default::default()
    };
    let result = crate::stability::davidson_lowest(n_aug, aug_matvec, &diag, &cfg)?;
    if crate::rohf::rohf_trace() {
        eprintln!(
            "ROHF AH: Davidson converged={} iterations={} residual={:.2e} (tol {:.2e}) lambda={:.3e}",
            result.converged, result.iterations, result.residual, cfg.conv_thresh, result.eigenvalue
        );
    }
    // 4. κ from the lowest eigenvector, a step along the negative mode the
    // gradient does not see, or the PCG Newton step (same inputs, exact
    // Hessian, Steihaug-guarded) from these orbitals when the eigenvector
    // cannot be used (`ah_kappa_or_fallback`).
    let kappa_flat = match ah_kappa_or_fallback(&result) {
        AhKappa::Step(k) => k,
        AhKappa::NegativeMode(v) => {
            AH_NEGATIVE_MODE_STEPS.with(|c| c.set(c.get() + 1));
            eprintln!(
                "[ferric] WARNING: ROHF AH lowest eigenvector has leading entry {:.2e} (Hessian eigenvalue {:.3e} along a mode orthogonal to the gradient); this iteration steps along that mode to the step bound",
                result.eigenvector[0], result.eigenvalue
            );
            negative_mode_step(&v, g_flat.view(), max_step)
        }
        AhKappa::Fallback(reason) => {
            AH_FALLBACKS.with(|c| c.set(c.get() + 1));
            report_ah_fallback(&reason, &result, cfg.conv_thresh);
            return crate::rohf_newton::rohf_newton_step(
                ctx,
                base,
                inp.newton_level_shift,
                inp.newton_max_step,
                20,
                davidson_conv,
            );
        }
    };

    let mut k_vc = Array2::<f64>::zeros((nvirt, nc));
    let mut k_vo = Array2::<f64>::zeros((nvirt, no));
    let mut k_oc = Array2::<f64>::zeros((no, nc));
    unpack_three(kappa_flat.view(), &mut k_vc, &mut k_vo, &mut k_oc);

    // 5. Step-length clip (NOT a trust region — see this function's doc).
    //    A uniform scale-down of the α = 1 AH step, not a re-solve at a larger
    //    level shift. `crate::trah` does the latter for RHF/UHF.
    let kmax = arr_max_abs(&k_vc)
        .max(arr_max_abs(&k_vo))
        .max(arr_max_abs(&k_oc));
    let scale = if kmax > max_step {
        max_step / kmax
    } else {
        1.0
    };
    if scale < 1.0 {
        k_vc.mapv_inplace(|x| x * scale);
        k_vo.mapv_inplace(|x| x * scale);
        k_oc.mapv_inplace(|x| x * scale);
    }

    // 6. Assemble antisymmetric κ in MO basis.
    let mut kappa_mo = Array2::<f64>::zeros((n, n));
    for (ir, p) in (nocc_a..n).enumerate() {
        for (ic, q) in (0..nc).enumerate() {
            kappa_mo[(p, q)] = k_vc[(ir, ic)];
            kappa_mo[(q, p)] = -k_vc[(ir, ic)];
        }
    }
    for (ir, p) in (nocc_a..n).enumerate() {
        for (ic, q) in (nc..nocc_a).enumerate() {
            kappa_mo[(p, q)] = k_vo[(ir, ic)];
            kappa_mo[(q, p)] = -k_vo[(ir, ic)];
        }
    }
    for (ir, p) in (nc..nocc_a).enumerate() {
        for (ic, q) in (0..nc).enumerate() {
            kappa_mo[(p, q)] = k_oc[(ir, ic)];
            kappa_mo[(q, p)] = -k_oc[(ir, ic)];
        }
    }

    // 7. Cayley unitary U = (I − κ/2)^{-1} (I + κ/2). Apply to C: C ← C·U.
    let c_new = crate::cayley::cayley_rotate(base.c, &kappa_mo)?;

    Ok((c_new, kmax))
}

/// Why an AH step falls back to the PCG Newton step.
#[derive(Debug, PartialEq)]
enum AhFallback {
    /// The Davidson did not converge. `endgame`: the residual reached is
    /// below [`AH_ENDGAME_RESIDUAL`], the tight-tolerance stall, and the
    /// fallback is silent.
    Unconverged { endgame: bool },
    /// The lowest eigenvector's leading entry is below
    /// [`AH_MIN_LEADING_ENTRY`] in magnitude, so κ = v[1..]/v[0] is undefined,
    /// and its eigenvalue is not negative, so there is no descent mode to
    /// step along.
    ZeroLead,
}

/// κ from the AH eigenproblem, or the reason it cannot be formed.
#[derive(Debug, PartialEq)]
enum AhKappa {
    Step(Array1<f64>),
    /// v of a converged lowest eigenvector [0, v] with a negative eigenvalue:
    /// a Hessian mode with negative curvature that the gradient does not see.
    NegativeMode(Array1<f64>),
    Fallback(AhFallback),
}

thread_local! {
    static AH_NEGATIVE_MODE_STEPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// AH steps taken along a negative mode orthogonal to the gradient on this
/// thread.
pub fn ah_negative_mode_steps_on_this_thread() -> usize {
    AH_NEGATIVE_MODE_STEPS.with(|c| c.get())
}

/// κ along the negative mode `v`, signed so gᵀκ ≤ 0 and scaled so its largest
/// component is `max_step`. The quadratic model changes by gᵀκ + ½λ|κ|², with
/// λ < 0 the mode's curvature and gᵀv ≈ 0 by construction, so the step lowers
/// the model by about ½|λ||κ|². Without this step the PCG Newton step, which
/// starts from −M⁻¹g with a diagonal preconditioner, can leave v unstepped,
/// and the SCF can converge onto the saddle.
fn negative_mode_step(v: &Array1<f64>, g: ndarray::ArrayView1<f64>, max_step: f64) -> Array1<f64> {
    let vmax = v.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    if vmax == 0.0 {
        return Array1::zeros(v.len());
    }
    let sign = if g.dot(v) > 0.0 { -1.0 } else { 1.0 };
    v.mapv(|x| sign * max_step * x / vmax)
}

/// Smallest |v₀| of the lowest augmented-Hessian eigenvector from which κ is
/// formed.
const AH_MIN_LEADING_ENTRY: f64 = 1e-10;

/// Decide between the AH step and the PCG Newton fallback from the Davidson
/// result `r` of the augmented eigenproblem.
///
/// An unconverged eigenvector is not used. At a tight `density_conv` that is
/// the normal endgame. MEASURED 2026-11-11, OH/6-31G ROHF at density_conv
/// 1e-10: the last AH Davidson stalled at residual 6.9e-9 against a tolerance
/// of 1e-10 (λ = −2.8e-13). The stall comes from `davidson_lowest`'s expansion
/// step, not from a fixed floor: t = r/(diag − θ) is discarded when its
/// Gram-Schmidt remainder has norm ≤ 1e-8, and on the augmented coordinate
/// diag − θ = −λ ≈ 0 is clamped to 1e-8, so r₀ is amplified ~1e8 into the
/// seed direction that Gram-Schmidt then removes. Near convergence AH and
/// Newton steps agree to O(λ/h). So the fallback is silent only when the
/// residual reached is that small (< [`AH_ENDGAME_RESIDUAL`]); a Davidson
/// stalled anywhere above it is a failure and warns on every step.
///
/// A converged eigenvector with a vanishing leading entry is [0, v] with v a
/// Hessian eigenvector orthogonal to g, whose eigenvalue lies below the lowest
/// root coupled to g. The AH ansatz (leading entry 1) has no solution there.
/// With a negative eigenvalue, v is a negative mode the gradient does not
/// see, and the step goes along it ([`negative_mode_step`]); otherwise the PCG
/// Newton step is taken.
fn ah_kappa_or_fallback(r: &crate::stability::LowestEigenpair) -> AhKappa {
    if !r.converged {
        return AhKappa::Fallback(AhFallback::Unconverged {
            endgame: r.residual < AH_ENDGAME_RESIDUAL,
        });
    }
    let first = r.eigenvector[0];
    if first.abs() < AH_MIN_LEADING_ENTRY {
        if r.eigenvalue < 0.0 {
            return AhKappa::NegativeMode(r.eigenvector[1..].iter().copied().collect());
        }
        return AhKappa::Fallback(AhFallback::ZeroLead);
    }
    // Rescale so the leading entry is 1: this realizes the AH ansatz.
    AhKappa::Step(r.eigenvector[1..].iter().map(|x| x / first).collect())
}

/// Warn about (or, for the endgame stall, trace) an AH step that falls back
/// to the PCG Newton step.
fn report_ah_fallback(
    reason: &AhFallback,
    r: &crate::stability::LowestEigenpair,
    conv_thresh: f64,
) {
    match reason {
        AhFallback::Unconverged { endgame: false } => eprintln!(
            "[ferric] WARNING: ROHF AH Davidson did not converge (residual {:.2e} > {:.2e} after {} iterations); this iteration takes the PCG Newton step instead of AH",
            r.residual, conv_thresh, r.iterations
        ),
        AhFallback::Unconverged { endgame: true } => {
            if crate::rohf::rohf_trace() {
                eprintln!(
                    "ROHF AH: endgame stall (residual {:.2e}, tol {conv_thresh:.2e}); PCG Newton step",
                    r.residual
                );
            }
        }
        AhFallback::ZeroLead => eprintln!(
            "[ferric] WARNING: ROHF AH lowest eigenvector has leading entry {:.2e} (eigenvalue {:.3e}, no descent mode); this iteration takes the PCG Newton step instead of AH",
            r.eigenvector[0], r.eigenvalue
        ),
    }
}

fn unpack_three(
    flat: ndarray::ArrayView1<f64>,
    vc: &mut Array2<f64>,
    vo: &mut Array2<f64>,
    oc: &mut Array2<f64>,
) {
    let mut idx = 0;
    for v in vc.iter_mut() {
        *v = flat[idx];
        idx += 1;
    }
    for v in vo.iter_mut() {
        *v = flat[idx];
        idx += 1;
    }
    for v in oc.iter_mut() {
        *v = flat[idx];
        idx += 1;
    }
}

fn arr_max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, &v| m.max(v.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stability::LowestEigenpair;

    fn pair(eigenvector: Vec<f64>, converged: bool, residual: f64) -> LowestEigenpair {
        pair_at(-0.1, eigenvector, converged, residual)
    }

    fn pair_at(
        eigenvalue: f64,
        eigenvector: Vec<f64>,
        converged: bool,
        residual: f64,
    ) -> LowestEigenpair {
        LowestEigenpair {
            eigenvalue,
            eigenvector,
            residual,
            iterations: 5,
            converged,
        }
    }

    /// A converged eigenvector [0, v] with a negative eigenvalue (a Hessian
    /// mode orthogonal to g below the gradient-coupled root) gives the
    /// negative mode v instead of dividing by the vanishing leading entry;
    /// with a non-negative eigenvalue it falls back to PCG Newton; a usable
    /// one gives κ = v[1..]/v[0]; an unconverged one falls back, silently
    /// only below the endgame residual.
    #[test]
    fn ah_kappa_falls_back_on_a_vanishing_leading_entry() {
        assert_eq!(
            ah_kappa_or_fallback(&pair(vec![1e-12, 0.6, -0.8], true, 1e-9)),
            AhKappa::NegativeMode(Array1::from(vec![0.6, -0.8]))
        );
        assert_eq!(
            ah_kappa_or_fallback(&pair(vec![-0.0, 1.0, 0.0], true, 1e-9)),
            AhKappa::NegativeMode(Array1::from(vec![1.0, 0.0]))
        );
        assert_eq!(
            ah_kappa_or_fallback(&pair_at(0.0, vec![0.0, 1.0, 0.0], true, 1e-9)),
            AhKappa::Fallback(AhFallback::ZeroLead)
        );
        assert_eq!(
            ah_kappa_or_fallback(&pair(vec![0.5, 0.1, -0.2], true, 1e-9)),
            AhKappa::Step(Array1::from(vec![0.2, -0.4]))
        );
        assert_eq!(
            ah_kappa_or_fallback(&pair(vec![0.5, 0.1, -0.2], false, 1e-9)),
            AhKappa::Fallback(AhFallback::Unconverged { endgame: true })
        );
        assert_eq!(
            ah_kappa_or_fallback(&pair(vec![0.5, 0.1, -0.2], false, 1e-5)),
            AhKappa::Fallback(AhFallback::Unconverged { endgame: false })
        );
    }

    /// The review's case: H = diag(−1, 1), g = (0, 1e-4). The lowest AH
    /// eigenvector is [0, ±1, 0] (eigenvalue −1). The step goes along the
    /// negative mode to the bound, signed against g, and lowers the
    /// quadratic model m(κ) = gᵀκ + ½κᵀHκ by ½·max_step²; a PCG step from
    /// −M⁻¹g = (0, −1e-4) would leave the first component at zero.
    #[test]
    fn negative_mode_step_goes_along_the_mode_to_the_bound_and_downhill() {
        let h = [-1.0, 1.0];
        let max_step = 0.1;
        let model = |k: &Array1<f64>, g: &Array1<f64>| -> f64 {
            g.dot(k) + 0.5 * (0..2).map(|i| h[i] * k[i] * k[i]).sum::<f64>()
        };
        for (g, v) in [
            (Array1::from(vec![0.0, 1e-4]), Array1::from(vec![1.0, 0.0])),
            (Array1::from(vec![0.0, 1e-4]), Array1::from(vec![-1.0, 0.0])),
            (Array1::from(vec![1e-6, 1e-4]), Array1::from(vec![1.0, 0.0])),
        ] {
            let k = negative_mode_step(&v, g.view(), max_step);
            let kmax = k.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
            assert!(
                (kmax - max_step).abs() <= 1e-15,
                "step {k:?} not at the bound"
            );
            assert!(g.dot(&k) <= 0.0, "step {k:?} not signed against g {g:?}");
            let dm = model(&k, &g);
            assert!(
                dm <= -0.5 * max_step * max_step + max_step * g[0].abs(),
                "model change {dm:.3e} for step {k:?}"
            );
        }
        assert_eq!(
            negative_mode_step(
                &Array1::zeros(2),
                Array1::from(vec![0.0, 1.0]).view(),
                max_step
            ),
            Array1::<f64>::zeros(2)
        );
    }
}
