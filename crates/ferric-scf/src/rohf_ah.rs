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
use ndarray_linalg::Solve;

thread_local! {
    /// AH steps on this thread whose Davidson did not converge (test seam).
    static AH_FALLBACKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Lowest-root residual below which an unconverged AH Davidson is the
/// expected tight-tolerance stall and the PCG Newton fallback is silent (see
/// `rohf_ah_step`). Chosen, not derived: it is the AH tolerance this solver
/// used before it followed `density_conv` (a residual below it was accepted
/// as converged), and 14× the stall measured at density_conv 1e-10.
const AH_ENDGAME_RESIDUAL: f64 = 1e-7;

/// Number of AH steps on the calling thread that fell back to the PCG Newton
/// step because the augmented-Hessian Davidson did not converge (test seam).
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
/// stall), with a warning otherwise.
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
    // An unconverged eigenvector is not used: take the PCG Newton step (same
    // inputs, exact Hessian, Steihaug-guarded) from these orbitals instead.
    //
    // At a tight `density_conv` that is the normal endgame. MEASURED
    // 2026-11-11, OH/6-31G ROHF at density_conv 1e-10: the last AH Davidson
    // stalled at residual 6.9e-9 against a tolerance of 1e-10 (λ = −2.8e-13).
    // The stall comes from `davidson_lowest`'s expansion step, not from a
    // fixed floor: t = r/(diag − θ) is discarded when its Gram-Schmidt
    // remainder has norm ≤ 1e-8, and on the augmented coordinate diag − θ =
    // −λ ≈ 0 is clamped to 1e-8, so r₀ is amplified ~1e8 into the seed
    // direction that Gram-Schmidt then removes. Near convergence AH and
    // Newton steps agree to O(λ/h). So the fallback is silent only when the
    // residual reached is that small (< AH_ENDGAME_RESIDUAL); a Davidson
    // stalled anywhere above it is a real failure and warns on every step.
    if !result.converged {
        AH_FALLBACKS.with(|c| c.set(c.get() + 1));
        let endgame = result.residual < AH_ENDGAME_RESIDUAL;
        if !endgame {
            eprintln!(
                "[ferric] WARNING: ROHF AH Davidson did not converge (residual {:.2e} > {:.2e} after {} iterations); this iteration takes the PCG Newton step instead of AH",
                result.residual, cfg.conv_thresh, result.iterations
            );
        } else if crate::rohf::rohf_trace() {
            eprintln!(
                "ROHF AH: endgame stall (residual {:.2e}, tol {:.2e}); PCG Newton step",
                result.residual, cfg.conv_thresh
            );
        }
        return crate::rohf_newton::rohf_newton_step(
            ctx,
            base,
            inp.newton_level_shift,
            inp.newton_max_step,
            20,
            davidson_conv,
        );
    }

    // 4. Extract κ from the lowest eigenvector. Rescale so the leading
    // entry is 1 — this realizes the AH ansatz.
    let evec = &result.eigenvector;
    let first = evec[0];
    if first.abs() < 1e-10 {
        return Err(FerricError::General(
            "AH eigenvector has near-zero leading entry — degeneracy or wrong sign".into(),
        ));
    }
    let kappa_flat: Array1<f64> = evec[1..].iter().map(|x| x / first).collect();

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
    let half_k = 0.5 * &kappa_mo;
    let i_eye = Array2::<f64>::eye(n);
    let a = &i_eye - &half_k;
    let b = &i_eye + &half_k;
    let mut u = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let bcol = b.column(col).to_owned();
        let sol = a
            .solve(&bcol)
            .map_err(|e| FerricError::Lapack(format!("AH Cayley solve: {e}")))?;
        for row in 0..n {
            u[(row, col)] = sol[row];
        }
    }
    let c_new = base.c.dot(&u);

    Ok((c_new, kmax))
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
