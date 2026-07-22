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

use crate::davidson_local::run_davidson_seeded;
use crate::engine_pool::EnginePool;
use crate::rohf_newton::{gradient_blocks, hessian_matvec, RohfNewtonInputs};
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

/// Inputs to one AH Newton step. Wraps `RohfNewtonInputs` directly — the AH
/// driver consumes exactly the same inputs as the PCG driver because the
/// orbital Hessian matvec is shared.
pub struct RohfAhInputs<'a, 'b> {
    pub base: &'a RohfNewtonInputs<'b>,
}

/// One augmented-Hessian Newton step on ROHF/ROKS MO coefficients.
///
/// Returns updated C and the κ ∞-norm (step-size diagnostic). The trust
/// region is enforced by componentwise clipping after Davidson; this is
/// simpler than the full Bacskay-Hendrickson "shift λ until ‖κ‖ ≤ Δ"
/// scheme and works well in practice because Davidson is computing the
/// optimal step direction.
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
        for &v in g_vc.iter() { g_flat[idx] = v; idx += 1; }
        for &v in g_vo.iter() { g_flat[idx] = v; idx += 1; }
        for &v in g_oc.iter() { g_flat[idx] = v; idx += 1; }
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
    let pool = EnginePool::new(base.bounds.op, base.prep, 1e-14)?;

    // 2. Build the augmented matvec closure. Davidson expects the closure
    // to return V^T A V given V (the trial subspace).
    let nc_local = nc;
    let no_local = no;
    let aug_matvec = |v: &Array2<f64>, _omega: f64| -> Array2<f64> {
        let m = v.ncols();
        let mut av = Array2::<f64>::zeros((n_aug, m));
        for col in 0..m {
            let v_col = v.column(col);
            let v_first = v_col[0];
            let mut k_vc_local = Array2::<f64>::zeros((nvirt, nc_local));
            let mut k_vo_local = Array2::<f64>::zeros((nvirt, no_local));
            let mut k_oc_local = Array2::<f64>::zeros((no_local, nc_local));
            unpack_three(v_col.slice(ndarray::s![1..]), &mut k_vc_local, &mut k_vo_local, &mut k_oc_local);

            let (h_vc, h_vo, h_oc) = hessian_matvec(
                ctx, base, &k_vc_local, &k_vo_local, &k_oc_local, &pool,
            ).expect("hessian_matvec failed inside Davidson closure");

            // av[0] = g · κ
            let mut gk: f64 = 0.0;
            {
                let mut idx = 0;
                for &vv in k_vc_local.iter() { gk += g_flat[idx] * vv; idx += 1; }
                for &vv in k_vo_local.iter() { gk += g_flat[idx] * vv; idx += 1; }
                for &vv in k_oc_local.iter() { gk += g_flat[idx] * vv; idx += 1; }
            }
            av[(0, col)] = gk;

            // av[1..] = g · v_first + H · κ (flattened)
            let mut idx = 0;
            for &vv in h_vc.iter() {
                av[(1 + idx, col)] = g_flat[idx] * v_first + vv;
                idx += 1;
            }
            for &vv in h_vo.iter() {
                av[(1 + idx, col)] = g_flat[idx] * v_first + vv;
                idx += 1;
            }
            for &vv in h_oc.iter() {
                av[(1 + idx, col)] = g_flat[idx] * v_first + vv;
                idx += 1;
            }
        }
        v.t().dot(&av)
    };

    // 3. Davidson seed: identity column [1, 0, …]^T is the natural seed
    // because the lowest eigenvector of the AH matrix has a large first
    // component (we'll rescale by it).
    let mut seed = Array2::<f64>::zeros((n_aug, 1));
    seed[(0, 0)] = 1.0;

    let result = run_davidson_seeded(
        seed,
        aug_matvec,
        davidson_conv,
        davidson_max_vecs,
        1,
        /* find_lowest = */ true,
    )?;

    // 4. Extract κ from the lowest eigenvector. Rescale so the leading
    // entry is 1 — this realizes the AH ansatz.
    let evec = result.eigenvectors.column(0);
    let first = evec[0];
    if first.abs() < 1e-10 {
        return Err(FerricError::General(
            "AH eigenvector has near-zero leading entry — degeneracy or wrong sign".into()
        ));
    }
    let kappa_flat: Array1<f64> = evec.slice(ndarray::s![1..]).mapv(|x| x / first);

    let mut k_vc = Array2::<f64>::zeros((nvirt, nc));
    let mut k_vo = Array2::<f64>::zeros((nvirt, no));
    let mut k_oc = Array2::<f64>::zeros((no, nc));
    unpack_three(kappa_flat.view(), &mut k_vc, &mut k_vo, &mut k_oc);

    // 5. Trust-region clip.
    let kmax = arr_max_abs(&k_vc).max(arr_max_abs(&k_vo)).max(arr_max_abs(&k_oc));
    let scale = if kmax > max_step { max_step / kmax } else { 1.0 };
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
        let sol = a.solve(&bcol)
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
