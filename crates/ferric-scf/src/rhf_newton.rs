//! Second-order RHF / RKS step: damped Newton via Hessian-vector products.
//!
//! The RESTRICTED closed-shell analogue of `uhf_newton.rs`. Where UHF carries
//! two independent MO sets (α, β) each with its own occ↔virt rotation κ^σ, a
//! closed-shell system has ONE MO set C with a single occupied↔virtual rotation
//! κ_{ai} (a ∈ virt, i ∈ occ). Both spins share C and share the occupation, so
//! the α/β bookkeeping of `uhf_newton.rs` collapses to one block plus explicit
//! factor-2 restricted-density bookkeeping.
//!
//! The gradient is the occupied→virtual block of the MO-basis Fock,
//!   g_{ai} = F_{ai}   (a ∈ virt, i ∈ occ),
//! which is exactly the RHF Brillouin condition (F_{ai} = 0 at convergence).
//! (Sign/scale note: the restricted orbital gradient is 4·F_ai; the constant
//! prefactor cancels between g and the diagonal-gap Hessian term below — the
//! Newton step κ = −g/(gap) and the full PCG solve are prefactor-invariant, so
//! we use g_ai = F_ai and gap = F_aa − F_ii consistently, matching the UHF path.)
//!
//! The Hessian matvec builds the AO restricted density perturbation
//!   δD = 2 · (C δD_MO Cᵀ),   δD_MO[a,i] = δD_MO[i,a] = κ_{ai}
//! (the factor 2 is the closed-shell double occupation, matching D = 2·C_occ·C_occᵀ),
//! rebuilds J on δD and K on δD, and — for KS — adds the XC-kernel response using
//! the SAME `FxcKernelStore`/`FxcResponse` closure the ROKS/UKS paths use, called
//! at the restricted point δD_α = δD_β = ½·δD (each spin carries half the total
//! restricted density perturbation). It then forms
//!   δF = δJ − ½·k_mix·δK + δV_xc,
//! projects δF to MO basis, reads the occ→virt block, and adds the diagonal
//! orbital-energy-gap term (F_aa − F_ii)·κ_ai.
//!
//! Rotation is applied via the Cayley unitary U = (I − κ/2)^{−1}(I + κ/2), which
//! exactly preserves orthonormality (identical to `uhf_newton.rs`).

use crate::engine_pool::EnginePool;
use crate::rhf::build_jk_with_pool;
use crate::rohf_newton::FxcResponse;
use crate::screening::SchwarzBounds;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;
use ndarray_linalg::Solve;

/// Inputs to one RHF/RKS Newton step.
pub struct RhfNewtonInputs<'a> {
    pub prep: &'a PreparedBasis,
    pub bounds: &'a SchwarzBounds,
    /// MO coeffs (n × n), columns ordered occ | virt.
    pub c: &'a Array2<f64>,
    /// Fock in the MO basis (current iter).
    pub f_mo: &'a Array2<f64>,
    pub nocc: usize,
    /// K mixing coefficient (1.0 for HF, c_HF for hybrid; ignored for RSH).
    pub k_mix_sr: f64,
    /// Optional XC-kernel response closure (None for pure RHF). Called with the
    /// per-spin restricted density perturbation (δD_α = δD_β = ½·δD_total).
    pub fxc: Option<&'a FxcResponse<'a>>,
    pub thresh: f64,
}

/// One damped-Newton step on RHF/RKS MO coefficients.
///
/// Returns `(C_new, kmax)` where `kmax` is the ∞-norm of the rotation (a
/// step-size diagnostic). `level_shift` is added to the diagonal preconditioner
/// gap; `max_step` is the componentwise trust radius on κ.
pub fn rhf_newton_step(
    ctx: &ParallelContext,
    inp: &RhfNewtonInputs,
    level_shift: f64,
    max_step: f64,
    cg_max_iter: usize,
    cg_conv: f64,
) -> Result<(Array2<f64>, f64), FerricError> {
    let n = inp.c.nrows();
    let no = inp.nocc;

    // EnginePool is geometry/basis-only (density-independent) — build ONCE
    // here and reuse across every hessian_matvec call in the PCG loop below,
    // instead of build_jk constructing a fresh pool per call. Reduction order
    // (grouped_deterministic_sum, inside build_jk_with_pool) is unchanged, so
    // results stay bit-identical across thread counts.
    let pool = EnginePool::new(inp.bounds.op, inp.prep, 1e-14)?;

    // Gradient g_{ai} = F_{ai}  (rows = virt, cols = occ).
    let g = occ_virt_block(inp.f_mo, no, n);

    // Diagonal preconditioner gap (F[a,a] − F[i,i]) + shift.
    let f_diag: Vec<f64> = (0..n).map(|i| inp.f_mo[(i, i)]).collect();
    let diag = build_gap(&f_diag, no, n, level_shift);

    // Initial guess: diagonal solve κ⁰ = −g / gap.
    let mut k = neg_div(&g, &diag);

    // Early-exit if the gradient is already below tolerance.
    let gmax = max_abs(&g);
    let cg_iters = if gmax < cg_conv { 0 } else { cg_max_iter };

    // Preconditioned conjugate-gradient solve for H·κ = −g.
    let h0 = hessian_matvec(ctx, inp, &k, &pool)?;
    let mut r = &neg(&g) - &h0;
    let mut z = &r / &diag;
    let mut p = z.clone();
    let mut rz_old = inner(&r, &z);

    for _it in 0..cg_iters {
        if max_abs(&r) < cg_conv {
            break;
        }
        let ap = hessian_matvec(ctx, inp, &p, &pool)?;
        let p_ap = inner(&p, &ap);
        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rz_old / p_ap;
        k.zip_mut_with(&p, |k, &v| *k += alpha * v);
        r.zip_mut_with(&ap, |r, &v| *r -= alpha * v);
        z = &r / &diag;
        let rz_new = inner(&r, &z);
        let beta = rz_new / rz_old;
        rz_old = rz_new;
        p.zip_mut_with(&z, |p, &v| *p = v + beta * *p);
    }

    // Trust-radius clip.
    let kmax = max_abs(&k);
    if kmax > max_step {
        let s = max_step / kmax;
        k.mapv_inplace(|x| x * s);
    }

    let c_new = apply_cayley(inp.c, &k, no, n)?;
    Ok((c_new, kmax))
}

/// H · κ for the occ→virt block. Public so tests can finite-difference-validate
/// the matvec directly against the orbital-gradient derivative.
pub fn hessian_matvec(
    ctx: &ParallelContext,
    inp: &RhfNewtonInputs,
    k: &Array2<f64>,
    pool: &EnginePool,
) -> Result<Array2<f64>, FerricError> {
    let n = inp.c.nrows();
    let no = inp.nocc;

    // Single-set MO density perturbation, then AO transform.
    // δD_single[a,i] = δD_single[i,a] = κ[a,i]; δD_single_ao = C·δD_MO·Cᵀ.
    let dd_single_ao = ao_from_ov(inp.c, k, no, n);
    // Restricted (closed-shell) total density perturbation: D = 2·D_closed
    // ⇒ δD = 2·δD_single. J and K in the RHF Fock are built on this total D.
    let dd_ao = 2.0 * &dd_single_ao;

    // δJ and δK on the total restricted density perturbation.
    let mut dj = Array2::<f64>::zeros((n, n));
    let mut dk = Array2::<f64>::zeros((n, n));
    build_jk_with_pool(ctx, inp.prep, inp.bounds, inp.thresh, &dd_ao, &mut dj, &mut dk, pool)?;

    // F = H + J − ½·k_mix·K  ⇒  δF = δJ − ½·k_mix·δK.
    let c_k = inp.k_mix_sr;
    let mut df: Array2<f64> = &dj - &(0.5 * c_k * &dk);

    if let Some(fxc) = inp.fxc {
        // The f_xc response expects the per-spin perturbation. For a restricted
        // point δD_α = δD_β = ½·δD_total = δD_single. Both spin outputs are
        // identical here; add one spin's δV_xc (the closed-shell restricted
        // response, matching V_xc appearing once in the restricted Fock).
        let (dvxc, _dvxc_b) = fxc(&dd_single_ao, &dd_single_ao);
        df = &df + &dvxc;
    }

    // Project to MO basis and read the occ→virt block.
    let df_mo = inp.c.t().dot(&df).dot(inp.c);
    let mut h = occ_virt_block(&df_mo, no, n);

    // Diagonal orbital-energy-gap term: + (F[a,a] − F[i,i]) · κ[a,i].
    let f_diag: Vec<f64> = (0..n).map(|i| inp.f_mo[(i, i)]).collect();
    for (ir, a) in (no..n).enumerate() {
        for (ic, i) in (0..no).enumerate() {
            h[(ir, ic)] += (f_diag[a] - f_diag[i]) * k[(ir, ic)];
        }
    }

    Ok(h)
}

// ---- helpers (mirror uhf_newton.rs) ----

/// The occ→virt block (rows = virt [nocc..n], cols = occ [0..nocc]) of a
/// square MO matrix.
fn occ_virt_block(m: &Array2<f64>, nocc: usize, n: usize) -> Array2<f64> {
    let nv = n - nocc;
    let mut out = Array2::<f64>::zeros((nv, nocc));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            out[(ir, i)] = m[(a, i)];
        }
    }
    out
}

/// AO single-set δD from an occ→virt block: δD^MO symmetric with the block in
/// both (virt,occ) and (occ,virt) corners, then C·δD^MO·Cᵀ.
fn ao_from_ov(c: &Array2<f64>, k_ov: &Array2<f64>, nocc: usize, n: usize) -> Array2<f64> {
    let mut dd_mo = Array2::<f64>::zeros((n, n));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            let v = k_ov[(ir, i)];
            dd_mo[(a, i)] = v;
            dd_mo[(i, a)] = v;
        }
    }
    c.dot(&dd_mo).dot(&c.t())
}

/// Diagonal gap (F[a,a] − F[i,i]) + shift, floored at 1e-6 for near-degeneracy.
fn build_gap(f_diag: &[f64], nocc: usize, n: usize, shift: f64) -> Array2<f64> {
    let nv = n - nocc;
    let mut out = Array2::<f64>::zeros((nv, nocc));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            let d = (f_diag[a] - f_diag[i]) + shift;
            out[(ir, i)] = if d.abs() < 1e-6 { 1e-6 } else { d };
        }
    }
    out
}

/// Apply the Cayley unitary built from the occ→virt block to C: C ← C·U.
fn apply_cayley(
    c: &Array2<f64>,
    k_ov: &Array2<f64>,
    nocc: usize,
    n: usize,
) -> Result<Array2<f64>, FerricError> {
    // Antisymmetric κ in MO basis from the occ→virt block.
    let mut kappa = Array2::<f64>::zeros((n, n));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            let v = k_ov[(ir, i)];
            kappa[(a, i)] = v;
            kappa[(i, a)] = -v;
        }
    }
    let half = 0.5 * &kappa;
    let eye = Array2::<f64>::eye(n);
    let a_mat = &eye - &half;
    let b_mat = &eye + &half;
    let mut u = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let bcol = b_mat.column(col).to_owned();
        let sol = a_mat
            .solve(&bcol)
            .map_err(|e| FerricError::Lapack(format!("RHF Cayley solve: {e}")))?;
        for row in 0..n {
            u[(row, col)] = sol[row];
        }
    }
    Ok(c.dot(&u))
}

fn neg(a: &Array2<f64>) -> Array2<f64> {
    -a
}
fn neg_div(num: &Array2<f64>, den: &Array2<f64>) -> Array2<f64> {
    -num / den
}
fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, &v| m.max(v.abs()))
}
fn inner(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
