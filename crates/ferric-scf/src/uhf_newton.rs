//! Second-order UHF / UKS step: damped Newton via Hessian-vector products.
//!
//! The unrestricted analogue of `rohf_newton.rs`. Where ROHF rotates a single
//! set of MOs partitioned into closed/open/virtual (Roothaan coupling), UHF has
//! two INDEPENDENT MO sets — α and β — each with its own occupied↔virtual
//! rotation κ^σ. The two spins are NOT decoupled, however: the Coulomb response
//! δJ depends on the total δD = δD_α + δD_β, and (for KS) the XC kernel carries
//! the f_αβ cross term. So we solve the coupled linear system
//!
//!   H · (κ_α, κ_β) = −(g_α, g_β)
//!
//! for both spin blocks simultaneously with one preconditioned-CG loop.
//!
//! Per spin the gradient is the occupied→virtual block of the MO-basis Fock,
//!   g^σ_{ai} = F^σ_{ai}   (a ∈ virt_σ, i ∈ occ_σ),
//! which is exactly the UHF Brillouin condition (F^σ_{ai} = 0 at convergence).
//!
//! The Hessian matvec builds the AO density perturbation δD^σ from κ^σ, rebuilds
//! J (on δD_total) and K^σ (on δD^σ), adds the optional XC-kernel response
//! (δV_xc^α, δV_xc^β) — the SAME `LdaFxcKernel`/`GgaFxcKernel` closure the ROKS
//! path uses — projects the resulting δF^σ to MO basis, reads the occ→virt
//! block, and adds the diagonal orbital-energy-gap term.
//!
//! Rotation is applied via the Cayley unitary U = (I − κ/2)^{−1}(I + κ/2) per
//! spin, which exactly preserves orthonormality. Sign convention matches
//! `rohf_newton.rs` (κ = −g/(gap+μ); C ← C·U).

use crate::rhf::build_jk;
use crate::rohf_newton::FxcResponse;
use crate::screening::SchwarzBounds;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;
use ndarray_linalg::Solve;

/// Inputs to one UHF/UKS Newton step.
pub struct UhfNewtonInputs<'a> {
    pub prep: &'a PreparedBasis,
    pub bounds: &'a SchwarzBounds,
    /// α MO coeffs (n × n), columns ordered occ | virt.
    pub c_a: &'a Array2<f64>,
    /// β MO coeffs (n × n), columns ordered occ | virt.
    pub c_b: &'a Array2<f64>,
    /// α Fock in the α-MO basis (current iter).
    pub f_a_mo: &'a Array2<f64>,
    /// β Fock in the β-MO basis (current iter).
    pub f_b_mo: &'a Array2<f64>,
    pub nocc_a: usize,
    pub nocc_b: usize,
    /// K mixing coefficient (1.0 for HF, c_HF for hybrid; ignored for RSH).
    pub k_mix_sr: f64,
    /// Optional XC-kernel response closure (None for pure UHF).
    pub fxc: Option<&'a FxcResponse<'a>>,
    pub thresh: f64,
}

/// One damped-Newton step on UHF/UKS MO coefficients.
///
/// Returns `(C_α_new, C_β_new, kmax)` where `kmax` is the ∞-norm of the packed
/// rotation (a step-size diagnostic). `level_shift` is added to the diagonal
/// preconditioner gap; `max_step` is the componentwise trust radius on κ.
#[allow(clippy::too_many_arguments)]
pub fn uhf_newton_step(
    ctx: &ParallelContext,
    inp: &UhfNewtonInputs,
    level_shift: f64,
    max_step: f64,
    cg_max_iter: usize,
    cg_conv: f64,
) -> Result<(Array2<f64>, Array2<f64>, f64), FerricError> {
    let n = inp.c_a.nrows();
    let na = inp.nocc_a;
    let nb = inp.nocc_b;

    // Gradient blocks g^σ_{ai} = F^σ_{ai}  (rows = virt, cols = occ).
    let g_a = occ_virt_block(inp.f_a_mo, na, n);
    let g_b = occ_virt_block(inp.f_b_mo, nb, n);

    // Diagonal preconditioner gaps: (F[a,a] − F[i,i]) + shift, per spin.
    let f_a_diag: Vec<f64> = (0..n).map(|i| inp.f_a_mo[(i, i)]).collect();
    let f_b_diag: Vec<f64> = (0..n).map(|i| inp.f_b_mo[(i, i)]).collect();
    let diag_a = build_gap(&f_a_diag, na, n, level_shift);
    let diag_b = build_gap(&f_b_diag, nb, n, level_shift);

    // Initial guess: diagonal solve κ⁰ = −g / gap.
    let mut k_a = neg_div(&g_a, &diag_a);
    let mut k_b = neg_div(&g_b, &diag_b);

    // Early-exit if both gradients are already below tolerance.
    let gmax = max_abs(&g_a).max(max_abs(&g_b));
    let cg_iters = if gmax < cg_conv { 0 } else { cg_max_iter };

    // PCG on the coupled (κ_α, κ_β) system H·κ = −g.
    let (h0_a, h0_b) = hessian_matvec(ctx, inp, &k_a, &k_b)?;
    let mut r_a = &neg(&g_a) - &h0_a;
    let mut r_b = &neg(&g_b) - &h0_b;
    let mut z_a = &r_a / &diag_a;
    let mut z_b = &r_b / &diag_b;
    let mut p_a = z_a.clone();
    let mut p_b = z_b.clone();
    let mut rz_old = inner(&r_a, &z_a) + inner(&r_b, &z_b);

    for _it in 0..cg_iters {
        let max_resid = max_abs(&r_a).max(max_abs(&r_b));
        if max_resid < cg_conv {
            break;
        }
        let (ap_a, ap_b) = hessian_matvec(ctx, inp, &p_a, &p_b)?;
        let p_ap = inner(&p_a, &ap_a) + inner(&p_b, &ap_b);
        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rz_old / p_ap;
        k_a.zip_mut_with(&p_a, |k, &v| *k += alpha * v);
        k_b.zip_mut_with(&p_b, |k, &v| *k += alpha * v);
        r_a.zip_mut_with(&ap_a, |r, &v| *r -= alpha * v);
        r_b.zip_mut_with(&ap_b, |r, &v| *r -= alpha * v);
        z_a = &r_a / &diag_a;
        z_b = &r_b / &diag_b;
        let rz_new = inner(&r_a, &z_a) + inner(&r_b, &z_b);
        let beta = rz_new / rz_old;
        rz_old = rz_new;
        p_a.zip_mut_with(&z_a, |p, &v| *p = v + beta * *p);
        p_b.zip_mut_with(&z_b, |p, &v| *p = v + beta * *p);
    }

    // Trust-radius clip (shared scale so the α/β step stays consistent).
    let kmax = max_abs(&k_a).max(max_abs(&k_b));
    if kmax > max_step {
        let s = max_step / kmax;
        k_a.mapv_inplace(|x| x * s);
        k_b.mapv_inplace(|x| x * s);
    }

    let c_a_new = apply_cayley(inp.c_a, &k_a, na, n)?;
    let c_b_new = apply_cayley(inp.c_b, &k_b, nb, n)?;
    Ok((c_a_new, c_b_new, kmax))
}

/// H · (κ_α, κ_β) for the two occ→virt blocks.
///
/// Public so tests can finite-difference-validate the matvec directly against
/// the orbital-gradient derivative. `k_a`/`k_b` are the occ→virt (virt×occ)
/// rotation blocks; returns the corresponding Hessian-vector-product blocks.
pub fn hessian_matvec(
    ctx: &ParallelContext,
    inp: &UhfNewtonInputs,
    k_a: &Array2<f64>,
    k_b: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let n = inp.c_a.nrows();
    let na = inp.nocc_a;
    let nb = inp.nocc_b;

    // δD^σ in AO from κ^σ. In σ-MO basis, δD^σ[a,i] = δD^σ[i,a] = κ^σ[a,i]
    // (occ→virt block only; occ=1, virt=0 ⇒ (occ[i]−occ[a])·κ = +κ). Transform
    // to AO via C_σ · δD_σ^MO · C_σ^T.
    let dd_a_ao = ao_from_ov(inp.c_a, k_a, na, n);
    let dd_b_ao = ao_from_ov(inp.c_b, k_b, nb, n);

    // δJ on δD_total; δK per spin.
    let dd_tot = &dd_a_ao + &dd_b_ao;
    let mut dj = Array2::<f64>::zeros((n, n));
    let mut dk_dum = Array2::<f64>::zeros((n, n));
    build_jk(ctx, inp.prep, inp.bounds, inp.thresh, &dd_tot, &mut dj, &mut dk_dum)?;

    let mut dk_a = Array2::<f64>::zeros((n, n));
    let mut dk_b = Array2::<f64>::zeros((n, n));
    let mut j_dum = Array2::<f64>::zeros((n, n));
    build_jk(ctx, inp.prep, inp.bounds, inp.thresh, &dd_a_ao, &mut j_dum, &mut dk_a)?;
    j_dum.fill(0.0);
    build_jk(ctx, inp.prep, inp.bounds, inp.thresh, &dd_b_ao, &mut j_dum, &mut dk_b)?;

    let c_k = inp.k_mix_sr;
    let mut df_a: Array2<f64> = &dj - &(c_k * &dk_a);
    let mut df_b: Array2<f64> = &dj - &(c_k * &dk_b);

    if let Some(fxc) = inp.fxc {
        let (dvxc_a, dvxc_b) = fxc(&dd_a_ao, &dd_b_ao);
        df_a = &df_a + &dvxc_a;
        df_b = &df_b + &dvxc_b;
    }

    // Project to each spin's own MO basis and read the occ→virt block.
    let df_a_mo = inp.c_a.t().dot(&df_a).dot(inp.c_a);
    let df_b_mo = inp.c_b.t().dot(&df_b).dot(inp.c_b);
    let mut h_a = occ_virt_block(&df_a_mo, na, n);
    let mut h_b = occ_virt_block(&df_b_mo, nb, n);

    // Diagonal orbital-energy-gap term: + (F[a,a] − F[i,i]) · κ[a,i].
    let f_a_diag: Vec<f64> = (0..n).map(|i| inp.f_a_mo[(i, i)]).collect();
    let f_b_diag: Vec<f64> = (0..n).map(|i| inp.f_b_mo[(i, i)]).collect();
    for (ir, a) in (na..n).enumerate() {
        for (ic, i) in (0..na).enumerate() {
            h_a[(ir, ic)] += (f_a_diag[a] - f_a_diag[i]) * k_a[(ir, ic)];
        }
    }
    for (ir, a) in (nb..n).enumerate() {
        for (ic, i) in (0..nb).enumerate() {
            h_b[(ir, ic)] += (f_b_diag[a] - f_b_diag[i]) * k_b[(ir, ic)];
        }
    }

    Ok((h_a, h_b))
}

// ---- helpers ----

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

/// AO δD from an occ→virt block: δD^MO symmetric with the block in both
/// (virt,occ) and (occ,virt) corners, then C·δD^MO·Cᵀ.
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
            .map_err(|e| FerricError::Lapack(format!("UHF Cayley solve: {e}")))?;
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
