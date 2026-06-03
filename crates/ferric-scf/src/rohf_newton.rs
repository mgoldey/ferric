//! Second-order ROHF / ROKS step: damped Newton via Hessian-vector products.
//!
//! Solves the linear system H·κ = −g for the orbital-rotation parameter κ,
//! where g is the MO-projected ROHF gradient (three blocks: closed→virt,
//! open→virt, closed→open) and H is the corresponding orbital Hessian.
//! The matvec H·κ is built in the AO basis by perturbing the α/β densities
//! by the κ-rotation, rebuilding J/K (+ XC kernel for ROKS) on those
//! perturbations, and reading back the three relevant MO blocks.
//!
//! The linear solver is the same DIIS-preconditioned fixed-point iteration
//! that the RI-MP2 Z-vector solver (`ferric-mp2::zvector`) uses; we don't
//! need a Davidson eigensolve because we want a single RHS, not eigenpairs.
//!
//! Rotation is applied via the Cayley unitary U = (I − κ/2)^{−1}(I + κ/2),
//! which exactly preserves orthonormality.

use crate::rhf::build_jk;
use crate::screening::SchwarzBounds;
use ferric_core::FerricError;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;
use ndarray_linalg::Solve;

/// Optional XC kernel callable for ROKS Newton: given (δD_α, δD_β) it returns
/// the response of the per-spin XC potential (δV_xc^α, δV_xc^β) in AO basis.
/// Pure ROHF passes `None`.
///
/// The lifetime parameter lets callers pass closures that borrow stack-local
/// data (e.g., a `LdaFxcKernel` + reference ρ); otherwise the trait-object
/// default would force `+ 'static`.
pub type FxcResponse<'a> = dyn Fn(&Array2<f64>, &Array2<f64>) -> (Array2<f64>, Array2<f64>) + Sync + 'a;

/// Inputs to one Newton step.
pub struct RohfNewtonInputs<'a> {
    pub prep: &'a PreparedBasis,
    pub bounds: &'a SchwarzBounds,
    pub c: &'a Array2<f64>,          // MO coeffs (n × n), columns ordered closed | open | virt
    pub eps: &'a [f64],              // MO eigenvalues (semicanonical, from diagonalising F_eff)
    pub f_a_mo: &'a Array2<f64>,     // α Fock in MO basis (current iter)
    pub f_b_mo: &'a Array2<f64>,     // β Fock in MO basis
    pub nocc_double: usize,
    pub nocc_open: usize,
    pub k_mix_sr: f64,               // K mixing coefficient (1.0 for HF, c_HF for hybrid; ignored for RSH)
    pub fxc: Option<&'a FxcResponse<'a>>,
    pub thresh: f64,
}

/// One damped-Newton step on ROHF/ROKS MO coefficients.
///
/// Returns updated C and the κ infinity-norm (a step-size diagnostic).
/// `level_shift` is added to the diagonal preconditioner |Δε| denominator;
/// `max_step` is the trust radius applied componentwise to κ.
pub fn rohf_newton_step(
    ctx: &ParallelContext,
    inp: &RohfNewtonInputs,
    level_shift: f64,
    max_step: f64,
    cg_max_iter: usize,
    cg_conv: f64,
) -> Result<(Array2<f64>, f64), FerricError> {
    let n = inp.c.nrows();
    let nc = inp.nocc_double;
    let no = inp.nocc_open;
    let nocc_a = nc + no;

    // Pack RHS −g in MO basis from the three blocks.
    //   g[v,c] = f_α[v,c] + f_β[v,c]
    //   g[v,o] = f_α[v,o]
    //   g[o,c] = f_β[o,c]
    let (g_vc, g_vo, g_oc) = gradient_blocks(inp);

    // Per-spin semicanonical diagonal entries — PySCF convention.
    // For each block, the diagonal Fock contribution is built from the spin
    // that actually changes occupation in that block:
    //   closed↔virt: both α and β change → (F_α[v,v] + F_β[v,v]) − (F_α[c,c] + F_β[c,c])
    //   open↔virt:   only α changes      → F_α[v,v] − F_α[o,o]
    //   closed↔open: only β changes      → F_β[o,o] − F_β[c,c]
    // The Hessian matvec applies these same per-spin Fock diagonals
    // multiplicatively to κ — using F_eff eigenvalues (the previous
    // implementation) was wrong because F_eff eigenvalues mix α and β
    // contributions in the wrong proportions for the off-diagonal blocks,
    // leading to a Newton step that points away from the minimum.
    let f_a_diag: Vec<f64> = (0..n).map(|i| inp.f_a_mo[(i, i)]).collect();
    let f_b_diag: Vec<f64> = (0..n).map(|i| inp.f_b_mo[(i, i)]).collect();
    let f_sum_diag: Vec<f64> = (0..n).map(|i| f_a_diag[i] + f_b_diag[i]).collect();

    let diag_vc = build_diag_perspin(&f_sum_diag, nocc_a..n, 0..nc, level_shift);
    let diag_vo = build_diag_perspin(&f_a_diag,   nocc_a..n, nc..nocc_a, level_shift);
    let diag_oc = build_diag_perspin(&f_b_diag,   nc..nocc_a, 0..nc, level_shift);

    // Initial guess: diagonal solve κ⁰ = −g / Δε.
    let mut k_vc = elemwise_div_neg(&g_vc, &diag_vc);
    let mut k_vo = elemwise_div_neg(&g_vo, &diag_vo);
    let mut k_oc = elemwise_div_neg(&g_oc, &diag_oc);

    // Early-exit if the gradient itself is below tolerance (we're already
    // at a stationary point — the off-diagonal matvec correction would
    // only amplify numerical noise).
    let gmax = arr_max_abs(&g_vc).max(arr_max_abs(&g_vo)).max(arr_max_abs(&g_oc));
    let cg_max_iter_effective = if gmax < cg_conv { 0 } else { cg_max_iter };

    // Preconditioned conjugate-gradient solve for H·κ = −g.
    //
    // The orbital Hessian is symmetric (and positive-definite near a
    // minimum), so PCG converges in O(√κ) iterations, vs Jacobi which
    // diverges whenever the off-diagonal coupling exceeds the |Δε|
    // diagonal. Preconditioner M = diag(|Δε| + shift); M^{-1} amplifies
    // long-wavelength modes the way Jacobi does, but inside the
    // conjugate-direction machinery the divergence is suppressed.
    //
    // Standard PCG (Saad, "Iterative Methods", Alg. 9.1) on the packed
    // (vc, vo, oc) triple.
    let (h0_vc, h0_vo, h0_oc) = hessian_matvec(ctx, inp, &k_vc, &k_vo, &k_oc)?;
    let mut r_vc = sub(&neg(&g_vc), &h0_vc);
    let mut r_vo = sub(&neg(&g_vo), &h0_vo);
    let mut r_oc = sub(&neg(&g_oc), &h0_oc);

    let mut z_vc = elemwise_div(&r_vc, &diag_vc);
    let mut z_vo = elemwise_div(&r_vo, &diag_vo);
    let mut z_oc = elemwise_div(&r_oc, &diag_oc);

    let mut p_vc = z_vc.clone();
    let mut p_vo = z_vo.clone();
    let mut p_oc = z_oc.clone();

    let mut rz_old: f64 =
        inner(&r_vc, &z_vc) + inner(&r_vo, &z_vo) + inner(&r_oc, &z_oc);

    for _it in 0..cg_max_iter_effective {
        let max_resid =
            arr_max_abs(&r_vc).max(arr_max_abs(&r_vo)).max(arr_max_abs(&r_oc));
        if max_resid < cg_conv {
            break;
        }
        // Ap = H · p
        let (ap_vc, ap_vo, ap_oc) = hessian_matvec(ctx, inp, &p_vc, &p_vo, &p_oc)?;
        let p_ap: f64 = inner(&p_vc, &ap_vc) + inner(&p_vo, &ap_vo) + inner(&p_oc, &ap_oc);
        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rz_old / p_ap;
        // κ ← κ + α p
        k_vc.zip_mut_with(&p_vc, |k, &v| *k += alpha * v);
        k_vo.zip_mut_with(&p_vo, |k, &v| *k += alpha * v);
        k_oc.zip_mut_with(&p_oc, |k, &v| *k += alpha * v);
        // r ← r − α (A p)
        r_vc.zip_mut_with(&ap_vc, |r, &v| *r -= alpha * v);
        r_vo.zip_mut_with(&ap_vo, |r, &v| *r -= alpha * v);
        r_oc.zip_mut_with(&ap_oc, |r, &v| *r -= alpha * v);
        // z = M^{-1} r
        z_vc = elemwise_div(&r_vc, &diag_vc);
        z_vo = elemwise_div(&r_vo, &diag_vo);
        z_oc = elemwise_div(&r_oc, &diag_oc);
        let rz_new: f64 =
            inner(&r_vc, &z_vc) + inner(&r_vo, &z_vo) + inner(&r_oc, &z_oc);
        let beta = rz_new / rz_old;
        rz_old = rz_new;
        // p ← z + β p
        p_vc.zip_mut_with(&z_vc, |p, &v| *p = v + beta * *p);
        p_vo.zip_mut_with(&z_vo, |p, &v| *p = v + beta * *p);
        p_oc.zip_mut_with(&z_oc, |p, &v| *p = v + beta * *p);
    }

    // Trust-radius clip.
    let kmax = arr_max_abs(&k_vc).max(arr_max_abs(&k_vo)).max(arr_max_abs(&k_oc));
    let scale = if kmax > max_step { max_step / kmax } else { 1.0 };
    if scale < 1.0 {
        k_vc.mapv_inplace(|x| x * scale);
        k_vo.mapv_inplace(|x| x * scale);
        k_oc.mapv_inplace(|x| x * scale);
    }

    // Assemble antisymmetric κ in MO basis.
    let mut kappa = Array2::<f64>::zeros((n, n));
    for (ir, p) in (nocc_a..n).enumerate() {
        for (ic, q) in (0..nc).enumerate() {
            kappa[(p, q)] = k_vc[(ir, ic)];
            kappa[(q, p)] = -k_vc[(ir, ic)];
        }
    }
    for (ir, p) in (nocc_a..n).enumerate() {
        for (ic, q) in (nc..nocc_a).enumerate() {
            kappa[(p, q)] = k_vo[(ir, ic)];
            kappa[(q, p)] = -k_vo[(ir, ic)];
        }
    }
    for (ir, p) in (nc..nocc_a).enumerate() {
        for (ic, q) in (0..nc).enumerate() {
            kappa[(p, q)] = k_oc[(ir, ic)];
            kappa[(q, p)] = -k_oc[(ir, ic)];
        }
    }

    // Cayley unitary U = (I − κ/2)^{−1} (I + κ/2). Apply to C: C ← C·U.
    let half_k = 0.5 * &kappa;
    let i_eye = Array2::<f64>::eye(n);
    let a = &i_eye - &half_k;
    let b = &i_eye + &half_k;
    // Solve A·U = B columnwise.
    let mut u = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let bcol = b.column(col).to_owned();
        let sol = a.solve(&bcol)
            .map_err(|e| FerricError::Lapack(format!("Cayley solve: {e}")))?;
        for row in 0..n {
            u[(row, col)] = sol[row];
        }
    }
    let c_new = inp.c.dot(&u);

    Ok((c_new, kmax))
}

/// Compute H · κ for the three κ-blocks.
///
/// Build α/β AO density perturbations δD_α, δD_β from κ, build J/K on those
/// (and add the optional XC kernel response), project the resulting δF_α,
/// δF_β to MO basis, and read off the three block components using the
/// PySCF Roothaan gradient pairing.
/// Build the three Roothaan-projected gradient blocks (vc, vo, oc) in MO basis
/// from per-spin Focks. Same convention as the DIIS error-vector in
/// `solve_rohf`:
///   g[v,c] = f_α[v,c] + f_β[v,c]
///   g[v,o] = f_α[v,o]
///   g[o,c] = f_β[o,c]
pub(crate) fn gradient_blocks(inp: &RohfNewtonInputs) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let n = inp.c.nrows();
    let nc = inp.nocc_double;
    let no = inp.nocc_open;
    let nocc_a = nc + no;
    let g_vc = pack_block(inp.f_a_mo, inp.f_b_mo, nocc_a..n, 0..nc, true);
    let g_vo = pack_block(inp.f_a_mo, inp.f_b_mo, nocc_a..n, nc..nocc_a, false);
    let g_oc = pack_block(inp.f_b_mo, inp.f_b_mo, nc..nocc_a, 0..nc, false);
    (g_vc, g_vo, g_oc)
}

// Returns the three coupling blocks (vc, vo, oc) of the Hessian-vector product;
// the tuple mirrors the three input blocks and reads clearer than an alias.
#[allow(clippy::type_complexity)]
pub(crate) fn hessian_matvec(
    ctx: &ParallelContext,
    inp: &RohfNewtonInputs,
    k_vc: &Array2<f64>,
    k_vo: &Array2<f64>,
    k_oc: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), FerricError> {
    let n = inp.c.nrows();
    let nc = inp.nocc_double;
    let no = inp.nocc_open;
    let nocc_a = nc + no;

    // δD_α, δD_β in AO basis from κ.
    //
    // Occupations: α has 1 in [0..nocc_a], β has 1 in [0..nc].
    // For a unitary U ≈ I + κ, δD_σ = κ_σ D_σ + D_σ κ_σ^T = [κ, D_σ] when
    // κ is antisymmetric and D_σ is the occupied projector.
    // In MO basis δD_σ[p,q] is nonzero only when one of p,q is occupied(σ)
    // and the other is not:
    //   δD_σ[p,q] = (occ_σ[q] − occ_σ[p]) · κ[p,q]
    // Then transform to AO via C·δD·C^T.

    let mut dd_a_mo = Array2::<f64>::zeros((n, n));
    let mut dd_b_mo = Array2::<f64>::zeros((n, n));

    // closed↔virt: changes both α and β by κ[v,c]
    for (ir, p) in (nocc_a..n).enumerate() {
        for (ic, q) in (0..nc).enumerate() {
            let v = k_vc[(ir, ic)];
            // virt - closed = 0 - 1 = -1 → δD[p,q] = (0-1)·κ[p,q] = -v
            // but we want δD symmetric in AO, so also δD[q,p] = +v − consistency check:
            // δD = κ D + D κ^T = κ D − D κ (since κ^T = −κ)
            // Component (q,p) with q closed, p virt:
            //   (κ D)[q,p] = Σ_r κ[q,r] D[r,p] = κ[q,p]·occ[p] = κ[q,p]·0 = 0
            //   (−D κ)[q,p] = −Σ_r D[q,r] κ[r,p] = −occ[q]·κ[q,p] = −κ[q,p]
            //   ⇒ δD[q,p] = −κ[q,p] = +κ[p,q] = +v
            // Component (p,q) with p virt, q closed:
            //   (κ D)[p,q] = κ[p,q]·1 = κ[p,q] = v
            //   (−D κ)[p,q] = −0·… = 0
            //   ⇒ δD[p,q] = v
            // δD is symmetric. ✓
            dd_a_mo[(p, q)] = v;
            dd_a_mo[(q, p)] = v;
            dd_b_mo[(p, q)] = v;
            dd_b_mo[(q, p)] = v;
        }
    }
    // open↔virt: α only (open is α-occupied, β-unoccupied)
    for (ir, p) in (nocc_a..n).enumerate() {
        for (ic, q) in (nc..nocc_a).enumerate() {
            let v = k_vo[(ir, ic)];
            dd_a_mo[(p, q)] = v;
            dd_a_mo[(q, p)] = v;
            // β: both p and q are β-unoccupied → δD_β = 0
        }
    }
    // closed↔open: β only (closed is β-occupied, open is β-unoccupied)
    // Indices: row = open (p), col = closed (q). occ_β: p=0, q=1.
    for (ir, p) in (nc..nocc_a).enumerate() {
        for (ic, q) in (0..nc).enumerate() {
            let v = k_oc[(ir, ic)];
            dd_b_mo[(p, q)] = v;
            dd_b_mo[(q, p)] = v;
            // α: both occupied → δD_α = 0
        }
    }

    // AO basis
    let dd_a_ao = inp.c.dot(&dd_a_mo).dot(&inp.c.t());
    let dd_b_ao = inp.c.dot(&dd_b_mo).dot(&inp.c.t());

    // δJ from δD_total = δD_α + δD_β.
    let dd_tot = &dd_a_ao + &dd_b_ao;
    let mut dj = Array2::<f64>::zeros((n, n));
    let mut dk_dum = Array2::<f64>::zeros((n, n));  // discarded — we want J only here
    // Build J on δD_total. We re-use build_jk but only keep J; K is rebuilt per-spin below.
    build_jk(ctx, inp.prep, inp.bounds, inp.thresh, &dd_tot, &mut dj, &mut dk_dum)?;

    // δK per spin.
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

    // Project to MO basis and read off the three blocks (using the same
    // pairing as the gradient).
    let df_a_mo = inp.c.t().dot(&df_a).dot(inp.c);
    let df_b_mo = inp.c.t().dot(&df_b).dot(inp.c);

    let mut h_vc = pack_block(&df_a_mo, &df_b_mo, nocc_a..n, 0..nc, true);
    let mut h_vo = pack_block(&df_a_mo, &df_b_mo, nocc_a..n, nc..nocc_a, false);  // α
    let mut h_oc = pack_block(&df_b_mo, &df_b_mo, nc..nocc_a, 0..nc, false);       // β

    // Per-spin diagonal Fock-commutator: each block uses the diagonal Fock
    // entries from the spin(s) that actually change occupation in that block.
    // See preconditioner notes above.
    let f_a_diag: Vec<f64> = (0..n).map(|i| inp.f_a_mo[(i, i)]).collect();
    let f_b_diag: Vec<f64> = (0..n).map(|i| inp.f_b_mo[(i, i)]).collect();
    for (ir, p) in (nocc_a..n).enumerate() {
        for (ic, q) in (0..nc).enumerate() {
            let gap = (f_a_diag[p] + f_b_diag[p]) - (f_a_diag[q] + f_b_diag[q]);
            h_vc[(ir, ic)] += gap * k_vc[(ir, ic)];
        }
    }
    for (ir, p) in (nocc_a..n).enumerate() {
        for (ic, q) in (nc..nocc_a).enumerate() {
            let gap = f_a_diag[p] - f_a_diag[q];
            h_vo[(ir, ic)] += gap * k_vo[(ir, ic)];
        }
    }
    for (ir, p) in (nc..nocc_a).enumerate() {
        for (ic, q) in (0..nc).enumerate() {
            let gap = f_b_diag[p] - f_b_diag[q];
            h_oc[(ir, ic)] += gap * k_oc[(ir, ic)];
        }
    }

    Ok((h_vc, h_vo, h_oc))
}

// ---- block helpers ----

/// Extract a block from MO matrices using the Roothaan pairing.
/// If `add` is true, returns A[rows,cols] + B[rows,cols]; otherwise returns A[rows,cols].
pub(crate) fn pack_block(
    a: &Array2<f64>,
    _b: &Array2<f64>,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    add: bool,
) -> Array2<f64> {
    let nr = rows.end - rows.start;
    let nc = cols.end - cols.start;
    let mut out = Array2::<f64>::zeros((nr, nc));
    if add {
        for (i, r) in rows.clone().enumerate() {
            for (j, c) in cols.clone().enumerate() {
                out[(i, j)] = a[(r, c)] + _b[(r, c)];
            }
        }
    } else {
        for (i, r) in rows.enumerate() {
            for (j, c) in cols.clone().enumerate() {
                out[(i, j)] = a[(r, c)];
            }
        }
    }
    out
}

fn build_diag_perspin(
    f_diag: &[f64],
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    shift: f64,
) -> Array2<f64> {
    let nr = rows.end - rows.start;
    let nc = cols.end - cols.start;
    let mut out = Array2::<f64>::zeros((nr, nc));
    for (i, r) in rows.enumerate() {
        for (j, c) in cols.clone().enumerate() {
            // (F[p,p] − F[q,q]) + shift, with a 1e-6 floor for near-degeneracies.
            // Signed (NOT abs) — for an occ→virt block this is positive and
            // gives the right Newton step direction when divided into the gradient.
            let d = (f_diag[r] - f_diag[c]) + shift;
            out[(i, j)] = if d.abs() < 1e-6 { 1e-6 } else { d };
        }
    }
    out
}

fn elemwise_div(num: &Array2<f64>, den: &Array2<f64>) -> Array2<f64> { num / den }
fn elemwise_div_neg(num: &Array2<f64>, den: &Array2<f64>) -> Array2<f64> { -num / den }
fn neg(a: &Array2<f64>) -> Array2<f64> { -a }
fn sub(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> { a - b }
fn arr_max_abs(a: &Array2<f64>) -> f64 { a.iter().fold(0.0f64, |m, &v| m.max(v.abs())) }
fn inner(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
