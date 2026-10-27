//! Integral-direct (RI) particle-particle ladder for closed-shell CCSD.
//!
//! # What this replaces
//!
//! The dense spin-adapted CCSD in [`crate::ccsd_closed_shell`] materializes the
//! chemist `vvvv` block `(ab|cd)` in core and builds the t1-dressed `W_abcd`
//! from a permuted `(ac|bd)` copy every iteration — **four `nv⁴` buffers at
//! the peak**, which is what its memory plan charges. Measured
//! peak RSS on alkane/cc-pVDZ scales cleanly as `nv⁴` and hits 7.90 GB at
//! `nv=109` (alkane_5, 130 basis functions) — a *basis-size* ceiling, not an
//! atom-count one (STO-3G reaches alkane_8 in well under a GB).
//!
//! This module computes the same contraction
//!
//! ```text
//!   Z[i,j,a,b] = Σ_cd W_abcd · tau[i,j,c,d]
//!   W_abcd     = (ac|bd) − Σ_k (kd|ac) t1[k,b] − Σ_k (kc|bd) t1[k,a]
//! ```
//!
//! **without holding the `nv⁴` chemist block or `W_abcd`.** Peak virtual-side
//! scratch is one `blk · nv³` panel for a block width `blk ≤ nv`: at most one
//! `nv⁴`-sized panel at full width (`blk = nv`; the dense route holds four
//! `nv⁴` buffers at its peak), and narrower panels when the budget requires, so
//! the memory floor is `nv³` × a constant instead of `4·nv⁴`.
//!
//! # The factorization
//!
//! With the dressed RI 3-index blocks `B^P_pq` (`(pq|rs) = Σ_P B^P_pq B^P_rs`),
//! define the **t1-dressed virtual-virtual block**
//!
//! ```text
//!   Bx[P,b,d] = − Σ_k B_ov[P,k,d] · t1[k,b]        (shape naux × nv × nv)
//! ```
//!
//! Then, term by term:
//!
//! ```text
//!   (ac|bd)              = Σ_P B_vv[P,a,c] · B_vv[P,b,d]
//!   − Σ_k (kc|bd) t1[k,a] = Σ_P Bx[P,a,c]   · B_vv[P,b,d]
//!   − Σ_k (kd|ac) t1[k,b] = Σ_P B_vv[P,a,c] · Bx[P,b,d]
//! ```
//!
//! so, summing,
//!
//! ```text
//!   W_abcd = Σ_P [ (B_vv + Bx)[P,a,c] · B_vv[P,b,d] + B_vv[P,a,c] · Bx[P,b,d] ]
//! ```
//!
//! Note the *absent* `Bx·Bx` term: that would be the `t1²` piece, which the
//! Hirata/PySCF `cc_Wvvvv` intermediate does not contain either (the quadratic
//! t1 contribution enters the T2 equation through `tau`, not through W). Keeping
//! the two cross terms and dropping `Bx·Bx` reproduces the dense `W_abcd`
//! **exactly**, not approximately — this is an algebraic refactor of the same
//! expression, so agreement with the dense path is at RI-*identical* level
//! (both sides use the same `B`), i.e. machine precision, not the RI floor.
//!
//! `Bx` costs `naux · nv²` — the same size class as `B_vv` itself, which the
//! dense path already holds. The only `nv⁴`-class buffer is the panel itself,
//! and only at full block width.
//!
//! # Blocking
//!
//! We loop over blocks of the first virtual index `a`. For a block `A` of width
//! `blk`:
//!
//! 1. Form `G[(a∈A, c), (b, d)] = Σ_P L[P,(a,c)] · B_vv[P,(b,d)]` as ONE wide
//!    GEMM of shape `(blk·nv) × naux × (nv·nv)`, where `L = B_vv + Bx` — this is
//!    the `(B_vv+Bx)·B_vv` half. Then a second wide GEMM with `L = B_vv`,
//!    `R = Bx` for the other half, accumulated into the same buffer.
//! 2. Contract `Z[i,j,a∈A,b] += Σ_cd G[(a,c),(b,d)] · tau[i,j,c,d]`, again as a
//!    GEMM by viewing `tau` as `(no·no) × (nv·nv)` and `G` reshaped to put
//!    `(c,d)` contiguous.
//!
//! Step 2 needs `G` indexed `[(a,c),(b,d)]` but contracted over `(c,d)` against
//! a `(b)`-free index — so per `a` we transpose the `nv × nv × nv` slab
//! `G[a][c][b][d] → G'[a][b][(c,d)]` and do `Z[:, :, a, :] = tau_mat · G'ᵀ`.
//! That transpose is `O(nv³)` per `a`, `O(nv⁴)` in *time* overall — the ladder
//! is `O(no²nv⁴)` in time regardless, so the permute is a lower-order cost, but
//! it does give up some BLAS3 shape versus the dense path's single huge GEMM.
//! **RI is slower than dense per iteration** (measured numbers in
//! `site/src/methods/cc.md`); the win is the ceiling, not speed, which is why
//! `CcsdLadder::Auto` only takes it when the dense plan does not fit.
//!
//! # Memory accounting
//!
//! Per block, resident virtual-side scratch is:
//!
//! ```text
//!   g_blk : blk · nv³ f64        (the (a,c),(b,d) panel)
//!   gp    : nv³ f64              (the per-a transposed slab)
//!   bx    : naux · nv² f64       (loop-invariant, built once per iteration)
//! ```
//!
//! The second GEMM accumulates into `g_blk` in place via
//! `ndarray::linalg::general_mat_mul` (BLAS `beta = 1`). Writing it as
//! `g_blk += &(...).dot(...)` instead would allocate a SECOND full `blk · nv³`
//! temporary and double the peak of the largest buffer here — that was measured
//! as a real ~1.1 GB of avoidable peak on alkane_5/cc-pVDZ before it was fixed,
//! so this is a load-bearing detail, not a micro-optimization.
//!
//! `.dot` on two row-stride-1 operands can return a column-major array, so
//! `g_blk` is only ever read by 2-index `[[i, j]]` access, never by flat
//! offset; the transposed slab `gp` is built that way for the same reason.
//!
//! `ccsd_closed_shell::plan_ladder` picks `blk` as the widest width whose
//! whole-run `MemoryPlan` still fits — the panel is one more reservation on
//! the same plan as everything else, not a separate fraction of the budget —
//! and the choice is *enforced*: a budget too small for even `blk = 1` is a
//! hard error naming this scratch, never a best-effort that walks into an OOM.
//!
//! # Measured: dense vs RI
//!
//! Alkane/cc-pVDZ, one route per process (`VmHWM` never resets), routes
//! interleaved, `OPENBLAS_NUM_THREADS=1`, `RAYON_NUM_THREADS=4`, load average
//! 3-6 from other jobs, measured 2026-10-26 with
//! `benchmarks/harness/examples/ri_ccsd_memory.rs`. RI ran at full block width
//! (`blk = nv`, the default-budget choice). Per-iteration time: 3 repeats per
//! route for alkane_3, 2 for alkane_4/5; peak RSS: one converged run each.
//!
//! ```text
//!   system    nv   peak RSS dense  RI      per-iteration dense  RI      RI/dense
//!   alkane_3  69   1.50 GB   0.70 GB       1.27 s    1.64 s     1.30
//!   alkane_4  89   3.79 GB   1.67 GB       3.2-3.7 s 5.2-5.3 s  1.50
//!   alkane_5  109  7.90 GB   3.40 GB       8.8-10.8 s 14.5 s    1.48
//! ```
//!
//! Energies are identical to all 10 printed digits. The rebuild costs
//! 1.3-1.5x per iteration, so `CcsdLadder::Auto` takes this route only when
//! the dense plan does not fit the budget.
//!
//! What is NOT removed: `ovvv` (`no·nv³`) is still built in core by
//! [`crate::ccsd_closed_shell`] and is now the largest single virtual-side
//! tensor. It is 0.22 GB at alkane_5 but ~597 GB at drug-fragment scale
//! (`no=170, nv=760`), where `vvvv` would have been ~2669 GB — so this change
//! removes the dominant wall, not the last one.

use ferric_core::FerricError;
use ndarray::{Array2, ArrayD, ArrayView2, IxDyn};

/// One RI particle-particle ladder contraction, integral-direct.
///
/// Computes `Z[i,j,a,b] = Σ_cd W_abcd · tau[i,j,c,d]` with
/// `W_abcd = (ac|bd) − Σ_k (kd|ac) t1[k,b] − Σ_k (kc|bd) t1[k,a]`, forming
/// `W` only in `blk × nv × nv × nv` panels.
///
/// # Arguments
/// * `b_vv` — dressed `B^P_ab`, shape `(naux, nv, nv)`.
/// * `b_ov` — dressed `B^P_ia`, shape `(naux, no, nv)`.
/// * `t1`   — `(no, nv)`.
/// * `tau`  — `t2 + t1⊗t1`, shape `(no, no, nv, nv)`.
/// * `blk`  — `a`-block width, chosen by `ccsd_closed_shell::plan_ladder`.
///
/// Returns `Z` with shape `(no, no, nv, nv)`.
pub fn ri_ladder_contract(
    b_vv: &ArrayD<f64>,
    b_ov: &ArrayD<f64>,
    t1: &ArrayD<f64>,
    tau: &ArrayD<f64>,
    blk: usize,
) -> Result<ArrayD<f64>, FerricError> {
    let naux = b_vv.shape()[0];
    let nv = b_vv.shape()[1];
    let no = b_ov.shape()[1];
    if b_vv.shape() != [naux, nv, nv] {
        return Err(FerricError::General(format!(
            "RI-CCSD ladder: b_vv shape {:?} is not (naux, nv, nv)",
            b_vv.shape()
        )));
    }
    if b_ov.shape() != [naux, no, nv] {
        return Err(FerricError::General(format!(
            "RI-CCSD ladder: b_ov shape {:?} is not (naux, no, nv) with nv={nv}",
            b_ov.shape()
        )));
    }
    if tau.shape() != [no, no, nv, nv] {
        return Err(FerricError::General(format!(
            "RI-CCSD ladder: tau shape {:?} is not (no, no, nv, nv) = ({no}, {no}, {nv}, {nv})",
            tau.shape()
        )));
    }
    if t1.shape() != [no, nv] {
        return Err(FerricError::General(format!(
            "RI-CCSD ladder: t1 shape {:?} is not (no, nv)",
            t1.shape()
        )));
    }
    if blk == 0 {
        return Err(FerricError::General("RI-CCSD ladder: block width 0".into()));
    }

    // --- loop-invariant: Bx[P,b,d] = − Σ_k B_ov[P,k,d] t1[k,b] ------------
    // Per P this is a (nv × no) · (no × nv) GEMM: (−t1)ᵀ_{b,k} · B_ov[P]_{k,d}.
    let bx = build_bx(b_ov, t1, naux, no, nv)?;

    // Flat (naux, nv*nv) views for the wide GEMMs. `as_standard_layout` is a
    // no-op for already-contiguous input but guarantees the reshape is valid.
    let b_vv_std = b_vv.as_standard_layout();
    let b_vv_flat: ArrayView2<f64> = b_vv_std
        .view()
        .into_shape_with_order((naux, nv * nv))
        .map_err(|e| FerricError::General(format!("RI-CCSD ladder: b_vv reshape: {e}")))?;
    let bx_flat: ArrayView2<f64> = bx
        .view()
        .into_shape_with_order((naux, nv * nv))
        .map_err(|e| FerricError::General(format!("RI-CCSD ladder: bx reshape: {e}")))?;

    // L = B_vv + Bx, the left factor of the first half.
    let l_flat: Array2<f64> = &b_vv_flat + &bx_flat;

    // tau viewed as (no*no) × (nv*nv), the fixed left operand of step 2.
    let tau_std = tau.as_standard_layout();
    let tau_mat: ArrayView2<f64> = tau_std
        .view()
        .into_shape_with_order((no * no, nv * nv))
        .map_err(|e| FerricError::General(format!("RI-CCSD ladder: tau reshape: {e}")))?;

    let mut z = ArrayD::<f64>::zeros(IxDyn(&[no, no, nv, nv]));

    let nvv = nv * nv;
    // Scratch reused across every block and every `a` — allocated once so the
    // per-iteration peak is the documented `blk·nv³ + nv³`, not a churn of
    // transient allocations the allocator may not return to the OS.
    let mut gp = Array2::<f64>::zeros((nv, nvv)); // G'[b, (c,d)] for one `a`

    let mut a0 = 0usize;
    while a0 < nv {
        // RAGGED FINAL BLOCK: the last block is `nv - a0` wide whenever `blk`
        // does not divide `nv`. Everything below is driven by `alen`, never by
        // `blk`, so a partial block is handled identically to a full one.
        let alen = blk.min(nv - a0);

        // g_blk[(a_local, c), (b, d)] = Σ_P L[P,(a,c)] B_vv[P,(b,d)]
        //                             + Σ_P B_vv[P,(a,c)] Bx[P,(b,d)]
        let rows = a0 * nv..(a0 + alen) * nv;
        let l_slab = l_flat.slice(ndarray::s![.., rows.clone()]); // (naux, alen*nv)
        let bvv_slab = b_vv_flat.slice(ndarray::s![.., rows.clone()]); // (naux, alen*nv)

        // In-place accumulation and layout: see "Memory accounting" above.
        let mut g_blk: Array2<f64> = l_slab.t().dot(&b_vv_flat);
        ndarray::linalg::general_mat_mul(1.0, &bvv_slab.t(), &bx_flat, 1.0, &mut g_blk);

        for a_loc in 0..alen {
            // g_blk rows for this `a` are [a_loc*nv + c], columns are [b*nv + d].
            // We need G'[b, (c,d)] = g_blk[(a_loc,c), (b,d)].
            for c in 0..nv {
                let src_row = a_loc * nv + c;
                for b in 0..nv {
                    let dst = c * nv; // (c,d) flat base
                    for d in 0..nv {
                        gp[[b, dst + d]] = g_blk[[src_row, b * nv + d]];
                    }
                }
            }
            // Z[(i,j), b] for this `a`: (no*no, nv*nv) · (nv*nv, nv)
            let zblk = tau_mat.dot(&gp.t()); // (no*no, nv)
            let zblk = zblk.as_standard_layout();
            let a = a0 + a_loc;
            for i in 0..no {
                for j in 0..no {
                    let r = i * no + j;
                    for b in 0..nv {
                        z[[i, j, a, b]] = zblk[[r, b]];
                    }
                }
            }
        }
        a0 += alen;
    }

    Ok(z)
}

/// `Bx[P,b,d] = − Σ_k B_ov[P,k,d] · t1[k,b]`, shape `(naux, nv, nv)`.
///
/// Split out so a test can pin it against the defining sum independently of the
/// ladder that consumes it.
pub fn build_bx(
    b_ov: &ArrayD<f64>,
    t1: &ArrayD<f64>,
    naux: usize,
    no: usize,
    nv: usize,
) -> Result<ArrayD<f64>, FerricError> {
    let b_ov_std = b_ov.as_standard_layout();
    let b_ov_flat: ArrayView2<f64> = b_ov_std
        .view()
        .into_shape_with_order((naux, no * nv))
        .map_err(|e| FerricError::General(format!("RI-CCSD ladder: b_ov reshape: {e}")))?;
    // t1n[b,k] = −t1[k,b]
    let mut t1n = Array2::<f64>::zeros((nv, no));
    for k in 0..no {
        for b in 0..nv {
            t1n[[b, k]] = -t1[[k, b]];
        }
    }
    let mut bx = ArrayD::<f64>::zeros(IxDyn(&[naux, nv, nv]));
    for p in 0..naux {
        let bp = b_ov_flat
            .slice(ndarray::s![p, ..])
            .into_shape_with_order((no, nv))
            .map_err(|e| FerricError::General(format!("RI-CCSD ladder: b_ov[P] reshape: {e}")))?;
        let prod = t1n.dot(&bp); // (nv, nv) = (nv,no)·(no,nv)
        let prod = prod.as_standard_layout();
        for b in 0..nv {
            for d in 0..nv {
                bx[[p, b, d]] = prod[[b, d]];
            }
        }
    }
    Ok(bx)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random filler (no rand dependency).
    fn filler(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }
    }

    /// Build `W_abcd` and `Z` the DENSE way, straight from the definition, with
    /// explicit index loops — no einsum, no permutes, nothing shared with the
    /// implementation under test. This is the oracle.
    fn dense_reference(
        b_vv: &ArrayD<f64>,
        b_ov: &ArrayD<f64>,
        t1: &ArrayD<f64>,
        tau: &ArrayD<f64>,
    ) -> ArrayD<f64> {
        let naux = b_vv.shape()[0];
        let nv = b_vv.shape()[1];
        let no = b_ov.shape()[1];
        // W[a,b,c,d]
        let mut w = ArrayD::<f64>::zeros(IxDyn(&[nv, nv, nv, nv]));
        for a in 0..nv {
            for b in 0..nv {
                for c in 0..nv {
                    for d in 0..nv {
                        // (ac|bd)
                        let mut v = 0.0;
                        for p in 0..naux {
                            v += b_vv[[p, a, c]] * b_vv[[p, b, d]];
                        }
                        // − Σ_k (kd|ac) t1[k,b]
                        for k in 0..no {
                            let mut kdac = 0.0;
                            for p in 0..naux {
                                kdac += b_ov[[p, k, d]] * b_vv[[p, a, c]];
                            }
                            v -= kdac * t1[[k, b]];
                        }
                        // − Σ_k (kc|bd) t1[k,a]
                        for k in 0..no {
                            let mut kcbd = 0.0;
                            for p in 0..naux {
                                kcbd += b_ov[[p, k, c]] * b_vv[[p, b, d]];
                            }
                            v -= kcbd * t1[[k, a]];
                        }
                        w[[a, b, c, d]] = v;
                    }
                }
            }
        }
        let mut z = ArrayD::<f64>::zeros(IxDyn(&[no, no, nv, nv]));
        for i in 0..no {
            for j in 0..no {
                for a in 0..nv {
                    for b in 0..nv {
                        let mut acc = 0.0;
                        for c in 0..nv {
                            for d in 0..nv {
                                acc += w[[a, b, c, d]] * tau[[i, j, c, d]];
                            }
                        }
                        z[[i, j, a, b]] = acc;
                    }
                }
            }
        }
        z
    }

    fn random_inputs(
        naux: usize,
        no: usize,
        nv: usize,
    ) -> (ArrayD<f64>, ArrayD<f64>, ArrayD<f64>, ArrayD<f64>) {
        let mut r = filler(0x9e37_79b9_7f4a_7c15);
        let b_vv = ArrayD::from_shape_fn(IxDyn(&[naux, nv, nv]), |_| r());
        let b_ov = ArrayD::from_shape_fn(IxDyn(&[naux, no, nv]), |_| r());
        let t1 = ArrayD::from_shape_fn(IxDyn(&[no, nv]), |_| r());
        // tau deliberately NOT symmetric in (c,d) or (i,j): an implementation
        // that transposed a B slice or swapped c/d must be detectable.
        let tau = ArrayD::from_shape_fn(IxDyn(&[no, no, nv, nv]), |_| r());
        (b_vv, b_ov, t1, tau)
    }

    fn max_abs_diff(a: &ArrayD<f64>, b: &ArrayD<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    /// The RI factorization must reproduce the explicit dense `W`-then-contract
    /// reference to machine precision — this is an algebraic identity over the
    /// SAME `B`, not an approximation, so the tolerance is tight.
    #[test]
    fn ri_ladder_matches_dense_reference() {
        let (naux, no, nv) = (11, 3, 5);
        let (b_vv, b_ov, t1, tau) = random_inputs(naux, no, nv);
        let want = dense_reference(&b_vv, &b_ov, &t1, &tau);
        let got = ri_ladder_contract(&b_vv, &b_ov, &t1, &tau, nv).unwrap();
        let d = max_abs_diff(&want, &got);
        let scale = want.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        println!("RI ladder vs dense: max|Δ| = {d:.3e} (scale {scale:.3e})");
        assert!(
            scale > 1e-3,
            "reference is ~zero ({scale:.3e}) — the check is vacuous"
        );
        assert!(
            d < 1e-11 * scale.max(1.0),
            "RI ladder deviates from dense by {d:.3e}"
        );
    }

    /// EVERY block width must give the same answer, including widths that do NOT
    /// divide `nv` (the ragged-final-block case) and the degenerate `blk = 1`.
    ///
    /// A silent truncation of the last partial block is the classic failure mode
    /// here: with `nv = 7` and `blk = 3` the final block is 1 wide, and dropping
    /// it would leave `Z[:,:,6,:]` zero while every other element stayed right.
    #[test]
    fn ri_ladder_is_invariant_to_block_width_including_ragged() {
        let (naux, no, nv) = (9, 2, 7); // nv=7 is prime: blk∈{2,3,4,5,6} are ALL ragged
        let (b_vv, b_ov, t1, tau) = random_inputs(naux, no, nv);
        let want = dense_reference(&b_vv, &b_ov, &t1, &tau);
        let scale = want.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(scale > 1e-3, "reference is ~zero — the check is vacuous");
        for blk in 1..=nv {
            let got = ri_ladder_contract(&b_vv, &b_ov, &t1, &tau, blk).unwrap();
            let d = max_abs_diff(&want, &got);
            assert!(
                d < 1e-11 * scale.max(1.0),
                "block width {blk} (nv={nv}, ragged tail {}) deviates by {d:.3e}",
                nv % blk
            );
            // Explicitly: the LAST `a` slice must carry signal, not zeros. A
            // dropped final block passes a mean-error check but fails this.
            let last: f64 = (0..no)
                .flat_map(|i| (0..no).flat_map(move |j| (0..nv).map(move |b| (i, j, b))))
                .map(|(i, j, b)| got[[i, j, nv - 1, b]].abs())
                .fold(0.0f64, f64::max);
            assert!(last > 1e-6, "blk={blk}: last a-slice is ~zero ({last:.3e})");
        }
    }

    /// Block width changes the answer only in the last bits — and this test
    /// pins HOW MUCH, because the honest answer is "not bitwise identical".
    ///
    /// `blk` sets the `M` dimension of the wide GEMM
    /// `(alen·nv, naux) · (naux, nv·nv)`. ndarray's `blas` feature is on, so
    /// these GEMMs run on OpenBLAS, whose kernel choice and blocking depend on
    /// the matrix shape (including `M`) and on the CPU kernel selected at run
    /// time. The order of the sum over `P` is therefore not guaranteed to be
    /// the same for different block widths, and bit-identity across widths is
    /// not promised.
    ///
    /// What we can and do guarantee is that the difference stays at rounding
    /// level (`<1e-14` relative), i.e. block width is a performance knob and
    /// never a physics knob. If a future change makes this drift materially,
    /// that is a real regression in the blocking, and this test catches it.
    #[test]
    fn ri_ladder_block_width_differs_only_at_rounding_level() {
        let (naux, no, nv) = (9, 2, 7);
        let (b_vv, b_ov, t1, tau) = random_inputs(naux, no, nv);
        let base = ri_ladder_contract(&b_vv, &b_ov, &t1, &tau, nv).unwrap();
        let scale = base.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(scale > 1e-3, "reference is ~zero — vacuous");
        let mut worst = 0.0f64;
        for blk in 1..=nv {
            let got = ri_ladder_contract(&b_vv, &b_ov, &t1, &tau, blk).unwrap();
            worst = worst.max(max_abs_diff(&base, &got));
        }
        println!("block-width reassociation: worst max|Δ| = {worst:.3e} (scale {scale:.3e})");
        assert!(
            worst < 1e-14 * scale,
            "block width perturbs the result by {worst:.3e} — far above rounding \
             ({:.3e}); the blocking is not just reassociating, it is changing the math",
            1e-14 * scale
        );
    }

    /// `Bx` must equal its defining sum element-by-element.
    #[test]
    fn build_bx_matches_definition() {
        let (naux, no, nv) = (7, 3, 4);
        let (_, b_ov, t1, _) = random_inputs(naux, no, nv);
        let bx = build_bx(&b_ov, &t1, naux, no, nv).unwrap();
        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for p in 0..naux {
            for b in 0..nv {
                for d in 0..nv {
                    let mut want = 0.0;
                    for k in 0..no {
                        want -= b_ov[[p, k, d]] * t1[[k, b]];
                    }
                    scale = scale.max(want.abs());
                    worst = worst.max((bx[[p, b, d]] - want).abs());
                }
            }
        }
        assert!(scale > 1e-3, "Bx reference is ~zero — vacuous");
        assert!(
            worst < 1e-13,
            "Bx deviates from its definition by {worst:.3e}"
        );
    }

    /// With `t1 = 0` the ladder must reduce to the bare `(ac|bd)·tau` contraction.
    /// This isolates the `Bx` cross terms: if either were mis-signed or dropped,
    /// the t1=0 case would still pass while the general case failed — so this is
    /// a *decomposition* check, not a duplicate of the main test.
    #[test]
    fn ri_ladder_with_zero_t1_is_the_bare_ppl() {
        let (naux, no, nv) = (8, 2, 4);
        let (b_vv, b_ov, _, tau) = random_inputs(naux, no, nv);
        let t1 = ArrayD::<f64>::zeros(IxDyn(&[no, nv]));
        let got = ri_ladder_contract(&b_vv, &b_ov, &t1, &tau, 3).unwrap();
        let mut want = ArrayD::<f64>::zeros(IxDyn(&[no, no, nv, nv]));
        let mut scale = 0.0f64;
        for i in 0..no {
            for j in 0..no {
                for a in 0..nv {
                    for b in 0..nv {
                        let mut acc = 0.0;
                        for c in 0..nv {
                            for d in 0..nv {
                                let mut acbd = 0.0;
                                for p in 0..naux {
                                    acbd += b_vv[[p, a, c]] * b_vv[[p, b, d]];
                                }
                                acc += acbd * tau[[i, j, c, d]];
                            }
                        }
                        want[[i, j, a, b]] = acc;
                        scale = scale.max(acc.abs());
                    }
                }
            }
        }
        assert!(scale > 1e-3, "vacuous");
        let d = max_abs_diff(&want, &got);
        assert!(
            d < 1e-11 * scale,
            "t1=0 ladder deviates from bare (ac|bd)·tau by {d:.3e}"
        );
    }

    /// Shape mismatches must be typed errors, not panics — these are reachable
    /// from user-facing config (a wrong frozen_core, a mismatched aux basis).
    #[test]
    fn shape_mismatches_are_typed_errors() {
        let (naux, no, nv) = (5, 2, 3);
        let (b_vv, b_ov, t1, tau) = random_inputs(naux, no, nv);
        let bad_tau = ArrayD::<f64>::zeros(IxDyn(&[no, no, nv, nv + 1]));
        assert!(ri_ladder_contract(&b_vv, &b_ov, &t1, &bad_tau, 2).is_err());
        let bad_t1 = ArrayD::<f64>::zeros(IxDyn(&[no + 1, nv]));
        assert!(ri_ladder_contract(&b_vv, &b_ov, &bad_t1, &tau, 2).is_err());
        assert!(ri_ladder_contract(&b_vv, &b_ov, &t1, &tau, 0).is_err());
    }
}
