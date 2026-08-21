//! Shared spin-orbital MO integral helpers: build dressed RI 3-index blocks and
//! expand spatial chemist integrals into antisymmetrized spin-orbital `<pq||rs>`
//! blocks (spin convention 2k=alpha, 2k+1=beta). Used by spin-orbital MP3 and CCD.

use crate::mo_transform::dress_3index;
use ferric_tensors::{Axis, Tensor};
use ndarray::{Array2, ArrayD, IxDyn};

/// Build a dressed RI 3-index MO tensor `B^P_{pq}` for a spatial block.
///
/// Delegates to [`dress_3index`] for the metric dressing, then wraps in a labeled `Tensor<3>`.
pub fn build_b(eri3_mo: &ndarray::Array3<f64>, v_inv_sqrt: &Array2<f64>, l1: Axis, l2: Axis) -> Tensor<3> {
    let b = dress_3index(eri3_mo, v_inv_sqrt);
    Tensor::new(b.into_dyn(), [Axis::Aux, l1, l2])
}

/// Spin of spin-orbital index `p`: 0=alpha (even), 1=beta (odd).
#[inline]
pub fn spin(p: usize) -> usize {
    p & 1
}
/// Spatial index of spin-orbital `p`.
#[inline]
pub fn spat(p: usize) -> usize {
    p >> 1
}

/// Antisymmetrized spin-orbital `<ij||ab>` (oovv) from spatial chemist `(ia|jb)`.
///
/// `g_iajb[i,a,j,b] = (ia|jb)`. Output shape `(2no, 2no, 2nv, 2nv)`.
#[allow(clippy::needless_range_loop)]
pub fn asym_oovv(g_iajb: &ArrayD<f64>, no: usize, nv: usize) -> ArrayD<f64> {
    let (no2, nv2) = (2 * no, 2 * nv);
    let mut out = ArrayD::zeros(IxDyn(&[no2, no2, nv2, nv2]));
    for i in 0..no2 {
        for j in 0..no2 {
            for a in 0..nv2 {
                for b in 0..nv2 {
                    // <ij|ab> = (ia|jb) with spin(i)=spin(a), spin(j)=spin(b)
                    let dir = if spin(i) == spin(a) && spin(j) == spin(b) {
                        g_iajb[[spat(i), spat(a), spat(j), spat(b)]]
                    } else {
                        0.0
                    };
                    // <ij|ba> = (ib|ja) with spin(i)=spin(b), spin(j)=spin(a)
                    let exc = if spin(i) == spin(b) && spin(j) == spin(a) {
                        g_iajb[[spat(i), spat(b), spat(j), spat(a)]]
                    } else {
                        0.0
                    };
                    out[[i, j, a, b]] = dir - exc;
                }
            }
        }
    }
    out
}

/// Antisymmetrized same-space `<pq||rs>` from spatial chemist `(pq|rs)`.
///
/// `g[p,q,r,s] = (pq|rs)`, all four indices in the SAME spatial space of size
/// `n` spatial / `2n` spin. `<pq||rs> = (pr|qs) - (ps|qr)`, i.e. index into `g`
/// as `g[p,r,q,s]` (direct) and `g[p,s,q,r]` (exchange). Used for vvvv and oooo.
#[allow(clippy::needless_range_loop)]
pub fn asym_same(g: &ArrayD<f64>, n: usize) -> ArrayD<f64> {
    let n2 = 2 * n;
    let mut out = ArrayD::zeros(IxDyn(&[n2, n2, n2, n2]));
    for p in 0..n2 {
        for q in 0..n2 {
            for r in 0..n2 {
                for s in 0..n2 {
                    // <pq|rs> = (pr|qs), spin(p)=spin(r) & spin(q)=spin(s)
                    let dir = if spin(p) == spin(r) && spin(q) == spin(s) {
                        g[[spat(p), spat(r), spat(q), spat(s)]]
                    } else {
                        0.0
                    };
                    // <pq|sr> = (ps|qr), spin(p)=spin(s) & spin(q)=spin(r)
                    let exc = if spin(p) == spin(s) && spin(q) == spin(r) {
                        g[[spat(p), spat(s), spat(q), spat(r)]]
                    } else {
                        0.0
                    };
                    out[[p, q, r, s]] = dir - exc;
                }
            }
        }
    }
    out
}

/// Antisymmetrized `<kb||cj>` (ovvo) for the particle-hole MP3 term.
///
/// k,j occupied; b,c virtual. Direct `<kb|cj> = (kc|bj)` from `g_kcbj[k,c,b,j]`;
/// exchange `<kb|jc> = (kj|bc)` from `g_kjbc[k,j,b,c]`.
/// Output shape `(2no, 2nv, 2nv, 2no)` indexed `[k,b,c,j]`.
#[allow(clippy::needless_range_loop)]
pub fn asym_ovvo(g_kcbj: &ArrayD<f64>, g_kjbc: &ArrayD<f64>, no: usize, nv: usize) -> ArrayD<f64> {
    let (no2, nv2) = (2 * no, 2 * nv);
    let mut out = ArrayD::zeros(IxDyn(&[no2, nv2, nv2, no2]));
    for k in 0..no2 {
        for b in 0..nv2 {
            for c in 0..nv2 {
                for j in 0..no2 {
                    // <kb|cj> = (kc|bj), spin(k)=spin(c) & spin(b)=spin(j)
                    let dir = if spin(k) == spin(c) && spin(b) == spin(j) {
                        g_kcbj[[spat(k), spat(c), spat(b), spat(j)]]
                    } else {
                        0.0
                    };
                    // <kb|jc> = (kj|bc), spin(k)=spin(j) & spin(b)=spin(c)
                    let exc = if spin(k) == spin(j) && spin(b) == spin(c) {
                        g_kjbc[[spat(k), spat(j), spat(b), spat(c)]]
                    } else {
                        0.0
                    };
                    out[[k, b, c, j]] = dir - exc;
                }
            }
        }
    }
    out
}

/// General antisymmetrized spin-orbital `<pq||rs>` from two spatial chemist
/// blocks. This is the reusable generalization of [`asym_oovv`] / [`asym_same`]
/// / [`asym_ovvo`] for arbitrary occupied/virtual index patterns.
///
/// In Dirac (physicist) notation `<pq||rs> = <pq|rs> - <pq|sr>` where
/// `<pq|rs> = (pr|qs)` and `<pq|sr> = (ps|qr)` (Mulliken/chemist).
///
/// - `g_dir[p,r,q,s] = (pr|qs)` — chemist integral feeding the direct term,
///   spatial shape `[np, nr, nq, ns]`. Nonzero only when `spin(p)==spin(r)` and
///   `spin(q)==spin(s)`.
/// - `g_exc[p,s,q,r] = (ps|qr)` — chemist integral feeding the exchange term,
///   spatial shape `[np, ns, nq, nr]`. Nonzero only when `spin(p)==spin(s)` and
///   `spin(q)==spin(r)`.
/// - `np,nq,nr,ns` are the *spatial* sizes of indices p,q,r,s respectively.
///
/// Output `<pq||rs>` has spin shape `[2np, 2nq, 2nr, 2ns]` indexed `[p,q,r,s]`.
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub fn asym_phys(
    g_dir: &ArrayD<f64>,
    g_exc: &ArrayD<f64>,
    np: usize,
    nq: usize,
    nr: usize,
    ns: usize,
) -> ArrayD<f64> {
    let (np2, nq2, nr2, ns2) = (2 * np, 2 * nq, 2 * nr, 2 * ns);
    let mut out = ArrayD::zeros(IxDyn(&[np2, nq2, nr2, ns2]));
    for p in 0..np2 {
        for q in 0..nq2 {
            for r in 0..nr2 {
                for s in 0..ns2 {
                    // <pq|rs> = (pr|qs)
                    let dir = if spin(p) == spin(r) && spin(q) == spin(s) {
                        g_dir[[spat(p), spat(r), spat(q), spat(s)]]
                    } else {
                        0.0
                    };
                    // <pq|sr> = (ps|qr)
                    let exc = if spin(p) == spin(s) && spin(q) == spin(r) {
                        g_exc[[spat(p), spat(s), spat(q), spat(r)]]
                    } else {
                        0.0
                    };
                    out[[p, q, r, s]] = dir - exc;
                }
            }
        }
    }
    out
}

/// Transpose a `[Aux, O, V]` RI block into `[Aux, V, O]` (i.e. B^P_{ai} = B^P_{ia}).
pub fn transpose_b(b_ov: &Tensor<3>) -> Tensor<3> {
    let out = b_ov.view().permuted_axes(IxDyn(&[0, 2, 1])).to_owned();
    Tensor::new(out, [Axis::Aux, Axis::V, Axis::O])
}
