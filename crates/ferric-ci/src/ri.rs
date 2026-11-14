//! Resolution-of-the-identity (density-fitted) two-electron integrals for
//! CASSCF and SC-NEVPT2.
//!
//! With the Coulomb metric `V_PQ = (P|Q)` of an auxiliary basis and its
//! Cholesky factor `V = L Lᵀ`, the fitted AO tensor is
//!
//! ```text
//!   B[P, mn] = Σ_Q (L⁻¹)_PQ (Q|mn),      (mn|ls) ≈ Σ_P B[P, mn] B[P, ls],
//! ```
//!
//! held once, `naux x nbas²` doubles. Every integral class the CASSCF and
//! SC-NEVPT2 drivers read is a GEMM on it:
//!
//! * the half transform `(P|ab) = Σ_mn C_l[m,a] C_r[n,b] B[P, mn]`
//!   ([`RiTensor::half`]): one GEMM over `n` for all `P`, then one GEMM over
//!   `m` per `P`;
//! * an MO block `(ab|cd) = Σ_P (P|ab) (P|cd)`: one GEMM over `P`;
//! * `J[D]_mn = Σ_P B[P, mn] Σ_ls B[P, ls] D_ls`: two GEMVs;
//! * `K[D]` for `D = X M Xᵀ` ([`RiTensor::jk_factored`]):
//!   `H[P, m, j] = Σ_n B[P, m, n] X[n, j]`, then
//!   `K_mn = Σ_{P j k} H[P, m, j] M_jk H[P, n, k]`, one GEMM over `(P, j)`.
//!   A general symmetric `D` ([`RiTensor::jk`]) contracts `B D B` per block of
//!   auxiliary functions.
//!
//! Each output element is written by one task and every reduction runs in a
//! fixed order, so the result does not depend on the thread count. The
//! fitted tensor reconstructs an AO block ([`RiTensor::reconstruct_ao_eri`],
//! `BᵀB`), the input on which the dense route reproduces the RI route to
//! rounding (`crates/ferric-ci/tests/casscf_ri.rs`).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex;
use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array2, ArrayView2};
use rayon::prelude::*;

/// Auxiliary functions per block of the general `K[D]` contraction
/// ([`RiTensor::jk`]); the blocks are summed in order.
const K_AUX_BLOCK: usize = 32;

/// The registry's primary name of the bundled basis `name` resolves to
/// (case-insensitive, aliases included). Any other name is an error: the
/// auxiliary basis of the RI route is always a bundled set.
pub fn resolve_aux_name(name: &str) -> Result<&'static str, String> {
    let stem = basis::bundled_stem(name).ok_or_else(|| {
        format!(
            "auxbasis {name:?} is not a bundled basis set (known: {})",
            basis::bundled_names().join(", ")
        )
    })?;
    basis::bundled_names()
        .into_iter()
        .find(|n| basis::bundled_stem(n) == Some(stem))
        .ok_or_else(|| format!("auxbasis {name:?}: no primary registry name for {stem}"))
}

/// The auxiliary basis `aux` prepared on `mol`.
pub(crate) fn aux_prepared(mol: &Molecule, aux: &str) -> Result<PreparedBasis, FerricError> {
    let bs = basis::bundled(aux)?;
    PreparedBasis::new(mol, &bs)
}

/// The fitted AO tensor `B[P, mn]` of one orbital and auxiliary basis.
pub struct RiTensor {
    /// `(naux, nbas²)`, row `P`, column `m nbas + n`.
    b: Array2<f64>,
    nbas: usize,
}

impl RiTensor {
    /// `B = L⁻¹ (Q|mn)` for the orbital basis `prep` and the bundled
    /// auxiliary basis `aux` on `mol`. The 3-index integrals stream from a
    /// [`ThreeIndexSource`] under `budget_bytes` and are dressed block by
    /// block into the resident tensor.
    pub fn build(
        mol: &Molecule,
        prep: &PreparedBasis,
        aux: &str,
        budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        let dfbs = aux_prepared(mol, aux)?;
        let op = Operator::coulomb();
        let v2c = threeindex::coulomb_metric_2c(op, &dfbs)?;
        let linv = ferric_mp2::rimp2::metric_inverse_sqrt(&v2c, op)?;
        drop(v2c);
        let nbas = prep.nbasis();
        let naux = dfbs.nbasis();
        let n2 = nbas * nbas;
        let mut b = Array2::<f64>::zeros((naux, n2));
        let mut src = ThreeIndexSource::build(op, prep, &dfbs, budget_bytes)?;
        src.for_each_block(|blk| {
            let qb = blk.data.shape()[0];
            let (q0, q1) = (blk.p0, blk.p0 + qb);
            let raw = blk
                .data
                .into_shape_with_order((qb, n2))
                .map_err(|e| FerricError::General(format!("RI (Q|mn) block shape: {e}")))?;
            // L⁻¹ is lower triangular: rows P < q0 get nothing from this block.
            let lb = linv.slice(s![q0.., q0..q1]);
            let mut out = b.slice_mut(s![q0.., ..]);
            with_blas_threads(opt_in_blas_threads(), || {
                general_mat_mul(1.0, &lb, &raw, 1.0, &mut out)
            });
            Ok(())
        })?;
        Ok(RiTensor { b, nbas })
    }

    /// Auxiliary functions.
    pub fn naux(&self) -> usize {
        self.b.nrows()
    }

    /// Orbital basis functions.
    pub fn nbas(&self) -> usize {
        self.nbas
    }

    /// `B` as `(naux, nbas²)`.
    pub fn tensor(&self) -> ArrayView2<'_, f64> {
        self.b.view()
    }

    /// The AO block `(mn|ls) = Σ_P B[P, mn] B[P, ls]`, row-major `nbas⁴`.
    pub fn reconstruct_ao_eri(&self) -> Vec<f64> {
        let g = with_blas_threads(opt_in_blas_threads(), || self.b.t().dot(&self.b));
        g.as_standard_layout().iter().copied().collect()
    }

    /// `B` as `(naux nbas, nbas)`: row `(P, m)`, column `n`.
    fn rows(&self) -> ArrayView2<'_, f64> {
        ArrayView2::from_shape(
            (self.naux() * self.nbas, self.nbas),
            self.b.as_slice().unwrap(),
        )
        .expect("B is standard layout (naux, nbas²) by construction")
    }

    /// `H[(P, m), b] = Σ_n B[P, m, n] C_r[n, b]`, `(naux nbas, n_r)`.
    pub(crate) fn right(&self, cr: ArrayView2<'_, f64>) -> Array2<f64> {
        let h = with_blas_threads(opt_in_blas_threads(), || self.rows().dot(&cr));
        h.as_standard_layout().into_owned()
    }

    /// `(P|ab) = Σ_m C_l[m, a] H[(P, m), b]` from `H = `[`Self::right`],
    /// `(naux, n_l n_r)`, column `a n_r + b`. One GEMM per `P`, in parallel.
    pub(crate) fn left(&self, h: &Array2<f64>, cl: ArrayView2<'_, f64>) -> Array2<f64> {
        let (nbas, naux) = (self.nbas, self.naux());
        let nr = h.ncols();
        let nl = cl.ncols();
        let clt = cl.t().as_standard_layout().into_owned();
        let hs = h.as_slice().expect("H is standard layout by construction");
        let mut out = Array2::<f64>::zeros((naux, nl * nr));
        if nl * nr == 0 {
            return out;
        }
        out.as_slice_mut()
            .expect("fresh array is standard layout")
            .par_chunks_mut(nl * nr)
            .zip(hs.par_chunks(nbas * nr))
            .for_each(|(o, hp)| {
                let hp = ArrayView2::from_shape((nbas, nr), hp)
                    .expect("one H_P slab is (nbas, n_r) by construction");
                let r = clt.dot(&hp);
                for (d, s) in o.iter_mut().zip(r.iter()) {
                    *d = *s;
                }
            });
        out
    }

    /// `(P|ab) = Σ_mn C_l[m, a] C_r[n, b] B[P, mn]`, `(naux, n_l n_r)`.
    pub fn half(&self, cl: ArrayView2<'_, f64>, cr: ArrayView2<'_, f64>) -> Array2<f64> {
        let h = self.right(cr);
        self.left(&h, cl)
    }

    /// `J[D]_mn = Σ_P B[P, mn] Σ_ls B[P, ls] D_ls`.
    fn coulomb(&self, d: &Array2<f64>) -> Array2<f64> {
        let n = self.nbas;
        let dv = d.as_standard_layout();
        let dv = ndarray::ArrayView1::from(dv.as_slice().expect("standard layout"));
        let fit = self.b.dot(&dv);
        let j = self.b.t().dot(&fit);
        j.into_shape_with_order((n, n))
            .expect("J has nbas² elements")
            .as_standard_layout()
            .into_owned()
    }

    /// `J[D]`, `K[D]` for a general symmetric `D`:
    /// `K = Σ_P B_P D B_P` over fixed blocks of `K_AUX_BLOCK` auxiliary
    /// functions, summed in order (the oracle path of the orbital Hessian).
    pub fn jk(&self, d: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let n = self.nbas;
        let j = self.coulomb(d);
        let rows = self.rows();
        // T[(P, m), l] = Σ_s B[P, m, s] D[s, l].
        let t = with_blas_threads(opt_in_blas_threads(), || rows.dot(d));
        let t = t.as_standard_layout().into_owned();
        let ts = t.as_slice().expect("standard layout");
        let naux = self.naux();
        let blocks: Vec<(usize, usize)> = (0..naux)
            .step_by(K_AUX_BLOCK)
            .map(|p0| (p0, (p0 + K_AUX_BLOCK).min(naux)))
            .collect();
        let parts: Vec<Array2<f64>> = blocks
            .par_iter()
            .map(|&(p0, p1)| {
                let mut k = Array2::<f64>::zeros((n, n));
                for p in p0..p1 {
                    let tp = ArrayView2::from_shape((n, n), &ts[p * n * n..(p + 1) * n * n])
                        .expect("T_P is (nbas, nbas)");
                    let bp = rows.slice(s![p * n..(p + 1) * n, ..]);
                    general_mat_mul(1.0, &tp, &bp, 1.0, &mut k);
                }
                k
            })
            .collect();
        let mut k = Array2::<f64>::zeros((n, n));
        for part in parts {
            k += &part;
        }
        (j, k)
    }

    /// `J[D]`, `K[D]` for `D = X M Xᵀ`, `X` `(nbas, k)` and `M` `(k, k)`
    /// symmetric: `K_mn = Σ_{P j} G[P, m, j] H[P, n, j]` with
    /// `H = B X`, `G = H M`, as one GEMM over `(P, j)`.
    pub fn jk_factored(
        &self,
        x: ArrayView2<'_, f64>,
        m: ArrayView2<'_, f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let n = self.nbas;
        let kk = x.ncols();
        let d = x.dot(&m).dot(&x.t());
        let j = self.coulomb(&d);
        if kk == 0 {
            return (j, Array2::zeros((n, n)));
        }
        let naux = self.naux();
        let h = self.right(x);
        let g = with_blas_threads(opt_in_blas_threads(), || h.dot(&m));
        let g = g.as_standard_layout().into_owned();
        // [(P, m), j] -> [m, (P, j)].
        let permute = |a: &Array2<f64>| {
            let a = a.as_slice().expect("standard layout");
            let mut out = Array2::<f64>::zeros((n, naux * kk));
            out.as_slice_mut()
                .expect("standard layout")
                .par_chunks_mut(naux * kk)
                .enumerate()
                .for_each(|(mu, row)| {
                    for p in 0..naux {
                        let src = &a[(p * n + mu) * kk..(p * n + mu + 1) * kk];
                        row[p * kk..(p + 1) * kk].copy_from_slice(src);
                    }
                });
            out
        };
        let gp = permute(&g);
        drop(g);
        let hp = permute(&h);
        drop(h);
        let k = with_blas_threads(opt_in_blas_threads(), || gp.dot(&hp.t()));
        (j, k.as_standard_layout().into_owned())
    }

    /// `(ab|cd) = Σ_P (P|ab) (P|cd)` from two half transforms, flat
    /// `[(a n_b + b) n_cd + (c n_d + d)]`.
    pub(crate) fn contract(x: &Array2<f64>, y: &Array2<f64>) -> Vec<f64> {
        let out = with_blas_threads(opt_in_blas_threads(), || x.t().dot(y));
        let out = out.as_standard_layout().into_owned();
        let (v, off) = out.into_raw_vec_and_offset();
        debug_assert_eq!(off.unwrap_or(0), 0);
        v
    }

    /// `(pu|vw)`, flat `[((p n_a + u) n_a + v) n_a + w]`, `p` over the
    /// columns of `c1`, `u, v, w` over those of `ca`.
    pub(crate) fn paaa(&self, c1: ArrayView2<'_, f64>, ca: ArrayView2<'_, f64>) -> Vec<f64> {
        let h = self.right(ca);
        let pu = self.left(&h, c1);
        let vw = self.left(&h, ca);
        drop(h);
        Self::contract(&pu, &vw)
    }

    /// `(pq|vw)` flat `[((p n + q) n_a + v) n_a + w]` and `(pu|qv)` flat
    /// `[((p n_a + u) n + q) n_a + v]`, `p, q` over the `n` columns of `c`,
    /// `u, v, w` over those of `ca`.
    pub(crate) fn ppaa_papa(
        &self,
        c: ArrayView2<'_, f64>,
        ca: ArrayView2<'_, f64>,
    ) -> (Vec<f64>, Vec<f64>) {
        let ha = self.right(ca);
        let vw = self.left(&ha, ca);
        let pu = self.left(&ha, c);
        drop(ha);
        let papa = Self::contract(&pu, &pu);
        drop(pu);
        let pq = self.half(c, c);
        let ppaa = Self::contract(&pq, &vw);
        (ppaa, papa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A random symmetric-in-(m,n) tensor and orbitals: every GEMM route
    /// equals its explicit loops.
    fn toy(naux: usize, n: usize) -> RiTensor {
        let mut st = 7u64;
        let mut r = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        let mut b = Array2::<f64>::zeros((naux, n * n));
        for p in 0..naux {
            for m in 0..n {
                for k in 0..=m {
                    let v = r();
                    b[(p, m * n + k)] = v;
                    b[(p, k * n + m)] = v;
                }
            }
        }
        RiTensor { b, nbas: n }
    }

    fn mat(r: usize, c: usize, seed: u64) -> Array2<f64> {
        Array2::from_shape_fn((r, c), |(i, j)| {
            (((i * 31 + j * 17) as u64 ^ seed) as f64 * 0.37).sin()
        })
    }

    fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn half_transforms_and_jk_match_explicit_loops() {
        let (naux, n) = (11, 7);
        let t = toy(naux, n);
        let eri = t.reconstruct_ao_eri();
        let g = |m: usize, k: usize, l: usize, s: usize| eri[((m * n + k) * n + l) * n + s];
        let c = mat(n, 5, 3);
        let ca = mat(n, 3, 9);
        let paaa = t.paaa(c.view(), ca.view());
        let mut want = Vec::new();
        for p in 0..5 {
            for u in 0..3 {
                for v in 0..3 {
                    for w in 0..3 {
                        let mut acc = 0.0;
                        for m in 0..n {
                            for k in 0..n {
                                for l in 0..n {
                                    for s in 0..n {
                                        acc += c[(m, p)]
                                            * ca[(k, u)]
                                            * ca[(l, v)]
                                            * ca[(s, w)]
                                            * g(m, k, l, s);
                                    }
                                }
                            }
                        }
                        want.push(acc);
                    }
                }
            }
        }
        assert!(maxdiff(&paaa, &want) < 1e-12, "(pu|vw)");

        let (ppaa, papa) = t.ppaa_papa(c.view(), ca.view());
        let tr = |c4: [&Array2<f64>; 4], idx: [usize; 4]| {
            let mut acc = 0.0;
            for m in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        for s in 0..n {
                            acc += c4[0][(m, idx[0])]
                                * c4[1][(k, idx[1])]
                                * c4[2][(l, idx[2])]
                                * c4[3][(s, idx[3])]
                                * g(m, k, l, s);
                        }
                    }
                }
            }
            acc
        };
        for p in 0..5 {
            for q in 0..5 {
                for v in 0..3 {
                    for w in 0..3 {
                        let got = ppaa[((p * 5 + q) * 3 + v) * 3 + w];
                        let e = tr([&c, &c, &ca, &ca], [p, q, v, w]);
                        assert!((got - e).abs() < 1e-12, "(pq|vw)");
                        let got = papa[((p * 3 + v) * 5 + q) * 3 + w];
                        let e = tr([&c, &ca, &c, &ca], [p, v, q, w]);
                        assert!((got - e).abs() < 1e-12, "(pu|qv)");
                    }
                }
            }
        }

        // General and factored J/K against the reconstructed block.
        let x = mat(n, 3, 5);
        let m = {
            let a = mat(3, 3, 1);
            &a + &a.t()
        };
        let d = x.dot(&m).dot(&x.t());
        let (j, k) = t.jk(&d);
        let (jf, kf) = t.jk_factored(x.view(), m.view());
        for a in 0..n {
            for b in 0..n {
                let (mut je, mut ke) = (0.0, 0.0);
                for l in 0..n {
                    for s in 0..n {
                        je += g(a, b, l, s) * d[(l, s)];
                        ke += g(a, l, b, s) * d[(l, s)];
                    }
                }
                for (got, e, what) in [
                    (j[(a, b)], je, "J"),
                    (k[(a, b)], ke, "K"),
                    (jf[(a, b)], je, "J factored"),
                    (kf[(a, b)], ke, "K factored"),
                ] {
                    assert!((got - e).abs() < 1e-12, "{what} {a} {b}: {got} vs {e}");
                }
            }
        }
    }

    #[test]
    fn aux_names_resolve_to_the_registry_primary_name() {
        assert_eq!(
            resolve_aux_name("DEF2-UNIVERSAL-JKFIT"),
            Ok("def2-universal-jkfit")
        );
        let e = resolve_aux_name("no-such-aux").unwrap_err();
        assert!(e.contains("not a bundled basis set"), "{e}");
    }
}
