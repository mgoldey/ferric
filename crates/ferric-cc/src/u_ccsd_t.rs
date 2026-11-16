//! Spin-blocked (T) correction for UCCSD on a UHF reference.
//!
//! The spin-orbital (T) of [`crate::ccsd_t`] is
//!
//! ```text
//! X(p,q,r; s,t,u) = sum_e t2[q,r,s,e] <tu||ep> - sum_m t2[p,m,t,u] <ms||qr>
//! Y(p,q,r; s,t,u) = t1[p,s] <tu||qr>
//! W = P(i/jk) P(a/bc) X,   V = P(i/jk) P(a/bc) Y,   D = e_i+e_j+e_k-e_a-e_b-e_c
//! E_(T) = 1/36 sum_{ijkabc} (W + V) W / D
//! ```
//!
//! This module evaluates exactly that formula, but only on the spin-allowed
//! blocks. `f = (W + V) W / D` is invariant under any permutation of the
//! occupied indices and under any permutation of the virtual indices, so the
//! ordered sum folds onto one canonical spin layout per class:
//!
//! ```text
//! class  occupied             virtual              weight
//! aaa    i<j<k   (a,a,a)      a,b,c all  (a,a,a)    1/6
//! aab    i<j a, k b           a,b all a, c b        1/2
//! abb    i a, j<k b           a a, b,c all b        1/2
//! bbb    i<j<k   (b,b,b)      all  (b,b,b)          1/6
//! ```
//!
//! The weight is (number of ordered spin-orbital triples in the class) / 36
//! divided by the number the canonical loop visits: for `aab`, 3! occupied
//! orderings times 3 positions of the beta virtual, over 36, is 1/2 (the
//! canonical loop visits each unordered alpha pair once and all alpha virtual
//! orderings). Derivation and the counting proof:
//! `crates/ferric-cc/theory.md` ("Unrestricted CCSD and CCSD(T)") and
//! `scripts/cc_uccsd_spin_blocks_sympy.py`.
//!
//! `X` and `Y` for a fixed occupied triple are built block by block from the
//! UCCSD amplitude blocks and per-spin chemist integrals; a term whose spin
//! labels cannot match is skipped. The loop streams over the canonical
//! occupied triples of all four classes, in parallel with a serial fold in a
//! fixed order, so the result does not depend on the thread count.

use crate::u_ccsd::{chem, UAmplitudes, UCcsdResult, URiBlocks};
use crate::CcConfig;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::with_blas_threads;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;
use ndarray::{s, Array2, Array3, Array4, ArrayView2};
use rayon::prelude::*;

const A: usize = 0;
const B: usize = 1;

/// The (T) energy and its four occupied spin classes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UTriples {
    /// `E_(T) = aaa + aab + abb + bbb` (Hartree).
    pub e_t: f64,
    /// Three alpha occupied spin orbitals.
    pub aaa: f64,
    /// Two alpha, one beta.
    pub aab: f64,
    /// One alpha, two beta.
    pub abb: f64,
    /// Three beta.
    pub bbb: f64,
}

/// A spin orbital: `(spin, index within its spin block)`.
type So = (usize, usize);

/// Per-spin chemist blocks the triples kernel slices, laid out so that a
/// fixed occupied index gives a contiguous matrix.
struct TriplesInts {
    /// `g1[st][su][p, e, t, u] = (te|up)`; `e,t` of spin `st`, `u,p` of `su`.
    g1: [[Array4<f64>; 2]; 2],
    /// `g2[sm][ss][q, r, m, s] = (mq|sr)`; `m,q` of spin `sm`, `s,r` of `ss`.
    g2: [[Array4<f64>; 2]; 2],
    /// `g3[st][su][q, r, t, u] = (tq|ur)`; `t,q` of spin `st`, `u,r` of `su`.
    g3: [[Array4<f64>; 2]; 2],
}

fn permute4(x: ndarray::ArrayD<f64>, axes: [usize; 4]) -> Array4<f64> {
    x.into_dimensionality::<ndarray::Ix4>()
        .expect("4-D")
        .permuted_axes(axes)
        .as_standard_layout()
        .into_owned()
}

impl TriplesInts {
    fn build(r: &URiBlocks) -> Self {
        let vo = [r.b_vo(A), r.b_vo(B)];
        let g1 = |st: usize, su: usize| {
            // chem(vv[st], vo[su]) = [t,e,u,p] -> [p,e,t,u]
            permute4(chem(&r.b_vv[st], &vo[su]), [3, 1, 0, 2])
        };
        let g2 = |sm: usize, ss: usize| {
            // chem(oo[sm], vo[ss]) = [m,q,s,r] -> [q,r,m,s]
            permute4(chem(&r.b_oo[sm], &vo[ss]), [1, 3, 0, 2])
        };
        let g3 = |st: usize, su: usize| {
            // chem(vo[st], vo[su]) = [t,q,u,r] -> [q,r,t,u]
            permute4(chem(&vo[st], &vo[su]), [1, 3, 0, 2])
        };
        Self {
            g1: [[g1(A, A), g1(A, B)], [g1(B, A), g1(B, B)]],
            g2: [[g2(A, A), g2(A, B)], [g2(B, A), g2(B, B)]],
            g3: [[g3(A, A), g3(A, B)], [g3(B, A), g3(B, B)]],
        }
    }

    fn bytes(no: [usize; 2], nv: [usize; 2]) -> usize {
        let mut n = 0usize;
        for x in 0..2 {
            for y in 0..2 {
                n = n
                    .saturating_add(no[y] * nv[x] * nv[x] * nv[y])
                    .saturating_add(no[x] * no[y] * no[x] * nv[y])
                    .saturating_add(no[x] * no[y] * nv[x] * nv[y]);
            }
        }
        n.saturating_mul(8)
    }
}

/// The amplitude blocks as spin-orbital slices.
struct T2View<'a> {
    t1: [&'a Array2<f64>; 2],
    t2aa: &'a Array4<f64>,
    t2ab: &'a Array4<f64>,
    t2bb: &'a Array4<f64>,
}

impl<'a> T2View<'a> {
    fn new(t: &'a UAmplitudes) -> Self {
        Self {
            t1: [&t.t1a, &t.t1b],
            t2aa: &t.t2aa,
            t2ab: &t.t2ab,
            t2bb: &t.t2bb,
        }
    }

    /// `t2[q, r, s, e]` over `(s, e)` with spins `(ss, se)`; `None` if the
    /// spins cannot match.
    fn pair(&self, q: So, r: So, ss: usize, se: usize) -> Option<Array2<f64>> {
        let ((sq, iq), (sr, ir)) = (q, r);
        let o = |x: ArrayView2<f64>| x.to_owned();
        match (sq, sr, ss, se) {
            (A, A, A, A) => Some(o(self.t2aa.slice(s![iq, ir, .., ..]))),
            (B, B, B, B) => Some(o(self.t2bb.slice(s![iq, ir, .., ..]))),
            (A, B, A, B) => Some(o(self.t2ab.slice(s![iq, ir, .., ..]))),
            (A, B, B, A) => Some(-o(self.t2ab.slice(s![iq, ir, .., ..]).t())),
            (B, A, A, B) => Some(-o(self.t2ab.slice(s![ir, iq, .., ..]))),
            (B, A, B, A) => Some(o(self.t2ab.slice(s![ir, iq, .., ..]).t())),
            _ => None,
        }
    }

    /// `t2[p, m, t, u]` over `(m, t*u)` with spins `(sm, st, su)`, as a
    /// matrix; `None` if the spins cannot match.
    fn row(&self, p: So, sm: usize, st: usize, su: usize) -> Option<Array2<f64>> {
        let (sp, ip) = p;
        let flat = |x: Array3<f64>| {
            let (m, t, u) = x.dim();
            x.into_shape_with_order((m, t * u))
                .expect("standard layout")
        };
        let own = |x: ndarray::ArrayView3<f64>| x.as_standard_layout().into_owned();
        let own_tr = |x: ndarray::ArrayView3<f64>| {
            x.permuted_axes([0, 2, 1]).as_standard_layout().into_owned()
        };
        match (sp, sm, st, su) {
            (A, A, A, A) => Some(flat(own(self.t2aa.slice(s![ip, .., .., ..])))),
            (B, B, B, B) => Some(flat(own(self.t2bb.slice(s![ip, .., .., ..])))),
            (A, B, A, B) => Some(flat(own(self.t2ab.slice(s![ip, .., .., ..])))),
            (A, B, B, A) => Some(-flat(own_tr(self.t2ab.slice(s![ip, .., .., ..])))),
            (B, A, A, B) => Some(-flat(own(self.t2ab.slice(s![.., ip, .., ..])))),
            (B, A, B, A) => Some(flat(own_tr(self.t2ab.slice(s![.., ip, .., ..])))),
            _ => None,
        }
    }
}

/// Swap two axes of a 3-index block, owned standard layout.
fn swap(x: &Array3<f64>, axes: [usize; 3]) -> Array3<f64> {
    x.view()
        .permuted_axes(axes)
        .as_standard_layout()
        .into_owned()
}

struct Kernel<'a> {
    ints: &'a TriplesInts,
    t: T2View<'a>,
    eo: [&'a [f64]; 2],
    ev: [&'a [f64]; 2],
    nv: [usize; 2],
}

impl Kernel<'_> {
    /// `X(p,q,r; s,t,u)` over the virtual spins `(ss, st, su)`.
    fn x_block(&self, p: So, q: So, r: So, ss: usize, st: usize, su: usize) -> Array3<f64> {
        let (nvs, nvt, nvu) = (self.nv[ss], self.nv[st], self.nv[su]);
        let mut out = Array2::<f64>::zeros((nvs, nvt * nvu));
        let ((sp, ip), (sq, iq), (sr, ir)) = (p, q, r);
        // term 1: sum_e t2[q,r,s,e] <tu||ep>
        for se in [A, B] {
            let Some(t2qr) = self.t.pair(q, r, ss, se) else {
                continue;
            };
            if st == se && su == sp {
                // (te|up) at g1[st][su][p] = [e, t, u]
                let g = self.ints.g1[st][su].slice(s![ip, .., .., ..]);
                let g = g
                    .into_shape_with_order((nvt, nvt * nvu))
                    .expect("contiguous");
                out += &t2qr.dot(&g);
            }
            if st == sp && su == se {
                // -(tp|ue) = -(ue|tp) at g1[su][st][p] = [e, u, t]
                let g = self.ints.g1[su][st].slice(s![ip, .., .., ..]);
                let g = g
                    .into_shape_with_order((nvu, nvu * nvt))
                    .expect("contiguous");
                let x = t2qr
                    .dot(&g)
                    .into_shape_with_order((nvs, nvu, nvt))
                    .expect("shape");
                let x = swap(&x, [0, 2, 1])
                    .into_shape_with_order((nvs, nvt * nvu))
                    .expect("shape");
                out -= &x;
            }
        }
        // term 2: - sum_m t2[p,m,t,u] <ms||qr>
        for sm in [A, B] {
            let Some(t2p) = self.t.row(p, sm, st, su) else {
                continue;
            };
            if sm == sq && ss == sr {
                // (mq|sr) at g2[sm][ss][q, r] = [m, s]
                let g = self.ints.g2[sm][ss].slice(s![iq, ir, .., ..]);
                out -= &g.t().dot(&t2p);
            }
            if sm == sr && ss == sq {
                // -(mr|sq) at g2[sm][ss][r, q] = [m, s]
                let g = self.ints.g2[sm][ss].slice(s![ir, iq, .., ..]);
                out += &g.t().dot(&t2p);
            }
        }
        out.into_shape_with_order((nvs, nvt, nvu)).expect("shape")
    }

    /// `Y(p,q,r; s,t,u) = t1[p,s] <tu||qr>` over the virtual spins.
    fn y_block(&self, p: So, q: So, r: So, ss: usize, st: usize, su: usize) -> Array3<f64> {
        let (nvs, nvt, nvu) = (self.nv[ss], self.nv[st], self.nv[su]);
        let mut out = Array3::<f64>::zeros((nvs, nvt, nvu));
        let ((sp, ip), (sq, iq), (sr, ir)) = (p, q, r);
        if sp != ss {
            return out;
        }
        let mut g = Array2::<f64>::zeros((nvt, nvu));
        if st == sq && su == sr {
            g += &self.ints.g3[st][su].slice(s![iq, ir, .., ..]);
        }
        if st == sr && su == sq {
            g -= &self.ints.g3[st][su].slice(s![ir, iq, .., ..]);
        }
        let t1p = self.t.t1[ss].row(ip);
        for (a, mut slab) in out.outer_iter_mut().enumerate() {
            slab.scaled_add(t1p[a], &g);
        }
        out
    }

    /// `P(i/jk) P(a/bc)` of `X` or `Y` on the canonical virtual layout `sv`.
    fn antisym(
        &self,
        i: So,
        j: So,
        k: So,
        sv: [usize; 3],
        f: impl Fn(&Self, So, So, So, usize, usize, usize) -> Array3<f64>,
    ) -> Array3<f64> {
        let [sa, sb, sc] = sv;
        let mut out = Array3::<f64>::zeros((self.nv[sa], self.nv[sb], self.nv[sc]));
        for ((p, q, r), sgn) in [((i, j, k), 1.0), ((j, i, k), -1.0), ((k, j, i), -1.0)] {
            out.scaled_add(sgn, &f(self, p, q, r, sa, sb, sc));
            out.scaled_add(-sgn, &swap(&f(self, p, q, r, sb, sa, sc), [1, 0, 2]));
            out.scaled_add(-sgn, &swap(&f(self, p, q, r, sc, sb, sa), [2, 1, 0]));
        }
        out
    }

    /// `sum_abc (W + V) W / D` for one occupied triple on layout `sv`.
    fn triple(&self, i: So, j: So, k: So, sv: [usize; 3]) -> f64 {
        let w = self.antisym(i, j, k, sv, Self::x_block);
        let v = self.antisym(i, j, k, sv, Self::y_block);
        let e_occ = self.eo[i.0][i.1] + self.eo[j.0][j.1] + self.eo[k.0][k.1];
        let [sa, sb, sc] = sv;
        let mut acc = 0.0f64;
        for (a, &ea) in self.ev[sa].iter().enumerate() {
            for (b, &eb) in self.ev[sb].iter().enumerate() {
                for (c, &ec) in self.ev[sc].iter().enumerate() {
                    let d = e_occ - ea - eb - ec;
                    let wv = w[[a, b, c]];
                    acc += (wv + v[[a, b, c]]) * wv / d;
                }
            }
        }
        acc
    }
}

/// One canonical occupied triple: class index (number of beta occupied
/// spin orbitals), the three spin orbitals, and the virtual layout.
type Job = (usize, So, So, So, [usize; 3]);

/// Every canonical occupied triple of the four classes, in a fixed order.
fn jobs(no: [usize; 2]) -> Vec<Job> {
    let mut out = Vec::new();
    let [na, nb] = no;
    for i in 0..na {
        for j in i + 1..na {
            for k in j + 1..na {
                out.push((0, (A, i), (A, j), (A, k), [A, A, A]));
            }
        }
    }
    for i in 0..na {
        for j in i + 1..na {
            for k in 0..nb {
                out.push((1, (A, i), (A, j), (B, k), [A, A, B]));
            }
        }
    }
    for i in 0..na {
        for j in 0..nb {
            for k in j + 1..nb {
                out.push((2, (A, i), (B, j), (B, k), [A, B, B]));
            }
        }
    }
    for i in 0..nb {
        for j in i + 1..nb {
            for k in j + 1..nb {
                out.push((3, (B, i), (B, j), (B, k), [B, B, B]));
            }
        }
    }
    out
}

/// Weight of a class: see the module docs.
pub(crate) const CLASS_WEIGHTS: [f64; 4] = [1.0 / 6.0, 0.5, 0.5, 1.0 / 6.0];

/// Peak bytes one triple holds: the `W` and `V` blocks plus three
/// `X`/`Y` temporaries and a transposed copy, each at most `nv_max^3`.
fn per_triple_bytes(nv: [usize; 2]) -> usize {
    nv[0]
        .max(nv[1])
        .saturating_pow(3)
        .saturating_mul(6)
        .saturating_mul(8)
}

/// Spin-blocked (T) correction for converged UCCSD amplitudes `cc`.
///
/// `cfg.frozen_core` must be the value the amplitudes were solved with (it
/// sets the occupied blocks); a shape mismatch is an error.
///
/// # Errors
///
/// A non-UHF reference, amplitudes that do not match the reference's
/// active spaces, or integral blocks over the memory budget.
pub fn u_ccsd_t(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    scf: &ScfResult,
    cc: &UCcsdResult,
    cfg: &CcConfig,
) -> Result<UTriples, FerricError> {
    let (nocc, nmo) = crate::u_ccsd::uhf_counts(mol, scf)?;
    let no = [
        ferric_mp2::rimp2::active_occ(nocc[0], cfg.frozen_core)?,
        ferric_mp2::rimp2::active_occ(nocc[1], cfg.frozen_core)?,
    ];
    let nv = [nmo[0] - nocc[0], nmo[1] - nocc[1]];
    let t = &cc.amplitudes;
    if t.t1a.dim() != (no[0], nv[0])
        || t.t1b.dim() != (no[1], nv[1])
        || t.t2ab.dim() != (no[0], no[1], nv[0], nv[1])
    {
        return Err(FerricError::General(format!(
            "u_ccsd_t: amplitudes {:?}/{:?} do not match the reference's active spaces \
             (no={no:?}, nv={nv:?}); pass the frozen_core the UCCSD run used",
            t.t1a.dim(),
            t.t1b.dim()
        )));
    }
    let budget = ferric_core::memory::resolve_budget_bytes(cfg.memory_budget_bytes);
    let floor = TriplesInts::bytes(no, nv).saturating_add(per_triple_bytes(nv));
    ferric_core::memory::check_alloc(
        &format!("UCCSD(T) integral blocks + one triple (no={no:?}, nv={nv:?})"),
        floor,
        budget,
    )?;
    let _floor_charge =
        ferric_core::memory::pool::reserve_global("UCCSD(T) integral blocks", floor)?;

    let ri = URiBlocks::build(mol, obs, dfbs, op, scf, cfg.frozen_core)?;
    let ints = TriplesInts::build(&ri);
    let kernel = Kernel {
        ints: &ints,
        t: T2View::new(t),
        eo: [&ri.eo[A], &ri.eo[B]],
        ev: [&ri.ev[A], &ri.ev[B]],
        nv,
    };
    let jobs = jobs(no);
    // Band width: a pure function of the budget and nv, never of the
    // thread count, so the serial fold order (and the bits) is fixed.
    let band = (budget.saturating_sub(floor) / per_triple_bytes(nv).max(1)).max(1);
    let mut classes = [0.0f64; 4];
    for chunk in jobs.chunks(band) {
        let partials: Vec<f64> = with_blas_threads(1, || {
            chunk
                .par_iter()
                .map(|&(_, i, j, k, sv)| kernel.triple(i, j, k, sv))
                .collect()
        });
        for (&(cls, ..), p) in chunk.iter().zip(partials) {
            classes[cls] += p;
        }
    }
    let [aaa, aab, abb, bbb] = [0, 1, 2, 3].map(|c| CLASS_WEIGHTS[c] * classes[c]);
    Ok(UTriples {
        e_t: aaa + aab + abb + bbb,
        aaa,
        aab,
        abb,
        bbb,
    })
}
