//! Spatial chemist MO integral blocks for the spin-blocked unrestricted MP3
//! driver ([`crate::u_mp3`]).
//!
//! Every block is `(pq|rs)` with `p, q` of one spin and `r, s` of one spin
//! (chemist integrals vanish otherwise), returned as a standard-layout
//! `[p, q, r, s]` array. Two sources give the same blocks:
//!
//! - **RI**: `(pq|rs) = Σ_P B^P_pq B^P_rs` with `B = V^{-1/2} (P|pq)` in the
//!   Coulomb metric, the same build [`crate::mp3::mp3_energy`] uses
//!   ([`crate::rimp2::cholesky_inverse_sqrt`], [`crate::mo_transform`]).
//! - **Exact**: the dense AO tensor of [`crate::canonical::dense_ao_eri`],
//!   transformed one index at a time ([`exact_mo_block`](crate::u_mp3_eri::exact_mo_block)).
//!
//! The particle-particle ladder never asks for a whole VVVV block from the RI
//! source: `EriSource::for_each_vvvv_slab` hands out one `(a c|b d)` slab per
//! first index `a`, so the RI path holds `O(nv³)` there, not `O(nv⁴)`.

use crate::mo_transform::{
    dress_3index, transform_3center_oo, transform_3center_ov, transform_3center_vv,
};
use ferric_core::FerricError;
use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayViewMut2};

/// Occupied or virtual orbital space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    /// Active occupied orbitals.
    Occ,
    /// Virtual orbitals.
    Vir,
}

/// Spin index: 0 = α, 1 = β.
pub type SpinIdx = usize;
/// α spin.
pub const ALPHA: SpinIdx = 0;
/// β spin.
pub const BETA: SpinIdx = 1;

/// One spin's active occupied and virtual orbitals.
#[derive(Debug, Clone)]
pub struct SpinOrbitals {
    /// Active occupied MO coefficients, `(nbas, nocc)`.
    pub c_occ: Array2<f64>,
    /// Virtual MO coefficients, `(nbas, nvir)`.
    pub c_vir: Array2<f64>,
    /// Active occupied orbital energies (Hartree).
    pub e_occ: Vec<f64>,
    /// Virtual orbital energies (Hartree).
    pub e_vir: Vec<f64>,
}

impl SpinOrbitals {
    /// Coefficients of `space`.
    pub fn coeffs(&self, space: Space) -> &Array2<f64> {
        match space {
            Space::Occ => &self.c_occ,
            Space::Vir => &self.c_vir,
        }
    }

    /// Number of orbitals in `space`.
    pub fn n(&self, space: Space) -> usize {
        self.coeffs(space).ncols()
    }
}

/// `c += alpha · a · b`, skipping empty products (a zero-width spin channel).
pub(crate) fn gemm_acc(
    alpha: f64,
    a: ArrayView2<f64>,
    b: ArrayView2<f64>,
    c: &mut ArrayViewMut2<f64>,
) {
    if a.nrows() == 0 || a.ncols() == 0 || b.ncols() == 0 {
        return;
    }
    general_mat_mul(alpha, &a, &b, 1.0, c);
}

/// `xᵀ · c` for `x` viewed as `(n, rest)`: contracts the leading index and
/// moves the new index to the back, `[μ, rest] → [rest, p]`.
fn rotate_transform(x: ArrayView2<f64>, c: &Array2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros((x.ncols(), c.ncols()));
    gemm_acc(1.0, x.t(), c.view(), &mut y.view_mut());
    y
}

/// Exact `(pq|rs) = Σ C_μp C_νq C_λr C_σs (μν|λσ)` from a dense chemist AO
/// tensor (`ao[((μ·n+ν)·n+λ)·n+σ]`, as [`crate::canonical::dense_ao_eri`]).
///
/// Four quarter transforms, each contracting the leading index and appending
/// the new one: `[μνλσ] → [νλσ p] → [λσ p q] → [σ p q r] → [p q r s]`. The
/// widest point is two consecutive intermediates ([`exact_block_transient`]).
pub fn exact_mo_block(ao: &[f64], nbas: usize, c: [&Array2<f64>; 4]) -> Array4<f64> {
    let n = nbas;
    let dims = [c[0].ncols(), c[1].ncols(), c[2].ncols(), c[3].ncols()];
    let x0 = ArrayView2::from_shape((n, n * n * n), ao).expect("dense AO ERI is nbas⁴");
    let y1 = rotate_transform(x0, c[0]);
    let y2 = {
        let v = y1
            .view()
            .into_shape_with_order((n, n * n * dims[0]))
            .expect("y1");
        rotate_transform(v, c[1])
    };
    drop(y1);
    let y3 = {
        let v = y2
            .view()
            .into_shape_with_order((n, n * dims[0] * dims[1]))
            .expect("y2");
        rotate_transform(v, c[2])
    };
    drop(y2);
    let y4 = {
        let v = y3
            .view()
            .into_shape_with_order((n, dims[0] * dims[1] * dims[2]))
            .expect("y3");
        rotate_transform(v, c[3])
    };
    drop(y3);
    y4.into_shape_with_order((dims[0], dims[1], dims[2], dims[3]))
        .expect("block shape")
}

/// Elements co-resident at the widest point of [`exact_mo_block`]: the max
/// over consecutive intermediate pairs `y_k + y_{k+1}` (the AO input is held
/// by the caller and not counted).
pub fn exact_block_transient(nbas: usize, d: [usize; 4]) -> usize {
    let n = nbas;
    let y1 = n.saturating_pow(3).saturating_mul(d[0]);
    let y2 = n
        .saturating_pow(2)
        .saturating_mul(d[0])
        .saturating_mul(d[1]);
    let y3 = n
        .saturating_mul(d[0])
        .saturating_mul(d[1])
        .saturating_mul(d[2]);
    let y4 = d[0]
        .saturating_mul(d[1])
        .saturating_mul(d[2])
        .saturating_mul(d[3]);
    y1.saturating_add(y2)
        .max(y2.saturating_add(y3))
        .max(y3.saturating_add(y4))
}

/// Dressed RI blocks of one spin: `B^P_oo`, `B^P_ov`, `B^P_vv`, each `[P, p, q]`.
#[derive(Debug)]
pub(crate) struct RiSpin {
    oo: Array3<f64>,
    ov: Array3<f64>,
    vv: Array3<f64>,
}

impl RiSpin {
    fn get(&self, x: Space, y: Space) -> Result<&Array3<f64>, FerricError> {
        match (x, y) {
            (Space::Occ, Space::Occ) => Ok(&self.oo),
            (Space::Occ, Space::Vir) => Ok(&self.ov),
            (Space::Vir, Space::Vir) => Ok(&self.vv),
            (Space::Vir, Space::Occ) => Err(FerricError::General(
                "u_mp3: RI blocks are stored (oo, ov, vv); a (vo| pair was requested".into(),
            )),
        }
    }
}

/// Build both spins' dressed RI blocks from the AO 3-index tensor and the
/// metric `V^{-1/2}`, in the order α (ov, oo, vv) then β (ov, oo, vv).
pub(crate) fn build_ri_spins(
    eri3_ao: &Array3<f64>,
    v_inv_sqrt: &Array2<f64>,
    orbs: &[SpinOrbitals; 2],
) -> [RiSpin; 2] {
    let one = |o: &SpinOrbitals| {
        let ov = dress_3index(
            &transform_3center_ov(eri3_ao, &o.c_occ, &o.c_vir),
            v_inv_sqrt,
        );
        let oo = dress_3index(&transform_3center_oo(eri3_ao, &o.c_occ), v_inv_sqrt);
        let vv = dress_3index(&transform_3center_vv(eri3_ao, &o.c_vir), v_inv_sqrt);
        RiSpin { oo, ov, vv }
    };
    let a = one(&orbs[ALPHA]);
    let b = one(&orbs[BETA]);
    [a, b]
}

/// Where the 4-index blocks come from.
#[derive(Debug)]
pub(crate) enum EriSource<'a> {
    /// RI factors per spin.
    Ri(Box<[RiSpin; 2]>),
    /// Dense AO tensor plus the orbitals to transform with.
    Exact {
        ao: Vec<f64>,
        nbas: usize,
        orbs: &'a [SpinOrbitals; 2],
    },
}

/// One side of a chemist integral: spin and the two orbital spaces.
pub(crate) type Pair = (SpinIdx, Space, Space);

/// `B` viewed as `[P, pq]` (standard layout from [`dress_3index`]).
fn as_aux_matrix(b: &Array3<f64>) -> ArrayView2<'_, f64> {
    let (p, d1, d2) = b.dim();
    b.view()
        .into_shape_with_order((p, d1 * d2))
        .expect("dressed B is standard layout")
}

impl EriSource<'_> {
    /// `(pq|rs)` as a standard-layout `[p, q, r, s]` array.
    pub(crate) fn block(&self, l: Pair, r: Pair) -> Result<Array4<f64>, FerricError> {
        match self {
            EriSource::Ri(spins) => {
                let bl = spins[l.0].get(l.1, l.2)?;
                let br = spins[r.0].get(r.1, r.2)?;
                let (d1, d2, d3, d4) = (bl.dim().1, bl.dim().2, br.dim().1, br.dim().2);
                let mut out = Array2::zeros((d1 * d2, d3 * d4));
                gemm_acc(
                    1.0,
                    as_aux_matrix(bl).t(),
                    as_aux_matrix(br),
                    &mut out.view_mut(),
                );
                Ok(out
                    .into_shape_with_order((d1, d2, d3, d4))
                    .expect("block shape"))
            }
            EriSource::Exact { ao, nbas, orbs } => Ok(exact_mo_block(
                ao,
                *nbas,
                [
                    orbs[l.0].coeffs(l.1),
                    orbs[l.0].coeffs(l.2),
                    orbs[r.0].coeffs(r.1),
                    orbs[r.0].coeffs(r.2),
                ],
            )),
        }
    }

    /// Call `f(a, slab)` for every virtual `a` of spin `s1`, with
    /// `slab[c, b, d] = (a c | b d)`, `a, c ∈ vir(s1)` and `b, d ∈ vir(s2)`.
    pub(crate) fn for_each_vvvv_slab(
        &self,
        s1: SpinIdx,
        s2: SpinIdx,
        f: &mut dyn FnMut(usize, ArrayView3<f64>),
    ) -> Result<(), FerricError> {
        match self {
            EriSource::Ri(spins) => {
                let b1 = &spins[s1].vv;
                let b2 = &spins[s2].vv;
                let (_, na, nc) = b1.dim();
                let (_, nb, nd) = b2.dim();
                let b2m = as_aux_matrix(b2);
                for a in 0..na {
                    let b1a = b1.slice(s![.., a, ..]);
                    let mut slab = Array2::zeros((nc, nb * nd));
                    gemm_acc(1.0, b1a.t(), b2m, &mut slab.view_mut());
                    let slab = slab
                        .into_shape_with_order((nc, nb, nd))
                        .expect("slab shape");
                    f(a, slab.view());
                }
                Ok(())
            }
            EriSource::Exact { .. } => {
                let g = self.block((s1, Space::Vir, Space::Vir), (s2, Space::Vir, Space::Vir))?;
                for a in 0..g.dim().0 {
                    f(a, g.slice(s![a, .., .., ..]));
                }
                Ok(())
            }
        }
    }
}
