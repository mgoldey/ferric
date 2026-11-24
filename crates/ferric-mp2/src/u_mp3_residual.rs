//! The spin-blocked MP3 doubles residual `R = D · c^(2)` from `c^(1)`.
//!
//! In spin orbitals (canonical or semicanonical, doubles only),
//!
//! ```text
//! R_ijab = ½ Σ_cd ⟨ab‖cd⟩ c_ijcd + ½ Σ_kl ⟨kl‖ij⟩ c_klab
//!        + P(ij) P(ab) Σ_kc ⟨kb‖cj⟩ c_ikac,      c^(2)_ijab = R_ijab / D_ijab
//! ```
//!
//! with `D = ε_i + ε_j − ε_a − ε_b` and `P(ij)P(ab) X = X_ijab − X_jiab − X_ijba
//! + X_jiba`. Splitting every spin-orbital sum by spin gives the three stored
//! blocks. Lower case is α, upper case β; `t` is `c^(1)`, `(pq|rs)` chemist.
//!
//! ```text
//! αα:  Σ_cd (ac|bd) t_ijcd + Σ_kl (ki|lj) t_klab + P(ij)P(ab) X_iajb
//!      X_iajb = Σ_kc [(kc|jb) − (kj|bc)] t^αα_ikac + Σ_KC (KC|jb) t^αβ_iKaC
//! ββ:  the same with α ↔ β; its mixed ring is Σ_kc (kc|JB) t^αβ_kIcA
//! αβ:  Σ_cD (ac|BD) t_iJcD + Σ_kL (ki|LJ) t_kLaB
//!      + Σ_kc (kc|JB) t^αα_ikac + Σ_KC [(KC|JB) − (KJ|BC)] t^αβ_iKaC
//!      + Σ_kc [(ia|kc) − (ik|ac)] t^αβ_kJcB + Σ_KC (ia|KC) t^ββ_JKBC
//!      − Σ_kC (ki|BC) t^αβ_kJaC − Σ_Kc (KJ|ac) t^αβ_iKcB
//! ```
//!
//! The ring terms are written as GEMMs over compound indices: with
//! `M^σ[(kc),(jb)] = (kc|jb) − (kj|bc)` (symmetric), `G[(ia),(JB)] = (ia|JB)`
//! and `T^σσ'[(ia),(KC)] = t[i,K,a,C]`,
//!
//! ```text
//! X^αα = T^αα M^α + T^αβ Gᵀ        X^ββ = T^ββ M^β + (T^αβ)ᵀ G
//! Y^αβ = T^αα G + T^αβ M^β + M^α T^αβ + G T^ββ
//! ```
//!
//! An RHF reference fed through this path reproduces the spin-orbital
//! [`crate::mp3`] energy (tests/u_mp3.rs); the αα and αβ blocks there also
//! satisfy the spin-adaptation identity `t^αα_ijab = t^αβ_ijab − t^αβ_ijba`.

use crate::u_mp3::SpinBlockedDoubles;
use crate::u_mp3_eri::{gemm_acc, EriSource, Space, SpinIdx, ALPHA, BETA};
use ferric_core::FerricError;
use ndarray::{s, Array2, Array4, ArrayView2, ArrayViewMut2};

const O: Space = Space::Occ;
const V: Space = Space::Vir;

/// Standard-layout copy of `a` with its axes permuted (`out[...] = a[axes...]`).
pub(crate) fn perm_std(a: &Array4<f64>, axes: [usize; 4]) -> Array4<f64> {
    let v = a.view().permuted_axes(axes);
    let mut out = Array4::zeros(v.raw_dim());
    out.assign(&v);
    out
}

/// A standard-layout 4-index array viewed as the matrix `[(d0 d1), (d2 d3)]`.
pub(crate) fn mat(a: &Array4<f64>) -> ArrayView2<'_, f64> {
    let (d0, d1, d2, d3) = a.dim();
    a.view()
        .into_shape_with_order((d0 * d1, d2 * d3))
        .expect("standard layout")
}

/// Mutable form of [`mat`].
pub(crate) fn mat_mut(a: &mut Array4<f64>) -> ArrayViewMut2<'_, f64> {
    let (d0, d1, d2, d3) = a.dim();
    a.view_mut()
        .into_shape_with_order((d0 * d1, d2 * d3))
        .expect("standard layout")
}

/// `M^σ[k,c,j,b] = (kc|jb) − (kj|bc)`, built in place from the `(ov|ov)` block.
fn same_spin_ring_kernel(
    src: &EriSource<'_>,
    s: SpinIdx,
    mut g_ovov: Array4<f64>,
) -> Result<Array4<f64>, FerricError> {
    let oovv = src.block((s, O, O), (s, V, V))?; // [k,j,b,c]
    g_ovov -= &oovv.view().permuted_axes([0, 3, 1, 2]);
    Ok(g_ovov)
}

/// Accumulate a same-spin ring product `X[(ia),(jb)]` into `R[i,j,a,b]` with
/// `P(ij)P(ab)`: `X_iajb − X_jaib − X_ibja + X_jbia`.
fn scatter_same_spin(x: &Array2<f64>, r: &mut Array4<f64>) {
    let (no, _, nv, _) = r.dim();
    let x4 = x
        .view()
        .into_shape_with_order((no, nv, no, nv))
        .expect("ring product shape");
    *r += &x4.view().permuted_axes([0, 2, 1, 3]);
    *r -= &x4.view().permuted_axes([2, 0, 1, 3]);
    *r -= &x4.view().permuted_axes([0, 2, 3, 1]);
    *r += &x4.view().permuted_axes([2, 0, 3, 1]);
}

/// The ovov blocks the ring terms consume: `(ia|jb)` αα and ββ (moved into
/// the ring kernels `M^σ`) and `(ia|JB)`.
pub(crate) struct RingInputs {
    pub(crate) g_aa: Array4<f64>,
    pub(crate) g_bb: Array4<f64>,
    pub(crate) g_ab: Array4<f64>,
}

/// All particle-hole ring terms that are GEMMs over `(ov)` compound indices.
pub(crate) fn ring_terms(
    src: &EriSource<'_>,
    g: RingInputs,
    t: &SpinBlockedDoubles,
    r: &mut SpinBlockedDoubles,
) -> Result<(), FerricError> {
    let m_a = same_spin_ring_kernel(src, ALPHA, g.g_aa)?;
    let m_b = same_spin_ring_kernel(src, BETA, g.g_bb)?;
    let g_ab = g.g_ab;
    let t_aa = perm_std(&t.aa, [0, 2, 1, 3]);
    let t_bb = perm_std(&t.bb, [0, 2, 1, 3]);
    let t_ab = perm_std(&t.ab, [0, 2, 1, 3]);
    let (ga, gb, gab) = (mat(&m_a), mat(&m_b), mat(&g_ab));
    let (taa, tbb, tab) = (mat(&t_aa), mat(&t_bb), mat(&t_ab));

    let mut x = Array2::zeros((taa.nrows(), ga.ncols()));
    gemm_acc(1.0, taa, ga, &mut x.view_mut());
    gemm_acc(1.0, tab, gab.t(), &mut x.view_mut());
    scatter_same_spin(&x, &mut r.aa);
    drop(x);

    let mut x = Array2::zeros((tbb.nrows(), gb.ncols()));
    gemm_acc(1.0, tbb, gb, &mut x.view_mut());
    gemm_acc(1.0, tab.t(), gab, &mut x.view_mut());
    scatter_same_spin(&x, &mut r.bb);
    drop(x);

    let mut y = Array2::zeros((taa.nrows(), gb.ncols()));
    gemm_acc(1.0, taa, gab, &mut y.view_mut());
    gemm_acc(1.0, tab, gb, &mut y.view_mut());
    gemm_acc(1.0, ga, tab, &mut y.view_mut());
    gemm_acc(1.0, gab, tbb, &mut y.view_mut());
    let (no_a, no_b, nv_a, nv_b) = r.ab.dim();
    let y4 = y
        .into_shape_with_order((no_a, nv_a, no_b, nv_b))
        .expect("ring product shape");
    r.ab += &y4.view().permuted_axes([0, 2, 1, 3]);
    Ok(())
}

/// The two αβ exchange rings, `− Σ_kC (ki|BC) t_kJaC − Σ_Kc (KJ|ac) t_iKcB`.
pub(crate) fn mixed_exchange_rings(
    src: &EriSource<'_>,
    t_ab: &Array4<f64>,
    r_ab: &mut Array4<f64>,
) -> Result<(), FerricError> {
    let (no_a, no_b, nv_a, nv_b) = r_ab.dim();
    {
        let z = perm_std(&src.block((ALPHA, O, O), (BETA, V, V))?, [1, 2, 0, 3]); // [i,B,k,C]
        let tc = perm_std(t_ab, [0, 3, 1, 2]); // [k,C,J,a]
        let mut u = Array2::zeros((no_a * nv_b, no_b * nv_a));
        gemm_acc(1.0, mat(&z), mat(&tc), &mut u.view_mut());
        let u4 = u
            .into_shape_with_order((no_a, nv_b, no_b, nv_a))
            .expect("exchange ring shape");
        *r_ab -= &u4.view().permuted_axes([0, 2, 3, 1]);
    }
    let z = perm_std(&src.block((ALPHA, V, V), (BETA, O, O))?, [3, 0, 2, 1]); // [J,a,K,c]
    let td = perm_std(t_ab, [1, 2, 0, 3]); // [K,c,i,B]
    let mut v = Array2::zeros((no_b * nv_a, no_a * nv_b));
    gemm_acc(1.0, mat(&z), mat(&td), &mut v.view_mut());
    let v4 = v
        .into_shape_with_order((no_b, nv_a, no_a, nv_b))
        .expect("exchange ring shape");
    *r_ab -= &v4.view().permuted_axes([2, 0, 1, 3]);
    Ok(())
}

/// Hole-hole ladder `R[i,j,a,b] += Σ_kl (ki|lj) t[k,l,a,b]` for the spin pair
/// `(s1, s2)` of `(i, j)`.
pub(crate) fn hole_ladder(
    src: &EriSource<'_>,
    s1: SpinIdx,
    s2: SpinIdx,
    t: &Array4<f64>,
    r: &mut Array4<f64>,
) -> Result<(), FerricError> {
    let m = perm_std(&src.block((s1, O, O), (s2, O, O))?, [1, 3, 0, 2]); // [i,j,k,l]
    gemm_acc(1.0, mat(&m), mat(t), &mut mat_mut(r));
    Ok(())
}

/// Particle-particle ladder `R[i,j,a,b] += Σ_cd (ac|bd) t[i,j,c,d]` for the
/// spin pair `(s1, s2)` of `(a, b)`, one first index `a` at a time.
pub(crate) fn particle_ladder(
    src: &EriSource<'_>,
    s1: SpinIdx,
    s2: SpinIdx,
    t: &Array4<f64>,
    r: &mut Array4<f64>,
) -> Result<(), FerricError> {
    let (n1, n2, _, nb) = r.dim();
    let tm = mat(t);
    src.for_each_vvvv_slab(s1, s2, &mut |a, slab| {
        let (nc, _, nd) = slab.dim();
        let mut sm = ndarray::Array3::zeros((nc, nd, nb));
        sm.assign(&slab.permuted_axes([0, 2, 1])); // [c,d,b]
        let smat = sm
            .view()
            .into_shape_with_order((nc * nd, nb))
            .expect("slab shape");
        let mut out = Array2::zeros((n1 * n2, nb));
        gemm_acc(1.0, tm, smat, &mut out.view_mut());
        let out3 = out
            .into_shape_with_order((n1, n2, nb))
            .expect("ladder shape");
        let mut ra = r.slice_mut(s![.., .., a, ..]);
        ra += &out3;
    })
}

/// The full residual `R` from `c^(1)` (`t`) and the ovov ring inputs.
pub(crate) fn mp3_residual(
    src: &EriSource<'_>,
    g: RingInputs,
    t: &SpinBlockedDoubles,
) -> Result<SpinBlockedDoubles, FerricError> {
    let mut r = SpinBlockedDoubles {
        aa: Array4::zeros(t.aa.raw_dim()),
        bb: Array4::zeros(t.bb.raw_dim()),
        ab: Array4::zeros(t.ab.raw_dim()),
    };
    ring_terms(src, g, t, &mut r)?;
    mixed_exchange_rings(src, &t.ab, &mut r.ab)?;
    hole_ladder(src, ALPHA, ALPHA, &t.aa, &mut r.aa)?;
    hole_ladder(src, BETA, BETA, &t.bb, &mut r.bb)?;
    hole_ladder(src, ALPHA, BETA, &t.ab, &mut r.ab)?;
    particle_ladder(src, ALPHA, ALPHA, &t.aa, &mut r.aa)?;
    particle_ladder(src, ALPHA, BETA, &t.ab, &mut r.ab)?;
    particle_ladder(src, BETA, BETA, &t.bb, &mut r.bb)?;
    Ok(r)
}
