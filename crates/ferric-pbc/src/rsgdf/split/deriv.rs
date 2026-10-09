//! Derivative walks of the range-split Gamma RS-GDF energy (forces and
//! stress): the Rust port of `reference/pbc/pbc_grad_gdf_split.py`
//! (FINDINGS "Iteration 26 (Python, range-split forces/stress/k-point)").
//!
//! # What is differentiated (grouping A, `twocall`, split metric)
//!
//! ```text
//! J3 = K + (2/Ω) Σ_{G∈half} Re[ P̄ (v_LR X + v_SR X_s) + v_SR P̄_ss X_c ] − c0 (S − S_ss) q_cᵀ
//! K  = Σ_{L,T} (μ_0 ν^c_L | P^c_T)_erfc + Σ_{L,T} (μ^c_0 ν^s_L | P^c_T)_erfc      (the two kept calls)
//! J2 = Σ_T (P^c_0|Q^c_T)_erfc + (2/Ω) Σ_{G∈half} Re[ v_LR XᴴX + v_SR (X_sᴴX + X_cᴴX_s) ] − c0 q_c q_cᵀ
//! ```
//!
//! with the fitted densities `Y`, `Wm` of [`crate::rsgdf::deriv`] UNCHANGED
//! (`Wm` symmetrised there: FINDINGS "Iteration 26" (e), an unsymmetrised
//! `Wm` shows up at 2.2e-11 in ΣF through the `v_SR`-weighted metric terms).
//! Every derivative walk follows the energy's partition:
//!
//! * **SR K**: the two kept calls on the piece shells of the build's combined
//!   libint bases, each call with its own pair bound and the compact-aux
//!   bound; every derivative block is scaled by the pieces' libint factors
//!   `f` (geometry-independent) and routed to the PARENT atoms / aux rows.
//! * **SR J2**: `(P^c|Q^c)` pairs only.
//! * **LR J3**: the full-pair derivative with the per-aux-column weight
//!   `v_LR X + v_SR X_s`; a SECOND pass over the smooth orbital pieces
//!   (`P_ss`) with weight `v_SR X_c`. Aux centres: `−iG` on `X`, `X_s`, `X_c`
//!   (all pieces of an aux function share its centre).
//! * **LR J2**: `d/dC_R` of `Re[Yᴴ Wm X] + v_SR Re[X_cᴴ Wm X_s]`, `Y = v_LR X
//!   + v_SR X_s` (the non-cancelling form the build uses).
//! * **G = 0**: `M_g0 = −c0 Σ_P Y_P q^c_P` contracted with `d(S − S_ss)`; the
//!   smooth-piece overlap derivative / virial is [`SplitG0`] (called from
//!   `crate::grad` / `crate::stress`).
//! * **Stress**: the same pieces under strain, plus the SR kernel-weight
//!   strain `dv_SR/dG² = −v_SR/G² + v_LR/(4ω²)` (= −4π/G⁴ + v_LR(1/G² +
//!   1/4ω²)), the smooth-piece pair-FT strain and the piece aux-FT strain;
//!   the G = 0 volume terms take `(q_c, S − S_ss)`.
//!
//! # Bit identity with the unsplit derivative
//!
//! A split that moves nothing (λ = 0, or λ below every exponent) walks the
//! SAME triplets in the same order through the combined bases (parents
//! only, every `f` exactly 1, `q_c` bitwise `q`), and its LR part is the
//! unsplit code path: the forces and stress are then bitwise the unsplit
//! ones.
//!
//! # Parallelism
//!
//! The SR walks are ordered-parallel ([`crate::ordered`]): units `(L, i1,
//! i2)` (parent pairs; the two calls inside a unit in call order) and `(P,
//! Q)` (parent aux pairs), each unit's screen + derivative blocks + per-
//! triplet subtotals in parallel, the accumulation serially in unit order.
//! There is no separate frozen serial oracle for the split walks; the
//! `serial` flag of the unsplit walks does not apply here.

use super::{kernel_weights, SmoothObs, SplitPlan};
use crate::budget::{bytes_of, Ledger};
use crate::hcore::ONE_E_ENGINE_PRECISION;
use crate::lattice::Cell;
use crate::ordered::{ordered_units, Stored};
use crate::pair_ft::{pair_ft_deriv_chunked, pair_ft_strain_chunked, DEFAULT_PAIR_FT_THRESH};
use crate::rsgdf::deriv::Y3;
use crate::rsgdf::strain::{aux_ft_strain_shells, FitStrainTerms};
use crate::rsgdf::{
    aux_ft_shells, check_obs_on_cell, dot3, gshells, pair_image_radius, require_pure_aux,
    LatticeWalker, RsGdf, Stage, ENGINE_PRECISION,
};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::engine_pool::EnginePool;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, Array3, ArrayView2};
use num_complex::Complex64;

type Mat3 = [[f64; 3]; 3];

/// `a b` (fresh output).
fn mm(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    let mut c = Array2::<f64>::zeros((a.nrows(), b.ncols()));
    general_mat_mul(1.0, &a, &b, 0.0, &mut c);
    c
}

/// Real and imaginary parts.
fn reim(a: &Array2<Complex64>) -> (Array2<f64>, Array2<f64>) {
    (a.mapv(|z| z.re), a.mapv(|z| z.im))
}

/// `(re, im)` of the pair FT, `(nao², ng)`, row `μ·nao+ν`.
fn pair_reim(p: &Array3<Complex64>) -> (Array2<f64>, Array2<f64>) {
    let (n, _, ng) = p.dim();
    let re = Array2::from_shape_fn((n * n, ng), |(mn, g)| p[[mn / n, mn % n, g]].re);
    let im = Array2::from_shape_fn((n * n, ng), |(mn, g)| p[[mn / n, mn % n, g]].im);
    (re, im)
}

/// AO → atom of `prep`.
fn ao_atoms(prep: &PreparedBasis) -> Vec<usize> {
    let sh2at = prep.shell_to_atom();
    let (dims, offs) = (prep.shell_dims(), prep.shell_offsets());
    let mut out = vec![0usize; prep.nbasis()];
    for sh in 0..prep.nshells() {
        for k in 0..dims[sh] {
            out[offs[sh] + k] = sh2at[sh];
        }
    }
    out
}

/// `(2/Ω) dv_LR/dG²`, `(2/Ω) dv_SR/dG²` from the weights of
/// [`kernel_weights`]; `sr_kernel = false` (the stress mutant) zeroes the
/// SR one.
fn kernel_slopes(g2: f64, w: (f64, f64), omega: f64, sr_kernel: bool) -> (f64, f64) {
    let inv4w2 = 1.0 / (4.0 * omega * omega);
    let d_lr = -w.0 * (1.0 / g2 + inv4w2);
    let d_sr = if sr_kernel {
        -w.1 / g2 + w.0 * inv4w2
    } else {
        0.0
    };
    (d_lr, d_sr)
}

// ---------------------------------------------------------------------------
// Ordered-parallel SR walks over the kept calls
// ---------------------------------------------------------------------------

/// One unit of a split SR walk: its screened count and, per kept entry
/// with a contribution, `(parent aux shell, T, value)` in walk order.
struct SplitUnit<V> {
    count: usize,
    items: Vec<(usize, [f64; 3], V)>,
}

impl<V: Stored> Stored for SplitUnit<V> {
    /// Heap bytes held: the struct plus 32 bytes of bookkeeping and the value's own bytes per item.
    fn stored_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .items
                .iter()
                .map(|(_, _, v)| 32 + v.stored_bytes())
                .sum::<usize>()
    }
}

/// One kept triplet's force contributions, each summed from zero:
/// `gpx[pp]` (parent aux row `P.off + pp`), `ga` / `gb` (the parent atoms of
/// the call's bra / ket shell).
struct Sr3Contrib {
    gpx: Vec<[f64; 3]>,
    ga: [f64; 3],
    gb: [f64; 3],
}

impl Stored for Sr3Contrib {
    /// Heap bytes held: the struct plus 24 bytes per aux-function force row.
    fn stored_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + 24 * self.gpx.len()
    }
}

impl SplitPlan {
    /// Whether any aux primitive moved (else the unsplit LR code runs).
    pub(in crate::rsgdf) fn moves_aux(&self) -> bool {
        !self.aux.s_sh.is_empty()
    }

    /// Pool of erfc(ω) 3-centre derivative engines over the pair-piece orbital basis and the aux
    /// pieces, one per rayon worker.
    fn sr3_deriv_pool(&self, st: &Stage<'_>) -> Result<EnginePool, FerricError> {
        EnginePool::from_fn(|| {
            Engine::new_3center_deriv(
                Operator::erfc(st.omega),
                &self.obs.x,
                &self.aux.x,
                ENGINE_PRECISION,
            )
        })
    }

    /// Pool of erfc(ω) 2-centre derivative engines over the aux pieces, one per rayon worker.
    fn sr2_deriv_pool(&self, st: &Stage<'_>) -> Result<EnginePool, FerricError> {
        EnginePool::from_fn(|| {
            Engine::new_2center_deriv(Operator::erfc(st.omega), &self.aux.x, ENGINE_PRECISION)
        })
    }

    /// One `(L, i1, i2)` unit: the kept calls of the parent pair in call
    /// order, each through [`SplitPlan::sr3_call`] (the energy's own walk),
    /// with `eval(engine, call, aux piece, L, T)`.
    fn sr3_unit<V, E>(
        &self,
        st: &Stage<'_>,
        eng: &mut Engine,
        (l, i1, i2): ([f64; 3], usize, usize),
        global: f64,
        eval: &E,
    ) -> Result<SplitUnit<V>, FerricError>
    where
        E: Fn(
            &mut Engine,
            (usize, usize),
            usize,
            [f64; 3],
            [f64; 3],
        ) -> Result<Option<V>, FerricError>,
    {
        let mut count = 0usize;
        let mut items = Vec::new();
        for &call in self.calls(i1, i2).iter().flatten() {
            let mut visit = |ip: usize, xp: usize, t: [f64; 3]| -> Result<(), FerricError> {
                if let Some(v) = eval(eng, call, xp, l, t)? {
                    items.push((ip, t, v));
                }
                Ok(())
            };
            self.sr3_call(st, call, &l, global, &mut count, &mut visit)?;
        }
        Ok(SplitUnit { count, items })
    }

    /// The kept SR 3-centre walk, ordered-parallel over units `(L, i1, i2)`
    /// in `L → i1 → i2` order: `eval` (pure) in parallel, then
    /// `apply(i1, i2, P, L, T, value)` serially in walk order. Returns the
    /// triplet count.
    fn sr3_ordered<V, E, A>(
        &self,
        st: &Stage<'_>,
        images: &[[f64; 3]],
        pool: &EnginePool,
        budget: usize,
        eval: E,
        mut apply: A,
    ) -> Result<usize, FerricError>
    where
        V: Send + Stored,
        E: Fn(
                &mut Engine,
                (usize, usize),
                usize,
                [f64; 3],
                [f64; 3],
            ) -> Result<Option<V>, FerricError>
            + Sync,
        A: FnMut(usize, usize, usize, [f64; 3], [f64; 3], V) -> Result<(), FerricError>,
    {
        let nsh = st.obs_sh.len();
        let global = self.sr3_global_radius(st);
        let unit_of = |u: usize| (images[u / (nsh * nsh)], (u / nsh) % nsh, u % nsh);
        let mut count = 0usize;
        ordered_units(
            images.len() * nsh * nsh,
            budget,
            0,
            |u| pool.with(|eng| self.sr3_unit(st, eng, unit_of(u), global, &eval)),
            |u, unit: SplitUnit<V>| {
                let (l, i1, i2) = unit_of(u);
                count += unit.count;
                for (ip, t, v) in unit.items {
                    apply(i1, i2, ip, l, t, v)?;
                }
                Ok(())
            },
        )?;
        Ok(count)
    }

    /// One parent aux pair `(ip, iq)` of the kept metric walk (the geometry
    /// and screen of the build's `sr2_task`).
    fn sr2_unit<V, E>(
        &self,
        st: &Stage<'_>,
        eng: &mut Engine,
        (ip, iq): (usize, usize),
        global: f64,
        eval: &E,
    ) -> Result<SplitUnit<V>, FerricError>
    where
        E: Fn(&mut Engine, (usize, usize), [f64; 3]) -> Result<Option<V>, FerricError>,
    {
        let empty = || SplitUnit {
            count: 0,
            items: Vec::new(),
        };
        let (Some(xp), Some(xq)) = (self.aux.compact[ip], self.aux.compact[iq]) else {
            return Ok(empty());
        };
        let (p, q) = (&self.aux.xsh[xp], &self.aux.xsh[xq]);
        let rad = if st.sr_screen {
            match st.radius(p.qbound, q.qbound, (p.amin, p.amax), (q.amin, q.amax)) {
                Some(r) => r,
                None => return Ok(empty()),
            }
        } else {
            global
        };
        let x0 = [
            p.center[0] - q.center[0],
            p.center[1] - q.center[1],
            p.center[2] - q.center[2],
        ];
        let mut count = 0usize;
        let mut items = Vec::new();
        st.walker.visit(x0, rad, |t| {
            count += 1;
            if let Some(v) = eval(eng, (xp, xq), t)? {
                items.push((iq, t, v));
            }
            Ok(())
        })?;
        Ok(SplitUnit { count, items })
    }

    /// The kept SR metric walk, ordered-parallel over parent aux pairs in
    /// `P → Q` order: `apply(P, Q, T, value)` serially in walk order.
    fn sr2_ordered<V, E, A>(
        &self,
        st: &Stage<'_>,
        pool: &EnginePool,
        budget: usize,
        eval: E,
        mut apply: A,
    ) -> Result<usize, FerricError>
    where
        V: Send + Stored,
        E: Fn(&mut Engine, (usize, usize), [f64; 3]) -> Result<Option<V>, FerricError> + Sync,
        A: FnMut(usize, usize, [f64; 3], V) -> Result<(), FerricError>,
    {
        let nsh = st.aux_sh.len();
        let global = self.sr2_global_radius(st);
        let mut count = 0usize;
        ordered_units(
            nsh * nsh,
            budget,
            0,
            |u| pool.with(|eng| self.sr2_unit(st, eng, (u / nsh, u % nsh), global, &eval)),
            |u, unit: SplitUnit<V>| {
                let (ip, iq) = (u / nsh, u % nsh);
                count += unit.count;
                for (_, t, v) in unit.items {
                    apply(ip, iq, t, v)?;
                }
                Ok(())
            },
        )?;
        Ok(count)
    }

    /// `(ga, gb, gpx)` of one kept-call derivative block (layout
    /// `[d/dP, d/d(bra), d/d(ket)]`, each `(nP, n1, n2)`), every `Y` element
    /// scaled by the pieces' libint factors (`1` exactly for parents, so an
    /// unmoved triplet is bitwise the unsplit contraction). `y` is the Gamma
    /// `Y[P, μν]` or a k-point phase-folded bin `Z[μν, P]`.
    fn sr3_contract(
        &self,
        blk: &[f64],
        (x1, x2, xp): (usize, usize, usize),
        n: usize,
        y: Y3<'_>,
    ) -> Sr3Contrib {
        let (a, b, p) = (&self.obs.xsh[x1], &self.obs.xsh[x2], &self.aux.xsh[xp]);
        let sc = self.obs.scale[x1] * self.obs.scale[x2] * self.aux.scale[xp];
        let nb = p.nfun * a.nfun * b.nfun;
        let mut ga = [0.0_f64; 3];
        let mut gb = [0.0_f64; 3];
        let mut gpxs = Vec::with_capacity(p.nfun);
        for pp in 0..p.nfun {
            let prow = p.off + pp;
            let mut gpx = [0.0_f64; 3];
            for i in 0..a.nfun {
                let r0 = (a.off + i) * n + b.off;
                for j in 0..b.nfun {
                    let yv = sc * y.at(prow, r0 + j);
                    if yv == 0.0 {
                        continue;
                    }
                    let idx = (pp * a.nfun + i) * b.nfun + j;
                    for x in 0..3 {
                        gpx[x] += yv * blk[x * nb + idx];
                        ga[x] += yv * blk[(3 + x) * nb + idx];
                        gb[x] += yv * blk[(6 + x) * nb + idx];
                    }
                }
            }
            gpxs.push(gpx);
        }
        Sr3Contrib { gpx: gpxs, ga, gb }
    }

    /// Kept SR 3-centre force pieces `(orbital natoms × 3, aux naux × 3,
    /// count)`: `Σ Y · d(kept call)` routed to the parent atoms and aux rows.
    pub(in crate::rsgdf) fn sr3_force(
        &self,
        st: &Stage<'_>,
        images: &[[f64; 3]],
        y: &Array2<f64>,
        natoms: usize,
        budget: usize,
    ) -> Result<(Array2<f64>, Array2<f64>, usize), FerricError> {
        self.sr3_force_with(st, images, natoms, budget, |_, _| Y3::AuxMajor(y))
    }

    /// [`SplitPlan::sr3_force`] with the weight of each triplet from
    /// `weights(L, T)` (the k-point forces: the phase-folded residue bin of
    /// `(L, T)`, pair-major).
    pub(in crate::rsgdf) fn sr3_force_with<'w, W>(
        &self,
        st: &Stage<'_>,
        images: &[[f64; 3]],
        natoms: usize,
        budget: usize,
        weights: W,
    ) -> Result<(Array2<f64>, Array2<f64>, usize), FerricError>
    where
        W: Fn([f64; 3], [f64; 3]) -> Y3<'w> + Sync,
    {
        let n = st.obs.nbasis();
        let sh2at = st.obs.shell_to_atom();
        let mut orb = Array2::<f64>::zeros((natoms, 3));
        let mut auxg = Array2::<f64>::zeros((st.aux.nbasis(), 3));
        let pool = self.sr3_deriv_pool(st)?;
        let count = self.sr3_ordered(
            st,
            images,
            &pool,
            budget,
            |eng, (x1, x2), xp, l, t| {
                Ok(eng
                    .compute_eri3_deriv_shifted(
                        &self.obs.x,
                        &self.aux.x,
                        xp,
                        x1,
                        x2,
                        [t, [0.0; 3], l],
                    )?
                    .map(|blk| self.sr3_contract(blk, (x1, x2, xp), n, weights(l, t))))
            },
            |i1, i2, ip, _, _, c: Sr3Contrib| {
                let p = &st.aux_sh[ip];
                for (pp, gpx) in c.gpx.iter().enumerate() {
                    for x in 0..3 {
                        auxg[(p.off + pp, x)] += gpx[x];
                    }
                }
                for x in 0..3 {
                    orb[(sh2at[i1], x)] += c.ga[x];
                    orb[(sh2at[i2], x)] += c.gb[x];
                }
                Ok(())
            },
        )?;
        Ok((orb, auxg, count))
    }

    /// Kept SR 3-centre strain: per triplet `ga ⊗ (A − C′) + gb ⊗ (B′ − C′)`
    /// (`B′ = B + L`, `C′ = C_P + T`; `imgs = 0` drops the image translations,
    /// the `NoSrImages` stress mutant).
    pub(in crate::rsgdf) fn sr3_strain(
        &self,
        st: &Stage<'_>,
        images: &[[f64; 3]],
        y: &Array2<f64>,
        imgs: f64,
        budget: usize,
    ) -> Result<(Mat3, usize), FerricError> {
        let n = st.obs.nbasis();
        let mut out = [[0.0_f64; 3]; 3];
        let pool = self.sr3_deriv_pool(st)?;
        let count = self.sr3_ordered(
            st,
            images,
            &pool,
            budget,
            |eng, (x1, x2), xp, l, t| {
                let Some(blk) = eng.compute_eri3_deriv_shifted(
                    &self.obs.x,
                    &self.aux.x,
                    xp,
                    x1,
                    x2,
                    [t, [0.0; 3], l],
                )?
                else {
                    return Ok(None);
                };
                let c = self.sr3_contract(blk, (x1, x2, xp), n, Y3::AuxMajor(y));
                let (a, b, p) = (&self.obs.xsh[x1], &self.obs.xsh[x2], &self.aux.xsh[xp]);
                let cp: [f64; 3] = std::array::from_fn(|k| p.center[k] + imgs * t[k]);
                let ra: [f64; 3] = std::array::from_fn(|k| a.center[k] - cp[k]);
                let rb: [f64; 3] = std::array::from_fn(|k| b.center[k] + imgs * l[k] - cp[k]);
                let add: Mat3 = std::array::from_fn(|x| {
                    std::array::from_fn(|z| c.ga[x] * ra[z] + c.gb[x] * rb[z])
                });
                Ok(Some(add))
            },
            |_, _, _, _, _, add: Mat3| {
                for x in 0..3 {
                    for z in 0..3 {
                        out[x][z] += add[x][z];
                    }
                }
                Ok(())
            },
        )?;
        Ok((out, count))
    }

    /// The `d/dP` third of one kept metric derivative block, scaled by the
    /// pieces' factors.
    fn sr2_block(
        &self,
        eng: &mut Engine,
        (xp, xq): (usize, usize),
        t: [f64; 3],
    ) -> Result<Option<Vec<f64>>, FerricError> {
        let (p, q) = (&self.aux.xsh[xp], &self.aux.xsh[xq]);
        let nb3 = 3 * p.nfun * q.nfun;
        let sc = self.aux.scale[xp] * self.aux.scale[xq];
        Ok(eng
            .compute_eri2_deriv_shifted(&self.aux.x, xp, xq, t)?
            .map(|blk| blk[..nb3].iter().map(|v| sc * v).collect()))
    }

    /// Kept SR metric force `(naux × 3, count)`: `Σ Wm d(P^c_0|Q^c_T)`,
    /// `d/dQ = −d/dP`, per element in walk order (as the unsplit `sr2_add`).
    pub(in crate::rsgdf) fn sr2_force(
        &self,
        st: &Stage<'_>,
        wm: &Array2<f64>,
        budget: usize,
    ) -> Result<(Array2<f64>, usize), FerricError> {
        self.sr2_force_with(st, budget, |_| wm)
    }

    /// [`SplitPlan::sr2_force`] with the weight of each pair image from
    /// `weights(T)` (the k-point forces: `Re Σ_q e^{iq·T} Wm(q)ᵀ` binned by
    /// the residue of `T`).
    pub(in crate::rsgdf) fn sr2_force_with<'w, W>(
        &self,
        st: &Stage<'_>,
        budget: usize,
        weights: W,
    ) -> Result<(Array2<f64>, usize), FerricError>
    where
        W: Fn([f64; 3]) -> &'w Array2<f64>,
    {
        let mut metric = Array2::<f64>::zeros((st.aux.nbasis(), 3));
        let pool = self.sr2_deriv_pool(st)?;
        let count = self.sr2_ordered(
            st,
            &pool,
            budget,
            |eng, xpq, t| self.sr2_block(eng, xpq, t),
            |ip, iq, t, blk: Vec<f64>| {
                let (p, q) = (&st.aux_sh[ip], &st.aux_sh[iq]);
                let nb = p.nfun * q.nfun;
                let wm = weights(t);
                for i in 0..p.nfun {
                    for j in 0..q.nfun {
                        let wv = wm[(p.off + i, q.off + j)];
                        if wv == 0.0 {
                            continue;
                        }
                        for x in 0..3 {
                            let v = wv * blk[x * nb + i * q.nfun + j];
                            metric[(p.off + i, x)] += v;
                            metric[(q.off + j, x)] -= v;
                        }
                    }
                }
                Ok(())
            },
        )?;
        Ok((metric, count))
    }

    /// Kept SR metric strain: `Σ_T gp ⊗ (C_P − C_Q − T)` per pair image.
    pub(in crate::rsgdf) fn sr2_strain(
        &self,
        st: &Stage<'_>,
        wm: &Array2<f64>,
        imgs: f64,
        budget: usize,
    ) -> Result<(Mat3, usize), FerricError> {
        let mut out = [[0.0_f64; 3]; 3];
        let pool = self.sr2_deriv_pool(st)?;
        let count = self.sr2_ordered(
            st,
            &pool,
            budget,
            |eng, (xp, xq), t| {
                let Some(blk) = self.sr2_block(eng, (xp, xq), t)? else {
                    return Ok(None);
                };
                let (p, q) = (&self.aux.xsh[xp], &self.aux.xsh[xq]);
                let nb = p.nfun * q.nfun;
                let mut gp3 = [0.0_f64; 3];
                for i in 0..p.nfun {
                    for j in 0..q.nfun {
                        let wv = wm[(p.off + i, q.off + j)];
                        if wv == 0.0 {
                            continue;
                        }
                        for x in 0..3 {
                            gp3[x] += wv * blk[x * nb + i * q.nfun + j];
                        }
                    }
                }
                let rel: [f64; 3] =
                    std::array::from_fn(|k| p.center[k] - q.center[k] - imgs * t[k]);
                let add: Mat3 = std::array::from_fn(|x| std::array::from_fn(|z| gp3[x] * rel[z]));
                Ok(Some(add))
            },
            |_, _, _, add: Mat3| {
                for x in 0..3 {
                    for z in 0..3 {
                        out[x][z] += add[x][z];
                    }
                }
                Ok(())
            },
        )?;
        Ok((out, count))
    }
}

// ---------------------------------------------------------------------------
// G space: forces
// ---------------------------------------------------------------------------

/// The LR (G ≠ 0) force pieces: orbital `natoms × 3`, J3 aux centre and
/// metric `naux × 3` (per aux function), and the `pair_ft_deriv` chunks.
pub(in crate::rsgdf) struct LrForce {
    pub(in crate::rsgdf) orb: Array2<f64>,
    pub(in crate::rsgdf) aux3: Array2<f64>,
    pub(in crate::rsgdf) metric: Array2<f64>,
    pub(in crate::rsgdf) n_chunks: usize,
}

impl LrForce {
    pub(in crate::rsgdf) fn zeros(natoms: usize, naux: usize) -> Self {
        Self {
            orb: Array2::zeros((natoms, 3)),
            aux3: Array2::zeros((naux, 3)),
            metric: Array2::zeros((naux, 3)),
            n_chunks: 0,
        }
    }
}

/// The per-chunk aux quantities of the moved-aux LR pass.
struct AuxChunk {
    /// `X`, `X_s`, `X_c` (re, im).
    x: (Array2<f64>, Array2<f64>),
    xs: (Array2<f64>, Array2<f64>),
    xc: (Array2<f64>, Array2<f64>),
    /// `Wt = w_LR X + w_SR X_s` (re, im).
    wt: (Array2<f64>, Array2<f64>),
    /// `Wm X`, `Wm X_s`, `Wm X_c` (re, im).
    z: (Array2<f64>, Array2<f64>),
    zs: (Array2<f64>, Array2<f64>),
    zc: (Array2<f64>, Array2<f64>),
}

impl AuxChunk {
    /// Computes the per-chunk aux FT quantities for the G chunk `gs` (Bohr⁻¹, Cartesian): `X`,
    /// `X_s`, `X_c` for the full, smooth and compact aux pieces, `Wt` from the `(w_LR, w_SR)`
    /// weights `w` per G, and the `Wm X` products with the `(naux, naux)` matrix `wm`.
    fn new(
        plan: &SplitPlan,
        st: &Stage<'_>,
        gs: &[[f64; 3]],
        w: &[(f64, f64)],
        wm: &Array2<f64>,
    ) -> Self {
        let naux = st.aux.nbasis();
        let x = reim(&aux_ft_shells(&st.aux_sh, naux, gs));
        let xs = reim(&aux_ft_shells(&plan.aux.s_sh, naux, gs));
        let xc = reim(&aux_ft_shells(&plan.aux.c_sh, naux, gs));
        let comb = |a: &Array2<f64>, b: &Array2<f64>| {
            Array2::from_shape_fn(a.dim(), |(p, g)| w[g].0 * a[(p, g)] + w[g].1 * b[(p, g)])
        };
        let wt = (comb(&x.0, &xs.0), comb(&x.1, &xs.1));
        let wmz =
            |a: &(Array2<f64>, Array2<f64>)| (mm(wm.view(), a.0.view()), mm(wm.view(), a.1.view()));
        let (z, zs, zc) = (wmz(&x), wmz(&xs), wmz(&xc));
        Self {
            x,
            xs,
            xc,
            wt,
            z,
            zs,
            zc,
        }
    }

    /// `G`-coefficient of `d/dC_R` of the chunk's metric energy at column
    /// `g` (module doc, LR J2): `(Y.im Z.re − Y.re Z.im) + (Zy.re X.im −
    /// Zy.im X.re) + w_SR [(Xc.im Zs.re − Xc.re Zs.im) + (Zc.re Xs.im − Zc.im
    /// Xs.re)]`, `Y = Wt`, `Zy = w_LR Z + w_SR Zs`.
    fn metric_coef(&self, r: usize, g: usize, (w_lr, w_sr): (f64, f64)) -> f64 {
        let (xr, xi) = (self.x.0[(r, g)], self.x.1[(r, g)]);
        let (xsr, xsi) = (self.xs.0[(r, g)], self.xs.1[(r, g)]);
        let (xcr, xci) = (self.xc.0[(r, g)], self.xc.1[(r, g)]);
        let (yr, yi) = (self.wt.0[(r, g)], self.wt.1[(r, g)]);
        let (zr, zi) = (self.z.0[(r, g)], self.z.1[(r, g)]);
        let (zsr, zsi) = (self.zs.0[(r, g)], self.zs.1[(r, g)]);
        let (zcr, zci) = (self.zc.0[(r, g)], self.zc.1[(r, g)]);
        let (zyr, zyi) = (w_lr * zr + w_sr * zsr, w_lr * zi + w_sr * zsi);
        (yi * zr - yr * zi)
            + (zyr * xi - zyi * xr)
            + w_sr * ((xci * zsr - xcr * zsi) + (zcr * xsi - zci * xsr))
    }
}

/// `acc[atom(μ)] += 2 Σ_ν Re[Q*_μν A_μν]` at column `g` (`a` = `(re, im)`
/// of `Σ_P Y_Pμν W_P`, rows `μ·n+ν`).
fn add_orbital(
    orb: &mut Array2<f64>,
    q: &[Array3<Complex64>; 3],
    a: &(Array2<f64>, Array2<f64>),
    aoat: &[usize],
    g: usize,
) {
    let n = aoat.len();
    for mu in 0..n {
        let mut acc = [0.0_f64; 3];
        for nu in 0..n {
            let mn = mu * n + nu;
            let (ar, ai) = (a.0[(mn, g)], a.1[(mn, g)]);
            for (c, qc) in q.iter().enumerate() {
                let qz = qc[[mu, nu, g]];
                acc[c] += qz.re * ar + qz.im * ai;
            }
        }
        for c in 0..3 {
            orb[(aoat[mu], c)] += 2.0 * acc[c];
        }
    }
}

impl SplitPlan {
    /// The moved-aux LR force pass over the FULL orbital pairs (module doc):
    /// orbital weight `Y (w_LR X + w_SR X_s)`, aux `−iG` on the same weight,
    /// metric of the split J2 form.
    pub(in crate::rsgdf) fn lr_force(
        &self,
        st: &Stage<'_>,
        gv: &[[f64; 3]],
        y: &Array2<f64>,
        wm: &Array2<f64>,
        natoms: usize,
        chunk_budget: usize,
    ) -> Result<LrForce, FerricError> {
        let n = st.obs.nbasis();
        let n2 = n * n;
        let naux = st.aux.nbasis();
        let (vol, omega) = (st.cell.volume(), st.omega);
        let aoat = ao_atoms(st.obs);
        let pair_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
        // P re/im + Σ_P Y Wt re/im (4 × 8 n²); X, X_s, X_c complex + their
        // re/im, Wt, Σ Y P, and the three Wm products (≈ 26 × 8 naux).
        let extra_per_g = n2
            .saturating_mul(32)
            .saturating_add(naux.saturating_mul(208))
            .saturating_add(64);
        let mut out = LrForce::zeros(natoms, naux);
        let n_chunks = pair_ft_deriv_chunked(
            st.cell,
            st.obs,
            gv,
            pair_thresh,
            chunk_budget,
            extra_per_g,
            |_g0, gs, p, q| {
                let w: Vec<(f64, f64)> = gs.iter().map(|g| kernel_weights(g, vol, omega)).collect();
                let ac = AuxChunk::new(self, st, gs, &w, wm);
                let (pr, pim) = pair_reim(p);
                let xy = (mm(y.t(), ac.wt.0.view()), mm(y.t(), ac.wt.1.view()));
                let (pyr, pyi) = (mm(y.view(), pr.view()), mm(y.view(), pim.view()));
                for (g, gvec) in gs.iter().enumerate() {
                    add_orbital(&mut out.orb, q, &xy, &aoat, g);
                    for pp in 0..naux {
                        // J3 aux: Re[(ΣYP)* (−iG Wt)] = G (PY.re Wt.im − PY.im Wt.re)
                        let t3 = pyr[(pp, g)] * ac.wt.1[(pp, g)] - pyi[(pp, g)] * ac.wt.0[(pp, g)];
                        let t2 = ac.metric_coef(pp, g, w[g]);
                        for c in 0..3 {
                            out.aux3[(pp, c)] += t3 * gvec[c];
                            out.metric[(pp, c)] += t2 * gvec[c];
                        }
                    }
                }
                Ok(())
            },
        )?;
        out.n_chunks = n_chunks;
        Ok(out)
    }

    /// `Y` restricted to the smooth AO pairs, `(naux, n_s²)`.
    fn smooth_y(sm: &SmoothObs, y: &Array2<f64>, n: usize) -> Array2<f64> {
        let ns = sm.prep.nbasis();
        Array2::from_shape_fn((y.nrows(), ns * ns), |(p, ab)| {
            y[(p, sm.ao_map[ab / ns] * n + sm.ao_map[ab % ns])]
        })
    }

    /// The NEW smooth-pair force pass (FINDINGS "Iteration 26"): the pair-FT
    /// derivative of the smooth orbital pieces with weight `w_SR X_c`
    /// (orbital), and `−iG` on `w_SR X_c` (aux). Added into `out`; a no-op
    /// when no orbital primitive is smooth or no aux primitive is compact
    /// (the build then has no `(ss | X_c)` block either).
    pub(in crate::rsgdf) fn lr_smooth_pair_force(
        &self,
        st: &Stage<'_>,
        gv: &[[f64; 3]],
        y: &Array2<f64>,
        chunk_budget: usize,
        out: &mut LrForce,
    ) -> Result<(), FerricError> {
        let Some(sm) = self
            .smooth_obs
            .as_ref()
            .filter(|_| !self.aux.c_sh.is_empty())
        else {
            return Ok(());
        };
        let naux = st.aux.nbasis();
        let ns2 = sm.prep.nbasis() * sm.prep.nbasis();
        let (vol, omega) = (st.cell.volume(), st.omega);
        let yss = Self::smooth_y(sm, y, st.obs.nbasis());
        let aoat = ao_atoms(&sm.prep);
        let pair_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
        let extra_per_g = ns2
            .saturating_mul(32)
            .saturating_add(naux.saturating_mul(64))
            .saturating_add(64);
        let n_chunks = pair_ft_deriv_chunked(
            st.cell,
            &sm.prep,
            gv,
            pair_thresh,
            chunk_budget,
            extra_per_g,
            |_g0, gs, p, q| {
                let w_sr: Vec<f64> = gs.iter().map(|g| kernel_weights(g, vol, omega).1).collect();
                let xc = aux_ft_shells(&self.aux.c_sh, naux, gs);
                let wc = (
                    Array2::from_shape_fn(xc.dim(), |(pp, g)| w_sr[g] * xc[(pp, g)].re),
                    Array2::from_shape_fn(xc.dim(), |(pp, g)| w_sr[g] * xc[(pp, g)].im),
                );
                let (pr, pim) = pair_reim(p);
                let xy = (mm(yss.t(), wc.0.view()), mm(yss.t(), wc.1.view()));
                let (pyr, pyi) = (mm(yss.view(), pr.view()), mm(yss.view(), pim.view()));
                for (g, gvec) in gs.iter().enumerate() {
                    add_orbital(&mut out.orb, q, &xy, &aoat, g);
                    for pp in 0..naux {
                        let t3 = pyr[(pp, g)] * wc.1[(pp, g)] - pyi[(pp, g)] * wc.0[(pp, g)];
                        for c in 0..3 {
                            out.aux3[(pp, c)] += t3 * gvec[c];
                        }
                    }
                }
                Ok(())
            },
        )?;
        out.n_chunks += n_chunks;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// G space: stress
// ---------------------------------------------------------------------------

/// The LR strain pieces `(2/Ω) Σ_G [dv Φ + v dΦ]` of J3 / J2 (WITHOUT the
/// `−δ E` volume terms) and the LR energies `e3`, `e2` those need.
pub(in crate::rsgdf) struct LrStrain {
    pub(in crate::rsgdf) s3: Mat3,
    pub(in crate::rsgdf) s2: Mat3,
    pub(in crate::rsgdf) e3: f64,
    pub(in crate::rsgdf) e2: f64,
    pub(in crate::rsgdf) n_chunks: usize,
}

/// `Re Σ conj(a) b` over rows at column `g` (`(re, im)` pairs).
fn re_dot(a: (&Array2<f64>, &Array2<f64>), b: (&Array2<f64>, &Array2<f64>), g: usize) -> f64 {
    let mut s = 0.0_f64;
    for r in 0..a.0.nrows() {
        s += a.0[(r, g)] * b.0[(r, g)] + a.1[(r, g)] * b.1[(r, g)];
    }
    s
}

/// `Re Σ_r conj(d[k, r, g]) b[r, g]`.
fn re_dot_d(d: &Array3<Complex64>, k: usize, b: (&Array2<f64>, &Array2<f64>), g: usize) -> f64 {
    let mut s = 0.0_f64;
    for r in 0..b.0.nrows() {
        let z = d[[k, r, g]];
        s += z.re * b.0[(r, g)] + z.im * b.1[(r, g)];
    }
    s
}

/// `Re Σ_mn conj(dP[mn, g]) b[mn, g]`.
fn re_dot_p(dp: &Array3<Complex64>, b: &(Array2<f64>, Array2<f64>), g: usize) -> f64 {
    let n = dp.dim().0;
    let mut s = 0.0_f64;
    for m in 0..n {
        for nu in 0..n {
            let z = dp[[m, nu, g]];
            let mn = m * n + nu;
            s += z.re * b.0[(mn, g)] + z.im * b.1[(mn, g)];
        }
    }
    s
}

/// The per-chunk aux quantities of the moved-aux LR strain pass.
struct AuxStrainChunk {
    x: (Array2<f64>, Array2<f64>),
    xs: (Array2<f64>, Array2<f64>),
    xc: (Array2<f64>, Array2<f64>),
    dx: Array3<Complex64>,
    dxs: Array3<Complex64>,
    dxc: Array3<Complex64>,
    z: (Array2<f64>, Array2<f64>),
    zs: (Array2<f64>, Array2<f64>),
    zc: (Array2<f64>, Array2<f64>),
}

/// Borrow both halves of a `(re, im)` pair.
fn v(a: &(Array2<f64>, Array2<f64>)) -> (&Array2<f64>, &Array2<f64>) {
    (&a.0, &a.1)
}

impl AuxStrainChunk {
    /// Computes the per-chunk aux FT values and their strain derivatives for the G chunk `gs`
    /// (Bohr⁻¹, Cartesian); `g_shape` is forwarded to `aux_ft_strain_shells`. `wm` is the `(naux,
    /// naux)` weight matrix.
    fn new(
        plan: &SplitPlan,
        st: &Stage<'_>,
        gs: &[[f64; 3]],
        wm: &Array2<f64>,
        g_shape: bool,
    ) -> Self {
        let naux = st.aux.nbasis();
        let (x, dx) = aux_ft_strain_shells(&st.aux_sh, naux, gs, g_shape);
        let (xs, dxs) = aux_ft_strain_shells(&plan.aux.s_sh, naux, gs, g_shape);
        let (xc, dxc) = aux_ft_strain_shells(&plan.aux.c_sh, naux, gs, g_shape);
        let (x, xs, xc) = (reim(&x), reim(&xs), reim(&xc));
        let wmz =
            |a: &(Array2<f64>, Array2<f64>)| (mm(wm.view(), a.0.view()), mm(wm.view(), a.1.view()));
        let (z, zs, zc) = (wmz(&x), wmz(&xs), wmz(&xc));
        Self {
            x,
            xs,
            xc,
            dx,
            dxs,
            dxc,
            z,
            zs,
            zc,
        }
    }

    /// `(Φ_A, Φ_M)` of the J2 form at column `g`: `Re X̄ Wm X` and
    /// `Re[X̄_s Wm X + X̄_c Wm X_s]`.
    fn metric_phis(&self, g: usize) -> (f64, f64) {
        let phi_a = re_dot(v(&self.x), v(&self.z), g);
        let phi_m = re_dot(v(&self.xs), v(&self.z), g) + re_dot(v(&self.xc), v(&self.zs), g);
        (phi_a, phi_m)
    }

    /// `(dΦ_A, dΦ_M)` for strain component `k`: `2 Re dX̄ Z` and
    /// `Re[dX̄_s Z + dX̄ Z_s + dX̄_c Z_s + dX̄_s Z_c]`.
    fn metric_dphis(&self, k: usize, g: usize) -> (f64, f64) {
        let d_a = 2.0 * re_dot_d(&self.dx, k, v(&self.z), g);
        let d_m = re_dot_d(&self.dxs, k, v(&self.z), g)
            + re_dot_d(&self.dx, k, v(&self.zs), g)
            + re_dot_d(&self.dxc, k, v(&self.zs), g)
            + re_dot_d(&self.dxs, k, v(&self.zc), g);
        (d_a, d_m)
    }
}

impl SplitPlan {
    /// The moved-aux LR strain pass over the FULL orbital pairs (module doc):
    /// J3 `w_LR Φ_L + w_SR Φ_S` with `Φ_L = Re P̄ Y X`, `Φ_S = Re P̄ Y X_s`
    /// (the smooth-pair part is [`SplitPlan::lr_smooth_pair_strain`]); J2
    /// `w_LR Φ_A + w_SR Φ_M`.
    pub(in crate::rsgdf) fn lr_strain(
        &self,
        st: &Stage<'_>,
        gv: &[[f64; 3]],
        y: &Array2<f64>,
        wm: &Array2<f64>,
        terms: &FitStrainTerms,
        chunk_budget: usize,
    ) -> Result<LrStrain, FerricError> {
        let n2 = st.obs.nbasis() * st.obs.nbasis();
        let naux = st.aux.nbasis();
        let (vol, omega) = (st.cell.volume(), st.omega);
        let pair_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
        // P re/im + two Σ_P Y X re/im + the weighted sum (8 × 8 n²); X, X_s,
        // X_c and their nine strain columns (30 × 16 naux) + re/im and the
        // Wm products (12 × 8 naux) + Σ Y P re/im.
        let extra_per_g = n2
            .saturating_mul(64)
            .saturating_add(naux.saturating_mul(592))
            .saturating_add(64);
        let mut out = LrStrain {
            s3: [[0.0; 3]; 3],
            s2: [[0.0; 3]; 3],
            e3: 0.0,
            e2: 0.0,
            n_chunks: 0,
        };
        let n_chunks = pair_ft_strain_chunked(
            st.cell,
            st.obs,
            gv,
            pair_thresh,
            chunk_budget,
            extra_per_g,
            terms.pair,
            |_g0, gs, p, dp| {
                let ac = AuxStrainChunk::new(self, st, gs, wm, terms.g_shape);
                let (pr, pim) = pair_reim(p);
                let xy = (mm(y.t(), ac.x.0.view()), mm(y.t(), ac.x.1.view()));
                let xys = (mm(y.t(), ac.xs.0.view()), mm(y.t(), ac.xs.1.view()));
                let (pyr, pyi) = (mm(y.view(), pr.view()), mm(y.view(), pim.view()));
                for (g, gvec) in gs.iter().enumerate() {
                    let w = kernel_weights(gvec, vol, omega);
                    let (d_lr, d_sr) = kernel_slopes(dot3(gvec, gvec), w, omega, terms.sr_kernel);
                    let phi_l = re_dot((&pr, &pim), (&xy.0, &xy.1), g);
                    let phi_s = re_dot((&pr, &pim), (&xys.0, &xys.1), g);
                    let (phi_a, phi_m) = ac.metric_phis(g);
                    out.e3 += w.0 * phi_l + w.1 * phi_s;
                    out.e2 += w.0 * phi_a + w.1 * phi_m;
                    for a in 0..3 {
                        for b in 0..3 {
                            let k = 3 * a + b;
                            let gg = if terms.g_shape {
                                -2.0 * gvec[a] * gvec[b]
                            } else {
                                0.0
                            };
                            // Re Σ dP̄ Y (w_LR X + w_SR X_s)
                            let mut dphi3 =
                                w.0 * re_dot_p(&dp[k], &xy, g) + w.1 * re_dot_p(&dp[k], &xys, g);
                            if terms.aux_ft3 {
                                for pp in 0..naux {
                                    let (dz, dzs) = (ac.dx[[k, pp, g]], ac.dxs[[k, pp, g]]);
                                    let (wr, wi) =
                                        (w.0 * dz.re + w.1 * dzs.re, w.0 * dz.im + w.1 * dzs.im);
                                    dphi3 += pyr[(pp, g)] * wr + pyi[(pp, g)] * wi;
                                }
                            }
                            let dphi2 = if terms.aux_ft2 {
                                let (d_a, d_m) = ac.metric_dphis(k, g);
                                w.0 * d_a + w.1 * d_m
                            } else {
                                0.0
                            };
                            out.s3[a][b] += gg * (d_lr * phi_l + d_sr * phi_s) + dphi3;
                            out.s2[a][b] += gg * (d_lr * phi_a + d_sr * phi_m) + dphi2;
                        }
                    }
                }
                Ok(())
            },
        )?;
        out.n_chunks = n_chunks;
        Ok(out)
    }

    /// The NEW smooth-pair strain pass: `Φ_ss = Re P̄_ss Y X_c` with `w_SR`,
    /// its kernel strain `dv_SR`, the smooth-piece pair-FT strain and the
    /// `X_c` strain; added into `out` (energy included, for the volume
    /// term). `smooth_pair = false` (the mutant) drops the `dP̄_ss` and
    /// `dX_c` terms only.
    pub(in crate::rsgdf) fn lr_smooth_pair_strain(
        &self,
        st: &Stage<'_>,
        gv: &[[f64; 3]],
        y: &Array2<f64>,
        terms: &FitStrainTerms,
        chunk_budget: usize,
        out: &mut LrStrain,
    ) -> Result<(), FerricError> {
        let Some(sm) = self
            .smooth_obs
            .as_ref()
            .filter(|_| !self.aux.c_sh.is_empty())
        else {
            return Ok(());
        };
        let naux = st.aux.nbasis();
        let ns2 = sm.prep.nbasis() * sm.prep.nbasis();
        let (vol, omega) = (st.cell.volume(), st.omega);
        let yss = Self::smooth_y(sm, y, st.obs.nbasis());
        let pair_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
        let extra_per_g = ns2
            .saturating_mul(32)
            .saturating_add(naux.saturating_mul(192))
            .saturating_add(64);
        let n_chunks = pair_ft_strain_chunked(
            st.cell,
            &sm.prep,
            gv,
            pair_thresh,
            chunk_budget,
            extra_per_g,
            terms.pair,
            |_g0, gs, p, dp| {
                let (xc, dxc) = aux_ft_strain_shells(&self.aux.c_sh, naux, gs, terms.g_shape);
                let (xcr, xci) = reim(&xc);
                let (pr, pim) = pair_reim(p);
                let xy = (mm(yss.t(), xcr.view()), mm(yss.t(), xci.view()));
                let (pyr, pyi) = (mm(yss.view(), pr.view()), mm(yss.view(), pim.view()));
                for (g, gvec) in gs.iter().enumerate() {
                    let w = kernel_weights(gvec, vol, omega);
                    let (_, d_sr) = kernel_slopes(dot3(gvec, gvec), w, omega, terms.sr_kernel);
                    let phi = re_dot((&pr, &pim), (&xy.0, &xy.1), g);
                    out.e3 += w.1 * phi;
                    for a in 0..3 {
                        for b in 0..3 {
                            let k = 3 * a + b;
                            let gg = if terms.g_shape {
                                -2.0 * gvec[a] * gvec[b]
                            } else {
                                0.0
                            };
                            let mut dphi = 0.0_f64;
                            if terms.smooth_pair {
                                dphi += re_dot_p(&dp[k], &xy, g);
                                if terms.aux_ft3 {
                                    for pp in 0..naux {
                                        let dz = dxc[[k, pp, g]];
                                        dphi += pyr[(pp, g)] * dz.re + pyi[(pp, g)] * dz.im;
                                    }
                                }
                            }
                            out.s3[a][b] += gg * d_sr * phi + w.1 * dphi;
                        }
                    }
                }
                Ok(())
            },
        )?;
        out.n_chunks += n_chunks;
        Ok(())
    }

    /// `(q, S)` of the two G = 0 volume terms: the kept `(q_c, S − S_ss)`,
    /// or the full `(q, S)` when `kept = false` (the `SplitFullG0` mutant).
    pub(in crate::rsgdf) fn g0_volume_inputs(
        &self,
        st: &Stage<'_>,
        s: &Array2<f64>,
        kept: bool,
    ) -> (Vec<f64>, Array2<f64>) {
        let naux = st.aux.nbasis();
        if !kept {
            let q = aux_ft_shells(&st.aux_sh, naux, &[[0.0; 3]])
                .column(0)
                .iter()
                .map(|z| z.re)
                .collect();
            return (q, s.clone());
        }
        let s_eff = match &self.s_ss {
            None => s.clone(),
            Some(s_ss) => s - s_ss,
        };
        (self.compact_charges(naux), s_eff)
    }
}

// ---------------------------------------------------------------------------
// G = 0: the smooth-piece overlap derivative (for crate::grad / crate::stress)
// ---------------------------------------------------------------------------

/// What `crate::grad` / `crate::stress` need of a split build's J3 G = 0
/// term `−c0 (S − S_ss) q_cᵀ` ([`split_g0`]): the compact aux charges and
/// the smooth-piece overlap `S_ss`'s derivative / virial over the build's
/// pair images (the image set `S_ss` is defined on — NOT hcore's).
pub(crate) struct SplitG0 {
    q_c: Vec<f64>,
    smooth: Option<SmoothObs>,
    images: Vec<[f64; 3]>,
}

/// The [`SplitG0`] of `gdf` (`None` for an unsplit build). `cell`, `obs`,
/// `aux` as for the forces (the build's).
pub(crate) fn split_g0(
    gdf: &RsGdf,
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    ledger: &mut Ledger,
) -> Result<Option<SplitG0>, FerricError> {
    if gdf.range_split().is_none() {
        return Ok(None);
    }
    let who = "RS-GDF range split G = 0";
    let gp = gdf.gradient_parts().ok_or_else(|| {
        FerricError::General(format!(
            "{who}: RsGdf has no gradient parts (build_for_gradient)"
        ))
    })?;
    check_obs_on_cell(cell, obs)?;
    require_pure_aux(aux, who)?;
    let stats = gdf.stats();
    if aux.nbasis() != stats.naux || obs.nbasis() != gdf.nao {
        return Err(FerricError::General(format!(
            "{who}: bases (nao {}, naux {}) are not the build's ({}, {})",
            obs.nbasis(),
            aux.nbasis(),
            gdf.nao,
            stats.naux
        )));
    }
    let st = Stage {
        cell,
        obs,
        aux,
        obs_sh: gshells(obs, "RS-GDF split G = 0 orbital basis")?,
        aux_sh: gshells(aux, "RS-GDF split G = 0 aux basis")?,
        omega: stats.omega,
        thresh: stats.precision,
        sr_screen: gp.sr_screen,
        walker: LatticeWalker::new(cell),
    };
    let rpair = pair_image_radius(&st, st.thresh);
    ledger.reserve(
        &format!("{who}: pair-image list (r_pair = {rpair:.2} Bohr)"),
        bytes_of(cell.translation_count_bound(rpair)?, 24),
    )?;
    let images = cell.translations(rpair)?;
    let Some(plan) = SplitPlan::for_derivatives(&st, gdf, &images, ledger)? else {
        return Ok(None);
    };
    Ok(Some(plan.into_split_g0(aux.nbasis(), images)))
}

impl SplitPlan {
    /// The [`SplitG0`] of this plan over the pair `images` the plan's `S_ss`
    /// is defined on (the k-point forces' image-resolved smooth-overlap
    /// derivative uses the same pieces).
    pub(in crate::rsgdf) fn into_split_g0(self, naux: usize, images: Vec<[f64; 3]>) -> SplitG0 {
        SplitG0 {
            q_c: self.compact_charges(naux),
            smooth: self.smooth_obs,
            images,
        }
    }
}

impl SplitG0 {
    /// `q_c` (naux).
    pub(crate) fn compact_charges(&self) -> &[f64] {
        &self.q_c
    }

    /// Visit every smooth-piece shifted overlap derivative block over the
    /// pair images: `visit(bra shell, ket shell, L, factor f_i f_j, block)`
    /// (`[bra xyz, ket xyz]`, each `n1 × n2`, smooth AO order).
    fn each_block<F>(&self, sm: &SmoothObs, mut visit: F) -> Result<(), FerricError>
    where
        F: FnMut(usize, usize, [f64; 3], f64, &[f64]),
    {
        let prep = &sm.prep;
        let nsh = prep.nshells();
        let mut eng = Engine::new_1e_deriv(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
        for l in &self.images {
            for s1 in 0..nsh {
                for s2 in 0..nsh {
                    if let Some(blk) = eng.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        visit(s1, s2, *l, sm.scale[s1] * sm.scale[s2], blk);
                    }
                }
            }
        }
        Ok(())
    }

    /// `Σ_mn w_mn dS_ss,mn/dR_A` (`natoms × 3`; `w` in the parent AO order,
    /// symmetric); zeros when no orbital primitive is smooth.
    pub(crate) fn overlap_force(
        &self,
        w: &Array2<f64>,
        natoms: usize,
    ) -> Result<Array2<f64>, FerricError> {
        self.overlap_force_by(|_| w, natoms)
    }

    /// `Σ_L Σ_mn w(L)_mn d⟨χ^s_m,0 | χ^s_n,L⟩/dR_A` with the weight of image
    /// `L` from `weight(L)` (parent AO order, row = bra at the origin; the
    /// k-point forces pass the phase-folded weight of `L`'s residue).
    pub(crate) fn overlap_force_by<'w, F>(
        &self,
        weight: F,
        natoms: usize,
    ) -> Result<Array2<f64>, FerricError>
    where
        F: Fn([f64; 3]) -> &'w Array2<f64>,
    {
        let mut g = Array2::<f64>::zeros((natoms, 3));
        let Some(sm) = &self.smooth else {
            return Ok(g);
        };
        let (dims, offs) = (sm.prep.shell_dims(), sm.prep.shell_offsets());
        let sh2at = sm.prep.shell_to_atom();
        self.each_block(sm, |s1, s2, l, f, blk| {
            let w = weight(l);
            let (n1, n2) = (dims[s1], dims[s2]);
            let bs = n1 * n2;
            let (a1, a2) = (sh2at[s1], sh2at[s2]);
            for i in 0..n1 {
                for j in 0..n2 {
                    let wv = f * w[(sm.ao_map[offs[s1] + i], sm.ao_map[offs[s2] + j])];
                    if wv == 0.0 {
                        continue;
                    }
                    let idx = i * n2 + j;
                    for c in 0..3 {
                        g[(a1, c)] += wv * blk[c * bs + idx];
                        g[(a2, c)] += wv * blk[(3 + c) * bs + idx];
                    }
                }
            }
        })?;
        Ok(g)
    }

    /// `Σ_mn w_mn dS_ss,mn/dε_ab` (the overlap virial of `crate::stress`:
    /// `½(∂_bra − ∂_ket)_a · (A − B − L)_b`); `drop_images` drops `L` from
    /// the pair vector (the `NoSrImages` stress mutant).
    pub(crate) fn overlap_virial(
        &self,
        w: &Array2<f64>,
        drop_images: bool,
    ) -> Result<Mat3, FerricError> {
        let mut out = [[0.0_f64; 3]; 3];
        let Some(sm) = &self.smooth else {
            return Ok(out);
        };
        let (dims, offs) = (sm.prep.shell_dims(), sm.prep.shell_offsets());
        let centers: Vec<[f64; 3]> = sm.prep.located_shells().iter().map(|s| s.center).collect();
        self.each_block(sm, |s1, s2, l, f, blk| {
            let (n1, n2) = (dims[s1], dims[s2]);
            let bs = n1 * n2;
            let li = if drop_images { [0.0; 3] } else { l };
            let rel: [f64; 3] = std::array::from_fn(|k| centers[s1][k] - centers[s2][k] - li[k]);
            let mut g = [0.0_f64; 3];
            for i in 0..n1 {
                for j in 0..n2 {
                    let wv = f * w[(sm.ao_map[offs[s1] + i], sm.ao_map[offs[s2] + j])];
                    if wv == 0.0 {
                        continue;
                    }
                    let idx = i * n2 + j;
                    for (c, gc) in g.iter_mut().enumerate() {
                        *gc += wv * 0.5 * (blk[c * bs + idx] - blk[(3 + c) * bs + idx]);
                    }
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    out[a][b] += g[a] * rel[b];
                }
            }
        })?;
        Ok(out)
    }
}
