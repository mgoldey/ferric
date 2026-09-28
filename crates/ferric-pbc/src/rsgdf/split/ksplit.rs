//! k-point pieces of the RS-GDF range split: the moved blocks of
//! [`crate::rsgdf::kpoint::KRsGdf::build`] per momentum transfer `q` and
//! their derivatives for [`crate::rsgdf::kpoint::kderiv`]. Rust port of
//! `reference/pbc/pbc_kgrad_gdf_split.py` (FINDINGS "Iteration 26 (Python,
//! range-split forces/stress/k-point)", item (f) and "For the Rust port"
//! items 4 and 5).
//!
//! # What changes at k (per q, `K = G + q`, `K ≠ 0`)
//!
//! ```text
//! J3^{kk'}_{ml,P} = SR_kept Σ_{L,T} e^{ik'·L} e^{−iq·T} [(m_0 l^c_L | P^c_T) + (m^c_0 l^s_L | P^c_T)]_erfc
//!                 + (1/Ω) Σ_K [ conj(v_LR X_P + v_SR X^s_P) a_ml(K) + v_SR conj(X^c_P) a^ss_ml(K) ]
//!                 − [q = 0] c0 q^c_P (S(k) − S_ss(k))_ml
//! J2(q)_{PQ}      = SR_kept Σ_T e^{iq·T} (P^c_0 | Q^c_T)_erfc
//!                 + (1/Ω) Σ_K [ v_LR conj(X_P) X_Q + v_SR (conj(X^s_P) X_Q + conj(X^c_P) X^s_Q) ]
//!                 − [q = 0] c0 q^c_P q^c_Q
//! ```
//!
//! `v_LR = 4π/K² e^{−K²/4ω²}`, `v_SR = 4π/K² − v_LR`, both at `|K| = |G + q|`:
//! the SR real-space sum of a moved block at momentum `q` IS
//! `Σ_{K∈G+q} v_SR(|K|) FT·FT` (Poisson summation with the `e^{iq·T}`
//! phase), so the Gamma kernel `v_SR(|G|)` is wrong at every `q ≠ 0`
//! (prototype: 1.05e-3 Ha in the energy at 1x1x3; invisible at 1x1x1).
//! `K = 0` lies only on the `q = 0` lattice, and `v_SR → π/ω²` is finite as
//! `K → 0`, so only `q = 0` subtracts a G = 0 term, with the KEPT inputs.
//! `a^ss` is the residue-binned Bloch pair FT of the smooth orbital pieces
//! (raw coefficients, no libint factor); `S_ss(k) = Σ_L e^{ik·L} S_ss(L)` is
//! the phase-folded image-resolved smooth-piece overlap over the build's
//! pair images.
//!
//! The J2 grouping is the non-cancelling `v_LR XᴴX + v_SR (X_sᴴX + X_cᴴX_s)`:
//! bitwise the unsplit metric when nothing moved (`X_s ≡ 0`); the cancelling
//! `v XᴴX − v_SR X_cᴴX_c` is not (2.8e-14 in J2, FINDINGS item (a)).
//!
//! # Derivatives (per q, on the FULL `{G + q}` set, as the unsplit k forces)
//!
//! * Full-pair residue derivative with the per-aux weight
//!   `conj(v_LR X + v_SR X_s)`; aux `+iK` on the same weight.
//! * NEW smooth-pair residue derivative with weight `conj(v_SR X_c)`, bra
//!   `Qb`, ket `−iK p − Qb` (translation identity at any `K`); aux `+iK`.
//! * Metric: `(v_LR, X, X) + (v_SR, X_s, X) + (v_SR, X_c, X_s)`, each
//!   `Re[vv conj(A1_P) iK (Wmᵀ A2)_P] + Re[vv A2_Q (−iK) (Wm conj A1)_Q]`.
//!
//! The SR walks are the Gamma split's kept calls binned by residue
//! ([`SplitPlan::sr_three_index_binned`], [`SplitPlan::sr_metric_binned`])
//! and their ordered-parallel derivative walks with the k weights.

use super::{SplitPlan, RANGE_SPLIT_NEG_EIG_GUARD};
use crate::budget::{bytes_of, Ledger};
use crate::hcore::ONE_E_ENGINE_PRECISION;
use crate::kpts::{lattice_coords, KPointMesh};
use crate::pair_ft::residues::{pair_ft_deriv_residues_chunked, pair_ft_residues_chunked};
use crate::pair_ft::DEFAULT_PAIR_FT_THRESH;
use crate::rsgdf::kpoint::kderiv::KLrForce;
use crate::rsgdf::{aux_ft_shells, dot3, RangeSplitMutant, RsGdfConfig, Stage};
use ferric_core::FerricError;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// `(scale v_LR(|K|), scale v_SR(|K|))`. `gamma_kernel` (TEST mutant):
/// `v_SR` at `|K − q|` (the Gamma kernel with the FTs still at `K`; limit
/// `π/ω²` at `K = q`).
fn k_kernel_weights(
    k: &[f64; 3],
    q: &[f64; 3],
    scale: f64,
    omega: f64,
    gamma_kernel: bool,
) -> (f64, f64) {
    let inv4w2 = 1.0 / (4.0 * omega * omega);
    let k2 = dot3(k, k);
    let full = scale * 4.0 * PI / k2;
    let w_lr = full * (-k2 * inv4w2).exp();
    if !gamma_kernel {
        return (w_lr, -full * (-k2 * inv4w2).exp_m1());
    }
    let d = [k[0] - q[0], k[1] - q[1], k[2] - q[2]];
    let d2 = dot3(&d, &d);
    let w_sr = if d2 > 1e-12 {
        -scale * 4.0 * PI / d2 * (-d2 * inv4w2).exp_m1()
    } else {
        scale * PI / (omega * omega)
    };
    (w_lr, w_sr)
}

/// Real and imaginary parts of a complex `(rows, cols)` array.
fn reim(a: &Array2<C64>) -> (Array2<f64>, Array2<f64>) {
    (a.mapv(|z| z.re), a.mapv(|z| z.im))
}

/// `(n², ng)` real and imaginary parts of one residue's pair FT `(n, n, ng)`.
fn pair_reim(p: &Array3<C64>) -> (Array2<f64>, Array2<f64>) {
    let (n, _, ng) = p.dim();
    let re = Array2::from_shape_fn((n * n, ng), |(mn, g)| p[[mn / n, mn % n, g]].re);
    let im = Array2::from_shape_fn((n * n, ng), |(mn, g)| p[[mn / n, mn % n, g]].im);
    (re, im)
}

/// `a[mn, g]` as a `(n², ng)` complex matrix.
fn pair_flat(p: &Array3<C64>) -> Array2<C64> {
    let (n, _, ng) = p.dim();
    Array2::from_shape_fn((n * n, ng), |(mn, g)| p[[mn / n, mn % n, g]])
}

impl SplitPlan {
    /// The plan of a k-point build at `cfg` (`None` without a range split):
    /// the Gamma plan's pieces, factors and screens on the build's stage and
    /// pair `images`, plus the `R_L` per-residue `(ss pair | X_c)`
    /// accumulators and the `N_k` kept overlaps `S(k) − S_ss(k)` reserved on
    /// `ledger` (`(r_l, nk)`). Only
    /// [`RangeSplitMutant::Production`] is accepted (the split mutants are
    /// Gamma energy anchors; the k-point negative controls are
    /// `KRsGdfMutation` / `KGradMutation`).
    pub(in crate::rsgdf) fn for_kpoint(
        st: &Stage<'_>,
        cfg: &RsGdfConfig,
        images: &[[f64; 3]],
        (r_l, nk): (usize, usize),
        ledger: &mut Ledger,
    ) -> Result<Option<Self>, FerricError> {
        let Some(rs) = cfg.range_split else {
            return Ok(None);
        };
        if rs.mutant != RangeSplitMutant::Production {
            return Err(FerricError::General(format!(
                "KRsGdf: the range-split mutant {:?} is a Gamma energy anchor; the k-point \
                 build takes RangeSplitMutant::Production only",
                rs.mutant
            )));
        }
        let plan = Self::new(st, rs, images, ledger)?;
        if let Some(sm) = &plan.smooth_obs {
            let ns = sm.prep.nbasis() as u64;
            let n = st.obs.nbasis() as u64;
            ledger.reserve(
                &format!("KRsGdf range split: S(k) − S_ss(k) (N_k = {nk}, nao = {n})"),
                bytes_of((nk as u64).saturating_mul(n.saturating_mul(n)), 16),
            )?;
            ledger.reserve(
                &format!(
                    "KRsGdf range split: per-residue (ss pair | X_c) accumulators (R_L = {r_l}, \
                     n_smooth = {ns}, naux = {})",
                    st.aux.nbasis()
                ),
                bytes_of(
                    (r_l as u64)
                        .saturating_mul(ns.saturating_mul(ns))
                        .saturating_mul(st.aux.nbasis() as u64),
                    16,
                ),
            )?;
        }
        Ok(Some(plan))
    }

    /// Whether the build has a moved `(ss pair | X_c)` block (some orbital
    /// primitive smooth AND some aux primitive compact).
    pub(in crate::rsgdf) fn has_smooth_pairs(&self) -> bool {
        self.smooth_obs.is_some() && !self.aux.c_sh.is_empty()
    }

    /// The counters of the partition (the Gamma build's names).
    pub(in crate::rsgdf) fn counters(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("rsgdf split orbital prims", self.obs.n_prims),
            ("rsgdf split orbital prims smooth", self.obs.n_smooth_prims),
            ("rsgdf split aux prims", self.aux.n_prims),
            ("rsgdf split aux prims smooth", self.aux.n_smooth_prims),
            (
                "rsgdf split smooth AOs",
                self.smooth_obs.as_ref().map_or(0, |s| s.prep.nbasis()),
            ),
        ]
    }

    /// `S(k) − S_ss(k)` per mesh point (the kept overlap of the `q = 0` G = 0
    /// term); `S_ss(k) = Σ_L e^{ik·L} f_i f_j ⟨χ^s_μ,0 | χ^s_ν,L⟩` over the
    /// pair `images`, Hermitised as `S(k)` is. `None` when no orbital
    /// primitive is smooth (the caller then uses `S(k)` itself, no
    /// arithmetic).
    pub(in crate::rsgdf) fn kept_overlaps(
        &self,
        st: &Stage<'_>,
        images: &[[f64; 3]],
        mesh: &KPointMesh,
        s_k: &[Array2<C64>],
    ) -> Result<Option<Vec<Array2<C64>>>, FerricError> {
        let Some(sm) = &self.smooth_obs else {
            return Ok(None);
        };
        let n = st.obs.nbasis();
        let nk = mesh.nk();
        let recip = st.cell.reciprocal();
        let prep = &sm.prep;
        let (offs, dims) = (prep.shell_offsets(), prep.shell_dims());
        let nsh = prep.nshells();
        let mut eng = Engine::new_1e(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
        let mut sss: Vec<Array2<C64>> = (0..nk).map(|_| Array2::zeros((n, n))).collect();
        for l in images {
            let nl = lattice_coords(&recip, l);
            let ph: Vec<C64> = (0..nk).map(|k| mesh.phase(k, nl)).collect();
            for i1 in 0..nsh {
                for i2 in 0..nsh {
                    let blk = eng.compute_1e_block_shifted(prep, i1, i2, *l)?;
                    let f = sm.scale[i1] * sm.scale[i2];
                    for a in 0..dims[i1] {
                        for b in 0..dims[i2] {
                            let (r, c) = (sm.ao_map[offs[i1] + a], sm.ao_map[offs[i2] + b]);
                            let v = f * blk[a * dims[i2] + b];
                            for (s, p) in sss.iter_mut().zip(&ph) {
                                s[(r, c)] += *p * v;
                            }
                        }
                    }
                }
            }
        }
        Ok(Some(
            s_k.iter()
                .zip(&sss)
                .map(|(s, x)| {
                    let h = Array2::from_shape_fn((n, n), |(i, j)| {
                        0.5 * (x[(i, j)] + x[(j, i)].conj())
                    });
                    s - &h
                })
                .collect(),
        ))
    }

    /// Moved-aux LR of one q class (module doc): `J2 += Σ_K conj(Y) X +
    /// v_SR conj(X_c) X_s`, `acc[r] += Σ_K a_r conj(Y)`, `Y = v_LR X + v_SR
    /// X_s`, weights `(fac/Ω)` (`fac = 2` on the half set of a
    /// time-reversal-invariant q, real `2 Re[·]` accumulation there).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rsgdf) fn k_lr_moved_aux(
        &self,
        st: &Stage<'_>,
        kv: &[[f64; 3]],
        half: bool,
        q: [f64; 3],
        moduli: [usize; 3],
        j2: &mut Array2<C64>,
        acc: &mut [Array2<C64>],
        chunk_budget: usize,
        gamma_kernel: bool,
    ) -> Result<usize, FerricError> {
        let n2 = st.obs.nbasis() * st.obs.nbasis();
        let naux = st.aux.nbasis();
        let (vol, omega) = (st.cell.volume(), st.omega);
        let fac = if half { 2.0 } else { 1.0 };
        // Pair re/im (or complex) copy per residue; X, X_s, X_c complex and
        // their eight real / four complex weighted copies, per K.
        let extra_per_g = n2
            .saturating_mul(16)
            .saturating_add(naux.saturating_mul(16 * 3 + 8 * 8 + 16 * 2))
            .saturating_add(64);
        let thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
        let one = C64::new(1.0, 0.0);
        let sink = |_k0: usize, ks: &[[f64; 3]], qs: &[Array3<C64>]| -> Result<(), FerricError> {
            let x = aux_ft_shells(&st.aux_sh, naux, ks);
            let xs = aux_ft_shells(&self.aux.s_sh, naux, ks);
            let xc = aux_ft_shells(&self.aux.c_sh, naux, ks);
            let w: Vec<(f64, f64)> = ks
                .iter()
                .map(|k| k_kernel_weights(k, &q, fac / vol, omega, gamma_kernel))
                .collect();
            let y =
                Array2::from_shape_fn(x.dim(), |(p, g)| x[(p, g)] * w[g].0 + xs[(p, g)] * w[g].1);
            let xcw = Array2::from_shape_fn(xc.dim(), |(p, g)| xc[(p, g)] * w[g].1);
            if half {
                let ((yr, yi), (xr, xi)) = (reim(&y), reim(&x));
                let ((xsr, xsi), (cr, ci)) = (reim(&xs), reim(&xcw));
                let mut j2r = Array2::<f64>::zeros((naux, naux));
                general_mat_mul(1.0, &yr, &xr.t(), 0.0, &mut j2r);
                general_mat_mul(1.0, &yi, &xi.t(), 1.0, &mut j2r);
                general_mat_mul(1.0, &cr, &xsr.t(), 1.0, &mut j2r);
                general_mat_mul(1.0, &ci, &xsi.t(), 1.0, &mut j2r);
                j2.zip_mut_with(&j2r, |z, &v| z.re += v);
                let mut m = Array2::<f64>::zeros((n2, naux));
                for (qr, a) in qs.iter().zip(acc.iter_mut()) {
                    let (pr, pim) = pair_reim(qr);
                    // Re[conj(Y) a] = a.re Y.re + a.im Y.im
                    general_mat_mul(1.0, &pr, &yr.t(), 0.0, &mut m);
                    general_mat_mul(1.0, &pim, &yi.t(), 1.0, &mut m);
                    a.zip_mut_with(&m, |z, &v| z.re += v);
                }
            } else {
                let ycw = y.mapv(|z| z.conj());
                let ccw = xcw.mapv(|z| z.conj());
                general_mat_mul(one, &ycw, &x.t(), one, &mut *j2);
                general_mat_mul(one, &ccw, &xs.t(), one, &mut *j2);
                for (qr, a) in qs.iter().zip(acc.iter_mut()) {
                    general_mat_mul(one, &pair_flat(qr), &ycw.t(), one, a);
                }
            }
            Ok(())
        };
        pair_ft_residues_chunked(
            st.cell,
            st.obs,
            kv,
            moduli,
            thresh,
            chunk_budget,
            extra_per_g,
            sink,
        )
    }

    /// The moved `(ss pair | X_c)` block of one q class:
    /// `acc[r][ml, P] += Σ_K a^ss_r,ml(K) v_SR conj(X_c,P(K))`, accumulated
    /// in the smooth AO basis and scattered into the parent rows at the end
    /// (a no-op without smooth pairs). Weights and half-set handling as
    /// [`SplitPlan::k_lr_moved_aux`].
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rsgdf) fn k_lr_smooth_pairs(
        &self,
        st: &Stage<'_>,
        kv: &[[f64; 3]],
        half: bool,
        q: [f64; 3],
        moduli: [usize; 3],
        acc: &mut [Array2<C64>],
        chunk_budget: usize,
        gamma_kernel: bool,
    ) -> Result<usize, FerricError> {
        let Some(sm) = self.smooth_obs.as_ref().filter(|_| self.has_smooth_pairs()) else {
            return Ok(0);
        };
        let n = st.obs.nbasis();
        let ns = sm.prep.nbasis();
        let ns2 = ns * ns;
        let naux = st.aux.nbasis();
        let (vol, omega) = (st.cell.volume(), st.omega);
        let fac = if half { 2.0 } else { 1.0 };
        let extra_per_g = ns2
            .saturating_mul(16)
            .saturating_add(naux.saturating_mul(16 * 3))
            .saturating_add(64);
        let thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
        let one = C64::new(1.0, 0.0);
        let mut acc_ss: Vec<Array2<C64>> = acc.iter().map(|_| Array2::zeros((ns2, naux))).collect();
        let sink = |_k0: usize, ks: &[[f64; 3]], qs: &[Array3<C64>]| -> Result<(), FerricError> {
            let xc = aux_ft_shells(&self.aux.c_sh, naux, ks);
            let w_sr: Vec<f64> = ks
                .iter()
                .map(|k| k_kernel_weights(k, &q, fac / vol, omega, gamma_kernel).1)
                .collect();
            let xcw = Array2::from_shape_fn(xc.dim(), |(p, g)| xc[(p, g)] * w_sr[g]);
            if half {
                let (cr, ci) = reim(&xcw);
                let mut m = Array2::<f64>::zeros((ns2, naux));
                for (qr, a) in qs.iter().zip(acc_ss.iter_mut()) {
                    let (pr, pim) = pair_reim(qr);
                    general_mat_mul(1.0, &pr, &cr.t(), 0.0, &mut m);
                    general_mat_mul(1.0, &pim, &ci.t(), 1.0, &mut m);
                    a.zip_mut_with(&m, |z, &v| z.re += v);
                }
            } else {
                let ccw = xcw.mapv(|z| z.conj());
                for (qr, a) in qs.iter().zip(acc_ss.iter_mut()) {
                    general_mat_mul(one, &pair_flat(qr), &ccw.t(), one, a);
                }
            }
            Ok(())
        };
        let n_chunks = pair_ft_residues_chunked(
            st.cell,
            &sm.prep,
            kv,
            moduli,
            thresh,
            chunk_budget,
            extra_per_g,
            sink,
        )?;
        for (a, a_ss) in acc.iter_mut().zip(&acc_ss) {
            for i in 0..ns {
                for j in 0..ns {
                    let row = sm.ao_map[i] * n + sm.ao_map[j];
                    let mut dst = a.row_mut(row);
                    dst += &a_ss.row(i * ns + j);
                }
            }
        }
        Ok(n_chunks)
    }

    /// The split full-pair LR force pass of one q class on the full
    /// `{G + q}` set `kfull` (module doc): orbital and aux `J3` pieces with
    /// the weight `conj(v_LR X + v_SR X_s)`, metric of the split J2 form
    /// (skipped when `no_metric`). `zr[r]` is the residue-folded `Z`
    /// `(n², naux)`, `wm` the metric weight `Wm(q)`. Added into `out`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rsgdf) fn k_lr_force_moved(
        &self,
        st: &Stage<'_>,
        kfull: &[[f64; 3]],
        q: [f64; 3],
        moduli: [usize; 3],
        zr: &[Array2<C64>],
        wm: &Array2<C64>,
        chunk_budget: usize,
        (gamma_kernel, no_metric): (bool, bool),
        out: &mut KLrForce,
    ) -> Result<usize, FerricError> {
        let n = st.obs.nbasis();
        let n2 = n * n;
        let naux = st.aux.nbasis();
        let (vol, omega) = (st.cell.volume(), st.omega);
        let pair_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
        let wmt = wm.t().to_owned();
        // Per K: X, X_s, X_c, the weight and the metric products (≈ 12
        // complex naux), the n² weight vector per residue.
        let extra_per_g = naux
            .saturating_mul(16 * 12)
            .saturating_add(n2.saturating_mul(16 * 2));
        pair_ft_deriv_residues_chunked(
            st.cell,
            st.obs,
            kfull,
            moduli,
            pair_thresh,
            chunk_budget,
            extra_per_g,
            |_k0, ks, p, qd| {
                let x = aux_ft_shells(&st.aux_sh, naux, ks);
                let xs = aux_ft_shells(&self.aux.s_sh, naux, ks);
                let xc = aux_ft_shells(&self.aux.c_sh, naux, ks);
                for (gi, kvec) in ks.iter().enumerate() {
                    let (vlr, vsr) = k_kernel_weights(kvec, &q, 1.0 / vol, omega, gamma_kernel);
                    let (xg, xsg, xcg) = (x.column(gi), xs.column(gi), xc.column(gi));
                    let wcv: Array1<C64> =
                        Array1::from_shape_fn(naux, |pp| (xg[pp] * vlr + xsg[pp] * vsr).conj());
                    let az = out.add_orbital(zr, p, qd, gi, kvec, &wcv, None);
                    out.add_aux(&wcv, &az, kvec);
                    if no_metric {
                        continue;
                    }
                    let (xg, xsg, xcg) = (xg.to_owned(), xsg.to_owned(), xcg.to_owned());
                    for (vv, a1, a2) in [(vlr, &xg, &xg), (vsr, &xsg, &xg), (vsr, &xcg, &xsg)] {
                        let wx = wmt.dot(a2); // (Wmᵀ A2)_P
                        let wca = wm.dot(&a1.mapv(|z| z.conj())); // (Wm conj A1)_Q
                        for pp in 0..naux {
                            for xx in 0..3 {
                                let ik = C64::new(0.0, kvec[xx]);
                                out.metric[(pp, xx)] += (a1[pp].conj() * ik * wx[pp] * vv).re;
                                out.metric[(pp, xx)] += (a2[pp] * (-ik) * wca[pp] * vv).re;
                            }
                        }
                    }
                }
                Ok(())
            },
        )
    }

    /// The NEW smooth-pair LR force pass of one q class (module doc): the
    /// residue pair-FT derivative of the smooth orbital pieces with weight
    /// `conj(v_SR X_c)` (orbital, routed to the parent AOs) and `+iK` on the
    /// same weight (aux). A no-op without smooth pairs.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rsgdf) fn k_lr_force_smooth(
        &self,
        st: &Stage<'_>,
        kfull: &[[f64; 3]],
        q: [f64; 3],
        moduli: [usize; 3],
        zr: &[Array2<C64>],
        chunk_budget: usize,
        gamma_kernel: bool,
        out: &mut KLrForce,
    ) -> Result<usize, FerricError> {
        let Some(sm) = self.smooth_obs.as_ref().filter(|_| self.has_smooth_pairs()) else {
            return Ok(0);
        };
        let n = st.obs.nbasis();
        let ns = sm.prep.nbasis();
        let naux = st.aux.nbasis();
        let (vol, omega) = (st.cell.volume(), st.omega);
        let pair_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
        // Z restricted to the smooth AO pairs, per residue.
        let zss: Vec<Array2<C64>> = zr
            .iter()
            .map(|z| {
                Array2::from_shape_fn((ns * ns, naux), |(ab, pp)| {
                    z[(sm.ao_map[ab / ns] * n + sm.ao_map[ab % ns], pp)]
                })
            })
            .collect();
        let extra_per_g = naux
            .saturating_mul(16 * 4)
            .saturating_add((ns * ns).saturating_mul(16 * 2));
        pair_ft_deriv_residues_chunked(
            st.cell,
            &sm.prep,
            kfull,
            moduli,
            pair_thresh,
            chunk_budget,
            extra_per_g,
            |_k0, ks, p, qd| {
                let xc = aux_ft_shells(&self.aux.c_sh, naux, ks);
                for (gi, kvec) in ks.iter().enumerate() {
                    let vsr = k_kernel_weights(kvec, &q, 1.0 / vol, omega, gamma_kernel).1;
                    let wcv: Array1<C64> = xc.column(gi).mapv(|z| (z * vsr).conj());
                    let az = out.add_orbital(&zss, p, qd, gi, kvec, &wcv, Some(&sm.ao_map[..]));
                    out.add_aux(&wcv, &az, kvec);
                }
                Ok(())
            },
        )
    }
}

/// The k-point split metric guard (FINDINGS "Iteration 23": a G = 0
/// bookkeeping error is one large negative metric eigenvalue that the lindep
/// cut would silently drop); a no-op without a split.
pub(in crate::rsgdf) fn check_metric_guard(
    split: bool,
    eig_min: f64,
    iq: usize,
) -> Result<(), FerricError> {
    if !split || eig_min >= -RANGE_SPLIT_NEG_EIG_GUARD {
        return Ok(());
    }
    Err(FerricError::General(format!(
        "KRsGdf range split: smallest metric eigenvalue {eig_min:.3e} < \
         -{RANGE_SPLIT_NEG_EIG_GUARD:e} at q class {iq}. A G = 0 bookkeeping error in the split \
         metric produces exactly this, and the lindep cut would silently drop it; refusing the fit"
    )))
}
