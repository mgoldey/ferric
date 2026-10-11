//! k-point STRAIN of the RS-GDF range split: the strain twins of
//! `ksplit`'s two LR force passes ([`SplitPlan::k_lr_force_moved`],
//! [`SplitPlan::k_lr_force_smooth`]) for [`crate::kstress`], per momentum
//! transfer `q` on the FULL `{G + q}` set.
//!
//! # What is differentiated (per `K = G + q ≠ 0`, scale `1/Ω` in the weights)
//!
//! ```text
//! E3 = Re Σ_K Σ_r Σ_ml p_r[ml] Σ_P Zr[ml,P] conj(w_LR X_P + w_SR Xs_P)   (full pairs)
//!    + Re Σ_K Σ_r Σ_ab pss_r[ab] Σ_P Zss_r[ab,P] conj(w_SR Xc_P)         (smooth pairs)
//! E2 = Σ_K [ w_LR Φ(X, X) + w_SR (Φ(Xs, X) + Φ(Xc, Xs)) ],  Φ(a1, a2) = Re Σ_P conj(a1_P) (Wmᵀ a2)_P
//! ```
//!
//! at FIXED `Z`, `Wm`. Under strain (Miller index and `k` fractions fixed, so
//! the Bloch and aux-image phases and the residue bins are invariant):
//!
//! * the kernel weights at `|K|`: `dw_LR/dK² = −w_LR (1/K² + 1/4ω²)`,
//!   `dw_SR/dK² = −w_SR/K² + w_LR/4ω²`, with `dK²/dε_ab = −2 K_a K_b` (the
//!   Gamma slopes of `split::deriv::kernel_slopes` at `|K|` instead of `|G|`;
//!   the `SplitNoSrKernelStrain` mutant zeroes the SR one);
//! * the residue pair-FT strain kernel `dp_r` of the full pairs (basis centres
//!   and G shape) and of the SMOOTH pieces (`dpss_r`);
//! * the aux-FT strain of `X`, `Xs`, `Xc` (G shape only: `K·C` is invariant);
//! * the `−δ E` volume terms, from the energies this pass accumulates (the
//!   smooth-pair energy included).
//!
//! The summed-over-residues pair kernels are the Gamma ones, so at `q = 0`,
//! `N_k = 1` this reproduces the Gamma split strain on the full sphere (the
//! Gamma pass runs the half sphere with weight 2).

use super::ksplit::k_kernel_weights;
use super::SplitPlan;
use crate::pair_ft::residues::pair_ft_strain_residues_chunked;
use crate::pair_ft::DEFAULT_PAIR_FT_THRESH;
use crate::rsgdf::kpoint::kderiv::{j3_strain_at_k, KFitStrainTerms, LrStrainAcc};
use crate::rsgdf::strain::aux_ft_strain_shells;
use crate::rsgdf::{dot3, Stage};
use ferric_core::FerricError;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64 as C64;

/// `(d w_LR/dK², d w_SR/dK²)` from the weights at `K²` (module doc);
/// `sr_kernel = false` zeroes the SR slope (the `SplitNoSrKernelStrain`
/// stress mutant).
fn k_kernel_slopes(
    k2: f64,
    (w_lr, w_sr): (f64, f64),
    omega: f64,
    terms: &KFitStrainTerms,
) -> (f64, f64) {
    let inv4w2 = 1.0 / (4.0 * omega * omega);
    let d_lr = -w_lr * (1.0 / k2 + inv4w2);
    let d_sr = if !terms.sr_kernel {
        0.0
    } else if terms.flip_slope {
        w_sr / k2 - w_lr * inv4w2
    } else {
        -w_sr / k2 + w_lr * inv4w2
    };
    (d_lr, d_sr)
}

/// `Re Σ_r Σ_ml p_r[ml] (Zr[r]·xc)[ml]` at column `gi` (the value part of
/// [`j3_strain_at_k`], without its derivatives).
fn j3_phi(zr: &[Array2<C64>], p: &[Array3<C64>], gi: usize, xc: &Array1<C64>) -> f64 {
    let n = p[0].dim().0;
    let mut phi = 0.0_f64;
    for (r, zrr) in zr.iter().enumerate() {
        let yv = zrr.dot(xc);
        let pr = &p[r];
        for ml in 0..n * n {
            phi += (pr[[ml / n, ml % n, gi]] * yv[ml]).re;
        }
    }
    phi
}

/// `Re Σ_P conj(a1_P) b_P`.
fn re_cdot(a1: &Array1<C64>, b: &Array1<C64>) -> f64 {
    a1.iter().zip(b).map(|(x, y)| (x.conj() * y).re).sum()
}

/// `Re Σ_P d_P c_P`.
fn re_dot(d: &[C64], c: &Array1<C64>) -> f64 {
    d.iter().zip(c).map(|(x, y)| (x * y).re).sum()
}

/// `Re Σ_P conj(d_P) c_P`.
fn re_dot_conj(d: &[C64], c: &Array1<C64>) -> f64 {
    d.iter().zip(c).map(|(x, y)| (x.conj() * y).re).sum()
}

/// One column of an `(naux, ng)` array.
fn col(x: &Array2<C64>, g: usize) -> Array1<C64> {
    x.column(g).to_owned()
}

/// One strain-component column `dx[k, :, g]`.
fn dcol(dx: &Array3<C64>, k: usize, g: usize) -> Vec<C64> {
    (0..dx.dim().1).map(|p| dx[[k, p, g]]).collect()
}

/// The aux FTs and their strain columns of one K chunk.
struct AuxPieces {
    x: Array2<C64>,
    dx: Array3<C64>,
    xs: Array2<C64>,
    dxs: Array3<C64>,
    xc: Array2<C64>,
    dxc: Array3<C64>,
}

impl SplitPlan {
    /// The split full-pair LR strain pass of one q class on `kfull` (module
    /// doc): `J3` with the pair weight `conj(w_LR X + w_SR Xs)` and `J2` with
    /// the three split metric terms; accumulates the strain and the energies
    /// into `out`. `zr[r]` is the residue-folded `Z` `(n², naux)`, `wm` the
    /// metric weight `Wm(q)`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rsgdf) fn k_lr_strain_moved(
        &self,
        st: &Stage<'_>,
        kfull: &[[f64; 3]],
        (q, moduli): ([f64; 3], [usize; 3]),
        (zr, wm): (&[Array2<C64>], &Array2<C64>),
        chunk_budget: usize,
        terms: &KFitStrainTerms,
        out: &mut LrStrainAcc,
    ) -> Result<usize, FerricError> {
        let n2 = st.obs.nbasis() * st.obs.nbasis();
        let naux = st.aux.nbasis();
        let (vol, omega) = (st.cell.volume(), st.omega);
        let pair_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
        let wmt = wm.t().to_owned();
        // Per K: X, Xs, Xc and their nine strain columns (30 complex naux),
        // the weights and metric products (≈ 14), the n² weight per residue.
        let extra_per_g = naux
            .saturating_mul(16 * 44)
            .saturating_add(n2.saturating_mul(16 * 2));
        pair_ft_strain_residues_chunked(
            st.cell,
            st.obs,
            kfull,
            moduli,
            pair_thresh,
            chunk_budget,
            extra_per_g,
            terms.pair,
            |_k0, ks, p, dp| {
                let gs = terms.g_shape;
                let (x, dx) = aux_ft_strain_shells(&st.aux_sh, naux, ks, gs);
                let (xs, dxs) = aux_ft_strain_shells(&self.aux.s_sh, naux, ks, gs);
                let (xc, dxc) = aux_ft_strain_shells(&self.aux.c_sh, naux, ks, gs);
                let aux = AuxPieces {
                    x,
                    dx,
                    xs,
                    dxs,
                    xc,
                    dxc,
                };
                for (gi, kvec) in ks.iter().enumerate() {
                    let w = k_kernel_weights(kvec, &q, 1.0 / vol, omega, false);
                    let slopes = k_kernel_slopes(dot3(kvec, kvec), w, omega, terms);
                    moved_at_k(
                        (&aux, gi, kvec),
                        (w, slopes),
                        (zr, p, dp),
                        (wm, &wmt),
                        terms,
                        out,
                    );
                }
                Ok(())
            },
        )
    }

    /// The smooth-pair LR strain pass of one q class (module doc): `Φ_ss =
    /// Re Σ pss Zss conj(w_SR Xc)` over the smooth orbital pieces, its kernel
    /// strain, the smooth-piece pair-FT strain and the `Xc` strain. A no-op
    /// without smooth pairs. `smooth_pair = false` (the stress mutant) drops
    /// the `dP̄_ss` and `dXc` terms only.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rsgdf) fn k_lr_strain_smooth(
        &self,
        st: &Stage<'_>,
        kfull: &[[f64; 3]],
        (q, moduli): ([f64; 3], [usize; 3]),
        zr: &[Array2<C64>],
        chunk_budget: usize,
        terms: &KFitStrainTerms,
        out: &mut LrStrainAcc,
    ) -> Result<usize, FerricError> {
        let Some(sm) = self.smooth_obs.as_ref().filter(|_| self.has_smooth_pairs()) else {
            return Ok(0);
        };
        let n = st.obs.nbasis();
        let ns = sm.prep.nbasis();
        let naux = st.aux.nbasis();
        let (vol, omega) = (st.cell.volume(), st.omega);
        let pair_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
        // Z restricted to the smooth AO pairs, per residue (as the force pass).
        let zss: Vec<Array2<C64>> = zr
            .iter()
            .map(|z| {
                Array2::from_shape_fn((ns * ns, naux), |(ab, pp)| {
                    z[(sm.ao_map[ab / ns] * n + sm.ao_map[ab % ns], pp)]
                })
            })
            .collect();
        let extra_per_g = naux
            .saturating_mul(16 * 14)
            .saturating_add((ns * ns).saturating_mul(16 * 2));
        pair_ft_strain_residues_chunked(
            st.cell,
            &sm.prep,
            kfull,
            moduli,
            pair_thresh,
            chunk_budget,
            extra_per_g,
            terms.pair,
            |_k0, ks, p, dp| {
                let (xc, dxc) = aux_ft_strain_shells(&self.aux.c_sh, naux, ks, terms.g_shape);
                for (gi, kvec) in ks.iter().enumerate() {
                    let w = k_kernel_weights(kvec, &q, 1.0 / vol, omega, false);
                    let (_, d_sr) = k_kernel_slopes(dot3(kvec, kvec), w, omega, terms);
                    let xcg = col(&xc, gi);
                    let wcv: Array1<C64> = xcg.mapv(|z| (z * w.1).conj());
                    let (phi_w, dphi, az) = j3_strain_at_k(&zss, p, dp, gi, &wcv);
                    // Φ_ss without the kernel weight (the slope's multiplier).
                    let phi = j3_phi(&zss, p, gi, &xcg.mapv(|z| z.conj()));
                    if terms.smooth_volume {
                        out.e3 += phi_w;
                    }
                    for a in 0..3 {
                        for b in 0..3 {
                            let k = 3 * a + b;
                            let gg = if terms.g_shape {
                                -2.0 * kvec[a] * kvec[b]
                            } else {
                                0.0
                            };
                            let mut d = 0.0_f64;
                            if terms.smooth_pair {
                                d += dphi[k];
                                if terms.aux_ft3 {
                                    // d conj(w_SR Xc) = w_SR conj(dXc)
                                    d += w.1 * re_dot_conj(&dcol(&dxc, k, gi), &az);
                                }
                            }
                            out.s3[a][b] += gg * d_sr * phi + d;
                        }
                    }
                }
                Ok(())
            },
        )
    }
}

/// One `K` of the moved full-pair strain pass: `J3` and `J2` contributions
/// into `out`. `w`, `slopes` are `(w_LR, w_SR)` and their `K²` slopes (scale
/// `1/Ω` inside).
fn moved_at_k(
    (aux, gi, kvec): (&AuxPieces, usize, &[f64; 3]),
    (w, (d_lr, d_sr)): ((f64, f64), (f64, f64)),
    (zr, p, dp): (&[Array2<C64>], &[Array3<C64>], &[[Array3<C64>; 9]]),
    (wm, wmt): (&Array2<C64>, &Array2<C64>),
    terms: &KFitStrainTerms,
    out: &mut LrStrainAcc,
) {
    let (xg, xsg, xcg) = (col(&aux.x, gi), col(&aux.xs, gi), col(&aux.xc, gi));
    // --- J3: the combined pair weight conj(w_LR X + w_SR Xs).
    let wcv: Array1<C64> =
        Array1::from_shape_fn(xg.len(), |pp| (xg[pp] * w.0 + xsg[pp] * w.1).conj());
    let (phi_w, dphi, az) = j3_strain_at_k(zr, p, dp, gi, &wcv);
    let phi_l = j3_phi(zr, p, gi, &xg.mapv(|z| z.conj()));
    let phi_s = j3_phi(zr, p, gi, &xsg.mapv(|z| z.conj()));
    // --- J2: Φ(a1, a2) = Re Σ conj(a1) (Wmᵀ a2) for the three split terms.
    let (wx, wxs) = (wmt.dot(&xg), wmt.dot(&xsg));
    let phi_a = re_cdot(&xg, &wx);
    let phi_m = re_cdot(&xsg, &wx) + re_cdot(&xcg, &wxs);
    out.e3 += phi_w;
    out.e2 += w.0 * phi_a + w.1 * phi_m;
    let (wcx, wcxs, wcxc) = (
        wm.dot(&xg.mapv(|z| z.conj())),
        wm.dot(&xsg.mapv(|z| z.conj())),
        wm.dot(&xcg.mapv(|z| z.conj())),
    );
    for a in 0..3 {
        for b in 0..3 {
            let k = 3 * a + b;
            let gg = if terms.g_shape {
                -2.0 * kvec[a] * kvec[b]
            } else {
                0.0
            };
            let (dx, dxs, dxc) = (
                dcol(&aux.dx, k, gi),
                dcol(&aux.dxs, k, gi),
                dcol(&aux.dxc, k, gi),
            );
            let mut d3 = dphi[k];
            let mut d2 = 0.0_f64;
            if terms.aux_ft3 {
                // d conj(w_LR X + w_SR Xs) = conj(w_LR dX + w_SR dXs)
                let dw: Vec<C64> = dx
                    .iter()
                    .zip(&dxs)
                    .map(|(d, s)| d * w.0 + s * w.1)
                    .collect();
                d3 += re_dot_conj(&dw, &az);
            }
            if terms.aux_ft2 {
                // A: (X, X); M1: (Xs, X); M2: (Xc, Xs).
                let d_a = re_dot_conj(&dx, &wx) + re_dot(&dx, &wcx);
                let d_m1 = re_dot_conj(&dxs, &wx) + re_dot(&dx, &wcxs);
                let d_m2 = re_dot_conj(&dxc, &wxs) + re_dot(&dxs, &wcxc);
                d2 = w.0 * d_a + w.1 * (d_m1 + d_m2);
            }
            out.s3[a][b] += gg * (d_lr * phi_l + d_sr * phi_s) + d3;
            out.s2[a][b] += gg * (d_lr * phi_a + d_sr * phi_m) + d2;
        }
    }
}
