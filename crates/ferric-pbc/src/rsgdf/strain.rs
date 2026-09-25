//! Strain derivative of the Gamma RS-GDF two-electron energy (the stress,
//! [`crate::stress`]'s `*_rsgdf` entry points) — the Rust port of
//! `reference/pbc/pbc_stress.py`'s `gdf_stress_2e` / `aux_ft_strain`
//! (FINDINGS "Iteration 19").
//!
//! # What is differentiated
//!
//! With the fitted densities `Y`, `Wm` of [`super::deriv`] (the forces'
//! Loewner metric weights, UNCHANGED), at fixed densities
//!
//! ```text
//! dE_2e/dε_ab = Σ Y dJ3/dε_ab + Σ Wm dJ2/dε_ab
//! ```
//!
//! under `r → (1 + ε) r` of the atoms, the aux centres (they must strain
//! homogeneously with the cell: atom-centred aux, or sites at fixed
//! fractional coordinates) and the lattice, G at fixed Miller index:
//!
//! * SR `J3` (erfc lattice sums, the energy's walk): each triplet
//!   `(μ_0 ν_L | P_T)` moves with `A`, `B′ = B + L`, `C′ = C_P + T`; with the
//!   origin at the aux image (translation invariance, no aux-slot block)
//!   `d/dε_ab = ∂_A,a (A − C′)_b + ∂_B′,a (B′ − C′)_b`.
//! * SR `J2`: `Σ_T ∂_P,a (P_0|Q_T) (C_P − C_Q − T)_b` (`d/dQ = −d/dP`).
//! * LR (`E = (2/Ω) Σ_{G∈half} v Φ`, `v = 4π/G² e^{−G²/4ω²}`):
//!   `−δ_ab E + (2/Ω) Σ [dv/dε_ab Φ + v dΦ/dε_ab]`, `dv/dε_ab = v′(G²)(−2 G_a G_b)`;
//!   `J3`: `Φ = Re Σ P̄_μν Y_Pμν X_P`, `dΦ = Re Σ dP̄ (YX) + Re Σ (Σ Y P̄)_P dX_P`;
//!   `J2`: `Φ = Re Σ X̄ Wm X`, `dΦ = 2 Re Σ dX̄ (Wm X)`.
//!   The pair-FT strain is [`crate::pair_ft::pair_ft_strain_chunked`]; the
//!   one-centre aux FT `X = e^{−iG·C} x(G)` has a strain-invariant phase, so
//!   only its G shape moves: `dX_ab = G_a G_b/(2α) X_prim − G_a X^g_b`.
//! * G = 0 (`c0 = π/(ω²Ω)`, `dc0/dε_ab = −c0 δ_ab`, `q` strain-invariant):
//!   `J3 −= c0 S qᵀ` gives `+δ_ab c0 Σ (Yq)·S` here and `−c0 Σ (Yq) dS`
//!   through the overlap virial (assembled in `crate::stress` as `M_g0`);
//!   `J2 −= c0 q qᵀ` gives `+δ_ab c0 qᵀ Wm q` — CONSTANT under atom motion
//!   (forces never see it), a first-class strain term (prototype 3–9e-2 if
//!   dropped).
//!
//! Every sum uses the build's ω, precision, SR screen mode, pair images, G
//! sphere and pair-FT threshold (the energy's truncation).

use super::{
    check_obs_on_cell, gshells, pair_image_radius, require_pure_aux, GShell, LatticeWalker, RsGdf,
    Stage, ENGINE_PRECISION,
};
use crate::budget::{bytes_of, Ledger};
use crate::hcore::{gvector_list_bytes, half_gvectors, G_CHUNK_BYTES};
use crate::lattice::Cell;
use crate::pair_ft::{pair_ft_strain_chunked, PairFtStrainTerms, DEFAULT_PAIR_FT_THRESH};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::md3c1e::{cart_components, e_table, ferric_cart2sph, MAX_L};
use ferric_integrals::operator::Operator;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::f64::consts::PI;

type Mat3 = [[f64; 3]; 3];

/// Which terms [`fit_strain`] includes (the stress mutation seams; production
/// sets every field).
#[derive(Debug, Clone, Copy)]
pub(crate) struct FitStrainTerms {
    /// Pair-FT parts (basis centres, G shape).
    pub(crate) pair: PairFtStrainTerms,
    /// The G-shape strain of the kernel `v(G)` and of the aux FT.
    pub(crate) g_shape: bool,
    /// The `−δ E_LR` volume terms of both LR parts.
    pub(crate) volume: bool,
    /// Image translations in the SR pair vectors.
    pub(crate) images: bool,
    /// Aux-FT strain in `J3` / in `J2`.
    pub(crate) aux_ft3: bool,
    pub(crate) aux_ft2: bool,
}

/// The RS-GDF strain pieces (each `dE/dε`, Hartree).
pub(crate) struct FitStrain {
    pub(crate) j3_sr: Mat3,
    pub(crate) j3_lr: Mat3,
    pub(crate) j2_sr: Mat3,
    pub(crate) j2_lr: Mat3,
    /// `+δ c0 Σ (Yq)·S`.
    pub(crate) j3_g0_volume: Mat3,
    /// `+δ c0 qᵀ Wm q`.
    pub(crate) j2_g0_volume: Mat3,
    pub(crate) n_sr3: usize,
    pub(crate) n_sr2: usize,
    pub(crate) n_g_half: usize,
    pub(crate) n_chunks: usize,
}

fn diag(x: f64) -> Mat3 {
    [[x, 0.0, 0.0], [0.0, x, 0.0], [0.0, 0.0, x]]
}

/// Contract `Y` with `dJ3/dε` and `Wm` with `dJ2/dε` (module doc). `obs`
/// must be the orbital basis B was built with (on `cell`'s atoms), `aux` its
/// aux basis; `y` `(naux, nao²)`, `wm` `(naux, naux)` from
/// [`super::deriv::fit_densities`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_strain(
    gdf: &RsGdf,
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    y: &Array2<f64>,
    wm: &Array2<f64>,
    terms: FitStrainTerms,
    ledger: &mut Ledger,
) -> Result<FitStrain, FerricError> {
    let gp = gdf
        .gradient_parts()
        .ok_or_else(|| FerricError::General("RS-GDF stress: RsGdf has no gradient parts".into()))?;
    check_obs_on_cell(cell, obs)?;
    require_pure_aux(aux, "RS-GDF stress")?;
    let stats = gdf.stats();
    let st = Stage {
        cell,
        obs,
        aux,
        obs_sh: gshells(obs, "RS-GDF stress orbital basis")?,
        aux_sh: gshells(aux, "RS-GDF stress aux basis")?,
        omega: stats.omega,
        thresh: stats.precision,
        sr_screen: gp.sr_screen,
        walker: LatticeWalker::new(cell),
    };
    let n = obs.nbasis();
    let n2 = n * n;
    let naux = aux.nbasis();
    if naux != stats.naux || y.dim() != (naux, n2) || wm.dim() != (naux, naux) || n != gdf.nao {
        return Err(FerricError::General(format!(
            "RS-GDF stress: aux basis has {naux} functions / orbital {n}, but B was built with \
             naux = {} / nao = {} (Y {:?}, Wm {:?})",
            stats.naux,
            gdf.nao,
            y.dim(),
            wm.dim()
        )));
    }
    let imgs = if terms.images { 1.0 } else { 0.0 };

    // --- SR 3-centre: the energy's pair images and triplet screen.
    let rpair = pair_image_radius(&st, st.thresh);
    ledger.reserve(
        &format!("RS-GDF stress pair-image list (r_pair = {rpair:.2} Bohr)"),
        bytes_of(cell.translation_count_bound(rpair)?, 24),
    )?;
    let images = cell.translations(rpair)?;
    let mut j3_sr = [[0.0_f64; 3]; 3];
    let mut eng3 = Engine::new_3center_deriv(Operator::erfc(st.omega), obs, aux, ENGINE_PRECISION)?;
    let n_sr3 = st.sr_three_index_walk(&images, |i1, i2, ip, l, t| {
        let blk = match eng3.compute_eri3_deriv_shifted(obs, aux, ip, i1, i2, [t, [0.0; 3], l])? {
            Some(b) => b,
            None => return Ok(()),
        };
        let (a, b, p) = (&st.obs_sh[i1], &st.obs_sh[i2], &st.aux_sh[ip]);
        let nb = p.nfun * a.nfun * b.nfun;
        let mut ga = [0.0_f64; 3];
        let mut gb = [0.0_f64; 3];
        for pp in 0..p.nfun {
            let prow = p.off + pp;
            for i in 0..a.nfun {
                let r0 = (a.off + i) * n + b.off;
                for j in 0..b.nfun {
                    let yv = y[(prow, r0 + j)];
                    if yv == 0.0 {
                        continue;
                    }
                    let idx = (pp * a.nfun + i) * b.nfun + j;
                    for x in 0..3 {
                        ga[x] += yv * blk[(3 + x) * nb + idx];
                        gb[x] += yv * blk[(6 + x) * nb + idx];
                    }
                }
            }
        }
        let cp = [
            p.center[0] + imgs * t[0],
            p.center[1] + imgs * t[1],
            p.center[2] + imgs * t[2],
        ];
        let bp = [
            b.center[0] + imgs * l[0],
            b.center[1] + imgs * l[1],
            b.center[2] + imgs * l[2],
        ];
        let ra = [
            a.center[0] - cp[0],
            a.center[1] - cp[1],
            a.center[2] - cp[2],
        ];
        let rb = [bp[0] - cp[0], bp[1] - cp[1], bp[2] - cp[2]];
        for x in 0..3 {
            for z in 0..3 {
                j3_sr[x][z] += ga[x] * ra[z] + gb[x] * rb[z];
            }
        }
        Ok(())
    })?;

    // --- SR metric: Σ_T ∂_P (P_0|Q_T) (C_P − C_Q − T).
    let mut j2_sr = [[0.0_f64; 3]; 3];
    let mut eng2 = Engine::new_2center_deriv(Operator::erfc(st.omega), aux, ENGINE_PRECISION)?;
    let n_sr2 = st.sr_metric_walk(|ip, iq, t| {
        let blk = match eng2.compute_eri2_deriv_shifted(aux, ip, iq, t)? {
            Some(b) => b,
            None => return Ok(()),
        };
        let (p, q) = (&st.aux_sh[ip], &st.aux_sh[iq]);
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
        let rel = [
            p.center[0] - q.center[0] - imgs * t[0],
            p.center[1] - q.center[1] - imgs * t[1],
            p.center[2] - q.center[2] - imgs * t[2],
        ];
        for x in 0..3 {
            for z in 0..3 {
                j2_sr[x][z] += gp3[x] * rel[z];
            }
        }
        Ok(())
    })?;

    // --- LR: the energy's half G sphere.
    let omega = st.omega;
    let gcut = 2.0 * omega * (1.0 / st.thresh).ln().sqrt();
    ledger.reserve(
        &format!("RS-GDF stress LR G list (|G| <= {gcut:.3})"),
        gvector_list_bytes(cell, gcut)?,
    )?;
    let gv = half_gvectors(cell, gcut)?;
    let vol = cell.volume();
    let pair_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
    // Per G: P re/im + Σ_P Y X re/im (4 × 8 n²); X and the nine dX (10 × 16
    // naux); Σ Y P re/im and Wm X re/im (4 × 8 naux).
    let extra_per_g = n2
        .saturating_mul(32)
        .saturating_add(naux.saturating_mul(192))
        .saturating_add(64);
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let aux_sh = &st.aux_sh;
    let (mut e3, mut e2) = (0.0_f64, 0.0_f64);
    let mut s3 = [[0.0_f64; 3]; 3];
    let mut s2 = [[0.0_f64; 3]; 3];
    let n_chunks = pair_ft_strain_chunked(
        cell,
        obs,
        &gv,
        pair_thresh,
        chunk_budget,
        extra_per_g,
        terms.pair,
        |_g0, gs, p, dp| {
            let ng = gs.len();
            let (x, dx) = aux_ft_strain_shells(aux_sh, naux, gs, terms.g_shape);
            let xr = x.mapv(|z| z.re);
            let xi = x.mapv(|z| z.im);
            let pr = Array2::from_shape_fn((n2, ng), |(mn, g)| p[[mn / n, mn % n, g]].re);
            let pim = Array2::from_shape_fn((n2, ng), |(mn, g)| p[[mn / n, mn % n, g]].im);
            // Σ_P Y[P,μν] X_P(G), (n², ng)
            let mut xyr = Array2::<f64>::zeros((n2, ng));
            let mut xyi = Array2::<f64>::zeros((n2, ng));
            general_mat_mul(1.0, &y.t(), &xr, 0.0, &mut xyr);
            general_mat_mul(1.0, &y.t(), &xi, 0.0, &mut xyi);
            // Σ_μν Y[P,μν] P_μν(G), (naux, ng)
            let mut pyr = Array2::<f64>::zeros((naux, ng));
            let mut pyi = Array2::<f64>::zeros((naux, ng));
            general_mat_mul(1.0, y, &pr, 0.0, &mut pyr);
            general_mat_mul(1.0, y, &pim, 0.0, &mut pyi);
            // (Wm X)_P(G), (naux, ng)
            let mut zr = Array2::<f64>::zeros((naux, ng));
            let mut zi = Array2::<f64>::zeros((naux, ng));
            general_mat_mul(1.0, wm, &xr, 0.0, &mut zr);
            general_mat_mul(1.0, wm, &xi, 0.0, &mut zi);
            for (g, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let v = 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp();
                let dvdg2 = -v * (1.0 / g2 + 1.0 / (4.0 * omega * omega));
                let fac = 2.0 / vol;
                let mut phi3 = 0.0_f64;
                for mn in 0..n2 {
                    phi3 += pr[(mn, g)] * xyr[(mn, g)] + pim[(mn, g)] * xyi[(mn, g)];
                }
                let mut phi2 = 0.0_f64;
                for pp in 0..naux {
                    phi2 += xr[(pp, g)] * zr[(pp, g)] + xi[(pp, g)] * zi[(pp, g)];
                }
                e3 += fac * v * phi3;
                e2 += fac * v * phi2;
                for a in 0..3 {
                    for b in 0..3 {
                        let k = 3 * a + b;
                        let dv = if terms.g_shape {
                            dvdg2 * (-2.0 * gvec[a] * gvec[b])
                        } else {
                            0.0
                        };
                        // J3: Re Σ dP̄ (YX) + Re Σ_P (Σ Y P̄)_P dX_P
                        let mut dphi3 = 0.0_f64;
                        let dpk = &dp[k];
                        for m in 0..n {
                            for nu in 0..n {
                                let z = dpk[[m, nu, g]];
                                let mn = m * n + nu;
                                dphi3 += z.re * xyr[(mn, g)] + z.im * xyi[(mn, g)];
                            }
                        }
                        let mut dphi2 = 0.0_f64;
                        for pp in 0..naux {
                            let dz = dx[[k, pp, g]];
                            if terms.aux_ft3 {
                                dphi3 += pyr[(pp, g)] * dz.re + pyi[(pp, g)] * dz.im;
                            }
                            if terms.aux_ft2 {
                                dphi2 += 2.0 * (dz.re * zr[(pp, g)] + dz.im * zi[(pp, g)]);
                            }
                        }
                        s3[a][b] += fac * (dv * phi3 + v * dphi3);
                        s2[a][b] += fac * (dv * phi2 + v * dphi2);
                    }
                }
            }
            Ok(())
        },
    )?;
    let mut j3_lr = s3;
    let mut j2_lr = s2;
    if terms.volume {
        for a in 0..3 {
            j3_lr[a][a] -= e3;
            j2_lr[a][a] -= e2;
        }
    }

    // --- G = 0 volume terms.
    let q: Vec<f64> = super::aux_ft_shells(&st.aux_sh, naux, &[[0.0; 3]])
        .column(0)
        .iter()
        .map(|z| z.re)
        .collect();
    let c0 = PI / (omega * omega * vol);
    let s_mat = gdf.overlap();
    let mut ysq = 0.0_f64;
    for (pp, qp) in q.iter().enumerate() {
        if *qp == 0.0 {
            continue;
        }
        let row = y.row(pp);
        let mut acc = 0.0_f64;
        for m in 0..n {
            for nu in 0..n {
                acc += row[m * n + nu] * s_mat[(m, nu)];
            }
        }
        ysq += qp * acc;
    }
    let mut qwq = 0.0_f64;
    for (pp, qp) in q.iter().enumerate() {
        for (rr, qr) in q.iter().enumerate() {
            qwq += qp * wm[(pp, rr)] * qr;
        }
    }
    let all = [&j3_sr, &j3_lr, &j2_sr, &j2_lr];
    if all
        .iter()
        .any(|m| m.iter().flatten().any(|v| !v.is_finite()))
    {
        return Err(FerricError::General(
            "RS-GDF stress: non-finite strain contraction".into(),
        ));
    }
    Ok(FitStrain {
        j3_sr,
        j3_lr,
        j2_sr,
        j2_lr,
        j3_g0_volume: diag(c0 * ysq),
        j2_g0_volume: diag(c0 * qwq),
        n_sr3,
        n_sr2,
        n_g_half: gv.len(),
        n_chunks,
    })
}

/// `(X, dX)`: `X[P, g]` the aux FT ([`super::aux_ft`]) and
/// `dX[3a + b, P, g] = dX_P(G_g)/dε_ab` at fixed Miller index. Per primitive
/// `X = e^{−iG·C} (π/α)^{3/2} e^{−G²/4α} Π_d F_d(G_d)`, `G·C` strain-invariant:
///
/// ```text
/// dX/dε_ab = G_a G_b/(2α) X_prim − G_a (common · ∂F_b/∂G_b · Π_{d≠b} F_d)
/// ```
///
/// `g_shape = false` (mutation seam) returns `dX = 0`.
fn aux_ft_strain_shells(
    shells: &[GShell],
    naux: usize,
    gvecs: &[[f64; 3]],
    g_shape: bool,
) -> (Array2<Complex64>, Array3<Complex64>) {
    let ng = gvecs.len();
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let mut out = Array2::<Complex64>::zeros((naux, ng));
    let mut outd = Array3::<Complex64>::zeros((9, naux, ng));
    let mut ebuf = vec![0.0_f64; (MAX_L + 1) * (MAX_L + 1)];
    for sh in shells {
        let l = sh.l;
        let comps = cart_components(l);
        let ncart = comps.len();
        let mut cart = vec![zero; ncart * ng];
        let mut cartd = vec![zero; 9 * ncart * ng];
        for (&a, &c) in sh.exps.iter().zip(&sh.coefs) {
            e_table(l, 0, a, 0.0, 0.0, &mut ebuf);
            let pref = c * (PI / a).powf(1.5);
            for (g, gv) in gvecs.iter().enumerate() {
                let g2 = gv[0] * gv[0] + gv[1] * gv[1] + gv[2] * gv[2];
                let mag = pref * (-g2 / (4.0 * a)).exp();
                let ph = gv[0] * sh.center[0] + gv[1] * sh.center[1] + gv[2] * sh.center[2];
                let common = Complex64::new(mag * ph.cos(), -mag * ph.sin());
                let mut f = [[zero; MAX_L + 1]; 3];
                let mut fg = [[zero; MAX_L + 1]; 3];
                for d in 0..3 {
                    let step = Complex64::new(0.0, -gv[d]);
                    for i in 0..=l {
                        let mut acc = zero;
                        let mut dacc = zero;
                        let mut pw = one; // (−iG)^t
                        let mut pwm1 = zero; // (−iG)^{t−1} (t ≥ 1)
                        for t in 0..=i {
                            let e = ebuf[i * (l + 1) + t];
                            acc += pw * e;
                            if t >= 1 {
                                // d/dG (−iG)^t = t (−i) (−iG)^{t−1}
                                dacc += pwm1 * Complex64::new(0.0, -(t as f64)) * e;
                            }
                            pwm1 = pw;
                            pw *= step;
                        }
                        f[d][i] = acc;
                        fg[d][i] = dacc;
                    }
                }
                for (u, lc) in comps.iter().enumerate() {
                    let (lx, ly, lz) = (lc[0] as usize, lc[1] as usize, lc[2] as usize);
                    let x0 = common * f[0][lx] * f[1][ly] * f[2][lz];
                    cart[u * ng + g] += x0;
                    if !g_shape {
                        continue;
                    }
                    let xg = [
                        common * fg[0][lx] * f[1][ly] * f[2][lz],
                        common * f[0][lx] * fg[1][ly] * f[2][lz],
                        common * f[0][lx] * f[1][ly] * fg[2][lz],
                    ];
                    for aa in 0..3 {
                        for bb in 0..3 {
                            let k = 3 * aa + bb;
                            cartd[(k * ncart + u) * ng + g] +=
                                x0 * (gv[aa] * gv[bb] / (2.0 * a)) - xg[bb] * gv[aa];
                        }
                    }
                }
            }
        }
        let emit = |src: &[Complex64], dst: &mut dyn FnMut(usize, usize, Complex64)| {
            if sh.pure {
                let c2s = ferric_cart2sph(l);
                let nf = sh.nfun;
                for j in 0..nf {
                    for g in 0..ng {
                        let mut acc = zero;
                        for m in 0..ncart {
                            let cm = c2s[m * nf + j];
                            if cm != 0.0 {
                                acc += src[m * ng + g] * cm;
                            }
                        }
                        dst(sh.off + j, g, acc);
                    }
                }
            } else {
                for u in 0..ncart {
                    for g in 0..ng {
                        dst(sh.off + u, g, src[u * ng + g]);
                    }
                }
            }
        };
        emit(&cart, &mut |pp: usize, g: usize, v: Complex64| {
            out[[pp, g]] = v
        });
        for k in 0..9 {
            let src = &cartd[k * ncart * ng..(k + 1) * ncart * ng];
            emit(src, &mut |pp: usize, g: usize, v: Complex64| {
                outd[[k, pp, g]] = v
            });
        }
    }
    (out, outd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rsgdf::aux_ft;
    use ferric_core::basis::{BasisSet, Shell};
    use ferric_core::mol::{Atom, Molecule};
    use std::collections::HashMap;

    fn one_shell(l: i32, pure: bool, a: f64, at: [f64; 3]) -> PreparedBasis {
        let mol = Molecule {
            atoms: vec![Atom {
                symbol: "H".into(),
                z: 1,
                x: at[0],
                y: at[1],
                zpos: at[2],
                ghost: false,
                n_core_ecp: 0,
            }],
            charge: 0,
            multiplicity: 2,
        };
        let mut m = HashMap::new();
        m.insert(
            1,
            vec![Shell {
                l,
                pure,
                exponents: vec![a],
                coefficients: vec![1.0],
            }],
        );
        let bs = BasisSet {
            name: "strain-test".into(),
            shells: m,
            ecps: HashMap::new(),
        };
        PreparedBasis::new(&mol, &bs).unwrap()
    }

    fn inv3(f: &Mat3) -> Mat3 {
        let det = f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1])
            - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0])
            + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0]);
        let mut o = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let (i1, i2) = ((j + 1) % 3, (j + 2) % 3);
                let (j1, j2) = ((i + 1) % 3, (i + 2) % 3);
                o[i][j] = (f[i1][j1] * f[i2][j2] - f[i1][j2] * f[i2][j1]) / det;
            }
        }
        o
    }

    /// `dX/dε_ab` vs central FD of the public `aux_ft` with the centre
    /// strained (`C → F C`) and G at fixed Miller index (`G → F^{−T} G`), for
    /// s, Cartesian p and pure d aux shells.
    #[test]
    fn aux_ft_strain_matches_fd() {
        let c = [0.4, -0.7, 1.1];
        let gs = [[0.9, -0.3, 0.5], [0.2, 1.4, -0.8], [-1.1, 0.6, 0.3]];
        let h = 1e-5;
        for (l, pure, a) in [(0, false, 0.7), (1, false, 1.3), (2, true, 0.9)] {
            let prep = one_shell(l, pure, a, c);
            let sh = gshells(&prep, "test").unwrap();
            let (x, dx) = aux_ft_strain_shells(&sh, prep.nbasis(), &gs, true);
            let x_ref = aux_ft(&prep, &gs).unwrap();
            let dmax = (&x - &x_ref).iter().fold(0.0_f64, |m, z| m.max(z.norm()));
            assert!(dmax < 1e-14, "X vs aux_ft: {dmax:e}");
            let (mut worst, mut fmax) = (0.0_f64, 0.0_f64);
            for ea in 0..3 {
                for eb in 0..3 {
                    let at = |s: f64| {
                        let mut f = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
                        f[ea][eb] += s * h;
                        let fi = inv3(&f);
                        let cs = [
                            f[0][0] * c[0] + f[0][1] * c[1] + f[0][2] * c[2],
                            f[1][0] * c[0] + f[1][1] * c[1] + f[1][2] * c[2],
                            f[2][0] * c[0] + f[2][1] * c[1] + f[2][2] * c[2],
                        ];
                        // G' = F^{−T} G:  G'_k = Σ_l (F^{−1})_{lk} G_l
                        let gp: Vec<[f64; 3]> = gs
                            .iter()
                            .map(|g| {
                                let mut o = [0.0; 3];
                                for (k, ok) in o.iter_mut().enumerate() {
                                    *ok = fi[0][k] * g[0] + fi[1][k] * g[1] + fi[2][k] * g[2];
                                }
                                o
                            })
                            .collect();
                        aux_ft(&one_shell(l, pure, a, cs), &gp).unwrap()
                    };
                    let (xp, xm) = (at(1.0), at(-1.0));
                    for p in 0..prep.nbasis() {
                        for g in 0..gs.len() {
                            let fd = (xp[[p, g]] - xm[[p, g]]) / (2.0 * h);
                            fmax = fmax.max(fd.norm());
                            worst = worst.max((dx[[3 * ea + eb, p, g]] - fd).norm());
                        }
                    }
                }
            }
            eprintln!("aux FT strain l={l}: max|dX − FD| {worst:.2e} (|FD| max {fmax:.2e})");
            assert!(fmax > 1e-3, "the strain must change X");
            assert!(worst < 1e-7 * fmax.max(1.0), "l={l}: {worst:e}");
        }
    }
}
