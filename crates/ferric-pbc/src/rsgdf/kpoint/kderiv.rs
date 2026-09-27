//! Two-electron derivative of the k-point RS-GDF energy for the k-point
//! forces ([`crate::kgrad`]) — the Rust port of
//! `reference/pbc/pbc_kgrad_gdf.py` (FINDINGS "Iteration 21 (Python, k-point
//! RHF/UHF forces)", item 6).
//!
//! # What is differentiated
//!
//! Per momentum-transfer class q and `k' ` (`k = k' − q`), with `J3`, `J2(q)`
//! EXACTLY as [`super::KRsGdf::build`] forms them (the same private helpers,
//! every q class built explicitly — the energy's time-reversal fill is
//! anchored to that at 5.8e-15):
//!
//! ```text
//! E_2e = Σ_q tr[f(J2(q)) H(q)],   f = U diag(1/s kept, 0 dropped) U^H
//! H(q) = ½ ρ ρ^H [q = 0] − (1/2N_k²) Σ_{k'} Σ_s tr[J3_P D_s(k') J3_Q^H D_s(k)]
//! ρ_P  = (1/N_k) Σ_k tr[J3^{kk}_P D(k)]
//! dE_2e = Re Σ_q Σ_{k'} Σ_P tr[dJ3_P Z^{k'}_P] + Re Σ_q tr[Wm(q) dJ2(q)]
//! Z^{k'}_P = −(1/N_k²) Σ_Q conj(f)_PQ T^{k'}_Q + [q = 0] conj(c_P) D(k')/N_k,
//!            T_Q = Σ_s D_s(k') J3_Q^H D_s(k),  c = f ρ
//!            (q = 0: Z → ½(Z + Z^H) in (l, m), the derivative of the energy's
//!            J3 Hermitisation)
//! Wm(q) = U (Lo ∘ U^H H U) U^H     (complex Daleckii–Krein; Lo kept–kept
//!          −f_i f_j, kept–dropped (f_i − f_j)/(s_i − s_j), dropped–dropped 0 —
//!          the FULL Loewner form, so an active lindep cut at any q is handled)
//! ```
//!
//! # Derivative integrals
//!
//! * SR `J3`: q- and k-independent 3-centre erfc derivatives, walked ONCE
//!   (the energy's pair images and screen) and contracted with the
//!   phase-folded `Re Zbin[r_L, r_T] = Re Σ_q Σ_{k'} e^{ik'·t_{r_L}}
//!   e^{−iq·t_{r_T}} Z^{k'}` (the energy's phases on the same residue bins).
//! * SR `J2`: the 2-centre erfc derivative walk, weight
//!   `Re Σ_q e^{iq·t_r} Wm(q)ᵀ` per aux-image residue; `d/dQ = −d/dP`.
//! * LR per q on the FULL `{G + q}` set (the energy's half sets at TRIM q are
//!   the same sum): orbital centres through the residue-resolved pair-FT
//!   derivative ([`crate::pair_ft::residues::pair_ft_deriv_residues_chunked`],
//!   ket `−iK p − Qb`), aux centres `d conj(X_P)/dC = +iK conj(X_P)`, metric
//!   `d conj(X_P)/dC_P = +iK conj(X_P)`, `dX_Q/dC_Q = −iK X_Q`.
//! * G = 0 (q = 0 only): `J3 −= c0 q_P S(k)` enters through `dS` as
//!   `M_g0(k) = −c0 Σ_P q_P Z^k_P`, returned per k for the caller's
//!   phase-weighted overlap-derivative pass (NO `1/N_k`: `Z` carries it).
//!
//! # Scope
//!
//! Aux centres must be the cell's atoms (no `aux_jac`); the energy build must
//! be the production one (`G0Handling::Consistent`, no mutation other than
//! `NoTimeReversal`). The rebuilt per-q `B` is checked against the given
//! [`super::KRsGdf`] (kept counts per q and `‖B(k,k')‖_F` to 1e-8 relative):
//! a force for a different build is refused, not returned.

use super::{
    hermitize_pairs, lr_accumulate_q, lr_kvectors, sr_metric_binned, sr_three_index_binned,
    subtract_g0_three_index, KRsGdf, KRsGdfConfig, KRsGdfMutation,
};
use crate::budget::{bytes_of, Ledger};
use crate::hcore::{gvector_list_bytes, G_CHUNK_BYTES};
use crate::kpts::{lattice_coords, unit_root, KPointMesh};
use crate::kscf::eigh_herm;
use crate::lattice::Cell;
use crate::pair_ft::residues::{pair_ft_deriv_residues_chunked, residue_coords, residue_index};
use crate::pair_ft::DEFAULT_PAIR_FT_THRESH;
use crate::rsgdf::{
    aux_ft_shells, check_obs_on_cell, dot3, gshells, pair_image_radius, require_pure_aux,
    subtract_g0, G0Handling, LatticeWalker, Stage, ENGINE_PRECISION,
};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ndarray::{Array1, Array2};
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// TEST-ONLY defects of the fitted derivative (set by
/// [`crate::kgrad::KGradMutation`]).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct KFitMutation {
    /// `e^{+iq·T}` on the SR aux images of `dJ3` (the energy has `e^{−iq·T}`).
    pub(crate) aux_phase: bool,
    /// Drop `Σ Wm dJ2` (SR and LR).
    pub(crate) no_metric: bool,
    /// Drop the `J3` G = 0 term `M_g0`.
    pub(crate) no_g0: bool,
}

/// The fitted two-electron force pieces. Orbital parts per cell atom; aux
/// parts per aux FUNCTION (`naux × 3`, folded by the caller).
pub(crate) struct KFitGrad {
    pub(crate) orb_sr: Array2<f64>,
    pub(crate) orb_lr: Array2<f64>,
    pub(crate) aux_sr: Array2<f64>,
    pub(crate) aux_lr: Array2<f64>,
    pub(crate) metric_sr: Array2<f64>,
    pub(crate) metric_lr: Array2<f64>,
    /// `M_g0(k)` per mesh point, `(n, n)` in the `tr[dS M]` layout.
    pub(crate) mg0: Vec<Array2<C64>>,
    /// Largest number of metric eigenvalues dropped at any q.
    pub(crate) n_dropped_max: usize,
    /// max relative `|‖B_rebuilt‖_F − ‖B_given‖_F|` over (k, k').
    pub(crate) b_norm_mismatch: f64,
    pub(crate) n_sr3: usize,
    pub(crate) n_sr2: usize,
    pub(crate) n_chunks: usize,
    pub(crate) n_k_lr: usize,
}

/// Row-major `(n, n)` view of column `p` of a `(n², naux)` J3/Z block.
fn column_matrix(x: &Array2<C64>, p: usize, n: usize) -> Array2<C64> {
    Array2::from_shape_fn((n, n), |(m, l)| x[(m * n + l, p)])
}

/// Loewner (Daleckii–Krein) matrix of `f(s) = 1/s` (kept) / 0 (dropped).
fn loewner(s: &[f64], keep: &[bool]) -> (Vec<f64>, Array2<f64>) {
    let na = s.len();
    let f: Vec<f64> = (0..na)
        .map(|i| if keep[i] { 1.0 / s[i] } else { 0.0 })
        .collect();
    let lo = Array2::from_shape_fn((na, na), |(i, j)| {
        if keep[i] && keep[j] {
            -f[i] * f[j]
        } else {
            let ds = s[i] - s[j];
            if ds.abs() < 1e-300 {
                0.0
            } else {
                (f[i] - f[j]) / ds
            }
        }
    });
    (f, lo)
}

fn herm(m: &Array2<C64>) -> Array2<C64> {
    m.t().mapv(|z| z.conj())
}

/// The k-point RS-GDF two-electron force pieces (module doc). `d_total[k]`
/// is `D(k)` (both spins); `exch` the exchange terms `Σ_s D_s X D_s =
/// Σ_(c, D) c·D X D` per k (restricted `[(½, D)]`, unrestricted
/// `[(1, D_α), (1, D_β)]`). `s_k` must be the `S(k)` the build used.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kpoint_fit_gradient(
    gdf: &KRsGdf,
    cfg: &KRsGdfConfig,
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    mesh: &KPointMesh,
    s_k: &[Array2<C64>],
    d_total: &[Array2<C64>],
    exch: &[(f64, &[Array2<C64>])],
    mutation: KFitMutation,
    ledger: &mut Ledger,
) -> Result<KFitGrad, FerricError> {
    let who = "k-point RS-GDF forces";
    let g = &cfg.gdf;
    g.validate()?;
    match cfg.mutation {
        None | Some(KRsGdfMutation::NoTimeReversal) => {}
        Some(m) => {
            return Err(FerricError::General(format!(
                "{who}: the KRsGdf config carries the energy mutant {m:?}; forces differentiate \
                 the production energy only"
            )))
        }
    }
    if g.g0 != G0Handling::Consistent {
        return Err(FerricError::General(format!(
            "{who}: G = 0 handling {:?} is a negative-control mutant; forces need Consistent",
            g.g0
        )));
    }
    require_pure_aux(aux, who)?;
    check_obs_on_cell(cell, obs)?;
    let n = obs.nbasis();
    let n2 = n * n;
    let naux = aux.nbasis();
    let nk = mesh.nk();
    if gdf.nk() != nk
        || gdf.stats().naux != naux
        || gdf.stats().per_q.len() != nk
        || s_k.len() != nk
        || d_total.len() != nk
        || s_k.iter().chain(d_total).any(|x| x.dim() != (n, n))
        || exch.iter().any(|(_, d)| d.len() != nk)
    {
        return Err(FerricError::General(format!(
            "{who}: inconsistent inputs (nao {n}, naux {naux} vs build {}, N_k {nk} vs build {})",
            gdf.stats().naux,
            gdf.nk()
        )));
    }
    let st = Stage {
        cell,
        obs,
        aux,
        obs_sh: gshells(obs, "k-point RS-GDF forces orbital basis")?,
        aux_sh: gshells(aux, "k-point RS-GDF forces aux basis")?,
        omega: g.omega,
        thresh: g.precision,
        sr_screen: g.sr_screen,
        walker: LatticeWalker::new(cell),
    };
    let mod_l = mesh.residue_moduli();
    let mod_t = mesh.n();
    let rl = mod_l[0] * mod_l[1] * mod_l[2];
    let rt = mod_t[0] * mod_t[1] * mod_t[2];
    let sat = |xs: &[usize]| xs.iter().fold(1u64, |a, &x| a.saturating_mul(x as u64));

    ledger.reserve(
        &format!("{who}: SR residue bins (R_L = {rl}, R_T = {rt}, naux = {naux}, nao = {n})"),
        bytes_of(
            sat(&[rt, naux, naux]).saturating_add(sat(&[rl, rt, n2, naux])),
            8,
        ),
    )?;
    ledger.reserve(
        &format!("{who}: phase-folded derivative weights Zbin + Wm bins"),
        bytes_of(
            sat(&[rl, rt, n2, naux]).saturating_add(sat(&[rt, naux, naux])),
            8,
        ),
    )?;
    ledger.reserve(
        &format!(
            "{who}: per-q working set (J3, T, Z per k' (N_k = {nk}), R_L residue folds, metric)"
        ),
        bytes_of(
            sat(&[3, nk, n2, naux])
                .saturating_add(sat(&[2, rl, n2, naux]))
                .saturating_add(sat(&[8, naux, naux])),
            16,
        ),
    )?;
    let rpair = pair_image_radius(&st, g.precision);
    ledger.reserve(
        &format!("{who}: pair-image list (r_pair = {rpair:.2} Bohr)"),
        bytes_of(cell.translation_count_bound(rpair)?, 24),
    )?;
    let images = cell.translations(rpair)?;
    let gcut = 2.0 * g.omega * (1.0 / g.precision).ln().sqrt();
    let qmax = (0..nk)
        .map(|iq| {
            let q = mesh.q_class(iq).0;
            dot3(&q, &q).sqrt()
        })
        .fold(0.0_f64, f64::max);
    ledger.reserve(
        &format!("{who}: LR K list (|K| <= {gcut:.3}, |q| <= {qmax:.3})"),
        gvector_list_bytes(cell, gcut + qmax)?.saturating_mul(3),
    )?;

    // --- The energy's SR bins (same walk, same order).
    let (j2res, _) = sr_metric_binned(&st, mod_t)?;
    let (j3res, _) = sr_three_index_binned(&st, &images, mod_l, mod_t)?;
    let qv: Vec<f64> = aux_ft_shells(&st.aux_sh, naux, &[[0.0; 3]])
        .column(0)
        .iter()
        .map(|z| z.re)
        .collect();
    let c0 = PI / (g.omega * g.omega * cell.volume());
    let phk: Vec<Vec<C64>> = (0..nk)
        .map(|k| {
            (0..rl)
                .map(|r| mesh.phase(k, residue_coords(r, mod_l)))
                .collect()
        })
        .collect();
    let dq = rt as i64;
    let q_phase = |mq: [i64; 3], r: usize| -> C64 {
        let c = residue_coords(r, mod_t);
        let mut t = 0i64;
        for i in 0..3 {
            t += mq[i] * c[i] * (dq / mod_t[i] as i64);
        }
        unit_root(t, dq)
    };

    let natoms = cell.positions().len();
    let aoat = crate::grad::ao_atoms(obs);
    let vol = cell.volume();
    let omega = g.omega;
    let inv_nk = 1.0 / nk as f64;
    let inv_nk2 = inv_nk * inv_nk;
    let pair_thresh = (0.01 * g.precision).min(DEFAULT_PAIR_FT_THRESH);
    let zero = C64::new(0.0, 0.0);

    let mut zbin: Vec<Array2<f64>> = (0..rl * rt).map(|_| Array2::zeros((n2, naux))).collect();
    let mut wmbin: Vec<Array2<f64>> = (0..rt).map(|_| Array2::zeros((naux, naux))).collect();
    let mut orb_ao = vec![[0.0_f64; 3]; n];
    let mut aux_lr = Array2::<f64>::zeros((naux, 3));
    let mut metric_lr = Array2::<f64>::zeros((naux, 3));
    let mut mg0: Vec<Array2<C64>> = (0..nk).map(|_| Array2::zeros((n, n))).collect();
    let mut n_dropped_max = 0usize;
    let mut b_norm_mismatch = 0.0_f64;
    let mut n_chunks = 0usize;
    let mut n_k_lr = 0usize;

    for iq in 0..nk {
        let (q, mq) = mesh.q_class(iq);
        let trim = mesh.q_minus(iq) == iq;
        let is_q0 = mq == [0, 0, 0];
        let php: Vec<C64> = (0..rt).map(|r| q_phase(mq, r)).collect();

        // --- J2(q), per-residue J3 accumulators: exactly the build's.
        let mut j2 = Array2::<C64>::zeros((naux, naux));
        for (bin, ph) in j2res.iter().zip(&php) {
            j2.zip_mut_with(bin, |z, &x| *z += *ph * x);
        }
        let mut acc: Vec<Array2<C64>> = (0..rl)
            .map(|r_l| {
                let mut a = Array2::<C64>::zeros((n2, naux));
                for (r_t, ph) in php.iter().enumerate() {
                    let ph = ph.conj();
                    a.zip_mut_with(&j3res[r_l * rt + r_t], |z, &x| *z += ph * x);
                }
                a
            })
            .collect();
        let (kv, half) = lr_kvectors(cell, mesh, iq, trim, gcut)?;
        let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
        n_chunks += lr_accumulate_q(&st, &kv, half, mod_l, &mut j2, &mut acc, chunk_budget)?;
        drop(kv);
        if is_q0 {
            let mut re = j2.mapv(|z| z.re);
            let mut no_j3 = Array2::<f64>::zeros((0, naux));
            subtract_g0(&mut re, &mut no_j3, &[], &qv, c0, g.g0);
            j2.zip_mut_with(&re, |z, &x| z.re = x);
        }
        let j2h = (&j2 + &herm(&j2)).mapv(|z| z * 0.5);
        drop(j2);
        let (evals, u) = eigh_herm(&j2h)
            .map_err(|e| FerricError::Lapack(format!("{who}: metric eigh at q class {iq}: {e}")))?;
        drop(j2h);
        let s: Vec<f64> = evals.to_vec();
        let keep: Vec<bool> = s.iter().map(|&x| x > g.lindep).collect();
        let nkeep = keep.iter().filter(|&&b| b).count();
        if nkeep == 0 {
            return Err(FerricError::General(format!(
                "{who}: every metric eigenvalue at q class {iq} is <= lindep {:e}",
                g.lindep
            )));
        }
        n_dropped_max = n_dropped_max.max(naux - nkeep);
        let kept_build = gdf.stats().per_q[iq].naux_kept;
        if kept_build != nkeep {
            return Err(FerricError::General(format!(
                "{who}: q class {iq} keeps {nkeep} metric eigenvectors here but {kept_build} in the \
                 given KRsGdf; pass the config the build used"
            )));
        }
        let kof: Vec<usize> = (0..nk).map(|j| mesh.k_minus_q(j, mq)).collect();

        // --- J3(k'−q, k') for every k'.
        let mut j3: Vec<Array2<C64>> = Vec::with_capacity(nk);
        for (j, ph_j) in phk.iter().enumerate() {
            let mut x = Array2::<C64>::zeros((n2, naux));
            for (a, ph) in acc.iter().zip(ph_j) {
                x.zip_mut_with(a, |z, &y| *z += *ph * y);
            }
            if is_q0 {
                subtract_g0_three_index(&mut x, &s_k[j], &qv, c0, g.g0);
                hermitize_pairs(&mut x, n);
            }
            j3.push(x);
        }
        drop(acc);

        // --- B check against the given build (rotation-invariant norm).
        let mut w = Array2::<C64>::zeros((naux, nkeep));
        let mut c = 0usize;
        for (k, &kp) in keep.iter().enumerate() {
            if !kp {
                continue;
            }
            let fct = 1.0 / s[k].sqrt();
            for r in 0..naux {
                w[(r, c)] = u[(r, k)].conj() * fct;
            }
            c += 1;
        }
        for (j, x) in j3.iter().enumerate() {
            let mine = x.dot(&w).iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            let given = gdf
                .block(kof[j], j)
                .iter()
                .map(|z| z.norm_sqr())
                .sum::<f64>()
                .sqrt();
            let rel = (mine - given).abs() / given.max(1e-300);
            b_norm_mismatch = b_norm_mismatch.max(rel);
        }
        drop(w);
        if b_norm_mismatch > 1e-8 {
            return Err(FerricError::General(format!(
                "{who}: the rebuilt fit differs from the given KRsGdf (relative ‖B‖ mismatch \
                 {b_norm_mismatch:.2e} at q class {iq}); pass the config and S(k) the build used"
            )));
        }

        // --- f(J2), T, H, Z.
        let (f, lo) = loewner(&s, &keep);
        let mut uf = u.clone();
        for (col, fv) in uf.columns_mut().into_iter().zip(&f) {
            let fv = *fv;
            for z in col {
                *z *= fv;
            }
        }
        let jinv = uf.dot(&herm(&u)); // U f U^H
        drop(uf);
        let mut hq = Array2::<C64>::zeros((naux, naux));
        let mut zt: Vec<Array2<C64>> = Vec::with_capacity(nk);
        for j in 0..nk {
            let k = kof[j];
            let mut tt = Array2::<C64>::zeros((n2, naux));
            for p in 0..naux {
                let jqh = herm(&column_matrix(&j3[j], p, n));
                let mut t = Array2::<C64>::zeros((n, n));
                for (cf, ds) in exch {
                    let x = ds[j].dot(&jqh).dot(&ds[k]);
                    t.scaled_add(C64::new(*cf, 0.0), &x);
                }
                for m in 0..n {
                    for l in 0..n {
                        tt[(m * n + l, p)] = t[(l, m)];
                    }
                }
            }
            // H += −(1/2N_k²) J3ᵀ Tt ; Z = −(1/N_k²) Tt f
            let h_j = j3[j].t().dot(&tt);
            hq.scaled_add(C64::new(-0.5 * inv_nk2, 0.0), &h_j);
            let mut z = tt.dot(&jinv);
            z.mapv_inplace(|v| v * (-inv_nk2));
            zt.push(z);
        }
        if is_q0 {
            let mut rho = Array1::<C64>::zeros(naux);
            for (j, x) in j3.iter().enumerate() {
                let d = &d_total[j];
                let v = Array1::from_shape_fn(n2, |mn| d[(mn % n, mn / n)]);
                rho += &x.t().dot(&v);
            }
            rho.mapv_inplace(|z| z * inv_nk);
            let cvec = jinv.dot(&rho);
            for p in 0..naux {
                for qq in 0..naux {
                    hq[(p, qq)] += rho[p] * rho[qq].conj() * 0.5;
                }
            }
            for (j, z) in zt.iter_mut().enumerate() {
                let d = &d_total[j];
                for m in 0..n {
                    for l in 0..n {
                        let dv = d[(l, m)] * inv_nk;
                        for p in 0..naux {
                            z[(m * n + l, p)] += cvec[p].conj() * dv;
                        }
                    }
                }
                hermitize_pairs(z, n);
                if !mutation.no_g0 {
                    let mg = &mut mg0[j];
                    for m in 0..n {
                        for l in 0..n {
                            let mut a = zero;
                            for (p, qp) in qv.iter().enumerate() {
                                a += z[(m * n + l, p)] * *qp;
                            }
                            mg[(l, m)] += a * (-c0);
                        }
                    }
                }
            }
        }
        drop(j3);

        // --- Metric weight Wm = U (Lo ∘ U^H H U) U^H.
        let wm = if mutation.no_metric {
            Array2::<C64>::zeros((naux, naux))
        } else {
            let a = herm(&u).dot(&hq).dot(&u);
            let b = Array2::from_shape_fn((naux, naux), |(i, j)| a[(i, j)] * lo[(i, j)]);
            u.dot(&b).dot(&herm(&u))
        };
        drop(hq);

        // --- Residue folds: Zr[r_L] = Σ_k' e^{ik'·t_rL} Z^{k'}; SR bins.
        let mut zr: Vec<Array2<C64>> = (0..rl).map(|_| Array2::zeros((n2, naux))).collect();
        for (j, z) in zt.iter().enumerate() {
            for (r_l, a) in zr.iter_mut().enumerate() {
                let ph = phk[j][r_l];
                a.zip_mut_with(z, |x, &y| *x += ph * y);
            }
        }
        drop(zt);
        for (r_l, a) in zr.iter().enumerate() {
            for (r_t, pt) in php.iter().enumerate() {
                let ph = if mutation.aux_phase { *pt } else { pt.conj() };
                zbin[r_l * rt + r_t].zip_mut_with(a, |x, &y| *x += (ph * y).re);
            }
        }
        for (r_t, pt) in php.iter().enumerate() {
            let bin = &mut wmbin[r_t];
            for p in 0..naux {
                for qq in 0..naux {
                    bin[(p, qq)] += (*pt * wm[(qq, p)]).re;
                }
            }
        }

        // --- LR on the FULL {G + q} set.
        let qn = dot3(&q, &q).sqrt();
        let g2cut = gcut * gcut;
        let kfull: Vec<[f64; 3]> = cell
            .gvectors(gcut + qn)?
            .into_iter()
            .map(|gv| [gv[0] + q[0], gv[1] + q[1], gv[2] + q[2]])
            .filter(|k| {
                let k2 = dot3(k, k);
                k2 > 0.0 && k2 <= g2cut
            })
            .collect();
        n_k_lr += kfull.len();
        let wmt = wm.t().to_owned();
        let aux_sh = &st.aux_sh;
        // Per K: X, conj(X) v, the n² weight vector per residue, aZ.
        let extra_per_g = naux
            .saturating_mul(16 * 6)
            .saturating_add(n2.saturating_mul(16 * 2));
        let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
        n_chunks += pair_ft_deriv_residues_chunked(
            cell,
            obs,
            &kfull,
            mod_l,
            pair_thresh,
            chunk_budget,
            extra_per_g,
            |_k0, ks, p, qd| {
                let x = aux_ft_shells(aux_sh, naux, ks);
                for (gi, kvec) in ks.iter().enumerate() {
                    let k2 = dot3(kvec, kvec);
                    let vlr = 4.0 * PI / k2 * (-k2 / (4.0 * omega * omega)).exp() / vol;
                    let xg: Array1<C64> = x.column(gi).to_owned();
                    let xcv: Array1<C64> = xg.mapv(|z| z.conj() * vlr);
                    let mut az = Array1::<C64>::zeros(naux);
                    for (r, zrr) in zr.iter().enumerate() {
                        // Yt_r[l, m] at index (m·n + l).
                        let yv = zrr.dot(&xcv);
                        let pr = &p[r];
                        let pv = Array1::from_shape_fn(n2, |ml| pr[[ml / n, ml % n, gi]]);
                        az += &zrr.t().dot(&pv);
                        for m in 0..n {
                            for l in 0..n {
                                let wv = yv[m * n + l];
                                if wv == zero {
                                    continue;
                                }
                                let pz = pr[[m, l, gi]];
                                for xx in 0..3 {
                                    let qb = qd[r][xx][[m, l, gi]];
                                    orb_ao[m][xx] += (qb * wv).re;
                                    let qk = C64::new(0.0, -kvec[xx]) * pz - qb;
                                    orb_ao[l][xx] += (qk * wv).re;
                                }
                            }
                        }
                    }
                    let wx = wmt.dot(&xg); // (Wmᵀ X)_P
                    let wcx = wm.dot(&xg.mapv(|z| z.conj())); // (Wm conj X)_Q
                    for pp in 0..naux {
                        let xc = xg[pp].conj();
                        for xx in 0..3 {
                            let ik = C64::new(0.0, kvec[xx]);
                            aux_lr[(pp, xx)] += (xc * ik * az[pp] * vlr).re;
                            if !mutation.no_metric {
                                metric_lr[(pp, xx)] += (xc * ik * wx[pp] * vlr).re;
                                metric_lr[(pp, xx)] += (xg[pp] * (-ik) * wcx[pp] * vlr).re;
                            }
                        }
                    }
                }
                Ok(())
            },
        )?;
    }

    // --- SR 3-centre derivative walk against the phase-folded Re Zbin.
    let b = cell.reciprocal();
    let sh2at = obs.shell_to_atom().to_vec();
    let mut orb_sr = Array2::<f64>::zeros((natoms, 3));
    let mut aux_sr = Array2::<f64>::zeros((naux, 3));
    let mut eng3 = Engine::new_3center_deriv(Operator::erfc(st.omega), obs, aux, ENGINE_PRECISION)?;
    let n_sr3 = st.sr_three_index_walk(&images, |i1, i2, ip, l, t| {
        let blk = match eng3.compute_eri3_deriv_shifted(obs, aux, ip, i1, i2, [t, [0.0; 3], l])? {
            Some(bk) => bk,
            None => return Ok(()),
        };
        let bin = &zbin[residue_index(lattice_coords(&b, &l), mod_l) * rt
            + residue_index(lattice_coords(&b, &t), mod_t)];
        let (a, bs, pa) = (&st.obs_sh[i1], &st.obs_sh[i2], &st.aux_sh[ip]);
        let nb = pa.nfun * a.nfun * bs.nfun;
        let mut ga = [0.0_f64; 3];
        let mut gb = [0.0_f64; 3];
        for pp in 0..pa.nfun {
            let prow = pa.off + pp;
            let mut gpx = [0.0_f64; 3];
            for i in 0..a.nfun {
                let r0 = (a.off + i) * n + bs.off;
                for j in 0..bs.nfun {
                    let yv = bin[(r0 + j, prow)];
                    if yv == 0.0 {
                        continue;
                    }
                    let idx = (pp * a.nfun + i) * bs.nfun + j;
                    for x in 0..3 {
                        gpx[x] += yv * blk[x * nb + idx];
                        ga[x] += yv * blk[(3 + x) * nb + idx];
                        gb[x] += yv * blk[(6 + x) * nb + idx];
                    }
                }
            }
            for x in 0..3 {
                aux_sr[(prow, x)] += gpx[x];
            }
        }
        for x in 0..3 {
            orb_sr[(sh2at[i1], x)] += ga[x];
            orb_sr[(sh2at[i2], x)] += gb[x];
        }
        Ok(())
    })?;

    // --- SR metric derivative walk against Re Σ_q e^{iq·t_r} Wm(q)ᵀ.
    let mut metric_sr = Array2::<f64>::zeros((naux, 3));
    let mut n_sr2 = 0usize;
    if !mutation.no_metric {
        let mut eng2 = Engine::new_2center_deriv(Operator::erfc(st.omega), aux, ENGINE_PRECISION)?;
        n_sr2 = st.sr_metric_walk(|ip, iq, t| {
            let blk = match eng2.compute_eri2_deriv_shifted(aux, ip, iq, t)? {
                Some(bk) => bk,
                None => return Ok(()),
            };
            let bin = &wmbin[residue_index(lattice_coords(&b, &t), mod_t)];
            let (p, q) = (&st.aux_sh[ip], &st.aux_sh[iq]);
            let nb = p.nfun * q.nfun;
            for i in 0..p.nfun {
                for j in 0..q.nfun {
                    let wv = bin[(p.off + i, q.off + j)];
                    if wv == 0.0 {
                        continue;
                    }
                    for x in 0..3 {
                        let v = wv * blk[x * nb + i * q.nfun + j];
                        metric_sr[(p.off + i, x)] += v;
                        metric_sr[(q.off + j, x)] -= v;
                    }
                }
            }
            Ok(())
        })?;
    }

    let mut orb_lr = Array2::<f64>::zeros((natoms, 3));
    for (mu, gv) in orb_ao.iter().enumerate() {
        for x in 0..3 {
            orb_lr[(aoat[mu], x)] += gv[x];
        }
    }
    let all = [&orb_sr, &orb_lr, &aux_sr, &aux_lr, &metric_sr, &metric_lr];
    if all.iter().any(|a| a.iter().any(|v| !v.is_finite()))
        || mg0
            .iter()
            .any(|m| m.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()))
    {
        return Err(FerricError::General(format!(
            "{who}: non-finite derivative contraction"
        )));
    }
    Ok(KFitGrad {
        orb_sr,
        orb_lr,
        aux_sr,
        aux_lr,
        metric_sr,
        metric_lr,
        mg0,
        n_dropped_max,
        b_norm_mismatch,
        n_sr3,
        n_sr2,
        n_chunks,
        n_k_lr,
    })
}
