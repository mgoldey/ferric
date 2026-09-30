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
//! # Range split
//!
//! A build with [`crate::rsgdf::RsGdfConfig::range_split`] is differentiated
//! along the same partition (FINDINGS "Iteration 26", k-point part; `split`'s
//! `ksplit`): the SR derivative walks follow the kept calls / compact pairs
//! (ordered-parallel, no separate serial oracle), the full-pair LR pass takes
//! the weight `conj(v_LR X + v_SR X_s)` at `|K| = |G + q|`, a NEW smooth-pair
//! residue pass the weight `conj(v_SR X_c)`, the metric the three split J2
//! terms, and `M_g0 = −c0 Σ_P q^c_P Z_P` enters through `d(S − S_ss)`: the
//! caller subtracts the image-resolved smooth-overlap derivative
//! ([`KFitGrad::smooth_g0`]). `Wm(q)` is Hermitised before use (exact:
//! `Re tr[Wm dJ2h] = Re tr[½(Wm + Wm^H) dJ2]` for the Hermitised metric;
//! the `v_SR`-weighted terms amplify its roundoff asymmetry, FINDINGS (e)).
//!
//! # Scope
//!
//! Aux centres must be the cell's atoms (no `aux_jac`); the energy build must
//! be the production one (`G0Handling::Consistent`, no mutation other than
//! `NoTimeReversal`, range split `Production` only). The rebuilt per-q `B` is checked against the given
//! [`super::KRsGdf`] (kept counts per q and `‖B(k,k')‖_F` to 1e-8 relative):
//! a force for a different build is refused, not returned.

use super::{
    aux_charges, g0_inputs, hermitize_pairs, lr_accumulate_q_any, lr_kvectors,
    refuse_column_rotation, sr_bins_of, subtract_g0_three_index, KPairWalk, KRsGdf, KRsGdfConfig,
    KRsGdfMutation,
};
use crate::budget::{bytes_of, Ledger};
use crate::hcore::{gvector_list_bytes, G_CHUNK_BYTES};
use crate::kpts::{lattice_coords, unit_root, KPointMesh};
use crate::kscf::eigh_herm;
use crate::lattice::Cell;
use crate::ordered::window_budget;
use crate::pair_ft::residues::{pair_ft_deriv_residues_chunked, residue_coords, residue_index};
use crate::pair_ft::DEFAULT_PAIR_FT_THRESH;
use crate::rsgdf::deriv::{sr2_force, sr3_force, Y3};
use crate::rsgdf::split::SplitPlan;
use crate::rsgdf::{
    aux_ft_shells, check_obs_on_cell, dot3, gshells, pair_image_radius, require_pure_aux,
    subtract_g0, G0Handling, LatticeWalker, RsGdfConfig, SplitG0, Stage, ENGINE_PRECISION,
};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ndarray::{Array1, Array2, Array3};
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
    /// NOT a defect: the frozen serial SR derivative walks (bit-identity
    /// oracle of the ordered-parallel ones; unsplit builds only).
    pub(crate) serial_sr: bool,
    /// Range split: the moved blocks' derivative weights with the Gamma
    /// kernel `v_SR(|K − q|)` (the energy rebuild keeps `v_SR(|K|)`).
    pub(crate) split_gamma_kernel: bool,
    /// Range split: `M_g0` with every aux charge `q` instead of `q_c` (the
    /// caller also drops the smooth-overlap derivative: the unsplit G = 0
    /// term).
    pub(crate) split_full_g0: bool,
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
    /// Range split: the smooth-piece overlap over the build's pair images,
    /// whose derivative the caller contracts with the phase-folded `M_g0`
    /// and SUBTRACTS (`J3 −= c0 (S − S_ss) q_c`); `None` unsplit.
    pub(crate) smooth_g0: Option<SplitG0>,
}

/// The LR (K ≠ 0) force accumulators of the k-point fit derivative: per
/// parent AO (orbital centres), per aux function (J3 aux centres) and the
/// metric.
pub(in crate::rsgdf) struct KLrForce {
    pub(in crate::rsgdf) orb_ao: Vec<[f64; 3]>,
    pub(in crate::rsgdf) aux: Array2<f64>,
    pub(in crate::rsgdf) metric: Array2<f64>,
}

impl KLrForce {
    fn zeros(n: usize, naux: usize) -> Self {
        Self {
            orb_ao: vec![[0.0; 3]; n],
            aux: Array2::zeros((naux, 3)),
            metric: Array2::zeros((naux, 3)),
        }
    }

    /// Orbital centres at column `gi`, `K = kvec` (range-split passes): per
    /// residue `r`, `yv = Z_r · wcv` (`wcv` = the conjugated per-aux weight),
    /// bra `Qb`, ket `−iK p − Qb`; the pair basis's AO `m` is parent AO
    /// `ao_map[m]` (`None`: the parent basis). Returns
    /// `aZ_P = Σ_r Σ_ml Z_r[ml, P] p_r[ml]` for the aux term.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::rsgdf) fn add_orbital(
        &mut self,
        zr: &[Array2<C64>],
        p: &[Array3<C64>],
        qd: &[[Array3<C64>; 3]],
        gi: usize,
        kvec: &[f64; 3],
        wcv: &Array1<C64>,
        ao_map: Option<&[usize]>,
    ) -> Array1<C64> {
        let nb = p[0].dim().0;
        let zero = C64::new(0.0, 0.0);
        let mut az = Array1::<C64>::zeros(wcv.len());
        let parent = |m: usize| ao_map.map_or(m, |a| a[m]);
        for (r, zrr) in zr.iter().enumerate() {
            let yv = zrr.dot(wcv);
            let pr = &p[r];
            let pv = Array1::from_shape_fn(nb * nb, |ml| pr[[ml / nb, ml % nb, gi]]);
            az += &zrr.t().dot(&pv);
            for m in 0..nb {
                for l in 0..nb {
                    let wv = yv[m * nb + l];
                    if wv == zero {
                        continue;
                    }
                    let pz = pr[[m, l, gi]];
                    let (pm, pl) = (parent(m), parent(l));
                    for xx in 0..3 {
                        let qb = qd[r][xx][[m, l, gi]];
                        self.orb_ao[pm][xx] += (qb * wv).re;
                        let qk = C64::new(0.0, -kvec[xx]) * pz - qb;
                        self.orb_ao[pl][xx] += (qk * wv).re;
                    }
                }
            }
        }
        az
    }

    /// Aux centres of `J3` (range-split passes): `Re[wcv_P · iK · aZ_P]`.
    pub(in crate::rsgdf) fn add_aux(
        &mut self,
        wcv: &Array1<C64>,
        az: &Array1<C64>,
        kvec: &[f64; 3],
    ) {
        for pp in 0..wcv.len() {
            for xx in 0..3 {
                let ik = C64::new(0.0, kvec[xx]);
                self.aux[(pp, xx)] += (wcv[pp] * ik * az[pp]).re;
            }
        }
    }
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
    check_fit_inputs(gdf, cfg, cell, obs, aux, mesh, s_k, d_total, exch)?;
    let n = obs.nbasis();
    let n2 = n * n;
    let naux = aux.nbasis();
    let nk = mesh.nk();
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
    let plan = SplitPlan::for_kpoint(&st, g, &images, (rl, nk), ledger)?;
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

    // --- The energy's SR bins (same walk, same order; the kept calls of a
    // range split) and G = 0 inputs.
    let ((j2res, j3res, _, _), _) = sr_bins_of(
        &st,
        plan.as_ref(),
        &images,
        (mod_l, mod_t),
        ledger,
        who,
        KPairWalk::S2(None),
    )?;
    let (qv, s_kept) = g0_inputs(&st, plan.as_ref(), &images, mesh, s_k)?;
    // M_g0's charges: the build's (q_c with a split), or every aux charge
    // under the SplitFullG0 mutant.
    let q_g0 = if mutation.split_full_g0 {
        aux_charges(&st)
    } else {
        qv.clone()
    };
    let phk: Vec<Vec<C64>> = (0..nk)
        .map(|k| {
            (0..rl)
                .map(|r| mesh.phase(k, residue_coords(r, mod_l)))
                .collect()
        })
        .collect();
    let ctx = FitCtx {
        st: &st,
        gdf,
        g,
        mesh,
        plan: plan.as_ref(),
        j2res: &j2res,
        j3res: &j3res,
        qv: &qv,
        q_g0: &q_g0,
        s_g0: s_kept.as_deref().unwrap_or(s_k),
        phk: &phk,
        d_total,
        exch,
        mutation,
        gcut,
        c0: PI / (g.omega * g.omega * cell.volume()),
    };
    let mut acc = FitAcc {
        zbin: (0..rl * rt).map(|_| Array2::zeros((n2, naux))).collect(),
        wmbin: (0..rt).map(|_| Array2::zeros((naux, naux))).collect(),
        lr: KLrForce::zeros(n, naux),
        mg0: (0..nk).map(|_| Array2::zeros((n, n))).collect(),
        n_dropped_max: 0,
        b_norm_mismatch: 0.0,
        n_chunks: 0,
        n_k_lr: 0,
    };
    for iq in 0..nk {
        q_class_pass(&ctx, iq, ledger, &mut acc)?;
    }

    // --- SR 3-centre derivative walk against the phase-folded Re Zbin, and
    // the SR metric derivative walk against Re Σ_q e^{iq·t_r} Wm(q)ᵀ:
    // ordered-parallel, BIT-IDENTICAL to the frozen serial walks.
    let natoms = cell.positions().len();
    let bins = KSrBins {
        recip: cell.reciprocal(),
        mod_l,
        mod_t,
        zbin: &acc.zbin,
        wmbin: &acc.wmbin,
    };
    let budget = window_budget(ledger.remaining());
    let ((orb_sr, aux_sr, n_sr3), (metric_sr, n_sr2)) =
        k_sr_force_parts(&st, plan.as_ref(), &images, &bins, natoms, budget, mutation)?;

    let aoat = crate::grad::ao_atoms(obs);
    let mut orb_lr = Array2::<f64>::zeros((natoms, 3));
    for (mu, gv) in acc.lr.orb_ao.iter().enumerate() {
        for x in 0..3 {
            orb_lr[(aoat[mu], x)] += gv[x];
        }
    }
    let (aux_lr, metric_lr) = (acc.lr.aux, acc.lr.metric);
    let all = [&orb_sr, &orb_lr, &aux_sr, &aux_lr, &metric_sr, &metric_lr];
    if all.iter().any(|a| a.iter().any(|v| !v.is_finite()))
        || acc
            .mg0
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
        mg0: acc.mg0,
        n_dropped_max: acc.n_dropped_max,
        b_norm_mismatch: acc.b_norm_mismatch,
        n_sr3,
        n_sr2,
        n_chunks: acc.n_chunks,
        n_k_lr: acc.n_k_lr,
        smooth_g0: plan.map(|p| p.into_split_g0(naux, images)),
    })
}

/// The input checks of [`kpoint_fit_gradient`].
#[allow(clippy::too_many_arguments)]
fn check_fit_inputs(
    gdf: &KRsGdf,
    cfg: &KRsGdfConfig,
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    mesh: &KPointMesh,
    s_k: &[Array2<C64>],
    d_total: &[Array2<C64>],
    exch: &[(f64, &[Array2<C64>])],
) -> Result<(), FerricError> {
    let who = "k-point RS-GDF forces";
    let g = &cfg.gdf;
    g.validate()?;
    refuse_column_rotation(g, who)?;
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
    Ok(())
}

/// The loop invariants of the per-q passes of [`kpoint_fit_gradient`].
struct FitCtx<'a> {
    st: &'a Stage<'a>,
    gdf: &'a KRsGdf,
    g: &'a RsGdfConfig,
    mesh: &'a KPointMesh,
    plan: Option<&'a SplitPlan>,
    j2res: &'a [Array2<f64>],
    j3res: &'a [Array2<f64>],
    /// The build's G = 0 charges (q, or q_c with a split).
    qv: &'a [f64],
    /// `M_g0`'s charges (`qv`, or every aux charge under a mutant).
    q_g0: &'a [f64],
    /// The build's G = 0 overlaps (`S(k)`, or `S(k) − S_ss(k)`).
    s_g0: &'a [Array2<C64>],
    phk: &'a [Vec<C64>],
    d_total: &'a [Array2<C64>],
    exch: &'a [(f64, &'a [Array2<C64>])],
    mutation: KFitMutation,
    gcut: f64,
    c0: f64,
}

/// What the per-q passes accumulate.
struct FitAcc {
    zbin: Vec<Array2<f64>>,
    wmbin: Vec<Array2<f64>>,
    lr: KLrForce,
    mg0: Vec<Array2<C64>>,
    n_dropped_max: usize,
    b_norm_mismatch: f64,
    n_chunks: usize,
    n_k_lr: usize,
}

/// `e^{+iq·t_r}` for aux-image residue `r` (`q = Σ m_i/N_i b_i`).
fn q_phase(mod_t: [usize; 3], mq: [i64; 3], r: usize) -> C64 {
    let dq: i64 = mod_t.iter().product::<usize>() as i64;
    let c = residue_coords(r, mod_t);
    let mut t = 0i64;
    for i in 0..3 {
        t += mq[i] * c[i] * (dq / mod_t[i] as i64);
    }
    unit_root(t, dq)
}

/// `J2(q)` and the per-residue `J3` accumulators exactly as the build forms
/// them, and `J3(k'−q, k')` for every `k'` (G = 0 and Hermitisation at
/// q = 0). Returns `(J2 Hermitised, J3 per k', chunks)`.
fn rebuild_q(
    ctx: &FitCtx<'_>,
    iq: usize,
    php: &[C64],
    ledger: &Ledger,
) -> Result<(Array2<C64>, Vec<Array2<C64>>, usize), FerricError> {
    let (st, mesh, g) = (ctx.st, ctx.mesh, ctx.g);
    let n = st.obs.nbasis();
    let (n2, naux) = (n * n, st.aux.nbasis());
    let (rl, rt) = (ctx.phk[0].len(), php.len());
    let (q, mq) = mesh.q_class(iq);
    let trim = mesh.q_minus(iq) == iq;
    let is_q0 = mq == [0, 0, 0];
    let mut j2 = Array2::<C64>::zeros((naux, naux));
    for (bin, ph) in ctx.j2res.iter().zip(php) {
        j2.zip_mut_with(bin, |z, &x| *z += *ph * x);
    }
    let mut acc: Vec<Array2<C64>> = (0..rl)
        .map(|r_l| {
            let mut a = Array2::<C64>::zeros((n2, naux));
            for (r_t, ph) in php.iter().enumerate() {
                let ph = ph.conj();
                a.zip_mut_with(&ctx.j3res[r_l * rt + r_t], |z, &x| *z += ph * x);
            }
            a
        })
        .collect();
    let (kv, half) = lr_kvectors(st.cell, mesh, iq, trim, ctx.gcut)?;
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let n_chunks = lr_accumulate_q_any(
        st,
        ctx.plan,
        (&kv, half, q),
        mesh.residue_moduli(),
        &mut j2,
        &mut acc,
        chunk_budget,
        false,
    )?;
    drop(kv);
    if is_q0 {
        let mut re = j2.mapv(|z| z.re);
        let mut no_j3 = Array2::<f64>::zeros((0, naux));
        subtract_g0(&mut re, &mut no_j3, &[], ctx.qv, ctx.c0, g.g0);
        j2.zip_mut_with(&re, |z, &x| z.re = x);
    }
    let j2h = (&j2 + &herm(&j2)).mapv(|z| z * 0.5);
    let mut j3: Vec<Array2<C64>> = Vec::with_capacity(mesh.nk());
    for (j, ph_j) in ctx.phk.iter().enumerate() {
        let mut x = Array2::<C64>::zeros((n2, naux));
        for (a, ph) in acc.iter().zip(ph_j) {
            x.zip_mut_with(a, |z, &y| *z += *ph * y);
        }
        if is_q0 {
            subtract_g0_three_index(&mut x, &ctx.s_g0[j], ctx.qv, ctx.c0, g.g0);
            hermitize_pairs(&mut x, n);
        }
        j3.push(x);
    }
    Ok((j2h, j3, n_chunks))
}

/// The largest relative `‖B‖_F` mismatch of the rebuilt fit against the
/// given build over the q class (rotation-invariant norm).
fn b_mismatch(
    gdf: &KRsGdf,
    j3: &[Array2<C64>],
    (u, s, keep): (&Array2<C64>, &[f64], &[bool]),
    kof: &[usize],
) -> f64 {
    let naux = u.nrows();
    let nkeep = keep.iter().filter(|&&b| b).count();
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
    let mut worst = 0.0_f64;
    for (j, x) in j3.iter().enumerate() {
        let mine = x.dot(&w).iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let given = gdf
            .block(kof[j], j)
            .iter()
            .map(|z| z.norm_sqr())
            .sum::<f64>()
            .sqrt();
        worst = worst.max((mine - given).abs() / given.max(1e-300));
    }
    worst
}

/// `H(q)` (exchange part) and `Z^{k'}` for every `k'` (module doc).
fn exchange_h_and_z(
    ctx: &FitCtx<'_>,
    j3: &[Array2<C64>],
    kof: &[usize],
    jinv: &Array2<C64>,
) -> (Array2<C64>, Vec<Array2<C64>>) {
    let n = ctx.st.obs.nbasis();
    let naux = ctx.st.aux.nbasis();
    let nk = ctx.mesh.nk();
    let inv_nk = 1.0 / nk as f64;
    let inv_nk2 = inv_nk * inv_nk;
    let mut hq = Array2::<C64>::zeros((naux, naux));
    let mut zt: Vec<Array2<C64>> = Vec::with_capacity(nk);
    for j in 0..nk {
        let k = kof[j];
        let mut tt = Array2::<C64>::zeros((n * n, naux));
        for p in 0..naux {
            let jqh = herm(&column_matrix(&j3[j], p, n));
            let mut t = Array2::<C64>::zeros((n, n));
            for (cf, ds) in ctx.exch {
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
        let mut z = tt.dot(jinv);
        z.mapv_inplace(|v| v * (-inv_nk2));
        zt.push(z);
    }
    (hq, zt)
}

/// The q = 0 Coulomb parts of `H` and `Z`, the Hermitisation of `Z`, and
/// `M_g0(k) = −c0 Σ_P q_P Z^k_P` (unless the `no_g0` mutant).
fn coulomb_q0(
    ctx: &FitCtx<'_>,
    j3: &[Array2<C64>],
    jinv: &Array2<C64>,
    hq: &mut Array2<C64>,
    zt: &mut [Array2<C64>],
    mg0: &mut [Array2<C64>],
) {
    let n = ctx.st.obs.nbasis();
    let naux = ctx.st.aux.nbasis();
    let inv_nk = 1.0 / ctx.mesh.nk() as f64;
    let zero = C64::new(0.0, 0.0);
    let mut rho = Array1::<C64>::zeros(naux);
    for (j, x) in j3.iter().enumerate() {
        let d = &ctx.d_total[j];
        let v = Array1::from_shape_fn(n * n, |mn| d[(mn % n, mn / n)]);
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
        let d = &ctx.d_total[j];
        for m in 0..n {
            for l in 0..n {
                let dv = d[(l, m)] * inv_nk;
                for p in 0..naux {
                    z[(m * n + l, p)] += cvec[p].conj() * dv;
                }
            }
        }
        hermitize_pairs(z, n);
        if !ctx.mutation.no_g0 {
            let mg = &mut mg0[j];
            for m in 0..n {
                for l in 0..n {
                    let mut a = zero;
                    for (p, qp) in ctx.q_g0.iter().enumerate() {
                        a += z[(m * n + l, p)] * *qp;
                    }
                    mg[(l, m)] += a * (-ctx.c0);
                }
            }
        }
    }
}

/// One q class of [`kpoint_fit_gradient`]: rebuild, check against the given
/// build, `H`/`Z`/`Wm`, the SR bin folds and the LR derivative on the full
/// `{G + q}` set.
fn q_class_pass(
    ctx: &FitCtx<'_>,
    iq: usize,
    ledger: &Ledger,
    acc: &mut FitAcc,
) -> Result<(), FerricError> {
    let who = "k-point RS-GDF forces";
    let (st, mesh, g) = (ctx.st, ctx.mesh, ctx.g);
    let naux = st.aux.nbasis();
    let n2 = st.obs.nbasis() * st.obs.nbasis();
    let nk = mesh.nk();
    let mod_t = mesh.n();
    let (q, mq) = mesh.q_class(iq);
    let is_q0 = mq == [0, 0, 0];
    let php: Vec<C64> = (0..php_len(mod_t)).map(|r| q_phase(mod_t, mq, r)).collect();

    // --- J2(q), J3(k'−q, k'): exactly the build's.
    let (j2h, j3, ch) = rebuild_q(ctx, iq, &php, ledger)?;
    acc.n_chunks += ch;
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
    acc.n_dropped_max = acc.n_dropped_max.max(naux - nkeep);
    let kept_build = ctx.gdf.stats().per_q[iq].naux_kept;
    if kept_build != nkeep {
        return Err(FerricError::General(format!(
            "{who}: q class {iq} keeps {nkeep} metric eigenvectors here but {kept_build} in the \
             given KRsGdf; pass the config the build used"
        )));
    }
    let kof: Vec<usize> = (0..nk).map(|j| mesh.k_minus_q(j, mq)).collect();

    // --- B check against the given build.
    let mism = b_mismatch(ctx.gdf, &j3, (&u, &s, &keep), &kof);
    acc.b_norm_mismatch = acc.b_norm_mismatch.max(mism);
    if acc.b_norm_mismatch > 1e-8 {
        return Err(FerricError::General(format!(
            "{who}: the rebuilt fit differs from the given KRsGdf (relative ‖B‖ mismatch \
             {:.2e} at q class {iq}); pass the config and S(k) the build used",
            acc.b_norm_mismatch
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
    let (mut hq, mut zt) = exchange_h_and_z(ctx, &j3, &kof, &jinv);
    if is_q0 {
        coulomb_q0(ctx, &j3, &jinv, &mut hq, &mut zt, &mut acc.mg0);
    }
    drop(j3);

    // --- Metric weight Wm = U (Lo ∘ U^H H U) U^H, Hermitised (module doc).
    let wm = if ctx.mutation.no_metric {
        Array2::<C64>::zeros((naux, naux))
    } else {
        let a = herm(&u).dot(&hq).dot(&u);
        let b = Array2::from_shape_fn((naux, naux), |(i, j)| a[(i, j)] * lo[(i, j)]);
        let w = u.dot(&b).dot(&herm(&u));
        (&w + &herm(&w)).mapv(|z| z * 0.5)
    };
    drop(hq);

    // --- Residue folds: Zr[r_L] = Σ_k' e^{ik'·t_rL} Z^{k'}; SR bins.
    let rl = ctx.phk[0].len();
    let rt = php.len();
    let mut zr: Vec<Array2<C64>> = (0..rl).map(|_| Array2::zeros((n2, naux))).collect();
    for (j, z) in zt.iter().enumerate() {
        for (r_l, a) in zr.iter_mut().enumerate() {
            let ph = ctx.phk[j][r_l];
            a.zip_mut_with(z, |x, &y| *x += ph * y);
        }
    }
    drop(zt);
    for (r_l, a) in zr.iter().enumerate() {
        for (r_t, pt) in php.iter().enumerate() {
            let ph = if ctx.mutation.aux_phase {
                *pt
            } else {
                pt.conj()
            };
            acc.zbin[r_l * rt + r_t].zip_mut_with(a, |x, &y| *x += (ph * y).re);
        }
    }
    for (r_t, pt) in php.iter().enumerate() {
        let bin = &mut acc.wmbin[r_t];
        for p in 0..naux {
            for qq in 0..naux {
                bin[(p, qq)] += (*pt * wm[(qq, p)]).re;
            }
        }
    }

    // --- LR on the FULL {G + q} set.
    let qn = dot3(&q, &q).sqrt();
    let g2cut = ctx.gcut * ctx.gcut;
    let kfull: Vec<[f64; 3]> = st
        .cell
        .gvectors(ctx.gcut + qn)?
        .into_iter()
        .map(|gv| [gv[0] + q[0], gv[1] + q[1], gv[2] + q[2]])
        .filter(|k| {
            let k2 = dot3(k, k);
            k2 > 0.0 && k2 <= g2cut
        })
        .collect();
    acc.n_k_lr += kfull.len();
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    acc.n_chunks += k_lr_force_q(ctx, &kfull, q, &zr, &wm, chunk_budget, &mut acc.lr)?;
    Ok(())
}

/// `R_T` of the aux-image residues.
fn php_len(mod_t: [usize; 3]) -> usize {
    mod_t[0] * mod_t[1] * mod_t[2]
}

/// The LR force passes of one q class: the unsplit pass when no aux
/// primitive moved, else the split full-pair pass; plus the split's
/// smooth-pair pass. Returns the chunk count.
fn k_lr_force_q(
    ctx: &FitCtx<'_>,
    kfull: &[[f64; 3]],
    q: [f64; 3],
    zr: &[Array2<C64>],
    wm: &Array2<C64>,
    chunk_budget: usize,
    out: &mut KLrForce,
) -> Result<usize, FerricError> {
    let (st, m) = (ctx.st, ctx.mutation);
    let moduli = ctx.mesh.residue_moduli();
    let gk = m.split_gamma_kernel;
    let mut n_chunks = match ctx.plan.filter(|p| p.moves_aux()) {
        Some(p) => p.k_lr_force_moved(
            st,
            kfull,
            q,
            moduli,
            zr,
            wm,
            chunk_budget,
            (gk, m.no_metric),
            out,
        )?,
        None => k_lr_force_unsplit(st, kfull, moduli, zr, wm, chunk_budget, m.no_metric, out)?,
    };
    if let Some(p) = ctx.plan {
        n_chunks += p.k_lr_force_smooth(st, kfull, q, moduli, zr, chunk_budget, gk, out)?;
    }
    Ok(n_chunks)
}

/// The unsplit LR force pass of one q class (Iteration 21b): orbital
/// centres through the residue pair-FT derivative with weight
/// `conj(X) v_LR`, aux centres `+iK conj(X)`, metric `(X, X)`.
#[allow(clippy::too_many_arguments)]
fn k_lr_force_unsplit(
    st: &Stage<'_>,
    kfull: &[[f64; 3]],
    mod_l: [usize; 3],
    zr: &[Array2<C64>],
    wm: &Array2<C64>,
    chunk_budget: usize,
    no_metric: bool,
    out: &mut KLrForce,
) -> Result<usize, FerricError> {
    let (cell, obs) = (st.cell, st.obs);
    let n = obs.nbasis();
    let n2 = n * n;
    let naux = st.aux.nbasis();
    let vol = cell.volume();
    let omega = st.omega;
    let pair_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
    let zero = C64::new(0.0, 0.0);
    let wmt = wm.t().to_owned();
    let aux_sh = &st.aux_sh;
    let (orb_ao, aux_lr, metric_lr) = (&mut out.orb_ao, &mut out.aux, &mut out.metric);
    // Per K: X, conj(X) v, the n² weight vector per residue, aZ.
    let extra_per_g = naux
        .saturating_mul(16 * 6)
        .saturating_add(n2.saturating_mul(16 * 2));
    pair_ft_deriv_residues_chunked(
        cell,
        obs,
        kfull,
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
                        if !no_metric {
                            metric_lr[(pp, xx)] += (xc * ik * wx[pp] * vlr).re;
                            metric_lr[(pp, xx)] += (xg[pp] * (-ik) * wcx[pp] * vlr).re;
                        }
                    }
                }
            }
            Ok(())
        },
    )
}

/// The SR derivative walks: the split's kept calls / compact pairs with a
/// plan (ordered-parallel only), else the unsplit ordered-parallel walks or
/// (`serial_sr`) their frozen serial oracles.
#[allow(clippy::type_complexity)]
fn k_sr_force_parts(
    st: &Stage<'_>,
    plan: Option<&SplitPlan>,
    images: &[[f64; 3]],
    bins: &KSrBins<'_>,
    natoms: usize,
    budget: usize,
    mutation: KFitMutation,
) -> Result<((Array2<f64>, Array2<f64>, usize), (Array2<f64>, usize)), FerricError> {
    let naux = st.aux.nbasis();
    let z = |l: [f64; 3], t: [f64; 3]| Y3::PairMajor(bins.z(l, t));
    let sr3 = match (plan, mutation.serial_sr) {
        (Some(p), _) => p.sr3_force_with(st, images, natoms, budget, z)?,
        (None, true) => k_sr3_force_serial(st, images, bins, natoms)?,
        (None, false) => sr3_force(st, images, natoms, budget, z)?,
    };
    let sr2 = match (plan, mutation.no_metric, mutation.serial_sr) {
        (_, true, _) => (Array2::<f64>::zeros((naux, 3)), 0),
        (Some(p), false, _) => p.sr2_force_with(st, budget, |t| bins.wm(t))?,
        (None, false, true) => k_sr2_force_serial(st, bins)?,
        (None, false, false) => sr2_force(st, budget, |t| bins.wm(t))?,
    };
    Ok((sr3, sr2))
}

/// The phase-folded SR derivative weights of [`kpoint_fit_gradient`]:
/// `zbin[r_L R_T + r_T]` `(n², naux)` and `wmbin[r_T]` `(naux, naux)`,
/// binned by the residues of `L` (mod `mod_l`) and `T` (mod `mod_t`).
struct KSrBins<'a> {
    recip: [[f64; 3]; 3],
    mod_l: [usize; 3],
    mod_t: [usize; 3],
    zbin: &'a [Array2<f64>],
    wmbin: &'a [Array2<f64>],
}

impl<'a> KSrBins<'a> {
    fn rt(&self) -> usize {
        self.mod_t.iter().product()
    }

    /// The `Z` bin of pair image `L`, aux image `T`.
    fn z(&self, l: [f64; 3], t: [f64; 3]) -> &'a Array2<f64> {
        &self.zbin[residue_index(lattice_coords(&self.recip, &l), self.mod_l) * self.rt()
            + residue_index(lattice_coords(&self.recip, &t), self.mod_t)]
    }

    /// The `Wm` bin of aux image `T`.
    fn wm(&self, t: [f64; 3]) -> &'a Array2<f64> {
        &self.wmbin[residue_index(lattice_coords(&self.recip, &t), self.mod_t)]
    }
}

/// FROZEN pre-parallel serial k-point SR 3-centre force walk (the
/// bit-identity oracle of [`sr3_force`]; do not "improve").
fn k_sr3_force_serial(
    st: &Stage<'_>,
    images: &[[f64; 3]],
    kb: &KSrBins<'_>,
    natoms: usize,
) -> Result<(Array2<f64>, Array2<f64>, usize), FerricError> {
    let (obs, aux) = (st.obs, st.aux);
    let (n, naux) = (obs.nbasis(), aux.nbasis());
    let (b, mod_l, mod_t, rt, zbin) = (kb.recip, kb.mod_l, kb.mod_t, kb.rt(), kb.zbin);
    let sh2at = obs.shell_to_atom().to_vec();
    let mut orb_sr = Array2::<f64>::zeros((natoms, 3));
    let mut aux_sr = Array2::<f64>::zeros((naux, 3));
    let mut eng3 = Engine::new_3center_deriv(Operator::erfc(st.omega), obs, aux, ENGINE_PRECISION)?;
    let n_sr3 = st.sr_three_index_walk(images, |i1, i2, ip, l, t| {
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
    Ok((orb_sr, aux_sr, n_sr3))
}

/// FROZEN pre-parallel serial k-point SR metric force walk (the
/// bit-identity oracle of [`sr2_force`]; do not "improve").
fn k_sr2_force_serial(
    st: &Stage<'_>,
    kb: &KSrBins<'_>,
) -> Result<(Array2<f64>, usize), FerricError> {
    let aux = st.aux;
    let (b, mod_t, wmbin) = (kb.recip, kb.mod_t, kb.wmbin);
    let mut metric_sr = Array2::<f64>::zeros((aux.nbasis(), 3));
    let mut eng2 = Engine::new_2center_deriv(Operator::erfc(st.omega), aux, ENGINE_PRECISION)?;
    let n_sr2 = st.sr_metric_walk(|ip, iq, t| {
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
    Ok((metric_sr, n_sr2))
}
