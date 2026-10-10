//! Derivative pieces of the Gamma RS-GDF two-electron energy for the analytic
//! forces ([`crate::grad`]'s `*_rsgdf` entry points) — the Rust port of
//! `reference/pbc/pbc_grad_gdf.py` (FINDINGS "Iteration 18").
//!
//! # What is differentiated
//!
//! ```text
//! E_2e = Σ Γ_μνλσ I^fit_μνλσ,   I^fit = J3 f(J2) J3ᵀ,
//! Γ = ½ D_μν D_λσ − (α/2) Σ_σ D^σ_μλ D^σ_νσ          (RHF: ½ DD − ¼ DD)
//! f(J2) = U diag(f(s)) Uᵀ,  f(s) = 1/s for s > lindep, 0 otherwise
//! ```
//!
//! with the primed (G = 0-dropped) `J2`, `J3` of [`super`] — exactly the
//! pseudo-inverse the build's eigenvalue cut applies (B = s^{-1/2} Uᵀ J3ᵀ
//! over the kept eigenvalues, so `BᵀB = J3 f(J2) J3ᵀ`). At fixed densities:
//!
//! ```text
//! dE_2e = Σ_{μν,P} Y[P,μν] dJ3[μν,P] + Σ_{PQ} Wm[P,Q] dJ2[P,Q]
//! Y_P  = D c_P − α Σ_σ D_σ C_P D_σ,     C = f(J2) J3ᵀ = W B,  c_P = C_P · D
//! Wm   = U (Lo ∘ Uᵀ H U) Uᵀ,            H = J3ᵀ Γ J3
//! Lo   = Loewner (Daleckii–Krein) matrix of f:
//!          kept–kept     −1/(s_i s_j)            → −W G_BB Wᵀ,  G_BB = Γ(B_k, B_l)
//!          kept–dropped  (1/s_k)/(s_k − s_d)     → W X' U_dᵀ + transpose,
//!                                                  X'[k,d] = Γ(B_k, J_d)/(s_k − s_d)
//!          dropped–dropped 0
//! ```
//!
//! (`W = U_kept s^{-1/2}` = [`RsGdf::metric_inv_sqrt`], `J_d = U_dᵀ J3ᵀ`
//! retained by [`RsGdf::build_for_gradient`].) Both Γ contractions come from
//! one tensor `Y^B_k = D (B_k·D) − α Σ_σ D_σ B_k D_σ`: `Γ(B_k, X) = ½ Y^B_k·X`,
//! and `Y = W Y^B` (Y is linear in C). The kept–kept block alone is the
//! textbook DF metric term `−½ cᵀ (P|Q)' c`; it differentiates a DIFFERENT
//! function whenever a dropped direction carries energy (the retained
//! eigenvectors rotate with R). Prototype: 4e-8..1.5e-7 Ha/Bohr on toys with
//! an active cut, 0 when nothing is dropped.
//!
//! # Derivative integrals (streamed, contracted immediately, never stored)
//!
//! * SR `J3`: `Σ_{L,T} (μ_0 ν_L | P_T)_erfc` over the energy's pair images and
//!   per-triplet screen ([`Stage::sr_three_index_walk`]); each triplet's
//!   `[d/dP, d/d(sh1), d/d(sh2)]` blocks ([`Engine::compute_eri3_deriv_shifted`])
//!   are contracted with the matching `Y` block: the two orbital blocks onto
//!   their atoms (all images move), the aux block onto the aux function.
//! * SR `J2`: `Σ_T (P_0 | Q_T)_erfc` over [`Stage::sr_metric_walk`], the
//!   `d/dP` block of [`Engine::compute_eri2_deriv_shifted`] with
//!   `d/dQ = −d/dP` (two-centre translation invariance).
//! * LR `J3` (half G sphere, weight `w = (2/Ω) 4π/G² e^{−G²/4ω²}`):
//!   orbital `2 w Σ_{μ∈A,ν} Re[Q*_μν (Σ_P Y_Pμν X_P)]` (pair-FT bra derivative
//!   [`pair_ft_deriv_chunked`], ×2 for the ket by `Y` symmetry and
//!   `P_μν = P_νμ` at reciprocal G); aux `w Re[(Σ_μν Y_Pμν P_μν)* (−iG) X_P]`
//!   (an aux function translates rigidly: `dX_P/dC = −iG X_P`, no raised-l
//!   term).
//! * LR `J2`: `2 w Re[(−iG X_R)* (Wm X)_R]` per aux function R.
//! * G = 0: `J2`'s `c0 q qᵀ` is position-free; `J3`'s `−c0 S_μν q_P` enters
//!   ONLY through `dS` as `M_g0 = −c0 Σ_P Y_P q_P` (assembled in
//!   [`crate::grad`], contracted with the lattice overlap derivative).
//!
//! Everything uses the build's ω, precision, SR screen mode, pair images and
//! G sphere, so the force is the derivative of the truncated energy up to the
//! screens (`≤ precision`).
//!
//! # Parallelism of the LR pass
//!
//! A G chunk's work (the pair-FT derivative producer, the `n² × naux × ng`
//! GEMMs, the per-G contractions) is a pure function of the chunk, while the
//! accumulation into the shared force rows (`natoms × 3` orbital, `naux × 3`
//! aux-centre and metric) is order-dependent. The chunks are therefore
//! evaluated in parallel, up to [`MAX_LR_INFLIGHT`] at a time, into small
//! per-chunk values ([`LrChunk`]: the exact scalars the serial pass added) and
//! replayed into the rows serially in ascending chunk order
//! ([`crate::ordered`]), so the forces are BIT-IDENTICAL to the serial pass at
//! any thread count. The chunk width is that of the serial pass (a function of
//! the memory budget and the basis only) and the in-flight count is a function
//! of the ledger only, never of the thread count.
//!
//! # Memory
//!
//! Each chunk in flight holds up to the chunk budget (`min(remaining,
//! G_CHUNK_BYTES)`); [`lr_inflight`] charges `in-flight × chunk budget` to the
//! ledger and narrows the in-flight count (down to the serial pass's one chunk)
//! rather than exceed it.
//!
//! # Memory (resident fit densities)
//!
//! `Y` (`naux × nao²`) and the transient `Y^B` (`naux_kept × nao²`) are
//! reserved on the caller's ledger, with the `naux²` metric weights; the LR
//! chunk's `P`, three `Q`, `X` and the per-G GEMM outputs are bounded by the
//! chunk budget.

use super::split::{LrAcc, LrChunk, LrForce, SplitPlan};
use super::{
    aux_ft_shells, check_obs_on_cell, gshells, pair_image_radius, require_pure_aux, LatticeWalker,
    RsGdf, Stage, ENGINE_PRECISION,
};
use crate::budget::{bytes_of, Ledger};
use crate::grad::ao_atoms;
use crate::hcore::{gvector_list_bytes, half_gvectors, G_CHUNK_BYTES};
use crate::lattice::Cell;
use crate::ordered::{window_budget, Stored};
use crate::pair_ft::{pair_ft_deriv_chunked_ordered, DEFAULT_PAIR_FT_THRESH};
use crate::sr_rotation::{AoTranspose, ColumnRotationMutant, DerivRotMutant, RotatedBasis};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2, ArrayView1};
use std::f64::consts::PI;

/// Metric-solve and derivative-walk diagnostics of an RS-GDF force
/// (reported with the gradient; FINDINGS "Iteration 18": a force is only as
/// good as the energy's own noise floor when the smallest KEPT eigenvalue is
/// within ~2 decades of the metric noise, ~1e-10 at precision 1e-13).
#[derive(Debug, Clone)]
pub struct RsGdfFitDiagnostics {
    /// Aux functions.
    pub naux: usize,
    /// Metric eigenvalues dropped by the `lindep` cut.
    pub n_dropped: usize,
    /// The absolute cut used.
    pub lindep: f64,
    /// Smallest kept / largest dropped metric eigenvalue.
    pub s_kept_min: f64,
    pub s_dropped_max: Option<f64>,
    /// `max |Wm_Loewner − Wm_textbook|`: the size of the kept–dropped block
    /// (0 when nothing is dropped). Reported even when the textbook mutant
    /// is requested.
    pub cross_norm: f64,
    /// Whether the kept–dropped block was OMITTED (test mutant only).
    pub textbook_metric: bool,
    /// Shifted 3-centre / 2-centre derivative calls (SR).
    pub n_sr3_deriv: usize,
    pub n_sr2_deriv: usize,
    /// Half-sphere G vectors and `pair_ft_deriv` chunks (LR).
    pub n_g_half: usize,
    pub n_g_chunks: usize,
}

/// The fitted two-electron densities of the module doc.
pub(crate) struct FitDensities {
    /// `(naux, nao²)`, row P, column `μ·nao+ν` (symmetric in μν).
    pub(crate) y: Array2<f64>,
    /// `(naux, naux)`, symmetric.
    pub(crate) wm: Array2<f64>,
    pub(crate) diag: RsGdfFitDiagnostics,
}

/// `Y`, `Wm` for total density `d` and exchange terms
/// `Σ_σ D_σ X D_σ = Σ_(c, D) c·D X D` (restricted: `[(½, D)]`; unrestricted:
/// `[(1, D_α), (1, D_β)]`), exact-exchange fraction `alpha`. `textbook`
/// omits the kept–dropped block (mutation tests only).
pub(crate) fn fit_densities(
    gdf: &RsGdf,
    d: &Array2<f64>,
    exch: &[(f64, &Array2<f64>)],
    alpha: f64,
    textbook: bool,
    ledger: &mut Ledger,
) -> Result<FitDensities, FerricError> {
    let gp = gdf.gradient_parts().ok_or_else(|| {
        FerricError::General(
            "RS-GDF forces: this RsGdf was built without gradient parts; \
             use RsGdf::build_for_gradient"
                .into(),
        )
    })?;
    let n = gdf.nao;
    let n2 = n * n;
    let b = gdf.b();
    let w = gdf.metric_inv_sqrt();
    let nkeep = b.nrows();
    let naux = w.nrows();
    let nd = gp.s_drop.len();
    if b.ncols() != n2
        || w.ncols() != nkeep
        || gp.s_kept.len() != nkeep
        || gp.u_drop.dim() != (naux, nd)
        || gp.jd.dim() != (nd, n2)
        || d.dim() != (n, n)
        || exch.iter().any(|(_, x)| x.dim() != (n, n))
    {
        return Err(FerricError::General(format!(
            "RS-GDF forces: inconsistent shapes (nao {n}, naux {naux}, kept {nkeep}, dropped {nd}, \
             B {:?}, W {:?}, D {:?})",
            b.dim(),
            w.dim(),
            d.dim()
        )));
    }
    ledger.reserve(
        &format!(
            "RS-GDF force 3-index densities Y + Y^B (naux = {naux}, kept = {nkeep}, nao = {n})"
        ),
        bytes_of((n2 as u64).saturating_mul((naux + nkeep) as u64), 8),
    )?;
    ledger.reserve(
        &format!("RS-GDF force metric weights + Γ blocks (naux = {naux})"),
        bytes_of(
            ((naux * naux) as u64)
                .saturating_mul(3)
                .saturating_add((nkeep * (nkeep + nd)) as u64),
            8,
        ),
    )?;

    // Y^B_k = D (B_k·D) − α Σ c D_σ B_k D_σ.
    let d_std = d.as_standard_layout();
    let dflat = ArrayView1::from(d_std.as_slice().expect("standard layout"));
    let cb: Array1<f64> = b.dot(&dflat);
    let mut yk = Array2::<f64>::zeros((nkeep, n2));
    for k in 0..nkeep {
        let bk = b
            .row(k)
            .into_shape_with_order((n, n))
            .map_err(|e| FerricError::General(format!("RS-GDF forces: B row reshape: {e}")))?;
        let mut t = Array2::<f64>::zeros((n, n));
        t.scaled_add(cb[k], d);
        if alpha != 0.0 {
            for &(c, ds) in exch {
                let s = ds.dot(&bk).dot(ds);
                t.scaled_add(-alpha * c, &s);
            }
        }
        let mut row = yk.row_mut(k);
        for m in 0..n {
            for l in 0..n {
                row[m * n + l] = t[(m, l)];
            }
        }
    }

    // Kept–kept: Wm = −W G_BB Wᵀ, G_BB = ½ Y^B Bᵀ.
    let mut g_bb = yk.dot(&b.t());
    g_bb *= 0.5;
    let g_bb = 0.5 * (&g_bb + &g_bb.t());
    let mut wm = -(w.dot(&g_bb).dot(&w.t()));

    // Kept–dropped (Loewner): W X' U_dᵀ + transpose.
    let mut cross_norm = 0.0_f64;
    if nd > 0 {
        let mut xp = yk.dot(&gp.jd.t()); // (nkeep, nd) = 2 Γ(B_k, J_d)
        for k in 0..nkeep {
            for dd in 0..nd {
                let gap = gp.s_kept[k] - gp.s_drop[dd];
                if !(gap > 0.0) {
                    return Err(FerricError::General(format!(
                        "RS-GDF forces: kept eigenvalue {} not above dropped {} (lindep {})",
                        gp.s_kept[k], gp.s_drop[dd], gp.lindep
                    )));
                }
                xp[(k, dd)] *= 0.5 / gap;
            }
        }
        let half = w.dot(&xp).dot(&gp.u_drop.t()); // (naux, naux)
        let cross = &half + &half.t();
        cross_norm = cross.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if !textbook {
            wm += &cross;
        }
    }
    let wm = 0.5 * (&wm + &wm.t());

    // Y = W Y^B (raw aux basis).
    let y = w.dot(&yk);
    drop(yk);
    if y.iter().chain(wm.iter()).any(|v| !v.is_finite()) {
        return Err(FerricError::General(
            "RS-GDF forces: non-finite fitted density (metric too ill-conditioned?)".into(),
        ));
    }
    let s_kept_min = gp.s_kept.iter().copied().fold(f64::INFINITY, f64::min);
    let s_dropped_max = gp.s_drop.iter().copied().reduce(f64::max);
    Ok(FitDensities {
        y,
        wm,
        diag: RsGdfFitDiagnostics {
            naux,
            n_dropped: nd,
            lindep: gp.lindep,
            s_kept_min,
            s_dropped_max,
            cross_norm,
            textbook_metric: textbook,
            n_sr3_deriv: 0,
            n_sr2_deriv: 0,
            n_g_half: 0,
            n_g_chunks: 0,
        },
    })
}

/// The six derivative contractions of the module doc. Orbital pieces are per
/// cell atom (`natoms × 3`); aux pieces per aux FUNCTION (`naux × 3`,
/// `dE/dC_P`), folded onto atoms by [`fold_aux`].
pub(crate) struct FitDerivatives {
    pub(crate) orb_sr: Array2<f64>,
    pub(crate) orb_lr: Array2<f64>,
    pub(crate) aux3_sr: Array2<f64>,
    pub(crate) aux3_lr: Array2<f64>,
    pub(crate) metric_sr: Array2<f64>,
    pub(crate) metric_lr: Array2<f64>,
    pub(crate) n_sr3: usize,
    pub(crate) n_sr2: usize,
    pub(crate) n_g_half: usize,
    pub(crate) n_chunks: usize,
}

/// Contract `Y` with `dJ3` and `Wm` with `dJ2` (module doc). `obs` must be
/// the orbital basis B was built with (on `cell`'s atoms), `aux` its aux
/// basis. The SR walks are ordered-parallel and BIT-IDENTICAL to the serial
/// walks ([`sr3_force`], [`sr2_force`]); `serial` runs the frozen serial
/// oracles instead (unsplit builds only). A range-split build walks the
/// split partition (`split`'s `deriv`, FINDINGS "Iteration 26");
/// `drop_smooth_pair` (TEST mutant) omits its smooth-pair LR pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_derivatives(
    gdf: &RsGdf,
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    y: &Array2<f64>,
    wm: &Array2<f64>,
    serial: bool,
    drop_smooth_pair: bool,
    rot_mutant: DerivRotMutant,
    ledger: &mut Ledger,
) -> Result<FitDerivatives, FerricError> {
    fit_derivatives_with(
        gdf,
        cell,
        obs,
        aux,
        (y, wm),
        (serial, drop_smooth_pair, rot_mutant),
        MAX_LR_INFLIGHT,
        ledger,
    )
}

/// Most LR G chunks in flight at once ([`lr_inflight`]). A constant, never a
/// function of the thread count (the in-flight count only bounds memory and
/// never changes a bit; 12 keeps a 6-core box busy across window barriers).
pub(crate) const MAX_LR_INFLIGHT: usize = 12;

/// How many LR chunks may be in flight: `min(cap, remaining / chunk_budget)`,
/// at least 1 (one chunk is what the serial pass already held). The in-flight
/// chunks are CHARGED to a scratch copy of the ledger (they are freed when
/// the pass ends, so the caller's ledger is not consumed), so a tight
/// `[memory]` budget narrows the parallelism instead of overrunning
/// `budget_gb`.
fn lr_inflight(ledger: &Ledger, chunk_budget: usize, cap: usize) -> Result<usize, FerricError> {
    let n = (ledger.remaining() / chunk_budget.max(1)).min(cap).max(1);
    if n > 1 {
        ledger.clone().reserve(
            &format!("RS-GDF force LR chunks in flight ({n} x {chunk_budget} bytes)"),
            n.saturating_mul(chunk_budget),
        )?;
    }
    Ok(n)
}

/// [`fit_derivatives`] with an explicit cap on the LR chunks in flight.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_derivatives_with(
    gdf: &RsGdf,
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    (y, wm): (&Array2<f64>, &Array2<f64>),
    (serial, drop_smooth_pair, rot_mutant): (bool, bool, DerivRotMutant),
    inflight_cap: usize,
    ledger: &mut Ledger,
) -> Result<FitDerivatives, FerricError> {
    let gp = gdf
        .gradient_parts()
        .ok_or_else(|| FerricError::General("RS-GDF forces: RsGdf has no gradient parts".into()))?;
    check_obs_on_cell(cell, obs)?;
    require_pure_aux(aux, "RS-GDF forces")?;
    let stats = gdf.stats();
    let st = Stage {
        cell,
        obs,
        aux,
        obs_sh: gshells(obs, "RS-GDF forces orbital basis")?,
        aux_sh: gshells(aux, "RS-GDF forces aux basis")?,
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
            "RS-GDF forces: aux basis has {naux} functions / orbital {n}, but B was built with \
             naux = {} / nao = {} (Y {:?}, Wm {:?})",
            stats.naux,
            gdf.nao,
            y.dim(),
            wm.dim()
        )));
    }
    let natoms = cell.positions().len();

    // --- SR 3-centre: the energy's pair images and triplet screen.
    let rpair = pair_image_radius(&st, st.thresh);
    ledger.reserve(
        &format!("RS-GDF force pair-image list (r_pair = {rpair:.2} Bohr)"),
        bytes_of(cell.translation_count_bound(rpair)?, 24),
    )?;
    let images = cell.translations(rpair)?;
    // A range-split build: the same partition (`split`'s `deriv`).
    let plan = SplitPlan::for_derivatives(&st, gdf, &images, ledger)?;
    let budget = window_budget(ledger.remaining());
    // A build whose energy walked the column-rotated shells is differentiated
    // on the same rotated shells (module doc "Column rotation").
    let rotd = SrRotatedDeriv::new(gdf, cell, obs, rot_mutant)?;
    let sr = sr_force_stage(
        &st,
        plan.as_ref(),
        rotd.as_ref(),
        &images,
        y,
        wm,
        natoms,
        budget,
        serial,
    )?;

    // --- LR: the energy's half G sphere; P, Q, X per chunk.
    let gcut = 2.0 * st.omega * (1.0 / st.thresh).ln().sqrt();
    ledger.reserve(
        &format!("RS-GDF force LR G list (|G| <= {gcut:.3})"),
        gvector_list_bytes(cell, gcut)?,
    )?;
    let gv = half_gvectors(cell, gcut)?;
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let inflight = lr_inflight(ledger, chunk_budget, inflight_cap)?;
    let lr = lr_pass(
        plan.as_ref(),
        &st,
        &gv,
        (y, wm),
        natoms,
        (chunk_budget, inflight, drop_smooth_pair),
    )?;
    let (orb_sr, aux3_sr, metric_sr) = (sr.orb, sr.aux3, sr.metric);
    let (orb_lr, aux3_lr, metric_lr, n_chunks) = (lr.orb, lr.aux3, lr.metric, lr.n_chunks);
    let (n_sr3, n_sr2) = (sr.n_sr3, sr.n_sr2);

    let all = [&orb_sr, &orb_lr, &aux3_sr, &aux3_lr, &metric_sr, &metric_lr];
    if all.iter().any(|a| a.iter().any(|v| !v.is_finite())) {
        return Err(FerricError::General(
            "RS-GDF forces: non-finite derivative contraction".into(),
        ));
    }
    Ok(FitDerivatives {
        orb_sr,
        orb_lr,
        aux3_sr,
        aux3_lr,
        metric_sr,
        metric_lr,
        n_sr3,
        n_sr2,
        n_g_half: gv.len(),
        n_chunks,
    })
}

/// TEST switch: run the frozen serial LR oracle instead of the parallel pass.
#[cfg(test)]
pub(crate) static LR_SERIAL_ORACLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The LR force pass (moved-aux split pass or unsplit, plus the smooth-pair
/// pass): chunk-parallel, or the frozen serial oracle under test.
fn lr_pass(
    plan: Option<&SplitPlan>,
    st: &Stage<'_>,
    gv: &[[f64; 3]],
    (y, wm): (&Array2<f64>, &Array2<f64>),
    natoms: usize,
    (chunk_budget, inflight, drop_smooth_pair): (usize, usize, bool),
) -> Result<LrForce, FerricError> {
    let moved = plan.filter(|p| p.moves_aux());
    #[cfg(test)]
    if LR_SERIAL_ORACLE.load(std::sync::atomic::Ordering::Relaxed) {
        let mut lr = match moved {
            Some(p) => p.lr_force_serial_oracle(st, gv, y, wm, natoms, chunk_budget)?,
            None => lr_force_unsplit_serial_oracle(st, gv, y, wm, natoms, chunk_budget)?,
        };
        if let Some(p) = plan.filter(|_| !drop_smooth_pair) {
            p.lr_smooth_pair_force_serial_oracle(st, gv, y, chunk_budget, &mut lr)?;
        }
        return Ok(lr);
    }
    let mut lr = match moved {
        Some(p) => p.lr_force(st, gv, y, wm, natoms, chunk_budget, inflight)?,
        None => lr_force_unsplit(st, gv, (y, wm), natoms, (chunk_budget, inflight))?,
    };
    if let Some(p) = plan.filter(|_| !drop_smooth_pair) {
        p.lr_smooth_pair_force(st, gv, y, chunk_budget, inflight, &mut lr)?;
    }
    Ok(lr)
}

/// The column-rotated SR 3-centre stage of a derivative of a build that ran
/// its energy walk rotated (module doc "Column rotation"; shared by the
/// force and stress walks): the rotated orbital basis and the AO transpose
/// map of its weights.
pub(super) struct SrRotatedDeriv {
    pub(super) rot: RotatedBasis,
    tr: AoTranspose,
}

impl SrRotatedDeriv {
    /// `Ok(None)` when `gdf`'s energy walk was unrotated. A build made with a
    /// rotation MUTANT has a deliberately wrong energy and no derivative here.
    /// `mutant` is a TEST defect of the forward transform only.
    pub(super) fn new(
        gdf: &RsGdf,
        cell: &Cell,
        obs: &PreparedBasis,
        mutant: DerivRotMutant,
    ) -> Result<Option<Self>, FerricError> {
        let Some(rot) = gdf.sr_rotation() else {
            return Ok(None);
        };
        if rot.mutant != ColumnRotationMutant::Production {
            return Err(FerricError::General(format!(
                "RS-GDF derivatives: the RsGdf was built with the column-rotation mutant {:?}; \
                 forces/stress exist only for ColumnRotationMutant::Production",
                rot.mutant
            )));
        }
        let Some(rb) = RotatedBasis::detect(cell, obs, rot, "RS-GDF derivatives")? else {
            return Err(FerricError::General(
                "RS-GDF derivatives: the build rotated its SR walk but the orbital basis given \
                 to the derivative has nothing to rotate (another basis than the build's)"
                    .into(),
            ));
        };
        let tr = rb.ao_transpose(mutant);
        Ok(Some(Self { rot: rb, tr }))
    }

    /// `Y' = Tᵀ Y T` as a lazily evaluated weight on the rotated shells.
    pub(super) fn weights<'w>(&'w self, y: &'w Array2<f64>, n: usize) -> RotY<'w> {
        RotY { y, tr: &self.tr, n }
    }
}

/// The SR force pieces on the energy's walk: the rotated stage (rotated
/// orbital shells, its own split pieces, `Y' = Tᵀ Y T`) when `rotd` is set,
/// else [`sr_force_parts`] on `st`.
#[allow(clippy::too_many_arguments)]
fn sr_force_stage(
    st: &Stage<'_>,
    plan: Option<&SplitPlan>,
    rotd: Option<&SrRotatedDeriv>,
    images: &[[f64; 3]],
    y: &Array2<f64>,
    wm: &Array2<f64>,
    natoms: usize,
    budget: usize,
    serial: bool,
) -> Result<SrForce, FerricError> {
    let Some(r) = rotd else {
        let y3 = Y3::AuxMajor(y);
        return sr_force_parts(st, plan, images, (y, y3), wm, natoms, budget, serial);
    };
    let st_rot = st.with_bases(&r.rot.prep, st.aux)?;
    let plan_rot = plan.map(|p| p.sr3_plan_on(&st_rot)).transpose()?;
    let ry = r.weights(y, st.obs.nbasis());
    let y3 = Y3::Rotated(&ry);
    sr_force_parts(
        &st_rot,
        plan_rot.as_ref(),
        images,
        (y, y3),
        wm,
        natoms,
        budget,
        false,
    )
}

/// SR force pieces of [`fit_derivatives`].
struct SrForce {
    orb: Array2<f64>,
    aux3: Array2<f64>,
    metric: Array2<f64>,
    n_sr3: usize,
    n_sr2: usize,
}

/// The SR 3-centre and metric force pieces: the split partition's kept
/// calls / compact pairs with a plan, else the unsplit walks (the frozen
/// serial oracles when `serial`).
#[allow(clippy::too_many_arguments)]
fn sr_force_parts(
    st: &Stage<'_>,
    plan: Option<&SplitPlan>,
    images: &[[f64; 3]],
    (y, y3): (&Array2<f64>, Y3<'_>),
    wm: &Array2<f64>,
    natoms: usize,
    budget: usize,
    serial: bool,
) -> Result<SrForce, FerricError> {
    let ((orb, aux3, n_sr3), (metric, n_sr2)) = match (plan, serial) {
        (Some(p), _) => (
            p.sr3_force_with(st, images, natoms, budget, |_, _| y3)?,
            p.sr2_force(st, wm, budget)?,
        ),
        (None, true) => (
            sr3_force_serial(st, images, y, natoms)?,
            sr2_force_serial(st, wm)?,
        ),
        (None, false) => (
            sr3_force(st, images, natoms, budget, |_, _| y3)?,
            sr2_force(st, budget, |_| wm)?,
        ),
    };
    Ok(SrForce {
        orb,
        aux3,
        metric,
        n_sr3,
        n_sr2,
    })
}

/// The unsplit LR force pass (Iteration 18): orbital `2 w Re[Q* (Y X)]`,
/// aux `w Re[(Y P)* (−iG X)]`, metric `2 w Re[(−iG X)* (Wm X)]`. G chunks
/// are evaluated `inflight` at a time and added in chunk order
/// ([`LrChunk`]): bit-identical to the serial pass at any thread count.
fn lr_force_unsplit(
    st: &Stage<'_>,
    gv: &[[f64; 3]],
    (y, wm): (&Array2<f64>, &Array2<f64>),
    natoms: usize,
    (chunk_budget, inflight): (usize, usize),
) -> Result<LrForce, FerricError> {
    let (cell, obs) = (st.cell, st.obs);
    let n = obs.nbasis();
    let n2 = n * n;
    let naux = st.aux.nbasis();
    let omega = st.omega;
    let vol = cell.volume();
    let pair_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
    // Per G: P re/im + Σ_P Y X re/im (4 × 8 n²); X (16 naux) + its re/im,
    // Σ_μν Y P re/im and Wm X re/im (6 × 8 naux).
    let extra_per_g = n2
        .saturating_mul(32)
        .saturating_add(naux.saturating_mul(64))
        .saturating_add(64);
    let aoat = ao_atoms(obs);
    let mut out = LrForce::zeros(natoms, naux);
    let mut acc = LrAcc::new(&mut out, &aoat);
    let aux_sh = &st.aux_sh;
    let n_chunks = pair_ft_deriv_chunked_ordered(
        cell,
        obs,
        gv,
        pair_thresh,
        chunk_budget,
        extra_per_g,
        inflight,
        |gs, p, q| {
            let ng = gs.len();
            let x = aux_ft_shells(aux_sh, naux, gs);
            let xr = x.mapv(|z| z.re);
            let xi = x.mapv(|z| z.im);
            drop(x);
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
            drop(pr);
            drop(pim);
            // (Wm X)_R(G), (naux, ng)
            let mut zr = Array2::<f64>::zeros((naux, ng));
            let mut zi = Array2::<f64>::zeros((naux, ng));
            general_mat_mul(1.0, wm, &xr, 0.0, &mut zr);
            general_mat_mul(1.0, wm, &xi, 0.0, &mut zi);
            let xyc = (xyr, xyi);
            let mut ck = LrChunk::new(ng, n, naux, true);
            for (g, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let w = 2.0 / vol * 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp();
                // Orbital: 2 w Σ_{μ∈A,ν} Re[Q* (Σ_P Y X)].
                ck.set_orbital(q, &xyc, g, 2.0 * w);
                for pp in 0..naux {
                    let (xre, xim) = (xr[(pp, g)], xi[(pp, g)]);
                    // J3 aux: w Re[(ΣYP)* (−iG X)] = w G (PY.re X.im − PY.im X.re)
                    let t3 = w * (pyr[(pp, g)] * xim - pyi[(pp, g)] * xre);
                    // J2: 2 w Re[(−iG X)* Z] = 2 w G (X.im Z.re − X.re Z.im)
                    let t2 = 2.0 * w * (xim * zr[(pp, g)] - xre * zi[(pp, g)]);
                    ck.set_aux(g, pp, t3, Some(t2));
                }
            }
            ck
        },
        |gs, ck| {
            acc.push(gs, ck);
            Ok(())
        },
    )?;
    acc.finish();
    out.n_chunks = n_chunks;
    Ok(out)
}

/// TEST ORACLE: the frozen serial unsplit LR pass (see `lr_tests`).
#[cfg(test)]
/// The unsplit LR force pass (Iteration 18): orbital `2 w Re[Q* (Y X)]`,
/// aux `w Re[(Y P)* (−iG X)]`, metric `2 w Re[(−iG X)* (Wm X)]`.
fn lr_force_unsplit_serial_oracle(
    st: &Stage<'_>,
    gv: &[[f64; 3]],
    y: &Array2<f64>,
    wm: &Array2<f64>,
    natoms: usize,
    chunk_budget: usize,
) -> Result<LrForce, FerricError> {
    let (cell, obs) = (st.cell, st.obs);
    let n = obs.nbasis();
    let n2 = n * n;
    let naux = st.aux.nbasis();
    let omega = st.omega;
    let vol = cell.volume();
    let pair_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
    // Per G: P re/im + Σ_P Y X re/im (4 × 8 n²); X (16 naux) + its re/im,
    // Σ_μν Y P re/im and Wm X re/im (6 × 8 naux).
    let extra_per_g = n2
        .saturating_mul(32)
        .saturating_add(naux.saturating_mul(64))
        .saturating_add(64);
    let sh2at = obs.shell_to_atom().to_vec();
    let mut aoat = vec![0usize; n];
    {
        let dims = obs.shell_dims();
        let offs = obs.shell_offsets();
        for sh in 0..obs.nshells() {
            for k in 0..dims[sh] {
                aoat[offs[sh] + k] = sh2at[sh];
            }
        }
    }
    let mut out = LrForce::zeros(natoms, naux);
    let (orb_lr, aux3_lr, metric_lr) = (&mut out.orb, &mut out.aux3, &mut out.metric);
    let aux_sh = &st.aux_sh;
    let n_chunks = crate::pair_ft::pair_ft_deriv_chunked(
        cell,
        obs,
        gv,
        pair_thresh,
        chunk_budget,
        extra_per_g,
        |_g0, gs, p, q| {
            let ng = gs.len();
            let x = aux_ft_shells(aux_sh, naux, gs);
            let xr = x.mapv(|z| z.re);
            let xi = x.mapv(|z| z.im);
            drop(x);
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
            drop(pr);
            drop(pim);
            // (Wm X)_R(G), (naux, ng)
            let mut zr = Array2::<f64>::zeros((naux, ng));
            let mut zi = Array2::<f64>::zeros((naux, ng));
            general_mat_mul(1.0, wm, &xr, 0.0, &mut zr);
            general_mat_mul(1.0, wm, &xi, 0.0, &mut zi);
            for (g, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let w = 2.0 / vol * 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp();
                // Orbital: 2 w Σ_{μ∈A,ν} Re[Q* (Σ_P Y X)].
                for mu in 0..n {
                    let a = aoat[mu];
                    let mut acc = [0.0_f64; 3];
                    for nu in 0..n {
                        let mn = mu * n + nu;
                        let (ar, ai) = (xyr[(mn, g)], xyi[(mn, g)]);
                        for (c, qc) in q.iter().enumerate() {
                            let qz = qc[[mu, nu, g]];
                            acc[c] += qz.re * ar + qz.im * ai;
                        }
                    }
                    for c in 0..3 {
                        orb_lr[(a, c)] += 2.0 * w * acc[c];
                    }
                }
                for pp in 0..naux {
                    let (xre, xim) = (xr[(pp, g)], xi[(pp, g)]);
                    // J3 aux: w Re[(ΣYP)* (−iG X)] = w G (PY.re X.im − PY.im X.re)
                    let t3 = w * (pyr[(pp, g)] * xim - pyi[(pp, g)] * xre);
                    // J2: 2 w Re[(−iG X)* Z] = 2 w G (X.im Z.re − X.re Z.im)
                    let t2 = 2.0 * w * (xim * zr[(pp, g)] - xre * zi[(pp, g)]);
                    for c in 0..3 {
                        aux3_lr[(pp, c)] += t3 * gvec[c];
                        metric_lr[(pp, c)] += t2 * gvec[c];
                    }
                }
            }
            Ok(())
        },
    )?;
    out.n_chunks = n_chunks;
    Ok(out)
}

// ---------------------------------------------------------------------------
// SR derivative walks: ordered-parallel (production) and frozen serial
// (oracle). Shared with the k-point forces (`kpoint::kderiv`).
// ---------------------------------------------------------------------------

/// The fitted 3-index weight of one SR triplet's `(P, μν)` elements:
/// Gamma `Y[P, μν]` (aux-major) or a k-point phase-folded bin `Z[μν, P]`
/// (pair-major). Both return the element the serial bodies read.
#[derive(Clone, Copy)]
pub(super) enum Y3<'w> {
    AuxMajor(&'w Array2<f64>),
    PairMajor(&'w Array2<f64>),
    /// Gamma `Y` on the COLUMN-ROTATED orbital basis (module doc, "Column
    /// rotation"): `Y'[P, mn] = Σ_{μν} T_μm T_νn Y[P, μν]`, read element by
    /// element from the parent `Y`.
    Rotated(&'w RotY<'w>),
}

/// `Y` (aux-major, parent AO basis) and the forward transform onto the
/// rotated AO basis of a rotated SR derivative walk.
pub(super) struct RotY<'w> {
    pub(super) y: &'w Array2<f64>,
    pub(super) tr: &'w AoTranspose,
    /// Orbital AOs.
    pub(super) n: usize,
}

impl RotY<'_> {
    /// `Y'[P, mn]` at `row = m·n + n'` (rotated AO indices).
    #[inline]
    fn at(&self, prow: usize, row: usize) -> f64 {
        let n = self.n;
        self.tr
            .pair(row / n, row % n, |a, b| self.y[(prow, a * n + b)])
    }
}

impl Y3<'_> {
    #[inline]
    pub(in crate::rsgdf) fn at(self, prow: usize, row: usize) -> f64 {
        match self {
            Y3::AuxMajor(y) => y[(prow, row)],
            Y3::PairMajor(z) => z[(row, prow)],
            Y3::Rotated(r) => r.at(prow, row),
        }
    }
}

/// One SR 3-centre triplet's force contributions, each summed FROM ZERO in
/// the serial body's order: `gpx[pp]` (aux function `P.off + pp`), `ga`
/// (bra atom), `gb` (ket atom). The serial walk adds each of them ONCE into
/// its shared row, so forming them in parallel keeps every row's scalar
/// sequence.
pub(super) struct Sr3Contrib {
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

/// The serial body's contraction of one derivative block `blk`
/// (`[d/dP, d/d(sh1), d/d(sh2)]`, each `(nP, n1, n2)`) with `y`.
fn sr3_contract(
    st: &Stage<'_>,
    blk: &[f64],
    (i1, i2, ip): (usize, usize, usize),
    y: Y3<'_>,
) -> Sr3Contrib {
    let n = st.obs.nbasis();
    let (a, b, p) = (&st.obs_sh[i1], &st.obs_sh[i2], &st.aux_sh[ip]);
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
                let yv = y.at(prow, r0 + j);
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

/// The serial body's accumulation of one triplet into the shared rows.
fn sr3_add(
    st: &Stage<'_>,
    c: &Sr3Contrib,
    (i1, i2, ip): (usize, usize, usize),
    orb: &mut Array2<f64>,
    auxg: &mut Array2<f64>,
) {
    let sh2at = st.obs.shell_to_atom();
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
}

/// SR 3-centre force pieces `(orbital natoms × 3, aux naux × 3, count)`:
/// `Σ y · d(μ_0 ν_L | P_T)` over the energy's triplets, `y` of each triplet
/// from `weights(L, T)`. ORDERED-PARALLEL (module [`crate::ordered`]):
/// derivative block + [`Sr3Contrib`] in parallel, [`sr3_add`] serially in
/// the serial walk's order — BIT-IDENTICAL to [`sr3_force_serial`] (Gamma)
/// and to the k-point oracle.
pub(super) fn sr3_force<'w, W>(
    st: &Stage<'_>,
    images: &[[f64; 3]],
    natoms: usize,
    budget: usize,
    weights: W,
) -> Result<(Array2<f64>, Array2<f64>, usize), FerricError>
where
    W: Fn([f64; 3], [f64; 3]) -> Y3<'w> + Sync,
{
    let mut orb = Array2::<f64>::zeros((natoms, 3));
    let mut auxg = Array2::<f64>::zeros((st.aux.nbasis(), 3));
    let pool = st.sr3_deriv_pool()?;
    let count = st.sr_three_index_ordered(
        images,
        &pool,
        budget,
        |eng, i1, i2, ip, l, t| {
            Ok(eng
                .compute_eri3_deriv_shifted(st.obs, st.aux, ip, i1, i2, [t, [0.0; 3], l])?
                .map(|blk| sr3_contract(st, blk, (i1, i2, ip), weights(l, t))))
        },
        |i1, i2, ip, _, _, c: Sr3Contrib| {
            sr3_add(st, &c, (i1, i2, ip), &mut orb, &mut auxg);
            Ok(())
        },
    )?;
    Ok((orb, auxg, count))
}

/// The serial body's metric accumulation of one `d/dP (P_0 | Q_T)` block
/// (`blk`, `[x · nb + i nQ + j]`), `d/dQ = −d/dP`.
fn sr2_add(
    st: &Stage<'_>,
    blk: &[f64],
    (ip, iq): (usize, usize),
    wm: &Array2<f64>,
    metric: &mut Array2<f64>,
) {
    let (p, q) = (&st.aux_sh[ip], &st.aux_sh[iq]);
    let nb = p.nfun * q.nfun;
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
}

/// SR metric force piece `(naux × 3, count)`: `Σ Wm · d(P_0 | Q_T)` with
/// `Wm` of each pair from `weights(T)`. ORDERED-PARALLEL: the derivative
/// block (its `d/dP` third) is computed in parallel and STORED; the serial
/// body [`sr2_add`] (a per-ELEMENT `+=`/`−=` into shared aux rows, so no
/// per-pair subtotal is allowed) replays in walk order — BIT-IDENTICAL to
/// [`sr2_force_serial`] and the k-point oracle.
pub(super) fn sr2_force<'w, W>(
    st: &Stage<'_>,
    budget: usize,
    weights: W,
) -> Result<(Array2<f64>, usize), FerricError>
where
    W: Fn([f64; 3]) -> &'w Array2<f64>,
{
    let mut metric = Array2::<f64>::zeros((st.aux.nbasis(), 3));
    let pool = st.sr2_deriv_pool()?;
    let count = st.sr_metric_ordered(
        &pool,
        budget,
        |eng, ip, iq, t| {
            let nb3 = 3 * st.aux_sh[ip].nfun * st.aux_sh[iq].nfun;
            Ok(eng
                .compute_eri2_deriv_shifted(st.aux, ip, iq, t)?
                .map(|blk| blk[..nb3].to_vec()))
        },
        |ip, iq, t, blk: Vec<f64>| {
            sr2_add(st, &blk, (ip, iq), weights(t), &mut metric);
            Ok(())
        },
    )?;
    Ok((metric, count))
}

/// FROZEN pre-parallel serial SR 3-centre force walk (Gamma; the
/// bit-identity oracle of [`sr3_force`]; do not "improve").
fn sr3_force_serial(
    st: &Stage<'_>,
    images: &[[f64; 3]],
    y: &Array2<f64>,
    natoms: usize,
) -> Result<(Array2<f64>, Array2<f64>, usize), FerricError> {
    let (obs, aux) = (st.obs, st.aux);
    let n = obs.nbasis();
    let naux = aux.nbasis();
    let sh2at = obs.shell_to_atom().to_vec();
    let mut orb_sr = Array2::<f64>::zeros((natoms, 3));
    let mut aux3_sr = Array2::<f64>::zeros((naux, 3));
    let mut eng3 = Engine::new_3center_deriv(Operator::erfc(st.omega), obs, aux, ENGINE_PRECISION)?;
    let n_sr3 = st.sr_three_index_walk(images, |i1, i2, ip, l, t| {
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
            let mut gpx = [0.0_f64; 3];
            for i in 0..a.nfun {
                let r0 = (a.off + i) * n + b.off;
                for j in 0..b.nfun {
                    let yv = y[(prow, r0 + j)];
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
            for x in 0..3 {
                aux3_sr[(prow, x)] += gpx[x];
            }
        }
        for x in 0..3 {
            orb_sr[(sh2at[i1], x)] += ga[x];
            orb_sr[(sh2at[i2], x)] += gb[x];
        }
        Ok(())
    })?;
    Ok((orb_sr, aux3_sr, n_sr3))
}

/// FROZEN pre-parallel serial SR metric force walk (Gamma; the
/// bit-identity oracle of [`sr2_force`]; do not "improve").
fn sr2_force_serial(st: &Stage<'_>, wm: &Array2<f64>) -> Result<(Array2<f64>, usize), FerricError> {
    let aux = st.aux;
    let naux = aux.nbasis();
    let mut metric_sr = Array2::<f64>::zeros((naux, 3));
    let mut eng2 = Engine::new_2center_deriv(Operator::erfc(st.omega), aux, ENGINE_PRECISION)?;
    let n_sr2 = st.sr_metric_walk(|ip, iq, t| {
        let blk = match eng2.compute_eri2_deriv_shifted(aux, ip, iq, t)? {
            Some(b) => b,
            None => return Ok(()),
        };
        let (p, q) = (&st.aux_sh[ip], &st.aux_sh[iq]);
        let nb = p.nfun * q.nfun;
        for i in 0..p.nfun {
            for j in 0..q.nfun {
                let wv = wm[(p.off + i, q.off + j)];
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

/// Check how aux centres map onto cell atoms: `jac` is `dC_k/dR_A`
/// (`n_aux_atoms × natoms`, the same for x/y/z) when the aux centres are
/// functions of the atom positions (e.g. ghost sites); `None` requires the
/// aux basis to sit on exactly the cell's atoms (aux atom k = cell atom k).
pub(crate) fn check_aux_map(
    cell: &Cell,
    aux: &PreparedBasis,
    jac: Option<&Array2<f64>>,
) -> Result<(), FerricError> {
    let pos = cell.positions();
    let naux_at = aux.atoms().len();
    match jac {
        Some(j) => {
            if j.dim() != (naux_at, pos.len()) {
                return Err(FerricError::General(format!(
                    "RS-GDF forces: aux_jac has shape {:?}, expected (aux atoms {naux_at}, cell atoms {})",
                    j.dim(),
                    pos.len()
                )));
            }
            if j.iter().any(|v| !v.is_finite()) {
                return Err(FerricError::General(
                    "RS-GDF forces: non-finite aux_jac".into(),
                ));
            }
        }
        None => {
            if naux_at != pos.len() {
                return Err(FerricError::General(format!(
                    "RS-GDF forces: aux basis has {naux_at} centres but the cell has {} atoms; \
                     pass aux_jac for aux centres that are not the cell's atoms",
                    pos.len()
                )));
            }
            for (k, (a, p)) in aux.atoms().iter().zip(&pos).enumerate() {
                let d = ((a.x - p[0]).powi(2) + (a.y - p[1]).powi(2) + (a.z - p[2]).powi(2)).sqrt();
                if d > 1e-10 {
                    return Err(FerricError::General(format!(
                        "RS-GDF forces: aux centre {k} is {d:.3e} Bohr from cell atom {k}; \
                         pass aux_jac for aux centres that are not the cell's atoms"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Fold a per-aux-function `dE/dC` (`naux × 3`) onto the cell atoms through
/// the aux shells' centres and `jac` ([`check_aux_map`]).
pub(crate) fn fold_aux(
    per_fun: &Array2<f64>,
    aux: &PreparedBasis,
    jac: Option<&Array2<f64>>,
    natoms: usize,
) -> Array2<f64> {
    let naux_at = aux.atoms().len();
    let sh2at = aux.shell_to_atom();
    let dims = aux.shell_dims();
    let offs = aux.shell_offsets();
    let mut per_center = Array2::<f64>::zeros((naux_at, 3));
    for sh in 0..aux.nshells() {
        for f in 0..dims[sh] {
            for x in 0..3 {
                per_center[(sh2at[sh], x)] += per_fun[(offs[sh] + f, x)];
            }
        }
    }
    match jac {
        None => per_center,
        Some(j) => {
            let mut out = Array2::<f64>::zeros((natoms, 3));
            for k in 0..naux_at {
                for a in 0..natoms {
                    let c = j[(k, a)];
                    if c == 0.0 {
                        continue;
                    }
                    for x in 0..3 {
                        out[(a, x)] += c * per_center[(k, x)];
                    }
                }
            }
            out
        }
    }
}
#[cfg(test)]
mod lr_tests;
