//! Amplitude-threshold LinLCCD in the localized basis — the ladder member
//! of the single-threshold family (MP2: `ferric_mp2::lmp2_amplitude`;
//! rings/dRPA: `ferric_mp2::drpa_amplitude`; ladders: here, applied by
//! `ferric_mp2::ladders`).
//!
//! Built AFTER the proof notebook
//! `wiki/notebooks/13-amplitude-threshold-linlccd.ipynb`, which establishes
//! by derivation-by-verification (spin-orbital vs candidate spatial residual
//! on random symmetric inputs, 1e-14):
//!
//! ```text
//! R_iajb = (ia|jb) + F(T)_iajb + Σ_kl (ik|jl) T_kalb   [hh ladder]
//!                              + Σ_cd (ac|bd) T_icjd   [pp ladder]
//! E      = Σ (ia|jb) (2 T_iajb − T_ibja)
//! ```
//!
//! and that the system is linear with a symmetric PD operator (Fock part PD
//! for gapped systems; both ladder blocks are RI Gram matrices, hence PSD),
//! so a Hylleraas (stationary) functional of the masked solve WOULD make the
//! solver error enter quadratically. NOTE: the energy computed here is the
//! LINEAR form E = Σ (ia|jb)(2T − T̃) (see `masked_solve`), whose solver
//! error is FIRST order in the CG residual; switching it to the stationary
//! form (as the molecular LMP2 drivers did in perf Stage D) is an open
//! decision, alongside ferric-pbc's Gamma LMP2.
//!
//! MEASURED CORRECTION to the notebook's PSD claim (2026-09-06,
//! proto_linlccd_pp_psd.py / WIKI-APPEND-linlccd-direct.md): in the
//! exchange pairing the ladder supermatrices are Σ_P Y_P ⊗ Y_P with Y_P
//! symmetric — NOT a manifest Gram. The exact-integral operator is PSD by
//! pointwise kernel positivity, but the RI projection breaks that
//! argument, and the RI blocks were MEASURED slightly indefinite for the
//! attenuated operator (erfc ω=1, alkane_4: λ_min = −6.9e-4) and
//! size-decreasing for Coulomb (+0.199 water → +0.045 C8). The OPERATIVE
//! CG license — global and direct paths alike — is that this bounded
//! indefiniteness stays orders below the Fock denominator floor (~2.4 Ha
//! on the measured systems), so the full operator remains SPD by Weyl.
//!
//! The canonical anchor is [`crate::linlccd::linlccd`] — ferric's
//! spin-orbital einsum implementation (Carter-Fenk JPCA 2025), an
//! independent formulation sharing only the RI integrals. `LadderVariant`
//! is reused directly so each tier (DriversOnly ≡ RI-MP2 / Hh / Full)
//! anchors against its canonical counterpart.
//!
//! V1 SCOPE: dense masked CG over the (no·nv)² compound space (prototype
//! parity with the MP2/dRPA rigs); the OOOO/VVVV ladder blocks are kept
//! DENSE — thresholding them is future work, so no cost or scaling claim
//! attaches to this module. Closed-shell RHF, library-only. `Hh` never
//! forms the VVVV block, matching the canonical implementation's memory
//! shape.

use ndarray::Array2;

use ferric_core::basis::BasisSet;
use ferric_core::error::FerricError;
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;

use crate::linlccd::LadderVariant;
use ferric_mp2::ladders::{apply_ladders_into, PpSource};
use ferric_mp2::lmp2_amplitude::{
    assemble_basis, assemble_ragged_direct, checked_vvhv, localized_spaces, AmplitudeLmp2Config,
    VvHv,
};
use ferric_mp2::lmp2_direct::{
    assemble_boo_direct, assemble_pp_fitted_direct, assemble_ragged_direct_local, DirectConfig,
    DirectStats, OoGram, PpFitted,
};
use ferric_mp2::mo_transform::whitened_oo_vv;
use ferric_mp2::ragged::{kept_fraction, matvec_indexed, solve_ragged_with, Ragged};

/// Configuration for amplitude-threshold LinLCCD.
#[derive(Debug, Clone)]
pub struct AmplitudeLinLccdConfig {
    /// Threshold ε on |(ia|jb)| (Eq-8 symmetric test, swap-closed); 0 keeps
    /// everything (the exactness-anchor limit).
    pub eps: f64,
    /// Number of lowest occupied MOs excluded from the localized occupied
    /// space.
    pub frozen_core: usize,
    /// Relative residual tolerance of the ragged PCG amplitude solve. Strict
    /// by design: LinLCCD's energy is the LINEAR form, first order in
    /// the residual, so LMP2's relaxed Hylleraas default does not apply.
    pub cg_rtol: f64,
    /// PCG iteration cap; reaching it without meeting `cg_rtol` is an error.
    pub cg_max_iter: usize,
    /// Byte budget forwarded to the RI 3-index build blocking; `None`
    /// auto-resolves.
    pub eri3_budget_bytes: Option<usize>,
    /// Integral-free pair gate (see `lmp2_amplitude`); direct path only —
    /// gated pairs are never assembled, and the hh Gram's gated columns
    /// are exactly zero (the Gram/PSD license is unchanged). `None`
    /// (default) keeps every pair (the trivial-anchor limit).
    pub pair_gate_cal: Option<f64>,
}

impl Default for AmplitudeLinLccdConfig {
    fn default() -> Self {
        Self {
            eps: 1e-4,
            frozen_core: 0,
            cg_rtol: 1e-11,
            cg_max_iter: 600,
            eri3_budget_bytes: None,
            pair_gate_cal: None,
        }
    }
}

/// Result of an amplitude-threshold LinLCCD calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct AmplitudeLinLccdResult {
    /// LinLCCD correlation energy (Hartree) from the linear energy form
    /// `Σ (ia|jb)(2 T_iajb − T_ibja)` over the kept pattern.
    pub e_corr: f64,
    /// Reference SCF energy plus `e_corr` (Hartree).
    pub e_total: f64,
    /// Kept `(ia, jb)` amplitude entries divided by `(no·nv)²`.
    pub keep_fraction: f64,
    /// PCG iterations taken by the amplitude solve.
    pub cg_iterations: usize,
    /// Final PCG relative residual `‖r‖ / ‖b‖`.
    pub cg_relres: f64,
    /// Always `true`: a solve that does not converge returns an error instead.
    pub cg_converged: bool,
}

/// Amplitude-threshold LinLCCD with the VV-HV space built internally.
#[allow(clippy::too_many_arguments)]
pub fn amplitude_linlccd(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLinLccdConfig,
    variant: LadderVariant,
) -> Result<AmplitudeLinLccdResult, FerricError> {
    let (vvhv, _) = checked_vvhv(mol, obs, obs_bs, rhf, "linlccd_amplitude")?;
    amplitude_linlccd_with_virtuals(mol, obs, dfbs, op, rhf, cfg, variant, &vvhv)
}

/// Mutation-test entry point.
#[allow(clippy::too_many_arguments)]
pub fn amplitude_linlccd_with_virtuals(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLinLccdConfig,
    variant: LadderVariant,
    vvhv: &VvHv,
) -> Result<AmplitudeLinLccdResult, FerricError> {
    let lcfg = AmplitudeLmp2Config::plain_boys(cfg.eps, cfg.frozen_core, cfg.eri3_budget_bytes);
    let lb = assemble_basis(mol, obs, dfbs, op, rhf, &lcfg, vvhv)?;
    let (no, nv) = (lb.no, lb.nv);
    // (ia|jb) directly onto ragged blocks — the dense (no·nv)² tensor is
    // never formed (the lmp2 direct-assembly path, scale 1.0)
    let (rg, _gated) = assemble_ragged_direct(mol, dfbs, op, &lb, cfg.eps, 1.0, None, None)?;

    // Fail-fast size guard for the dense terms THIS function allocates
    // beyond `assemble_basis`/`assemble_ragged_direct` (which gate their own
    // `eri3` blocking internally via `lcfg.eri3_budget_bytes`).
    //
    // Before this, none of the four public entry points in this file gated
    // anything: `eri3_ao` (naux·nbas², built once for `oo_g` and, for
    // `LadderVariant::Full`, a SECOND time for `bvv_t` — each transient and
    // non-overlapping, so charged once as the dominant transient, not
    // doubled), `oo_g` (no²·no², resident) and `bvv_t` (naux·nv², resident,
    // Full only) were all unbounded allocations.
    let naux = dfbs.nbasis();
    let nbas = obs.nbasis();
    let mut plan = MemoryPlan::resolve(
        cfg.eri3_budget_bytes,
        format!("amplitude LinLCCD {variant:?} (no={no}, nv={nv}, naux={naux})"),
    );
    plan.reserve(
        "eri3_ao (P|mn) (built once for oo_g, and again for bvv_t under Full — non-overlapping)",
        naux.saturating_mul(nbas).saturating_mul(nbas),
        Lifetime::Transient,
    );
    plan.reserve(
        "oo_g (ik|jl) whitened Gram",
        no.saturating_pow(2).saturating_mul(no.saturating_pow(2)),
        Lifetime::Resident,
    );
    if variant.needs_vvvv_pub() {
        plan.reserve(
            "bvv_t whitened (naux, nv^2)",
            naux.saturating_mul(nv.saturating_pow(2)),
            Lifetime::Resident,
        );
    }
    // COMMIT against the process-global pool rather than merely CHECK against
    // a ceiling every other subsystem may re-read in full. `commit()` is
    // `check()` plus a DEBIT plus an RAII guard; with no pool installed it
    // degenerates to exactly the old `check()` plus an inert guard, which is
    // the trivial limit pinned by `mwe_cc_pool_is_inert_without_a_pool.rs`.
    //
    // HARD: every reservation on this plan is a dense tensor with no streaming
    // fallback in this driver, so there is no None branch to take and a soft
    // gate would be decoration. `_charge` is held to the end of the function,
    // which is where those tensors die.
    if let Some(pool) = ferric_core::memory::pool::global() {
        plan = plan.with_pool(&pool);
    }
    let _charge = plan.commit()?;

    // hh ladder coefficients (ik|jl): the OOOO block is no⁴ — tiny — via
    // the same whitened RI Gram as the canonical implementation. pp ladder:
    // whitened B over VV pairs, gathered per pair on demand — the (nv²)²
    // VVVV tensor is never formed.
    let (boo_t, bvv_t) = whitened_oo_vv(
        op,
        obs,
        dfbs,
        &lb.v2c,
        &lb.c_locc,
        variant.needs_vvvv_pub().then_some(&vvhv.c_vloc),
    )?;
    let oo_g = boo_t.t().dot(&boo_t); // rows (i,k), cols (j,l): (ik|jl)
    drop(boo_t);

    let oo = if matches!(variant, LadderVariant::DriversOnly) {
        None
    } else {
        Some(OoGram::dense(oo_g, no))
    };
    let pp = match &bvv_t {
        Some(bvv) => PpSource::GlobalWhitened { bvv_t: bvv, nv },
        None => PpSource::None,
    };
    let (e_corr, it, relres) = linlccd_masked_solve(&rg, &lb.f_oo, oo.as_ref(), pp, cfg)?;

    let keep_fraction = kept_fraction(&rg, no, nv);
    Ok(AmplitudeLinLccdResult {
        e_corr,
        e_total: rhf.energy + e_corr,
        keep_fraction,
        cg_iterations: it,
        cg_relres: relres,
        cg_converged: true,
    })
}

/// Masked ragged CG on the LinLCCD linear system, the solver
/// shared by the global-B path ([`amplitude_linlccd_with_virtuals`]) and
/// the integral-direct path ([`amplitude_linlccd_direct_with_virtuals`]),
/// extracted verbatim (the `riccati_masked_solve` pattern from the dRPA
/// port): F(t) + optional hh gather (coefficients via [`OoGram`]) +
/// optional pp per-pair domain Grams from a whitened global Bvv, then the
/// linear energy form Σ (ia|jb)(2T − T̃) on the pattern (not the stationary
/// Hylleraas functional: its solver error is first order in the residual).
/// Non-convergence is a hard error, so
/// `Ok` implies convergence.
fn linlccd_masked_solve(
    rg: &Ragged,
    f_oo: &Array2<f64>,
    oo: Option<&OoGram>,
    pp: PpSource<'_>,
    cfg: &AmplitudeLinLccdConfig,
) -> Result<(f64, usize, f64), FerricError> {
    let matvec = |t: &[Array2<f64>], flops: &mut u64| -> Vec<Array2<f64>> {
        // F(t) (rayon-parallel, deterministic), then the hh/pp ladders
        // (ferric_mp2::ladders: per-output-pair tasks, pattern-projected).
        let mut r = matvec_indexed(rg, f_oo, t, flops);
        apply_ladders_into(rg, t, oo, &pp, &mut r, flops);
        r
    };

    let (t, it, relres, converged, _flops) =
        solve_ragged_with(rg, cfg.cg_rtol, cfg.cg_max_iter, matvec);
    if !converged {
        return Err(FerricError::General(format!(
            "linlccd_amplitude(ragged): CG failed to converge (relres {relres:.2e} after {it} iters)"
        )));
    }

    // E = Σ (ia|jb)(2 T_iajb − T_ibja) over the pattern (proof notebook §1):
    // the LINEAR form, exact at convergence but NOT stationary (its error is
    // first order in the CG residual, which cg_rtol must keep small). The
    // stationary Hylleraas form is an open decision (see the module doc).
    let mut e_corr = 0.0;
    for (p, pb) in rg.pairs.iter().enumerate() {
        let nbb = pb.db.len();
        for (r_, &a) in pb.da.iter().enumerate() {
            for (c_, &b) in pb.db.iter().enumerate() {
                if !pb.pat[r_ * nbb + c_] {
                    continue;
                }
                let jv = pb.j_blk[(r_, c_)];
                let sr = pb.pos_da[b];
                let sc = pb.pos_db[a];
                let t_swap = if sr != usize::MAX && sc != usize::MAX {
                    t[p][(sr, sc)]
                } else {
                    0.0
                };
                e_corr += jv * (2.0 * t[p][(r_, c_)] - t_swap);
            }
        }
    }
    Ok((e_corr, it, relres))
}

/// Integral-direct amplitude-threshold LinLCCD — the direct-path sibling
/// of [`amplitude_linlccd`] (the `amplitude_drpa_direct` pattern): the
/// global (naux, no·nv) B, the (naux, nao²) AO `eri3_tensor`, and the N⁵
/// whitening GEMM are never formed. RHS (ia|jb) via per-occupied sparse
/// strips ([`assemble_ragged_direct_local`], scale = 1.0); the hh OOOO
/// block via the occ-occ direct pass ([`assemble_boo_direct`] — full aux
/// rows + GLOBAL whitening, so the hh matvec stays the exact Gram the
/// proof-notebook license requires). `Full` adds the pp ladder as
/// per-pair domain-FITTED factors ([`assemble_pp_fitted_direct`]) — that
/// block's SPD contract is the MEASURED one from the Phase-1 prototype
/// (proto_linlccd_pp_psd.py; GO verdict in WIKI-APPEND-linlccd-direct.md),
/// not a Gram identity: re-run the measurement before extending it to new
/// operators or regimes.
#[allow(clippy::too_many_arguments)]
pub fn amplitude_linlccd_direct(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLinLccdConfig,
    dcfg: &DirectConfig,
    variant: LadderVariant,
) -> Result<(AmplitudeLinLccdResult, DirectStats), FerricError> {
    let (vvhv, _) = checked_vvhv(mol, obs, obs_bs, rhf, "linlccd_amplitude(direct)")?;
    amplitude_linlccd_direct_with_virtuals(mol, obs, dfbs, op, rhf, cfg, dcfg, variant, &vvhv)
}

/// [`amplitude_linlccd_direct`] with a caller-supplied virtual space (the
/// mutation-test entry point, mirroring the global path).
#[allow(clippy::too_many_arguments)]
pub fn amplitude_linlccd_direct_with_virtuals(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLinLccdConfig,
    dcfg: &DirectConfig,
    variant: LadderVariant,
    vvhv: &VvHv,
) -> Result<(AmplitudeLinLccdResult, DirectStats), FerricError> {
    let spaces = localized_spaces(mol, obs, rhf, cfg.frozen_core, vvhv)?;
    let (rg, _n_gated, dstats) = assemble_ragged_direct_local(
        mol,
        obs,
        dfbs,
        op,
        &spaces,
        cfg.eps,
        1.0,
        cfg.pair_gate_cal,
        dcfg,
    )?;
    let oo = if matches!(variant, LadderVariant::DriversOnly) {
        None
    } else {
        Some(assemble_boo_direct(
            obs,
            dfbs,
            op,
            &spaces,
            cfg.eps,
            cfg.pair_gate_cal,
            dcfg,
        )?)
    };
    // Full tier: per-pair domain-FITTED pp factors — ported on the
    // strength of the Phase-1 PSD measurement (GO verdict recorded in
    // WIKI-APPEND-linlccd-direct.md); see PpFitted's doc for the license.
    let pp_factors: Option<Vec<PpFitted>> = if variant.needs_vvvv_pub() {
        Some(assemble_pp_fitted_direct(
            mol, obs, dfbs, op, &spaces, &rg, dcfg,
        )?)
    } else {
        None
    };
    let pp = match &pp_factors {
        Some(f) => PpSource::Fitted(f),
        None => PpSource::None,
    };
    let (e_corr, it, relres) = linlccd_masked_solve(&rg, &spaces.f_oo, oo.as_ref(), pp, cfg)?;
    let keep_fraction = kept_fraction(&rg, spaces.no, spaces.nv);
    Ok((
        AmplitudeLinLccdResult {
            e_corr,
            e_total: rhf.energy + e_corr,
            keep_fraction,
            cg_iterations: it,
            cg_relres: relres,
            cg_converged: true,
        },
        dstats,
    ))
}

impl LadderVariant {
    /// Public mirror of the crate-private `needs_vvvv` used by the canonical
    /// implementation, so the amplitude-threshold port keeps the same
    /// memory shape per variant.
    pub fn needs_vvvv_pub(self) -> bool {
        matches!(self, LadderVariant::Full)
    }
}
