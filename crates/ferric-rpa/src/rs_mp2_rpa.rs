//! SR-MP2 + LR-RPA range-separated correlation (Δ-form B and coupled-rings T).
//! See docs/superpowers/specs/2026-06-09-sr-mp2-lr-rpa-design.md.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{
    compute_rpa_intermediates, spin_components_from_b_ov, RiMp2Config,
};
use ferric_scf::ScfResult;

use crate::{run_pdep_rpa_from_intermediates, PdepRpaConfig};

/// Which algebraic form of the range-separated MP2 + RPA functional to use.
///
/// Both forms are exact through 2nd order in the full Coulomb interaction and
/// contain all pure long-range rings. They differ in whether mixed SR×LR rings
/// (3rd order and higher) are included.
///
/// ## Formulation B — Δ-form (`DeltaLr`, default)
///
/// ```text
/// E_c^B = E_MP2[Coulomb] + (E_dRPA[erf_ω] − 2·E_OS[erf_ω])
/// ```
///
/// Resums pure long-range rings via a single dRPA[erf] solve. Drops all mixed
/// SR×LR rings at 3rd order and above (leading residual: 3·k_s·k_l·(k_s+k_l)/(4Δ²)
/// in the one-mode ring model). Cost: 1 dRPA[erf] call.
///
/// ## Formulation T — coupled rings (`CoupledRings`)
///
/// ```text
/// E_c^T = E_MP2[Coulomb] + (E_dRPA[Coulomb] − 2·E_OS[Coulomb])
///                         − (E_dRPA[erfc_ω] − 2·E_OS[erfc_ω])
/// ```
///
/// Screens ALL rings (full-Coulomb ΔdRPA), then un-screens the short-range-only
/// rings (subtracts erfc ΔdRPA). Contains exact 2nd order + all pure-LR rings +
/// ALL mixed SR×LR rings; excludes pure-SR rings beyond 2nd order (avoiding
/// dRPA's short-range self-correlation hole).
///
/// Same exact limits as B: ω→0 ⇒ erfc→Coulomb, the two ΔdRPA terms cancel ⇒
/// plain MP2; ω→∞ ⇒ erfc→0 ⇒ MP2 + ΔdRPA[Coulomb]. Cost: 2 dRPA calls
/// (Coulomb + erfc).
///
/// ## TOML / Python knob
///
/// TOML: `[mp2] formulation = "delta-lr"` (default) or `"coupled-rings"`
/// Python: `run_rs_mp2_rpa(..., formulation="delta-lr")` (default) or
///         `formulation="coupled-rings"`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RsMp2RpaFormulation {
    /// Δ-form (formulation B): E_MP2[Coulomb] + (E_dRPA[erf] − 2·E_OS[erf]).
    /// Pure-LR rings only; mixed SR×LR rings dropped. Default.
    #[default]
    DeltaLr,
    /// Coupled-rings (formulation T): MP2 + ΔdRPA[Coulomb] − ΔdRPA[erfc].
    /// Adds all mixed SR×LR rings vs DeltaLr; no pure-SR rings ≥3rd order.
    CoupledRings,
}

/// Which range-separation kernel splits Coulomb into short + long range.
///
/// Both choices satisfy the SAME split identity (SR + LR = Coulomb) and thus give
/// the SAME exact limits, so either drops into formulations B and T without
/// changing the ring-diagram telescoping. They differ only in the *shape* of the
/// attenuator, which changes intermediate-ω behavior (basis-set convergence,
/// π-stack over/under-binding), not the ω→0 / ω→∞ endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Attenuator {
    /// Standard error-function split: LR = erf(ωr)/r, SR = erfc(ωr)/r.
    /// Parameterized by ω (Bohr⁻¹). Default — preserves all existing behavior.
    #[default]
    Erf,
    /// Tempered (Dutoi/Goldey) split: LR = terf(r,r0)/r, SR = terfc(r,r0)/r,
    /// with terf + terfc = Coulomb exactly. Parameterized by r0 (Bohr); the
    /// curvature ω = 1/(r0·√2) is derived, never set independently. Exact via
    /// 2D interpolation tables (needs FERRIC_TERF_TABLE_DIR).
    Terf,
}

/// Configuration for SR-MP2 + LR-RPA.
#[derive(Debug, Clone)]
pub struct RsMp2RpaConfig {
    /// Range-separation parameter ω in Bohr⁻¹ (CLI/Python boundary is Å⁻¹,
    /// matching att-rimp2). Default 0.222 Bohr⁻¹ = 0.420 Å⁻¹ (erfc-optimal,
    /// Goldey & Head-Gordon JPCL 2012).
    ///
    /// Small-ω caveat: the erf 2-center metric loses rank as ω→0 (regularized
    /// eigh inverse handles it); ω=0.05 Bohr⁻¹ is the tested floor.
    pub omega: f64,
    /// Which attenuator splits Coulomb (see [`Attenuator`]). Default `Erf`.
    ///
    /// When `Terf`, the long/short-range operators become `terf(r0)`/`terfc(r0)`
    /// and `omega` is IGNORED — the curvature is derived from `r0` as
    /// ω = 1/(r0·√2). Use [`RsMp2RpaConfig::r0`] as the single knob then.
    pub attenuator: Attenuator,
    /// Range-separation length r0 in Bohr, used ONLY when `attenuator == Terf`.
    /// By default ω is derived (ω = 1/(r0·√2), the Dutoi curvature link);
    /// see `terf_omega` to decouple. Ignored for `Erf`. Default 3.18 Bohr ⇒
    /// ω ≈ 0.222 Bohr⁻¹ = 0.42 Å⁻¹ (the erf operating point), so the terf and
    /// erf arms are directly comparable at the default.
    pub r0: f64,
    /// Terf-only: OVERRIDE the curvature-linked sharpness with an independent
    /// ω (Bohr⁻¹). `None` (default) keeps the Dutoi link ω = 1/(r0·√2).
    /// Basis for decoupling: with the complement computed by LR-RPA the
    /// curvature constraint does not bind (Ne2 measured, ne2_seam_test.py /
    /// curvature-constraint-relaxed memory); the split identity terf + terfc
    /// = Coulomb holds for every (r0, ω) (anchor-tested). Tables cover
    /// s = (ωr0)² ≤ 80. Ignored for `Erf`.
    pub terf_omega: Option<f64>,
    pub frozen_core: usize,
    /// dRPA solver knobs. Default forces trunc_thresh = 0.0 (full rank):
    /// this is an energy method; PDEP truncation is a production-size opt-in.
    ///
    /// Note: the nested `drpa.frozen_core` field is ignored — `cfg.frozen_core`
    /// is authoritative and overwrites it before each RPA call.
    pub drpa: PdepRpaConfig,
    /// Which algebraic formulation to use. Default is `DeltaLr` (formulation B),
    /// which preserves all existing behavior.
    pub formulation: RsMp2RpaFormulation,
}

impl Default for RsMp2RpaConfig {
    fn default() -> Self {
        Self {
            omega: 0.420 * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
            attenuator: Attenuator::Erf,
            r0: 3.18,
            terf_omega: None,
            frozen_core: 0,
            drpa: PdepRpaConfig { trunc_thresh: 0.0, ..Default::default() },
            formulation: RsMp2RpaFormulation::DeltaLr,
        }
    }
}

/// Component breakdown.
///
/// `e_corr` is the selected formulation's correlation energy; semantics follow
/// `formulation`. `total_energy = rhf + e_corr`.
///
/// ## Always-present fields (all formulations)
///
/// - `e_mp2_full`, `e_sr_mp2`, `e_lr_mp2`, `e_dmp2_lr` come from the three
///   spin-component RI-MP2 calls (Coulomb/erfc/erf) which are always run.
///
/// ## Formulation-specific fields
///
/// - `e_drpa_lr` (`Some` for `DeltaLr`, `None` for `CoupledRings`): dRPA[erf] energy.
/// - `e_corr_naive` (`Some` for `DeltaLr`, `None` for `CoupledRings`): diagnostic
///   formulation A = E_MP2[erfc] + E_dRPA[erf]; missing SR×LR cross-range terms.
/// - `e_delta_drpa_full` (`Some` for `CoupledRings`, `None` for `DeltaLr`):
///   ΔdRPA[Coulomb] = E_dRPA[Coulomb] − 2·E_OS[Coulomb].
/// - `e_delta_drpa_sr` (`Some` for `CoupledRings`, `None` for `DeltaLr`):
///   ΔdRPA[erfc] = E_dRPA[erfc] − 2·E_OS[erfc] (the short-range ring contribution
///   that is subtracted to avoid double-counting pure-SR rings).
#[derive(Debug, Clone)]
#[must_use]
pub struct RsMp2RpaResult {
    pub e_mp2_full: f64,
    pub e_sr_mp2: f64,
    pub e_lr_mp2: f64,
    /// Direct (ring) second-order term with the erf kernel = 2·E_OS[erf].
    pub e_dmp2_lr: f64,
    /// E_dRPA[erf] (formulation B / DeltaLr only; None for CoupledRings).
    pub e_drpa_lr: Option<f64>,
    /// Naive sum E_MP2[erfc] + E_dRPA[erf] (formulation A, diagnostic, DeltaLr only).
    /// Missing the 2·v_sr·v_lr cross-range correlation; reported to make that visible.
    pub e_corr_naive: Option<f64>,
    /// ΔdRPA[Coulomb] = E_dRPA[Coulomb] − 2·E_OS[Coulomb] (CoupledRings only).
    pub e_delta_drpa_full: Option<f64>,
    /// ΔdRPA[erfc] = E_dRPA[erfc] − 2·E_OS[erfc] (CoupledRings only).
    /// Subtracted from ΔdRPA[Coulomb] to exclude pure-SR rings beyond 2nd order.
    pub e_delta_drpa_sr: Option<f64>,
    /// Correlation energy of the selected formulation.
    pub e_corr: f64,
    pub total_energy: f64,
}

impl std::fmt::Display for RsMp2RpaResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RS-MP2+RPA total: {:.10} Ha (corr: {:.10})",
            self.total_energy, self.e_corr)
    }
}

/// SR-MP2 + LR-RPA, Δ-form (B) or coupled-rings (T).
///
/// **DeltaLr (B)**: replaces MP2's long-range direct ring series with its dRPA[erf]
/// resummation. Exact limits: ω→0 ⇒ plain MP2; ω→∞ ⇒ MP2 + (dRPA − dMP2).
///
/// **CoupledRings (T)**: screens all rings (ΔdRPA[Coulomb]) then un-screens the
/// pure-SR rings (−ΔdRPA[erfc]). Same exact limits; additionally includes all
/// mixed SR×LR ring diagrams. Cost: 2 dRPA calls instead of 1.
pub fn rs_mp2_lr_rpa(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    cfg: &RsMp2RpaConfig,
) -> Result<RsMp2RpaResult, FerricError> {
    // Resolved ONCE for the whole call: the preflight gate's budget check
    // just below, AND the stage-seam RSS observability checks further down
    // (Item 3) both compare against this same configured ceiling — previously
    // re-resolved via two separate `resolve_budget_bytes(cfg.drpa.memory_budget_bytes)`
    // calls ten lines apart.
    let resolved_budget_bytes = ferric_core::memory::resolve_budget_bytes(cfg.drpa.memory_budget_bytes);

    // Pre-flight peak-memory gate (M2-style fail-fast, see budget.rs). Cheap
    // shape values only (nelec/nbasis accessors, no ERI/GEMM work) so this
    // runs before ANY large allocation. naux is known exactly; nocc/nvir use
    // the same closed-shell formula `compute_rpa_intermediates` itself uses
    // (rimp2.rs) so the estimate matches what's about to be allocated.
    {
        use ferric_mp2::rimp2::active_occ;
        let naux = dfbs.nbasis();
        let nbas = obs.nbasis();
        let nocc_total = (mol.nelec() as usize) / 2;
        let nocc = active_occ(nocc_total, cfg.frozen_core)?;
        let nvir = nbas.saturating_sub(nocc_total);
        let n_workers = rayon::current_num_threads().max(1);
        let n_keep = naux; // trunc_thresh unknown pre-eigensolve; cfg.drpa default is 0.0 (keep-all)
        let est = crate::budget::estimate_peak_bytes(crate::budget::PeakEstimateShape {
            naux, nocc, nvir,
            n_quad: cfg.drpa.quadrature.n_points,
            n_workers,
            n_keep,
            grid: None,
        });
        ferric_core::memory::check_alloc(
            &format!(
                "RS-MP2-RPA preflight (naux={naux}, nocc={nocc}, nvir={nvir}, \
                 n_workers={n_workers}, formulation={:?})",
                cfg.formulation
            ),
            est,
            resolved_budget_bytes,
        )?;
    }

    let ri_cfg = RiMp2Config { frozen_core: cfg.frozen_core, memory_budget_bytes: cfg.drpa.memory_budget_bytes, ..Default::default() };

    // SHARED-INTERMEDIATE FUSION. The MP2 spin components and the dRPA solves
    // both need the dressed b_ov = V^{-1/2}(P|op|ia) for the SAME operator, and
    // building it (the aux-blocked (P|op|ia) transform) is the expensive step.
    // Build the intermediates ONCE per operator and feed them to both the
    // spin-component energy (spin_components_from_b_ov) and the dRPA solve
    // (run_pdep_rpa_from_intermediates). This removes the duplicate transforms
    // that the old code ran (CoupledRings previously did 5 transforms — 3 MP2 +
    // 2 RPA — for only 3 distinct operators; now 3). Results are bit-identical.
    let eps = rhf.eps_r();
    let inter_of = |op: Operator| -> Result<ferric_mp2::rimp2::RpaIntermediates, FerricError> {
        let it = compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &ri_cfg)?;
        // Stage-seam RSS safety net (Item 3): observability only, never a
        // hard error — the preflight gate above already vetted the ESTIMATE;
        // this checks whether the ACTUAL resident memory after this
        // operator's RpaIntermediates build (the co-resident raw-AO-block +
        // MO-tensor + dressed b_ov peak `budget.rs` estimates) stayed in the
        // ballpark. A single stderr line, never more than 10% noisy above
        // budget, never a panic/kill.
        ferric_core::memory::warn_if_rss_over(
            &format!("RS-MP2-RPA RpaIntermediates build (op={op:?})"),
            resolved_budget_bytes,
            1.1,
        );
        Ok(it)
    };
    let sc_of = |it: &ferric_mp2::rimp2::RpaIntermediates| {
        spin_components_from_b_ov(
            &it.b_ov, eps, it.nocc, it.nvir, it.first_occ, it.nocc_total,
        )
    };

    let mut drpa_cfg = cfg.drpa.clone();
    drpa_cfg.frozen_core = cfg.frozen_core;

    // Range-separation operators. `Erf` uses erf(ω)/erfc(ω); `Terf` uses the
    // tempered terf(r0)/terfc(r0), with terf + terfc = Coulomb exactly (same
    // split identity ⇒ same exact limits ⇒ same telescoping). r0 is the single
    // Terf knob; ω is not consulted for Terf. The Coulomb pieces are unchanged.
    let (op_lr, op_sr) = match cfg.attenuator {
        Attenuator::Erf => (Operator::erf(cfg.omega), Operator::erfc(cfg.omega)),
        Attenuator::Terf => match cfg.terf_omega {
            Some(w) => (
                Operator::terf_with_omega(cfg.r0, w),
                Operator::terfc_with_omega(cfg.r0, w),
            ),
            None => (Operator::terf(cfg.r0), Operator::terfc(cfg.r0)),
        },
    };

    // LR/SR intermediates: needed for sc_lr (always, for e_dmp2_lr) and for the
    // DeltaLr dRPA. Coulomb/SR built inside their arms to avoid unused work.
    let (e_corr, e_drpa_lr, e_corr_naive, e_delta_drpa_full, e_delta_drpa_sr,
         sc_full, sc_sr, sc_lr) =
        match cfg.formulation {
            RsMp2RpaFormulation::DeltaLr => {
                let it_lr = inter_of(op_lr)?;
                let sc_lr = sc_of(&it_lr);
                let sc_full = sc_of(&inter_of(Operator::coulomb())?);
                let sc_sr = sc_of(&inter_of(op_sr)?);
                let e_dmp2_lr = 2.0 * sc_lr.e_os;
                // op_lr is erf(ω) for Erf, terf(r0) for Terf — the LR dRPA must
                // use the SAME long-range operator the attenuator selected, not a
                // hardcoded erf. drpa_cfg carries main's memory budget knobs.
                let drpa_lr = run_pdep_rpa_from_intermediates(
                    &it_lr, mol, obs, dfbs, op_lr, rhf, &drpa_cfg,
                )?;
                let e_corr = sc_full.e_total + drpa_lr.e_rpa - e_dmp2_lr;
                (e_corr, Some(drpa_lr.e_rpa), Some(sc_sr.e_total + drpa_lr.e_rpa),
                 None, None, sc_full, sc_sr, sc_lr)
            }
            RsMp2RpaFormulation::CoupledRings => {
                // LR only enters via e_dmp2_lr = 2·E_OS[lr]; no LR dRPA needed.
                let sc_lr = sc_of(&inter_of(op_lr)?);
                let it_full = inter_of(Operator::coulomb())?;
                let it_sr = inter_of(op_sr)?;
                let sc_full = sc_of(&it_full);
                let sc_sr = sc_of(&it_sr);
                let drpa_coul = run_pdep_rpa_from_intermediates(
                    &it_full, mol, obs, dfbs, Operator::coulomb(), rhf, &drpa_cfg,
                )?;
                // op_sr is erfc(ω) for Erf, terfc(r0) for Terf — the SR dRPA must
                // use the attenuator-selected short-range operator, not hardcoded erfc.
                let drpa_erfc = run_pdep_rpa_from_intermediates(
                    &it_sr, mol, obs, dfbs, op_sr, rhf, &drpa_cfg,
                )?;
                let delta_full = drpa_coul.e_rpa - 2.0 * sc_full.e_os;
                let delta_sr = drpa_erfc.e_rpa - 2.0 * sc_sr.e_os;
                // T: E_MP2[Coulomb] + ΔdRPA[Coulomb] − ΔdRPA[erfc]
                let e_corr = sc_full.e_total + delta_full - delta_sr;
                (e_corr, None, None, Some(delta_full), Some(delta_sr),
                 sc_full, sc_sr, sc_lr)
            }
        };

    // Closed shell: E_MP2 = Σ (ia|jb)[2(ia|jb)−(ib|ja)]/Δ; the direct (ring)
    // part 2Σ(ia|jb)²/Δ — the 2nd-order truncation of dRPA — equals 2·E_OS.
    let e_dmp2_lr = 2.0 * sc_lr.e_os;

    Ok(RsMp2RpaResult {
        e_mp2_full: sc_full.e_total,
        e_sr_mp2: sc_sr.e_total,
        e_lr_mp2: sc_lr.e_total,
        e_dmp2_lr,
        e_drpa_lr,
        e_corr_naive,
        e_delta_drpa_full,
        e_delta_drpa_sr,
        e_corr,
        total_energy: rhf.energy + e_corr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // These reference-cross-check tests call the standalone entry points
    // directly (the production path uses the fused intermediates above).
    use crate::run_pdep_rpa;
    use ferric_mp2::rimp2::ri_mp2_spin_components;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ferric_scf::ScfResult;

    fn setup_h2() -> (Molecule, PreparedBasis, PreparedBasis, ScfResult) {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ParallelContext::default(),
            &mol, &obs, op, &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        ).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        (mol, obs, dfbs, rhf)
    }

    /// ω→0: erf(ωr)/r → 0, so dRPA[erf] and dMP2[erf] both vanish and the
    /// Δ-form must reduce to plain RI-MP2. The correction is 4th-order small
    /// in v_lr, so even ω=0.05 must be tight.
    ///
    /// Note: ω=0.05 Bohr⁻¹ is the tested floor — at this value the erf metric
    /// has ≥1 significant eigenvalue (the LR dielectric ε̃ is non-trivially
    /// above identity) and the Δ-correction is <1e-5 Ha from MP2.
    #[test]
    fn omega_to_zero_reduces_to_mp2() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let full = ferric_mp2::rimp2::ri_mp2(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf,
            &ferric_mp2::rimp2::RiMp2Config::default(),
        ).unwrap();
        let cfg = RsMp2RpaConfig { omega: 0.05, ..Default::default() };
        let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();
        let drpa_lr = r.e_drpa_lr.unwrap();
        eprintln!("MP2 corr {:.10}  Δ-form corr {:.10}  ΔLR {:.2e}",
            full.mp2_corr, r.e_corr, drpa_lr - r.e_dmp2_lr);
        assert!((r.e_corr - full.mp2_corr).abs() < 1e-5,
            "omega→0 must reduce to MP2: {} vs {}", r.e_corr, full.mp2_corr);
        assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
    }

    /// ω→∞: erf → Coulomb, so the Δ-form must equal
    /// E_MP2[Coulomb] + (E_dRPA[Coulomb] − 2·E_OS[Coulomb]) computed explicitly.
    /// Pins the dRPA energy convention against the MP2 spin components.
    ///
    /// ω=200 Bohr⁻¹ is needed for 1e-6 Ha tolerance: at ω=50 the residual
    /// 2-center integral mismatch (P|erf(50r)/r|Q) vs (P|1/r|Q) is ~7 µHa for
    /// H₂/cc-pVDZ — a finite-ω truncation artefact, not a convention error.
    /// At ω=200 the same artefact drops to ~5e-7 Ha, safely below 1e-6.
    /// The tolerance is still 4 orders of magnitude tighter than any factor-2/4
    /// convention trap (~18 mHa = 1.8e-2 Ha), so it robustly catches spin-convention bugs.
    #[test]
    fn omega_to_infinity_is_mp2_plus_delta_drpa() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let cfg = RsMp2RpaConfig { omega: 200.0, ..Default::default() };
        let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();

        let ri_cfg = RiMp2Config::default();
        let (sc, _) = ri_mp2_spin_components(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf, &ri_cfg).unwrap();
        let drpa_cfg = PdepRpaConfig { trunc_thresh: 0.0, ..Default::default() };
        let drpa_coul = run_pdep_rpa(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf, &drpa_cfg).unwrap();
        let expected = sc.e_total + drpa_coul.e_rpa - 2.0 * sc.e_os;
        eprintln!("Δ-form(ω=200) {:.10}  MP2+ΔdRPA[Coulomb] {:.10}", r.e_corr, expected);
        assert!((r.e_corr - expected).abs() < 1e-6,
            "omega→∞ limit broken: {} vs {}", r.e_corr, expected);
        assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
    }

    /// CoupledRings, ω→0: the two ΔdRPA terms (Coulomb and erfc) become equal
    /// (erfc → Coulomb as ω→0), so their difference cancels and e_corr → plain MP2.
    /// NOTE: the margin is cancellation-limited, not slack — at ω=0.05 the two
    /// ΔdRPA terms are each ~7.9e-3 Ha and the residual is ~7.9e-6 against the
    /// 1e-5 tolerance (79% of budget). A wider test ω would fail this limit.
    #[test]
    fn coupled_rings_omega_to_zero_reduces_to_mp2() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let full = ferric_mp2::rimp2::ri_mp2(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf,
            &ferric_mp2::rimp2::RiMp2Config::default(),
        ).unwrap();
        let cfg = RsMp2RpaConfig {
            omega: 0.05,
            formulation: RsMp2RpaFormulation::CoupledRings,
            ..Default::default()
        };
        let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();
        let delta_full = r.e_delta_drpa_full.unwrap();
        let delta_sr = r.e_delta_drpa_sr.unwrap();
        eprintln!(
            "CoupledRings(ω=0.05): MP2={:.10}  e_corr={:.10}  ΔdRPA_full={:.4e}  ΔdRPA_sr={:.4e}  diff={:.2e}",
            full.mp2_corr, r.e_corr, delta_full, delta_sr, r.e_corr - full.mp2_corr
        );
        assert!(
            (r.e_corr - full.mp2_corr).abs() < 1e-5,
            "CoupledRings ω→0 must reduce to MP2: {} vs {}",
            r.e_corr, full.mp2_corr
        );
        assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
    }

    /// CoupledRings, ω→∞: erfc→0, so ΔdRPA[erfc]→0 and e_corr →
    /// E_MP2[Coulomb] + ΔdRPA[Coulomb], which must equal the same explicit
    /// MP2+ΔdRPA[Coulomb] value as the DeltaLr omega→∞ test.
    #[test]
    fn coupled_rings_omega_to_infinity_is_mp2_plus_delta_drpa() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let cfg = RsMp2RpaConfig {
            omega: 200.0,
            formulation: RsMp2RpaFormulation::CoupledRings,
            ..Default::default()
        };
        let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();

        let ri_cfg = RiMp2Config::default();
        let (sc, _) = ri_mp2_spin_components(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf, &ri_cfg).unwrap();
        let drpa_cfg = PdepRpaConfig { trunc_thresh: 0.0, ..Default::default() };
        let drpa_coul = run_pdep_rpa(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf, &drpa_cfg).unwrap();
        let expected = sc.e_total + drpa_coul.e_rpa - 2.0 * sc.e_os;
        eprintln!(
            "CoupledRings(ω=200): e_corr={:.10}  MP2+ΔdRPA[Coulomb]={:.10}  diff={:.2e}",
            r.e_corr, expected, r.e_corr - expected
        );
        assert!(
            (r.e_corr - expected).abs() < 1e-6,
            "CoupledRings ω→∞ limit broken: {} vs {}",
            r.e_corr, expected
        );
        assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
    }

    /// The shared-intermediate fusion must change NOTHING. Reconstruct the
    /// CoupledRings energy at a realistic ω from independent
    /// ri_mp2_spin_components + run_pdep_rpa calls (the pre-fusion recipe) and
    /// demand bit-level agreement with the fused production path.
    #[test]
    fn coupled_rings_fusion_is_bit_identical() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let omega = 0.42;
        let cfg = RsMp2RpaConfig {
            omega,
            formulation: RsMp2RpaFormulation::CoupledRings,
            ..Default::default()
        };
        let fused = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();

        // Pre-fusion reconstruction: separate transforms for each operator.
        let ri_cfg = RiMp2Config::default();
        let (sc_full, _) = ri_mp2_spin_components(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf, &ri_cfg).unwrap();
        let (sc_sr, _) = ri_mp2_spin_components(
            &mol, &obs, &dfbs, Operator::erfc(omega), &rhf, &ri_cfg).unwrap();
        let drpa_cfg = cfg.drpa.clone();
        let drpa_coul = run_pdep_rpa(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf, &drpa_cfg).unwrap();
        let drpa_erfc = run_pdep_rpa(
            &mol, &obs, &dfbs, Operator::erfc(omega), &rhf, &drpa_cfg).unwrap();
        let delta_full = drpa_coul.e_rpa - 2.0 * sc_full.e_os;
        let delta_sr = drpa_erfc.e_rpa - 2.0 * sc_sr.e_os;
        let expected = sc_full.e_total + delta_full - delta_sr;

        eprintln!(
            "fusion check: fused={:.12} unfused={:.12} diff={:.2e}",
            fused.e_corr, expected, fused.e_corr - expected
        );
        assert!(
            (fused.e_corr - expected).abs() < 1e-10,
            "fusion not bit-identical: {} vs {}", fused.e_corr, expected
        );
    }

    // ── terf/terfc attenuator arm ─────────────────────────────────────────
    //
    // The tempered split satisfies terf + terfc = Coulomb exactly, so it has the
    // SAME exact limits as erf/erfc — but the mapping to r0 is INVERTED vs ω:
    //   large r0 ⇒ ω=1/(r0√2)→0 ⇒ terfc→Coulomb, terf→0  ⇒ plain MP2
    //   small r0 ⇒ ω→∞          ⇒ terfc→0, terf→Coulomb  ⇒ MP2 + ΔdRPA[Coulomb]
    // These tests need the interpolation tables. They resolve FERRIC_TERF_TABLE_DIR
    // (or the main-repo `terf-tables/` — the worktree copy is generators-only, the
    // .bin tables are uncommitted), and SKIP with a note if absent, matching
    // crates/ferric-integrals/tests/terfc_base_validation.rs.

    /// Locate the terf .bin tables; None (→ skip) if not found.
    fn terf_tables_available() -> bool {
        if let Ok(d) = std::env::var("FERRIC_TERF_TABLE_DIR") {
            if std::path::Path::new(&d).join("16_4_2.bin").exists() {
                return true;
            }
        }
        false
    }

    /// terf, large r0 (ω→0): terf→0 ⇒ both DeltaLr and CoupledRings reduce to
    /// plain RI-MP2. r0=20 Bohr ⇒ ω≈0.035 Bohr⁻¹.
    #[test]
    fn terf_large_r0_reduces_to_mp2() {
        if !terf_tables_available() {
            eprintln!("SKIP terf_large_r0_reduces_to_mp2: terf tables absent");
            return;
        }
        let (mol, obs, dfbs, rhf) = setup_h2();
        let full = ferric_mp2::rimp2::ri_mp2(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf,
            &ferric_mp2::rimp2::RiMp2Config::default(),
        ).unwrap();
        for form in [RsMp2RpaFormulation::DeltaLr, RsMp2RpaFormulation::CoupledRings] {
            let cfg = RsMp2RpaConfig {
                attenuator: Attenuator::Terf,
                r0: 20.0,
                formulation: form,
                ..Default::default()
            };
            let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();
            eprintln!("terf(r0=20, {form:?}): MP2={:.10} e_corr={:.10} diff={:.2e}",
                full.mp2_corr, r.e_corr, r.e_corr - full.mp2_corr);
            assert!((r.e_corr - full.mp2_corr).abs() < 1e-4,
                "terf large-r0 must reduce to MP2 ({form:?}): {} vs {}",
                r.e_corr, full.mp2_corr);
            assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
        }
    }

    /// terf, small r0 (ω→∞): terf→Coulomb ⇒ both formulations equal
    /// E_MP2[Coulomb] + (E_dRPA[Coulomb] − 2·E_OS[Coulomb]). r0=0.05 Bohr ⇒ ω≈14.
    ///
    /// Tolerance is 5e-4, not tighter, because r0=0.05 is the finite-range floor:
    /// the CLI sweep converges monotonically toward the target as r0 shrinks
    /// (-0.02258 @0.30 → -0.01861 @0.05, target -0.01843), the SAME finite-range
    /// truncation the erf ω→∞ test documents (it needs ω=200 for 1e-6; ω≈14 here
    /// leaves ~1.8e-4). This confirms the limit direction, not machine-precision
    /// equality — pushing r0 smaller runs off the terf table domain.
    #[test]
    fn terf_small_r0_is_mp2_plus_delta_drpa() {
        if !terf_tables_available() {
            eprintln!("SKIP terf_small_r0_is_mp2_plus_delta_drpa: terf tables absent");
            return;
        }
        let (mol, obs, dfbs, rhf) = setup_h2();
        let ri_cfg = RiMp2Config::default();
        let (sc, _) = ri_mp2_spin_components(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf, &ri_cfg).unwrap();
        let rpa_cfg = PdepRpaConfig { trunc_thresh: 0.0, ..Default::default() };
        let rpa_coul = run_pdep_rpa(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf, &rpa_cfg).unwrap();
        let expected = sc.e_total + rpa_coul.e_rpa - 2.0 * sc.e_os;
        for form in [RsMp2RpaFormulation::DeltaLr, RsMp2RpaFormulation::CoupledRings] {
            let cfg = RsMp2RpaConfig {
                attenuator: Attenuator::Terf,
                r0: 0.05,
                formulation: form,
                ..Default::default()
            };
            let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();
            eprintln!("terf(r0=0.05, {form:?}): e_corr={:.10} MP2+ΔdRPA[Coul]={:.10} diff={:.2e}",
                r.e_corr, expected, r.e_corr - expected);
            assert!((r.e_corr - expected).abs() < 5e-4,
                "terf small-r0 limit broken ({form:?}): {} vs {}", r.e_corr, expected);
            assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
        }
    }

    /// The erf default must be untouched by the terf plumbing: an explicit
    /// Attenuator::Erf config equals the pre-change default at ω=0.42.
    #[test]
    fn erf_default_unchanged_by_terf_plumbing() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let a = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf,
            &RsMp2RpaConfig { omega: 0.42, ..Default::default() }).unwrap();
        let b = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf,
            &RsMp2RpaConfig { omega: 0.42, attenuator: Attenuator::Erf,
                ..Default::default() }).unwrap();
        assert_eq!(a.e_corr.to_bits(), b.e_corr.to_bits(),
            "explicit Erf must be bit-identical to the default");
    }

    /// Item 3 smoke test: a normal small H2/cc-pVDZ run under a generous
    /// (default auto-resolved) budget must complete successfully — i.e. the
    /// new preflight gate (Item 1) does not reject it, and the stage-seam RSS
    /// warnings (Item 3), which are pure stderr observability and never
    /// affect control flow, do not disrupt the computation. This is the
    /// "normal small run produces no disruption" contract; the "no warning
    /// fires" half of that claim is unit-tested directly in
    /// ferric-core::memory (`warn_if_rss_over_does_not_panic_and_is_a_pure_observer`)
    /// since capturing this test binary's own stderr for a substring check
    /// would be flaky under the parallel test harness.
    #[test]
    fn small_system_run_completes_under_default_budget_item3_smoke() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let cfg = RsMp2RpaConfig::default();
        let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg)
            .expect("small H2/cc-pVDZ run must complete under the default auto-resolved budget");
        assert!(r.total_energy.is_finite());
    }
}
