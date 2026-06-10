//! SR-MP2 + LR-RPA range-separated correlation (Δ-form B and coupled-rings T).
//! See docs/superpowers/specs/2026-06-09-sr-mp2-lr-rpa-design.md.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::ScfResult;

use crate::{run_pdep_rpa, PdepRpaConfig};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RsMp2RpaFormulation {
    /// Δ-form (formulation B): E_MP2[Coulomb] + (E_dRPA[erf] − 2·E_OS[erf]).
    /// Pure-LR rings only; mixed SR×LR rings dropped. Default.
    DeltaLr,
    /// Coupled-rings (formulation T): MP2 + ΔdRPA[Coulomb] − ΔdRPA[erfc].
    /// Adds all mixed SR×LR rings vs DeltaLr; no pure-SR rings ≥3rd order.
    CoupledRings,
}

impl Default for RsMp2RpaFormulation {
    fn default() -> Self {
        RsMp2RpaFormulation::DeltaLr
    }
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
    pub frozen_core: usize,
    /// dRPA solver knobs. Default forces trunc_thresh = 0.0 (full rank):
    /// this is an energy method; PDEP truncation is a production-size opt-in.
    ///
    /// Note: the nested `rpa.frozen_core` field is ignored — `cfg.frozen_core`
    /// is authoritative and overwrites it before each RPA call.
    pub rpa: PdepRpaConfig,
    /// Which algebraic formulation to use. Default is `DeltaLr` (formulation B),
    /// which preserves all existing behavior.
    pub formulation: RsMp2RpaFormulation,
}

impl Default for RsMp2RpaConfig {
    fn default() -> Self {
        Self {
            omega: 0.420 * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
            frozen_core: 0,
            rpa: PdepRpaConfig { trunc_thresh: 0.0, ..Default::default() },
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
#[derive(Debug)]
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
    let ri_cfg = RiMp2Config { frozen_core: cfg.frozen_core };
    let (sc_full, _) =
        ri_mp2_spin_components(mol, obs, dfbs, Operator::coulomb(), rhf, &ri_cfg)?;
    let (sc_sr, _) =
        ri_mp2_spin_components(mol, obs, dfbs, Operator::erfc(cfg.omega), rhf, &ri_cfg)?;
    let (sc_lr, _) =
        ri_mp2_spin_components(mol, obs, dfbs, Operator::erf(cfg.omega), rhf, &ri_cfg)?;

    // Closed shell: E_MP2 = Σ (ia|jb)[2(ia|jb)−(ib|ja)]/Δ; the direct (ring)
    // part 2Σ(ia|jb)²/Δ — the 2nd-order truncation of dRPA — equals 2·E_OS.
    let e_dmp2_lr = 2.0 * sc_lr.e_os;

    let mut rpa_cfg = cfg.rpa.clone();
    rpa_cfg.frozen_core = cfg.frozen_core;

    let (e_corr, e_drpa_lr, e_corr_naive, e_delta_drpa_full, e_delta_drpa_sr) =
        match cfg.formulation {
            RsMp2RpaFormulation::DeltaLr => {
                let rpa = run_pdep_rpa(mol, obs, dfbs, Operator::erf(cfg.omega), rhf, &rpa_cfg)?;
                let e_corr = sc_full.e_total + rpa.e_rpa - e_dmp2_lr;
                (
                    e_corr,
                    Some(rpa.e_rpa),
                    Some(sc_sr.e_total + rpa.e_rpa),
                    None,
                    None,
                )
            }
            RsMp2RpaFormulation::CoupledRings => {
                let rpa_coul =
                    run_pdep_rpa(mol, obs, dfbs, Operator::coulomb(), rhf, &rpa_cfg)?;
                let rpa_erfc =
                    run_pdep_rpa(mol, obs, dfbs, Operator::erfc(cfg.omega), rhf, &rpa_cfg)?;
                let delta_full = rpa_coul.e_rpa - 2.0 * sc_full.e_os;
                let delta_sr = rpa_erfc.e_rpa - 2.0 * sc_sr.e_os;
                // T: E_MP2[Coulomb] + ΔdRPA[Coulomb] − ΔdRPA[erfc]
                let e_corr = sc_full.e_total + delta_full - delta_sr;
                (e_corr, None, None, Some(delta_full), Some(delta_sr))
            }
        };

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
        let rpa_cfg = PdepRpaConfig { trunc_thresh: 0.0, ..Default::default() };
        let rpa_coul = run_pdep_rpa(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf, &rpa_cfg).unwrap();
        let expected = sc.e_total + rpa_coul.e_rpa - 2.0 * sc.e_os;
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
        let rpa_cfg = PdepRpaConfig { trunc_thresh: 0.0, ..Default::default() };
        let rpa_coul = run_pdep_rpa(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf, &rpa_cfg).unwrap();
        let expected = sc.e_total + rpa_coul.e_rpa - 2.0 * sc.e_os;
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
}
