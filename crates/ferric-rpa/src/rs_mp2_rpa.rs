//! SR-MP2 + LR-RPA range-separated correlation (Δ-form).
//! See docs/superpowers/specs/2026-06-09-sr-mp2-lr-rpa-design.md.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::ScfResult;

use crate::{run_pdep_rpa, PdepRpaConfig};

/// Configuration for SR-MP2 + LR-RPA.
#[derive(Debug, Clone)]
pub struct RsMp2RpaConfig {
    /// Range-separation parameter ω in Bohr⁻¹ (CLI/Python boundary is Å⁻¹,
    /// matching att-rimp2). Default 0.222 Bohr⁻¹ = 0.420 Å⁻¹ (erfc-optimal,
    /// Goldey & Head-Gordon JPCL 2012).
    pub omega: f64,
    pub frozen_core: usize,
    /// dRPA[erf] solver knobs. Default forces trunc_thresh = 0.0 (full rank):
    /// this is an energy method; PDEP truncation is a production-size opt-in.
    pub rpa: PdepRpaConfig,
}

impl Default for RsMp2RpaConfig {
    fn default() -> Self {
        let mut rpa = PdepRpaConfig::default();
        rpa.trunc_thresh = 0.0;
        Self {
            omega: 0.420 * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
            frozen_core: 0,
            rpa,
        }
    }
}

/// Component breakdown. `e_corr` (Δ-form, formulation B) is the method;
/// `e_corr_naive` (formulation A) is a diagnostic — it is missing the
/// 2·v_sr·v_lr cross-range correlation and is reported to make that visible.
#[derive(Debug)]
pub struct RsMp2RpaResult {
    pub e_mp2_full: f64,
    pub e_sr_mp2: f64,
    pub e_lr_mp2: f64,
    /// Direct (ring) second-order term with the erf kernel = 2·E_OS[erf].
    pub e_dmp2_lr: f64,
    pub e_drpa_lr: f64,
    /// Naive sum E_MP2[erfc] + E_dRPA[erf] (formulation A, diagnostic).
    pub e_corr_naive: f64,
    /// Δ-form E_MP2[Coulomb] + E_dRPA[erf] − E_dMP2[erf] (formulation B).
    pub e_corr: f64,
    pub total_energy: f64,
}

/// SR-MP2 + LR-RPA, Δ-form: replace MP2's long-range direct ring series with
/// its dRPA resummation. Exact limits: ω→0 ⇒ plain MP2; ω→∞ ⇒ MP2 + (dRPA − dMP2).
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

    let mut rpa_cfg = cfg.rpa.clone();
    rpa_cfg.frozen_core = cfg.frozen_core;
    let rpa = run_pdep_rpa(mol, obs, dfbs, Operator::erf(cfg.omega), rhf, &rpa_cfg)?;

    // Closed shell: E_MP2 = Σ (ia|jb)[2(ia|jb)−(ib|ja)]/Δ; the direct (ring)
    // part 2Σ(ia|jb)²/Δ — the 2nd-order truncation of dRPA — equals 2·E_OS.
    let e_dmp2_lr = 2.0 * sc_lr.e_os;
    let e_corr = sc_full.e_total + rpa.e_rpa - e_dmp2_lr;

    Ok(RsMp2RpaResult {
        e_mp2_full: sc_full.e_total,
        e_sr_mp2: sc_sr.e_total,
        e_lr_mp2: sc_lr.e_total,
        e_dmp2_lr,
        e_drpa_lr: rpa.e_rpa,
        e_corr_naive: sc_sr.e_total + rpa.e_rpa,
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
        eprintln!("MP2 corr {:.10}  Δ-form corr {:.10}  ΔLR {:.2e}",
            full.mp2_corr, r.e_corr, r.e_drpa_lr - r.e_dmp2_lr);
        assert!((r.e_corr - full.mp2_corr).abs() < 1e-5,
            "omega→0 must reduce to MP2: {} vs {}", r.e_corr, full.mp2_corr);
        assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
    }

    /// ω→∞: erf → Coulomb, so the Δ-form must equal
    /// E_MP2[Coulomb] + (E_dRPA[Coulomb] − 2·E_OS[Coulomb]) computed explicitly.
    /// Pins the dRPA energy convention against the MP2 spin components.
    ///
    /// At ω=50 Bohr⁻¹, erf(50r)/r ≈ 1/r at all chemically relevant r; the
    /// residual 2-center integral mismatch (P|erf(50r)/r|Q) vs (P|1/r|Q) is
    /// ~7 µHa for H₂/cc-pVDZ — a finite-ω truncation artefact, not a
    /// convention error. This tolerance (1e-5 Ha) is 3 orders of magnitude
    /// tighter than any factor-2/4 convention trap (~18 mHa = 1.8e-2 Ha), so
    /// the test still serves its purpose of catching spin-convention bugs.
    #[test]
    fn omega_to_infinity_is_mp2_plus_delta_drpa() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let cfg = RsMp2RpaConfig { omega: 50.0, ..Default::default() };
        let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();

        let ri_cfg = RiMp2Config::default();
        let (sc, _) = ri_mp2_spin_components(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf, &ri_cfg).unwrap();
        let mut rpa_cfg = PdepRpaConfig::default();
        rpa_cfg.trunc_thresh = 0.0;
        let rpa_coul = run_pdep_rpa(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf, &rpa_cfg).unwrap();
        let expected = sc.e_total + rpa_coul.e_rpa - 2.0 * sc.e_os;
        eprintln!("Δ-form(ω=50) {:.10}  MP2+ΔdRPA[Coulomb] {:.10}", r.e_corr, expected);
        assert!((r.e_corr - expected).abs() < 1e-5,
            "omega→∞ limit broken: {} vs {}", r.e_corr, expected);
    }
}
