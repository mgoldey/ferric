//! Spin-component scaled MP2 variants.
//!
//! - SCS-MP2: E = c_OS * E_OS + c_SS * E_SS (Grimme, JCP 2003)
//! - SCS-MP2(2terfc): dual-attenuated SCS (Goldey, Dutoi, Head-Gordon, PCCP 2013)
//!   E = c_OS * E_OS(r0_1) + c_SS * [E_SS(r0_2) - E_SS(r0_1)], using the EXACT
//!   `terfc` operator (2D interpolation tables), not the erfc approximation.

use crate::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;

/// Angstrom to Bohr conversion factor.
const ANGSTROM_TO_BOHR: f64 = 1.8897259886;


/// Standard SCS-MP2 configuration (Grimme, JCP 2003).
#[derive(Debug, Clone)]
pub struct ScsMp2Config {
    /// Opposite-spin scaling coefficient (Grimme default: 6/5).
    pub c_os: f64,
    /// Same-spin scaling coefficient (Grimme default: 1/3).
    pub c_ss: f64,
    /// Frozen core orbitals.
    pub frozen_core: usize,
    /// Optional resident-bytes ceiling for the 3-index MO transform, propagated
    /// into the internal `RiMp2Config`. `None` → resolved via
    /// [`ferric_core::memory::resolve_budget_bytes`].
    pub memory_budget_bytes: Option<usize>,
}

impl Default for ScsMp2Config {
    fn default() -> Self {
        Self { c_os: 6.0 / 5.0, c_ss: 1.0 / 3.0, frozen_core: 0, memory_budget_bytes: None }
    }
}



/// Result from SCS-MP2 or SCS-MP2(2terfc).
#[derive(Debug, Clone)]
#[must_use]
pub struct ScsMp2Result {
    /// Total energy: E_RHF + scs_corr.
    pub total_energy: f64,
    /// SCS correlation energy.
    pub scs_corr: f64,
    /// Opposite-spin component (possibly scaled/attenuated).
    pub e_os: f64,
    /// Same-spin component (possibly scaled/attenuated).
    pub e_ss: f64,
}

impl std::fmt::Display for ScsMp2Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SCS-MP2 total: {:.10} Ha (OS: {:.10}, SS: {:.10})",
            self.total_energy, self.e_os, self.e_ss)
    }
}

/// Standard SCS-MP2: E = c_OS * E_OS + c_SS * E_SS.
///
/// Uses full Coulomb operator (no attenuation). With c_OS=1, c_SS=1 this
/// recovers standard MP2.
pub fn scs_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &ScsMp2Config,
) -> Result<ScsMp2Result, FerricError> {
    let ri_config = RiMp2Config { frozen_core: config.frozen_core, memory_budget_bytes: config.memory_budget_bytes, ..Default::default() };
    let (sc, _) = ri_mp2_spin_components(mol, obs, dfbs, Operator::coulomb(), rhf, &ri_config)?;
    let scs_corr = config.c_os * sc.e_os + config.c_ss * sc.e_ss;
    Ok(ScsMp2Result {
        total_energy: rhf.energy + scs_corr,
        scs_corr,
        e_os: sc.e_os,
        e_ss: sc.e_ss,
    })
}

/// SCS-MP2(2terfc) configuration (Goldey, Dutoi, Head-Gordon, PCCP 2013; thesis Eq 5.6).
#[derive(Debug, Clone)]
pub struct ScsMp2TerfcConfig {
    /// Bonded attenuation distance r0(1) in Bohr.
    pub r0_bonded: f64,
    /// Non-bonded attenuation distance r0(2) in Bohr (must be > r0(1)).
    pub r0_nonbonded: f64,
    /// Opposite-spin scaling coefficient.
    pub c_os: f64,
    /// Same-spin scaling coefficient.
    pub c_ss: f64,
    /// Frozen core orbitals.
    pub frozen_core: usize,
    /// Optional resident-bytes ceiling for the 3-index MO transform, propagated
    /// into the internal `RiMp2Config`. `None` → resolved via
    /// [`ferric_core::memory::resolve_budget_bytes`].
    pub memory_budget_bytes: Option<usize>,
}

impl Default for ScsMp2TerfcConfig {
    /// The published **SCS-MP2(2terfc, aTZ)** parameterization — the NON-VV10
    /// variant.
    ///
    /// r₀(1) = 0.75 Å, r₀(2) = 1.05 Å, c_OS = 1.27, c_SS = 4.05, in
    /// **aug-cc-pVTZ**. Reported S66 RMSD 0.228 kcal/mol (Goldey/Belzunces/
    /// Head-Gordon, JCTC 11, 4159 (2015), Table 2; the parameters themselves are
    /// from the earlier Goldey/Dutoi/Head-Gordon PCCP 2013 work / thesis Eq 5.6).
    ///
    /// **Do NOT confuse these with SCS-MP2-V(2terfc, aTZ)**, the VV10-corrected
    /// variant, which is a genuinely different fit: r₀^SR = 0.70 Å,
    /// r₀^MR = 0.90 Å, b = 14.0, c_OS = 1.267, c_SS = 4.444 (S66 RMSD 0.194).
    /// The paper describes the VV10 variant's attenuation parameters as
    /// "contracted" relative to these — the difference is physical, not
    /// rounding, because the VV10 tail lets you attenuate harder. That variant
    /// belongs in `att_vv10.rs` (it carries a VV10 term); this function has no
    /// VV10 term at all.
    ///
    /// Like every attenuated parameterization in this repo these are
    /// **no-counterpoise**, frozen-core, aug-cc-pVTZ values; using them in
    /// another basis or with CP-corrected interaction energies is extrapolation.
    fn default() -> Self {
        Self {
            r0_bonded: 0.75 * ANGSTROM_TO_BOHR,
            r0_nonbonded: 1.05 * ANGSTROM_TO_BOHR,
            c_os: 1.27,
            c_ss: 4.05,
            frozen_core: 0,
            memory_budget_bytes: None,
        }
    }
}

/// SCS-MP2(2terfc): E = c_OS * E_OS(r0_1) + c_SS * [E_SS(r0_2) - E_SS(r0_1)].
///
/// Dual-attenuated SCS-MP2 from Goldey, Dutoi, Head-Gordon (PCCP 2013). Calls
/// `ri_mp2_spin_components` twice with the EXACT `terfc` operator at the bonded
/// (r0_1) and non-bonded (r0_2) attenuation distances. Unlike the deprecated
/// spike, this uses the interpolation-table terfc integrals, not an erfc fit.
pub fn scs_mp2_2terfc(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &ScsMp2TerfcConfig,
) -> Result<ScsMp2Result, FerricError> {
    // Config validation returns an error rather than panicking: r0_bonded /
    // r0_nonbonded are unvalidated user config, and the repo convention is that
    // bad config hard-ERRORS, never aborts the process (and never silently
    // proceeds to a meaningless number). The ordering matters physically — the
    // same-spin term is the DIFFERENCE E_SS(r0_MR) - E_SS(r0_SR), which changes
    // sign if the two are swapped.
    #[allow(clippy::nonminimal_bool)] // NaN-aware guard: !(x > 0 && finite) must REJECT NaN
    if !(config.r0_bonded > 0.0 && config.r0_bonded.is_finite())
        || !(config.r0_nonbonded > 0.0 && config.r0_nonbonded.is_finite())
    {
        return Err(FerricError::General(format!(
            "scs_mp2_2terfc: r0 values must be finite and > 0 (got r0(1)={} Bohr, \
             r0(2)={} Bohr)",
            config.r0_bonded, config.r0_nonbonded
        )));
    }
    if config.r0_nonbonded <= config.r0_bonded {
        return Err(FerricError::General(format!(
            "scs_mp2_2terfc: the midrange r0(2)={} Bohr must exceed the short-range \
             r0(1)={} Bohr — the same-spin term is the difference \
             E_SS(r0(2)) - E_SS(r0(1)) and inverts sign if these are swapped.",
            config.r0_nonbonded, config.r0_bonded
        )));
    }
    let ri_config = RiMp2Config { frozen_core: config.frozen_core, memory_budget_bytes: config.memory_budget_bytes, ..Default::default() };

    // Spin components at r0(1) (bonded, shorter range) via exact terfc.
    let (sc1, _) =
        ri_mp2_spin_components(mol, obs, dfbs, Operator::terfc(config.r0_bonded), rhf, &ri_config)?;
    // Spin components at r0(2) (non-bonded, longer range).
    let (sc2, _) = ri_mp2_spin_components(
        mol, obs, dfbs, Operator::terfc(config.r0_nonbonded), rhf, &ri_config,
    )?;

    // Thesis Eq 5.6: E = c_OS * E_OS(r0_1) + c_SS * [E_SS(r0_2) - E_SS(r0_1)].
    let e_ss = sc2.e_ss - sc1.e_ss;
    let scs_corr = config.c_os * sc1.e_os + config.c_ss * e_ss;
    Ok(ScsMp2Result {
        total_energy: rhf.energy + scs_corr,
        scs_corr,
        e_os: sc1.e_os,
        e_ss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    /// Water/cc-pVDZ — used by the 2terfc tests instead of H2 because **H2 has
    /// nocc = 1, so same-spin correlation is identically zero**: the
    /// antisymmetrized numerator K = (ia|jb) − (ib|ja) needs two distinct
    /// occupied orbitals. On H2 both E_SS(r0_SR) and E_SS(r0_MR) come out 0.000,
    /// so the entire c_SS·[E_SS(MR) − E_SS(SR)] half of Eq. 12 is untested and a
    /// mutation to it cannot fail. (Caught by mutation testing: an earlier
    /// revision of `scs_2terfc_matches_eq12_assembled_by_hand` used H2 and
    /// survived dropping the difference.) Water has nocc = 5 — a real same-spin
    /// space. Do not swap this back to H2.
    fn setup_h2o() -> (Molecule, PreparedBasis, PreparedBasis, ScfResult) {
        let xyz = "3\nwater\nO 0.000 0.000 0.118\nH 0.000 0.755 -0.471\nH 0.000 -0.755 -0.471\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol, &obs, op, &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        ).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        (mol, obs, dfbs, rhf)
    }

    fn setup_h2() -> (Molecule, PreparedBasis, PreparedBasis, ScfResult) {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol, &obs, op, &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        ).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        (mol, obs, dfbs, rhf)
    }

    #[test]
    fn test_scs_mp2_unit_scaling_equals_standard() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let full = crate::rimp2::ri_mp2(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf,
            &RiMp2Config::default(),
        ).unwrap();
        let scs = scs_mp2(
            &mol, &obs, &dfbs, &rhf,
            &ScsMp2Config { c_os: 1.0, c_ss: 1.0, frozen_core: 0, memory_budget_bytes: None },
        ).unwrap();
        eprintln!(
            "SCS (c_OS=1, c_SS=1) corr: {:.10}, standard RI-MP2 corr: {:.10}",
            scs.scs_corr, full.mp2_corr
        );
        assert!(
            (scs.scs_corr - full.mp2_corr).abs() < 1e-10,
            "SCS with c_OS=c_SS=1 ({}) should equal standard ({})",
            scs.scs_corr, full.mp2_corr
        );
    }

    // -----------------------------------------------------------------------
    // SCS-MP2(2terfc): published-parameter and formula validation
    //
    // `scs_mp2_2terfc` predates these tests and had NO coverage at all. It is
    // NOT reimplemented here — these tests validate what is already there.
    // -----------------------------------------------------------------------

    /// The defaults must be the published SCS-MP2(2terfc, aTZ) values, in the
    /// right units, and must NOT be the VV10-corrected variant's values.
    #[test]
    fn scs_2terfc_defaults_are_the_published_non_vv10_parameters() {
        let c = ScsMp2TerfcConfig::default();
        // r0 in Angstrom, recovered by dividing out the conversion.
        let r0_1_ang = c.r0_bonded / ANGSTROM_TO_BOHR;
        let r0_2_ang = c.r0_nonbonded / ANGSTROM_TO_BOHR;
        eprintln!(
            "SCS-MP2(2terfc, aTZ) defaults: r0(1)={r0_1_ang:.4} A, r0(2)={r0_2_ang:.4} A, \
             c_OS={}, c_SS={}",
            c.c_os, c.c_ss
        );
        assert!((r0_1_ang - 0.75).abs() < 1e-12, "r0(1) must be 0.75 A");
        assert!((r0_2_ang - 1.05).abs() < 1e-12, "r0(2) must be 1.05 A");
        assert_eq!(c.c_os, 1.27, "c_OS must be 1.27");
        assert_eq!(c.c_ss, 4.05, "c_SS must be 4.05");

        // ABSOLUTE Bohr anchors too. `r0_bonded / ANGSTROM_TO_BOHR` round-trips
        // through the same constant it was built with, so it would still pass if
        // that constant were inverted. (This exact blind spot was found by
        // mutation testing in att_vv10.rs.) 0.75 A = 1.4173 Bohr, not 0.397.
        assert!(
            (c.r0_bonded - 1.417_294_491_45).abs() < 1e-8,
            "0.75 A must be ~1.41729 Bohr, got {} (a value near 0.397 means the \
             Angstrom->Bohr conversion is inverted)",
            c.r0_bonded
        );
        assert!(
            (c.r0_nonbonded - 1.984_212_287_1).abs() < 1e-8,
            "1.05 A must be ~1.98421 Bohr, got {}",
            c.r0_nonbonded
        );

        // Must NOT be the VV10-corrected variant (r0 0.70/0.90, c 1.267/4.444).
        assert!(
            (r0_1_ang - 0.70).abs() > 1e-6 && (r0_2_ang - 0.90).abs() > 1e-6,
            "these are SCS-MP2-V(2terfc) parameters, not SCS-MP2(2terfc)"
        );
        assert_ne!(c.c_os, 1.267, "1.267 is the VV10-corrected c_OS");
        assert_ne!(c.c_ss, 4.444, "4.444 is the VV10-corrected c_SS");
        // Ordering invariant the formula depends on.
        assert!(c.r0_nonbonded > c.r0_bonded);
    }

    /// Paper Eq. 12: `E = c_OS·E_OS(r0_SR) + c_SS·[E_SS(r0_MR) − E_SS(r0_SR)]`.
    ///
    /// Verified by recomputing both terfc spin-component calls independently and
    /// reassembling the formula by hand, rather than trusting the implementation
    /// to agree with itself. Requires the terfc interpolation tables.
    #[test]
    fn scs_2terfc_matches_eq12_assembled_by_hand() {
        if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
            eprintln!("SKIP: FERRIC_TERF_TABLE_DIR not set; terfc tables unavailable");
            return;
        }
        let (mol, obs, dfbs, rhf) = setup_h2o();
        let cfg = ScsMp2TerfcConfig::default();
        let got = match scs_mp2_2terfc(&mol, &obs, &dfbs, &rhf, &cfg) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIP: terfc path unavailable at runtime: {e}");
                return;
            }
        };

        // Independent reassembly.
        let ri = RiMp2Config { frozen_core: cfg.frozen_core, ..Default::default() };
        let (sc_sr, _) =
            ri_mp2_spin_components(&mol, &obs, &dfbs, Operator::terfc(cfg.r0_bonded), &rhf, &ri)
                .unwrap();
        let (sc_mr, _) =
            ri_mp2_spin_components(&mol, &obs, &dfbs, Operator::terfc(cfg.r0_nonbonded), &rhf, &ri)
                .unwrap();
        let expect = cfg.c_os * sc_sr.e_os + cfg.c_ss * (sc_mr.e_ss - sc_sr.e_ss);

        eprintln!(
            "SCS-MP2(2terfc)/H2O/cc-pVDZ: E_OS(SR)={:.12}, E_SS(SR)={:.12}, E_SS(MR)={:.12}\n  \
             Eq12 by hand = {expect:.12}, implementation = {:.12}",
            sc_sr.e_os, sc_sr.e_ss, sc_mr.e_ss, got.scs_corr
        );
        // Guard against the H2 trap: if same-spin is ~0 the c_SS half of Eq. 12
        // is untested and every assertion below about it is vacuous.
        assert!(
            sc_sr.e_ss.abs() > 1e-4 && sc_mr.e_ss.abs() > 1e-4,
            "this test is meaningless unless same-spin correlation is real: \
             E_SS(SR)={}, E_SS(MR)={}",
            sc_sr.e_ss, sc_mr.e_ss
        );
        assert!(
            (got.scs_corr - expect).abs() < 1e-12,
            "implementation ({}) must equal Eq. 12 assembled by hand ({expect})",
            got.scs_corr
        );
        // The reported components must be the pieces Eq. 12 actually used: e_os
        // at the SHORT range, e_ss as the midrange-minus-short DIFFERENCE.
        assert!((got.e_os - sc_sr.e_os).abs() < 1e-12, "e_os must be E_OS(r0_SR)");
        assert!(
            (got.e_ss - (sc_mr.e_ss - sc_sr.e_ss)).abs() < 1e-12,
            "e_ss must be the difference E_SS(r0_MR) - E_SS(r0_SR), not a raw E_SS"
        );
        assert!((got.total_energy - (rhf.energy + got.scs_corr)).abs() < 1e-12);

        // DISCRIMINATION. The two assertions above compare against expressions
        // rebuilt from the same `sc_sr`/`sc_mr` this test just computed, so they
        // pin the arithmetic but not the STRUCTURE: a mutation that drops the
        // `- E_SS(r0_SR)` difference changes `got.e_ss` and `expect` together
        // and slips through. (Found by mutation testing — M10 survived the
        // checks above.) So assert the structure directly: the SS term must be
        // a genuine difference, numerically distinct from EITHER raw E_SS, and
        // the OS term must come from the short range, not the midrange.
        let raw_mr = sc_mr.e_ss;
        let raw_sr = sc_sr.e_ss;
        eprintln!(
            "  structure check: e_ss(diff)={:.12} vs raw E_SS(MR)={raw_mr:.12} vs raw \
             E_SS(SR)={raw_sr:.12}",
            got.e_ss
        );
        assert!(
            (got.e_ss - raw_mr).abs() > 1e-9,
            "e_ss ({}) must NOT be the raw midrange E_SS ({raw_mr}) — the Eq. 12 \
             difference has been dropped",
            got.e_ss
        );
        assert!(
            (got.e_ss - raw_sr).abs() > 1e-9,
            "e_ss ({}) must NOT be the raw short-range E_SS ({raw_sr})",
            got.e_ss
        );
        assert!(
            (got.e_os - sc_mr.e_os).abs() > 1e-9,
            "e_os ({}) must come from the SHORT range, not the midrange ({})",
            got.e_os,
            sc_mr.e_os
        );
        // The attenuated same-spin difference is small and positive-ish relative
        // to the raw pieces (longer range keeps more correlation, so
        // E_SS(MR) is more negative => the difference is negative). Pin the sign
        // so a swapped subtraction is caught even if magnitudes coincide.
        assert!(
            got.e_ss < 0.0,
            "E_SS(r0_MR) - E_SS(r0_SR) must be negative (the longer range retains \
             more same-spin correlation), got {}",
            got.e_ss
        );
    }

    /// Bad r0 config must ERROR, not panic and not silently produce a number.
    /// The swapped-r0 case matters physically: the same-spin term is a
    /// difference and inverts sign if r0(1) and r0(2) are exchanged.
    #[test]
    fn scs_2terfc_rejects_bad_r0_config() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let base = ScsMp2TerfcConfig::default();

        let swapped = ScsMp2TerfcConfig {
            r0_bonded: base.r0_nonbonded,
            r0_nonbonded: base.r0_bonded,
            ..base.clone()
        };
        let err = scs_mp2_2terfc(&mol, &obs, &dfbs, &rhf, &swapped)
            .expect_err("swapped r0 ordering must be rejected");
        eprintln!("swapped-r0 rejection: {err}");

        for bad in [0.0_f64, -1.0, f64::NAN] {
            let cfg = ScsMp2TerfcConfig { r0_bonded: bad, ..base.clone() };
            assert!(
                scs_mp2_2terfc(&mol, &obs, &dfbs, &rhf, &cfg).is_err(),
                "r0(1)={bad} must be rejected"
            );
        }
    }
}
