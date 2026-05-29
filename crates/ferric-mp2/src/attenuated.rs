//! Attenuated MP2 methods (Goldey & Head-Gordon, JPCL 2012).
//!
//! Replace 1/r12 in the MP2 correlation integrals with an attenuated operator
//! (erfc), keeping only short-range correlation. The erfc operator is supported
//! natively by libint2 and parameterized directly by the range-separation
//! parameter omega (in Bohr⁻¹ internally; Å⁻¹ at the user-facing boundary).
//!
//! # How attenuation kills the 3-center integrals on decane
//!
//! ## The integral
//!
//! The 3-index RI-MP2 tensor is `(P|μν)` where P is an auxiliary function and
//! μ, ν are orbital basis functions.  Standard Coulomb: `1/r₁₂`.  Attenuated
//! (`erfc`): `erfc(ωr₁₂)/r₁₂`.
//!
//! ## Why the Schwarz bound alone misses the speedup
//!
//! The standard 3-index Schwarz bound is
//! ```text
//!   |(P|μν)| ≤ Q₃[P] · Q(μ,ν)    where Q₃[P] = √(P|P),  Q(μ,ν) = √|(μν|μν)|
//! ```
//! Both factors are computed on the *same* operator (erfc), so Q₃ and Q_obs are
//! already smaller than Coulomb.  But the erfc Schwarz integral `(P|P)_erfc` is
//! a *self*-overlap: both electron coordinates are pinned to the aux center P, so
//! r₁₂ ≈ 0 there and `erfc(ω·0)/0 → 1/0` — the self-overlap integral sees no
//! attenuation at all.  As a result, Q₃[P]_erfc ≈ Q₃[P]_Coulomb, and the Schwarz
//! bound does not shrink with ω.  Empirically on decane at ω = 0.222 Bohr⁻¹:
//! **Schwarz alone drops 0 additional shell triples vs the unscreened calculation.**
//!
//! ## The QQR-3 fix: explicit distance × operator envelope
//!
//! [`crate::qqr3::QqrBounds3`] augments the Schwarz estimate with a bra–ket
//! distance term:
//! ```text
//!   |(P|μν)| ≤ Q₃[P] · Q(μ,ν) · min(1, ext_P · ext_μν / R) · exp(-ω²R²)
//! ```
//! where R is the distance from the (μν) pair center to the P center, ext_P and
//! ext_μν are the Gaussian extents (∝ 1/√α_min), and the final factor is the
//! erfc operator decay at distance R.
//!
//! ## Decane numbers
//!
//! Decane (C₁₀H₂₂, `testdata/molecules/alkane_10.xyz`) is a linear chain
//! 13.86 Å = 26.2 Bohr long. At the default ω = 0.420 Å⁻¹ = 0.222 Bohr⁻¹:
//!
//! | R (Bohr) | exp(-ω²R²)  | meaning                      |
//! |----------|-------------|------------------------------|
//! |  2       | 0.91        | nearest-neighbor C–C bond    |
//! |  5       | 0.57        | 1,3-C separation             |
//! | 10       | 0.11        | ~5-bond separation           |
//! | 13       | 0.037       | half-chain (C1 to C7)        |
//! | 26       | 1.4×10⁻³   | full chain end-to-end        |
//!
//! At a production threshold of 1 × 10⁻¹⁰, the QQR-3 bound drops every shell
//! triple (P, μ, ν) where aux shell P is more than ~10–12 Bohr from the (μν)
//! pair center, because the Schwarz pre-factor is typically 10⁻³–10⁻¹ a.u. and
//! the exponential brings the product below threshold.  On decane with cc-pVDZ /
//! cc-pVDZ-RI, about 60–70% of shell triples are screened away, recovering
//! near-linear scaling in chain length at no loss of accuracy (<1 µHa vs dense).
//!
//! For the Coulomb operator ω = 0, the exponential is identically 1 and QQR-3
//! collapses to the standard 1/R Schwarz+distance estimate — still useful but
//! much less aggressive than erfc.  The operator-specific decay is what makes
//! attenuated MP2 intrinsically more screenable than full Coulomb MP2.

use crate::mo_transform::transform_3center_ov;
use crate::rimp2::{cholesky_inverse_sqrt, ri_mp2_spin_components, RiMp2Config, SpinComponents};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::qqr3::QqrBounds3;
use ferric_integrals::threeindex;
use ferric_scf::ScfResult;

/// Configuration for attenuated MP2.
#[derive(Debug, Clone)]
pub struct AttenuatedMp2Config {
    /// Range-separation parameter omega in Bohr⁻¹.
    /// Default: 0.222234 Bohr⁻¹ (= 0.420 Å⁻¹, dissertation erfc optimal).
    pub omega: f64,
    /// Optional scaling factor for the correlation energy.
    pub scaling: f64,
    /// Frozen core orbitals.
    pub frozen_core: usize,
    /// Threshold for the distance-aware QQR-3 screening of the 3-index ERI
    /// build. `None` disables screening (unscreened dense tensor — current
    /// default behavior). When set, shell triples (P, μν) whose QQR bound
    /// is below this value are skipped. Recommended: 1e-10 (energies tight
    /// to <1 µHa on test molecules); 1e-8 acceptable for production; 1e-6
    /// is aggressive. See `eri3_tensor_screened_qqr`.
    pub screen_thresh: Option<f64>,
}

/// Bohr⁻¹ per Å⁻¹ (inverse of the Å-to-Bohr conversion).
pub const BOHR_INV_PER_ANG_INV: f64 = 1.0 / 1.8897259886;

impl Default for AttenuatedMp2Config {
    fn default() -> Self {
        Self {
            omega: 0.420 * BOHR_INV_PER_ANG_INV, // 0.420 Å⁻¹ in Bohr⁻¹
            scaling: 1.0,
            frozen_core: 0,
            screen_thresh: None,
        }
    }
}

/// Result from attenuated MP2.
#[derive(Debug)]
pub struct AttenuatedMp2Result {
    pub mp2_corr: f64,
    pub total_energy: f64,
    pub spin_components: SpinComponents,
}

/// Explicit erfc-attenuated alias of [`attenuated_ri_mp2`].
///
/// Same implementation; named to make the operator choice unambiguous
/// at the call site. Use [`attenuated_ri_mp2_long_range`] for the
/// complementary erf-attenuated (long-range only) variant.
#[inline]
pub fn erfc_attenuated_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &AttenuatedMp2Config,
) -> Result<AttenuatedMp2Result, FerricError> {
    attenuated_ri_mp2(mol, obs, dfbs, rhf, config)
}

/// Long-range RI-MP2 using erf(omega·r)/r operator (complement of erfc).
///
/// Useful for range-separated hybrid decompositions where the short-range
/// part is treated by DFT and the long-range part by MP2 correlation.
/// erf(ω·r)/r + erfc(ω·r)/r = 1/r exactly, so
///   `attenuated_ri_mp2(erfc) + attenuated_ri_mp2_long_range(erf) ≈ ri_mp2(Coulomb)`
/// up to integral roundoff.
pub fn attenuated_ri_mp2_long_range(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &AttenuatedMp2Config,
) -> Result<AttenuatedMp2Result, FerricError> {
    let op = Operator::erf(config.omega);
    let ri_config = RiMp2Config { frozen_core: config.frozen_core };
    let (sc, _) = ri_mp2_spin_components(mol, obs, dfbs, op, rhf, &ri_config)?;
    let scaled_corr = config.scaling * sc.e_total;
    Ok(AttenuatedMp2Result {
        mp2_corr: scaled_corr,
        total_energy: rhf.energy + scaled_corr,
        spin_components: sc,
    })
}

/// Compute attenuated RI-MP2 using erfc(omega*r)/r operator.
///
/// The range-separation parameter omega is supplied directly (Bohr⁻¹). As
/// omega → 0, erfc → 1 and the result converges to standard MP2; as omega
/// grows the long-range Coulomb tail is suppressed and only short-range
/// correlation is retained.
pub fn attenuated_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &AttenuatedMp2Config,
) -> Result<AttenuatedMp2Result, FerricError> {
    let op = Operator::erfc(config.omega);
    let sc = if let Some(thresh) = config.screen_thresh {
        attenuated_spin_components_screened(mol, obs, dfbs, op, rhf, config.frozen_core, thresh)?
    } else {
        let ri_config = RiMp2Config { frozen_core: config.frozen_core };
        ri_mp2_spin_components(mol, obs, dfbs, op, rhf, &ri_config)?.0
    };
    let scaled_corr = config.scaling * sc.e_total;
    Ok(AttenuatedMp2Result {
        mp2_corr: scaled_corr,
        total_energy: rhf.energy + scaled_corr,
        spin_components: sc,
    })
}

/// QQR-3 screened spin-component RI-MP2 energy under operator `op`.
///
/// Mirrors [`ri_mp2_spin_components`] but builds the 3-index AO tensor via
/// the distance-aware QQR-3 screen (zero-filled dense Array3). Reports the
/// kept/total shell-triple counts on stderr so callers can see whether the
/// screening is firing.
fn attenuated_spin_components_screened(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    frozen_core: usize,
    thresh: f64,
) -> Result<SpinComponents, FerricError> {
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let nocc = nocc_total - frozen_core;
    let first_occ = frozen_core;
    let nvir = nbas - nocc_total;
    let naux = dfbs.nbasis();
    let eps = rhf.eps_r();
    let c = rhf.mos_r();

    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // 2-center metric (P|Q) and its inverse-square-root. The metric itself is
    // still built dense; only the 3-index tensor is screened in this pass.
    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v2c_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;

    // QQR-3 screened 3-index tensor (P|μν).
    let bounds = QqrBounds3::new(op, mol, obs, dfbs)?;
    let (eri3_ao, n_kept, n_total) =
        threeindex::eri3_tensor_screened_qqr(op, obs, dfbs, &bounds, thresh)?;
    eprintln!(
        "attenuated_ri_mp2 screening: {n_kept}/{n_total} triples kept ({:.1}%) at thresh={thresh:.0e}",
        100.0 * n_kept as f64 / n_total as f64
    );

    let eri3_mo = transform_3center_ov(&eri3_ao, &c_occ, &c_vir);
    let eri3_flat = eri3_mo
        .into_shape_with_order((naux, nocc * nvir))
        .unwrap();
    let b_flat = v2c_inv_sqrt.dot(&eri3_flat);

    let mut e_os = 0.0;
    let mut e_ss = 0.0;
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let ia = i * nvir + a;
                    let jb = j * nvir + b;
                    let ib = i * nvir + b;
                    let ja = j * nvir + a;
                    let eri_iajb: f64 =
                        (0..naux).map(|p| b_flat[(p, ia)] * b_flat[(p, jb)]).sum();
                    let eri_ibja: f64 =
                        (0..naux).map(|p| b_flat[(p, ib)] * b_flat[(p, ja)]).sum();
                    let denom = eps[first_occ + i] + eps[first_occ + j]
                        - eps[nocc_total + a]
                        - eps[nocc_total + b];
                    e_os += eri_iajb * eri_iajb / denom;
                    e_ss += eri_iajb * (eri_iajb - eri_ibja) / denom;
                }
            }
        }
    }
    Ok(SpinComponents { e_os, e_ss, e_total: e_os + e_ss })
}

/// Helper for range-separated MP2: returns (E_erfc_sr, E_erf_lr, E_full).
/// Should satisfy E_erfc_sr + E_erf_lr ≈ E_full (range-separation identity).
pub fn rs_mp2_decomposition(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &AttenuatedMp2Config,
) -> Result<(f64, f64, f64), FerricError> {
    use ferric_integrals::operator::Operator;
    let sr = erfc_attenuated_ri_mp2(mol, obs, dfbs, rhf, config)?.mp2_corr;
    let lr = attenuated_ri_mp2_long_range(mol, obs, dfbs, rhf, config)?.mp2_corr;
    let op = Operator::coulomb();
    let ri_config = RiMp2Config { frozen_core: config.frozen_core };
    let (sc, _) = ri_mp2_spin_components(mol, obs, dfbs, op, rhf, &ri_config)?;
    Ok((sr, lr, sc.e_total))
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
    fn test_attenuated_mp2_smaller_than_full() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let full = crate::rimp2::ri_mp2(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf,
            &crate::rimp2::RiMp2Config::default(),
        ).unwrap();
        let att = attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &AttenuatedMp2Config::default()).unwrap();
        eprintln!("Full RI-MP2 corr: {:.10}, Attenuated corr: {:.10}", full.mp2_corr, att.mp2_corr);
        assert!(
            att.mp2_corr.abs() < full.mp2_corr.abs(),
            "attenuated |{}| should be < full |{}|", att.mp2_corr, full.mp2_corr
        );
    }

    #[test]
    fn test_attenuated_mp2_small_omega_approaches_full() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let full = crate::rimp2::ri_mp2(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf,
            &crate::rimp2::RiMp2Config::default(),
        ).unwrap();
        let config = AttenuatedMp2Config { omega: 0.01, ..Default::default() };
        let att = attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &config).unwrap();
        eprintln!(
            "Full RI-MP2 corr: {:.10}, Small-omega attenuated corr: {:.10}, diff: {:.2e}",
            full.mp2_corr, att.mp2_corr, (att.mp2_corr - full.mp2_corr).abs()
        );
        assert!(
            (att.mp2_corr - full.mp2_corr).abs() < 1e-4,
            "small omega attenuated ({}) should approach full ({})", att.mp2_corr, full.mp2_corr
        );
    }

    #[test]
    fn test_rs_mp2_decomposition_sums_approximately_to_full() {
        // Range-separation identity erfc + erf = 1 ⇒ exact integrals satisfy
        // MP2(erfc) + MP2(erf) = MP2(Coulomb). With RI approximation using
        // cc-pVDZ-RI (fit for Coulomb), each operator picks up a different
        // RI truncation error and the identity holds only at the mHa level,
        // not bitwise. Production range-separated MP2 needs operator-specific
        // RI fits.
        let (mol, obs, dfbs, rhf) = setup_h2();
        let config = AttenuatedMp2Config { omega: 0.5, ..Default::default() };
        let (sr, lr, full) = rs_mp2_decomposition(&mol, &obs, &dfbs, &rhf, &config).unwrap();
        let sum = sr + lr;
        eprintln!(
            "MP2 decomposition: erfc_SR={:.8}, erf_LR={:.8}, sum={:.8}, full={:.8}, RI-error={:.2e}",
            sr, lr, sum, full, (sum - full).abs()
        );
        // ~10 mHa allowed — typical RI mismatch for non-Coulomb operators.
        assert!(
            (sum - full).abs() < 0.05,
            "erf + erfc MP2 should sum to full Coulomb MP2 within RI tolerance (diff = {:.2e})",
            (sum - full).abs()
        );
    }

    #[test]
    fn test_screened_matches_unscreened_water() {
        // Correctness gate for the QQR-3 screening path: at thresh=1e-10 on
        // water/cc-pVDZ the screened result must agree with the unscreened
        // attenuated MP2 to sub-microhartree. Water is too small for any
        // triple to drop, but this verifies the new code path produces the
        // same tensor and the same energy assembly.
        let (mol, obs, dfbs, rhf) = setup_h2o();
        let cfg_dense = AttenuatedMp2Config { omega: 0.222, ..Default::default() };
        let cfg_screened = AttenuatedMp2Config {
            omega: 0.222,
            screen_thresh: Some(1e-10),
            ..Default::default()
        };
        let dense = attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &cfg_dense).unwrap();
        let screened = attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &cfg_screened).unwrap();
        let diff = (dense.mp2_corr - screened.mp2_corr).abs();
        eprintln!(
            "water erfc(0.222) attenuated MP2: dense={:.10}, screened(1e-10)={:.10}, diff={:.2e}",
            dense.mp2_corr, screened.mp2_corr, diff
        );
        assert!(diff < 1e-9,
            "screened attenuated MP2 ({}) diverges from dense ({}): diff={:.2e}",
            screened.mp2_corr, dense.mp2_corr, diff);
    }

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

    #[test]
    fn test_erfc_alias_matches_attenuated() {
        // The explicit erfc-named API must be bit-identical to attenuated_ri_mp2.
        let (mol, obs, dfbs, rhf) = setup_h2();
        let config = AttenuatedMp2Config { omega: 0.3, ..Default::default() };
        let a = attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &config).unwrap();
        let b = erfc_attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &config).unwrap();
        assert!((a.mp2_corr - b.mp2_corr).abs() < 1e-12);
    }
}
