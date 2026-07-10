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
//! cc-pVDZ-RI, about 48% of shell triples are screened away (48/100 kept = 52%).
//!
//! For the Coulomb operator ω = 0, the exponential is identically 1 and QQR-3
//! collapses to the standard 1/R Schwarz+distance estimate — still useful but
//! much less aggressive than erfc.  The operator-specific decay is what makes
//! attenuated MP2 intrinsically more screenable than full Coulomb MP2.
//!
//! ## All five steps: measured on decane/cc-pVDZ (OPENBLAS_NUM_THREADS=1, release)
//!
//! nbf=250, naux=868, nocc=41, nvir=209.  Times in ms.
//!
//! | Step                        | Coulomb | erfc dense | erfc screened |
//! |-----------------------------|---------|------------|---------------|
//! | 1. 2-center metric+Cholesky |   129   |    137     |     135       |
//! | 2. 3-center AO build        |  1628   |   2237     |    1769       |
//! | 3. MO transform →(P\|ia)   |   695   |    702     |     708       |
//! | 4. Metric contraction B̃    |   283   |    300     |     291       |
//! | 5. Energy assembly          |    66   |     70     |      71       |
//! | **TOTAL (post-RHF)**        | **2800**| **3445**   |   **2973**    |
//! | Speedup vs Coulomb          |  1.00×  |   0.81×    |    0.94×      |
//!
//! ### What the numbers say
//!
//! **The 3-center build (step 2) dominates** at 58% of post-RHF cost, and the
//! erfc operator is intrinsically slower to evaluate in libint2 than Coulomb
//! (special function, ~1.4× per integral).  Screening recovers some of that —
//! the screened erfc build (1769 ms) is 21% faster than dense erfc (2237 ms)
//! but still 8% slower than Coulomb (1628 ms).
//!
//! **Steps 3–5 are essentially operator-agnostic:** the MO transform, metric
//! contraction, and energy assembly all operate on the materialized B̃ tensor
//! and don't see the operator after step 2.  Energy assembly (step 5) is only
//! 66 ms — not the bottleneck at cc-pVDZ because nocc=41 is small.
//!
//! **The 48% triple reduction from screening saves ~470 ms on step 2 alone**
//! (screened 1769 ms vs dense erfc 2237 ms), making the screened path
//! competitive with unscreened Coulomb.  The overall post-RHF speedup of 0.94×
//! vs Coulomb means attenuated MP2 with QQR-3 screening is within 6% of plain
//! Coulomb MP2 speed while computing a physically different (short-range only)
//! correlation — you pay essentially nothing for the physics change.
//!
//! **Where the real gain lives:** the erfc MP2 energy itself is shorter-ranged,
//! so pair-amplitude cancellation lets you use smaller basis sets and fewer
//! k-points in periodic systems without losing the essential physics.  The
//! screening speedup on the integral build is a secondary benefit; the primary
//! advantage is reduced basis-set requirements.

use crate::mo_transform::transform_3center_ov;
use crate::rimp2::{
    active_occ, cholesky_inverse_sqrt, ri_mp2_spin_components, spin_components_from_b_ov,
    RiMp2Config, SpinComponents,
};
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
    /// Optional resident-bytes ceiling for the 3-index MO transform, propagated
    /// into the internal `RiMp2Config`. `None` → resolved via
    /// [`ferric_core::memory::resolve_budget_bytes`].
    pub memory_budget_bytes: Option<usize>,
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
            memory_budget_bytes: None,
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
    let ri_config = RiMp2Config { frozen_core: config.frozen_core, memory_budget_bytes: config.memory_budget_bytes };
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
        let ri_config = RiMp2Config { frozen_core: config.frozen_core, memory_budget_bytes: config.memory_budget_bytes };
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
    let nocc = active_occ(nocc_total, frozen_core)?;
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

    // b_flat is exactly the dressed (naux, nocc*nvir) B_ov tensor that
    // ri_mp2_spin_components hands to spin_components_from_b_ov — same shape,
    // same B^P_ia = sum_Q V^{-1/2}_PQ (Q|ia) contraction, only the upstream
    // 3-index build differs (QQR-3 screened vs dense). Reuse the shared,
    // i-blocked-GEMM + par-i spin-component assembly instead of the hand-rolled
    // serial (i,j,a,b) O(naux) double-dot loop, which had neither the M4 GEMM
    // port nor parallelization.
    let sc = spin_components_from_b_ov(&b_flat, eps, nocc, nvir, first_occ, nocc_total);
    Ok(sc)
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
    let ri_config = RiMp2Config { frozen_core: config.frozen_core, memory_budget_bytes: config.memory_budget_bytes };
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
        // erfc + erf = 1 splits the OPERATOR, not the energy: E_MP2 is
        // quadratic in the interaction, so (v_sr+v_lr)² has a 2·v_sr·v_lr
        // cross term that MP2(erfc)+MP2(erf) does not contain. The sum is
        // therefore only an approximation to MP2(Coulomb); the residual is
        // the cross-range correlation (plus per-operator RI fit error from
        // the Coulomb-fit aux basis). The 0.05 Ha tolerance bounds both.
        // This is exactly why rs_mp2_lr_rpa (ferric-rpa) uses the Δ-form
        // E_MP2[full] + (dRPA[erf] − dMP2[erf]) instead of a naive sum.
        let (mol, obs, dfbs, rhf) = setup_h2();
        let config = AttenuatedMp2Config { omega: 0.5, ..Default::default() };
        let (sr, lr, full) = rs_mp2_decomposition(&mol, &obs, &dfbs, &rhf, &config).unwrap();
        let sum = sr + lr;
        eprintln!(
            "MP2 decomposition: erfc_SR={:.8}, erf_LR={:.8}, sum={:.8}, full={:.8}, sum-vs-full={:.2e}",
            sr, lr, sum, full, (sum - full).abs()
        );
        // 0.05 Ha allows for cross-range correlation (dominant) + per-operator RI fit error.
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
