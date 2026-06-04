//! SINGLE-MOLECULE SANITY PROBE (preflight, not a CI assertion).
//!
//! Hypothesis under test: attenuating the Coulomb response kernel
//! (erfc(ωr)/r in the (A+B) dielectric) moves RPA@HF static α and molecular
//! C6 toward reference (CRC α, DOSD C6), at potentially lower cost.
//!
//! This is lever (1): swap `Operator::coulomb()` → `Operator::erfc(ω)` in the
//! existing α/C6 path. SCF stays full-Coulomb; only the response is attenuated.
//!
//! Reference values for water:
//!   * CRC static α_iso  ≈ 9.8  a.u. (1.45 Å³)
//!   * DOSD molecular C6 ≈ 45.4 a.u.
//!
//! Run with:  cargo test -p ferric-rpa --test attenuated_alpha_c6_probe -- --nocapture --ignored
//! Ignored by default so it never gates CI (it prints, it does not assert physics).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{casimir_polder_c6, pdep_dynamic_polarizability, DispersionPartition};
use ferric_rpa::properties::pdep_polarizability_static;
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

#[test]
#[ignore]
fn attenuated_water_alpha_c6_probe() {
    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();

    // SCF is ALWAYS full Coulomb — we attenuate only the response.
    let scf_op = Operator::coulomb();
    let scf_bounds = SchwarzBounds::compute(scf_op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, scf_op, &scf_bounds, &RhfConfig::default()).unwrap();

    let cfg = PdepRpaConfig::default();

    // ω in Bohr⁻¹ (library convention). 0 = Coulomb baseline, then a small sweep.
    // CLI default attenuation for att-rimp2 is 0.420 Å⁻¹ ≈ 0.222 Bohr⁻¹.
    let omegas_bohr: &[f64] = &[0.0, 0.1, 0.2, 0.3, 0.5, 1.0];

    println!("\n=== water aug-cc-pVDZ : attenuated RPA@HF α and C6 ===");
    println!("  ref: CRC α_iso ≈ 9.8 a.u. ;  DOSD molecular C6 ≈ 45.4 a.u.\n");
    println!(
        "  {:>8}  {:>12}  {:>14}",
        "ω(Bohr⁻¹)", "α_iso(a.u.)", "C6_mol(a.u.)"
    );

    for &w in omegas_bohr {
        let op = if w == 0.0 {
            Operator::coulomb()
        } else {
            Operator::erfc(w)
        };

        let alpha = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, op, &cfg).unwrap();

        let dp = pdep_dynamic_polarizability(
            &mol,
            &obs,
            &obs_bs,
            &dfbs,
            &rhf,
            op,
            &cfg,
            DispersionPartition::Becke,
            None,
        )
        .unwrap();
        let c6 = casimir_polder_c6(&dp);

        println!(
            "  {:>8.3}  {:>12.4}  {:>14.4}",
            w, alpha.iso, c6.c6_molecular_iso
        );
    }
    println!();
}

/// CLINCHER: attenuation sweep on RPA@PBE — a GOOD-baseline correlated method.
///
/// RPA@HF C6 is ~−46% (broken baseline); attenuation moving it "up toward DOSD"
/// is consistent with error cancellation. RPA@PBE is ~3× better (memory:
/// rpa-vs-ts-c6-molecular, ~−15% on C6). If attenuation is real physics it
/// should NOT systematically overshoot a good baseline; if it's pure
/// over-correction it will push the already-decent C6 PAST DOSD as ω rises.
///
/// Run: cargo test --release -p ferric-rpa --test attenuated_alpha_c6_probe \
///        attenuated_water_rpa_pbe_c6_probe -- --ignored --nocapture
#[test]
#[ignore]
fn attenuated_water_rpa_pbe_c6_probe() {
    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();

    // PBE reference SCF (RI-J/RI-K with def2-universal-jkfit, per CLI convention).
    let scf_op = Operator::coulomb();
    let scf_bounds = SchwarzBounds::compute(scf_op, &obs).unwrap();
    let scf_cfg = RhfConfig {
        energy_conv: 1e-9,
        xc: Some("PBE".to_string()),
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        df_k_aux: Some("def2-universal-jkfit".to_string()),
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, &mol, &obs, scf_op, &scf_bounds, &scf_cfg).unwrap();
    assert!(rhf.converged, "RPA@PBE reference SCF did not converge");

    let cfg = PdepRpaConfig::default();
    // Wider ω range to catch a possible OVERSHOOT past DOSD (the clincher).
    let omegas_bohr: &[f64] = &[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0];

    println!("\n=== water aug-cc-pVDZ : attenuated RPA@PBE α and C6 ===");
    println!("  ref: CRC α_iso ≈ 9.8 a.u. ;  DOSD molecular C6 ≈ 45.4 a.u.");
    println!("  (RPA@PBE is a GOOD baseline ~−15%; watch for overshoot past 45.4)\n");
    println!("  {:>8}  {:>12}  {:>14}  {:>10}", "ω(Bohr⁻¹)", "α_iso(a.u.)", "C6_mol(a.u.)", "C6 err%");

    for &w in omegas_bohr {
        let op = if w == 0.0 { Operator::coulomb() } else { Operator::erfc(w) };
        let alpha = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, op, &cfg).unwrap();
        let dp = pdep_dynamic_polarizability(
            &mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg, DispersionPartition::Becke, None,
        )
        .unwrap();
        let c6 = casimir_polder_c6(&dp);
        let err = 100.0 * (c6.c6_molecular_iso - 45.4) / 45.4;
        println!(
            "  {:>8.3}  {:>12.4}  {:>14.4}  {:>+9.2}",
            w, alpha.iso, c6.c6_molecular_iso, err
        );
    }
    println!();
}

#[test]
#[ignore]
fn rpa_static_alpha_ccpvdz_water_for_mp2_compare() {
    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let cfg = PdepRpaConfig::default();
    let a = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, op, &cfg).unwrap();
    println!("\n[compare] RPA static α_iso (cc-pVDZ water) = {:.5} a.u.; tensor diag = {:.4} {:.4} {:.4}\n",
        a.iso, a.tensor[0][0], a.tensor[1][1], a.tensor[2][2]);
}

/// DIAGNOSTIC: does attenuation shrink the ideal response (PDEP) basis?
///
/// Counts significant dielectric eigenmodes (the response rank) as a function
/// of the attenuation ω. If attenuation shrinks the ideal basis, `rank` should
/// DROP as ω rises (shorter-ranged screening → fewer significant modes), while
/// |trace_log| (total screening strength) also shrinks. Ignored; prints a table.
#[test]
#[ignore]
fn pdep_rank_vs_attenuation_water() {
    use ferric_rpa::properties::dielectric_spectrum_static;

    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let scf_op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(scf_op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, scf_op, &bounds, &RhfConfig::default()).unwrap();

    let thresh = 1e-4;
    println!("\n=== water aug-cc-pVDZ : PDEP response rank vs attenuation ===");
    println!("  rank = #{{λ_α − 1 > {thresh:.0e}}} (the 'ideal basis' size for the response)\n");
    println!(
        "  {:>10}  {:>6}  {:>6}  {:>10}  {:>12}",
        "ω(Bohr⁻¹)", "naux", "rank", "rank/naux", "|trace_log|"
    );
    for &w in &[0.0_f64, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5] {
        let op = if w == 0.0 { Operator::coulomb() } else { Operator::erfc(w) };
        let spec = dielectric_spectrum_static(&mol, &obs, &dfbs, &rhf, op, thresh).unwrap();
        println!(
            "  {:>10.3}  {:>6}  {:>6}  {:>10.4}  {:>12.5}",
            w,
            spec.naux,
            spec.rank,
            spec.rank as f64 / spec.naux as f64,
            spec.trace_log.abs()
        );
    }
    println!();
}
