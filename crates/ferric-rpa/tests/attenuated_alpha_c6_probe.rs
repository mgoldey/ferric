//! Mostly SINGLE-MOLECULE SANITY PROBES (preflight, not CI assertions) for
//! attenuated/screened RPA, PLUS (as of the S3-TIGHTEN pass) a handful of
//! real, tight, `#[ignore]`-FREE regression tests that assert a number.
//!
//! "Attenuated/screened RPA" is NOT a separate implementation from plain
//! dRPA: it is `run_pdep_rpa`/`pdep_dynamic_polarizability` fed
//! `Operator::erfc(ω)` (or `erf(ω)`) instead of `Operator::coulomb()` —
//! verified by reading the call graph (`rs_mp2_rpa.rs`'s formulation T calls
//! the SAME `run_pdep_rpa`). So the exact-limit test convention already used
//! for `rs_mp2_lr_rpa` (ω→0/ω→∞ unit tests in `rs_mp2_rpa.rs`) applies
//! directly to this file's erfc-attenuated `run_pdep_rpa` calls too — see
//! `erfc_rpa_omega_to_zero_matches_coulomb_rpa` /
//! `erfc_rpa_omega_to_infinity_vanishes` below, both real assertions, not
//! `#[ignore]`d, run as part of plain `cargo test -p ferric-rpa`.
//!
//! What remains `#[ignore]`d (genuinely no ground truth to assert against —
//! PySCF has no equivalent attenuated-RPA α/C6 method, and CRC α / DOSD C6
//! are external empirical anchors, not a first-principles reference for
//! *this specific* method): the α/C6 sweeps at production ω below.
//! Hypothesis under test there: attenuating the Coulomb response kernel
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
//! Run the sweeps with:  cargo test -p ferric-rpa --test attenuated_alpha_c6_probe -- --nocapture --ignored
//! Those remain `#[ignore]`d so they never gate CI (they print, they do not
//! assert physics) — see `docs/VALIDATION.md`'s "Attenuated / screened RPA"
//! row for the current grade split (limits: Proven (narrow); α/C6 sweeps: Smoke).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{casimir_polder_c6, pdep_dynamic_polarizability, DispersionPartition};
use ferric_rpa::properties::pdep_polarizability_static;
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
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

/// SCREENING-PHYSICS TEST: is the C6-optimal attenuation ω tied to the gap?
///
/// Matt's hypothesis: attenuation should behave like physical screening, whose
/// length is set by the system (HOMO–LUMO gap), NOT a free knob. If so, the ω
/// that makes RPA@PBE C6 hit DOSD should satisfy ω_needed ≈ c·Δ_gap with ONE
/// universal constant c across molecules. If c varies per molecule, ω is still
/// just a per-system fit (the cancellation we already found).
///
/// For each molecule: PBE reference, report Δ_gap (HOMO–LUMO, a.u.), bisect for
/// ω_needed where C6(ω)=DOSD, print ω_needed and the ratio ω_needed/Δ_gap.
/// A constant ratio column ⟹ screening physics; a scattered one ⟹ fit.
///
/// CLEAN VERSION (replaces the N2-stretch test, which was confounded by
/// multireference breakdown — the stretched gap is meaningless past Coulson–
/// Fischer ~1.2 Å). Here the gap is varied across an EQUILIBRIUM, all-single-
/// reference, closed-shell set spanning a wide VALENCE-gap range. DOSD targets
/// from testdata/reference/dosd_c6.json.
///
/// Run: cargo test --release -p ferric-rpa --test attenuated_alpha_c6_probe \
///        screening_omega_vs_gap -- --ignored --nocapture
#[test]
#[ignore]
fn screening_omega_vs_gap() {
    // (label, xyz [Å], DOSD molecular C6_AA). Equilibrium, closed-shell, SR.
    // Ordered roughly large-gap → small-gap.
    let mols: &[(&str, &str, f64)] = &[
        ("hf",   "2\nhf\nF 0 0 0\nH 0 0 0.917\n", 19.0),
        ("h2o",  "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n", 45.3),
        ("ch4",  "5\nch4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n", 129.7),
        ("nh3",  "4\nnh3\nN 0 0 0.0\nH 0 0.9377 -0.3816\nH 0.8120 -0.4689 -0.3816\nH -0.8120 -0.4689 -0.3816\n", 89.0),
        ("co",   "2\nco\nC 0 0 0\nO 0 0 1.128\n", 81.4),
        ("n2",   "2\nn2\nN 0 0 0.0\nN 0 0 1.0977\n", 73.3),
        ("co2",  "3\nco2\nC 0 0 0.0\nO 0 0 1.1621\nO 0 0 -1.1621\n", 158.7),
        ("c2h4", "6\nc2h4\nC 0 0 0.6695\nC 0 0 -0.6695\nH 0 0.9289 1.2321\nH 0 -0.9289 1.2321\nH 0 0.9289 -1.2321\nH 0 -0.9289 -1.2321\n", 300.2),
    ];
    let ctx = ParallelContext::default();
    let cfg = PdepRpaConfig::default();

    println!("\n=== Screening test: ω needed for RPA@PBE C6 = DOSD, vs HOMO–LUMO gap ===");
    println!("  (constant ω/Δ ratio ⟹ screening physics; scattered ⟹ per-system fit)\n");
    println!(
        "  {:>5}  {:>8}  {:>8}  {:>10}  {:>10}  {:>10}",
        "mol", "Δ_gap", "C6(ω=0)", "DOSD", "ω_needed", "ω/Δ"
    );

    for (label, xyz, dosd) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let scf_op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(scf_op, &obs).unwrap();
        let scf_cfg = RhfConfig {
            energy_conv: 1e-9,
            xc: Some("PBE".to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()),
            ..Default::default()
        };
        let rhf = solve_rhf(&ctx, &mol, &obs, scf_op, &bounds, &scf_cfg).unwrap();
        assert!(rhf.converged, "{label}: PBE SCF not converged");

        let eps = rhf.eps_r();
        let nocc = (mol.nelec() / 2) as usize;
        let gap = eps[nocc] - eps[nocc - 1]; // LUMO − HOMO (a.u.)

        let c6_at = |w: f64| -> f64 {
            let op = if w == 0.0 { Operator::coulomb() } else { Operator::erfc(w) };
            let dp = pdep_dynamic_polarizability(
                &mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg, DispersionPartition::Becke, None,
            )
            .unwrap();
            casimir_polder_c6(&dp).c6_molecular_iso
        };

        let c6_0 = c6_at(0.0);
        // Bisect ω in [0, 3] for C6(ω) = DOSD (C6 increases monotonically with ω
        // for RPA@PBE — confirmed in the clincher probe).
        let (mut lo, mut hi) = (0.0_f64, 3.0_f64);
        let target = *dosd;
        let omega_needed = if c6_0 >= target {
            0.0 // already at/over DOSD at Coulomb — no attenuation needed
        } else if c6_at(hi) < target {
            f64::NAN // can't reach even at ω=3
        } else {
            for _ in 0..24 {
                let mid = 0.5 * (lo + hi);
                if c6_at(mid) < target { lo = mid; } else { hi = mid; }
            }
            0.5 * (lo + hi)
        };
        let ratio = omega_needed / gap;
        println!(
            "  {:>5}  {:>8.4}  {:>8.3}  {:>10.3}  {:>10.4}  {:>10.4}",
            label, gap, c6_0, target, omega_needed, ratio
        );
    }
    println!("\n  Verdict: compare the ω/Δ column. Constant ⟹ screening; scattered ⟹ fit.");
}

// NOTE: a prior `screening_n2_stretch_gap_response` test was removed — using a
// bond stretch to vary the gap is confounded by multireference breakdown past
// the Coulson–Fischer point (~1.2 Å for N2), where the HOMO–LUMO gap is no
// longer a meaningful valence gap and RHF/PBE+RPA is qualitatively wrong. The
// gap must be varied by electronic structure at equilibrium instead — that is
// what the expanded `screening_omega_vs_gap` valence set above now does.

/// PHYSICS vs CANCELLATION: is the attenuation C6 correction STRUCTURED (real)
/// or UNIFORM (a scale factor in disguise)?
///
/// A transferable ω≈0.3 Bohr⁻¹ corrects RPA@PBE C6 to ~few-% across molecules.
/// Two stories:
///   (A) cancellation/scale: RPA@PBE is ~−16% uniformly; attenuation lifts α(iω)
///       by a uniform fraction, so a plain scalar C6×1.18 does just as well and
///       the correction ratio α_att(iω)/α_0(iω) is FLAT across frequency.
///   (B) real physics: attenuation restores short-range correlation (the known
///       dRPA short-range over-screening that RSH-RPA fixes), so the lift is
///       CONCENTRATED at high iω (short-time/short-range response) — a scalar
///       canNOT reproduce it.
///
/// Two diagnostics per molecule at the transferable ω=0.30:
///   1. shape of r(iω) = α_iso_att(iω) / α_iso_0(iω) across the CP grid
///      (flat ⟹ scale; rising/falling ⟹ structured).
///   2. does a global scalar (C6×s, s = mean baseline ratio) match DOSD as well
///      as the ω=0.30 attenuation? (compare per-molecule |err|).
///
/// Anchor: the dissertation erfc optimum is ω=0.222 Bohr⁻¹; our C6 ω≈0.31 is the
/// same ballpark — a hint this touches real range-separation physics.
///
/// Run: cargo test --release -p ferric-rpa --test attenuated_alpha_c6_probe \
///        attenuation_structure_vs_scalar -- --ignored --nocapture
#[test]
#[ignore]
fn attenuation_structure_vs_scalar() {
    let mols: &[(&str, &str, f64)] = &[
        ("hf",   "2\nhf\nF 0 0 0\nH 0 0 0.917\n", 19.0),
        ("h2o",  "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n", 45.3),
        ("ch4",  "5\nch4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n", 129.7),
        ("n2",   "2\nn2\nN 0 0 0.0\nN 0 0 1.0977\n", 73.3),
        ("co2",  "3\nco2\nC 0 0 0.0\nO 0 0 1.1621\nO 0 0 -1.1621\n", 158.7),
        ("c2h4", "6\nc2h4\nC 0 0 0.6695\nC 0 0 -0.6695\nH 0 0.9289 1.2321\nH 0 -0.9289 1.2321\nH 0 0.9289 -1.2321\nH 0 -0.9289 -1.2321\n", 300.2),
    ];
    let ctx = ParallelContext::default();
    let cfg = PdepRpaConfig::default();
    let omega_fixed = 0.30_f64; // the transferable optimum (Bohr⁻¹)

    let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;

    // First pass: collect baseline & attenuated C6 + the α(iω) profiles.
    struct Row { label: String, c6_0: f64, c6_w: f64, dosd: f64, ratios: Vec<(f64, f64)> }
    let mut rows: Vec<Row> = Vec::new();

    for (label, xyz, dosd) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let scf_op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(scf_op, &obs).unwrap();
        let scf_cfg = RhfConfig {
            energy_conv: 1e-9,
            xc: Some("PBE".to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()),
            ..Default::default()
        };
        let rhf = solve_rhf(&ctx, &mol, &obs, scf_op, &bounds, &scf_cfg).unwrap();

        let dp0 = pdep_dynamic_polarizability(
            &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::coulomb(), &cfg, DispersionPartition::Becke, None,
        ).unwrap();
        let dpw = pdep_dynamic_polarizability(
            &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::erfc(omega_fixed), &cfg, DispersionPartition::Becke, None,
        ).unwrap();

        let c6_0 = casimir_polder_c6(&dp0).c6_molecular_iso;
        let c6_w = casimir_polder_c6(&dpw).c6_molecular_iso;

        // r(iω) = α_att(iω) / α_0(iω) across the grid.
        let ratios: Vec<(f64, f64)> = dp0.freqs.iter().enumerate().map(|(k, &w)| {
            (w, iso(&dpw.molecular[k]) / iso(&dp0.molecular[k]))
        }).collect();

        rows.push(Row { label: label.to_string(), c6_0, c6_w, dosd: *dosd, ratios });
    }

    // Global scalar s = mean(DOSD/C6_0) — the best single multiplicative correction.
    let s: f64 = rows.iter().map(|r| r.dosd / r.c6_0).sum::<f64>() / rows.len() as f64;

    println!("\n=== Attenuation structure vs scalar (RPA@PBE C6, ω={omega_fixed} Bohr⁻¹) ===");
    println!("  global scalar s = mean(DOSD/C6₀) = {s:.4}\n");
    println!("  {:>5}  {:>9}  {:>9}  {:>9}  {:>10}  {:>10}",
        "mol", "C6₀ err%", "C6(ω) err%", "scalar err%", "att |err|", "scalar |err|");
    let (mut att_mae, mut sca_mae) = (0.0, 0.0);
    for r in &rows {
        let e0 = 100.0 * (r.c6_0 - r.dosd) / r.dosd;
        let ew = 100.0 * (r.c6_w - r.dosd) / r.dosd;
        let es = 100.0 * (s * r.c6_0 - r.dosd) / r.dosd;
        att_mae += ew.abs(); sca_mae += es.abs();
        println!("  {:>5}  {:>+8.2}  {:>+9.2}  {:>+10.2}  {:>10.2}  {:>10.2}",
            r.label, e0, ew, es, ew.abs(), es.abs());
    }
    att_mae /= rows.len() as f64; sca_mae /= rows.len() as f64;
    println!("\n  MAE: attenuation(ω=0.30) = {att_mae:.2}%   global scalar = {sca_mae:.2}%");
    println!("  (attenuation MAE < scalar MAE ⟹ structured/real; ≈ ⟹ scale in disguise)\n");

    // Frequency STRUCTURE of the correction (the decisive diagnostic).
    println!("  --- correction shape r(iω)=α_att/α₀ across the CP grid (flat ⟹ scalar) ---");
    // Use a representative subset of grid points (first molecule's grid).
    let grid = &rows[0].ratios;
    let idxs: Vec<usize> = {
        let n = grid.len();
        vec![0, n/4, n/2, 3*n/4, n-1]
    };
    print!("  {:>5} ", "ω→");
    for &k in &idxs { print!("{:>9.3}", grid[k].0); }
    println!("   span%");
    for r in &rows {
        print!("  {:>5} ", r.label);
        let vals: Vec<f64> = idxs.iter().map(|&k| r.ratios[k].1).collect();
        for v in &vals { print!("{:>9.4}", v); }
        let (lo, hi) = (vals.iter().cloned().fold(f64::MAX, f64::min),
                        vals.iter().cloned().fold(f64::MIN, f64::max));
        println!("   {:>5.1}", 100.0 * (hi - lo) / lo);
    }
    println!("\n  span% = spread of r(iω) over frequency. Near 0 ⟹ uniform lift (scalar).");
    println!("  Large/systematic ⟹ frequency-structured (short-range physics).");
}

/// erf(ωr)/r RPA — the LONG-RANGE operator (complement of erfc), the one tied
/// to real range-separated RPA (Toulouse/Ángyán/Janesko-Scuseria: SR=DFT,
/// LR=RPA on erf). erfc inflated the long-range/static α (wrong lever, scalar-
/// like). erf acts on the SHORT-range part, where RPA actually fails — so IF
/// attenuation has real dispersion physics, the erf correction should be
/// frequency-STRUCTURED (high-iω) and a scalar should NOT reproduce it.
///
/// Limits (opposite to erfc!): erf(ωr)/r → 0 as ω→0 (no interaction),
///   → 1/r as ω→∞ (full Coulomb). So large ω ≈ baseline; small ω = far tail
///   only. The long-range-RPA regime is small-to-moderate ω.
///
/// Run: cargo test --release -p ferric-rpa --test attenuated_alpha_c6_probe \
///        erf_rpa_c6_and_structure -- --ignored --nocapture
#[test]
#[ignore]
fn erf_rpa_c6_and_structure() {
    let mols: &[(&str, &str, f64)] = &[
        ("hf",   "2\nhf\nF 0 0 0\nH 0 0 0.917\n", 19.0),
        ("h2o",  "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n", 45.3),
        ("ch4",  "5\nch4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n", 129.7),
        ("n2",   "2\nn2\nN 0 0 0.0\nN 0 0 1.0977\n", 73.3),
        ("co2",  "3\nco2\nC 0 0 0.0\nO 0 0 1.1621\nO 0 0 -1.1621\n", 158.7),
        ("c2h4", "6\nc2h4\nC 0 0 0.6695\nC 0 0 -0.6695\nH 0 0.9289 1.2321\nH 0 -0.9289 1.2321\nH 0 0.9289 -1.2321\nH 0 -0.9289 -1.2321\n", 300.2),
    ];
    let ctx = ParallelContext::default();
    let cfg = PdepRpaConfig::default();
    let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;

    // erf ω grid: small=far-tail-only ... large≈full Coulomb. Sweep to see if
    // any erf regime gives a STRUCTURED correction (not just monotone-to-baseline).
    let omegas = [0.3_f64, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0];

    println!("\n=== erf(ωr)/r RPA@PBE : C6 sweep (long-range operator) ===");
    println!("  erf: ω→0 = no interaction, ω→∞ = full Coulomb. DOSD targets.\n");

    // For one molecule (water), also dump the correction SHAPE vs full-Coulomb
    // baseline at a representative ω, to compare against erfc's low-iω lift.
    for (label, xyz, dosd) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let scf_op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(scf_op, &obs).unwrap();
        let scf_cfg = RhfConfig {
            energy_conv: 1e-9,
            xc: Some("PBE".to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()),
            ..Default::default()
        };
        let rhf = solve_rhf(&ctx, &mol, &obs, scf_op, &bounds, &scf_cfg).unwrap();

        // Full-Coulomb baseline C6 for reference.
        let dp_cb = pdep_dynamic_polarizability(
            &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::coulomb(), &cfg, DispersionPartition::Becke, None,
        ).unwrap();
        let c6_cb = casimir_polder_c6(&dp_cb).c6_molecular_iso;

        print!("  {:>5}  DOSD={:>7.1}  C6(full)={:>7.2}({:+5.1}%) | erf ω:",
            label, dosd, c6_cb, 100.0*(c6_cb-dosd)/dosd);
        for &w in &omegas {
            let dp = pdep_dynamic_polarizability(
                &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::erf(w), &cfg, DispersionPartition::Becke, None,
            ).unwrap();
            let c6 = casimir_polder_c6(&dp).c6_molecular_iso;
            print!("  {:.1}→{:+.0}%", w, 100.0*(c6-dosd)/dosd);
        }
        println!();

        // Structure diagnostic on water only: r(iω)=α_erf(iω)/α_full(iω) at ω=1.0.
        if *label == "h2o" {
            let dpw = pdep_dynamic_polarizability(
                &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::erf(1.0), &cfg, DispersionPartition::Becke, None,
            ).unwrap();
            println!("\n  --- erf correction shape r(iω)=α_erf(ω=1.0)/α_full, water ---");
            let n = dp_cb.freqs.len();
            let idxs = [0usize, n/4, n/2, 3*n/4, n-1];
            print!("    {:>5}", "ω→");
            for &k in &idxs { print!("{:>10.3}", dp_cb.freqs[k]); }
            println!();
            print!("    {:>5}", "r");
            for &k in &idxs { print!("{:>10.4}", iso(&dpw.molecular[k]) / iso(&dp_cb.molecular[k])); }
            println!("\n    (erfc was HIGH at low-iω→1.0 at high-iω. erf rising at high-iω ⟹ short-range, real.)\n");
        }
    }
    println!("\n  Read: does any erf ω land C6 near DOSD with a HIGH-iω-structured correction");
    println!("  (real short-range physics) vs erfc's scalar-like low-iω lift?");
}

/// S3 spike (open triage item #3): pin the SR-RPA CORRELATION-ENERGY recovery
/// ratio claimed in the `attenuated-rpa-recovers-most-correlation` memory
/// file: "SR-RPA with erfc(ω=0.222 Bohr⁻¹) on H2O/cc-pVDZ recovers 97.7% of
/// full Coulomb-RPA correlation energy." That number previously existed only
/// as a memory-file record from an ad-hoc investigation; no test computed and
/// asserted this specific ratio (grepped — no `e_rpa` recovery-ratio
/// computation at ω=0.222 existed anywhere in ferric-rpa's tests or src).
///
/// Distinct quantity from `attenuated_water_alpha_c6_probe` above: this test
/// is about the RPA CORRELATION ENERGY (`run_pdep_rpa(...).e_rpa`, the
/// dielectric-trace-log quantity consumed by SCS/RS-MP2-RPA-style methods),
/// not static polarizability or C6.
///
/// SCF (RHF) is always full Coulomb; only the RPA response kernel is
/// attenuated (erfc(ωr)/r replaces 1/r in the dielectric build), matching the
/// convention used throughout this file and in
/// `ferric_mp2::attenuated::attenuated_ri_mp2` (ω in Bohr⁻¹, library-native
/// unit — the CLI/Python Å⁻¹→Bohr⁻¹ conversion happens only at that
/// boundary).
///
/// `PdepRpaConfig::default()` (Lanczos eigensolver, no diagnostics) is the
/// same cheap config the non-`#[ignore]`d `h2o_cc_pvdz_pdep_rpa_matches_pyscf`
/// test in `pdep_rpa.rs` uses at cc-pVDZ — so this is cheap enough to run
/// un-ignored in plain `cargo test -p ferric-rpa`.
#[test]
fn attenuated_rpa_correlation_recovery_water_ccpvdz() {
    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();

    // SCF is always full Coulomb.
    let scf_op = Operator::coulomb();
    let scf_bounds = SchwarzBounds::compute(scf_op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, scf_op, &scf_bounds, &RhfConfig::default()).unwrap();
    assert!(rhf.converged, "H2O/cc-pVDZ RHF reference did not converge");

    let cfg = PdepRpaConfig::default();

    // Full Coulomb-RPA correlation energy.
    let full = run_pdep_rpa(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &cfg).unwrap();
    // SR-RPA at the production omega (0.420 A^-1 -> 0.222 Bohr^-1, the same
    // value used throughout attenuated_ri_mp2's default and this file's
    // erfc probes).
    let omega_bohr = 0.222_f64;
    let sr = run_pdep_rpa(&mol, &obs, &dfbs, Operator::erfc(omega_bohr), &rhf, &cfg).unwrap();

    assert!(full.e_rpa < 0.0, "full-Coulomb E_c should be negative, got {}", full.e_rpa);
    assert!(sr.e_rpa < 0.0, "SR (erfc) E_c should be negative, got {}", sr.e_rpa);

    let recovery = sr.e_rpa / full.e_rpa;
    println!(
        "\nH2O/cc-pVDZ RPA correlation: full(Coulomb)={:.10} Ha, SR(erfc ω=0.222)={:.10} Ha, recovery={:.4}%\n",
        full.e_rpa, sr.e_rpa, 100.0 * recovery
    );

    // Memory-claimed value: 97.7% recovery. Measured here from a real
    // run_pdep_rpa call (not carried over from the prior ad-hoc
    // investigation) — pin with a tolerance wide enough to survive minor
    // eigensolver/config drift since the original measurement, but tight
    // enough to catch a qualitatively different picture (e.g. off by several
    // percentage points).
    let expected_recovery_pct = 97.7;
    let measured_pct = 100.0 * recovery;
    let diff_pct = (measured_pct - expected_recovery_pct).abs();
    assert!(
        diff_pct < 1.0,
        "SR-RPA recovery = {measured_pct:.3}%, expected ~{expected_recovery_pct}% \
         (memory: attenuated-rpa-recovers-most-correlation); diff={diff_pct:.3} pct pts"
    );
}

/// S3-TIGHTEN (attenuated/screened RPA grade upgrade): the two ANALYTIC LIMITS
/// of `run_pdep_rpa` under `Operator::erfc(ω)`, which is exactly the response
/// kernel `attenuated_water_alpha_c6_probe`/`attenuated_water_rpa_pbe_c6_probe`
/// above sweep — this file's "attenuated RPA" is NOT a separate implementation,
/// it is plain `run_pdep_rpa`/`pdep_dynamic_polarizability` fed
/// `Operator::erfc(ω)` instead of `Operator::coulomb()` (verified by reading
/// `lib.rs`/`rs_mp2_rpa.rs`: `rs_mp2_lr_rpa`'s formulation T calls this SAME
/// `run_pdep_rpa` with `Operator::erfc`/`Operator::coulomb`). So the exact
/// ω→0/ω→∞ limit-test convention already used and Proven for
/// `rs_mp2_lr_rpa` (`crate::rs_mp2_rpa::tests::omega_to_zero_reduces_to_mp2` /
/// `omega_to_infinity_is_mp2_plus_delta_drpa`) applies verbatim here, one level
/// down the call stack, to the plain dRPA correlation energy itself:
///
///   * ω→0: `erfc(ωr)/r → 1/r` pointwise (libint2's native range-separated
///     Coulomb kernel; `OperatorKind::ErfcCoulomb` in `operator.rs`), so
///     `run_pdep_rpa(erfc(ω))` must converge to `run_pdep_rpa(coulomb())`
///     as ω→0. This is a DIFFERENT limit from `rs_mp2_lr_rpa`'s (which
///     collapses the whole SR-MP2+LR-RPA composite to plain MP2 because the
///     LR *dRPA correction relative to MP2* vanishes) — here it is the raw
///     attenuated dRPA energy itself converging to the raw Coulomb dRPA
///     energy, the literal quantity this probe file's α/C6 sweeps consume.
///   * ω→∞: `erfc(ωr)/r → 0` pointwise (no interaction), so the RPA dielectric
///     response vanishes (all eigenvalues λ_α(iω) → 1) and
///     `e_rpa = Σ_k w_k Σ_α [ln λ_α + (1−λ_α)] → 0` (each bracket → ln 1 + 0 = 0
///     as λ_α → 1; see `energy::rpa_correlation_energy`).
///
/// Measured on H2/cc-pVDZ (same cheap system `rs_mp2_rpa.rs`'s own limit
/// tests use) before picking tolerances:
///   ω=0.01 Bohr⁻¹: |e_rpa(erfc) − e_rpa(coulomb)| = 1.94e-7 Ha (monotone ↓ in ω)
///   ω=20   Bohr⁻¹: |e_rpa(erfc)|                  = 2.95e-7 Ha (monotone ↓ in ω)
/// Tolerances below are set at 5e-7 (~2.5x the measured residual) to leave
/// headroom for eigensolver/config drift while still catching a qualitatively
/// broken kernel (a convention bug here is off by orders of magnitude, not a
/// few×1e-7 — see the `rs_mp2_rpa.rs` precedent's own tolerance commentary).
#[test]
fn erfc_rpa_omega_to_zero_matches_coulomb_rpa() {
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let scf_op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(scf_op, &obs).unwrap();
    let rhf = solve_rhf(
        &ctx, &mol, &obs, scf_op, &bounds,
        &RhfConfig { energy_conv: 1e-10, ..Default::default() },
    ).unwrap();
    assert!(rhf.converged, "H2/cc-pVDZ RHF reference did not converge");

    // Full rank (trunc_thresh = 0.0): this is an energy comparison, not a
    // production-size perf run, so no truncation noise should enter.
    let cfg = PdepRpaConfig { trunc_thresh: 0.0, ..Default::default() };

    let coul = run_pdep_rpa(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &cfg).unwrap();
    let sr = run_pdep_rpa(&mol, &obs, &dfbs, Operator::erfc(0.01), &rhf, &cfg).unwrap();

    let diff = (sr.e_rpa - coul.e_rpa).abs();
    println!(
        "\nH2/cc-pVDZ dRPA: coulomb e_rpa={:.12}  erfc(ω=0.01) e_rpa={:.12}  diff={:.3e}\n",
        coul.e_rpa, sr.e_rpa, diff
    );
    assert!(
        diff < 5e-7,
        "erfc(ω→0) must reduce to plain Coulomb dRPA: coulomb={:.10}, erfc(0.01)={:.10}, diff={:.3e}",
        coul.e_rpa, sr.e_rpa, diff
    );
}

/// ω→∞ complement of the above: `erfc(ωr)/r → 0` everywhere, so the response
/// vanishes and `e_rpa → 0`. See the doc comment on
/// `erfc_rpa_omega_to_zero_matches_coulomb_rpa` for the full derivation and
/// measured-residual justification of the tolerance.
#[test]
fn erfc_rpa_omega_to_infinity_vanishes() {
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let scf_op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(scf_op, &obs).unwrap();
    let rhf = solve_rhf(
        &ctx, &mol, &obs, scf_op, &bounds,
        &RhfConfig { energy_conv: 1e-10, ..Default::default() },
    ).unwrap();
    assert!(rhf.converged, "H2/cc-pVDZ RHF reference did not converge");

    let cfg = PdepRpaConfig { trunc_thresh: 0.0, ..Default::default() };

    let sr = run_pdep_rpa(&mol, &obs, &dfbs, Operator::erfc(20.0), &rhf, &cfg).unwrap();

    println!("\nH2/cc-pVDZ dRPA: erfc(ω=20) e_rpa={:.12} (should -> 0)\n", sr.e_rpa);
    assert!(
        sr.e_rpa.abs() < 5e-7,
        "erfc(ω→∞) must vanish: e_rpa={:.10}",
        sr.e_rpa
    );
}
