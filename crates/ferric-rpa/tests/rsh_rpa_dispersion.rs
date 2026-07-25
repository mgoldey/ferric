//! Consistent range-separated RPA (RSH-RPA) for dispersion.
//!
//! See docs/superpowers/specs/2026-06-04-rsh-rpa-dispersion-design.md.
//! The honest test of "does range-separation help dispersion": a long-range-
//! corrected hybrid REFERENCE (SR-DFT + 100% LR-HF exchange via erf) paired with
//! LONG-RANGE (erf) RPA correlation at a MATCHED ω — not a response-only kernel
//! swap (which we showed is error cancellation).
//!
//! Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{casimir_polder_c6, pdep_dynamic_polarizability, DispersionPartition};
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Option B probe: which LC references are valid for RSH-RPA (c_lr=1.0 = 100%
/// long-range HF, so LR-RPA supplies ALL long-range correlation, no double-count)
/// AND give better compact-system α? Print CAM coeffs + converged water α(0).
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        lc_reference_survey -- --ignored --nocapture
#[test]
#[ignore]
fn lc_reference_survey() {
    use ferric_dft::libxc::xc_def_from_name;
    use ferric_rpa::properties::pdep_polarizability_static;

    let candidates = [
        "HYB_GGA_XC_LC_WPBE",   // baseline (c_lr=1.0, ω=0.40)
        "HYB_GGA_XC_LC_BLYP",   // pure LC-BLYP (c_lr=1.0)
        "HYB_GGA_XC_LRC_WPBE",  // LRC-ωPBE (c_lr=1.0, different ω)
        "HYB_GGA_XC_LRC_WPBEH", // LRC-ωPBEh (c_lr=1.0, short-range HF)
        "HYB_GGA_XC_CAM_B3LYP", // CAM (c_lr=0.65 — partial, double-count risk)
        "WB97XV",               // c_lr=1.0 but has VV10 (dispersion double-count)
    ];
    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = PdepRpaConfig::default();

    eprintln!("\n=== LC reference survey (water; RSH-RPA needs c_lr=1.0, good α) ===");
    eprintln!("  ref CRC α(H2O)≈9.8; PBE static-RPA α≈8.7; want LC α closer to 9.8\n");
    eprintln!("  {:>22}  {:>6} {:>6} {:>6}  {:>9} {:>9}  note", "functional", "ω", "c_sr", "c_lr", "E", "α(erf)");
    for name in candidates {
        let cam = xc_def_from_name(name).ok().and_then(|d| d.cam);
        let (om, csr, clr) = cam.map(|c| (c.omega, c.c_sr, c.c_lr)).unwrap_or((f64::NAN, f64::NAN, f64::NAN));
        let scf_cfg = RhfConfig {
            energy_conv: 1e-9, xc: Some(name.to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()),
            ..Default::default()
        };
        match solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg) {
            Ok(r) if r.converged && om.is_finite() => {
                // static LR-RPA α at the native ω (the dispersion-relevant α magnitude).
                let a = pdep_polarizability_static(&mol, &obs, &dfbs, &r, Operator::erf(om), &cfg)
                    .map(|x| x.iso).unwrap_or(f64::NAN);
                let note = if (clr - 1.0).abs() < 1e-6 { "RSH-ok" } else { "c_lr≠1 (double-count)" };
                eprintln!("  {:>22}  {:>6.3} {:>6.3} {:>6.3}  {:>9.4} {:>9.4}  {note}", name, om, csr, clr, r.energy, a);
            }
            Ok(_) => eprintln!("  {:>22}  {:>6.3} {:>6.3} {:>6.3}  not converged / no CAM", name, om, csr, clr),
            Err(e) => eprintln!("  {:>22}  error: {}", name, e),
        }
    }
}

/// Step 1 (gating smoke): an LC-ωPBE reference must converge. Validates the LC
/// functional is reachable (raw libxc name) and the SR-hybrid SCF works — the
/// foundation of the whole RSH-RPA build.
#[test]
#[ignore]
fn rsh_reference_lc_wpbe_converges() {
    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvdz").unwrap()).unwrap();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    // Try LC-ωPBE by raw libxc name (pure SR-PBE + 100% LR-HF exchange, no extra
    // correlation → no dispersion double-counting). Fall back to wB97X-V (has VV10,
    // documented double-count caveat) if LC_WPBE isn't in this libxc build.
    for name in ["HYB_GGA_XC_LC_WPBE", "LC-wPBE", "WB97XV"] {
        let cfg = RhfConfig {
            energy_conv: 1e-9,
            xc: Some(name.to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()),
            ..Default::default()
        };
        match solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg) {
            Ok(r) if r.converged => {
                let eps = r.eps_r();
                let nocc = (mol.nelec() / 2) as usize;
                eprintln!(
                    "LC reference '{name}' CONVERGED: E={:.6}  HOMO={:.4}  LUMO={:.4}  gap={:.4}",
                    r.energy, eps[nocc - 1], eps[nocc], eps[nocc] - eps[nocc - 1]
                );
                return; // first working LC functional is enough for the gate
            }
            Ok(_) => eprintln!("LC reference '{name}': SCF did not converge"),
            Err(e) => eprintln!("LC reference '{name}': error {e}"),
        }
    }
    panic!("no LC functional converged — RSH-RPA reference unavailable");
}

/// THE DECISIVE EXPERIMENT: consistent RSH-RPA C6 at the functional's NATIVE ω.
///
/// LC-ωPBE reference (SR-PBE + 100% LR-HF exchange) + LR-erf RPA correlation, BOTH
/// at the functional's native ω (read from its CAM coefficients). No per-molecule
/// fitting — ω is fixed by the functional. If RSH-RPA C6 lands near DOSD at this
/// fixed ω across molecules, range-separation genuinely helps dispersion (real).
/// If it only works at a C6-fit ω ≠ ω_native, it's still a knob.
///
/// Compare MAE vs: RPA@PBE (−15% uniform), scalar-corrected RPA@PBE (2.4%),
/// erfc/erf response-only knobs (5%+).
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        rsh_rpa_c6_native_omega -- --ignored --nocapture
/// Shared RSH-RPA C6 driver: LC `lc_name` reference + LR-erf RPA at the
/// functional's native ω, valence set vs DOSD. Returns the MAE%.
fn run_rsh_rpa_c6(lc_name: &str) -> f64 {
    use ferric_dft::libxc::xc_def_from_name;

    let cam = xc_def_from_name(lc_name).ok().and_then(|d| d.cam)
        .expect("LC functional must expose CAM omega");
    let omega = cam.omega;
    eprintln!("\n[{lc_name}] native ω={:.3}  c_sr={:.3}  c_lr={:.3} {}",
        omega, cam.c_sr, cam.c_lr,
        if (cam.c_lr - 1.0).abs() > 1e-6 { "⚠ c_lr≠1 (LR double-count)" } else { "" });

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

    eprintln!("  {:>5}  {:>9}  {:>9}  {:>8}", "mol", "C6_RSH", "DOSD", "err%");
    let mut mae = 0.0;
    let mut n = 0;
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
            xc: Some(lc_name.to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()),
            ..Default::default()
        };
        let rhf = match solve_rhf(&ctx, &mol, &obs, scf_op, &bounds, &scf_cfg) {
            Ok(r) if r.converged => r,
            _ => { eprintln!("  {label:>5}  (LC SCF not converged — skipped)"); continue; }
        };

        // LR-erf RPA correlation at the SAME ω → α(iω) → molecular C6.
        let dp = pdep_dynamic_polarizability(
            &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::erf(omega), &cfg,
            DispersionPartition::Becke, None,
        ).unwrap();
        let c6 = casimir_polder_c6(&dp).c6_molecular_iso;
        let err = 100.0 * (c6 - dosd) / dosd;
        eprintln!("  {:>5}  {:>9.3}  {:>9.1}  {:>+7.2}", label, c6, dosd, err);
        mae += err.abs(); n += 1;
    }
    let mae = if n > 0 { mae / n as f64 } else { f64::NAN };
    eprintln!("  → MAE (fixed native ω, NO fit) = {mae:.2}%");
    mae
}

#[test]
#[ignore]
fn rsh_rpa_c6_native_omega() {
    eprintln!("=== Consistent RSH-RPA C6 vs DOSD ===");
    eprintln!("  baselines: RPA@PBE ~16% | scalar-fix 2.4% | response-only knobs 5%+");
    // Baseline LC-ωPBE (worst α) vs LRC-ωPBEh (best valid RSH α from the survey).
    let m_lcwpbe = run_rsh_rpa_c6("HYB_GGA_XC_LC_WPBE");
    let m_lrcwpbeh = run_rsh_rpa_c6("HYB_GGA_XC_LRC_WPBEH");
    eprintln!("\n  SUMMARY  LC-ωPBE MAE={m_lcwpbe:.2}%   LRC-ωPBEh MAE={m_lrcwpbeh:.2}%");
    eprintln!("  Option B win ⟺ LRC-ωPBEh (better compact-α reference) lowers MAE.");
}

/// NOVELTY SPIKE: dielectric-dependent (Brawand-style) ω for RSH-RPA C6.
///
/// Prior art (verified): RSH-RPA C6 with FIXED ω (Toulouse 2013); Brawand/Galli
/// system-dielectric ω for GAPS not dispersion; MBD@rsSCS self-consistent screening
/// but atomic-TS one universal param. The combination — SYSTEM-DIELECTRIC ω driving
/// the range-separation of RPA C6 — appears open. My option-B finding (optimal ω is
/// bonding-type-dependent) is the MOTIVATION: a dielectric ω should supply exactly
/// that dependence.
///
/// CALIBRATION FIRST (this test): for each molecule print α (static RPA), molecular
/// volume V (Σ Becke atomic eff. volumes), the dielectric proxy α/V, and the
/// per-molecule OPTIMAL ω (bisect C6=DOSD on the LC reference). If ω_opt correlates
/// monotonically with α/V, a dielectric ω = f(α/V) CAN fix the split — and the data
/// fixes the functional form + sign. If ω_opt is uncorrelated with α/V, the
/// dielectric hypothesis fails (clean null).
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        dielectric_omega_calibration -- --ignored --nocapture
#[test]
#[ignore]
fn dielectric_omega_calibration() {
    use ferric_rpa::properties::{atomic_effective_volumes_becke, pdep_polarizability_static};

    let lc_name = "HYB_GGA_XC_LC_WPBE";
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

    eprintln!("\n=== Dielectric-ω calibration: does optimal ω track α/V (the dielectric proxy)? ===");
    eprintln!("  ω_opt = ω s.t. RSH-RPA C6 = DOSD (LC-ωPBE ref). α from static RPA@ω_native.\n");
    eprintln!("  {:>5}  {:>8}  {:>9}  {:>9}  {:>9}  {:>8}", "mol", "α", "V", "α/V", "ω_opt", "bond");

    let mut rows: Vec<(String, f64, f64, f64)> = Vec::new(); // label, a_over_v, omega_opt, alpha
    for (label, xyz, dosd) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let scf_op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(scf_op, &obs).unwrap();
        let scf_cfg = RhfConfig {
            energy_conv: 1e-9, xc: Some(lc_name.to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()),
            ..Default::default()
        };
        let rhf = match solve_rhf(&ctx, &mol, &obs, scf_op, &bounds, &scf_cfg) {
            Ok(r) if r.converged => r,
            _ => { eprintln!("  {label:>5}  (SCF not converged)"); continue; }
        };

        // α: static Coulomb RPA (the dielectric-relevant magnitude).
        let alpha = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, Operator::coulomb(), &cfg)
            .map(|x| x.iso).unwrap_or(f64::NAN);
        // V: sum of Becke atomic effective volumes (a.u.³).
        let vol: f64 = atomic_effective_volumes_becke(&mol, &obs, &obs_bs, rhf.density_r())
            .map(|v| v.iter().sum()).unwrap_or(f64::NAN);
        let a_over_v = alpha / vol;

        // ω_opt: bisect erf-RPA C6(ω)=DOSD (C6 increases with ω for erf).
        let c6_at = |w: f64| -> f64 {
            let dp = pdep_dynamic_polarizability(
                &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::erf(w), &cfg,
                DispersionPartition::Becke, None,
            ).unwrap();
            casimir_polder_c6(&dp).c6_molecular_iso
        };
        // erf-RPA C6 DECREASES with ω (small ω → near-bare KS response, large C6;
        // large ω → full Coulomb screening, small C6). Bracket [lo=large C6, hi=small].
        let (mut lo, mut hi) = (0.05_f64, 2.5_f64);
        let omega_opt = if c6_at(lo) < *dosd { f64::NAN }      // even tiny ω undershoots
            else if c6_at(hi) > *dosd { f64::NAN }             // even ω=2.5 overshoots
            else {
                for _ in 0..22 { let m = 0.5*(lo+hi); if c6_at(m) > *dosd { lo = m } else { hi = m } }
                0.5*(lo+hi)
            };
        let bond = if ["co","n2","co2","c2h4"].contains(label) { "π/multi" } else { "saturated" };
        eprintln!("  {:>5}  {:>8.3}  {:>9.2}  {:>9.4}  {:>9.4}  {:>8}", label, alpha, vol, a_over_v, omega_opt, bond);
        rows.push((label.to_string(), a_over_v, omega_opt, alpha));
    }

    // Correlation of ω_opt with α/V (the falsifiable test).
    let valid: Vec<_> = rows.iter().filter(|r| r.2.is_finite()).collect();
    if valid.len() >= 3 {
        let n = valid.len() as f64;
        let mx: f64 = valid.iter().map(|r| r.1).sum::<f64>()/n;
        let my: f64 = valid.iter().map(|r| r.2).sum::<f64>()/n;
        let sxy: f64 = valid.iter().map(|r| (r.1-mx)*(r.2-my)).sum();
        let sxx: f64 = valid.iter().map(|r| (r.1-mx).powi(2)).sum();
        let syy: f64 = valid.iter().map(|r| (r.2-my).powi(2)).sum();
        let pearson = sxy/(sxx.sqrt()*syy.sqrt());
        eprintln!("\n  Pearson r(α/V, ω_opt) = {pearson:+.3}  (slope {:+.4})", sxy/sxx);
        eprintln!("  |r|→1 ⟹ ω_opt IS dielectric-determined → build ω=f(α/V). |r|→0 ⟹ null.");
    }
}

/// DEBUG: actual erf-RPA C6(ω) curve on the LC reference, to fix monotonicity
/// and bracket before trusting the dielectric-ω calibration. The calibration
/// bisection collapsed to the floor → C6(small ω) already ≥ DOSD; verify the
/// real shape.
#[test]
#[ignore]
fn debug_erf_c6_curve_lc_ref() {
    let lc_name = "HYB_GGA_XC_LC_WPBE";
    let mols: &[(&str, &str, f64)] = &[
        ("ch4", "5\nch4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n", 129.7),
        ("n2",  "2\nn2\nN 0 0 0.0\nN 0 0 1.0977\n", 73.3),
    ];
    let ctx = ParallelContext::default();
    let cfg = PdepRpaConfig::default();
    let ws = [0.05_f64, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0];
    for (label, xyz, dosd) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let scf_cfg = RhfConfig { energy_conv: 1e-9, xc: Some(lc_name.to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()), ..Default::default() };
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();
        eprint!("  {label:>4} (DOSD {dosd}): ");
        for &w in &ws {
            let dp = pdep_dynamic_polarizability(&mol,&obs,&obs_bs,&dfbs,&rhf,Operator::erf(w),&cfg,DispersionPartition::Becke,None).unwrap();
            eprint!("{:.1}→{:.0} ", w, casimir_polder_c6(&dp).c6_molecular_iso);
        }
        eprintln!();
    }
}

/// DECISIVE: leave-one-out dielectric-ω RSH-RPA C6. For each molecule, predict ω
/// from ω=a(α/V)+b fit on the OTHER molecules (honest transferability, no in-sample
/// circularity), compute C6 at that predicted ω, report MAE vs DOSD. Beats the
/// fixed-ω 12.4%? → dielectric-ω is a real, transferable improvement (the novelty).
/// Doesn't beat it? → α/V correlation doesn't transfer to C6 accuracy (null).
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        dielectric_omega_loo_c6 -- --ignored --nocapture
#[test]
#[ignore]
fn dielectric_omega_loo_c6() {
    use ferric_rpa::properties::{atomic_effective_volumes_becke, pdep_polarizability_static};

    let lc_name = "HYB_GGA_XC_LC_WPBE";
    let mols: &[(&str, &str, f64)] = &[
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

    // Pass 1: gather (α/V, ω_opt, C6 sampler) per molecule. Reuse the calibration.
    struct M { label: String, dosd: f64, aov: f64, omega_opt: f64,
               mol: Molecule, obs: PreparedBasis, dfbs: PreparedBasis, obs_bs: ferric_core::basis::BasisSet, rhf: ferric_scf::ScfResult }
    let mut data: Vec<M> = Vec::new();
    for (label, xyz, dosd) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let scf_cfg = RhfConfig { energy_conv: 1e-9, xc: Some(lc_name.to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()), ..Default::default() };
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();
        let alpha = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, Operator::coulomb(), &cfg).unwrap().iso;
        let vol: f64 = atomic_effective_volumes_becke(&mol, &obs, &obs_bs, rhf.density_r()).unwrap().iter().sum();
        let aov = alpha / vol;
        // ω_opt by bisection (C6 decreasing in ω).
        let c6_at = |w: f64, m:&Molecule, o:&PreparedBasis, ob:&ferric_core::basis::BasisSet, d:&PreparedBasis, r:&ferric_scf::ScfResult| {
            let dp = pdep_dynamic_polarizability(m,o,ob,d,r,Operator::erf(w),&cfg,DispersionPartition::Becke,None).unwrap();
            casimir_polder_c6(&dp).c6_molecular_iso
        };
        let (mut lo, mut hi) = (0.05_f64, 2.5_f64);
        let omega_opt = if c6_at(lo,&mol,&obs,&obs_bs,&dfbs,&rhf) < *dosd { f64::NAN }
            else { for _ in 0..22 { let m=0.5*(lo+hi); if c6_at(m,&mol,&obs,&obs_bs,&dfbs,&rhf) > *dosd {lo=m} else {hi=m} } 0.5*(lo+hi) };
        data.push(M{label:label.to_string(),dosd:*dosd,aov,omega_opt,mol,obs,dfbs,obs_bs,rhf});
    }

    // Pass 2: LOO — predict ω for each from the linear fit on the others, get C6.
    eprintln!("\n=== LOO dielectric-ω RSH-RPA C6 (ω=a·(α/V)+b fit on others, no in-sample) ===");
    eprintln!("  {:>5}  {:>7}  {:>9}  {:>9}  {:>9}  {:>8}", "mol", "α/V", "ω_opt", "ω_pred", "C6_pred", "err%");
    let fit = |idx: usize| -> (f64, f64) {
        let (mut sx, mut sy, mut sxx, mut sxy, mut n) = (0.0,0.0,0.0,0.0,0.0);
        for (j,m) in data.iter().enumerate() {
            if j==idx || !m.omega_opt.is_finite() { continue; }
            sx+=m.aov; sy+=m.omega_opt; sxx+=m.aov*m.aov; sxy+=m.aov*m.omega_opt; n+=1.0;
        }
        let slope = (n*sxy - sx*sy)/(n*sxx - sx*sx);
        (slope, (sy - slope*sx)/n)
    };
    let (mut mae, mut cnt) = (0.0, 0);
    for i in 0..data.len() {
        let (s,b) = fit(i);
        let w_pred = (s*data[i].aov + b).clamp(0.05, 2.5);
        let dp = pdep_dynamic_polarizability(&data[i].mol,&data[i].obs,&data[i].obs_bs,&data[i].dfbs,&data[i].rhf,
            Operator::erf(w_pred),&cfg,DispersionPartition::Becke,None).unwrap();
        let c6 = casimir_polder_c6(&dp).c6_molecular_iso;
        let err = 100.0*(c6-data[i].dosd)/data[i].dosd;
        eprintln!("  {:>5}  {:>7.4}  {:>9.4}  {:>9.4}  {:>9.2}  {:>+7.2}", data[i].label, data[i].aov, data[i].omega_opt, w_pred, c6, err);
        mae += err.abs(); cnt += 1;
    }
    eprintln!("\n  LOO dielectric-ω C6 MAE = {:.2}%   (fixed-ω was 12.4%; scalar-fix 2.4%)", mae/cnt as f64);
    eprintln!("  Beats 12.4% ⟹ dielectric-ω transfers (novelty real). Else ⟹ correlation ≠ accuracy.");
}

/// COST/CONSISTENCY TEST (Matt's objection): does the C6 actually depend on the
/// REFERENCE's ω, or only on the CORRELATION ω? If C6 is ~flat vs reference-ω at
/// fixed correlation-ω, then the cheap construction (one fixed reference SCF, vary
/// only the post-SCF RPA correlation ω) is justified — NO self-consistent SCF loop.
/// If C6 moves a lot with reference-ω, we'd be forced into the expensive iterated
/// SCF (recompute the density every ω).
///
/// For water + n2: fix correlation ω at a few values, sweep the REFERENCE LC ω
/// (by using different LC functionals as ω-proxies isn't clean, so we instead hold
/// ONE reference and report C6 sensitivity to reference choice at matched vs
/// mismatched correlation ω).
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        reference_omega_sensitivity -- --ignored --nocapture
#[test]
#[ignore]
fn reference_omega_sensitivity() {
    // Reference functionals with DIFFERENT native ω (0.40, 0.30, 0.20) but all
    // valid RSH (c_lr=1.0). Hold the CORRELATION ω fixed; see how much C6 moves
    // with the reference. Small spread ⟹ cheap fixed-reference construction is OK.
    let refs = [
        ("LC-ωPBE  ω=0.40", "HYB_GGA_XC_LC_WPBE"),
        ("LRC-ωPBE ω=0.30", "HYB_GGA_XC_LRC_WPBE"),
        ("LRC-ωPBEh ω=0.20", "HYB_GGA_XC_LRC_WPBEH"),
    ];
    let mols: &[(&str, &str)] = &[
        ("h2o", "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n"),
        ("n2",  "2\nn2\nN 0 0 0.0\nN 0 0 1.0977\n"),
    ];
    let corr_omegas = [0.20_f64, 0.35, 0.50]; // fixed correlation ω values
    let ctx = ParallelContext::default();
    let cfg = PdepRpaConfig::default();

    eprintln!("\n=== C6 sensitivity to REFERENCE ω at FIXED correlation ω ===");
    eprintln!("  (flat across references ⟹ cheap fixed-reference OK, no SCF-in-loop)\n");
    for (mlabel, xyz) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();

        eprintln!("  --- {mlabel} ---");
        eprint!("  {:>18}", "reference \\ corr ω");
        for w in &corr_omegas { eprint!("   ω={:.2}", w); }
        eprintln!();
        for (rlabel, rname) in &refs {
            let scf_cfg = RhfConfig { energy_conv: 1e-9, xc: Some(rname.to_string()),
                df_j_aux: Some("def2-universal-jkfit".to_string()),
                df_k_aux: Some("def2-universal-jkfit".to_string()), ..Default::default() };
            let rhf = match solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg) {
                Ok(r) if r.converged => r, _ => { eprintln!("  {rlabel:>18}  (no SCF)"); continue; } };
            eprint!("  {:>18}", rlabel);
            for &cw in &corr_omegas {
                let dp = pdep_dynamic_polarizability(&mol,&obs,&obs_bs,&dfbs,&rhf,Operator::erf(cw),&cfg,DispersionPartition::Becke,None).unwrap();
                eprint!("  {:>6.1}", casimir_polder_c6(&dp).c6_molecular_iso);
            }
            eprintln!();
        }
        eprintln!();
    }
    eprintln!("  Read each COLUMN: spread down a column = C6 sensitivity to the reference");
    eprintln!("  at that fixed correlation ω. Small ⟹ reference-ω decouples from C6.");
}

/// REPLACE Clausius-Mossotti: does the COMPUTED dielectric spectrum carry the
/// dynamic range that scalar α/V loses? CM catastrophes at molecular density;
/// scalar non-local-field ε (1+4π α/V) fixes the pole but COMPRESSES the range
/// (ε 3.2–4.8 can't reproduce ω_opt's 4.6× span). Hypothesis: the dielectric
/// EIGENVALUE structure (anisotropy / soft modes) distinguishes N₂ from CH₄ where
/// the scalar can't. Pull λ_max, top-3 sum, trace-log, and correlate vs ω_opt.
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        dielectric_eigenvalue_descriptors -- --ignored --nocapture
#[test]
#[ignore]
fn dielectric_eigenvalue_descriptors() {
    use ferric_rpa::properties::{atomic_effective_volumes_becke, dielectric_spectrum_static, pdep_polarizability_static};

    let lc_name = "HYB_GGA_XC_LC_WPBE";
    // ω_opt from the calibration (RSH-RPA C6=DOSD on LC-ωPBE ref).
    let mols: &[(&str, &str, f64)] = &[
        ("h2o",  "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n", 0.1759),
        ("ch4",  "5\nch4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n", 0.1255),
        ("nh3",  "4\nnh3\nN 0 0 0.0\nH 0 0.9377 -0.3816\nH 0.8120 -0.4689 -0.3816\nH -0.8120 -0.4689 -0.3816\n", 0.1035),
        ("co",   "2\nco\nC 0 0 0\nO 0 0 1.128\n", 0.3651),
        ("n2",   "2\nn2\nN 0 0 0.0\nN 0 0 1.0977\n", 0.4640),
        ("co2",  "3\nco2\nC 0 0 0.0\nO 0 0 1.1621\nO 0 0 -1.1621\n", 0.3712),
        ("c2h4", "6\nc2h4\nC 0 0 0.6695\nC 0 0 -0.6695\nH 0 0.9289 1.2321\nH 0 -0.9289 1.2321\nH 0 0.9289 -1.2321\nH 0 -0.9289 -1.2321\n", 0.2267),
    ];
    let ctx = ParallelContext::default();
    let cfg = PdepRpaConfig::default();

    eprintln!("\n=== Dielectric eigenvalue descriptors vs ω_opt (replace CM) ===");
    eprintln!("  {:>5}  {:>8}  {:>8}  {:>8}  {:>9}  {:>9}  {:>8}", "mol", "λ_max", "top3", "trace_ln", "aniso", "α/V", "ω_opt");

    let mut lmax=vec![]; let mut top3=vec![]; let mut tln=vec![]; let mut aniso=vec![]; let mut aov=vec![]; let mut wopt=vec![];
    for (label, xyz, wo) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let scf_cfg = RhfConfig { energy_conv: 1e-9, xc: Some(lc_name.to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()), ..Default::default() };
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();

        // Static dielectric spectrum (full Coulomb). λ are screening eigenvalues (>1).
        let spec = dielectric_spectrum_static(&mol, &obs, &dfbs, &rhf, Operator::coulomb(), 1e-6, None).unwrap();
        let mut ev = spec.eigenvalues.clone();
        ev.sort_by(|a,b| b.partial_cmp(a).unwrap()); // descending
        let l1 = ev[0];
        let t3: f64 = ev.iter().take(3).sum();
        let trace_ln = spec.trace_log;

        // α tensor anisotropy (the shape descriptor scalar α/V loses).
        let at = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, Operator::coulomb(), &cfg).unwrap();
        let d = [at.tensor[0][0], at.tensor[1][1], at.tensor[2][2]];
        let amean = (d[0]+d[1]+d[2])/3.0;
        let an = ((d[0]-amean).powi(2)+(d[1]-amean).powi(2)+(d[2]-amean).powi(2)).sqrt()/amean; // rel anisotropy
        let vol: f64 = atomic_effective_volumes_becke(&mol,&obs,&obs_bs,rhf.density_r()).unwrap().iter().sum();
        let av = at.iso/vol;

        eprintln!("  {:>5}  {:>8.3}  {:>8.2}  {:>8.3}  {:>9.4}  {:>9.4}  {:>8.4}", label, l1, t3, trace_ln, an, av, wo);
        lmax.push(l1); top3.push(t3); tln.push(trace_ln); aniso.push(an); aov.push(av); wopt.push(*wo);
    }

    let pear = |x:&[f64], y:&[f64]| -> f64 {
        let n=x.len() as f64; let mx=x.iter().sum::<f64>()/n; let my=y.iter().sum::<f64>()/n;
        let sxy:f64=x.iter().zip(y).map(|(a,b)|(a-mx)*(b-my)).sum();
        let sxx:f64=x.iter().map(|a|(a-mx).powi(2)).sum();
        let syy:f64=y.iter().map(|b|(b-my).powi(2)).sum();
        sxy/(sxx.sqrt()*syy.sqrt())
    };
    eprintln!("\n  Pearson r vs ω_opt:");
    eprintln!("    λ_max     = {:+.3}", pear(&lmax,&wopt));
    eprintln!("    top3      = {:+.3}", pear(&top3,&wopt));
    eprintln!("    trace_ln  = {:+.3}", pear(&tln,&wopt));
    eprintln!("    aniso(α)  = {:+.3}", pear(&aniso,&wopt));
    eprintln!("    α/V       = {:+.3}  (scalar baseline)", pear(&aov,&wopt));
    eprintln!("  Any |r| clearly beating α/V's = the descriptor that carries the lost range.");
}

/// DECISIVE: leave-one-out C6 with ω = a·λ_max + b (λ_max = largest computed
/// dielectric eigenvalue, the CM replacement). Mirrors dielectric_omega_loo_c6
/// exactly but uses λ_max instead of α/V. Beats the α/V LOO MAE (6.6%)? → λ_max's
/// tighter descriptor correlation (r=0.97 vs 0.85) translates to better C6, and
/// λ_max is the right nonempirical screening descriptor (computed dielectric, no CM).
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        lambda_max_loo_c6 -- --ignored --nocapture
#[test]
#[ignore]
fn lambda_max_loo_c6() {
    use ferric_rpa::properties::dielectric_spectrum_static;

    let lc_name = "HYB_GGA_XC_LC_WPBE";
    let mols: &[(&str, &str, f64)] = &[
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

    struct M { label: String, dosd: f64, lmax: f64, omega_opt: f64,
               mol: Molecule, obs: PreparedBasis, dfbs: PreparedBasis, obs_bs: ferric_core::basis::BasisSet, rhf: ferric_scf::ScfResult }
    let mut data: Vec<M> = Vec::new();
    for (label, xyz, dosd) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let scf_cfg = RhfConfig { energy_conv: 1e-9, xc: Some(lc_name.to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()), ..Default::default() };
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();

        // λ_max of the static Coulomb dielectric.
        let spec = dielectric_spectrum_static(&mol, &obs, &dfbs, &rhf, Operator::coulomb(), 1e-6, None).unwrap();
        let lmax = spec.eigenvalues.iter().cloned().fold(f64::MIN, f64::max);

        // ω_opt by bisection (erf-RPA C6 decreasing in ω).
        let c6_at = |w: f64, m:&Molecule, o:&PreparedBasis, ob:&ferric_core::basis::BasisSet, d:&PreparedBasis, r:&ferric_scf::ScfResult| {
            let dp = pdep_dynamic_polarizability(m,o,ob,d,r,Operator::erf(w),&cfg,DispersionPartition::Becke,None).unwrap();
            casimir_polder_c6(&dp).c6_molecular_iso
        };
        let (mut lo, mut hi) = (0.05_f64, 2.5_f64);
        let omega_opt = if c6_at(lo,&mol,&obs,&obs_bs,&dfbs,&rhf) < *dosd { f64::NAN }
            else { for _ in 0..22 { let m=0.5*(lo+hi); if c6_at(m,&mol,&obs,&obs_bs,&dfbs,&rhf) > *dosd {lo=m} else {hi=m} } 0.5*(lo+hi) };
        data.push(M{label:label.to_string(),dosd:*dosd,lmax,omega_opt,mol,obs,dfbs,obs_bs,rhf});
    }

    eprintln!("\n=== LOO C6 with ω = a·λ_max + b (computed dielectric, replaces CM) ===");
    eprintln!("  {:>5}  {:>7}  {:>9}  {:>9}  {:>9}  {:>8}", "mol", "λ_max", "ω_opt", "ω_pred", "C6_pred", "err%");
    let fit = |idx: usize| -> (f64, f64) {
        let (mut sx,mut sy,mut sxx,mut sxy,mut n)=(0.0,0.0,0.0,0.0,0.0);
        for (j,m) in data.iter().enumerate() {
            if j==idx || !m.omega_opt.is_finite() { continue; }
            sx+=m.lmax; sy+=m.omega_opt; sxx+=m.lmax*m.lmax; sxy+=m.lmax*m.omega_opt; n+=1.0;
        }
        let slope=(n*sxy-sx*sy)/(n*sxx-sx*sx);
        (slope,(sy-slope*sx)/n)
    };
    let (mut mae, mut cnt)=(0.0,0);
    for i in 0..data.len() {
        let (s,b)=fit(i);
        let w_pred=(s*data[i].lmax+b).clamp(0.05,2.5);
        let dp=pdep_dynamic_polarizability(&data[i].mol,&data[i].obs,&data[i].obs_bs,&data[i].dfbs,&data[i].rhf,
            Operator::erf(w_pred),&cfg,DispersionPartition::Becke,None).unwrap();
        let c6=casimir_polder_c6(&dp).c6_molecular_iso;
        let err=100.0*(c6-data[i].dosd)/data[i].dosd;
        eprintln!("  {:>5}  {:>7.3}  {:>9.4}  {:>9.4}  {:>9.2}  {:>+7.2}", data[i].label, data[i].lmax, data[i].omega_opt, w_pred, c6, err);
        mae+=err.abs(); cnt+=1;
    }
    eprintln!("\n  λ_max LOO C6 MAE = {:.2}%   (α/V LOO 6.6%; fixed-ω 12.4%; scalar 2.4%)", mae/cnt as f64);
    eprintln!("  Beats 6.6% ⟹ computed-dielectric λ_max is the better screening descriptor.");
}

/// PARAMETER-FREE TEST: ω = (1/3)(λ_max − 2), ZERO fitted constants. The free
/// linear fit landed at slope 0.338≈1/3, intercept −0.684≈−2/3 (= −(1/3)·2), i.e.
/// ω ≈ (1/3)(λ_max−2). Matt spotted the round numbers; pinning them costs almost
/// nothing on the ω fit (MAE 0.0273 vs 0.0263 free). Does the parameter-FREE form
/// give the same C6 MAE (~2.4%)? If yes, the screening law is nonempirical — the
/// 1/3 and the λ₀=2 threshold are physical, not fitted.
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        parameter_free_lambda_omega_c6 -- --ignored --nocapture
#[test]
#[ignore]
fn parameter_free_lambda_omega_c6() {
    use ferric_rpa::properties::dielectric_spectrum_static;

    let lc_name = "HYB_GGA_XC_LC_WPBE";
    let mols: &[(&str, &str, f64)] = &[
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

    eprintln!("\n=== PARAMETER-FREE  ω = (1/3)(λ_max − 2)  →  C6 vs DOSD ===");
    eprintln!("  (zero fitted constants; compare to LOO-fit 2.4%, scalar 2.4%, fixed-ω 12.4%)\n");
    eprintln!("  {:>5}  {:>7}  {:>8}  {:>9}  {:>8}", "mol", "λ_max", "ω_pf", "C6_pf", "err%");
    let (mut mae, mut cnt) = (0.0, 0);
    for (label, xyz, dosd) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let scf_cfg = RhfConfig { energy_conv: 1e-9, xc: Some(lc_name.to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()), ..Default::default() };
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();

        let spec = dielectric_spectrum_static(&mol, &obs, &dfbs, &rhf, Operator::coulomb(), 1e-6, None).unwrap();
        let lmax = spec.eigenvalues.iter().cloned().fold(f64::MIN, f64::max);

        // PARAMETER-FREE omega. Clamp to physical (>0) — CH4 with lmax 2.23 gives 0.078.
        let w_pf = ((1.0/3.0)*(lmax - 2.0)).clamp(0.02, 2.5);
        let dp = pdep_dynamic_polarizability(&mol,&obs,&obs_bs,&dfbs,&rhf,Operator::erf(w_pf),&cfg,DispersionPartition::Becke,None).unwrap();
        let c6 = casimir_polder_c6(&dp).c6_molecular_iso;
        let err = 100.0*(c6-dosd)/dosd;
        eprintln!("  {:>5}  {:>7.3}  {:>8.4}  {:>9.2}  {:>+7.2}", label, lmax, w_pf, c6, err);
        mae += err.abs(); cnt += 1;
    }
    eprintln!("\n  PARAMETER-FREE (1/3)(λ_max−2) C6 MAE = {:.2}%", mae/cnt as f64);
    eprintln!("  ≈2.4% with ZERO fitted constants ⟹ the 1/3 and λ₀=2 are physical, not fit.");
}

/// FULL DOSD SET, parameter-free ω = (1/3)(λ_max − 2), ZERO fitted constants.
/// The decisive test: does the law found on 7 molecules hold on the full set,
/// including molecules never used to find it (H2, C2H2, C2H6, HF, HCl, H2S, C6H6)?
/// O2 skipped (open-shell, closed-shell path only).
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        parameter_free_full_dosd -- --ignored --nocapture
#[test]
#[ignore]
fn parameter_free_full_dosd() {
    use ferric_rpa::properties::dielectric_spectrum_static;

    // label, xyz (Å, equilibrium), DOSD molecular C6, in-original-7?
    let mols: &[(&str, &str, f64, bool)] = &[
        // --- the original calibration 7 ---
        ("h2o",  "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n", 45.3, true),
        ("ch4",  "5\nch4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n", 129.7, true),
        ("nh3",  "4\nnh3\nN 0 0 0.0\nH 0 0.9377 -0.3816\nH 0.8120 -0.4689 -0.3816\nH -0.8120 -0.4689 -0.3816\n", 89.0, true),
        ("co",   "2\nco\nC 0 0 0\nO 0 0 1.128\n", 81.4, true),
        ("n2",   "2\nn2\nN 0 0 0.0\nN 0 0 1.0977\n", 73.3, true),
        ("co2",  "3\nco2\nC 0 0 0.0\nO 0 0 1.1621\nO 0 0 -1.1621\n", 158.7, true),
        ("c2h4", "6\nc2h4\nC 0 0 0.6695\nC 0 0 -0.6695\nH 0 0.9289 1.2321\nH 0 -0.9289 1.2321\nH 0 0.9289 -1.2321\nH 0 -0.9289 -1.2321\n", 300.2, true),
        // --- NEW (never used to find the law) ---
        ("h2",   "2\nh2\nH 0 0 0\nH 0 0 0.741\n", 12.1, false),
        ("hf",   "2\nhf\nF 0 0 0\nH 0 0 0.917\n", 19.0, false),
        ("hcl",  "2\nhcl\nCl 0 0 0\nH 0 0 1.275\n", 130.4, false),
        ("h2s",  "3\nh2s\nS 0 0 0.1030\nH 0 0.9659 -0.8253\nH 0 -0.9659 -0.8253\n", 216.8, false),
        ("c2h2", "4\nc2h2\nC 0 0 0.6015\nC 0 0 -0.6015\nH 0 0 1.6615\nH 0 0 -1.6615\n", 204.1, false),
        ("c2h6", "8\nc2h6\nC 0 0 0.7680\nC 0 0 -0.7680\nH 1.0192 0 1.1573\nH -0.5096 0.8826 1.1573\nH -0.5096 -0.8826 1.1573\nH -1.0192 0 -1.1573\nH 0.5096 0.8826 -1.1573\nH 0.5096 -0.8826 -1.1573\n", 381.9, false),
        ("c6h6", "12\nc6h6\nC 0 1.3970 0\nC 1.2098 0.6985 0\nC 1.2098 -0.6985 0\nC 0 -1.3970 0\nC -1.2098 -0.6985 0\nC -1.2098 0.6985 0\nH 0 2.4810 0\nH 2.1486 1.2405 0\nH 2.1486 -1.2405 0\nH 0 -2.4810 0\nH -2.1486 -1.2405 0\nH -2.1486 1.2405 0\n", 1765.0, false),
    ];
    let ctx = ParallelContext::default();
    let cfg = PdepRpaConfig::default();
    let lc_name = "HYB_GGA_XC_LC_WPBE";

    eprintln!("\n=== FULL DOSD: parameter-free ω = (1/3)(λ_max − 2), ZERO fitted constants ===");
    eprintln!("  (law found on the 7 marked *; rest are pure prediction. O2 skipped: open-shell)\n");
    eprintln!("  {:>5} {:>3}  {:>7}  {:>8}  {:>9}  {:>9}  {:>8}", "mol", "new", "λ_max", "ω_pf", "C6_pf", "DOSD", "err%");
    let (mut mae_all, mut mae_new, mut n_all, mut n_new) = (0.0, 0.0, 0, 0);
    for (label, xyz, dosd, in7) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let scf_cfg = RhfConfig { energy_conv: 1e-9, xc: Some(lc_name.to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()), ..Default::default() };
        let rhf = match solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg) {
            Ok(r) if r.converged => r,
            _ => { eprintln!("  {label:>5}  (LC SCF not converged — skipped)"); continue; }
        };
        let spec = match dielectric_spectrum_static(&mol, &obs, &dfbs, &rhf, Operator::coulomb(), 1e-6, None) {
            Ok(s) => s, Err(e) => { eprintln!("  {label:>5}  (spectrum err: {e})"); continue; }
        };
        let lmax = spec.eigenvalues.iter().cloned().fold(f64::MIN, f64::max);
        let w_pf = ((1.0/3.0)*(lmax - 2.0)).clamp(0.02, 2.5);
        let dp = pdep_dynamic_polarizability(&mol,&obs,&obs_bs,&dfbs,&rhf,Operator::erf(w_pf),&cfg,DispersionPartition::Becke,None).unwrap();
        let c6 = casimir_polder_c6(&dp).c6_molecular_iso;
        let err = 100.0*(c6-dosd)/dosd;
        let tag = if *in7 { "" } else { "NEW" };
        eprintln!("  {:>5} {:>3}  {:>7.3}  {:>8.4}  {:>9.2}  {:>9.1}  {:>+7.2}", label, tag, lmax, w_pf, c6, dosd, err);
        mae_all += err.abs(); n_all += 1;
        if !in7 { mae_new += err.abs(); n_new += 1; }
    }
    eprintln!("\n  Parameter-free C6 MAE — ALL {} mols = {:.2}%   |   NEW-only ({}) = {:.2}%",
        n_all, mae_all/n_all as f64, n_new, mae_new/n_new.max(1) as f64);
    eprintln!("  (7-mol value was 1.74%. NEW-only is the true out-of-sample test of the law.)");
}

/// Full-DOSD calibration data dump: per-molecule λ_max, α/V, trace_log, rank,
/// AND the bisected ω_opt (RSH-RPA C6=DOSD). For fitting ω(descriptor) on the
/// whole set. Prints CSV-ish rows. O2 skipped (open-shell).
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        full_dosd_calibration_dump -- --ignored --nocapture
#[test]
#[ignore]
fn full_dosd_calibration_dump() {
    use ferric_rpa::properties::{atomic_effective_volumes_becke, dielectric_spectrum_static, pdep_polarizability_static};
    let mols: &[(&str, &str, f64)] = &[
        ("h2o",  "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n", 45.3),
        ("ch4",  "5\nch4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n", 129.7),
        ("nh3",  "4\nnh3\nN 0 0 0.0\nH 0 0.9377 -0.3816\nH 0.8120 -0.4689 -0.3816\nH -0.8120 -0.4689 -0.3816\n", 89.0),
        ("co",   "2\nco\nC 0 0 0\nO 0 0 1.128\n", 81.4),
        ("n2",   "2\nn2\nN 0 0 0.0\nN 0 0 1.0977\n", 73.3),
        ("co2",  "3\nco2\nC 0 0 0.0\nO 0 0 1.1621\nO 0 0 -1.1621\n", 158.7),
        ("c2h4", "6\nc2h4\nC 0 0 0.6695\nC 0 0 -0.6695\nH 0 0.9289 1.2321\nH 0 -0.9289 1.2321\nH 0 0.9289 -1.2321\nH 0 -0.9289 -1.2321\n", 300.2),
        ("h2",   "2\nh2\nH 0 0 0\nH 0 0 0.741\n", 12.1),
        ("hf",   "2\nhf\nF 0 0 0\nH 0 0 0.917\n", 19.0),
        ("hcl",  "2\nhcl\nCl 0 0 0\nH 0 0 1.275\n", 130.4),
        ("h2s",  "3\nh2s\nS 0 0 0.1030\nH 0 0.9659 -0.8253\nH 0 -0.9659 -0.8253\n", 216.8),
        ("c2h2", "4\nc2h2\nC 0 0 0.6015\nC 0 0 -0.6015\nH 0 0 1.6615\nH 0 0 -1.6615\n", 204.1),
        ("c2h6", "8\nc2h6\nC 0 0 0.7680\nC 0 0 -0.7680\nH 1.0192 0 1.1573\nH -0.5096 0.8826 1.1573\nH -0.5096 -0.8826 1.1573\nH -1.0192 0 -1.1573\nH 0.5096 0.8826 -1.1573\nH 0.5096 -0.8826 -1.1573\n", 381.9),
        ("c6h6", "12\nc6h6\nC 0 1.3970 0\nC 1.2098 0.6985 0\nC 1.2098 -0.6985 0\nC 0 -1.3970 0\nC -1.2098 -0.6985 0\nC -1.2098 0.6985 0\nH 0 2.4810 0\nH 2.1486 1.2405 0\nH 2.1486 -1.2405 0\nH 0 -2.4810 0\nH -2.1486 -1.2405 0\nH -2.1486 1.2405 0\n", 1765.0),
    ];
    let ctx = ParallelContext::default();
    let cfg = PdepRpaConfig::default();
    let lc = "HYB_GGA_XC_LC_WPBE";
    eprintln!("\nCSV label,lmax,top3,trace_ln,aoV,omega_opt,dosd");
    for (label, xyz, dosd) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let scf_cfg = RhfConfig { energy_conv: 1e-9, xc: Some(lc.to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()), ..Default::default() };
        let rhf = match solve_rhf(&ctx,&mol,&obs,op,&bounds,&scf_cfg) { Ok(r) if r.converged=>r, _=>{eprintln!("{label},SCF_FAIL");continue;} };
        let spec = dielectric_spectrum_static(&mol,&obs,&dfbs,&rhf,Operator::coulomb(),1e-6, None).unwrap();
        let mut ev = spec.eigenvalues.clone(); ev.sort_by(|a,b| b.partial_cmp(a).unwrap());
        let lmax = ev[0]; let top3: f64 = ev.iter().take(3).sum();
        let at = pdep_polarizability_static(&mol,&obs,&dfbs,&rhf,Operator::coulomb(),&cfg).unwrap();
        let vol: f64 = atomic_effective_volumes_becke(&mol,&obs,&obs_bs,rhf.density_r()).unwrap().iter().sum();
        let aov = at.iso/vol;
        let c6 = |w: f64| { let dp=pdep_dynamic_polarizability(&mol,&obs,&obs_bs,&dfbs,&rhf,Operator::erf(w),&cfg,DispersionPartition::Becke,None).unwrap(); casimir_polder_c6(&dp).c6_molecular_iso };
        let (mut lo,mut hi)=(0.02_f64,2.5_f64);
        let wopt = if c6(lo) < *dosd { 0.02 } else if c6(hi) > *dosd { f64::NAN }
            else { for _ in 0..22 { let m=0.5*(lo+hi); if c6(m) > *dosd {lo=m} else {hi=m} } 0.5*(lo+hi) };
        eprintln!("CSV {label},{:.4},{:.4},{:.4},{:.4},{:.4},{}", lmax, top3, spec.trace_log, aov, wopt, dosd);
    }
}

/// DIRECTIONALITY TEST (Matt's hypothesis): λ_max is scalar and loses WHICH
/// direction screens — but C6 integrates the anisotropic α tensor. Dump the α
/// PRINCIPAL VALUES (directional screening strengths) + ω_opt, to test whether a
/// direction-resolved descriptor beats scalar λ_max (0.72) and fixes the
/// multi-directional failures (benzene, H2S, HCl).
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        directional_descriptor_dump -- --ignored --nocapture
#[test]
#[ignore]
fn directional_descriptor_dump() {
    use ferric_rpa::properties::{atomic_effective_volumes_becke, dielectric_spectrum_static, pdep_polarizability_static};
    let mols: &[(&str, &str, f64)] = &[
        ("h2o",  "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n", 45.3),
        ("ch4",  "5\nch4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n", 129.7),
        ("nh3",  "4\nnh3\nN 0 0 0.0\nH 0 0.9377 -0.3816\nH 0.8120 -0.4689 -0.3816\nH -0.8120 -0.4689 -0.3816\n", 89.0),
        ("co",   "2\nco\nC 0 0 0\nO 0 0 1.128\n", 81.4),
        ("n2",   "2\nn2\nN 0 0 0.0\nN 0 0 1.0977\n", 73.3),
        ("co2",  "3\nco2\nC 0 0 0.0\nO 0 0 1.1621\nO 0 0 -1.1621\n", 158.7),
        ("c2h4", "6\nc2h4\nC 0 0 0.6695\nC 0 0 -0.6695\nH 0 0.9289 1.2321\nH 0 -0.9289 1.2321\nH 0 0.9289 -1.2321\nH 0 -0.9289 -1.2321\n", 300.2),
        ("h2",   "2\nh2\nH 0 0 0\nH 0 0 0.741\n", 12.1),
        ("hf",   "2\nhf\nF 0 0 0\nH 0 0 0.917\n", 19.0),
        ("hcl",  "2\nhcl\nCl 0 0 0\nH 0 0 1.275\n", 130.4),
        ("h2s",  "3\nh2s\nS 0 0 0.1030\nH 0 0.9659 -0.8253\nH 0 -0.9659 -0.8253\n", 216.8),
        ("c2h2", "4\nc2h2\nC 0 0 0.6015\nC 0 0 -0.6015\nH 0 0 1.6615\nH 0 0 -1.6615\n", 204.1),
        ("c2h6", "8\nc2h6\nC 0 0 0.7680\nC 0 0 -0.7680\nH 1.0192 0 1.1573\nH -0.5096 0.8826 1.1573\nH -0.5096 -0.8826 1.1573\nH -1.0192 0 -1.1573\nH 0.5096 0.8826 -1.1573\nH 0.5096 -0.8826 -1.1573\n", 381.9),
        ("c6h6", "12\nc6h6\nC 0 1.3970 0\nC 1.2098 0.6985 0\nC 1.2098 -0.6985 0\nC 0 -1.3970 0\nC -1.2098 -0.6985 0\nC -1.2098 0.6985 0\nH 0 2.4810 0\nH 2.1486 1.2405 0\nH 2.1486 -1.2405 0\nH 0 -2.4810 0\nH -2.1486 -1.2405 0\nH -2.1486 1.2405 0\n", 1765.0),
    ];
    let ctx = ParallelContext::default();
    let cfg = PdepRpaConfig::default();
    let lc = "HYB_GGA_XC_LC_WPBE";
    eprintln!("\nCSV label,a1,a2,a3,V,lmax,omega_opt");  // a1>=a2>=a3 principal alphas
    for (label, xyz, dosd) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let scf_cfg = RhfConfig { energy_conv: 1e-9, xc: Some(lc.to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()), ..Default::default() };
        let rhf = match solve_rhf(&ctx,&mol,&obs,op,&bounds,&scf_cfg){Ok(r) if r.converged=>r,_=>{eprintln!("{label},SCF_FAIL");continue;}};
        let at = pdep_polarizability_static(&mol,&obs,&dfbs,&rhf,Operator::coulomb(),&cfg).unwrap();
        let mut pr = at.principal; pr.sort_by(|a,b| b.partial_cmp(a).unwrap()); // desc
        let spec = dielectric_spectrum_static(&mol,&obs,&dfbs,&rhf,Operator::coulomb(),1e-6, None).unwrap();
        let lmax = spec.eigenvalues.iter().cloned().fold(f64::MIN,f64::max);
        let vol: f64 = atomic_effective_volumes_becke(&mol,&obs,&obs_bs,rhf.density_r()).unwrap().iter().sum();
        let c6 = |w: f64| { let dp=pdep_dynamic_polarizability(&mol,&obs,&obs_bs,&dfbs,&rhf,Operator::erf(w),&cfg,DispersionPartition::Becke,None).unwrap(); casimir_polder_c6(&dp).c6_molecular_iso };
        let (mut lo,mut hi)=(0.02_f64,2.5_f64);
        let wopt = if c6(lo)<*dosd {0.02} else if c6(hi)>*dosd {f64::NAN}
            else { for _ in 0..22 { let m=0.5*(lo+hi); if c6(m)>*dosd {lo=m} else {hi=m} } 0.5*(lo+hi) };
        eprintln!("CSV {label},{:.4},{:.4},{:.4},{:.3},{:.4},{:.4}", pr[0],pr[1],pr[2],vol,lmax,wopt);
    }
}

/// LOO-C6 with ω = a·(a3/V) + b (a3 = WEAKEST-direction α principal value). Tests
/// whether the directional descriptor (Matt: λ_max loses directionality; a3/V
/// captures perpendicular screening, r_fit=−0.83) gives low out-of-sample C6 MAE on
/// the FULL fittable set (drop hf,h2 = floored/unfittable). The honest test of the
/// directionality hypothesis — beating the fixed-ω 12.4% on the full set would be the
/// first non-overfit win.
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        a3_directional_loo_c6 -- --ignored --nocapture
#[test]
#[ignore]
fn a3_directional_loo_c6() {
    use ferric_rpa::properties::{atomic_effective_volumes_becke, pdep_polarizability_static};
    // fittable set only (hf,h2 floored — C6 overshoots at ω→0)
    let mols: &[(&str, &str, f64)] = &[
        ("h2o",  "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n", 45.3),
        ("ch4",  "5\nch4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n", 129.7),
        ("nh3",  "4\nnh3\nN 0 0 0.0\nH 0 0.9377 -0.3816\nH 0.8120 -0.4689 -0.3816\nH -0.8120 -0.4689 -0.3816\n", 89.0),
        ("co",   "2\nco\nC 0 0 0\nO 0 0 1.128\n", 81.4),
        ("n2",   "2\nn2\nN 0 0 0.0\nN 0 0 1.0977\n", 73.3),
        ("co2",  "3\nco2\nC 0 0 0.0\nO 0 0 1.1621\nO 0 0 -1.1621\n", 158.7),
        ("c2h4", "6\nc2h4\nC 0 0 0.6695\nC 0 0 -0.6695\nH 0 0.9289 1.2321\nH 0 -0.9289 1.2321\nH 0 0.9289 -1.2321\nH 0 -0.9289 -1.2321\n", 300.2),
        ("hcl",  "2\nhcl\nCl 0 0 0\nH 0 0 1.275\n", 130.4),
        ("h2s",  "3\nh2s\nS 0 0 0.1030\nH 0 0.9659 -0.8253\nH 0 -0.9659 -0.8253\n", 216.8),
        ("c2h2", "4\nc2h2\nC 0 0 0.6015\nC 0 0 -0.6015\nH 0 0 1.6615\nH 0 0 -1.6615\n", 204.1),
        ("c2h6", "8\nc2h6\nC 0 0 0.7680\nC 0 0 -0.7680\nH 1.0192 0 1.1573\nH -0.5096 0.8826 1.1573\nH -0.5096 -0.8826 1.1573\nH -1.0192 0 -1.1573\nH 0.5096 0.8826 -1.1573\nH 0.5096 -0.8826 -1.1573\n", 381.9),
        ("c6h6", "12\nc6h6\nC 0 1.3970 0\nC 1.2098 0.6985 0\nC 1.2098 -0.6985 0\nC 0 -1.3970 0\nC -1.2098 -0.6985 0\nC -1.2098 0.6985 0\nH 0 2.4810 0\nH 2.1486 1.2405 0\nH 2.1486 -1.2405 0\nH 0 -2.4810 0\nH -2.1486 -1.2405 0\nH -2.1486 1.2405 0\n", 1765.0),
    ];
    let ctx = ParallelContext::default();
    let cfg = PdepRpaConfig::default();
    let lc = "HYB_GGA_XC_LC_WPBE";

    struct M { label:String, dosd:f64, x:f64, wopt:f64,
               mol:Molecule, obs:PreparedBasis, dfbs:PreparedBasis, obs_bs:ferric_core::basis::BasisSet, rhf:ferric_scf::ScfResult }
    let mut data:Vec<M>=Vec::new();
    for (label,xyz,dosd) in mols {
        let mol=Molecule::parse_xyz(xyz,0,1).unwrap();
        let obs_bs=basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs=PreparedBasis::new(&mol,&basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let obs=PreparedBasis::new(&mol,&obs_bs).unwrap();
        let op=Operator::coulomb(); let bounds=SchwarzBounds::compute(op,&obs).unwrap();
        let scf_cfg=RhfConfig{energy_conv:1e-9,xc:Some(lc.to_string()),
            df_j_aux:Some("def2-universal-jkfit".to_string()),
            df_k_aux:Some("def2-universal-jkfit".to_string()),..Default::default()};
        let rhf=solve_rhf(&ctx,&mol,&obs,op,&bounds,&scf_cfg).unwrap();
        let at=pdep_polarizability_static(&mol,&obs,&dfbs,&rhf,Operator::coulomb(),&cfg).unwrap();
        let a3=at.principal.iter().cloned().fold(f64::MAX,f64::min); // smallest principal
        let vol:f64=atomic_effective_volumes_becke(&mol,&obs,&obs_bs,rhf.density_r()).unwrap().iter().sum();
        let x=a3/vol;
        let c6=|w:f64,m:&Molecule,o:&PreparedBasis,ob:&ferric_core::basis::BasisSet,d:&PreparedBasis,r:&ferric_scf::ScfResult|{
            let dp=pdep_dynamic_polarizability(m,o,ob,d,r,Operator::erf(w),&cfg,DispersionPartition::Becke,None).unwrap();
            casimir_polder_c6(&dp).c6_molecular_iso };
        let (mut lo,mut hi)=(0.02_f64,2.5_f64);
        let wopt=if c6(lo,&mol,&obs,&obs_bs,&dfbs,&rhf)<*dosd {0.02} else { for _ in 0..22 {let m=0.5*(lo+hi); if c6(m,&mol,&obs,&obs_bs,&dfbs,&rhf)>*dosd {lo=m} else {hi=m}} 0.5*(lo+hi)};
        data.push(M{label:label.to_string(),dosd:*dosd,x,wopt,mol,obs,dfbs,obs_bs,rhf});
    }
    eprintln!("\n=== LOO-C6, ω = a·(a3/V) + b  (weakest-direction descriptor, fittable set) ===");
    eprintln!("  {:>5}  {:>8}  {:>8}  {:>9}  {:>8}", "mol", "a3/V", "ω_pred", "C6", "err%");
    let fit=|idx:usize|->(f64,f64){
        let(mut sx,mut sy,mut sxx,mut sxy,mut n)=(0.,0.,0.,0.,0.);
        for(j,m)in data.iter().enumerate(){ if j==idx{continue;} sx+=m.x;sy+=m.wopt;sxx+=m.x*m.x;sxy+=m.x*m.wopt;n+=1.;}
        let s=(n*sxy-sx*sy)/(n*sxx-sx*sx); (s,(sy-s*sx)/n)};
    let(mut mae,mut cnt)=(0.,0);
    for i in 0..data.len(){
        let(s,b)=fit(i); let w=(s*data[i].x+b).clamp(0.02,2.5);
        let dp=pdep_dynamic_polarizability(&data[i].mol,&data[i].obs,&data[i].obs_bs,&data[i].dfbs,&data[i].rhf,Operator::erf(w),&cfg,DispersionPartition::Becke,None).unwrap();
        let c6=casimir_polder_c6(&dp).c6_molecular_iso; let err=100.*(c6-data[i].dosd)/data[i].dosd;
        eprintln!("  {:>5}  {:>8.4}  {:>8.4}  {:>9.2}  {:>+7.2}", data[i].label, data[i].x, w, c6, err);
        mae+=err.abs(); cnt+=1;
    }
    eprintln!("\n  a3/V LOO-C6 MAE = {:.2}%  (fittable set, n={})", mae/cnt as f64, cnt);
    eprintln!("  vs: fixed-ω 12.4%, λ_max-full ~weak, parameter-free(1/3)(λ-2) NEW-only 12%.");
}

/// VALIDATION: dump the static dielectric spectrum (Coulomb) for water + HCl so it
/// can be cross-checked against an independent PySCF build of the SAME ε̃ = I +
/// 4·L·diag(1/Δε)·Lᵀ (L = (P|Q)^{-½}(Q|ia), cc-pVDZ-RI aux). The question this
/// answers: is HCl's λ_max genuinely low (third-row screening physics) or an
/// RI/basis artifact of dielectric_spectrum_static? Also dumps orbital energies +
/// occupancy so PySCF can reproduce Δε exactly (same LC-ωPBE reference).
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        dielectric_spectrum_validation_dump -- --ignored --nocapture
#[test]
#[ignore]
fn dielectric_spectrum_validation_dump() {
    use ferric_rpa::properties::dielectric_spectrum_static;

    let lc_name = "HYB_GGA_XC_LC_WPBE";
    let mols: &[(&str, &str)] = &[
        ("h2o", "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n"),
        ("hcl", "2\nhcl\nH 0 0 0\nCl 0 0 1.2746\n"),
    ];
    let ctx = ParallelContext::default();

    for (label, xyz) in mols {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let scf_cfg = RhfConfig { energy_conv: 1e-9, xc: Some(lc_name.to_string()),
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()), ..Default::default() };
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();

        let spec = dielectric_spectrum_static(&mol, &obs, &dfbs, &rhf, Operator::coulomb(), 1e-6, None).unwrap();
        let mut ev = spec.eigenvalues.clone();
        ev.sort_by(|a,b| b.partial_cmp(a).unwrap()); // descending

        eprintln!("\n=== FERRIC dielectric spectrum: {} (aug-cc-pVDZ / cc-pVDZ-RI, LC-ωPBE) ===", label);
        eprintln!("  naux={}  rank(λ-1>1e-6)={}  trace_log={:.6}", spec.naux, spec.rank, spec.trace_log);
        eprintln!("  λ_max={:.6}  top5={:?}", ev[0],
            ev.iter().take(5).map(|x| (x*1e4).round()/1e4).collect::<Vec<_>>());
        // count modes by screening strength
        let n_gt_2 = ev.iter().filter(|&&l| l > 2.0).count();
        let n_gt_15 = ev.iter().filter(|&&l| l > 1.5).count();
        let n_gt_11 = ev.iter().filter(|&&l| l > 1.1).count();
        eprintln!("  modes>2.0={}  >1.5={}  >1.1={}", n_gt_2, n_gt_15, n_gt_11);

        // SCF eigenvalues + nocc so PySCF can rebuild Δε exactly.
        let eps = rhf.eps_r();
        let nocc = (mol.nelec() / 2) as usize;
        eprintln!("  nelec={} nocc={} nao={}", mol.nelec(), nocc, eps.len());
        eprintln!("  HOMO={:.6} LUMO={:.6} gap={:.6}", eps[nocc-1], eps[nocc], eps[nocc]-eps[nocc-1]);
        eprintln!("  E_scf={:.8}", rhf.energy);
    }
}

/// SPIKE (a): LOO-C6 with the MODE-SUMMED screening descriptor Σ(λ−1)/λ, the fix the
/// PySCF validation pointed at (λ_max is a peak detector and under-reads third-row
/// broadband screening; Σ(λ−1)/λ ranks HCl>H2O correctly). Mirrors a3_directional_loo_c6
/// EXACTLY (same set, same ω_opt bisection, same LOO linear fit, same C6) so the only
/// variable is the descriptor. Tests three forms: raw Σ(λ−1)/λ (extensive),
/// Σ(λ−1)/λ / nocc (size-normalized), and trace_log. Beats a3/V's 4.94%?
///
/// Run: cargo test --release -p ferric-rpa --test rsh_rpa_dispersion \
///        screen_sum_loo_c6 -- --ignored --nocapture
#[test]
#[ignore]
fn screen_sum_loo_c6() {
    use ferric_rpa::properties::{atomic_effective_volumes_becke, dielectric_spectrum_static, pdep_polarizability_static};
    let mols: &[(&str, &str, f64)] = &[
        ("h2o",  "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n", 45.3),
        ("ch4",  "5\nch4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n", 129.7),
        ("nh3",  "4\nnh3\nN 0 0 0.0\nH 0 0.9377 -0.3816\nH 0.8120 -0.4689 -0.3816\nH -0.8120 -0.4689 -0.3816\n", 89.0),
        ("co",   "2\nco\nC 0 0 0\nO 0 0 1.128\n", 81.4),
        ("n2",   "2\nn2\nN 0 0 0.0\nN 0 0 1.0977\n", 73.3),
        ("co2",  "3\nco2\nC 0 0 0.0\nO 0 0 1.1621\nO 0 0 -1.1621\n", 158.7),
        ("c2h4", "6\nc2h4\nC 0 0 0.6695\nC 0 0 -0.6695\nH 0 0.9289 1.2321\nH 0 -0.9289 1.2321\nH 0 0.9289 -1.2321\nH 0 -0.9289 -1.2321\n", 300.2),
        ("hcl",  "2\nhcl\nCl 0 0 0\nH 0 0 1.275\n", 130.4),
        ("h2s",  "3\nh2s\nS 0 0 0.1030\nH 0 0.9659 -0.8253\nH 0 -0.9659 -0.8253\n", 216.8),
        ("c2h2", "4\nc2h2\nC 0 0 0.6015\nC 0 0 -0.6015\nH 0 0 1.6615\nH 0 0 -1.6615\n", 204.1),
        ("c2h6", "8\nc2h6\nC 0 0 0.7680\nC 0 0 -0.7680\nH 1.0192 0 1.1573\nH -0.5096 0.8826 1.1573\nH -0.5096 -0.8826 1.1573\nH -1.0192 0 -1.1573\nH 0.5096 0.8826 -1.1573\nH 0.5096 -0.8826 -1.1573\n", 381.9),
        ("c6h6", "12\nc6h6\nC 0 1.3970 0\nC 1.2098 0.6985 0\nC 1.2098 -0.6985 0\nC 0 -1.3970 0\nC -1.2098 -0.6985 0\nC -1.2098 0.6985 0\nH 0 2.4810 0\nH 2.1486 1.2405 0\nH 2.1486 -1.2405 0\nH 0 -2.4810 0\nH -2.1486 -1.2405 0\nH -2.1486 1.2405 0\n", 1765.0),
    ];
    let ctx = ParallelContext::default();
    let cfg = PdepRpaConfig::default();
    let lc = "HYB_GGA_XC_LC_WPBE";

    struct M { label:String, dosd:f64, ssum:f64, ssum_n:f64, tlog:f64, wopt:f64,
               mol:Molecule, obs:PreparedBasis, dfbs:PreparedBasis, obs_bs:ferric_core::basis::BasisSet, rhf:ferric_scf::ScfResult }
    let mut data:Vec<M>=Vec::new();
    for (label,xyz,dosd) in mols {
        let mol=Molecule::parse_xyz(xyz,0,1).unwrap();
        let obs_bs=basis::bundled("aug-cc-pvdz").unwrap();
        let dfbs=PreparedBasis::new(&mol,&basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let obs=PreparedBasis::new(&mol,&obs_bs).unwrap();
        let op=Operator::coulomb(); let bounds=SchwarzBounds::compute(op,&obs).unwrap();
        let scf_cfg=RhfConfig{energy_conv:1e-9,xc:Some(lc.to_string()),
            df_j_aux:Some("def2-universal-jkfit".to_string()),
            df_k_aux:Some("def2-universal-jkfit".to_string()),..Default::default()};
        let rhf=solve_rhf(&ctx,&mol,&obs,op,&bounds,&scf_cfg).unwrap();
        let _at=pdep_polarizability_static(&mol,&obs,&dfbs,&rhf,Operator::coulomb(),&cfg).unwrap();
        let spec=dielectric_spectrum_static(&mol,&obs,&dfbs,&rhf,Operator::coulomb(),1e-6, None).unwrap();
        // mode-summed physical screening Σ(λ-1)/λ  (only λ>1 contribute meaningfully)
        let ssum:f64=spec.eigenvalues.iter().map(|&l| if l>1.0 {(l-1.0)/l} else {0.0}).sum();
        let nocc=(mol.nelec()/2) as f64;
        let ssum_n=ssum/nocc;
        let tlog=spec.trace_log;
        let _vol:f64=atomic_effective_volumes_becke(&mol,&obs,&obs_bs,rhf.density_r()).unwrap().iter().sum();
        let c6=|w:f64,m:&Molecule,o:&PreparedBasis,ob:&ferric_core::basis::BasisSet,d:&PreparedBasis,r:&ferric_scf::ScfResult|{
            let dp=pdep_dynamic_polarizability(m,o,ob,d,r,Operator::erf(w),&cfg,DispersionPartition::Becke,None).unwrap();
            casimir_polder_c6(&dp).c6_molecular_iso };
        let (mut lo,mut hi)=(0.02_f64,2.5_f64);
        let wopt=if c6(lo,&mol,&obs,&obs_bs,&dfbs,&rhf)<*dosd {0.02} else { for _ in 0..22 {let m=0.5*(lo+hi); if c6(m,&mol,&obs,&obs_bs,&dfbs,&rhf)>*dosd {lo=m} else {hi=m}} 0.5*(lo+hi)};
        eprintln!("CSV {label},ssum={:.4},ssum/n={:.4},tlog={:.4},wopt={:.4}", ssum, ssum_n, tlog, wopt);
        data.push(M{label:label.to_string(),dosd:*dosd,ssum,ssum_n,tlog,wopt,mol,obs,dfbs,obs_bs,rhf});
    }

    // generic LOO-C6 over a chosen descriptor accessor
    let run=|name:&str, xs:&[f64], data:&[M]|{
        eprintln!("\n=== LOO-C6, ω = a·({}) + b ===", name);
        eprintln!("  {:>5}  {:>9}  {:>8}  {:>9}  {:>8}", "mol", name, "ω_pred", "C6", "err%");
        let fit=|idx:usize|->(f64,f64){
            let(mut sx,mut sy,mut sxx,mut sxy,mut n)=(0.,0.,0.,0.,0.);
            for j in 0..data.len(){ if j==idx{continue;} sx+=xs[j];sy+=data[j].wopt;sxx+=xs[j]*xs[j];sxy+=xs[j]*data[j].wopt;n+=1.;}
            let s=(n*sxy-sx*sy)/(n*sxx-sx*sx); (s,(sy-s*sx)/n)};
        let(mut mae,mut cnt)=(0.,0);
        for i in 0..data.len(){
            let(s,b)=fit(i); let w=(s*xs[i]+b).clamp(0.02,2.5);
            let dp=pdep_dynamic_polarizability(&data[i].mol,&data[i].obs,&data[i].obs_bs,&data[i].dfbs,&data[i].rhf,Operator::erf(w),&cfg,DispersionPartition::Becke,None).unwrap();
            let c6=casimir_polder_c6(&dp).c6_molecular_iso; let err=100.*(c6-data[i].dosd)/data[i].dosd;
            eprintln!("  {:>5}  {:>9.4}  {:>8.4}  {:>9.2}  {:>+7.2}", data[i].label, xs[i], w, c6, err);
            mae+=err.abs(); cnt+=1;
        }
        eprintln!("  --> {} LOO-C6 MAE = {:.2}%  (n={})", name, mae/cnt as f64, cnt);
    };
    let ssum:Vec<f64>=data.iter().map(|m|m.ssum).collect();
    let ssum_n:Vec<f64>=data.iter().map(|m|m.ssum_n).collect();
    let tlog:Vec<f64>=data.iter().map(|m|m.tlog).collect();
    run("Sum(l-1)/l", &ssum, &data);
    run("Sum(l-1)/l/nocc", &ssum_n, &data);
    run("trace_log", &tlog, &data);
    eprintln!("\n  BASELINE to beat: a3/V LOO-C6 MAE = 4.94%; fixed-ω 12.4%.");
}
