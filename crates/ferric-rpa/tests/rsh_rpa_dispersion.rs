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
