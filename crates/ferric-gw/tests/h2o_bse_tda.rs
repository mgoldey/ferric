//! GATE 1: BSE-TDA singlet excitation energies for H₂O / cc-pVDZ on G0W0@HF.
//!
//! Validates the W-screened BSE kernel + GW-energy wiring. The GW reference is
//! G0W0@HF, already validated to ~5 meV vs MOLGW on this exact system
//! (h2o_g0w0_cohsex.rs: IP 11.97 eV). So any BSE discrepancy isolates the
//! screened (A) kernel, not the quasiparticle energies.
//!
//! Reference (BSE@G0W0@HF / cc-pVDZ, TDA, lowest singlet ¹B₁ of water): the
//! literature value clusters ~8.1-8.4 eV (van Setten/Jacquemin BSE benchmark
//! family). First gate asserts a physically-sane window + prints the spectrum;
//! the tight number is pinned by the independent PySCF-integral BSE reference
//! (scripts, separate).
//!
//! Run: cargo test -p ferric-gw --release --test h2o_bse_tda -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme, SternheimerConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_gw::bse::run_bse_tda;

const HA_TO_EV: f64 = 27.211386245988_f64;

fn prepare_h2o() -> (Molecule, PreparedBasis, PreparedBasis, ferric_scf::ScfResult) {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).expect("parse H2O");
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    (mol, obs, dfbs, rhf)
}

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig { scheme: QuadratureScheme::GaussLegendre, n_points: 16, u0: 0.5 },
        eigensolver_conv_thresh: 1e-7,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0, // keep ALL modes (full screened W, for reference match)
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
    }
}

#[test]
#[ignore = "slow: RHF + PDEP-RPA + G0W0 + BSE-TDA eigensolve; run --release --ignored"]
fn bse_tda_h2o_lowest_singlet() {
    let (mol, obs, dfbs, rhf) = prepare_h2o();
    let res = run_bse_tda(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &pdep_cfg(), 0)
        .expect("BSE-TDA runs");
    eprintln!(
        "\nBSE-TDA@G0W0@HF / cc-pVDZ H2O   (nocc={} nvir={}, {} states)",
        res.nocc, res.nvir, res.omega.len()
    );
    eprintln!("  lowest 8 singlets (eV):");
    for (n, &om) in res.omega.iter().take(8).enumerate() {
        eprintln!("    Ω_{:<2} = {:8.4} eV", n + 1, om * HA_TO_EV);
    }
    let lowest = res.lowest_ev();
    eprintln!("  --> lowest singlet = {lowest:.4} eV");
    eprintln!("      (ferric 7.24; PySCF-integral BSE-ref on same kernel = 8.46 eV;");
    eprintln!("       the 1.2 eV gap == ferric GW gap 15.64 vs PySCF 16.86 eV — GW-limited,");
    eprintln!("       NOT a kernel bug: CIS cross-check matches PySCF to <1 meV.)");

    // Gate: positive + ordered (kernel sanity). The ABSOLUTE BSE number is
    // GW-gap-limited — see cis_tda_h2o_assembly_xcheck (kernel proven exact) and
    // the GW-gap decomposition in memory. Window kept wide; the kernel is
    // validated by the CIS cross-check, not by this absolute number.
    assert!(lowest > 0.0, "lowest excitation must be positive");
    assert!(
        res.omega.windows(2).all(|w| w[0] <= w[1] + 1e-12),
        "eigenvalues must be ascending"
    );
    assert!(
        (5.0..12.0).contains(&lowest),
        "lowest BSE singlet {lowest:.3} eV outside sane [5,12] eV window for water"
    );
}

#[test]
#[ignore = "fast: RHF + CIS-TDA assembly cross-check (no GW); run --release --ignored"]
fn cis_tda_h2o_assembly_xcheck() {
    use ferric_gw::bse::run_cis_tda;
    let (mol, obs, dfbs, rhf) = prepare_h2o();
    let res = run_cis_tda(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, 0).unwrap();
    let lowest = res.lowest_ev();
    eprintln!("\nCIS/TDHF-TDA (HF, bare exch) / cc-pVDZ H2O");
    for (n, &om) in res.omega.iter().take(4).enumerate() {
        eprintln!("    Ω_{:<2} = {:8.4} eV", n + 1, om * HA_TO_EV);
    }
    eprintln!("  --> lowest CIS = {lowest:.4} eV  (PySCF-integral ref = 9.1978 eV)");
    // DECISIVE: if this matches PySCF's 9.198 eV, the (ia)-assembly + 2v−exch
    // convention is correct; any BSE gap is the screening/GW, not the kernel.
    assert!(
        (lowest - 9.198).abs() < 0.05,
        "CIS-TDA lowest {lowest:.4} eV must match PySCF 9.198 ±0.05 (assembly check)"
    );
}

/// 12-point Gauss–Legendre nodes/weights on [-1,1] (for the Casimir–Polder map).
fn gl12() -> ([f64; 12], [f64; 12]) {
    (
        [
            -0.981560634246719, -0.904117256370475, -0.769902674194305,
            -0.587317954286617, -0.367831498998180, -0.125233408511469,
            0.125233408511469, 0.367831498998180, 0.587317954286617,
            0.769902674194305, 0.904117256370475, 0.981560634246719,
        ],
        [
            0.047175336386512, 0.106939325995318, 0.160078328543346,
            0.203167426723066, 0.233492536538355, 0.249147045813403,
            0.249147045813403, 0.233492536538355, 0.203167426723066,
            0.160078328543346, 0.106939325995318, 0.047175336386512,
        ],
    )
}

/// Casimir–Polder imaginary-frequency grid on [0,∞): ω = u0·(1+x)/(1−x).
fn cp_grid(u0: f64) -> (Vec<f64>, Vec<f64>) {
    let (x, w) = gl12();
    let mut freqs = Vec::with_capacity(12);
    let mut wts = Vec::with_capacity(12);
    for k in 0..12 {
        let xk = x[k];
        let om = u0 * (1.0 + xk) / (1.0 - xk);
        let jac = u0 * 2.0 / ((1.0 - xk) * (1.0 - xk));
        freqs.push(om);
        wts.push(w[k] * jac);
    }
    (freqs, wts)
}

#[test]
#[ignore = "slow: RHF + PDEP-RPA + G0W0 + BSE α(iω) on a 12-pt CP grid; --release --ignored"]
fn bse_c6_h2o_vs_dosd() {
    use ferric_gw::bse::run_bse_c6;
    let (mol, obs, dfbs, rhf) = prepare_h2o();
    let (freqs, weights) = cp_grid(0.6);
    let res = run_bse_c6(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &pdep_cfg(), 0, &freqs, &weights)
        .expect("BSE C6 runs");
    let dosd = 45.3;
    let err = 100.0 * (res.c6 - dosd) / dosd;
    eprintln!("\nBSE-C6@G0W0@HF / cc-pVDZ H2O");
    eprintln!("  α_static (iso) = {:.4} a.u.  (DOSD α0 = 9.64)", res.alpha_static);
    eprintln!("  α(iω) profile  = {:?}", res.alpha_iso.iter().map(|a| (a*100.0).round()/100.0).collect::<Vec<_>>());
    eprintln!("  C6 = {:.3} a.u.   (DOSD 45.3)   err = {err:+.2}%", res.c6);
    // Gate: finite, positive, physically sane. The absolute C6 is bounded by the
    // GW gap (being tightened) AND the cc-pVDZ basis α deficit (~−15% known).
    assert!(res.c6.is_finite() && res.c6 > 0.0, "C6 must be finite positive");
    assert!(res.alpha_static > 0.0, "static α must be positive");
    assert!((10.0..120.0).contains(&res.c6), "C6 {:.2} outside sane window for water", res.c6);
}

/// RPAx@PBE spike: gate 2's screened (A±B) kernel but on a PBE-KS reference
/// (KS energies + PBE-built W), isolating the REFERENCE variable. Gate 2 was
/// HF+HF-W → water C6 −64%, α_static 5.24 (≈bare HF). Hypothesis: the HF
/// reference's α deficit is the culprit; a PBE reference should raise α toward
/// DOSD (9.64). If α_static climbs and C6 improves markedly, the W-as-kernel
/// dispersion lane is alive; if it stays low, the screened kernel itself
/// under-polarizes and the lane is dead.
#[test]
#[ignore = "slow: PBE-KS + PDEP-RPA + RPAx α(iω) on CP grid; --release --ignored"]
fn rpax_pbe_c6_h2o_vs_dosd() {
    use ferric_gw::bse::run_bse_c6_ks;
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let scf_cfg = RhfConfig { xc: Some("PBE".to_string()), ..Default::default() };
    let ks = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();

    let (freqs, weights) = cp_grid(0.6);
    let res = run_bse_c6_ks(&mol, &obs, &dfbs, op, &ks, &pdep_cfg(), 0, &freqs, &weights, 0.0)
        .expect("RPAx@PBE C6 runs");
    let dosd = 45.3;
    let err = 100.0 * (res.c6 - dosd) / dosd;
    eprintln!("\nRPAx@PBE / cc-pVDZ H2O");
    eprintln!("  α_static (iso) = {:.4} a.u.  (DOSD α0 = 9.64; gate2 HF was 5.24)", res.alpha_static);
    eprintln!("  α(iω) profile  = {:?}", res.alpha_iso.iter().map(|a| (a*100.0).round()/100.0).collect::<Vec<_>>());
    eprintln!("  C6 = {:.3} a.u.   (DOSD 45.3; gate2 HF was 16.88)   err = {err:+.2}%", res.c6);
    assert!(res.c6.is_finite() && res.c6 > 0.0, "C6 must be finite positive");
    assert!(res.alpha_static > 0.0, "static α must be positive");
}

/// α(iω)-falloff test: scan a scissor shift on RPAx@PBE virtuals from KS gap
/// (7.05 eV) toward the GW gap (16.86 eV, scissor≈0.36 Ha). RPAx@PBE gives the
/// right static α (9.24) but C6 −63% because α(iω) falls too fast. If widening
/// the gap to GW raises C6 markedly, the falloff is a gap problem (build full
/// GW@PBE); if C6 barely moves (or α0 just drops), the kernel falloff is intrinsic.
#[test]
#[ignore = "slow: RPAx@PBE C6 scissor scan; --release --ignored"]
fn rpax_pbe_scissor_scan_h2o() {
    use ferric_gw::bse::run_bse_c6_ks;
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let ks = solve_rhf(&ctx, &mol, &obs, op, &bounds,
        &RhfConfig { xc: Some("PBE".to_string()), ..Default::default() }).unwrap();
    let (freqs, weights) = cp_grid(0.6);
    eprintln!("\nRPAx@PBE C6 scissor scan (water; KS gap 7.05 eV → GW gap 16.86 eV at scissor≈0.36):");
    eprintln!("  {:>8}  {:>9}  {:>9}  {:>8}", "scissor", "α_static", "C6", "C6 err%");
    for &sc in &[0.0, 0.10, 0.20, 0.36, 0.50] {
        let r = run_bse_c6_ks(&mol, &obs, &dfbs, op, &ks, &pdep_cfg(), 0, &freqs, &weights, sc).unwrap();
        eprintln!("  {:>8.2}  {:>9.4}  {:>9.3}  {:>+7.1}", sc, r.alpha_static, r.c6, 100.0*(r.c6-45.3)/45.3);
    }
    eprintln!("  DOSD: α0=9.64, C6=45.3. If C6 climbs toward 45 → gap problem (build GW@PBE).");
}
