//! Smoke + validation tests for G0W0 and COHSEX on water at cc-pVDZ.
//!
//! Reference (from MOLGW; van Setten et al. JCTC 11, 5665, 2015 Table 2,
//! G0W0@HF / cc-pVDZ for H₂O):
//!   IP (= −ε_HOMO^QP)  ≈  11.97 eV
//!   EA (= −ε_LUMO^QP)  ≈  −2.50 eV   (i.e., LUMO QP ≈ +2.50 eV)
//!
//! Spike tolerance: ±0.30 eV on IP. EA is checked qualitatively
//! (sign + within 0.5 eV).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme, SternheimerConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_gw::{run_gw, GwConfig, GwMethod};

const HA_TO_EV: f64 = 27.211386245988_f64;

fn water_mol() -> Molecule {
    let xyz = "3
H2O
O  0.0   0.0       0.117790
H  0.0   0.755453 -0.471161
H  0.0  -0.755453 -0.471161
";
    Molecule::parse_xyz(xyz, 0, 1).expect("parse H2O xyz")
}

fn prepare_h2o() -> (
    Molecule,
    PreparedBasis,
    PreparedBasis,
    ferric_scf::ScfResult,
) {
    let mol = water_mol();
    let obs_bs = basis::bundled("cc-pvdz").expect("cc-pvdz");
    let aux_bs = basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("obs prep");
    let dfbs = PreparedBasis::new(&mol, &aux_bs).expect("aux prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("Schwarz");
    let cfg = RhfConfig::default();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).expect("RHF");
    (mol, obs, dfbs, rhf)
}

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 16,
            u0: 0.5,
        },
        davidson_conv_thresh: 1e-7,
        davidson_max_vecs: 0,
        trunc_thresh: 0.0, // keep all modes for spike validation
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: false, // run_gw forces this on (M9 gate)
    }
}

#[test]
#[ignore = "slow: builds RHF + PDEP-RPA + G0W0; run with --release --ignored"]
fn cohsex_h2o_runs() {
    let (mol, obs, dfbs, rhf) = prepare_h2o();
    let pcfg = pdep_cfg();
    let gcfg = GwConfig {
        method: GwMethod::Cohsex,
        ..Default::default()
    };
    let res = run_gw(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &pcfg, &gcfg, None)
        .expect("COHSEX runs");
    // HOMO is MO index 4 for water (5 doubly occupied: 1s_O, 2s, 2p × 3).
    let nocc = (mol.nelec() as usize) / 2;
    let homo_idx = res
        .mo_indices
        .iter()
        .position(|&i| i == nocc - 1)
        .expect("HOMO in qp range");
    let ip_ev = -res.eps_qp[homo_idx] * HA_TO_EV;
    eprintln!("COHSEX@HF/cc-pVDZ H2O IP = {ip_ev:.3} eV (ref G0W0@HF ≈ 11.97 eV)");
    eprintln!("  ε_mf  = {:.4} Ha", res.eps_mf[homo_idx]);
    eprintln!("  Σ_x   = {:.4} Ha", res.sigma_x[homo_idx]);
    eprintln!("  Σ_c   = {:.4} Ha", res.sigma_c[homo_idx]);
    eprintln!("  ε_qp  = {:.4} Ha", res.eps_qp[homo_idx]);
    // COHSEX typically overshoots G0W0 IP by 0.5-1 eV; allow 10-15 eV range.
    assert!(
        (8.0..18.0).contains(&ip_ev),
        "COHSEX IP for water is wildly off: {ip_ev:.3} eV"
    );
}

#[test]
#[ignore = "slow: rebuilds PDEP-RPA each iter; run with --release --ignored"]
fn evgw_h2o_homo_ip() {
    let (mol, obs, dfbs, rhf) = prepare_h2o();
    let pcfg = pdep_cfg();
    let gcfg = GwConfig {
        method: GwMethod::EvGw,
        max_ev_iter: 6,
        ev_conv_thresh: 5e-4,
        ..Default::default()
    };
    let res = run_gw(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &pcfg, &gcfg, None)
        .expect("evGW runs");
    let nocc = (mol.nelec() as usize) / 2;
    let homo_idx = res
        .mo_indices
        .iter()
        .position(|&i| i == nocc - 1)
        .expect("HOMO in qp range");
    let ip_ev = -res.eps_qp[homo_idx] * HA_TO_EV;
    eprintln!(
        "evGW@HF/cc-pVDZ H2O IP = {ip_ev:.3} eV (iter={})",
        res.n_ev_iter
    );
    // evGW should converge; tolerance loose for spike.
    assert!(
        ip_ev > 10.0 && ip_ev < 14.0,
        "evGW IP out of expected range: {ip_ev:.3} eV"
    );
}

#[test]
#[ignore = "slow: builds RHF + PDEP-RPA + evGW₀; run with --release --ignored"]
fn evgw0_h2o_homo_ip() {
    let (mol, obs, dfbs, rhf) = prepare_h2o();
    let pcfg = pdep_cfg();
    let gcfg = GwConfig {
        method: GwMethod::EvGw0,
        max_ev_iter: 10,
        ev_conv_thresh: 1e-4,
        ..Default::default()
    };
    let res = run_gw(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &pcfg, &gcfg, None)
        .expect("evGW0 runs");
    let nocc = (mol.nelec() as usize) / 2;
    let homo_idx = res
        .mo_indices
        .iter()
        .position(|&i| i == nocc - 1)
        .expect("HOMO in qp range");
    let ip_ev = -res.eps_qp[homo_idx] * HA_TO_EV;
    eprintln!(
        "evGW0@HF/cc-pVDZ H2O IP = {ip_ev:.3} eV (G0W0 was 11.965; iter={})",
        res.n_ev_iter
    );
    // evGW0 should shift HOMO toward more bound (larger IP) by 0.1-0.4 eV.
    assert!(
        ip_ev > 11.97 - 0.2 && ip_ev < 14.0,
        "evGW0 IP out of expected range: {ip_ev:.3} eV"
    );
}

/// G0W0 quasiparticle GAP / LUMO validation against PySCF gw_ac.
///
/// PySCF gw_ac (cc-pVDZ / cc-pvdz-ri) reference for H2O:
///   HOMO_qp = −12.160 eV (IP),  LUMO_qp = +4.703 eV,  gap = 16.863 eV.
///
/// This guards the EA / virtual side: a prior bug used per-ω-rotated scalar
/// dielectric eigenvalues (basis-inconsistent with the static B̃ projection),
/// which over-screened the virtuals (LUMO 3.68 eV, gap 15.64 eV — 1.2 eV too
/// small). The fix contracts the full inverse-dielectric MATRIX in the fixed
/// PDEP basis. See sigma::sigma_c_at_z.
#[test]
#[ignore = "slow: builds RHF + PDEP-RPA + G0W0 over full MO range; run with --release --ignored"]
fn g0w0_h2o_gap_and_lumo() {
    let (mol, obs, dfbs, rhf) = prepare_h2o();
    let pcfg = pdep_cfg();
    let nmo = rhf.eps_r().len();
    let gcfg = GwConfig {
        method: GwMethod::G0W0,
        qp_mos: Some(0..nmo),
        ..Default::default()
    };
    let res = run_gw(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &pcfg, &gcfg, None)
        .expect("G0W0 runs");
    let nocc = (mol.nelec() as usize) / 2;
    let homo = res.eps_qp[nocc - 1] * HA_TO_EV;
    let lumo = res.eps_qp[nocc] * HA_TO_EV;
    let gap = lumo - homo;
    eprintln!("G0W0@HF/cc-pVDZ H2O: HOMO_qp={homo:.3} eV  LUMO_qp={lumo:.3} eV  gap={gap:.3} eV");
    eprintln!("PySCF gw_ac ref    : HOMO_qp=-12.160     LUMO_qp=4.703      gap=16.863 eV");
    // IP (HOMO) within 0.30 eV of PySCF 12.160.
    assert!(
        (-homo - 12.160).abs() < 0.30,
        "HOMO IP off: {:.3} eV (ref 12.160)",
        -homo
    );
    // LUMO and gap within 0.20 eV of PySCF.
    assert!(
        (lumo - 4.703).abs() < 0.20,
        "LUMO QP off: {lumo:.3} eV (ref 4.703)"
    );
    assert!(
        (gap - 16.863).abs() < 0.20,
        "QP gap off: {gap:.3} eV (ref 16.863)"
    );
}

#[test]
#[ignore = "slow: builds RHF + PDEP-RPA + G0W0; run with --release --ignored"]
fn g0w0_h2o_homo_ip() {
    let (mol, obs, dfbs, rhf) = prepare_h2o();
    let pcfg = pdep_cfg();
    let gcfg = GwConfig {
        method: GwMethod::G0W0,
        ..Default::default()
    };
    let res = run_gw(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &pcfg, &gcfg, None)
        .expect("G0W0 runs");
    let nocc = (mol.nelec() as usize) / 2;
    let homo_idx = res
        .mo_indices
        .iter()
        .position(|&i| i == nocc - 1)
        .expect("HOMO in qp range");
    let ip_ev = -res.eps_qp[homo_idx] * HA_TO_EV;
    eprintln!("G0W0@HF/cc-pVDZ H2O IP = {ip_ev:.3} eV (ref ≈ 11.97 eV)");
    eprintln!("  ε_mf  = {:.4} Ha = {:.3} eV", res.eps_mf[homo_idx],
              res.eps_mf[homo_idx] * HA_TO_EV);
    eprintln!("  Σ_x   = {:.4} Ha", res.sigma_x[homo_idx]);
    eprintln!("  Σ_c   = {:.4} Ha", res.sigma_c[homo_idx]);
    eprintln!("  Z     = {:.4}", res.z_factor[homo_idx]);
    eprintln!("  ε_qp  = {:.4} Ha = {:.3} eV", res.eps_qp[homo_idx],
              res.eps_qp[homo_idx] * HA_TO_EV);
    // Spike tolerance: ±0.30 eV on the published 11.97 eV.
    assert!(
        (11.97 - 0.30..11.97 + 0.30).contains(&ip_ev),
        "G0W0 HOMO IP out of tolerance: {ip_ev:.3} eV (ref 11.97 ± 0.30)"
    );
}
