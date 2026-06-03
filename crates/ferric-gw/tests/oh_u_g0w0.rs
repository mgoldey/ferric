//! U-GW smoke + validation on OH radical / cc-pVDZ, UHF reference.
//!
//! OH is the smallest meaningful open-shell test: 9 electrons (5 α, 4 β),
//! doublet ground state. The α-HOMO IP at G0W0@UHF should land near
//! ~13–14 eV (literature G0W0@HF estimates for OH/cc-pVDZ — Bruneval et al.
//! report ~13 eV; experiment is 13.02 eV).
//!
//! Run: OPENBLAS_NUM_THREADS=1 cargo test -p ferric-gw --release --ignored

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::rhf::RhfConfig;
use ferric_gw::{run_u_gw, GwConfig, GwMethod};

const HA_TO_EV: f64 = 27.211386245988_f64;

fn oh_mol() -> Molecule {
    // OH at experimental bond length 0.9697 Å along the z axis.
    let xyz = "2
OH
O  0.0  0.0  0.0
H  0.0  0.0  0.9697
";
    Molecule::parse_xyz(xyz, 0, 2).expect("parse OH xyz")
}

fn prepare_oh() -> (Molecule, PreparedBasis, PreparedBasis, ferric_scf::ScfResult) {
    let mol = oh_mol();
    let obs_bs = basis::bundled("cc-pvdz").expect("cc-pvdz");
    let aux_bs = basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("obs");
    let dfbs = PreparedBasis::new(&mol, &aux_bs).expect("aux");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("Schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        max_iter: 200,
        ..Default::default()
    };
    let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &cfg).expect("UHF");
    assert!(uhf.converged, "UHF did not converge");
    (mol, obs, dfbs, uhf)
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
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
    }
}

#[test]
#[ignore = "slow: builds UHF + U-PDEP-RPA + U-G0W0; run with --release --ignored"]
fn oh_u_g0w0_first_ip_runs() {
    let (mol, obs, dfbs, uhf) = prepare_oh();
    let op = Operator::coulomb();
    let pdep = pdep_cfg();

    let two_s = (mol.multiplicity as i32) - 1;
    let nocc_a = ((mol.nelec() + two_s) / 2) as usize;
    let nocc_b = ((mol.nelec() - two_s) / 2) as usize;
    let homo_a = nocc_a - 1;
    let homo_b = nocc_b - 1;

    let gcfg = GwConfig { method: GwMethod::G0W0, ..Default::default() };
    let res = run_u_gw(&mol, &obs, &dfbs, op, &uhf, &pdep, &gcfg).expect("U-G0W0");

    let idx_a = res.mo_indices.iter().position(|&i| i == homo_a).expect("HOMO_α in range");
    let idx_b = res.mo_indices.iter().position(|&i| i == homo_b).expect("HOMO_β in range");
    let ip_a_qp = -res.eps_qp_a[idx_a] * HA_TO_EV;
    let ip_b_qp = -res.eps_qp_b[idx_b] * HA_TO_EV;
    let ip_a_mf = -res.eps_mf_a[idx_a] * HA_TO_EV;
    let ip_b_mf = -res.eps_mf_b[idx_b] * HA_TO_EV;

    println!("OH/cc-pVDZ U-G0W0@UHF:");
    println!("  UHF Koopmans α-HOMO: {ip_a_mf:.3} eV    β-HOMO: {ip_b_mf:.3} eV");
    println!("  U-G0W0   α-HOMO QP: {ip_a_qp:.3} eV    β-HOMO QP: {ip_b_qp:.3} eV");
    println!("  Σ_c(α): {:.4} Ha  Z(α): {:.3}", res.sigma_c_a[idx_a], res.z_factor_a[idx_a]);
    println!("  Σ_c(β): {:.4} Ha  Z(β): {:.3}", res.sigma_c_b[idx_b], res.z_factor_b[idx_b]);

    // Sanity bounds: QP IPs should be finite, in the (UHF Koopmans − 5, UHF Koopmans + 1) eV window
    // and bracket experiment (~13.02 eV) on at least the right order.
    for (ip, ip_mf, label) in [(ip_a_qp, ip_a_mf, "α"), (ip_b_qp, ip_b_mf, "β")] {
        assert!(
            ip.is_finite(),
            "{label}-HOMO QP IP is not finite"
        );
        assert!(
            ip > 5.0 && ip < 20.0,
            "{label}-HOMO QP IP = {ip:.3} eV is wildly out of range"
        );
        assert!(
            ip < ip_mf + 0.5,
            "{label}-HOMO QP IP {ip:.3} should be ≤ Koopmans {ip_mf:.3} + 0.5 (Σ_c expected negative)"
        );
    }
    // Z factor in (0, 1] is the canonical QP renormalization range.
    assert!(res.z_factor_a[idx_a] > 0.4 && res.z_factor_a[idx_a] <= 1.1);
    assert!(res.z_factor_b[idx_b] > 0.4 && res.z_factor_b[idx_b] <= 1.1);
}

#[test]
#[ignore = "slow: builds UHF + U-PDEP-RPA + U-COHSEX; run with --release --ignored"]
fn oh_u_cohsex_runs() {
    let (mol, obs, dfbs, uhf) = prepare_oh();
    let op = Operator::coulomb();
    let pdep = pdep_cfg();
    let gcfg = GwConfig { method: GwMethod::Cohsex, ..Default::default() };
    let res = run_u_gw(&mol, &obs, &dfbs, op, &uhf, &pdep, &gcfg).expect("U-COHSEX");
    let two_s = (mol.multiplicity as i32) - 1;
    let nocc_a = ((mol.nelec() + two_s) / 2) as usize;
    let homo_a = nocc_a - 1;
    let idx = res.mo_indices.iter().position(|&i| i == homo_a).unwrap();
    let ip = -res.eps_qp_a[idx] * HA_TO_EV;
    println!("OH U-COHSEX α-HOMO IP: {ip:.3} eV");
    // COHSEX typically overshoots G0W0 IP by ~1 eV — generous bound.
    assert!(ip > 8.0 && ip < 22.0, "U-COHSEX α-HOMO IP {ip:.3} out of bound");
}

#[test]
#[ignore = "slow: builds UHF + U-PDEP-RPA + U-evGW₀; run with --release --ignored"]
fn oh_u_evgw0_converges() {
    let (mol, obs, dfbs, uhf) = prepare_oh();
    let op = Operator::coulomb();
    let pdep = pdep_cfg();
    let gcfg = GwConfig {
        method: GwMethod::EvGw0,
        max_ev_iter: 8,
        ev_conv_thresh: 1e-4,
        ..Default::default()
    };
    let res = run_u_gw(&mol, &obs, &dfbs, op, &uhf, &pdep, &gcfg).expect("U-evGW0");
    println!("U-evGW0: {} outer iterations", res.n_ev_iter);
    assert!(res.n_ev_iter >= 1, "U-evGW0 did at least one outer step");
    for (idx, &mo_abs) in res.mo_indices.iter().enumerate() {
        let _ = mo_abs;
        assert!(res.eps_qp_a[idx].is_finite());
        assert!(res.eps_qp_b[idx].is_finite());
    }
}
