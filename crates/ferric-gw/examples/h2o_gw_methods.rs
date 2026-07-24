//! Compare G0W0, COHSEX, evGW₀, evGW first-IP on H₂O/cc-pVDZ.
//!
//! Run:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release --example h2o_gw_methods -p ferric-gw

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const HA_TO_EV: f64 = 27.211386245988_f64;

fn main() {
    let xyz = "3
H2O
O  0.0   0.0       0.117790
H  0.0   0.755453 -0.471161
H  0.0  -0.755453 -0.471161
";
    let mol = Molecule::parse_xyz(xyz, 0, 1).expect("xyz");
    let obs_bs = basis::bundled("cc-pvdz").expect("cc-pvdz");
    let aux_bs = basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("obs");
    let dfbs = PreparedBasis::new(&mol, &aux_bs).expect("aux");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("Schwarz");
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("RHF");
    let nocc = (mol.nelec() as usize) / 2;
    let homo_abs = nocc - 1;
    println!("Water/cc-pVDZ — first IP (eV) by method:");
    println!("  RHF Koopmans (−ε_HOMO): {:.3}", -rhf.eps_r()[homo_abs] * HA_TO_EV);

    let pdep_cfg = PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 16,
            u0: 0.5,
        },
        eigensolver_conv_thresh: 1e-7,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        // run_gw forces this on internally; false is fine here (M9 gate).
        need_inv_dielectric_freq: false,
        need_eigenvalues_freq: true,
        verbose: false,
    };

    for method in [
        GwMethod::Cohsex,
        GwMethod::G0W0,
        GwMethod::EvGw0,
        GwMethod::EvGw,
    ] {
        let gcfg = GwConfig {
            method,
            max_ev_iter: 8,
            ev_conv_thresh: 1e-4,
            ..Default::default()
        };
        let res = run_gw(&mol, &obs, &dfbs, op, &rhf, &pdep_cfg, &gcfg, None).expect("gw run");
        let homo_local = res
            .mo_indices
            .iter()
            .position(|&i| i == homo_abs)
            .expect("HOMO in qp range");
        let ip = -res.eps_qp[homo_local] * HA_TO_EV;
        let tag = match method {
            GwMethod::G0W0 => "G0W0   ",
            GwMethod::Cohsex => "COHSEX ",
            GwMethod::EvGw0 => "evGW0  ",
            GwMethod::EvGw => "evGW   ",
            GwMethod::ScCohsex => "scCOHSEX",
        };
        println!(
            "  {tag} = {ip:.3}   (Σ_c = {:.4} Ha, Z = {:.3}, n_iter = {})",
            res.sigma_c[homo_local], res.z_factor[homo_local], res.n_ev_iter
        );
    }
    println!("  MOLGW G0W0@HF reference: 11.97  (van Setten 2015)");
    println!("  Experiment:              12.62  (NIST)");
}
