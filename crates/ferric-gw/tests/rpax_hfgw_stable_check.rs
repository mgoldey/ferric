//! Investigation (task #68) cross-check: run_bse_c6 (HF-reference + REAL
//! G0W0@HF quasiparticle diagonal, gate-2/library-only path) should NOT show
//! the negative-(A-B) instability that run_bse_c6_ks/
//! run_rpax_static_polarizability (KS-reference, bare-KS diagonal, no GW
//! correction) show -- because its diagonal is already gap-corrected before
//! screening is mixed in. Confirms the root cause is "narrow uncorrected
//! diagonal + full static screening", not the screened-kernel formula
//! itself (which run_bse_tda's gate-1 validation, and rpax_bare_ab_check.rs's
//! bare-exchange match to PySCF, already showed is correct).
//!
//! Run: cargo test -p ferric-gw --release --test rpax_hfgw_stable_check -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::bse::run_bse_c6;
use ferric_gw::{mo_b, run_gw, w_pdep, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig { scheme: QuadratureScheme::GaussLegendre, n_points: 16, u0: 0.5 },
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
        need_inv_dielectric_freq: false,
        verbose: false,
    }
}

#[test]
#[ignore = "slow: G0W0@HF + PDEP + BSE C6 min-eig check on water; --release --ignored --nocapture"]
fn hf_gw_diagonal_is_stable_water() {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    // Just run run_bse_c6 at freq=0 only, to get alpha_static without the full CP grid.
    let (freqs, weights) = (vec![0.0_f64], vec![1.0_f64]);
    let res = run_bse_c6(&mol, &obs, &dfbs, op, &rhf, &pdep_cfg(), 0, &freqs, &weights).unwrap();
    eprintln!("run_bse_c6 (HF+G0W0@HF diagonal) water: alpha_static = {:.4} (must be positive, DOSD=9.64)", res.alpha_static);
    assert!(res.alpha_static > 0.0, "HF+GW-diagonal path must give POSITIVE static alpha (sanity re-confirmation)");

    // Now directly rebuild A and A-B with the REAL GW diagonal to check min eig sign.
    let nmo = rhf.eps_r().len();
    let nocc_total = (mol.nelec() as usize) / 2;
    let gw_cfg = GwConfig {
        method: GwMethod::G0W0,
        qp_mos: Some(0..nmo),
        max_ev_iter: 0,
        ev_conv_thresh: 1e-4,
        pade_npts: 0,
        qp_newton_damp: 1.0,
        frozen_core: 0,
        memory_budget_bytes: None,
        verbose: false,
    };
    let gw = run_gw(&mol, &obs, &dfbs, op, &rhf, &pdep_cfg(), &gw_cfg, None).unwrap();
    let mut eps_qp = rhf.eps_r().to_vec();
    for (k, &mo) in gw.mo_indices.iter().enumerate() { eps_qp[mo] = gw.eps_qp[k]; }
    let gap_ev = (eps_qp[nocc_total] - eps_qp[nocc_total-1]) * 27.211386245988;
    eprintln!("G0W0@HF gap = {gap_ev:.3} eV");

    let nocc = nocc_total;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;
    let (v_dressed, _dev) = w_pdep::redress_with_check(&mo_b::build_full_b(&mol, &obs, &dfbs, op, &rhf, 0).unwrap().v_inv_sqrt, &gw.pdep.eigenpotentials).unwrap();
    let mob = mo_b::build_full_b(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
    let m_proj = ferric_gw::cohsex::project_b_into_pdep(&mob, &v_dressed);
    let m_modes = m_proj.shape()[0];
    let w_red: Vec<f64> = gw.pdep.eigenvalues_static.iter().map(|&l| 1.0/l - 1.0).collect();
    let b = &mob.b_full;
    let naux = mob.naux;
    let bare = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = 0.0;
        for pp in 0..naux { acc += b[(pp,p,q)]*b[(pp,r,s)]; }
        acc
    };
    let screened = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = bare(p,q,r,s);
        for al in 0..m_modes { acc += w_red[al]*m_proj[(al,p,q)]*m_proj[(al,r,s)]; }
        acc
    };
    let mut amb = Array2::<f64>::zeros((n,n));
    for i in 0..nocc {
        let eps_i = eps_qp[i];
        for a in 0..nvir {
            let ia = i*nvir+a;
            let a_loc = nocc+a;
            let eps_a = eps_qp[nocc_total+a];
            for j in 0..nocc {
                for bb in 0..nvir {
                    let b_loc = nocc+bb;
                    let jb = j*nvir+bb;
                    let w_abij = screened(a_loc,b_loc,i,j);
                    let w_ibaj = screened(i,b_loc,a_loc,j);
                    amb[(ia,jb)] = w_ibaj - w_abij;
                }
            }
            amb[(ia,ia)] += eps_a - eps_i;
        }
    }
    let (evals,_) = amb.eigh(UPLO::Upper).unwrap();
    let min_amb = evals.iter().cloned().fold(f64::MAX, f64::min);
    eprintln!("min eig (A-B) with REAL G0W0@HF diagonal = {min_amb:+.6}  (vs KS-diagonal water case: -0.0804 at scissor=0)");
    assert!(min_amb > 0.0, "with the real GW-corrected diagonal, (A-B) should be positive-definite for water");
}
