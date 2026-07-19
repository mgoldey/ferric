//! Investigation (task #68): isolate whether the negative-(A-B) result is a
//! genuine consequence of applying STATIC W to the exchange kernel (which is
//! a well-known BSE/TDA subtlety -- see doc below) or an assembly bug.
//! Builds A and B SEPARATELY with the PDEP-screened kernel (same formula
//! run_bse_tda validates for A alone), forms A-B and A+B externally, and
//! compares against bse.rs's in-place apb/amb.
//!
//! Run: cargo test -p ferric-gw --release --test rpax_screened_ab_isolate -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::mo_b;
use ferric_gw::w_pdep;
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
    }
}

#[test]
#[ignore = "fast: screened-W A/B cross-check; --release --ignored --nocapture"]
fn screened_a_minus_b_vs_direct_amb() {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let ks = solve_rhf(&ctx, &mol, &obs, op, &bounds,
        &RhfConfig { xc: Some("PBE".to_string()), ..Default::default() }).unwrap();

    let nmo = ks.eps_r().len();
    let nocc_total = (mol.nelec() as usize) / 2;
    let eps = ks.eps_r().to_vec();
    let nocc = nocc_total;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;

    let pdep = ferric_rpa::run_pdep_rpa(&mol, &obs, &dfbs, op, &ks, &pdep_cfg()).unwrap();
    let mob = mo_b::build_full_b(&mol, &obs, &dfbs, op, &ks, 0).unwrap();
    let (v_dressed, _dev) = w_pdep::redress_with_check(&mob.v_inv_sqrt, &pdep.eigenpotentials).unwrap();
    let m_proj = ferric_gw::cohsex::project_b_into_pdep(&mob, &v_dressed);
    let m_modes = m_proj.shape()[0];
    let w_red: Vec<f64> = pdep.eigenvalues_static.iter().map(|&l| 1.0 / l - 1.0).collect();
    let b = &mob.b_full;
    let naux = mob.naux;
    let bare = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = 0.0;
        for pp in 0..naux { acc += b[(pp, p, q)] * b[(pp, r, s)]; }
        acc
    };
    let screened = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = bare(p, q, r, s);
        for alpha in 0..m_modes { acc += w_red[alpha] * m_proj[(alpha, p, q)] * m_proj[(alpha, r, s)]; }
        acc
    };

    // Build A and B SEPARATELY (A exactly like run_bse_tda's validated formula;
    // B via the textbook-derived (ib|aj) screened exchange), each as a full
    // n x n matrix, THEN form A-B / A+B externally.
    let mut a_mat = Array2::<f64>::zeros((n, n));
    let mut b_mat = Array2::<f64>::zeros((n, n));
    for i in 0..nocc {
        let eps_i = eps[i];
        for a in 0..nvir {
            let ia = i * nvir + a;
            let a_loc = nocc + a;
            let eps_a = eps[nocc_total + a];
            for j in 0..nocc {
                for bb in 0..nvir {
                    let b_loc = nocc + bb;
                    let jb = j * nvir + bb;
                    let coul = bare(i, a_loc, j, bb + nocc); // (ia|jb)
                    let a_exch = screened(a_loc, b_loc, i, j); // (ab|W|ij)
                    let b_exch = screened(i, b_loc, a_loc, j); // (ib|W|aj)
                    a_mat[(ia, jb)] = 2.0 * coul - a_exch;
                    b_mat[(ia, jb)] = 2.0 * coul - b_exch;
                }
            }
            a_mat[(ia, ia)] += eps_a - eps_i;
        }
    }
    // symmetrize (small numerical asymmetry from float ops)
    let a_sym = 0.5 * (&a_mat + &a_mat.t());
    let b_sym = 0.5 * (&b_mat + &b_mat.t());
    let amb_external = &a_sym - &b_sym;
    let apb_external = &a_sym + &b_sym;

    let (ea, _) = a_sym.eigh(UPLO::Upper).unwrap();
    let (eb, _) = b_sym.eigh(UPLO::Upper).unwrap();
    let (eamb, _) = amb_external.eigh(UPLO::Upper).unwrap();
    let (eapb, _) = apb_external.eigh(UPLO::Upper).unwrap();
    eprintln!("Separately-built A: min eig = {:+.6}", ea.iter().cloned().fold(f64::MAX, f64::min));
    eprintln!("Separately-built B: min eig = {:+.6}  max eig = {:+.6}", eb.iter().cloned().fold(f64::MAX, f64::min), eb.iter().cloned().fold(f64::MIN, f64::max));
    eprintln!("External (A-B): min eig = {:+.6}", eamb.iter().cloned().fold(f64::MAX, f64::min));
    eprintln!("External (A+B): min eig = {:+.6}", eapb.iter().cloned().fold(f64::MAX, f64::min));

    // Now build amb/apb the SAME way bse.rs does in-place (fused loop),
    // to check whether that path agrees with the externally-formed A-B.
    let mut apb2 = Array2::<f64>::zeros((n, n));
    let mut amb2 = Array2::<f64>::zeros((n, n));
    for i in 0..nocc {
        let eps_i = eps[i];
        for a in 0..nvir {
            let ia = i * nvir + a;
            let a_loc = nocc + a;
            let eps_a = eps[nocc_total + a];
            for j in 0..nocc {
                for bb in 0..nvir {
                    let b_loc = nocc + bb;
                    let jb = j * nvir + bb;
                    let coul = bare(i, a_loc, j, bb + nocc);
                    let w_abij = screened(a_loc, b_loc, i, j);
                    let w_ibaj = screened(i, b_loc, a_loc, j);
                    apb2[(ia, jb)] = 4.0 * coul - w_abij - w_ibaj;
                    amb2[(ia, jb)] = w_ibaj - w_abij;
                }
            }
            apb2[(ia, ia)] += eps_a - eps_i;
            amb2[(ia, ia)] += eps_a - eps_i;
        }
    }
    let diff_amb = (&amb2 - &amb_external).mapv(f64::abs).into_iter().fold(0.0, f64::max);
    let diff_apb = (&apb2 - &apb_external).mapv(f64::abs).into_iter().fold(0.0, f64::max);
    eprintln!("max|amb2 - amb_external| = {diff_amb:e}");
    eprintln!("max|apb2 - apb_external| = {diff_apb:e}");
}
