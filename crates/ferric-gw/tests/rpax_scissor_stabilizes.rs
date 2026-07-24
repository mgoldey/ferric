//! Investigation (task #68): does widening the KS gap toward the GW gap (via
//! the existing `scissor` knob) restore positive-definiteness of A and
//! (A-B)? If yes, this confirms the negative-diagonal alpha is a genuine
//! TDA/BSE electronic instability driven by screening the exchange kernel at
//! the SAME too-narrow KS gap used for the diagonal (a known pathology in
//! the BSE/TDDFT literature: over-screened electron-hole attraction at a
//! DFT-gap-scale energy denominator), not a ferric assembly bug (already
//! ruled out separately: bare-exchange kernel is positive-definite and
//! matches PySCF to 3e-4, see rpax_bare_ab_check.rs).
//!
//! Run: cargo test -p ferric-gw --release --test rpax_scissor_stabilizes -- --ignored --nocapture

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

const HA_TO_EV: f64 = 27.211_386_245_988;

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
        need_eigenvalues_freq: true,
        verbose: false,
    }
}

fn min_eigs_at_scissor(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    ks: &ferric_scf::ScfResult,
    scissor: f64,
) -> (f64, f64, f64, f64) {
    let nmo = ks.eps_r().len();
    let nocc_total = (mol.nelec() as usize) / 2;
    let mut eps = ks.eps_r().to_vec();
    for p in nocc_total..nmo { eps[p] += scissor; }
    let gap_ev = (eps[nocc_total] - eps[nocc_total - 1]) * HA_TO_EV;
    let nocc = nocc_total;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;

    let pdep = ferric_rpa::run_pdep_rpa(mol, obs, dfbs, op, ks, &pdep_cfg()).unwrap();
    let mob = mo_b::build_full_b(mol, obs, dfbs, op, ks, 0).unwrap();
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

    let mut apb = Array2::<f64>::zeros((n, n));
    let mut amb = Array2::<f64>::zeros((n, n));
    let mut a_mat = Array2::<f64>::zeros((n, n));
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
                    apb[(ia, jb)] = 4.0 * coul - w_abij - w_ibaj;
                    amb[(ia, jb)] = w_ibaj - w_abij;
                    a_mat[(ia, jb)] = 2.0 * coul - w_abij;
                }
            }
            apb[(ia, ia)] += eps_a - eps_i;
            amb[(ia, ia)] += eps_a - eps_i;
            a_mat[(ia, ia)] += eps_a - eps_i;
        }
    }
    let a_sym = 0.5 * (&a_mat + &a_mat.t());
    let (ea, _) = a_sym.eigh(UPLO::Upper).unwrap();
    let (eapb, _) = apb.eigh(UPLO::Upper).unwrap();
    let (eamb, _) = amb.eigh(UPLO::Upper).unwrap();
    (
        gap_ev,
        ea.iter().cloned().fold(f64::MAX, f64::min),
        eapb.iter().cloned().fold(f64::MAX, f64::min),
        eamb.iter().cloned().fold(f64::MAX, f64::min),
    )
}

#[test]
#[ignore = "slow: scissor scan of A/(A+-B) min-eig on water; --release --ignored --nocapture"]
fn scissor_scan_min_eigs_water() {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let ks = solve_rhf(&ctx, &mol, &obs, op, &bounds,
        &RhfConfig { xc: Some("PBE".to_string()), ..Default::default() }).unwrap();

    eprintln!("water/cc-pVDZ RPAx@PBE, scissor scan (KS gap 7.05eV -> GW gap ~16.86eV at scissor~0.36 Ha):");
    eprintln!("{:>8}  {:>9}  {:>12}  {:>12}  {:>12}", "scissor", "gap(eV)", "min_eig(A)", "min_eig(A+B)", "min_eig(A-B)");
    for &sc in &[0.0, 0.05, 0.10, 0.20, 0.30, 0.36, 0.50, 0.80] {
        let (gap, ea, eapb, eamb) = min_eigs_at_scissor(&mol, &obs, &dfbs, op, &ks, sc);
        eprintln!("{sc:>8.2}  {gap:>9.3}  {ea:>+12.6}  {eapb:>+12.6}  {eamb:>+12.6}");
    }
}
