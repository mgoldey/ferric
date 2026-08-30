//! Isolates whether the negative-(A-B) bug (task #68 investigation) lives in
//! the BARE-exchange TDHF assembly itself (shared with the already-validated
//! `run_cis_tda`) or only appears once PDEP screening (`w_red`/`m_proj`) is
//! mixed in. Builds A+B/A-B using ONLY the bare Coulomb integral (mirrors
//! run_cis_tda's `bare()` closure exactly, W -> v), independent of any PDEP
//! machinery, then reports min eigenvalues.
//!
//! Run: cargo test -p ferric-gw --release --test rpax_bare_ab_check -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::mo_b;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};

#[test]
#[ignore = "fast: bare-exchange-only A+-B min-eig check; --release --ignored --nocapture"]
fn bare_ab_min_eig_water() {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let nmo = rhf.eps_r().len();
    let nocc_total = (mol.nelec() as usize) / 2;
    let eps = rhf.eps_r().to_vec();
    let nocc = nocc_total;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;

    let mob = mo_b::build_full_b(&mol, &obs, &dfbs, op, &rhf, 0, None).unwrap();
    let b = &mob.b_full;
    let naux = mob.naux;
    let bare = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = 0.0;
        for pp in 0..naux {
            acc += b[(pp, p, q)] * b[(pp, r, s)];
        }
        acc
    };

    let mut apb = Array2::<f64>::zeros((n, n));
    let mut amb = Array2::<f64>::zeros((n, n));
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
                    let w_abij = bare(a_loc, b_loc, i, j); // (ab|ij)  bare, no screening
                    let w_ibaj = bare(i, b_loc, a_loc, j); // (ib|aj) bare
                    apb[(ia, jb)] = 4.0 * coul - w_abij - w_ibaj;
                    amb[(ia, jb)] = w_ibaj - w_abij;
                }
            }
            apb[(ia, ia)] += eps_a - eps_i;
            amb[(ia, ia)] += eps_a - eps_i;
        }
    }
    let (evals_apb, _) = apb.eigh(UPLO::Upper).unwrap();
    let (evals_amb, _) = amb.eigh(UPLO::Upper).unwrap();
    let min_apb = evals_apb.iter().cloned().fold(f64::MAX, f64::min);
    let min_amb = evals_amb.iter().cloned().fold(f64::MAX, f64::min);
    eprintln!("BARE-exchange-only (no PDEP/W at all), water/cc-pVDZ RHF:");
    eprintln!("  min eig (A+B) = {min_apb:+.6}");
    eprintln!("  min eig (A-B) = {min_amb:+.6}");
    eprintln!("  (PySCF TDHF reference: min eig A-B = +0.321393)");
}
