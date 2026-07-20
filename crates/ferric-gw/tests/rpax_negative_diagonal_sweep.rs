//! Investigation (task #68): sweep for a negative-diagonal alpha tensor from
//! `run_rpax_static_polarizability` (the CLI/Python-wired RPAx@KS static
//! polarizability path). See docs/rpax-negative-diagonal-investigation.md for
//! the write-up. This test file is diagnostic/exploratory, not a pinned
//! regression gate (grades are printed, not all asserted).
//!
//! Sweeps a handful of small molecules/geometries (closed-shell only, since
//! `run_rpax_static_polarizability` hard-errors on non-Restricted references)
//! at STO-3G and cc-pVDZ, looking for ANY negative diagonal element of the
//! resulting 3x3 static alpha tensor. For each system also reports the min
//! eigenvalue of (A-B) and (A+B) directly (rebuilt inline, mirroring bse.rs's
//! private fill_row closures) to distinguish "kernel is genuinely
//! non-positive-definite" from "kernel is fine but the alpha contraction has
//! a sign/indexing bug".
//!
//! Run: cargo test -p ferric-gw --release --test rpax_negative_diagonal_sweep -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::bse::run_rpax_static_polarizability;
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
        need_inv_dielectric_freq: false,
        verbose: false,
    }
}

/// Rebuild (A+B)/(A-B) exactly as `run_rpax_static_polarizability` does
/// internally (duplicated here since bse.rs doesn't expose them), then report
/// (min_eig_apb, min_eig_amb, homo_lumo_gap_ev).
fn ab_min_eigs(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    ks: &ferric_scf::ScfResult,
    pdep_cfg: &PdepRpaConfig,
    scissor: f64,
) -> (f64, f64, f64) {
    let nmo = ks.eps_r().len();
    let nocc_total = (mol.nelec() as usize) / 2;
    let mut eps = ks.eps_r().to_vec();
    for p in nocc_total..nmo {
        eps[p] += scissor;
    }
    let gap_ev = (eps[nocc_total] - eps[nocc_total - 1]) * HA_TO_EV;

    let pdep = ferric_rpa::run_pdep_rpa(mol, obs, dfbs, op, ks, pdep_cfg).unwrap();
    let nocc = nocc_total;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;

    let mob = mo_b::build_full_b(mol, obs, dfbs, op, ks, 0).unwrap();
    let (v_dressed, _dev) = w_pdep::redress_with_check(&mob.v_inv_sqrt, &pdep.eigenpotentials).unwrap();
    let m_proj = ferric_gw::cohsex::project_b_into_pdep(&mob, &v_dressed);
    let m_modes = m_proj.shape()[0];
    let w_red: Vec<f64> = pdep.eigenvalues_static.iter().map(|&l| 1.0 / l - 1.0).collect();
    let b = &mob.b_full;
    let naux = mob.naux;
    let bare = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = 0.0;
        for pp in 0..naux {
            acc += b[(pp, p, q)] * b[(pp, r, s)];
        }
        acc
    };
    let screened = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = bare(p, q, r, s);
        for alpha in 0..m_modes {
            acc += w_red[alpha] * m_proj[(alpha, p, q)] * m_proj[(alpha, r, s)];
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
                    let coul = bare(i, a_loc, j, bb + nocc);
                    let w_abij = screened(a_loc, b_loc, i, j);
                    let w_ibaj = screened(i, b_loc, a_loc, j);
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
    (min_apb, min_amb, gap_ev)
}

struct Case {
    name: &'static str,
    xyz: &'static str,
    charge: i32,
    mult: usize,
    basis: &'static str,
    xc: &'static str,
}

#[test]
#[ignore = "slow: sweeps ~10 RPAx@KS static-alpha systems; --release --ignored --nocapture"]
fn sweep_negative_diagonal() {
    let cases = vec![
        Case { name: "water/cc-pvdz/PBE", xyz: "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n", charge: 0, mult: 1, basis: "cc-pvdz", xc: "PBE" },
        Case { name: "water/sto-3g/PBE", xyz: "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n", charge: 0, mult: 1, basis: "sto-3g", xc: "PBE" },
        Case { name: "methane/cc-pvdz/PBE", xyz: "5\nCH4\nC 0.000000 0.000000 0.000000\nH 0.629118 0.629118 0.629118\nH -0.629118 -0.629118 0.629118\nH -0.629118 0.629118 -0.629118\nH 0.629118 -0.629118 -0.629118\n", charge: 0, mult: 1, basis: "cc-pvdz", xc: "PBE" },
        Case { name: "H2/cc-pvdz/PBE (eq)", xyz: "2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.7414\n", charge: 0, mult: 1, basis: "cc-pvdz", xc: "PBE" },
        Case { name: "H2/cc-pvdz/PBE (stretched 1.35x)", xyz: "2\nH2 stretched\nH 0.0 0.0 0.0\nH 0.0 0.0 1.0000\n", charge: 0, mult: 1, basis: "cc-pvdz", xc: "PBE" },
        Case { name: "H2/cc-pvdz/PBE (stretched 2x)", xyz: "2\nH2 stretched 2x\nH 0.0 0.0 0.0\nH 0.0 0.0 1.4828\n", charge: 0, mult: 1, basis: "cc-pvdz", xc: "PBE" },
        Case { name: "H2/cc-pvdz/PBE (stretched 3x, near-dissociation)", xyz: "2\nH2 stretched 3x\nH 0.0 0.0 0.0\nH 0.0 0.0 2.2242\n", charge: 0, mult: 1, basis: "cc-pvdz", xc: "PBE" },
        Case { name: "N2/cc-pvdz/PBE stretched 1.5x", xyz: "2\nN2 stretched\nN 0.0 0.0 0.0\nN 0.0 0.0 1.6900\n", charge: 0, mult: 1, basis: "cc-pvdz", xc: "PBE" },
        Case { name: "N2/cc-pvdz/PBE (eq)", xyz: "2\nN2\nN 0.0 0.0 0.0\nN 0.0 0.0 1.0977\n", charge: 0, mult: 1, basis: "cc-pvdz", xc: "PBE" },
        Case { name: "LiH/cc-pvdz/PBE", xyz: "2\nLiH\nLi 0.0 0.0 0.0\nH 0.0 0.0 1.5949\n", charge: 0, mult: 1, basis: "cc-pvdz", xc: "PBE" },
        Case { name: "water/cc-pvdz/HF-as-xc(plain HF)", xyz: "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n", charge: 0, mult: 1, basis: "cc-pvdz", xc: "HF" },
    ];

    let ctx = ParallelContext::default();
    let mut any_negative = false;

    for c in &cases {
        let mol = match Molecule::parse_xyz(c.xyz, c.charge, c.mult) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[{}] SKIP: mol parse failed: {e}", c.name);
                continue;
            }
        };
        let obs = match PreparedBasis::new(&mol, &basis::bundled(c.basis).unwrap()) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[{}] SKIP: basis failed: {e}", c.name);
                continue;
            }
        };
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();

        let scf_cfg = RhfConfig {
            xc: Some(c.xc.to_string()),
            ..Default::default()
        };
        let ks = match solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg) {
            Ok(r) if r.converged => r,
            Ok(r) => {
                eprintln!("[{}] SKIP: SCF did not converge (energy={:.6})", c.name, r.energy);
                continue;
            }
            Err(e) => {
                eprintln!("[{}] SKIP: SCF failed: {e}", c.name);
                continue;
            }
        };

        let res = run_rpax_static_polarizability(&mol, &obs, &dfbs, op, &ks, &pdep_cfg(), 0, 0.0);
        match res {
            Ok(r) => {
                let (min_apb, min_amb, gap_ev) =
                    ab_min_eigs(&mol, &obs, &dfbs, op, &ks, &pdep_cfg(), 0.0);
                let diag_neg = r.tensor[0][0] < 0.0 || r.tensor[1][1] < 0.0 || r.tensor[2][2] < 0.0;
                if diag_neg {
                    any_negative = true;
                }
                eprintln!(
                    "[{}] gap={:.3}eV  diag=({:+.4},{:+.4},{:+.4})  iso={:+.4}  min_eig(A+B)={:+.3e}  min_eig(A-B)={:+.3e}{}",
                    c.name,
                    gap_ev,
                    r.tensor[0][0],
                    r.tensor[1][1],
                    r.tensor[2][2],
                    r.iso,
                    min_apb,
                    min_amb,
                    if diag_neg { "  <<< NEGATIVE DIAGONAL" } else { "" }
                );
                if min_apb < 0.0 || min_amb < 0.0 {
                    eprintln!("    ^^^ (A+B) or (A-B) is NOT positive-definite for this system (genuine TDHF instability territory)");
                }
            }
            Err(e) => {
                eprintln!("[{}] run_rpax_static_polarizability ERRORED: {e}", c.name);
            }
        }
    }

    eprintln!("\n=== SWEEP SUMMARY: any_negative_diagonal = {any_negative} ===");
    // Deliberately not asserting here -- this test's job is to REPORT, see
    // docs/rpax-negative-diagonal-investigation.md for the verdict drawn from
    // the printed output.
}

#[test]
#[ignore = "slow: same sweep but with scissor=0.36 (typical GW-gap proxy); --release --ignored --nocapture"]
fn sweep_with_scissor_correction() {
    let cases = vec![
        Case { name: "water/cc-pvdz/PBE", xyz: "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n", charge: 0, mult: 1, basis: "cc-pvdz", xc: "PBE" },
        Case { name: "water/sto-3g/PBE", xyz: "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n", charge: 0, mult: 1, basis: "sto-3g", xc: "PBE" },
        Case { name: "N2/cc-pvdz/PBE (eq)", xyz: "2\nN2\nN 0.0 0.0 0.0\nN 0.0 0.0 1.0977\n", charge: 0, mult: 1, basis: "cc-pvdz", xc: "PBE" },
        Case { name: "LiH/cc-pvdz/PBE", xyz: "2\nLiH\nLi 0.0 0.0 0.0\nH 0.0 0.0 1.5949\n", charge: 0, mult: 1, basis: "cc-pvdz", xc: "PBE" },
    ];
    let ctx = ParallelContext::default();
    let mut any_negative = false;
    let scissor = 0.36_f64; // Ha, typical KS->GW gap-widening proxy already used elsewhere in bse.rs

    for c in &cases {
        let mol = Molecule::parse_xyz(c.xyz, c.charge, c.mult).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled(c.basis).unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let scf_cfg = RhfConfig { xc: Some(c.xc.to_string()), ..Default::default() };
        let ks = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();
        assert!(ks.converged);

        let res = run_rpax_static_polarizability(&mol, &obs, &dfbs, op, &ks, &pdep_cfg(), 0, scissor);
        match res {
            Ok(r) => {
                let diag_neg = r.tensor[0][0] < 0.0 || r.tensor[1][1] < 0.0 || r.tensor[2][2] < 0.0;
                if diag_neg { any_negative = true; }
                eprintln!(
                    "[{} @ scissor={scissor}] diag=({:+.4},{:+.4},{:+.4}) iso={:+.4}{}",
                    c.name, r.tensor[0][0], r.tensor[1][1], r.tensor[2][2], r.iso,
                    if diag_neg { "  <<< STILL NEGATIVE" } else { "  (fixed)" }
                );
            }
            Err(e) => eprintln!("[{}] ERRORED: {e}", c.name),
        }
    }
    eprintln!("\n=== SCISSOR-CORRECTED SWEEP: any_negative_diagonal = {any_negative} ===");
}
