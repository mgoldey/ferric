//! Internal-consistency cross-check for U-COHSEX and U-evGW/U-evGW₀
//! (open-shell self-consistent GW), widening validation beyond OH-only
//! sanity bounds (`oh_u_g0w0.rs`).
//!
//! ## Why this check and not a PySCF cross-check
//!
//! U-G0W0 already has a genuine external reference: PySCF's `pyscf.gw.ugw_ac`
//! (spin-unrestricted analytic-continuation GW) — see `u_g0w0_radicals.rs`.
//! U-COHSEX and U-evGW/U-evGW₀ do NOT have that option: as of PySCF 2.13.0,
//! `pyscf.gw` ships exactly four kernels (`gw_ac`, `gw_cd`, `gw_exact`,
//! `ugw_ac`) and NONE of them implement a COHSEX self-energy or an
//! eigenvalue-self-consistent (evGW/evGW₀) outer loop, restricted OR
//! unrestricted. Verified by grep across the installed
//! `pyscf/gw/*.py` (zero hits for "cohsex"/"COHSEX"/"evgw"/"EVGW" anywhere
//! in the package) and by reading `GWAC`/`UGWAC` source: `UGWAC.kernel()`
//! calls straight into the module-level `kernel()` in `ugw_ac.py`, which
//! only ever does perturbative (or Z-linearized) one-shot G0W0 — there is no
//! outer eigenvalue loop and no static-screening (COHSEX) branch. This is a
//! package-wide gap, not an open-shell-specific one: ferric's own
//! CLOSED-shell COHSEX/evGW (`docs/VALIDATION.md`) is likewise validated
//! against experimental IPs on the GW100 subset, not PySCF, for the same
//! reason.
//!
//! ## What this check does instead
//!
//! ferric's closed-shell COHSEX/evGW₀/evGW are GW100-benchmarked against
//! experiment (69-molecule subset, `docs/gw100-whitepaper.md`) — a much
//! larger and independently-anchored validation than any single-molecule
//! spike. The open-shell (U-) variants share the same `mo_b`/`w_pdep`
//! machinery per spin channel (`u_cohsex.rs`, `u_sigma.rs::run_u_evgw0/
//! run_u_evgw`) but had no check tying them back to that already-validated
//! closed-shell code path.
//!
//! This test closes that gap the same way `u_pdep_rpa.rs` already validates
//! U-PDEP-RPA: run a closed-shell singlet (H₂O) through
//!   (a) the closed-shell path (`solve_rhf` → `run_gw`), and
//!   (b) the open-shell path with a SPIN-SYMMETRIC UHF reference
//!       (`solve_uhf` on the same singlet, where α and β orbitals coincide
//!       by symmetry) → `run_u_gw`,
//! for both `GwMethod::Cohsex` and `GwMethod::EvGw0`/`GwMethod::EvGw`, and
//! assert the α-channel (≡ β-channel) QP energies from (b) reproduce (a) to
//! tight tolerance. This is a genuine, independent-of-PySCF correctness
//! check on the open-shell self-energy assembly and (for evGW/evGW₀) the
//! open-shell outer self-consistency loop: any sign, factor-of-2, or
//! spin-summation bug in `u_cohsex.rs`/`u_sigma.rs`'s COHSEX or evGW
//! branches would break this limit even though it's invisible to the
//! OH/CH₃/NH₂ G0W0-only PySCF cross-check.
//!
//! Run: OPENBLAS_NUM_THREADS=1 cargo test -p ferric-gw --release --ignored \
//!      u_cohsex_evgw_closed_shell_limit

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_gw, run_u_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

const HA_TO_EV: f64 = 27.211_386_245_988_f64;
/// Tight: this is a closed-shell-via-UHF identity, not a cross-code check.
/// evGW/evGW₀ run independent Newton/outer loops per spin channel with their
/// own convergence thresholds, so allow a little more slack than COHSEX's
/// closed-form (no-iteration) exact match.
const TOL_HA_COHSEX: f64 = 1e-6;
const TOL_HA_EVGW: f64 = 5e-5;

fn water_mol() -> Molecule {
    let xyz = "3
H2O
O  0.0   0.0       0.117790
H  0.0   0.755453 -0.471161
H  0.0  -0.755453 -0.471161
";
    Molecule::parse_xyz(xyz, 0, 1).expect("parse H2O xyz")
}

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        need_eigenvalues_freq: true, // GW reads eigenvalues_freq
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 16,
            u0: 0.5,
        },
        eigensolver_conv_thresh: 1e-9,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0, // full rank: truncation would be a confound here
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: false, // run_gw/run_u_gw force this on
        verbose: false,
    }
}

/// Shared harness: run `method` both via the closed-shell RHF path and the
/// spin-symmetric-UHF open-shell path on H₂O/cc-pVDZ, and assert the U-GW
/// α/β channels each reproduce the closed-shell QP energies (mean-field,
/// Σx, Σc, QP) within `tol_ha`.
fn check_closed_shell_limit(method: GwMethod, tol_ha: f64, max_ev_iter: usize) {
    let ctx = ParallelContext::default();
    let mol = water_mol();
    let obs_bs = basis::bundled("cc-pvdz").expect("cc-pvdz");
    let aux_bs = basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("obs");
    let dfbs = PreparedBasis::new(&mol, &aux_bs).expect("aux");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("Schwarz");

    // (a) Closed-shell RHF reference.
    let rhf_cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &rhf_cfg).expect("RHF");
    assert!(rhf.converged, "RHF did not converge");

    // (b) Spin-symmetric UHF on the SAME closed-shell singlet: α and β
    // orbitals must coincide by symmetry, so this is the identical physical
    // reference fed through the open-shell code path.
    let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &rhf_cfg).expect("UHF");
    assert!(uhf.converged, "UHF did not converge");
    assert!(
        (uhf.energy - rhf.energy).abs() < 1e-8,
        "UHF/RHF energy disagreement on closed-shell H2O: uhf={:.10} rhf={:.10}",
        uhf.energy,
        rhf.energy
    );

    let pdep = pdep_cfg();
    let gcfg_closed = GwConfig {
        method,
        max_ev_iter,
        ev_conv_thresh: 1e-9,
        ..Default::default()
    };
    let gcfg_open = GwConfig {
        method,
        max_ev_iter,
        ev_conv_thresh: 1e-9,
        ..Default::default()
    };

    let res_closed = run_gw(&mol, &obs, &dfbs, op, &rhf, &pdep, &gcfg_closed, None)
        .unwrap_or_else(|e| panic!("closed-shell run_gw({method:?}) failed: {e}"));
    let res_open = run_u_gw(&mol, &obs, &dfbs, op, &uhf, &pdep, &gcfg_open)
        .unwrap_or_else(|e| panic!("open-shell run_u_gw({method:?}) failed: {e}"));

    assert_eq!(res_closed.mo_indices, res_open.mo_indices, "{method:?}: QP MO ranges differ");

    let mut max_dev_qp = 0.0_f64;
    for (idx, &mo_abs) in res_closed.mo_indices.iter().enumerate() {
        for (spin, eps_qp_spin, sigma_c_spin, sigma_x_spin) in [
            ("alpha", &res_open.eps_qp_a, &res_open.sigma_c_a, &res_open.sigma_x_a),
            ("beta", &res_open.eps_qp_b, &res_open.sigma_c_b, &res_open.sigma_x_b),
        ] {
            let dev_qp = (res_closed.eps_qp[idx] - eps_qp_spin[idx]).abs();
            let dev_sx = (res_closed.sigma_x[idx] - sigma_x_spin[idx]).abs();
            let dev_sc = (res_closed.sigma_c[idx] - sigma_c_spin[idx]).abs();
            max_dev_qp = max_dev_qp.max(dev_qp);
            assert!(
                dev_qp < tol_ha,
                "{method:?}: MO {mo_abs} {spin}-channel QP energy diverges from closed-shell: \
                 closed={:.8} Ha, open={:.8} Ha, dev={dev_qp:.2e} Ha (tol {tol_ha:.1e})",
                res_closed.eps_qp[idx],
                eps_qp_spin[idx]
            );
            assert!(
                dev_sx < tol_ha,
                "{method:?}: MO {mo_abs} {spin}-channel Sigma_x diverges: dev={dev_sx:.2e} Ha"
            );
            assert!(
                dev_sc < tol_ha,
                "{method:?}: MO {mo_abs} {spin}-channel Sigma_c diverges: dev={dev_sc:.2e} Ha"
            );
        }
    }
    let homo = (mol.nelec() as usize) / 2 - 1;
    let idx_homo = res_closed
        .mo_indices
        .iter()
        .position(|&i| i == homo)
        .expect("HOMO in qp range");
    println!(
        "{method:?} closed-shell-via-UHF limit on H2O/cc-pVDZ: max|dev QP| = {max_dev_qp:.2e} Ha \
         (tol {tol_ha:.1e}); HOMO IP closed={:.4} eV, open(alpha)={:.4} eV, open(beta)={:.4} eV",
        -res_closed.eps_qp[idx_homo] * HA_TO_EV,
        -res_open.eps_qp_a[idx_homo] * HA_TO_EV,
        -res_open.eps_qp_b[idx_homo] * HA_TO_EV,
    );
}

#[test]
#[ignore = "slow: builds RHF+UHF + PDEP-RPA + COHSEX/U-COHSEX twice; run with --release --ignored"]
fn u_cohsex_matches_closed_shell_limit() {
    check_closed_shell_limit(GwMethod::Cohsex, TOL_HA_COHSEX, 1);
}

#[test]
#[ignore = "slow: builds RHF+UHF + PDEP-RPA + evGW0/U-evGW0 twice; run with --release --ignored"]
fn u_evgw0_matches_closed_shell_limit() {
    check_closed_shell_limit(GwMethod::EvGw0, TOL_HA_EVGW, 10);
}

#[test]
#[ignore = "slow: builds RHF+UHF + PDEP-RPA + evGW/U-evGW twice (W re-solved each outer \
            iteration); run with --release --ignored"]
fn u_evgw_matches_closed_shell_limit() {
    check_closed_shell_limit(GwMethod::EvGw, TOL_HA_EVGW, 8);
}
