//! Integration tests for the block-Lanczos PDEP eigensolver.
//!
//! Validates that the Lanczos backend produces RPA correlation energies
//! within 1e-9 Ha of the Davidson default on small systems, and within
//! 1e-7 Ha on a larger system (H₂O/aug-cc-pVTZ — naux=198).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Eigensolver, QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::ScfResult;
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

fn setup(
    xyz: &str,
    obs_name: &str,
    dfbs_name: &str,
) -> (Molecule, PreparedBasis, PreparedBasis, Operator, ScfResult) {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(xyz).unwrap();
    let obs_bs = basis::bundled(obs_name).unwrap();
    let dfbs_bs = basis::bundled(dfbs_name).unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    (mol, obs, dfbs, op, rhf)
}

fn pyscf_compat_config(n_quad: usize) -> PdepRpaConfig {
    let mut cfg = PdepRpaConfig::default();
    cfg.quadrature = QuadratureConfig {
        scheme: QuadratureScheme::GaussLegendre,
        n_points: n_quad,
        u0: 0.5,
    };
    cfg.frozen_core = 0;
    cfg.trunc_thresh = 0.0;
    cfg.davidson_conv_thresh = 1e-10;
    cfg
}

#[test]
fn h2o_cc_pvdz_lanczos_matches_davidson() {
    let (mol, obs, dfbs, op, rhf) =
        setup("../../testdata/molecules/water.xyz", "cc-pvdz", "cc-pvdz-ri");

    // Davidson reference
    let mut cfg_dav = pyscf_compat_config(40);
    cfg_dav.eigensolver = Eigensolver::Davidson;
    let t0 = Instant::now();
    let r_dav = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_dav).unwrap();
    let dt_dav = t0.elapsed();

    // Lanczos
    let mut cfg_lz = pyscf_compat_config(40);
    cfg_lz.eigensolver = Eigensolver::Lanczos;
    let t0 = Instant::now();
    let r_lz = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_lz).unwrap();
    let dt_lz = t0.elapsed();

    let diff = (r_dav.e_rpa - r_lz.e_rpa).abs();
    println!(
        "H2O/cc-pVDZ Davidson E_c={:.12} ({:?}, M={}); Lanczos E_c={:.12} ({:?}, M={}); |Δ| = {:.2e}",
        r_dav.e_rpa,
        dt_dav,
        r_dav.n_eigenpotentials,
        r_lz.e_rpa,
        dt_lz,
        r_lz.n_eigenpotentials,
        diff,
    );
    assert!(
        diff < 1e-9,
        "Lanczos vs Davidson energy disagreement on H2O/cc-pVDZ: |Δ| = {:.2e} > 1e-9",
        diff
    );
}

#[test]
fn h2o_aug_cc_pvtz_lanczos_matches_davidson() {
    // Larger system: aug-cc-pVTZ on water (naux=198). Looser threshold (1e-7 Ha)
    // since this is a stress test for the larger Krylov subspace.
    let (mol, obs, dfbs, op, rhf) = setup(
        "../../testdata/molecules/water.xyz",
        "aug-cc-pvtz",
        "aug-cc-pvtz-rifit",
    );

    let mut cfg_dav = pyscf_compat_config(40);
    cfg_dav.eigensolver = Eigensolver::Davidson;
    let t0 = Instant::now();
    let r_dav = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_dav).unwrap();
    let dt_dav = t0.elapsed();

    let mut cfg_lz = pyscf_compat_config(40);
    cfg_lz.eigensolver = Eigensolver::Lanczos;
    let t0 = Instant::now();
    let r_lz = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg_lz).unwrap();
    let dt_lz = t0.elapsed();

    let diff = (r_dav.e_rpa - r_lz.e_rpa).abs();
    println!(
        "H2O/aug-cc-pVTZ Davidson E_c={:.12} ({:?}, M={}); Lanczos E_c={:.12} ({:?}, M={}); |Δ| = {:.2e}",
        r_dav.e_rpa,
        dt_dav,
        r_dav.n_eigenpotentials,
        r_lz.e_rpa,
        dt_lz,
        r_lz.n_eigenpotentials,
        diff,
    );
    assert!(
        diff < 1e-7,
        "Lanczos vs Davidson energy disagreement on H2O/aug-cc-pVTZ: |Δ| = {:.2e} > 1e-7",
        diff
    );
}
