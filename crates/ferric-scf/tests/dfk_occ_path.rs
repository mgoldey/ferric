//! End-to-end correctness gate for the DF-K C_occ half-transform optimization.
//!
//! `DfK::build_from_occ` contracts the fitted 3-index tensor against the
//! occupied MO coefficients (O(naux·n²·nocc)) instead of the assembled density
//! (O(naux·n³)). The K-matrix-level equivalence is proven in df_k.rs unit tests
//! (`df_k_build_from_occ_matches_density_build`, <1e-10). These tests close the
//! loop at the SCF-energy level for BOTH the closed-shell (RHF, √2·C_occ
//! convention) and open-shell (UHF, two independent spin channels) drivers:
//!
//!   1. Run the identical SCF twice — once via the default C_occ fast path, once
//!      forcing the density path with FERRIC_DFK_FORCE_DENSITY — and assert the
//!      converged energies agree to ≤1e-8 Ha. The two paths contract the SAME
//!      fitted B tensor, so per-iteration K agrees to the DF-K reassociation
//!      floor (~1e-13, proven bit-level in df_k.rs); the residual ≤1e-8 gap is
//!      pure SCF-trajectory drift bounded by the density-convergence threshold,
//!      NOT an algebra difference. A wrong scaling factor, a wrong MO-block
//!      slice, or a swapped α/β channel would shift the energy by mHa — five
//!      orders of magnitude above this bar. This is the single most important
//!      gate: DF-JK SCF is one of the most heavily used paths in the engine.
//!   2. Anchor the absolute energy against the PySCF reference so the test also
//!      catches a bug common to BOTH paths (not just their difference).
//!
//! The env var is process-global, so the two runs are serialized under a mutex
//! and the var is always removed before the lock is released.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_core::FerricError;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, UhfConfig};
use std::sync::Mutex;

/// Extract the (best-effort) total energy whether the solver converged or
/// plateaued on the DF-JK noise floor. DF-JK SCF routinely parks just above a
/// strict density threshold while the energy is already essentially final (the
/// documented DF-JK noise floor), so — like the sibling dft_b3lyp / mpi_dfjk
/// tests — we compare the energy, not the converged flag.
fn energy_or_plateau(r: Result<ferric_scf::ScfResult, FerricError>) -> f64 {
    match r {
        Ok(res) => res.energy,
        Err(FerricError::ScfConvergence { last_energy, .. }) => last_energy,
        Err(e) => panic!("unexpected SCF error: {e:?}"),
    }
}

/// Serializes the two runs that toggle FERRIC_DFK_FORCE_DENSITY so a parallel
/// test (or the two calls here) never observe each other's env state.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn rhf_dfjk_energy(mol: &Molecule, prep: &PreparedBasis, force_density: bool) -> f64 {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, prep).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-8,
        max_iter: 200,
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if force_density {
        std::env::set_var("FERRIC_DFK_FORCE_DENSITY", "1");
    } else {
        std::env::remove_var("FERRIC_DFK_FORCE_DENSITY");
    }
    let e = energy_or_plateau(solve_rhf(&ctx, mol, prep, op, &bounds, &cfg));
    std::env::remove_var("FERRIC_DFK_FORCE_DENSITY");
    e
}

fn uhf_dfjk_energy(mol: &Molecule, prep: &PreparedBasis, force_density: bool) -> f64 {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, prep).unwrap();
    let cfg = UhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-8,
        max_iter: 300,
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if force_density {
        std::env::set_var("FERRIC_DFK_FORCE_DENSITY", "1");
    } else {
        std::env::remove_var("FERRIC_DFK_FORCE_DENSITY");
    }
    let e = energy_or_plateau(solve_uhf(&ctx, mol, prep, &bounds, &cfg));
    std::env::remove_var("FERRIC_DFK_FORCE_DENSITY");
    e
}

#[test]
fn rhf_dfk_occ_path_matches_density_path() {
    // Water / cc-pVDZ, DF-JK. Closed shell: exercises the √2·C_occ convention.
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();

    let e_occ = rhf_dfjk_energy(&mol, &prep, false);
    let e_den = rhf_dfjk_energy(&mol, &prep, true);
    eprintln!("RHF DF-JK water/cc-pVDZ: occ={e_occ:.12}  density={e_den:.12}  diff={:.2e}", (e_occ - e_den).abs());
    assert!(
        (e_occ - e_den).abs() < 1e-8,
        "DF-K C_occ path vs density path SCF energy diff = {:.3e} (occ={e_occ}, density={e_den})",
        (e_occ - e_den).abs()
    );

    // Absolute anchor: DF-K carries RI fitting error vs the exact PySCF number,
    // so allow a few mHa here — this only guards against a gross bug shared by
    // BOTH paths (the occ/density agreement above is the tight gate).
    let pyscf = -76.0267679973766_f64;
    assert!(
        (e_occ - pyscf).abs() < 3e-3,
        "DF-JK RHF water energy {e_occ} too far from PySCF {pyscf} (RI floor is ~1e-3)"
    );
}

#[test]
fn uhf_dfk_occ_path_matches_density_path() {
    // OH radical / cc-pVDZ, DF-JK. Open shell: the trickier case — build_from_occ
    // is called independently for the α (5 occ) and β (4 occ) channels, so a
    // swapped or mis-sliced channel would surface here as an occ-vs-density gap.
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();

    let e_occ = uhf_dfjk_energy(&mol, &prep, false);
    let e_den = uhf_dfjk_energy(&mol, &prep, true);
    eprintln!("UHF DF-JK OH/cc-pVDZ: occ={e_occ:.12}  density={e_den:.12}  diff={:.2e}", (e_occ - e_den).abs());
    assert!(
        (e_occ - e_den).abs() < 1e-8,
        "UHF DF-K C_occ path vs density path SCF energy diff = {:.3e} (occ={e_occ}, density={e_den})",
        (e_occ - e_den).abs()
    );

    let pyscf = -75.39383892655523_f64;
    assert!(
        (e_occ - pyscf).abs() < 3e-3,
        "DF-JK UHF OH energy {e_occ} too far from PySCF {pyscf} (RI floor is ~1e-3)"
    );
}
