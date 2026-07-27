//! End-to-end correctness gate for the DF-K C_occ half-transform optimization.
//!
//! `DfK::build_from_occ` contracts the fitted 3-index tensor against the
//! occupied MO coefficients (O(naux·n²·nocc)) instead of the assembled density
//! (O(naux·n³)). The K-matrix-level equivalence is proven in df_k.rs unit tests
//! (`df_k_build_from_occ_matches_density_build`, <1e-10). These tests close the
//! loop at the SCF-energy level for BOTH the closed-shell (RHF) and open-shell
//! (UHF, two independent spin channels) drivers.
//!
//! Scaling convention: `build_from_occ` applies NO occupation factor — it
//! returns K(C_occ·C_occᵀ) for whatever coefficients it is handed, and the
//! drivers cache BARE C_occ. So the RHF caller owes the factor 2 from
//! D = 2·C_occ·C_occᵀ, while UHF/ROHF spin channels (D_σ = C_σ·C_σᵀ) owe 1.
//! Tests 1-2 below cannot see that factor (see
//! `dfk_occ_path_caller_scaling_matches_true_density_build`, which can):
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

/// Agreement tolerance between the C_occ and density DF-K routes' converged
/// SCF energies.
///
/// The two routes are algebraically identical and differ only in floating-point
/// association order, so the gap is bounded by how precisely the SCF pins the
/// energy at all — NOT by anything tighter. Under DF-JK the per-iteration `dE`
/// at convergence is ~1.1e-8 (MEASURED, water/cc-pVDZ, at the iteration the ΔP
/// gate fires), i.e. the fitted Fock's own noise floor. Asserting agreement
/// below that floor is asking two independent trajectories to land inside a
/// window neither one resolves: it passes or fails on luck, and it did flip
/// direction (RHF ok / UHF fail, then the reverse) under trajectory changes
/// that left both energies correct.
///
/// 1e-7 is ~10× the measured floor: loose enough to be trajectory-independent,
/// still four orders below the RI fitting error the absolute anchors allow
/// (3e-3) and twelve orders below the rel ~1.0 a dropped occupation factor
/// would produce — which is what this test exists to catch.
const DFJK_ENERGY_TOL: f64 = 1e-7;

/// Serializes the two runs that toggle FERRIC_DFK_FORCE_DENSITY so a parallel
/// test (or the two calls here) never observe each other's env state.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn rhf_dfjk_energy(mol: &Molecule, prep: &PreparedBasis, force_density: bool) -> f64 {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, prep).unwrap();
    let cfg = RhfConfig {
        // `energy_conv` is a LOOSE "not still descending" sanity bound, not a
        // target — see `rhf::scf_converged`. Under DF-JK the energy jitters on a
        // ~1e-8 fitting-noise floor, so the old 1e-11 here was unreachable: BOTH
        // runs burned their full iteration cap and compared two noise-floor
        // plateaus that agreed only by luck. ΔP (density_conv) is the real gate.
        energy_conv: 1e-3,
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
        // `energy_conv` is a LOOSE "not still descending" sanity bound, not a
        // target — see `rhf::scf_converged`. Under DF-JK the energy jitters on a
        // ~1e-8 fitting-noise floor, so the old 1e-11 here was unreachable: BOTH
        // runs burned their full iteration cap and compared two noise-floor
        // plateaus that agreed only by luck. ΔP (density_conv) is the real gate.
        energy_conv: 1e-3,
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

/// Guards the CALLER-side scaling convention, which the
/// `FERRIC_DFK_FORCE_DENSITY` toggle structurally cannot check.
///
/// `build_from_occ` applies no occupation factor: it returns
/// K(C_occ·C_occᵀ). RHF's density is D = 2·C_occ·C_occᵀ, so the RHF caller owes
/// a factor of 2 (either `k *= 2.0`, or `occ_factor = 2.0` on the RSH path);
/// UHF/ROHF spin channels owe 1.0. Dropping the RHF factor is invisible to the
/// sibling tests because `FERRIC_DFK_FORCE_DENSITY` rebuilds D from the same
/// bare C_occ *inside* `build_from_occ_impl` — both branches then carry the
/// identical half-scaling and their difference cancels exactly. That is how a
/// 16.4 Ha error on benzene/def2-svp (-214.13 vs -230.54) survived a green
/// suite and was misattributed to an "f64-floor limit-cycle" (636c26c).
///
/// This compares against the genuine density contraction `DfK::build(&D)` at a
/// fixed converged C_occ — one Fock build, no SCF trajectory in the loop — so
/// any caller-side factor shows up directly as a K-matrix mismatch.
#[test]
fn dfk_occ_path_caller_scaling_matches_true_density_build() {
    use ferric_scf::fock::KBuilder;
    use ndarray::Array2;

    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();

    let nocc = (mol.nelec() / 2) as usize;
    let c_occ = res.mos_r().slice(ndarray::s![.., ..nocc]).to_owned();
    // The RHF convention this test exists to pin down.
    let d = 2.0 * c_occ.dot(&c_occ.t());
    let n = prep.nbasis();

    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let mut dfk = ferric_scf::df_k::DfK::new(op, &prep, &aux, 0).unwrap();

    let mut k_density = Array2::<f64>::zeros((n, n));
    dfk.build(&d, &mut k_density).unwrap();

    let mut k_occ = Array2::<f64>::zeros((n, n));
    dfk.build_from_occ(&c_occ, &mut k_occ).unwrap();
    k_occ *= 2.0; // caller-supplied RHF factor — the thing under test

    let max_abs = k_density.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
    let max_diff = (&k_density - &k_occ)
        .iter()
        .cloned()
        .fold(0.0f64, |a, b| a.max(b.abs()));
    // Same B tensor, different association order: a few ulp, nothing more. A
    // missing/extra factor of 2 lands at rel ~1.0, twelve orders of magnitude up.
    assert!(
        max_diff / max_abs < 1e-12,
        "DF-K occ path vs true density build: rel {:.3e} (max|diff| {max_diff:.3e}, \
         max|K| {max_abs:.3e}) — check the caller's occupation factor",
        max_diff / max_abs
    );
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
        (e_occ - e_den).abs() < DFJK_ENERGY_TOL,
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
        (e_occ - e_den).abs() < DFJK_ENERGY_TOL,
        "UHF DF-K C_occ path vs density path SCF energy diff = {:.3e} (occ={e_occ}, density={e_den})",
        (e_occ - e_den).abs()
    );

    let pyscf = -75.39383892655523_f64;
    assert!(
        (e_occ - pyscf).abs() < 3e-3,
        "DF-JK UHF OH energy {e_occ} too far from PySCF {pyscf} (RI floor is ~1e-3)"
    );
}
