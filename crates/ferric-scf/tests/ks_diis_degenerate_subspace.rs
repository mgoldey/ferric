//! Regression: KS-DFT must reach the default density threshold.
//!
//! # The bug
//!
//! DIIS builds a B-matrix from `B[i][j] = <err_i, err_j>`, whose entries scale
//! as ‖err‖². A *converging* SCF therefore drives the whole matrix toward zero.
//! Two defects compounded on that:
//!
//! 1. `solve_linear` rejected a system on an ABSOLUTE pivot floor (`< 1e-14`).
//!    Once ‖err‖ ~ 1e-8 the B entries are ~1e-16, so a perfectly well-formed
//!    system was reported singular purely because it had been rescaled.
//! 2. On a rejected solve, `Diis::step` returned the un-extrapolated Fock and
//!    then never recovered — each later iteration pushed another near-dependent
//!    vector onto an already-degenerate history.
//!
//! MEASURED before the fix (water / STO-3G / PBE, default config): the solve
//! first failed at subspace size 7 and failed on all 73 remaining iterations,
//! leaving the SCF running bare Roothaan steps. `dp_rms` bottomed at 1.8e-8 —
//! just above the 1e-8 gate — then drifted *upward* by 10× over 110 iterations
//! while the energy stayed pinned at −75.2261674 ± 2e-8. The run never
//! converged at any iteration cap.
//!
//! Plain HF was unaffected (its error vectors stay large enough to clear the
//! absolute floor), which is why this presented as "DFT can't converge".
//!
//! # The fix
//!
//! `solve_linear` scales the B-matrix by its largest entry before elimination,
//! making the pivot test relative (genuine conditioning) rather than absolute;
//! and `solve_diis_coeffs` shrinks the subspace from the oldest end and retries
//! when the system really is rank-deficient, instead of giving up permanently.

use ferric_core::{basis, mol::Molecule, parallel::ParallelContext};
use ferric_integrals::{basis_bridge::PreparedBasis, operator::Operator};
use ferric_scf::{
    rhf::{solve_rhf, RhfConfig},
    screening::SchwarzBounds,
};

fn water_sto3g() -> (Molecule, PreparedBasis, SchwarzBounds) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    (mol, obs, bounds)
}

/// THE REGRESSION: RKS/PBE must converge at the DEFAULT diis_size.
///
/// Pinned at the default (not an explicit small subspace) because that is
/// precisely what broke: a *smaller* `diis_size` masked the bug by evicting the
/// degenerate directions before they could poison the solve, so a test that set
/// `diis_size` explicitly would have passed against the unfixed code.
#[test]
fn rks_pbe_converges_at_default_diis_size() {
    let (mol, obs, bounds) = water_sto3g();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: Some("PBE".into()),
        density_conv: 1e-8,
        // Generous, but far past the ~8 iterations a healthy run needs: the
        // unfixed code failed at ANY cap (it drifted away from the solution
        // rather than approaching it slowly).
        max_iter: 100,
        ..Default::default()
    };

    let res = solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &cfg).unwrap();
    eprintln!(
        "RKS/PBE water/STO-3G: converged={} iters={} E={:.12}",
        res.converged, res.iterations, res.energy
    );

    assert!(
        res.converged,
        "RKS/PBE must reach density_conv=1e-8; got {} iterations at E={:.12} \
         (DIIS subspace conditioning regression)",
        res.iterations, res.energy
    );
    // A healthy DIIS converges this system in well under 20 iterations; the
    // unfixed code degenerated into unaccelerated Roothaan steps.
    assert!(
        res.iterations < 20,
        "converged but took {} iterations — DIIS is not accelerating",
        res.iterations
    );
}

/// The converged energy must be independent of `diis_size`.
///
/// DIIS is an accelerator: it may change the path, never the fixed point. Under
/// the bug the default-size run had no fixed point to report at all, while
/// `diis_size = 4` converged — so agreement across sizes is the statement that
/// the fix restored the accelerator rather than merely perturbing it.
#[test]
fn converged_energy_is_independent_of_diis_size() {
    let (mol, obs, bounds) = water_sto3g();
    let ctx = ParallelContext::default();

    let mut energies = Vec::new();
    for size in [4usize, 8, 12] {
        let cfg = RhfConfig {
            xc: Some("PBE".into()),
            density_conv: 1e-8,
            max_iter: 100,
            diis_size: size,
            ..Default::default()
        };
        let res = solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &cfg).unwrap();
        eprintln!(
            "diis_size={size:2}: converged={} iters={:3} E={:.12}",
            res.converged, res.iterations, res.energy
        );
        assert!(res.converged, "diis_size={size} failed to converge");
        energies.push(res.energy);
    }

    let lo = energies.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // Different subspace sizes take different paths to the same fixed point, so
    // they agree only to the SCF's own resolution (the ΔP gate at 1e-8), not to
    // machine precision.
    assert!(
        hi - lo < 1e-7,
        "converged energy depends on diis_size: spread {:.3e} over {energies:?}",
        hi - lo
    );
}

/// Plain HF must be untouched by the DIIS change.
///
/// HF converged fine before the fix (its error vectors never shrank into the
/// absolute-floor trap), so it is the control: the fix must not perturb the
/// path or the answer on a case that was already healthy.
#[test]
fn plain_hf_still_converges_quickly() {
    let (mol, obs, bounds) = water_sto3g();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig { density_conv: 1e-8, max_iter: 100, ..Default::default() };

    let res = solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &cfg).unwrap();
    eprintln!("RHF water/STO-3G: iters={} E={:.12}", res.iterations, res.energy);
    assert!(res.converged, "plain RHF must converge");
    assert!(res.iterations < 20, "RHF took {} iterations", res.iterations);
}
