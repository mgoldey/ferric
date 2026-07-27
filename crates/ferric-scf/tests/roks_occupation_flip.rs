//! ROKS/PBE oscillates between two electronic states; MOM fixes it.
//!
//! # Distinct from the DIIS subspace bug
//!
//! `ks_diis_degenerate_subspace.rs` covers a *different* defect (the DIIS
//! B-matrix solve failing on a converging SCF). Fixing that one cured ROKS for
//! B3LYP and ωB97X-V, which now converge in 13 iterations with no MOM. Pure-GGA
//! ROKS/PBE does not: it fails by another mechanism entirely.
//!
//! # The mechanism: occupation flipping, not a noise floor
//!
//! MEASURED (OH doublet / STO-3G / PBE, default config): the SCF reaches
//! `err_max = 6.6e-6` at iteration 6 — essentially converged — then the energy
//! JUMPS from −74.5719 to −73.9911, a swing of 0.58 Ha, and back again, over
//! and over to `max_iter`. The tell is `dp_rms`, which sits pinned at ≈0.2357
//! across iterations 4–16: the *total* density barely moves while the energy
//! swings wildly, i.e. the α/β occupied sets are trading orbitals rather than
//! the density failing to settle.
//!
//! That is aufbau occupation flip-flop, which is exactly what MOM (tracking the
//! occupied set by AO overlap with the previous accepted occupation, instead of
//! re-selecting by energy each iteration) exists to break.
//!
//! # Why the spurious state matters
//!
//! −73.9911 is not a less-precise −74.5719; it is a different, higher solution.
//! MEASURED: `diis_size = 4` *does* report `converged = true` on this system —
//! at −73.9911296313, the wrong state. So "make it converge" is not the bar; a
//! fix has to reach the RIGHT state, which is why this test asserts the energy
//! and not merely the flag.

use ferric_core::{basis, mol::Molecule, parallel::ParallelContext};
use ferric_integrals::{basis_bridge::PreparedBasis, operator::Operator};
use ferric_scf::{rhf::RhfConfig, rohf::solve_rohf, screening::SchwarzBounds};

/// The ROKS/PBE ground state for OH / STO-3G, from the MOM-converged run.
const OH_PBE_ENERGY: f64 = -74.5719102621;

/// The spurious higher state the un-MOM'd oscillation visits — asserted against
/// so a future change that "converges" to it fails loudly instead of silently
/// reporting success on the wrong solution.
const OH_PBE_SPURIOUS: f64 = -73.9911296313;

fn oh_doublet() -> (Molecule, PreparedBasis, SchwarzBounds) {
    let mol = Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    (mol, obs, bounds)
}

/// MOM converges ROKS/PBE to the correct state.
#[test]
fn roks_pbe_converges_with_mom() {
    let (mol, obs, bounds) = oh_doublet();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: Some("PBE".into()),
        mom_after_iter: 3,
        density_conv: 1e-8,
        max_iter: 100,
        ..Default::default()
    };

    let res = solve_rohf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &cfg)
        .expect("ROKS/PBE with MOM must converge");
    eprintln!("ROKS/PBE + MOM: converged={} iters={} E={:.10}", res.converged, res.iterations, res.energy);

    assert!(res.converged, "ROKS/PBE with MOM did not converge in {} iterations", res.iterations);
    assert!(
        (res.energy - OH_PBE_ENERGY).abs() < 1e-7,
        "converged to {:.10}, expected the ROKS/PBE ground state {OH_PBE_ENERGY:.10}",
        res.energy
    );
    assert!(
        (res.energy - OH_PBE_SPURIOUS).abs() > 1e-3,
        "converged to the SPURIOUS state {:.10} — the oscillation's other fixed point",
        res.energy
    );
    assert!(res.iterations < 30, "took {} iterations; MOM should settle quickly", res.iterations);
}

/// Hybrid and range-separated ROKS converge WITHOUT MOM.
///
/// These regressed together with closed-shell RKS on the DIIS subspace bug and
/// were fixed by it; pinning them here guards that fix from the open-shell side
/// and documents that MOM is a PBE-specific need, not a blanket ROKS
/// requirement.
#[test]
fn roks_hybrids_converge_without_mom() {
    let (mol, obs, bounds) = oh_doublet();
    let ctx = ParallelContext::default();

    for xc in ["B3LYP", "wB97X-V"] {
        let cfg = RhfConfig {
            xc: Some(xc.into()),
            density_conv: 1e-8,
            max_iter: 100,
            ..Default::default()
        };
        let res = solve_rohf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &cfg)
            .unwrap_or_else(|e| panic!("ROKS/{xc} errored: {e:?}"));
        eprintln!("ROKS/{xc}: converged={} iters={} E={:.10}", res.converged, res.iterations, res.energy);
        assert!(res.converged, "ROKS/{xc} must converge without MOM (took {})", res.iterations);
        assert!(res.iterations < 30, "ROKS/{xc} took {} iterations", res.iterations);
    }
}

/// Plain ROHF is unaffected — the control.
#[test]
fn plain_rohf_converges() {
    let (mol, obs, bounds) = oh_doublet();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig { density_conv: 1e-8, max_iter: 100, ..Default::default() };

    let res = solve_rohf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &cfg).unwrap();
    eprintln!("ROHF: converged={} iters={} E={:.10}", res.converged, res.iterations, res.energy);
    assert!(res.converged, "plain ROHF must converge");
    assert!(res.iterations < 20, "ROHF took {} iterations", res.iterations);
}
