//! EXACTNESS ANCHOR: with TRAH disabled, every SCF result is BIT-IDENTICAL.
//!
//! `trah_trigger: None` is the default, so this is the trivial limit of the
//! whole feature: "off = unchanged". It is asserted on the raw IEEE-754 bits
//! (`f64::to_bits`), not on a tolerance — a tolerance-based version of this
//! test would pass even if the TRAH branch perturbed the arithmetic slightly,
//! which is exactly the failure mode it exists to exclude.
//!
//! Per this repo's experimental protocol the anchor is written and passing
//! BEFORE any TRAH measurement is believed. It covers every solver TRAH
//! touches (RHF, RKS, UHF, UKS) plus the interaction with the OTHER
//! second-order paths (`newton_trigger`), because the TRAH branch sits
//! immediately above the Newton branch in both loops and a mis-scoped gate
//! there would silently change which one fires.
//!
//! # Mutation record — and why the energy comparison alone was NOT enough
//!
//! Mutation 1 (`trah_armed = true` everywhere) is killed: all six cases fail.
//!
//! Mutation 2 was the informative one. Arming TRAH from a merely *present*
//! `TrahConfig` — the realistic "config present = permission to run" bug this
//! file claims to exclude — **SURVIVED** the energy-only version of this test.
//! The `FERRIC_SCF_TRACE` output showed 8 TRAH steps had actually run, with the
//! trust radius visibly differing between the two runs (Δ = 0.4 vs 0.01), yet
//! every energy stayed bit-identical: on water/cc-pVDZ the TRAH path converges
//! quadratically (‖κ‖ = 9.9e-3 → 9.7e-5 → 7.2e-7 → 6.2e-9) onto the SAME
//! stationary point DIIS finds, so at a 1e-10 convergence threshold the answers
//! agree to the last bit.
//!
//! That is a clean demonstration that comparing converged energies proves the
//! ANSWER is unchanged, not that the code path was unused. The fix is
//! [`ferric_scf::trah::TRAH_STEPS_TAKEN`], a process-wide engagement counter:
//! every case below now also asserts the counter did not advance, which is the
//! strictly stronger claim "the branch never executed" and which mutation 2
//! does not survive.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n",
        0,
        1,
    )
    .unwrap()
}

/// OH doublet — an open-shell case with a genuinely small frontier gap.
fn oh_doublet() -> Molecule {
    Molecule::parse_xyz("2\noh\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap()
}

/// Snapshot of the process-wide TRAH engagement counter.
///
/// Tests in one binary share a process and may run concurrently, so the
/// counter is read as a DELTA around the solve rather than compared to zero.
fn trah_steps() -> usize {
    ferric_scf::trah::TRAH_STEPS_TAKEN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Assert no TRAH step ran across a region.
///
/// This is the assertion that mutation 2 does not survive: it is about the
/// CODE PATH, not the answer. See the module docs for why the energy
/// comparison alone was insufficient.
fn assert_no_trah_steps(before: usize, what: &str) {
    let ran = trah_steps().saturating_sub(before);
    assert_eq!(
        ran, 0,
        "{what}: TRAH is disabled (trah_trigger = None) so the TRAH branch must \
         never execute, but {ran} TRAH step(s) were taken. A bit-identical energy \
         does NOT prove the branch was skipped — on an easy system both paths \
         converge to the same bits."
    );
}

/// Assert two energies are the SAME BITS, with a diagnostic that shows the
/// bit patterns when they are not.
fn assert_bit_identical(a: f64, b: f64, what: &str) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "{what}: TRAH-disabled must be BIT-IDENTICAL to a build without TRAH. \
         got {a:.17e} (bits {:#x}) vs {b:.17e} (bits {:#x}); Δ = {:.3e}",
        a.to_bits(),
        b.to_bits(),
        a - b
    );
}

/// The core anchor: a default config (`trah_trigger: None`) and a config that
/// differs ONLY in carrying an explicit `TrahConfig` must produce bit-identical
/// energies, densities and iteration counts.
///
/// Carrying a `TrahConfig` while `trah_trigger` is `None` must be a complete
/// no-op — the config struct being present is not permission to run.
#[test]
fn rhf_trah_off_is_bit_identical() {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let base = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    assert!(
        base.trah_trigger.is_none(),
        "TRAH must default to OFF — a default-on convergence knob would change \
         every existing result in the workspace"
    );

    // Identical except for a non-default TrahConfig that must never be read.
    let with_cfg = RhfConfig {
        trah: ferric_scf::trah::TrahConfig {
            radius0: 0.01,
            shrink: 0.1,
            grow: 5.0,
            ..Default::default()
        },
        ..base.clone()
    };

    let before = trah_steps();
    let r0 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &base).unwrap();
    let r1 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &with_cfg).unwrap();
    assert_no_trah_steps(before, "RHF/H2O/cc-pVDZ");

    assert_bit_identical(r0.energy, r1.energy, "RHF/H2O/cc-pVDZ energy");
    assert_eq!(
        r0.iterations, r1.iterations,
        "RHF iteration count must match"
    );
    assert_eq!(r0.converged, r1.converged);
    for (x, y) in r0.density_total.iter().zip(r1.density_total.iter()) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "RHF density must be bit-identical"
        );
    }
    for (x, y) in r0.mos_alpha.iter().zip(r1.mos_alpha.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "RHF MOs must be bit-identical");
    }
}

/// Same anchor for RKS (a GGA functional), where the TRAH branch would
/// additionally build an f_xc kernel if it ever fired.
#[test]
fn rks_pbe_trah_off_is_bit_identical() {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let base = RhfConfig {
        xc: Some("PBE".into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        level_shift: 0.2,
        ..Default::default()
    };
    let with_cfg = RhfConfig {
        trah: ferric_scf::trah::TrahConfig {
            radius0: 0.005,
            ..Default::default()
        },
        ..base.clone()
    };

    let before = trah_steps();
    let r0 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &base).unwrap();
    let r1 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &with_cfg).unwrap();
    assert_no_trah_steps(before, "RKS/PBE/H2O");

    assert_bit_identical(r0.energy, r1.energy, "RKS/PBE/H2O energy");
    assert_eq!(
        r0.iterations, r1.iterations,
        "RKS iteration count must match"
    );
}

/// Same anchor for UHF on an open-shell doublet.
#[test]
fn uhf_trah_off_is_bit_identical() {
    let mol = oh_doublet();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let base = RhfConfig {
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 300,
        level_shift: 0.2,
        ..Default::default()
    };
    let with_cfg = RhfConfig {
        trah: ferric_scf::trah::TrahConfig {
            radius0: 0.02,
            rho_reject: 0.9,
            ..Default::default()
        },
        ..base.clone()
    };

    let before = trah_steps();
    let r0 = solve_uhf(&ctx, &mol, &prep, &bounds, &base).unwrap();
    let r1 = solve_uhf(&ctx, &mol, &prep, &bounds, &with_cfg).unwrap();
    assert_no_trah_steps(before, "UHF/OH/6-31G");

    assert_bit_identical(r0.energy, r1.energy, "UHF/OH/6-31G energy");
    assert_eq!(
        r0.iterations, r1.iterations,
        "UHF iteration count must match"
    );
    for (x, y) in r0.density_alpha.iter().zip(r1.density_alpha.iter()) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "UHF α density must be bit-identical"
        );
    }
}

/// Same anchor for UKS/PBE.
#[test]
fn uks_pbe_trah_off_is_bit_identical() {
    let mol = oh_doublet();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let base = RhfConfig {
        xc: Some("PBE".into()),
        energy_conv: 1e-8,
        density_conv: 1e-6,
        max_iter: 300,
        level_shift: 0.3,
        ..Default::default()
    };
    let with_cfg = RhfConfig {
        trah: ferric_scf::trah::TrahConfig {
            radius0: 0.02,
            ..Default::default()
        },
        ..base.clone()
    };

    let before = trah_steps();
    let r0 = solve_uhf(&ctx, &mol, &prep, &bounds, &base).unwrap();
    let r1 = solve_uhf(&ctx, &mol, &prep, &bounds, &with_cfg).unwrap();
    assert_no_trah_steps(before, "UKS/PBE/OH");

    assert_bit_identical(r0.energy, r1.energy, "UKS/PBE/OH energy");
    assert_eq!(
        r0.iterations, r1.iterations,
        "UKS iteration count must match"
    );
}

/// The EXISTING second-order path must be untouched: with `newton_trigger` set
/// and TRAH off, the result must be bit-identical to a pre-TRAH build.
///
/// This is the interaction case. The TRAH branch sits immediately above the
/// Newton branch in both SCF loops and `continue`s when it fires, so a
/// mis-scoped TRAH gate would silently steal iterations from Newton. Comparing
/// two Newton runs that differ only in a TRAH config proves it does not.
#[test]
fn newton_trigger_path_is_unaffected_by_trah_being_compiled_in() {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let newton = RhfConfig {
        newton_trigger: 1e-2,
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let newton_plus_trah_cfg = RhfConfig {
        trah: ferric_scf::trah::TrahConfig {
            radius0: 0.001,
            ..Default::default()
        },
        ..newton.clone()
    };

    let before = trah_steps();
    let r0 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &newton).unwrap();
    let r1 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &newton_plus_trah_cfg).unwrap();
    assert_no_trah_steps(before, "RHF+Newton");

    assert_bit_identical(r0.energy, r1.energy, "RHF+Newton energy");
    assert_eq!(
        r0.iterations, r1.iterations,
        "the Newton path must take the same number of iterations whether or not a \
         TrahConfig is present"
    );
}

/// Thread-count independence of the disabled path.
///
/// Constants frozen at one worker count have broken this repo's tests before,
/// so the OFF path is checked at more than one rayon width. The energies must
/// agree bitwise across widths, which they do because the TRAH branch is not
/// merely inactive but entirely unevaluated.
#[test]
fn trah_off_is_bit_identical_across_rayon_widths() {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();

    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        trah: ferric_scf::trah::TrahConfig {
            radius0: 0.01,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut energies = Vec::new();
    for width in [1usize, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(width)
            .build()
            .unwrap();
        let e = pool.install(|| {
            let ctx = ParallelContext::default();
            solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg)
                .unwrap()
                .energy
        });
        energies.push((width, e));
    }
    let (_, e0) = energies[0];
    for &(w, e) in &energies[1..] {
        assert_eq!(
            e.to_bits(),
            e0.to_bits(),
            "TRAH-off energy must be bit-identical across rayon widths: \
             {w} threads gave {e:.17e}, 1 thread gave {e0:.17e}"
        );
    }
}
