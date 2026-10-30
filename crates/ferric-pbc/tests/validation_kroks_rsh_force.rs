//! Weekly validation of the restricted-open-shell range-separated k-point
//! ROKS force (dense AFT): central FD of ferric's own k-ROKS energy (fresh SCF
//! per displacement; PySCF has no periodic ROKS gradient) on EVERY component
//! of the H3 doublet and the H4 triplet for HSE06 and CAM-B3LYP, the FD floor
//! sweep, ewald vs none, the full mutation ledger, the 1x1x2 mesh vs the
//! explicit supercell, Gamma vs 1x1x1 in all four cases, the equal-fraction
//! anchor under exxdiv none and the HSE06 singlet anchor. The per-commit
//! subset and the hypotheses are in `pbc_kroks_rsh_force.rs`.
//!
//! MEASURED (test profile, 2026-10-23, 512 s for the file at 2 threads):
//! analytic - FD (h = 1e-4, ewald) worst over every component 4.47e-9 /
//! 4.56e-9 (HSE06, H3 doublet / H4 triplet), 4.52e-9 / 4.57e-9 (CAM-B3LYP);
//! the CAM-B3LYP H3 (0,2) residual falls 1.79e-8 -> 4.51e-9 -> 1.11e-9 for h =
//! 2e-4 -> 1e-4 -> 5e-5 (ratios 4.0, 4.1: h^2); ewald vs none 5.8e-16 /
//! 1.7e-16, FD under none <= 4.5e-9; 1x1x2 vs supercell 4.1e-15 / 4.2e-15;
//! Gamma vs 1x1x1 <= 5.1e-15 (routed spin-Fock guard <= 1.0e-15); equal
//! fractions under none 2.8e-16; HSE06 singlet ROKS vs RKS <= 1.2e-15 per
//! part. Bars: FD 1e-7 (>= 20x the worst residual), anchors 1e-9.
//!
//! MUTATION LEDGER (miss = max |F_mutant - F_ref|, Ha/Bohr; H3 doublet 1x1x2
//! ewald, open shell / H2 singlet 1x1x3):
//!
//! | mutant                       | CAM-B3LYP          | HSE06              |
//! |------------------------------|--------------------|--------------------|
//! | ExchBetaUsesAlpha            | 3.10e-3 / 0        | 2.64e-3 / 0        |
//! | ExchDropBeta                 | 2.16e-2 / 4.06e-2  | 1.77e-2 / 2.73e-2  |
//! | MTermTotalDensity            | 1.15e-2 / 0        | 3.77e-3 / 0        |
//! | PulayKSwap                   | 2.83e-2 / 0        | 1.23e-2 / 0        |
//! | RshKernelOnJ                 | 1.30e-1 / 1.01e-1  | 1.20e-1 / 9.03e-2  |
//! | RshPlainMadelung             | 4.37e-2 / 3.40e-2  | 7.27e-2 / 5.27e-2  |
//! | RshSwapFractions             | 3.65e-2 / 3.00e-2  | 3.45e-2 / 5.44e-2  |
//! | RshNoKernelFactor            | 9.88e-2 / 1.48e-1  | 1.03e-1 / 1.64e-1  |
//! | RshNoErfPiece                | 1.37e-2 / 3.77e-2  | 2.80e-6 / 1.22e-4  |
//! | RshOmegaScaled (w x 1.01)    | 2.80e-4 / 5.20e-4  | 5.33e-7 / 1.45e-5  |
//! | RoWDropCross                 | 8.53e-2 / 7.65e-2  | 7.22e-2 / 6.34e-2  |
//!
//! HSE06's erf piece is nearly dead on the H3 1x1x2 K set (largest weight
//! 3.6e-5 vs 0.172 for CAM-B3LYP), so its erf and w rows are resolved only at
//! 28 and 5 FD bars there (not asserted on the open shell); on the singlet
//! (1.22e-4, 1.45e-5) they are asserted > 100 FD bars. Every other
//! open-shell miss is > 100 FD bars (asserted). The `W_RO` -> unrestricted
//! `W` swap is an identity at convergence (1.1e-14, H4 triplet) and is tested
//! off stationarity in `pbc_kroks_rsh_force.rs` (refused by the gate).

mod common;
#[path = "kroks_rsh_force_support/mod.rs"]
mod ro;
#[path = "kuks_rsh_force_support/mod.rs"]
mod uks;

use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kuxcgrad::KUSpinMutation;
use ro::*;
use uks::*;

/// (d) Central FD (h = 1e-4) of the k-ROKS energy on every component: H3
/// doublet (9) and H4 triplet (12), 1x1x2, exxdiv ewald, HSE06 and
/// CAM-B3LYP. Physics: the h^2 floor; artifact: every ledger mutant misses by
/// more than the bar (`full_mutation_ledger`).
#[test]
#[ignore = "validation: every force component of H3 doublet + H4 triplet x HSE06/CAM-B3LYP by central FD, 84 fresh k-ROKS SCFs"]
fn rsh_roks_force_matches_fd_every_component() {
    for f in FUNCTIONALS {
        for (kind, pos) in [(Kind::Tri(2), &H3D[..]), (Kind::Tri(3), &H4T[..])] {
            let fx = ro_fixture(kind, pos, [1, 1, 2], f, ExxDiv::Ewald);
            assert!(ro_spin_asymmetry(&fx.res.scf) > 0.1, "open shell");
            let g = ro_force(&fx);
            let mut worst = 0.0_f64;
            for (a, x) in all_comps(pos.len()) {
                let d = ro_fd(kind, pos, a, x, FD_H, [1, 1, 2], f, ExxDiv::Ewald);
                let r = g.grad[(a, x)] - d;
                eprintln!(
                    "ROKS {f} {kind:?} ({a},{x}): analytic {:+.10e} FD {d:+.10e} diff {r:+.2e}",
                    g.grad[(a, x)]
                );
                worst = worst.max(r.abs());
            }
            eprintln!("ROKS {f} {kind:?}: worst |analytic - FD| {worst:.2e}");
            assert!(worst < FD_BAR, "{f} {kind:?}: {worst:e}");
        }
    }
}

/// FD floor of the CAM-B3LYP ROKS force (H3 doublet, component (0, 2), the
/// bond direction): the residual must fall ~4x per halving of h down to the
/// SCF noise. Artifact: a force defect gives an h-independent residual.
#[test]
#[ignore = "validation: ROKS FD floor sweep h = 2e-4, 1e-4, 5e-5 (6 fresh k-ROKS SCFs)"]
fn roks_cam_b3lyp_fd_residual_follows_h_squared() {
    let fx = ro_fixture(Kind::Tri(2), &H3D, [1, 1, 2], CAM, ExxDiv::Ewald);
    let g = ro_force(&fx).grad[(0, 2)];
    let r = |h: f64| ro_fd(Kind::Tri(2), &H3D, 0, 2, h, [1, 1, 2], CAM, ExxDiv::Ewald) - g;
    let (r2, r1, r05) = (r(2e-4), r(1e-4), r(5e-5));
    eprintln!("ROKS CAM-B3LYP H3 (0,2) residual h=2e-4 {r2:+.2e}, 1e-4 {r1:+.2e}, 5e-5 {r05:+.2e}");
    assert!(r2 / r1 > 3.5 && r2 / r1 < 6.0, "not h^2: ratio {}", r2 / r1);
    assert!(r05.abs() < FD_BAR);
}

/// exxdiv ewald vs none on the H3 doublet: the mixed Madelung shift moves
/// the energy by a geometry-independent constant, so the forces agree (the
/// M-term and the Pulay term cancel it per spin). Plus FD of the exxdiv-none
/// energy on two components. Artifact: an M-term on the wrong density or the
/// plain constant leaves an O(1e-3) gap (ledger).
#[test]
#[ignore = "validation: ROKS ewald vs none forces + FD under none (2 functionals, 10 k-ROKS SCFs)"]
fn rsh_roks_force_ewald_equals_none() {
    for f in FUNCTIONALS {
        let ge = ro_force(&ro_fixture(Kind::Tri(2), &H3D, [1, 1, 2], f, ExxDiv::Ewald));
        let gn = ro_force(&ro_fixture(Kind::Tri(2), &H3D, [1, 1, 2], f, ExxDiv::None));
        let d = diff(&ge.grad, &gn.grad);
        eprintln!("ROKS {f}: ewald vs none force {d:.2e}");
        assert!(d < 1e-9, "{f}: {d:e}");
        for (a, x) in [(0, 2), (2, 0)] {
            let r = (gn.grad[(a, x)]
                - ro_fd(Kind::Tri(2), &H3D, a, x, FD_H, [1, 1, 2], f, ExxDiv::None))
            .abs();
            eprintln!("ROKS {f} none ({a},{x}): |analytic - FD| {r:.2e}");
            assert!(r < FD_BAR, "{f} ({a},{x}): {r:e}");
        }
    }
}

/// The full mutation ledger on the H3 doublet (1x1x2, ewald), both
/// functionals, and the singlet column (H2, 1x1x3).
#[test]
#[ignore = "validation: ROKS full mutation ledger (2 functionals x 11 mutants, open shell + singlet)"]
fn full_mutation_ledger() {
    let cell = cell_of(Kind::Tri(2), &H3D);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    for f in FUNCTIONALS {
        eprintln!(
            "{f}: max erf weight on the H3 1x1x2 K set {:.2e}",
            max_erf_weight(&cell, &mesh, rsh_of(f))
        );
        let fx = ro_fixture(Kind::Tri(2), &H3D, [1, 1, 2], f, ExxDiv::Ewald);
        let base = ro_force(&fx);
        let (_, sfx, su) = ro_singlet_rows(f);
        for (n, m, sm) in ro_ledger_mutants() {
            let miss = diff(&ro_force_m(&fx, m, sm).unwrap().grad, &base.grad);
            let smiss = diff(&ro_force_m(&sfx, m, sm).unwrap().grad, &su.grad);
            eprintln!("LEDGER ROKS {f} {n}: open shell {miss:.2e} singlet {smiss:.2e}");
            let weak = f == "HSE06" && matches!(n, "RshNoErfPiece" | "RshOmegaScaled");
            if weak {
                // Nearly dead on the open-shell K set; resolved on the singlet.
                assert!(smiss > 100.0 * FD_BAR, "{f} {n}: singlet {smiss:e}");
            } else {
                assert!(miss > 100.0 * FD_BAR, "{f} {n}: {miss:e}");
            }
            if RO_SPIN_ONLY.contains(&n) {
                assert!(smiss < 1e-9, "{f} {n}: singlet {smiss:e}");
            }
        }
    }
}

/// (c) SUPERCELL: the 1x1x2 mesh ROKS force on the H3 doublet equals the
/// force on one copy in the explicit 2-cell supercell (6 H, high-spin ROKS
/// triplet). Physics: the SCF floor; artifact: a q-dependent kernel argument
/// or a mesh-vs-primitive Madelung slip is O(1e-3).
#[test]
#[ignore = "validation: ROKS 1x1x2 vs 6-atom supercell (2 functionals)"]
fn roks_mesh_force_is_the_supercell_force() {
    for f in FUNCTIONALS {
        let d = ro_mesh_vs_supercell(f);
        eprintln!("ROKS {f}: 1x1x2 vs 2-cell supercell force {d:.2e}");
        assert!(d < ANCHOR_TOL, "{f}: {d:e}");
    }
}

/// (c) Gamma == 1x1x1 in all four cases (HSE06 / CAM-B3LYP x ewald / none).
#[test]
#[ignore = "validation: Gamma vs 1x1x1 ROKS force, 4 cases (8 SCFs)"]
fn gamma_roks_rsh_force_is_the_one_point_mesh_force_all_cases() {
    for f in FUNCTIONALS {
        for exxdiv in [ExxDiv::Ewald, ExxDiv::None] {
            let (d, guard, fmax) = ro_gamma_vs_one_point(f, exxdiv);
            eprintln!(
                "ROKS {f} {exxdiv:?}: Gamma vs 1x1x1 force {d:.2e} (Fock guard {guard:.1e}, \
                 max|F| {fmax:.3e})"
            );
            assert!(d < ANCHOR_TOL, "{f} {exxdiv:?}: {d:e}");
            assert!(fmax > 1e-2);
        }
    }
}

/// (b) The equal-fraction anchor under exxdiv none (ewald runs per commit):
/// the Madelung defects are blind there by construction (asserted).
#[test]
#[ignore = "validation: ROKS equal-fraction anchor, exxdiv none (2 zero-XC k-ROKS SCFs)"]
fn equal_fractions_is_the_global_hybrid_roks_force_exxdiv_none() {
    ro_equal_fraction_case(ExxDiv::None);
}

/// (a) The HSE06 singlet anchor (CAM-B3LYP runs per commit).
#[test]
#[ignore = "validation: HSE06 singlet ROKS vs RKS (2 SCFs)"]
fn hse06_singlet_roks_is_the_rks_rsh_force() {
    let (rows, _, _) = ro_singlet_rows("HSE06");
    for (n, d) in rows {
        eprintln!("HSE06 singlet ROKS vs RKS {n}: {d:.2e}");
        assert!(d < ANCHOR_TOL, "{n}: {d:e}");
    }
}

/// The unrestricted-form `W = Σ_σ D_σ F_σ D_σ` instead of `W_RO`
/// (`RoWUhfForm`) is an identity at a converged ROKS state (they differ by ½
/// the closed-open β orbital gradient) and is therefore NOT a defect there;
/// off stationarity it is wrong, and the orbital-gradient gate is what
/// refuses that state. On the H4 triplet (1x1x1, CAM-B3LYP; NOT the H3
/// doublet, whose inversion symmetry makes the closed-open block, and with
/// it this difference, vanish for any state: measured 1.1e-16 on a loose
/// one): the production force and `RoWUhfForm` on a loosely converged state
/// are refused with the gate message. With the gate off neither `W` form is
/// right on that state (both miss the tight force by more than 10 FD bars:
/// the state is not stationary), and the two forms differ (`RoWUhfFormGateOff`
/// vs `RoGateOff`), which is the assertion that sees the swap. Measured
/// (loose SCF 1e-3 / 1e-1): g 3.6e-5, gate-off miss 1.0e-5 (W_UKS) and
/// 1.5e-5 (W_RO), W_UKS - W_RO 4.9e-6; at convergence 1.1e-14.
#[test]
#[ignore = "validation: ROKS RSH unrestricted-form W identity at convergence and refused / distinguishable off it (H4 triplet, tight + loose SCFs); CI ~128 s"]
fn unrestricted_w_is_an_identity_at_convergence_and_refused_off_it() {
    let tight = ro_fixture(Kind::Tri(3), &H4T, [1, 1, 1], CAM, ExxDiv::Ewald);
    let gt = ro_force(&tight);
    let id = diff(
        &ro_force_m(&tight, None, Some(KUSpinMutation::RoWUhfForm))
            .unwrap()
            .grad,
        &gt.grad,
    );
    eprintln!(
        "RoWUhfForm at convergence: {id:.2e} (gate value {:.1e})",
        gt.hf.commutator
    );
    assert!(id < 1e-8, "identity at convergence: {id:e}");
    let loose = ro_fixture_conv(
        cell_of(Kind::Tri(3), &H4T),
        [1, 1, 1],
        CAM,
        ExxDiv::Ewald,
        LOOSE,
    );
    for sm in [None, Some(KUSpinMutation::RoWUhfForm)] {
        let e = ro_force_m(&loose, None, sm)
            .expect_err("loose state accepted")
            .to_string();
        assert!(e.contains("ROHF orbital gradient"), "{sm:?}: {e}");
    }
    let off = ro_force_m(&loose, None, Some(KUSpinMutation::RoGateOff)).unwrap();
    let uhf = ro_force_m(&loose, None, Some(KUSpinMutation::RoWUhfFormGateOff)).unwrap();
    let g = off.hf.commutator;
    let (err_uhf, err_ro) = (diff(&uhf.grad, &gt.grad), diff(&off.grad, &gt.grad));
    let w_gap = diff(&uhf.grad, &off.grad);
    eprintln!(
        "loose CAM-B3LYP ROKS H4: g = {g:.2e}; gate-off error W_UKS {err_uhf:.2e}, W_RO \
         {err_ro:.2e}; W_UKS - W_RO {w_gap:.2e}"
    );
    // Off stationarity NEITHER form is right (both gate-off forces miss the
    // tight one); the gate's refusal is the protection. What distinguishes
    // the two W forms there is their gap.
    assert!(g > 1e-5, "state is not past the gate: {g:e}");
    assert!(
        err_ro > 10.0 * FD_BAR,
        "the loose state is not off stationarity: {err_ro:e}"
    );
    assert!(
        w_gap > 1e-6,
        "the two W forms coincide off stationarity: {w_gap:e}"
    );
}
