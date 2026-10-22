//! Weekly validation of the open-shell range-separated k-point UKS force
//! (dense AFT): central FD of ferric's own polarized energy (fresh SCF per
//! displacement) on EVERY component of the H3 doublet and the H4 triplet for
//! HSE06 and CAM-B3LYP, the FD floor sweep, ewald vs none, the full mutation
//! ledger, the 1x1x2 mesh vs the explicit supercell, Gamma vs 1x1x1 in all
//! four cases, and the HSE06 singlet anchor. The per-commit subset, the
//! hypotheses and the ledger table are in `pbc_kuks_rsh_force.rs`.
//!
//! MEASURED (release test profile, 2026-10-19, 633 s for the file): analytic
//! - FD (h = 1e-4, ewald) worst over every component 3.79e-9 / 4.63e-9 (HSE06,
//! H3 doublet / H4 triplet), 3.82e-9 / 4.70e-9 (CAM-B3LYP); the CAM-B3LYP H3
//! (0,2) residual falls 1.50e-8 -> 3.82e-9 -> 9.16e-10 for h = 2e-4 -> 1e-4 ->
//! 5e-5 (ratio 3.9, 4.2: h^2); ewald vs none force 2.8e-14 / 5.6e-14, FD under
//! none <= 3.8e-9; 1x1x2 vs supercell 7.2e-15 / 6.7e-15; Gamma vs 1x1x1
//! <= 8.5e-14 (routed Fock guard <= 8.9e-16); HSE06 singlet UKS vs RKS
//! <= 1.2e-14 per part.

mod common;
#[path = "kuks_rsh_force_support/mod.rs"]
mod support;

use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::kpts::KPointMesh;
use support::*;

/// (2) Central FD (h = 1e-4) of the polarized k-UKS energy on every
/// component: H3 doublet (9) and H4 triplet (12), 1x1x2, exxdiv ewald, HSE06
/// and CAM-B3LYP. Physics: the h^2 floor (~4e-9); artifact: every ledger
/// mutant misses by >= 5.1e-7 (HSE06 w x 1.01), 2.7e-6 (HSE06 erf piece),
/// 2.7e-4 (CAM-B3LYP w x 1.01), all others by >= 2.5e-3.
#[test]
#[ignore = "validation: every force component of H3 doublet + H4 triplet x HSE06/CAM-B3LYP by central FD, 84 fresh k-UKS SCFs"]
fn rsh_uks_force_matches_fd_every_component() {
    for f in FUNCTIONALS {
        for (kind, pos) in [(Kind::Tri(2), &H3D[..]), (Kind::Tri(3), &H4T[..])] {
            let fx = fixture(kind, pos, [1, 1, 2], f, ExxDiv::Ewald);
            assert!(spin_asymmetry(&fx.res.scf) > 0.1, "polarized");
            let g = force(&fx);
            let mut worst = 0.0_f64;
            for (a, x) in all_comps(pos.len()) {
                let d = fd(kind, pos, a, x, 1e-4, [1, 1, 2], f, ExxDiv::Ewald);
                let r = g.grad[(a, x)] - d;
                eprintln!(
                    "{f} {kind:?} ({a},{x}): analytic {:+.10e} FD {d:+.10e} diff {r:+.2e}",
                    g.grad[(a, x)]
                );
                worst = worst.max(r.abs());
            }
            eprintln!("{f} {kind:?}: worst |analytic - FD| {worst:.2e}");
            assert!(worst < FD_BAR, "{f} {kind:?}: {worst:e}");
        }
    }
}

/// FD floor of the polarized CAM-B3LYP force (H3 doublet, component (0, 2),
/// the bond direction): the residual must fall ~4x per halving of h down to
/// the SCF noise. Artifact: a force defect gives an h-independent residual.
#[test]
#[ignore = "validation: FD floor sweep h = 2e-4, 1e-4, 5e-5 (6 fresh k-UKS SCFs)"]
fn polarized_cam_b3lyp_fd_residual_follows_h_squared() {
    let fx = fixture(Kind::Tri(2), &H3D, [1, 1, 2], CAM, ExxDiv::Ewald);
    let g = force(&fx).grad[(0, 2)];
    let r = |h: f64| fd(Kind::Tri(2), &H3D, 0, 2, h, [1, 1, 2], CAM, ExxDiv::Ewald) - g;
    let (r2, r1, r05) = (r(2e-4), r(1e-4), r(5e-5));
    eprintln!("CAM-B3LYP H3 (0,2) residual h=2e-4 {r2:+.2e}, 1e-4 {r1:+.2e}, 5e-5 {r05:+.2e}");
    assert!(r2 / r1 > 3.5 && r2 / r1 < 6.0, "not h^2: ratio {}", r2 / r1);
    assert!(r05.abs() < FD_BAR);
}

/// exxdiv ewald vs none on the polarized H3 doublet: the mixed Madelung shift
/// moves each spin's occupied levels rigidly and the energy by a
/// geometry-independent constant, so the forces agree (the M-term and the
/// Pulay term cancel it per spin). Plus FD of the exxdiv-none energy on two
/// components. Artifact: an M-term on the wrong density or the plain constant
/// leaves an O(1e-3) gap (ledger).
#[test]
#[ignore = "validation: ewald vs none forces + FD under none (2 functionals, 10 k-UKS SCFs)"]
fn rsh_uks_force_ewald_equals_none() {
    for f in FUNCTIONALS {
        let ge = force(&fixture(Kind::Tri(2), &H3D, [1, 1, 2], f, ExxDiv::Ewald));
        let gn = force(&fixture(Kind::Tri(2), &H3D, [1, 1, 2], f, ExxDiv::None));
        let d = diff(&ge.grad, &gn.grad);
        eprintln!("{f}: ewald vs none UKS force {d:.2e}");
        assert!(d < 1e-9, "{f}: {d:e}");
        for (a, x) in [(0, 2), (2, 0)] {
            let r = (gn.grad[(a, x)]
                - fd(Kind::Tri(2), &H3D, a, x, 1e-4, [1, 1, 2], f, ExxDiv::None))
            .abs();
            eprintln!("{f} none ({a},{x}): |analytic - FD| {r:.2e}");
            assert!(r < FD_BAR, "{f} ({a},{x}): {r:e}");
        }
    }
}

/// The full mutation ledger (module doc of `pbc_kuks_rsh_force.rs`) on the
/// polarized H3 doublet, both functionals, and the singlet column.
#[test]
#[ignore = "validation: full mutation ledger (2 functionals x 10 mutants, polarized + singlet)"]
fn full_mutation_ledger() {
    let cell = cell_of(Kind::Tri(2), &H3D);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    for f in FUNCTIONALS {
        eprintln!(
            "{f}: max erf weight on the H3 1x1x2 K set {:.2e}",
            max_erf_weight(&cell, &mesh, rsh_of(f))
        );
        let fx = fixture(Kind::Tri(2), &H3D, [1, 1, 2], f, ExxDiv::Ewald);
        let base = force(&fx);
        let (_, sfx, su) = singlet_rows(f);
        for (n, m, sm) in ledger_mutants() {
            let miss = diff(&force_m(&fx, m, sm).unwrap().grad, &base.grad);
            let smiss = diff(&force_m(&sfx, m, sm).unwrap().grad, &su.grad);
            eprintln!("LEDGER {f} {n}: polarized {miss:.2e} singlet {smiss:.2e}");
            let weak = f == "HSE06" && matches!(n, "RshNoErfPiece" | "RshOmegaScaled");
            if !weak {
                assert!(miss > 100.0 * FD_BAR, "{f} {n}: {miss:e}");
            }
            if SPIN_ONLY.contains(&n) {
                assert!(smiss < 1e-9, "{f} {n}: singlet {smiss:e}");
            }
        }
    }
}

/// (4) SUPERCELL: the 1x1x2 mesh force on the polarized H3 doublet equals the
/// force on one copy in the explicit 2-cell supercell (6 H, multiplicity 3).
/// Physics: the SCF floor; artifact: a q-dependent kernel argument or a
/// mesh-vs-primitive Madelung slip shows as O(1e-3).
#[test]
#[ignore = "validation: 1x1x2 vs 6-atom supercell (2 functionals)"]
fn polarized_mesh_force_is_the_supercell_force() {
    for f in FUNCTIONALS {
        let d = mesh_vs_supercell(f);
        eprintln!("{f}: 1x1x2 vs 2-cell supercell force {d:.2e}");
        assert!(d < 1e-9, "{f}: {d:e}");
    }
}

/// (5) Gamma == 1x1x1 in all four cases (HSE06 / CAM-B3LYP x ewald / none).
#[test]
#[ignore = "validation: Gamma vs 1x1x1 UKS force, 4 cases (8 SCFs)"]
fn gamma_uks_rsh_force_is_the_one_point_mesh_force_all_cases() {
    for f in FUNCTIONALS {
        for exxdiv in [ExxDiv::Ewald, ExxDiv::None] {
            let (d, guard, fmax) = gamma_vs_one_point(f, exxdiv);
            eprintln!(
                "{f} {exxdiv:?}: Gamma vs 1x1x1 UKS force {d:.2e} (Fock guard {guard:.1e}, \
                 max|F| {fmax:.3e})"
            );
            assert!(d < 1e-9, "{f} {exxdiv:?}: {d:e}");
            assert!(fmax > 1e-2);
        }
    }
}

/// (3) The equal-fraction anchor under exxdiv none (ewald runs per commit):
/// the Madelung defects are blind there by construction (asserted).
#[test]
#[ignore = "validation: equal-fraction anchor, exxdiv none (2 zero-XC k-UKS SCFs)"]
fn equal_fractions_is_the_global_hybrid_uks_force_exxdiv_none() {
    equal_fraction_case(ExxDiv::None);
}

/// (1) The HSE06 singlet anchor (CAM-B3LYP runs per commit).
#[test]
#[ignore = "validation: HSE06 singlet UKS vs RKS (2 SCFs)"]
fn hse06_singlet_uks_is_the_rks_rsh_force() {
    let (rows, _, _) = singlet_rows("HSE06");
    for (n, d) in rows {
        eprintln!("HSE06 singlet UKS vs RKS {n}: {d:.2e}");
        assert!(d < 1e-9, "{n}: {d:e}");
    }
}
