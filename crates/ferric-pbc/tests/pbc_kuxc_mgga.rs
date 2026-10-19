//! Analytic force and stress of the k-point open-shell UKS energy for the pure
//! meta-GGAs SCAN and r2SCAN (`ferric_pbc::kuxcgrad`, `ferric_pbc::kuxcstress`,
//! "Meta-GGA (UKS)"; issue #353), the per-commit half. The heavy anchors (FD of
//! the SCF energy, k-mesh vs supercell, the PySCF stress pin) are in
//! `validation_pbc_kuxc_mgga.rs`.
//!
//! # Artifact hypothesis next to the physics hypothesis
//!
//! * Physics right: (1) a singlet k-UKS force/stress equals the k-RKS one per
//!   part (independent SCF loops); (2) on a POLARIZED cell the XC part of the
//!   force and of the stress equals the central FD of `E_xc[D_α, D_β]` at
//!   FIXED densities (no SCF, an independent construction: the energy code's
//!   polarized kernel on rebuilt grids); (3) PBE pushed through the tau code
//!   path (`v_τ ≡ 0`) is the PBE force/stress bit for bit.
//! * Artifact: an α/β mix-up in the tau term (swapped `v_τ`, a dropped β tau
//!   term, the TOTAL τ in both libxc channels, a dropped β `V^τ`) is INVISIBLE
//!   in the singlet limit (there `τ_α = τ_β`, `v_τα = v_τβ`), so (1) can only
//!   say "no gross break"; every spin mutant must fail the POLARIZED (2), on
//!   the H3 doublet, the #479 cell, and on the non-TRIM 1x1x3 mesh (complex
//!   Bloch AOs, so a missing conjugate shows). The mutant runs themselves
//!   (and their ledger) are in `validation_pbc_kuxc_mgga.rs`, on every
//!   component; here only the production values and the Fock-guard refusal of
//!   a missing `V^τ` run per commit.
//!
//! Cost: five tests, 95 s wall in the test profile (four test threads,
//! shared box).

mod common;
#[path = "kuxc_mgga_support/mod.rs"]
mod support;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kgrad::{KGradJk, KRsGdfGradSource};
use ferric_pbc::kroks::{solve_kroks, KRoksConfig};
use ferric_pbc::kstress::KStressJk;
use ferric_pbc::kuxcgrad::{kpoint_roks_gradient, kpoint_uks_gradient};
use ferric_pbc::kuxcstress::{kpoint_roks_stress, kpoint_uks_stress};
use ferric_pbc::kxcgrad::KXcMutation;
use ferric_pbc::kxcstress::KXcStressMutation;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use std::collections::HashMap;
use support::*;

/// Singlet k-UKS vs k-RKS, per part (independent SCF loops). Measured
/// worst 8.9e-16 (SCAN and r2SCAN, H2 1x1x3); the spin-factor defects of
/// `pbc_kuxcgrad.rs` miss this anchor by >= 1e-2.
const BAR_SINGLET: f64 = 1e-11;
/// Fixed-density XC part vs its central FD. Measured worst on the H3 doublet
/// 1x1x3 (atom grid 20x50): force 6.5e-10 at h = 1e-4; stress 5.5e-8 at
/// h = 1e-4, which is the stencil's h^2 truncation (|dE_xc/deps| ~ 1), so the
/// stress uses h = 2.5e-5 (h sweep in `validation_pbc_kuxc_mgga.rs`).
const BAR_FIXED_D: f64 = 1e-7;
const FORCE_H: f64 = 1e-4;
const STRESS_H: f64 = 2.5e-5;
/// Per-commit subsets (every component, both functionals, in the validation
/// file): one component of each atom and the three diagonal plus one shear
/// strain component.
const FORCE_COMPS: [(usize, usize); 3] = [(0, 0), (1, 2), (2, 1)];
const STRAIN_COMPS: [(usize, usize); 4] = [(0, 0), (1, 1), (2, 2), (1, 2)];

fn h3(pos: &[[f64; 3]]) -> ferric_pbc::lattice::Cell {
    cube_cell(pos, 2)
}

// ---------------------------------------------------------- singlet anchor

/// (1) Singlet limit, per part, SCAN, 1x1x3 (complex Bloch AOs); r2SCAN
/// in the validation file. Blind to every α/β mix-up by construction (module
/// doc).
#[test]
fn singlet_uks_force_and_stress_equal_rks_per_part() {
    let fx = fixture(cube_cell(&H2, 1), [1, 1, 3], "SCAN", atom_grid());
    for (n, d) in singlet_rows(&fx) {
        eprintln!("SCAN singlet UKS vs RKS: {n:<17} {d:.2e}");
        assert!(d < BAR_SINGLET, "{n}: {d:e}");
    }
}

// -------------------------------------------------------- null-tau hook

/// (3) PBE through the tau path on the POLARIZED cell. Artifact hypothesis:
/// a tau path that perturbs the non-tau terms (a stray accumulation, a
/// mis-sized vector) moves the PBE numbers; physics predicts exactly 0.
#[test]
fn null_tau_path_reproduces_the_pbe_force_and_stress_exactly() {
    let fx = fixture(h3(&H3), [1, 1, 2], "PBE", atom_grid());
    let d = diff(
        &force(&fx, Some(KXcMutation::ForceTauPath), None).grad,
        &force(&fx, None, None).grad,
    );
    let s = mdiff(
        &stress(&fx, Some(KXcStressMutation::StressTauPath), None).de_deps,
        &stress(&fx, None, None).de_deps,
    );
    eprintln!("PBE through the UKS tau path: force {d:.2e}, stress {s:.2e}");
    assert_eq!(d, 0.0);
    assert_eq!(s, 0.0);
}

// ------------------------------------------------ polarized fixed-D anchors

/// (2) Force: X1 + X3 + X4 against the central FD of the fixed-density
/// `E_xc[D_α, D_β]`, H3 doublet 1x1x3, r2SCAN, one component per atom.
#[test]
fn polarized_fixed_density_xc_force_matches_fd() {
    let fx = fixture(h3(&H3), [1, 1, 3], "R2SCAN", atom_grid());
    assert!(spin_asymmetry(&fx) > 0.1, "not polarized");
    let (ok, scale, fd) = fixed_d_force_check(&h3, &H3, &fx, FORCE_H, &FORCE_COMPS);
    eprintln!("R2SCAN H3 1x1x3 XC force vs fixed-D FD: {ok:.2e} (|xc| max {scale:.2e})");
    assert!(ok < BAR_FIXED_D, "{ok:e}");
    assert!(fd.iter().any(|x| x.abs() > 1e-2), "vacuous FD");
    // Pulay side: `V^tau` missing from a Fock is refused by the guard (the
    // guard-skipped variants are measured in the weekly ledger).
    let e = try_force_as(
        &fx,
        "R2SCAN",
        &gcfg(&fx.grid, Some(KXcMutation::TauNoVxc), None),
    )
    .expect_err("the guard must refuse a Fock without V^tau")
    .to_string();
    assert!(e.contains("differs"), "{e}");
}

/// (2) Stress: `xc_ao + xc_weight` against the central FD of the
/// fixed-density `E_xc` under cell strain, SCAN, four components.
#[test]
fn polarized_fixed_density_xc_stress_matches_fd() {
    let fx = fixture(h3(&H3), [1, 1, 3], "SCAN", atom_grid());
    let fd = fd_xc_stress(&fx, STRESS_H, &STRAIN_COMPS);
    let ok = mdiff_at(&xc_stress(&stress(&fx, None, None)), &fd, &STRAIN_COMPS);
    eprintln!(
        "SCAN H3 1x1x3 XC stress vs fixed-D FD: {ok:.2e} (|xc| max {:.2e})",
        mmax(&fd)
    );
    assert!(ok < BAR_FIXED_D, "{ok:e}");
}

// ------------------------------------------------------------ refusals

fn et_sp_aux() -> BasisSet {
    let mut shells = Vec::new();
    for a in [4.7, 1.9, 0.75, 0.3] {
        shells.push(Shell {
            l: 0,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-s-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

/// Outside the validated envelope, by name: meta-GGA on the RS-GDF J (the
/// open-shell meta-GGA energy is refused there), any other meta-GGA (raw
/// TPSS components), and meta-GGA ROKS (its SCF has no tau term). Each
/// refusal precedes the Fock guard, so a PBE state exercises it.
#[test]
fn unvalidated_open_shell_meta_gga_derivatives_are_refused() {
    let fx = fixture(h3(&H3), [1, 1, 1], "SCAN", atom_grid());
    for f in ["MGGA_X_TPSS", "MGGA_C_TPSS"] {
        let e = try_force_as(&fx, f, &gcfg(&fx.grid, None, None))
            .expect_err(f)
            .to_string();
        assert!(e.contains("SCAN and r2SCAN"), "{f}: {e}");
        let e = try_stress_as(&fx, f, &scfg(&fx.grid, None, None))
            .expect_err(f)
            .to_string();
        assert!(e.contains("SCAN and r2SCAN"), "{f}: {e}");
    }
    // RS-GDF J/K.
    let aux = prep_for(&fx.cell, &et_sp_aux());
    let kc = KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    };
    let gdf = KRsGdf::build(&fx.cell, &fx.prep, &aux, &fx.mesh, &fx.hk.s, &kc).unwrap();
    let src = || KRsGdfGradSource {
        gdf: &gdf,
        cfg: &kc,
        aux: &aux,
    };
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let e = kpoint_uks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        KGradJk::RsGdf(src()),
        &fx.res.scf,
        "SCAN",
        &gcfg(&fx.grid, None, None),
    )
    .expect_err("SCAN force on RS-GDF")
    .to_string();
    eprintln!("UKS SCAN force on RS-GDF: {e}");
    assert!(e.contains("RS-GDF") && e.contains("dense"), "{e}");
    let e = kpoint_uks_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        KStressJk::RsGdf(src()),
        &fx.res.scf,
        "SCAN",
        &scfg(&fx.grid, None, None),
    )
    .expect_err("SCAN stress on RS-GDF")
    .to_string();
    assert!(e.contains("RS-GDF") && e.contains("dense"), "{e}");
    // ROKS: a PBE ROKS state handed in as SCAN.
    let mut rcfg = KRoksConfig::new(&fx.cell, ExxDiv::None, "PBE");
    rcfg.rohf.hcore = hcfg;
    if let ferric_pbc::kxcgrad::KXcGrid::Atom(g) = &fx.grid {
        rcfg.grid = g.clone();
    }
    let ro = solve_kroks(&fx.cell, &fx.prep, None, &fx.mesh, &rcfg).unwrap();
    let e = kpoint_roks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        KGradJk::Dense(&fx.eri),
        &ro.scf,
        "SCAN",
        &gcfg(&fx.grid, None, None),
    )
    .expect_err("ROKS SCAN force")
    .to_string();
    eprintln!("ROKS SCAN force: {e}");
    assert!(e.contains("meta-GGA") && e.contains("ROKS"), "{e}");
    let e = kpoint_roks_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        KStressJk::Dense(&fx.eri),
        &ro.scf,
        "SCAN",
        &scfg(&fx.grid, None, None),
    )
    .expect_err("ROKS SCAN stress")
    .to_string();
    assert!(e.contains("meta-GGA") && e.contains("ROKS"), "{e}");
}
