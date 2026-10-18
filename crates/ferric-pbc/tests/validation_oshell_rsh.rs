//! Weekly validation of the open-shell range-separated hybrids (UKS) against
//! PySCF 2.13.1 `pbc.dft.UKS` / `KUKS` and against the library's own anchors
//! (k-mesh = supercell, RS-GDF = dense to the plain-Coulomb fit error, the
//! mutant ledger). The cheap subset that every shard runs is
//! `pbc_oshell_rsh.rs`; the hypotheses, the measured ledger and the limits of
//! each anchor are in that file's header. Pins are read from
//! `testdata/reference/pbc_oshell_sto3g_rsh.json`
//! (`scripts/gen_pyscf_oshell_rsh_refs.py`).

mod common;
#[path = "oshell_rsh_support/mod.rs"]
mod support;

use common::*;
use support::*;

use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::{gamma_uks_rsh_on_grid, GammaUksConfig, PeriodicGrid};
use ferric_pbc::kuscf::KUhfMutation;

/// The k-point rows of the PySCF pin file (cell, functional, mesh, exxdiv).
const K_ROWS: [(&str, &str, [usize; 3], ExxDiv); 8] = [
    ("h3d", "HSE06", [1, 1, 1], ExxDiv::Ewald),
    ("h3d", "HSE06", [1, 1, 1], ExxDiv::None),
    ("h3d", "CAM_B3LYP", [1, 1, 1], ExxDiv::Ewald),
    ("h3d", "HSE06", [1, 1, 2], ExxDiv::Ewald),
    ("h3d", "CAM_B3LYP", [1, 1, 2], ExxDiv::Ewald),
    ("h3d", "HSE06", [1, 1, 2], ExxDiv::None),
    ("h4t", "CAM_B3LYP", [1, 1, 1], ExxDiv::Ewald),
    ("h4t", "HSE06", [1, 1, 2], ExxDiv::Ewald),
];

fn kmesh_pin_row(cell: &str, f: &str, n: [usize; 3], x: ExxDiv) -> f64 {
    let c = cell_named(cell);
    let r = kuks_run(&c, f, n, x, false, None);
    let d = r.scf.energy - pin(cell, f, n, x);
    eprintln!(
        "{cell} {f} {n:?} {x:?}: E {:.12} dE vs PySCF {d:+.2e} ({} it, <S2> {:.4})",
        r.scf.energy, r.scf.iterations, r.scf.s2
    );
    d
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (UKS); 8 SCFs, measured ~30 s"]
fn kpoint_polarized_matches_pyscf() {
    for (cell, f, n, x) in K_ROWS {
        let d = kmesh_pin_row(cell, f, n, x);
        assert!(d.abs() < PIN_TOL, "{cell} {f} {n:?} {x:?}: {d:.2e}");
    }
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (UKS); 8 SCF pairs, measured ~30 s"]
fn kpoint_closed_shell_uks_is_rks() {
    let h2 = cell_of(&H2_ATOMS, 1);
    for n in [[1, 1, 1], [1, 1, 2]] {
        for f in ["HSE06", "CAM_B3LYP"] {
            for x in [ExxDiv::Ewald, ExxDiv::None] {
                let u = kuks_run(&h2, f, n, x, false, None);
                let r = krks_energy(&h2, f, n, x, false);
                eprintln!("closed k {n:?} {f} {x:?}: U-R {:.2e}", u.scf.energy - r);
                assert!((u.scf.energy - r).abs() < CLOSED_TOL, "{n:?} {f} {x:?}");
            }
        }
    }
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (UKS); two UKS SCFs, ~30 s"]
fn kmesh_is_the_gamma_supercell_rsh() {
    let cell = cell_named("h3d");
    let n = [1, 1, 2];
    // (N_a - N_b) N_k + 1 with (N_a, N_b) = (2, 1) per cell and N_k = 2.
    let (sc, _) = supercell(&cell, n, 3);
    for f in ["HSE06", "CAM_B3LYP"] {
        let sup = GSetup::new(&sc, f, ExxDiv::Ewald, GRun::default());
        // 41 x 41 x 82 uniform points = two copies of the cell grid.
        let grid = PeriodicGrid::uniform(&sc, [MESH, MESH, 2 * MESH]).unwrap();
        let mut cfg = GammaUksConfig::new(name_of(f));
        cfg.scf.density_conv = 1e-11;
        cfg.scf.energy_conv = 1e-12;
        let g = gamma_uks_rsh_on_grid(&sc, &sup.prep, &sup.hc, sup.ints(), &sup.ex, &grid, &cfg)
            .unwrap();
        let k = kuks_run(&cell, f, n, ExxDiv::Ewald, false, None);
        let d = k.scf.energy - g.scf.energy / 2.0;
        eprintln!(
            "{f}: E_k {:.12} E_super/2 {:.12} d {d:.2e}",
            k.scf.energy,
            g.scf.energy / 2.0
        );
        assert!(d.abs() < SUPERCELL_TOL, "{f}: {d:.2e}");
    }
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (UKS); six k-point mutants, ~1 min"]
fn kpoint_spin_mutants_fail_the_polarized_pin() {
    let cell = cell_named("h3d");
    let want = pin("h3d", "HSE06", [1, 1, 2], ExxDiv::Ewald);
    for m in [
        KUhfMutation::KAlphaOnBeta,
        KUhfMutation::KSwapSpin,
        KUhfMutation::KDropBeta,
        KUhfMutation::MadelungHalfPerSpin,
        KUhfMutation::PlainMadelungForRsh,
        KUhfMutation::KFromTotalDensity,
    ] {
        let d = kuks_run(&cell, "HSE06", [1, 1, 2], ExxDiv::Ewald, false, Some(m))
            .scf
            .energy
            - want;
        eprintln!("k mutant {m:?}: E - PySCF pin {d:+.3e}");
        assert!(d.abs() > MUT_MIN, "{m:?} not detected ({d:.2e})");
    }
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (UKS); 3 SCFs, ~20 s"]
fn alpha_on_beta_and_swap_are_invisible_in_the_closed_shell_limit() {
    // Documents the artifact hypothesis: the spin mutants cannot be caught by
    // the closed-shell anchor, which is why they are measured on polarized
    // cells above.
    let h2 = cell_of(&H2_ATOMS, 1);
    let base = kuks_run(&h2, "HSE06", [1, 1, 2], ExxDiv::Ewald, false, None)
        .scf
        .energy;
    for m in [KUhfMutation::KAlphaOnBeta, KUhfMutation::KSwapSpin] {
        let e = kuks_run(&h2, "HSE06", [1, 1, 2], ExxDiv::Ewald, false, Some(m))
            .scf
            .energy;
        eprintln!("closed-shell {m:?}: dE {:.2e}", e - base);
        assert!((e - base).abs() < CLOSED_TOL, "{m:?}");
    }
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (UKS); 4 SCFs, ~20 s"]
fn gamma_rsh_parameter_mutants_fail_the_polarized_cam_pin() {
    let cell = cell_named("h3d");
    let want = pin("h3d", "CAM_B3LYP", [1, 1, 1], ExxDiv::Ewald);
    let r = rsh_of("CAM_B3LYP");
    let mut shifts = Vec::new();
    for (label, p) in [
        (
            "omega x1.01",
            ferric_pbc::rsh::RshParams {
                omega: r.omega * 1.01,
                ..r
            },
        ),
        (
            "omega x0.01",
            ferric_pbc::rsh::RshParams {
                omega: r.omega * 0.01,
                ..r
            },
        ),
        (
            "c_lr := c_sr",
            ferric_pbc::rsh::RshParams { c_lr: r.c_sr, ..r },
        ),
        (
            "c_sr x0.99",
            ferric_pbc::rsh::RshParams {
                c_sr: r.c_sr * 0.99,
                ..r
            },
        ),
    ] {
        let run = GRun {
            rsh: Some(p),
            ..Default::default()
        };
        let d = gamma_uks(&cell, "CAM_B3LYP", ExxDiv::Ewald, run).scf.energy - want;
        eprintln!("Gamma RSH mutant {label}: E - pin {d:+.3e}");
        shifts.push(d);
    }
    // omega x1.01 is the finest (1% in the range-separation parameter).
    assert!(shifts.iter().all(|d| d.abs() > OMEGA_MUT_MIN), "{shifts:?}");
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (UKS); 1 SCF, ~10 s"]
fn polarized_global_hybrid_pbe0_matches_pyscf_and_hse06_is_distinct() {
    let cell = cell_named("h3d");
    let e_pbe0 = gamma_pbe0(&cell, false);
    let pyscf_pbe0 = pbe0_pin();
    let e_hse = pin("h3d", "HSE06", [1, 1, 1], ExxDiv::Ewald);
    eprintln!(
        "PBE0 {e_pbe0:.12} vs PySCF {pyscf_pbe0:.12}: {:.2e}; HSE06 - PBE0 {:.3e}",
        e_pbe0 - pyscf_pbe0,
        e_hse - e_pbe0
    );
    assert!((e_pbe0 - pyscf_pbe0).abs() < PIN_TOL, "polarized PBE0 pin");
    assert!(
        (e_hse - e_pbe0).abs() > 1e5 * PIN_TOL,
        "HSE06 must not collapse onto PBE0"
    );
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (UKS); 4 fitted SCFs + PBE0 references, ~30 s"]
fn rsgdf_oshell_rsh_is_dense_to_the_plain_coulomb_fit_error() {
    // PBE0 (global hybrid) RS-GDF - dense is the plain-Coulomb fit error; the
    // RSH RS-GDF - dense must be of the same order (the attenuated K adds a
    // G-space fit on top of the same full-Coulomb one).
    let cell = cell_named("h3d");
    let n = [1, 1, 2];
    let dense = kuks_run(&cell, "HSE06", n, ExxDiv::Ewald, false, None)
        .scf
        .energy;
    let fit = kuks_run(&cell, "HSE06", n, ExxDiv::Ewald, true, None)
        .scf
        .energy;
    let pbe0_d = kuks_pbe0(&cell, n, false);
    let pbe0_f = kuks_pbe0(&cell, n, true);
    let (e_rsh, e_c) = (fit - dense, pbe0_f - pbe0_d);
    eprintln!(
        "k 1x1x2: RS-GDF - dense: HSE06 {e_rsh:.3e}, PBE0 {e_c:.3e}, ratio {:.3}",
        e_rsh / e_c
    );
    assert!(e_rsh.abs() < 2.0 * e_c.abs() && e_rsh.abs() > 0.2 * e_c.abs());
    let gd = gamma_uks(&cell, "HSE06", ExxDiv::Ewald, GRun::default())
        .scf
        .energy;
    let gf = gamma_uks(
        &cell,
        "HSE06",
        ExxDiv::Ewald,
        GRun {
            rsgdf: true,
            ..Default::default()
        },
    )
    .scf
    .energy;
    let g_c = gamma_pbe0(&cell, true) - gamma_pbe0(&cell, false);
    eprintln!(
        "Gamma: RS-GDF - dense HSE06 {:.3e}, PBE0 {g_c:.3e}, ratio {:.3}",
        gf - gd,
        (gf - gd) / g_c
    );
    assert!((gf - gd).abs() < 2.0 * g_c.abs() && (gf - gd).abs() > 0.2 * g_c.abs());
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (UKS); ~10 s"]
fn gamma_closed_shell_uks_is_rks_other_rows() {
    closed_limit_case("HSE06", ExxDiv::None);
    closed_limit_case("CAM_B3LYP", ExxDiv::Ewald);
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (UKS); ~15 s"]
fn gamma_polarized_matches_pyscf_h4_triplet() {
    let cell = cell_named("h4t");
    for f in ["HSE06", "CAM_B3LYP"] {
        let r = gamma_uks(&cell, f, ExxDiv::Ewald, GRun::default());
        let d = r.scf.energy - pin("h4t", f, [1, 1, 1], ExxDiv::Ewald);
        eprintln!(
            "H4 triplet {f} Gamma: dE vs PySCF {d:.2e}, <S2> {:.4}",
            r.s2
        );
        assert!(d.abs() < PIN_TOL, "{f}: {d:.2e}");
        assert!((r.s2 - 2.0).abs() < 1e-3);
    }
}
