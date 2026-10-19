//! Weekly validation of the restricted open-shell range-separated hybrids
//! (ROKS) against PySCF 2.13.1 `pbc.dft.ROKS` / `KROKS` and against the
//! library's own anchors (k-mesh = supercell, RS-GDF = dense to the
//! plain-Coulomb fit error, closed-shell limit on more rows). The cheap subset
//! that every shard runs is `pbc_roks_rsh.rs`; the hypotheses, the measured
//! ledger and the limits of each anchor are in that file's header. Pins are
//! read from `testdata/reference/pbc_roks_sto3g_rsh.json`
//! (`scripts/gen_pyscf_roks_rsh_refs.py`).

mod common;
#[path = "oshell_rsh_support/mod.rs"]
mod support;

use common::*;
use support::*;

use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGrid;

/// Every PySCF ROKS row (cell, functional, mesh, exxdiv).
const ROWS: [(&str, &str, [usize; 3], ExxDiv); 11] = [
    ("h3d", "HSE06", [1, 1, 1], ExxDiv::Ewald),
    ("h3d", "HSE06", [1, 1, 1], ExxDiv::None),
    ("h3d", "CAM_B3LYP", [1, 1, 1], ExxDiv::Ewald),
    ("h3d", "CAM_B3LYP", [1, 1, 1], ExxDiv::None),
    ("h4t", "HSE06", [1, 1, 1], ExxDiv::Ewald),
    ("h4t", "CAM_B3LYP", [1, 1, 1], ExxDiv::Ewald),
    ("h3d", "HSE06", [1, 1, 2], ExxDiv::Ewald),
    ("h3d", "HSE06", [1, 1, 2], ExxDiv::None),
    ("h3d", "CAM_B3LYP", [1, 1, 2], ExxDiv::Ewald),
    ("h4t", "HSE06", [1, 1, 2], ExxDiv::Ewald),
    ("h4t", "CAM_B3LYP", [1, 1, 2], ExxDiv::Ewald),
];

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (ROKS); 11 SCFs at Gamma and on 1x1x2, ~2 min"]
fn roks_matches_pyscf_every_row() {
    for (cell, f, n, x) in ROWS {
        let c = cell_named(cell);
        let (e, it) = if n == [1, 1, 1] {
            let r = gamma_roks(&c, f, x, GRun::default());
            (r.scf.energy, r.scf.iterations)
        } else {
            let r = kroks_run(&c, f, n, x, false, None);
            (r.scf.energy, r.scf.iterations)
        };
        match roks_pin(cell, f, n, x) {
            Some(want) => {
                let d = e - want;
                eprintln!("{cell} {f} {n:?} {x:?}: E {e:.12} dE vs PySCF {d:+.2e} ({it} it)");
                assert!(d.abs() < PIN_TOL, "{cell} {f} {n:?} {x:?}: {d:.2e}");
            }
            None => eprintln!("{cell} {f} {n:?} {x:?}: E {e:.12}; PySCF did not converge"),
        }
    }
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (ROKS); 8 SCFs, ~1 min"]
fn roks_is_above_uks_on_every_uks_pinned_row() {
    // Variational ordering (UKS relaxes the ROKS constraint): a sanity check
    // on the same cell, functional, mesh and exxdiv, not a precision bar.
    for (cell, f, n, x) in ROWS {
        let Some(ro) = roks_pin(cell, f, n, x) else {
            continue;
        };
        let has_uks = !(f == "CAM_B3LYP" && x == ExxDiv::None)
            && !(cell == "h4t" && f == "CAM_B3LYP" && n == [1, 1, 2]);
        if !has_uks {
            continue;
        }
        let u = pin(cell, f, n, x);
        eprintln!("{cell} {f} {n:?} {x:?}: ROKS - UKS {:+.3e}", ro - u);
        assert!(ro >= u, "{cell} {f} {n:?} {x:?}");
    }
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (ROKS); two ROKS SCFs per functional, ~2 min"]
fn kmesh_roks_is_the_gamma_supercell_rsh() {
    let cell = cell_named("h3d");
    let n = [1, 1, 2];
    // (N_a - N_b) N_k + 1 with (N_a, N_b) = (2, 1) per cell and N_k = 2.
    let (sc, _) = supercell(&cell, n, 3);
    for f in ["HSE06", "CAM_B3LYP"] {
        let sup = GSetup::new(&sc, f, ExxDiv::Ewald, GRun::default());
        // 41 x 41 x 82 uniform points = two copies of the cell grid.
        let grid = PeriodicGrid::uniform(&sc, [MESH, MESH, 2 * MESH]).unwrap();
        let g = gamma_roks_on(&sc, f, ExxDiv::Ewald, &sup, &grid);
        let k = kroks_run(&cell, f, n, ExxDiv::Ewald, false, None);
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
#[ignore = "validation: Periodic open-shell range-separated hybrids (ROKS); 8 SCF pairs, ~30 s"]
fn closed_shell_roks_is_rks_other_rows() {
    let h2 = cell_of(&H2_ATOMS, 1);
    for f in ["HSE06", "CAM_B3LYP"] {
        for x in [ExxDiv::Ewald, ExxDiv::None] {
            let o = gamma_roks(&h2, f, x, GRun::default()).scf.energy;
            let r = gamma_rks_energy(&h2, f, x, GRun::default());
            eprintln!("closed Gamma {f} {x:?}: RO-R {:.2e}", o - r);
            assert!((o - r).abs() < CLOSED_TOL, "Gamma {f} {x:?}");
            let n = [1, 1, 2];
            let o = kroks_run(&h2, f, n, x, false, None).scf.energy;
            let r = krks_energy(&h2, f, n, x, false);
            eprintln!("closed k {n:?} {f} {x:?}: RO-R {:.2e}", o - r);
            assert!((o - r).abs() < CLOSED_TOL, "k {f} {x:?}");
        }
    }
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (ROKS); 2 SCFs, ~20 s"]
fn alpha_on_beta_and_bare_alpha_are_invisible_in_the_closed_shell_limit() {
    // Documents the artifact hypothesis: these mutants cannot be caught by
    // the closed-shell anchor, which is why they are measured on the
    // polarized cell in pbc_roks_rsh.rs.
    use ferric_pbc::kroks::KRohfMutation;
    let h2 = cell_of(&H2_ATOMS, 1);
    let n = [1, 1, 2];
    let base = kroks_run(&h2, "CAM_B3LYP", n, ExxDiv::Ewald, false, None)
        .scf
        .energy;
    for m in [KRohfMutation::KAlphaOnBeta, KRohfMutation::BareAlphaFock] {
        let e = kroks_run(&h2, "CAM_B3LYP", n, ExxDiv::Ewald, false, Some(m))
            .scf
            .energy;
        eprintln!("closed-shell {m:?}: dE {:.2e}", e - base);
        assert!((e - base).abs() < CLOSED_TOL, "{m:?}");
    }
}

#[test]
#[ignore = "validation: Periodic open-shell range-separated hybrids (ROKS); 4 fitted SCFs + PBE0 UKS references, ~1 min"]
fn rsgdf_roks_rsh_is_dense_to_the_plain_coulomb_fit_error() {
    // The PBE0 (global hybrid) UKS RS-GDF - dense is the plain-Coulomb fit
    // error on the same cell, aux and mesh; the ROKS RSH RS-GDF - dense must
    // be of the same order (the attenuated K adds a G-space fit on top of the
    // same full-Coulomb one).
    let cell = cell_named("h3d");
    let n = [1, 1, 2];
    for f in ["HSE06", "CAM_B3LYP"] {
        let dense = kroks_run(&cell, f, n, ExxDiv::Ewald, false, None)
            .scf
            .energy;
        let fit = kroks_run(&cell, f, n, ExxDiv::Ewald, true, None).scf.energy;
        let e_c = kuks_pbe0(&cell, n, true) - kuks_pbe0(&cell, n, false);
        let e_rsh = fit - dense;
        eprintln!(
            "k 1x1x2 {f}: RS-GDF - dense ROKS {e_rsh:.3e}, PBE0 UKS {e_c:.3e}, ratio {:.3}",
            e_rsh / e_c
        );
        assert!(e_rsh.abs() < RSGDF_RATIO_MAX * e_c.abs(), "{f}");
        let gd = gamma_roks(&cell, f, ExxDiv::Ewald, GRun::default())
            .scf
            .energy;
        let gf = gamma_roks(
            &cell,
            f,
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
            "Gamma {f}: RS-GDF - dense ROKS {:.3e}, PBE0 UKS {g_c:.3e}, ratio {:.3}",
            gf - gd,
            (gf - gd) / g_c
        );
        assert!((gf - gd).abs() < RSGDF_RATIO_MAX * g_c.abs(), "{f}");
    }
}

/// Bar of the RS-GDF check: |RSH ROKS fit error| below this multiple of the
/// PBE0 UKS fit error on the same cell. Measured 0.92 .. 1.84 (CAM-B3LYP adds
/// its own erf G-space fit); the K mutants move E by >= 1e-2, ~3e3 x the
/// 3e-6 fit error (ledger in `pbc_roks_rsh.rs`).
const RSGDF_RATIO_MAX: f64 = 5.0;
