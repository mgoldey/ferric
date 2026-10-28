//! Weekly validation of the k-point ROKS SCAN / r2SCAN energy (issue #492).
//! Every test here is `#[ignore]`d. Per commit, `pbc_kroks_mgga.rs` runs the
//! singlet (== k-RKS) and fully polarized (== k-UKS) anchors, the 1x1x3
//! PySCF `KROKS` pins of both functionals and the four defects against the
//! SCAN 1x1x3 pin. Here: all four pins (1x1x1 too) and the full mutation
//! ledger (every defect against every pin AND against both anchors, so the
//! anchors' blindness is measured, not assumed).
//!
//! System: the H3 doublet (N_alpha = 2, N_beta = 1) in the a = 4 cube, one s
//! per H (exponent 0.5), dense-AFT J, uniform 32^3 XC grid, gamma-centred
//! meshes; oracle PySCF 2.13.1 `pbc.dft.KROKS` (FFTDF 40^3, UniformGrids
//! 32^3), `scripts/gen_pyscf_kroks_mgga_refs.py`.
//!
//! MEASURED: the eprintln table of each test (numbers in the test docs).

mod common;
#[path = "kroks_mgga_support/mod.rs"]
mod support;

use support::*;

const PIN_BAR: f64 = 1e-9;
const MUTANT_BAR: f64 = 1e-6;
const MESHES: [[usize; 3]; 2] = [[1, 1, 1], [1, 1, 3]];

/// All four PySCF `KROKS` pins (SCAN / r2SCAN at 1x1x1 and 1x1x3), and every
/// defect against every pin.
#[test]
#[ignore = "validation: k-ROKS SCAN / r2SCAN energy vs PySCF KROKS, 4 pins x 4 defects (20 SCFs)"]
fn all_pins_and_every_defect() {
    let cell = cube_cell(&H3, 2);
    let bs = s05();
    let grid = uniform_grid(&cell);
    for n in MESHES {
        for f in FUNCTIONALS {
            let pin = pyscf_pin("e_tot", &key(f, n));
            let r = kroks(&cell, &bs, n, f, &grid);
            let de = r.scf.energy - pin;
            eprintln!(
                "{}: E {:.12} pin {pin:.12} dE {de:.2e}; ROKS - KUKS pin {:.3e}",
                key(f, n),
                r.scf.energy,
                r.scf.energy - pyscf_pin("e_tot_kuks", &key(f, n))
            );
            assert!(de.abs() < PIN_BAR, "{}: {de:e}", key(f, n));
            for d in DEFECTS {
                let miss = defect_miss(f, n, d, pin);
                eprintln!("  {} {d:?}: |dE| vs pin {miss:.6e}", key(f, n));
                assert!(miss > MUTANT_BAR, "{} {d:?}: {miss:e}", key(f, n));
            }
        }
    }
}

/// The anchors' reach, measured: each defect against the singlet (== k-RKS)
/// and the fully polarized (== k-UKS) anchor of `pbc_kroks_mgga.rs`. Only
/// `DropTau` is asserted to be caught; the others are expected (module doc of
/// `pbc_kroks_mgga.rs`) to be invisible there, which is why the polarized
/// pins exist. Nothing is asserted about the blind ones.
#[test]
#[ignore = "validation: k-ROKS meta-GGA defects against the singlet and fully polarized anchors (ledger)"]
fn defects_against_the_anchors() {
    let n = [1, 1, 3];
    let singlet = cube_cell(&H2, 1);
    let triplet = cube_cell(&H2, 3);
    let (s_bs, t_bs) = (s05(), two_s());
    let (s_grid, t_grid) = (uniform_grid(&singlet), uniform_grid(&triplet));
    for f in FUNCTIONALS {
        let e_rks = krks(&singlet, &s_bs, n, f, &s_grid).scf.energy;
        let e_uks = kuks(&triplet, &t_bs, n, f, &t_grid).scf.energy;
        for d in DEFECTS {
            let miss =
                |cell, bs, grid, reference: f64| match try_kroks(cell, bs, n, f, grid, Some(d)) {
                    Ok(r) => (r.scf.energy - reference).abs(),
                    Err(e) => {
                        eprintln!("  {f} {d:?}: refused ({e})");
                        f64::INFINITY
                    }
                };
            let ms = miss(&singlet, &s_bs, &s_grid, e_rks);
            let mt = miss(&triplet, &t_bs, &t_grid, e_uks);
            eprintln!("{f} {d:?}: singlet anchor {ms:.2e}, fully polarized anchor {mt:.2e}");
            if d == Defect::Tau(ferric_pbc::kuks::KUksMggaMutation::DropTau) {
                assert!(ms > MUTANT_BAR && mt > MUTANT_BAR, "{f} {d:?}");
            }
        }
    }
}
