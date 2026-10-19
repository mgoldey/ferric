//! Validation (weekly) of the k-point open-shell UKS meta-GGA (SCAN, r2SCAN)
//! force and stress (`ferric_pbc::kuxcgrad`, `ferric_pbc::kuxcstress`; issue
//! #353). Per-commit anchors are in `pbc_kuxc_mgga.rs`.
//!
//! * FD of ferric's OWN converged k-UKS energy on the polarized cells of the
//!   #479 energy pins: every force component and all nine strain components,
//!   with the FD noise floor measured by an h sweep; the production-wired
//!   mutation ledger (spin and tau defects, Pulay-side tau defects included)
//!   is measured against the SAME finite differences.
//! * k-mesh vs supercell: the 1x1x3 force (per part) and stress against the
//!   explicit 3-cell supercell at 1x1x1 (same code; checks the k sum and the
//!   Bloch phases of the per-spin tau terms).
//! * PySCF 2.13.1 `pbc.grad.kuks_stress` (FFTDF + UniformGrids 61^3) for
//!   SCAN / r2SCAN on the H3 doublet (`scripts/gen_pyscf_kuks_mgga_stress_refs.py`,
//!   `testdata/reference/pbc_h3_sto3g_kuks_stress_mgga.json`). PySCF's k-point
//!   UKS FORCE refuses meta-GGA, so the force has no external pin.

mod common;
#[path = "kuxc_mgga_support/mod.rs"]
mod support;

use common::*;
use ferric_pbc::kuxcgrad::KUSpinMutation;
use ferric_pbc::kuxcstress::KUStressMutation;
use ferric_pbc::kxcgrad::{KXcGrid, KXcMutation};
use ferric_pbc::kxcstress::KXcStressMutation;
use ferric_pbc::lattice::Cell;
use ndarray::Array2;
use support::*;

/// FD of the SCF energy (atom grid, cutoff 14 Bohr, SCF to 1e-13). Measured
/// worst: force 8.4e-9 (h = 2e-4) falling to 7.7e-10 (h = 5e-5); stress
/// 6.4e-8 (h = 1e-4) falling to 5.3e-9 (h = 2.5e-5), both h^2 (the stencil);
/// the smallest mutant miss is 1.95e-3, 2e4x above the bar.
const FD_BAR: f64 = 1e-7;
const MUTANT_BAR: f64 = 1e-5;
const H_SWEEP: [f64; 3] = [2e-4, 1e-4, 5e-5];
const H_MAIN: f64 = 1e-4;

fn h3(pos: &[[f64; 3]]) -> Cell {
    cube_cell(pos, 2)
}

fn to_array(n: usize, comps: &[(usize, usize)], v: &[f64]) -> Array2<f64> {
    let mut o = Array2::<f64>::zeros((n, 3));
    for (&(a, x), &y) in comps.iter().zip(v) {
        o[(a, x)] = y;
    }
    o
}

fn to_mat(comps: &[(usize, usize)], v: &[f64]) -> [[f64; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for (&(i, j), &y) in comps.iter().zip(v) {
        o[i][j] = y;
    }
    o
}

/// The force ledger against one FD reference: XC-side (spin, tau) and
/// Pulay-side defects, all on the polarized fixture.
fn force_ledger(fx: &Fx, fd: &Array2<f64>, label: &str) {
    use KUSpinMutation as U;
    let xm = [
        KXcMutation::TauNoAo,
        KXcMutation::TauDoubled,
        KXcMutation::TauNoVxcUnguarded,
    ];
    let sm = [
        U::TauSpinSwap,
        U::TauDropBeta,
        U::TauTotalBothChannels,
        U::TauNoVxcBeta,
        U::XcSpinSwap,
        U::XcSpinAveraged,
    ];
    let cases = xm
        .iter()
        .map(|m| (format!("{m:?}"), Some(*m), None))
        .chain(sm.iter().map(|m| (format!("{m:?}"), None, Some(*m))));
    for (name, x, s) in cases {
        let miss = diff(&force(fx, x, s).grad, fd);
        eprintln!("  LEDGER {label} force {name:<22} |F - FD| {miss:.2e}");
        assert!(miss > MUTANT_BAR, "{label} {name} survives: {miss:e}");
    }
}

fn stress_ledger(fx: &Fx, fd: &[[f64; 3]; 3], label: &str) {
    use KUStressMutation as U;
    let xm = [
        KXcStressMutation::NoTauStrain,
        KXcStressMutation::TauDoubled,
        KXcStressMutation::TauNoVxc,
    ];
    let sm = [
        U::TauSpinSwap,
        U::TauDropBeta,
        U::TauTotalBothChannels,
        U::TauNoVxcBeta,
        U::XcSpinSwap,
        U::XcSpinAveraged,
    ];
    let cases = xm
        .iter()
        .map(|m| (format!("{m:?}"), Some(*m), None))
        .chain(sm.iter().map(|m| (format!("{m:?}"), None, Some(*m))));
    for (name, x, s) in cases {
        let miss = mdiff(&stress(fx, x, s).de_deps, fd);
        eprintln!("  LEDGER {label} stress {name:<22} |dE/deps - FD| {miss:.2e}");
        assert!(miss > MUTANT_BAR, "{label} {name} survives: {miss:e}");
    }
}

/// Force vs FD of the SCF energy, H3 doublet 1x1x3, all nine components, h
/// sweep; then the mutation ledger against the h = 1e-4 FD.
#[test]
#[ignore = "validation: k-UKS meta-GGA force vs FD of the SCF energy (H3 doublet 1x1x3, 9 components x 3 steps x 2 functionals) plus the mutation ledger"]
fn h3_force_matches_fd_of_the_scf_energy_with_h_sweep() {
    let comps = all_comps(3);
    for f in FUNCTIONALS {
        let grid = atom_grid();
        let fx = fixture(h3(&H3), [1, 1, 3], f, grid.clone());
        let g = force(&fx, None, None);
        let mut at_main = None;
        for h in H_SWEEP {
            let fd = to_array(
                3,
                &comps,
                &fd_scf_force(&h3, &H3, [1, 1, 3], f, &grid, h, &comps),
            );
            let d = diff(&g.grad, &fd);
            eprintln!("{f} H3 1x1x3 force vs SCF FD, h = {h:.0e}: max |F - FD| {d:.2e}");
            assert!(d < FD_BAR, "{f} h {h}: {d:e}");
            if h == H_MAIN {
                at_main = Some(fd);
            }
        }
        eprintln!("{f} |F| max {:.3e}", amax(&g.grad));
        force_ledger(&fx, &at_main.unwrap(), &format!("{f} H3"));
    }
}

/// Force vs FD of the SCF energy on the triclinic H4 triplet (N_alpha = 3,
/// N_beta = 1), 1x1x2, all twelve components, h = 1e-4.
#[test]
#[ignore = "validation: k-UKS meta-GGA force vs FD of the SCF energy (triclinic H4 triplet 1x1x2, 12 components x 2 functionals)"]
fn h4_triplet_force_matches_fd_of_the_scf_energy() {
    let comps = all_comps(4);
    for f in FUNCTIONALS {
        let grid = atom_grid();
        let fx = fixture(tri_cell(&TRI_ATOMS), [1, 1, 2], f, grid.clone());
        assert!(spin_asymmetry(&fx) > 0.1);
        let g = force(&fx, None, None);
        let fd = to_array(
            4,
            &comps,
            &fd_scf_force(&tri_cell, &TRI_ATOMS, [1, 1, 2], f, &grid, H_MAIN, &comps),
        );
        let d = diff(&g.grad, &fd);
        eprintln!(
            "{f} H4 triplet 1x1x2 force vs SCF FD: {d:.2e} (|F| max {:.3e})",
            amax(&g.grad)
        );
        assert!(d < FD_BAR, "{f}: {d:e}");
    }
}

/// Stress vs FD of the strained SCF energy, all nine components, h sweep, H3
/// doublet 1x1x3; then the ledger against the smallest-h FD. The strain FD
/// carries an h^2 stencil error that is NOT negligible at h = 2e-4 (|dE/deps|
/// ~ 1; the fixed-density stress alone shows 2.2e-7 there, the SCF FD
/// 2.6e-7), so the sweep is h = 1e-4 .. 2.5e-5 and the test also asserts that
/// the residual falls by > 2.5x per halving of h from 1e-4 (the stencil, not
/// the analytic stress, carries it).
#[test]
#[ignore = "validation: k-UKS meta-GGA stress vs FD of the strained SCF energy (H3 doublet 1x1x3, 9 components x 3 steps x 2 functionals) plus the mutation ledger"]
fn h3_stress_matches_fd_of_the_scf_energy_with_h_sweep() {
    for f in FUNCTIONALS {
        let grid = atom_grid();
        let fx = fixture(h3(&H3), [1, 1, 3], f, grid.clone());
        let s = stress(&fx, None, None);
        let mut res = Vec::new();
        let mut last = None;
        for h in STRESS_SWEEP {
            let fd = to_mat(
                &ALL9,
                &fd_scf_stress(&fx.cell, [1, 1, 3], f, &grid, h, &ALL9),
            );
            let d = mdiff(&s.de_deps, &fd);
            eprintln!("{f} H3 1x1x3 dE/deps vs SCF FD, h = {h:.1e}: {d:.2e}");
            assert!(d < FD_BAR, "{f} h {h}: {d:e}");
            res.push(d);
            last = Some(fd);
        }
        assert!(res[0] / res[1] > 2.5, "{f}: not stencil-dominated {res:?}");
        eprintln!(
            "{f} |dE/deps| max {:.3e}, antisymmetry {:.2e}",
            mmax(&s.de_deps),
            s.antisymmetry()
        );
        stress_ledger(&fx, &last.unwrap(), &format!("{f} H3"));
    }
}

const STRESS_SWEEP: [f64; 3] = [1e-4, 5e-5, 2.5e-5];

/// Stress vs FD on the triclinic H4 triplet, 1x1x2, all nine components.
#[test]
#[ignore = "validation: k-UKS meta-GGA stress vs FD of the strained SCF energy (triclinic H4 triplet 1x1x2, 9 components x 2 functionals)"]
fn h4_triplet_stress_matches_fd_of_the_scf_energy() {
    for f in FUNCTIONALS {
        let grid = atom_grid();
        let fx = fixture(tri_cell(&TRI_ATOMS), [1, 1, 2], f, grid.clone());
        let s = stress(&fx, None, None);
        let fd = to_mat(
            &ALL9,
            &fd_scf_stress(&fx.cell, [1, 1, 2], f, &grid, H_MAIN, &ALL9),
        );
        let d = mdiff(&s.de_deps, &fd);
        eprintln!(
            "{f} H4 triplet 1x1x2 dE/deps vs SCF FD: {d:.2e} (|dE/deps| max {:.3e})",
            mmax(&s.de_deps)
        );
        assert!(d < FD_BAR, "{f}: {d:e}");
    }
}

// ----------------------------------------- fixed-density XC, every component

/// The per-commit fixed-density anchors of `pbc_kuxc_mgga.rs` on EVERY
/// component, both functionals, with an h sweep (the FD floor): force (all
/// nine) and stress (all nine) of the H3 doublet 1x1x3, plus the XC-side
/// spin/tau ledger against the smallest-h FD; and the singlet anchor for
/// r2SCAN.
#[test]
#[ignore = "validation: k-UKS meta-GGA fixed-density XC force and stress vs FD on every component, h sweep, 2 functionals, plus the r2SCAN singlet anchor"]
fn fixed_density_xc_force_and_stress_every_component() {
    let comps = all_comps(3);
    for f in FUNCTIONALS {
        let fx = fixture(h3(&H3), [1, 1, 3], f, atom_grid());
        let mut last_force = None;
        for h in [2e-4, 1e-4, 5e-5, 2.5e-5] {
            let (ok, scale, fd) = fixed_d_force_check(&h3, &H3, &fx, h, &comps);
            eprintln!("{f} fixed-D XC force h = {h:.1e}: {ok:.2e} (|xc| max {scale:.2e})");
            assert!(ok < 1e-7, "{f} h {h}: {ok:e}");
            last_force = Some(fd);
        }
        for (name, miss) in force_xc_mutant_misses(&fx, &last_force.unwrap(), &comps) {
            eprintln!("  LEDGER {f} fixed-D force {name:<22} {miss:.2e}");
            assert!(miss > MUTANT_BAR, "{f} {name}: {miss:e}");
        }
        let xc = xc_stress(&stress(&fx, None, None));
        let mut last_stress = None;
        for h in [2e-4, 1e-4, 5e-5, 2.5e-5] {
            let fd = fd_xc_stress(&fx, h, &ALL9);
            let ok = mdiff(&xc, &fd);
            eprintln!(
                "{f} fixed-D XC stress h = {h:.1e}: {ok:.2e} (|xc| max {:.2e})",
                mmax(&fd)
            );
            last_stress = Some(fd);
        }
        let fd = last_stress.unwrap();
        let ok = mdiff(&xc, &fd);
        assert!(ok < 1e-7, "{f} stress at h = 2.5e-5: {ok:e}");
        for (name, miss) in stress_xc_mutant_misses(&fx, &fd, &ALL9) {
            eprintln!("  LEDGER {f} fixed-D stress {name:<22} {miss:.2e}");
            assert!(miss > MUTANT_BAR, "{f} {name}: {miss:e}");
        }
    }
    let fx = fixture(cube_cell(&H2, 1), [1, 1, 3], "R2SCAN", atom_grid());
    for (n, d) in singlet_rows(&fx) {
        eprintln!("R2SCAN singlet UKS vs RKS: {n:<17} {d:.2e}");
        assert!(d < 1e-11, "{n}: {d:e}");
    }
}

// --------------------------------------------------- k-mesh vs supercell

fn supercell_of(pos: &[[f64; 3]], n: usize, mult: usize) -> Cell {
    let mut atoms = Vec::new();
    for m in 0..n {
        for p in pos {
            atoms.push([p[0], p[1], p[2] + 4.0 * m as f64]);
        }
    }
    let mut lat = cubic(4.0);
    lat[2][2] = 4.0 * n as f64;
    let mut mol = hydrogens(&atoms);
    mol.multiplicity = mult;
    Cell::new(mol, lat).unwrap()
}

/// 1x1x3 (non-TRIM: complex Bloch AOs) vs the explicit 3-cell supercell at
/// 1x1x1, multiplicity 4 (N_alpha - N_beta = 3), per force part on one copy of
/// each atom and the stress tensor.
#[test]
#[ignore = "validation: k-UKS meta-GGA 1x1x3 vs the 3-cell supercell (force per part and stress), 2 functionals"]
fn k_mesh_equals_the_supercell_force_and_stress() {
    for f in FUNCTIONALS {
        let k = fixture(h3(&H3), [1, 1, 3], f, atom_grid());
        let sc = fixture(supercell_of(&H3, 3, 4), [1, 1, 1], f, atom_grid());
        assert!((k.res.scf.energy - sc.res.scf.energy / 3.0).abs() < 1e-8);
        let (gk, gs) = (force(&k, None, None), force(&sc, None, None));
        let first = |a: &Array2<f64>| a.slice(ndarray::s![0..3, ..]).to_owned();
        for (n, a, b) in [
            ("xc_ao", &gk.xc_ao, &gs.xc_ao),
            ("xc_point", &gk.xc_point, &gs.xc_point),
            ("xc_weight", &gk.xc_weight, &gs.xc_weight),
            ("total", &gk.grad, &gs.grad),
        ] {
            let d = diff(a, &first(b));
            eprintln!("{f} 1x1x3 vs supercell force {n:<9} {d:.2e}");
            // Measured <= 1.1e-13.
            assert!(d < 1e-9, "{f} {n}: {d:e}");
        }
        let (sk, ss) = (stress(&k, None, None), stress(&sc, None, None));
        let d = mdiff(&sk.sigma, &ss.sigma);
        eprintln!("{f} 1x1x3 vs supercell sigma {d:.2e}");
        assert!(d < 1e-10, "{f} sigma: {d:e}");
    }
}

// --------------------------------------------------------- PySCF anchor

fn read_pin(key: &str, field: &str) -> [[f64; 3]; 3] {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/pbc_h3_sto3g_kuks_stress_mgga.json"
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("reference json")).unwrap();
    assert_eq!(doc["pyscf_version"], "2.13.1");
    let rows = &doc["rows"][key][field];
    let mut want = [[0.0; 3]; 3];
    for (i, row) in want.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = rows[i][j].as_f64().expect("entry");
        }
    }
    want
}

/// PySCF 2.13.1 `kuks_stress` vs ferric on the uniform 61^3 grid, H3
/// doublet 1x1x2 (the settings of the closed-shell meta-GGA anchor in
/// `pbc_kxcstress_mgga.rs`, plus `cell.precision` 1e-12 and per-mesh
/// `nelec`): the full sigma, and the fixed-density XC part (`get_vxc`
/// without J and the nuclear term) against `xc_ao + xc_weight`. The tau and
/// spin defects miss the same reference.
#[test]
#[ignore = "validation: k-UKS meta-GGA stress vs PySCF kuks_stress (61^3 uniform grid, H3 doublet 1x1x2, SCAN and r2SCAN)"]
fn uniform_grid_stress_matches_pyscf_kuks_stress() {
    for f in FUNCTIONALS {
        let want = read_pin(f, "sigma");
        let want_xc = read_pin(f, "xc_de_deps");
        let fx = fixture(h3(&H3), [1, 1, 2], f, KXcGrid::Uniform([61, 61, 61]));
        let s = stress(&fx, None, None);
        let d = mdiff(&s.sigma, &want);
        let dx = mdiff(&xc_stress(&s), &want_xc);
        eprintln!(
            "{f} 61^3 1x1x2 vs PySCF: |sigma - pyscf| {d:.2e} (xx {:.9e} vs {:.9e}); \
             XC part {dx:.2e} (|xc| max {:.3e})",
            s.sigma[0][0],
            want[0][0],
            mmax(&want_xc)
        );
        assert!(d < PYSCF_BAR, "{f}: {d:e}");
        assert!(dx < PYSCF_XC_BAR, "{f} xc: {dx:e}");
        for (name, x, sm) in [
            ("NoTauStrain", Some(KXcStressMutation::NoTauStrain), None),
            ("TauSpinSwap", None, Some(KUStressMutation::TauSpinSwap)),
            (
                "TauTotalBothChannels",
                None,
                Some(KUStressMutation::TauTotalBothChannels),
            ),
        ] {
            let m = mdiff(&stress(&fx, x, sm).sigma, &want);
            eprintln!("  {f} {name} vs PySCF: {m:.2e}");
            assert!(m > 1e3 * PYSCF_BAR, "{f} {name}: {m:e}");
        }
    }
}

/// Measured 5.8e-11 (SCAN) / 7.7e-11 (r2SCAN), the closed-shell anchor's
/// order; the tau/spin mutants miss by 1.9e-4 to 1.4e-3.
const PYSCF_BAR: f64 = 1e-9;
/// The fixed-density XC part is compared at each code's OWN converged
/// density (a ferric density cannot be fed to PySCF's `get_vxc` here), so it
/// carries the density difference: measured 3.7e-9 / 5.1e-9.
const PYSCF_XC_BAR: f64 = 1e-7;
