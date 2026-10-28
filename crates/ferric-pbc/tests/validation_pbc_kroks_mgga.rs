//! Weekly validation of the k-point ROKS SCAN / r2SCAN energy, force and
//! stress (issue #492).
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
//! Force and stress (item 3; per commit, `pbc_kroks_mgga.rs` runs the SCAN
//! singlet / fully polarized anchors and one FD row): every force and strain
//! component of the H3 doublet at 1x1x3 against the central FD of ferric's
//! own converged k-ROKS energy, for SCAN and r2SCAN, with the mutation
//! ledger against the same FD; both anchors for r2SCAN too, with each
//! defect's reach there measured; and the FD of the GAMMA-point ROKS SCAN
//! energy (an independent energy code) against the 1x1x1 k-ROKS force and
//! stress. No external pin: PySCF has no k-point ROKS gradient and its KUKS
//! force refuses meta-GGA.
//!
//! MEASURED: the eprintln table of each test (numbers in the test docs and
//! the PR).

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

// ------------------------------------------- force and stress (#492, item 3)

use common::*;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{PeriodicGrid, PeriodicXc, PeriodicXcConfig};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kroks::KRoScfResult;
use ferric_pbc::kuxcgrad::KUSpinMutation;
use ferric_pbc::kuxcstress::KUStressMutation;
use ferric_pbc::kxcgrad::KXcMutation;
use ferric_pbc::kxcstress::KXcStressMutation;
use ferric_pbc::lattice::Cell;
use ferric_pbc::rohf::{gamma_roks_with_xc, GammaRoksConfig};
use ferric_pbc::uhf::GammaUhfIntegrals;

/// FD of the converged SCF energy (uniform 32^3, SCF to 1e-13 / orbital
/// gradient ~1e-11): bar on every component, the force at `H_MAIN`, the
/// stress on the Richardson extrapolation of `H_MAIN` and `H_MAIN / 2`. The
/// diagonal strains (1,1) and (2,2) carry an h^2 stencil error of 1.3e-7 /
/// 6e-8 at h = 5e-5 (falling 4x per halving, the same for PBE), which the
/// extrapolation removes; the force and the other strains are below 1e-9 at
/// h = 5e-5.
const FD_BAR: f64 = 1e-8;
const DERIV_MUTANT_BAR: f64 = 1e-5;
const H_SWEEP: [f64; 2] = [1e-4, 5e-5];
const H_MAIN: f64 = 5e-5;

/// `(4 FD(h/2) - FD(h)) / 3`: the central difference with its h^2 term
/// removed.
fn richardson(fd_h: &[f64], fd_half: &[f64]) -> Vec<f64> {
    fd_h.iter()
        .zip(fd_half)
        .map(|(a, b)| (4.0 * b - a) / 3.0)
        .collect()
}

fn all_force_comps(n: usize) -> Vec<(usize, usize)> {
    (0..n).flat_map(|a| (0..3).map(move |x| (a, x))).collect()
}

fn all_strain_comps() -> Vec<(usize, usize)> {
    (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).collect()
}

/// One ledger row: `Ok(miss)` against the FD, or the refusal text.
type Row = (String, Result<f64, String>);

fn force_ledger(fx: &DerivFx, scf: &KRoScfResult, f: &str, fd: &[f64]) -> Vec<Row> {
    use KUSpinMutation as U;
    let comps = all_force_comps(3);
    let cases: Vec<(String, Option<KXcMutation>, Option<U>)> = vec![
        ("TauNoVxcBeta".into(), None, Some(U::TauNoVxcBeta)),
        ("TauSpinSwap".into(), None, Some(U::TauSpinSwap)),
        ("TauDropBeta".into(), None, Some(U::TauDropBeta)),
        (
            "TauTotalBothChannels".into(),
            None,
            Some(U::TauTotalBothChannels),
        ),
        ("RoWDropCross".into(), None, Some(U::RoWDropCross)),
        ("TauDoubled".into(), Some(KXcMutation::TauDoubled), None),
        ("TauNoAo".into(), Some(KXcMutation::TauNoAo), None),
        (
            "TauNoVxcUnguarded+RoGateOff".into(),
            Some(KXcMutation::TauNoVxcUnguarded),
            Some(U::RoGateOff),
        ),
        ("RoWUhfForm".into(), None, Some(U::RoWUhfForm)),
    ];
    cases
        .into_iter()
        .map(|(name, x, s)| {
            let r = roks_force(fx, scf, f, &gcfg(fx, x, s))
                .map(|g| worst_force(&g.grad, fd, &comps))
                .map_err(|e| e.to_string());
            (name, r)
        })
        .collect()
}

fn stress_ledger(fx: &DerivFx, scf: &KRoScfResult, f: &str, fd: &[f64]) -> Vec<Row> {
    use KUStressMutation as U;
    let comps = all_strain_comps();
    let cases: Vec<(String, Option<KXcStressMutation>, Option<U>)> = vec![
        ("TauNoVxcBeta".into(), None, Some(U::TauNoVxcBeta)),
        ("TauSpinSwap".into(), None, Some(U::TauSpinSwap)),
        ("TauDropBeta".into(), None, Some(U::TauDropBeta)),
        (
            "TauTotalBothChannels".into(),
            None,
            Some(U::TauTotalBothChannels),
        ),
        ("RoWDropCross".into(), None, Some(U::RoWDropCross)),
        (
            "TauDoubled".into(),
            Some(KXcStressMutation::TauDoubled),
            None,
        ),
        (
            "NoTauStrain".into(),
            Some(KXcStressMutation::NoTauStrain),
            None,
        ),
        (
            "TauTranspose".into(),
            Some(KXcStressMutation::TauTranspose),
            None,
        ),
        (
            "TauNoVxc+RoGateOff".into(),
            Some(KXcStressMutation::TauNoVxc),
            Some(U::RoGateOff),
        ),
        ("RoWUhfForm".into(), None, Some(U::RoWUhfForm)),
    ];
    cases
        .into_iter()
        .map(|(name, x, s)| {
            let r = roks_stress(fx, scf, f, &scfg(fx, x, s))
                .map(|g| worst_stress(&g.de_deps, fd, &comps))
                .map_err(|e| e.to_string());
            (name, r)
        })
        .collect()
}

/// Every defect must miss by more than the bar; `TauNoVxcBeta` must instead
/// be refused by the ROHF orbital-gradient gate (it changes the rebuilt `F_β`
/// on which the gate is evaluated); `RoWUhfForm` is an identity at
/// convergence (`W_UHF` and `W_RO` differ only by orbital-gradient terms) and
/// must stay within the FD bar. `TauTranspose` is recorded, not asserted: on
/// the uniform grid the stress is symmetric, so the tau strain term equals its
/// transpose and the defect is invisible by construction (measured 7.3e-10,
/// the production residual); it is caught on the atom grid of
/// `validation_pbc_kuxc_mgga.rs`, where the stress is not symmetric.
fn check_ledger(label: &str, rows: &[Row]) {
    for (name, r) in rows {
        match r {
            Ok(miss) => eprintln!("  LEDGER {label} {name:<28} |d - FD| {miss:.2e}"),
            Err(e) => eprintln!("  LEDGER {label} {name:<28} refused: {e}"),
        }
        match (name.as_str(), r) {
            ("TauNoVxcBeta", Err(e)) => assert!(e.contains("ROHF orbital gradient"), "{e}"),
            ("TauNoVxcBeta", Ok(m)) => panic!("{label} TauNoVxcBeta not refused ({m:e})"),
            ("RoWUhfForm", Ok(m)) => assert!(*m < FD_BAR, "{label} RoWUhfForm moved: {m:e}"),
            ("TauTranspose", Ok(_)) => {}
            (_, Ok(m)) => assert!(*m > DERIV_MUTANT_BAR, "{label} {name} survives: {m:e}"),
            (_, Err(e)) => panic!("{label} {name}: {e}"),
        }
    }
}

/// k-ROKS SCAN / r2SCAN force (all nine components) and `dE/dε` (all nine
/// strain components) on the polarized H3 doublet, 1x1x3, against the
/// central FD of ferric's own converged k-ROKS energy (h sweep for SCAN), then
/// the mutation ledger against the `H_MAIN` FD.
#[test]
#[ignore = "validation: k-ROKS SCAN / r2SCAN force and stress vs FD of the SCF energy (H3 doublet 1x1x3, 9 + 9 components, h sweep, ~126 SCFs) plus the mutation ledger"]
fn h3_kroks_force_and_stress_match_fd_with_ledger() {
    let cell = cube_cell(&H3, 2);
    let n = [1, 1, 3];
    let fx = deriv_fx(&cell, &s05(), n);
    let (fc, sc) = (all_force_comps(3), all_strain_comps());
    for f in FUNCTIONALS {
        let r = kroks(&cell, &s05(), n, f, &uniform_grid(&cell));
        assert!(spin_asymmetry(&r.scf) > 0.1, "not polarized");
        let g = roks_force(&fx, &r.scf, f, &gcfg(&fx, None, None)).unwrap();
        let s = roks_stress(&fx, &r.scf, f, &scfg(&fx, None, None)).unwrap();
        eprintln!("{f}: orbital gradient {:.2e}", g.hf.commutator);
        let hs: &[f64] = if f == "SCAN" { &H_SWEEP } else { &[H_MAIN] };
        let mut fd_f = vec![];
        for &h in hs {
            fd_f = fd_roks_force(&H3, n, f, h, &fc);
            let wf = worst_force(&g.grad, &fd_f, &fc);
            eprintln!("{f} h = {h:.1e}: force vs FD {wf:.2e}");
        }
        let wf = worst_force(&g.grad, &fd_f, &fc);
        assert!(wf < FD_BAR, "{f} force {wf:e}");
        let s_h = fd_roks_stress(&cell, n, f, H_MAIN, &sc);
        let s_half = fd_roks_stress(&cell, n, f, H_MAIN / 2.0, &sc);
        let fd_s = richardson(&s_h, &s_half);
        let (w1, w2, ws) = (
            worst_stress(&s.de_deps, &s_h, &sc),
            worst_stress(&s.de_deps, &s_half, &sc),
            worst_stress(&s.de_deps, &fd_s, &sc),
        );
        eprintln!(
            "{f}: dE/deps vs FD {w1:.2e} (h = {H_MAIN:.1e}), {w2:.2e} (h/2), {ws:.2e} (Richardson)"
        );
        assert!(ws < FD_BAR, "{f} stress {ws:e}");
        check_ledger(&format!("{f} force"), &force_ledger(&fx, &r.scf, f, &fd_f));
        check_ledger(
            &format!("{f} stress"),
            &stress_ledger(&fx, &r.scf, f, &fd_s),
        );
    }
}

/// The derivative anchors for both functionals (SCAN per commit), and the
/// reach of each defect there, measured (numbers in the PR): the singlet
/// (== k-RKS) anchor is blind to `TauSpinSwap` only (`v_τα = v_τβ`); the
/// fully polarized (== k-UKS) anchor to `TauDropBeta`, `TauNoVxcBeta` and
/// `RoWDropCross` (all multiply `D_β = 0`). Only the production anchors are
/// asserted.
#[test]
#[ignore = "validation: k-ROKS meta-GGA force / stress anchors (singlet == k-RKS, N_beta = 0 == k-UKS) for SCAN and r2SCAN, with the defects' reach"]
fn derivative_anchors_and_their_reach() {
    let n = [1, 1, 3];
    for (label, cell, bs) in [
        ("singlet", cube_cell(&H2, 1), s05()),
        ("fully polarized", cube_cell(&H2, 3), two_s()),
    ] {
        let fx = deriv_fx(&cell, &bs, n);
        let grid = uniform_grid(&cell);
        for f in FUNCTIONALS {
            let ro = kroks(&cell, &bs, n, f, &grid);
            let oracle = if label == "singlet" {
                rks_derivs(&fx, &krks(&cell, &bs, n, f, &grid).scf, f)
            } else {
                uks_derivs(&fx, &kuks(&cell, &bs, n, f, &grid).scf, f)
            };
            let prod = (
                roks_force(&fx, &ro.scf, f, &gcfg(&fx, None, None)).unwrap(),
                roks_stress(&fx, &ro.scf, f, &scfg(&fx, None, None)).unwrap(),
            );
            let worst = part_rows(&prod, &oracle)
                .iter()
                .fold(0.0_f64, |m, (_, d)| m.max(*d));
            eprintln!("{label} {f}: worst per-part |ROKS - oracle| {worst:.2e}");
            assert!(worst < 1e-11, "{label} {f}: {worst:e}");
            let fcomps = all_force_comps(2);
            let fo: Vec<f64> = fcomps.iter().map(|&(a, x)| oracle.0.grad[(a, x)]).collect();
            let so: Vec<f64> = all_strain_comps()
                .iter()
                .map(|&(i, j)| oracle.1.de_deps[i][j])
                .collect();
            use KUSpinMutation as U;
            for m in [
                U::TauSpinSwap,
                U::TauDropBeta,
                U::TauTotalBothChannels,
                U::TauNoVxcBeta,
                U::RoWDropCross,
            ] {
                let r = roks_force(&fx, &ro.scf, f, &gcfg(&fx, None, Some(m)))
                    .map(|g| format!("{:.2e}", worst_force(&g.grad, &fo, &fcomps)))
                    .unwrap_or_else(|e| format!("refused ({e})"));
                eprintln!("  REACH {label} {f} force {m:?}: {r}");
            }
            for m in [
                KUStressMutation::TauSpinSwap,
                KUStressMutation::RoWDropCross,
            ] {
                let r = roks_stress(&fx, &ro.scf, f, &scfg(&fx, None, Some(m)))
                    .map(|s| format!("{:.2e}", worst_stress(&s.de_deps, &so, &all_strain_comps())))
                    .unwrap_or_else(|e| format!("refused ({e})"));
                eprintln!("  REACH {label} {f} stress {m:?}: {r}");
            }
        }
    }
}

/// An independent construction of the derivative's 1x1x1 limit: the central
/// FD of the GAMMA-point ROKS SCAN energy (`gamma_roks_with_xc`, real AOs,
/// ferric-dft's `eval_tau_uks`; its own SCF) against the 1x1x1 k-ROKS force
/// and `dE/dε` (complex Bloch AOs, `chunk_tau`), H3 doublet, s05, uniform
/// 32^3, every component, h = 5e-5 (the stress on the Richardson
/// extrapolation with h / 2, see `FD_BAR`). The 1x1x1 k-ROKS energy == Gamma ROKS
/// anchor is in `pbc_gamma_open_mgga.rs`; it is re-checked here on this cell.
#[test]
#[ignore = "validation: FD of the Gamma ROKS SCAN energy vs the 1x1x1 k-ROKS force and stress (H3 doublet, 9 + 9 components, 55 Gamma SCFs)"]
fn gamma_roks_scan_fd_matches_the_one_point_kroks_force_and_stress() {
    let f = "SCAN";
    let n = [1, 1, 1];
    let cell = cube_cell(&H3, 2);
    let fx = deriv_fx(&cell, &s05(), n);
    let r = kroks(&cell, &s05(), n, f, &uniform_grid(&cell));
    let g = roks_force(&fx, &r.scf, f, &gcfg(&fx, None, None)).unwrap();
    let s = roks_stress(&fx, &r.scf, f, &scfg(&fx, None, None)).unwrap();
    let e0 = gamma_roks_energy(&cell, f);
    eprintln!(
        "H3 {f}: E_gamma {e0:.12} E_k111 {:.12} dE {:.2e}",
        r.scf.energy,
        e0 - r.scf.energy
    );
    assert!((e0 - r.scf.energy).abs() < 1e-9);
    let (fc, sc) = (all_force_comps(3), all_strain_comps());
    let h = H_MAIN;
    let fd_f: Vec<f64> = fc
        .iter()
        .map(|&(a, x)| {
            let ep = gamma_roks_energy(&cube_cell(&moved(&H3, a, x, h), 2), f);
            let em = gamma_roks_energy(&cube_cell(&moved(&H3, a, x, -h), 2), f);
            (ep - em) / (2.0 * h)
        })
        .collect();
    let fd_strain = |h: f64| -> Vec<f64> {
        sc.iter()
            .map(|&(i, j)| {
                let ep = gamma_roks_energy(&strained(&cell, i, j, h), f);
                let em = gamma_roks_energy(&strained(&cell, i, j, -h), f);
                (ep - em) / (2.0 * h)
            })
            .collect()
    };
    let (s_h, s_half) = (fd_strain(h), fd_strain(h / 2.0));
    eprintln!(
        "H3 {f}: dE/deps vs the Gamma FD {:.2e} (h), {:.2e} (h/2)",
        worst_stress(&s.de_deps, &s_h, &sc),
        worst_stress(&s.de_deps, &s_half, &sc)
    );
    let fd_s = richardson(&s_h, &s_half);
    let wf = worst_force(&g.grad, &fd_f, &fc);
    let ws = worst_stress(&s.de_deps, &fd_s, &sc);
    eprintln!("H3 {f}: 1x1x1 k-ROKS force vs FD of the Gamma energy {wf:.2e}, dE/deps {ws:.2e}");
    assert!(fd_f.iter().any(|x| x.abs() > 1e-2), "vacuous FD");
    assert!(wf < FD_BAR, "force {wf:e}");
    assert!(ws < FD_BAR, "stress {ws:e}");
}

/// The Gamma-point ROKS energy of `cell` (s05, the pins' uniform grid,
/// dense-AFT J/K, its own SCF to 1e-13).
fn gamma_roks_energy(cell: &Cell, f: &str) -> f64 {
    let bs = s05();
    let prep = prep_for(cell, &bs);
    let hc = periodic_hcore(cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA_HCORE)).unwrap();
    let eri = DenseAftEri::build(
        cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let grid = PeriodicGrid::uniform(cell, [PIN_GRID; 3]).unwrap();
    let mut xc =
        PeriodicXc::new_open_mgga(cell, &bs, f, &grid, &PeriodicXcConfig::default()).unwrap();
    let mut cfg = GammaRoksConfig::new(f);
    cfg.scf.energy_conv = 1e-13;
    cfg.scf.density_conv = 1e-10;
    cfg.scf.max_iter = 300;
    let r = gamma_roks_with_xc(
        cell,
        &prep,
        &hc,
        GammaUhfIntegrals::DenseAft(&eri),
        &mut xc,
        &cfg,
    )
    .unwrap_or_else(|e| panic!("gamma_roks {f}: {e}"));
    assert!(r.scf.converged);
    r.scf.energy
}
