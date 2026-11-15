//! The analytic SCS-MP2 nuclear gradient against a central finite
//! difference of ferric's own SCS-MP2 energy.
//!
//! # What is compared
//!
//! [`scs_mp2_gradient_analytical`] (the relaxed RI-MP2 gradient with the pair
//! weight `T̃ = (c_os + c_ss) t − c_ss t̄` and the matching unrelaxed
//! densities; derivation on `ferric_mp2::rimp2::SpinScaling`) against
//! `(E(+h) − E(−h)) / 2h`, `E` = [`scs_mp2`]`.total_energy` on an RHF
//! converged to `ΔP ≤ 1e-9`, the same RI aux basis, every Cartesian component.
//! Systems: distorted H2O in STO-3G and cc-pVDZ, distorted NH3 in 6-31G, all
//! with cc-pvdz-ri; scalings Grimme's (6/5, 1/3) and the SOS limit (1, 0).
//! Also H2O/cc-pVDZ with one frozen core orbital, and H2O/cc-pVDZ on an RI-JK
//! reference ([`total_scs_mp2_gradient_on`], `scf_jk = "rijk"`), whose FD
//! energy is the SCS-MP2 energy of the fitted SCF.
//!
//! # Bar: 2e-7 Ha/Bohr, from the FD step and the SCF and Z-vector tolerances
//!
//! The derivation of `rimp2_gradient_rijk.rs`, unchanged by the scaling
//! (`c_os, c_ss ≤ 6/5` scale the correlation part, which is a small fraction
//! of `E'''` and of the residual couplings):
//!
//! * central-difference truncation `h²|E'''|/6`, `h = 5e-4` Bohr,
//!   `|E'''| ≲ 2 Ha/Bohr³`: ≤ 8e-8;
//! * energy noise between the displaced SCFs (`ΔP ≤ 1e-9`, energy converged to
//!   `O(ΔP)` in the non-variational correlation part times an O(0.1) coupling,
//!   plus ≲ 1e-11 Ha integral-screen noise) over `2h`: ≤ 2e-8;
//! * the analytic side's orbital residual and Z-vector residual
//!   (`max|r| < 1e-8`) times O(1) couplings: ≤ 2e-8.
//!
//! Sum ≤ 1.2e-7, rounded up to 2e-7.
//!
//! # Exactness anchor
//!
//! `scs_unit_scaling_is_the_rimp2_gradient_bit_for_bit`: `(c_os, c_ss) =
//! (1, 1)` reproduces [`rimp2_gradient_analytical`] bit for bit on H2O and
//! NH3 (and H2O with a frozen core). Both run the same intermediates code, so
//! this pins the SCS entry point's route and configuration (frozen core,
//! J/K route), not the arithmetic of the shared code; that the RI-MP2
//! gradient itself is unchanged by the pair-weight threading was checked
//! bit for bit against a build of the parent commit.
//!
//! # Mutations (each fails on an assertion; source restored)
//!
//! * `c_os` and `c_ss` swapped in the SCS entry: H2O/STO-3G Grimme misses FD
//!   by 2.3e-2;
//! * the same-spin part dropped from the unrelaxed densities only
//!   (`P = c_os P_os`): 2.7e-4;
//! * the 3c/2c RI response on the MP2 pair weight `2t − t̄`: 5.1e-3;
//! * frozen core not passed to the SCS intermediates: the frozen-core FD case
//!   misses by 5.6e-4 and the anchor's fc=1 case differs in 9 of 9 bits;
//! * the scaled path on four-centre response J/K under an RI-JK reference:
//!   the RI-JK case misses by 2.5e-5.
//!
//! # Negative control
//!
//! The former construction, the RI-MP2 gradient with its correlation part
//! scaled by `(c_os + c_ss)/2`, is rebuilt here and must miss the FD of the
//! SCS energy by more than `NEGATIVE_FACTOR` times the bar whenever
//! `c_os ≠ c_ss`.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::gradient::{
    rimp2_gradient_analytical, scs_mp2_gradient_analytical, total_scs_mp2_gradient_on,
};
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::scs::{scs_mp2, ScsMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::scf_jk::{ScfJk, ScfJkMode};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::warm_start::ScfWarmStart;
use ferric_scf::ScfResult;
use ndarray::Array2;

const H: f64 = 5e-4;
const BAR: f64 = 2e-7;
/// The old `(c_os + c_ss)/2` scaling must miss FD by more than this many bars.
const NEGATIVE_FACTOR: f64 = 100.0;
const AUX: &str = "cc-pvdz-ri";

const WATER: &str = "3\nwater, off C2v\nO 0.02 -0.01 0.117790\nH 0.0 0.785453 -0.451161\nH 0.03 -0.735453 -0.491161\n";
const AMMONIA: &str = "4\nammonia, distorted\nN 0.02 -0.01 0.11\nH 0.0 0.95 -0.26\nH 0.84 -0.46 -0.30\nH -0.80 -0.50 -0.24\n";

const GRIMME: (f64, f64) = (6.0 / 5.0, 1.0 / 3.0);
const SOS: (f64, f64) = (1.0, 0.0);

fn scs_config(scaling: (f64, f64), frozen_core: usize) -> ScsMp2Config {
    ScsMp2Config {
        c_os: scaling.0,
        c_ss: scaling.1,
        frozen_core,
        memory_budget_bytes: None,
    }
}

fn rijk() -> ScfJk {
    ScfJk {
        mode: ScfJkMode::RiJk,
        aux: Some("def2-universal-jkfit".to_string()),
        k_builder: None,
    }
}

/// One geometry's prepared bases, bounds and tight RHF on `jk`.
struct Point {
    prep: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
    rhf: ScfResult,
}

fn point(mol: &Molecule, obs: &BasisSet, aux: &BasisSet, jk: Option<&ScfJk>) -> Point {
    let op = Operator::coulomb();
    let prep = PreparedBasis::new(mol, obs).unwrap();
    let dfbs = PreparedBasis::new(mol, aux).unwrap();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut cfg = RhfConfig {
        density_conv: 1e-9,
        ..Default::default()
    }
    .for_geometry_task();
    if let Some(jk) = jk {
        jk.apply(&mut cfg);
    }
    let ctx = ferric_core::parallel::ParallelContext::default();
    let rhf = solve_rhf(&ctx, mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged, "reference SCF did not converge");
    Point {
        prep,
        dfbs,
        bounds,
        rhf,
    }
}

fn displaced(mol: &Molecule, atom: usize, coord: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match coord {
        0 => m.atoms[atom].x += h,
        1 => m.atoms[atom].y += h,
        _ => m.atoms[atom].zpos += h,
    }
    m
}

/// Central FD of `scs_mp2`'s total energy for every config in `cfgs`
/// (one SCF per displaced geometry, shared by the configs).
fn fd_gradients(
    mol: &Molecule,
    obs: &BasisSet,
    aux: &BasisSet,
    jk: Option<&ScfJk>,
    cfgs: &[ScsMp2Config],
) -> Vec<Array2<f64>> {
    let n = mol.atoms.len();
    let mut out = vec![Array2::<f64>::zeros((n, 3)); cfgs.len()];
    for atom in 0..n {
        for coord in 0..3 {
            let energies = |h: f64| {
                let m = displaced(mol, atom, coord, h);
                let p = point(&m, obs, aux, jk);
                cfgs.iter()
                    .map(|c| {
                        scs_mp2(&m, &p.prep, &p.dfbs, &p.rhf, c)
                            .unwrap()
                            .total_energy
                    })
                    .collect::<Vec<_>>()
            };
            let (ep, em) = (energies(H), energies(-H));
            for (k, g) in out.iter_mut().enumerate() {
                g[(atom, coord)] = (ep[k] - em[k]) / (2.0 * H);
            }
        }
    }
    out
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// The former SCS gradient: RI-MP2's correlation part scaled by
/// `(c_os + c_ss)/2` on top of the four-centre RHF gradient.
fn old_average_scaling(mol: &Molecule, p: &Point, cfg: &ScsMp2Config) -> Array2<f64> {
    let op = Operator::coulomb();
    let mp2 = RiMp2Config {
        frozen_core: cfg.frozen_core,
        ..Default::default()
    };
    let g_mp2 = rimp2_gradient_analytical(mol, &p.prep, &p.dfbs, op, &p.bounds, &p.rhf, &mp2, None)
        .unwrap();
    let g_hf =
        ferric_scf::gradient::rhf_gradient_exact_jk(mol, &p.prep, op, &p.bounds, &p.rhf, None)
            .unwrap();
    &g_hf + &(0.5 * (cfg.c_os + cfg.c_ss) * (&g_mp2 - &g_hf))
}

/// Analytic vs FD for each scaling on a four-centre reference, plus the
/// negative control for every scaling with `c_os ≠ c_ss`.
fn check(label: &str, xyz: &str, basis_name: &str, scalings: &[(f64, f64)], frozen_core: usize) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = basis::bundled(basis_name).unwrap();
    let aux = basis::bundled(AUX).unwrap();
    let cfgs: Vec<_> = scalings
        .iter()
        .map(|&s| scs_config(s, frozen_core))
        .collect();
    let fds = fd_gradients(&mol, &obs, &aux, None, &cfgs);
    let p = point(&mol, &obs, &aux, None);
    let op = Operator::coulomb();
    for (cfg, fd) in cfgs.iter().zip(&fds) {
        let g =
            scs_mp2_gradient_analytical(&mol, &p.prep, &p.dfbs, op, &p.bounds, &p.rhf, cfg, None)
                .unwrap();
        let worst = max_abs_diff(&g, fd);
        let old = max_abs_diff(&old_average_scaling(&mol, &p, cfg), fd);
        eprintln!(
            "{label} (c_os {:.4}, c_ss {:.4}, fc {frozen_core}): max |analytic - FD| = \
             {worst:.3e} (bar {BAR:.0e}); old (c_os+c_ss)/2 scaling: {old:.3e} ({:.0}x bar); \
             max |g| = {:.3e}",
            cfg.c_os,
            cfg.c_ss,
            old / BAR,
            g.iter().fold(0.0f64, |m, v| m.max(v.abs()))
        );
        assert!(
            worst < BAR,
            "{label} ({}, {}): max |analytic - FD| = {worst:.3e}",
            cfg.c_os,
            cfg.c_ss
        );
        if cfg.c_os != cfg.c_ss {
            assert!(
                old > NEGATIVE_FACTOR * BAR,
                "{label}: the (c_os+c_ss)/2 scaling passes within {NEGATIVE_FACTOR}x the bar \
                 ({old:.3e}); the FD comparison cannot tell it from the real gradient"
            );
        }
    }
}

#[test]
fn scs_gradient_matches_fd_water_sto3g() {
    check("H2O/STO-3G", WATER, "sto-3g", &[GRIMME, SOS], 0);
}

#[test]
fn scs_gradient_matches_fd_water_ccpvdz() {
    check("H2O/cc-pVDZ", WATER, "cc-pvdz", &[GRIMME, SOS], 0);
}

#[test]
fn scs_gradient_matches_fd_ammonia_631g() {
    check("NH3/6-31G", AMMONIA, "6-31g", &[GRIMME, SOS], 0);
}

#[test]
fn scs_gradient_matches_fd_water_ccpvdz_frozen_core() {
    check("H2O/cc-pVDZ", WATER, "cc-pvdz", &[GRIMME], 1);
}

/// The RI-JK reference: [`total_scs_mp2_gradient_on`] with `scf_jk =
/// "rijk"` against FD of the SCS-MP2 energy of the fitted SCF; its energy is
/// that FD energy at the reference geometry.
fn check_rijk(label: &str, xyz: &str, basis_name: &str, frozen_core: usize) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = basis::bundled(basis_name).unwrap();
    let aux = basis::bundled(AUX).unwrap();
    let cfg = scs_config(GRIMME, frozen_core);
    let jk = rijk();
    let (e, g) = total_scs_mp2_gradient_on(
        &mol,
        &obs,
        &aux,
        &cfg,
        None,
        &jk,
        &mut ScfWarmStart::new(false),
    )
    .unwrap();
    let p = point(&mol, &obs, &aux, Some(&jk));
    let e_direct = scs_mp2(&mol, &p.prep, &p.dfbs, &p.rhf, &cfg)
        .unwrap()
        .total_energy;
    assert!(
        (e - e_direct).abs() < 1e-9,
        "gradient driver energy {e} vs scs_mp2 {e_direct}"
    );
    let fd = fd_gradients(&mol, &obs, &aux, Some(&jk), std::slice::from_ref(&cfg));
    let worst = max_abs_diff(&g, &fd[0]);
    eprintln!("{label} RI-JK SCS-MP2 (fc {frozen_core}): max |analytic - FD| = {worst:.3e} (bar {BAR:.0e})");
    assert!(
        worst < BAR,
        "{label} RI-JK: max |analytic - FD| = {worst:.3e}"
    );
}

#[test]
fn scs_gradient_matches_fd_water_ccpvdz_rijk() {
    check_rijk("H2O/cc-pVDZ", WATER, "cc-pvdz", 0);
}

#[test]
fn scs_gradient_matches_fd_ammonia_631g_rijk() {
    check_rijk("NH3/6-31G", AMMONIA, "6-31g", 0);
}

#[test]
fn scs_gradient_matches_fd_water_ccpvdz_rijk_frozen_core() {
    check_rijk("H2O/cc-pVDZ", WATER, "cc-pvdz", 1);
}

/// The exactness anchor: `(c_os, c_ss) = (1, 1)` is the RI-MP2 gradient bit
/// for bit, frozen core included.
#[test]
fn scs_unit_scaling_is_the_rimp2_gradient_bit_for_bit() {
    let aux = basis::bundled(AUX).unwrap();
    let op = Operator::coulomb();
    for (label, xyz, basis_name, fc) in [
        ("H2O/cc-pVDZ", WATER, "cc-pvdz", 0),
        ("H2O/cc-pVDZ fc=1", WATER, "cc-pvdz", 1),
        ("NH3/6-31G", AMMONIA, "6-31g", 0),
    ] {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs = basis::bundled(basis_name).unwrap();
        let p = point(&mol, &obs, &aux, None);
        let scs = scs_mp2_gradient_analytical(
            &mol,
            &p.prep,
            &p.dfbs,
            op,
            &p.bounds,
            &p.rhf,
            &scs_config((1.0, 1.0), fc),
            None,
        )
        .unwrap();
        let mp2 = RiMp2Config {
            frozen_core: fc,
            ..Default::default()
        };
        let rimp2 =
            rimp2_gradient_analytical(&mol, &p.prep, &p.dfbs, op, &p.bounds, &p.rhf, &mp2, None)
                .unwrap();
        let mismatched = scs
            .iter()
            .zip(rimp2.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "{label}: (1, 1) vs rimp2: {mismatched} of {} components differ in bits",
            scs.len()
        );
        assert_eq!(
            mismatched, 0,
            "{label}: SCS(1,1) != RI-MP2 gradient bitwise"
        );
    }
}
