//! Analytic `∂E_pcm/∂R` at a FIXED AO density against central finite
//! differences of `pcm_step`'s energy (`crates/ferric-pcm/theory.md` §8).
//!
//! The AO density `P` is held fixed in the AO basis while the atoms (and
//! with them the basis functions and the cavity) move, so `E_pcm(R; P)` is a
//! well-defined function of the nuclear coordinates and its derivative is
//! exactly what `pcm_gradient` returns. No SCF enters: the SCF-level
//! checks (and the PySCF comparison) are in
//! `crates/ferric-scf/tests/pcm_gradient.rs`.
//!
//! Physics vs artifact hypothesis: if the gradient is right, analytic − FD
//! sits inside the step-derived truncation bound for every coordinate, on
//! both S/D kinds and both probe kinds. A dropped term (switching
//! derivative, D derivative, tessera-position derivative of the
//! potential, ...) shifts components by 1e-4..1e-2 Ha/Bohr, far outside it.
//!
//! FD tolerance: central differences at steps `h` and `2h` differ by
//! `≈ 3·(h²/6)·E'''`, i.e. three times the truncation error at `h`; the bar
//! is that difference plus a 1e-9 Ha/Bohr round-off floor (`E_pcm` ~1e-2 Ha
//! through two ~10³-dimensional LU solves, ~1e-13 Ha, divided by `2h`).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pcm::matrices::SdKind;
use ferric_pcm::{pcm_gradient_parts, pcm_step, PcmConfig, PcmContext, ProbeKind};
use ndarray::Array2;

const H: f64 = 1e-3;
const ROUNDOFF_FLOOR: f64 = 1e-9;

/// Water (testdata geometry) in Bohr.
fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap()
}

/// Methanol with one methyl H moved off the mirror plane: no symmetry, so
/// no gradient component is zero by construction.
fn methanol_c1() -> Molecule {
    let xyz = "6\nmethanol C1\n\
        C -0.046800  0.663000  0.000000\n\
        O -0.046800 -0.758000  0.000000\n\
        H -1.092600  0.969000  0.120000\n\
        H  0.438000  1.080000  0.890000\n\
        H  0.438000  1.080000 -0.890000\n\
        H  0.861000 -1.075000  0.050000\n";
    Molecule::parse_xyz(xyz, 0, 1).unwrap()
}

/// A fixed, non-idempotent but positive AO density: diagonal plus a
/// nearest-neighbour coupling. Physicality is irrelevant to the identity.
fn model_density(n: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        d[(i, i)] = 0.6 + 0.05 * (i % 3) as f64;
        if i + 1 < n {
            d[(i, i + 1)] = 0.12;
            d[(i + 1, i)] = 0.12;
        }
    }
    d
}

fn displaced(mol: &Molecule, a: usize, ax: usize, dx: f64) -> Molecule {
    let mut m = mol.clone();
    match ax {
        0 => m.atoms[a].x += dx,
        1 => m.atoms[a].y += dx,
        _ => m.atoms[a].zpos += dx,
    }
    m
}

fn e_pcm(mol: &Molecule, d: &Array2<f64>, cfg: &PcmConfig) -> f64 {
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let ctx = PcmContext::new(mol, cfg).unwrap();
    pcm_step(&ctx, mol, &prep, d).unwrap().1
}

fn central(mol: &Molecule, d: &Array2<f64>, cfg: &PcmConfig, a: usize, ax: usize, h: f64) -> f64 {
    (e_pcm(&displaced(mol, a, ax, h), d, cfg) - e_pcm(&displaced(mol, a, ax, -h), d, cfg))
        / (2.0 * h)
}

/// Analytic vs FD on every coordinate; returns the max |analytic − FD|.
fn check_fd(label: &str, mol: &Molecule, cfg: &PcmConfig) -> f64 {
    check_fd_h(label, mol, cfg, H)
}

fn check_fd_h(label: &str, mol: &Molecule, cfg: &PcmConfig, h: f64) -> f64 {
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let d = model_density(prep.nbasis());
    let g = pcm_gradient_parts(mol, &prep, &d, cfg).unwrap().total();
    let mut worst = 0.0_f64;
    for a in 0..mol.atoms.len() {
        for ax in 0..3 {
            let fd_h = central(mol, &d, cfg, a, ax, h);
            let fd_2h = central(mol, &d, cfg, a, ax, 2.0 * h);
            let tol = (fd_2h - fd_h).abs() + ROUNDOFF_FLOOR;
            let err = (g[(a, ax)] - fd_h).abs();
            // The bar is only meaningful while the FD is in its O(h²) regime.
            assert!(
                (fd_2h - fd_h).abs() < 0.05 * fd_h.abs() + 1e-6,
                "{label}: atom {a} axis {ax}: FD not converged at h = {h} ({fd_h:e} vs {fd_2h:e})"
            );
            assert!(
                err < tol,
                "{label}: atom {a} axis {ax}: analytic {:.12e} FD {:.12e} |diff| {err:.3e} > tol {tol:.3e}",
                g[(a, ax)],
                fd_h
            );
            worst = worst.max(err);
        }
    }
    eprintln!("{label}: max |analytic - FD| = {worst:.3e} Ha/Bohr");
    worst
}

fn cfg(sd: SdKind, probe: ProbeKind, order: usize) -> PcmConfig {
    PcmConfig {
        sd_kind: sd,
        probe,
        ..PcmConfig::water().with_lebedev_order(order).unwrap()
    }
}

#[test]
fn fixed_density_gradient_matches_fd_water_default() {
    check_fd(
        "water default",
        &water(),
        &cfg(SdKind::GaussianSmeared, ProbeKind::Point, 110),
    );
}

#[test]
fn fixed_density_gradient_matches_fd_methanol_both_probes() {
    let mol = methanol_c1();
    for probe in [ProbeKind::Point, ProbeKind::GaussianSmeared] {
        check_fd(
            &format!("methanol {probe:?}"),
            &mol,
            &cfg(SdKind::GaussianSmeared, probe, 50),
        );
    }
}

/// `SdKind::PointCharge` (`S_ii ∝ a_i^{-1/2}`, `D_ii = −S_ii/2R_i`): its
/// energy is far rougher in the switching zone than the Gaussian kind's
/// (on C1 methanol at Lebedev 50, components reach ~10 Ha/Bohr and the FD
/// is not yet in its O(h²) regime at h = 2.5e-4), so it is checked on water
/// at Lebedev 110 with h = 2.5e-4.
#[test]
fn fixed_density_gradient_matches_fd_water_point_charge_sd() {
    check_fd_h(
        "water PointCharge S/D",
        &water(),
        &cfg(SdKind::PointCharge, ProbeKind::Point, 110),
        2.5e-4,
    );
}

/// Rigid translation leaves `E_pcm(R; P)` unchanged, so each part's rows sum
/// to zero: the response term through tessera pairs and the switching
/// function, the potential term through nuclei, AO centres and tesserae.
#[test]
fn each_part_is_translationally_invariant() {
    let mol = methanol_c1();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let d = model_density(prep.nbasis());
    for probe in [ProbeKind::Point, ProbeKind::GaussianSmeared] {
        let p =
            pcm_gradient_parts(&mol, &prep, &d, &cfg(SdKind::GaussianSmeared, probe, 50)).unwrap();
        for (name, part) in [("potential", &p.potential), ("response", &p.response)] {
            let scale = part.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            assert!(scale > 1e-5, "{probe:?} {name}: part is trivially zero");
            for ax in 0..3 {
                let sum: f64 = part.column(ax).sum();
                assert!(
                    sum.abs() < 1e-9 * scale.max(1.0),
                    "{probe:?} {name} axis {ax}: Σ_A g = {sum:e} (scale {scale:e})"
                );
            }
        }
    }
}

/// ε → 1: `f(ε) = (ε−1)/(ε+1) → 0`, the charges and every term vanish
/// linearly in `f`. At ε = 1 + 2e-9 (`f` = 1e-9) the gradient is ≤ 1e-8
/// of its ε = 78.4 value relative to `f(78.4)`.
#[test]
fn gradient_vanishes_linearly_as_eps_goes_to_one() {
    let mol = water();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let d = model_density(prep.nbasis());
    let g_of = |eps: f64| {
        let c = PcmConfig::with_epsilon(eps).unwrap();
        pcm_gradient_parts(&mol, &prep, &d, &c).unwrap().total()
    };
    let f = |eps: f64| (eps - 1.0) / (eps + 1.0);
    let max_abs = |g: &Array2<f64>| g.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let g_water = g_of(78.4);
    let eps_small = 1.0 + 2e-9;
    let g_small = g_of(eps_small);
    let ratio = (max_abs(&g_small) / f(eps_small)) / (max_abs(&g_water) / f(78.4));
    assert!(max_abs(&g_water) > 1e-4);
    assert!(
        max_abs(&g_small) < 1e-8,
        "ε→1 gradient {:e}",
        max_abs(&g_small)
    );
    // Linear in f, with an O(1) slope ratio (the ε = 78.4 response is
    // screened: Kirkwood's k_l saturates).
    assert!(ratio > 0.5 && ratio < 10.0, "slope ratio {ratio}");
}

/// An injected cavity has no recorded geometry dependence: refused.
#[test]
fn injected_cavity_is_refused() {
    let mol = water();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let d = model_density(prep.nbasis());
    let ctx = PcmContext::new(&mol, &PcmConfig::water()).unwrap();
    let c = PcmConfig {
        cavity: Some(ctx.tesserae().to_vec()),
        ..PcmConfig::water()
    };
    assert!(pcm_gradient_parts(&mol, &prep, &d, &c).is_err());
}
