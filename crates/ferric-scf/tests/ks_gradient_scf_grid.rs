//! The KS analytic gradient differentiates the grid the SCF ran on (#428).
//!
//! `restricted_scf_gradient` / `unrestricted_scf_gradient` /
//! `restricted_open_scf_gradient` used to build the XC grid with
//! `AtomicGridConfig::default()` whatever `config.dft_grid` said, so a SCF on a
//! non-default (or pruned) grid was paired with the gradient of a different
//! surface. Each case here compares the production gradient entry point with a
//! central finite difference of the converged energy on the SAME grid.
//!
//! Bar derivation. `default_grid_floor_*` measures the FD floor of this harness
//! (step 1e-3 Bohr, SCF converged to 1e-11 Ha) on the default grid, where the
//! old code was already right: agreement is 0.9e-7..1.4e-7 Ha/Bohr, so
//! `BAR = 3e-7` is ~2x the floor. The non-default rows must meet the same bar. The second
//! assertion of `check` takes the plain `ks_gradient_*` entry points (which
//! differentiate the default grid by construction, i.e. the pre-fix pairing) and
//! requires them to miss by >= 10x the bar, so the anchor can discriminate.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::prune::PruneScheme;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{
    restricted_open_scf_gradient, restricted_scf_gradient, unrestricted_scf_gradient,
};
use ferric_scf::ks_gradient::{ks_gradient_closed, ks_gradient_roks, ks_gradient_uks};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

const BAR: f64 = 3e-7;
/// Measured FD floor of this harness on the default grid (largest of the rows).
const FD_FLOOR: f64 = 1.5e-7;
const STEP: f64 = 1e-3;
const BASIS: &str = "6-31g";

const WATER: &str = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
const NH2: &str = "3\n\nN 0.0 0.0 0.1289\nH 0.0 0.8033 -0.5157\nH 0.0 -0.8033 -0.5157\n";

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Rks,
    Uks,
    Roks,
}

fn config(xc: &str, grid: Option<AtomicGridConfig>) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        dft_grid: grid,
        energy_conv: 1e-11,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    }
}

fn energy(kind: Kind, mol: &Molecule, cfg: &RhfConfig) -> f64 {
    let bs = basis::bundled(BASIS).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    match kind {
        Kind::Rks => {
            solve_rhf(&ctx, mol, &prep, op, &bounds, cfg)
                .unwrap()
                .energy
        }
        Kind::Uks => solve_uhf(&ctx, mol, &prep, &bounds, cfg).unwrap().energy,
        Kind::Roks => {
            solve_rohf(&ctx, mol, &prep, op, &bounds, cfg)
                .unwrap()
                .energy
        }
    }
}

/// (production gradient, plain default-grid-gradient) at `mol`.
fn gradients(kind: Kind, mol: &Molecule, xc: &str, cfg: &RhfConfig) -> (Array2<f64>, Array2<f64>) {
    let bs = basis::bundled(BASIS).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    match kind {
        Kind::Rks => {
            let r = solve_rhf(&ctx, mol, &prep, op, &bounds, cfg).unwrap();
            (
                restricted_scf_gradient(mol, &prep, &bs, op, &bounds, cfg, &r).unwrap(),
                ks_gradient_closed(mol, &prep, &bs, op, &bounds, xc, &r, None).unwrap(),
            )
        }
        Kind::Uks => {
            let r = solve_uhf(&ctx, mol, &prep, &bounds, cfg).unwrap();
            (
                unrestricted_scf_gradient(mol, &prep, &bs, op, &bounds, cfg, &r).unwrap(),
                ks_gradient_uks(mol, &prep, &bs, op, &bounds, xc, &r, None).unwrap(),
            )
        }
        Kind::Roks => {
            let r = solve_rohf(&ctx, mol, &prep, op, &bounds, cfg).unwrap();
            (
                restricted_open_scf_gradient(mol, &prep, &bs, op, &bounds, cfg, &r).unwrap(),
                ks_gradient_roks(mol, &prep, &bs, op, &bounds, xc, &r, None).unwrap(),
            )
        }
    }
}

/// (max|production - FD|, max|plain default-grid gradient - FD|) over all coordinates.
fn residuals(
    kind: Kind,
    xyz: &str,
    mult: usize,
    xc: &str,
    grid: Option<AtomicGridConfig>,
) -> (f64, f64) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let cfg = config(xc, grid);
    let (g, g_plain) = gradients(kind, &mol, xc, &cfg);
    let mut worst = 0.0f64;
    let mut worst_plain = 0.0f64;
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let mut e = [0.0f64; 2];
            for (i, sgn) in [1.0, -1.0].into_iter().enumerate() {
                let mut m = mol.clone();
                match c {
                    0 => m.atoms[a].x += sgn * STEP,
                    1 => m.atoms[a].y += sgn * STEP,
                    _ => m.atoms[a].zpos += sgn * STEP,
                }
                e[i] = energy(kind, &m, &cfg);
            }
            let fd = (e[0] - e[1]) / (2.0 * STEP);
            worst = worst.max((g[(a, c)] - fd).abs());
            worst_plain = worst_plain.max((g_plain[(a, c)] - fd).abs());
        }
    }
    (worst, worst_plain)
}

fn flat() -> Option<AtomicGridConfig> {
    Some(AtomicGridConfig {
        n_radial: 40,
        n_angular: 194,
        prune: None,
    })
}

/// Aggressive pruning (ORCA GridX / PySCF `sgx_prune` regions) on a coarse
/// radial grid, chosen so that the pruned and flat energies differ enough for
/// a gradient that ignored `prune` to be visible above the FD floor (see
/// `prune_sensitivity`, required >= 10x the FD floor).
fn pruned() -> Option<AtomicGridConfig> {
    Some(AtomicGridConfig {
        n_radial: 20,
        n_angular: 194,
        prune: Some(PruneScheme::Sgx),
    })
}

fn pruned_nwchem() -> Option<AtomicGridConfig> {
    Some(AtomicGridConfig {
        n_radial: 30,
        n_angular: 194,
        prune: Some(PruneScheme::NwchemLike),
    })
}

/// max |g(pruned) - g(same grid, prune = None)|: what a gradient that dropped
/// `prune` would be off by. Must exceed 10x the bar or the pruned rows cannot
/// tell "honours prune" from "ignores prune".
fn prune_sensitivity(kind: Kind, xyz: &str, mult: usize, xc: &str, grid: AtomicGridConfig) -> f64 {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let mut flat_twin = grid.clone();
    flat_twin.prune = None;
    let (gp, _) = gradients(kind, &mol, xc, &config(xc, Some(grid)));
    let (gf, _) = gradients(kind, &mol, xc, &config(xc, Some(flat_twin)));
    gp.iter()
        .zip(gf.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn check(
    label: &str,
    kind: Kind,
    xyz: &str,
    mult: usize,
    xc: &str,
    grid: Option<AtomicGridConfig>,
) {
    let pruned_cfg = grid.clone().filter(|g| g.prune.is_some());
    let (res, plain) = residuals(kind, xyz, mult, xc, grid);
    if let Some(g) = pruned_cfg {
        // NwchemLike changes the energy by ~1e-10 Ha by design, so a gradient
        // that ignored it is invisible at FD precision; that row is a
        // consistency check only. Sgx is aggressive enough to discriminate.
        let discriminates = g.prune == Some(PruneScheme::Sgx);
        let sens = prune_sensitivity(kind, xyz, mult, xc, g);
        eprintln!("{label}: prune sensitivity {sens:.3e}");
        if discriminates {
            assert!(
                sens >= 10.0 * FD_FLOOR,
                "{label}: prune sensitivity {sens:.3e} too small"
            );
        }
    }
    eprintln!("{label}: production-vs-FD {res:.3e}  default-grid-gradient-vs-FD {plain:.3e}");
    assert!(res < BAR, "{label}: analytic vs FD {res:.3e} >= {BAR:.0e}");
    assert!(
        plain >= 10.0 * BAR,
        "{label}: the pre-fix pairing (default-grid gradient, SCF on another grid) only \
         misses by {plain:.3e}; the anchor cannot discriminate"
    );
}

#[test]
fn default_grid_floor_rks_lda() {
    let (res, plain) = residuals(Kind::Rks, WATER, 1, "LDA", None);
    eprintln!("default grid RKS/LDA: {res:.3e} (plain {plain:.3e})");
    assert!(res < BAR && plain < BAR);
}

#[test]
fn default_grid_floor_uks_pbe() {
    let (res, plain) = residuals(Kind::Uks, NH2, 2, "PBE", None);
    eprintln!("default grid UKS/PBE: {res:.3e} (plain {plain:.3e})");
    assert!(res < BAR && plain < BAR);
}

#[test]
fn rks_lda_non_default_grid() {
    check("RKS LDA flat 40x194", Kind::Rks, WATER, 1, "LDA", flat());
}

#[test]
fn rks_pbe_non_default_grid() {
    check("RKS PBE flat 40x194", Kind::Rks, WATER, 1, "PBE", flat());
}

#[test]
fn rks_pbe_pruned_grid() {
    check(
        "RKS PBE pruned sgx 20x194",
        Kind::Rks,
        WATER,
        1,
        "PBE",
        pruned(),
    );
}

#[test]
fn uks_pbe_non_default_grid() {
    check("UKS PBE flat 40x194", Kind::Uks, NH2, 2, "PBE", flat());
}

#[test]
fn uks_pbe_pruned_grid() {
    check(
        "UKS PBE pruned sgx 20x194",
        Kind::Uks,
        NH2,
        2,
        "PBE",
        pruned(),
    );
}

#[test]
fn roks_pbe_non_default_grid() {
    check("ROKS PBE flat 40x194", Kind::Roks, NH2, 2, "PBE", flat());
}

#[test]
fn roks_pbe_pruned_grid() {
    check(
        "ROKS PBE pruned sgx 20x194",
        Kind::Roks,
        NH2,
        2,
        "PBE",
        pruned(),
    );
}

#[test]
fn rks_pbe_pruned_nwchem_grid() {
    check(
        "RKS PBE pruned nwchem 30x194",
        Kind::Rks,
        WATER,
        1,
        "PBE",
        pruned_nwchem(),
    );
}
