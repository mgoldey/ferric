//! S8 spike (item #8): diagnose the soft-atom (Si/S/O) blowup of the MOLECULAR
//! Becke effective-volume integral `v_A = ∫ w^A_Becke(r) ρ(r) |r−R_A|³ dV`.
//!
//! This is the NUMERATOR of the TS volume ratio (the molecular volume computed on
//! the actual molecule) — deliberately separate from G8, which fixes the free-atom
//! REFERENCE (denominator). We want the numerator's own basis / grid sensitivity.
//!
//! Two hypotheses tested with real numbers:
//!   (H1) basis: 6-31g / def2-svp (DZ, no diffuse) vs aug-cc-pVDZ (same zeta,
//!        +diffuse) vs def2-tzvp (TZ, no diffuse) vs aug-cc-pVTZ (converged ref).
//!        Separates "diffuseness" from "zeta level". NB the bundled cc-pVDZ has
//!        no second-row (Si/S) shells, so 6-31g is the bare-DZ baseline for those.
//!   (H2) grid: re-integrate the SAME density on 50×50, 75×110 (production default),
//!        99×302 grids. Separates grid quality from basis quality.
//!
//! VERDICT (2026-07-17): numerator does NOT blow up. Worst DZ deviation is O at
//! def2-svp, -10.1% (mild UNDER-prediction); Si is ±0.7% across the whole ladder
//! incl. bare 6-31g. Diffuse functions close the DZ gap (def2-svp→aug-cc-pvdz:
//! O -10.1%→+0.6%). Grid is converged (75×110 vs 99×302 ≤0.09%). The SiH4 C6
//! blowup is entirely the free-atom denominator (item #36/G8), not this integral.
//!
//! Run:  flock ... cargo test -p ferric-rpa --release --test s8_becke_volume_basis_grid -- --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::ao_grid::eval_basis_on_points;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Molecular Becke volume with a CONFIGURABLE grid (the production function
/// hardcodes AtomicGridConfig::default()). Faithful copy of
/// properties::atomic_effective_volumes_becke otherwise.
fn becke_volume_with_grid(
    mol: &Molecule,
    obs_bs: &basis::BasisSet,
    density: &Array2<f64>,
    cfg: &AtomicGridConfig,
) -> Vec<f64> {
    let natoms = mol.atoms.len();
    let grid = build_atomic_grid(mol, cfg);
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let home_atom: Vec<usize> = grid.iter().map(|g| g.home_atom).collect();
    let npts = points.len();
    let chi = eval_basis_on_points(mol, obs_bs, &points).unwrap();
    let nbf = chi.nrows();
    let pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let mut vol = vec![0.0_f64; natoms];
    for g in 0..npts {
        let a = home_atom[g];
        let mut rho = 0.0;
        for mu in 0..nbf {
            let cm = chi[(mu, g)];
            if cm.abs() < 1e-30 {
                continue;
            }
            for nu in 0..nbf {
                rho += density[(mu, nu)] * cm * chi[(nu, g)];
            }
        }
        let dx = points[g][0] - pos[a][0];
        let dy = points[g][1] - pos[a][1];
        let dz = points[g][2] - pos[a][2];
        let r3 = (dx * dx + dy * dy + dz * dz).powf(1.5);
        vol[a] += weights[g] * rho * r3;
    }
    vol
}

/// Also report total integrated electron count per atom (∫ w^A ρ dV) and the
/// mean ⟨r³⟩ = v_A / N_A, to see whether a blowup is in the population or in
/// the r³ tail moment.
fn becke_vol_and_pop(
    mol: &Molecule,
    obs_bs: &basis::BasisSet,
    density: &Array2<f64>,
    cfg: &AtomicGridConfig,
) -> (Vec<f64>, Vec<f64>) {
    let natoms = mol.atoms.len();
    let grid = build_atomic_grid(mol, cfg);
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let home_atom: Vec<usize> = grid.iter().map(|g| g.home_atom).collect();
    let npts = points.len();
    let chi = eval_basis_on_points(mol, obs_bs, &points).unwrap();
    let nbf = chi.nrows();
    let pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let mut vol = vec![0.0_f64; natoms];
    let mut pop = vec![0.0_f64; natoms];
    for g in 0..npts {
        let a = home_atom[g];
        let mut rho = 0.0;
        for mu in 0..nbf {
            let cm = chi[(mu, g)];
            if cm.abs() < 1e-30 {
                continue;
            }
            for nu in 0..nbf {
                rho += density[(mu, nu)] * cm * chi[(nu, g)];
            }
        }
        let dx = points[g][0] - pos[a][0];
        let dy = points[g][1] - pos[a][1];
        let dz = points[g][2] - pos[a][2];
        let r2 = dx * dx + dy * dy + dz * dz;
        let r3 = r2.powf(1.5);
        vol[a] += weights[g] * rho * r3;
        pop[a] += weights[g] * rho;
    }
    (vol, pop)
}

fn scf_density(mol: &Molecule, obs_bs: &basis::BasisSet) -> Array2<f64> {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(mol, obs_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    rhf.density_r().clone()
}

fn sih4() -> Molecule {
    // Si-H = 1.480 Å tetrahedral, from testdata conventions. xyz in Angstrom.
    let xyz = "5\nSiH4\n\
        Si  0.000000  0.000000  0.000000\n\
        H   0.854400  0.854400  0.854400\n\
        H  -0.854400 -0.854400  0.854400\n\
        H  -0.854400  0.854400 -0.854400\n\
        H   0.854400 -0.854400 -0.854400\n";
    Molecule::parse_xyz(xyz, 0, 1).unwrap()
}

fn h2s() -> Molecule {
    let xyz = "3\nH2S\n\
        S   0.000000  0.000000  0.102135\n\
        H   0.000000  0.966700 -0.816200\n\
        H   0.000000 -0.966700 -0.816200\n";
    Molecule::parse_xyz(xyz, 0, 1).unwrap()
}

fn h2o() -> Molecule {
    let xyz = "3\nH2O\n\
        O   0.000000  0.000000  0.117790\n\
        H   0.000000  0.755453 -0.471161\n\
        H   0.000000 -0.755453 -0.471161\n";
    Molecule::parse_xyz(xyz, 0, 1).unwrap()
}

/// Basis ladder. NOTE the bundled `cc-pvdz.json` has NO second-row (Si/S)
/// shells, so the honest DZ baseline for Si/S is `6-31g` (bare split-valence
/// double-zeta, no polarization, no diffuse) and `def2-svp` (DZ + polarization,
/// no diffuse). The diffuseness test is `aug-cc-pvdz` (DZ + diffuse) against
/// `def2-svp`/`6-31g` (DZ, no diffuse) at the SAME zeta level; the zeta test is
/// `def2-tzvp` (TZ, no diffuse) against `def2-svp` (DZ, no diffuse). Reference
/// is `aug-cc-pvtz` (TZ + diffuse, treated as converged).
///
/// Columns, in ascending "quality" so the trend is legible:
///   6-31g  def2-svp  aug-cc-pvdz  def2-tzvp  aug-cc-pvtz(ref)
const LADDER: &[&str] = &["6-31g", "def2-svp", "aug-cc-pvdz", "def2-tzvp", "aug-cc-pvtz"];

/// H1 — basis sensitivity of the molecular Becke heavy-atom volume, production grid.
#[test]
fn s8_h1_basis_sensitivity() {
    let cases: &[(&str, fn() -> Molecule, usize)] = &[
        ("SiH4 (Si)", sih4, 0),
        ("H2S  (S) ", h2s, 0),
        ("H2O  (O) ", h2o, 0),
    ];
    let cfg = AtomicGridConfig::default(); // production 75×110

    println!("\n=== S8 H1: molecular Becke heavy-atom volume v_A [Bohr^5] vs basis (grid 75x110) ===");
    println!("ladder (ascending): 6-31g(DZ,no-diff) | def2-svp(DZ,no-diff) | aug-cc-pvdz(DZ,+diff) | def2-tzvp(TZ,no-diff) | aug-cc-pvtz(TZ,+diff=REF)");
    println!(
        "{:<12} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "molecule", "6-31g", "def2-svp", "aug-ccpvdz", "def2-tzvp", "aug-ccpvtz"
    );
    for (label, build, heavy_idx) in cases {
        let mol = build();
        let mut row = Vec::new();
        for b in LADDER {
            let bs = basis::bundled(b).unwrap();
            let d = scf_density(&mol, &bs);
            let v = becke_volume_with_grid(&mol, &bs, &d, &cfg);
            row.push(v[*heavy_idx]);
        }
        println!(
            "{:<12} {:>11.4} {:>11.4} {:>11.4} {:>11.4} {:>11.4}",
            label, row[0], row[1], row[2], row[3], row[4]
        );
        // % deviation of each basis from the aug-cc-pVTZ reference (last column).
        let refv = row[4];
        println!(
            "   %dev vs aTZ:  6-31g {:+7.1}%  def2-svp {:+7.1}%  aug-ccpvdz {:+7.1}%  def2-tzvp {:+7.1}%",
            100.0 * (row[0] - refv) / refv,
            100.0 * (row[1] - refv) / refv,
            100.0 * (row[2] - refv) / refv,
            100.0 * (row[3] - refv) / refv,
        );
    }
}

/// H2 — grid sensitivity: re-integrate the SAME (6-31g DZ and aug-cc-pVTZ)
/// density on progressively finer grids. If the blowup is basis (density-tail)
/// not grid, the volume is grid-converged already at 75×110 and both bases move
/// by the same small % under grid refinement.
#[test]
fn s8_h2_grid_sensitivity() {
    let grids = [
        ("50x50", AtomicGridConfig { n_radial: 50, n_angular: 50 }),
        ("75x110", AtomicGridConfig { n_radial: 75, n_angular: 110 }),
        ("99x302", AtomicGridConfig { n_radial: 99, n_angular: 302 }),
    ];
    let cases: &[(&str, fn() -> Molecule, usize)] = &[
        ("SiH4 (Si)", sih4, 0),
        ("H2S  (S) ", h2s, 0),
    ];

    println!("\n=== S8 H2: molecular Becke heavy-atom volume v_A [Bohr^5] vs GRID (fixed basis) ===");
    for basis_name in ["6-31g", "aug-cc-pvtz"] {
        println!("--- basis = {basis_name} ---");
        for (label, build, heavy_idx) in cases {
            let mol = build();
            let bs = basis::bundled(basis_name).unwrap();
            let d = scf_density(&mol, &bs);
            let mut row = Vec::new();
            let mut poprow = Vec::new();
            for (_gname, gcfg) in &grids {
                let (v, pop) = becke_vol_and_pop(&mol, &bs, &d, gcfg);
                row.push(v[*heavy_idx]);
                poprow.push(pop[*heavy_idx]);
            }
            println!(
                "  {:<10} v_A: 50x50 {:>10.4}  75x110 {:>10.4}  99x302 {:>10.4}   (Δgrid vs fine {:+.2}%)",
                label, row[0], row[1], row[2],
                100.0 * (row[1] - row[2]) / row[2]
            );
            println!(
                "  {:<10} N_A: 50x50 {:>10.4}  75x110 {:>10.4}  99x302 {:>10.4}",
                "", poprow[0], poprow[1], poprow[2]
            );
        }
    }
}
