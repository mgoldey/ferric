//! MEASUREMENT (not a gate): how many radial and angular points the Hirshfeld
//! VOLUME integral `∫ w_A ρ r³` needs, against the dense (200,590) reference
//! the validation row uses.
//!
//! The question this answers: the production 75x110 Becke-Lebedev grid is
//! tuned for `∫ρ ε_xc`, where the density tail contributes almost nothing.
//! The `r³` weight moves the integrand's mass OUTWARD, so the volume needs a
//! radial rule accurate much further out — and a free H atom, whose density is
//! the most diffuse per unit of it, is the worst case.
//!
//!   OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-rpa \
//!       --test measure_volume_grid_convergence -- --ignored --nocapture

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_dft::ao_grid::eval_basis_on_points;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_scf::properties::{scf_proatom_radii, slater_xi_for_z, RadialProatom};
use ndarray::Array2;
use serde_json::Value;

fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("testdata").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf()
}

fn reference(name: &str) -> Value {
    let path = workspace_root()
        .join("testdata/reference/validation/hirshfeld")
        .join(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

fn mat(v: &Value, ptr: &str) -> Array2<f64> {
    let rows = v.pointer(ptr).unwrap().as_array().unwrap();
    let n = rows.len();
    let m = rows[0].as_array().unwrap().len();
    Array2::from_shape_fn((n, m), |(i, j)| rows[i][j].as_f64().unwrap())
}

/// `atomic_effective_volumes_hirshfeld` with the grid config exposed.
fn volumes_on(
    mol: &Molecule,
    bs: &BasisSet,
    d: &Array2<f64>,
    proatoms: &BTreeMap<i32, RadialProatom>,
    cfg: &AtomicGridConfig,
) -> (Vec<f64>, f64, usize) {
    let natoms = mol.atoms.len();
    let grid = build_atomic_grid(mol, cfg);
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(mol, bs, &points).unwrap();
    let d_chi = d.dot(&chi);
    let nbf = chi.nrows();
    let pos: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let mut vol = vec![0.0_f64; natoms];
    let mut n_e = 0.0_f64;
    for (g, gp) in grid.iter().enumerate() {
        let mut rho = 0.0_f64;
        for mu in 0..nbf {
            rho += chi[(mu, g)] * d_chi[(mu, g)];
        }
        n_e += gp.weight * rho;
        let mut r0 = vec![0.0_f64; natoms];
        let mut sum = 0.0_f64;
        for a in 0..natoms {
            let z = mol.atoms[a].z;
            let dx = gp.xyz[0] - pos[a][0];
            let dy = gp.xyz[1] - pos[a][1];
            let dz = gp.xyz[2] - pos[a][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let v = match proatoms.get(&z) {
                Some(p) => p.at(r),
                None => {
                    let xi = slater_xi_for_z(z);
                    z as f64 * xi.powi(3) / std::f64::consts::PI * (-2.0 * xi * r).exp()
                }
            };
            r0[a] = v;
            sum += v;
        }
        let inv = 1.0 / (sum + 1e-12);
        for a in 0..natoms {
            let dx = gp.xyz[0] - pos[a][0];
            let dy = gp.xyz[1] - pos[a][1];
            let dz = gp.xyz[2] - pos[a][2];
            vol[a] += r0[a] * inv * gp.weight * rho * (dx * dx + dy * dy + dz * dz).powf(1.5);
        }
    }
    (vol, n_e, grid.len())
}

/// `Σ_g w_g ρ r³` with NO Hirshfeld weight: what the reference's
/// `volume_dense_grid` computes, so the two can be compared like for like.
fn volumes_weight_one(
    mol: &Molecule,
    bs: &BasisSet,
    d: &Array2<f64>,
    cfg: &AtomicGridConfig,
) -> f64 {
    let grid = build_atomic_grid(mol, cfg);
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(mol, bs, &points).unwrap();
    let d_chi = d.dot(&chi);
    let nbf = chi.nrows();
    let pos = [mol.atoms[0].x, mol.atoms[0].y, mol.atoms[0].zpos];
    let mut acc = 0.0_f64;
    for (g, gp) in grid.iter().enumerate() {
        let mut rho = 0.0_f64;
        for mu in 0..nbf {
            rho += chi[(mu, g)] * d_chi[(mu, g)];
        }
        let dx = gp.xyz[0] - pos[0];
        let dy = gp.xyz[1] - pos[1];
        let dz = gp.xyz[2] - pos[2];
        acc += gp.weight * rho * (dx * dx + dy * dy + dz * dz).powf(1.5);
    }
    acc
}

fn proatom_tables(r: &Value, zs: &[i32]) -> BTreeMap<i32, RadialProatom> {
    let radii = scf_proatom_radii();
    let mut out = BTreeMap::new();
    for &z in zs {
        let rho: Vec<f64> = r
            .pointer(&format!("/free_atoms/{z}/proatom_rho"))
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        out.insert(z, RadialProatom::new(radii.clone(), rho).unwrap());
    }
    out
}

#[test]
#[ignore = "measurement: Hirshfeld volume grid convergence"]
fn measure_volume_grid_convergence() {
    let grids: [(usize, usize); 8] = [
        (75, 110),
        (75, 194),
        (75, 302),
        (75, 434),
        (75, 590),
        (99, 434),
        (99, 590),
        (200, 590),
    ];

    // Molecules (SCF proatom, reference density).
    for sys in ["h2o", "co", "ch3oh"] {
        for b in ["cc-pvdz", "def2-svp"] {
            let r = reference(&format!("{sys}_{b}"));
            let xyz = workspace_root()
                .join("testdata/molecules/validation")
                .join(format!("{sys}.xyz"));
            let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1).unwrap();
            let bs = basis::bundled(b).unwrap();
            let d = mat(&r, "/density_total_ferric_order");
            let mut zs: Vec<i32> = mol.atoms.iter().map(|a| a.z).collect();
            zs.sort_unstable();
            zs.dedup();
            let tabs = proatom_tables(&r, &zs);
            let dense: Vec<f64> = r
                .pointer("/hirshfeld_volumes/scf_proatom_dense_grid")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let nelec = r.pointer("/nelectron").unwrap().as_f64().unwrap();
            println!("\n== {sys}/{b} (dense ref {dense:?})");
            for (nr, na) in grids {
                let cfg = AtomicGridConfig {
                    n_radial: nr,
                    n_angular: na,
                    prune: None,
                };
                let (v, n_e, npts) = volumes_on(&mol, &bs, &d, &tabs, &cfg);
                let worst = v
                    .iter()
                    .zip(&dense)
                    .map(|(a, bb)| (a - bb).abs() / bb.abs())
                    .fold(0.0_f64, f64::max);
                let which = v
                    .iter()
                    .zip(&dense)
                    .enumerate()
                    .max_by(|x, y| {
                        let fx = (x.1 .0 - x.1 .1).abs() / x.1 .1.abs();
                        let fy = (y.1 .0 - y.1 .1).abs() / y.1 .1.abs();
                        fx.partial_cmp(&fy).unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap();
                println!(
                    "  ({nr:3},{na:3}) npts {npts:7}: worst rel {worst:.3e} at atom {which} \
                     ({}); dN_e {:+.2e}",
                    mol.atoms[which].symbol,
                    n_e - nelec
                );
            }
        }
    }

    // Free atoms (None proatom, reference density): the TS denominator.
    for sym in ["h", "c", "o"] {
        for b in ["cc-pvdz", "def2-svp"] {
            let r = reference(&format!("{sym}_atom_volume_{b}"));
            let mult = r["multiplicity"].as_i64().unwrap() as usize;
            let xyz = workspace_root()
                .join("testdata/molecules/validation")
                .join(format!("{sym}_atom.xyz"));
            let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, mult).unwrap();
            let bs = basis::bundled(b).unwrap();
            let d = mat(&r, "/density_total_ferric_order");
            let dense = r.pointer("/volume_dense_grid").unwrap().as_f64().unwrap();
            let empty: BTreeMap<i32, RadialProatom> = BTreeMap::new();
            println!("\n== {sym}_atom/{b} (dense ref {dense:.9})");
            for (nr, na) in grids {
                let cfg = AtomicGridConfig {
                    n_radial: nr,
                    n_angular: na,
                    prune: None,
                };
                let (v, _n, npts) = volumes_on(&mol, &bs, &d, &empty, &cfg);
                // The reference `volume_dense_grid` applies NO Hirshfeld
                // weight (gen_hirshfeld.py line ~596: `rho * rr**3 * wg`),
                // while `volumes_on` applies ρ⁰/(ρ⁰+1e-12). On one atom that
                // factor is 1 only where ρ⁰ > 1e-12; for H, ρ⁰_Slater drops
                // below the floor around 6.6 Bohr and the r³ weight makes
                // that tail matter. So the weight-one integral is the
                // like-for-like comparison, and the gap between the two IS
                // the floor loss.
                let w1 = volumes_weight_one(&mol, &bs, &d, &cfg);
                println!(
                    "  ({nr:3},{na:3}) npts {npts:7}: floored {:.9} ({:+.3e} rel) \
                     weight_one {:.9} ({:+.3e} rel); floor loss {:.3e} rel",
                    v[0],
                    (v[0] - dense) / dense,
                    w1,
                    (w1 - dense) / dense,
                    (w1 - v[0]) / w1
                );
            }
        }
    }
}
