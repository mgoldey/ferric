//! Periodic XC grid diagnostics: partition, neighbour cutoff and radial /
//! angular ladders at a FIXED density (issue #228, dry-ice PBE grid ladder).
//!
//! Modes:
//! * `scf`: one Gamma RKS on RS-GDF J at `--grid`, printing the energy,
//!   `E_xc` and `∫ρ`; `--save-dm PATH` writes the converged total density
//!   (`u64` nbf, then nbf² little-endian `f64`, row-major) and the lattice
//!   overlap `S` to `PATH.S` in the same format.
//! * `fixed`: load `--dm PATH` and, for every `--grid` spec, build the grid
//!   and print the point count, `∫ρ` and `E_xc[D]` on that one density.
//! * `uniform`: `fixed` on `PeriodicGrid::uniform` meshes `--mesh n` (repeat).
//! * `wdiff`: for every `--grid` spec (hard cutoff), the partition weight of
//!   every generated point at the spec's `D` against the same point at
//!   `--dref` (a larger cutoff): max |Δw|, Σ w_rl |Δw| and the count of
//!   points whose weight moves by more than 1e-14.
//!
//! A grid spec is `partition:n_radial:n_angular[:D[:W]]` with partition
//! `ssf` or `becke`, `D` the neighbour cutoff in Bohr (`auto` or omitted =
//! the default) and `W` the neighbour switch width in Bohr (default 0).
//!
//! Cells: `h2` (cubic 4 Bohr, STO-3G, aux cc-pvdz-ri), `lih` (triclinic,
//! STO-3G, aux cc-pvdz-ri), `dryice` (the #228 bench cell) and `diamond`
//! (fcc primitive), both cc-pVDZ with aux cc-pvdz-ri as in
//! `reference/pbc/bench/toml_ri`.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-pbc --example grid_ladder_diag -- \
//!     --mode fixed --cell dryice --dm /path/d.bin --grid ssf:150:590 --grid becke:150:590:16
//! ```

use ferric_core::basis;
use ferric_core::mol::{Atom, Molecule};
use ferric_dft::becke::{partition_weight_over, NeighbourAtom, PartitionScheme};
use ferric_dft::lebedev::lebedev;
use ferric_dft::radial::treutler_ahlrichs_m4;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::{
    gamma_rks, resolve_neighbour_cutoff, GammaRksConfig, PeriodicGrid, PeriodicGridConfig,
    PeriodicXc, PeriodicXcConfig,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
use ferric_pbc::uhf::GammaUhfIntegrals;
use ferric_pbc::Cell;
use ndarray::Array2;
use rayon::prelude::*;
use std::io::{Read, Write};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Args {
    mode: String,
    cell: String,
    functional: String,
    grids: Vec<String>,
    meshes: Vec<usize>,
    dm: Option<String>,
    save_dm: Option<String>,
    dref: f64,
}

fn parse_args() -> Res<Args> {
    let mut a = Args {
        mode: "fixed".into(),
        cell: "h2".into(),
        functional: "PBE".into(),
        grids: Vec::new(),
        meshes: Vec::new(),
        dm: None,
        save_dm: None,
        dref: 24.0,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        let val = it.next().ok_or(format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--mode" => a.mode = val.clone(),
            "--cell" => a.cell = val.clone(),
            "--functional" => a.functional = val.clone(),
            "--grid" => a.grids.push(val.clone()),
            "--mesh" => a.meshes.push(val.parse()?),
            "--dm" => a.dm = Some(val.clone()),
            "--save-dm" => a.save_dm = Some(val.clone()),
            "--dref" => a.dref = val.parse()?,
            _ => return Err(format!("unknown flag {flag}").into()),
        }
    }
    Ok(a)
}

fn atom(symbol: &str, z: i32, r: [f64; 3]) -> Atom {
    Atom {
        symbol: symbol.into(),
        z,
        x: r[0],
        y: r[1],
        zpos: r[2],
        ghost: false,
        n_core_ecp: 0,
    }
}

fn read_lattice(path: &str) -> Res<[[f64; 3]; 3]> {
    let text = std::fs::read_to_string(path)?;
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|x| x.parse()).collect())
        .collect::<Result<_, _>>()?;
    let mut a = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] = rows[i][j];
        }
    }
    Ok(a)
}

/// `(cell, orbital basis, aux basis)`.
fn cell_of(name: &str) -> Res<(Cell, &'static str, &'static str)> {
    let mol = |atoms| Molecule {
        atoms,
        charge: 0,
        multiplicity: 1,
    };
    let bench = concat!(env!("CARGO_MANIFEST_DIR"), "/../../reference/pbc/bench");
    Ok(match name {
        "h2" => (
            Cell::new(
                mol(vec![
                    atom("H", 1, [0.1, 0.2, 0.3]),
                    atom("H", 1, [0.1, 0.2, 1.7]),
                ]),
                [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            )?,
            "sto-3g",
            "cc-pvdz-ri",
        ),
        "lih" => (
            Cell::new(
                mol(vec![
                    atom("Li", 3, [0.3, 0.2, 0.4]),
                    atom("H", 1, [0.3, 0.2, 3.4]),
                ]),
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.4, 0.3, 6.0]],
            )?,
            "sto-3g",
            "cc-pvdz-ri",
        ),
        "dryice" | "diamond" => {
            let stem = if name == "dryice" {
                "dryice"
            } else {
                "diamond_prim"
            };
            let m = Molecule::load_xyz(&format!("{bench}/{stem}.xyz"))?;
            let lat = read_lattice(&format!("{bench}/{stem}.lattice"))?;
            (Cell::new(m, lat)?, "cc-pvdz", "cc-pvdz-ri")
        }
        _ => return Err(format!("unknown cell {name}").into()),
    })
}

fn parse_spec(spec: &str) -> Res<PeriodicGridConfig> {
    let f: Vec<&str> = spec.split(':').collect();
    if f.len() < 3 {
        return Err(format!("grid spec {spec:?}: need partition:n_radial:n_angular").into());
    }
    let mut c = PeriodicGridConfig::with_size(f[1].parse()?, f[2].parse()?);
    c.partition = PartitionScheme::parse_config_str(f[0])?;
    if let Some(d) = f.get(3) {
        if *d != "auto" {
            c.neighbour_cutoff = Some(d.parse()?);
        }
    }
    if let Some(w) = f.get(4) {
        c.neighbour_switch = w.parse()?;
    }
    Ok(c)
}

fn save_dm(path: &str, d: &Array2<f64>) -> Res<()> {
    let n = d.nrows();
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(n as u64).to_le_bytes())?;
    for v in d.as_standard_layout().iter() {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn load_dm(path: &str) -> Res<Array2<f64>> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    let n = u64::from_le_bytes(buf[0..8].try_into()?) as usize;
    let vals: Vec<f64> = buf[8..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    Ok(Array2::from_shape_vec((n, n), vals)?)
}

fn run_scf(cell: &Cell, bs: &basis::BasisSet, aux_name: &str, a: &Args) -> Res<()> {
    let prep = PreparedBasis::new(cell.mol(), bs)?;
    let hc = periodic_hcore(cell, &prep, &PeriodicHcoreConfig::for_cell(cell))?;
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled(aux_name)?)?;
    let gdf = RsGdf::build(
        cell,
        &prep,
        &aux,
        &hc.s,
        &RsGdfConfig {
            exxdiv: ExxDiv::Ewald,
            ..Default::default()
        },
    )?;
    for spec in &a.grids {
        let mut cfg = GammaRksConfig::new(&a.functional);
        cfg.grid = parse_spec(spec)?;
        let t = std::time::Instant::now();
        let r = gamma_rks(cell, &prep, &hc, GammaUhfIntegrals::RsGdf(&gdf), &cfg)?;
        println!(
            "SCF {spec} D={:.4} npts={} iters={} E={:.10} Exc={:.10} N={:.10} wall={:.1}s",
            r.neighbour_cutoff,
            r.n_grid_points,
            r.scf.iterations,
            r.scf.energy,
            r.e_xc,
            r.electrons_on_grid,
            t.elapsed().as_secs_f64()
        );
        if let Some(p) = &a.save_dm {
            save_dm(p, &r.scf.density_total)?;
            save_dm(&format!("{p}.S"), &hc.s)?;
            println!("saved density to {p} and the lattice overlap to {p}.S");
        }
    }
    Ok(())
}

fn fixed_on(
    cell: &Cell,
    bs: &basis::BasisSet,
    a: &Args,
    d: &Array2<f64>,
    label: &str,
    grid: &PeriodicGrid,
) -> Res<()> {
    let t = std::time::Instant::now();
    let mut xc = PeriodicXc::new(cell, bs, &a.functional, grid, &PeriodicXcConfig::default())?;
    let n = xc.integrate_density(d)?;
    let (exc, _) = xc.eval(d)?;
    println!(
        "FIXED {label} D={:.4} npts={} ngen={} wsum={:.12} N={:.12} Exc={:.12} wall={:.1}s",
        grid.neighbour_cutoff(),
        grid.len(),
        grid.n_generated(),
        grid.weight_sum(),
        n,
        exc,
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Partition weights at the spec's cutoff against `dref`, point by point.
fn wdiff(cell: &Cell, spec: &str, dref: f64) -> Res<()> {
    let cfg = parse_spec(spec)?;
    let (d1, _) = resolve_neighbour_cutoff(cell, cfg.neighbour_cutoff)?;
    let d2 = dref.max(d1);
    let pos = cell.positions();
    let zs: Vec<i32> = cell.mol().atoms.iter().map(|x| x.z).collect();
    let (leb, lw) = lebedev(cfg.n_angular);
    let trans = cell.translations(2.0 * d2)?;
    let dist = |p: &[f64; 3], q: &[f64; 3]| {
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    };
    let (mut max_dw, mut sum_dw, mut n_moved, mut n_pts) = (0.0_f64, 0.0_f64, 0usize, 0usize);
    let mut worst = (0usize, 0.0_f64, 0.0_f64, 0.0_f64);
    for (home, ra) in pos.iter().enumerate() {
        let mut cands: Vec<NeighbourAtom> = Vec::new();
        let mut home_c = usize::MAX;
        for (li, l) in trans.iter().enumerate() {
            for (b, rb) in pos.iter().enumerate() {
                let xyz = [rb[0] + l[0], rb[1] + l[1], rb[2] + l[2]];
                if dist(&xyz, ra) <= 2.0 * d2 {
                    if b == home && li == 0 {
                        home_c = cands.len();
                    }
                    cands.push(NeighbourAtom { xyz, z: zs[b] });
                }
            }
        }
        let (rs, ws) = treutler_ahlrichs_m4(zs[home], cfg.n_radial);
        let mut pre = Vec::new();
        for (r, wr) in rs.iter().zip(&ws) {
            if *r > d1 {
                continue;
            }
            for (u, wl) in leb.iter().zip(&lw) {
                pre.push((
                    [ra[0] + r * u[0], ra[1] + r * u[1], ra[2] + r * u[2]],
                    wr * wl,
                    *r,
                ));
            }
        }
        let res: Vec<(f64, f64, f64)> = pre
            .par_iter()
            .map(|&(xyz, wrl, r)| {
                let w_at = |dc: f64| {
                    let mut nb = Vec::new();
                    let mut h = usize::MAX;
                    for (ci, c) in cands.iter().enumerate() {
                        if ci == home_c {
                            h = nb.len();
                            nb.push(*c);
                        } else if dist(&c.xyz, &xyz) <= dc {
                            nb.push(*c);
                        }
                    }
                    partition_weight_over(cfg.partition, &nb, h, xyz)
                };
                let w1 = w_at(d1);
                let w2 = w_at(d2);
                (wrl, (w1 - w2).abs(), r)
            })
            .collect();
        for (wrl, dw, r) in res {
            n_pts += 1;
            sum_dw += wrl * dw;
            if dw > 1e-14 {
                n_moved += 1;
            }
            if dw > max_dw {
                max_dw = dw;
                worst = (home, r, wrl, dw);
            }
        }
    }
    println!(
        "WDIFF {spec} D={d1:.4} vs Dref={d2:.4}: points={n_pts} moved(>1e-14)={n_moved} \
         max|dw|={max_dw:.3e} sum w_rl|dw|={sum_dw:.3e} worst(home={}, r={:.3}, w_rl={:.3e})",
        worst.0, worst.1, worst.2
    );
    Ok(())
}

fn main() -> Res<()> {
    let a = parse_args()?;
    let (cell, bname, aux) = cell_of(&a.cell)?;
    let bs = basis::bundled(bname)?;
    println!(
        "cell {} basis {bname} functional {} threads {}",
        a.cell,
        a.functional,
        rayon::current_num_threads()
    );
    match a.mode.as_str() {
        "scf" => run_scf(&cell, &bs, aux, &a),
        "fixed" | "uniform" => {
            let d = load_dm(a.dm.as_deref().ok_or("--dm is required")?)?;
            if a.mode == "fixed" {
                for spec in &a.grids {
                    let g = PeriodicGrid::build(&cell, &parse_spec(spec)?)?;
                    fixed_on(&cell, &bs, &a, &d, spec, &g)?;
                }
            } else {
                for &m in &a.meshes {
                    let g = PeriodicGrid::uniform(&cell, [m, m, m])?;
                    fixed_on(&cell, &bs, &a, &d, &format!("uniform:{m}"), &g)?;
                }
            }
            Ok(())
        }
        "wdiff" => {
            for spec in &a.grids {
                wdiff(&cell, spec, a.dref)?;
            }
            Ok(())
        }
        m => Err(format!("unknown mode {m}").into()),
    }
}
