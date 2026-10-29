//! Per-iteration k-point XC build benchmark (issue #356, Performance).
//!
//! Builds the Bloch AO cache once ([`KPeriodicXc::new`], timed as stage
//! `"k XC AO cache"`), then calls the closed-shell `build` and the open-shell
//! `build_polarized` `--reps` times each on a fixed density and prints the
//! phase breakdown of [`KPeriodicXc::xc_timings`] (density GEMM, density
//! contraction, tau, libxc kernel, Vxc scale / GEMM / add). Nothing here
//! feeds an energy; the printed `E_xc` is a fingerprint that two builds of the
//! same code must reproduce bit for bit.
//!
//! The density is `D(k) = c I` with `c` set so that `∫ρ = N_e` (positive,
//! atom-shaped; the build cost does not depend on its values beyond the
//! density floor). The open shell uses `D_α = 0.6 D`, `D_β = 0.4 D`.
//!
//! Cells (bundled bases, the default atom-centred (75, 302) grid):
//! * `h2`: the test suite's H2 cell, cubic 4 Bohr, STO-3G;
//! * `lih`: the triclinic all-electron LiH cell of
//!   `tests/validation_pbc_kks_core.rs`, STO-3G;
//! * `diamond`: the fcc primitive diamond of `reference/pbc/bench`, 6-31G.
//!
//! `--scf dense|rsgdf` instead runs one closed-shell `solve_krks` on that J/K
//! (RS-GDF aux `cc-pvdz-ri`) and prints its stage record, so the XC build's
//! share of the SCF (sub-stage `"k XC build"` inside stage `"k SCF"`) and the
//! AO cache's share of the run are measured, not estimated.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-pbc --example kxc_build_bench -- \
//!     [--cell h2|lih|diamond] [--mesh 2,2,2] [--functional PBE] [--reps 3] \
//!     [--radial 75] [--angular 302] [--scf dense|rsgdf]
//! ```

use ferric_core::basis;
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::{PeriodicGrid, PeriodicGridConfig, PeriodicXcConfig};
use ferric_pbc::kuks::KPeriodicXcPolarized;
use ferric_pbc::kuscf::KPointXcPolarized;
use ferric_pbc::timing::{PbcTimings, StageClock};
use ferric_pbc::{solve_krks, Cell, KJkKind, KPeriodicXc, KPointMesh, KPointXc, KRksConfig};
use ndarray::Array2;
use num_complex::Complex64;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Args {
    cell: String,
    mesh: [usize; 3],
    functional: String,
    reps: usize,
    radial: usize,
    angular: usize,
    scf: Option<String>,
}

fn parse_args() -> Res<Args> {
    let mut a = Args {
        cell: "lih".into(),
        mesh: [2, 2, 2],
        functional: "PBE".into(),
        reps: 3,
        radial: 75,
        angular: 302,
        scf: None,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        let val = it.next().ok_or(format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--cell" => a.cell = val.clone(),
            "--functional" => a.functional = val.clone(),
            "--reps" => a.reps = val.parse()?,
            "--radial" => a.radial = val.parse()?,
            "--angular" => a.angular = val.parse()?,
            "--scf" => a.scf = Some(val.clone()),
            "--mesh" => {
                let v: Vec<usize> = val
                    .split(',')
                    .map(|x| x.parse())
                    .collect::<Result<_, _>>()?;
                a.mesh = v.try_into().map_err(|_| "--mesh takes n1,n2,n3")?;
            }
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

/// `(cell, bundled basis name)` of a named benchmark cell.
fn cell_of(name: &str) -> Res<(Cell, &'static str)> {
    let mol = |atoms| Molecule {
        atoms,
        charge: 0,
        multiplicity: 1,
    };
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
        ),
        "diamond" => {
            let a = 3.3703265431617879;
            let x = 0.89175 * ferric_core::units::ANGSTROM_TO_BOHR;
            (
                Cell::new(
                    mol(vec![atom("C", 6, [0.0; 3]), atom("C", 6, [x, x, x])]),
                    [[0.0, a, a], [a, 0.0, a], [a, a, 0.0]],
                )?,
                "6-31g",
            )
        }
        _ => return Err(format!("unknown cell {name} (h2, lih, diamond)").into()),
    })
}

fn print_timings(label: &str, t: &PbcTimings, reps: usize) {
    let build = t.stage("k XC build").map_or(0.0, |s| s.wall_s);
    let cache = t.stage("k XC AO cache").map_or(0.0, |s| s.wall_s);
    println!("[{label}] AO cache build {cache:.3} s (once)");
    println!(
        "[{label}] build: {:.4} s/call over {reps} calls",
        build / reps.max(1) as f64
    );
    // Phase clocks are thread-summed: compare them as shares of their own sum.
    let sub: f64 = t.substages.iter().map(|s| s.wall_s).sum();
    for s in &t.substages {
        println!(
            "[{label}]   {:<34} {:>9.4} thread-s/call  {:>5.1}% of phases",
            s.name,
            s.wall_s / reps.max(1) as f64,
            100.0 * s.wall_s / sub.max(1e-300)
        );
    }
    if rayon::current_num_threads() == 1 {
        println!(
            "[{label}]   {:<34} {:>9.4} s/call (build minus phases)",
            "other (finite check, hermitise)",
            (build - sub) / reps.max(1) as f64
        );
    }
}

/// One closed-shell SCF on `jk`; prints every stage and sub-stage.
fn run_scf(cell: &Cell, bs: &basis::BasisSet, mesh: &KPointMesh, a: &Args, jk: &str) -> Res<()> {
    let prep = PreparedBasis::new(cell.mol(), bs)?;
    let mut cfg = KRksConfig::new(cell, ExxDiv::None, &a.functional);
    cfg.grid.n_radial = a.radial;
    cfg.grid.n_angular = a.angular;
    let aux = match jk {
        "dense" => None,
        "rsgdf" => {
            cfg.krhf.jk = KJkKind::RsGdf;
            Some(PreparedBasis::new(
                cell.mol(),
                &basis::bundled("cc-pvdz-ri")?,
            )?)
        }
        _ => return Err(format!("--scf takes dense or rsgdf, got {jk}").into()),
    };
    let total = StageClock::start();
    let r = solve_krks(cell, &prep, aux.as_ref(), mesh, &cfg)?;
    println!(
        "SCF {jk}: E {:.12} in {} iterations, {:.2} s wall",
        r.scf.energy,
        r.scf.iterations,
        total.elapsed().0
    );
    for s in &r.scf.timings.stages {
        println!(
            "  stage    {:<34} {:>9.3} s ({} calls)",
            s.name, s.wall_s, s.calls
        );
    }
    for s in &r.scf.timings.substages {
        println!(
            "  substage {:<34} {:>9.3} s ({} calls)",
            s.name, s.wall_s, s.calls
        );
    }
    Ok(())
}

fn main() -> Res<()> {
    let a = parse_args()?;
    let (cell, bname) = cell_of(&a.cell)?;
    let bs = basis::bundled(bname)?;
    let mesh = KPointMesh::gamma_centred(&cell, a.mesh)?;
    if let Some(jk) = &a.scf {
        return run_scf(&cell, &bs, &mesh, &a, jk);
    }
    let gcfg = PeriodicGridConfig {
        n_radial: a.radial,
        n_angular: a.angular,
        ..Default::default()
    };
    let t = StageClock::start();
    let grid = PeriodicGrid::build(&cell, &gcfg)?;
    println!(
        "cell {} basis {bname} mesh {:?} functional {} grid ({}, {}): {} points, built in {:.2} s; \
         rayon threads {}",
        a.cell,
        a.mesh,
        a.functional,
        a.radial,
        a.angular,
        grid.len(),
        t.elapsed().0,
        rayon::current_num_threads()
    );
    let mut xc = KPeriodicXc::new(
        &cell,
        &bs,
        &a.functional,
        &grid,
        &mesh,
        &PeriodicXcConfig::default(),
    )?;
    let n = xc.nbasis();
    println!("nbf {n} nk {}", mesh.nk());
    let unit: Vec<Array2<Complex64>> = (0..mesh.nk())
        .map(|_| Array2::<Complex64>::eye(n))
        .collect();
    let ne = cell.mol().nelec() as f64;
    let c = ne / xc.integrate_density(&unit)?;
    let dm: Vec<Array2<Complex64>> = unit.iter().map(|m| m.mapv(|z| z * c)).collect();

    let mut e = 0.0;
    for _ in 0..a.reps {
        e = xc.build(&dm)?.0;
    }
    println!("closed E_xc fingerprint {e:.15e}");
    print_timings("closed", &xc.xc_timings(), a.reps);

    let da: Vec<_> = dm.iter().map(|m| m.mapv(|z| z * 0.6)).collect();
    let db: Vec<_> = dm.iter().map(|m| m.mapv(|z| z * 0.4)).collect();
    let fresh = KPeriodicXc::new(
        &cell,
        &bs,
        &a.functional,
        &grid,
        &mesh,
        &PeriodicXcConfig::default(),
    )?;
    let mut pol = KPeriodicXcPolarized::from_closed(fresh)?;
    for _ in 0..a.reps {
        e = pol.build_polarized(&da, &db)?.0;
    }
    println!("open E_xc fingerprint {e:.15e}");
    print_timings("open", &pol.xc_timings(), a.reps);
    Ok(())
}
