//! Per-triplet cost microbenchmark for the periodic builds (FINDINGS
//! "Performance plan (research) — 2026-09-25", item 0). The benchmark plan
//! GUESSED 1–10 µs per shifted erfc 3-centre shell triplet; this measures it,
//! split by angular-momentum class, and separates per-call from
//! per-primitive cost, so every later optimisation (batched shim call,
//! unique-primitive shells, range split) is judged as a measured ratio.
//!
//! What it times (same engines, operators and precisions as the production
//! builds; nothing here feeds an energy):
//!
//! 1. RS-GDF SR 3-centre: one `Engine::compute_eri3_shifted` call on
//!    `erfc(ω_gdf)` per `(μ, ν | P)` shell triple, at two geometries (all
//!    shifts 0, and the aux shell displaced by the first lattice vector),
//!    `--reps` times each. Reported per `(l_μ, l_ν, l_P)` class: µs/call and
//!    µs per primitive triple (nonzero-coefficient primitives, which is what
//!    libint2 evaluates), plus a least-squares split
//!    `t_call = t_overhead + t_prim · N_prim` over all triples.
//! 2. RS-GDF SR metric: one `compute_eri2_shifted` per `(P | Q)` pair.
//! 3. hcore SR attraction: one shifted erfc(ω_h) 3-centre call against the
//!    Gaussian nucleus (ζ = 1e16, engine precision `f64::MIN_POSITIVE` as in
//!    `hcore.rs`) per `(μ, ν | C)` triple.
//! 4. One LR pair-FT chunk: `pair_ft_with_thresh` on the chunk width the
//!    RS-GDF build would use (64 MiB cap, RS-GDF's per-G scratch), for the
//!    first, middle and last chunk of the `|G|`-sorted list at
//!    `gcut = 2 ω √ln(1/precision)`, with the projected serial total.
//!
//! Run (release; single-threaded BLAS as always):
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-pbc --example pbc_triplet_bench -- \
//!     [--xyz reference/pbc/bench/diamond_prim.xyz --lattice reference/pbc/bench/diamond_prim.lattice] \
//!     [--basis sto-3g] [--aux cc-pvdz-ri] [--reps 20] [--omega 1.0]
//! ```
//! The default cell is the STO-3G diamond preflight (`diamond_prim`: fcc
//! primitive, a = 3.567 Å, embedded below so no file is needed). `--xyz` is
//! Ångström (the ferric XYZ convention); `--lattice` is three rows of Bohr
//! (the `reference/pbc/bench/*.lattice` format).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::ewald::default_ewald_omega;
use ferric_pbc::hcore::GAUSSIAN_NUCLEUS_EXPONENT;
use ferric_pbc::pair_ft::{pair_ft_bytes_per_g, pair_ft_with_thresh, DEFAULT_PAIR_FT_THRESH};
use ferric_pbc::rsgdf::{DEFAULT_RSGDF_OMEGA, DEFAULT_RSGDF_PRECISION};
use ferric_pbc::Cell;
use std::collections::BTreeMap;
use std::time::Instant;

/// `diamond_prim.xyz` (Å) as written by `reference/pbc/bench/make_geometries.py`.
const DIAMOND_PRIM_XYZ: &str = "2\ndiamond_prim\nC 0.0 0.0 0.0\nC 0.89175 0.89175 0.89175\n";
/// `diamond_prim.lattice` (Bohr rows).
const DIAMOND_PRIM_LATTICE: [[f64; 3]; 3] = [
    [0.0, 3.3703265431617879, 3.3703265431617879],
    [3.3703265431617879, 0.0, 3.3703265431617879],
    [3.3703265431617879, 3.3703265431617879, 0.0],
];
/// The RS-GDF / hcore libint2 precisions (`rsgdf.rs` ENGINE_PRECISION,
/// `hcore.rs` ERI3_ENGINE_PRECISION).
const RSGDF_ENGINE_PRECISION: f64 = 1e-20;
const HCORE_ENGINE_PRECISION: f64 = f64::MIN_POSITIVE;
/// `hcore.rs` G_CHUNK_BYTES.
const G_CHUNK_BYTES: usize = 64 << 20;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Args {
    xyz: Option<String>,
    lattice: Option<String>,
    basis: String,
    aux: String,
    reps: usize,
    omega: f64,
}

fn parse_args() -> Res<Args> {
    let mut a = Args {
        xyz: None,
        lattice: None,
        basis: "sto-3g".into(),
        aux: "cc-pvdz-ri".into(),
        reps: 20,
        omega: DEFAULT_RSGDF_OMEGA,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--xyz" => a.xyz = Some(v()?),
            "--lattice" => a.lattice = Some(v()?),
            "--basis" => a.basis = v()?,
            "--aux" => a.aux = v()?,
            "--reps" => a.reps = v()?.parse()?,
            "--omega" => a.omega = v()?.parse()?,
            other => return Err(format!("unknown argument {other:?} (see the file header)").into()),
        }
    }
    if a.xyz.is_some() != a.lattice.is_some() {
        return Err("--xyz and --lattice go together".into());
    }
    if a.reps == 0 {
        return Err("--reps must be >= 1".into());
    }
    Ok(a)
}

fn read_lattice(path: &str) -> Res<[[f64; 3]; 3]> {
    let text = std::fs::read_to_string(path)?;
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .map(|l| l.split_whitespace().map(str::parse::<f64>).collect())
        .collect::<Result<_, _>>()?;
    if rows.len() != 3 || rows.iter().any(|r| r.len() != 3) {
        return Err(format!("{path}: expected 3 rows of 3 numbers (Bohr)").into());
    }
    Ok([
        [rows[0][0], rows[0][1], rows[0][2]],
        [rows[1][0], rows[1][1], rows[1][2]],
        [rows[2][0], rows[2][1], rows[2][2]],
    ])
}

/// Per shell: `(l, nonzero-coefficient primitive count)`.
fn shell_info(p: &PreparedBasis) -> Vec<(i32, usize)> {
    p.located_shells()
        .iter()
        .map(|s| (s.l, s.coefficients.iter().filter(|c| **c != 0.0).count()))
        .collect()
}

#[derive(Default)]
struct ClassStat {
    calls: u64,
    secs: f64,
    prims: f64,
}

fn print_classes(title: &str, classes: &BTreeMap<Vec<i32>, ClassStat>) {
    println!("\n{title}");
    println!(
        "  {:<12} {:>10} {:>12} {:>14} {:>16}",
        "class", "calls", "us/call", "mean N_prim", "us/prim-triple"
    );
    let (mut calls, mut secs) = (0u64, 0.0);
    for (k, c) in classes {
        let per = c.secs / c.calls as f64;
        let np = c.prims / c.calls as f64;
        let key = format!("{k:?}");
        println!(
            "  {:<12} {:>10} {:>12.3} {:>14.1} {:>16.4}",
            key,
            c.calls,
            per * 1e6,
            np,
            per * 1e6 / np.max(1.0)
        );
        calls += c.calls;
        secs += c.secs;
    }
    println!(
        "  {:<12} {:>10} {:>12.3}   (unweighted over the classes' triples)",
        "all",
        calls,
        secs / calls.max(1) as f64 * 1e6
    );
}

/// Least squares `t = a + b x` over `(x, t)`.
fn fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let (sx, st) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), (x, t)| (a + x, b + t));
    let (mx, mt) = (sx / n, st / n);
    let (mut sxx, mut sxt) = (0.0, 0.0);
    for (x, t) in points {
        sxx += (x - mx) * (x - mx);
        sxt += (x - mx) * (t - mt);
    }
    let b = if sxx > 0.0 { sxt / sxx } else { 0.0 };
    (mt - b * mx, b)
}

fn main() -> Res<()> {
    let args = parse_args()?;
    let (mol, lattice) = match (&args.xyz, &args.lattice) {
        (Some(x), Some(l)) => (Molecule::load_xyz(x)?, read_lattice(l)?),
        _ => (
            Molecule::parse_xyz(DIAMOND_PRIM_XYZ, 0, 1)?,
            DIAMOND_PRIM_LATTICE,
        ),
    };
    let cell = Cell::new(mol, lattice)?;
    let bs = basis::bundled(&args.basis)?;
    let aux_bs = basis::bundled(&args.aux)?;
    let obs = PreparedBasis::new(cell.mol(), &bs)?;
    let aux = PreparedBasis::new(cell.mol(), &aux_bs)?;
    let (oi, ai) = (shell_info(&obs), shell_info(&aux));
    let a1 = cell.lattice()[0];
    println!(
        "cell: {} atoms, volume {:.4} Bohr^3; basis {} (nao {}, {} shells); aux {} (naux {}, {} shells)",
        cell.mol().atoms.len(),
        cell.volume(),
        args.basis,
        obs.nbasis(),
        oi.len(),
        args.aux,
        aux.nbasis(),
        ai.len()
    );
    println!(
        "RS-GDF omega {} Bohr^-1, precision {:e}; {} reps per triple and geometry",
        args.omega, DEFAULT_RSGDF_PRECISION, args.reps
    );

    // --- 1. SR 3-centre (μ ν | P)_erfc
    let mut eng = Engine::new_3center(
        Operator::erfc(args.omega),
        &obs,
        &aux,
        RSGDF_ENGINE_PRECISION,
    )?;
    let geoms = [[[0.0; 3]; 3], [a1, [0.0; 3], [0.0; 3]]];
    let mut classes: BTreeMap<Vec<i32>, ClassStat> = BTreeMap::new();
    let mut points = Vec::new();
    for (i1, &(l1, n1)) in oi.iter().enumerate() {
        for (i2, &(l2, n2)) in oi.iter().enumerate() {
            for (ip, &(lp, np)) in ai.iter().enumerate() {
                let nprim = (n1 * n2 * np) as f64;
                for shifts in geoms {
                    // One untimed warm-up call per (triple, geometry).
                    let _ = std::hint::black_box(
                        eng.compute_eri3_shifted(&obs, &aux, ip, i1, i2, shifts)?,
                    );
                    let t0 = Instant::now();
                    for _ in 0..args.reps {
                        let _ = std::hint::black_box(
                            eng.compute_eri3_shifted(&obs, &aux, ip, i1, i2, shifts)?,
                        );
                    }
                    let per = t0.elapsed().as_secs_f64() / args.reps as f64;
                    let c = classes.entry(vec![l1, l2, lp]).or_default();
                    c.calls += 1;
                    c.secs += per;
                    c.prims += nprim;
                    points.push((nprim, per));
                }
            }
        }
    }
    print_classes(
        "RS-GDF SR 3-centre shifted erfc eri3 (per (mu, nu | P) shell triple; class = (l_mu, l_nu, l_P))",
        &classes,
    );
    let (t0, t1) = fit(&points);
    println!(
        "  fit t_call = {:.3} us + {:.5} us x N_prim  (N_prim = nonzero-coefficient primitive triples)",
        t0 * 1e6,
        t1 * 1e6
    );

    // --- 2. SR metric (P | Q_T)_erfc
    let mut eng2 = Engine::new_2center(Operator::erfc(args.omega), &aux, RSGDF_ENGINE_PRECISION)?;
    let mut classes2: BTreeMap<Vec<i32>, ClassStat> = BTreeMap::new();
    for (ip, &(lp, np)) in ai.iter().enumerate() {
        for (iq, &(lq, nq)) in ai.iter().enumerate() {
            for t in [[0.0; 3], a1] {
                let _ = std::hint::black_box(eng2.compute_eri2_shifted(&aux, ip, iq, t)?);
                let t0 = Instant::now();
                for _ in 0..args.reps {
                    let _ = std::hint::black_box(eng2.compute_eri2_shifted(&aux, ip, iq, t)?);
                }
                let c = classes2.entry(vec![lp, lq]).or_default();
                c.calls += 1;
                c.secs += t0.elapsed().as_secs_f64() / args.reps as f64;
                c.prims += (np * nq) as f64;
            }
        }
    }
    print_classes(
        "RS-GDF SR metric shifted erfc eri2 (per (P | Q) shell pair; class = (l_P, l_Q))",
        &classes2,
    );

    // --- 3. hcore SR attraction (μ ν | Gaussian nucleus)_erfc(ω_h)
    let omega_h = default_ewald_omega(&cell);
    let sites: Vec<[f64; 4]> = cell
        .positions()
        .iter()
        .map(|r| [r[0], r[1], r[2], GAUSSIAN_NUCLEUS_EXPONENT])
        .collect();
    let site = SiteBasis::new(&sites, 0)?;
    let mut eng_n = Engine::new_3center(
        Operator::erfc(omega_h),
        &obs,
        &site.prep,
        HCORE_ENGINE_PRECISION,
    )?;
    let mut classes_n: BTreeMap<Vec<i32>, ClassStat> = BTreeMap::new();
    for (i1, &(l1, n1)) in oi.iter().enumerate() {
        for (i2, &(l2, n2)) in oi.iter().enumerate() {
            for &sh in &site.site_shell {
                for m in [[0.0; 3], a1] {
                    let shifts = [m, [0.0; 3], [0.0; 3]];
                    let _ = std::hint::black_box(
                        eng_n.compute_eri3_shifted(&obs, &site.prep, sh, i1, i2, shifts)?,
                    );
                    let t0 = Instant::now();
                    for _ in 0..args.reps {
                        let _ = std::hint::black_box(
                            eng_n.compute_eri3_shifted(&obs, &site.prep, sh, i1, i2, shifts)?,
                        );
                    }
                    let c = classes_n.entry(vec![l1, l2]).or_default();
                    c.calls += 1;
                    c.secs += t0.elapsed().as_secs_f64() / args.reps as f64;
                    c.prims += (n1 * n2) as f64;
                }
            }
        }
    }
    print_classes(
        &format!(
            "hcore SR attraction erfc(omega_h = {omega_h:.4}) eri3 vs the Gaussian nucleus \
             (per (mu, nu | C) triple; class = (l_mu, l_nu))"
        ),
        &classes_n,
    );

    // --- 4. One LR pair-FT chunk at the RS-GDF chunk width.
    let n = obs.nbasis();
    let naux = aux.nbasis();
    let lmax = oi
        .iter()
        .map(|&(l, _)| l.max(0) as usize)
        .max()
        .unwrap_or(0);
    let gcut = 2.0 * args.omega * (1.0 / DEFAULT_RSGDF_PRECISION).ln().sqrt();
    let gall: Vec<[f64; 3]> = cell
        .gvectors(gcut)?
        .into_iter()
        .filter(|g| g.iter().any(|v| *v != 0.0))
        .collect();
    let n_half = gall.len() / 2;
    let per_g = pair_ft_bytes_per_g(n, lmax) + n * n * 16 + naux * 48 + 64;
    let width = (G_CHUNK_BYTES / per_g).max(1);
    let n_chunks = n_half.div_ceil(width);
    let thresh = (0.01 * DEFAULT_RSGDF_PRECISION).min(DEFAULT_PAIR_FT_THRESH);
    println!(
        "\nLR pair FT (pair_ft_with_thresh, thresh {thresh:e}): |G| <= {gcut:.3}, ~{n_half} half-sphere G, \
         chunk width {width} G (64 MiB cap), {n_chunks} chunks"
    );
    let starts = [
        0,
        (gall.len() / 2).saturating_sub(width / 2),
        gall.len().saturating_sub(width),
    ];
    let mut per_g_secs = Vec::new();
    for (label, s0) in ["first", "middle", "last"].into_iter().zip(starts) {
        let chunk = &gall[s0..(s0 + width).min(gall.len())];
        let t0 = Instant::now();
        let _ = std::hint::black_box(pair_ft_with_thresh(&cell, &obs, chunk, thresh)?);
        let secs = t0.elapsed().as_secs_f64();
        per_g_secs.push(secs / chunk.len() as f64);
        println!(
            "  {label:<6} chunk ({} G, |G| {:.3}..{:.3}): {:.4} s  ({:.2} us/G)",
            chunk.len(),
            norm(chunk[0]),
            norm(chunk[chunk.len() - 1]),
            secs,
            secs / chunk.len() as f64 * 1e6
        );
    }
    let mean = per_g_secs.iter().sum::<f64>() / per_g_secs.len() as f64;
    println!(
        "  projected serial LR pair-FT total ~ {:.2} s ({n_half} G x mean {:.2} us/G; the |G| \
         window makes early chunks costlier, so this is a rough estimate)",
        mean * n_half as f64,
        mean * 1e6
    );
    Ok(())
}

fn norm(g: [f64; 3]) -> f64 {
    (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt()
}
