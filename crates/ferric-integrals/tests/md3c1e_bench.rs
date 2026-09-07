//! md3c1e vs cosx_a per-grid-point cost (butane, def2-SVP/TZVP/QZVP).
//!
//! Pre-registration: `scripts/queue/out/md3c1e_prereg.md` (committed before
//! any number here was taken). One process per cell:
//!
//! ```text
//! MD_BENCH_BASIS=def2-qzvp MD_BENCH_NPTS=2000 \
//!   cargo test -p ferric-integrals --release --test md3c1e_bench -- --ignored --nocapture
//! ```
//!
//! Points: `MD_BENCH_NPTS` (default 2000) drawn uniformly at random without
//! replacement (fixed-seed LCG, seed 20260907 — the L-axis harness's sampler)
//! from the (50,110) Becke grid, whose point POSITIONS are rebuilt here from
//! the Treutler-Ahlrichs M4 radial formula and `ferric_quadrature::lebedev`
//! (ferric-integrals cannot depend on ferric-dft; the count is asserted to be
//! `natoms * 50 * 110`, i.e. 77000 on butane, matching l_axis_results.md).
//!
//! Single thread, release; PSI `full avg10` printed before/after every timed
//! segment (discard nonzero); cpu-seconds printed next to wall.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::cosx_a::{a_matrix_at_point_with, CosxScreen};
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::md3c1e::{self, Md3c1e};
use ferric_quadrature::lebedev::lebedev;
use std::time::Instant;

const SEED: u64 = 20260907;
const N_RADIAL: usize = 50;
const N_ANGULAR: usize = 110;

fn testdata(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn env_num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn psi_full_avg10() -> String {
    std::fs::read_to_string("/proc/pressure/memory")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("full"))
                .and_then(|l| l.split_whitespace().nth(1).map(|kv| kv.trim_start_matches("avg10=").to_string()))
        })
        .unwrap_or_else(|| "n/a".to_string())
}

fn cpu_seconds() -> f64 {
    std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| {
            let rest = s.rsplit(')').next()?;
            let f: Vec<&str> = rest.split_whitespace().collect();
            let ut: f64 = f.get(11)?.parse().ok()?;
            let st: f64 = f.get(12)?.parse().ok()?;
            Some((ut + st) / 100.0)
        })
        .unwrap_or(f64::NAN)
}

/// Returns (wall s, cpu s, value); prints PSI before/after.
fn timed<T>(label: &str, f: impl FnOnce() -> T) -> (f64, f64, T) {
    let before = psi_full_avg10();
    let c0 = cpu_seconds();
    let t0 = Instant::now();
    let v = f();
    let secs = t0.elapsed().as_secs_f64();
    let cpu = cpu_seconds() - c0;
    let after = psi_full_avg10();
    println!("  [timed:{label}] wall {secs:.3} s  cpu {cpu:.2} s  PSI full avg10 before={before} after={after}");
    (secs, cpu, v)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

fn sample_indices(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = Lcg(seed);
    let k = k.min(n);
    for i in 0..k {
        let j = i + (rng.next() as usize) % (n - i);
        idx.swap(i, j);
    }
    idx.truncate(k);
    idx
}

/// Treutler-Ahlrichs ξ (JCP 102, 346 (1995) Table I) for Z = 1..10, as in
/// `ferric_dft::radial`.
fn ta_xi(z: i32) -> f64 {
    const XI: [f64; 11] = [1.0, 0.8, 0.9, 1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9];
    if (z as usize) < XI.len() { XI[z as usize] } else { 1.5 }
}

/// TA-M4 radii (small→large), the same formula as `ferric_dft::radial::treutler_ahlrichs_m4`.
fn ta_m4_radii(z: i32, n: usize) -> Vec<f64> {
    let alpha = 0.6_f64;
    let ln2 = ta_xi(z) / 2.0_f64.ln();
    let np1 = (n + 1) as f64;
    let mut rs: Vec<f64> = (1..=n)
        .map(|k| {
            let x = (std::f64::consts::PI * k as f64 / np1).cos();
            -ln2 * (1.0 + x).powf(alpha) * ((1.0 - x) / 2.0).ln()
        })
        .collect();
    rs.reverse();
    rs
}

/// Point positions of the flat (N_RADIAL, N_ANGULAR) Becke grid.
fn grid_points(mol: &Molecule) -> Vec<[f64; 3]> {
    let (leb, _) = lebedev(N_ANGULAR);
    let mut pts = Vec::with_capacity(mol.atoms.len() * N_RADIAL * leb.len());
    for atom in &mol.atoms {
        for r in ta_m4_radii(atom.z, N_RADIAL) {
            for p in &leb {
                pts.push([atom.x + r * p[0], atom.y + r * p[1], atom.zpos + r * p[2]]);
            }
        }
    }
    pts
}

fn measure_md_batched(kern: &Md3c1e, pts: &[[f64; 3]], batch: usize) -> f64 {
    let mut scr = kern.scratch();
    let (secs, _cpu, checksum) = timed(&format!("md3c1e batched B={batch}"), || {
        let mut checksum = 0.0_f64;
        for chunk in pts.chunks(batch) {
            kern.for_each_pair(chunk, None, CosxScreen::none(), &mut scr, |_s1, _s2, blk| {
                checksum += blk[0] + blk[blk.len() - 1];
            })
            .expect("md3c1e batch");
        }
        checksum
    });
    println!("    checksum {checksum:.6e} (touches every block; prevents elimination)");
    secs / pts.len() as f64
}

#[test]
#[ignore = "measurement; run explicitly with --ignored --nocapture, one basis per process"]
fn md3c1e_vs_cosx_a_per_point_cell() {
    let system = std::env::var("MD_BENCH_SYSTEM").unwrap_or_else(|_| "alkane_4".into());
    let basis = std::env::var("MD_BENCH_BASIS").unwrap_or_else(|_| "def2-svp".into());
    let npts: usize = env_num("MD_BENCH_NPTS", 2000);
    let n_dropin: usize = env_num("MD_BENCH_DROPIN_NPTS", 200);

    let mol = Molecule::load_xyz(&testdata(&format!("testdata/molecules/{system}.xyz"))).expect("xyz");
    let bs = bundled(&basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let kern = Md3c1e::new(&prep).expect("md3c1e");
    let nbf = prep.nbasis();
    println!(
        "\n=== md3c1e bench: {system} / {basis}  natoms={} nsh={} nbf={nbf} L_max={} fma={} PSI now={} ===",
        mol.atoms.len(),
        prep.nshells(),
        prep.max_l(),
        kern.uses_fma(),
        psi_full_avg10()
    );

    let grid = grid_points(&mol);
    assert_eq!(grid.len(), mol.atoms.len() * N_RADIAL * N_ANGULAR, "grid point count");
    let idx = sample_indices(grid.len(), npts, SEED);
    let pts: Vec<[f64; 3]> = idx.iter().map(|&i| grid[i]).collect();
    println!("grid (50,110): {} points; sampled {} at random (seed {SEED})", grid.len(), pts.len());

    // ---- preflight: the kernel being timed IS the anchored kernel ----
    let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, &prep, 1e-14).expect("nuclear engine");
    {
        let mut scr = kern.scratch();
        let batch = kern.a_matrices(&pts[..3], None, CosxScreen::none(), &mut scr).expect("md batch");
        let mut worst = 0.0_f64;
        for (k, r) in pts[..3].iter().enumerate() {
            let want = a_matrix_at_point_with(&mut eng, &prep, r, None, CosxScreen::none()).expect("cosx_a").a;
            let got = batch.a.index_axis(ndarray::Axis(0), k);
            let d = (&got - &want).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v));
            worst = worst.max(d);
        }
        println!("preflight: max|A_md - A_cosx_a| over 3 sampled points = {worst:.3e}");
        assert!(worst <= 1e-12, "refusing to time a kernel that disagrees with cosx_a");
    }

    // ---- FLOP count (spec convention) ----
    let flops = kern.flops_per_point();
    println!(
        "operation count per point: R-tensor {:.0}  contraction {:.0}  total {:.0}",
        flops.r_tensor, flops.contraction, flops.total()
    );

    // ---- (1) cosx_a per point ----
    let (secs_cosx, _, _) = timed("cosx_a unscreened, one reused engine", || {
        let mut acc = 0.0_f64;
        for r in &pts {
            let p = a_matrix_at_point_with(&mut eng, &prep, r, None, CosxScreen::none()).expect("cosx_a");
            acc += p.a[(0, 0)];
        }
        acc
    });
    let cosx_per_pt = secs_cosx / pts.len() as f64;

    // ---- (2) md3c1e batched, three batch sizes ----
    let md_64 = measure_md_batched(&kern, &pts, 64);
    let md_256 = measure_md_batched(&kern, &pts, 256);
    let md_1024 = measure_md_batched(&kern, &pts, 1024);

    // ---- (3) md3c1e drop-in single point ----
    let n_dropin = n_dropin.min(pts.len());
    let (secs_dropin, _, _) = timed(&format!("md3c1e drop-in single point x{n_dropin}"), || {
        let mut scr = kern.scratch();
        let mut acc = 0.0_f64;
        for r in &pts[..n_dropin] {
            let p = md3c1e::a_matrix_at_point_with(&kern, &mut scr, r, None, CosxScreen::none()).expect("md drop-in");
            acc += p.a[(0, 0)];
        }
        acc
    });
    let dropin_per_pt = secs_dropin / n_dropin as f64;

    let gf = |s_per_pt: f64| flops.total() / s_per_pt / 1e9;
    println!(
        "\nper point: cosx_a {:.4e} s | md3c1e B=64 {:.4e} s  B=256 {:.4e} s  B=1024 {:.4e} s | drop-in {:.4e} s",
        cosx_per_pt, md_64, md_256, md_1024, dropin_per_pt
    );
    println!(
        "speedup vs cosx_a: B=64 {:.2}x  B=256 {:.2}x  B=1024 {:.2}x  (drop-in {:.2}x)",
        cosx_per_pt / md_64,
        cosx_per_pt / md_256,
        cosx_per_pt / md_1024,
        cosx_per_pt / dropin_per_pt
    );
    println!(
        "effective GFLOP/s (spec op count): cosx_a/libint2 {:.2} | md3c1e B=64 {:.2}  B=256 {:.2}  B=1024 {:.2}",
        gf(cosx_per_pt),
        gf(md_64),
        gf(md_256),
        gf(md_1024)
    );
    println!(
        "\nROW | {system} | {basis} | L={} | nbf={nbf} | npts={} | cosx_a/pt={:.4e} | md/pt B=256={:.4e} | speedup B=64/256/1024={:.2}/{:.2}/{:.2} | flops/pt={:.0} | GF/s md(B=256)={:.2} libint2={:.2} | PSI end={}",
        prep.max_l(),
        pts.len(),
        cosx_per_pt,
        md_256,
        cosx_per_pt / md_64,
        cosx_per_pt / md_256,
        cosx_per_pt / md_1024,
        flops.total(),
        gf(md_256),
        gf(cosx_per_pt),
        psi_full_avg10()
    );
}

/// Operation count for water/cc-pVDZ, printed next to the spec's numbers
/// (§7: R-tensor 13,360; contraction 27,801; total 41,161) so the counting
/// convention used for GFLOP/s can be compared. Not asserted equal — the
/// spec does not state its exact per-element convention.
#[test]
fn md3c1e_flop_count_calibration_water_ccpvdz() {
    let mol = Molecule::load_xyz(&testdata("testdata/molecules/water.xyz")).expect("xyz");
    let bs = bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let kern = Md3c1e::new(&prep).expect("md3c1e");
    let f = kern.flops_per_point();
    println!(
        "water/cc-pVDZ: nsh={} pairs={}  R-tensor {:.0} (spec 13,360)  contraction {:.0} (spec 27,801)  total {:.0} (spec 41,161)",
        prep.nshells(),
        prep.nshells() * (prep.nshells() + 1) / 2,
        f.r_tensor,
        f.contraction,
        f.total()
    );
    assert_eq!(prep.nshells(), 12, "spec's segmented shell count for water/cc-pVDZ");
    assert!(f.total() > 0.0);
}

/// Time attribution without `perf` (blocked on this box): the Boys evaluator
/// costs one `exp` + table Taylor + downward recursion per (primitive pair,
/// point), which the FLOP count excludes. Times `md3c1e::boys` in isolation
/// at `n_max = 0, 4, 8` over T drawn from the actual T distribution of the
/// cell's own sweep is NOT available here, so T is uniform on [0, 60) (both
/// branches exercised), and multiplies by the cell's surviving primitive
/// pairs per point.
#[test]
#[ignore = "measurement; run with --ignored --nocapture"]
fn md3c1e_boys_cost_attribution() {
    let system = std::env::var("MD_BENCH_SYSTEM").unwrap_or_else(|_| "alkane_4".into());
    let basis = std::env::var("MD_BENCH_BASIS").unwrap_or_else(|_| "def2-svp".into());
    let mol = Molecule::load_xyz(&testdata(&format!("testdata/molecules/{system}.xyz"))).expect("xyz");
    let bs = bundled(&basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let kern = Md3c1e::new(&prep).expect("md3c1e");
    let f = kern.flops_per_point();
    let n_eval = 2_000_000usize;
    let mut rng = Lcg(SEED);
    let ts: Vec<f64> = (0..n_eval).map(|_| 60.0 * (rng.next() as f64) / ((1u64 << 53) as f64)).collect();
    let mut out = [0.0_f64; 9];
    let mut ns = [0.0_f64; 3];
    for (k, nmax) in [0usize, 4, 8].iter().enumerate() {
        let (secs, _, acc) = timed(&format!("boys n_max={nmax} x{n_eval}"), || {
            let mut acc = 0.0;
            for &t in &ts {
                md3c1e::boys(*nmax, t, &mut out);
                acc += out[*nmax];
            }
            acc
        });
        ns[k] = secs / n_eval as f64 * 1e9;
        println!("    n_max={nmax}: {:.1} ns/eval (acc {acc:.3e})", ns[k]);
    }
    println!(
        "{system}/{basis}: surviving primitive pairs per point = {:.0}; Boys share of per-point time if every pair cost the n_max=0 / 4 / 8 evaluator: {:.3e} / {:.3e} / {:.3e} s",
        f.prim_pairs,
        f.prim_pairs * ns[0] * 1e-9,
        f.prim_pairs * ns[1] * 1e-9,
        f.prim_pairs * ns[2] * 1e-9
    );
}
