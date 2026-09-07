//! Sparsity measurement for the VALID COSX shell-pair screen
//! (`cosx_a::PairBounds`, Hölder bound) — pre-registered in
//! `scripts/queue/out/screen_fix_prereg.md`.
//!
//! Counts, not timings, answer the sparsity question (they are
//! load-independent): for a fixed-seed sample of `SCREEN_NPTS` (default 2000)
//! positions of the flat (50,110) Becke grid, the per-POINT number of shell
//! pairs with `estimate >= t` at `t = 1e-7, 1e-8`, reported as mean / min /
//! max and as a fraction of `nsh(nsh+1)/2`. Also reported: the batch-level
//! kept count (`any_exceeds` over batches of 256 UNSORTED random points —
//! an upper bound on what an unsorted-batch implementation realizes, not the
//! intrinsic sparsity), and the screen's own cost per point (all pairs,
//! `estimate`) — optionally against the md3c1e A-build per point
//! (`SCREEN_TIME_ABUILD=1`, one batch of 256 points, unscreened).
//!
//! One system per process: `SCREEN_SYSTEM=alkane_12 SCREEN_BASIS=def2-svp`.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::cosx_a::{CosxScreen, PairBounds};
use ferric_integrals::md3c1e::Md3c1e;
use ferric_quadrature::lebedev::lebedev;
use std::time::Instant;

const SEED: u64 = 20260907;

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

fn ta_xi(z: i32) -> f64 {
    const XI: [f64; 11] = [1.0, 0.8, 0.9, 1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9];
    if (z as usize) < XI.len() {
        XI[z as usize]
    } else {
        1.5
    }
}

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

fn grid_points(mol: &Molecule) -> Vec<[f64; 3]> {
    let (leb, _) = lebedev(110);
    let mut pts = Vec::new();
    for atom in &mol.atoms {
        for r in ta_m4_radii(atom.z, 50) {
            for p in &leb {
                pts.push([atom.x + r * p[0], atom.y + r * p[1], atom.zpos + r * p[2]]);
            }
        }
    }
    pts
}

#[test]
#[ignore = "measurement; run explicitly with --ignored --nocapture, one system per process"]
fn cosx_screen_sparsity_cell() {
    let system = std::env::var("SCREEN_SYSTEM").unwrap_or_else(|_| "alkane_4".into());
    let basis = std::env::var("SCREEN_BASIS").unwrap_or_else(|_| "def2-svp".into());
    let npts: usize = env_num("SCREEN_NPTS", 2000);
    let time_abuild = std::env::var("SCREEN_TIME_ABUILD").map(|v| v == "1").unwrap_or(false);

    let mol = Molecule::load_xyz(&testdata(&format!("testdata/molecules/{system}.xyz"))).expect("xyz");
    let bs = bundled(&basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let t0 = Instant::now();
    let bounds = PairBounds::build(&prep).expect("bounds");
    let build_s = t0.elapsed().as_secs_f64();
    let nsh = prep.nshells();
    let total = nsh * (nsh + 1) / 2;
    let grid = grid_points(&mol);
    let idx = sample_indices(grid.len(), npts, SEED);
    let pts: Vec<[f64; 3]> = idx.iter().map(|&g| grid[g]).collect();
    println!(
        "\n=== screen sparsity: {system} / {basis}  natoms={} nsh={nsh} nbf={} pairs={total} prim-pair terms={} (build {build_s:.3} s)  sample {} of {} grid pts  PSI={} ===",
        mol.atoms.len(),
        prep.nbasis(),
        bounds.nterms(),
        pts.len(),
        grid.len(),
        psi_full_avg10()
    );

    for &t in &[1e-7_f64, 1e-8] {
        // Per-point counts (the intrinsic sparsity).
        let psi_b = psi_full_avg10();
        let t1 = Instant::now();
        let mut sum = 0usize;
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        for r in &pts {
            let mut kept = 0usize;
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    if bounds.exceeds(s1, s2, r, t) {
                        kept += 1;
                    }
                }
            }
            sum += kept;
            lo = lo.min(kept);
            hi = hi.max(kept);
        }
        let per_point_s = t1.elapsed().as_secs_f64() / pts.len() as f64;
        let psi_a = psi_full_avg10();
        let mean = sum as f64 / pts.len() as f64;
        // Coarse-only (one sqrt per pair) kept count, for the K-builder's cost/benefit.
        let mut csum = 0usize;
        for r in &pts {
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    if bounds.coarse_estimate(s1, s2, r) >= t {
                        csum += 1;
                    }
                }
            }
        }
        let cmean = csum as f64 / pts.len() as f64;
        // Grid-INDEPENDENT significant-pair list: max_estimate (the R = 0
        // value) >= t. Computed once per geometry, O(nsh^2), no per-point cost.
        let mut gsum = 0usize;
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                gsum += (bounds.max_estimate(s1, s2) >= t) as usize;
            }
        }
        // Batch-level (any over 256 unsorted random points).
        let mut bsum = 0usize;
        let mut nb = 0usize;
        for chunk in pts.chunks(256) {
            let mut kept = 0usize;
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    if bounds.any_exceeds(s1, s2, chunk, t) {
                        kept += 1;
                    }
                }
            }
            bsum += kept;
            nb += 1;
        }
        let bmean = bsum as f64 / nb as f64;
        println!(
            "SPARSITY {system} {basis} t={t:e} natoms={} nsh={nsh} pairs={total} kept_mean={mean:.1} kept_frac={:.4} kept_min={lo} kept_max={hi} coarse_only_kept_mean={cmean:.1} coarse_only_frac={:.4} batch256_kept_mean={bmean:.1} batch256_frac={:.4} geometry_only_kept={gsum} geometry_only_frac={:.4} screen_s_per_point={per_point_s:.3e} PSI_before={psi_b} PSI_after={psi_a}",
            mol.atoms.len(),
            mean / total as f64,
            cmean / total as f64,
            bmean / total as f64,
            gsum as f64 / total as f64
        );
    }

    if time_abuild {
        let kern = Md3c1e::new(&prep).expect("md3c1e");
        let mut scr = kern.scratch();
        let chunk = &pts[..pts.len().min(256)];
        let psi_b = psi_full_avg10();
        let t2 = Instant::now();
        let mut checksum = 0.0_f64;
        kern.for_each_pair(chunk, None, CosxScreen::none(), &mut scr, |_s1, _s2, blk| {
            checksum += blk[0] + blk[blk.len() - 1];
        })
        .expect("md3c1e");
        let abuild = t2.elapsed().as_secs_f64() / chunk.len() as f64;
        let psi_a = psi_full_avg10();
        // Screen per point on the same chunk, for a like-for-like ratio.
        let t3 = Instant::now();
        let mut kept = 0usize;
        for r in chunk {
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    if bounds.exceeds(s1, s2, r, 1e-7) {
                        kept += 1;
                    }
                }
            }
        }
        let scr_pp = t3.elapsed().as_secs_f64() / chunk.len() as f64;
        println!(
            "COST {system} {basis} abuild_s_per_point={abuild:.3e} (B={}, unscreened, checksum {checksum:.3e}) screen_s_per_point={scr_pp:.3e} ratio_screen/abuild={:.4} kept@1e-7={kept} PSI_before={psi_b} PSI_after={psi_a}",
            chunk.len(),
            scr_pp / abuild
        );
    }
}
