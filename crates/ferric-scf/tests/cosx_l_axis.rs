//! COSX L-axis measurement: A-build vs analytic (LinK) K vs DF-K as a function
//! of basis-set angular momentum (def2-SVP L=2, def2-TZVP L=3, def2-QZVP L=4).
//!
//! Every earlier COSX deficit in this repo was taken at DZ/TZ. The literature's
//! case for seminumerical exchange lives at HIGH L: an analytic quartet's cost
//! explodes with L, a 3c1e block over a shell pair does not. This harness
//! measures that axis directly. Pre-registration: scripts/queue/out/l_axis_prereg.md
//! (committed before any number here was taken).
//!
//! # Protocol (fixed before measuring)
//!
//! * One process per (system, basis) cell; selected via env `COSX_L_SYSTEM`
//!   (`alkane_4` | `alkane_8`) and `COSX_L_BASIS` (`def2-svp` | ...).
//! * A-build: `>= 500` grid points drawn uniformly at random without
//!   replacement (fixed-seed LCG) from the full (50,110) Becke grid; per-point
//!   cost measured serially with ONE reused engine, unscreened and with the
//!   Stage 2 screen at 1e-7. Per-K-build cost = per-point x full point count
//!   (measured x known count, NOT an nbf extrapolation).
//! * Analytic K: `LinkK` with Schwarz bound at thresh 1e-12 (production
//!   `integral_thresh`), on the converged RHF density. Warm build: the engine
//!   pool exists, `update_density` + `build` are inside the timed region (an
//!   SCF iteration pays both).
//! * DF-K: `DfK` with def2-universal-jkfit; warm `build` (density path) and
//!   warm `build_from_occ` (the path the SCF actually uses once MOs exist).
//!   The resident dressed tensor is naux*nbf^2*8 B; when that exceeds
//!   `COSX_L_DFK_MAX_GB` (default 2.5) the SIZE is the finding and DF-K is
//!   skipped.
//! * Every timed segment runs inside a fresh rayon pool of `COSX_L_LINK_THREADS`
//!   threads (default 1) so A and K are on equal footing (both parallelise
//!   trivially in production). Wall AND process CPU seconds are printed for
//!   each segment (cpu ~= wall on one thread confirms no contention; on more
//!   threads CPU-seconds is the thread-invariant figure). PSI `full avg10`
//!   from /proc/pressure/memory is printed before and after each.
//! * The density is converged with DF-JK on the default pool; its provenance
//!   cancels out of every ratio because all builders contract the same D.
//! * Window-splitting knobs (the box rule is <= 8 min per foreground run):
//!   `COSX_L_SKIP_SCF` (A-build only), `COSX_L_SKIP_LINK` (no analytic K),
//!   `COSX_L_LINK_BUILDS` (default 2: cold + warm), `COSX_L_SKIP_DFK`.
//!
//! Run with `--ignored --nocapture`; this is a measurement, not a CI gate.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::cosx_a::{a_matrix_at_point_with, CosxScreen, PairBounds};
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use ferric_scf::df_k::DfK;
use ferric_scf::fock::KBuilder;
use ferric_scf::link_k::LinkK;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use std::time::Instant;

const SCREEN_THRESH: f64 = 1e-7; // Stage 2 production screen, unchanged
const LINK_THRESH: f64 = 1e-12; // RhfConfig::default().integral_thresh
const JKFIT: &str = "def2-universal-jkfit";
const N_SAMPLE: usize = 500;
const SEED: u64 = 20260907;

fn testdata(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok()
}

fn env_num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// `full avg10` from /proc/pressure/memory, as a string ("n/a" if unreadable).
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

/// Process CPU seconds (utime + stime) from /proc/self/stat; Linux userland
/// CLK_TCK is 100 on every supported target here.
fn cpu_seconds() -> f64 {
    std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| {
            // Field 2 (comm) may contain spaces; split after the closing paren.
            let rest = s.rsplit(')').next()?;
            let f: Vec<&str> = rest.split_whitespace().collect();
            // After ')' the fields start at index 0 == state (field 3), so
            // utime (field 14) is index 11 and stime (field 15) is index 12.
            let ut: f64 = f.get(11)?.parse().ok()?;
            let st: f64 = f.get(12)?.parse().ok()?;
            Some((ut + st) / 100.0)
        })
        .unwrap_or(f64::NAN)
}

/// Time `f` inside a fresh `threads`-thread rayon pool; returns (wall s, cpu s, value).
fn timed<T>(label: &str, threads: usize, f: impl FnOnce() -> T + Send) -> (f64, f64, T)
where
    T: Send,
{
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("rayon pool");
    let before = psi_full_avg10();
    let c0 = cpu_seconds();
    let t0 = Instant::now();
    let v = pool.install(f);
    let secs = t0.elapsed().as_secs_f64();
    let cpu = cpu_seconds() - c0;
    let after = psi_full_avg10();
    println!("  [timed:{label}] wall {secs:.3} s  cpu {cpu:.2} s  threads={threads}  PSI full avg10 before={before} after={after}");
    (secs, cpu, v)
}

/// Fixed-seed LCG (Knuth MMIX constants); we only need a reproducible shuffle.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

/// Uniform sample of `k` distinct indices from `0..n` (partial Fisher-Yates).
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

struct ACost {
    secs_per_point: f64,
    kept_frac: f64,
}

fn measure_a(prep: &PreparedBasis, pts: &[[f64; 3]], screen: CosxScreen, bounds: Option<&PairBounds>) -> ACost {
    let label = if screen.is_vacuous() { "A-build unscreened" } else { "A-build screened" };
    let (secs, _cpu, (kept, total)) = timed(label, 1, || {
        let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14).expect("nuclear engine");
        let mut kept = 0usize;
        let mut total = 0usize;
        for r in pts {
            let p = a_matrix_at_point_with(&mut eng, prep, r, bounds, screen).expect("A build");
            kept += p.pairs_kept;
            total += p.pairs_total;
        }
        (kept, total)
    });
    ACost { secs_per_point: secs / pts.len() as f64, kept_frac: kept as f64 / total as f64 }
}

/// Raw little-endian f64 dump: `D` (nbf*nbf) followed by `C_occ` (nbf*nocc).
/// Shapes are re-derived from (nbf, nocc) on load and cross-checked by length.
fn save_density(path: &str, d: &Array2<f64>, c_occ: &Array2<f64>) {
    let mut bytes = Vec::with_capacity((d.len() + c_occ.len()) * 8);
    for v in d.iter().chain(c_occ.iter()) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write density file");
}

fn load_density(path: &str, nbf: usize, nocc: usize) -> (Array2<f64>, Array2<f64>) {
    let bytes = std::fs::read(path).expect("read density file");
    let n_d = nbf * nbf;
    let n_c = nbf * nocc;
    assert_eq!(bytes.len(), (n_d + n_c) * 8, "density file length does not match nbf={nbf}, nocc={nocc}");
    let vals: Vec<f64> = bytes.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect();
    let d = Array2::from_shape_vec((nbf, nbf), vals[..n_d].to_vec()).expect("D shape");
    let c = Array2::from_shape_vec((nbf, nocc), vals[n_d..].to_vec()).expect("C_occ shape");
    (d, c)
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.3}")).unwrap_or_else(|| "skip".into())
}
fn ratio_opt(num: f64, den: Option<f64>) -> String {
    den.map(|x| format!("{:.2}", num / x)).unwrap_or_else(|| "skip".into())
}

#[test]
#[ignore = "L-axis measurement; run explicitly with --ignored --nocapture"]
fn cosx_l_axis_cell() {
    let system = std::env::var("COSX_L_SYSTEM").unwrap_or_else(|_| "alkane_4".into());
    let basis = std::env::var("COSX_L_BASIS").unwrap_or_else(|_| "def2-svp".into());
    let dfk_max_gb: f64 = env_num("COSX_L_DFK_MAX_GB", 2.5);
    let skip_scf = env_flag("COSX_L_SKIP_SCF");
    let skip_link = env_flag("COSX_L_SKIP_LINK");
    let skip_dfk = env_flag("COSX_L_SKIP_DFK");
    let link_threads: usize = env_num("COSX_L_LINK_THREADS", 1);
    let link_builds: usize = env_num("COSX_L_LINK_BUILDS", 2);
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(None);

    let mol = Molecule::load_xyz(&testdata(&format!("testdata/molecules/{system}.xyz"))).expect("xyz");
    let bs = bundled(&basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let nbf = prep.nbasis();
    let nsh = prep.nshells();
    let lmax = prep.max_l();
    let aux_bs = bundled(JKFIT).expect("jkfit");
    let aux = PreparedBasis::new(&mol, &aux_bs).expect("aux prep");
    let naux = aux.nbasis();
    let tensor_gb = naux as f64 * (nbf * nbf) as f64 * 8.0 / 1e9;

    println!("\n=== COSX L-axis cell: {system} / {basis}  natoms={} nsh={nsh} nbf={nbf} L_max={lmax}  \
              jkfit naux={naux} dressed 3-index = {tensor_gb:.3} GB  budget={:.2} GB  PSI now={} ===",
        mol.atoms.len(), budget_bytes as f64 / 1e9, psi_full_avg10());

    // ---------- (1) A-build on a random sample of the (50,110) grid ----------
    let cfg = AtomicGridConfig { n_radial: 50, n_angular: 110, ..Default::default() };
    let grid = build_atomic_grid(&mol, &cfg);
    let npts = grid.len();
    let idx = sample_indices(npts, N_SAMPLE, SEED);
    let pts: Vec<[f64; 3]> = idx.iter().map(|&i| grid[i].xyz).collect();
    println!("grid (50,110): {npts} points; sampled {} at random (seed {SEED})", pts.len());

    let bounds = PairBounds::build(&prep).expect("pair bounds");
    let (a_un, a_sc) = if env_flag("COSX_L_SKIP_A") {
        println!("COSX_L_SKIP_A set: A-build not measured in this process (A columns below are NaN).");
        let nan = ACost { secs_per_point: f64::NAN, kept_frac: f64::NAN };
        (nan, ACost { secs_per_point: f64::NAN, kept_frac: f64::NAN })
    } else {
        (
            measure_a(&prep, &pts, CosxScreen::none(), None),
            measure_a(&prep, &pts, CosxScreen::at(SCREEN_THRESH), Some(&bounds)),
        )
    };
    let a_full_un = a_un.secs_per_point * npts as f64;
    let a_full_sc = a_sc.secs_per_point * npts as f64;
    println!(
        "A-build/point: unscreened {:.4e} s, screened(1e-7) {:.4e} s (kept frac {:.4}); \
         per full K build (x{npts} pts): unscreened {:.1} s, screened {:.1} s; \
         A/pt/nbf^2: unscreened {:.3e}, screened {:.3e}",
        a_un.secs_per_point, a_sc.secs_per_point, a_sc.kept_frac, a_full_un, a_full_sc,
        a_un.secs_per_point / (nbf * nbf) as f64, a_sc.secs_per_point / (nbf * nbf) as f64
    );

    if skip_scf {
        println!("COSX_L_SKIP_SCF set: A-build only for this cell.");
        return;
    }

    // ---------- converged density (DF-JK, default pool) ----------
    // `COSX_L_DENSITY_IN=<file>` loads (D, C_occ) saved by an earlier process
    // via `COSX_L_DENSITY_OUT=<file>`, so a cell whose SCF + K build do not fit
    // one 8-minute foreground window can be split across two processes.
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let schwarz = SchwarzBounds::compute(op, &prep).expect("Schwarz bounds");
    let nocc = (mol.nelec() / 2) as usize;
    let (d, c_occ): (Array2<f64>, Array2<f64>) = if let Ok(path) = std::env::var("COSX_L_DENSITY_IN") {
        let (d, c) = load_density(&path, nbf, nocc);
        println!("density loaded from {path} (nbf={nbf}, nocc={nocc}); tr(DS) check skipped, provenance = earlier DF-JK SCF process");
        (d, c)
    } else {
        let scf_cfg = RhfConfig {
            df_j_aux: Some(JKFIT.to_string()),
            df_k_aux: Some(JKFIT.to_string()),
            ..RhfConfig::default()
        };
        let t0 = Instant::now();
        let res = solve_rhf(&ctx, &mol, &prep, op, &schwarz, &scf_cfg).expect("DF-JK RHF");
        println!("DF-JK RHF: E={:.8} converged={} iters={} in {:.1} s (default pool, not a timing)",
            res.energy, res.converged, res.iterations, t0.elapsed().as_secs_f64());
        assert!(res.converged, "refusing to time K builds on an unconverged density");
        let d = res.density_total.clone();
        let c_occ = res.mos_r().slice(ndarray::s![.., ..nocc]).to_owned();
        if let Ok(path) = std::env::var("COSX_L_DENSITY_OUT") {
            save_density(&path, &d, &c_occ);
            println!("density saved to {path}");
        }
        (d, c_occ)
    };
    if env_flag("COSX_L_SCF_ONLY") {
        println!("COSX_L_SCF_ONLY set: stopping after the SCF.");
        return;
    }

    // ---------- (2) analytic K via LinK ----------
    let mut k_link: Option<Array2<f64>> = None;
    let mut link_warm: Option<f64> = None;
    let mut link_cpu: Option<f64> = None;
    if !skip_link {
        let mut k = Array2::<f64>::zeros((nbf, nbf));
        let mut link = LinkK::new(&ctx, &prep, &schwarz, op, LINK_THRESH, budget_bytes);
        for b in 0..link_builds.max(1) {
            let label = if b == 0 { "LinK K build #1 (cold: engine pool + pairs + build)" } else { "LinK K build (warm: update_density + build)" };
            let (wall, cpu, _) = timed(label, link_threads, || {
                link.update_density(&d);
                link.build(&d, &mut k).expect("LinK build")
            });
            link_warm = Some(wall);
            link_cpu = Some(cpu);
        }
        println!("LinK K: last build wall {:.3} s cpu {:.2} s ({link_threads} thr); K/nbf^2 (wall) {:.3e} s",
            link_warm.unwrap(), link_cpu.unwrap(), link_warm.unwrap() / (nbf * nbf) as f64);
        k_link = Some(k);
    } else {
        println!("COSX_L_SKIP_LINK set: analytic K not built in this process.");
    }

    // ---------- (2b) scope check: direct J+K (ferric's DEFAULT k_builder) ----------
    // `build_jk` computes J AND K in one quartet sweep, so its time is an UPPER
    // bound on a direct K alone. If it came out far below LinK, LinK would be
    // the wrong denominator for the ratios above.
    if env_flag("COSX_L_DIRECT_JK") {
        let mut j = Array2::<f64>::zeros((nbf, nbf));
        let mut k = Array2::<f64>::zeros((nbf, nbf));
        let (wall, cpu, _) = timed("direct build_jk (J+K, default builder)", link_threads, || {
            ferric_scf::rhf::build_jk(&ctx, &prep, &schwarz, LINK_THRESH, &d, &mut j, &mut k).expect("build_jk")
        });
        let dev = k_link.as_ref().map(|kl| (kl - &k).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v)));
        println!("direct J+K: wall {wall:.3} s cpu {cpu:.2} s ({link_threads} thr); max|K_link - K_direct| = {}",
            dev.map(|x| format!("{x:.3e}")).unwrap_or_else(|| "n/a".into()));
    }

    // ---------- (3) DF-K, 1 thread ----------
    let mut dfk_dens: Option<f64> = None;
    let mut dfk_occ: Option<f64> = None;
    let mut dfk_setup: Option<f64> = None;
    if skip_dfk {
        println!("COSX_L_SKIP_DFK set: DF-K not built in this process.");
    } else if tensor_gb <= dfk_max_gb {
        let (setup, _, mut dfk) = timed("DF-K setup (3-index build + V^-1/2 dressing)", 1, || {
            DfK::new(op, &prep, &aux, budget_bytes).expect("DfK::new")
        });
        let mut k_dfk = Array2::<f64>::zeros((nbf, nbf));
        let mut k_dfk_occ = Array2::<f64>::zeros((nbf, nbf));
        let _ = timed("DF-K density-path build (warm-up)", 1, || dfk.build(&d, &mut k_dfk).expect("dfk build"));
        let (t_dens, _, _) = timed("DF-K density-path build", 1, || dfk.build(&d, &mut k_dfk).expect("dfk build"));
        let _ = timed("DF-K occ-path build (warm-up)", 1, || {
            dfk.build_from_occ(&c_occ, &mut k_dfk_occ).expect("dfk occ build")
        });
        let (t_occ, _, _) = timed("DF-K occ-path build", 1, || {
            dfk.build_from_occ(&c_occ, &mut k_dfk_occ).expect("dfk occ build")
        });
        // Sanity: DF-K must agree with analytic K to fit accuracy (jkfit ~1e-3 abs on K),
        // and the occ path must equal the density path up to D vs 2*C_occ*C_occ^T
        // (they differ by the final SCF density-convergence residual ~1e-6).
        let dev = k_link.as_ref().map(|kl| (kl - &k_dfk).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v)));
        let dev_occ = (&k_dfk - &(&k_dfk_occ * 2.0)).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v));
        println!("DF-K: setup {setup:.3} s; density-path {t_dens:.3} s; occ-path {t_occ:.3} s; \
                  max|K_link - K_dfk| = {}; max|K_dfk - 2*K_occ| = {dev_occ:.3e}",
            dev.map(|x| format!("{x:.3e}")).unwrap_or_else(|| "n/a".into()));
        dfk_dens = Some(t_dens);
        dfk_occ = Some(t_occ);
        dfk_setup = Some(setup);
    } else {
        println!("DF-K SKIPPED: dressed 3-index tensor {tensor_gb:.2} GB exceeds {dfk_max_gb} GB cap (the size IS the finding)");
    }

    // ---------- summary row ----------
    println!(
        "\nROW | {system} | {basis} | L={lmax} | nbf={nbf} | npts={npts} | A/pt un={:.4e} sc={:.4e} | \
         A/K un={a_full_un:.1}s sc={a_full_sc:.1}s | LinK wall={} cpu={} thr={link_threads} | DF-K setup={} dens={} occ={} tensor={tensor_gb:.3}GB | \
         A/LinK un={}x sc={}x | A/DFK(occ) un={}x sc={}x | A/DFK(dens) un={}x sc={}x | \
         K/nbf^2={} | A/pt/nbf^2 un={:.3e} sc={:.3e} | PSI end={}",
        a_un.secs_per_point, a_sc.secs_per_point,
        fmt_opt(link_warm), fmt_opt(link_cpu),
        fmt_opt(dfk_setup), fmt_opt(dfk_dens), fmt_opt(dfk_occ),
        ratio_opt(a_full_un, link_warm), ratio_opt(a_full_sc, link_warm),
        ratio_opt(a_full_un, dfk_occ), ratio_opt(a_full_sc, dfk_occ),
        ratio_opt(a_full_un, dfk_dens), ratio_opt(a_full_sc, dfk_dens),
        link_warm.map(|w| format!("{:.3e}", w / (nbf * nbf) as f64)).unwrap_or_else(|| "skip".into()),
        a_un.secs_per_point / (nbf * nbf) as f64, a_sc.secs_per_point / (nbf * nbf) as f64,
        psi_full_avg10()
    );
}
