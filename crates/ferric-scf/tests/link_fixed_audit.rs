//! Fixed-LinK audit harness (ignored; run explicitly, one process per cell).
//!
//! PR #50 fixed three pair-list defects in `LinkK` that made the old kernel
//! silently SKIP quartets — so every LinK timing quoted in this repo's COSX
//! work was against a too-fast kernel. This harness re-measures, on ONE
//! converged density from a DIRECT SCF, warm K builds by
//!
//! * `DirectK` — K only, Schwarz + global max|D| screen: the like-for-like
//!   comparator (identical per-quartet screen to LinK, no pair lists);
//! * `build_jk` — ferric's DEFAULT builder (J AND K in one sweep, Häser-
//!   Ahlrichs shell-pair density max): an upper bound on a direct K alone;
//! * `LinkK` — the fixed kernel, with its quartet count and the kept
//!   significant-pair / density-pair totals of the FIXED lists and of the
//!   PRE-fix criteria recomputed on the same density;
//! * `CosxK` — production config (md3c1e, (50,110), fit on, density screen
//!   at the default threshold).
//!
//! and prints the quartet counts side by side (the over-inclusion check: a
//! correct LinK evaluates a SUBSET of `DirectK`'s quartets).
//!
//! Protocol (mirrors cosx_full_k.rs): every timed segment runs in a fresh
//! 1-thread rayon pool (`LFA_THREADS`) with OPENBLAS_NUM_THREADS=1 set by the
//! caller; cpu-s and wall-s are both printed; `/proc/pressure/memory`
//! `full avg10` is printed before and after and any nonzero segment is to be
//! discarded. Pre-registration: scripts/queue/out/link_fixed_prereg.md.
//!
//! Env:
//!   LFA_SYSTEM        xyz stem under testdata/molecules (default alkane_4)
//!   LFA_BASIS         bundled basis (default def2-svp)
//!   LFA_THRESH        integral threshold for DirectK / build_jk / LinK (default 1e-12)
//!   LFA_DENSITY_OUT   run the direct SCF (ambient pool) and save D here, then stop
//!   LFA_DENSITY_IN    load D saved by an earlier process
//!   LFA_SCF_MAXITER / LFA_SCF_RESTART_IN / LFA_SCF_SAVE_UNCONVERGED=1
//!                     window-chaining for an SCF that does not fit one window
//!   LFA_BUILDERS      comma list of direct_k, direct_jk, link, cosx (default "direct_k,link")
//!   LFA_BUILDS        builds per builder (default 2: cold, then warm; warm is reported)
//!   LFA_K_OUT         save the DirectK K here (reference for other processes)
//!   LFA_K_IN          load a reference K to report max|K_builder - K_ref|
//!   LFA_THREADS       rayon threads for timed segments (default 1)

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{CosxConfig, CosxK};
use ferric_scf::direct_k::DirectK;
use ferric_scf::fock::KBuilder;
use ferric_scf::link_k::LinkK;
use ferric_scf::pairs::{DensityPairs, SignificantPairs};
use ferric_scf::rhf::{build_jk, solve_rhf, RhfConfig};
use ferric_scf::screening::{Bound, SchwarzBounds};
use ndarray::Array2;
use std::time::Instant;

fn testdata(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}

fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).ok().as_deref(), Some("1") | Some("true") | Some("yes"))
}

fn env_num<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn psi_full_avg10() -> String {
    std::fs::read_to_string("/proc/pressure/memory")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("full")).and_then(|l| {
                l.split_whitespace().find_map(|t| t.strip_prefix("avg10=").map(str::to_string))
            })
        })
        .unwrap_or_else(|| "n/a".into())
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

/// (wall s, cpu s, psi before, psi after, value)
fn timed<T>(label: &str, threads: usize, f: impl FnOnce() -> T + Send) -> (f64, f64, String, String, T)
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
    (secs, cpu, before, after, v)
}

fn save_matrix(path: &str, m: &Array2<f64>) {
    let mut bytes = Vec::with_capacity(m.len() * 8);
    for v in m.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write matrix file");
}

fn load_matrix(path: &str, nbf: usize) -> Array2<f64> {
    let bytes = std::fs::read(path).expect("read matrix file");
    assert_eq!(bytes.len(), nbf * nbf * 8, "matrix file length does not match nbf={nbf}");
    let vals: Vec<f64> = bytes.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect();
    Array2::from_shape_vec((nbf, nbf), vals).expect("matrix shape")
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// PRE-fix pair-list criteria (origin/main a131b69f `pairs.rs`), recomputed on
/// the same density so the kept fractions can be compared like for like:
/// old sp `estimate(i,j,i,j) > thresh` (i.e. `Q² > thresh`); old dp
/// `max|D[j,σ]| · Q(j,σ) > thresh`. Returns (sp_total, dp_total).
fn prefix_pair_totals(bound: &dyn Bound, prep: &PreparedBasis, d: &Array2<f64>, thresh: f64) -> (usize, usize) {
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut sp = 0usize;
    let mut dp = 0usize;
    for i in 0..nsh {
        for j in 0..nsh {
            let est = bound.estimate(i, j, i, j);
            if est > thresh {
                sp += 1;
            }
            let mut dmax = 0.0f64;
            for mu in offs[i]..offs[i] + dims[i] {
                for nu in offs[j]..offs[j] + dims[j] {
                    dmax = dmax.max(d[(mu, nu)].abs());
                }
            }
            if dmax * est.sqrt() > thresh {
                dp += 1;
            }
        }
    }
    (sp, dp)
}

#[test]
#[ignore = "fixed-LinK audit timing; run explicitly with --ignored --nocapture"]
fn link_fixed_audit_cell() {
    let system = std::env::var("LFA_SYSTEM").unwrap_or_else(|_| "alkane_4".into());
    let basis = std::env::var("LFA_BASIS").unwrap_or_else(|_| "def2-svp".into());
    let thresh: f64 = env_num("LFA_THRESH", 1e-12);
    let threads: usize = env_num("LFA_THREADS", 1);
    let builds: usize = env_num("LFA_BUILDS", 2).max(1);
    let builders: Vec<String> = std::env::var("LFA_BUILDERS")
        .unwrap_or_else(|_| "direct_k,link".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(None);

    let mol = Molecule::load_xyz(&testdata(&format!("testdata/molecules/{system}.xyz"))).expect("xyz");
    let bs = bundled(&basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let nbf = prep.nbasis();
    let nsh = prep.nshells();
    println!(
        "\n=== fixed-LinK audit cell: {system} / {basis}  natoms={} nsh={nsh} nbf={nbf} L_max={} thresh={thresh:.0e}  budget={:.2} GB  PSI now={} ===",
        mol.atoms.len(),
        prep.max_l(),
        budget_bytes as f64 / 1e9,
        psi_full_avg10()
    );

    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let schwarz = SchwarzBounds::compute(op, &prep).expect("Schwarz bounds");

    // ---------- density: DIRECT SCF (ambient pool; not a timing) ----------
    let d: Array2<f64> = if let Ok(path) = std::env::var("LFA_DENSITY_IN") {
        let d = load_matrix(&path, nbf);
        println!("density loaded from {path} (provenance: an earlier direct-SCF process of this harness)");
        d
    } else {
        let scf_cfg = RhfConfig {
            max_iter: env_num("LFA_SCF_MAXITER", RhfConfig::default().max_iter),
            verbose: true,
            init_guess_density: std::env::var("LFA_SCF_RESTART_IN").ok().map(|p| {
                let d0 = load_matrix(&p, nbf);
                println!("SCF restart density loaded from {p}");
                d0
            }),
            ..RhfConfig::default()
        };
        println!("direct SCF pool: {} rayon threads (not a timing)", rayon::current_num_threads());
        let t0 = Instant::now();
        let res = solve_rhf(&ctx, &mol, &prep, op, &schwarz, &scf_cfg).expect("direct RHF");
        println!(
            "direct RHF: E={:.10} converged={} iters={} in {:.1} s (ambient pool, not a timing)",
            res.energy,
            res.converged,
            res.iterations,
            t0.elapsed().as_secs_f64()
        );
        let d = res.density_total.clone();
        if let Ok(path) = std::env::var("LFA_DENSITY_OUT") {
            if res.converged || env_flag("LFA_SCF_SAVE_UNCONVERGED") {
                save_matrix(&path, &d);
                println!("density saved to {path} (converged={}); stopping (run again with LFA_DENSITY_IN)", res.converged);
                return;
            }
        }
        assert!(res.converged, "refusing to time K builds on an unconverged density");
        d
    };

    let mut k_ref: Option<Array2<f64>> = std::env::var("LFA_K_IN").ok().map(|p| {
        let k = load_matrix(&p, nbf);
        println!("reference K loaded from {p}");
        k
    });

    for builder in &builders {
        match builder.as_str() {
            "direct_k" => {
                let mut dk = DirectK::new(&ctx, &prep, &schwarz, thresh, budget_bytes);
                let mut k = Array2::<f64>::zeros((nbf, nbf));
                let mut last = (f64::NAN, f64::NAN, String::new(), String::new(), 0usize);
                for b in 0..builds {
                    let label = if b == 0 { "DirectK build #1 (cold: pool + build)" } else { "DirectK build (warm)" };
                    k.fill(0.0);
                    last = timed(label, threads, || dk.build(&d, &mut k).expect("DirectK build"));
                }
                let (wall, cpu, pb, pa, n) = last;
                let dev = k_ref.as_ref().map(|r| format!("{:.3e}", max_abs_diff(r, &k))).unwrap_or_else(|| "n/a".into());
                println!("ROW | {system} | {basis} | direct_k | nbf={nbf} | wall={wall:.3} | cpu={cpu:.2} | quartets={n} | max|K-Kref|={dev} | PSI {pb}/{pa}");
                if let Ok(path) = std::env::var("LFA_K_OUT") {
                    save_matrix(&path, &k);
                    println!("DirectK K saved to {path}");
                }
                if k_ref.is_none() {
                    k_ref = Some(k);
                }
            }
            "direct_jk" => {
                let mut j = Array2::<f64>::zeros((nbf, nbf));
                let mut k = Array2::<f64>::zeros((nbf, nbf));
                let mut last = (f64::NAN, f64::NAN, String::new(), String::new(), 0usize);
                for b in 0..builds {
                    let label = if b == 0 { "build_jk #1 (J+K, default builder; constructs its own pool)" } else { "build_jk (J+K, default builder)" };
                    j.fill(0.0);
                    k.fill(0.0);
                    last = timed(label, threads, || build_jk(&ctx, &prep, &schwarz, thresh, &d, &mut j, &mut k).expect("build_jk"));
                }
                let (wall, cpu, pb, pa, n) = last;
                let dev = k_ref.as_ref().map(|r| format!("{:.3e}", max_abs_diff(r, &k))).unwrap_or_else(|| "n/a".into());
                println!("ROW | {system} | {basis} | direct_jk | nbf={nbf} | wall={wall:.3} | cpu={cpu:.2} | quartets={n} | max|K-Kref|={dev} | PSI {pb}/{pa}");
            }
            "link" => {
                let mut link = LinkK::new(&ctx, &prep, &schwarz, op, thresh, budget_bytes);
                let mut k = Array2::<f64>::zeros((nbf, nbf));
                let mut last = (f64::NAN, f64::NAN, String::new(), String::new(), 0usize);
                for b in 0..builds {
                    let label = if b == 0 { "LinK build #1 (cold: pool + pairs + build)" } else { "LinK build (warm: update_density + build)" };
                    k.fill(0.0);
                    last = timed(label, threads, || {
                        link.update_density(&d);
                        link.build(&d, &mut k).expect("LinK build")
                    });
                }
                let (wall, cpu, pb, pa, n) = last;
                // Pair-list totals: fixed criteria (the lists LinK just used) and
                // the pre-fix criteria on the same density.
                let sp = SignificantPairs::build(&schwarz, nsh, thresh).total_pairs();
                let dp = DensityPairs::build(&d, &schwarz, &prep, thresh).total_pairs();
                let (sp_old, dp_old) = prefix_pair_totals(&schwarz, &prep, &d, thresh);
                let sq = nsh * nsh;
                let dev = k_ref.as_ref().map(|r| format!("{:.3e}", max_abs_diff(r, &k))).unwrap_or_else(|| "n/a".into());
                println!(
                    "ROW | {system} | {basis} | link | nbf={nbf} | wall={wall:.3} | cpu={cpu:.2} | quartets={n} | max|K-Kref|={dev} | PSI {pb}/{pa} \
                     | sp fixed {sp}/{sq} ({:.4}) prefix {sp_old}/{sq} ({:.4}) | dp fixed {dp}/{sq} ({:.4}) prefix {dp_old}/{sq} ({:.4})",
                    sp as f64 / sq as f64,
                    sp_old as f64 / sq as f64,
                    dp as f64 / sq as f64,
                    dp_old as f64 / sq as f64
                );
            }
            "cosx" => {
                let cfg = CosxConfig::default();
                let mut cosx = CosxK::new(&ctx, &mol, &prep, cfg, budget_bytes).expect("CosxK::new");
                let npts = cosx.npts();
                let mut k = Array2::<f64>::zeros((nbf, nbf));
                let mut last = (f64::NAN, f64::NAN, String::new(), String::new(), 0usize);
                for b in 0..builds {
                    let label = if b == 0 { "COSX build #1 (pool + S_num factor + build)" } else { "COSX build (warm)" };
                    k.fill(0.0);
                    last = timed(label, threads, || cosx.build(&d, &mut k).expect("COSX build"));
                }
                let (wall, cpu, pb, pa, _) = last;
                let t = *cosx.last_timings();
                let dev = k_ref.as_ref().map(|r| format!("{:.3e}", max_abs_diff(r, &k))).unwrap_or_else(|| "n/a".into());
                println!(
                    "ROW | {system} | {basis} | cosx | nbf={nbf} | wall={wall:.3} | cpu={cpu:.2} | total={:.3} A-build={:.3} GEMMs={:.3} | npts={npts} pairs kept {}/{} ({:.4}) | max|K-Kref|={dev} | PSI {pb}/{pa}",
                    t.total_s,
                    t.a_build_s,
                    t.blas_s,
                    t.pairs_kept,
                    t.pairs_total,
                    t.pairs_kept as f64 / t.pairs_total as f64
                );
            }
            other => panic!("LFA_BUILDERS: unknown builder {other:?} (direct_k, direct_jk, link, cosx)"),
        }
    }
    println!("PSI end={}", psi_full_avg10());
}
