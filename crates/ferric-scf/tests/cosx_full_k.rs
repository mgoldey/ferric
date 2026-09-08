//! FULL COSX K build timing (ignored; run explicitly, one process per cell).
//!
//! Measures what the L-axis harness (`cosx_l_axis.rs`) deliberately did NOT:
//! the whole `CosxK::build` on a converged density — AO-on-grid evaluation,
//! per-point A-build, per-point GEMVs, block GEMMs, overlap-fit finalization —
//! and reports the split, so the working estimate "A-build is ~95% of the full
//! K" is confirmed or refuted by a measurement rather than a proxy.
//!
//! Protocol (mirrors cosx_l_axis.rs): every timed segment runs in a fresh
//! `COSX_FK_THREADS`-thread rayon pool (default 1) with OPENBLAS_NUM_THREADS=1
//! set by the caller; cpu-s and wall-s are both printed (must agree to ~1% on
//! one thread); `/proc/pressure/memory` full avg10 is printed before and after
//! and any segment with a nonzero value is to be discarded.
//!
//! Env:
//!   COSX_FK_SYSTEM      xyz stem under testdata/molecules (default alkane_4)
//!   COSX_FK_BASIS       bundled basis (default def2-qzvp)
//!   COSX_FK_DENSITY_OUT run the DF-JK SCF (default pool) and save (D, C_occ) here, then stop
//!   COSX_FK_DENSITY_IN  load (D, C_occ) saved by an earlier process
//!   COSX_FK_THREADS     rayon threads for timed segments (default 1)
//!   COSX_FK_FIT         "0" disables the overlap fit (default on)
//!   COSX_FK_LINK        "1" also times LinK builds and prints max|K_cosx - K_link|
//!   COSX_FK_LINK_BUILDS number of LinK builds (default 2: cold = pool + pairs + build, then warm)
//!   COSX_FK_BUILDS      number of COSX builds (default 1; the first also factorizes S_num)
//!   COSX_FK_SCREEN      density-driven screen threshold: a number, or "none" (unscreened);
//!                       absent = the library default (`COSX_DEFAULT_SCREEN_THRESH`)

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{CosxConfig, CosxK};
use ferric_scf::fock::KBuilder;
use ferric_scf::link_k::LinkK;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use std::time::Instant;

const JKFIT: &str = "def2-universal-jkfit";
const LINK_THRESH: f64 = 1e-12;

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

#[test]
#[ignore = "full COSX K timing; run explicitly with --ignored --nocapture"]
fn cosx_full_k_cell() {
    let system = std::env::var("COSX_FK_SYSTEM").unwrap_or_else(|_| "alkane_4".into());
    let basis = std::env::var("COSX_FK_BASIS").unwrap_or_else(|_| "def2-qzvp".into());
    let threads: usize = env_num("COSX_FK_THREADS", 1);
    let builds: usize = env_num("COSX_FK_BUILDS", 1);
    let fit = !matches!(std::env::var("COSX_FK_FIT").ok().as_deref(), Some("0"));
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(None);

    let mol = Molecule::load_xyz(&testdata(&format!("testdata/molecules/{system}.xyz"))).expect("xyz");
    let bs = bundled(&basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let nbf = prep.nbasis();
    let nocc = (mol.nelec() / 2) as usize;
    println!(
        "\n=== COSX full-K cell: {system} / {basis}  natoms={} nbf={nbf} nocc={nocc} L_max={}  budget={:.2} GB  PSI now={} ===",
        mol.atoms.len(),
        prep.max_l(),
        budget_bytes as f64 / 1e9,
        psi_full_avg10()
    );

    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let (d, _c_occ): (Array2<f64>, Array2<f64>) = if let Ok(path) = std::env::var("COSX_FK_DENSITY_IN") {
        let (d, c) = load_density(&path, nbf, nocc);
        println!("density loaded from {path} (provenance: an earlier SCF process of this harness; see its log for mode/convergence)");
        (d, c)
    } else {
        let schwarz = SchwarzBounds::compute(op, &prep).expect("Schwarz bounds");
        // COSX_FK_SCF: "dfjk" (default; two budget-bounded 3-index tensors —
        // at nbf >= ~400 they no longer fit a 3.6 GB cgroup and spill/thrash)
        // or "link" (LinK-K + direct J, no 3-index tensors; density_conv from
        // COSX_FK_SCF_DCONV, default 1e-5 — the SCF is not the timed quantity).
        let scf_mode = std::env::var("COSX_FK_SCF").unwrap_or_else(|_| "dfjk".into());
        let scf_cfg = match scf_mode.as_str() {
            "dfjk" => RhfConfig { df_j_aux: Some(JKFIT.to_string()), df_k_aux: Some(JKFIT.to_string()), ..RhfConfig::default() },
            // Window-chaining knobs (an SCF that does not fit one foreground
            // window): COSX_FK_SCF_MAXITER caps the iterations, the density is
            // saved UNCONVERGED (flagged) when COSX_FK_SCF_SAVE_UNCONVERGED=1,
            // and COSX_FK_SCF_RESTART_IN seeds the next window from it.
            // "direct": the plain direct J+K SCF (incremental Fock), no
            // 3-index tensors — the exact, low-memory route for nbf >= ~400.
            "link" | "cosx" | "direct" => RhfConfig {
                k_builder: (scf_mode != "direct").then(|| scf_mode.clone()),
                density_conv: env_num("COSX_FK_SCF_DCONV", 1e-5),
                max_iter: env_num("COSX_FK_SCF_MAXITER", RhfConfig::default().max_iter),
                verbose: true,
                init_guess_density: std::env::var("COSX_FK_SCF_RESTART_IN").ok().map(|p| {
                    let (d0, _) = load_density(&p, nbf, nocc);
                    println!("SCF restart density loaded from {p}");
                    d0
                }),
                ..RhfConfig::default()
            },
            other => panic!("COSX_FK_SCF = {other:?}: expected \"dfjk\", \"direct\", \"link\" or \"cosx\""),
        };
        println!("SCF pool: {} rayon threads (not a timing)", rayon::current_num_threads());
        let t0 = Instant::now();
        let res = solve_rhf(&ctx, &mol, &prep, op, &schwarz, &scf_cfg).expect("RHF");
        println!(
            "{scf_mode} RHF: E={:.8} converged={} iters={} in {:.1} s (default pool, not a timing)",
            res.energy, res.converged, res.iterations, t0.elapsed().as_secs_f64()
        );
        let d = res.density_total.clone();
        let c_occ = res.mos_r().slice(ndarray::s![.., ..nocc]).to_owned();
        if let Ok(path) = std::env::var("COSX_FK_DENSITY_OUT") {
            if res.converged || env_flag("COSX_FK_SCF_SAVE_UNCONVERGED") {
                save_density(&path, &d, &c_occ);
                println!("density saved to {path} (converged={}); stopping (run again with COSX_FK_DENSITY_IN)", res.converged);
                return;
            }
        }
        assert!(res.converged, "refusing to time K builds on an unconverged density");
        (d, c_occ)
    };

    // ---------- full COSX K ----------
    let screen_thresh = match std::env::var("COSX_FK_SCREEN").ok().as_deref() {
        None => CosxConfig::default().screen_thresh,
        Some("none") => None,
        Some(s) => Some(s.parse::<f64>().expect("COSX_FK_SCREEN: number or 'none'")),
    };
    // COSX_FK_HALF: "sparse" (default) or "dense" — the block half transforms.
    let half = match std::env::var("COSX_FK_HALF").ok().as_deref() {
        None | Some("sparse") => ferric_scf::cosx_k::CosxHalfTransform::SPARSE_DEFAULT,
        Some("dense") => ferric_scf::cosx_k::CosxHalfTransform::Dense,
        Some(other) => panic!("COSX_FK_HALF = {other:?}: expected \"sparse\" or \"dense\""),
    };
    let cfg = CosxConfig { overlap_fit: fit, screen_thresh, half_transform: half, ..CosxConfig::default() };
    let mut cosx = CosxK::new(&ctx, &mol, &prep, cfg, budget_bytes).expect("CosxK::new");
    let npts = cosx.npts();
    println!("COSX grid (50,110): {npts} points; overlap_fit={fit}; screen_thresh={screen_thresh:?}");
    let mut k_cosx = Array2::<f64>::zeros((nbf, nbf));
    // COSX_FK_BUILDS=0: skip the COSX build (LinK-only process); K_cosx then
    // comes from COSX_FK_K_IN if given.
    for b in 0..builds {
        let label = if b == 0 { "COSX K build #1 (engine pool + S_num factor + build)" } else { "COSX K build (warm)" };
        let (wall, cpu, _) = timed(label, threads, || cosx.build(&d, &mut k_cosx).expect("COSX build"));
        let t = *cosx.last_timings();
        let acc = t.ao_eval_s + t.a_build_s + t.contract_s + t.blas_s + t.fit_s;
        println!(
            "COSX split (s): total {:.3} | ao_eval {:.3} | A-build {:.3} ({:.1}% of total) | per-point GEMV {:.3} | block GEMMs {:.3} | fit {:.3} | accounted {:.3} ({:.1}%) | pairs kept {}/{} (dd {:.4}, geom-only {:.4}) | wall {wall:.3} cpu {cpu:.2}",
            t.total_s,
            t.ao_eval_s,
            t.a_build_s,
            100.0 * t.a_build_s / t.total_s,
            t.contract_s,
            t.blas_s,
            t.fit_s,
            acc,
            100.0 * acc / t.total_s,
            t.pairs_kept,
            t.pairs_total,
            t.pairs_kept as f64 / t.pairs_total as f64,
            t.pairs_kept_geom as f64 / t.pairs_total as f64
        );
        println!(
            "COSX sparse counters: half {:.3} s | ktilde {:.3} s | snum {:.3} s | gather {:.3} s | |A|/nbf {:.4} (min {:.4} max {:.4}) | |Λ|/nbf {:.4} | |B|/nbf {:.4} | blocks {}",
            t.half_s, t.ktilde_s, t.snum_s, t.gather_s,
            t.active_ao_frac, t.active_ao_frac_min, t.active_ao_frac_max, t.lambda_frac, t.out_ao_frac, t.n_blocks
        );
        println!(
            "COSX per-point A-build {:.4e} s/pt ({:.3e} s/pt/nbf^2); K/nbf^2 {:.3e} s",
            t.a_build_s / npts as f64,
            t.a_build_s / npts as f64 / (nbf * nbf) as f64,
            t.total_s / (nbf * nbf) as f64
        );
    }

    if let Ok(path) = std::env::var("COSX_FK_K_OUT") {
        if builds > 0 {
            save_density(&path, &k_cosx, &Array2::<f64>::zeros((nbf, 0)));
            println!("K_cosx saved to {path}");
        }
    }
    if let Ok(path) = std::env::var("COSX_FK_K_IN") {
        let (k, _) = load_density(&path, nbf, 0);
        k_cosx = k;
        println!("K_cosx loaded from {path}");
    }
    let cosx_total_s = cosx.last_timings().total_s;

    // ---------- optional LinK denominator + accuracy ----------
    if env_flag("COSX_FK_LINK") {
        let schwarz = SchwarzBounds::compute(op, &prep).expect("Schwarz bounds");
        let mut link = LinkK::new(&ctx, &prep, &schwarz, op, LINK_THRESH, budget_bytes);
        let mut k_link = Array2::<f64>::zeros((nbf, nbf));
        let link_builds: usize = env_num("COSX_FK_LINK_BUILDS", 2);
        let mut wall = f64::NAN;
        let mut cpu = f64::NAN;
        for b in 0..link_builds.max(1) {
            let label = if b == 0 { "LinK K build (cold: pool + pairs + build)" } else { "LinK K build (warm: update_density + build)" };
            (wall, cpu, _) = timed(label, threads, || {
                link.update_density(&d);
                link.build(&d, &mut k_link).expect("LinK build")
            });
        }
        let dev = (&k_cosx - &k_link).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v));
        let kmax = k_link.mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v));
        let ratio = if cosx_total_s > 0.0 { format!("{:.3}", cosx_total_s / wall) } else { "n/a (COSX not built in this process)".into() };
        println!(
            "LinK K (last build): wall {wall:.3} s cpu {cpu:.2} s; COSX/LinK = {ratio}; max|K_cosx - K_link| = {dev:.3e} (||K||max {kmax:.3e})"
        );
    }
}
