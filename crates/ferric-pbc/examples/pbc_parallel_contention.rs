//! Parallel-contention microbenchmark for the SR lattice-sum loops (FINDINGS
//! "Parallel SR loops — quiet-box speed-up (measured 2026-09-27)": 6 threads
//! gave 2.3x, and CPU per integral call inflated ~2.4x (SR 3-centre) /
//! ~2.9x (hcore SR) against serial). This isolates the integral CALL from
//! ferric's loop, screen and output mutex: every thread runs `--calls`
//! identical calls over a fixed triple list, with NO shared output, at each
//! thread count, and reports per-call wall AND per-thread CPU time.
//!
//! Variants (`--variants`, comma list, default all):
//!
//! * `sr3-pool`   the production layout: RS-GDF `erfc(ω)` 3-centre engines
//!   from `EnginePool::from_fn` (all built back-to-back on ONE thread, so
//!   their heap blocks are packed into one malloc arena), used via
//!   `pool.with`, calling `compute_eri3_shifted` on shifted triples.
//! * `sr3-own`    the same calls, but each worker builds its OWN engine on its
//!   own thread (glibc gives each thread its own arena). `sr3-pool` slower
//!   than `sr3-own` at n > 1 ⇒ false sharing between packed engines.
//! * `sr3-zero-shifted` / `sr3-zero-plain`  own engines, the SAME triples at
//!   all-zero shifts, through `compute_eri3_shifted` (shim copies three
//!   `libint2::Shell`s per call) vs `compute_eri3` (no copy). The integrals
//!   are bitwise equal (checked at setup), so the gap is exactly the per-call
//!   shell-copy cost; if the gap GROWS with threads, the copies contend.
//! * `hcore-pool` the production hcore SR layout: `erfc(ω_h)` engines at
//!   precision `f64::MIN_POSITIVE` against the Gaussian-nucleus site basis.
//! * `ctl-fp`     control, pure register FP (+ exp/ln): no memory, no locks.
//!   If THIS inflates with threads, the box (SMT sibling sharing, turbo bins,
//!   other sessions' load) is the "contended resource", not ferric.
//! * `ctl-alloc`  control, 6 small heap alloc/free per call (the shell-copy
//!   pattern): allocator contention shows here.
//! * `ctl-mem`    control, streams 32 KiB per call from a per-thread
//!   `--mem-kb` buffer (default 1024 KiB > the 256 KiB L2): L3 / DRAM
//!   bandwidth sharing shows here (x6 threads: 6 MiB, inside the 15 MiB L3;
//!   pass `--mem-kb 4096` to exceed it).
//!
//! Output per (variant, threads): wall ns/call (min and median over
//! `--reps`), CPU ns/call (thread CPU clock), CPU/wall (< 1 ⇒ the thread was
//! PREEMPTED — oversubscription; ≈ 1 with inflated CPU ⇒ the core itself ran
//! slower: SMT sibling, lower turbo bin, cache/bandwidth), inflation vs the
//! variant's 1-thread wall ns/call, and aggregate speed-up. Also a bitwise
//! checksum of every block the SR / hcore lists produce — identical before
//! and after a shim change ⇔ that change is bit-identical on these calls.
//!
//! Run (release; single-threaded BLAS as always):
//! ```text
//! cargo build --release -p ferric-pbc --example pbc_parallel_contention
//! OPENBLAS_NUM_THREADS=1 target/release/examples/pbc_parallel_contention \
//!     [--basis cc-pvdz] [--aux cc-pvdz-ri] [--calls 100000] [--reps 3] \
//!     [--threads 1,2,3,6] [--pin none|phys|smt] [--variants all|sr3-pool,...] \
//!     [--mem-kb 1024] [--omega 1.0]
//! ```
//! `--pin phys` pins thread k to a distinct PHYSICAL core (sysfs
//! `thread_siblings_list`; this box: CPUs 0-5, siblings k and k+6); `--pin
//! smt` packs threads onto sibling pairs (0,6,1,7,...) to show what SMT
//! co-scheduling alone costs. `none` is what production does.
//! Allocator experiments via env (no rebuild): `MALLOC_ARENA_MAX=1` (forces
//! one shared arena: makes any allocator contention WORSE, a positive
//! control), `MALLOC_ARENA_MAX=12`, `GLIBC_TUNABLES=glibc.malloc.tcache_count=0`
//! (disables the per-thread cache: again a positive control), or
//! `LD_PRELOAD=<libjemalloc.so.2|libtcmalloc_minimal.so.4>` if installed.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::engine_pool::EnginePool;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::hcore::{PeriodicHcoreConfig, GAUSSIAN_NUCLEUS_EXPONENT};
use ferric_pbc::rsgdf::DEFAULT_RSGDF_OMEGA;
use ferric_pbc::Cell;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::Barrier;
use std::time::Instant;

/// `diamond_prim.xyz` (Å), as `pbc_triplet_bench`.
const DIAMOND_PRIM_XYZ: &str = "2\ndiamond_prim\nC 0.0 0.0 0.0\nC 0.89175 0.89175 0.89175\n";
/// `diamond_prim.lattice` (Bohr rows).
const DIAMOND_PRIM_LATTICE: [[f64; 3]; 3] = [
    [0.0, 3.3703265431617879, 3.3703265431617879],
    [3.3703265431617879, 0.0, 3.3703265431617879],
    [3.3703265431617879, 3.3703265431617879, 0.0],
];
/// `rsgdf.rs` ENGINE_PRECISION / `hcore.rs` ERI3_ENGINE_PRECISION (both
/// crate-private; mirrored here as in `pbc_triplet_bench`).
const RSGDF_ENGINE_PRECISION: f64 = 1e-20;
const HCORE_ENGINE_PRECISION: f64 = f64::MIN_POSITIVE;
/// `ctl-fp` inner iterations (~2 µs per call, the integral calls' scale).
const FP_ITERS: usize = 512;
/// `ctl-mem` cache lines read per call (32 KiB).
const MEM_LINES_PER_CALL: usize = 512;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Sr3Pool,
    Sr3Own,
    Sr3ZeroShifted,
    Sr3ZeroPlain,
    HcorePool,
    CtlFp,
    CtlAlloc,
    CtlMem,
}

const ALL_KINDS: [(&str, Kind); 8] = [
    ("sr3-pool", Kind::Sr3Pool),
    ("sr3-own", Kind::Sr3Own),
    ("sr3-zero-shifted", Kind::Sr3ZeroShifted),
    ("sr3-zero-plain", Kind::Sr3ZeroPlain),
    ("hcore-pool", Kind::HcorePool),
    ("ctl-fp", Kind::CtlFp),
    ("ctl-alloc", Kind::CtlAlloc),
    ("ctl-mem", Kind::CtlMem),
];

fn kind_name(k: Kind) -> &'static str {
    ALL_KINDS
        .iter()
        .find(|(_, v)| *v == k)
        .map(|(n, _)| *n)
        .unwrap_or("?")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pin {
    None,
    Phys,
    Smt,
}

struct Args {
    basis: String,
    aux: String,
    calls: usize,
    reps: usize,
    threads: Vec<usize>,
    pin: Pin,
    kinds: Vec<Kind>,
    mem_kb: usize,
    omega: f64,
}

fn parse_args() -> Res<Args> {
    let mut a = Args {
        basis: "cc-pvdz".into(),
        aux: "cc-pvdz-ri".into(),
        calls: 100_000,
        reps: 3,
        threads: vec![1, 2, 3, 6],
        pin: Pin::None,
        kinds: ALL_KINDS.iter().map(|(_, k)| *k).collect(),
        mem_kb: 1024,
        omega: DEFAULT_RSGDF_OMEGA,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--basis" => a.basis = v()?,
            "--aux" => a.aux = v()?,
            "--calls" => a.calls = v()?.parse()?,
            "--reps" => a.reps = v()?.parse()?,
            "--threads" => {
                a.threads = v()?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<_, _>>()?
            }
            "--pin" => {
                a.pin = match v()?.as_str() {
                    "none" => Pin::None,
                    "phys" => Pin::Phys,
                    "smt" => Pin::Smt,
                    other => return Err(format!("--pin {other:?}: none|phys|smt").into()),
                }
            }
            "--variants" => {
                let s = v()?;
                if s != "all" {
                    let mut ks = Vec::new();
                    for name in s.split(',') {
                        let name = name.trim();
                        let k = ALL_KINDS
                            .iter()
                            .find(|(n, _)| *n == name)
                            .map(|(_, k)| *k)
                            .ok_or_else(|| format!("unknown variant {name:?}"))?;
                        ks.push(k);
                    }
                    a.kinds = ks;
                }
            }
            "--mem-kb" => a.mem_kb = v()?.parse()?,
            "--omega" => a.omega = v()?.parse()?,
            other => return Err(format!("unknown argument {other:?} (see the file header)").into()),
        }
    }
    if a.calls == 0 || a.reps == 0 || a.threads.is_empty() || a.threads.contains(&0) {
        return Err("--calls, --reps and every --threads entry must be >= 1".into());
    }
    if a.mem_kb < 64 {
        return Err("--mem-kb must be >= 64".into());
    }
    Ok(a)
}

// ---------------------------------------------------------------- system ---

/// Parse a sysfs CPU list ("0,6", "0-1", "3").
fn parse_cpu_list(s: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                out.extend(a..=b);
            }
        } else if let Ok(a) = part.parse::<usize>() {
            out.push(a);
        }
    }
    out
}

/// `(physical-core-first order, sibling-pair order)` of the online CPUs.
fn cpu_orders() -> (Vec<usize>, Vec<usize>) {
    let n = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    let mut cores: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for cpu in 0..n {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let sib = std::fs::read_to_string(&path)
            .map(|s| parse_cpu_list(&s))
            .unwrap_or_default();
        let key = sib.iter().copied().min().unwrap_or(cpu);
        cores.entry(key).or_default().push(cpu);
    }
    for v in cores.values_mut() {
        v.sort_unstable();
    }
    let depth = cores.values().map(Vec::len).max().unwrap_or(1);
    let mut phys = Vec::new();
    for d in 0..depth {
        for v in cores.values() {
            if let Some(&c) = v.get(d) {
                phys.push(c);
            }
        }
    }
    let smt: Vec<usize> = cores.values().flatten().copied().collect();
    (phys, smt)
}

fn pin_to(cpu: usize) -> bool {
    // SAFETY: `set` is a zeroed, writable cpu_set_t; CPU_SET bounds-checks
    // against the set size; sched_setaffinity(0, ..) only affects the
    // calling thread and reports failure through its return code.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}

/// This thread's CPU seconds.
fn thread_cpu_secs() -> f64 {
    // SAFETY: `ts` is a valid, writable timespec; clock_gettime only writes
    // it and reports failure through the return code (0.0 on failure).
    unsafe {
        let mut ts: libc::timespec = std::mem::zeroed();
        if libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) != 0 {
            return 0.0;
        }
        ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
    }
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "?".into())
}

fn env_or(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| "-".into())
}

// ------------------------------------------------------------- workloads ---

#[derive(Clone, Copy)]
struct Triple {
    ip: usize,
    i1: usize,
    i2: usize,
    shifts: [[f64; 3]; 3],
}

struct Ctx {
    obs: PreparedBasis,
    aux: PreparedBasis,
    site: SiteBasis,
    omega: f64,
    omega_h: f64,
    /// SR 3-centre, shifted (`[T, 0, L]`), unscreened by libint2.
    sr: Vec<Triple>,
    /// The same `(P | μ ν)` shells at all-zero shifts.
    sr_zero: Vec<Triple>,
    /// hcore SR: `(g_{C,M} | μ_0 ν_L)`, `[M, 0, L]`.
    hc: Vec<Triple>,
    calls: usize,
    mem_kb: usize,
}

impl Ctx {
    fn sr3_engine(&self) -> Result<Engine, String> {
        Engine::new_3center(
            Operator::erfc(self.omega),
            &self.obs,
            &self.aux,
            RSGDF_ENGINE_PRECISION,
        )
        .map_err(|e| e.to_string())
    }

    fn hcore_engine(&self) -> Result<Engine, String> {
        Engine::new_3center(
            Operator::erfc(self.omega_h),
            &self.obs,
            &self.site.prep,
            HCORE_ENGINE_PRECISION,
        )
        .map_err(|e| e.to_string())
    }
}

fn call_shifted(
    eng: &mut Engine,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    t: &Triple,
) -> Result<f64, String> {
    eng.compute_eri3_shifted(obs, aux, t.ip, t.i1, t.i2, t.shifts)
        .map(|o| o.map_or(0.0, |b| b[0]))
        .map_err(|e| e.to_string())
}

fn fp_work(k: usize) -> f64 {
    let mut a = [1.0 + k as f64 * 1e-12, 1.1, 1.2, 1.3];
    for i in 0..FP_ITERS {
        for x in a.iter_mut() {
            *x = *x * 0.999_999 + 1e-7;
        }
        if i % 16 == 0 {
            a[0] = a[0].exp().ln();
        }
    }
    a.iter().sum()
}

fn alloc_work(k: usize) -> f64 {
    let mut s = 0.0;
    for j in 0..6 {
        let v: Vec<f64> = black_box(vec![k as f64 + j as f64; 9]);
        s += v[8];
    }
    s
}

struct ThreadStat {
    wall: f64,
    cpu: f64,
    calls: usize,
}

/// `warm` untimed calls, the barrier, then `calls` timed calls of `f`.
fn timed_loop(
    barrier: &Barrier,
    warm: usize,
    calls: usize,
    mut f: impl FnMut(usize) -> Result<f64, String>,
) -> Result<ThreadStat, String> {
    let mut sink = 0.0;
    let mut err = None;
    for k in 0..warm {
        match f(k) {
            Ok(v) => sink += v,
            Err(e) => {
                err = Some(e);
                break;
            }
        }
    }
    barrier.wait();
    if let Some(e) = err {
        return Err(e);
    }
    let c0 = thread_cpu_secs();
    let t0 = Instant::now();
    for k in 0..calls {
        sink += f(k)?;
    }
    let wall = t0.elapsed().as_secs_f64();
    let cpu = thread_cpu_secs() - c0;
    black_box(sink);
    Ok(ThreadStat { wall, cpu, calls })
}

/// One worker's share of one (variant, threads) run.
fn thread_body(
    ctx: &Ctx,
    kind: Kind,
    epool: Option<&EnginePool>,
    barrier: &Barrier,
) -> Result<ThreadStat, String> {
    let calls = ctx.calls;
    // Build the thread-private state BEFORE the barrier; on failure still
    // meet the barrier so the other workers are not left waiting.
    let bail = |e: String| -> Result<ThreadStat, String> {
        barrier.wait();
        Err(e)
    };
    match kind {
        Kind::Sr3Pool | Kind::HcorePool => {
            let Some(pool) = epool else {
                return bail("internal: no engine pool".into());
            };
            let (list, aux) = if kind == Kind::Sr3Pool {
                (&ctx.sr, &ctx.aux)
            } else {
                (&ctx.hc, &ctx.site.prep)
            };
            pool.with(|eng| {
                timed_loop(barrier, list.len(), calls, |k| {
                    call_shifted(eng, &ctx.obs, aux, &list[k % list.len()])
                })
            })
        }
        Kind::Sr3Own | Kind::Sr3ZeroShifted => {
            let mut eng = match ctx.sr3_engine() {
                Ok(e) => e,
                Err(e) => return bail(e),
            };
            let list = if kind == Kind::Sr3Own {
                &ctx.sr
            } else {
                &ctx.sr_zero
            };
            timed_loop(barrier, list.len(), calls, |k| {
                call_shifted(&mut eng, &ctx.obs, &ctx.aux, &list[k % list.len()])
            })
        }
        Kind::Sr3ZeroPlain => {
            let mut eng = match ctx.sr3_engine() {
                Ok(e) => e,
                Err(e) => return bail(e),
            };
            let list = &ctx.sr_zero;
            timed_loop(barrier, list.len(), calls, |k| {
                let t = &list[k % list.len()];
                Ok(eng
                    .compute_eri3(&ctx.obs, &ctx.aux, t.ip, t.i1, t.i2)
                    .map_or(0.0, |b| b[0]))
            })
        }
        Kind::CtlFp => timed_loop(barrier, 1000, calls, |k| Ok(fp_work(k))),
        Kind::CtlAlloc => timed_loop(barrier, 1000, calls, |k| Ok(alloc_work(k))),
        Kind::CtlMem => {
            // Allocated and first-touched on this thread.
            let buf = vec![1.0_f64; ctx.mem_kb * 128];
            let lines = buf.len() / 8;
            let mut pos = 0usize;
            timed_loop(barrier, lines / MEM_LINES_PER_CALL + 1, calls, |_| {
                let mut s = 0.0;
                for _ in 0..MEM_LINES_PER_CALL {
                    s += buf[pos * 8];
                    pos += 1;
                    if pos == lines {
                        pos = 0;
                    }
                }
                Ok(s)
            })
        }
    }
}

/// One (variant, threads) run in a fresh rayon pool of `n` workers.
fn run_once(ctx: &Ctx, kind: Kind, n: usize, pins: &[usize]) -> Res<Vec<ThreadStat>> {
    let pins: Vec<usize> = pins.to_vec();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .start_handler(move |idx| {
            if let Some(&cpu) = pins.get(idx) {
                if !pin_to(cpu) {
                    eprintln!("warning: could not pin worker {idx} to CPU {cpu}");
                }
            }
        })
        .build()?;
    // As production: `from_fn` inside the pool (slot count n + 1), all
    // engines built back-to-back on one thread.
    let epool = match kind {
        Kind::Sr3Pool => Some(pool.install(|| {
            EnginePool::from_fn(|| {
                Engine::new_3center(
                    Operator::erfc(ctx.omega),
                    &ctx.obs,
                    &ctx.aux,
                    RSGDF_ENGINE_PRECISION,
                )
            })
        })?),
        Kind::HcorePool => Some(pool.install(|| {
            EnginePool::from_fn(|| {
                Engine::new_3center(
                    Operator::erfc(ctx.omega_h),
                    &ctx.obs,
                    &ctx.site.prep,
                    HCORE_ENGINE_PRECISION,
                )
            })
        })?),
        _ => None,
    };
    let barrier = Barrier::new(n);
    let out: Vec<Result<ThreadStat, String>> =
        pool.broadcast(|_| thread_body(ctx, kind, epool.as_ref(), &barrier));
    let mut stats = Vec::with_capacity(n);
    for r in out {
        stats.push(r?);
    }
    Ok(stats)
}

// ----------------------------------------------------------------- setup ---

/// FNV-1a over the bit patterns of `x`.
fn fnv(h: &mut u64, x: &[f64]) {
    for v in x {
        for b in v.to_bits().to_le_bytes() {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

const FNV_SEED: u64 = 0xcbf2_9ce4_8422_2325;

fn add3(a: [f64; 3], b: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
}

fn build_ctx(args: &Args) -> Res<Ctx> {
    let mol = Molecule::parse_xyz(DIAMOND_PRIM_XYZ, 0, 1)?;
    let cell = Cell::new(mol, DIAMOND_PRIM_LATTICE)?;
    let bs = basis::bundled(&args.basis)?;
    let aux_bs = basis::bundled(&args.aux)?;
    let obs = PreparedBasis::new(cell.mol(), &bs)?;
    let aux = PreparedBasis::new(cell.mol(), &aux_bs)?;
    // The production default split (hcore.rs `default_hcore_omega`). The
    // hcore checksum depends on it: the 2026-09-27 FINDINGS checksums were
    // taken at the older √π/Ω^{1/3}, so compare checksums at one ω_h only.
    let omega_h = PeriodicHcoreConfig::for_cell(&cell).omega;
    let sites: Vec<[f64; 4]> = cell
        .positions()
        .iter()
        .map(|r| [r[0], r[1], r[2], GAUSSIAN_NUCLEUS_EXPONENT])
        .collect();
    let site = SiteBasis::new(&sites, 0)?;
    let [a1, a2, a3] = *cell.lattice();
    let z = [0.0; 3];
    let nobs = obs.nshells();
    let naux = aux.nshells();

    let mut ctx = Ctx {
        obs,
        aux,
        site,
        omega: args.omega,
        omega_h,
        sr: Vec::new(),
        sr_zero: Vec::new(),
        hc: Vec::new(),
        calls: args.calls,
        mem_kb: args.mem_kb,
    };

    // SR 3-centre: [T, 0, L] for a few near images (every class, a mix of
    // screened / unscreened triples as in the production walk).
    let sr_geoms = [
        (a1, z),
        (z, a1),
        (a2, a3),
        (add3(z, a1, -1.0), add3(a2, a3, 1.0)),
    ];
    let mut eng = ctx.sr3_engine()?;
    let mut h_sr = FNV_SEED;
    for &(t, l) in &sr_geoms {
        for i1 in 0..nobs {
            for i2 in 0..nobs {
                for ip in 0..naux {
                    let tr = Triple {
                        ip,
                        i1,
                        i2,
                        shifts: [t, z, l],
                    };
                    if let Some(b) =
                        eng.compute_eri3_shifted(&ctx.obs, &ctx.aux, ip, i1, i2, tr.shifts)?
                    {
                        fnv(&mut h_sr, b);
                        ctx.sr.push(tr);
                    }
                }
            }
        }
    }
    // Zero shifts: shifted (shell copies) vs plain (none) must agree BITWISE.
    let mut h_zero = FNV_SEED;
    let mut zero_mismatch = 0usize;
    for i1 in 0..nobs {
        for i2 in 0..nobs {
            for ip in 0..naux {
                let shifted: Option<Vec<f64>> = eng
                    .compute_eri3_shifted(&ctx.obs, &ctx.aux, ip, i1, i2, [z; 3])?
                    .map(<[f64]>::to_vec);
                let plain: Option<Vec<f64>> = eng
                    .compute_eri3(&ctx.obs, &ctx.aux, ip, i1, i2)
                    .map(<[f64]>::to_vec);
                let same = match (&shifted, &plain) {
                    (Some(a), Some(b)) => {
                        a.len() == b.len()
                            && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
                    }
                    (None, None) => true,
                    _ => false,
                };
                if !same {
                    zero_mismatch += 1;
                }
                if let Some(b) = shifted {
                    fnv(&mut h_zero, &b);
                    ctx.sr_zero.push(Triple {
                        ip,
                        i1,
                        i2,
                        shifts: [z; 3],
                    });
                }
            }
        }
    }
    // hcore SR: (g_{C,M} | μ_0 ν_L), [M, 0, L].
    let mut eng_h = ctx.hcore_engine()?;
    let mut h_hc = FNV_SEED;
    let hc_geoms = [(z, z), (a1, z), (z, a3), (add3(z, a2, -1.0), a1)];
    for &(m, l) in &hc_geoms {
        for i1 in 0..nobs {
            for i2 in 0..nobs {
                for &sh in &ctx.site.site_shell {
                    let tr = Triple {
                        ip: sh,
                        i1,
                        i2,
                        shifts: [m, z, l],
                    };
                    if let Some(b) = eng_h.compute_eri3_shifted(
                        &ctx.obs,
                        &ctx.site.prep,
                        sh,
                        i1,
                        i2,
                        tr.shifts,
                    )? {
                        fnv(&mut h_hc, b);
                        ctx.hc.push(tr);
                    }
                }
            }
        }
    }
    println!(
        "cell diamond_prim; basis {} ({} shells, nao {}), aux {} ({} shells, naux {}); omega {} / omega_h {:.4}",
        args.basis,
        nobs,
        ctx.obs.nbasis(),
        args.aux,
        naux,
        ctx.aux.nbasis(),
        ctx.omega,
        ctx.omega_h
    );
    println!(
        "lists: sr3 {} shifted triples, sr3-zero {} triples, hcore {} triples",
        ctx.sr.len(),
        ctx.sr_zero.len(),
        ctx.hc.len()
    );
    println!("checksum sr3 {h_sr:016x}  sr3-zero {h_zero:016x}  hcore {h_hc:016x}");
    println!(
        "anchor: zero-shift compute_eri3_shifted == compute_eri3 bitwise on every triple: {}",
        if zero_mismatch == 0 {
            "PASS".to_string()
        } else {
            format!("FAIL ({zero_mismatch} triples differ)")
        }
    );
    if ctx.sr.is_empty() || ctx.sr_zero.is_empty() || ctx.hc.is_empty() {
        return Err("an empty triple list (everything screened?)".into());
    }
    Ok(ctx)
}

// ------------------------------------------------------------------ main ---

struct Row {
    wall_min: f64,
    wall_med: f64,
    cpu_min: f64,
    cpu_over_wall: f64,
    throughput: f64,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn main() -> Res<()> {
    let args = parse_args()?;
    let (phys, smt) = cpu_orders();
    let pins: Vec<usize> = match args.pin {
        Pin::None => Vec::new(),
        Pin::Phys => phys.clone(),
        Pin::Smt => smt.clone(),
    };
    println!(
        "loadavg at start: {}  | nproc {}  | phys order {:?} | smt order {:?}",
        loadavg(),
        phys.len(),
        phys,
        smt
    );
    println!(
        "env: OPENBLAS_NUM_THREADS={} RAYON_NUM_THREADS={} MALLOC_ARENA_MAX={} GLIBC_TUNABLES={} LD_PRELOAD={}",
        env_or("OPENBLAS_NUM_THREADS"),
        env_or("RAYON_NUM_THREADS"),
        env_or("MALLOC_ARENA_MAX"),
        env_or("GLIBC_TUNABLES"),
        env_or("LD_PRELOAD")
    );
    println!(
        "governor: {}",
        std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "?".into())
    );
    let ctx = build_ctx(&args)?;
    println!(
        "{} calls per thread per run, best/median of {} reps, pin {}",
        args.calls,
        args.reps,
        match args.pin {
            Pin::None => "none",
            Pin::Phys => "phys",
            Pin::Smt => "smt",
        }
    );
    println!(
        "\n{:<17} {:>3} {:>11} {:>11} {:>11} {:>8} {:>9} {:>8} {:>6}",
        "variant",
        "thr",
        "wall ns min",
        "wall ns med",
        "cpu ns min",
        "cpu/wall",
        "inflation",
        "speed-up",
        "eff"
    );
    for &kind in &args.kinds {
        let mut base: Option<(f64, f64)> = None; // (wall ns/call, throughput) at the first thread count
        for &n in &args.threads {
            let mut walls = Vec::new();
            let mut cpus = Vec::new();
            let mut ratios = Vec::new();
            let mut thr = Vec::new();
            for _ in 0..args.reps {
                let st = run_once(&ctx, kind, n, &pins)?;
                let k = st.len() as f64;
                let wall_ns = st.iter().map(|s| s.wall / s.calls as f64).sum::<f64>() / k * 1e9;
                let cpu_ns = st.iter().map(|s| s.cpu / s.calls as f64).sum::<f64>() / k * 1e9;
                let ratio = st.iter().map(|s| s.cpu / s.wall.max(1e-12)).sum::<f64>() / k;
                let max_wall = st.iter().map(|s| s.wall).fold(0.0_f64, f64::max);
                let total: usize = st.iter().map(|s| s.calls).sum();
                walls.push(wall_ns);
                cpus.push(cpu_ns);
                ratios.push(ratio);
                thr.push(total as f64 / max_wall.max(1e-12));
            }
            let row = Row {
                wall_min: walls.iter().copied().fold(f64::INFINITY, f64::min),
                wall_med: median(&mut walls.clone()),
                cpu_min: cpus.iter().copied().fold(f64::INFINITY, f64::min),
                cpu_over_wall: median(&mut ratios),
                throughput: thr.iter().copied().fold(0.0_f64, f64::max),
            };
            let (b_wall, b_thr) = *base.get_or_insert((row.wall_min, row.throughput / n as f64));
            let speedup = row.throughput / b_thr;
            println!(
                "{:<17} {:>3} {:>11.1} {:>11.1} {:>11.1} {:>8.3} {:>8.2}x {:>7.2}x {:>5.0}%",
                kind_name(kind),
                n,
                row.wall_min,
                row.wall_med,
                row.cpu_min,
                row.cpu_over_wall,
                row.wall_min / b_wall,
                speedup,
                100.0 * speedup / n as f64
            );
        }
    }
    println!("\nloadavg at end: {}", loadavg());
    println!(
        "inflation = wall ns/call over the variant's first thread count; speed-up = aggregate \
         calls/s over (first-count per-thread calls/s); cpu/wall < 1 => preempted."
    );
    Ok(())
}
