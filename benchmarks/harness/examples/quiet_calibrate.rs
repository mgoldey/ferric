//! Calibration of the external-load sampler (`ferric_benchmarks::quiet`) on its
//! own output. No GPU, no features, std only. Starts the shared sampler,
//! sleeps `FERRIC_QUIET_SECS` seconds (default 120) with no workload, and
//! prints the summary, the top-3 offenders, per-second percentiles, the count
//! of seconds with a single process above 0.10 / 0.16 / 0.21 / 0.30 cores, the
//! sampler's own cost, and a final machine-readable line
//! `CALIBRATION mean=.. max=.. spikes010=.. spikes016=.. spikes021=.. spikes030=.. secs=..`.
//!
//!   cargo run --release -p ferric-benchmarks --example quiet_calibrate
//!
//! Clean side: run it on the quiet box with nothing else active (several runs;
//! take the worst value of each statistic). Contested side:
//! `FERRIC_QUIET_BUSY=1` starts 2 busy-loop processes that are NOT descendants
//! of this process (a detached `sh` is started with `setsid`, backgrounds a
//! `timeout`-bounded busy loop and exits, so the loop is reparented to init
//! or a subreaper; its pid is written to a file and it is killed at the end).
//! Take the mildest value of each statistic (see the derivation procedure in
//! `quiet.rs`).
//!
//! Controls (exclusion checks; the summary must read as the idle value, not as
//! +N cores): `FERRIC_QUIET_BUSY_THREADS=N` spins N threads in this process,
//! `FERRIC_QUIET_BUSY_CHILD=N` spawns N busy child processes of this process.
//! `FERRIC_QUIET_DUMP=<path>` writes the per-second series.
use ferric_benchmarks::quiet;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const BUSY_LOOP: &str = "while :; do :; done";

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn own_cpu_seconds() -> f64 {
    std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| quiet::parse_stat(&s))
        .map(|p| p.ticks as f64 / quiet::CLK_TCK)
        .unwrap_or(f64::NAN)
}

/// Two busy loops outside this process's tree. Returns the pid file path.
fn spawn_detached_busy(secs: usize) -> std::path::PathBuf {
    let pidfile = std::env::temp_dir().join(format!("quiet_busy_{}.pids", std::process::id()));
    let _ = std::fs::remove_file(&pidfile);
    let script = format!(
        "for i in 1 2; do setsid sh -c 'echo $$ >> {pf}; exec timeout {t} sh -c \"{busy}\"' >/dev/null 2>&1 & done",
        pf = pidfile.display(),
        t = secs + 10,
        busy = BUSY_LOOP
    );
    let status = Command::new("sh")
        .args(["-c", &script])
        .stdin(Stdio::null())
        .status()
        .expect("spawn detached busy loops");
    assert!(status.success());
    pidfile
}

fn kill_detached(pidfile: &std::path::Path) {
    if let Ok(s) = std::fs::read_to_string(pidfile) {
        for pid in s.split_whitespace() {
            let _ = Command::new("kill").arg(pid).status();
        }
    }
    let _ = std::fs::remove_file(pidfile);
}

fn main() {
    let secs = env_usize("FERRIC_QUIET_SECS", 120);
    let busy_threads = env_usize("FERRIC_QUIET_BUSY_THREADS", 0);
    let busy_child = env_usize("FERRIC_QUIET_BUSY_CHILD", 0);
    let detached = std::env::var_os("FERRIC_QUIET_BUSY").is_some();
    println!(
        "quiet_calibrate: {secs} s, busy threads {busy_threads}, busy children {busy_child}, \
         detached busy processes {}",
        if detached { 2 } else { 0 }
    );

    let stop = Arc::new(AtomicBool::new(false));
    let spinners: Vec<_> = (0..busy_threads)
        .map(|_| {
            let flag = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut x = 1.0f64;
                while !flag.load(Ordering::Relaxed) {
                    x = std::hint::black_box(x * 1.0000001 + 1e-9);
                }
                x
            })
        })
        .collect();
    let mut children: Vec<Child> = (0..busy_child)
        .map(|_| {
            Command::new("sh")
                .args(["-c", BUSY_LOOP])
                .spawn()
                .expect("spawn busy child")
        })
        .collect();
    let pidfile = detached.then(|| spawn_detached_busy(secs));

    let cpu0 = own_cpu_seconds();
    let t0 = Instant::now();
    let sampler = quiet::Sampler::start();
    std::thread::sleep(Duration::from_secs(secs as u64));
    let acc = sampler.finish_acc();
    let wall = t0.elapsed().as_secs_f64();
    let own_cpu = own_cpu_seconds() - cpu0;

    stop.store(true, Ordering::Relaxed);
    for s in spinners {
        let _ = s.join();
    }
    for c in &mut children {
        let _ = c.kill();
        let _ = c.wait();
    }
    if let Some(pf) = &pidfile {
        kill_detached(pf);
    }

    let (summary, top, series) = match &acc {
        Some(a) => (a.summary(), a.top_n(10), a.series().to_vec()),
        None => (quiet::summary_of(None), Vec::new(), Vec::new()),
    };
    if busy_threads > 0 {
        // The busy threads' CPU belongs to this process and must not appear.
        println!("(control: {busy_threads} busy threads in this process are excluded)");
    }
    for line in quiet::calibration_lines(&summary, &top, &series, own_cpu, wall) {
        println!("{line}");
    }
    if let Ok(path) = std::env::var("FERRIC_QUIET_DUMP") {
        let mut text = String::from("second total_cores max_single_cores\n");
        for (i, s) in series.iter().enumerate() {
            text.push_str(&format!(
                "{i} {:.4} {:.4}\n",
                s.total_cores, s.max_single_cores
            ));
        }
        std::fs::write(&path, text).expect("write dump");
    }
}
