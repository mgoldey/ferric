use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

/// One unit of work: a molecule's TOML config plus where to write its logs.
struct Job {
    file_stem: String,
    toml_path: PathBuf,
    out_log_path: PathBuf,
    err_log_path: PathBuf,
}

/// Result of running one job, reported back to the coordinating thread so
/// output from concurrent children never interleaves — each job's stdout is
/// captured to a file and only printed (as a whole) once the child exits.
struct JobResult {
    file_stem: String,
    success: bool,
}

fn print_usage() {
    eprintln!(
        "Usage: ferric-batch [--jobs N] <template.toml> <xyz_dir> <output_dir>"
    );
}

/// Extract `(file_stem, full_path)` as UTF-8 strings, or `None` if either is
/// unavailable. `file_stem()` is `None` for a path with no stem (e.g. `.xyz`);
/// `to_str()` is `None` for any path component that isn't valid UTF-8. Both
/// are real possibilities for a directory of externally-supplied xyz files
/// (not internal invariants this binary controls), so the caller skips the
/// entry with a diagnostic instead of the previous bare `.unwrap().unwrap()`
/// panicking the whole batch run over one bad filename.
fn utf8_stem_and_path(path: &Path) -> Option<(String, &str)> {
    let file_stem = path.file_stem()?.to_str()?.to_string();
    let path_str = path.to_str()?;
    Some((file_stem, path_str))
}

fn main() {
    let raw_args: Vec<String> = env::args().collect();

    let mut jobs_n: usize = 1;
    let mut positional: Vec<String> = Vec::new();
    let mut i = 1;
    while i < raw_args.len() {
        let arg = &raw_args[i];
        if arg == "--jobs" {
            let val = raw_args.get(i + 1).unwrap_or_else(|| {
                eprintln!("Error: --jobs requires a value");
                print_usage();
                std::process::exit(1);
            });
            jobs_n = val.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("Error: --jobs value must be a positive integer, got '{}'", val);
                print_usage();
                std::process::exit(1);
            });
            if jobs_n == 0 {
                eprintln!("Error: --jobs must be >= 1");
                std::process::exit(1);
            }
            i += 2;
        } else if let Some(val) = arg.strip_prefix("--jobs=") {
            jobs_n = val.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("Error: --jobs value must be a positive integer, got '{}'", val);
                print_usage();
                std::process::exit(1);
            });
            if jobs_n == 0 {
                eprintln!("Error: --jobs must be >= 1");
                std::process::exit(1);
            }
            i += 1;
        } else {
            positional.push(arg.clone());
            i += 1;
        }
    }

    if positional.len() != 3 {
        print_usage();
        std::process::exit(1);
    }

    let template_path = &positional[0];
    let xyz_dir = Path::new(&positional[1]);
    let out_dir = Path::new(&positional[2]);

    if !xyz_dir.is_dir() {
        eprintln!("Error: {} is not a directory", xyz_dir.display());
        std::process::exit(1);
    }

    if !out_dir.exists() {
        fs::create_dir_all(out_dir).unwrap_or_else(|e| {
            eprintln!("Failed to create output directory: {}", e);
            std::process::exit(1);
        });
    }

    let template = fs::read_to_string(template_path).unwrap_or_else(|e| {
        eprintln!("Failed to read template: {}", e);
        std::process::exit(1);
    });

    let entries = fs::read_dir(xyz_dir).unwrap_or_else(|e| {
        eprintln!("Failed to read xyz directory: {}", e);
        std::process::exit(1);
    });

    // Locate the `ferric` binary once, relative to this executable (same
    // logic as before: prefer the sibling binary, fall back to PATH).
    let current_exe = env::current_exe().unwrap_or_default();
    let ferric_bin: PathBuf = if let Some(parent) = current_exe.parent() {
        let bin = parent.join("ferric");
        if bin.exists() {
            bin
        } else {
            PathBuf::from("ferric")
        }
    } else {
        PathBuf::from("ferric")
    };

    // Build the job list: write each molecule's TOML up front (this was
    // already done eagerly in the serial version; keeping it serial here
    // avoids concurrent writers and keeps job construction simple).
    let mut jobs: Vec<Job> = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();

        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("xyz") {
            let (file_stem, path_str) = match utf8_stem_and_path(&path) {
                Some(pair) => pair,
                None => {
                    eprintln!(
                        "Warning: skipping {} (non-UTF-8 or missing file stem)",
                        path.display()
                    );
                    continue;
                }
            };

            let config_content = template.replace("{XYZ_FILE}", path_str);

            let toml_path = out_dir.join(format!("{}.toml", file_stem));
            let out_log_path = out_dir.join(format!("{}.out", file_stem));
            let err_log_path = out_dir.join(format!("{}.err", file_stem));

            fs::write(&toml_path, config_content).unwrap();

            jobs.push(Job {
                file_stem,
                toml_path,
                out_log_path,
                err_log_path,
            });
        }
    }

    // Resolve a per-child memory budget: divide the parent's resolved budget
    // by --jobs so N concurrent children can't collectively exceed the box.
    // Mirrors ferric_core::memory::resolve_budget_bytes's own precedence
    // (explicit > FERRIC_MEM_BUDGET_GB > legacy env > auto-detected RAM >
    // 2 GiB fallback); ferric-cli already depends on ferric-core so no new
    // dependency edge is needed.
    let njobs = jobs_n.max(1);
    let per_child_budget_gb = resolve_per_child_budget_gib(njobs);

    let (result_tx, result_rx) = mpsc::channel::<JobResult>();
    let job_queue = Mutex::new(jobs.into_iter());
    let ferric_bin = &ferric_bin;
    let njobs_effective = njobs.min(job_queue.lock().unwrap().len().max(1));

    thread::scope(|scope| {
        for _ in 0..njobs_effective {
            let result_tx = result_tx.clone();
            let job_queue = &job_queue;
            let ferric_bin = ferric_bin.clone();
            scope.spawn(move || loop {
                let job = {
                    let mut q = job_queue.lock().unwrap();
                    q.next()
                };
                let job = match job {
                    Some(j) => j,
                    None => break,
                };
                run_job(&ferric_bin, &job, per_child_budget_gb, &result_tx);
            });
        }
        drop(result_tx);

        let mut success_count = 0;
        let mut fail_count = 0;
        let mut failures: Vec<String> = Vec::new();

        for result in result_rx {
            if result.success {
                println!("{}: Success.", result.file_stem);
                success_count += 1;
            } else {
                println!("{}: Failed. Check {}.err", result.file_stem, result.file_stem);
                fail_count += 1;
                failures.push(result.file_stem);
            }
        }

        println!("\nBatch run complete.");
        println!("Successful: {}", success_count);
        println!("Failed:     {}", fail_count);

        if !failures.is_empty() {
            println!("Failed jobs: {}", failures.join(", "));
            std::process::exit(1);
        }
    });
}

/// Run a single child `ferric` job, capturing stdout/stderr to per-job files
/// (so concurrent children never interleave on the terminal) and reporting
/// the outcome over `result_tx`.
fn run_job(
    ferric_bin: &Path,
    job: &Job,
    per_child_budget_gb: Option<f64>,
    result_tx: &mpsc::Sender<JobResult>,
) {
    println!("Running {}...", job.file_stem);

    let stdout_file = match fs::File::create(&job.out_log_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{}: failed to create stdout log: {}", job.file_stem, e);
            let _ = result_tx.send(JobResult {
                file_stem: job.file_stem.clone(),
                success: false,
            });
            return;
        }
    };
    let stderr_file = match fs::File::create(&job.err_log_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{}: failed to create stderr log: {}", job.file_stem, e);
            let _ = result_tx.send(JobResult {
                file_stem: job.file_stem.clone(),
                success: false,
            });
            return;
        }
    };

    let mut cmd = Command::new(ferric_bin);
    cmd.arg(&job.toml_path)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .stdout(stdout_file)
        .stderr(stderr_file);

    if let Some(budget_gb) = per_child_budget_gb {
        cmd.env("FERRIC_MEM_BUDGET_GB", format!("{}", budget_gb));
    }

    let timeout = job_timeout();
    let success = match run_with_timeout(&mut cmd, timeout) {
        Ok(RunOutcome::Exited(status)) => status.success(),
        Ok(RunOutcome::TimedOut) => {
            eprintln!("{}: timed out after {:?}, killed", job.file_stem, timeout);
            false
        }
        Err(e) => {
            eprintln!("{}: failed to run job: {}", job.file_stem, e);
            false
        }
    };
    let _ = result_tx.send(JobResult {
        file_stem: job.file_stem.clone(),
        success,
    });
}

/// Per-job wall-clock budget: a hung child (stuck SCF, deadlocked solver,
/// zombie waiting on a resource) previously blocked its worker thread
/// forever inside `cmd.status()`, silently stalling the whole batch (the
/// job queue never advances past a job whose worker never returns).
/// Overridable via `FERRIC_BATCH_JOB_TIMEOUT_S`; 4 hours is comfortably
/// above any single-molecule job this binary is used for today (the
/// gw100 driver's own stall watchdog uses 30 min per molecule for a much
/// finer-grained workload — this is a coarser last-resort backstop).
fn job_timeout() -> Duration {
    let secs = env::var("FERRIC_BATCH_JOB_TIMEOUT_S")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(4 * 3600);
    Duration::from_secs(secs)
}

enum RunOutcome {
    Exited(std::process::ExitStatus),
    TimedOut,
}

/// Spawn `cmd` and poll `try_wait()` on a deadline instead of blocking
/// indefinitely in `Command::status()`. No `wait-timeout`-style crate is in
/// this workspace's dependency tree (checked Cargo.toml/Cargo.lock) and
/// pulling one in for a single call site is out of proportion here, so this
/// is a plain spawn + poll loop: cheap, no new dependency, and this
/// function's only caller runs on a per-job worker thread (not the async/perf
/// hot path), so the poll interval's latency doesn't matter.
fn run_with_timeout(cmd: &mut Command, timeout: Duration) -> std::io::Result<RunOutcome> {
    let mut child = cmd.spawn()?;
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(RunOutcome::Exited(status));
        }
        if Instant::now() >= deadline {
            // Best-effort kill; if the process already exited between the
            // try_wait() above and here, `kill()` returns an error we can
            // ignore (nothing left to signal). Reap it either way so it
            // doesn't linger as a zombie.
            let _ = child.kill();
            let _ = child.wait();
            return Ok(RunOutcome::TimedOut);
        }
        thread::sleep(Duration::from_millis(500).min(deadline - Instant::now()));
    }
}

/// Resolve the per-child FERRIC_MEM_BUDGET_GB to pass down to each of the
/// `njobs` concurrent children, dividing the parent's resolved budget by
/// `njobs` so concurrent children can't collectively exceed the box.
///
/// Uses `ferric_core::memory::resolve_budget_bytes` (same precedence a
/// single `ferric` invocation would use: explicit > FERRIC_MEM_BUDGET_GB >
/// legacy FERRIC_OOC_BUDGET_GB/FERRIC_ERI3_BUDGET_GB > auto-detected
/// available RAM > 2 GiB fallback) so `ferric-batch --jobs N` divides
/// whatever budget the parent resolves to across its children.
fn resolve_per_child_budget_gib(njobs: usize) -> Option<f64> {
    let total_bytes = ferric_core::memory::resolve_budget_bytes(None);
    let per_child_bytes = total_bytes / njobs.max(1);
    if per_child_bytes == 0 {
        return None;
    }
    Some(per_child_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_stem_and_path_normal_file() {
        let path = Path::new("/tmp/water.xyz");
        let (stem, s) = utf8_stem_and_path(path).expect("valid UTF-8 path");
        assert_eq!(stem, "water");
        assert_eq!(s, "/tmp/water.xyz");
    }

    #[test]
    fn utf8_stem_and_path_no_stem_returns_none() {
        // file_stem() is None exactly when file_name() is None (per
        // std::path docs). A path with no file-name component at all --
        // root -- is the reliable way to hit that: "/tmp/" normalizes its
        // trailing slash away (file_name() == Some("tmp")), so it does NOT
        // trigger this case; "/" has no file-name component at all.
        let path = Path::new("/");
        assert!(utf8_stem_and_path(path).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn utf8_stem_and_path_non_utf8_returns_none_not_panic() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        // 0xFF is not valid UTF-8 in any position; build a non-UTF-8 filename
        // the way a real (externally supplied) directory entry could contain
        // one. Before the fix, path.file_stem().unwrap().to_str().unwrap()
        // (or path.to_str().unwrap()) would panic here.
        let bad_name = OsStr::from_bytes(&[0x66, 0x6f, 0xff, 0x2e, 0x78, 0x79, 0x7a]); // "fo\xFF.xyz"
        let path = Path::new(bad_name);
        assert!(utf8_stem_and_path(path).is_none());
    }
}
