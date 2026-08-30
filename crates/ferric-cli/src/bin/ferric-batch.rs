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
    // by the number of children that will actually run CONCURRENTLY, so they
    // can't collectively exceed the box. Mirrors
    // ferric_core::memory::resolve_budget_bytes's own precedence (explicit >
    // FERRIC_MEM_BUDGET_GB > legacy env > auto-detected RAM > 2 GiB fallback);
    // ferric-cli already depends on ferric-core so no new dependency edge is
    // needed.
    //
    // The divisor is the EFFECTIVE worker count, not the requested `--jobs`.
    // `thread::scope` below spawns `min(--jobs, job_count)` workers, so with
    // `--jobs 16` over 2 molecules only 2 children ever run at once; dividing
    // by 16 told each of those 2 children it could use 1/16 of the box — an 8x
    // under-budget that forces needless `ThreeIndexSource` disk spilling for no
    // safety benefit whatsoever.
    let njobs = jobs_n.max(1);
    let job_count = jobs.len();
    let njobs_effective = njobs.min(job_count.max(1));
    let per_child_budget_gb = resolve_per_child_budget_gib(njobs_effective);

    // The env var alone is NOT sufficient to hold a child to its share: a
    // template carrying `[memory] budget_gb` copies verbatim into every child
    // TOML, and `resolve_budget` ranks explicit config ABOVE
    // FERRIC_MEM_BUDGET_GB. Rewrite the generated per-child TOMLs so the
    // config agrees with the env var. See `apply_per_child_budget_to_toml`.
    if let Some(budget_gb) = per_child_budget_gb {
        for job in &jobs {
            rewrite_child_toml_budget(&job.toml_path, budget_gb);
        }
    }

    let (result_tx, result_rx) = mpsc::channel::<JobResult>();
    let job_queue = Mutex::new(jobs.into_iter());
    let ferric_bin = &ferric_bin;

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

/// The smallest per-child budget this binary will ever hand a child, in GiB.
///
/// A budget that rounds to zero must NOT become "no budget at all": see
/// [`resolve_per_child_budget_gib`]. 64 MiB is small enough that the child
/// spills essentially everything to disk (which is the honest behaviour when
/// the caller has asked for more concurrency than the box has memory for) and
/// large enough that `ThreeIndexSource`'s block sizing still has something to
/// work with instead of a 0-byte ceiling under which every single allocation
/// looks oversized.
const MIN_PER_CHILD_BUDGET_GIB: f64 = 0.0625;

/// Resolve the per-child FERRIC_MEM_BUDGET_GB to pass down to each of the
/// `njobs_effective` CONCURRENT children, dividing the parent's resolved budget
/// by that count so concurrent children can't collectively exceed the box.
///
/// Uses `ferric_core::memory::resolve_budget_bytes` (same precedence a
/// single `ferric` invocation would use: explicit > FERRIC_MEM_BUDGET_GB >
/// legacy FERRIC_OOC_BUDGET_GB/FERRIC_ERI3_BUDGET_GB > auto-detected
/// available RAM > 2 GiB fallback) so `ferric-batch --jobs N` divides
/// whatever budget the parent resolves to across its children.
///
/// # Why the zero case clamps instead of returning `None`
///
/// This used to `return None` when the division floored to 0 bytes, and `None`
/// means "set no env var", which means the child falls all the way through to
/// `0.8 x available RAM` — i.e. asking for *more* concurrency than the box can
/// divide (a huge `--jobs`, or a tiny parent budget) silently turned the
/// overcommit guard completely OFF, giving every child the WHOLE box. That is
/// the exact inverse of this function's purpose. A degenerate division now
/// clamps to [`MIN_PER_CHILD_BUDGET_GIB`] and warns; the result is always
/// `Some`, so the guard can never be disabled by arithmetic.
fn resolve_per_child_budget_gib(njobs_effective: usize) -> Option<f64> {
    let total_bytes = ferric_core::memory::resolve_budget_bytes(None);
    let per_child_bytes = total_bytes / njobs_effective.max(1);
    let per_child_gib = per_child_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if per_child_gib < MIN_PER_CHILD_BUDGET_GIB {
        eprintln!(
            "ferric-batch WARNING: {} concurrent jobs divide the {:.2} GiB resolved budget down \
             to {:.4} GiB each; clamping to the {:.4} GiB floor. Expect heavy disk spilling — \
             lower --jobs or raise the budget.",
            njobs_effective, total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            per_child_gib, MIN_PER_CHILD_BUDGET_GIB,
        );
        return Some(MIN_PER_CHILD_BUDGET_GIB);
    }
    Some(per_child_gib)
}

/// Rewrite one generated per-child TOML in place so its `[memory]` budget keys
/// carry `budget_gb`, reporting on stderr if a template value was overridden.
/// I/O failures are warnings, not fatal: the env var is still set, so the worst
/// case is the pre-existing (buggy) behaviour for that one job rather than an
/// aborted batch.
fn rewrite_child_toml_budget(toml_path: &Path, budget_gb: f64) {
    let contents = match fs::read_to_string(toml_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "ferric-batch WARNING: could not re-read {} to apply the per-child memory \
                 budget: {e}",
                toml_path.display()
            );
            return;
        }
    };
    let Some(rewritten) = apply_per_child_budget_to_toml(&contents, budget_gb, &mut |old, key| {
        // Silently "correcting" a value the user wrote is its own bug — say so,
        // naming both numbers, so a surprising budget in a child log is
        // traceable to this line rather than looking like a ferric-core bug.
        eprintln!(
            "ferric-batch: {}: overriding template [memory] {key} = {old} with the per-child \
             share {budget_gb:.4} GiB (concurrent children may not each take the full \
             configured budget)",
            toml_path.display(),
        );
    }) else {
        return;
    };
    if let Err(e) = fs::write(toml_path, rewritten) {
        eprintln!(
            "ferric-batch WARNING: could not write the per-child memory budget into {}: {e}",
            toml_path.display()
        );
    }
}

/// Force the per-child budget into a generated TOML's `[memory]` table.
///
/// Returns `Some(new_contents)` when a rewrite was needed, `None` when the
/// document already has nothing to override (no `[memory]` budget key at all),
/// in which case `FERRIC_MEM_BUDGET_GB` alone already governs the child.
///
/// # Why this exists (the N-times-overcommit bug)
///
/// `ferric-batch` builds each child's config by copying the template verbatim
/// (`template.replace("{XYZ_FILE}", ...)`), and `ferric_core::memory::
/// resolve_budget` ranks an **explicit** config value ABOVE the
/// `FERRIC_MEM_BUDGET_GB` env var. So a template containing
///
/// ```toml
/// [memory]
/// budget_gb = 24.0
/// ```
///
/// silently defeated the whole per-child division: every one of the N
/// concurrent children read 24 GiB out of its own TOML, ignored the env var
/// entirely, and the batch could collectively claim N x 24 GiB with no warning.
/// This was the one path in the tree that could OOM the machine. Making the
/// generated config *agree* with the env var is the fix, because it is the only
/// way to win against a precedence chain that (correctly, for a single job)
/// puts config first.
///
/// Both `budget_gb` and its deprecated alias `three_index_budget_gb` are
/// rewritten: `MemoryCfg::budget_gb()` falls back to the alias, so leaving the
/// alias alone would reopen the same hole for older templates.
///
/// # Why a line rewrite rather than a parse-and-reserialize
///
/// Round-tripping through `toml::Value` would reorder tables, drop every
/// comment, and normalize float formatting across the whole document — a large,
/// lossy diff to a file a user may well read when debugging a batch. This walks
/// the lines instead, tracking the current table header so it edits ONLY keys
/// inside `[memory]`; a `budget_gb` under any other table (say a
/// `[some_other_section]`, or a nested `[memory.something]`) is left untouched.
fn apply_per_child_budget_to_toml(
    contents: &str,
    budget_gb: f64,
    on_override: &mut dyn FnMut(&str, &str),
) -> Option<String> {
    let mut out = String::with_capacity(contents.len() + 32);
    let mut in_memory_table = false;
    let mut changed = false;
    // Whether the canonical replacement key has already been written. TOML
    // forbids a table header appearing twice, so one flag for the whole
    // document is enough to keep exactly one `budget_gb` in `[memory]`.
    let mut emitted = false;
    // Preserve the input's exact final-newline behaviour: `lines()` drops it.
    let had_trailing_newline = contents.ends_with('\n');

    for line in contents.lines() {
        let trimmed = line.trim();
        if let Some(header) = trimmed.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
            // Exactly `[memory]` — NOT `[memory.sub]` (a different table) and
            // not `[[memory]]` (an array of tables, which `header` would leave
            // as "[memory"). Both correctly fail this comparison.
            in_memory_table = header.trim() == "memory";
            out.push_str(line);
            out.push('\n');
            continue;
        }
        if in_memory_table {
            if let Some((key, value)) = split_toml_key_value(trimmed) {
                if key == "budget_gb" || key == "three_index_budget_gb" {
                    on_override(value, key);
                    changed = true;
                    // Emit the canonical `budget_gb` once, at the position of
                    // the FIRST budget key seen; any further budget key (e.g. a
                    // template setting both `budget_gb` and the deprecated
                    // `three_index_budget_gb`) is dropped rather than rewritten
                    // — emitting it twice would be a duplicate-key TOML parse
                    // error, and leaving the stale alias in place would be a
                    // misleading artifact in a file users read while debugging.
                    if !emitted {
                        out.push_str(&format_toml_float(budget_gb));
                        emitted = true;
                    }
                    continue;
                }
            }
        }
        out.push_str(line);
        out.push('\n');
    }

    if !changed {
        return None;
    }
    if !had_trailing_newline {
        out.pop();
    }
    Some(out)
}

/// Render the `budget_gb = <value>` line, forcing a TOML **float** literal.
///
/// `MemoryCfg::budget_gb` is an `Option<f64>` and TOML (unlike JSON) has
/// distinct integer and float types with no implicit widening — serde rejects
/// `budget_gb = 3` for an `f64` field with an "invalid type: integer" error. A
/// plain `format!("{}", 3.0f64)` produces exactly that `3`, so a whole-number
/// per-child share (the common case: 16 GiB over 2 jobs) would have written a
/// config that fails to parse and kills every child in the batch. `{:?}` on an
/// f64 always keeps the decimal point (`3.0`), and stays exact for
/// non-whole values (`0.0625`), so it is the right formatter here.
fn format_toml_float(value: f64) -> String {
    format!("budget_gb = {value:?}\n")
}

/// Split a trimmed TOML line into `(key, value)` for a simple bare-key
/// assignment, or `None` for anything else (a comment, a blank line, a quoted
/// or dotted key, a line with no `=`). Deliberately conservative: this drives
/// an in-place edit, so anything it does not confidently understand must be
/// left exactly as written rather than guessed at.
fn split_toml_key_value(trimmed: &str) -> Option<(&str, &str)> {
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    let (key, value) = trimmed.split_once('=')?;
    let key = key.trim();
    if key.is_empty()
        || !key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return None;
    }
    Some((key, value.trim()))
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

    // ---- per-child memory budget: the N-times-overcommit fix ----

    /// FERRIC_* budget env vars are process-global and the harness runs tests
    /// in parallel; serialize every test that touches them.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Apply the rewrite, collecting the (old_value, key) pairs the override
    /// notice was fired for, so a test can assert BOTH the rewritten text and
    /// that the user was actually told about the correction.
    fn apply(contents: &str, budget_gb: f64) -> (Option<String>, Vec<(String, String)>) {
        let mut notices: Vec<(String, String)> = Vec::new();
        let out = apply_per_child_budget_to_toml(contents, budget_gb, &mut |old, key| {
            notices.push((old.to_string(), key.to_string()));
        });
        (out, notices)
    }

    /// REGRESSION (the OOM bug): a template carrying `[memory] budget_gb`
    /// copied verbatim into every child, and `resolve_budget` ranks explicit
    /// config above `FERRIC_MEM_BUDGET_GB` — so all N concurrent children took
    /// the FULL configured budget and the batch could claim N x it.
    #[test]
    fn template_memory_budget_is_replaced_by_the_per_child_value() {
        let template = "\
[molecule]
xyz_file = \"/tmp/water.xyz\"

[memory]
budget_gb = 24.0

[method]
kind = \"rimp2\"
";
        let (out, notices) = apply(template, 3.0);
        let out = out.expect("a template WITH [memory] budget_gb must be rewritten");
        assert!(out.contains("budget_gb = 3"), "got:\n{out}");
        assert!(!out.contains("24.0"), "the template's 24.0 GiB must be gone:\n{out}");
        // The user must be TOLD their config was overridden, with both numbers.
        assert_eq!(notices, vec![("24.0".to_string(), "budget_gb".to_string())]);
        // Nothing outside [memory] may be disturbed.
        assert!(out.contains("xyz_file = \"/tmp/water.xyz\""));
        assert!(out.contains("kind = \"rimp2\""));
        assert!(out.contains("[molecule]") && out.contains("[method]"));
        // And the result must still be valid, parseable TOML.
        let v: toml::Value = toml::from_str(&out).expect("rewritten TOML must still parse");
        assert_eq!(v["memory"]["budget_gb"].as_float(), Some(3.0));
    }

    /// The deprecated alias reaches the SAME explicit slot via
    /// `MemoryCfg::budget_gb()`'s fallback, so leaving it alone would reopen
    /// the identical hole for older templates.
    #[test]
    fn deprecated_three_index_alias_is_also_overridden() {
        let template = "[memory]\nthree_index_budget_gb = 16.0\n";
        let (out, notices) = apply(template, 2.5);
        let out = out.expect("the deprecated alias must be rewritten too");
        assert_eq!(notices, vec![("16.0".to_string(), "three_index_budget_gb".to_string())]);
        // Rewritten as the canonical key, and the stale alias must not survive
        // (`budget_gb` wins over it, but leaving 16.0 in the file would be a
        // misleading artifact in a config a user may read while debugging).
        assert!(!out.contains("three_index_budget_gb"), "got:\n{out}");
        let v: toml::Value = toml::from_str(&out).unwrap();
        assert_eq!(v["memory"]["budget_gb"].as_float(), Some(2.5));
    }

    /// No `[memory]` section: nothing to override, so the file is left exactly
    /// as written and `FERRIC_MEM_BUDGET_GB` alone governs the child (which it
    /// does correctly — the env var only loses to an *explicit* config value).
    #[test]
    fn template_without_memory_section_is_left_alone() {
        let template = "[molecule]\nxyz_file = \"/tmp/a.xyz\"\n\n[method]\nkind = \"rhf\"\n";
        let (out, notices) = apply(template, 3.0);
        assert!(out.is_none(), "a template with no [memory] budget must not be rewritten");
        assert!(notices.is_empty(), "and must produce no override notice");
    }

    /// A `[memory]` table that exists but sets no budget key is also untouched.
    #[test]
    fn memory_section_without_a_budget_key_is_left_alone() {
        let template = "[memory]\n# no budget set here\n";
        let (out, _) = apply(template, 3.0);
        assert!(out.is_none());
    }

    /// The rewrite must be scoped to the `[memory]` table: a `budget_gb` key
    /// belonging to some other table is a different setting entirely and
    /// clobbering it would be a silent config corruption.
    #[test]
    fn budget_gb_under_a_different_table_is_not_rewritten() {
        let template = "\
[memory]
budget_gb = 24.0

[other]
budget_gb = 99.0
";
        let (out, notices) = apply(template, 1.5);
        let out = out.unwrap();
        assert_eq!(notices.len(), 1, "only the [memory] key may be overridden");
        let v: toml::Value = toml::from_str(&out).unwrap();
        assert_eq!(v["memory"]["budget_gb"].as_float(), Some(1.5));
        assert_eq!(
            v["other"]["budget_gb"].as_float(),
            Some(99.0),
            "a budget_gb in another table must survive untouched"
        );

        // Same guarantee when the foreign table comes FIRST (the walker must
        // not still be "inside" [memory] from a previous document position).
        let template = "[other]\nbudget_gb = 99.0\n\n[memory]\nbudget_gb = 24.0\n";
        let (out, notices) = apply(template, 1.5);
        let v: toml::Value = toml::from_str(&out.unwrap()).unwrap();
        assert_eq!(notices.len(), 1);
        assert_eq!(v["other"]["budget_gb"].as_float(), Some(99.0));
        assert_eq!(v["memory"]["budget_gb"].as_float(), Some(1.5));
    }

    /// A nested `[memory.foo]` sub-table is NOT `[memory]`; its keys are a
    /// different setting and must not be rewritten either.
    #[test]
    fn nested_memory_subtable_is_not_rewritten() {
        let template = "[memory.spill]\nbudget_gb = 7.0\n";
        let (out, notices) = apply(template, 1.0);
        assert!(out.is_none(), "[memory.spill] is not [memory]");
        assert!(notices.is_empty());
    }

    /// Comments and formatting outside the rewritten key survive: a user
    /// debugging a batch reads these generated files, so the diff against the
    /// template must be exactly the one intended line.
    #[test]
    fn comments_and_other_keys_are_preserved() {
        let template = "\
# batch template
[memory]
# how much RAM one job may hold resident
budget_gb = 24.0
three_index_budget_gb = 8.0
";
        let (out, notices) = apply(template, 4.0);
        let out = out.unwrap();
        assert!(out.contains("# batch template"));
        assert!(out.contains("# how much RAM one job may hold resident"));
        // Both budget keys were present; both are overridden (and both flagged).
        assert_eq!(notices.len(), 2);
        let v: toml::Value = toml::from_str(&out).unwrap();
        assert_eq!(v["memory"]["budget_gb"].as_float(), Some(4.0));
    }

    /// A file with no trailing newline must not silently gain one (and one
    /// with a trailing newline must keep exactly one).
    #[test]
    fn trailing_newline_behaviour_is_preserved() {
        let (out, _) = apply("[memory]\nbudget_gb = 9.0", 1.0);
        let out = out.unwrap();
        assert!(!out.ends_with('\n'), "input had no trailing newline: {out:?}");

        let (out, _) = apply("[memory]\nbudget_gb = 9.0\n", 1.0);
        let out = out.unwrap();
        assert!(out.ends_with('\n'));
        assert!(!out.ends_with("\n\n"), "must not gain a blank line: {out:?}");
    }

    /// REGRESSION: the emitted value must be a TOML *float*, not an integer.
    /// TOML has no implicit int->float widening, so `budget_gb = 8` makes serde
    /// reject the whole config ("invalid type: integer") for the `Option<f64>`
    /// field — which would have killed every child whenever the per-child share
    /// came out a whole number (16 GiB over 2 jobs being the obvious case).
    #[test]
    fn whole_number_budget_is_written_as_a_toml_float() {
        for (share, want) in [(8.0f64, "8.0"), (1.0, "1.0"), (MIN_PER_CHILD_BUDGET_GIB, "0.0625")] {
            let (out, _) = apply("[memory]\nbudget_gb = 99.0\n", share);
            let out = out.unwrap();
            assert!(out.contains(&format!("budget_gb = {want}")), "share {share}, got:\n{out}");
            // And it must deserialize into the real `Option<f64>` config field.
            #[derive(serde::Deserialize)]
            struct Mem {
                budget_gb: Option<f64>,
            }
            #[derive(serde::Deserialize)]
            struct Doc {
                memory: Mem,
            }
            let doc: Doc = toml::from_str(&out)
                .unwrap_or_else(|e| panic!("share {share} produced unparseable TOML: {e}\n{out}"));
            assert_eq!(doc.memory.budget_gb, Some(share));
        }
    }

    /// REGRESSION: the divisor is the EFFECTIVE concurrency, not the requested
    /// `--jobs`. With `--jobs 16` over 2 molecules only 2 children ever run at
    /// once, so dividing by 16 gave each an 8x under-budget and forced
    /// needless disk spilling.
    #[test]
    fn divisor_is_the_effective_job_count_not_the_requested_one() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("FERRIC_MEM_BUDGET_GB", "16");

        // What main() computes: min(--jobs, job_count).
        let njobs_requested = 16usize;
        let job_count = 2usize;
        let njobs_effective = njobs_requested.min(job_count.max(1));
        assert_eq!(njobs_effective, 2);

        let per_child = resolve_per_child_budget_gib(njobs_effective).unwrap();
        assert!(
            (per_child - 8.0).abs() < 1e-9,
            "16 GiB over 2 concurrent children is 8 GiB each, got {per_child}"
        );
        // The old (buggy) divisor would have produced 1 GiB.
        let buggy = resolve_per_child_budget_gib(njobs_requested).unwrap();
        assert!((buggy - 1.0).abs() < 1e-9);

        std::env::remove_var("FERRIC_MEM_BUDGET_GB");
    }

    /// More jobs than molecules never shrinks the share below one-per-molecule.
    #[test]
    fn effective_count_saturates_at_the_molecule_count() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("FERRIC_MEM_BUDGET_GB", "12");
        // 1 molecule, --jobs 100 -> the single child gets the WHOLE budget.
        let per_child = resolve_per_child_budget_gib(100usize.min(1usize.max(1))).unwrap();
        assert!((per_child - 12.0).abs() < 1e-9, "got {per_child}");
        std::env::remove_var("FERRIC_MEM_BUDGET_GB");
    }

    /// REGRESSION: a degenerate division used to `return None`, which means
    /// "set no env var", which means the child auto-detects and takes the
    /// WHOLE box — the exact inverse of this guard's purpose. It must clamp to
    /// a positive floor instead, so the guard can never be switched off by
    /// arithmetic.
    #[test]
    fn zero_per_child_budget_clamps_instead_of_going_unlimited() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("FERRIC_MEM_BUDGET_GB", "1");

        // A divisor large enough to floor the byte division to 0.
        let huge = 4 * 1024 * 1024 * 1024usize;
        let per_child = resolve_per_child_budget_gib(huge);
        assert_eq!(
            per_child,
            Some(MIN_PER_CHILD_BUDGET_GIB),
            "an over-divided budget must clamp to the floor, never become None/unlimited"
        );
        // Anything short of the floor clamps the same way.
        assert_eq!(
            resolve_per_child_budget_gib(1024),
            Some(MIN_PER_CHILD_BUDGET_GIB)
        );
        // And the function NEVER returns None, for any divisor.
        for n in [1usize, 2, 7, 64, 1_000_000, usize::MAX] {
            let got = resolve_per_child_budget_gib(n).expect("must always be Some");
            assert!(got >= MIN_PER_CHILD_BUDGET_GIB, "n={n} gave {got}");
        }

        std::env::remove_var("FERRIC_MEM_BUDGET_GB");
    }

    /// End-to-end on the shape `main()` actually produces: template ->
    /// `{XYZ_FILE}` substitution -> budget rewrite. The child's TOML must carry
    /// the per-child share, not the template's, AND still name the right xyz.
    #[test]
    fn generated_child_config_carries_the_per_child_budget() {
        let template = "\
[molecule]
xyz_file = \"{XYZ_FILE}\"

[memory]
budget_gb = 32.0
";
        let substituted = template.replace("{XYZ_FILE}", "/data/mol_07.xyz");
        let (out, _) = apply(&substituted, 32.0 / 4.0);
        let v: toml::Value = toml::from_str(&out.unwrap()).unwrap();
        assert_eq!(v["molecule"]["xyz_file"].as_str(), Some("/data/mol_07.xyz"));
        assert_eq!(
            v["memory"]["budget_gb"].as_float(),
            Some(8.0),
            "4 concurrent children of a 32 GiB budget get 8 GiB each, not 32"
        );
    }
}
