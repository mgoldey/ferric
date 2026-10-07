//! Quotability check for timing harnesses: an INDEPENDENT measure of other
//! processes' CPU use during a run.
//!
//! `/proc/pressure/cpu` `some avg10` cannot be the "after" criterion: a
//! harness that itself keeps 6 threads busy raises it (control: 6 busy loops
//! for 30 s on a quiet box read 0.13 immediately after). A sampler thread
//! instead reads `/proc/<pid>/stat` for every pid once a second and totals the
//! CPU time (utime + stime) of every process that is NOT this process or one
//! of its descendants (the ccsd harness spawns child arms).
//!
//! What counts as external load: every other process, including a controller's
//! own monitoring loops (`ps`, `htop`) and kernel/driver threads (a small
//! effect). Run nothing alongside a quotable run.
//!
//! The sampled span is whatever lies between `Sampler::start` and
//! `Sampler::finish`; each harness starts it immediately before its timed
//! region (see the harness docs for what that covers). The sampler fails
//! CLOSED: a panicked thread, an unreadable or incomplete `/proc` (the table
//! must contain this process), an unreadable `/proc/uptime`, or fewer than
//! [`MIN_SAMPLES`] one-second samples all give an invalid summary that is
//! never quotable.
//!
//! ## Thresholds: PROVISIONAL (not yet calibrated on this sampler)
//!
//! The numbers below came from a DIFFERENT instrument: the old `ps` contender
//! logger, whose %CPU is a lifetime average (cputime/etime) over processes above
//! 3%. This sampler reads one-second deltas of every process, quantised at 0.01
//! core and burstier (a background service reading 0.084 cores lifetime-average
//! reads 0.169 per second live). They are placeholders and `CALIBRATED` stays
//! `false` until they are replaced.
//!
//! Derivation procedure (done with this sampler, on the quiet box, via the
//! `quiet_calibrate` example):
//! 1. Clean side: several runs with no workload; take the WORST value of each
//!    statistic (mean total cores, max single process in any second, fraction
//!    of seconds with a process above `SPIKE_CORES`).
//! 2. Contested side: `FERRIC_QUIET_BUSY=1` runs (two busy loops outside the
//!    harness's process tree); take the MILDEST value of each statistic.
//! 3. Each threshold is the geometric midpoint, sqrt(clean_worst *
//!    contested_mildest). Then set the constants and `CALIBRATED = true`.
//!
//! The spike tolerance is a FRACTION of the sampled seconds (floor 1 s), so a
//! multi-minute run is not failed by the first cron or updater burst.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// `USER_HZ` on Linux (clock ticks per second in `/proc/<pid>/stat`).
pub const CLK_TCK: f64 = 100.0;

// ---- Thresholds: one block; replace the numbers and set CALIBRATED = true
// in a data-only commit once derived from this sampler's own output. ----
/// `false` while the numbers below are placeholders from a different instrument.
pub const CALIBRATED: bool = false;
/// Quotable only if the mean total non-harness CPU is at most this (cores).
pub const MAX_MEAN_CORES: f64 = 0.27;
/// Quotable only if no single process used more than this in any second.
pub const MAX_SINGLE_CORES: f64 = 0.21;
/// A second counts as a spike when a non-harness process used more than this.
pub const SPIKE_CORES: f64 = 0.16;
/// Quotable only if spike seconds <= max(1, ceil(this * sampled seconds)).
pub const MAX_SPIKE_FRACTION: f64 = 0.02;
/// PSI `some avg10` ceiling for the BEFORE reading.
pub const MAX_PSI_BEFORE: f64 = 0.05;
// ---- end of threshold block ----

/// Fewer one-second samples than this is not quotable ("run too short").
pub const MIN_SAMPLES: usize = 3;

/// One row of the process table.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcStat {
    pub pid: u32,
    pub ppid: u32,
    pub name: String,
    /// Start time in clock ticks since boot.
    pub start: u64,
    /// utime + stime in clock ticks.
    pub ticks: u64,
}

/// Parse the text of `/proc/<pid>/stat`. The name sits between the first `(`
/// and the LAST `)` and may itself contain spaces and parentheses.
pub fn parse_stat(s: &str) -> Option<ProcStat> {
    let open = s.find('(')?;
    let close = s.rfind(')')?;
    if close < open {
        return None;
    }
    let pid: u32 = s[..open].trim().parse().ok()?;
    let name = s[open + 1..close].to_string();
    let f: Vec<&str> = s[close + 1..].split_whitespace().collect();
    // After the name: state(0) ppid(1) ... utime(11) stime(12) ... starttime(19).
    let ppid: u32 = f.get(1)?.parse().ok()?;
    let utime: u64 = f.get(11)?.parse().ok()?;
    let stime: u64 = f.get(12)?.parse().ok()?;
    let start: u64 = f.get(19)?.parse().ok()?;
    Some(ProcStat {
        pid,
        ppid,
        name,
        start,
        ticks: utime + stime,
    })
}

/// Read every readable `/proc/<pid>/stat`.
pub fn read_table() -> Vec<ProcStat> {
    let Ok(rd) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    rd.filter_map(|e| {
        let e = e.ok()?;
        e.file_name().to_str()?.parse::<u32>().ok()?;
        parse_stat(&std::fs::read_to_string(e.path().join("stat")).ok()?)
    })
    .collect()
}

/// Uptime in clock ticks (same unit as a process start time).
pub fn read_uptime_ticks() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/uptime").ok()?;
    let secs: f64 = s.split_whitespace().next()?.parse().ok()?;
    Some((secs * CLK_TCK) as u64)
}

/// `root` and every descendant of it in `table` (walks ppid).
pub fn descendants(table: &[ProcStat], root: u32) -> HashSet<u32> {
    let mut set: HashSet<u32> = HashSet::from([root]);
    loop {
        let before = set.len();
        for p in table {
            if set.contains(&p.ppid) {
                set.insert(p.pid);
            }
        }
        if set.len() == before {
            return set;
        }
    }
}

/// Result of a sampling run. `valid == false` is never quotable.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Summary {
    pub valid: bool,
    /// Why the summary is invalid (empty when valid).
    pub failure: String,
    pub samples: usize,
    pub seconds: f64,
    pub mean_total_cores: f64,
    pub max_single_cores: f64,
    pub spike_seconds: usize,
    /// Top offenders by total CPU seconds: (name truncated, cpu seconds).
    pub top3: Vec<(String, f64)>,
}

impl Summary {
    /// An invalid summary: NaN statistics, never quotable.
    pub fn failed(reason: &str) -> Self {
        Self {
            valid: false,
            failure: reason.to_string(),
            samples: 0,
            seconds: 0.0,
            mean_total_cores: f64::NAN,
            max_single_cores: f64::NAN,
            spike_seconds: 0,
            top3: Vec::new(),
        }
    }
}

/// One one-second sample of the non-harness load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub total_cores: f64,
    pub max_single_cores: f64,
}

type Key = (u32, u64);

/// Aggregates per-interval CPU deltas of non-harness processes.
#[derive(Default)]
pub struct Accumulator {
    prev: HashMap<Key, u64>,
    prev_uptime: u64,
    names: HashMap<Key, String>,
    totals: HashMap<Key, u64>,
    elapsed: f64,
    sum_ticks: u64,
    max_single: f64,
    spikes: usize,
    series: Vec<Sample>,
    invalid: Option<String>,
}

impl Accumulator {
    /// Mark the run invalid (first reason wins).
    pub fn mark_invalid(&mut self, reason: &str) {
        self.invalid.get_or_insert_with(|| reason.to_string());
    }

    fn check_table(&mut self, table: &[ProcStat], root: u32) {
        if !table.iter().any(|p| p.pid == root) {
            self.mark_invalid("process table incomplete (does not contain this process)");
        }
    }

    /// Record the baseline table (no interval is accounted).
    pub fn baseline(&mut self, table: &[ProcStat], root: u32, uptime_ticks: u64) {
        self.check_table(table, root);
        let own = descendants(table, root);
        self.prev = table
            .iter()
            .filter(|p| !own.contains(&p.pid))
            .map(|p| ((p.pid, p.start), p.ticks))
            .collect();
        self.prev_uptime = uptime_ticks;
    }

    /// Account one interval of `dt` seconds ending at `table` (taken at
    /// `uptime_ticks`). A process not in the previous sample is charged its
    /// whole CPU time only if it STARTED after the previous sample; otherwise
    /// it is a baseline (first sight of an old process, charge 0). A pid reused
    /// with a different start time is a different process.
    pub fn step(&mut self, table: &[ProcStat], root: u32, dt: f64, uptime_ticks: u64) {
        self.check_table(table, root);
        let own = descendants(table, root);
        let mut interval_max = 0.0f64;
        let mut interval_sum = 0u64;
        let mut cur = HashMap::new();
        for p in table.iter().filter(|p| !own.contains(&p.pid)) {
            let key = (p.pid, p.start);
            cur.insert(key, p.ticks);
            let delta = match self.prev.get(&key) {
                Some(t) => p.ticks.saturating_sub(*t),
                None if p.start > self.prev_uptime => p.ticks,
                None => 0,
            };
            if delta == 0 {
                continue;
            }
            interval_sum += delta;
            *self.totals.entry(key).or_insert(0) += delta;
            self.names.entry(key).or_insert_with(|| p.name.clone());
            interval_max = interval_max.max(delta as f64 / CLK_TCK / dt);
        }
        self.prev = cur;
        self.prev_uptime = uptime_ticks;
        self.elapsed += dt;
        self.sum_ticks += interval_sum;
        self.max_single = self.max_single.max(interval_max);
        if interval_max > SPIKE_CORES {
            self.spikes += 1;
        }
        self.series.push(Sample {
            total_cores: interval_sum as f64 / CLK_TCK / dt,
            max_single_cores: interval_max,
        });
    }

    /// The per-interval series.
    pub fn series(&self) -> &[Sample] {
        &self.series
    }

    /// Top `n` processes by total CPU seconds (names truncated to 15 chars).
    pub fn top_n(&self, n: usize) -> Vec<(String, f64)> {
        let mut tops: Vec<(String, f64)> = self
            .totals
            .iter()
            .map(|(k, t)| {
                let name: String = self.names[k].chars().take(15).collect();
                (name, *t as f64 / CLK_TCK)
            })
            .collect();
        tops.sort_by(|a, b| b.1.total_cmp(&a.1));
        tops.truncate(n);
        tops
    }

    pub fn summary(&self) -> Summary {
        if let Some(reason) = &self.invalid {
            return Summary::failed(reason);
        }
        if self.series.len() < MIN_SAMPLES {
            let mut s = Summary::failed("run too short for the sampler");
            s.samples = self.series.len();
            s.seconds = self.elapsed;
            return s;
        }
        Summary {
            valid: true,
            failure: String::new(),
            samples: self.series.len(),
            seconds: self.elapsed,
            mean_total_cores: self.sum_ticks as f64 / CLK_TCK / self.elapsed,
            max_single_cores: self.max_single,
            spike_seconds: self.spikes,
            top3: self.top_n(3),
        }
    }
}

/// Spike seconds tolerated in a run of `seconds`: max(1, ceil(fraction * s)).
pub fn allowed_spike_seconds(seconds: f64) -> usize {
    ((MAX_SPIKE_FRACTION * seconds - 1e-9).ceil().max(1.0)) as usize
}

/// Verdict of the sampler alone: an invalid summary is never quotable.
pub fn sampler_quotable(s: &Summary) -> bool {
    s.valid
        && s.mean_total_cores <= MAX_MEAN_CORES
        && s.max_single_cores <= MAX_SINGLE_CORES
        && s.spike_seconds <= allowed_spike_seconds(s.seconds)
}

/// `/proc/pressure/cpu` `some avg10`, NaN if unreadable.
pub fn psi_cpu_some_avg10() -> f64 {
    std::fs::read_to_string("/proc/pressure/cpu")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("some"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kv| kv.strip_prefix("avg10="))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(f64::NAN)
}

/// Background sampler: one sample per second until `finish`.
pub struct Sampler {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<Accumulator>,
}

fn sample_once(acc: &mut Accumulator, me: u32, dt: f64) {
    match read_uptime_ticks() {
        Some(up) => acc.step(&read_table(), me, dt, up),
        None => acc.mark_invalid("/proc/uptime unreadable"),
    }
}

impl Sampler {
    pub fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            let me = std::process::id();
            let mut acc = Accumulator::default();
            match read_uptime_ticks() {
                Some(up) => acc.baseline(&read_table(), me, up),
                None => acc.mark_invalid("/proc/uptime unreadable"),
            }
            let mut last = Instant::now();
            while !flag.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(50));
                let dt = last.elapsed().as_secs_f64();
                if dt >= 1.0 {
                    sample_once(&mut acc, me, dt);
                    last = Instant::now();
                }
            }
            let dt = last.elapsed().as_secs_f64();
            if dt >= 0.25 {
                sample_once(&mut acc, me, dt);
            }
            acc
        });
        Self { stop, handle }
    }

    /// Stop and return the accumulator; `None` if the thread panicked.
    pub fn finish_acc(self) -> Option<Accumulator> {
        self.stop.store(true, Ordering::Relaxed);
        self.handle.join().ok()
    }

    /// Stop and summarise; a panicked thread gives an invalid summary.
    pub fn finish(self) -> Summary {
        summary_of(self.finish_acc())
    }
}

/// Summary of an optional accumulator (`None` = the sampler thread panicked).
pub fn summary_of(acc: Option<Accumulator>) -> Summary {
    match acc {
        Some(a) => a.summary(),
        None => Summary::failed("sampler thread panicked"),
    }
}

/// Final verdict: PSI before within the ceiling AND the sampler within its
/// thresholds.
pub fn quotable(psi_before: f64, s: &Summary) -> bool {
    psi_before.is_finite() && psi_before <= MAX_PSI_BEFORE && sampler_quotable(s)
}

/// The sampler verdict line.
pub fn verdict_line(s: &Summary) -> String {
    let provisional = if CALIBRATED {
        ""
    } else {
        "; thresholds provisional: not yet calibrated on the sampler's own output"
    };
    if !s.valid {
        return format!(
            "quotable by external-load sampler: no (sampler failed: {}; {} samples){provisional}",
            s.failure, s.samples
        );
    }
    format!(
        "quotable by external-load sampler: {} (mean {:.3} cores, max {:.3} cores, {} seconds above {SPIKE_CORES}); \
         thresholds mean<={MAX_MEAN_CORES}, max<={MAX_SINGLE_CORES}, seconds above<={}{provisional}",
        if sampler_quotable(s) { "yes" } else { "no" },
        s.mean_total_cores,
        s.max_single_cores,
        s.spike_seconds,
        allowed_spike_seconds(s.seconds)
    )
}

/// Print the PSI readings (informational), the sampler summary and its
/// verdict. Returns the combined verdict; the caller prints its own
/// `NOT QUOTABLE: box contested` wording when this is false.
pub fn print_report(psi_before: f64, psi_after: f64, s: &Summary) -> bool {
    println!(
        "PSI cpu some avg10 before = {psi_before:.2}, after = {psi_after:.2} \
         (informational: includes this harness's own load; only `before` is a criterion)"
    );
    let top: Vec<String> = s
        .top3
        .iter()
        .map(|(n, c)| format!("{n} {c:.2} cpu-s"))
        .collect();
    println!(
        "external-load sampler: {:.0} s, top offenders: {}",
        s.seconds,
        if top.is_empty() {
            "none".to_string()
        } else {
            top.join(", ")
        }
    );
    println!("{}", verdict_line(s));
    quotable(psi_before, s)
}

/// Nearest-rank percentile of an ascending slice (NaN when empty).
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (q * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

/// Single-process thresholds (cores) the calibration counts seconds above.
pub const CALIBRATION_LEVELS: [f64; 4] = [0.10, 0.16, 0.21, 0.30];

/// Calibration report: the printed lines, ending with the single
/// machine-readable `CALIBRATION ...` line.
pub fn calibration_lines(
    s: &Summary,
    top: &[(String, f64)],
    series: &[Sample],
    own_cpu_s: f64,
    wall_s: f64,
) -> Vec<String> {
    let n = series.len();
    let sorted = |f: fn(&Sample) -> f64| {
        let mut v: Vec<f64> = series.iter().map(f).collect();
        v.sort_by(|a, b| a.total_cmp(b));
        v
    };
    let tot = sorted(|x| x.total_cores);
    let single = sorted(|x| x.max_single_cores);
    let mut out = vec![format!(
        "summary: valid={} samples={} mean {:.4} cores, max single {:.4} cores ({})",
        s.valid,
        s.samples,
        s.mean_total_cores,
        s.max_single_cores,
        if s.valid { "ok" } else { s.failure.as_str() }
    )];
    let tops: Vec<String> = top
        .iter()
        .map(|(name, c)| format!("{name} {c:.2} cpu-s"))
        .collect();
    out.push(format!("top offenders: {}", tops.join(", ")));
    for (label, v) in [("total cores", &tot), ("max-single cores", &single)] {
        out.push(format!(
            "per-second {label}: p50 {:.3} p95 {:.3} p99 {:.3} max {:.3}",
            percentile(v, 0.5),
            percentile(v, 0.95),
            percentile(v, 0.99),
            percentile(v, 1.0)
        ));
    }
    let mut counts = Vec::new();
    for level in CALIBRATION_LEVELS {
        let c = series.iter().filter(|x| x.max_single_cores > level).count();
        out.push(format!(
            "seconds with a process above {level:.2} cores: {c} of {n} ({:.3})",
            c as f64 / n.max(1) as f64
        ));
        counts.push(c);
    }
    out.push(format!(
        "sampler cost (whole harness process, no workload): {:.4} cores ({own_cpu_s:.2} cpu-s in {wall_s:.0} s)",
        own_cpu_s / wall_s.max(1e-9)
    ));
    out.push(format!(
        "CALIBRATION mean={:.4} max={:.4} spikes010={} spikes016={} spikes021={} spikes030={} secs={n}",
        s.mean_total_cores, s.max_single_cores, counts[0], counts[1], counts[2], counts[3]
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p_at(pid: u32, ppid: u32, name: &str, start: u64, ticks: u64) -> ProcStat {
        ProcStat {
            pid,
            ppid,
            name: name.into(),
            start,
            ticks,
        }
    }

    fn p(pid: u32, ppid: u32, name: &str, ticks: u64) -> ProcStat {
        p_at(pid, ppid, name, 1, ticks)
    }

    fn valid_summary() -> Summary {
        Summary {
            valid: true,
            failure: String::new(),
            samples: 60,
            seconds: 60.0,
            mean_total_cores: 0.2,
            max_single_cores: 0.1,
            spike_seconds: 0,
            top3: vec![],
        }
    }

    /// An accumulator with `n` empty samples taken from a table of just `me`.
    fn idle_acc(n: usize) -> Accumulator {
        let t = [p(10, 1, "harness", 0)];
        let mut a = Accumulator::default();
        a.baseline(&t, 10, 1000);
        for i in 0..n {
            a.step(&t, 10, 1.0, 1100 + 100 * i as u64);
        }
        a
    }

    #[test]
    fn quiet_parse_stat_plain_and_tricky_names() {
        let tail =
            "S 7 1 1 0 -1 4194560 100 0 0 0 30 12 0 0 20 0 1 0 555 1000 10 18446744073709551615";
        let s = parse_stat(&format!("42 (bash) {tail}")).unwrap();
        assert_eq!((s.pid, s.ppid, s.ticks, s.start), (42, 7, 42, 555));
        assert_eq!(s.name, "bash");
        let s = parse_stat(&format!("43 (Web Content (x) y) {tail}")).unwrap();
        assert_eq!(
            (s.pid, s.name.as_str(), s.ticks),
            (43, "Web Content (x) y", 42)
        );
        assert!(parse_stat("garbage").is_none());
        assert!(parse_stat("1 (x) S 2").is_none());
    }

    #[test]
    fn quiet_descendants_exclude_the_whole_subtree() {
        let t = vec![
            p(1, 0, "init", 0),
            p(10, 1, "harness", 0),
            p(11, 10, "arm", 0),
            p(12, 11, "grandarm", 0),
            p(20, 1, "other", 0),
            p(21, 20, "otherchild", 0),
        ];
        assert_eq!(descendants(&t, 10), HashSet::from([10, 11, 12]));
        // Order independence: child listed before its parent.
        let t2 = vec![p(12, 11, "g", 0), p(11, 10, "c", 0), p(10, 1, "h", 0)];
        assert_eq!(descendants(&t2, 10), HashSet::from([10, 11, 12]));
    }

    #[test]
    fn quiet_accumulator_deltas_exclusion_and_top3() {
        let mut a = Accumulator::default();
        a.baseline(
            &[
                p(10, 1, "harness", 0),
                p(11, 10, "arm", 0),
                p(20, 1, "svc", 100),
                p(21, 1, "idle", 5),
            ],
            10,
            1000,
        );
        // Harness and child burn 600 ticks (excluded); svc 20 ticks.
        a.step(
            &[
                p(10, 1, "harness", 300),
                p(11, 10, "arm", 300),
                p(20, 1, "svc", 120),
                p(21, 1, "idle", 5),
            ],
            10,
            1.0,
            1100,
        );
        // A process born inside the interval (start 1150 > 1100) is charged in
        // full: 30 ticks.
        a.step(
            &[
                p(10, 1, "harness", 600),
                p(20, 1, "svc", 120),
                p(21, 1, "idle", 5),
                p_at(30, 1, "a-very-long-process-name", 1150, 30),
            ],
            10,
            1.0,
            1200,
        );
        a.step(&[p(10, 1, "harness", 600)], 10, 1.0, 1300);
        let s = a.summary();
        assert!(s.valid, "{}", s.failure);
        assert_eq!(s.seconds, 3.0);
        assert!((s.mean_total_cores - 50.0 / 100.0 / 3.0).abs() < 1e-12);
        assert!((s.max_single_cores - 0.30).abs() < 1e-12);
        assert_eq!(s.spike_seconds, 2); // 0.2 and 0.3 both exceed SPIKE_CORES
        assert_eq!(s.top3[0], ("a-very-long-pro".to_string(), 0.30));
        assert_eq!(s.top3[1].0, "svc");
        assert_eq!(s.top3.len(), 2);
        assert_eq!(a.series().len(), 3);
    }

    #[test]
    fn quiet_first_seen_old_process_is_a_baseline_not_a_charge() {
        let mut a = Accumulator::default();
        a.baseline(&[p(10, 1, "harness", 0)], 10, 1000);
        // `old` started long before the previous sample but was not in it
        // (unreadable then): first sight is a baseline, charge 0. `young`
        // started after the previous sample: charged in full.
        a.step(
            &[
                p(10, 1, "harness", 0),
                p_at(20, 1, "old", 500, 9000),
                p_at(21, 1, "young", 1050, 40),
            ],
            10,
            1.0,
            1100,
        );
        assert_eq!(a.top_n(5), vec![("young".to_string(), 0.40)]);
        // Next interval: old now has a baseline and is charged only its delta.
        a.step(
            &[
                p(10, 1, "harness", 0),
                p_at(20, 1, "old", 500, 9010),
                p_at(21, 1, "young", 1050, 40),
            ],
            10,
            1.0,
            1200,
        );
        let top = a.top_n(5);
        assert_eq!(top[0], ("young".to_string(), 0.40));
        assert_eq!(top[1], ("old".to_string(), 0.10));
    }

    #[test]
    fn quiet_reused_pid_with_a_new_start_time_is_a_new_process() {
        let mut a = Accumulator::default();
        a.baseline(
            &[p(10, 1, "harness", 0), p_at(20, 1, "first", 100, 5000)],
            10,
            1000,
        );
        // pid 20 is reused by a process that started at 1050 with 30 ticks: not
        // the 5000-tick process (a keyed-by-pid delta would saturate to 0).
        a.step(
            &[p(10, 1, "harness", 0), p_at(20, 1, "second", 1050, 30)],
            10,
            1.0,
            1100,
        );
        assert_eq!(a.top_n(5), vec![("second".to_string(), 0.30)]);
        // The same pid and start time keeps its identity (delta only).
        a.step(
            &[p(10, 1, "harness", 0), p_at(20, 1, "second", 1050, 35)],
            10,
            1.0,
            1200,
        );
        assert_eq!(a.top_n(5), vec![("second".to_string(), 0.35)]);
    }

    #[test]
    fn quiet_verdict_thresholds() {
        let ok = valid_summary();
        assert!(sampler_quotable(&ok));
        assert!(quotable(0.0, &ok));
        assert!(!quotable(0.06, &ok));
        assert!(!quotable(f64::NAN, &ok));
        let mean = Summary {
            mean_total_cores: 0.3,
            ..ok.clone()
        };
        assert!(!sampler_quotable(&mean));
        let single = Summary {
            max_single_cores: 0.5,
            ..ok.clone()
        };
        assert!(!sampler_quotable(&single));
        // 60 s run: allowed = max(1, ceil(0.02 * 60)) = 2.
        let two = Summary {
            spike_seconds: 2,
            ..ok.clone()
        };
        assert!(sampler_quotable(&two));
        let three = Summary {
            spike_seconds: 3,
            ..ok.clone()
        };
        assert!(!sampler_quotable(&three));
    }

    #[test]
    fn quiet_spike_tolerance_scales_with_the_run() {
        assert_eq!(allowed_spike_seconds(40.0), 1);
        assert_eq!(allowed_spike_seconds(300.0), 6);
        assert_eq!(allowed_spike_seconds(800.0), 16);
        assert_eq!(allowed_spike_seconds(3.0), 1);
    }

    #[test]
    fn quiet_fails_closed() {
        // Default and explicit failure summaries are never quotable.
        assert!(!sampler_quotable(&Summary::default()));
        assert!(!quotable(0.0, &Summary::failed("x")));
        // A panicked sampler thread.
        let s = summary_of(None);
        assert!(!s.valid && s.failure.contains("panicked") && s.mean_total_cores.is_nan());
        assert!(!sampler_quotable(&s));
        // Empty process table, and one without this process.
        let mut a = Accumulator::default();
        a.baseline(&[], 10, 1000);
        for i in 0..5 {
            a.step(&[], 10, 1.0, 1100 + i);
        }
        assert!(!a.summary().valid);
        let mut a = Accumulator::default();
        a.baseline(&[p(11, 1, "other", 0)], 10, 1000);
        for i in 0..5 {
            a.step(&[p(11, 1, "other", 0)], 10, 1.0, 1100 + i);
        }
        let s = a.summary();
        assert!(!s.valid && s.failure.contains("incomplete"));
        // Unreadable uptime.
        let mut a = idle_acc(5);
        a.mark_invalid("/proc/uptime unreadable");
        assert!(!a.summary().valid);
        // Too short: 2 samples invalid, 3 valid.
        let s = idle_acc(2).summary();
        assert!(!s.valid && s.failure == "run too short for the sampler");
        assert!(!sampler_quotable(&s));
        assert!(sampler_quotable(&idle_acc(3).summary()));
        // The verdict line names the failure and the provisional status.
        let line = verdict_line(&s);
        assert!(line.contains("run too short for the sampler"), "{line}");
        assert!(line.contains("thresholds provisional"), "{line}");
        assert!(verdict_line(&valid_summary()).contains("thresholds provisional"));
    }

    #[test]
    fn quiet_percentile_nearest_rank() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&v, 0.5), 50.0);
        assert_eq!(percentile(&v, 0.95), 95.0);
        assert_eq!(percentile(&v, 0.99), 99.0);
        assert_eq!(percentile(&v, 1.0), 100.0);
        assert_eq!(percentile(&[7.0], 0.5), 7.0);
        assert!(percentile(&[], 0.5).is_nan());
    }

    #[test]
    fn quiet_calibration_lines_format() {
        let series: Vec<Sample> = [0.0, 0.05, 0.12, 0.25, 0.35]
            .iter()
            .map(|&m| Sample {
                total_cores: m + 0.1,
                max_single_cores: m,
            })
            .collect();
        let s = valid_summary();
        let lines = calibration_lines(&s, &[("svc".to_string(), 1.5)], &series, 0.5, 5.0);
        let last = lines.last().unwrap();
        assert_eq!(
            last,
            "CALIBRATION mean=0.2000 max=0.1000 spikes010=3 spikes016=2 spikes021=2 spikes030=1 secs=5"
        );
        assert!(lines.iter().any(|l| l.contains("p50 0.220 p95 0.450")));
        assert!(lines
            .iter()
            .any(|l| l.contains("above 0.16 cores: 2 of 5 (0.400)")));
        assert!(lines.iter().any(|l| l.contains("svc 1.50 cpu-s")));
        assert!(lines
            .iter()
            .any(|l| l.contains("0.1000 cores (0.50 cpu-s in 5 s)")));
    }

    #[test]
    fn quiet_live_table_contains_this_process() {
        let t = read_table();
        let me = std::process::id();
        assert!(t.iter().any(|p| p.pid == me));
        assert!(descendants(&t, me).contains(&me));
        assert!(read_uptime_ticks().is_some());
    }
}
