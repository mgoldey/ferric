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
//! ## Thresholds (derived from logged runs)
//!
//! Data: the old external-contender sampler (`ps`, processes above 3% CPU once
//! a second), persistent processes only (seen in >= 10 samples; `ps` %CPU is a
//! lifetime average, so spike magnitudes are smoothed and short-lived
//! processes read as hundreds of percent and are excluded).
//!
//! Known-clean runs (crossover reruns 1 and 3, three morning crossovers, the
//! mixed sweep): background total 0.09-0.20 cores (mean per run 0.195, 0.175, 0.095,
//! 0.095, 0.095, 0.093), largest single persistent process 0.084 cores
//! (htop), and 0 seconds with any persistent process above 0.16 cores.
//! Known-contested run (crossover rerun 2, a new claude session at 23:54:46):
//! mean 0.366 cores, a single-process spike of 0.51 cores (0.30 at its lowest
//! plateau), and 19 of 58 seconds with a persistent process above 0.16 cores.
//! (The morning ccsd run was logged "clean" but its log shows a flatpak update
//! worker and mint-refresh-cache above 0.16 cores in 21 of 814 seconds: real
//! contention, which this criterion is expected to flag.)
//!
//! Each threshold is the geometric midpoint of the worst clean value and the
//! mildest contested value:
//! - mean total: sqrt(0.195 * 0.366) = 0.267, rounded to 0.27 cores.
//! - one process in one second: sqrt(0.084 * 0.51) = 0.207, rounded to 0.21
//!   cores.
//! - "spike" second (any process above): sqrt(0.084 * 0.30) = 0.159, rounded
//!   to 0.16 cores. Clean runs show 0 such seconds and the contested run 19,
//!   so one transient second is tolerated.
//!
//! The ps data is smoothed and the clean sample is 6 runs, so these are
//! derived, not calibrated; the sampler reads per-second deltas, which are
//! sharper than the logged lifetime averages.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// `USER_HZ` on Linux (clock ticks per second in `/proc/<pid>/stat`).
pub const CLK_TCK: f64 = 100.0;
/// Quotable only if the mean total non-harness CPU is at most this (cores).
pub const MAX_MEAN_CORES: f64 = 0.27;
/// Quotable only if no single process used more than this in any second.
pub const MAX_SINGLE_CORES: f64 = 0.21;
/// A second counts as a spike when a non-harness process used more than this.
pub const SPIKE_CORES: f64 = 0.16;
/// Quotable only if at most this many seconds were spikes.
pub const MAX_SPIKE_SECONDS: usize = 1;
/// PSI `some avg10` ceiling for the BEFORE reading.
pub const MAX_PSI_BEFORE: f64 = 0.05;

/// One row of the process table.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcStat {
    pub pid: u32,
    pub ppid: u32,
    pub name: String,
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

/// Result of a sampling run.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Summary {
    pub seconds: f64,
    pub mean_total_cores: f64,
    pub max_single_cores: f64,
    pub spike_seconds: usize,
    /// Top offenders by total CPU seconds: (name truncated, cpu seconds).
    pub top3: Vec<(String, f64)>,
}

type Key = (u32, u64);

/// Aggregates per-interval CPU deltas of non-harness processes.
#[derive(Default)]
pub struct Accumulator {
    prev: HashMap<Key, u64>,
    names: HashMap<Key, String>,
    totals: HashMap<Key, u64>,
    elapsed: f64,
    sum_ticks: u64,
    max_single: f64,
    spikes: usize,
}

impl Accumulator {
    /// Record the baseline table (no interval is accounted).
    pub fn baseline(&mut self, table: &[ProcStat], root: u32) {
        let own = descendants(table, root);
        self.prev = table
            .iter()
            .filter(|p| !own.contains(&p.pid))
            .map(|p| ((p.pid, p.start), p.ticks))
            .collect();
    }

    /// Account one interval of `dt` seconds ending at `table`. A process first
    /// seen in this interval was born inside it and is charged its whole CPU
    /// time.
    pub fn step(&mut self, table: &[ProcStat], root: u32, dt: f64) {
        let own = descendants(table, root);
        let mut interval_max = 0.0f64;
        let mut cur = HashMap::new();
        for p in table.iter().filter(|p| !own.contains(&p.pid)) {
            let key = (p.pid, p.start);
            cur.insert(key, p.ticks);
            let delta = p
                .ticks
                .saturating_sub(self.prev.get(&key).copied().unwrap_or(0));
            if delta == 0 {
                continue;
            }
            self.sum_ticks += delta;
            *self.totals.entry(key).or_insert(0) += delta;
            self.names.entry(key).or_insert_with(|| p.name.clone());
            interval_max = interval_max.max(delta as f64 / CLK_TCK / dt);
        }
        self.prev = cur;
        self.elapsed += dt;
        self.max_single = self.max_single.max(interval_max);
        if interval_max > SPIKE_CORES {
            self.spikes += 1;
        }
    }

    pub fn summary(&self) -> Summary {
        let mut tops: Vec<(String, f64)> = self
            .totals
            .iter()
            .map(|(k, t)| {
                let n: String = self.names[k].chars().take(15).collect();
                (n, *t as f64 / CLK_TCK)
            })
            .collect();
        tops.sort_by(|a, b| b.1.total_cmp(&a.1));
        tops.truncate(3);
        Summary {
            seconds: self.elapsed,
            mean_total_cores: if self.elapsed > 0.0 {
                self.sum_ticks as f64 / CLK_TCK / self.elapsed
            } else {
                f64::NAN
            },
            max_single_cores: self.max_single,
            spike_seconds: self.spikes,
            top3: tops,
        }
    }
}

/// Verdict of the sampler alone (NaN, i.e. nothing sampled, is not quotable).
pub fn sampler_quotable(s: &Summary) -> bool {
    s.mean_total_cores <= MAX_MEAN_CORES
        && s.max_single_cores <= MAX_SINGLE_CORES
        && s.spike_seconds <= MAX_SPIKE_SECONDS
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
    handle: JoinHandle<Summary>,
}

impl Sampler {
    pub fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            let me = std::process::id();
            let mut acc = Accumulator::default();
            acc.baseline(&read_table(), me);
            let mut last = Instant::now();
            while !flag.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(50));
                let dt = last.elapsed().as_secs_f64();
                if dt >= 1.0 {
                    acc.step(&read_table(), me, dt);
                    last = Instant::now();
                }
            }
            let dt = last.elapsed().as_secs_f64();
            if dt >= 0.25 {
                acc.step(&read_table(), me, dt);
            }
            acc.summary()
        });
        Self { stop, handle }
    }

    pub fn finish(self) -> Summary {
        self.stop.store(true, Ordering::Relaxed);
        self.handle.join().unwrap_or_default()
    }
}

/// Final verdict: PSI before within the ceiling AND the sampler within its
/// thresholds.
pub fn quotable(psi_before: f64, s: &Summary) -> bool {
    psi_before.is_finite() && psi_before <= MAX_PSI_BEFORE && sampler_quotable(s)
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
    println!(
        "quotable by external-load sampler: {} (mean {:.3} cores, max {:.3} cores, {} seconds above {SPIKE_CORES}); thresholds mean<={MAX_MEAN_CORES}, max<={MAX_SINGLE_CORES}, seconds above<={MAX_SPIKE_SECONDS}",
        if sampler_quotable(s) { "yes" } else { "no" },
        s.mean_total_cores,
        s.max_single_cores,
        s.spike_seconds
    );
    quotable(psi_before, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(pid: u32, ppid: u32, name: &str, ticks: u64) -> ProcStat {
        ProcStat {
            pid,
            ppid,
            name: name.into(),
            start: 1,
            ticks,
        }
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
        );
        // A process born inside the interval is charged in full: 30 ticks.
        a.step(
            &[
                p(10, 1, "harness", 600),
                p(20, 1, "svc", 120),
                p(21, 1, "idle", 5),
                p(30, 1, "a-very-long-process-name", 30),
            ],
            10,
            1.0,
        );
        let s = a.summary();
        assert_eq!(s.seconds, 2.0);
        assert!((s.mean_total_cores - 0.25).abs() < 1e-12); // 50 ticks / 100 / 2 s
        assert!((s.max_single_cores - 0.30).abs() < 1e-12);
        assert_eq!(s.spike_seconds, 2); // 0.2 and 0.3 both exceed SPIKE_CORES
        assert_eq!(s.top3[0], ("a-very-long-pro".to_string(), 0.30));
        assert_eq!(s.top3[1].0, "svc");
        assert_eq!(s.top3.len(), 2);
    }

    #[test]
    fn quiet_verdict_thresholds() {
        let ok = Summary {
            seconds: 60.0,
            mean_total_cores: 0.2,
            max_single_cores: 0.1,
            spike_seconds: 0,
            top3: vec![],
        };
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
        let spikes = Summary {
            spike_seconds: 2,
            ..ok.clone()
        };
        assert!(!sampler_quotable(&spikes));
        let one = Summary {
            spike_seconds: 1,
            ..ok.clone()
        };
        assert!(sampler_quotable(&one));
        let nan = Summary {
            mean_total_cores: f64::NAN,
            ..ok
        };
        assert!(!sampler_quotable(&nan));
    }

    #[test]
    fn quiet_live_table_contains_this_process() {
        let t = read_table();
        let me = std::process::id();
        assert!(t.iter().any(|p| p.pid == me));
        assert!(descendants(&t, me).contains(&me));
    }
}
