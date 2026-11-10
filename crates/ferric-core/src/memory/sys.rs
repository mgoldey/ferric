//! Per-platform probes for system memory, this process's resident memory, and
//! its CPU time.
//!
//! Every probe returns `None` when the platform cannot answer; callers decide
//! what a missing reading means. The budget resolver treats a missing
//! available-RAM reading as a reason to fall back LOUDLY (see
//! [`crate::memory::resolve_budget`]); the RSS readers are observability and
//! degrade to "unknown".
//!
//! | probe                     | Linux                              | macOS                                         | Windows                                   |
//! |---------------------------|------------------------------------|-----------------------------------------------|-------------------------------------------|
//! | [`available_ram_bytes`](crate::memory::sys::available_ram_bytes)   | `/proc/meminfo` `MemAvailable`     | `host_statistics64`: (free − speculative + inactive) pages, capped at `hw.memsize` | `GlobalMemoryStatusEx` `ullAvailPhys` |
//! | [`total_ram_bytes`](crate::memory::sys::total_ram_bytes)       | `/proc/meminfo` `MemTotal`         | `sysctl hw.memsize`                           | `GlobalMemoryStatusEx` `ullTotalPhys`     |
//! | [`own_rss_bytes`](crate::memory::sys::own_rss_bytes)         | `/proc/self/status` `VmRSS`        | `proc_pidinfo(PROC_PIDTASKINFO)` resident size | `K32GetProcessMemoryInfo` `WorkingSetSize` |
//! | [`own_peak_rss_bytes`](crate::memory::sys::own_peak_rss_bytes)    | `/proc/self/status` `VmHWM`        | `getrusage` `ru_maxrss` (BYTES on macOS)      | `K32GetProcessMemoryInfo` `PeakWorkingSetSize` |
//! | [`own_cpu_seconds`](crate::memory::sys::own_cpu_seconds)       | `/proc/self/stat` utime + stime    | `getrusage` `ru_utime + ru_stime`             | `GetProcessTimes` kernel + user           |
//!
//! The cgroup limit stays a Linux-only input to the budget (see
//! `detect_cgroup_limit_bytes` in the parent module); there is no macOS
//! equivalent, and Windows job-object limits are not read.
//!
//! The macOS "available" figure follows psutil's `virtual_memory().available`
//! convention (inactive + free, where `free_count` already counts the
//! speculative pages, which are subtracted). It is an estimate of what can be
//! handed out without paging, the same role `MemAvailable` plays on Linux.

/// Units `getrusage`'s `ru_maxrss` is reported in. The field is KILOBYTES on
/// Linux and the BSDs but BYTES on macOS, so a peak read on the wrong
/// assumption is off by 1024x.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuMaxrssUnit {
    /// Kibibytes (Linux, FreeBSD, ...).
    Kibibytes,
    /// Bytes (macOS / iOS).
    Bytes,
}

/// The `ru_maxrss` unit of the platform this crate was compiled for.
#[must_use]
pub const fn native_ru_maxrss_unit() -> RuMaxrssUnit {
    if cfg!(any(target_os = "macos", target_os = "ios")) {
        RuMaxrssUnit::Bytes
    } else {
        RuMaxrssUnit::Kibibytes
    }
}

/// Convert a raw `ru_maxrss` value to bytes. `None` for a non-positive value
/// (the kernel did not report one); saturates instead of overflowing.
#[must_use]
pub fn ru_maxrss_to_bytes(raw: i64, unit: RuMaxrssUnit) -> Option<usize> {
    if raw <= 0 {
        return None;
    }
    let raw = raw as u64;
    let bytes = match unit {
        RuMaxrssUnit::Kibibytes => raw.saturating_mul(1024),
        RuMaxrssUnit::Bytes => raw,
    };
    Some(clamp_u64(bytes))
}

/// The macOS available-memory estimate from `vm_statistics64` page counts:
/// `(free − speculative + inactive) × page_size`, capped at `total_bytes`
/// when that is known. `None` when the page size is zero.
#[must_use]
pub fn macos_available_bytes(
    free_pages: u64,
    speculative_pages: u64,
    inactive_pages: u64,
    page_size: u64,
    total_bytes: Option<u64>,
) -> Option<usize> {
    if page_size == 0 {
        return None;
    }
    let pages = free_pages
        .saturating_sub(speculative_pages)
        .saturating_add(inactive_pages);
    let bytes = pages.saturating_mul(page_size);
    let bytes = total_bytes.map_or(bytes, |t| bytes.min(t));
    Some(clamp_u64(bytes))
}

/// A Windows `FILETIME` duration (100-nanosecond ticks split into two 32-bit
/// halves) in seconds.
#[must_use]
pub fn filetime_ticks_to_seconds(high: u32, low: u32) -> f64 {
    let ticks = (u64::from(high) << 32) | u64::from(low);
    ticks as f64 * 1e-7
}

/// Parse a `/proc/<pid>/status`-shaped string for a `<field>:  <n> kB` line,
/// returning bytes. `None` on a missing field or malformed number.
#[must_use]
pub fn parse_status_kb_field(contents: &str, field: &str) -> Option<usize> {
    for line in contents.lines() {
        if let Some(rest) = line
            .strip_prefix(field)
            .and_then(|r| r.strip_prefix(':'))
        {
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(clamp_u64(kb.saturating_mul(1024)));
        }
    }
    None
}

/// Parse `/proc/self/stat` for the process's user + system CPU seconds
/// (fields 14 and 15, in USER_HZ = 100 ticks per second).
#[must_use]
pub fn parse_proc_stat_cpu_seconds(contents: &str) -> Option<f64> {
    // The comm field may itself contain spaces and parentheses, so split at
    // the LAST ')'. `rest` then starts at field 3 (state), which puts utime
    // and stime at indices 11 and 12.
    let rest = contents.rsplit_once(')')?.1;
    let f: Vec<&str> = rest.split_whitespace().collect();
    let utime = f.get(11)?.parse::<u64>().ok()?;
    let stime = f.get(12)?.parse::<u64>().ok()?;
    // USER_HZ is 100 on every Linux target ferric builds for.
    Some((utime + stime) as f64 / 100.0)
}

fn clamp_u64(v: u64) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

/// Memory the system can hand out without paging, in bytes (the
/// `MemAvailable` role). Does NOT include any cgroup limit.
#[must_use]
pub fn available_ram_bytes() -> Option<usize> {
    imp::available_ram_bytes()
}

/// Physical RAM installed, in bytes.
#[must_use]
pub fn total_ram_bytes() -> Option<usize> {
    imp::total_ram_bytes()
}

/// This process's current resident set size, in bytes.
#[must_use]
pub fn own_rss_bytes() -> Option<usize> {
    imp::own_rss_bytes()
}

/// This process's peak resident set size so far, in bytes.
#[must_use]
pub fn own_peak_rss_bytes() -> Option<usize> {
    imp::own_peak_rss_bytes()
}

/// This process's total (user + system) CPU time, in seconds.
#[must_use]
pub fn own_cpu_seconds() -> Option<f64> {
    imp::own_cpu_seconds()
}

#[cfg(target_os = "linux")]
mod imp {
    use super::{parse_proc_stat_cpu_seconds, parse_status_kb_field};

    fn read(path: &str) -> Option<String> {
        std::fs::read_to_string(path).ok()
    }

    pub(super) fn available_ram_bytes() -> Option<usize> {
        read("/proc/meminfo").and_then(|s| crate::memory::parse_meminfo_available(&s))
    }

    pub(super) fn total_ram_bytes() -> Option<usize> {
        read("/proc/meminfo").and_then(|s| parse_status_kb_field(&s, "MemTotal"))
    }

    pub(super) fn own_rss_bytes() -> Option<usize> {
        read("/proc/self/status").and_then(|s| crate::memory::parse_vm_rss_bytes(&s))
    }

    pub(super) fn own_peak_rss_bytes() -> Option<usize> {
        read("/proc/self/status").and_then(|s| parse_status_kb_field(&s, "VmHWM"))
    }

    pub(super) fn own_cpu_seconds() -> Option<f64> {
        read("/proc/self/stat").and_then(|s| parse_proc_stat_cpu_seconds(&s))
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use super::{macos_available_bytes, native_ru_maxrss_unit, ru_maxrss_to_bytes};
    use std::ffi::{c_int, c_void, CStr};
    use std::sync::OnceLock;

    fn sysctl_u64(name: &CStr) -> Option<u64> {
        let mut value: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: `value` is a live u64 and `len` holds its size; sysctlbyname
        // writes at most `len` bytes and reports the written length back.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&raw mut value).cast::<c_void>(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0 && len == std::mem::size_of::<u64>()).then_some(value)
    }

    fn rusage_self() -> Option<libc::rusage> {
        // SAFETY: an all-zero rusage is a valid value; getrusage fills it.
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: `ru` is a valid, writable rusage.
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
        (rc == 0).then_some(ru)
    }

    /// The host port, fetched once: every `mach_host_self` call adds a user
    /// reference to the send right, so caching it avoids leaking one per probe.
    fn host_port() -> libc::mach_port_t {
        static HOST: OnceLock<libc::mach_port_t> = OnceLock::new();
        // SAFETY: mach_host_self has no preconditions. libc marks it
        // deprecated only in favour of the `mach2` crate; the call is stable.
        #[allow(deprecated)]
        *HOST.get_or_init(|| unsafe { libc::mach_host_self() })
    }

    pub(super) fn total_ram_bytes() -> Option<usize> {
        sysctl_u64(c"hw.memsize").map(|b| usize::try_from(b).unwrap_or(usize::MAX))
    }

    pub(super) fn available_ram_bytes() -> Option<usize> {
        // SAFETY: an all-zero vm_statistics64 is a valid value.
        let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        let mut count = libc::HOST_VM_INFO64_COUNT;
        // SAFETY: `stats` is HOST_VM_INFO64_COUNT integers long, as `count` says.
        let kr = unsafe {
            libc::host_statistics64(
                host_port(),
                libc::HOST_VM_INFO64,
                (&raw mut stats).cast::<libc::integer_t>(),
                &mut count,
            )
        };
        if kr != libc::KERN_SUCCESS {
            return None;
        }
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page = u64::try_from(page).ok()?;
        // Packed struct: copy each field out by value.
        let free = u64::from(stats.free_count);
        let spec = u64::from(stats.speculative_count);
        let inactive = u64::from(stats.inactive_count);
        macos_available_bytes(free, spec, inactive, page, sysctl_u64(c"hw.memsize"))
    }

    pub(super) fn own_rss_bytes() -> Option<usize> {
        // SAFETY: an all-zero proc_taskinfo is a valid value.
        let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_taskinfo>() as c_int;
        // SAFETY: `info` is `size` bytes; proc_pidinfo writes at most that
        // and returns the count written.
        let n = unsafe {
            libc::proc_pidinfo(
                libc::getpid(),
                libc::PROC_PIDTASKINFO,
                0,
                (&raw mut info).cast::<c_void>(),
                size,
            )
        };
        (n == size).then(|| usize::try_from(info.pti_resident_size).unwrap_or(usize::MAX))
    }

    pub(super) fn own_peak_rss_bytes() -> Option<usize> {
        let ru = rusage_self()?;
        // `c_long` is i64 on every macOS target.
        ru_maxrss_to_bytes(ru.ru_maxrss, native_ru_maxrss_unit())
    }

    pub(super) fn own_cpu_seconds() -> Option<f64> {
        let ru = rusage_self()?;
        let tv = |t: libc::timeval| t.tv_sec as f64 + f64::from(t.tv_usec) * 1e-6;
        Some(tv(ru.ru_utime) + tv(ru.ru_stime))
    }
}

#[cfg(windows)]
mod imp {
    use super::filetime_ticks_to_seconds;
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    fn memory_status() -> Option<MEMORYSTATUSEX> {
        // SAFETY: an all-zero MEMORYSTATUSEX is valid; dwLength must be set.
        let mut st: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        st.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        // SAFETY: `st` is a valid MEMORYSTATUSEX with dwLength set.
        let ok = unsafe { GlobalMemoryStatusEx(&mut st) };
        (ok != 0).then_some(st)
    }

    fn process_counters() -> Option<PROCESS_MEMORY_COUNTERS> {
        // SAFETY: an all-zero PROCESS_MEMORY_COUNTERS is valid.
        let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        let cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        c.cb = cb;
        // SAFETY: GetCurrentProcess returns a pseudo-handle that needs no
        // closing; `c` is `cb` bytes.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, cb) };
        (ok != 0).then_some(c)
    }

    fn to_usize(v: u64) -> usize {
        usize::try_from(v).unwrap_or(usize::MAX)
    }

    pub(super) fn available_ram_bytes() -> Option<usize> {
        memory_status().map(|s| to_usize(s.ullAvailPhys))
    }

    pub(super) fn total_ram_bytes() -> Option<usize> {
        memory_status().map(|s| to_usize(s.ullTotalPhys))
    }

    pub(super) fn own_rss_bytes() -> Option<usize> {
        process_counters().map(|c| c.WorkingSetSize)
    }

    pub(super) fn own_peak_rss_bytes() -> Option<usize> {
        process_counters().map(|c| c.PeakWorkingSetSize)
    }

    pub(super) fn own_cpu_seconds() -> Option<f64> {
        let zero = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let (mut creation, mut exit, mut kernel, mut user) = (zero, zero, zero, zero);
        // SAFETY: four valid, writable FILETIMEs and the current-process
        // pseudo-handle.
        let ok = unsafe {
            GetProcessTimes(
                GetCurrentProcess(),
                &mut creation,
                &mut exit,
                &mut kernel,
                &mut user,
            )
        };
        if ok == 0 {
            return None;
        }
        let secs = |t: FILETIME| filetime_ticks_to_seconds(t.dwHighDateTime, t.dwLowDateTime);
        Some(secs(kernel) + secs(user))
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
mod imp {
    pub(super) fn available_ram_bytes() -> Option<usize> {
        None
    }
    pub(super) fn total_ram_bytes() -> Option<usize> {
        None
    }
    pub(super) fn own_rss_bytes() -> Option<usize> {
        None
    }
    pub(super) fn own_peak_rss_bytes() -> Option<usize> {
        None
    }
    pub(super) fn own_cpu_seconds() -> Option<f64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ru_maxrss_units_differ_by_1024() {
        assert_eq!(
            ru_maxrss_to_bytes(2048, RuMaxrssUnit::Kibibytes),
            Some(2048 * 1024)
        );
        assert_eq!(ru_maxrss_to_bytes(2048, RuMaxrssUnit::Bytes), Some(2048));
        assert_eq!(ru_maxrss_to_bytes(0, RuMaxrssUnit::Bytes), None);
        assert_eq!(ru_maxrss_to_bytes(-5, RuMaxrssUnit::Kibibytes), None);
        assert_eq!(
            ru_maxrss_to_bytes(i64::MAX, RuMaxrssUnit::Kibibytes),
            Some(clamp_u64(u64::MAX))
        );
    }

    #[test]
    fn native_unit_is_bytes_only_on_apple() {
        let expect = if cfg!(target_os = "macos") {
            RuMaxrssUnit::Bytes
        } else {
            RuMaxrssUnit::Kibibytes
        };
        assert_eq!(native_ru_maxrss_unit(), expect);
    }

    #[test]
    fn macos_available_subtracts_speculative_and_caps_at_total() {
        // 16 KiB pages (Apple silicon): (1000 - 200 + 3000) pages.
        assert_eq!(
            macos_available_bytes(1000, 200, 3000, 16384, None),
            Some(3800 * 16384)
        );
        assert_eq!(
            macos_available_bytes(1000, 200, 3000, 16384, Some(1 << 20)),
            Some(1 << 20)
        );
        // speculative > free must not wrap.
        assert_eq!(macos_available_bytes(10, 20, 5, 4096, None), Some(5 * 4096));
        assert_eq!(macos_available_bytes(10, 0, 5, 0, None), None);
    }

    #[test]
    fn filetime_is_100ns_ticks() {
        assert!((filetime_ticks_to_seconds(0, 10_000_000) - 1.0).abs() < 1e-12);
        // 2^32 ticks = 429.4967296 s.
        assert!((filetime_ticks_to_seconds(1, 0) - 429.496_729_6).abs() < 1e-9);
    }

    #[test]
    fn status_kb_field_requires_the_exact_field_name() {
        let s = "VmPeak:  10 kB\nVmHWM:   20 kB\nVmRSS:   15 kB\n";
        assert_eq!(parse_status_kb_field(s, "VmHWM"), Some(20 * 1024));
        assert_eq!(parse_status_kb_field(s, "VmRSS"), Some(15 * 1024));
        assert_eq!(parse_status_kb_field(s, "VmH"), None);
        assert_eq!(parse_status_kb_field("VmHWM: x kB\n", "VmHWM"), None);
    }

    #[test]
    fn proc_stat_cpu_handles_parenthesised_comm() {
        // comm "(a) b)" contains a space and a ')'.
        let s = "123 (a) b) S 1 2 3 4 5 6 7 8 9 10 250 50 0 0";
        assert_eq!(parse_proc_stat_cpu_seconds(s), Some(3.0));
        assert_eq!(parse_proc_stat_cpu_seconds("garbage"), None);
    }

    /// The live probes must answer on the platforms ferric ships for, and the
    /// numbers must be mutually consistent (a unit slip shows up as rss >
    /// total or peak < rss by a factor of ~1024).
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", windows))]
    fn live_probes_answer_and_agree() {
        let total = total_ram_bytes().expect("total RAM");
        let avail = available_ram_bytes().expect("available RAM");
        let rss = own_rss_bytes().expect("rss");
        let peak = own_peak_rss_bytes().expect("peak rss");
        assert!(avail <= total, "available {avail} > total {total}");
        assert!(rss > 0 && rss < total, "rss {rss} vs total {total}");
        // Peak is a high-water mark; sampled after rss it can only be >= it,
        // up to page-accounting slack.
        assert!(peak + (1 << 20) >= rss, "peak {peak} < rss {rss}");
        assert!(peak < total, "peak {peak} vs total {total}");
        assert!(own_cpu_seconds().expect("cpu seconds") >= 0.0);
    }
}
