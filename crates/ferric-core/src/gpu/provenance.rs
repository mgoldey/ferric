//! Precision provenance: which mixed-precision kernels a result's numbers went
//! through, recorded on the result itself (the first field of the Pillar 1
//! certificate; plan `plan_mixed_precision.md` §1 "Provenance", rank 1).
//!
//! A [`PrecisionProbe`] snapshots the per-kernel counters of
//! [`crate::gpu::stats`](mod@crate::gpu::stats) when a method starts; [`PrecisionProbe::finish`]
//! turns the difference into a [`PrecisionRecord`] when it returns. A run in
//! which no kernel ran mixed reports `precision = f64` explicitly.
//!
//! Scope and limits, stated rather than hidden:
//!
//! * The counters are process-wide. Two calculations running at the same time
//!   in one process (Python threads with the GIL released) see each other's
//!   counts. A sequential run is exact.
//! * Nothing here is consulted by any calculation: the record observes, it
//!   never changes a number (the f64 path is byte-identical with or without
//!   it, pinned by the `precision_provenance` tests).
//! * No a-posteriori error bound is computed by any kernel yet. Each mixed
//!   kernel's record says so ([`ErrorBound::NoneComputed`] with the reason);
//!   a made-up number is never written. The schema has a slot for a computed
//!   bound ([`ErrorBound::APosteriori`]) for the kernels the plan says can
//!   carry one cheaply (the RI-MP2 Cauchy-Schwarz bound, §5.3; the RPA
//!   dielectric bound, §7.3, whose kernel does not exist yet).
use std::fmt;

use super::precision::{MixedKernel, Precision};
use super::stats::{kernel_counters, KernelCounters, MixedFallback};
use super::{installed_settings, installed_status, GpuStatus};

/// Kernels that can run inside an SCF (RHF/UHF/ROHF and their KS-DFT forms).
pub const SCF_KERNELS: &[MixedKernel] = &[
    MixedKernel::XcVxc,
    MixedKernel::DfjPack,
    MixedKernel::DfkOcc,
    MixedKernel::CosxKern,
];

/// Kernels that can run inside an RI-MP2 energy (closed shell and
/// unrestricted; Coulomb, erfc and terfc).
pub const RIMP2_KERNELS: &[MixedKernel] = &[MixedKernel::RiMp2Energy];

/// Where a kernel's mixed calls ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelDevice {
    /// The host CPU (a CPU kernel, or a device kernel with no usable device).
    Cpu,
    /// A CUDA device, as probed at install.
    Gpu {
        ordinal: usize,
        name: String,
        cc_major: i32,
        cc_minor: i32,
    },
}

impl fmt::Display for KernelDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelDevice::Cpu => f.write_str("cpu"),
            KernelDevice::Gpu {
                ordinal,
                name,
                cc_major,
                cc_minor,
            } => write!(f, "gpu {ordinal} {name} (cc {cc_major}.{cc_minor})"),
        }
    }
}

/// An error bound for a kernel's contribution to the result.
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorBound {
    /// No bound was computed for this run; `why` says what exists instead.
    NoneComputed { why: &'static str },
    /// A bound computed at runtime from this run's own quantities, in
    /// Hartree, and the name of the estimate that produced it.
    APosteriori { eh: f64, method: &'static str },
}

/// The precision ladder's counts for one kernel (only kernels behind a ladder
/// carry one: `xc-vxc`, `dfj-pack`, `dfk-occ`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LadderRecord {
    /// Builds that ran mixed.
    pub builds_mixed: u64,
    /// Builds the ladder was consulted for while mixed was requested.
    pub builds_total: u64,
    /// Converged exits whose last mixed contribution was rebuilt in f64.
    pub exit_rebuilt_f64: u64,
}

/// One kernel's provenance for one result.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRecord {
    pub kernel: MixedKernel,
    /// What the installed settings allowed: `Mixed` when `precision = mixed`
    /// and the kernel is in `mixed_kernels`, else `F64` (the kernel is
    /// listed only because a test seam ran it).
    pub requested: Precision,
    /// What ran: `Mixed` when at least one call ran mixed, else `F64`.
    pub mode: Precision,
    /// Where the mixed calls ran: `None` when none did (`calls_mixed == 0`),
    /// so the record never names a device that computed nothing in mixed.
    pub device: Option<KernelDevice>,
    /// Calls that ran mixed.
    pub calls_mixed: u64,
    /// Calls (or parts of calls) that fell back to f64, per
    /// [`MixedFallback`] (indexed by `MixedFallback as usize`).
    pub calls_f64_fallback: [u64; MixedFallback::ALL.len()],
    /// The precision ladder, for a kernel behind one.
    pub ladder: Option<LadderRecord>,
    pub error_bound: ErrorBound,
}

/// The precision provenance of one result.
#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionRecord {
    /// `F64` when no kernel ran mixed (every number is IEEE f64 arithmetic),
    /// `Mixed` when at least one did.
    pub precision: Precision,
    /// One record per kernel that was allowed to run mixed for this result
    /// or that ran mixed; empty for an f64 run with no mixed request.
    pub kernels: Vec<KernelRecord>,
}

impl Default for PrecisionRecord {
    fn default() -> Self {
        Self::f64()
    }
}

impl PrecisionRecord {
    /// The record of a run in which nothing was allowed to run mixed.
    pub fn f64() -> Self {
        PrecisionRecord {
            precision: Precision::F64,
            kernels: Vec::new(),
        }
    }

    /// One short line for printouts: `f64`, or `mixed (kernel: calls, ...)`.
    pub fn summary(&self) -> String {
        if self.kernels.is_empty() {
            return self.precision.to_string();
        }
        let parts: Vec<String> = self
            .kernels
            .iter()
            .map(|k| match &k.device {
                Some(d) => format!(
                    "{} {} on {d}: {} mixed call(s)",
                    k.kernel.name(),
                    k.mode,
                    k.calls_mixed
                ),
                None => format!("{} {}: 0 mixed calls", k.kernel.name(), k.mode),
            })
            .collect();
        format!("{} ({})", self.precision, parts.join("; "))
    }

    /// The record as JSON (CLI run log, Python): see the input reference for
    /// the schema. `counter_scope` is always `"process"`: the counts are
    /// process-wide, so a calculation running CONCURRENTLY with this one in
    /// the same process would be counted here too (sequential runs are exact).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "precision": self.precision.to_string(),
            "counter_scope": COUNTER_SCOPE,
            "kernels": self.kernels.iter().map(KernelRecord::to_json).collect::<Vec<_>>(),
        })
    }
}

impl KernelRecord {
    pub fn to_json(&self) -> serde_json::Value {
        let device = match &self.device {
            None => serde_json::Value::Null,
            Some(KernelDevice::Cpu) => serde_json::json!("cpu"),
            Some(KernelDevice::Gpu {
                ordinal,
                name,
                cc_major,
                cc_minor,
            }) => serde_json::json!({
                "ordinal": ordinal,
                "name": name,
                "cc": format!("{cc_major}.{cc_minor}"),
            }),
        };
        let fallback: serde_json::Map<String, serde_json::Value> = MixedFallback::ALL
            .iter()
            .map(|&r| {
                (
                    r.name().to_string(),
                    self.calls_f64_fallback[r as usize].into(),
                )
            })
            .collect();
        let ladder = self.ladder.map(|l| {
            serde_json::json!({
                "builds_mixed": l.builds_mixed,
                "builds_total": l.builds_total,
                "exit_rebuilt_f64": l.exit_rebuilt_f64,
            })
        });
        let bound = match &self.error_bound {
            ErrorBound::NoneComputed { why } => serde_json::json!({
                "kind": "none_computed",
                "why": why,
            }),
            ErrorBound::APosteriori { eh, method } => serde_json::json!({
                "kind": "a_posteriori",
                "eh": if eh.is_finite() { serde_json::json!(eh) } else { serde_json::json!(eh.to_string()) },
                "method": method,
            }),
        };
        serde_json::json!({
            "kernel": self.kernel.name(),
            "requested": self.requested.to_string(),
            "mode": self.mode.to_string(),
            "device": device,
            "calls_mixed": self.calls_mixed,
            "calls_f64_fallback": fallback,
            "ladder": ladder,
            "error_bound": bound,
        })
    }
}

/// Whether `k` sits behind an SCF precision ladder (mixed only while the
/// density moves, f64 exit rebuild).
fn has_ladder(k: MixedKernel) -> bool {
    matches!(
        k,
        MixedKernel::XcVxc | MixedKernel::DfjPack | MixedKernel::DfkOcc
    )
}

/// Why no bound is attached to `k`'s record.
fn no_bound_reason(k: MixedKernel) -> &'static str {
    match k {
        MixedKernel::RiMp2Energy => {
            "no per-run bound is computed (an a-posteriori Cauchy-Schwarz energy bound \
             is designed but not implemented); the kernel's measured error map is on \
             the validation page"
        }
        MixedKernel::XcVxc => {
            "no per-run bound is computed; ladder.exit_rebuilt_f64 counts the SCF exits \
             whose last mixed V_xc was rebuilt in f64 (zero when the run did not \
             converge). Measured error map on the validation page"
        }
        _ => "no per-run bound is computed; this kernel is not shipped and has no runtime bound",
    }
}

/// The `counter_scope` of every record: the counters are process-wide.
pub const COUNTER_SCOPE: &str = "process";

/// A snapshot of the per-kernel counters at the start of a method.
#[derive(Debug, Clone)]
pub struct PrecisionProbe {
    before: [KernelCounters; MixedKernel::ALL.len()],
}

impl PrecisionProbe {
    pub fn start() -> Self {
        PrecisionProbe {
            before: kernel_counters(),
        }
    }

    /// The record of everything `stage` ran since [`Self::start`]. Reads the
    /// installed settings without initialising them
    /// ([`installed_settings`]): with nothing installed no mixed kernel can
    /// have run.
    pub fn finish(&self, stage: &[MixedKernel]) -> PrecisionRecord {
        let now = kernel_counters();
        let settings = installed_settings();
        let gpu = match installed_status() {
            Some(GpuStatus::Ready(info)) => Some(KernelDevice::Gpu {
                ordinal: info.ordinal,
                name: info.name.clone(),
                cc_major: info.cc_major,
                cc_minor: info.cc_minor,
            }),
            _ => None,
        };
        let mut kernels = Vec::new();
        for &k in stage {
            let d = now[k.index()].since(&self.before[k.index()]);
            let allowed = settings.is_some_and(|s| s.mixed_allows(k));
            if !allowed && !d.any() {
                continue;
            }
            let device = (d.mixed > 0).then(|| match (&gpu, k.runs_on_cpu()) {
                (Some(g), false) => g.clone(),
                _ => KernelDevice::Cpu,
            });
            kernels.push(KernelRecord {
                kernel: k,
                requested: if allowed {
                    Precision::Mixed
                } else {
                    Precision::F64
                },
                mode: if d.mixed > 0 {
                    Precision::Mixed
                } else {
                    Precision::F64
                },
                device,
                calls_mixed: d.mixed,
                calls_f64_fallback: d.fallback,
                ladder: has_ladder(k).then_some(LadderRecord {
                    builds_mixed: d.mixed,
                    builds_total: d.ladder_builds,
                    exit_rebuilt_f64: d.exit_rebuilds,
                }),
                error_bound: ErrorBound::NoneComputed {
                    why: no_bound_reason(k),
                },
            });
        }
        let precision = if kernels.iter().any(|k| k.mode == Precision::Mixed) {
            Precision::Mixed
        } else {
            Precision::F64
        };
        PrecisionRecord { precision, kernels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::stats::{
        note_exit_rebuild, note_kernel_mixed, note_ladder_build, note_mixed_fallback,
    };

    #[test]
    fn f64_record_is_explicit_f64_with_no_kernels() {
        let r = PrecisionRecord::f64();
        assert_eq!(r.precision, Precision::F64);
        assert!(r.kernels.is_empty());
        assert_eq!(
            r.to_json(),
            serde_json::json!({"precision": "f64", "counter_scope": "process", "kernels": []})
        );
        assert_eq!(r.summary(), "f64");
    }

    /// The kernels behind an SCF ladder carry a `ladder` entry, the others do
    /// not. This test is the only writer of these kernels' counters here.
    #[test]
    fn the_jk_kernels_report_their_ladder() {
        for k in [MixedKernel::DfjPack, MixedKernel::DfkOcc] {
            let probe = PrecisionProbe::start();
            note_ladder_build(k);
            note_ladder_build(k);
            note_kernel_mixed(k);
            note_exit_rebuild(k);
            let r = probe.finish(&[k]);
            assert_eq!(
                r.kernels[0].ladder,
                Some(LadderRecord {
                    builds_mixed: 1,
                    builds_total: 2,
                    exit_rebuilt_f64: 1,
                }),
                "{k:?}"
            );
        }
    }

    /// The probe reports exactly the counts noted between start and finish,
    /// for a kernel no installed setting allows (so it is listed only because
    /// it ran), on the CPU. A process-wide counter: this test is the only
    /// writer of the cosx-kern counters in this binary.
    #[test]
    fn probe_reports_the_delta_of_a_kernel_that_ran() {
        let k = MixedKernel::CosxKern;
        let probe = PrecisionProbe::start();
        assert!(probe.finish(&[k]).kernels.is_empty());
        note_kernel_mixed(k);
        note_kernel_mixed(k);
        note_mixed_fallback(k, MixedFallback::F32Range);
        note_ladder_build(k);
        note_exit_rebuild(k);
        let r = probe.finish(&[k]);
        assert_eq!(r.precision, Precision::Mixed);
        assert_eq!(r.kernels.len(), 1);
        let kr = &r.kernels[0];
        assert_eq!(kr.kernel, k);
        assert_eq!(kr.mode, Precision::Mixed);
        assert_eq!(kr.device, Some(KernelDevice::Cpu));
        assert_eq!(kr.calls_mixed, 2);
        assert_eq!(kr.calls_f64_fallback, [0, 1, 0]);
        assert_eq!(kr.ladder, None, "cosx-kern has no ladder");
        assert!(matches!(kr.error_bound, ErrorBound::NoneComputed { .. }));
        let j = r.to_json();
        assert_eq!(j["precision"], "mixed");
        assert_eq!(j["kernels"][0]["kernel"], "cosx-kern");
        assert_eq!(j["kernels"][0]["device"], "cpu");
        assert_eq!(j["kernels"][0]["calls_mixed"], 2);
        assert_eq!(j["kernels"][0]["calls_f64_fallback"]["f32_range"], 1);
        assert_eq!(j["kernels"][0]["error_bound"]["kind"], "none_computed");
        // A second probe started now sees none of it.
        assert!(PrecisionProbe::start().finish(&[k]).kernels.is_empty());
    }

    /// A kernel listed with no mixed call names no device: the record never
    /// claims a device computed something in mixed when nothing did.
    #[test]
    fn a_kernel_with_no_mixed_call_names_no_device() {
        let k = MixedKernel::CcsdAmplitudes;
        let probe = PrecisionProbe::start();
        note_mixed_fallback(k, MixedFallback::CpuF64);
        let r = probe.finish(&[k]);
        assert_eq!(r.precision, Precision::F64);
        assert_eq!(r.kernels[0].device, None);
        assert_eq!(r.kernels[0].calls_f64_fallback, [0, 0, 1]);
        let j = r.to_json();
        assert!(j["kernels"][0]["device"].is_null());
        assert_eq!(j["counter_scope"], "process");
    }
}
