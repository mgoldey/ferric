//! Device-memory ledger, the same contract as `crate::memory::pool::MemoryPool`
//! (debited on reserve, credited on drop, refuses past the ceiling, reports
//! occupancy). Kept separate from the host pool: device bytes are a different
//! resource and must not reduce the host budget.
//!
//! The ledger only knows what ferric reserved. Another process can take device
//! memory after the probe, so a successful `reserve` does not guarantee the
//! allocation succeeds; callers must treat an allocation error as a per-call
//! CPU fallback, never an abort.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::device::GpuError;

#[derive(Debug)]
struct Inner {
    capacity: usize,
    outstanding: AtomicUsize,
    peak: AtomicUsize,
    by_label: Mutex<BTreeMap<String, usize>>,
}

#[derive(Clone, Debug)]
pub struct DevicePool {
    inner: Arc<Inner>,
}

/// Holds `bytes` of the pool until dropped.
#[derive(Debug)]
pub struct DeviceReservation {
    pool: DevicePool,
    label: String,
    bytes: usize,
}

impl Drop for DeviceReservation {
    fn drop(&mut self) {
        self.pool
            .inner
            .outstanding
            .fetch_sub(self.bytes, Ordering::AcqRel);
        if let Ok(mut m) = self.pool.inner.by_label.lock() {
            if let Some(v) = m.get_mut(&self.label) {
                *v = v.saturating_sub(self.bytes);
            }
        }
    }
}

impl DevicePool {
    pub fn with_capacity_bytes(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                capacity,
                outstanding: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                by_label: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    pub fn capacity_bytes(&self) -> usize {
        self.inner.capacity
    }

    pub fn available_bytes(&self) -> usize {
        self.inner
            .capacity
            .saturating_sub(self.inner.outstanding.load(Ordering::Acquire))
    }

    pub fn peak_bytes(&self) -> usize {
        self.inner.peak.load(Ordering::Acquire)
    }

    /// Debit `bytes` or refuse, naming `label` and the shortfall. A refusal
    /// leaves the ledger untouched.
    pub fn reserve(&self, label: &str, bytes: usize) -> Result<DeviceReservation, GpuError> {
        let cap = self.inner.capacity;
        let outcome =
            self.inner
                .outstanding
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                    let next = cur.saturating_add(bytes);
                    (next <= cap).then_some(next)
                });
        match outcome {
            Ok(prev) => {
                self.inner
                    .peak
                    .fetch_max(prev.saturating_add(bytes), Ordering::AcqRel);
                if let Ok(mut m) = self.inner.by_label.lock() {
                    *m.entry(label.to_string()).or_insert(0) += bytes;
                }
                Ok(DeviceReservation {
                    pool: self.clone(),
                    label: label.to_string(),
                    bytes,
                })
            }
            Err(cur) => Err(GpuError::PoolFull {
                label: label.to_string(),
                detail: format!(
                    "needs {:.3} GB but only {:.3} GB of the {:.3} GB device pool is free (short by {:.3} GB) — \
                     raise [gpu] memory_gb / FERRIC_GPU_MEM_GB or let the GEMM run on the CPU\n{}",
                    bytes as f64 / 1e9,
                    cap.saturating_sub(cur) as f64 / 1e9,
                    cap as f64 / 1e9,
                    bytes.saturating_sub(cap.saturating_sub(cur)) as f64 / 1e9,
                    self.occupancy_report()
                ),
            }),
        }
    }

    pub fn try_reserve(&self, label: &str, bytes: usize) -> Option<DeviceReservation> {
        self.reserve(label, bytes).ok()
    }

    pub fn occupancy_report(&self) -> String {
        let mut s = format!(
            "  device pool: {:.3} GB outstanding of {:.3} GB (peak {:.3} GB)\n",
            self.inner.outstanding.load(Ordering::Acquire) as f64 / 1e9,
            self.inner.capacity as f64 / 1e9,
            self.peak_bytes() as f64 / 1e9
        );
        if let Ok(m) = self.inner.by_label.lock() {
            let mut rows: Vec<_> = m.iter().filter(|(_, b)| **b > 0).collect();
            rows.sort_by(|a, b| b.1.cmp(a.1));
            for (l, b) in rows {
                s.push_str(&format!("    {:.3} GB  {l}\n", *b as f64 / 1e9));
            }
        }
        s
    }
}
