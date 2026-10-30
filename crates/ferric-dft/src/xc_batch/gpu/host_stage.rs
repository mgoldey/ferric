//! Host staging of the device XC waves, allocated once per grid and reused by
//! every wave of every Fock build.
//!
//! A fresh `vec![0.0; n]` per wave cost a page fault per 4 KiB page on first
//! touch (≈59 000 faults and ≈120 ms of system time per benzene PBE/cc-pVDZ
//! build on the GTX 1080 this was measured on), and the device-to-host copy
//! into it ran at the pageable rate. The buffers are page-locked while the
//! grid's total stays within [`PINNED_CAP_BYTES`] and the driver grants it;
//! otherwise ordinary heap memory, still allocated once. Each refused
//! page-locked allocation is counted (`stats::xc_pinned_fallback`). They are
//! held for the life of the evaluator (`KsXc` / `KsXcUks`) that owns the grid
//! and released with the grid's device state (decline or drop). The page-locked
//! bytes are NOT charged to `[memory] budget_gb`, cannot be swapped, and each
//! concurrent calculation pins its own copy (see the parent module's
//! "Page-locked host memory").
//!
//! Only memory changes: the same values go up and come back, so results are
//! bit-identical to per-wave heap buffers.
use ferric_core::gpu::device::{Device, GpuError};
use ferric_core::gpu::host::{HostBuf, HostElem};
use ferric_core::gpu::stats;

/// Test seam (`test-seams` feature): fill a wave's operand staging with NaN
/// before the wave writes it. The buffers are reused, so an element a wave
/// fails to write (the zero-fill of the padding, say) would otherwise hold
/// whatever an earlier wave left there, or the allocation's zeros on the first
/// one; with the seam set it is NaN, and NaN times the AO copy's zero padding
/// is NaN in the real part of the result.
#[cfg(feature = "test-seams")]
#[doc(hidden)]
pub static POISON_XC_STAGING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Most page-locked bytes one grid's staging may hold (all buffers together).
pub(crate) const PINNED_CAP_BYTES: usize = 256 << 20;

/// The host side of one grid's waves: operands up, results down, and (mixed
/// `xc-vxc` only) the f32-rounded operands.
pub(crate) struct HostStage {
    pub(crate) inp: HostBuf<f64>,
    pub(crate) out: HostBuf<f64>,
    pub(crate) inp32: Option<HostBuf<f32>>,
}

/// Which buffers of a stage of `stage` elements fit the page-locked budget,
/// first come first served: `[operands, results, f32 operands]`.
pub(crate) fn pinned_plan(stage: usize, with_f32: bool) -> [bool; 3] {
    let mut left = PINNED_CAP_BYTES;
    let bytes = [8 * stage, 8 * stage, if with_f32 { 4 * stage } else { 0 }];
    bytes.map(|b| {
        let fits = b <= left;
        if fits {
            left -= b;
        }
        fits
    })
}

fn buf<T: HostElem>(dev: &Device, len: usize, pinned: bool) -> HostBuf<T> {
    if !pinned {
        return HostBuf::heap(len);
    }
    HostBuf::pinned(dev, len).unwrap_or_else(|_| {
        stats::note_xc_pinned_fallback();
        HostBuf::heap(len)
    })
}

impl HostStage {
    /// Buffers of `stage` elements each (see [`pinned_plan`]).
    pub(crate) fn new(dev: &Device, stage: usize, with_f32: bool) -> Self {
        let [pi, po, p32] = pinned_plan(stage, with_f32);
        Self {
            inp: buf(dev, stage, pi),
            out: buf(dev, stage, po),
            inp32: with_f32.then(|| buf(dev, stage, p32)),
        }
    }

    /// The first `n` operand elements, for a new wave to fill (NaN first under
    /// the [`POISON_XC_STAGING`] seam).
    pub(crate) fn inp_for_wave(&mut self, n: usize) -> Result<&mut [f64], GpuError> {
        let s = self.inp.head_mut(n)?;
        #[cfg(feature = "test-seams")]
        if POISON_XC_STAGING.load(std::sync::atomic::Ordering::Relaxed) {
            s.fill(f64::NAN);
        }
        Ok(s)
    }

    /// Page-locked bytes held.
    pub(crate) fn pinned_bytes(&self) -> usize {
        let b = |pinned: bool, bytes: usize| if pinned { bytes } else { 0 };
        b(self.inp.is_pinned(), self.inp.bytes())
            + b(self.out.is_pinned(), self.out.bytes())
            + self
                .inp32
                .as_ref()
                .map_or(0, |x| b(x.is_pinned(), x.bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_plan_stays_within_the_cap() {
        let st = PINNED_CAP_BYTES / 20;
        assert_eq!(pinned_plan(st, true), [true, true, true]);
        assert_eq!(pinned_plan(st, false), [true, true, true]);
        // 8+8 fit, the f32 copy (4 more) does not.
        let st = PINNED_CAP_BYTES / 18;
        assert_eq!(pinned_plan(st, true), [true, true, false]);
        let st = PINNED_CAP_BYTES / 8;
        assert_eq!(pinned_plan(st, true), [true, false, false]);
        assert_eq!(pinned_plan(PINNED_CAP_BYTES, true), [false, false, false]);
    }
}
