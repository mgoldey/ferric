//! Which device paths `auto` takes: a pure function of the mode, the path, the
//! basis size and the functional class. `on` always takes a path the device
//! can run, `off` never does; `auto` takes it only where the device was
//! measured to win.
//!
//! Scope: the SCF's RI-J and DF-K builds. The `einsum!` GEMM offload has its
//! own threshold ([`FERRIC_GPU_MIN_FLOPS_DEFAULT`](super::config::FERRIC_GPU_MIN_FLOPS_DEFAULT));
//! the XC grid and the RI-MP2 energy are not `auto` paths.
//!
//! # Measured crossover
//!
//! Per-build time of the SCF's RI-J (`jk:ri_j`) and DF-K (`jk:df_k`) rows of
//! `FERRIC_TIMING`, CPU default (`off`: f32 RI-J tensor, packed DF-K) against
//! `on`, RHF, n-alkanes, aux `def2-universal-jkfit`, GTX 1080 against 6 CPU
//! cores. `nbf` is the basis size, `naux` the aux size, `nocc` the occupied
//! count; the DF-K FLOP count per build is `3 * naux * nocc * nbf^2` (the
//! `Y = C_occ^T B` GEMM plus the SYRK). Ratios are CPU time over device
//! time, so above 1 the device is faster.
//!
//! ```text
//!  system        nbf  naux nocc  DF-K FLOP   RI-J ms/build      DF-K ms/build      whole run s
//!                                              off   on  off/on   off    on  off/on   off    on
//!  C10/def2-TZVP  442  1146   36   2.5e10      44   43  1.03     296   320  0.92     8.2   8.4
//!  C12/def2-TZVP  528  1368   49   5.6e10      78   71  1.10     586   518  1.13    15.1  13.9
//!  C14/def2-TZVP  614  1590   57   1.0e11     122  368  0.33    1102   775  1.42    26.0  25.8
//!  C12/def2-SVP   298  1368   49   1.3e10      24   26  0.94     166   179  0.93     4.9   5.2
//!  C16/def2-SVP   394  1812   65   5.5e10      57   55  1.04     435   680  0.64    11.9  14.8
//!  C20/def2-SVP   490  2232   81   1.3e11     105  101  1.05    1053  1160  0.91    25.6  26.6
//! ```
//!
//! Mean of 2 runs per cell (off and on interleaved, in-window foreign CPU
//! load 0.2-0.8 cores of 6, rows from `FERRIC_TIMING` divided by the 13-15
//! build count; the two runs of a cell differ by under 3% on the DF-K rows).
//! The C14/TZVP RI-J row under `on` is the CPU f64 tensor: the dressed DF-K
//! tensor (4.8 GB) takes the pool and J yields to it. Earlier T019 rows
//! (C8/SVP RI-J 0.15 s device against 0.11 s CPU, C12/TZVP 1.15) agree.
//!
//! # Why the measure is `nbf`
//!
//! A FLOP-count gate does not describe the data: n-hexadecane/def2-SVP and
//! n-dodecane/def2-TZVP need the same DF-K work per build (5.5e10 FLOP) and
//! the device loses on one (0.64) and wins on the other (1.13), and
//! n-eicosane/def2-SVP at 2.4x that work loses (0.91). The derived rule that
//! [`derive_min_flops`](super::config::derive_min_flops) uses (the smallest
//! size at which the device wins and keeps winning at every larger measured
//! size) has no solution in FLOPs and has one in `nbf` for DF-K: it loses at
//! every measured `nbf` up to 490 and wins at 528 and 614, so the threshold
//! is 512. RI-J per build is at parity with the CPU default (f32 tensor) at
//! every measured size (0.94-1.10, 0.33 at `nbf` 614 where J yields the
//! card to K and runs on the f64 CPU tensor), so its per-build rows do not
//! set a threshold. It takes the DF-K threshold because the two share one raw
//! tensor: with K on the device and J on the CPU the host builds the f32 J
//! tensor in a second integral pass (C12/def2-TZVP RHF, one run each, before
//! and after this rule: `setup:df_fitters` 4.5 s with J and K on the device,
//! 7.3 s with K alone; whole run 13.8 s against 17.6 s). Below the threshold
//! neither path is taken.
//! The device needs the
//! dressed tensor resident (`naux * nbf^2 * 8` bytes against a pool of about
//! 6.7 GB on the 8 GB card), so the sizes above the largest measured one
//! decline on the pool and are not a separate regime.
//!
//! The 16-carbon def2-SVP DF-K build is 1.5-1.6x slower on the device while its
//! neighbours are at parity (an unexplained, unprofiled point; reproduced in
//! both runs here). It sits below the threshold (nbf 394 against 512), so the
//! rule excludes it without being fitted to it: the threshold is the same with
//! that point removed, because the next-largest losing size is 490.
//!
//! # Pure functionals
//!
//! For a pure functional (or a range-separated one) the SCF has no K in its
//! main Fock build and runs the RI-J build and the XC digest concurrently on
//! the CPU (the J || XC overlap). The device RI-J path declines on a rayon
//! worker, so taking it switches the overlap off and runs J and XC in
//! sequence. Under `auto` a pure functional therefore keeps J on the CPU
//! (T019: `auto` was 6-17% slower than `off` for PBE at every size). The
//! rule is explicit in [`routes_to_device`] and independent of the RI-J
//! size gate, so it holds if that gate is later lowered.

use super::config::GpuMode;

/// A device path whose use `auto` gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoPath {
    /// The RI-J build (`df_j_gpu`).
    RiJ,
    /// The DF-K occupied-orbital build (`df_k_gpu`).
    DfK,
}

/// What `auto` needs to know about the problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceSite {
    /// Number of basis functions.
    pub nbf: usize,
    /// The SCF builds J with no K beside it and has an XC functional: a
    /// pure or range-separated functional, whose J build overlaps the XC
    /// digest on the CPU.
    pub jxc_overlap: bool,
}

impl DeviceSite {
    /// A site with no J || XC overlap (a hybrid or Hartree-Fock SCF, or a
    /// path that does not build J).
    pub fn plain(nbf: usize) -> Self {
        Self {
            nbf,
            jxc_overlap: false,
        }
    }
}

/// Smallest `nbf` at which `auto` takes the device RI-J build.
pub const AUTO_MIN_NBF_RI_J: usize = 512;

/// Smallest `nbf` at which `auto` takes the device DF-K build.
pub const AUTO_MIN_NBF_DF_K: usize = 512;

/// Whether `path` goes to the device for `site` under `mode`. Whether the
/// device can actually run it (status, pool, layout) is decided by the path
/// itself and can still send the build to the CPU.
pub fn routes_to_device(mode: GpuMode, path: AutoPath, site: DeviceSite) -> bool {
    match mode {
        GpuMode::Off => false,
        GpuMode::On => true,
        GpuMode::Auto => match path {
            AutoPath::RiJ => auto_ri_j(site, AUTO_MIN_NBF_RI_J),
            AutoPath::DfK => site.nbf >= AUTO_MIN_NBF_DF_K,
        },
    }
}

/// The `auto` RI-J rule with the size threshold given: never for a pure or
/// range-separated functional (the J || XC overlap stays on the CPU), else
/// from `min_nbf`.
fn auto_ri_j(site: DeviceSite, min_nbf: usize) -> bool {
    !site.jxc_overlap && site.nbf >= min_nbf
}

#[cfg(test)]
mod tests {
    use super::*;

    const BIG: usize = 4096;
    const SMALL: usize = 64;

    fn site(nbf: usize, jxc_overlap: bool) -> DeviceSite {
        DeviceSite { nbf, jxc_overlap }
    }

    #[test]
    fn off_never_and_on_always() {
        for path in [AutoPath::RiJ, AutoPath::DfK] {
            for s in [site(SMALL, false), site(BIG, false), site(BIG, true)] {
                assert!(!routes_to_device(GpuMode::Off, path, s));
                assert!(routes_to_device(GpuMode::On, path, s));
            }
        }
    }

    #[test]
    fn auto_dfk_boundary_is_the_threshold() {
        let at = |n| routes_to_device(GpuMode::Auto, AutoPath::DfK, site(n, false));
        assert!(!at(SMALL));
        assert!(!at(AUTO_MIN_NBF_DF_K - 1));
        assert!(at(AUTO_MIN_NBF_DF_K));
        assert!(at(BIG));
    }

    #[test]
    fn dfk_threshold_separates_the_measured_sizes() {
        // Sweep rows in the module doc: the device loses at nbf <= 490 and
        // wins at 528 and 614.
        let at = |n| routes_to_device(GpuMode::Auto, AutoPath::DfK, site(n, false));
        for n in [298, 394, 442, 490] {
            assert!(!at(n), "nbf {n}");
        }
        for n in [528, 614] {
            assert!(at(n), "nbf {n}");
        }
    }

    #[test]
    fn auto_ri_j_boundary_is_the_threshold_for_a_hybrid_and_never_for_a_pure_functional() {
        let at = |n, pure| routes_to_device(GpuMode::Auto, AutoPath::RiJ, site(n, pure));
        assert!(!at(AUTO_MIN_NBF_RI_J - 1, false));
        assert!(at(AUTO_MIN_NBF_RI_J, false));
        for n in [298, 394, 442, 490, 528, 614, BIG] {
            assert!(!at(n, true), "pure functional, nbf {n}");
        }
    }

    #[test]
    fn ri_j_rule_keeps_a_pure_functional_on_the_cpu_and_gates_the_rest_by_size() {
        assert!(!auto_ri_j(site(BIG, true), 512));
        assert!(auto_ri_j(site(BIG, false), 512));
        assert!(!auto_ri_j(site(511, false), 512));
        assert!(auto_ri_j(site(512, false), 512));
    }

    #[test]
    fn the_pure_functional_rule_does_not_gate_k() {
        assert!(routes_to_device(
            GpuMode::Auto,
            AutoPath::DfK,
            site(BIG, true)
        ));
    }
}
