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
//! # RI-J against the f64 CPU tensor
//!
//! A geometry or frequency task (and an explicit `ri_j_precision = "f64"`)
//! keeps the f64 tensor on the CPU, which is slower per build than the f32
//! default. Same protocol, `task = "optimize"` with `max_steps = 1` (two SCFs
//! and a gradient; `jk:ri_j` rows summed over all builds), RHF, `off` against
//! `on`, 2 interleaved runs per cell, foreign load 0.25-0.50 cores:
//!
//! ```text
//!  system         nbf   RI-J ms/build      DF-K ms/build      wall s
//!                       off   on  off/on   off    on  off/on   off    on
//!  C12/def2-SVP   298    67   29  2.34     146   169  0.86    12.8  12.8
//!  C16/def2-SVP   394   150   61  2.45     405   648  0.62    30.1  34.4
//!  C10/def2-TZVP  442   124   49  2.52     265   288  0.92    21.5  20.8
//!  C12/def2-TZVP  528   215   80  2.68     527   476  1.11    37.3  33.9
//!  C14/def2-TZVP  614   349  352  0.99    1016   700  1.45    62.6  56.8
//! ```
//!
//! The device wins by 2.3-2.7x at every size from 298 through 528. At 614 J
//! yields the card to K and runs on the CPU in both arms. `auto` therefore
//! takes RI-J at `nbf` >= [`AUTO_MIN_NBF_RI_J_F64`] (298, the smallest size
//! measured) when the CPU tensor is f64, and at [`AUTO_MIN_NBF_RI_J`] when it
//! is f32. DF-K keeps one threshold: its ratios here repeat the table above.
//! The wall column includes the device DF-K that `on` also takes, which loses
//! at the SVP sizes and is gated off under `auto`.
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
//!
//! The class is the functional's, from [`DeviceSite::scf`]: no ω = 0 exact
//! exchange in the main Fock build. A global hybrid builds K beside J
//! whichever builder makes it (DF-K, COSX, LinK, direct), so it is not in
//! the class. A functional with a VV10 term is classed by its exchange like
//! any other, although the SCF does not run the overlap for it: the device
//! RI-J was not measured for those functionals, and the class keeps their
//! J on the CPU. The site is carried into the RI-J build, so the f64
//! tensor's pair screen and the device build ask the gate with it.

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceSite {
    /// Number of basis functions.
    pub nbf: usize,
    /// The SCF builds J with no K beside it and has an XC functional: a
    /// pure or range-separated functional ([`DeviceSite::scf`]), whose J
    /// build overlaps the XC digest on the CPU.
    pub jxc_overlap: bool,
    /// The CPU RI-J tensor is f64 (a geometry or frequency task, or an
    /// explicit `ri_j_precision = "f64"`), not the f32 default.
    pub cpu_f64: bool,
}

impl DeviceSite {
    /// A site with no J || XC overlap (a hybrid or Hartree-Fock SCF, or a
    /// path that does not build J).
    pub fn plain(nbf: usize) -> Self {
        Self {
            nbf,
            jxc_overlap: false,
            cpu_f64: false,
        }
    }

    /// A site with no J || XC overlap whose CPU RI-J tensor is f64.
    pub fn f64_tensor(nbf: usize) -> Self {
        Self {
            nbf,
            jxc_overlap: false,
            cpu_f64: true,
        }
    }

    /// The RI-J site of an SCF on `nbf` basis functions. `xc` is `None` for
    /// Hartree-Fock, else the functional's exact exchange. The site is a
    /// J || XC overlap site exactly when the functional puts no ω = 0
    /// exact exchange into the main Fock build: a pure functional (no exact
    /// exchange) or a range-separated one (its K comes from its own SR/LR
    /// fitters). A global hybrid builds K beside J, whatever builds it
    /// (DF-K, COSX, LinK or the direct path).
    pub fn scf(nbf: usize, xc: Option<ExactExchange>) -> Self {
        Self {
            nbf,
            jxc_overlap: xc.is_some_and(|k| !(k.coefficient > 0.0 && k.omega == 0.0)),
            cpu_f64: false,
        }
    }

    /// This site with an f64 CPU RI-J tensor: what the f64 tier of the RI-J
    /// build asks the gate with, whichever precision was requested.
    pub fn with_f64_tensor(self) -> Self {
        Self {
            cpu_f64: true,
            ..self
        }
    }
}

/// A functional's exact exchange as the RI-J gate reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactExchange {
    /// Exact-exchange coefficient on the short-range K, the global fraction
    /// when `omega == 0`.
    pub coefficient: f64,
    /// Range-separation parameter ω (Bohr⁻¹); `0` for no range separation.
    pub omega: f64,
}

/// Smallest `nbf` at which `auto` takes the device RI-J build.
pub const AUTO_MIN_NBF_RI_J: usize = 512;

/// Smallest `nbf` at which `auto` takes the device RI-J build when the CPU
/// tensor is f64 (the smallest measured size, where the device is 2.3-2.7x
/// faster per build; sizes below it were not measured).
pub const AUTO_MIN_NBF_RI_J_F64: usize = 298;

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
            AutoPath::RiJ => auto_ri_j(
                site,
                if site.cpu_f64 {
                    AUTO_MIN_NBF_RI_J_F64
                } else {
                    AUTO_MIN_NBF_RI_J
                },
            ),
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
        DeviceSite {
            nbf,
            jxc_overlap,
            cpu_f64: false,
        }
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
    fn auto_ri_j_threshold_depends_on_the_cpu_tensor_precision() {
        let at = |n, f64t| {
            routes_to_device(
                GpuMode::Auto,
                AutoPath::RiJ,
                DeviceSite {
                    nbf: n,
                    jxc_overlap: false,
                    cpu_f64: f64t,
                },
            )
        };
        // f64 CPU tensor: the device wins at every measured size from 298.
        assert!(!at(AUTO_MIN_NBF_RI_J_F64 - 1, true));
        for n in [298, 394, 442, 490, 528] {
            assert!(at(n, true), "f64 tensor, nbf {n}");
        }
        // f32 CPU tensor: the same sizes below 512 stay on the CPU.
        for n in [298, 394, 442, 490] {
            assert!(!at(n, false), "f32 tensor, nbf {n}");
        }
        assert!(at(AUTO_MIN_NBF_RI_J, false));
        // The pure-functional rule outranks the precision branch.
        assert!(!routes_to_device(
            GpuMode::Auto,
            AutoPath::RiJ,
            DeviceSite {
                nbf: BIG,
                jxc_overlap: true,
                cpu_f64: true
            }
        ));
    }

    #[test]
    fn ri_j_rule_keeps_a_pure_functional_on_the_cpu_and_gates_the_rest_by_size() {
        assert!(!auto_ri_j(site(BIG, true), 512));
        assert!(auto_ri_j(site(BIG, false), 512));
        assert!(!auto_ri_j(site(511, false), 512));
        assert!(auto_ri_j(site(512, false), 512));
    }

    // Exact exchange of the functional classes the SCF sees: PBE (none),
    // B3LYP (global 0.2, also the COSX case: its K is not a DF-K), wB97X-V
    // (range-separated, SR 0.167 at ω 0.3).
    const PURE: ExactExchange = ExactExchange {
        coefficient: 0.0,
        omega: 0.0,
    };
    const GLOBAL_HYBRID: ExactExchange = ExactExchange {
        coefficient: 0.2,
        omega: 0.0,
    };
    const RSH: ExactExchange = ExactExchange {
        coefficient: 0.167,
        omega: 0.3,
    };

    #[test]
    fn scf_site_marks_pure_and_range_separated_functionals_as_overlap_sites() {
        assert!(!DeviceSite::scf(BIG, None).jxc_overlap, "Hartree-Fock");
        assert!(DeviceSite::scf(BIG, Some(PURE)).jxc_overlap, "pure");
        assert!(
            DeviceSite::scf(BIG, Some(RSH)).jxc_overlap,
            "range-separated"
        );
        assert!(
            !DeviceSite::scf(BIG, Some(GLOBAL_HYBRID)).jxc_overlap,
            "global hybrid"
        );
        assert!(!DeviceSite::scf(BIG, None).cpu_f64);
        assert!(DeviceSite::scf(BIG, None).with_f64_tensor().cpu_f64);
    }

    #[test]
    fn auto_keeps_a_pure_functional_on_the_cpu_with_the_f64_tensor() {
        for n in [AUTO_MIN_NBF_RI_J_F64, AUTO_MIN_NBF_RI_J] {
            let s = DeviceSite::scf(n, Some(PURE)).with_f64_tensor();
            assert!(
                !routes_to_device(GpuMode::Auto, AutoPath::RiJ, s),
                "auto, nbf {n}"
            );
            assert!(
                routes_to_device(GpuMode::On, AutoPath::RiJ, s),
                "on, nbf {n}"
            );
        }
    }

    #[test]
    fn auto_takes_a_global_hybrid_and_keeps_a_range_separated_one_on_the_cpu() {
        let at =
            |xc, n| routes_to_device(GpuMode::Auto, AutoPath::RiJ, DeviceSite::scf(n, Some(xc)));
        // A B3LYP COSX run has no DF-K aux but builds K beside J.
        assert!(at(GLOBAL_HYBRID, AUTO_MIN_NBF_RI_J));
        assert!(routes_to_device(
            GpuMode::Auto,
            AutoPath::RiJ,
            DeviceSite::scf(AUTO_MIN_NBF_RI_J_F64, Some(GLOBAL_HYBRID)).with_f64_tensor()
        ));
        for n in [AUTO_MIN_NBF_RI_J, BIG] {
            assert!(!at(RSH, n), "range-separated, nbf {n}");
            assert!(!at(PURE, n), "pure, nbf {n}");
        }
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
