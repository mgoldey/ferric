//! Mixed-precision knobs. `precision = f64` (default) means every device GEMM is
//! plain IEEE dgemm. `precision = mixed` enables, FOR THE KERNELS NAMED IN
//! `mixed_kernels` ONLY, the k-panelled SGEMM with f64 accumulation
//! (`crate::gpu::mixed`). There is no global f32 flip: a kernel is a name here
//! only once its error-budget row (see the Phase 4 plan §3.2) has shipped, and
//! kernels on the never-f32 list (Fock diagonalisation, DIIS, V^{-1/2}, metric
//! lindep, GW Newton, dielectric log-det, Becke weights, meta-GGA tau) are not
//! names at all, which is how "cannot opt in" is enforced.
//!
//! `einsum!` callers opt a block in with [`MixedScope::enter`]; the dispatch in
//! `ferric_tensors::einsum::try_device_gemm` consults [`MixedScope::current`]
//! AND the allowlist. A GEMM outside any scope is f64 even under `mixed`.
use std::cell::Cell;
use std::fmt;
use std::str::FromStr;

use crate::config::{accept_any, ConfigVar};

/// The default precision, in ONE place. Every path that resolves the default
/// (`FERRIC_GPU_PRECISION` unset, the CLI `[gpu]` table without `precision`, the
/// lazy library fallback, the Python status) reads this constant, so changing
/// the default is a one-line edit here.
pub const PRECISION_DEFAULT: Precision = Precision::F64;

/// Which arithmetic the device GEMMs use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// cublasDgemm throughout (the current default, see [`PRECISION_DEFAULT`]).
    F64,
    /// f32 storage + k-panelled cublasSgemm with f64 accumulation, for the
    /// kernels in the allowlist only.
    Mixed,
}

impl FromStr for Precision {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f64" => Ok(Precision::F64),
            "mixed" => Ok(Precision::Mixed),
            _ => Err(format!(
                "invalid GPU precision {s:?} (expected f64 or mixed)"
            )),
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Precision::F64 => "f64",
            Precision::Mixed => "mixed",
        })
    }
}

/// A kernel that MAY run in mixed precision. The discriminant is its bit in
/// [`MixedKernelSet`]; the name is the TOML/env spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MixedKernel {
    /// RI-MP2 energy: `G_i = B_i^T · B_tail` with f32-resident `B_ov` (Task 4.2b).
    RiMp2Energy = 1,
    /// Closed-shell CCSD amplitude-update contractions (Task 4.3).
    CcsdAmplitudes = 2,
    /// DF-K occupied path with f32-resident dressed B (Task 4.4).
    DfkOcc = 4,
}

impl MixedKernel {
    pub const ALL: [MixedKernel; 3] = [
        MixedKernel::RiMp2Energy,
        MixedKernel::CcsdAmplitudes,
        MixedKernel::DfkOcc,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            MixedKernel::RiMp2Energy => "rimp2-energy",
            MixedKernel::CcsdAmplitudes => "ccsd-amplitudes",
            MixedKernel::DfkOcc => "dfk-occ",
        }
    }
}

impl FromStr for MixedKernel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let t = s.trim();
        MixedKernel::ALL
            .into_iter()
            .find(|k| k.name() == t)
            .ok_or_else(|| {
                let valid: Vec<&str> = MixedKernel::ALL.iter().map(|k| k.name()).collect();
                format!(
                    "unknown mixed-precision kernel {t:?} (valid: {})",
                    valid.join(", ")
                )
            })
    }
}

/// A set of [`MixedKernel`]s as a bit mask (so `GpuSettings` stays `Copy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MixedKernelSet(u8);

impl MixedKernelSet {
    pub const EMPTY: Self = Self(0);
    /// The kernels whose error-budget row has shipped. Each Phase 4 task that
    /// ships a row extends this constant in the same commit as its gates:
    /// Task 4.2b adds `RiMp2Energy`, Task 4.3 `CcsdAmplitudes`, Task 4.4 `DfkOcc`.
    /// It is the default of `FERRIC_GPU_MIXED_KERNELS`; while it is empty,
    /// `precision = mixed` is refused.
    pub const SHIPPED: Self = Self::EMPTY;

    pub const fn with(self, k: MixedKernel) -> Self {
        Self(self.0 | k as u8)
    }
    pub const fn contains(self, k: MixedKernel) -> bool {
        self.0 & (k as u8) != 0
    }
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub fn iter(self) -> impl Iterator<Item = MixedKernel> {
        MixedKernel::ALL
            .into_iter()
            .filter(move |&k| self.contains(k))
    }
}

impl FromStr for MixedKernelSet {
    type Err = String;
    /// Comma-separated names; whitespace around names is ignored; an empty
    /// string or an empty item is an error (an explicit empty list cannot mean
    /// "default").
    fn from_str(s: &str) -> Result<Self, String> {
        let mut set = MixedKernelSet::EMPTY;
        let mut any = false;
        for item in s.split(',') {
            let k: MixedKernel = item.parse()?;
            set = set.with(k);
            any = true;
        }
        if !any {
            return Err("empty mixed-precision kernel list".into());
        }
        Ok(set)
    }
}

impl fmt::Display for MixedKernelSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("none");
        }
        let names: Vec<&str> = self.iter().map(|k| k.name()).collect();
        f.write_str(&names.join(","))
    }
}

pub static GPU_PRECISION: ConfigVar<Precision> = ConfigVar {
    env_name: "FERRIC_GPU_PRECISION",
    default: PRECISION_DEFAULT,
    parse: |s| s.parse::<Precision>(),
    validate: accept_any,
};

pub static GPU_MIXED_KERNELS: ConfigVar<MixedKernelSet> = ConfigVar {
    env_name: "FERRIC_GPU_MIXED_KERNELS",
    default: MixedKernelSet::SHIPPED,
    parse: |s| s.parse::<MixedKernelSet>(),
    validate: accept_any,
};

/// Env-only measurement/mutation knob: the k-panel width of the mixed GEMM
/// (0 = the compiled `MIXED_K_PANEL_DEFAULT`). It changes the last digits of a
/// mixed result and exists so the panel sweep and the "f32 accumulator" mutant
/// (`FERRIC_GPU_MIXED_K_PANEL=1000000000`, i.e. one panel) need no rebuild.
static GPU_MIXED_K_PANEL: ConfigVar<usize> = ConfigVar {
    env_name: "FERRIC_GPU_MIXED_K_PANEL",
    default: 0,
    parse: |s| {
        s.trim()
            .parse::<usize>()
            .map_err(|e| format!("invalid panel width {s:?}: {e}"))
    },
    validate: accept_any,
};

/// `Some(width)` when `FERRIC_GPU_MIXED_K_PANEL` is set to a non-zero value.
pub fn mixed_k_panel_override() -> Option<usize> {
    match GPU_MIXED_K_PANEL.resolve(None, crate::config::env_lookup) {
        Ok(r) if r.value > 0 => Some(r.value),
        _ => None,
    }
}

thread_local! {
    static SCOPE: Cell<Option<MixedKernel>> = const { Cell::new(None) };
}

/// RAII marker: while alive on this thread, `einsum!` GEMMs may use the mixed
/// path if the kernel is allowed. Nests; the previous value is restored on
/// drop, including during unwinding.
#[must_use = "the scope ends when this guard is dropped"]
pub struct MixedScope {
    prev: Option<MixedKernel>,
}

impl MixedScope {
    pub fn enter(k: MixedKernel) -> Self {
        let prev = SCOPE.replace(Some(k));
        Self { prev }
    }
    pub fn current() -> Option<MixedKernel> {
        SCOPE.get()
    }
}

impl Drop for MixedScope {
    fn drop(&mut self) {
        SCOPE.set(self.prev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_round_trips_through_display_and_parse() {
        let s = MixedKernelSet::EMPTY
            .with(MixedKernel::DfkOcc)
            .with(MixedKernel::RiMp2Energy);
        assert_eq!(s.to_string(), "rimp2-energy,dfk-occ");
        assert_eq!(s.to_string().parse::<MixedKernelSet>().unwrap(), s);
    }

    #[test]
    fn shipped_is_a_subset_of_all() {
        for k in MixedKernelSet::SHIPPED.iter() {
            assert!(MixedKernel::ALL.contains(&k));
        }
    }
}
