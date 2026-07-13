//! Opt-in stage timing. Zero cost unless `FERRIC_TIMING` is set in the
//! environment — then each `Stage` prints `[timing] <label> <ms>` to stderr at
//! `.end()`. Used to MEASURE (not estimate) where the GW/RPA pipeline spends
//! wall-time, so optimization decisions (e.g. sharing PDEP setup across GW
//! columns) are backed by numbers. See crates/ferric-gw/examples/gw_profile.rs.

use std::sync::OnceLock;
use std::time::Instant;

/// `FERRIC_TIMING` descriptor: opt-in stage timing (env-only debug toggle).
/// Routed through the shared `parse_toggle` so `FERRIC_TIMING=0` means OFF,
/// matching every other ferric toggle. (This is a deliberate behavior change
/// from the prior `var_os(..).is_some()` read, under which `FERRIC_TIMING=0`
/// was ON — the cross-flag "0 means off everywhere" consistency fix.) Resolved
/// once and cached in a `OnceLock` since it gates a hot-path timer.
static TIMING: ferric_core::config::ConfigVar<bool> = ferric_core::config::ConfigVar {
    env_name: "FERRIC_TIMING",
    default: false,
    parse: ferric_core::config::parse_toggle,
    validate: ferric_core::config::accept_any,
};

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| TIMING.toggle())
}

/// A scoped stage timer. Construct with `start`, call `end` when the stage
/// finishes. No-op (and no allocation of the label cost beyond the &'static str)
/// when `FERRIC_TIMING` is unset.
pub struct Stage {
    label: &'static str,
    t0: Option<Instant>,
}

impl Stage {
    #[inline]
    pub fn start(label: &'static str) -> Self {
        Stage {
            label,
            t0: if enabled() { Some(Instant::now()) } else { None },
        }
    }

    #[inline]
    pub fn end(self) {
        if let Some(t0) = self.t0 {
            eprintln!(
                "[timing] {:<48} {:>9.1} ms",
                self.label,
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}
