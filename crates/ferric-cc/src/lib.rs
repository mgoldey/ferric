//! Coupled-Cluster methods for ferric.
//!
//! Spin-orbital RI-CCD, RI-CCSD, and the CCSD(T) perturbative-triples
//! correction, all validated against exact-integral / PySCF references.
//! The infrastructure reuses RI integrals from ferric-mp2.

use ndarray::{Array2, Array4};

/// Spin-orbital coupled-cluster doubles (CCD).
/// Elements held by an amplitude DIIS ring at steady state.
///
/// `Diis::step` clones BOTH of its arguments into SEPARATE ring histories on
/// every call (`ferric_scf::diis` :231-233):
///
/// ```text
/// self.fock_hist.push(f.clone());
/// let new_slot = self.err_hist.push(err.clone());
/// ```
///
/// so a subspace of `n` holds `2n` full-size tensors, not `n`. For the CC
/// amplitude drivers each argument is one `oovv_elems`
/// (`(no2·nv2, no2·nv2)`), giving 12 copies at the default
/// `diis_subspace = 6` — against the 5 the plans charged:
///
/// ```text
///   system             oovv_elems    charged (5x)    DIIS (12x)
///   ethane/cc-pVDZ        1498176        0.060 GB       0.144 GB
///   benzene/cc-pVDZ      61027344        2.441 GB       5.859 GB
/// ```
///
/// `subspace = 0` charges one slot, not none: every driver constructs with
/// `cfg.diis_subspace.max(1)`, so the charge must match what the code does
/// rather than what the config literally says.
pub fn diis_history_elems(per_tensor_elems: usize, subspace: usize) -> usize {
    per_tensor_elems
        .saturating_mul(2)
        .saturating_mul(subspace.max(1))
}

pub mod ccd;
/// Spin-orbital CCSD (coupled-cluster singles and doubles).
pub mod ccsd;
/// Spin-adapted closed-shell CCSD (spatial orbitals, ~8-10x faster).
pub mod ccsd_closed_shell;
/// Spin-orbital CCSD(T) perturbative triples (streaming per-triple-block).
pub mod ccsd_t;
/// Spin-adapted closed-shell CCSD(T) (spatial orbitals, ~10-40x faster).
pub mod ccsd_t_closed_shell;
/// DLPNO-CCSD (domain-based local pair natural orbital CCSD).
pub mod dlpno_ccsd;
/// DLPNO-CCSD amplitude update kernel.
pub mod dlpno_ccsd_kernel;
/// DLPNO-CCSD(T) perturbative triples with local domains.
pub mod dlpno_ccsd_t;
/// DLPNO-CCSD(T) triples kernel.
pub mod dlpno_ccsd_t_kernel;
/// DLPNO-CCSD(T) virtual-domain machinery.
pub mod dlpno_ccsd_t_virtual;
/// DLPNO-CCSD virtual-domain construction.
pub mod dlpno_ccsd_virtual;
/// DLPNO-LinLCCD (linearized local coupled-cluster doubles).
pub mod dlpno_linlccd;
/// Double-hybrid DFT: wB97X-L-V and similar post-KS correlation.
pub mod double_hybrid;
/// Shared CC helpers: MO integrals, amplitude utilities.
pub mod helpers;
/// Linearized LCCD (LinLCCD) from RI intermediates.
pub mod linlccd;
/// LinLCCD amplitude update kernel.
pub mod linlccd_amplitude;
/// LinLCCD via exact (non-RI) integrals (validation only).
pub mod linlccd_exact;
/// Unrestricted LinLCCD (α/β spin channels).
pub mod linlccd_u;
/// Integral-direct (RI) particle-particle ladder for closed-shell CCSD.
pub mod ri_ladder;

pub use ferric_mp2::local_pno as pno;
/// Local-correlation machinery, re-exported from `ferric-mp2`.
///
/// These live in `ferric-mp2` rather than here because MP2 and RPA need them too
/// and `ferric-mp2` is UPSTREAM of this crate — a module here would be unreachable
/// from either. Re-exported so `ferric_cc::pair_domains::…` keeps resolving.
pub use ferric_mp2::pair_domains;
pub use ferric_mp2::pair_energy_screen;

/// Configuration for coupled-cluster iterations (CCD, CCSD).
#[derive(Debug, Clone, PartialEq)]
pub struct CcConfig {
    pub frozen_core: usize,
    pub max_iter: usize,
    pub energy_conv: f64,
    pub diis_start: usize,
    pub diis_subspace: usize,
    /// Optional resident-bytes ceiling for the dense tensors a CC method
    /// materializes. `None` → resolved via
    /// [`ferric_core::memory::resolve_budget_bytes`].
    ///
    /// This field used to be documented as "wiring-only", and for the
    /// local-correlation family that was literally true: it was threaded through
    /// the whole crate and read by nobody on the `dlpno_*` path. The consequence
    /// was a production incident — the LNO-coupled path predicted a 0.055 GB
    /// working set and peaked at 7.3 GB. The predicted number was not wrong
    /// about what it described; it described the *compressed, pair-shaped*
    /// iteration (`HhFlopCount`, `virtual_retention`), which is genuinely tiny,
    /// while the allocator was serving the *dense canonical setup* that has to
    /// run before the first PNO exists. Nothing reconciled the two, so the only
    /// feedback was the OOM killer.
    ///
    /// It is now read on every path that materializes a size-dependent dense
    /// tensor, via [`ferric_core::memory::plan::MemoryPlan`], which fails fast
    /// with a per-reservation breakdown naming the term that blew up:
    ///
    /// * `ccd`, `ccsd`, `ccsd_closed_shell`, `ccsd_t`, `ccsd_t_closed_shell`,
    ///   `linlccd`, `linlccd_u`, `linlccd_exact` — via `check_alloc`.
    /// * [`dlpno_linlccd::dlpno_linlccd_hh`] — a `MemoryPlan` over the dense
    ///   setup, composed with the `S^{P,Q}` cache guard below.
    /// * [`dlpno_ccsd_kernel::PairOverlaps::build_within_budget`] — the
    ///   `O(nocc⁴·nvir²)` pair-pair overlap cache, the worst asymptotic here.
    /// * [`dlpno_ccsd_t_virtual::TripleTnoBasis::build_within_budget`] — the
    ///   pair-PNO and per-triple-TNO transform sets.
    pub memory_budget_bytes: Option<usize>,
    /// How closed-shell CCSD ([`ccsd_closed_shell`]) evaluates the
    /// particle-particle ladder. See [`CcsdLadder`]; the default is
    /// [`CcsdLadder::Auto`]. Read by `ccsd_closed_shell` only — the
    /// spin-orbital solvers have no RI ladder.
    pub ladder: CcsdLadder,
    /// Explicit `a`-block width for the RI particle-particle ladder.
    ///
    /// `None` (the default) takes the widest block the memory plan still fits
    /// (see `ccsd_closed_shell::plan_ladder`). Set it only to pin a width — the
    /// tests use it to force narrow and deliberately RAGGED blocks (widths that
    /// do not divide `nv`) without having to starve the whole calculation's
    /// budget. It requires `ladder: CcsdLadder::Ri` (a pinned width on a route
    /// that may not use it would be a silently ignored knob), and a value of 0
    /// or one exceeding `nv` is a hard error rather than a clamp.
    pub ri_ladder_block: Option<usize>,
}

/// How closed-shell CCSD evaluates the particle-particle ladder
/// `Σ_cd W_abcd τ_ij^cd`.
///
/// Both routes compute the same number from the same RI tensors: the RI route
/// is an exact algebraic refactor of the dense `W_abcd`, not an approximation
/// (H2O/cc-pVDZ: `|ΔE|` and `max|Δt|` ≲ 3e-17, i.e. round-off). They differ
/// only in memory and time:
///
/// * [`Dense`](CcsdLadder::Dense) holds the chemist `(ab|cd)` block for the
///   whole run and builds `W_abcd` every iteration — `nv⁴`-class residents
///   plus three `nv⁴` temporaries.
/// * [`Ri`](CcsdLadder::Ri) holds neither the `(ab|cd)` block nor `W_abcd`;
///   it rebuilds the `(ac|bd)` entries from the 3-index blocks each iteration
///   in `blk·nv³` panels ([`crate::ri_ladder`]) — at most one `nv⁴`-sized
///   panel at full width (dense holds four), narrower panels when the budget
///   requires. The rebuild costs about `naux/no` times the dense path's
///   per-iteration `W` build (measured 1.3–1.5× per iteration at `nv ≥ 69`),
///   so it is a memory-for-time trade.
/// * [`Auto`](CcsdLadder::Auto) (the default) takes Dense whenever the dense
///   memory plan fits the budget and RI otherwise, so it is never refused
///   where RI would run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CcsdLadder {
    /// Dense when its memory plan fits the budget, RI otherwise.
    #[default]
    Auto,
    /// Always the in-core `nv⁴` ladder.
    Dense,
    /// Always the integral-direct (RI) ladder.
    Ri,
}

impl CcsdLadder {
    /// Strict parser for the CLI/Python spelling: `"auto"`, `"dense"` or
    /// `"ri"`. Anything else is an error, never a silent default.
    pub fn parse_config_str(s: &str) -> Result<Self, ferric_core::FerricError> {
        match s {
            "auto" => Ok(Self::Auto),
            "dense" => Ok(Self::Dense),
            "ri" => Ok(Self::Ri),
            other => Err(ferric_core::FerricError::General(format!(
                "unknown CCSD ladder {other:?}: expected \"auto\", \"dense\" or \"ri\""
            ))),
        }
    }

    /// The config spelling, the inverse of [`CcsdLadder::parse_config_str`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Dense => "dense",
            Self::Ri => "ri",
        }
    }
}

impl Default for CcConfig {
    fn default() -> Self {
        Self {
            frozen_core: 0,
            max_iter: 50,
            energy_conv: 1e-8,
            diis_start: 5,
            diis_subspace: 6,
            memory_budget_bytes: None,
            ladder: CcsdLadder::Auto,
            ri_ladder_block: None,
        }
    }
}

impl CcConfig {
    /// Set the number of frozen core orbitals.
    pub fn with_frozen_core(mut self, n: usize) -> Self {
        self.frozen_core = n;
        self
    }
    /// Set the maximum number of CC iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    /// Set the energy convergence threshold (Ha).
    pub fn with_energy_conv(mut self, thresh: f64) -> Self {
        self.energy_conv = thresh;
        self
    }
    /// Set the memory budget in bytes.
    pub fn with_memory_budget_bytes(mut self, bytes: usize) -> Self {
        self.memory_budget_bytes = Some(bytes);
        self
    }
}

/// Converged coupled-cluster result: correlation energy and amplitudes.
#[derive(Debug, Clone)]
#[must_use]
pub struct CcResult {
    pub correlation_energy: f64,
    pub t1: Option<Array2<f64>>,
    pub t2: Array4<f64>,
    /// The particle-particle ladder route closed-shell CCSD took (with
    /// [`CcsdLadder::Auto`] resolved). `None` for every other solver.
    pub ladder_route: Option<ccsd_closed_shell::LadderRoute>,
}

impl std::fmt::Display for CcResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let method = if self.t1.is_some() { "CCSD" } else { "CCD" };
        write!(
            f,
            "{} correlation: {:.10} Ha",
            method, self.correlation_energy
        )
    }
}
