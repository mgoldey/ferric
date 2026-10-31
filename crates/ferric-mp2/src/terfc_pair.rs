//! MP2-V and SCS-MP2(2terfc) (plus optional plain RI-MP2) from ONE converged
//! closed-shell RHF reference — the S66 benchmark's shared-SCF entry point
//! (mgoldey/ferric#278).
//!
//! # What is shared, and what cannot be
//!
//! The per-method entry points ([`crate::att_vv10::att_mp2_vv10`],
//! [`crate::scs::scs_mp2_2terfc`], [`crate::rimp2::ri_mp2`]) already take a
//! converged `ScfResult`; it is the per-method Python/CLI drivers that each
//! re-run the SCF. This function exists so a benchmark can pay for the SCF
//! (and the `PreparedBasis` / Schwarz setup) once per system instead of once
//! per method.
//!
//! The 3-index integrals are **not** shared, and cannot be without changing
//! the method: RI-MP2 under an operator `op` fits `(P|op|μν)` against the
//! metric `(P|op|Q)` (`RiMp2Config::metric_op = None` → `op`). The four
//! operators in play are all different —
//!
//! | quantity | operator |
//! |---|---|
//! | MP2-V attenuated half | terfc(r₀ = 1.00 Å) |
//! | SCS-MP2(2terfc) bonded | terfc(r₀(1) = 0.75 Å) |
//! | SCS-MP2(2terfc) non-bonded | terfc(r₀(2) = 1.05 Å) |
//! | plain RI-MP2 control | Coulomb |
//!
//! — so each needs its own `(P|op|μν)` build and metric. Re-using one
//! (e.g. a Coulomb-metric fit contracted with an attenuated 2-centre kernel)
//! is a different approximation and would not reproduce the published
//! per-method numbers. The two published parameterizations also share no r₀.
//! What both methods DO share: the RHF reference (energy, MOs, density), the
//! orbital and auxiliary basis, and the frozen-core count — all enforced here.
//!
//! # Exactness
//!
//! Each half is computed by calling the existing public function with the
//! caller's config on the shared reference, so given the same `ScfResult` the
//! output is bit-identical to the separate calls by construction. The
//! non-trivial claim — that one SCF reproduces the per-method drivers' SCFs —
//! is pinned by `tests/terfc_pair_shared_scf.rs` (fresh SCF per method vs one
//! shared SCF, bitwise).

use std::time::Instant;

use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::Spin;
use ferric_scf::ScfResult;

use crate::att_vv10::{att_mp2_vv10, AttVv10Config, AttVv10Result};
use crate::rimp2::{ri_mp2, RiMp2Config, RiMp2Result};
use crate::scs::{scs_mp2_2terfc, ScsMp2Result, ScsMp2TerfcConfig};

/// Wall-clock seconds per post-SCF phase of [`mp2_v_and_scs_2terfc`].
///
/// Reported so a benchmark can refit its cost model from production runs
/// (the SCF time is the caller's to measure — it happens before this call).
#[derive(Debug, Clone, Copy, Default)]
pub struct TerfcPairTimings {
    /// MP2-V: one terfc RI-MP2 plus the damped VV10 grid integration.
    pub mp2_v_s: f64,
    /// SCS-MP2(2terfc): two terfc RI-MP2 evaluations.
    pub scs_2terfc_s: f64,
    /// Plain Coulomb RI-MP2 (0 when not requested).
    pub mp2_coulomb_s: f64,
}

/// Output of [`mp2_v_and_scs_2terfc`].
#[derive(Debug, Clone)]
#[must_use]
pub struct TerfcPairResult {
    /// Reference RHF total energy shared by every method below.
    pub e_hf: f64,
    /// MP2-V, exactly as [`att_mp2_vv10`] returns it on this reference.
    pub mp2_v: AttVv10Result,
    /// SCS-MP2(2terfc), exactly as [`scs_mp2_2terfc`] returns it.
    pub scs_2terfc: ScsMp2Result,
    /// Plain Coulomb RI-MP2 ([`ri_mp2`]) on the same reference, aux basis and
    /// frozen core, when requested.
    pub mp2_coulomb: Option<RiMp2Result>,
    /// Per-phase wall-clock seconds.
    pub timings: TerfcPairTimings,
}

/// MP2-V and SCS-MP2(2terfc) — and optionally plain RI-MP2 — on ONE
/// converged closed-shell RHF reference.
///
/// The two configs must agree on `frozen_core` and `memory_budget_bytes`
/// (hard error otherwise): a benchmark that froze a different core in each
/// method would compare two different correlation spaces. The plain RI-MP2
/// control runs under the same `frozen_core` and budget with every other
/// [`RiMp2Config`] field at its default — the configuration the `run_rimp2`
/// binding uses for the exact closed-shell method.
#[allow(clippy::too_many_arguments)]
pub fn mp2_v_and_scs_2terfc(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    mp2_v_config: &AttVv10Config,
    scs_config: &ScsMp2TerfcConfig,
    include_coulomb_mp2: bool,
) -> Result<TerfcPairResult, FerricError> {
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(format!(
            "mp2_v_and_scs_2terfc requires a closed-shell restricted (RHF) reference, got {:?}",
            rhf.spin
        )));
    }
    if mp2_v_config.frozen_core != scs_config.frozen_core {
        return Err(FerricError::General(format!(
            "mp2_v_and_scs_2terfc: frozen_core differs between the MP2-V ({}) and \
             SCS-MP2(2terfc) ({}) configs; a shared reference with two correlation \
             spaces is not a like-for-like comparison",
            mp2_v_config.frozen_core, scs_config.frozen_core
        )));
    }
    if mp2_v_config.memory_budget_bytes != scs_config.memory_budget_bytes {
        return Err(FerricError::General(format!(
            "mp2_v_and_scs_2terfc: memory_budget_bytes differs between the MP2-V ({:?}) and \
             SCS-MP2(2terfc) ({:?}) configs",
            mp2_v_config.memory_budget_bytes, scs_config.memory_budget_bytes
        )));
    }

    let t = Instant::now();
    let mp2_v = att_mp2_vv10(mol, obs, obs_bs, dfbs, rhf, mp2_v_config)?;
    let mp2_v_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let scs_2terfc = scs_mp2_2terfc(mol, obs, dfbs, rhf, scs_config)?;
    let scs_2terfc_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mp2_coulomb = if include_coulomb_mp2 {
        let cfg = RiMp2Config {
            frozen_core: scs_config.frozen_core,
            memory_budget_bytes: scs_config.memory_budget_bytes,
            ..Default::default()
        };
        Some(ri_mp2(mol, obs, dfbs, Operator::coulomb(), rhf, &cfg)?)
    } else {
        None
    };
    let mp2_coulomb_s = if include_coulomb_mp2 {
        t.elapsed().as_secs_f64()
    } else {
        0.0
    };

    Ok(TerfcPairResult {
        e_hf: rhf.energy,
        mp2_v,
        scs_2terfc,
        mp2_coulomb,
        timings: TerfcPairTimings {
            mp2_v_s,
            scs_2terfc_s,
            mp2_coulomb_s,
        },
    })
}
