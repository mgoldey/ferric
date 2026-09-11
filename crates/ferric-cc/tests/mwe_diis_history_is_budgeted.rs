//! MWE: the CC amplitude DIIS history must be charged to the memory budget.
//!
//! # The defect
//!
//! `linlccd.rs` reserves seven `oovv`-sized buffers and stops:
//!
//! ```text
//! plan.reserve("v_oovv <ij||ab> + oovv_t clone", oovv_elems * 2, Resident);
//! plan.reserve("d denominator + t/r/x amplitude working set (x5)",
//!              oovv_elems * 5, Resident);
//! plan.check()?;
//! ```
//!
//! Then, with no further reservation, it builds a DIIS ring:
//!
//! ```text
//! let mut diis = ferric_scf::diis::Diis::new(cfg.diis_subspace.max(1));
//! ...
//! let t_ext = diis.step(&t_flat, &err_flat);
//! ```
//!
//! and `diis.rs:231-233` clones **both** arguments into **two** separate ring
//! histories on every call:
//!
//! ```text
//! self.fock_hist.push(f.clone());
//! let new_slot = self.err_hist.push(err.clone());
//! ```
//!
//! `t_flat` and `err_flat` are each `(no2·nv2, no2·nv2)`, i.e. exactly one
//! `oovv_elems`. With the default `diis_subspace = 6` (`lib.rs:100`) the
//! steady-state history is **12 full amplitude tensors** — against the 5 the
//! plan charges:
//!
//! ```text
//!   system             oovv_elems    charged (5x)    DIIS (12x)
//!   ethane/cc-pVDZ        1498176        0.060 GB       0.144 GB
//!   benzene/cc-pVDZ      61027344        2.441 GB       5.859 GB
//! ```
//!
//! 2.4x the charged working set, uncharged. And the plan's `check()` runs
//! BEFORE the loop that fills the ring, so nothing later notices.
//!
//! All seven ferric-cc drivers that run amplitudes share the shape
//! (`ccd`, `ccsd`, `ccsd_closed_shell`, `linlccd`, `linlccd_u`,
//! `linlccd_exact`, `dlpno_linlccd`).
//!
//! # Scope
//!
//! Pins the ARITHMETIC of the charge — `diis_history_elems` — so it can be
//! tested without running a CC iteration. Whether each driver calls it is a
//! separate, per-driver question.

use ferric_cc::diis_history_elems;

/// benzene/cc-pVDZ-scale spin-orbital dimensions.
const NO2: usize = 42;
const NV2: usize = 186;

fn oovv_elems() -> usize {
    NO2 * NO2 * NV2 * NV2
}

/// CONTRACT 1: the charge is TWO tensors per subspace slot.
///
/// The crux. `step` pushes into `fock_hist` AND `err_hist`, so a charge of one
/// per slot under-counts by exactly half — and half of 5.9 GB is still 2.9 GB
/// unaccounted at benzene scale.
#[test]
fn the_charge_is_two_tensors_per_subspace_slot() {
    let one = oovv_elems();
    assert_eq!(
        diis_history_elems(one, 6),
        12 * one,
        "diis_subspace = 6 holds 6 amplitude copies AND 6 error copies — \
         diis.rs:231-233 pushes both on every step()."
    );
}

/// CONTRACT 2: it scales with the subspace size.
///
/// A constant charge would be wrong in both directions: too small at the
/// default 6, and needlessly refusing jobs at `diis_subspace = 1`.
#[test]
fn the_charge_scales_with_the_subspace() {
    let one = oovv_elems();
    assert!(
        diis_history_elems(one, 8) > diis_history_elems(one, 2),
        "a larger DIIS subspace holds more history and must cost more"
    );
    assert_eq!(diis_history_elems(one, 1), 2 * one, "one slot is still two tensors");
}

/// CONTRACT 3: a zero subspace is treated as one slot.
///
/// The drivers all construct with `cfg.diis_subspace.max(1)`, so a configured
/// 0 still allocates one slot's worth. The charge must match what the code
/// actually does, not what the config literally says — otherwise a
/// `diis_subspace = 0` run is under-charged by two full tensors.
#[test]
fn a_zero_subspace_still_charges_one_slot() {
    let one = oovv_elems();
    assert_eq!(
        diis_history_elems(one, 0),
        2 * one,
        "the drivers build with `.max(1)`, so 0 means one slot, not none"
    );
}

/// CONTRACT 4: the charge is material next to the working set.
///
/// The reachability guard. If the DIIS history were a rounding term beside the
/// 5 charged buffers, adding it would be noise and CONTRACTS 1-3 would be
/// pinning something that does not matter. At the default subspace it is 2.4x
/// the working set, which is the whole reason this finding is high severity.
#[test]
fn the_history_dominates_the_charged_working_set() {
    let one = oovv_elems();
    let working_set = 5 * one;
    let history = diis_history_elems(one, 6);
    assert!(
        history > 2 * working_set,
        "the DIIS history ({history} elems) should dominate the charged working set \
         ({working_set} elems) at the default subspace — if it did not, this would not be \
         worth a budget term"
    );
}
