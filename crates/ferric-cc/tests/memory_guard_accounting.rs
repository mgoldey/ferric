//! Pre-flight memory guards in `ferric-cc` must charge what the driver
//! actually allocates — in BOTH directions.
//!
//! # The defect class
//!
//! Every method in this crate used to carry a hand-written pre-flight byte sum:
//! a SECOND implementation of its own allocation shapes, kept in sync with the
//! allocator by hand. They drift silently until an OOM. `ferric-core`'s
//! `MemoryPlan` exists to retire that class by making the estimate and the
//! allocation one expression; these tests pin the shapes the old hand-written
//! sums had already drifted past.
//!
//! Two concrete drifts covered here:
//!
//! * `linlccd` and `linlccd_u` charged only their MO-basis blocks and omitted
//!   the dense AO 3-center tensor `eri3_ao` (`naux·nbas²`) entirely — although
//!   the sibling `ccsd.rs` guard charges exactly that term, and `linlccd`'s own
//!   explicit `drop(eri3_ao)` proves the tensor is live across every MO block
//!   build below the guard.
//! * The same omission in `ccd`, covered by that module's own unit tests.
//!
//! # Why these tests do not re-derive the plan's arithmetic
//!
//! Re-implementing the byte sum in the test would reintroduce exactly the
//! second-estimator drift being removed: the test would then agree with a wrong
//! plan as happily as with a right one. Instead each test probes the guard with
//! a budget computed from the INPUT dimensions (`naux·nbas²·8` is a property of
//! the basis sets, not of the plan) and asserts the guard's verdict. Since the
//! driver allocates `eri3_ao` PLUS the MO blocks PLUS the amplitudes, a budget
//! of exactly `eri3_ao`'s own size must be refused by any correct guard — and
//! was ACCEPTED by the old ones.
//!
//! # Over-counting is also a bug
//!
//! An over-estimating guard refuses jobs that would have fit. This pass ADDED
//! terms to several estimates, so every guard tested here also gets an
//! ample-budget case asserting both that the job runs and that the energy is
//! bit-identical to the unbudgeted run: these are accounting changes only, and
//! no numerical result may move.

use ferric_cc::linlccd::{linlccd, LadderVariant};
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::result::ScfResult;
use ferric_scf::screening::SchwarzBounds;

const H2O: &str = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    op: Operator,
    rhf: ScfResult,
}

fn setup(xyz: &str, obs_name: &str, aux_name: &str) -> Setup {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .unwrap();
    Setup { mol, obs, dfbs, op, rhf }
}

/// `eri3_ao`'s own size, from the basis dimensions. NOT a restatement of the
/// plan — this is what the AO 3-center tensor measurably is.
fn eri3_ao_bytes(s: &Setup) -> usize {
    s.dfbs.nbasis() * s.obs.nbasis() * s.obs.nbasis() * 8
}

/// Did the guard REFUSE this budget? Any other outcome (including a converged
/// energy, or an unrelated error) means the guard let the job through.
fn refused(s: &Setup, variant: LadderVariant, budget: usize) -> bool {
    let cfg = CcConfig {
        frozen_core: 0,
        max_iter: 1,
        memory_budget_bytes: Some(budget),
        ..Default::default()
    };
    match linlccd(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &cfg, variant) {
        Err(e) => e.to_string().contains("budget is"),
        Ok(_) => false,
    }
}

/// A budget of exactly `eri3_ao`'s size leaves nothing for the MO blocks or the
/// amplitudes, so every variant must refuse it. The old guard charged neither
/// `eri3_ao` nor the `v_oovv` clone, and accepted it.
#[test]
fn linlccd_guard_charges_eri3_ao_for_every_variant() {
    // A large aux basis makes eri3_ao the dominant term.
    let s = setup(H2O, "sto-3g", "def2-qzvpp-rifit");
    let probe = eri3_ao_bytes(&s);
    for variant in [LadderVariant::DriversOnly, LadderVariant::Hh, LadderVariant::Full] {
        assert!(
            refused(&s, variant, probe),
            "LinLCCD {variant:?} accepted a budget of exactly eri3_ao's own size \
             ({probe} bytes) — that tensor is still uncharged"
        );
    }
}

/// The refusal must NAME the terms. That per-reservation breakdown is the point
/// of the `MemoryPlan` rewrite: a bare "needs X GB, have Y GB" is what made the
/// historical incidents slow to diagnose.
#[test]
fn linlccd_refusal_names_the_terms() {
    let s = setup(H2O, "sto-3g", "def2-qzvpp-rifit");
    let cfg = CcConfig {
        frozen_core: 0,
        max_iter: 1,
        memory_budget_bytes: Some(1024),
        ..Default::default()
    };
    let err = linlccd(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &cfg, LadderVariant::Full)
        .expect_err("a 1 KB budget must be refused");
    let msg = err.to_string();
    assert!(msg.contains("memory plan"), "no plan breakdown: {msg}");
    assert!(msg.contains("eri3_ao"), "breakdown must name eri3_ao: {msg}");
    assert!(msg.contains("LinLCCD"), "breakdown must name the method: {msg}");
}

/// The variant conditioning must be PRESERVED: `DriversOnly` and `Hh` never
/// build the VVVV block, so charging it would refuse systems those variants run
/// comfortably. Asserted by the breakdown's contents rather than by arithmetic.
#[test]
fn linlccd_does_not_charge_blocks_the_variant_never_builds() {
    let s = setup(H2O, "sto-3g", "def2-qzvpp-rifit");
    let refusal = |variant| {
        let cfg = CcConfig {
            frozen_core: 0,
            max_iter: 1,
            memory_budget_bytes: Some(1024),
            ..Default::default()
        };
        linlccd(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &cfg, variant)
            .expect_err("a 1 KB budget must be refused")
            .to_string()
    };

    let drivers = refusal(LadderVariant::DriversOnly);
    assert!(
        !drivers.contains("v_vvvv") && !drivers.contains("v_oooo"),
        "DriversOnly builds neither ladder block; charging one would refuse jobs \
         that fit: {drivers}"
    );

    let hh = refusal(LadderVariant::Hh);
    assert!(hh.contains("v_oooo"), "Hh DOES build the hh ladder: {hh}");
    assert!(
        !hh.contains("v_vvvv"),
        "Hh never forms the VVVV block; charging it would refuse jobs that fit: {hh}"
    );

    let full = refusal(LadderVariant::Full);
    assert!(full.contains("v_vvvv"), "Full DOES build the pp ladder: {full}");
    assert!(full.contains("v_oooo"), "Full DOES build the hh ladder: {full}");
}

/// An ample budget must still run AND give the identical energy. This pass only
/// changed accounting; a guard that moves a number is a bug regardless of which
/// direction it moves it.
#[test]
fn linlccd_ample_budget_runs_and_energy_is_unchanged() {
    let s = setup(H2O, "sto-3g", "cc-pvdz-ri");
    let base = CcConfig { frozen_core: 0, max_iter: 60, energy_conv: 1e-9, ..Default::default() };
    for variant in [LadderVariant::DriversOnly, LadderVariant::Hh, LadderVariant::Full] {
        let unbudgeted =
            linlccd(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &base, variant).unwrap();
        let budgeted = linlccd(
            &s.mol,
            &s.obs,
            &s.dfbs,
            s.op,
            &s.rhf,
            &CcConfig {
                memory_budget_bytes: Some(ferric_core::memory::gib_to_bytes(4.0)),
                ..base
            },
            variant,
        )
        .unwrap_or_else(|e| panic!("4 GiB must be ample for {variant:?}: {e}"));
        assert_eq!(
            budgeted.correlation_energy, unbudgeted.correlation_energy,
            "the memory guard changed the {variant:?} energy"
        );
    }
}
