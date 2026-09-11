//! MWE: `amplitude_linlccd_with_virtuals` must gate its own dense terms.
//!
//! # The defect
//!
//! `linlccd_amplitude.rs` has four public entry points
//! (`amplitude_linlccd`, `amplitude_linlccd_with_virtuals`,
//! `amplitude_linlccd_direct`, `amplitude_linlccd_direct_with_virtuals`) and,
//! before this fix, NONE of them called `check_alloc`/`MemoryPlan` — even
//! though `amplitude_linlccd_with_virtuals` directly builds:
//!
//! * `eri3_ao` (`naux·nbas²`) — TWICE (once for `oo_g`, again for `bvv_t`
//!   under `LadderVariant::Full`), each transient and non-overlapping;
//! * `oo_g`, the whitened `(ik|jl)` Gram, `no²·no²` and RESIDENT;
//! * `bvv_t`, `naux·nv²` and RESIDENT, under `Full`.
//!
//! This test pins the gate that was added at the top of
//! `amplitude_linlccd_with_virtuals` in `crates/ferric-cc/src/linlccd_amplitude.rs`.
//!
//! # Scope
//!
//! The sibling entry points `amplitude_linlccd_direct` /
//! `amplitude_linlccd_direct_with_virtuals` are OUT of this test's scope on
//! purpose: their dense-tensor-avoiding design means the only allocations of
//! comparable scale happen inside `ferric_mp2::lmp2_direct` helpers
//! (`assemble_boo_direct`, `assemble_pp_fitted_direct`), which are a
//! different crate and have no memory gating of their own — a real gap, but
//! one that cannot be closed from `ferric-cc` alone. See the driver's
//! module doc / this session's report for that refutation-shaped finding.

use ferric_cc::linlccd::LadderVariant;
use ferric_cc::linlccd_amplitude::{amplitude_linlccd_with_virtuals, AmplitudeLinLccdConfig};
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::build_vvhv;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn h2o() -> Molecule {
    Molecule::parse_xyz(
        "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .unwrap()
}

/// CONTRACT 1: a tiny budget is refused, and the error names the method.
#[test]
fn amplitude_linlccd_fails_fast_under_tiny_budget() {
    let mol = h2o();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .unwrap();
    let vvhv = build_vvhv(&mol, &obs, &obs_bs, &rhf).unwrap();
    let cfg = AmplitudeLinLccdConfig {
        eri3_budget_bytes: Some(ferric_core::memory::gib_to_bytes(1e-6)),
        ..Default::default()
    };
    let err = match amplitude_linlccd_with_virtuals(
        &mol,
        &obs,
        &dfbs,
        op,
        &rhf,
        &cfg,
        LadderVariant::Full,
        &vvhv,
    ) {
        Err(e) => e,
        Ok(_) => panic!("amplitude LinLCCD should fail fast under tiny budget"),
    };
    let msg = err.to_string();
    assert!(msg.contains("LinLCCD") && msg.contains("budget is"), "unexpected: {msg}");
    assert!(msg.contains("memory plan"), "no plan breakdown: {msg}");
}

/// CONTRACT 2 (REACHABILITY): `bvv_t` is charged under `Full` — a budget
/// that holds `oo_g` + one `eri3_ao` but NOT `bvv_t` must be refused for
/// `Full`, even though the same driver never touches `bvv_t` for `Hh`.
#[test]
fn amplitude_linlccd_full_charges_bvv_t() {
    let mol = h2o();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .unwrap();
    let vvhv = build_vvhv(&mol, &obs, &obs_bs, &rhf).unwrap();

    let naux = dfbs.nbasis();
    let nbas = obs.nbasis();
    let nocc_total = (mol.nelec() as usize) / 2;
    let nv = nbas - nocc_total; // matches lb.nv (no frozen core here)

    // A budget that holds eri3_ao (naux*nbas^2) comfortably but is smaller
    // than eri3_ao + bvv_t (naux*nv^2) combined: sized as 1.2x eri3_ao,
    // which is less than eri3_ao + bvv_t for this system (bvv_t is not tiny
    // relative to eri3_ao once nv approaches nbas).
    let eri3_bytes = naux * nbas * nbas * 8;
    let bvv_bytes = naux * nv * nv * 8;
    let budget = eri3_bytes + bvv_bytes / 2; // between "just eri3_ao" and "eri3_ao + bvv_t"

    let cfg = AmplitudeLinLccdConfig { eri3_budget_bytes: Some(budget), ..Default::default() };

    let refused_full = match amplitude_linlccd_with_virtuals(
        &mol,
        &obs,
        &dfbs,
        op,
        &rhf,
        &cfg,
        LadderVariant::Full,
        &vvhv,
    ) {
        Err(e) => e.to_string().contains("budget is"),
        Ok(_) => false,
    };
    assert!(
        refused_full,
        "Full variant accepted a budget ({budget} bytes) below eri3_ao ({eri3_bytes}) + \
         bvv_t ({bvv_bytes}) — bvv_t is still uncharged"
    );
}

/// CONTRACT 3 (OVER-REJECTION): an ample budget must still run to
/// completion for both Hh and Full.
#[test]
fn amplitude_linlccd_ample_budget_still_runs() {
    let mol = h2o();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { energy_conv: 1e-10, ..Default::default() },
    )
    .unwrap();
    let vvhv = build_vvhv(&mol, &obs, &obs_bs, &rhf).unwrap();
    let cfg = AmplitudeLinLccdConfig {
        eri3_budget_bytes: Some(ferric_core::memory::gib_to_bytes(4.0)),
        ..Default::default()
    };
    for variant in [LadderVariant::Hh, LadderVariant::Full] {
        let r = amplitude_linlccd_with_virtuals(&mol, &obs, &dfbs, op, &rhf, &cfg, variant, &vvhv)
            .unwrap_or_else(|e| panic!("an ample 4 GiB budget must not be refused ({variant:?}): {e}"));
        assert!(r.e_corr.is_finite());
    }
}
