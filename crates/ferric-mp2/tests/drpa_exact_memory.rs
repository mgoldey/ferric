//! Memory accounting of the EXACT (ε = 0) dRPA Riccati path.
//!
//! The ε = 0 solve is exact dRPA (anchored to the canonical plasmon formula
//! in `drpa_amplitude.rs`), and it is the memory ceiling of the solver: with a
//! full pattern the ring-product plan alone is `no` times the size of B (C12
//! thrashed, then was OOM-killed). Two things are pinned here:
//!
//! 1. `exact_drpa_peak_bytes(no, nv, diis)` — the closed form callers use to
//!    refuse an exact run BEFORE the SCF — equals the counted
//!    `ragged_bytes + riccati_solve_bytes` of a real ε = 0 assembly. A wrong
//!    closed form would let a too-large exact run through, or refuse one that
//!    fits.
//! 2. The solve's HARD pool charge is real: a pool smaller than the solve's
//!    own allocation refuses with this solve's label, and an ample pool
//!    admits and returns to a zero ledger.
//!
//! Artifact hypothesis, stated before measuring: if the charge were not
//! wired (or charged some other plane), the tight pool below — sized from
//! `riccati_solve_bytes` itself, so it covers every upstream plane — would
//! admit the run, and the refusal-label assertion would fail.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::drpa_amplitude::{
    amplitude_drpa, exact_drpa_peak_bytes, ragged_bytes, riccati_solve_bytes, AmplitudeDrpaConfig,
};
use ferric_mp2::lmp2_amplitude::{
    assemble_basis, assemble_ragged_direct, build_vvhv, AmplitudeLmp2Config,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn global_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    match L.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    obs_bs: basis::BasisSet,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
}

fn water_631g() -> Setup {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/water.xyz",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
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
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    Setup {
        mol,
        obs,
        obs_bs,
        dfbs,
        rhf,
    }
}

/// The ragged pair space the dRPA path assembles at `eps` (B = 2(ia|jb), no
/// pair gate), with `frozen_core` frozen.
fn ragged_at(
    su: &Setup,
    eps: f64,
    frozen_core: usize,
) -> (ferric_mp2::ragged::Ragged, usize, usize) {
    let vvhv = build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
    let lcfg = AmplitudeLmp2Config {
        eps,
        frozen_core,
        ..Default::default()
    };
    let lb = assemble_basis(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &lcfg,
        &vvhv,
    )
    .unwrap();
    let (no, nv) = (lb.no, lb.nv);
    let (rg, _) = assemble_ragged_direct(
        &su.mol,
        &su.dfbs,
        Operator::coulomb(),
        &lb,
        eps,
        2.0,
        None,
        None,
    )
    .unwrap();
    (rg, no, nv)
}

/// Closed form == counted at ε = 0, for DIIS on and off and two frozen-core
/// settings (so `no` enters the plan term twice over). At a finite ε the
/// counted size must be strictly SMALLER — a closed form that ignored ε would
/// be a correct but useless upper bound, while one that UNDER-counted the
/// full pattern would show up as `counted > closed` here.
#[test]
fn exact_closed_form_equals_the_counted_assembly_and_solve() {
    let su = water_631g();
    for frozen_core in [0usize, 1] {
        let (rg, no, nv) = ragged_at(&su, 0.0, frozen_core);
        assert_eq!(rg.pairs.len(), no * no, "eps = 0 keeps every ordered pair");
        for diis in [None, Some(8)] {
            let counted = ragged_bytes(&rg) + riccati_solve_bytes(&rg, diis);
            let closed = exact_drpa_peak_bytes(no, nv, diis);
            assert_eq!(
                counted, closed,
                "fc={frozen_core} diis={diis:?}: counted {counted} vs closed form {closed}"
            );
        }
        let (rg_loose, _, _) = ragged_at(&su, 1e-2, frozen_core);
        let loose = ragged_bytes(&rg_loose) + riccati_solve_bytes(&rg_loose, Some(8));
        assert!(
            loose < exact_drpa_peak_bytes(no, nv, Some(8)),
            "a truncated pattern must cost less than the exact one ({loose})"
        );
    }
}

/// The ring plan dominates at ε = 0 and grows with `no` relative to B: the
/// reason the exact path, not the local one, hits the memory ceiling.
#[test]
fn exact_peak_grows_faster_than_the_amplitudes() {
    let b = |no: usize, nv: usize| 8 * no * no * nv * nv;
    let r_small = exact_drpa_peak_bytes(10, 50, Some(8)) as f64 / b(10, 50) as f64;
    let r_large = exact_drpa_peak_bytes(40, 200, Some(8)) as f64 / b(40, 200) as f64;
    assert!(
        r_large > r_small + 25.0,
        "peak/B must grow with no: {r_small:.1} -> {r_large:.1}"
    );
}

/// The solve's hard charge: a pool too small for `riccati_solve_bytes` (but
/// large enough for everything else this path charges) refuses with the
/// solve's own label; an ample pool admits and drains to zero.
#[test]
fn the_riccati_solve_charges_the_pool() {
    let su = water_631g();
    let (rg, _, _) = ragged_at(&su, 0.0, 1);
    let solve = riccati_solve_bytes(&rg, Some(8));
    let cfg = AmplitudeDrpaConfig {
        eps: 0.0,
        frozen_core: 1,
        diis: Some(8),
        eps_rtol_factor: Some(0.1),
        ..Default::default()
    };
    let run = || {
        amplitude_drpa(
            &su.mol,
            &su.obs,
            &su.obs_bs,
            &su.dfbs,
            Operator::coulomb(),
            &su.rhf,
            &cfg,
        )
    };
    let _g = global_lock();

    clear_global();
    let reference = run().expect("no pool: the solve runs");

    // Ample: admitted, same energy, ledger drained.
    install_global(MemoryPool::with_capacity_bytes(64 * solve + (1 << 30)));
    let admitted = run().expect("ample pool admits");
    assert_eq!(admitted.e_corr, reference.e_corr);
    assert_eq!(
        global().unwrap().outstanding_bytes(),
        0,
        "ledger must drain"
    );

    // Tight: big enough for the upstream 3-index planes, not for the solve.
    install_global(MemoryPool::with_capacity_bytes(solve - 1));
    let err = run().expect_err("a pool below the solve's own size must refuse");
    clear_global();
    let msg = err.to_string();
    assert!(
        msg.contains("dRPA Riccati solve"),
        "refusal must name the solve's plane: {msg}"
    );
}
