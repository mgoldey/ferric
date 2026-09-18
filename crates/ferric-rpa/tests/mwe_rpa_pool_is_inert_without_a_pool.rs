//! EXACTNESS ANCHOR for wiring the shared memory pool into the ferric-rpa
//! paths.
//!
//! Written and made to pass BEFORE any ferric-rpa gate was migrated, per
//! `CLAUDE.md`'s Experimental Protocol ("EXACTNESS ANCHOR FIRST"), and
//! mirroring `ferric-scf/tests/mwe_ksdft_pool_is_inert_without_a_pool.rs`.
//!
//! The trivial limit of the pool is "no pool installed". In that limit an RPA
//! run must be BIT-IDENTICAL to the pre-pool tree: every migrated gate must
//! take the same branch, hold an inert reservation, and produce the same
//! correlation energy to the last bit.
//!
//! ## Artifact hypothesis (stated before measuring, per the protocol)
//!
//! If the no-op property is REAL: with no pool installed, `reserve_global`
//! hands back an inert reservation, `MemoryPlan::commit` degenerates to
//! `check()` plus an inert guard, and every panel/band width in this crate is
//! resolved from exactly the ceiling it was resolved from before — so two runs
//! of the same input differ by exactly 0 bits.
//!
//! If the implementation is BROKEN in the most likely way — a migrated gate
//! that treats "no pool" as "no memory available" (`try_reserve` returning
//! `None`, or `global_available_bytes()` unwrapped to 0) — then
//! `lanczos_panel_width` and `quad_panel_width` would collapse to their
//! minimum widths. Those widths fix the accumulation order of the paneled
//! `naux x naux` dielectric assembly and of the frequency-quadrature fold, so
//! the correlation energy would move at ~1e-12..1e-9 Ha. That prediction
//! (nonzero bit delta) differs from the real one (0 bits), so this test can
//! distinguish them.
//!
//! The second most likely break is over-charging: a plane charged against a
//! pool that is already holding it, which self-refuses. `an_ample_pool_*`
//! catches that as an `Err` rather than a bit move.
//!
//! The pool is process-global, so these tests serialize on `GLOBAL_LOCK` and
//! clear the slot on entry and exit.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn global_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    match L.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

/// Clear the process-global pool on entry AND exit so one test's pool can
/// never leak into another's "unbudgeted" assertions.
struct CleanSlot(#[allow(dead_code)] MutexGuard<'static, ()>);

impl CleanSlot {
    fn acquire() -> Self {
        let g = global_lock();
        clear_global();
        Self(g)
    }
}

impl Drop for CleanSlot {
    fn drop(&mut self) {
        clear_global();
    }
}

/// A PDEP-RPA correlation energy on water/cc-pVDZ with the cc-pvdz-ri aux
/// basis. Big enough that the preflight, the Lanczos panel and the frequency
/// quadrature all run for real; small enough to be a test.
fn run_rpa() -> f64 {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
    let obs_bs = basis::bundled("cc-pvdz").expect("cc-pvdz");
    let dfbs_bs = basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("obs");
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).expect("dfbs");
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("rhf");
    let cfg = PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 8,
            u0: 0.5,
        },
        frozen_core: 0,
        trunc_thresh: 0.0,
        eigensolver_conv_thresh: 1e-10,
        ..Default::default()
    };
    run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg)
        .expect("PDEP-RPA must run")
        .e_rpa
}

/// A static Becke-partitioned per-atom polarizability: the grid property path,
/// which is the OTHER family of migrated gates (`preflight_grid_path`). Its
/// `chi` plane is the one that reached the OOM killer in 2026-07.
fn run_becke_alpha() -> [[f64; 3]; 3] {
    use ferric_rpa::properties::pdep_polarizability_becke;
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz("../../testdata/molecules/h2.xyz").expect("h2.xyz");
    let obs_bs = basis::bundled("sto-3g").expect("sto-3g");
    let dfbs_bs = basis::bundled("sto-3g").expect("sto-3g");
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("obs");
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).expect("dfbs");
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("rhf");
    let cfg = PdepRpaConfig::default();
    let per_atom =
        pdep_polarizability_becke(&mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg).expect("becke alpha");
    per_atom[0]
}

#[test]
fn unbudgeted_rpa_energy_is_bit_identical_across_repeated_runs() {
    let _s = CleanSlot::acquire();
    // No pool installed: every migrated gate must be inert.
    let a = run_rpa();
    let b = run_rpa();
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "unbudgeted PDEP-RPA must be deterministic to the BIT (got {a:.17e} vs {b:.17e})"
    );
}

#[test]
fn installing_an_ample_pool_does_not_move_the_rpa_energy_one_bit() {
    let _s = CleanSlot::acquire();
    let unbudgeted = run_rpa();

    // An ample pool: every reservation fits, so every gate must take the SAME
    // branch it took unbudgeted. This is the property that makes the pool safe
    // to install by default -- charging a plane must not change which code
    // path runs when the plane fits.
    install_global(MemoryPool::with_capacity_bytes(64_000_000_000));
    let budgeted = run_rpa();
    clear_global();

    assert_eq!(
        unbudgeted.to_bits(),
        budgeted.to_bits(),
        "an AMPLE pool must not perturb the RPA energy by even one bit \
         (unbudgeted {unbudgeted:.17e} vs pooled {budgeted:.17e}) -- if this \
         fails, charging a plane changed a panel width, which moves an energy"
    );
}

#[test]
fn installing_an_ample_pool_does_not_move_the_becke_polarizability_one_bit() {
    let _s = CleanSlot::acquire();
    let unbudgeted = run_becke_alpha();

    install_global(MemoryPool::with_capacity_bytes(64_000_000_000));
    let budgeted = run_becke_alpha();
    clear_global();

    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(
                unbudgeted[i][j].to_bits(),
                budgeted[i][j].to_bits(),
                "an AMPLE pool must not perturb alpha[{i}][{j}] by one bit \
                 ({:.17e} vs {:.17e})",
                unbudgeted[i][j],
                budgeted[i][j]
            );
        }
    }
}
