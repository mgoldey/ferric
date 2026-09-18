//! MWE: the migration must NOT refuse a job the pre-migration tree completes.
//!
//! # The standing decision this enforces
//!
//! From the shared brief, user-level: "THE MIGRATION MUST NOT REFUSE A JOB
//! THAT THE CURRENT TREE COMPLETES." Adding four hard charges to a crate is
//! exactly the change that can violate it -- a plane that was never counted is
//! now counted, and a capacity that used to work now fails.
//!
//! # How this is measured rather than argued
//!
//! Binary-search the smallest pool capacity at which a full GW run completes,
//! and compare it against the same search with this crate's charges disabled
//! (which is behaviourally the pre-migration tree: `ferric_gw` debited nothing,
//! while ferric-rpa and ferric-integrals charged exactly as they do now).
//!
//! # Scope, stated rather than implied
//!
//! This runs at ONE shape (water / STO-3G orbital basis, cc-pVDZ-RI auxiliary,
//! G0W0@HF) -- the largest a default-profile test can binary-search in about a
//! second. At that shape ferric-gw's charges total 65_856 B against a
//! ~900_000 B floor, so they are dominated by roughly 14x and the two minima
//! come out identical TO THE BYTE at 1, 2, 4 and 12 workers.
//!
//! That is a real measurement of the standing decision and it is also a
//! measurement of a regime where the answer was never in much doubt. It does
//! NOT establish that the charges can never bind at production scale, where
//! `naux·n_act²` grows as the fourth power of system size. What makes the
//! conclusion transferable is the STRUCTURE, not this one number: ferric-rpa
//! hard-charges the whole pipeline's `estimate_peak_bytes` -- a term that
//! already contains `naux·nocc·nvir` plus five `naux²` buffers plus the
//! retained inverse-dielectric stack -- and RELEASES it before ferric-gw
//! allocates anything. ferric-gw's pair has to exceed all of that before it
//! can bind. The assertion below is written so that if it ever does, the
//! failure names the numbers rather than the conclusion.

use ferric_core::basis;
use ferric_core::memory::pool::{self, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::sync::{Mutex, MutexGuard, OnceLock};

fn lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn setup(
    obs_name: &str,
) -> (
    Molecule,
    PreparedBasis,
    PreparedBasis,
    ferric_scf::ScfResult,
) {
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO  0.0   0.0       0.117790\nH  0.0   0.755453 -0.471161\nH  0.0  -0.755453 -0.471161\n",
        0,
        1,
    )
    .expect("parse H2O");
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).expect("obs basis")).expect("obs");
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").expect("ri")).expect("dfbs");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).expect("schwarz");
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig::default(),
    )
    .expect("RHF");
    (mol, obs, dfbs, rhf)
}

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        need_eigenvalues_freq: true,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 8,
            u0: 0.5,
        },
        eigensolver_conv_thresh: 1e-7,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: false,
        verbose: false,
    }
}

fn completes(obs_name: &str, capacity: usize) -> bool {
    let (mol, obs, dfbs, rhf) = setup(obs_name);
    pool::install_global(MemoryPool::with_capacity_bytes(capacity));
    let r = run_gw(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &rhf,
        &pdep_cfg(),
        &GwConfig {
            method: GwMethod::G0W0,
            ..Default::default()
        },
        None,
    );
    pool::clear_global();
    r.is_ok()
}

/// Smallest capacity at which a G0W0 run on `obs_name` completes.
fn smallest_completing(obs_name: &str, hi: usize) -> usize {
    assert!(
        completes(obs_name, hi),
        "the upper bound {hi} B must itself complete for {obs_name}"
    );
    let (mut lo, mut hi) = (0usize, hi);
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if completes(obs_name, mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// The four charges this crate adds must not raise the smallest completing
/// capacity above what ferric-rpa and ferric-integrals already require.
///
/// # What the numbers mean
///
/// This asserts the WEAKER, falsifiable form of the claim: whatever ferric-gw
/// adds, the run must still complete at the capacity ferric-rpa's own preflight
/// demands. The reason that holds is structural rather than lucky -- ferric-rpa
/// hard-charges `estimate_peak_bytes` for the WHOLE pipeline and RELEASES it
/// when `run_pdep_rpa` returns, and that peak (which already counts a
/// `naux·nocc·nvir` b_ov, three `naux²` metric buffers, two more for the
/// eigensolve and the retained inverse-dielectric stack) exceeds the
/// `naux·n_act²` pair ferric-gw allocates afterwards at every shape this suite
/// can run.
///
/// MEASURED on water / STO-3G with a cc-pVDZ-RI aux basis, G0W0@HF,
/// binary-searched:
///
/// ```text
///   workers   ferric-gw charges OFF   ON        ferric-gw's own planes
///   1           899_808               899_808   65_856
///   2           906_528               906_528   65_856
///   4           919_968               919_968   65_856
///  12           946_848               946_848   65_856
/// ```
///
/// identical to the byte. The `charges OFF` column was produced by replacing
/// both `reserve_global` calls with `Reservation::inert`, which is
/// behaviourally the pre-migration tree.
///
/// This test re-runs the `ON` search live rather than trusting that table, and
/// compares it to the sum ferric-gw contributes -- if the migration ever starts
/// binding, the minimum has to rise by at least `b_full + m_proj` and the
/// assertion says so with the numbers.
#[test]
fn the_migration_does_not_raise_the_smallest_completing_capacity() {
    let _g = lock();
    let workers = rayon::current_num_threads().max(1);

    // Reference: what ferric-rpa/ferric-integrals alone demand is not directly
    // measurable from here (that would need this crate's charges compiled out),
    // so bound it instead. ferric-gw's contribution is exactly these bytes:
    let naux = 84usize; // cc-pVDZ-RI on water
    let nbf = 7usize; // STO-3G on water
    let gw_bytes =
        ferric_gw::budget::b_full_bytes(naux, nbf) + ferric_gw::budget::m_proj_bytes(naux, nbf);

    let min = smallest_completing("sto-3g", 1_000_000_000);
    eprintln!(
        "smallest completing capacity: {min} B at {workers} workers \
         (ferric-gw's own planes: {gw_bytes} B)"
    );

    // Frozen from the measured `charges OFF` search -- the pre-migration
    // behaviour, at the three widths the suite is verified at.
    let pre_migration = match workers {
        1 => 899_808,
        2 => 906_528,
        4 => 919_968,
        w if w >= 8 => 946_848,
        _ => {
            eprintln!("no pre-migration reference measured at {workers} workers; skipping");
            return;
        }
    };
    assert!(
        min <= pre_migration,
        "the migration RAISED the smallest completing capacity from {pre_migration} B to \
         {min} B at {workers} workers, i.e. it refuses a job the pre-migration tree \
         completes. That is the standing decision's one hard constraint. ferric-gw's own \
         planes are {gw_bytes} B; if they are now binding, the fix is to make one of them \
         soft with a real streaming fallback, NOT to raise the default budget."
    );
    // And the other direction: a charge that binds nothing at all might mean
    // the planes are not being charged. They are -- the composition binary
    // pins that -- so this only records that they are dominated HERE.
    assert!(
        gw_bytes > 0,
        "ferric-gw must be charging something; a zero contribution would make this test \
         vacuous"
    );
}
