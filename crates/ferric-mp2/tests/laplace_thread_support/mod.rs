//! Shared fixture of `laplace_energy_is_thread_independent.rs` (per commit)
//! and `validation_laplace_energy_is_thread_independent.rs` (weekly):
//! benzene/cc-pVDZ at a 0.5 GiB budget, where the Laplace panel widths bind.
#![allow(dead_code, unused_imports)]

pub use ferric_core::basis;
pub use ferric_core::mol::Molecule;
pub use ferric_integrals::basis_bridge::PreparedBasis;
pub use ferric_integrals::operator::Operator;
pub use ferric_mp2::laplace::laplace_ri_mp2;
pub use ferric_scf::result::ScfResult;
pub use ferric_scf::rhf::{solve_rhf, RhfConfig};
pub use ferric_scf::screening::SchwarzBounds;

/// Worker counts to compare. 1 vs 12 spans the interesting range on this box;
/// 3 is included because an odd split exercises a different rayon tree shape
/// than a power of two, which is what the `.reduce()` defect keyed on.
pub const THREAD_COUNTS: [usize; 4] = [1, 2, 3, 12];

/// 0.5 GiB: chosen so the panel widths BIND at this fixture (see the module
/// doc). Held FIXED across every run — the budget is allowed to affect the
/// numerics, the thread count is not.
pub const BUDGET: Option<usize> = Some(512 * 1024 * 1024);

/// The fixture's shape, for the reachability guard. Benzene/cc-pVDZ.
pub const NBAS: usize = 114;
pub const NAUX: usize = 420;
pub const NOCC: usize = 21;

pub struct Fixture {
    pub mol: Molecule,
    pub obs: PreparedBasis,
    pub dfbs: PreparedBasis,
    pub rhf: ScfResult,
}

/// Benzene/cc-pVDZ: the smallest system where the panel widths actually bind
/// (see the module doc). Water leaves them clamped at full width for every
/// worker count, so the mechanism is inert there and the test measures nothing
/// — the "a passing test may be measuring inertness" failure mode, hit for real
/// by this test's first version.
pub fn fixture() -> Fixture {
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/benzene.xyz"
    ))
    .unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ferric_core::parallel::ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    Fixture {
        mol,
        obs,
        dfbs,
        rhf,
    }
}

pub fn energy_at(f: &Fixture, n_quad: usize, threads: usize) -> f64 {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(|| {
            laplace_ri_mp2(
                &f.mol,
                &f.obs,
                &f.dfbs,
                Operator::coulomb(),
                &f.rhf,
                n_quad,
                0,
                BUDGET,
            )
            .unwrap()
            .mp2_corr
        })
}
