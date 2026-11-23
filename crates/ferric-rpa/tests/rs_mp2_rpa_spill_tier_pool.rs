//! rs-mp2-rpa under an installed memory pool at the packed and spill tiers.
//!
//! The CLI installs a process-global pool with capacity equal to the budget.
//! At a budget below the in-core need, the driver must hand the build the
//! budget it planned with: the AO tensor's in-core hard charge would
//! otherwise meet a pool that already holds the persistent planes and be
//! refused. The test asserts the run is admitted at each tier and that the
//! pool's high-water mark stays within its capacity.
//!
//! One test in its own binary: the pool is process-global.
//!
//! Mutation (commit report): the driver handing the full budget to the build
//! (`ri_cfg.memory_budget_bytes = cfg.drpa.memory_budget_bytes`) fails the
//! admission assertion at the packed budget.

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::jk_storage::Tier;
use ferric_rpa::rs_mp2_rpa::{
    rs_mp2_lr_rpa_scan, scan_charge_for, RsMp2RpaConfig, RsMp2RpaFormulation,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const ANG_INV: f64 = ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV;

fn cfg_at(budget: usize) -> RsMp2RpaConfig {
    let mut cfg = RsMp2RpaConfig {
        frozen_core: 2,
        ..Default::default()
    };
    cfg.drpa.memory_budget_bytes = Some(budget);
    cfg
}

#[test]
fn packed_and_spill_tiers_are_admitted_by_the_pool() {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/molecules/s22/water_dimer.xyz");
    let mol = Molecule::parse_xyz(&std::fs::read_to_string(p).unwrap(), 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvdz-rifit").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig::default()
            .with_max_iter(400)
            .with_energy_conv(1e-11)
            .with_density_conv(1e-10),
    )
    .unwrap();
    assert!(rhf.converged);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        let charge = |b: usize| scan_charge_for(&mol, &obs, &dfbs, &cfg_at(b)).unwrap().1;
        let full = charge(1 << 40);
        assert_eq!(full.ao_tier, Tier::InCore);
        let mut seen = Vec::new();
        let mut b = full.total;
        while b > full.persistent && seen.len() < 2 {
            let c = charge(b);
            let wanted = [Tier::InCorePacked, Tier::Spill];
            if wanted.contains(&c.ao_tier) && !seen.contains(&c.ao_tier) {
                seen.push(c.ao_tier);
                let ledger = MemoryPool::with_capacity_bytes(b);
                install_global(ledger.clone());
                let r = rs_mp2_lr_rpa_scan(
                    &mol,
                    &obs,
                    &dfbs,
                    &rhf,
                    &cfg_at(b),
                    &[0.42 * ANG_INV],
                    &[
                        RsMp2RpaFormulation::DeltaLr,
                        RsMp2RpaFormulation::CoupledRings,
                    ],
                );
                clear_global();
                eprintln!(
                    "{:?} at {b} B: pool peak {} B, charge {} B",
                    c.ao_tier,
                    ledger.peak_bytes(),
                    c.total
                );
                assert!(r.is_ok(), "{:?} at {b} B refused: {:?}", c.ao_tier, r.err());
                assert!(ledger.peak_bytes() <= b);
            }
            b = b * 49 / 50;
        }
        assert_eq!(seen.len(), 2, "both tiers must be reachable: {seen:?}");
    });
}
