//! The memory budget changes how rs-mp2-rpa blocks its work, never its
//! energies.
//!
//! On one fixed RHF reference, the scan runs at an ample budget and at
//! budgets down to the smallest one its pre-flight admits. Smaller budgets
//! narrow the frequency panel (the charge's `drpa_soft` falls below its
//! full-width value) and move the AO tensor out of core (packed or spilled);
//! the test asserts both at the smallest budget. Each only regroups sums, so
//! every energy must
//! agree with the ample run to reassociation noise. The bar, 1e-10 Ha, is
//! well above the reassociation differences measured here (water dimer /
//! aug-cc-pVDZ, at most 2e-15 Ha; printed by the test) and three orders
//! below the 2.2e-7 Ha a
//! reported budget-dependent run moved by.
//!
//! That report was not a correlation-side effect: the CLI's reference SCF
//! stores its RI-J tensor in f32 by default and keeps f64 when the f32 tensor
//! does not fit the budget (announced on stderr and in the certificate), so
//! the reference energy, and everything correlated on it, follows the budget.
//! With the reference fixed, as here, nothing does.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::rs_mp2_rpa::{
    rs_mp2_lr_rpa_scan, scan_charge_for, RsMp2RpaConfig, RsMp2RpaFormulation, RsMp2RpaResult,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const ANG_INV: f64 = ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV;
const AMPLE: usize = 1 << 40;
const BAR: f64 = 1e-10;

fn cfg_at(budget: usize) -> RsMp2RpaConfig {
    let mut cfg = RsMp2RpaConfig {
        frozen_core: 2,
        ..Default::default()
    };
    cfg.drpa.memory_budget_bytes = Some(budget);
    cfg
}

fn energies(r: &RsMp2RpaResult) -> Vec<(&'static str, f64)> {
    let mut v = vec![
        ("e_mp2_full", r.e_mp2_full),
        ("e_sr_mp2", r.e_sr_mp2),
        ("e_lr_mp2", r.e_lr_mp2),
        ("e_dmp2_lr", r.e_dmp2_lr),
        ("e_corr", r.e_corr),
        ("total_energy", r.total_energy),
    ];
    for (name, e) in [
        ("e_drpa_lr", r.e_drpa_lr),
        ("e_delta_drpa_full", r.e_delta_drpa_full),
        ("e_delta_drpa_sr", r.e_delta_drpa_sr),
    ] {
        if let Some(e) = e {
            v.push((name, e));
        }
    }
    v
}

#[test]
fn rs_mp2_rpa_energies_do_not_depend_on_the_budget() {
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
    let both = [
        RsMp2RpaFormulation::DeltaLr,
        RsMp2RpaFormulation::CoupledRings,
    ];
    let omegas = [0.42 * ANG_INV];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        let scan =
            |b: usize| rs_mp2_lr_rpa_scan(&mol, &obs, &dfbs, &rhf, &cfg_at(b), &omegas, &both);
        let charge = |b: usize| scan_charge_for(&mol, &obs, &dfbs, &cfg_at(b)).unwrap().1;
        let reference = scan(AMPLE).unwrap();
        let full = charge(AMPLE);

        // Budgets from the ample run's need down to the smallest admitted
        // one, in 5% steps; the run is checked at three of them.
        let mut admitted = Vec::new();
        let mut b = full.total;
        while charge(b).total <= b {
            admitted.push(b);
            b = b * 19 / 20;
        }
        let smallest = *admitted.last().expect("the full charge must be admitted");
        let narrowed = charge(smallest);
        eprintln!(
            "full charge {} B (dRPA scratch {} B); smallest admitted budget {smallest} B \
             (dRPA scratch {} B, AO tier {:?})",
            full.total, full.drpa_soft, narrowed.drpa_soft, narrowed.ao_tier
        );
        assert!(
            narrowed.drpa_soft < full.drpa_soft,
            "the smallest admitted budget must narrow the frequency panel"
        );
        assert_ne!(
            narrowed.ao_tier,
            ferric_integrals::three_index_source::jk_storage::Tier::InCore,
            "the smallest admitted budget must move the AO tensor out of core"
        );
        let budgets = [admitted[0], admitted[admitted.len() / 2], smallest];
        for &b in &budgets {
            let got = scan(b).unwrap_or_else(|e| panic!("refused at {b} B: {e}"));
            assert_eq!(got.len(), reference.len());
            for (k, (r, g)) in reference.iter().zip(&got).enumerate() {
                for ((name, want), (_, have)) in energies(r).into_iter().zip(energies(g)) {
                    let d = (have - want).abs();
                    eprintln!("budget {b} B result {k} {name}: |dE| = {d:.3e}");
                    assert!(d <= BAR, "{name} moved by {d:.3e} Ha at a {b} B budget");
                }
            }
        }
    });
}
