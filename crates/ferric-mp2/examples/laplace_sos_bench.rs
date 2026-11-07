//! Laplace SOS-MP2 formulation harness: energy, wall time and structural
//! counters of each formulation on one RHF reference.
//!
//! usage: laplace_sos_bench <xyz> <basis> <aux> <frozen_core> <budget_gb> <spec>...
//!   spec = mo | ao | ao-sparse:R | local:R | local:VIR:TAU:OCC   (radii in Bohr, TAU Schwarz)
//!
//! `local:R` is the production radius rule (`LocalRadii::from_cutoff`);
//! `local:VIR:TAU:OCC` sets the three truncation thresholds separately.
//! One line per spec on stdout:
//!   SPEC <spec> e_os=<Ha> wall=<s>
//! `local` also prints its counter line. Run one spec per process when peak
//! RSS matters (the process peak covers everything run before it).

use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::laplace::{laplace_sos_mp2_with_basis, LaplaceMp2, SosFormulation, SosMp2Config};
use ferric_mp2::laplace_local::LocalRadii;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 7 {
        eprintln!(
            "usage: laplace_sos_bench <xyz> <basis> <aux> <frozen_core> <budget_gb> <spec>..."
        );
        std::process::exit(2);
    }
    let mol = Molecule::load_xyz(&a[1]).expect("xyz");
    let obs_bs = basis::bundled(&a[2]).expect("basis");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("obs");
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(&a[3]).expect("aux")).expect("aux");
    let frozen_core: usize = a[4].parse().expect("frozen_core");
    let budget_gb: f64 = a[5].parse().expect("budget_gb");
    let op = Operator::coulomb();
    println!(
        "SYS natom={} nbas={} naux={} nocc={} frozen_core={frozen_core} threads={}",
        mol.atoms.len(),
        obs.nbasis(),
        dfbs.nbasis(),
        mol.nelec() / 2,
        rayon::current_num_threads()
    );

    let t = Instant::now();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
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
    .expect("rhf");
    println!(
        "SCF e={:.10} wall={:.3}",
        rhf.energy,
        t.elapsed().as_secs_f64()
    );

    let budget = Some((budget_gb * 1e9) as usize);
    let cfg = SosMp2Config {
        c_os: 1.0,
        frozen_core,
        n_quad: 7,
        memory_budget_bytes: budget,
        domain_cutoff_bohr: None,
    };
    for spec in &a[6..] {
        let parts: Vec<&str> = spec.split(':').collect();
        let t = Instant::now();
        let e_os = if parts.len() == 4 && parts[0] == "local" {
            let f = |k: usize| -> f64 { parts[k].parse().expect("radius") };
            let radii = LocalRadii {
                vir: f(1),
                pair_tau: f(2),
                occ: f(3),
            };
            let mut lp = LaplaceMp2::new(7);
            lp.memory_budget_bytes = budget;
            let (e, st) = lp
                .compute_sos_local(&mol, &obs, &obs_bs, &dfbs, op, &rhf, frozen_core, radii)
                .expect("local");
            println!("{st}");
            e
        } else {
            let form = match parts.as_slice() {
                [name] => SosFormulation::parse_config_str(Some(name), None),
                [name, r] => {
                    SosFormulation::parse_config_str(Some(name), Some(r.parse().expect("radius")))
                }
                _ => panic!("bad spec {spec}"),
            }
            .expect("spec");
            laplace_sos_mp2_with_basis(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg, form)
                .expect("sos")
                .e_os
        };
        println!(
            "SPEC {spec} e_os={e_os:.12} wall={:.3}",
            t.elapsed().as_secs_f64()
        );
    }
}
