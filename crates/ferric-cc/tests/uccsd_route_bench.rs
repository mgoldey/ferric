//! Wall time and peak RSS of spin-blocked UCCSD(T) against the spin-orbital
//! route on the same UHF orbitals (CH3 / cc-pVDZ, cc-pvdz-ri, all electrons).
//!
//! Not a correctness test (those are `uccsd_anchors.rs`); `#[ignore]`d so it
//! runs only on demand, one route per process so peak RSS is per route:
//!
//! ```text
//! UCCSD_ROUTE=blocked OPENBLAS_NUM_THREADS=1 /usr/bin/time -v \
//!     target/release/deps/uccsd_route_bench-<hash> --ignored --exact uccsd_route_timing
//! ```
//!
//! `UCCSD_ROUTE` is `blocked` or `spin-orbital`. The test prints the UHF,
//! CCSD and (T) wall times and the energies (the two routes must agree).

use std::time::Instant;

use ferric_cc::ccsd::ccsd_uhf_spin_orbital;
use ferric_cc::ccsd_t::ccsd_t_uhf_spin_orbital;
use ferric_cc::u_ccsd::u_ccsd;
use ferric_cc::u_ccsd_t::u_ccsd_t;
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

const CH3: &str =
    "4\n\nC 0.0 0.0 0.0\nH 0.0 1.0790 0.0\nH 0.9344 -0.5395 0.0\nH -0.9344 -0.5395 0.0\n";

#[test]
#[ignore = "benchmark: run on demand with UCCSD_ROUTE set"]
fn uccsd_route_timing() {
    let route = std::env::var("UCCSD_ROUTE").expect("set UCCSD_ROUTE=blocked|spin-orbital");
    let mol = Molecule::parse_xyz(CH3, 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let t0 = Instant::now();
    let scf = solve_uhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        &bounds,
        &RhfConfig {
            density_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    let t_scf = t0.elapsed().as_secs_f64();
    let cfg = CcConfig {
        energy_conv: 1e-10,
        ..Default::default()
    };
    let t1 = Instant::now();
    let (e_cc, e_t, t_cc) = match route.as_str() {
        "blocked" => {
            let cc = u_ccsd(&mol, &obs, &dfbs, op, &scf, &cfg).unwrap();
            let t_cc = t1.elapsed().as_secs_f64();
            let t = u_ccsd_t(&mol, &obs, &dfbs, op, &scf, &cc, &cfg).unwrap();
            (cc.correlation_energy, t.e_t, t_cc)
        }
        "spin-orbital" => {
            let cc = ccsd_uhf_spin_orbital(&mol, &obs, &dfbs, op, &scf, &cfg).unwrap();
            let t_cc = t1.elapsed().as_secs_f64();
            let t = ccsd_t_uhf_spin_orbital(&mol, &obs, &dfbs, op, &scf, &cc, &cfg).unwrap();
            (cc.correlation_energy, t.iter().sum(), t_cc)
        }
        other => panic!("unknown UCCSD_ROUTE {other}"),
    };
    let t_total = t1.elapsed().as_secs_f64();
    println!(
        "ROUTE {route} nbasis {} uhf_s {t_scf:.2} ccsd_s {t_cc:.2} triples_s {:.2} \
         e_ccsd {e_cc:.12} e_t {e_t:.12}",
        obs.nbasis(),
        t_total - t_cc
    );
}
