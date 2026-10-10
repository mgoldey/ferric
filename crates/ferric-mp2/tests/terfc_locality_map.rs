//! RESEARCH HARNESS (ignored): locality of amplitude-threshold LMP2 versus the
//! terfc operator (r0, r0*omega), at a FAIR criterion. See
//! benchmarks/terfc_locality_map/ (prereg + results). Not a gate test.
//!
//! One SCF per process; for each operator in OPS it prints
//!   ROWOP  <mol> <op> E_op(eps=0, ri_mp2) E_coul(ri_mp2) ratio t_ri t_eri3 t_eri3_coul
//!   ROW    <mol> <op> eps e_corr relerr keep% pairs dom_mean dom_max wall_s
//! Exactness anchor (ANCHOR=1): direct eps=0 with trivial maps must equal the
//! same-operator ri_mp2 to <= 1e-9 Ha, asserted per operator BEFORE its sweep.
//!
//! Env: MOL (xyz file name under testdata/molecules), FC (frozen core count),
//! OPS ("coul;terfc:r0_ang:r0w;erfc:omega_inv_ang;..."), EPS (comma list),
//! MAPS ("trivial" | "prod"), ANCHOR ("1"), FERRIC_TERF_TABLE_DIR.
//!   FERRIC_TERF_TABLE_DIR=$PWD/terf-tables OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=3 \
//!   MOL=alkane_8.xyz FC=8 OPS="coul;terfc:1.0:0.7071" EPS=1e-3,1e-4 ANCHOR=1 \
//!   cargo test -p ferric-mp2 --release --test terfc_locality_map -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::AmplitudeLmp2Config;
use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

const ANG: f64 = ferric_core::units::ANGSTROM_TO_BOHR;

fn parse_op(s: &str) -> Operator {
    let p: Vec<&str> = s.split(':').collect();
    match p[0] {
        "coul" => Operator::coulomb(),
        "terfc" => {
            let r0 = p[1].parse::<f64>().unwrap() * ANG;
            let r0w: f64 = p[2].parse().unwrap();
            Operator::terfc_with_omega(r0, r0w / r0)
        }
        // omega in Angstrom^-1 on the CLI, Bohr^-1 in the operator.
        "erfc" => Operator::erfc(p[1].parse::<f64>().unwrap() / ANG),
        o => panic!("unknown op {o}"),
    }
}

fn maps(kind: &str) -> DirectConfig {
    match kind {
        "trivial" => DirectConfig {
            aux_radius_bohr: 1e6,
            virt_radius_bohr: None,
            ao_tail: 0.0,
            schwarz_skip: 0.0,
            ..Default::default()
        },
        // CLI production defaults (config.rs) except schwarz_skip, which the
        // terfc table engine requires to be 0.
        "prod" => DirectConfig {
            aux_radius_bohr: 10.0,
            virt_radius_bohr: Some(12.0),
            ao_tail: 1e-3,
            schwarz_skip: 0.0,
            batch_merge: 4,
            ..Default::default()
        },
        k => panic!("MAPS={k}"),
    }
}

#[test]
#[ignore]
fn terfc_locality_map() {
    if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
        panic!("set FERRIC_TERF_TABLE_DIR");
    }
    let molf = std::env::var("MOL").unwrap_or("alkane_8.xyz".into());
    let fc: usize = std::env::var("FC").unwrap().parse().unwrap();
    let ops = std::env::var("OPS").unwrap();
    let eps_list: Vec<f64> = std::env::var("EPS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let mapkind = std::env::var("MAPS").unwrap_or("trivial".into());
    let anchor = std::env::var("ANCHOR").map(|v| v == "1").unwrap_or(false);
    let skip_ri = std::env::var("SKIP_RI").map(|v| v == "1").unwrap_or(false);

    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{molf}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let c = Operator::coulomb();
    let bounds = SchwarzBounds::compute(c, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        c,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    let no = (mol.nelec() as usize) / 2 - fc;
    eprintln!("SETUP {molf} nbas={} naux={} no={no} fc={fc} maps={mapkind}", obs.nbasis(), dfbs.nbasis());

    let ri = |op: Operator| -> (f64, f64) {
        let t = Instant::now();
        let e = ri_mp2(
            &mol,
            &obs,
            &dfbs,
            op,
            &rhf,
            &RiMp2Config {
                frozen_core: fc,
                ..Default::default()
            },
        )
        .unwrap()
        .mp2_corr;
        (e, t.elapsed().as_secs_f64())
    };
    let t3 = |op: Operator| -> f64 {
        let t = Instant::now();
        let x = ferric_integrals::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        std::hint::black_box(&x);
        t.elapsed().as_secs_f64()
    };
    let (e_coul, _) = ri(c);
    let t3_coul = t3(c);

    for spec in ops.split(';') {
        let op = parse_op(spec);
        let (e0, t_ri, t_e3) = if spec == "coul" {
            (e_coul, 0.0, t3_coul)
        } else {
            let t_e3 = t3(op);
            if skip_ri {
                (f64::NAN, 0.0, t_e3)
            } else {
                let (e, t) = ri(op);
                (e, t, t_e3)
            }
        };
        println!(
            "ROWOP {molf} {spec} e0 {e0:.10} ecoul {e_coul:.10} ratio {:.5} t_ri {t_ri:.1} t_eri3 {t_e3:.2} t_eri3_coul {t3_coul:.2} cost_ratio {:.2}",
            e0 / e_coul,
            t_e3 / t3_coul
        );
        let run = |eps: f64, mp: &DirectConfig| {
            let cfg = AmplitudeLmp2Config {
                eps,
                frozen_core: fc,
                compute_reference: false,
                ..Default::default()
            };
            let t = Instant::now();
            let (r, _) =
                amplitude_lmp2_direct(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg, mp).unwrap();
            assert!(r.cg_converged, "CG not converged {spec} eps {eps}");
            (r, t.elapsed().as_secs_f64())
        };
        if anchor {
            let (r, w) = run(0.0, &maps("trivial"));
            let d = (r.e_corr - e0).abs();
            println!("ANCHOR {molf} {spec} direct_eps0 {:.10} ri {e0:.10} |d| {d:.3e} wall {w:.1}", r.e_corr);
            assert!(d < 1e-9, "{spec}: eps=0 direct vs ri_mp2 {d:.3e}");
        }
        let mp = maps(&mapkind);
        for &eps in &eps_list {
            let (r, w) = run(eps, &mp);
            println!(
                "ROW {molf} {spec} eps {eps:e} e_corr {:.10} relerr {:.4e} keep {:.3} pairs {:.1} dom_mean {:.2} dom_max {} wall {w:.1}",
                r.e_corr,
                ((r.e_corr - e0) / e0).abs(),
                r.keep_fraction * 100.0,
                r.pair_fraction * (no * no) as f64,
                r.dom_mean,
                r.dom_max
            );
        }
    }
}
