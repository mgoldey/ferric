//! Default-on decision probe for the Fock-propagated screen
//! (`AmplitudeLmp2Config::fock_screen`) on the integral-direct and the
//! in-core LMP2 paths.
//!
//! Anchors (protocol order: must pass before the sweep):
//! `fock_screen_is_inert_at_eps0_butane` — at ε = 0 the screen on equals the
//! screen off on the production maps, and with every map inert both equal
//! canonical `ri_mp2` (butane / 6-31G / cc-pVDZ-RI, all-electron);
//! `fock_screen_is_inert_at_eps0_butane_incore` — the same on the in-core
//! path, which has no maps, so both arms equal canonical `ri_mp2`.
//!
//! Probe (`probe_fock_default`, #[ignore]): one process per molecule list,
//! the production maps (aux 10, virt 20, tail 1e-4, skip 1e-5, merge 4, no
//! pair gate, all-electron — the CLI defaults), ε = 0 once as the same-path
//! reference, then every (ε, κ) of the grid `FD_REPEATS` times, the κ order
//! reversed on odd repeats. VmHWM is reset before each run. Environment:
//! - `FD_XYZS` (comma-separated; relative to testdata/molecules);
//! - `FD_BASIS` (6-31g), `FD_AUX` (cc-pvdz-ri), `FD_FC` (0);
//! - `FD_EPS` (comma list), `FD_KAPPAS` (comma list, `off` = None);
//! - `FD_REPEATS` (1);
//! - `FD_PATH` (`direct`): `direct` runs `amplitude_lmp2_direct` on the
//!   production maps, `incore` runs `amplitude_lmp2` (no maps; its ε = 0 run
//!   is canonical RI-MP2, so `err` and `err_canon` coincide up to the PCG stop);
//! - `FD_EPS0` (1): run the ε = 0 same-path reference first (`err`);
//! - `FD_CANON` (1): also run canonical `ri_mp2` (`err_canon`).
//! A run the memory pre-flight refuses prints a row with `"refused"`.
//!
//! Rows are `FD {json}` lines on stdout.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{amplitude_lmp2, AmplitudeLmp2Config, CgEnergyTarget};
use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
#[allow(dead_code)]
#[path = "common/bench_rhf_cache.rs"]
mod bench_rhf_cache;

fn xyz_path(xyz: &str) -> String {
    if xyz.starts_with('/') {
        xyz.to_string()
    } else {
        format!(
            "{}/../../testdata/molecules/{xyz}",
            env!("CARGO_MANIFEST_DIR")
        )
    }
}

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    obs_bs: basis::BasisSet,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
}

fn setup(xyz: &str, basis_name: &str, aux_name: &str) -> Setup {
    let path = xyz_path(xyz);
    let mol = Molecule::load_xyz(&path).unwrap();
    let obs_bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let aux_bs = basis::bundled(aux_name).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let rhf = bench_rhf_cache::setup_rhf(
        &path,
        &mol,
        &obs,
        (basis_name, &obs_bs),
        (aux_name, &aux_bs),
    );
    Setup {
        mol,
        obs,
        obs_bs,
        dfbs,
        rhf,
    }
}

/// The CLI defaults of `[local] integral_direct = true`
/// (`ferric-cli/src/config.rs`, `LocalDirectKnobs`).
fn prod_maps() -> DirectConfig {
    DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(20.0),
        ao_tail: 1e-4,
        schwarz_skip: 1e-5,
        batch_merge: 4,
        virt_schwarz_kappa: None,
        ..Default::default()
    }
}

fn trivial_maps() -> DirectConfig {
    DirectConfig {
        aux_radius_bohr: 1e6,
        virt_radius_bohr: None,
        ao_tail: 0.0,
        schwarz_skip: 0.0,
        batch_merge: 1,
        virt_schwarz_kappa: None,
        ..Default::default()
    }
}

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn reset_hwm() {
    std::fs::write("/proc/self/clear_refs", "5").expect("clear_refs");
}

fn vm_hwm_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap();
    s.lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn run(su: &Setup, eps: f64, kappa: Option<f64>, fc: usize, d: &DirectConfig) -> f64 {
    let cfg = AmplitudeLmp2Config {
        eps,
        frozen_core: fc,
        fock_screen: kappa,
        // The eps = 0 checks compare with canonical RI-MP2 at 1e-10 Eh.
        cg_energy_target: CgEnergyTarget::Strict,
        ..Default::default()
    };
    let (r, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        d,
    )
    .unwrap();
    assert!(r.cg_converged);
    r.e_corr
}

/// ε = 0 anchor: screen on ≡ screen off on the production maps (the screen
/// is inert there), and with every map inert both ≡ canonical `ri_mp2`.
#[test]
fn fock_screen_is_inert_at_eps0_butane() {
    let su = setup("alkane_4.xyz", "6-31g", "cc-pvdz-ri");
    let e_off = run(&su, 0.0, None, 0, &prod_maps());
    let e_on = run(&su, 0.0, Some(1.0), 0, &prod_maps());
    let t_off = run(&su, 0.0, None, 0, &trivial_maps());
    let t_on = run(&su, 0.0, Some(1.0), 0, &trivial_maps());
    let canon = ri_mp2(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &RiMp2Config::default(),
    )
    .unwrap()
    .mp2_corr;
    eprintln!(
        "ANCHOR butane: prod on-off {:+.3e}; trivial on-off {:+.3e}; trivial off-canon {:+.3e}; trivial on-canon {:+.3e}",
        e_on - e_off,
        t_on - t_off,
        t_off - canon,
        t_on - canon
    );
    assert!((e_on - e_off).abs() < 1e-12);
    assert!((t_on - t_off).abs() < 1e-12);
    assert!((t_off - canon).abs() < 1e-10);
    assert!((t_on - canon).abs() < 1e-10);
}

fn run_incore(su: &Setup, eps: f64, kappa: Option<f64>, fc: usize) -> f64 {
    let cfg = AmplitudeLmp2Config {
        eps,
        frozen_core: fc,
        fock_screen: kappa,
        // The eps = 0 checks compare with canonical RI-MP2 at 1e-10 Eh.
        cg_energy_target: CgEnergyTarget::Strict,
        ..Default::default()
    };
    let r = amplitude_lmp2(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
    )
    .unwrap();
    assert!(r.cg_converged);
    r.e_corr
}

/// ε = 0 anchor on the in-core path: screen on ≡ screen off (the screen is
/// inert at ε = 0), and both ≡ canonical `ri_mp2` (the in-core path has no
/// locality maps).
#[test]
fn fock_screen_is_inert_at_eps0_butane_incore() {
    let su = setup("alkane_4.xyz", "6-31g", "cc-pvdz-ri");
    let e_off = run_incore(&su, 0.0, None, 0);
    let e_on = run_incore(&su, 0.0, Some(1.0), 0);
    let canon = ri_mp2(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &RiMp2Config::default(),
    )
    .unwrap()
    .mp2_corr;
    eprintln!(
        "ANCHOR butane in-core: on-off {:+.3e}; off-canon {:+.3e}; on-canon {:+.3e}",
        e_on - e_off,
        e_off - canon,
        e_on - canon
    );
    assert!((e_on - e_off).abs() < 1e-12);
    assert!((e_off - canon).abs() < 1e-10);
    assert!((e_on - canon).abs() < 1e-10);
}

fn parse_kappa(s: &str) -> Option<f64> {
    match s {
        "off" => None,
        k => Some(k.parse().unwrap_or_else(|_| panic!("bad kappa {k:?}"))),
    }
}

#[test]
#[ignore = "measurement probe: release build, exclusive box lease"]
fn probe_fock_default() {
    let basis_name = env_or("FD_BASIS", "6-31g");
    let aux_name = env_or("FD_AUX", "cc-pvdz-ri");
    let fc: usize = env_or("FD_FC", "0").parse().unwrap();
    let eps_list: Vec<f64> = env_or("FD_EPS", "1e-4,1e-5,1e-6,1e-7")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let kappas: Vec<Option<f64>> = env_or("FD_KAPPAS", "off,1")
        .split(',')
        .map(|s| parse_kappa(s.trim()))
        .collect();
    let repeats: usize = env_or("FD_REPEATS", "1").parse().unwrap();
    let xyzs = env_or("FD_XYZS", "alkane_8.xyz");
    let op = Operator::coulomb();
    let d = prod_maps();
    let path = env_or("FD_PATH", "direct");
    let incore = match path.as_str() {
        "incore" => true,
        "direct" => false,
        other => panic!("FD_PATH must be direct or incore, got {other:?}"),
    };
    let kstr = |k: Option<f64>| k.map_or("\"off\"".to_string(), |x| format!("{x:e}"));
    for xyz in xyzs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let su = setup(xyz, &basis_name, &aux_name);
        let nc = su.mol.atoms.iter().filter(|a| a.z == 6).count();
        let mut e0 = f64::NAN;
        let mut e_canon = f64::NAN;
        if env_or("FD_CANON", "1") == "1" {
            reset_hwm();
            let t0 = std::time::Instant::now();
            e_canon = ri_mp2(
                &su.mol,
                &su.obs,
                &su.dfbs,
                op,
                &su.rhf,
                &RiMp2Config {
                    frozen_core: fc,
                    ..Default::default()
                },
            )
            .unwrap()
            .mp2_corr;
            println!(
                "FD {{\"xyz\":\"{xyz}\",\"nc\":{nc},\"fc\":{fc},\"spec\":\"canon\",\"e_corr\":{e_canon:.12},\"t\":{:.3},\"hwm_kb\":{}}}",
                t0.elapsed().as_secs_f64(),
                vm_hwm_kb()
            );
        }
        let mut runs: Vec<(f64, Option<f64>, usize)> = Vec::new();
        if env_or("FD_EPS0", "1") == "1" {
            runs.push((0.0, None, 0));
        }
        for rep in 0..repeats {
            for &eps in &eps_list {
                let ks: Vec<Option<f64>> = if rep % 2 == 0 {
                    kappas.clone()
                } else {
                    kappas.iter().rev().copied().collect()
                };
                for k in ks {
                    runs.push((eps, k, rep));
                }
            }
        }
        for (eps, kappa, rep) in runs {
            let cfg = AmplitudeLmp2Config {
                eps,
                frozen_core: fc,
                fock_screen: kappa,
                ..Default::default()
            };
            reset_hwm();
            let t0 = std::time::Instant::now();
            let res = if incore {
                amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg)
                    .map(|r| (r, None))
            } else {
                amplitude_lmp2_direct(
                    &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg, &d,
                )
                .map(|(r, st)| (r, Some(st)))
            };
            let (r, st) = match res {
                Ok(x) => x,
                Err(e) => {
                    let msg = format!("{e}").replace('"', "'");
                    println!(
                        "FD {{\"xyz\":\"{xyz}\",\"nc\":{nc},\"fc\":{fc},\"eps\":{eps:e},\"kappa\":{},\"rep\":{rep},\"refused\":\"{msg}\"}}",
                        kstr(kappa)
                    );
                    continue;
                }
            };
            let t = t0.elapsed().as_secs_f64();
            if eps == 0.0 {
                e0 = r.e_corr;
            }
            println!(
                "FD {{\"path\":\"{path}\",\"xyz\":\"{xyz}\",\"basis\":\"{basis_name}\",\"aux\":\"{aux_name}\",\"nc\":{nc},\"fc\":{fc},\"eps\":{eps:e},\"kappa\":{},\"rep\":{rep},\"e_corr\":{:.12},\"err\":{:.6e},\"err_canon\":{:.6e},\"keep\":{:.6e},\"t\":{t:.3},\"t_asm\":{:.3},\"t_solve\":{:.3},\"hwm_kb\":{},\"cg\":{},\"cg_conv\":{},\"group_scratch_bound\":{},\"group_conc\":{},\"threads\":{}}}",
                kstr(kappa),
                r.e_corr,
                r.e_corr - e0,
                r.e_corr - e_canon,
                r.keep_fraction,
                r.timings.t_assembly_s,
                r.timings.t_solve_s,
                vm_hwm_kb(),
                r.cg_iterations,
                r.cg_converged,
                st.as_ref().map_or(0, |s| s.group_scratch_bound_bytes),
                st.as_ref().map_or(0, |s| s.group_concurrency),
                rayon::current_num_threads(),
            );
        }
    }
}
