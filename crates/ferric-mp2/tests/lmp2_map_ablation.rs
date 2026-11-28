//! Map-error ablation for the integral-direct amplitude-threshold LMP2 at
//! ε = 0, where the amplitude mask keeps everything and any difference from
//! canonical RI-MP2 comes from the locality maps (`DirectConfig`).
//!
//! Anchors (protocol order: these come first and must pass before a sweep):
//! - `trivial_maps_match_canonical_octane_631g`: ε = 0, every map inert,
//!   alkane_8 / 6-31G / cc-pVDZ-RI ≡ `ri_mp2` to 1e-9 Ha;
//! - `trivial_maps_match_canonical_aconf20_sto3g`: same at C20H42 (ACONF20
//!   A2000) / STO-3G.
//!
//! Probe (`probe_map_ablation`, #[ignore]): one process, many molecules,
//! many configurations. Peak RSS is reset per configuration via
//! `/proc/self/clear_refs`, so each row's `hwm_kb` is that configuration's
//! own peak. Environment:
//! - `ABL_XYZS`: comma-separated xyz paths (absolute, or relative to
//!   testdata/molecules);
//! - `ABL_BASIS` (6-31g), `ABL_AUX` (cc-pvdz-ri), `ABL_EPS` (0);
//! - `ABL_FC`: frozen core count (default: number of non-H atoms);
//! - `ABL_REF` (1): also run canonical `ri_mp2`;
//! - `ABL_CONFIGS`: `;`-separated specs, each a `,`-separated list applied
//!   left to right to the all-trivial base: `prod` (CLI defaults
//!   aux=10,virt=20,tail=1e-4,skip=1e-5,merge=4), `oldprod` (the defaults
//!   before 2026-11-17, virt=12,tail=1e-3, the 2026-11-17 ablation's
//!   `prod`), `aux=R` (R ≥ 1e5 or
//!   `off` = global), `virt=R|off`, `tail=x`, `skip=x`, `merge=k`,
//!   `kappa=x|off`, `trivial` (reset).
//!
//! Rows are printed as `ABL {json}` lines on stdout.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{AmplitudeLmp2Config, CgEnergyTarget};
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

fn heavy_atoms(mol: &Molecule) -> usize {
    mol.atoms.iter().filter(|a| a.z > 1).count()
}

/// Every map at its no-op limit (global aux fit, all virtuals, every AO
/// shell, every integral triple, per-atom batches).
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

/// Apply one ablation spec (see the module doc) to the trivial base.
fn parse_spec(spec: &str) -> DirectConfig {
    let mut d = trivial_maps();
    for tok in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let (k, v) = tok.split_once('=').unwrap_or((tok, ""));
        let num = || -> f64 { v.parse().unwrap_or_else(|_| panic!("bad value in {tok:?}")) };
        match k {
            "trivial" => d = trivial_maps(),
            "prod" => {
                d.aux_radius_bohr = 10.0;
                d.virt_radius_bohr = Some(20.0);
                d.ao_tail = 1e-4;
                d.schwarz_skip = 1e-5;
                d.batch_merge = 4;
            }
            "oldprod" => {
                d.aux_radius_bohr = 10.0;
                d.virt_radius_bohr = Some(12.0);
                d.ao_tail = 1e-3;
                d.schwarz_skip = 1e-5;
                d.batch_merge = 4;
            }
            "aux" => d.aux_radius_bohr = if v == "off" { 1e6 } else { num() },
            "virt" => d.virt_radius_bohr = if v == "off" { None } else { Some(num()) },
            "tail" => d.ao_tail = num(),
            "skip" => d.schwarz_skip = num(),
            "merge" => d.batch_merge = num() as usize,
            "kappa" => d.virt_schwarz_kappa = if v == "off" { None } else { Some(num()) },
            _ => panic!("unknown ablation key {k:?} in {spec:?}"),
        }
    }
    d
}

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

/// Reset the kernel's peak-RSS counter (VmHWM) for this process.
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

fn anchor(xyz: &str, basis_name: &str) {
    let su = setup(xyz, basis_name, "cc-pvdz-ri");
    let fc = heavy_atoms(&su.mol);
    let cfg = AmplitudeLmp2Config {
        compute_reference: true,
        eps: 0.0,
        frozen_core: fc,
        // 1e-9 against canonical RI-MP2: below what the 1e-7 Ha default target guarantees
        cg_energy_target: CgEnergyTarget::Strict,
        ..Default::default()
    };
    let (r, st) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        &trivial_maps(),
    )
    .unwrap();
    let de = r.e_corr - r.e_corr_canonical_ri;
    eprintln!(
        "ANCHOR {xyz}/{basis_name}: E={:.12} canonical={:.12} dE={de:+.3e} \
         strips rows max {} of {} cols max {} of {}",
        r.e_corr,
        r.e_corr_canonical_ri,
        st.strip_rows_max,
        su.dfbs.nbasis(),
        st.strip_cols_max,
        r.n_valence_virt + r.n_hard_virt
    );
    assert!(r.cg_converged);
    // the maps must be inert: every aux function and virtual in every strip
    assert_eq!(st.strip_rows_max, su.dfbs.nbasis());
    assert_eq!(st.strip_cols_max, r.n_valence_virt + r.n_hard_virt);
    assert!(
        de.abs() < 1e-9,
        "{xyz}: direct vs canonical ri_mp2 {de:+.3e}"
    );
}

/// ε = 0 trivial-map anchor at octane / 6-31G (the probe's smallest system).
#[ignore = "tier slow (>=60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn trivial_maps_match_canonical_octane_631g() {
    anchor("alkane_8.xyz", "6-31g");
}

/// ε = 0 trivial-map anchor at C20H42 (ACONF20 A2000) / STO-3G: the
/// ablation's molecule size, every pair and every aux function live.
#[ignore = "tier slow (>=60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn trivial_maps_match_canonical_aconf20_sto3g() {
    anchor("aconfl/A2000.xyz", "sto-3g");
}

#[test]
fn spec_parser_is_strict_and_composes() {
    let p = parse_spec("prod,aux=off");
    assert_eq!(p.aux_radius_bohr, 1e6);
    assert_eq!(p.virt_radius_bohr, Some(20.0));
    assert_eq!(p.ao_tail, 1e-4);
    let o = parse_spec("oldprod");
    assert_eq!(o.virt_radius_bohr, Some(12.0));
    assert_eq!(o.ao_tail, 1e-3);
    assert_eq!(p.schwarz_skip, 1e-5);
    assert_eq!(p.batch_merge, 4);
    let t = parse_spec("");
    assert_eq!(t.aux_radius_bohr, 1e6);
    assert_eq!(t.virt_radius_bohr, None);
    assert_eq!(t.ao_tail, 0.0);
    assert_eq!(t.schwarz_skip, 0.0);
    assert_eq!(t.batch_merge, 1);
    assert!(std::panic::catch_unwind(|| parse_spec("auxx=3")).is_err());
    assert!(std::panic::catch_unwind(|| parse_spec("aux=ten")).is_err());
}

#[test]
#[ignore = "measurement probe: release build, exclusive box lease"]
fn probe_map_ablation() {
    let basis_name = env_or("ABL_BASIS", "6-31g");
    let aux_name = env_or("ABL_AUX", "cc-pvdz-ri");
    let eps: f64 = env_or("ABL_EPS", "0").parse().unwrap();
    let with_ref = env_or("ABL_REF", "1") == "1";
    let xyzs = env_or("ABL_XYZS", "alkane_8.xyz");
    let specs = env_or("ABL_CONFIGS", "trivial;prod");
    let op = Operator::coulomb();
    for xyz in xyzs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let su = setup(xyz, &basis_name, &aux_name);
        let fc: usize = match std::env::var("ABL_FC") {
            Ok(v) => v.parse().unwrap(),
            Err(_) => heavy_atoms(&su.mol),
        };
        let nc = su.mol.atoms.iter().filter(|a| a.z == 6).count();
        let mut e_ref = f64::NAN;
        if with_ref {
            reset_hwm();
            let t0 = std::time::Instant::now();
            let r = ri_mp2(
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
            .unwrap();
            e_ref = r.mp2_corr;
            println!(
                "ABL {{\"xyz\":\"{xyz}\",\"basis\":\"{basis_name}\",\"aux\":\"{aux_name}\",\"nc\":{nc},\"fc\":{fc},\"spec\":\"ref\",\"e_corr\":{e_ref:.12},\"t\":{:.3},\"hwm_kb\":{}}}",
                t0.elapsed().as_secs_f64(),
                vm_hwm_kb()
            );
        }
        for spec in specs.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            let dcfg = parse_spec(spec);
            let cfg = AmplitudeLmp2Config {
                eps,
                frozen_core: fc,
                ..Default::default()
            };
            reset_hwm();
            let t0 = std::time::Instant::now();
            let (r, st) = amplitude_lmp2_direct(
                &su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg, &dcfg,
            )
            .unwrap();
            let t = t0.elapsed().as_secs_f64();
            println!(
                "ABL {{\"xyz\":\"{xyz}\",\"basis\":\"{basis_name}\",\"aux\":\"{aux_name}\",\"nc\":{nc},\"fc\":{fc},\"eps\":{eps:e},\"spec\":\"{spec}\",\"e_corr\":{:.12},\"err\":{:.6e},\"t\":{t:.3},\"t_asm\":{:.3},\"t_solve\":{:.3},\"hwm_kb\":{},\"cg\":{},\"strip_rows_mean\":{:.1},\"strip_cols_mean\":{:.1},\"virt_cand_mean\":{:.1},\"triples\":{},\"skipped\":{}}}",
                r.e_corr,
                r.e_corr - e_ref,
                r.timings.t_assembly_s,
                r.timings.t_solve_s,
                vm_hwm_kb(),
                r.cg_iterations,
                st.strip_rows_mean,
                st.strip_cols_mean,
                st.virt_cand_mean,
                st.n_eri3_shell_triples,
                st.n_eri3_skipped,
            );
        }
    }
}

/// Diagnostic for the `ao_tail` map: per-orbital coefficient mass the
/// per-shell max|C| < tail rule drops (Σ C² over dropped shells, no metric),
/// split occupied / virtual (valence vs hard), mean and max over orbitals.
/// A localized orbital family gives a size-independent per-orbital mass;
/// growth with the chain length means the tails are not localized.
#[test]
#[ignore = "measurement probe: release build, exclusive box lease"]
fn probe_tail_mass() {
    use ferric_mp2::lmp2_amplitude::{build_vvhv, localized_spaces};
    let basis_name = env_or("ABL_BASIS", "6-31g");
    let xyzs = env_or("ABL_XYZS", "alkane_8.xyz");
    for xyz in xyzs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let su = setup(xyz, &basis_name, "cc-pvdz-ri");
        let fc = heavy_atoms(&su.mol);
        let vv = build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
        let sp = localized_spaces(&su.mol, &su.obs, &su.rhf, fc, &vv).unwrap();
        let n_val = vv.n_valence;
        let dims = su.obs.shell_dims();
        let offs = su.obs.shell_offsets();
        let dropped = |c: ndarray::ArrayView1<f64>, tail: f64| -> f64 {
            let mut m = 0.0;
            for sh in 0..su.obs.nshells() {
                let blk = (0..dims[sh]).map(|f| c[offs[sh] + f]);
                if blk.clone().fold(0.0f64, |a, x| a.max(x.abs())) < tail {
                    m += blk.map(|x| x * x).sum::<f64>();
                }
            }
            m
        };
        for tail in [1e-3, 1e-4, 1e-5] {
            let stat = |cols: Vec<ndarray::ArrayView1<f64>>| -> (f64, f64) {
                let v: Vec<f64> = cols.into_iter().map(|c| dropped(c, tail)).collect();
                let n = v.len().max(1) as f64;
                (
                    v.iter().sum::<f64>() / n,
                    v.iter().cloned().fold(0.0, f64::max),
                )
            };
            let occ = stat(sp.c_locc.columns().into_iter().collect());
            let val = stat(sp.c_vloc.columns().into_iter().take(n_val).collect());
            let hard = stat(sp.c_vloc.columns().into_iter().skip(n_val).collect());
            println!(
                "TAILMASS {xyz} tail={tail:e} occ mean {:.3e} max {:.3e} | valvirt mean {:.3e} max {:.3e} | hardvirt mean {:.3e} max {:.3e}",
                occ.0, occ.1, val.0, val.1, hard.0, hard.1
            );
        }
    }
}
