//! The J/K digest of the RI-MP2-family geometry gradient
//! (`ScfJk::jk_digest`, CLI `[scf] jk_digest`) reaches the direct builds of
//! the reference it re-solves at each geometry: the closed-shell driver
//! (`total_rimp2_gradient_on`, which the SCS-MP2, attenuated and 2-terfc
//! closed-shell gradients share through `solve_rhf_reference`) and the
//! open-shell one (`total_u_rimp2_gradient_on`, `solve_reference`, shared by
//! the unrestricted variants).
//!
//! The four-centre reference of a small molecule normally runs on the
//! in-core ERI store (`[scf] eri = "auto"`), which no digest reaches, so each
//! run is a child process of this binary with a memory budget
//! (`FERRIC_MEM_BUDGET_GB`) whose quarter the store exceeds: the reference
//! SCF and its response then build J/K integral-direct. The correlation part
//! gets an explicit budget (`RiMp2Config::memory_budget_bytes`). No test sets
//! a variable in its own process.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::RiMp2Config;
use ferric_scf::direct_jk::gc_kernel::kernel_probe;
use ferric_scf::direct_jk::GcKernel;
use ferric_scf::quartet_scatter::{pair_probe, JkDigest};
use ferric_scf::scf_jk::ScfJk;
use ferric_scf::warm_start::ScfWarmStart;

/// Set in the environment of the child runs: the config value to run
/// (`pair`, `legacy` or `unset`).
const CHILD: &str = "FERRIC_TEST_RIMP2_JK_DIGEST_CHILD";

/// The children's memory budget, GiB (about 107 kB): a quarter of it is
/// below the in-core store of water and OH in cc-pVDZ (361 kB and 145 kB),
/// so `eri = "auto"` builds J/K direct.
const BUDGET_GB: &str = "0.0001";

/// The two drivers.
const CASES: [&str; 2] = ["closed", "open"];

/// One child's run of `case`: the energy, then the gradient, and the
/// pair-digest builds the probe recorded.
#[derive(Debug, Clone, PartialEq)]
struct Run {
    case: String,
    builds: usize,
    /// Quartets the shared-primitive kernel served (`gc_kernel`).
    kernel: usize,
    values: Vec<f64>,
}

/// The energy and gradient of `case` under the config value `digest`, with
/// the pair-digest builds of the whole run.
fn run(case: &str, digest: Option<JkDigest>, gck: Option<GcKernel>) -> Run {
    let (xyz, mult) = match case {
        "closed" => (
            "3\nwater\nO 0.02 -0.01 0.117790\nH 0.0 0.785453 -0.451161\nH 0.03 -0.735453 -0.491161\n",
            1,
        ),
        _ => ("2\nOH\nO 0.0 0.0 0.0\nH 0.03 0.0 0.98\n", 2),
    };
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let obs = basis::bundled("cc-pvdz").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let cfg = RiMp2Config {
        frozen_core: 1,
        memory_budget_bytes: Some(2 << 30),
        ..Default::default()
    };
    let jk = ScfJk::exact().with_jk_digest(digest).with_gc_kernel(gck);
    let mut warm = ScfWarmStart::new(false);
    let op = Operator::coulomb();
    pair_probe::start();
    kernel_probe::start();
    let (e, g) = if case == "closed" {
        ferric_mp2::gradient::total_rimp2_gradient_on(
            &mol, &obs, &aux, op, &cfg, None, &jk, &mut warm,
        )
    } else {
        ferric_mp2::u_rimp2_gradient::total_u_rimp2_gradient_on(
            &mol, &obs, &aux, op, &cfg, None, &jk, &mut warm,
        )
    }
    .unwrap();
    let builds = pair_probe::take().len();
    let kernel = kernel_probe::take().iter().map(|c| c.quartets).sum();
    let mut values = vec![e];
    values.extend(g.iter().copied());
    Run {
        case: case.into(),
        builds,
        kernel,
        values,
    }
}

/// Child entry point of [`config_digest_reaches_the_rimp2_gradient_builds`]:
/// every case under the config value `$CHILD`, printed as hex bits. A no-op
/// outside the child.
#[test]
fn child_prints_rimp2_gradient_bits() {
    let Some(value) = std::env::var_os(CHILD) else {
        return;
    };
    let (digest, gck) = value.to_str().unwrap().split_once('/').unwrap();
    let digest = match digest {
        "unset" => None,
        v => Some(JkDigest::parse_str(v).unwrap()),
    };
    let gck = match gck {
        "unset" => None,
        v => Some(GcKernel::parse_str(v).unwrap()),
    };
    for case in CASES {
        let r = run(case, digest, gck);
        let bits: Vec<String> = r
            .values
            .iter()
            .map(|v| format!("{:016x}", v.to_bits()))
            .collect();
        println!(
            "RIMP2_DIGEST {} {} {} {}",
            r.case,
            r.builds,
            r.kernel,
            bits.join(" ")
        );
    }
}

/// Run every case in a child with the config value `config` and, when
/// given, `FERRIC_JK_DIGEST = env`.
fn child(config: &str, env: Option<&str>) -> Vec<Run> {
    let envs: Vec<(&str, &str)> = env.map(|v| ("FERRIC_JK_DIGEST", v)).into_iter().collect();
    child_with(&format!("{config}/unset"), &envs)
}

/// Run every case in a child with the config values `config`
/// (`<jk_digest>/<gc_kernel>`, each a value or `unset`) and the variables
/// `envs` (`FERRIC_JK_DIGEST` and `FERRIC_GC_KERNEL` unset unless named).
fn child_with(config: &str, envs: &[(&str, &str)]) -> Vec<Run> {
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args(["child_prints_rimp2_gradient_bits", "--exact", "--nocapture"])
        .env(CHILD, config)
        .env("FERRIC_MEM_BUDGET_GB", BUDGET_GB)
        .env_remove("FERRIC_JK_DIGEST")
        .env_remove("FERRIC_GC_KERNEL");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn the child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child {config} failed: {stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let runs: Vec<Run> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("RIMP2_DIGEST "))
        .map(|l| {
            let mut it = l.split_whitespace();
            let case = it.next().unwrap().to_string();
            let builds = it.next().unwrap().parse().unwrap();
            let kernel = it.next().unwrap().parse().unwrap();
            let values = it
                .map(|h| f64::from_bits(u64::from_str_radix(h, 16).unwrap()))
                .collect();
            Run {
                case,
                builds,
                kernel,
                values,
            }
        })
        .collect();
    assert_eq!(runs.len(), CASES.len(), "child {config}: {stdout}");
    runs
}

/// `ScfJk::jk_digest` reaches the reference SCF and response builds of both
/// drivers: config `legacy` is bitwise an unset config under
/// `FERRIC_JK_DIGEST=legacy` (energy and gradient) and ran no pair-digest
/// build; config `pair` ran pair-digest builds, is not bitwise legacy, and
/// agrees with it to roundoff (1e-9 relative to the largest component),
/// signed and finite.
#[test]
fn config_digest_reaches_the_rimp2_gradient_builds() {
    let legacy = child("legacy", None);
    let env_legacy = child("unset", Some("legacy"));
    let pair = child("pair", None);
    for ((l, e), p) in legacy.iter().zip(&env_legacy).zip(&pair) {
        let case = &l.case;
        assert_eq!(
            l.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            e.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{case}: config legacy is not the unset config under FERRIC_JK_DIGEST=legacy"
        );
        assert_eq!(l.builds, 0, "{case}: config legacy ran the pair digest");
        assert_eq!(
            e.builds, 0,
            "{case}: FERRIC_JK_DIGEST=legacy ran the pair digest"
        );
        eprintln!("{case}: {} pair-digest builds under config pair", p.builds);
        assert!(p.builds > 0, "{case}: config pair did not reach the builds");
        assert!(
            p.values
                .iter()
                .zip(&l.values)
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "{case}: config pair is bitwise legacy"
        );
        let scale = l.values.iter().skip(1).fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (a, b)) in p.values.iter().zip(&l.values).enumerate() {
            assert!(a.is_finite() && b.is_finite(), "{case}[{i}]: {a} vs {b}");
            let d = (a - b).abs();
            let bar = if i == 0 {
                1e-10 * b.abs()
            } else {
                1e-9 * scale
            };
            assert!(d <= bar, "{case}[{i}]: pair {a:e} vs legacy {b:e}");
        }
        let dmax = p
            .values
            .iter()
            .zip(&l.values)
            .skip(1)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        eprintln!(
            "{case}: pair vs legacy |dE| {:.2e}, max|dG| {dmax:.2e} (max|G| {scale:.2e})",
            (p.values[0] - l.values[0]).abs()
        );
    }
}

/// `gc_kernel` (`ScfJk::gc_kernel`) reaches the reference builds the
/// same way: config `auto` is bitwise an unset config under
/// `FERRIC_GC_KERNEL=auto` and served kernel quartets; config `off` served
/// none, and agrees with `auto` to roundoff (the bars of the digest test).
#[test]
fn config_gc_kernel_reaches_the_rimp2_gradient_builds() {
    let auto = child_with("unset/auto", &[]);
    let env_auto = child_with("unset/unset", &[("FERRIC_GC_KERNEL", "auto")]);
    let off = child_with("unset/off", &[]);
    for ((a, e), o) in auto.iter().zip(&env_auto).zip(&off) {
        let case = &a.case;
        assert_eq!(
            a.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            e.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{case}: config auto is not the unset config under FERRIC_GC_KERNEL=auto"
        );
        eprintln!("{case}: {} kernel quartets under config auto", a.kernel);
        assert!(
            a.kernel > 0 && e.kernel == a.kernel,
            "{case}: auto did not reach the builds"
        );
        assert_eq!(o.kernel, 0, "{case}: config off ran the kernel");
        let scale = o.values.iter().skip(1).fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (x, y)) in a.values.iter().zip(&o.values).enumerate() {
            assert!(x.is_finite() && y.is_finite(), "{case}[{i}]: {x} vs {y}");
            let bar = if i == 0 {
                1e-10 * y.abs()
            } else {
                1e-9 * scale
            };
            assert!((x - y).abs() <= bar, "{case}[{i}]: auto {x:e} vs off {y:e}");
        }
    }
}
