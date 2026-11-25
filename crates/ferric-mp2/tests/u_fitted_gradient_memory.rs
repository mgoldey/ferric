//! The memory pre-flight of the UNRESTRICTED RI-MP2 gradient on an RI-JK
//! reference (`ferric_mp2::gradient_memory::fitted_gradient_plan_unrestricted`),
//! against the peak resident set of the process that runs it.
//!
//! Each measurement runs in its own child process (this test binary,
//! re-executed on `child_runs_one_u_fitted_gradient`), so `VmHWM` is the peak of
//! that one gradient and not of whatever else the harness runs in parallel.
//! Systems: the n-butyl and n-octyl radicals (n-butane and n-octane with the
//! last H removed, doublets), cc-pVDZ / cc-pVDZ-RI, JK aux cc-pVTZ-JKFIT (the
//! CLI's matched set for cc-pVDZ).
//!
//! Bars:
//! * bound: the declared peak must be at least the measured `VmHWM` (the
//!   claim the pre-flight makes; no tolerance below it);
//! * refusal: a budget below the declared peak errors before the reference
//!   SCF, so the child's `VmHWM` grows by less than the smallest declared
//!   tensor term at n-butyl (the RI-J raw tensor, 64 MB), not by the gradient;
//! * admission: a budget above the declared peak runs, and its energy and
//!   gradient equal those of the unbudgeted run to 1e-10 (the budget only
//!   moves block widths, which reorder sums: rounding, not a change of
//!   method).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_mp2::gradient_memory::{
    fitted_gradient_plan_unrestricted, UnrestrictedFittedGradientShape,
};
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::u_rimp2_gradient::total_u_rimp2_gradient_on;
use ferric_scf::scf_jk::ScfJk;
use ferric_scf::warm_start::ScfWarmStart;

const CHILD: &str = "FERRIC_U_FITTED_GRAD_MEM_CHILD";
const JK_AUX: &str = "cc-pvtz-jkfit";

/// The alkyl radical C_nH_{2n+1}: alkane `n` without its last H, a doublet.
fn radical(n: usize) -> Molecule {
    let path = format!(
        "{}/../../testdata/molecules/alkane_{n}.xyz",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).unwrap();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let natoms: usize = lines[0].trim().parse().unwrap();
    assert!(lines[1 + natoms].trim_start().starts_with('H'));
    let body = lines[2..1 + natoms].join("\n");
    let xyz = format!("{}\nradical\n{body}\n", natoms - 1);
    Molecule::parse_xyz(&xyz, 0, 2).unwrap()
}

/// Peak resident set of this process (`VmHWM`), in bytes.
fn vm_hwm() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    ferric_core::memory::sys::parse_status_kb_field(&status, "VmHWM").expect("VmHWM")
}

/// The declared peak for `mol` under `budget` with `baseline` resident.
fn declared_peak(mol: &Molecule, budget: usize, baseline: usize) -> usize {
    let obs = PreparedBasis::new(mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let shape = UnrestrictedFittedGradientShape::new(mol, &obs, &dfbs, JK_AUX, 0).unwrap();
    fitted_gradient_plan_unrestricted(shape, budget, baseline).peak_bytes()
}

/// Bytes either side of the admission threshold the refused and admitted
/// runs sit at.
const MARGIN: usize = 64 << 20;

/// The smallest budget the plan admits (`declared_peak(budget) <= budget`),
/// by bisection. The declared peak grows with the budget only through block
/// widths and storage tiers, at a slope below one, so the admitted budgets
/// are the ones above this threshold.
fn admission_threshold(mol: &Molecule, baseline: usize) -> usize {
    let (mut lo, mut hi) = (0usize, 1usize << 40);
    assert!(declared_peak(mol, hi, baseline) <= hi);
    while hi - lo > 1 << 20 {
        let mid = lo + (hi - lo) / 2;
        if declared_peak(mol, mid, baseline) <= mid {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// One child's report.
#[derive(Debug)]
struct Child {
    declared: usize,
    hwm_before: usize,
    hwm_after: usize,
    /// `Ok((energy, gradient))` or the error text.
    result: Result<(f64, Vec<f64>), String>,
}

/// Run `child_runs_one_u_fitted_gradient` on the radical of alkane `n` with `budget`
/// (`"auto"`, `"below"`, `"above"` or a byte count).
fn run_child(n: usize, budget: &str) -> Child {
    let spec = format!("{n}:{budget}");
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["child_runs_one_u_fitted_gradient", "--ignored", "--exact"])
        .args(["--nocapture", "--test-threads=1"])
        .env(CHILD, &spec)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child {spec} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let line = stdout
        .lines()
        .find_map(|l| l.split_once("FGM ").map(|(_, r)| r.to_string()))
        .unwrap_or_else(|| panic!("child {spec} printed no report: {stdout}"));
    let mut f = line.splitn(4, ' ');
    let mut num = || f.next().unwrap().parse::<usize>().unwrap();
    let (declared, hwm_before, hwm_after) = (num(), num(), num());
    let rest = f.next().unwrap();
    let result = match rest.split_once(' ') {
        Some(("ok", v)) => {
            let vals: Vec<f64> = v.split(' ').map(|x| x.parse().unwrap()).collect();
            Ok((vals[0], vals[1..].to_vec()))
        }
        _ => Err(rest.to_string()),
    };
    Child {
        declared,
        hwm_before,
        hwm_after,
        result,
    }
}

/// The child: one fitted gradient of the radical and budget in `CHILD`, then
/// `FGM <declared> <VmHWM before> <VmHWM after> ok <E> <g...>` or `... err <msg>`.
#[test]
#[ignore = "child process of the tests in this file (needs FERRIC_U_FITTED_GRAD_MEM_CHILD)"]
fn child_runs_one_u_fitted_gradient() {
    let Ok(spec) = std::env::var(CHILD) else {
        return;
    };
    let (n, budget) = spec.split_once(':').unwrap();
    let mol = radical(n.parse().unwrap());
    // "below" / "above": MARGIN either side of this process's own admission
    // threshold (its baseline, not the parent's).
    let entry = ferric_core::memory::read_own_rss_bytes().unwrap();
    let explicit = match budget {
        "auto" => None,
        "below" => Some(admission_threshold(&mol, entry) - MARGIN),
        "above" => Some(admission_threshold(&mol, entry) + MARGIN),
        b => Some(b.parse::<usize>().unwrap()),
    };
    let obs = basis::bundled("cc-pvdz").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let scf_jk = ScfJk::rijk(JK_AUX);
    let config = RiMp2Config {
        memory_budget_bytes: explicit,
        ..Default::default()
    };
    let resolved = ferric_core::memory::resolve_budget_bytes(explicit);
    let baseline = ferric_core::memory::read_own_rss_bytes().unwrap();
    let declared = declared_peak(&mol, resolved, baseline);
    let hwm_before = vm_hwm();
    let r = total_u_rimp2_gradient_on(
        &mol,
        &obs,
        &aux,
        ferric_integrals::operator::Operator::coulomb(),
        &config,
        None,
        &scf_jk,
        &mut ScfWarmStart::new(false),
    );
    let hwm_after = vm_hwm();
    let tail = match r {
        Ok((e, g)) => {
            let g: Vec<String> = g.iter().map(|x| format!("{x:e}")).collect();
            format!("ok {e:e} {}", g.join(" "))
        }
        Err(e) => format!("err {}", e.to_string().replace('\n', " | ")),
    };
    println!("FGM {declared} {hwm_before} {hwm_after} {tail}");
}

/// The declared peak bounds the measured peak RSS at n-butyl and n-octyl.
#[test]
fn declared_peak_bounds_the_measured_peak_rss() {
    // An explicit budget just above this system's admission threshold, not
    // "auto": the auto budget follows the machine's RAM (3.2 GB on a CI
    // runner, which refuses octane), and the bound under test does not
    // depend on which budget admitted the run.
    for n in [4, 8] {
        let c = run_child(n, "above");
        assert!(c.result.is_ok(), "alkane_{n}: {:?}", c.result);
        let gb = |b: usize| b as f64 / 1e9;
        eprintln!(
            "C{n}H{} radical/cc-pVDZ: declared {:.3} GB, VmHWM {:.3} GB, ratio {:.2}",
            2 * n + 1,
            gb(c.declared),
            gb(c.hwm_after),
            c.declared as f64 / c.hwm_after as f64
        );
        assert!(
            c.declared >= c.hwm_after,
            "alkane_{n}: declared peak {:.3} GB is below the measured VmHWM {:.3} GB",
            gb(c.declared),
            gb(c.hwm_after)
        );
    }
}

/// A budget just below the admission threshold is refused before the
/// reference SCF; one just above it runs, with the energy and gradient of the
/// unbudgeted run.
#[test]
fn a_budget_below_the_declared_peak_errors_and_one_above_runs_unchanged() {
    let refused = run_child(4, "below");
    let msg = refused
        .result
        .expect_err("a budget below the declared peak ran");
    assert!(msg.contains("fitted U-MP2 gradient requires"), "{msg}");
    assert!(msg.contains("jk = \"exact\""), "{msg}");
    assert!(
        refused.hwm_after - refused.hwm_before < MARGIN,
        "the refusal allocated {} bytes first",
        refused.hwm_after - refused.hwm_before
    );

    let (e_ref, g_ref) = run_child(4, "auto").result.unwrap();
    let admitted = run_child(4, "above");
    assert!(
        admitted.hwm_after <= admitted.declared,
        "admitted run peaked at {} B over its declared {} B",
        admitted.hwm_after,
        admitted.declared
    );
    let (e, g) = admitted
        .result
        .expect("a budget above the declared peak was refused");
    assert!((e - e_ref).abs() < 1e-10, "energy moved by {:e}", e - e_ref);
    let dg = g
        .iter()
        .zip(&g_ref)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(dg < 1e-10, "gradient moved by {dg:e}");
}
