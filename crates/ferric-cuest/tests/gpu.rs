//! GPU tests: need the `cuest` feature and a device cuEST supports (cuEST 0.3
//! is tested on compute capability 8.0 and newer).
//!
//! On an unsupported device (or none) the tests print SKIP and pass, unless
//! FERRIC_CUEST_REQUIRE_DEVICE=1, which turns a skip into a failure. A GPU
//! tester should always set it, so a broken device setup cannot pass.
//!
//!     FERRIC_CUEST_REQUIRE_DEVICE=1 cargo test -p ferric-cuest --features cuest \
//!         --test gpu -- --nocapture --test-threads=1
//!
//! `one_electron_matches_libint` writes a JSON report (path printed at the
//! end), pass or fail, including when device setup or the stored references
//! fail, with everything needed to diagnose a mismatch from a single run:
//! the AO order and phases cuEST actually uses, any normalization factor,
//! and the max errors with both our guessed and the data-inferred AO map.
#![cfg(feature = "cuest")]

use ferric_cuest::cuda::DeviceInfo;
use ferric_cuest::handle::CuestHandle;
use ferric_cuest::layout::{classify, infer_map, max_abs_diff, AoMap, ClassVerdict};
use ferric_cuest::oe::compute_one_electron;
use ferric_cuest::reference::all_references;
use ferric_cuest::CuestError;
use serde::Serialize;

/// Max-abs-error bars against libint.
///
/// libint runs at engine precision 1e-14 and cuEST in fp64 (V with ffloat
/// forced off); with at most a few hundred primitive pairs per element the
/// accumulated error of either code is ~1e-12. S and T elements are O(1),
/// V elements up to ~sum Z = 10. A convention error (order, phase,
/// normalization, sign of V) is >= 1e-3. 1e-10 for S, T and 1e-9 for V keep
/// two orders of margin over rounding and six under the smallest real defect.
const BAR_ST: f64 = 1e-10;
const BAR_V: f64 = 1e-9;
/// Inference tolerance: residual and |scale| - 1 below this count as exact.
const INFER_TOL: f64 = 1e-8;

fn require_device() -> bool {
    std::env::var("FERRIC_CUEST_REQUIRE_DEVICE").is_ok_and(|v| v == "1")
}

/// Ok(Some(handle)); Ok(None) after printing SKIP; Err when a usable device
/// is required (FERRIC_CUEST_REQUIRE_DEVICE=1) or creation failed otherwise.
fn handle_or_skip(test: &str) -> Result<Option<CuestHandle>, String> {
    match CuestHandle::new() {
        Ok(h) => Ok(Some(h)),
        Err(CuestError::DeviceUnavailable(why)) if !require_device() => {
            println!(
                "SKIP {test}: {why} (set FERRIC_CUEST_REQUIRE_DEVICE=1 to make this a failure)"
            );
            Ok(None)
        }
        Err(e) => Err(format!("{test}: cuEST handle creation failed: {e}")),
    }
}

#[test]
fn device_probe_and_handle() {
    match DeviceInfo::probe() {
        Ok(d) => println!("CUDA device: {d:?}"),
        Err(e) => println!("CUDA device probe failed: {e}"),
    }
    let Some(h) = handle_or_skip("device_probe_and_handle").unwrap_or_else(|e| panic!("{e}"))
    else {
        return;
    };
    let v = h.version().expect("cuestGet*Version");
    println!("cuEST library version {}.{}.{}", v.0, v.1, v.2);
    assert_eq!((v.0, v.1), (0, 3), "ffi.rs targets cuEST 0.3.x");
}

#[derive(Serialize)]
struct SystemReport {
    file: String,
    basis: String,
    nao: usize,
    cuest_pure: bool,
    /// setup, S, T, V seconds.
    timings: [f64; 4],
    error: Option<String>,
    classes: Vec<ClassVerdict>,
    /// Max |cuEST - libint| after reordering with the GUESSED map.
    hypothesis_max_err: Option<[f64; 3]>,
    /// The same with the map INFERRED from the data (when every class is clean).
    inferred_max_err: Option<[f64; 3]>,
    /// Range of diag(S) straight from cuEST (normalization check).
    s_diag_min_max: Option<[f64; 2]>,
    pass: bool,
}

#[derive(Serialize)]
struct Report {
    device: Option<String>,
    cuest_version: Option<(u32, u32, u32)>,
    /// Why the run stopped before comparing any system (device setup, or the
    /// stored references); `None` when every system ran.
    setup_error: Option<String>,
    systems: Vec<SystemReport>,
}

impl Report {
    fn write(&self) -> std::path::PathBuf {
        let out = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("cuest-oe-report.json");
        std::fs::write(&out, serde_json::to_string_pretty(self).unwrap()).unwrap();
        println!("REPORT: {}", out.display());
        out
    }

    /// Record a setup failure, write the report, then fail.
    fn fail_setup(mut self, why: String) -> ! {
        self.setup_error = Some(why.clone());
        let out = self.write();
        panic!("{why}; send {}", out.display());
    }
}

fn run_one(file: &str, r: &ferric_cuest::reference::OneElectronReference) -> SystemReport {
    let mol = r.molecule();
    let bs = r.basis_set().expect("bundled basis");
    let mut rep = SystemReport {
        file: file.into(),
        basis: r.basis.clone(),
        nao: r.nao,
        cuest_pure: false,
        timings: [0.0; 4],
        error: None,
        classes: vec![],
        hypothesis_max_err: None,
        inferred_max_err: None,
        s_diag_min_max: None,
        pass: false,
    };
    let m = match compute_one_electron(&mol, &bs) {
        Ok(m) => m,
        Err(e) => {
            rep.error = Some(e.to_string());
            return rep;
        }
    };
    rep.cuest_pure = m.layout.cuest_pure;
    rep.timings = m.timings;
    let d = m.s.diag();
    rep.s_diag_min_max = Some([
        d.fold(f64::INFINITY, |a, &b| a.min(b)),
        d.fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
    ]);
    let (rs, rt, rv) = (r.s_matrix(), r.t_matrix(), r.v_matrix());
    let errs = |map: &AoMap| {
        [
            max_abs_diff(&map.to_ferric(&m.s), &rs),
            max_abs_diff(&map.to_ferric(&m.t), &rt),
            max_abs_diff(&map.to_ferric(&m.v), &rv),
        ]
    };
    let hyp = AoMap::hypothesis(&m.layout);
    let he = errs(&hyp);
    rep.hypothesis_max_err = Some(he);
    let inferred = infer_map(&m.layout, &[&m.s, &m.t, &m.v], &[&rs, &rt, &rv]);
    rep.classes = classify(&m.layout, &inferred, INFER_TOL);
    if rep.classes.iter().all(|c| c.components.is_some()) {
        let classes = &rep.classes;
        let map = AoMap::from_components(&m.layout, |sh| {
            classes
                .iter()
                .find(|c| {
                    (c.l, c.cuest_pure, c.ferric_pure) == (sh.l, sh.cuest_pure, sh.ferric_pure)
                })
                .and_then(|c| c.components.clone())
                .expect("class present")
        });
        rep.inferred_max_err = Some(errs(&map));
    }
    rep.pass = he[0] < BAR_ST
        && he[1] < BAR_ST
        && he[2] < BAR_V
        && rep.classes.iter().all(|c| c.matches_hypothesis);
    rep
}

#[test]
fn one_electron_matches_libint() {
    let mut report = Report {
        device: DeviceInfo::probe().ok().map(|d| format!("{d:?}")),
        cuest_version: None,
        setup_error: None,
        systems: vec![],
    };
    let h = match handle_or_skip("one_electron_matches_libint") {
        Ok(Some(h)) => h,
        Ok(None) => return,
        Err(why) => report.fail_setup(why),
    };
    report.cuest_version = h.version().ok();
    drop(h);
    let refs = match all_references() {
        Ok(r) => r,
        Err(why) => report.fail_setup(why),
    };
    for (path, r) in refs {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let rep = run_one(&name, &r);
        println!(
            "{:<32} nao={:<3} pure={:<5} pass={:<5} hyp_err={:?} inferred_err={:?} sdiag={:?} error={:?}",
            name, rep.nao, rep.cuest_pure, rep.pass, rep.hypothesis_max_err, rep.inferred_max_err, rep.s_diag_min_max,
            rep.error
        );
        for c in rep.classes.iter().filter(|c| !c.matches_hypothesis) {
            println!(
                "    class l={} cuest_pure={} ferric_pure={}: inferred {:?}, guessed {:?}, problem {:?}",
                c.l, c.cuest_pure, c.ferric_pure, c.components, c.hypothesis, c.problem
            );
        }
        report.systems.push(rep);
    }
    let out = report.write();
    let failed: Vec<&str> = report
        .systems
        .iter()
        .filter(|s| !s.pass)
        .map(|s| s.file.as_str())
        .collect();
    assert!(
        failed.is_empty(),
        "cuEST vs libint failed for {failed:?}; send {}",
        out.display()
    );
}
