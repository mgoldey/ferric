//! τ-only meta-GGA (r2SCAN) orbital-Hessian consumers: stability vs PySCF and
//! Newton vs DIIS.
//!
//! The orbital Hessian of a meta-GGA reference is the HF-like part plus the
//! XC response δV_xc = f_xc·δD, which `rohf::FxcKernelStore::Mgga`
//! (`ferric_dft::fxc::MggaFxcKernel`) supplies with the ρ×τ, σ×τ and τ×τ
//! blocks. References: `scripts/validation/gen_mgga_response.py` (PySCF
//! `newton_ah.gen_g_hop_uhf` / `gen_g_hop_rhf` on `dft.UKS` / `dft.RKS`
//! r2SCAN, dense), 6-31G, exact J, the (75,110) unpruned Becke grid with
//! Becke-1988 radii on both sides.
//!
//! * `nh2_uks_r2scan_hessian_matches_pyscf` — UKS-internal: the dense
//!   spectrum built from ferric's `uhf_newton::hessian_matvec` with the τ
//!   kernel, and the `check_stability` Davidson λ_min, against PySCF's dense
//!   spectrum.
//! * `water_rks_r2scan_singlet_hessian_matches_pyscf` — RKS-internal
//!   (singlet), the same two checks against PySCF `gen_g_hop_rhf` / 2.
//! * `*_newton_matches_diis` — RKS and UKS Newton converge to the DIIS
//!   energy, and the τ kernel was built (`MGGA_FXC_KERNEL_BUILDS`).
//! * `roks_r2scan_newton_request_keeps_diis` — ROKS Newton stays on DIIS for
//!   a meta-GGA (gate in `rohf.rs`).
//!
//! Artifact hypothesis: dropping δτ from the kernel's linearization moves the
//! water singlet λ_min by 9.8e-4 Ha (measured by mutation), five orders above
//! the 1.5e-10 a correct kernel leaves. Newton-vs-DIIS cannot see a kernel defect (an
//! approximate Hessian still converges to the same stationary point); it pins
//! only that the meta-GGA Newton path engages and lands on the DIIS state.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Mutex;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::engine_pool::EnginePool;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::{solve_rohf, FxcKernelStore, MGGA_FXC_KERNEL_BUILDS};
use ferric_scf::rohf_newton::FxcResponse;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::{StabilityKind, StabilityVerdict};
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/mgga_response";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "6-31g";
const XC: &str = "r2SCAN";
const N_COMPARE: usize = 6;

// TOLERANCES (measured values in the comments).
/// Nuclear repulsion vs the reference (geometry/unit check).
const TOL_ENUC: f64 = 1e-9;
/// r2SCAN energy vs PySCF on the matched grid. Measured 5.7e-14 (NH2, water).
const TOL_E: f64 = 1e-9;
/// Hessian eigenvalues vs PySCF (dense and Davidson), first order in the SCF
/// density error. Measured max 1.5e-10 (NH2 dense[2]).
const TOL_LAMBDA: f64 = 1e-8;
/// Newton vs DIIS energy. Measured 0 (water RKS), 1.4e-14 (NH2 UKS).
const TOL_NEWTON: f64 = 1e-9;

/// Serializes the tests that read the global kernel-build counter.
static COUNTER_LOCK: Mutex<()> = Mutex::new(());

fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("testdata").is_dir() && p.join("crates").is_dir()
    };
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| looks_like_root(p))
        .expect("workspace root above ferric-scf")
        .to_path_buf()
}

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing reference {} ({e}); regenerate with scripts/validation/gen_mgga_response.py",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn nums(v: &Value, ptr: &str) -> Vec<f64> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("reference field {ptr} missing"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn num(v: &Value, ptr: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("reference field {ptr} missing"))
}

struct System {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn load(system: &str, mult: usize, r: &Value) -> System {
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, mult).unwrap();
    let enuc = mol.nuclear_repulsion();
    let enuc_ref = num(r, "/nuclear_repulsion");
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{system}: E_nuc {enuc:.12} vs {enuc_ref:.12}"
    );
    let bs = basis::bundled(BASIS).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    assert_eq!(prep.nbasis() as u64, r["nao"].as_u64().unwrap());
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    System {
        mol,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

/// The reference grid: (75,110), unpruned, Becke partition and radii.
fn grid() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 75,
        n_angular: 110,
        ..Default::default()
    }
}

/// r2SCAN, exact J (`Some("")` forces four-centre J on the restricted path;
/// the open-shell solvers read `None` as exact J).
fn config(restricted: bool) -> RhfConfig {
    RhfConfig {
        xc: Some(XC.into()),
        dft_grid: Some(grid()),
        df_j_aux: if restricted {
            Some(String::new())
        } else {
            None
        },
        df_k_aux: None,
        max_iter: 500,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    }
}

fn close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<14} ferric {got:+.12e} ref {want:+.12e} |d| {d:.2e}");
    assert!(d < tol, "{ctx}: {what} |d| {d:.2e} >= {tol:.0e}");
}

fn sym_eigh(h: &Array2<f64>) -> Vec<f64> {
    let asym = (h - &h.t()).iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    assert!(asym < 1e-8, "orbital Hessian not symmetric ({asym:.2e})");
    (0.5 * (h + &h.t())).eigh(UPLO::Lower).unwrap().0.to_vec()
}

fn builds() -> usize {
    MGGA_FXC_KERNEL_BUILDS.load(Ordering::Relaxed)
}

#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn nh2_uks_r2scan_hessian_matches_pyscf() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let r = reference("nh2");
    let s = load("nh2", 2, &r);
    let ctx = "nh2/6-31g/UKS-r2SCAN";
    let cfg = RhfConfig {
        check_stability: true,
        ..config(false)
    };
    let before = builds();
    let res = solve_uhf(&s.ctx, &s.mol, &s.prep, &s.bounds, &cfg).unwrap();
    assert!(res.converged, "{ctx}: SCF not converged");
    assert!(
        builds() > before,
        "{ctx}: stability did not build the τ kernel"
    );
    close(
        ctx,
        "energy",
        res.energy,
        num(&r, "/uks_r2scan/energy"),
        TOL_E,
    );
    let want = nums(&r, "/uks_r2scan/hessian/lowest");

    let st = res.stability.as_ref().expect("check_stability verdict");
    eprintln!("{ctx}: {}", st.summary());
    assert_eq!(st.kind, StabilityKind::UhfInternal);
    assert!(st.converged);
    assert_eq!(st.verdict(), StabilityVerdict::Stable, "{}", st.summary());
    close(
        ctx,
        "Davidson λmin",
        st.lowest_eigenvalue,
        want[0],
        TOL_LAMBDA,
    );

    // Dense spectrum from the public matvec with the τ kernel.
    // Doublet: one unpaired α electron.
    let (na, nb) = {
        let ne = s.mol.nelec() as usize;
        (ne.div_ceil(2), ne / 2)
    };
    let c_a = res.mos_alpha.clone();
    let c_b = res.mos_beta.clone().unwrap();
    let d_a = c_a
        .slice(ndarray::s![.., ..na])
        .dot(&c_a.slice(ndarray::s![.., ..na]).t());
    let d_b = c_b
        .slice(ndarray::s![.., ..nb])
        .dot(&c_b.slice(ndarray::s![.., ..nb]).t());
    let store = FxcKernelStore::build(&s.mol, &s.prep, &grid(), XC, None, &d_a, &d_b).unwrap();
    assert!(matches!(store, FxcKernelStore::Mgga { .. }));
    let resp = store.response();
    let fxc: &FxcResponse<'_> = &*resp;
    let f_a = c_a.t().dot(&res.fock_alpha).dot(&c_a);
    let f_b = c_b.t().dot(res.fock_beta.as_ref().unwrap()).dot(&c_b);
    let inp = ferric_scf::uhf_newton::UhfNewtonInputs {
        prep: &s.prep,
        bounds: &s.bounds,
        c_a: &c_a,
        c_b: &c_b,
        f_a_mo: &f_a,
        f_b_mo: &f_b,
        nocc_a: na,
        nocc_b: nb,
        k_mix_sr: 0.0,
        rsh: None,
        fxc: Some(fxc),
        thresh: cfg.integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let n = c_a.nrows();
    let (da, db) = ((n - na) * na, (n - nb) * nb);
    let pool = EnginePool::new(s.bounds.op, &s.prep, 1e-14).unwrap();
    let mut h = Array2::<f64>::zeros((da + db, da + db));
    for col in 0..da + db {
        let mut v = vec![0.0; da + db];
        v[col] = 1.0;
        let ka = Array2::from_shape_vec((n - na, na), v[..da].to_vec()).unwrap();
        let kb = Array2::from_shape_vec((n - nb, nb), v[da..].to_vec()).unwrap();
        let (ha, hb) =
            ferric_scf::uhf_newton::hessian_matvec(&s.ctx, &inp, &ka, &kb, &pool).unwrap();
        for (row, x) in ha.iter().chain(hb.iter()).enumerate() {
            h[(row, col)] = *x;
        }
    }
    let got = sym_eigh(&h);
    for k in 0..N_COMPARE {
        close(ctx, &format!("dense[{k}]"), got[k], want[k], TOL_LAMBDA);
    }
}

#[test]
fn water_rks_r2scan_singlet_hessian_matches_pyscf() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let r = reference("water_eq");
    let s = load("water_eq", 1, &r);
    let ctx = "water/6-31g/RKS-r2SCAN";
    let cfg = RhfConfig {
        check_stability: true,
        ..config(true)
    };
    let before = builds();
    let res = solve_rhf(
        &s.ctx,
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &cfg,
    )
    .unwrap();
    assert!(res.converged, "{ctx}: SCF not converged");
    assert!(
        builds() > before,
        "{ctx}: stability did not build the τ kernel"
    );
    close(
        ctx,
        "energy",
        res.energy,
        num(&r, "/rks_r2scan/energy"),
        TOL_E,
    );
    let want = nums(&r, "/rks_r2scan/singlet_hessian/lowest");

    let st = res.stability.as_ref().expect("check_stability verdict");
    eprintln!("{ctx}: {}", st.summary());
    assert_eq!(st.kind, StabilityKind::RhfInternal);
    assert!(st.converged);
    assert_eq!(st.verdict(), StabilityVerdict::Stable, "{}", st.summary());
    close(
        ctx,
        "Davidson λmin",
        st.lowest_eigenvalue,
        want[0],
        TOL_LAMBDA,
    );

    let c = res.mos_r().clone();
    let nocc = (s.mol.nelec() / 2) as usize;
    let d_half = 0.5 * res.density_r();
    let store =
        FxcKernelStore::build(&s.mol, &s.prep, &grid(), XC, None, &d_half, &d_half).unwrap();
    let resp = store.response();
    let fxc: &FxcResponse<'_> = &*resp;
    let f_mo = c.t().dot(res.fock_r()).dot(&c);
    let inp = ferric_scf::rhf_newton::RhfNewtonInputs {
        prep: &s.prep,
        bounds: &s.bounds,
        c: &c,
        f_mo: &f_mo,
        nocc,
        k_mix_sr: 0.0,
        rsh: None,
        fxc: Some(fxc),
        thresh: cfg.integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let n = c.nrows();
    let (no, nv) = (nocc, n - nocc);
    let pool = EnginePool::new(s.bounds.op, &s.prep, 1e-14).unwrap();
    let mut h = Array2::<f64>::zeros((nv * no, nv * no));
    for col in 0..nv * no {
        let mut v = vec![0.0; nv * no];
        v[col] = 1.0;
        let k = Array2::from_shape_vec((nv, no), v).unwrap();
        let hv = ferric_scf::rhf_newton::hessian_matvec(&s.ctx, &inp, &k, &pool).unwrap();
        for (row, x) in hv.iter().enumerate() {
            h[(row, col)] = *x;
        }
    }
    let got = sym_eigh(&h);
    for k in 0..N_COMPARE {
        close(ctx, &format!("dense[{k}]"), got[k], want[k], TOL_LAMBDA);
    }
}

/// Newton (`newton_trigger`) vs DIIS-only on the same system: same energy,
/// and the Newton run built the τ kernel.
fn newton_vs_diis(ctx: &str, run: impl Fn(&RhfConfig) -> ferric_scf::ScfResult, diis: RhfConfig) {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let newton = RhfConfig {
        newton_trigger: 1e-2,
        ..diis.clone()
    };
    let before = builds();
    let r_diis = run(&diis);
    assert_eq!(builds(), before, "{ctx}: DIIS-only built a kernel");
    let r_newton = run(&newton);
    let n_builds = builds() - before;
    eprintln!(
        "{ctx}: DIIS E={:.12} ({} it), Newton E={:.12} ({} it, {n_builds} τ-kernel builds)",
        r_diis.energy, r_diis.iterations, r_newton.energy, r_newton.iterations
    );
    assert!(
        r_diis.converged && r_newton.converged,
        "{ctx}: not converged"
    );
    assert!(n_builds >= 1, "{ctx}: Newton fell back to DIIS");
    close(
        ctx,
        "Newton-DIIS",
        r_newton.energy,
        r_diis.energy,
        TOL_NEWTON,
    );
}

#[test]
fn rks_r2scan_newton_matches_diis() {
    let r = reference("water_eq");
    let s = load("water_eq", 1, &r);
    newton_vs_diis(
        "water/RKS-r2SCAN",
        |c| solve_rhf(&s.ctx, &s.mol, &s.prep, Operator::coulomb(), &s.bounds, c).unwrap(),
        config(true),
    );
}

#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn uks_r2scan_newton_matches_diis() {
    let r = reference("nh2");
    let s = load("nh2", 2, &r);
    newton_vs_diis(
        "nh2/UKS-r2SCAN",
        |c| solve_uhf(&s.ctx, &s.mol, &s.prep, &s.bounds, c).unwrap(),
        config(false),
    );
}

/// ROKS Newton keeps DIIS for a meta-GGA (its Newton step on the τ kernel is
/// not anchored, see `rohf.rs`): a newton_trigger run builds no τ kernel and
/// lands on the DIIS energy.
#[test]
fn roks_r2scan_newton_request_keeps_diis() {
    let _g = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let r = reference("nh2");
    let s = load("nh2", 2, &r);
    let diis = RhfConfig {
        density_conv: 1e-7,
        energy_conv: 1e-10,
        ..config(false)
    };
    let newton = RhfConfig {
        newton_trigger: 1e-2,
        ..diis.clone()
    };
    let run = |c: &RhfConfig| {
        solve_rohf(&s.ctx, &s.mol, &s.prep, Operator::coulomb(), &s.bounds, c).unwrap()
    };
    let r_diis = run(&diis);
    let before = builds();
    let r_newton = run(&newton);
    assert_eq!(builds(), before, "ROKS Newton built the τ kernel");
    close(
        "nh2/ROKS-r2SCAN",
        "Newton-DIIS",
        r_newton.energy,
        r_diis.energy,
        TOL_NEWTON,
    );
}
