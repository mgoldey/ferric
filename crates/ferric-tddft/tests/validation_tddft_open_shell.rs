//! Spin-conserving TDA / TDDFT on UHF and UKS references against PySCF.
//!
//! References: `testdata/reference/validation/tddft_open_shell/<mol>_6-31g.json`,
//! from `scripts/validation/gen_tddft_open_shell_refs.py`. Every case is
//! `#[ignore]`d:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-tddft --test validation_tddft_open_shell \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! # What is matched, and so what the bar is
//!
//! As `validation_tddft_rsh.rs`: the bundled basis, the libxc identifiers,
//! the flat (75, 110) Becke grid on both sides, the RI-J/K reference SCF
//! (UHF / UKS here, the range-separated exchange fitted as ferric fits it)
//! and the response integrals by ferric's Coulomb- and erf-metric RI. What
//! is left is SCF convergence (ferric's density to 1e-8, PySCF's orbital
//! gradient to 1e-7): a few 1e-6 eV on an excitation energy, first order in
//! the orbitals. The energy bar is `TOL_EV = 2e-5` eV, the bar of the
//! closed-shell comparisons.
//!
//! `<S²>`: a perturbation `δA` of the response matrix rotates an eigenvector
//! by `sin θ ≤ ‖δA‖/gap`, and `<S²>` (a quadratic form with `‖S²‖ ≤ 4` on
//! these spaces) moves by at most `2·4·sin θ`. With `‖δA‖` at the energy
//! bar, the `<S²>` bar of a root is `8·TOL_HA/gap`, `gap` its distance to
//! the nearest other root; it is compared on roots isolated by
//! `ISOLATED_GAP_EV`. For TDA the sum of `<S²>` over a cluster of
//! near-degenerate roots (spaced below `ISOLATED_GAP_EV`) is invariant under
//! rotations inside the cluster and is compared with the cluster's gap to
//! the rest. The reference values are checked in the generator against an
//! independent construction (pyscf.fci.spin_op on the expanded determinant
//! vector, OH and NH2).
//!
//! OH is compared by TDA on its UHF only: its ²Π ground state is doubly
//! degenerate, so the β π → π rotation between the two components is a zero
//! mode of A+B and A−B and its Casida Ω² sits at the noise floor with either
//! sign; and its UKS does not converge in PySCF (the occupied and the empty
//! β π component are degenerate in KS), so there is no KS reference.
//!
//! # Teeth
//!
//! * The α-β kernel block: for each KS case the reference's TDA roots with
//!   the α-β f_xc block dropped (`tda_no_fxc_ab`) sit far outside the bar,
//!   and ferric must too.
//! * The reference SCF: ferric's UHF / UKS energy and `<S²>_0` equal
//!   PySCF's (a different UHF solution would fail here first).
//! * Coefficients: c_sr, c_lr and ω equal PySCF's.
//!
//! Mutations (applied and reverted, 2026-11-14):
//! * the α-β f_xc block dropped in `UksOvFxc::apply`: `nh2_pbe` fails (TDA
//!   state 2 oscillator strength 3.30e-3 vs PySCF 4.07e-3, before the energy
//!   check; the negative control above sits 0.06 eV off);
//! * the cross-spin Coulomb block zeroed in the dense matrices: `nh2_hf`
//!   fails (state 0 f off by 1.1e-4);
//! * the cross term of the `<S²>` formula with its sign flipped: `nh2_hf`
//!   fails on `<S²>` (roots 0..1: 0.7620 vs 0.7584, bar 2.8e-5);
//! * Olsen's projection removed from the Davidson corrections (the solver
//!   before this suite): `o2_lda`, `o2_pbe`, `o2_r2scan` fail (one root
//!   stuck at residual 0.3-0.4 Ha after 100 iterations, TDA and Casida).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ferric_tddft::{run_tddft, TddftConfig, TddftMethod, TddftResult, TddftSolver};
use serde_json::Value;

const HARTREE_TO_EV: f64 = 27.211_386_245_988;
/// Excitation-energy bar; derived in the module docs.
const TOL_EV: f64 = 2e-5;
const TOL_HA: f64 = TOL_EV / HARTREE_TO_EV;
/// SCF energy agreement (Ha): both SCFs converge the energy far below it,
/// and a different UHF solution sits mHa away.
const SCF_TOL: f64 = 1e-6;
/// How many of the lowest roots are compared.
const N_CHECK: usize = 8;
const ISOLATED_GAP_EV: f64 = 1e-2;
const OSC_ABS: f64 = 1e-4;
const OSC_REL: f64 = 1e-3;
const NEG_CONTROL_FACTOR: f64 = 100.0;
const SCF_JK_AUX: &str = "def2-universal-jkfit";
const BASIS: &str = "6-31g";
const SOLVERS: [TddftSolver; 2] = [TddftSolver::Dense, TddftSolver::Davidson];

fn flat_grid() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 75,
        n_angular: 110,
        ..Default::default()
    }
}

fn load_ref(mol: &str) -> Value {
    let path = format!(
        "{}/../../testdata/reference/validation/tddft_open_shell/{mol}_{BASIS}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let txt = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing reference {path}: {e}\n  generate it with `OPENBLAS_NUM_THREADS=1 uv run \
             --no-sync python scripts/validation/gen_tddft_open_shell_refs.py`"
        )
    });
    serde_json::from_str(&txt).unwrap()
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected an array, got {v}"))
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    scf: ScfResult,
}

/// The UHF / UKS reference SCF on the flat grid, RI-J/K, J in f64.
fn setup(refj: &Value, xc: Option<&str>) -> Setup {
    let xyz = format!(
        "{}/../../{}",
        env!("CARGO_MANIFEST_DIR"),
        refj["xyz"].as_str().unwrap()
    );
    let mult = refj["multiplicity"].as_u64().unwrap() as usize;
    let mol = Molecule::load_xyz_with_charge(&xyz, 0, mult).unwrap();
    assert_eq!(refj["basis"].as_str().unwrap(), BASIS);
    assert_eq!(refj["scf_jk_aux"].as_str().unwrap(), SCF_JK_AUX);
    let aux_name = refj["response_aux"].as_str().unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(BASIS).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let mut cfg = RhfConfig::default()
        .with_dft_grid(flat_grid())
        .with_df_j_aux(SCF_JK_AUX.to_string())
        .with_df_k_aux(SCF_JK_AUX.to_string())
        .with_ri_j_precision(ferric_scf::df_j::RiJPrecision::F64)
        .with_density_conv(1e-8)
        .with_max_iter(300);
    cfg.xc = xc.map(str::to_string);
    let scf = solve_uhf(&ParallelContext::default(), &mol, &obs, &bounds, &cfg).unwrap();
    assert!(scf.converged, "{xc:?}: reference SCF did not converge");
    Setup {
        mol,
        obs,
        dfbs,
        scf,
    }
}

fn run(
    s: &Setup,
    xc: Option<&str>,
    method: TddftMethod,
    solver: TddftSolver,
) -> Result<TddftResult, ferric_core::error::FerricError> {
    let cfg = TddftConfig {
        n_roots: N_CHECK,
        method,
        grid: flat_grid(),
        xc: xc.map(str::to_string),
        solver,
        ..Default::default()
    };
    run_tddft(&s.mol, &s.obs, &s.dfbs, &s.scf, &cfg)
}

fn ev(r: &TddftResult) -> Vec<f64> {
    r.excitation_energies
        .iter()
        .map(|w| w * HARTREE_TO_EV)
        .collect()
}

fn max_dev(a: &[f64], b: &[f64], n: usize) -> f64 {
    (0..n).map(|k| (a[k] - b[k]).abs()).fold(0.0_f64, f64::max)
}

/// Clusters of consecutive roots spaced below `ISOLATED_GAP_EV` among the
/// first `n`, each with its gap (Ha) to the roots outside it; clusters that
/// touch root `n − 1` are dropped (the cluster may continue past `n`).
fn clusters(w_ev: &[f64], n: usize) -> Vec<(std::ops::Range<usize>, f64)> {
    let mut out = Vec::new();
    let mut lo = 0;
    for k in 1..=w_ev.len().min(n + 1) {
        if k == w_ev.len() || w_ev[k] - w_ev[k - 1] > ISOLATED_GAP_EV {
            if k < n {
                let below = if lo == 0 {
                    f64::INFINITY
                } else {
                    w_ev[lo] - w_ev[lo - 1]
                };
                let above = if k < w_ev.len() {
                    w_ev[k] - w_ev[k - 1]
                } else {
                    f64::INFINITY
                };
                out.push((lo..k, below.min(above) / HARTREE_TO_EV));
            }
            lo = k;
        }
    }
    out
}

/// `<S²>` of isolated roots (Casida) or of TDA clusters.
fn compare_s2(label: &str, r: &TddftResult, refm: &Value, n: usize, tda: bool) -> f64 {
    let s2_ref = f64s(&refm["s2"]);
    let w_ref = f64s(&refm["omega_ev"]);
    let mut worst = 0.0_f64;
    for (range, gap) in clusters(&w_ref, n) {
        if !tda && range.len() > 1 {
            continue;
        }
        let ours: f64 = r.s2[range.clone()].iter().sum();
        let theirs: f64 = s2_ref[range.clone()].iter().sum();
        let bar = 8.0 * TOL_HA / gap * range.len() as f64;
        let d = (ours - theirs).abs();
        assert!(
            d <= bar,
            "{label} roots {range:?}: <S2> {ours} vs PySCF {theirs} (|d| {d:.2e}, bar {bar:.2e})"
        );
        worst = worst.max(d);
    }
    println!("  {label}: max |d<S2>| = {worst:.3e}");
    worst
}

/// Compare the lowest `N_CHECK` roots; returns max |dΩ| (eV).
fn compare(label: &str, r: &TddftResult, refm: &Value, tda: bool) -> f64 {
    let w_ref = f64s(&refm["omega_ev"]);
    let f_ref = f64s(&refm["osc"]);
    let n = N_CHECK.min(w_ref.len());
    assert!(r.excitation_energies.len() >= n, "{label}: too few roots");
    let w = ev(r);
    let dev = max_dev(&w, &w_ref, n);
    println!("  {label}: ferric {:?}", &w[..n]);
    println!("  {label}: pyscf  {:?}", &w_ref[..n]);
    println!("  {label}: <S2>   {:?}", &r.s2[..n]);
    println!("  {label}: max |dOmega| = {dev:.3e} eV");
    for k in 0..n {
        let lo = k == 0 || w_ref[k] - w_ref[k - 1] > ISOLATED_GAP_EV;
        let hi = k + 1 >= w_ref.len() || w_ref[k + 1] - w_ref[k] > ISOLATED_GAP_EV;
        if lo && hi {
            let df = (r.oscillator_strengths[k] - f_ref[k]).abs();
            assert!(
                df < OSC_ABS + OSC_REL * f_ref[k].abs(),
                "{label} state {k}: f {} vs PySCF {} (|df| {df:.2e})",
                r.oscillator_strengths[k],
                f_ref[k]
            );
        }
    }
    compare_s2(label, r, refm, n, tda);
    assert!(
        dev < TOL_EV,
        "{label}: max |dOmega| {dev:.3e} eV exceeds the {TOL_EV:.0e} eV bar"
    );
    dev
}

/// The reference's `key` roots must sit far from the true ones, and so must
/// ferric's.
fn negative_control(label: &str, tda: &TddftResult, case: &Value, key: &str) {
    let ctrl = f64s(&case[key]["omega_ev"]);
    let truth = f64s(&case["tda"]["omega_ev"]);
    let n = N_CHECK.min(ctrl.len());
    let ref_gap = max_dev(&truth, &ctrl, n);
    let fer_gap = max_dev(&ev(tda), &ctrl, n);
    println!("  {label} {key}: reference {ref_gap:.4} eV, ferric {fer_gap:.4} eV away");
    assert!(
        ref_gap > NEG_CONTROL_FACTOR * TOL_EV,
        "{label}: control {key} moves the roots by only {ref_gap:.2e} eV, too little to detect"
    );
    assert!(
        fer_gap > NEG_CONTROL_FACTOR * TOL_EV,
        "{label}: ferric sits within {fer_gap:.2e} eV of control {key}"
    );
}

fn check_reference(label: &str, s: &Setup, case: &Value, r: &TddftResult) {
    let e_ref = case["e_scf"].as_f64().unwrap();
    let de = (s.scf.energy - e_ref).abs();
    println!(
        "  {label}: E_SCF ferric {:.10} PySCF {e_ref:.10} (|d| {de:.1e})",
        s.scf.energy
    );
    assert!(
        de < SCF_TOL,
        "{label}: reference SCF energy differs by {de:e} Ha"
    );
    let s2_ref = case["s2_reference"].as_f64().unwrap();
    assert!(
        (r.s2_reference - s2_ref).abs() < 1e-6,
        "{label}: <S2>_0 {} vs PySCF {s2_ref}",
        r.s2_reference
    );
    for (name, ours) in [("c_sr", r.c_hf), ("c_lr", r.c_lr), ("omega", r.omega)] {
        let theirs = case[name].as_f64().unwrap();
        assert!(
            (ours - theirs).abs() < 1e-12,
            "{label}: {name} {ours} vs PySCF {theirs}"
        );
    }
    assert_eq!(
        r.spin, None,
        "{label}: an unrestricted root has no spin label"
    );
    assert_eq!(r.reference, ferric_scf::Spin::Unrestricted);
}

fn check_case(mol: &str, key: &str, xc: Option<&str>) {
    let refj = load_ref(mol);
    let case = &refj["cases"][key];
    assert!(case.is_object(), "{mol}: reference has no case {key}");
    let s = setup(&refj, xc);
    let mut worst = 0.0_f64;
    for solver in SOLVERS {
        let label = format!("{mol}/{BASIS}/{key} [{}]", solver.name());
        println!("\n== {label} ==");
        let tda = run(&s, xc, TddftMethod::Tda, solver).unwrap();
        check_reference(&label, &s, case, &tda);
        worst = worst.max(compare(&format!("{label} TDA"), &tda, &case["tda"], true));
        if case.get("tda_no_fxc_ab").is_some() {
            negative_control(&label, &tda, case, "tda_no_fxc_ab");
        }
        if case["casida"].get("skipped").is_some() {
            // OH: the zero mode of its degenerate ²Π ground state (see the
            // generator); TDA only.
            println!(
                "  {label} Casida: not compared ({})",
                case["casida"]["skipped"]
            );
            continue;
        }
        let cas = run(&s, xc, TddftMethod::Casida, solver);
        if case["casida"]["unstable"].as_bool().unwrap() {
            let w2 = case["casida"]["omega2_min"].as_f64().unwrap();
            let err = cas.expect_err("an unstable Casida problem must be refused");
            println!("  {label} Casida: reference Omega^2_min {w2:.3e}; ferric refuses: {err}");
        } else {
            let cas = cas.unwrap();
            worst = worst.max(compare(
                &format!("{label} Casida"),
                &cas,
                &case["casida"],
                false,
            ));
        }
    }
    println!("\n{mol}/{BASIS}/{key}: worst |dOmega| = {worst:.3e} eV (bar {TOL_EV:.0e})");
}

macro_rules! case {
    ($name:ident, $mol:literal, $key:literal, $xc:expr) => {
        #[test]
        #[ignore = "validation: open-shell TDDFT vs PySCF"]
        fn $name() {
            check_case($mol, $key, $xc);
        }
    };
}

const CAM: Option<&str> = Some("HYB_GGA_XC_CAM_B3LYP");

case!(oh_hf, "oh", "HF", None);

case!(nh2_hf, "nh2", "HF", None);
case!(nh2_lda, "nh2", "LDA", Some("LDA"));
case!(nh2_pbe, "nh2", "PBE", Some("PBE"));
case!(nh2_b3lyp, "nh2", "B3LYP", Some("B3LYP"));
case!(nh2_r2scan, "nh2", "r2SCAN", Some("r2SCAN"));
case!(nh2_tpssh, "nh2", "TPSSh", Some("TPSSh"));
case!(nh2_cam_b3lyp, "nh2", "HYB_GGA_XC_CAM_B3LYP", CAM);

case!(o2_hf, "o2", "HF", None);
case!(o2_lda, "o2", "LDA", Some("LDA"));
case!(o2_pbe, "o2", "PBE", Some("PBE"));
case!(o2_b3lyp, "o2", "B3LYP", Some("B3LYP"));
case!(o2_r2scan, "o2", "r2SCAN", Some("r2SCAN"));
case!(o2_tpssh, "o2", "TPSSh", Some("TPSSh"));
case!(o2_cam_b3lyp, "o2", "HYB_GGA_XC_CAM_B3LYP", CAM);
