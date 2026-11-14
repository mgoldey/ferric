//! Range-separated hybrids and triplets in `run_tddft` against PySCF.
//!
//! References: `testdata/reference/validation/tddft_rsh/<mol>_6-31g.json`, from
//! `scripts/validation/gen_tddft_rsh_refs.py`. Every case is `#[ignore]`d:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-tddft --test validation_tddft_rsh \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! # What is matched, and so what the bar is
//!
//! The generator reproduces ferric's integrals instead of leaving them in the
//! residual:
//!
//! * the bundled basis JSON, identical libxc identifiers, and the flat
//!   (75, 110) Becke grid with no pruning on both sides (ferric's SCF and f_xc
//!   run on it here), so grid error cancels;
//! * the reference SCF: RI-J/K over def2-universal-jkfit with J fitted in
//!   f64, and for a range-separated functional the exchange fitted as
//!   c_sr K[erfc] + c_lr K[erf], each in its own metric, as ferric fits it;
//! * the response integrals: (pq|rs) by ferric's Coulomb-metric RI and
//!   (pq|rs)_ω by ferric's erf-metric RI (eigenvalues below 1e-10 dropped),
//!   both over the response aux.
//!
//! The RI error therefore cancels. What is left is SCF convergence: ferric
//! converges the density to 1e-8 and PySCF the orbital gradient to 1e-7, so
//! each code's orbitals are off by O(1e-7) and an excitation energy, first
//! order in them, by a few 1e-6 eV. The Davidson solver adds its residual
//! bar, below 1e-6 eV here (`davidson_roots_equal_the_dense_roots`). The bar
//! is `TOL_EV = 2e-5` eV, the same as the existing same-grid comparison
//! (`validation_tddft.rs`, `SAME_GRID_BAR_EV`, measured there at 5.7e-7 to
//! 2.8e-6 eV). Each case prints its measured maximum.
//!
//! # Teeth
//!
//! * Long range: for each range-separated case the reference's roots with the
//!   long-range exchange dropped (`tda_no_lr`) sit far outside the bar, and
//!   ferric must too. Dropping the erf term fails this AND the energies.
//! * Triplet kernel sign: for each KS triplet the reference's roots with
//!   f_αα + f_αβ in place of f_αα − f_αβ (`tda_singlet_kernel`) sit far
//!   outside the bar, and ferric must too.
//! * Coefficients: c_sr, c_lr and ω must equal PySCF's
//!   `rsh_and_hybrid_coeff`.
//! * Triplet oscillator strengths are exactly zero.
//! * Instability: a triplet Casida problem with Ω² < 0 in the reference must
//!   be refused by ferric, with both solvers.
//!
//! Mutations (applied and reverted, 2026-11-14): the long-range exchange
//! dropped (`ExchangeMix::long_range` returning `None`) fails
//! `water_cam_b3lyp` at 3.39 eV; the dense triplet kernel fed δD_β = +δD
//! fails `water_pbe` dense triplet TDA at 0.144 eV; the matrix-free triplet γ
//! with the singlet sign fails `water_pbe` Davidson triplet TDA at 0.136 eV
//! (the dense arm still passes, so each solver's triplet route is pinned).
//!
//! Water and C2H4 run through both `TddftSolver::Dense` and
//! `TddftSolver::Davidson` against the same bar; benzene through Davidson
//! only (see [`DAVIDSON`]).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ferric_tddft::{run_tddft, ResponseSpin, TddftConfig, TddftMethod, TddftResult, TddftSolver};
use serde_json::Value;

const HARTREE_TO_EV: f64 = 27.211_386_245_988;
/// Excitation-energy bar; derived in the module docs.
const TOL_EV: f64 = 2e-5;
/// How many of the lowest roots are compared.
const N_CHECK: usize = 5;
/// A state's oscillator strength is compared only when both neighbours are
/// at least this far away (inside a near-degenerate pair f depends on an
/// arbitrary rotation).
const ISOLATED_GAP_EV: f64 = 1e-2;
const OSC_ABS: f64 = 1e-4;
const OSC_REL: f64 = 1e-3;
/// A negative control must sit at least this many bars from the reference.
const NEG_CONTROL_FACTOR: f64 = 100.0;
const SCF_JK_AUX: &str = "def2-universal-jkfit";
const BASIS: &str = "6-31g";
const SOLVERS: [TddftSolver; 2] = [TddftSolver::Dense, TddftSolver::Davidson];
const SPINS: [ResponseSpin; 2] = [ResponseSpin::Singlet, ResponseSpin::Triplet];

fn flat_grid() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 75,
        n_angular: 110,
        ..Default::default()
    }
}

fn load_ref(mol: &str) -> Value {
    let path = format!(
        "{}/../../testdata/reference/validation/tddft_rsh/{mol}_{BASIS}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let txt = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing reference {path}: {e}\n  generate it with `OPENBLAS_NUM_THREADS=1 uv run \
             --no-sync python scripts/validation/gen_tddft_rsh_refs.py`"
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

/// The reference SCF on the flat grid, RI-J/K as the CLI builds it, J in f64.
fn setup(refj: &Value, xc: Option<&str>) -> Setup {
    let xyz = format!(
        "{}/../../{}",
        env!("CARGO_MANIFEST_DIR"),
        refj["xyz"].as_str().unwrap()
    );
    let mol = Molecule::load_xyz(&xyz).unwrap();
    assert_eq!(refj["basis"].as_str().unwrap(), BASIS);
    assert_eq!(refj["scf_jk_aux"].as_str().unwrap(), SCF_JK_AUX);
    let aux_name = refj["response_aux"].as_str().unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(BASIS).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        xc: xc.map(str::to_string),
        dft_grid: Some(flat_grid()),
        df_j_aux: Some(SCF_JK_AUX.to_string()),
        df_k_aux: Some(SCF_JK_AUX.to_string()),
        ri_j_precision: Some(ferric_scf::df_j::RiJPrecision::F64),
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let scf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
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
    spin: ResponseSpin,
    solver: TddftSolver,
) -> Result<TddftResult, ferric_core::error::FerricError> {
    let cfg = TddftConfig {
        n_roots: N_CHECK,
        method,
        grid: flat_grid(),
        xc: xc.map(str::to_string),
        solver,
        spin,
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

/// Compare the lowest `N_CHECK` roots; returns max |dΩ| (eV).
fn compare(label: &str, r: &TddftResult, refm: &Value, spin: ResponseSpin) -> f64 {
    let w_ref = f64s(&refm["omega_ev"]);
    let f_ref = f64s(&refm["osc"]);
    let n = N_CHECK.min(w_ref.len());
    assert!(r.excitation_energies.len() >= n, "{label}: too few roots");
    let w = ev(r);
    let dev = max_dev(&w, &w_ref, n);
    println!("  {label}: ferric {:?}", &w[..n]);
    println!("  {label}: pyscf  {:?}", &w_ref[..n]);
    println!("  {label}: max |dOmega| = {dev:.3e} eV");
    if spin == ResponseSpin::Triplet {
        assert!(
            r.oscillator_strengths.iter().all(|&f| f == 0.0),
            "{label}: a triplet has no transition dipole, f must be exactly 0"
        );
    } else {
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
    }
    assert!(
        dev < TOL_EV,
        "{label}: max |dOmega| {dev:.3e} eV exceeds the {TOL_EV:.0e} eV bar"
    );
    dev
}

/// A negative control: the reference's `key` roots must sit far from the true
/// ones, and so must ferric's. Returns (reference gap, ferric gap) in eV.
fn negative_control(label: &str, tda: &TddftResult, rec: &Value, key: &str) {
    let ctrl = f64s(&rec[key]["omega_ev"]);
    let truth = f64s(&rec["tda"]["omega_ev"]);
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

fn check_case(mol: &str, key: &str, xc: Option<&str>, solvers: &[TddftSolver]) {
    let refj = load_ref(mol);
    let case = &refj["cases"][key];
    assert!(case.is_object(), "{mol}: reference has no case {key}");
    let s = setup(&refj, xc);
    let mut worst = 0.0_f64;
    for (solver, spin) in solvers.iter().flat_map(|&v| SPINS.map(|sp| (v, sp))) {
        let label = format!("{mol}/{BASIS}/{key} {} [{}]", spin.name(), solver.name());
        println!("\n== {label} ==");
        let rec = &case[spin.name()];
        let tda = run(&s, xc, TddftMethod::Tda, spin, solver).unwrap();
        assert_eq!(tda.spin, Some(spin));
        for (name, ours) in [("c_sr", tda.c_hf), ("c_lr", tda.c_lr), ("omega", tda.omega)] {
            let theirs = case[name].as_f64().unwrap();
            assert!(
                (ours - theirs).abs() < 1e-12,
                "{label}: {name} {ours} vs PySCF {theirs}"
            );
        }
        worst = worst.max(compare(&format!("{label} TDA"), &tda, &rec["tda"], spin));
        if rec.get("tda_no_lr").is_some() {
            negative_control(&label, &tda, rec, "tda_no_lr");
        }
        if rec.get("tda_singlet_kernel").is_some() {
            negative_control(&label, &tda, rec, "tda_singlet_kernel");
        }

        let cas = run(&s, xc, TddftMethod::Casida, spin, solver);
        if rec["casida"]["unstable"].as_bool().unwrap() {
            let w2 = rec["casida"]["omega2_min"].as_f64().unwrap();
            let err = cas.expect_err("an unstable Casida problem must be refused");
            println!("  {label} Casida: reference Omega^2_min {w2:.3e}; ferric refuses: {err}");
        } else {
            let cas = cas.unwrap();
            worst = worst.max(compare(
                &format!("{label} Casida"),
                &cas,
                &rec["casida"],
                spin,
            ));
        }
    }
    println!("\n{mol}/{BASIS}/{key}: worst |dOmega| = {worst:.3e} eV (bar {TOL_EV:.0e})");
}

macro_rules! case {
    ($name:ident, $mol:literal, $key:literal, $xc:expr) => {
        case!($name, $mol, $key, $xc, &SOLVERS);
    };
    ($name:ident, $mol:literal, $key:literal, $xc:expr, $solvers:expr) => {
        #[test]
        #[ignore = "validation: range-separated and triplet TDDFT vs PySCF"]
        fn $name() {
            check_case($mol, $key, $xc, $solvers);
        }
    };
}

/// Benzene runs through the Davidson solver only. The dense solver's f_xc
/// block costs one kernel application per (jb) column (dim = 945 here),
/// and the dense and Davidson roots are already pinned to each other
/// (`davidson_roots_equal_the_dense_roots`) and to PySCF on water and C2H4.
const DAVIDSON: &[TddftSolver] = &[TddftSolver::Davidson];

const CAM: Option<&str> = Some("HYB_GGA_XC_CAM_B3LYP");
const WB97X: Option<&str> = Some("HYB_GGA_XC_WB97X");

case!(water_hf, "water", "HF", None);
case!(water_lda, "water", "LDA", Some("LDA"));
case!(water_pbe, "water", "PBE", Some("PBE"));
case!(water_b3lyp, "water", "B3LYP", Some("B3LYP"));
case!(water_r2scan, "water", "r2SCAN", Some("r2SCAN"));
case!(water_cam_b3lyp, "water", "HYB_GGA_XC_CAM_B3LYP", CAM);
case!(water_wb97x, "water", "HYB_GGA_XC_WB97X", WB97X);

case!(c2h4_hf, "c2h4", "HF", None);
case!(c2h4_lda, "c2h4", "LDA", Some("LDA"));
case!(c2h4_pbe, "c2h4", "PBE", Some("PBE"));
case!(c2h4_b3lyp, "c2h4", "B3LYP", Some("B3LYP"));
case!(c2h4_r2scan, "c2h4", "r2SCAN", Some("r2SCAN"));
case!(c2h4_cam_b3lyp, "c2h4", "HYB_GGA_XC_CAM_B3LYP", CAM);
case!(c2h4_wb97x, "c2h4", "HYB_GGA_XC_WB97X", WB97X);

case!(benzene_hf, "benzene", "HF", None, DAVIDSON);
case!(benzene_lda, "benzene", "LDA", Some("LDA"), DAVIDSON);
case!(benzene_pbe, "benzene", "PBE", Some("PBE"), DAVIDSON);
case!(
    benzene_r2scan,
    "benzene",
    "r2SCAN",
    Some("r2SCAN"),
    DAVIDSON
);
case!(benzene_b3lyp, "benzene", "B3LYP", Some("B3LYP"), DAVIDSON);
case!(
    benzene_cam_b3lyp,
    "benzene",
    "HYB_GGA_XC_CAM_B3LYP",
    CAM,
    DAVIDSON
);
case!(
    benzene_wb97x,
    "benzene",
    "HYB_GGA_XC_WB97X",
    WB97X,
    DAVIDSON
);
