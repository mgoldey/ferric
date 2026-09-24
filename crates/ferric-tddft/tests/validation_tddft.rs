//! Validation row V-TDDFT (campaign defect F1): the user-facing `run_tddft`
//! (CLI `tda`/`tddft`, Python `run_tddft`) against PySCF `TDA` / `TDDFT`.
//!
//! References: `testdata/reference/validation/tddft/<mol>_<basis>.json`, from
//! `scripts/validation/gen_tddft_refs.py`. Every case is `#[ignore]`d — run
//! with `--ignored` (see the command in the generator's docstring and below).
//!
//! # What the reference matches, and so what the tolerance is
//!
//! The generator reproduces ferric's integral treatment rather than leaving it
//! as a residual: ferric's own bundled basis JSON, RI-JK (def2-universal-jkfit)
//! reference SCF, the (75, 110) unpruned Becke grid, identical libxc
//! identifiers, and — the part that dominated the older library comparison —
//! the response `(pq|rs)` built by the SAME Coulomb-metric RI over the SAME aux
//! (`get_ab` with `ao2mo.general` replaced). So the RI error that set the
//! 3e-3 / 3e-2 eV bars in `ferric-gw/tests/tda_dft_vs_pyscf.rs` cancels here,
//! and what remains is SCF convergence and grid round-off.
//!
//! ASSUMPTION (to be confirmed by the first run, then tightened): the residual
//! is ≪ 1e-3 eV, so the campaign's 1e-3 eV bar applies unchanged. The printed
//! max deviation per case is the measurement; if a case sits near the bar,
//! that is a finding (e.g. a grid/radial-scheme mismatch), not slack to add.
//!
//! # State matching
//!
//! Both sides solve the dense eigenproblem, so every state is present on both
//! sides and SORTED energies are compared: a missing or extra root would show
//! as an O(0.1 eV) deviation, not hide. Near-degenerate pairs (e.g. NH3's E
//! states) are therefore fine for ENERGIES. Oscillator strengths are only
//! compared for states separated from both neighbours by more than
//! `ISOLATED_GAP_EV`, because inside a (near-)degenerate pair the individual
//! `f` depends on an arbitrary rotation of the pair.
//!
//! # Teeth (each assertion here has a defect it catches)
//!
//! * `(b)` negative control, every KS case: the reference's own kernel-LESS
//!   energies (`tda_nofxc`, what `run_tddft` returned before F1) must differ
//!   from the kernel-full ones by ≫ the tolerance (the pass condition is
//!   reachable, i.e. this comparison CAN tell the two apart), and ferric's
//!   answer must be far from the kernel-less one. If f_xc is dropped again the
//!   energy comparison fails AND this does.
//! * `(c)` Casida vs TDA must differ (lowest root: Casida strictly below TDA by
//!   more than the tolerance, on both sides) — catches a B matrix that is
//!   silently zero, or the Casida arm falling through to TDA.
//! * `(a)` anchor: `run_tddft` TDA equals `ferric_gw::tddft::run_tda_dft` (the
//!   PySCF-validated library path; at HF, bit-identical to CIS per its own
//!   anchor) — for HF, PBE and B3LYP — to 1e-8 Ha. Both use the shared
//!   `ferric_dft::lr_kernel` block, so this pins the A-matrix assembly and
//!   c_HF resolution of THIS driver against the validated one, independent of
//!   PySCF.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-tddft --test validation_tddft \
//!     -- --ignored --nocapture --test-threads=1
//! ```

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ferric_tddft::{run_tddft, TddftConfig, TddftMethod, TddftResult};
use serde_json::Value;

const HARTREE_TO_EV: f64 = 27.211_386_245_988;
/// Excitation-energy bar (campaign §5.4 row V-TDDFT). See the module docs for
/// why the RI-matched reference makes this the right order of magnitude.
const TOL_EV: f64 = 1e-3;
/// How many of the lowest roots are compared.
const N_CHECK: usize = 5;
/// A state is "isolated" (its oscillator strength is comparable) when both
/// neighbouring roots are at least this far away.
const ISOLATED_GAP_EV: f64 = 1e-2;
/// Oscillator strengths: absolute + relative bar. With RI-matched integrals
/// they should agree far better; this is loose enough not to fail on grid
/// round-off yet catches a missing sqrt(2) (a factor of 2) or an un-normalized
/// Casida (X+Y) (an O(1) factor).
const OSC_ABS: f64 = 1e-3;
const OSC_REL: f64 = 1e-2;
/// The kernel-less answer must sit at least this many tolerances away from the
/// kernel-full one for the negative control to mean anything.
const NEG_CONTROL_FACTOR: f64 = 10.0;
const SCF_JK_AUX: &str = "def2-universal-jkfit";

fn load_ref(mol: &str, basis: &str) -> Value {
    let path = format!(
        "{}/../../testdata/reference/validation/tddft/{mol}_{basis}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let txt = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing reference {path}: {e}\n  generate it with \
             `OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/validation/gen_tddft_refs.py`"
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

/// The reference SCF exactly as the CLI/Python `run_tddft` builds it: RI-J/K
/// with def2-universal-jkfit, default (75, 110) grid, the named functional.
fn setup(refj: &Value, xc: Option<&str>) -> Setup {
    let xyz = format!(
        "{}/../../{}",
        env!("CARGO_MANIFEST_DIR"),
        refj["xyz"].as_str().unwrap()
    );
    let mol = Molecule::load_xyz(&xyz).unwrap();
    let obs_name = refj["basis"].as_str().unwrap();
    let aux_name = refj["response_aux"].as_str().unwrap();
    assert_eq!(refj["scf_jk_aux"].as_str().unwrap(), SCF_JK_AUX);
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    // density_conv 1e-8: far below what moves an excitation energy by 1e-6 eV;
    // energy_conv stays at its default (a sanity bound, not a target).
    let cfg = RhfConfig {
        xc: xc.map(str::to_string),
        df_j_aux: Some(SCF_JK_AUX.to_string()),
        df_k_aux: Some(SCF_JK_AUX.to_string()),
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let scf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(
        scf.converged,
        "{obs_name}/{xc:?}: reference SCF did not converge"
    );
    Setup {
        mol,
        obs,
        dfbs,
        scf,
    }
}

fn run(s: &Setup, xc: Option<&str>, method: TddftMethod, n_roots: usize) -> TddftResult {
    let cfg = TddftConfig {
        n_roots,
        method,
        xc: xc.map(str::to_string),
        ..Default::default()
    };
    run_tddft(&s.mol, &s.obs, &s.dfbs, &s.scf, &cfg).unwrap()
}

/// Compare one method's lowest `N_CHECK` roots; returns max |dΩ| (eV).
fn compare(label: &str, r: &TddftResult, refm: &Value) -> f64 {
    let w_ref = f64s(&refm["omega_ev"]);
    let f_ref = f64s(&refm["osc"]);
    let n = N_CHECK.min(w_ref.len());
    assert!(
        r.excitation_energies.len() >= n,
        "{label}: ferric returned {} roots, need {n}",
        r.excitation_energies.len()
    );
    let w_fer: Vec<f64> = r
        .excitation_energies
        .iter()
        .map(|w| w * HARTREE_TO_EV)
        .collect();

    println!(
        "  {label}\n  {:>3} {:>11} {:>11} {:>9}  {:>9} {:>9}",
        "st", "ferric eV", "pyscf eV", "dev eV", "f_fer", "f_py"
    );
    let mut max_dev = 0.0_f64;
    for k in 0..n {
        let dev = (w_fer[k] - w_ref[k]).abs();
        max_dev = max_dev.max(dev);
        let gap_lo = if k == 0 {
            f64::INFINITY
        } else {
            w_ref[k] - w_ref[k - 1]
        };
        let gap_hi = w_ref
            .get(k + 1)
            .map_or(f64::INFINITY, |next| next - w_ref[k]);
        let isolated = gap_lo > ISOLATED_GAP_EV && gap_hi > ISOLATED_GAP_EV;
        let f_f = r.oscillator_strengths[k];
        println!(
            "  {k:>3} {:>11.6} {:>11.6} {dev:>9.2e}  {f_f:>9.5} {:>9.5}{}",
            w_fer[k],
            w_ref[k],
            f_ref[k],
            if isolated {
                ""
            } else {
                "  (near-degenerate: f not compared)"
            }
        );
        if isolated {
            let df = (f_f - f_ref[k]).abs();
            assert!(
                df < OSC_ABS + OSC_REL * f_ref[k].abs(),
                "{label} state {k}: oscillator strength {f_f:.6} vs PySCF {:.6} \
                 (|df| = {df:.2e})",
                f_ref[k]
            );
        }
    }
    println!("  {label}: max |dOmega| = {max_dev:.3e} eV over {n} roots");
    assert!(
        max_dev < TOL_EV,
        "{label}: max excitation-energy deviation {max_dev:.3e} eV exceeds {TOL_EV:.0e} eV"
    );
    max_dev
}

/// Full check of one (molecule, basis, functional) case.
///
/// `ferric_xc` is ferric's name (`None` = HF); `key` is the reference's case
/// key (`"HF"`, `"LDA"`, `"PBE"`, `"B3LYP"`).
fn check_case(mol: &str, basis: &str, key: &str, ferric_xc: Option<&str>) {
    let refj = load_ref(mol, basis);
    let case = &refj["cases"][key];
    assert!(
        case.is_object(),
        "{mol}/{basis}: reference has no case {key}"
    );
    let s = setup(&refj, ferric_xc);
    let label = format!("{mol}/{basis}/{key}");
    println!("\n== {label} ==");

    let tda = run(&s, ferric_xc, TddftMethod::Tda, N_CHECK);
    let cas = run(&s, ferric_xc, TddftMethod::Casida, N_CHECK);

    // Bookkeeping that must match before any energy is meaningful.
    let c_hf_ref = case["c_hf"].as_f64().unwrap();
    assert!(
        (tda.c_hf - c_hf_ref).abs() < 1e-12,
        "{label}: c_HF {} vs PySCF hybrid coefficient {c_hf_ref}",
        tda.c_hf
    );
    assert_eq!(
        tda.fxc_included,
        ferric_xc.is_some(),
        "{label}: f_xc must be included exactly for a KS reference"
    );
    assert_eq!(cas.fxc_included, ferric_xc.is_some(), "{label} (Casida)");

    compare(&format!("{label} TDA"), &tda, &case["tda"]);
    compare(&format!("{label} Casida"), &cas, &case["casida"]);

    // (c) Casida must differ from TDA, on both sides. For a stable reference
    // the lowest Casida root lies BELOW the lowest TDA root.
    let tda0 = tda.excitation_energies[0] * HARTREE_TO_EV;
    let cas0 = cas.excitation_energies[0] * HARTREE_TO_EV;
    let ref_tda0 = f64s(&case["tda"]["omega_ev"])[0];
    let ref_cas0 = f64s(&case["casida"]["omega_ev"])[0];
    assert!(
        ref_tda0 - ref_cas0 > TOL_EV,
        "{label}: reference Casida/TDA split {:.2e} eV is below the tolerance — \
         this case cannot tell the two methods apart",
        ref_tda0 - ref_cas0
    );
    assert!(
        tda0 - cas0 > TOL_EV,
        "{label}: ferric Casida ({cas0:.6} eV) is not below TDA ({tda0:.6} eV) by more \
         than {TOL_EV:.0e} eV — B is missing or the Casida arm fell through"
    );

    // (b) Negative control: the kernel-less answer (pre-F1 behaviour). Taken
    // over the N_CHECK lowest roots (sorted on both sides), because a single
    // diffuse Rydberg-like S1 can carry a small f_xc shift while the valence
    // roots above it carry a large one.
    if ferric_xc.is_some() {
        let nofxc = f64s(&case["tda_nofxc"]["omega_ev"]);
        let ref_tda = f64s(&case["tda"]["omega_ev"]);
        let n = N_CHECK.min(nofxc.len()).min(ref_tda.len());
        let max_gap = |w: &[f64]| {
            (0..n)
                .map(|k| (w[k] - nofxc[k]).abs())
                .fold(0.0_f64, f64::max)
        };
        let ref_gap = max_gap(ref_tda.as_slice());
        assert!(
            ref_gap > NEG_CONTROL_FACTOR * TOL_EV,
            "{label}: the f_xc term moves the lowest {n} roots by at most {ref_gap:.2e} eV \
             in the reference — too small for this case to detect a dropped kernel"
        );
        let fer: Vec<f64> = tda
            .excitation_energies
            .iter()
            .map(|w| w * HARTREE_TO_EV)
            .collect();
        let fer_gap = max_gap(fer.as_slice());
        assert!(
            fer_gap > NEG_CONTROL_FACTOR * TOL_EV,
            "{label}: ferric's lowest {n} roots sit within {fer_gap:.2e} eV of the \
             KERNEL-LESS reference — the f_xc term has been dropped again (F1)"
        );
        println!(
            "  {label}: f_xc moves the lowest {n} roots by up to {ref_gap:.4} eV \
             (reference); ferric is up to {fer_gap:.4} eV from the kernel-less values"
        );
    }
}

macro_rules! validation_case {
    ($name:ident, $mol:literal, $basis:literal, $key:literal, $xc:expr) => {
        #[test]
        #[ignore = "validation: TDA-DFT / TDDFT"]
        fn $name() {
            check_case($mol, $basis, $key, $xc);
        }
    };
}

validation_case!(water_631g_hf, "water", "6-31g", "HF", None);
validation_case!(water_631g_lda, "water", "6-31g", "LDA", Some("LDA"));
validation_case!(water_631g_pbe, "water", "6-31g", "PBE", Some("PBE"));
validation_case!(water_631g_b3lyp, "water", "6-31g", "B3LYP", Some("B3LYP"));
validation_case!(water_augdz_hf, "water", "aug-cc-pvdz", "HF", None);
validation_case!(water_augdz_lda, "water", "aug-cc-pvdz", "LDA", Some("LDA"));
validation_case!(water_augdz_pbe, "water", "aug-cc-pvdz", "PBE", Some("PBE"));
validation_case!(
    water_augdz_b3lyp,
    "water",
    "aug-cc-pvdz",
    "B3LYP",
    Some("B3LYP")
);

validation_case!(h2co_631g_hf, "formaldehyde", "6-31g", "HF", None);
validation_case!(h2co_631g_lda, "formaldehyde", "6-31g", "LDA", Some("LDA"));
validation_case!(h2co_631g_pbe, "formaldehyde", "6-31g", "PBE", Some("PBE"));
validation_case!(
    h2co_631g_b3lyp,
    "formaldehyde",
    "6-31g",
    "B3LYP",
    Some("B3LYP")
);
validation_case!(h2co_augdz_hf, "formaldehyde", "aug-cc-pvdz", "HF", None);
validation_case!(
    h2co_augdz_lda,
    "formaldehyde",
    "aug-cc-pvdz",
    "LDA",
    Some("LDA")
);
validation_case!(
    h2co_augdz_pbe,
    "formaldehyde",
    "aug-cc-pvdz",
    "PBE",
    Some("PBE")
);
validation_case!(
    h2co_augdz_b3lyp,
    "formaldehyde",
    "aug-cc-pvdz",
    "B3LYP",
    Some("B3LYP")
);

validation_case!(nh3_631g_hf, "nh3", "6-31g", "HF", None);
validation_case!(nh3_631g_lda, "nh3", "6-31g", "LDA", Some("LDA"));
validation_case!(nh3_631g_pbe, "nh3", "6-31g", "PBE", Some("PBE"));
validation_case!(nh3_631g_b3lyp, "nh3", "6-31g", "B3LYP", Some("B3LYP"));
validation_case!(nh3_augdz_hf, "nh3", "aug-cc-pvdz", "HF", None);
validation_case!(nh3_augdz_lda, "nh3", "aug-cc-pvdz", "LDA", Some("LDA"));
validation_case!(nh3_augdz_pbe, "nh3", "aug-cc-pvdz", "PBE", Some("PBE"));
validation_case!(
    nh3_augdz_b3lyp,
    "nh3",
    "aug-cc-pvdz",
    "B3LYP",
    Some("B3LYP")
);

/// (a) Anchor against the PySCF-validated library path, independent of any
/// external reference: `run_tddft` TDA must reproduce
/// `ferric_gw::tddft::run_tda_dft` (same reference, same RI aux, same default
/// grid, shared f_xc block). At HF that path is bit-identical to `run_cis_tda`
/// (its own anchor), so this is also the CIS check.
///
/// The two drivers build their RI tensors through different code (`build_b_tensors`
/// here, `mo_b::build_full_b` there), so agreement is to round-off, not bits:
/// 1e-8 Ha is ~3e-7 eV, far below anything physical and far above round-off.
#[test]
#[ignore = "validation: TDA-DFT / TDDFT"]
fn tda_matches_the_validated_library_tda_path() {
    use ferric_gw::tddft::{run_tda_dft, TdaDftConfig};
    let refj = load_ref("water", "6-31g");
    for xc in [None, Some("PBE"), Some("B3LYP")] {
        let s = setup(&refj, xc);
        let ours = run(&s, xc, TddftMethod::Tda, 10);
        let lib = run_tda_dft(
            &s.mol,
            &s.obs,
            &s.dfbs,
            Operator::coulomb(),
            &s.scf,
            xc,
            &TdaDftConfig::default(),
        )
        .unwrap();
        assert_eq!(lib.fxc_included, xc.is_some());
        assert!((lib.c_hf - ours.c_hf).abs() < 1e-15, "{xc:?}: c_HF differs");
        let mut max_d = 0.0_f64;
        for (k, w) in ours.excitation_energies.iter().enumerate() {
            let d = (w - lib.omega[k]).abs();
            max_d = max_d.max(d);
        }
        println!("{xc:?}: run_tddft vs run_tda_dft max |dOmega| = {max_d:.2e} Ha");
        assert!(
            max_d < 1e-8,
            "{xc:?}: user-facing TDA disagrees with the validated library TDA by {max_d:.2e} Ha"
        );
        // Oscillator strengths share the PySCF convention (sqrt(2) factor).
        for k in 0..ours.oscillator_strengths.len() {
            let d = (ours.oscillator_strengths[k] - lib.oscillator_strength[k]).abs();
            // Degenerate pairs can rotate; only compare where the library's
            // neighbours are well separated.
            let lo = k == 0 || lib.omega[k] - lib.omega[k - 1] > 1e-4;
            let hi = k + 1 >= lib.omega.len() || lib.omega[k + 1] - lib.omega[k] > 1e-4;
            if lo && hi {
                assert!(
                    d < 1e-6,
                    "{xc:?} state {k}: f {} vs library {}",
                    ours.oscillator_strengths[k],
                    lib.oscillator_strength[k]
                );
            }
        }
    }
}
