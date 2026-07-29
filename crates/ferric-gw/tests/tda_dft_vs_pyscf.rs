//! TDA-DFT excitation energies vs PySCF `tddft.TDA`.
//!
//! This is the half of the validation the CIS anchor cannot cover. The anchor
//! (`tda_dft_cis_anchor.rs`) pins the A-matrix ASSEMBLY by switching the f_xc
//! term off; it says nothing about whether the AO→(ia) f_xc adapter — the one
//! genuinely new piece of physics — is right. That is what this file tests.
//!
//! # State matching is by CHARACTER, not by energy order
//!
//! `docs/VALIDATION.md` records that BSE-TDA's reported MAE is only a LOWER
//! BOUND because naive nearest-energy matching assigned a bright π→π* to a dark
//! root and double-assigned three N-heterocycle states. That failure mode is
//! avoided here: each PySCF state carries its dominant `(i, a)` pair, and we
//! match on that, then check the energies of the matched pairs. A test that
//! sorted both lists and zipped them could report a small MAE while comparing
//! physically different states.
//!
//! Oscillator strengths are compared too, at a looser tolerance — they are far
//! more sensitive to basis/grid details than the energies, so a tight bound
//! would produce noise failures rather than signal.
//!
//! # Why the energy tolerance is 3e-3 eV and not tighter
//!
//! ferric evaluates the `(ia|jb)` Coulomb coupling through the RI/density-fitted
//! 3-index path; PySCF's `tddft.TDA` uses exact 4-index integrals. So a residual
//! offset is EXPECTED and is not implementation error. The measured signature
//! confirms that reading rather than assuming it — on HF/STO-3G every state
//! deviates in the SAME direction (ferric low) with a tight relative spread:
//!
//! ```text
//!   state   dev (eV)    relative
//!     0     -1.33e-3    -1.01e-4
//!     1     -1.06e-3    -7.01e-5
//!     2     -1.30e-3    -7.76e-5
//!     3     -1.12e-3    -5.82e-5
//!     4     -1.24e-3    -5.61e-5
//!     5     -1.45e-3    -4.99e-5
//!   mean relative deviation -6.9e-5, stdev 1.9e-5
//! ```
//!
//! Uniform sign and a tight relative spread across six states of different
//! character is one systematic mechanism, not scatter. The same ~1.0-1.5e-3 eV
//! band appears for LDA and PBE, i.e. it does not depend on the functional —
//! which is what an RI-on-the-Coulomb-term explanation predicts and what a
//! broken f_xc adapter would NOT produce.
//!
//! 3e-3 eV therefore clears the measured RI floor with ~2x margin while still
//! catching a real defect: the GGA symmetrization bug this suite found showed up
//! as a 3.1e-2 RELATIVE (ia)-block asymmetry, and a wrong kernel term moves
//! excitation energies by tenths of an eV, not thousandths.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::tddft::{run_tda_dft, TdaDftConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const WATER: &str = "3\nwater\nO 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692\n";
const HARTREE_TO_EV: f64 = 27.211_386_245_988;

struct PyState {
    omega_ev: f64,
    osc_length: f64,
    /// `[i, a]` of the largest-|X| amplitude, in PySCF's LOCAL active indexing.
    dominant_ia: [usize; 2],
}

struct PyCase {
    basis: String,
    nocc: usize,
    nvir: usize,
    states: Vec<PyState>,
}

/// Hand-parsed rather than serde-derived: `ferric-gw` carries `serde_json` as a
/// dev-dependency but not `serde` with the derive feature, and a test is not a
/// good reason to add one to the crate graph.
fn load(key: &str) -> PyCase {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/water_tda_dft_pyscf.json"
    );
    let txt = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("missing PySCF TDA reference {path}: {e}"));
    let all: serde_json::Value = serde_json::from_str(&txt).unwrap();
    let c = all
        .get(key)
        .unwrap_or_else(|| panic!("reference has no case {key}"));
    let states = c["states"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let d = s["dominant_ia"].as_array().unwrap();
            PyState {
                omega_ev: s["omega_ev"].as_f64().unwrap(),
                osc_length: s["osc_length"].as_f64().unwrap(),
                dominant_ia: [
                    d[0].as_u64().unwrap() as usize,
                    d[1].as_u64().unwrap() as usize,
                ],
            }
        })
        .collect();
    PyCase {
        basis: c["basis"].as_str().unwrap().to_string(),
        nocc: c["nocc"].as_u64().unwrap() as usize,
        nvir: c["nvir"].as_u64().unwrap() as usize,
        states,
    }
}

/// ferric's dominant (i, a) for state `n`, from the retained eigenvectors.
fn dominant_ia(x: &ndarray::Array2<f64>, n: usize, nvir: usize) -> (usize, usize) {
    let col = x.column(n);
    let mut best = 0usize;
    let mut best_w = -1.0f64;
    for (ia, &v) in col.iter().enumerate() {
        if v.abs() > best_w {
            best_w = v.abs();
            best = ia;
        }
    }
    (best / nvir, best % nvir)
}

/// Run ferric TDA-DFT for one functional and compare against PySCF.
///
/// `xc` is ferric's functional name; `key` selects the PySCF case. `tol_ev` is
/// the excitation-energy bound; `n_check` limits how many of the lowest states
/// are compared (the high-lying ones are basis-set artifacts on a minimal basis
/// and are not physically meaningful to bound tightly).
fn compare(xc: Option<&str>, key: &str, obs_name: &str, tol_ev: f64, n_check: usize) {
    let py = load(key);
    assert_eq!(py.basis, obs_name, "reference basis mismatch for {key}");

    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    // cc-pvdz-ri throughout. NOTE this aux is undersized for EXCITATION
    // energies at cc-pVDZ: an aux sweep on water/cc-pVDZ/PBE moves the lowest
    // state 7.3472 (cc-pvdz-ri, naux=84) -> 7.3670 (aug-cc-pvdz-rifit, 118)
    // -> 7.3676 (def2-universal-jkfit, 113), i.e. the two larger sets agree
    // with each other to 6e-4 eV while cc-pvdz-ri sits ~2e-2 eV away. The
    // series converges, so this is genuine RI fitting error, not a bug -- but
    // it is why the cc-pVDZ case carries a looser bound than the STO-3G ones.
    let aux = "cc-pvdz-ri";
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux).unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    // NOTE the SCF thresholds are looser for KS than for HF. A KS density
    // cannot converge past the XC grid's own noise floor, so demanding
    // density_conv = 1e-9 makes PBE/B3LYP fail to converge on this grid while
    // HF converges fine (cf. the `energy_conv-is-a-sanity-bound` note in
    // CLAUDE.md). 1e-8 is comfortably tighter than the ~1e-3 eV agreement being
    // asserted downstream.
    // SCF settings. `energy_conv` is a NOT-DESCENDING sanity bound, not a
    // tightness target (see CLAUDE.md). Setting it to 1e-10 puts it below the
    // XC grid's own noise floor, so `de < energy_conv` can never be satisfied
    // and a KS SCF spins to max_iter no matter how well the DENSITY converges.
    //
    // MEASURED on water/cc-pVDZ/PBE, sweeping energy_conv at fixed
    // density_conv = 1e-8:
    //
    //     energy_conv   converged   iters   E (Ha)
    //        1e-10         NO        200    -76.3334713912
    //        1e-8          yes        13    -76.3334713822
    //        1e-6          yes        12    -76.3334713801
    //        1e-3 (deflt)  yes        12    -76.3334713801
    //
    // Same energy to ~1e-8 Ha in every case — it was converged all along and
    // only the exit gate was unreachable. The density threshold is what
    // actually bounds accuracy here, and 1e-8 is far tighter than the ~1e-3 eV
    // agreement asserted downstream.
    let cfg = RhfConfig {
        density_conv: 1e-8,
        xc: xc.map(|s| s.to_string()),
        max_iter: 200,
        ..Default::default()
    };
    let ks = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(ks.converged, "{key}: reference SCF did not converge");

    let tda = run_tda_dft(&mol, &obs, &dfbs, op, &ks, xc, &TdaDftConfig::default()).unwrap();

    assert_eq!(tda.nocc, py.nocc, "{key}: nocc mismatch");
    assert_eq!(tda.nvir, py.nvir, "{key}: nvir mismatch");

    // TEETH: for a DFT functional the f_xc term must actually be active, else
    // this is silently re-testing the CIS path the anchor already covers.
    if xc.is_some() {
        assert!(tda.fxc_included, "{key}: f_xc must be included for a functional");
    }

    let n = n_check.min(py.states.len()).min(tda.omega.len());
    let mut max_dev = 0.0f64;
    let mut max_f_dev = 0.0f64;
    println!("\n== {key} ==");
    println!(
        "{:>4} {:>12} {:>12} {:>10}  {:>10} {:>10}  match",
        "st", "ferric(eV)", "pyscf(eV)", "dev(eV)", "f_ferric", "f_pyscf"
    );
    for s in 0..n {
        let w_f = tda.omega[s] * HARTREE_TO_EV;
        let w_p = py.states[s].omega_ev;
        let dev = (w_f - w_p).abs();
        let (i_f, a_f) = dominant_ia(&tda.x, s, tda.nvir);
        let [i_p, a_p] = py.states[s].dominant_ia;
        let same = i_f == i_p && a_f == a_p;
        let f_f = tda.oscillator_strength[s];
        let f_p = py.states[s].osc_length;
        println!(
            "{s:>4} {w_f:>12.6} {w_p:>12.6} {dev:>10.2e}  {f_f:>10.6} {f_p:>10.6}  \
             ({i_f},{a_f}) vs ({i_p},{a_p}) {}",
            if same { "OK" } else { "**DIFFER**" }
        );
        // The character check is the load-bearing one: if the dominant (i,a)
        // disagrees we are comparing different states and the energy deviation
        // below is meaningless, however small it looks.
        assert!(
            same,
            "{key} state {s}: dominant (i,a) differs — ferric ({i_f},{a_f}) vs \
             PySCF ({i_p},{a_p}). Comparing different states; do NOT read the \
             energy agreement as validation."
        );
        max_dev = max_dev.max(dev);
        max_f_dev = max_f_dev.max((f_f - f_p).abs());
    }
    println!("max |dOmega| = {max_dev:.3e} eV over {n} states; max |df| = {max_f_dev:.3e}");

    assert!(
        max_dev < tol_ev,
        "{key}: max excitation-energy deviation {max_dev:.3e} eV exceeds {tol_ev:.1e} eV"
    );
    // Oscillator strengths are much more grid/basis sensitive than energies;
    // a loose bound here still catches a wrong transition dipole (which shows
    // up at O(1)) without failing on integration noise.
    assert!(
        max_f_dev < 5e-2,
        "{key}: max oscillator-strength deviation {max_f_dev:.3e} exceeds 5e-2"
    );
}

/// HF reference — this is CIS, and duplicates the anchor's coverage against an
/// EXTERNAL reference rather than an internal one. If this fails while the
/// anchor passes, the bug is in the (ia) Coulomb/exchange assembly, not f_xc.
#[test]
fn tda_hf_water_sto3g_matches_pyscf() {
    compare(None, "sto-3g__HF", "sto-3g", 3e-3, 6);
}

/// Pure LDA — the first case where the f_xc adapter is load-bearing, and the
/// simplest kernel (no sigma coupling).
#[test]
fn tda_lda_water_sto3g_matches_pyscf() {
    compare(Some("LDA"), "sto-3g__lda,vwn", "sto-3g", 3e-3, 6);
}

/// Pure GGA — exercises the sigma = |grad rho|^2 coupling terms
/// (v2rhosigma / v2sigma2) that LDA does not touch.
#[test]
fn tda_pbe_water_sto3g_matches_pyscf() {
    compare(Some("pbe"), "sto-3g__pbe,pbe", "sto-3g", 3e-3, 6);
}

/// Hybrid — both a nonzero c_HF on the (ij|ab) term AND the GGA f_xc kernel,
/// so it catches a wrong exact-exchange fraction that the pure functionals
/// cannot (their c_HF is 0, so an error there is invisible).
#[test]
fn tda_b3lyp_water_sto3g_matches_pyscf() {
    compare(Some("b3lyp"), "sto-3g__b3lyp", "sto-3g", 3e-3, 6);
}

/// One real-basis case. STO-3G alone could hide a bug that only appears with a
/// larger virtual space or with polarization functions on the grid.
///
/// Bound is 3e-2 eV, an order looser than the STO-3G cases, because
/// `cc-pvdz-ri` is undersized for excitation energies at this orbital basis
/// (see the aux-sweep note in `compare`). The looseness is an RI-basis
/// limitation, NOT slack for implementation error: the deviation shrinks
/// monotonically toward the exact-integral answer as the aux grows.
#[test]
fn tda_pbe_water_ccpvdz_matches_pyscf() {
    compare(Some("pbe"), "cc-pvdz__pbe,pbe", "cc-pvdz", 3e-2, 6);
}
