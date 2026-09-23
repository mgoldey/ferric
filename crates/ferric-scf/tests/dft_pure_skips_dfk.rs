//! A DF-K fitter is built only when its K reaches the Fock matrix.
//!
//! `run_dft` names `def2-universal-jkfit` for `df_k_aux` on EVERY functional,
//! and `solve_rhf` used to honour that for a pure GGA: it built the DfK and ran
//! `build_from_occ` every iteration, then multiplied the K by k_mix = 0
//! (danuglipron PBE/STO-3G: setup:df_fitters 91 s, jk_build 33 s). The gate
//! that removes it must (a) not move a pure functional's energy and (b) not
//! take RI-K away from HF or a plain hybrid. The "is a DfK built at all" check
//! lives in `rhf::tests` (it needs a crate-private counter).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const JKFIT: &str = "def2-universal-jkfit";

/// Water/STO-3G energy with RI-J on and `df_k_aux` as given
/// (`Some("")` = explicit conventional K).
fn water(xc: Option<&str>, df_k_aux: &str) -> f64 {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        xc: xc.map(str::to_string),
        df_j_aux: Some(JKFIT.to_string()),
        df_k_aux: Some(df_k_aux.to_string()),
        // Default convergence thresholds on purpose: 1e-10 Ha sits below the
        // RI noise floor and never converges (see energy_conv's doc).
        ..Default::default()
    };
    let res = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(
        res.converged,
        "{xc:?} df_k_aux={df_k_aux:?} did not converge"
    );
    res.energy
}

/// Energy anchor for the gate: PBE with a named DF-K aux and PBE with DF-K
/// explicitly off must be BIT-identical, because the K never entered F.
///
/// What this catches: a gate that changes more than the dead K, e.g. one that
/// drops the DF routing for J along with the DfK, or a K that leaks into F.
/// It would ALSO have passed before the fix (the unused K was multiplied by
/// zero then too), so it is a no-regression anchor, not the detector of the
/// gate itself; that is `rhf::tests::pure_gga_with_named_df_k_aux_builds_no_dfk`.
#[test]
fn pbe_named_df_k_aux_is_bit_identical_to_df_k_off() {
    let e_named = water(Some("PBE"), JKFIT);
    let e_off = water(Some("PBE"), "");
    assert_eq!(
        e_named.to_bits(),
        e_off.to_bits(),
        "PBE energy moved with df_k_aux: named={e_named:.15} off={e_off:.15}"
    );
}

/// The K-fitting error, MEASURED with PySCF on this system (RI-JK vs RI-J-only,
/// def2-universal-jkfit, conv_tol 1e-11): HF 3.5e-4 Ha, B3LYP 7.0e-5 Ha. The
/// lower bar sits 70x below the smaller of the two and ~1e4x above the SCF
/// convergence noise at energy_conv 1e-10; the upper bar catches a K that is
/// wrong rather than merely fitted.
const MIN_K_FIT_ERR: f64 = 1e-6;
const MAX_K_FIT_ERR: f64 = 1e-2;

fn assert_ri_k_active(xc: Option<&str>) {
    let e_ri = water(xc, JKFIT);
    let e_direct = water(xc, "");
    let d = (e_ri - e_direct).abs();
    eprintln!("{xc:?}: RI-K {e_ri:.12} direct-K {e_direct:.12} |d|={d:.3e}");
    assert!(
        d > MIN_K_FIT_ERR,
        "{xc:?}: RI-K and direct K agree to {d:.3e} -- the DF-K was not used \
         (gate dropped a K that IS consumed)"
    );
    assert!(
        d < MAX_K_FIT_ERR,
        "{xc:?}: RI-K vs direct K differ by {d:.3e}"
    );
}

/// HF: xc = None, so the DFT-specific `needs_k` is FALSE while K is consumed.
/// FAILS (|d| == 0, both runs take direct K) if the gate keys on `needs_k`.
#[test]
fn hf_with_df_k_aux_still_uses_ri_k() {
    assert_ri_k_active(None);
}

/// Plain ω = 0 hybrid. FAILS if the gate drops hybrids (e.g. keys on k_mix.sr
/// being 1, or on ω alone).
#[test]
fn b3lyp_with_df_k_aux_still_uses_ri_k() {
    assert_ri_k_active(Some("B3LYP"));
}
