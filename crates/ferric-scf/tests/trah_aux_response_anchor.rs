//! **Exactness anchor for AURORA's auxiliary two-electron response (#279).**
//!
//! The original auxiliary-curvature negative (`trah_aux_rho.rs`) was UNANCHORED:
//! its only anchor zeroed the two-electron response and checked the 0.5 scale,
//! so the response CONTENT (Coulomb 8, both exchange contractions, the dressed
//! tensor's index order, the MO refresh) was never compared with a known limit.
//! "STO-3G underfits the response" (physics) and "the aux response is built
//! wrong" (artifact) both predict rho < 1.
//!
//! This file compares `AURORA_TO_TRAH_CURVATURE * AuxCurvature::base_apply_ref`
//! with `rhf_newton::hessian_matvec` at a converged water/cc-pVDZ RHF point.
//!
//! # Hypotheses, written before running
//!
//! * Physics (construction correct): the two-electron part differs from exact by
//!   the density-fitting error of the aux basis, small (<1e-2) for a rich RI/JK
//!   basis and O(0.1-1) for STO-3G, ordered by aux quality.
//! * Artifact (construction wrong): the rich-aux error is O(1) and does NOT fall
//!   as the aux basis is enriched; or a clean factor/transposition appears.
//!
//! A numpy/PySCF prototype of the same comparison (same dressed-tensor
//! construction, exact response cross-checked against `mf.gen_response` to 1e-15)
//! measured, for the 2e part on a random kappa: def2-universal-jkfit 2.4e-3,
//! cc-pvdz-ri 1.5e-3, cc-pvtz-jkfit 5.5e-4, cc-pvqz-jkfit 2.1e-4, sto-3g 0.25.
//!
//! The comparison is on the 2e part only (the Fock-gap term is shared exactly and
//! would dilute the error).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::aurora::{AuroraConfig, AuxCurvature};
use ferric_scf::rhf::{build_jk, solve_rhf, RhfConfig};
use ferric_scf::rhf_newton::{hessian_matvec, RhfNewtonInputs};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::trah::AURORA_TO_TRAH_CURVATURE;
use ndarray::{s, Array2};

fn fro(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Relative error of the auxiliary 2e response vs the exact one, worst over
/// `seeds` random kappa. Returns (worst 2e rel err, worst total rel err).
fn two_e_error(auxbasis: &str, seeds: &[u64]) -> (f64, f64) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(res.converged);
    let n = prep.nbasis();
    let nocc = mol.nelec() as usize / 2;
    let nv = n - nocc;
    let c = res.mos_alpha.clone();

    let h = oneelectron::hcore(&prep);
    let cocc = c.slice(s![.., ..nocc]);
    let dtot = 2.0 * cocc.dot(&cocc.t());
    let mut j = Array2::<f64>::zeros((n, n));
    let mut k = Array2::<f64>::zeros((n, n));
    build_jk(&ctx, &prep, &bounds, 1e-12, &dtot, &mut j, &mut k).unwrap();
    let f_mo = c.t().dot(&(&h + &j - &(0.5 * &k))).dot(&c);

    let inputs = RhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c: &c,
        f_mo: &f_mo,
        nocc,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: 1e-12,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let pool = ferric_scf::engine_pool::EnginePool::new(op, &prep, 1e-14).unwrap();

    let acfg = AuroraConfig {
        auxbasis: auxbasis.to_string(),
        ..Default::default()
    };
    let aux = AuxCurvature::new(&mol, &prep, c.view(), nocc, 1.0, &acfg).unwrap();
    let f_vv = f_mo.slice(s![nocc.., nocc..]).to_owned();
    let f_oo = f_mo.slice(s![..nocc, ..nocc]).to_owned();

    let (mut worst2, mut worst_t) = (0.0f64, 0.0f64);
    for &seed in seeds {
        let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut x = Array2::<f64>::zeros((nv, nocc));
        for v in x.iter_mut() {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            *v = ((st >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0;
        }
        let exact = hessian_matvec(&ctx, &inputs, &x, &pool).unwrap();
        let y = aux.base_apply_ref(x.view(), &f_vv, &f_oo, 0.0) * AURORA_TO_TRAH_CURVATURE;
        // Shared Fock-gap term, subtracted from both to isolate the 2e part.
        let mut gap = Array2::<f64>::zeros((nv, nocc));
        for a in 0..nv {
            for i in 0..nocc {
                gap[(a, i)] = (f_mo[(nocc + a, nocc + a)] - f_mo[(i, i)]) * x[(a, i)];
            }
        }
        let e2 = fro(&(&exact - &gap));
        assert!(e2 > 1e-3, "inert anchor: exact 2e response ~0 ({e2:.3e})");
        worst2 = worst2.max(fro(&(&y - &exact)) / e2);
        worst_t = worst_t.max(fro(&(&y - &exact)) / fro(&exact));
    }
    (worst2, worst_t)
}

/// Rich-aux anchor. Bar = ~10x the measured value (2e-3 .. 3e-3 across seeds).
#[test]
fn rich_aux_response_matches_exact_hessian_matvec() {
    for (aux, bar) in [("def2-universal-jkfit", 3e-2), ("cc-pvdz-ri", 2e-2)] {
        let (e2, et) = two_e_error(aux, &[1, 2, 3]);
        eprintln!("ANCHOR aux={aux:22} 2e_rel_err={e2:.3e} total_rel_err={et:.3e} bar={bar:.0e}");
        assert!(
            e2 < bar,
            "{aux}: 2e response rel err {e2:.3e} >= bar {bar:.0e}"
        );
    }
}

/// Negative control: STO-3G must miss by far more than the rich-aux bar, or the
/// anchor cannot see aux quality.
#[test]
fn sto3g_aux_misses_the_exact_matvec() {
    let (e2, et) = two_e_error("sto-3g", &[1, 2, 3]);
    eprintln!("ANCHOR aux=sto-3g                 2e_rel_err={e2:.3e} total_rel_err={et:.3e}");
    assert!(
        e2 > 0.1,
        "sto-3g 2e error {e2:.3e} should exceed 0.1 (10x the rich bar scale)"
    );
}
