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
use ferric_scf::trah::{
    offdiag_gap_correction, rhf_trah_step, rhf_trah_step_aux_cached, TrahConfig,
    AURORA_TO_TRAH_CURVATURE,
};
use ndarray::{s, Array2};

fn fro(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Cayley rotation of `c` by a random occ->virt kappa scaled to max |kappa| = `size`.
fn rotate_c(c: &Array2<f64>, nocc: usize, size: f64) -> Array2<f64> {
    use ndarray_linalg::Solve;
    let n = c.nrows();
    let mut kap = Array2::<f64>::zeros((n, n));
    let mut st = 0xC0FF_EE11_u64;
    let mut mx = 0.0f64;
    for a in nocc..n {
        for i in 0..nocc {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            let v = ((st >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0;
            kap[(a, i)] = v;
            kap[(i, a)] = -v;
            mx = mx.max(v.abs());
        }
    }
    kap.mapv_inplace(|v| v * size / mx);
    let half = 0.5 * &kap;
    let eye = Array2::<f64>::eye(n);
    let (am, bm) = (&eye - &half, &eye + &half);
    let mut u = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let sol = am.solve(&bm.column(col).to_owned()).unwrap();
        u.column_mut(col).assign(&sol);
    }
    c.dot(&u)
}

/// A converged water/cc-pVDZ RHF point, optionally rotated off the canonical
/// orbitals, with everything `RhfNewtonInputs` needs.
struct Point {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
    c: Array2<f64>,
    /// The converged canonical orbitals (== `c` when `rot == 0`).
    c_canon: Array2<f64>,
    f_mo: Array2<f64>,
    nocc: usize,
}

impl Point {
    fn new(rot: f64) -> Self {
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
        let c = if rot > 0.0 {
            rotate_c(&res.mos_alpha, nocc, rot)
        } else {
            res.mos_alpha.clone()
        };
        let h = oneelectron::hcore(&prep);
        let cocc = c.slice(s![.., ..nocc]);
        let dtot = 2.0 * cocc.dot(&cocc.t());
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        build_jk(&ctx, &prep, &bounds, 1e-12, &dtot, &mut j, &mut k).unwrap();
        let f_mo = c.t().dot(&(&h + &j - &(0.5 * &k))).dot(&c);
        Self {
            mol,
            prep,
            bounds,
            ctx,
            c,
            c_canon: res.mos_alpha.clone(),
            f_mo,
            nocc,
        }
    }

    fn inputs(&self) -> RhfNewtonInputs<'_> {
        RhfNewtonInputs {
            prep: &self.prep,
            bounds: &self.bounds,
            c: &self.c,
            f_mo: &self.f_mo,
            nocc: self.nocc,
            k_mix_sr: 1.0,
            rsh: None,
            fxc: None,
            thresh: 1e-12,
            ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
        }
    }

    fn aux_cfg(auxbasis: &str) -> AuroraConfig {
        AuroraConfig {
            auxbasis: auxbasis.to_string(),
            ..Default::default()
        }
    }

    /// Aux model built at the CURRENT (possibly rotated) orbitals.
    fn aux(&self, auxbasis: &str) -> AuxCurvature {
        let acfg = Self::aux_cfg(auxbasis);
        AuxCurvature::new(&self.mol, &self.prep, self.c.view(), self.nocc, 1.0, &acfg).unwrap()
    }

    /// Aux model built at the CANONICAL orbitals, as the SCF driver builds it
    /// at its first TRAH step; the step must refresh it onto `c`.
    fn aux_stale(&self, auxbasis: &str) -> AuxCurvature {
        let acfg = Self::aux_cfg(auxbasis);
        let cc = self.c_canon.view();
        AuxCurvature::new(&self.mol, &self.prep, cc, self.nocc, 1.0, &acfg).unwrap()
    }

    fn blocks(&self) -> (Array2<f64>, Array2<f64>) {
        let no = self.nocc;
        (
            self.f_mo.slice(s![no.., no..]).to_owned(),
            self.f_mo.slice(s![..no, ..no]).to_owned(),
        )
    }
}

fn random_kappa(seed: u64, nv: usize, nocc: usize) -> Array2<f64> {
    let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut x = Array2::<f64>::zeros((nv, nocc));
    for v in x.iter_mut() {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        *v = ((st >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0;
    }
    x
}

/// Errors of the auxiliary operator, all relative to the exact 2e response norm.
struct AnchorErr {
    /// aux vs exact with the FULL Fock-gap term (the like-for-like control).
    vs_full_gap: f64,
    /// aux vs `rhf_newton::hessian_matvec` as shipped (diagonal gap only).
    vs_diag_gap: f64,
    /// size of the one-electron diag-vs-full difference itself.
    gap_mismatch: f64,
    /// aux vs `hessian_matvec`, relative to the TOTAL matvec norm.
    total: f64,
}

/// Worst over `seeds` random kappa, at the converged point (`rot = 0`) or at a
/// non-canonical point rotated by max |kappa| = `rot`.
fn anchor_err(auxbasis: &str, seeds: &[u64], rot: f64) -> AnchorErr {
    let pt = Point::new(rot);
    let inputs = pt.inputs();
    let pool =
        ferric_scf::engine_pool::EnginePool::new(Operator::coulomb(), &pt.prep, 1e-14).unwrap();
    let aux = pt.aux(auxbasis);
    let (f_vv, f_oo) = pt.blocks();
    let nv = pt.prep.nbasis() - pt.nocc;

    let mut w = AnchorErr {
        vs_full_gap: 0.0,
        vs_diag_gap: 0.0,
        gap_mismatch: 0.0,
        total: 0.0,
    };
    for &seed in seeds {
        let x = random_kappa(seed, nv, pt.nocc);
        let exact_diag = hessian_matvec(&pt.ctx, &inputs, &x, &pool).unwrap();
        let corr = offdiag_gap_correction(&pt.f_mo, &x, pt.nocc);
        let exact_full = &exact_diag + &corr;
        let y = aux.base_apply_ref(x.view(), &f_vv, &f_oo, 0.0) * AURORA_TO_TRAH_CURVATURE;
        // 2e part of the exact response = full matvec minus the full gap term.
        let gap_full = f_vv.dot(&x) - x.dot(&f_oo);
        let e2 = fro(&(&exact_full - &gap_full));
        assert!(e2 > 1e-3, "inert anchor: exact 2e response ~0 ({e2:.3e})");
        w.vs_full_gap = w.vs_full_gap.max(fro(&(&y - &exact_full)) / e2);
        w.vs_diag_gap = w.vs_diag_gap.max(fro(&(&y - &exact_diag)) / e2);
        w.gap_mismatch = w.gap_mismatch.max(fro(&corr) / e2);
        w.total = w.total.max(fro(&(&y - &exact_diag)) / fro(&exact_diag));
    }
    w
}

/// Rich-aux anchor at the converged (canonical) point. Bar = ~10x measured.
#[test]
fn rich_aux_response_matches_exact_hessian_matvec() {
    for (aux, bar) in [("def2-universal-jkfit", 3e-2), ("cc-pvdz-ri", 2e-2)] {
        let e = anchor_err(aux, &[1, 2, 3], 0.0);
        eprintln!(
            "ANCHOR canonical aux={aux:22} vs_full={:.3e} vs_diag={:.3e} gap_mismatch={:.1e} total={:.3e} bar={bar:.0e}",
            e.vs_full_gap, e.vs_diag_gap, e.gap_mismatch, e.total
        );
        assert!(
            e.vs_full_gap < bar,
            "{aux}: rel err {:.3e} >= bar {bar:.0e}",
            e.vs_full_gap
        );
    }
}

/// The same anchor at a NON-canonical point (max |kappa| = 0.1), where the
/// diagonal-gap `hessian_matvec` and the full-gap aux operator differ in the
/// one-electron term. Against the like-for-like full-gap exact control the aux
/// error must stay at the fitting level; against the diagonal-gap matvec it is
/// contaminated by the one-electron mismatch, and this test pins that too.
#[test]
fn rich_aux_matches_full_gap_exact_at_noncanonical_point() {
    let e = anchor_err("def2-universal-jkfit", &[1, 2, 3], 0.1);
    eprintln!(
        "ANCHOR rot=0.1 def2-universal-jkfit vs_full={:.3e} vs_diag={:.3e} gap_mismatch={:.3e} total={:.3e}",
        e.vs_full_gap, e.vs_diag_gap, e.gap_mismatch, e.total
    );
    assert!(
        e.vs_full_gap < 3e-2,
        "vs full-gap exact: {:.3e}",
        e.vs_full_gap
    );
    // The anchor can SEE the confound: the diag-gap comparison is worse by the
    // mismatch, and the mismatch is a resolvable fraction of the 2e response.
    assert!(
        e.gap_mismatch > 1e-3 && e.vs_diag_gap > e.vs_full_gap,
        "non-canonical point should expose the diag-vs-full gap mismatch: {:.3e}",
        e.gap_mismatch
    );
}

/// Negative control: STO-3G must miss by far more than the rich-aux bar, or the
/// anchor cannot see aux quality.
#[test]
fn sto3g_aux_misses_the_exact_matvec() {
    let e = anchor_err("sto-3g", &[1, 2, 3], 0.0);
    eprintln!("ANCHOR canonical aux=sto-3g vs_full={:.3e}", e.vs_full_gap);
    assert!(
        e.vs_full_gap > 0.1,
        "sto-3g 2e error {:.3e} should exceed 0.1",
        e.vs_full_gap
    );
}

/// **Convergence anchor (the full-rank limit of #279).** An exact-B route does
/// not exist here (the auxiliary tensor is a fit by construction, and the
/// orbital basis is not a valid aux for the 4-index limit), so the honest
/// full-rank statement is a CONVERGENCE one: along one aux family the error of
/// the response against the full-gap exact matvec must fall as the aux basis is
/// enriched, and the richest member bounds any construction defect from above.
///
/// Both-sides derivation. Fitting error shrinks with the aux basis; a
/// construction defect (a wrong factor, a dropped term, a transposed index) is
/// aux-independent and sets a FLOOR. So (1) the error must fall monotonically
/// def2-universal-jkfit -> cc-pvtz-jkfit -> cc-pvqz-jkfit, and (2) the qz error
/// must sit under an absolute bar. What it can resolve: a defect of relative
/// size >= the bar (about 2x the worst measured qz error, 2.7e-4) in the 2e response; a
/// smaller one hides under the qz fitting error. Mutation: Coulomb prefactor
/// 8 -> 8.04 (0.5 % of the Coulomb term) is below the def2 error but above the
/// qz floor, so only this test (not the 3e-2 bars above) can see it.
#[test]
fn aux_error_falls_with_aux_enrichment_to_a_derived_floor() {
    for rot in [0.0, 0.1] {
        let ladder = ["def2-universal-jkfit", "cc-pvtz-jkfit", "cc-pvqz-jkfit"];
        let errs: Vec<f64> = ladder
            .iter()
            .map(|a| anchor_err(a, &[1, 2, 3], rot).vs_full_gap)
            .collect();
        eprintln!("LADDER rot={rot} {ladder:?} vs_full = {errs:?}");
        assert!(
            errs[1] < errs[0] && errs[2] < errs[1],
            "rot={rot}: error not monotone in aux enrichment: {errs:?}"
        );
        assert!(
            errs[2] < QZ_FLOOR_BAR,
            "rot={rot}: qz-jkfit error {:.3e} >= floor bar {QZ_FLOOR_BAR:.0e}",
            errs[2]
        );
    }
}

/// Upper bound on the qz-jkfit 2e-response error (about 2x the worst measured, 2.7e-4, and 1.4x the independent PySCF DF error 3.6e-4).
const QZ_FLOOR_BAR: f64 = 5e-4;

/// Relative differences between the aux TRAH step and the full-gap exact TRAH
/// step taken from the same non-canonical point.
struct StepDiff {
    kappa: f64,
    predicted: f64,
    exact_kappa_norm: f64,
}

/// One TRAH step each way from a point rotated by `rot`. The aux model is built
/// at the CANONICAL orbitals and reaches the rotated ones only through
/// `refresh_orbitals`, exactly as in the SCF driver.
fn step_diff(auxbasis: &str, rot: f64, radius: f64) -> StepDiff {
    let pt = Point::new(rot);
    let inp = pt.inputs();
    let exact_cfg = TrahConfig {
        exact_full_gap: true,
        ..Default::default()
    };
    let (_, ex) = rhf_trah_step(&pt.ctx, &inp, radius, &exact_cfg).unwrap();
    let mut slot = Some(pt.aux_stale(auxbasis));
    let rcfg = RhfConfig {
        trah: TrahConfig {
            aux_curvature: true,
            ..Default::default()
        },
        aurora: Point::aux_cfg(auxbasis),
        ..Default::default()
    };
    let geom = (&pt.mol, &pt.prep, pt.c.view(), pt.nocc, 1.0);
    let (_, ax) = rhf_trah_step_aux_cached(&mut slot, geom, &rcfg, &inp, radius).unwrap();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let dk: Vec<f64> = ex.kappa.iter().zip(&ax.kappa).map(|(a, b)| a - b).collect();
    StepDiff {
        kappa: norm(&dk) / norm(&ex.kappa),
        predicted: (ex.predicted - ax.predicted).abs() / ex.predicted.abs(),
        exact_kappa_norm: norm(&ex.kappa),
    }
}

/// **Step-level anchor.** A full TRAH step from a non-canonical start: the aux
/// step must agree with the full-gap exact step at the fitting level. Unlike the
/// operator anchors this exercises the wiring in `rhf_trah_step_aux` (which Fock
/// block is `f_oo`) and `rhf_trah_step_aux_cached` (`refresh_orbitals` onto the
/// rotated orbitals), and it needs a NONZERO gradient, so it starts off the
/// converged point. Mutants of each give an O(1) step difference.
#[test]
fn aux_trah_step_matches_full_gap_exact_step_off_canonical() {
    let d = step_diff("def2-universal-jkfit", 0.1, 0.3);
    eprintln!(
        "STEP rot=0.1 radius=0.3 |kappa|={:.3e} dkappa_rel={:.3e} dpred_rel={:.3e}",
        d.exact_kappa_norm, d.kappa, d.predicted
    );
    assert!(
        d.exact_kappa_norm > 1e-3,
        "inert step test: |kappa| = {:.3e}",
        d.exact_kappa_norm
    );
    assert!(d.kappa < STEP_KAPPA_BAR, "kappa rel diff {:.3e}", d.kappa);
    assert!(
        d.predicted < STEP_PRED_BAR,
        "pred rel diff {:.3e}",
        d.predicted
    );
}

/// Bars ~10x the measured 1.5e-4 / 1.1e-5 (def2-universal-jkfit, rot 0.1, radius 0.3).
const STEP_KAPPA_BAR: f64 = 2e-3;
const STEP_PRED_BAR: f64 = 2e-4;
