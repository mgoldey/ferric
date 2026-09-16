//! SCF/KS internal stability analysis: does ferric know a saddle from a minimum?
//!
//! # The gap this closes
//!
//! Ferric's SCF converges on the Brillouin condition `F_ai = 0`, which makes a
//! solution STATIONARY, not MINIMAL. Before this module there was no way to
//! tell the two apart, and the following was measured on 2026-09-16:
//!
//! HeNe⁺ / def2-SVP / UHF at R = 2.0 Å. Ferric converges in 14 iterations to
//! `E = -130.5003466413`. PySCF 2.13.0 from its default guess reaches
//! `E = -130.5053405386` and reports that state internally AND externally
//! stable. Forcing PySCF onto ferric's state with `scf.addons.mom_occ`
//! reproduces `E = -130.50034664` (agreement 1.3e-9 Ha) and `mf.stability()`
//! returns `internal stable = False` for it. Ferric was converging cleanly and
//! reproducibly onto a SADDLE POINT 4.99e-3 Ha = 0.136 eV above the true
//! minimum, and had no diagnostic capable of noticing.
//!
//! # Test design (protocol order is deliberate)
//!
//! 1. **EXACTNESS ANCHOR FIRST** — `anchor_*`: on systems with a known-stable
//!    minimum, λ_min must be POSITIVE, and must equal the lowest eigenvalue of
//!    the EXPLICITLY BUILT dense Hessian (built by applying the matvec to every
//!    unit vector and diagonalizing). The dense cross-check is an INDEPENDENT
//!    construction of the same operator: it shares the matvec but not the
//!    eigensolver, so it separates "the Davidson is wrong" from "the operator
//!    is wrong". Written and passing before any HeNe⁺ number was believed.
//! 2. **BOTH DIRECTIONS REACHABLE** — a checker that always says one thing is
//!    useless. `anchor_*` proves STABLE is reachable, `hene_*` proves UNSTABLE
//!    is reachable, on real converged SCF states.
//! 3. **THE DECISIVE TEST** — `hene_plus_pi_state_is_internally_unstable`:
//!    ferric's own converged HeNe⁺ UHF state must come out UNSTABLE, matching
//!    PySCF's `stability()` verdict on the numerically identical state.
//! 4. **FOLLOW THE INSTABILITY** — `hene_plus_following_the_instability_lowers_the_energy`:
//!    rotating along the negative eigenvector and re-converging must reach a
//!    LOWER energy. This is the strongest available evidence that the
//!    eigenVECTOR is meaningful and not just its sign.
//!
//! # Artifact hypothesis (written before measuring)
//!
//! If the analysis is REAL, I expect λ_min > 0 on water/H₂ and λ_min < 0 on
//! HeNe⁺, with the HeNe⁺ eigenvector carrying β-spin character (the hole is a
//! β hole) and following it reaching PySCF's -130.50534054.
//!
//! If my implementation is BROKEN in the most likely way — a sign error, or
//! driving the wrong block — I expect λ_min < 0 EVERYWHERE (including on the
//! anchors), or λ_min > 0 everywhere. Both are distinguishable from the real
//! signal by the anchors, which is why the anchors run first and on separate
//! systems. The one failure mode the anchors do NOT catch is an operator that
//! is correct but restricted to a subspace excluding the instability; test 4
//! (following the vector to a lower energy) is what closes that hole, because
//! a spurious negative direction would not re-converge anywhere better.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{build_jk, solve_rhf, RhfConfig};
use ferric_scf::rhf_newton::RhfNewtonInputs;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::{
    rhf_internal_stability, uhf_internal_stability, StabilityConfig, StabilityKind,
};
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess};
use ferric_scf::uhf_newton::UhfNewtonInputs;
use ndarray::{s, Array2};
use ndarray_linalg::{Eigh, Solve, UPLO};

/// A converged UHF state plus everything the stability analysis needs.
struct UhfState {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    c_a: Array2<f64>,
    c_b: Array2<f64>,
    f_a_mo: Array2<f64>,
    f_b_mo: Array2<f64>,
    nocc_a: usize,
    nocc_b: usize,
    energy: f64,
}

/// Converge UHF on `xyz` and rebuild the MO-basis Fock matrices at the
/// converged orbitals (the Hessian is evaluated AT the stationary point, so
/// the Fock must correspond to exactly the MOs handed to the matvec).
fn converge_uhf(xyz: &str, charge: i32, mult: usize, basis_name: &str) -> UhfState {
    converge_uhf_from(xyz, charge, mult, basis_name, None)
}

/// As [`converge_uhf`], but optionally seeded from explicit starting MOs
/// (used to re-converge after rotating along an instability).
fn converge_uhf_from(
    xyz: &str,
    charge: i32,
    mult: usize,
    basis_name: &str,
    guess: Option<(Array2<f64>, Array2<f64>)>,
) -> UhfState {
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 400,
        ..Default::default()
    };
    let res = match guess.as_ref() {
        Some((a, b)) => {
            solve_uhf_with_guess(&ctx, &mol, &prep, &bounds, &cfg, Some((a, b))).unwrap()
        }
        None => solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap(),
    };
    assert!(res.converged, "UHF must converge for a stability analysis");

    let c_a = res.mos_alpha.clone();
    let c_b = res.mos_beta.clone().unwrap();
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    let nocc_a = (nelec + two_s) / 2;
    let nocc_b = (nelec - two_s) / 2;

    let (f_a, f_b) = uhf_fock(&ctx, &prep, &bounds, &c_a, &c_b, nocc_a, nocc_b);
    let f_a_mo = c_a.t().dot(&f_a).dot(&c_a);
    let f_b_mo = c_b.t().dot(&f_b).dot(&c_b);

    UhfState {
        mol,
        prep,
        bounds,
        c_a,
        c_b,
        f_a_mo,
        f_b_mo,
        nocc_a,
        nocc_b,
        energy: res.energy,
    }
}

/// AO-basis α/β UHF Fock matrices at given MOs, built from scratch. Kept local
/// (rather than read off `ScfResult`) so the Fock is unambiguously the one at
/// the MOs the Hessian is evaluated with, including after an explicit
/// rotation.
fn uhf_fock(
    ctx: &ParallelContext,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    nocc_a: usize,
    nocc_b: usize,
) -> (Array2<f64>, Array2<f64>) {
    let n = prep.nbasis();
    let h = oneelectron::hcore(prep);
    let d_a = c_a
        .slice(s![.., ..nocc_a])
        .dot(&c_a.slice(s![.., ..nocc_a]).t());
    let d_b = c_b
        .slice(s![.., ..nocc_b])
        .dot(&c_b.slice(s![.., ..nocc_b]).t());
    let d_t = &d_a + &d_b;

    let mut j = Array2::<f64>::zeros((n, n));
    let mut kdum = Array2::<f64>::zeros((n, n));
    build_jk(ctx, prep, bounds, 1e-12, &d_t, &mut j, &mut kdum).unwrap();
    let mut k_a = Array2::<f64>::zeros((n, n));
    let mut k_b = Array2::<f64>::zeros((n, n));
    let mut jdum = Array2::<f64>::zeros((n, n));
    build_jk(ctx, prep, bounds, 1e-12, &d_a, &mut jdum, &mut k_a).unwrap();
    jdum.fill(0.0);
    build_jk(ctx, prep, bounds, 1e-12, &d_b, &mut jdum, &mut k_b).unwrap();

    (&h + &j - &k_a, &h + &j - &k_b)
}

fn uhf_inputs<'a>(st: &'a UhfState) -> UhfNewtonInputs<'a> {
    UhfNewtonInputs {
        prep: &st.prep,
        bounds: &st.bounds,
        c_a: &st.c_a,
        c_b: &st.c_b,
        f_a_mo: &st.f_a_mo,
        f_b_mo: &st.f_b_mo,
        nocc_a: st.nocc_a,
        nocc_b: st.nocc_b,
        k_mix_sr: 1.0, // pure HF
        fxc: None,
        thresh: 1e-12,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    }
}

/// Build the FULL dense Hessian by applying the matvec to every unit vector,
/// then diagonalize it. This is the INDEPENDENT construction that separates a
/// broken eigensolver from a broken operator: it shares the matvec with the
/// Davidson path but nothing else — no preconditioner, no subspace, no
/// iteration. Only affordable because these systems are deliberately tiny
/// (HeNe⁺ dim = 78 + 70 = 148).
fn dense_uhf_hessian_lowest(st: &UhfState) -> (f64, Vec<f64>) {
    let ctx = ParallelContext::default();
    let inp = uhf_inputs(st);
    let n = st.c_a.nrows();
    let (nva, nvb) = (n - st.nocc_a, n - st.nocc_b);
    let dim_a = nva * st.nocc_a;
    let dim_b = nvb * st.nocc_b;
    let dim = dim_a + dim_b;
    let pool = ferric_scf::engine_pool::EnginePool::new(st.bounds.op, &st.prep, 1e-14).unwrap();

    let mut hess = Array2::<f64>::zeros((dim, dim));
    for col in 0..dim {
        let mut v = vec![0.0; dim];
        v[col] = 1.0;
        let k_a = Array2::from_shape_vec((nva, st.nocc_a), v[..dim_a].to_vec()).unwrap();
        let k_b = Array2::from_shape_vec((nvb, st.nocc_b), v[dim_a..].to_vec()).unwrap();
        let (h_a, h_b) =
            ferric_scf::uhf_newton::hessian_matvec(&ctx, &inp, &k_a, &k_b, &pool).unwrap();
        for (r, x) in h_a.iter().chain(h_b.iter()).enumerate() {
            hess[(r, col)] = *x;
        }
    }
    // Symmetrize: the exact Hessian IS symmetric; the asymmetry is the
    // integral-threshold noise, and its size is reported so a silently broken
    // (genuinely non-symmetric) operator cannot hide.
    let asym = (&hess - &hess.t())
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()));
    eprintln!("  dense Hessian dim = {dim}, max asymmetry = {asym:.3e}");
    assert!(
        asym < 1e-6,
        "orbital Hessian must be symmetric; max |H - Hᵀ| = {asym:e}. A larger \
         asymmetry means the matvec is not a second derivative at all."
    );
    let sym = 0.5 * (&hess + &hess.t());
    let (evals, evecs) = sym.eigh(UPLO::Lower).unwrap();
    let lo = (0..dim).fold(0usize, |b, i| if evals[i] < evals[b] { i } else { b });
    (evals[lo], evecs.column(lo).to_vec())
}

/// Cayley rotation of MO coefficients by an occ→virt block scaled by `eps`.
/// `U = (I − κ/2)⁻¹(I + κ/2)` exactly preserves orthonormality — the same
/// construction `uhf_newton::apply_cayley` uses.
fn rotate(c: &Array2<f64>, k_ov: &Array2<f64>, nocc: usize, eps: f64) -> Array2<f64> {
    let n = c.nrows();
    let mut kappa = Array2::<f64>::zeros((n, n));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            let v = eps * k_ov[(ir, i)];
            kappa[(a, i)] = v;
            kappa[(i, a)] = -v;
        }
    }
    let half = 0.5 * &kappa;
    let eye = Array2::<f64>::eye(n);
    let am = &eye - &half;
    let bm = &eye + &half;
    let mut u = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let sol = am.solve(&bm.column(col).to_owned()).unwrap();
        for row in 0..n {
            u[(row, col)] = sol[row];
        }
    }
    c.dot(&u)
}

/// UHF total electronic + nuclear energy at explicit MOs, used to confirm the
/// negative direction really is downhill in the ENERGY, independently of the
/// Hessian that predicted it.
fn uhf_energy_at(st: &UhfState, c_a: &Array2<f64>, c_b: &Array2<f64>) -> f64 {
    let ctx = ParallelContext::default();
    let n = st.prep.nbasis();
    let h = oneelectron::hcore(&st.prep);
    let d_a = c_a
        .slice(s![.., ..st.nocc_a])
        .dot(&c_a.slice(s![.., ..st.nocc_a]).t());
    let d_b = c_b
        .slice(s![.., ..st.nocc_b])
        .dot(&c_b.slice(s![.., ..st.nocc_b]).t());
    let (f_a, f_b) = uhf_fock(&ctx, &st.prep, &st.bounds, c_a, c_b, st.nocc_a, st.nocc_b);
    let _ = n;
    // E = ½ Σ_σ Tr[D_σ (h + F_σ)] + E_nuc
    let e_el = 0.5 * ((&d_a * &(&h + &f_a)).sum() + (&d_b * &(&h + &f_b)).sum());
    e_el + st.mol.nuclear_repulsion()
}

// ===========================================================================
// 1. EXACTNESS ANCHORS — a genuine minimum must come out STABLE.
// ===========================================================================

/// ANCHOR A1 (H₂ at equilibrium, STO-3G, closed-shell singlet run through the
/// UHF code path). The closed-shell H₂ minimum at r = 0.74 Å is the textbook
/// stable SCF solution. λ_min must be POSITIVE, and must match the lowest
/// eigenvalue of the explicitly built dense Hessian.
///
/// Also anchors the DIAGONAL LIMIT: with only one occupied orbital per spin
/// and a small basis, λ_min must be bounded below by nothing pathological —
/// reported, not asserted, so this test measures rather than merely passing.
#[test]
fn anchor_h2_equilibrium_uhf_is_internally_stable() {
    let st = converge_uhf("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1, "sto-3g");
    let ctx = ParallelContext::default();
    let inp = uhf_inputs(&st);
    let res = uhf_internal_stability(&ctx, &inp, &StabilityConfig::default()).unwrap();
    eprintln!("ANCHOR A1  H2/STO-3G  E = {:.10}", st.energy);
    eprintln!("ANCHOR A1  {}", res.summary());

    let (dense_lo, _) = dense_uhf_hessian_lowest(&st);
    eprintln!(
        "ANCHOR A1  dense lambda_min = {dense_lo:+.10e}  (Davidson {:+.10e})",
        res.lowest_eigenvalue
    );

    assert!(res.converged, "Davidson must converge: {}", res.summary());
    assert_eq!(res.kind, StabilityKind::UhfInternal);
    assert!(
        res.lowest_eigenvalue > 0.0,
        "H2 at equilibrium is a genuine SCF minimum; lambda_min = {:+e} must be \
         positive. A negative value here means the analysis reports UNSTABLE \
         unconditionally and the HeNe+ verdict would be worthless.",
        res.lowest_eigenvalue
    );
    assert!(
        res.is_stable && !res.is_marginal(),
        "H2 at equilibrium must be reported STABLE, not marginal: {}",
        res.summary()
    );
    assert!(
        (res.lowest_eigenvalue - dense_lo).abs() < 1e-6,
        "Davidson lambda_min {:+e} must match the independently built dense \
         Hessian's {:+e} (diff {:e}). A mismatch separates a broken EIGENSOLVER \
         from a broken OPERATOR.",
        res.lowest_eigenvalue,
        dense_lo,
        (res.lowest_eigenvalue - dense_lo).abs()
    );
}

/// ANCHOR A2 (water / STO-3G, closed shell through the UHF path). A second,
/// larger, chemically different stable minimum — so A1 passing cannot be a
/// one-system coincidence. Same two assertions: positive, and equal to dense.
#[test]
fn anchor_water_uhf_is_internally_stable() {
    let st = converge_uhf(
        "3\nwater\nO 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692\n",
        0,
        1,
        "sto-3g",
    );
    let ctx = ParallelContext::default();
    let inp = uhf_inputs(&st);
    let res = uhf_internal_stability(&ctx, &inp, &StabilityConfig::default()).unwrap();
    eprintln!("ANCHOR A2  water/STO-3G  E = {:.10}", st.energy);
    eprintln!("ANCHOR A2  {}", res.summary());

    let (dense_lo, _) = dense_uhf_hessian_lowest(&st);
    eprintln!(
        "ANCHOR A2  dense lambda_min = {dense_lo:+.10e}  (Davidson {:+.10e})",
        res.lowest_eigenvalue
    );
    assert!(res.converged, "Davidson must converge: {}", res.summary());
    assert!(
        res.lowest_eigenvalue > 0.0,
        "water/STO-3G RHF is a genuine minimum; lambda_min = {:+e} must be positive",
        res.lowest_eigenvalue
    );
    assert!(res.is_stable && !res.is_marginal(), "{}", res.summary());
    assert!(
        (res.lowest_eigenvalue - dense_lo).abs() < 1e-6,
        "Davidson {:+e} vs dense {:+e}",
        res.lowest_eigenvalue,
        dense_lo
    );
}

/// ANCHOR A3 (RHF path, water / STO-3G). The RHF singlet-channel Hessian is a
/// DIFFERENT operator from the UHF one (one rotation applied to both spins,
/// not two independent ones), so it needs its own anchor. Positive + matches
/// its own dense build.
#[test]
fn anchor_water_rhf_singlet_channel_is_internally_stable() {
    let mol = Molecule::parse_xyz(
        "3\nwater\nO 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692\n",
        0,
        1,
    )
    .unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    };
    let res_scf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(res_scf.converged);

    let n = prep.nbasis();
    let nocc = mol.nelec() as usize / 2;
    let c = res_scf.mos_alpha.clone();
    let f_mo = c.t().dot(&res_scf.fock_alpha).dot(&c);

    let inp = RhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c: &c,
        f_mo: &f_mo,
        nocc,
        k_mix_sr: 1.0,
        fxc: None,
        thresh: 1e-12,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let res = rhf_internal_stability(&ctx, &inp, &StabilityConfig::default()).unwrap();
    eprintln!("ANCHOR A3  water/STO-3G RHF  E = {:.10}", res_scf.energy);
    eprintln!("ANCHOR A3  {}", res.summary());
    assert_eq!(res.kind, StabilityKind::RhfInternal);
    assert!(res.eigenvector_beta.is_none(), "RHF has one MO set only");

    // Dense cross-check of the RHF operator.
    let nv = n - nocc;
    let dim = nv * nocc;
    let pool = ferric_scf::engine_pool::EnginePool::new(bounds.op, &prep, 1e-14).unwrap();
    let mut hess = Array2::<f64>::zeros((dim, dim));
    for col in 0..dim {
        let mut v = vec![0.0; dim];
        v[col] = 1.0;
        let k = Array2::from_shape_vec((nv, nocc), v).unwrap();
        let h = ferric_scf::rhf_newton::hessian_matvec(&ctx, &inp, &k, &pool).unwrap();
        for (r, x) in h.iter().enumerate() {
            hess[(r, col)] = *x;
        }
    }
    let sym = 0.5 * (&hess + &hess.t());
    let (evals, _) = sym.eigh(UPLO::Lower).unwrap();
    let dense_lo = evals.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!(
        "ANCHOR A3  dense lambda_min = {dense_lo:+.10e}  (Davidson {:+.10e})",
        res.lowest_eigenvalue
    );

    assert!(res.converged, "{}", res.summary());
    assert!(
        res.lowest_eigenvalue > 0.0,
        "water RHF singlet channel must be stable; got {:+e}",
        res.lowest_eigenvalue
    );
    assert!(
        (res.lowest_eigenvalue - dense_lo).abs() < 1e-6,
        "RHF Davidson {:+e} vs dense {:+e}",
        res.lowest_eigenvalue,
        dense_lo
    );
}

// ===========================================================================
// 2. THE DECISIVE TEST — the known saddle must come out UNSTABLE.
// ===========================================================================

/// THE DECISIVE TEST.
///
/// Ferric's own unconstrained UHF on HeNe⁺ / def2-SVP / R = 2.0 Å converges to
/// the ²Π state at `E = -130.5003466413`. PySCF 2.13.0, forced onto the
/// numerically identical state (`E = -130.50034664`, agreement 1.3e-9 Ha) via
/// `scf.addons.mom_occ`, reports `mf.stability()` → `internal stable = False`.
/// PySCF's own default-guess solution is a DIFFERENT state at
/// `E = -130.5053405386`, which it reports internally and externally STABLE.
///
/// This test asserts ferric reaches the same verdict PySCF does on the state
/// ferric actually converges to. A stability analysis that calls this state
/// stable is wrong, and the assertion is written so that outcome FAILS.
#[test]
fn hene_plus_pi_state_is_internally_unstable() {
    /// PySCF 2.13.0, MOM-forced ²Π state, def2-SVP, R = 2.0 Å, conv_tol 1e-11.
    const PYSCF_PI_STATE: f64 = -130.50034664;
    /// PySCF 2.13.0 default-guess solution — internally AND externally stable.
    const PYSCF_STABLE_MIN: f64 = -130.5053405386;
    /// PySCF's OWN orbital Hessian eigenvalue on this state, from
    /// `pyscf.soscf.newton_ah.gen_g_hop_uhf(mf, mo_coeff, mo_occ)` applied to
    /// every unit vector and diagonalized (`scripts/gen_pyscf_hene_stability.py`).
    /// PySCF reports the same 148-dimensional rotation space and
    /// λ_min = -4.8706729960e-3 at |gradient| = 9.0e-9.
    ///
    /// This is the genuinely INDEPENDENT check: not just the boolean verdict,
    /// but the NUMBER, from a separate code base's own Hessian construction.
    const PYSCF_LAMBDA_MIN: f64 = -4.8706729960e-3;

    let st = converge_uhf("2\nHeNe+\nHe 0 0 0\nNe 0 0 2.0\n", 1, 2, "def2-svp");
    eprintln!("DECISIVE  ferric HeNe+ UHF  E = {:.10}", st.energy);
    eprintln!(
        "DECISIVE  PySCF pi state        E = {PYSCF_PI_STATE:.10}  (diff {:.3e})",
        (st.energy - PYSCF_PI_STATE).abs()
    );
    eprintln!("DECISIVE  PySCF stable min      E = {PYSCF_STABLE_MIN:.10}  (ferric is {:.3e} Ha = {:.4} eV above)",
        st.energy - PYSCF_STABLE_MIN, (st.energy - PYSCF_STABLE_MIN) * 27.211386245988);

    // Confirm ferric is on the state PySCF called unstable, not some third
    // state — otherwise the verdict comparison below is meaningless.
    assert!(
        (st.energy - PYSCF_PI_STATE).abs() < 1e-6,
        "ferric must be on PySCF's MOM-forced pi state for the stability verdict \
         comparison to mean anything: ferric {:.10} vs PySCF {PYSCF_PI_STATE:.10}",
        st.energy
    );

    let ctx = ParallelContext::default();
    let inp = uhf_inputs(&st);
    let res = uhf_internal_stability(&ctx, &inp, &StabilityConfig::default()).unwrap();
    eprintln!("DECISIVE  {}", res.summary());

    let (dense_lo, _) = dense_uhf_hessian_lowest(&st);
    eprintln!(
        "DECISIVE  dense lambda_min = {dense_lo:+.10e}  (Davidson {:+.10e})",
        res.lowest_eigenvalue
    );

    assert!(res.converged, "Davidson must converge: {}", res.summary());
    assert!(
        (res.lowest_eigenvalue - dense_lo).abs() < 1e-6,
        "Davidson {:+e} vs dense {:+e} — the verdict must not depend on the \
         eigensolver",
        res.lowest_eigenvalue,
        dense_lo
    );
    assert!(
        res.lowest_eigenvalue < 0.0,
        "THE DECISIVE TEST FAILED. PySCF's mf.stability() reports internal \
         stable = False for this exact state (E agreement 1.3e-9 Ha), and it \
         sits 4.99e-3 Ha above PySCF's stable minimum. lambda_min = {:+e} says \
         ferric thinks it is a minimum. A stability analysis that cannot see \
         THIS saddle has no value.",
        res.lowest_eigenvalue
    );
    assert!(
        !res.is_stable,
        "verdict must be UNSTABLE: {}",
        res.summary()
    );
    assert!(
        !res.is_marginal(),
        "this instability must be well clear of the noise floor, not marginal: {}",
        res.summary()
    );

    // THE INDEPENDENT NUMERIC CHECK. Matching PySCF's boolean verdict is good;
    // matching PySCF's own orbital-Hessian EIGENVALUE, from a separate code
    // base's independent construction, is what makes this a measurement rather
    // than an agreement of conventions.
    eprintln!(
        "DECISIVE  PySCF gen_g_hop_uhf lambda_min = {PYSCF_LAMBDA_MIN:+.10e} \
         (ferric {:+.10e}, diff {:.3e})",
        res.lowest_eigenvalue,
        (res.lowest_eigenvalue - PYSCF_LAMBDA_MIN).abs()
    );
    assert!(
        (res.lowest_eigenvalue - PYSCF_LAMBDA_MIN).abs() < 1e-8,
        "ferric lambda_min {:+e} must match PySCF's own orbital-Hessian lowest \
         eigenvalue {PYSCF_LAMBDA_MIN:+e} (diff {:e}). Both are built on the \
         same 148-dimensional UHF rotation space by independent code.",
        res.lowest_eigenvalue,
        (res.lowest_eigenvalue - PYSCF_LAMBDA_MIN).abs()
    );
}

// ===========================================================================
// 3. FOLLOW THE INSTABILITY — the eigenVECTOR, not just its sign.
// ===========================================================================

/// Rotate along the negative-eigenvalue direction and re-converge. Reaching a
/// LOWER energy proves the eigenvector is a real downhill direction rather
/// than a sign artifact — a spurious negative mode would not lead anywhere
/// better.
///
/// Two separate claims are checked, and they fail for different reasons:
///   (a) the energy DIRECTLY at the rotated (unconverged) orbitals is already
///       below the saddle for small rotations — the pure Hessian statement,
///       independent of any re-convergence;
///   (b) re-converging from there reaches a strictly lower stationary point,
///       which should be PySCF's -130.50534054 stable minimum.
#[test]
fn hene_plus_following_the_instability_lowers_the_energy() {
    const PYSCF_STABLE_MIN: f64 = -130.5053405386;

    let st = converge_uhf("2\nHeNe+\nHe 0 0 0\nNe 0 0 2.0\n", 1, 2, "def2-svp");
    let ctx = ParallelContext::default();
    let inp = uhf_inputs(&st);
    let res = uhf_internal_stability(&ctx, &inp, &StabilityConfig::default()).unwrap();
    assert!(
        res.converged && res.lowest_eigenvalue < 0.0,
        "{}",
        res.summary()
    );
    eprintln!("FOLLOW    saddle E = {:.10}   {}", st.energy, res.summary());

    let k_a = &res.eigenvector_alpha;
    let k_b = res.eigenvector_beta.as_ref().unwrap();

    // Report which spin channel carries the mode (diagnostic: the HeNe+ hole
    // is a beta hole, so the instability is expected to be beta-dominated).
    let na2: f64 = k_a.iter().map(|x| x * x).sum();
    let nb2: f64 = k_b.iter().map(|x| x * x).sum();
    eprintln!("FOLLOW    mode weight: alpha = {na2:.4}, beta = {nb2:.4}");

    // (a) The pure Hessian statement: E(εκ) < E(0) for a small step along the
    //     negative direction, with no re-convergence involved.
    let mut best_direct = f64::INFINITY;
    for &eps in &[0.05_f64, 0.1, 0.2, 0.4] {
        let ca = rotate(&st.c_a, k_a, st.nocc_a, eps);
        let cb = rotate(&st.c_b, k_b, st.nocc_b, eps);
        let e = uhf_energy_at(&st, &ca, &cb);
        eprintln!(
            "FOLLOW    direct rotation eps={eps:.2}: E = {e:.10}  (dE = {:+.3e})",
            e - st.energy
        );
        best_direct = best_direct.min(e);
    }
    assert!(
        best_direct < st.energy - 1e-8,
        "rotating along the NEGATIVE Hessian eigenvector must lower the energy \
         without any re-convergence: best {best_direct:.10} vs saddle {:.10}. \
         If this fails the eigenvector is not a downhill direction and the \
         sign of lambda_min is not evidence of anything.",
        st.energy
    );

    // (b) Re-converge from the rotated orbitals and check we land strictly
    //     lower — ideally on PySCF's stable minimum.
    //
    // MEASURED, and worth recording because it is a real property of the
    // basin rather than a tuning detail: SMALL steps along the correct
    // downhill direction are NOT enough. At eps = 0.2 and 0.5 the SCF falls
    // straight back to the saddle (dE = 0 to 6e-14), even though (a) above
    // proves those same rotations lower the energy. The saddle's DIIS basin is
    // wide enough to recapture a small displacement. From eps >= 1.0 rad the
    // re-converged energy is -130.5053405386 every time — PySCF's stable
    // minimum to ~2e-11 Ha. The ladder therefore spans both regimes on
    // purpose; a ladder that only probed eps <= 0.5 would have concluded,
    // wrongly, that the eigenvector was useless.
    let mut best_reconverged = f64::INFINITY;
    for &eps in &[0.2_f64, 0.5, 1.0, std::f64::consts::FRAC_PI_2, 2.0, 3.0] {
        let ca = rotate(&st.c_a, k_a, st.nocc_a, eps);
        let cb = rotate(&st.c_b, k_b, st.nocc_b, eps);
        let st2 = converge_uhf_from(
            "2\nHeNe+\nHe 0 0 0\nNe 0 0 2.0\n",
            1,
            2,
            "def2-svp",
            Some((ca, cb)),
        );
        eprintln!(
            "FOLLOW    re-converged from eps={eps:.2}: E = {:.10}  (dE vs saddle {:+.3e}, vs PySCF min {:+.3e})",
            st2.energy,
            st2.energy - st.energy,
            st2.energy - PYSCF_STABLE_MIN
        );
        best_reconverged = best_reconverged.min(st2.energy);
    }
    assert!(
        best_reconverged < st.energy - 1e-6,
        "re-converging from the rotated orbitals must reach a STRICTLY LOWER \
         stationary point: best {best_reconverged:.10} vs saddle {:.10}",
        st.energy
    );
    assert!(
        (best_reconverged - PYSCF_STABLE_MIN).abs() < 1e-5,
        "following the instability should reach PySCF's stable minimum \
         {PYSCF_STABLE_MIN:.10}; got {best_reconverged:.10} (diff {:.3e})",
        (best_reconverged - PYSCF_STABLE_MIN).abs()
    );

    // And the state we landed on must itself be STABLE — closing the loop:
    // the analysis says UNSTABLE on the saddle and STABLE on the minimum it
    // was used to find. That is both verdicts, on the SAME system, from the
    // SAME code path, which no always-one-answer checker can produce.
    let ca = rotate(&st.c_a, k_a, st.nocc_a, std::f64::consts::FRAC_PI_2);
    let cb = rotate(&st.c_b, k_b, st.nocc_b, std::f64::consts::FRAC_PI_2);
    let st_min = converge_uhf_from(
        "2\nHeNe+\nHe 0 0 0\nNe 0 0 2.0\n",
        1,
        2,
        "def2-svp",
        Some((ca, cb)),
    );
    let inp_min = uhf_inputs(&st_min);
    let res_min = uhf_internal_stability(&ctx, &inp_min, &StabilityConfig::default()).unwrap();
    eprintln!(
        "FOLLOW    minimum E = {:.10}  {}",
        st_min.energy,
        res_min.summary()
    );
    assert!(res_min.converged, "{}", res_min.summary());
    assert!(
        res_min.lowest_eigenvalue > 0.0 && res_min.is_stable,
        "the state reached by following the instability must itself be STABLE \
         (PySCF reports internal AND external stable for it): {}",
        res_min.summary()
    );
}

// ===========================================================================
// 4. REACHABILITY OF THE PASS CONDITION.
// ===========================================================================

/// A gate whose pass condition cannot be violated returns arithmetic, not
/// measurement. This test proves BOTH verdicts are reachable from the SAME
/// code path within one test body, on real converged SCF states — so
/// `is_stable` is genuinely a function of the physics and not a constant.
#[test]
fn both_verdicts_are_reachable_from_the_same_code_path() {
    let ctx = ParallelContext::default();
    let cfg = StabilityConfig::default();

    let stable = converge_uhf("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1, "sto-3g");
    let r_stable = uhf_internal_stability(&ctx, &uhf_inputs(&stable), &cfg).unwrap();

    let saddle = converge_uhf("2\nHeNe+\nHe 0 0 0\nNe 0 0 2.0\n", 1, 2, "def2-svp");
    let r_saddle = uhf_internal_stability(&ctx, &uhf_inputs(&saddle), &cfg).unwrap();

    eprintln!(
        "REACHABLE  H2   -> is_stable = {}  lambda_min = {:+.6e}",
        r_stable.is_stable, r_stable.lowest_eigenvalue
    );
    eprintln!(
        "REACHABLE  HeNe+-> is_stable = {}  lambda_min = {:+.6e}",
        r_saddle.is_stable, r_saddle.lowest_eigenvalue
    );

    assert!(
        r_stable.is_stable,
        "the STABLE verdict must be reachable; H2 gave {:+e}",
        r_stable.lowest_eigenvalue
    );
    assert!(
        !r_saddle.is_stable,
        "the UNSTABLE verdict must be reachable; HeNe+ gave {:+e}",
        r_saddle.lowest_eigenvalue
    );
    assert_ne!(
        r_stable.is_stable, r_saddle.is_stable,
        "is_stable returned the same answer for a known minimum and a known \
         saddle — it is a constant, not a measurement"
    );
}

// ===========================================================================
// 5. THE EIGENSOLVER ITSELF.
// ===========================================================================

/// The eigensolver is the one component whose silent failure would defeat the
/// whole module (an iteration-starved Davidson returning a wrong λ_min). Two
/// properties are pinned on a synthetic operator with an EXACTLY KNOWN
/// spectrum, so a regression here is caught without running any SCF:
///   * it finds the true lowest eigenvalue of a known matrix;
///   * when starved of iterations it reports `converged = false` with a large
///     residual, rather than a confident wrong answer.
#[test]
fn davidson_finds_the_known_lowest_eigenvalue_and_reports_starvation() {
    use ferric_scf::stability::davidson_lowest;

    // Diagonally dominant symmetric matrix with a planted lowest eigenvalue:
    // A = diag(d) + small symmetric off-diagonal coupling.
    let dim = 60;
    let mut a = Array2::<f64>::zeros((dim, dim));
    for i in 0..dim {
        a[(i, i)] = -0.35 + i as f64 * 0.5; // A_00 = -0.35 is the target region
        for j in 0..i {
            let v = 0.01 / (1.0 + (i as f64 - j as f64).abs());
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
    }
    let (evals, _) = a.eigh(UPLO::Lower).unwrap();
    let exact_lo = evals.iter().cloned().fold(f64::INFINITY, f64::min);
    let diag: Vec<f64> = (0..dim).map(|i| a[(i, i)]).collect();

    let mv = |v: &[f64]| -> Result<Vec<f64>, ferric_core::FerricError> {
        let x = ndarray::Array1::from(v.to_vec());
        Ok(a.dot(&x).to_vec())
    };

    let cfg = StabilityConfig::default();
    let r = davidson_lowest(dim, &mv, &diag, &cfg).unwrap();
    eprintln!(
        "EIGENSOLVER  exact lambda_min = {exact_lo:+.12e}, Davidson = {:+.12e} \
         (resid {:.2e}, {} iters, converged {})",
        r.eigenvalue, r.residual, r.iterations, r.converged
    );
    assert!(
        r.converged,
        "Davidson must converge on a well-conditioned 60x60"
    );
    assert!(
        (r.eigenvalue - exact_lo).abs() < 1e-8,
        "Davidson lambda_min {:+e} must match the exact {exact_lo:+e}",
        r.eigenvalue
    );
    assert!(
        r.residual < cfg.conv_thresh,
        "a converged=true result must carry a residual below the threshold: \
         {:.3e} vs {:.3e}",
        r.residual,
        cfg.conv_thresh
    );

    // Starvation: one iteration cannot resolve this. The solver must SAY so.
    let starved = StabilityConfig {
        max_iter: 1,
        ..StabilityConfig::default()
    };
    let rs = davidson_lowest(dim, &mv, &diag, &starved).unwrap();
    eprintln!(
        "EIGENSOLVER  starved (max_iter=1): lambda = {:+.6e}, resid = {:.3e}, converged = {}",
        rs.eigenvalue, rs.residual, rs.converged
    );
    assert!(
        !rs.converged,
        "an iteration-starved Davidson must report converged = false, not a \
         confident wrong answer — this flag is the only thing standing between \
         a starved eigensolve and a silently wrong stability verdict"
    );
    assert!(
        rs.residual > cfg.conv_thresh,
        "a converged = false result must carry a residual ABOVE the threshold \
         (got {:.3e}); otherwise the flag and the residual disagree",
        rs.residual
    );
    // Rayleigh-Ritz is variational: even starved, the estimate is an UPPER
    // bound on the truth. This is what makes an unconverged NEGATIVE verdict
    // still meaningful (documented on StabilityResult::converged).
    assert!(
        rs.eigenvalue >= exact_lo - 1e-10,
        "the Rayleigh-Ritz estimate {:+e} must be an upper bound on the true \
         lowest eigenvalue {exact_lo:+e}",
        rs.eigenvalue
    );
}

/// The empty rotation space must be a clean error, not a panic or a fabricated
/// verdict. (He atom / STO-3G: one occupied, one basis function ⇒ zero
/// virtuals ⇒ nothing to rotate.)
#[test]
fn empty_rotation_space_is_a_clean_error() {
    let st = converge_uhf("1\nHe\nHe 0 0 0\n", 0, 1, "sto-3g");
    let ctx = ParallelContext::default();
    let inp = uhf_inputs(&st);
    let r = uhf_internal_stability(&ctx, &inp, &StabilityConfig::default());
    match r {
        Err(e) => eprintln!("EMPTY  clean error as required: {e}"),
        Ok(res) => panic!(
            "He/STO-3G has no virtual orbitals; stability is undefined and must \
             be an Err, not a fabricated verdict ({})",
            res.summary()
        ),
    }
}

// ===========================================================================
// 6. REGRESSION TESTS FOR THE TWO EIGENSOLVER BUGS FOUND WHILE BUILDING THIS.
// ===========================================================================
//
// Both were caught by the water/STO-3G ANCHOR, before any HeNe⁺ number was
// believed — which is the entire argument for writing the exactness anchor
// first. Both produced `converged = true` on a WRONG eigenvalue, so neither
// would have been caught by checking the residual of the returned root. They
// are pinned here on the real operator, because a default quietly reverted is
// a silent regression of the stability verdict itself.

/// BUG 1 (single-root Davidson converges inside a SYMMETRY BLOCK).
///
/// The orbital Hessian of a symmetric molecule is block diagonal in the
/// irreps. On water/STO-3G through the UHF path, `H·e_4` has support only on
/// indices {4, 14} — a closed 2×2 block whose lowest root is +3.688705e-1,
/// while the true λ_min is +3.625617e-1 in a DIFFERENT block. Handed a unit
/// vector inside that block, single-root Davidson converges to residual 6e-11
/// and reports `converged = true` on +3.688705e-1: a genuinely converged
/// eigenpair that is not the lowest one. No residual check on the returned
/// root can detect this, which is what makes it dangerous.
///
/// The failure needs BOTH ingredients, and the parameter sweep below is what
/// established that rather than assumption:
///
/// ```text
///   n_block=1 n_seed=1  -> +3.6256168663e-1  OK    (19 iters)
///   n_block=1 n_seed=2  -> +3.6887050762e-1  WRONG ( 1 iter)
///   n_block=1 n_seed=8  -> +3.6887050762e-1  WRONG ( 1 iter)
///   n_block=2 n_seed=1  -> +3.6256168663e-1  OK    (11 iters)
///   n_block=6 n_seed=8  -> +3.6256168663e-1  OK    ( 7 iters)
/// ```
///
/// i.e. the smallest-diagonal UNIT-VECTOR seeds are what hand the solver a
/// block-local eigenvector in the first place (they are a convergence
/// accelerator, and with `n_block = 1` they are actively harmful), and
/// `n_block >= 2` is what makes the result correct regardless. That is the
/// justification for the `n_block` default, and this test pins it.
#[test]
fn davidson_single_root_converges_inside_a_symmetry_block() {
    let st = converge_uhf(
        "3\nwater\nO 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692\n",
        0,
        1,
        "sto-3g",
    );
    let ctx = ParallelContext::default();
    let inp = uhf_inputs(&st);
    let (dense_lo, _) = dense_uhf_hessian_lowest(&st);

    // The exact configuration that fails: one tracked root, plus the
    // unit-vector seeds that put a block-local eigenvector in the subspace.
    let trap = StabilityConfig {
        n_block: 1,
        n_seed: 8,
        ..StabilityConfig::default()
    };
    let bad = uhf_internal_stability(&ctx, &inp, &trap).unwrap();
    eprintln!(
        "REGRESSION-1  n_block=1,n_seed=8: lambda = {:+.10e}, resid = {:.3e}, \
         converged = {}, iters = {}  (TRUE lambda_min = {dense_lo:+.10e})",
        bad.lowest_eigenvalue, bad.residual, bad.converged, bad.iterations
    );

    // PREMISE CHECK — the pass condition must be REACHABLE in both directions.
    // If single-root Davidson ever stops missing this root, the n_block
    // default's justification has changed and must be re-derived, not silently
    // inherited.
    assert!(
        bad.lowest_eigenvalue > dense_lo + 1e-9,
        "PREMISE CHECK: n_block=1 was expected to miss the true lowest root on \
         this symmetry-blocked system (that is why n_block defaults to > 1). It \
         returned {:+e} vs true {dense_lo:+e}. If the trap is genuinely gone, \
         re-derive the default rather than deleting this test.",
        bad.lowest_eigenvalue
    );
    assert!(
        bad.converged && bad.residual < trap.conv_thresh,
        "and it reported converged = true (resid {:.3e}) on that wrong value — \
         which is precisely why the `converged` flag alone cannot detect a \
         missed root, and why the whole BLOCK must converge",
        bad.residual
    );

    // The defended default must get it right.
    let good = uhf_internal_stability(&ctx, &inp, &StabilityConfig::default()).unwrap();
    eprintln!(
        "REGRESSION-1  default:            lambda = {:+.10e}, resid = {:.3e}, \
         converged = {}, iters = {}",
        good.lowest_eigenvalue, good.residual, good.converged, good.iterations
    );
    assert!(
        (good.lowest_eigenvalue - dense_lo).abs() < 1e-9,
        "the DEFAULT config must find the true lowest root {dense_lo:+e}, got {:+e}",
        good.lowest_eigenvalue
    );
    assert!(
        StabilityConfig::default().n_block >= 2,
        "the n_block default must stay >= 2: n_block = 1 silently returns the \
         wrong lambda_min on this system with converged = true"
    );
}

/// BUG 2 (collapse-regrow deadlock on a small rotation space).
///
/// water/STO-3G RHF has only nv × nocc = 2 × 5 = 10 rotation dimensions. A
/// collapse rule that fires on `m + n_block > max_sub` while retaining
/// `max(n_block, n_seed) = 8` vectors pins the basis at 8 forever: every
/// iteration collapses and regrows the identical subspace, burning all 100
/// iterations while the lowest root sits converged at 5e-16 and a higher block
/// root stalls at 5.3e-2.
///
/// The fix is that a collapse must leave room for at least one new expansion
/// vector, and that `m >= dim` is EXACT convergence (the Ritz values are the
/// eigenvalues) rather than a stall. This test pins both: the RHF analysis must
/// converge in a small number of iterations, not burn the cap.
#[test]
fn small_rotation_space_converges_without_collapse_deadlock() {
    let mol = Molecule::parse_xyz(
        "3\nwater\nO 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692\n",
        0,
        1,
    )
    .unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    };
    let res_scf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let nocc = mol.nelec() as usize / 2;
    let c = res_scf.mos_alpha.clone();
    let f_mo = c.t().dot(&res_scf.fock_alpha).dot(&c);
    let inp = RhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c: &c,
        f_mo: &f_mo,
        nocc,
        k_mix_sr: 1.0,
        fxc: None,
        thresh: 1e-12,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let scfg = StabilityConfig::default();
    let r = rhf_internal_stability(&ctx, &inp, &scfg).unwrap();
    let dim = (prep.nbasis() - nocc) * nocc;
    eprintln!(
        "REGRESSION-2  rotation-space dim = {dim} (< n_seed = {}), converged in {} iters (cap {}), resid {:.3e}",
        scfg.n_seed, r.iterations, scfg.max_iter, r.residual
    );
    assert!(
        dim < scfg.n_seed.max(scfg.n_block) * 2,
        "PREMISE CHECK: this test needs a rotation space small enough to \
         trigger the collapse-regrow path (dim = {dim}, n_seed = {}, \
         n_block = {}). If the defaults grew, pick a smaller system.",
        scfg.n_seed,
        scfg.n_block
    );
    assert!(r.converged, "{}", r.summary());
    assert!(
        r.iterations < scfg.max_iter / 4,
        "a {dim}-dimensional eigenproblem must not burn {} of {} iterations — \
         that is the collapse-regrow deadlock, in which the basis is pinned at \
         a fixed size and the same subspace is rebuilt every iteration",
        r.iterations,
        scfg.max_iter
    );
}
