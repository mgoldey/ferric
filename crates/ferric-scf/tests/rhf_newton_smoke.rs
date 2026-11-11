//! Newton-accelerated RHF / RKS (closed-shell): correctness + engagement tests.
//!
//! The closed-shell analogue of `uhf_newton_smoke.rs` / `rohf_newton_smoke.rs`.
//! Covers:
//!   1. RHF (HF, no XC): Newton must reach the same energy as DIIS-only.
//!   2. RKS/PBE: the Newton path must engage a *GGA* f_xc kernel (not silently
//!      fall back to DIIS) and reach the same energy as DIIS-only.
//!   3. Hessian-matvec finite-difference check: at a converged RHF state, the
//!      analytic H·κ must match the central difference of the orbital gradient
//!      g(κ) = F_ai(κ) along a random rotation.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Deterministic xorshift64 PRNG (no external rand dep).
struct Xorshift64(u64);
impl Xorshift64 {
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n",
        0,
        1,
    )
    .unwrap()
}

/// RHF (no XC): Newton-accelerated run must reach the same energy as DIIS-only
/// on closed-shell water/cc-pVDZ.
#[test]
fn rhf_newton_water_matches_diis_only() {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let cfg_newton = RhfConfig {
        newton_trigger: 1e-2,
        ..cfg_diis.clone()
    };

    let r_diis = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();
    let r_newton = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_newton).unwrap();

    eprintln!(
        "RHF/H2O  DIIS:   E = {:.10}, iters = {}",
        r_diis.energy, r_diis.iterations
    );
    eprintln!(
        "RHF/H2O  Newton: E = {:.10}, iters = {}",
        r_newton.energy, r_newton.iterations
    );

    assert!(r_diis.converged && r_newton.converged);
    assert!(
        (r_diis.energy - r_newton.energy).abs() < 1e-8,
        "RHF Newton must match DIIS energy: ΔE = {:.3e}",
        (r_diis.energy - r_newton.energy).abs()
    );
}

/// RKS/PBE: the Newton path must engage a GGA f_xc kernel (proved via the
/// `GGA_FXC_KERNEL_BUILDS` counter) and reach the same energy as DIIS-only.
#[test]
fn rks_pbe_newton_engages_gga_fxc_and_matches_diis() {
    use std::sync::atomic::Ordering;

    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    // Root cause of the historical flake here: plain Pulay DIIS with NO
    // level shift genuinely does not converge RKS/PBE water/cc-pVDZ within
    // 200 iterations — it settles into a textbook oscillating limit cycle
    // (traced via FERRIC_SCF_TRACE: dp_rms repeatedly drains to ~1e-10 then
    // blows back up to ~1e-2 on a ~33-iteration period, deterministically,
    // every run — not a race). This is the same class of DIIS oscillation
    // `rhf_level_shift_converges_cose` regression-guards for RHF; a modest
    // virtual-block level shift (ramped to 0 at convergence, see
    // `RhfConfig::level_shift` doc) is the established cure and does not
    // perturb the converged energy: DIIS-with-shift here reaches
    // -76.3334866645, matching the Newton/GGA-fxc path (which does not use
    // this config's `level_shift` for anything but its own damping) to
    // 2.1e-8, and matching plain unassisted Newton's -76.3334866667 to 2e-9.
    let cfg_diis = RhfConfig {
        xc: Some("PBE".into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        level_shift: 0.2,
        ..Default::default()
    };
    let cfg_newton = RhfConfig {
        newton_trigger: 1e-2,
        ..cfg_diis.clone()
    };

    let r_diis = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();

    let before = ferric_scf::rohf::GGA_FXC_KERNEL_BUILDS.load(Ordering::Relaxed);
    let r_newton = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_newton).unwrap();
    let after = ferric_scf::rohf::GGA_FXC_KERNEL_BUILDS.load(Ordering::Relaxed);
    let gga_builds = after.saturating_sub(before);

    eprintln!(
        "RKS/H2O/PBE  DIIS:   E = {:.10}, iters = {}",
        r_diis.energy, r_diis.iterations
    );
    eprintln!(
        "RKS/H2O/PBE  Newton: E = {:.10}, iters = {}, GGA-fxc builds = {}",
        r_newton.energy, r_newton.iterations, gga_builds
    );

    assert!(r_diis.converged, "DIIS-only RKS/PBE must converge");
    assert!(r_newton.converged, "Newton+GGA-fxc RKS/PBE must converge");
    assert!(
        gga_builds >= 1,
        "RKS Newton must engage the GGA f_xc kernel for PBE (got {gga_builds}); \
         it must not silently fall back to DIIS"
    );
    assert!(
        (r_diis.energy - r_newton.energy).abs() < 1e-7,
        "RKS Newton+GGA-fxc must match DIIS energy: ΔE = {:.3e}",
        (r_diis.energy - r_newton.energy).abs()
    );
}

/// Hessian-matvec finite-difference check.
///
/// At a converged RHF state, rotate the MOs by ε·κ (random antisymmetric occ→virt
/// block), rebuild the Fock, and read the occ→virt gradient block g(ε). The
/// central difference [g(ε) − g(−ε)]/(2ε) must match the analytic Hessian-vector
/// product `rhf_newton::hessian_matvec`.
#[test]
fn rhf_hessian_matvec_matches_finite_difference() {
    use ferric_integrals::oneelectron;
    use ferric_scf::rhf::build_jk;
    use ndarray::Array2;
    use ndarray_linalg::Solve;

    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(res.converged);

    let n = prep.nbasis();
    let h = oneelectron::hcore(&prep);
    let c = res.mos_alpha.clone();
    let nocc = mol.nelec() as usize / 2;

    // Occ→virt gradient g_{ai} = F_{ai} at MOs c, via the restricted Fock
    // F = H + J[2·D_c] − ½·K[2·D_c], with D_c = C_occ·C_occᵀ.
    let grad_at = |cc: &Array2<f64>| -> Array2<f64> {
        let cocc = cc.slice(ndarray::s![.., ..nocc]);
        let dtot = 2.0 * cocc.dot(&cocc.t());
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        build_jk(&ctx, &prep, &bounds, 1e-12, &dtot, &mut j, &mut k).unwrap();
        let f = &h + &j - &(0.5 * &k);
        let f_mo = cc.t().dot(&f).dot(cc);
        let nv = n - nocc;
        let mut o = Array2::<f64>::zeros((nv, nocc));
        for (ir, a) in (nocc..n).enumerate() {
            for i in 0..nocc {
                o[(ir, i)] = f_mo[(a, i)];
            }
        }
        o
    };

    // Random antisymmetric rotation direction (occ→virt block).
    let mut rng = Xorshift64(0x9E3779B97F4A7C15);
    let nv = n - nocc;
    let k_dir = Array2::<f64>::from_shape_fn((nv, nocc), |_| 0.01 * rng.next_f64());

    // Apply Cayley rotation for the occ→virt block to C.
    let rotate = |c: &Array2<f64>, k_ov: &Array2<f64>, eps: f64| -> Array2<f64> {
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
    };

    // Analytic H·κ from the solver's own matvec.
    let f_mo = {
        let cocc = c.slice(ndarray::s![.., ..nocc]);
        let dtot = 2.0 * cocc.dot(&cocc.t());
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        build_jk(&ctx, &prep, &bounds, 1e-12, &dtot, &mut j, &mut k).unwrap();
        let f = &h + &j - &(0.5 * &k);
        c.t().dot(&f).dot(&c)
    };

    let inputs = ferric_scf::rhf_newton::RhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c: &c,
        f_mo: &f_mo,
        nocc,
        k_mix_sr: 1.0, // pure HF
        rsh: None,
        fxc: None,
        thresh: 1e-12,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let pool = ferric_scf::engine_pool::EnginePool::new(bounds.op, &prep, 1e-14).unwrap();
    let hk = ferric_scf::rhf_newton::hessian_matvec(&ctx, &inputs, &k_dir, &pool).unwrap();

    // Central-difference the orbital gradient along κ:  [g(εκ) − g(−εκ)]/(2ε) → H·κ.
    let fd_hk = |eps: f64| -> Array2<f64> {
        let cp = rotate(&c, &k_dir, eps);
        let cm = rotate(&c, &k_dir, -eps);
        &(&grad_at(&cp) - &grad_at(&cm)) / (2.0 * eps)
    };

    let fro = |a: &Array2<f64>| -> f64 { a.iter().map(|&x| x * x).sum::<f64>().sqrt() };
    let fro_diff = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    };
    let scale = fro(&hk).max(1e-30);

    let eps_ladder = [1e-3, 5e-4, 2.5e-4, 1.25e-4];
    let mut rels = Vec::new();
    for &eps in &eps_ladder {
        let fa = fd_hk(eps);
        let rel = fro_diff(&hk, &fa) / scale;
        eprintln!("RHF H·κ analytic-vs-FD: eps={eps:.2e}  rel_err={rel:.3e}");
        rels.push(rel);
    }
    // The Hessian matvec is the EXACT analytic derivative of the orbital
    // gradient, so its central-difference residual sits at the FD round-off floor
    // for every ε. Require tight agreement at every ε on the ladder.
    let worst = rels.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        worst < 1e-6,
        "RHF analytic H·κ must match the gradient central difference at the FD \
         round-off floor for every ε: worst rel err {worst:e} (want < 1e-6). \
         ladder={rels:?}"
    );
}

/// Converged water/cc-pVDZ RHF plus the MO-basis Fock rebuilt at its MOs: the
/// inputs a Newton step sees at a converged point.
fn converged_water_rhf_inputs() -> (
    Molecule,
    PreparedBasis,
    SchwarzBounds,
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    usize,
) {
    use ferric_integrals::oneelectron;
    use ferric_scf::rhf::build_jk;
    use ndarray::Array2;

    let mol = water();
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
    let c = res.mos_alpha.clone();
    let cocc = c.slice(ndarray::s![.., ..nocc]);
    let dtot = 2.0 * cocc.dot(&cocc.t());
    let mut j = Array2::<f64>::zeros((n, n));
    let mut k = Array2::<f64>::zeros((n, n));
    build_jk(&ctx, &prep, &bounds, 1e-12, &dtot, &mut j, &mut k).unwrap();
    let f = &oneelectron::hcore(&prep) + &j - &(0.5 * &k);
    let f_mo = c.t().dot(&f).dot(&c);
    (mol, prep, bounds, c, f_mo, nocc)
}

/// A Newton step whose orbital gradient is already below `cg_conv` returns the
/// diagonal step κ⁰ = −g/gap and never reads a Hessian-vector product, so it
/// must not compute one (each costs a full J/K build plus the f_xc response).
///
/// Counts singlet HVPs on this thread through the `hvp_calls_on_this_thread`
/// seam. Positive control: the same inputs with `cg_conv = 0` (CG forced) must
/// count at least one HVP, so a seam that never counts cannot pass.
///
/// Mutation (applied and reverted, 2026-11-11): restoring the unconditional
/// `h0 = hessian_matvec(κ⁰)` before the CG loop makes the converged-point count
/// 1 and this test fails at the `== 0` assert.
#[test]
fn rhf_newton_step_computes_no_hvp_at_a_converged_point() {
    use ferric_scf::rhf_newton::{hvp_calls_on_this_thread, rhf_newton_step, RhfNewtonInputs};

    let (_mol, prep, bounds, c, f_mo, nocc) = converged_water_rhf_inputs();
    let ctx = ParallelContext::default();
    let n = prep.nbasis();
    let cg_conv = 1e-7; // the value solve_rhf passes
    let gmax = (nocc..n)
        .flat_map(|a| (0..nocc).map(move |i| (a, i)))
        .fold(0.0f64, |m, (a, i)| m.max(f_mo[(a, i)].abs()));
    assert!(
        gmax < cg_conv,
        "precondition: converged gradient {gmax:e} must sit below cg_conv {cg_conv:e}"
    );
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

    let before = hvp_calls_on_this_thread();
    let (c_conv, _) = rhf_newton_step(&ctx, &inputs, None, 1e-6, 0.2, 20, cg_conv).unwrap();
    let at_converged = hvp_calls_on_this_thread() - before;
    assert_eq!(
        at_converged, 0,
        "a step with |g|max {gmax:e} < cg_conv {cg_conv:e} uses only κ⁰ and must compute no HVP"
    );
    assert!(c_conv.iter().all(|v| v.is_finite()));

    let before = hvp_calls_on_this_thread();
    rhf_newton_step(&ctx, &inputs, None, 1e-6, 0.2, 20, 0.0).unwrap();
    let forced = hvp_calls_on_this_thread() - before;
    assert!(
        forced >= 1,
        "positive control: a forced CG solve must count its HVPs (got {forced})"
    );
}

/// Cayley rotation of `c` by `eps·κ` (occ→virt block), as `apply_cayley` does.
fn cayley_rotate(
    c: &ndarray::Array2<f64>,
    k_ov: &ndarray::Array2<f64>,
    eps: f64,
    nocc: usize,
) -> ndarray::Array2<f64> {
    use ndarray::Array2;
    use ndarray_linalg::Solve;
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

/// The occ→virt block (rows virt, cols occ) of a square MO matrix.
fn ov_block(m: &ndarray::Array2<f64>, nocc: usize) -> ndarray::Array2<f64> {
    m.slice(ndarray::s![nocc.., ..nocc]).to_owned()
}

fn frob(a: &ndarray::Array2<f64>) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Fitted RHF-like Fock pieces at MOs `c`: `(F, |H| + |J| + ½·k_mix·|K|)` with
/// `F = H + J_RI(D) − ½·k_mix·K_RI(D)`, `D = 2·C_occ·C_occᵀ`. The second matrix
/// is the elementwise magnitude sum the rounding bound reads.
fn fitted_fock(
    h: &ndarray::Array2<f64>,
    dfj: &mut ferric_scf::df_j::DfJ<'_>,
    dfk: &mut ferric_scf::df_k::DfK<'_>,
    k_mix: f64,
    c: &ndarray::Array2<f64>,
    nocc: usize,
) -> (ndarray::Array2<f64>, ndarray::Array2<f64>) {
    use ferric_scf::fock::{JBuilder, KBuilder};
    use ndarray::Array2;
    let n = c.nrows();
    let cocc = c.slice(ndarray::s![.., ..nocc]);
    let d = 2.0 * cocc.dot(&cocc.t());
    let mut j = Array2::<f64>::zeros((n, n));
    dfj.build(&d, &mut j).unwrap();
    let mut k = Array2::<f64>::zeros((n, n));
    if k_mix != 0.0 {
        dfk.build(&d, &mut k).unwrap();
    }
    let f = h + &j - &(0.5 * k_mix * &k);
    let mag = h.mapv(f64::abs) + j.mapv(f64::abs) + (0.5 * k_mix.abs()) * k.mapv(f64::abs);
    (f, mag)
}

/// Rotate `c` within its occupied and its virtual blocks so that the fitted
/// Fock at `c` is diagonal in both (semicanonical). The diagonal-gap Hessian
/// in `rhf_newton` is then the exact second derivative at `c` for ANY occupied
/// space, converged or not: g'(κ) = [δF]_ai + Σ_b F_ab κ_bi − Σ_j κ_aj F_ji.
fn semicanonical(
    f: &ndarray::Array2<f64>,
    c: &ndarray::Array2<f64>,
    nocc: usize,
) -> ndarray::Array2<f64> {
    use ndarray_linalg::{Eigh, UPLO};
    let f_mo = c.t().dot(f).dot(c);
    let (_, uo) = f_mo
        .slice(ndarray::s![..nocc, ..nocc])
        .to_owned()
        .eigh(UPLO::Lower)
        .unwrap();
    let (_, uv) = f_mo
        .slice(ndarray::s![nocc.., nocc..])
        .to_owned()
        .eigh(UPLO::Lower)
        .unwrap();
    let mut out = c.clone();
    out.slice_mut(ndarray::s![.., ..nocc])
        .assign(&c.slice(ndarray::s![.., ..nocc]).dot(&uo));
    out.slice_mut(ndarray::s![.., nocc..])
        .assign(&c.slice(ndarray::s![.., nocc..]).dot(&uv));
    out
}

/// Exactness anchor for the density-fitted Newton HVP: with the SCF's RI-J
/// (and DF-K when exact exchange is consumed) lent through `ScfJkResponse`,
/// `hessian_matvec_jk` must equal the central finite difference of the FITTED
/// orbital gradient g(κ) = [Cᵀ F_RI C]_ai, to the FD error bound, at
/// k_mix = 1 (HF), 0.2 (a global hybrid's fraction) and 0 (no exact exchange:
/// no K may be built).
///
/// Bound at step h/2 (Frobenius, derived, not fitted): central differences
/// have error T(h) = c·h² + O(h⁴) plus rounding ρ(h), so
///   ‖Hκ − FD(h/2)‖ ≤ ‖FD(h) − FD(h/2)‖/3 + 2·R(h/2),
/// dropping O(h⁴). R(h) = ‖E_g‖/h bounds ρ, where E_g = γ_m·|C|ᵀ M |C| (ov
/// block) bounds the rounding of one evaluation of g, M = |H|+|J|+½k|K|
/// elementwise and m = n² + naux + 2n (density contraction, aux sum, the two
/// MO-transform sums), γ_m = m·u/(1 − m·u).
///
/// The same check against the DIRECT HVP (what the solver used before) fails:
/// it differs from the fitted Hessian by the fitting error, which no step
/// refines. That difference is printed as "fitted vs direct HVP".
///
/// Measured (2026-11-11): fitted HVP vs FD 1.3e-10..8.4e-10 (bound
/// 5.3e-8..1.1e-7, the error shrinking as h², i.e. truncation only); fitted vs
/// direct HVP 5.0e-5..5.8e-5 (relative 0.9e-4..1.3e-4), ~500x the bound.
///
/// Mutation (applied and reverted, 2026-11-11): making `response_jk` take δJ
/// from the direct build although a fitted J is lent (the "direct J on an RI
/// SCF" bug, with the direct-build counter bypassed) fails the k_mix = 1 case
/// at the FD assert: |Hk − FD| = 5.39e-5 against a bound of 1.1e-7
/// (err/bound 961).
#[test]
fn fitted_newton_hvp_matches_finite_difference_of_the_fitted_gradient() {
    use ferric_integrals::oneelectron;
    use ferric_scf::df_j::{DfJ, RiJPrecision};
    use ferric_scf::df_k::DfK;
    use ferric_scf::rhf_newton::{
        direct_response_builds_on_this_thread, hessian_matvec, hessian_matvec_jk, RhfNewtonInputs,
        ScfJkResponse,
    };
    use ndarray::Array2;

    let aux_name = "def2-universal-jkfit";
    let mol = water();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let n = prep.nbasis();
    let naux = aux.nbasis();
    let nocc = mol.nelec() as usize / 2;
    let nv = n - nocc;
    let h = oneelectron::hcore(&prep);

    // ferric's own fitted-HF SCF supplies a sensible occupied space.
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        df_j_aux: Some(aux_name.into()),
        df_k_aux: Some(aux_name.into()),
        ri_j_precision: Some(RiJPrecision::F64),
        ri_k_precision: Some(ferric_scf::df_k::RiKPrecision::F64),
        ..Default::default()
    };
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(res.converged);

    let mut dfj = DfJ::new_with_precision(op, &prep, &aux, budget, RiJPrecision::F64).unwrap();
    let mut dfk = DfK::new(op, &prep, &aux, budget).unwrap();

    let mut rng = Xorshift64(0x9E3779B97F4A7C15);
    let k_dir = Array2::<f64>::from_shape_fn((nv, nocc), |_| 0.01 * rng.next_f64());
    let u = f64::EPSILON / 2.0;
    let m = (n * n + naux + 2 * n) as f64;
    let gamma = m * u / (1.0 - m * u);
    let pool = ferric_scf::engine_pool::EnginePool::new(op, &prep, 1e-14).unwrap();

    for &k_mix in &[1.0, 0.2, 0.0] {
        let (f0, _) = fitted_fock(&h, &mut dfj, &mut dfk, k_mix, &res.mos_alpha, nocc);
        let c = semicanonical(&f0, &res.mos_alpha, nocc);
        let (f, mag) = fitted_fock(&h, &mut dfj, &mut dfk, k_mix, &c, nocc);
        let f_mo = c.t().dot(&f).dot(&c);
        let ca = c.mapv(f64::abs);
        let e_g = frob(&ov_block(&ca.t().dot(&mag).dot(&ca), nocc)) * gamma;

        let mut grad = |cc: &Array2<f64>| {
            let (ff, _) = fitted_fock(&h, &mut dfj, &mut dfk, k_mix, cc, nocc);
            ov_block(&cc.t().dot(&ff).dot(cc), nocc)
        };
        let mut fd = |eps: f64| {
            let gp = grad(&cayley_rotate(&c, &k_dir, eps, nocc));
            let gm = grad(&cayley_rotate(&c, &k_dir, -eps, nocc));
            (&gp - &gm) / (2.0 * eps)
        };
        // Steps where truncation (~2e-4·h² here) is far below the rounding
        // bound, so the dropped O(h⁴) term cannot matter; at h ≥ 0.025 the FD
        // error is pure truncation and the Richardson estimate is tight to it.
        let ladder = [4e-3, 2e-3, 1e-3];
        let fds: Vec<Array2<f64>> = ladder.iter().map(|&e| fd(e)).collect();

        let inputs = RhfNewtonInputs {
            prep: &prep,
            bounds: &bounds,
            c: &c,
            f_mo: &f_mo,
            nocc,
            k_mix_sr: k_mix,
            rsh: None,
            fxc: None,
            thresh: 1e-12,
            ooc_budget: budget,
        };
        let hk_direct = hessian_matvec(&ctx, &inputs, &k_dir, &pool).unwrap();
        let direct_before = direct_response_builds_on_this_thread();
        let hk = {
            // At k_mix = 0 no DF-K is lent: a K request would then fall to the
            // direct build and trip the counter below.
            let resp = if k_mix == 0.0 {
                ScfJkResponse::new(Some(&mut dfj), None)
            } else {
                ScfJkResponse::new(Some(&mut dfj), Some(&mut dfk))
            };
            hessian_matvec_jk(&ctx, &inputs, &k_dir, &pool, Some(&resp)).unwrap()
        };
        let direct_builds = direct_response_builds_on_this_thread() - direct_before;
        assert_eq!(
            direct_builds, 0,
            "k_mix={k_mix}: with RI-J and DF-K lent, no direct four-centre build may run"
        );

        let scale = frob(&hk);
        let mut worst_ratio = 0.0f64;
        for w in 1..ladder.len() {
            let hh = ladder[w];
            let d = frob(&(&fds[w - 1] - &fds[w]));
            let bound = d / 3.0 + 2.0 * e_g / hh;
            let err = frob(&(&hk - &fds[w]));
            let err_direct = frob(&(&hk_direct - &fds[w]));
            eprintln!(
                "k_mix={k_mix} eps={hh:.3e}: |Hk_fit-FD|={err:.3e} (rel {:.3e}) bound={bound:.3e} \
                 [trunc {:.3e}, round {:.3e}]  |Hk_direct-FD|={err_direct:.3e}",
                err / scale,
                d / 3.0,
                2.0 * e_g / hh
            );
            worst_ratio = worst_ratio.max(err / bound);
        }
        let fit_vs_direct = frob(&(&hk - &hk_direct));
        eprintln!(
            "k_mix={k_mix}: fitted vs direct HVP |diff|={fit_vs_direct:.3e} (rel {:.3e}), \
             |Hk|={scale:.3e}, worst err/bound={worst_ratio:.3}",
            fit_vs_direct / scale
        );
        assert!(
            worst_ratio <= 1.0,
            "k_mix={k_mix}: the fitted HVP must match the FD of the fitted gradient within the \
             derived bound at every step (worst err/bound {worst_ratio:.3})"
        );
        // The direct HVP is a different operator (off by the fitting error):
        // the anchor must be able to tell them apart at the finest step...
        let hh = ladder[ladder.len() - 1];
        let d = frob(&(&fds[ladder.len() - 2] - &fds[ladder.len() - 1]));
        assert!(
            fit_vs_direct > d / 3.0 + 2.0 * e_g / hh,
            "k_mix={k_mix}: fitted and direct HVPs differ by {fit_vs_direct:.3e}, inside the FD \
             bound, so this anchor could not detect a direct response"
        );
        // ...while staying at the fitting-error scale (a sanity ceiling, not a
        // precision claim: a JK-fit basis is far better than 1%).
        assert!(
            fit_vs_direct / scale < 1e-2,
            "k_mix={k_mix}: fitted vs direct rel too large"
        );
    }
}

/// Newton-converged energies equal DIIS-converged energies when the Newton
/// HVP takes its J/K from the SCF's own RI-J / DF-K, and the Newton steps run
/// no direct four-centre build at all.
///
/// Bar: 1e-7 Ha, as in `rks_pbe_newton_engages_gga_fxc_and_matches_diis`. Both
/// runs minimise the same fitted functional on the same grid; each stops at
/// energy_conv 1e-9 / density_conv 1e-7 on its own path, and 100·energy_conv
/// covers two independently terminated iterations (the energy is second order
/// in the residual density error, so the stationary-point difference itself
/// is far below it).
///
/// Mutation (applied and reverted, 2026-11-11): passing `None` instead of the
/// `ScfJkResponse` at the solve_rhf call site (the pre-fix direct response)
/// fails all three cases at the direct-build assert (9, 9 and 7 direct
/// builds); the energies themselves agree either way, since both Hessians
/// lead to the same stationary point.
fn newton_matches_diis_with_fitted_response(xyz: &str, basis_name: &str, xc: &str, hybrid: bool) {
    use ferric_scf::rhf_newton::{direct_response_builds_on_this_thread, hvp_calls_on_this_thread};
    let aux = "def2-universal-jkfit";
    let mol = Molecule::load_xyz(xyz).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg_diis = RhfConfig {
        xc: Some(xc.into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        level_shift: 0.2,
        df_j_aux: Some(aux.into()),
        df_k_aux: hybrid.then(|| aux.into()),
        ..Default::default()
    };
    let cfg_newton = RhfConfig {
        newton_trigger: 1e-2,
        ..cfg_diis.clone()
    };
    let r_diis = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();
    let (hvp0, dir0) = (
        hvp_calls_on_this_thread(),
        direct_response_builds_on_this_thread(),
    );
    let r_newton = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_newton).unwrap();
    let hvps = hvp_calls_on_this_thread() - hvp0;
    let direct = direct_response_builds_on_this_thread() - dir0;
    let de = (r_diis.energy - r_newton.energy).abs();
    eprintln!(
        "{xyz} {basis_name} {xc}: E_diis={:.10} ({} it) E_newton={:.10} ({} it) dE={de:.2e} \
         hvps={hvps} direct_builds={direct}",
        r_diis.energy, r_diis.iterations, r_newton.energy, r_newton.iterations
    );
    assert!(r_diis.converged && r_newton.converged);
    assert!(hvps >= 1, "the Newton branch must engage (0 HVPs)");
    assert_eq!(
        direct, 0,
        "a fitted SCF's Newton HVPs must build no direct J/K"
    );
    assert!(de < 1e-7, "Newton vs DIIS energy differ by {de:.3e}");
}

#[test]
fn rks_newton_fitted_response_water_pbe() {
    newton_matches_diis_with_fitted_response(
        "../../testdata/molecules/water.xyz",
        "cc-pvdz",
        "PBE",
        false,
    );
}

#[test]
fn rks_newton_fitted_response_water_b3lyp() {
    newton_matches_diis_with_fitted_response(
        "../../testdata/molecules/water.xyz",
        "cc-pvdz",
        "B3LYP",
        true,
    );
}

#[test]
fn rks_newton_fitted_response_benzene_b3lyp() {
    newton_matches_diis_with_fitted_response(
        "../../testdata/molecules/benzene.xyz",
        "def2-svp",
        "B3LYP",
        true,
    );
}
