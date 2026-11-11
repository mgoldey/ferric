//! The RKS Newton Hessian-vector product's XC-kernel response on the SCF's own
//! screened XC grid (`ferric_dft::xc_batch::ScreenedFxc`, selected by
//! `rohf::restricted_fxc_response`).
//!
//! Anchor: the HVP, with RI-J / DF-K lent as the SCF has them and the f_xc
//! response from the SCF's `KsXc`, equals the central finite difference of
//! the orbital gradient of the FITTED, SCREENED Kohn-Sham Fock
//! `F = H + J_RI − ½·k·K_RI + V_xc[KsXc]` along a random rotation, within a
//! derived FD error bound. The screened response is the exact derivative of
//! the screened V_xc (same batches, same kept shells, same density floor), so
//! no screening tolerance enters the bound.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::ks::KsXc;
use ferric_dft::xc_batch::XcBatchConfig;
use ferric_dft::xc_trait::XcContribution;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j::{DfJ, RiJPrecision};
use ferric_scf::df_k::DfK;
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rhf_newton::{hessian_matvec_jk, RhfNewtonInputs, ScfJkResponse};
use ferric_scf::rohf::{
    dense_fxc_builds_on_this_thread, restricted_fxc_response, screened_fxc_builds_on_this_thread,
    FxcKernelStore,
};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{s, Array2, Axis};

const AUX: &str = "def2-universal-jkfit";

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

fn frob(a: &Array2<f64>) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

fn ov_block(m: &Array2<f64>, nocc: usize) -> Array2<f64> {
    m.slice(s![nocc.., ..nocc]).to_owned()
}

/// Cayley rotation of `c` by `eps·κ` (occ→virt block), as `apply_cayley` does.
fn cayley_rotate(c: &Array2<f64>, k_ov: &Array2<f64>, eps: f64, nocc: usize) -> Array2<f64> {
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
        u.column_mut(col).assign(&sol);
    }
    c.dot(&u)
}

/// Rotate `c` within occ and virt so the Fock `f` is diagonal in both: the
/// diagonal-gap Hessian of `rhf_newton` is then the exact second derivative.
fn semicanonical(f: &Array2<f64>, c: &Array2<f64>, nocc: usize) -> Array2<f64> {
    use ndarray_linalg::{Eigh, UPLO};
    let f_mo = c.t().dot(f).dot(c);
    let (_, uo) = f_mo
        .slice(s![..nocc, ..nocc])
        .to_owned()
        .eigh(UPLO::Lower)
        .unwrap();
    let (_, uv) = f_mo
        .slice(s![nocc.., nocc..])
        .to_owned()
        .eigh(UPLO::Lower)
        .unwrap();
    let mut out = c.clone();
    out.slice_mut(s![.., ..nocc])
        .assign(&c.slice(s![.., ..nocc]).dot(&uo));
    out.slice_mut(s![.., nocc..])
        .assign(&c.slice(s![.., nocc..]).dot(&uv));
    out
}

/// Elementwise magnitude of the closed-shell V_xc integrand at `d`, the `S`
/// of the XC part of the gradient's rounding bound:
/// `Σ_g w [|v_ρ||χ_μ||χ_ν| + 2|v_σ| Σ_a |∂_aρ| (|χ_μ||∂_aχ_ν| + |∂_aχ_μ||χ_ν|)
///  + ½|v_τ| Σ_a |∂_aχ_μ||∂_aχ_ν|]`, on the evaluator's own grid.
fn xc_magnitude(mol: &Molecule, bs: &basis::BasisSet, ks: &KsXc, d: &Array2<f64>) -> Array2<f64> {
    use ferric_dft::density_on_grid::{eval_density_closed, eval_tau_closed};
    let pts: Vec<[f64; 3]> = ks.grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = ferric_dft::ao_grid::eval_basis_and_grad_on_points(mol, bs, &pts).unwrap();
    let dens = eval_density_closed(d, &chi, &dchi);
    let tau = eval_tau_closed(d, &dchi);
    let k = ferric_dft::vxc::closed_kernel(&dens, Some(&tau), &ks.xc);
    let w: Vec<f64> = ks.grid.iter().map(|g| g.weight).collect();
    let ca = chi.mapv(f64::abs);
    let scaled = |fac: &dyn Fn(usize) -> f64, m: &Array2<f64>| {
        let mut out = m.clone();
        for (g, mut col) in out.axis_iter_mut(Axis(1)).enumerate() {
            col *= fac(g);
        }
        out
    };
    let mut v = scaled(&|g| w[g] * k.vrho[g].abs(), &ca).dot(&ca.t());
    for ax in 0..3 {
        let da = dchi.index_axis(Axis(0), ax).mapv(f64::abs);
        let m = scaled(
            &|g| 2.0 * w[g] * k.vsigma[g].abs() * dens.grad[(ax, g)].abs(),
            &ca,
        )
        .dot(&da.t());
        v = v + &m + m.t();
        v += &scaled(&|g| 0.5 * w[g] * k.vtau[g].abs(), &da).dot(&da.t());
    }
    v
}

/// γ_m = m·u/(1 − m·u).
fn gamma(m: usize) -> f64 {
    let u = f64::EPSILON / 2.0;
    m as f64 * u / (1.0 - m as f64 * u)
}

/// What one anchor case measured.
struct Anchor {
    worst_ratio: f64,
    worst_ratio_dense: f64,
    screened_vs_dense: f64,
    scale: f64,
}

/// FD anchor for one functional at one XC screening threshold; see the module
/// docs. Bound at step h/2 (Frobenius, derived as in rhf_newton_smoke's
/// fitted J/K anchor): ‖Hκ − FD(h/2)‖ ≤ ‖FD(h) − FD(h/2)‖/3 + 2·E_g/(h/2),
/// E_g = ‖ov(|C|ᵀ (γ_m·(|H|+|J|+½k|K|) + γ_x·S_xc) |C|)‖, m = n² + naux + 2n,
/// x = npts + 6n + 64 (the quadrature, density and MO-transform sums, and an
/// allowance of 64 per point for libxc and the propagation of the rounded
/// density into v_xc — an assumption, not a derived constant), S_xc =
/// [`xc_magnitude`].
///
/// Also evaluates the dense, UNSCREENED `FxcKernelStore` HVP on the same
/// grid, i.e. the pre-change response, against the same FD.
fn fd_anchor(xc: &str, k_mix: f64, screen_thresh: f64) -> Anchor {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let (n, naux) = (prep.nbasis(), aux.nbasis());
    let nocc = mol.nelec() as usize / 2;
    let nv = n - nocc;
    let h = oneelectron::hcore(&prep);
    let grid = AtomicGridConfig::production();

    let cfg = RhfConfig {
        xc: Some(xc.into()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        level_shift: 0.2,
        dft_grid: Some(grid.clone()),
        df_j_aux: Some(AUX.into()),
        df_k_aux: (k_mix != 0.0).then(|| AUX.into()),
        ri_j_precision: Some(RiJPrecision::F64),
        ri_k_precision: Some(ferric_scf::df_k::RiKPrecision::F64),
        ..Default::default()
    };
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(res.converged);

    let ks = KsXc::new_with_batch_config(
        &mol,
        &bs,
        xc,
        &grid,
        &grid,
        None,
        None,
        XcBatchConfig {
            screen_thresh,
            ..Default::default()
        },
    )
    .unwrap();
    let mut dfj = DfJ::new_with_precision(op, &prep, &aux, budget, RiJPrecision::F64).unwrap();
    let mut dfk = DfK::new(op, &prep, &aux, budget).unwrap();
    let mut fock = |c: &Array2<f64>| -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let co = c.slice(s![.., ..nocc]);
        let d = 2.0 * co.dot(&co.t());
        let mut j = Array2::<f64>::zeros((n, n));
        dfj.build(&d, &mut j).unwrap();
        let mut k = Array2::<f64>::zeros((n, n));
        if k_mix != 0.0 {
            dfk.build(&d, &mut k).unwrap();
        }
        let mut f = &h + &j - &(0.5 * k_mix * &k);
        ks.add_xc(&d, &mut f);
        let mag = h.mapv(f64::abs) + j.mapv(f64::abs) + (0.5 * k_mix.abs()) * k.mapv(f64::abs);
        (f, mag, d)
    };

    let (f0, _, _) = fock(&res.mos_alpha);
    let c = semicanonical(&f0, &res.mos_alpha, nocc);
    let (f, mag, d) = fock(&c);
    let f_mo = c.t().dot(&f).dot(&c);
    let ca = c.mapv(f64::abs);
    let s_xc = xc_magnitude(&mol, &bs, &ks, &d);
    let e_jk = gamma(n * n + naux + 2 * n) * &mag;
    let e_xc = gamma(ks.grid.len() + 6 * n + 64) * &s_xc;
    let e_g = frob(&ov_block(&ca.t().dot(&(e_jk + e_xc)).dot(&ca), nocc));

    let mut rng = Xorshift64(0x9E3779B97F4A7C15);
    let k_dir = Array2::<f64>::from_shape_fn((nv, nocc), |_| 0.01 * rng.next_f64());
    let ladder = [4e-3, 2e-3, 1e-3];
    let fds: Vec<Array2<f64>> = ladder
        .iter()
        .map(|&e| {
            let mut g = |cc: &Array2<f64>| {
                let (ff, _, _) = fock(cc);
                ov_block(&cc.t().dot(&ff).dot(cc), nocc)
            };
            let gp = g(&cayley_rotate(&c, &k_dir, e, nocc));
            let gm = g(&cayley_rotate(&c, &k_dir, -e, nocc));
            (&gp - &gm) / (2.0 * e)
        })
        .collect();

    let pool = ferric_scf::engine_pool::EnginePool::new(op, &prep, 1e-14).unwrap();
    let inputs = |fxc| RhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c: &c,
        f_mo: &f_mo,
        nocc,
        k_mix_sr: k_mix,
        rsh: None,
        fxc: Some(fxc),
        thresh: 1e-12,
        ooc_budget: budget,
    };
    let (sc0, de0) = (
        screened_fxc_builds_on_this_thread(),
        dense_fxc_builds_on_this_thread(),
    );
    let fxc = restricted_fxc_response(&ks, (&mol, &prep, &cfg, xc), &d).unwrap();
    assert_eq!(
        (
            screened_fxc_builds_on_this_thread() - sc0,
            dense_fxc_builds_on_this_thread() - de0
        ),
        (1, 0),
        "{xc}: the response must come from the SCF's screened grid, with no dense kernel"
    );
    let mut dfj2 = DfJ::new_with_precision(op, &prep, &aux, budget, RiJPrecision::F64).unwrap();
    let mut dfk2 = DfK::new(op, &prep, &aux, budget).unwrap();
    let hk = hvp(&ctx, &inputs(&*fxc), &k_dir, &pool, &mut dfj2, &mut dfk2);
    let half = 0.5 * &d;
    let store = FxcKernelStore::build(&mol, &prep, &grid, xc, None, &half, &half).unwrap();
    let dense_resp = store.response();
    let hk_dense = hvp(
        &ctx,
        &inputs(&*dense_resp),
        &k_dir,
        &pool,
        &mut dfj2,
        &mut dfk2,
    );

    let scale = frob(&hk);
    let (mut worst, mut worst_dense) = (0.0f64, 0.0f64);
    for w in 1..ladder.len() {
        let hh = ladder[w];
        let bound = frob(&(&fds[w - 1] - &fds[w])) / 3.0 + 2.0 * e_g / hh;
        let err = frob(&(&hk - &fds[w]));
        let err_dense = frob(&(&hk_dense - &fds[w]));
        eprintln!(
            "{xc} thresh={screen_thresh:.0e} h={hh:.0e}: |Hk-FD|={err:.3e} (rel {:.2e}) \
             |Hk_dense-FD|={err_dense:.3e} bound={bound:.3e} [round {:.3e}]",
            err / scale,
            2.0 * e_g / hh
        );
        worst = worst.max(err / bound);
        worst_dense = worst_dense.max(err_dense / bound);
    }
    let screened_vs_dense = frob(&(&hk - &hk_dense));
    eprintln!(
        "{xc} thresh={screen_thresh:.0e}: |Hk|={scale:.3e} screened vs dense {screened_vs_dense:.3e} \
         worst err/bound screened {worst:.3e}, dense {worst_dense:.3e}"
    );
    Anchor {
        worst_ratio: worst,
        worst_ratio_dense: worst_dense,
        screened_vs_dense,
        scale,
    }
}

/// One HVP with RI-J (and DF-K when the step consumes exact exchange) lent.
fn hvp<'l>(
    ctx: &ParallelContext,
    inputs: &RhfNewtonInputs,
    k_dir: &Array2<f64>,
    pool: &ferric_scf::engine_pool::EnginePool,
    dfj: &mut DfJ<'l>,
    dfk: &mut DfK<'l>,
) -> Array2<f64> {
    let resp = if inputs.k_mix_sr == 0.0 {
        ScfJkResponse::new(Some(dfj), None)
    } else {
        ScfJkResponse::new(Some(dfj), Some(dfk))
    };
    hessian_matvec_jk(ctx, inputs, k_dir, pool, Some(&resp)).unwrap()
}

fn assert_anchor(a: &Anchor, what: &str) {
    assert!(
        a.worst_ratio <= 1.0,
        "{what}: the screened-f_xc HVP must match the FD of the fitted, screened gradient \
         (worst err/bound {:.3})",
        a.worst_ratio
    );
    assert!(a.scale > 0.0 && a.screened_vs_dense.is_finite());
}

/// Water/cc-pVDZ PBE (k_mix 0, RI-J), production grid and screening.
///
/// Mutation (applied and reverted, 2026-11-11): see the report; dropping the
/// cross-spin `H_i,ρβ` column in `fxc::restricted_coefficients` fails here.
#[test]
fn screened_fxc_hvp_matches_fd_water_pbe() {
    assert_anchor(&fd_anchor("PBE", 0.0, 1e-10), "PBE");
}

/// Water/cc-pVDZ B3LYP (k_mix 0.2, RI-J + DF-K).
#[test]
fn screened_fxc_hvp_matches_fd_water_b3lyp() {
    assert_anchor(&fd_anchor("B3LYP", 0.2, 1e-10), "B3LYP");
}

/// Water/cc-pVDZ r2SCAN: the τ response (k_mix 0, RI-J).
#[test]
fn screened_fxc_hvp_matches_fd_water_r2scan() {
    assert_anchor(&fd_anchor("r2SCAN", 0.0, 1e-10), "r2SCAN");
}

/// The anchor sees screening. At a coarse XC screening threshold (1e-4, so
/// the screening error is far above the FD resolution) the screened response
/// still matches the FD of the equally screened gradient, while the dense,
/// unscreened response — what the Newton step used before, and what a
/// response mixing in an unscreened grid would give — fails it.
#[test]
fn screened_fxc_hvp_matches_fd_at_coarse_screening_and_dense_does_not() {
    let a = fd_anchor("PBE", 0.0, 1e-4);
    assert_anchor(&a, "PBE coarse");
    assert!(
        a.worst_ratio_dense > 1.0,
        "the unscreened response must fail the coarse-screening anchor (err/bound {:.3}); \
         otherwise this anchor cannot detect a screening mismatch",
        a.worst_ratio_dense
    );
}

/// Newton-converged energy equals the DIIS-converged one when the Newton HVP's
/// f_xc response comes from the SCF's screened grid, and every Newton step
/// built it there (no dense kernel). Water/cc-pVDZ, RI-J (+ DF-K for the
/// hybrid), default grid and screening. Bar 1e-7 Ha, as in
/// `rhf_newton_smoke::newton_matches_diis_with_fitted_response` (two
/// independently terminated runs at energy_conv 1e-9 of the same functional).
fn newton_matches_diis_screened(xc: &str, hybrid: bool) {
    use ferric_scf::rhf_newton::hvp_calls_on_this_thread;
    let mol = water();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg_diis = RhfConfig {
        xc: Some(xc.into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        level_shift: 0.2,
        df_j_aux: Some(AUX.into()),
        df_k_aux: hybrid.then(|| AUX.into()),
        ..Default::default()
    };
    let cfg_newton = RhfConfig {
        newton_trigger: 1e-2,
        ..cfg_diis.clone()
    };
    let r_diis = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();
    let (h0, s0, d0) = (
        hvp_calls_on_this_thread(),
        screened_fxc_builds_on_this_thread(),
        dense_fxc_builds_on_this_thread(),
    );
    let r_newton = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_newton).unwrap();
    let hvps = hvp_calls_on_this_thread() - h0;
    let screened = screened_fxc_builds_on_this_thread() - s0;
    let dense = dense_fxc_builds_on_this_thread() - d0;
    let de = (r_diis.energy - r_newton.energy).abs();
    eprintln!(
        "{xc}: E_diis={:.10} ({} it) E_newton={:.10} ({} it) dE={de:.2e} hvps={hvps} \
         screened_fxc={screened} dense_fxc={dense}",
        r_diis.energy, r_diis.iterations, r_newton.energy, r_newton.iterations
    );
    assert!(r_diis.converged && r_newton.converged);
    assert!(hvps >= 1, "{xc}: the Newton branch must engage");
    assert!(screened >= 1, "{xc}: no screened f_xc response was built");
    assert_eq!(dense, 0, "{xc}: a dense f_xc kernel was built");
    assert!(de < 1e-7, "{xc}: Newton vs DIIS energy differ by {de:.3e}");
}

#[test]
fn rks_newton_screened_fxc_matches_diis_pbe() {
    newton_matches_diis_screened("PBE", false);
}

#[test]
fn rks_newton_screened_fxc_matches_diis_b3lyp() {
    newton_matches_diis_screened("B3LYP", true);
}

#[test]
fn rks_newton_screened_fxc_matches_diis_r2scan() {
    newton_matches_diis_screened("r2SCAN", false);
}
