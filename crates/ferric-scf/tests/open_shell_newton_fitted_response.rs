//! Open-shell (UHF/UKS, ROHF/ROKS) Newton Hessian-vector products built from
//! the SCF's own density-fitted J/K, the open-shell sibling of
//! `rhf_newton_smoke.rs`'s fitted-response tests (PR #642).
//!
//! Covers:
//!   1. FD anchor: with RI-J (and DF-K when exact exchange is consumed) lent as
//!      `j_response`/`k_response`, the UHF and ROHF matvecs equal the central
//!      finite difference of the FITTED orbital gradient to a derived bound,
//!      with no direct four-centre build.
//!   2. Newton-converged energies equal DIIS-converged ones (fitted SCFs), with
//!      no direct four-centre build in any Newton HVP.
//!   3. A Newton step at a converged point computes no HVP.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j::{DfJ, RiJPrecision};
use ferric_scf::df_k::{DfK, RiKPrecision};
use ferric_scf::engine_pool::EnginePool;
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf_newton::{direct_response_builds_on_this_thread, AoResponse, OccKResponse};
use ndarray::{s, Array2};
use ndarray_linalg::Solve;
use std::cell::RefCell;

const AUX: &str = "def2-universal-jkfit";
const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";
const H2_TRIPLET: &str = "2\nH2\nH 0 0 0\nH 0 0 2.0\n";
const NH2: &str = "3\nNH2\nN 0 0 0\nH 0 0.80 -0.62\nH 0 -0.80 -0.62\n";

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

fn frob(a: &Array2<f64>) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Cayley `C·(I − κ/2)⁻¹(I + κ/2)` for an antisymmetric MO-basis κ.
fn cayley(c: &Array2<f64>, kappa: &Array2<f64>) -> Array2<f64> {
    let n = c.nrows();
    let eye = Array2::<f64>::eye(n);
    let am = &eye - &(0.5 * kappa);
    let bm = &eye + &(0.5 * kappa);
    let mut u = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let sol = am.solve(&bm.column(col).to_owned()).unwrap();
        u.column_mut(col).assign(&sol);
    }
    c.dot(&u)
}

/// Antisymmetric κ with `blk` in rows `rows`, cols `cols` (and −blkᵀ mirrored).
fn put_block(kappa: &mut Array2<f64>, blk: &Array2<f64>, r0: usize, c0: usize) {
    for ((i, j), &v) in blk.indexed_iter() {
        kappa[(r0 + i, c0 + j)] = v;
        kappa[(c0 + j, r0 + i)] = -v;
    }
}

/// The fitted SCF pieces shared by both anchors.
struct Fit<'p> {
    h: Array2<f64>,
    dfj: DfJ<'p>,
    dfk: DfK<'p>,
}

impl<'p> Fit<'p> {
    fn new(prep: &'p PreparedBasis, aux: &'p PreparedBasis) -> Self {
        let budget = ferric_core::memory::resolve_budget_bytes(None);
        let op = Operator::coulomb();
        Fit {
            h: oneelectron::hcore(prep),
            dfj: DfJ::new_with_precision(op, prep, aux, budget, RiJPrecision::F64).unwrap(),
            dfk: DfK::new(op, prep, aux, budget).unwrap(),
        }
    }

    /// Fitted spin Focks `F_σ = H + J[C_αoC_αoᵀ + C_βoC_βoᵀ] − k·K[C_σoC_σoᵀ]`
    /// and the elementwise magnitudes `|H| + |J| + k|K_σ|` the rounding bound reads.
    fn focks(
        &mut self,
        k_mix: f64,
        (ca, na): (&Array2<f64>, usize),
        (cb, nb): (&Array2<f64>, usize),
    ) -> [(Array2<f64>, Array2<f64>); 2] {
        let n = ca.nrows();
        let oa = ca.slice(s![.., ..na]).to_owned();
        let ob = cb.slice(s![.., ..nb]).to_owned();
        let d = oa.dot(&oa.t()) + ob.dot(&ob.t());
        let mut j = Array2::<f64>::zeros((n, n));
        self.dfj.build(&d, &mut j).unwrap();
        let mut spin = |o: &Array2<f64>| {
            let mut k = Array2::<f64>::zeros((n, n));
            if k_mix != 0.0 && o.ncols() > 0 {
                self.dfk.build_from_occ(o, &mut k).unwrap();
            }
            let f = &self.h + &j - &(k_mix * &k);
            let mag = self.h.mapv(f64::abs) + j.mapv(f64::abs) + k_mix.abs() * k.mapv(f64::abs);
            (f, mag)
        };
        [spin(&oa), spin(&ob)]
    }
}

/// Fitted `j_response` / `k_response` closures over a SEPARATE pair of
/// builders (same tensors, so the same fitted operator as `Fit`).
#[allow(clippy::type_complexity)]
fn lend<'b, 'p: 'b>(
    dfj: &'b RefCell<DfJ<'p>>,
    dfk: &'b RefCell<DfK<'p>>,
) -> (Box<AoResponse<'b>>, Box<OccKResponse<'b>>) {
    let j = Box::new(move |dd: &Array2<f64>| {
        let mut out = Array2::<f64>::zeros(dd.raw_dim());
        dfj.borrow_mut().build(dd, &mut out)?;
        Ok(out)
    }) as Box<AoResponse<'b>>;
    let k = Box::new(move |c: &Array2<f64>| {
        let n = c.nrows();
        let mut out = Array2::<f64>::zeros((n, n));
        dfk.borrow_mut().build_from_occ(c, &mut out)?;
        Ok(out)
    }) as Box<OccKResponse<'b>>;
    (j, k)
}

/// One FD-anchor case: `analytic` is the matvec, `fd(h)` the central FD of the
/// fitted gradient, `corr` the (exact commutator − diagonal gap) correction,
/// `e_g` the rounding bound of one gradient evaluation. Returns worst err/bound.
///
/// Bound at step h (Frobenius, derived): central differences have error
/// T(h) = c·h² + O(h⁴) plus rounding ≤ e_g/h, so with the ladder halving h,
///   ‖Hκ − FD(h)‖ ≤ ‖FD(2h) − FD(h)‖/3 + 2·e_g/h,
/// dropping O(h⁴) (the ladder keeps truncation ≪ rounding). The matvec's own
/// rounding is not in the bound (as in the closed-shell anchor); it is
/// γ·|δF|-sized, below e_g/h by the factor h·|δF|/|F|.
fn check_against_fd(
    label: &str,
    analytic: &[Array2<f64>],
    corr: &[Array2<f64>],
    e_g: f64,
    fd: &mut dyn FnMut(f64) -> Vec<Array2<f64>>,
) -> f64 {
    let pack = |v: &[Array2<f64>]| -> Array2<f64> {
        let flat: Vec<f64> = v.iter().flat_map(|a| a.iter().copied()).collect();
        Array2::from_shape_vec((1, flat.len()), flat).unwrap()
    };
    // `analytic + corr` is the exact derivative of the gradient (the
    // Jacobian the FD measures); `corr` is each caller's documented term.
    let hk = &pack(analytic) + &pack(corr);
    let ladder = [4e-3, 2e-3, 1e-3];
    let fds: Vec<Array2<f64>> = ladder.iter().map(|&h| pack(&fd(h))).collect();
    let scale = frob(&hk);
    let mut worst = 0.0f64;
    for w in 1..ladder.len() {
        let h = ladder[w];
        let trunc = frob(&(&fds[w - 1] - &fds[w])) / 3.0;
        let round = 2.0 * e_g / h;
        let err = frob(&(&hk - &fds[w]));
        eprintln!(
            "{label} h={h:.0e}: |Hk-FD|={err:.3e} (rel {:.2e}) bound={:.3e} [trunc {trunc:.2e}, round {round:.2e}]",
            err / scale,
            trunc + round
        );
        worst = worst.max(err / (trunc + round));
    }
    worst
}

fn gamma(m: usize) -> f64 {
    let u = f64::EPSILON / 2.0;
    let m = m as f64;
    m * u / (1.0 - m * u)
}

fn converged_scf(
    xyz: &str,
    mult: usize,
    restricted_open: bool,
) -> (
    Molecule,
    PreparedBasis,
    PreparedBasis,
    ferric_scf::result::ScfResult,
) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        df_j_aux: Some(AUX.into()),
        df_k_aux: Some(AUX.into()),
        ri_j_precision: Some(RiJPrecision::F64),
        ri_k_precision: Some(RiKPrecision::F64),
        ..Default::default()
    };
    let res = if restricted_open {
        ferric_scf::rohf::solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap()
    } else {
        ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap()
    };
    assert!(res.converged);
    (mol, prep, aux, res)
}

/// Exactness anchor, UHF/UKS: the fitted matvec equals the central FD of the
/// fitted per-spin gradient g_σ = [C_σᵀ F_σ C_σ]_ai along a random coupled
/// rotation (κ_α, κ_β), at k_mix = 1 (UHF), 0.2 (B3LYP's exact-exchange
/// fraction) and 0 (PBE: no K may be built, and no DF-K is lent). The XC
/// kernel is not part of this anchor (`fxc = None`): it is unchanged here and
/// FD-validated in `uhf_newton_smoke.rs`/`mgga_response_vs_pyscf.rs`.
///
/// `e_g = γ_m·‖|C|ᵀ(|H|+|J|+k|K_σ|)|C|‖_ov` per spin (added in quadrature),
/// m = n² + naux + 2n; the commutator correction adds γ_{2n}‖|F||κ| + |κ||F|‖·h
/// at most, far below and not counted.
///
/// Measured (2026-11-11, OH ²Π cc-pVDZ): |Hκ − FD| 1.4e-10..9.1e-10 (rel
/// ≤ 1.3e-9) against bounds 2.5e-8..5.3e-8, worst err/bound 0.033, the error
/// falling 4× per halving of h (truncation only); fitted vs direct HVP
/// 3.0e-5..3.1e-5 (rel 4.5e-5..5.4e-5), ~600× the bound; no direct build.
///
/// Mutation (applied and reverted, 2026-11-11): `response_fock` taking δJ from
/// an uncounted direct build although `j_response` is lent (the "direct J on
/// an RI SCF" bug) fails at k_mix = 1 with err/bound 1128.
/// K-path mutation (applied and reverted, 2026-11-11): dropping the ½ on the
/// per-spin occupied-pair δK (`½·(K(X+Y) − K(X−Y))` → `K(X+Y) − K(X−Y)`)
/// fails at k_mix = 1 with |Hκ − FD| 4.75e-2 (rel 7.2e-2), err/bound 1.7e6;
/// it also exceeds the fitted-vs-direct ceiling below.
#[test]
fn uhf_fitted_hvp_matches_finite_difference_of_the_fitted_gradient() {
    use ferric_scf::uhf_newton::{hessian_matvec, UhfNewtonInputs};
    let (mol, prep, aux, res) = converged_scf(OH, 2, false);
    let n = prep.nbasis();
    let na = (mol.nelec() as usize).div_ceil(2);
    let nb = mol.nelec() as usize / 2;
    let ca = res.mos_alpha.clone();
    let cb = res.mos_beta.clone().expect("UHF has β MOs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let pool = EnginePool::new(op, &prep, 1e-14).unwrap();
    let g = gamma(n * n + aux.nbasis() + 2 * n);

    let mut rng = Xorshift64(0x9E3779B97F4A7C15);
    let ka = Array2::from_shape_fn((n - na, na), |_| 0.01 * rng.next_f64());
    let kb = Array2::from_shape_fn((n - nb, nb), |_| 0.01 * rng.next_f64());
    let mut kap_a = Array2::<f64>::zeros((n, n));
    let mut kap_b = Array2::<f64>::zeros((n, n));
    put_block(&mut kap_a, &ka, na, 0);
    put_block(&mut kap_b, &kb, nb, 0);

    let mut fit = Fit::new(&prep, &aux);
    let dfj_h =
        RefCell::new(DfJ::new_with_precision(op, &prep, &aux, budget, RiJPrecision::F64).unwrap());
    let dfk_h = RefCell::new(DfK::new(op, &prep, &aux, budget).unwrap());
    let (jr, kr) = lend(&dfj_h, &dfk_h);

    for &k_mix in &[1.0, 0.2, 0.0] {
        let [(fa, ma), (fb, mb)] = fit.focks(k_mix, (&ca, na), (&cb, nb));
        let fa_mo = ca.t().dot(&fa).dot(&ca);
        let fb_mo = cb.t().dot(&fb).dot(&cb);
        let inp = UhfNewtonInputs {
            prep: &prep,
            bounds: &bounds,
            c_a: &ca,
            c_b: &cb,
            f_a_mo: &fa_mo,
            f_b_mo: &fb_mo,
            nocc_a: na,
            nocc_b: nb,
            k_mix_sr: k_mix,
            rsh: None,
            fxc: None,
            thresh: 1e-12,
            ooc_budget: budget,
            j_response: Some(&*jr),
            k_response: (k_mix != 0.0).then_some(&*kr),
        };
        let dir0 = direct_response_builds_on_this_thread();
        let (ha, hb) = hessian_matvec(&ctx, &inp, &ka, &kb, &pool).unwrap();
        assert_eq!(
            direct_response_builds_on_this_thread() - dir0,
            0,
            "k_mix={k_mix}: with RI-J (and DF-K) lent, no direct four-centre build may run"
        );
        // Old direct HVP, for the fitting-error scale.
        let inp_dir = UhfNewtonInputs {
            j_response: None,
            k_response: None,
            ..inp
        };
        let (da, db) = hessian_matvec(&ctx, &inp_dir, &ka, &kb, &pool).unwrap();

        // corr = [Fκ − κF]_ov − gap·κ, per spin.
        let corr = |f_mo: &Array2<f64>, kap: &Array2<f64>, k: &Array2<f64>, no: usize| {
            let comm = f_mo.dot(kap) - kap.dot(f_mo);
            let mut c = comm.slice(s![no.., ..no]).to_owned();
            for ((r, i), v) in c.indexed_iter_mut() {
                *v -= (f_mo[(no + r, no + r)] - f_mo[(i, i)]) * k[(r, i)];
            }
            c
        };
        let corr_v = [corr(&fa_mo, &kap_a, &ka, na), corr(&fb_mo, &kap_b, &kb, nb)];
        let eg = |c: &Array2<f64>, m: &Array2<f64>, no: usize| {
            let a = c.mapv(f64::abs);
            frob(&a.t().dot(m).dot(&a).slice(s![no.., ..no]).to_owned()) * g
        };
        let e_g = eg(&ca, &ma, na).hypot(eg(&cb, &mb, nb));

        let mut fd = |h: f64| -> Vec<Array2<f64>> {
            let mut at = |e: f64| {
                let (ra, rb) = (cayley(&ca, &(e * &kap_a)), cayley(&cb, &(e * &kap_b)));
                let [(fa, _), (fb, _)] = fit.focks(k_mix, (&ra, na), (&rb, nb));
                let ga = ra.t().dot(&fa).dot(&ra).slice(s![na.., ..na]).to_owned();
                let gb = rb.t().dot(&fb).dot(&rb).slice(s![nb.., ..nb]).to_owned();
                [ga, gb]
            };
            let [pa, pb] = at(h);
            let [ma_, mb_] = at(-h);
            vec![(&pa - &ma_) / (2.0 * h), (&pb - &mb_) / (2.0 * h)]
        };
        let label = format!("UHF k_mix={k_mix}");
        let worst = check_against_fd(&label, &[ha.clone(), hb.clone()], &corr_v, e_g, &mut fd);
        let fit_vs_dir = frob(&(&ha - &da)).hypot(frob(&(&hb - &db)));
        let scale = frob(&ha).hypot(frob(&hb));
        eprintln!(
            "{label}: fitted vs direct HVP |diff|={fit_vs_dir:.3e} (rel {:.2e}), worst err/bound={worst:.3}",
            fit_vs_dir / scale
        );
        assert!(
            worst <= 1.0,
            "{label}: fitted HVP outside the FD bound (err/bound {worst:.3})"
        );
        // Sanity ceiling, not a precision claim: the fitted and direct HVPs
        // differ by the JK-fit error, measured at 4.5e-5..1.6e-4 relative.
        let rel = fit_vs_dir / scale;
        assert!(
            rel < 1e-3,
            "{label}: fitted vs direct HVP rel {rel:.2e} >= 1e-3"
        );
    }
}

/// Exactness anchor, ROHF/ROKS (NH2 ²B₁): the fitted Roothaan matvec equals
/// the central FD of the fitted HALF gradient blocks
/// (g_vc = [F_α+F_β]_vc, g_vo = [F_α]_vo, g_oc = [F_β]_oc, one MO set),
/// plus the Hessian-vs-Jacobian term (the matvec is the exact Hessian since
/// 2026-11; before, this test added the commutator its diagonal gap omitted),
/// at k_mix = 1 (ROHF), 0.2 (B3LYP's fraction) and 0. This exercises
/// the per-spin occupied/unoccupied split the DF-K path builds from (α: occ =
/// closed+open, κ_α = [κ_vc | κ_vo]; β: occ = closed, κ_β = [κ_oc ; κ_vc]).
/// Bound as in the UHF anchor (one gradient evaluation's rounding is the sum
/// over its blocks; the vc block reads both spins).
///
/// Measured (2026-11-11, NH2 ²B₁ cc-pVDZ): |Hκ − FD| 1.7e-10..1.2e-9 against
/// bounds 8.7e-8..1.9e-7, worst err/bound 0.013; fitted vs direct HVP
/// 8.2e-5..1.0e-4 (rel 1.0e-4..1.6e-4); no direct build.
///
/// Mutation (applied and reverted, 2026-11-11): the same uncounted direct δJ in
/// `rohf_newton::response_fock` fails at k_mix = 1 with err/bound 1083.
/// K-path mutation (applied and reverted, 2026-11-11): κ_β assembled as
/// [κ_vc ; κ_oc] instead of [κ_oc ; κ_vc] (open rows must come first) fails at
/// k_mix = 1 with |Hκ − FD| 4.11e-2 (rel 5.0e-2), err/bound 4.3e5.
#[test]
fn rohf_fitted_hvp_matches_finite_difference_of_the_fitted_gradient() {
    use ferric_scf::rohf_newton::{hessian_matvec, RohfNewtonInputs};
    let (mol, prep, aux, res) = converged_scf(NH2, 2, true);
    let n = prep.nbasis();
    let no = 1;
    let nc = (mol.nelec() as usize - no) / 2;
    let na = nc + no;
    let nv = n - na;
    let c = res.mos_alpha.clone();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let pool = EnginePool::new(op, &prep, 1e-14).unwrap();
    let g = gamma(n * n + aux.nbasis() + 2 * n);

    let mut rng = Xorshift64(0xD1B54A32D192ED03);
    let mut rand = |r, cc| Array2::from_shape_fn((r, cc), |_| 0.01 * rng.next_f64());
    let (k_vc, k_vo, k_oc) = (rand(nv, nc), rand(nv, no), rand(no, nc));
    let mut kap = Array2::<f64>::zeros((n, n));
    put_block(&mut kap, &k_vc, na, 0);
    put_block(&mut kap, &k_vo, na, nc);
    put_block(&mut kap, &k_oc, nc, 0);

    let blocks = |a: &Array2<f64>, b: &Array2<f64>| -> [Array2<f64>; 3] {
        [
            (a + b).slice(s![na.., ..nc]).to_owned(),
            a.slice(s![na.., nc..na]).to_owned(),
            b.slice(s![nc..na, ..nc]).to_owned(),
        ]
    };

    let mut fit = Fit::new(&prep, &aux);
    let dfj_h =
        RefCell::new(DfJ::new_with_precision(op, &prep, &aux, budget, RiJPrecision::F64).unwrap());
    let dfk_h = RefCell::new(DfK::new(op, &prep, &aux, budget).unwrap());
    let (jr, kr) = lend(&dfj_h, &dfk_h);

    for &k_mix in &[1.0, 0.2, 0.0] {
        let [(fa, ma), (fb, mb)] = fit.focks(k_mix, (&c, na), (&c, nc));
        let fa_mo = c.t().dot(&fa).dot(&c);
        let fb_mo = c.t().dot(&fb).dot(&c);
        let inp = RohfNewtonInputs {
            prep: &prep,
            bounds: &bounds,
            c: &c,
            f_a_mo: &fa_mo,
            f_b_mo: &fb_mo,
            nocc_double: nc,
            nocc_open: no,
            k_mix_sr: k_mix,
            rsh: None,
            fxc: None,
            thresh: 1e-12,
            ooc_budget: budget,
            j_response: Some(&*jr),
            k_response: (k_mix != 0.0).then_some(&*kr),
        };
        let dir0 = direct_response_builds_on_this_thread();
        let (h_vc, h_vo, h_oc) = hessian_matvec(&ctx, &inp, &k_vc, &k_vo, &k_oc, &pool).unwrap();
        assert_eq!(
            direct_response_builds_on_this_thread() - dir0,
            0,
            "k_mix={k_mix}: with RI-J (and DF-K) lent, no direct four-centre build may run"
        );
        let inp_dir = RohfNewtonInputs {
            j_response: None,
            k_response: None,
            ..inp
        };
        let (d_vc, d_vo, d_oc) =
            hessian_matvec(&ctx, &inp_dir, &k_vc, &k_vo, &k_oc, &pool).unwrap();

        // corr = Jacobian − Hessian. Since 2026-11 the matvec is the exact
        // (symmetric) Hessian of E(C exp κ); the FD here is the Jacobian of
        // the half gradient, which differs from it by ¼ read-off(Σ_σ [X, G_σ]),
        // G_σ[i,j] = (n_i − n_j) f_σ[i,j] (Jacobi identity; derived and
        // FD-verified off stationarity in `rohf_hessian_matvec_fd.rs`). At this
        // converged point the term is O(|g|·|κ|), but it is kept exact.
        // (Before the fix this held the omitted commutator − diagonal gap.)
        let occ_g = |f: &Array2<f64>, nocc: usize| {
            Array2::from_shape_fn((n, n), |(i, j)| {
                let o = |k: usize| if k < nocc { 1.0 } else { 0.0 };
                (o(i) - o(j)) * f[(i, j)]
            })
        };
        let (ga, gb) = (occ_g(&fa_mo, na), occ_g(&fb_mo, nc));
        let m = kap.dot(&ga) - ga.dot(&kap) + kap.dot(&gb) - gb.dot(&kap);
        let readoff = -0.25 * (&m.t() - &m);
        let corr = [
            readoff.slice(s![na.., ..nc]).to_owned(),
            readoff.slice(s![na.., nc..na]).to_owned(),
            readoff.slice(s![nc..na, ..nc]).to_owned(),
        ];
        let cabs = c.mapv(f64::abs);
        let e_g = blocks(&cabs.t().dot(&ma).dot(&cabs), &cabs.t().dot(&mb).dot(&cabs))
            .iter()
            .map(frob)
            .sum::<f64>()
            * g;

        let mut fd = |h: f64| -> Vec<Array2<f64>> {
            let mut at = |e: f64| {
                let r = cayley(&c, &(e * &kap));
                let [(fa, _), (fb, _)] = fit.focks(k_mix, (&r, na), (&r, nc));
                blocks(&r.t().dot(&fa).dot(&r), &r.t().dot(&fb).dot(&r))
            };
            let p = at(h);
            let m = at(-h);
            (0..3).map(|i| (&p[i] - &m[i]) / (2.0 * h)).collect()
        };
        let label = format!("ROHF k_mix={k_mix}");
        let an = [h_vc.clone(), h_vo.clone(), h_oc.clone()];
        let worst = check_against_fd(&label, &an, &corr, e_g, &mut fd);
        let fit_vs_dir = [(&h_vc, &d_vc), (&h_vo, &d_vo), (&h_oc, &d_oc)]
            .iter()
            .map(|(a, b)| frob(&(*a - *b)).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = an.iter().map(|a| frob(a).powi(2)).sum::<f64>().sqrt();
        eprintln!(
            "{label}: fitted vs direct HVP |diff|={fit_vs_dir:.3e} (rel {:.2e}), worst err/bound={worst:.3}",
            fit_vs_dir / scale
        );
        assert!(
            worst <= 1.0,
            "{label}: fitted HVP outside the FD bound (err/bound {worst:.3})"
        );
        // Sanity ceiling, not a precision claim: the fitted and direct HVPs
        // differ by the JK-fit error, measured at 4.5e-5..1.6e-4 relative.
        let rel = fit_vs_dir / scale;
        assert!(
            rel < 1e-3,
            "{label}: fitted vs direct HVP rel {rel:.2e} >= 1e-3"
        );
    }
}

/// Newton-converged vs DIIS-converged energy on a fitted SCF, and the Newton
/// HVPs' direct-build count. Bar 1e-7 Ha as in the closed-shell tests: both
/// runs minimise the same fitted functional on the same grid and stop at
/// energy_conv 1e-9 / density_conv 1e-7 on their own paths.
///
/// Measured (2026-11-11, cc-pVDZ, def2-universal-jkfit): UHF OH |ΔE| 2.8e-14
/// (13 HVPs), UKS OH PBE 1.4e-11 (60), UKS OH B3LYP 4.9e-11 (57), ROHF OH
/// 5.7e-14 (33), ROKS H2 ³Σ B3LYP 2.0e-15 (1); 0 direct builds in all.
///
/// Mutation (applied and reverted, 2026-11-11): passing `None` for the fitted
/// responses at the solver's Newton call site (the pre-fix direct response)
/// fails at the direct-build assert: UHF 42, UKS PBE 60 (δJ only: no K is built
/// for a pure functional), UKS B3LYP 198 (in `solve_uhf`); ROHF OH 114 (in
/// `solve_rohf`). The energies agree either way.
fn newton_matches_diis(xyz: &str, mult: usize, xc: Option<&str>, hybrid: bool, rohf: bool) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg_diis = RhfConfig {
        xc: xc.map(Into::into),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        // As in uhf_newton_smoke / rohf_newton_smoke: the UKS tests shift,
        // the ROKS ones do not (the ROKS Newton step uses its own shift floor).
        level_shift: if rohf { 0.0 } else { 0.2 },
        df_j_aux: Some(AUX.into()),
        df_k_aux: hybrid.then(|| AUX.into()),
        ..Default::default()
    };
    let cfg_newton = RhfConfig {
        newton_trigger: 1e-2,
        ..cfg_diis.clone()
    };
    let solve = |cfg: &RhfConfig| {
        if rohf {
            ferric_scf::rohf::solve_rohf(&ctx, &mol, &prep, op, &bounds, cfg).unwrap()
        } else {
            ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, cfg).unwrap()
        }
    };
    let hvp_count = || {
        if rohf {
            ferric_scf::rohf_newton::hvp_calls_on_this_thread()
        } else {
            ferric_scf::uhf_newton::hvp_calls_on_this_thread()
        }
    };
    let r_diis = solve(&cfg_diis);
    let (h0, d0) = (hvp_count(), direct_response_builds_on_this_thread());
    let r_newton = solve(&cfg_newton);
    let hvps = hvp_count() - h0;
    let direct = direct_response_builds_on_this_thread() - d0;
    let de = (r_diis.energy - r_newton.energy).abs();
    let kind = if rohf { "RO" } else { "U" };
    eprintln!(
        "{kind} {} {:?}: E_diis={:.10} ({} it) E_newton={:.10} ({} it) dE={de:.2e} hvps={hvps} direct={direct}",
        mol.atoms.len(),
        xc,
        r_diis.energy,
        r_diis.iterations,
        r_newton.energy,
        r_newton.iterations
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
fn uhf_newton_fitted_response_oh_hf() {
    newton_matches_diis(OH, 2, None, true, false);
}

#[test]
fn uks_newton_fitted_response_oh_pbe() {
    newton_matches_diis(OH, 2, Some("PBE"), false, false);
}

#[test]
fn uks_newton_fitted_response_oh_b3lyp() {
    newton_matches_diis(OH, 2, Some("B3LYP"), true, false);
}

#[test]
fn rohf_newton_fitted_response_oh_hf() {
    newton_matches_diis(OH, 2, None, true, true);
}

/// H2 ³Σ (the system `rohf_newton_smoke` runs ROKS Newton on), not NH2 or OH:
/// at these settings (newton_trigger 1e-2) ROKS/B3LYP Newton on NH2/cc-pVDZ
/// and OH/cc-pVDZ does not converge in 200 iterations with the old direct
/// response or the fitted one (DIIS: 9 and 18; measured 2026-11-11, counts in
/// the gating comment in `solve_rohf`, which also records NH2/6-31G PBE
/// Newton 185 vs DIIS 8). NH2 is covered by the ROHF FD anchor above.
#[test]
fn roks_newton_fitted_response_h2_triplet_b3lyp() {
    newton_matches_diis(H2_TRIPLET, 3, Some("B3LYP"), true, true);
}

/// A Newton step at a converged point computes no HVP (UHF and ROHF), and a
/// forced CG solve (cg_conv = 0) does: the positive control.
///
/// Measured (2026-11-11): UHF OH |g|max 8.8e-10, ROHF NH2 1.9e-9 (cg_conv
/// 1e-7): 0 HVPs each; forced: 11 and 9.
///
/// Mutation (applied and reverted, 2026-11-11): the old unconditional
/// `h0 = H·κ⁰` before the PCG loop fails with a count of 1 (UHF; and ROHF with
/// the UHF half unmutated).
#[test]
fn open_shell_newton_step_computes_no_hvp_at_a_converged_point() {
    use ferric_scf::rohf_newton::{rohf_newton_step, RohfNewtonInputs};
    use ferric_scf::uhf_newton::{uhf_newton_step, UhfNewtonInputs};
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let cg_conv = 1e-7;

    // UHF/OH.
    let (mol, prep, aux, res) = converged_scf(OH, 2, false);
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let na = (mol.nelec() as usize).div_ceil(2);
    let nb = mol.nelec() as usize / 2;
    let ca = res.mos_alpha.clone();
    let cb = res.mos_beta.clone().unwrap();
    let mut fit = Fit::new(&prep, &aux);
    let [(fa, _), (fb, _)] = fit.focks(1.0, (&ca, na), (&cb, nb));
    let (fa_mo, fb_mo) = (ca.t().dot(&fa).dot(&ca), cb.t().dot(&fb).dot(&cb));
    let gmax = fa_mo
        .slice(s![na.., ..na])
        .iter()
        .chain(fb_mo.slice(s![nb.., ..nb]).iter())
        .fold(0.0f64, |m, &v| m.max(v.abs()));
    let inp = UhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c_a: &ca,
        c_b: &cb,
        f_a_mo: &fa_mo,
        f_b_mo: &fb_mo,
        nocc_a: na,
        nocc_b: nb,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: 1e-12,
        ooc_budget: budget,
        j_response: None,
        k_response: None,
    };
    let h0 = ferric_scf::uhf_newton::hvp_calls_on_this_thread();
    uhf_newton_step(&ctx, &inp, 1e-6, 0.2, 20, cg_conv).unwrap();
    let at_conv = ferric_scf::uhf_newton::hvp_calls_on_this_thread() - h0;
    let h0 = ferric_scf::uhf_newton::hvp_calls_on_this_thread();
    uhf_newton_step(&ctx, &inp, 1e-6, 0.2, 20, 0.0).unwrap();
    let forced = ferric_scf::uhf_newton::hvp_calls_on_this_thread() - h0;
    eprintln!("UHF: |g|max={gmax:.2e} HVPs at converged={at_conv} forced={forced}");
    assert!(
        gmax < cg_conv,
        "precondition: converged point (|g|max {gmax:.2e})"
    );
    assert_eq!(
        at_conv, 0,
        "UHF: a converged Newton step must compute no HVP"
    );
    assert!(
        forced >= 1,
        "UHF positive control: a forced CG solve must count HVPs"
    );

    // ROHF/NH2.
    let (mol, prep, aux, res) = converged_scf(NH2, 2, true);
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let no = 1;
    let nc = (mol.nelec() as usize - no) / 2;
    let c = res.mos_alpha.clone();
    let mut fit = Fit::new(&prep, &aux);
    let [(fa, _), (fb, _)] = fit.focks(1.0, (&c, nc + no), (&c, nc));
    let (fa_mo, fb_mo) = (c.t().dot(&fa).dot(&c), c.t().dot(&fb).dot(&c));
    let inp = RohfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c: &c,
        f_a_mo: &fa_mo,
        f_b_mo: &fb_mo,
        nocc_double: nc,
        nocc_open: no,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: 1e-12,
        ooc_budget: budget,
        j_response: None,
        k_response: None,
    };
    let (g_vc, g_vo, g_oc) = (
        (&fa_mo + &fb_mo).slice(s![nc + no.., ..nc]).to_owned(),
        fa_mo.slice(s![nc + no.., nc..nc + no]).to_owned(),
        fb_mo.slice(s![nc..nc + no, ..nc]).to_owned(),
    );
    let gmax = [g_vc, g_vo, g_oc]
        .iter()
        .flat_map(|a| a.iter().copied().collect::<Vec<_>>())
        .fold(0.0f64, |m, v| m.max(v.abs()));
    let h0 = ferric_scf::rohf_newton::hvp_calls_on_this_thread();
    rohf_newton_step(&ctx, &inp, 1e-6, 0.1, 20, cg_conv).unwrap();
    let at_conv = ferric_scf::rohf_newton::hvp_calls_on_this_thread() - h0;
    let h0 = ferric_scf::rohf_newton::hvp_calls_on_this_thread();
    rohf_newton_step(&ctx, &inp, 1e-6, 0.1, 20, 0.0).unwrap();
    let forced = ferric_scf::rohf_newton::hvp_calls_on_this_thread() - h0;
    eprintln!("ROHF: |g|max={gmax:.2e} HVPs at converged={at_conv} forced={forced}");
    assert!(
        gmax < cg_conv,
        "precondition: converged point (|g|max {gmax:.2e})"
    );
    assert_eq!(
        at_conv, 0,
        "ROHF: a converged Newton step must compute no HVP"
    );
    assert!(
        forced >= 1,
        "ROHF positive control: a forced CG solve must count HVPs"
    );
}
