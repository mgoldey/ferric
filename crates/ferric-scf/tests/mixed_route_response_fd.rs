//! Second-order products on MIXED J/K routes: the Hessian-vector product
//! builds each half of the response with the builder the SCF used for that
//! half of its Fock.
//!
//! Routes (the fitted half is f64; the other half is the SCF's own unfitted
//! builder, made homogeneous in δD by `rhf_newton::unfitted_response`):
//!   * `rijcosx`      RI-J + COSX K
//!   * `dfj+directk`  RI-J + `DirectK`
//!   * `dfk+directj`  `DirectJ` + DF-K
//!
//! Exactness anchors: for each route, the closed-shell singlet matvec
//! (`hessian_matvec_jk`, used by Newton, TRAH and singlet stability), the
//! triplet matvec (`triplet_hessian_matvec_jk`, external stability), the UHF
//! matvec (Newton, TRAH, stability) and the ROHF matvec (Newton, AH) equal the
//! central finite difference of THAT route's orbital gradient, built from the
//! same builders, to a derived bound, with no direct four-centre build.
//!
//! COSX runs here in its trivial-screen limit (no density-driven pair screen,
//! dense half transform; overlap fit ON as in production): its screens switch
//! on the density, so with them on `K(D ± hδD)` drops different pairs on the
//! two sides of the difference, and that error (~1e-7/h) is not a property of
//! the linear response. In this limit COSX is an exactly linear map, which is
//! what the anchor needs.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{CosxConfig, CosxHalfTransform, CosxK};
use ferric_scf::df_j::{DfJ, RiJPrecision};
use ferric_scf::df_k::{DfK, RiKPrecision};
use ferric_scf::direct_j::DirectJ;
use ferric_scf::direct_k::DirectK;
use ferric_scf::engine_pool::EnginePool;
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rhf_newton::unfitted_response_builds_on_this_thread;
use ferric_scf::screening::SchwarzBounds;
use ndarray::{s, Array2};
use ndarray_linalg::Solve;

const AUX: &str = "def2-universal-jkfit";
const WATER: &str = "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n";
const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";
const NH2: &str = "3\nNH2\nN 0 0 0\nH 0 0.80 -0.62\nH 0 -0.80 -0.62\n";
/// Integral screening threshold of the direct builders (Fock and response).
const THRESH: f64 = 1e-12;

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

/// Antisymmetric κ with `blk` at rows `r0..`, cols `c0..` (and −blkᵀ mirrored).
fn put_block(kappa: &mut Array2<f64>, blk: &Array2<f64>, r0: usize, c0: usize) {
    for ((i, j), &v) in blk.indexed_iter() {
        kappa[(r0 + i, c0 + j)] = v;
        kappa[(c0 + j, r0 + i)] = -v;
    }
}

/// γ_m = m·u/(1 − m·u), the relative rounding bound of an m-term computation.
fn gamma(m: usize) -> f64 {
    let u = f64::EPSILON / 2.0;
    let m = m as f64;
    m * u / (1.0 - m * u)
}

/// `[Fκ − κF]_vo − (F_aa − F_ii)·k`: the exact MO-rotation commutator minus
/// the diagonal gap the RHF/UHF matvecs carry, for one occ→virt block.
fn gap_correction(
    f_mo: &Array2<f64>,
    kap: &Array2<f64>,
    k: &Array2<f64>,
    no: usize,
) -> Array2<f64> {
    let comm = f_mo.dot(kap) - kap.dot(f_mo);
    let mut c = comm.slice(s![no.., ..no]).to_owned();
    for ((r, i), v) in c.indexed_iter_mut() {
        *v -= (f_mo[(no + r, no + r)] - f_mo[(i, i)]) * k[(r, i)];
    }
    c
}

/// Worst `‖Hκ − FD(h)‖ / bound(h)` over the ladder h = 4e-3, 2e-3, 1e-3.
///
/// Bound at step h (Frobenius, derived): a central difference has error
/// `c·h² + O(h⁴)` plus rounding `≤ e_g/h`, so with h halving,
/// `‖Hκ − FD(h)‖ ≤ ‖FD(2h) − FD(h)‖/3 + 2·e_g/h`, dropping O(h⁴) (the ladder
/// keeps truncation far below rounding). `analytic + corr` is the exact
/// derivative (the matvec's diagonal gap replaced by the full commutator).
/// Also returns the bound at the finest step.
fn check_against_fd(
    label: &str,
    analytic: &[Array2<f64>],
    corr: &[Array2<f64>],
    e_g: f64,
    fd: &mut dyn FnMut(f64) -> Vec<Array2<f64>>,
) -> (f64, f64) {
    let pack = |v: &[Array2<f64>]| -> Array2<f64> {
        let flat: Vec<f64> = v.iter().flat_map(|a| a.iter().copied()).collect();
        Array2::from_shape_vec((1, flat.len()), flat).unwrap()
    };
    let hk = &pack(analytic) + &pack(corr);
    let ladder = [4e-3, 2e-3, 1e-3];
    let fds: Vec<Array2<f64>> = ladder.iter().map(|&h| pack(&fd(h))).collect();
    let scale = frob(&hk);
    let mut worst = 0.0f64;
    let mut last = 0.0;
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
        last = trunc + round;
    }
    (worst, last)
}

/// The rounding bound of one gradient block: `γ_m·‖[|C|ᵀ·mag·|C|]_vo‖`.
fn rounding(c: &Array2<f64>, mag: &Array2<f64>, no: usize, m: usize) -> f64 {
    let a = c.mapv(f64::abs);
    frob(&a.t().dot(mag).dot(&a).slice(s![no.., ..no]).to_owned()) * gamma(m)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Route {
    RiJCosx,
    DfJDirectK,
    DfKDirectJ,
}

const ROUTES: [Route; 3] = [Route::RiJCosx, Route::DfJDirectK, Route::DfKDirectJ];

/// The COSX configuration of the anchors: the flat (50,110) grid with the
/// production overlap fit, its density-dependent screens at their trivial
/// limit (see the module doc), no final-grid pass.
///
/// The grid is deliberately COARSE, (20, 26) unpruned: the anchor must be
/// able to tell COSX's δK from the exact four-centre δK the old code built,
/// i.e. their difference (the grid error) must exceed the FD bound, whose
/// rounding term grows with the number of grid points. On flat (50, 110) the
/// two differ by 2.1e-6 against a bound of 2.6e-6 (measured 2026-11-11), too
/// close to separate. The map is linear on any grid, so a coarse one tests
/// the same construction.
fn cosx_anchor_config() -> CosxConfig {
    let mut cfg = CosxConfig {
        screen_thresh: None,
        half_transform: CosxHalfTransform::Dense,
        final_grid: None,
        ..CosxConfig::flat_reference()
    };
    cfg.grid.n_radial = 20;
    cfg.grid.n_angular = 26;
    cfg
}

/// One route's J and K builders: a fitted half and an unfitted half.
struct Builders<'p> {
    route: Route,
    dfj: Option<DfJ<'p>>,
    dfk: Option<DfK<'p>>,
    dj: Option<DirectJ<'p>>,
    kb: Option<Box<dyn KBuilder + 'p>>,
    /// Inner-product length bound for one Fock element (see `rounding`).
    m_fock: usize,
}

impl<'p> Builders<'p> {
    fn new(
        route: Route,
        (ctx, mol): (&'p ParallelContext, &'p Molecule),
        (prep, aux, bounds): (&'p PreparedBasis, &'p PreparedBasis, &'p SchwarzBounds),
    ) -> Self {
        let budget = ferric_core::memory::resolve_budget_bytes(None);
        let op = Operator::coulomb();
        let n = prep.nbasis();
        let dfj = || DfJ::new_with_precision(op, prep, aux, budget, RiJPrecision::F64).unwrap();
        let dfk = || DfK::new(op, prep, aux, budget).unwrap();
        let fit_m = n * n + aux.nbasis() + 2 * n;
        match route {
            Route::RiJCosx => {
                let cosx = CosxK::new(ctx, mol, prep, cosx_anchor_config(), budget).unwrap();
                // K = Σ_g X·A(g)·(D·X): sums of length n, n and npts, then
                // the S_num fit (n) and the MO transform (n²).
                let m = (n * n + 3 * n + cosx.npts()).max(fit_m);
                Builders {
                    route,
                    dfj: Some(dfj()),
                    dfk: None,
                    dj: None,
                    kb: Some(Box::new(cosx)),
                    m_fock: m,
                }
            }
            Route::DfJDirectK => Builders {
                route,
                dfj: Some(dfj()),
                dfk: None,
                dj: None,
                kb: Some(Box::new(DirectK::new(ctx, prep, bounds, THRESH, budget))),
                m_fock: fit_m,
            },
            Route::DfKDirectJ => Builders {
                route,
                dfj: None,
                dfk: Some(dfk()),
                dj: Some(DirectJ::new(ctx, prep, bounds, THRESH, budget)),
                kb: None,
                m_fock: fit_m,
            },
        }
    }

    /// J of the total density `d`, from this route's J builder.
    fn j(&mut self, d: &Array2<f64>) -> Array2<f64> {
        let mut j = Array2::<f64>::zeros(d.raw_dim());
        match (self.dfj.as_mut(), self.dj.as_mut()) {
            (Some(b), _) => b.build(d, &mut j).unwrap(),
            (None, Some(b)) => b.build(d, &mut j).unwrap(),
            (None, None) => unreachable!(),
        };
        j
    }

    /// K of `C_o·C_oᵀ`, as the SCF builds it: DF-K through the occupied
    /// factor, an unfitted builder on the density.
    fn k(&mut self, c_o: &Array2<f64>) -> Array2<f64> {
        let n = c_o.nrows();
        let mut k = Array2::<f64>::zeros((n, n));
        if c_o.ncols() == 0 {
            return k;
        }
        match (self.dfk.as_mut(), self.kb.as_mut()) {
            (Some(b), _) => {
                b.build_from_occ(c_o, &mut k).unwrap();
            }
            (None, Some(b)) => {
                let d = c_o.dot(&c_o.t());
                b.update_density(&d);
                b.build(&d, &mut k).unwrap();
            }
            (None, None) => unreachable!(),
        }
        k
    }

    /// Spin Focks `F_σ = H + J[D_α + D_β] − k·K[D_σ]` and the magnitudes
    /// `|H| + |J| + k|K_σ|` of each (for the rounding bound).
    fn spin_focks(
        &mut self,
        h: &Array2<f64>,
        k_mix: f64,
        (ca, na): (&Array2<f64>, usize),
        (cb, nb): (&Array2<f64>, usize),
    ) -> [(Array2<f64>, Array2<f64>); 2] {
        let oa = ca.slice(s![.., ..na]).to_owned();
        let ob = cb.slice(s![.., ..nb]).to_owned();
        let j = self.j(&(oa.dot(&oa.t()) + ob.dot(&ob.t())));
        let mut spin = |o: &Array2<f64>| {
            let k = if k_mix != 0.0 {
                self.k(o)
            } else {
                Array2::<f64>::zeros(h.raw_dim())
            };
            let f = h + &j - &(k_mix * &k);
            let mag = h.mapv(f64::abs) + j.mapv(f64::abs) + k_mix.abs() * k.mapv(f64::abs);
            (f, mag)
        };
        let fa = spin(&oa);
        let fb = spin(&ob);
        [fa, fb]
    }

    /// The closed-shell Fock `H + J[2·C_oC_oᵀ] − ½k·K[2·C_oC_oᵀ]` (= either
    /// spin Fock at α = β) and its magnitudes.
    fn rhf_fock(
        &mut self,
        h: &Array2<f64>,
        k_mix: f64,
        c: &Array2<f64>,
        no: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        let [fa, _] = self.spin_focks(h, k_mix, (c, no), (c, no));
        fa
    }

    /// The route's builders lent to the closed-shell products, exactly as
    /// `solve_rhf` lends them (`lend_scf_jk`).
    fn closed_shell(&mut self) -> ferric_scf::rhf_newton::ScfJkResponse<'_, 'p> {
        ferric_scf::rhf_newton::ScfJkResponse::new(self.dfj.as_mut(), self.dfk.as_mut())
            .with_unfitted_j(self.dj.as_mut().map(|j| j as &mut (dyn JBuilder + 'p)))
            .with_unfitted_k(self.kb.as_deref_mut())
    }

    /// The OLD lending for this route (before the mixed routes were wired):
    /// only the fitted half, so the other half falls to the direct build.
    fn closed_shell_fitted_only(&mut self) -> ferric_scf::rhf_newton::ScfJkResponse<'_, 'p> {
        ferric_scf::rhf_newton::ScfJkResponse::new(self.dfj.as_mut(), self.dfk.as_mut())
    }

    /// [`Self::closed_shell_fitted_only`] for the open-shell hooks.
    fn open_shell_fitted_only(&mut self) -> ferric_scf::uhf_newton::ScfResponses<'_> {
        ferric_scf::uhf_newton::ScfResponses::lend(
            self.dfj.as_mut().map(|j| j as &mut dyn JBuilder),
            self.dfk.as_mut(),
            None,
            None,
        )
    }

    /// The route's builders as the open-shell response hooks, exactly as
    /// `solve_uhf`/`solve_rohf` lend them (`uhf::scf_responses`).
    fn open_shell(&mut self) -> ferric_scf::uhf_newton::ScfResponses<'_> {
        ferric_scf::uhf_newton::ScfResponses::lend(
            self.dfj.as_mut().map(|j| j as &mut dyn JBuilder),
            self.dfk.as_mut(),
            self.dj.as_mut().map(|j| j as &mut dyn JBuilder),
            self.kb.as_deref_mut(),
        )
    }

    /// Whether this route's matvec at `k_mix` is a DIFFERENT operator from
    /// the old one (fitted half lent, other half direct), so the FD anchor
    /// itself can tell them apart: only COSX's δK differs from the exact δK
    /// (by its grid error). `DirectJ`/`DirectK` are the exact four-centre
    /// operator, differing from the old direct build only in their screening
    /// (≲ 1e-12); on those routes the counters are what catch a mis-wiring.
    fn differs_from_old(&self, k_mix: f64) -> bool {
        self.route == Route::RiJCosx && k_mix != 0.0
    }
}

struct System {
    mol: Molecule,
    prep: PreparedBasis,
    aux: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
    res: ferric_scf::result::ScfResult,
}

/// A fitted SCF supplies sensible MOs (the anchor holds at any C).
fn system(xyz: &str, mult: usize, kind: &str) -> System {
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
    let res = match kind {
        "rhf" => ferric_scf::rhf::solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg),
        "uhf" => ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &cfg),
        "rohf" => ferric_scf::rohf::solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg),
        _ => unreachable!(),
    }
    .unwrap();
    assert!(res.converged);
    System {
        mol,
        prep,
        aux,
        bounds,
        ctx,
        res,
    }
}

/// The anchor's verdict for one case, with the counters. `route_vs_old` is
/// `‖H_route·κ − H_old·κ‖` with `H_old` the old lending (fitted half only).
fn assert_case(
    label: &str,
    (worst, bound): (f64, f64),
    route_vs_old: f64,
    differs: bool,
    (direct, unfitted): (usize, usize),
    expect_unfitted: bool,
) {
    eprintln!(
        "{label}: route vs old HVP |diff|={route_vs_old:.3e}, worst err/bound={worst:.3} \
         direct_builds={direct} unfitted_builds={unfitted}"
    );
    assert!(
        worst <= 1.0,
        "{label}: route HVP outside the FD bound (err/bound {worst:.3})"
    );
    assert_eq!(
        direct, 0,
        "{label}: no direct four-centre response build may run"
    );
    assert_eq!(
        unfitted > 0,
        expect_unfitted,
        "{label}: the route's unfitted builder must answer its half of the response \
         exactly when that half is consumed ({unfitted} unfitted builds)"
    );
    if differs {
        // The anchor can tell this route's operator from the old one.
        assert!(
            route_vs_old > bound,
            "{label}: route and old HVPs differ by {route_vs_old:.3e}, inside the FD \
             bound {bound:.3e}, so the anchor could not detect the old response"
        );
    }
}

/// Exactness anchor, closed-shell SINGLET matvec (`hessian_matvec_jk`, the
/// operator of the RHF/RKS Newton step, TRAH and internal stability): on each
/// mixed route the matvec with the route's builders lent equals the central
/// FD of that route's gradient `g = [CᵀFC]_vo`, at k_mix = 1 (HF), 0.2
/// (B3LYP's fraction) and 0 (no K built; the K builder is lent but unused).
#[test]
fn closed_shell_singlet_matvec_matches_fd_on_mixed_routes() {
    use ferric_scf::rhf_newton::{
        direct_response_builds_on_this_thread, hessian_matvec_jk, RhfNewtonInputs,
    };
    let sys = system(WATER, 1, "rhf");
    let (prep, n) = (&sys.prep, sys.prep.nbasis());
    let no = sys.mol.nelec() as usize / 2;
    let c = sys.res.mos_alpha.clone();
    let h = oneelectron::hcore(prep);
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let pool = EnginePool::new(Operator::coulomb(), prep, 1e-14).unwrap();
    let mut rng = Xorshift64(0x9E3779B97F4A7C15);
    let k = Array2::from_shape_fn((n - no, no), |_| 0.01 * rng.next_f64());
    let mut kap = Array2::<f64>::zeros((n, n));
    put_block(&mut kap, &k, no, 0);

    for route in ROUTES {
        let mut b = Builders::new(route, (&sys.ctx, &sys.mol), (prep, &sys.aux, &sys.bounds));
        for &k_mix in &[1.0, 0.2, 0.0] {
            let (f, mag) = b.rhf_fock(&h, k_mix, &c, no);
            let f_mo = c.t().dot(&f).dot(&c);
            let inp = RhfNewtonInputs {
                prep,
                bounds: &sys.bounds,
                c: &c,
                f_mo: &f_mo,
                nocc: no,
                k_mix_sr: k_mix,
                rsh: None,
                fxc: None,
                thresh: THRESH,
                ooc_budget: budget,
            };
            let (dir0, unf0) = (
                direct_response_builds_on_this_thread(),
                unfitted_response_builds_on_this_thread(),
            );
            let hk = {
                let resp = b.closed_shell();
                hessian_matvec_jk(&sys.ctx, &inp, &k, &pool, Some(&resp)).unwrap()
            };
            let counts = (
                direct_response_builds_on_this_thread() - dir0,
                unfitted_response_builds_on_this_thread() - unf0,
            );
            let hk_old = {
                let resp = b.closed_shell_fitted_only();
                hessian_matvec_jk(&sys.ctx, &inp, &k, &pool, Some(&resp)).unwrap()
            };
            let corr = gap_correction(&f_mo, &kap, &k, no);
            let e_g = rounding(&c, &mag, no, b.m_fock);
            let mut fd = |hh: f64| -> Vec<Array2<f64>> {
                let mut at = |e: f64| {
                    let r = cayley(&c, &(e * &kap));
                    let (ff, _) = b.rhf_fock(&h, k_mix, &r, no);
                    r.t().dot(&ff).dot(&r).slice(s![no.., ..no]).to_owned()
                };
                vec![(&at(hh) - &at(-hh)) / (2.0 * hh)]
            };
            let label = format!("RHF singlet {route:?} k_mix={k_mix}");
            let res = check_against_fd(&label, std::slice::from_ref(&hk), &[corr], e_g, &mut fd);
            let expect_unfitted = match route {
                Route::DfKDirectJ => true,
                _ => k_mix != 0.0,
            };
            assert_case(
                &label,
                res,
                frob(&(&hk - &hk_old)),
                b.differs_from_old(k_mix),
                counts,
                expect_unfitted,
            );
        }
    }
}

/// Exactness anchor, closed-shell TRIPLET matvec (`triplet_hessian_matvec_jk`,
/// RHF→UHF external stability): with the route's exchange builder lent, it
/// equals the central FD of the UHF α gradient `[C_αᵀF_αC_α]_vo` along the
/// spin-flip rotation `C_α = C·U(hκ)`, `C_β = C·U(−hκ)`, where
/// `F_α = H + J[D_α + D_β] − K[D_α]` (HF, k_mix = 1). `δD_α + δD_β = 0`, so
/// no Coulomb response exists, and none may be built: on the DF-K + direct J
/// route the lent `DirectJ` must stay unused.
#[test]
fn closed_shell_triplet_matvec_matches_fd_on_mixed_routes() {
    use ferric_scf::rhf_newton::{
        direct_response_builds_on_this_thread, triplet_hessian_matvec_jk, RhfNewtonInputs,
    };
    let sys = system(WATER, 1, "rhf");
    let (prep, n) = (&sys.prep, sys.prep.nbasis());
    let no = sys.mol.nelec() as usize / 2;
    let c = sys.res.mos_alpha.clone();
    let h = oneelectron::hcore(prep);
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let pool = EnginePool::new(Operator::coulomb(), prep, 1e-14).unwrap();
    let mut rng = Xorshift64(0x2545F4914F6CDD1D);
    let k = Array2::from_shape_fn((n - no, no), |_| 0.01 * rng.next_f64());
    let mut kap = Array2::<f64>::zeros((n, n));
    put_block(&mut kap, &k, no, 0);
    let k_mix = 1.0;

    for route in ROUTES {
        let mut b = Builders::new(route, (&sys.ctx, &sys.mol), (prep, &sys.aux, &sys.bounds));
        let [(fa, mag), _] = b.spin_focks(&h, k_mix, (&c, no), (&c, no));
        let f_mo = c.t().dot(&fa).dot(&c);
        let inp = RhfNewtonInputs {
            prep,
            bounds: &sys.bounds,
            c: &c,
            f_mo: &f_mo,
            nocc: no,
            k_mix_sr: k_mix,
            rsh: None,
            fxc: None,
            thresh: THRESH,
            ooc_budget: budget,
        };
        let (dir0, unf0) = (
            direct_response_builds_on_this_thread(),
            unfitted_response_builds_on_this_thread(),
        );
        let hk = {
            let resp = b.closed_shell();
            triplet_hessian_matvec_jk(&sys.ctx, &inp, &k, &pool, Some(&resp)).unwrap()
        };
        let counts = (
            direct_response_builds_on_this_thread() - dir0,
            unfitted_response_builds_on_this_thread() - unf0,
        );
        let hk_old = {
            let resp = b.closed_shell_fitted_only();
            triplet_hessian_matvec_jk(&sys.ctx, &inp, &k, &pool, Some(&resp)).unwrap()
        };
        let corr = gap_correction(&f_mo, &kap, &k, no);
        let e_g = rounding(&c, &mag, no, b.m_fock);
        let mut fd = |hh: f64| -> Vec<Array2<f64>> {
            let mut at = |e: f64| {
                let (ra, rb) = (cayley(&c, &(e * &kap)), cayley(&c, &(-e * &kap)));
                let [(ff, _), _] = b.spin_focks(&h, k_mix, (&ra, no), (&rb, no));
                ra.t().dot(&ff).dot(&ra).slice(s![no.., ..no]).to_owned()
            };
            vec![(&at(hh) - &at(-hh)) / (2.0 * hh)]
        };
        let label = format!("RHF triplet {route:?}");
        let res = check_against_fd(&label, std::slice::from_ref(&hk), &[corr], e_g, &mut fd);
        assert_case(
            &label,
            res,
            frob(&(&hk - &hk_old)),
            b.differs_from_old(k_mix),
            counts,
            // An unfitted builder answers only on the routes whose K is unfitted.
            route != Route::DfKDirectJ,
        );
    }
}

/// Exactness anchor, UHF/UKS matvec (`uhf_newton::hessian_matvec`, the
/// operator of the UHF Newton step, TRAH and stability): on each mixed route,
/// with the route's builders lent through `ScfResponses::lend`, it equals the
/// central FD of the per-spin gradients `[C_σᵀF_σC_σ]_vo` along a random
/// coupled rotation (κ_α, κ_β), at k_mix = 1, 0.2 and 0. OH ²Π.
#[test]
fn uhf_matvec_matches_fd_on_mixed_routes() {
    use ferric_scf::uhf_newton::{
        direct_response_builds_on_this_thread, hessian_matvec, UhfNewtonInputs,
    };
    let sys = system(OH, 2, "uhf");
    let (prep, n) = (&sys.prep, sys.prep.nbasis());
    let na = (sys.mol.nelec() as usize).div_ceil(2);
    let nb = sys.mol.nelec() as usize / 2;
    let ca = sys.res.mos_alpha.clone();
    let cb = sys.res.mos_beta.clone().expect("UHF has β MOs");
    let h = oneelectron::hcore(prep);
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let pool = EnginePool::new(Operator::coulomb(), prep, 1e-14).unwrap();
    let mut rng = Xorshift64(0x9E3779B97F4A7C15);
    let ka = Array2::from_shape_fn((n - na, na), |_| 0.01 * rng.next_f64());
    let kb = Array2::from_shape_fn((n - nb, nb), |_| 0.01 * rng.next_f64());
    let mut kap_a = Array2::<f64>::zeros((n, n));
    let mut kap_b = Array2::<f64>::zeros((n, n));
    put_block(&mut kap_a, &ka, na, 0);
    put_block(&mut kap_b, &kb, nb, 0);

    for route in ROUTES {
        let mut b = Builders::new(route, (&sys.ctx, &sys.mol), (prep, &sys.aux, &sys.bounds));
        for &k_mix in &[1.0, 0.2, 0.0] {
            let [(fa, ma), (fb, mb)] = b.spin_focks(&h, k_mix, (&ca, na), (&cb, nb));
            let fa_mo = ca.t().dot(&fa).dot(&ca);
            let fb_mo = cb.t().dot(&fb).dot(&cb);
            let inp_dir = UhfNewtonInputs {
                prep,
                bounds: &sys.bounds,
                c_a: &ca,
                c_b: &cb,
                f_a_mo: &fa_mo,
                f_b_mo: &fb_mo,
                nocc_a: na,
                nocc_b: nb,
                k_mix_sr: k_mix,
                rsh: None,
                fxc: None,
                thresh: THRESH,
                ooc_budget: budget,
                j_response: None,
                k_response: None,
            };
            let (dir0, unf0) = (
                direct_response_builds_on_this_thread(),
                unfitted_response_builds_on_this_thread(),
            );
            let (ha, hb) = {
                let resp = b.open_shell();
                let inp = UhfNewtonInputs {
                    j_response: resp.j(),
                    k_response: resp.k(),
                    ..inp_dir
                };
                hessian_matvec(&sys.ctx, &inp, &ka, &kb, &pool).unwrap()
            };
            let counts = (
                direct_response_builds_on_this_thread() - dir0,
                unfitted_response_builds_on_this_thread() - unf0,
            );
            let (da, db) = {
                let resp = b.open_shell_fitted_only();
                let inp = UhfNewtonInputs {
                    j_response: resp.j(),
                    k_response: resp.k(),
                    ..inp_dir
                };
                hessian_matvec(&sys.ctx, &inp, &ka, &kb, &pool).unwrap()
            };
            let corr = [
                gap_correction(&fa_mo, &kap_a, &ka, na),
                gap_correction(&fb_mo, &kap_b, &kb, nb),
            ];
            let e_g = rounding(&ca, &ma, na, b.m_fock).hypot(rounding(&cb, &mb, nb, b.m_fock));
            let mut fd = |hh: f64| -> Vec<Array2<f64>> {
                let mut at = |e: f64| {
                    let (ra, rb) = (cayley(&ca, &(e * &kap_a)), cayley(&cb, &(e * &kap_b)));
                    let [(fa, _), (fb, _)] = b.spin_focks(&h, k_mix, (&ra, na), (&rb, nb));
                    [
                        ra.t().dot(&fa).dot(&ra).slice(s![na.., ..na]).to_owned(),
                        rb.t().dot(&fb).dot(&rb).slice(s![nb.., ..nb]).to_owned(),
                    ]
                };
                let [pa, pb] = at(hh);
                let [ma_, mb_] = at(-hh);
                vec![(&pa - &ma_) / (2.0 * hh), (&pb - &mb_) / (2.0 * hh)]
            };
            let label = format!("UHF {route:?} k_mix={k_mix}");
            let res = check_against_fd(&label, &[ha.clone(), hb.clone()], &corr, e_g, &mut fd);
            let expect_unfitted = match route {
                Route::DfKDirectJ => true,
                _ => k_mix != 0.0,
            };
            assert_case(
                &label,
                res,
                frob(&(&ha - &da)).hypot(frob(&(&hb - &db))),
                b.differs_from_old(k_mix),
                counts,
                expect_unfitted,
            );
        }
    }
}

/// Exactness anchor, ROHF/ROKS matvec (`rohf_newton::hessian_matvec`, the
/// operator of the ROHF Newton and AH steps): on each mixed route it equals
/// the central FD of the HALF gradient blocks (g_vc = [F_α+F_β]_vc,
/// g_vo = [F_α]_vo, g_oc = [F_β]_oc, one MO set) once its diagonal gap is
/// replaced by the full commutator (as in
/// `open_shell_newton_fitted_response.rs`), at k_mix = 1, 0.2 and 0. NH2 ²B₁.
#[test]
fn rohf_matvec_matches_fd_on_mixed_routes() {
    use ferric_scf::rohf_newton::{hessian_matvec, RohfNewtonInputs};
    use ferric_scf::uhf_newton::direct_response_builds_on_this_thread;
    let sys = system(NH2, 2, "rohf");
    let (prep, n) = (&sys.prep, sys.prep.nbasis());
    let no = 1;
    let nc = (sys.mol.nelec() as usize - no) / 2;
    let na = nc + no;
    let nv = n - na;
    let c = sys.res.mos_alpha.clone();
    let h = oneelectron::hcore(prep);
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let pool = EnginePool::new(Operator::coulomb(), prep, 1e-14).unwrap();
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

    for route in ROUTES {
        let mut b = Builders::new(route, (&sys.ctx, &sys.mol), (prep, &sys.aux, &sys.bounds));
        for &k_mix in &[1.0, 0.2, 0.0] {
            let [(fa, ma), (fb, mb)] = b.spin_focks(&h, k_mix, (&c, na), (&c, nc));
            let fa_mo = c.t().dot(&fa).dot(&c);
            let fb_mo = c.t().dot(&fb).dot(&c);
            let inp_dir = RohfNewtonInputs {
                prep,
                bounds: &sys.bounds,
                c: &c,
                f_a_mo: &fa_mo,
                f_b_mo: &fb_mo,
                nocc_double: nc,
                nocc_open: no,
                k_mix_sr: k_mix,
                rsh: None,
                fxc: None,
                thresh: THRESH,
                ooc_budget: budget,
                j_response: None,
                k_response: None,
            };
            let (dir0, unf0) = (
                direct_response_builds_on_this_thread(),
                unfitted_response_builds_on_this_thread(),
            );
            let (h_vc, h_vo, h_oc) = {
                let resp = b.open_shell();
                let inp = RohfNewtonInputs {
                    j_response: resp.j(),
                    k_response: resp.k(),
                    ..inp_dir
                };
                hessian_matvec(&sys.ctx, &inp, &k_vc, &k_vo, &k_oc, &pool).unwrap()
            };
            let counts = (
                direct_response_builds_on_this_thread() - dir0,
                unfitted_response_builds_on_this_thread() - unf0,
            );
            let (d_vc, d_vo, d_oc) = {
                let resp = b.open_shell_fitted_only();
                let inp = RohfNewtonInputs {
                    j_response: resp.j(),
                    k_response: resp.k(),
                    ..inp_dir
                };
                hessian_matvec(&sys.ctx, &inp, &k_vc, &k_vo, &k_oc, &pool).unwrap()
            };

            // corr = commutator blocks − the matvec's diagonal gaps.
            let ca_ = fa_mo.dot(&kap) - kap.dot(&fa_mo);
            let cb_ = fb_mo.dot(&kap) - kap.dot(&fb_mo);
            let mut corr = blocks(&ca_, &cb_);
            let fa_d = |p: usize| fa_mo[(p, p)];
            let fb_d = |p: usize| fb_mo[(p, p)];
            for ((r, q), v) in corr[0].indexed_iter_mut() {
                let p = na + r;
                *v -= (fa_d(p) + fb_d(p) - fa_d(q) - fb_d(q)) * k_vc[(r, q)];
            }
            for ((r, j), v) in corr[1].indexed_iter_mut() {
                *v -= (fa_d(na + r) - fa_d(nc + j)) * k_vo[(r, j)];
            }
            for ((r, q), v) in corr[2].indexed_iter_mut() {
                *v -= (fb_d(nc + r) - fb_d(q)) * k_oc[(r, q)];
            }
            // One gradient evaluation's rounding: the vc block reads both spins.
            let a = c.mapv(f64::abs);
            let g = gamma(b.m_fock);
            let [ra, rb] = [a.t().dot(&ma).dot(&a), a.t().dot(&mb).dot(&a)];
            let rblk = blocks(&ra, &rb);
            let e_g = g * rblk.iter().map(|x| frob(x).powi(2)).sum::<f64>().sqrt();

            let mut fd = |hh: f64| -> Vec<Array2<f64>> {
                let mut at = |e: f64| {
                    let r = cayley(&c, &(e * &kap));
                    let [(fa, _), (fb, _)] = b.spin_focks(&h, k_mix, (&r, na), (&r, nc));
                    blocks(&r.t().dot(&fa).dot(&r), &r.t().dot(&fb).dot(&r))
                };
                let p = at(hh);
                let m = at(-hh);
                (0..3).map(|i| (&p[i] - &m[i]) / (2.0 * hh)).collect()
            };
            let label = format!("ROHF {route:?} k_mix={k_mix}");
            let res = check_against_fd(
                &label,
                &[h_vc.clone(), h_vo.clone(), h_oc.clone()],
                &corr,
                e_g,
                &mut fd,
            );
            let route_vs_dir = [(&h_vc, &d_vc), (&h_vo, &d_vo), (&h_oc, &d_oc)]
                .iter()
                .map(|(x, y)| frob(&(*x - *y)).powi(2))
                .sum::<f64>()
                .sqrt();
            let expect_unfitted = match route {
                Route::DfKDirectJ => true,
                _ => k_mix != 0.0,
            };
            assert_case(
                &label,
                res,
                route_vs_dir,
                b.differs_from_old(k_mix),
                counts,
                expect_unfitted,
            );
        }
    }
}
