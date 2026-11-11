//! MEASUREMENT harness (ignored): wall time of one second-order product on
//! each J/K route, with the response J/K built the old way and the new way.
//!
//! "Before" is the exact pre-change code path, reached through the same API:
//! * mixed routes: only the fitted half lent (`ScfJkResponse::new(j, k)` /
//!   `ScfResponses::lend(j, k, None, None)`), so the other half fell to the
//!   combined direct four-centre build;
//! * fitted route, TRAH / stability: nothing lent (the direct build both used).
//!
//! The `RiJCosx` rows time the option NOT taken: lending the SCF's COSX K.
//! It measured slower than the direct build it would replace (2026-11-11,
//! C8H18/def2-TZVP: RHF 13.2 → 17.0 s, UHF 24.0 → 30.2-31.3 s per product; one
//! warmed COSX build 15-16 s vs 13 s for the combined direct J+K), so the SCF
//! does not lend it (`rhf::lend_scf_jk`). Re-run these rows if COSX gets
//! cheaper.
//!
//! Run (one heavy op at a time; box-lease serialises):
//!   OPENBLAS_NUM_THREADS=1 box-lease exec -- cargo test --release -p ferric-scf \
//!     --test hvp_route_timing -- --ignored --nocapture
//! `FERRIC_HVP_XYZ` (default testdata/molecules/alkane_8.xyz) and
//! `FERRIC_HVP_BASIS` (default def2-tzvp) pick the system. The open-shell
//! products run on the radical made by dropping the molecule's last atom.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{CosxConfig, CosxK};
use ferric_scf::df_j::{DfJ, RiJPrecision};
use ferric_scf::df_k::DfK;
use ferric_scf::direct_j::DirectJ;
use ferric_scf::direct_k::DirectK;
use ferric_scf::engine_pool::EnginePool;
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rhf_newton::{
    hessian_matvec_jk, triplet_hessian_matvec_jk, RhfNewtonInputs, ScfJkResponse,
};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf_newton::{ScfResponses, UhfNewtonInputs};
use ndarray::Array2;
use std::time::Instant;

const AUX: &str = "def2-universal-jkfit";

fn frob(a: &Array2<f64>) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64())
}

/// The route's unfitted builders, built once (geometry-only setup excluded
/// from the product timings, as in the SCF where they are built before the loop).
struct Unfitted<'p> {
    cosx: CosxK<'p>,
    dk: DirectK<'p>,
    dj: DirectJ<'p>,
}

#[derive(Clone, Copy, Debug)]
enum Route {
    RiJCosx,
    DfJDirectK,
    DfKDirectJ,
}

#[ignore = "measurement harness: minutes at C8H18/def2-TZVP"]
#[test]
fn hvp_route_timing() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../");
    let xyz = std::env::var("FERRIC_HVP_XYZ")
        .unwrap_or_else(|_| format!("{root}testdata/molecules/alkane_8.xyz"));
    let basis_name = std::env::var("FERRIC_HVP_BASIS").unwrap_or_else(|_| "def2-tzvp".into());
    let text = std::fs::read_to_string(&xyz).unwrap();
    let mol = Molecule::parse_xyz(&text, 0, 1).unwrap();
    // The radical: drop the last atom (a terminal H for the n-alkanes).
    let lines: Vec<&str> = text.lines().collect();
    let nat: usize = lines[0].trim().parse().unwrap();
    let rad_text = format!(
        "{}\nradical\n{}\n",
        nat - 1,
        lines[2..2 + nat - 1].join("\n")
    );
    let rad = Molecule::parse_xyz(&rad_text, 0, 2).unwrap();
    eprintln!(
        "system {xyz} / {basis_name}; radical {} atoms",
        rad.atoms.len()
    );

    run_closed_shell(&mol, &basis_name);
    run_open_shell(&rad, &basis_name);
}

/// Warm the COSX builder the way the SCF already has by the time it is lent
/// (workers made, `S_num` factored), then report its steady per-build cost on
/// the SCF-style operand (`build_from_occ(C_occ)`) next to one response operand
/// (`δD = C_v κ C_oᵀ + h.c.`, normalized to `max|δD| = 1` as
/// `unfitted_response` does).
fn warm_cosx(cosx: &mut CosxK<'_>, c: &Array2<f64>, nocc: usize, kappa: &Array2<f64>) {
    use ndarray::s;
    let n = c.nrows();
    let c_o = c.slice(s![.., ..nocc]).to_owned();
    let c_v = c.slice(s![.., nocc..]).to_owned();
    let mut kb = Array2::<f64>::zeros((n, n));
    let (_, t_first) = timed(|| cosx.build_from_occ(&c_o, &mut kb).unwrap());
    let (_, t_occ) = timed(|| cosx.build_from_occ(&c_o, &mut kb).unwrap());
    let t_occ_d = *cosx.last_timings();
    let x = c_v.dot(kappa).dot(&c_o.t());
    let mut dd = &x + &x.t();
    let m = dd.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    dd /= m;
    let (_, t_dd) = timed(|| cosx.build(&dd, &mut kb).unwrap());
    let t_dd_d = *cosx.last_timings();
    eprintln!(
        "COSX per build: first (setup) {t_first:.2} s; SCF-style build_from_occ {t_occ:.2} s \
         (pairs {} lambda {:.2} a_build {:.1} s); response build(dD) {t_dd:.2} s (pairs {} lambda {:.2} \
         a_build {:.1} s)",
        t_occ_d.pairs_kept, t_occ_d.lambda_frac, t_occ_d.a_build_s,
        t_dd_d.pairs_kept, t_dd_d.lambda_frac, t_dd_d.a_build_s
    );
}

fn run_closed_shell(mol: &Molecule, basis_name: &str) {
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let prep = PreparedBasis::new(mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let aux = PreparedBasis::new(mol, &basis::bundled(AUX).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let thresh = RhfConfig::default().integral_thresh;
    let n = prep.nbasis();
    let no = mol.nelec() as usize / 2;
    eprintln!("RHF: nbf={n} naux={} nocc={no}", aux.nbasis());

    let cfg = RhfConfig {
        energy_conv: 1e-7,
        density_conv: 1e-5,
        max_iter: 100,
        df_j_aux: Some(AUX.into()),
        df_k_aux: Some(AUX.into()),
        ri_j_precision: Some(RiJPrecision::F64),
        ..Default::default()
    };
    let (res, t_scf) =
        timed(|| ferric_scf::rhf::solve_rhf(&ctx, mol, &prep, op, &bounds, &cfg).unwrap());
    eprintln!("RHF reference SCF: {} it, {t_scf:.1} s", res.iterations);
    let c = res.mos_alpha.clone();
    let f_mo = c.t().dot(&res.fock_alpha).dot(&c);
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let k = Array2::from_shape_fn((n - no, no), |_| 0.01 * rng.next());
    let pool = EnginePool::new(op, &prep, ferric_integrals::engine_pool::eri_precision()).unwrap();
    let mut dfj = DfJ::new_with_precision(op, &prep, &aux, budget, RiJPrecision::F64).unwrap();
    let mut dfk = DfK::new(op, &prep, &aux, budget).unwrap();
    let (mut un, t_setup) = timed(|| Unfitted {
        cosx: CosxK::new(&ctx, mol, &prep, CosxConfig::default(), budget).unwrap(),
        dk: DirectK::new(&ctx, &prep, &bounds, thresh, budget),
        dj: DirectJ::new(&ctx, &prep, &bounds, thresh, budget),
    });
    eprintln!(
        "unfitted builders setup {t_setup:.1} s (COSX npts {})",
        un.cosx.npts()
    );
    warm_cosx(&mut un.cosx, &c, no, &k);

    for &k_mix in &[1.0, 0.0] {
        let inp = RhfNewtonInputs {
            prep: &prep,
            bounds: &bounds,
            c: &c,
            f_mo: &f_mo,
            nocc: no,
            k_mix_sr: k_mix,
            rsh: None,
            fxc: None,
            thresh,
            ooc_budget: budget,
        };
        // Fitted route, TRAH and stability: before = direct, after = RI-J + DF-K.
        let (h_dir, t_dir) = timed(|| hessian_matvec_jk(&ctx, &inp, &k, &pool, None).unwrap());
        let (h_fit, t_fit) = timed(|| {
            let r = ScfJkResponse::new(Some(&mut dfj), (k_mix != 0.0).then_some(&mut dfk));
            hessian_matvec_jk(&ctx, &inp, &k, &pool, Some(&r)).unwrap()
        });
        eprintln!(
            "RHF singlet k_mix={k_mix} fitted (TRAH/stability): before {t_dir:.2} s, after {t_fit:.2} s, \
             x{:.1}; rel diff {:.2e}",
            t_dir / t_fit,
            frob(&(&h_fit - &h_dir)) / frob(&h_dir)
        );
        if k_mix != 0.0 {
            let (_, t_dir) =
                timed(|| triplet_hessian_matvec_jk(&ctx, &inp, &k, &pool, None).unwrap());
            let (_, t_fit) = timed(|| {
                let r = ScfJkResponse::new(Some(&mut dfj), Some(&mut dfk));
                triplet_hessian_matvec_jk(&ctx, &inp, &k, &pool, Some(&r)).unwrap()
            });
            eprintln!(
                "RHF triplet fitted (external stability): before {t_dir:.2} s, after {t_fit:.2} s, x{:.1}",
                t_dir / t_fit
            );
        }
        for route in [Route::RiJCosx, Route::DfJDirectK, Route::DfKDirectJ] {
            let (h_b, t_b) = timed(|| {
                let r = match route {
                    Route::DfKDirectJ => ScfJkResponse::new::<DfJ>(None, Some(&mut dfk)),
                    _ => ScfJkResponse::new(Some(&mut dfj), None),
                };
                hessian_matvec_jk(&ctx, &inp, &k, &pool, Some(&r)).unwrap()
            });
            let (h_a, t_a) = timed(|| {
                let r = match route {
                    Route::RiJCosx => ScfJkResponse::new(Some(&mut dfj), None)
                        .with_unfitted_k(Some(&mut un.cosx as &mut dyn KBuilder)),
                    Route::DfJDirectK => ScfJkResponse::new(Some(&mut dfj), None)
                        .with_unfitted_k(Some(&mut un.dk as &mut dyn KBuilder)),
                    Route::DfKDirectJ => ScfJkResponse::new::<DfJ>(None, Some(&mut dfk))
                        .with_unfitted_j(Some(&mut un.dj as &mut dyn JBuilder)),
                };
                hessian_matvec_jk(&ctx, &inp, &k, &pool, Some(&r)).unwrap()
            });
            eprintln!(
                "RHF singlet k_mix={k_mix} {route:?} (Newton): before {t_b:.2} s, after {t_a:.2} s, \
                 x{:.1}; rel diff {:.2e}",
                t_b / t_a,
                frob(&(&h_a - &h_b)) / frob(&h_b)
            );
        }
    }
}

fn run_open_shell(mol: &Molecule, basis_name: &str) {
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let prep = PreparedBasis::new(mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let aux = PreparedBasis::new(mol, &basis::bundled(AUX).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let thresh = RhfConfig::default().integral_thresh;
    let n = prep.nbasis();
    let na = (mol.nelec() as usize).div_ceil(2);
    let nb = mol.nelec() as usize / 2;
    // Any orthonormal MOs serve for a timing: the core-Hamiltonian guess of
    // the radical (the product's cost does not depend on convergence).
    let h = oneelectron::hcore(&prep);
    let s = oneelectron::overlap(&prep);
    use ndarray_linalg::{Eigh, UPLO};
    let (sv, su) = s.eigh(UPLO::Lower).unwrap();
    let mut c = su.clone();
    for (j, &l) in sv.iter().enumerate() {
        c.column_mut(j).mapv_inplace(|x| x / l.sqrt());
    }
    let hx = c.t().dot(&h).dot(&c);
    let (_, v) = hx.eigh(UPLO::Lower).unwrap();
    let c = c.dot(&v);
    assert_eq!(c.ncols(), n, "timing harness assumes no linear dependence");
    let f_mo = c.t().dot(&h).dot(&c);
    let mut rng = Rng(0xD1B54A32D192ED03);
    let ka = Array2::from_shape_fn((n - na, na), |_| 0.01 * rng.next());
    let kb = Array2::from_shape_fn((n - nb, nb), |_| 0.01 * rng.next());
    let pool = EnginePool::new(op, &prep, ferric_integrals::engine_pool::eri_precision()).unwrap();
    let mut dfj = DfJ::new_with_precision(op, &prep, &aux, budget, RiJPrecision::F64).unwrap();
    let mut dfk = DfK::new(op, &prep, &aux, budget).unwrap();
    let mut un = Unfitted {
        cosx: CosxK::new(&ctx, mol, &prep, CosxConfig::default(), budget).unwrap(),
        dk: DirectK::new(&ctx, &prep, &bounds, thresh, budget),
        dj: DirectJ::new(&ctx, &prep, &bounds, thresh, budget),
    };
    eprintln!("UHF radical: nbf={n} nocc_a={na} nocc_b={nb}");
    warm_cosx(&mut un.cosx, &c, na, &ka);
    let base = UhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c_a: &c,
        c_b: &c,
        f_a_mo: &f_mo,
        f_b_mo: &f_mo,
        nocc_a: na,
        nocc_b: nb,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh,
        ooc_budget: budget,
        j_response: None,
        k_response: None,
    };
    let hvp = |inp: &UhfNewtonInputs| {
        timed(|| ferric_scf::uhf_newton::hessian_matvec(&ctx, inp, &ka, &kb, &pool).unwrap())
    };
    // Fitted route, stability: before = direct, after = RI-J + DF-K.
    let (_, t_dir) = hvp(&base);
    let (_, t_fit) = {
        let r = ScfResponses::lend(
            Some(&mut dfj as &mut dyn JBuilder),
            Some(&mut dfk),
            None,
            None,
        );
        hvp(&UhfNewtonInputs {
            j_response: r.j(),
            k_response: r.k(),
            ..base
        })
    };
    eprintln!(
        "UHF fitted (stability): before {t_dir:.2} s, after {t_fit:.2} s, x{:.1}",
        t_dir / t_fit
    );
    for route in [Route::RiJCosx, Route::DfJDirectK, Route::DfKDirectJ] {
        let (_, t_b) = {
            let r = match route {
                Route::DfKDirectJ => ScfResponses::lend(None, Some(&mut dfk), None, None),
                _ => ScfResponses::lend(Some(&mut dfj as &mut dyn JBuilder), None, None, None),
            };
            hvp(&UhfNewtonInputs {
                j_response: r.j(),
                k_response: r.k(),
                ..base
            })
        };
        let (_, t_a) = {
            let r = match route {
                Route::RiJCosx => ScfResponses::lend(
                    Some(&mut dfj as &mut dyn JBuilder),
                    None,
                    None,
                    Some(&mut un.cosx as &mut dyn KBuilder),
                ),
                Route::DfJDirectK => ScfResponses::lend(
                    Some(&mut dfj as &mut dyn JBuilder),
                    None,
                    None,
                    Some(&mut un.dk as &mut dyn KBuilder),
                ),
                Route::DfKDirectJ => ScfResponses::lend(
                    None,
                    Some(&mut dfk),
                    Some(&mut un.dj as &mut dyn JBuilder),
                    None,
                ),
            };
            hvp(&UhfNewtonInputs {
                j_response: r.j(),
                k_response: r.k(),
                ..base
            })
        };
        eprintln!(
            "UHF {route:?} (Newton/TRAH; the ROHF matvec makes the same J + 2 K builds): \
             before {t_b:.2} s, after {t_a:.2} s, x{:.1}",
            t_b / t_a
        );
    }
}
