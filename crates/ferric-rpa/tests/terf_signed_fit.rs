//! The terf long-range interaction of rs-mp2-rpa against independent fits.
//!
//! # What is claimed
//!
//! The terf kernel is not positive definite (its Fourier transform carries
//! `cos(k r0)`), so its auxiliary metric `V_t = (P|terf|Q)` has negative
//! eigenvalues. rs-mp2-rpa builds the long-range interaction as Coulomb RI
//! minus terfc RI, `K_lr = B_Cᵀ B_C − B_sᵀ B_s`, both fits in a positive
//! definite metric, and never factors `V_t`. On Ne2 (R = 3.6 Å, aug-cc-pVDZ /
//! aug-cc-pVDZ-RIFIT, r0 = 1.35 Å, r0·ω ∈ {2, 4}) its `e_dmp2_lr` and
//! `e_drpa_lr` equal those of three constructions this file builds itself
//! from the raw integrals `J_x = (P|x|ia)`, `V_x = (P|x|Q)`:
//!
//! * `c_minus_sr`: `J_Cᵀ V_C⁻¹ J_C − J_sᵀ V_s⁻¹ J_s` (triangular solves);
//! * `signed_pinv`: `J_tᵀ V_t⁺ J_t`, pseudo-inverse over `|λ| > 1e-10`, both signs;
//! * `sandwich`: `(V_C⁻¹ J_C)ᵀ V_t (V_C⁻¹ J_C)`, Coulomb-fitted densities.
//!
//! dMP2 is `−2 Σ K²/(D_ia + D_jb)` and the dRPA energy the plasmon formula
//! (no frequency quadrature). The fourth construction, `drop`, is the
//! positive-subspace fit `J_tᵀ V_t,+⁺ J_t` the eigh branch of
//! `metric_inverse_sqrt` used to give terf; the test requires it to MISS the
//! bar, so the pass condition is shown reachable and discriminating.
//!
//! # Bars (derived, not copied)
//!
//! * Identity bar (ferric vs `c_minus_sr`, same algebra): `10 κ ε |E|`
//!   with κ the larger condition number of `V_C`, `V_s` and ε the machine
//!   epsilon, plus the quadrature bar for dRPA terms.
//! * Quadrature bar: 48-point Gauss–Legendre (`EXACT_DRPA_QUAD_POINTS`, the
//!   order at which `tests/validation_drpa_frequency.rs` reaches the rounding
//!   floor against the plasmon formula). Set to 1e-8 Ha and shown REACHED by
//!   the same check on the Coulomb dRPA of the same system.
//! * Fit bar (ferric vs `signed_pinv`, `sandwich` on the counterpoise
//!   interaction of `E_dRPA − dMP2`): 0.5 µHa. `scripts/gen_terf_signed_fit_refs.py`
//!   (output `testdata/reference/ne2_terf_signed_fit.json`) measures the
//!   construction spread at ≤ 0.20 µHa with aug-cc-pVDZ-RIFIT and moves
//!   `c_minus_sr` by ≤ 0.001 µHa to aug-cc-pVTZ-RIFIT; the `drop` construction
//!   misses by 27 µHa (r0·ω = 2) and 88 µHa (r0·ω = 4).
//!
//! # Mutations (each makes `terf_lr_matches_independent_constructions_ne2` fail by assertion)
//!
//! * Fit terf in its own metric again (`rs_mp2_rpa.rs` builds the LR
//!   intermediates with `compute_rpa_intermediates(op_lr)` and
//!   `metric_inverse_sqrt` sends `Terf` to `eigh_inverse_sqrt`, which drops
//!   `evals[k] < LINDEP_THRESH`): the dimer `e_dmp2_lr` identity assertion
//!   fails at r0·ω = 2 (−1013.0 vs −4352.7 µHa; run on 0fe427de1, which is
//!   this state).
//! * Flip the signature to `+` (`K = B_Cᵀ B_C + B_sᵀ B_s`) in
//!   `ferric_mp2::rimp2::signed_pair_block`: the same assertion fails
//!   (e_dmp2_lr −2.5471 vs −0.0043527 Ha).
//! * Drop the row negation in `ferric_rpa::energy::eval_signed_trace_log_summands`
//!   (signed dRPA with signature +): the dimer `e_drpa_lr` assertion fails
//!   (−1.6842 vs −0.0044846 Ha); the energy.rs plasmon unit tests fail too.
//!
//! Needs the terf interpolation tables (`FERRIC_TERF_TABLE_DIR`); skips with
//! a note without them.
//!
//! `paths_without_a_terf_fit_are_unchanged` pins the erf (B and T), terfc
//! RI-MP2 and terf-T numbers that do not involve a terf fit; the values were
//! recorded on the commit before the signed fit landed and checked to the
//! last bit there and after (see the commit report). The committed bar is
//! 1e-12 Ha so that a BLAS build that rounds differently does not fail it.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ferric_rpa::rs_mp2_rpa::{
    rs_mp2_lr_rpa, rs_mp2_lr_rpa_scan, Attenuator, RsMp2RpaConfig, RsMp2RpaFormulation,
};
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, Eigh, SolveTriangular, UPLO};

const A2B: f64 = 1.8897261246257702;
const R0_ANG: f64 = 1.35;
const CS: [f64; 2] = [2.0, 4.0];
const QUAD_BAR: f64 = 1e-8;
const FIT_BAR: f64 = 0.5e-6;

fn tables() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR")
        .map(|d| std::path::Path::new(&d).join("16_4_2.bin").exists())
        .unwrap_or(false)
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn setup(xyz: &str, obs: &str, aux: &str) -> Sys {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig {
            max_iter: 400,
            energy_conv: 1e-11,
            density_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(rhf.converged);
    Sys {
        mol,
        obs,
        dfbs,
        rhf,
    }
}

/// Raw `(P|op|ia)` as a `(naux, nocc·nvir)` matrix and the metric `(P|op|Q)`.
fn raw(sys: &Sys, op: Operator) -> (Array2<f64>, Array2<f64>) {
    let nocc = (sys.mol.nelec() as usize) / 2;
    let c = sys.rhf.mos_r();
    let co = c.slice(ndarray::s![.., ..nocc]).to_owned();
    let cv = c.slice(ndarray::s![.., nocc..]).to_owned();
    let j =
        ferric_mp2::rimp2::eri3_mo_ov_blocked(op, &sys.obs, &sys.dfbs, &co, &cv, 1 << 30).unwrap();
    let naux = j.shape()[0];
    let nov = j.shape()[1] * j.shape()[2];
    let j = j.into_shape_with_order((naux, nov)).unwrap();
    (j, coulomb_metric_2c(op, &sys.dfbs).unwrap())
}

fn gaps(sys: &Sys) -> Array1<f64> {
    let nocc = (sys.mol.nelec() as usize) / 2;
    let e = sys.rhf.eps_r();
    let nvir = e.len() - nocc;
    Array1::from_shape_fn(nocc * nvir, |ia| e[nocc + ia % nvir] - e[ia / nvir])
}

/// `L⁻¹ J` with `V = L Lᵀ` (LAPACK triangular solve), so `K = XᵀX = Jᵀ V⁻¹ J`.
fn chol_fit(j: &Array2<f64>, v: &Array2<f64>) -> Array2<f64> {
    let l = v.cholesky(UPLO::Lower).unwrap();
    l.solve_triangular(UPLO::Lower, ndarray_linalg::Diag::NonUnit, j)
        .unwrap()
}

fn cond(v: &Array2<f64>) -> f64 {
    let (w, _) = v.eigh(UPLO::Lower).unwrap();
    w[w.len() - 1] / w[0]
}

/// `U f(λ) Uᵀ` over the kept eigenvalues of `v`.
fn pinv(v: &Array2<f64>, signed: bool) -> (Array2<f64>, usize) {
    let (w, u) = v.eigh(UPLO::Lower).unwrap();
    let mut us = u.clone();
    let mut n_neg = 0;
    for k in 0..w.len() {
        if w[k] < -1e-10 {
            n_neg += 1;
        }
        let keep = if signed {
            w[k].abs() > 1e-10
        } else {
            w[k] > 1e-10
        };
        let s = if keep { 1.0 / w[k] } else { 0.0 };
        us.column_mut(k).mapv_inplace(|x| x * s);
    }
    (us.dot(&u.t()), n_neg)
}

fn dmp2(k: &Array2<f64>, d: &Array1<f64>) -> f64 {
    let n = d.len();
    let mut e = 0.0;
    for p in 0..n {
        for q in 0..n {
            e += k[(p, q)] * k[(p, q)] / (d[p] + d[q]);
        }
    }
    -2.0 * e
}

/// Closed-shell dRPA by the plasmon formula, `½(ΣΩ − ΣD − 2 tr K)`,
/// `Ω²` the eigenvalues of `D^½ (D + 4K) D^½`.
fn plasmon(k: &Array2<f64>, d: &Array1<f64>) -> f64 {
    let n = d.len();
    let sd = d.mapv(f64::sqrt);
    let m = Array2::from_shape_fn((n, n), |(p, q)| {
        let a = if p == q { d[p] } else { 0.0 } + 4.0 * k[(p, q)];
        sd[p] * a * sd[q]
    });
    let (w, _) = m.eigh(UPLO::Lower).unwrap();
    assert!(w[0] > 0.0, "unstable dRPA: {}", w[0]);
    0.5 * (w.iter().map(|x| x.sqrt()).sum::<f64>()
        - d.sum()
        - 2.0 * (0..n).map(|p| k[(p, p)]).sum::<f64>())
}

fn drpa_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        trunc_thresh: 0.0,
        quadrature: ferric_rpa::config::QuadratureConfig {
            scheme: ferric_rpa::config::QuadratureScheme::GaussLegendre,
            n_points: ferric_rpa::energy_route::EXACT_DRPA_QUAD_POINTS,
            u0: 0.5,
        },
        ..Default::default()
    }
}

/// Per r0·ω: (dmp2, drpa) for ferric and the four constructions.
struct Point {
    ferric: (f64, f64),
    c_minus_sr: (f64, f64),
    signed_pinv: (f64, f64),
    sandwich: (f64, f64),
    drop: (f64, f64),
    identity_bar: f64,
    n_neg: usize,
}

fn evaluate(sys: &Sys) -> Vec<Point> {
    let r0 = R0_ANG * A2B;
    let d = gaps(sys);
    let (jc, vc) = raw(sys, Operator::coulomb());
    let xc = chol_fit(&jc, &vc);
    let kc = xc.t().dot(&xc);
    // Coulomb-fitted densities V_C⁻¹ J_C = L⁻ᵀ L⁻¹ J_C.
    let lc = vc.cholesky(UPLO::Lower).unwrap();
    let fitted = lc
        .t()
        .to_owned()
        .solve_triangular(UPLO::Upper, ndarray_linalg::Diag::NonUnit, &xc)
        .unwrap();

    // The quadrature bar is reached on the Coulomb dRPA of the same system.
    let e_c = ferric_rpa::energy_route::run_pdep_rpa_energy(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        &sys.rhf,
        &PdepRpaConfig {
            need_eigenvalues_freq: false,
            ..drpa_cfg()
        },
    )
    .unwrap()
    .e_rpa;
    let p_c = plasmon(&kc, &d);
    eprintln!(
        "Coulomb dRPA: ferric {e_c:.12} plasmon {p_c:.12} diff {:.2e}",
        e_c - p_c
    );
    assert!(
        (e_c - p_c).abs() < QUAD_BAR,
        "quadrature bar not reached on the Coulomb dRPA: {e_c} vs {p_c}"
    );

    let omegas: Vec<f64> = CS.iter().map(|c| c / r0).collect();
    let cfg = RsMp2RpaConfig {
        attenuator: Attenuator::Terf,
        r0,
        drpa: drpa_cfg(),
        ..Default::default()
    };
    let res = rs_mp2_lr_rpa_scan(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        &sys.rhf,
        &cfg,
        &omegas,
        &[RsMp2RpaFormulation::DeltaLr],
    )
    .unwrap();

    omegas
        .iter()
        .zip(res)
        .map(|(&w, r)| {
            let lr = Operator::terf_with_omega(r0, w);
            let sr = Operator::terfc_with_omega(r0, w);
            let (jt, vt) = raw(sys, lr);
            let (js, vs) = raw(sys, sr);
            let xs = chol_fit(&js, &vs);
            let k3 = &kc - &xs.t().dot(&xs);
            let (ps, n_neg) = pinv(&vt, true);
            let k1 = jt.t().dot(&ps.dot(&jt));
            let k2 = fitted.t().dot(&vt.dot(&fitted));
            let (pd, _) = pinv(&vt, false);
            let k4 = jt.t().dot(&pd.dot(&jt));
            let both = |k: &Array2<f64>| (dmp2(k, &d), plasmon(k, &d));
            let kappa = cond(&vc).max(cond(&vs));
            let c3 = both(&k3);
            Point {
                ferric: (r.e_dmp2_lr, r.e_drpa_lr.unwrap()),
                c_minus_sr: c3,
                signed_pinv: both(&k1),
                sandwich: both(&k2),
                drop: both(&k4),
                identity_bar: 10.0 * kappa * f64::EPSILON * c3.0.abs(),
                n_neg,
            }
        })
        .collect()
}

#[test]
fn terf_lr_matches_independent_constructions_ne2() {
    if !tables() {
        eprintln!("SKIP terf_lr_matches_independent_constructions_ne2: terf tables absent");
        return;
    }
    let dim = setup(
        "2\nNe2\nNe 0 0 0\nNe 0 0 3.6\n",
        "aug-cc-pvdz",
        "aug-cc-pvdz-rifit",
    );
    let mon = setup(
        "2\nNe2\nNe 0 0 0\n@Ne 0 0 3.6\n",
        "aug-cc-pvdz",
        "aug-cc-pvdz-rifit",
    );
    let pd = evaluate(&dim);
    let pm = evaluate(&mon);
    for (k, (d, m)) in pd.iter().zip(&pm).enumerate() {
        let c = CS[k];
        assert!(
            d.n_neg > 0,
            "r0w={c}: the terf metric must be indefinite here"
        );
        // Identity: ferric is the c_minus_sr algebra, on the dimer and the monomer.
        for (who, p) in [("dimer", d), ("monomer", m)] {
            let e_mp2 = (p.ferric.0 - p.c_minus_sr.0).abs();
            let e_rpa = (p.ferric.1 - p.c_minus_sr.1).abs();
            eprintln!(
                "r0w={c} {who}: dMP2 ferric {:.10} c-sr {:.10} signed {:.10} sandwich {:.10} drop {:.10} | |d dMP2| {e_mp2:.2e} (bar {:.2e}) |d dRPA| {e_rpa:.2e} | n_neg {}",
                p.ferric.0, p.c_minus_sr.0, p.signed_pinv.0, p.sandwich.0, p.drop.0, p.identity_bar, p.n_neg
            );
            assert!(
                e_mp2 <= p.identity_bar,
                "r0w={c} {who}: e_dmp2_lr {} is not Coulomb RI minus terfc RI {} (bar {:.2e})",
                p.ferric.0,
                p.c_minus_sr.0,
                p.identity_bar
            );
            assert!(
                e_rpa <= p.identity_bar + QUAD_BAR,
                "r0w={c} {who}: e_drpa_lr {} is not the plasmon dRPA of K_C - K_sr {}",
                p.ferric.1,
                p.c_minus_sr.1
            );
        }
        // Counterpoise interaction of E_dRPA - dMP2 per construction.
        let e_int = |f: &dyn Fn(&Point) -> (f64, f64)| {
            let (a, b) = (f(d), f(m));
            (a.1 - a.0) - 2.0 * (b.1 - b.0)
        };
        let fe = e_int(&|p| p.ferric);
        let rows = [
            ("c_minus_sr", e_int(&|p| p.c_minus_sr)),
            ("signed_pinv", e_int(&|p| p.signed_pinv)),
            ("sandwich", e_int(&|p| p.sandwich)),
        ];
        let dr = e_int(&|p| p.drop);
        eprintln!(
            "r0w={c}: E_int[dRPA - dMP2](lr) uHa: ferric {:.4} c_minus_sr {:.4} signed_pinv {:.4} sandwich {:.4} drop {:.4}",
            fe * 1e6,
            rows[0].1 * 1e6,
            rows[1].1 * 1e6,
            rows[2].1 * 1e6,
            dr * 1e6
        );
        for (name, e) in rows {
            assert!(
                (fe - e).abs() <= FIT_BAR,
                "r0w={c}: ferric E_int {:.4} uHa vs {name} {:.4} uHa (bar {} uHa)",
                fe * 1e6,
                e * 1e6,
                FIT_BAR * 1e6
            );
        }
        // Reachability: the positive-subspace fit misses the bar.
        assert!(
            (dr - rows[0].1).abs() > FIT_BAR,
            "r0w={c}: the drop construction ({:.4} uHa) passes the bar; the test cannot discriminate",
            dr * 1e6
        );
    }
}

/// Prints the value and its bits, and records a miss (asserted at the end so
/// every value is printed).
fn pin(misses: &mut Vec<String>, label: &str, got: f64, want: f64) {
    eprintln!("PIN {label} = {got:.17e} bits=0x{:016x}", got.to_bits());
    if (got - want).abs() > 1e-12 {
        misses.push(format!(
            "{label}: {got:.17e} moved from the pinned {want:.17e}"
        ));
    }
}

#[test]
fn paths_without_a_terf_fit_are_unchanged() {
    let water = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
    let sys = setup(water, "cc-pvdz", "cc-pvdz-ri");
    let both = [
        RsMp2RpaFormulation::DeltaLr,
        RsMp2RpaFormulation::CoupledRings,
    ];
    let erf = rs_mp2_lr_rpa_scan(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        &sys.rhf,
        &RsMp2RpaConfig::default(),
        &[0.222, 0.5],
        &both,
    )
    .unwrap();
    let mut misses = Vec::new();
    for (k, r) in erf.iter().enumerate() {
        pin(
            &mut misses,
            &format!("erf[{k}].e_corr"),
            r.e_corr,
            PIN_ERF[k],
        );
    }
    pin(
        &mut misses,
        "erf[0].e_lr_mp2",
        erf[0].e_lr_mp2,
        PIN_ERF_LR_MP2,
    );
    if !tables() {
        eprintln!("SKIP terfc / terf-T pins: terf tables absent");
        assert!(misses.is_empty(), "{misses:#?}");
        return;
    }
    let (sc, _) = ferric_mp2::rimp2::ri_mp2_spin_components(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::terfc(2.0),
        &sys.rhf,
        &ferric_mp2::rimp2::RiMp2Config::default(),
    )
    .unwrap();
    pin(
        &mut misses,
        "terfc RI-MP2 e_total",
        sc.e_total,
        PIN_TERFC_MP2,
    );
    let t = rs_mp2_lr_rpa(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        &sys.rhf,
        &RsMp2RpaConfig {
            attenuator: Attenuator::Terf,
            formulation: RsMp2RpaFormulation::CoupledRings,
            ..Default::default()
        },
    )
    .unwrap();
    pin(&mut misses, "terf T e_corr", t.e_corr, PIN_TERF_T[0]);
    pin(&mut misses, "terf T e_sr_mp2", t.e_sr_mp2, PIN_TERF_T[1]);
    let full = t.e_delta_drpa_full.unwrap();
    pin(&mut misses, "terf T e_delta_drpa_full", full, PIN_TERF_T[2]);
    let sr = t.e_delta_drpa_sr.unwrap();
    pin(&mut misses, "terf T e_delta_drpa_sr", sr, PIN_TERF_T[3]);
    assert!(misses.is_empty(), "{misses:#?}");
}

// Recorded on 22770952b (before the signed fit), OPENBLAS_NUM_THREADS=1.
// Recorded on 0fe427de1 (before the signed fit), as bit patterns.
const PIN_ERF: [f64; 4] = [
    f64::from_bits(0xbfca1db018451b41), // -2.04031001902160752e-1
    f64::from_bits(0xbfc9d746f4871f86), // -2.01882237814285215e-1
    f64::from_bits(0xbfca0ece275f8b86), // -2.03576821541983033e-1
    f64::from_bits(0xbfc7865e0fc3bb29), // -1.83788068482919870e-1
];
const PIN_ERF_LR_MP2: f64 = f64::from_bits(0xbf2332b60d0a4bd5); // -1.46469805731712588e-4
const PIN_TERFC_MP2: f64 = f64::from_bits(0xbfca07d3bd890585); // -2.03363864466194283e-1
const PIN_TERF_T: [f64; 4] = [
    f64::from_bits(0xbfca2057c7b54dfe), // -2.04112026704578342e-1
    f64::from_bits(0xbfca1f26936e14af), // -2.04075643527322964e-1
    f64::from_bits(0x3fb2db37151e6678), // 7.36574579772425464e-2
    f64::from_bits(0x3fb2e05d43314824), // 7.37360276264387715e-2
];
