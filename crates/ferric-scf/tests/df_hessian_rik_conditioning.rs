//! Accuracy of the analytic RI-K Hessian as a function of the conditioning of
//! the exchange metric, and the routing gate derived from it.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test df_hessian_rik_conditioning
//! ```
//!
//! The measurement harness (`measure_one_system`, ignored, driven by
//! environment variables) writes, for one molecule / basis / fitting basis,
//! the analytic RI-JK Hessian, the same Hessian with the metric inverse built
//! by LU instead of `V^{-1/2} V^{-1/2}` (an independent construction of `M`:
//! their difference is roundoff amplified by the conditioning, with no finite
//! difference involved), and central differences of ferric's own analytic DF
//! gradient at several steps (SCF re-converged at every displaced geometry).
//! The analysis and the derived bound live in `hessian_df_k`'s module doc,
//! section "Conditioning".

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::hessian::rhf_hessian_parts;
use ferric_scf::hessian::rij_support::RikMutants;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};

fn has_deriv2() -> bool {
    ferric_integrals::engine::libint_max_deriv_order() >= 2
}

/// RHF with RI-J and RI-K over the same fitting basis (the route a hybrid
/// Kohn-Sham SCF takes by default), converged far below the FD noise.
fn scf_config(aux: &str) -> RhfConfig {
    RhfConfig::default()
        .with_ri_j_precision(ferric_scf::df_j::RiJPrecision::F64)
        .with_max_iter(300)
        .with_energy_conv(1e-14)
        .with_density_conv(1e-12)
        .with_df_j_aux(aux.to_string())
        .with_df_k_aux(aux.to_string())
}

struct Setup {
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    rhf: ScfResult,
    cfg: RhfConfig,
}

fn setup(mol: &Molecule, basis_name: &str, aux: &str) -> Setup {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = scf_config(aux);
    let rhf = solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged, "{basis_name}/{aux}: SCF did not converge");
    Setup {
        prep,
        bounds,
        rhf,
        cfg,
    }
}

fn displaced(mol: &Molecule, atom: usize, coord: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match coord {
        0 => m.atoms[atom].x += h,
        1 => m.atoms[atom].y += h,
        _ => m.atoms[atom].zpos += h,
    }
    m
}

/// Unsymmetrized central difference of the analytic DF gradient.
fn fd_hessian(mol: &Molecule, basis_name: &str, aux: &str, h: f64) -> Array2<f64> {
    let n3 = 3 * mol.atoms.len();
    let grad = |m: &Molecule| {
        let s = setup(m, basis_name, aux);
        rhf_gradient(m, &s.prep, Operator::coulomb(), &s.bounds, &s.rhf, None).unwrap()
    };
    let mut out = Array2::<f64>::zeros((n3, n3));
    for b in 0..n3 {
        let gp = grad(&displaced(mol, b / 3, b % 3, h));
        let gm = grad(&displaced(mol, b / 3, b % 3, -h));
        for a in 0..n3 {
            out[(a, b)] = (gp[(a / 3, a % 3)] - gm[(a / 3, a % 3)]) / (2.0 * h);
        }
    }
    out
}

fn rows(h: &Array2<f64>) -> Vec<Vec<f64>> {
    h.outer_iter().map(|r| r.to_vec()).collect()
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// One measurement row, written as JSON to `$RIKCOND_OUT`.
///
/// `RIKCOND_MOL` (xyz path), `RIKCOND_BASIS`, `RIKCOND_AUX`,
/// `RIKCOND_STEPS` (comma-separated Bohr; empty = analytic only).
#[test]
#[ignore = "measurement harness: one molecule per run, driven by RIKCOND_* environment variables"]
fn measure_one_system() {
    if !has_deriv2() {
        return;
    }
    let path = std::env::var("RIKCOND_MOL").expect("RIKCOND_MOL");
    let basis_name = env_or("RIKCOND_BASIS", "def2-svp");
    let aux = env_or("RIKCOND_AUX", "def2-universal-jkfit");
    let steps: Vec<f64> = env_or("RIKCOND_STEPS", "")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let out = std::env::var("RIKCOND_OUT").expect("RIKCOND_OUT");
    let mol = Molecule::load_xyz(&path).unwrap();
    let auxbs = PreparedBasis::new(&mol, &basis::bundled(&aux).unwrap()).unwrap();
    let v = coulomb_metric_2c(Operator::coulomb(), &auxbs).unwrap();
    let (lam, _) = v.eigh(UPLO::Upper).unwrap();
    let (lmin, lmax) = (lam[0], lam[lam.len() - 1]);
    let band = lam.iter().filter(|&&l| (1e-10..1e-7).contains(&l)).count();
    let below = lam.iter().filter(|&&l| l < 1e-10).count();
    eprintln!(
        "RIKCOND {path} {basis_name}/{aux}: naux {} lambda_min {lmin:.3e} lambda_max {lmax:.3e} \
         cond {:.3e} band {band} dropped {below}",
        lam.len(),
        lmax / lmin
    );
    let t0 = std::time::Instant::now();
    let s = setup(&mol, &basis_name, &aux);
    let t_scf = t0.elapsed().as_secs_f64();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let t1 = std::time::Instant::now();
    let parts = rhf_hessian_parts(&ctx, &mol, &s.prep, op, &s.bounds, &s.rhf, &s.cfg).unwrap();
    let t_hess = t1.elapsed().as_secs_f64();
    let analytic = parts.total();
    let lu = ferric_scf::hessian::rij_support::rhf_hessian_parts_rik_mutated(
        &ctx,
        &mol,
        &s.prep,
        op,
        &s.bounds,
        &s.rhf,
        &s.cfg,
        &RikMutants {
            explicit_inverse: true,
            allow_ill_conditioned: true,
            ..Default::default()
        },
    )
    .unwrap()
    .total();
    let no_metric_second = ferric_scf::hessian::rij_support::rhf_hessian_parts_rik_mutated(
        &ctx,
        &mol,
        &s.prep,
        op,
        &s.bounds,
        &s.rhf,
        &s.cfg,
        &RikMutants {
            drop_metric_second: true,
            allow_ill_conditioned: true,
            ..Default::default()
        },
    )
    .unwrap()
    .total();
    let mut fds = serde_json::Map::new();
    for &h in &steps {
        let t = std::time::Instant::now();
        let fd = fd_hessian(&mol, &basis_name, &aux, h);
        eprintln!("RIKCOND step {h:e}: {:.1} s", t.elapsed().as_secs_f64());
        fds.insert(format!("{h:e}"), serde_json::json!(rows(&fd)));
    }
    let payload = serde_json::json!({
        "molecule": path,
        "basis": basis_name,
        "aux": aux,
        "nbf": s.prep.nbasis(),
        "naux": lam.len(),
        "lambda_min": lmin,
        "lambda_max": lmax,
        "cond": lmax / lmin,
        "band": band,
        "dropped": below,
        "energy": s.rhf.energy,
        "t_scf_s": t_scf,
        "t_hessian_s": t_hess,
        "cphf_max_residual": parts.cphf_max_residual,
        "response_asymmetry": parts.response_asymmetry,
        "analytic": rows(&analytic),
        "analytic_lu_inverse": rows(&lu),
        "analytic_no_metric_second": rows(&no_metric_second),
        "fd": fds,
    });
    std::fs::write(&out, serde_json::to_string(&payload).unwrap()).unwrap();
}

// ---------------------------------------------------------------------------
// The routing gate
// ---------------------------------------------------------------------------

/// An ascending synthetic spectrum with extreme eigenvalues `lambda_min` and
/// `lambda_max`.
fn spectrum(lambda_min: f64, lambda_max: f64) -> ndarray::Array1<f64> {
    ndarray::Array1::from(vec![lambda_min, 1e-3, 1.0, lambda_max])
}

#[test]
fn the_gate_is_where_the_bound_reaches_the_finite_difference_error() {
    use ferric_scf::df_gradient::DFK_GRADIENT_NOISE_FLOOR;
    use ferric_scf::hessian::rij_support::{
        auto_conditioning_gate, exchange_spectrum_conditioning,
    };
    use ferric_scf::hessian_df_k::{RikConditioning, FD_ROUTE_ERROR, RIK_ROUNDOFF_COEFF};
    // The gate is derived, not set: the lambda_min at which the roundoff bound
    // equals the error of the finite-difference route Auto falls back to.
    let gate = (RIK_ROUNDOFF_COEFF / FD_ROUTE_ERROR).sqrt();
    let at = RikConditioning::from_spectrum(gate, 1e3);
    assert!((at.bound - FD_ROUTE_ERROR).abs() <= 1e-12 * FD_ROUTE_ERROR);
    assert!(RikConditioning::from_spectrum(1.01 * gate, 1e3).admits_analytic());
    let below = RikConditioning::from_spectrum(0.99 * gate, 1e3);
    let e = auto_conditioning_gate(below).unwrap_err().to_string();
    assert!(e.contains("ABOVE"), "{e}");
    // cond is recorded but does not decide: the same lambda_min with a 1000x
    // larger lambda_max has the same bound.
    let wide = RikConditioning::from_spectrum(1.01 * gate, 1e6);
    assert_eq!(
        wide.bound,
        RikConditioning::from_spectrum(1.01 * gate, 1e3).bound
    );
    assert!(wide.admits_analytic());
    // With the measured constants the gate (1.2e-7) lies just above the
    // gradient's noise floor (1e-7): a spectrum between the two passes the hard
    // checks and is refused by Auto alone; one above the gate is admitted.
    assert!(
        gate > DFK_GRADIENT_NOISE_FLOOR && gate < 2.0 * DFK_GRADIENT_NOISE_FLOOR,
        "gate {gate:.3e}"
    );
    let between =
        exchange_spectrum_conditioning(&spectrum(0.5 * (gate + DFK_GRADIENT_NOISE_FLOOR), 1e4))
            .expect("between the floor and the gate: the hard checks pass");
    let e = auto_conditioning_gate(between).unwrap_err().to_string();
    assert!(e.contains("ABOVE"), "{e}");
    let over = exchange_spectrum_conditioning(&spectrum(1.05 * gate, 1e4)).unwrap();
    auto_conditioning_gate(over).expect("above the gate: analytic");
    // The hard checks are unchanged: a kept mode in the gradient's noise band
    // is refused whatever the bound.
    let e = exchange_spectrum_conditioning(&spectrum(0.5 * DFK_GRADIENT_NOISE_FLOOR, 1.0))
        .unwrap_err()
        .to_string();
    assert!(e.contains("lie in"), "{e}");
}

/// cond(V) of def2-universal-jkfit on the testdata propane and benzene
/// (measured: 1.681e8 and 7.146e8; the metric depends on the geometry and the
/// fitting basis only). Both were refused by the earlier fixed cond gate
/// (1e8); the probe admits them and records the bound.
#[test]
fn propane_and_benzene_jkfit_metrics_pass_the_auto_probe() {
    use ferric_scf::hessian::rij_support::probe_exchange_metric;
    use ferric_scf::hessian_df_k::{FD_ROUTE_ERROR, RIK_ROUNDOFF_COEFF};
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/molecules/");
    for (name, want) in [("alkane_3.xyz", 1.681e8), ("benzene.xyz", 7.146e8)] {
        let mol = Molecule::load_xyz(&format!("{dir}{name}")).unwrap();
        let aux =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let c = probe_exchange_metric(&aux, Operator::coulomb())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        eprintln!("{name}: {}", c.describe());
        assert!(
            (c.cond / want - 1.0).abs() < 1e-3,
            "{name}: cond {:e}",
            c.cond
        );
        let b = RIK_ROUNDOFF_COEFF / (c.lambda_min * c.lambda_min);
        assert!((c.bound - b).abs() <= 1e-12 * b);
        assert!(c.bound < FD_ROUTE_ERROR, "{name}: {}", c.describe());
    }
}

// ---------------------------------------------------------------------------
// Accuracy above the earlier fixed gate (cond 1.7e8), and the mutant
// ---------------------------------------------------------------------------

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

/// Analytic vs Richardson FD of the own DF gradient (`h` = 2e-3 / 1e-3, SCF
/// re-converged to 1e-14 / 1e-12), Ha/Bohr². Derived from both sides:
/// measured 0.9e-9 and 2.1e-9 in two runs (the reference floor: its
/// asymmetry on propane def2-SVP is 9.7e-10, and the eigen-vs-LU inverse gap
/// is 0.7e-9 .. 2.2e-9), so 1e-8 is 5x above a correct Hessian; the dropped
/// 2c second derivative misses by 0.23, 2e7 bars.
const BAR_ABOVE_GATE: f64 = 1e-8;

#[test]
#[ignore = "tier slow (>=60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
fn propane_above_the_old_gate_matches_finite_differences_and_the_metric_mutant_does_not() {
    // Physics: the Hessian is the derivative of the energy; its conditioning
    // roundoff is bounded by RIK_ROUNDOFF_COEFF / lambda_min^2, far below
    // the bar. Artifact: a K-specific loss at cond 1.7e8 would show as a miss
    // above the bar AND a gap between the eigen-built and the LU-built metric
    // inverse above the bound.
    use ferric_scf::hessian_df_k::RikConditioning;
    if !has_deriv2() {
        return;
    }
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/alkane_3.xyz"
    );
    let mol = Molecule::load_xyz(path).unwrap();
    let (basis_name, aux) = ("sto-3g", "def2-universal-jkfit");
    let auxbs = PreparedBasis::new(&mol, &basis::bundled(aux).unwrap()).unwrap();
    let v = coulomb_metric_2c(Operator::coulomb(), &auxbs).unwrap();
    let (lam, _) = v.eigh(UPLO::Upper).unwrap();
    let cond = RikConditioning::from_spectrum(lam[0], lam[lam.len() - 1]);
    assert!(
        cond.cond > 1e8,
        "the fixture must sit above the old gate: {:e}",
        cond.cond
    );
    assert!(cond.admits_analytic(), "{}", cond.describe());
    let s = setup(&mol, basis_name, aux);
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let mutated = |m: RikMutants| {
        ferric_scf::hessian::rij_support::rhf_hessian_parts_rik_mutated(
            &ctx, &mol, &s.prep, op, &s.bounds, &s.rhf, &s.cfg, &m,
        )
        .unwrap()
        .total()
    };
    let analytic = rhf_hessian_parts(&ctx, &mol, &s.prep, op, &s.bounds, &s.rhf, &s.cfg)
        .unwrap()
        .total();
    let lu = mutated(RikMutants {
        explicit_inverse: true,
        ..Default::default()
    });
    let rich = {
        let a = fd_hessian(&mol, basis_name, aux, 2e-3);
        let b = fd_hessian(&mol, basis_name, aux, 1e-3);
        let r = (4.0 * b - a) / 3.0;
        (&r + &r.t()) * 0.5
    };
    let miss = max_abs_diff(&analytic, &rich);
    let lu_gap = max_abs_diff(&analytic, &lu);
    let mutant = max_abs_diff(
        &mutated(RikMutants {
            drop_metric_second: true,
            ..Default::default()
        }),
        &rich,
    );
    eprintln!(
        "propane {basis_name}/{aux}: {} | analytic vs FD {miss:.2e}, eigen vs LU inverse \
         {lu_gap:.2e}, drop-2c-second-derivative mutant {mutant:.2e}",
        cond.describe()
    );
    assert!(miss < BAR_ABOVE_GATE, "analytic misses FD by {miss:.2e}");
    assert!(
        lu_gap < cond.bound,
        "the two metric inverses differ by {lu_gap:.2e}, above the bound {:.2e}",
        cond.bound
    );
    assert!(
        mutant > 10.0 * BAR_ABOVE_GATE,
        "the dropped 2c second derivative misses by only {mutant:.2e}"
    );
}

/// `base` on every atom plus a near-duplicate s function (exponent
/// x (1 + eps)) on each hydrogen: pushes `lambda_min` down at fixed
/// `lambda_max`.
fn twin_aux(mol: &Molecule, base: &str, eps: f64) -> PreparedBasis {
    use ferric_core::basis::Shell;
    let mut aux = basis::bundled(base).unwrap();
    let h = aux.shells.get_mut(&1).unwrap();
    let first = h.iter().find(|s| s.l == 0).unwrap().clone();
    let a = first.exponents[0];
    h.push(Shell {
        exponents: vec![a * (1.0 + eps)],
        coefficients: vec![(2.0 * a / std::f64::consts::PI).powf(0.75)],
        ..first
    });
    PreparedBasis::new(mol, &aux).unwrap()
}

/// The roundoff law past the measured molecules: the RI-K skeleton and `K^x`
/// with the metric inverse built two ways (eigen `W W` vs LU) at the SCF
/// density of `RIKCOND_MOL` / `RIKCOND_BASIS`, over twin-augmented
/// def2-universal-jkfit bases (`RIKCOND_EPS`, comma-separated) that push
/// cond(V) up to where `lambda_min` reaches the gradient's noise band.
#[test]
#[ignore = "measurement harness: eigen vs LU metric inverse over twin-augmented fitting bases"]
fn measure_inverse_construction_gap_versus_cond() {
    use ferric_scf::hessian::rij_support::rik_terms_with_aux;
    if !has_deriv2() {
        return;
    }
    let path = std::env::var("RIKCOND_MOL").expect("RIKCOND_MOL");
    let basis_name = env_or("RIKCOND_BASIS", "sto-3g");
    let eps: Vec<f64> = env_or("RIKCOND_EPS", "1e-1,1e-2,3e-3")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let mol = Molecule::load_xyz(&path).unwrap();
    let base = "def2-universal-jkfit";
    let s = setup(&mol, &basis_name, base);
    let nocc = (mol.nelec() / 2) as usize;
    let x = s.rhf.mos_r().slice(ndarray::s![.., ..nocc]).to_owned() * std::f64::consts::SQRT_2;
    let op = Operator::coulomb();
    let n = mol.atoms.len();
    for e in eps {
        let aux = twin_aux(&mol, base, e);
        let v = coulomb_metric_2c(op, &aux).unwrap();
        let (lam, _) = v.eigh(UPLO::Upper).unwrap();
        let (lmin, cond) = (lam[0], lam[lam.len() - 1] / lam[0]);
        let band = lam.iter().filter(|&&l| l < 1e-7).count();
        let m = |inv: bool| {
            rik_terms_with_aux(
                n,
                &s.prep,
                twin_aux(&mol, base, e),
                op,
                usize::MAX,
                &x,
                &RikMutants {
                    explicit_inverse: inv,
                    allow_ill_conditioned: true,
                    ..Default::default()
                },
            )
        };
        let (a, b) = match (m(false), m(true)) {
            (Ok(a), Ok(b)) => (a, b),
            (r1, r2) => {
                eprintln!(
                    "TWINLU eps {e:.0e}: lambda_min {lmin:.2e} cond {cond:.2e} band {band}: \
                     refused ({:?} / {:?})",
                    r1.err().map(|e| e.to_string()),
                    r2.err().map(|e| e.to_string())
                );
                continue;
            }
        };
        let sk = a
            .skeleton
            .iter()
            .zip(b.skeleton.iter())
            .fold(0.0f64, |acc, (p, q)| acc.max((p - q).abs()));
        let kx = a
            .first_order_k
            .iter()
            .zip(&b.first_order_k)
            .flat_map(|(p, q)| p.iter().zip(q.iter()).map(|(u, w)| (u - w).abs()))
            .fold(0.0f64, f64::max);
        eprintln!(
            "TWINLU eps {e:.0e}: lambda_min {lmin:.2e} cond {cond:.2e} band {band} | skeleton gap \
             {sk:.2e} ({:.2e} per cond), K^x gap {kx:.2e} ({:.2e} per cond)",
            sk / cond,
            kx / cond
        );
    }
}
