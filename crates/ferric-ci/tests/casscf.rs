//! Two-step CASSCF (`ferric_ci::casscf`) anchors.
//!
//! Systems: `testdata/molecules/water.xyz` and `n2.xyz` (the geometries of
//! `scripts/prototypes/casscf_proto.py`), ferric exact-J/K RHF orbitals as
//! the start, no frozen orbitals beyond the CAS core. The CI step is
//! ferric-ci's Davidson with spin-summed RDMs.
//!
//! # Anchor 1: active space = full space
//!
//! With no core and no virtual orbitals every rotation is redundant, so the
//! energy at the RHF orbitals is FCI and the whole matrix `2(F − Fᵀ)` must
//! vanish. Compared with `run_cas_ci` over the full window, which builds its
//! integrals by a different path (MO transform of `h` and the ERIs, no `J`/`K`).
//!
//! Bars:
//!
//! * `2(F_pq − F_qp) = ⟨c|[E_pq − E_qp, H]|c⟩` for the normalized Ritz vector
//!   `c`, and with `r = Hc − θc` this is `2 cᵀ A r`, `A = E_pq − E_qp`. With
//!   `‖E_pq‖ ≤ 2` for `p ≠ q`, `|2(F_pq − F_qp)| ≤ 8 ‖r‖`. Rounding adds
//!   `4 ε n² max|F|` (a random walk over the `n⁴`-term contraction).
//! * Two Ritz values of the same Hamiltonian lie within `‖r₁‖ + ‖r₂‖` of its
//!   lowest eigenvalue; rounding adds `4 ε |E| √N_det` per solve.
//!
//! # Anchor 2: analytic gradient vs finite difference, per class
//!
//! Point: RHF orbitals rotated by a random `κ` (normal, scale 0.05, a fixed
//! LCG seed) over all nonredundant pairs. Each element: 5-point central
//! difference of the CI-relaxed energy at `h = 1e-3` and `h/2`,
//! Richardson-extrapolated `(16 FD(h/2) − FD(h))/15`.
//!
//! Bar 1e-8 (the derivation in `scripts/prototypes/casscf_derivation.md`,
//! re-checked for this solver):
//!
//! * truncation: Richardson removes the `h⁴ f⁽⁵⁾/30` term. Its size,
//!   `FD(h) − FD(h/2)`, is printed per class; the largest is 2.0e-7 (H2O
//!   core-active). The remaining term is `O(h⁶)` and measured below 2.5e-10;
//! * energy noise: `δE ≈ 4 ε |E| √N_det` (2e-12 for N2, 400 determinants)
//!   amplified by `3.3/h` gives 6e-9 at worst;
//! * CI error in the analytic gradient: first order in the Davidson
//!   residual, below `10 × ci_conv_thresh = 1e-10` (Davidson accepts
//!   `10 ×` its threshold when the correction vector vanishes; measured
//!   2e-11 and 6e-11).
//!
//! # Anchor 3: converged energies
//!
//! The prototype's committed energies (`casscf_derivation.md`, anchor 3),
//! which match `pyscf.mcscf.CASSCF` on ferric's integrals to 1.4e-12 /
//! 1.8e-13 Ha. Bar: near a minimum `E − E_min ≤ ‖g‖²/(2 λ_min)` for each
//! run, so `|ΔE| ≤ (‖g‖² + ‖g_proto‖²)/(2 λ_min)` plus the rounding of the
//! two energies, `4 ε |E| √N_det` each. `λ_min` is the lowest eigenvalue of
//! the CI-relaxed orbital Hessian, computed here by central differences of
//! the analytic gradient (`h = 1e-4`) at the converged orbitals and compared
//! with the derivation note's value.
//!
//! # Measured (2026-11-11) and mutations
//!
//! Tables and the mutation record are in `crates/ferric-ci/theory.md`
//! ("Two-step CASSCF").

use ferric_ci::casscf::{optimize, CasScfConfig, CasScfProblem, RotationClass};
use ferric_ci::rdm::rdm_energy;
use ferric_ci::{run_cas_ci, run_casscf, CasCiConfig};
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
use std::path::PathBuf;

const EPS: f64 = f64::EPSILON;
const CLASSES: [RotationClass; 3] = [
    RotationClass::CoreActive,
    RotationClass::CoreVirtual,
    RotationClass::ActiveVirtual,
];

fn setup(xyz: &str, basis_name: &str) -> (Molecule, PreparedBasis, ScfResult) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/molecules")
        .join(xyz);
    let mol = Molecule::load_xyz(path.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: {e:?}", path.display()));
    let prep = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        max_iter: 300,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{xyz}/{basis_name}: RHF failed: {e:?}"));
    assert!(rhf.converged, "{xyz}/{basis_name}: RHF did not converge");
    assert!(
        rhf.df_jk.is_none(),
        "{xyz}/{basis_name}: RHF must use exact J/K"
    );
    (mol, prep, rhf)
}

fn maxabs<'a>(it: impl IntoIterator<Item = &'a f64>) -> f64 {
    it.into_iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

// ------------------------------------------------------------ anchor 1

fn full_space_case(xyz: &str, basis_name: &str, nel: (usize, usize)) {
    let (mol, prep, rhf) = setup(xyz, basis_name);
    let nmo = rhf.mos_r().ncols();
    assert_eq!(nel.0 + nel.1, mol.nelec() as usize);
    let cfg = CasScfConfig {
        n_active: nmo,
        n_elec_active: nel,
        n_core: 0,
        ..Default::default()
    };
    let problem = CasScfProblem::new(&mol, &prep, nmo, &cfg).unwrap();
    assert!(
        problem.pairs().is_empty(),
        "full space has no nonredundant pair"
    );
    let pt = problem.evaluate(rhf.mos_r()).unwrap();
    assert!(pt.ci.converged, "CI did not converge");
    let r = pt.ci.residual_norm;
    let fmax = maxabs(&pt.f_general);
    let gmax = maxabs(&pt.gradient_matrix);
    let g_bar = 8.0 * r + 4.0 * EPS * (nmo * nmo) as f64 * fmax;

    let fci = run_cas_ci(
        &mol,
        &prep,
        &rhf,
        CasCiConfig {
            n_active: nmo,
            n_elec_active: nel,
            active_start: 0,
            conv_thresh: cfg.ci_conv_thresh,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(fci.converged);
    let de = pt.e_total - fci.e_total;
    let e_bar =
        r + fci.residual_norm + 8.0 * EPS * pt.e_total.abs() * (fci.n_determinants as f64).sqrt();
    eprintln!(
        "{xyz}/{basis_name}: ndet {} E {:.12} dE(vs run_cas_ci) {de:.2e} (bar {e_bar:.1e}) \
         max|2(F-F^T)| {gmax:.2e} (bar {g_bar:.1e}) CI residual {r:.1e} / {:.1e}",
        fci.n_determinants, pt.e_total, fci.residual_norm
    );
    assert!(gmax <= g_bar, "max|2(F-F^T)| {gmax:e} > {g_bar:e}");
    // The RDMs the gradient is built from rebuild the energy (bar of
    // tests/casci_rdms.rs): this covers the sector's RDMs, n_alpha != n_beta
    // included.
    let er = rdm_energy(&pt.ci.integrals, pt.rdms());
    let n4 = (nmo as f64).powi(4);
    let r_bar = 4.0 * EPS * er.abs_terms * (fci.n_determinants as f64 + n4).sqrt() + r;
    assert!(
        (er.e_total - pt.e_total).abs() <= r_bar,
        "RDM energy {} vs {}",
        er.e_total,
        pt.e_total
    );
    assert!(
        de.abs() <= e_bar,
        "E {} vs FCI {}: {de:e} > {e_bar:e}",
        pt.e_total,
        fci.e_total
    );

    // The driver stops at the first evaluation and returns the same energy.
    let res = optimize(&problem, rhf.mos_r().clone(), &cfg).unwrap();
    assert!(res.converged && res.iterations == 1 && res.grad_norm == 0.0);
    assert_eq!(res.e_total, pt.e_total);
}

#[test]
fn full_space_h2o_sto3g_is_fci_with_zero_orbital_gradient() {
    full_space_case("water.xyz", "sto-3g", (5, 5));
}

#[test]
fn full_space_n2_sto3g_is_fci_with_zero_orbital_gradient() {
    full_space_case("n2.xyz", "sto-3g", (7, 7));
}

/// `M_s = 1` sector (245 determinants), whose lowest root is a triplet above
/// the singlet ground state, so a CI run in the wrong sector misses
/// `run_cas_ci` by the singlet-triplet gap.
#[test]
fn full_space_h2o_triplet_sector_sto3g_is_fci_with_zero_orbital_gradient() {
    full_space_case("water.xyz", "sto-3g", (6, 4));
}

/// `M_s = 1` sector of O2 (1200 determinants) from singlet RHF orbitals.
#[test]
fn full_space_o2_triplet_sector_sto3g_is_fci_with_zero_orbital_gradient() {
    full_space_case("o2.xyz", "sto-3g", (9, 7));
}

// ------------------------------------------------------------ anchor 2

/// Normal deviates from a 64-bit LCG and Box-Muller (fixed seed).
fn normals(n: usize, seed: u64) -> Vec<f64> {
    let mut st = seed;
    let mut uni = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (uni(), uni());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect()
}

fn fd_gradient(problem: &CasScfProblem, c: &Array2<f64>, h: f64) -> Vec<f64> {
    let n = problem.pairs().len();
    (0..n)
        .map(|k| {
            let e = |m: f64| {
                let mut kv = vec![0.0; n];
                kv[k] = m * h;
                problem.energy(&problem.rotate(c, &kv).unwrap()).unwrap()
            };
            (e(-2.0) - 8.0 * e(-1.0) + 8.0 * e(1.0) - e(2.0)) / (12.0 * h)
        })
        .collect()
}

const FD_BAR: f64 = 1e-8;

fn fd_case(xyz: &str, basis_name: &str, n_core: usize, n_active: usize, nel: (usize, usize)) {
    let (mol, prep, rhf) = setup(xyz, basis_name);
    let nmo = rhf.mos_r().ncols();
    let cfg = CasScfConfig {
        n_active,
        n_elec_active: nel,
        n_core,
        ci_conv_thresh: 1e-11,
        ..Default::default()
    };
    let problem = CasScfProblem::new(&mol, &prep, nmo, &cfg).unwrap();
    let npair = problem.pairs().len();
    let kappa: Vec<f64> = normals(npair, 7).iter().map(|x| 0.05 * x).collect();
    let c = problem.rotate(rhf.mos_r(), &kappa).unwrap();
    let pt = problem.evaluate(&c).unwrap();
    assert!(
        pt.ci.converged,
        "CI did not converge (residual {:e})",
        pt.ci.residual_norm
    );
    let h = 1e-3;
    let fd1 = fd_gradient(&problem, &c, h);
    let fd2 = fd_gradient(&problem, &c, h / 2.0);
    let rich: Vec<f64> = fd1
        .iter()
        .zip(&fd2)
        .map(|(a, b)| (16.0 * b - a) / 15.0)
        .collect();
    let cls = problem.classes();
    let per_class = |v: &dyn Fn(usize) -> f64| -> [f64; 3] {
        CLASSES.map(|want| {
            (0..npair)
                .filter(|&k| cls[k] == want)
                .map(v)
                .fold(0.0f64, |m, x| m.max(x.abs()))
        })
    };
    let gmax = per_class(&|k| pt.gradient[k]);
    let trunc = per_class(&|k| fd1[k] - fd2[k]);
    let err = per_class(&|k| pt.gradient[k] - rich[k]);
    eprintln!(
        "{xyz}/{basis_name} CAS({},{n_active}) {npair} rotations, CI residual {:.1e}\n  \
         max|g|          {:.3e} {:.3e} {:.3e}\n  \
         FD(h)-FD(h/2)   {:.2e} {:.2e} {:.2e}\n  \
         analytic-FD     {:.2e} {:.2e} {:.2e}",
        nel.0 + nel.1,
        pt.ci.residual_norm,
        gmax[0],
        gmax[1],
        gmax[2],
        trunc[0],
        trunc[1],
        trunc[2],
        err[0],
        err[1],
        err[2]
    );
    for (i, cl) in CLASSES.iter().enumerate() {
        // Every class is populated and nonzero at this point, so an
        // all-zero analytic class cannot pass.
        assert!(gmax[i] > 1e-2, "{cl:?}: max|g| = {:e}", gmax[i]);
        assert!(
            err[i] <= FD_BAR,
            "{cl:?}: |analytic − FD| = {:e} > {FD_BAR:e}",
            err[i]
        );
    }
}

#[test]
fn orbital_gradient_matches_fd_h2o_cas44_631g() {
    fd_case("water.xyz", "6-31g", 3, 4, (2, 2));
}

#[test]
fn orbital_gradient_matches_fd_n2_cas66_631g() {
    fd_case("n2.xyz", "6-31g", 4, 6, (3, 3));
}

#[test]
fn orbital_gradient_matches_fd_h2o_triplet_cas44_631g() {
    fd_case("water.xyz", "6-31g", 3, 4, (3, 1));
}

#[test]
fn orbital_gradient_matches_fd_o2_triplet_cas86_631g() {
    fd_case("o2.xyz", "6-31g", 4, 6, (5, 3));
}

// ------------------------------------------------------------ anchor 3

/// Lowest eigenvalue of the CI-relaxed orbital Hessian by central
/// differences of the analytic gradient.
fn hessian_min_eig(problem: &CasScfProblem, c: &Array2<f64>, h: f64) -> f64 {
    let n = problem.pairs().len();
    let mut hes = Array2::<f64>::zeros((n, n));
    for k in 0..n {
        let mut kv = vec![0.0; n];
        kv[k] = h;
        let gp = problem
            .evaluate(&problem.rotate(c, &kv).unwrap())
            .unwrap()
            .gradient;
        kv[k] = -h;
        let gm = problem
            .evaluate(&problem.rotate(c, &kv).unwrap())
            .unwrap()
            .gradient;
        for r in 0..n {
            hes[(r, k)] = (gp[r] - gm[r]) / (2.0 * h);
        }
    }
    let sym = 0.5 * (&hes + &hes.t());
    let (ev, _) = sym.eigh(UPLO::Upper).unwrap();
    ev[0]
}

struct Reference {
    xyz: &'static str,
    n_core: usize,
    n_active: usize,
    nel: (usize, usize),
    e_proto: f64,
    g_proto: f64,
    lambda_note: f64,
    iters_proto: usize,
}

fn converged_case(r: &Reference) {
    let (mol, prep, rhf) = setup(r.xyz, "6-31g");
    let cfg = CasScfConfig {
        n_active: r.n_active,
        n_elec_active: r.nel,
        n_core: r.n_core,
        ..Default::default()
    };
    let res = run_casscf(&mol, &prep, &rhf, &cfg).unwrap();
    assert!(
        res.converged,
        "{}: not converged in {} iterations",
        r.xyz, res.iterations
    );
    let problem = CasScfProblem::new(&mol, &prep, res.mo_coeff.ncols(), &cfg).unwrap();
    let lam = hessian_min_eig(&problem, &res.mo_coeff, 1e-4);
    assert!(
        lam > 0.0,
        "{}: lowest orbital Hessian eigenvalue {lam:e} <= 0",
        r.xyz
    );
    let round = 4.0 * EPS * res.e_total.abs() * (res.n_determinants as f64).sqrt();
    let bar = (res.grad_norm.powi(2) + r.g_proto.powi(2)) / (2.0 * lam) + 2.0 * round;
    let de = res.e_total - r.e_proto;
    eprintln!(
        "{}: E {:.13} vs prototype {:.13}: dE {de:.2e} (bar {bar:.2e}); |g| {:.1e} \
         (prototype {:.1e}); lambda_min {lam:.4} (note {}); iterations {} (prototype {}); \
         CI residual {:.1e}",
        r.xyz,
        res.e_total,
        r.e_proto,
        res.grad_norm,
        r.g_proto,
        r.lambda_note,
        res.iterations,
        r.iters_proto,
        res.ci_residual_norm
    );
    assert!(
        de.abs() <= bar,
        "{}: |dE| {:e} > bar {bar:e}",
        r.xyz,
        de.abs()
    );
}

#[test]
fn converged_energy_matches_prototype_h2o_cas44_631g() {
    converged_case(&Reference {
        xyz: "water.xyz",
        n_core: 3,
        n_active: 4,
        nel: (2, 2),
        e_proto: -76.0369690965018,
        g_proto: 9.5e-8,
        lambda_note: 0.0195,
        iters_proto: 62,
    });
}

#[test]
fn converged_energy_matches_prototype_n2_cas66_631g() {
    converged_case(&Reference {
        xyz: "n2.xyz",
        n_core: 4,
        n_active: 6,
        nel: (3, 3),
        e_proto: -109.0155467897007,
        g_proto: 3.4e-8,
        lambda_note: 0.0402,
        iters_proto: 21,
    });
}

// ------------------------------------------------------------ input checks

#[test]
fn casscf_refuses_bad_bookkeeping() {
    let (mol, prep, rhf) = setup("water.xyz", "sto-3g");
    let nmo = rhf.mos_r().ncols();
    let bad = [
        (4, 1, (2, 0)), // 2 alpha electrons in 1 active orbital
        (2, 4, (2, 2)), // 2*2 + 4 != 10
        (3, 9, (2, 2)), // window exceeds nmo
    ];
    for (n_core, n_active, nel) in bad {
        let cfg = CasScfConfig {
            n_active,
            n_elec_active: nel,
            n_core,
            ..Default::default()
        };
        assert!(
            CasScfProblem::new(&mol, &prep, nmo, &cfg).is_err(),
            "accepted n_core={n_core} n_active={n_active} nel={nel:?}"
        );
    }
}

/// A CI solve that misses its residual threshold is an error, never a
/// silently used root: one Davidson iteration cannot reach 1e-10 on 36
/// determinants.
#[test]
fn unconverged_ci_is_an_error() {
    let (mol, prep, rhf) = setup("water.xyz", "6-31g");
    let cfg = CasScfConfig {
        n_active: 4,
        n_elec_active: (2, 2),
        n_core: 3,
        ci_max_iter: 1,
        ..Default::default()
    };
    let msg = match run_casscf(&mol, &prep, &rhf, &cfg) {
        Ok(r) => panic!("accepted an unconverged CI (E = {})", r.e_total),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("ci_conv_thresh"), "{msg}");
}
