//! Analytic ROHF Hessian for density-fitted Coulomb and/or exchange (RI-J,
//! RI-K, RI-JK; issue #408) against INDEPENDENT constructions.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test df_hessian_rohf
//! ```
//!
//! # What is compared to what
//!
//! * EXACTNESS ANCHOR (closed-shell limit): a singlet ROHF with RI has an
//!   empty open shell, so its Hessian must equal the RHF RI-J / RI-K / RI-JK
//!   Hessian term by term. BLIND to everything that needs open orbitals (a
//!   swapped spin, the β factor taken from α, one spin's density for K): the
//!   open-shell FD test and the mutants carry those.
//! * The TOTAL Hessian vs a central difference (Richardson) of ferric's own
//!   analytic ROHF DF gradient (`rohf_gradient`, validated against
//!   `df_jk_gradient.rs`) with the SCF re-converged at every displaced
//!   geometry; NH2 / HO2 doublets and a CH2 triplet.
//! * DF off: `df_j_aux = None` and `Some("")` take the exact path and give
//!   bit-identical Hessians (the comparison against the pre-change commit is
//!   a measurement recorded in the PR, not a committed test).
//! * Negative controls (exact-integral, RI-J-only, RI-K-only Hessians of the
//!   same molecule) and production-wired mutants of the DF constructions.
//!
//! PySCF has no density-fitted ROHF Hessian (`df.hessian` covers RHF/UHF/RKS/
//! UKS only), so there is NO PySCF leg: the references are ferric's own
//! gradient (FD) and the RHF/UHF-validated DF code the ROHF path shares.
//!
//! # Hypotheses (written before measuring)
//!
//! * Physics (right): every column matches the FD of the own DF gradient at
//!   the FD floor; exact / RI-J-only / RI-K-only Hessians miss it by far more
//!   than the bar.
//! * Artifact (a spin-channel defect): the closed-shell anchor PASSES (the
//!   two spins coincide, no open shell) and only the open-shell FD and the
//!   mutants fail. Artifact (the FD shares a defect with the analytic code):
//!   both use `rohf_gradient`'s fitted route; the independent legs are the
//!   exact-integral ROHF Hessian (validated against PySCF) in the trivial
//!   limit of the shared per-spin DF code (the UHF tests) and the negative
//!   controls.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::rohf_gradient;
use ferric_scf::hessian::rij_support::{RijMutants, RikMutants, UhfDfMutants};
use ferric_scf::hessian::rohf_support::{
    rohf_hessian_parts_df_mutated, rohf_hessian_parts_mutated, RohfMutation,
};
use ferric_scf::hessian::{
    analytic_rohf_hessian_available, rhf_hessian_parts, rohf_hessian_parts, rohf_hessian_preflight,
    RhfHessianParts,
};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const CH2_DISTORTED: &str = "3\nCH2 triplet distorted off C2v\n\
C      0.000000     0.000000     0.000000\n\
H      0.990000     0.020000     0.420000\n\
H     -0.970000    -0.030000     0.440000\n";

const NH2_DISTORTED: &str = "3\nNH2 doublet distorted off C2v\n\
N      0.000000     0.000000     0.000000\n\
H      0.010000     0.810000    -0.600000\n\
H      0.020000    -0.790000    -0.640000\n";

const HO2_LIFTED: &str = "3\nHO2 doublet, H out of plane\n\
O      0.000000     0.000000     0.000000\n\
O      1.331000     0.000000     0.050000\n\
H     -0.239672     0.940956     0.300000\n";

const WATER_DISTORTED: &str = "3\nH2O distorted off C2v\n\
O      0.010000    -0.020000     0.120000\n\
H      0.030000     0.770000    -0.460000\n\
H     -0.020000    -0.740000    -0.490000\n";

fn has_deriv2() -> bool {
    let order = ferric_integrals::engine::libint_max_deriv_order();
    if order < 2 {
        eprintln!("SKIP: libint2 generated with derivative order {order}; no analytic Hessian");
    }
    order >= 2
}

fn scf_config(j_aux: &str, k_aux: &str) -> RhfConfig {
    RhfConfig {
        max_iter: 500,
        energy_conv: 1e-14,
        density_conv: 1e-12,
        df_j_aux: Some(j_aux.to_string()),
        df_k_aux: Some(k_aux.to_string()),
        ..Default::default()
    }
}

struct Setup {
    mol: Molecule,
    basis_name: &'static str,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    rohf: ScfResult,
    cfg: RhfConfig,
}

fn prep_for(m: &Molecule, basis_name: &str) -> (PreparedBasis, SchwarzBounds) {
    let prep = PreparedBasis::new(m, &basis::bundled(basis_name).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (prep, bounds)
}

fn setup(xyz: &str, mult: usize, basis_name: &'static str, j: &str, k: &str) -> Setup {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let (prep, bounds) = prep_for(&mol, basis_name);
    let cfg = scf_config(j, k);
    let rohf = solve_rohf(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &cfg,
    )
    .unwrap();
    assert!(rohf.converged);
    if j.is_empty() && k.is_empty() {
        assert!(rohf.df_jk.is_none(), "exact SCF carries no route");
    } else {
        let route = rohf.df_jk.as_ref().expect("a DF route");
        assert_eq!(route.j_aux.as_deref(), Some(j).filter(|v| !v.is_empty()));
        assert_eq!(route.k_aux.as_deref(), Some(k).filter(|v| !v.is_empty()));
    }
    Setup {
        mol,
        basis_name,
        prep,
        bounds,
        rohf,
        cfg,
    }
}

fn parts(s: &Setup) -> RhfHessianParts {
    rohf_hessian_parts(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &s.rohf,
        &s.cfg,
    )
    .unwrap()
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

fn fd_hessian(
    mol: &Molecule,
    h: f64,
    grad: &mut impl FnMut(&Molecule) -> Array2<f64>,
) -> Array2<f64> {
    let n3 = 3 * mol.atoms.len();
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

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn assert_close(ctx: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) -> f64 {
    let d = max_abs_diff(got, want);
    eprintln!(
        "{ctx}: max|analytic - ref| = {d:.3e} (max|ref| {:.3e}, tol {tol:.0e})",
        max_abs(want)
    );
    assert!(d < tol, "{ctx}: max|analytic - ref| = {d:.3e} >= {tol:.0e}");
    d
}

/// Richardson FD Hessian (h = 2e-3 and 1e-3) of the own DF ROHF gradient, the
/// SCF re-converged (default guess) at every displaced geometry.
fn fd_own_gradient(s: &Setup) -> Array2<f64> {
    let mut grad = |m: &Molecule| {
        let (prep, bounds) = prep_for(m, s.basis_name);
        let op = Operator::coulomb();
        let r = solve_rohf(&ParallelContext::default(), m, &prep, op, &bounds, &s.cfg).unwrap();
        assert!(r.converged, "displaced ROHF not converged");
        assert!(
            (r.energy - s.rohf.energy).abs() < 1e-3,
            "displaced ROHF changed state"
        );
        rohf_gradient(m, &prep, op, &bounds, &r, None).unwrap()
    };
    let coarse = fd_hessian(&s.mol, 2e-3, &mut grad);
    let fine = fd_hessian(&s.mol, 1e-3, &mut grad);
    (4.0 * fine - coarse) / 3.0
}

/// Agreement bar with the Richardson FD of the own DF ROHF gradient (same as
/// the UHF bar: FD floor 1e-10..3e-9 with SCF converged to 1e-14).
const BAR_OWN: f64 = 1e-8;
/// Closed-shell limit bar (per term and total).
const TOL_ANCHOR: f64 = 1e-10;

fn check_full(xyz: &str, mult: usize, basis_name: &'static str, j: &str, k: &str) -> f64 {
    if !has_deriv2() {
        return 0.0;
    }
    let s = setup(xyz, mult, basis_name, j, k);
    let p = parts(&s);
    let fd = fd_own_gradient(&s);
    eprintln!(
        "{basis_name}/J={j:?}/K={k:?}: CPHF iterations max {}, residual {:.2e}, asym {:.2e}",
        p.cphf_iterations.iter().max().unwrap(),
        p.cphf_max_residual,
        p.response_asymmetry
    );
    assert!(
        max_abs_diff(&p.skeleton(), &fd) > 100.0 * BAR_OWN,
        "skeleton alone matches FD: the full test cannot see the response"
    );
    let miss = assert_close(
        &format!("{basis_name}/J={j:?}/K={k:?}: full ROHF Hessian"),
        &p.total(),
        &fd,
        BAR_OWN,
    );
    assert!(p.response_asymmetry < 1e-9);
    miss
}

// ---------------------------------------------------------------------------
// Exactness anchor: closed-shell limit
// ---------------------------------------------------------------------------

fn closed_shell_anchor(basis_name: &'static str, j: &str, k: &str) {
    if !has_deriv2() {
        return;
    }
    let mol = Molecule::parse_xyz(WATER_DISTORTED, 0, 1).unwrap();
    let (prep, bounds) = prep_for(&mol, basis_name);
    let op = Operator::coulomb();
    let cfg = scf_config(j, k);
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let rohf = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(rohf.converged && rohf.df_jk.is_some());
    let r = rhf_hessian_parts(&ctx, &mol, &prep, op, &bounds, &rhf, &cfg).unwrap();
    let o = rohf_hessian_parts(&ctx, &mol, &prep, op, &bounds, &rohf, &cfg).unwrap();
    let tag = format!("closed-shell anchor {basis_name} J={j:?} K={k:?}");
    assert!(max_abs(&r.response) > 1e-3, "vacuous: RHF response ~ 0");
    assert!(max_abs(&r.two_electron) > 1e-3, "vacuous: 2e skeleton ~ 0");
    for (what, got, want) in [
        ("nuclear", &o.nuclear, &r.nuclear),
        ("one_electron", &o.one_electron, &r.one_electron),
        ("overlap", &o.overlap, &r.overlap),
        ("two_electron", &o.two_electron, &r.two_electron),
        ("response", &o.response, &r.response),
    ] {
        assert_close(&format!("{tag} {what}"), got, want, TOL_ANCHOR);
    }
    assert_close(&format!("{tag} total"), &o.total(), &r.total(), TOL_ANCHOR);
}

#[test]
fn closed_shell_limit_ri_j() {
    closed_shell_anchor("6-31g", "cc-pvdz-ri", "");
}

#[test]
fn closed_shell_limit_ri_k() {
    closed_shell_anchor("6-31g", "", "cc-pvdz-ri");
}

#[test]
fn closed_shell_limit_ri_jk() {
    closed_shell_anchor("6-31g", "cc-pvdz-ri", "cc-pvdz-ri");
}

// ---------------------------------------------------------------------------
// Own-gradient FD (real open shells)
// ---------------------------------------------------------------------------

#[test]
fn full_hessian_matches_fd_nh2_sto3g_ri_jk() {
    check_full(
        NH2_DISTORTED,
        2,
        "sto-3g",
        "def2-universal-jkfit",
        "def2-universal-jkfit",
    );
}

#[test]
fn full_hessian_matches_fd_ch2_triplet_sto3g_ri_j() {
    check_full(CH2_DISTORTED, 3, "sto-3g", "def2-universal-jkfit", "");
}

#[test]
fn full_hessian_matches_fd_ch2_triplet_sto3g_ri_k() {
    check_full(CH2_DISTORTED, 3, "sto-3g", "", "def2-universal-jkfit");
}

#[test]
fn full_hessian_matches_fd_nh2_631g_ri_jk() {
    check_full(NH2_DISTORTED, 2, "6-31g", "cc-pvdz-ri", "cc-pvdz-ri");
}

#[test]
fn full_hessian_matches_fd_ch2_triplet_631g_ri_jk() {
    check_full(CH2_DISTORTED, 3, "6-31g", "cc-pvdz-ri", "cc-pvdz-ri");
}

#[test]
fn full_hessian_matches_fd_ho2_sto3g_ri_jk() {
    check_full(
        HO2_LIFTED,
        2,
        "sto-3g",
        "def2-universal-jkfit",
        "def2-universal-jkfit",
    );
}

#[test]
#[ignore = "slow: NH2 cc-pVDZ / def2-universal-jkfit RI-JK own-gradient FD (ROHF)"]
fn full_hessian_matches_fd_nh2_ccpvdz_jkfit() {
    check_full(
        NH2_DISTORTED,
        2,
        "cc-pvdz",
        "def2-universal-jkfit",
        "def2-universal-jkfit",
    );
}

#[test]
#[ignore = "slow: CH2 triplet cc-pVDZ / def2-universal-jkfit RI-JK own-gradient FD (ROHF)"]
fn full_hessian_matches_fd_ch2_ccpvdz_jkfit() {
    check_full(
        CH2_DISTORTED,
        3,
        "cc-pvdz",
        "def2-universal-jkfit",
        "def2-universal-jkfit",
    );
}

// ---------------------------------------------------------------------------
// DF off, negative controls, mutants
// ---------------------------------------------------------------------------

#[test]
fn df_off_configurations_take_the_exact_path_bit_identically() {
    if !has_deriv2() {
        return;
    }
    for (xyz, mult) in [(NH2_DISTORTED, 2), (CH2_DISTORTED, 3)] {
        let unset = {
            let mut s = setup(xyz, mult, "sto-3g", "", "");
            s.cfg.df_j_aux = None;
            s.cfg.df_k_aux = None;
            s
        };
        let empty = setup(xyz, mult, "sto-3g", "", "");
        let a = parts(&unset).total();
        let b = parts(&empty).total();
        assert!(
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "unset and empty DF names must give bit-identical Hessians"
        );
    }
}

/// Mutation bar (NH2 6-31G / cc-pVDZ-RI, RI-JK). Derived from both sides: the
/// production agreement with the differenced own gradient on this system is
/// measured in `production_agrees_with_fd_at_the_mutation_bar`; the stencil
/// mutants of the shared DF code are the closest to the pass line (K x10 4.8e-8,
/// J x10 2.4e-7 on the UHF system); every ROHF-specific mutant misses by
/// orders of magnitude more.
const BAR_MUT: f64 = 3e-9;

type Mutant = (
    &'static str,
    RijMutants,
    RikMutants,
    UhfDfMutants,
    RohfMutation,
);

fn mutated_total(s: &Setup, m: &Mutant) -> Result<Array2<f64>, ferric_core::FerricError> {
    rohf_hessian_parts_df_mutated(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &s.rohf,
        &s.cfg,
        m.4,
        (&m.1, &m.2, &m.3),
    )
    .map(|p| p.total())
}

fn all_mutants() -> Vec<Mutant> {
    let (rj, rk, uh) = (
        RijMutants::default(),
        RikMutants::default(),
        UhfDfMutants::default(),
    );
    let n = RohfMutation::None;
    let u = |m: UhfDfMutants| (rj, rk, m);
    let mut v: Vec<Mutant> = Vec::new();
    let mut push = |name: &'static str, r: (RijMutants, RikMutants, UhfDfMutants)| {
        v.push((name, r.0, r.1, r.2, n));
    };
    push(
        "J from alpha density only",
        u(UhfDfMutants {
            j_from_alpha: true,
            ..uh
        }),
    );
    push(
        "alpha and beta K^x swapped",
        u(UhfDfMutants {
            swap_k_channels: true,
            ..uh
        }),
    );
    push(
        "closed-shell channel (alpha -1/4, X = sqrt2 C) for each spin",
        u(UhfDfMutants {
            closed_shell_channel: true,
            ..uh
        }),
    );
    push(
        "beta channel dropped",
        u(UhfDfMutants {
            drop_beta_channel: true,
            ..uh
        }),
    );
    push(
        "beta occupied factor = alpha (open orbitals in the beta channel)",
        u(UhfDfMutants {
            beta_factor_is_alpha: true,
            ..uh
        }),
    );
    push(
        "CPHF: exchange of the other spin",
        u(UhfDfMutants {
            cphf_wrong_spin_k: true,
            ..uh
        }),
    );
    push(
        "CPHF: exact K where the SCF fitted it",
        u(UhfDfMutants {
            cphf_exact_k: true,
            ..uh
        }),
    );
    push(
        "CPHF: exchange of the TOTAL response density per spin",
        u(UhfDfMutants {
            cphf_k_from_total: true,
            ..uh
        }),
    );
    push(
        "CPHF: Coulomb from the alpha response density only",
        u(UhfDfMutants {
            cphf_j_from_alpha: true,
            ..uh
        }),
    );
    push(
        "K: drop 2c second derivative",
        (
            rj,
            RikMutants {
                drop_metric_second: true,
                ..rk
            },
            uh,
        ),
    );
    push(
        "K: drop cross terms",
        (
            rj,
            RikMutants {
                drop_cross: true,
                ..rk
            },
            uh,
        ),
    );
    push(
        "K: drop K^x altogether",
        (
            rj,
            RikMutants {
                drop_kx: true,
                ..rk
            },
            uh,
        ),
    );
    push(
        "K: drop K^x metric term",
        (
            rj,
            RikMutants {
                drop_kx_metric: true,
                ..rk
            },
            uh,
        ),
    );
    push(
        "K: exchange weight x2",
        (
            rj,
            RikMutants {
                alpha_factor: 2.0,
                ..rk
            },
            uh,
        ),
    );
    push(
        "J: drop 2c second derivative",
        (
            RijMutants {
                drop_metric_second: true,
                ..rj
            },
            rk,
            uh,
        ),
    );
    push(
        "J: drop cross terms",
        (
            RijMutants {
                drop_cross: true,
                ..rj
            },
            rk,
            uh,
        ),
    );
    push(
        "J: omit aux-fit part of J^x",
        (
            RijMutants {
                omit_aux_jx: true,
                ..rj
            },
            rk,
            uh,
        ),
    );
    // ROHF response mutants (shared with the exact path) must still fail with DF.
    for (name, m) in [
        ("response: kappa_oc dropped", RohfMutation::DropOcBlock),
        (
            "response: diagonal-gap operator",
            RohfMutation::DiagonalGapOperator,
        ),
        ("response: -Tr[W^y S^x] omitted", RohfMutation::DropWOverlap),
        (
            "response: F_alpha^x for both spins",
            RohfMutation::OneSpinFockX,
        ),
        (
            "response: F_alpha^x <-> F_beta^x",
            RohfMutation::SwapSpinFockX,
        ),
        (
            "response: G(D^S) dropped",
            RohfMutation::DropOverlapDensityG,
        ),
    ] {
        v.push((name, rj, rk, uh, m));
    }
    v
}

#[test]
fn every_wired_mutant_misses_by_more_than_ten_bars() {
    if !has_deriv2() {
        return;
    }
    let s = setup(NH2_DISTORTED, 2, "6-31g", "cc-pvdz-ri", "cc-pvdz-ri");
    let prod = parts(&s).total();
    let again = mutated_total(
        &s,
        &(
            "production",
            RijMutants::default(),
            RikMutants::default(),
            UhfDfMutants::default(),
            RohfMutation::None,
        ),
    )
    .unwrap();
    assert_eq!(
        max_abs_diff(&prod, &again),
        0.0,
        "production not reproducible"
    );
    // The exact-path wrapper with DF mutants defaulted is the same function.
    let via_old = rohf_hessian_parts_mutated(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &s.rohf,
        &s.cfg,
        RohfMutation::None,
    )
    .unwrap()
    .total();
    assert_eq!(max_abs_diff(&prod, &via_old), 0.0);
    for m in all_mutants() {
        match mutated_total(&s, &m) {
            Ok(h) => {
                let miss = max_abs_diff(&h, &prod);
                eprintln!(
                    "MUTANT {}: miss {miss:.3e} ({:.1e} bars)",
                    m.0,
                    miss / BAR_MUT
                );
                assert!(
                    miss > 10.0 * BAR_MUT,
                    "mutant {} misses by only {miss:.3e}",
                    m.0
                );
            }
            Err(e) => {
                // A kernel made indefinite is caught by the solver's typed error.
                let msg = e.to_string();
                eprintln!("MUTANT {}: typed error {msg}", m.0);
                assert!(
                    msg.contains("curvature")
                        || msg.contains("positive definite")
                        || msg.contains("converge")
                        || msg.contains("rotation"),
                    "mutant {} failed with an unexpected error: {msg}",
                    m.0
                );
            }
        }
    }
}

#[test]
fn production_agrees_with_fd_at_the_mutation_bar() {
    if !has_deriv2() {
        return;
    }
    let s = setup(NH2_DISTORTED, 2, "6-31g", "cc-pvdz-ri", "cc-pvdz-ri");
    let fd = fd_own_gradient(&s);
    let prod = max_abs_diff(&parts(&s).total(), &fd);
    eprintln!("production vs FD on the mutation system: {prod:.3e} (bar {BAR_MUT:.0e})");
    assert!(prod < BAR_MUT);
}

#[test]
fn negative_controls_miss_the_ri_jk_reference() {
    if !has_deriv2() {
        return;
    }
    let aux = "cc-pvdz-ri";
    let jk = setup(NH2_DISTORTED, 2, "6-31g", aux, aux);
    let prod = parts(&jk).total();
    for (name, j, k) in [
        ("exact-integral ROHF", "", ""),
        ("RI-J only (exact K)", aux, ""),
        ("RI-K only (exact J)", "", aux),
    ] {
        let h = parts(&setup(NH2_DISTORTED, 2, "6-31g", j, k)).total();
        let miss = max_abs_diff(&h, &prod);
        eprintln!(
            "NEGATIVE CONTROL {name}: miss {miss:.3e} ({:.1e} bars of {BAR_OWN:.0e})",
            miss / BAR_OWN
        );
        assert!(miss > 10.0 * BAR_OWN, "{name}: {miss:.3e}");
    }
    let miss_resp = max_abs_diff(&parts(&jk).skeleton(), &prod);
    assert!(miss_resp > 10.0 * BAR_OWN);
}

// ---------------------------------------------------------------------------
// Dispatch and remaining refusals
// ---------------------------------------------------------------------------

#[test]
fn ri_is_accepted_and_the_remaining_refusals_are_typed() {
    let op = Operator::coulomb();
    let cfg = scf_config("def2-universal-jkfit", "def2-universal-jkfit");
    rohf_hessian_preflight(op, &cfg).expect("ROHF RI-JK preflight");
    let mol = Molecule::parse_xyz(NH2_DISTORTED, 0, 2).unwrap();
    let (prep, _) = prep_for(&mol, "sto-3g");
    if has_deriv2() {
        analytic_rohf_hessian_available(&mol, &prep, op, &cfg).expect("ROHF RI probe");
    }
    // An external potential next to RI stays refused.
    let mut with_ext = cfg.clone();
    with_ext.external_potential = Some(ferric_core::external_potential::ExternalPotential {
        point_charges: vec![ferric_core::external_potential::PointCharge {
            q: 0.3,
            x: 5.0,
            y: 0.0,
            z: 0.0,
        }],
        ..Default::default()
    });
    let e = rohf_hessian_preflight(op, &with_ext)
        .unwrap_err()
        .to_string();
    assert!(e.contains("external potential"), "{e}");
    // Kohn-Sham, COSX and an ECP stay refused.
    let mut ks = cfg.clone();
    ks.xc = Some("pbe".into());
    assert!(rohf_hessian_preflight(op, &ks).is_err());
    let svp = basis::bundled("def2-svp").unwrap();
    let mut hi = Molecule::parse_xyz("2\nHI\nH 0 0 0\nI 0 0 1.75\n", 0, 1).unwrap();
    hi.apply_ecp(&svp);
    let hi_prep = PreparedBasis::new(&hi, &svp).unwrap();
    let e = analytic_rohf_hessian_available(&hi, &hi_prep, op, &cfg)
        .unwrap_err()
        .to_string();
    assert!(e.contains("effective core potential"), "{e}");
    // A route whose fitting basis differs from the configuration's.
    let s = setup(NH2_DISTORTED, 2, "sto-3g", "def2-universal-jkfit", "");
    let other = scf_config("cc-pvdz-ri", "");
    let e = rohf_hessian_parts(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        op,
        &s.bounds,
        &s.rohf,
        &other,
    )
    .map(|_| ())
    .unwrap_err()
    .to_string();
    assert!(e.contains("differs from the configuration"), "{e}");
}
