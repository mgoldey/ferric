//! Closed-shell PDEP-RPA: the Lanczos static eigensolve honours
//! `Chi0Backend::Laplace`, like the Davidson arm and the open-shell Lanczos arm.
//!
//! Before this, `(Eigensolver::Lanczos, None)` always diagonalized the DENSE
//! ε̃(0) and ignored the backend, so `eigenvalues_static` / the eigenbasis of a
//! Laplace run came from a different operator than the one the energy integrates.
//!
//! Bars. n_quad = 3 is the coarse table (relative error ~1e-3 per kernel
//! element), so the Laplace and dense ε̃(0) spectra differ by ~1e-4 or more:
//! orders of magnitude above the Davidson convergence threshold (1e-9) that
//! sets the agreement bar between two solvers on the SAME operator.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Chi0Backend, Eigensolver, QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig, PdepRpaResult};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, UhfConfig};

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    op: Operator,
    bounds: SchwarzBounds,
}

fn h2o() -> Sys {
    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    Sys {
        mol,
        obs,
        dfbs,
        op,
        bounds,
    }
}

fn cfg(solver: Eigensolver, n_quad: Option<usize>, trunc: f64) -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 20,
            u0: 0.5,
        },
        frozen_core: 0,
        trunc_thresh: trunc,
        eigensolver: solver,
        eigensolver_conv_thresh: 1e-9,
        chi0_backend: match n_quad {
            Some(n_quad) => Chi0Backend::Laplace { n_quad },
            None => Chi0Backend::Dense,
        },
        ..Default::default()
    }
}

fn run(s: &Sys, rhf: &ferric_scf::result::ScfResult, c: &PdepRpaConfig) -> PdepRpaResult {
    run_pdep_rpa(&s.mol, &s.obs, &s.dfbs, s.op, rhf, c).unwrap()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// The static spectrum of a Laplace/Lanczos run is the Laplace operator's
/// (equal to Laplace/Davidson), not the dense one's.
#[test]
fn lanczos_static_spectrum_follows_the_laplace_backend() {
    let s = h2o();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &s.mol,
        &s.obs,
        s.op,
        &s.bounds,
        &RhfConfig::default(),
    )
    .unwrap();
    let n_top = 8;
    let top = |r: &PdepRpaResult| r.eigenvalues_static[..n_top].to_vec();

    let lz_lap = run(&s, &rhf, &cfg(Eigensolver::Lanczos, Some(3), 1e-3));
    let dav_lap = run(&s, &rhf, &cfg(Eigensolver::Davidson, Some(3), 1e-3));
    let lz_dense = run(&s, &rhf, &cfg(Eigensolver::Lanczos, None, 1e-3));

    let same_operator = max_abs_diff(&top(&lz_lap), &top(&dav_lap));
    let other_operator = max_abs_diff(&top(&lz_lap), &top(&lz_dense));
    assert!(
        same_operator < 1e-7,
        "Lanczos+Laplace must diagonalize the Laplace ε̃(0) (Davidson+Laplace agrees): {same_operator:.3e}"
    );
    assert!(
        other_operator > 100.0 * same_operator.max(1e-9),
        "the Laplace spectrum must be distinguishable from the dense one at n_quad=3 \
         (else this test cannot see the backend): {other_operator:.3e}"
    );
}

/// Energy anchor at matched quadrature: full basis, so the eigenbasis is
/// irrelevant and the Laplace energy must agree across (a) Lanczos vs Davidson
/// closed-shell and (b) the open-shell path on a closed-shell molecule
/// (UHF = RHF). Bar 1e-7 Ha: Davidson/SCF convergence, not quadrature (the
/// quadrature is identical on every side).
#[test]
fn closed_shell_laplace_lanczos_energy_matches_davidson_and_open_shell() {
    let s = h2o();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &s.mol, &s.obs, s.op, &s.bounds, &RhfConfig::default()).unwrap();
    let e_lz = run(&s, &rhf, &cfg(Eigensolver::Lanczos, Some(3), 0.0)).e_rpa;
    let e_dav = run(&s, &rhf, &cfg(Eigensolver::Davidson, Some(3), 0.0)).e_rpa;
    assert!((e_lz - e_dav).abs() < 1e-7, "lz={e_lz} dav={e_dav}");

    let uhf = solve_uhf(
        &ctx,
        &s.mol,
        &s.obs,
        &s.bounds,
        &UhfConfig::default().with_max_iter(200),
    )
    .unwrap();
    let e_u = ferric_rpa::run_u_pdep_rpa(
        &s.mol,
        &s.obs,
        &s.dfbs,
        s.op,
        &uhf,
        &cfg(Eigensolver::Lanczos, Some(3), 0.0),
    )
    .unwrap()
    .e_rpa;
    assert!((e_lz - e_u).abs() < 1e-7, "closed={e_lz} open-shell={e_u}");
}
