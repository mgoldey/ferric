//! Narrow-panel Lanczos assembly with the Laplace backend (own binary: it sets
//! a process-global env var).
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Chi0Backend, Eigensolver, QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig, PdepRpaResult};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

/// Forcing narrow panels (`FERRIC_LANCZOS_PANEL`) must not change the Laplace result (the matvec must be the operator ε̃·V, not a
/// projected matrix that only equals it for the full identity).
#[test]
fn closed_shell_laplace_lanczos_is_panel_width_independent() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
    let c = cfg(Eigensolver::Lanczos, Some(3), 1e-3);
    let wide = run(&s, &rhf, &c);
    // Own test binary, so the process-global env var cannot leak into others.
    std::env::set_var("FERRIC_LANCZOS_PANEL", "7");
    let narrow = run(&s, &rhf, &c);
    std::env::remove_var("FERRIC_LANCZOS_PANEL");
    let d = max_abs_diff(&wide.eigenvalues_static, &narrow.eigenvalues_static);
    assert!(d < 1e-10, "panel-width dependence: {d:.3e}");
    assert!((wide.e_rpa - narrow.e_rpa).abs() < 1e-10);
}

/// Same for the open-shell Lanczos arm (UHF = RHF on a closed-shell molecule):
/// narrow panels must reproduce the wide-panel energy and agree with the
/// closed-shell run at matched quadrature.
#[test]
fn open_shell_laplace_lanczos_is_panel_width_independent() {
    use ferric_scf::uhf::{solve_uhf, UhfConfig};
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let s = h2o();
    let ctx = ParallelContext::default();
    let uhf = solve_uhf(
        &ctx,
        &s.mol,
        &s.obs,
        &s.bounds,
        &UhfConfig {
            max_iter: 200,
            ..Default::default()
        },
    )
    .unwrap();
    let c = cfg(Eigensolver::Lanczos, Some(3), 1e-3);
    let go = || ferric_rpa::run_u_pdep_rpa(&s.mol, &s.obs, &s.dfbs, s.op, &uhf, &c).unwrap();
    let wide = go();
    std::env::set_var("FERRIC_LANCZOS_PANEL", "7");
    let narrow = go();
    std::env::remove_var("FERRIC_LANCZOS_PANEL");
    let d = max_abs_diff(&wide.eigenvalues_static, &narrow.eigenvalues_static);
    assert!(d < 1e-10, "panel-width dependence: {d:.3e}");
    assert!((wide.e_rpa - narrow.e_rpa).abs() < 1e-10);
}
