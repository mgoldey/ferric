//! The analytic ECP Hessian on the libecpint backend (issue #406).
//!
//! The default (and validated) ECP backend is ferric's `ecp_quad`; libecpint
//! (`FERRIC_ECP_BACKEND=libecpint`) is the cross-check. The skeleton stencil
//! differentiates whichever backend's first-derivative kernel the gradient
//! uses, so the same FD-of-the-analytic-gradient anchor applies per backend.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test ecp_hessian_libecpint
//! ```
//!
//! ONE test function: the backend is a process-wide environment variable read
//! on every ECP call, so a second concurrent test would race on it.
//!
//! * libecpint: analytic Hessian vs central FD of the libecpint analytic
//!   gradient (columns of a non-ECP atom and of the ECP atom), and the
//!   no-skeleton control (must miss by `MUST_MISS` x bar);
//! * libecpint vs quadrature analytic Hessians on the same system: bounded by
//!   the two backends' integral agreement (the gradient-level agreement of
//!   the two backends is `ecp_deriv_block.rs`);
//! * PySCF's analytic ECP Hessian on this backend: run the ignored
//!   `validation_ecp_hessian.rs` with `FERRIC_ECP_BACKEND=libecpint`.
//!
//! Artifact hypothesis: if the stencil is run on libecpint's first
//! derivatives with step `h` and libecpint's first derivatives carry integral
//! noise `e`, the stencil error is ~ `e / h`. FD of the libecpint gradient
//! carries the same noise divided by `H_FD`, so the two see the SAME defect
//! and the anchor cannot detect a libecpint first-derivative error itself;
//! that is what the PySCF run and the quadrature comparison are for.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::hessian::rhf_hessian_parts;
use ferric_scf::hessian_ecp::ecp_skeleton_hessian;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const TOL_FD: f64 = 2e-6;
const MUST_MISS: f64 = 10.0;
const H_FD: f64 = 1e-3;
/// libecpint vs quadrature Hessians (Ha/Bohr²; measured 1.4e-8, so ~70x
/// margin): both are ~1e-7 level kernels
/// and the stencil divides their disagreement by `h`.
const TOL_BACKENDS: f64 = 1e-6;

const HOI: &str = "3\nHOI tilted\nH 0.30 -0.40 0.20\nO 0.00 0.10 0.95\nI 0.45 0.20 2.80\n";

fn config() -> RhfConfig {
    RhfConfig {
        max_iter: 300,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        ..Default::default()
    }
}

struct Run {
    hessian: Array2<f64>,
    skeleton: Array2<f64>,
    fd: Array2<f64>,
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

fn run(cols: &[usize]) -> Run {
    let bs = basis::bundled("def2-svp").unwrap();
    let mut mol = Molecule::parse_xyz(HOI, 0, 1).unwrap();
    mol.apply_ecp(&bs);
    let (ctx, op, cfg) = (ParallelContext::default(), Operator::coulomb(), config());
    let solve = |m: &Molecule| {
        let prep = PreparedBasis::new(m, &bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let scf = solve_rhf(&ctx, m, &prep, op, &bounds, &cfg).unwrap();
        assert!(scf.converged);
        (prep, bounds, scf)
    };
    let (prep, bounds, scf) = solve(&mol);
    let parts = rhf_hessian_parts(&ctx, &mol, &prep, op, &bounds, &scf, &cfg).unwrap();
    let skeleton = ecp_skeleton_hessian(&mol, &bs, scf.density_r()).unwrap();
    let n3 = 3 * mol.atoms.len();
    let mut fd = Array2::<f64>::zeros((n3, n3));
    for &b in cols {
        let g = |h: f64| {
            let m = displaced(&mol, b / 3, b % 3, h);
            let (p, bd, s) = solve(&m);
            rhf_gradient(&m, &p, op, &bd, &s, None).unwrap()
        };
        let (gp, gm) = (g(H_FD), g(-H_FD));
        for a in 0..n3 {
            fd[(a, b)] = (gp[(a / 3, a % 3)] - gm[(a / 3, a % 3)]) / (2.0 * H_FD);
        }
    }
    Run {
        hessian: parts.total(),
        skeleton,
        fd,
    }
}

fn cols_diff(a: &Array2<f64>, b: &Array2<f64>, cols: &[usize]) -> f64 {
    cols.iter()
        .flat_map(|&c| (0..a.nrows()).map(move |r| (r, c)))
        .fold(0.0f64, |m, rc| {
            let (x, y) = (a[rc], b[rc]);
            assert!(x.is_finite() && y.is_finite());
            m.max((x - y).abs())
        })
}

#[test]
fn libecpint_backend_hessian_matches_fd_and_the_quadrature_backend() {
    if ferric_integrals::engine::libint_max_deriv_order() < 2 {
        return;
    }
    let cols = [4usize, 8];
    std::env::set_var("FERRIC_ECP_BACKEND", "libecpint");
    assert_eq!(
        ferric_integrals::ecp::ecp_backend().unwrap(),
        ferric_integrals::ecp::EcpBackend::Libecpint
    );
    let lib = run(&cols);
    std::env::set_var("FERRIC_ECP_BACKEND", "quadrature");
    let quad = run(&cols);
    std::env::remove_var("FERRIC_ECP_BACKEND");

    let d_lib = cols_diff(&lib.hessian, &lib.fd, &cols);
    let d_quad = cols_diff(&quad.hessian, &quad.fd, &cols);
    eprintln!("analytic vs FD: libecpint {d_lib:.3e}, quadrature {d_quad:.3e} (bar {TOL_FD:.0e})");
    assert!(d_quad < TOL_FD, "quadrature control: {d_quad:.3e}");
    assert!(d_lib < TOL_FD, "libecpint: {d_lib:.3e}");

    let miss = cols_diff(&(&lib.hessian - &lib.skeleton), &lib.fd, &cols);
    eprintln!("libecpint, skeleton removed: misses FD by {miss:.3e}");
    assert!(miss > MUST_MISS * TOL_FD);

    let across: f64 = lib
        .hessian
        .iter()
        .zip(&quad.hessian)
        .fold(0.0, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("libecpint vs quadrature analytic Hessian: max|d| = {across:.3e}");
    assert!(across < TOL_BACKENDS, "{across:.3e}");
}
