//! Classify CASSCF stationary points of N2 CAS(10,10)/cc-pVDZ (2 core
//! orbitals, `testdata/molecules/n2.xyz`).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-ci \
//!     --example casscf_stationary_points -- <bfgs|ah> <grad_tol> [follow <scale> <max>] [swap <i>:<j> ...]
//! ```
//!
//! From the exact-J/K RHF orbitals (with the listed MO columns swapped
//! first) the CASSCF is converged to `grad_tol` and canonicalized (natural
//! active orbitals). At the result the CI-relaxed orbital Hessian is built
//! by central differences of the analytic gradient (`h = 1e-4`, every
//! nonredundant pair, symmetrized) and its lowest eigenvalues printed, with
//! the natural occupations and the character of each active natural orbital:
//! its `|m|` about the bond axis (the dominant Fourier component of the
//! orbital on circles about `z`) and its parity under inversion through the
//! bond midpoint. With `follow <scale> <max>`, while the lowest eigenvalue
//! is below `-1e-4` (eigenvalues of size `1e-6` are the zero modes and the
//! finite-difference error) the point is displaced by `scale` along that
//! eigenvector, converged and classified again, at most `max` times.

use ferric_ci::casscf::{optimize, CasScfConfig, CasScfOptimizer, CasScfProblem, CasScfResult};
use ferric_ci::driver::{rhf_start, ScfStart};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};
use std::path::PathBuf;

const H: f64 = 1e-4;

fn fd_hessian(problem: &CasScfProblem, c: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = problem.pairs().len();
    let mut hes = Array2::<f64>::zeros((n, n));
    for k in 0..n {
        let mut kv = vec![0.0; n];
        kv[k] = H;
        let gp = problem
            .evaluate(&problem.rotate(c, &kv).unwrap())
            .unwrap()
            .gradient;
        kv[k] = -H;
        let gm = problem
            .evaluate(&problem.rotate(c, &kv).unwrap())
            .unwrap()
            .gradient;
        for r in 0..n {
            hes[(r, k)] = (gp[r] - gm[r]) / (2.0 * H);
        }
    }
    let sym = 0.5 * (&hes + &hes.t());
    sym.eigh(UPLO::Upper).unwrap()
}

/// `|m|` about `z` and inversion parity (`g`/`u`) of the orbital `col`.
fn character(mol: &Molecule, prep: &PreparedBasis, col: ndarray::ArrayView1<'_, f64>) -> String {
    let zmid = 0.5 * mol.atoms.iter().map(|a| a.zpos).sum::<f64>();
    let nth = 16usize;
    let mut points = Vec::new();
    for &rho in &[0.6, 1.2] {
        for &dz in &[0.4, 1.1] {
            for t in 0..nth {
                let th = 2.0 * std::f64::consts::PI * (t as f64 + 0.37) / nth as f64;
                points.push([rho * th.cos(), rho * th.sin(), zmid + dz]);
            }
        }
    }
    let mirror: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [-p[0], -p[1], 2.0 * zmid - p[2]])
        .collect();
    let chi = ferric_integrals::ao_grid::eval_basis_on_points(mol, prep.basis_set(), &points)
        .expect("AO values");
    let chim = ferric_integrals::ao_grid::eval_basis_on_points(mol, prep.basis_set(), &mirror)
        .expect("AO values");
    let phi = col.dot(&chi);
    let phim = col.dot(&chim);
    let mut power = [0.0f64; 5];
    for ring in 0..4 {
        let v = &phi.as_slice().unwrap()[ring * nth..(ring + 1) * nth];
        for (m, pw) in power.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0, 0.0);
            for (t, &x) in v.iter().enumerate() {
                let th = 2.0 * std::f64::consts::PI * (t as f64 + 0.37) / nth as f64;
                re += x * (m as f64 * th).cos();
                im += x * (m as f64 * th).sin();
            }
            *pw += re * re + im * im;
        }
    }
    let total: f64 = power.iter().sum();
    let m = (0..5)
        .max_by(|&a, &b| power[a].partial_cmp(&power[b]).unwrap())
        .unwrap();
    let parity = phi.dot(&phim) / phi.dot(&phi);
    let name = ["σ", "π", "δ", "φ", "γ"][m];
    format!(
        "{name}{} (|m| weight {:.3}, parity {parity:+.3})",
        if parity > 0.0 { "g" } else { "u" },
        power[m] / total
    )
}

fn report(
    tag: &str,
    mol: &Molecule,
    prep: &PreparedBasis,
    problem: &CasScfProblem,
    r: &CasScfResult,
) -> Array2<f64> {
    println!(
        "[{tag}] E {:.10} |g| {:.2e} iterations {} converged {}",
        r.e_total, r.grad_norm, r.iterations, r.converged
    );
    let (occ, _) = r.rdms.dm1.eigh(UPLO::Upper).unwrap();
    let mut occ: Vec<f64> = occ.to_vec();
    occ.sort_by(|a, b| b.partial_cmp(a).unwrap());
    println!(
        "[{tag}] natural occupations {}",
        occ.iter()
            .map(|x| format!("{x:.4}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let (nc, na) = (2usize, 10usize);
    for t in 0..na {
        let col = r.mo_coeff.column(nc + t);
        println!(
            "[{tag}]   active {t}: occupation {:.4}  {}",
            r.rdms.dm1[(t, t)],
            character(mol, prep, col)
        );
    }
    let (ev, vecs) = fd_hessian(problem, &r.mo_coeff);
    println!(
        "[{tag}] lowest CI-relaxed orbital Hessian eigenvalues {}",
        ev.iter()
            .take(5)
            .map(|x| format!("{x:.5}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let v0 = vecs.column(0);
    let mut idx: Vec<usize> = (0..v0.len()).collect();
    idx.sort_by(|&a, &b| v0[b].abs().partial_cmp(&v0[a].abs()).unwrap());
    println!(
        "[{tag}] lowest mode, largest pairs: {}",
        idx.iter()
            .take(4)
            .map(|&k| format!("{:?} {:+.3}", problem.pairs()[k], v0[k]))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut out = Array2::<f64>::zeros((v0.len(), 2));
    out.column_mut(0).assign(&v0);
    out[(0, 1)] = ev[0];
    out
}

fn main() {
    ferric_core::blas_threads::init_threading();
    let a: Vec<String> = std::env::args().collect();
    let optimizer = CasScfOptimizer::parse_config_str(&a[1]).expect("optimizer");
    let grad_tol: f64 = a[2].parse().expect("grad_tol");
    let mut follow = None;
    let mut swaps = Vec::new();
    let mut i = 3;
    while i < a.len() {
        match a[i].as_str() {
            "follow" => {
                follow = Some((
                    a[i + 1].parse::<f64>().expect("scale"),
                    a[i + 2].parse::<usize>().expect("max"),
                ));
                i += 3;
            }
            "swap" => {
                let (p, q) = a[i + 1].split_once(':').expect("i:j");
                swaps.push((p.parse::<usize>().unwrap(), q.parse::<usize>().unwrap()));
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/molecules/n2.xyz");
    let mol = Molecule::load_xyz(path.to_str().unwrap()).unwrap();
    let bs = ferric_core::basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let rhf = rhf_start(
        &ParallelContext::default(),
        &mol,
        &prep,
        ScfStart::default(),
        None,
    )
    .unwrap();
    let cfg = CasScfConfig {
        n_core: 2,
        n_active: 10,
        n_elec_active: (5, 5),
        grad_tol,
        max_iter: 1000,
        optimizer,
        canonicalize: true,
        ..Default::default()
    };
    let mut c0 = rhf.mos_r().clone();
    for &(p, q) in &swaps {
        let cp = c0.column(p).to_owned();
        let cq = c0.column(q).to_owned();
        c0.column_mut(p).assign(&cq);
        c0.column_mut(q).assign(&cp);
    }
    let problem = CasScfProblem::new(&mol, &prep, c0.ncols(), &cfg).unwrap();
    let r = optimize(&problem, c0, &cfg).unwrap();
    let mut low = report("start", &mol, &prep, &problem, &r);
    let mut cur = r;
    if let Some((scale, max)) = follow {
        for step in 1..=max {
            if low[(0, 1)] >= -1e-4 {
                println!(
                    "lowest eigenvalue {:.2e} >= -1e-4: a local minimum",
                    low[(0, 1)]
                );
                break;
            }
            let kv: Vec<f64> = low.column(0).iter().map(|x| scale * x).collect();
            let c = problem.rotate(&cur.mo_coeff, &kv).unwrap();
            cur = optimize(&problem, c, &cfg).unwrap();
            low = report(&format!("follow {step}"), &mol, &prep, &problem, &cur);
        }
    }
}
