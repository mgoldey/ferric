//! The spin-polarized XC pieces of the analytic UKS Hessian
//! (`ferric_scf::ks_fixed_grid::PolarizedKs`) and its CPKS operator, one at a
//! time, on radicals with UNEQUAL α/β densities.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test uks_xc_pieces
//! ```
//!
//! # What is compared to what
//!
//! * Closed-shell limit (exactness anchor): at `D^α = D^β = D/2` the polarized
//!   skeleton, first-order Fock and kernel equal the closed-shell
//!   `FixedGridKs` ones (libxc polarized vs unpolarized at equal spins).
//! * Frozen spin densities (no SCF): the moving skeleton vs a central
//!   difference of `ferric_dft`'s UKS XC gradient (an independent code, riding
//!   `weight1`); the moving `F^{s,x}` vs a central difference of the polarized
//!   `V_xc` on a grid rebuilt at each displaced geometry; the fixed-grid ones vs
//!   differences on the frozen grid.
//! * The kernel `V1^s[D1^α, D1^β]` vs a central difference of `V^s_xc` along an
//!   unequal two-spin density change `D1^α ≠ D1^β`.
//! * The CPKS operator `v_s = J[D1^α + D1^β] − a K[D1^s] + V1^s` vs a difference
//!   of the rebuilt UKS Fock matrices, and its singlet limit vs the closed-shell
//!   CPKS operator `J − (a/2) K + V1`.
//!
//! # Artifact hypothesis next to the physics hypothesis
//!
//! Right: every difference agrees at the stencil's truncation. A construction
//! right only for spin-symmetric perturbations (the "total-density kernel",
//! the α kernel rows reused for β, a dropped αβ cross block) PASSES the
//! closed-shell limit (shown below for the total-density kernel) and must miss
//! the unequal-spin anchors; each mutant is required to miss by >= 10 bars.

mod uks_hessian_common;

use ferric_core::parallel::ParallelContext;
use ferric_scf::ks_fixed_grid::{FixedGridKs, SpinMutants, XcGeometry, XcMutants};
use ndarray::Array2;
use uks_hessian_common::*;

/// A mutant must miss by at least this many bars.
const MUTANT_FACTOR: f64 = 10.0;

/// Closed-shell limit bar, Ha/Bohr² (libxc polarized vs unpolarized at
/// equal spins, rounding).
const TOL_CLOSED_LIMIT: f64 = 1e-10;
/// Moving skeleton vs FD of the `ferric_dft` UKS XC gradient, Ha/Bohr².
const TOL_SKELETON: f64 = 1e-9;
/// `F^{s,x}` vs FD of the rebuilt polarized `V_xc`, Ha/Bohr.
const TOL_FOCK: f64 = 1e-9;
/// Kernel vs FD of `V_xc` along `D1`, Ha.
const TOL_KERNEL: f64 = 1e-9;

/// A symmetric, deterministic, O(0.1) density change; `seed` makes α and β differ.
fn d1(n: usize, seed: usize) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let v = (((3 * i + 5 * j + 7 * seed) % 11) as f64 - 5.0) * 0.02;
            m[(i, j)] = v;
            m[(j, i)] = v;
        }
    }
    m
}

fn worst(a: &[Array2<f64>], b: &[Array2<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| max_abs_diff(x, y))
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// Closed-shell limit
// ---------------------------------------------------------------------------

/// At `D^α = D^β = D/2` (and `D1^α = D1^β = D1/2`) the polarized pieces equal
/// the closed-shell ones. The total-density-kernel mutant passes this anchor
/// too: it cannot tell a construction right only for spin-symmetric
/// perturbations from the right one.
#[test]
fn polarized_pieces_reduce_to_closed_shell_at_equal_spins() {
    let mol = water();
    for xc in ["LDA", "PBE", "B3LYP"] {
        let bs = ferric_core::basis::bundled("6-31g").unwrap();
        let prep = ferric_integrals::basis_bridge::PreparedBasis::new(&mol, &bs).unwrap();
        let n = prep.nbasis();
        // A physical closed-shell density: the occupied projector of the
        // overlap's leading vectors is overkill; any symmetric D works here,
        // but keep it positive by squaring.
        let x = d1(n, 3);
        let d = x.dot(&x.t()) * 0.5;
        let h = 0.5 * &d;
        let grid = FixedGridKs::new(&mol, xc, &grid_cfg()).unwrap();
        let pol = grid.polarized().unwrap();
        let geo = XcGeometry::new(&mol, &prep).unwrap();
        let m = XcMutants::default();
        let skel_c = grid.skeleton_hessian_moving(&geo, &d, &m).unwrap();
        let skel_p = pol.skeleton_hessian_moving(&geo, [&h, &h], &m).unwrap();
        let fock_c = grid.first_order_fock_moving(&geo, &d, &m).unwrap();
        let fock_p = pol.first_order_fock_moving(&geo, [&h, &h], &m).unwrap();
        let pert = d1(n, 1);
        let half = 0.5 * &pert;
        let v1_c = grid.response(&geo, &d, 1.0, 1.0).unwrap().apply(&pert);
        let v1_p = pol
            .response(&geo, [&h, &h], 1.0, &m)
            .unwrap()
            .apply([&half, &half]);
        let total = XcMutants {
            spin: SpinMutants {
                total_kernel: true,
                ..SpinMutants::default()
            },
            ..XcMutants::default()
        };
        let v1_t = pol
            .response(&geo, [&h, &h], 1.0, &total)
            .unwrap()
            .apply([&half, &half]);
        let e = [
            max_abs_diff(&skel_c, &skel_p),
            worst(&fock_c, &fock_p[0]).max(worst(&fock_c, &fock_p[1])),
            max_abs_diff(&v1_c, &v1_p[0]).max(max_abs_diff(&v1_c, &v1_p[1])),
            max_abs_diff(&v1_c, &v1_t[0]),
        ];
        eprintln!(
            "{xc}: closed-shell limit skeleton {:.2e} (max |H| {:.2e}), F^x {:.2e}, V1 {:.2e}; \
             total-density-kernel mutant V1 {:.2e}",
            e[0],
            max_abs(&skel_c),
            e[1],
            e[2],
            e[3]
        );
        for (what, v) in ["skeleton", "F^x", "V1", "total-kernel V1"].iter().zip(e) {
            assert!(v < TOL_CLOSED_LIMIT, "{xc} {what}: {v:.3e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Frozen spin densities
// ---------------------------------------------------------------------------

fn moving_misses(p: &Pt, xc: &str, r: &FrozenRef, m: &XcMutants) -> (f64, f64) {
    let grid = FixedGridKs::new(&p.mol, xc, &p.grid).unwrap();
    let pol = grid.polarized().unwrap();
    let geo = XcGeometry::new(&p.mol, &p.prep).unwrap();
    let (da, db) = spin_densities(p);
    let skel = pol.skeleton_hessian_moving(&geo, [&da, &db], m).unwrap();
    let fock = pol.first_order_fock_moving(&geo, [&da, &db], m).unwrap();
    let f = worst(&fock[0], &r.fock[0]).max(worst(&fock[1], &r.fock[1]));
    (max_abs_diff(&skel, &r.skel), f)
}

/// Moving skeleton and `F^{s,x}` vs the frozen-density differences: NH2
/// (STO-3G, 6-31G) and O2 triplet (STO-3G), LDA and PBE.
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn polarized_moving_pieces_match_frozen_density_differences() {
    let cases = [(nh2(), "sto-3g"), (nh2(), "6-31g"), (o2(), "sto-3g")];
    for (mol, basis_name) in &cases {
        for xc in ["LDA", "PBE"] {
            let p = scf(mol, basis_name, &config(xc));
            let (da, db) = spin_densities(&p);
            let r = frozen_reference(&p, xc);
            let (skel, fock) = moving_misses(&p, xc, &r, &XcMutants::default());
            let grid = FixedGridKs::new(&p.mol, xc, &p.grid).unwrap();
            let geo = XcGeometry::new(&p.mol, &p.prep).unwrap();
            let fixed = grid
                .polarized()
                .unwrap()
                .skeleton_hessian(&geo, [&da, &db], &XcMutants::default())
                .unwrap();
            eprintln!(
                "{xc} {basis_name} {}: |Da-Db| {:.2e}; moving skeleton vs FD {skel:.3e} (max |H| \
                 {:.2e}), moving F^x vs FD {fock:.3e}; fixed-grid skeleton vs the same FD {:.3e}",
                p.mol.atoms.len(),
                max_abs_diff(&da, &db),
                max_abs(&r.skel),
                max_abs_diff(&fixed, &r.skel)
            );
            assert!(
                skel < TOL_SKELETON,
                "{xc} {basis_name}: skeleton {skel:.3e}"
            );
            assert!(fock < TOL_FOCK, "{xc} {basis_name}: F^x {fock:.3e}");
            assert!(max_abs_diff(&fixed, &r.skel) > MUTANT_FACTOR * TOL_SKELETON);
        }
    }
}

/// The fixed-grid pieces vs differences on the FROZEN grid (points and
/// weights of the reference geometry): the skeleton vs a difference of the
/// polarized fixed-grid XC gradient, `F^{s,x}` vs a difference of `V_xc`.
#[test]
fn polarized_fixed_pieces_match_frozen_grid_differences() {
    let p = scf(&nh2(), "sto-3g", &config("PBE"));
    let (da, db) = spin_densities(&p);
    let built = FixedGridKs::new(&p.mol, "PBE", &p.grid).unwrap();
    let (pts, w) = built.points_and_weights();
    let frozen = FixedGridKs::from_points("PBE", pts.to_vec(), w.to_vec()).unwrap();
    let pol = frozen.polarized().unwrap();
    let n3 = 3 * p.mol.atoms.len();
    let at = |b: usize, s: f64| {
        let m = displaced(&p.mol, b, s);
        let prep = ferric_integrals::basis_bridge::PreparedBasis::new(&m, &p.bs).unwrap();
        let geo = XcGeometry::new(&m, &prep).unwrap();
        (
            pol.gradient(&geo, [&da, &db]).unwrap(),
            pol.energy_vxc(&geo, [&da, &db]).unwrap().1,
        )
    };
    let mut skel_fd = Array2::<f64>::zeros((n3, n3));
    let mut fock_fd: [Vec<Array2<f64>>; 2] = [Vec::new(), Vec::new()];
    for b in 0..n3 {
        let col = rich(H_FD, |s| flat(&at(b, s).0));
        skel_fd.column_mut(b).assign(&col.column(0));
        for (sp, f) in fock_fd.iter_mut().enumerate() {
            f.push(rich(H_FD, |s| at(b, s).1[sp].clone()));
        }
    }
    let geo = XcGeometry::new(&p.mol, &p.prep).unwrap();
    let m = XcMutants::default();
    let skel = pol.skeleton_hessian(&geo, [&da, &db], &m).unwrap();
    let fock = pol.first_order_fock(&geo, [&da, &db], &m).unwrap();
    let es = max_abs_diff(&skel, &skel_fd);
    let ef = worst(&fock[0], &fock_fd[0]).max(worst(&fock[1], &fock_fd[1]));
    eprintln!(
        "fixed grid: skeleton vs FD {es:.3e} (max |H| {:.2e}), F^x vs FD {ef:.3e}",
        max_abs(&skel)
    );
    assert!(es < TOL_SKELETON, "skeleton {es:.3e}");
    assert!(ef < TOL_FOCK, "F^x {ef:.3e}");
}

// ---------------------------------------------------------------------------
// Kernel and CPKS operator
// ---------------------------------------------------------------------------

/// `V1^s[D1^α, D1^β]` vs `(V^s(D + h D1) − V^s(D − h D1)) / 2h` (Richardson)
/// for `D1^α ≠ D1^β`, worst over both spins.
fn kernel_miss(p: &Pt, xc: &str, m: &XcMutants) -> (f64, f64) {
    let (da, db) = spin_densities(p);
    let n = da.nrows();
    let (pa, pb) = (d1(n, 1), d1(n, 4));
    let grid = FixedGridKs::new(&p.mol, xc, &p.grid).unwrap();
    let pol = grid.polarized().unwrap();
    let geo = XcGeometry::new(&p.mol, &p.prep).unwrap();
    let v1 = pol
        .response(&geo, [&da, &db], 1.0, m)
        .unwrap()
        .apply([&pa, &pb]);
    let mut miss = 0.0f64;
    for (sp, v) in v1.iter().enumerate() {
        let fd = rich(1e-3, |s| {
            let a = &da + &(s * &pa);
            let b = &db + &(s * &pb);
            pol.energy_vxc(&geo, [&a, &b]).unwrap().1[sp].clone()
        });
        miss = miss.max(max_abs_diff(v, &fd));
    }
    (miss, v1.iter().map(max_abs).fold(0.0, f64::max))
}

#[test]
fn polarized_kernel_matches_density_differences_for_unequal_spin_changes() {
    for xc in ["LDA", "PBE", "B3LYP"] {
        let p = scf(&nh2(), "6-31g", &config(xc));
        let (miss, scale) = kernel_miss(&p, xc, &XcMutants::default());
        eprintln!("{xc}: kernel vs FD {miss:.3e} (max |V1| {scale:.2e})");
        assert!(miss < TOL_KERNEL, "{xc}: {miss:.3e}");
    }
}

/// The UKS Fock matrices `h + J[D] − a K[D^s] + V^s_xc` rebuilt at `D`.
fn uks_focks(p: &Pt, xc: &str, d: [&Array2<f64>; 2]) -> [Array2<f64>; 2] {
    let ctx = ParallelContext::default();
    let n = d[0].nrows();
    let jk = |m: &Array2<f64>| {
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        ferric_scf::rhf::build_jk(&ctx, &p.prep, &p.bounds, 1e-14, m, &mut j, &mut k).unwrap();
        (j, k)
    };
    let grid = FixedGridKs::new(&p.mol, xc, &p.grid).unwrap();
    let a = grid.exact_exchange();
    let geo = XcGeometry::new(&p.mol, &p.prep).unwrap();
    let v = grid.polarized().unwrap().energy_vxc(&geo, d).unwrap().1;
    let h = ferric_integrals::oneelectron::hcore_ecp_with_external(
        &p.prep,
        &p.mol,
        p.prep.basis_set(),
        None,
    )
    .unwrap();
    let j = jk(&(d[0] + d[1])).0;
    [0, 1].map(|s| &h + &j + &v[s] - &(a * &jk(d[s]).1))
}

/// The CPKS operator the UKS Hessian uses vs a difference of the rebuilt UKS
/// Fock matrices along an unequal two-spin density change, and the rebuilt
/// Focks vs the SCF's own (the Fock-consistency check of the Hessian); then
/// its singlet limit vs the closed-shell CPKS operator `J − (a/2)K + V1`.
#[test]
fn uks_cpks_operator_matches_fock_differences_and_the_closed_shell_limit() {
    let ctx = ParallelContext::default();
    for xc in ["PBE", "B3LYP"] {
        let p = scf(&nh2(), "6-31g", &config(xc));
        let (da, db) = spin_densities(&p);
        let n = da.nrows();
        let grid = FixedGridKs::new(&p.mol, xc, &p.grid).unwrap();
        let residual =
            ferric_scf::hessian::uks_fock_residual(&ctx, &p.mol, &p.prep, &p.bounds, &p.res, &grid)
                .unwrap();
        let (pa, pb) = (d1(n, 2), d1(n, 5));
        let g = ferric_scf::hessian::uks_response_operator(
            &ctx,
            &p.mol,
            &p.prep,
            &p.bounds,
            &p.res,
            &grid,
            &XcMutants::default(),
            [&pa, &pb],
        )
        .unwrap();
        let mut miss = 0.0f64;
        for (sp, gs) in g.iter().enumerate() {
            let fd = rich(1e-3, |s| {
                let a = &da + &(s * &pa);
                let b = &db + &(s * &pb);
                uks_focks(&p, xc, [&a, &b])[sp].clone()
            });
            miss = miss.max(max_abs_diff(gs, &fd));
        }
        eprintln!(
            "{xc}: Fock residual {residual:.2e}; CPKS operator vs FD of the UKS Fock {miss:.3e}"
        );
        assert!(residual < 1e-8, "{xc}: Fock residual {residual:.3e}");
        assert!(miss < TOL_KERNEL, "{xc}: {miss:.3e}");
    }
    // Singlet limit: closed-shell water with a UKS SCF.
    for xc in ["PBE", "B3LYP"] {
        let p = scf(&water(), "6-31g", &config(xc));
        let (da, db) = spin_densities(&p);
        assert!(
            max_abs_diff(&da, &db) < 1e-8,
            "UKS singlet broke spin symmetry"
        );
        let n = da.nrows();
        let pert = d1(n, 3);
        let half = 0.5 * &pert;
        let grid = FixedGridKs::new(&p.mol, xc, &p.grid).unwrap();
        let g = ferric_scf::hessian::uks_response_operator(
            &ctx,
            &p.mol,
            &p.prep,
            &p.bounds,
            &p.res,
            &grid,
            &XcMutants::default(),
            [&half, &half],
        )
        .unwrap();
        let d = &da + &db;
        let geo = XcGeometry::new(&p.mol, &p.prep).unwrap();
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        ferric_scf::rhf::build_jk(&ctx, &p.prep, &p.bounds, 1e-14, &pert, &mut j, &mut k).unwrap();
        let closed = j - &(0.5 * grid.exact_exchange() * &k)
            + grid.response(&geo, &d, 1.0, 1.0).unwrap().apply(&pert);
        let e = max_abs_diff(&g[0], &closed).max(max_abs_diff(&g[1], &closed));
        eprintln!("{xc}: singlet UKS CPKS operator vs closed-shell {e:.3e}");
        assert!(e < 1e-9, "{xc}: {e:.3e}");
    }
}

// ---------------------------------------------------------------------------
// Mutation ledger
// ---------------------------------------------------------------------------

/// Each spin-polarized and moving-grid mutant misses an unequal-spin anchor
/// (frozen-density pieces or the kernel) by >= 10 bars. NH2 / 6-31G, PBE.
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn polarized_mutants_each_fail_an_unequal_spin_anchor() {
    let xc = "PBE";
    let p = scf(&nh2(), "6-31g", &config(xc));
    let r = frozen_reference(&p, xc);
    let spin = |f: fn(&mut SpinMutants)| {
        let mut m = XcMutants::default();
        f(&mut m.spin);
        m
    };
    let mut cases: Vec<(&str, XcMutants)> = vec![
        (
            "drop the alpha-beta kernel block",
            spin(|m| m.cross_kernel = 0.0),
        ),
        ("total-density kernel", spin(|m| m.total_kernel = true)),
        (
            "alpha kernel rows for beta",
            spin(|m| m.alpha_rows_for_beta = true),
        ),
    ];
    let mut w = XcMutants::default();
    w.moving.weight_hess = 0.0;
    cases.push(("drop W''", w));
    let mut w = XcMutants::default();
    w.moving.weight_cross = 0.0;
    cases.push(("drop the W' f' cross terms", w));
    let mut w = XcMutants::default();
    w.moving.fock_weight = 0.0;
    cases.push(("drop the weight term of F^x", w));
    cases.push((
        "GGA treated as LDA",
        XcMutants {
            gradient_terms: 0.0,
            ..XcMutants::default()
        },
    ));
    let mut weakest = f64::INFINITY;
    for (name, m) in cases {
        let (skel, fock) = moving_misses(&p, xc, &r, &m);
        let (kern, _) = kernel_miss(&p, xc, &m);
        let bars = (skel / TOL_SKELETON)
            .max(fock / TOL_FOCK)
            .max(kern / TOL_KERNEL);
        eprintln!(
            "mutant [{name}]: skeleton {skel:.3e}, F^x {fock:.3e}, kernel {kern:.3e} ({bars:.1e} bars)"
        );
        weakest = weakest.min(bars);
        assert!(
            bars >= MUTANT_FACTOR,
            "mutant [{name}] misses by only {bars:.1} bars"
        );
    }
    eprintln!("weakest mutant: {weakest:.1e} bars");
}
