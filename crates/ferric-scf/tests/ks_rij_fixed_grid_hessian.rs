//! Opt-in analytic closed-shell Kohn-Sham Hessian with RI-J (density-fitted
//! Coulomb) on a grid FIXED IN SPACE: unit 3 of issue #407, the DF extension of
//! `ks_fixed_grid_hessian.rs`.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test ks_rij_fixed_grid_hessian
//! ```
//!
//! # What is compared to what
//!
//! The energy is `E[R] = E_KS(R)` with the Coulomb term `1/2 Tr[D J_DF(D)]`
//! (`DfJ`, Coulomb-metric fit over `def2-universal-jkfit`), exact `a`-scaled K
//! for a hybrid, and the XC on the grid frozen at the reference geometry.
//!
//! * The fixed-grid DF-J gradient vs a central difference of that ENERGY from
//!   SCFs re-converged at displaced nuclei on the reference grid (validates the
//!   gradient the Hessian is differenced against).
//! * The analytic Hessian vs a central difference (Richardson) of that
//!   gradient with a fresh DF-J SCF at every displaced geometry.
//! * The exact-J Hessian vs the same difference: the NEGATIVE CONTROL that
//!   proves the fitting error is visible at the bar.
//! * DF-J vs exact-J: the Hessian difference is the fitting error, tied to the
//!   energy-level DF error and shrinking with a richer auxiliary basis.
//! * Production-wired mutants (every RI-J piece, and each exact-J fallback).
//!
//! # Hypotheses
//!
//! Physics (RI-J Hessian is right): Hessian == FD of the DF-J gradient at the
//! FD floor (~1e-7..1e-6); `|H_DF - H_exact|` is a small multiple of
//! `|E_DF - E_exact|` (measured ratio ~3.7 for water/6-31G/PBE in PySCF) and
//! falls with a richer auxiliary basis. Artifact (the DF terms are silently
//! exact): `H_DF == H_exact` exactly, and the FD of the DF gradient is missed
//! by the fitting error. These differ by more than 10x the bar, so the test
//! distinguishes them.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::hessian::{
    rks_fixed_grid_hessian, rks_fixed_grid_hessian_parts_with, KsFixedGridMutants,
};
use ferric_scf::hessian_df::RijMutants;
use ferric_scf::ks_fixed_grid::{rks_fixed_grid_gradient_df, solve_rks_fixed_grid_df, FixedGridKs};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const WATER: &str = "3\nH2O distorted off C2v\n\
O      0.010000    -0.020000     0.120000\n\
H      0.030000     0.770000    -0.460000\n\
H     -0.020000    -0.740000    -0.490000\n";

const AUX: &str = "def2-universal-jkfit";
/// A coarser fit (the MP2 RI basis used as a Coulomb fit), for the convergence leg.
const AUX_COARSE: &str = "cc-pvdz-ri";

fn grid_cfg() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 50,
        n_angular: 110,
        prune: None,
    }
}

fn has_deriv2() -> bool {
    ferric_integrals::engine::libint_max_deriv_order() >= 2
}

fn water() -> Molecule {
    Molecule::parse_xyz(WATER, 0, 1).unwrap()
}

/// Config handed to the Hessian: functional, the SCF's RI-J basis (`None` =
/// exact J), exact K.
fn hessian_config(xc: &str, aux: Option<&str>) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some(aux.unwrap_or("").into()),
        df_k_aux: Some(String::new()),
        dft_grid: Some(grid_cfg()),
        ..Default::default()
    }
}

struct Pt {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    dfbs: Option<PreparedBasis>,
    res: ScfResult,
}

/// Fixed-grid SCF at `mol` (grid frozen at the reference geometry), Coulomb
/// fitted over `aux` (None = exact).
fn scf(grid: &FixedGridKs, mol: &Molecule, basis_name: &str, aux: Option<&str>) -> Pt {
    let ctx = ParallelContext::default();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let dfbs = aux.map(|a| PreparedBasis::new(mol, &basis::bundled(a).unwrap()).unwrap());
    let cfg = RhfConfig {
        use_sad_guess: false,
        max_iter: 200,
        energy_conv: 1e-13,
        density_conv: 1e-11,
        ..Default::default()
    };
    let res =
        solve_rks_fixed_grid_df(&ctx, mol, &prep, op, &bounds, grid, dfbs.as_ref(), &cfg).unwrap();
    assert!(
        res.converged,
        "{basis_name}: fixed-grid SCF did not converge"
    );
    Pt {
        mol: mol.clone(),
        prep,
        bounds,
        dfbs,
        res,
    }
}

fn displaced(mol: &Molecule, coord: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match coord % 3 {
        0 => m.atoms[coord / 3].x += h,
        1 => m.atoms[coord / 3].y += h,
        _ => m.atoms[coord / 3].zpos += h,
    }
    m
}

fn rich(h: f64, mut f: impl FnMut(f64) -> Array2<f64>) -> Array2<f64> {
    let mut d = |h: f64| (f(h) - f(-h)) / (2.0 * h);
    let coarse = d(h);
    let fine = d(0.5 * h);
    (4.0 * fine - coarse) / 3.0
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    max_abs(&(a - b))
}

fn flat(g: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_vec((g.len(), 1), g.iter().copied().collect()).unwrap()
}

struct Ref {
    mol: Molecule,
    basis: String,
    xc: String,
    aux: Option<String>,
    grid: FixedGridKs,
    pt: Pt,
}

fn reference(basis_name: &str, xc: &str, aux: Option<&str>) -> Ref {
    let mol = water();
    let grid = FixedGridKs::new(&mol, xc, &grid_cfg()).unwrap();
    let pt = scf(&grid, &mol, basis_name, aux);
    Ref {
        mol,
        basis: basis_name.into(),
        xc: xc.into(),
        aux: aux.map(String::from),
        grid,
        pt,
    }
}

fn gradient_at(r: &Ref, pt: &Pt) -> Array2<f64> {
    rks_fixed_grid_gradient_df(
        &pt.mol,
        &pt.prep,
        Operator::coulomb(),
        &pt.bounds,
        &r.grid,
        &pt.res,
        pt.dfbs.as_ref(),
    )
    .unwrap()
}

/// Central difference of the fixed-grid gradient, each from its own SCF.
fn fd_hessian_of_gradient(r: &Ref, h: f64) -> Array2<f64> {
    let n3 = 3 * r.mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    for b in 0..n3 {
        let col = rich(h, |s| {
            let pt = scf(
                &r.grid,
                &displaced(&r.mol, b, s),
                &r.basis,
                r.aux.as_deref(),
            );
            flat(&gradient_at(r, &pt))
        });
        out.column_mut(b).assign(&col.column(0));
    }
    0.5 * (&out + &out.t())
}

fn fd_gradient_of_energy(r: &Ref, h: f64) -> Array2<f64> {
    let n3 = 3 * r.mol.atoms.len();
    let mut out = Array2::<f64>::zeros((r.mol.atoms.len(), 3));
    for b in 0..n3 {
        let e = rich(h, |s| {
            let pt = scf(
                &r.grid,
                &displaced(&r.mol, b, s),
                &r.basis,
                r.aux.as_deref(),
            );
            Array2::from_elem((1, 1), pt.res.energy)
        });
        out[(b / 3, b % 3)] = e[(0, 0)];
    }
    out
}

fn analytic(r: &Ref, m: &KsFixedGridMutants) -> Array2<f64> {
    rks_fixed_grid_hessian_parts_with(
        &ParallelContext::default(),
        &r.mol,
        &r.pt.prep,
        Operator::coulomb(),
        &r.pt.bounds,
        &r.pt.res,
        &hessian_config(&r.xc, r.aux.as_deref()),
        &r.grid,
        m,
    )
    .unwrap()
    .total()
}

/// Step of the finite differences, Bohr.
const H_FD: f64 = 5e-4;
/// Gradient vs FD of the energy, Ha/Bohr.
const TOL_GRAD_FD: f64 = 1e-7;
/// Hessian vs FD of the DF-J gradient, Ha/Bohr^2 (re-converged SCFs, stencil
/// second derivatives of the 3c integrals).
const TOL_HESS_FD: f64 = 2e-7;
/// A broken construction must miss by this multiple of the bar.
const MUTANT_FACTOR: f64 = 10.0;

#[test]
fn rij_gradient_matches_central_difference_of_the_fixed_grid_energy() {
    for (basis_name, xc) in [("sto-3g", "PBE"), ("6-31g", "PBE0")] {
        let r = reference(basis_name, xc, Some(AUX));
        let an = gradient_at(&r, &r.pt);
        let fd = fd_gradient_of_energy(&r, H_FD);
        let diff = max_abs_diff(&an, &fd);
        eprintln!(
            "{basis_name} {xc} RI-J: |grad - FD(E)| = {diff:.3e} (max |g| {:.3e})",
            max_abs(&an)
        );
        assert!(diff < TOL_GRAD_FD, "{basis_name} {xc}: {diff:.3e}");
    }
}

#[test]
fn rij_hessian_matches_central_difference_of_the_df_gradient() {
    if !has_deriv2() {
        return;
    }
    for (basis_name, xc) in [("sto-3g", "PBE"), ("sto-3g", "PBE0"), ("6-31g", "LDA")] {
        let r = reference(basis_name, xc, Some(AUX));
        let an = analytic(&r, &KsFixedGridMutants::default());
        let fd = fd_hessian_of_gradient(&r, H_FD);
        let diff = max_abs_diff(&an, &fd);
        eprintln!(
            "{basis_name} {xc} RI-J: |H - FD(g_DF)| = {diff:.3e} (max |H| {:.3e})",
            max_abs(&an)
        );
        assert!(diff < TOL_HESS_FD, "{basis_name} {xc}: {diff:.3e}");
    }
}

/// DF-J vs exact-J: the Hessian differs by the fitting error (a few times the
/// energy-level error), the exact-J Hessian misses the DF difference by that
/// amount (negative control), and a richer auxiliary basis shrinks both (def2-universal-jkfit vs the cc-pVDZ-RI functions).
#[test]
fn rij_vs_exact_j_is_the_fitting_error_and_shrinks_with_the_aux_basis() {
    if !has_deriv2() {
        return;
    }
    let exact = reference("6-31g", "PBE", None);
    let h_ex = analytic(&exact, &KsFixedGridMutants::default());
    let mut prev: Option<(f64, f64)> = None;
    for aux in [AUX_COARSE, AUX] {
        let df = reference("6-31g", "PBE", Some(aux));
        let h_df = analytic(&df, &KsFixedGridMutants::default());
        let de = (df.pt.res.energy - exact.pt.res.energy).abs();
        let dh = max_abs_diff(&h_df, &h_ex);
        eprintln!(
            "{aux}: |E_DF - E_exact| = {de:.3e}, max |H_DF - H_exact| = {dh:.3e} (ratio {:.2})",
            dh / de
        );
        if aux == AUX {
            // Tied to the energy-level fitting error from both sides (measured
            // ratio 3.70; PySCF's DF-J Hessian gives 3.69 on the same grid).
            assert!(
                dh < 30.0 * de && dh > 0.3 * de,
                "{aux}: {dh:.3e} vs {de:.3e}"
            );
            // Negative control: the exact-J Hessian misses the FD of the DF
            // gradient by the fitting error.
            let fd = fd_hessian_of_gradient(&df, H_FD);
            let miss_ex = max_abs_diff(&h_ex, &fd);
            let miss_df = max_abs_diff(&h_df, &fd);
            eprintln!("exact-J misses FD(g_DF) by {miss_ex:.3e}; RI-J by {miss_df:.3e}");
            assert!(miss_df < TOL_HESS_FD);
            assert!(miss_ex > MUTANT_FACTOR * TOL_HESS_FD);
        }
        if let Some((pe, ph)) = prev {
            assert!(
                de < 0.1 * pe && dh < 0.7 * ph,
                "{aux} does not shrink the error"
            );
        }
        prev = Some((de, dh));
    }
}

#[test]
fn production_wired_rij_mutants_each_fail_the_hessian_anchor() {
    if !has_deriv2() {
        return;
    }
    let r = reference("6-31g", "PBE", Some(AUX));
    let fd = fd_hessian_of_gradient(&r, H_FD);
    let good = max_abs_diff(&analytic(&r, &KsFixedGridMutants::default()), &fd);
    assert!(good < TOL_HESS_FD, "production {good:.3e}");
    let rij = |f: fn(&mut RijMutants)| {
        let mut m = KsFixedGridMutants::default();
        f(&mut m.rij);
        m
    };
    let cases: Vec<(&str, KsFixedGridMutants)> = vec![
        (
            "drop the 2c metric second derivative",
            rij(|m| m.drop_metric_second = true),
        ),
        ("drop the 3c-2c cross terms", rij(|m| m.drop_cross = true)),
        (
            "drop the aux-fit part of J^x",
            rij(|m| m.omit_aux_jx = true),
        ),
        ("erfc metric", rij(|m| m.swap_metric = true)),
        (
            "exact four-centre J skeleton",
            KsFixedGridMutants {
                exact_j_skeleton: true,
                ..Default::default()
            },
        ),
        (
            "exact four-centre J^x",
            KsFixedGridMutants {
                exact_j_first_order: true,
                ..Default::default()
            },
        ),
        (
            "exact J in the CPHF",
            KsFixedGridMutants {
                exact_j_response: true,
                ..Default::default()
            },
        ),
    ];
    for (name, m) in cases {
        let miss = max_abs_diff(&analytic(&r, &m), &fd);
        eprintln!(
            "mutant [{name}]: {miss:.3e} ({:.1e} x bar)",
            miss / TOL_HESS_FD
        );
        assert!(
            miss >= MUTANT_FACTOR * TOL_HESS_FD,
            "mutant [{name}] misses by only {miss:.3e}"
        );
    }
}

/// The production SCF (grid built by `solve_rhf`, RI-J route recorded) gives
/// the same Hessian as the injected fixed-grid DF-J SCF.
#[test]
fn production_ri_j_scf_result_gives_the_same_hessian() {
    if !has_deriv2() {
        return;
    }
    let r = reference("6-31g", "PBE", Some(AUX));
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        max_iter: 200,
        energy_conv: 1e-13,
        density_conv: 1e-11,
        ..hessian_config("PBE", Some(AUX))
    };
    let prod = solve_rhf(
        &ctx,
        &r.mol,
        &r.pt.prep,
        Operator::coulomb(),
        &r.pt.bounds,
        &cfg,
    )
    .unwrap();
    let from_prod = rks_fixed_grid_hessian(
        &ctx,
        &r.mol,
        &r.pt.prep,
        Operator::coulomb(),
        &r.pt.bounds,
        &prod,
        &cfg,
    )
    .unwrap();
    let from_inj = analytic(&r, &KsFixedGridMutants::default());
    let diff = max_abs_diff(&from_prod, &from_inj);
    eprintln!("RI-J Hessian from production SCF vs injected: {diff:.3e}");
    assert!(diff < 5e-10, "{diff:.3e}");

    // A route naming another basis than the configuration is refused.
    let err = rks_fixed_grid_hessian(
        &ctx,
        &r.mol,
        &r.pt.prep,
        Operator::coulomb(),
        &r.pt.bounds,
        &prod,
        &hessian_config("PBE", None),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("RI-J basis"), "{err}");
}

#[test]
fn ri_k_and_inconsistent_fits_are_refused() {
    if !has_deriv2() {
        return;
    }
    let r = reference("sto-3g", "PBE0", Some(AUX));
    let ctx = ParallelContext::default();
    let run = |cfg: &RhfConfig, res: &ScfResult| {
        rks_fixed_grid_hessian(
            &ctx,
            &r.mol,
            &r.pt.prep,
            Operator::coulomb(),
            &r.pt.bounds,
            res,
            cfg,
        )
        .unwrap_err()
        .to_string()
    };
    // RI-K named in the configuration.
    let msg = run(
        &RhfConfig {
            df_k_aux: Some(AUX.into()),
            ..hessian_config("PBE0", Some(AUX))
        },
        &r.pt.res,
    );
    assert!(msg.contains("RI-K") && msg.contains("issue #407"), "{msg}");
    // A production SCF that fitted K.
    let prod = solve_rhf(
        &ctx,
        &r.mol,
        &r.pt.prep,
        Operator::coulomb(),
        &r.pt.bounds,
        &RhfConfig {
            df_k_aux: Some(AUX.into()),
            ..hessian_config("PBE0", Some(AUX))
        },
    )
    .unwrap();
    let msg = run(
        &RhfConfig {
            df_k_aux: Some(AUX.into()),
            ..hessian_config("PBE0", Some(AUX))
        },
        &prod,
    );
    assert!(msg.contains("RI-K") && msg.contains("issue #407"), "{msg}");
    // A fitted SCF presented as exact J, and an exact SCF presented as fitted:
    // the Fock matrix rebuilt with the stated Coulomb matrix does not match.
    let msg = run(&hessian_config("PBE0", None), &r.pt.res);
    assert!(msg.contains("not a converged"), "{msg}");
    let exact = reference("sto-3g", "PBE0", None);
    let msg = run(&hessian_config("PBE0", Some(AUX)), &exact.pt.res);
    assert!(msg.contains("not a converged"), "{msg}");
}
