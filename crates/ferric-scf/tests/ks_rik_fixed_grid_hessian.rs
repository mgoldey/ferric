//! Opt-in analytic closed-shell Kohn-Sham Hessian with RI-K (density-fitted
//! exchange of a global hybrid) on a grid FIXED IN SPACE: the RI-K item of
//! issue #490, the exchange extension of `ks_rij_fixed_grid_hessian.rs`.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test ks_rik_fixed_grid_hessian
//! ```
//!
//! # What is compared to what
//!
//! The energy is `E[R] = E_KS(R)` with the exchange term `−(a/4) Tr[D K_DF(D)]`
//! (`DfK`, Coulomb-metric fit), Coulomb exact or RI-J, and the XC on the grid
//! frozen at the reference geometry.
//!
//! * The fixed-grid RI-K gradient vs a central difference of that ENERGY from
//!   SCFs re-converged at displaced nuclei on the reference grid (validates the
//!   gradient the Hessian is differenced against).
//! * The analytic Hessian vs a central difference (Richardson) of that
//!   gradient with a fresh RI-K SCF at every displaced geometry, with exact J
//!   and with RI-J.
//! * Exactness anchor (`a = 0`): the RI-K machinery handed a pure functional
//!   adds nothing, so RI-J + RI-K equals the RI-J Hessian.
//! * RI-K vs exact K: the Hessian difference is the fitting error, the
//!   exact-K Hessian misses the FD of the RI-K gradient by it (negative
//!   control), and a richer auxiliary basis shrinks it toward the exact-K Hessian.
//! * Production-wired mutants (every RI-K piece, each exact-K fallback, and the
//!   `a` scaling of each fitted piece).
//! * The production SCF (route recorded, default auxiliary basis) gives the
//!   same Hessian as the injected one.
//!
//! # Hypotheses
//!
//! Physics (the RI-K Hessian is right): Hessian == FD of the RI-K gradient at
//! the FD floor (~1e-7); `H_RIK − H_exactK` is a real, fit-sized difference
//! that falls with a richer auxiliary basis. Artifact (the fitted pieces are
//! silently exact, or unscaled by `a`): `H_RIK == H_exactK`, and the FD of the
//! RI-K gradient is missed by the fitting error; or (unscaled) the `a = 0`
//! anchor fails by the full HF exchange Hessian and the FD is missed by
//! `(1 − a)` of it. Both differ from the physics prediction by more than 10x
//! the bar, so the tests distinguish them.
//!
//! # Mutation ledger
//!
//! Each row is a deliberately broken construction run through the production
//! code path (`KsFixedGridMutants`), measured against the FD of the RI-K
//! gradient (6-31G PBE0, RI-J + RI-K; production 4.2e-8, bar 5e-7, a mutant
//! must miss by >= 10x the bar):
//!
//! | mutant | miss (Ha/Bohr^2) |
//! |---|---|
//! | 2c metric second derivative dropped | 6.8e-2 |
//! | 3c-2c cross terms dropped | 2.8e2 |
//! | metric response of `K^x` dropped | 3.9e-2 |
//! | auxiliary-centre derivative in `K^x` omitted | 2.8e-2 |
//! | transposed pair in `K^x` | 1.4e-1 |
//! | erfc metric | 2.1 |
//! | exact four-centre `Γ_K` skeleton | 3.0e-5 |
//! | exact four-centre `K^x` | 2.0e-5 |
//! | exact K in the CPHF | 5.2e-6 (the closest, 10.4x) |
//! | RI-K weight `−1/4` instead of `−a/4` | 1.6 |
//! | exchange dropped from the skeleton / `F^x` / CPHF | 0.17 / 0.14 / 5.5e-2 |
//! | exact-K Hessian vs FD of the RI-K gradient (negative control) | 2.3e-5 |
//!
//! The `a = 0` anchor (`ri_k_adds_nothing_to_a_pure_functional`, measured 0)
//! has its own mutant: the RI-K weight ignoring `a` misses it by 2.9.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::hessian::{
    rks_fixed_grid_hessian, rks_fixed_grid_hessian_parts_with,
    rks_fixed_grid_hessian_parts_with_aux, KsFit, KsFits, KsFixedGridMutants,
};
use ferric_scf::hessian_df_k::RikMutants;
use ferric_scf::ks_fixed_grid::{
    rks_fixed_grid_gradient_dfjk, solve_rks_fixed_grid_dfjk, FixedGridKs,
};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const WATER: &str = "3\nH2O distorted off C2v\n\
O      0.010000    -0.020000     0.120000\n\
H      0.030000     0.770000    -0.460000\n\
H     -0.020000    -0.740000    -0.490000\n";

const AUX: &str = "def2-universal-jkfit";

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

/// Config handed to the Hessian: functional, the SCF's RI-J / RI-K bases
/// (`None` = exact).
fn hessian_config(xc: &str, j: Option<&str>, k: Option<&str>) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some(j.unwrap_or("").into()),
        df_k_aux: Some(k.unwrap_or("").into()),
        dft_grid: Some(grid_cfg()),
        ..Default::default()
    }
}

struct Pt {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    jbs: Option<PreparedBasis>,
    kbs: Option<PreparedBasis>,
    res: ScfResult,
}

fn aux_at(mol: &Molecule, name: Option<&str>) -> Option<PreparedBasis> {
    name.map(|a| PreparedBasis::new(mol, &basis::bundled(a).unwrap()).unwrap())
}

/// Fixed-grid SCF at `mol` (grid frozen at the reference geometry), Coulomb
/// fitted over `j`, exchange over `k` (None = exact).
fn scf(
    grid: &FixedGridKs,
    mol: &Molecule,
    basis_name: &str,
    j: Option<&str>,
    k: Option<&str>,
) -> Pt {
    let ctx = ParallelContext::default();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let (jbs, kbs) = (aux_at(mol, j), aux_at(mol, k));
    let cfg = RhfConfig {
        use_sad_guess: false,
        max_iter: 200,
        energy_conv: 1e-13,
        density_conv: 1e-11,
        ..Default::default()
    };
    let res = solve_rks_fixed_grid_dfjk(
        &ctx,
        mol,
        &prep,
        op,
        &bounds,
        grid,
        jbs.as_ref(),
        kbs.as_ref(),
        &cfg,
    )
    .unwrap();
    assert!(
        res.converged,
        "{basis_name}: fixed-grid SCF did not converge"
    );
    Pt {
        mol: mol.clone(),
        prep,
        bounds,
        jbs,
        kbs,
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
    j: Option<String>,
    k: Option<String>,
    grid: FixedGridKs,
    pt: Pt,
}

fn reference(basis_name: &str, xc: &str, j: Option<&str>, k: Option<&str>) -> Ref {
    let mol = water();
    let grid = FixedGridKs::new(&mol, xc, &grid_cfg()).unwrap();
    let pt = scf(&grid, &mol, basis_name, j, k);
    Ref {
        mol,
        basis: basis_name.into(),
        xc: xc.into(),
        j: j.map(String::from),
        k: k.map(String::from),
        grid,
        pt,
    }
}

fn scf_at(r: &Ref, mol: &Molecule) -> Pt {
    scf(&r.grid, mol, &r.basis, r.j.as_deref(), r.k.as_deref())
}

fn gradient_at(r: &Ref, pt: &Pt) -> Array2<f64> {
    rks_fixed_grid_gradient_dfjk(
        &pt.mol,
        &pt.prep,
        Operator::coulomb(),
        &pt.bounds,
        &r.grid,
        &pt.res,
        pt.jbs.as_ref(),
        pt.kbs.as_ref(),
    )
    .unwrap()
}

/// Central difference of the fixed-grid gradient, each from its own SCF.
fn fd_hessian_of_gradient(r: &Ref, h: f64) -> Array2<f64> {
    let n3 = 3 * r.mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    for b in 0..n3 {
        let col = rich(h, |s| {
            let pt = scf_at(r, &displaced(&r.mol, b, s));
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
            let pt = scf_at(r, &displaced(&r.mol, b, s));
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
        &hessian_config(&r.xc, r.j.as_deref(), r.k.as_deref()),
        &r.grid,
        m,
    )
    .unwrap()
    .total()
}

/// Step of the finite differences, Bohr.
const H_FD: f64 = 5e-4;
/// Gradient vs FD of the energy, Ha/Bohr.
const TOL_GRAD_FD: f64 = 3e-9;
/// Hessian vs FD of the RI-K gradient, Ha/Bohr^2 (re-converged SCFs, stencil
/// second derivatives of the 3c and 2c integrals). Measured 7.3e-11 to
/// 8.9e-11 at STO-3G and 1.4e-8 (hybrid LDA) to 4.2e-8 (PBE0) at 6-31G, where
/// the h^4 truncation of the FD of the GGA XC gradient sets the floor (B3LYP
/// at 6-31G measured 1.2e-7, which is why it runs at STO-3G here); the bar is
/// 12x the PBE0 6-31G number, and the closest production-wired mutant (exact K
/// in the CPHF, the RI-K fitting error of the response alone) misses by 5.2e-6.
const TOL_HESS_FD: f64 = 5e-7;
/// A broken construction must miss by this multiple of the bar.
const MUTANT_FACTOR: f64 = 10.0;

#[test]
fn rik_gradient_matches_central_difference_of_the_fixed_grid_energy() {
    for (basis_name, xc, j) in [
        ("sto-3g", "PBE0", Some(AUX)),
        ("6-31g", "B3LYP", Some(AUX)),
        ("6-31g", "PBE0", None),
    ] {
        let r = reference(basis_name, xc, j, Some(AUX));
        let an = gradient_at(&r, &r.pt);
        let fd = fd_gradient_of_energy(&r, H_FD);
        let diff = max_abs_diff(&an, &fd);
        eprintln!(
            "{basis_name} {xc} RI-J {:?} + RI-K: |grad - FD(E)| = {diff:.3e} (max |g| {:.3e})",
            j.is_some(),
            max_abs(&an)
        );
        assert!(diff < TOL_GRAD_FD, "{basis_name} {xc}: {diff:.3e}");
    }
}

#[test]
fn rik_hessian_matches_central_difference_of_the_df_gradient() {
    if !has_deriv2() {
        return;
    }
    for (basis_name, xc, j) in [
        ("sto-3g", "PBE0", Some(AUX)),
        ("sto-3g", "PBE0", None),
        ("sto-3g", "B3LYP", Some(AUX)),
        ("6-31g", "HYB_LDA_XC_LDA0", Some(AUX)),
        ("6-31g", "PBE0", None),
    ] {
        let r = reference(basis_name, xc, j, Some(AUX));
        let an = analytic(&r, &KsFixedGridMutants::default());
        let fd = fd_hessian_of_gradient(&r, H_FD);
        let diff = max_abs_diff(&an, &fd);
        eprintln!(
            "{basis_name} {xc} RI-J {:?} + RI-K: |H - FD(g_DF)| = {diff:.3e} (max |H| {:.3e})",
            j.is_some(),
            max_abs(&an)
        );
        assert!(diff < TOL_HESS_FD, "{basis_name} {xc}: {diff:.3e}");
    }
}

/// Exactness anchor: with `a = 0` the RI-K pieces (skeleton, `K^x`, CPHF
/// exchange) are all multiplied by zero, so the Hessian handed an RI-K fit is
/// the RI-J one. Mutant: an RI-K weight that ignores `a` (the HF `−1/4`).
#[test]
fn ri_k_adds_nothing_to_a_pure_functional() {
    if !has_deriv2() {
        return;
    }
    let r = reference("6-31g", "PBE", Some(AUX), None);
    let ctx = ParallelContext::default();
    let aux = || PreparedBasis::new(&r.mol, &basis::bundled(AUX).unwrap());
    let fit = KsFit {
        aux: &aux,
        budget_bytes: usize::MAX,
    };
    let run = |fits: KsFits<'_>, m: &KsFixedGridMutants| {
        rks_fixed_grid_hessian_parts_with_aux(
            &ctx,
            &r.mol,
            &r.pt.prep,
            Operator::coulomb(),
            &r.pt.bounds,
            &r.pt.res,
            &hessian_config("PBE", Some(AUX), None),
            &r.grid,
            fits,
            m,
        )
        .unwrap()
        .total()
    };
    let rij = run(
        KsFits {
            j: Some(&fit),
            k: None,
        },
        &KsFixedGridMutants::default(),
    );
    let both = KsFits {
        j: Some(&fit),
        k: Some(&fit),
    };
    let rijk = run(both, &KsFixedGridMutants::default());
    let diff = max_abs_diff(&rijk, &rij);
    eprintln!("a = 0: |H(RI-J + RI-K) - H(RI-J)| = {diff:.3e}");
    assert!(diff < 1e-12, "{diff:.3e}");
    let mutant = run(
        both,
        &KsFixedGridMutants {
            rik_alpha_ignores_exx: true,
            ..Default::default()
        },
    );
    let miss = max_abs_diff(&mutant, &rij);
    eprintln!("a = 0 mutant [RI-K weight ignores a]: {miss:.3e}");
    assert!(miss > 1e-3, "{miss:.3e}");
}

/// RI-K vs exact K (RI-J both): the Hessian differs by the fitting error, the
/// exact-K Hessian misses the FD of the RI-K gradient by it (negative
/// control), and a richer exchange fit brings the RI-K Hessian toward the
/// exact-K one. Measured (PBE0 / 6-31G): |dE| 1.22e-5 / 4.31e-6 / 3.33e-6 Ha
/// and max |dH| 2.26e-5 / 1.41e-6 / 1.53e-6 for def2-universal-jkfit /
/// cc-pVTZ-JKFIT / cc-pVQZ-JKFIT: the Hessian difference falls 16x and then
/// sits at the level where the energy difference also stops falling (the
/// fit's floor for this orbital basis), so the assertion is a 10x drop from
/// the jkfit difference for BOTH richer fits and a monotone energy, not a
/// monotone Hessian difference (1.41e-6 vs 1.53e-6 is inside that floor).
#[test]
fn rik_vs_exact_k_is_the_fitting_error_and_shrinks_with_the_aux_basis() {
    if !has_deriv2() {
        return;
    }
    let exact = reference("6-31g", "PBE0", Some(AUX), None);
    let h_ex = analytic(&exact, &KsFixedGridMutants::default());
    let mut prev_de: Option<f64> = None;
    let mut first_dh: Option<f64> = None;
    for k in [AUX, "cc-pvtz-jkfit", "cc-pvqz-jkfit"] {
        let df = reference("6-31g", "PBE0", Some(AUX), Some(k));
        let h_df = analytic(&df, &KsFixedGridMutants::default());
        let de = (df.pt.res.energy - exact.pt.res.energy).abs();
        let dh = max_abs_diff(&h_df, &h_ex);
        eprintln!(
            "RI-K {k}: |E_RIK - E_exactK| = {de:.3e}, max |H_RIK - H_exactK| = {dh:.3e} \
             (ratio {:.2})",
            dh / de
        );
        if k == AUX {
            let fd = fd_hessian_of_gradient(&df, H_FD);
            let miss_ex = max_abs_diff(&h_ex, &fd);
            let miss_df = max_abs_diff(&h_df, &fd);
            eprintln!("exact-K misses FD(g_RIK) by {miss_ex:.3e}; RI-K by {miss_df:.3e}");
            assert!(miss_df < TOL_HESS_FD);
            assert!(miss_ex > MUTANT_FACTOR * TOL_HESS_FD);
        }
        match first_dh {
            None => first_dh = Some(dh),
            Some(d0) => assert!(
                dh < 0.1 * d0,
                "{k}: |dH| {dh:.3e} not 10x below the jkfit {d0:.3e}"
            ),
        }
        if let Some(pe) = prev_de {
            assert!(de < pe, "{k}: |dE| {de:.3e} does not fall below {pe:.3e}");
        }
        prev_de = Some(de);
    }
}

#[test]
fn production_wired_rik_mutants_each_fail_the_hessian_anchor() {
    if !has_deriv2() {
        return;
    }
    let r = reference("6-31g", "PBE0", Some(AUX), Some(AUX));
    let fd = fd_hessian_of_gradient(&r, H_FD);
    let good = max_abs_diff(&analytic(&r, &KsFixedGridMutants::default()), &fd);
    eprintln!("production: {good:.3e}");
    assert!(good < TOL_HESS_FD, "production {good:.3e}");
    let rik = |f: fn(&mut RikMutants)| {
        let mut m = KsFixedGridMutants::default();
        f(&mut m.rik);
        m
    };
    let ks = |f: fn(&mut KsFixedGridMutants)| {
        let mut m = KsFixedGridMutants::default();
        f(&mut m);
        m
    };
    let cases: Vec<(&str, KsFixedGridMutants)> = vec![
        (
            "drop the 2c metric second derivative",
            rik(|m| m.drop_metric_second = true),
        ),
        ("drop the 3c-2c cross terms", rik(|m| m.drop_cross = true)),
        (
            "drop the metric response of K^x",
            rik(|m| m.drop_kx_metric = true),
        ),
        (
            "omit the aux-centre derivative in K^x",
            rik(|m| m.omit_aux_centre_kx = true),
        ),
        (
            "transposed pair in K^x",
            rik(|m| m.transpose_pair_kx = true),
        ),
        ("erfc metric", rik(|m| m.swap_metric = true)),
        (
            "exact four-centre K skeleton",
            ks(|m| m.exact_k_skeleton = true),
        ),
        (
            "exact four-centre K^x",
            ks(|m| m.exact_k_first_order = true),
        ),
        ("exact K in the CPHF", ks(|m| m.exact_k_response = true)),
        (
            "RI-K weight ignores a",
            ks(|m| m.rik_alpha_ignores_exx = true),
        ),
        ("no exchange in the skeleton", ks(|m| m.skeleton_exx = 0.0)),
        ("no exchange in F^x", ks(|m| m.fock_exx = 0.0)),
        ("no exchange in the CPHF", ks(|m| m.response_exx = 0.0)),
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

/// The production SCF (grid built by `solve_rhf`, RI-J + RI-K route recorded,
/// the default basis for both when the configuration leaves them unset) gives
/// the same Hessian as the injected fixed-grid RI-JK SCF; a configuration
/// naming a different exchange fit is refused.
#[test]
fn production_ri_jk_scf_result_gives_the_same_hessian() {
    if !has_deriv2() {
        return;
    }
    let r = reference("sto-3g", "PBE0", Some(AUX), Some(AUX));
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: Some("PBE0".into()),
        dft_grid: Some(grid_cfg()),
        max_iter: 200,
        energy_conv: 1e-13,
        density_conv: 1e-11,
        ..Default::default()
    };
    assert!(cfg.df_j_aux.is_none() && cfg.df_k_aux.is_none());
    let prod = solve_rhf(
        &ctx,
        &r.mol,
        &r.pt.prep,
        Operator::coulomb(),
        &r.pt.bounds,
        &cfg,
    )
    .unwrap();
    let route = prod
        .df_jk
        .as_ref()
        .expect("the hybrid SCF should have fitted");
    assert_eq!(route.k_aux.as_deref(), Some(AUX));
    let run = |c: &RhfConfig| {
        rks_fixed_grid_hessian(
            &ctx,
            &r.mol,
            &r.pt.prep,
            Operator::coulomb(),
            &r.pt.bounds,
            &prod,
            c,
        )
    };
    let from_prod = run(&cfg).unwrap();
    let from_inj = analytic(&r, &KsFixedGridMutants::default());
    let diff = max_abs_diff(&from_prod, &from_inj);
    eprintln!("RI-JK Hessian from production SCF vs injected: {diff:.3e}");
    assert!(diff < 5e-10, "{diff:.3e}");

    // A configuration stating exact K for a fitted-K SCF.
    let err = run(&RhfConfig {
        df_k_aux: Some(String::new()),
        ..cfg.clone()
    })
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("fitted K over") && err.contains("df_k_aux"),
        "{err}"
    );
    // The injected RI-K SCF presented as exact K: the rebuilt Fock matrix
    // differs by the fitting error.
    let err = rks_fixed_grid_hessian(
        &ctx,
        &r.mol,
        &r.pt.prep,
        Operator::coulomb(),
        &r.pt.bounds,
        &r.pt.res,
        &hessian_config("PBE0", Some(AUX), None),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("not a converged"), "{err}");
}
