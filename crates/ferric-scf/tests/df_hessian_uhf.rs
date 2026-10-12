//! Analytic UHF Hessian for density-fitted Coulomb and/or exchange (RI-J,
//! RI-K, RI-JK; `ferric_scf::hessian_df`, `ferric_scf::hessian_df_k`,
//! `ferric_scf::hessian::uhf_hessian`) against INDEPENDENT constructions.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test df_hessian_uhf
//! ```
//!
//! # What is compared to what
//!
//! * EXACTNESS ANCHOR (closed-shell limit): a converged closed-shell RHF
//!   result re-labelled as UHF (`D_α = D_β = D/2`) must give the merged RHF
//!   RI-J / RI-K / RI-JK Hessian term by term. BLIND to everything that needs
//!   `D_α ≠ D_β` (a swapped spin, one spin's density used for both): the
//!   open-shell FD tests and the mutants carry those.
//! * Channel additivity: the multi-channel RI-K terms equal the sum of the
//!   one-channel terms (the RHF-validated code path), and the weights of two
//!   channels with one `X` equal one channel with the summed weight.
//! * `K^x` of each spin channel vs a central difference of the SCF's own `DfK`
//!   applied to that spin's density.
//! * The skeleton pieces (3c, 2c, cross) summed over spin channels, and the
//!   whole J + K skeleton, vs central differences at FIXED per-spin densities
//!   of the first-derivative pieces and of the validated `df_jk_gradient`.
//! * The TOTAL Hessian vs a central difference (Richardson) of ferric's own
//!   analytic UHF DF gradient with the SCF re-converged (seeded from the
//!   reference orbitals, basin-checked) at every displaced geometry.
//! * Translational and rotational invariance, negative controls and mutants.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * Right: every piece matches its finite difference at the FD floor; the
//!   exact-integral, RI-J-only and RI-K-only Hessians miss the RI-JK reference
//!   by far more than the bar.
//! * Spin-channel defect (α/β swapped, the closed-shell weight, one spin's
//!   density for the Coulomb part): the closed-shell anchor PASSES (the two
//!   spins coincide there) and the open-shell tests and mutants fail.
//! * FD reference broken (a displaced SCF on another state, a metric mode
//!   crossing the DF-K cut): the full test fails column-wise while the
//!   analytic-piece tests stay green; the systems here have a clean metric gap
//!   (`clean_gap`) and the SCF states are basin-checked.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ferric_scf::df_gradient::{df_jk_gradient, DfKChannel};
use ferric_scf::df_k::DfK;
use ferric_scf::fock::KBuilder;
use ferric_scf::gradient::uhf_gradient;
use ferric_scf::hessian::rij_support::{
    rik_channel_terms, rik_first, uhf_hessian_parts_mutated, RijMutants, RikChannel, RikMutants,
    UhfDfMutants, RIK_ALPHA_SPIN,
};
use ferric_scf::hessian::{
    rhf_hessian_parts, uhf_hessian_parts, RhfHessianParts, UhfHessianParts,
};
use ferric_scf::result::{ScfResult, Spin};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess};
use ndarray::{s, Array2};
use ndarray_linalg::{Eigh, UPLO};

/// CH2 triplet distorted off C2v.
const CH2_DISTORTED: &str = "3\nCH2 triplet distorted off C2v\n\
C      0.000000     0.000000     0.000000\n\
H      0.990000     0.020000     0.420000\n\
H     -0.970000    -0.030000     0.440000\n";

/// NH2 doublet distorted off C2v.
const NH2_DISTORTED: &str = "3\nNH2 doublet distorted off C2v\n\
N      0.000000     0.000000     0.000000\n\
H      0.010000     0.810000    -0.630000\n\
H     -0.020000    -0.790000    -0.640000\n";

/// CH3 doublet, pyramidalised and distorted.
const CH3_DISTORTED: &str = "4\nCH3 doublet distorted\n\
C      0.000000     0.000000     0.000000\n\
H      1.080000     0.020000     0.100000\n\
H     -0.530000     0.930000     0.090000\n\
H     -0.560000    -0.920000     0.110000\n";

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

fn has_deriv1() -> bool {
    let order = ferric_integrals::engine::libint_max_deriv_order();
    if order < 1 {
        eprintln!("SKIP: libint2 generated without first derivatives");
    }
    order >= 1
}

fn nh2() -> Molecule {
    Molecule::parse_xyz(NH2_DISTORTED, 0, 2).unwrap()
}

fn ch2() -> Molecule {
    Molecule::parse_xyz(CH2_DISTORTED, 0, 3).unwrap()
}

fn ch3() -> Molecule {
    Molecule::parse_xyz(CH3_DISTORTED, 0, 2).unwrap()
}

fn water() -> Molecule {
    Molecule::parse_xyz(WATER_DISTORTED, 0, 1).unwrap()
}

/// SCF with RI-J in `j_aux` and RI-K in `k_aux` ("" = exact).
fn scf_config(j_aux: &str, k_aux: &str) -> RhfConfig {
    RhfConfig {
        max_iter: 300,
        energy_conv: 1e-14,
        density_conv: 1e-12,
        df_j_aux: Some(j_aux.to_string()),
        df_k_aux: Some(k_aux.to_string()),
        ..Default::default()
    }
}

struct Setup {
    mol: Molecule,
    basis_name: String,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    uhf: ScfResult,
    cfg: RhfConfig,
    j_aux: String,
    k_aux: String,
}

fn prep_at(m: &Molecule, basis_name: &str) -> PreparedBasis {
    PreparedBasis::new(m, &basis::bundled(basis_name).unwrap()).unwrap()
}

fn aux_at(m: &Molecule, aux: &str) -> PreparedBasis {
    PreparedBasis::new(m, &basis::bundled(aux).unwrap()).unwrap()
}

fn setup(mol: &Molecule, basis_name: &str, j_aux: &str, k_aux: &str) -> Setup {
    let prep = prep_at(mol, basis_name);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = scf_config(j_aux, k_aux);
    let uhf = solve_uhf(&ParallelContext::default(), mol, &prep, &bounds, &cfg).unwrap();
    assert!(uhf.converged, "{basis_name}: UHF did not converge");
    if j_aux.is_empty() && k_aux.is_empty() {
        assert!(uhf.df_jk.is_none(), "exact SCF carries no route");
    } else {
        let route = uhf.df_jk.as_ref().expect("a DF route");
        assert_eq!(route.j_aux.as_deref(), Some(j_aux).filter(|v| !v.is_empty()));
        assert_eq!(route.k_aux.as_deref(), Some(k_aux).filter(|v| !v.is_empty()));
    }
    Setup {
        mol: mol.clone(),
        basis_name: basis_name.to_string(),
        prep,
        bounds,
        uhf,
        cfg,
        j_aux: j_aux.to_string(),
        k_aux: k_aux.to_string(),
    }
}

fn parts(s: &Setup) -> UhfHessianParts {
    uhf_hessian_parts(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &s.uhf,
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
    mut grad: impl FnMut(&Molecule) -> Array2<f64>,
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

fn fd_hessian_richardson(
    mol: &Molecule,
    h: f64,
    mut grad: impl FnMut(&Molecule) -> Array2<f64>,
) -> Array2<f64> {
    let a = fd_hessian(mol, h, &mut grad);
    let b = fd_hessian(mol, h / 2.0, &mut grad);
    (4.0 * b - a) / 3.0
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

/// Spin occupation counts of the molecule.
fn nocc(mol: &Molecule) -> [usize; 2] {
    let nelec = mol.nelec() as usize;
    let unpaired = mol.multiplicity - 1;
    [(nelec + unpaired) / 2, (nelec - unpaired) / 2]
}

/// `[X_α, X_β] = [C_α,occ, C_β,occ]`.
fn occ_factors(s: &Setup) -> [Array2<f64>; 2] {
    let n = nocc(&s.mol);
    [
        s.uhf.mos_alpha.slice(s![.., ..n[0]]).to_owned(),
        s.uhf
            .mos_beta
            .as_ref()
            .unwrap()
            .slice(s![.., ..n[1]])
            .to_owned(),
    ]
}

fn spin_densities(r: &ScfResult) -> [Array2<f64>; 2] {
    [
        r.density_alpha.clone(),
        r.density_beta.clone().expect("UHF β density"),
    ]
}

/// Smallest and largest metric eigenvalue of the fit at this geometry, and
/// whether any lies in the band the gradient treats as dropped although the
/// energy keeps it (`[1e-10, 1e-7)`).
fn clean_gap(mol: &Molecule, aux: &str) -> (f64, f64, bool) {
    let v = coulomb_metric_2c(Operator::coulomb(), &aux_at(mol, aux)).unwrap();
    let (lam, _) = v.eigh(UPLO::Upper).unwrap();
    let band = lam.iter().any(|&l| (1e-10..1e-7).contains(&l));
    (lam[0], lam[lam.len() - 1], !band)
}

const H_PIECE: f64 = 1e-4;
/// Step for finite differences of the fit coefficients `Γ = V⁻¹ L`.
const H_C: f64 = 3e-4;
/// Basin check on displaced SCFs (Ha).
const BASIN_TOL: f64 = 1e-3;

// ---------------------------------------------------------------------------
// Exactness anchor: closed-shell UHF == RHF, RI-J / RI-K / RI-JK
// ---------------------------------------------------------------------------

/// A converged RHF result presented as a UHF one with identical α/β orbitals.
fn rhf_as_uhf(rhf: &ScfResult) -> ScfResult {
    let mut u = rhf.clone();
    let half = 0.5 * rhf.density_r();
    u.spin = Spin::Unrestricted;
    u.density_alpha = half.clone();
    u.density_beta = Some(half);
    u.mos_beta = Some(rhf.mos_r().clone());
    u.eps_beta = Some(rhf.eps_r().to_vec());
    u.fock_beta = Some(rhf.fock_r().clone());
    u
}

/// Closed-shell limit bar (per term and total). Measured (release, water
/// 6-31G / cc-pVDZ-RI): RI-J 5.6e-17, RI-K 2.9e-13, RI-JK 1.2e-13 (worst term;
/// the K and JK differences are the summation order of the two channels against
/// one). The bar is 350x the worst; a spin-channel defect moves the open-shell
/// Hessian by >= 5e-2 (see the mutants), which this anchor cannot see.
const TOL_ANCHOR: f64 = 1e-10;

fn closed_shell_anchor(basis_name: &str, j_aux: &str, k_aux: &str) {
    if !has_deriv2() {
        return;
    }
    let mol = water();
    let prep = prep_at(&mol, basis_name);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = scf_config(j_aux, k_aux);
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged);
    let r = rhf_hessian_parts(&ctx, &mol, &prep, op, &bounds, &rhf, &cfg).unwrap();
    let u = uhf_hessian_parts(&ctx, &mol, &prep, op, &bounds, &rhf_as_uhf(&rhf), &cfg).unwrap();
    let tag = format!("closed-shell anchor {basis_name} J={j_aux:?} K={k_aux:?}");
    assert!(max_abs(&r.response) > 1e-3, "vacuous: RHF response ~ 0");
    assert!(max_abs(&r.two_electron) > 1e-3, "vacuous: 2e skeleton ~ 0");
    let mut worst = 0.0f64;
    for (what, got, want) in [
        ("nuclear", &u.nuclear, &r.nuclear),
        ("one_electron", &u.one_electron, &r.one_electron),
        ("overlap", &u.overlap, &r.overlap),
        ("two_electron", &u.two_electron, &r.two_electron),
        ("response", &u.response, &r.response),
    ] {
        worst = worst.max(assert_close(&format!("{tag} {what}"), got, want, TOL_ANCHOR));
    }
    assert_close(
        &format!("{tag} total"),
        &u.total(),
        &r.total(),
        TOL_ANCHOR,
    );
    eprintln!("{tag}: worst term {worst:.3e}");
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

#[test]
fn closed_shell_solve_uhf_equals_rhf_ri_jk() {
    // A real UHF SCF on the closed-shell molecule converges back to RHF: the
    // SCF-side plumbing (β fields, per-spin DfK) is covered as well.
    if !has_deriv2() {
        return;
    }
    let mol = water();
    let prep = prep_at(&mol, "sto-3g");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = scf_config("def2-universal-jkfit", "def2-universal-jkfit");
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let uhf = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
    assert!((uhf.energy - rhf.energy).abs() < 1e-9);
    let hr = rhf_hessian_parts(&ctx, &mol, &prep, op, &bounds, &rhf, &cfg)
        .unwrap()
        .total();
    let hu = uhf_hessian_parts(&ctx, &mol, &prep, op, &bounds, &uhf, &cfg)
        .unwrap()
        .total();
    assert_close("solve_uhf closed-shell RI-JK anchor", &hu, &hr, 1e-9);
}

// ---------------------------------------------------------------------------
// Channel additivity (independent of the open-shell wiring)
// ---------------------------------------------------------------------------

#[test]
fn multi_channel_terms_are_the_sum_of_one_channel_terms() {
    // Hypothesis: the multi-channel accumulation (summed 3c weights, summed
    // 2c weight, summed cross terms, per-channel K^x) equals the sum over
    // separate one-channel runs, which is the RHF-validated path with the
    // exchange weight scaled (every RI-K piece is linear in alpha at fixed X).
    // Artifact: an accumulator bug (second channel overwriting the first,
    // alpha of channel 0 used for both) would break this identity. BLIND to
    // anything the one-channel path itself gets wrong.
    if !has_deriv1() {
        return;
    }
    let s = setup(&ch2(), "sto-3g", "", "def2-universal-jkfit");
    let route = s.uhf.df_jk.as_ref().unwrap();
    let [xa, xb] = occ_factors(&s);
    assert_ne!(xa.ncols(), xb.ncols(), "triplet: unequal channels");
    let mu = RikMutants::default();
    let both = rik_channel_terms(
        &s.mol,
        &s.prep,
        route,
        &[
            RikChannel { x: &xa, alpha: -0.5 },
            RikChannel { x: &xb, alpha: -0.3 },
        ],
        &mu,
    )
    .unwrap();
    let one = |x: &Array2<f64>, alpha: f64| {
        rik_channel_terms(&s.mol, &s.prep, route, &[RikChannel { x, alpha }], &mu).unwrap()
    };
    let (ta, tb) = (one(&xa, -0.5), one(&xb, -0.3));
    for (what, got, want) in [
        ("3c", &both.three_centre, &(&ta.three_centre + &tb.three_centre)),
        ("2c", &both.metric_second, &(&ta.metric_second + &tb.metric_second)),
        ("cross", &both.cross, &(&ta.cross + &tb.cross)),
        ("skeleton", &both.skeleton, &(&ta.skeleton + &tb.skeleton)),
    ] {
        assert!(max_abs(want) > 1e-3, "vacuous {what}");
        assert_close(&format!("additivity {what}"), got, want, 1e-12);
    }
    for (c, t) in [(0, &ta), (1, &tb)] {
        for x in 0..9 {
            let d = max_abs_diff(
                &both.channels[c].first_order_fock[x],
                &t.channels[0].first_order_fock[x],
            );
            assert!(d < 1e-13, "channel {c} K^x {x}: {d:.3e}");
        }
    }
    // Two channels with the SAME X add their weights.
    let twice = rik_channel_terms(
        &s.mol,
        &s.prep,
        route,
        &[
            RikChannel { x: &xa, alpha: -0.2 },
            RikChannel { x: &xa, alpha: -0.3 },
        ],
        &mu,
    )
    .unwrap();
    let once = one(&xa, -0.5);
    assert_close("same-X weights add", &twice.skeleton, &once.skeleton, 1e-12);
}

// ---------------------------------------------------------------------------
// K^x of each spin channel vs the SCF's own DfK
// ---------------------------------------------------------------------------

fn check_first_order_k(mol: &Molecule, basis_name: &str, aux: &str) {
    if !has_deriv1() {
        return;
    }
    let s = setup(mol, basis_name, "", aux);
    let route = s.uhf.df_jk.as_ref().unwrap();
    let xs = occ_factors(&s);
    let dens = spin_densities(&s.uhf);
    let terms = rik_channel_terms(
        &s.mol,
        &s.prep,
        route,
        &[
            RikChannel {
                x: &xs[0],
                alpha: RIK_ALPHA_SPIN,
            },
            RikChannel {
                x: &xs[1],
                alpha: RIK_ALPHA_SPIN,
            },
        ],
        &RikMutants::default(),
    )
    .unwrap();
    let n3 = 3 * s.mol.atoms.len();
    for (sp, d) in dens.iter().enumerate() {
        let k_at = |m: &Molecule| {
            let prep = prep_at(m, basis_name);
            let mut dfk =
                DfK::new(Operator::coulomb(), &prep, &aux_at(m, aux), usize::MAX).unwrap();
            let mut k = Array2::<f64>::zeros(d.dim());
            dfk.build(d, &mut k).unwrap();
            k
        };
        for xi in 0..n3 {
            let kp = k_at(&displaced(&s.mol, xi / 3, xi % 3, H_PIECE));
            let km = k_at(&displaced(&s.mol, xi / 3, xi % 3, -H_PIECE));
            let fd = (kp - km) / (2.0 * H_PIECE);
            assert_close(
                &format!("{basis_name}/{aux}: spin {sp} K^x, x = {xi}"),
                &terms.channels[sp].first_order_k[xi],
                &fd,
                5e-7,
            );
            assert!(max_abs(&fd) > 1e-3, "vacuous: K^x ~ 0 at x = {xi}");
        }
    }
}

#[test]
fn first_order_k_per_spin_matches_dfk_fd_ch2_sto3g() {
    check_first_order_k(&ch2(), "sto-3g", "def2-universal-jkfit");
}

#[test]
fn first_order_k_per_spin_matches_dfk_fd_nh2_631g() {
    check_first_order_k(&nh2(), "6-31g", "cc-pvdz-ri");
}

// ---------------------------------------------------------------------------
// Skeleton pieces vs fixed-density finite differences
// ---------------------------------------------------------------------------

fn check_skeleton_pieces(mol: &Molecule, basis_name: &str, aux: &str) {
    if !has_deriv1() {
        return;
    }
    let s = setup(mol, basis_name, "", aux);
    let route = s.uhf.df_jk.as_ref().unwrap();
    let xs = occ_factors(&s);
    let dens = spin_densities(&s.uhf);
    let alpha = RIK_ALPHA_SPIN;
    let terms = rik_channel_terms(
        &s.mol,
        &s.prep,
        route,
        &[
            RikChannel { x: &xs[0], alpha },
            RikChannel { x: &xs[1], alpha },
        ],
        &RikMutants::default(),
    )
    .unwrap();
    let n3 = 3 * s.mol.atoms.len();
    let f0: Vec<_> = xs
        .iter()
        .map(|x| rik_first(&s.mol, &s.prep, route, x, None).unwrap())
        .collect();
    let first_at = |m: &Molecule, ch: usize, fixed: Option<&Array2<f64>>| {
        rik_first(m, &prep_at(m, basis_name), route, &xs[ch], fixed).unwrap()
    };
    let mut three = Array2::<f64>::zeros((n3, n3));
    let mut metric = Array2::<f64>::zeros((n3, n3));
    let mut cross = Array2::<f64>::zeros((n3, n3));
    for y in 0..n3 {
        let (mp, mm) = (
            displaced(&s.mol, y / 3, y % 3, H_PIECE),
            displaced(&s.mol, y / 3, y % 3, -H_PIECE),
        );
        for ch in 0..2 {
            let g0 = &f0[ch].gamma;
            let (fp, fm) = (first_at(&mp, ch, Some(g0)), first_at(&mm, ch, Some(g0)));
            let (cp, cm) = (
                first_at(&displaced(&s.mol, y / 3, y % 3, H_C), ch, None),
                first_at(&displaced(&s.mol, y / 3, y % 3, -H_C), ch, None),
            );
            let dgamma = (&cp.gamma - &cm.gamma) / (2.0 * H_C);
            for xi in 0..n3 {
                three[(xi, y)] += 2.0 * alpha * (fp.lg[xi] - fm.lg[xi]) / (2.0 * H_PIECE);
                metric[(xi, y)] += -alpha * (fp.gvg[xi] - fm.gvg[xi]) / (2.0 * H_PIECE);
                cross[(xi, y)] += 2.0
                    * alpha
                    * f0[ch]
                        .z
                        .row(xi)
                        .iter()
                        .zip(dgamma.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
            }
        }
    }
    let tol = 2e-6;
    let tag = format!("{basis_name}/{aux}");
    assert_close(&format!("{tag}: 3c piece"), &terms.three_centre, &three, tol);
    assert_close(&format!("{tag}: 2c piece"), &terms.metric_second, &metric, tol);
    assert_close(&format!("{tag}: cross piece"), &terms.cross, &cross, tol);
    for (name, m) in [("3c", &three), ("2c", &metric), ("cross", &cross)] {
        assert!(max_abs(m) > 1e-3, "vacuous: {name} piece ~ 0");
    }
    // The whole K skeleton vs d/dy of the validated DF gradient at fixed
    // per-spin densities.
    let natoms = s.mol.atoms.len();
    let fd = fd_hessian(&s.mol, H_PIECE, |m| {
        df_jk_gradient(
            natoms,
            &prep_at(m, basis_name),
            &aux_at(m, aux),
            Operator::coulomb(),
            None,
            &[
                DfKChannel {
                    density: &dens[0],
                    alpha,
                },
                DfKChannel {
                    density: &dens[1],
                    alpha,
                },
            ],
            route.budget_bytes,
        )
        .unwrap()
    });
    assert_close(&format!("{tag}: K skeleton"), &terms.skeleton, &fd, tol);
    assert!(
        terms.skeleton_asymmetry < 1e-9,
        "raw skeleton asymmetry {:.3e}",
        terms.skeleton_asymmetry
    );
}

#[test]
fn skeleton_pieces_match_fd_ch2_sto3g() {
    check_skeleton_pieces(&ch2(), "sto-3g", "def2-universal-jkfit");
}

#[test]
fn skeleton_pieces_match_fd_nh2_631g() {
    check_skeleton_pieces(&nh2(), "6-31g", "cc-pvdz-ri");
}

fn check_jk_skeleton(mol: &Molecule, basis_name: &str, aux: &str) {
    // RI-J on the TOTAL density + RI-K per spin: the whole two-electron
    // skeleton of the production Hessian (no four-centre term remains) vs
    // d/dy of the DF gradient at fixed densities.
    if !has_deriv2() {
        return;
    }
    let s = setup(mol, basis_name, aux, aux);
    let route = s.uhf.df_jk.as_ref().unwrap();
    let dens = spin_densities(&s.uhf);
    let total = &dens[0] + &dens[1];
    let natoms = s.mol.atoms.len();
    let fd = fd_hessian(&s.mol, H_PIECE, |m| {
        df_jk_gradient(
            natoms,
            &prep_at(m, basis_name),
            &aux_at(m, aux),
            Operator::coulomb(),
            Some(&total),
            &[
                DfKChannel {
                    density: &dens[0],
                    alpha: RIK_ALPHA_SPIN,
                },
                DfKChannel {
                    density: &dens[1],
                    alpha: RIK_ALPHA_SPIN,
                },
            ],
            route.budget_bytes,
        )
        .unwrap()
    });
    let p = parts(&s);
    assert_close(
        &format!("{basis_name}/{aux}: J + K two-electron skeleton"),
        &p.two_electron,
        &fd,
        2e-6,
    );
}

#[test]
fn jk_skeleton_matches_fd_ch2_sto3g() {
    check_jk_skeleton(&ch2(), "sto-3g", "def2-universal-jkfit");
}

#[test]
fn jk_skeleton_matches_fd_nh2_631g() {
    check_jk_skeleton(&nh2(), "6-31g", "cc-pvdz-ri");
}

// ---------------------------------------------------------------------------
// Total vs FD of ferric's own UHF DF gradient
// ---------------------------------------------------------------------------

fn uhf_df_gradient_seeded(
    m: &Molecule,
    s: &Setup,
    ca: &Array2<f64>,
    cb: &Array2<f64>,
) -> Array2<f64> {
    let prep = prep_at(m, &s.basis_name);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let r = solve_uhf_with_guess(
        &ParallelContext::default(),
        m,
        &prep,
        &bounds,
        &s.cfg,
        Some((ca, cb)),
    )
    .unwrap();
    assert!(r.converged, "displaced UHF did not converge");
    assert!(
        (r.energy - s.uhf.energy).abs() < BASIN_TOL,
        "displaced UHF changed state: E {:.10} vs reference {:.10}",
        r.energy,
        s.uhf.energy
    );
    uhf_gradient(m, &prep, op, &bounds, &r, None).unwrap()
}

fn fd_own_gradient(s: &Setup) -> Array2<f64> {
    let ca = s.uhf.mos_alpha.clone();
    let cb = s.uhf.mos_beta.clone().expect("β MOs");
    fd_hessian_richardson(&s.mol, 2e-3, |m| uhf_df_gradient_seeded(m, s, &ca, &cb))
}

/// Agreement bar of the production Hessian with the Richardson FD of the own
/// UHF DF gradient (SCF re-converged at every displaced geometry, to
/// energy_conv 1e-14 / density_conv 1e-12: at 1e-12 / 1e-10 the FD reference
/// itself carries ~5e-8 of SCF noise and every configuration, exact K included,
/// misses at that level). Floors measured (release): CH2 triplet STO-3G RI-J
/// 6.6e-11, RI-K 7.2e-10, 6-31G RI-JK 3.2e-10; NH2 STO-3G RI-JK 5.0e-10, 6-31G
/// RI-K 7.3e-10; cc-pVDZ / def2-universal-jkfit RI-JK: CH2 7.1e-10, NH2 2.8e-9
/// (worst, 3.6x below the bar), CH3 9.4e-10. Metric lambda_min >= 9.6e-6 (cond
/// <= 3.5e7), no mode in the gradient's noise band.
const BAR_OWN: f64 = 1e-8;

fn check_full(mol: &Molecule, basis_name: &str, j_aux: &str, k_aux: &str, tol: f64) -> f64 {
    if !has_deriv2() {
        return 0.0;
    }
    let s = setup(mol, basis_name, j_aux, k_aux);
    for aux in [&j_aux, &k_aux] {
        if !aux.is_empty() {
            let (lmin, lmax, clean) = clean_gap(mol, aux);
            eprintln!(
                "{basis_name}/{aux}: metric lambda_min {lmin:.3e}, cond {:.2e}",
                lmax / lmin
            );
            assert!(clean, "metric has a mode in the gradient's noise band");
        }
    }
    let p = parts(&s);
    let fd = fd_own_gradient(&s);
    eprintln!(
        "{basis_name}/J={j_aux:?}/K={k_aux:?}: CPHF iterations {:?}, residual {:.2e}, response asym {:.2e}",
        p.cphf_iterations, p.cphf_max_residual, p.response_asymmetry
    );
    assert!(
        max_abs_diff(&p.skeleton(), &fd) > 100.0 * tol,
        "skeleton alone matches FD: the full test cannot see the response"
    );
    let miss = assert_close(
        &format!("{basis_name}/J={j_aux:?}/K={k_aux:?}: full UHF Hessian"),
        &p.total(),
        &fd,
        tol,
    );
    assert!(
        p.response_asymmetry < 1e-9,
        "asym {:.3e}",
        p.response_asymmetry
    );
    miss
}

#[test]
fn full_hessian_matches_fd_nh2_sto3g_ri_jk() {
    check_full(&nh2(), "sto-3g", "def2-universal-jkfit", "def2-universal-jkfit", BAR_OWN);
}

#[test]
fn full_hessian_matches_fd_ch2_sto3g_ri_j() {
    check_full(&ch2(), "sto-3g", "def2-universal-jkfit", "", BAR_OWN);
}

#[test]
fn full_hessian_matches_fd_ch2_sto3g_ri_k() {
    check_full(&ch2(), "sto-3g", "", "def2-universal-jkfit", BAR_OWN);
}

#[test]
fn full_hessian_matches_fd_ch2_631g_ri_jk() {
    check_full(&ch2(), "6-31g", "cc-pvdz-ri", "cc-pvdz-ri", BAR_OWN);
}

#[test]
fn full_hessian_matches_fd_nh2_631g_ri_k() {
    check_full(&nh2(), "6-31g", "", "cc-pvdz-ri", BAR_OWN);
}

#[test]
#[ignore = "slow: NH2 cc-pVDZ / def2-universal-jkfit RI-JK own-gradient FD; measured 53 s for the three cc-pVDZ tests run together (3 test threads, box load ~8)"]
fn full_hessian_matches_fd_nh2_ccpvdz_jkfit() {
    check_full(&nh2(), "cc-pvdz", "def2-universal-jkfit", "def2-universal-jkfit", BAR_OWN);
}

#[test]
#[ignore = "slow: CH2 triplet cc-pVDZ / def2-universal-jkfit RI-JK own-gradient FD; measured 53 s for the three cc-pVDZ tests run together (3 test threads, box load ~8)"]
fn full_hessian_matches_fd_ch2_ccpvdz_jkfit() {
    check_full(&ch2(), "cc-pvdz", "def2-universal-jkfit", "def2-universal-jkfit", BAR_OWN);
}

#[test]
#[ignore = "slow: CH3 doublet cc-pVDZ / def2-universal-jkfit RI-JK own-gradient FD; measured 53 s for the three cc-pVDZ tests run together (3 test threads, box load ~8)"]
fn full_hessian_matches_fd_ch3_ccpvdz_jkfit() {
    check_full(&ch3(), "cc-pvdz", "def2-universal-jkfit", "def2-universal-jkfit", BAR_OWN);
}

#[test]
fn translational_and_rotational_invariance() {
    if !has_deriv2() {
        return;
    }
    let s = setup(&ch2(), "6-31g", "cc-pvdz-ri", "cc-pvdz-ri");
    let total = parts(&s).total();
    let n = s.mol.atoms.len();
    let n3 = 3 * n;
    let mut worst = 0.0f64;
    for a in 0..n3 {
        for axis in 0..3 {
            let sum: f64 = (0..n).map(|b| total[(a, 3 * b + axis)]).sum();
            worst = worst.max(sum.abs());
        }
    }
    eprintln!("translational invariance: worst row sum {worst:.3e}");
    assert!(worst < 1e-9, "row sum {worst:.3e}");
    let g = uhf_gradient(
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &s.uhf,
        None,
    )
    .unwrap();
    let r: Vec<[f64; 3]> = s.mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let cross = |e: [f64; 3], v: [f64; 3]| {
        [
            e[1] * v[2] - e[2] * v[1],
            e[2] * v[0] - e[0] * v[2],
            e[0] * v[1] - e[1] * v[0],
        ]
    };
    let mut worst_rot = 0.0f64;
    for axis in 0..3 {
        let mut e = [0.0; 3];
        e[axis] = 1.0;
        let u: Vec<f64> = r.iter().flat_map(|&ri| cross(e, ri)).collect();
        for atom in 0..n {
            let gv = cross(e, [g[(atom, 0)], g[(atom, 1)], g[(atom, 2)]]);
            for k in 0..3 {
                let hu: f64 = (0..n3).map(|b| total[(3 * atom + k, b)] * u[b]).sum();
                worst_rot = worst_rot.max((hu - gv[k]).abs());
            }
        }
    }
    eprintln!("rotational invariance: worst |H (e x r) - e x g| = {worst_rot:.3e}");
    assert!(worst_rot < 1e-7, "rotational {worst_rot:.3e}");
}

// ---------------------------------------------------------------------------
// Negative controls and mutants
// ---------------------------------------------------------------------------

/// Bar of the mutation suite (NH2 6-31G / cc-pVDZ-RI, RI-JK). Derived from
/// both sides: the production agreement with the differenced own gradient on
/// this system is 7.0e-10, so 3e-9 is 4x above what a correct Hessian shows;
/// the closest mutants are the stencil-step-x10 ones (K 4.8e-8 = 16 bars, J
/// 2.4e-7 = 81 bars, i.e. 1.6x and 8x the 10-bar pass line, margins stated, not
/// assumed); every other mutant misses by >= 2.1e-5 (the CPHF exact-K one) and
/// mostly by > 5e-2. One mutant (the CPHF Coulomb matrix from the alpha response
/// density only) makes the orbital Hessian indefinite and is caught by the
/// solver's typed error instead of a number.
const BAR_MUT: f64 = 3e-9;

type Mutant = (
    &'static str,
    RijMutants,
    RikMutants,
    UhfDfMutants,
);

fn mutated_total(s: &Setup, m: &Mutant) -> Result<Array2<f64>, ferric_core::FerricError> {
    uhf_hessian_parts_mutated(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &s.uhf,
        &s.cfg,
        &m.1,
        &m.2,
        &m.3,
    )
    .map(|p| p.total())
}

fn all_mutants() -> Vec<Mutant> {
    use ferric_scf::hessian_df_stencil::{StencilConfig, Variant};
    let (rj, rk, uh) = (
        RijMutants::default(),
        RikMutants::default(),
        UhfDfMutants::default(),
    );
    let st = |v| StencilConfig {
        variant: v,
        ..Default::default()
    };
    vec![
        (
            "alpha and beta K^x swapped",
            rj,
            rk,
            UhfDfMutants {
                swap_k_channels: true,
                ..uh
            },
        ),
        (
            "closed-shell channel (alpha -1/4, X = sqrt2 C) for each spin",
            rj,
            rk,
            UhfDfMutants {
                closed_shell_channel: true,
                ..uh
            },
        ),
        (
            "J built from the alpha density only",
            rj,
            rk,
            UhfDfMutants {
                j_from_alpha: true,
                ..uh
            },
        ),
        (
            "beta channel dropped",
            rj,
            rk,
            UhfDfMutants {
                drop_beta_channel: true,
                ..uh
            },
        ),
        (
            "CPHF: exchange of the wrong spin",
            rj,
            rk,
            UhfDfMutants {
                cphf_wrong_spin_k: true,
                ..uh
            },
        ),
        (
            "CPHF: exact K where the SCF fitted it",
            rj,
            rk,
            UhfDfMutants {
                cphf_exact_k: true,
                ..uh
            },
        ),
        (
            "CPHF: Coulomb from the alpha response density only",
            rj,
            rk,
            UhfDfMutants {
                cphf_j_from_alpha: true,
                ..uh
            },
        ),
        (
            "beta channel with a different metric cut",
            rj,
            RikMutants {
                channel_metric_cut: Some((1, 1e-2)),
                ..rk
            },
            uh,
        ),
        (
            "drop K 2c second derivative",
            rj,
            RikMutants {
                drop_metric_second: true,
                ..rk
            },
            uh,
        ),
        (
            "drop K cross terms",
            rj,
            RikMutants {
                drop_cross: true,
                ..rk
            },
            uh,
        ),
        (
            "drop K^x metric term",
            rj,
            RikMutants {
                drop_kx_metric: true,
                ..rk
            },
            uh,
        ),
        (
            "drop K^x altogether",
            rj,
            RikMutants {
                drop_kx: true,
                ..rk
            },
            uh,
        ),
        (
            "omit aux-centre part of K^x",
            rj,
            RikMutants {
                omit_aux_centre_kx: true,
                ..rk
            },
            uh,
        ),
        (
            "transpose of the pair in K^x",
            rj,
            RikMutants {
                transpose_pair_kx: true,
                ..rk
            },
            uh,
        ),
        (
            "exchange weight x2",
            rj,
            RikMutants {
                alpha_factor: 2.0,
                ..rk
            },
            uh,
        ),
        (
            "J: drop 2c second derivative",
            RijMutants {
                drop_metric_second: true,
                ..rj
            },
            rk,
            uh,
        ),
        (
            "J: drop cross terms",
            RijMutants {
                drop_cross: true,
                ..rj
            },
            rk,
            uh,
        ),
        (
            "J: omit aux-fit part of J^x",
            RijMutants {
                omit_aux_jx: true,
                ..rj
            },
            rk,
            uh,
        ),
        (
            "stencil step x10 (J)",
            RijMutants {
                stencil: st(Variant::Step10),
                ..rj
            },
            rk,
            uh,
        ),
        (
            "one-shell displacement (J)",
            RijMutants {
                stencil: st(Variant::SingleShell),
                ..rj
            },
            rk,
            uh,
        ),
        (
            "stencil step x10 (K)",
            rj,
            RikMutants {
                stencil: st(Variant::Step10),
                ..rk
            },
            uh,
        ),
        (
            "one-shell displacement (K)",
            rj,
            RikMutants {
                stencil: st(Variant::SingleShell),
                ..rk
            },
            uh,
        ),
    ]
}

#[test]
fn every_wired_mutant_misses_by_more_than_ten_bars() {
    if !has_deriv2() {
        return;
    }
    let s = setup(&nh2(), "6-31g", "cc-pvdz-ri", "cc-pvdz-ri");
    let prod = parts(&s).total();
    let again = mutated_total(
        &s,
        &(
            "production",
            RijMutants::default(),
            RikMutants::default(),
            UhfDfMutants::default(),
        ),
    )
    .unwrap();
    assert_eq!(
        max_abs_diff(&prod, &again),
        0.0,
        "production is not reproducible"
    );
    let mut caught_by_error = Vec::new();
    for m in all_mutants() {
        // A mutant whose broken kernel makes the CPHF solver refuse (a
        // non-positive-definite orbital Hessian, no convergence) is caught by
        // a typed error: reported, and counted as caught, not as a number.
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
                eprintln!("MUTANT {}: typed error {e}", m.0);
                caught_by_error.push(m.0);
            }
        }
    }
    eprintln!("mutants caught by a typed error: {caught_by_error:?}");
}

#[test]
fn production_agrees_with_fd_at_the_mutation_bar() {
    // The mutation bar is only meaningful if the production Hessian meets it
    // against the independent reference on the very system the mutants use.
    if !has_deriv2() {
        return;
    }
    let s = setup(&nh2(), "6-31g", "cc-pvdz-ri", "cc-pvdz-ri");
    let fd = fd_own_gradient(&s);
    let prod = max_abs_diff(&parts(&s).total(), &fd);
    eprintln!("production vs FD on the mutation system: {prod:.3e} (bar {BAR_MUT:.0e})");
    assert!(prod < BAR_MUT);
}

#[test]
fn negative_controls_miss_the_ri_jk_reference() {
    // Exact-integral, RI-J-only and RI-K-only Hessians of the same molecule
    // and basis must miss the RI-JK Hessian by far more than the bar; so must
    // dropping K^x, the beta channel or the whole response.
    if !has_deriv2() {
        return;
    }
    let aux = "cc-pvdz-ri";
    let jk = setup(&nh2(), "6-31g", aux, aux);
    let prod = parts(&jk).total();
    let exact = parts(&setup(&nh2(), "6-31g", "", "")).total();
    let j_only = parts(&setup(&nh2(), "6-31g", aux, "")).total();
    let k_only = parts(&setup(&nh2(), "6-31g", "", aux)).total();
    for (name, h) in [
        ("exact-integral UHF", &exact),
        ("RI-J only (exact K)", &j_only),
        ("RI-K only (exact J)", &k_only),
    ] {
        let miss = max_abs_diff(h, &prod);
        eprintln!(
            "NEGATIVE CONTROL {name}: miss {miss:.3e} ({:.1e} bars of {BAR_OWN:.0e})",
            miss / BAR_OWN
        );
        assert!(miss > 10.0 * BAR_OWN, "{name}: {miss:.3e}");
    }
    let p = parts(&jk);
    let miss_resp = max_abs_diff(&p.skeleton(), &prod);
    eprintln!("NEGATIVE CONTROL no response: miss {miss_resp:.3e}");
    assert!(miss_resp > 10.0 * BAR_OWN);
}

// ---------------------------------------------------------------------------
// Refusals and the pre-SCF probe
// ---------------------------------------------------------------------------

#[test]
fn rohf_with_ri_and_other_unsupported_configurations_are_refused() {
    use ferric_scf::hessian::{
        analytic_rohf_hessian_available, analytic_uhf_hessian_available, rohf_hessian_preflight,
        uhf_hessian_preflight,
    };
    let op = Operator::coulomb();
    let mol = nh2();
    let prep = prep_at(&mol, "sto-3g");
    let ri = |j: &str, k: &str| scf_config(j, k);
    // UHF RI-J, RI-K, RI-JK: accepted by the preflight and the probe (the
    // latter needs the second-derivative engines).
    for (j, k) in [
        ("def2-universal-jkfit", ""),
        ("", "def2-universal-jkfit"),
        ("def2-universal-jkfit", "def2-universal-jkfit"),
    ] {
        uhf_hessian_preflight(op, &ri(j, k)).expect("UHF RI preflight");
        if has_deriv2() {
            analytic_uhf_hessian_available(&mol, &prep, op, &ri(j, k)).expect("UHF RI probe");
        }
        // ROHF with RI is refused by both.
        let e = rohf_hessian_preflight(op, &ri(j, k)).unwrap_err().to_string();
        eprintln!("ROHF RI config: {e}");
        assert!(e.contains("ROHF") && e.contains("RI"), "{e}");
        assert!(analytic_rohf_hessian_available(&mol, &prep, op, &ri(j, k)).is_err());
    }
    // Other refusals stay for UHF with RI.
    let mut with_ext = ri("def2-universal-jkfit", "def2-universal-jkfit");
    with_ext.external_potential = Some(ferric_core::external_potential::ExternalPotential {
        point_charges: vec![ferric_core::external_potential::PointCharge {
            q: 0.3,
            x: 5.0,
            y: 0.0,
            z: 0.0,
        }],
        ..Default::default()
    });
    let e = uhf_hessian_preflight(op, &with_ext).unwrap_err().to_string();
    eprintln!("UHF RI + external potential: {e}");
    assert!(e.contains("external potential"), "{e}");
    let ks = RhfConfig {
        xc: Some("pbe".into()),
        ..ri("def2-universal-jkfit", "def2-universal-jkfit")
    };
    assert!(uhf_hessian_preflight(op, &ks).is_err());
    let cosmo = RhfConfig {
        cosmo: Some(ferric_scf::cosmo::CosmoConfig::default()),
        ..ri("def2-universal-jkfit", "def2-universal-jkfit")
    };
    assert!(uhf_hessian_preflight(op, &cosmo).is_err());
}

#[test]
fn route_and_state_inconsistencies_are_refused() {
    if !has_deriv2() {
        return;
    }
    let s = setup(&nh2(), "sto-3g", "def2-universal-jkfit", "def2-universal-jkfit");
    let ctx = ParallelContext::default();
    let hess = |r: &ScfResult, cfg: &RhfConfig| {
        uhf_hessian_parts(&ctx, &s.mol, &s.prep, Operator::coulomb(), &s.bounds, r, cfg)
    };
    // A route whose RI-K basis differs from the configuration's.
    let mut bad = s.uhf.clone();
    bad.df_jk.as_mut().unwrap().k_aux = Some("cc-pvdz-ri".into());
    let e = hess(&bad, &s.cfg).unwrap_err().to_string();
    assert!(e.contains("RI-K") && e.contains("differs from the configuration"), "{e}");
    // Range-separated fitted exchange in the route.
    let mut rsh = s.uhf.clone();
    rsh.df_jk.as_mut().unwrap().rsh_k = Some(("def2-universal-jkfit".into(), 0.3));
    assert!(hess(&rsh, &s.cfg).is_err());
    // A spin density that is not built from its first nocc MOs (here the α
    // occupied and a virtual MO exchanged): typed error, not a wrong Hessian.
    let mut inconsistent = s.uhf.clone();
    let n = nocc(&s.mol);
    let (a, b) = (
        inconsistent.mos_alpha.column(0).to_owned(),
        inconsistent.mos_alpha.column(n[0]).to_owned(),
    );
    inconsistent.mos_alpha.column_mut(0).assign(&b);
    inconsistent.mos_alpha.column_mut(n[0]).assign(&a);
    let e = hess(&inconsistent, &s.cfg).unwrap_err().to_string();
    eprintln!("inconsistent UHF state: {e}");
    assert!(!e.is_empty());
}

#[test]
fn df_off_uhf_is_unchanged_by_the_density_fitting_branches() {
    // Exact SCFs carry no route, never enter the RI code, and give the same
    // Hessian bit for bit with df_*_aux unset or "". The hash of every part
    // against origin/main is recorded in the PR description.
    if !has_deriv2() {
        return;
    }
    let mol = ch2();
    let prep = prep_at(&mol, "sto-3g");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let hess = |aux: Option<String>| {
        let cfg = RhfConfig {
            max_iter: 300,
            energy_conv: 1e-12,
            density_conv: 1e-10,
            df_j_aux: aux.clone(),
            df_k_aux: aux,
            ..Default::default()
        };
        let uhf = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();
        assert!(uhf.df_jk.is_none());
        uhf_hessian_parts(&ParallelContext::default(), &mol, &prep, op, &bounds, &uhf, &cfg)
            .unwrap()
    };
    let (a, b) = (hess(None), hess(Some(String::new())));
    assert_eq!(max_abs_diff(&a.total(), &b.total()), 0.0);
}
