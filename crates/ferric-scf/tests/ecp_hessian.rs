//! Analytic RHF/UHF Hessian with effective core potentials (issue #285).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test ecp_hessian
//! ```
//!
//! # What is tested, and what each anchor can and cannot see
//!
//! The ECP Hessian is two terms (`src/hessian_ecp.rs`): the skeleton
//! `Σ D ∂²V_ECP` (a masked fourth-order stencil of the first-derivative ECP
//! blocks) and `∂V_ECP/∂x` in the CPHF Fock derivative.
//!
//! * `masked_stencil_matches_unmasked_stencil_of_the_gradient_matrices`: the
//!   skeleton against an INDEPENDENT construction of the same quantity (a
//!   stencil of `ecp_potential_deriv`, the gradient's own per-atom matrices,
//!   unmasked, folded by the other code path). It sees the mask and the
//!   per-atom fold (bra/ket/ECP centre), not the integral kernel.
//! * `translational_invariance`: Σ over atoms of a skeleton column is zero.
//!   EXACT BY INVARIANCE (bra + ket + centre = 0 per triple): a bookkeeping
//!   guard that every centre was folded somewhere, blind to folding a centre
//!   onto the WRONG atom.
//! * `stencil_step_sweep`: `|S(h) − S(2h)|` stays below `TOL_STEP`: the
//!   truncation error of the stencil. It cannot see a sign error (that flips
//!   `S(h)` and `S(2h)` together); the sign is pinned by the masked-vs-unmasked
//!   test (an independent fold of the same stencil weights) and by the FD
//!   tests.
//! * `*_hessian_matches_fd_of_analytic_gradient`: the full analytic Hessian
//!   against central FD of ferric's analytic ECP gradient (validated against
//!   PySCF and ORCA, #206). It tests the ASSEMBLY (skeleton + `F^x` + CPHF),
//!   not the integrals: the skeleton stencil and the gradient share the
//!   first-derivative kernel. The independent anchor is PySCF's analytic ECP
//!   Hessian (`validation_ecp_hessian.rs`). Negative control: dropping the
//!   skeleton term must miss by `MUST_MISS` × bar, and so must dropping
//!   `∂V_ECP/∂x` from `F^x` (negative control 2, through
//!   `rij_support::with_ecp_first_order_dropped`; #406).
//! * `all_electron_system_is_bit_identical_with_or_without_basis_ecps`.
//! * `ecp_g_shell_is_a_typed_error`. (An ECP not applied to the molecule is
//!   still refused, and `Auto` frequencies route ECP systems to the analytic
//!   Hessian: `frequency_hessian_dispatch.rs`.)
//!
//! # MUTATION (each must turn a named test red; see `src/hessian_ecp.rs`)
//!
//! * centre derivative folded onto the wrong atom (`ecp_atom[u]` → shell atom);
//! * `∂V_ECP/∂x` dropped from `add_extra_first_order`;
//! * mask drops the triples with a centre on the displaced atom;
//! * stencil weight sign flipped.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::ecp_potential_deriv;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{rhf_gradient, uhf_gradient};
use ferric_scf::hessian::{rhf_hessian_parts, uhf_hessian_parts};
use ferric_scf::hessian_ecp::{ecp_skeleton_hessian, ecp_skeleton_hessian_step};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ndarray::Array2;

/// Analytic vs FD of the analytic gradient, Ha/Bohr².
const TOL_FD: f64 = 2e-6;
/// A negative control must miss by this factor of the bar.
const MUST_MISS: f64 = 10.0;
/// Stencil self-consistency `|S(h) − S(2h)|`, Ha/Bohr².
const TOL_STEP: f64 = 1e-8;
const H_FD: f64 = 1e-3;

/// H–O–I (closed shell), tilted off the axes so every Cartesian block is
/// populated, with three distinct atoms around the ECP centre.
const HOI: &str = "3\nHOI tilted\nH 0.30 -0.40 0.20\nO 0.00 0.10 0.95\nI 0.45 0.20 2.80\n";
/// H–I–H doublet (odd valence electron count), tilted likewise.
const HIH: &str = "3\nHIH tilted\nH 0.30 -0.40 0.20\nI 0.00 0.00 1.61\nH -1.10 0.70 2.90\n";

fn has_deriv2() -> bool {
    ferric_integrals::engine::libint_max_deriv_order() >= 2
}

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

fn ecp_mol(xyz: &str, charge: i32, mult: usize, bs: &BasisSet) -> Molecule {
    let mut mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    mol.apply_ecp(bs);
    mol
}

struct Solved {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    scf: ScfResult,
}

fn solve(mol: &Molecule, bs: &BasisSet, open_shell: bool) -> Solved {
    let prep = PreparedBasis::new(mol, bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = config();
    let scf = if open_shell {
        solve_uhf(&ctx, mol, &prep, &bounds, &cfg).unwrap()
    } else {
        solve_rhf(&ctx, mol, &prep, op, &bounds, &cfg).unwrap()
    };
    assert!(scf.converged);
    Solved {
        mol: mol.clone(),
        prep,
        bounds,
        scf,
    }
}

fn total_density(s: &Solved, open_shell: bool) -> Array2<f64> {
    if open_shell {
        &s.scf.density_alpha + s.scf.density_beta.as_ref().unwrap()
    } else {
        s.scf.density_r().clone()
    }
}

fn hessian(s: &Solved, open_shell: bool) -> ferric_scf::hessian::RhfHessianParts {
    let (ctx, op, cfg) = (ParallelContext::default(), Operator::coulomb(), config());
    if open_shell {
        uhf_hessian_parts(&ctx, &s.mol, &s.prep, op, &s.bounds, &s.scf, &cfg).unwrap()
    } else {
        rhf_hessian_parts(&ctx, &s.mol, &s.prep, op, &s.bounds, &s.scf, &cfg).unwrap()
    }
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

fn gradient(mol: &Molecule, bs: &BasisSet, open_shell: bool) -> Array2<f64> {
    let s = solve(mol, bs, open_shell);
    let op = Operator::coulomb();
    if open_shell {
        uhf_gradient(&s.mol, &s.prep, op, &s.bounds, &s.scf, None).unwrap()
    } else {
        rhf_gradient(&s.mol, &s.prep, op, &s.bounds, &s.scf, None).unwrap()
    }
}

/// Columns `cols` of the central FD of the analytic gradient (the other
/// columns stay zero): each column costs two SCF + gradient evaluations.
fn fd_columns(mol: &Molecule, bs: &BasisSet, open_shell: bool, cols: &[usize]) -> Array2<f64> {
    let n3 = 3 * mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    for &b in cols {
        let g = |h: f64| gradient(&displaced(mol, b / 3, b % 3, h), bs, open_shell);
        let (gp, gm) = (g(H_FD), g(-H_FD));
        for a in 0..n3 {
            out[(a, b)] = (gp[(a / 3, a % 3)] - gm[(a / 3, a % 3)]) / (2.0 * H_FD);
        }
    }
    out
}

/// Largest |a - b| over the listed columns.
fn max_diff_cols(a: &Array2<f64>, b: &Array2<f64>, cols: &[usize]) -> f64 {
    let mut m = 0.0f64;
    for &c in cols {
        for r in 0..a.nrows() {
            assert!(a[(r, c)].is_finite() && b[(r, c)].is_finite(), "non-finite");
            m = m.max((a[(r, c)] - b[(r, c)]).abs());
        }
    }
    m
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| {
        assert!(x.is_finite() && y.is_finite(), "non-finite Hessian entry");
        m.max((x - y).abs())
    })
}

/// The same skeleton from the gradient's per-atom matrices (`ecp_potential_deriv`,
/// a different fold), differentiated by the same stencil, unmasked.
fn unmasked_skeleton(mol: &Molecule, bs: &BasisSet, d: &Array2<f64>, h: f64) -> Array2<f64> {
    let n3 = 3 * mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    let w = [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)];
    for b in 0..n3 {
        for (off, wt) in w {
            let m = displaced(mol, b / 3, b % 3, off * h);
            let derivs = ecp_potential_deriv(&m, bs).unwrap().unwrap();
            for (a, blk) in derivs.iter().enumerate() {
                for (x, dv) in blk.iter().enumerate() {
                    let g: f64 = d.iter().zip(dv).map(|(p, q)| p * q).sum();
                    out[(3 * a + x, b)] += wt * g / (12.0 * h);
                }
            }
        }
    }
    out
}

#[test]
fn masked_stencil_matches_unmasked_stencil_of_the_gradient_matrices() {
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(HOI, 0, 1, &bs);
    let s = solve(&mol, &bs, false);
    let d = total_density(&s, false);
    let masked = ecp_skeleton_hessian(&mol, &bs, &d).unwrap();
    // The skeleton is returned symmetrised; so is the reference.
    let raw = unmasked_skeleton(&mol, &bs, &d, 2e-3);
    let reference = (&raw + &raw.t()) * 0.5;
    let diff = max_diff(&masked, &reference);
    let scale = masked.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    eprintln!("masked vs unmasked skeleton: {diff:.3e} (max element {scale:.3e})");
    assert!(scale > 1e-3, "vacuous skeleton");
    assert!(diff < 1e-10, "masked != unmasked: {diff:.3e}");
}

#[test]
fn translational_invariance() {
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(HOI, 0, 1, &bs);
    let d = total_density(&solve(&mol, &bs, false), false);
    let h = ecp_skeleton_hessian(&mol, &bs, &d).unwrap();
    let n = mol.atoms.len();
    let mut worst = 0.0f64;
    for col in 0..3 * n {
        for x in 0..3 {
            let s: f64 = (0..n).map(|a| h[(3 * a + x, col)]).sum();
            worst = worst.max(s.abs());
        }
    }
    eprintln!("ECP skeleton column sums (bookkeeping guard): {worst:.3e}");
    assert!(worst < 1e-10, "column sum {worst:.3e}");
}

#[test]
fn stencil_step_sweep() {
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(HOI, 0, 1, &bs);
    let d = total_density(&solve(&mol, &bs, false), false);
    let s1 = ecp_skeleton_hessian_step(&mol, &bs, &d, 2e-3).unwrap();
    let s2 = ecp_skeleton_hessian_step(&mol, &bs, &d, 4e-3).unwrap();
    let step = max_diff(&s1, &s2);
    let scale = s1.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    eprintln!("|S(2e-3) - S(4e-3)| = {step:.3e} (max element {scale:.3e})");
    assert!(scale > 1e-3, "vacuous skeleton");
    assert!(step < TOL_STEP, "stencil not converged in h: {step:.3e}");
}

/// Analytic Hessian vs FD columns `cols` of ferric's analytic gradient (two SCF
/// + gradient evaluations each), and the no-skeleton negative control. The
/// column is a coordinate of an atom that is NOT the ECP atom (HOI: O.y = 4,
/// HIH: H.y = 1): the column of the ECP atom itself cannot see a centre
/// derivative folded onto the displaced atom (mutant 1, see the module doc).
/// The whole matrix was checked once at development time (`max|analytic - FD|`
/// 3.8e-7 RHF, 1.6e-7 UHF).
fn check_against_fd(open_shell: bool, xyz: &str, charge: i32, mult: usize, cols: &[usize]) {
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(xyz, charge, mult, &bs);
    let s = solve(&mol, &bs, open_shell);
    let parts = hessian(&s, open_shell);
    let an = parts.total();
    let fd = fd_columns(&mol, &bs, open_shell, cols);
    let d = max_diff_cols(&an, &fd, cols);
    eprintln!("ECP Hessian (UHF: {open_shell}): max|analytic - FD| = {d:.3e} (bar {TOL_FD:.0e})");
    assert!(d < TOL_FD, "max|analytic - FD| = {d:.3e}");

    // Negative control 1: without the skeleton term.
    let dens = total_density(&s, open_shell);
    let skel = ecp_skeleton_hessian(&mol, &bs, &dens).unwrap();
    let miss_skel = max_diff_cols(&(&an - &skel), &fd, cols);
    eprintln!("  without the ECP skeleton the Hessian misses FD by {miss_skel:.3e}");
    assert!(
        miss_skel > MUST_MISS * TOL_FD,
        "skeleton too small: {miss_skel:.3e}"
    );

    // Negative control 2: `∂V_ECP/∂x` left out of the CPHF `F^x` (the
    // skeleton stays). Same density, so the miss is the response term alone.
    let dropped =
        ferric_scf::hessian::rij_support::with_ecp_first_order_dropped(|| hessian(&s, open_shell))
            .total();
    let miss_fx = max_diff_cols(&dropped, &fd, cols);
    eprintln!("  without dV_ECP/dx in F^x the Hessian misses FD by {miss_fx:.3e}");
    assert!(
        miss_fx > MUST_MISS * TOL_FD,
        "dV_ECP/dx in F^x too small to be guarded: {miss_fx:.3e}"
    );
}

#[test]
fn rhf_hessian_matches_fd_of_analytic_gradient() {
    if !has_deriv2() {
        return;
    }
    check_against_fd(false, HOI, 0, 1, &[4]);
}

#[test]
fn uhf_hessian_matches_fd_of_analytic_gradient() {
    if !has_deriv2() {
        return;
    }
    check_against_fd(true, HIH, 0, 2, &[1]);
}

#[test]
fn availability_probe_refuses_an_ecp_g_shell_without_an_scf() {
    // No SCF and no libint second-derivative engine: the ECP angular-momentum
    // check runs first, so `Auto` frequencies fall back to finite differences.
    let bs = basis::bundled("def2-qzvp").unwrap();
    let mol = ecp_mol(HOI, 0, 1, &bs);
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    assert!(prep.max_l() > 3, "def2-QZVP iodine must carry g functions");
    let err = ferric_scf::hessian::analytic_hessian_available(
        &mol,
        &prep,
        Operator::coulomb(),
        &RhfConfig::default(),
    )
    .expect_err("an ECP system with g shells must be unavailable");
    eprintln!("probe: {err}");
    assert!(err.to_string().contains("ECP"), "{err}");
    let err = ferric_scf::hessian::analytic_uhf_hessian_available(
        &mol,
        &prep,
        Operator::coulomb(),
        &RhfConfig::default(),
    )
    .expect_err("UHF probe likewise");
    assert!(err.to_string().contains("ECP"), "{err}");
}

/// The probe's last gate is the libint second-derivative engines (one-body
/// and four-centre, `l <= 3`): the ECP Hessian itself needs only those plus
/// the FIRST-derivative ECP kernel, no three-/two-centre second derivatives.
/// So an ECP system is analytic on every libint that has 4c/1e second
/// derivatives to f (the wheel's trimmed 2.7.2 export, conda-forge 2.13.1)
/// and is refused by the probe, before any SCF, on a first-derivative-only
/// libint (the stock mpqc4 export), so `Auto` frequencies fall back to FD.
/// Either branch runs on whichever libint is linked; the other is not
/// reachable in one build.
#[test]
fn probe_outcome_for_an_ecp_system_tracks_the_linked_libint() {
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(HOI, 0, 1, &bs);
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    assert!(prep.max_l() <= 3, "def2-SVP iodine must stay within f");
    let probe = ferric_scf::hessian::analytic_hessian_available(
        &mol,
        &prep,
        Operator::coulomb(),
        &RhfConfig::default(),
    );
    if has_deriv2() {
        probe.expect("an f-bounded ECP system is analytic where libint has second derivatives");
    } else {
        let err = probe.expect_err("first-derivative libint must be refused by the probe");
        eprintln!("probe: {err}");
        assert!(err.to_string().contains("second-derivative"), "{err}");
    }
}

#[test]
fn ecp_g_shell_is_a_typed_error() {
    // def2-QZVP iodine carries g functions: above the second-derivative
    // angular momentum, so a typed error before any integral, not a panic.
    let bs = basis::bundled("def2-qzvp").unwrap();
    let mol = ecp_mol(HOI, 0, 1, &bs);
    let err = ecp_skeleton_hessian(&mol, &bs, &Array2::zeros((1, 1)))
        .expect_err("a g shell must be refused");
    eprintln!("g-shell refusal: {err}");
    assert!(err.to_string().contains("l = 4"), "{err}");
}

#[test]
fn all_electron_system_is_bit_identical_with_or_without_basis_ecps() {
    if !has_deriv2() {
        return;
    }
    let water = "3\nwater\nO 0.01 -0.02 0.12\nH 0.03 0.77 -0.46\nH -0.02 -0.74 -0.49\n";
    let with = basis::bundled("def2-svp").unwrap();
    assert!(
        !with.ecps.is_empty(),
        "def2-SVP must carry ECPs for this test"
    );
    let mut without = with.clone();
    without.ecps.clear();
    let (m1, m2) = (ecp_mol(water, 0, 1, &with), ecp_mol(water, 0, 1, &without));
    let (h1, h2) = (
        hessian(&solve(&m1, &with, false), false).total(),
        hessian(&solve(&m2, &without, false), false).total(),
    );
    assert!(
        h1.iter().zip(&h2).all(|(a, b)| a.to_bits() == b.to_bits()),
        "all-electron Hessian changed by an unused ECP table"
    );
}

/// Wall-clock cost of the analytic ECP Hessian against the 6N-gradient FD
/// Hessian it replaces (whole `harmonic_frequencies`, one SCF included), and
/// the ECP skeleton alone. Run with `-- --ignored --nocapture` on a quiet box.
#[test]
#[ignore = "cost probe: prints wall-clock times, asserts nothing about speed"]
fn cost_vs_finite_difference_hessian() {
    use ferric_scf::frequencies::{
        harmonic_frequencies, FrequencyConfig, HessianMethod, HessianSource,
    };
    use std::time::Instant;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bs = basis::bundled("def2-svp").unwrap();
    for system in ["hi", "ch3i"] {
        let xyz = root.join(format!("testdata/molecules/validation/ecp/{system}.xyz"));
        let mut mol = Molecule::load_xyz(xyz.to_str().unwrap()).unwrap();
        mol.apply_ecp(&bs);
        let freq = |hessian| {
            let t = Instant::now();
            let r = harmonic_frequencies(
                &ParallelContext::default(),
                &mol,
                "def2-svp",
                Operator::coulomb(),
                &RhfConfig::default(),
                &FrequencyConfig {
                    hessian,
                    ..Default::default()
                },
            )
            .unwrap();
            (t.elapsed().as_secs_f64(), r)
        };
        let d = total_density(&solve(&mol, &bs, false), false);
        for rep in 0..3 {
            let t = Instant::now();
            ecp_skeleton_hessian(&mol, &bs, &d).unwrap();
            let t_skel = t.elapsed().as_secs_f64();
            let (t_an, an) = freq(HessianMethod::Analytic);
            let (t_fd, fd) = freq(HessianMethod::FiniteDifference);
            assert_eq!(an.hessian_source, HessianSource::Analytic);
            let dev = an
                .frequencies
                .iter()
                .zip(&fd.frequencies)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            eprintln!(
                "{system} rep {rep}: analytic {t_an:.2} s (ECP skeleton alone {t_skel:.2} s), \
                 FD {t_fd:.2} s ({} gradients), ratio {:.1}x, max |dfreq| {dev:.3e} cm^-1",
                fd.n_gradient_evaluations,
                t_fd / t_an
            );
        }
    }
}

/// The ECP terms alone: one evaluation of the ECP first-derivative matrices
/// (`ecp_potential_deriv`, the gradient's kernel) against the skeleton stencil.
#[test]
#[ignore = "cost probe: prints wall-clock times, asserts nothing about speed"]
fn cost_of_the_ecp_terms() {
    use std::time::Instant;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bs = basis::bundled("def2-svp").unwrap();
    for system in ["hi", "ch3i"] {
        let xyz = root.join(format!("testdata/molecules/validation/ecp/{system}.xyz"));
        let mut mol = Molecule::load_xyz(xyz.to_str().unwrap()).unwrap();
        mol.apply_ecp(&bs);
        let d = total_density(&solve(&mol, &bs, false), false);
        for rep in 0..3 {
            let t = Instant::now();
            ecp_potential_deriv(&mol, &bs).unwrap();
            let t_grad = t.elapsed().as_secs_f64();
            let t = Instant::now();
            ecp_skeleton_hessian(&mol, &bs, &d).unwrap();
            let t_skel = t.elapsed().as_secs_f64();
            eprintln!(
                "{system} rep {rep}: ECP first-derivative matrices {t_grad:.3} s, \
                 skeleton stencil {t_skel:.3} s ({:.1}x, 12N = {} calls)",
                t_skel / t_grad,
                12 * mol.atoms.len()
            );
        }
    }
}
