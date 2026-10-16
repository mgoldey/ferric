//! Analytic RHF/UHF Hessian with effective core potentials: the combinations
//! the base suite (`ecp_hessian.rs`) does not cover (issue #406).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test ecp_hessian_combinations
//! ```
//!
//! # What is tested, and what each anchor can and cannot see
//!
//! * `two_ecp_elements_*` (synthetic second ECP on Br, def2-SVP) and the
//!   `#[ignore]`d `two_real_ecp_elements_*` (Ag + I, aug-cc-pVDZ-PP): H–Ag–I (UHF doublet), TWO different ECP elements
//!   (Z = 47 and 53, different ECP tables, different atoms). Anchors: the
//!   masked skeleton stencil against the unmasked stencil of the gradient's
//!   per-atom matrices (1e-10: sees the mask and the per-ECP-centre fold, which
//!   is where a "second ECP lands on the first ECP's atom" defect lives),
//!   translational invariance of the skeleton, and the full analytic Hessian
//!   against central FD of ferric's analytic ECP gradient. The FD columns are
//!   a coordinate of the non-ECP atom, of Ag and of I, so each ECP atom is
//!   displaced once and the OTHER ECP's centre derivative must fold onto the
//!   displaced atom.
//! * `ecp_with_{point,smeared,field}_*`: an ECP system in an external
//!   environment. The ECP terms (skeleton + `dV_ECP/dx` in `F^x`) and the
//!   environment terms (charge–nucleus, `dV_ext/dx`) are independent code, so
//!   the only place they can interact wrongly is the screened nuclear charge
//!   `Z - n_core` in the charge–nucleus term: an unscreened `Z` is invisible
//!   without an ECP, and the ECP suite has no charges. The FD anchor therefore
//!   carries the interaction. Negative controls: the same system's ECP-only
//!   Hessian (no environment) and the environment-only terms removed must each
//!   miss FD by `MUST_MISS` × bar.
//! * `charge_nuclear_hessian_uses_the_screened_nuclear_charge`: screened vs
//!   bare-Z charge-nucleus Hessian differ by >= 10x the bar (the unscreened-Z
//!   mutant's full-Hessian miss, 1.3e-1 to 1.4e-1, is a one-time measurement).
//! * `empty_environment_is_bit_identical_*` / `zero_charge_limit_*`: the
//!   no-charge limit of the ECP+environment route.
//! * `ecp_with_ri_and_environment_is_refused_by_the_probe`: typed refusal.
//!
//! NOT covered: the synthetic Br table is iodine's scaled (same shells and
//! `n_core` = 28; the real H-Ag-I also has `n_core` 28 on both), so
//! table-structure defects and per-element `n_core` mix-ups are untested.
//!
//! Artifact hypothesis, written before measuring: if the combination is
//! broken (a) the ECP+environment FD residual is O(q·Z/R³) ~ 1e-2 (an
//! unscreened nuclear charge) or O(skeleton) ~ 1e-1, never the 1e-7 FD floor;
//! (b) a physics-correct implementation sits at the FD floor. These differ by
//! 4 orders, so the bar can tell them apart.
//!
//! # MUTATION (each run once at development time; measured residuals are in
//! the PR / issue record, not asserted in code)
//!
//! * `ecp_atom[u]` -> `ecp_atom[0]` in `folded_first_derivative`
//!   (two-element test);
//! * `effective_z()` -> `z` in `ExternalPotential::charge_nuclear_hessian`
//!   (ECP + point/smeared charges test).

use ferric_core::basis::{self, BasisSet};
use ferric_core::external_potential::{ExternalPotential, PointCharge, SmearedCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::ecp_potential_deriv;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{rhf_gradient, uhf_gradient};
use ferric_scf::hessian::{rhf_hessian_parts, uhf_hessian_parts};
use ferric_scf::hessian_ecp::ecp_skeleton_hessian;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ndarray::Array2;

/// Analytic vs FD of the analytic gradient, Ha/Bohr² (base suite: 2e-6; the
/// environment adds charge–nucleus terms of the same FD floor, ~5e-7).
const TOL_FD: f64 = 5e-6;
const MUST_MISS: f64 = 10.0;
const H_FD: f64 = 1e-3;

/// H–O–I, tilted (as in `ecp_hessian.rs`).
const HOI: &str = "3\nHOI tilted\nH 0.30 -0.40 0.20\nO 0.00 0.10 0.95\nI 0.45 0.20 2.80\n";
/// H–I–H doublet, tilted.
const HIH: &str = "3\nHIH tilted\nH 0.30 -0.40 0.20\nI 0.00 0.00 1.61\nH -1.10 0.70 2.90\n";
/// H–Ag–I doublet: two DIFFERENT ECP elements (Z = 47 and 53), tilted off
/// every axis. Needs `aug-cc-pVDZ-PP`: it is the only bundled basis that
/// carries an ECP for two elements (def2-ECP is bundled for iodine only).
const HAGI: &str = "3\nHAgI tilted\nH 0.30 -0.40 0.20\nAg 0.00 0.10 1.60\nI 0.45 0.20 4.15\n";
/// H-Br-I cation (closed shell) for the synthetic two-ECP basis (see `two_ecp_system`).
const HBRI: &str = "3\nHBrI tilted\nH 0.30 -0.40 0.20\nBr 0.00 0.10 1.50\nI 0.45 0.20 4.10\n";
const BS_TWO: &str = "aug-cc-pvdz-pp";

fn has_deriv2() -> bool {
    ferric_integrals::engine::libint_max_deriv_order() >= 2
}

fn config(ext: Option<&ExternalPotential>) -> RhfConfig {
    RhfConfig {
        max_iter: 300,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        external_potential: ext.cloned(),
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

fn solve(mol: &Molecule, bs: &BasisSet, open: bool, ext: Option<&ExternalPotential>) -> Solved {
    let prep = PreparedBasis::new(mol, bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = config(ext);
    let scf = if open {
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

fn density(s: &Solved, open: bool) -> Array2<f64> {
    if open {
        &s.scf.density_alpha + s.scf.density_beta.as_ref().unwrap()
    } else {
        s.scf.density_r().clone()
    }
}

fn hessian(s: &Solved, open: bool, ext: Option<&ExternalPotential>) -> Array2<f64> {
    let (ctx, op, cfg) = (ParallelContext::default(), Operator::coulomb(), config(ext));
    let parts = if open {
        uhf_hessian_parts(&ctx, &s.mol, &s.prep, op, &s.bounds, &s.scf, &cfg).unwrap()
    } else {
        rhf_hessian_parts(&ctx, &s.mol, &s.prep, op, &s.bounds, &s.scf, &cfg).unwrap()
    };
    parts.total()
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

fn gradient(
    mol: &Molecule,
    bs: &BasisSet,
    open: bool,
    ext: Option<&ExternalPotential>,
) -> Array2<f64> {
    let s = solve(mol, bs, open, ext);
    let op = Operator::coulomb();
    if open {
        uhf_gradient(&s.mol, &s.prep, op, &s.bounds, &s.scf, ext).unwrap()
    } else {
        rhf_gradient(&s.mol, &s.prep, op, &s.bounds, &s.scf, ext).unwrap()
    }
}

/// Columns `cols` of the central FD of the analytic gradient (others zero).
fn fd_columns(
    mol: &Molecule,
    bs: &BasisSet,
    open: bool,
    ext: Option<&ExternalPotential>,
    cols: &[usize],
) -> Array2<f64> {
    let n3 = 3 * mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    for &b in cols {
        let g = |h: f64| gradient(&displaced(mol, b / 3, b % 3, h), bs, open, ext);
        let (gp, gm) = (g(H_FD), g(-H_FD));
        for a in 0..n3 {
            out[(a, b)] = (gp[(a / 3, a % 3)] - gm[(a / 3, a % 3)]) / (2.0 * H_FD);
        }
    }
    out
}

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
        assert!(x.is_finite() && y.is_finite(), "non-finite entry");
        m.max((x - y).abs())
    })
}

/// Unmasked stencil of the gradient's per-atom ECP matrices (the independent
/// construction of the skeleton, as in `ecp_hessian.rs`).
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

fn point_environment() -> ExternalPotential {
    ExternalPotential {
        point_charges: vec![
            PointCharge {
                q: 0.55,
                x: 4.0,
                y: 1.5,
                z: -2.0,
            },
            PointCharge {
                q: -0.35,
                x: -3.2,
                y: 2.4,
                z: 6.0,
            },
        ],
        ..Default::default()
    }
}

fn smeared_environment() -> ExternalPotential {
    ExternalPotential {
        smeared_charges: vec![
            SmearedCharge {
                q: 0.45,
                x: -2.5,
                y: 3.0,
                z: 4.0,
                width: 0.9,
            },
            SmearedCharge {
                q: -0.6,
                x: 3.0,
                y: -2.8,
                z: 7.0,
                width: 1.3,
            },
        ],
        ..Default::default()
    }
}

fn field_environment() -> ExternalPotential {
    ExternalPotential {
        field: Some([0.004, -0.003, 0.005]),
        ..Default::default()
    }
}

/// Analytic vs FD (columns `cols`) with environment `ext`; returns the
/// analytic Hessian and the FD columns for the controls.
fn check_with_environment(
    label: &str,
    open: bool,
    xyz: &str,
    mult: usize,
    ext: &ExternalPotential,
    cols: &[usize],
) {
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(xyz, 0, mult, &bs);
    let s = solve(&mol, &bs, open, Some(ext));
    let an = hessian(&s, open, Some(ext));
    let fd = fd_columns(&mol, &bs, open, Some(ext), cols);
    let d = max_diff_cols(&an, &fd, cols);
    eprintln!("{label}: max|analytic - FD| = {d:.3e} (bar {TOL_FD:.0e})");
    assert!(d < TOL_FD, "{label}: max|analytic - FD| = {d:.3e}");

    // Control 1: the environment's Hessian terms removed, same ECP terms and
    // same density: the Hessian of the SAME SCF without the environment.
    let gas = hessian(&s, open, None);
    let miss_env = max_diff_cols(&gas, &fd, cols);
    eprintln!("  {label}: environment terms removed misses FD by {miss_env:.3e}");
    assert!(
        miss_env > MUST_MISS * TOL_FD,
        "{label}: the environment barely moves the Hessian ({miss_env:.3e}); vacuous"
    );
    // Control 2: ECP skeleton removed.
    let skel = ecp_skeleton_hessian(&mol, &bs, &density(&s, open)).unwrap();
    let miss_skel = max_diff_cols(&(&an - &skel), &fd, cols);
    eprintln!("  {label}: ECP skeleton removed misses FD by {miss_skel:.3e}");
    assert!(miss_skel > MUST_MISS * TOL_FD, "{label}: skeleton vacuous");
}

/// The two-ECP system and its basis.
///
/// `real = true`: H-Ag-I in `aug-cc-pVDZ-PP`, the only bundled basis with
/// ECPs on two real elements. It costs ~29 min wall for the FD column set
/// (measured 2026-10-15, 6 cores), so it runs only under `--ignored`.
///
/// `real = false` (default suite): H-Br-I in def2-SVP where Br is GIVEN a
/// second, DIFFERENT ECP table (iodine's def2 ECP with every coefficient
/// scaled by 0.6, 28 core electrons). Not chemistry; the Hessian/gradient
/// consistency being tested is a property of two distinct ECP definitions on
/// two distinct centres, which this keeps, at def2-SVP cost.
fn two_ecp_system(real: bool) -> (BasisSet, Molecule, bool) {
    if real {
        let bs = basis::bundled(BS_TWO).unwrap();
        let mol = ecp_mol(HAGI, 0, 2, &bs);
        return (bs, mol, true);
    }
    let mut bs = basis::bundled("def2-svp").unwrap();
    let mut def = bs.ecp_for_element(53).expect("def2 iodine ECP").clone();
    for sh in &mut def.shells {
        for t in &mut sh.terms {
            t.coef *= 0.6;
        }
    }
    bs.ecps.insert(35, def);
    // The cation is closed shell (32 valence electrons); the neutral UHF
    // doublet of this synthetic system converges to an unstable solution.
    let mol = ecp_mol(HBRI, 1, 1, &bs);
    (bs, mol, false)
}

fn check_two_ecp_skeleton(real: bool) {
    let (bs, mol, open) = two_ecp_system(real);
    assert!(
        mol.atoms.iter().filter(|a| a.n_core_ecp > 0).count() == 2,
        "both heavy atoms must carry an ECP"
    );
    assert_ne!(mol.atoms[1].z, mol.atoms[2].z);
    let s = solve(&mol, &bs, open, None);
    let d = density(&s, open);
    let masked = ecp_skeleton_hessian(&mol, &bs, &d).unwrap();
    let raw = unmasked_skeleton(&mol, &bs, &d, 2e-3);
    let reference = (&raw + &raw.t()) * 0.5;
    let diff = max_diff(&masked, &reference);
    let scale = masked.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    eprintln!("two ECPs, masked vs unmasked skeleton: {diff:.3e} (max element {scale:.3e})");
    assert!(scale > 1e-3, "vacuous skeleton");
    assert!(diff < 1e-10, "masked != unmasked: {diff:.3e}");

    let n = mol.atoms.len();
    let mut worst = 0.0f64;
    for col in 0..3 * n {
        for x in 0..3 {
            let sum: f64 = (0..n).map(|a| masked[(3 * a + x, col)]).sum();
            worst = worst.max(sum.abs());
        }
    }
    eprintln!("two ECPs, skeleton column sums: {worst:.3e}");
    assert!(worst < 1e-10, "column sum {worst:.3e}");
}

fn check_two_ecp_fd(real: bool) {
    if !has_deriv2() {
        return;
    }
    let (bs, mol, open) = two_ecp_system(real);
    let s = solve(&mol, &bs, open, None);
    let an = hessian(&s, open, None);
    // H.y, heavy-1.x, I.z: every atom once; a column of an ECP atom sees the
    // OTHER ECP's centre derivative folded onto the displaced atom.
    let cols = [1, 3, 8];
    let fd = fd_columns(&mol, &bs, open, None, &cols);
    let d = max_diff_cols(&an, &fd, &cols);
    eprintln!("two ECPs (open shell: {open}, real: {real}): max|analytic - FD| = {d:.3e} (bar {TOL_FD:.0e})");
    assert!(d < TOL_FD, "max|analytic - FD| = {d:.3e}");
    let skel = ecp_skeleton_hessian(&mol, &bs, &density(&s, open)).unwrap();
    let miss = max_diff_cols(&(&an - &skel), &fd, &cols);
    eprintln!("  without the skeleton: misses FD by {miss:.3e}");
    assert!(miss > MUST_MISS * TOL_FD);
}

#[test]
fn two_ecp_elements_masked_stencil_matches_unmasked_and_is_translation_invariant() {
    check_two_ecp_skeleton(false);
}

#[test]
fn two_ecp_elements_hessian_matches_fd_of_analytic_gradient() {
    check_two_ecp_fd(false);
}

#[test]
#[ignore = "~29 min: H-Ag-I / aug-cc-pVDZ-PP, two real ECP elements"]
fn two_real_ecp_elements_masked_stencil_matches_unmasked() {
    check_two_ecp_skeleton(true);
}

#[test]
#[ignore = "~29 min: H-Ag-I / aug-cc-pVDZ-PP, two real ECP elements"]
fn two_real_ecp_elements_uhf_hessian_matches_fd_of_analytic_gradient() {
    check_two_ecp_fd(true);
}

#[test]
fn ecp_with_point_charges_rhf_matches_fd() {
    if !has_deriv2() {
        return;
    }
    // O.y and I.z: the I column also sees the screened charge on the ECP atom.
    check_with_environment("HOI + point", false, HOI, 1, &point_environment(), &[4, 8]);
}

#[test]
fn ecp_with_smeared_charges_rhf_matches_fd() {
    if !has_deriv2() {
        return;
    }
    check_with_environment(
        "HOI + smeared",
        false,
        HOI,
        1,
        &smeared_environment(),
        &[4, 8],
    );
}

#[test]
fn ecp_with_uniform_field_rhf_matches_fd() {
    if !has_deriv2() {
        return;
    }
    // The nuclear dipole term of the field is linear in R, so the field
    // Hessian is purely electronic; the ECP enters through the screened Z in
    // the (zero) nuclear second derivative and through the density response.
    let ext = field_environment();
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(HOI, 0, 1, &bs);
    let s = solve(&mol, &bs, false, Some(&ext));
    let an = hessian(&s, false, Some(&ext));
    let cols = [4, 8];
    let fd = fd_columns(&mol, &bs, false, Some(&ext), &cols);
    let d = max_diff_cols(&an, &fd, &cols);
    eprintln!("HOI + field: max|analytic - FD| = {d:.3e} (bar {TOL_FD:.0e})");
    assert!(d < TOL_FD, "max|analytic - FD| = {d:.3e}");
    // Negative control: the field terms removed (same SCF, no environment).
    let miss = max_diff_cols(&hessian(&s, false, None), &fd, &cols);
    eprintln!("  HOI + field: field terms removed misses FD by {miss:.3e}");
    assert!(
        miss > MUST_MISS * TOL_FD,
        "field barely matters: {miss:.3e}"
    );
}

/// The screened nuclear charge `Z - n_core` is what the charge-nucleus
/// Hessian must use: with the ECP core removed it differs from the bare-Z
/// Hessian by far more than the bar, so an unscreened-Z mutant is visible
/// (the FD tests above carry the same defect through the full Hessian).
#[test]
fn charge_nuclear_hessian_uses_the_screened_nuclear_charge() {
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(HOI, 0, 1, &bs);
    let mut bare = mol.clone();
    bare.atoms.iter_mut().for_each(|a| a.n_core_ecp = 0);
    for (label, ext) in [
        ("point", point_environment()),
        ("smeared", smeared_environment()),
    ] {
        let d = max_diff(
            &ext.charge_nuclear_hessian(&mol),
            &ext.charge_nuclear_hessian(&bare),
        );
        eprintln!("{label}: screened vs bare-Z charge-nucleus Hessian differ by {d:.3e}");
        assert!(d > MUST_MISS * TOL_FD, "{label}: {d:.3e}");
    }
}

#[test]
fn ecp_with_all_environments_uhf_matches_fd() {
    if !has_deriv2() {
        return;
    }
    let ext = ExternalPotential {
        point_charges: point_environment().point_charges,
        smeared_charges: smeared_environment().smeared_charges,
        field: field_environment().field,
    };
    // H.y of the doublet, and the iodine y.
    check_with_environment("HIH + all (UHF)", true, HIH, 2, &ext, &[1, 4]);
}

#[test]
fn empty_environment_is_bit_identical_to_none_and_zero_charge_is_1e10() {
    if !has_deriv2() {
        return;
    }
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(HOI, 0, 1, &bs);
    let s = solve(&mol, &bs, false, None);
    let none = hessian(&s, false, None);
    let empty = ExternalPotential::default();
    let h_empty = hessian(&s, false, Some(&empty));
    assert!(
        none.iter()
            .zip(&h_empty)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "empty ExternalPotential changed the ECP Hessian"
    );
    // A zero-valued charge of each kind (same SCF, so the difference is the
    // Hessian terms alone).
    let zero = ExternalPotential {
        point_charges: vec![PointCharge {
            q: 0.0,
            x: 4.0,
            y: 1.0,
            z: -2.0,
        }],
        smeared_charges: vec![SmearedCharge {
            q: 0.0,
            x: -3.0,
            y: 2.0,
            z: 5.0,
            width: 1.0,
        }],
        field: Some([0.0; 3]),
    };
    let h_zero = hessian(&s, false, Some(&zero));
    let d = max_diff(&none, &h_zero);
    eprintln!("zero-charge limit: |H(q=0) - H(none)| = {d:.3e}");
    assert!(d < 1e-10, "{d:.3e}");
}

#[test]
fn ecp_with_ri_and_environment_is_refused_by_the_probe() {
    // RI J/K + external potential is refused ("untested") before any SCF, and
    // RI + ECP is refused too; both reasons must reach the caller as typed
    // errors so `Auto` frequencies fall back to FD.
    let bs = basis::bundled("def2-svp").unwrap();
    let mol = ecp_mol(HOI, 0, 1, &bs);
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let cfg = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        external_potential: Some(point_environment()),
        ..Default::default()
    };
    let err =
        ferric_scf::hessian::analytic_hessian_available(&mol, &prep, Operator::coulomb(), &cfg)
            .expect_err("RI + environment + ECP must be refused");
    eprintln!("probe: {err}");
    assert!(
        err.to_string()
            .contains("combined with an external potential"),
        "{err}"
    );

    // RI + ECP alone (RHF): the ECP reason itself.
    let cfg = RhfConfig {
        external_potential: None,
        ..cfg
    };
    let err =
        ferric_scf::hessian::analytic_hessian_available(&mol, &prep, Operator::coulomb(), &cfg)
            .expect_err("RHF RI + ECP must be refused");
    eprintln!("probe: {err}");
    assert!(
        err.to_string().contains("effective core potential"),
        "{err}"
    );
}
