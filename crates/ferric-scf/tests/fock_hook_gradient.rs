//! Nuclear gradient of a Fock-hooked SCF
//! (`gradient::with_fock_hook_gradient`; derivation in this crate's
//! `book/src/07-hartree-fock/06-fock-hooks.md#fock-hooks-equations`).
//!
//! The hook reproduces a built-in feature: one fixed point charge, entered as
//! the hook `V = h_ext - h` (its nuclear-attraction matrix) with
//! `E_hook = tr(D V) + E_charge-nuclear`, against the same charge as
//! `RhfConfig.external_potential`. The hook's explicit term is the
//! Hellmann-Feynman derivative at fixed D (basis-centre derivatives of
//! `V`, the charge does not move) plus the charge-nuclear repulsion
//! derivative: `oneelectron_gradient(D, 0, ext) - oneelectron_gradient(D, 0,
//! None)`. The built-in gradient of the hooked result supplies every other
//! term, including the Pulay term of `V` through its energy-weighted density.
//! Covered: RHF and RKS (LDA) water, UHF and ROHF NH2, all 6-31G.
//!
//! Bars. The two routes solve the same equations; they differ by the rounding
//! of `h + V` against `h_ext` (a few ulp of |h|, ~1e-15 Ha) and by where each
//! SCF stops. Energy: variational, so the stopping error enters squared; both
//! energies sit within `ENERGY_CONV` of the stationary value: bar
//! `2 ENERGY_CONV`. Gradient: linear in the density error dD at fixed
//! integrals, |dg| <= sum_{mu nu} |dD_{mu nu}| max|d(integral)/dR|; with
//! `DENSITY_CONV` = 1e-10 per element, at most 13^2 (water) / 13^2 (NH2)
//! elements and derivative integrals below ~10 Ha/Bohr, |dg| <= 1.7e-7. The
//! bar is that bound, 2e-7 Ha/Bohr. MEASURED (2026-11-14): |dE| <= 2.8e-14
//! Ha and max|dg| = 1.2e-14 (RHF), 4.7e-15 (RKS), 6.2e-11 (UHF), 2.0e-11
//! (ROHF) Ha/Bohr, against gradients of 2.5e-2..3.4e-2.
//!
//! Artifact hypothesis. If the explicit term were dropped the hooked gradient
//! would differ from the built-in one by the Hellmann-Feynman force of the
//! charge. If the energy-weighted density missed `V` (built from the
//! un-hooked Fock) it would differ by `tr[1/2 D V D dS/dR]`. Mutations run
//! (each failed on the gradient assertion, then restored):
//! `with_fock_hook_gradient` returning `grad` without the explicit term
//! (max|dg| 4.5e-2 RHF, 4.7e-2 RKS, 3.7e-2 UHF, 3.7e-2 ROHF); ROHF spin Fock
//! matrices recorded before the hook is added (`rohf.rs`, the
//! `spin_focks_last` capture, so W lacks V: max|dg| 3.0e-2).

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{
    oneelectron_gradient, restricted_open_scf_gradient, restricted_scf_gradient,
    unrestricted_scf_gradient, with_fock_hook_gradient,
};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, solve_rhf_with_hook, RhfConfig};
use ferric_scf::rohf::{solve_rohf, solve_rohf_with_hook};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_hook};
use ndarray::Array2;

const ENERGY_CONV: f64 = 1e-10;
const DENSITY_CONV: f64 = 1e-10;
const GRAD_BAR: f64 = 2e-7;

const WATER: &str = "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692";
const NH2: &str = "N 0.0 0.0 0.1404\nH 0.0 0.8012 -0.4914\nH 0.0 -0.8012 -0.4914";

#[derive(Clone, Copy, Debug)]
enum Kind {
    Rhf,
    Rks,
    Uhf,
    Rohf,
}

fn mol(xyz: &str, mult: usize) -> Molecule {
    let n = xyz.lines().count();
    Molecule::parse_xyz(&format!("{n}\n\n{xyz}\n"), 0, mult).unwrap()
}

/// One point charge off every symmetry plane, ~2.5 Bohr from the nearest atom.
fn charge() -> ExternalPotential {
    ExternalPotential {
        point_charges: vec![PointCharge {
            q: -0.6,
            x: 1.9,
            y: 1.4,
            z: 2.1,
        }],
        ..Default::default()
    }
}

fn config(kind: Kind) -> RhfConfig {
    let mut c = RhfConfig::default()
        .with_energy_conv(ENERGY_CONV)
        .with_density_conv(DENSITY_CONV)
        .with_max_iter(300);
    c.xc = matches!(kind, Kind::Rks).then(|| "LDA".to_string());
    c
}

fn gradient(
    kind: Kind,
    m: &Molecule,
    prep: &PreparedBasis,
    b: &SchwarzBounds,
    c: &RhfConfig,
    r: &ScfResult,
) -> Array2<f64> {
    let bs = prep.basis_set();
    let op = Operator::coulomb();
    match kind {
        Kind::Rhf | Kind::Rks => restricted_scf_gradient(m, prep, bs, op, b, c, r),
        Kind::Uhf => unrestricted_scf_gradient(m, prep, bs, op, b, c, r),
        Kind::Rohf => restricted_open_scf_gradient(m, prep, bs, op, b, c, r),
    }
    .unwrap()
}

/// The built-in route: the charge in `RhfConfig.external_potential`.
fn builtin(
    kind: Kind,
    m: &Molecule,
    prep: &PreparedBasis,
    b: &SchwarzBounds,
) -> (f64, Array2<f64>) {
    let ctx = ParallelContext::default();
    let mut c = config(kind);
    c.external_potential = Some(charge());
    let r = match kind {
        Kind::Rhf | Kind::Rks => solve_rhf(&ctx, m, prep, Operator::coulomb(), b, &c),
        Kind::Uhf => solve_uhf(&ctx, m, prep, b, &c),
        Kind::Rohf => solve_rohf(&ctx, m, prep, Operator::coulomb(), b, &c),
    }
    .unwrap();
    assert!(r.converged, "{kind:?} built-in SCF did not converge");
    (r.energy, gradient(kind, m, prep, b, &c, &r))
}

/// The explicit term of the point-charge hook at fixed total density `d`.
fn explicit_term(m: &Molecule, prep: &PreparedBasis, d: &Array2<f64>) -> Array2<f64> {
    let zero = Array2::<f64>::zeros(d.dim());
    let ext = charge();
    oneelectron_gradient(m, prep, d, &zero, Some(&ext)).unwrap()
        - oneelectron_gradient(m, prep, d, &zero, None).unwrap()
}

/// The hooked route: the charge as a Fock hook, plus its explicit term.
fn hooked_gradient(
    kind: Kind,
    m: &Molecule,
    prep: &PreparedBasis,
    b: &SchwarzBounds,
) -> (f64, Array2<f64>) {
    let ctx = ParallelContext::default();
    let ext = charge();
    let v = oneelectron::hcore_with_external(prep, Some(&ext)).unwrap()
        - oneelectron::hcore_with_external(prep, None).unwrap();
    let e_cn = ext.charge_nuclear_energy(m);
    let c = config(kind);
    let r = match kind {
        Kind::Rhf | Kind::Rks => {
            let mut h = |d: &Array2<f64>| Ok(((d * &v).sum() + e_cn, v.clone()));
            solve_rhf_with_hook(&ctx, m, prep, Operator::coulomb(), b, &c, &mut h)
        }
        Kind::Uhf | Kind::Rohf => {
            let mut h = |da: &Array2<f64>, db: &Array2<f64>| {
                Ok((((da + db) * &v).sum() + e_cn, v.clone(), v.clone()))
            };
            if matches!(kind, Kind::Uhf) {
                solve_uhf_with_hook(&ctx, m, prep, b, &c, &mut h)
            } else {
                solve_rohf_with_hook(&ctx, m, prep, b, &c, &mut h)
            }
        }
    }
    .unwrap();
    assert!(r.converged, "{kind:?} hooked SCF did not converge");
    let explicit = explicit_term(m, prep, r.density_total());
    let g = with_fock_hook_gradient(gradient(kind, m, prep, b, &c, &r), &explicit).unwrap();
    (r.energy, g)
}

fn check(kind: Kind, xyz: &str, mult: usize) {
    let m = mol(xyz, mult);
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&m, &bs).unwrap();
    let b = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let (e_ref, g_ref) = builtin(kind, &m, &prep, &b);
    let (e_hook, g_hook) = hooked_gradient(kind, &m, &prep, &b);
    let de = (e_hook - e_ref).abs();
    let dg = (&g_hook - &g_ref)
        .iter()
        .fold(0.0f64, |a, x| a.max(x.abs()));
    let g_size = g_ref.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    eprintln!("{kind:?}: |dE| = {de:.3e} Ha, max|dg| = {dg:.3e} Ha/Bohr (max|g| = {g_size:.3e})");
    assert!(de <= 2.0 * ENERGY_CONV, "{kind:?}: |dE| = {de:.3e}");
    assert!(
        dg <= GRAD_BAR,
        "{kind:?}: max|dg| = {dg:.3e} > {GRAD_BAR:.1e}"
    );
}

#[test]
fn point_charge_hook_gradient_matches_external_potential_rhf() {
    check(Kind::Rhf, WATER, 1);
}

#[test]
fn point_charge_hook_gradient_matches_external_potential_rks() {
    check(Kind::Rks, WATER, 1);
}

#[test]
fn point_charge_hook_gradient_matches_external_potential_uhf() {
    check(Kind::Uhf, NH2, 2);
}

#[test]
fn point_charge_hook_gradient_matches_external_potential_rohf() {
    check(Kind::Rohf, NH2, 2);
}

#[test]
fn a_misshaped_or_nonfinite_explicit_term_is_refused() {
    let g = Array2::<f64>::zeros((3, 3));
    let e = with_fock_hook_gradient(g.clone(), &Array2::zeros((2, 3))).unwrap_err();
    assert!(e.to_string().contains("shape"), "{e}");
    let mut bad = Array2::<f64>::zeros((3, 3));
    bad[(1, 2)] = f64::NAN;
    let e = with_fock_hook_gradient(g, &bad).unwrap_err();
    assert!(e.to_string().contains("not finite"), "{e}");
}
