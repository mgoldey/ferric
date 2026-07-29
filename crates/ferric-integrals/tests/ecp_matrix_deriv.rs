//! Exactness anchor for the libecpint first-derivative binding
//! (`ferric_ecp_matrix_deriv` / [`ecp_potential_deriv`]).
//!
//! These tests check the DERIVATIVE INTEGRALS in isolation, before any SCF
//! machinery is involved: analytic `dV_ECP/dR` vs central finite difference of
//! `V_ECP` itself. If this fails, an SCF gradient built on it cannot be right,
//! and the FD-vs-analytic SCF test in `ferric-scf/tests/ecp_rhf.rs` would not be
//! able to tell a binding bug from a gradient-assembly bug.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::oneelectron::{ecp_potential, ecp_potential_deriv};
use ndarray::Array2;

/// Central finite difference of `V_ECP` w.r.t. atom `a`, coordinate `c`.
fn fd_vecp(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    a: usize,
    c: usize,
    h: f64,
) -> Array2<f64> {
    let shift = |sign: f64| {
        let mut m = mol.clone();
        match c {
            0 => m.atoms[a].x += sign * h,
            1 => m.atoms[a].y += sign * h,
            _ => m.atoms[a].zpos += sign * h,
        }
        ecp_potential(&m, bs).expect("V_ECP")
    };
    let vp = shift(1.0);
    let vm = shift(-1.0);
    (vp - vm) / (2.0 * h)
}

/// Compare analytic `dV_ECP/dR` with central FD, scaled by the magnitude of the
/// derivative block.
///
/// The bar is RELATIVE, not absolute, and that is deliberate — see the
/// FD-step sweep recorded on [`ecp_matrix_deriv_i2_matches_finite_difference`].
/// An absolute bound would penalise a system merely for having larger `V_ECP`
/// entries: I2's derivatives are ~25x larger than HI's, so the same relative
/// accuracy shows up as a proportionally larger absolute difference.
fn check_deriv_vs_fd(mol: &Molecule, bs: &ferric_core::basis::BasisSet, rel_tol: f64, label: &str) {
    let derivs = ecp_potential_deriv(mol, bs)
        .expect("ecp_potential_deriv")
        .expect("basis carries ECPs");
    assert_eq!(derivs.len(), mol.atoms.len(), "{label}: one entry per atom");

    // h = 1e-4 Bohr sits in the measured flat basin for both systems.
    let h = 1e-4;
    let mut worst_abs = 0.0f64;
    let mut worst_rel = 0.0f64;
    let mut worst_where = (0usize, 0usize);
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let fd = fd_vecp(mol, bs, a, c, h);
            let an = &derivs[a][c];
            assert_eq!(an.dim(), fd.dim(), "{label}: shape mismatch");
            let scale = an.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
            let abs = (an - &fd).iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let rel = abs / scale;
            worst_abs = worst_abs.max(abs);
            if rel > worst_rel {
                worst_rel = rel;
                worst_where = (a, c);
            }
        }
    }
    eprintln!(
        "{label}: dV_ECP/dR vs FD(h={h:.0e}) -> max abs {worst_abs:.3e}, \
         max rel {worst_rel:.3e} (worst at atom {}, coord {})",
        worst_where.0, worst_where.1
    );
    assert!(
        worst_rel < rel_tol,
        "{label}: dV_ECP/dR disagrees with finite difference by {worst_rel:.3e} \
         relative (abs {worst_abs:.3e}), tol {rel_tol:.1e}"
    );
}

/// I2: a real, NON-symmetric-per-atom case. Each iodine feels the other's ECP,
/// so the cross-center (C) derivative terms are exercised, not just the
/// same-center ones. This is the test with teeth.
///
/// # Why the bound is relative, and why 1e-6
///
/// This started as an absolute `1e-7` bar and failed at `1.413e-7`. Measured
/// FD-step sweep (`max |analytic - FD|`, def2-SVP, release build):
///
/// ```text
///   h (Bohr)      I2          HI
///     3.0e-3   1.334e-6    6.608e-6
///     1.0e-3   1.551e-7    7.342e-7
///     5.0e-4   1.449e-7    1.835e-7
///     2.0e-4   1.427e-7    5.775e-8
///     1.0e-4   1.413e-7    5.553e-8
///     5.0e-5   1.394e-7    1.802e-6   <- round-off takes over
///     2.0e-5   8.531e-7    3.039e-7
/// ```
///
/// From 3e-3 to 1e-3 the error falls ~9x, the O(h^2) law for a central
/// difference, so the derivative tracks the true slope. Below that, I2 does not
/// keep falling — it PLATEAUS at ~1.4e-7. That plateau is neither truncation
/// (which would predict 3.7e-10 at h=5e-5) nor plain round-off (whose implied
/// noise floor would have to shrink with h, which is incoherent). It is
/// h-independent jitter in `V_ECP` itself: libecpint's radial quadrature makes
/// slightly different internal grid/cutoff decisions as the geometry shifts.
/// Corroborating evidence — at h=1e-4 the FD matrix loses its exact symmetry
/// (`max|fd_ij - fd_ji|` = 5.3e-12) while the analytic matrix is symmetric to
/// 2.8e-17, i.e. machine precision. The noise is in the FD reference, not in the
/// derivative being tested.
///
/// Scaled by magnitude the discrepancy is bounded and flat: max relative error
/// 2.16e-7 for I2 and ~5.3e-8 for HI, stable across h — consistent with
/// libecpint's own integral accuracy. Hence a relative bound of 1e-6, which
/// clears the measured 2.2e-7 with margin while still catching any real
/// derivative error (a wrong term shows up at O(1) relative, not O(1e-7)).
#[test]
fn ecp_matrix_deriv_i2_matches_finite_difference() {
    let bs = basis::bundled("def2-svp").unwrap();
    // I2 near equilibrium (~2.67 A).
    let mut mol = Molecule::parse_xyz("2\n\nI 0.0 0.0 0.0\nI 0.0 0.0 2.67\n", 0, 1).unwrap();
    mol.apply_ecp(&bs);
    assert!(
        mol.atoms.iter().all(|a| a.n_core_ecp > 0),
        "def2-svp must carry an ECP for I; otherwise this test is vacuous"
    );
    check_deriv_vs_fd(&mol, &bs, 1e-6, "I2/def2-SVP");
}

/// A heteronuclear ECP/all-electron mix: only the heavy atom carries an ECP, so
/// the light atom's derivative comes ENTIRELY from bra/ket (A/B) terms with no
/// ECP of its own. This catches an atom-id mapping bug that a homonuclear
/// diatomic cannot: the two atoms are no longer interchangeable.
///
/// Same relative bound as the I2 test (see its doc comment for the FD-step
/// sweep); HI measures ~5.3e-8 relative, comfortably inside it.
#[test]
fn ecp_matrix_deriv_hi_matches_finite_difference() {
    let bs = basis::bundled("def2-svp").unwrap();
    let mut mol = Molecule::parse_xyz("2\n\nH 0.0 0.0 0.0\nI 0.0 0.0 1.61\n", 0, 1).unwrap();
    mol.apply_ecp(&bs);
    assert!(mol.atoms[0].n_core_ecp == 0, "H must have no ECP");
    assert!(mol.atoms[1].n_core_ecp > 0, "I must have an ECP");
    check_deriv_vs_fd(&mol, &bs, 1e-6, "HI/def2-SVP");
}

/// Translational invariance: rigidly translating the molecule cannot change
/// V_ECP, so `Σ_A dV_ECP/dR_A = 0` for every matrix element. This is an exact
/// identity independent of any finite-difference step, and it is INSENSITIVE to
/// a permutation of atom ids -- which is exactly why it is not sufficient on its
/// own and is paired with the FD tests above.
#[test]
fn ecp_matrix_deriv_sums_to_zero_over_atoms() {
    let bs = basis::bundled("def2-svp").unwrap();
    let mut mol = Molecule::parse_xyz("2\n\nH 0.0 0.0 0.0\nI 0.0 0.0 1.61\n", 0, 1).unwrap();
    mol.apply_ecp(&bs);
    let derivs = ecp_potential_deriv(&mol, &bs).unwrap().unwrap();

    for c in 0..3 {
        let mut sum: Array2<f64> = Array2::zeros(derivs[0][c].dim());
        for d in &derivs {
            sum += &d[c];
        }
        let worst = sum.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        eprintln!("HI: max |Σ_A dV_ECP/dR_A| coord {c} = {worst:.3e}");
        assert!(
            worst < 1e-10,
            "translational invariance violated on coord {c}: {worst:.3e}"
        );
    }
}

/// No ECP in the basis -> `Ok(None)`, zero work, mirroring `ecp_potential`.
#[test]
fn ecp_matrix_deriv_is_none_without_ecps() {
    let bs = basis::bundled("sto-3g").unwrap();
    let mol = Molecule::parse_xyz("1\n\nHe 0.0 0.0 0.0\n", 0, 1).unwrap();
    assert!(ecp_potential_deriv(&mol, &bs).unwrap().is_none());
}
