//! A ghost atom of an ECP element carries its basis functions and nothing else:
//! no nuclear charge, no core electrons removed, no ECP operator.
//!
//! Found by the smeltery #115 review (counterpoise with ghost heavy atoms):
//! `He + @I` at 1.5 Å sat ~1.9e-3 Ha ABOVE bare He, which a basis-only addition
//! cannot do variationally. The ghost's ECP was being added to `hcore`.
//!
//! Artifact hypothesis: if the ghost's ECP is still applied, the variational
//! anchor fails by ~1e-3 Ha and the ECP-free-basis comparison differs by the same
//! amount; if the fix is right both agree to SCF convergence (~1e-10 Ha).

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::{ecp_inputs_with_atoms, ecp_potential, ecp_potential_deriv};
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const Z_IODINE: i32 = 53;
const Z_XENON: i32 = 54;

const HE: &str = "1\nHe\nHe 0.0 0.0 0.0\n";
const HE_GHOST_I: &str = "2\nHe + ghost I\nHe 0.0 0.0 0.0\n@I 0.0 0.0 1.5\n";
const HI: &str = "2\nHI\nH 0.0 0.0 0.0\nI 0.0 0.0 1.61\n";
const HI_GHOST_I: &str = "3\nHI + ghost I\nH 0.0 0.0 0.0\nI 0.0 0.0 1.61\n@I 0.0 0.0 -1.5\n";
const HI_GHOST_XE: &str = "3\nHI + ghost Xe\nH 0.0 0.0 0.0\nI 0.0 0.0 1.61\n@Xe 0.0 0.0 4.3\n";

fn def2_svp() -> BasisSet {
    basis::bundled("def2-svp").unwrap()
}

/// def2-SVP with iodine's ECP deleted: the ghost's basis functions, and no ECP
/// code path at all for it. Only valid where no REAL iodine is present.
fn def2_svp_without_iodine_ecp() -> BasisSet {
    let mut bs = def2_svp();
    assert!(
        bs.ecps.remove(&Z_IODINE).is_some(),
        "def2-svp carries an I ECP"
    );
    bs
}

fn molecule(xyz: &str, bs: &BasisSet) -> Molecule {
    let mut mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    mol.apply_ecp(bs);
    mol
}

fn rhf_energy(xyz: &str, bs: &BasisSet) -> f64 {
    let mol = molecule(xyz, bs);
    let prep = PreparedBasis::new(&mol, bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let config = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
    assert!(res.converged, "RHF did not converge for {xyz}");
    res.energy
}

/// Variational anchor: a ghost adds basis functions, never operators, so the
/// energy can only go down.
#[test]
fn ghost_ecp_atom_cannot_raise_the_energy() {
    let bs = def2_svp();
    let e_he = rhf_energy(HE, &bs);
    let e_ghost = rhf_energy(HE_GHOST_I, &bs);
    eprintln!(
        "He {e_he:.12}  He+@I {e_ghost:.12}  diff {:+.3e}",
        e_ghost - e_he
    );
    assert!(
        e_ghost <= e_he + 1e-10,
        "He + @I = {e_ghost:.12} is above He = {e_he:.12} by {:.3e} Ha: the ghost changed \
         the Hamiltonian",
        e_ghost - e_he
    );
}

/// Exactness: He + @I equals the same calculation with iodine's ECP removed
/// from the basis set (an independent construction with no ECP code path).
#[test]
fn ghost_ecp_atom_matches_an_ecp_free_basis() {
    let e_with = rhf_energy(HE_GHOST_I, &def2_svp());
    let e_without = rhf_energy(HE_GHOST_I, &def2_svp_without_iodine_ecp());
    let diff = (e_with - e_without).abs();
    eprintln!("He+@I with I ECP {e_with:.12}  without {e_without:.12}  |diff| {diff:.3e}");
    assert_eq!(
        e_with.to_bits(),
        e_without.to_bits(),
        "ghost I's ECP still contributes {diff:.3e} Ha"
    );
}

/// Exactness with a NON-EMPTY ECP list: a real I (whose V_ECP stays on) plus a
/// ghost Xe, against the same run with ONLY Xe's ECP deleted from the basis.
/// Unlike the all-ghost case, both runs build a V_ECP here, so the ghost really
/// goes through the per-atom filter rather than the empty-list early return.
#[test]
fn ghost_ecp_atom_beside_a_real_ecp_atom_matches_a_basis_without_its_ecp() {
    let bs = def2_svp();
    let mut bs_no_xe = def2_svp();
    assert!(
        bs_no_xe.ecps.remove(&Z_XENON).is_some(),
        "def2-svp carries a Xe ECP"
    );
    let mol = molecule(HI_GHOST_XE, &bs);
    let inp = ecp_inputs_with_atoms(&mol, &bs_no_xe).expect("the real I carries an ECP");
    assert_eq!(inp.ecp_atom, vec![1]);

    let e_with = rhf_energy(HI_GHOST_XE, &bs);
    let e_without = rhf_energy(HI_GHOST_XE, &bs_no_xe);
    eprintln!(
        "HI+@Xe with Xe ECP {e_with:.12}  without {e_without:.12}  diff {:+.3e}",
        e_with - e_without
    );
    assert_eq!(
        e_with.to_bits(),
        e_without.to_bits(),
        "ghost Xe's ECP contributes {:+.3e} Ha",
        e_with - e_without
    );
}

/// Counterpoise with a REAL ECP atom of the same element present: the real I
/// keeps its ECP, the ghost I (1.5 Å off the H) gets none, and the energy is
/// still variational.
#[test]
fn ghost_and_real_ecp_atom_of_the_same_element() {
    let bs = def2_svp();
    let e_hi = rhf_energy(HI, &bs);
    let e_cp = rhf_energy(HI_GHOST_I, &bs);
    eprintln!("HI {e_hi:.12}  HI+@I {e_cp:.12}  diff {:+.3e}", e_cp - e_hi);
    assert!(
        e_cp <= e_hi + 1e-10,
        "HI + @I = {e_cp:.12} is above HI = {e_hi:.12}"
    );

    let mol = molecule(HI_GHOST_I, &bs);
    let inp = ecp_inputs_with_atoms(&mol, &bs).expect("the real I carries an ECP");
    assert_eq!(
        inp.ecp_atom,
        vec![1],
        "only the real I (atom 1) is an ECP centre"
    );
    assert_eq!(inp.ecps.len(), 1);
}

/// Electron bookkeeping: a ghost heavy atom removes no core electrons and adds
/// none, and freezes no core orbitals. Only the stored `n_core_ecp == 0` failed
/// before the fix; the `nelec` / frozen-core equalities are regression guards
/// (readers already zeroed a ghost through `effective_z` and the ghost filter).
#[test]
fn ghost_ecp_atom_leaves_the_electron_count_alone() {
    let bs = def2_svp();
    let he = molecule(HE, &bs);
    let ghost = molecule(HE_GHOST_I, &bs);
    assert_eq!(
        ghost.atoms[1].n_core_ecp, 0,
        "a ghost has no core to replace"
    );
    assert_eq!(ghost.atoms[1].effective_z(), 0);
    assert_eq!(ghost.nelec(), he.nelec());
    assert_eq!(ghost.nelec(), 2);
    assert_eq!(ghost.auto_frozen_core(), he.auto_frozen_core());

    let hi = molecule(HI, &bs);
    let hi_cp = molecule(HI_GHOST_I, &bs);
    assert_eq!(hi_cp.atoms[1].n_core_ecp, 28, "the real I keeps its core");
    assert_eq!(hi_cp.nelec(), hi.nelec());
    assert_eq!(hi_cp.auto_frozen_core(), hi.auto_frozen_core());
}

/// Ghost-only ECP elements: no ECP operator, so no ECP force on any atom.
#[test]
fn ghost_only_ecp_elements_give_no_ecp_operator_or_force() {
    let bs = def2_svp();
    let mol = molecule(HE_GHOST_I, &bs);
    assert!(ecp_potential(&mol, &bs).is_none(), "V_ECP must be absent");
    assert!(
        ecp_potential_deriv(&mol, &bs).unwrap().is_none(),
        "dV_ECP/dR must be absent"
    );
}

fn contract(d: &Array2<f64>, v: &Array2<f64>) -> f64 {
    (d * v).sum()
}

/// CONSISTENCY GUARD, not a regression test (it passed before the fix too, as
/// energy and derivative shared the same wrong inputs). The user-facing
/// gradients refuse ghosts, so `ecp_potential_deriv` is the only ghost-reachable
/// ECP derivative. Gradient consistency with a ghost present: the analytic `Σ D dV_ECP/dR` on
/// every atom (ghost included) matches a central finite difference of the
/// energy-path `V_ECP`. The ghost's row is non-zero (its basis functions move
/// through the REAL I's ECP) but carries no term from an ECP of its own.
#[test]
fn ecp_derivative_with_a_ghost_matches_finite_difference() {
    let bs = def2_svp();
    let mol = molecule(HI_GHOST_I, &bs);
    let v0 = ecp_potential(&mol, &bs).unwrap();
    let n = v0.nrows();
    let d = Array2::from_shape_fn((n, n), |(i, j)| {
        let (a, b) = (i.min(j) as f64, i.max(j) as f64);
        0.3 * (0.7 * a + 1.3 * b).sin() / (1.0 + (a - b).abs())
    });
    let derivs = ecp_potential_deriv(&mol, &bs).unwrap().unwrap();
    let h = 1e-4;
    let mut worst: f64 = 0.0;
    for (atom, blocks) in derivs.iter().enumerate() {
        for (c, block) in blocks.iter().enumerate() {
            let shifted = |s: f64| {
                let mut m = mol.clone();
                let a = &mut m.atoms[atom];
                match c {
                    0 => a.x += s,
                    1 => a.y += s,
                    _ => a.zpos += s,
                }
                contract(&d, &ecp_potential(&m, &bs).unwrap())
            };
            let fd = (shifted(h) - shifted(-h)) / (2.0 * h);
            let an = contract(&d, block);
            eprintln!("atom {atom} c {c}: analytic {an:+.10e}  fd {fd:+.10e}");
            worst = worst.max((an - fd).abs());
        }
    }
    assert!(
        worst < 1e-6,
        "analytic vs FD ECP derivative differs by {worst:.3e}"
    );
}
