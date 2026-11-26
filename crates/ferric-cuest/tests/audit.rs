use ferric_core::basis::bundled;
use ferric_cuest::audit::{audit_basis, audit_basis_with_aux, ViolationKind};

#[test]
fn audit_runs_on_all_bundled_bases() {
    let mut total_violations = 0usize;
    for name in ferric_core::basis::bundled_names() {
        let bs = bundled(name).expect(name);
        let a = audit_basis(&bs);
        if !a.is_ok() {
            total_violations += a.violations.len();
            eprintln!("basis {} violations: {:?}", name, a.violations);
        }
    }
    // This is a report, not a hard gate: print counts. The number of violating
    // bases is recorded for Phase 0.4 decision-making.
    eprintln!(
        "total violation entries across bundled bases: {}",
        total_violations
    );
}

#[test]
fn mixed_representation_detected() {
    // Build a basis by cloning a real one and flipping one shell's `pure`.
    let mut bs = bundled("cc-pvdz").unwrap(); // has high-l shells
                                              // flip a d-shell (l>=2) of the first element to create a mixed representation
    let z = *bs.shells.keys().next().unwrap();
    for sh in bs.shells.get_mut(&z).unwrap() {
        if sh.l >= 2 {
            sh.pure = !sh.pure;
            break;
        }
    }
    let a = audit_basis(&bs);
    assert!(a
        .violations
        .iter()
        .any(|v| v.kind == ViolationKind::MixedRepresentation));
}

#[test]
fn contiguous_l_no_violation_for_sto3g() {
    let bs = bundled("sto-3g").unwrap();
    let a = audit_basis(&bs);
    assert!(
        !a.violations
            .iter()
            .any(|v| v.kind == ViolationKind::LContiguityHole),
        "sto-3g should have contiguous L: {:?}",
        a.violations
    );
}

#[test]
fn aux_not_spherical_detected() {
    let primary = bundled("sto-3g").unwrap();
    let mut aux = bundled("sto-3g").unwrap();
    aux.name = "aux".into();
    let z = *aux.shells.keys().next().unwrap();
    aux.shells.get_mut(&z).unwrap()[0].pure = false;
    // make sure there is at least one pure to be a "mixed"? For aux rule, any cart shell violates.
    // ensure the shell we flipped is cart (done) ; but if all were cart originally, flipping one to? we want cart present.
    let a = audit_basis_with_aux(&primary, Some(&aux));
    assert!(a
        .violations
        .iter()
        .any(|v| v.kind == ViolationKind::AuxNotSpherical));
}
