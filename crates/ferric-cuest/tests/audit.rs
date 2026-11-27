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
    // Flip one l >= 2 shell of a real basis. `shells` is a HashMap and some
    // elements (H) have no d shell, so search every element and check a shell
    // actually changed.
    let mut bs = bundled("cc-pvdz").unwrap();
    let flipped = bs
        .shells
        .values_mut()
        .flat_map(|v| v.iter_mut())
        .find(|sh| sh.l >= 2)
        .map(|sh| sh.pure = !sh.pure)
        .is_some();
    assert!(flipped, "cc-pvdz has no l >= 2 shell to flip");
    let a = audit_basis(&bs);
    assert!(a
        .violations
        .iter()
        .any(|v| v.kind == ViolationKind::MixedRepresentation));
}

fn element_with(bs: &mut ferric_core::basis::BasisSet, z: i32, keep: impl Fn(i32) -> bool) {
    bs.shells.get_mut(&z).unwrap().retain(|sh| keep(sh.l));
}

/// The rule is per element: {s} on H and {p} on O is a hole on O even though
/// the union {s, p} is contiguous; {s} on H and {s, p} on O is fine.
#[test]
fn contiguity_is_checked_per_element() {
    let base = bundled("sto-3g").unwrap();
    let holes = |bs: &ferric_core::basis::BasisSet| {
        audit_basis(bs)
            .violations
            .into_iter()
            .filter(|v| v.kind == ViolationKind::LContiguityHole)
            .collect::<Vec<_>>()
    };
    let mut ok = base.clone();
    element_with(&mut ok, 1, |l| l == 0);
    element_with(&mut ok, 8, |l| l <= 1);
    assert!(holes(&ok).is_empty(), "{:?}", holes(&ok));

    let mut bad = base.clone();
    element_with(&mut bad, 1, |l| l == 0);
    element_with(&mut bad, 8, |l| l == 1);
    let h = holes(&bad);
    assert_eq!(h.len(), 1, "{h:?}");
    assert!(h[0].detail.contains("Z=8"), "{}", h[0].detail);
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
    let a = audit_basis_with_aux(&primary, Some(&aux));
    assert!(a
        .violations
        .iter()
        .any(|v| v.kind == ViolationKind::AuxNotSpherical));
}
