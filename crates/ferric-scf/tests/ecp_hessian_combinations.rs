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
//! * `ecp_with_ri_and_environment_is_refused_by_the_probe`: typed refusal (RI + ECP
//!   alone is lifted: `ecp_df_hessian.rs`).
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
//!
//! # Tiers
//!
//! `two_ecp_elements_hessian_matches_fd_of_analytic_gradient` (synthetic
//! H-Br-I cation, def2-SVP; ~163 s on CI) runs weekly in
//! `validation_ecp_hessian_combinations.rs`. Its per-commit canary is
//! `two_ecp_elements_masked_stencil_matches_unmasked_and_is_translation_invariant`
//! (same system and SCF; the masked skeleton against the unmasked stencil at
//! 1e-10; it exercises the same per-ECP-centre fold, `hessian_ecp::
//! folded_first_derivative`, where the `ecp_atom[u] -> ecp_atom[0]` mutant
//! above lives — not re-mutation-tested when the FD half moved). The
//! environment FD combinations stay per commit below. Shared harness:
//! `ecp_hessian_combinations_support/mod.rs`.

#[path = "ecp_hessian_combinations_support/mod.rs"]
mod support;
use support::*;

#[test]
fn two_ecp_elements_masked_stencil_matches_unmasked_and_is_translation_invariant() {
    check_two_ecp_skeleton(false);
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
    // RI J/K + external potential is still refused ("untested") before any SCF,
    // as a typed error so `Auto` frequencies fall back to FD. RI + ECP alone
    // is analytic (`ecp_df_hessian.rs`).
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
}
