//! Weekly validation half of `pbc_kgrad.rs` (every test
//! `#[ignore = "validation: ..."]`): the tests whose per-commit CI time
//! was >= 120 s (mean of runs 40957025137 and 40953725119), moved so the
//! fast tier stays balanced over 8 shards. The module docs, hypotheses and
//! bars are in `pbc_kgrad.rs`; the shared helpers are
//! `pbc_kgrad_support/mod.rs`.
//!
//! Fast-tier canaries that keep these invariants per commit:
//! * `h2_1x1x3_force_is_the_supercell_force_on_one_copy` (dense supercell
//!   anchor + mesh mutants, non-TRIM)
//! * `h2_1x1x2_trim_mesh_is_blind_to_phase_mutants` (the TRIM blind spot shrunk
//!   to 1x1x2)
//! * `h3_uhf_1x1x3_rsgdf_force_is_the_supercell_rsgdf_force` (UHF 1x1x3
//!   supercell anchor, RS-GDF J/K)
//! * `closed_shell_kuhf_force_is_the_krhf_force`,
//!   `one_point_mesh_uhf_is_the_gamma_force` (dense k-UHF force)

mod common;
#[path = "pbc_kgrad_support/mod.rs"]
mod support;

use support::*;

#[test]
#[ignore = "validation: PBC k-point RHF force, H2 2x2x2 supercell anchor + TRIM blind spot; CI ~274 s"]
fn h2_2x2x2_force_is_the_supercell_force_and_trim_mesh_is_blind_to_phase_mutants() {
    let (ks, eri, scf, refs) =
        rhf_supercell_anchor("H2", h2_cell_g(), &pyscf_sto3g_h(), [2, 2, 2], None, SC_BAR);
    // BLIND SPOT (FINDINGS Iteration 21): every n_i <= 2 ⇒ every k is a
    // TRIM, D(k) is real and q ≡ −q, so ConjD and WrongQ are algebraic
    // identities. A port validated on such meshes has NOT tested its phases;
    // the 1×1×3 tests above are the ones that do.
    for exx in EXX {
        let f0 = kg_rhf(&ks, KGradJk::Dense(&eri), &scf, exx, None);
        for m in [KGradMutation::ConjD, KGradMutation::WrongQ] {
            let f = kg_rhf(&ks, KGradJk::Dense(&eri), &scf, exx, Some(m));
            let d = max_diff(&f.grad, &f0.grad);
            eprintln!("  H2 2x2x2 {exx:?} {m:?}: |F_mut − F| {d:.2e} (identity on TRIM meshes)");
            assert!(
                d <= IDENTITY_BAR,
                "{m:?} should be an identity on 2x2x2 ({d:e})"
            );
        }
    }
    // NoPhase and NoInvNk are NOT blind there (prototype 6.7e-2 / 3.4).
    for m in [KGradMutation::NoPhase, KGradMutation::NoInvNk] {
        let f = kg_rhf(&ks, KGradJk::Dense(&eri), &scf, ExxDiv::None, Some(m));
        let miss = sc_miss(&f.grad, &refs[0]);
        eprintln!("  H2 2x2x2 {m:?}: miss {miss:.2e}");
        assert!(miss > MUTANT_BAR, "{m:?} not caught on 2x2x2 ({miss:e})");
    }
}

#[test]
#[ignore = "validation: PBC k-point UHF force, dense H3 1x1x3 supercell anchor + mesh mutants; CI ~166 s"]
fn h3_uhf_1x1x3_force_is_the_supercell_force_on_one_copy() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h3_cell(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), 2, 1);
    eprintln!(
        "H3 UHF {n:?}: {} it, E {:.12}, nocc/k α {:?} β {:?}",
        scf.iterations, scf.energy, scf.nocc_per_k_alpha, scf.nocc_per_k_beta
    );
    // Supercell (6, 3): multiplicity 4.
    let g = gsys(
        supercell(&h3_cell(), n, 4),
        &bs,
        Some(ANCHOR_PRECISION),
        None,
    );
    let gscf = gamma_unrestricted(
        unfold(&ks.mesh, &scf.density_alpha),
        unfold(&ks.mesh, &scf.density_beta),
    );
    let mut refs: Vec<Array2<f64>> = Vec::new();
    let mut fks: Vec<Array2<f64>> = Vec::new();
    for exx in EXX {
        let fk = kg_uhf(&ks, KGradJk::Dense(&eri), &scf, exx, None);
        let fsc = gamma_grad(&g, &gscf, exx);
        let miss = sc_miss(&fk.grad, &fsc);
        report(&format!("H3 UHF {n:?} {exx:?}"), &fk);
        eprintln!("  H3 UHF {n:?} {exx:?}: supercell miss {miss:.2e} (prototype 7.2e-14)");
        assert!(miss <= SC_BAR, "{exx:?}: {miss:e}");
        assert!(fk.net_force <= NET_FORCE_BAR, "ΣF {:e}", fk.net_force);
        fks.push(fk.grad);
        refs.push(fsc);
    }
    let de = max_diff(&fks[0], &fks[1]);
    eprintln!("  H3 UHF: |F(ewald) − F(none)| {de:.2e}");
    assert!(de <= EWALD_NONE_BAR, "ewald vs none {de:e}");
    for m in MESH_MUTANTS {
        for (i, exx) in EXX.into_iter().enumerate() {
            let f = kg_uhf(&ks, KGradJk::Dense(&eri), &scf, exx, Some(m));
            let miss = sc_miss(&f.grad, &refs[i]);
            eprintln!(
                "  H3 UHF mutant {m:?} {exx:?}: miss {miss:.2e} (ΣF {:.1e})",
                f.net_force
            );
            assert!(
                miss > MUTANT_BAR,
                "mutant {m:?} {exx:?} not caught ({miss:e})"
            );
        }
    }
    let f = kg_uhf(
        &ks,
        KGradJk::Dense(&eri),
        &scf,
        ExxDiv::Ewald,
        Some(KGradMutation::GammaMadelung),
    );
    let miss = sc_miss(&f.grad, &refs[1]);
    eprintln!("  H3 UHF mutant GammaMadelung ewald: miss {miss:.2e}");
    assert!(miss > MUTANT_BAR, "GammaMadelung not caught ({miss:e})");
}
