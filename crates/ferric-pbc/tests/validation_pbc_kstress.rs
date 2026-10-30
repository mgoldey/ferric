//! Weekly validation half of `pbc_kstress.rs` (every test
//! `#[ignore = "validation: ..."]`): the tests whose per-commit CI time
//! was >= 120 s (mean of runs 40957025137 and 40953725119), moved so the
//! fast tier stays balanced over 8 shards. The module docs, hypotheses and
//! bars are in `pbc_kstress.rs`; the shared helpers are
//! `pbc_kstress_support/mod.rs`.
//!
//! Fast-tier canaries that keep these invariants per commit:
//! * `h2_1x1x3_stress_matches_fd_of_own_k_energy_all_nine` (dense 1x1x3, non-
//!   TRIM, FD of the own energy)
//! * `h2_1x1x3_split_rsgdf_stress_is_the_supercell_stress_per_cell` (the same
//!   RS-GDF supercell anchor and k-fit mutants on the range-split fit)
//! * `h2_1x1x3_rsgdf_stress_matches_fd_of_own_k_energy_all_nine`
//! * `h2_mp_1x1x2_stress_matches_fd_of_own_k_energy_zz_and_xz` (the MP 1x1x2 FD
//!   shrunk to two strains)

mod common;
#[path = "pbc_kstress_support/mod.rs"]
mod support;

use support::*;

/// PHYSICS: a Gamma-centred k mesh IS the diag(N) supercell, so per cell
/// `dE/dε = dE/dε_sc/N` and `σ = σ_sc` (Ω_sc = N Ω), with the supercell's
/// Madelung constant as the exchange-divergence constant. ARTIFACT: a wrong
/// phase / q sign / 1/N_k / Madelung lattice would NOT reproduce an
/// explicit supercell. NON-TRIM mesh (n_2 = 3): ConjD and WrongQ are
/// identities on every n_i <= 2 mesh.
#[test]
#[ignore = "validation: PBC k-point RHF stress, dense H2 1x1x3 supercell anchor + k mutants; CI ~225 s"]
fn h2_1x1x3_stress_is_the_supercell_stress_per_cell() {
    let bs = pyscf_sto3g_h();
    let (ks, eri, scfs, refs) = rhf_supercell_anchor("H2", h2_cell_g(), &bs, [1, 1, 3]);
    assert_mutants_fail(
        "H2 1x1x3",
        &|exx, m| ks_rhf(&ks, &eri, &scfs[exx_idx(exx)], exx, m),
        &refs,
        &K_MUTANTS,
        &K_EWALD_MUTANTS,
    );
}

// ====================================================== MP mesh, p shells

/// PHYSICS: a Monkhorst-Pack even mesh (k = ±b3/4 for N = 2, residue modulus
/// 4) is a twisted-boundary supercell with NO explicit-supercell anchor, so
/// only the FD of ferric's own k energy pins it. Non-TRIM (2k ∉ G).
#[test]
#[ignore = "validation: PBC k-point RHF stress, Monkhorst-Pack 1x1x2 FD all nine; CI ~121 s"]
fn h2_mp_1x1x2_stress_matches_fd_of_own_k_energy_all_nine() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 2];
    let c = MeshCentring::MonkhorstPack;
    let ks = ksys_c(h2_cell_g(), &bs, n, c);
    assert_eq!(ks.mesh.residue_moduli(), [1, 1, 4]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scf = [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))),
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))),
    ];
    let e = |cell: &Cell| k_energies_c(cell, &bs, n, c);
    let fd1 = fd_all9(&ks.cell, FD_H, &e);
    let fd2 = fd_all9(&ks.cell, 2.0 * FD_H, &e);
    for (x, exx) in EXX.into_iter().enumerate() {
        let s = ks_rhf(&ks, &eri, &scf[x], exx, None);
        let (m1, m2) = (max_mat(&s.de_deps, &fd1[x]), max_mat(&s.de_deps, &fd2[x]));
        eprintln!("  H2 MP 1x1x2 {exx:?}: |analytic − FD| h {m1:.2e}, 2h {m2:.2e}");
        assert!(m1 <= K_FD_BAR, "{exx:?}: {m1:e}");
        assert!(s.antisymmetry() <= ANTISYM_BAR);
        for m in [KStressMutation::NoPhase, KStressMutation::NoInvNk] {
            let mm = max_mat(&ks_rhf(&ks, &eri, &scf[x], exx, Some(m)).de_deps, &fd1[x]);
            eprintln!("  H2 MP 1x1x2 {exx:?} mutant {m:?}: miss {mm:.2e}");
            assert!(mm > MUTANT_BAR, "{m:?} {exx:?}: {mm:e}");
        }
        for m in K_EWALD_MUTANTS {
            let mm = max_mat(&ks_rhf(&ks, &eri, &scf[x], exx, Some(m)).de_deps, &fd1[x]);
            eprintln!("  H2 MP 1x1x2 {exx:?} mutant {m:?}: miss {mm:.2e}");
            if exx == ExxDiv::Ewald {
                assert!(mm > MUTANT_BAR, "{m:?}: {mm:e}");
            }
        }
    }
}

/// PHYSICS: a Gamma-centred k mesh IS the diag(N) supercell, so per cell
/// `dE/dε = dE/dε_sc/N` with the RS-GDF fit built on either side. ARTIFACT: a
/// wrong phase / q sign / 1/N_k / Loewner per-q Wm / residue bin / conjugated
/// density would not reproduce an explicit supercell. NON-TRIM mesh (n_2 = 3):
/// ConjD and the q-sign defects are identities on every n_i <= 2 mesh. The k
/// defects here are the new risk of this module, so they run in CI.
#[test]
#[ignore = "validation: PBC k-point RHF stress, RS-GDF H2 1x1x3 supercell anchor + k-fit mutants; CI ~122 s"]
fn h2_1x1x3_rsgdf_stress_is_the_supercell_stress_per_cell() {
    let a = fit_supercell_anchor("H2", h2_cell_g(), &pyscf_sto3g_h(), [1, 1, 3]);
    let k_defects = [
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
        KStressMutation::NoInvNk,
        KStressMutation::FitAuxPhase,
        KStressMutation::FitLrWrongQ,
    ];
    for m in k_defects {
        let x = a.mutant(ExxDiv::None, m);
        eprintln!(
            "  H2 RS-GDF mutant {m:?}: miss {:.2e} (off-diagonal {:.2e})",
            x.total, x.off
        );
        assert!(x.total > MUTANT_BAR, "{m:?} not caught ({:e})", x.total);
    }
}
