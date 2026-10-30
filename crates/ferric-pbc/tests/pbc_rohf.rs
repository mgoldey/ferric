//! Stage 5b: Gamma-point ROHF / ROKS (`ferric_pbc::rohf::{gamma_rohf,
//! gamma_roks}` over `ferric_scf::rohf::solve_rohf_injected`), ported from
//! `reference/pbc/pbc_uks.py::roks` (FINDINGS "Iteration 10").
//!
//! Per spin σ (a = 1, V = 0 for ROHF), then the Guest-Saunders Roothaan
//! combination on one MO set:
//!
//! ```text
//! F_σ = h + J[D_α + D_β] − a·(K[D_σ] + v_M S D_σ S) + V_σ[ρ_α, ρ_β]
//! ```
//!
//! # What is independent of what
//!
//! * (1) Closed shell: ROHF ≡ Gamma RHF (`solve_rhf_injected`, a different
//!   solver: canonical orthogonalizer, RHF DIIS) for both exxdiv, 1e-12;
//!   ROKS(PBE0) ≡ `gamma_rks` on the same grid, 1e-11. ⟨S²⟩ = 0.
//! * (2) High spin: ROHF ≥ UHF (`gamma_uhf`, the Stage-4 path; the UHF
//!   variational space contains ROHF's). With N_β = 0 (H atom doublet, H2
//!   triplet) the two spaces COINCIDE, so ROHF == UHF == the FINDINGS
//!   It. 6 PySCF pins; on the tri 4H s+p triplet (3, 1) ROHF is strictly
//!   higher and pinned against PySCF 2.13 `pbc.scf.ROHF` (AFTDF 61³, the
//!   construction that matched our UHF to 1e-12). ⟨S²⟩ is S(S+1) from the
//!   MOs (UHF's is not). Plus the identity `E_ewald − E_none = −v_M N/2` at
//!   the same density and the per-spin gap identity `gap_ewald = gap_none +
//!   v_M` (exact for ROHF MOs: occupations are 0/1 per spin).
//! * (3) ROKS tri 4H s+p triplet LDA/PBE/PBE0 (ewald, staged): PySCF's A1
//!   (50, 146) grid is not reproducible here, so the tight pins are the
//!   PROTOTYPE'S OWN construction (`pbc_uks.roks` on
//!   `pbc_dft.PeriodicGrid(cell, 75, 302, D=10, scheme="ssf")`, dense pure
//!   AFT; `reference/pbc/run_roks_ssf_pins.py`, 2026-09-24); the PySCF
//!   `pbc.dft.ROKS` pins (FINDINGS It. 10) only to the grid-construction
//!   gap. ROKS ≥ UKS on the same grid (the pbc_uks.rs prototype pins).
//! * (4) `gamma_roks_with_xc` with an `a = 1`, zero-XC builder ≡
//!   `gamma_rohf` (1e-12): the `c_k = a` / `V_σ` plumbing at the HF limit.
//!
//! # Mutation plan (each must turn ≥ 1 test red)
//!
//! * `solve_rohf_impl`: `c_k = 0.5 * k_mix.sr` on the injected path → (1)
//!   PBE0, (3) PBE0, (4).
//! * `solve_rohf_impl`: injected K built from `D_total` for both spins →
//!   (2) tri (N_β > 0), (1) closed shell.
//! * `DenseAftEri::k_builder_with_madelung(0.5 v_M)` in `uhf::builders` →
//!   (2) ewald pins and the `−v_M N/2` identity.
//! * `rohf_occupation_gaps`: classify by sorted index / use `F_eff` → (2)
//!   gap identity (β open orbital is unoccupied for β).
//! * `gamma_roks_with_xc` not passing `xc` (ROHF run) → (3), (1) ROKS.
//!
//! # (5) Second-order solver (`RohfSolver::SecondOrder`, FINDINGS "Iteration
//! 24 (Python, second-order ROHF on the injected path) — 2026-09-27")
//!
//! * Exactness anchor: where DIIS converges (H3 doublet ROHF none/ewald, tri
//!   ROKS PBE0 none) the trust-region AH solver reaches the DIIS energy to
//!   1e-10 and the same per-spin densities, at a true minimum (lowest exact
//!   orbital-Hessian eigenvalue > 0).
//! * Target: tri ROHF triplet `exxdiv = none` from the core guess reaches
//!   PySCF's −0.581222768976 (1e-8) with λ_min in the prototype's +0.123
//!   basin (the second, trapped minimum has +0.030), and the staged ewald
//!   stage reaches −1.826096526949. The root cause is asserted directly: at
//!   that minimum a virtual lies BELOW the second open orbital in the
//!   Roothaan `F_eff` spectrum (prototype: 0.966763 < 1.049073).
//! * Negative control: DIIS alone does NOT reach it (the known failure).
//! * Artifact hypotheses (prototype A1–A5): a converged state ABOVE PySCF
//!   (second minimum, λ_min ≈ 0.03) or a saddle (λ_min < 0) would fail the
//!   λ_min window; a packed-scale gradient or a diagonal-Fock Hessian shows
//!   up as ρ ≠ 1 at the tail and in `ferric_scf::rohf_trah`'s FD unit tests.
//!
//! The tests whose per-commit CI time was >= 120 s
//! (`second_order_reaches_the_diis_state_on_tri_roks_pbe0_none`) run weekly
//! from `validation_pbc_rohf.rs`; the shared helpers are
//! `pbc_rohf_support/mod.rs`.

mod common;
#[path = "pbc_rohf_support/mod.rs"]
mod support;

use support::*;

// ===========================================================================
// (1) Closed shell: ROHF ≡ Gamma RHF, ROKS ≡ gamma_rks.
// ===========================================================================

#[test]
fn closed_shell_rohf_equals_gamma_rhf_both_exxdiv() {
    for (label, su) in [
        ("H2 a=4 STO-3G", setup(h2_cell(4.0), pyscf_sto3g_h())),
        ("tri 4H s+p", setup(triclinic_cell(), sp_basis_h())),
    ] {
        let vm = madelung_constant(&su.cell).unwrap();
        for exx in EXX {
            let applied = if exx == ExxDiv::Ewald { vm } else { 0.0 };
            let rhf = gamma_rhf_jk(
                &su.cell,
                &su.prep,
                &su.hc,
                Box::new(su.eri.j_builder()),
                Box::new(su.eri.k_builder_with_madelung(applied)),
            );
            for start in [EwaldStart::Staged, EwaldStart::Direct] {
                let ro = rohf(&su, exx, start);
                eprintln!(
                    "{label} {exx:?} {start:?}: E_rhf {:.14} E_rohf {:.14} dE {:.2e} <S2> {:.1e}",
                    rhf.energy,
                    ro.scf.energy,
                    ro.scf.energy - rhf.energy,
                    ro.s2
                );
                close(
                    ro.scf.energy,
                    rhf.energy,
                    1e-12,
                    &format!("{label} {exx:?} ROHF vs RHF"),
                );
                assert!(
                    max_abs_diff(&ro.scf.density_total, &rhf.density_total) < 1e-7,
                    "{label} {exx:?}: density"
                );
                assert!(ro.s2.abs() < 1e-12, "{label}: <S2> {}", ro.s2);
                assert_eq!(ro.nocc.0, ro.nocc.1);
                assert!(ro.gaps.satisfied(), "{label} {exx:?}: {:?}", ro.gaps);
                assert_eq!(
                    ro.none_stage.is_some(),
                    exx == ExxDiv::Ewald && start == EwaldStart::Staged
                );
            }
        }
    }
}

#[test]
fn closed_shell_roks_pbe0_equals_gamma_rks() {
    let su = setup(h2_cell(4.0), pyscf_sto3g_h());
    let grid = ssf_grid(30, 50, 10.0);
    let vm = madelung_constant(&su.cell).unwrap();
    let n = su.cell.mol().nelec() as f64;
    for exx in EXX {
        let mut rcfg = GammaRksConfig::new("PBE0");
        rcfg.grid = grid.clone();
        rcfg.exxdiv = exx;
        let r = gamma_rks(&su.cell, &su.prep, &su.hc, ints(&su), &rcfg).unwrap();
        let o = roks(&su, "PBE0", grid.clone(), exx);
        close(
            o.scf.energy,
            r.scf.energy,
            1e-11,
            &format!("H2 PBE0 {exx:?} ROKS vs RKS"),
        );
        assert!((o.e_xc - r.e_xc).abs() < 1e-10, "E_xc");
        assert!(o.s2.abs() < 1e-12);
        assert_eq!(o.exact_exchange_fraction, 0.25);
        if exx == ExxDiv::Ewald {
            let none = o.none_stage.as_ref().expect("staged for a > 0");
            close(
                o.scf.energy - none.energy,
                -0.25 * vm * n / 2.0,
                1e-10,
                "ROKS PBE0 ewald - none (= -a v_M N/2)",
            );
        }
    }
    // a = 0: no K, so no staging.
    assert!(roks(&su, "PBE", grid, ExxDiv::Ewald).none_stage.is_none());
}

// ===========================================================================
// (2) High spin: ROHF ≥ UHF, spin purity, pins, ewald identities.
// ===========================================================================

#[test]
fn high_spin_rohf_bounds_uhf_and_is_spin_pure() {
    type Row = (
        &'static str,
        Cell,
        fn() -> BasisSet,
        (usize, usize),
        [f64; 2],
    );
    let rows: [Row; 2] = [
        (
            "H atom a=4 doublet",
            cell_mult(&H_ATOM, cubic(4.0), 2),
            pyscf_sto3g_h,
            (1, 0),
            H_UHF_E,
        ),
        (
            "H2 a=4 triplet",
            cell_mult(&H2_ATOMS, cubic(4.0), 3),
            pyscf_sto3g_h,
            (2, 0),
            H2T_UHF_E,
        ),
    ];
    for (label, cell, basis, nocc, pins) in rows {
        let su = setup(cell, basis());
        let vm = madelung_constant(&su.cell).unwrap();
        let n = (nocc.0 + nocc.1) as f64;
        for (i, exx) in EXX.into_iter().enumerate() {
            let ro = rohf(&su, exx, EwaldStart::Staged);
            let u = gamma_uhf(
                &su.cell,
                &su.prep,
                &su.hc,
                ints(&su),
                &GammaUhfConfig {
                    exxdiv: exx,
                    ..Default::default()
                },
            )
            .unwrap();
            let de = ro.scf.energy - u.scf.energy;
            eprintln!(
                "{label} {exx:?}: E_rohf {:.12} <S2> {:.12} | E_uhf {:.12} <S2> {:.10} | \
                 E_rohf - E_uhf {de:.3e} gaps {:?}",
                ro.scf.energy, ro.s2, u.scf.energy, u.s2, ro.gaps
            );
            assert_eq!(ro.nocc, nocc);
            // Variational bound: the UHF space contains the ROHF space.
            assert!(de > -1e-10, "{label} {exx:?}: ROHF below UHF by {de:.3e}");
            if nocc.1 == 0 {
                // No beta electrons: the two spaces coincide.
                assert!(de.abs() < 1e-10, "{label} {exx:?}: N_b = 0 but dE {de:.3e}");
            } else {
                assert!(de > 1e-6, "{label} {exx:?}: ROHF not above UHF ({de:.3e})");
                assert!(u.s2 - s_s1(nocc.0, nocc.1) > 1e-4, "UHF is contaminated");
            }
            close(
                ro.scf.energy,
                pins[i],
                1e-9,
                &format!("{label} {exx:?} vs PySCF"),
            );
            // Spin purity from the MOs (exact by construction).
            close(ro.s2, s_s1(nocc.0, nocc.1), 1e-12, &format!("{label} <S2>"));
            assert!(ro.gaps.satisfied(), "{label} {exx:?}: {:?}", ro.gaps);
            if exx == ExxDiv::Ewald {
                let none = ro.none_stage.as_ref().expect("staged for HF");
                close(
                    ro.scf.energy - none.energy,
                    -vm * n / 2.0,
                    1e-10,
                    &format!("{label} ewald - none (= -v_M N/2)"),
                );
                // Per-spin gap identity (occupations are 0/1 per spin).
                let g0 = rohf_occupation_gaps(none, &su.hc.s, 0.0);
                for (ge, gn, spin) in [
                    (ro.gaps.gap_alpha, g0.gap_alpha, "alpha"),
                    (ro.gaps.gap_beta, g0.gap_beta, "beta"),
                ] {
                    match (ge, gn) {
                        (Some(ge), Some(gn)) => close(
                            ge - gn,
                            vm,
                            1e-7,
                            &format!("{label} {spin} gap_ewald - gap_none (= v_M)"),
                        ),
                        // No occupied or no unoccupied level of this spin
                        // (e.g. H2/STO-3G triplet: alpha fills both AOs).
                        (None, None) => {}
                        other => panic!("{label} {spin}: gap presence differs {other:?}"),
                    }
                }
            }
        }
    }
}

/// The ROHF Ewald trap: a single-stage ewald run from the core guess may land
/// in a hole state (the DIIS prototype does, at `TRI_ROHF_EWALD_TRAPPED`,
/// which Iteration 24 showed is a genuine second minimum, λ_min +0.030,
/// α occupation-aware gap −0.0093 at none). Run on the SECOND-ORDER solver:
/// ferric's DIIS ROHF does not converge this system at all (FINDINGS
/// Iteration 24: PySCF's minimum is not an `F_eff`-aufbau fixed point; see
/// `diis_alone_does_not_converge_tri_rohf_none`). Whether the Direct
/// trajectory reaches the trapped minimum depends on the solver path (the
/// prototype's TRAH reached PySCF's), so the assertion is conditional: a
/// Direct run ABOVE the staged state must be flagged by the
/// occupation-aware gap, and the staged run is never flagged.
#[test]
fn rohf_ewald_trap_is_flagged_when_hit() {
    let su = setup(cell_mult(&TRI_ATOMS, TRI_A, 3), sp_basis_h());
    let staged = rohf_so(&su, ExxDiv::Ewald, EwaldStart::Staged);
    let direct = rohf_so(&su, ExxDiv::Ewald, EwaldStart::Direct);
    let de = direct.scf.energy - staged.scf.energy;
    eprintln!(
        "tri ROHF ewald: staged {:.12} gaps {:?} | direct {:.12} (dE {de:.3e}) gaps {:?}",
        staged.scf.energy, staged.gaps, direct.scf.energy, direct.gaps
    );
    assert!(staged.gaps.satisfied(), "staged: {:?}", staged.gaps);
    close(
        staged.scf.energy,
        TRI_ROHF_PYSCF_E[1],
        1e-9,
        "staged vs PySCF",
    );
    if de > 1e-6 {
        assert!(
            !direct.gaps.satisfied(),
            "direct run landed {de:.3e} above the staged state without a gap flag: {:?}",
            direct.gaps
        );
        if (direct.scf.energy - TRI_ROHF_EWALD_TRAPPED).abs() < 1e-8 {
            eprintln!("  direct run reached the prototype's trapped state");
        }
    } else {
        assert!(de.abs() < 1e-9, "direct below staged by {de:.3e}");
    }
}

// ===========================================================================
// (3) ROKS tri 4H s+p triplet vs the prototype construction and PySCF.
// ===========================================================================

#[test]
fn roks_tri_triplet_matches_the_prototype_construction_and_pyscf() {
    let su = setup(cell_mult(&TRI_ATOMS, TRI_A, 3), sp_basis_h());
    let vm = madelung_constant(&su.cell).unwrap();
    for (i, xc) in ["LDA", "PBE", "PBE0"].into_iter().enumerate() {
        let o = roks(&su, xc, ssf_grid(75, 302, 10.0), ExxDiv::Ewald);
        eprintln!(
            "tri ROKS {xc}: E {:.12} <S2> {:.12} a {} gaps {:?} grid {:?}",
            o.scf.energy, o.s2, o.exact_exchange_fraction, o.gaps, o.grid
        );
        assert_eq!(o.nocc, (3, 1));
        assert_eq!(o.grid.as_ref().unwrap().n_grid_points, TRI_SSF_NPTS);
        close(
            o.scf.energy,
            TRI_ROKS_SSF_E[i],
            1e-8,
            &format!("tri ROKS {xc} vs prototype"),
        );
        close(o.s2, 2.0, 1e-12, &format!("tri ROKS {xc} <S2>"));
        let d_py = o.scf.energy - TRI_ROKS_PYSCF_E[i];
        eprintln!("  vs PySCF ROKS A1 (50,146): {d_py:+.3e}");
        assert!(d_py.abs() < GRID_CONSTRUCTION_GAP, "{xc}: {d_py:.3e}");
        // ROKS above UKS on the identical grid (variational bound).
        assert!(
            o.scf.energy - TRI_UKS_SSF_E[i] > 1e-6,
            "{xc}: ROKS not above UKS"
        );
        assert!(o.gaps.satisfied(), "{xc}: {:?}", o.gaps);
        let a = o.exact_exchange_fraction;
        if let Some(none) = o.none_stage.as_ref() {
            close(
                o.scf.energy - none.energy,
                -a * vm * 4.0 / 2.0,
                1e-10,
                &format!("tri ROKS {xc} ewald - none"),
            );
        } else {
            assert_eq!(a, 0.0);
        }
    }
}

/// `GammaRoksConfig::new` sets the ramped level shift and the 600-iteration
/// cap for a hybrid only (FINDINGS 2026-09-25); LDA/GGA and unresolvable
/// names keep the `GammaUksConfig::new` SCF defaults.
#[test]
fn roks_config_new_shifts_hybrids_only() {
    let uks = GammaUksConfig::new("PBE0").scf;
    for (name, shift, cap) in [
        ("PBE0", ROKS_HYBRID_LEVEL_SHIFT, ROKS_HYBRID_MAX_ITER),
        ("LDA", uks.level_shift, uks.max_iter),
        ("PBE", uks.level_shift, uks.max_iter),
        ("unused", uks.level_shift, uks.max_iter),
    ] {
        let c = GammaRoksConfig::new(name).scf;
        assert_eq!((c.level_shift, c.max_iter), (shift, cap), "{name}");
        assert_eq!(c.density_conv, uks.density_conv, "{name}");
        assert!(!c.use_sad_guess, "{name}");
    }
    assert_eq!((ROKS_HYBRID_LEVEL_SHIFT, ROKS_HYBRID_MAX_ITER), (0.05, 600));
    assert_eq!((uks.level_shift, uks.max_iter), (0.0, 200));
}

/// Regression for the CI failure (FINDINGS "ROKS PBE0 CI non-convergence
/// (Python diagnosis) — 2026-09-25"): with the `GammaRoksConfig::new` hybrid
/// defaults (ramped level shift 0.05, 600 iterations) every perturbed start
/// reaches the prototype pin. Its negative control is
/// `roks_pbe0_tri_none_stage_perturbed_starts_fail_without_the_shift`.
#[test]
#[ignore = "slow: 6 PBE0 ROKS SCFs of ~100-500 iterations each on the 70784-point SSF grid; run with --ignored"]
fn roks_pbe0_tri_none_stage_converges_from_perturbed_starts() {
    let runs = pbe0_none_from_perturbed_starts(|_| {});
    for (t, seed, res) in &runs {
        match res {
            Ok((e, _)) => close(
                *e,
                TRI_ROKS_PBE0_NONE,
                1e-8,
                &format!("PBE0 none t {t:.0e} seed {seed}"),
            ),
            Err(why) => panic!("PBE0 none t {t:.0e} seed {seed}: {why}"),
        }
    }
}

/// Negative control: the SAME starts with the pre-2026-09-25 defaults (no
/// shift, 200 iterations). At least one must fail to reach the pin, otherwise
/// the positive test above does not show that the shift is what makes it
/// pass (the replica fails 2 of 3 starts at t = 1e-6 without the shift; this
/// is not guaranteed under ferric's arithmetic, hence the explicit check).
#[test]
#[ignore = "slow: 6 PBE0 ROKS SCFs of up to 200 iterations each on the 70784-point SSF grid; run with --ignored"]
fn roks_pbe0_tri_none_stage_perturbed_starts_fail_without_the_shift() {
    let runs = pbe0_none_from_perturbed_starts(|scf| {
        scf.level_shift = 0.0;
        scf.max_iter = 200;
    });
    let failed = runs
        .iter()
        .filter(|(_, _, r)| !matches!(r, Ok((e, _)) if (e - TRI_ROKS_PBE0_NONE).abs() < 1e-8))
        .count();
    eprintln!(
        "  unshifted: {failed} of {} starts miss the pin",
        runs.len()
    );
    assert!(
        failed >= 1,
        "every perturbed start reached the pin WITHOUT the level shift, so the shifted \
         regression test does not isolate the shift; use stronger perturbations"
    );
}

// ===========================================================================
// (4) a = 1, zero XC ≡ gamma_rohf.
// ===========================================================================

/// Both sides on the second-order solver (ferric's DIIS ROHF does not
/// converge the tri triplet: FINDINGS Iteration 24). With `a = 1` and a zero
/// XC builder the second-order trajectory is the ROHF one operation for
/// operation (the FD XC response adds exact zeros), hence 1e-12.
#[test]
fn roks_with_an_hf_equivalent_builder_is_gamma_rohf() {
    let su = setup(cell_mult(&TRI_ATOMS, TRI_A, 3), sp_basis_h());
    for exx in EXX {
        let h = rohf_so(&su, exx, EwaldStart::Staged);
        let cfg = GammaRoksConfig {
            exxdiv: exx,
            solver: second_order(false),
            ..GammaRoksConfig::new("unused")
        };
        let k = gamma_roks_with_xc(
            &su.cell,
            &su.prep,
            &su.hc,
            ints(&su),
            &mut ZeroXc { frac: 1.0 },
            &cfg,
        )
        .unwrap();
        close(
            k.scf.energy,
            h.scf.energy,
            1e-12,
            &format!("tri {exx:?} ROKS(a=1, xc=0) vs ROHF"),
        );
        assert_eq!(k.e_xc, 0.0);
        assert_eq!(k.none_stage.is_some(), h.none_stage.is_some());
        assert!(k.second_order.as_ref().is_some_and(|i| i.converged));
    }
}

// ===========================================================================
// (5) Second-order solver (RohfSolver::SecondOrder).
// ===========================================================================

/// DIIS stays the default everywhere, and a DIIS run reports no
/// second-order diagnostics.
#[test]
fn diis_is_the_default_rohf_solver() {
    assert_eq!(GammaRohfConfig::default().solver, RohfSolver::Diis);
    for name in ["PBE0", "PBE", "LDA", "unused"] {
        assert_eq!(
            GammaRoksConfig::new(name).solver,
            RohfSolver::Diis,
            "{name}"
        );
    }
    let su = setup(cell_mult(&H3_ATOMS, cubic(H3_A), 2), pyscf_sto3g_h());
    assert!(rohf(&su, ExxDiv::None, EwaldStart::Staged)
        .second_order
        .is_none());
}

/// EXACTNESS ANCHOR (H3 doublet ROHF, where DIIS converges): the
/// second-order solver from the same core guess reaches the DIIS energy to
/// 1e-10, the same per-spin densities, and a true minimum (prototype λ_min
/// +0.4931 for none and ewald).
#[test]
fn second_order_reaches_the_diis_state_on_h3_rohf() {
    let su = setup(cell_mult(&H3_ATOMS, cubic(H3_A), 2), pyscf_sto3g_h());
    let vm = madelung_constant(&su.cell).unwrap();
    for exx in EXX {
        let d = rohf(&su, exx, EwaldStart::Staged);
        let s = rohf_so(&su, exx, EwaldStart::Staged);
        close(
            s.scf.energy,
            d.scf.energy,
            1e-10,
            &format!("H3 ROHF {exx:?} second order vs DIIS"),
        );
        same_state(&s.scf, &d.scf, &format!("H3 ROHF {exx:?}"));
        let info = s.second_order.as_ref().unwrap();
        let lam = info.lowest_hessian_eigenvalue.expect("lambda_min reported");
        assert!(lam > 0.0, "H3 {exx:?}: not a minimum, lambda_min {lam:e}");
        close(s.s2, 0.75, 1e-12, "H3 <S2>");
        assert_eq!(s.none_stage.is_some(), d.none_stage.is_some());
        if let Some(none) = s.none_stage.as_ref() {
            close(
                s.scf.energy - none.energy,
                -vm * 3.0 / 2.0,
                1e-10,
                "H3 second order ewald - none (= -v_M N/2)",
            );
        }
    }
}

/// Per-commit canary of `validation_pbc_rohf.rs`'s
/// `second_order_reaches_the_diis_state_on_tri_roks_pbe0_none`: the same
/// exactness anchor with a REAL XC builder (the second-order solver's FD XC
/// response), shrunk to the H3 doublet / STO-3G on a coarse (30, 110) grid.
/// DIIS converges there, so the second-order solver must reach the DIIS
/// energy (1e-10) and per-spin densities at a true minimum (λ_min > 0).
#[test]
fn second_order_reaches_the_diis_state_on_h3_roks_pbe0_none() {
    let su = setup(cell_mult(&H3_ATOMS, cubic(H3_A), 2), pyscf_sto3g_h());
    let gcfg = ssf_grid(30, 110, 10.0);
    let grid = PeriodicGrid::build(&su.cell, &gcfg).expect("grid");
    let mut pxc = PeriodicXc::new(
        &su.cell,
        su.prep.basis_set(),
        "PBE0",
        &grid,
        &PeriodicXcConfig::default(),
    )
    .expect("PeriodicXc PBE0");
    let base = GammaRoksConfig {
        exxdiv: ExxDiv::None,
        grid: gcfg,
        ..GammaRoksConfig::new("PBE0")
    };
    let d = gamma_roks_with_xc(&su.cell, &su.prep, &su.hc, ints(&su), &mut pxc, &base)
        .expect("DIIS PBE0 none");
    let so_cfg = GammaRoksConfig {
        solver: second_order(true),
        ..base.clone()
    };
    let s = gamma_roks_with_xc(&su.cell, &su.prep, &su.hc, ints(&su), &mut pxc, &so_cfg)
        .expect("second-order PBE0 none");
    let info = s.second_order.as_ref().expect("second-order diagnostics");
    eprintln!("  H3 PBE0 none second order: {info:?}");
    assert!(info.converged);
    close(s.scf.energy, d.scf.energy, 1e-10, "H3 PBE0 none SO vs DIIS");
    same_state(&s.scf, &d.scf, "H3 PBE0 none");
    let lam = info.lowest_hessian_eigenvalue.expect("lambda_min reported");
    assert!(lam > 0.0, "H3 PBE0 none: not a minimum, lambda_min {lam:e}");
    close(s.s2, 0.75, 1e-12, "H3 PBE0 <S2>");
}

/// TARGET (the open item): tri ROHF triplet `exxdiv = none` from the core
/// guess reaches PySCF, at the prototype's minimum (λ_min window), with a
/// virtual BELOW the second open orbital in the `F_eff` spectrum — the reason
/// the aufbau DIIS loop cannot hold it. Then the staged ewald run reaches
/// PySCF's ewald energy in a few macro iterations, `−v_M N/2` below.
#[test]
fn second_order_converges_tri_rohf_to_pyscf() {
    let su = setup(cell_mult(&TRI_ATOMS, TRI_A, 3), sp_basis_h());
    let vm = madelung_constant(&su.cell).unwrap();
    let none = rohf_so(&su, ExxDiv::None, EwaldStart::Direct);
    close(
        none.scf.energy,
        TRI_ROHF_PYSCF_E[0],
        1e-8,
        "tri ROHF none (core guess) vs PySCF",
    );
    let info = none.second_order.as_ref().unwrap();
    let lam = info.lowest_hessian_eigenvalue.expect("lambda_min reported");
    eprintln!("  tri ROHF none lambda_min {lam:.6} (prototype +0.1231)");
    assert!(
        lam > TRI_ROHF_LAMBDA_MIN_WINDOW.0 && lam < TRI_ROHF_LAMBDA_MIN_WINDOW.1,
        "tri ROHF none: lambda_min {lam} outside {TRI_ROHF_LAMBDA_MIN_WINDOW:?}"
    );
    let rho = info.last_rho.expect("at least one step");
    assert!((rho - 1.0).abs() < 0.2, "tail rho {rho} is not ~1");
    close(none.s2, 2.0, 1e-12, "tri ROHF <S2>");
    assert!(none.gaps.satisfied(), "{:?}", none.gaps);
    // eps_alpha: closed | open | virtual (semicanonical F_eff per block).
    let eps = &none.scf.eps_alpha;
    let open_max = eps[1].max(eps[2]);
    let virt_min = eps[3..].iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!("  F_eff: open max {open_max:.6}, virtual min {virt_min:.6}");
    assert!(
        virt_min < open_max,
        "expected a virtual below the second open orbital in F_eff at PySCF's none minimum          (prototype 0.966763 < 1.049073), got {virt_min} vs {open_max}"
    );

    let ew = rohf_so(&su, ExxDiv::Ewald, EwaldStart::Staged);
    close(
        ew.scf.energy,
        TRI_ROHF_PYSCF_E[1],
        1e-8,
        "tri ROHF ewald (staged) vs PySCF",
    );
    let stage0 = ew.none_stage.as_ref().expect("staged");
    close(stage0.energy, none.scf.energy, 1e-10, "staged none stage");
    close(
        ew.scf.energy - stage0.energy,
        -vm * 4.0 / 2.0,
        1e-10,
        "tri ROHF ewald - none (= -v_M N/2)",
    );
    let einfo = ew.second_order.as_ref().unwrap();
    assert!(
        einfo.macro_iterations <= 5,
        "ewald stage from the none MOs took {} macro iterations (prototype 1)",
        einfo.macro_iterations
    );
    let lam = einfo
        .lowest_hessian_eigenvalue
        .expect("lambda_min reported");
    assert!(
        lam > TRI_ROHF_LAMBDA_MIN_WINDOW.0 && lam < TRI_ROHF_LAMBDA_MIN_WINDOW.1,
        "tri ROHF ewald: lambda_min {lam}"
    );
    assert!(ew.gaps.satisfied(), "{:?}", ew.gaps);
}

/// NEGATIVE CONTROL for the target: ferric's DIIS loop (defaults) does NOT
/// reach PySCF's tri ROHF none minimum — it errors (not converged) or lands
/// elsewhere. If this ever passes by converging to PySCF, the target test no
/// longer isolates the second-order solver.
#[test]
fn diis_alone_does_not_converge_tri_rohf_none() {
    let su = setup(cell_mult(&TRI_ATOMS, TRI_A, 3), sp_basis_h());
    let cfg = GammaRohfConfig {
        exxdiv: ExxDiv::None,
        ..Default::default()
    };
    assert_eq!(cfg.solver, RohfSolver::Diis);
    match gamma_rohf(&su.cell, &su.prep, &su.hc, ints(&su), &cfg) {
        Err(e) => eprintln!("  DIIS tri ROHF none: {e}"),
        Ok(r) => {
            let de = r.scf.energy - TRI_ROHF_PYSCF_E[0];
            eprintln!(
                "  DIIS tri ROHF none: converged {} E {:.12} (E - PySCF {de:+.3e})",
                r.scf.converged, r.scf.energy
            );
            assert!(
                !r.scf.converged || de.abs() > 1e-6,
                "DIIS reached PySCF's tri ROHF none minimum: the negative control is void"
            );
        }
    }
}

// ===========================================================================
// Refusals.
// ===========================================================================

#[test]
fn gamma_rohf_and_roks_refuse_unsupported_features_by_name() {
    let su = setup(cell_mult(&H_ATOM, cubic(4.0), 2), pyscf_sto3g_h());
    let rohf_cases: [(&str, &dyn Fn(&mut GammaRohfConfig)); 6] = [
        ("RhfConfig.xc", &|c| c.scf.xc = Some("PBE".into())),
        ("RhfConfig.ah_trigger", &|c| c.scf.ah_trigger = 1e-2),
        ("RhfConfig.newton_trigger", &|c| c.scf.newton_trigger = 1e-2),
        ("RhfConfig.scf_stability_descent", &|c| {
            c.scf.scf_stability_descent = true
        }),
        ("RhfConfig.check_stability", &|c| {
            c.scf.check_stability = true
        }),
        ("RhfConfig.use_sad_guess", &|c| c.scf.use_sad_guess = true),
    ];
    for (word, edit) in rohf_cases {
        let mut cfg = GammaRohfConfig::default();
        edit(&mut cfg);
        let e = gamma_rohf(&su.cell, &su.prep, &su.hc, ints(&su), &cfg)
            .unwrap_err()
            .to_string();
        assert!(e.contains(word), "{word}: {e}");
    }
    let roks_cases: [(&str, &dyn Fn(&mut GammaRoksConfig)); 6] = [
        ("RhfConfig.ah_trigger", &|c| c.scf.ah_trigger = 1e-2),
        ("scf.xc_omega", &|c| c.scf.xc_omega = Some(0.3)),
        ("scf.dft_grid", &|c| {
            c.scf.dft_grid = Some(ferric_dft::grid::AtomicGridConfig::default())
        }),
        ("VV10", &|c| c.functional = "wB97X-V".into()),
        ("range-separated hybrid", &|c| c.functional = "HSE06".into()),
        ("TPSS", &|c| c.functional = "TPSS".into()),
    ];
    for (word, edit) in roks_cases {
        let mut cfg = GammaRoksConfig::new("PBE");
        cfg.grid = ssf_grid(30, 50, 10.0);
        edit(&mut cfg);
        let e = gamma_roks(&su.cell, &su.prep, &su.hc, ints(&su), &cfg)
            .unwrap_err()
            .to_string();
        assert!(e.contains(word), "{word}: {e}");
    }
    let cfg = GammaRoksConfig::new("unused");
    let e = gamma_roks_with_xc(
        &su.cell,
        &su.prep,
        &su.hc,
        ints(&su),
        &mut ZeroXc { frac: 1.5 },
        &cfg,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("exact-exchange fraction"), "{e}");
    let e = gamma_roks_with_xc(
        &su.cell,
        &su.prep,
        &su.hc,
        ints(&su),
        &mut ClosedOnlyXc,
        &cfg,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("closed-shell XcBuilder"), "{e}");
    // Wrong-shape initial MOs are a named error.
    let n = su.prep.nbasis();
    let bad = GammaRohfConfig {
        initial_mos: Some(Array2::zeros((n + 1, n))),
        ..Default::default()
    };
    let e = gamma_rohf(&su.cell, &su.prep, &su.hc, ints(&su), &bad)
        .unwrap_err()
        .to_string();
    assert!(e.contains("initial MO shape"), "{e}");
}
