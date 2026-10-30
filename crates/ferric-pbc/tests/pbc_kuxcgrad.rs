//! Analytic forces of the k-point open-shell KS energy (`ferric_pbc::kuxcgrad`,
//! module doc) for LDA, GGA and global hybrids.
//!
//! Anchors, each with its artifact hypothesis next to the physics hypothesis:
//! a = 1 with a zero XC == the k-point UHF force per part; the closed-shell
//! limit == the merged RKS force per part (BLIND to every spin swap or
//! one-spin-dropped defect: both spins are equal there); 1x1x1 and the
//! non-TRIM 1x1x3 supercell identities against the Gamma UKS gradient per
//! part on a genuinely open-shell cell (H3 doublet, Da != Db); central FD of
//! the energy (fresh SCF per displacement) at a verified safe step with an
//! explicit shared neighbour cutoff; thread bit-identity; the per-spin Fock
//! guard; typed refusals; production-wired spin-specific mutants.
//!
//! NOT covered: p/d shells and heavy atoms / ECP, shifted meshes, metals,
//! range-separated hybrids / VV10 (refused), meta-GGA (`pbc_kuxc_mgga.rs`), the production
//! (75, 302) grid (a (40, 110) FD is in the ignored tier), stress.
//!
//! WEEKLY HALF: the two tests whose CI time is >= 120 s (the PBE0 1x1x1 vs
//! Gamma UKS anchor with its spin / scaling mutants, and the full refusals /
//! per-spin Fock guard sweep) run in `validation_pbc_kuxcgrad.rs`; the
//! helpers are `kuxcgrad_support/mod.rs`. Per-commit canaries for them:
//! `one_point_mesh_matches_the_gamma_uks_gradient_per_part_pbe` (the same
//! per-part 1x1x1 vs Gamma anchor, PBE: the polarized semilocal XC force,
//! no mutants), `one_point_mesh_matches_the_gamma_uks_gradient_per_part_pbe0`
//! (the polarized PBE0 row itself, exact exchange included, no mutants: its
//! `XcSpinSwap` / `PulayKSwap` / `ExchScaledTwice` checks are weekly only) and
//! `refusals_and_the_per_spin_fock_guard_canary` (named, exxdiv and per-spin
//! tamper refusals). All are 1x1x1: blind to the Bloch phases.

mod common;
#[path = "kuxcgrad_support/mod.rs"]
mod support;

use common::*;
use support::*;

/// (a) EXACTNESS ANCHOR. a = 1 with a zero XC IS the k-point UHF force: same
/// `assemble(Unrestricted)`, but the KS entry rebuilds F_sigma itself and takes
/// its exxdiv from the config. Artifact hypothesis: a leaked XC term, a K
/// scaled by anything but 1, a swapped spin Fock would show as nonzero; physics
/// predicts 0 to roundoff.
#[test]
#[ignore = "dense+RS-GDF x none/ewald; measured 165 s (box load 15-60: inflated)"]
fn a_equal_one_with_zero_xc_is_the_uhf_force_per_part() {
    for jk in [Jk::Dense, Jk::Gdf] {
        for exxdiv in [ExxDiv::None, ExxDiv::Ewald] {
            let z = zero_xc(&H3, [1, 1, 3], jk, exxdiv, 1.0);
            let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
            let ks = with_jk(&z, |j| {
                kpoint_null_xc_uks_gradient(
                    &z.cell,
                    &z.prep,
                    &z.mesh,
                    &hcfg,
                    &z.hk,
                    j,
                    &z.scf,
                    1.0,
                    &gcfg(Some(exxdiv), None, None, None),
                )
                .unwrap()
            });
            let hf = with_jk(&z, |j| {
                kpoint_uhf_gradient(
                    &z.cell,
                    &z.prep,
                    &z.mesh,
                    &hcfg,
                    &z.hk,
                    j,
                    &z.scf,
                    exxdiv,
                    &kcfg(),
                )
                .unwrap()
            });
            let (p, q) = (&ks.hf.parts, &hf.parts);
            let rows = [
                ("overlap", parts_diff(&p.overlap, &q.overlap)),
                ("kinetic", parts_diff(&p.kinetic, &q.kinetic)),
                ("vsr", parts_diff(&p.vsr_basis, &q.vsr_basis)),
                ("coulomb", parts_diff(&p.coulomb, &q.coulomb)),
                ("exchange", parts_diff(&p.exchange, &q.exchange)),
                ("fit_orb_sr", parts_diff(&p.fit_orb_sr, &q.fit_orb_sr)),
                ("fit_g0", parts_diff(&p.fit_g0, &q.fit_g0)),
                ("total", parts_diff(&ks.grad, &hf.grad)),
            ];
            for (n, d) in rows {
                eprintln!("{jk:?} {exxdiv:?}: a=1 null-XC vs UHF force, {n:<10} {d:.1e}");
                assert!(d < 1e-13, "{jk:?} {exxdiv:?} {n}: {d:e}");
            }
            // Not vacuous: open shell (Da != Db) and the exchange walks matter.
            let na: f64 = z
                .scf
                .density_alpha
                .iter()
                .map(|m| m.iter().map(|x| x.norm()).sum::<f64>())
                .sum();
            let nb: f64 = z
                .scf
                .density_beta
                .iter()
                .map(|m| m.iter().map(|x| x.norm()).sum::<f64>())
                .sum();
            assert!((na - nb).abs() > 0.1, "closed-shell-like fixture");
            assert!(max_abs(&hf.grad) > 1e-3);
            match jk {
                Jk::Dense => assert!(max_abs(&hf.parts.exchange) > 1e-4),
                Jk::Gdf => assert!(max_abs(&hf.parts.fit_orb_sr) > 1e-4),
            }
        }
    }
}

// ------------------------------------------------------ closed-shell limit

/// (b) CLOSED-SHELL LIMIT == the merged RKS force, per part. Independent SCF
/// loops (k-UHF skeleton with polarized kernel vs the closed-shell loop). Artifact
/// hypothesis: a spin-factor slip (2 vs 1) or a mishandled gate would break
/// equality; physics predicts agreement at the SCF-density floor. BLIND to:
/// alpha/beta swaps, a dropped spin that is doubled, anything symmetric in the
/// two spins.
#[test]
fn closed_shell_limit_equals_the_rks_force_per_part() {
    closed_shell_vs_rks(&["PBE0"]);
}

#[test]
#[ignore = "closed-shell UKS vs RKS for LDA and PBE; measured ~25 s (box load 15-60: inflated)"]
fn closed_shell_limit_equals_the_rks_force_per_part_lda_pbe() {
    closed_shell_vs_rks(&["LDA", "PBE"]);
}

/// Per-commit canary of the weekly PBE0 row
/// (`validation_pbc_kuxcgrad.rs::one_point_mesh_matches_the_gamma_uks_gradient_per_part`):
/// the same per-part 1x1x1 vs Gamma UKS anchor on the H3 doublet with PBE
/// (the polarized semilocal XC force of both spins at one real k-point, no
/// exact exchange, no mutants; Bloch phases are not exercised).
#[test]
fn one_point_mesh_matches_the_gamma_uks_gradient_per_part_pbe() {
    one_point_vs_gamma("PBE", &[]);
}

/// Per-commit canary of the weekly PBE0 row with its exact exchange: the
/// polarized PBE0 1x1x1 vs Gamma UKS anchor per part, WITHOUT the weekly
/// row's three mutants (each adds a force, ~15 s locally; with two of them
/// this canary would sit near the 120 s per-commit line on CI). Measured
/// 2026-10-30: 32 s locally alone, worst part 1.2e-11.
#[test]
fn one_point_mesh_matches_the_gamma_uks_gradient_per_part_pbe0() {
    one_point_vs_gamma("PBE0", &[]);
}

/// Per-commit canary of the weekly `refusals_and_the_per_spin_fock_guard`:
/// the typed name and exxdiv refusals and the per-spin Fock tampers (each
/// spin, named) on the H3 doublet PBE0 state. The weekly test adds the
/// accepted-state check, the functional-mismatch guard cases, the spin swap
/// and the pure-functional entry.
#[test]
fn refusals_and_the_per_spin_fock_guard_canary() {
    let fx = fixture(&H3, [1, 1, 1], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let ok_cfg = gcfg(Some(ExxDiv::Ewald), None, None, None);
    let run = |f: &str, s: &KUScfResult, cfg: &KUksGradConfig| force_with(&fx, f, s, cfg);
    for (name, key) in [("HSE06", "bare Coulomb exchange"), ("wB97X-V", "VV10")] {
        let e = run(name, &fx.res.scf, &ok_cfg)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        assert!(e.contains(key), "{name}: {e}");
    }
    let e = run("PBE0", &fx.res.scf, &gcfg(None, None, None, None))
        .expect_err("no exxdiv")
        .to_string();
    assert!(
        e.contains("kpoint_uks_gradient: a hybrid functional"),
        "{e}"
    );
    let mut t = fx.res.scf.clone();
    t.fock_alpha[0][(0, 1)] += Complex64::new(1e-6, 0.0);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("alpha tamper")
        .to_string();
    assert!(e.contains("fock_alpha"), "{e}");
    let mut t = fx.res.scf.clone();
    t.fock_beta[0][(0, 1)] += Complex64::new(1e-6, 0.0);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("beta tamper")
        .to_string();
    assert!(e.contains("fock_beta"), "{e}");
}

/// (d) SUPERCELL ANCHOR on the non-TRIM 1x1x3 mesh, open shell: the k force on
/// atom A is the Gamma UKS force of the explicit supercell (giant-determinant
/// multiplicity (Na - Nb) N_k + 1 = 4) on ONE copy of A. Carries the Bloch
/// phases and the mesh Madelung constant in the scaled M-term; same explicit
/// neighbour cutoff on both sides.
#[test]
#[ignore = "9-atom Gamma UKS supercell; measured 53 s (box load 15-60: inflated)"]
fn k_mesh_uks_force_is_the_gamma_supercell_force_per_part() {
    let f = "PBE0";
    let gam = gamma_uks_side(&supercell_pos(&H3, 4.0, [1, 1, 3], 4), &grid_cfg(), f);
    let fx = fixture(&H3, [1, 1, 3], f, Jk::Dense, ExxDiv::Ewald);
    let k = force(&fx);
    let first = |a: &Array2<f64>| a.slice(ndarray::s![0..3, ..]).to_owned();
    let kp = &k.hf.parts;
    let rows = [
        ("xc_ao", parts_diff(&k.xc_ao, &first(&gam.parts.xc_ao))),
        (
            "xc_weight",
            parts_diff(&k.xc_weight, &first(&gam.parts.xc_weight)),
        ),
        (
            "overlap",
            parts_diff(&kp.overlap, &first(&gam.parts.overlap)),
        ),
        (
            "eri=J+K",
            parts_diff(&(&kp.coulomb + &kp.exchange), &first(&gam.parts.eri)),
        ),
        ("total", parts_diff(&k.grad, &first(&gam.grad))),
    ];
    for (n, d) in rows {
        eprintln!("{f} H3 1x1x3 vs supercell: {n:<10} {d:.2e}");
        // Worst measured 1.6e-11.
        assert!(d < 1e-9, "{f} {n}: {d:e}");
    }
}

// ----------------------------------------------------------------------- FD

/// (e) Central FD of the open-shell energy (fresh SCF per displacement) at
/// h = 1e-4 with the shared explicit cutoff.
#[test]
#[ignore = "FD of 3 open-shell energies, 12 fresh SCFs; measured ~60 s (box load 15-60: inflated)"]
fn uks_force_matches_fd() {
    for (f, jk, exxdiv) in [
        ("PBE", Jk::Dense, ExxDiv::None),
        ("PBE0", Jk::Dense, ExxDiv::Ewald),
        ("PBE0", Jk::Gdf, ExxDiv::Ewald),
    ] {
        let fx = fixture(&H3, [1, 1, 3], f, jk, exxdiv);
        let g = force(&fx);
        let mut worst = 0.0_f64;
        for (a, x) in [(0, 2), (2, 1)] {
            let d = fd(&H3, a, x, FD_H, [1, 1, 3], f, jk, exxdiv);
            worst = worst.max((g.grad[(a, x)] - d).abs());
        }
        eprintln!("{f} {jk:?} {exxdiv:?}: UKS analytic vs FD worst {worst:.2e}");
        assert!(worst < FD_BAR, "{f} {jk:?} {exxdiv:?}: {worst:e}");
    }
}

#[test]
#[ignore = "translation invariance, PBE H3 1x1x3; measured 15 s (box load 15-60: inflated)"]
fn net_force_vanishes() {
    let fx = fixture(&H3, [1, 1, 3], "PBE", Jk::Dense, ExxDiv::None);
    let g = force(&fx);
    let net: f64 = (0..3)
        .map(|x| g.grad.column(x).sum().abs())
        .fold(0.0, f64::max);
    eprintln!("net force {net:.2e}");
    assert!(net < 1e-6, "{net:e}");
}

#[test]
#[ignore = "uniform-grid FD; measured 44 s (box load 15-60: inflated)"]
fn uniform_grid_force_has_no_grid_terms_and_matches_fd() {
    let n = [24, 24, 24];
    let pos = &H3;
    let run = |p: &[[f64; 3]]| {
        let cell = cell_of(p, 2);
        let prep = prep_for(&cell, &pyscf_sto3g_h());
        let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
        let cfg = kuks_cfg(&cell, "PBE", Jk::Dense, ExxDiv::None);
        let grid = PeriodicGrid::uniform(&cell, n).unwrap();
        let r = solve_kuks_on_grid(&cell, &prep, None, &mesh, &grid, &cfg).unwrap();
        (cell, prep, mesh, r)
    };
    let (cell, prep, mesh, r) = run(pos);
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::None,
        &Default::default(),
    )
    .unwrap();
    let mut cfg = gcfg(None, None, None, None);
    cfg.base.grid = KXcGrid::Uniform(n);
    let g = kpoint_uks_gradient(
        &cell,
        &prep,
        &mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &hk,
        KGradJk::Dense(&eri),
        &r.scf,
        "PBE",
        &cfg,
    )
    .unwrap();
    assert_eq!(max_abs(&g.xc_point), 0.0);
    assert_eq!(max_abs(&g.xc_weight), 0.0);
    let mut worst = 0.0_f64;
    for (a, x) in [(0, 2), (2, 1)] {
        let e = |p: &[[f64; 3]]| run(p).3.scf.energy;
        let h = FD_H;
        let d = (e(&moved(pos, a, x, h)) - e(&moved(pos, a, x, -h))) / (2.0 * h);
        worst = worst.max((g.grad[(a, x)] - d).abs());
    }
    eprintln!("uniform-grid UKS analytic vs FD worst {worst:.2e}");
    assert!(worst < FD_BAR, "{worst:e}");
}

// -------------------------------------------------------- bitwise / guard

#[test]
#[ignore = "4 force runs; measured 107 s (box load 15-60: inflated)"]
fn uks_force_is_bit_identical_across_threads() {
    for jk in [Jk::Dense, Jk::Gdf] {
        let fx = fixture(&H3, [1, 1, 3], "PBE0", jk, ExxDiv::Ewald);
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| force(&fx))
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.grad, b.grad, "{jk:?}");
        assert_eq!(a.xc_ao, b.xc_ao, "{jk:?}");
    }
}

/// Every budget either reproduces the unconstrained force bit-identically or
/// fails typed. The Fock-guard charge (10 N_k n^2 complex) is NOT tripped by
/// this sweep on a cell this small (the XC chunk gate trips first): the charge
/// is covered by the sizing arithmetic only (stated, not claimed).
#[test]
#[ignore = "budget sweep; measured 140 s (box load 15-60: inflated)"]
fn budgets_either_reproduce_the_force_or_fail_typed() {
    let fx = fixture(&H3, [1, 1, 3], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let full = force(&fx);
    let (mut ok, mut err) = (0, 0);
    let mut b = 20_000usize;
    while b <= AMPLE {
        let mut cfg = gcfg(Some(ExxDiv::Ewald), None, None, None);
        cfg.base.grad.budget_bytes = Some(b);
        match force_with(&fx, &fx.functional, &fx.res.scf, &cfg) {
            Ok(g) => {
                ok += 1;
                assert_eq!(g.grad, full.grad, "budget {b}");
            }
            Err(e) => {
                err += 1;
                assert!(
                    e.to_string().contains("budget") || e.to_string().contains("memory"),
                    "{b}: {e}"
                );
            }
        }
        b *= 3;
    }
    eprintln!("budget sweep: {ok} runs identical, {err} typed refusals");
    assert!(ok > 0 && err > 0);
}

/// Production-wired spin-specific and shared defects against the FD-anchored
/// force on the open-shell H3 cell. Each must miss by far more than the FD bar;
/// the closed-shell limit is blind to the spin ones (stated in (b)). The
/// control `GuardSkippedOnly` must be exactly 0.
#[test]
#[ignore = "25 production-wired mutants on H3 PBE0; measured 598 s (box load 15-60: inflated)"]
fn smoke_mutants_are_caught() {
    let fx = fixture(&H3, [1, 1, 3], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let base = force(&fx);
    assert_eq!(
        miss(&fx, Some(KXcMutation::GuardSkippedOnly), None, None, &base),
        0.0
    );
    for sm in [
        KUSpinMutation::XcSpinSwap,
        KUSpinMutation::XcDropBeta,
        KUSpinMutation::XcCrossDropped,
        KUSpinMutation::XcCrossOwn,
        KUSpinMutation::XcSpinAveraged,
        KUSpinMutation::PulayFockSwap,
        KUSpinMutation::PulayKSwap,
        KUSpinMutation::PulayJOneSpin,
        KUSpinMutation::PulayNoVxcBeta,
    ] {
        let d = miss(&fx, None, Some(sm), None, &base);
        eprintln!("spin mutant {sm:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{sm:?} survives: {d:e}");
    }
    for xm in [
        KXcMutation::NoAo,
        KXcMutation::NoPointMotion,
        KXcMutation::NoWeightDeriv,
        KXcMutation::NoGga,
        KXcMutation::NoHessian,
        KXcMutation::WrongFactor,
        KXcMutation::NoInvNk,
        KXcMutation::NoPhase,
        KXcMutation::PulayNoVxc,
        KXcMutation::WrongRemainder,
    ] {
        let d = miss(&fx, Some(xm), None, None, &base);
        eprintln!("xc mutant {xm:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{xm:?} survives: {d:e}");
    }
    for gm in [
        KGradMutation::ExchDenseUnscaled,
        KGradMutation::ExchMadelungUnscaled,
        KGradMutation::ExchJScaled,
        KGradMutation::ExchScaledTwice,
        KGradMutation::ExchSpinFactor,
    ] {
        let d = miss(&fx, None, None, Some(gm), &base);
        eprintln!("hybrid-scale mutant {gm:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{gm:?} survives: {d:e}");
    }
}

// ------------------------------------------------- UKS stationarity gate

/// UKS stationarity probe: for SCFs stopped at looser `grad_conv`, the largest
/// `|F D S - S D F|` the force reports against the force error relative to the
/// tightly converged force at the same geometry (gate off). The refusal
/// threshold `UKS_COMMUTATOR_TOL` is read off this table.
#[test]
#[ignore = "probe: 8 open-shell SCFs"]
fn uks_stationarity_probe() {
    let tight = {
        let fx = fixture(&H3, [1, 1, 2], "PBE", Jk::Dense, ExxDiv::None);
        force(&fx).grad
    };
    for gconv in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5] {
        SETTINGS.with(|st| st.set((CUTOFF, 1e-3 * gconv, gconv)));
        let fx = fixture(&H3, [1, 1, 2], "PBE", Jk::Dense, ExxDiv::None);
        let g = force_mut(&fx, None, Some(KUSpinMutation::StationarityGateOff), None);
        eprintln!(
            "grad_conv {gconv:.0e}: commutator {:.3e}, max |dF| vs tight {:.3e}",
            g.hf.commutator,
            parts_diff(&g.grad, &tight)
        );
    }
}

/// A loosely converged UKS state is refused with the commutator message; the
/// same state with the gate off returns a force that is wrong by far more than
/// the FD bar (so the gate matters), and a tight state passes the gate.
#[test]
fn loose_uks_scf_is_refused_by_the_stationarity_gate() {
    let tight = fixture(&H3, [1, 1, 1], "PBE", Jk::Dense, ExxDiv::None);
    let gt = force(&tight);
    assert!(
        gt.hf.commutator < UKS_COMMUTATOR_TOL / 100.0,
        "{:e}",
        gt.hf.commutator
    );
    SETTINGS.with(|st| st.set((CUTOFF, LOOSE_UKS.0, LOOSE_UKS.1)));
    let loose = fixture(&H3, [1, 1, 1], "PBE", Jk::Dense, ExxDiv::None);
    let e = force_with(&loose, "PBE", &loose.res.scf, &gcfg(None, None, None, None))
        .expect_err("loose SCF accepted")
        .to_string();
    eprintln!("loose UKS: {e}");
    assert!(
        e.contains("kpoint_uks_gradient: the spin Fock/density commutator")
            && e.contains("exceeds 1e-5"),
        "{e}"
    );
    let off = force_mut(
        &loose,
        None,
        Some(KUSpinMutation::StationarityGateOff),
        None,
    );
    let miss = parts_diff(&off.grad, &gt.grad);
    eprintln!(
        "loose UKS: commutator {:.3e}, force error with the gate off {miss:.3e}",
        off.hf.commutator
    );
    assert!(off.hf.commutator > UKS_COMMUTATOR_TOL);
    assert!(
        miss > 10.0 * FD_BAR,
        "gate off is not wrong enough: {miss:e}"
    );
}
