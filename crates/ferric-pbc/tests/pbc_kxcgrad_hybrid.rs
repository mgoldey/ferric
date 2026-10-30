//! Analytic forces of the k-point closed-shell RKS energy for GLOBAL HYBRID
//! functionals (`ferric_pbc::kxcgrad`, module doc "Global hybrids").
//!
//! Anchors (each with its artifact hypothesis next to the physics hypothesis):
//! a = 1 with a zero XC remainder == the k-point HF force; the force at a fixed
//! density is affine in a; 1x1x1 and the non-TRIM 1x1x3 supercell identities
//! against the Gamma PBE0 gradient per part; central FD of the PBE0 / B3LYP
//! energy (fresh SCF per displacement), dense AFT and RS-GDF, exxdiv none and
//! ewald; Ewald == None force equality; thread bit-identity; the Fock guard
//! from every side; typed refusals; production-wired mutants.
//!
//! NOT covered: p/d shells on the hybrid K walks (every fixture here is
//! H2/STO-3G, s functions only), range-separated hybrids (refused), VV10/double
//! hybrids (refused; meta-GGA has its own file, `pbc_kxcgrad_mgga.rs`), range-split RS-GDF with a hybrid, heavy atoms / ECP with a
//! hybrid, shifted meshes, open shell, stress.
//!
//! WEEKLY HALF: `smoke_mutants_are_caught` (CI >= 120 s: dense ewald, RS-GDF
//! ewald and dense none mutant sweeps on 1x1x3) runs in
//! `validation_pbc_kxcgrad_hybrid.rs`; the helpers are
//! `kxcgrad_hybrid_support/mod.rs`. Per-commit canary:
//! `dense_exchange_scale_mutants_are_caught_on_the_one_point_mesh` (every
//! dense-reachable exchange-scale defect, the remainder defect and the
//! identity controls, dense ewald, 1x1x1); the RS-GDF fit-scale defects stay
//! per commit in `gamma_rsgdf_anchor_catches_the_fit_scale_mutants`.

mod common;
#[path = "kxcgrad_hybrid_support/mod.rs"]
mod support;

use support::*;

/// (a) EXACTNESS ANCHOR. a = 1 with a zero XC remainder IS the k-point HF
/// force: same `assemble`, but the KS entry rebuilds F itself (h + J - K/2 with
/// the Madelung shift, exact guard) and takes its exxdiv from the config.
/// Artifact hypothesis: a leaked XC term, a dropped Madelung, a K scaled by
/// anything but 1 would show as a nonzero difference; physics predicts 0 to
/// roundoff (F is hermitised on one side only: ~1e-16).
#[test]
fn a_equal_one_with_zero_xc_is_the_hf_force_per_part() {
    for jk in [Jk::Dense, Jk::Gdf] {
        for exxdiv in [ExxDiv::None, ExxDiv::Ewald] {
            let z = zero_xc(&H2, [1, 1, 3], jk, exxdiv, 1.0);
            let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
            let ks = with_jk(&z, |j| {
                kpoint_null_xc_gradient(
                    &z.cell,
                    &z.prep,
                    &z.mesh,
                    &hcfg,
                    &z.hk,
                    j,
                    &z.scf,
                    1.0,
                    &gcfg(Some(exxdiv), None, None),
                )
                .unwrap()
            });
            let hf = with_jk(&z, |j| {
                kpoint_rhf_gradient(
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
                ("fit_aux_lr", parts_diff(&p.fit_aux_lr, &q.fit_aux_lr)),
                (
                    "fit_metric_sr",
                    parts_diff(&p.fit_metric_sr, &q.fit_metric_sr),
                ),
                ("fit_g0", parts_diff(&p.fit_g0, &q.fit_g0)),
                ("total", parts_diff(&ks.grad, &hf.grad)),
            ];
            for (n, d) in rows {
                eprintln!("{jk:?} {exxdiv:?}: a=1 null-XC vs HF force, {n:<13} {d:.1e}");
                assert!(d < 1e-13, "{jk:?} {exxdiv:?} {n}: {d:e}");
            }
            // The anchor is not vacuous.
            assert!(max_abs(&hf.grad) > 1e-3);
            match jk {
                Jk::Dense => assert!(max_abs(&hf.parts.exchange) > 1e-4),
                Jk::Gdf | Jk::GdfSplit => assert!(max_abs(&hf.parts.fit_orb_sr) > 1e-4),
            }
        }
    }
}

/// (b) AFFINITY IN a at a FIXED density. The energy's two-electron part is
/// J-terms + a (K-terms) and F = h + J - (a/2) K, so the force is affine in
/// a. WHAT THIS CATCHES: only a-NONLINEAR defects (a applied twice, a^2).
/// WHAT IT DOES NOT: `a` applied to J, an unscaled piece, a 2a spin factor, an
/// unscaled Madelung are all still affine in a; they are pinned by the Gamma
/// anchors, the supercell identity, FD and the mutant tests below. (The a = 1
/// HF anchor is likewise blind to every scaling defect except the spin
/// factor.) Density: the PBE0 SCF's; the stored Fock is replaced by the
/// rebuilt F(a) so the guard passes (the formula, not self-consistency, is
/// under test).
#[test]
fn force_is_affine_in_the_exchange_fraction_at_fixed_density() {
    for jk in [Jk::Dense, Jk::Gdf] {
        let z = zero_xc(&H2, [1, 1, 3], jk, ExxDiv::Ewald, 0.25);
        let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
        // J and K of the fixed density.
        let (jm, km) = jk_mats(&z);
        let at = |a: f64| -> KRksGradient {
            let mut s = z.scf.clone();
            for q in 0..z.mesh.nk() {
                s.fock[q] = &(&z.hk.h[q] + &jm[q]) - &km[q].mapv(|x| x * (0.5 * a));
            }
            with_jk(&z, |j| {
                kpoint_null_xc_gradient(
                    &z.cell,
                    &z.prep,
                    &z.mesh,
                    &hcfg,
                    &z.hk,
                    j,
                    &s,
                    a,
                    &gcfg(Some(ExxDiv::Ewald), None, None),
                )
                .unwrap()
            })
        };
        let (g0, g1, gh, gq) = (at(0.0), at(1.0), at(0.5), at(0.25));
        let aff = |x0: &Array2<f64>, x1: &Array2<f64>, xa: &Array2<f64>, a: f64| {
            parts_diff(xa, &(&x0.mapv(|v| v * (1.0 - a)) + &x1.mapv(|v| v * a)))
        };
        let rows = [
            ("total a=.5", aff(&g0.grad, &g1.grad, &gh.grad, 0.5)),
            ("total a=.25", aff(&g0.grad, &g1.grad, &gq.grad, 0.25)),
            (
                "overlap a=.5",
                aff(
                    &g0.hf.parts.overlap,
                    &g1.hf.parts.overlap,
                    &gh.hf.parts.overlap,
                    0.5,
                ),
            ),
            (
                "exchange a=.5",
                aff(
                    &g0.hf.parts.exchange,
                    &g1.hf.parts.exchange,
                    &gh.hf.parts.exchange,
                    0.5,
                ),
            ),
            (
                "fit_orb_sr a=.5",
                aff(
                    &g0.hf.parts.fit_orb_sr,
                    &g1.hf.parts.fit_orb_sr,
                    &gh.hf.parts.fit_orb_sr,
                    0.5,
                ),
            ),
            (
                "fit_metric_lr a=.25",
                aff(
                    &g0.hf.parts.fit_metric_lr,
                    &g1.hf.parts.fit_metric_lr,
                    &gq.hf.parts.fit_metric_lr,
                    0.25,
                ),
            ),
        ];
        for (n, d) in rows {
            eprintln!("{jk:?}: affinity residual {n:<20} {d:.1e}");
            assert!(d < 1e-12, "{jk:?} {n}: {d:e}");
        }
        // Not vacuous: the a-dependent part is large against the residual, and
        // J-only (a = 0) is far from HF.
        let span = parts_diff(&g0.grad, &g1.grad);
        eprintln!("{jk:?}: |g(1) - g(0)| = {span:.3e}");
        assert!(span > 1e-3);
        assert_eq!(max_abs(&g0.hf.parts.exchange), 0.0);
    }
}

/// (c) 1x1x1 == the Gamma hybrid KS gradient, per part (Ewald exxdiv, so the
/// scaled Madelung M-term is inside `overlap`). Independent construction: real
/// lattice sums vs complex Bloch AOs and a separate SCF. Floor: the two SCFs
/// converge to ~1e-10 in the density.
#[test]
fn one_point_mesh_matches_the_gamma_hybrid_gradient_per_part() {
    for f in ["PBE0", "B3LYP"] {
        let grid = grid_cfg();
        let gam = gamma_pbe0(&cell_of(&H2), &grid, f);
        let fx = fixture(&H2, [1, 1, 1], f, Jk::Dense, ExxDiv::Ewald);
        let k = force(&fx);
        let (p, kp) = (&gam.parts, &k.hf.parts);
        let rows = [
            ("xc_ao", parts_diff(&k.xc_ao, &p.xc_ao)),
            ("xc_point", parts_diff(&k.xc_point, &p.xc_point)),
            ("xc_weight", parts_diff(&k.xc_weight, &p.xc_weight)),
            ("overlap", parts_diff(&kp.overlap, &p.overlap)),
            ("kinetic", parts_diff(&kp.kinetic, &p.kinetic)),
            ("eri=J+K", parts_diff(&(&kp.coulomb + &kp.exchange), &p.eri)),
            ("total", parts_diff(&k.grad, &gam.grad)),
        ];
        for (n, d) in rows {
            eprintln!("{f} 1x1x1 vs Gamma: {n:<10} {d:.2e}");
            assert!(d < 1e-10, "{f} {n}: {d:e}");
        }
        assert!(max_abs(&kp.exchange) > 1e-4, "K walks are exercised");
        assert!(max_abs(&k.xc_weight) > 1e-3);
    }
}

/// (d) SUPERCELL ANCHOR on the non-TRIM 1x1x3 mesh (carries the Bloch phases,
/// and with Ewald the MESH Madelung constant in the scaled M-term: the
/// primitive-cell constant would differ). The k force on atom A is the Gamma
/// force of the explicit supercell on ONE copy of A. Both sides use the same
/// explicit `neighbour_cutoff`.
#[test]
fn k_mesh_hybrid_force_is_the_gamma_supercell_force_per_part() {
    let grid = grid_cfg();
    let f = "PBE0";
    let gam = gamma_pbe0(&supercell_pos(&H2, 4.0, [1, 1, 3]), &grid, f);
    let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, ExxDiv::Ewald);
    let k = force(&fx);
    let first = |a: &Array2<f64>| a.slice(ndarray::s![0..2, ..]).to_owned();
    let kp = &k.hf.parts;
    let rows = [
        ("xc_ao", parts_diff(&k.xc_ao, &first(&gam.parts.xc_ao))),
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
        eprintln!("{f} 1x1x3 vs supercell: {n:<10} {d:.2e}");
        assert!(d < 1e-9, "{f} {n}: {d:e}");
    }
}

// ----------------------------------------------------------------------- FD

/// (e) Central FD of the hybrid energy (fresh SCF per displacement), h = 1e-4
/// with the shared explicit cutoff (no neighbour discontinuity inside +-h).
/// Bar: see `FD_BAR`. Both exxdiv modes (f): each is checked against its OWN
/// energy's FD, and the two analytic forces must agree (derived equal: the
/// shift is a constant on the occupied space, the density does not change, and
/// its force contribution cancels between K and the M-term).
#[test]
fn pbe0_force_matches_fd_dense_and_rsgdf_both_exxdiv() {
    let mut by_mode: HashMap<(Jk, bool), Array2<f64>> = HashMap::new();
    for jk in [Jk::Dense, Jk::Gdf] {
        for exxdiv in [ExxDiv::None, ExxDiv::Ewald] {
            let fx = fixture(&H2, [1, 1, 3], "PBE0", jk, exxdiv);
            let g = force(&fx);
            let mut worst = 0.0_f64;
            for (a, x) in [(0, 2), (1, 1)] {
                let d = fd(&H2, a, x, 1e-4, [1, 1, 3], "PBE0", jk, exxdiv);
                worst = worst.max((g.grad[(a, x)] - d).abs());
            }
            eprintln!("{jk:?} {exxdiv:?}: PBE0 analytic vs FD worst {worst:.2e}");
            assert!(worst < FD_BAR, "{jk:?} {exxdiv:?}: {worst:e}");
            by_mode.insert((jk, exxdiv == ExxDiv::Ewald), g.grad);
        }
        let dd = parts_diff(&by_mode[&(jk, false)], &by_mode[&(jk, true)]);
        eprintln!("{jk:?}: Ewald vs None force difference {dd:.2e}");
        assert!(dd < 1e-9, "{jk:?}: {dd:e}");
    }
    // The RS-GDF fit is really wired: it differs from dense AFT by a fit-sized
    // amount (not 0, not O(1)).
    let d = parts_diff(&by_mode[&(Jk::Dense, false)], &by_mode[&(Jk::Gdf, false)]);
    eprintln!("dense vs RS-GDF PBE0 force difference {d:.2e}");
    assert!(d > 1e-9 && d < 1e-2);
}

/// A second global hybrid with a different a and a different semilocal
/// remainder (B3LYP, a = 0.2): dense AFT, Ewald.
#[test]
fn b3lyp_force_matches_fd() {
    let fx = fixture(&H2, [1, 1, 3], "B3LYP", Jk::Dense, ExxDiv::Ewald);
    assert!((fx.res.exact_exchange_fraction - 0.2).abs() < 1e-12);
    let g = force(&fx);
    let mut worst = 0.0_f64;
    for (a, x) in [(0, 2), (1, 1)] {
        let d = fd(
            &H2,
            a,
            x,
            1e-4,
            [1, 1, 3],
            "B3LYP",
            Jk::Dense,
            ExxDiv::Ewald,
        );
        worst = worst.max((g.grad[(a, x)] - d).abs());
    }
    eprintln!("B3LYP analytic vs FD worst {worst:.2e}");
    assert!(worst < FD_BAR, "{worst:e}");
    // FD FLOOR, MEASURED (component (0,2), signed residual FD - analytic):
    //   h = 2e-4: +1.08e-8   1e-4: +2.31e-9   5e-5: +1.9e-10   2.5e-5: -2.5e-10
    // The two largest h give a ratio of 4.7 (c h^2, c ~ 0.23-0.27 Ha/Bohr^3: the
    // third derivative of the energy along the H-H bond); below h ~ 5e-5 the
    // residual is the SCF-energy noise eps/(2h) with eps ~ 1.2e-14 Ha (it
    // changes sign and does not shrink). The 2.31e-9 is the SAME component in
    // every row (PBE0/B3LYP, dense/RS-GDF, none/ewald): the bond-direction
    // third derivative is almost purely electrostatic, so it barely depends on
    // the functional (plain PBE: 2.22e-9); every other component is <= 1.2e-10.
    // Artifact hypothesis: a force error would not follow h^2; here it does
    // until the noise floor.
    let r =
        |h: f64| fd(&H2, 0, 2, h, [1, 1, 3], "B3LYP", Jk::Dense, ExxDiv::Ewald) - g.grad[(0, 2)];
    let (r2, r1) = (r(2e-4), r(1e-4));
    eprintln!(
        "B3LYP (0,2) residual h=2e-4 {r2:+.2e}, h=1e-4 {r1:+.2e}, ratio {:.2}",
        r2 / r1
    );
    assert!(r2 / r1 > 3.5 && r2 / r1 < 6.0, "not h^2: ratio {}", r2 / r1);
}

/// Range-split RS-GDF (lambda = 1) with a hybrid: the exchange coefficients go
/// through the same fit-gradient plumbing as the unsplit fit; FD of the
/// hybrid energy on the split build decides.
#[test]
fn pbe0_force_with_the_range_split_fit_matches_fd() {
    let fx = fixture(&H2, [1, 1, 3], "PBE0", Jk::GdfSplit, ExxDiv::Ewald);
    let g = force(&fx);
    let mut worst = 0.0_f64;
    for (a, x) in [(0, 2), (1, 1)] {
        let d = fd(
            &H2,
            a,
            x,
            1e-4,
            [1, 1, 3],
            "PBE0",
            Jk::GdfSplit,
            ExxDiv::Ewald,
        );
        worst = worst.max((g.grad[(a, x)] - d).abs());
    }
    eprintln!("split RS-GDF PBE0 analytic vs FD worst {worst:.2e}");
    assert!(worst < FD_BAR, "{worst:e}");
}

// -------------------------------------------------------------- bitwise etc.

#[test]
fn hybrid_force_is_bit_identical_across_threads() {
    for jk in [Jk::Dense, Jk::Gdf] {
        let fx = fixture(&H2, [1, 1, 3], "PBE0", jk, ExxDiv::Ewald);
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| force(&fx))
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.grad, b.grad, "{jk:?}");
        assert_eq!(a.hf.parts.exchange, b.hf.parts.exchange, "{jk:?}");
    }
}

// ---------------------------------------------------- refusals / Fock guard

#[test]
fn refusals_and_the_hybrid_fock_guard() {
    let fx = fixture(&H2, [1, 1, 3], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let ok_cfg = gcfg(Some(ExxDiv::Ewald), None, None);
    let run = |f: &str, s: &KScfResult, cfg: &KRksGradConfig| force_with(&fx, f, s, cfg);
    // Typed refusals.
    for (name, key) in [("HSE06", "range-separated"), ("wB97X-V", "VV10")] {
        let e = run(name, &fx.res.scf, &ok_cfg)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        eprintln!("{name}: {e}");
        assert!(e.contains(key), "{name}: {e}");
    }
    // A hybrid needs the exxdiv: a typed error, not a silent default.
    let e = run("PBE0", &fx.res.scf, &gcfg(None, None, None))
        .expect_err("no exxdiv")
        .to_string();
    eprintln!("no exxdiv: {e}");
    assert!(e.contains("exxdiv"));
    // Matching inputs pass at roundoff.
    let ok = run("PBE0", &fx.res.scf, &ok_cfg).unwrap();
    assert!(ok.hf.fock_mismatch < 1e-12, "{:e}", ok.hf.fock_mismatch);
    // Guard, every side: wrong exxdiv; wrong functional with another a and
    // another remainder (B3LYP on a PBE0 SCF); a pure functional's force asked
    // of a hybrid SCF and vice versa; a tamper.
    let pbe = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, ExxDiv::Ewald);
    let cases: Vec<(&str, Result<KRksGradient, ferric_core::FerricError>)> = vec![
        (
            "exxdiv None on an Ewald SCF",
            run("PBE0", &fx.res.scf, &gcfg(Some(ExxDiv::None), None, None)),
        ),
        ("B3LYP on a PBE0 SCF", run("B3LYP", &fx.res.scf, &ok_cfg)),
        ("PBE force on a PBE0 SCF", run("PBE", &fx.res.scf, &ok_cfg)),
        (
            "PBE0 force on a PBE SCF",
            run("PBE0", &pbe.res.scf, &ok_cfg),
        ),
    ];
    for (n, r) in cases {
        let e = r
            .err()
            .unwrap_or_else(|| panic!("{n} accepted"))
            .to_string();
        eprintln!("{n}: {e}");
        assert!(e.contains("differs"), "{n}: {e}");
    }
    // Wrong a with the RIGHT remainder, via the zero-XC entry (a = 0.5 asked of
    // an a = 1 SCF).
    let z = zero_xc(&H2, [1, 1, 3], Jk::Dense, ExxDiv::Ewald, 1.0);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let r = with_jk(&z, |j| {
        kpoint_null_xc_gradient(
            &z.cell, &z.prep, &z.mesh, &hcfg, &z.hk, j, &z.scf, 0.5, &ok_cfg,
        )
    });
    let e = r.expect_err("wrong a accepted").to_string();
    eprintln!("wrong a: {e}");
    assert!(e.contains("differs"));
    let mut t = fx.res.scf.clone();
    t.fock[1][(0, 1)] += Complex64::new(1e-6, 0.0);
    assert!(
        run("PBE0", &t, &ok_cfg).is_err(),
        "1e-6 tamper must be refused"
    );
    // A pure functional ignores the exxdiv entirely.
    let a = force_with(&pbe, "PBE", &pbe.res.scf, &gcfg(None, None, None)).unwrap();
    let b = force_with(
        &pbe,
        "PBE",
        &pbe.res.scf,
        &gcfg(Some(ExxDiv::None), None, None),
    )
    .unwrap();
    assert_eq!(a.grad, b.grad);
    assert_eq!(max_abs(&a.hf.parts.exchange), 0.0);
}

/// Memory: every budget either runs and gives the bit-identical force of the
/// unconstrained run, or fails with a typed error (never a panic or a wrong
/// number). The Fock-guard matrices (D, J, K, F, V_xc) are charged to the
/// ledger after the XC pass (see `kxcgrad` "Memory"); on this tiny cell the XC
/// chunk gate (>= 10 kB) always trips before that 1 kB charge could, so the
/// charge itself is covered by the sizing arithmetic, not by a tripping budget
/// (stated, not claimed).
#[test]
fn budgets_either_reproduce_the_force_or_fail_typed() {
    let fx = fixture(&H2, [1, 1, 3], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let full = force(&fx);
    let (mut ok, mut err) = (0, 0);
    let mut b = 20_000usize;
    while b <= AMPLE {
        let mut cfg = gcfg(Some(ExxDiv::Ewald), None, None);
        cfg.grad.budget_bytes = Some(b);
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

/// Production-class grid (40, 110) and a 2x2x2 mesh (TRIM-only: phases are
/// real there; the 1x1x3 row carries them), dense AFT PBE0 with Ewald, FD of
/// the energy.
#[test]
#[ignore = "production-class grid + 2x2x2 mesh FD, PBE0 dense, all 6 components at h = 2.5e-5; measured 55 s"]
fn pbe0_force_matches_fd_on_the_production_grid_and_a_2x2x2_mesh() {
    GRID.with(|g| g.set((40, 110)));
    for mesh_n in [[1, 1, 3], [2, 2, 2]] {
        let fx = fixture(&H2, mesh_n, "PBE0", Jk::Dense, ExxDiv::Ewald);
        let g = force(&fx);
        let mut worst = 0.0_f64;
        for (a, x) in (0..2).flat_map(|a| (0..3).map(move |x| (a, x))) {
            let d = fd(&H2, a, x, H_STEP, mesh_n, "PBE0", Jk::Dense, ExxDiv::Ewald);
            eprintln!(
                "  comp ({a},{x}): analytic {:+.9e} FD {d:+.9e} diff {:+.2e}",
                g.grad[(a, x)],
                g.grad[(a, x)] - d
            );
            worst = worst.max((g.grad[(a, x)] - d).abs());
        }
        eprintln!("(40,110) {mesh_n:?}: PBE0 analytic vs FD worst {worst:.2e}");
        assert!(worst < FD_BAR, "{mesh_n:?}: {worst:e}");
    }
}

/// (c') RS-GDF 1x1x1 k force == the Gamma RS-GDF hybrid gradient, per part
/// (fit_*, overlap, total), exxdiv None and Ewald, PBE0 and B3LYP. The Gamma
/// fit gradient is separate code, and scales J and K independently, so unlike
/// the a = 1 anchor this one pins the fit's a-scaling. Floor: SCFs converge to
/// ~1e-10 in the density.
#[test]
fn one_point_rsgdf_hybrid_force_matches_the_gamma_rsgdf_gradient_per_part() {
    for f in ["PBE0", "B3LYP"] {
        for exxdiv in [ExxDiv::None, ExxDiv::Ewald] {
            let gam = gamma_rsgdf_hybrid(&cell_of(&H2), f, exxdiv);
            let fx = fixture(&H2, [1, 1, 1], f, Jk::Gdf, exxdiv);
            let k = force(&fx);
            for (n, d) in fit_rows(&k, &gam) {
                eprintln!("{f} {exxdiv:?} RS-GDF 1x1x1 vs Gamma: {n:<14} {d:.2e}");
                assert!(d < 1e-9, "{f} {exxdiv:?} {n}: {d:e}");
            }
            assert!(max_abs(&k.hf.parts.fit_orb_sr) > 1e-4, "fit exercised");
        }
    }
}

/// The fit-side defects are caught by the Gamma anchor itself, not only by FD:
/// `a` applied to the fit's Coulomb pieces and the fit exchange left unscaled
/// both miss the Gamma gradient by far more than the anchor's floor.
#[test]
fn gamma_rsgdf_anchor_catches_the_fit_scale_mutants() {
    use KGradMutation as G;
    let gam = gamma_rsgdf_hybrid(&cell_of(&H2), "PBE0", ExxDiv::Ewald);
    let fx = fixture(&H2, [1, 1, 1], "PBE0", Jk::Gdf, ExxDiv::Ewald);
    let base = parts_diff(&force(&fx).grad, &gam.grad);
    for m in [
        G::ExchJScaled,
        G::ExchFitUnscaled,
        G::ExchScaledTwice,
        G::ExchSpinFactor,
    ] {
        let d = parts_diff(&force_mut(&fx, None, Some(m)).grad, &gam.grad);
        eprintln!("{m:?}: miss vs the Gamma RS-GDF gradient {d:.2e} (unmutated {base:.1e})");
        assert!(d > MUTANT_BAR, "{m:?} survives the Gamma anchor: {d:e}");
    }
}

/// Per-commit canary of the weekly `smoke_mutants_are_caught`
/// (`validation_pbc_kxcgrad_hybrid.rs`): the dense-AFT + Ewald block of that
/// sweep on the 1x1x1 mesh. Every dense-reachable exchange-scale defect and
/// the PBE-for-PBE0 remainder must move the force by more than the mutant
/// bar; the fit-only defect and the guard-skip control must be EXACTLY 0.
/// The production force it is measured against is anchored per commit to the
/// Gamma PBE0 gradient (`one_point_mesh_matches_the_gamma_hybrid_gradient_per_part`).
#[test]
fn dense_exchange_scale_mutants_are_caught_on_the_one_point_mesh() {
    use KGradMutation as G;
    let fx = fixture(&H2, [1, 1, 1], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let base = force(&fx);
    for m in [
        G::ExchDenseUnscaled,
        G::ExchMadelungUnscaled,
        G::ExchJScaled,
        G::ExchScaledTwice,
        G::ExchSpinFactor,
    ] {
        let d = miss_g(&fx, m, &base);
        eprintln!("1x1x1 dense ewald {m:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{m:?} survives: {d:e}");
    }
    assert_eq!(miss_g(&fx, G::ExchFitUnscaled, &base), 0.0);
    let d = parts_diff(
        &force_mut(&fx, Some(KXcMutation::WrongRemainder), None).grad,
        &base.grad,
    );
    eprintln!("1x1x1 WrongRemainder: miss {d:.2e}");
    assert!(d > MUTANT_BAR);
    let d0 = parts_diff(
        &force_mut(&fx, Some(KXcMutation::GuardSkippedOnly), None).grad,
        &base.grad,
    );
    assert_eq!(d0, 0.0, "the guard-skip control must not change the force");
}

// ----------------------------------------------- net force, rsh, validation

/// Translation invariance: sum over atoms ~ 0 for the hybrid, dense and
/// RS-GDF. It covers the components FD never probes, and is BLIND to a
/// dropped XC term on this cell (X1, X3, X4 each sum to zero over the atoms,
/// PR A) and to a-scaling defects (all still translation invariant).
#[test]
fn hybrid_net_force_vanishes() {
    for jk in [Jk::Dense, Jk::Gdf] {
        let fx = fixture(&H2, [1, 1, 3], "PBE0", jk, ExxDiv::Ewald);
        let g = force(&fx);
        let net = (0..3)
            .map(|c| g.grad.column(c).sum().abs())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "{jk:?}: hybrid net force {net:.2e}, max |F| {:.3e}",
            max_abs(&g.grad)
        );
        assert!(net < 1e-10, "{jk:?}: {net:e}");
        assert!(max_abs(&g.grad) > 1e-2);
    }
}

/// An injected SCF built from dense kernels that carry an attenuated exchange
/// (rsh) passes the Fock guard (same kernels) but would get the bare 4pi/k^2
/// K derivative: refused with a typed error naming rsh, for the HF entry and
/// the KS entry alike.
#[test]
fn dense_kernels_with_rsh_are_refused() {
    let fx = fixture(&H2, [1, 1, 3], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let mut dc = KDenseAftConfig::default();
    dc.rsh = Some(ferric_pbc::rsh::RshParams::from_cam(
        &ferric_dft::libxc::xc_def_from_name("HSE06")
            .unwrap()
            .cam
            .unwrap(),
    ));
    let eri = KDenseAftEri::build(&fx.cell, &fx.prep, &fx.mesh, &fx.hk.s, fx.exxdiv, &dc).unwrap();
    assert!(eri.rsh().is_some());
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let e = kpoint_rks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        KGradJk::Dense(&eri),
        &fx.res.scf,
        "PBE0",
        &gcfg(Some(ExxDiv::Ewald), None, None),
    )
    .expect_err("rsh kernels accepted")
    .to_string();
    eprintln!("rsh dense kernels (KS entry): {e}");
    assert!(e.contains("rsh"));
    let e = kpoint_rhf_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        KGradJk::Dense(&eri),
        &fx.res.scf,
        ExxDiv::Ewald,
        &kcfg(),
    )
    .expect_err("rsh kernels accepted by HF")
    .to_string();
    assert!(e.contains("rsh"));
}

/// The exact-exchange fraction is validated (typed error, NaN included).
#[test]
fn exchange_fraction_is_validated() {
    let z = zero_xc(&H2, [1, 1, 3], Jk::Dense, ExxDiv::None, 1.0);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    for bad in [-0.1, 1.5, f64::NAN] {
        let r = with_jk(&z, |j| {
            kpoint_null_xc_gradient(
                &z.cell,
                &z.prep,
                &z.mesh,
                &hcfg,
                &z.hk,
                j,
                &z.scf,
                bad,
                &gcfg(Some(ExxDiv::None), None, None),
            )
        });
        let e = r.expect_err("bad fraction accepted").to_string();
        assert!(e.contains("outside [0, 1]"), "{bad}: {e}");
    }
}
