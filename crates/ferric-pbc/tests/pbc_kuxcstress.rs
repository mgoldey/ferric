//! Analytic stress of the k-point open-shell UKS energy (`ferric_pbc::kuxcstress`,
//! module doc) for LDA, GGA and global hybrids. Predictions are written next to
//! each test before measuring; measured numbers are recorded after.
//!
//! WHICH ANCHOR COVERS WHICH DEFECT (the closed-shell limit is blind to every
//! alpha/beta asymmetry; a = 1 with a zero XC is blind to every a-scaling):
//!
//! * (1) a = 1 + zero XC == `kpoint_uhf_stress` per part: the KS entry's F
//!   rebuild, exxdiv routing, spin plumbing of the HF part. Blind to a-scaling
//!   (ExchUnscaled / ExchScaledTwice are asserted identities there).
//! * (2) closed-shell UKS (Da == Db) == the merged RKS stress per part: spin
//!   FACTORS (2 vs 1, a dropped spin that is not doubled), the shared XC terms.
//!   BLIND to XcSpinSwap, XcCrossOwn, XcSpinAveraged, PulayFockSwap, PulayKSwap
//!   (asserted blind below, so the claim is measured, not assumed).
//! * (3) 1x1x1 vs the Gamma UKS stress on H3 (Da != Db) and the 1x1x3 vs the
//!   explicit 3-copy supercell: every spin defect and every a-scaling defect at
//!   nk = 1; the supercell adds the Bloch phases and mesh Madelung constant.
//! * (4) FD of the energy under symmetric strain: the stress of the energy as
//!   computed, any wrong factor.
//! * (5) the trace against isotropic scaling.
//!
//! NOT covered: p/d shells and heavy atoms / ECP (every fixture is H/STO-3G),
//! ROKS/ROHF (see `pbc_kuxcstress_ro.rs`), range-separated hybrids / VV10 (refused), meta-GGA (`pbc_kuxc_mgga.rs`),
//! shifted meshes except one ignored FD row, the production (75, 302) grid,
//! no external (PySCF) anchor (its k stress refuses open-shell hybrids).
//!
//! WEEKLY HALF: the three tests whose CI time is >= 120 s (the full
//! refusals / per-spin Fock guard sweep, the 1x1x3 a = 1 wiring anchor and
//! the spin / scaling mutants against the Gamma anchor) run in
//! `validation_pbc_kuxcstress.rs`; the helpers are
//! `kuxcstress_support/mod.rs`. Per-commit canaries for them:
//! `refusals_and_the_per_spin_fock_guard_canary` (named, exxdiv and
//! per-spin tamper refusals), `a_equal_one_with_zero_xc_is_the_uhf_stress_on_the_one_point_mesh`
//! (the same wiring anchor on 1x1x1) and
//! `xc_spin_swap_mutant_is_caught_on_the_one_point_mesh`.

mod common;
#[path = "kuxcstress_support/mod.rs"]
mod support;

use common::*;
use support::*;

#[test]
#[ignore = "the other three rows (dense none, RS-GDF none/ewald) of the a = 1 anchor; ~90 s"]
fn a_equal_one_with_zero_xc_is_the_uhf_stress_per_part_other_rows() {
    for (jk, exxdiv) in [
        (Jk::Dense, ExxDiv::None),
        (Jk::Gdf, ExxDiv::None),
        (Jk::Gdf, ExxDiv::Ewald),
    ] {
        a_equal_one_case(jk, exxdiv);
    }
}

/// (2) CLOSED-SHELL LIMIT == the merged RKS stress, per part. Independent SCF
/// loops (k-UHF skeleton with a polarized kernel vs the closed-shell loop).
/// Artifact hypothesis: a spin-factor slip (2 vs 1) or a mishandled gate breaks
/// equality; physics predicts agreement at the SCF-density floor. BLIND to the
/// alpha/beta asymmetric defects (see `closed_shell_limit_is_blind_to_...`).
#[test]
fn closed_shell_limit_equals_the_rks_stress_per_part() {
    closed_shell_vs_rks(&["PBE0"]);
}

#[test]
#[ignore = "closed-shell UKS vs RKS for LDA and PBE; ~40 s"]
fn closed_shell_limit_equals_the_rks_stress_per_part_lda_pbe() {
    closed_shell_vs_rks(&["LDA", "PBE"]);
}

/// The closed-shell limit is BLIND to the alpha/beta-asymmetric defects (so
/// the Gamma/FD anchors, not this one, must carry them): asserted, not assumed.
/// A dropped beta, J of one spin, and the other exact-exchange/Fock defects are
/// NOT blind and must be visible there.
#[test]
#[ignore = "9 PBE0 stresses; measured 111 s"]
fn closed_shell_limit_is_blind_to_spin_swaps_and_sees_dropped_spins() {
    use KUStressMutation as U;
    let fx = fixture(&H2, G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let base = stress(&fx);
    for sm in [
        U::XcSpinSwap,
        U::XcCrossOwn,
        U::XcSpinAveraged,
        U::PulayFockSwap,
        U::PulayKSwap,
    ] {
        let d = mdiff(
            &stress_mut(&fx, None, Some(sm), None).de_deps,
            &base.de_deps,
        );
        eprintln!("closed-shell blindness {sm:?}: {d:.2e}");
        assert!(d < BAR_CLOSED, "{sm:?} is visible closed-shell: {d:e}");
    }
    for sm in [
        U::XcDropBeta,
        U::XcCrossDropped,
        U::PulayJOneSpin,
        U::PulayNoVxcBeta,
    ] {
        let d = mdiff(
            &stress_mut(&fx, None, Some(sm), None).de_deps,
            &base.de_deps,
        );
        eprintln!("closed-shell visibility {sm:?}: {d:.2e}");
        assert!(d > 1e-6, "{sm:?} is invisible closed-shell: {d:e}");
    }
}

/// Per-commit canary of the weekly (1x1x3)
/// `a_equal_one_with_zero_xc_is_the_uhf_stress_per_part`
/// (`validation_pbc_kuxcstress.rs`): the same wiring anchor (a = 1 + zero XC
/// == `kpoint_uhf_stress` per part, a-scaling mutants exact identities) on the
/// 1x1x1 mesh of the H3 doublet, dense, exxdiv ewald.
#[test]
fn a_equal_one_with_zero_xc_is_the_uhf_stress_on_the_one_point_mesh() {
    a_equal_one_case_on(Jk::Dense, ExxDiv::Ewald, G111);
}

/// Per-commit canary of the weekly
/// `spin_and_scaling_mutants_are_caught_by_the_gamma_anchor`: the XC strain
/// spin swap (`XcSpinSwap`, 1.6e-1 against the Gamma anchor in the full
/// sweep) moves the 1x1x1 PBE0 stress of the H3 doublet by far more than the
/// mutant bar. The production stress it is measured against is the one
/// `one_point_mesh_matches_the_gamma_uks_stress_per_part` anchors to the
/// Gamma UKS stress per commit.
#[test]
fn xc_spin_swap_mutant_is_caught_on_the_one_point_mesh() {
    let fx = fixture(&H3, G111, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let base = stress(&fx);
    let d = mdiff(
        &stress_mut(&fx, None, Some(KUStressMutation::XcSpinSwap), None).de_deps,
        &base.de_deps,
    );
    eprintln!("1x1x1 XcSpinSwap: miss {d:.2e}");
    assert!(d > MUTANT_BAR, "XcSpinSwap survives: {d:e}");
}

/// Per-commit canary of the weekly `refusals_and_the_per_spin_fock_guard`:
/// the typed name and exxdiv refusals and the per-spin Fock tampers (each
/// spin, named), on the H3 doublet PBE0 state. The weekly test adds the
/// accepted-state checks, the functional-mismatch guard cases, the spin
/// swap, the pure-functional entry and the exchange-fraction validation.
#[test]
fn refusals_and_the_per_spin_fock_guard_canary() {
    let fx = fixture(&H3, G111, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let ok_cfg = scfg(Some(ExxDiv::Ewald), None, None, None);
    let run = |f: &str, s: &KUScfResult, cfg: &KUksStressConfig| stress_with(&fx, f, s, cfg);
    for (name, key) in [
        (
            "HSE06",
            "but the dense kernels carry the bare Coulomb exchange; build the J/K with KDenseAftConfig::rsh set",
        ),
        ("wB97X-V", "VV10"),
        ("B2PLYP", "double hybrid"),
    ] {
        let e = run(name, &fx.res.scf, &ok_cfg)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        assert!(e.contains(key), "{name}: {e}");
    }
    let e = run("PBE0", &fx.res.scf, &scfg(None, None, None, None))
        .expect_err("no exxdiv")
        .to_string();
    assert!(
        e.contains("kpoint_uks_stress: a hybrid functional (exact exchange 0.25) needs KRksStressConfig::exxdiv"),
        "{e}"
    );
    let guard_msg = "kpoint_uks_stress: the Fock matrix rebuilt";
    let mut t = fx.res.scf.clone();
    t.fock_alpha[0][(0, 1)] += Complex64::new(1e-6, 0.0);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("alpha tamper")
        .to_string();
    assert!(e.contains(guard_msg) && e.contains("fock_alpha"), "{e}");
    let mut t = fx.res.scf.clone();
    t.fock_beta[0][(0, 1)] += Complex64::new(1e-6, 0.0);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("beta tamper")
        .to_string();
    assert!(e.contains(guard_msg) && e.contains("fock_beta"), "{e}");
    let mut t = fx.res.scf.clone();
    t.converged = false;
    assert!(run("PBE0", &t, &ok_cfg).is_err());
}

/// (3a) 1x1x1 == the Gamma UKS stress, PER PART, on the open-shell H3 doublet
/// (real lattice sums vs complex Bloch AOs; separate SCFs). Sees the spin
/// channels (Da != Db) that the closed-shell limit cannot. Prediction: every
/// part <= 1e-9 (both SCFs converge to ~1e-10 in the density).
#[test]
fn one_point_mesh_matches_the_gamma_uks_stress_per_part() {
    one_point_case("PBE0");
}

#[test]
#[ignore = "PBE and LDA rows of the 1x1x1 Gamma UKS anchor; ~40 s"]
fn one_point_mesh_matches_the_gamma_uks_stress_per_part_pbe_lda() {
    one_point_case("PBE");
    one_point_case("LDA");
}

/// (3b) SUPERCELL ANCHOR on the non-TRIM 1x1x3 mesh, open shell: the k stress
/// per cell is the Gamma UKS stress of the explicit 3-copy supercell
/// (multiplicity (Na - Nb) N_k + 1 = 4; sigma equal, dE/de / 3), per part. With
/// Ewald the supercell Gamma Madelung constant IS the mesh constant. Same
/// explicit cutoff on both sides. Prediction <= 1e-9 per part.
#[test]
#[ignore = "9-atom Gamma UKS supercell; ~60 s"]
fn k_mesh_uks_stress_is_the_gamma_supercell_stress_per_part() {
    let f = "PBE0";
    let gam = gamma_side(&supercell_pos(&H3, 4.0, [1, 1, 3], 4), f);
    let fx = fixture(&H3, G113, f, Jk::Dense, ExxDiv::Ewald);
    let k = stress(&fx);
    for (n, d) in gamma_rows(&k, &gam, 3.0) {
        eprintln!("{f} H3 1x1x3 vs supercell: {n:<10} {d:.2e}");
        assert!(d < BAR_GAMMA, "{f} {n}: {d:e}");
    }
    assert!(mmax(&k.hf.parts.madelung) > 1e-6);
}

/// (4) FD of ferric's own UKS energy under symmetric strain at fixed
/// fractional coordinates (fresh SCF per displaced cell, mesh rebuilt on the
/// strained cell with the same integer numerators, shared cutoff 14). Three
/// components (diagonal along the mesh axis, a shear, the transverse diagonal).
#[test]
#[ignore = "3 components x 2 SCFs; measured 36 s"]
fn pbe0_uks_stress_matches_fd_dense_ewald_1x1x3() {
    fd_case(
        "PBE0",
        Jk::Dense,
        ExxDiv::Ewald,
        G113,
        &[(2, 2), (0, 2), (0, 0)],
    );
}

#[test]
#[ignore = "PBE row of the FD anchor (6 SCFs); ~60 s"]
fn pbe_uks_stress_matches_fd_dense_none_1x1x3() {
    fd_case(
        "PBE",
        Jk::Dense,
        ExxDiv::None,
        G113,
        &[(2, 2), (0, 2), (1, 0)],
    );
}

#[test]
#[ignore = "RS-GDF row of the FD anchor (6 SCFs); ~120 s"]
fn pbe0_uks_stress_matches_fd_rsgdf_ewald_1x1x3() {
    fd_case("PBE0", Jk::Gdf, ExxDiv::Ewald, G113, &[(2, 2), (0, 1)]);
}

#[test]
#[ignore = "8-point shifted mesh, 2 components x 2 SCFs; needs a quiet box"]
fn pbe0_uks_stress_matches_fd_on_the_monkhorst_pack_mesh() {
    let m = Mesh {
        n: [2, 2, 2],
        mp: true,
    };
    fd_case("PBE0", Jk::Dense, ExxDiv::Ewald, m, &[(2, 2), (0, 1)]);
}

/// (5) Pressure from isotropic scaling: sum_a dE/de_aa equals the FD of E under
/// eps = h*I (PBE, dense, none: the pure-GGA twin of the PBE0 FD above).
#[test]
fn trace_matches_isotropic_scaling_fd() {
    let fx = fixture(&H3, G113, "PBE", Jk::Dense, ExxDiv::None);
    let s = stress(&fx);
    let tr = s.de_deps[0][0] + s.de_deps[1][1] + s.de_deps[2][2];
    let iso = |h: f64| {
        let e = [[h, 0.0, 0.0], [0.0, h, 0.0], [0.0, 0.0, h]];
        let c = fx.cell.strained(&e).unwrap();
        energy_on(&c, G113, "PBE", Jk::Dense, ExxDiv::None)
    };
    let fd = (iso(FD_H) - iso(-FD_H)) / (2.0 * FD_H);
    eprintln!(
        "PBE trace: analytic {tr:.9e} FD(isotropic) {fd:.9e} miss {:.2e}",
        (tr - fd).abs()
    );
    assert!((tr - fd).abs() < 3.0 * FD_BAR);
}

/// PROBE: choose the FD step. All nine components of PBE0 (dense, Ewald,
/// 1x1x3 H3) at four steps; the printout is the floor and the window.
#[test]
#[ignore = "probe: 4 steps x 18 SCFs, needs a quiet box; run with --nocapture"]
fn fd_window_probe() {
    let fx = fixture(&H3, G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let s = stress(&fx);
    for h in [1e-5, 2.5e-5, 5e-5, 1e-4] {
        let fd = fd_components(&fx.cell, G113, "PBE0", Jk::Dense, ExxDiv::Ewald, h, &ALL9);
        let mut row = String::new();
        let mut worst = 0.0_f64;
        for (&(i, j), d) in ALL9.iter().zip(&fd) {
            let m = (d - s.de_deps[i][j]).abs();
            worst = worst.max(m);
            row += &format!(" {m:.1e}");
        }
        eprintln!("h={h:.1e} worst {worst:.2e}; per component (00 01 02 10 11 12 20 21 22):{row}");
    }
}

#[test]
#[ignore = "all-nine FD for PBE0 dense Ewald and PBE dense none (36 SCFs); needs a quiet box"]
fn uks_stress_matches_fd_all_nine() {
    for (f, exxdiv) in [("PBE0", ExxDiv::Ewald), ("PBE", ExxDiv::None)] {
        let fx = fixture(&H3, G113, f, Jk::Dense, exxdiv);
        let s = stress(&fx);
        let fd = fd_components(&fx.cell, G113, f, Jk::Dense, exxdiv, FD_H, &ALL9);
        let mut worst = 0.0_f64;
        for (&(i, j), d) in ALL9.iter().zip(&fd) {
            worst = worst.max((d - s.de_deps[i][j]).abs());
        }
        eprintln!("{f} {exxdiv:?}: all-nine FD miss {worst:.2e}");
        assert!(worst < FD_BAR, "{f} {exxdiv:?}: {worst:e}");
    }
}

// ------------------------------------------------------ bitwise / guards

#[test]
#[ignore = "two full PBE0 stresses at 1 and 4 threads; thread bit-identity of the shared assembly is also pinned by pbc_parallel_bitwise"]
fn uks_stress_is_bit_identical_across_threads() {
    let fx = fixture(&H3, G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| stress(&fx))
    };
    let (a, b) = (run(1), run(4));
    assert_eq!(a.de_deps, b.de_deps);
    assert_eq!(a.xc_ao, b.xc_ao);
    assert_eq!(a.e_xc, b.e_xc);
}

/// A loosely converged UKS state is refused with the commutator message; the
/// same state with the gate off returns a stress that is wrong by far more than
/// the FD bar (so the gate matters), and a tight state passes the gate.
#[test]
#[ignore = "two UKS SCFs + 2 stresses; measured 45 s"]
fn loose_uks_scf_is_refused_by_the_stationarity_gate() {
    let tight = fixture(&H3, G111, "PBE", Jk::Dense, ExxDiv::None);
    let st = stress(&tight);
    assert!(
        st.hf.commutator < UKS_COMMUTATOR_TOL / 100.0,
        "{:e}",
        st.hf.commutator
    );
    SETTINGS.with(|s| s.set(LOOSE_UKS));
    let loose = fixture(&H3, G111, "PBE", Jk::Dense, ExxDiv::None);
    let e = stress_with(&loose, "PBE", &loose.res.scf, &scfg(None, None, None, None))
        .expect_err("loose SCF accepted")
        .to_string();
    eprintln!("loose UKS: {e}");
    assert!(
        e.contains("kpoint_uks_stress: the spin Fock/density commutator")
            && e.contains("exceeds 1e-5"),
        "{e}"
    );
    let off = stress_mut(
        &loose,
        None,
        Some(KUStressMutation::StationarityGateOff),
        None,
    );
    let miss = mdiff(&off.de_deps, &st.de_deps);
    eprintln!(
        "loose UKS: commutator {:.3e}, stress error with the gate off {miss:.3e}",
        off.hf.commutator
    );
    assert!(off.hf.commutator > UKS_COMMUTATOR_TOL);
    // First-order law miss ~ slope * commutator with the measured slope 0.42
    // (6.6e-5 / 1.57e-4, one point); two-sided, a factor of 2 each way.
    let slope = miss / off.hf.commutator;
    eprintln!("loose UKS: slope miss/commutator = {slope:.3}");
    assert!(
        (0.21..=0.84).contains(&slope),
        "gate-off miss {miss:e} is not first order in the commutator {:e} (slope {slope})",
        off.hf.commutator
    );
    assert!(miss > 10.0 * FD_BAR);
}

/// The stationarity gate fires in the DEFAULT set: a deliberately loose SCF
/// (one SCF, one stress) is refused with the exact message. Deleting the gate
/// in `uks_stress_with` makes this fail (the stress is returned).
#[test]
fn loose_uks_scf_fires_the_stationarity_gate_with_the_exact_message() {
    SETTINGS.with(|s| s.set(LOOSE_UKS));
    let loose = fixture(&H3, G111, "PBE", Jk::Dense, ExxDiv::None);
    let e = stress_with(&loose, "PBE", &loose.res.scf, &scfg(None, None, None, None))
        .expect_err("loose SCF accepted")
        .to_string();
    let head = "kpoint_uks_stress: the spin Fock/density commutator max|F D S - S D F| = ";
    let tail = " exceeds 1e-5: not a stationary state (the stress is first order in it); \
                converge the SCF tighter";
    let body = e.strip_prefix("General error: ").unwrap_or(&e);
    let mid = body
        .strip_prefix(head)
        .and_then(|r| r.strip_suffix(tail))
        .unwrap_or_else(|| panic!("message differs: {e}"));
    let c: f64 = mid.parse().unwrap_or_else(|_| panic!("number {mid}: {e}"));
    assert!(c > UKS_COMMUTATOR_TOL, "{c:e}");
}

/// Budget gates of the UKS stress. (i) A 4 MB budget fails typed and a 16 MB
/// budget reproduces the AMPLE-budget stress BITWISE (null XC, a = 0.5, no grid
/// pass). (ii) The Fock-guard reservation is read from the typed error the
/// PRODUCTION path raises when the budget holds the per-k arrays but not the
/// guard matrices: it must be ten N_k n^2 complex arrays
/// (D_a, D_b, J_a, J_b, K_a, K_b, F_a, F_b, V_a, V_b). A charge reverted to the
/// closed-shell five fails the equality.
#[test]
#[ignore = "null-XC budget bisection; measured 38 s"]
fn budgets_either_reproduce_the_uks_stress_or_fail_typed() {
    let z = zero_xc(&H3, G113, Jk::Dense, ExxDiv::Ewald, 0.5);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let mut scf = z.scf.clone();
    let (ja, ka) = jk_mats(&z, &z.scf.density_alpha);
    let (jb, kb) = jk_mats(&z, &z.scf.density_beta);
    for q in 0..scf.fock_alpha.len() {
        for (f, k) in [(&mut scf.fock_alpha, &ka), (&mut scf.fock_beta, &kb)] {
            let mut m = &z.hk.h[q] + &ja[q] + &jb[q];
            m.scaled_add(Complex64::new(-0.5, 0.0), &k[q]);
            f[q] = m;
        }
    }
    let z = Zx { scf, ..z };
    let at = |b: usize| {
        let mut cfg = scfg(Some(ExxDiv::Ewald), None, None, None);
        cfg.base.stress.budget_bytes = Some(b);
        null_stress(&z, 0.5, &cfg)
    };
    let _ = hcfg;
    let full = at(AMPLE).unwrap();
    let e = at(4 << 20).expect_err("4 MB must not hold the strain K list");
    eprintln!("4 MB: {e}");
    assert_eq!(at(16 << 20).unwrap().de_deps, full.de_deps);
    let (mut lo, mut hi) = (0usize, 1usize << 20);
    let mut guard_msg = None;
    for _ in 0..40 {
        let mid = lo + (hi - lo) / 2;
        match at(mid) {
            Err(e) => {
                let m = e.to_string();
                if m.contains("Fock-guard matrices") {
                    guard_msg = Some(m);
                    break;
                } else if m.contains("k stress per-k complex") || m.contains("residue-folded") {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            Ok(_) => hi = mid,
        }
    }
    let m = guard_msg.expect("the bisection must land in the guard band");
    eprintln!("guard gate: {m}");
    assert!(
        m.contains("D_a, D_b, J_a, J_b, K_a, K_b, F_a, F_b, V_a, V_b"),
        "{m}"
    );
    let n = z.prep.nbasis();
    assert_eq!(charge_of(&m), 10 * z.mesh.nk() * n * n * 16, "{m}");
}

/// The XC chunk charge, read from the typed error of the PRODUCTION PBE0 path.
/// Stage order in production: open_ledger, strain grid / AO list, XC chunk gate,
/// Fock-guard reservation, then the HF assembly (strain K list, ...). A budget
/// error is therefore "too small" iff it comes from a stage BEFORE the chunk gate
/// (matched by marker), "too big" iff it comes from a later stage or the run
/// succeeds. At most 16 probes over [0, 8 MB]. The charge is compared with the
/// module-doc arithmetic (40 AO planes per k and point, eight work planes, four
/// N_k nbf^2 arrays).
#[test]
#[ignore = "five-probe budget bisection plus the PBE0 SCF fixture; measured 73 s (over the 60 s default-set guideline); fails if the chunk charge term is removed (verified)"]
fn xc_chunk_charge_is_read_from_the_production_error() {
    let fx = fixture(&H3, G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let at = |b: usize| {
        let mut cfg = scfg(Some(ExxDiv::Ewald), None, None, None);
        cfg.base.stress.budget_bytes = Some(b);
        stress_with(&fx, &fx.functional, &fx.res.scf, &cfg)
    };
    let early = [
        "per-k complex",
        "residue-folded",
        "grid",
        "AO list",
        "translation",
    ];
    let (mut lo, mut hi) = (0usize, 8usize << 20);
    let mut found = None;
    for _ in 0..16 {
        let mid = lo + (hi - lo) / 2;
        match at(mid) {
            Err(e) => {
                let m = e.to_string();
                eprintln!("b={mid}: {}", &m[..m.len().min(140)]);
                if m.contains("UKS XC stress AO moment chunk") {
                    found = Some(m);
                    break;
                }
                if early.iter().any(|k| m.contains(k)) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            Ok(_) => {
                eprintln!("b={mid}: ok");
                hi = mid;
            }
        }
    }
    let m = found.expect("the bisection must land on the XC chunk gate");
    eprintln!("xc chunk gate: {m}");
    let pts: usize = m
        .split('(')
        .nth(1)
        .unwrap()
        .split(' ')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let (nk, n) = (fx.mesh.nk(), fx.prep.nbasis());
    let want = nk * n * pts * 16 * 40 + n * pts * 16 * 8 + nk * n * n * 16 * 4;
    assert_eq!(charge_of(&m), want, "{m}");
}

/// Production-wired spin-specific, XC and exchange-scale defects against the
/// Gamma UKS stress at 1x1x1 (H3, PBE0, Ewald): the anchor that sees BOTH the
/// spin defects (Da != Db) and the a-scaling defects (a = 0.25, Madelung on).
/// Each must miss the Gamma reference by far more than the Gamma bar (the
/// correct residual is <= BAR_GAMMA = 1e-9; the mutant bar is 1e-5).
#[test]
#[ignore = "full mutant sweep against the Gamma UKS stress; measured 384 s"]
fn mutants_are_caught_by_the_gamma_anchor() {
    use KStressMutation as S;
    use KXcStressMutation as X;
    let f = "PBE0";
    let gam = gamma_side(&cell_of(&H3), f);
    let fx = fixture(&H3, G111, f, Jk::Dense, ExxDiv::Ewald);
    let sigma_gam = gam.de_deps;
    let check = |label: String, s: KUksStress| {
        let d = mdiff(&s.de_deps, &sigma_gam);
        eprintln!("1x1x1 vs Gamma under {label}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{label} survives the Gamma anchor: {d:e}");
    };
    // Control: the unmutated stress passes the same comparison.
    let d0 = mdiff(&stress(&fx).de_deps, &sigma_gam);
    assert!(d0 < BAR_GAMMA, "{d0:e}");
    for sm in all_spin_mutants() {
        check(format!("{sm:?}"), stress_mut(&fx, None, Some(sm), None));
    }
    for xm in [
        X::NoAo,
        X::NoPointMotion,
        X::NoWeightDeriv,
        X::NoGga,
        X::NoHessian,
        X::NoStrainMoment,
        X::WrongFactor,
        X::NoInvNk,
        X::PulayNoVxc,
        X::ExchUnscaled,
        X::ExchScaledTwice,
    ] {
        // NoInvNk is an identity at one k-point (1/N_k = 1): excluded.
        if xm == X::NoInvNk {
            assert_eq!(
                mdiff(
                    &stress_mut(&fx, Some(xm), None, None).de_deps,
                    &stress(&fx).de_deps
                ),
                0.0
            );
            continue;
        }
        check(format!("{xm:?}"), stress_mut(&fx, Some(xm), None, None));
    }
    for hm in [
        S::MadelungUnscaled,
        S::CrossSpinK,
        S::SwapSpinFock,
        S::NoPulay,
    ] {
        check(format!("{hm:?}"), stress_mut(&fx, None, None, Some(hm)));
    }
}

/// The k-dependent defects need nk > 1: against the unmutated 1x1x3 stress
/// (itself anchored by the supercell and the FD). The correct residuals are
/// <= 1e-9; every mutant below misses by far more.
#[test]
#[ignore = "7 PBE0 UKS stresses on 1x1x3; ~120 s"]
fn k_dependent_mutants_are_caught_on_the_mesh() {
    use KXcStressMutation as X;
    let fx = fixture(&H3, G113, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let base = stress(&fx);
    for (xm, sm, hm) in [
        (Some(X::NoInvNk), None, None),
        (Some(X::NoPhase), None, None),
        (Some(X::ConjPhase), None, None),
        (Some(X::KWrongDensity), None, None),
        (None, None, Some(KStressMutation::GammaMadelung)),
        (None, None, Some(KStressMutation::CrossSpinK)),
        (None, Some(KUStressMutation::XcSpinSwap), None),
    ] {
        let d = mdiff(&stress_mut(&fx, xm, sm, hm).de_deps, &base.de_deps);
        eprintln!("1x1x3 {xm:?}/{sm:?}/{hm:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{xm:?}/{sm:?}/{hm:?} survives: {d:e}");
    }
}

/// Uniform grid: no partition (symmetric stress to roundoff) and the volume
/// weight term; FD of the uniform-grid energy.
#[test]
#[ignore = "uniform-grid FD (2 components x 2 SCFs); needs a quiet box"]
fn uniform_grid_stress_is_symmetric_and_matches_fd() {
    use ferric_pbc::kuks::solve_kuks_on_grid;
    let n = [24, 24, 24];
    let run = |cell: &Cell| {
        let prep = prep_for(cell, &pyscf_sto3g_h());
        let mesh = KPointMesh::gamma_centred(cell, [1, 1, 3]).unwrap();
        let cfg = kuks_cfg(cell, "PBE", Jk::Dense, ExxDiv::None);
        let grid = ferric_pbc::PeriodicGrid::uniform(cell, n).unwrap();
        solve_kuks_on_grid(cell, &prep, None, &mesh, &grid, &cfg).unwrap()
    };
    let cell = cell_of(&H3);
    let r = run(&cell);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
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
    let mut cfg = scfg(None, None, None, None);
    cfg.base.grid = KXcGrid::Uniform(n);
    let s = kpoint_uks_stress(
        &cell,
        &prep,
        &mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &hk,
        KStressJk::Dense(&eri),
        &r.scf,
        "PBE",
        &cfg,
    )
    .unwrap();
    eprintln!("uniform-grid antisymmetry {:.2e}", s.antisymmetry());
    assert!(s.antisymmetry() < 1e-9);
    let mut worst = 0.0_f64;
    for (i, j) in [(2, 2), (0, 2)] {
        let e = |h: f64| run(&strained(&cell, i, j, h)).scf.energy;
        let d = (e(FD_H) - e(-FD_H)) / (2.0 * FD_H);
        worst = worst.max((s.de_deps[i][j] - d).abs());
    }
    eprintln!("uniform-grid UKS stress vs FD worst {worst:.2e}");
    assert!(worst < FD_BAR, "{worst:e}");
}
