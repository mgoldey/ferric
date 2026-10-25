//! Analytic forces of the RESTRICTED-OPEN-SHELL k-point ROKS energy for
//! RANGE-SEPARATED hybrids on the dense-AFT J/K
//! (`ferric_pbc::kuxcgrad::kpoint_roks_gradient`, module doc
//! "Range-separated hybrids (UKS, ROKS)"), and the Gamma-point ROKS force
//! routed through the 1x1x1 mesh
//! (`ferric_pbc::gamma_rsh::gamma_roks_rsh_gradient`).
//!
//! ROKS builds the SAME per-spin Focks as UKS (`F_σ = h + J[D] − K[D_σ] +
//! V_σ`, the attenuated `K[D_σ]` scaled by 1 with the mixed Madelung constant
//! inside) before the Roothaan coupling, so the force is the UKS RSH
//! assembly (#505: exchange terms `(1, D_α), (1, D_β)` weighted by `f(K) =
//! c_sr + (c_lr − c_sr) e^{−K²/4ω²}`, `J` bare, M-term `−v_M^mix Σ_σ D_σ S
//! D_σ`) on the ROHF spin densities with `W_RO` and the ROHF orbital-gradient
//! gate. Oracles (artifact hypothesis in each test's doc):
//!
//! (a) singlet k-ROKS == k-RKS RSH force (separate SCFs; blind to every α/β
//!     mix-up and to the open-shell part of `W_RO`, asserted, which is why
//!     the rest run on cells with an open shell);
//! (b) `c_sr == c_lr == c` reproduces the global-hybrid ROKS force with
//!     fraction `c` (zero XC, two ROKS SCFs, H3 doublet; ewald here, none
//!     weekly). Both sides share the M-term sandwich and `W_RO`, so this
//!     anchor is blind to spin errors in them; FD is not;
//! (c) the 1x1x2 mesh force equals the force on one copy in the explicit
//!     2-cell supercell (weekly), and Gamma (`gamma_roks_rsh` +
//!     `gamma_roks_rsh_gradient`) equals the 1x1x1 mesh (CAM-B3LYP / ewald
//!     here, all four cases weekly);
//! (d) central FD of ferric's own k-ROKS energy (fresh SCF per displacement;
//!     PySCF has no periodic ROKS gradient): two CAM-B3LYP components of the
//!     H3 doublet here; every component of the H3 doublet and the H4 triplet
//!     for both functionals, the h sweep and ewald vs none weekly
//!     (`validation_kroks_rsh_force.rs`).
//!
//! MEASURED here (2026-10-23): singlet ROKS - RKS <= 1.2e-13 per part; FD
//! 4.51e-9 / 2.02e-10 (bar 1e-7); equal fractions 3.4e-16; Gamma - 1x1x1
//! 3.4e-15 (spin-Fock guard 1.0e-15; Roothaan fallback identical, 0.0); all
//! under ANCHOR_TOL 1e-9.
//!
//! MUTATION LEDGER: measured numbers in `validation_kroks_rsh_force.rs`
//! (`full_mutation_ledger`, module doc table). Per commit: the spin, erf-piece
//! and `W_RO` defects on CAM-B3LYP, the singlet blindness, and the
//! unrestricted-form `W` off stationarity (refused by the gate).
//!
//! NOT covered: RS-GDF (`pbc_kuks_rsh_rsgdf_force.rs`), the stress
//! (refused), wB97X (accepted, not force-tested), p/d shells (H/STO-3G only),
//! shifted meshes, a Gamma ROKS on a caller-supplied grid, Python / CLI (the
//! binding derivative gate still refuses open-shell range separation).

mod common;
#[path = "kroks_rsh_force_support/mod.rs"]
mod ro;
#[path = "kuks_rsh_force_support/mod.rs"]
mod uks;

use common::*;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::kgrad::KGradJk;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kuxcgrad::KUSpinMutation;
use ro::*;
use uks::*;

// ------------------------------------------------------- (a) singlet limit

/// (a) SINGLET (CAM-B3LYP; HSE06 weekly): on the H2 singlet the k-ROKS RSH
/// force equals the k-RKS RSH force (#486; independent closed-shell SCF and
/// entry), per part. Physics: the SCF floor; artifact: a per-spin factor slip
/// (a ½ on the spin exchange terms, a scale other than 1, half of `W`) is
/// O(1e-2) (ExchDropBeta, RoWDropCross). The spin-assignment defects are
/// INVISIBLE here (`D_α = D_β`), asserted.
#[test]
fn singlet_roks_is_the_rks_rsh_force() {
    let (rows, fx, u) = ro_singlet_rows(CAM);
    for (n, d) in rows {
        eprintln!("CAM-B3LYP singlet ROKS vs RKS {n}: {d:.2e}");
        assert!(d < ANCHOR_TOL, "{n}: {d:e}");
    }
    for (n, m, sm) in ro_ledger_mutants() {
        if RO_SPIN_ONLY.contains(&n) || n == "ExchDropBeta" || n == "RoWDropCross" {
            let miss = diff(&ro_force_m(&fx, m, sm).unwrap().grad, &u.grad);
            eprintln!("CAM-B3LYP ROKS singlet: {n} miss {miss:.2e}");
            if !RO_SPIN_ONLY.contains(&n) {
                assert!(miss > 1e-3, "{n}: {miss:e}");
            } else {
                assert!(
                    miss < ANCHOR_TOL,
                    "{n} should be blind on a singlet: {miss:e}"
                );
            }
        }
    }
}

// -------------------------------------------------- (d) FD, per-commit row

/// (d) Central FD (h = 1e-4) of the CAM-B3LYP k-ROKS energy, H3 doublet,
/// 1x1x2, exxdiv ewald, two components. Physics: the h^2 floor (weekly
/// sweep); artifact: every ledger mutant misses by > 100 FD bars
/// (`full_mutation_ledger`).
#[test]
fn cam_b3lyp_roks_force_matches_fd() {
    let fx = ro_fixture(Kind::Tri(2), &H3D, [1, 1, 2], CAM, ExxDiv::Ewald);
    assert!(ro_spin_asymmetry(&fx.res.scf) > 0.1, "open shell");
    let g = ro_force(&fx);
    for (a, x) in [(0, 2), (2, 0)] {
        let d = ro_fd(
            Kind::Tri(2),
            &H3D,
            a,
            x,
            FD_H,
            [1, 1, 2],
            CAM,
            ExxDiv::Ewald,
        );
        let r = (g.grad[(a, x)] - d).abs();
        eprintln!(
            "CAM-B3LYP ROKS H3 ({a},{x}): analytic {:+.10e} FD {d:+.10e} |diff| {r:.2e}",
            g.grad[(a, x)]
        );
        assert!(r < FD_BAR, "({a},{x}): {r:e}");
    }
    assert!(max_abs(&g.hf.parts.exchange) > 1e-4, "K walks exercised");
}

// ----------------------------------------------- (b) equal-fraction anchor

/// (b) EQUAL FRACTIONS on the H3 doublet: `c_sr = c_lr = c` makes the
/// attenuated kernel `c 4π/K²` and the mixed Madelung `c v_M`, i.e. the
/// global-hybrid ROKS with fraction `c`. Two ROKS SCFs (RSH kernels at scale
/// 1 vs bare kernels at scale c, zero XC) and two force paths (kernel factor
/// vs exchange-coefficient scaling). Artifact: the factor on J or dropped, a
/// plain Madelung, the M-term on the total density, or `D_α` in the β
/// exchange walk move it by > 1e-3 (the Madelung ones are blind under exxdiv
/// none by construction).
#[test]
fn equal_fractions_is_the_global_hybrid_roks_force() {
    // exxdiv none (where the Madelung defects are blind) runs weekly.
    ro_equal_fraction_case(ExxDiv::Ewald);
}

// ----------------------------------------------------------- (c) Gamma

/// (c) GAMMA == 1x1x1 on the H3 doublet (CAM-B3LYP, ewald; the other three
/// cases weekly): the Gamma ROKS RSH force (Gamma SCF, J, hcore and grid,
/// routed into the 1x1x1 ROKS force, Fock guard on the SCF's stored spin
/// Focks `rohf_spin_focks`) against the 1x1x1 k-ROKS SCF + force. Physics:
/// the SCF floor; artifact: a glue slip (spin densities swapped, `F_eff` read
/// as a spin Fock, the wrong Madelung) is O(1e-3) or fails the routed guard.
/// The guard is mutation-checked: another grid, another `exxdiv` (with a
/// matching exchange object), another functional (likewise) and an
/// equal-and-opposite spin-Fock tamper on one AO element (whose
/// closed/open/virtual diagonal-block part the Roothaan `F_eff` cannot see)
/// are each refused by the spin guard; the
/// Roothaan fallback (no stored spin Focks) gives the same force and still
/// refuses another grid. Also: an unrestricted result, an RS-GDF exchange.
#[test]
fn gamma_roks_rsh_force_is_the_one_point_mesh_force() {
    let g = ro_gamma_fixture(CAM, ExxDiv::Ewald);
    let call = |scf: &ferric_scf::result::ScfResult,
                gc: &ferric_pbc::rohf::GammaRoksConfig,
                ex: &ferric_pbc::gamma_rsh::GammaRshExchange| {
        ferric_pbc::gamma_rsh::gamma_roks_rsh_gradient(
            &g.cell,
            &g.prep,
            &g.hcfg,
            scf,
            gc,
            ex,
            &kgrad_cfg(),
        )
    };
    let spin_guard = |e: String, what: &str| {
        assert!(
            e.contains("differs from the SCF's stored fock_"),
            "{what}: {e}"
        )
    };
    assert!(
        g.scf.rohf_spin_focks.is_some(),
        "Gamma ROKS keeps spin Focks"
    );
    let hc = ferric_pbc::hcore::periodic_hcore(&g.cell, &g.prep, &g.hcfg).unwrap();
    let dense_ex = |f: &str, x: ExxDiv| {
        ferric_pbc::gamma_rsh::GammaRshExchange::build_dense(
            &g.cell,
            &g.prep,
            &hc.s,
            x,
            &dense_cfg(None),
            rsh_of(f),
        )
        .unwrap()
    };
    // Another grid.
    let mut other = g.gc.clone();
    other.grid = ferric_pbc::dft::PeriodicGridConfig::with_size(30, 110);
    other.grid.neighbour_cutoff = Some(CUTOFF);
    spin_guard(
        call(&g.scf, &other, &g.ex)
            .expect_err("another grid")
            .to_string(),
        "grid",
    );
    // Another exxdiv (exchange object consistent with the config, not the SCF).
    let mut none = g.gc.clone();
    none.exxdiv = ExxDiv::None;
    spin_guard(
        call(&g.scf, &none, &dense_ex(CAM, ExxDiv::None))
            .expect_err("another exxdiv")
            .to_string(),
        "exxdiv",
    );
    // Another functional (likewise).
    let mut hse = g.gc.clone();
    hse.functional = "HSE06".into();
    spin_guard(
        call(&g.scf, &hse, &dense_ex("HSE06", ExxDiv::Ewald))
            .expect_err("another functional")
            .to_string(),
        "functional",
    );
    // Equal-and-opposite spin change on one AO element: its part on the
    // closed/open/virtual diagonal blocks is invisible to F_eff (½(δF_α +
    // δF_β) = 0 there); refused by the spin guard.
    let mut tamper = g.scf.clone();
    if let Some((fa, fb)) = tamper.rohf_spin_focks.as_mut() {
        fa[(0, 0)] += 1e-6;
        fb[(0, 0)] -= 1e-6;
    }
    spin_guard(
        call(&tamper, &g.gc, &g.ex)
            .expect_err("spin-Fock tamper")
            .to_string(),
        "tamper",
    );
    let mut uhf = g.scf.clone();
    uhf.spin = ferric_scf::result::Spin::Unrestricted;
    let e = call(&uhf, &g.gc, &g.ex)
        .expect_err("unrestricted result accepted")
        .to_string();
    assert!(e.contains("restricted-open-shell (ROKS) results"), "{e}");
    let aux = prep_for(&g.cell, &ferric_core::basis::bundled("cc-pvdz-ri").unwrap());
    let rcfg = ferric_pbc::rsgdf::kpoint::KRsGdfConfig {
        gdf: ferric_pbc::rsgdf::RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::Ewald,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    };
    let ex_gdf = ferric_pbc::gamma_rsh::GammaRshExchange::build_rsgdf(
        &g.cell,
        &g.prep,
        &aux,
        &hc.s,
        ExxDiv::Ewald,
        &rcfg,
        rsh_of(CAM),
    )
    .unwrap();
    let e = call(&g.scf, &g.gc, &ex_gdf)
        .expect_err("RS-GDF Gamma exchange accepted")
        .to_string();
    assert!(e.contains("RS-GDF") && e.contains("no derivative"), "{e}");
    let gf = ro_gamma_force(&g).unwrap();
    let k = ro_force(&ro_fixture(
        Kind::Tri(2),
        &H3D,
        [1, 1, 1],
        CAM,
        ExxDiv::Ewald,
    ));
    let d = diff(&gf.grad, &k.grad);
    // The Roothaan fallback (a result without stored spin Focks).
    let mut bare = g.scf.clone();
    bare.rohf_spin_focks = None;
    let gr = call(&bare, &g.gc, &g.ex).unwrap();
    let dr = diff(&gr.grad, &gf.grad);
    let e = call(&bare, &other, &g.ex)
        .expect_err("another grid (Roothaan fallback)")
        .to_string();
    assert!(e.contains("stored Roothaan fock_eff"), "{e}");
    eprintln!(
        "CAM-B3LYP ewald: Gamma vs 1x1x1 ROKS force {d:.2e} (spin guard {:.1e}, gate {:.1e}); \
         Roothaan fallback vs spin guard {dr:.2e} (guard {:.1e})",
        gf.hf.fock_mismatch, gf.hf.commutator, gr.hf.fock_mismatch
    );
    assert!(d < ANCHOR_TOL, "{d:e}");
    assert!(dr < ANCHOR_TOL, "{dr:e}");
    assert!(max_abs(&gf.grad) > 1e-2);
}

// ----------------------------------------------------------- mutants

/// The spin, erf-piece and `W_RO` defects on the H3 doublet (CAM-B3LYP,
/// 1x1x2, ewald; the full ledger for both functionals is weekly): each moves
/// the FD-validated force by > 100 FD bars, and the erf piece is live on this
/// K set (largest weight 0.172).
#[test]
fn open_shell_spin_erf_and_w_ro_mutants_are_caught() {
    let cell = cell_of(Kind::Tri(2), &H3D);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    assert!(max_erf_weight(&cell, &mesh, rsh_of(CAM)) > 0.1);
    let fx = ro_fixture(Kind::Tri(2), &H3D, [1, 1, 2], CAM, ExxDiv::Ewald);
    let base = ro_force(&fx);
    for (n, m, sm) in ro_ledger_mutants() {
        if RO_SPIN_ONLY.contains(&n)
            || matches!(n, "ExchDropBeta" | "RshNoErfPiece" | "RoWDropCross")
        {
            let miss = diff(&ro_force_m(&fx, m, sm).unwrap().grad, &base.grad);
            eprintln!("CAM-B3LYP ROKS H3 1x1x2 ewald: {n} miss {miss:.2e}");
            assert!(miss > 100.0 * FD_BAR, "{n}: {miss:e}");
        }
    }
}

/// The unrestricted-form `W = Σ_σ D_σ F_σ D_σ` instead of `W_RO`
/// (`RoWUhfForm`) is an identity at a converged ROKS state (they differ by ½
/// the closed-open β orbital gradient) and is therefore NOT a defect there;
/// off stationarity it is wrong, and the orbital-gradient gate is what
/// refuses that state. On the H4 triplet (1x1x1, CAM-B3LYP; NOT the H3
/// doublet, whose inversion symmetry makes the closed-open block, and with
/// it this difference, vanish for any state: measured 1.1e-16 on a loose
/// one): the production force and `RoWUhfForm` on a loosely converged state
/// are refused with the gate message. With the gate off neither `W` form is
/// right on that state (both miss the tight force by more than 10 FD bars:
/// the state is not stationary), and the two forms differ (`RoWUhfFormGateOff`
/// vs `RoGateOff`), which is the assertion that sees the swap. Measured
/// (loose SCF 1e-3 / 1e-1): g 3.6e-5, gate-off miss 1.0e-5 (W_UKS) and
/// 1.5e-5 (W_RO), W_UKS - W_RO 4.9e-6; at convergence 1.1e-14.
#[test]
fn unrestricted_w_is_an_identity_at_convergence_and_refused_off_it() {
    let tight = ro_fixture(Kind::Tri(3), &H4T, [1, 1, 1], CAM, ExxDiv::Ewald);
    let gt = ro_force(&tight);
    let id = diff(
        &ro_force_m(&tight, None, Some(KUSpinMutation::RoWUhfForm))
            .unwrap()
            .grad,
        &gt.grad,
    );
    eprintln!(
        "RoWUhfForm at convergence: {id:.2e} (gate value {:.1e})",
        gt.hf.commutator
    );
    assert!(id < 1e-8, "identity at convergence: {id:e}");
    let loose = ro_fixture_conv(
        cell_of(Kind::Tri(3), &H4T),
        [1, 1, 1],
        CAM,
        ExxDiv::Ewald,
        LOOSE,
    );
    for sm in [None, Some(KUSpinMutation::RoWUhfForm)] {
        let e = ro_force_m(&loose, None, sm)
            .expect_err("loose state accepted")
            .to_string();
        assert!(e.contains("ROHF orbital gradient"), "{sm:?}: {e}");
    }
    let off = ro_force_m(&loose, None, Some(KUSpinMutation::RoGateOff)).unwrap();
    let uhf = ro_force_m(&loose, None, Some(KUSpinMutation::RoWUhfFormGateOff)).unwrap();
    let g = off.hf.commutator;
    let (err_uhf, err_ro) = (diff(&uhf.grad, &gt.grad), diff(&off.grad, &gt.grad));
    let w_gap = diff(&uhf.grad, &off.grad);
    eprintln!(
        "loose CAM-B3LYP ROKS H4: g = {g:.2e}; gate-off error W_UKS {err_uhf:.2e}, W_RO \
         {err_ro:.2e}; W_UKS - W_RO {w_gap:.2e}"
    );
    // Off stationarity NEITHER form is right (both gate-off forces miss the
    // tight one); the gate's refusal is the protection. What distinguishes
    // the two W forms there is their gap.
    assert!(g > 1e-5, "state is not past the gate: {g:e}");
    assert!(
        err_ro > 10.0 * FD_BAR,
        "the loose state is not off stationarity: {err_ro:e}"
    );
    assert!(
        w_gap > 1e-6,
        "the two W forms coincide off stationarity: {w_gap:e}"
    );
}

// ----------------------------------------------------------- refusals

/// Typed refusals of the ROKS RSH force: bare kernels with a range-separated
/// name, kernels of another functional, a global-hybrid name on
/// range-separated kernels, RS-GDF J/K carrying the functional's range
/// separation (open-shell), a missing exxdiv; the ROKS stress with a
/// range-separated name; and the Python / CLI derivative gate (unchanged).
#[test]
fn roks_rsh_force_refusals() {
    use ferric_pbc::kuxcstress::{kpoint_roks_stress, KUksStressConfig};
    let fx = ro_fixture(Kind::Tri(2), &H3D, [1, 1, 1], "HSE06", ExxDiv::Ewald);
    let cfg = gcfg(ExxDiv::Ewald, None, None);
    let build = |rsh| {
        ferric_pbc::kdense_aft::KDenseAftEri::build(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &fx.hk.s,
            ExxDiv::None,
            &dense_cfg(rsh),
        )
        .unwrap()
    };
    let call = |jk: KGradJk<'_>, f: &str, c: &ferric_pbc::kuxcgrad::KUksGradConfig| {
        ferric_pbc::kuxcgrad::kpoint_roks_gradient(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &ferric_pbc::hcore::PeriodicHcoreConfig::with_omega(OMEGA),
            &fx.hk,
            jk,
            &fx.res.scf,
            f,
            c,
        )
        .unwrap_err()
        .to_string()
    };
    let e = call(KGradJk::Dense(&build(None)), "HSE06", &cfg);
    assert!(e.contains("bare Coulomb exchange"), "{e}");
    let e = call(KGradJk::Dense(&build(Some(rsh_of(CAM)))), "HSE06", &cfg);
    assert!(e.contains("functional's is"), "{e}");
    let e = call(KGradJk::Dense(&fx.eri), "PBE0", &cfg);
    assert!(e.contains("does not differentiate it"), "{e}");
    let mut no_exx = cfg.clone();
    no_exx.base.exxdiv = None;
    let e = call(KGradJk::Dense(&fx.eri), "HSE06", &no_exx);
    assert!(e.contains("needs KRksGradConfig::exxdiv"), "{e}");
    let aux = prep_for(
        &fx.cell,
        &ferric_core::basis::bundled("cc-pvdz-ri").unwrap(),
    );
    let kc = ferric_pbc::rsgdf::kpoint::KRsGdfConfig {
        gdf: ferric_pbc::rsgdf::RsGdfConfig {
            omega: OMEGA,
            exxdiv: ExxDiv::Ewald,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    };
    let gdf_rsh = ferric_pbc::rsgdf::kpoint::KRsGdf::build_rsh(
        &fx.cell,
        &fx.prep,
        &aux,
        &fx.mesh,
        &fx.hk.s,
        &kc,
        rsh_of("HSE06"),
    )
    .unwrap();
    let src = ferric_pbc::kgrad::KRsGdfGradSource {
        gdf: &gdf_rsh,
        cfg: &kc,
        aux: &aux,
    };
    // Differentiated since the open-shell RS-GDF force landed
    // (`pbc_kuks_rsh_rsgdf_force.rs`): this DENSE SCF's spin Focks are not
    // that J/K's, so the Fock guard refuses it, and the ROKS stress refuses
    // the range-separated source.
    let e = call(KGradJk::RsGdf(src), "HSE06", &cfg);
    assert!(e.contains("Fock matrix rebuilt"), "{e}");
    let e = kpoint_roks_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &ferric_pbc::hcore::PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        ferric_pbc::kstress::KStressJk::RsGdf(src),
        &fx.res.scf,
        "HSE06",
        &KUksStressConfig::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("range-separated"), "{e}");
    let e = kpoint_roks_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &ferric_pbc::hcore::PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        ferric_pbc::kstress::KStressJk::Dense(&fx.eri),
        &fx.res.scf,
        "HSE06",
        &KUksStressConfig::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("range-separated"), "{e}");
    // The binding derivative gate keeps refusing ROKS range separation.
    use ferric_pbc::kxcgrad::{check_deriv_functional, Deriv, DerivSpin};
    for f in FUNCTIONALS {
        for d in [Deriv::Force, Deriv::Stress] {
            assert!(
                check_deriv_functional(DerivSpin::RestrictedOpen, d, false, f).is_err(),
                "{f} {d:?}"
            );
        }
    }
}
