//! Analytic forces of the OPEN-SHELL k-point UKS energy for RANGE-SEPARATED
//! hybrids on the dense-AFT J/K (`ferric_pbc::kuxcgrad`, module doc
//! "Range-separated hybrids (UKS)"), and the Gamma-point UKS force routed
//! through the 1x1x1 mesh (`ferric_pbc::gamma_rsh::gamma_uks_rsh_gradient`).
//!
//! The derivative is the closed-shell RSH one (#486) with the UHF spin
//! bookkeeping: exchange terms `(1, D_alpha), (1, D_beta)` weighted by
//! `f(K) = c_sr + (c_lr - c_sr) e^{-K^2/4w^2}`, `J[D_alpha + D_beta]` bare, and
//! the M-term `-v_M^mix sum_s D_s S D_s`. Oracles (artifact hypothesis in
//! each test's doc):
//!
//! 1. singlet k-UKS == k-RKS RSH force (separate SCFs; blind to every
//!    alpha/beta mix-up, which is why the rest run on POLARIZED cells);
//! 2. central FD of ferric's own polarized k-UKS energy (fresh SCF per
//!    displacement): two CAM-B3LYP components of the H3 doublet here; every
//!    component of the H3 doublet and the H4 triplet for both functionals,
//!    the h sweep and ewald vs none in `validation_kuks_rsh_force.rs`;
//! 3. `c_sr == c_lr == c` reproduces the global-hybrid UKS force with fraction
//!    `c` (zero XC, two SCFs, polarized H3; ewald here, none weekly). Both
//!    sides share the M-term density sandwich (`Σ_σ D_σ S D_σ`), so this
//!    anchor cannot see an M-term SPIN error; only FD does (a reviewer's
//!    cross-spin M-term mutant missed FD by 2.3e-2);
//! 4. the 1x1x2 mesh force equals the force on one copy in the explicit 2-cell
//!    supercell (weekly file);
//! 5. Gamma (`gamma_uks_rsh` + `gamma_uks_rsh_gradient`) equals the 1x1x1 mesh
//!    (CAM-B3LYP / ewald here, all four cases weekly).
//!
//! MUTATION LEDGER, measured 2026-10-19 (release test profile), H3 doublet
//! (triclinic `TRI_A`, 1x1x2, exxdiv ewald, POLARIZED) and H2 singlet (cube,
//! 1x1x3); miss = max |F_mutant - F_ref| in Ha/Bohr against the production
//! force, which agrees with FD of the SCF energy to <= 4.7e-9 on every
//! component (weekly file). The spin-only defects are invisible on the
//! singlet by construction:
//!
//! | mutant                                   | H3 CAM-B3LYP | H3 HSE06 | H2 singlet CAM / HSE06 |
//! |------------------------------------------|--------------|----------|------------------------|
//! | ExchBetaUsesAlpha (D_a in the beta K walk) | 2.55e-3    | 2.80e-3  | 0 / 2.8e-17            |
//! | ExchDropBeta (beta exchange dropped)     | 1.96e-2      | 1.63e-2  | 4.06e-2 / 2.73e-2      |
//! | MTermTotalDensity (1/2 D S D, total D)   | 1.17e-2      | 3.86e-3  | 0 / 0                  |
//! | PulayKSwap (alpha K in the beta Fock)    | 2.66e-2      | 1.09e-2  | 0 / 0                  |
//! | RshKernelOnJ                             | 1.29e-1      | 1.20e-1  | 1.01e-1 / 9.03e-2      |
//! | RshPlainMadelung                         | 4.34e-2      | 7.22e-2  | 3.40e-2 / 5.27e-2      |
//! | RshSwapFractions (c_sr <-> c_lr)         | 3.58e-2      | 3.34e-2  | 3.00e-2 / 5.44e-2      |
//! | RshNoKernelFactor                        | 9.61e-2      | 1.00e-1  | 1.48e-1 / 1.64e-1      |
//! | RshNoErfPiece                            | 1.31e-2      | 2.66e-6  | 3.77e-2 / 1.22e-4      |
//! | RshOmegaScaled (w x 1.01)                | 2.68e-4      | 5.07e-7  | 5.20e-4 / 1.45e-5      |
//!
//! The erf piece is live for CAM-B3LYP (largest `|c_lr - c_sr|
//! e^{-K^2/4w^2}` on the H3 1x1x2 K set 0.172) and nearly dead for HSE06 on
//! this cell (3.6e-5): HSE06's erf-piece and w defects are resolved only at
//! 27 and 5 FD bars there, so they are asserted on CAM-B3LYP (HSE06's erf
//! piece is resolved on the H2 singlet, 1.2e-4). Bars asserted (weekly
//! `full_mutation_ledger`): every polarized miss > 1e-5 (100 FD bars) except
//! those two HSE06 rows; per commit the spin defects and the erf piece on
//! CAM-B3LYP.
//!
//! NOT covered: RS-GDF (refused), ROKS / ROHF (refused), the stress
//! (refused), wB97X (accepted, not force-tested), p/d shells (H/STO-3G only),
//! shifted meshes, a PySCF force pin, Python / CLI (the shared open-shell
//! derivative gate still refuses range-separated hybrids).

mod common;
#[path = "kuks_rsh_force_support/mod.rs"]
mod support;

use common::*;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::kgrad::KGradJk;
use ferric_pbc::kpts::KPointMesh;
use support::*;

// ------------------------------------------------------- (1) singlet limit

/// (1) SINGLET (CAM-B3LYP; HSE06 weekly): on the H2 singlet the k-UKS RSH
/// force (UKS SCF, UKS entry) equals the k-RKS RSH force (#486; independent
/// closed-shell SCF and entry), per part. Physics: the SCF floor; artifact: a
/// per-spin factor slip (a ½ on the spin exchange terms or the M-term, a
/// scale other than 1) shows as O(1e-2) (ExchDropBeta: 4.1e-2). The pure
/// spin-assignment defects are INVISIBLE here (`D_a = D_b`), asserted, which
/// is why the remaining oracles run on polarized cells.
#[test]
fn singlet_uks_is_the_rks_rsh_force() {
    let (rows, fx, u) = singlet_rows(CAM);
    for (n, d) in rows {
        eprintln!("CAM-B3LYP singlet UKS vs RKS {n}: {d:.2e}");
        assert!(d < ANCHOR_TOL, "{n}: {d:e}");
    }
    for (n, m, sm) in ledger_mutants() {
        if SPIN_ONLY.contains(&n) || n == "ExchDropBeta" {
            let miss = diff(&force_m(&fx, m, sm).unwrap().grad, &u.grad);
            eprintln!("CAM-B3LYP singlet: {n} miss {miss:.2e}");
            if n == "ExchDropBeta" {
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

// -------------------------------------------------- (2) FD, per-commit row

/// (2) Central FD (h = 1e-4) of the polarized CAM-B3LYP k-UKS energy, H3
/// doublet, 1x1x2, exxdiv ewald, two components. Physics: the h^2 floor
/// (3.8e-9 / 2.1e-10 measured); artifact: every ledger mutant misses by
/// >= 2.7e-4 (CAM-B3LYP ledger).
#[test]
fn cam_b3lyp_polarized_force_matches_fd() {
    let fx = fixture(Kind::Tri(2), &H3D, [1, 1, 2], CAM, ExxDiv::Ewald);
    assert!(spin_asymmetry(&fx.res.scf) > 0.1, "polarized");
    let g = force(&fx);
    for (a, x) in [(0, 2), (2, 0)] {
        let d = fd(
            Kind::Tri(2),
            &H3D,
            a,
            x,
            1e-4,
            [1, 1, 2],
            CAM,
            ExxDiv::Ewald,
        );
        let r = (g.grad[(a, x)] - d).abs();
        eprintln!(
            "CAM-B3LYP H3 ({a},{x}): analytic {:+.10e} FD {d:+.10e} |diff| {r:.2e}",
            g.grad[(a, x)]
        );
        assert!(r < FD_BAR, "({a},{x}): {r:e}");
    }
    assert!(max_abs(&g.hf.parts.exchange) > 1e-4, "K walks exercised");
}

// ----------------------------------------------- (3) equal-fraction anchor

/// (3) EQUAL FRACTIONS on the polarized H3 doublet: `c_sr = c_lr = c` makes
/// the attenuated kernel `c 4pi/K^2` and the mixed Madelung `c v_M`, i.e. the
/// global-hybrid UKS with fraction `c`. Two SCFs (RSH kernels at scale 1 vs
/// bare kernels at scale c, zero XC) and two force paths (kernel factor vs
/// exchange-coefficient scaling). Artifact: the factor on J or dropped, a
/// plain Madelung, the M-term on the total density, or the alpha density in
/// the beta exchange walk move it by >= 4e-3 (the Madelung ones are blind
/// under exxdiv none by construction).
#[test]
fn equal_fractions_is_the_global_hybrid_uks_force() {
    // exxdiv none (where the Madelung defects are blind) runs weekly.
    equal_fraction_case(ExxDiv::Ewald);
}

// ----------------------------------------------------------- (5) Gamma

/// (5) GAMMA == 1x1x1 on the polarized H3 doublet (CAM-B3LYP, ewald; the
/// other three cases weekly): the Gamma UKS RSH force (Gamma SCF, J, hcore
/// and grid, routed into the 1x1x1 UKS force) against the 1x1x1 k-UKS SCF +
/// force. Physics: the SCF floor; artifact: a glue slip (spin densities or
/// Focks swapped, a factor 2 on a spin density, the wrong Madelung) shows as
/// O(1e-3) or fails the routed Fock guard. A glue check (the kernel is
/// shared). Also the routed refusals: another grid (Fock guard), a
/// restricted result.
#[test]
fn gamma_uks_rsh_force_is_the_one_point_mesh_force() {
    let g = gamma_fixture(CAM, ExxDiv::Ewald);
    let mut other = g.gc.clone();
    other.grid = ferric_pbc::dft::PeriodicGridConfig::with_size(30, 110);
    other.grid.neighbour_cutoff = Some(CUTOFF);
    let e = ferric_pbc::gamma_rsh::gamma_uks_rsh_gradient(
        &g.cell,
        &g.prep,
        &g.hcfg,
        &g.scf,
        &other,
        &g.ex,
        &kgrad_cfg(),
    )
    .expect_err("another grid accepted")
    .to_string();
    assert!(e.contains("differs from the SCF's stored fock"), "{e}");
    let mut closed = g.scf.clone();
    closed.spin = ferric_scf::result::Spin::Restricted;
    let e = ferric_pbc::gamma_rsh::gamma_uks_rsh_gradient(
        &g.cell,
        &g.prep,
        &g.hcfg,
        &closed,
        &g.gc,
        &g.ex,
        &kgrad_cfg(),
    )
    .expect_err("restricted result accepted")
    .to_string();
    assert!(e.contains("unrestricted (UKS) results only"), "{e}");
    // An RS-GDF exchange is refused (the Gamma routes differentiate dense AFT only).
    let hc = ferric_pbc::hcore::periodic_hcore(&g.cell, &g.prep, &g.hcfg).unwrap();
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
    let e = ferric_pbc::gamma_rsh::gamma_uks_rsh_gradient(
        &g.cell,
        &g.prep,
        &g.hcfg,
        &g.scf,
        &g.gc,
        &ex_gdf,
        &kgrad_cfg(),
    )
    .expect_err("RS-GDF Gamma exchange accepted")
    .to_string();
    assert!(e.contains("RS-GDF") && e.contains("no derivative"), "{e}");
    let gf = gamma_force(&g).unwrap();
    let k = force(&fixture(Kind::Tri(2), &H3D, [1, 1, 1], CAM, ExxDiv::Ewald));
    let d = diff(&gf.grad, &k.grad);
    eprintln!(
        "CAM-B3LYP ewald: Gamma vs 1x1x1 UKS force {d:.2e} (Fock guard {:.1e})",
        gf.hf.fock_mismatch
    );
    assert!(d < ANCHOR_TOL, "{d:e}");
    assert!(max_abs(&gf.grad) > 1e-2);
}

// ----------------------------------------------------------- mutants

/// The spin and erf-piece defects on the POLARIZED H3 doublet (CAM-B3LYP,
/// 1x1x2, ewald; the full ledger for both functionals is weekly): each moves
/// the FD-validated force by > 100 FD bars, and the erf piece is live on this
/// K set (largest weight 0.172).
#[test]
fn polarized_spin_and_erf_mutants_are_caught() {
    let cell = cell_of(Kind::Tri(2), &H3D);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    assert!(max_erf_weight(&cell, &mesh, rsh_of(CAM)) > 0.1);
    let fx = fixture(Kind::Tri(2), &H3D, [1, 1, 2], CAM, ExxDiv::Ewald);
    let base = force(&fx);
    for (n, m, sm) in ledger_mutants() {
        if SPIN_ONLY.contains(&n) || n == "ExchDropBeta" || n == "RshNoErfPiece" {
            let miss = diff(&force_m(&fx, m, sm).unwrap().grad, &base.grad);
            eprintln!("CAM-B3LYP H3 1x1x2 ewald: {n} miss {miss:.2e}");
            assert!(miss > 100.0 * FD_BAR, "{n}: {miss:e}");
        }
    }
}

// ----------------------------------------------------------- refusals

/// Typed refusals: bare kernels with a range-separated name, kernels of
/// another functional, a global-hybrid name on range-separated kernels,
/// RS-GDF J/K with a range-separated name (UKS force), the ROKS force and the
/// open-shell stress with a range-separated name, and the Python / CLI
/// open-shell derivative gate (unchanged).
#[test]
fn oshell_rsh_force_refusals() {
    use ferric_pbc::kuxcstress::{kpoint_uks_stress, KUksStressConfig};
    let fx = fixture(Kind::Tri(2), &H3D, [1, 1, 1], "HSE06", ExxDiv::Ewald);
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
    let call = |jk: KGradJk<'_>, f: &str| try_force_on(&fx, jk, f, &cfg).unwrap_err().to_string();
    let e = call(KGradJk::Dense(&build(None)), "HSE06");
    assert!(e.contains("bare Coulomb exchange"), "{e}");
    let e = call(KGradJk::Dense(&build(Some(rsh_of(CAM)))), "HSE06");
    assert!(e.contains("functional's is"), "{e}");
    let e = call(KGradJk::Dense(&fx.eri), "PBE0");
    assert!(e.contains("does not differentiate it"), "{e}");
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
    let gdf =
        ferric_pbc::rsgdf::kpoint::KRsGdf::build(&fx.cell, &fx.prep, &aux, &fx.mesh, &fx.hk.s, &kc)
            .unwrap();
    let src = ferric_pbc::kgrad::KRsGdfGradSource {
        gdf: &gdf,
        cfg: &kc,
        aux: &aux,
    };
    let e = call(KGradJk::RsGdf(src), "HSE06");
    assert!(e.contains("RS-GDF") && e.contains("no derivative"), "{e}");
    // An RS-GDF source that carries the functional's own range separation
    // (the closed-shell RKS force differentiates it) is refused for UKS too.
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
    let e = call(KGradJk::RsGdf(src), "HSE06");
    assert!(
        e.contains("RS-GDF") && e.contains("open-shell state"),
        "{e}"
    );
    let e = kpoint_uks_stress(
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
    let e = ferric_pbc::kuxcgrad::kpoint_roks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &ferric_pbc::hcore::PeriodicHcoreConfig::with_omega(OMEGA),
        &fx.hk,
        KGradJk::Dense(&fx.eri),
        &roks_shaped(&fx),
        "HSE06",
        &cfg,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("ROKS / ROHF force"), "{e}");
    for f in FUNCTIONALS {
        assert!(ferric_pbc::kxcgrad::check_open_shell_force_functional(f).is_err());
    }
}

/// The UKS state in the ROKS result shape. Only the refusal is exercised: the
/// ROKS force checks the functional before it reads any density, so this
/// relabelled state is never differentiated.
fn roks_shaped(fx: &Fx) -> ferric_pbc::kroks::KRoScfResult {
    let s = &fx.res.scf;
    ferric_pbc::kroks::KRoScfResult {
        energy: s.energy,
        e_nuc: s.e_nuc,
        e_xc: 0.0,
        nalpha: s.nalpha,
        nbeta: s.nbeta,
        eps: s.eps_alpha.clone(),
        eps_alpha: s.eps_alpha.clone(),
        eps_beta: s.eps_beta.clone(),
        mos: s.mos_alpha.clone(),
        density_alpha: s.density_alpha.clone(),
        density_beta: s.density_beta.clone(),
        fock_alpha: s.fock_alpha.clone(),
        fock_beta: s.fock_beta.clone(),
        fock_eff: s.fock_alpha.clone(),
        occupations: s.occ_alpha.clone(),
        nclosed_per_k: s.nocc_per_k_beta.clone(),
        nopen_per_k: vec![1; s.kpts.len()],
        gap_alpha: None,
        gap_beta: None,
        s2: s.s2,
        converged: s.converged,
        iterations: s.iterations,
        max_error: 0.0,
        kpts: s.kpts.clone(),
        lindep: Default::default(),
    }
}
