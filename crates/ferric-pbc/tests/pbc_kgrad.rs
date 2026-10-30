//! k-point RHF / UHF analytic forces (`ferric_pbc::kgrad`), the Rust port of
//! `reference/pbc/pbc_kgrad.py` + `pbc_kgrad_gdf.py` (FINDINGS "Iteration 21
//! (Python, k-point RHF/UHF forces)").
//!
//! What is anchored against what (every anchor is taken at ONE fixed
//! density, so SCF convergence does not enter the comparison):
//! (1) 1×1×1 mesh ≡ the Gamma force (`gamma_rhf_gradient_with` /
//!     `gamma_uhf_gradient_with`, and the RS-GDF twins) at the same density,
//!     both exxdiv, ≤ 1e-12 (prototype 4.7e-16..2.8e-15; RS-GDF 1.7e-13).
//!     An independent construction: real half-sphere Gamma code vs complex
//!     full-sphere residue code, image-summed vs phase-folded weights.
//! (2) SUPERCELL ANCHOR: the k-mesh force on atom A ≡ the Gamma force of the
//!     explicit diag(N) supercell on EVERY copy of A, at the k density
//!     unfolded to the supercell (`D_sc[(c,m),(c',n)] = (1/N_k) Σ_k
//!     e^{ik·(T_c − T_c')} D(k)_mn`). Prototype ≤ 1.6e-13 dense, ≤ 6e-12
//!     RS-GDF. NOT the N× sum over copies (printed: it is (N−1)× off). The
//!     Rust `h(k)` (SR erfc V_ne on Gaussian nuclei + LR + c0 Z S(k)) was NOT
//!     prototyped at k — this anchor is what pins its derivative, since the
//!     supercell side runs the Gamma `h` derivative of `crate::grad`.
//!     Meshes: H2 1×1×3 and 2×2×2 RHF, H3 1×1×3 UHF (2,1), triclinic s+p
//!     tri2 1×1×3 RHF (`#[ignore]`d, 135 s release). tri2 runs at the default
//!     gradient nucleus exponent like the s-only cells and the same `SC_BAR`:
//!     measured miss 1.3e-14 at the default.
//! (3) Analytic vs central FD (h = 1e-4) of ferric's OWN k energy, H2 1×1×3,
//!     both exxdiv (dense and RS-GDF). The prototype's floor is the h²
//!     truncation (−2.99e-9 on (0,z), Richardson 3e-12) with a pure-AFT h;
//!     the Rust Ewald-split h adds the smeared-nucleus derivative (the
//!     default `GRAD_NUCLEUS_EXPONENT` is the energy's 1e16), so the bar is
//!     the Rust Gamma FD bar 1e-7 (`pbc_grad.rs`), residuals printed.
//! (4) ΣF ≤ 1e-10; F(ewald) ≡ F(none) on one density ≤ 1e-11; closed-shell
//!     kUHF(D/2, D/2) ≡ kRHF(D) ≤ 1e-12.
//! (5) Mutations (`KGradMutation`), miss vs the supercell anchor on 1×1×3
//!     (> 1e-4; prototype H2 1×1×3: NoPhase 9.0e-2, ConjD 7.4e-3 / 4.7e-4
//!     (none / ewald), WrongQ 9.1e-3, GammaMadelung 1.9e-1 (ewald only),
//!     NoInvNk 7.8e-1; RS-GDF FitAuxPhase 7.7e-4, FitNoMetric 4.6e-2,
//!     FitNoG0 2.8e-2). BLIND SPOT, asserted: on the TRIM-only 2×2×2 mesh
//!     `D(k)` is real and `q ≡ −q`, so ConjD and WrongQ are IDENTITIES
//!     (|ΔF| ≤ 1e-12) — a port validated only on n_i ≤ 2 meshes has not
//!     tested its phases. GammaMadelung is blind under exxdiv = none
//!     (asserted).
//!
//! Note: on H2 at TRIM meshes the SCF converges in one iteration (inversion
//! symmetry fixes the one occupied band per TRIM k); those rows test the
//! force formula, not SCF response. H2 1×1×3, H3 and tri2 carry the SCF.
//!
//! The tests whose per-commit CI time was >= 120 s (`h2_2x2x2_force_is_the_supe
//! rcell_force_and_trim_mesh_is_blind_to_phase_mutants`,
//! `h3_uhf_1x1x3_force_is_the_supercell_force_on_one_copy`) run weekly from
//! `validation_pbc_kgrad.rs`; the shared helpers are
//! `pbc_kgrad_support/mod.rs`.

mod common;
#[path = "pbc_kgrad_support/mod.rs"]
mod support;

use support::*;

// ======================================================= (1) 1×1×1 ≡ Gamma

#[test]
fn one_point_mesh_rhf_is_the_gamma_force() {
    for (name, cell, bs) in [
        ("H2", h2_cell_g(), pyscf_sto3g_h()),
        ("tri2 s+p", tri2_cell(), sp_basis_h()),
    ] {
        let ks = ksys(cell.clone(), &bs, [1, 1, 1]);
        let eri = kdense(&ks, ANCHOR_PRECISION);
        let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
        let g = gsys(cell, &bs, Some(ANCHOR_PRECISION), None);
        let d = real_of(&scf.densities[0]);
        let gscf = gamma_restricted(d);
        for exx in EXX {
            let fk = kg_rhf(&ks, KGradJk::Dense(&eri), &scf, exx, None);
            let fg = gamma_grad(&g, &gscf, exx);
            let dd = max_diff(&fk.grad, &fg);
            report(&format!("{name} 1x1x1 {exx:?}"), &fk);
            eprintln!("  {name} 1x1x1 {exx:?}: |F_k − F_gamma| {dd:.2e} (prototype 4.7e-16)");
            assert!(dd <= GAMMA_BAR, "{name} {exx:?}: {dd:e}");
            assert!(
                fk.net_force <= NET_FORCE_BAR,
                "{name}: ΣF {:e}",
                fk.net_force
            );
        }
    }
}

#[test]
fn one_point_mesh_uhf_is_the_gamma_force() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h3_cell(), &bs, [1, 1, 1]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), 2, 1);
    let g = gsys(h3_cell(), &bs, Some(ANCHOR_PRECISION), None);
    let gscf = gamma_unrestricted(
        real_of(&scf.density_alpha[0]),
        real_of(&scf.density_beta[0]),
    );
    for exx in EXX {
        let fk = kg_uhf(&ks, KGradJk::Dense(&eri), &scf, exx, None);
        let fg = gamma_grad(&g, &gscf, exx);
        let dd = max_diff(&fk.grad, &fg);
        report(&format!("H3 UHF 1x1x1 {exx:?}"), &fk);
        eprintln!("  H3 UHF 1x1x1 {exx:?}: |F_k − F_gamma| {dd:.2e} (prototype 2.8e-15)");
        assert!(dd <= GAMMA_BAR, "{exx:?}: {dd:e}");
    }
}

#[test]
fn one_point_mesh_rsgdf_is_the_gamma_rsgdf_force() {
    let bs = pyscf_sto3g_h();
    let aux_bs = et_sp_aux();
    // RHF H2 and UHF H3.
    for (name, cell, uhf) in [("H2 RHF", h2_cell_g(), false), ("H3 UHF", h3_cell(), true)] {
        let ks = ksys(cell.clone(), &bs, [1, 1, 1]);
        let auxp = PreparedBasis::new(cell.mol(), &aux_bs).unwrap();
        let gdf = kgdf(&ks, &auxp);
        let cfg = kgdf_cfg();
        let src = KRsGdfGradSource {
            gdf: &gdf,
            cfg: &cfg,
            aux: &auxp,
        };
        let g = gsys(cell, &bs, None, Some(&aux_bs));
        for exx in EXX {
            let (fk, fg) = if uhf {
                let scf = kuhf(&ks, Box::new(gdf.jk_builder_with_madelung(0.0)), 2, 1);
                let gscf = gamma_unrestricted(
                    real_of(&scf.density_alpha[0]),
                    real_of(&scf.density_beta[0]),
                );
                (
                    kg_uhf(&ks, KGradJk::RsGdf(src), &scf, exx, None),
                    gamma_grad(&g, &gscf, exx),
                )
            } else {
                let scf = krhf(&ks, Box::new(gdf.jk_builder_with_madelung(0.0)));
                let gscf = gamma_restricted(real_of(&scf.densities[0]));
                (
                    kg_rhf(&ks, KGradJk::RsGdf(src), &scf, exx, None),
                    gamma_grad(&g, &gscf, exx),
                )
            };
            let dd = max_diff(&fk.grad, &fg);
            report(&format!("{name} RS-GDF 1x1x1 {exx:?}"), &fk);
            eprintln!(
                "  {name} RS-GDF 1x1x1 {exx:?}: |F_k − F_gamma| {dd:.2e} (prototype 1.7e-13..2.1e-13)"
            );
            assert!(dd <= GAMMA_BAR_GDF, "{name} {exx:?}: {dd:e}");
        }
    }
}

#[test]
fn h2_1x1x3_force_is_the_supercell_force_on_one_copy() {
    let (ks, eri, scf, refs) =
        rhf_supercell_anchor("H2", h2_cell_g(), &pyscf_sto3g_h(), [1, 1, 3], None, SC_BAR);
    assert_rhf_mutants_fail("H2 1x1x3", &ks, &eri, &scf, &refs, None, SC_BAR);
}

/// Per-commit canary of `validation_pbc_kgrad.rs`'s 2x2x2 supercell test:
/// its BLIND-SPOT half on the cheapest TRIM mesh, 1x1x2. Every k is a TRIM, so
/// ConjD and WrongQ are identities (|ΔF| <= 1e-12, both exxdiv), while NoPhase
/// and NoInvNk still move the force by > MUTANT_BAR. The phases themselves are
/// pinned on the non-TRIM 1x1x3 mesh by
/// `h2_1x1x3_force_is_the_supercell_force_on_one_copy`.
#[test]
fn h2_1x1x2_trim_mesh_is_blind_to_phase_mutants() {
    let ks = ksys(h2_cell_g(), &pyscf_sto3g_h(), [1, 1, 2]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    for exx in EXX {
        let f0 = kg_rhf(&ks, KGradJk::Dense(&eri), &scf, exx, None);
        for m in [KGradMutation::ConjD, KGradMutation::WrongQ] {
            let f = kg_rhf(&ks, KGradJk::Dense(&eri), &scf, exx, Some(m));
            let d = max_diff(&f.grad, &f0.grad);
            eprintln!("  H2 1x1x2 {exx:?} {m:?}: |F_mut − F| {d:.2e} (identity on TRIM meshes)");
            assert!(
                d <= IDENTITY_BAR,
                "{m:?} should be an identity on 1x1x2 ({d:e})"
            );
        }
    }
    let f0 = kg_rhf(&ks, KGradJk::Dense(&eri), &scf, ExxDiv::None, None);
    for m in [KGradMutation::NoPhase, KGradMutation::NoInvNk] {
        let f = kg_rhf(&ks, KGradJk::Dense(&eri), &scf, ExxDiv::None, Some(m));
        let d = max_diff(&f.grad, &f0.grad);
        eprintln!("  H2 1x1x2 {m:?}: |F_mut − F| {d:.2e}");
        assert!(d > MUTANT_BAR, "{m:?} not caught on 1x1x2 ({d:e})");
    }
}

#[test]
#[ignore = "slow (unmeasured): 6-atom s+p supercell SR sum at 1e-14; run with --ignored"]
fn triclinic_sp_1x1x3_force_is_the_supercell_force_on_one_copy() {
    let nuc = None;
    let (ks, eri, scf, refs) = rhf_supercell_anchor(
        "tri2 s+p",
        tri2_cell(),
        &sp_basis_h(),
        [1, 1, 3],
        nuc,
        SC_BAR,
    );
    assert_rhf_mutants_fail("tri2 1x1x3", &ks, &eri, &scf, &refs, nuc, SC_BAR);
}

#[test]
fn closed_shell_kuhf_force_is_the_krhf_force() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let r = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let mut u = kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), 1, 1);
    let half: Vec<Array2<Complex64>> = r.densities.iter().map(|d| d.mapv(|z| z * 0.5)).collect();
    u.density_alpha = half.clone();
    u.density_beta = half;
    for exx in EXX {
        let fr = kg_rhf(&ks, KGradJk::Dense(&eri), &r, exx, None);
        let fu = kg_uhf(&ks, KGradJk::Dense(&eri), &u, exx, None);
        let d = max_diff(&fr.grad, &fu.grad);
        eprintln!("  H2 1x1x3 {exx:?}: |F_kUHF(D/2,D/2) − F_kRHF(D)| {d:.2e}");
        assert!(d <= IDENTITY_BAR, "{exx:?}: {d:e}");
    }
}

#[test]
fn h2_1x1x3_dense_force_matches_fd_of_own_k_energy() {
    let ks = ksys(h2_cell_g(), &pyscf_sto3g_h(), [1, 1, 3]);
    let eri = kdense(&ks, DEFAULT_DENSE_AFT_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scf = [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))),
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))),
    ];
    let ident = scf[0].energy - scf[1].energy - vm;
    eprintln!("H2 1x1x3: E(none) − E(ewald) − nocc v_M = {ident:.1e}");
    assert!(ident.abs() < 1e-10, "{ident:e}");
    let fd = fd_rows(&H2_COMPS, &H2_ATOMS, h2_dense_energies);
    for (i, exx) in EXX.into_iter().enumerate() {
        let g = kg_rhf(&ks, KGradJk::Dense(&eri), &scf[i], exx, None);
        report(&format!("H2 1x1x3 dense {exx:?}"), &g);
        let mut worst = 0.0_f64;
        for ((a, x), v) in &fd {
            let d = g.grad[(*a, *x)] - v[i];
            eprintln!(
                "  ({a},{x}) {exx:?}: analytic {:+.10e} FD {:+.10e} diff {d:+.2e} (prototype h² floor −3e-9 on (0,z))",
                g.grad[(*a, *x)],
                v[i]
            );
            worst = worst.max(d.abs());
        }
        assert!(worst <= FD_BAR, "{exx:?}: FD miss {worst:e}");
        assert!(g.net_force <= NET_FORCE_BAR, "ΣF {:e}", g.net_force);
    }
}

#[test]
fn h2_1x1x3_rsgdf_force_matches_fd_of_own_k_energy() {
    let cell = h2_cell_g();
    let ks = ksys(cell.clone(), &pyscf_sto3g_h(), [1, 1, 3]);
    let auxp = PreparedBasis::new(cell.mol(), &et_sp_aux()).unwrap();
    let gdf = kgdf(&ks, &auxp);
    let cfg = kgdf_cfg();
    let src = KRsGdfGradSource {
        gdf: &gdf,
        cfg: &cfg,
        aux: &auxp,
    };
    let scf = krhf(&ks, Box::new(gdf.jk_builder_with_madelung(0.0)));
    let comps = [(0, 2), (1, 1)];
    let fd = fd_rows(&comps, &H2_ATOMS, h2_gdf_energies);
    for (i, exx) in EXX.into_iter().enumerate() {
        let g = kg_rhf(&ks, KGradJk::RsGdf(src), &scf, exx, None);
        report(&format!("H2 1x1x3 RS-GDF {exx:?}"), &g);
        let mut worst = 0.0_f64;
        for ((a, x), v) in &fd {
            let d = g.grad[(*a, *x)] - v[i];
            eprintln!(
                "  ({a},{x}) {exx:?}: analytic {:+.10e} FD {:+.10e} diff {d:+.2e} (prototype −2.99e-9 / +9.5e-11)",
                g.grad[(*a, *x)],
                v[i]
            );
            worst = worst.max(d.abs());
        }
        assert!(worst <= FD_BAR, "{exx:?}: FD miss {worst:e}");
    }
}

// ================================================ RS-GDF supercell anchors

#[test]
fn h2_1x1x3_rsgdf_force_is_the_supercell_rsgdf_force_and_catches_fit_mutants() {
    let n = [1, 1, 3];
    let bs = pyscf_sto3g_h();
    let aux_bs = et_sp_aux();
    let cell = h2_cell_g();
    let ks = ksys(cell.clone(), &bs, n);
    let auxp = PreparedBasis::new(cell.mol(), &aux_bs).unwrap();
    let gdf = kgdf(&ks, &auxp);
    let cfg = kgdf_cfg();
    let src = KRsGdfGradSource {
        gdf: &gdf,
        cfg: &cfg,
        aux: &auxp,
    };
    let scf = krhf(&ks, Box::new(gdf.jk_builder_with_madelung(0.0)));
    let g = gsys(supercell(&cell, n, 1), &bs, None, Some(&aux_bs));
    let gscf = gamma_restricted(unfold(&ks.mesh, &scf.densities));
    let mut refs: Vec<Array2<f64>> = Vec::new();
    let mut fks: Vec<Array2<f64>> = Vec::new();
    for exx in EXX {
        let fk = kg_rhf(&ks, KGradJk::RsGdf(src), &scf, exx, None);
        let fsc = gamma_grad(&g, &gscf, exx);
        let miss = sc_miss(&fk.grad, &fsc);
        report(&format!("H2 RS-GDF {n:?} {exx:?}"), &fk);
        eprintln!(
            "  H2 RS-GDF {n:?} {exx:?}: supercell miss {miss:.2e} (prototype 3.5e-12 / 1.6e-12)"
        );
        assert!(miss <= SC_BAR_GDF, "{exx:?}: {miss:e}");
        assert!(fk.net_force <= NET_FORCE_BAR, "ΣF {:e}", fk.net_force);
        fks.push(fk.grad);
        refs.push(fsc);
    }
    let de = max_diff(&fks[0], &fks[1]);
    eprintln!("  H2 RS-GDF: |F(ewald) − F(none)| {de:.2e} (prototype 5.5e-13)");
    assert!(de <= EWALD_NONE_BAR, "ewald vs none {de:e}");
    // WrongQ acts on the dense exchange derivative only (an identity here);
    // ConjD is unmeasured on the fitted path, so it is not asserted.
    let mesh_on_fit = [KGradMutation::NoPhase, KGradMutation::NoInvNk];
    for m in FIT_MUTANTS.into_iter().chain(mesh_on_fit) {
        let f = kg_rhf(&ks, KGradJk::RsGdf(src), &scf, ExxDiv::None, Some(m));
        let miss = sc_miss(&f.grad, &refs[0]);
        eprintln!("  H2 RS-GDF mutant {m:?}: miss {miss:.2e}");
        assert!(miss > MUTANT_BAR, "mutant {m:?} not caught ({miss:e})");
    }
}

#[test]
fn h3_uhf_1x1x3_rsgdf_force_is_the_supercell_rsgdf_force() {
    let n = [1, 1, 3];
    let bs = pyscf_sto3g_h();
    let aux_bs = et_sp_aux();
    let cell = h3_cell();
    let ks = ksys(cell.clone(), &bs, n);
    let auxp = PreparedBasis::new(cell.mol(), &aux_bs).unwrap();
    let gdf = kgdf(&ks, &auxp);
    let cfg = kgdf_cfg();
    let src = KRsGdfGradSource {
        gdf: &gdf,
        cfg: &cfg,
        aux: &auxp,
    };
    let scf = kuhf(&ks, Box::new(gdf.jk_builder_with_madelung(0.0)), 2, 1);
    let g = gsys(supercell(&cell, n, 4), &bs, None, Some(&aux_bs));
    let gscf = gamma_unrestricted(
        unfold(&ks.mesh, &scf.density_alpha),
        unfold(&ks.mesh, &scf.density_beta),
    );
    let mut refs: Vec<Array2<f64>> = Vec::new();
    for exx in EXX {
        let fk = kg_uhf(&ks, KGradJk::RsGdf(src), &scf, exx, None);
        let fsc = gamma_grad(&g, &gscf, exx);
        let miss = sc_miss(&fk.grad, &fsc);
        report(&format!("H3 UHF RS-GDF {n:?} {exx:?}"), &fk);
        eprintln!("  H3 UHF RS-GDF {n:?} {exx:?}: supercell miss {miss:.2e} (prototype 4.9e-12 / 6.1e-12)");
        assert!(miss <= SC_BAR_GDF, "{exx:?}: {miss:e}");
        refs.push(fsc);
    }
    for m in FIT_MUTANTS {
        let f = kg_uhf(&ks, KGradJk::RsGdf(src), &scf, ExxDiv::None, Some(m));
        let miss = sc_miss(&f.grad, &refs[0]);
        eprintln!("  H3 UHF RS-GDF mutant {m:?}: miss {miss:.2e}");
        assert!(miss > MUTANT_BAR, "mutant {m:?} not caught ({miss:e})");
    }
}

// ======================================================== input checks

#[test]
fn kpoint_gradient_refuses_a_mismatched_mesh_and_an_unconverged_scf() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let mut scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let other = KPointMesh::gamma_centred(&ks.cell, [1, 3, 1]).unwrap();
    let err = kpoint_rhf_gradient(
        &ks.cell,
        &ks.prep,
        &other,
        &hcore_cfg(),
        &ks.hk,
        KGradJk::Dense(&eri),
        &scf,
        ExxDiv::None,
        &gcfg(None),
    )
    .expect_err("a different mesh must be refused");
    eprintln!("mesh mismatch: {err}");
    assert!(err.to_string().contains("k-points"), "{err}");
    scf.converged = false;
    let err = kg_rhf_try(&ks, &eri, &scf).expect_err("unconverged must be refused");
    assert!(err.to_string().contains("did not converge"), "{err}");
}
