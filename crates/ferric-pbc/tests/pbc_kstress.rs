//! k-point RHF / UHF analytic stress (`ferric_pbc::kstress`), dense AFT.
//!
//! What is anchored against what (every anchor at ONE fixed density, so SCF
//! convergence does not enter the comparison unless stated):
//! (0) KERNEL: the residue-resolved pair-FT strain kernel, summed over the
//!     residues, is the Gamma strain kernel at the same vectors, and EACH
//!     residue's `dP_r` is the finite difference of the independent residue
//!     pair FT ([`pair_ft_residues_chunked`]) on strained cells with `K`
//!     held at fixed Miller index + fixed fractional `q`.
//! (1) HCORE GATE (before any ERI term is trusted): the one-electron + `E_nn`
//!     strain at a FIXED `D(k)` against central FD of
//!     `(1/N_k) Σ_k tr[D h(k)] + E_nn` on strained cells with the mesh rebuilt
//!     from the same integer numerators.
//! (2) 1×1×1 mesh ≡ the Gamma stress, term by term.
//!
//! ARTIFACT vs PHYSICS hypotheses are stated next to each test.
//!
//! The tests whose per-commit CI time was >= 120 s
//! (`h2_1x1x3_stress_is_the_supercell_stress_per_cell`,
//! `h2_1x1x3_rsgdf_stress_is_the_supercell_stress_per_cell`,
//! `h2_mp_1x1x2_stress_matches_fd_of_own_k_energy_all_nine`) run weekly from
//! `validation_pbc_kstress.rs`; the shared helpers are
//! `pbc_kstress_support/mod.rs`.

mod common;
#[path = "pbc_kstress_support/mod.rs"]
mod support;

use support::*;

/// PHYSICS: `P^{k}(K) = Σ_r e^{ik·t_r} P_r(K)` with `P_r` a pure function of
/// `K` and the image set, and `dP_r/dε` its strain derivative at fixed Miller
/// index, so (a) `Σ_r P_r`, `Σ_r dP_r` are the Gamma kernel's `P`, `dP` at the
/// same `K` and (b) each `dP_r` is the FD of the independent residue FT.
/// ARTIFACT: a wrong bucket map (e.g. residues swapped between images) leaves
/// (a) intact (a sum over all residues) but breaks (b); a bucketed copy that
/// drops a term breaks both. (b) is therefore the sharp check.
#[test]
fn residue_strain_kernel_is_gamma_kernel_summed_and_fd_per_residue() {
    let cell = triclinic_cell();
    let bs = sp_basis_h();
    let prep = prep_for(&cell, &bs);
    let ks = shifted_ks(&cell, [0.31, 0.17, 1.0 / 3.0], 8);
    let thresh = 1e-15;
    let moduli = [1, 2, 3];
    let nr = 6;
    let (p, dp) = collect_strain(&cell, &prep, &ks, moduli, thresh, PairFtStrainTerms::ALL);
    assert_eq!(p.len(), nr);
    // (a) sum over residues = Gamma kernel.
    let mut gp = None;
    pair_ft_strain_chunked(
        &cell,
        &prep,
        &ks,
        thresh,
        1 << 30,
        0,
        PairFtStrainTerms::ALL,
        |_, _, p, dp| {
            gp = Some((p.clone(), dp.to_vec()));
            Ok(())
        },
    )
    .unwrap();
    let (gp, gdp) = gp.unwrap();
    let mut sum_p = Array3::<Complex64>::zeros(gp.dim());
    for r in &p {
        sum_p += r;
    }
    let ea = (&sum_p - &gp).iter().fold(0.0_f64, |m, z| m.max(z.norm()));
    let mut ed = 0.0_f64;
    for k in 0..9 {
        let mut s = Array3::<Complex64>::zeros(gp.dim());
        for r in &dp {
            s += &r[k];
        }
        ed = ed.max((&s - &gdp[k]).iter().fold(0.0_f64, |m, z| m.max(z.norm())));
    }
    eprintln!("residue strain kernel: |ΣP_r − P_gamma| {ea:.2e}, |ΣdP_r − dP_gamma| {ed:.2e}");
    assert!(ea < 1e-13 && ed < 1e-12, "{ea:e} {ed:e}");
    // (b) per residue vs FD of the independent residue FT.
    let (_, dp_nc) = collect_strain(
        &cell,
        &prep,
        &ks,
        moduli,
        thresh,
        PairFtStrainTerms {
            centres: false,
            g_shape: true,
        },
    );
    let (_, dp_ng) = collect_strain(
        &cell,
        &prep,
        &ks,
        moduli,
        thresh,
        PairFtStrainTerms {
            centres: true,
            g_shape: false,
        },
    );
    let h = 1e-5;
    let n = prep.nbasis();
    let (mut worst, mut fmax) = (0.0_f64, 0.0_f64);
    let (mut w_nc, mut w_ng, mut w_swap) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (a, b) in ALL9 {
        let fd_p = |s: f64| {
            let c = strained(&cell, a, b, s * h);
            let pr = prep_for(&c, &bs);
            let kk = k_on(&cell, &c, &ks);
            let mut out = None;
            pair_ft_residues_chunked(&c, &pr, &kk, moduli, thresh, 1 << 30, 0, |_, _, q| {
                out = Some(q.to_vec());
                Ok(())
            })
            .unwrap();
            out.unwrap()
        };
        let (pp, pm) = (fd_p(1.0), fd_p(-1.0));
        let k = 3 * a + b;
        for r in 0..nr {
            for m in 0..n {
                for q in 0..n {
                    for g in 0..ks.len() {
                        let f = (pp[r][[m, q, g]] - pm[r][[m, q, g]]) / (2.0 * h);
                        fmax = fmax.max(f.norm());
                        worst = worst.max((dp[r][k][[m, q, g]] - f).norm());
                        w_nc = w_nc.max((dp_nc[r][k][[m, q, g]] - f).norm());
                        w_ng = w_ng.max((dp_ng[r][k][[m, q, g]] - f).norm());
                        // MUTANT: residue buckets rotated by one.
                        w_swap = w_swap.max((dp[(r + 1) % nr][k][[m, q, g]] - f).norm());
                    }
                }
            }
        }
    }
    eprintln!("residue strain kernel vs FD per residue: {worst:.2e} (|FD| max {fmax:.2e})");
    assert!(fmax > 1e-3, "the strain must change P_r");
    eprintln!(
        "  kernel mutants: no centres {w_nc:.2e}, no G shape {w_ng:.2e}, rotated buckets {w_swap:.2e}"
    );
    assert!(worst < 1e-7 * fmax.max(1.0), "{worst:e}");
    for (tag, w) in [
        ("FtNoCentres", w_nc),
        ("GUnstrained", w_ng),
        ("rotated buckets", w_swap),
    ] {
        assert!(w > 1e-3 * fmax, "{tag} escaped the per-residue FD: {w:e}");
    }
}

/// PHYSICS: with `D(k)` fixed and the mesh at fixed integer numerators, the
/// strain derivative of `(1/N_k) Σ tr[D h(k)] + E_nn` is kinetic + V_SR + V_LR
/// + c0 (volume AND `c0 Z_tot Σ D dS`) + Ewald. ARTIFACT: a missing/wrong
/// phase weight (NoPhase), a missing 1/N_k, or a pair vector without its
/// image shift (NoSrImages) all leave 1×1×1 intact but move this by O(1).
#[test]
fn hcore_strain_at_fixed_density_matches_fd_h2_1x1x3() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h2_cell_g(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let an = kpoint_hcore_stress_fixed_d(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        &scf.densities,
        &scfg(None),
    )
    .unwrap();
    let fd = fd_hcore(&ks.cell, &bs, n, &scf.densities);
    let tot = an.hcore_total();
    let miss = max_mat(&tot, &fd);
    eprintln!("H2 1x1x3 hcore fixed-D: |analytic − FD| {miss:.2e}");
    for a in 0..3 {
        eprintln!(
            "  [{a}] analytic {:+.8e} {:+.8e} {:+.8e} | FD {:+.8e} {:+.8e} {:+.8e}",
            tot[a][0], tot[a][1], tot[a][2], fd[a][0], fd[a][1], fd[a][2]
        );
    }
    assert!(miss < HCORE_FD_BAR, "hcore FD miss {miss:e}");
    for m in [
        KStressMutation::NoPhase,
        KStressMutation::NoInvNk,
        KStressMutation::NoSrImages,
        KStressMutation::NoC0Volume,
        KStressMutation::NoVolume,
        KStressMutation::NoEwaldLr,
        KStressMutation::NoEwaldBg,
        KStressMutation::FtNoCentres,
        KStressMutation::GUnstrained,
    ] {
        let a = kpoint_hcore_stress_fixed_d(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &ks.hk,
            &scf.densities,
            &scfg(Some(m)),
        )
        .unwrap();
        let t = a.hcore_total();
        let mm = max_mat(&t, &fd);
        // Off-diagonal miss (the volume-only mutants are diagonal by construction).
        let mut off = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    off = off.max((t[i][j] - fd[i][j]).abs());
                }
            }
        }
        eprintln!("  hcore mutant {m:?}: miss {mm:.2e} (off-diagonal {off:.2e})");
        assert!(mm > 1e-4, "{m:?} not caught by the hcore gate ({mm:e})");
    }
}

// ============================================================ (2) 1x1x1 = Gamma

/// PHYSICS: at 1×1×1 every phase is 1, the residue is the only bucket, and
/// the k code is the Gamma code over the full sphere instead of the half
/// sphere. ARTIFACT: independent constructions (real half-sphere vs complex
/// full-sphere residue kernel; image-summed vs phase-folded weights), so
/// agreement is not an identity of one code path.
#[test]
fn one_point_mesh_hcore_parts_are_the_gamma_parts() {
    // (The triclinic s+p cell is covered by the `#[ignore]`d full 1x1x1 anchor.)
    for (name, cell, bs) in [("H2", h2_cell_g(), pyscf_sto3g_h())] {
        let ks = ksys(cell.clone(), &bs, [1, 1, 1]);
        let eri = kdense(&ks, ANCHOR_PRECISION);
        let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
        let g = gsys(cell, &bs);
        let gscf = gamma_restricted(real_of(&scf.densities[0]));
        let gs = gamma_rhf_stress(
            &g.cell,
            &g.prep,
            &hcore_cfg(),
            &g.hc,
            &g.eri,
            &gscf,
            ExxDiv::None,
            &GammaStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
        )
        .unwrap();
        let an = kpoint_hcore_stress_fixed_d(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &ks.hk,
            &scf.densities,
            &scfg(None),
        )
        .unwrap();
        let pairs: [(&str, &Mat3, &Mat3); 7] = [
            ("kinetic", &an.kinetic, &gs.parts.kinetic),
            ("vsr", &an.vsr, &gs.parts.vsr),
            ("vlr", &an.vlr, &gs.parts.vlr),
            ("c0_volume", &an.c0_volume, &gs.parts.c0_volume),
            ("nn_sr", &an.nn_sr, &gs.parts.nn_sr),
            ("nn_lr", &an.nn_lr, &gs.parts.nn_lr),
            ("nn_background", &an.nn_background, &gs.parts.nn_background),
        ];
        for (tag, a, b) in pairs {
            let d = max_mat(a, b);
            eprintln!("  {name} 1x1x1 {tag}: |k − Gamma| {d:.2e}");
            assert!(d < 1e-10, "{name} {tag}: {d:e}");
        }
    }
}

/// PHYSICS: at 1×1×1 the k stress is the Gamma stress. ARTIFACT: independent
/// constructions (half-sphere real Gamma kernel vs full-sphere complex
/// residue kernel; image-summed vs phase-folded weights), but BLIND to the
/// phases (all 1), to 1/N_k (N_k = 1) and to the supercell Madelung.
#[test]
fn one_point_mesh_rhf_stress_is_the_gamma_stress() {
    one_point_rhf(("H2", h2_cell_g(), pyscf_sto3g_h(), ANTISYM_BAR));
}

#[test]
#[ignore = "slow (~1 min release): triclinic s+p, 2.6M SR triplets"]
fn one_point_mesh_rhf_stress_is_the_gamma_stress_triclinic_sp() {
    one_point_rhf(("tri2 s+p", triclinic_cell(), sp_basis_h(), P_ANTISYM_BAR));
}

#[test]
fn one_point_mesh_uhf_stress_is_the_gamma_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h3_cell(), &bs, [1, 1, 1]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = uscf_pair(&ks, &eri, 2, 1);
    let g = gsys(h3_cell(), &bs);
    for exx in EXX {
        let scf = &scfs[exx_idx(exx)];
        let gscf = gamma_unrestricted(
            real_of(&scf.density_alpha[0]),
            real_of(&scf.density_beta[0]),
        );
        let k = ks_uhf(&ks, &eri, scf, exx, None);
        let gs = gamma_stress(&g, &gscf, exx);
        assert_one_point("H3 UHF", &k, &gs, exx, ANTISYM_BAR);
    }
}

// ====================================================== energy bookkeeping

/// The pieces the stress differentiates add up to the SCF energy: a check
/// that `e_vlr`, `e_coulomb`, `e_exchange`, `e_madelung` are the energy's
/// terms with the energy's prefactors (the strain of a mis-scaled `Φ` would
/// still be self-consistent against nothing).
#[test]
fn stress_energy_pieces_sum_to_the_scf_energy_h2_1x1x3() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    for (exx, m) in [(ExxDiv::None, 0.0), (ExxDiv::Ewald, vm)] {
        let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(m)));
        let st = ks_rhf(&ks, &eri, &scf, exx, None);
        // (1/N_k) Σ_k tr[h D] + E_J + E_K + E_M + E_nn.
        let nk = ks.mesh.nk();
        let n = ks.prep.nbasis();
        let mut e1 = 0.0;
        for (d, h) in scf.densities.iter().zip(&ks.hk.h) {
            for a in 0..n {
                for b in 0..n {
                    e1 += (d[(a, b)] * h[(b, a)]).re;
                }
            }
        }
        e1 /= nk as f64;
        let (ej, ek) = (st.e_coulomb.unwrap(), st.e_exchange.unwrap());
        let e = e1 + ej + ek + st.e_madelung + ks.hk.enn;
        eprintln!(
            "  {exx:?}: E_pieces {e:.12} vs SCF {:.12} (diff {:.2e}); E_J {:.8} E_K {:.8} E_M {:.8}",
            scf.energy,
            e - scf.energy,
            ej,
            ek,
            st.e_madelung
        );
        assert!(
            (e - scf.energy).abs() < 1e-9,
            "{exx:?}: {:e}",
            e - scf.energy
        );
    }
}

/// The Gamma-style term mutants against the same supercell anchor.
#[test]
#[ignore = "slow (~2 min release): 22 more stress evaluations after the supercell anchor"]
fn h2_1x1x3_term_mutants_miss_the_supercell_anchor() {
    let bs = pyscf_sto3g_h();
    let (ks, eri, scfs, refs) = rhf_supercell_anchor("H2", h2_cell_g(), &bs, [1, 1, 3]);
    assert_mutants_fail(
        "H2 1x1x3",
        &|exx, m| ks_rhf(&ks, &eri, &scfs[exx_idx(exx)], exx, m),
        &refs,
        &TERM_MUTANTS,
        &TERM_EWALD_MUTANTS,
    );
}

/// UHF H3 doublet: supercell (6, 3), multiplicity 4.
#[test]
#[ignore = "slow (~8 min release): 9-atom supercell Gamma stress x2 + 36 k stress evaluations"]
fn h3_uhf_1x1x3_stress_is_the_supercell_stress_per_cell() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h3_cell(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = uscf_pair(&ks, &eri, 2, 1);
    let g = gsys(supercell(&h3_cell(), n, 4), &bs);
    let mut de = [ZERO; 2];
    for (i, exx) in EXX.into_iter().enumerate() {
        let gscf = gamma_unrestricted(
            unfold(&ks.mesh, &scfs[i].density_alpha),
            unfold(&ks.mesh, &scfs[i].density_beta),
        );
        let k = ks_uhf(&ks, &eri, &scfs[i], exx, None);
        let gs = gamma_stress(&g, &gscf, exx);
        let r = scale3(&gs.de_deps, 1.0 / 3.0);
        let miss = max_mat(&k.de_deps, &r);
        report(&format!("H3 UHF {n:?} {exx:?}"), &k);
        eprintln!("  H3 UHF {n:?} {exx:?}: |dE/dε_k − dE/dε_sc/N| {miss:.2e}");
        assert!(miss <= SC_BAR, "{exx:?}: {miss:e}");
        assert!(k.antisymmetry() <= ANTISYM_BAR);
        de[i] = r;
    }
    let refs = ScRef { de };
    let run = |exx, m| ks_uhf(&ks, &eri, &scfs[exx_idx(exx)], exx, m);
    assert_mutants_fail("H3 UHF 1x1x3", &run, &refs, &K_MUTANTS, &K_EWALD_MUTANTS);
    assert_mutants_fail(
        "H3 UHF 1x1x3",
        &run,
        &refs,
        &TERM_MUTANTS,
        &TERM_EWALD_MUTANTS,
    );
}

/// BLIND SPOT, asserted: on the TRIM-only 2×2×2 mesh `D(k)` is real and
/// `q ≡ −q`, so ConjD and WrongQ are IDENTITIES; NoPhase and NoInvNk are not.
#[test]
#[ignore = "slow (~2.5 min release): 8-point mesh, 10 stress evaluations"]
fn h2_2x2x2_trim_mesh_is_blind_to_phase_mutants() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [2, 2, 2]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = scf_pair(&ks, &eri);
    for exx in EXX {
        let scf = &scfs[exx_idx(exx)];
        let f0 = ks_rhf(&ks, &eri, scf, exx, None);
        for m in [KStressMutation::ConjD, KStressMutation::WrongQ] {
            let f = ks_rhf(&ks, &eri, scf, exx, Some(m));
            let d = max_mat(&f.de_deps, &f0.de_deps);
            eprintln!("  H2 2x2x2 {exx:?} {m:?}: |mut − ref| {d:.2e} (identity on TRIM meshes)");
            assert!(
                d <= IDENTITY_BAR,
                "{m:?} should be an identity on 2x2x2 ({d:e})"
            );
        }
        for m in [KStressMutation::NoPhase, KStressMutation::NoInvNk] {
            let f = ks_rhf(&ks, &eri, scf, exx, Some(m));
            let d = max_mat(&f.de_deps, &f0.de_deps);
            eprintln!("  H2 2x2x2 {exx:?} {m:?}: |mut − ref| {d:.2e}");
            assert!(d > MUTANT_BAR, "{m:?} not caught on 2x2x2 ({d:e})");
        }
        assert!(f0.antisymmetry() <= ANTISYM_BAR);
    }
}

/// closed-shell kUHF(D/2, D/2) ≡ kRHF(D).
#[test]
fn closed_shell_kuhf_stress_is_the_krhf_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let rs = scf_pair(&ks, &eri);
    let mut us = uscf_pair(&ks, &eri, 1, 1);
    for exx in EXX {
        let (r, u) = (&rs[exx_idx(exx)], &mut us[exx_idx(exx)]);
        let half: Vec<Array2<Complex64>> =
            r.densities.iter().map(|d| d.mapv(|z| z * 0.5)).collect();
        u.density_alpha = half.clone();
        u.density_beta = half;
        let sr = ks_rhf(&ks, &eri, r, exx, None);
        let su = ks_uhf(&ks, &eri, u, exx, None);
        let d = max_mat(&sr.de_deps, &su.de_deps);
        eprintln!("  H2 1x1x3 {exx:?}: |σ_kUHF(D/2,D/2) − σ_kRHF(D)| {d:.2e}");
        assert!(d <= IDENTITY_BAR, "{exx:?}: {d:e}");
    }
}

/// PHYSICS: analytic `dE/dε` of the k energy vs central FD (h = 1e-4) of
/// ferric's own k energy on strained cells (lattice AND atoms strained, index
/// sets frozen, k at fixed fractional coordinates), all nine components, both
/// exxdiv, with a fresh SCF at every strained point (the analytic value is at
/// the converged density, so the density response drops out by stationarity).
#[test]
fn h2_1x1x3_stress_matches_fd_of_own_k_energy_all_nine() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h2_cell_g(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scf = [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))),
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))),
    ];
    let mut fd = [[[0.0; 3]; 3]; 2];
    for (i, j) in ALL9 {
        let ep = k_energies(&strained(&ks.cell, i, j, FD_H), &bs, n);
        let em = k_energies(&strained(&ks.cell, i, j, -FD_H), &bs, n);
        for x in 0..2 {
            fd[x][i][j] = (ep[x] - em[x]) / (2.0 * FD_H);
        }
    }
    for (x, exx) in EXX.into_iter().enumerate() {
        let s = ks_rhf(&ks, &eri, &scf[x], exx, None);
        let miss = max_mat(&s.de_deps, &fd[x]);
        eprintln!("  H2 1x1x3 {exx:?}: |analytic − FD| {miss:.2e}");
        assert!(miss <= K_FD_BAR, "{exx:?}: FD miss {miss:e}");
    }
}

// ====================================================== refusals

#[test]
fn stress_refuses_unconverged_scf_and_ecp() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let mut scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    // Unconverged SCF.
    scf.converged = false;
    let e = kpoint_rhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        KStressJk::Dense(&eri),
        &scf,
        ExxDiv::None,
        &scfg(None),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("did not converge"), "{e}");
    scf.converged = true;
    // An hcore carrying V_ECP for a cell whose basis has no ECP centre: the ECP walk is
    // driven by the basis and cross-checked against hk.v_ecp.
    let mut hk_ecp = ks.hk.clone();
    hk_ecp.v_ecp = Some(hk_ecp.h.clone());
    let e = kpoint_rhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &hk_ecp,
        KStressJk::Dense(&eri),
        &scf,
        ExxDiv::None,
        &scfg(None),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("carries V_ECP but the basis"), "{e}");
}

/// The UHF twin of the FD anchor (H3 doublet 1x1x3, both exxdiv), at h and 2h:
/// the h = 1e-4 miss is the central-FD h² truncation (Richardson removes it),
/// and the two UHF-specific mutants (alpha/beta cross-paired exchange, swapped
/// spin in the Pulay weight; both identities for RHF) must miss.
#[test]
#[ignore = "slow (~90 s release): 36 strained dense-AFT builds + 72 k-UHF SCFs (h and 2h)"]
fn h3_uhf_1x1x3_stress_matches_fd_of_own_k_energy_all_nine() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h3_cell(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scf = [
        kuhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)), 2, 1),
        kuhf(&ks, Box::new(eri.jk_builder_with_madelung(vm)), 2, 1),
    ];
    let e = |c: &Cell| k_energies_uhf(c, &bs, n);
    let fd1 = fd_all9(&ks.cell, FD_H, &e);
    let fd2 = fd_all9(&ks.cell, 2.0 * FD_H, &e);
    for (x, exx) in EXX.into_iter().enumerate() {
        let s = ks_uhf(&ks, &eri, &scf[x], exx, None);
        let mut rich = ZERO;
        for a in 0..3 {
            for b in 0..3 {
                rich[a][b] = (4.0 * fd1[x][a][b] - fd2[x][a][b]) / 3.0;
            }
        }
        let miss = max_mat(&s.de_deps, &fd1[x]);
        let miss2 = max_mat(&s.de_deps, &fd2[x]);
        let missr = max_mat(&s.de_deps, &rich);
        eprintln!(
            "  H3 UHF 1x1x3 {exx:?}: |analytic − FD| h {miss:.2e}, 2h {miss2:.2e} (ratio {:.2}), \
             Richardson {missr:.2e}",
            miss2 / miss
        );
        assert!(miss <= K_FD_BAR_UHF, "{exx:?}: FD miss {miss:e}");
        // h² scaling: the miss is truncation, not a formula error.
        assert!(
            (3.0..5.0).contains(&(miss2 / miss)),
            "not h²: {}",
            miss2 / miss
        );
        assert!(missr <= K_FD_BAR_UHF / 10.0, "Richardson miss {missr:e}");
        for m in [KStressMutation::CrossSpinK, KStressMutation::SwapSpinFock] {
            let sm = ks_uhf(&ks, &eri, &scf[x], exx, Some(m));
            let mm = max_mat(&sm.de_deps, &fd1[x]);
            eprintln!("  H3 UHF 1x1x3 {exx:?} mutant {m:?}: miss vs FD {mm:.2e}");
            assert!(mm > MUTANT_BAR, "{m:?} {exx:?} not caught ({mm:e})");
        }
    }
}

/// PHYSICS: a stress is only meaningful for the energy whose SCF produced
/// `D`; the stored Fock tells which one that was. ARTIFACT: a wrong `exxdiv`
/// or a KS result both give a FINITE σ (nothing else fails), so the guard is
/// the stored-Fock comparison. Measured mismatches are printed below; the
/// matching case sits at the 1e-12 level (see `FOCK_CONSISTENCY_BAR`).
#[test]
fn stress_refuses_a_wrong_exxdiv_and_a_ks_result() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 2];
    let ks = ksys(h2_cell_g(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scf_none = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let scf_ewald = krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm)));
    // Matching inputs pass (and the match is far below the bar).
    let a = rhf_stress_result(&ks, &eri, &scf_none, ExxDiv::None).expect("none/none");
    let b = rhf_stress_result(&ks, &eri, &scf_ewald, ExxDiv::Ewald).expect("ewald/ewald");
    eprintln!(
        "  matching inputs: Fock mismatch {:.2e} (none), {:.2e} (ewald)",
        a.fock_mismatch, b.fock_mismatch
    );
    assert!(a.fock_mismatch < 1e-11 && b.fock_mismatch < 1e-11);
    // (a) wrong exxdiv, both directions.
    for (tag, scf, exx) in [
        ("SCF none, stress ewald", &scf_none, ExxDiv::Ewald),
        ("SCF ewald, stress none", &scf_ewald, ExxDiv::None),
    ] {
        let e = rhf_stress_result(&ks, &eri, scf, exx)
            .unwrap_err()
            .to_string();
        eprintln!("  {tag}: {e}");
        assert!(e.contains("differs from the SCF's stored fock"), "{e}");
    }
    // (b) a KS-DFT (LDA) result on the same mesh: its Fock carries V_xc.
    let mut kcfg = KRksConfig::new(&ks.cell, ExxDiv::None, "LDA");
    kcfg.krhf.hcore = hcore_cfg();
    kcfg.krhf.scf.energy_conv = 1e-10;
    kcfg.krhf.scf.grad_conv = 1e-7;
    kcfg.grid = PeriodicGridConfig::with_size(30, 50);
    let kres = solve_krks(&ks.cell, &ks.prep, None, &ks.mesh, &kcfg).expect("k-LDA");
    let e = rhf_stress_result(&ks, &eri, &kres.scf, ExxDiv::None)
        .unwrap_err()
        .to_string();
    eprintln!("  KS (LDA) result: {e}");
    assert!(e.contains("differs from the SCF's stored fock"), "{e}");
}

/// A dense source built with a range-separated K kernel is refused (its
/// strain is not implemented), even though its own SCF/Fock are consistent.
#[test]
fn stress_refuses_a_range_separated_dense_source() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 2]);
    let eri = KDenseAftEri::build(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &ks.hk.s,
        ExxDiv::None,
        &KDenseAftConfig {
            precision: ANCHOR_PRECISION,
            rsh: Some(ferric_pbc::rsh::RshParams {
                omega: 0.4,
                c_sr: 1.0,
                c_lr: 0.25,
            }),
            ..Default::default()
        },
    )
    .expect("rsh dense kernels");
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let e = rhf_stress_result(&ks, &eri, &scf, ExxDiv::None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("range-separated"), "{e}");
}

/// The fixed-D hcore entry applies the same input checks.
#[test]
fn hcore_fixed_d_entry_checks_its_inputs() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 2]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scf = krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0)));
    let call = |cfg: &PeriodicHcoreConfig, hk: &PeriodicHcoreK, d: &[Array2<Complex64>]| {
        kpoint_hcore_stress_fixed_d(&ks.cell, &ks.prep, &ks.mesh, cfg, hk, d, &scfg(None))
            .err()
            .map(|e| e.to_string())
    };
    assert!(call(&hcore_cfg(), &ks.hk, &scf.densities).is_none());
    let mut other = hcore_cfg();
    other.omega = 0.9;
    assert!(call(&other, &ks.hk, &scf.densities)
        .unwrap()
        .contains("omega"));
    assert!(call(&hcore_cfg(), &ks.hk, &scf.densities[..1])
        .unwrap()
        .contains("shape mismatch"));
    let mut hk_ecp = ks.hk.clone();
    hk_ecp.v_ecp = Some(hk_ecp.h.clone());
    assert!(call(&hcore_cfg(), &hk_ecp, &scf.densities)
        .unwrap()
        .contains("ECP"));
}

// ====================================================== measured mutant claims

/// The Gamma-style term mutants against the 1x1x1 Gamma anchor (H2): each
/// must miss it, and the k-specific ones (phases, q, 1/N_k, mesh Madelung)
/// are BLIND there (asserted) — which is why the 1x1x3 supercell anchor, not
/// this one, carries them.
#[test]
#[ignore = "slow-ish (~1 min release): ~30 stress evaluations on H2 1x1x1"]
fn term_mutants_miss_the_one_point_gamma_anchor() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 1]);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = scf_pair(&ks, &eri);
    let g = gsys(h2_cell_g(), &bs);
    for (i, exx) in EXX.into_iter().enumerate() {
        let scf = &scfs[i];
        let gscf = gamma_restricted(real_of(&scf.densities[0]));
        let gs = gamma_stress(&g, &gscf, exx);
        let mut all: Vec<KStressMutation> = TERM_MUTANTS.to_vec();
        all.extend(TERM_EWALD_MUTANTS);
        for m in all {
            if TERM_EWALD_MUTANTS.contains(&m) && i == 0 {
                continue;
            }
            let miss = max_mat(&ks_rhf(&ks, &eri, scf, exx, Some(m)).de_deps, &gs.de_deps);
            eprintln!("  1x1x1 anchor, mutant {m:?} {exx:?}: miss {miss:.2e}");
            assert!(miss > MUTANT_BAR, "{m:?} {exx:?}: {miss:e}");
        }
        let mut blind: Vec<KStressMutation> = K_MUTANTS.to_vec();
        blind.extend(K_EWALD_MUTANTS);
        for m in blind {
            let miss = max_mat(&ks_rhf(&ks, &eri, scf, exx, Some(m)).de_deps, &gs.de_deps);
            eprintln!("  1x1x1 anchor, k-mutant {m:?} {exx:?}: miss {miss:.2e} (blind)");
            // Identities at N_k = 1 (phases 1, 1/N_k = 1, mesh = primitive v_M).
            assert!(miss <= GAMMA_BAR * 100.0, "{m:?} {exx:?}: {miss:e}");
        }
    }
}

/// Per-commit canary of `validation_pbc_kstress.rs`'s
/// `h2_mp_1x1x2_stress_matches_fd_of_own_k_energy_all_nine`: the same
/// Monkhorst-Pack 1x1x2 mesh (residue modulus 4, no supercell anchor exists),
/// FD of ferric's own k energy on the zz and xz strains only (h = 1e-4, both
/// exxdiv), and the NoPhase / NoInvNk / Madelung mutants against the same
/// two components.
#[test]
fn h2_mp_1x1x2_stress_matches_fd_of_own_k_energy_zz_and_xz() {
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
    let comps = [(2, 2), (0, 2)];
    let fd: Vec<[f64; 2]> = comps
        .iter()
        .map(|&(i, j)| {
            let ep = k_energies_c(&strained(&ks.cell, i, j, FD_H), &bs, n, c);
            let em = k_energies_c(&strained(&ks.cell, i, j, -FD_H), &bs, n, c);
            [0, 1].map(|x| (ep[x] - em[x]) / (2.0 * FD_H))
        })
        .collect();
    let miss = |s: &KStress, x: usize| {
        comps
            .iter()
            .zip(&fd)
            .map(|(&(i, j), f)| (s.de_deps[i][j] - f[x]).abs())
            .fold(0.0_f64, f64::max)
    };
    for (x, exx) in EXX.into_iter().enumerate() {
        let s = ks_rhf(&ks, &eri, &scf[x], exx, None);
        let m = miss(&s, x);
        eprintln!("  H2 MP 1x1x2 {exx:?}: |analytic − FD| (zz, xz) {m:.2e}");
        assert!(m <= K_FD_BAR, "{exx:?}: {m:e}");
        for mu in [KStressMutation::NoPhase, KStressMutation::NoInvNk] {
            let mm = miss(&ks_rhf(&ks, &eri, &scf[x], exx, Some(mu)), x);
            eprintln!("  H2 MP 1x1x2 {exx:?} mutant {mu:?}: miss {mm:.2e}");
            assert!(mm > MUTANT_BAR, "{mu:?} {exx:?}: {mm:e}");
        }
        if exx == ExxDiv::Ewald {
            for mu in K_EWALD_MUTANTS {
                let mm = miss(&ks_rhf(&ks, &eri, &scf[x], exx, Some(mu)), x);
                eprintln!("  H2 MP 1x1x2 {exx:?} mutant {mu:?}: miss {mm:.2e}");
                assert!(mm > MUTANT_BAR, "{mu:?}: {mm:e}");
            }
        }
    }
}

/// l > 0 at N_k > 1: H2 with an s+p basis (nao 8), Gamma-centred 1x1x3
/// (non-TRIM), FD of the own k energy, analytic at the default gradient
/// nucleus exponent and at 1e8 (both agree with the FD at the FD-truncation
/// level; the former exponent-dependent p-shell SR-derivative noise is gone,
/// #434).
#[test]
#[ignore = "slow: s+p H2 1x1x3, 18 strained dense-AFT builds + 36 k-SCFs"]
fn h2_sp_1x1x3_stress_matches_fd_of_own_k_energy_all_nine() {
    let bs = sp_basis_h();
    let n = [1, 1, 3];
    let c = MeshCentring::Gamma;
    let ks = ksys_c(h2_cell_g(), &bs, n, c);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scf = [
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(0.0))),
        krhf(&ks, Box::new(eri.jk_builder_with_madelung(vm))),
    ];
    let e = |cell: &Cell| k_energies_c(cell, &bs, n, c);
    let fd1 = fd_all9(&ks.cell, FD_H, &e);
    for (x, exx) in EXX.into_iter().enumerate() {
        for nuc in [None, Some(1e8)] {
            let s = kpoint_rhf_stress(
                &ks.cell,
                &ks.prep,
                &ks.mesh,
                &hcore_cfg(),
                &ks.hk,
                KStressJk::Dense(&eri),
                &scf[x],
                exx,
                &KStressConfig {
                    nucleus_exponent: nuc,
                    ..scfg(None)
                },
            )
            .unwrap();
            let m1 = max_mat(&s.de_deps, &fd1[x]);
            eprintln!(
                "  H2 s+p 1x1x3 {exx:?} nucleus exponent {nuc:?}: |analytic − FD| {m1:.2e}, \
                 antisym(σ) {:.2e}",
                s.antisymmetry()
            );
            assert!(m1 <= K_FD_BAR_P, "{exx:?} {nuc:?}: {m1:e}");
        }
        for m in [
            KStressMutation::NoPhase,
            KStressMutation::WrongQ,
            KStressMutation::ConjD,
        ] {
            let mm = max_mat(&ks_rhf(&ks, &eri, &scf[x], exx, Some(m)).de_deps, &fd1[x]);
            eprintln!("  H2 s+p 1x1x3 {exx:?} mutant {m:?}: miss {mm:.2e}");
            assert!(mm > MUTANT_BAR, "{m:?} {exx:?}: {mm:e}");
        }
    }
}

/// PHYSICS: at 1x1x1 the k RS-GDF stress is the Gamma RS-GDF stress, term by
/// term (SR3, SR2, LR3, LR2, G = 0 overlap, both G = 0 volumes). ARTIFACT: the
/// two are independent constructions (half-sphere real LR vs full-sphere
/// complex per-q LR; one Y vs per-(l, t) bins; `Y q` vs `Z`-based `M_g0`), but
/// the anchor is BLIND to phases, 1/N_k, q != 0 (only the q = 0 class exists)
/// and the supercell Madelung; the 1x1x3 supercell anchor covers those.
#[test]
fn one_point_mesh_rsgdf_stress_is_the_gamma_rsgdf_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 1]);
    let fs = fitsys(&ks);
    let scfs = fit_scfs(&ks, &fs);
    let g = gfit(h2_cell_g(), &bs);
    for exx in EXX {
        let scf = &scfs[exx_idx(exx)];
        let gscf = gamma_restricted(real_of(&scf.densities[0]));
        let k = ks_rhf_fit(&ks, &fs, scf, exx, None);
        let gs = gamma_fit_stress(&g, &gscf, exx);
        report(&format!("H2 RS-GDF 1x1x1 {exx:?}"), &k);
        let total = max_mat(&k.de_deps, &gs.de_deps);
        eprintln!("  H2 RS-GDF 1x1x1 {exx:?}: |dE/dε k − Gamma| {total:.2e}");
        for ((name, kp), gp) in fit_parts(&k.parts).iter().zip(gamma_fit_parts(&gs.parts)) {
            let d = max_mat(kp, &gp);
            eprintln!(
                "    {name}: |k − Gamma| {d:.2e} (|part| {:.2e})",
                max_mat(kp, &ZERO)
            );
            assert!(d <= GAMMA_FIT_BAR, "{exx:?} {name}: {d:e}");
        }
        assert!(total <= GAMMA_FIT_BAR, "{exx:?}: {total:e}");
        assert!(k.antisymmetry() <= ANTISYM_BAR);
        if exx == ExxDiv::None {
            for m in [
                KStressMutation::NoMetric,
                KStressMutation::NoAuxFt3,
                KStressMutation::NoJ2G0Vol,
                KStressMutation::NoJ3G0Vol,
                KStressMutation::NoG0,
            ] {
                let x = ks_rhf_fit(&ks, &fs, scf, exx, Some(m));
                let miss = max_mat(&x.de_deps, &gs.de_deps);
                eprintln!("  1x1x1 mutant {m:?}: miss vs Gamma {miss:.2e}");
                assert!(miss > MUTANT_BAR, "{m:?} not caught ({miss:e})");
            }
        }
    }
}

/// The remaining fit-term defects against the same anchor, both exxdiv. The
/// volume-only mutants must leave the off-diagonal at the floor. `NoAuxFt`
/// (aux FT strain-free in J3 and J2) errs by the fit error only in the TOTAL
/// (aux stationarity: measured 5.0e-5, the Gamma prototype 7.7e-6 to 1.2e-4),
/// but each of the J3-LR and J2-LR parts misses its supercell value sharply,
/// so the per-part comparison catches it (and any other defect in the shared
/// `aux_ft_strain_shells`). `WrongQ` is an identity on the fitted path (it
/// acts on the dense exchange pairs only). Aux coverage: s and Cartesian p
/// shells here; pure-d aux is in `h2_1x1x3_rsgdf_stress_with_a_d_aux_*`.
#[test]
#[ignore = "slow (~140 s release, loaded box): 22 stress evaluations after the supercell anchor"]
fn h2_1x1x3_rsgdf_term_mutants_miss_the_supercell_anchor() {
    let a = fit_supercell_anchor("H2", h2_cell_g(), &pyscf_sto3g_h(), [1, 1, 3]);
    let terms = FIT_MUTANTS.into_iter().chain([
        KStressMutation::NoG0,
        KStressMutation::NoSrImages,
        KStressMutation::FtNoCentres,
        KStressMutation::GUnstrained,
        KStressMutation::NoVolume,
    ]);
    for m in terms {
        for exx in EXX {
            let x = a.mutant(exx, m);
            eprintln!(
                "  H2 RS-GDF mutant {m:?} {exx:?}: miss {:.2e} (off-diagonal {:.2e}, worst part {:.2e})",
                x.total, x.off, x.part
            );
            assert!(
                x.total > MUTANT_BAR,
                "{m:?} {exx:?} not caught ({:e})",
                x.total
            );
            assert!(x.part > MUTANT_BAR, "{m:?} {exx:?}: parts ({:e})", x.part);
            if FIT_DIAGONAL_ONLY.contains(&m) || m == KStressMutation::NoVolume {
                assert!(x.off <= SC_FIT_BAR, "{m:?} diagonal-only ({:e})", x.off);
            }
        }
    }
    let x = a.mutant(ExxDiv::None, KStressMutation::NoAuxFt);
    eprintln!(
        "  H2 RS-GDF mutant NoAuxFt: total miss {:.2e}, worst part {:.2e}",
        x.total, x.part
    );
    assert!(
        x.total < MUTANT_BAR,
        "total hidden by aux stationarity ({:e})",
        x.total
    );
    assert!(
        x.part > MUTANT_BAR,
        "NoAuxFt not caught per part ({:e})",
        x.part
    );
    let x = a.mutant(ExxDiv::None, KStressMutation::WrongQ);
    assert!(
        x.total <= SC_FIT_BAR,
        "WrongQ must be an identity ({:e})",
        x.total
    );
}

/// PHYSICS: analytic `dE/dε` of the RS-GDF k energy vs central FD (h = 1e-4,
/// fresh SCF + fresh fit at every strained point, lattice AND atoms strained,
/// index sets frozen, k at fixed fractional coordinates), all nine
/// components, both exxdiv. ARTIFACT: the FD sees the energy's truncation
/// (frozen K sets, the energy's own screening), the supercell anchor does
/// not; a piecewise energy (a metric eigenvalue crossing `lindep`, a K at the
/// sphere edge flipping) would show as FD(h) vs FD(2h) disagreeing faster than
/// h², asserted below.
#[test]
fn h2_1x1x3_rsgdf_stress_matches_fd_of_own_k_energy_all_nine() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h2_cell_g(), &bs, n);
    let fs = fitsys(&ks);
    let scfs = fit_scfs(&ks, &fs);
    let kept0: Vec<usize> = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    let e = |c: &Cell| {
        let (e, kept) = k_fit_energies(c, &bs, n);
        assert_eq!(
            kept, kept0,
            "the metric cut changed under strain (E(ε) piecewise)"
        );
        e
    };
    let fd1 = fd_all9(&ks.cell, FD_H, &e);
    let fd2 = fd_all9(&ks.cell, 2.0 * FD_H, &e);
    for exx in EXX {
        let x = exx_idx(exx);
        let s = ks_rhf_fit(&ks, &fs, &scfs[x], exx, None);
        let (m1, m2) = (max_mat(&s.de_deps, &fd1[x]), max_mat(&s.de_deps, &fd2[x]));
        eprintln!(
            "  H2 RS-GDF 1x1x3 {exx:?}: |analytic − FD| h {m1:.2e}, 2h {m2:.2e} (ratio {:.2})",
            m2 / m1
        );
        assert!(m1 <= K_FD_BAR_FIT, "{exx:?}: FD miss {m1:e}");
        assert!((3.0..5.0).contains(&(m2 / m1)), "not h²: {}", m2 / m1);
        // The FD sees the sub-1e-4 fit-error mutant the supercell anchor
        // cannot (both sides carry the same fit).
        let aux_ft = ks_rhf_fit(&ks, &fs, &scfs[x], exx, Some(KStressMutation::NoAuxFt));
        let ma = max_mat(&aux_ft.de_deps, &fd1[x]);
        eprintln!("  H2 RS-GDF 1x1x3 {exx:?} mutant NoAuxFt: |mut − FD| {ma:.2e}");
        assert!(ma > 1e-5, "NoAuxFt not caught by FD ({ma:e})");
    }
}

/// PHYSICS/ARTIFACT as the RHF 1x1x1 anchor, on the H3 doublet (per-spin
/// exchange pairs `D_s Z^H D_s`, spin-summed Coulomb).
#[test]
fn one_point_mesh_rsgdf_uhf_stress_is_the_gamma_rsgdf_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h3_cell(), &bs, [1, 1, 1]);
    let fs = fitsys(&ks);
    let scfs = fit_uscfs(&ks, &fs, 2, 1);
    let g = gfit(h3_cell(), &bs);
    for exx in EXX {
        let scf = &scfs[exx_idx(exx)];
        let gscf = gamma_unrestricted(
            real_of(&scf.density_alpha[0]),
            real_of(&scf.density_beta[0]),
        );
        let k = ks_uhf_fit(&ks, &fs, scf, exx, None);
        let gs = gamma_fit_stress(&g, &gscf, exx);
        let total = max_mat(&k.de_deps, &gs.de_deps);
        report(&format!("H3 UHF RS-GDF 1x1x1 {exx:?}"), &k);
        eprintln!("  H3 UHF RS-GDF 1x1x1 {exx:?}: |dE/dε k − Gamma| {total:.2e}");
        for ((name, kp), gp) in fit_parts(&k.parts).iter().zip(gamma_fit_parts(&gs.parts)) {
            let d = max_mat(kp, &gp);
            eprintln!("    {name}: |k − Gamma| {d:.2e}");
            assert!(d <= GAMMA_FIT_BAR, "{exx:?} {name}: {d:e}");
        }
        assert!(total <= GAMMA_FIT_BAR, "{exx:?}: {total:e}");
    }
}

/// closed-shell kUHF(D/2, D/2) ≡ kRHF(D) on the fitted path.
#[test]
fn closed_shell_kuhf_rsgdf_stress_is_the_krhf_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let fs = fitsys(&ks);
    let rs = fit_scfs(&ks, &fs);
    let mut us = fit_uscfs(&ks, &fs, 1, 1);
    for exx in EXX {
        let (r, u) = (&rs[exx_idx(exx)], &mut us[exx_idx(exx)]);
        let half: Vec<Array2<Complex64>> =
            r.densities.iter().map(|d| d.mapv(|z| z * 0.5)).collect();
        u.density_alpha = half.clone();
        u.density_beta = half;
        let sr = ks_rhf_fit(&ks, &fs, r, exx, None);
        let su = ks_uhf_fit(&ks, &fs, u, exx, None);
        let d = max_mat(&sr.de_deps, &su.de_deps);
        eprintln!("  H2 RS-GDF 1x1x3 {exx:?}: |σ_kUHF(D/2,D/2) − σ_kRHF(D)| {d:.2e}");
        assert!(d <= IDENTITY_BAR, "{exx:?}: {d:e}");
    }
}

/// H3 doublet 1x1x3: the supercell (9 atoms, multiplicity 4) Gamma RS-GDF UHF
/// stress per cell, both exxdiv, and the UHF-specific mutants.
#[test]
#[ignore = "slow (~115 s release, loaded box): 9-atom supercell Gamma RS-GDF UHF stress x2 + k stress"]
fn h3_uhf_1x1x3_rsgdf_stress_is_the_supercell_stress_per_cell() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h3_cell(), &bs, n);
    let fs = fitsys(&ks);
    let scfs = fit_uscfs(&ks, &fs, 2, 1);
    let g = gfit(supercell(&h3_cell(), n, 4), &bs);
    let mut refs = [ZERO; 2];
    for exx in EXX {
        let i = exx_idx(exx);
        let gscf = gamma_unrestricted(
            unfold(&ks.mesh, &scfs[i].density_alpha),
            unfold(&ks.mesh, &scfs[i].density_beta),
        );
        let k = ks_uhf_fit(&ks, &fs, &scfs[i], exx, None);
        let gs = gamma_fit_stress(&g, &gscf, exx);
        refs[i] = scale3(&gs.de_deps, 1.0 / 3.0);
        let miss = max_mat(&k.de_deps, &refs[i]);
        report(&format!("H3 UHF RS-GDF {n:?} {exx:?}"), &k);
        eprintln!("  H3 UHF RS-GDF {n:?} {exx:?}: |dE/dε_k − dE/dε_sc/N| {miss:.2e}");
        assert!(miss <= SC_FIT_BAR, "{exx:?}: {miss:e}");
    }
    // CrossSpinK acts on the dense exchange pairs only: an identity here.
    let s = ks_uhf_fit(
        &ks,
        &fs,
        &scfs[0],
        ExxDiv::None,
        Some(KStressMutation::CrossSpinK),
    );
    assert!(
        max_mat(&s.de_deps, &refs[0]) <= SC_FIT_BAR,
        "CrossSpinK must be an identity"
    );
    for m in [
        KStressMutation::SwapSpinFock,
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
        KStressMutation::FitAuxPhase,
        KStressMutation::FitLrWrongQ,
    ] {
        let s = ks_uhf_fit(&ks, &fs, &scfs[0], ExxDiv::None, Some(m));
        let miss = max_mat(&s.de_deps, &refs[0]);
        eprintln!("  H3 UHF RS-GDF mutant {m:?}: miss {miss:.2e}");
        assert!(miss > MUTANT_BAR, "{m:?} not caught ({miss:e})");
    }
}

// ----------------------------------------------------------- guards

/// Typed refusal of the fitted path: an ECP in the hcore the basis does not
/// carry, on the unsplit AND the range-split source (a range-split build
/// itself is supported). Column-rotation request refusals live in
/// `pbc_sr_rotation_kderiv.rs`.
#[test]
fn rsgdf_stress_refuses_ecp() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let fs = fitsys(&ks);
    let scf = krhf(&ks, fs.jk(&ks, ExxDiv::None));
    let run = |hk: &PeriodicHcoreK, hc: &PeriodicHcoreConfig, src: KRsGdfGradSource<'_>| {
        kpoint_rhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            hc,
            hk,
            KStressJk::RsGdf(src),
            &scf,
            ExxDiv::None,
            &scfg(None),
        )
        .unwrap_err()
        .to_string()
    };
    // ECP in the hcore.
    let mut hk_ecp = ks.hk.clone();
    hk_ecp.v_ecp = Some(hk_ecp.h.clone());
    let e = run(&hk_ecp, &hcore_cfg(), fs.src());
    assert!(e.contains("carries V_ECP but the basis"), "{e}");
    // The same refusal on a range-split source (the SCF density is the
    // unsplit one: the refusal fires before the Fock guard).
    let sp = fitsys_cfg(&ks, split_kcfg(1.0));
    let e = run(&hk_ecp, &hcore_cfg(), sp.src());
    assert!(e.contains("carries V_ECP but the basis"), "{e}");
}

/// The Fock-consistency guard on the fitted source: it must refuse an SCF run
/// at another exxdiv, a fit on another aux basis than the SCF's, and an SCF
/// whose stored Fock was perturbed by 1e-6, and ACCEPT the matching input.
/// Both sides of the bar: matching inputs 1e-15-level (printed), the wrong
/// exxdiv / wrong fit orders of magnitude above `FOCK_CONSISTENCY_BAR`.
#[test]
fn rsgdf_stress_fock_guard_refuses_inconsistent_inputs() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let fs = fitsys(&ks);
    let scf = krhf(&ks, fs.jk(&ks, ExxDiv::None));
    let ok = ks_rhf_fit(&ks, &fs, &scf, ExxDiv::None, None);
    eprintln!(
        "  matching RS-GDF inputs: Fock mismatch {:.2e}",
        ok.fock_mismatch
    );
    assert!(ok.fock_mismatch < 1e-12);
    // Wrong exxdiv (the SCF ran without the Madelung shift).
    let stress = |scf: &KScfResult, fs: &FitSys, exx: ExxDiv| {
        kpoint_rhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &ks.hk,
            KStressJk::RsGdf(fs.src()),
            scf,
            exx,
            &scfg(None),
        )
    };
    let e = stress(&scf, &fs, ExxDiv::Ewald).unwrap_err().to_string();
    eprintln!("  wrong exxdiv: {e}");
    assert!(e.contains("Fock matrix rebuilt"), "{e}");
    // A different fit (another aux basis) than the SCF's. (Another omega is
    // NOT refused, see `rsgdf_stress_accepts_a_fit_at_another_omega`.)
    let other = fitsys_with(&ks, kgdf_cfg(), &coarse_aux());
    let e = stress(&scf, &other, ExxDiv::None).unwrap_err().to_string();
    eprintln!("  wrong aux basis: {e}");
    assert!(e.contains("Fock matrix rebuilt"), "{e}");
    // A stored Fock perturbed by 1e-6 (above the 1e-8 bar).
    let mut bad = scf.clone();
    bad.fock[0][(0, 0)] += Complex64::new(1e-6, 0.0);
    let e = stress(&bad, &fs, ExxDiv::None).unwrap_err().to_string();
    assert!(e.contains("Fock matrix rebuilt"), "{e}");
}

/// PHYSICS: a Monkhorst-Pack even mesh (k = ±b3/4 for N = 2) is a twisted
/// supercell with NO explicit-supercell anchor, so only the FD of ferric's
/// own k energy pins it. It is also the case where the pair-image residue
/// modulus (4) differs from the aux-image one (N = 2): a bin mix-up in either
/// dimension of the (l, t) weights fails here. Non-TRIM (2k ∉ G).
#[test]
#[ignore = "slow (~75 s release on a loaded box): 36 strained RS-GDF builds + 72 k-SCFs (h and 2h)"]
fn h2_mp_1x1x2_rsgdf_stress_matches_fd_of_own_k_energy_all_nine() {
    let c = MeshCentring::MonkhorstPack;
    let (ks, fs, scfs, fd1) = fit_fd_anchor("H2 MP", h2_cell_g(), &pyscf_sto3g_h(), [1, 1, 2], c);
    assert_eq!(ks.mesh.residue_moduli(), [1, 1, 4]);
    for exx in EXX {
        let x = exx_idx(exx);
        let s0 = ks_rhf_fit(&ks, &fs, &scfs[x], exx, None);
        for m in [KStressMutation::NoPhase, KStressMutation::NoInvNk] {
            let mm = max_mat(
                &ks_rhf_fit(&ks, &fs, &scfs[x], exx, Some(m)).de_deps,
                &fd1[x],
            );
            eprintln!("  H2 MP 1x1x2 RS-GDF {exx:?} mutant {m:?}: miss {mm:.2e}");
            assert!(mm > MUTANT_BAR, "{m:?} {exx:?}: {mm:e}");
        }
        // BLIND SPOT, asserted: every q = k' - k of this mesh is a TRIM
        // (2q is a reciprocal-lattice vector), so e^{+iq.T} = e^{-iq.T} and
        // {G + q} = {G - q} up to the sphere edge.
        for m in [KStressMutation::FitAuxPhase, KStressMutation::FitLrWrongQ] {
            let d = max_mat(
                &ks_rhf_fit(&ks, &fs, &scfs[x], exx, Some(m)).de_deps,
                &s0.de_deps,
            );
            eprintln!("  H2 MP 1x1x2 RS-GDF {exx:?} {m:?}: |mut − ref| {d:.2e} (TRIM q: blind)");
            assert!(d <= 1e-9, "{m:?} {exx:?} should be blind on TRIM q ({d:e})");
        }
    }
}

/// A triclinic lattice (TRI_A: no cell vector along a Cartesian axis, so a
/// Cartesian/fractional index mix-up in the K sets, residue bins or q phases
/// cannot cancel as it can in a cubic cell) with two s-only hydrogens, 1x1x3.
#[test]
#[ignore = "slow (~40 s release): triclinic H2 1x1x3 RS-GDF, 36 strained builds + 72 k-SCFs"]
fn triclinic_h2_1x1x3_rsgdf_stress_matches_fd_of_own_k_energy_all_nine() {
    let cell = cell_at(&[TRI_ATOMS[0], TRI_ATOMS[1]], TRI_A, 1);
    let (ks, fs, scfs, fd1) = fit_fd_anchor(
        "tri H2",
        cell,
        &pyscf_sto3g_h(),
        [1, 1, 3],
        MeshCentring::Gamma,
    );
    for exx in EXX {
        let x = exx_idx(exx);
        for m in [KStressMutation::NoPhase, KStressMutation::FitLrWrongQ] {
            let mm = max_mat(
                &ks_rhf_fit(&ks, &fs, &scfs[x], exx, Some(m)).de_deps,
                &fd1[x],
            );
            eprintln!("  tri H2 RS-GDF {exx:?} mutant {m:?}: miss {mm:.2e}");
            assert!(mm > MUTANT_BAR, "{m:?} {exx:?}: {mm:e}");
        }
    }
}

/// An ACTIVE metric lindep cut with a NEAR-dependent dropped mode: every atom
/// carries s(0.75) and s(0.76) aux shells, so each q class has one small but
/// NONZERO metric eigenvalue per atom (asserted below the cut, the rest above
/// it). An exact duplicate would make the dropped mode an exact null vector of
/// `J3`, multiplying the kept-dropped Loewner weight by zero; here that branch
/// carries weight (at the ~1e-8 level, see below), and `FitNoLoewnerKd` (kept-dropped entries zeroed) must miss
/// the supercell anchor. The Gamma supercell side drops the same modes.
#[test]
#[ignore = "slow (~50 s release): supercell anchor with a near-duplicate aux shell"]
fn h2_1x1x3_rsgdf_stress_with_an_active_lindep_cut_is_the_supercell_stress() {
    let mut aux = et_sp_aux();
    let mut near = aux.shells[&1][2].clone();
    near.exponents = vec![0.95];
    aux.shells.get_mut(&1).unwrap().push(near);
    let gdf = RsGdfConfig {
        lindep: NEAR_DUP_LINDEP,
        ..gdf_cfg()
    };
    let a = fit_supercell_anchor_cfg(
        "H2 near-dup aux",
        h2_cell_g(),
        &pyscf_sto3g_h(),
        &aux,
        &gdf,
        [1, 1, 3],
    );
    let per_q = &a.fs.gdf.stats().per_q;
    for q in per_q {
        eprintln!(
            "  q class {}: dropped {} (min eig {:.2e}), kept {}",
            q.iq, q.n_dropped, q.metric_eig_min, q.naux_kept
        );
        assert_eq!(q.n_dropped, 2, "one near-dependent mode per atom");
        assert!(
            q.metric_eig_min > 1e-14 && q.metric_eig_min < NEAR_DUP_LINDEP,
            "the dropped mode must be small but NONZERO"
        );
    }
    for m in [
        KStressMutation::FitNoLoewnerKd,
        KStressMutation::NoMetric,
        KStressMutation::NoPhase,
    ] {
        let x = a.mutant(ExxDiv::None, m);
        eprintln!(
            "  H2 near-dup aux mutant {m:?}: miss {:.2e} (worst part {:.2e})",
            x.total, x.part
        );
        // The kept-dropped weight multiplies the dropped component of H, which
        // is small for a near-dependent mode, so this defect is inherently a
        // ~1e-8 effect on the total (measured 3.6e-8; worst part 3.2e-6) -
        // far above the 1e-12 anchor floor but below MUTANT_BAR.
        let bar = if m == KStressMutation::FitNoLoewnerKd {
            100.0 * SC_FIT_BAR
        } else {
            MUTANT_BAR
        };
        assert!(x.total > bar, "{m:?} not caught ({:e})", x.total);
    }
}

/// PHYSICS/ARTIFACT as the main supercell anchor, with a pure-d aux shell so
/// the l = 2 aux-FT strain (`aux_ft_strain_shells`, pure cart2sph) and the l = 2
/// SR derivative blocks are on the production path at k. `NoAuxFt` must be a
/// sharp per-part miss here too.
#[test]
#[ignore = "slow (~45 s release): supercell anchor with a pure-d aux shell"]
fn h2_1x1x3_rsgdf_stress_with_a_d_aux_is_the_supercell_stress() {
    let a = fit_supercell_anchor_aux(
        "H2 spd-aux",
        h2_cell_g(),
        &pyscf_sto3g_h(),
        &et_spd_aux(),
        [1, 1, 3],
    );
    for m in [KStressMutation::NoAuxFt, KStressMutation::NoAuxFt3] {
        let x = a.mutant(ExxDiv::None, m);
        eprintln!(
            "  H2 spd-aux mutant {m:?}: total miss {:.2e}, worst part {:.2e}",
            x.total, x.part
        );
        assert!(
            x.part > MUTANT_BAR,
            "{m:?} not caught per part ({:e})",
            x.part
        );
    }
}

/// The energy is omega-independent (the fit of the same aux basis, SR + LR
/// split at another omega), so an SCF run at omega 1.0 and a stress with a fit
/// at omega 1.3 is NOT inconsistent: the guard must accept it (mismatch far
/// below the bar) and the stress must agree with the omega-1.0 stress to the
/// measured level (printed; the two splits differ by the SR/LR truncation).
#[test]
fn rsgdf_stress_accepts_a_fit_at_another_omega() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let fs = fitsys(&ks);
    let scf = krhf(&ks, fs.jk(&ks, ExxDiv::None));
    let base = ks_rhf_fit(&ks, &fs, &scf, ExxDiv::None, None);
    let other = fitsys_cfg(
        &ks,
        KRsGdfConfig {
            gdf: RsGdfConfig {
                omega: 1.3,
                ..gdf_cfg()
            },
            mutation: None,
        },
    );
    let s = ks_rhf_fit(&ks, &other, &scf, ExxDiv::None, None);
    let d = max_mat(&s.de_deps, &base.de_deps);
    eprintln!(
        "  omega 1.3 fit on an omega 1.0 SCF: Fock mismatch {:.2e}, |dE/dε − dE/dε(omega 1.0)| {d:.2e}",
        s.fock_mismatch
    );
    assert!(s.fock_mismatch < FOCK_CONSISTENCY_BAR);
    assert!(d <= OMEGA_STRESS_BAR, "{d:e}");
}

/// Auto column rotation: the energy builds and the stress walks rotate the SR
/// columns (`Auto`) and agree with the unrotated pipeline to the screening
/// precision. On a p-bearing, GENERALLY CONTRACTED k cell (cc-pVDZ / cc-pvdz-ri H2; a
/// segmented basis such as STO-3G or the s+p test basis has no rotating
/// column, so Auto is bit-identical to Off there), the stress
/// of a fully-Auto pipeline (hcore, fit, SCF) vs the fully-Off one is printed
/// together with the energy difference (zero energy difference would mean Auto
/// resolved to off here).
#[test]
#[ignore = "slow (~80 s release, loaded box): H2 cc-pVDZ 1x1x2 RS-GDF, two full pipelines"]
fn rsgdf_stress_under_auto_rotation_agrees_with_off_on_a_p_cell() {
    let bs = ferric_core::basis::bundled("cc-pvdz").unwrap();
    let aux_bs = ferric_core::basis::bundled("cc-pvdz-ri").unwrap();
    let n = [1, 1, 2];
    let run = |auto: bool| {
        let cell = h2_cell_g();
        let prep = prep_for(&cell, &bs);
        let mesh = KPointMesh::new(&cell, n, MeshCentring::Gamma).unwrap();
        let hc = if auto {
            hcore_cfg().with_sr_column_rotation(SrColumnRotation::Auto)
        } else {
            hcore_cfg()
        };
        let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hc).unwrap();
        let ks = KSys {
            cell,
            prep,
            mesh,
            hk,
        };
        let cfg = KRsGdfConfig {
            gdf: RsGdfConfig {
                sr_column_rotation: if auto {
                    SrColumnRotation::Auto
                } else {
                    SrColumnRotation::Off
                },
                ..gdf_cfg()
            },
            mutation: None,
        };
        let fs = fitsys_with(&ks, cfg, &aux_bs);
        let scf = krhf(&ks, fs.jk(&ks, ExxDiv::None));
        let st = kpoint_rhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hc,
            &ks.hk,
            KStressJk::RsGdf(fs.src()),
            &scf,
            ExxDiv::None,
            &scfg(None),
        )
        .expect("stress");
        // The stress walked the columns the energy rotated (nonzero under Auto).
        assert_eq!(st.hcore_rotated_columns, ks.hk.sr_rotated_columns);
        assert_eq!(st.hcore_rotated_columns > 0, auto);
        (scf.energy, st)
    };
    let (e_auto, s_auto) = run(true);
    let (e_off, s_off) = run(false);
    let d = max_mat(&s_auto.de_deps, &s_off.de_deps);
    eprintln!(
        "  H2 cc-pVDZ 1x1x2 RS-GDF: |E(Auto) − E(Off)| {:.2e}, |dE/dε(Auto) − dE/dε(Off)| {d:.2e}",
        (e_auto - e_off).abs()
    );
    assert!(d <= AUTO_ROT_BAR, "{d:e}");
}

/// PHYSICS: analytic UHF `dE/dε` on the fitted path vs central FD (h = 1e-4,
/// fresh fit and SCF per point, all nine components, exxdiv none) of the own
/// k-UHF RS-GDF energy: a spin-specific anchor that does not rest on the
/// supercell. The fit-path spin mutant `FitCrossSpin` (exchange pairing
/// D_α J3^H D_β) and `SwapSpinFock` must miss it.
#[test]
#[ignore = "slow (~50 s release): 18 strained RS-GDF builds + 18 k-UHF SCFs on H3"]
fn h3_uhf_1x1x3_rsgdf_stress_matches_fd_of_own_k_energy_all_nine() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h3_cell(), &bs, n);
    let fs = fitsys(&ks);
    let scf = kuhf(&ks, fs.jk(&ks, ExxDiv::None), 2, 1);
    let kept0: Vec<usize> = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    let mut fd = ZERO;
    for (i, j) in ALL9 {
        let at = |h: f64| {
            let (e, kept) = k_fit_energy_uhf(&strained(&ks.cell, i, j, h), &bs, n);
            assert_eq!(kept, kept0, "the metric cut changed under strain");
            e
        };
        fd[i][j] = (at(FD_H) - at(-FD_H)) / (2.0 * FD_H);
    }
    let s = ks_uhf_fit(&ks, &fs, &scf, ExxDiv::None, None);
    let miss = max_mat(&s.de_deps, &fd);
    eprintln!("  H3 UHF RS-GDF 1x1x3: |analytic − FD| {miss:.2e}");
    assert!(miss <= K_FD_BAR_FIT, "{miss:e}");
    for m in [KStressMutation::FitCrossSpin, KStressMutation::SwapSpinFock] {
        let mm = max_mat(
            &ks_uhf_fit(&ks, &fs, &scf, ExxDiv::None, Some(m)).de_deps,
            &fd,
        );
        eprintln!("  H3 UHF RS-GDF mutant {m:?}: |mut − FD| {mm:.2e}");
        assert!(mm > MUTANT_BAR, "{m:?} not caught ({mm:e})");
    }
}

/// PHYSICS: at 1x1x1 the k split stress is the Gamma split stress, part by
/// part (SR3, SR2, LR3, LR2, G0 overlap incl. the smooth-overlap virial, both
/// G0 volumes). ARTIFACT: independent constructions (half sphere real with
/// weight 2 vs full sphere complex), but BLIND to phases, 1/N_k and q != 0.
#[test]
fn one_point_mesh_split_rsgdf_stress_is_the_gamma_split_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 1]);
    let fs = fitsys_cfg(&ks, split_kcfg(1.0));
    let un = fitsys(&ks);
    assert_moved("H2 1x1x1", &fs.gdf, &un.gdf);
    let scfs = fit_scfs(&ks, &fs);
    let g = gfit_aux(h2_cell_g(), &bs, &et_sp_aux(), &split_gdf_cfg(1.0));
    for exx in EXX {
        let scf = &scfs[exx_idx(exx)];
        let gscf = gamma_restricted(real_of(&scf.densities[0]));
        let k = ks_rhf_fit(&ks, &fs, scf, exx, None);
        let gs = gamma_fit_stress(&g, &gscf, exx);
        let total = max_mat(&k.de_deps, &gs.de_deps);
        eprintln!("  H2 split 1x1x1 {exx:?}: |dE/dε k − Gamma| {total:.2e}");
        for ((name, kp), gp) in fit_parts(&k.parts).iter().zip(gamma_fit_parts(&gs.parts)) {
            let d = max_mat(kp, &gp);
            eprintln!(
                "    {name}: |k − Gamma| {d:.2e} (|part| {:.2e})",
                max_mat(kp, &ZERO)
            );
            assert!(d <= GAMMA_FIT_BAR, "{exx:?} {name}: {d:e}");
        }
        assert!(total <= GAMMA_FIT_BAR, "{exx:?}: {total:e}");
        if exx == ExxDiv::None {
            for m in SPLIT_MUTANTS {
                let x = ks_rhf_fit(&ks, &fs, scf, exx, Some(m));
                let miss = max_mat(&x.de_deps, &gs.de_deps);
                eprintln!("  1x1x1 split mutant {m:?}: miss vs Gamma {miss:.2e}");
                assert!(miss > MUTANT_BAR, "{m:?} not caught ({miss:e})");
            }
        }
    }
}

// ----------------------------------------------- supercell anchor (split)

/// PHYSICS: a Gamma-centred k mesh IS the diag(N) supercell, so per cell
/// `dE/dε = dE/dε_sc/N`, here with the RANGE-SPLIT fit on both sides (the
/// k-point kernel v_SR(|G + q|) per q vs the supercell's Gamma kernel).
/// ARTIFACT: a split stress that used the Gamma kernel at every q (the
/// k-specific defect of the energy, 1.05e-3 Ha) or mis-binned the smooth
/// pairs could not reproduce an explicit supercell; the 1x1x1 anchor cannot
/// see either (q = 0 only). NON-TRIM mesh (n_2 = 3).
#[test]
fn h2_1x1x3_split_rsgdf_stress_is_the_supercell_stress_per_cell() {
    let bs = pyscf_sto3g_h();
    let a = fit_supercell_anchor_cfg(
        "H2 split",
        h2_cell_g(),
        &bs,
        &et_sp_aux(),
        &split_gdf_cfg(1.0),
        [1, 1, 3],
    );
    assert_moved("H2 1x1x3", &a.fs.gdf, &fitsys(&a.ks).gdf);
    for m in [
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
        KStressMutation::FitAuxPhase,
        KStressMutation::FitLrWrongQ,
        KStressMutation::SplitNoSrKernelStrain,
        KStressMutation::SplitNoSmoothPair,
    ] {
        let x = a.mutant(ExxDiv::None, m);
        eprintln!(
            "  H2 split mutant {m:?}: miss {:.2e} (off-diagonal {:.2e}, worst part {:.2e})",
            x.total, x.off, x.part
        );
        assert!(x.total > MUTANT_BAR, "{m:?} not caught ({:e})", x.total);
    }
}

/// Every NEW split hook against the supercell anchor, both exxdiv, per part
/// (a part-level defect hidden in the total by aux stationarity is sharp
/// per part). `SplitNoSmoothVolume` and `SplitFullG0` carry diagonal volume
/// terms: asserted diagonal-only for the volume one.
#[test]
#[ignore = "slow (~90 s release, 55 s on a quiet box): 26 stress evaluations after the supercell anchor"]
fn h2_1x1x3_split_mutants_miss_the_supercell_anchor() {
    let bs = pyscf_sto3g_h();
    let a = fit_supercell_anchor_cfg(
        "H2 split",
        h2_cell_g(),
        &bs,
        &et_sp_aux(),
        &split_gdf_cfg(1.0),
        [1, 1, 3],
    );
    for m in SPLIT_MUTANTS.into_iter().chain([
        KStressMutation::NoInvNk,
        KStressMutation::NoMetric,
        KStressMutation::NoAuxFt3,
        KStressMutation::NoJ2G0Vol,
        KStressMutation::NoJ3G0Vol,
        KStressMutation::NoG0,
        KStressMutation::NoVolume,
    ]) {
        for exx in EXX {
            let x = a.mutant(exx, m);
            eprintln!(
                "  H2 split mutant {m:?} {exx:?}: miss {:.2e} (off-diagonal {:.2e}, worst part {:.2e})",
                x.total, x.off, x.part
            );
            assert!(
                x.total > MUTANT_BAR,
                "{m:?} {exx:?} not caught ({:e})",
                x.total
            );
            // NoInvNk acts on the one-electron overlap weights, not on the
            // seven fitted parts (its part miss is the 1e-12 floor).
            if m != KStressMutation::NoInvNk {
                assert!(x.part > MUTANT_BAR, "{m:?} {exx:?}: parts ({:e})", x.part);
            }
            if m == KStressMutation::SplitNoSmoothVolume {
                assert!(x.off <= SC_FIT_BAR, "{m:?} diagonal-only ({:e})", x.off);
            }
        }
    }
}

/// PHYSICS: the split is an exact decomposition of the same fitted energy, so
/// the split k stress equals the UNSPLIT k stress up to the split's
/// own tolerance (each side at its own converged density). ARTIFACT: equality
/// to 1e-13 would mean the split ran the unsplit path (the moved counters are
/// asserted); the difference is the split's truncation, MEASURED here.
#[test]
fn h2_1x1x3_split_k_stress_matches_the_unsplit_k_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let sp = fitsys_cfg(&ks, split_kcfg(1.0));
    let un = fitsys(&ks);
    assert_moved("H2 1x1x3", &sp.gdf, &un.gdf);
    let (scf_s, scf_u) = (fit_scfs(&ks, &sp), fit_scfs(&ks, &un));
    for exx in EXX {
        let i = exx_idx(exx);
        let s = ks_rhf_fit(&ks, &sp, &scf_s[i], exx, None);
        let u = ks_rhf_fit(&ks, &un, &scf_u[i], exx, None);
        let d = max_mat(&s.de_deps, &u.de_deps);
        eprintln!(
            "  H2 1x1x3 {exx:?}: |split − unsplit| dE/dε {d:.2e}, energies {:.3e}",
            (scf_s[i].energy - scf_u[i].energy).abs()
        );
        assert!(d <= SPLIT_UNSPLIT_BAR, "{exx:?}: {d:e}");
        assert!(d > 1e-15, "{exx:?}: split ran the unsplit path ({d:e})");
    }
}

/// PHYSICS: analytic split `dE/dε` vs central FD of ferric's own split k
/// energy (fresh split build + SCF at every strained point; lattice and atoms
/// strained, index sets frozen, k at fixed fractional coordinates). ARTIFACT:
/// the split's partition is geometry-dependent (primitive criteria are
/// strain-invariant, the pair screens are not); FD(h) vs FD(2h) must stay at
/// h² (asserted) or the split energy is piecewise under strain.
#[test]
#[ignore = "slow (~37 s release): 36 strained split RS-GDF builds + 72 k-SCFs (h and 2h)"]
fn h2_1x1x3_split_stress_matches_fd_of_own_k_energy_all_nine() {
    let (ks, fs, scfs, fd1) = split_fd_anchor(
        "H2",
        h2_cell_g(),
        &pyscf_sto3g_h(),
        [1, 1, 3],
        MeshCentring::Gamma,
    );
    for m in SPLIT_MUTANTS {
        let s = ks_rhf_fit(&ks, &fs, &scfs[0], ExxDiv::None, Some(m));
        let mm = max_mat(&s.de_deps, &fd1[0]);
        eprintln!("  H2 split FD mutant {m:?}: |mut − FD| {mm:.2e}");
        assert!(mm > MUTANT_BAR, "{m:?} not caught by FD ({mm:e})");
    }
}

/// An even MP mesh (twisted supercell, no supercell anchor; pair-image modulus
/// 4 vs aux modulus 2) with the split: only the FD pins it.
#[test]
#[ignore = "slow (~16 s release): 36 strained split RS-GDF builds + 72 k-SCFs (h and 2h)"]
fn h2_mp_1x1x2_split_stress_matches_fd_of_own_k_energy_all_nine() {
    let (ks, ..) = split_fd_anchor(
        "H2 MP",
        h2_cell_g(),
        &pyscf_sto3g_h(),
        [1, 1, 2],
        MeshCentring::MonkhorstPack,
    );
    assert_eq!(ks.mesh.residue_moduli(), [1, 1, 4]);
}

// ------------------------------------------------------------ split UHF

/// 1x1x1 H3 doublet: the k split UHF stress is the Gamma split UHF stress.
#[test]
fn one_point_mesh_split_rsgdf_uhf_stress_is_the_gamma_split_stress() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h3_cell(), &bs, [1, 1, 1]);
    let fs = fitsys_cfg(&ks, split_kcfg(1.0));
    assert_moved("H3 1x1x1", &fs.gdf, &fitsys(&ks).gdf);
    let scfs = fit_uscfs(&ks, &fs, 2, 1);
    let g = gfit_aux(h3_cell(), &bs, &et_sp_aux(), &split_gdf_cfg(1.0));
    for exx in EXX {
        let scf = &scfs[exx_idx(exx)];
        let gscf = gamma_unrestricted(
            real_of(&scf.density_alpha[0]),
            real_of(&scf.density_beta[0]),
        );
        let k = ks_uhf_fit(&ks, &fs, scf, exx, None);
        let gs = gamma_fit_stress(&g, &gscf, exx);
        let total = max_mat(&k.de_deps, &gs.de_deps);
        eprintln!("  H3 UHF split 1x1x1 {exx:?}: |dE/dε k − Gamma| {total:.2e}");
        for ((name, kp), gp) in fit_parts(&k.parts).iter().zip(gamma_fit_parts(&gs.parts)) {
            let d = max_mat(kp, &gp);
            eprintln!("    {name}: |k − Gamma| {d:.2e}");
            assert!(d <= GAMMA_FIT_BAR, "{exx:?} {name}: {d:e}");
        }
        assert!(total <= GAMMA_FIT_BAR, "{exx:?}: {total:e}");
    }
}

/// H3 doublet 1x1x3: the split supercell (9 atoms, multiplicity 4) Gamma UHF
/// stress per cell, both exxdiv, per part, and the split + spin mutants.
#[test]
#[ignore = "slow (~98 s release): 9-atom split supercell Gamma RS-GDF UHF stress x2 + k stress"]
fn h3_uhf_1x1x3_split_stress_is_the_supercell_stress_per_cell() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h3_cell(), &bs, n);
    let fs = fitsys_cfg(&ks, split_kcfg(1.0));
    assert_moved("H3 1x1x3", &fs.gdf, &fitsys(&ks).gdf);
    let scfs = fit_uscfs(&ks, &fs, 2, 1);
    let g = gfit_aux(
        supercell(&h3_cell(), n, 4),
        &bs,
        &et_sp_aux(),
        &split_gdf_cfg(1.0),
    );
    let mut refs = [ZERO; 2];
    let mut ref_parts = [[ZERO; 7]; 2];
    for exx in EXX {
        let i = exx_idx(exx);
        let gscf = gamma_unrestricted(
            unfold(&ks.mesh, &scfs[i].density_alpha),
            unfold(&ks.mesh, &scfs[i].density_beta),
        );
        let k = ks_uhf_fit(&ks, &fs, &scfs[i], exx, None);
        let gs = gamma_fit_stress(&g, &gscf, exx);
        refs[i] = scale3(&gs.de_deps, 1.0 / 3.0);
        let miss = max_mat(&k.de_deps, &refs[i]);
        eprintln!("  H3 UHF split {n:?} {exx:?}: |dE/dε_k − dE/dε_sc/N| {miss:.2e}");
        for (pi, ((pn, kp), gp)) in fit_parts(&k.parts)
            .iter()
            .zip(gamma_fit_parts(&gs.parts))
            .enumerate()
        {
            ref_parts[i][pi] = scale3(&gp, 1.0 / 3.0);
            let d = max_mat(kp, &ref_parts[i][pi]);
            eprintln!("    {pn}: |k − sc/N| {d:.2e}");
            assert!(d <= SC_FIT_BAR, "{exx:?} {pn}: {d:e}");
        }
        assert!(miss <= SC_FIT_BAR, "{exx:?}: {miss:e}");
    }
    for m in SPLIT_MUTANTS.into_iter().chain([
        KStressMutation::SwapSpinFock,
        KStressMutation::FitCrossSpin,
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
    ]) {
        let s = ks_uhf_fit(&ks, &fs, &scfs[0], ExxDiv::None, Some(m));
        let miss = max_mat(&s.de_deps, &refs[0]);
        eprintln!("  H3 UHF split mutant {m:?}: miss {miss:.2e}");
        assert!(miss > MUTANT_BAR, "{m:?} not caught ({miss:e})");
    }
}

/// A pure-d aux shell (l = 2) with the split, 1x1x3 supercell anchor, both
/// exxdiv, per part. MEASURED at the default `lindep`: total 7.4e-12 / 7.2e-12
/// but the two J2 parts (SR and LR) are 7.7e-11..1.1e-10 and cancel in the
/// total (unsplit d-aux J2 parts: 1.7e-12..2.5e-12); the floor does not move
/// with the build precision. With the single near-null metric direction
/// (eigenvalue 2.1e-4) dropped on both sides (`lindep` 1e-3) every part is
/// <= `SC_FIT_BAR` (2.6e-12, see
/// `d_aux_part_floor_collapses_when_the_near_null_mode_is_dropped`): the floor
/// is the fit's sensitivity along that mode, not a defect. Both are asserted.
#[test]
#[ignore = "slow (~35 s release): two supercell anchors with a pure-d aux shell and the split"]
fn h2_1x1x3_split_stress_with_a_d_aux_is_the_supercell_stress() {
    let split = split_gdf_cfg(1.0);
    let bs = pyscf_sto3g_h();
    let _ = fit_supercell_anchor_bars(
        "H2 split d-aux",
        h2_cell_g(),
        &bs,
        &et_spd_aux(),
        &split,
        [1, 1, 3],
        5e-10,
    );
    let _ = fit_supercell_anchor_cfg(
        "H2 split d-aux lindep 1e-3",
        h2_cell_g(),
        &bs,
        &et_spd_aux(),
        &RsGdfConfig {
            lindep: 1e-3,
            ..split
        },
        [1, 1, 3],
    );
}

// ----------------------------------------------------------------- guard

/// The Fock-consistency guard on a SPLIT source: accepts the matching SCF,
/// refuses the wrong exxdiv and a perturbed Fock. The split SCF against the
/// UNSPLIT source is NOT inconsistent (an exact decomposition): the printed
/// mismatch is the split tolerance, far below the 1e-8 bar.
#[test]
fn split_stress_fock_guard_accepts_matching_and_refuses_inconsistent_inputs() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let sp = fitsys_cfg(&ks, split_kcfg(1.0));
    let un = fitsys(&ks);
    let scf = krhf(&ks, sp.jk(&ks, ExxDiv::None));
    let ok = ks_rhf_fit(&ks, &sp, &scf, ExxDiv::None, None);
    eprintln!(
        "  matching split inputs: Fock mismatch {:.2e}",
        ok.fock_mismatch
    );
    assert!(ok.fock_mismatch < 1e-12);
    let cross = ks_rhf_fit(&ks, &un, &scf, ExxDiv::None, None);
    eprintln!(
        "  split SCF vs unsplit source: Fock mismatch {:.2e}",
        cross.fock_mismatch
    );
    assert!(cross.fock_mismatch < FOCK_CONSISTENCY_BAR);
    let stress = |scf: &KScfResult, exx: ExxDiv| {
        kpoint_rhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &ks.hk,
            KStressJk::RsGdf(sp.src()),
            scf,
            exx,
            &scfg(None),
        )
    };
    let e = stress(&scf, ExxDiv::Ewald).unwrap_err().to_string();
    assert!(e.contains("Fock matrix rebuilt"), "{e}");
    let mut bad = scf.clone();
    bad.fock[0][(0, 0)] += Complex64::new(1e-6, 0.0);
    let e = stress(&bad, ExxDiv::None).unwrap_err().to_string();
    assert!(e.contains("Fock matrix rebuilt"), "{e}");
}

/// The pure-d aux PART floor of the split supercell anchor (totals agree to
/// 7e-12; the J2 SR / J2 LR parts differ by 1e-10 and cancel). The smallest
/// metric eigenvalue is 2.1e-4 (> `lindep`, so nothing is dropped by default).
/// MEASURED discriminators (release, H2 1x1x3): the floor does NOT move with
/// the build PRECISION (1.1e-10 at 1e-13, 5.1e-11 at 1e-15, 1.05e-10 at 1e-17,
/// so it is not a truncation floor), is 4.4e-11 with a COMPACT (unmoved) d
/// exponent 1.6 and 2.9e-12 unsplit; but dropping the single near-null metric
/// direction (`lindep` 1e-3, on BOTH sides) collapses it (asserted below). A
/// defect independent of that one mode would not move; this floor is the
/// fit's sensitivity along it, amplified by the split's larger SR/LR parts.
#[test]
#[ignore = "slow (~75 s release): 3 supercell anchors on H2 1x1x3"]
fn d_aux_part_floor_collapses_when_the_near_null_mode_is_dropped() {
    let split = split_gdf_cfg(1.0);
    let run = |tag: &str, g: &RsGdfConfig| {
        let r = d_aux_floor(&et_sp_d_aux(0.8), g);
        eprintln!(
            "  [{tag}] worst part {:.2e}, worst total {:.2e}, min metric eig {:.2e}",
            r.0, r.1, r.2
        );
        r
    };
    let u = run("unsplit lindep 1e-10", &gdf_cfg());
    let s = run("split   lindep 1e-10", &split);
    let l = run(
        "split   lindep 1e-3 ",
        &RsGdfConfig {
            lindep: 1e-3,
            ..split
        },
    );
    assert!(s.0 > 10.0 * u.0, "the split carries the larger part floor");
    assert!(
        l.0 < 0.1 * s.0,
        "floor must collapse when the mode is dropped"
    );
    assert!(l.0 <= SC_FIT_BAR, "{:e}", l.0);
}

/// 1x1x1 d-aux split vs the Gamma split, part by part (q = 0 only: the sphere
/// is the same set on both sides, so edge-set differences cannot enter).
/// MEASURED at the default `lindep`: total <= 4.6e-13 but the J2 SR / LR parts
/// differ by 8e-12 / 1.1e-11 (ET-sp aux without d: <= 1e-13), cancelling; with
/// the single near-null metric direction (eigenvalue 2.1e-4) dropped by
/// `lindep` = 1e-3 on both sides every part collapses to <= 4.3e-14, so the
/// floor is that mode's sensitivity, not a defect. Both are asserted.
#[test]
fn one_point_mesh_split_d_aux_stress_is_the_gamma_split_stress() {
    let (part, total, dropped, emin) = one_point_part_floor(&et_spd_aux(), &split_gdf_cfg(1.0));
    eprintln!(
        "  d-aux 1x1x1 default lindep: part {part:.2e}, total {total:.2e}, min eig {emin:.2e}"
    );
    assert_eq!(dropped, 0);
    assert!(total <= GAMMA_FIT_BAR, "{total:e}");
    assert!(part <= 5e-11, "{part:e}");
    let g = RsGdfConfig {
        lindep: 1e-3,
        ..split_gdf_cfg(1.0)
    };
    let (part, total, dropped, _) = one_point_part_floor(&et_spd_aux(), &g);
    eprintln!("  d-aux 1x1x1 lindep 1e-3 (dropped {dropped}): part {part:.2e}, total {total:.2e}");
    assert_eq!(dropped, 1);
    assert!(
        total <= GAMMA_FIT_BAR && part <= GAMMA_FIT_BAR,
        "{part:e} {total:e}"
    );
}

/// The split is an exact decomposition at ANY omega and lambda <= 1: the
/// split k stress equals the unsplit one at omega 1.3, lambda 0.7 (the
/// omega-1 / lambda-1 test cannot see an omega or lambda that is hard-wired
/// somewhere in the new passes). Measured 3.6e-12 / 3.7e-12 (none / ewald).
#[test]
#[ignore = "slow-ish (~17 s release): 2 split + 2 unsplit builds and 4 SCFs at omega 1.3"]
fn h2_1x1x3_split_k_stress_matches_the_unsplit_one_at_omega_1p3_lambda_0p7() {
    let at = |split: Option<RangeSplit>| RsGdfConfig {
        omega: 1.3,
        range_split: split,
        ..gdf_cfg()
    };
    let (d, fs) = split_vs_unsplit(&et_sp_aux(), &at(Some(RangeSplit::new(0.7))), &at(None));
    let (o, a) = moved_counts(&fs.gdf);
    eprintln!(
        "  omega 1.3, lambda 0.7: moved orbital {o}, aux {a}; |split − unsplit| {:?}",
        d
    );
    assert!(o > 0 && a > 0, "nothing moved");
    for x in d {
        assert!(x <= SPLIT_UNSPLIT_BAR, "{x:e}");
        assert!(x > 1e-15, "split ran the unsplit path ({x:e})");
    }
}

/// The two partial-move branches of the LR strain dispatch. AUX moves, no
/// orbital does (ET-sp aux, lambda 0.32: aux s(0.3) <= 0.32 but the orbital
/// primitive 0.1689 > 0.16): the moved pass runs without a smooth-pair pass.
/// ORBITALS move, aux does not (aux s(1.9, 0.5), lambda 0.4: orbital 0.1689
/// <= 0.2 but the smallest aux exponent 0.5 > 0.4): the unsplit LR pass plus
/// the smooth-pair pass with `Xc = X`. Each agrees with the unsplit stress.
#[test]
fn split_k_stress_partial_move_branches_match_the_unsplit_stress() {
    let split = |l: f64| Some(RangeSplit::new(l));
    let at = |s: Option<RangeSplit>| RsGdfConfig {
        range_split: s,
        ..gdf_cfg()
    };
    let (d, fs) = split_vs_unsplit(&et_sp_aux(), &at(split(0.32)), &at(None));
    let (o, a) = moved_counts(&fs.gdf);
    eprintln!(
        "  aux-only move (lambda 0.32): orbital {o}, aux {a}; diff {:?}",
        d
    );
    assert_eq!(o, 0, "an orbital primitive moved");
    assert!(a > 0, "no aux primitive moved");
    for x in d {
        assert!(x <= SPLIT_UNSPLIT_BAR, "aux-only: {x:e}");
        assert!(x > 1e-15, "aux-only: split ran the unsplit path ({x:e})");
    }
    let (d, fs) = split_vs_unsplit(&coarse_aux(), &at(split(0.4)), &at(None));
    let (o, a) = moved_counts(&fs.gdf);
    eprintln!(
        "  orbital-only move (lambda 0.4): orbital {o}, aux {a}; diff {:?}",
        d
    );
    assert!(o > 0, "no orbital primitive moved");
    assert_eq!(a, 0, "an aux primitive moved");
    for x in d {
        assert!(x <= SPLIT_UNSPLIT_BAR, "orbital-only: {x:e}");
        assert!(
            x > 1e-15,
            "orbital-only: split ran the unsplit path ({x:e})"
        );
    }
}

/// TRIVIAL LIMIT: lambda = 0 builds the split plan but moves nothing, so the
/// k stress (every part and the total) is the unsplit one BIT FOR BIT (the
/// LR strain is the unsplit pass, every libint factor is 1, `q_c` is `q`, no
/// smooth overlap).
#[test]
fn nothing_moved_split_k_stress_is_the_unsplit_stress_bitwise() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let un = fitsys(&ks);
    let sp = fitsys_cfg(&ks, split_kcfg(0.0));
    assert_eq!(moved_counts(&sp.gdf), (0, 0), "lambda = 0 moved something");
    let scfs = fit_scfs(&ks, &un);
    for exx in EXX {
        let i = exx_idx(exx);
        let u = ks_rhf_fit(&ks, &un, &scfs[i], exx, None);
        let s = ks_rhf_fit(&ks, &sp, &scfs[i], exx, None);
        let bits = |m: &Mat3| m.iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>();
        let d = max_mat(&s.de_deps, &u.de_deps);
        eprintln!("  lambda 0 {exx:?}: |split − unsplit| {d:.2e}");
        assert_eq!(bits(&s.de_deps), bits(&u.de_deps), "{exx:?} total ({d:e})");
        for ((n, a), (_, b)) in fit_parts(&s.parts).iter().zip(fit_parts(&u.parts)) {
            assert_eq!(bits(a), bits(&b), "{exx:?} {n}");
        }
    }
}

/// Typed refusals of the split source: a build and a config that disagree
/// about the split (either way), and lambda > 1.
#[test]
fn split_stress_refuses_a_mismatched_config_and_lambda_above_one() {
    let bs = pyscf_sto3g_h();
    let ks = ksys(h2_cell_g(), &bs, [1, 1, 3]);
    let sp = fitsys_cfg(&ks, split_kcfg(1.0));
    let un = fitsys(&ks);
    let scf = krhf(&ks, un.jk(&ks, ExxDiv::None));
    let run = |gdf: &KRsGdf, cfg: &KRsGdfConfig, aux: &PreparedBasis| {
        kpoint_rhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &ks.hk,
            KStressJk::RsGdf(KRsGdfGradSource { gdf, cfg, aux }),
            &scf,
            ExxDiv::None,
            &scfg(None),
        )
    };
    // Split build, unsplit config; unsplit build, split config.
    let e = run(&sp.gdf, &kgdf_cfg(), &sp.aux).unwrap_err().to_string();
    assert!(
        e.contains("ran the range split but the config does not"),
        "{e}"
    );
    let e = run(&un.gdf, &split_kcfg(1.0), &un.aux)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("did not run the range split but the config requests"),
        "{e}"
    );
    // Matching split build and config is accepted (an SCF on the split build
    // is not needed: the refusals fire first, the accept path is covered by
    // the anchors).
    // lambda = 2 (build and config agree): refused with the reason.
    let l2 = fitsys_cfg(&ks, split_kcfg(2.0));
    let e = run(&l2.gdf, &split_kcfg(2.0), &l2.aux)
        .unwrap_err()
        .to_string();
    assert!(e.contains("lambda = 2 > 1"), "{e}");
}

// ----------------------------------- cheap q != 0 split FD (non-ignored)

/// Central FD of the split k energy at h for two components of the H2 1x1x3
/// non-TRIM mesh: zz (the mesh direction) and xz (off-diagonal), both exxdiv.
/// A reduced version of the all-nine FD (which stays ignored) so the q != 0
/// split strain is pinned by an independent anchor in the default test run.
#[test]
fn h2_1x1x3_split_stress_matches_fd_for_zz_and_xz() {
    let bs = pyscf_sto3g_h();
    let n = [1, 1, 3];
    let ks = ksys(h2_cell_g(), &bs, n);
    let fs = fitsys_cfg(&ks, split_kcfg(1.0));
    let scfs = fit_scfs(&ks, &fs);
    let kept0: Vec<usize> = fs.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
    for (i, j) in [(2, 2), (0, 2)] {
        let at = |h: f64| {
            let (e, kept) =
                k_split_energies(&strained(&ks.cell, i, j, h), &bs, n, MeshCentring::Gamma);
            assert_eq!(kept, kept0, "the metric cut changed under strain");
            e
        };
        let (ep, em) = (at(FD_H), at(-FD_H));
        for exx in EXX {
            let x = exx_idx(exx);
            let fd = (ep[x] - em[x]) / (2.0 * FD_H);
            let s = ks_rhf_fit(&ks, &fs, &scfs[x], exx, None);
            let miss = (s.de_deps[i][j] - fd).abs();
            eprintln!(
                "  split FD ({i},{j}) {exx:?}: analytic {:.8e}, |analytic − FD| {miss:.2e}",
                s.de_deps[i][j]
            );
            assert!(miss <= K_FD_BAR_FIT, "({i},{j}) {exx:?}: {miss:e}");
        }
    }
}

#[test]
fn open_shell_p_h2_triplet_one_point_mesh_uhf_stress_is_the_gamma_stress() {
    open_shell_p_one_point(s1_h2_triplet());
}

#[test]
#[ignore = "slow (~2 min release): H4 s+p, 2.8M SR triplets, k and Gamma, both exxdiv"]
fn open_shell_p_h4_triplet_one_point_mesh_uhf_stress_is_the_gamma_stress() {
    open_shell_p_one_point(s2_h4_triplet());
}

/// PHYSICS: the 1x1x3 (non-TRIM) k-UHF stress per cell is the explicit
/// 3-cell supercell's Gamma-UHF stress / 3, both exxdiv, at the default
/// model-nucleus exponent; the
/// k-specific mutants and the Gamma-style term mutants all miss it. S1 only
/// (the supercell has 24 AOs).
#[test]
#[ignore = "slow (~12 min release at load 10): 6-atom s+p supercell Gamma UHF stress x2 + 26 k stress mutant evaluations"]
fn open_shell_p_1x1x3_uhf_stress_is_the_supercell_stress_per_cell() {
    let bs = sp_basis_h();
    let sys = s1_h2_triplet();
    let n = [1, 1, 3];
    let ks = ksys(sys.cell.clone(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = uscf_pair(&ks, &eri, sys.na, sys.nb);
    assert_spin_polarised(&sys, &scfs[0]);
    // 3 cells x 2 unpaired electrons: multiplicity 7.
    let g = gsys(supercell(&sys.cell, n, 7), &bs);
    let mut de = [ZERO; 2];
    for (i, exx) in EXX.into_iter().enumerate() {
        let gscf = gamma_unrestricted(
            unfold(&ks.mesh, &scfs[i].density_alpha),
            unfold(&ks.mesh, &scfs[i].density_beta),
        );
        let k = ks_uhf(&ks, &eri, &scfs[i], exx, None);
        let gs = gamma_uhf_stress(
            &g.cell,
            &g.prep,
            &hcore_cfg(),
            &g.hc,
            &g.eri,
            &gscf,
            exx,
            &GammaStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
        )
        .expect("gamma uhf stress");
        let r = scale3(&gs.de_deps, 1.0 / 3.0);
        let miss = max_mat(&k.de_deps, &r);
        report(&format!("{} {n:?} {exx:?}", sys.name), &k);
        eprintln!(
            "  {} {n:?} {exx:?}: |dE/dε_k − dE/dε_sc/N| {miss:.2e} (bar {OS_SC_BAR:.0e})",
            sys.name
        );
        assert!(miss <= OS_SC_BAR, "{exx:?}: {miss:e}");
        assert!(k.antisymmetry() <= P_ANTISYM_BAR);
        de[i] = r;
    }
    let refs = ScRef { de };
    let run = |exx: ExxDiv, m: Option<KStressMutation>| {
        kpoint_uhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(),
            &ks.hk,
            KStressJk::Dense(&eri),
            &scfs[exx_idx(exx)],
            exx,
            &scfg(m),
        )
        .expect("kpoint_uhf_stress")
    };
    assert_mutants_fail_floor(
        sys.name,
        &run,
        &refs,
        &K_MUTANTS,
        &K_EWALD_MUTANTS,
        OS_SC_BAR,
    );
    assert_mutants_fail_floor(
        sys.name,
        &run,
        &refs,
        &TERM_MUTANTS,
        &TERM_EWALD_MUTANTS,
        OS_SC_BAR,
    );
}

/// PHYSICS: all nine components of the k-UHF stress of S1 vs central FD (h =
/// 1e-4, 2h) of its own k energy, a fresh UHF SCF per point, both exxdiv,
/// plus the UHF-specific and k-specific mutants. Bar K_FD_BAR_UHF (the H3
/// doublet's own measured floor, same h^2 truncation); cf. K_FD_BAR_P.
#[test]
#[ignore = "slow (~5.5 min release at load 10): 36 strained dense-AFT builds + 72 k-UHF SCFs (h and 2h), s+p H2 triplet"]
fn open_shell_p_h2_triplet_1x1x3_uhf_stress_matches_fd_all_nine() {
    let bs = sp_basis_h();
    let sys = s1_h2_triplet();
    let n = [1, 1, 3];
    let e = |c: &Cell| k_energies_uhf_nanb(c, &bs, n, sys.na, sys.nb);
    let fd1 = fd_all9(&sys.cell, FD_H, &e);
    let fd2 = fd_all9(&sys.cell, 2.0 * FD_H, &e);
    let ks = ksys(sys.cell.clone(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = uscf_pair(&ks, &eri, sys.na, sys.nb);
    assert_spin_polarised(&sys, &scfs[0]);
    for (x, exx) in EXX.into_iter().enumerate() {
        let s = ks_uhf(&ks, &eri, &scfs[x], exx, None);
        let miss = max_mat(&s.de_deps, &fd1[x]);
        let miss2 = max_mat(&s.de_deps, &fd2[x]);
        eprintln!(
            "  {} 1x1x3 {exx:?}: |analytic − FD| h {miss:.2e}, 2h {miss2:.2e} (ratio {:.2})",
            sys.name,
            miss2 / miss
        );
        assert!(miss <= K_FD_BAR_UHF, "{exx:?}: FD miss {miss:e}");
        assert!(
            (3.0..5.0).contains(&(miss2 / miss)),
            "not h²: {}",
            miss2 / miss
        );
        for m in [
            KStressMutation::CrossSpinK,
            KStressMutation::SwapSpinFock,
            KStressMutation::NoPhase,
            KStressMutation::WrongQ,
            KStressMutation::ConjD,
        ] {
            let sm = ks_uhf(&ks, &eri, &scfs[x], exx, Some(m));
            let mm = max_mat(&sm.de_deps, &fd1[x]);
            eprintln!("  {} {exx:?} mutant {m:?}: miss vs FD {mm:.2e}", sys.name);
            assert!(mm > MUTANT_BAR, "{m:?} {exx:?} not caught ({mm:e})");
        }
    }
}

/// S2 (3 alpha, 1 beta, 16 AOs, both sectors populated): zz, xz and xx of the
/// k-UHF stress vs central FD at h = 1e-4 on the 1x1x3 mesh, exxdiv none and
/// ewald.
#[test]
#[ignore = "slow (~20 min release at load 10): 6 strained dense-AFT builds + 12 k-UHF SCFs on H4 s+p"]
fn open_shell_p_h4_triplet_1x1x3_uhf_stress_matches_fd_zz_xz_xx() {
    let bs = sp_basis_h();
    let sys = s2_h4_triplet();
    let n = [1, 1, 3];
    let ks = ksys(sys.cell.clone(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = uscf_pair(&ks, &eri, sys.na, sys.nb);
    assert_spin_polarised(&sys, &scfs[0]);
    let mut worst_mut = [[0.0_f64; 2]; 2];
    for (i, j) in [(2usize, 2usize), (0, 2), (0, 0)] {
        let ep = k_energies_uhf_nanb(&strained(&sys.cell, i, j, FD_H), &bs, n, sys.na, sys.nb);
        let em = k_energies_uhf_nanb(&strained(&sys.cell, i, j, -FD_H), &bs, n, sys.na, sys.nb);
        for (x, exx) in EXX.into_iter().enumerate() {
            let fd = (ep[x] - em[x]) / (2.0 * FD_H);
            let s = ks_uhf(&ks, &eri, &scfs[x], exx, None);
            let miss = (s.de_deps[i][j] - fd).abs();
            eprintln!(
                "  {} 1x1x3 ({i},{j}) {exx:?}: analytic {:.8e} FD {fd:.8e} miss {miss:.2e}",
                sys.name, s.de_deps[i][j]
            );
            assert!(miss <= K_FD_BAR_UHF, "({i},{j}) {exx:?}: {miss:e}");
            for (mi, m) in [KStressMutation::CrossSpinK, KStressMutation::SwapSpinFock]
                .into_iter()
                .enumerate()
            {
                let sm = ks_uhf(&ks, &eri, &scfs[x], exx, Some(m));
                let mm = (sm.de_deps[i][j] - fd).abs();
                worst_mut[mi][x] = worst_mut[mi][x].max(mm);
            }
        }
    }
    eprintln!(
        "  {} spin mutants worst miss vs FD [none, ewald]: CrossSpinK {:?}, SwapSpinFock {:?}",
        sys.name, worst_mut[0], worst_mut[1]
    );
    for w in worst_mut.iter().flatten() {
        assert!(*w > MUTANT_BAR, "a spin mutant escaped the FD ({w:e})");
    }
}

/// PHYSICS (#434): the Gamma stress's SR-attraction term of a cell is
/// invariant under a rigid translation of every atom (the cell is periodic,
/// the stress is a derivative of a translation-invariant energy). Before the
/// SR derivative walk evaluated each triplet in the frame of its nucleus,
/// libint2 combined O(cell-size) absolute coordinates with the 1e9-sized
/// model-nucleus exponent and the p-shell `vsr_zz` of this 3-cell H2 triplet
/// supercell moved by 8.5e-8 (per cell) between translations (and sat 5.9e-8
/// off the k-point value at the default exponent). ARTIFACT vs PHYSICS: a
/// physical effect would survive a rigid translation, a numerical-noise
/// artifact does not; after the fix the translated values agree to 1e-14.
/// Density: the unfolded s+p k-UHF triplet density (`parts.vsr` is a
/// fixed-density quantity, so nothing needs to be SCF-converged Gamma-side).
#[test]
#[ignore = "slow (~1 min release): k-UHF SCF + 3 Gamma supercell hcore/ERI builds"]
fn gamma_sr_stress_is_translation_invariant_for_p_shells_at_the_default_exponent() {
    let bs = sp_basis_h();
    let sys = s1_h2_triplet();
    let n = [1, 1, 3];
    let ks = ksys(sys.cell.clone(), &bs, n);
    let eri = kdense(&ks, ANCHOR_PRECISION);
    let scfs = uscf_pair(&ks, &eri, sys.na, sys.nb);
    let sc = supercell(&sys.cell, n, 7);
    let gscf = gamma_unrestricted(
        unfold(&ks.mesh, &scfs[0].density_alpha),
        unfold(&ks.mesh, &scfs[0].density_beta),
    );
    let vsr_zz = |shift: [f64; 3]| {
        let pos: Vec<[f64; 3]> = sc
            .positions()
            .iter()
            .map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]])
            .collect();
        let g = gsys(cell_at(&pos, *sc.lattice(), 7), &bs);
        gamma_uhf_stress(
            &g.cell,
            &g.prep,
            &hcore_cfg(),
            &g.hc,
            &g.eri,
            &gscf,
            ExxDiv::None,
            &GammaStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
        )
        .expect("gamma uhf stress")
        .parts
        .vsr[2][2]
            / 3.0
    };
    let dk: Vec<Array2<Complex64>> = scfs[0]
        .density_alpha
        .iter()
        .zip(&scfs[0].density_beta)
        .map(|(a, b)| a + b)
        .collect();
    let k = kpoint_hcore_stress_fixed_d(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        &dk,
        &scfg(None),
    )
    .expect("k hcore stress")
    .vsr[2][2];
    let v0 = vsr_zz([0.0; 3]);
    let mut worst = (v0 - k).abs();
    eprintln!("  vsr_zz/3 unshifted {v0:.12}, k {k:.12}: {worst:.2e}");
    for shift in [[0.37, -0.21, 1.3], [0.0, 0.0, -3.0]] {
        let v = vsr_zz(shift);
        eprintln!("  shifted {shift:?}: {v:.12}");
        worst = worst.max((v - v0).abs()).max((v - k).abs());
    }
    assert!(
        worst <= 1e-10,
        "translation/k mismatch {worst:e} (was 8.5e-8)"
    );
}
