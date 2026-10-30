//! k-point closed-shell KS-DFT (`ferric_pbc::kdft`).
//!
//! What is independent of what:
//! * `bloch_ao_overlap_on_grid_matches_s_of_k` — the Bloch AO cache against
//!   the k-point overlap `S(k)` of `periodic_hcore_kpts` (analytic lattice
//!   sums): pins the phase convention `e^{+ik·L}` of the XC AOs to the one
//!   every other k-point quantity uses, with no SCF involved.
//! * `one_point_mesh_is_gamma_rks` — a 1×1×1 mesh against the Gamma
//!   `gamma_rks` on the SAME grid (different density/Vxc code: complex Bloch
//!   AOs vs real lattice-summed AOs).
//! * `k_mesh_is_the_gamma_supercell` — E/cell of an N-point mesh against the
//!   Gamma KS energy of the explicit N-cell supercell / N, on the supercell
//!   grid made of N translated copies of the cell grid (so the two
//!   integrations are the same sum, [`PeriodicGrid::replicated`]).
//! * `uniform_grid_krks_matches_pinned_pyscf` — PySCF 2.13 `pbc.dft.KRKS`
//!   with `UniformGrids` (AFTDF, mesh 61³, exxdiv ewald): an external pin,
//!   a different integration construction through our Bloch AOs, libxc
//!   kernel and hybrid-K scaling.
//!
//! The tests whose per-commit CI time was >= 120 s
//! (`mgga_uniform_grid_krks_matches_pinned_pyscf`,
//! `mgga_scan_and_r2scan_are_distinguished_by_the_pins`) run weekly from
//! `validation_pbc_krks.rs`; the shared helpers are `pbc_krks_support/mod.rs`.

mod common;
#[path = "pbc_krks_support/mod.rs"]
mod support;

use support::*;

// ---------------------------------------------------------------- Bloch AOs

#[test]
fn bloch_ao_overlap_on_grid_matches_s_of_k() {
    for (cell, bs, mesh_n, name) in [
        (h2_cell(4.0), pyscf_sto3g_h(), [1, 1, 3], "H2 1x1x3"),
        (triclinic_cell(), sp_basis_h(), [1, 2, 1], "tri 1x2x1"),
    ] {
        let prep = prep_for(&cell, &bs);
        let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
        let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA))
            .unwrap();
        let grid = PeriodicGrid::uniform(&cell, [32; 3]).unwrap();
        let pxc = KPeriodicXc::new(
            &cell,
            &bs,
            "PBE",
            &grid,
            &mesh,
            &PeriodicXcConfig::default(),
        )
        .unwrap();
        let mut worst = 0.0_f64;
        for k in 0..mesh.nk() {
            let sg = pxc.overlap_on_grid(k);
            let d = (&sg - &hk.s[k])
                .iter()
                .fold(0.0_f64, |m, z| m.max(z.norm()));
            worst = worst.max(d);
        }
        eprintln!("{name}: max|S_grid(k) - S(k)| = {worst:.3e}");
        assert!(worst < 1e-5, "{name}: {worst}");
    }
}

// ------------------------------------------------------ 1-point mesh = Gamma

#[test]
fn one_point_mesh_is_gamma_rks() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    for functional in ["LDA", "PBE", "PBE0"] {
        let mut gcfg = GammaRksConfig::new(functional);
        gcfg.grid = grid_cfg(50, 110);
        let g = gamma_rks(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &gcfg).unwrap();
        let mut cfg = krks_cfg(&cell, functional, ExxDiv::Ewald);
        cfg.grid = grid_cfg(50, 110);
        let k = solve_krks(&cell, &prep, None, &mesh, &cfg).unwrap();
        eprintln!(
            "{functional}: E_k {:.12} E_gamma {:.12} dE {:.2e}; E_xc dE {:.2e}; \
             |N_grid dE| {:.2e}",
            k.scf.energy,
            g.scf.energy,
            k.scf.energy - g.scf.energy,
            k.e_xc - g.e_xc,
            k.electrons_on_grid - g.electrons_on_grid
        );
        assert_eq!(k.n_grid_points, g.n_grid_points);
        assert!((k.scf.energy - g.scf.energy).abs() < 1e-9, "{functional}");
        assert!((k.e_xc - g.e_xc).abs() < 1e-9, "{functional} E_xc");
        assert!((k.electrons_on_grid - g.electrons_on_grid).abs() < 1e-9);
    }
}

#[test]
fn k_mesh_is_the_gamma_supercell() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    for functional in ["LDA", "PBE", "PBE0"] {
        supercell_anchor("H2", &cell, &bs, [1, 1, 3], functional);
    }
    supercell_anchor("H2", &cell, &bs, [2, 2, 1], "PBE0");
}

// ------------------------------------------------------------- PySCF pins

#[test]
fn uniform_grid_krks_matches_pinned_pyscf() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    for &(functional, n, e_ref) in &PYSCF {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        let r = krks_uniform(&cell, &bs, &mesh, functional, 40);
        eprintln!(
            "H2 {functional} {n:?}: {:.12} PySCF {e_ref:.12} dE {:.2e}",
            r.scf.energy,
            r.scf.energy - e_ref
        );
        assert!(
            (r.scf.energy - e_ref).abs() < 1e-8,
            "{functional} {n:?}: {} vs {e_ref}",
            r.scf.energy
        );
    }
}

/// The Gamma entry's XC (`PeriodicXc::new_closed_mgga`, real lattice-summed
/// AOs, the object `gamma_rks` injects) against PySCF's DIRECT Gamma-point RKS
/// on the same uniform grid: a pin that does not pass through the k-point code.
#[test]
fn gamma_mgga_matches_pinned_pyscf_gamma_rks() {
    let cell = triclinic_cell();
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::uniform(&cell, [61; 3]).unwrap();
    for &(functional, e_ref) in &PYSCF_MGGA_TRI_GAMMA {
        let e = gamma_rks_on_grid(&cell, &bs, &grid, functional);
        eprintln!(
            "H4 tri Gamma {functional}: {e:.12} PySCF {e_ref:.12} dE {:.2e}",
            e - e_ref
        );
        assert!((e - e_ref).abs() < 1e-9, "{functional}: {e} vs {e_ref}");
    }
}

/// TPSS closed-shell ENERGY (issue #355) pinned to PySCF at the k-point
/// (1x1x1, 1x1x2) and the direct Gamma-point RKS, on the cell where the
/// meta-GGAs differ. Measured: 2.5e-13 (1x1x1), 1.6e-14 (1x1x2), 2.4e-13
/// (Gamma); bar 1e-9.
///
/// Mutation ledger (source mutants, run and reverted 2026-10-18): the friendly
/// name `TPSS` mapped to the SCAN pair misses by 1.41e-2 Ha (the TPSS - SCAN
/// gap, also asserted below); V^tau zeroed in `KPeriodicXc` misses by 2.38e-5
/// Ha at 1x1x1.
#[test]
fn tpss_krks_and_gamma_match_pinned_pyscf() {
    let cell = triclinic_cell();
    let bs = pyscf_sto3g_h();
    for &(n, e_ref) in &PYSCF_TPSS_TRI {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        let r = krks_uniform(&cell, &bs, &mesh, "TPSS", 61);
        eprintln!(
            "H4 tri TPSS {n:?}: {:.12} PySCF {e_ref:.12} dE {:.2e}",
            r.scf.energy,
            r.scf.energy - e_ref
        );
        assert!(
            (r.scf.energy - e_ref).abs() < 1e-9,
            "TPSS {n:?}: {} vs {e_ref}",
            r.scf.energy
        );
        let scan = PYSCF_MGGA_TRI
            .iter()
            .find(|(f, m, _)| *f == "SCAN" && *m == n)
            .unwrap()
            .2;
        assert!((e_ref - scan).abs() > 1e-3, "{n:?}: TPSS - SCAN gap");
    }
    let grid = PeriodicGrid::uniform(&cell, [61; 3]).unwrap();
    let e = gamma_rks_on_grid(&cell, &bs, &grid, "TPSS");
    let e_ref = PYSCF_TPSS_TRI_GAMMA;
    eprintln!(
        "H4 tri Gamma TPSS: {e:.12} PySCF {e_ref:.12} dE {:.2e}",
        e - e_ref
    );
    assert!((e - e_ref).abs() < 1e-9, "Gamma TPSS: {e} vs {e_ref}");
}

/// Gamma-point meta-GGA (`gamma_rks`, lattice-summed real AOs) against the
/// 1x1x1 k-mesh (complex Bloch AOs) on the same grid (issue #355): two
/// independent energy constructions (the Gamma meta-GGA force and stress are
/// the k-point code routed through 1x1x1, `pbc_gamma_mgga_derivs.rs`).
#[test]
fn mgga_one_point_mesh_is_gamma_rks() {
    let cell = triclinic_cell();
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    for functional in ["SCAN", "R2SCAN", "TPSS"] {
        let mut gcfg = GammaRksConfig::new(functional);
        gcfg.grid = grid_cfg(50, 110);
        gcfg.scf.energy_conv = 1e-12;
        let g = gamma_rks(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &gcfg).unwrap();
        let mut cfg = krks_cfg(&cell, functional, ExxDiv::Ewald);
        cfg.grid = grid_cfg(50, 110);
        let k = solve_krks(&cell, &prep, None, &mesh, &cfg).unwrap();
        eprintln!(
            "{functional}: E_k {:.12} E_gamma {:.12} dE {:.2e}; E_xc dE {:.2e}",
            k.scf.energy,
            g.scf.energy,
            k.scf.energy - g.scf.energy,
            k.e_xc - g.e_xc
        );
        assert!((k.scf.energy - g.scf.energy).abs() < 1e-8, "{functional}");
        assert!((k.e_xc - g.e_xc).abs() < 1e-8, "{functional} E_xc");
    }
}

/// Meta-GGA k-mesh == supercell / N: the supercell is solved by the same
/// k-point code at a 1x1x1 mesh on the replicated grid (the Gamma meta-GGA
/// comparison is `mgga_one_point_mesh_is_gamma_rks`).
#[test]
fn mgga_k_mesh_is_the_supercell() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let cell_grid = PeriodicGrid::build(&cell, &grid_cfg(40, 110)).unwrap();
    for (functional, n) in [("SCAN", [1, 1, 3]), ("R2SCAN", [2, 1, 1])] {
        let nk = (n[0] * n[1] * n[2]) as f64;
        let (sc, shifts) = supercell(&cell, n);
        let sc_grid = cell_grid.replicated(&shifts);
        let sprep = prep_for(&sc, &bs);
        let one = KPointMesh::gamma_centred(&sc, [1, 1, 1]).unwrap();
        let es = solve_krks_on_grid(
            &sc,
            &sprep,
            None,
            &one,
            &sc_grid,
            &krks_cfg(&sc, functional, ExxDiv::Ewald),
        )
        .unwrap()
        .scf
        .energy
            / nk;
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        let r = solve_krks_on_grid(
            &cell,
            &prep,
            None,
            &mesh,
            &cell_grid,
            &krks_cfg(&cell, functional, ExxDiv::Ewald),
        )
        .unwrap();
        eprintln!(
            "{functional} {n:?}: E_k {:.12} E_super/N {es:.12} dE {:.2e}",
            r.scf.energy,
            r.scf.energy - es
        );
        assert!((r.scf.energy - es).abs() < 1e-8, "{functional} {n:?}");
    }
}

/// `V_xc(k)` is the derivative of `E_xc` w.r.t. `D(k)`: central finite
/// difference of `E_xc` along a Hermitian perturbation against `Re Tr(V Δ)`.
/// Exercises the tau term of `V` independently of any SCF.
#[test]
fn mgga_vxc_is_the_derivative_of_exc() {
    use ferric_pbc::kscf::KPointXc;
    use ndarray::Array2;
    use num_complex::Complex64;
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let grid = PeriodicGrid::build(&cell, &grid_cfg(30, 110)).unwrap();
    for functional in ["SCAN", "R2SCAN", "TPSS"] {
        let r = solve_krks_on_grid(
            &cell,
            &prep,
            None,
            &mesh,
            &grid,
            &krks_cfg(&cell, functional, ExxDiv::Ewald),
        )
        .unwrap();
        let mut pxc = KPeriodicXc::new(
            &cell,
            &bs,
            functional,
            &grid,
            &mesh,
            &PeriodicXcConfig::default(),
        )
        .unwrap();
        let d0 = r.scf.densities.clone();
        let (_, v) = pxc.build(&d0).unwrap();
        // Deterministic Hermitian perturbation.
        let delta: Vec<Array2<Complex64>> = d0
            .iter()
            .enumerate()
            .map(|(k, d)| {
                let n = d.nrows();
                let mut m = Array2::<Complex64>::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        let a = ((i * 7 + j * 3 + k * 5) % 11) as f64 - 5.0;
                        let b = ((i * 2 + j * 9 + k) % 7) as f64 - 3.0;
                        m[(i, j)] += Complex64::new(a, b);
                        m[(j, i)] += Complex64::new(a, -b);
                    }
                }
                m
            })
            .collect();
        let eps = 1e-4;
        let shift = |s: f64| -> Vec<Array2<Complex64>> {
            d0.iter()
                .zip(&delta)
                .map(|(d, x)| d + &x.mapv(|z| z * s))
                .collect()
        };
        let ep = pxc.build(&shift(eps)).unwrap().0;
        let em = pxc.build(&shift(-eps)).unwrap().0;
        let fd = (ep - em) / (2.0 * eps);
        // V carries no 1/N_k (k-point Fock convention); dE/dD(k) = V/N_k.
        let mut an = 0.0;
        for (vk, xk) in v.iter().zip(&delta) {
            for i in 0..vk.nrows() {
                for j in 0..vk.ncols() {
                    an += (vk[(i, j)] * xk[(j, i)]).re;
                }
            }
        }
        let an = an / mesh.nk() as f64;
        eprintln!(
            "{functional}: FD {fd:.10} analytic {an:.10} diff {:.2e}",
            fd - an
        );
        assert!((fd - an).abs() < 1e-6 * an.abs().max(1.0), "{functional}");
    }
}
