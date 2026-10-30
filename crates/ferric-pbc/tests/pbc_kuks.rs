//! k-point open-shell KS-DFT (`ferric_pbc::kuks::solve_kuks`): LDA, GGA and
//! global hybrids on a k-mesh, built on the k-point UHF loop
//! (`run_kuhf`) with a spin-polarized Bloch-AO XC builder.
//!
//! What is independent of what:
//! * `closed_shell_limit_is_krks` — `solve_kuks` on a singlet against
//!   `solve_krks` (closed-shell code, different SCF loop and XC kernel
//!   entry: unpolarized `closed_kernel` vs `polarized_kernel`), same grid.
//! * `one_point_mesh_is_gamma_uks` — 1x1x1 mesh against `gamma_uks` (real
//!   lattice-summed AOs, `ferric_scf::uhf`), open shell, same grid.
//! * `k_mesh_is_the_gamma_supercell` — E/cell of an N-point mesh against the
//!   Gamma UKS energy of the explicit supercell / N on the replicated grid,
//!   open shell (giant-determinant multiplicity `(N_a - N_b) N_k + 1`).
//! * `uniform_grid_kuks_matches_pinned_pyscf` — PySCF 2.13 `pbc.dft.KUKS`
//!   with `UniformGrids` (AFTDF, mesh 61^3, exxdiv ewald, `mf.nelec` set to
//!   `(N_a N_k, N_b N_k)`): an external pin of the whole path.
//! * `mutants_are_detected` — the k-UHF mutants applied through the KS path
//!   move the closed-shell hybrid energy off `solve_krks`.
//!
//! The tests whose per-commit CI time was >= 120 s
//! (`uniform_grid_kuks_matches_pinned_pyscf`) run weekly from
//! `validation_pbc_kuks.rs`; the shared helpers are `pbc_kuks_support/mod.rs`.

mod common;
#[path = "pbc_kuks_support/mod.rs"]
mod support;

use support::*;

// -------------------------------------------------- closed-shell limit

#[test]
fn closed_shell_limit_is_krks() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let grid = PeriodicGrid::build(&cell, &grid_cfg(40, 110)).unwrap();
    for n in [[1, 1, 3], [2, 1, 1]] {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        for functional in ["LDA", "PBE", "PBE0"] {
            let mut rcfg = KRksConfig::new(&cell, ExxDiv::Ewald, functional);
            rcfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
            rcfg.krhf.scf.energy_conv = 1e-12;
            rcfg.krhf.scf.grad_conv = 1e-9;
            let r = solve_krks_on_grid(&cell, &prep, None, &mesh, &grid, &rcfg).unwrap();
            let u = kuks_on(&cell, &bs, &mesh, &grid, functional);
            eprintln!(
                "H2 {n:?} {functional}: E_u {:.12} E_r {:.12} dE {:.2e}; E_xc dE {:.2e}; <S2> {:.1e}",
                u.scf.energy,
                r.scf.energy,
                u.scf.energy - r.scf.energy,
                u.e_xc - r.e_xc,
                u.scf.s2
            );
            assert!((u.scf.energy - r.scf.energy).abs() < 1e-9, "{functional}");
            assert!((u.e_xc - r.e_xc).abs() < 1e-9, "{functional} E_xc");
            assert!(
                (u.electrons_on_grid - r.electrons_on_grid).abs() < 1e-9,
                "{functional} N_grid"
            );
            assert!(u.scf.s2.abs() < 1e-8);
        }
    }
}

#[test]
fn mutants_are_detected() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let grid = PeriodicGrid::build(&cell, &grid_cfg(40, 110)).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let rcfg = {
        let mut c = KRksConfig::new(&cell, ExxDiv::Ewald, "PBE0");
        c.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
        c.krhf.scf.energy_conv = 1e-12;
        c.krhf.scf.grad_conv = 1e-9;
        c
    };
    let r = solve_krks_on_grid(&cell, &prep, None, &mesh, &grid, &rcfg).unwrap();
    for m in [
        KUhfMutation::KFromTotalDensity,
        KUhfMutation::MadelungHalfPerSpin,
    ] {
        let mut cfg = kuks_cfg(&cell, "PBE0", ExxDiv::Ewald);
        cfg.kuhf.mutation = Some(m);
        let u = solve_kuks_on_grid(&cell, &prep, None, &mesh, &grid, &cfg).unwrap();
        let d = (u.scf.energy - r.scf.energy).abs();
        eprintln!("{m:?}: |dE| {d:.3e}");
        assert!(d > 1e-3, "{m:?} not detected");
    }
}

// ----------------------------------------------------- 1-point = Gamma

#[test]
fn one_point_mesh_is_gamma_uks() {
    let cases: Vec<(&str, Cell, BasisSet)> = vec![
        (
            "H doublet sp",
            cell_mult(&[[0.3, 0.2, 0.1]], 4.0, 2),
            sp_basis_h(),
        ),
        ("H2 triplet", cell_mult(&H2_ATOMS, 4.0, 3), pyscf_sto3g_h()),
    ];
    for (label, cell, bs) in cases {
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
            let mut gcfg = GammaUksConfig::new(functional);
            gcfg.grid = grid_cfg(50, 110);
            gcfg.scf.density_conv = 1e-11;
            let g = gamma_uks(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &gcfg).unwrap();
            let grid = PeriodicGrid::build(&cell, &grid_cfg(50, 110)).unwrap();
            let k = kuks_on(&cell, &bs, &mesh, &grid, functional);
            eprintln!(
                "{label} {functional}: E_k {:.12} E_gamma {:.12} dE {:.2e}; E_xc dE {:.2e}; \
                 S2 {:.6} vs {:.6}",
                k.scf.energy,
                g.scf.energy,
                k.scf.energy - g.scf.energy,
                k.e_xc - g.e_xc,
                k.scf.s2,
                g.s2
            );
            assert_eq!(k.n_grid_points, g.grid.as_ref().unwrap().n_grid_points);
            assert!(
                (k.scf.energy - g.scf.energy).abs() < 1e-9,
                "{label} {functional}"
            );
            assert!((k.e_xc - g.e_xc).abs() < 1e-9, "{label} {functional} E_xc");
            assert!((k.scf.s2 - g.s2).abs() < 1e-8, "{label} {functional} S2");
        }
    }
}

#[test]
fn k_mesh_is_the_gamma_supercell() {
    let h2t = cell_mult(&H2_ATOMS, 4.0, 3);
    let bs = pyscf_sto3g_h();
    for functional in ["LDA", "PBE", "PBE0"] {
        supercell_anchor("H2 triplet", &h2t, &bs, [1, 1, 3], functional, 2, 0);
    }
    let hd = cell_mult(&[[0.3, 0.2, 0.1]], 4.0, 2);
    let sp = sp_basis_h();
    for functional in ["PBE", "PBE0"] {
        supercell_anchor("H doublet sp", &hd, &sp, [1, 1, 3], functional, 1, 0);
    }
    // Both spins populated (H2 cell, alpha 2 / beta 1 impossible for H2;
    // use a 3-H cell: doublet, N_a = 2, N_b = 1).
    let h3 = cell_mult(&[[0.3, 0.2, 0.1], [0.3, 0.2, 1.5], [0.3, 1.9, 0.4]], 4.0, 2);
    supercell_anchor("H3 doublet", &h3, &bs, [1, 1, 2], "PBE0", 2, 1);
}

/// Per-commit canary of `validation_pbc_kuks.rs`'s 15-pin
/// `uniform_grid_kuks_matches_pinned_pyscf`: the non-TRIM open-shell hybrid
/// (Hd PBE0 1x1x3: phases, Madelung-scaled exchange, spin-polarized XC) and
/// the triplet GGA (H2t PBE 1x1x1), the two pins that exercise the most of
/// the path per second.
#[test]
fn uniform_grid_kuks_matches_a_pinned_pyscf_subset() {
    let subset: Vec<_> = PYSCF
        .iter()
        .copied()
        .filter(|&(sys, functional, n, _)| {
            (sys, functional, n) == ("Hd", "PBE0", [1, 1, 3])
                || (sys, functional, n) == ("H2t", "PBE", [1, 1, 1])
        })
        .collect();
    assert_eq!(subset.len(), 2);
    assert_kuks_pins(&subset);
}

/// The open-shell solver's meta-GGA gate (issue #355): ONLY the pure SCAN /
/// r2SCAN are accepted (see `pbc_kuks_mgga.rs`); every other meta-GGA is
/// refused on both entry points even where the closed-shell XC accepts it.
#[test]
fn kuks_refuses_unvalidated_meta_gga() {
    let cell = cell_mult(&[[0.0, 0.0, 0.0]], 4.0, 2);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let cfg = kuks_cfg(&cell, "MGGA_X_TPSS", ExxDiv::None);
    let grid = PeriodicGrid::build(&cell, &cfg.grid).unwrap();
    assert!(
        solve_kuks_on_grid(&cell, &prep, None, &mesh, &grid, &cfg).is_err(),
        "solve_kuks_on_grid accepted a raw TPSS exchange"
    );
    assert!(
        ferric_pbc::kuks::solve_kuks(&cell, &prep, None, &mesh, &cfg).is_err(),
        "solve_kuks accepted a raw TPSS exchange"
    );
}
