//! k-point open-shell meta-GGA (`ferric_pbc::kuks::solve_kuks`, pure SCAN and
//! r2SCAN, ENERGY ONLY; issue #355).
//!
//! # Artifact hypothesis next to the physics hypothesis
//!
//! * Physics right: (1) a singlet (`N_alpha == N_beta`) k-UKS equals the
//!   closed-shell k-RKS to rounding; (2) genuinely polarized energies match
//!   PySCF `pbc.dft.KUKS` per functional; (3) an N-point mesh equals the
//!   N-cell supercell / N.
//! * Artifact: a spin-channel or tau error (swapped tau, a dropped beta
//!   `V^tau`, the TOTAL tau in both channels, the spin-summed evaluator) is
//!   INVISIBLE in the singlet limit, because there `tau_a == tau_b == tau/2`
//!   and `rho_a == rho_b`. So the singlet anchor can only say "no gross
//!   break"; every discriminating pin here is polarized (`N_alpha != N_beta`)
//!   and the mutants are required to fail THOSE pins, not the closed-shell
//!   one.
//!
//! # Independence
//!
//! * `singlet_k_uks_is_k_rks` (anchor, written and passed BEFORE any PySCF
//!   number): the polarized kernel/tau code against the unpolarized
//!   `closed_kernel` + `chunk_tau` of k-RKS.
//! * `one_point_mesh_singlet_is_gamma_rks`: 1x1x1 against `gamma_rks`
//!   (real lattice-summed AOs).
//! * `polarized_vxc_is_the_derivative_of_exc`: `V_sigma(k)` against a central
//!   finite difference of `E_xc` along Hermitian perturbations of ONE spin
//!   density at a time (checks each spin's `V^tau` separately, no SCF).
//! * `polarized_k_mesh_is_the_supercell`: polarized mesh energy / cell
//!   against the same code at 1x1x1 on the explicit supercell.
//! * `h3_scan_matches_pinned_pyscf` (always on) and, in
//!   `validation_pbc_kuks_mgga.rs` (weekly validation job), the full table of
//!   eight PySCF `KUKS` pins plus the mutant-versus-pin checks.
//!
//! Caveat on the singlet anchor: `chunk_tau` is shared by both of its sides
//! (k-RKS and k-UKS), so a tau-FORMULA bug cancels there; that is covered by
//! the closed-shell PySCF pins of `pbc_krks.rs`. Meshes with a 1/3 component
//! (complex Bloch AOs) are included so a missing complex conjugate shows.
//!
//! No SCF level shift was needed: every SCF converged in 7 to 12 iterations,
//! but only on these 3 to 4 orbital minimal-basis cells (no evidence for
//! larger or near-degenerate cells).

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{
    gamma_rks, GammaRksConfig, PeriodicGrid, PeriodicGridConfig, PeriodicXcConfig,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kuks::{
    solve_kuks_on_grid, KPeriodicXcPolarized, KUksConfig, KUksMggaMutation, KUksResult,
};
use ferric_pbc::kuscf::KPointXcPolarized;
use ferric_pbc::lattice::Cell;
use ferric_pbc::uhf::GammaUhfIntegrals;
use ndarray::Array2;
use num_complex::Complex64;

const OMEGA: f64 = 0.8;

fn cell_mult(atoms: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(atoms);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

/// Three hydrogens in the cubic a = 4 cell: doublet, N_alpha = 2, N_beta = 1.
const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.3, 0.2, 1.5], [0.3, 1.9, 0.4]];

fn h3_doublet() -> Cell {
    cell_mult(&H3_ATOMS, cubic(4.0), 2)
}

/// The four-hydrogen triclinic cell as a triplet: N_alpha = 3, N_beta = 1.
fn h4_triplet() -> Cell {
    cell_mult(&TRI_ATOMS, TRI_A, 3)
}

fn kuks_cfg(cell: &Cell, functional: &str) -> KUksConfig {
    let mut cfg = KUksConfig::new(cell, ExxDiv::Ewald, functional);
    cfg.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.kuhf.scf.energy_conv = 1e-12;
    cfg.kuhf.scf.grad_conv = 1e-9;
    cfg
}

fn krks_cfg(cell: &Cell, functional: &str) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, ExxDiv::Ewald, functional);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-12;
    cfg.krhf.scf.grad_conv = 1e-9;
    cfg
}

fn grid_cfg(n_rad: usize, n_ang: usize) -> PeriodicGridConfig {
    PeriodicGridConfig::with_size(n_rad, n_ang)
}

fn kuks_on(
    cell: &Cell,
    bs: &BasisSet,
    mesh: &KPointMesh,
    grid: &PeriodicGrid,
    functional: &str,
) -> KUksResult {
    let prep = prep_for(cell, bs);
    solve_kuks_on_grid(cell, &prep, None, mesh, grid, &kuks_cfg(cell, functional))
        .unwrap_or_else(|e| panic!("solve_kuks {functional}: {e}"))
}

// ------------------------------------------------------------ the anchors

/// EXACTNESS ANCHOR (run before any PySCF number): at `N_alpha == N_beta` the
/// open-shell code must reproduce k-RKS, on the 1x1x2, 1x1x3 (complex Bloch
/// AOs) and 2x2x2 meshes.
/// Blind to a tau/spin-channel error by construction (module doc), so it is
/// necessary, not sufficient.
#[test]
fn singlet_k_uks_is_k_rks() {
    let bs = pyscf_sto3g_h();
    for (label, cell, meshes) in [
        ("H2", h2_cell(4.0), vec![[1, 1, 2], [1, 1, 3], [2, 2, 2]]),
        ("H4 tri", triclinic_cell(), vec![[1, 1, 2], [1, 1, 3]]),
    ] {
        let prep = prep_for(&cell, &bs);
        let grid = PeriodicGrid::build(&cell, &grid_cfg(40, 110)).unwrap();
        for n in meshes {
            let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
            for functional in ["SCAN", "R2SCAN"] {
                let r = solve_krks_on_grid(
                    &cell,
                    &prep,
                    None,
                    &mesh,
                    &grid,
                    &krks_cfg(&cell, functional),
                )
                .unwrap();
                let u = kuks_on(&cell, &bs, &mesh, &grid, functional);
                eprintln!(
                    "{label} {n:?} {functional}: E_u {:.12} E_r {:.12} dE {:.2e}; E_xc dE {:.2e}; \
                     it {} (u) / {} (r)",
                    u.scf.energy,
                    r.scf.energy,
                    u.scf.energy - r.scf.energy,
                    u.e_xc - r.e_xc,
                    u.scf.iterations,
                    r.scf.iterations
                );
                assert!(
                    (u.scf.energy - r.scf.energy).abs() < 1e-10,
                    "{label} {n:?} {functional}"
                );
                assert!((u.e_xc - r.e_xc).abs() < 1e-10, "{label} {functional} E_xc");
                assert!(u.scf.s2.abs() < 1e-8);
            }
        }
    }
}

/// The 1x1x1 singlet against the Gamma-point `gamma_rks` (real
/// lattice-summed AOs; the Gamma entry has no open-shell meta-GGA).
#[test]
fn one_point_mesh_singlet_is_gamma_rks() {
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
    let grid = PeriodicGrid::build(&cell, &grid_cfg(50, 110)).unwrap();
    for functional in ["SCAN", "R2SCAN"] {
        let mut gcfg = GammaRksConfig::new(functional);
        gcfg.grid = grid_cfg(50, 110);
        gcfg.scf.energy_conv = 1e-12;
        let g = gamma_rks(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &gcfg).unwrap();
        let u = kuks_on(&cell, &bs, &mesh, &grid, functional);
        eprintln!(
            "{functional}: E_k-uks {:.12} E_gamma-rks {:.12} dE {:.2e}",
            u.scf.energy,
            g.scf.energy,
            u.scf.energy - g.scf.energy
        );
        assert!((u.scf.energy - g.scf.energy).abs() < 1e-10, "{functional}");
        assert!((u.e_xc - g.e_xc).abs() < 1e-10, "{functional} E_xc");
    }
}

// ------------------------------------------------- polarized V = dE/dD

/// Worst relative miss, over the two spins, between a central finite
/// difference of `E_xc` along a Hermitian perturbation of ONE spin density
/// (the other fixed) and `Re Tr(V_sigma Delta) / N_k` of `pxc` itself.
fn fd_vs_v_worst_rel(
    pxc: &mut KPeriodicXcPolarized,
    mesh: &KPointMesh,
    da: &[Array2<Complex64>],
    db: &[Array2<Complex64>],
    label: &str,
) -> f64 {
    let (_, va, vb) = pxc.build_polarized(da, db).unwrap();
    let delta: Vec<Array2<Complex64>> = da
        .iter()
        .enumerate()
        .map(|(k, d)| {
            let n = d.nrows();
            let mut m = Array2::<Complex64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    let a = ((i * 7 + j * 3 + k * 5) % 11) as f64 - 5.0;
                    let b = ((i * 5 + j * 2 + k * 3) % 7) as f64 - 3.0;
                    m[(i, j)] = Complex64::new(a, b) * 1e-2;
                }
            }
            let h = m.mapv(|z| z.conj()).t().to_owned();
            (&m + &h).mapv(|z| z * 0.5)
        })
        .collect();
    let h = 1e-4;
    let mut worst = 0.0_f64;
    for spin in 0..2 {
        let shift = |s: f64| -> Vec<Array2<Complex64>> {
            let base = if spin == 0 { da } else { db };
            base.iter()
                .zip(&delta)
                .map(|(d, x)| d + &x.mapv(|z| z * s))
                .collect()
        };
        let mut e_at = |s: f64| {
            if spin == 0 {
                pxc.build_polarized(&shift(s), db).unwrap().0
            } else {
                pxc.build_polarized(da, &shift(s)).unwrap().0
            }
        };
        let fd = (e_at(h) - e_at(-h)) / (2.0 * h);
        let v = if spin == 0 { &va } else { &vb };
        let an: f64 = v
            .iter()
            .zip(&delta)
            .map(|(vk, dk)| {
                vk.iter()
                    .zip(dk.t().iter())
                    .map(|(a, b)| (a * b).re)
                    .sum::<f64>()
            })
            .sum::<f64>()
            / mesh.nk() as f64;
        let rel = (fd - an).abs() / fd.abs().max(1e-12);
        eprintln!("{label} spin {spin}: FD {fd:.10} analytic {an:.10} rel {rel:.2e}");
        worst = worst.max(rel);
    }
    worst
}

/// `V_sigma(k)` is the derivative of `E_xc` w.r.t. `D_sigma(k)`, at a
/// polarized converged state, on the real-AO meshes 1x1x2 and the complex-AO
/// mesh 1x1x3 (a missing conjugate in `V^tau` or `chunk_tau(D_sigma)` is
/// invisible at k = 0, 1/2). Measured worst correct relative miss 4.2e-9
/// (1x1x2) and 1.2e-8 (1x1x3); bar 2e-7 (16x). Under the mutants the miss is
/// 5.5e-4 / 9.3e-4 (`DropBetaVtau`) and 1.8e-2 (`SwapTau`) at 1x1x2, and 3.7
/// / 5.0 (`DropBetaVtau`) and 1.1 to 1.5 (`SwapTau`) at 1x1x3 (SCAN / r2SCAN
/// ranges merged), required > 1e-5.
///
/// The same check must FAIL under `DropBetaVtau` and `SwapTau` (no 61^3 SCF
/// needed): energy pins are inherently weak for Fock-only errors (dropping
/// beta `V^tau` is second order in the energy, 2.0e-5 at 1x1x1 and 2.1e-6 at
/// 1x1x2), the FD check is first order.
#[test]
fn polarized_vxc_is_the_derivative_of_exc() {
    let cell = h3_doublet();
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::build(&cell, &grid_cfg(30, 110)).unwrap();
    for n in [[1, 1, 2], [1, 1, 3]] {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        for functional in ["SCAN", "R2SCAN"] {
            let r = kuks_on(&cell, &bs, &mesh, &grid, functional);
            let (da, db) = (r.scf.density_alpha.clone(), r.scf.density_beta.clone());
            let new_pxc = || {
                KPeriodicXcPolarized::new(
                    &cell,
                    &bs,
                    functional,
                    &grid,
                    &mesh,
                    &PeriodicXcConfig::default(),
                )
                .unwrap()
            };
            let ok = fd_vs_v_worst_rel(
                &mut new_pxc(),
                &mesh,
                &da,
                &db,
                &format!("{functional} {n:?}"),
            );
            assert!(ok < 2e-7, "{functional} {n:?}: worst rel {ok:.2e}");
            for m in [KUksMggaMutation::DropBetaVtau, KUksMggaMutation::SwapTau] {
                let bad = fd_vs_v_worst_rel(
                    &mut new_pxc().with_mgga_mutation(m),
                    &mesh,
                    &da,
                    &db,
                    &format!("{functional} {n:?} {m:?}"),
                );
                eprintln!("{functional} {n:?} {m:?}: FD miss {bad:.2e} (correct {ok:.2e})");
                assert!(
                    bad > 1e-5,
                    "{functional} {n:?} {m:?} not detected: {bad:.2e}"
                );
            }
        }
    }
}

// ------------------------------------------------------ k-mesh = supercell

fn supercell(cell: &Cell, n: [usize; 3], mult: usize) -> (Cell, Vec<[f64; 3]>) {
    let a = *cell.lattice();
    let mut atoms = Vec::new();
    let mut shifts = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                let t: Vec<f64> = (0..3)
                    .map(|d| m0 as f64 * a[0][d] + m1 as f64 * a[1][d] + m2 as f64 * a[2][d])
                    .collect();
                shifts.push([t[0], t[1], t[2]]);
                for p in cell.positions() {
                    atoms.push([p[0] + t[0], p[1] + t[1], p[2] + t[2]]);
                }
            }
        }
    }
    let mut lat = a;
    for (i, row) in lat.iter_mut().enumerate() {
        for v in row.iter_mut() {
            *v *= n[i] as f64;
        }
    }
    let mut mol = hydrogens(&atoms);
    mol.multiplicity = mult;
    (Cell::new(mol, lat).expect("supercell"), shifts)
}

/// Polarized N-point mesh == the explicit N-cell supercell / N, both through
/// the k-point code (the supercell at 1x1x1 on the replicated grid; there is
/// no open-shell Gamma meta-GGA to compare with). Open shell, multiplicity
/// `(N_a - N_b) N_k + 1`.
#[test]
fn polarized_k_mesh_is_the_supercell() {
    let cell = h3_doublet();
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let cell_grid = PeriodicGrid::build(&cell, &grid_cfg(40, 110)).unwrap();
    let n = [1, 1, 2];
    let nk = 2.0;
    for functional in ["SCAN", "R2SCAN"] {
        let (sc, shifts) = supercell(&cell, n, 3);
        let sc_grid = cell_grid.replicated(&shifts);
        let one = KPointMesh::gamma_centred(&sc, [1, 1, 1]).unwrap();
        let es = kuks_on(&sc, &bs, &one, &sc_grid, functional).scf.energy / nk;
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        let r = solve_kuks_on_grid(
            &cell,
            &prep,
            None,
            &mesh,
            &cell_grid,
            &kuks_cfg(&cell, functional),
        )
        .unwrap();
        eprintln!(
            "H3 doublet {functional} {n:?}: E_k {:.12} E_super/N {es:.12} dE {:.2e}",
            r.scf.energy,
            r.scf.energy - es
        );
        assert!((r.scf.energy - es).abs() < 1e-8, "{functional}");
    }
}

fn pin_cell(name: &str) -> Cell {
    match name {
        "h3" => h3_doublet(),
        "h4t" => h4_triplet(),
        _ => unreachable!(),
    }
}

fn kuks_uniform(
    cell: &Cell,
    n: [usize; 3],
    functional: &str,
    mutation: Option<KUksMggaMutation>,
) -> KUksResult {
    let bs = pyscf_sto3g_h();
    let prep = prep_for(cell, &bs);
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let grid = PeriodicGrid::uniform(cell, [61; 3]).unwrap();
    let mut cfg = kuks_cfg(cell, functional);
    cfg.mgga_mutation = mutation;
    solve_kuks_on_grid(cell, &prep, None, &mesh, &grid, &cfg)
        .unwrap_or_else(|e| panic!("solve_kuks {functional} {n:?}: {e}"))
}

/// ONE polarized PySCF pin that runs on every commit (`h3` SCAN 1x1x1, the
/// cheapest; PySCF 2.13.1 `KUKS`, setup in
/// `scripts/gen_pyscf_kuks_mgga_refs.py`). The other seven pins and the
/// mutant-versus-pin checks live in `validation_pbc_kuks_mgga.rs` (weekly).
/// Measured miss 4.0e-13; bar 1e-9.
#[test]
fn h3_scan_matches_pinned_pyscf() {
    let cell = pin_cell("h3");
    let r = kuks_uniform(&cell, [1, 1, 1], "SCAN", None);
    let e_ref = -1.524551950364;
    eprintln!(
        "h3 SCAN 1x1x1: {:.12} dE {:.2e}",
        r.scf.energy,
        r.scf.energy - e_ref
    );
    assert!((r.scf.energy - e_ref).abs() < 1e-9, "{}", r.scf.energy);
}

/// A one-electron doublet (`N_beta = 0`, an empty beta channel): documented
/// behaviour is refuse-or-finite. The code errors on a non-finite E_xc / V_xc
/// instead of returning one, so the outcome is either a finite energy or a
/// typed error; it must never be NaN.
#[test]
fn empty_beta_channel_is_finite_or_refused() {
    let cell = cell_mult(&[[0.3, 0.2, 0.1]], cubic(4.0), 2);
    let bs = pyscf_sto3g_h();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let grid = PeriodicGrid::build(&cell, &grid_cfg(30, 110)).unwrap();
    let prep = prep_for(&cell, &bs);
    for f in ["SCAN", "R2SCAN"] {
        match solve_kuks_on_grid(&cell, &prep, None, &mesh, &grid, &kuks_cfg(&cell, f)) {
            Ok(r) => {
                eprintln!("H atom {f}: finite E {:.10}", r.scf.energy);
                assert!(r.scf.energy.is_finite() && r.e_xc.is_finite(), "{f}");
            }
            Err(e) => eprintln!("H atom {f}: refused ({e})"),
        }
    }
}

/// Everything outside the validated envelope is refused BY NAME: other
/// meta-GGAs (raw TPSS), meta-GGA with RS-GDF J/K, and (the library gates the
/// Python / CLI bindings share) the open-shell meta-GGA forces and stress
/// outside the wired UKS SCAN / r2SCAN on the dense-AFT J/K: UKS on RS-GDF
/// and every ROKS meta-GGA derivative.
#[test]
fn unvalidated_meta_gga_requests_are_refused() {
    let cell = h3_doublet();
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let grid = PeriodicGrid::build(&cell, &grid_cfg(20, 50)).unwrap();
    // Raw TPSS component names (no SCF exists for them).
    for f in ["MGGA_X_TPSS", "MGGA_C_TPSS"] {
        let e =
            solve_kuks_on_grid(&cell, &prep, None, &mesh, &grid, &kuks_cfg(&cell, f)).expect_err(f);
        assert!(e.to_string().contains("meta-GGA"), "{f}: {e}");
    }
    // RS-GDF J/K (the aux basis is only reached after the gate).
    let mut cfg = kuks_cfg(&cell, "SCAN");
    cfg.kuhf.jk = ferric_pbc::kscf::KJkKind::RsGdf;
    let e = solve_kuks_on_grid(&cell, &prep, Some(&prep), &mesh, &grid, &cfg)
        .expect_err("SCAN with RS-GDF");
    assert!(
        e.to_string().contains("meta-GGA with RS-GDF") && e.to_string().contains("dense"),
        "{e}"
    );
    // Forces / stress: the Python / CLI derivative gate takes the UKS
    // meta-GGA on the dense-AFT J/K only, and ROKS never.
    use ferric_pbc::kxcgrad::{check_kpoint_deriv_functional, DerivSpin};
    for f in ["SCAN", "R2SCAN"] {
        check_kpoint_deriv_functional(DerivSpin::Unrestricted, false, f, "with_gradient")
            .unwrap_or_else(|e| panic!("{f}: {e}"));
        let e = check_kpoint_deriv_functional(DerivSpin::Unrestricted, true, f, "with_gradient")
            .expect_err(f);
        assert!(e.contains("run on the dense-AFT J/K only"), "{f}: {e}");
        let e = check_kpoint_deriv_functional(DerivSpin::RestrictedOpen, false, f, "with_stress")
            .expect_err(f);
        assert!(e.contains("ROKS has no meta-GGA"), "{f}: {e}");
    }
}
