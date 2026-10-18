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
//! * `uniform_grid_kuks_mgga_matches_pinned_pyscf`: PySCF `KUKS` pins.

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
/// open-shell code must reproduce k-RKS, on the 1x1x2 and 2x2x2 meshes.
/// Blind to a tau/spin-channel error by construction (module doc), so it is
/// necessary, not sufficient.
#[test]
fn singlet_k_uks_is_k_rks() {
    let bs = pyscf_sto3g_h();
    for (label, cell, meshes) in [
        ("H2", h2_cell(4.0), vec![[1, 1, 2], [2, 2, 2]]),
        ("H4 tri", triclinic_cell(), vec![[1, 1, 2]]),
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

/// `V_sigma(k)` is the derivative of `E_xc` w.r.t. `D_sigma(k)`: central
/// finite difference of `E_xc` along a Hermitian perturbation of ONE spin
/// density (the other fixed) against `Re Tr(V_sigma Delta)`, at a polarized
/// converged state. Each spin's `V^tau` is checked on its own, so a dropped
/// beta `V^tau` or a swapped channel fails here without any SCF.
#[test]
fn polarized_vxc_is_the_derivative_of_exc() {
    let cell = h3_doublet();
    let bs = pyscf_sto3g_h();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let grid = PeriodicGrid::build(&cell, &grid_cfg(30, 110)).unwrap();
    for functional in ["SCAN", "R2SCAN"] {
        let r = kuks_on(&cell, &bs, &mesh, &grid, functional);
        let mut pxc = KPeriodicXcPolarized::new(
            &cell,
            &bs,
            functional,
            &grid,
            &mesh,
            &PeriodicXcConfig::default(),
        )
        .unwrap();
        let (da, db) = (r.scf.density_alpha.clone(), r.scf.density_beta.clone());
        let (_, va, vb) = pxc.build_polarized(&da, &db).unwrap();
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
        for spin in 0..2 {
            let shift = |s: f64| -> Vec<Array2<Complex64>> {
                let base = if spin == 0 { &da } else { &db };
                base.iter()
                    .zip(&delta)
                    .map(|(d, x)| d + &x.mapv(|z| z * s))
                    .collect()
            };
            let (ep, _, _) = if spin == 0 {
                pxc.build_polarized(&shift(h), &db).unwrap()
            } else {
                pxc.build_polarized(&da, &shift(h)).unwrap()
            };
            let (em, _, _) = if spin == 0 {
                pxc.build_polarized(&shift(-h), &db).unwrap()
            } else {
                pxc.build_polarized(&da, &shift(-h)).unwrap()
            };
            let fd = (ep - em) / (2.0 * h);
            let v = if spin == 0 { &va } else { &vb };
            let nk = mesh.nk() as f64;
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
                / nk;
            eprintln!(
                "{functional} spin {spin}: FD {fd:.10} analytic {an:.10} rel {:.2e}",
                (fd - an).abs() / fd.abs().max(1e-12)
            );
            assert!(
                (fd - an).abs() < 1e-7 * fd.abs(),
                "{functional} spin {spin}: {fd} vs {an}"
            );
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

// ------------------------------------------------------------- PySCF pins

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

/// PySCF 2.13.1 `pbc.dft.KUKS` (AFTDF + UniformGrids 61^3, exxdiv ewald,
/// conv_tol 1e-12, `cell.precision` 1e-12, STO-3G, `mf.nelec = (N_a N_k,
/// N_b N_k)`), generated by `scripts/gen_pyscf_kuks_mgga_refs.py`
/// (2026-10-18; the script's PBE control row reproduces the existing
/// `pbc_kuks.rs` pin to 9e-12). System, functional, mesh, E/cell.
const PYSCF: &[(&str, &str, [usize; 3], f64)] = &[
    ("h3", "SCAN", [1, 1, 1], -1.524551950364),
    ("h3", "R2SCAN", [1, 1, 1], -1.523392968513),
    ("h3", "SCAN", [1, 1, 2], -1.613699448896),
    ("h3", "R2SCAN", [1, 1, 2], -1.611529707848),
    ("h4t", "SCAN", [1, 1, 1], -1.676171388791),
    ("h4t", "R2SCAN", [1, 1, 1], -1.673910059973),
    ("h4t", "SCAN", [1, 1, 2], -1.748364159384),
    ("h4t", "R2SCAN", [1, 1, 2], -1.745764402810),
];

// Bar derivation (tolerance 1e-9): measured worst miss 7.4e-13 (>= 1000x
// headroom for platform rounding). The other side: the weakest mutant
// (`DropBetaVtau`) misses these same pins by 2.1e-6 (H3 1x1x2 SCAN) to 1.1e-4,
// >= 2000x the bar; the SCAN - r2SCAN gaps are 1.2e-3 (H3 1x1x1) and 2.3e-3
// (H4 triplet 1x1x1), >= 1e6x the bar.

#[test]
#[ignore = "heavy: ~2 min release (eight 61^3 polarized SCFs); run with --ignored"]
fn uniform_grid_kuks_mgga_matches_pinned_pyscf() {
    let mut e = std::collections::HashMap::new();
    for &(sys, functional, n, e_ref) in PYSCF {
        let cell = pin_cell(sys);
        let r = kuks_uniform(&cell, n, functional, None);
        eprintln!(
            "{sys} {functional} {n:?}: {:.12} PySCF {e_ref:.12} dE {:.2e} ({} it, S2 {:.4})",
            r.scf.energy,
            r.scf.energy - e_ref,
            r.scf.iterations,
            r.scf.s2
        );
        assert!(
            (r.scf.energy - e_ref).abs() < 1e-9,
            "{sys} {functional} {n:?}: {} vs {e_ref}",
            r.scf.energy
        );
        e.insert((sys, functional, n), r.scf.energy);
    }
    // SCAN and r2SCAN must be told apart by these pins (they are identical on
    // the one-orbital-per-spin cells where alpha = 0).
    for sys in ["h3", "h4t"] {
        let gap = e[&(sys, "SCAN", [1, 1, 1])] - e[&(sys, "R2SCAN", [1, 1, 1])];
        assert!(gap.abs() > 1e-3, "{sys}: SCAN - r2SCAN = {gap:.3e}");
    }
}

/// Mutation ledger (measured 2026-10-18 against the pins above, 61^3 grid; dE
/// = E_mutant - E_correct, Ha):
///
/// ```text
///                        H3 SCAN 1x1x1   H3 r2SCAN 1x1x1  H4 trip SCAN 1x1x1  H4 trip r2SCAN 1x1x1
/// SwapTau                -2.093e-2       -2.201e-2        -6.659e-2           -6.899e-2
/// DropBetaVtau            2.024e-5        2.509e-5         9.523e-5            1.144e-4
/// TotalTauBothChannels    4.916e-2        5.091e-2         4.536e-2            4.756e-2
/// UnpolarizedEvaluator    8.138e-2        8.215e-2         1.471e-1            1.479e-1
/// ```
///
/// (H3 1x1x2: SwapTau -2.0e-2, DropBetaVtau 2.1e-6 (SCAN) / 2.7e-6 (r2SCAN),
/// TotalTau 5.6e-2, Unpolarized 5.5e-2; H4 triplet 1x1x2 DropBetaVtau 5.4e-5.)
/// Every mutant is far outside the 1e-9 pin bar; this test requires >= 1e-6
/// (the smallest measured miss is 2.0e-5 on the cells run here, 20x above
/// the requirement and 1000x above the pin bar). None of them is visible in
/// the singlet limit (`singlet_k_uks_is_k_rks` cannot see them: `tau_a ==
/// tau_b`), which is why they are run on the POLARIZED cells.
#[test]
#[ignore = "heavy: ~4 min release (sixteen 61^3 polarized SCFs); run with --ignored"]
fn tau_mutants_fail_the_polarized_pins() {
    for &(sys, functional, n, e_ref) in PYSCF {
        if n != [1, 1, 1] {
            continue;
        }
        let cell = pin_cell(sys);
        for m in [
            KUksMggaMutation::SwapTau,
            KUksMggaMutation::DropBetaVtau,
            KUksMggaMutation::TotalTauBothChannels,
            KUksMggaMutation::UnpolarizedEvaluator,
        ] {
            let r = kuks_uniform(&cell, n, functional, Some(m));
            let miss = (r.scf.energy - e_ref).abs();
            eprintln!("{sys} {functional} {n:?} {m:?}: miss {miss:.3e}");
            assert!(
                miss > 1e-6,
                "{sys} {functional} {m:?} not detected: {miss:.3e}"
            );
        }
    }
}

/// Everything outside the validated envelope is refused BY NAME: other
/// meta-GGAs (raw TPSS), meta-GGA with RS-GDF J/K, and (library gates the
/// Python / CLI bindings share) open-shell forces and stress.
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
    // Forces / stress: the open-shell allowlist has no meta-GGA.
    for f in ["SCAN", "R2SCAN"] {
        let e = ferric_pbc::kxcgrad::check_open_shell_force_functional(f).expect_err(f);
        assert!(e.to_string().contains("no meta-GGA"), "{f}: {e}");
        let e = ferric_pbc::kxcgrad::check_kpoint_deriv_functional(true, f, "with_gradient")
            .expect_err(f);
        assert!(e.contains("no meta-GGA"), "{f}: {e}");
    }
}
