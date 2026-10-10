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

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{
    gamma_rks, GammaRksConfig, PeriodicGrid, PeriodicGridConfig, PeriodicXc, PeriodicXcConfig,
};
use ferric_pbc::hcore::kpoint::periodic_hcore_kpts;
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdft::{solve_krks, solve_krks_on_grid, KPeriodicXc, KRksConfig, KRksResult};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::lattice::Cell;
use ferric_pbc::uhf::GammaUhfIntegrals;
use ferric_scf::screening::SchwarzBounds;

const OMEGA: f64 = 0.8;

// PySCF 2.13 pbc.dft.KRKS, gamma-centred mesh, AFTDF mesh 61^3,
// exxdiv='ewald', UniformGrids(61^3), conv_tol 1e-12, STO-3G H2 a = 4
// (scratchpad `ref/krks.py`, 2026-10-09).
const PYSCF: [(&str, [usize; 3], f64); 9] = [
    ("LDA", [1, 1, 1], -1.521492168319),
    ("LDA", [1, 1, 2], -1.313021155290),
    ("LDA", [2, 2, 2], -1.054361178633),
    ("PBE", [1, 1, 1], -1.527136159909),
    ("PBE", [1, 1, 2], -1.325652288662),
    ("PBE", [2, 2, 2], -1.080236280438),
    ("PBE0", [1, 1, 1], -1.576992652863),
    ("PBE0", [1, 1, 2], -1.347642203115),
    ("PBE0", [2, 2, 2], -1.086899600638),
];

fn krks_cfg(cell: &Cell, functional: &str, exxdiv: ExxDiv) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, exxdiv, functional);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-12;
    cfg.krhf.scf.grad_conv = 1e-9;
    cfg
}

fn grid_cfg(n_rad: usize, n_ang: usize) -> PeriodicGridConfig {
    PeriodicGridConfig::with_size(n_rad, n_ang)
}

fn krks_uniform(
    cell: &Cell,
    bs: &BasisSet,
    mesh: &KPointMesh,
    functional: &str,
    n: usize,
) -> KRksResult {
    let prep = prep_for(cell, bs);
    let grid = PeriodicGrid::uniform(cell, [n; 3]).unwrap();
    let cfg = krks_cfg(cell, functional, ExxDiv::Ewald);
    solve_krks_on_grid(cell, &prep, None, mesh, &grid, &cfg)
        .unwrap_or_else(|e| panic!("solve_krks {functional}: {e}"))
}

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

// ----------------------------------------------------- k-mesh = supercell

/// Explicit diag(n) supercell and the cell-translation vectors of its copies
/// (atoms of cell m at `R + Σ m_i a_i`, m0 outer).
fn supercell(cell: &Cell, n: [usize; 3]) -> (Cell, Vec<[f64; 3]>) {
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
    (
        Cell::new(hydrogens(&atoms), lat).expect("supercell"),
        shifts,
    )
}

/// Gamma RKS energy of `sc` on an explicit grid (`gamma_rks` with the grid
/// supplied), dense AFT J/K.
fn gamma_rks_on_grid(sc: &Cell, bs: &BasisSet, grid: &PeriodicGrid, functional: &str) -> f64 {
    use ferric_pbc::ewald::madelung_constant;
    use ferric_scf::rhf::{solve_rhf_injected, PeriodicInjection};
    let prep = prep_for(sc, bs);
    let hc = periodic_hcore(sc, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let eri = DenseAftEri::build(
        sc,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let pxc = PeriodicXc::new(sc, bs, functional, grid, &PeriodicXcConfig::default()).unwrap();
    let vm = madelung_constant(sc).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let inj = PeriodicInjection {
        s: hc.s.clone(),
        h: hc.h.clone(),
        vnn: hc.enn,
        j: Box::new(eri.j_builder()),
        k: Box::new(eri.k_builder_with_madelung(vm)),
        xc: Some(Box::new(pxc)),
    };
    let mut scf = GammaRksConfig::new(functional).scf;
    scf.density_conv = 1e-11;
    let r = solve_rhf_injected(
        &ParallelContext::default(),
        sc.mol(),
        &prep,
        op,
        &bounds,
        &scf,
        inj,
    )
    .unwrap();
    assert!(r.converged, "supercell {functional} did not converge");
    r.energy
}

fn supercell_anchor(name: &str, cell: &Cell, bs: &BasisSet, n: [usize; 3], functional: &str) {
    let nk = (n[0] * n[1] * n[2]) as f64;
    let prep = prep_for(cell, bs);
    let cell_grid = PeriodicGrid::build(cell, &grid_cfg(40, 110)).unwrap();
    let (sc, shifts) = supercell(cell, n);
    let sc_grid = cell_grid.replicated(&shifts);
    let e_sc = gamma_rks_on_grid(&sc, bs, &sc_grid, functional) / nk;
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let cfg = krks_cfg(cell, functional, ExxDiv::Ewald);
    let r = solve_krks_on_grid(cell, &prep, None, &mesh, &cell_grid, &cfg).unwrap();
    eprintln!(
        "{name} {n:?} {functional}: E_k {:.12} E_super/N {e_sc:.12} dE {:.2e} ({} it)",
        r.scf.energy,
        r.scf.energy - e_sc,
        r.scf.iterations
    );
    assert!((r.scf.energy - e_sc).abs() < 1e-8, "{name} {functional}");
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

// ------------------------------------------------------------- meta-GGA

// PySCF 2.13 pbc.dft.KRKS, same setup as `PYSCF` above, xc = SCAN / R2SCAN
// (scratchpad `ref/kmgga.py`, 2026-10-09).
const PYSCF_MGGA: [(&str, [usize; 3], f64); 6] = [
    ("SCAN", [1, 1, 1], -1.557880049858),
    ("SCAN", [1, 1, 2], -1.343585556975),
    ("SCAN", [2, 2, 2], -1.085236512624),
    ("R2SCAN", [1, 1, 1], -1.557880049858),
    ("R2SCAN", [1, 1, 2], -1.343079341892),
    ("R2SCAN", [2, 2, 2], -1.084507483367),
];

#[test]
fn mgga_uniform_grid_krks_matches_pinned_pyscf() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    for &(functional, n, e_ref) in &PYSCF_MGGA {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        let r = krks_uniform(&cell, &bs, &mesh, functional, 61);
        eprintln!(
            "H2 {functional} {n:?}: {:.12} PySCF {e_ref:.12} dE {:.2e}",
            r.scf.energy,
            r.scf.energy - e_ref
        );
        assert!(
            (r.scf.energy - e_ref).abs() < 1e-7,
            "{functional} {n:?}: {} vs {e_ref}",
            r.scf.energy
        );
    }
}

/// Meta-GGA k-mesh == supercell / N: the supercell is solved by the same
/// k-point code at a 1x1x1 mesh on the replicated grid (the Gamma periodic
/// path has no meta-GGA).
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
    for functional in ["SCAN", "R2SCAN"] {
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
