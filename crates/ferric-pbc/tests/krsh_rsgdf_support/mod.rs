//! Shared fixtures and helpers of `pbc_krsh_rsgdf.rs` (per commit) and
//! `validation_pbc_krsh_rsgdf.rs` (weekly): the RS-GDF / dense range-separated
//! k-RKS drivers, the fit-error scale, the PySCF pins and the anchor bodies.
#![allow(dead_code, unused_imports)]

use crate::common::*;
pub use ferric_core::basis::{self, BasisSet};
pub use ferric_integrals::basis_bridge::PreparedBasis;
pub use ferric_integrals::site_basis::SiteBasis;
pub use ferric_pbc::dense_aft::ExxDiv;
pub use ferric_pbc::dft::{PeriodicGrid, PeriodicGridConfig};
pub use ferric_pbc::hcore::kpoint::periodic_hcore_kpts;
pub use ferric_pbc::hcore::PeriodicHcoreConfig;
pub use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
pub use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig, KRksResult};
pub use ferric_pbc::kpts::KPointMesh;
pub use ferric_pbc::kscf::KJkKind;
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::rsgdf::kpoint::{
    KRsGdf, KRsGdfConfig, KRsGdfMutation, KRshMutant, DEFAULT_K_FITTED_KERNELS_MAX_BYTES,
};
pub use ferric_pbc::rsgdf::RsGdfConfig;
pub use ferric_pbc::rsh::RshParams;
pub use num_complex::Complex64;

pub const OMEGA_HCORE: f64 = 0.8;
pub const ANCHOR_ALPHA: f64 = 0.5;
pub const AMPLE: usize = 1 << 30;

pub fn gdf_cfg(exxdiv: ExxDiv, mutation: Option<KRsGdfMutation>) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 0.8,
            exxdiv,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation,
    }
}

/// The 24 anchor aux centres of `pbc_krsgdf.rs` (one s per H pair midpoint
/// class): the fit is exact for the full Coulomb kernel there.
pub fn anchor_sites(cell: &Cell) -> Vec<[f64; 4]> {
    let r = cell.positions();
    let a = cell.lattice();
    let mut out = Vec::new();
    for (i, j) in [(0usize, 0usize), (1, 1), (0, 1)] {
        for k in 0..8usize {
            let h = [(k >> 2) & 1, (k >> 1) & 1, k & 1];
            let mut c = [0.0; 3];
            for d in 0..3 {
                let ha: f64 = (0..3).map(|x| h[x] as f64 * a[x][d]).sum();
                c[d] = 0.5 * (r[i][d] + r[j][d] + ha);
            }
            out.push([c[0], c[1], c[2], 2.0 * ANCHOR_ALPHA]);
        }
    }
    out
}

pub fn cmax(a: &ndarray::Array2<Complex64>, b: &ndarray::Array2<Complex64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).norm()))
}

/// The mixed exchange-divergence constant from its closed forms, computed
/// here independently of `RshParams::madelung`: the full-Coulomb constant of
/// the diag(N) supercell (`ewald::madelung_constant`, pinned to PySCF in
/// `pbc_krsh.rs`) and `v_M^{LR}(w) = 2w/sqrt(pi) - (4 pi/Omega) sum_{G != 0}
/// e^{-G^2/4w^2}/G^2` summed here.
pub fn closed_form_mixed_madelung(cell: &Cell, mesh: &KPointMesh, r: RshParams) -> f64 {
    let sc = Cell::new(cell.mol().clone(), mesh.supercell_lattice()).unwrap();
    let w = r.omega;
    let gcut = 2.0 * w * (20.0 * std::f64::consts::LN_10).sqrt();
    let mut sum = 0.0;
    for g in sc.gvectors(gcut).unwrap() {
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        if g2 > 1e-12 {
            sum += (-g2 / (4.0 * w * w)).exp() / g2;
        }
    }
    let lr = 2.0 * w / std::f64::consts::PI.sqrt() - 4.0 * std::f64::consts::PI / sc.volume() * sum;
    r.c_sr * ferric_pbc::ewald::madelung_constant(&sc).unwrap() + (r.c_lr - r.c_sr) * lr
}

/// Max |ΔK| of a PRODUCTION `jk_builder().build` (add_k_blocks +
/// mix_rsh_exchange + Madelung) against the dense builder, on fixed
/// Hermitian per-k densities.
pub fn production_k_gap(mesh: &KPointMesh, dense: &KDenseAftEri, gdf: &KRsGdf, nao: usize) -> f64 {
    use ferric_pbc::kscf::KPointJk;
    let nk = mesh.nk();
    let dm: Vec<ndarray::Array2<Complex64>> = (0..nk)
        .map(|k| {
            let mut d = ndarray::Array2::<Complex64>::zeros((nao, nao));
            for m in 0..nao {
                for l in 0..=m {
                    let v = Complex64::new(
                        0.3 + 0.1 * (m + 2 * l + k) as f64,
                        if m == l { 0.0 } else { 0.05 * (k + 1) as f64 },
                    );
                    d[(m, l)] = v;
                    d[(l, m)] = v.conj();
                }
            }
            d
        })
        .collect();
    let zero = || vec![ndarray::Array2::<Complex64>::zeros((nao, nao)); nk];
    let (mut jd, mut kd, mut jg, mut kg) = (zero(), zero(), zero(), zero());
    dense.jk_builder().build(&dm, &mut jd, &mut kd).unwrap();
    gdf.jk_builder().build(&dm, &mut jg, &mut kg).unwrap();
    kd.iter()
        .zip(&kg)
        .fold(0.0_f64, |m, (a, b)| m.max(cmax(a, b)))
}

/// `(max |ΔK kernel|, |v_M - closed form|, |K|, production-path max |ΔK|)` of
/// the fitted RSH K vs the dense RSH K over every (k,k'). The kernel
/// comparison goes through `fitted_kernels` / `mixed_k_kernel`; the last entry
/// is the production contraction.
pub fn kernel_gap(
    n: [usize; 3],
    rsh: RshParams,
    exxdiv: ExxDiv,
    mutation: Option<KRsGdfMutation>,
) -> (f64, f64, f64, f64) {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    let hk = periodic_hcore_kpts(
        &cell,
        &prep,
        &mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA_HCORE),
    )
    .unwrap();
    let dense = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        exxdiv,
        &KDenseAftConfig {
            rsh: Some(rsh),
            ..Default::default()
        },
    )
    .unwrap();
    let site = SiteBasis::new(&anchor_sites(&cell), 0).unwrap();
    let gdf = KRsGdf::build_rsh(
        &cell,
        &prep,
        &site.prep,
        &mesh,
        &hk.s,
        &gdf_cfg(exxdiv, mutation),
        rsh,
    )
    .unwrap();
    let (_, kf) = gdf
        .fitted_kernels(DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
        .unwrap();
    let nk = mesh.nk();
    let mut worst = 0.0_f64;
    for k in 0..nk {
        for kp in 0..nk {
            worst = worst.max(cmax(&kf[k * nk + kp], dense.kker(k, kp)));
        }
    }
    let expect = match exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => closed_form_mixed_madelung(&cell, &mesh, rsh),
    };
    let dv = (gdf.madelung() - expect).abs();
    let prod = production_k_gap(&mesh, &dense, &gdf, 2);
    let scale = (0..nk)
        .flat_map(|k| (0..nk).map(move |kp| (k, kp)))
        .map(|(k, kp)| {
            dense
                .kker(k, kp)
                .iter()
                .fold(0.0_f64, |m, z| m.max(z.norm()))
        })
        .fold(0.0_f64, f64::max);
    (worst, dv, scale, prod)
}

pub const HSE_LIKE: (f64, f64) = (0.25, 0.0);
pub const CAM_LIKE: (f64, f64) = (0.19, 0.65);

pub fn rsh(omega: f64, f: (f64, f64)) -> RshParams {
    RshParams {
        omega,
        c_sr: f.0,
        c_lr: f.1,
    }
}

/// Fit-free bar of the trivial-aux anchor (measured 2e-13 at |K| ~ 0.4).
pub const KERNEL_BAR: f64 = 1e-11;
/// Smallest kernel / Madelung shift any mutant produces (measured 0.157 /
/// 0.10); the bar sits two decades below, five above the production residual.
pub const MUTANT_BAR: f64 = 1e-3;

pub fn krks_cfg(
    cell: &Cell,
    functional: &str,
    aux: bool,
    mutation: Option<KRsGdfMutation>,
) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, ExxDiv::Ewald, functional);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    cfg.krhf.scf.energy_conv = 1e-12;
    cfg.krhf.scf.grad_conv = 1e-9;
    if aux {
        cfg.krhf.jk = KJkKind::RsGdf;
        cfg.krhf.rsgdf = gdf_cfg(ExxDiv::Ewald, mutation);
    }
    cfg
}

pub fn run(
    cell: &Cell,
    bs: &BasisSet,
    aux_name: Option<&str>,
    mesh: &KPointMesh,
    grid: &PeriodicGrid,
    functional: &str,
    mutation: Option<KRsGdfMutation>,
) -> KRksResult {
    let prep = prep_for(cell, bs);
    let aux =
        aux_name.map(|n| PreparedBasis::new(cell.mol(), &basis::bundled(n).unwrap()).unwrap());
    let cfg = krks_cfg(cell, functional, aux.is_some(), mutation);
    solve_krks_on_grid(cell, &prep, aux.as_ref(), mesh, grid, &cfg)
        .unwrap_or_else(|e| panic!("solve_krks {functional}: {e}"))
}

pub const GRID: usize = 32;
/// The RSH fit error must sit within this fraction of the plain-Coulomb
/// (PBE0) RS-GDF-vs-dense energy gap of the same aux basis and mesh. A defect
/// moves E by >= 1.7e-2 against fit scales <= 1e-4 (`mutants_break_the_energy`).
pub const FIT_BAR_RATIO: f64 = 0.4;

/// Energy gap RS-GDF - dense on the same grid.
pub fn fit_gap(cell: &Cell, mesh: &KPointMesh, aux: &str, f: &str) -> f64 {
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::uniform(cell, [GRID; 3]).unwrap();
    let d = run(cell, &bs, None, mesh, &grid, f, None).scf.energy;
    run(cell, &bs, Some(aux), mesh, &grid, f, None).scf.energy - d
}

pub fn sc_cell(cell: &Cell, n: [usize; 3]) -> (Cell, Vec<[f64; 3]>) {
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

/// E/cell of the N-point mesh on the primitive cell minus the Gamma energy of
/// the N-cell supercell / N, both RS-GDF (aux on every image); `mutation`
/// acts on the mesh side only.
pub fn supercell_gap(f: &str, n: [usize; 3], mutation: Option<KRsGdfMutation>) -> f64 {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let nk = (n[0] * n[1] * n[2]) as f64;
    let grid = PeriodicGrid::build(&cell, &PeriodicGridConfig::with_size(40, 110)).unwrap();
    let (sc, shifts) = sc_cell(&cell, n);
    let sc_grid = grid.replicated(&shifts);
    let sc_mesh = KPointMesh::gamma_centred(&sc, [1, 1, 1]).unwrap();
    let e_sc = run(&sc, &bs, Some("cc-pvdz-ri"), &sc_mesh, &sc_grid, f, None)
        .scf
        .energy
        / nk;
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    run(&cell, &bs, Some("cc-pvdz-ri"), &mesh, &grid, f, mutation)
        .scf
        .energy
        - e_sc
}

// PySCF 2.13 pbc.dft.KRKS pins of `pbc_krsh.rs` (AFTDF, UniformGrids 61^3,
// exxdiv ewald, STO-3G H2 a = 4 Bohr); 2x2x2 meshes omitted for the reason
// stated there (PySCF's attenuated AFTDF K did not finish in 10 minutes).
pub const PYSCF: [(&str, [usize; 3], f64); 6] = [
    ("HSE06", [1, 1, 1], -1.575428343697),
    ("HSE06", [1, 1, 2], -1.347311737653),
    ("HYB_GGA_XC_CAM_B3LYP", [1, 1, 1], -1.591829863303),
    ("HYB_GGA_XC_CAM_B3LYP", [1, 1, 2], -1.354469613012),
    ("HYB_GGA_XC_WB97X", [1, 1, 1], -1.605512318088),
    ("HYB_GGA_XC_WB97X", [1, 1, 2], -1.363565753524),
];

/// Cross-surface pin shared with `test_pbc_krks.py::
/// test_range_separated_hybrid_runs_on_rsgdf_energy_only` (the Python binding
/// produced the number; the library reproduces it on the same input): H2 a = 4
/// Bohr, 1x1x2 Gamma mesh, cc-pVDZ-RI aux, hcore omega 0.8, atom-centred grid
/// (40, 110), HSE06.
pub const HSE06_RSGDF_112_ATOM_GRID: f64 = -1.3484931727;

/// RS-GDF vs dense RSH SCF energy on the same grid, for every mesh x aux x
/// functional: the gap must sit within `FIT_BAR_RATIO` of the plain-Coulomb
/// (PBE0) fit error of the same aux and mesh. The weekly test runs both
/// meshes, both aux sets and three functionals; the per-commit canary one
/// mesh, one aux and two functionals.
pub fn fit_error_case(meshes: &[[usize; 3]], auxes: &[&str], fs: &[&str]) {
    let cell = h2_cell(4.0);
    for &n in meshes {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        for &aux in auxes {
            // The plain-Coulomb fit error of the same aux/mesh sets the scale.
            let scale = fit_gap(&cell, &mesh, aux, "PBE0");
            assert!(scale.abs() > 1e-7, "fit error scale is not measurable");
            for &f in fs {
                let d = fit_gap(&cell, &mesh, aux, f);
                eprintln!(
                    "{n:?} {aux} {f}: dE {d:.3e} (plain-Coulomb PBE0 fit error {scale:.3e}, ratio {:.3})",
                    d / scale
                );
                assert!(
                    (d - scale).abs() < FIT_BAR_RATIO * scale.abs(),
                    "{n:?} {aux} {f}: {d:e} vs plain {scale:e}"
                );
            }
        }
    }
}

/// RS-GDF (cc-pVDZ-RI) energies against the PySCF pins in `rows`: E - pin
/// must be the plain-Coulomb fit error of that aux at the same mesh to within
/// `FIT_BAR_RATIO` (the dense energy equals the pin to 1e-9; a defect moves E
/// by >= 1.7e-2 against scales <= 3e-4).
pub fn pyscf_pin_case(rows: &[(&str, [usize; 3], f64)]) {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::uniform(&cell, [61; 3]).unwrap();
    for &(f, n, e_ref) in rows {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        let scale = fit_gap(&cell, &mesh, "cc-pvdz-ri", "PBE0");
        let e = run(&cell, &bs, Some("cc-pvdz-ri"), &mesh, &grid, f, None)
            .scf
            .energy;
        eprintln!(
            "{f} {n:?}: E {e:.12} pyscf {e_ref:.12} dE {:.2e} (plain fit scale {scale:.2e})",
            e - e_ref
        );
        assert!(
            (e - e_ref - scale).abs() < FIT_BAR_RATIO * scale.abs(),
            "{f} {n:?}"
        );
    }
}
