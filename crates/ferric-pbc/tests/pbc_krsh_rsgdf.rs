//! Range-separated exact exchange on the k-point RS-GDF J/K
//! (`ferric_pbc::rsgdf::kpoint::klr`, `KRsGdf::build_rsh`).
//!
//! # Physics hypothesis / artifact hypothesis (written before measuring)
//!
//! `K = c_sr K[1/r] + (c_lr − c_sr) K[erf(ωr)/r] + v_M^{mix} S D S`, the erf
//! piece fitted in G-space only. If the construction is right, the fitted
//! kernels equal the dense-AFT RSH kernels to the aux-fit error (zero in the
//! trivial-aux limit), the E/cell of an N-point mesh equals the Gamma energy
//! of the N-cell supercell / N, and PySCF's pinned RSH energies are
//! reproduced to the fit error. If the build is broken in one of the ways the
//! mutants model (no damping, wrong ω units, erf/erfc swapped, no or
//! primitive-cell Madelung), the kernel-level anchors move by O(1e-3..1e-1)
//! and the supercell anchor by the Madelung or kernel difference. The
//! anchors that can NOT see a defect: the `ω → 0` anchor sees only the
//! Madelung bookkeeping (the erf K set is empty, so the attenuation, the ω
//! units and the role swap are invisible there), and the equal-fraction
//! anchor sees nothing about the erf piece (its coefficient is 0).
//!
//! The kernel-level anchors compare `fitted_kernels` (`mixed_k_kernel`); the
//! production contraction (`add_k_blocks` + `mix_rsh_exchange` + Madelung) is
//! checked against the dense builder on fixed densities by the `prod` column of
//! `kernel_gap`. The Madelung constant is compared with a closed form summed in
//! this file, not with a second call of `RshParams::madelung`.
//!
//! Anchors:
//! * `kernel_matches_dense_in_the_trivial_aux_limit` — fitted `K` vs the
//!   dense RSH `K` (same layout), non-TRIM 1×1×3 mesh, HSE-like and
//!   CAM-like fractions, plus the mixed Madelung constant.
//! * `small_omega_is_the_global_hybrid_without_exxdiv` — `ω → 0`,
//!   `ExxDiv::None`: RSH ≡ global hybrid `c_sr`; with Ewald the difference is
//!   exactly `(c_lr − c_sr) v_M^{LR}(ω) ~ 2ω/√π`, not 0.
//! * `rsgdf_rsh_is_the_dense_rsh_to_the_fit_error` — SCF energies, HSE06 and
//!   CAM-B3LYP, RS-GDF vs dense on the same grid; the difference must be of
//!   the order of the plain-Coulomb (PBE0) RS-GDF-vs-dense difference.
//! * `rsgdf_rsh_k_mesh_is_the_supercell` — 1×1×3 mesh E/cell vs the Gamma
//!   energy of the 3-cell supercell / 3 (both RS-GDF).
//! * `rsgdf_rsh_matches_pinned_pyscf` — PySCF 2.13 KRKS pins of `pbc_krsh`.
//! * `rsgdf_rsh_mutations_break_the_anchors`, the derivative source checks
//!   (the force and stress themselves: `pbc_krsh_rsgdf_force.rs`,
//!   `validation_krsh_rsgdf_force.rs`).

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::{PeriodicGrid, PeriodicGridConfig};
use ferric_pbc::hcore::kpoint::periodic_hcore_kpts;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig, KRksResult};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::KJkKind;
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{
    KRsGdf, KRsGdfConfig, KRsGdfMutation, KRshMutant, DEFAULT_K_FITTED_KERNELS_MAX_BYTES,
};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::rsh::RshParams;
use num_complex::Complex64;

const OMEGA_HCORE: f64 = 0.8;
const ANCHOR_ALPHA: f64 = 0.5;
const AMPLE: usize = 1 << 30;

fn gdf_cfg(exxdiv: ExxDiv, mutation: Option<KRsGdfMutation>) -> KRsGdfConfig {
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
fn anchor_sites(cell: &Cell) -> Vec<[f64; 4]> {
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

fn cmax(a: &ndarray::Array2<Complex64>, b: &ndarray::Array2<Complex64>) -> f64 {
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
fn closed_form_mixed_madelung(cell: &Cell, mesh: &KPointMesh, r: RshParams) -> f64 {
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
fn production_k_gap(mesh: &KPointMesh, dense: &KDenseAftEri, gdf: &KRsGdf, nao: usize) -> f64 {
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
fn kernel_gap(
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

const HSE_LIKE: (f64, f64) = (0.25, 0.0);
const CAM_LIKE: (f64, f64) = (0.19, 0.65);

fn rsh(omega: f64, f: (f64, f64)) -> RshParams {
    RshParams {
        omega,
        c_sr: f.0,
        c_lr: f.1,
    }
}

#[test]
fn kernel_matches_dense_in_the_trivial_aux_limit() {
    for (omega, f) in [
        (0.11, HSE_LIKE),
        (0.2, HSE_LIKE),
        (0.2, CAM_LIKE),
        (0.35, CAM_LIKE),
    ] {
        for exxdiv in [ExxDiv::None, ExxDiv::Ewald] {
            let (d, dv, scale, prod) = kernel_gap([1, 1, 3], rsh(omega, f), exxdiv, None);
            eprintln!(
                "trivial aux 1x1x3 w={omega} f={f:?} {exxdiv:?}: max|dK| {d:.2e} (|K| {scale:.2e}), |d v_M| {dv:.1e}"
            );
            assert!(d < KERNEL_BAR, "w={omega} {f:?} {exxdiv:?}: {d:e}");
            // v_M against the test's own closed-form sum (not RshParams::madelung).
            assert!(dv < 1e-10, "w={omega} {f:?} {exxdiv:?}: v_M {dv:e}");
            assert!(
                prod < KERNEL_BAR,
                "w={omega} {f:?} {exxdiv:?}: production K {prod:e}"
            );
            assert!(scale > 0.1, "kernel is not trivially small: {scale}");
        }
    }
}

/// Fit-free bar of the trivial-aux anchor (measured 2e-13 at |K| ~ 0.4).
const KERNEL_BAR: f64 = 1e-11;
/// Smallest kernel / Madelung shift any mutant produces (measured 0.157 /
/// 0.10); the bar sits two decades below, five above the production residual.
const MUTANT_BAR: f64 = 1e-3;

#[test]
fn kernel_anchor_sees_every_mutant() {
    use KRshMutant::*;
    let r = rsh(0.2, CAM_LIKE);
    for m in [
        NoAttenuation,
        OmegaUnits,
        SwappedRoles,
        NoMadelung,
        PrimitiveMadelung,
    ] {
        let (d, dv, _, prod) =
            kernel_gap([1, 1, 3], r, ExxDiv::Ewald, Some(KRsGdfMutation::Rsh(m)));
        eprintln!("{m:?}: max|dK| {d:.2e}, |d v_M| {dv:.2e}");
        // Madelung mutants shift v_M (and the production K by v_M S D S);
        // kernel mutants shift the fitted kernel and the production K.
        let seen = match m {
            NoMadelung | PrimitiveMadelung => dv > MUTANT_BAR && prod > MUTANT_BAR,
            _ => d > MUTANT_BAR && prod > MUTANT_BAR,
        };
        assert!(seen, "{m:?} left the anchor green (dK {d:e}, dvM {dv:e})");
    }
}

/// `ω → 0`: no K vector lies inside the erf Gaussian cut, so the fitted RSH
/// kernel is exactly `c_sr` times the full-Coulomb fit, i.e. the global
/// hybrid; with Ewald the only difference is the constant
/// `(c_lr − c_sr) v_M^{LR}(ω)`, which tends to 0 only linearly (`2ω/√π`).
#[test]
fn small_omega_is_the_global_hybrid_without_exxdiv() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let hk = periodic_hcore_kpts(
        &cell,
        &prep,
        &mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA_HCORE),
    )
    .unwrap();
    let site = SiteBasis::new(&anchor_sites(&cell), 0).unwrap();
    let omega = 0.01;
    let r = rsh(omega, CAM_LIKE);
    for exxdiv in [ExxDiv::None, ExxDiv::Ewald] {
        let cfg = gdf_cfg(exxdiv, None);
        let plain = KRsGdf::build(&cell, &prep, &site.prep, &mesh, &hk.s, &cfg).unwrap();
        let mix = KRsGdf::build_rsh(&cell, &prep, &site.prep, &mesh, &hk.s, &cfg, r).unwrap();
        let (_, kp) = plain
            .fitted_kernels(DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
            .unwrap();
        let (_, km) = mix
            .fitted_kernels(DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
            .unwrap();
        let mut worst = 0.0_f64;
        for (a, b) in kp.iter().zip(&km) {
            worst = worst.max(cmax(&b.mapv(|z| z / r.c_sr), a));
        }
        let sc = Cell::new(cell.mol().clone(), mesh.supercell_lattice()).unwrap();
        let lr = ferric_pbc::rsh::madelung_lr(&sc, omega).unwrap();
        let want = match exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => closed_form_mixed_madelung(&cell, &mesh, r),
        };
        eprintln!(
            "w={omega} {exxdiv:?}: max|K_rsh/c_sr - K_full| {worst:.2e}, v_M {:.12} (formula {want:.12}), v_LR {lr:.3e}",
            mix.madelung()
        );
        assert!(worst < 1e-13, "{exxdiv:?}: {worst:e}");
        assert!((mix.madelung() - want).abs() < 1e-10, "{exxdiv:?}");
        if exxdiv == ExxDiv::Ewald {
            // The erf self-term does not vanish at finite omega: v_M is NOT the
            // global hybrid's c_sr v_M (this is physics, not an artifact).
            assert!((lr - 2.0 * omega / std::f64::consts::PI.sqrt()).abs() < 1e-6 * lr);
            assert!((mix.madelung() - r.c_sr * plain.madelung_ewald()).abs() > 1e-3 * lr);
        }
    }
}

fn krks_cfg(
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

fn run(
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

const GRID: usize = 32;
/// The RSH fit error must sit within this fraction of the plain-Coulomb
/// (PBE0) RS-GDF-vs-dense energy gap of the same aux basis and mesh. A defect
/// moves E by >= 1.7e-2 against fit scales <= 1e-4 (`mutants_break_the_energy`).
const FIT_BAR_RATIO: f64 = 0.4;

/// Energy gap RS-GDF - dense on the same grid.
fn fit_gap(cell: &Cell, mesh: &KPointMesh, aux: &str, f: &str) -> f64 {
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::uniform(cell, [GRID; 3]).unwrap();
    let d = run(cell, &bs, None, mesh, &grid, f, None).scf.energy;
    run(cell, &bs, Some(aux), mesh, &grid, f, None).scf.energy - d
}

#[test]
fn rsgdf_rsh_is_the_dense_rsh_to_the_fit_error() {
    let cell = h2_cell(4.0);
    for n in [[1, 1, 1], [1, 1, 3]] {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        for aux in ["cc-pvdz-ri", "def2-universal-jkfit"] {
            // The plain-Coulomb fit error of the same aux/mesh sets the scale.
            let scale = fit_gap(&cell, &mesh, aux, "PBE0");
            assert!(scale.abs() > 1e-7, "fit error scale is not measurable");
            for f in ["HSE06", "HYB_GGA_XC_CAM_B3LYP", "HYB_GGA_XC_WB97X"] {
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

#[test]
fn mutants_break_the_energy() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::uniform(&cell, [GRID; 3]).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    for f in ["HSE06", "HYB_GGA_XC_CAM_B3LYP"] {
        let d = run(&cell, &bs, None, &mesh, &grid, f, None).scf.energy;
        for m in [
            KRshMutant::NoAttenuation,
            KRshMutant::OmegaUnits,
            KRshMutant::SwappedRoles,
            KRshMutant::NoMadelung,
            KRshMutant::PrimitiveMadelung,
        ] {
            let e = run(
                &cell,
                &bs,
                Some("cc-pvdz-ri"),
                &mesh,
                &grid,
                f,
                Some(KRsGdfMutation::Rsh(m)),
            )
            .scf
            .energy;
            eprintln!("{f} {m:?}: E - E_dense {:.3e}", e - d);
            assert!(
                (e - d).abs() > 1e-3,
                "{f} {m:?} left the energy anchor green"
            );
        }
    }
}

fn sc_cell(cell: &Cell, n: [usize; 3]) -> (Cell, Vec<[f64; 3]>) {
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
fn supercell_gap(f: &str, n: [usize; 3], mutation: Option<KRsGdfMutation>) -> f64 {
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

/// The k-mesh/supercell identity is a statement about the fit construction
/// (the mesh fit IS the supercell Gamma fit), so it holds to round-off, not to
/// the fit error. Non-TRIM 1x1x3 (`-1 != 1 mod 3`) and the TRIM 1x1x2.
#[test]
fn rsgdf_rsh_k_mesh_is_the_supercell() {
    for f in ["HSE06", "HYB_GGA_XC_CAM_B3LYP"] {
        for n in [[1, 1, 2], [1, 1, 3]] {
            let d = supercell_gap(f, n, None);
            eprintln!("{f} {n:?}: E_k - E_super/N {d:.2e}");
            assert!(d.abs() < 1e-8, "{f} {n:?}: {d:e}");
        }
    }
}

#[test]
fn supercell_anchor_sees_every_mutant() {
    use KRshMutant::*;
    for m in [
        NoAttenuation,
        OmegaUnits,
        SwappedRoles,
        NoMadelung,
        PrimitiveMadelung,
    ] {
        let d = supercell_gap(
            "HYB_GGA_XC_CAM_B3LYP",
            [1, 1, 3],
            Some(KRsGdfMutation::Rsh(m)),
        );
        eprintln!("CAM-B3LYP 1x1x3 {m:?}: {d:.2e}");
        assert!(
            d.abs() > 1e-4,
            "{m:?} left the supercell anchor green: {d:e}"
        );
    }
}

// PySCF 2.13 pbc.dft.KRKS pins of `pbc_krsh.rs` (AFTDF, UniformGrids 61^3,
// exxdiv ewald, STO-3G H2 a = 4 Bohr); 2x2x2 meshes omitted for the reason
// stated there (PySCF's attenuated AFTDF K did not finish in 10 minutes).
const PYSCF: [(&str, [usize; 3], f64); 6] = [
    ("HSE06", [1, 1, 1], -1.575428343697),
    ("HSE06", [1, 1, 2], -1.347311737653),
    ("HYB_GGA_XC_CAM_B3LYP", [1, 1, 1], -1.591829863303),
    ("HYB_GGA_XC_CAM_B3LYP", [1, 1, 2], -1.354469613012),
    ("HYB_GGA_XC_WB97X", [1, 1, 1], -1.605512318088),
    ("HYB_GGA_XC_WB97X", [1, 1, 2], -1.363565753524),
];
/// RS-GDF with the bundled cc-pVDZ-RI aux is not PySCF's AFT: E - pin is the
/// fit error of that aux, whose scale is the plain-Coulomb (PBE0) RS-GDF vs
/// dense gap at the same mesh (the dense energy equals the pin to 1e-9). A
/// defect moves E by >= 1.7e-2 against scales <= 3e-4.
#[test]
fn rsgdf_rsh_matches_pinned_pyscf_to_the_fit_error() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::uniform(&cell, [61; 3]).unwrap();
    for (f, n, e_ref) in PYSCF {
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

/// Forces and stress of a range-separated RS-GDF SCF: the HF entries and a KS
/// entry with another functional refuse the attenuated J/K (typed); the
/// closed-shell RKS force and stress with the functional's own range
/// separation run on them. The Python / CLI allowlist still refuses the name.
#[test]
fn rsh_rsgdf_derivatives_need_the_rks_entry_with_the_functionals_rsh() {
    use ferric_pbc::kdft::solve_krks_keep_on_grid;
    use ferric_pbc::kgrad::{kpoint_rhf_gradient, KGradConfig};
    use ferric_pbc::kstress::{kpoint_rhf_stress, KStressConfig};
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let grid = PeriodicGrid::uniform(&cell, [24; 3]).unwrap();
    let cfg = krks_cfg(&cell, "HSE06", true, None);
    let (r, kept) =
        solve_krks_keep_on_grid(&cell, &prep, Some(&aux), &mesh, &grid, &cfg).expect("hse keep");
    let g = kpoint_rhf_gradient(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &kept.hk,
        kept.grad_jk(Some(&aux), &cfg.krhf.rsgdf).unwrap(),
        &r.scf,
        ExxDiv::Ewald,
        &KGradConfig::default(),
    );
    let e = g.expect_err("force on an RSH RS-GDF SCF must be refused");
    assert!(e.to_string().contains("range-separated"), "{e}");
    let s = kpoint_rhf_stress(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &kept.hk,
        kept.stress_jk(Some(&aux), &cfg.krhf.rsgdf).unwrap(),
        &r.scf,
        ExxDiv::Ewald,
        &KStressConfig::default(),
    );
    let e = s.expect_err("stress on an RSH RS-GDF SCF must be refused");
    assert!(e.to_string().contains("range-separated"), "{e}");
    // The KS entry with a NON-range-separated name (so the functional guard
    // does not fire) must still be refused on the RSH-carrying J/K.
    let kgrad = ferric_pbc::kxcgrad::kpoint_rks_gradient(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &kept.hk,
        kept.grad_jk(Some(&aux), &cfg.krhf.rsgdf).unwrap(),
        &r.scf,
        "PBE0",
        &ferric_pbc::kxcgrad::KRksGradConfig {
            exxdiv: Some(ExxDiv::Ewald),
            ..Default::default()
        },
    );
    let e = kgrad.expect_err("KS force on an RSH RS-GDF SCF must be refused");
    assert!(e.to_string().contains("range-separated"), "{e}");
    // The KS entry WITH the range-separated name runs on the RS-GDF RSH J/K
    // (the Fock guard passes: it is the SCF's own source); the forces sum to
    // zero. Its FD validation is validation_krsh_rsgdf_force.rs.
    let kgrad = ferric_pbc::kxcgrad::kpoint_rks_gradient(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &kept.hk,
        kept.grad_jk(Some(&aux), &cfg.krhf.rsgdf).unwrap(),
        &r.scf,
        "HSE06",
        &ferric_pbc::kxcgrad::KRksGradConfig {
            exxdiv: Some(ExxDiv::Ewald),
            grid: ferric_pbc::kxcgrad::KXcGrid::Uniform([24; 3]),
            ..Default::default()
        },
    );
    let g = kgrad.expect("HSE06 KS force on its own RS-GDF RSH J/K");
    assert!(g.hf.net_force < 1e-10, "{:e}", g.hf.net_force);
    assert!(g.hf.fock_mismatch < 1e-8, "{:e}", g.hf.fock_mismatch);
    let st = ferric_pbc::kxcstress::kpoint_rks_stress(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &kept.hk,
        kept.stress_jk(Some(&aux), &cfg.krhf.rsgdf).unwrap(),
        &r.scf,
        "HSE06",
        &ferric_pbc::kxcstress::KRksStressConfig {
            exxdiv: Some(ExxDiv::Ewald),
            grid: ferric_pbc::kxcgrad::KXcGrid::Uniform([24; 3]),
            ..Default::default()
        },
    );
    assert!(st.is_ok(), "{:?}", st.err());
    // Another range separation than the build's is refused.
    let e = ferric_pbc::kxcgrad::kpoint_rks_gradient(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &kept.hk,
        kept.grad_jk(Some(&aux), &cfg.krhf.rsgdf).unwrap(),
        &r.scf,
        "HYB_GGA_XC_CAM_B3LYP",
        &ferric_pbc::kxcgrad::KRksGradConfig {
            exxdiv: Some(ExxDiv::Ewald),
            ..Default::default()
        },
    )
    .expect_err("CAM-B3LYP on HSE06 J/K must be refused")
    .to_string();
    assert!(e.contains("functional's"), "{e}");
    // The binding derivative gate refuses RSH on RS-GDF before any SCF (the
    // library has it, above; not wired to the bindings yet), and open-shell
    // RSH on either J/K.
    use ferric_pbc::kxcgrad::{check_deriv_functional, DerivSpin};
    assert!(check_deriv_functional(DerivSpin::Restricted, true, "HSE06").is_err());
    assert!(
        check_deriv_functional(DerivSpin::Unrestricted, false, "HYB_GGA_XC_CAM_B3LYP").is_err()
    );
}

/// k-point ROKS takes the attenuated RS-GDF K per spin as well
/// (mgoldey/ferric#354): in the closed-shell limit it must be the RKS energy
/// on the same fit.
#[test]
fn roks_rsgdf_closed_shell_rsh_is_krks() {
    use ferric_pbc::kroks::{solve_kroks_on_grid, KRoksConfig};
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let grid = PeriodicGrid::uniform(&cell, [24; 3]).unwrap();
    let k = krks_cfg(&cell, "HSE06", true, None);
    let mut roks = KRoksConfig::new(&cell, ExxDiv::Ewald, "HSE06");
    roks.rohf.hcore = k.krhf.hcore;
    roks.rohf.scf.energy_conv = 1e-12;
    roks.rohf.scf.grad_conv = 1e-9;
    roks.rohf.jk = KJkKind::RsGdf;
    roks.rohf.rsgdf = k.krhf.rsgdf;
    let ro = solve_kroks_on_grid(&cell, &prep, Some(&aux), &mesh, &grid, &roks)
        .unwrap()
        .scf
        .energy;
    let r = run(&cell, &bs, Some("cc-pvdz-ri"), &mesh, &grid, "HSE06", None)
        .scf
        .energy;
    eprintln!("closed-shell HSE06 RS-GDF: KROKS - KRKS {:.2e}", ro - r);
    assert!((ro - r).abs() < 1e-11, "{:.2e}", ro - r);
}

/// Cross-surface pin shared with `test_pbc_krks.py::
/// test_range_separated_hybrid_runs_on_rsgdf_energy_only` (the Python binding
/// produced the number; the library reproduces it on the same input): H2 a = 4
/// Bohr, 1x1x2 Gamma mesh, cc-pVDZ-RI aux, hcore omega 0.8, atom-centred grid
/// (40, 110), HSE06.
const HSE06_RSGDF_112_ATOM_GRID: f64 = -1.3484931727;

#[test]
fn rsgdf_hse06_matches_the_python_binding_pin() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::build(&cell, &PeriodicGridConfig::with_size(40, 110)).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let e = run(&cell, &bs, Some("cc-pvdz-ri"), &mesh, &grid, "HSE06", None)
        .scf
        .energy;
    eprintln!("HSE06 RS-GDF 1x1x2 atom grid: {e:.10} (pin {HSE06_RSGDF_112_ATOM_GRID})");
    assert!((e - HSE06_RSGDF_112_ATOM_GRID).abs() < 1e-7, "{e:.10}");
}

/// Regression: a q class whose K vectors lie inside the Gaussian cut but whose
/// kernel weights are all under the absolute lindep used to abort the build
/// ("every metric eigenvalue ... <= lindep"). H2 cubic a = 5.3 Bohr, Gamma,
/// HSE06 (w = 0.11 -> cut 1.2 Bohr^-1, smallest |G| = 1.185, weights ~1e-12).
/// The class contributes below the build precision, so the fit must run, agree
/// with the dense RSH to the plain-Coulomb fit scale, and give a sane HSE06 vs
/// PBE0 gap.
#[test]
fn hse06_rsgdf_runs_when_every_erf_weight_is_below_lindep() {
    let cell = h2_cell(5.3);
    let bs = pyscf_sto3g_h();
    let grid = PeriodicGrid::uniform(&cell, [GRID; 3]).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let aux = "cc-pvdz-ri";
    let hse_d = run(&cell, &bs, None, &mesh, &grid, "HSE06", None)
        .scf
        .energy;
    let hse_g = run(&cell, &bs, Some(aux), &mesh, &grid, "HSE06", None)
        .scf
        .energy;
    let pbe_d = run(&cell, &bs, None, &mesh, &grid, "PBE0", None).scf.energy;
    let pbe_g = run(&cell, &bs, Some(aux), &mesh, &grid, "PBE0", None)
        .scf
        .energy;
    let (scale, d) = (pbe_g - pbe_d, hse_g - hse_d);
    eprintln!(
        "a = 5.3: HSE06 gdf {hse_g:.10} dense {hse_d:.10} (dE {d:.3e}); PBE0 gdf {pbe_g:.10} dense {pbe_d:.10} (plain fit {scale:.3e})"
    );
    assert!(scale.abs() > 1e-7);
    assert!(
        (d - scale).abs() < FIT_BAR_RATIO * scale.abs(),
        "{d:e} vs {scale:e}"
    );
    // HSE06 and PBE0 differ by the short-range-screened exchange: finite, and
    // the same on both J/K up to the fit error.
    let gap = hse_g - pbe_g;
    assert!(gap.abs() > 1e-4 && gap.abs() < 0.1, "HSE06 - PBE0 = {gap}");
    assert!((gap - (hse_d - pbe_d)).abs() < 3.0 * scale.abs());
}

/// The other side of the criterion: when the kernel weights are NOT negligible
/// but the aux basis cannot represent them (one extremely tight s function,
/// |X(K)|^2 ~ 1e-12), the metric is genuinely singular and the typed error
/// stays.
#[test]
fn singular_attenuated_metric_with_real_weights_is_still_an_error() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let hk = periodic_hcore_kpts(
        &cell,
        &prep,
        &mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA_HCORE),
    )
    .unwrap();
    let site = SiteBasis::new(&[[0.3, 0.2, 0.1, 2.0e8]], 0).unwrap();
    let e = KRsGdf::build_rsh(
        &cell,
        &prep,
        &site.prep,
        &mesh,
        &hk.s,
        &gdf_cfg(ExxDiv::Ewald, None),
        rsh(0.5, HSE_LIKE),
    )
    .expect_err("a singular attenuated metric with real weights must be an error")
    .to_string();
    assert!(e.contains("not negligible"), "{e}");
}
