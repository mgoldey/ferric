//! Stage 3: k-point RS-GDF (`ferric_pbc::rsgdf::kpoint::KRsGdf`) against the
//! dense k-point AFT oracle (`KDenseAftEri`), the Gamma RS-GDF (`RsGdf`) and
//! the Python prototype (`reference/pbc/pbc_kgdf.py`, FINDINGS "Iteration 11
//! (Python, k-point RS-GDF)"; tests at the end of `test_prototype.py`).
//!
//! (a) Trivial-aux anchor (the Gamma `pbc_rsgdf.rs` anchor: one s per H,
//!     α = 0.5, 24 aux s on the 8 half-lattice classes per pair type) on H2
//!     1×1×3: the SAME 24 centres span every q-Bloch pair density, so
//!     `B Bᴴ` must reproduce the dense k-point kernels at every (k,k')
//!     (prototype 1.8e-12; asserted 1e-11) and the SCF energy (8e-13; 1e-10).
//!     Negative controls, localised to q ≠ 0 (the q = 0 blocks stay exact):
//!     `QPhaseSign` (e^{+iq·T} on the SR aux images of J3; prototype
//!     max|ΔKker| 1.8e2) and `G0AllQ` (Gamma G = 0 bookkeeping at every q;
//!     6.4e-2). The phase mutant needs N >= 3 on an axis (at N = 2 every
//!     e^{iq·T} is real). Also: time-reversal fill ≡ every q built
//!     (1×2×3, 4 of 6 classes; prototype 5.8e-15); an exactly duplicated aux
//!     is dropped and COUNTED at every q and leaves the fit exact.
//! (b) 1×1×1 ≡ Gamma `RsGdf` (fitted kernels and both-exxdiv energies,
//!     ≤ 1e-12; prototype 1.9e-13 / 1.4e-14) — an independent construction
//!     (complex eigh_herm + residue code vs the real Gamma path).
//! (c) H2 1×1×3 k-mesh RS-GDF RHF ≡ Gamma RS-GDF RHF of the explicit supercell
//!     (aux on every image) per cell, both exxdiv, ≤ 1e-12 (prototype
//!     4.7e-15: EXACT by construction — the supercell metric is block-diagonal
//!     in q with blocks J2(q), so the same lindep cut keeps 3 × 28 = 84); the
//!     `QPhaseSign` mutant is off by > 1e-2 (prototype −8.8e-2).
//! (d) Pins: H2/STO-3G a = 4 1×1×2 with CARTESIAN cc-pvdz-ri (30) and
//!     def2-universal-jkfit (40), triclinic 4H s+p 1×1×2 with both (60, 80),
//!     all from `pbc_kgdf` (cart aux, w = 1, prec 1e-13), which equals
//!     PySCF 2.13 KRHF + GDF to <= 7e-12 and + RSDF to <= 5e-11 on exactly
//!     these systems (FINDINGS Iteration 11 oracle table; H2 cc-pvdz-ri is
//!     `KGDF_REF_H2_112` digit for digit). The prototype pins are CART aux,
//!     so the Rust side builds the bundled sets with `pure = false` (same
//!     function count as the prototype asserted). Run through the
//!     `solve_krhf(jk = rsgdf)` entry point.
//! (e) The Gamma SR sums are BIT-identical to the single-bin walk of the
//!     binned k-point mode (the refactor that shares the walk must not move
//!     a bit of `RsGdf`; the existing `pbc_rsgdf.rs` tests pin the rest).
//! (f) Budget gates compose and name quantity + bytes; K-chunking is
//!     invariant.
//! (g) `KJkKind` strict parse; `solve_krhf` refuses rsgdf without aux and
//!     dense with an (unused) aux.
//!
//! Mutation plan for the main agent (production-code edits): drop the
//! `if g0_here` J2 branch → (a) q = 0 blocks, (b), (c) fail; J3 phase
//! `ph.conj()` → `*ph` (= `QPhaseSign`, in-test); remove the q = 0
//! `hermitize_pairs` → exact tests still pass (a HERMITIAN anchor), the loose
//! precision SCF of the prototype would stall (not ported: needs the diffuse
//! even-tempered aux); skip the mirror in `contract` → (a) energies and (c)
//! fail (1×1×3 has a mirrored q pair; the 1×1×2 pins are all-TRIM).

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::ewald::madelung_constant;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{
    solve_krhf, solve_krhf_injected, KJkKind, KPointInjection, KRhfConfig, KScfConfig, KScfResult,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{
    sr_sums_gamma_and_single_bin, KRsGdf, KRsGdfConfig, KRsGdfMutation,
    DEFAULT_K_FITTED_KERNELS_MAX_BYTES,
};
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig, DEFAULT_FITTED_ERI_MAX_BYTES};
use ndarray::Array2;
use num_complex::Complex64;

const HCORE_OMEGA: f64 = 0.8;
const ANCHOR_ALPHA: f64 = 0.5;
const AMPLE: usize = 1 << 30;

// ---- (d) pbc_kgdf pins (cart aux, w = 1, prec 1e-13; scratch run of
// build_kgdf + krhf(conv 1e-12) 2026-09-24): (none, ewald) -----------------
// Spherical (pure) aux, as ferric bundles it: reference/pbc/run_kgdf_sph_pins.py.
const H2_112_CCPVDZ_RI: (f64, f64) = (-0.9027998778875217, -1.354260330500599);
const H2_112_JKFIT: (f64, f64) = (-0.9026909165418127, -1.3541513691548905);
const TRI_112_CCPVDZ_RI: (f64, f64) = (-1.5875876466563348, -2.327058254972801);
const TRI_112_JKFIT: (f64, f64) = (-1.5876606977729906, -2.3271313060894583);
/// Same tolerance as the dense-AFT PySCF pins of `pbc_krhf.rs`.
const PIN_TOL: f64 = 1e-9;

fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig::with_omega(HCORE_OMEGA)
}

fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        ..Default::default()
    }
}

fn gdf_cfg(omega: f64) -> RsGdfConfig {
    RsGdfConfig {
        omega,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

fn kcfg(omega: f64, mutation: Option<KRsGdfMutation>) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: gdf_cfg(omega),
        mutation,
    }
}

/// The bundled aux set with every shell Cartesian. libint2's 2/3-centre
/// ERIs require pure shells above l = 1 and ABORT on Cartesian ones, so the
/// builders must refuse such an aux basis with a typed error.
fn cart_aux(name: &str) -> BasisSet {
    let mut b = basis::bundled(name).unwrap();
    for shells in b.shells.values_mut() {
        for s in shells.iter_mut() {
            s.pure = false;
        }
    }
    b
}

#[test]
fn cartesian_aux_above_p_is_refused_not_aborted() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let auxp = PreparedBasis::new(cell.mol(), &cart_aux("cc-pvdz-ri")).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let mut cfg = KRhfConfig::for_cell(&cell, ExxDiv::None);
    cfg.scf = kscf_cfg();
    cfg.hcore = hcore_cfg();
    cfg.jk = KJkKind::parse_config_str("rsgdf").unwrap();
    cfg.rsgdf = kcfg(1.0, None);
    let err = solve_krhf(&cell, &prep, Some(&auxp), &mesh, &cfg)
        .expect_err("a Cartesian d aux shell must be refused");
    let msg = err.to_string();
    assert!(msg.contains("Cartesian") && msg.contains("l = 2"), "{msg}");
}

fn cmax(a: &Array2<Complex64>, b: &Array2<Complex64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter().zip(b.iter()).fold(0.0_f64, |m, (x, y)| {
        let d = (x - y).norm();
        assert!(d.is_finite());
        m.max(d)
    })
}

/// The 24 anchor aux centres (copy of `pbc_rsgdf.rs::anchor_sites`).
fn anchor_sites(cell: &Cell, classes: &[usize]) -> Vec<[f64; 4]> {
    let r = cell.positions();
    let a = cell.lattice();
    let mut out = Vec::new();
    for (i, j) in [(0usize, 0usize), (1, 1), (0, 1)] {
        for k in 0..8usize {
            if !classes.contains(&k) {
                continue;
            }
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

fn all_classes() -> Vec<usize> {
    (0..8).collect()
}

fn krhf_with(
    cell: &Cell,
    mesh: &KPointMesh,
    hk: &PeriodicHcoreK,
    jk: Box<dyn ferric_pbc::kscf::KPointJk + '_>,
) -> KScfResult {
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk,
    };
    let r = solve_krhf_injected(cell, mesh, &kscf_cfg(), inj).expect("k-point RHF");
    assert!(
        r.converged,
        "k-point RHF did not converge ({} it)",
        r.iterations
    );
    r
}

struct KAnchor {
    cell: Cell,
    prep: PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
}

fn k_anchor(n: [usize; 3]) -> KAnchor {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).unwrap();
    KAnchor {
        cell,
        prep,
        mesh,
        hk,
    }
}

/// (max |ΔJker|, per-(k,k') max |ΔKker|) of the fit vs the dense oracle.
fn kernel_errors(gdf: &KRsGdf, eri: &KDenseAftEri) -> (f64, Vec<Vec<f64>>) {
    let nk = gdf.nk();
    let (jf, kf) = gdf
        .fitted_kernels(DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
        .unwrap();
    let mut dj = 0.0_f64;
    let mut dk = vec![vec![0.0; nk]; nk];
    for k in 0..nk {
        for kp in 0..nk {
            dj = dj.max(cmax(&jf[k * nk + kp], eri.jker(k, kp)));
            dk[k][kp] = cmax(&kf[k * nk + kp], eri.kker(k, kp));
        }
    }
    (dj, dk)
}

fn anchor_dense(an: &KAnchor) -> KDenseAftEri {
    KDenseAftEri::build(
        &an.cell,
        &an.prep,
        &an.mesh,
        &an.hk.s,
        ExxDiv::None,
        &KDenseAftConfig::default(),
    )
    .expect("dense k kernels")
}

// ---------------------------------------------------------------- (a)

#[test]
fn krsgdf_matches_dense_kpoint_aft_in_the_trivial_aux_limit() {
    let an = k_anchor([1, 1, 3]);
    let eri = anchor_dense(&an);
    let site = SiteBasis::new(&anchor_sites(&an.cell, &all_classes()), 0).unwrap();
    let gdf = KRsGdf::build(
        &an.cell,
        &an.prep,
        &site.prep,
        &an.mesh,
        &an.hk.s,
        &kcfg(0.8, None),
    )
    .unwrap();
    let st = gdf.stats();
    for q in &st.per_q {
        eprintln!(
            "q class {} (trim {}, mirrored {:?}): kept {}/{} eig {:.2e}..{:.2e}, {} K, asym J2 {:.1e} J3 {:.1e}",
            q.iq,
            q.trim,
            q.mirrored_from,
            q.naux_kept,
            q.naux,
            q.metric_eig_min,
            q.metric_eig_max,
            q.n_k_vectors,
            q.asym_j2,
            q.asym_j3
        );
        assert_eq!(q.naux_kept, 24, "prototype kept 24/24 at every q");
    }
    assert_eq!(st.n_q_built, 2);
    let (dj, dk) = kernel_errors(&gdf, &eri);
    let dkmax = dk.iter().flatten().fold(0.0_f64, |a, &b| a.max(b));
    eprintln!("anchor 1x1x3: max|ΔJker| {dj:.2e}, max|ΔKker| {dkmax:.2e}");
    assert!(dj < 1e-11 && dkmax < 1e-11, "dJ {dj:.3e} dK {dkmax:.3e}");
    let vm = eri.madelung_ewald();
    assert!((gdf.madelung_ewald() - vm).abs() < 1e-15);
    for v in [0.0, vm] {
        let e0 = krhf_with(
            &an.cell,
            &an.mesh,
            &an.hk,
            Box::new(eri.jk_builder_with_madelung(v)),
        );
        let e1 = krhf_with(
            &an.cell,
            &an.mesh,
            &an.hk,
            Box::new(gdf.jk_builder_with_madelung(v)),
        );
        eprintln!(
            "  v_M {v:.6}: E_dense {:.12} E_gdf {:.12}",
            e0.energy, e1.energy
        );
        assert!((e1.energy - e0.energy).abs() < 1e-10);
    }
}

#[test]
fn krsgdf_anchor_catches_q_phase_and_g0_mutants_at_q_nonzero_only() {
    let an = k_anchor([1, 1, 3]);
    let eri = anchor_dense(&an);
    let site = SiteBasis::new(&anchor_sites(&an.cell, &all_classes()), 0).unwrap();
    for (mutation, floor) in [
        (KRsGdfMutation::QPhaseSign, 1.0), // prototype 1.8e2
        (KRsGdfMutation::G0AllQ, 1e-2),    // prototype 6.4e-2
    ] {
        let gdf = KRsGdf::build(
            &an.cell,
            &an.prep,
            &site.prep,
            &an.mesh,
            &an.hk.s,
            &kcfg(0.8, Some(mutation)),
        )
        .unwrap();
        let (_, dk) = kernel_errors(&gdf, &eri);
        let nk = gdf.nk();
        let q0 = (0..nk).map(|k| dk[k][k]).fold(0.0_f64, f64::max);
        let qn = (0..nk)
            .flat_map(|k| (0..nk).filter(move |&j| j != k).map(move |j| (k, j)))
            .map(|(k, j)| dk[k][j])
            .fold(0.0_f64, f64::max);
        eprintln!("{mutation:?}: q=0 blocks {q0:.2e}, q!=0 blocks {qn:.2e}");
        assert!(q0 < 1e-10, "{mutation:?} leaked into q = 0: {q0:.3e}");
        assert!(qn > floor, "{mutation:?} not detected: {qn:.3e}");
    }
}

#[test]
fn krsgdf_time_reversal_fill_equals_every_q_built() {
    let an = k_anchor([1, 2, 3]);
    let site = SiteBasis::new(&anchor_sites(&an.cell, &all_classes()), 0).unwrap();
    let build = |m| {
        KRsGdf::build(
            &an.cell,
            &an.prep,
            &site.prep,
            &an.mesh,
            &an.hk.s,
            &kcfg(0.8, m),
        )
        .unwrap()
    };
    let tr = build(None);
    let all = build(Some(KRsGdfMutation::NoTimeReversal));
    assert_eq!(tr.stats().n_q_built, 4);
    assert_eq!(all.stats().n_q_built, 6);
    let (j1, k1) = tr
        .fitted_kernels(DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
        .unwrap();
    let (j2, k2) = all
        .fitted_kernels(DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
        .unwrap();
    let d = j1
        .iter()
        .zip(&j2)
        .chain(k1.iter().zip(&k2))
        .map(|(a, b)| cmax(a, b))
        .fold(0.0_f64, f64::max);
    eprintln!("time reversal 1x2x3: max|Δkernel| {d:.2e} (prototype 5.8e-15)");
    assert!(d < 1e-12, "{d:.3e}");
}

#[test]
fn krsgdf_duplicate_aux_is_dropped_and_counted_at_every_q() {
    let an = k_anchor([1, 1, 3]);
    let eri = anchor_dense(&an);
    let base_sites = anchor_sites(&an.cell, &all_classes());
    let mut dup_sites = base_sites.clone();
    dup_sites.push(base_sites[5]);
    let dup = SiteBasis::new(&dup_sites, 0).unwrap();
    let gdf = KRsGdf::build(
        &an.cell,
        &an.prep,
        &dup.prep,
        &an.mesh,
        &an.hk.s,
        &kcfg(0.8, None),
    )
    .unwrap();
    for q in &gdf.stats().per_q {
        eprintln!(
            "dup aux q class {}: dropped {}/{} (min eig {:.2e})",
            q.iq, q.n_dropped, q.naux, q.metric_eig_min
        );
        assert_eq!(q.naux, 25);
        assert_eq!(
            q.n_dropped, 1,
            "q class {}: the duplicate must be dropped and counted",
            q.iq
        );
        assert_eq!(q.naux_kept + q.n_dropped, q.naux);
    }
    let (dj, dk) = kernel_errors(&gdf, &eri);
    let dkmax = dk.iter().flatten().fold(0.0_f64, |a, &b| a.max(b));
    assert!(dj < 1e-10 && dkmax < 1e-10, "dJ {dj:.3e} dK {dkmax:.3e}");
}

// ---------------------------------------------------------------- (b)

#[test]
fn one_point_mesh_is_the_gamma_rsgdf() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).unwrap();
    let gamma = RsGdf::build(&cell, &prep, &aux, &hc.s, &gdf_cfg(1.0)).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).unwrap();
    let kg = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kcfg(1.0, None)).unwrap();
    let q0 = &kg.stats().per_q[0];
    assert_eq!(q0.naux_kept, gamma.stats().naux_kept);
    let eri = gamma.fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES).unwrap();
    let (jf, kf) = kg
        .fitted_kernels(DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
        .unwrap();
    let mut d = 0.0_f64;
    for (x, y) in jf[0]
        .iter()
        .zip(eri.iter())
        .chain(kf[0].iter().zip(eri.iter()))
    {
        d = d.max((x - Complex64::new(*y, 0.0)).norm());
    }
    eprintln!("1x1x1 vs Gamma RsGdf: max|Δkernel| {d:.2e} (prototype 1.9e-13)");
    assert!(d < 1e-12, "{d:.3e}");
    assert!((kg.madelung_ewald() - madelung_constant(&cell).unwrap()).abs() < 1e-15);
    for exx in [ExxDiv::None, ExxDiv::Ewald] {
        let g = gamma.clone().with_exxdiv(&cell, exx).unwrap();
        let eg = gamma_rhf_jk(
            &cell,
            &prep,
            &hc,
            Box::new(g.j_builder()),
            Box::new(g.k_builder()),
        )
        .energy;
        let k1 = kg.clone().with_exxdiv(exx);
        let ek = krhf_with(&cell, &mesh, &hk, Box::new(k1.jk_builder())).energy;
        eprintln!(
            "  {exx:?}: E_gamma {eg:.12} E_k {ek:.12} dE {:.1e}",
            ek - eg
        );
        assert!((ek - eg).abs() < 1e-12, "{exx:?}: {:.3e}", ek - eg);
    }
}

// ---------------------------------------------------------------- (c)

/// Explicit diag(n) supercell (copy of `pbc_krhf.rs::supercell`).
fn supercell(cell: &Cell, n: [usize; 3]) -> Cell {
    let a = *cell.lattice();
    let mut atoms = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                let t: Vec<f64> = (0..3)
                    .map(|d| m0 as f64 * a[0][d] + m1 as f64 * a[1][d] + m2 as f64 * a[2][d])
                    .collect();
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
    Cell::new(hydrogens(&atoms), lat).expect("supercell")
}

#[test]
fn kmesh_rsgdf_is_the_gamma_rsgdf_of_the_supercell() {
    let n = [1, 1, 3];
    let cell = h2_cell(4.0);
    let obs = pyscf_sto3g_h();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    // Supercell Gamma side (aux on every image).
    let sc = supercell(&cell, n);
    let prep_sc = prep_for(&sc, &obs);
    let aux_sc = PreparedBasis::new(sc.mol(), &aux_bs).unwrap();
    let hc_sc = periodic_hcore(&sc, &prep_sc, &hcore_cfg()).unwrap();
    let gs = RsGdf::build(&sc, &prep_sc, &aux_sc, &hc_sc.s, &gdf_cfg(1.0)).unwrap();
    // k-mesh side.
    let prep = prep_for(&cell, &obs);
    let aux = PreparedBasis::new(cell.mol(), &aux_bs).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).unwrap();
    let kg = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kcfg(1.0, None)).unwrap();
    let kept: usize = kg.stats().per_q.iter().map(|q| q.naux_kept).sum();
    eprintln!(
        "kept per q {:?} (sum {kept}) vs supercell {}",
        kg.stats()
            .per_q
            .iter()
            .map(|q| q.naux_kept)
            .collect::<Vec<_>>(),
        gs.stats().naux_kept
    );
    assert_eq!(kept, gs.stats().naux_kept);
    let vm_sc = madelung_constant(&sc).unwrap();
    assert!((kg.madelung_ewald() - vm_sc).abs() < 1e-14);
    let nk = 3.0;
    let mut e_sc_ewald = 0.0;
    for exx in [ExxDiv::None, ExxDiv::Ewald] {
        let g = gs.clone().with_exxdiv(&sc, exx).unwrap();
        let es = gamma_rhf_jk(
            &sc,
            &prep_sc,
            &hc_sc,
            Box::new(g.j_builder()),
            Box::new(g.k_builder()),
        )
        .energy
            / nk;
        let ek = krhf_with(
            &cell,
            &mesh,
            &hk,
            Box::new(kg.clone().with_exxdiv(exx).jk_builder()),
        )
        .energy;
        eprintln!(
            "  {exx:?}: E_k {ek:.12} E_sc/N {es:.12} dE {:.1e} (prototype 4.7e-15)",
            ek - es
        );
        assert!((ek - es).abs() < 1e-12, "{exx:?}: {:.3e}", ek - es);
        if exx == ExxDiv::Ewald {
            e_sc_ewald = es;
        }
    }
    let km = KRsGdf::build(
        &cell,
        &prep,
        &aux,
        &mesh,
        &hk.s,
        &kcfg(1.0, Some(KRsGdfMutation::QPhaseSign)),
    )
    .unwrap()
    .with_exxdiv(ExxDiv::Ewald);
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: Box::new(km.jk_builder()),
    };
    // The mutant may not even converge; any energy it reaches must be off.
    let e = solve_krhf_injected(&cell, &mesh, &kscf_cfg(), inj)
        .map(|r| r.energy)
        .unwrap_or(f64::NAN);
    eprintln!("QPhaseSign mutant: E_k {e:.12} vs E_sc/N {e_sc_ewald:.12} (prototype −8.8e-2)");
    assert!(
        !((e - e_sc_ewald).abs() <= 1e-2),
        "phase mutant not caught: {e}"
    );
}

// ---------------------------------------------------------------- (d)

fn pinned(name: &str, cell: &Cell, obs: &BasisSet, aux: &str, naux: usize, pins: (f64, f64)) {
    let prep = prep_for(cell, obs);
    let auxp = PreparedBasis::new(cell.mol(), &basis::bundled(aux).unwrap()).unwrap();
    assert_eq!(
        auxp.nbasis(),
        naux,
        "{aux}: pure aux count differs from the prototype's"
    );
    let mesh = KPointMesh::gamma_centred(cell, [1, 1, 2]).unwrap();
    for (exx, e_ref) in [(ExxDiv::None, pins.0), (ExxDiv::Ewald, pins.1)] {
        let mut cfg = KRhfConfig::for_cell(cell, exx);
        cfg.scf = kscf_cfg();
        cfg.hcore = hcore_cfg();
        cfg.jk = KJkKind::parse_config_str("rsgdf").unwrap();
        cfg.rsgdf = kcfg(1.0, None);
        let r = solve_krhf(cell, &prep, Some(&auxp), &mesh, &cfg).expect("krhf rsgdf");
        assert!(r.converged);
        eprintln!(
            "{name} {aux} {exx:?}: E {:.12} pin {e_ref:.12} dE {:.1e} ({} it)",
            r.energy,
            r.energy - e_ref,
            r.iterations
        );
        assert!(
            (r.energy - e_ref).abs() < PIN_TOL,
            "{name} {aux} {exx:?}: {}",
            r.energy
        );
    }
}

#[test]
fn h2_1x1x2_matches_pinned_kgdf_cc_pvdz_ri_and_jkfit() {
    let cell = h2_cell(4.0);
    pinned(
        "H2 1x1x2",
        &cell,
        &pyscf_sto3g_h(),
        "cc-pvdz-ri",
        28,
        H2_112_CCPVDZ_RI,
    );
    pinned(
        "H2 1x1x2",
        &cell,
        &pyscf_sto3g_h(),
        "def2-universal-jkfit",
        36,
        H2_112_JKFIT,
    );
}

#[test]
#[ignore = "slow: triclinic 4H s+p 1x1x2 RS-GDF with two pure aux sets (prototype 150 s / 80 s per set; \
            unmeasured in Rust); run with --release -- --ignored, serially, on a quiet box"]
fn triclinic_sp_1x1x2_matches_pinned_kgdf() {
    let cell = triclinic_cell();
    pinned(
        "tri 1x1x2",
        &cell,
        &sp_basis_h(),
        "cc-pvdz-ri",
        56,
        TRI_112_CCPVDZ_RI,
    );
    pinned(
        "tri 1x1x2",
        &cell,
        &sp_basis_h(),
        "def2-universal-jkfit",
        72,
        TRI_112_JKFIT,
    );
}

// ---------------------------------------------------------------- (e)

#[test]
fn gamma_sr_sums_are_bitwise_the_single_bin_walk() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    // Production screen only: the unscreened global radius over diffuse
    // cc-pvdz-ri shells is a much larger walk and adds nothing here.
    let [(j2g, j3g), (j2b, j3b)] =
        sr_sums_gamma_and_single_bin(&cell, &prep, &aux, &gdf_cfg(1.0)).unwrap();
    for (a, b, what) in [(&j2g, &j2b, "J2"), (&j3g, &j3b, "J3")] {
        assert_eq!(a.dim(), b.dim());
        assert!(a.iter().any(|x| *x != 0.0), "{what}: vacuous");
        let differ = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(differ, 0, "{what}: {differ} elements differ in bits");
    }
}

// ---------------------------------------------------------------- (f)

#[test]
fn krsgdf_memory_gates_name_the_quantity_and_bytes() {
    let an = k_anchor([1, 1, 3]);
    let site = SiteBasis::new(&anchor_sites(&an.cell, &all_classes()), 0).unwrap();
    let (n2, naux, nk, rl, rt, n_built) = (4usize, 24usize, 3usize, 3usize, 3usize, 2usize);
    let build = |b: usize| {
        KRsGdf::build(
            &an.cell,
            &an.prep,
            &site.prep,
            &an.mesh,
            &an.hk.s,
            &KRsGdfConfig {
                gdf: RsGdfConfig {
                    budget_bytes: Some(b),
                    ..gdf_cfg(0.8)
                },
                mutation: None,
            },
        )
    };
    let s_bytes = (nk + 2) * n2 * 16;
    let bins = (rt * naux * naux + rl * rt * n2 * naux) * 8;
    let bres = n_built * nk * naux * n2 * 16;
    let msg = build(1).unwrap_err().to_string();
    assert!(
        msg.contains("KRsGdf S(k) copies") && msg.contains(&format!("({s_bytes} bytes")),
        "{msg}"
    );
    let msg = build(s_bytes + 100).unwrap_err().to_string();
    assert!(
        msg.contains("KRsGdf SR residue bins") && msg.contains(&format!("({bins} bytes")),
        "{msg}"
    );
    let msg = build(s_bytes + bins + 100).unwrap_err().to_string();
    assert!(
        msg.contains("KRsGdf resident B") && msg.contains(&format!("({bres} bytes")),
        "{msg}"
    );
    // K-chunk invariance: a budget just above the resident set forces many
    // LR chunks; only the GEMM summation order changes.
    let ample = build(AMPLE).unwrap();
    let tight = build(ample.stats().resident_bytes + (64 << 10)).unwrap();
    let (c1, c2) = (ample.stats().n_lr_chunks, tight.stats().n_lr_chunks);
    let (ja, ka) = ample
        .fitted_kernels(DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
        .unwrap();
    let (jt, kt) = tight
        .fitted_kernels(DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
        .unwrap();
    let d = ja
        .iter()
        .zip(&jt)
        .chain(ka.iter().zip(&kt))
        .map(|(a, b)| cmax(a, b))
        .fold(0.0_f64, f64::max);
    eprintln!("LR chunks {c1} (ample) vs {c2} (tight); max|Δkernel| {d:.2e}");
    assert_eq!(c1, ample.stats().n_q_built, "one chunk per built q class");
    assert!(c2 > c1, "tight budget did not force chunking");
    assert!(d < 1e-11, "{d:.3e}");
}

// ---------------------------------------------------------------- (g)

#[test]
fn jk_kind_parse_is_strict_and_rsgdf_needs_an_aux_basis() {
    assert_eq!(KJkKind::parse_config_str("dense").unwrap(), KJkKind::Dense);
    assert_eq!(KJkKind::parse_config_str("RsGdf").unwrap(), KJkKind::RsGdf);
    for bad in ["gdf", "rs-gdf", "", "aft"] {
        assert!(KJkKind::parse_config_str(bad).is_err(), "{bad:?} accepted");
    }
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let mut cfg = KRhfConfig::for_cell(&cell, ExxDiv::None);
    cfg.jk = KJkKind::RsGdf;
    let msg = solve_krhf(&cell, &prep, None, &mesh, &cfg)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("requires an auxbasis"), "{msg}");
    cfg.jk = KJkKind::Dense;
    let msg = solve_krhf(&cell, &prep, Some(&aux), &mesh, &cfg)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("jk = dense does not use it"), "{msg}");
}
