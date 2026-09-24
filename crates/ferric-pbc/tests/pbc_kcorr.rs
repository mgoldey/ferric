//! Stage 9: k-point MP2 / dRPA (`ferric_pbc::kcorr`). Port of the
//! prototype's Iteration-12 tests (`reference/pbc/pbc_kcorr.py`,
//! `run_kcorr_anchor.py`, `run_kcorr_oracle.py`, the end of
//! `test_prototype.py`; FINDINGS "Iteration 12").
//!
//! (a) 1×1×1 ≡ `gamma_mp2` / `gamma_drpa` (dense AFT, plasmon) on the SAME
//!     Gamma orbitals, both reference exxdiv × both conventions, ≤ 1e-12
//!     (prototype ≤ 6e-16): the complex residue pair code + the k loop vs the
//!     real Gamma `Wᵀ I W` + ferric-mp2's `spin_components_from_g`. The
//!     `NkNormalization` mutant must be INVISIBLE here (N_k = 1 cannot see a
//!     power of N_k — the stated artifact hypothesis; the supercell anchor
//!     and the PySCF pins can).
//! (b) Gamma-centred mesh per cell ≡ Gamma MP2/dRPA of the explicit diag(N)
//!     supercell / N (H2 1×1×3 — complex q — and 2×2×2), both reference
//!     exxdiv × both conventions, ≤ 1e-11 (prototype: exact, 2e-15; the two
//!     sides are separate SCFs here). Loose AFT precision: both sides cut the
//!     same K sphere. Also: `KDenseAftPairs` blocks reproduce
//!     `KDenseAftEri::kker` (an independent build of the same kernels),
//!     realified GL-40 ≡ plasmon (≤ 1e-11; prototype 4e-17) and
//! (d) the O(Π²) term of k-dRPA ≡ direct KMP2 (≤ 1e-12; prototype 3e-17).
//! (c) Trivial-aux k-point RS-GDF (the `pbc_krsgdf.rs` anchor aux) ≡ the dense
//!     pair oracle on the same orbitals, ≤ 1e-11 (prototype 8e-14), with
//!     DIFFERENT energy constructions on the two sides for dRPA (realified
//!     quadrature vs plasmon).
//! (e) PySCF 2.13 `pbc.mp.KMP2` on KRHF + AFTDF (mesh 61³) pins from
//!     `run_kcorr_oracle.py aft`: H2/STO-3G a = 4, 1×1×2 and 1×1×3 (complex
//!     q). exxdiv=None ↔ ours unshifted, exxdiv='ewald' ↔ ours shifted
//!     (prototype d ≤ 2e-14). Asserted at 1e-9 (the Rust KRHF pin tolerance;
//!     the Rust-vs-PySCF MP2 residual is UNMEASURED until this runs).
//! (f) Mutants (`KCorrMutation`, H2 1×1×3 vs the supercell, none reference,
//!     shifted; prototype magnitudes): `KbIndex` dMP2 −1.1e-3 (dRPA exact),
//!     `NoConj` +1.1e-2 (dRPA exact), `MissingMadelungShift` dMP2 −5.6e-3 /
//!     dRPA −4.4e-3 with the unshifted row exact, `RpaOccupiedKPlusQ` dRPA
//!     +1.4e-3 (MP2 exact), `NkNormalization` (both off; code mutation in
//!     the prototype, 6/8 tests failed).
//!
//! Cost: everything but the triclinic s+p anchor is H2/minimal-basis
//! scale. The triclinic case (12-atom supercell, nao 48) is marked slow by
//! analogy with `pbc_krhf.rs` (~70 s release there); UNMEASURED in Rust.

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::drpa::{gamma_drpa, GammaDrpaConfig, GammaDrpaIntegrals};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::kcorr::{
    kpoint_drpa, kpoint_mp2, KCorrIntegrals, KCorrMutation, KDenseAftPairs, KDrpaConfig,
    KDrpaEnergy, KMp2Config, KMp2Result,
};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KScfConfig, KScfResult};
use ferric_pbc::lattice::Cell;
use ferric_pbc::mp2::{gamma_mp2, GammaMp2Config, GammaMp2Integrals, Mp2Denominators};
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf_injected, PeriodicInjection, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use num_complex::Complex64;

const OMEGA: f64 = 0.8;
/// Loose AFT precision for the supercell anchors (exact at any precision).
const ANCHOR_PRECISION: f64 = 1e-6;
const AMPLE: usize = 1 << 30;
const ANCHOR_ALPHA: f64 = 0.5;

const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];
const DEN: [Mp2Denominators; 2] = [Mp2Denominators::MadelungShifted, Mp2Denominators::Unshifted];

// ---- (e) PySCF 2.13 KMP2 (AFTDF 61³) on KRHF, FINDINGS Iteration 12 ------
/// (unshifted = exxdiv None, shifted = exxdiv ewald).
const KMP2_H2_112: (f64, f64) = (-2.8883196728367e-02, -1.8228154905146e-02);
const KMP2_H2_113: (f64, f64) = (-3.2507280126487e-02, -2.6906779192508e-02);
const PIN_TOL: f64 = 1e-9;

fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig::with_omega(OMEGA)
}

fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-11,
        ..Default::default()
    }
}

fn kdense_cfg(precision: f64) -> KDenseAftConfig {
    KDenseAftConfig {
        precision,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

fn cmax(a: &Array2<Complex64>, b: &Array2<Complex64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).norm()))
}

fn krhf(
    cell: &Cell,
    mesh: &KPointMesh,
    hk: &PeriodicHcoreK,
    eri: &KDenseAftEri,
    madelung: f64,
) -> KScfResult {
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: Box::new(eri.jk_builder_with_madelung(madelung)),
    };
    let r = solve_krhf_injected(cell, mesh, &kscf_cfg(), inj).expect("k-point RHF");
    assert!(
        r.converged,
        "k-point RHF did not converge ({} it)",
        r.iterations
    );
    r
}

/// Gamma RHF on the dense AFT with a tighter density threshold than
/// `common::gamma_rhf` (correlation energies are first order in the orbital
/// error, HF is second order).
fn gamma_rhf_tight(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
) -> ScfResult {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, prep).expect("schwarz");
    let inj = PeriodicInjection {
        s: hc.s.clone(),
        h: hc.h.clone(),
        vnn: hc.enn,
        j: Box::new(eri.j_builder()),
        k: Box::new(eri.k_builder()),
        xc: None,
    };
    let cfg = RhfConfig {
        density_conv: 1e-11,
        ..gamma_config()
    };
    let r = solve_rhf_injected(&ctx, cell.mol(), prep, op, &bounds, &cfg, inj)
        .expect("gamma-point RHF");
    assert!(r.converged, "gamma-point RHF did not converge");
    r
}

fn mp2_cfg(exx: ExxDiv, den: Mp2Denominators, mutation: Option<KCorrMutation>) -> KMp2Config {
    KMp2Config {
        frozen_core: 0,
        reference_exxdiv: exx,
        denominators: den,
        budget_bytes: Some(AMPLE),
        mutation,
    }
}

fn drpa_cfg(
    exx: ExxDiv,
    den: Mp2Denominators,
    energy: KDrpaEnergy,
    quad_points: usize,
    mutation: Option<KCorrMutation>,
) -> KDrpaConfig {
    KDrpaConfig {
        frozen_core: 0,
        reference_exxdiv: exx,
        denominators: den,
        quad_points,
        energy,
        budget_bytes: Some(AMPLE),
        mutation,
    }
}

fn kmp2(
    cell: &Cell,
    mesh: &KPointMesh,
    k: &KScfResult,
    ints: KCorrIntegrals<'_>,
    exx: ExxDiv,
    den: Mp2Denominators,
    mutation: Option<KCorrMutation>,
) -> KMp2Result {
    kpoint_mp2(cell, mesh, k, ints, &mp2_cfg(exx, den, mutation)).expect("k-point MP2")
}

#[allow(clippy::too_many_arguments)]
fn kdrpa(
    cell: &Cell,
    mesh: &KPointMesh,
    k: &KScfResult,
    ints: KCorrIntegrals<'_>,
    exx: ExxDiv,
    den: Mp2Denominators,
    energy: KDrpaEnergy,
    quad_points: usize,
    mutation: Option<KCorrMutation>,
) -> f64 {
    kpoint_drpa(
        cell,
        mesh,
        k,
        ints,
        &drpa_cfg(exx, den, energy, quad_points, mutation),
    )
    .expect("k-point dRPA")
    .drpa_corr
}

fn gamma_corr(
    cell: &Cell,
    rhf: &ScfResult,
    eri: &DenseAftEri,
    exx: ExxDiv,
    den: Mp2Denominators,
) -> (f64, f64) {
    let m = gamma_mp2(
        cell,
        rhf,
        GammaMp2Integrals::DenseAft(eri),
        &GammaMp2Config {
            frozen_core: 0,
            reference_exxdiv: exx,
            denominators: den,
            budget_bytes: Some(AMPLE),
        },
    )
    .expect("gamma MP2")
    .mp2_corr;
    let r = gamma_drpa(
        cell,
        rhf,
        GammaDrpaIntegrals::DenseAft(eri),
        &GammaDrpaConfig {
            frozen_core: 0,
            reference_exxdiv: exx,
            denominators: den,
            quad_points: 40,
            budget_bytes: Some(AMPLE),
        },
    )
    .expect("gamma dRPA")
    .drpa_corr;
    (m, r)
}

/// A one-point `KScfResult` carrying the Gamma RHF's own (real) orbitals.
fn kscf_from_gamma(g: &ScfResult, mesh: &KPointMesh, nocc: usize) -> KScfResult {
    let eps: Vec<f64> = g.eps_r().to_vec();
    let n = eps.len();
    KScfResult {
        energy: g.energy,
        e_nuc: 0.0,
        eps: vec![eps.clone()],
        mos: vec![g.mos_r().mapv(|x| Complex64::new(x, 0.0))],
        densities: Vec::new(),
        fock: Vec::new(),
        occupations: vec![(0..n).map(|i| if i < nocc { 2.0 } else { 0.0 }).collect()],
        nocc_per_k: vec![nocc],
        homo: eps[nocc - 1],
        lumo: eps[nocc],
        converged: true,
        iterations: 0,
        max_error: 0.0,
        kpts: mesh.kpts().to_vec(),
    }
}

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

// ============================================================== (a)

#[test]
fn one_point_mesh_is_the_gamma_mp2_and_drpa() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).unwrap();
    let base = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let pairs = KDenseAftPairs::build(
        &cell,
        &prep,
        &mesh,
        &kdense_cfg(DEFAULT_DENSE_AFT_PRECISION),
    )
    .unwrap();
    let ints = KCorrIntegrals::DenseAft(&pairs);
    for exx in EXX {
        let eri = base.clone().with_exxdiv(&cell, exx).unwrap();
        let g = gamma_rhf_tight(&cell, &prep, &hc, &eri);
        let k = kscf_from_gamma(&g, &mesh, 1);
        for den in DEN {
            let (gm, gr) = gamma_corr(&cell, &g, &eri, exx, den);
            let km = kmp2(&cell, &mesh, &k, ints, exx, den, None).mp2_corr;
            let kpl = kdrpa(
                &cell,
                &mesh,
                &k,
                ints,
                exx,
                den,
                KDrpaEnergy::Plasmon,
                40,
                None,
            );
            let kqd = kdrpa(
                &cell,
                &mesh,
                &k,
                ints,
                exx,
                den,
                KDrpaEnergy::Quadrature,
                40,
                None,
            );
            eprintln!(
                "1x1x1 {exx:?} {den:?}: MP2 k {km:.15} gamma {gm:.15} d {:.1e}; dRPA plasmon d {:.1e}, \
                 quad-40 d {:.1e}",
                km - gm,
                kpl - gr,
                kqd - gr
            );
            assert!((km - gm).abs() <= 1e-12, "MP2 {exx:?} {den:?}");
            assert!((kpl - gr).abs() <= 1e-12, "dRPA plasmon {exx:?} {den:?}");
            assert!((kqd - gr).abs() <= 1e-11, "dRPA quad {exx:?} {den:?}");
            // Artifact hypothesis: N_k = 1 cannot see a power of N_k.
            let mut_m = kmp2(
                &cell,
                &mesh,
                &k,
                ints,
                exx,
                den,
                Some(KCorrMutation::NkNormalization),
            )
            .mp2_corr;
            let mut_r = kdrpa(
                &cell,
                &mesh,
                &k,
                ints,
                exx,
                den,
                KDrpaEnergy::Plasmon,
                40,
                Some(KCorrMutation::NkNormalization),
            );
            assert!((mut_m - km).abs() < 1e-15 && (mut_r - kpl).abs() < 1e-15);
        }
    }
}

// ============================================================== (b), (d)

struct Anchor {
    cell: Cell,
    mesh: KPointMesh,
    pairs: KDenseAftPairs,
    /// k-point RHF per reference exxdiv (`EXX` order).
    k: Vec<KScfResult>,
    /// Supercell Gamma per cell, `[exx][den] = (MP2, dRPA plasmon)`.
    sc: [[(f64, f64); 2]; 2],
}

fn build_anchor(cell: &Cell, bs: &BasisSet, n: [usize; 3], precision: f64) -> Anchor {
    let nk = (n[0] * n[1] * n[2]) as f64;
    let scell = supercell(cell, n);
    let sprep = prep_for(&scell, bs);
    let shc = periodic_hcore(&scell, &sprep, &hcore_cfg()).unwrap();
    let base = DenseAftEri::build(
        &scell,
        &sprep,
        &shc.s,
        ExxDiv::None,
        precision,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let nocc_sc = (scell.mol().nelec() / 2) as usize;
    let mut sc = [[(0.0, 0.0); 2]; 2];
    for (x, &exx) in EXX.iter().enumerate() {
        let eri = base.clone().with_exxdiv(&scell, exx).unwrap();
        let rhf = gamma_rhf_tight(&scell, &sprep, &shc, &eri);
        assert!(rhf.eps_r()[nocc_sc] > rhf.eps_r()[nocc_sc - 1]);
        for (d, &den) in DEN.iter().enumerate() {
            let (m, r) = gamma_corr(&scell, &rhf, &eri, exx, den);
            sc[x][d] = (m / nk, r / nk);
        }
    }

    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let prep = prep_for(cell, bs);
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &hcore_cfg()).unwrap();
    let eri = KDenseAftEri::build(
        cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::None,
        &kdense_cfg(precision),
    )
    .unwrap();
    let pairs = KDenseAftPairs::build(cell, &prep, &mesh, &kdense_cfg(precision)).unwrap();
    // The factored pair tensors reproduce the independently built kernels.
    let nkp = mesh.nk();
    let mut dk = 0.0_f64;
    for k in 0..nkp {
        for kp in 0..nkp {
            let b = pairs.block(k, kp);
            let kk = b.t().dot(&b.mapv(|z| z.conj()));
            dk = dk.max(cmax(&kk, eri.kker(k, kp)));
        }
    }
    let vm = eri.madelung_ewald();
    eprintln!(
        "{n:?}: pairs vs Kker max|d| {dk:.2e}; ranks {:?}; v_M {vm:.12}",
        (0..nkp).map(|q| pairs.rank(q)).collect::<Vec<_>>()
    );
    assert!(dk < 1e-11, "pair factor vs Kker {dk:e}");
    assert!((pairs.madelung_ewald() - vm).abs() < 1e-15);
    let k = EXX
        .iter()
        .map(|&exx| {
            krhf(
                cell,
                &mesh,
                &hk,
                &eri,
                if exx == ExxDiv::Ewald { vm } else { 0.0 },
            )
        })
        .collect();
    Anchor {
        cell: cell.clone(),
        mesh,
        pairs,
        k,
        sc,
    }
}

fn check_anchor(name: &str, a: &Anchor, tol: f64) {
    let ints = KCorrIntegrals::DenseAft(&a.pairs);
    for (x, &exx) in EXX.iter().enumerate() {
        let k = &a.k[x];
        for (d, &den) in DEN.iter().enumerate() {
            let m = kmp2(&a.cell, &a.mesh, k, ints, exx, den, None);
            let pl = kdrpa(
                &a.cell,
                &a.mesh,
                k,
                ints,
                exx,
                den,
                KDrpaEnergy::Plasmon,
                40,
                None,
            );
            let qd = kdrpa(
                &a.cell,
                &a.mesh,
                k,
                ints,
                exx,
                den,
                KDrpaEnergy::Quadrature,
                40,
                None,
            );
            let so = kdrpa(
                &a.cell,
                &a.mesh,
                k,
                ints,
                exx,
                den,
                KDrpaEnergy::SecondOrder,
                256,
                None,
            );
            let (rm, rr) = a.sc[x][d];
            eprintln!(
                "{name} {exx:?} {den:?}: MP2/cell {:.15} sc/N {rm:.15} d {:.1e}; dRPA plasmon d {:.1e}, \
                 quad-40 - plasmon {:.1e}; O(Pi^2) - direct MP2 {:.1e}",
                m.mp2_corr,
                m.mp2_corr - rm,
                pl - rr,
                qd - pl,
                so - m.e_direct
            );
            assert!((m.mp2_corr - rm).abs() <= tol, "{name} MP2 {exx:?} {den:?}");
            assert!((pl - rr).abs() <= tol, "{name} dRPA {exx:?} {den:?}");
            assert!(
                (qd - pl).abs() <= 1e-11,
                "{name} quad vs plasmon {exx:?} {den:?}"
            );
            assert!(
                (so - m.e_direct).abs() <= 1e-12,
                "{name} O(Pi^2) {exx:?} {den:?}"
            );
        }
    }
}

#[test]
fn h2_1x1x3_mesh_is_the_gamma_supercell() {
    let a = build_anchor(&h2_cell(4.0), &pyscf_sto3g_h(), [1, 1, 3], ANCHOR_PRECISION);
    check_anchor("H2 1x1x3", &a, 1e-11);
}

#[test]
fn h2_2x2x2_mesh_is_the_gamma_supercell() {
    let a = build_anchor(&h2_cell(4.0), &pyscf_sto3g_h(), [2, 2, 2], ANCHOR_PRECISION);
    check_anchor("H2 2x2x2", &a, 1e-11);
}

#[test]
#[ignore = "slow: triclinic s+p 1x1x3 k-mesh plus its 3-cell Gamma supercell (pbc_krhf.rs's analogue is ~70 s release); run with --release -- --ignored, serially"]
fn triclinic_sp_1x1x3_mesh_is_the_gamma_supercell() {
    let a = build_anchor(
        &triclinic_cell(),
        &sp_basis_h(),
        [1, 1, 3],
        ANCHOR_PRECISION,
    );
    check_anchor("tri s+p 1x1x3", &a, 1e-10);
}

// ============================================================== (c)

/// The 24 anchor aux centres (copy of `pbc_krsgdf.rs::anchor_sites`).
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

#[test]
fn trivial_aux_krsgdf_equals_the_dense_pair_oracle() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).unwrap();
    let kc = kdense_cfg(DEFAULT_DENSE_AFT_PRECISION);
    let eri = KDenseAftEri::build(&cell, &prep, &mesh, &hk.s, ExxDiv::None, &kc).unwrap();
    let pairs = KDenseAftPairs::build(&cell, &prep, &mesh, &kc).unwrap();
    let site = SiteBasis::new(&anchor_sites(&cell), 0).unwrap();
    let gdf = KRsGdf::build(
        &cell,
        &prep,
        &site.prep,
        &mesh,
        &hk.s,
        &KRsGdfConfig {
            gdf: RsGdfConfig {
                omega: 0.8,
                exxdiv: ExxDiv::None,
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            mutation: None,
        },
    )
    .unwrap();
    let k = krhf(&cell, &mesh, &hk, &eri, 0.0);
    let (dense, fit) = (
        KCorrIntegrals::DenseAft(&pairs),
        KCorrIntegrals::RsGdf(&gdf),
    );
    for den in DEN {
        let md = kmp2(&cell, &mesh, &k, dense, ExxDiv::None, den, None);
        let mg = kmp2(&cell, &mesh, &k, fit, ExxDiv::None, den, None);
        let pd = kdrpa(
            &cell,
            &mesh,
            &k,
            dense,
            ExxDiv::None,
            den,
            KDrpaEnergy::Plasmon,
            40,
            None,
        );
        let qg = kdrpa(
            &cell,
            &mesh,
            &k,
            fit,
            ExxDiv::None,
            den,
            KDrpaEnergy::Quadrature,
            40,
            None,
        );
        eprintln!(
            "trivial aux {den:?}: MP2 dense {:.15} fit d {:.1e} (direct d {:.1e}); dRPA dense plasmon \
             {pd:.15} fit quad d {:.1e}",
            md.mp2_corr,
            mg.mp2_corr - md.mp2_corr,
            mg.e_direct - md.e_direct,
            qg - pd
        );
        assert!((mg.mp2_corr - md.mp2_corr).abs() <= 1e-11, "{den:?}");
        assert!((mg.e_direct - md.e_direct).abs() <= 1e-11, "{den:?}");
        assert!((qg - pd).abs() <= 1e-11, "{den:?}");
    }
}

// ============================================================== (e)

fn pinned_kmp2(n: [usize; 3], pins: (f64, f64)) {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).unwrap();
    let kc = kdense_cfg(DEFAULT_DENSE_AFT_PRECISION);
    let eri = KDenseAftEri::build(&cell, &prep, &mesh, &hk.s, ExxDiv::None, &kc).unwrap();
    let pairs = KDenseAftPairs::build(&cell, &prep, &mesh, &kc).unwrap();
    let ints = KCorrIntegrals::DenseAft(&pairs);
    let vm = eri.madelung_ewald();
    let k_none = krhf(&cell, &mesh, &hk, &eri, 0.0);
    let k_ew = krhf(&cell, &mesh, &hk, &eri, vm);
    let un = kmp2(&cell, &mesh, &k_none, ints, ExxDiv::None, DEN[1], None).mp2_corr;
    let sh = kmp2(&cell, &mesh, &k_ew, ints, ExxDiv::Ewald, DEN[0], None).mp2_corr;
    // The same numbers through the stated-reference conversion.
    let sh_from_none = kmp2(&cell, &mesh, &k_none, ints, ExxDiv::None, DEN[0], None).mp2_corr;
    eprintln!(
        "{n:?}: unshifted {un:.15} PySCF(none) {:.15} d {:.1e}; shifted {sh:.15} PySCF(ewald) {:.15} \
         d {:.1e}; shifted from the none reference d {:.1e}",
        pins.0,
        un - pins.0,
        pins.1,
        sh - pins.1,
        sh_from_none - sh
    );
    assert!((un - pins.0).abs() < PIN_TOL, "{n:?} unshifted");
    assert!((sh - pins.1).abs() < PIN_TOL, "{n:?} shifted");
    assert!(
        (sh_from_none - sh).abs() < 1e-10,
        "{n:?} none->shifted conversion"
    );
}

#[test]
fn kmp2_h2_1x1x2_matches_pinned_pyscf_kmp2_both_exxdiv() {
    pinned_kmp2([1, 1, 2], KMP2_H2_112);
}

#[test]
fn kmp2_h2_1x1x3_complex_q_matches_pinned_pyscf_kmp2_both_exxdiv() {
    pinned_kmp2([1, 1, 3], KMP2_H2_113);
}

// ============================================================== (f)

#[test]
fn supercell_anchor_catches_momentum_conj_madelung_occupied_k_and_normalisation_mutants() {
    let a = build_anchor(&h2_cell(4.0), &pyscf_sto3g_h(), [1, 1, 3], ANCHOR_PRECISION);
    let ints = KCorrIntegrals::DenseAft(&a.pairs);
    let k = &a.k[0]; // exxdiv = none reference
    let run = |den: Mp2Denominators, mu: KCorrMutation| {
        let m = kmp2(&a.cell, &a.mesh, k, ints, ExxDiv::None, den, Some(mu)).mp2_corr;
        let r = kdrpa(
            &a.cell,
            &a.mesh,
            k,
            ints,
            ExxDiv::None,
            den,
            KDrpaEnergy::Plasmon,
            40,
            Some(mu),
        );
        (m, r)
    };
    let (rm, rr) = a.sc[0][0]; // none reference, shifted
                               // (mutation, min |dMP2| or None = must stay exact, same for dRPA)
    let cases = [
        (KCorrMutation::KbIndex, Some(5e-4), None),
        (KCorrMutation::NoConj, Some(5e-3), None),
        (KCorrMutation::MissingMadelungShift, Some(2e-3), Some(2e-3)),
        (KCorrMutation::RpaOccupiedKPlusQ, None, Some(5e-4)),
        (KCorrMutation::NkNormalization, Some(1e-3), Some(1e-3)),
    ];
    for (mu, mmin, rmin) in cases {
        let (m, r) = run(DEN[0], mu);
        let (dm, dr) = (m - rm, r - rr);
        eprintln!("mutant {mu:?} (shifted): dMP2 {dm:+.2e} d-dRPA {dr:+.2e}");
        match mmin {
            Some(t) => assert!(dm.abs() > t, "{mu:?}: MP2 not caught ({dm:e})"),
            None => assert!(dm.abs() < 1e-11, "{mu:?}: MP2 moved ({dm:e})"),
        }
        match rmin {
            Some(t) => assert!(dr.abs() > t, "{mu:?}: dRPA not caught ({dr:e})"),
            None => assert!(dr.abs() < 1e-11, "{mu:?}: dRPA moved ({dr:e})"),
        }
    }
    // The missing Madelung shift is invisible in the unshifted row.
    let (m, r) = run(DEN[1], KCorrMutation::MissingMadelungShift);
    let (um, ur) = a.sc[0][1];
    eprintln!(
        "mutant MissingMadelungShift (unshifted): dMP2 {:+.2e} d-dRPA {:+.2e}",
        m - um,
        r - ur
    );
    assert!((m - um).abs() < 1e-11 && (r - ur).abs() < 1e-11);
}

// ============================================================== refusals

#[test]
fn kcorr_refuses_bad_references_configs_and_budgets() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).unwrap();
    let kc = kdense_cfg(ANCHOR_PRECISION);
    let eri = KDenseAftEri::build(&cell, &prep, &mesh, &hk.s, ExxDiv::None, &kc).unwrap();
    let pairs = KDenseAftPairs::build(&cell, &prep, &mesh, &kc).unwrap();
    let ints = KCorrIntegrals::DenseAft(&pairs);
    let k = krhf(&cell, &mesh, &hk, &eri, 0.0);
    let ok = mp2_cfg(ExxDiv::None, DEN[0], None);
    assert!(kpoint_mp2(&cell, &mesh, &k, ints, &ok).is_ok());

    let mut bad = k.clone();
    bad.converged = false;
    assert!(kpoint_mp2(&cell, &mesh, &bad, ints, &ok).is_err());
    let mut bad = k.clone();
    bad.nocc_per_k = vec![0, 2];
    let msg = format!("{}", kpoint_mp2(&cell, &mesh, &bad, ints, &ok).unwrap_err());
    assert!(msg.contains("non-uniform occupation"), "{msg}");
    // frozen_core = nocc leaves nothing to correlate.
    let fc = KMp2Config {
        frozen_core: 1,
        ..ok
    };
    assert!(kpoint_mp2(&cell, &mesh, &k, ints, &fc).is_err());
    // Another mesh.
    let m1 = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    assert!(kpoint_mp2(&cell, &m1, &k, ints, &ok).is_err());
    // A tiny budget names the quantity.
    let tiny = KMp2Config {
        budget_bytes: Some(64),
        ..ok
    };
    let msg = format!("{}", kpoint_mp2(&cell, &mesh, &k, ints, &tiny).unwrap_err());
    assert!(msg.contains("k-point MP2"), "{msg}");
    let dtiny = KDrpaConfig {
        budget_bytes: Some(64),
        ..drpa_cfg(ExxDiv::None, DEN[0], KDrpaEnergy::Quadrature, 40, None)
    };
    let msg = format!(
        "{}",
        kpoint_drpa(&cell, &mesh, &k, ints, &dtiny).unwrap_err()
    );
    assert!(msg.contains("k-point dRPA"), "{msg}");
    // Quadrature size is validated strictly.
    let dq = drpa_cfg(ExxDiv::None, DEN[0], KDrpaEnergy::Quadrature, 3, None);
    assert!(kpoint_drpa(&cell, &mesh, &k, ints, &dq).is_err());
    // The dense oracle refuses an oversize request instead of allocating.
    let capped = KDenseAftConfig {
        max_bytes: 1024,
        ..kc
    };
    assert!(KDenseAftPairs::build(&cell, &prep, &mesh, &capped).is_err());
}
