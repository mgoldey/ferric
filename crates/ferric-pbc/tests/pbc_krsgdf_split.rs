//! k-point RS-GDF energy and forces of the RANGE-SPLIT fit
//! (`RsGdfConfig::range_split` on `KRsGdf::build`, `rsgdf::split`'s
//! `ksplit`, `kpoint::kderiv`, `kgrad`): the Rust port of
//! `reference/pbc/pbc_kgrad_gdf_split.py`, FINDINGS "Iteration 26 (Python,
//! range-split forces/stress/k-point)" item (f) and "For the Rust port"
//! items 4 and 5.
//!
//! # Anchors (written before this file's first run)
//!
//! 1. NOTHING MOVED (λ = 0 and λ = 1e-8, below every exponent) on H2 1x1x3:
//!    the split machinery runs (kept-call binned walks on parents-only piece
//!    bases, `q_c`, the kept-input G = 0 path, the split force walks) and
//!    must reproduce the unsplit `B(k, k')`, SR counts, energy and force BIT
//!    FOR BIT (the LR is the unsplit code path when no aux primitive moved,
//!    every libint factor is exactly 1, `q_c` is bitwise `q`, no smooth
//!    pieces; prototype 6.9e-18 in the force).
//! 2. 1x1x1 ≡ the Gamma split (an independent construction: real half-sphere
//!    Gamma code vs the complex residue code): fitted kernel, energy, and the
//!    force at the same density ≤ 1e-12 (prototype ≤ 6.7e-13), H2 RHF and
//!    H3 UHF.
//! 3. SUPERCELL: the 1x1x3 split k force on atom A ≡ the Gamma split force
//!    of the explicit supercell on EVERY copy of A at the unfolded density,
//!    ≤ 1e-11 (prototype 7.0e-12 / 3.0e-12 H2, 6.9e-12 / 5.4e-12 H3, with
//!    Wm symmetrised on both sides).
//! 4. FD of ferric's OWN split k energy (H2 1x1x3, h = 1e-4, both exxdiv):
//!    the unsplit k bar 1e-7 (prototype residuals reproduce Iteration 21b's
//!    −2.99e-9 on (0,z)).
//! 5. Split vs unsplit k energy, each at its own density: ≤ 1e-9 (prototype
//!    +1.9e-13 H2 1x1x3); the triclinic s+p cell at 1x1x2 (release).
//!
//! # Artifact hypotheses
//!
//! * "The split k force passes because it is secretly the unsplit one"
//!   (F_split ≡ F_unsplit to ~1e-12): every λ = 1 test asserts the moved
//!   counters are > 0 on BOTH sides and the SR triplet count DROPPED, and
//!   the mutants below mis-weight or delete only the NEW k terms.
//! * "A mutant passes because it is invisible here": each is asserted only
//!   where FINDINGS measured it visible:
//!   - `KGradMutation::SplitGammaKernel` / `KRsGdfMutation::SplitGammaKernel`
//!     (moved blocks at `v_SR(|K − q|)`): an IDENTITY at q = 0, so BLIND at
//!     1x1x1 (asserted blind there, anchor 2); H2 1x1x3 force 1.1e-3,
//!     energy +1.05e-3 Ha; H3 1.1e-3.
//!   - `SplitNoSmoothOverlap` (drop `d S_ss`): H2 1x1x3 1.1e-3, H3 2.5e-3.
//!   - `SplitFullG0` (every aux charge and `dS`): H2 2.4e-2, H3 4.8e-2.
//!   All three are asserted against the supercell anchor on 1x1x3 (> 1e-4).
//!
//! h is the Ewald-split `PeriodicHcoreK` (as `pbc_kgrad.rs`), not the
//! prototype's pure-AFT h; the fit derivative composes additively.
//!
//! NOT covered: k stress, meshes beyond 1x1x3, KS/XC, dropped metric
//! eigenvalues with a split, d orbitals, non-H atoms.

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::grad::{
    gamma_rhf_gradient_rsgdf, gamma_uhf_gradient_rsgdf, GammaGradConfig, RsGdfGradSource,
};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::kgrad::{
    kpoint_rhf_gradient, kpoint_uhf_gradient, KGradConfig, KGradJk, KGradMutation, KGradient,
    KRsGdfGradSource,
};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KPointJk, KScfConfig, KScfResult};
use ferric_pbc::kuscf::{solve_kuhf_injected, KUScfResult};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{
    KRsGdf, KRsGdfConfig, KRsGdfMutation, DEFAULT_K_FITTED_KERNELS_MAX_BYTES,
};
use ferric_pbc::rsgdf::{RangeSplit, RsGdf, RsGdfConfig, DEFAULT_FITTED_ERI_MAX_BYTES};
use ferric_scf::result::{ScfExit, ScfResult, Spin};
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

/// ω of the nuclear-attraction split (as `pbc_kgrad.rs`).
const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
/// RS-GDF ω (the prototype's w = 1; λ = 1 moves orbital a ≤ 0.5, aux α ≤ 1).
const GDF_OMEGA: f64 = 1.0;
const AMPLE: usize = 1 << 31;
const FD_H: f64 = 1e-4;
/// The unsplit k suite's FD bar (`pbc_kgrad.rs`).
const FD_BAR: f64 = 1e-7;
/// 1x1x1 ≡ Gamma split: kernel, energy, force at one density (prototype
/// ≤ 6.7e-13 in the force; the unsplit twin's bar).
const GAMMA_BAR: f64 = 1e-12;
/// 1x1x3 ≡ supercell (prototype ≤ 7.0e-12).
const SC_BAR: f64 = 1e-11;
const NET_FORCE_BAR: f64 = 1e-10;
const EWALD_NONE_BAR: f64 = 1e-11;
/// Split vs unsplit k energy, each SCF at its own density (prototype
/// +1.9e-13 / −7.9e-14).
const SPLIT_UNSPLIT_E_BAR: f64 = 1e-9;
/// Split vs unsplit k force, each at its own density (prototype 1.0e-12 /
/// 3.3e-12).
const SPLIT_UNSPLIT_F_BAR: f64 = 1e-10;
/// Smallest asserted k mutant miss in FINDINGS is 1.1e-3.
const MUTANT_BAR: f64 = 1e-4;
/// A mutant that is an identity by construction.
const BLIND_BAR: f64 = 1e-12;

/// `run_kgrad_anchor.py` H2 (Bohr), a = 4, STO-3G.
const H2_ATOMS_K: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const H2_COMPS: [(usize, usize); 3] = [(0, 0), (0, 2), (1, 1)];
/// H3 (Bohr), a = 4.5, STO-3G, doublet (2, 1) per cell.
const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
const H3_A: f64 = 4.5;
const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];

const K_SPLIT_MUTANTS: [KGradMutation; 3] = [
    KGradMutation::SplitGammaKernel,
    KGradMutation::SplitNoSmoothOverlap,
    KGradMutation::SplitFullG0,
];

fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
}

fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    }
}

fn gdf_cfg(split: Option<RangeSplit>) -> RsGdfConfig {
    RsGdfConfig {
        omega: GDF_OMEGA,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        range_split: split,
        ..Default::default()
    }
}

fn kgdf_cfg(split: Option<RangeSplit>, mutation: Option<KRsGdfMutation>) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: gdf_cfg(split),
        mutation,
    }
}

fn split(lambda: f64) -> Option<RangeSplit> {
    Some(RangeSplit::new(lambda))
}

fn gcfg(mutation: Option<KGradMutation>) -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        mutation,
        ..Default::default()
    }
}

fn gamma_gcfg() -> GammaGradConfig {
    GammaGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

/// The prototype's ET-sp aux on H: s(4.7, 1.9, 0.75, 0.3) + p(1.25, 0.5),
/// one unit-normalised primitive per shell. λ = 1 moves s 0.75, 0.3 and
/// p 0.5 (plus the STO-3G 0.1689 orbital primitive).
fn et_sp_aux() -> BasisSet {
    let mut shells = Vec::new();
    for (l, a) in [(0, 4.7), (0, 1.9), (0, 0.75), (0, 0.3), (1, 1.25), (1, 0.5)] {
        shells.push(Shell {
            l,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-sp-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

fn cell_at(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

fn moved_pos(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

fn h2_cell_k() -> Cell {
    cell_at(&H2_ATOMS_K, cubic(4.0), 1)
}

fn h3_cell() -> Cell {
    cell_at(&H3_ATOMS, cubic(H3_A), 2)
}

/// Explicit diag(n) supercell (copy c = (m0, m1, m2), m0 outer) with
/// multiplicity `mult` (as `pbc_kgrad.rs`).
fn supercell(cell: &Cell, n: [usize; 3], mult: usize) -> Cell {
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
    cell_at(&atoms, lat, mult)
}

/// The k density unfolded to the explicit supercell (as `pbc_kgrad.rs`).
fn unfold(mesh: &KPointMesh, dk: &[Array2<Complex64>]) -> Array2<f64> {
    let n = mesh.n();
    let nao = dk[0].nrows();
    let nk = mesh.nk();
    let copies: Vec<[i64; 3]> = (0..n[0])
        .flat_map(|m0| {
            (0..n[1]).flat_map(move |m1| (0..n[2]).map(move |m2| [m0 as i64, m1 as i64, m2 as i64]))
        })
        .collect();
    let nc = copies.len();
    let mut out = Array2::<f64>::zeros((nc * nao, nc * nao));
    let mut max_im = 0.0_f64;
    for (c, tc) in copies.iter().enumerate() {
        for (c2, tc2) in copies.iter().enumerate() {
            let nl = [tc[0] - tc2[0], tc[1] - tc2[1], tc[2] - tc2[2]];
            for m in 0..nao {
                for nn in 0..nao {
                    let mut z = Complex64::new(0.0, 0.0);
                    for (k, d) in dk.iter().enumerate() {
                        z += mesh.phase(k, nl) * d[(m, nn)];
                    }
                    z /= nk as f64;
                    max_im = max_im.max(z.im.abs());
                    out[(c * nao + m, c2 * nao + nn)] = z.re;
                }
            }
        }
    }
    assert!(max_im < 1e-10, "unfolded density not real: {max_im:e}");
    out
}

fn gamma_restricted(d: Array2<f64>) -> ScfResult {
    let n = d.nrows();
    ScfResult {
        spin: Spin::Restricted,
        energy: 0.0,
        density_alpha: &d * 0.5,
        density_total: d,
        density_beta: None,
        mos_alpha: Array2::eye(n),
        mos_beta: None,
        eps_alpha: vec![0.0; n],
        eps_beta: None,
        fock_alpha: Array2::zeros((n, n)),
        fock_beta: None,
        converged: true,
        exit: ScfExit::Converged,
        iterations: 0,
        computed_quartets: 0,
        induced_dipoles: None,
        stability: None,
        df_jk: None,
        rohf_spin_focks: None,
    }
}

fn gamma_unrestricted(da: Array2<f64>, db: Array2<f64>) -> ScfResult {
    let n = da.nrows();
    ScfResult {
        spin: Spin::Unrestricted,
        energy: 0.0,
        density_total: &da + &db,
        density_alpha: da,
        density_beta: Some(db),
        mos_alpha: Array2::eye(n),
        mos_beta: Some(Array2::eye(n)),
        eps_alpha: vec![0.0; n],
        eps_beta: Some(vec![0.0; n]),
        fock_alpha: Array2::zeros((n, n)),
        fock_beta: Some(Array2::zeros((n, n))),
        converged: true,
        exit: ScfExit::Converged,
        iterations: 0,
        computed_quartets: 0,
        induced_dipoles: None,
        stability: None,
        df_jk: None,
        rohf_spin_focks: None,
    }
}

fn real_of(d: &Array2<Complex64>) -> Array2<f64> {
    let im = d.iter().fold(0.0_f64, |m, z| m.max(z.im.abs()));
    assert!(im < 1e-12, "Gamma density has an imaginary part {im:e}");
    d.mapv(|z| z.re)
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

fn bitwise(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.dim() == b.dim()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn c_bitwise(a: &Array2<Complex64>, b: &Array2<Complex64>) -> bool {
    a.dim() == b.dim()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits())
}

fn c_max_diff(a: &Array2<Complex64>, b: &Array2<Complex64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).norm()))
}

/// `max_{c, A, x} |F_k[A, x] − F_sc[c·natoms + A, x]|` (every copy).
fn sc_miss(gk: &Array2<f64>, gsc: &Array2<f64>) -> f64 {
    let na = gk.nrows();
    let nc = gsc.nrows() / na;
    assert_eq!(nc * na, gsc.nrows());
    let mut w = 0.0_f64;
    for c in 0..nc {
        for a in 0..na {
            for x in 0..3 {
                w = w.max((gk[(a, x)] - gsc[(c * na + a, x)]).abs());
            }
        }
    }
    w
}

// ---------------------------------------------------------------- setups

struct KSys {
    cell: Cell,
    prep: PreparedBasis,
    aux: PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
}

fn ksys(cell: Cell, bs: &BasisSet, aux_bs: &BasisSet, n: [usize; 3]) -> KSys {
    let prep = prep_for(&cell, bs);
    let aux = PreparedBasis::new(cell.mol(), aux_bs).expect("aux prep");
    let mesh = KPointMesh::gamma_centred(&cell, n).expect("mesh");
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg()).expect("hcore(k)");
    KSys {
        cell,
        prep,
        aux,
        mesh,
        hk,
    }
}

fn kgdf(ks: &KSys, cfg: &KRsGdfConfig) -> KRsGdf {
    KRsGdf::build(&ks.cell, &ks.prep, &ks.aux, &ks.mesh, &ks.hk.s, cfg).expect("KRsGdf")
}

fn inj<'a>(ks: &KSys, jk: Box<dyn KPointJk + 'a>) -> KPointInjection<'a> {
    KPointInjection {
        s: ks.hk.s.clone(),
        h: ks.hk.h.clone(),
        vnn: ks.hk.enn,
        jk,
    }
}

fn krhf(ks: &KSys, jk: Box<dyn KPointJk + '_>) -> KScfResult {
    let r = solve_krhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj(ks, jk)).expect("k-RHF");
    assert!(r.converged, "k-RHF not converged ({} it)", r.iterations);
    r
}

fn kuhf(ks: &KSys, jk: Box<dyn KPointJk + '_>, na: usize, nb: usize) -> KUScfResult {
    let r =
        solve_kuhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj(ks, jk), na, nb).expect("k-UHF");
    assert!(r.converged, "k-UHF not converged ({} it)", r.iterations);
    r
}

/// k-RHF on `gdf` with the Madelung term of `exx`.
fn krhf_exx(ks: &KSys, gdf: &KRsGdf, exx: ExxDiv) -> KScfResult {
    let vm = match exx {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => ks.mesh.madelung(&ks.cell).expect("madelung"),
    };
    krhf(ks, Box::new(gdf.jk_builder_with_madelung(vm)))
}

fn src<'a>(ks: &'a KSys, gdf: &'a KRsGdf, cfg: &'a KRsGdfConfig) -> KRsGdfGradSource<'a> {
    KRsGdfGradSource {
        gdf,
        cfg,
        aux: &ks.aux,
    }
}

fn kg_rhf(
    ks: &KSys,
    jk: KGradJk<'_>,
    scf: &KScfResult,
    exx: ExxDiv,
    m: Option<KGradMutation>,
) -> KGradient {
    kpoint_rhf_gradient(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        jk,
        scf,
        exx,
        &gcfg(m),
    )
    .expect("kpoint_rhf_gradient")
}

fn kg_uhf(
    ks: &KSys,
    jk: KGradJk<'_>,
    scf: &KUScfResult,
    exx: ExxDiv,
    m: Option<KGradMutation>,
) -> KGradient {
    kpoint_uhf_gradient(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(),
        &ks.hk,
        jk,
        scf,
        exx,
        &gcfg(m),
    )
    .expect("kpoint_uhf_gradient")
}

/// A counter of the range split's partition (panics when the split path
/// did not run).
fn counter(gdf: &KRsGdf, name: &str) -> usize {
    gdf.stats()
        .split_counters
        .iter()
        .find(|(n, _)| *n == name)
        .map(|&(_, v)| v)
        .unwrap_or_else(|| panic!("counter {name:?} missing (split path did not run?)"))
}

/// `(orbital smooth, orbital total, aux smooth, aux total)` primitives.
fn moved(gdf: &KRsGdf) -> [usize; 4] {
    [
        counter(gdf, "rsgdf split orbital prims smooth"),
        counter(gdf, "rsgdf split orbital prims"),
        counter(gdf, "rsgdf split aux prims smooth"),
        counter(gdf, "rsgdf split aux prims"),
    ]
}

/// Something moved on BOTH sides and the SR walk shrank vs `unsplit`.
fn assert_split_active(tag: &str, gdf: &KRsGdf, unsplit: &KRsGdf) {
    let m = moved(gdf);
    let (s, u) = (gdf.stats(), unsplit.stats());
    eprintln!(
        "{tag}: moved [orb smooth, orb, aux smooth, aux] = {m:?}; SR3 {} → {}, SR2 {} → {}",
        u.n_sr3_triplets, s.n_sr3_triplets, u.n_sr2_pairs, s.n_sr2_pairs
    );
    assert!(m[0] > 0 && m[2] > 0, "{tag}: nothing moved {m:?}");
    assert!(
        s.n_sr3_triplets < u.n_sr3_triplets,
        "{tag}: SR3 did not shrink"
    );
    assert!(
        unsplit.stats().split_counters.is_empty(),
        "{tag}: the unsplit build ran the split path"
    );
}

/// The Gamma split setup of `cell` (the explicit supercell, or the cell
/// itself for 1x1x1): hcore, aux, and the split gradient build.
struct GSys {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    aux: PreparedBasis,
    gdf: RsGdf,
}

fn gsys(cell: Cell, bs: &BasisSet, aux_bs: &BasisSet, rs: Option<RangeSplit>) -> GSys {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).expect("gamma hcore");
    let aux = PreparedBasis::new(cell.mol(), aux_bs).expect("aux prep");
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &gdf_cfg(rs))
        .expect("RsGdf::build_for_gradient");
    GSys {
        cell,
        prep,
        hc,
        aux,
        gdf,
    }
}

fn gamma_grad(g: &GSys, scf: &ScfResult, exx: ExxDiv) -> Array2<f64> {
    let src = RsGdfGradSource {
        gdf: &g.gdf,
        aux: &g.aux,
        aux_jac: None,
    };
    let gc = gamma_gcfg();
    if scf.spin == Spin::Restricted {
        gamma_rhf_gradient_rsgdf(&g.cell, &g.prep, &hcore_cfg(), &g.hc, &src, scf, exx, &gc)
    } else {
        gamma_uhf_gradient_rsgdf(&g.cell, &g.prep, &hcore_cfg(), &g.hc, &src, scf, exx, &gc)
    }
    .expect("gamma split gradient")
    .grad
}

fn report(tag: &str, g: &KGradient) {
    eprintln!(
        "  {tag}: |F| {:.3e} ΣF {:.1e} comm {:.1e} v_M {:.6} (SR {}, K {}, chunks {}, dropped \
         {:?})",
        g.grad.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
        g.net_force,
        g.commutator,
        g.madelung,
        g.n_sr_triplets,
        g.n_k_eri,
        g.n_chunks,
        g.fit_dropped_max
    );
}

// ================================================= (1) nothing moved

/// λ = 0 and λ = 1e-8 move nothing on H2 1x1x3: every `B(k, k')`, the SR
/// counts, the energy (both exxdiv) and the force are the unsplit ones BIT
/// FOR BIT (module doc).
#[test]
fn nothing_moved_split_k_build_energy_and_force_are_the_unsplit_ones() {
    let ks = ksys(h2_cell_k(), &pyscf_sto3g_h(), &et_sp_aux(), [1, 1, 3]);
    let cfg_u = kgdf_cfg(None, None);
    let un = kgdf(&ks, &cfg_u);
    assert!(un.stats().split_counters.is_empty());
    let nk = ks.mesh.nk();
    let scf_u: Vec<KScfResult> = EXX.iter().map(|&e| krhf_exx(&ks, &un, e)).collect();
    for lam in [0.0, 1e-8] {
        let cfg_s = kgdf_cfg(split(lam), None);
        let sp = kgdf(&ks, &cfg_s);
        let m = moved(&sp);
        assert!(
            m[1] > 0 && m[3] > 0,
            "λ = {lam}: split path did not run {m:?}"
        );
        assert_eq!((m[0], m[2]), (0, 0), "λ = {lam} moved something: {m:?}");
        let (s0, s1) = (un.stats(), sp.stats());
        assert_eq!(s1.n_sr3_triplets, s0.n_sr3_triplets);
        assert_eq!(s1.n_sr2_pairs, s0.n_sr2_pairs);
        let mut worst = 0.0_f64;
        let mut all_bitwise = true;
        for k in 0..nk {
            for kp in 0..nk {
                let (bs, bu) = (sp.block(k, kp), un.block(k, kp));
                worst = worst.max(c_max_diff(&bs, &bu));
                all_bitwise &= c_bitwise(&bs, &bu);
            }
        }
        eprintln!("λ = {lam}: max|B_split − B_unsplit| = {worst:.2e} (bitwise: {all_bitwise})");
        assert!(all_bitwise, "λ = {lam}: B(k, k') not bitwise ({worst:.2e})");
        for (i, exx) in EXX.into_iter().enumerate() {
            let scf_s = krhf_exx(&ks, &sp, exx);
            assert_eq!(
                scf_s.energy.to_bits(),
                scf_u[i].energy.to_bits(),
                "λ = {lam} {exx:?}: E {} vs {}",
                scf_s.energy,
                scf_u[i].energy
            );
            let gu = kg_rhf(
                &ks,
                KGradJk::RsGdf(src(&ks, &un, &cfg_u)),
                &scf_u[i],
                exx,
                None,
            );
            let gs = kg_rhf(
                &ks,
                KGradJk::RsGdf(src(&ks, &sp, &cfg_s)),
                &scf_s,
                exx,
                None,
            );
            let d = max_diff(&gs.grad, &gu.grad);
            eprintln!(
                "λ = {lam} {exx:?}: |F_split − F_unsplit| = {d:.2e} (bitwise: {})",
                bitwise(&gs.grad, &gu.grad)
            );
            let (cs, cu) = (
                gs.fit_checks.as_ref().unwrap(),
                gu.fit_checks.as_ref().unwrap(),
            );
            assert_eq!((cs.n_sr3, cs.n_sr2), (cu.n_sr3, cu.n_sr2));
            assert!(
                bitwise(&gs.grad, &gu.grad),
                "λ = {lam} {exx:?}: force not bitwise ({d:.2e}); the code paths should coincide"
            );
        }
    }
}

// ================================================= (2) 1x1x1 ≡ Gamma split

/// H2 RHF at 1x1x1, λ = 1: the k split fitted kernel and SCF energy equal
/// the Gamma split's; the k split force equals the Gamma split force at the
/// same density (both exxdiv); the Gamma-kernel mutants are IDENTITIES at
/// q = 0 (energy B bitwise, force ≤ 1e-12).
#[test]
fn one_point_mesh_split_is_the_gamma_split_rhf() {
    let (bs, aux_bs) = (pyscf_sto3g_h(), et_sp_aux());
    let cell = h2_cell_k();
    let ks = ksys(cell.clone(), &bs, &aux_bs, [1, 1, 1]);
    let cfg_s = kgdf_cfg(split(1.0), None);
    let kg = kgdf(&ks, &cfg_s);
    assert_split_active("H2 1x1x1 λ=1", &kg, &kgdf(&ks, &kgdf_cfg(None, None)));
    let g = gsys(cell.clone(), &bs, &aux_bs, split(1.0));

    // Fitted kernel (the Gamma ERI BᵀB; rotation-invariant, so eigenvector
    // phases do not enter).
    let (jf, kf) = kg
        .fitted_kernels(DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
        .unwrap();
    let eri = g.gdf.fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES).unwrap();
    let dker = jf[0]
        .iter()
        .zip(eri.iter())
        .chain(kf[0].iter().zip(eri.iter()))
        .fold(0.0_f64, |m, (x, y)| {
            m.max((x - Complex64::new(*y, 0.0)).norm())
        });
    eprintln!("H2 1x1x1 split vs Gamma split: max|Δkernel| {dker:.2e}");
    assert!(dker < GAMMA_BAR, "kernel {dker:.3e}");

    // The Gamma-kernel energy mutant is an identity at q = 0.
    let km = kgdf(
        &ks,
        &kgdf_cfg(split(1.0), Some(KRsGdfMutation::SplitGammaKernel)),
    );
    let db = c_max_diff(&km.block(0, 0), &kg.block(0, 0));
    eprintln!(
        "  energy mutant SplitGammaKernel at 1x1x1: max|ΔB| {db:.2e} (blind by construction)"
    );
    assert!(
        db <= BLIND_BAR,
        "SplitGammaKernel is not an identity at q = 0: {db:e}"
    );

    for exx in EXX {
        // Energy: each SCF at its own density.
        let gg = g.gdf.clone().with_exxdiv(&g.cell, exx).unwrap();
        let eg = gamma_rhf_jk(
            &g.cell,
            &g.prep,
            &g.hc,
            Box::new(gg.j_builder()),
            Box::new(gg.k_builder()),
        )
        .energy;
        let scf = krhf_exx(&ks, &kg, exx);
        eprintln!(
            "  {exx:?}: E_gamma {eg:.12} E_k {:.12} dE {:.1e}",
            scf.energy,
            scf.energy - eg
        );
        assert!(
            (scf.energy - eg).abs() < GAMMA_BAR,
            "{exx:?}: dE {:e}",
            scf.energy - eg
        );

        // Force at ONE density.
        let fk = kg_rhf(&ks, KGradJk::RsGdf(src(&ks, &kg, &cfg_s)), &scf, exx, None);
        let fg = gamma_grad(&g, &gamma_restricted(real_of(&scf.densities[0])), exx);
        let d = max_diff(&fk.grad, &fg);
        report(&format!("H2 split 1x1x1 {exx:?}"), &fk);
        eprintln!(
            "  H2 split 1x1x1 {exx:?}: |F_k − F_gamma| {d:.2e} (prototype 1.5e-13 / 2.2e-13)"
        );
        assert!(d <= GAMMA_BAR, "{exx:?}: {d:e}");
        assert!(fk.net_force <= NET_FORCE_BAR, "ΣF {:e}", fk.net_force);
        let fm = kg_rhf(
            &ks,
            KGradJk::RsGdf(src(&ks, &kg, &cfg_s)),
            &scf,
            exx,
            Some(KGradMutation::SplitGammaKernel),
        );
        let dm = max_diff(&fm.grad, &fk.grad);
        eprintln!(
            "  force mutant SplitGammaKernel at 1x1x1: |ΔF| {dm:.2e} (blind by construction)"
        );
        assert!(
            dm <= BLIND_BAR,
            "{exx:?}: SplitGammaKernel not blind at 1x1x1: {dm:e}"
        );
    }
}

/// H3 UHF (2,1) at 1x1x1, λ = 1: the k split force equals the Gamma split
/// force at the same spin densities (prototype 4.4e-13 / 6.7e-13).
#[test]
fn one_point_mesh_split_is_the_gamma_split_uhf() {
    let (bs, aux_bs) = (pyscf_sto3g_h(), et_sp_aux());
    let cell = h3_cell();
    let ks = ksys(cell.clone(), &bs, &aux_bs, [1, 1, 1]);
    let cfg_s = kgdf_cfg(split(1.0), None);
    let kg = kgdf(&ks, &cfg_s);
    let m = moved(&kg);
    assert!(m[0] > 0 && m[2] > 0, "nothing moved {m:?}");
    let g = gsys(cell, &bs, &aux_bs, split(1.0));
    let scf = kuhf(&ks, Box::new(kg.jk_builder_with_madelung(0.0)), 2, 1);
    let gscf = gamma_unrestricted(
        real_of(&scf.density_alpha[0]),
        real_of(&scf.density_beta[0]),
    );
    for exx in EXX {
        let fk = kg_uhf(&ks, KGradJk::RsGdf(src(&ks, &kg, &cfg_s)), &scf, exx, None);
        let fg = gamma_grad(&g, &gscf, exx);
        let d = max_diff(&fk.grad, &fg);
        report(&format!("H3 UHF split 1x1x1 {exx:?}"), &fk);
        eprintln!("  H3 UHF split 1x1x1 {exx:?}: |F_k − F_gamma| {d:.2e}");
        assert!(d <= GAMMA_BAR, "{exx:?}: {d:e}");
    }
}

// ============================================ (5) split vs unsplit k energy

/// H2 1x1x3, λ = 1 vs unsplit, each SCF at its own density, both exxdiv.
#[test]
fn h2_1x1x3_split_k_energy_matches_the_unsplit_one() {
    let ks = ksys(h2_cell_k(), &pyscf_sto3g_h(), &et_sp_aux(), [1, 1, 3]);
    let sp = kgdf(&ks, &kgdf_cfg(split(1.0), None));
    let un = kgdf(&ks, &kgdf_cfg(None, None));
    assert_split_active("H2 1x1x3 λ=1", &sp, &un);
    for exx in EXX {
        let (es, eu) = (
            krhf_exx(&ks, &sp, exx).energy,
            krhf_exx(&ks, &un, exx).energy,
        );
        eprintln!(
            "  H2 1x1x3 {exx:?}: E_split − E_unsplit = {:+.2e} (prototype +1.9e-13)",
            es - eu
        );
        assert!(
            (es - eu).abs() <= SPLIT_UNSPLIT_E_BAR,
            "{exx:?}: {:e}",
            es - eu
        );
    }
}

#[test]
#[ignore = "slow (unmeasured): triclinic 4H s+p / cc-pvdz-ri k builds at 1x1x2, split and \
            unsplit, and their SCFs; run with --release -- --ignored"]
fn triclinic_1x1x2_split_k_energy_matches_the_unsplit_one() {
    let ks = ksys(
        triclinic_cell(),
        &sp_basis_h(),
        &basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri"),
        [1, 1, 2],
    );
    let sp = kgdf(&ks, &kgdf_cfg(split(1.0), None));
    let un = kgdf(&ks, &kgdf_cfg(None, None));
    assert_split_active("tri s+p 1x1x2 λ=1", &sp, &un);
    let m = moved(&sp);
    assert!(m[0] < m[1] && m[2] < m[3], "moved everything: {m:?}");
    let (es, eu) = (
        krhf_exx(&ks, &sp, ExxDiv::None).energy,
        krhf_exx(&ks, &un, ExxDiv::None).energy,
    );
    eprintln!(
        "  tri 1x1x2: E_split − E_unsplit = {:+.2e} (Gamma tri: −1.9e-14)",
        es - eu
    );
    assert!((es - eu).abs() <= SPLIT_UNSPLIT_E_BAR, "{:e}", es - eu);
}

// ========================================== (4) FD of the own split k energy

/// Split k-RHF energies `[none, ewald]` of H2 at `pos`, 1x1x3, λ = 1 (the
/// split build rebuilt at every displacement).
fn h2_split_energies(pos: &[[f64; 3]]) -> [f64; 2] {
    let ks = ksys(
        cell_at(pos, cubic(4.0), 1),
        &pyscf_sto3g_h(),
        &et_sp_aux(),
        [1, 1, 3],
    );
    let gdf = kgdf(&ks, &kgdf_cfg(split(1.0), None));
    [
        krhf_exx(&ks, &gdf, ExxDiv::None).energy,
        krhf_exx(&ks, &gdf, ExxDiv::Ewald).energy,
    ]
}

#[test]
#[ignore = "slow (unmeasured): 6 displaced H2 1x1x3 split k builds + 12 k-SCFs; run with \
            --release -- --ignored"]
fn h2_1x1x3_split_force_matches_fd_of_own_k_energy() {
    let ks = ksys(h2_cell_k(), &pyscf_sto3g_h(), &et_sp_aux(), [1, 1, 3]);
    let cfg_s = kgdf_cfg(split(1.0), None);
    let gdf = kgdf(&ks, &cfg_s);
    let m = moved(&gdf);
    assert!(m[0] > 0 && m[2] > 0, "nothing moved {m:?}");
    let fd: Vec<((usize, usize), [f64; 2])> = H2_COMPS
        .iter()
        .map(|&(a, x)| {
            let ep = h2_split_energies(&moved_pos(&H2_ATOMS_K, a, x, FD_H));
            let em = h2_split_energies(&moved_pos(&H2_ATOMS_K, a, x, -FD_H));
            (
                (a, x),
                [
                    (ep[0] - em[0]) / (2.0 * FD_H),
                    (ep[1] - em[1]) / (2.0 * FD_H),
                ],
            )
        })
        .collect();
    for (i, exx) in EXX.into_iter().enumerate() {
        let scf = krhf_exx(&ks, &gdf, exx);
        let g = kg_rhf(&ks, KGradJk::RsGdf(src(&ks, &gdf, &cfg_s)), &scf, exx, None);
        report(&format!("H2 1x1x3 split {exx:?}"), &g);
        let mut worst = 0.0_f64;
        for ((a, x), v) in &fd {
            let d = g.grad[(*a, *x)] - v[i];
            eprintln!(
                "  ({a},{x}) {exx:?}: analytic {:+.10e} FD {:+.10e} diff {d:+.2e} (prototype \
                 +6.8e-11 / −2.99e-9 / +9.7e-11)",
                g.grad[(*a, *x)],
                v[i]
            );
            worst = worst.max(d.abs());
        }
        assert!(worst <= FD_BAR, "{exx:?}: FD miss {worst:e}");
        assert!(g.net_force <= NET_FORCE_BAR, "ΣF {:e}", g.net_force);
    }
}

// ============================= (3) + (6) supercell anchor and the k mutants

/// H2 RHF 1x1x3, λ = 1: the k split force ≡ the split supercell force on
/// every copy (both exxdiv); ΣF; ewald ≡ none; split vs unsplit force at
/// their own densities; the three k split force mutants and the
/// Gamma-kernel ENERGY mutant are caught (and the energy mutant leaves the
/// q = 0 class bitwise).
#[test]
#[ignore = "slow (unmeasured): H2 1x1x3 split + unsplit + mutant k builds and SCFs, and a \
            6-atom split supercell RS-GDF gradient build; run with --release -- --ignored"]
fn h2_1x1x3_split_force_is_the_split_supercell_force_and_catches_the_k_mutants() {
    let n = [1, 1, 3];
    let (bs, aux_bs) = (pyscf_sto3g_h(), et_sp_aux());
    let cell = h2_cell_k();
    let ks = ksys(cell.clone(), &bs, &aux_bs, n);
    let cfg_s = kgdf_cfg(split(1.0), None);
    let cfg_u = kgdf_cfg(None, None);
    let kg = kgdf(&ks, &cfg_s);
    let ku = kgdf(&ks, &cfg_u);
    assert_split_active("H2 1x1x3 λ=1", &kg, &ku);
    let g = gsys(supercell(&cell, n, 1), &bs, &aux_bs, split(1.0));
    let mut refs: Vec<Array2<f64>> = Vec::new();
    let mut fks: Vec<Array2<f64>> = Vec::new();
    let mut scfs: Vec<KScfResult> = Vec::new();
    for exx in EXX {
        let scf = krhf_exx(&ks, &kg, exx);
        let fk = kg_rhf(&ks, KGradJk::RsGdf(src(&ks, &kg, &cfg_s)), &scf, exx, None);
        let fsc = gamma_grad(&g, &gamma_restricted(unfold(&ks.mesh, &scf.densities)), exx);
        let miss = sc_miss(&fk.grad, &fsc);
        report(&format!("H2 split {n:?} {exx:?}"), &fk);
        eprintln!(
            "  H2 split {n:?} {exx:?}: supercell miss {miss:.2e} (prototype 7.0e-12 / 3.0e-12)"
        );
        assert!(miss <= SC_BAR, "{exx:?}: {miss:e}");
        assert!(fk.net_force <= NET_FORCE_BAR, "ΣF {:e}", fk.net_force);
        // Split vs unsplit force, each at its own density (Q5).
        let scf_u = krhf_exx(&ks, &ku, exx);
        let fu = kg_rhf(
            &ks,
            KGradJk::RsGdf(src(&ks, &ku, &cfg_u)),
            &scf_u,
            exx,
            None,
        );
        let du = max_diff(&fk.grad, &fu.grad);
        eprintln!(
            "  E_split − E_unsplit {:+.2e}; |F_split − F_unsplit| {du:.2e} (prototype 1.0e-12)",
            scf.energy - scf_u.energy
        );
        assert!(
            du <= SPLIT_UNSPLIT_F_BAR,
            "{exx:?}: split vs unsplit {du:e}"
        );
        fks.push(fk.grad);
        refs.push(fsc);
        scfs.push(scf);
    }
    let de = max_diff(&fks[0], &fks[1]);
    eprintln!("  H2 split: |F(ewald) − F(none)| {de:.2e} (prototype 2.7e-13)");
    assert!(de <= EWALD_NONE_BAR, "ewald vs none {de:e}");
    for m in K_SPLIT_MUTANTS {
        let f = kg_rhf(
            &ks,
            KGradJk::RsGdf(src(&ks, &kg, &cfg_s)),
            &scfs[0],
            ExxDiv::None,
            Some(m),
        );
        let miss = sc_miss(&f.grad, &refs[0]);
        eprintln!(
            "  H2 split mutant {m:?}: supercell miss {miss:.2e}, ΣF {:.1e}",
            f.net_force
        );
        // Prototype: 1.1e-3 / 1.1e-3 / 2.4e-2.
        assert!(miss > MUTANT_BAR, "mutant {m:?} not caught ({miss:e})");
    }
    // The ENERGY with the Gamma kernel on the moved blocks. Measured
    // 2026-09-28: on this cell the wrong q ≠ 0 kernel makes the split metric
    // indefinite (smallest eigenvalue −8.5e-3 at q class 1), so the
    // negative-eigenvalue guard REFUSES the build — the mutant is caught by
    // the guard. If a build ever gets through, the q = 0 class must be
    // untouched (bitwise) and the energy must move by > MUTANT_BAR.
    let built = KRsGdf::build(
        &ks.cell,
        &ks.prep,
        &ks.aux,
        &ks.mesh,
        &ks.hk.s,
        &kgdf_cfg(split(1.0), Some(KRsGdfMutation::SplitGammaKernel)),
    );
    match built {
        Err(e) => {
            let msg = format!("{e:?}");
            eprintln!("  energy mutant SplitGammaKernel: refused by the metric guard ({msg})");
            assert!(
                msg.contains("smallest metric eigenvalue") && !msg.contains("q class 0"),
                "mutant refused for the wrong reason: {msg}"
            );
        }
        Ok(km) => {
            let mut dq0 = 0.0_f64;
            for k in 0..ks.mesh.nk() {
                dq0 = dq0.max(c_max_diff(&km.block(k, k), &kg.block(k, k)));
            }
            let em = krhf_exx(&ks, &km, ExxDiv::None).energy;
            eprintln!(
                "  energy mutant SplitGammaKernel: E_mut − E = {:+.2e} (prototype +1.05e-3 at the \
                 same D); q = 0 max|ΔB| {dq0:.1e}",
                em - scfs[0].energy
            );
            assert!(dq0 <= BLIND_BAR, "the mutant touched q = 0: {dq0:e}");
            assert!(
                (em - scfs[0].energy).abs() > MUTANT_BAR,
                "energy mutant not caught ({:e})",
                em - scfs[0].energy
            );
        }
    }
}

/// H3 UHF (2,1) 1x1x3, λ = 1: the k split force ≡ the split supercell force
/// on every copy (both exxdiv) and the k split mutants are caught.
#[test]
#[ignore = "slow (unmeasured): H3 1x1x3 split k build + k-UHF and a 9-atom split supercell \
            RS-GDF gradient build; run with --release -- --ignored"]
fn h3_uhf_1x1x3_split_force_is_the_split_supercell_force() {
    let n = [1, 1, 3];
    let (bs, aux_bs) = (pyscf_sto3g_h(), et_sp_aux());
    let cell = h3_cell();
    let ks = ksys(cell.clone(), &bs, &aux_bs, n);
    let cfg_s = kgdf_cfg(split(1.0), None);
    let kg = kgdf(&ks, &cfg_s);
    let m = moved(&kg);
    assert!(m[0] > 0 && m[2] > 0, "nothing moved {m:?}");
    let scf = kuhf(&ks, Box::new(kg.jk_builder_with_madelung(0.0)), 2, 1);
    let g = gsys(supercell(&cell, n, 4), &bs, &aux_bs, split(1.0));
    let gscf = gamma_unrestricted(
        unfold(&ks.mesh, &scf.density_alpha),
        unfold(&ks.mesh, &scf.density_beta),
    );
    let mut refs: Vec<Array2<f64>> = Vec::new();
    for exx in EXX {
        let fk = kg_uhf(&ks, KGradJk::RsGdf(src(&ks, &kg, &cfg_s)), &scf, exx, None);
        let fsc = gamma_grad(&g, &gscf, exx);
        let miss = sc_miss(&fk.grad, &fsc);
        report(&format!("H3 UHF split {n:?} {exx:?}"), &fk);
        eprintln!(
            "  H3 UHF split {n:?} {exx:?}: supercell miss {miss:.2e} (prototype 6.9e-12 / 5.4e-12)"
        );
        assert!(miss <= SC_BAR, "{exx:?}: {miss:e}");
        assert!(fk.net_force <= NET_FORCE_BAR, "ΣF {:e}", fk.net_force);
        refs.push(fsc);
    }
    for m in K_SPLIT_MUTANTS {
        let f = kg_uhf(
            &ks,
            KGradJk::RsGdf(src(&ks, &kg, &cfg_s)),
            &scf,
            ExxDiv::None,
            Some(m),
        );
        let miss = sc_miss(&f.grad, &refs[0]);
        eprintln!("  H3 UHF split mutant {m:?}: supercell miss {miss:.2e}");
        // Prototype: 1.1e-3 / 2.5e-3 / 4.8e-2.
        assert!(miss > MUTANT_BAR, "mutant {m:?} not caught ({miss:e})");
    }
}
