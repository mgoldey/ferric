//! Gamma RS-GDF forces and stress of the RANGE-SPLIT fit
//! (`RsGdfConfig::range_split`, `rsgdf::split`'s `deriv`): the Rust port of
//! `reference/pbc/pbc_grad_gdf_split.py`, FINDINGS "Iteration 26 (Python,
//! range-split forces/stress/k-point)".
//!
//! # Anchors (written before any split derivative code ran)
//!
//! 1. NOTHING MOVED (λ = 0 and λ = 1e-8, below every exponent): the split
//!    machinery runs (combined piece bases, the kept-call walks, `q_c`, the
//!    split G = 0 path) and must reproduce the unsplit Iteration-18/19 force
//!    and stress. Expected BITWISE: the SR walks visit the same triplets in
//!    the same order through parents-only combined bases with every libint
//!    factor exactly 1, the LR part IS the unsplit code path (no aux
//!    primitive moved), `q_c` is bitwise `q`, and there is no smooth-piece
//!    pass. Asserted bitwise, with ≤ 1e-13 as the reported fallback.
//! 2. FD of ferric's OWN split energy at λ = 1 (the split `RsGdf::build`
//!    rebuilt at every displacement / strain): the unsplit suites' bars
//!    (forces 1e-7, p shells 5e-7; stress 1e-7, triclinic Richardson 5e-6).
//!    The prototype's FD residuals reproduce the unsplit ones to 2–3 digits.
//! 3. INDEPENDENT CONSTRUCTION: the exact aux span with EVERYTHING moved
//!    (ω = 1.2: the kept SR walks are empty) vs the dense pure-AFT force
//!    (prototype 3.1e-13).
//!
//! # Artifact hypotheses
//!
//! * "The split force passes FD because it is secretly the unsplit force"
//!   (F_split ≡ F_unsplit to ~1e-12, prototype Q5): every λ = 1 test asserts
//!   the moved counters are > 0 and the SR triplet count DROPPED, and the
//!   mutants below delete or mis-weight only the NEW moved / G = 0 terms.
//! * "A mutant passes because it is invisible on this cell": each mutant is
//!   asserted only where FINDINGS measured it visible:
//!   - `SplitNoSmoothOverlap` (drop dS_ss): H3 2.8e-4, tri 4.8e-4; H2 at
//!     Gamma only 4.7e-6 (reported, not asserted); BLIND in the span limit
//!     (q_c ≡ 0, asserted blind there).
//!   - `SplitFullG0`: H2 2.9e-3, H3 1.8e-2; NOT blind in the span.
//!   - `SplitNoSmoothPair`: H2 2.8e-4, H3 5.8e-3; BLIND in the span (X_c ≡ 0).
//!   - stress `SplitNoSmoothOverlap`: DIAGONAL-only on the cubic H2 cell
//!     (off-diagonal 7.3e-9), 1.6e-3 off-diagonal on the triclinic cell —
//!     every mutant is checked on all nine components.
//!   - stress `SplitNoSrKernelStrain` 2.5e-3 (H2), `SplitFullG0` 1.1e-1,
//!     `SplitNoSmoothPair` 2.0e-3.
//! * "Wm symmetry does not matter": FINDINGS (e) — the v_SR-weighted metric
//!   derivative of an unsymmetrised Wm puts 2.2e-11 into Σ_A of the G-space
//!   J2 part on the H2 1x1x3 supercell (symmetrised: 2.2e-16). Asserted on
//!   that part alone (bar 1e-13).
//!
//! h here is the Ewald-split `PeriodicHcore` (as `pbc_grad_rsgdf.rs`), not
//! the prototype's pure-AFT h; the fit derivative composes additively.
//!
//! NOT covered: k-points (stage 2), KS/XC with a split, dropped metric
//! eigenvalues with a split, d orbitals, non-H atoms.

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::grad::{
    gamma_rhf_gradient_rsgdf, gamma_rhf_gradient_with, gamma_uhf_gradient_rsgdf, GammaGradConfig,
    GammaGradient, GradMutation, RsGdfGradSource,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::{RangeSplit, RsGdf, RsGdfConfig, DEFAULT_RSGDF_LINDEP};
use ferric_pbc::stress::{gamma_rhf_stress_rsgdf, GammaStress, GammaStressConfig, StressMutation};
use ferric_pbc::uhf::{gamma_uhf, GammaUhfConfig, GammaUhfIntegrals, GammaUhfResult};
use ferric_scf::result::ScfResult;
use ndarray::Array2;
use std::collections::HashMap;
use std::sync::OnceLock;

type Mat3 = [[f64; 3]; 3];

/// ω of the nuclear-attraction split (as `pbc_grad_rsgdf.rs`).
const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
/// RS-GDF ω (the prototype's w = 1; λ = 1 moves orbital a ≤ 0.5, aux α ≤ 1).
const GDF_OMEGA: f64 = 1.0;
const AMPLE: usize = 1 << 31;
const FD_H: f64 = 1e-4;
/// The unsplit suite's bars (`pbc_grad_rsgdf.rs`, `pbc_stress.rs`).
const FD_BAR: f64 = 1e-7;
const TRI_FD_BAR: f64 = 5e-7;
const STRESS_FD_BAR: f64 = 1e-7;
const TRI_STRESS_BAR: f64 = 5e-6;
/// Smallest asserted mutant miss in FINDINGS is 2.8e-4 (forces) / 2.0e-3
/// (stress); both bars sit ≥ 3 decades above the FD bars.
const MUTANT_BAR: f64 = 1e-4;
const STRESS_MUTANT_BAR: f64 = 1e-4;
const NET_FORCE_BAR: f64 = 1e-10;
const EWALD_NONE_BAR: f64 = 1e-11;
/// Nothing moved: ≤ 1e-13 (FINDINGS measured ≤ 6.3e-14 with Wm symmetrised
/// in BOTH; here both paths symmetrise identically, so bitwise is expected).
const ANCHOR_BAR: f64 = 1e-13;
/// Split vs unsplit force, each at its own converged D (FINDINGS Q5:
/// ≤ 4.4e-12 at Gamma).
const SPLIT_UNSPLIT_BAR: f64 = 1e-10;
/// Exact span, everything moved, vs dense AFT (prototype 3.1e-13; FINDINGS'
/// own bar table says 1e-11).
const SPAN_BAR: f64 = 1e-12;
/// Σ_A of the G-space metric force on the split supercell (FINDINGS (e):
/// 2.2e-16 symmetrised, 2.2e-11 not).
const WM_SYM_BAR: f64 = 1e-13;

/// `run_grad_gdf_anchor.py` H2 (Bohr), a = 4.
const H2_ATOMS_G: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const H2_COMPS: [(usize, usize); 3] = [(0, 0), (0, 2), (1, 1)];
/// H3 (Bohr), a = 4.5, doublet.
const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
const H3_A: f64 = 4.5;
const H3_COMPS: [(usize, usize); 3] = [(0, 0), (2, 1), (1, 2)];
/// `run_grad_oracle.py` TRI_MOVED (Bohr), s+p basis.
const TRI_MOVED: [[f64; 3]; 4] = [
    [0.13, 0.25, 0.31],
    [0.02, 0.27, 1.66],
    [2.47, 2.41, 2.25],
    [3.52, 2.98, 2.71],
];
const TRI_COMPS: [(usize, usize); 2] = [(2, 1), (0, 2)];

const ALL9: [(usize, usize); 9] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
];

const SPLIT_MUTANTS: [GradMutation; 3] = [
    GradMutation::SplitNoSmoothOverlap,
    GradMutation::SplitFullG0,
    GradMutation::SplitNoSmoothPair,
];

const SPLIT_STRESS_MUTANTS: [StressMutation; 4] = [
    StressMutation::SplitNoSmoothOverlap,
    StressMutation::SplitFullG0,
    StressMutation::SplitNoSrKernelStrain,
    StressMutation::SplitNoSmoothPair,
];

/// Hcore config at the file's `OMEGA` with `precision = HCORE_PRECISION`; the
/// other fields are the defaults of `with_omega`.
fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
}

/// RS-GDF config at splitting `omega` with the optional range split
/// `split`, `DEFAULT_RSGDF_LINDEP`, `ExxDiv::None` and budget `AMPLE`.
fn gdf_cfg(omega: f64, split: Option<RangeSplit>) -> RsGdfConfig {
    RsGdfConfig {
        omega,
        lindep: DEFAULT_RSGDF_LINDEP,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        range_split: split,
        ..Default::default()
    }
}

/// `Some(RangeSplit::new(lambda))`.
fn split(lambda: f64) -> Option<RangeSplit> {
    Some(RangeSplit::new(lambda))
}

/// Even-tempered s + Cartesian p aux on H (as `pbc_grad_rsgdf.rs`).
fn et_aux(a0: f64, beta: f64, n: usize) -> BasisSet {
    let mut shells = Vec::new();
    for k in 0..n {
        let a = a0 * beta.powi(k as i32);
        for l in [0, 1] {
            shells.push(Shell {
                l,
                pure: false,
                exponents: vec![a],
                coefficients: vec![1.0],
            });
        }
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

/// H2 ET l≤1 a0 0.3, β 2.5, 5 exponents: 0.3 and 0.75 are smooth at λ = 1.
fn et40() -> BasisSet {
    et_aux(0.3, 2.5, 5)
}

/// The bundled `cc-pvdz-ri` auxiliary basis; panics if it is not bundled.
fn cc_pvdz_ri() -> BasisSet {
    basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri")
}

/// A neutral hydrogen cell with atoms at `pos` (Bohr), lattice rows `lattice`
/// (Bohr) and spin multiplicity `mult`.
fn cell_at(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

/// `pos` (Bohr) with Cartesian component `x` of atom `a` displaced by `h` Bohr.
fn moved_pos(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

struct Setup {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    aux: PreparedBasis,
    gdf: RsGdf,
}

/// Builds the setup: the periodic hcore of `cell` in `bs`, the aux basis
/// `aux_bs` on the same atoms, and the gradient-capable `RsGdf` at
/// `GDF_OMEGA` with the optional range split `rs`.
fn setup(cell: Cell, bs: &BasisSet, aux_bs: &BasisSet, rs: Option<RangeSplit>) -> Setup {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).expect("hcore");
    let aux = PreparedBasis::new(cell.mol(), aux_bs).expect("aux prep");
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &gdf_cfg(GDF_OMEGA, rs))
        .expect("RsGdf::build_for_gradient");
    Setup {
        cell,
        prep,
        hc,
        aux,
        gdf,
    }
}

/// Energy-only split build at a displaced / strained geometry.
fn energy_gdf(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    aux_bs: &BasisSet,
    rs: Option<RangeSplit>,
) -> RsGdf {
    let aux = PreparedBasis::new(cell.mol(), aux_bs).expect("aux prep");
    RsGdf::build(cell, prep, &aux, &hc.s, &gdf_cfg(GDF_OMEGA, rs)).expect("RsGdf")
}

/// The named timing counter of `gdf`; panics, naming the counter, if it is
/// missing.
fn counter(gdf: &RsGdf, name: &str) -> u64 {
    gdf.timings()
        .counter(name)
        .unwrap_or_else(|| panic!("counter {name:?} missing (split path did not run?)"))
}

/// `(orbital smooth, orbital total, aux smooth, aux total)` primitives.
fn moved(gdf: &RsGdf) -> [u64; 4] {
    [
        counter(gdf, "rsgdf split orbital prims smooth"),
        counter(gdf, "rsgdf split orbital prims"),
        counter(gdf, "rsgdf split aux prims smooth"),
        counter(gdf, "rsgdf split aux prims"),
    ]
}

/// Something moved on BOTH sides and the SR walk shrank vs `unsplit`.
fn assert_split_active(tag: &str, gdf: &RsGdf, unsplit: &RsGdf) {
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
}

fn rhf_on(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    gdf: &RsGdf,
    exx: ExxDiv,
) -> ScfResult {
    let g = gdf.clone().with_exxdiv(cell, exx).expect("with_exxdiv");
    gamma_rhf_jk(
        cell,
        prep,
        hc,
        Box::new(g.j_builder()),
        Box::new(g.k_builder()),
    )
}

/// The RS-GDF gradient source of the setup's `gdf` and `aux`, with no aux
/// Jacobian (aux centres on the cell's atoms).
fn src(su: &Setup) -> RsGdfGradSource<'_> {
    RsGdfGradSource {
        gdf: &su.gdf,
        aux: &su.aux,
        aux_jac: None,
    }
}

/// Default Gamma gradient config with budget `AMPLE` and the test-only
/// `mutation`.
fn gcfg(mutation: Option<GradMutation>) -> GammaGradConfig {
    GammaGradConfig {
        mutation,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

/// Default Gamma stress config with the test-only `mutation` and budget
/// `AMPLE`.
fn scfg(mutation: Option<StressMutation>) -> GammaStressConfig {
    GammaStressConfig {
        mutation,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

/// The RS-GDF Gamma RHF gradient of `scf` at exchange divergence `exx` with
/// test-only mutation `m`; panics on error.
fn rhf_grad(su: &Setup, scf: &ScfResult, exx: ExxDiv, m: Option<GradMutation>) -> GammaGradient {
    gamma_rhf_gradient_rsgdf(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &src(su),
        scf,
        exx,
        &gcfg(m),
    )
    .expect("gamma_rhf_gradient_rsgdf")
}

/// The RS-GDF Gamma UHF gradient of `scf` at exchange divergence `exx` with
/// test-only mutation `m`; panics on error.
fn uhf_grad(su: &Setup, scf: &ScfResult, exx: ExxDiv, m: Option<GradMutation>) -> GammaGradient {
    gamma_uhf_gradient_rsgdf(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &src(su),
        scf,
        exx,
        &gcfg(m),
    )
    .expect("gamma_uhf_gradient_rsgdf")
}

/// The RS-GDF Gamma RHF stress of `scf` at exchange divergence `exx` with
/// test-only mutation `m`; panics on error.
fn rhf_stress(su: &Setup, scf: &ScfResult, exx: ExxDiv, m: Option<StressMutation>) -> GammaStress {
    gamma_rhf_stress_rsgdf(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &src(su),
        scf,
        exx,
        &scfg(m),
    )
    .expect("gamma_rhf_stress_rsgdf")
}

type Mos = Option<(Array2<f64>, Array2<f64>)>;

/// The `(α, β)` MO coefficients of `r` as a starting guess; panics without
/// beta MOs.
fn mos_of(r: &ScfResult) -> Mos {
    Some((r.mos_alpha.clone(), r.mos_beta.clone().expect("beta MOs")))
}

/// UHF config with exchange-divergence treatment `exx` and starting MOs
/// `init`; the other fields are the defaults.
fn uhf_cfg(exx: ExxDiv, init: Mos) -> GammaUhfConfig {
    GammaUhfConfig {
        exxdiv: exx,
        initial_mos: init,
        ..GammaUhfConfig::default()
    }
}

/// Gamma UHF on the RS-GDF integrals `gdf`; panics on error or if the SCF
/// does not converge.
fn uhf_on(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    gdf: &RsGdf,
    cfg: &GammaUhfConfig,
) -> GammaUhfResult {
    let r = gamma_uhf(cell, prep, hc, GammaUhfIntegrals::RsGdf(gdf), cfg).expect("gamma_uhf");
    assert!(r.scf.converged);
    r
}

/// Largest element-wise `|a − b|`. `zip` truncates silently if the shapes
/// differ.
fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Whether `a` and `b` have equal shape and equal bit patterns in every
/// element.
fn bitwise(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.dim() == b.dim()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Largest element-wise `|a|` (0 for an empty matrix).
fn amax(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

type Fd = Vec<((usize, usize), Vec<f64>)>;

/// Central FD `(E(+h) − E(−h))/2h` of every energy `energies` returns.
fn fd<F>(
    pos: &[[f64; 3]],
    lattice: [[f64; 3]; 3],
    mult: usize,
    comps: &[(usize, usize)],
    energies: F,
) -> Fd
where
    F: Fn(&Cell) -> Vec<f64>,
{
    comps
        .iter()
        .map(|&(a, x)| {
            let ep = energies(&cell_at(&moved_pos(pos, a, x, FD_H), lattice, mult));
            let em = energies(&cell_at(&moved_pos(pos, a, x, -FD_H), lattice, mult));
            let d = ep
                .iter()
                .zip(&em)
                .map(|(p, m)| (p - m) / (2.0 * FD_H))
                .collect();
            ((a, x), d)
        })
        .collect()
}

/// Largest `|g[a, x] − fd[(a, x)][k]|` over the finite-difference entries:
/// `g` is the analytic `(natoms, 3)` gradient and `k` selects the energy
/// column.
fn max_fd_err(g: &Array2<f64>, fdv: &Fd, k: usize) -> f64 {
    fdv.iter()
        .map(|((a, x), v)| (g[(*a, *x)] - v[k]).abs())
        .fold(0.0_f64, f64::max)
}

fn report(tag: &str, g: &GammaGradient, err: f64) {
    eprintln!(
        "{tag}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, comm {:.1e}\n\
         parts: orb_sr {:.3e} orb_lr {:.3e} aux_sr {:.3e} aux_lr {:.3e} met_sr {:.3e} \
         met_lr {:.3e} g0 {:.3e}\n{:.10}",
        g.net_force,
        g.commutator,
        amax(&g.parts.fit_orb_sr),
        amax(&g.parts.fit_orb_lr),
        amax(&g.parts.fit_aux_sr),
        amax(&g.parts.fit_aux_lr),
        amax(&g.parts.fit_metric_sr),
        amax(&g.parts.fit_metric_lr),
        amax(&g.parts.fit_g0),
        g.grad
    );
}

// ------------------------------------------------------------ stress helpers

/// `cell` strained by the matrix that is zero except `ε[i][j] = h`.
fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    cell.strained(&e).expect("strained cell")
}

/// Central FD `dE/dε_ij`, all nine components, of every energy returned.
fn fd_strain<F>(cell: &Cell, h: f64, energies: F) -> Vec<Mat3>
where
    F: Fn(&Cell) -> Vec<f64>,
{
    let mut out: Vec<Mat3> = Vec::new();
    for (i, j) in ALL9 {
        let ep = energies(&strained(cell, i, j, h));
        let em = energies(&strained(cell, i, j, -h));
        if out.is_empty() {
            out = vec![[[0.0; 3]; 3]; ep.len()];
        }
        for k in 0..ep.len() {
            out[k][i][j] = (ep[k] - em[k]) / (2.0 * h);
        }
    }
    out
}

/// Element-wise `(4·fd_h2 − fd_h)/3`: the Richardson combination that
/// cancels the `O(h²)` term when `fd_h2` is the central difference at half
/// the step of `fd_h`.
fn richardson(fd_h: &Mat3, fd_h2: &Mat3) -> Mat3 {
    std::array::from_fn(|a| std::array::from_fn(|b| (4.0 * fd_h2[a][b] - fd_h[a][b]) / 3.0))
}

/// Largest element-wise `|a − b|` of two 3×3 matrices.
fn max_err(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

/// Largest off-diagonal `|a_ij − b_ij|` of two 3×3 matrices.
fn max_offdiag_err(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                m = m.max((a[i][j] - b[i][j]).abs());
            }
        }
    }
    m
}

/// `max_ij |a_ij − a_ji|` of a 3×3 matrix.
fn antisym(a: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - a[j][i]).abs());
        }
    }
    m
}

/// Whether two 3×3 matrices agree in the bit pattern of every element.
fn mat3_bitwise(a: &Mat3, b: &Mat3) -> bool {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .all(|(x, y)| x.to_bits() == y.to_bits())
}

// ------------------------------------------------- anchor 1: nothing moved

/// The H2 singlet cell: `H2_ATOMS_G` in a cubic lattice of edge 4 Bohr.
fn h2_cell() -> Cell {
    cell_at(&H2_ATOMS_G, cubic(4.0), 1)
}

/// λ = 0 and λ = 1e-8 move nothing: the split force (both exxdiv) and
/// stress equal the unsplit Iteration-18/19 results, bitwise (module doc).
#[test]
fn nothing_moved_split_force_and_stress_are_the_unsplit_ones() {
    let un = setup(h2_cell(), &pyscf_sto3g_h(), &et40(), None);
    for lam in [0.0, 1e-8] {
        let sp = setup(h2_cell(), &pyscf_sto3g_h(), &et40(), split(lam));
        let m = moved(&sp.gdf);
        assert!(
            m[1] > 0 && m[3] > 0,
            "λ = {lam}: split path did not run {m:?}"
        );
        assert_eq!((m[0], m[2]), (0, 0), "λ = {lam} moved something: {m:?}");
        for exx in [ExxDiv::None, ExxDiv::Ewald] {
            let scf_u = rhf_on(&un.cell, &un.prep, &un.hc, &un.gdf, exx);
            let scf_s = rhf_on(&sp.cell, &sp.prep, &sp.hc, &sp.gdf, exx);
            assert_eq!(
                scf_s.energy.to_bits(),
                scf_u.energy.to_bits(),
                "λ = {lam}: E"
            );
            let gu = rhf_grad(&un, &scf_u, exx, None);
            let gs = rhf_grad(&sp, &scf_s, exx, None);
            let d = max_diff(&gs.grad, &gu.grad);
            let fu = gu.fit.as_ref().unwrap();
            let fs = gs.fit.as_ref().unwrap();
            eprintln!(
                "λ = {lam} {exx:?}: |F_split − F_unsplit| = {d:.2e} (bitwise: {}); SR3 deriv {} vs {}",
                bitwise(&gs.grad, &gu.grad),
                fs.n_sr3_deriv,
                fu.n_sr3_deriv
            );
            assert!(d <= ANCHOR_BAR, "λ = {lam} {exx:?}: {d:e}");
            assert_eq!(fs.n_sr3_deriv, fu.n_sr3_deriv);
            assert_eq!(fs.n_sr2_deriv, fu.n_sr2_deriv);
            assert!(
                bitwise(&gs.grad, &gu.grad),
                "λ = {lam} {exx:?}: not bitwise ({d:.2e}); the code paths should coincide"
            );
        }
        let scf_u = rhf_on(&un.cell, &un.prep, &un.hc, &un.gdf, ExxDiv::None);
        let scf_s = rhf_on(&sp.cell, &sp.prep, &sp.hc, &sp.gdf, ExxDiv::None);
        let su = rhf_stress(&un, &scf_u, ExxDiv::None, None);
        let ss = rhf_stress(&sp, &scf_s, ExxDiv::None, None);
        let d = max_err(&ss.de_deps, &su.de_deps);
        eprintln!(
            "λ = {lam}: |dE/dε split − unsplit| = {d:.2e} (bitwise: {})",
            mat3_bitwise(&ss.de_deps, &su.de_deps)
        );
        assert!(d <= ANCHOR_BAR, "λ = {lam} stress: {d:e}");
        assert!(
            mat3_bitwise(&ss.de_deps, &su.de_deps),
            "λ = {lam} stress: not bitwise ({d:.2e})"
        );
    }
}

// ------------------------------------------------------------ H2 RHF, λ = 1

static H2_FD: OnceLock<Fd> = OnceLock::new();

/// FD `[none, ewald]` of the H2 ET-40 split RS-GDF RHF energy.
fn h2_fd() -> &'static Fd {
    H2_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let aux_bs = et40();
        fd(&H2_ATOMS_G, cubic(4.0), 1, &H2_COMPS, |c| {
            let prep = prep_for(c, &bs);
            let hc = periodic_hcore(c, &prep, &hcore_cfg()).unwrap();
            let gdf = energy_gdf(c, &prep, &hc, &aux_bs, split(1.0));
            [ExxDiv::None, ExxDiv::Ewald]
                .iter()
                .map(|&e| rhf_on(c, &prep, &hc, &gdf, e).energy)
                .collect()
        })
    })
}

#[test]
fn h2_split_force_matches_fd_and_the_unsplit_force() {
    let sp = setup(h2_cell(), &pyscf_sto3g_h(), &et40(), split(1.0));
    let un = setup(h2_cell(), &pyscf_sto3g_h(), &et40(), None);
    assert_split_active("H2 ET-40 λ=1", &sp.gdf, &un.gdf);
    let fdv = h2_fd();
    for (k, exx) in [ExxDiv::None, ExxDiv::Ewald].into_iter().enumerate() {
        let scf = rhf_on(&sp.cell, &sp.prep, &sp.hc, &sp.gdf, exx);
        let g = rhf_grad(&sp, &scf, exx, None);
        let err = max_fd_err(&g.grad, fdv, k);
        report(&format!("H2 ET-40 split RHF {exx:?}"), &g, err);
        // Prototype: −2.43e-9 vs the unsplit −2.42e-9.
        assert!(err < FD_BAR, "{exx:?}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        // Each at its own D (prototype Q5: 1.7e-12 / 4.4e-12).
        let scf_u = rhf_on(&un.cell, &un.prep, &un.hc, &un.gdf, exx);
        let gu = rhf_grad(&un, &scf_u, exx, None);
        let d = max_diff(&g.grad, &gu.grad);
        eprintln!(
            "  E_split − E_unsplit = {:.2e}; |F_split − F_unsplit| = {d:.2e}",
            scf.energy - scf_u.energy
        );
        assert!(d < SPLIT_UNSPLIT_BAR, "{exx:?}: split vs unsplit {d:e}");
    }
    let scf = rhf_on(&sp.cell, &sp.prep, &sp.hc, &sp.gdf, ExxDiv::None);
    for m in SPLIT_MUTANTS {
        let e = max_fd_err(&rhf_grad(&sp, &scf, ExxDiv::None, Some(m)).grad, fdv, 0);
        eprintln!("  H2 mutant {m:?}: max|analytic − FD| = {e:.2e}");
        // SplitNoSmoothOverlap is weak on H2 at Gamma (prototype 4.7e-6:
        // q_c is small, the smooth ET-40 aux carry most of the charge).
        if m != GradMutation::SplitNoSmoothOverlap {
            assert!(e > MUTANT_BAR, "{m:?} escaped the H2 FD anchor: {e:e}");
        }
    }
}

// ------------------------------------------------------------ H3 UHF, λ = 1

/// The H3 doublet setup: `H3_ATOMS` in a cubic cell of edge `H3_A` Bohr,
/// PySCF STO-3G with `cc-pvdz-ri`, and the optional range split `rs`.
fn h3_setup(rs: Option<RangeSplit>) -> Setup {
    setup(
        cell_at(&H3_ATOMS, cubic(H3_A), 2),
        &pyscf_sto3g_h(),
        &cc_pvdz_ri(),
        rs,
    )
}

/// `(none stage, ewald)` of the staged H3 doublet UHF.
fn h3_uhf_ref(su: &Setup) -> (ScfResult, ScfResult) {
    let r = uhf_on(
        &su.cell,
        &su.prep,
        &su.hc,
        &su.gdf,
        &uhf_cfg(ExxDiv::Ewald, None),
    );
    (r.none_stage.clone().expect("none stage"), r.scf)
}

static H3_FD: OnceLock<Fd> = OnceLock::new();

fn h3_fd() -> &'static Fd {
    H3_FD.get_or_init(|| {
        let bs = pyscf_sto3g_h();
        let aux_bs = cc_pvdz_ri();
        let (none, _) = h3_uhf_ref(&h3_setup(split(1.0)));
        let init = mos_of(&none);
        fd(&H3_ATOMS, cubic(H3_A), 2, &H3_COMPS, |c| {
            let prep = prep_for(c, &bs);
            let hc = periodic_hcore(c, &prep, &hcore_cfg()).unwrap();
            let gdf = energy_gdf(c, &prep, &hc, &aux_bs, split(1.0));
            let r = uhf_on(c, &prep, &hc, &gdf, &uhf_cfg(ExxDiv::Ewald, init.clone()));
            vec![r.none_stage.expect("none stage").energy, r.scf.energy]
        })
    })
}

#[test]
fn h3_uhf_split_force_matches_fd_and_catches_the_split_mutants() {
    let sp = h3_setup(split(1.0));
    assert_split_active("H3 cc-pvdz-ri λ=1", &sp.gdf, &h3_setup(None).gdf);
    let (none, ewald) = h3_uhf_ref(&sp);
    let fdv = h3_fd();
    for (k, (scf, exx)) in [(&none, ExxDiv::None), (&ewald, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let g = uhf_grad(&sp, scf, exx, None);
        let err = max_fd_err(&g.grad, fdv, k);
        report(&format!("H3 split UHF doublet {exx:?}"), &g, err);
        // Prototype: +2.74e-9 / −1.61e-9 (unsplit 2.7e-9 / 1.6e-9).
        assert!(err < FD_BAR, "{exx:?}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        for m in SPLIT_MUTANTS {
            let gm = uhf_grad(&sp, scf, exx, Some(m));
            let e = max_fd_err(&gm.grad, fdv, k);
            eprintln!(
                "  mutant {m:?} ({exx:?}): {e:.2e}, |ΣF| {:.1e}",
                gm.net_force
            );
            // Prototype: 2.8e-4 / 1.8e-2 / 5.8e-3; ΣF is blind to all three.
            assert!(
                e > MUTANT_BAR,
                "{m:?} ({exx:?}) escaped the FD anchor: {e:e}"
            );
        }
    }
}

/// F(ewald) ≡ F(none) on ONE open-shell density (split fit).
#[test]
fn h3_uhf_split_ewald_and_none_forces_agree_on_one_density() {
    let sp = h3_setup(split(1.0));
    let (_, ewald) = h3_uhf_ref(&sp);
    let gn = uhf_grad(&sp, &ewald, ExxDiv::None, None);
    let ge = uhf_grad(&sp, &ewald, ExxDiv::Ewald, None);
    let d = max_diff(&gn.grad, &ge.grad);
    eprintln!(
        "H3 UHF split: |F_ewald − F_none| = {d:.2e} (v_M = {})",
        ge.madelung
    );
    assert!(ge.madelung > 0.1);
    assert!(d < EWALD_NONE_BAR, "{d:e}");
}

// ------------------------------------------- anchor 3: exact span, all moved

/// The 24 exact-span aux sites (all 8 half-lattice classes of the 3 pair
/// types, α = 1) and their Jacobian `dC_k/dR_A` (as `pbc_grad_rsgdf.rs`).
fn span_sites(cell: &Cell) -> (Vec<[f64; 4]>, Array2<f64>) {
    let r = cell.positions();
    let a = cell.lattice();
    let mut out = Vec::new();
    let mut jac = Array2::<f64>::zeros((24, 2));
    let mut row = 0;
    for (i, j) in [(0usize, 0usize), (1, 1), (0, 1)] {
        for k in 0..8usize {
            let h = [(k >> 2) & 1, (k >> 1) & 1, k & 1];
            let mut c = [0.0; 3];
            for d in 0..3 {
                let ha: f64 = (0..3).map(|x| h[x] as f64 * a[x][d]).sum();
                c[d] = 0.5 * (r[i][d] + r[j][d] + ha);
            }
            out.push([c[0], c[1], c[2], 1.0]);
            jac[(row, i)] += 0.5;
            jac[(row, j)] += 0.5;
            row += 1;
        }
    }
    (out, jac)
}

/// ω = 1.2, λ = 1: orbital 0.5 ≤ 0.72 and aux 1 ≤ 1.44 — EVERYTHING moved
/// (no SR call left): the split force must equal the dense pure-AFT force.
/// `SplitFullG0` must fail it; `SplitNoSmoothOverlap` / `SplitNoSmoothPair`
/// are blind here BY CONSTRUCTION (every aux primitive smooth ⇒ X_c ≡ 0,
/// q_c ≡ 0), which is why the H3 / tri FD anchors carry them.
#[test]
fn exact_span_everything_moved_split_force_equals_dense_aft() {
    let cell = h2_cell();
    let prep = prep_for(&cell, &single_s_h(0.5));
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg()).unwrap();
    let (sites, jac) = span_sites(&cell);
    let site = SiteBasis::new(&sites, 0).unwrap();
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &site.prep, &hc.s, &gdf_cfg(1.2, split(1.0)))
        .unwrap();
    let m = moved(&gdf);
    let st = gdf.stats();
    eprintln!(
        "span ω=1.2: moved {m:?}; SR3 {} SR2 {}",
        st.n_sr3_triplets, st.n_sr2_pairs
    );
    assert_eq!((m[0], m[2]), (m[1], m[3]), "not everything moved: {m:?}");
    assert_eq!(st.n_sr3_triplets, 0);
    assert_eq!(st.n_sr2_pairs, 0);
    let dense = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let fit = RsGdfGradSource {
        gdf: &gdf,
        aux: &site.prep,
        aux_jac: Some(&jac),
    };
    for exx in [ExxDiv::None, ExxDiv::Ewald] {
        let eri = dense.clone().with_exxdiv(&cell, exx).unwrap();
        let scf_d = gamma_rhf(&cell, &prep, &hc, &eri);
        let scf_f = rhf_on(&cell, &prep, &hc, &gdf, exx);
        let gd = gamma_rhf_gradient_with(
            &cell,
            &prep,
            &hcore_cfg(),
            &hc,
            &eri,
            &scf_d,
            exx,
            &gcfg(None),
        )
        .unwrap();
        let run = |mm: Option<GradMutation>| {
            gamma_rhf_gradient_rsgdf(
                &cell,
                &prep,
                &hcore_cfg(),
                &hc,
                &fit,
                &scf_f,
                exx,
                &gcfg(mm),
            )
            .unwrap()
        };
        let gf = run(None);
        let d = max_diff(&gf.grad, &gd.grad);
        let fdiag = gf.fit.as_ref().unwrap();
        eprintln!(
            "span split {exx:?}: E − E_dense = {:.2e}, max|F − F_dense| = {d:.2e} (|F| {:.3}), \
             SR3 deriv {} SR2 deriv {}, ΣF {:.1e}",
            scf_f.energy - scf_d.energy,
            amax(&gd.grad),
            fdiag.n_sr3_deriv,
            fdiag.n_sr2_deriv,
            gf.net_force
        );
        assert!(amax(&gd.grad) > 0.05, "vacuous comparison");
        assert_eq!((fdiag.n_sr3_deriv, fdiag.n_sr2_deriv), (0, 0));
        assert!(d < SPAN_BAR, "{exx:?}: {d:e}");
        for mm in SPLIT_MUTANTS {
            let dm = max_diff(&run(Some(mm)).grad, &gd.grad);
            eprintln!("  mutant {mm:?}: max|F − F_dense| = {dm:.2e}");
            if mm == GradMutation::SplitFullG0 {
                // Prototype 7.1e-3.
                assert!(dm > 1e-3, "{mm:?} escaped the span anchor: {dm:e}");
            } else {
                assert!(
                    dm < 1e-10,
                    "{mm:?} should be blind in the span limit: {dm:e}"
                );
            }
        }
    }
}

// ------------------------------------------------ Wm symmetry (supercell)

/// FINDINGS "Iteration 26" (e): on the H2 1x1x3 supercell the G-space
/// metric force of the split fit sums to Re[iG Xᴴ(Wm − Wmᵀ)X] over atoms,
/// zero only for a symmetric Wm (unsymmetrised: 2.2e-11; symmetrised
/// 2.2e-16). Asserted on that part alone so the other terms' own
/// translation noise cannot hide it.
#[test]
fn split_supercell_metric_force_is_translation_invariant() {
    let pos: Vec<[f64; 3]> = (0..3)
        .flat_map(|k| {
            H2_ATOMS_G
                .iter()
                .map(move |r| [r[0], r[1], r[2] + 4.0 * k as f64])
        })
        .collect();
    let lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 12.0]];
    let sp = setup(
        cell_at(&pos, lattice, 1),
        &pyscf_sto3g_h(),
        &et40(),
        split(1.0),
    );
    let m = moved(&sp.gdf);
    assert!(m[0] > 0 && m[2] > 0, "nothing moved {m:?}");
    let scf = rhf_on(&sp.cell, &sp.prep, &sp.hc, &sp.gdf, ExxDiv::None);
    let g = rhf_grad(&sp, &scf, ExxDiv::None, None);
    let sum_part = |a: &Array2<f64>| {
        (0..3)
            .map(|c| a.column(c).sum().abs())
            .fold(0.0_f64, f64::max)
    };
    let s_met = sum_part(&g.parts.fit_metric_lr);
    // Copies of one H2 must see the same force (the supercell is H2 x3).
    let mut spread = 0.0_f64;
    for k in 1..3 {
        for a in 0..2 {
            for c in 0..3 {
                spread = spread.max((g.grad[(2 * k + a, c)] - g.grad[(a, c)]).abs());
            }
        }
    }
    eprintln!(
        "H2 x3 split supercell: |Σ_A F_metric_lr| = {s_met:.2e} (|part| {:.2e}), |ΣF| {:.1e}, \
         copy spread {spread:.2e}",
        amax(&g.parts.fit_metric_lr),
        g.net_force
    );
    assert!(amax(&g.parts.fit_metric_lr) > 1e-4, "metric part vacuous");
    assert!(s_met < WM_SYM_BAR, "Σ_A metric LR force {s_met:e}");
    assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
}

// ---------------------------------------------------- H2 stress (slow)

#[test]
#[ignore = "slow: H2 ET-40 split RS-GDF, 18 strained split builds + 36 SCFs; run in release \
            with --ignored"]
fn h2_split_stress_matches_fd_all_nine_and_catches_the_split_mutants() {
    let bs = pyscf_sto3g_h();
    let aux_bs = et40();
    let sp = setup(h2_cell(), &bs, &aux_bs, split(1.0));
    let fdv = fd_strain(&sp.cell, FD_H, |c| {
        let p = prep_for(c, &bs);
        let h = periodic_hcore(c, &p, &hcore_cfg()).unwrap();
        let g = energy_gdf(c, &p, &h, &aux_bs, split(1.0));
        [ExxDiv::None, ExxDiv::Ewald]
            .iter()
            .map(|&e| rhf_on(c, &p, &h, &g, e).energy)
            .collect()
    });
    let scfs = [
        rhf_on(&sp.cell, &sp.prep, &sp.hc, &sp.gdf, ExxDiv::None),
        rhf_on(&sp.cell, &sp.prep, &sp.hc, &sp.gdf, ExxDiv::Ewald),
    ];
    let un = setup(h2_cell(), &bs, &aux_bs, None);
    for (k, exx) in [ExxDiv::None, ExxDiv::Ewald].into_iter().enumerate() {
        let st = rhf_stress(&sp, &scfs[k], exx, None);
        let err = max_err(&st.de_deps, &fdv[k]);
        let scf_u = rhf_on(&un.cell, &un.prep, &un.hc, &un.gdf, exx);
        let su = rhf_stress(&un, &scf_u, exx, None);
        eprintln!(
            "H2 split stress {exx:?}: max|an − FD| = {err:.2e}, |an − anᵀ| {:.1e}, \
             |split − unsplit| {:.2e}\n{:?}",
            antisym(&st.de_deps),
            max_err(&st.de_deps, &su.de_deps),
            st.de_deps
        );
        // Prototype 6.73e-9 (unsplit 6.7e-9).
        assert!(err < STRESS_FD_BAR, "{exx:?}: {err:e}");
        assert!(antisym(&st.de_deps) < 1e-10);
    }
    for m in SPLIT_STRESS_MUTANTS {
        let st = rhf_stress(&sp, &scfs[0], ExxDiv::None, Some(m));
        let err = max_err(&st.de_deps, &fdv[0]);
        let off = max_offdiag_err(&st.de_deps, &fdv[0]);
        eprintln!("H2 split stress mutant {m:?}: max|an − FD| {err:.2e} (off-diagonal {off:.2e})");
        assert!(err > STRESS_MUTANT_BAR, "{m:?} escaped: {err:e}");
    }
}

// --------------------------------------------------- triclinic s+p (slow)

/// The triclinic setup: `TRI_MOVED` in the lattice `TRI_A`, singlet, the
/// s+p hydrogen basis with `cc-pvdz-ri`, and the optional range split `rs`.
fn tri_setup(rs: Option<RangeSplit>) -> Setup {
    setup(
        cell_at(&TRI_MOVED, TRI_A, 1),
        &sp_basis_h(),
        &cc_pvdz_ri(),
        rs,
    )
}

#[test]
#[ignore = "slow: triclinic 4H s+p, cc-pvdz-ri, 4 displaced split RS-GDF builds + SCFs and the \
            split mutants; run with --release -- --ignored"]
fn triclinic_split_forces_match_fd_and_catch_the_split_mutants() {
    let bs = sp_basis_h();
    let aux_bs = cc_pvdz_ri();
    let sp = tri_setup(split(1.0));
    assert_split_active("tri s+p cc-pvdz-ri λ=1", &sp.gdf, &tri_setup(None).gdf);
    let fdv = fd(&TRI_MOVED, TRI_A, 1, &TRI_COMPS, |c| {
        let prep = prep_for(c, &bs);
        let hc = periodic_hcore(c, &prep, &hcore_cfg()).unwrap();
        let gdf = energy_gdf(c, &prep, &hc, &aux_bs, split(1.0));
        vec![rhf_on(c, &prep, &hc, &gdf, ExxDiv::None).energy]
    });
    let scf = rhf_on(&sp.cell, &sp.prep, &sp.hc, &sp.gdf, ExxDiv::None);
    let g = rhf_grad(&sp, &scf, ExxDiv::None, None);
    let err = max_fd_err(&g.grad, &fdv, 0);
    report("tri s+p split RHF", &g, err);
    // Prototype +4.8e-10 / −2.72e-9; p-shell libint floor → the unsplit bar.
    assert!(err < TRI_FD_BAR, "{err:e}");
    assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
    for m in SPLIT_MUTANTS {
        let e = max_fd_err(&rhf_grad(&sp, &scf, ExxDiv::None, Some(m)).grad, &fdv, 0);
        eprintln!("  tri mutant {m:?}: {e:.2e}");
        // Prototype: 4.8e-4 / 8.8e-4 / 3.4e-3.
        assert!(e > MUTANT_BAR, "{m:?} escaped the tri FD anchor: {e:e}");
    }
}

#[test]
#[ignore = "slow: triclinic 4H s+p split RS-GDF, 36 strained split builds + SCFs (FD h and h/2); \
            run with --release -- --ignored"]
fn triclinic_split_stress_matches_fd_with_richardson_and_sees_the_offdiagonal_virial() {
    let bs = sp_basis_h();
    let aux_bs = cc_pvdz_ri();
    let sp = tri_setup(split(1.0));
    let energies = |c: &Cell| -> Vec<f64> {
        let p = prep_for(c, &bs);
        let h = periodic_hcore(c, &p, &hcore_cfg()).unwrap();
        let g = energy_gdf(c, &p, &h, &aux_bs, split(1.0));
        vec![rhf_on(c, &p, &h, &g, ExxDiv::None).energy]
    };
    let fd_h = fd_strain(&sp.cell, FD_H, &energies);
    let fd_h2 = fd_strain(&sp.cell, 0.5 * FD_H, &energies);
    let rich = richardson(&fd_h[0], &fd_h2[0]);
    let scf = rhf_on(&sp.cell, &sp.prep, &sp.hc, &sp.gdf, ExxDiv::None);
    let st = rhf_stress(&sp, &scf, ExxDiv::None, None);
    let (err_h, err_r) = (max_err(&st.de_deps, &fd_h[0]), max_err(&st.de_deps, &rich));
    eprintln!(
        "tri split stress: FD(h) {err_h:.2e}, Richardson {err_r:.2e}, |an − anᵀ| {:.1e}\n{:?}",
        antisym(&st.de_deps),
        st.de_deps
    );
    assert!(err_r < TRI_STRESS_BAR, "Richardson {err_r:e}");
    assert!(err_h < TRI_STRESS_BAR, "FD(h) {err_h:e}");
    for m in SPLIT_STRESS_MUTANTS {
        let sm = rhf_stress(&sp, &scf, ExxDiv::None, Some(m));
        let err = max_err(&sm.de_deps, &rich);
        let off = max_offdiag_err(&sm.de_deps, &rich);
        eprintln!("  tri stress mutant {m:?}: {err:.2e} (off-diagonal {off:.2e})");
        assert!(err > STRESS_MUTANT_BAR, "{m:?} escaped: {err:e}");
        if m == StressMutation::SplitNoSmoothOverlap {
            // Diagonal-only on the cubic H2 cell; off-diagonal 1.6e-3 here.
            assert!(off > STRESS_MUTANT_BAR, "{m:?} off-diagonal {off:e}");
        }
    }
}
