//! Linear dependence in the periodic SCF (FINDINGS "Iteration 15 (Python,
//! linear dependence)", `reference/pbc/pbc_lindep.py`, the end of
//! `test_prototype.py`): the per-k canonical cut on RAW `S(k)` eigenvalues,
//! its `LindepReport`, the noise-floor flag, and the opt-in `exp_to_discard`.
//!
//! Model (the prototype's `lindep_model`): H2 cubic a = 4, PySCF STO-3G on
//! each H plus ONE diffuse s per H with DIFFERENT exponents (0.06 on H1,
//! 0.069 on H2 — the asymmetry breaks inversion symmetry, so dropping the
//! Γ near-null vector is not free by symmetry). ferric keys bases by Z, so
//! each diffuse s sits on a GHOST atom (Z = 2 / Z = 3, no charge, no
//! electrons) at its hydrogen's position: the same functions. 1×1×3
//! Gamma-centred mesh, exxdiv none, dense pure-AFT J/K.
//! Prototype numbers: λ_min S(Γ) 2.2e-7, S(±k) 1.6e-3; τ = 1e-6 keeps
//! [3, 4, 4] on the mesh and 11 on the supercell; k-mesh − supercell
//! −3.4e-14 Ha/cell; dropping the Γ vector costs +2.28e-6 (loose ERIs) ..
//! +2.55e-6 (tight) Ha/cell.
//!
//! ARTIFACT HYPOTHESIS (stated before measuring, per the repo protocol): a
//! cut that is not k-consistent (normalised S, pivoted Cholesky, a per-k
//! count forced common) breaks k-mesh ≡ supercell by ≥ 1e-9; a truncated
//! 1e lattice sum fakes λ_min (the prototype's rcut 22 artifact read 8.7e-6)
//! — the premise asserts pin λ_min to the Poisson-consistent range.
//!
//! Cost: UNMEASURED in Rust. The diffuse pairs need ~10³ lattice images in
//! the pair FT (both sides) and the supercell SR nuclear sum is 12 AOs; if
//! the SCF tests exceed ~60 s in release mark them
//! `#[ignore = "slow: ..."]`.

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::{Atom, Molecule};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{
    complex_canonical_orthogonalizer_with_stats, solve_krhf_injected, KPointInjection, KScfConfig,
    KScfResult,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::lindep::{exp_to_discard, prepare_cell_basis, ExpToDiscardError, LindepReport};
use num_complex::Complex64;
use std::collections::HashMap;
use std::sync::OnceLock;

const A: f64 = 4.0;
const AD: f64 = 0.06;
const ASYM: f64 = 1.15;
const OMEGA: f64 = 0.8;
/// Dense-AFT K sphere / pair screen (`0.1 · precision` = 1e-9, the
/// prototype's pair threshold; FINDINGS precondition 2: integrals precise
/// well below τ). The anchor is exact at any precision.
const ERI_PRECISION: f64 = 1e-8;
const MESH: [usize; 3] = [1, 1, 3];
/// The recommended default cut.
const TAU: f64 = 1e-6;

fn atom(symbol: &str, z: i32, r: [f64; 3], ghost: bool) -> Atom {
    Atom {
        symbol: symbol.into(),
        z,
        x: r[0],
        y: r[1],
        zpos: r[2],
        ghost,
        n_core_ecp: 0,
    }
}

/// H1, H2 of `H2_ATOMS` shifted by `t`, plus the ghost carriers of their
/// diffuse s (Z = 2 on H1, Z = 3 on H2).
fn cell_atoms(t: [f64; 3]) -> Vec<Atom> {
    let p = |r: [f64; 3]| [r[0] + t[0], r[1] + t[1], r[2] + t[2]];
    vec![
        atom("H", 1, p(H2_ATOMS[0]), false),
        atom("H", 1, p(H2_ATOMS[1]), false),
        atom("He", 2, p(H2_ATOMS[0]), true),
        atom("Li", 3, p(H2_ATOMS[1]), true),
    ]
}

fn molecule(atoms: Vec<Atom>) -> Molecule {
    Molecule {
        atoms,
        charge: 0,
        multiplicity: 1,
    }
}

fn model_cell() -> Cell {
    Cell::new(molecule(cell_atoms([0.0; 3])), cubic(A)).expect("model cell")
}

/// Explicit diag(1,1,3) supercell of [`model_cell`].
fn model_supercell() -> Cell {
    let mut atoms = Vec::new();
    for m in 0..MESH[2] {
        atoms.extend(cell_atoms([0.0, 0.0, m as f64 * A]));
    }
    let mut lat = cubic(A);
    lat[2][2] *= MESH[2] as f64;
    Cell::new(molecule(atoms), lat).expect("model supercell")
}

fn s_shell(alpha: f64) -> Shell {
    single_s_h(alpha).shells[&1][0].clone()
}

fn model_basis() -> BasisSet {
    let mut shells = HashMap::new();
    shells.insert(1, pyscf_sto3g_h().shells[&1].clone());
    shells.insert(2, vec![s_shell(AD)]);
    shells.insert(3, vec![s_shell(AD * ASYM)]);
    BasisSet {
        name: "lindep-model".into(),
        shells,
        ecps: HashMap::new(),
    }
}

struct Ints {
    cell: Cell,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    eri: KDenseAftEri,
}

fn build_ints(cell: Cell, n: [usize; 3]) -> Ints {
    let prep = prep_for(&cell, &model_basis());
    let mesh = KPointMesh::gamma_centred(&cell, n).expect("mesh");
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA))
        .expect("hcore(k)");
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::None,
        &KDenseAftConfig {
            precision: ERI_PRECISION,
            ..Default::default()
        },
    )
    .expect("dense k-point kernels");
    Ints {
        cell,
        mesh,
        hk,
        eri,
    }
}

fn kmesh() -> &'static Ints {
    static K: OnceLock<Ints> = OnceLock::new();
    K.get_or_init(|| build_ints(model_cell(), MESH))
}

fn supercell() -> &'static Ints {
    static S: OnceLock<Ints> = OnceLock::new();
    S.get_or_init(|| build_ints(model_supercell(), [1, 1, 1]))
}

fn scf(ints: &Ints, lindep: f64) -> KScfResult {
    let cfg = KScfConfig {
        max_iter: 300,
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        lindep,
        ..Default::default()
    };
    let inj = KPointInjection {
        s: ints.hk.s.clone(),
        h: ints.hk.h.clone(),
        vnn: ints.hk.enn,
        jk: Box::new(ints.eri.jk_builder_with_madelung(0.0)),
    };
    let r = solve_krhf_injected(&ints.cell, &ints.mesh, &cfg, inj).expect("k-point RHF");
    assert!(
        r.converged,
        "SCF (lindep {lindep:e}) not converged in {} it (max error {:e})",
        r.iterations, r.max_error
    );
    r
}

fn kept(r: &LindepReport) -> Vec<usize> {
    r.per_k.iter().map(|k| k.kept).collect()
}

// ------------------------------------------------------------ overlap only

/// eig S_sc(Γ) = ∪_k eig S(k): the report's Σ_k kept equals the supercell's
/// kept count at EVERY τ although per-k counts differ; the smallest and
/// largest-dropped eigenvalues agree too. The real-Γ `from_real_overlap`
/// (the `solve_rhf_injected` path's S) is an independent construction of
/// the supercell side.
#[test]
#[ignore = "slow: diffuse-shell supercell SCF (~100 s release); run with --release -- --ignored, serially"]
fn report_total_kept_equals_the_supercell_kept_count_at_every_threshold() {
    let k = kmesh();
    let sc = supercell();
    let sc_prep = prep_for(&sc.cell, &model_basis());
    let hc = periodic_hcore(&sc.cell, &sc_prep, &PeriodicHcoreConfig::with_omega(OMEGA))
        .expect("supercell Gamma hcore");
    let lam_min = |r: &LindepReport| {
        r.per_k
            .iter()
            .map(|p| p.min_eig)
            .fold(f64::INFINITY, f64::min)
    };
    for tau in [1e-9, 1e-6, 1e-4, 1e-3] {
        let rk = LindepReport::from_overlaps(&k.hk.s, tau).unwrap();
        let rs = LindepReport::from_overlaps(&sc.hk.s, tau).unwrap();
        let rg = LindepReport::from_real_overlap(&hc.s, tau).unwrap();
        assert_eq!(rk.total_kept, rs.total_kept, "tau {tau:e}: {:?}", kept(&rk));
        assert_eq!(
            rs.total_kept, rg.total_kept,
            "tau {tau:e}: complex vs real Gamma"
        );
        assert!((lam_min(&rk) - lam_min(&rs)).abs() < 1e-12, "tau {tau:e}");
        let md = |r: &LindepReport| {
            r.per_k
                .iter()
                .filter_map(|p| p.max_dropped)
                .fold(f64::NEG_INFINITY, f64::max)
        };
        if rk.total_dropped() > 0 {
            assert!((md(&rk) - md(&rs)).abs() < 1e-12, "tau {tau:e}");
        }
    }
    let r6 = LindepReport::from_overlaps(&k.hk.s, TAU).unwrap();
    // Premise (Poisson-consistent S; a truncated lattice sum fakes λ_min):
    // Γ holds the one near-null vector, ±k are O(1e-3).
    let g = &r6.per_k[0];
    assert!(
        g.min_eig > 1e-8 && g.min_eig < 1e-6,
        "lam_min(Gamma) {:e}",
        g.min_eig
    );
    assert!(
        r6.per_k[1].min_eig > 1e-4,
        "lam_min(k1) {:e}",
        r6.per_k[1].min_eig
    );
    assert_eq!(kept(&r6), vec![3, 4, 4]);
    assert_eq!((r6.min_kept, r6.max_kept, r6.total_kept), (3, 4, 11));
    assert_eq!(r6.total_dropped(), 1);
}

/// The flag fires for a threshold at the roundoff floor of S(k) and not at
/// the default. Estimate here: Σ ≈ max row ℓ1 of S(Γ) ~ 40 (diffuse
/// diagonal (1/Ω)(2π/α)^{3/2} ≈ 16.7) → floor ≈ 9e-12, flag below ~9e-10.
#[test]
fn noise_floor_flag_fires_on_a_pathological_threshold_only() {
    let k = kmesh();
    let bad = LindepReport::from_overlaps(&k.hk.s, 1e-13).unwrap();
    assert!(bad.near_noise_floor, "floor {:e}", bad.noise_floor);
    let ok = LindepReport::from_overlaps(&k.hk.s, TAU).unwrap();
    assert!(!ok.near_noise_floor, "floor {:e}", ok.noise_floor);
    assert_eq!(bad.noise_floor, ok.noise_floor);
    assert!(
        ok.overlap_abs_sum > 10.0 && ok.overlap_abs_sum < 200.0,
        "sum|S| estimate {}",
        ok.overlap_abs_sum
    );
}

/// τ below every eigenvalue: nothing dropped and X X^H = S(k)^{-1}
/// (S X X^H = 1; tolerance ε·κ with κ(S(Γ)) ~ 1e2/2e-7).
#[test]
fn orthogonalizer_below_every_eigenvalue_is_the_full_inverse() {
    let k = kmesh();
    for s in &k.hk.s {
        let (x, st) = complex_canonical_orthogonalizer_with_stats(s, 1e-12).unwrap();
        assert_eq!((st.kept, st.max_dropped), (s.nrows(), None));
        let xxh = x.dot(&x.t().mapv(|z| z.conj()));
        let id = s.dot(&xxh);
        let n = s.nrows();
        let err = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| {
                let e = if i == j { 1.0 } else { 0.0 };
                (id[(i, j)] - Complex64::new(e, 0.0)).norm()
            })
            .fold(0.0_f64, f64::max);
        assert!(err < 1e-7, "S X X^H - 1 = {err:e}");
    }
}

// ------------------------------------------------------------------ SCF

/// Trivial limit: two thresholds below every eigenvalue of every S(k) give
/// the SAME kept space, hence bitwise the same SCF (≤ 1e-13 asserted); the
/// report says nothing was dropped, and only the pathological one is
/// flagged. Negative control: τ = 1e-6 drops the Γ vector and costs
/// +1e-7..1e-5 Ha/cell (variational, prototype +2.3e-6..2.6e-6) — the
/// threshold is live.
#[test]
fn threshold_below_every_eigenvalue_is_no_cut() {
    let k = kmesh();
    let a = scf(k, 1e-8);
    let b = scf(k, 1e-13);
    for r in [&a, &b] {
        assert_eq!(kept(&r.lindep), vec![4, 4, 4]);
        assert_eq!(r.lindep.total_dropped(), 0);
        assert!(r.lindep.per_k.iter().all(|p| p.max_dropped.is_none()));
    }
    assert!(
        a.lindep.per_k[0].min_eig > 1e-8,
        "premise: {:e}",
        a.lindep.per_k[0].min_eig
    );
    assert!(!a.lindep.near_noise_floor);
    assert!(
        b.lindep.near_noise_floor,
        "floor {:e}",
        b.lindep.noise_floor
    );
    assert!(
        (a.energy - b.energy).abs() <= 1e-13,
        "{:e}",
        a.energy - b.energy
    );
    let c = scf(k, TAU);
    let d = c.energy - a.energy;
    assert!(
        d > 1e-7 && d < 1e-5,
        "cost of dropping the Gamma vector {d:e}"
    );
}

/// k-mesh ≡ supercell with a k-DEPENDENT kept count ([3, 4, 4] vs 11 at
/// τ = 1e-6): the absolute per-k cut keeps the supercell's space, so
/// E/cell agrees to ≤ 1e-12 (prototype −3.4e-14). The SCF's own report
/// equals the overlap-only diagnosis.
#[test]
#[ignore = "slow: diffuse-shell supercell SCF (~100 s release); run with --release -- --ignored, serially"]
fn kmesh_equals_supercell_with_k_dependent_kept_counts() {
    let k = kmesh();
    let sc = supercell();
    let rk = scf(k, TAU);
    let rs = scf(sc, TAU);
    assert_eq!(kept(&rk.lindep), vec![3, 4, 4]);
    assert_eq!(rs.lindep.total_kept, 11);
    assert_eq!(rk.lindep.total_kept, rs.lindep.total_kept);
    // Partners carry their representative's stats (S(-k) = S(k)*), so
    // compare counts, not last-bit eigenvalues.
    let direct = LindepReport::from_overlaps(&k.hk.s, TAU).unwrap();
    assert_eq!(kept(&rk.lindep), kept(&direct));
    assert_eq!(rk.lindep.overlap_abs_sum, direct.overlap_abs_sum);
    assert!(!rk.lindep.near_noise_floor);
    let ncell = (MESH[0] * MESH[1] * MESH[2]) as f64;
    let d = rk.energy - rs.energy / ncell;
    assert!(d.abs() <= 1e-12, "k-mesh - supercell {d:e} Ha/cell");
}

// -------------------------------------------------------- exp_to_discard

fn self_overlap(sh: &Shell) -> f64 {
    let lf = sh.l as f64;
    let mut s = 0.0;
    for (a, ca) in sh.exponents.iter().zip(&sh.coefficients) {
        for (b, cb) in sh.exponents.iter().zip(&sh.coefficients) {
            s += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(lf + 1.5);
        }
    }
    s
}

/// H: STO-3G (3.43, 0.624, 0.169), s 0.06, p 0.08, s 0.5.
fn h_basis_with_diffuse() -> BasisSet {
    let mut p = s_shell(0.08);
    p.l = 1;
    let mut shells = HashMap::new();
    shells.insert(
        1,
        vec![
            pyscf_sto3g_h().shells[&1][0].clone(),
            s_shell(0.06),
            p,
            s_shell(0.5),
        ],
    );
    BasisSet {
        name: "h-diffuse".into(),
        shells,
        ecps: HashMap::new(),
    }
}

#[test]
fn exp_to_discard_drops_exactly_the_requested_shells() {
    let bs = h_basis_with_diffuse();
    let orig = bs.shells[&1].clone();
    // 0.1: the whole diffuse s and p go; STO-3G (min 0.169) and s 0.5 stay
    // bit-for-bit.
    let (out, rep) = exp_to_discard(&bs, 0.1, &[1]).unwrap();
    assert_eq!(out.shells[&1], vec![orig[0].clone(), orig[3].clone()]);
    let idx: Vec<(usize, bool)> = rep
        .shells
        .iter()
        .map(|s| (s.shell_index, s.whole_shell))
        .collect();
    assert_eq!(idx, vec![(1, true), (2, true)]);
    assert_eq!((rep.n_shells_removed(), rep.n_primitives_removed()), (2, 2));
    assert_eq!(rep.shells[1].l, 1);
    // 0.2: STO-3G additionally loses its 0.169 primitive (partial, renormalised).
    let (out, rep) = exp_to_discard(&bs, 0.2, &[1]).unwrap();
    assert_eq!(out.shells[&1].len(), 2);
    let sto = &out.shells[&1][0];
    assert_eq!(sto.exponents, orig[0].exponents[..2].to_vec());
    assert!(
        (self_overlap(sto) - 1.0).abs() < 1e-14,
        "{}",
        self_overlap(sto)
    );
    assert_eq!(rep.shells[0].shell_index, 0);
    assert!(!rep.shells[0].whole_shell);
    assert_eq!(rep.shells[0].exponents_removed, vec![orig[0].exponents[2]]);
    assert_eq!((rep.n_shells_removed(), rep.n_primitives_removed()), (2, 3));
    // Below every exponent: identity, empty report.
    let (out, rep) = exp_to_discard(&bs, 1e-3, &[1]).unwrap();
    assert_eq!(out, bs);
    assert!(rep.shells.is_empty());
    // Through the cell: None is exactly PreparedBasis::new; Some(0.1) removes
    // 1 + 3 functions per H.
    let cell = h2_cell(A);
    let (p0, r0) = prepare_cell_basis(&cell, &bs, None).unwrap();
    assert!(r0.is_none());
    assert_eq!(p0.nbasis(), prep_for(&cell, &bs).nbasis());
    assert_eq!(p0.nbasis(), 2 * 6);
    let (p1, r1) = prepare_cell_basis(&cell, &bs, Some(0.1)).unwrap();
    assert_eq!(p1.nbasis(), 2 * 2);
    assert_eq!(r1.unwrap().n_shells_removed(), 2);
    let (p2, _) = prepare_cell_basis(&cell, &bs, Some(1e-3)).unwrap();
    assert_eq!(p2.nbasis(), p0.nbasis());
}

#[test]
fn exp_to_discard_errors_on_an_emptied_element_and_bad_input() {
    // The model's ghost carriers hold ONLY a diffuse s: 0.1 empties Z = 2.
    let bs = model_basis();
    match exp_to_discard(&bs, 0.1, &[1, 2, 3]) {
        Err(ExpToDiscardError::EmptiedElement { z, n_shells, .. }) => {
            assert_eq!((z, n_shells), (2, 1))
        }
        other => panic!("expected EmptiedElement, got {other:?}"),
    }
    assert!(prepare_cell_basis(&model_cell(), &bs, Some(0.1)).is_err());
    // An element outside the list is untouched even if it would empty.
    assert!(exp_to_discard(&bs, 0.1, &[1]).is_ok());
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            exp_to_discard(&bs, bad, &[1]),
            Err(ExpToDiscardError::InvalidThreshold(_))
        ));
    }
    assert_eq!(
        exp_to_discard(&bs, 0.1, &[9]).unwrap_err(),
        ExpToDiscardError::MissingElement { z: 9 }
    );
}
