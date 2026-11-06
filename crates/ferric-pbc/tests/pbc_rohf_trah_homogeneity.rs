//! Homogeneity of the second-order (TRAH) ROHF/ROKS orbital-Hessian product
//! on the injected periodic path, `H(s·x) = s·H(x)`.
//!
//! The molecular response matvecs (Z-vector, CPHF, Newton, stability) lost
//! two-electron coupling on small operands because `build_jk_with_pool`
//! screens quartets on the ABSOLUTE `bound · max|δD| < thresh`; they now go
//! through `rhf::build_jk_with_pool_homogeneous` (relative miss at s = 1e-7
//! was 4.5e-6 .. 5.4e-5 before that fix, 2.4e-3 .. 2.4e-2 at 2⁻³⁰;
//! `ferric-scf/tests/response_matvec_homogeneity.rs`). The ROHF TRAH product
//! (`rohf_trah::rohf_exact_hessian_matvec` with the injected model's
//! `response`) does NOT use that builder: its J/K are the injected periodic
//! builders (dense-AFT `Σ (mk|ls) D_ls`, RS-GDF `B(B·D)` / `Σ B D Bᵀ`, plus
//! the Madelung `v_M S D S`), plain linear contractions with no
//! density-dependent screen, and its `δV_xc` is a central FD whose step is
//! `fxc_fd_step / max|δD|`, i.e. normalized to the operand.
//!
//! Hypotheses, stated before measuring:
//! * if the product is homogeneous by construction (no absolute screen
//!   anywhere), power-of-two scaling is exact through every operation, so the
//!   miss is EXACTLY 0 at s = 2⁻²⁰ and 2⁻³⁰, and at decimal s only the one
//!   rounding of the operand remains (≲1e-14; the FD XC perturbation
//!   `t·δD` is the same matrix up to that rounding at every s);
//! * if an absolute screen hides in a builder (the molecular defect), the
//!   miss is nonzero at 2⁻²⁰ and GROWS as s shrinks (1e-6 .. 1e-2 in the
//!   fixed paths).
//!
//! The two are distinguishable at the 2⁻²⁰ / 2⁻³⁰ points alone. Measured
//! (2026-11-06): 0 exactly at 2⁻²⁰ and 2⁻³⁰ in all three cases; decades
//! 1e-2 .. 1e-9 at ≤ 1.7e-15 (tri dense-AFT), ≤ 2.8e-16 (H3 RS-GDF),
//! ≤ 3.3e-16 (H3 ROKS/PBE0), flat in s. Mutation
//! check (done when this test was written): an absolute `|δD| < 1e-10` screen
//! in the model's response, or an absolute (un-normalized) FD step, makes
//! these tests fail.

mod common;
#[path = "pbc_rohf_support/mod.rs"]
mod support;

use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::rhf::PeriodicInjection;
use ferric_scf::rohf_trah::{
    injected_rohf_model, rohf_exact_hessian_matvec, rohf_rotation_count, RohfPoint, RohfTrahConfig,
};
use support::*;

/// A smooth deterministic packed rotation, max ~0.1.
fn packed(np: usize, phase: f64) -> Vec<f64> {
    (0..np)
        .map(|p| 0.1 * (phase + 0.9 * p as f64).sin() / (1.0 + 0.03 * p as f64))
        .collect()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

const DECADES: [f64; 6] = [1e-2, 1e-4, 1e-6, 1e-7, 1e-8, 1e-9];
const POW2: [i32; 2] = [-20, -30];

/// Relative misses `max|H(s·x)/s − H(x)| / max|H(x)|` at the production
/// TRAH product of `inj` at the point `c` (closed | open | virtual), for the
/// power-of-two scales `2^POW2` then the decimal `DECADES`.
fn misses(
    inj: PeriodicInjection<'_>,
    c: Array2<f64>,
    nd: usize,
    no: usize,
) -> (Vec<f64>, Vec<f64>) {
    let model = injected_rohf_model(inj, RohfTrahConfig::default().fxc_fd_step).expect("model");
    let pt = RohfPoint::at(&*model, c, nd, no).expect("point");
    let resp = |a: &Array2<f64>, b: &Array2<f64>| model.response(&pt.d_a, &pt.d_b, a, b);
    let h = |v: &[f64]| {
        rohf_exact_hessian_matvec(&pt.c, &pt.f_a_mo, &pt.f_b_mo, nd, no, &resp, v).expect("H x")
    };
    let np = rohf_rotation_count(pt.c.ncols(), nd, no);
    assert!(np > 0);
    let x = packed(np, 0.4);
    let full = h(&x);
    let hmax = max_abs(&full);
    assert!(hmax > 1e-3, "degenerate product, max|Hx| = {hmax:e}");
    let miss = |s: f64| {
        let xs: Vec<f64> = x.iter().map(|v| s * v).collect();
        let small = h(&xs);
        let d: Vec<f64> = small.iter().zip(&full).map(|(a, b)| a / s - b).collect();
        max_abs(&d) / hmax
    };
    (
        POW2.iter().map(|&e| miss(2f64.powi(e))).collect(),
        DECADES.iter().map(|&s| miss(s)).collect(),
    )
}

/// Prints the sweep; asserts bit-exactness at every power of two and
/// `decade_bar` at every decade.
fn check(what: &str, (pow2, dec): (Vec<f64>, Vec<f64>), decade_bar: f64) {
    for (e, m) in POW2.iter().zip(&pow2) {
        eprintln!("{what}: s = 2^{e}  rel miss = {m:.3e}");
    }
    for (s, m) in DECADES.iter().zip(&dec) {
        eprintln!("{what}: s = {s:.0e}  rel miss = {m:.3e}");
    }
    for (e, m) in POW2.iter().zip(&pow2) {
        assert!(*m == 0.0, "{what}: H(2^{e} x)/2^{e} - H(x) rel {m:.3e}");
    }
    let worst = dec.iter().fold(0.0f64, |m, v| m.max(*v));
    assert!(
        worst < decade_bar,
        "{what}: worst decade rel miss {worst:.3e}"
    );
}

fn injection<'a>(
    su: &'a Setup,
    j: Box<dyn JBuilder + 'a>,
    k: Box<dyn KBuilder + 'a>,
    xc: Option<Box<dyn XcBuilder + 'a>>,
) -> PeriodicInjection<'a> {
    PeriodicInjection {
        s: su.hc.s.clone(),
        h: su.hc.h.clone(),
        vnn: su.hc.enn,
        j,
        k,
        xc,
    }
}

/// A generic (non-stationary) point: the core guess rotated by a small
/// random antisymmetric generator, so the MO Focks carry off-diagonal blocks.
fn point(su: &Setup) -> Array2<f64> {
    rotated(&core_mos(&su.hc.s, &su.hc.h), 0.05, 7)
}

/// Tri 4H s+p ROHF triplet (nd 1, no 2) on the dense-AFT J/K with the
/// Madelung term in K (exxdiv = ewald builder).
#[test]
fn rohf_trah_product_is_homogeneous_on_dense_aft() {
    let su = setup(cell_mult(&TRI_ATOMS, TRI_A, 3), sp_basis_h());
    let vm = madelung_constant(&su.cell).unwrap();
    let inj = injection(
        &su,
        Box::new(su.eri.j_builder()),
        Box::new(su.eri.k_builder_with_madelung(vm)),
        None,
    );
    check(
        "ROHF TRAH tri dense-AFT",
        misses(inj, point(&su), 1, 2),
        1e-12,
    );
}

/// H3 doublet ROHF (nd 1, no 1) on the production RS-GDF J/K (+ Madelung).
#[test]
fn rohf_trah_product_is_homogeneous_on_rsgdf() {
    let su = setup(cell_mult(&H3_ATOMS, cubic(H3_A), 2), pyscf_sto3g_h());
    let aux = ferric_integrals::basis_bridge::PreparedBasis::new(
        su.cell.mol(),
        &ferric_core::basis::bundled("cc-pvdz-ri").unwrap(),
    )
    .unwrap();
    let gdf = RsGdf::build(
        &su.cell,
        &su.prep,
        &aux,
        &su.hc.s,
        &RsGdfConfig {
            exxdiv: ExxDiv::Ewald,
            ..Default::default()
        },
    )
    .unwrap();
    let vm = madelung_constant(&su.cell).unwrap();
    let inj = injection(
        &su,
        Box::new(gdf.j_builder()),
        Box::new(gdf.k_builder_with_madelung(vm)),
        None,
    );
    check("ROHF TRAH H3 RS-GDF", misses(inj, point(&su), 1, 1), 1e-12);
}

/// H3 doublet ROKS/PBE0 on the dense-AFT J/K: exercises the central-FD
/// `δV_xc` (step normalized by `max|δD|`) on a real periodic XC builder.
#[test]
fn roks_trah_product_is_homogeneous_with_fd_xc() {
    let su = setup(cell_mult(&H3_ATOMS, cubic(H3_A), 2), pyscf_sto3g_h());
    let grid = PeriodicGrid::build(&su.cell, &ssf_grid(30, 110, 10.0)).expect("grid");
    let pxc = PeriodicXc::new(
        &su.cell,
        su.prep.basis_set(),
        "PBE0",
        &grid,
        &PeriodicXcConfig::default(),
    )
    .expect("PeriodicXc PBE0");
    let inj = injection(
        &su,
        Box::new(su.eri.j_builder()),
        Box::new(su.eri.k_builder_with_madelung(0.0)),
        Some(Box::new(pxc)),
    );
    check(
        "ROKS/PBE0 TRAH H3 dense-AFT",
        misses(inj, point(&su), 1, 1),
        1e-12,
    );
}
