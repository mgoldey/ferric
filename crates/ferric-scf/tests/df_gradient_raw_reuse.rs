//! The DF-JK gradient reuses the SCF's raw `(P|μν)` tensor
//! (`ferric_scf::raw_handoff`) instead of regenerating it in `dfg:3c-pass`,
//! and gets the gradient it would have recomputed.
//!
//! Exactness: the kept tiers hold exactly what `eri3_block` returns (the
//! unpacked tier the block itself; the packed tier the lower triangle of an
//! exactly symmetric block, `eri3_block` writes `(μν)` and `(νμ)` from one
//! value), and each aux row is contracted independently of the blocking
//! (`df_gradient_one_pass_blocking.rs`). So reuse and recompute contract
//! bit-identical rows in the same order, and the bar is the brief's 1e-12
//! Ha/Bohr with an expected difference of exactly 0. What would move it: a
//! wrong row offset or band (O(1e-2) Ha/Bohr), a stale tensor from another
//! geometry (O(step) Ha/Bohr), or a pair-screened / f32 tensor (which the
//! hand-off refuses to keep).
//!
//! Each test checks the pass condition is reachable: the slot is FILLED after
//! the SCF (else "reuse" would silently be a recompute equal to itself), and
//! the `dfg:3c-pass` block counter tells reuse (0 blocks) from recompute.
//!
//! Mutations seen to fail (2026-11-10):
//! - `routed_two_electron_gradient` passing `None` instead of
//!   `take_active()`: both `*_equals_recomputed` tests fail on the counter
//!   ("reuse computed raw blocks", left 1).
//! - `ThreeIndexSource::rows_f64` viewing the unpacked tier's rows reversed
//!   (`s![l0..l1;-1, ..]`): water fails (4.97e4) and OH fails (1.51e4).
//!   (A row OFFSET is not a mutation here: water's whole aux range is one
//!   block, so `l0 / 2` was an identity and survived.)
//! - The packed tier unpacking only the lower triangle (upper left zero):
//!   n-octane fails (5.84e-1 Ha/Bohr).
//! - No slot clearing (solver entry, `keep_for_gradient`'s take-first) and no
//!   stripping in `basis2::secondary_config`:
//!   `guess_basis_tensor_never_reaches_the_gradient` fails ("the slot holds a
//!   foreign tensor: Err(\"tensor shape differs\")").
//! - `KeptRaw::check` skipping the geometry comparison:
//!   `mismatched_geometry_recomputes` fails ("a tensor from another geometry
//!   was used").

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_gradient::dfg_3c_blocks_computed_for_test;
use ferric_scf::gradient::{
    restricted_open_scf_gradient, restricted_scf_gradient, unrestricted_scf_gradient,
};
use ferric_scf::raw_handoff::{with_fresh_handoff, RawHandoff};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

const AUX: &str = "def2-universal-jkfit";
const BASIS: &str = "def2-svp";
/// The brief's bar; the expected difference is exactly 0 (module docs).
const BAR: f64 = 1e-12;

#[derive(Clone, Copy)]
enum Ref {
    Rks,
    Uks,
    Roks,
}

fn b3lyp_geometry_config() -> RhfConfig {
    RhfConfig {
        xc: Some("B3LYP".into()),
        df_j_aux: Some(AUX.into()),
        df_k_aux: Some(AUX.into()),
        ..RhfConfig::default()
    }
    .for_geometry_task()
}

fn handoff(cfg: &RhfConfig) -> &RawHandoff {
    cfg.raw_handoff.as_ref().expect("hand-off installed")
}

struct Setup {
    mol: Molecule,
    bs: ferric_core::basis::BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
}

fn setup(mol: Molecule) -> Setup {
    let bs = ferric_core::basis::bundled(BASIS).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).expect("schwarz");
    Setup {
        mol,
        bs,
        prep,
        bounds,
    }
}

fn scf(s: &Setup, r: Ref, cfg: &RhfConfig) -> ScfResult {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let res = match r {
        Ref::Rks => solve_rhf(&ctx, &s.mol, &s.prep, op, &s.bounds, cfg),
        Ref::Uks => solve_uhf(&ctx, &s.mol, &s.prep, &s.bounds, cfg),
        Ref::Roks => solve_rohf(&ctx, &s.mol, &s.prep, op, &s.bounds, cfg),
    }
    .expect("SCF");
    assert!(res.converged, "SCF did not converge");
    res
}

/// The gradient and how many raw blocks its `dfg:3c-pass` computed.
fn gradient(s: &Setup, r: Ref, cfg: &RhfConfig, res: &ScfResult) -> (Array2<f64>, usize) {
    let op = Operator::coulomb();
    let before = dfg_3c_blocks_computed_for_test();
    let g = match r {
        Ref::Rks => restricted_scf_gradient(&s.mol, &s.prep, &s.bs, op, &s.bounds, cfg, res),
        Ref::Uks => unrestricted_scf_gradient(&s.mol, &s.prep, &s.bs, op, &s.bounds, cfg, res),
        Ref::Roks => restricted_open_scf_gradient(&s.mol, &s.prep, &s.bs, op, &s.bounds, cfg, res),
    }
    .expect("gradient");
    (g, dfg_3c_blocks_computed_for_test() - before)
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// SCF with a hand-off, gradient reusing it, gradient recomputing; returns
/// max |g_reuse − g_recompute| after asserting the reuse really happened.
/// `expect_bytes`: the kept tensor's size, which names its tier.
fn reuse_vs_recompute(s: &Setup, r: Ref, cfg: &RhfConfig, expect_bytes: Option<usize>) -> f64 {
    let cfg = with_fresh_handoff(cfg);
    let res = scf(s, r, &cfg);
    let slot = handoff(&cfg);
    let kept = slot
        .take()
        .expect("the SCF kept its raw tensor (reachability)");
    if let Some(b) = expect_bytes {
        assert_eq!(kept.bytes(), b, "kept tensor is not the expected tier");
    }
    slot.put_for_test(kept);

    let (g_reuse, computed_reuse) = gradient(s, r, &cfg, &res);
    assert!(!slot.is_filled(), "the gradient took the tensor");
    assert_eq!(computed_reuse, 0, "reuse computed raw blocks");
    let (g_rec, computed_rec) = gradient(s, r, &cfg, &res);
    assert!(computed_rec > 0, "recompute computed no raw blocks");
    assert!(
        g_rec.iter().any(|v| v.abs() > 1e-3),
        "vacuous: the gradient is ~zero"
    );
    max_abs_diff(&g_reuse, &g_rec)
}

fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz")
}

#[test]
fn reused_raw_gradient_equals_recomputed() {
    // Water, default budget: the unpacked in-core tier.
    let s = setup(water());
    let d = reuse_vs_recompute(&s, Ref::Rks, &b3lyp_geometry_config(), None);
    eprintln!("water/def2-SVP B3LYP: max|g_reuse - g_recompute| = {d:.3e}");
    assert!(d <= BAR, "water: {d:e}");

    // n-octane under a budget that holds the raw tensor packed (the unpacked
    // tensor and DF-K's dressed band do not fit together, the packed one
    // does): the packed tier, read by unpacking.
    let s = setup(Molecule::load_xyz("../../testdata/molecules/alkane_8.xyz").expect("octane"));
    let (nao, naux) = (s.prep.nbasis(), {
        let aux = ferric_core::basis::bundled(AUX).unwrap();
        PreparedBasis::new(&s.mol, &aux).unwrap().nbasis()
    });
    let full = naux * nao * nao * 8;
    let packed = naux * nao * (nao + 1) / 2 * 8;
    let dressed = ferric_scf::df_k::DfK::packed_band_bytes(
        naux,
        nao,
        ferric_scf::df_k::RiKPrecision::resolve(None),
    );
    let cfg = RhfConfig {
        three_index_budget_bytes: full + dressed - 1,
        ..b3lyp_geometry_config()
    };
    let d = reuse_vs_recompute(&s, Ref::Rks, &cfg, Some(packed));
    eprintln!("n-octane/def2-SVP B3LYP (packed): max|g_reuse - g_recompute| = {d:.3e}");
    assert!(d <= BAR, "octane: {d:e}");
}

#[test]
fn reused_raw_open_shell_gradients_equal_recomputed() {
    let oh = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
    let s = setup(Molecule::parse_xyz(oh, 0, 2).expect("OH"));
    for r in [Ref::Uks, Ref::Roks] {
        let d = reuse_vs_recompute(&s, r, &b3lyp_geometry_config(), None);
        assert!(d <= BAR, "OH: {d:e}");
    }
}

#[test]
fn mismatched_geometry_recomputes() {
    let a = setup(water());
    let mut moved = water();
    moved.atoms[1].y += 0.01;
    let b = setup(moved);
    let base = b3lyp_geometry_config();

    let cfg_a = with_fresh_handoff(&base);
    let _ = scf(&a, Ref::Rks, &cfg_a);
    let kept_a = handoff(&cfg_a).take().expect("A kept its tensor");

    // Geometry B, with A's tensor in the slot instead of its own.
    let cfg_b = with_fresh_handoff(&base);
    let res_b = scf(&b, Ref::Rks, &cfg_b);
    assert!(handoff(&cfg_b).is_filled(), "B kept its tensor");
    handoff(&cfg_b).put_for_test(kept_a);
    let (g_stale, computed) = gradient(&b, Ref::Rks, &cfg_b, &res_b);
    assert!(computed > 0, "a tensor from another geometry was used");

    let (g_plain, _) = gradient(&b, Ref::Rks, &base, &res_b);
    let d = max_abs_diff(&g_stale, &g_plain);
    assert!(d <= BAR, "mismatched hand-off moved the gradient by {d:e}");
}

#[test]
fn guess_basis_tensor_never_reaches_the_gradient() {
    // The guess_basis secondary SCF (STO-3G) runs through a clone of the
    // config. Under a budget that keeps ITS tensor but not the main one
    // (def2-SVP), a slot the secondary SCF could fill — and nobody cleared —
    // would hand the gradient a tensor in the wrong basis. The slot must hold
    // the main SCF's tensor or nothing.
    let s = setup(water());
    let aux = ferric_core::basis::bundled(AUX).unwrap();
    let naux = PreparedBasis::new(&s.mol, &aux).unwrap().nbasis();
    let sto = ferric_core::basis::bundled("sto-3g").unwrap();
    let nao_s = PreparedBasis::new(&s.mol, &sto).unwrap().nbasis();
    let nocc = (s.mol.nelec() / 2) as usize;
    // Secondary unpacked tensor + intermediates fit; the main one does not.
    let budget = naux * nao_s * nao_s * 8
        + ferric_scf::raw_handoff::gradient_intermediate_bytes(naux, &[nocc])
        + 1;
    let cfg = with_fresh_handoff(&RhfConfig {
        three_index_budget_bytes: budget,
        guess_basis: Some(ferric_scf::basis2::GuessBasis::bundled("sto-3g", "fock").unwrap()),
        ..b3lyp_geometry_config()
    });
    let res = scf(&s, Ref::Rks, &cfg);
    assert!(res.guess_basis.is_some(), "the guess-basis SCF did not run");
    let dfbs = PreparedBasis::new(&s.mol, &aux).unwrap();
    if let Some(k) = handoff(&cfg).take() {
        let check = k.check(&s.mol, &s.prep, &dfbs, AUX, Operator::coulomb());
        assert!(check.is_ok(), "the slot holds a foreign tensor: {check:?}");
    }
}
