//! A pure functional fits no exchange, and its DF gradient still reads the
//! SCF's RI-J tensor (`ferric_scf::raw_handoff`) instead of regenerating it
//! in `dfg:3c-pass`, with the gradient it would have recomputed.
//!
//! Exactness: the kept tier holds exactly what `eri3_block` returns when it
//! is held whole and unscreened in f64, and every aux row is contracted
//! independently of the blocking, so reuse and recompute contract
//! bit-identical rows in the same order. The bar is 0 bits.
//!
//! Each test checks the pass condition is reachable: the slot is FILLED
//! after the SCF (else "reuse" would be a recompute equal to itself), and the
//! `dfg:3c-pass` block counter tells reuse (0 blocks) from recompute.
//!
//! A pair-screened tensor is read too, with its own anchors in
//! `df_gradient_screened_j_handoff.rs`.
//!
//! Artifact hypothesis. A stale tensor would move the gradient by O(step) Ha/Bohr; a gradient that silently
//! recomputed would pass the equality and fail the counter.

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_gradient::dfg_3c_blocks_computed_for_test;
use ferric_scf::gradient::{
    restricted_open_scf_gradient, restricted_scf_gradient, unrestricted_scf_gradient,
};
use ferric_scf::raw_handoff::with_fresh_handoff;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_testkit::molecules::water_xyz;
use ndarray::Array2;

const AUX: &str = "def2-universal-jkfit";
const BASIS: &str = "def2-svp";

#[derive(Clone, Copy)]
enum Ref {
    Rks,
    Uks,
    Roks,
}

/// PBE with RI-J only: a pure functional builds no K.
fn pbe_geometry_config() -> RhfConfig {
    {
        let mut c = RhfConfig::default().with_xc("PBE").with_df_j_aux(AUX);
        c.df_k_aux = None;
        c
    }
    .for_geometry_task()
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

fn scf(s: &Setup, r: Ref, cfg: &RhfConfig) -> ferric_scf::result::ScfResult {
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
fn gradient(
    s: &Setup,
    r: Ref,
    cfg: &RhfConfig,
    res: &ferric_scf::result::ScfResult,
) -> (Array2<f64>, usize) {
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

fn bits(g: &Array2<f64>) -> Vec<u64> {
    g.iter().map(|x| x.to_bits()).collect()
}

/// SCF with a hand-off, gradient reusing it, gradient recomputing: asserts
/// the reuse really happened and returns both gradients.
fn reuse_and_recompute(s: &Setup, r: Ref) -> (Array2<f64>, Array2<f64>) {
    let cfg = with_fresh_handoff(&pbe_geometry_config());
    let res = scf(s, r, &cfg);
    let slot = cfg.raw_handoff.as_ref().expect("hand-off installed");
    assert!(
        slot.is_filled(),
        "a pure functional's RI-J tensor was not kept (reachability)"
    );
    let (g_reuse, computed_reuse) = gradient(s, r, &cfg, &res);
    assert!(!slot.is_filled(), "the gradient took the tensor");
    assert_eq!(computed_reuse, 0, "reuse computed raw blocks");
    let (g_rec, computed_rec) = gradient(s, r, &cfg, &res);
    assert!(computed_rec > 0, "recompute computed no raw blocks");
    assert!(
        g_rec.iter().any(|v| v.abs() > 1e-3),
        "vacuous: the gradient is ~zero"
    );
    (g_reuse, g_rec)
}

fn hydroxyl() -> Molecule {
    Molecule::parse_xyz("2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).expect("OH")
}

#[test]
fn pure_functional_gradient_reads_the_kept_ri_j_tensor_bitwise() {
    let s = setup(water_xyz());
    let (reuse, recompute) = reuse_and_recompute(&s, Ref::Rks);
    assert_eq!(bits(&reuse), bits(&recompute));
}

#[test]
fn pure_functional_open_shell_gradients_read_the_kept_tensor_bitwise() {
    let s = setup(hydroxyl());
    for r in [Ref::Uks, Ref::Roks] {
        let (reuse, recompute) = reuse_and_recompute(&s, r);
        assert_eq!(bits(&reuse), bits(&recompute));
    }
}
