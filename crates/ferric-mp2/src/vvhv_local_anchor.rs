//! The trivial-limit anchor harness of the local VV-HV build (design T-A):
//! the local build against the dense build on the same SCF, column by column
//! and in the energy. Test-only (`#[cfg(test)]` module of
//! [`crate::vvhv_local`]).

use super::*;
use crate::lmp2_amplitude::{
    build_vvhv_traced_dense, check_vvhv, match_columns_up_to_sign_and_order, AmplitudeLmp2Config,
    LMP2_CG_RTOL_STRICT,
};
use crate::lmp2_direct::{amplitude_lmp2_direct_with_virtuals, DirectConfig};
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

pub(crate) struct Su {
    pub mol: Molecule,
    pub obs: PreparedBasis,
    pub obs_bs: BasisSet,
    pub dfbs: PreparedBasis,
    pub rhf: ScfResult,
    pub nocc: usize,
}

/// 6-31G / cc-pVDZ-RI RHF of a test molecule, converged tightly.
pub(crate) fn setup(xyz: &str) -> Su {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = ferric_core::basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs =
        PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
    let nocc = (mol.nelec() as usize) / 2;
    Su {
        mol,
        obs,
        obs_bs,
        dfbs,
        rhf,
        nocc,
    }
}

/// E_corr and kept fraction with production integral-direct maps (the
/// bench's) and strict CG, on a given virtual space.
pub(crate) fn energy(su: &Su, eps: f64, vv: &VvHv) -> (f64, f64) {
    let cfg = AmplitudeLmp2Config {
        eps,
        frozen_core: 0,
        cg_rtol: Some(LMP2_CG_RTOL_STRICT),
        pair_gate_cal: Some(0.7),
        ..Default::default()
    };
    let dcfg = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5,
        batch_merge: 4,
        ..Default::default()
    };
    let (r, _) = amplitude_lmp2_direct_with_virtuals(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        &dcfg,
        vv,
    )
    .unwrap();
    (r.e_corr, r.keep_fraction)
}

/// Largest over the columns of `a` of the distance (max-abs, up to sign) to
/// the nearest column of `b`: the size of a column mismatch.
pub(crate) fn max_column_distance(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.columns()
        .into_iter()
        .map(|ca| {
            b.columns()
                .into_iter()
                .map(|cb| {
                    let s = if ca.dot(&cb) < 0.0 { -1.0 } else { 1.0 };
                    ca.iter()
                        .zip(cb.iter())
                        .fold(0.0f64, |m, (x, y)| m.max((x - s * y).abs()))
                })
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

/// What one anchor comparison measured.
#[derive(Debug)]
pub(crate) struct AnchorReport {
    pub same_counts: bool,
    pub same_pivots: bool,
    pub columns_match: bool,
    pub max_col_dist: f64,
    pub dev_orth: f64,
    pub dev_span: f64,
    /// (eps, |E_dense − E_local|, same kept fraction)
    pub energies: Vec<(f64, f64, bool)>,
    pub stats: LocalVvhvStats,
}

impl AnchorReport {
    /// Every T-A condition at the 1e-10 bars.
    pub fn passes(&self) -> bool {
        self.same_counts
            && self.same_pivots
            && self.columns_match
            && self.dev_orth <= 1e-10
            && self.dev_span <= 1e-10
            && self.energies.iter().all(|&(_, de, k)| de <= 1e-10 && k)
    }
}

/// The dense build of `su` with its hard-virtual trace.
pub(crate) fn dense(su: &Su) -> (VvHv, Vec<usize>) {
    build_vvhv_traced_dense(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap()
}

/// Compare a local build against the dense build on `su`; energies at each
/// of `eps` (production maps, strict CG).
pub(crate) fn compare_built(
    su: &Su,
    dense: &VvHv,
    dense_hard: &[usize],
    loc: &LocalVvhv,
    eps: &[f64],
) -> AnchorReport {
    let same_counts = (loc.vv.n_valence, loc.vv.n_hard) == (dense.n_valence, dense.n_hard);
    let columns_match = loc.vv.c_vloc.dim() == dense.c_vloc.dim()
        && match_columns_up_to_sign_and_order(&loc.vv.c_vloc, &dense.c_vloc).0;
    let max_col_dist = max_column_distance(&loc.vv.c_vloc, &dense.c_vloc);
    let (dev_orth, dev_span) = check_vvhv(&su.obs, &su.rhf, su.nocc, &loc.vv.c_vloc);
    let energies = eps
        .iter()
        .map(|&e| {
            let (ed, kd) = energy(su, e, dense);
            let (el, kl) = energy(su, e, &loc.vv);
            (e, (ed - el).abs(), kd == kl)
        })
        .collect();
    AnchorReport {
        same_counts,
        same_pivots: loc.hard_aos == dense_hard,
        columns_match,
        max_col_dist,
        dev_orth,
        dev_span,
        energies,
        stats: loc.stats.clone(),
    }
}

/// Local build at `cfg` against the dense build.
pub(crate) fn compare(su: &Su, cfg: &LocalVvhvConfig, eps: &[f64]) -> AnchorReport {
    let (d, dh) = dense(su);
    let loc = build_vvhv_local(&su.mol, &su.obs, &su.obs_bs, &su.rhf, cfg).unwrap();
    compare_built(su, &d, &dh, &loc, eps)
}

/// The counters every sparse stage that has landed must leave nonzero in
/// the trivial limit (so the anchor cannot pass on a silent fallback to the
/// dense build), and nothing dropped.
pub(crate) fn assert_reached(st: &LocalVvhvStats) {
    assert_eq!(st.blocks_dropped, 0, "trivial limit dropped blocks: {st:?}");
}

/// T-A: in the trivial limit the local build reproduces the dense build
/// (counts, hard-virtual pivot sequence, columns up to sign/order <= 1e-10,
/// the construction check <= 1e-10, and the energy <= 1e-10 at eps = 0 and
/// 1e-4 with production maps and strict CG) on water, C4 and C8.
#[test]
fn local_vvhv_trivial_limit_reproduces_build_vvhv_columns_and_energies() {
    for xyz in ["water.xyz", "alkane_4.xyz", "alkane_8.xyz"] {
        let su = setup(xyz);
        let r = compare(&su, &LocalVvhvConfig::trivial(), &[0.0, 1e-4]);
        eprintln!("T-A {xyz}: max column distance {:.2e}; {r:?}", r.max_col_dist);
        assert!(r.passes(), "{xyz}: {r:?}");
        assert_reached(&r.stats);
    }
}
