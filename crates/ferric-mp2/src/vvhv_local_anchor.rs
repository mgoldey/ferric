//! The trivial-limit anchor harness of the local VV-HV build (design T-A):
//! the local build against the dense build on the same SCF, column by column
//! and in the energy. Test-only (`#[cfg(test)]` module of
//! [`crate::vvhv_local`]).

use super::ValenceStart;
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
    /// max |C_occᵀ S C_v|: the occupied leak, which check_vvhv sees only at
    /// SECOND order (in dev_orth; U = C_canᵀ S C_v annihilates it).
    pub occ_leak: f64,
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
            && self.occ_leak <= 1e-10
            && self.energies.iter().all(|&(_, de, k)| de <= 1e-10 && k)
    }
}

/// max |C_occᵀ S C_v| over the canonical occupieds.
pub(crate) fn occupied_leak(su: &Su, c_v: &Array2<f64>) -> f64 {
    let s = ferric_integrals::oneelectron::overlap(&su.obs);
    let c_o = su.rhf.mos_r().slice(ndarray::s![.., ..su.nocc]).to_owned();
    c_o.t()
        .dot(&s.dot(c_v))
        .iter()
        .fold(0.0f64, |m, x| m.max(x.abs()))
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
    let occ_leak = occupied_leak(su, &loc.vv.c_vloc);
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
        occ_leak,
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
    assert!(st.solve_domain_aos_max > 0, "A1 not reached: {st:?}");
    assert!(st.n_products >= 3, "A2 products not reached: {st:?}");
    assert!(st.heap_pops > 0, "A6 heap pivots not reached: {st:?}");
    assert!(
        st.ns_iterations_hard > 0,
        "A7 hard Löwdin not reached: {st:?}"
    );
    assert!(
        st.n_fock_atom_blocks > 0,
        "A8 Fock blocks not reached: {st:?}"
    );
    assert!(
        st.boys_sweeps > 0 && st.boys_pairs_evaluated > 0,
        "F not reached: {st:?}"
    );
    assert!(st.ns_iterations_guard > 0, "A9 guard not reached: {st:?}");
}

/// The SCF's returned density lags its returned MOs by one
/// diagonalization: max |density_alpha − C_o C_oᵀ| is at the convergence
/// level, not at rounding (why the local build forms C_o C_oᵀ itself).
#[test]
fn density_alpha_lags_the_mos() {
    let su = setup("alkane_8.xyz");
    let c = su.rhf.mos_r().slice(ndarray::s![.., ..su.nocc]).to_owned();
    let d = c.dot(&c.t());
    let dev = (&su.rhf.density_alpha - &d)
        .iter()
        .fold(0.0f64, |m, x| m.max(x.abs()));
    eprintln!("C8: max |density_alpha - C_o C_o^T| = {dev:.3e}");
    assert!(dev.is_finite());
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
        eprintln!(
            "T-A {xyz}: max column distance {:.2e}; {r:?}",
            r.max_col_dist
        );
        assert!(r.passes(), "{xyz}: {r:?}");
        assert_reached(&r.stats);
    }
}

/// E_corr at eps = 0 with every integral-direct map at its trivial limit
/// (strict CG), and the canonical RI-MP2 on the same inputs: on these maps
/// the eps = 0 energy depends only on the SPAN of the virtual space.
pub(crate) fn energy_span(su: &Su, vv: &VvHv) -> (f64, f64) {
    let cfg = AmplitudeLmp2Config {
        eps: 0.0,
        frozen_core: 0,
        cg_rtol: Some(LMP2_CG_RTOL_STRICT),
        compute_reference: true,
        ..Default::default()
    };
    let dcfg = DirectConfig {
        aux_radius_bohr: 1e6,
        virt_radius_bohr: None,
        ao_tail: 0.0,
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
    (r.e_corr, r.e_corr_canonical_ri)
}

/// T-B (classification, non-vacuous): loose settings on the BASIS-CHOICE
/// steps only (span-critical steps exact) rotate the columns (> 1e-6 from
/// the dense build: the knob did something) but leave the span exact
/// (check <= 1e-10) and so the eps = 0 energy fixed: <= 1e-10 vs the dense
/// build and <= 1e-9 vs canonical RI-MP2, C8.
#[test]
fn basis_choice_truncation_leaves_the_eps0_energy_fixed() {
    let su = setup("alkane_8.xyz");
    let cfg = LocalVvhvConfig {
        solve_radius_bohr: Some(6.0),
        ..LocalVvhvConfig::trivial()
    };
    let (d, _) = dense(&su);
    let loc = build_vvhv_local(&su.mol, &su.obs, &su.obs_bs, &su.rhf, &cfg).unwrap();
    let dist = max_column_distance(&loc.vv.c_vloc, &d.c_vloc);
    let (o, sp) = check_vvhv(&su.obs, &su.rhf, su.nocc, &loc.vv.c_vloc);
    let (e_d, e_can) = energy_span(&su, &d);
    let (e_l, _) = energy_span(&su, &loc.vv);
    eprintln!(
        "T-B C8: column distance {dist:.2e}, orth {o:.2e} span {sp:.2e}, |dE| vs dense \
         {:.2e}, vs canonical {:.2e}; {:?}",
        (e_l - e_d).abs(),
        (e_l - e_can).abs(),
        loc.stats
    );
    assert!(
        dist > 1e-6,
        "the basis-choice knobs changed nothing ({dist:.2e})"
    );
    assert!(
        loc.stats.solve_domain_aos_max < su.obs.nbasis(),
        "domain solve not reached"
    );
    assert!(o <= 1e-10 && sp <= 1e-10, "orth {o:.2e} span {sp:.2e}");
    assert!((e_l - e_d).abs() <= 1e-10, "{:.2e}", (e_l - e_d).abs());
    assert!((e_l - e_can).abs() <= 1e-9, "{:.2e}", (e_l - e_can).abs());
}

/// The Boys functional Σ_k |⟨k|r|k⟩|² of the columns of `c`.
fn boys_functional(su: &Su, c: &Array2<f64>) -> f64 {
    let dip = ferric_integrals::oneelectron::dipole(&su.obs, [0.0; 3]).unwrap();
    let mut f = 0.0;
    for dm in &dip {
        let dc = dm.dot(c);
        for k in 0..c.ncols() {
            f += c.column(k).dot(&dc.column(k)).powi(2);
        }
    }
    f
}

/// T-A2: the LOCAL valence route (purification + column-normalized pivots +
/// Newton-Schulz Löwdin) at zero thresholds. The span is the dense build's
/// (check <= 1e-10), the Boys functional of the valence block is equal to
/// 1e-12 relative (same maximum), the hard-virtual pivot sequence and counts
/// are unchanged and the energies agree to 1e-10 at eps 0 and 1e-4 with
/// production maps. The COLUMNS differ at Boys's stopping tolerance (the
/// start's imprint; REPORTED, bounded loosely at 1e-8 so that a different
/// maximum, which moves columns at O(1), still fails).
#[test]
fn purified_valence_start_reaches_the_same_space_and_boys_maximum() {
    let cfg = LocalVvhvConfig {
        valence_start: ValenceStart::Purified,
        ..LocalVvhvConfig::trivial()
    };
    for xyz in ["water.xyz", "alkane_4.xyz", "alkane_8.xyz"] {
        let su = setup(xyz);
        let (d, dh) = dense(&su);
        let loc = build_vvhv_local(&su.mol, &su.obs, &su.obs_bs, &su.rhf, &cfg).unwrap();
        let r = compare_built(&su, &d, &dh, &loc, &[0.0, 1e-4]);
        let nl = d.n_valence;
        let fd = boys_functional(&su, &d.c_vloc.slice(ndarray::s![.., ..nl]).to_owned());
        let fl = boys_functional(&su, &loc.vv.c_vloc.slice(ndarray::s![.., ..nl]).to_owned());
        eprintln!(
            "T-A2 {xyz}: column distance {:.2e}, Boys functional rel diff {:.2e}; {r:?}",
            r.max_col_dist,
            ((fl - fd) / fd).abs()
        );
        assert!(r.same_counts && r.same_pivots, "{xyz}: {r:?}");
        assert!(r.dev_orth <= 1e-10 && r.dev_span <= 1e-10, "{xyz}: {r:?}");
        assert!(
            ((fl - fd) / fd).abs() <= 1e-12,
            "{xyz}: Boys maximum differs"
        );
        assert!(
            r.max_col_dist <= 1e-8,
            "{xyz}: columns at a different maximum"
        );
        assert!(
            r.energies.iter().all(|&(_, de, k)| de <= 1e-10 && k),
            "{xyz}: {r:?}"
        );
        assert!(loc.stats.purification_iterations > 0 && loc.stats.ns_iterations_valence > 0);
        assert!(loc.stats.heap_pops > 0 && loc.stats.blocks_dropped == 0);
    }
}

/// T-F (screened Boys alone), on C8's canonical occupieds: boys_tau = 0 is
/// BITWISE `boys_localize` (same d); boys_tau = 1e-8 screens pairs, reaches
/// the same functional to 1e-12 relative and the same columns to 1e-8; a
/// coarse screen (1e-2) is seen (functional lower by > 1e-10 relative).
#[test]
fn screened_boys_is_boys_in_the_trivial_limit_and_screens_safely() {
    use crate::boys::boys_localize;
    let su = setup("alkane_8.xyz");
    let c = su.rhf.mos_r().slice(ndarray::s![.., ..su.nocc]).to_owned();
    let dip = ferric_integrals::oneelectron::dipole(&su.obs, [0.0; 3]).unwrap();
    let d0: [Array2<f64>; 3] = std::array::from_fn(|x| c.t().dot(&dip[x].dot(&c)));
    let reference = boys_localize(&c, &dip, 200);
    let run = |tau: f64| {
        let (mut cc, mut d) = (c.clone(), d0.clone());
        let r = boys_screened(&mut cc, &mut d, tau, 200);
        (cc, r)
    };
    let (c0, (_, ev0, sc0, conv0)) = run(0.0);
    assert!(conv0 && sc0 == 0 && ev0 > 0);
    assert!(
        c0.iter()
            .zip(reference.c_loc.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "boys_tau = 0 is not bitwise boys_localize"
    );
    let f_ref = boys_functional(&su, &reference.c_loc);
    let (c8, (_, _, sc8, conv8)) = run(1e-8);
    let f8 = boys_functional(&su, &c8);
    let dist = max_column_distance(&c8, &reference.c_loc);
    eprintln!(
        "T-F C8 occ: screened {sc8} pairs at 1e-8, functional rel {:.2e}, columns {dist:.2e}",
        ((f8 - f_ref) / f_ref).abs()
    );
    assert!(conv8 && sc8 > 0, "the 1e-8 screen skipped nothing");
    assert!(((f8 - f_ref) / f_ref).abs() <= 1e-12);
    assert!(dist <= 1e-8);
    let (c2, _) = run(1e-2);
    let f2 = boys_functional(&su, &c2);
    eprintln!(
        "T-F C8 occ: functional rel at 1e-2 {:.2e}",
        (f_ref - f2) / f_ref
    );
    assert!((f_ref - f2) / f_ref > 1e-10, "a coarse screen is not seen");
}

/// The trivial-limit configuration with one deliberate defect.
fn mutated(su: &Su, knobs: Knobs) -> LocalVvhv {
    build_vvhv_local_with(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.rhf,
        &LocalVvhvConfig::trivial(),
        knobs,
    )
    .unwrap()
}

/// MUTATION (c1, design T-B′): the occupied density projector truncated at
/// 1e-2 (everything else exact) must fail the anchor: the span leaves the
/// virtual space (> 1e-8) and the eps = 0 energy moves by > 1e-8, C8. (At
/// 1e-4 nothing is dropped on C8: every D block is inside D's 1e-4 reach of
/// ~30 Bohr, step 0; the C8 chain is ~17 Bohr long.) A truncation of a
/// basis-choice step leaves both unchanged (T-B).
#[test]
fn mutation_truncated_density_projector_fails_the_anchor() {
    let su = setup("alkane_8.xyz");
    let (d, dh) = dense(&su);
    let loc = mutated(
        &su,
        Knobs {
            density_tau: Some(1e-2),
            ..Knobs::default()
        },
    );
    let r = compare_built(&su, &d, &dh, &loc, &[]);
    let (e_d, _) = energy_span(&su, &d);
    let (e_l, _) = energy_span(&su, &loc.vv);
    eprintln!(
        "MUT density C8: span {:.2e} orth {:.2e}, eps=0 |dE| {:.2e}, dropped {}",
        r.dev_span,
        r.dev_orth,
        (e_l - e_d).abs(),
        loc.stats.blocks_dropped
    );
    assert!(loc.stats.blocks_dropped > 0, "the mutation dropped nothing");
    assert!(!r.passes() && r.dev_span > 1e-8, "{r:?}");
    assert!((e_l - e_d).abs() > 1e-8);
}

/// MUTATION (c2): every Newton-Schulz iteration stopped one step before
/// convergence must fail the anchor (orthonormality above 1e-10), C8.
#[test]
fn mutation_newton_schulz_stopped_early_fails_the_anchor() {
    let su = setup("alkane_8.xyz");
    let (d, dh) = dense(&su);
    let loc = mutated(
        &su,
        Knobs {
            ns_stop_early: true,
            ..Knobs::default()
        },
    );
    let r = compare_built(&su, &d, &dh, &loc, &[]);
    eprintln!(
        "MUT NS-early C8: orth {:.2e} span {:.2e}, column distance {:.2e}",
        r.dev_orth, r.dev_span, r.max_col_dist
    );
    assert!(!r.passes() && r.dev_orth > 1e-10, "{r:?}");
}

/// MUTATION (c3): the pivot tie rule reversed (highest index within the
/// PIVOT_TIE_RTOL band) must fail the anchor on water, whose two hydrogens'
/// candidates tie exactly: a different hard-virtual pivot sequence.
#[test]
fn mutation_reversed_tie_rule_fails_the_anchor() {
    let su = setup("water.xyz");
    let (d, dh) = dense(&su);
    let loc = mutated(
        &su,
        Knobs {
            tie: TieRule::Highest,
            ..Knobs::default()
        },
    );
    let r = compare_built(&su, &d, &dh, &loc, &[]);
    eprintln!(
        "MUT tie water: pivots dense {dh:?} local {:?}; same {}",
        loc.hard_aos, r.same_pivots
    );
    assert!(!r.same_pivots && !r.passes(), "{r:?}");
}

/// (d) production thresholds, MEASURED (not a pass/fail gate; `--ignored
/// --nocapture`): per alkane in FERRIC_LVVHV_SIZES (default 8,16,24,32) and
/// per local configuration, the eps = 1e-4 energy shift of the local build
/// against the dense build (production maps, strict CG), as a fraction of
/// the eps truncation error |E_dense(1e-4) − E_canonical|, plus the
/// construction check and the counters. Configs: "default" =
/// `LocalVvhvConfig::default()`; the others vary one knob
/// (FERRIC_LVVHV_CONFIGS = a comma list of names, or "none", selects).
#[test]
#[ignore]
fn production_threshold_energy_shift() {
    let sizes: Vec<usize> = std::env::var("FERRIC_LVVHV_SIZES")
        .unwrap_or_else(|_| "8,16,24,32".into())
        .split(',')
        .map(|x| x.trim().parse().unwrap())
        .collect();
    let d = LocalVvhvConfig::default();
    let configs: Vec<(&str, LocalVvhvConfig)> = vec![
        ("default", d.clone()),
        (
            "tau1e-8",
            LocalVvhvConfig {
                tau: 1e-8,
                ..d.clone()
            },
        ),
        (
            "loose1e-4",
            LocalVvhvConfig {
                tau_loose: 1e-4,
                ..d.clone()
            },
        ),
        (
            "loose1e-8",
            LocalVvhvConfig {
                tau_loose: 1e-8,
                ..d.clone()
            },
        ),
        (
            "radius15",
            LocalVvhvConfig {
                solve_radius_bohr: Some(15.0),
                ..d.clone()
            },
        ),
        (
            "radius_global",
            LocalVvhvConfig {
                solve_radius_bohr: None,
                ..d.clone()
            },
        ),
        (
            "boys1e-6",
            LocalVvhvConfig {
                boys_tau: 1e-6,
                ..d.clone()
            },
        ),
    ];
    for nc in sizes {
        let su = setup(&format!("alkane_{nc}.xyz"));
        let t0 = std::time::Instant::now();
        let (dv, _) = dense(&su);
        let t_dense = t0.elapsed().as_secs_f64();
        let cfg = AmplitudeLmp2Config {
            eps: 1e-4,
            frozen_core: 0,
            cg_rtol: Some(LMP2_CG_RTOL_STRICT),
            pair_gate_cal: Some(0.7),
            compute_reference: true,
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
        let run = |vv: &VvHv, reference: bool| {
            let c = AmplitudeLmp2Config {
                compute_reference: reference,
                ..cfg.clone()
            };
            amplitude_lmp2_direct_with_virtuals(
                &su.mol,
                &su.obs,
                &su.dfbs,
                Operator::coulomb(),
                &su.rhf,
                &c,
                &dcfg,
                vv,
            )
            .unwrap()
            .0
        };
        let rd = run(&dv, true);
        let trunc = (rd.e_corr - rd.e_corr_canonical_ri).abs();
        println!(
            "PROD C{nc} dense: E(1e-4) {:.10} canonical {:.10} eps-error {trunc:.3e} t_build {t_dense:.3}",
            rd.e_corr, rd.e_corr_canonical_ri
        );
        let only = std::env::var("FERRIC_LVVHV_CONFIGS").ok();
        for (name, lc) in &configs {
            if only
                .as_deref()
                .is_some_and(|o| !o.split(',').any(|x| x == *name))
            {
                continue;
            }
            let t0 = std::time::Instant::now();
            let loc = build_vvhv_local(&su.mol, &su.obs, &su.obs_bs, &su.rhf, lc).unwrap();
            let t_loc = t0.elapsed().as_secs_f64();
            let (o, sp) = check_vvhv(&su.obs, &su.rhf, su.nocc, &loc.vv.c_vloc);
            let lk = occupied_leak(&su, &loc.vv.c_vloc);
            let rl = run(&loc.vv, false);
            let de = (rl.e_corr - rd.e_corr).abs();
            println!(
                "PROD C{nc} {name:13} |dE| {de:.3e} ratio {:.3e} orth {o:.2e} span {sp:.2e} \
                 leak {lk:.2e} keep {:.5}/{:.5} t_build {t_loc:.3} stats {:?}",
                de / trunc,
                rl.keep_fraction,
                rd.keep_fraction,
                loc.stats
            );
        }
        // the guard's effect at production thresholds and at tau = 1e-6
        for tau in [1e-9, 1e-6] {
            let lc = LocalVvhvConfig { tau, ..d.clone() };
            for skip in [false, true] {
                let knobs = Knobs {
                    skip_guard: skip,
                    ..Knobs::default()
                };
                let loc = build_vvhv_local_with(&su.mol, &su.obs, &su.obs_bs, &su.rhf, &lc, knobs)
                    .unwrap();
                let (o, sp) = check_vvhv(&su.obs, &su.rhf, su.nocc, &loc.vv.c_vloc);
                let lk = occupied_leak(&su, &loc.vv.c_vloc);
                println!(
                    "GUARD C{nc} tau {tau:.0e} skip_guard {skip}: orth {o:.3e} span {sp:.3e} \
                     leak {lk:.3e}"
                );
            }
        }
    }
}

/// MUTATION (c4): the guard's job is to remove what the truncated steps
/// leave. With every threshold 0 there is nothing to remove (the guard is
/// then the identity, which T-A checks), so the defect is injected: an
/// occupied component of size 1e-3 added to every virtual column before the
/// guard. WITH the guard the anchor still passes on C8 (the re-projection
/// removes it exactly); with the guard SKIPPED it fails (occupied leak >
/// 1e-8). NOTE check_vvhv's span measure is BLIND to this defect (U =
/// C_canᵀ S C_v annihilates an occupied component; it shows only as δ² in
/// the orthonormality), which is why the anchor also bounds the leak.
#[test]
fn mutation_skipped_guard_fails_the_anchor() {
    let su = setup("alkane_8.xyz");
    let (d, dh) = dense(&su);
    let leak = |skip_guard| {
        let loc = mutated(
            &su,
            Knobs {
                occ_leak: Some(1e-3),
                skip_guard,
                ..Knobs::default()
            },
        );
        compare_built(&su, &d, &dh, &loc, &[0.0])
    };
    let guarded = leak(false);
    let skipped = leak(true);
    eprintln!(
        "MUT guard C8 (1e-3 occupied leak): guarded leak {:.2e} orth {:.2e} span {:.2e} cols \
         {:.2e} pass {}; skipped leak {:.2e} orth {:.2e} span {:.2e} cols {:.2e} pass {}",
        guarded.occ_leak,
        guarded.dev_orth,
        guarded.dev_span,
        guarded.max_col_dist,
        guarded.passes(),
        skipped.occ_leak,
        skipped.dev_orth,
        skipped.dev_span,
        skipped.max_col_dist,
        skipped.passes()
    );
    assert!(
        guarded.passes(),
        "the guard did not remove the leak: {guarded:?}"
    );
    assert!(!skipped.passes() && skipped.occ_leak > 1e-8, "{skipped:?}");
}
