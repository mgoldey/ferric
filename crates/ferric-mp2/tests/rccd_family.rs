//! Anchors for SOSEX and rCCD (RPAx).
//!
//! The identities behind these bars are PROVED against a full spin-orbital
//! antisymmetrized Riccati oracle in `wiki/notebooks/15-sosex-rccd.ipynb`,
//! so a failure here is an implementation bug by construction.
//!
//! EXACTNESS ANCHOR FIRST (repo Experimental Protocol): the trivial limit
//! of SOSEX is the first-order amplitude `T⁽¹⁾ = −B/D`, where the SOSEX
//! functional must reproduce full MP2 *with* exchange while the drCCD ring
//! functional reproduces direct-only MP2. That is a SHARP factor identity —
//! it holds exactly, not to a tolerance — and it pins the `−½` weight and
//! the `(ib|ja)` index placement, the two things most likely to be wrong.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rccd_family::{
    first_order_amplitude, localized_problem_and_denominators, rccd, sosex, sosex_energy,
    RccdConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    obs_bs: basis::BasisSet,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
}

fn setup(xyz: &str, obs_name: &str, aux_name: &str) -> Setup {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled(obs_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { energy_conv: 1e-10, ..Default::default() },
    )
    .unwrap();
    Setup { mol, obs, obs_bs, dfbs, rhf }
}

/// Independent-construction MP2 energies from the SAME localized (ia|jb)
/// and denominators the SOSEX functional sees, in the closed-shell spatial
/// form:
///   direct-only:  E = Σ_iajb (ia|jb)·[−2(ia|jb)/D]   (the drCCD ring limit)
///   with exchange: E = Σ_iajb [2(ia|jb) − (ib|ja)]·[−(ia|jb)/D]  (full MP2)
/// Written out longhand here on purpose: this must NOT reuse the module's
/// own contraction helpers, or the anchor would be checking a function
/// against itself.
fn mp2_limits_longhand(
    j_dense: &ndarray::Array2<f64>,
    d2: &ndarray::Array2<f64>,
    no: usize,
    nv: usize,
) -> (f64, f64) {
    let idx = |i: usize, a: usize| i * nv + a;
    let mut e_direct = 0.0;
    let mut e_full = 0.0;
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    let iajb = j_dense[(idx(i, a), idx(j, b))];
                    let ibja = j_dense[(idx(i, b), idx(j, a))];
                    let d = d2[(idx(i, a), idx(j, b))];
                    e_direct += iajb * (-2.0 * iajb / d);
                    e_full += (2.0 * iajb - ibja) * (-iajb / d);
                }
            }
        }
    }
    (e_direct, e_full)
}

/// THE EXACTNESS ANCHOR. On the first-order amplitude built from the drCCD
/// kernel `B = 2(ia|jb)`:
///   - the drCCD ring functional `½ Σ B T⁽¹⁾` must give direct-only MP2
///   - the SOSEX functional `Σ [(ia|jb) − ½(ib|ja)] T⁽¹⁾` must give FULL MP2
/// Both to machine precision. This is the sharp factor check of notebook §3
/// and it is what pins the `−½` and the `(ib|ja)` axis permutation.
#[test]
fn sosex_functional_reproduces_full_mp2_in_the_first_order_limit() {
    for (xyz, obs_name, aux, fc) in [
        ("water.xyz", "6-31g", "cc-pvdz-ri", 1),
        ("h2.xyz", "sto-3g", "sto-3g", 0),
    ] {
        let su = setup(xyz, obs_name, aux);
        let cfg = RccdConfig { frozen_core: fc, ..Default::default() };
        let (lp, d2) = localized_problem_and_denominators(
            &su.mol,
            &su.obs,
            &su.obs_bs,
            &su.dfbs,
            Operator::coulomb(),
            &su.rhf,
            &cfg,
        )
        .unwrap();
        let (no, nv) = (lp.no, lp.nv);

        let b2 = lp.j_dense.mapv(|x| 2.0 * x);
        let t1 = first_order_amplitude(&b2, &d2);

        let ring = 0.5 * b2.iter().zip(t1.iter()).map(|(b, t)| b * t).sum::<f64>();
        let sos = sosex_energy(&lp.j_dense, &t1, no, nv);
        let (want_direct, want_full) = mp2_limits_longhand(&lp.j_dense, &d2, no, nv);

        eprintln!(
            "{xyz}/{obs_name}: ring={ring:.12} (want {want_direct:.12}, d={:+.2e})  \
             sosex={sos:.12} (want {want_full:.12}, d={:+.2e})",
            ring - want_direct,
            sos - want_full
        );
        assert!(
            (ring - want_direct).abs() < 1e-12,
            "drCCD ring functional is not direct-only MP2 at first order: {:+.3e}",
            ring - want_direct
        );
        assert!(
            (sos - want_full).abs() < 1e-12,
            "SOSEX functional is not full MP2 at first order: {:+.3e}",
            sos - want_full
        );
        // The two MUST differ, or the test is vacuous — an exchange term
        // that silently evaluated to zero would pass both asserts above.
        assert!(
            (want_full - want_direct).abs() > 1e-6,
            "vacuous anchor: direct and full MP2 coincide on {xyz}"
        );
    }
}

/// A TEST YOU HAVE NEVER SEEN FAIL IS AN ASSUMPTION: the anchor above must
/// FAIL under a mutation that can actually change the answer.
///
/// NOTE on a mutation that was TRIED AND REJECTED as unreachable: swapping
/// the OCCUPIED labels `(i,a,j,b)->(j,a,i,b)` instead of the virtual ones
/// is NOT a mutation at all. `(ia|jb)` carries the chemist compound
/// symmetry `J[(ia),(jb)] = J[(jb),(ia)]`, and composing the occupied swap
/// with the virtual swap IS that transpose — so the two permutations are
/// identically equal for any valid integral tensor. Asserting they differ
/// is asserting `0 != 0`; it fails on correct code and measures nothing.
/// The reachable mutations are the exchange WEIGHT and dropping the
/// exchange term outright, both checked below.
#[test]
fn wrong_exchange_weight_or_axis_breaks_the_mp2_limit() {
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");
    let cfg = RccdConfig { frozen_core: 1, ..Default::default() };
    let (lp, d2) = localized_problem_and_denominators(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
    )
    .unwrap();
    let (no, nv) = (lp.no, lp.nv);
    let b2 = lp.j_dense.mapv(|x| 2.0 * x);
    let t1 = first_order_amplitude(&b2, &d2);
    let (_, want_full) = mp2_limits_longhand(&lp.j_dense, &d2, no, nv);

    let j4 = lp
        .j_dense
        .clone()
        .into_shape_with_order((no, nv, no, nv))
        .unwrap();
    let t4 = t1.into_shape_with_order((no, nv, no, nv)).unwrap();
    let contract = |kernel: &ndarray::Array4<f64>| -> f64 {
        kernel.iter().zip(t4.iter()).map(|(k, t)| k * t).sum()
    };

    let swapped = j4.view().permuted_axes([0, 3, 2, 1]).to_owned();
    // mutation 1: weight −1 instead of −½
    let bad_weight = contract(&(&j4 - &swapped));
    // mutation 2: no exchange term at all (the direct-only functional)
    let no_exchange = contract(&j4);

    eprintln!(
        "mutations: weight-1 -> {bad_weight:.10}, no-exchange -> {no_exchange:.10} \
         (want-full {want_full:.10})"
    );
    assert!(
        (bad_weight - want_full).abs() > 1e-6,
        "MUTATION SURVIVED: exchange weight −1 also reproduces full MP2, so the \
         anchor does not pin the −½"
    );
    assert!(
        (no_exchange - want_full).abs() > 1e-6,
        "MUTATION SURVIVED: dropping the exchange term also reproduces full MP2, \
         so the anchor does not pin the exchange contribution at all"
    );
}


/// Converged-amplitude references from an INDEPENDENT construction: the
/// spin-orbital antisymmetrized Riccati oracle of notebook 15, re-run at
/// ferric's OWN testdata geometries with EXACT (non-RI) integrals.
///
/// Regenerate with `scripts/gen_rccd_oracle_refs.py`. The notebook's own
/// §9 table uses slightly different geometries (H2 at 0.74 Å vs ferric's
/// 0.7414 Å, water at a different optimized structure), so its literals are
/// NOT directly assertable here — that mismatch is why these are generated
/// at ferric's geometry rather than hand-copied.
///
/// Bars are the RI FLOOR, not machine precision: ferric fits (ia|jb) in an
/// auxiliary basis while the oracle uses exact ERIs, so the residual is a
/// real basis-incompleteness difference (MEASURED 1.2e-6..3.6e-5 across
/// these systems), not solver error. The dRPA lane's own exact-vs-RI bar is
/// the same ~1e-4 scale for the same reason.
mod oracle_refs {
    /// (drCCD, SOSEX, E_S, E_T) at ferric's geometry, exact ERIs.
    pub const H2_STO3G: (f64, f64, f64, f64) = (
        -0.020_675_730_6,
        -0.010_337_865_3,
        -0.005_773_237_2,
        -0.007_761_408_9,
    );
    pub const WATER_631G_FC1: (f64, f64, f64, f64) = (
        -0.137_127_740_1,
        -0.085_066_226_6,
        -0.081_404_825_9,
        -0.053_910_875_2,
    );
    /// Measured RI floor for these systems/aux basis (see module docs).
    pub const RI_BAR: f64 = 1e-4;
}

/// SOSEX and rCCD on the CONVERGED amplitudes vs the exact-ERI oracle.
///
/// A REAL auxiliary basis is used deliberately. Fitting STO-3G in STO-3G
/// (which the dRPA lane's H2 test does, legitimately, because its notebook
/// oracle used the same fit) carries a ~40% RI error on these energies —
/// MEASURED here during the port: H2 drCCD −0.01261 with STO-3G aux vs
/// −0.02067 exact. An absolute comparison against an exact-ERI reference is
/// only meaningful once the fitting basis is adequate.
#[test]
fn sosex_and_rccd_match_the_exact_eri_oracle() {
    for (xyz, obs_name, aux, fc, want) in [
        ("h2.xyz", "sto-3g", "cc-pvdz-ri", 0, oracle_refs::H2_STO3G),
        ("water.xyz", "6-31g", "cc-pvdz-ri", 1, oracle_refs::WATER_631G_FC1),
    ] {
        let su = setup(xyz, obs_name, aux);
        let cfg = RccdConfig { frozen_core: fc, ..Default::default() };
        let s = sosex(
            &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        )
        .unwrap();
        let r = rccd(
            &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        )
        .unwrap();
        let (w_drccd, w_sosex, w_s, w_t) = want;
        eprintln!(
            "{xyz}/{obs_name}/{aux}: drCCD={:.10} (d={:+.2e}) SOSEX={:.10} (d={:+.2e}) \
             E_S={:.10} (d={:+.2e}) E_T={:.10} (d={:+.2e}) rCCD={:.10}",
            s.e_drccd,
            s.e_drccd - w_drccd,
            s.e_sosex,
            s.e_sosex - w_sosex,
            r.singlet.e_corr,
            r.singlet.e_corr - w_s,
            r.triplet.e_corr,
            r.triplet.e_corr - w_t,
            r.e_corr
        );
        assert!(s.converged && r.singlet.converged && r.triplet.converged);
        let bar = oracle_refs::RI_BAR;
        assert!((s.e_drccd - w_drccd).abs() < bar, "drCCD off oracle on {xyz}");
        assert!((s.e_sosex - w_sosex).abs() < bar, "SOSEX off oracle on {xyz}");
        assert!((r.singlet.e_corr - w_s).abs() < bar, "E_S off oracle on {xyz}");
        assert!((r.triplet.e_corr - w_t).abs() < bar, "E_T off oracle on {xyz}");
        assert!((r.e_corr - (w_s + w_t)).abs() < 2.0 * bar, "E_rCCD off oracle on {xyz}");
    }
}

/// SOSEX is a DIFFERENT number from drCCD, and rCCD is different from both.
/// Guards against a wiring slip where one energy is silently returned for
/// another — every assert above would still pass if all three coincided.
#[test]
fn the_three_energies_are_actually_distinct() {
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");
    let cfg = RccdConfig { frozen_core: 1, ..Default::default() };
    let s = sosex(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
    )
    .unwrap();
    let r = rccd(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
    )
    .unwrap();
    eprintln!(
        "water: drCCD={:.10} SOSEX={:.10} rCCD={:.10}",
        s.e_drccd, s.e_sosex, r.e_corr
    );
    assert!((s.e_drccd - s.e_sosex).abs() > 1e-3);
    assert!((s.e_drccd - r.e_corr).abs() > 1e-4);
    assert!((s.e_sosex - r.e_corr).abs() > 1e-3);
}

/// The channel weight is 1:1, NOT the naive singlet/triplet multiplicity
/// 1:3. On a system with a NONZERO triplet channel the two differ, so this
/// pins the choice rather than restating it.
#[test]
fn rccd_channel_weight_is_one_to_one_not_one_to_three() {
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");
    let cfg = RccdConfig { frozen_core: 1, ..Default::default() };
    let r = rccd(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
    )
    .unwrap();
    let one_to_three = r.singlet.e_corr + 3.0 * r.triplet.e_corr;
    eprintln!(
        "water rCCD: E_S={:.10} E_T={:.10} -> 1:1 {:.10} vs 1:3 {:.10}",
        r.singlet.e_corr, r.triplet.e_corr, r.e_corr, one_to_three
    );
    assert!(
        r.triplet.e_corr.abs() > 1e-6,
        "vacuous: water's triplet channel is ~zero, so the weights coincide"
    );
    assert!((r.e_corr - (r.singlet.e_corr + r.triplet.e_corr)).abs() < 1e-15);
    assert!(
        (r.e_corr - one_to_three).abs() > 1e-6,
        "1:1 and 1:3 weights coincide here — this test cannot pin the choice"
    );
}
