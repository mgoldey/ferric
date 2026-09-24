//! Stage 8b: Gamma-point local MP2 (`ferric_pbc::lmp2::gamma_lmp2`), ported
//! from `reference/pbc/pbc_lmp2.py` (FINDINGS "Iteration 5 (Python, Gamma
//! LMP2)"). Pinned numbers are that iteration's (`run_lmp2_anchor.py`,
//! `run_lmp2_sweep.py`, `test_prototype.py`).
//!
//! System: primitive cubic a0 = 7 Bohr, one tilted H2 (R = 1.4) per cell,
//! PySCF 6-31G (s only), cc-pvdz-ri aux, 1x1x4 needle supercells built
//! DIRECTLY as a `Cell` (no primitive-cell fold: the explicit RS-GDF build is
//! affordable at nao = 16, naux = 112). "Wrapped" = the last atom moved by
//! −a_sc(z), so one molecule straddles the supercell boundary.
//!
//! # Why 1x1x4 and not 1x1x2 or 2x2x2
//!
//! With n = 2 cells along every axis each raw Cartesian distance already IS a
//! minimum image, so a minimum-image bug is invisible (prototype: the raw
//! pair-cutoff mutant passes the anchor exactly on 2x2x1). Every distance test
//! below therefore runs on n = 4 along z and ALSO asserts that the raw
//! mutant fails (reachable failure, not a vacuous pass).
//!
//! # Artifact hypotheses (stated before measuring)
//!
//! * Anchor (eps = 0 ≡ canonical `gamma_mp2`): if the periodic VV-HV span is
//!   wrong the anchor moves (dropping one hard virtual: prototype +3.4e-3);
//!   if localisation is non-periodic the anchor CANNOT see it (unitary
//!   invariance) — translation equivalence is the guard, and molecular Boys
//!   must fail it while still passing the anchor (pinned blind spot).
//! * Trivial-radius domain fit ≡ global B: a non-minimum-image domain drops
//!   aux across the boundary (prototype 5.8e-5) — asserted to fail.
//! * The eps-gate "all pairs kept" count below onset is PHYSICS (uniform
//!   G → 0 field), not arithmetic: the farthest pair's integral block matches
//!   the uniform-field formula (prototype rel. dev. 4.1e-2 at N = 4), and a
//!   minimum-image distance cutoff on the same system DOES drop pairs.
//!
//! # Byte-identity of the ferric-mp2 seams (run with this file)
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-mp2 --lib \
//!     lmp2_amplitude::tests ragged::tests
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-mp2 \
//!     --test lmp2_amplitude --test lmp2_direct --test drpa_amplitude \
//!     --test drpa_direct --test ragged_ring_identity
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-cc \
//!     --test linlccd_amplitude --test linlccd_direct
//! ```

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_mp2::boys::boys_localize;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::lmp2::{
    centroid_distances, gamma_lmp2, gamma_lmp2_with_spaces, gamma_localized_spaces,
    mp2_closed_form_local, needle_axis, resta_matrix_at, resta_operator, uniform_coupling,
    uniform_head, GammaEpsGate, GammaLmp2Config, GammaLmp2Inputs, GammaLmp2Result,
    GammaLocalSpaces, GammaPairIntegrals, MinImage, PeriodicDistance, UniformHead,
};
use ferric_pbc::mp2::{b_ov_from_ao_b, gamma_mp2, GammaMp2Config, GammaMp2Integrals};
use ferric_pbc::rsgdf::{PeriodicFitParts, RsGdf, RsGdfConfig};
use ferric_scf::result::ScfResult;
use ndarray::{s, Array2};
use std::f64::consts::PI;
use std::sync::OnceLock;

const A0: f64 = 7.0;
const NCELL: usize = 4;
const AMPLE: usize = 1 << 30;

// --- FINDINGS Iteration 5, 1x1x4 (straight): run_lmp2_anchor / run_lmp2_sweep.
const PIN_E_MP2_CANON_1X1X4: f64 = -7.437860250848e-2;
/// dE = E_LMP2 − E_canonical (Ha, 3 significant figures) and partners per
/// molecule, eps-mask rows then minimum-image pair-cutoff rows (eps = 0).
const PIN_EPS_ROWS: [(f64, f64, usize); 5] = [
    (1e-2, 1.05e-3, 1),
    (3e-3, 6.88e-4, 4),
    (1e-3, 1.87e-4, 4),
    (3e-4, 3.18e-5, 4),
    (1e-4, 2.00e-5, 4),
];
const PIN_RC_ROWS: [(f64, f64, usize); 3] = [(3.5, 8.98e-4, 1), (10.5, 2.73e-4, 3), (17.5, 0.0, 4)];

struct System {
    cell: Cell,
    obs_bs: BasisSet,
    min_bs: BasisSet,
    obs: PreparedBasis,
    gdf: RsGdf,
    parts: PeriodicFitParts,
    rhf: ScfResult,
}

impl System {
    fn inputs(&self) -> GammaLmp2Inputs<'_> {
        GammaLmp2Inputs {
            obs: &self.obs,
            obs_bs: &self.obs_bs,
            minimal_bs: &self.min_bs,
            gdf: &self.gdf,
            fit: Some(&self.parts),
        }
    }

    fn canonical(&self) -> f64 {
        let cfg = GammaMp2Config {
            budget_bytes: Some(AMPLE),
            ..GammaMp2Config::shifted(ExxDiv::Ewald)
        };
        gamma_mp2(
            &self.cell,
            &self.rhf,
            GammaMp2Integrals::RsGdf(&self.gdf),
            &cfg,
        )
        .expect("canonical gamma MP2")
        .mp2_corr
    }

    fn spaces(&self, cfg: &GammaLmp2Config) -> GammaLocalSpaces {
        gamma_localized_spaces(&self.cell, &self.rhf, &self.inputs(), cfg).expect("spaces")
    }

    fn run(&self, cfg: &GammaLmp2Config, spaces: &GammaLocalSpaces) -> GammaLmp2Result {
        gamma_lmp2_with_spaces(&self.cell, &self.rhf, &self.inputs(), cfg, spaces)
            .expect("gamma LMP2")
    }

    fn nao_prim(&self) -> usize {
        self.obs.nbasis() / NCELL
    }
}

fn cfg(eps: f64) -> GammaLmp2Config {
    GammaLmp2Config {
        budget_bytes: Some(AMPLE),
        compute_reference: false,
        ..GammaLmp2Config::shifted(ExxDiv::Ewald, eps)
    }
}

/// [`cfg`] with the opt-in A-drop gate (restored q = 0 head).
fn cfg_head(eps: f64) -> GammaLmp2Config {
    GammaLmp2Config {
        eps_gate: GammaEpsGate::UniformHeadRestored,
        ..cfg(eps)
    }
}

/// `pbc_lmp2`/`run_lmp2_anchor` H2: (1.0, 1.2, 1.1) and that + 1.4·û,
/// û ∝ (0.5, 0.6, 1.2).
fn h2_prim() -> [[f64; 3]; 2] {
    let u: [f64; 3] = [0.5, 0.6, 1.2];
    let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let r0 = [1.0, 1.2, 1.1];
    [
        r0,
        [
            r0[0] + 1.4 * u[0] / n,
            r0[1] + 1.4 * u[1] / n,
            r0[2] + 1.4 * u[2] / n,
        ],
    ]
}

fn needle_cell(n: usize, wrapped: bool) -> Cell {
    let prim = h2_prim();
    let mut pos = Vec::new();
    for c in 0..n {
        for r in prim {
            pos.push([r[0], r[1], r[2] + A0 * c as f64]);
        }
    }
    if wrapped {
        let last = pos.len() - 1;
        pos[last][2] -= A0 * n as f64;
    }
    let lat = [[A0, 0.0, 0.0], [0.0, A0, 0.0], [0.0, 0.0, A0 * n as f64]];
    Cell::new(hydrogens(&pos), lat).expect("needle cell")
}

fn build(n: usize, wrapped: bool) -> System {
    let t0 = std::time::Instant::now();
    let cell = needle_cell(n, wrapped);
    let obs_bs = pyscf_631g_h();
    let min_bs = pyscf_sto3g_h();
    let obs = prep_for(&cell, &obs_bs);
    let hc = periodic_hcore(&cell, &obs, &PeriodicHcoreConfig::with_omega(0.8)).expect("hcore");
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let gcfg = RsGdfConfig {
        omega: 1.0,
        exxdiv: ExxDiv::Ewald,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    let (gdf, parts) =
        RsGdf::build_with_fit_parts(&cell, &obs, &aux, &hc.s, &gcfg).expect("RS-GDF");
    let rhf = gamma_rhf_jk(
        &cell,
        &obs,
        &hc,
        Box::new(gdf.j_builder()),
        Box::new(gdf.k_builder()),
    );
    eprintln!(
        "1x1x{n}{}: nao {} naux {} (kept {}), E_HF {:.12} [{:.1} s]",
        if wrapped { " wrapped" } else { "" },
        obs.nbasis(),
        gdf.stats().naux,
        gdf.stats().naux_kept,
        rhf.energy,
        t0.elapsed().as_secs_f64()
    );
    System {
        cell,
        obs_bs,
        min_bs,
        obs,
        gdf,
        parts,
        rhf,
    }
}

fn wrapped4() -> &'static System {
    static S: OnceLock<System> = OnceLock::new();
    S.get_or_init(|| build(NCELL, true))
}

fn straight4() -> &'static System {
    static S: OnceLock<System> = OnceLock::new();
    S.get_or_init(|| build(NCELL, false))
}

fn max_min_image_distance(sys: &System, sp: &GammaLocalSpaces) -> f64 {
    centroid_distances(&sys.cell, &sp.occ_centers, PeriodicDistance::MinimumImage)
        .iter()
        .fold(0.0_f64, |m, &d| m.max(d))
}

// ===========================================================================
// (a) Exactness anchor: eps = 0 ≡ canonical Gamma MP2 (same B, shifted).
// ===========================================================================

#[test]
fn eps0_matches_canonical_gamma_mp2_and_detects_construction_and_min_image_mutants() {
    let sys = wrapped4();
    let e_can = sys.canonical();
    let c0 = cfg(0.0);
    let sp = sys.spaces(&c0);
    eprintln!(
        "VV {} HV {}; orth {:.1e} span {:.1e}; Berghold f {:.6} -> {:.6} in {} sweeps, |grad| {:.1e}",
        sp.n_valence,
        sp.n_hard,
        sp.vvhv_dev_orth,
        sp.vvhv_dev_span,
        sp.loc_occ.f_initial,
        sp.loc_occ.f_final,
        sp.loc_occ.sweeps,
        sp.loc_occ.max_gradient
    );
    assert_eq!((sp.no(), sp.nv(), sp.n_valence, sp.n_hard), (4, 12, 4, 8));
    assert!(sp.vvhv_dev_orth < 1e-10 && sp.vvhv_dev_span < 1e-10);
    assert!(sp.loc_occ.max_gradient < 1e-8);

    let r = sys.run(&c0, &sp);
    let d = r.e_corr - e_can;
    eprintln!(
        "ANCHOR eps=0: E {:.12e} canonical {e_can:.12e} dE {d:+.2e} (cg {})",
        r.e_corr, r.cg_iterations
    );
    assert!(d.abs() < 1e-10, "anchor dE {d:.3e}");
    assert_eq!(r.pairs_kept, 16);
    let esum: f64 = r.pair_energies.sum();
    assert!(
        (esum - r.e_corr).abs() < 1e-12,
        "pair energies sum {esum} vs {}",
        r.e_corr
    );
    // The driver's own reference path is the same canonical number.
    let with_ref = gamma_lmp2(
        &sys.cell,
        &sys.rhf,
        &sys.inputs(),
        &GammaLmp2Config {
            compute_reference: true,
            ..c0
        },
    )
    .unwrap();
    assert!((with_ref.e_corr_canonical - e_can).abs() < 1e-14);

    // M1: drop one hard virtual -> the anchor must fail (prototype +3.4e-3).
    let mut m1 = sp.clone();
    let nv = sp.nv();
    m1.c_vir = sp.c_vir.slice(s![.., ..nv - 1]).to_owned();
    m1.f_vv = sp.f_vv.slice(s![..nv - 1, ..nv - 1]).to_owned();
    m1.vir_centers.pop();
    m1.n_hard -= 1;
    let dm1 = sys.run(&c0, &m1).e_corr - e_can;
    eprintln!("M1 drop one HV: dE {dm1:+.2e} (prototype +3.4e-3)");
    assert!(
        dm1.abs() > 1e-4,
        "dropped hard virtual invisible: {dm1:.3e}"
    );

    // M2: pair cutoff at its trivial radius R* = max minimum-image distance.
    let rstar = max_min_image_distance(sys, &sp) + 1e-6;
    let cut = GammaLmp2Config {
        pair_cutoff_bohr: Some(rstar),
        ..c0
    };
    let rmi = sys.run(&cut, &sp);
    let raw = sys.run(
        &GammaLmp2Config {
            distance: PeriodicDistance::RawCartesianMutant,
            ..cut
        },
        &sp,
    );
    eprintln!(
        "M2 cutoff R* = {rstar:.4}: min-image dE {:+.2e} ({} pairs), raw dE {:+.2e} ({} pairs)",
        rmi.e_corr - e_can,
        rmi.pairs_kept,
        raw.e_corr - e_can,
        raw.pairs_kept
    );
    assert!((rmi.e_corr - e_can).abs() < 1e-10 && rmi.pairs_kept == 16 && rmi.n_pairs_cut == 0);
    assert!(
        raw.pairs_kept < 16,
        "raw distances dropped no pair: the mutant is unreachable here"
    );
    assert!(
        (raw.e_corr - e_can).abs() > 1e-5,
        "raw-distance cutoff invisible"
    );
}

// ===========================================================================
// (b) Per-pair domain fit in the periodic metric: trivial radius ≡ global B.
// ===========================================================================

#[test]
fn periodic_domain_fit_at_the_trivial_radius_is_the_global_fit() {
    let sys = wrapped4();
    let e_can = sys.canonical();
    let c0 = cfg(0.0);
    let sp = sys.spaces(&c0);
    let mi = MinImage::new(&sys.cell, PeriodicDistance::MinimumImage);
    let mi = &mi;
    let rstar = sp
        .occ_centers
        .iter()
        .flat_map(|c| {
            sys.parts
                .aux_centers
                .iter()
                .map(move |p| mi.between(*c, *p))
        })
        .fold(0.0_f64, f64::max)
        + 1e-6;
    let src = |radius: Option<f64>, metric| {
        GammaPairIntegrals::new(
            &sys.cell,
            &sp,
            &sys.gdf,
            Some(&sys.parts),
            radius,
            metric,
            Some(AMPLE),
        )
        .expect("pair integrals")
    };
    let glob = src(None, PeriodicDistance::MinimumImage);
    let dom = src(Some(rstar), PeriodicDistance::MinimumImage);
    let raw = src(Some(rstar), PeriodicDistance::RawCartesianMutant);
    let naux = sys.parts.j2.nrows();
    let (mut d_mi, mut d_raw) = (0.0_f64, 0.0_f64);
    let mut raw_dom_min = usize::MAX;
    for i in 0..sp.no() {
        for j in i..sp.no() {
            let (g, _) = glob.block(i, j).unwrap();
            let (x, nd) = dom.block(i, j).unwrap();
            let (y, nr) = raw.block(i, j).unwrap();
            assert_eq!(nd, naux, "trivial radius must cover every aux function");
            raw_dom_min = raw_dom_min.min(nr);
            d_mi = d_mi.max(max_abs_diff(&x, &g));
            d_raw = d_raw.max(max_abs_diff(&y, &g));
        }
    }
    eprintln!(
        "domain fit at R* = {rstar:.3}: min-image max|dJ| {d_mi:.1e}; raw max|dJ| {d_raw:.1e} \
         (smallest raw domain {raw_dom_min}/{naux}; prototype 3e-16 / 5.8e-5)"
    );
    assert!(
        d_mi < 1e-12,
        "trivial-radius periodic fit != global B: {d_mi:.3e}"
    );
    assert!(
        raw_dom_min < naux,
        "raw distances dropped no aux function: mutant unreachable"
    );
    assert!(
        d_raw > 1e-6,
        "non-minimum-image domain invisible: {d_raw:.3e}"
    );

    // End to end through the driver: eps = 0 with the domain fit is the anchor.
    let r = sys.run(
        &GammaLmp2Config {
            fit_radius_bohr: Some(rstar),
            ..c0
        },
        &sp,
    );
    assert_eq!(r.aux_dom_max, naux);
    assert!(
        (r.e_corr - e_can).abs() < 1e-10,
        "domain-fit anchor {:.3e}",
        r.e_corr - e_can
    );
    // A finite radius really shrinks the domains (not a vacuous knob).
    let r5 = sys.run(
        &GammaLmp2Config {
            fit_radius_bohr: Some(5.0),
            ..c0
        },
        &sp,
    );
    eprintln!(
        "fit radius 5 Bohr: aux domains mean {:.1} max {} of {naux}; dE {:+.2e} (not pinned)",
        r5.aux_dom_mean,
        r5.aux_dom_max,
        r5.e_corr - e_can
    );
    assert!(r5.aux_dom_max < naux);
}

// ===========================================================================
// (c) Translation equivalence (the guard the anchor cannot provide).
// ===========================================================================

/// Translate every column of `c` by one primitive vector along z (an exact
/// AO permutation at Gamma, cell-major AO order) and match it to the best
/// column: returns (perm, max over columns of 1 − max_j |⟨c_j|S|T c_i⟩|).
fn translation_map(sys: &System, c: &Array2<f64>) -> (Vec<usize>, f64) {
    let np = sys.nao_prim();
    let nao = c.nrows();
    let mut tc = Array2::<f64>::zeros(c.dim());
    for k in 0..nao {
        let (cell, m) = (k / np, k % np);
        let tgt = ((cell + 1) % NCELL) * np + m;
        tc.row_mut(tgt).assign(&c.row(k));
    }
    let o = c.t().dot(&sys.gdf.overlap().dot(&tc)).mapv(f64::abs);
    let mut perm = Vec::with_capacity(c.ncols());
    let mut dev = 0.0_f64;
    for col in 0..c.ncols() {
        let (arg, best) = o
            .column(col)
            .iter()
            .enumerate()
            .fold(
                (0, -1.0),
                |(ai, bv), (i, &v)| if v > bv { (i, v) } else { (ai, bv) },
            );
        perm.push(arg);
        dev = dev.max(1.0 - best);
    }
    (perm, dev)
}

#[test]
fn berghold_lmos_virtuals_and_pair_energies_are_translation_equivalent() {
    for (name, sys) in [("wrapped", wrapped4()), ("straight", straight4())] {
        let c = cfg(1e-4);
        let sp = sys.spaces(&c);
        let r = sys.run(&c, &sp);
        let (po, occ_dev) = translation_map(sys, &sp.c_occ);
        let (_, vir_dev) = translation_map(sys, &sp.c_vir);
        let mut sorted = po.clone();
        sorted.sort_unstable();
        sorted.dedup();
        let bijective = sorted.len() == po.len();
        let no = sp.no();
        let (mut pair_dev, mut spread_dev) = (0.0_f64, 0.0_f64);
        for i in 0..no {
            spread_dev = spread_dev.max((sp.occ_spreads[po[i]] - sp.occ_spreads[i]).abs());
            for j in 0..no {
                pair_dev =
                    pair_dev.max((r.pair_energies[(po[i], po[j])] - r.pair_energies[(i, j)]).abs());
            }
        }
        eprintln!(
            "{name}: occ_dev {occ_dev:.1e} vir_dev {vir_dev:.1e} pair_dev {pair_dev:.1e} \
             spread_dev {spread_dev:.1e} bijective {bijective} partners {:?}",
            r.partners
        );
        assert!(
            bijective,
            "{name}: translation does not map LMOs one-to-one"
        );
        assert!(
            occ_dev < 1e-10 && vir_dev < 1e-10,
            "{name}: {occ_dev:.2e} / {vir_dev:.2e}"
        );
        assert!(
            pair_dev < 1e-10 && spread_dev < 1e-10,
            "{name}: {pair_dev:.2e} / {spread_dev:.2e}"
        );
        assert!(
            r.partners.iter().all(|&p| p == r.partners[0]),
            "{name}: unequal partner counts"
        );

        // Start independence: a random orthogonal start reaches the same maximum.
        let cs = GammaLmp2Config {
            localization: ferric_pbc::lmp2::LocalizationConfig {
                random_start_seed: Some(3),
                ..c.localization
            },
            ..c
        };
        let sps = sys.spaces(&cs);
        let rs = sys.run(&cs, &sps);
        eprintln!(
            "{name}: Berghold f canonical start {:.12} / random start {:.12}; dE(eps 1e-4) {:+.1e}",
            sp.loc_occ.f_final,
            sps.loc_occ.f_final,
            rs.e_corr - r.e_corr
        );
        assert!((sps.loc_occ.f_final - sp.loc_occ.f_final).abs() < 1e-8);
        assert!((rs.e_corr - r.e_corr).abs() < 1e-10);
    }

    // Negative control: molecular Boys (L = 0 <mu|r|nu>, sees the boundary)
    // breaks equivalence (prototype 3.2e-4 wrapped) yet passes the eps = 0
    // anchor — the pinned blind spot of the anchor.
    let sys = wrapped4();
    let c0 = cfg(0.0);
    let sp = sys.spaces(&c0);
    let c_can = sys.rhf.mos_r().slice(s![.., ..sp.nocc_total]).to_owned();
    let dip = oneelectron::dipole(&sys.obs, [0.0; 3]).unwrap();
    let boys = boys_localize(&c_can, &dip, 200).c_loc;
    let (_, boys_dev) = translation_map(sys, &boys);
    let mut sb = sp.clone();
    sb.f_oo = boys.t().dot(&sys.rhf.fock_r().dot(&boys));
    for i in 0..sb.no() {
        sb.f_oo[(i, i)] += sb.occ_shift;
    }
    sb.c_occ = boys;
    let db = sys.run(&c0, &sb).e_corr - sys.canonical();
    eprintln!(
        "molecular Boys (MUTANT): occ_dev {boys_dev:.2e}; eps=0 anchor dE {db:+.1e} (blind spot)"
    );
    assert!(
        boys_dev > 1e-6,
        "molecular Boys stayed translation-equivalent: the guard is vacuous"
    );
    assert!(
        db.abs() < 1e-10,
        "the anchor was expected to be blind to localisation"
    );
}

// ===========================================================================
// (d) Prototype pins, straight 1x1x4.
// ===========================================================================

fn half_unit_3sig(x: f64) -> f64 {
    0.5 * 10f64.powi(x.abs().log10().floor() as i32 - 2)
}

#[test]
fn straight_1x1x4_reproduces_the_prototype_sweep_row() {
    let sys = straight4();
    let e_can = sys.canonical();
    eprintln!("E_MP2 canonical (shifted) {e_can:.12e} (prototype {PIN_E_MP2_CANON_1X1X4:.12e})");
    // Python fold (w = 0.5) vs Rust explicit RS-GDF (w = 1.0): same kernel,
    // same lindep, different truncation paths.
    assert!((e_can - PIN_E_MP2_CANON_1X1X4).abs() < 1e-8, "{e_can:.12e}");
    let c0 = cfg(0.0);
    let sp = sys.spaces(&c0);
    assert_eq!((sp.no(), sp.nv(), sp.n_valence, sp.n_hard), (4, 12, 4, 8));
    let r0 = sys.run(&c0, &sp);
    assert!((r0.e_corr - e_can).abs() < 1e-10);
    for (eps, de_pin, partners) in PIN_EPS_ROWS {
        let r = sys.run(&cfg(eps), &sp);
        let de = r.e_corr - e_can;
        eprintln!(
            "eps {eps:<7}: dE {de:+.4e} (pin {de_pin:+.3e}) partners {:?}",
            r.partners
        );
        assert!(
            (de - de_pin).abs() < half_unit_3sig(de_pin),
            "eps {eps}: dE {de:.4e}"
        );
        assert!(
            r.partners.iter().all(|&p| p == partners),
            "eps {eps}: {:?}",
            r.partners
        );
    }
    for (rc, de_pin, partners) in PIN_RC_ROWS {
        let r = sys.run(
            &GammaLmp2Config {
                pair_cutoff_bohr: Some(rc),
                ..c0
            },
            &sp,
        );
        let de = r.e_corr - e_can;
        eprintln!(
            "R_c {rc:<5}: dE {de:+.4e} (pin {de_pin:+.3e}) partners {:?}",
            r.partners
        );
        let tol = if de_pin == 0.0 {
            1e-10
        } else {
            half_unit_3sig(de_pin)
        };
        assert!((de - de_pin).abs() < tol, "R_c {rc}: dE {de:.4e}");
        assert!(
            r.partners.iter().all(|&p| p == partners),
            "R_c {rc}: {:?}",
            r.partners
        );
    }
}

// ===========================================================================
// (e) KNOWN LIMITATION (documents the Gamma uniform-field coupling) of the
//     DEFAULT raw gate, and its deliberate flip under the opt-in A-drop gate.
// ===========================================================================

#[test]
fn eps_gate_keeps_every_pair_below_the_uniform_field_onset() {
    let sys = straight4();
    let c = cfg(3e-3);
    let sp = sys.spaces(&c);
    let no = sp.no();

    // Why: far pairs carry (ia|jb) -> -(4π/Ω_sc) μ_ia,z μ_jb,z. μ from the
    // Resta matrix at the smallest b: μ_ia ≈ Im(z_ia e^{-i arg z_ii}) / b.
    let op = resta_operator(&sys.cell, &sys.obs).unwrap();
    let bz = 2.0 * PI / (A0 * NCELL as f64);
    let (co, cv) = (&sp.c_occ, &sp.c_vir);
    let mu = Array2::from_shape_fn((no, sp.nv()), |(i, a)| {
        let (ci, ca) = (co.column(i), cv.column(a));
        let (zr_ii, zi_ii) = (ci.dot(&op.re[2].dot(&ci)), ci.dot(&op.im[2].dot(&ci)));
        let (zr, zi) = (ci.dot(&op.re[2].dot(&ca)), ci.dot(&op.im[2].dot(&ca)));
        let ph = zi_ii.atan2(zr_ii);
        // Im[(zr + i zi) e^{-i ph}] = zi cos ph − zr sin ph
        (zi * ph.cos() - zr * ph.sin()) / bz
    });
    let mu_star = mu.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let omega_prim = A0 * A0 * A0;
    let nstar = 4.0 * PI * mu_star * mu_star / (omega_prim * c.eps);
    let dist = centroid_distances(&sys.cell, &sp.occ_centers, PeriodicDistance::MinimumImage);
    let (mut fi, mut fj, mut dmax) = (0, 0, -1.0);
    for i in 0..no {
        for j in 0..no {
            if dist[(i, j)] > dmax {
                (fi, fj, dmax) = (i, j, dist[(i, j)]);
            }
        }
    }
    let src = GammaPairIntegrals::new(
        &sys.cell,
        &sp,
        &sys.gdf,
        None,
        None,
        PeriodicDistance::MinimumImage,
        Some(AMPLE),
    )
    .unwrap();
    let (jf, _) = src.block(fi, fj).unwrap();
    let vol = sys.cell.volume();
    let pred = Array2::from_shape_fn(jf.dim(), |(a, b)| {
        -(4.0 * PI / vol) * mu[(fi, a)] * mu[(fj, b)]
    });
    let jmax = jf.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let rel = max_abs_diff(&jf, &pred) / jmax;
    eprintln!(
        "mu*_z {mu_star:.4}; N*(eps 3e-3) = {nstar:.2} (N = {NCELL}); farthest pair ({fi},{fj}) at \
         {dmax:.2} Bohr: max|J| {jmax:.3e}, rel |J - J_unif| {rel:.2e} (prototype 4.1e-2)"
    );
    assert!(
        rel < 0.06,
        "farthest-pair block is not the uniform-field term: {rel:.3e}"
    );
    assert!(
        nstar > NCELL as f64 + 1.0,
        "N* {nstar} no longer above N: the premise changed"
    );

    // The limitation itself: every one of the N² pairs survives the eps gate.
    let r = sys.run(&c, &sp);
    eprintln!(
        "eps 3e-3: pairs kept {}/{} partners {:?}",
        r.pairs_kept,
        no * no,
        r.partners
    );
    assert_eq!(
        r.pairs_kept,
        no * no,
        "KNOWN LIMITATION changed: see lmp2.rs module doc"
    );
    // ...while a minimum-image distance cutoff on the SAME system is local,
    // so the count above is reachable, not arithmetic.
    let rc = sys.run(
        &GammaLmp2Config {
            pair_cutoff_bohr: Some(3.5),
            ..c
        },
        &sp,
    );
    assert_eq!(rc.pairs_kept, no, "{:?}", rc.partners);

    // THE DELIBERATE FLIP (FINDINGS Iteration 5b, A-drop): with the q = 0
    // head restored the far pairs lose their uniform coupling, so the SAME
    // eps gate on the SAME system is local. Prototype: 1 partner/molecule at
    // eps 1e-3 for every N; eps 3e-3 keeps a subset of that (monotone mask),
    // and self pairs are always far above 3e-3. The raw gate above stays
    // reachable as GammaEpsGate::RawIntegral and still keeps all N².
    let rh = sys.run(
        &GammaLmp2Config {
            eps_gate: GammaEpsGate::UniformHeadRestored,
            ..c
        },
        &sp,
    );
    eprintln!(
        "A-drop eps 3e-3: pairs kept {}/{} partners {:?}",
        rh.pairs_kept,
        no * no,
        rh.partners
    );
    assert_eq!(
        rh.pairs_kept, no,
        "A-drop no longer removes the onset: {:?}",
        rh.partners
    );
    assert!(rh.partners.iter().all(|&p| p == 1), "{:?}", rh.partners);
    assert!(rh.uniform_head.is_some());
}

// ===========================================================================
// Honesty: refusals.
// ===========================================================================

#[test]
fn gamma_lmp2_refuses_what_it_cannot_do() {
    // Non-orthorhombic cell: Resta weights refused, not mis-weighted.
    let tri = triclinic_cell();
    let prep = prep_for(&tri, &pyscf_sto3g_h());
    let err = resta_operator(&tri, &prep).unwrap_err().to_string();
    assert!(err.contains("orthorhombic"), "{err}");

    let sys = wrapped4();
    // Domain fit without the fit parts.
    let no_parts = GammaLmp2Inputs {
        fit: None,
        ..sys.inputs()
    };
    let err = gamma_lmp2(
        &sys.cell,
        &sys.rhf,
        &no_parts,
        &GammaLmp2Config {
            fit_radius_bohr: Some(5.0),
            ..cfg(0.0)
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("fit parts"), "{err}");
    // frozen_core = nocc: active_occ refuses.
    let err = gamma_lmp2(
        &sys.cell,
        &sys.rhf,
        &sys.inputs(),
        &GammaLmp2Config {
            frozen_core: 4,
            ..cfg(0.0)
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("frozen_core"), "{err}");
    // Unconverged reference.
    let mut bad = sys.rhf.clone();
    bad.converged = false;
    assert!(gamma_lmp2(&sys.cell, &bad, &sys.inputs(), &cfg(0.0)).is_err());
}

// ===========================================================================
// (f) A-drop (FINDINGS Iteration 5b): restored q = 0 head, J' = J + c μ μᵀ.
//
// Artifact hypotheses (stated before measuring):
// * eps = 0: if the extra column is not applied (vacuum) the LMP2 equals
//   canonical Gamma MP2 and the closed-form E_head differs by ~3e-3 — the
//   anchor compares against E_head AND asserts |E_head − E_can| > 1e-3, so a
//   vacuous head fails. If the head is applied with a wrong scale the pinned
//   E_can − E_head (prototype +2.955e-3 at 1x1x4) moves (Ω·8 mutant below).
// * Onset: if A-drop were a relabelled raw gate the partner count stays N;
//   the raw gate on the same spaces must still keep N (reachable contrast).
// * Richardson: if the 2b matrix were wrong (sign/multiple), the Richardson
//   residual would be WORSE than single-b, not ~6x (1x1x4) / 20x (1x1x8)
//   better.
// ===========================================================================

/// Prototype (run_lmp2_uniform.py, FINDINGS 5b): E_can − E_head at 1x1x4.
const PIN_E_CAN_MINUS_E_HEAD_1X1X4: f64 = 2.955e-3;
/// FINDINGS Iteration 5 anchor table, 1x1x8 straight.
const PIN_E_MP2_CANON_1X1X8: f64 = -1.528153223461e-1;
/// FINDINGS 5b A-drop rows at N = 8 (reference E_head): (eps, dE, sig figs,
/// partners per molecule).
const PIN_ADROP_ROWS_1X1X8: [(f64, f64, i32, usize); 4] = [
    (1e-3, 1.01e-4, 3, 1),
    (3e-4, 5.49e-5, 3, 3),
    (1e-4, 2.28e-6, 3, 3),
    (3e-5, 4.6e-7, 2, 3),
];

fn half_unit(x: f64, sig: i32) -> f64 {
    0.5 * 10f64.powi(x.abs().log10().floor() as i32 - (sig - 1))
}

fn local_b_ov(sys: &System, sp: &GammaLocalSpaces) -> Array2<f64> {
    b_ov_from_ao_b(
        sys.gdf.b(),
        sp.c_occ.nrows(),
        sp.c_occ.view(),
        sp.c_vir.view(),
    )
    .expect("local B_ov")
}

/// E_head = closed-form MP2 on the head-augmented B (the A-drop eps = 0 target).
fn e_head_closed_form(sys: &System, sp: &GammaLocalSpaces, head: &UniformHead) -> f64 {
    mp2_closed_form_local(
        &local_b_ov(sys, sp),
        Some(&head.column()),
        &sp.f_oo,
        &sp.f_vv,
    )
    .expect("closed form")
}

/// `(rel |J − J_unif(μ_Richardson)|, rel |J − J_unif(f(b))|)` at the
/// farthest (minimum-image) occupied pair, relative to max |J| of that block;
/// also checks the head-augmented source block IS `J − J_unif` there.
fn farthest_pair_residuals(sys: &System, sp: &GammaLocalSpaces, head: &UniformHead) -> (f64, f64) {
    let no = sp.no();
    let dist = centroid_distances(&sys.cell, &sp.occ_centers, PeriodicDistance::MinimumImage);
    let (mut fi, mut fj, mut dmax) = (0, 0, -1.0);
    for i in 0..no {
        for j in 0..no {
            if dist[(i, j)] > dmax {
                (fi, fj, dmax) = (i, j, dist[(i, j)]);
            }
        }
    }
    let src = GammaPairIntegrals::new(
        &sys.cell,
        sp,
        &sys.gdf,
        None,
        None,
        PeriodicDistance::MinimumImage,
        Some(AMPLE),
    )
    .unwrap();
    let (jf, _) = src.block(fi, fj).unwrap();
    let jmax = jf.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let rel_r = max_abs_diff(&jf, &head.coupling(fi, fj)) / jmax;
    let rel_1 = max_abs_diff(
        &jf,
        &uniform_coupling(&head.mu_single_b, head.scale, fi, fj),
    ) / jmax;
    // The head-augmented source is J − J_unif at that block (same residual).
    let (jp, _) = src.with_uniform_head(head).unwrap().block(fi, fj).unwrap();
    let rel_p = jp.iter().fold(0.0_f64, |m, &x| m.max(x.abs())) / jmax;
    assert!((rel_p - rel_r).abs() < 1e-12, "{rel_p:.3e} vs {rel_r:.3e}");
    eprintln!(
        "farthest pair ({fi},{fj}) at {dmax:.2} Bohr: max|J| {jmax:.3e}; rel |J'| Richardson \
         {rel_r:.2e}, single-b {rel_1:.2e} (ratio {:.1})",
        rel_1 / rel_r
    );
    (rel_r, rel_1)
}

#[test]
fn a_drop_eps0_anchor_is_closed_form_mp2_on_the_head_augmented_b() {
    let sys = straight4();
    let e_can = sys.canonical();
    let sp = sys.spaces(&cfg(0.0));

    // Anchor of the anchor: the closed form WITHOUT the head is canonical
    // Gamma MP2 (independent code: eigenbasis rotation vs gamma_mp2).
    let b_ov = local_b_ov(sys, &sp);
    let e_cf = mp2_closed_form_local(&b_ov, None, &sp.f_oo, &sp.f_vv).unwrap();
    eprintln!("closed form (no head) {e_cf:.12e} vs canonical {e_can:.12e}");
    assert!((e_cf - e_can).abs() < 1e-10, "{:.3e}", e_cf - e_can);

    let head = uniform_head(&sys.cell, &sys.obs, &sp).expect("uniform head");
    assert_eq!(head.axis, 2);
    let e_head = e_head_closed_form(sys, &sp, &head);

    // eps = 0 A-drop == E_head (CG on ragged blocks vs eigenbasis closed form).
    let r = sys.run(&cfg_head(0.0), &sp);
    let d = r.e_corr - e_head;
    eprintln!(
        "A-drop eps=0: E {:.12e} E_head {e_head:.12e} dE {d:+.2e}; E_can - E_head {:+.4e} \
         (prototype {PIN_E_CAN_MINUS_E_HEAD_1X1X4:+.3e})",
        r.e_corr,
        e_can - e_head
    );
    assert!(d.abs() < 1e-10, "A-drop anchor dE {d:.3e}");
    assert_eq!(r.pairs_kept, 16);
    assert!((r.pair_energies.sum() - r.e_corr).abs() < 1e-12);
    // Not a vacuum test: the target really is NOT canonical Gamma MP2, and
    // the difference is the prototype's uniform term (4 sig figs quoted;
    // Rust vs prototype E_can agree to 1e-8, so 1e-6 is two half-units).
    assert!((e_can - e_head).abs() > 1e-3);
    assert!(
        (e_can - e_head - PIN_E_CAN_MINUS_E_HEAD_1X1X4).abs() < 1e-6,
        "E_can - E_head {:.6e}",
        e_can - e_head
    );
    // Mutation (prototype's "J_unif volume Ω/8"): the pin must see it.
    let mut m = head.clone();
    m.scale /= 8.0;
    let e_m = e_head_closed_form(sys, &sp, &m);
    eprintln!("MUTANT scale/8: E_can - E_head {:+.4e}", e_can - e_m);
    assert!((e_can - e_m - PIN_E_CAN_MINUS_E_HEAD_1X1X4).abs() > 1e-4);

    // The driver reports the right target for each gate.
    let with_ref = gamma_lmp2(
        &sys.cell,
        &sys.rhf,
        &sys.inputs(),
        &GammaLmp2Config {
            compute_reference: true,
            ..cfg_head(0.0)
        },
    )
    .unwrap();
    assert!((with_ref.e_corr_target - e_head).abs() < 1e-12);
    assert!((with_ref.e_corr_canonical - e_can).abs() < 1e-12);
    let info = with_ref.uniform_head.expect("head diagnostics");
    assert_eq!(info.axis, 2);
    assert!((info.scale - 4.0 * PI / sys.cell.volume()).abs() < 1e-15);
    let raw_ref = gamma_lmp2(
        &sys.cell,
        &sys.rhf,
        &sys.inputs(),
        &GammaLmp2Config {
            compute_reference: true,
            ..cfg(0.0)
        },
    )
    .unwrap();
    assert!(raw_ref.uniform_head.is_none());
    assert_eq!(
        raw_ref.e_corr_target.to_bits(),
        raw_ref.e_corr_canonical.to_bits()
    );
    // Default unchanged (config honesty): shifted() is still the raw gate.
    assert_eq!(
        GammaLmp2Config::shifted(ExxDiv::Ewald, 1e-4).eps_gate,
        GammaEpsGate::RawIntegral
    );

    // The domain fit carries the same head: trivial radius == global, with head.
    let mi = MinImage::new(&sys.cell, PeriodicDistance::MinimumImage);
    let mi = &mi;
    let rstar = sp
        .occ_centers
        .iter()
        .flat_map(|c| {
            let c = *c;
            sys.parts.aux_centers.iter().map(move |p| mi.between(c, *p))
        })
        .fold(0.0_f64, f64::max)
        + 1e-6;
    let rd = sys.run(
        &GammaLmp2Config {
            fit_radius_bohr: Some(rstar),
            ..cfg_head(0.0)
        },
        &sp,
    );
    assert!(
        (rd.e_corr - e_head).abs() < 1e-10,
        "{:.3e}",
        rd.e_corr - e_head
    );
}

#[test]
fn a_drop_removes_the_onset_on_1x1x4_and_richardson_dipoles_beat_single_b() {
    let sys = straight4();
    let sp = sys.spaces(&cfg(1e-4));
    let no = sp.no();

    // resta_matrix_at(m = 1) is the Berghold operator's matrix.
    let op = resta_operator(&sys.cell, &sys.obs).unwrap();
    let (re1, im1) = resta_matrix_at(&sys.cell, &sys.obs, 2, 1).unwrap();
    assert!(max_abs_diff(&re1, &op.re[2]) < 1e-13 && max_abs_diff(&im1, &op.im[2]) < 1e-13);

    let head = uniform_head(&sys.cell, &sys.obs, &sp).unwrap();
    // Prototype X1 at N = 4: 3.7e-5 / 2.4e-4 of max|J| 5.8e-3 -> 6.4e-3 / 4.1e-2.
    let (rel_r, rel_1) = farthest_pair_residuals(sys, &sp, &head);
    assert!(rel_r < 1.5e-2, "Richardson residual {rel_r:.3e}");
    assert!(
        rel_1 > 3e-2,
        "single-b residual {rel_1:.3e} (premise changed)"
    );
    assert!(
        rel_1 > 3.0 * rel_r,
        "Richardson gains only {:.2}x",
        rel_1 / rel_r
    );

    // Onset: raw gate keeps all N² = 16 at eps 1e-4 (PIN_EPS_ROWS); A-drop
    // keeps 3 partners per molecule (prototype 3/3/3/3 at N = 4/8/16/32).
    let raw = sys.run(&cfg(1e-4), &sp);
    let adrop = sys.run(&cfg_head(1e-4), &sp);
    let e_head = e_head_closed_form(sys, &sp, &head);
    let de = adrop.e_corr - e_head;
    eprintln!(
        "eps 1e-4: raw partners {:?} ({} pairs); A-drop partners {:?} ({} pairs), dE vs E_head \
         {de:+.3e}",
        raw.partners, raw.pairs_kept, adrop.partners, adrop.pairs_kept
    );
    assert_eq!(raw.pairs_kept, no * no);
    assert!(
        adrop.partners.iter().all(|&p| p == 3),
        "{:?}",
        adrop.partners
    );
    // Prototype per-molecule a = 2.5e-7, |b| <= 7e-6 at eps 1e-4: N = 4
    // error is O(1e-6); positive (correlation lost). Loose bound, not a pin.
    assert!(de > 0.0 && de < 1e-5, "A-drop truncation error {de:.3e}");
}

#[test]
fn a_drop_refuses_non_needle_supercells() {
    // The needle itself is accepted.
    assert_eq!(needle_axis(&needle_cell(4, false)).unwrap(), 2);
    let msg = |c: &Cell| needle_axis(c).unwrap_err().to_string();
    // Cubic (W = I/3, not implemented), too-short needle, slab: refused.
    let cube = Cell::new(
        hydrogens(&h2_prim()),
        [[A0, 0.0, 0.0], [0.0, A0, 0.0], [0.0, 0.0, A0]],
    )
    .unwrap();
    let slab = Cell::new(
        hydrogens(&h2_prim()),
        [[A0, 0.0, 0.0], [0.0, 4.0 * A0, 0.0], [0.0, 0.0, 4.0 * A0]],
    )
    .unwrap();
    let short = needle_cell(2, false);
    for (name, c) in [("cubic", &cube), ("1x1x2", &short), ("slab", &slab)] {
        let e = msg(c);
        eprintln!("{name}: {e}");
        assert!(e.contains("NEEDLE"), "{name}: {e}");
    }
    assert!(msg(&triclinic_cell()).contains("orthorhombic"));

    // The driver refuses before any localisation work (cube cell, any inputs).
    let sys = wrapped4();
    let err = gamma_lmp2(&cube, &sys.rhf, &sys.inputs(), &cfg_head(1e-4))
        .unwrap_err()
        .to_string();
    assert!(err.contains("NEEDLE"), "{err}");
    let sp4 = sys.spaces(&cfg(1e-4));
    let err = gamma_lmp2_with_spaces(&cube, &sys.rhf, &sys.inputs(), &cfg_head(1e-4), &sp4)
        .unwrap_err()
        .to_string();
    assert!(err.contains("NEEDLE"), "{err}");
    // Applying the head twice is refused (it would double it).
    let sp = sys.spaces(&cfg(0.0));
    let head = uniform_head(&sys.cell, &sys.obs, &sp).unwrap();
    let src = GammaPairIntegrals::new(
        &sys.cell,
        &sp,
        &sys.gdf,
        None,
        None,
        PeriodicDistance::MinimumImage,
        Some(AMPLE),
    )
    .unwrap()
    .with_uniform_head(&head)
    .unwrap();
    assert!(src.has_uniform_head());
    assert!(src.with_uniform_head(&head).is_err());
}

#[test]
#[ignore = "slow: explicit 1x1x8 RS-GDF build + SCF (nao 32, naux 224), time unmeasured; \
            run with --ignored"]
fn a_drop_1x1x8_reproduces_the_prototype_rows() {
    let sys = build(8, false);
    let e_can = sys.canonical();
    eprintln!("1x1x8 E_MP2 canonical {e_can:.12e} (prototype {PIN_E_MP2_CANON_1X1X8:.12e})");
    assert!((e_can - PIN_E_MP2_CANON_1X1X8).abs() < 1e-8, "{e_can:.12e}");
    let sp = sys.spaces(&cfg(0.0));
    let no = sp.no();
    assert_eq!(no, 8);
    let head = uniform_head(&sys.cell, &sys.obs, &sp).unwrap();
    let e_head = e_head_closed_form(&sys, &sp, &head);
    let r0 = sys.run(&cfg_head(0.0), &sp);
    assert!(
        (r0.e_corr - e_head).abs() < 1e-10,
        "{:.3e}",
        r0.e_corr - e_head
    );

    // Richardson vs single-b (prototype port test: <= 2e-3 vs >= 1e-2 fails;
    // measured 4.4e-4 vs 1.0e-2, ~20x).
    let (rel_r, rel_1) = farthest_pair_residuals(&sys, &sp, &head);
    assert!(rel_r < 2e-3 && rel_1 > 5e-3, "{rel_r:.3e} / {rel_1:.3e}");
    assert!(
        rel_1 > 10.0 * rel_r,
        "Richardson gains only {:.1}x",
        rel_1 / rel_r
    );

    // Onset: raw keeps all 64 at eps 1e-4 (N* = 232); A-drop 3 per molecule.
    let raw = sys.run(&cfg(1e-4), &sp);
    assert_eq!(raw.pairs_kept, no * no);

    for (eps, de_pin, sig, partners) in PIN_ADROP_ROWS_1X1X8 {
        let r = sys.run(&cfg_head(eps), &sp);
        let de = r.e_corr - e_head;
        eprintln!(
            "A-drop eps {eps:<7}: dE {de:+.4e} (pin {de_pin:+.3e}) partners {:?}",
            r.partners
        );
        assert!(
            r.partners.iter().all(|&p| p == partners),
            "eps {eps}: {:?}",
            r.partners
        );
        assert!(
            (de - de_pin).abs() < half_unit(de_pin, sig),
            "eps {eps}: dE {de:.4e} vs {de_pin:.3e}"
        );
    }

    // Finite-size term b = 2 E4 − E8 (prototype test: canonical > 3e-3,
    // head-restored in (0.5e-3, 1.6e-3)).
    let s4 = straight4();
    let sp4 = s4.spaces(&cfg(0.0));
    let e4_head = e_head_closed_form(s4, &sp4, &uniform_head(&s4.cell, &s4.obs, &sp4).unwrap());
    let (b_can, b_head) = (2.0 * s4.canonical() - e_can, 2.0 * e4_head - e_head);
    eprintln!("b = 2E4 - E8: canonical {b_can:+.3e}, head restored {b_head:+.3e}");
    assert!(b_can > 3e-3 && b_head > 0.5e-3 && b_head < 1.6e-3);
}
