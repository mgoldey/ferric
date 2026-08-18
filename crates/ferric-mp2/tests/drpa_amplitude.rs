//! Anchors for the amplitude-threshold dRPA (Rust port of
//! scripts/amplitude_rpa_proto.py). The formulation identities behind these
//! bars are PROVED in wiki/notebooks/12-amplitude-threshold-drpa.ipynb
//! (Riccati ≡ plasmon exactly; rotation invariance; first-order error
//! structure), so a failure here is an implementation bug by construction.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::drpa_amplitude::{
    amplitude_drpa, amplitude_drpa_with_virtuals, AmplitudeDrpaConfig,
};
use ferric_mp2::lmp2_amplitude::{build_vvhv, VvHv};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::s;

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

/// H2/STO-3G with STO-3G aux is the EXACT setup of the proof notebook's
/// live cell (single (i,a) pair): the localized Riccati energy must land on
/// the notebook's plasmon value -0.0126072623 AND on its own canonical
/// plasmon reference.
#[test]
fn h2_single_pair_matches_the_proof_notebook() {
    let su = setup("h2.xyz", "sto-3g", "sto-3g");
    let cfg = AmplitudeDrpaConfig { eps: 0.0, ..Default::default() };
    let r = amplitude_drpa(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg)
        .unwrap();
    let de = r.e_corr - r.e_corr_plasmon_canonical;
    eprintln!(
        "H2: E_corr={:.10} plasmon={:.10} dE={de:+.3e} notebook=-0.0126072623 (fp {} iters)",
        r.e_corr, r.e_corr_plasmon_canonical, r.iterations
    );
    assert!(r.converged);
    assert!(de.abs() < 1e-9, "H2 anchor FAILED: dE={de:+.3e}");
    assert!(
        (r.e_corr - (-0.012_607_262_3)).abs() < 1e-8,
        "H2 disagrees with the proof notebook: {:.10}",
        r.e_corr
    );
}

/// ε=0 anchor on a multi-pair system: localized Riccati fixed point vs the
/// canonical semicanonicalized plasmon formula — independent constructions
/// sharing only the RI integrals and the Fock operator.
#[test]
fn eps_zero_matches_canonical_plasmon_on_water() {
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");
    let cfg = AmplitudeDrpaConfig { eps: 0.0, frozen_core: 1, ..Default::default() };
    let r = amplitude_drpa(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg)
        .unwrap();
    let de = r.e_corr - r.e_corr_plasmon_canonical;
    eprintln!(
        "ANCHOR water: E_corr={:.10} plasmon={:.10} dE={de:+.3e} (fp {} iters)",
        r.e_corr, r.e_corr_plasmon_canonical, r.iterations
    );
    assert!(r.converged);
    assert!(r.keep_fraction == 1.0 && r.pair_fraction == 1.0);
    assert!(de.abs() < 1e-9, "eps=0 anchor FAILED: dE={de:+.3e}");
}

/// A TEST YOU HAVE NEVER SEEN FAIL IS AN ASSUMPTION: dropping one hard
/// virtual must break the anchor — and because dRPA is FIRST order in the
/// perturbation (proof notebook §3), the bar is the anchor's own 1e-9, not
/// the MP2 rig's 1e-6.
#[test]
fn mutated_virtual_space_fails_the_anchor() {
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");
    let vvhv = build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
    let nvir = vvhv.c_vloc.ncols();
    let broken = VvHv {
        c_vloc: vvhv.c_vloc.slice(s![.., ..nvir - 1]).to_owned(),
        n_valence: vvhv.n_valence,
        n_hard: vvhv.n_hard - 1,
    };
    let cfg = AmplitudeDrpaConfig { eps: 0.0, frozen_core: 1, ..Default::default() };
    let r = amplitude_drpa_with_virtuals(
        &su.mol, &su.obs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg, &broken,
    )
    .unwrap();
    let de = (r.e_corr - r.e_corr_plasmon_canonical).abs();
    eprintln!("MUTATION water: |dE|={de:.3e} (must exceed 1e-7)");
    assert!(de > 1e-7, "mutation NOT detected: |dE|={de:.3e}");
}

/// Masked-solve behavior: converges, error one-sided (under-correlation)
/// and — per the proof notebook — expected ~linear in ε; iteration count
/// must NOT grow when the mask tightens (masking removes ring coupling,
/// measured in the Python rig).
#[test]
fn masked_sweep_is_one_sided_and_iterations_shrink() {
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");
    let full = AmplitudeDrpaConfig { eps: 0.0, frozen_core: 1, ..Default::default() };
    let r0 = amplitude_drpa(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &full)
        .unwrap();
    let cfg = AmplitudeDrpaConfig { eps: 1e-3, frozen_core: 1, ..Default::default() };
    let r = amplitude_drpa(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg)
        .unwrap();
    let de = r.e_corr - r0.e_corr;
    eprintln!(
        "water eps=1e-3: dE={de:+.3e} keep={:.4} iters {} (full-mask {})",
        r.keep_fraction, r.iterations, r0.iterations
    );
    assert!(r.converged);
    assert!(de > 0.0, "threshold error must be one-sided, got {de:+.3e}");
    assert!(r.keep_fraction < 1.0, "mask did not bite");
    assert!(
        r.iterations <= r0.iterations,
        "masking should not increase ring-coupling iterations ({} > {})",
        r.iterations,
        r0.iterations
    );
}

/// VARIANT-CONSISTENCY bar for the ragged port: at finite ε the ragged
/// path differs from the dense V1 in two DOCUMENTED ways (swap-closed
/// Eq-8 pattern instead of plain |B|>ε, and pattern-projected ring
/// intermediates). The requirement is that this variant difference stays
/// SUB-DOMINANT to the threshold truncation error itself — otherwise the
/// port changed the method, not the data structure. At ε=0 both are
/// exactly the full equations (covered by the anchors above).
#[test]
fn ragged_variant_difference_is_subdominant_to_truncation() {
    use ferric_mp2::drpa_amplitude::amplitude_drpa_dense;
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");
    let full = AmplitudeDrpaConfig { eps: 0.0, frozen_core: 1, ..Default::default() };
    let r_full = amplitude_drpa(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &full)
        .unwrap();
    let cfg = AmplitudeDrpaConfig { eps: 1e-3, frozen_core: 1, ..Default::default() };
    let vvhv = ferric_mp2::lmp2_amplitude::build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
    let r_ragged = amplitude_drpa(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg)
        .unwrap();
    let r_dense = amplitude_drpa_dense(&su.mol, &su.obs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg, &vvhv)
        .unwrap();
    let trunc = (r_dense.e_corr - r_full.e_corr).abs();
    let variant = (r_ragged.e_corr - r_dense.e_corr).abs();
    eprintln!(
        "eps=1e-3: truncation={trunc:.3e}, ragged-vs-dense variant diff={variant:.3e}"
    );
    assert!(variant > 0.0, "variant difference vanished — patterns identical? check the swap closure");
    assert!(
        variant < trunc,
        "variant difference ({variant:.3e}) DOMINATES truncation ({trunc:.3e}) — the port changed the method"
    );
}

/// DIIS acceleration must land on the SAME fixed point as the plain damped
/// iteration: at eps=0 both solve the exact Riccati equation, so the DIIS
/// coefficients (which only reweight iterates spanning one linear subspace)
/// cannot change the root — agreement should be at the 1e-12 fp_rtol floor.
/// At finite eps the ragged variant/pattern-projection noise (see the
/// subdominance test above) sets the bar instead, so this checks DIIS-on vs
/// DIIS-off agree to well inside the ~1e-3 truncation error at eps=1e-3.
#[test]
fn diis_matches_plain_fixed_point() {
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");

    // eps = 0: identical root, tight bound.
    let cfg0 = AmplitudeDrpaConfig { eps: 0.0, frozen_core: 1, ..Default::default() };
    let r0_plain = amplitude_drpa(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg0)
        .unwrap();
    let cfg0_diis = AmplitudeDrpaConfig { diis: Some(6), ..cfg0.clone() };
    let r0_diis = amplitude_drpa(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg0_diis)
        .unwrap();
    let de0 = (r0_diis.e_corr - r0_plain.e_corr).abs();
    eprintln!(
        "eps=0: plain iters={} E={:.12} | DIIS iters={} E={:.12} dE={de0:.3e}",
        r0_plain.iterations, r0_plain.e_corr, r0_diis.iterations, r0_diis.e_corr
    );
    assert!(r0_diis.converged);
    assert!(de0 < 1e-11, "DIIS root disagrees with the plain fixed point at eps=0: dE={de0:.3e}");
    assert!(
        r0_diis.iterations <= r0_plain.iterations,
        "DIIS should not need MORE iterations than the plain damped fixed point ({} > {})",
        r0_diis.iterations,
        r0_plain.iterations
    );

    // finite eps: subdominant to the eps=1e-3 truncation error itself
    // (measured ~1e-3 scale on this system; require well inside it).
    let cfg = AmplitudeDrpaConfig { eps: 1e-3, frozen_core: 1, ..Default::default() };
    let r_plain = amplitude_drpa(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg)
        .unwrap();
    let cfg_diis = AmplitudeDrpaConfig { diis: Some(6), ..cfg.clone() };
    let r_diis = amplitude_drpa(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg_diis)
        .unwrap();
    let de = (r_diis.e_corr - r_plain.e_corr).abs();
    eprintln!(
        "eps=1e-3: plain iters={} E={:.12} | DIIS iters={} E={:.12} dE={de:.3e}",
        r_plain.iterations, r_plain.e_corr, r_diis.iterations, r_diis.e_corr
    );
    assert!(r_diis.converged);
    assert!(de < 1e-9, "DIIS disagrees with the plain fixed point beyond fp_rtol noise: dE={de:.3e}");

    // dense path: same convention, same bound, independent construction.
    use ferric_mp2::drpa_amplitude::amplitude_drpa_dense;
    let vvhv = ferric_mp2::lmp2_amplitude::build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
    let rd_plain = amplitude_drpa_dense(&su.mol, &su.obs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg, &vvhv)
        .unwrap();
    let rd_diis = amplitude_drpa_dense(&su.mol, &su.obs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg_diis, &vvhv)
        .unwrap();
    let de_dense = (rd_diis.e_corr - rd_plain.e_corr).abs();
    eprintln!(
        "dense eps=1e-3: plain iters={} E={:.12} | DIIS iters={} E={:.12} dE={de_dense:.3e}",
        rd_plain.iterations, rd_plain.e_corr, rd_diis.iterations, rd_diis.e_corr
    );
    assert!(rd_diis.converged);
    assert!(de_dense < 1e-9, "dense DIIS disagrees with the plain fixed point: dE={de_dense:.3e}");
}

/// ε-linked stopping tolerance: loosening the fixed-point rtol down to
/// ~ε must stay sub-dominant to the eps truncation error itself — the
/// REQUIRED measurement before eps_rtol_factor gets a non-None default.
/// Calibration table + the c value chosen are in
/// wiki/amplitude-threshold-drpa.md (dated section).
#[test]
fn eps_linked_rtol_is_subdominant_to_truncation() {
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");
    let tight = AmplitudeDrpaConfig { eps: 1e-3, frozen_core: 1, fp_rtol: 1e-12, ..Default::default() };
    let r_tight = amplitude_drpa(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &tight)
        .unwrap();
    // reference: canonical plasmon carries the "true" answer at this eps
    // (r_tight.e_corr_plasmon_canonical is the eps=0-class continuum
    // limit); the truncation error is what tightening rtol can never fix.
    let trunc = (r_tight.e_corr - r_tight.e_corr_plasmon_canonical).abs();
    // calibrated c: see wiki/amplitude-threshold-drpa.md. c=1.0 looked
    // safe on alkane_8/alkane_12 (1.5-2.5% of truncation) but FAILS on
    // water at eps=1e-3 (86% of a much smaller truncation error, 4.2e-5
    // vs the alkanes' ~1e-2) — a genuine per-system truncation-scale
    // effect (per the reliability conventions: measure per-system, don't
    // trust a single-molecule sweep). c=0.1 stays under the 10% bar on
    // ALL three systems measured (water/alkane_8/alkane_12), worst case
    // 8.2% on water/eps=1e-3.
    let c = 0.1;
    let loose = AmplitudeDrpaConfig { eps_rtol_factor: Some(c), ..tight.clone() };
    let r_loose = amplitude_drpa(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &loose)
        .unwrap();
    let diff = (r_loose.e_corr - r_tight.e_corr).abs();
    eprintln!(
        "eps=1e-3 c={c:.0e}: tight iters={} loose iters={} diff={diff:.3e} trunc={trunc:.3e} frac={:.4}",
        r_tight.iterations, r_loose.iterations, diff / trunc
    );
    assert!(r_loose.converged);
    assert!(
        r_loose.iterations <= r_tight.iterations,
        "eps-linked rtol should not need MORE iterations ({} > {})",
        r_loose.iterations,
        r_tight.iterations
    );
    assert!(
        diff <= 0.10 * trunc,
        "eps-linked rtol diff ({diff:.3e}) exceeds 10% of the truncation error ({trunc:.3e})"
    );
}
