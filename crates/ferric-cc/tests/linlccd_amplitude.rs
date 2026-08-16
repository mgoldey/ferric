//! Anchors for amplitude-threshold LinLCCD. Formulation proved in
//! wiki/notebooks/13-amplitude-threshold-linlccd.ipynb; the canonical
//! reference is ferric_cc::linlccd::linlccd — the spin-orbital einsum
//! implementation, an INDEPENDENT formulation sharing only RI integrals.
use ferric_cc::linlccd::{linlccd, LadderVariant};
use ferric_cc::linlccd_amplitude::{
    amplitude_linlccd, amplitude_linlccd_with_virtuals, AmplitudeLinLccdConfig,
};
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
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

fn setup(xyz: &str) -> Setup {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
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

fn canonical(su: &Setup, variant: LadderVariant, fc: usize) -> f64 {
    linlccd(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &CcConfig { frozen_core: fc, energy_conv: 1e-11, max_iter: 200, ..Default::default() },
        variant,
    )
    .unwrap()
    .correlation_energy
}

/// ε=0 anchors for all three tiers on water/6-31G: localized masked CG vs
/// the canonical spin-orbital solver. DriversOnly pins the shared machinery
/// (≡ RI-MP2); Hh and Full pin the two ladder contractions the proof
/// notebook derived.
#[test]
fn eps_zero_matches_canonical_spin_orbital_all_variants() {
    let su = setup("water.xyz");
    let cfg = AmplitudeLinLccdConfig { eps: 0.0, frozen_core: 1, ..Default::default() };
    for variant in [LadderVariant::DriversOnly, LadderVariant::Hh, LadderVariant::Full] {
        let r = amplitude_linlccd(
            &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg, variant,
        )
        .unwrap();
        let e_can = canonical(&su, variant, 1);
        let de = r.e_corr - e_can;
        eprintln!(
            "ANCHOR water {variant:?}: E_corr={:.10} canonical={:.10} dE={de:+.3e} (cg {})",
            r.e_corr, e_can, r.cg_iterations
        );
        assert!(r.cg_converged);
        assert!(de.abs() < 5e-9, "{variant:?} eps=0 anchor FAILED: dE={de:+.3e}");
    }
}

/// A TEST YOU HAVE NEVER SEEN FAIL IS AN ASSUMPTION: dropping one hard
/// virtual must break the Hh anchor.
#[test]
fn mutated_virtual_space_fails_the_anchor() {
    let su = setup("water.xyz");
    let vvhv = build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
    let nvir = vvhv.c_vloc.ncols();
    let broken = VvHv {
        c_vloc: vvhv.c_vloc.slice(s![.., ..nvir - 1]).to_owned(),
        n_valence: vvhv.n_valence,
        n_hard: vvhv.n_hard - 1,
    };
    let cfg = AmplitudeLinLccdConfig { eps: 0.0, frozen_core: 1, ..Default::default() };
    let r = amplitude_linlccd_with_virtuals(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        LadderVariant::Hh,
        &broken,
    )
    .unwrap();
    let e_can = canonical(&su, LadderVariant::Hh, 1);
    let de = (r.e_corr - e_can).abs();
    eprintln!("MUTATION water Hh: |dE|={de:.3e} (must exceed 1e-6)");
    assert!(de > 1e-6, "mutation NOT detected: |dE|={de:.3e}");
}

/// Masked sweep smoke on the Hh tier: converges, error one-sided
/// (under-correlation, the Hylleraas argument from the proof notebook),
/// counters live.
#[test]
fn masked_sweep_is_one_sided_hh() {
    let su = setup("water.xyz");
    let full = AmplitudeLinLccdConfig { eps: 0.0, frozen_core: 1, ..Default::default() };
    let r0 = amplitude_linlccd(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &full,
        LadderVariant::Hh,
    )
    .unwrap();
    let cfg = AmplitudeLinLccdConfig { eps: 1e-3, frozen_core: 1, ..Default::default() };
    let r = amplitude_linlccd(
        &su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg,
        LadderVariant::Hh,
    )
    .unwrap();
    let de = r.e_corr - r0.e_corr;
    eprintln!(
        "water Hh eps=1e-3: dE={de:+.3e} keep={:.4} (cg {} vs full {})",
        r.keep_fraction, r.cg_iterations, r0.cg_iterations
    );
    assert!(r.cg_converged);
    assert!(r.keep_fraction < 1.0, "mask did not bite");
    assert!(de > 0.0, "threshold error must be one-sided, got {de:+.3e}");
}
