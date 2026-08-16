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
