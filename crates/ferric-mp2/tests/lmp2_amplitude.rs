//! Exactness anchors for the amplitude-threshold local MP2 (Rust Phase 2).
//!
//! Protocol order (CLAUDE.md Experimental Protocol): the trivial-limit
//! anchor and its mutation test come FIRST; the finite-ε behavior checks
//! ride behind them. The anchor pair is canonical-orbital closed-form RI-MP2
//! vs localized-orbital ragged CG — independent constructions sharing only
//! the RI integrals, so the bar is CG tolerance, not the RI floor.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{
    amplitude_lmp2, amplitude_lmp2_with_virtuals, build_vvhv, check_vvhv, AmplitudeLmp2Config,
    VvHv,
};
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

#[test]
fn vvhv_construction_is_orthonormal_and_spans_the_virtual_space() {
    let su = setup("water.xyz");
    let vvhv = build_vvhv(&su.mol, &su.obs, &su.obs_bs, &su.rhf).unwrap();
    let nocc = (su.mol.nelec() as usize) / 2;
    let (dev_orth, dev_span) = check_vvhv(&su.obs, &su.rhf, nocc, &vvhv.c_vloc);
    assert!(dev_orth < 1e-8, "orthonormality dev {dev_orth:.2e}");
    assert!(dev_span < 1e-8, "span dev {dev_span:.2e}");
    assert_eq!(vvhv.n_valence + vvhv.n_hard, su.obs.nbasis() - nocc);
}

#[test]
fn eps_zero_matches_canonical_ri_mp2() {
    let su = setup("water.xyz");
    let cfg = AmplitudeLmp2Config { eps: 0.0, frozen_core: 1, ..Default::default() };
    let r = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg)
        .unwrap();
    let de = r.e_corr - r.e_corr_canonical_ri;
    eprintln!(
        "ANCHOR water/6-31G: E_corr={:.10} canonical={:.10} dE={de:+.3e} (cg {} iters)",
        r.e_corr, r.e_corr_canonical_ri, r.cg_iterations
    );
    assert!(r.cg_converged);
    assert!(r.keep_fraction == 1.0 && r.pair_fraction == 1.0);
    assert!(de.abs() < 1e-9, "eps=0 anchor FAILED: dE={de:+.3e}");
}

/// A TEST YOU HAVE NEVER SEEN FAIL IS AN ASSUMPTION: breaking the virtual
/// space (dropping one hard virtual) must break the anchor. Bar matches the
/// Python rig's measured mutation scale (span loss moves E by >1e-6).
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
    let cfg = AmplitudeLmp2Config { eps: 0.0, frozen_core: 1, ..Default::default() };
    let r = amplitude_lmp2_with_virtuals(
        &su.mol, &su.obs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg, &broken,
    )
    .unwrap();
    let de = (r.e_corr - r.e_corr_canonical_ri).abs();
    eprintln!("MUTATION water/6-31G: |dE|={de:.3e} (must exceed 1e-6)");
    assert!(de > 1e-6, "mutation NOT detected: |dE|={de:.3e}");
}

/// Finite-ε behavior on C4: error one-sided (under-correlation), counters
/// live, CG converged, and the keep fraction in the same band the Python
/// rig measured (0.0785 at ε=1e-3 with PySCF's auto aux; the aux here is
/// cc-pvdz-ri so J magnitudes — and hence the mask — differ slightly).
#[test]
fn eps_sweep_on_c4_is_one_sided_with_live_counters() {
    let su = setup("alkane_4.xyz");
    let cfg = AmplitudeLmp2Config { eps: 1e-3, frozen_core: 4, ..Default::default() };
    let r = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, Operator::coulomb(), &su.rhf, &cfg)
        .unwrap();
    eprintln!(
        "C4 eps=1e-3: dE={:+.3e} keep={:.4} pairs={:.3} dom(mean/max)={:.1}/{} cg={} raggedx={}",
        r.e_corr - r.e_corr_canonical_ri,
        r.keep_fraction,
        r.pair_fraction,
        r.dom_mean,
        r.dom_max,
        r.cg_iterations,
        r.dense_flops_per_matvec / r.ragged_flops_per_matvec.max(1),
    );
    assert!(r.cg_converged);
    let de = r.e_corr - r.e_corr_canonical_ri;
    assert!(de > 0.0, "threshold error must be one-sided (under-correlation), got {de:+.3e}");
    assert!(de < 5e-2, "eps=1e-3 error implausibly large: {de:+.3e}");
    assert!(
        r.keep_fraction > 0.03 && r.keep_fraction < 0.20,
        "keep fraction {:.4} outside the Python-measured band (0.0785 ±aux)",
        r.keep_fraction
    );
    // C4 is BELOW the locality onset, so dom_max may touch the full virtual
    // space (Python measured the same); the mask biting shows in the MEAN.
    let nv = su.obs.nbasis() - (su.mol.nelec() as usize) / 2;
    assert!(r.dom_mean < nv as f64, "dom mean {:.1} not below nv={nv}", r.dom_mean);
}
