//! Is the coupled (ALDA) response of a monomer pole-free on the imaginary axis?
//!
//! `Π(iω) = Π⁰^{1/2} M⁻¹ Π⁰^{1/2}` with `M = I + Π⁰^{1/2} K Π⁰^{1/2}`. For a
//! physical (stable) reference `M ≻ 0` at every ω ≥ 0; a negative eigenvalue
//! means the Dyson resolvent crosses a pole on the integration path and the
//! Casimir–Polder integral is not defined.
//!
//! Hypotheses, stated before measuring:
//! * PHYSICS: a Kohn–Sham reference built with the SAME functional as the
//!   kernel (Slater exchange, `LDA_X`, with the ALDAX kernel) sits at a stable
//!   minimum of that functional, so `M(0) ≻ 0`. An RHF reference with an ALDA
//!   kernel has no such guarantee (orbitals and kernel come from different
//!   functionals).
//! * ARTIFACT: if the kernel projection carried a wrong factor or sign, `M(0)`
//!   would go negative for the matched KS reference too.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::cks_dispersion::{build_monomer_response, CksCoupling, CksDispersionConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const WATER: &str = "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.467\nH 0.0 -0.757 -0.467\n";
const METHANE: &str = "5\nmethane\nC 0.000000 -0.000140 1.859161\nH -0.888551 0.513060 1.494685\n\
H 0.888551 0.513060 1.494685\nH 0.000000 -1.026339 1.494868\nH 0.000000 0.000089 2.948284\n";

/// `min eig M(ω = 0)` under the ALDAX kernel, for an RHF (`xc = None`) or KS
/// (`xc = Some(..)`) reference.
fn static_dyson_min(xyz: &str, obs_name: &str, aux_name: &str, scf_xc: Option<&str>) -> f64 {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let scf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            xc: scf_xc.map(str::to_string),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(scf.converged);
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let cfg = CksDispersionConfig {
        n_freq: 4,
        ..Default::default()
    };
    let r = build_monomer_response(&mol, &obs, &dfbs, &scf, "LDA_X", &cfg).unwrap();
    r.dyson_min_eigenvalue(0.0, CksCoupling::Cks)
        .unwrap()
        .unwrap()
}

/// Isolated monomers (no ghost centers): both references are stable.
///
/// MEASURED: water/cc-pVDZ RHF 0.982353, KS(LDA_X) 0.975453; methane/aug-cc-pVDZ
/// RHF 0.968647, KS(LDA_X) 0.957123. The bar (> 0.5) is far from both the
/// measured values and the unstable ghosted values below.
#[test]
fn isolated_monomer_static_response_is_stable() {
    for (label, xyz, obs, aux) in [
        ("water/cc-pVDZ", WATER, "cc-pvdz", "cc-pvdz-ri"),
        (
            "methane/aug-cc-pVDZ",
            METHANE,
            "aug-cc-pvdz",
            "aug-cc-pvdz-rifit",
        ),
    ] {
        let hf = static_dyson_min(xyz, obs, aux, None);
        let ks = static_dyson_min(xyz, obs, aux, Some("LDA_X"));
        eprintln!("{label}: min eig M(0), ALDAX kernel: RHF ref {hf:.6}, KS(LDA_X) ref {ks:.6}");
        assert!(
            hf > 0.5 && ks > 0.5,
            "{label}: isolated monomer response unstable"
        );
    }
}

/// The SAME methane monomer in the dimer basis (ghost partner at the S22-08
/// position), which is how `mp2c_cks_interaction` builds it. KNOWN OPEN: the
/// static Dyson matrix goes negative, so the coupled Casimir–Polder integrand
/// crosses a pole. Asserts stability, so it fails today.
#[test]
#[ignore = "known open: ghosted monomer ALDA response is unstable (methane/aug-cc-pVDZ M(0) < 0)"]
fn ghosted_monomer_static_response_is_stable() {
    let ghost = "@C 0.000000 0.000140 -1.859161\n@H 0.000000 -0.000089 -2.948284\n\
@H -0.888551 -0.513060 -1.494685\n@H 0.888551 -0.513060 -1.494685\n@H 0.000000 1.026339 -1.494868\n";
    let body: String = METHANE.lines().skip(2).map(|l| format!("{l}\n")).collect();
    let xyz = format!("10\nmethane + ghost methane\n{body}{ghost}");
    let hf = static_dyson_min(&xyz, "aug-cc-pvdz", "aug-cc-pvdz-rifit", None);
    let ks = static_dyson_min(&xyz, "aug-cc-pvdz", "aug-cc-pvdz-rifit", Some("LDA_X"));
    eprintln!("ghosted methane/aug-cc-pVDZ: min eig M(0): RHF ref {hf:.6}, KS(LDA_X) ref {ks:.6}");
    assert!(
        hf > 0.0 && ks > 0.0,
        "ghosted monomer response unstable: RHF {hf:.4}, KS {ks:.4}"
    );
}
