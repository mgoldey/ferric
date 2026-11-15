//! [`Certificate`](ferric_core::certificate::Certificate)s of the MP2-family results: capability and grade from
//! `ferric_core::certificate::METHOD_KIND_GRADES`, folded with the SCF
//! reference's certificate ([`Certificate::for_post_scf`](ferric_core::certificate::Certificate::for_post_scf)), plus the
//! settings each method ran with.
//!
//! Built after the solve from the config the method ran with; nothing here
//! changes an energy. The build stamp comes from the reference certificate
//! (`ferric-cli` and `ferric-python` stamp it).

use ferric_core::certificate::{energy_anchor, Certificate, EnergyAnchor, Grade, RunEnvelope};
use ferric_core::gpu::provenance::PrecisionRecord;
use ferric_core::mol::Molecule;
use ferric_integrals::operator::{Operator, OperatorKind};

use crate::attenuated::AttenuatedMp2Config;
use crate::double_hybrid::{DoubleHybridKind, Mp2DoubleHybridConfig};
use crate::laplace::{SosFormulation, SosMp2Config};
use crate::lmp2_certificate::{operator_label, Lmp2Certificate};
use crate::rimp2::RiMp2Config;
use crate::scs::{ScsMp2Config, ScsMp2TerfcConfig};

/// The anchor of the RI correlation kinds whose evidence scope is the common
/// one (RHF, all electrons, Coulomb) and nothing else: `mp3` and `laplace-mp2`
/// (every `n_quad` the minimax table supports is run; others are an error).
/// `""` for every other kind.
fn scope_only_anchor(kind: &str) -> &'static str {
    match kind {
        "mp3" => "MP3",
        "laplace-mp2" => "Laplace RI-MP2",
        _ => "",
    }
}

/// The certificate every RI correlation method shares: `kind` on
/// `reference`, the auxiliary basis, frozen core, and the correlation
/// operator when it is not Coulomb. `mp3` and `laplace-mp2` on an RHF
/// reference with every electron correlated carry their energy anchor.
pub fn ri_correlation_certificate(
    kind: &str,
    reference: &Certificate,
    aux: &str,
    frozen_core: usize,
    op: &Operator,
) -> Certificate {
    let c = Certificate::for_post_scf(kind, reference)
        .with_ri(aux)
        .with_frozen_core(frozen_core);
    let in_scope = rhf_all_electron_coulomb(reference, frozen_core, op);
    let c = with_anchor_in_scope(c, scope_only_anchor(kind), in_scope);
    if op.kind == OperatorKind::Coulomb {
        c
    } else {
        c.with_approximation("operator", operator_label(op))
    }
}

/// The headline energy anchor of canonical RI-MP2 on the SCF reference
/// that `reference` certifies: `RI-MP2` on RHF, `U-RI-MP2` on UHF, none on
/// any other reference.
pub fn rimp2_energy_anchor(reference: &Certificate) -> Option<&'static EnergyAnchor> {
    match reference.capability.as_str() {
        "rhf" => energy_anchor("RI-MP2"),
        "uhf" => energy_anchor("U-RI-MP2"),
        _ => None,
    }
}

/// `c` (an RI-MP2 certificate on `reference`) with a warning for each way the
/// run lies outside the anchor of [`rimp2_energy_anchor`]: an element of
/// `mol`, the orbital basis `basis`, the auxiliary basis `aux`. A `c` that
/// does not carry that anchor (its settings are outside the anchor's scope,
/// see [`rimp2_certificate`]) is returned as is. The grade is not changed.
pub fn with_rimp2_envelope(
    c: Certificate,
    reference: &Certificate,
    mol: &Molecule,
    basis: &ferric_core::basis::BasisSet,
    aux: &ferric_core::basis::BasisSet,
) -> Certificate {
    let Some(a) = rimp2_energy_anchor(reference) else {
        return c;
    };
    with_anchor_envelope(c, a.method, mol, basis, Some(aux))
}

/// `c` with a warning for each way the run lies outside the anchor named
/// `method` (a [`EnergyAnchor::method`], e.g. `"CCSD"`): an element of `mol`,
/// the orbital basis `basis`, the auxiliary basis `aux` (`None` for methods
/// without one). A `c` that does not carry that anchor (its settings are
/// outside the anchor's scope) is returned as is: a run outside the anchor's
/// settings has no anchor to be outside of. The grade is not changed.
pub fn with_anchor_envelope(
    c: Certificate,
    method: &str,
    mol: &Molecule,
    basis: &ferric_core::basis::BasisSet,
    aux: Option<&ferric_core::basis::BasisSet>,
) -> Certificate {
    let Some(a) = energy_anchor(method) else {
        return c;
    };
    if c.measured_error.as_ref() != Some(&a.measured_error()) {
        return c;
    }
    let zs: Vec<i32> = mol.atoms.iter().map(|at| at.z).collect();
    c.with_envelope(
        a,
        &RunEnvelope {
            atomic_numbers: &zs,
            basis: Some(basis),
            aux_basis: aux,
            functional: None,
        },
    )
}

/// `c` with the anchor named `method` attached when `in_scope`. The anchors of
/// the correlated methods other than RI-MP2 are attached by this one rule:
/// only inside the settings their evidence test runs.
pub fn with_anchor_in_scope(c: Certificate, method: &str, in_scope: bool) -> Certificate {
    match energy_anchor(method) {
        Some(a) if in_scope => c.with_energy_anchor(a),
        _ => c,
    }
}

/// Whether a correlation run is on an RHF `reference` with every electron
/// correlated and the Coulomb operator: the common part of the scope of the
/// closed-shell anchors (MP3, CCSD, attenuated, SCS and Laplace MP2), whose
/// evidence tests run an exact-J/K RHF and no frozen core.
fn rhf_all_electron_coulomb(reference: &Certificate, frozen_core: usize, op: &Operator) -> bool {
    reference.capability == "rhf" && frozen_core == 0 && op.kind == OperatorKind::Coulomb
}

/// Whether a run with `cfg` and `op` is what the anchor's evidence test runs:
/// the Coulomb operator, no κ regularization, no separate fitting metric, no
/// 3-index screening (or the threshold 0, which drops nothing), and all
/// electrons correlated on an RHF reference (`validation_basis_coverage.rs`
/// runs `RiMp2Config::default()`). The UHF evidence
/// (`validation_u_rimp2.rs`) runs frozen core 0 on every system and frozen
/// core 2 on HO2, so a UHF reference is in scope at exactly those two counts.
fn in_anchor_scope(reference: &Certificate, cfg: &RiMp2Config, op: &Operator) -> bool {
    op.kind == OperatorKind::Coulomb
        && cfg.kappa.is_none()
        && cfg.metric_op.is_none()
        && cfg.eri3_screen_thresh.is_none_or(|t| t == 0.0)
        && (cfg.frozen_core == 0 || (reference.capability == "uhf" && cfg.frozen_core == 2))
}

/// Canonical RI-MP2 (`rimp2`, closed or open shell) run with `cfg`.
/// `measured_error` is the anchor of [`rimp2_energy_anchor`] when the run is
/// in the scope its evidence covers (see `in_anchor_scope`); otherwise none.
pub fn rimp2_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &RiMp2Config,
    op: &Operator,
    precision: &PrecisionRecord,
) -> Certificate {
    let mut c = ri_correlation_certificate("rimp2", reference, aux, cfg.frozen_core, op)
        .with_precision(precision);
    if let Some(a) = rimp2_energy_anchor(reference).filter(|_| in_anchor_scope(reference, cfg, op))
    {
        c = c.with_energy_anchor(a);
    }
    if let Some(m) = cfg.metric_op.as_ref() {
        c = c.with_approximation("ri-metric", operator_label(m));
    }
    if let Some(t) = cfg.eri3_screen_thresh {
        c = c.with_approximation("eri3-screen", format!("threshold {t:e}"));
    }
    if let Some(k) = cfg.kappa {
        c = c.with_approximation("kappa", format!("kappa-regularized denominators, {k}"));
    }
    c
}

/// Local RI-MP2 (`[local] scheme = "amplitude-threshold"`): the model
/// certificate ([`Lmp2Certificate::certificate`]) on `reference`.
pub fn local_rimp2_certificate(lmp2: &Lmp2Certificate, reference: &Certificate) -> Certificate {
    lmp2.certificate().with_reference(reference)
}

/// Orbital-optimized RI-MP2 (`oo-rimp2`). A run whose orbital optimization
/// did not converge is capped to [`Grade::Ungraded`].
pub fn oo_rimp2_certificate(
    reference: &Certificate,
    aux: &str,
    frozen_core: usize,
    op: &Operator,
    converged: bool,
    iterations: usize,
    grad_norm: f64,
) -> Certificate {
    let c = ri_correlation_certificate("oo-rimp2", reference, aux, frozen_core, op);
    if converged {
        c
    } else {
        c.cap_grade(
            Grade::Ungraded,
            format!(
                "orbital optimization did not converge ({iterations} iterations, \
                 gradient norm {grad_norm:.2e})"
            ),
        )
    }
}

/// Attenuated RI-MP2 (`att-rimp2`) with the erfc operator of `cfg`.
pub fn att_rimp2_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &AttenuatedMp2Config,
) -> Certificate {
    let op = Operator::erfc(cfg.omega);
    let c = ri_correlation_certificate("att-rimp2", reference, aux, cfg.frozen_core, &op);
    let c = with_anchor_in_scope(c, "Attenuated RI-MP2", att_in_anchor_scope(reference, cfg));
    if cfg.scaling == 1.0 {
        c
    } else {
        c.with_approximation("scaling", format!("correlation scaled by {}", cfg.scaling))
    }
}

/// The attenuation parameters (Bohr⁻¹) that `validation_attenuated_mp2.rs`
/// runs besides the default: 0.2, 0.42 and 1.0 (a test compares them with the
/// reference files).
const ATT_EVIDENCE_OMEGAS: [f64; 3] = [0.2, 0.42, 1.0];

/// Whether an attenuated RI-MP2 run is what the anchor's evidence runs: RHF,
/// all electrons correlated, unscaled, unscreened, at an evidence omega.
fn att_in_anchor_scope(reference: &Certificate, cfg: &AttenuatedMp2Config) -> bool {
    reference.capability == "rhf"
        && cfg.frozen_core == 0
        && cfg.scaling == 1.0
        && cfg.screen_thresh.is_none_or(|t| t == 0.0)
        && att_omega_in_evidence(cfg.omega)
}

/// Whether `omega` is one of the evidence's: the default or an entry of
/// `ATT_EVIDENCE_OMEGAS`.
fn att_omega_in_evidence(omega: f64) -> bool {
    let near = |w: f64| (omega - w).abs() < 1e-12;
    near(AttenuatedMp2Config::default().omega) || ATT_EVIDENCE_OMEGAS.into_iter().any(near)
}

/// Whether an SCS-MP2 run is the evidence's: RHF, all electrons, Coulomb and
/// the default spin-component scales.
fn scs_in_anchor_scope(reference: &Certificate, cfg: &ScsMp2Config, op: &Operator) -> bool {
    let defaults = ScsMp2Config::default();
    rhf_all_electron_coulomb(reference, cfg.frozen_core, op)
        && cfg.c_os == defaults.c_os
        && cfg.c_ss == defaults.c_ss
}

/// SCS-MP2 (`scs-mp2`) with the spin-component scales of `cfg`.
pub fn scs_mp2_certificate(reference: &Certificate, aux: &str, cfg: &ScsMp2Config) -> Certificate {
    let op = Operator::coulomb();
    let c = ri_correlation_certificate("scs-mp2", reference, aux, cfg.frozen_core, &op);
    with_anchor_in_scope(c, "SCS-MP2", scs_in_anchor_scope(reference, cfg, &op))
        .with_approximation("scs", format!("c_os = {}, c_ss = {}", cfg.c_os, cfg.c_ss))
}

/// SCS-MP2 with two terfc operators (`scs-mp2-2terfc`).
pub fn scs_mp2_2terfc_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &ScsMp2TerfcConfig,
) -> Certificate {
    ri_correlation_certificate(
        "scs-mp2-2terfc",
        reference,
        aux,
        cfg.frozen_core,
        &Operator::coulomb(),
    )
    .with_approximation("scs", format!("c_os = {}, c_ss = {}", cfg.c_os, cfg.c_ss))
    .with_approximation(
        "operator",
        format!(
            "terfc pair, r0 = {} / {} Bohr (bonded / nonbonded)",
            cfg.r0_bonded, cfg.r0_nonbonded
        ),
    )
}

/// Laplace-transformed RI-MP2 (`laplace-mp2`) on `n_quad` points.
pub fn laplace_mp2_certificate(
    reference: &Certificate,
    aux: &str,
    frozen_core: usize,
    op: &Operator,
    n_quad: usize,
) -> Certificate {
    ri_correlation_certificate("laplace-mp2", reference, aux, frozen_core, op)
        .with_approximation("laplace", format!("{n_quad}-point minimax quadrature"))
}

/// Whether a Laplace SOS-MP2 run is the evidence's: RHF, all electrons,
/// Coulomb, c_os = 1 (the opposite-spin energy) or the default 1.3, on the
/// dense MO or AO formulation.
fn sos_in_anchor_scope(
    reference: &Certificate,
    cfg: &SosMp2Config,
    formulation: SosFormulation,
    op: &Operator,
) -> bool {
    rhf_all_electron_coulomb(reference, cfg.frozen_core, op)
        && (cfg.c_os == 1.0 || cfg.c_os == SosMp2Config::default().c_os)
        && matches!(formulation, SosFormulation::Mo | SosFormulation::Ao)
}

/// `c` with the domain formulation of `formulation` recorded and the grade
/// capped when it is not the dense MO or AO one.
fn with_sos_formulation(c: Certificate, formulation: SosFormulation) -> Certificate {
    match formulation {
        SosFormulation::Mo | SosFormulation::Ao => c,
        SosFormulation::AoSparse(r) => c
            .with_approximation(
                "ao-domain",
                format!("AO pseudo-densities restricted to {r} Bohr Boys domains"),
            )
            .cap_grade(
                Grade::Ungraded,
                "the ao-sparse formulation is not validated",
            ),
        SosFormulation::Local(r) => c
            .with_approximation(
                "local-domain",
                format!("Boys LMO x VV-HV LVO with {r} Bohr virtual domains"),
            )
            .cap_grade(Grade::Ungraded, "the local formulation is not validated"),
    }
}

/// Laplace SOS-MP2 (`laplace-sos-mp2`) with `cfg` and `formulation`.
pub fn sos_mp2_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &SosMp2Config,
    formulation: SosFormulation,
    op: &Operator,
) -> Certificate {
    let c = ri_correlation_certificate("laplace-sos-mp2", reference, aux, cfg.frozen_core, op);
    let c = with_anchor_in_scope(
        c,
        "Laplace SOS-MP2",
        sos_in_anchor_scope(reference, cfg, formulation, op),
    )
    .with_approximation("sos", format!("c_os = {}, c_ss = 0", cfg.c_os))
    .with_approximation(
        "laplace",
        format!("{}-point minimax quadrature", cfg.n_quad),
    );
    with_sos_formulation(c, formulation)
}

/// Record an amplitude-threshold `[local]` scheme at `eps` with the kept
/// amplitude fraction. A finite `eps` is capped to [`Grade::Ungraded`]: the
/// validation rows anchor the local schemes by their `eps = 0` limit only.
pub fn with_amplitude_threshold(c: Certificate, eps: f64, keep_fraction: f64) -> Certificate {
    let c = c.with_approximation(
        "local",
        format!("amplitude-threshold eps = {eps:e}, keep_fraction = {keep_fraction}"),
    );
    if eps == 0.0 {
        c
    } else {
        c.cap_grade(
            Grade::Ungraded,
            format!(
                "finite eps = {eps:e}: the local scheme is anchored at eps = 0 only; its \
                 finite-eps error is a measured map, not validated"
            ),
        )
    }
}

/// dRPA by the drCCD Riccati solve (`drpa`). `local` is `(eps, keep_fraction)`
/// under `[local]`, `None` for the exact method. A fixed point that did not
/// converge is capped to [`Grade::Ungraded`].
#[allow(clippy::too_many_arguments)]
pub fn drpa_certificate(
    reference: &Certificate,
    aux: &str,
    frozen_core: usize,
    op: &Operator,
    local: Option<(f64, f64)>,
    converged: bool,
    iterations: usize,
    relres: f64,
) -> Certificate {
    let mut c = ri_correlation_certificate("drpa", reference, aux, frozen_core, op);
    if let Some((eps, keep)) = local {
        c = with_amplitude_threshold(c, eps, keep);
    }
    if converged {
        c
    } else {
        c.cap_grade(
            Grade::Ungraded,
            format!(
                "drCCD fixed point did not converge (relres {relres:.2e} after {iterations} \
                 iterations)"
            ),
        )
    }
}

/// Attenuated MP2 + VV10 (`mp2-v`) run with `cfg`.
pub fn mp2_v_certificate(
    reference: &Certificate,
    aux: &str,
    cfg: &crate::att_vv10::AttVv10Config,
) -> Certificate {
    let omega = cfg
        .omega
        .map_or(String::new(), |w| format!(", omega = {w} Bohr^-1"));
    ri_correlation_certificate(
        "mp2-v",
        reference,
        aux,
        cfg.frozen_core,
        &Operator::coulomb(),
    )
    .with_approximation(
        "operator",
        format!("{:?}, r0 = {} Bohr{omega}", cfg.attenuator, cfg.r0_bohr).to_lowercase(),
    )
    .with_approximation(
        "vv10",
        format!(
            "{:?} damping, NLC grid ({}, {})",
            cfg.vv10_damping, cfg.nlc_grid.n_radial, cfg.nlc_grid.n_angular
        ),
    )
}

/// MP3 (`mp3`).
pub fn mp3_certificate(
    reference: &Certificate,
    aux: &str,
    frozen_core: usize,
    op: &Operator,
) -> Certificate {
    ri_correlation_certificate("mp3", reference, aux, frozen_core, op)
}

/// The orbital-fitting basis of the double hybrid's Kohn-Sham reference and
/// the correlation aux bases the double-hybrid evidence tests run
/// (`validation_b2plyp_pyscf.rs`, `validation_dsd_pbep86_pyscf.rs`).
const DOUBLE_HYBRID_JK_AUX: &str = "def2-universal-jkfit";

/// The anchored double hybrids: the `method.kind`, its anchor name in
/// `ferric_core::certificate::anchors::ENERGY_ANCHORS` and its parameter set.
const ANCHORED_DOUBLE_HYBRIDS: [(&str, &str, DoubleHybridKind); 2] = [
    ("b2plyp", "B2PLYP", DoubleHybridKind::B2plyp),
    ("dsd-pbep86", "DSD-PBEP86", DoubleHybridKind::DsdPbep86),
];

/// Whether a double-hybrid run of `dk` is what its anchor's evidence runs: a
/// Kohn-Sham `reference` on the flat (75, 110) grid with RI-J and RI-K over
/// def2-universal-jkfit (J tensor stored in f64), the published scaling of
/// `dk`, and frozen core 0 or 1. The pruned production grid and the f32 RI-J
/// default are outside it.
fn in_double_hybrid_anchor_scope(
    reference: &Certificate,
    cfg: &Mp2DoubleHybridConfig,
    dk: DoubleHybridKind,
) -> bool {
    let detail = |name: &str| {
        reference
            .approximations
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.detail.as_str())
    };
    let grid = format!(
        "{} on {}",
        dk.xc_name(),
        ferric_scf::certificate::grid_text(&ferric_dft::grid::AtomicGridConfig::default())
    );
    let published = dk.mp2_config();
    reference.capability == "ksdft"
        && reference
            .approximations
            .iter()
            .all(|a| matches!(a.name.as_str(), "ri-j" | "ri-k" | "xc-grid"))
        && detail("xc-grid") == Some(grid.as_str())
        && detail("ri-j").is_some_and(|d| {
            d.starts_with(&format!("aux {DOUBLE_HYBRID_JK_AUX}, f64 tensor storage"))
        })
        && detail("ri-k").is_some_and(|d| d.starts_with(&format!("aux {DOUBLE_HYBRID_JK_AUX}")))
        && cfg.c_os == published.c_os
        && cfg.c_ss == published.c_ss
        && cfg.frozen_core <= 1
}

/// OP-MP3 (`method.kind = "opmp3"`) on `reference`: RI with `aux` when given
/// (exact four-centre otherwise), `frozen_core`, and the SSD-OPMP3 same-spin
/// damping `x_ss`.
pub fn opmp3_certificate(
    reference: &Certificate,
    aux: Option<&str>,
    frozen_core: usize,
    x_ss: f64,
) -> Certificate {
    let c = Certificate::for_post_scf("opmp3", reference);
    let c = match aux {
        Some(a) => c.with_ri(a),
        None => c,
    };
    c.with_frozen_core(frozen_core)
        .with_approximation("ssd-opmp3", format!("x_ss = {x_ss}"))
}

/// MP2-based double hybrid (`kind` is `b2plyp` or `dsd-pbep86`) run with
/// `cfg`: scaled RI-MP2 on the functional's own Kohn-Sham `reference`.
/// Each kind carries its anchor inside the scope of its evidence (see
/// `in_double_hybrid_anchor_scope`).
pub fn mp2_double_hybrid_certificate(
    kind: &str,
    reference: &Certificate,
    aux: &str,
    cfg: &Mp2DoubleHybridConfig,
) -> Certificate {
    let base = double_hybrid_base(kind, reference, aux, cfg);
    match scoped_double_hybrid_anchor(kind, reference, cfg) {
        Some(anchor) => with_anchor_in_scope(base, anchor, true),
        None => base,
    }
}

/// The anchor name of double-hybrid `kind` when `reference` and `cfg` are
/// inside the scope of that anchor's evidence; `None` for a kind without an
/// anchor or a run outside it.
fn scoped_double_hybrid_anchor(
    kind: &str,
    reference: &Certificate,
    cfg: &Mp2DoubleHybridConfig,
) -> Option<&'static str> {
    let (_, anchor, dk) = ANCHORED_DOUBLE_HYBRIDS
        .iter()
        .find(|(k, _, _)| *k == kind)?;
    in_double_hybrid_anchor_scope(reference, cfg, *dk).then_some(*anchor)
}

/// The double-hybrid certificate without an anchor: the reference, the
/// correlation aux basis, the frozen core and the PT2 scaling.
fn double_hybrid_base(
    kind: &str,
    reference: &Certificate,
    aux: &str,
    cfg: &Mp2DoubleHybridConfig,
) -> Certificate {
    let scaling = format!("c_os = {}, c_ss = {}", cfg.c_os, cfg.c_ss);
    Certificate::for_post_scf(kind, reference)
        .with_ri(aux)
        .with_frozen_core(cfg.frozen_core)
        .with_approximation("spin-component-scaling", scaling)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::certificate::{BuildStamp, GRADE_SOURCE_PAGE};

    fn rhf() -> Certificate {
        Certificate::for_method_kind("rhf")
            .with_build(BuildStamp::new("v", "c", Some(false)))
            .with_approximation("ri-j", "aux def2-universal-jkfit")
    }

    fn names(c: &Certificate) -> Vec<&str> {
        c.approximations.iter().map(|a| a.name.as_str()).collect()
    }

    #[test]
    fn rimp2_carries_grade_aux_and_settings() {
        let cfg = RiMp2Config {
            frozen_core: 1,
            kappa: Some(1.1),
            ..Default::default()
        };
        let c = rimp2_certificate(
            &rhf(),
            "cc-pvdz-ri",
            &cfg,
            &Operator::coulomb(),
            &PrecisionRecord::f64(),
        );
        assert_eq!(c.capability, "rimp2");
        assert_eq!(c.grade, Grade::Proven);
        assert_eq!(c.grade_source.as_deref(), Some(GRADE_SOURCE_PAGE));
        assert_eq!(names(&c), ["scf ri-j", "ri", "frozen-core", "kappa"]);
        assert_eq!(c.approximations[1].detail, "aux cc-pvdz-ri");
        assert_eq!(c.build.as_ref().map(|b| b.commit.as_str()), Some("c"));
        assert!(c.precision.is_some());
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
        // A non-Coulomb correlation operator is named.
        let e = rimp2_certificate(
            &rhf(),
            "x",
            &RiMp2Config::default(),
            &Operator::erfc(0.5),
            &PrecisionRecord::f64(),
        );
        assert_eq!(names(&e), ["scf ri-j", "ri", "operator"]);
        assert_eq!(e.approximations[2].detail, "erfc(omega=0.5)");
    }

    #[test]
    fn rimp2_measured_error_follows_the_reference_spin() {
        let run = |reference: &Certificate| {
            rimp2_certificate(
                reference,
                "cc-pvdz-ri",
                &RiMp2Config::default(),
                &Operator::coulomb(),
                &PrecisionRecord::f64(),
            )
        };
        let closed = run(&rhf());
        let e = closed.measured_error.as_ref().expect("RI-MP2 anchor");
        assert_eq!(
            (e.quantity.as_str(), e.value),
            ("correlation energy", 2.4e-10)
        );
        assert_eq!(
            e.evidence,
            "crates/ferric-mp2/tests/validation_basis_coverage.rs"
        );
        let open = run(&Certificate::for_method_kind("uhf"));
        assert_eq!(open.measured_error.as_ref().unwrap().value, 9.2e-10);
        // A KS reference has no RI-MP2 anchor.
        assert!(run(&Certificate::for_method_kind("ksdft"))
            .measured_error
            .is_none());
    }

    /// The anchor is attached only inside its evidence's settings: each
    /// setting outside them drops it on its own. Frozen core drops it on an
    /// RHF reference; on UHF it keeps it at the evidence's counts 0 and 2 only.
    #[test]
    fn rimp2_anchor_is_gated_on_the_evidence_settings() {
        let cert = |reference: &Certificate, cfg: RiMp2Config, op: Operator| {
            rimp2_certificate(reference, "cc-pvdz-ri", &cfg, &op, &PrecisionRecord::f64())
        };
        let uhf = Certificate::for_method_kind("uhf");
        let coul = Operator::coulomb;
        let d = RiMp2Config::default;
        assert!(cert(&rhf(), d(), coul()).measured_error.is_some());
        let outside = [
            ("kappa", rhf(), d().with_kappa(1.1), coul()),
            ("erfc", rhf(), d(), Operator::erfc(0.5)),
            ("rhf frozen core", rhf(), d().with_frozen_core(1), coul()),
            (
                "metric",
                rhf(),
                RiMp2Config {
                    metric_op: Some(Operator::erfc(0.5)),
                    ..d()
                },
                coul(),
            ),
            (
                "eri3 screen",
                rhf(),
                RiMp2Config {
                    eri3_screen_thresh: Some(1e-10),
                    ..d()
                },
                coul(),
            ),
            ("uhf kappa", uhf.clone(), d().with_kappa(1.1), coul()),
            (
                "uhf frozen core 1",
                uhf.clone(),
                d().with_frozen_core(1),
                coul(),
            ),
            (
                "uhf frozen core 3",
                uhf.clone(),
                d().with_frozen_core(3),
                coul(),
            ),
        ];
        for (what, reference, cfg, op) in outside {
            let c = cert(&reference, cfg, op);
            assert!(c.measured_error.is_none(), "{what}: anchor attached");
            let n_atom = Molecule::parse_xyz("1\n\nN 0 0 0\n", 0, 4).unwrap();
            let b = ferric_core::basis::bundled("6-31g").unwrap();
            assert_eq!(
                with_rimp2_envelope(c.clone(), &reference, &n_atom, &b, &b),
                c,
                "{what}: envelope warnings without an anchor"
            );
        }
        let inside = [
            ("uhf all electron", uhf.clone(), d()),
            ("uhf frozen core 2", uhf.clone(), d().with_frozen_core(2)),
            (
                "eri3 screen 0",
                rhf(),
                RiMp2Config {
                    eri3_screen_thresh: Some(0.0),
                    ..d()
                },
            ),
        ];
        for (what, reference, cfg) in inside {
            assert!(
                cert(&reference, cfg, coul()).measured_error.is_some(),
                "{what}: anchor missing"
            );
        }
    }

    #[test]
    fn rimp2_envelope_checks_elements_and_both_basis_sets() {
        let water = Molecule::parse_xyz("3\n\nO 0 0 0\nH 0 0 1\nH 0 1 0\n", 0, 1).unwrap();
        let c = rimp2_certificate(
            &rhf(),
            "cc-pvdz-ri",
            &RiMp2Config::default(),
            &Operator::coulomb(),
            &PrecisionRecord::f64(),
        );
        // Covered: water / cc-pVDZ / cc-pvdz-ri (an alias of cc-pvdz-rifit).
        let b = |n: &str| ferric_core::basis::bundled(n).unwrap();
        let inside =
            with_rimp2_envelope(c.clone(), &rhf(), &water, &b("cc-pVDZ"), &b("cc-pvdz-ri"));
        assert_eq!(inside, c);
        let h2s = Molecule::parse_xyz("3\n\nS 0 0 0\nH 0 0 1\nH 0 1 0\n", 0, 1).unwrap();
        let out = with_rimp2_envelope(
            c.clone(),
            &rhf(),
            &h2s,
            &b("6-31g"),
            &b("aug-cc-pvdz-jkfit"),
        );
        assert_eq!(out.grade, c.grade);
        let w: Vec<&str> = out.warnings.iter().map(String::as_str).collect();
        assert_eq!(w.len(), 3, "{w:?}");
        assert!(
            w[0].starts_with("element S (covered: H, C, N, O, F, Pd)"),
            "{w:?}"
        );
        assert!(
            w[1].starts_with("basis 6-31g is outside the RI-MP2"),
            "{w:?}"
        );
        assert!(
            w[2].starts_with("auxiliary basis aug-cc-pvdz-jkfit is outside the RI-MP2"),
            "{w:?}"
        );
    }

    /// Each of the six closed-shell anchors is attached inside the settings
    /// its evidence test runs and dropped by each setting outside them.
    #[test]
    fn closed_shell_anchors_are_gated_on_the_evidence_settings() {
        let uhf = Certificate::for_method_kind("uhf");
        let ks = Certificate::for_method_kind("ksdft");
        let coul = Operator::coulomb;
        let has = |c: &Certificate, evidence: &str| {
            c.measured_error.as_ref().map(|m| m.evidence.as_str()) == Some(evidence)
        };
        let none = |c: Certificate, what: &str| {
            assert!(c.measured_error.is_none(), "{what}: anchor attached");
        };
        let sos = |r: &Certificate, cfg: &SosMp2Config, f: SosFormulation, op: &Operator| {
            sos_mp2_certificate(r, "a", cfg, f, op)
        };
        let att = |r: &Certificate, cfg: &AttenuatedMp2Config| att_rimp2_certificate(r, "a", cfg);

        // MP3
        assert!(has(
            &mp3_certificate(&rhf(), "a", 0, &coul()),
            "crates/ferric-cc/tests/validation_cc.rs"
        ));
        none(mp3_certificate(&rhf(), "a", 1, &coul()), "mp3 frozen core");
        none(
            mp3_certificate(&rhf(), "a", 0, &Operator::erfc(0.3)),
            "mp3 erfc",
        );
        none(mp3_certificate(&uhf, "a", 0, &coul()), "mp3 uhf");
        none(mp3_certificate(&ks, "a", 0, &coul()), "mp3 ks");

        // Attenuated RI-MP2: the evidence sweeps omega over four values.
        let a_ev = "crates/ferric-mp2/tests/validation_attenuated_mp2.rs";
        let dflt = AttenuatedMp2Config::default;
        assert!(has(&att(&rhf(), &dflt()), a_ev));
        for omega in [0.2, 0.42, 1.0] {
            let cfg = AttenuatedMp2Config { omega, ..dflt() };
            assert!(has(&att(&rhf(), &cfg), a_ev), "omega {omega}");
        }
        for (what, cfg) in [
            (
                "omega",
                AttenuatedMp2Config {
                    omega: 0.3,
                    ..dflt()
                },
            ),
            (
                "frozen core",
                AttenuatedMp2Config {
                    frozen_core: 1,
                    ..dflt()
                },
            ),
            (
                "scaling",
                AttenuatedMp2Config {
                    scaling: 1.2,
                    ..dflt()
                },
            ),
            (
                "screen",
                AttenuatedMp2Config {
                    screen_thresh: Some(1e-8),
                    ..dflt()
                },
            ),
        ] {
            none(att(&rhf(), &cfg), &format!("att {what}"));
        }
        none(att(&uhf, &dflt()), "att uhf");

        // SCS-MP2: the default coefficients.
        let s_ev = "crates/ferric-mp2/tests/validation_scs_mp2.rs";
        let scs = |r: &Certificate, cfg: &ScsMp2Config| scs_mp2_certificate(r, "a", cfg);
        assert!(has(&scs(&rhf(), &ScsMp2Config::default()), s_ev));
        for (what, cfg) in [
            (
                "c_os",
                ScsMp2Config {
                    c_os: 1.0,
                    ..Default::default()
                },
            ),
            (
                "c_ss",
                ScsMp2Config {
                    c_ss: 0.0,
                    ..Default::default()
                },
            ),
            (
                "frozen core",
                ScsMp2Config {
                    frozen_core: 1,
                    ..Default::default()
                },
            ),
        ] {
            none(scs(&rhf(), &cfg), &format!("scs {what}"));
        }
        none(scs(&uhf, &ScsMp2Config::default()), "scs uhf");

        // Laplace-MP2: every supported n_quad.
        let l_ev = "crates/ferric-mp2/tests/validation_laplace_mp2.rs";
        for n in [3, 5, 7] {
            assert!(has(
                &laplace_mp2_certificate(&rhf(), "a", 0, &coul(), n),
                l_ev
            ));
        }
        none(
            laplace_mp2_certificate(&rhf(), "a", 1, &coul(), 7),
            "laplace frozen core",
        );
        none(
            laplace_mp2_certificate(&rhf(), "a", 0, &Operator::erfc(0.3), 7),
            "laplace erfc",
        );
        none(
            laplace_mp2_certificate(&uhf, "a", 0, &coul(), 7),
            "laplace uhf",
        );

        // Laplace SOS-MP2.
        let o_ev = "crates/ferric-mp2/tests/validation_sos_mp2.rs";
        let d = SosMp2Config::default;
        let mo = SosFormulation::Mo;
        assert!(has(&sos(&rhf(), &d(), mo, &coul()), o_ev));
        assert!(has(
            &sos(
                &rhf(),
                &SosMp2Config {
                    c_os: 1.0,
                    n_quad: 3,
                    ..d()
                },
                SosFormulation::Ao,
                &coul()
            ),
            o_ev
        ));
        for (what, cfg, f) in [
            ("c_os", SosMp2Config { c_os: 1.7, ..d() }, mo),
            (
                "frozen core",
                SosMp2Config {
                    frozen_core: 1,
                    ..d()
                },
                mo,
            ),
            ("ao-sparse", d(), SosFormulation::AoSparse(6.0)),
            ("local", d(), SosFormulation::Local(6.0)),
        ] {
            none(sos(&rhf(), &cfg, f, &coul()), &format!("sos {what}"));
        }
        none(sos(&rhf(), &d(), mo, &Operator::erfc(0.3)), "sos erfc");
        none(sos(&uhf, &d(), mo, &coul()), "sos uhf");
    }

    /// The attenuation values the certificate gates on are those of the
    /// reference files the evidence test reads.
    #[test]
    fn attenuated_evidence_omegas_are_the_reference_blocks() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for system in ["h2o", "nh3"] {
            let path = root.join(format!(
                "testdata/reference/validation/attenuated_mp2/{system}_aug-cc-pvdz.json"
            ));
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let omegas: Vec<f64> = v["attenuated"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b["omega_bohr_inv"].as_f64().unwrap())
                .collect();
            assert_eq!(omegas.len(), 4, "{system}");
            for w in omegas {
                assert!(att_omega_in_evidence(w), "{system}: omega {w}");
            }
        }
    }

    /// The envelope warns outside an attached anchor's covered elements and
    /// basis sets, and says nothing for a run with no anchor.
    #[test]
    fn anchor_envelope_warns_only_on_an_anchored_run() {
        let nh3 = Molecule::parse_xyz("4\n\nN 0 0 0\nH 0 0 1\nH 0 1 0\nH 1 0 0\n", 0, 1).unwrap();
        let b = ferric_core::basis::bundled("6-31g").unwrap();
        let aux = ferric_core::basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let anchored = laplace_mp2_certificate(&rhf(), "cc-pvdz-ri", 0, &op, 7);
        let w = with_anchor_envelope(anchored, "Laplace RI-MP2", &nh3, &b, Some(&aux));
        let msgs: Vec<&str> = w.warnings.iter().map(String::as_str).collect();
        assert_eq!(msgs.len(), 2, "{msgs:?}");
        assert!(
            msgs[0].starts_with("element N (covered: H, C, O)"),
            "{msgs:?}"
        );
        assert!(
            msgs[1].starts_with("basis 6-31g is outside the Laplace RI-MP2"),
            "{msgs:?}"
        );
        let unanchored = laplace_mp2_certificate(&rhf(), "cc-pvdz-ri", 1, &op, 7);
        let w = with_anchor_envelope(unanchored.clone(), "Laplace RI-MP2", &nh3, &b, Some(&aux));
        assert_eq!(w, unanchored);
    }

    #[test]
    fn family_grades_come_from_the_table() {
        let r = rhf();
        let op = Operator::coulomb();
        let cases = [
            (mp3_certificate(&r, "a", 0, &op), "mp3", Grade::Proven),
            (
                oo_rimp2_certificate(&r, "a", 0, &op, true, 5, 1e-7),
                "oo-rimp2",
                Grade::ProvenNarrow,
            ),
            (
                att_rimp2_certificate(&r, "a", &AttenuatedMp2Config::default()),
                "att-rimp2",
                Grade::Proven,
            ),
            (
                scs_mp2_certificate(&r, "a", &ScsMp2Config::default()),
                "scs-mp2",
                Grade::Proven,
            ),
            (
                laplace_mp2_certificate(&r, "a", 0, &op, 7),
                "laplace-mp2",
                Grade::Proven,
            ),
            (
                sos_mp2_certificate(&r, "a", &SosMp2Config::default(), SosFormulation::Mo, &op),
                "laplace-sos-mp2",
                Grade::Proven,
            ),
            (
                mp2_v_certificate(&r, "a", &crate::att_vv10::AttVv10Config::default()),
                "mp2-v",
                Grade::ProvenNarrow,
            ),
        ];
        for (c, kind, g) in cases {
            assert_eq!(c.capability, kind);
            assert_eq!(c.grade, g, "{kind}");
            assert_eq!(
                c.grade,
                ferric_core::certificate::method_kind_grade(kind),
                "{kind}"
            );
        }
    }

    #[test]
    fn sos_domain_formulations_are_not_graded() {
        let r = rhf();
        let op = Operator::coulomb();
        let cfg = SosMp2Config::default();
        for f in [SosFormulation::AoSparse(4.0), SosFormulation::Local(6.0)] {
            let c = sos_mp2_certificate(&r, "a", &cfg, f, &op);
            assert_eq!(c.grade, Grade::Ungraded, "{f:?}");
        }
        let exact = sos_mp2_certificate(&r, "a", &cfg, SosFormulation::Ao, &op);
        assert_eq!(exact.grade, Grade::Proven);
    }

    #[test]
    fn method_settings_are_recorded() {
        let r = rhf();
        let op = Operator::coulomb();
        let l = laplace_mp2_certificate(&r, "a", 0, &op, 5);
        assert_eq!(l.approximations[2].detail, "5-point minimax quadrature");
        let s = sos_mp2_certificate(
            &r,
            "a",
            &SosMp2Config::default(),
            SosFormulation::AoSparse(4.0),
            &op,
        );
        assert_eq!(names(&s), ["scf ri-j", "ri", "sos", "laplace", "ao-domain"]);
        let a = att_rimp2_certificate(
            &r,
            "a",
            &AttenuatedMp2Config {
                omega: 0.25,
                ..Default::default()
            },
        );
        assert_eq!(a.approximations[2].detail, "erfc(omega=0.25)");
        let scs = scs_mp2_certificate(
            &r,
            "a",
            &ScsMp2Config {
                c_os: 1.2,
                c_ss: 0.5,
                ..Default::default()
            },
        );
        assert_eq!(scs.approximations[2].detail, "c_os = 1.2, c_ss = 0.5");
    }

    #[test]
    fn drpa_exact_local_and_unconverged() {
        let op = Operator::coulomb();
        let exact = drpa_certificate(&rhf(), "a", 0, &op, None, true, 9, 1e-9);
        assert_eq!(exact.capability, "drpa");
        assert_eq!(exact.grade, Grade::ProvenNarrow);
        assert!(!names(&exact).contains(&"local"));
        let anchor = drpa_certificate(&rhf(), "a", 0, &op, Some((0.0, 1.0)), true, 9, 1e-9);
        assert_eq!(anchor.grade, Grade::ProvenNarrow, "eps = 0 is the anchor");
        assert_eq!(
            anchor.approximations[2].detail,
            "amplitude-threshold eps = 0e0, keep_fraction = 1"
        );
        let finite = drpa_certificate(&rhf(), "a", 0, &op, Some((1e-4, 0.3)), true, 9, 1e-9);
        assert_eq!(finite.grade, Grade::Ungraded);
        assert!(finite.warnings[0].contains("finite eps = 1e-4"));
        let bad = drpa_certificate(&rhf(), "a", 0, &op, None, false, 40, 1e-3);
        assert_eq!(bad.grade, Grade::Ungraded);
        assert!(bad.warnings[0].contains("did not converge (relres 1.00e-3 after 40"));
    }

    #[test]
    fn unconverged_orbital_optimization_or_reference_is_ungraded() {
        let op = Operator::coulomb();
        let c = oo_rimp2_certificate(&rhf(), "a", 0, &op, false, 50, 3e-3);
        assert_eq!(c.grade, Grade::Ungraded);
        assert!(
            c.warnings[0].contains("orbital optimization did not converge (50 iterations"),
            "{:?}",
            c.warnings
        );
        let bad = rhf().cap_grade(Grade::Ungraded, "SCF did not converge");
        let m = mp3_certificate(&bad, "a", 0, &op);
        assert_eq!(m.grade, Grade::Ungraded);
    }

    /// A Kohn-Sham reference of `xc` as the double-hybrid evidence tests run it.
    fn dh_reference(xc: &str) -> Certificate {
        let grid =
            ferric_scf::certificate::grid_text(&ferric_dft::grid::AtomicGridConfig::default());
        Certificate::for_method_kind("ksdft")
            .with_build(BuildStamp::new("v", "c", Some(false)))
            .with_approximation(
                "ri-j",
                "aux def2-universal-jkfit, f64 tensor storage (requested f64: explicit)",
            )
            .with_approximation("ri-k", "aux def2-universal-jkfit, f64 tensor storage")
            .with_approximation("xc-grid", format!("{xc} on {grid}"))
    }

    #[test]
    fn double_hybrids_carry_their_grade_and_scaling() {
        let ks = Certificate::for_method_kind("ksdft")
            .with_build(BuildStamp::new("v", "c", Some(false)))
            .with_approximation("xc-grid", "(75, 302)");
        for (kind, dk) in [
            ("b2plyp", DoubleHybridKind::B2plyp),
            ("dsd-pbep86", DoubleHybridKind::DsdPbep86),
        ] {
            let mut cfg = dk.mp2_config();
            cfg.frozen_core = 1;
            let c = mp2_double_hybrid_certificate(kind, &ks, "cc-pvdz-ri", &cfg);
            assert_eq!(c.capability, kind);
            assert_eq!(c.grade, Grade::Proven, "{kind}");
            assert_eq!(c.grade, ferric_core::certificate::method_kind_grade(kind));
            assert_eq!(
                names(&c),
                ["scf xc-grid", "ri", "frozen-core", "spin-component-scaling"]
            );
            assert_eq!(
                c.approximations[3].detail,
                format!("c_os = {}, c_ss = {}", cfg.c_os, cfg.c_ss)
            );
            assert!(c.measured_error.is_none(), "{kind}: outside the evidence");
        }
        // An unconverged reference caps the grade below Spike.
        let bad = ks.cap_grade(Grade::Ungraded, "SCF did not converge");
        let c = mp2_double_hybrid_certificate(
            "b2plyp",
            &bad,
            "a",
            &DoubleHybridKind::B2plyp.mp2_config(),
        );
        assert_eq!(c.grade, Grade::Ungraded);
    }

    /// Each double hybrid's anchor rides only on the settings its evidence runs.
    #[test]
    fn double_hybrid_anchors_are_attached_inside_the_evidence_scope_only() {
        for (kind, anchor_name, dk) in ANCHORED_DOUBLE_HYBRIDS {
            let xc = dk.xc_name();
            let cert = |reference: &Certificate, fc: usize| {
                let mut cfg = dk.mp2_config();
                cfg.frozen_core = fc;
                mp2_double_hybrid_certificate(kind, reference, "cc-pvdz-ri", &cfg)
            };
            let anchor = energy_anchor(anchor_name).unwrap().measured_error();
            for fc in [0, 1] {
                assert_eq!(
                    cert(&dh_reference(xc), fc).measured_error,
                    Some(anchor.clone()),
                    "{kind} fc{fc}"
                );
            }
            assert!(
                cert(&dh_reference(xc), 2).measured_error.is_none(),
                "{kind}: frozen core 2"
            );
            let pruned = Certificate::for_method_kind("ksdft")
                .with_approximation("ri-j", "aux def2-universal-jkfit, f64 tensor storage")
                .with_approximation("ri-k", "aux def2-universal-jkfit, f64 tensor storage")
                .with_approximation(
                    "xc-grid",
                    format!("{xc} on (75, 110), nwchem, weight cut 1e-14"),
                );
            assert!(
                cert(&pruned, 0).measured_error.is_none(),
                "{kind}: pruned grid"
            );
            let f32_j = Certificate::for_method_kind("ksdft")
                .with_approximation("ri-j", "aux def2-universal-jkfit, f32 tensor storage")
                .with_approximation("ri-k", "aux def2-universal-jkfit, f64 tensor storage");
            assert!(cert(&f32_j, 0).measured_error.is_none(), "{kind}: f32 RI-J");
            assert!(cert(&Certificate::for_method_kind("rhf"), 0)
                .measured_error
                .is_none());
            // A reference run on the other functional is not this evidence.
            let other = if kind == "b2plyp" {
                "DSD-PBEP86"
            } else {
                "B2PLYP"
            };
            assert!(
                cert(&dh_reference(other), 0).measured_error.is_none(),
                "{kind}: other functional"
            );
            // Another kind's scaling is not this evidence.
            let mut wrong = dk.mp2_config();
            wrong.c_ss += 0.01;
            let w = mp2_double_hybrid_certificate(kind, &dh_reference(xc), "x", &wrong);
            assert!(w.measured_error.is_none(), "{kind}: altered scaling");
        }
    }
}
