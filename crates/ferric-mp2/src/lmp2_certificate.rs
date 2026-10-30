//! The model certificate of an amplitude-threshold local MP2 result: what
//! was computed (scheme, threshold, maps, precision, solver state), how far
//! the result is validated, and every way this run sits outside the
//! measured record. CLI JSON (`"local"`) and Python (`result.local`) both
//! serialize [`fields`](crate::lmp2_certificate::Lmp2Certificate::fields),
//! so the two surfaces cannot drift; each adds the build commit/dirty flag
//! (the library does not link `ferric-build-info`).
//!
//! The ε-trust line quotes the MEASURED local-error record (C4/cc-pVDZ and
//! C8 at ε = 1e-3, Coulomb operator; `wiki/amplitude-threshold-lmp2.md`;
//! memory "local error is superlinear in eps"). It is a record of
//! measurements on n-alkanes, not a bound, and it was measured for the
//! Coulomb operator only: a run on any other operator (`att-rimp2`, erfc)
//! gets the same table plus an explicit caveat and warning.

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_integrals::operator::{Operator, OperatorKind};

use crate::lmp2_amplitude::{
    AmplitudeLmp2Config, AmplitudeLmp2Result, CgRtolOutcome, CgRtolSource,
    LMP2_CG_ENERGY_ERROR_TARGET,
};
use crate::lmp2_direct::{DirectConfig, DirectStats};

/// Largest ε whose MEASURED recovery is comparable to ORCA DLPNO-MP2
/// NormalPNO (C4/cc-pVDZ: 99.980% at 1e-5 vs NormalPNO's 99.96-99.98%).
/// One system, one basis, no independent finite-ε reference: a comparison
/// point, NOT a validated range (the name is kept for API stability).
pub const VALIDATED_EPS_MAX: f64 = 1e-5;

/// Heuristic "alkane-like 1-D" cut on √(λ₂/λ₁) of the unweighted atomic
/// gyration tensor. Chosen so the measured n-alkane series passes (C4 0.48,
/// C8 0.28, C20 0.12) and benzene (1.00) does not. A shape label, not a
/// locality measure.
pub const ONE_D_RATIO_MAX: f64 = 0.5;

/// Smallest orbital-basis primitive exponent below which the basis counts
/// as diffuse. The measured record is 6-31G / cc-pVDZ (smallest exponents
/// ~0.12-0.17 for H/C); aug-cc-pVDZ (0.03-0.05) is flagged.
pub const DIFFUSE_EXPONENT_MIN: f64 = 0.08;

/// The measured C4/cc-pVDZ recovery of the canonical DF-MP2 correlation
/// energy, as (ε, percent as recorded).
pub const MEASURED_RECOVERY_C4: [(f64, &str); 3] =
    [(1e-5, "99.980"), (1e-4, "99.671"), (1e-3, "94.99")];

/// The measured C8 recovery at ε = 1e-3 (the size trend: it LOSES more
/// than C4 at the same ε), as (ε, percent as recorded).
pub const MEASURED_RECOVERY_C8: (f64, &str) = (1e-3, "93.81");

/// Log-log slope of the C4 lost fraction per decade of ε, from
/// [`MEASURED_RECOVERY_C4`]: 0.020% -> 0.329% -> 5.01% lost gives 1.22 and
/// 1.18, i.e. ~1.2 (superlinear). The 1.2-1.8 range sometimes quoted is the
/// octane local-dRPA record, not MP2.
pub const MEASURED_SLOPE_C4: &str = "~1.2 (1.22, 1.18)";

/// Short label of the two-electron operator, for the certificate.
pub fn operator_label(op: &Operator) -> String {
    match op.kind {
        OperatorKind::Coulomb => "coulomb".to_string(),
        OperatorKind::ErfcCoulomb => format!("erfc(omega={})", op.omega),
        OperatorKind::ErfCoulomb => format!("erf(omega={})", op.omega),
        k => format!("{k:?}(omega={}, distance={})", op.omega, op.distance).to_lowercase(),
    }
}

/// The integral-direct locality maps of a run.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectMaps {
    pub aux_radius: f64,
    pub virt_radius: Option<f64>,
    pub ao_tail: f64,
    pub schwarz_skip: f64,
    pub batch_merge: usize,
    pub virt_schwarz_kappa: Option<f64>,
}

impl From<&DirectConfig> for DirectMaps {
    fn from(d: &DirectConfig) -> Self {
        Self {
            aux_radius: d.aux_radius_bohr,
            virt_radius: d.virt_radius_bohr,
            ao_tail: d.ao_tail,
            schwarz_skip: d.schwarz_skip,
            batch_merge: d.batch_merge,
            virt_schwarz_kappa: d.virt_schwarz_kappa,
        }
    }
}

/// Stage-5 pair-fit work counters of an integral-direct run (from
/// [`DirectStats`]): pair-domain groups, mean distinct occupieds per group,
/// triangular solves, and whitenings reused from the κ screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairFitCounters {
    pub n_pair_groups: usize,
    pub n_distinct_occ_per_group: f64,
    pub n_trsm: usize,
    pub n_w_reused: usize,
}

impl From<&DirectStats> for PairFitCounters {
    fn from(st: &DirectStats) -> Self {
        Self {
            n_pair_groups: st.n_pair_groups,
            n_distinct_occ_per_group: st.n_distinct_occ_per_group,
            n_trsm: st.n_trsm,
            n_w_reused: st.n_w_reused,
        }
    }
}

/// Geometry/basis classification the out-of-range warnings use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SystemShape {
    /// √(λ₂/λ₁) of the atomic gyration tensor (0 for a single atom).
    pub gyration_ratio: f64,
    /// Smallest primitive exponent of the orbital basis over the molecule.
    pub min_exponent: f64,
}

impl SystemShape {
    pub fn new(mol: &Molecule, obs_bs: &BasisSet) -> Self {
        Self {
            gyration_ratio: gyration_ratio(mol),
            min_exponent: min_exponent(mol, obs_bs),
        }
    }

    pub fn one_d(&self) -> bool {
        self.gyration_ratio <= ONE_D_RATIO_MAX
    }

    pub fn diffuse(&self) -> bool {
        self.min_exponent < DIFFUSE_EXPONENT_MIN
    }
}

fn gyration_ratio(mol: &Molecule) -> f64 {
    use ndarray_linalg::Eigh;
    let n = mol.atoms.len();
    if n < 2 {
        return 0.0;
    }
    let mut x = ndarray::Array2::from_shape_fn((n, 3), |(i, k)| {
        let a = &mol.atoms[i];
        [a.x, a.y, a.zpos][k]
    });
    let mean = x.mean_axis(ndarray::Axis(0)).expect("n >= 2 rows");
    x -= &mean;
    let g = x.t().dot(&x) / n as f64;
    match g.eigh(ndarray_linalg::UPLO::Lower) {
        Ok((w, _)) if w[2] > 0.0 => (w[1].max(0.0) / w[2]).sqrt(),
        _ => 0.0,
    }
}

fn min_exponent(mol: &Molecule, obs_bs: &BasisSet) -> f64 {
    mol.atoms
        .iter()
        .filter_map(|a| obs_bs.for_element(a.z))
        .flat_map(|shells| shells.iter().flat_map(|s| s.exponents.iter().copied()))
        .fold(f64::INFINITY, f64::min)
}

/// One certificate value, typed for both serializers.
#[derive(Debug, Clone, PartialEq)]
pub enum CertValue {
    Null,
    Bool(bool),
    Usize(usize),
    F64(f64),
    Str(String),
    StrList(Vec<String>),
    Map(Vec<(&'static str, CertValue)>),
}

impl From<Option<f64>> for CertValue {
    fn from(v: Option<f64>) -> Self {
        v.map_or(CertValue::Null, CertValue::F64)
    }
}

/// The certificate itself; see the module doc.
#[derive(Debug, Clone, PartialEq)]
pub struct Lmp2Certificate {
    pub scheme: &'static str,
    pub eps: f64,
    pub keep_fraction: f64,
    pub pair_fraction: f64,
    pub n_pairs_gated: usize,
    pub gate_cal: Option<f64>,
    pub integral_direct: bool,
    /// `None` on the in-core (global-B) path.
    pub maps: Option<DirectMaps>,
    /// Stage-5 counters; `None` on the in-core path (set by
    /// [`Lmp2Certificate::with_direct_stats`]).
    pub pair_fit: Option<PairFitCounters>,
    /// Arithmetic precision of every kernel; `"f64"` until a mixed kernel
    /// ships.
    pub precision: &'static str,
    /// The last PCG relative-residual target the solve ran to. The energy is
    /// the stationary Hylleraas form, so its solver error is second order in
    /// the achieved residual (`cg_relres`).
    pub cg_rtol: f64,
    /// Where it came from ([`CgRtolSource::as_str`]).
    pub cg_rtol_source: &'static str,
    /// Whether the error bound forced the PCG past its starting tolerance.
    pub cg_tightened: bool,
    /// The bound target (Ha); `None` unless the bound governed the solve.
    pub cg_energy_error_target: Option<f64>,
    /// Whether the target was met; `None` when no target applied.
    pub cg_energy_target_met: Option<bool>,
    pub cg_relres: f64,
    /// A-priori bound (Ha) on the Hylleraas energy's solver error at
    /// `cg_relres`: `3·‖PJ‖²·relres²/λ_min`, λ_min ≥ 2(ε_LUMO − ε_HOMO).
    pub cg_energy_error_bound: f64,
    pub cg_converged: bool,
    pub frozen_core: usize,
    /// [`operator_label`] of the run's operator; the ε record is Coulomb.
    pub operator: String,
    /// Size of the PCG gather plan (bytes) and the matvec path the solve
    /// ran ("planned" or "indexed", the plan-free fallback when the plan
    /// did not fit the memory budget).
    pub solve_plan_bytes: usize,
    /// Bytes resident beside the plan for the whole solve (ragged system +
    /// PCG vectors); the plan was admitted against the budget minus this.
    pub solve_resident_bytes: usize,
    pub matvec_path: &'static str,
    pub validation_grade: String,
    pub eps_trust: String,
    pub warnings: Vec<String>,
}

impl Lmp2Certificate {
    pub fn new(
        cfg: &AmplitudeLmp2Config,
        direct: Option<&DirectConfig>,
        r: &AmplitudeLmp2Result,
        shape: &SystemShape,
        op: &Operator,
    ) -> Self {
        let coulomb = op.kind == OperatorKind::Coulomb;
        Self {
            scheme: "amplitude-threshold",
            eps: cfg.eps,
            keep_fraction: r.keep_fraction,
            pair_fraction: r.pair_fraction,
            n_pairs_gated: r.n_pairs_gated,
            gate_cal: cfg.pair_gate_cal,
            integral_direct: direct.is_some(),
            maps: direct.map(DirectMaps::from),
            pair_fit: None,
            precision: "f64",
            cg_rtol: r.cg_rtol_outcome.rtol_used,
            cg_rtol_source: r.cg_rtol_outcome.source.as_str(),
            cg_tightened: r.cg_rtol_outcome.tightened,
            cg_energy_error_target: r.cg_rtol_outcome.target_ha,
            cg_energy_target_met: r.cg_rtol_outcome.target_met,
            cg_relres: r.cg_relres,
            cg_energy_error_bound: r.cg_energy_error_bound,
            cg_converged: r.cg_converged,
            frozen_core: cfg.frozen_core,
            operator: operator_label(op),
            solve_plan_bytes: r.solve_plan_bytes,
            solve_resident_bytes: r.solve_resident_bytes,
            matvec_path: r.matvec_path.as_str(),
            validation_grade: validation_grade(cfg.eps),
            eps_trust: eps_trust(cfg.eps, coulomb),
            warnings: {
                let mut w = warnings(cfg.eps, cfg.pair_gate_cal, shape, coulomb);
                w.extend(solver_warnings(
                    &r.cg_rtol_outcome,
                    r.cg_energy_error_bound,
                    r.cg_relres,
                ));
                w
            },
        }
    }

    /// Attach an integral-direct run's stage-5 counters (`None`, the
    /// in-core path, leaves `pair_fit` unset).
    pub fn with_direct_stats(mut self, st: Option<&DirectStats>) -> Self {
        self.pair_fit = st.map(PairFitCounters::from);
        self
    }

    /// Every certificate entry in serialization order. The single source
    /// both the CLI JSON and the Python dict are built from.
    pub fn fields(&self) -> Vec<(&'static str, CertValue)> {
        use CertValue as V;
        let maps = self.maps.as_ref().map_or(V::Null, |m| {
            V::Map(vec![
                ("aux_radius", V::F64(m.aux_radius)),
                ("virt_radius", m.virt_radius.into()),
                ("ao_tail", V::F64(m.ao_tail)),
                ("schwarz_skip", V::F64(m.schwarz_skip)),
                ("batch_merge", V::Usize(m.batch_merge)),
                ("virt_schwarz_kappa", m.virt_schwarz_kappa.into()),
            ])
        });
        let pair_fit = self.pair_fit.map_or(V::Null, |c| {
            V::Map(vec![
                ("n_pair_groups", V::Usize(c.n_pair_groups)),
                (
                    "n_distinct_occ_per_group",
                    V::F64(c.n_distinct_occ_per_group),
                ),
                ("n_trsm", V::Usize(c.n_trsm)),
                ("n_w_reused", V::Usize(c.n_w_reused)),
            ])
        });
        vec![
            ("scheme", V::Str(self.scheme.to_string())),
            ("eps", V::F64(self.eps)),
            ("keep_fraction", V::F64(self.keep_fraction)),
            ("pair_fraction", V::F64(self.pair_fraction)),
            ("n_pairs_gated", V::Usize(self.n_pairs_gated)),
            ("gate_cal", self.gate_cal.into()),
            ("integral_direct", V::Bool(self.integral_direct)),
            ("maps", maps),
            ("pair_fit", pair_fit),
            ("precision", V::Str(self.precision.to_string())),
            ("cg_rtol", V::F64(self.cg_rtol)),
            ("cg_rtol_source", V::Str(self.cg_rtol_source.to_string())),
            ("cg_tightened", V::Bool(self.cg_tightened)),
            ("cg_energy_error_target", self.cg_energy_error_target.into()),
            (
                "cg_energy_target_met",
                self.cg_energy_target_met.map_or(V::Null, V::Bool),
            ),
            ("cg_relres", V::F64(self.cg_relres)),
            ("cg_energy_error_bound", V::F64(self.cg_energy_error_bound)),
            ("cg_converged", V::Bool(self.cg_converged)),
            ("frozen_core", V::Usize(self.frozen_core)),
            ("operator", V::Str(self.operator.clone())),
            ("solve_plan_bytes", V::Usize(self.solve_plan_bytes)),
            ("solve_resident_bytes", V::Usize(self.solve_resident_bytes)),
            ("matvec_path", V::Str(self.matvec_path.to_string())),
            ("validation_grade", V::Str(self.validation_grade.clone())),
            ("eps_trust", V::Str(self.eps_trust.clone())),
            ("warnings", V::StrList(self.warnings.clone())),
        ]
    }
}

fn validation_grade(eps: f64) -> String {
    if eps == 0.0 {
        "Proven (narrow): eps = 0 reproduces canonical RI-MP2 to CG tolerance (anchor <= 1e-9)"
            .to_string()
    } else {
        "finite eps: measured error map, not validated".to_string()
    }
}

fn eps_trust(eps: f64, coulomb: bool) -> String {
    if eps == 0.0 {
        return "eps = 0: nothing truncated (exact limit)".to_string();
    }
    let table = MEASURED_RECOVERY_C4
        .iter()
        .map(|(e, pct)| format!("{pct}% at {e:.0e}"))
        .collect::<Vec<_>>()
        .join(", ");
    let (lo, hi) = (MEASURED_RECOVERY_C4[0].0, MEASURED_RECOVERY_C4[2].0);
    let where_ = if eps < lo * 0.999 || eps > hi * 1.001 {
        "outside the measured range"
    } else {
        "inside the measured range"
    };
    let (e8, pct8) = MEASURED_RECOVERY_C8;
    let op_note = if coulomb {
        ""
    } else {
        "; CAVEAT: the record is the Coulomb operator and this run's operator is not, so \
         it does not cover this run"
    };
    format!(
        "measured Coulomb C4/cc-pVDZ recovery of the canonical DF-MP2 correlation energy: \
         {table}; error superlinear in eps (C4 slope {MEASURED_SLOPE_C4} per decade); \
         larger systems lose more (C8 recovers {pct8}% at {e8:.0e}); this run's eps = \
         {eps:.1e} is {where_}{op_note}"
    )
}

fn warnings(eps: f64, gate_cal: Option<f64>, shape: &SystemShape, coulomb: bool) -> Vec<String> {
    let mut w = Vec::new();
    if eps > VALIDATED_EPS_MAX {
        w.push(format!(
            "eps = {eps:.1e} > {VALIDATED_EPS_MAX:.0e}: looser than the DLPNO-NormalPNO-comparable \
             range (C4/cc-pVDZ loses 0.33% of the correlation energy at 1e-4, 5.0% at 1e-3)"
        ));
    }
    if !coulomb && eps > 0.0 {
        w.push(
            "non-Coulomb operator: the finite-eps error record is Coulomb only; this \
             operator's local error is unmeasured"
                .to_string(),
        );
    }
    if let (Some(cal), true) = (gate_cal, eps > 0.0) {
        w.push(format!(
            "pair gate on (gate_cal = {cal}): its effect on interaction energies is unvalidated"
        ));
    }
    if !shape.one_d() {
        w.push(format!(
            "geometry is not alkane-like 1-D (gyration ratio {:.2} > {ONE_D_RATIO_MAX}): the \
             finite-eps record is n-alkanes only; 3-D systems are untested",
            shape.gyration_ratio
        ));
    }
    if shape.diffuse() {
        w.push(format!(
            "diffuse basis (smallest exponent {:.3} < {DIFFUSE_EXPONENT_MIN}): the measured record \
             is 6-31G / cc-pVDZ; diffuse bases are untested",
            shape.min_exponent
        ));
    }
    w
}

/// Every way the solve's tolerance leaves its energy error unbounded by the
/// default target.
fn solver_warnings(o: &CgRtolOutcome, bound: f64, relres: f64) -> Vec<String> {
    let mut w = Vec::new();
    match o.source {
        CgRtolSource::StrictNoGap => w.push(format!(
            "no positive orbital gap: the PCG energy-error bound is undefined, so the solve ran \
             at the strict cg_rtol = {:.0e}",
            o.rtol_used
        )),
        CgRtolSource::AdaptiveDefault if o.target_met == Some(false) => w.push(format!(
            "PCG energy-error target NOT met: bound {bound:.2e} Ha > {:.0e} Ha at relres \
             {relres:.2e} (cg_rtol {:.0e}; floor or iteration limit reached)",
            o.target_ha.unwrap_or(f64::NAN),
            o.rtol_used
        )),
        CgRtolSource::Explicit if bound.is_nan() => w.push(format!(
            "explicit cg_rtol = {:.0e}: no positive orbital gap, the PCG energy-error bound is \
             undefined",
            o.rtol_used
        )),
        CgRtolSource::Explicit if bound > LMP2_CG_ENERGY_ERROR_TARGET => w.push(format!(
            "explicit cg_rtol = {:.0e}: PCG energy-error bound {bound:.2e} Ha exceeds \
             {LMP2_CG_ENERGY_ERROR_TARGET:.0e} Ha; not tightened, as asked",
            o.rtol_used
        )),
        _ => {}
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(source: CgRtolSource, met: Option<bool>) -> CgRtolOutcome {
        CgRtolOutcome {
            source,
            rtol_used: 1e-6,
            tightened: false,
            target_ha: met.map(|_| 1e-10),
            target_met: met,
        }
    }

    /// Each tolerance outcome that weakens or voids the bound says so; a run
    /// inside it says nothing.
    #[test]
    fn solver_warnings_fire_exactly_on_their_conditions() {
        use CgRtolSource::*;
        assert!(solver_warnings(&outcome(AdaptiveDefault, Some(true)), 1e-13, 5e-7).is_empty());
        assert!(solver_warnings(&outcome(Explicit, None), 1e-13, 5e-7).is_empty());
        let w = solver_warnings(&outcome(AdaptiveDefault, Some(false)), 1e-9, 5e-7);
        assert!(w.len() == 1 && w[0].contains("NOT met"), "{w:?}");
        let w = solver_warnings(&outcome(StrictNoGap, None), f64::NAN, 1e-12);
        assert!(w.len() == 1 && w[0].contains("undefined"), "{w:?}");
        let w = solver_warnings(&outcome(Explicit, None), 1e-9, 5e-7);
        assert!(w.len() == 1 && w[0].contains("not tightened"), "{w:?}");
        let w = solver_warnings(&outcome(Explicit, None), f64::NAN, 5e-7);
        assert!(w.len() == 1 && w[0].contains("undefined"), "{w:?}");
    }

    fn shape(ratio: f64, emin: f64) -> SystemShape {
        SystemShape {
            gyration_ratio: ratio,
            min_exponent: emin,
        }
    }

    /// Each out-of-range condition raises exactly its own warning, and an
    /// in-range run (eps <= 1e-5, no gate, 1-D, compact basis) raises none.
    #[test]
    fn warnings_fire_exactly_on_their_conditions() {
        assert!(warnings(1e-5, None, &shape(0.3, 0.12), true).is_empty());
        assert!(warnings(0.0, Some(0.7), &shape(0.3, 0.12), true).is_empty());
        assert!(warnings(0.0, None, &shape(0.3, 0.12), false).is_empty());
        let w = warnings(1e-4, Some(0.7), &shape(0.9, 0.03), false);
        assert_eq!(w.len(), 5, "{w:?}");
        assert!(w[0].contains("1.0e-4") && w[1].contains("Coulomb") && w[2].contains("gate"));
        assert!(w[3].contains("1-D") && w[4].contains("diffuse"));
        assert_eq!(warnings(1e-6, None, &shape(0.51, 0.12), true).len(), 1);
        assert_eq!(warnings(1e-6, None, &shape(0.3, 0.079), true).len(), 1);
        assert_eq!(warnings(1e-6, None, &shape(0.3, 0.12), false).len(), 1);
    }

    #[test]
    fn grade_and_trust_distinguish_the_exact_limit() {
        assert!(validation_grade(0.0).starts_with("Proven"));
        assert!(validation_grade(1e-6).contains("not validated"));
        assert!(eps_trust(0.0, true).contains("exact"));
        let t = eps_trust(1e-4, true);
        assert!(t.contains("99.671% at 1e-4") && t.contains("inside"), "{t}");
        assert!(
            t.contains("slope ~1.2") && t.contains("93.81% at 1e-3"),
            "{t}"
        );
        assert!(!t.contains("1.8") && !t.contains("CAVEAT"), "{t}");
        assert!(eps_trust(1e-6, true).contains("outside"));
        assert!(eps_trust(1e-4, false).contains("CAVEAT"));
        assert!(!eps_trust(0.0, false).contains("CAVEAT"));
    }

    /// The quoted slope is the one the C4 table implies (lost fraction,
    /// log10 ratio per decade), not a constant typed in separately.
    #[test]
    fn quoted_slope_matches_the_c4_table() {
        let lost: Vec<f64> = MEASURED_RECOVERY_C4
            .iter()
            .map(|(_, p)| 100.0 - p.parse::<f64>().unwrap())
            .collect();
        let s1 = (lost[1] / lost[0]).log10();
        let s2 = (lost[2] / lost[1]).log10();
        assert!(
            (s1 - 1.22).abs() < 0.01 && (s2 - 1.18).abs() < 0.01,
            "{s1} {s2}"
        );
        assert!(MEASURED_SLOPE_C4.contains("1.22") && MEASURED_SLOPE_C4.contains("1.18"));
    }

    #[test]
    fn operator_labels() {
        assert_eq!(operator_label(&Operator::coulomb()), "coulomb");
        assert_eq!(operator_label(&Operator::erfc(0.5)), "erfc(omega=0.5)");
    }
}
