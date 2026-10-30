//! The model certificate of an amplitude-threshold local MP2 result: what
//! was computed (scheme, threshold, maps, precision, solver state), how far
//! the result is validated, and every way this run sits outside the
//! measured record. CLI JSON (`"local"`) and Python (`result.local`) both
//! serialize [`fields`](crate::lmp2_certificate::Lmp2Certificate::fields),
//! so the two surfaces cannot drift; each adds the build commit/dirty flag
//! (the library does not link `ferric-build-info`).
//!
//! The ε-trust line quotes the MEASURED local-error record (C4/cc-pVDZ,
//! `wiki/amplitude-threshold-lmp2.md`; memory "local error is superlinear
//! in eps"). It is a record of measurements on n-alkanes, not a bound.

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;

use crate::lmp2_amplitude::{AmplitudeLmp2Config, AmplitudeLmp2Result};
use crate::lmp2_direct::DirectConfig;

/// Largest ε inside the range whose accuracy matches DLPNO NormalPNO on
/// the measured record (C4/cc-pVDZ: 99.980% recovery at 1e-5).
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
    /// Arithmetic precision of every kernel; `"f64"` until a mixed kernel
    /// ships.
    pub precision: &'static str,
    pub cg_relres: f64,
    pub cg_converged: bool,
    pub frozen_core: usize,
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
    ) -> Self {
        Self {
            scheme: "amplitude-threshold",
            eps: cfg.eps,
            keep_fraction: r.keep_fraction,
            pair_fraction: r.pair_fraction,
            n_pairs_gated: r.n_pairs_gated,
            gate_cal: cfg.pair_gate_cal,
            integral_direct: direct.is_some(),
            maps: direct.map(DirectMaps::from),
            precision: "f64",
            cg_relres: r.cg_relres,
            cg_converged: r.cg_converged,
            frozen_core: cfg.frozen_core,
            validation_grade: validation_grade(cfg.eps),
            eps_trust: eps_trust(cfg.eps),
            warnings: warnings(cfg.eps, cfg.pair_gate_cal, shape),
        }
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
        vec![
            ("scheme", V::Str(self.scheme.to_string())),
            ("eps", V::F64(self.eps)),
            ("keep_fraction", V::F64(self.keep_fraction)),
            ("pair_fraction", V::F64(self.pair_fraction)),
            ("n_pairs_gated", V::Usize(self.n_pairs_gated)),
            ("gate_cal", self.gate_cal.into()),
            ("integral_direct", V::Bool(self.integral_direct)),
            ("maps", maps),
            ("precision", V::Str(self.precision.to_string())),
            ("cg_relres", V::F64(self.cg_relres)),
            ("cg_converged", V::Bool(self.cg_converged)),
            ("frozen_core", V::Usize(self.frozen_core)),
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

fn eps_trust(eps: f64) -> String {
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
    format!(
        "measured C4/cc-pVDZ recovery of the canonical DF-MP2 correlation energy: {table}; \
         error superlinear in eps (slope 1.2-1.8 per decade); this run's eps = {eps:.1e} is \
         {where_}"
    )
}

fn warnings(eps: f64, gate_cal: Option<f64>, shape: &SystemShape) -> Vec<String> {
    let mut w = Vec::new();
    if eps > VALIDATED_EPS_MAX {
        w.push(format!(
            "eps = {eps:.1e} > {VALIDATED_EPS_MAX:.0e}: looser than the DLPNO-NormalPNO-comparable \
             range (C4/cc-pVDZ loses 0.33% of the correlation energy at 1e-4, 5.0% at 1e-3)"
        ));
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(warnings(1e-5, None, &shape(0.3, 0.12)).is_empty());
        assert!(warnings(0.0, Some(0.7), &shape(0.3, 0.12)).is_empty());
        let w = warnings(1e-4, Some(0.7), &shape(0.9, 0.03));
        assert_eq!(w.len(), 4, "{w:?}");
        assert!(w[0].contains("1.0e-4") && w[1].contains("gate") && w[2].contains("1-D"));
        assert!(w[3].contains("diffuse"));
        assert_eq!(warnings(1e-6, None, &shape(0.51, 0.12)).len(), 1);
        assert_eq!(warnings(1e-6, None, &shape(0.3, 0.079)).len(), 1);
    }

    #[test]
    fn grade_and_trust_distinguish_the_exact_limit() {
        assert!(validation_grade(0.0).starts_with("Proven"));
        assert!(validation_grade(1e-6).contains("not validated"));
        assert!(eps_trust(0.0).contains("exact"));
        let t = eps_trust(1e-4);
        assert!(t.contains("99.671% at 1e-4") && t.contains("inside"), "{t}");
        assert!(eps_trust(1e-6).contains("outside"));
    }
}
