//! The result certificate: which build produced a number, how strongly the
//! capability behind it is validated, its recorded error against an
//! independent reference, the approximations in effect, the arithmetic
//! precision, and every warning raised along the way.
//!
//! One type for every method. The pieces that already exist are fields of
//! it, not parallel concepts:
//!
//! * precision provenance ([`crate::gpu::provenance::PrecisionRecord`]) is
//!   attached with [`Certificate::with_precision`];
//! * a method-specific certificate (e.g. the local-MP2 model certificate in
//!   `ferric_mp2::lmp2_certificate`) maps onto this type and keeps its own
//!   detail fields for its own surfaces;
//! * the build stamp is a [`BuildStamp`] the top-level artifact fills in
//!   (`ferric-cli` and `ferric-python` link `ferric-build-info`; library
//!   crates do not, by design, so a library-built certificate says
//!   `unstamped` rather than guessing).
//!
//! # Grades are never fabricated
//!
//! The grade of a `method.kind` comes from [`METHOD_KIND_GRADES`], a copy of
//! the Grade column of the `method.kind` matrix on
//! `site/src/reference/validation.md`; a unit test fails if the two differ.
//! Anything not in that table is [`Grade::Ungraded`]. A method can only
//! LOWER a grade ([`Certificate::cap_grade`]), and doing so records why as a
//! warning; there is no API that raises one.
//!
//! # Example
//!
//! ```
//! use ferric_core::certificate::{BuildStamp, Certificate, Grade, MeasuredError};
//!
//! let cert = Certificate::for_method_kind("rimp2")
//!     .with_build(BuildStamp::new("0.5.0.dev1", "testcommit", Some(false)))
//!     .with_approximation("ri", "auxiliary basis cc-pvdz-ri")
//!     .with_measured_error(
//!         MeasuredError::new("energy", 1e-8, "Ha", "PySCF", "testdata/reference/*_rimp2.json")
//!             .unwrap(),
//!     )
//!     .warn("frozen_core = 1 is outside the reference set");
//! assert_eq!(cert.grade, Grade::Proven);
//!
//! // A finite local threshold the page does not grade: cap, never raise.
//! let local = cert.clone().cap_grade(Grade::Ungraded, "local eps = 1e-4 is not graded");
//! assert_eq!(local.grade, Grade::Ungraded);
//! assert_eq!(local.warnings.len(), 2);
//!
//! // Unknown capabilities are Ungraded, not Proven.
//! assert_eq!(Certificate::for_method_kind("no-such-kind").grade, Grade::Ungraded);
//!
//! // JSON round trip.
//! let s = serde_json::to_string(&cert).unwrap();
//! let back: Certificate = serde_json::from_str(&s).unwrap();
//! assert_eq!(back, cert);
//! println!("{cert}");
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::gpu::provenance::PrecisionRecord;

/// Where [`METHOD_KIND_GRADES`] is copied from.
pub const GRADE_SOURCE_PAGE: &str = "site/src/reference/validation.md";
/// The heading of the table on [`GRADE_SOURCE_PAGE`] that holds the grades.
pub const GRADE_SOURCE_HEADING: &str = "## CLI `method.kind` matrix";

/// Validation grade, in the vocabulary of the validation pages.
///
/// Ordered from weakest to strongest, so `min` is the meaningful combinator.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Grade {
    /// No grade is recorded for this capability. The default.
    #[default]
    Ungraded,
    /// No validating test, or a placeholder ("Stub / none").
    Stub,
    /// New code with no comparison to a reference code yet.
    Spike,
    /// A test runs it and checks an invariant; the value is not pinned to a
    /// reference.
    Smoke,
    /// Asserted against an independent reference, for one or two small
    /// systems or one basis. Generalization unverified.
    ProvenNarrow,
    /// Asserted against an independent reference to a tight tolerance in a
    /// committed test.
    Proven,
}

impl Grade {
    /// The page spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Grade::Ungraded => "not graded",
            Grade::Stub => "Stub",
            Grade::Spike => "Spike",
            Grade::Smoke => "Smoke",
            Grade::ProvenNarrow => "Proven (narrow)",
            Grade::Proven => "Proven",
        }
    }

    /// Parse one Grade cell of a validation-page table. Strict: an
    /// unrecognised cell is an error, never a default. Accepts a markdown
    /// link (`[Proven](#anchors)`), bold, and the qualified narrow form
    /// (`Proven (narrow, closed shell)`).
    pub fn parse_page_cell(cell: &str) -> Result<Grade, String> {
        let mut g = cell.trim();
        if let (Some(rest), Some(close)) = (g.strip_prefix('['), g.find("](")) {
            if g.ends_with(')') {
                g = &rest[..close - 1];
            }
        }
        let g = g.trim_matches('*').trim();
        match g {
            "Proven" => Ok(Grade::Proven),
            "Smoke" => Ok(Grade::Smoke),
            "Spike" => Ok(Grade::Spike),
            "Stub" | "Stub / none" => Ok(Grade::Stub),
            "not graded" => Ok(Grade::Ungraded),
            _ if g.starts_with("Proven (narrow") && g.ends_with(')') => Ok(Grade::ProvenNarrow),
            _ => Err(format!("unrecognised grade cell {cell:?}")),
        }
    }
}

impl fmt::Display for Grade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The grade of every CLI `method.kind`, copied from the Grade column of
/// [`GRADE_SOURCE_HEADING`] on [`GRADE_SOURCE_PAGE`]. When a grade changes,
/// change the page and this table together; the
/// `method_kind_grades_match_the_page` test fails if they disagree.
pub const METHOD_KIND_GRADES: &[(&str, Grade)] = &[
    ("rhf", Grade::Proven),
    ("uhf", Grade::Proven),
    ("rohf", Grade::Proven),
    ("ksdft", Grade::Proven),
    ("rimp2", Grade::Proven),
    ("mp3", Grade::Proven),
    ("oo-rimp2", Grade::ProvenNarrow),
    ("att-rimp2", Grade::Proven),
    ("mp2-v", Grade::ProvenNarrow),
    ("scs-mp2", Grade::Proven),
    ("scs-mp2-2terfc", Grade::ProvenNarrow),
    ("laplace-mp2", Grade::Proven),
    ("laplace-sos-mp2", Grade::Ungraded),
    ("pdep-rpa", Grade::Proven),
    ("rs-mp2-rpa", Grade::Smoke),
    ("gw", Grade::Smoke),
    ("bse-tda", Grade::Smoke),
    ("tdhf-static-polarizability", Grade::Smoke),
    ("ccsd", Grade::Proven),
    ("ccd", Grade::ProvenNarrow),
    ("ccsd(t)", Grade::ProvenNarrow),
    ("linlccd", Grade::ProvenNarrow),
    ("drpa", Grade::ProvenNarrow),
    ("wb97x-l-v", Grade::ProvenNarrow),
    ("b2plyp", Grade::Spike),
    ("dsd-pbep86", Grade::Spike),
    ("tda", Grade::ProvenNarrow),
    ("tddft", Grade::ProvenNarrow),
];

/// The recorded grade of a CLI `method.kind`; [`Grade::Ungraded`] when the
/// table has no row for it.
pub fn method_kind_grade(kind: &str) -> Grade {
    METHOD_KIND_GRADES
        .iter()
        .find(|(k, _)| *k == kind)
        .map_or(Grade::Ungraded, |&(_, g)| g)
}

/// The build that produced a result. Filled by the top-level artifact from
/// `ferric-build-info` (`VERSION`, `COMMIT`, `DIRTY`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildStamp {
    /// Package version, or `"unknown"`.
    pub version: String,
    /// Full git commit, or `"unknown"`.
    pub commit: String,
    /// Whether tracked sources differed from `commit`; `None` when unknown.
    pub dirty: Option<bool>,
}

impl BuildStamp {
    pub fn new(version: impl Into<String>, commit: impl Into<String>, dirty: Option<bool>) -> Self {
        BuildStamp {
            version: version.into(),
            commit: commit.into(),
            dirty,
        }
    }
}

impl fmt::Display for BuildStamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let short = &self.commit[..12.min(self.commit.len())];
        let dirty = match self.dirty {
            Some(true) => " (dirty)",
            Some(false) => "",
            None => " (dirty: unknown)",
        };
        write!(f, "{} {short}{dirty}", self.version)
    }
}

/// A capability's error against an independent reference, as recorded by a
/// committed test (not a per-run bound).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredError {
    /// What was compared, e.g. `"energy"`, `"gradient (max |dE/dR|)"`.
    pub quantity: String,
    /// The largest absolute deviation recorded, in `units`. Finite, >= 0.
    pub value: f64,
    pub units: String,
    /// The reference code or data, e.g. `"PySCF 2.6"`.
    pub reference: String,
    /// The committed test or reference file that records it.
    pub evidence: String,
}

impl MeasuredError {
    /// Errors on a non-finite or negative `value` (JSON cannot carry NaN or
    /// infinity, and a negative deviation is meaningless).
    pub fn new(
        quantity: impl Into<String>,
        value: f64,
        units: impl Into<String>,
        reference: impl Into<String>,
        evidence: impl Into<String>,
    ) -> Result<Self, String> {
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "measured error must be finite and >= 0, got {value}"
            ));
        }
        Ok(MeasuredError {
            quantity: quantity.into(),
            value,
            units: units.into(),
            reference: reference.into(),
            evidence: evidence.into(),
        })
    }
}

/// One approximation in effect for a result, e.g. `("ri", "aux cc-pvdz-ri")`
/// or `("local", "amplitude-threshold eps = 1e-5")`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Approximation {
    pub name: String,
    pub detail: String,
}

/// The certificate of one result; see the module doc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Certificate {
    /// What was computed: a CLI `method.kind`, or a library capability name.
    pub capability: String,
    /// `None` until a top-level artifact stamps it.
    pub build: Option<BuildStamp>,
    pub grade: Grade,
    /// Where `grade` came from: [`GRADE_SOURCE_PAGE`] for a table hit,
    /// `None` for [`Grade::Ungraded`] with no recorded source.
    pub grade_source: Option<String>,
    pub measured_error: Option<MeasuredError>,
    pub approximations: Vec<Approximation>,
    /// [`PrecisionRecord::to_json`] of the result, when attached.
    pub precision: Option<serde_json::Value>,
    /// In the order raised; exact duplicates are dropped.
    pub warnings: Vec<String>,
}

impl Certificate {
    /// A certificate with no recorded grade.
    pub fn ungraded(capability: impl Into<String>) -> Self {
        Certificate {
            capability: capability.into(),
            build: None,
            grade: Grade::Ungraded,
            grade_source: None,
            measured_error: None,
            approximations: Vec::new(),
            precision: None,
            warnings: Vec::new(),
        }
    }

    /// A certificate graded from [`METHOD_KIND_GRADES`]. A kind with no row
    /// is [`Grade::Ungraded`] with no source.
    pub fn for_method_kind(kind: &str) -> Self {
        let mut c = Self::ungraded(kind);
        if let Some(&(_, g)) = METHOD_KIND_GRADES.iter().find(|(k, _)| *k == kind) {
            c.grade = g;
            c.grade_source = Some(GRADE_SOURCE_PAGE.to_string());
        }
        c
    }

    /// The certificate of a post-SCF `kind` (MP2, CC, RPA, GW, ...) computed
    /// on the SCF result that `reference` certifies:
    /// [`Self::for_method_kind`]`(kind)`, then [`Self::with_reference`].
    pub fn for_post_scf(kind: &str, reference: &Certificate) -> Self {
        Self::for_method_kind(kind).with_reference(reference)
    }

    /// Fold in the certificate of the SCF reference this result is built on.
    ///
    /// * The grade is capped at the reference's grade (an unconverged
    ///   reference is [`Grade::Ungraded`], so its post-SCF result is too).
    /// * Each reference approximation is kept with its name prefixed
    ///   `scf ` (`scf ri-j`, `scf xc-grid`, ...).
    /// * Each reference warning is kept, prefixed `scf reference: `.
    /// * The reference's build stamp is taken when this one has none.
    ///
    /// The reference's precision record is not copied: it describes the
    /// SCF, not this result.
    pub fn with_reference(mut self, reference: &Certificate) -> Self {
        if self.build.is_none() {
            self.build = reference.build.clone();
        }
        for a in &reference.approximations {
            self.approximations.push(Approximation {
                name: format!("scf {}", a.name),
                detail: a.detail.clone(),
            });
        }
        for w in &reference.warnings {
            self.push_warning(format!("scf reference: {w}"));
        }
        let why = format!(
            "the SCF reference ({}) is {}",
            reference.capability, reference.grade
        );
        self.cap_grade(reference.grade, why)
    }

    /// Record density fitting with auxiliary basis `aux` (approximation `ri`).
    pub fn with_ri(self, aux: &str) -> Self {
        self.with_approximation("ri", format!("aux {aux}"))
    }

    /// Record `n` frozen core orbitals (approximation `frozen-core`); `0`
    /// records nothing.
    pub fn with_frozen_core(self, n: usize) -> Self {
        if n == 0 {
            self
        } else {
            self.with_approximation("frozen-core", format!("{n} core orbital(s) frozen"))
        }
    }

    pub fn with_build(mut self, build: BuildStamp) -> Self {
        self.build = Some(build);
        self
    }

    pub fn with_precision(mut self, record: &PrecisionRecord) -> Self {
        self.precision = Some(record.to_json());
        self
    }

    pub fn with_measured_error(mut self, e: MeasuredError) -> Self {
        self.measured_error = Some(e);
        self
    }

    pub fn with_approximation(
        mut self,
        name: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        self.approximations.push(Approximation {
            name: name.into(),
            detail: detail.into(),
        });
        self
    }

    /// Add one warning (builder form of [`Self::push_warning`]).
    pub fn warn(mut self, w: impl Into<String>) -> Self {
        self.push_warning(w);
        self
    }

    /// Add one warning; an exact duplicate of one already held is dropped.
    pub fn push_warning(&mut self, w: impl Into<String>) {
        let w = w.into();
        if !self.warnings.contains(&w) {
            self.warnings.push(w);
        }
    }

    /// Add every warning of `ws`, in order, through [`Self::push_warning`].
    pub fn extend_warnings<I, S>(&mut self, ws: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for w in ws {
            self.push_warning(w);
        }
    }

    /// Lower the grade to at most `ceiling`. When this lowers it, the
    /// reason is recorded as a warning; a ceiling at or above the current
    /// grade changes nothing. There is no way to raise a grade.
    pub fn cap_grade(mut self, ceiling: Grade, reason: impl Into<String>) -> Self {
        if ceiling < self.grade {
            let from = self.grade;
            self.grade = ceiling;
            self.push_warning(format!(
                "grade capped from {from} to {ceiling}: {}",
                reason.into()
            ));
        }
        self
    }

    /// The certificate as JSON (CLI run log, Python).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("Certificate serializes (no non-finite floats)")
    }
}

impl fmt::Display for Certificate {
    /// A fixed-width block for printouts, one `key = value` per line.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "certificate")?;
        writeln!(f, "  capability  = {}", self.capability)?;
        match &self.build {
            Some(b) => writeln!(f, "  build       = {b}")?,
            None => writeln!(f, "  build       = unstamped")?,
        }
        match &self.grade_source {
            Some(s) => writeln!(f, "  grade       = {} ({s})", self.grade)?,
            None => writeln!(f, "  grade       = {}", self.grade)?,
        }
        match &self.measured_error {
            Some(e) => writeln!(
                f,
                "  error       = {} {:.1e} {} vs {} ({})",
                e.quantity, e.value, e.units, e.reference, e.evidence
            )?,
            None => writeln!(f, "  error       = none recorded")?,
        }
        if self.approximations.is_empty() {
            writeln!(f, "  approx      = none")?;
        }
        for a in &self.approximations {
            writeln!(f, "  approx      = {}: {}", a.name, a.detail)?;
        }
        match self.precision.as_ref().and_then(|p| p.get("precision")) {
            Some(serde_json::Value::String(p)) => writeln!(f, "  precision   = {p}")?,
            _ => writeln!(f, "  precision   = not recorded")?,
        }
        for w in &self.warnings {
            writeln!(f, "  warning     = {w}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    fn full() -> Certificate {
        Certificate::for_method_kind("rimp2")
            .with_build(BuildStamp::new(
                "0.5.0.dev1",
                "testcommit-0123456789",
                Some(true),
            ))
            .with_approximation("ri", "aux cc-pvdz-ri")
            .with_measured_error(
                MeasuredError::new("energy", 1e-8, "Ha", "PySCF", "x_rimp2.json").unwrap(),
            )
            .with_precision(&PrecisionRecord::f64())
            .warn("w1")
    }

    #[test]
    fn serialization_round_trips() {
        for c in [full(), Certificate::ungraded("x")] {
            let s = serde_json::to_string(&c).unwrap();
            let back: Certificate = serde_json::from_str(&s).unwrap();
            assert_eq!(back, c);
            assert_eq!(
                c.to_json(),
                serde_json::from_str::<serde_json::Value>(&s).unwrap()
            );
        }
        let j = full().to_json();
        assert_eq!(j["grade"], "proven");
        assert_eq!(j["build"]["dirty"], true);
        assert_eq!(j["precision"]["precision"], "f64");
        // Strict: an unknown key is an error, not ignored.
        let mut bad = j.clone();
        bad["grde"] = serde_json::json!("proven");
        assert!(serde_json::from_value::<Certificate>(bad).is_err());
    }

    #[test]
    fn display_formats_every_field() {
        let s = full().to_string();
        let want = "certificate\n\
                    \x20 capability  = rimp2\n\
                    \x20 build       = 0.5.0.dev1 testcommit-0 (dirty)\n\
                    \x20 grade       = Proven (site/src/reference/validation.md)\n\
                    \x20 error       = energy 1.0e-8 Ha vs PySCF (x_rimp2.json)\n\
                    \x20 approx      = ri: aux cc-pvdz-ri\n\
                    \x20 precision   = f64\n\
                    \x20 warning     = w1\n";
        assert_eq!(s, want);
        let u = Certificate::ungraded("x").to_string();
        assert!(u.contains("build       = unstamped"), "{u}");
        assert!(u.contains("grade       = not graded\n"), "{u}");
        assert!(u.contains("error       = none recorded"), "{u}");
        assert!(u.contains("approx      = none"), "{u}");
        assert!(u.contains("precision   = not recorded"), "{u}");
        assert!(!u.contains("warning"), "{u}");
    }

    #[test]
    fn warnings_accumulate_in_order_without_duplicates() {
        let mut c = Certificate::ungraded("x").warn("a").warn("b");
        c.push_warning("a");
        c.extend_warnings(["c", "b", "d"]);
        assert_eq!(c.warnings, ["a", "b", "c", "d"]);
    }

    #[test]
    fn ungraded_is_the_default_and_unknown_kinds_get_it() {
        assert_eq!(Grade::default(), Grade::Ungraded);
        let c = Certificate::for_method_kind("not-a-kind");
        assert_eq!(c.grade, Grade::Ungraded);
        assert_eq!(c.grade_source, None);
        assert_eq!(method_kind_grade("not-a-kind"), Grade::Ungraded);
        assert_eq!(Certificate::ungraded("rhf").grade, Grade::Ungraded);
        // A table row of "not graded" is still Ungraded, but sourced.
        let l = Certificate::for_method_kind("laplace-sos-mp2");
        assert_eq!(l.grade, Grade::Ungraded);
        assert_eq!(l.grade_source.as_deref(), Some(GRADE_SOURCE_PAGE));
    }

    #[test]
    fn cap_grade_only_lowers_and_says_why() {
        let c = Certificate::for_method_kind("rhf");
        let same = c.clone().cap_grade(Grade::Proven, "no-op");
        assert_eq!(same, c, "a ceiling at the grade changes nothing");
        let low = c.cap_grade(Grade::Smoke, "because");
        assert_eq!(low.grade, Grade::Smoke);
        assert_eq!(low.warnings, ["grade capped from Proven to Smoke: because"]);
        let still = low.cap_grade(Grade::Proven, "try to raise");
        assert_eq!(still.grade, Grade::Smoke, "cap_grade must never raise");
    }

    #[test]
    fn post_scf_inherits_the_reference() {
        let stamp = BuildStamp::new("0.5.0", "abc", Some(false));
        let scf = Certificate::for_method_kind("rhf")
            .with_build(stamp.clone())
            .with_approximation("ri-j", "aux def2-universal-jkfit")
            .warn("w");
        let c = Certificate::for_post_scf("ccsd(t)", &scf)
            .with_ri("cc-pvdz-ri")
            .with_frozen_core(0)
            .with_frozen_core(1);
        assert_eq!(c.capability, "ccsd(t)");
        assert_eq!(c.grade, Grade::ProvenNarrow, "Proven reference: no cap");
        assert_eq!(c.build, Some(stamp));
        let names: Vec<&str> = c.approximations.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["scf ri-j", "ri", "frozen-core"]);
        assert_eq!(c.approximations[1].detail, "aux cc-pvdz-ri");
        assert_eq!(c.warnings, ["scf reference: w"]);
        assert_eq!(c.precision, None);

        // An unconverged (ungraded) reference caps the post-SCF grade.
        let bad = Certificate::for_method_kind("rhf").cap_grade(Grade::Ungraded, "not converged");
        let c = Certificate::for_post_scf("rimp2", &bad);
        assert_eq!(c.grade, Grade::Ungraded);
        assert_eq!(
            c.warnings,
            [
                "scf reference: grade capped from Proven to not graded: not converged",
                "grade capped from Proven to not graded: the SCF reference (rhf) is not graded",
            ]
        );
        // An own build stamp is kept over the reference's.
        let own = BuildStamp::new("x", "own", None);
        let c = Certificate::ungraded("gw")
            .with_build(own.clone())
            .with_reference(&scf);
        assert_eq!(c.build, Some(own));
    }

    #[test]
    fn measured_error_rejects_non_finite_and_negative() {
        for v in [f64::NAN, f64::INFINITY, -1e-9] {
            assert!(MeasuredError::new("e", v, "Ha", "r", "t").is_err(), "{v}");
        }
        assert!(MeasuredError::new("e", 0.0, "Ha", "r", "t").is_ok());
    }

    #[test]
    fn grade_cells_parse_strictly() {
        let cases = [
            ("[Proven](#anchors)", Grade::Proven),
            ("Proven", Grade::Proven),
            ("**Smoke**", Grade::Smoke),
            (
                "[Proven (narrow, closed shell)](#anchors)",
                Grade::ProvenNarrow,
            ),
            ("Proven (narrow)", Grade::ProvenNarrow),
            ("Spike", Grade::Spike),
            ("Stub / none", Grade::Stub),
            ("not graded", Grade::Ungraded),
        ];
        for (cell, g) in cases {
            assert_eq!(Grade::parse_page_cell(cell), Ok(g), "{cell}");
        }
        for bad in ["", "proven", "Proven (pending PR #1)", "Smoke-ish"] {
            assert!(Grade::parse_page_cell(bad).is_err(), "{bad:?}");
        }
        // Every variant's page spelling parses back to itself.
        for g in [
            Grade::Ungraded,
            Grade::Stub,
            Grade::Spike,
            Grade::Smoke,
            Grade::ProvenNarrow,
            Grade::Proven,
        ] {
            assert_eq!(Grade::parse_page_cell(g.as_str()), Ok(g));
        }
    }

    /// Workspace root at RUN time (see ferric-cli's
    /// `grades_match_cli_warnings.rs` for why not `CARGO_MANIFEST_DIR` alone).
    fn workspace_root() -> PathBuf {
        let looks_like_root = |p: &Path| {
            p.join("Cargo.toml").is_file() && p.join("site").is_dir() && p.join("crates").is_dir()
        };
        if let Ok(cwd) = std::env::current_dir() {
            let mut here: Option<&Path> = Some(cwd.as_path());
            while let Some(p) = here {
                if looks_like_root(p) {
                    return p.to_path_buf();
                }
                here = p.parent();
            }
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("crates/ferric-core sits two levels below the workspace root")
            .to_path_buf()
    }

    /// kind -> grade cell of the table under [`GRADE_SOURCE_HEADING`]; the
    /// Grade column is located by its header.
    fn page_grades(page: &str) -> BTreeMap<String, Grade> {
        let start = page.find(GRADE_SOURCE_HEADING).expect("matrix heading");
        let cells = |row: &str| -> Vec<String> {
            row.trim()
                .trim_matches('|')
                .replace("\\|", "\u{0}")
                .split('|')
                .map(|c| c.replace('\u{0}', "|").trim().to_string())
                .collect()
        };
        let mut rows = page[start..]
            .lines()
            .skip_while(|l| !l.trim_start().starts_with('|'))
            .take_while(|l| l.trim_start().starts_with('|'));
        let header = cells(rows.next().expect("header row"));
        let col = header
            .iter()
            .position(|h| h == "Grade")
            .expect("Grade column");
        let mut out = BTreeMap::new();
        for row in rows.skip(1) {
            let c = cells(row);
            let kind = c[0].trim_matches('`').to_string();
            let g = Grade::parse_page_cell(&c[col]).unwrap_or_else(|e| panic!("`{kind}`: {e}"));
            assert!(out.insert(kind.clone(), g).is_none(), "`{kind}` twice");
        }
        out
    }

    #[test]
    fn method_kind_grades_match_the_page() {
        let path = workspace_root().join(GRADE_SOURCE_PAGE);
        let page =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let on_page = page_grades(&page);
        assert!(on_page.len() >= 20, "parsed only {} rows", on_page.len());
        let ours: BTreeMap<String, Grade> = METHOD_KIND_GRADES
            .iter()
            .map(|&(k, g)| (k.to_string(), g))
            .collect();
        assert_eq!(
            ours.len(),
            METHOD_KIND_GRADES.len(),
            "duplicate kind in METHOD_KIND_GRADES"
        );
        assert_eq!(
            ours, on_page,
            "METHOD_KIND_GRADES disagrees with {GRADE_SOURCE_PAGE}"
        );
    }
}
