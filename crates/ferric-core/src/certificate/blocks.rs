//! Provenance of building-block results, and the certificate a caller's
//! script composes from it.
//!
//! The Python building blocks (`AoIntegrals` and the objects it builds)
//! return bare arrays. Each `AoIntegrals` object keeps one [`BlockRecord`]:
//! the build, the orbital basis with the source of every element's data, the
//! object-level approximations (ECPs), and every distinct block
//! configuration called on it ([`BlockUse`]: operator, auxiliary basis, XC
//! functional and grid, block-specific approximations). Recording never
//! touches the arrays.
//!
//! [`Certificate::from_blocks`] turns records into a certificate. It is
//! always [`Grade::Ungraded`](super::Grade::Ungraded): a method assembled
//! in the caller's code is not covered by any row of the validation page,
//! and no API raises the grade.
//!
//! ```
//! use ferric_core::basis::bundled;
//! use ferric_core::certificate::{BasisRecord, BlockRecord, BlockUse, BuildStamp, Certificate, Grade};
//!
//! let obs = bundled("cc-pvdz").unwrap();
//! let aux = bundled("cc-pvdz-ri").unwrap();
//! let mut rec = BlockRecord::new(
//!     "AoIntegrals",
//!     BuildStamp::new("0.5.0", "abc", Some(false)),
//!     BasisRecord::of(&obs, &[1, 8]),
//! );
//! rec.record_use(BlockUse::new("df_mo").with_operator("coulomb").with_aux(BasisRecord::of(&aux, &[1, 8])));
//! rec.record_use(BlockUse::new("df_mo").with_operator("coulomb").with_aux(BasisRecord::of(&aux, &[1, 8])));
//! assert_eq!(rec.uses.len(), 1);
//! assert_eq!(rec.uses[0].calls, 2);
//!
//! let cert = Certificate::from_blocks("my-rimp2", vec![rec]).unwrap();
//! assert_eq!(cert.capability, "blocks:my-rimp2");
//! assert_eq!(cert.grade, Grade::Ungraded);
//! assert!(cert.approximations.iter().any(|a| a.name == "ri" && a.detail.contains("cc-pvdz-ri")));
//! ```

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::{Approximation, BuildStamp, Certificate};
use crate::basis::{bundled, bundled_stem, BasisSet};

/// The prefix [`Certificate::from_blocks`] puts on the caller's capability.
pub const BLOCKS_CAPABILITY_PREFIX: &str = "blocks:";

/// Where one element's basis data came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElementSource {
    /// Atomic number.
    pub z: i32,
    /// e.g. `"BSE cc-pVDZ v1, BSE package 0.12"`, `"user-supplied"`.
    pub source: String,
}

/// A basis set by name, with the source of each element's data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BasisRecord {
    /// [`BasisSet::name`].
    pub name: String,
    /// One entry per requested element, ascending `z`.
    pub elements: Vec<ElementSource>,
}

impl BasisRecord {
    /// The record of `bs` for `elements` (duplicates collapse).
    ///
    /// An element is reported as bundled only when its shells and ECP equal
    /// those of the bundled set `bs.name` resolves to; its source is then
    /// read from `basis/provenance.json`. Any other element (a set loaded
    /// from a file, an edited set, a name that is not bundled) is
    /// `"user-supplied"`, and an element `bs` has no shells for is
    /// `"not in this basis"`.
    pub fn of(bs: &BasisSet, elements: &[i32]) -> Self {
        let mut zs = elements.to_vec();
        zs.sort_unstable();
        zs.dedup();
        let shipped = bundled_stem(&bs.name).and_then(|stem| {
            let set = bundled(&bs.name).ok()?;
            Some((set, provenance_entry(stem)?))
        });
        let elements = zs
            .into_iter()
            .map(|z| {
                let source = if bs.for_element(z).is_none() {
                    "not in this basis".to_string()
                } else {
                    match &shipped {
                        Some((set, entry))
                            if set.for_element(z) == bs.for_element(z)
                                && set.ecp_for_element(z) == bs.ecp_for_element(z) =>
                        {
                            bundled_source(entry, z)
                        }
                        _ => "user-supplied".to_string(),
                    }
                };
                ElementSource { z, source }
            })
            .collect();
        BasisRecord {
            name: bs.name.clone(),
            elements,
        }
    }
}

/// `basis/provenance.json`, parsed once.
fn provenance() -> &'static serde_json::Value {
    static P: OnceLock<serde_json::Value> = OnceLock::new();
    P.get_or_init(|| {
        serde_json::from_str(include_str!("../basis/provenance.json"))
            .expect("basis/provenance.json is valid JSON (checked by its own tests)")
    })
}

/// The `sets` entry of the bundled file `<stem>.json`.
fn provenance_entry(stem: &str) -> Option<&'static serde_json::Value> {
    let file = format!("{stem}.json");
    provenance()["sets"]
        .as_array()?
        .iter()
        .find(|s| s["file"].as_str() == Some(file.as_str()))
}

fn lists(v: &serde_json::Value, z: i32) -> bool {
    v.as_array()
        .is_some_and(|a| a.iter().any(|x| x.as_i64() == Some(i64::from(z))))
}

/// The source text of element `z` of the bundled set `entry`.
fn bundled_source(entry: &serde_json::Value, z: i32) -> String {
    let text = |k: &str| entry[k].as_str().unwrap_or("unknown").to_string();
    if entry["source"].as_str() == Some("psi4") {
        let commit = text("psi4_commit");
        let mut s = format!("Psi4 {} @{}", text("name"), &commit[..12.min(commit.len())]);
        if let Some(t) = entry["truncated_from"].as_str() {
            s += &format!(", truncated from {}", t.split(':').next().unwrap_or(t));
        }
        if lists(&entry["extrapolated_elements"], z) {
            s += ", extrapolated";
        }
        return s;
    }
    let pkg = provenance()["bse_package_version"]
        .as_str()
        .unwrap_or("unknown");
    let name = text("bse_name");
    let fill = &entry["filled_from"];
    if lists(&fill["elements"], z) {
        let from = fill["bse_name"].as_str().unwrap_or("unknown");
        return format!("BSE {from} (filled into {name}), BSE package {pkg}");
    }
    if lists(&entry["shipped_only_elements"], z) {
        return format!("local extension of {name} (not in BSE)");
    }
    let mut s = format!(
        "BSE {name} v{}, BSE package {pkg}",
        text("bse_data_version")
    );
    if lists(&entry["function_type_override"]["elements_changed"], z) {
        s += ", l >= 2 shells spherical (Gaussian convention)";
    }
    s
}

/// One distinct block configuration called on a building-block object.
///
/// Every key is required in JSON, the optional ones as `null`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockUse {
    /// The method called, e.g. `"df_mo"`, `"jk"`, `"veff"`.
    pub block: String,
    /// Two-electron operator, e.g. `"coulomb"`, `"erf omega=0.3"`; `None`
    /// for a block with no operator argument.
    #[serde(deserialize_with = "Option::deserialize")]
    pub operator: Option<String>,
    /// The auxiliary basis of a density-fitted block.
    #[serde(deserialize_with = "Option::deserialize")]
    pub aux: Option<BasisRecord>,
    /// `"<functional> on <grid>"` of an exchange-correlation block.
    #[serde(deserialize_with = "Option::deserialize")]
    pub xc: Option<String>,
    /// Block-specific approximations, e.g. `screening`, `x2c`.
    pub approximations: Vec<Approximation>,
    /// How many calls had this configuration.
    pub calls: u64,
}

impl BlockUse {
    /// A use of `block` with no operator, aux, XC or approximation.
    pub fn new(block: impl Into<String>) -> Self {
        BlockUse {
            block: block.into(),
            operator: None,
            aux: None,
            xc: None,
            approximations: Vec::new(),
            calls: 1,
        }
    }

    /// Set the operator text.
    pub fn with_operator(mut self, op: impl Into<String>) -> Self {
        self.operator = Some(op.into());
        self
    }

    /// Set the auxiliary basis.
    pub fn with_aux(mut self, aux: BasisRecord) -> Self {
        self.aux = Some(aux);
        self
    }

    /// Set the XC text.
    pub fn with_xc(mut self, xc: impl Into<String>) -> Self {
        self.xc = Some(xc.into());
        self
    }

    /// Append one approximation.
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

    /// Equal in every field but `calls`.
    fn same_configuration(&self, other: &BlockUse) -> bool {
        self.block == other.block
            && self.operator == other.operator
            && self.aux == other.aux
            && self.xc == other.xc
            && self.approximations == other.approximations
    }
}

/// The provenance record of one building-block object; see the module doc.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockRecord {
    /// The Python class, e.g. `"AoIntegrals"`.
    pub object: String,
    /// The build that computed the blocks.
    pub build: BuildStamp,
    /// The orbital basis, for the molecule's elements.
    pub basis: BasisRecord,
    /// Approximations of the object itself (e.g. `ecp`).
    pub approximations: Vec<Approximation>,
    /// Distinct block configurations, in first-call order.
    pub uses: Vec<BlockUse>,
}

impl BlockRecord {
    /// A record with no approximations and no uses.
    pub fn new(object: impl Into<String>, build: BuildStamp, basis: BasisRecord) -> Self {
        BlockRecord {
            object: object.into(),
            build,
            basis,
            approximations: Vec::new(),
            uses: Vec::new(),
        }
    }

    /// Append one object-level approximation.
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

    /// The record as JSON (the Python `provenance()` dict).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("BlockRecord serializes (no floats)")
    }

    /// Parse a record from JSON, strictly: an unknown or missing key is an
    /// error.
    pub fn from_json_str(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }

    /// Count `u`: a use with the same configuration gains `u.calls`, a new
    /// one is appended.
    pub fn record_use(&mut self, u: BlockUse) {
        match self.uses.iter_mut().find(|e| e.same_configuration(&u)) {
            Some(e) => e.calls += u.calls,
            None => self.uses.push(u),
        }
    }

    /// The approximations this record implies, in order: the object's own,
    /// then per use `ri` (auxiliary basis), `operator` (any non-Coulomb
    /// operator), `xc-grid`, and the use's own.
    pub fn implied_approximations(&self) -> Vec<Approximation> {
        let mut out = self.approximations.clone();
        let a = |name: &str, detail: String| Approximation {
            name: name.to_string(),
            detail,
        };
        for u in &self.uses {
            if let Some(aux) = &u.aux {
                out.push(a("ri", format!("aux {} ({})", aux.name, u.block)));
            }
            if let Some(op) = u.operator.as_deref().filter(|o| *o != "coulomb") {
                out.push(a("operator", format!("{op} ({})", u.block)));
            }
            if let Some(xc) = &u.xc {
                out.push(a("xc-grid", xc.clone()));
            }
            out.extend(u.approximations.iter().cloned());
        }
        out
    }
}

impl Certificate {
    /// Parse a certificate from JSON (e.g. a Python `.certificate` dict),
    /// strictly: an unknown key is an error.
    pub fn from_json_str(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }

    /// The certificate of a result the caller composed from building
    /// blocks.
    ///
    /// * capability `blocks:<capability>`, grade [`Grade::Ungraded`] with no
    ///   source, whatever the records hold;
    /// * `blocks`: the records, unchanged;
    /// * approximations: [`BlockRecord::implied_approximations`] of each
    ///   record, exact duplicates dropped;
    /// * build: the records' stamp; when they differ, the first record's,
    ///   with a warning naming each other stamp.
    ///
    /// Errors when `records` is empty.
    ///
    /// [`Grade::Ungraded`]: super::Grade::Ungraded
    pub fn from_blocks(capability: &str, records: Vec<BlockRecord>) -> Result<Certificate, String> {
        let first = records
            .first()
            .ok_or("a block certificate needs at least one block record")?;
        let mut c = Certificate::ungraded(format!("{BLOCKS_CAPABILITY_PREFIX}{capability}"))
            .with_build(first.build.clone());
        for r in &records {
            if r.build != first.build {
                c.push_warning(format!(
                    "block records come from different builds: {} and {}",
                    first.build, r.build
                ));
            }
            for a in r.implied_approximations() {
                if !c.approximations.contains(&a) {
                    c.approximations.push(a);
                }
            }
        }
        c.blocks = records;
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::Grade;

    fn stamp(commit: &str) -> BuildStamp {
        BuildStamp::new("0.5.0", commit, Some(false))
    }

    fn rimp2_record() -> BlockRecord {
        let obs = bundled("cc-pvdz").unwrap();
        let aux = bundled("cc-pvdz-ri").unwrap();
        let mut r = BlockRecord::new(
            "AoIntegrals",
            stamp("abc"),
            BasisRecord::of(&obs, &[8, 1, 1]),
        );
        r.record_use(BlockUse::new("hcore"));
        r.record_use(
            BlockUse::new("df_mo")
                .with_operator("coulomb")
                .with_aux(BasisRecord::of(&aux, &[1, 8])),
        );
        r.record_use(
            BlockUse::new("jk")
                .with_operator("erf omega=0.3")
                .with_approximation("screening", "Schwarz threshold 1e-12"),
        );
        r.record_use(BlockUse::new("xc_functional").with_xc("pbe on (75, 110), flat"));
        r.record_use(BlockUse::new("hcore"));
        r
    }

    #[test]
    fn record_round_trips_through_json_and_is_strict() {
        let r = rimp2_record();
        let s = serde_json::to_string(&r).unwrap();
        let back: BlockRecord = serde_json::from_str(&s).unwrap();
        assert_eq!(back, r);
        let mut bad: serde_json::Value = serde_json::from_str(&s).unwrap();
        bad["uses"][0]["blok"] = serde_json::json!("x");
        assert!(serde_json::from_value::<BlockRecord>(bad).is_err());
        let mut missing: serde_json::Value = serde_json::from_str(&s).unwrap();
        missing["uses"][1].as_object_mut().unwrap().remove("aux");
        assert!(serde_json::from_value::<BlockRecord>(missing).is_err());
    }

    #[test]
    fn identical_configurations_collapse_and_count() {
        let r = rimp2_record();
        let blocks: Vec<&str> = r.uses.iter().map(|u| u.block.as_str()).collect();
        assert_eq!(blocks, ["hcore", "df_mo", "jk", "xc_functional"]);
        assert_eq!(r.uses[0].calls, 2);
        assert_eq!(r.uses[1].calls, 1);
    }

    #[test]
    fn element_sources_name_the_shipped_data() {
        let src = |name: &str, z: i32| {
            BasisRecord::of(&bundled(name).unwrap(), &[z]).elements[0]
                .source
                .clone()
        };
        assert_eq!(src("cc-pvdz", 8), "BSE cc-pVDZ v1, BSE package 0.12");
        assert_eq!(
            src("cc-pvdz-pp", 8),
            "BSE cc-pVDZ (filled into cc-pVDZ-PP), BSE package 0.12"
        );
        assert!(src("aug-cc-pvdz-rifit", 3).starts_with("local extension of aug-cc-pVDZ-RIFIT"));
        assert!(src("6-311g**", 17).ends_with("spherical (Gaussian convention)"));
        assert!(!src("6-311g**", 8).contains("spherical"));
        let jk = src("aug-cc-pvdz-jkfit", 18);
        assert!(
            jk.starts_with("Psi4 aug-cc-pVDZ-JKFIT @1ea3b9552909"),
            "{jk}"
        );
        assert!(jk.contains("truncated from aug-cc-pVTZ-JKFIT"), "{jk}");
        assert!(jk.ends_with(", extrapolated"), "{jk}");
        assert!(!src("aug-cc-pvdz-jkfit", 8).contains("extrapolated"));
    }

    #[test]
    fn edited_or_unknown_sets_are_user_supplied_per_element() {
        let mut bs = bundled("cc-pvdz").unwrap();
        bs.shells.get_mut(&1).unwrap()[0].exponents[0] *= 1.0001;
        let r = BasisRecord::of(&bs, &[1, 8, 92]);
        let s: Vec<&str> = r.elements.iter().map(|e| e.source.as_str()).collect();
        assert_eq!(s[0], "user-supplied");
        assert!(s[1].starts_with("BSE cc-pVDZ"), "{}", s[1]);
        assert_eq!(s[2], "not in this basis");
        bs.name = "my-set".into();
        assert_eq!(
            BasisRecord::of(&bs, &[8]).elements[0].source,
            "user-supplied"
        );
    }

    #[test]
    fn composed_certificate_is_ungraded_and_names_the_aux_basis() {
        let cert = Certificate::from_blocks("rimp2", vec![rimp2_record()]).unwrap();
        assert_eq!(cert.capability, "blocks:rimp2");
        assert_eq!(cert.grade, Grade::Ungraded);
        assert_eq!(cert.grade_source, None);
        assert_eq!(cert.build, Some(stamp("abc")));
        assert!(cert.warnings.is_empty(), "{:?}", cert.warnings);
        let ap: Vec<(&str, &str)> = cert
            .approximations
            .iter()
            .map(|a| (a.name.as_str(), a.detail.as_str()))
            .collect();
        assert_eq!(
            ap,
            [
                ("ri", "aux cc-pvdz-ri (df_mo)"),
                ("operator", "erf omega=0.3 (jk)"),
                ("screening", "Schwarz threshold 1e-12"),
                ("xc-grid", "pbe on (75, 110), flat"),
            ]
        );
        // A Proven reference cannot raise it.
        let rhf = Certificate::for_method_kind("rhf");
        assert_eq!(cert.clone().with_reference(&rhf).grade, Grade::Ungraded);
        // The certificate round-trips with its records.
        let s = serde_json::to_string(&cert).unwrap();
        assert_eq!(serde_json::from_str::<Certificate>(&s).unwrap(), cert);
        assert!(cert
            .to_string()
            .contains("block       = AoIntegrals cc-pvdz (4 uses)"));
    }

    #[test]
    fn mixed_builds_warn_and_empty_input_errors() {
        let mut other = rimp2_record();
        other.build = stamp("def");
        let cert = Certificate::from_blocks("x", vec![rimp2_record(), other]).unwrap();
        assert_eq!(cert.warnings.len(), 1, "{:?}", cert.warnings);
        assert!(cert.warnings[0].contains("abc") && cert.warnings[0].contains("def"));
        assert_eq!(cert.approximations.len(), 4, "duplicates dropped");
        assert!(Certificate::from_blocks("x", vec![]).is_err());
    }

    #[test]
    fn certificates_without_blocks_serialize_as_before() {
        let j = Certificate::for_method_kind("rhf").to_json();
        assert!(j.get("blocks").is_none(), "{j}");
        let back: Certificate = serde_json::from_value(j).unwrap();
        assert!(back.blocks.is_empty());
    }
}
