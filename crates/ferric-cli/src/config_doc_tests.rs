//! The published docs must agree with the config structs they describe.
//!
//! Every `[section]` key accepted by `Config` is a public promise: the CLI
//! rejects unknown keys, so a key missing from `site/src/reference/input.md`
//! is one a user cannot discover, and a documented key that no struct
//! accepts is one that makes the user's run fail. Neither drift is visible
//! in review, so both are asserted here, in the test binary that already
//! runs on every PR.
//!
//! The field lists come from serde itself, not from a hand-kept copy: the
//! derived `Deserialize` impl hands its `FIELDS` table to
//! `Deserializer::deserialize_struct`, and [`FieldRecorder`] captures that
//! table and stops. Adding a field to a config struct therefore changes what
//! this test demands of the page, with no list to update.

use super::*;
use serde::de::{DeserializeOwned, Visitor};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Error type whose only job is to stop deserialization once the field table
/// has been recorded.
#[derive(Debug)]
struct Stop;

impl std::fmt::Display for Stop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("field recorder stop")
    }
}

impl std::error::Error for Stop {}

impl serde::de::Error for Stop {
    fn custom<T: std::fmt::Display>(_msg: T) -> Self {
        Stop
    }
}

/// A `Deserializer` that records the `FIELDS` table a derived struct impl
/// passes to `deserialize_struct`. `Option<T>` is looked through, so
/// `Option<QmmmCfg>` records `QmmmCfg`'s fields.
struct FieldRecorder(Option<&'static [&'static str]>);

impl<'de> serde::Deserializer<'de> for &mut FieldRecorder {
    type Error = Stop;

    fn deserialize_any<V: Visitor<'de>>(self, _visitor: V) -> Result<V::Value, Stop> {
        Err(Stop)
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, Stop> {
        self.0 = Some(fields);
        Err(Stop)
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Stop> {
        visitor.visit_some(self)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf unit unit_struct newtype_struct seq tuple tuple_struct
        map enum identifier ignored_any
    }
}

/// The key names `T` accepts, as serde's derived impl declares them
/// (aliases included).
fn fields_of<T: DeserializeOwned>() -> BTreeSet<&'static str> {
    let mut rec = FieldRecorder(None);
    let _ = T::deserialize(&mut rec);
    rec.0
        .unwrap_or_else(|| panic!("{} is not a struct", std::any::type_name::<T>()))
        .iter()
        .copied()
        .collect()
}

/// The accepted keys of every TOML table the input reference documents,
/// keyed by the table name as written in a heading (`scf.ladder` for the
/// `[[scf.ladder]]` array of tables).
fn accepted_keys_by_section() -> BTreeMap<&'static str, BTreeSet<&'static str>> {
    BTreeMap::from([
        ("molecule", fields_of::<MoleculeCfg>()),
        ("basis", fields_of::<BasisCfg>()),
        ("method", fields_of::<MethodCfg>()),
        ("scf", fields_of::<ScfCfg>()),
        ("scf.ladder", fields_of::<LadderRungCfg>()),
        ("dft", fields_of::<DftCfg>()),
        ("mp2", fields_of::<Mp2Cfg>()),
        ("rpa", fields_of::<RpaCfg>()),
        ("gw", fields_of::<GwCfg>()),
        ("tddft", fields_of::<TddftCfg>()),
        ("optimize", fields_of::<OptimizeCfg>()),
        ("frequencies", fields_of::<FrequenciesCfg>()),
        ("memory", fields_of::<MemoryCfg>()),
        ("output", fields_of::<OutputCfg>()),
        ("qmmm", fields_of::<QmmmCfg>()),
        ("cosmo", fields_of::<ferric_scf::cosmo::CosmoConfig>()),
        ("pcm", fields_of::<PcmCfg>()),
        ("external_potential", fields_of::<ExternalPotentialCfg>()),
    ])
}

/// `#[serde(alias = ...)]` names, per table. serde lists aliases among a
/// struct's fields with nothing to tell them apart, so they are named here;
/// these alone may be documented in another key's Notes instead of in a row
/// of their own. A missing entry fails loudly (the alias has no row); a stale
/// one is caught by the check at the end of the sync test.
const SERDE_ALIASES: &[(&str, &str)] = &[("rpa", "davidson_conv_thresh")];

/// The repository root, found at RUN time (see `all_shipped_examples_parse`
/// for why `CARGO_MANIFEST_DIR` alone breaks under `cargo nextest archive`).
fn repo_root() -> PathBuf {
    let looks_like_root = |p: &Path| p.join("Cargo.toml").is_file() && p.join("site/src").is_dir();
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// One `##`/`###` section of a Markdown page whose heading names a TOML table.
struct DocSection {
    table: String,
    /// Backticked names in the first column of the section's table rows.
    documented: Vec<String>,
    /// The section's full text, for finding aliases, which are documented in
    /// the Notes column of the key they alias rather than as rows of their own.
    text: String,
}

/// Split a page into the sections whose heading contains a backticked
/// `[table]` or `[[table]]`.
fn table_sections(page: &str) -> Vec<DocSection> {
    let mut out: Vec<DocSection> = Vec::new();
    for line in page.lines() {
        if line.starts_with("## ") || line.starts_with("### ") {
            let table = line
                .split('`')
                .nth(1)
                .filter(|s| s.starts_with('['))
                .map(|s| s.trim_matches(|c| c == '[' || c == ']').to_string());
            match table {
                Some(table) => out.push(DocSection {
                    table,
                    documented: Vec::new(),
                    text: String::new(),
                }),
                // A heading that names no table ends the current section.
                None => out.push(DocSection {
                    table: String::new(),
                    documented: Vec::new(),
                    text: String::new(),
                }),
            }
            continue;
        }
        let Some(sec) = out.last_mut() else { continue };
        sec.text.push_str(line);
        sec.text.push('\n');
        if let Some(first_cell) = line.strip_prefix('|').and_then(|r| r.split('|').next()) {
            sec.documented.extend(
                first_cell
                    .split('`')
                    .skip(1)
                    .step_by(2)
                    .map(|s| s.to_string()),
            );
        }
    }
    out.retain(|s| !s.table.is_empty());
    out
}

#[test]
fn input_reference_documents_exactly_the_accepted_keys() {
    let path = repo_root().join("site/src/reference/input.md");
    let page = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let sections = table_sections(&page);
    let accepted = accepted_keys_by_section();

    let mut problems = Vec::new();

    // Every top-level table the CLI accepts has a section, and vice versa.
    let top_level: BTreeSet<&str> = fields_of::<Config>();
    let documented_top: BTreeSet<&str> = sections
        .iter()
        .map(|s| s.table.as_str())
        .filter(|t| !t.contains('.'))
        .collect();
    for t in top_level.difference(&documented_top) {
        problems.push(format!(
            "[{t}] is accepted by Config but has no section in input.md"
        ));
    }
    for t in documented_top.difference(&top_level) {
        problems.push(format!(
            "input.md documents [{t}], which Config does not accept"
        ));
    }

    for sec in &sections {
        let Some(keys) = accepted.get(sec.table.as_str()) else {
            problems.push(format!(
                "input.md documents [{}], which this test does not map to a config struct; \
                 add it to accepted_keys_by_section()",
                sec.table
            ));
            continue;
        };
        for key in &sec.documented {
            if !keys.contains(key.as_str()) {
                problems.push(format!(
                    "input.md documents `{key}` in [{}], which the config struct does not accept",
                    sec.table
                ));
            }
        }
        for key in keys {
            // Each key needs its own row. The only exception is a serde alias
            // listed in SERDE_ALIASES, which is documented in the Notes of the
            // key it aliases. Merely appearing in another row's text does not
            // count: that is how a deleted row would go unnoticed.
            let has_row = sec.documented.iter().any(|d| d == key);
            let is_documented_alias = SERDE_ALIASES.contains(&(sec.table.as_str(), *key))
                && sec.text.contains(&format!("`{key}`"));
            if !has_row && !is_documented_alias {
                problems.push(format!(
                    "[{}] accepts `{key}`, but input.md has no row for it",
                    sec.table
                ));
            }
        }
    }

    // A stale entry would silently exempt a key forever.
    for (table, alias) in SERDE_ALIASES {
        if !accepted.get(table).is_some_and(|k| k.contains(alias)) {
            problems.push(format!(
                "SERDE_ALIASES lists [{table}] `{alias}`, which the config struct does not accept"
            ));
        }
    }

    assert!(
        problems.is_empty(),
        "site/src/reference/input.md is out of sync with config.rs:\n  {}",
        problems.join("\n  ")
    );
}

/// Every Markdown file the docs build publishes, plus the README (which is
/// also the PyPI description).
fn published_markdown() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let p = entry.unwrap().path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().and_then(|e| e.to_str()) == Some("md") {
                out.push(p);
            }
        }
    }
    let root = repo_root();
    let mut out = vec![root.join("README.md")];
    walk(&root.join("site/src"), &mut out);
    out.sort();
    out
}

/// The fenced ```toml blocks of a page, skipping any block whose preceding
/// line is `<!-- doctest: skip -->` (for deliberately invalid input).
fn toml_blocks(page: &str) -> Vec<(usize, String)> {
    let mut blocks = Vec::new();
    let mut lines = page.lines().enumerate().peekable();
    let mut prev = "";
    while let Some((i, line)) = lines.next() {
        if line.trim_start().starts_with("```toml") {
            let skip = prev.trim() == "<!-- doctest: skip -->";
            let mut body = String::new();
            for (_, l) in lines.by_ref() {
                if l.trim_start().starts_with("```") {
                    break;
                }
                body.push_str(l);
                body.push('\n');
            }
            if !skip {
                blocks.push((i + 1, body));
            }
            prev = "";
            continue;
        }
        prev = line;
    }
    blocks
}

#[test]
fn every_toml_block_in_the_docs_parses() {
    let root = repo_root();
    let mut n = 0;
    let mut failures = Vec::new();
    for path in published_markdown() {
        let page = std::fs::read_to_string(&path).unwrap();
        for (line, body) in toml_blocks(&page) {
            // A snippet may show only the tables it is about. Complete it
            // with the three required tables, appended so that any bare keys
            // at the top of the snippet stay where the author put them.
            let mut doc = body.clone();
            for (table, filler) in [
                ("[molecule]", "[molecule]\nxyz = \"doc-snippet.xyz\"\n"),
                ("[basis]", "[basis]\nname = \"sto-3g\"\n"),
                ("[method]", "[method]\nkind = \"rhf\"\n"),
            ] {
                if !body.lines().any(|l| l.trim() == table) {
                    doc.push('\n');
                    doc.push_str(filler);
                }
            }
            if let Err(e) = toml::from_str::<Config>(&doc) {
                failures.push(format!(
                    "{}:{line}: {e}",
                    path.strip_prefix(&root).unwrap_or(&path).display()
                ));
            }
            n += 1;
        }
    }
    assert!(
        n > 0,
        "found no ```toml blocks at all; is the docs path wrong?"
    );
    assert!(
        failures.is_empty(),
        "TOML blocks in the docs that the CLI would reject (mark a deliberately \
         invalid block with `<!-- doctest: skip -->` on the line before it):\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn examples_index_lists_every_example() {
    let root = repo_root();
    let index = std::fs::read_to_string(root.join("site/src/reference/examples.md")).unwrap();
    let mut missing = Vec::new();
    for entry in std::fs::read_dir(root.join("examples")).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if (name.ends_with(".toml") || name.ends_with(".py")) && !index.contains(&name) {
            missing.push(name);
        }
    }
    missing.sort();
    assert!(
        missing.is_empty(),
        "site/src/reference/examples.md does not mention: {}",
        missing.join(", ")
    );
}

#[test]
fn field_recorder_sees_config_structs() {
    // Guard on the recorder itself: if serde ever stopped routing derived
    // structs through deserialize_struct, every sync test above would compare
    // against an empty set and could pass while checking nothing.
    let method = fields_of::<MethodCfg>();
    assert!(
        method.contains("kind") && method.contains("task"),
        "{method:?}"
    );
    let top = fields_of::<Config>();
    assert!(top.contains("scf") && top.contains("qmmm"), "{top:?}");
    // Option<T> fields must be looked through, not recorded as empty.
    assert!(!fields_of::<ferric_scf::cosmo::CosmoConfig>().is_empty());
}
