//! Build script for `ferric-core`. Two jobs:
//!
//! 1. **Link `libopenblas` for `ferric-core`'s own test/doc binaries.**
//!    `ferric-core::blas_threads` (moved here from `ferric-integrals` so any
//!    crate can reach it without a new dependency edge — see that module's
//!    doc) calls the OpenBLAS runtime API (`openblas_get_num_threads` /
//!    `openblas_set_num_threads`) directly via `extern "C"`. In every
//!    downstream binary (ferric-cli, the Python extension, every
//!    integration-test binary) this symbol is already resolved because
//!    `ferric-integrals`'s build script emits
//!    `cargo:rustc-link-lib=dylib=openblas` and every such binary depends on
//!    `ferric-integrals` too. But `ferric-core` itself has NO dependency on
//!    `ferric-integrals` (that would be backwards), so `ferric-core`'s OWN
//!    `cargo test -p ferric-core` binary has nothing upstream emitting that
//!    link directive. Mirror it here so the unit tests link standalone.
//!
//! 2. **Pack the bundled basis sets.** Every `src/basis/bundled/*.json`
//!    (except `def2-ecp.json`, which `ecp.rs` embeds on its own) is reduced
//!    to its physics content — per element, the shells' function type,
//!    angular momenta, exponents and coefficients (the exact BSE digit
//!    strings, untouched) and any ECP — with every distinct shell and ECP
//!    stored ONCE in a shared pool. Most sets are supersets or near-copies of
//!    another (cc-pVTZ inside aug-cc-pVTZ, def2-TZVP inside def2-TZVPP and
//!    def2-TZVPD, the def2 RIFIT family, the light elements of the PP sets),
//!    and a deflate stream's 32 KiB window cannot see repeats that far
//!    apart, so the pool is what keeps the wheel small. The pooled JSON is
//!    then deflated (level 9) and embedded with `include_bytes!`; see
//!    `basis::packed` for the reader.
//!
//!    MEASURED 2026-10-29 on the 64 packed files (26.4 MB of pretty JSON):
//!    embedded blob 0.61 MB (pooled JSON ~3.1 MB before deflate). The release
//!    `libferric.so` went from 53.72 MB (origin/main, 27 sets embedded with
//!    `include_str!`) to 42.39 MB, and its deflate-6 size, which is what the
//!    wheel pays, from 19.35 MB to 19.44 MB (+0.09 MB). Embedding all 64
//!    pretty files with `include_str!` would add ~1.0 MB to the wheel.
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use serde_json::{json, Value};

const BUNDLED_DIR: &str = "src/basis/bundled";
/// Embedded separately by `ecp::bundled`; not a basis set.
const NOT_PACKED: &[&str] = &["def2-ecp.json"];

fn main() {
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={BUNDLED_DIR}");
    pack_bundled_bases();
}

/// Index of `v` in `pool`, appending it if new.
fn intern(v: Value, pool: &mut Vec<Value>, index: &mut HashMap<String, usize>) -> usize {
    let key = v.to_string();
    *index.entry(key).or_insert_with(|| {
        pool.push(v);
        pool.len() - 1
    })
}

fn pick(obj: &Value, keys: &[&str]) -> Value {
    let mut out = serde_json::Map::new();
    for k in keys {
        if let Some(v) = obj.get(*k) {
            out.insert((*k).to_string(), v.clone());
        }
    }
    Value::Object(out)
}

fn pack_bundled_bases() {
    let mut files: Vec<_> = fs::read_dir(BUNDLED_DIR)
        .expect("read src/basis/bundled")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();

    let mut shells: Vec<Value> = Vec::new();
    let mut shell_index = HashMap::new();
    let mut ecps: Vec<Value> = Vec::new();
    let mut ecp_index = HashMap::new();
    let mut sets = BTreeMap::new();

    for path in &files {
        let fname = path.file_name().unwrap().to_string_lossy().into_owned();
        if NOT_PACKED.contains(&fname.as_str()) {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let text = fs::read_to_string(path).expect("read bundled basis");
        let doc: Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{}: invalid JSON: {e}", path.display()));
        let elements = doc["elements"]
            .as_object()
            .unwrap_or_else(|| panic!("{}: no `elements` object", path.display()));
        let mut els = BTreeMap::new();
        for (z, el) in elements {
            let ids: Vec<usize> = el
                .get("electron_shells")
                .and_then(Value::as_array)
                .map(|a| a.as_slice())
                .unwrap_or_default()
                .iter()
                .map(|sh| {
                    let v = pick(
                        sh,
                        &[
                            "function_type",
                            "angular_momentum",
                            "exponents",
                            "coefficients",
                        ],
                    );
                    intern(v, &mut shells, &mut shell_index)
                })
                .collect();
            let ecp = match (el.get("ecp_electrons"), el.get("ecp_potentials")) {
                (Some(n), Some(pots)) => {
                    let pots: Vec<Value> = pots
                        .as_array()
                        .expect("ecp_potentials array")
                        .iter()
                        .map(|p| {
                            pick(
                                p,
                                &[
                                    "angular_momentum",
                                    "r_exponents",
                                    "gaussian_exponents",
                                    "coefficients",
                                ],
                            )
                        })
                        .collect();
                    let v = json!({"ecp_electrons": n, "ecp_potentials": pots});
                    Some(intern(v, &mut ecps, &mut ecp_index))
                }
                _ => None,
            };
            els.insert(z.clone(), json!({"s": ids, "e": ecp}));
        }
        let stem = fname.trim_end_matches(".json").to_string();
        sets.insert(stem, json!({"name": doc.get("name"), "elements": els}));
    }

    let packed = json!({"shells": shells, "ecps": ecps, "sets": sets});
    let raw = serde_json::to_vec(&packed).expect("serialize packed bases");
    let deflated = miniz_oxide::deflate::compress_to_vec(&raw, 9);
    let out = Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR")).join("bundled_bases.bin");
    fs::write(&out, deflated).expect("write packed bases");
}
