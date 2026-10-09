//! Guard: no Å ↔ Bohr conversion factor may be written as a float literal in
//! any crate's `src/` or `examples/` outside `ferric-core/src/units.rs`.
//!
//! ferric once carried four different factors (1/0.52917721092 for geometry,
//! 1.8897259886 for r₀/ω/radii, 1.8897259885799238 for internal coordinates,
//! 1/0.529177210903 in an example), so a length given in Å landed in a frame
//! ~7e-8 relative away from the geometry's (#329). Every conversion now goes
//! through `ferric_core::units`; this test keeps it that way.
//!
//! Detection is by VALUE, not spelling: every numeric literal with a decimal
//! point is parsed (underscores stripped) and flagged if it lies within 1e-4
//! relative of 1/0.52917721092 or 0.52917721092. That catches `1.8897`,
//! `1.889_725_988_6`, `0.529177`, `0.529_177_210_903`, ... alike. Line
//! comments are not scanned (prose may quote a value). The file set is DERIVED
//! by walking `crates/*/{src,examples}`, never hand-listed, so a new crate or
//! file is covered the moment it exists.

use std::path::{Path, PathBuf};

const TARGETS: [f64; 2] = [1.0 / 0.529_177_210_92, 0.529_177_210_92];
const REL_TOL: f64 = 1e-4;

/// Every float literal on `line` (before any `//`) whose value is an Å ↔ Bohr
/// factor.
fn offending_literals(line: &str) -> Vec<String> {
    let code = line.split("//").next().unwrap_or("");
    let b = code.as_bytes();
    let mut hits = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let starts = b[i].is_ascii_digit()
            && (i == 0
                || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'.'));
        if !starts {
            i += 1;
            continue;
        }
        let mut j = i;
        let mut seen_dot = false;
        while j < b.len() {
            let c = b[j];
            if c.is_ascii_digit() || c == b'_' {
                j += 1;
            } else if c == b'.' && !seen_dot && j + 1 < b.len() && b[j + 1].is_ascii_digit() {
                seen_dot = true;
                j += 1;
            } else if (c == b'e' || c == b'E') && seen_dot {
                let mut k = j + 1;
                if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
                    k += 1;
                }
                if k < b.len() && b[k].is_ascii_digit() {
                    j = k;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        if seen_dot {
            let text: String = code[i..j].chars().filter(|&c| c != '_').collect();
            if let Ok(v) = text.parse::<f64>() {
                if TARGETS.iter().any(|t| ((v - t) / t).abs() < REL_TOL) {
                    hits.push(code[i..j].to_string());
                }
            }
        }
        i = j.max(i + 1);
    }
    hits
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn detector_flags_every_spelling_and_ignores_neighbours() {
    for s in [
        "let r = x * 1.8897259886;",
        "const A: f64 = 1.889_725_988_579_923_8;",
        "1.0 / 0.529_177_210_92",
        "a.x * 0.529177210903",
        "BOND * 0.529177",
        "y*1.8897e0",
    ] {
        assert_eq!(offending_literals(s).len(), 1, "missed: {s}");
    }
    for s in [
        "let x = 0.529;",             // 3.3e-4 relative away: not the factor
        "// 1 A = 1.8897259886 Bohr", // a comment
        "v1.8897",                    // part of an identifier
        "let n = 18897259886;",       // no decimal point
        "let w = 0.75 * ANGSTROM_TO_BOHR;",
    ] {
        assert!(offending_literals(s).is_empty(), "false hit: {s}");
    }
}

#[test]
fn no_angstrom_bohr_literal_outside_the_units_module() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let units = crates
        .join("ferric-core/src/units.rs")
        .canonicalize()
        .unwrap();
    let mut files = Vec::new();
    let mut n_crates = 0;
    for e in std::fs::read_dir(&crates).unwrap().flatten() {
        if e.path().join("Cargo.toml").is_file() {
            n_crates += 1;
            rust_files(&e.path().join("src"), &mut files);
            rust_files(&e.path().join("examples"), &mut files);
        }
    }
    // Reachability: the walk must actually have found the workspace.
    assert!(
        n_crates >= 10,
        "walked only {n_crates} crates under {crates:?}"
    );
    assert!(files.len() >= 100, "walked only {} .rs files", files.len());
    assert!(files.iter().any(|f| f.canonicalize().unwrap() == units));

    let mut bad = Vec::new();
    for f in &files {
        if f.canonicalize().unwrap() == units {
            continue;
        }
        let text = std::fs::read_to_string(f).unwrap();
        for (n, line) in text.lines().enumerate() {
            for lit in offending_literals(line) {
                bad.push(format!("{}:{}: {lit}", f.display(), n + 1));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "Å↔Bohr factor written as a literal; use ferric_core::units::{{ANGSTROM_TO_BOHR, \
         BOHR_TO_ANGSTROM}} instead:\n{}",
        bad.join("\n")
    );
}
