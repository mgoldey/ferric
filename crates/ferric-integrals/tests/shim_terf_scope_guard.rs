//! Guard: every `scf_compute_*` entry point in shim.cc that drives a libint2
//! engine through `eng->engine` must open `FERRIC_TERF_HOOK_SCOPE(eng)` before
//! its first use of that engine.
//!
//! libint2's terf hook carries no engine context; it reads the thread-local
//! table set that `FERRIC_TERF_HOOK_SCOPE` publishes. An entry point that
//! drives `eng->engine` without the scope computes with whatever the thread
//! last published (nullptr or another engine's tables) when the engine was
//! built for `FERRIC_LIBINT2_TERF`. The macro expands to a no-op in builds
//! without that define, so no ordinary test notices a missing scope.
//!
//! Entry points that never touch `eng->engine` (the emultipole dipole and
//! second-moment functions build a local engine of their own, and the
//! standalone `scf_compute_terf*` family evaluates from its own tables) are
//! exempt by that criterion, and the set of exempt names is pinned so a new
//! entry point cannot become exempt by accident.

const SHIM: &str = include_str!("../shim/shim.cc");

const SCOPE: &str = "FERRIC_TERF_HOOK_SCOPE(eng)";

/// Entry points that do not drive `eng->engine` and so need no scope.
const EXEMPT: &[&str] = &[
    "scf_compute_dipole",
    "scf_compute_second_moment",
    "scf_compute_terfc_eri3",
    "scf_compute_terfc_eri2",
    "scf_compute_terf_eri3",
    "scf_compute_terfc_eri4",
    "scf_compute_terf_eri2",
    "scf_compute_terf_eri4",
];

/// Replace `//` and `/* */` comments with spaces, keeping newlines.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        if b[i..].starts_with(b"//") {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i..].starts_with(b"/*") {
            i += 2;
            while i < b.len() && !b[i..].starts_with(b"*/") {
                if b[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i = (i + 2).min(b.len());
        } else {
            let ch = src[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// (name, body) of every top-level `scf_compute_*` definition in comment-free
/// source. A definition starts at a line beginning with `int scf_compute_` or
/// `extern "C" int scf_compute_`; its body is the brace-balanced block after
/// the parameter list. Braces inside string literals do not occur in these
/// bodies.
fn entry_points(code: &str) -> Vec<(String, String)> {
    let mut v = Vec::new();
    let mut pos = 0;
    while pos < code.len() {
        let line_end = code[pos..].find('\n').map_or(code.len(), |n| pos + n);
        let line = &code[pos..line_end];
        let rest = line
            .strip_prefix("extern \"C\" ")
            .unwrap_or(line)
            .strip_prefix("int scf_compute_");
        if let Some(rest) = rest {
            let name_len = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            let name = format!("scf_compute_{}", &rest[..name_len]);
            let open = pos + code[pos..].find('{').expect("definition has a body");
            let mut depth = 0usize;
            let mut end = open;
            for (k, ch) in code[open..].char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = open + k;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            v.push((name, code[open..=end].to_string()));
            pos = end;
            continue;
        }
        pos = line_end + 1;
    }
    v
}

/// Violations: entry points that use `eng->engine` without opening the scope
/// first, and exempt-list drift.
fn violations(src: &str) -> Vec<String> {
    let eps = entry_points(&strip_comments(src));
    let mut bad = Vec::new();
    for (name, body) in &eps {
        let exempt = EXEMPT.contains(&name.as_str());
        match (body.find("eng->engine"), body.find(SCOPE)) {
            (Some(_), None) => bad.push(format!("{name}: uses eng->engine, no {SCOPE}")),
            (Some(u), Some(s)) if s > u => {
                bad.push(format!("{name}: {SCOPE} comes after the first eng->engine"))
            }
            (Some(_), Some(_)) if exempt => bad.push(format!("{name}: listed exempt but scoped")),
            (None, _) if !exempt => bad.push(format!(
                "{name}: never uses eng->engine but is not in EXEMPT; add it or scope it"
            )),
            _ => {}
        }
    }
    for e in EXEMPT {
        if !eps.iter().any(|(n, _)| n == e) {
            bad.push(format!("EXEMPT names {e}, which shim.cc does not define"));
        }
    }
    bad
}

#[test]
fn every_engine_driving_entry_point_opens_the_terf_hook_scope() {
    let eps = entry_points(&strip_comments(SHIM));
    // Anchor: the scan finds the entry points (a broken parser that finds
    // none would pass vacuously). 25 definitions at the time of writing.
    assert!(
        eps.len() >= 20,
        "scanner found only {} scf_compute_* definitions",
        eps.len()
    );
    let scoped = eps.iter().filter(|(_, b)| b.contains(SCOPE)).count();
    assert!(scoped >= 15, "only {scoped} entry points open the scope");
    let bad = violations(SHIM);
    assert!(bad.is_empty(), "terf hook scope violations:\n{bad:#?}");
}

#[test]
fn the_guard_fails_on_a_shim_missing_one_scope() {
    // Mutated copy: drop the scope from scf_compute_eri2 only.
    let start = SHIM
        .find("int scf_compute_eri2(")
        .expect("scf_compute_eri2 exists");
    let rel = SHIM[start..].find(SCOPE).expect("it opens the scope");
    let at = start + rel;
    let mutated = format!("{}{}", &SHIM[..at], &SHIM[at + SCOPE.len() + 1..]);
    let bad = violations(&mutated);
    assert_eq!(bad.len(), 1, "{bad:#?}");
    assert!(bad[0].starts_with("scf_compute_eri2:"), "{bad:#?}");
}
