//! Guard: shim.cc may only reach a libint2 `::instance(` singleton through
//! `core_eval_instance` (which holds `libint_ctor_mutex`) or through the
//! known-safe `SolidHarmonicsCoefficients::instance` (a function-local static
//! that is built once and never replaced, so it is thread-safe per C++11).
//!
//! libint2's core-evaluator `instance(m_max)` replaces its static
//! `shared_ptr` when a larger `m_max` is requested, while other callers copy
//! it unlocked. An unlocked call in the shim reopens the use-after-free that
//! `core_eval_singleton_race.rs` reproduces. Engine constructors reach
//! `instance()` inside libint2 and are covered by holding `libint_ctor_mutex`
//! at every construction site, which this scan does not see.
//!
//! The scan strips comments, then requires every `::instance(` to match an
//! allowed form. Mutation-checked 2026-11-11: adding
//! `auto x = libint2::FmEval_Chebyshev7<double>::instance(3);` to a shim
//! function makes this test fail and name that line.

const SHIM: &str = include_str!("../shim/shim.cc");

/// Replace `//` and `/* */` comments with spaces, keeping newlines, so line
/// numbers survive. String literals in shim.cc never contain `::instance(`,
/// so the scan does not track them.
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

/// The `::instance(` call sites (1-based line, trimmed text) in comment-free code.
fn instance_calls(src: &str) -> Vec<(usize, String)> {
    strip_comments(src)
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains("::instance("))
        .map(|(i, l)| (i + 1, l.trim().to_string()))
        .collect()
}

fn allowed(line: &str) -> bool {
    line == "return CoreEval::instance(mmax);"
        || line.contains("SolidHarmonicsCoefficients<double>::instance(")
}

#[test]
fn every_libint2_singleton_call_in_the_shim_is_locked_or_known_safe() {
    let calls = instance_calls(SHIM);
    let bad: Vec<_> = calls.iter().filter(|(_, l)| !allowed(l)).collect();
    assert!(
        bad.is_empty(),
        "unlocked libint2 ::instance( call(s) in shim.cc; route them through \
         core_eval_instance<...>(mmax) (holds libint_ctor_mutex):\n{bad:#?}"
    );
    let helper = calls
        .iter()
        .filter(|(_, l)| l == "return CoreEval::instance(mmax);")
        .count();
    assert_eq!(
        helper, 1,
        "core_eval_instance's own call must exist exactly once"
    );
}

/// The scanner itself: comments are ignored, and an unlocked call is caught.
#[test]
fn scanner_ignores_comments_and_flags_an_unlocked_call() {
    let src = "// FmEval_Chebyshev7<double>::instance(3) in a comment\n\
               /* TennoGmEval<double>::instance(2)\n   still a comment */\n\
               int f() { auto x = libint2::FmEval_Chebyshev7<double>::instance(3); }\n";
    let calls = instance_calls(src);
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert_eq!(calls[0].0, 4);
    assert!(!allowed(&calls[0].1));
}
