//! `libint_max_am` / `libint_g12_max_am` report exactly what the
//! `libint2_params.h` the shim was compiled against says, for whichever
//! libint2 this build uses (no hard-coded version numbers).

use ferric_integrals::{libint_g12_max_am, libint_linkage, libint_max_am, libint_max_deriv_order};

/// `# define NAME <int>` from the compiled-against params header, if present.
fn header_macro(header: &str, name: &str) -> Option<i32> {
    header.lines().find_map(|line| {
        let mut words = line.trim_start_matches('#').split_whitespace();
        match (words.next(), words.next(), words.next()) {
            (Some("define"), Some(n), Some(v)) if n == name => v.parse().ok(),
            _ => None,
        }
    })
}

fn params_header() -> String {
    let path = concat!(
        env!("FERRIC_LIBINT_ROOT"),
        "/include/libint2/libint2_params.h"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

#[test]
fn max_am_matches_the_compiled_against_header() {
    let header = params_header();
    let report = libint_max_am();
    let classes = [
        ("eri", "eri"),
        ("eri3", "3eri"),
        ("eri2", "2eri"),
        ("onebody", "overlap"),
    ];
    assert_eq!(report.len(), classes.len());
    for (key, stem) in classes {
        let expected: Vec<i32> = ["", "1", "2"]
            .iter()
            .map_while(|d| header_macro(&header, &format!("LIBINT2_MAX_AM_{stem}{d}")))
            .collect();
        assert!(
            !expected.is_empty(),
            "header defines no LIBINT2_MAX_AM_{stem}"
        );
        assert_eq!(report[key], expected, "{key}");
    }
}

#[test]
fn deriv_entries_never_exceed_max_deriv_order() {
    let order = libint_max_deriv_order();
    for (key, ams) in libint_max_am() {
        assert!(!ams.is_empty(), "{key}");
        assert!(
            ams.len() as i32 <= order + 1,
            "{key}: {ams:?} vs deriv order {order}"
        );
        assert!(ams.iter().all(|&l| l >= 0), "{key}: {ams:?}");
    }
}

#[test]
fn g12_and_linkage_match_the_build() {
    let header = params_header();
    let expected = header_macro(&header, "LIBINT2_SUPPORT_G12")
        .and_then(|_| header_macro(&header, "LIBINT2_MAX_AM_R12kG12"));
    assert_eq!(libint_g12_max_am(), expected);
    let root = env!("FERRIC_LIBINT_ROOT");
    let has_static = std::path::Path::new(&format!("{root}/lib/libint2.a")).exists();
    assert_eq!(
        libint_linkage(),
        if has_static { "static" } else { "shared" }
    );
}
