//! ferric's whole D3(BJ) parameter table against a snapshot of simple-dftd3's.
//!
//! `testdata/d3bj_params_snapshot.json` is written by
//! `scripts/gen_d3bj_param_snapshot.py` from the installed `dftd3` package
//! (its `parameters.toml`, cross-checked there against the compiled library).
//! Every name `d3bj_tabulated_functionals()` lists must be in the snapshot with
//! bit-identical `(s6, s8, a1, a2)`, every snapshot name must be tabulated, and
//! every `no_upstream_bj` name must be refused. Any drift fails here.

use ferric_d3::{d3bj_params_for_functional, d3bj_tabulated_functionals};

/// `(name, [s6, s8, a1, a2])` rows and the refused names, from the snapshot.
///
/// Strict hand reader for the generator's fixed `json.dumps(indent=1,
/// sort_keys=True)` shape: a missing key panics.
fn snapshot() -> (Vec<(String, [f64; 4])>, Vec<String>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/d3bj_params_snapshot.json"
    );
    let text = std::fs::read_to_string(path).expect("read snapshot");
    let num = |obj: &str, key: &str| -> f64 {
        let pat = format!("\"{key}\":");
        let at = obj
            .find(&pat)
            .unwrap_or_else(|| panic!("no {key} in {obj}"))
            + pat.len();
        let rest = obj[at..].trim_start();
        let end = rest.find([',', '\n', '}']).expect("number end");
        rest[..end].trim().parse().expect("number")
    };

    let p0 = text.find("\"params\": {").expect("params") + "\"params\": {".len();
    let mut rows = Vec::new();
    let mut cur = p0;
    loop {
        let rest = &text[cur..];
        let open = rest.find('{');
        let close = rest.find("\n }");
        match (open, close) {
            (Some(o), Some(c)) if o < c => {
                let head = &rest[..o];
                let q1 = head.rfind("\":").expect("key end");
                let q0 = head[..q1].rfind('"').expect("key start");
                let name = head[q0 + 1..q1].to_string();
                let body_end = rest[o..].find('}').expect("entry end") + o;
                let body = &rest[o..body_end];
                rows.push((
                    name,
                    [
                        num(body, "s6"),
                        num(body, "s8"),
                        num(body, "a1"),
                        num(body, "a2"),
                    ],
                ));
                cur += body_end + 1;
            }
            _ => break,
        }
    }

    let r0 =
        text.find("\"no_upstream_bj\": [").expect("no_upstream_bj") + "\"no_upstream_bj\": [".len();
    let r1 = text[r0..].find(']').expect("list end") + r0;
    let refused = text[r0..r1]
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty())
        .collect();
    (rows, refused)
}

#[test]
fn table_matches_simple_dftd3_snapshot() {
    let (rows, _) = snapshot();
    assert!(
        rows.len() >= 16,
        "snapshot parse found only {} rows",
        rows.len()
    );
    for (name, [s6, s8, a1, a2]) in &rows {
        let p = d3bj_params_for_functional(name)
            .unwrap_or_else(|e| panic!("{name}: upstream has a D3(BJ) fit, ferric errors: {e}"));
        assert_eq!(
            [p.s6, p.s8, p.a1, p.a2],
            [*s6, *s8, *a1, *a2],
            "{name}: ferric (s6, s8, a1, a2) differs from simple-dftd3"
        );
        assert_eq!(p.s9, 0.0, "{name}: named lookup must load with ATM off");
    }
}

#[test]
fn every_tabulated_name_is_in_the_snapshot() {
    let (rows, _) = snapshot();
    let mut snap: Vec<&str> = rows.iter().map(|(n, _)| n.as_str()).collect();
    let mut ours = d3bj_tabulated_functionals();
    snap.sort_unstable();
    ours.sort_unstable();
    assert_eq!(
        ours, snap,
        "ferric's tabulated names and the simple-dftd3 snapshot differ; \
         regenerate with scripts/gen_d3bj_param_snapshot.py after updating its map"
    );
}

#[test]
fn functionals_without_an_upstream_fit_are_refused() {
    let (_, refused) = snapshot();
    assert!(!refused.is_empty(), "snapshot lists no refused names");
    for name in &refused {
        assert!(
            d3bj_params_for_functional(name).is_err(),
            "{name}: upstream has no D3(BJ) fit, ferric must refuse it"
        );
    }
}
