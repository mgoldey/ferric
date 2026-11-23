//! Typed accessors for the JSON reference files under `testdata/reference`.
//! A missing or mistyped field panics with the pointer, so a stale reference
//! file fails loudly at the field that moved.

use serde_json::Value;

/// The `f64` at JSON pointer `ptr` in `v`. `ctx` names the case in the panic.
pub fn num_ctx(v: &Value, ptr: &str, ctx: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a number"))
}

/// As [`num_ctx`], for call sites with no case name to report.
pub fn num(v: &Value, ptr: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("reference field {ptr} missing or not a number"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reads_numbers() {
        let v = json!({"a": {"b": 1.5}});
        assert_eq!(num_ctx(&v, "/a/b", "case"), 1.5);
        assert_eq!(num(&v, "/a/b"), 1.5);
    }

    #[test]
    #[should_panic(expected = "case: reference field /a/c missing or not a number")]
    fn num_ctx_names_case_and_field() {
        num_ctx(&json!({"a": {"b": 1.5}}), "/a/c", "case");
    }

    #[test]
    #[should_panic(expected = "reference field /a/b missing or not a number")]
    fn num_rejects_non_number() {
        num(&json!({"a": {"b": "x"}}), "/a/b");
    }
}
