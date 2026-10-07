//! The pre-registered decision rules of the mixed-GEMM panel sweep
//! (`benchmarks/harness/examples/gpu_mixed_gemm_sweep.rs`), as pure functions
//! over the measured table so they are unit-tested without a device and a
//! naming mismatch between the sweep's arm names and the rule lookups fails a
//! CI test (and the smoke run) instead of a full sweep.
//!
//! Arm names (resident arms and `cpu_f64` match EXACTLY; `cpu_mixed_b*` and
//! `gpu_split3_resident_128*` carry a variable suffix and match by prefix):
//! `cpu_f64`, `cpu_mixed_b<b>`, `gpu_sgemm_resident`,
//! `gpu_mixed_resident_b<p>`, `gpu_split3_resident_128 (...)`.

/// Panel widths the sweep tries, ascending (ties go to the smaller, the more accurate).
pub const PANELS: [usize; 5] = [64, 128, 256, 512, 1024];

/// One arm's row for one shape: median seconds and rms normalized error.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmRow {
    pub name: String,
    pub median_s: f64,
    pub rms: f64,
}

/// All arm rows measured at one `(m, k, n)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeRows {
    pub shape: (usize, usize, usize),
    pub rows: Vec<ArmRow>,
}

/// Verdicts of the three rules.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleOutcome {
    /// Smallest panel whose resident mixed throughput is >= 0.25 x the resident
    /// plain SGEMM at every `panel_rule_shapes`; `None` if no panel qualifies.
    pub candidate: Option<usize>,
    /// rms(b)/rms(c) >= 2 at b = 128 AND t(c) <= 2 t(b), at every shape.
    pub variant_c_fires: bool,
    /// t(cpu_mixed) <= 0.6 t(cpu_f64) at every shape.
    pub cpu_counterpart_fires: bool,
}

fn lookup(shape: &ShapeRows, name: &str, prefix: bool) -> Result<ArmRow, String> {
    shape
        .rows
        .iter()
        .find(|r| {
            if prefix {
                r.name.starts_with(name)
            } else {
                r.name == name
            }
        })
        .cloned()
        .ok_or_else(|| {
            let have: Vec<&str> = shape.rows.iter().map(|r| r.name.as_str()).collect();
            format!(
                "shape {:?}: arm {name:?} ({}) not in the table; have {have:?}",
                shape.shape,
                if prefix {
                    "prefix match"
                } else {
                    "exact match"
                }
            )
        })
}

/// Evaluate the three rules. `panel_rule_shapes` are the shapes the panel rule
/// must see (the sweep passes `(393,912,5895)` and `(256,8192,256)`; the smoke
/// run passes the one shape it ran); a missing shape or arm is an `Err`
/// naming it, never a panic.
pub fn evaluate_rules(
    table: &[ShapeRows],
    panel_rule_shapes: &[(usize, usize, usize)],
) -> Result<RuleOutcome, String> {
    let row_of = |sh: (usize, usize, usize)| {
        table
            .iter()
            .find(|r| r.shape == sh)
            .ok_or_else(|| format!("shape {sh:?} required by the panel rule is not in the table"))
    };
    // Look up EVERY panel at every rule shape (no early exit), so a missing
    // arm is reported even when a smaller panel already qualified.
    let mut candidate = None;
    for &p in &PANELS {
        let mut qualifies = true;
        for &sh in panel_rule_shapes {
            let row = row_of(sh)?;
            let sgemm = lookup(row, "gpu_sgemm_resident", false)?;
            let mixed = lookup(row, &format!("gpu_mixed_resident_b{p}"), false)?;
            // same flops, so throughput ratio = inverse time ratio
            qualifies &= sgemm.median_s / mixed.median_s >= 0.25;
        }
        if qualifies && candidate.is_none() {
            candidate = Some(p);
        }
    }
    // Validate every arm the all-shape rules use, even if an earlier shape
    // already decided the verdict (no short-circuit hiding a missing arm).
    let (mut c_time_ok, mut c_err_ok, mut cpu_ok) = (true, true, true);
    for sh in table {
        let b = lookup(sh, "gpu_mixed_resident_b128", false)?;
        let c = lookup(sh, "gpu_split3_resident_128", true)?;
        let f64_cpu = lookup(sh, "cpu_f64", false)?;
        let mixed_cpu = lookup(sh, "cpu_mixed_b", true)?;
        c_time_ok &= c.median_s <= 2.0 * b.median_s;
        c_err_ok &= b.rms / c.rms >= 2.0;
        cpu_ok &= mixed_cpu.median_s <= 0.6 * f64_cpu.median_s;
    }
    Ok(RuleOutcome {
        candidate,
        variant_c_fires: c_time_ok && c_err_ok,
        cpu_counterpart_fires: cpu_ok,
    })
}

/// The lines the sweep prints for an outcome.
pub fn render(o: &RuleOutcome) -> Vec<String> {
    let fire = |b: bool| if b { "fires" } else { "does not fire" };
    vec![
        match o.candidate {
            Some(p) => format!("candidate MIXED_K_PANEL_DEFAULT = {p}"),
            None => format!("candidate MIXED_K_PANEL_DEFAULT = none of {PANELS:?} meets the 0.25 rule"),
        },
        format!(
            "variant (c) rule: ship only if rms(b)/rms(c) >= 2 at b=128 AND t(c) <= 2 t(b) at every shape -> {}",
            fire(o.variant_c_fires)
        ),
        format!(
            "CPU counterpart rule (§3.6): t(cpu_mixed) <= 0.6 t(cpu_f64) at every shape -> {}",
            fire(o.cpu_counterpart_fires)
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const S1: (usize, usize, usize) = (256, 8192, 256);
    const S2: (usize, usize, usize) = (393, 912, 5895);

    fn row(name: &str, t: f64, rms: f64) -> ArmRow {
        ArmRow {
            name: name.into(),
            median_s: t,
            rms,
        }
    }

    /// Every arm with the names the sweep really prints. `t_mixed(p)` gives the
    /// resident mixed time for panel p; sgemm takes 1.0 s.
    fn shape(sh: (usize, usize, usize), t_mixed: impl Fn(usize) -> f64) -> ShapeRows {
        let mut rows = vec![
            row("cpu_f64", 1.0, 0.0),
            row("cpu_mixed_b128", 0.5, 1e-8),
            row("gpu_f64_resident", 5.0, 1e-16),
            row("gpu_sgemm_resident", 1.0, 1e-7),
        ];
        for &p in &PANELS {
            rows.push(row(&format!("gpu_mixed_resident_b{p}"), t_mixed(p), 5e-9));
        }
        rows.push(row("gpu_mixed_percall_128", 6.0, 5e-9));
        rows.push(row("gpu_split3_resident_128 (host sum untimed)", 3.0, 5e-9));
        ShapeRows { shape: sh, rows }
    }

    #[test]
    fn smallest_qualifying_panel_wins_when_the_rule_fires() {
        // b=64 too slow (ratio 0.2), b=128 qualifies (0.5) at both shapes.
        let t = |p: usize| if p == 64 { 5.0 } else { 2.0 };
        let o = evaluate_rules(&[shape(S1, t), shape(S2, t)], &[S1, S2]).unwrap();
        assert_eq!(o.candidate, Some(128));
    }

    #[test]
    fn rule_does_not_fire_when_no_panel_reaches_a_quarter_of_sgemm() {
        let o = evaluate_rules(&[shape(S1, |_| 8.0), shape(S2, |_| 8.0)], &[S1, S2]).unwrap();
        assert_eq!(o.candidate, None);
        assert!(render(&o)[0].contains("none of"));
    }

    #[test]
    fn a_panel_must_qualify_at_both_shapes() {
        // b=64 fine at S1, too slow at S2 -> b=128.
        let t1 = |_p: usize| 2.0;
        let t2 = |p: usize| if p == 64 { 5.0 } else { 2.0 };
        let o = evaluate_rules(&[shape(S1, t1), shape(S2, t2)], &[S1, S2]).unwrap();
        assert_eq!(o.candidate, Some(128));
    }

    #[test]
    fn an_exact_tie_at_the_threshold_qualifies_and_ties_go_to_the_smaller_panel() {
        // sgemm/mixed = 1.0/4.0 = 0.25 exactly, for every panel.
        let o = evaluate_rules(&[shape(S1, |_| 4.0), shape(S2, |_| 4.0)], &[S1, S2]).unwrap();
        assert_eq!(o.candidate, Some(64));
    }

    #[test]
    fn variant_c_needs_both_the_accuracy_gain_and_the_time_bound() {
        let mut s = shape(S1, |_| 2.0);
        let o = evaluate_rules(&[s.clone()], &[S1]).unwrap();
        assert!(!o.variant_c_fires, "rms ratio 1.0 < 2 must not fire");
        // c twice as accurate and t(c) = 3.0 <= 2*2.0
        s.rows.last_mut().unwrap().rms = 2.5e-9;
        assert!(evaluate_rules(&[s.clone()], &[S1]).unwrap().variant_c_fires);
        // accurate but too slow: t(c) = 4.5 > 2*2.0
        s.rows.last_mut().unwrap().median_s = 4.5;
        assert!(!evaluate_rules(&[s], &[S1]).unwrap().variant_c_fires);
    }

    #[test]
    fn cpu_rule_is_checked_at_every_shape() {
        let fast = shape(S1, |_| 2.0); // cpu_mixed 0.5 <= 0.6 * 1.0
        let mut slow = shape(S2, |_| 2.0);
        assert!(
            evaluate_rules(&[fast.clone(), slow.clone()], &[S1, S2])
                .unwrap()
                .cpu_counterpart_fires
        );
        slow.rows[1].median_s = 0.7;
        assert!(
            !evaluate_rules(&[fast, slow], &[S1, S2])
                .unwrap()
                .cpu_counterpart_fires
        );
    }

    #[test]
    fn a_missing_arm_or_shape_is_a_clear_error_not_a_panic() {
        let mut s = shape(S1, |_| 2.0);
        s.rows.retain(|r| r.name != "gpu_mixed_resident_b256");
        let e = evaluate_rules(&[s.clone()], &[S1]).unwrap_err();
        assert!(
            e.contains("gpu_mixed_resident_b256") && e.contains("not in the table"),
            "{e}"
        );
        let e = evaluate_rules(&[s], &[S1, S2]).unwrap_err();
        assert!(
            e.contains("not in the table") || e.contains("required"),
            "{e}"
        );
        // the old trailing-space lookup must NOT match a real name
        let ok = shape(S1, |_| 2.0);
        assert!(lookup(&ok, "gpu_mixed_resident_b128 ", false).is_err());
        assert!(lookup(&ok, "gpu_mixed_resident_b128", false).is_ok());
        // exact match: b128 must not be satisfied by the b1024 row
        let mut only_1024 = ok;
        only_1024
            .rows
            .retain(|r| r.name != "gpu_mixed_resident_b128");
        assert!(lookup(&only_1024, "gpu_mixed_resident_b128", false).is_err());
    }
}
