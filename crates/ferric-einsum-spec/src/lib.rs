//! Parse a binary einsum spec (`"ijcd,abcd->ijab"`) into axis positions.
//!
//! Shared by the compile-time `einsum!` macro and the run-time
//! `ferric_tensors::einsum::einsum_dyn`, so both accept exactly the same specs
//! and lower them to the same `einsum_binary_batched` call.
//!
//! Index classification (an index is one character):
//! - in both inputs, NOT in the output -> contracted (the GEMM axis)
//! - in both inputs AND in the output  -> batch / diagonal (element-wise)
//! - in one input and in the output    -> free
//!
//! The runtime emits output axes as (batch..., left-free..., right-free...).
//! The output must list its indices in that GROUP order, but within a group
//! any order is allowed: batch and free positions are reported in OUTPUT
//! order, and the runtime permutes each operand to match.

/// Axis positions for one binary contraction. Every `left_*` / `right_*`
/// vector holds positions into that operand; `out_from_*` give, in output
/// order, the operand axis each output axis takes its extent from (batch
/// extents are read from the left operand).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spec {
    pub left_rank: usize,
    pub right_rank: usize,
    pub left_batch: Vec<usize>,
    pub left_free: Vec<usize>,
    pub left_contr: Vec<usize>,
    pub right_batch: Vec<usize>,
    pub right_free: Vec<usize>,
    pub right_contr: Vec<usize>,
    pub out: Vec<char>,
    pub out_from_batch: Vec<usize>,
    pub out_from_left: Vec<usize>,
    pub out_from_right: Vec<usize>,
}

fn no_repeats(term: &[char], what: &str) -> Result<(), String> {
    for (i, c) in term.iter().enumerate() {
        if term[..i].contains(c) {
            return Err(format!(
                "index '{c}' is repeated in the {what} (traces / diagonals of one operand are \
                 not supported)"
            ));
        }
    }
    Ok(())
}

fn pos(term: &[char], c: char) -> usize {
    term.iter()
        .position(|&x| x == c)
        .expect("index taken from this term")
}

/// Parse `spec`. Spaces are ignored. Errors name the offending index.
pub fn parse(spec: &str) -> Result<Spec, String> {
    let spec = spec.replace(' ', "");
    let (ins, out) = spec
        .split_once("->")
        .ok_or_else(|| "einsum spec must contain '->'".to_string())?;
    let (la, ra) = ins
        .split_once(',')
        .ok_or_else(|| "einsum spec must have exactly two comma-separated inputs".to_string())?;
    if ra.contains(',') {
        return Err("einsum is binary: exactly two inputs".to_string());
    }
    let l: Vec<char> = la.chars().collect();
    let r: Vec<char> = ra.chars().collect();
    let o: Vec<char> = out.chars().collect();
    no_repeats(&l, "left operand")?;
    no_repeats(&r, "right operand")?;
    no_repeats(&o, "output")?;

    // Every non-shared input index must reach the output; one that does not
    // would be an implicit single-operand sum, which is not supported.
    for c in l
        .iter()
        .filter(|c| !r.contains(c))
        .chain(r.iter().filter(|c| !l.contains(c)))
    {
        if !o.contains(c) {
            return Err(format!(
                "index '{c}' appears in an input but is neither contracted nor in the output \
                 (implicit single-operand sums are not supported)"
            ));
        }
    }

    // Walk the output: each index is batch (in both inputs), left free or
    // right free, and the groups must come in that order.
    let (mut batch, mut lfree, mut rfree) = (Vec::new(), Vec::new(), Vec::new());
    let mut max_group = 0;
    for &c in &o {
        let group = match (l.contains(&c), r.contains(&c)) {
            (true, true) => {
                batch.push(c);
                0
            }
            (true, false) => {
                lfree.push(c);
                1
            }
            (false, true) => {
                rfree.push(c);
                2
            }
            (false, false) => {
                return Err(format!(
                    "output index '{c}' is not a free or batch index of either input"
                ))
            }
        };
        if group < max_group {
            return Err(
                "output indices must be listed as (batch indices, then left-operand free \
                        indices, then right-operand free indices) to match the runtime axis order"
                    .to_string(),
            );
        }
        max_group = group;
    }
    let contr: Vec<char> = l
        .iter()
        .copied()
        .filter(|c| r.contains(c) && !o.contains(c))
        .collect();

    let left_batch: Vec<usize> = batch.iter().map(|&c| pos(&l, c)).collect();
    let left_free: Vec<usize> = lfree.iter().map(|&c| pos(&l, c)).collect();
    let right_free: Vec<usize> = rfree.iter().map(|&c| pos(&r, c)).collect();
    Ok(Spec {
        left_rank: l.len(),
        right_rank: r.len(),
        right_batch: batch.iter().map(|&c| pos(&r, c)).collect(),
        left_contr: contr.iter().map(|&c| pos(&l, c)).collect(),
        right_contr: contr.iter().map(|&c| pos(&r, c)).collect(),
        out_from_batch: left_batch.clone(),
        out_from_left: left_free.clone(),
        out_from_right: right_free.clone(),
        left_batch,
        left_free,
        right_free,
        out: o,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_ccsd_vvvv() {
        let s = parse("ijcd,abcd->ijab").unwrap();
        assert_eq!((s.left_free, s.left_contr), (vec![0, 1], vec![2, 3]));
        assert_eq!((s.right_free, s.right_contr), (vec![0, 1], vec![2, 3]));
        assert!(s.left_batch.is_empty());
    }

    #[test]
    fn free_and_batch_follow_output_order() {
        let s = parse("ijk,kl->jil").unwrap();
        assert_eq!(s.left_free, vec![1, 0]);
        let s = parse("kij,kji->ji").unwrap();
        assert_eq!((s.left_batch, s.right_batch), (vec![2, 1], vec![1, 2]));
    }

    #[test]
    fn rejects_bad_specs() {
        for (spec, msg) in [
            ("ij,jk", "->"),
            ("ij->ij", "two comma"),
            ("ij,jk,kl->il", "binary"),
            ("ii,ij->j", "repeated"),
            ("ij,jk->k", "neither contracted"),
            ("ij,jk->ikz", "not a free or batch"),
            ("ij,jk->ki", "group order"),
        ] {
            let err = parse(spec).unwrap_err();
            assert!(
                err.contains(msg) || (msg == "group order" && err.contains("listed as")),
                "{spec}: {err}"
            );
        }
    }
}
