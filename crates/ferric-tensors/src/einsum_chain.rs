//! Three or more operands as a fixed left-to-right chain of binary
//! contractions: `"ij,jk,kl->il"` is `((A.B).C)`.
//!
//! No contraction-order search. The chain is the order the spec is written in,
//! so the result is a deterministic function of the spec. Each step is one
//! [`einsum_dyn`] call under the binary spec rules, so those rules (no repeated
//! index in a term, no implicit single-operand sums, output listed as batch,
//! left-free, right-free) apply per step:
//!
//! - an index shared by the running result and the next operand is summed at
//!   that step unless a LATER operand or the final output still carries it, in
//!   which case it is a batch (element-wise) index of the step;
//! - an index in only one of the two that no later operand or the output
//!   carries would be a single-operand sum, which the engine refuses; the
//!   error names the step;
//! - the final output must follow the binary group order relative to the last
//!   step's two terms.
//!
//! The intermediate's index order is (batch, left-free, right-free), each in
//! term order. A spec with two operands is exactly [`einsum_dyn`].

use crate::einsum::{einsum_dyn, TensorError};
use ndarray::{ArrayD, ArrayViewD};
use std::collections::HashMap;

/// Contract `operands` (two or more) under `spec` as a left-to-right chain,
/// returning `scale` times the result. `scale` is applied in the last step.
pub fn einsum_chain_dyn(
    spec: &str,
    operands: &[ArrayViewD<f64>],
    scale: f64,
) -> Result<ArrayD<f64>, TensorError> {
    if let Some((ins, _)) = spec.split_once("->") {
        let n_terms = ins.split(',').count();
        if n_terms != operands.len() {
            return Err(TensorError::Operand(format!(
                "spec '{spec}' has {n_terms} input terms but {} operands were given",
                operands.len()
            )));
        }
    }
    match operands {
        [a, b] => return einsum_dyn(spec, a.view(), b.view(), scale),
        [] | [_] => {
            return Err(TensorError::Operand(
                "einsum needs at least two operands".into(),
            ))
        }
        _ => {}
    }
    let spec: String = spec.chars().filter(|c| *c != ' ').collect();
    let (ins, out) = spec
        .split_once("->")
        .ok_or_else(|| TensorError::Spec("einsum spec must contain '->'".into()))?;
    let terms: Vec<Vec<char>> = ins.split(',').map(|t| t.chars().collect()).collect();
    check_extents(&spec, &terms, operands)?;
    let out_term: Vec<char> = out.chars().collect();
    let mut acc_term = terms[0].clone();
    let mut acc: Option<ArrayD<f64>> = None;
    for k in 1..operands.len() {
        let last = k == operands.len() - 1;
        let right = &terms[k];
        let step_out: Vec<char> = if last {
            out_term.clone()
        } else {
            let later =
                |c: &char| terms[k + 1..].iter().any(|t| t.contains(c)) || out_term.contains(c);
            let batch = acc_term.iter().filter(|c| right.contains(c) && later(c));
            let lfree = acc_term.iter().filter(|c| !right.contains(c) && later(c));
            let rfree = right.iter().filter(|c| !acc_term.contains(c) && later(c));
            batch.chain(lfree).chain(rfree).copied().collect()
        };
        let step_spec = format!(
            "{},{}->{}",
            acc_term.iter().collect::<String>(),
            right.iter().collect::<String>(),
            step_out.iter().collect::<String>()
        );
        let left_view = match &acc {
            Some(a) => a.view(),
            None => operands[0].view(),
        };
        let step_scale = if last { scale } else { 1.0 };
        let next = einsum_dyn(&step_spec, left_view, operands[k].view(), step_scale)
            .map_err(|e| step_error(&spec, k, &step_spec, e))?;
        acc = Some(next);
        acc_term = step_out;
    }
    Ok(acc.expect("a chain of at least three operands runs a step"))
}

/// Every index has one extent across all terms, and each operand's rank is its
/// term's length. Checked before any step runs.
fn check_extents(
    spec: &str,
    terms: &[Vec<char>],
    operands: &[ArrayViewD<f64>],
) -> Result<(), TensorError> {
    let mut extent: HashMap<char, usize> = HashMap::new();
    for (k, (term, op)) in terms.iter().zip(operands).enumerate() {
        if term.len() != op.ndim() {
            return Err(TensorError::Operand(format!(
                "operand {k} has {} axes, spec '{spec}' gives it {}",
                op.ndim(),
                term.len()
            )));
        }
        for (&c, &n) in term.iter().zip(op.shape()) {
            if let Some(&prev) = extent.get(&c) {
                if prev != n {
                    return Err(TensorError::Operand(format!(
                        "index '{c}' has extent {prev} in an earlier operand but {n} in operand {k}"
                    )));
                }
            } else {
                extent.insert(c, n);
            }
        }
    }
    Ok(())
}

fn step_error(spec: &str, k: usize, step_spec: &str, e: TensorError) -> TensorError {
    let note = format!(
        "in step {k} of the left-to-right chain for '{spec}' (the binary contraction \
         '{step_spec}')"
    );
    match e {
        TensorError::Spec(m) => TensorError::Spec(format!("{m} {note}")),
        TensorError::Operand(m) => TensorError::Operand(format!("{m} {note}")),
        other => other,
    }
}
