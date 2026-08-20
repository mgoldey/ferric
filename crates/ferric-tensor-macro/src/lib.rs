//! Compile-time `einsum!` macro for ferric tensor contractions.
//!
//! `einsum!("ijcd,abcd->ijab", &a, &b)` parses the spec, derives the batch,
//! free, and contracted axis positions for each operand, and emits a call to
//! `ferric_tensors::einsum::einsum_binary_batched`. A scalar spec (`"ij,ij->"`)
//! emits code that returns the single element as `f64`.
//!
//! Index classification (an index is one letter):
//! - appears in both inputs, NOT in output -> contracted (summed, the GEMM axis)
//! - appears in both inputs AND in output   -> batch / diagonal (element-wise)
//! - appears in one input and in the output -> free
//!
//! An optional trailing argument (after the two operands) is a scale factor
//! applied to the whole result: `einsum!("ai,ai->", &mu, &u, -4.0)`.
//!
//! ## Panic-on-shape-mismatch is deliberate, not an oversight
//!
//! The emitted call to `einsum_binary_batched` (and the scalar-result unwrap)
//! `.expect()`s the `Result<_, TensorError>` rather than propagating it. This
//! was evaluated against this codebase's general "never panic on
//! data-dependent conditions, propagate `Result`" convention (see
//! `docs/guide/dev/03-reliability-conventions.md`) and judged a deliberate,
//! narrower exception — analogous to Rust's own array-index-out-of-bounds
//! panic:
//!
//! - `TensorError::ContractedDimMismatch` / `OutputReshape` fire only when
//!   the two operands of *one* contraction disagree on a dimension the macro
//!   itself computed should match (from the spec string's shared index
//!   letters). Every audited call site (RI-MP2, MP3, CCD, CCSD,
//!   CCSD(T) — `crates/ferric-mp2/src/{rimp2,mp3,canonical}.rs`,
//!   `crates/ferric-cc/src/{ccd,ccsd,ccsd_closed_shell,ccsd_t}.rs`, ~200
//!   call sites total) computes its occupied/virtual/aux dimensions ONCE per
//!   function from validated inputs (`nocc_total`,
//!   `ferric_mp2::rimp2::active_occ`-checked `frozen_core`, `nbas`) and
//!   threads those same dimensions into every `Tensor`/`ArrayD` built
//!   downstream in that function — so a genuine runtime shape mismatch
//!   inside one `einsum!` call can only happen via an internal
//!   contraction-spec bug (wrong axis order, a copy-pasted spec string, an
//!   intermediate built from the wrong variable), never from a user varying
//!   molecule size, basis set, or `frozen_core`: every legal value of those
//!   inputs scales all operands in a given function consistently.
//! - The *labeling* variant of this same bug class (contracting the wrong
//!   semantic axis, e.g. an occupied index against a virtual one that
//!   happens to share a dimension size) is already caught earlier and more
//!   precisely by the `debug_assert_eq!` axis-label check emitted above
//!   (compares `Tensor` axis labels, not just raw dimension sizes) — that
//!   check gives a clearer message and fires in every debug build. The
//!   release-only `.expect()` on `TensorError` is the backstop for the rarer
//!   case of two *different* semantic axes that happen to share a dimension
//!   size, or a bare unlabeled `ArrayD` operand (which skips the label check
//!   entirely).
//! - Cross-call contract violations (e.g. calling `ccsd_t` with a `CcConfig`
//!   whose `frozen_core` differs from the one used to produce the `CcResult`
//!   it's fed) are a different, pre-existing sharp edge in those functions'
//!   *call contracts*, not something `einsum!`'s error handling reaches: any
//!   two operands inside one `einsum!` call there are still built from the
//!   SAME (possibly wrong) dimensions within that one function body, so
//!   there is nothing for `TensorError` to observe — the mismatch would
//!   surface, if at all, as a plain out-of-bounds `ndarray` index panic (or
//!   worse, a silent short read) at the point the stale tensor is sliced,
//!   entirely outside `einsum!`. Propagating `TensorError` would not close
//!   that gap; it is out of scope for this macro.
//!
//! This is therefore treated the same as an out-of-bounds slice index: a
//! crash that can only indicate a programming bug in the contraction being
//! assembled, not a legitimate-but-unusual runtime state a caller could hit
//! by varying input data. It does NOT fall under the "solver honesty" /
//! "TS/MBD hard-error" family of conventions, which exist specifically for
//! conditions a valid user input CAN trigger (non-convergence, an element
//! outside a physical model's validity domain, a singular dielectric).
//!
//! If you add a new `einsum!` call site whose two operands are NOT both
//! derived from the same in-scope dimension variables within one function
//! (e.g. a helper that receives two tensors built by different,
//! independently configured callers), reconsider this reasoning for that
//! call site specifically — the safe default above assumes
//! single-function-scoped dimension provenance, which may not hold there.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse::Parser, punctuated::Punctuated, Expr, Token};

/// Compile-time einsum: `einsum!("ij,jk->ik", &a, &b)` lowers to BLAS3 GEMM.
#[proc_macro]
pub fn einsum(input: TokenStream) -> TokenStream {
    let parser = Punctuated::<Expr, Token![,]>::parse_terminated;
    let args = match parser.parse(input) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    let mut it = args.into_iter();
    let spec_expr = match it.next() { Some(e) => e, None => return compile_err("einsum! needs a spec string") };
    let lhs = match it.next() { Some(e) => e, None => return compile_err("einsum! needs a left operand") };
    let rhs = match it.next() { Some(e) => e, None => return compile_err("einsum! needs a right operand") };
    // Optional scale factor as a 4th argument; defaults to 1.0.
    let scale_expr: Expr = match it.next() {
        Some(e) => e,
        None => syn::parse_quote!(1.0_f64),
    };
    if it.next().is_some() {
        return syn::Error::new_spanned(
            &spec_expr,
            "einsum! takes 2 operands and an optional scale: einsum!(spec, &a, &b[, scale])",
        )
        .to_compile_error().into();
    }

    let spec = match &spec_expr {
        Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }) => s.value(),
        _ => return syn::Error::new_spanned(&spec_expr, "einsum! spec must be a string literal")
            .to_compile_error().into(),
    };

    let parsed = match parse_spec(&spec) {
        Ok(p) => p,
        Err(msg) => return syn::Error::new_spanned(&spec_expr, msg).to_compile_error().into(),
    };

    let l_batch_v: Vec<usize> = parsed.left_batch.iter().map(|&x| x as usize).collect();
    let l_free_v: Vec<usize> = parsed.left_free.iter().map(|&x| x as usize).collect();
    let l_contr_v: Vec<usize> = parsed.left_contr.iter().map(|&x| x as usize).collect();
    let r_batch_v: Vec<usize> = parsed.right_batch.iter().map(|&x| x as usize).collect();
    let r_free_v: Vec<usize> = parsed.right_free.iter().map(|&x| x as usize).collect();
    let r_contr_v: Vec<usize> = parsed.right_contr.iter().map(|&x| x as usize).collect();
    let out_from_batch_v: Vec<usize> = parsed.out_from_batch.iter().map(|&x| x as usize).collect();
    let out_from_left_v: Vec<usize> = parsed.out_from_left.iter().map(|&x| x as usize).collect();
    let out_from_right_v: Vec<usize> = parsed.out_from_right.iter().map(|&x| x as usize).collect();
    let scalar = parsed.out.is_empty();

    // For the debug axis-label check, batch axes are also paired across operands.
    let contr_l: Vec<usize> = parsed.left_contr.iter().map(|&x| x as usize).collect();
    let contr_r: Vec<usize> = parsed.right_contr.iter().map(|&x| x as usize).collect();
    let batch_l: Vec<usize> = parsed.left_batch.iter().map(|&x| x as usize).collect();
    let batch_r: Vec<usize> = parsed.right_batch.iter().map(|&x| x as usize).collect();

    let core = quote! {{
        let __lhs = &(#lhs);
        let __rhs = &(#rhs);
        let __scale: f64 = (#scale_expr);
        #[cfg(debug_assertions)]
        {
            use ::ferric_tensors::MaybeLabeled;
            #(
                if let (Some(__la), Some(__ra)) =
                    (__lhs.axis_label(#contr_l), __rhs.axis_label(#contr_r)) {
                    debug_assert_eq!(
                        __la, __ra,
                        "einsum axis mismatch on contracted index: left {:?} vs right {:?}",
                        __la, __ra
                    );
                }
            )*
            #(
                if let (Some(__la), Some(__ra)) =
                    (__lhs.axis_label(#batch_l), __rhs.axis_label(#batch_r)) {
                    debug_assert_eq!(
                        __la, __ra,
                        "einsum axis mismatch on batch index: left {:?} vs right {:?}",
                        __la, __ra
                    );
                }
            )*
        }
        let __l = __lhs.view();
        let __r = __rhs.view();
        let mut __out_shape: ::std::vec::Vec<usize> = ::std::vec::Vec::new();
        #( __out_shape.push(__l.shape()[#out_from_batch_v]); )*
        #( __out_shape.push(__l.shape()[#out_from_left_v]); )*
        #( __out_shape.push(__r.shape()[#out_from_right_v]); )*
        ::ferric_tensors::einsum::einsum_binary_batched(
            __l,
            &[#(#l_batch_v),*],
            &[#(#l_free_v),*],
            &[#(#l_contr_v),*],
            __r,
            &[#(#r_batch_v),*],
            &[#(#r_free_v),*],
            &[#(#r_contr_v),*],
            &__out_shape,
            __scale,
        ).expect("einsum_binary_batched failed")
    }};

    if scalar {
        quote! {{
            let __res = #core;
            *__res.iter().next().expect("einsum scalar: empty result")
        }}.into()
    } else {
        core.into()
    }
}

fn compile_err(msg: &str) -> TokenStream {
    syn::Error::new(proc_macro2::Span::call_site(), msg).to_compile_error().into()
}

struct Parsed {
    left_batch: Vec<u8>,
    left_free: Vec<u8>,
    left_contr: Vec<u8>,
    right_batch: Vec<u8>,
    right_free: Vec<u8>,
    right_contr: Vec<u8>,
    out: Vec<char>,
    out_from_batch: Vec<u8>,
    out_from_left: Vec<u8>,
    out_from_right: Vec<u8>,
}

fn parse_spec(spec: &str) -> Result<Parsed, String> {
    let spec = spec.replace(' ', "");
    let (ins, out) = spec.split_once("->").ok_or_else(|| "einsum spec must contain '->'".to_string())?;
    let (la, ra) = ins.split_once(',').ok_or_else(|| "einsum spec must have exactly two comma-separated inputs".to_string())?;
    if ra.contains(',') { return Err("einsum! is binary: exactly two inputs".to_string()); }
    let lchars: Vec<char> = la.chars().collect();
    let rchars: Vec<char> = ra.chars().collect();
    let ochars: Vec<char> = out.chars().collect();

    let in_out = |c: char| ochars.contains(&c);
    let in_right = |c: char| rchars.contains(&c);
    let in_left = |c: char| lchars.contains(&c);

    // Classify each left index. A shared index (in both inputs) is a BATCH axis
    // when it also appears in the output, otherwise it is CONTRACTED. A
    // non-shared index is FREE (and must appear in the output, checked below).
    let mut left_batch = Vec::new();
    let mut left_free = Vec::new();
    let mut left_contr = Vec::new();
    for (pos, &c) in lchars.iter().enumerate() {
        if in_right(c) {
            if in_out(c) { left_batch.push((c, pos as u8)); } else { left_contr.push((c, pos as u8)); }
        } else {
            left_free.push((c, pos as u8));
        }
    }
    let mut right_batch = Vec::new();
    let mut right_free = Vec::new();
    let mut right_contr = Vec::new();
    for (pos, &c) in rchars.iter().enumerate() {
        if in_left(c) {
            if in_out(c) { right_batch.push((c, pos as u8)); } else { right_contr.push((c, pos as u8)); }
        } else {
            right_free.push((c, pos as u8));
        }
    }
    // order right_contr to match left_contr letters
    let mut right_contr_ordered = Vec::new();
    for (lc, _) in &left_contr {
        let p = right_contr.iter().find(|(rc, _)| rc == lc)
            .ok_or_else(|| format!("contracted index '{lc}' missing from right operand"))?;
        right_contr_ordered.push(*p);
    }
    if right_contr_ordered.len() != right_contr.len() {
        return Err("mismatched contracted indices between operands".to_string());
    }
    // order right_batch to match left_batch letters
    let mut right_batch_ordered = Vec::new();
    for (lc, _) in &left_batch {
        let p = right_batch.iter().find(|(rc, _)| rc == lc)
            .ok_or_else(|| format!("batch index '{lc}' missing from right operand"))?;
        right_batch_ordered.push(*p);
    }

    // Every free (non-contracted, non-batch) input index must appear in the
    // output; an input index that is neither contracted, batch, nor output
    // would be an implicit single-operand sum, which we do not support.
    for (c, _) in left_free.iter().chain(right_free.iter()) {
        if !ochars.contains(c) {
            return Err(format!(
                "index '{c}' appears in an input but is neither contracted nor in the output \
                 (implicit single-operand sums are not supported)"
            ));
        }
    }

    #[derive(PartialEq)]
    enum Side { Batch, Left, Right }
    let mut out_from_batch = Vec::new();
    let mut out_from_left = Vec::new();
    let mut out_from_right = Vec::new();
    let mut provenance = Vec::new();
    for &oc in &ochars {
        if let Some((_, p)) = left_batch.iter().find(|(c, _)| *c == oc) {
            // Batch positions are reported as left-operand axis positions; the
            // runtime reads batch extents from the left operand.
            out_from_batch.push(*p);
            provenance.push(Side::Batch);
        } else if let Some((_, p)) = left_free.iter().find(|(c, _)| *c == oc) {
            out_from_left.push(*p);
            provenance.push(Side::Left);
        } else if let Some((_, p)) = right_free.iter().find(|(c, _)| *c == oc) {
            out_from_right.push(*p);
            provenance.push(Side::Right);
        } else {
            return Err(format!("output index '{oc}' is not a free or batch index of either input"));
        }
    }
    // The runtime emits axes in (batch..., left-free..., right-free...) order.
    // The output spec must list them in that order.
    let rank = |s: &Side| match s { Side::Batch => 0, Side::Left => 1, Side::Right => 2 };
    let mut max_rank = 0;
    for side in &provenance {
        let r = rank(side);
        if r < max_rank {
            return Err(
                "output indices must be listed as (batch indices, then left-operand free \
                 indices, then right-operand free indices) to match the runtime axis order"
                    .to_string(),
            );
        }
        max_rank = r;
    }

    Ok(Parsed {
        left_batch: left_batch.iter().map(|(_, p)| *p).collect(),
        left_free: left_free.iter().map(|(_, p)| *p).collect(),
        left_contr: left_contr.iter().map(|(_, p)| *p).collect(),
        right_batch: right_batch_ordered.iter().map(|(_, p)| *p).collect(),
        right_free: right_free.iter().map(|(_, p)| *p).collect(),
        right_contr: right_contr_ordered.iter().map(|(_, p)| *p).collect(),
        out: ochars,
        out_from_batch,
        out_from_left,
        out_from_right,
    })
}
