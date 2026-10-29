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
    let spec_expr = match it.next() {
        Some(e) => e,
        None => return compile_err("einsum! needs a spec string"),
    };
    let lhs = match it.next() {
        Some(e) => e,
        None => return compile_err("einsum! needs a left operand"),
    };
    let rhs = match it.next() {
        Some(e) => e,
        None => return compile_err("einsum! needs a right operand"),
    };
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
        .to_compile_error()
        .into();
    }

    let spec = match &spec_expr {
        Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(s),
            ..
        }) => s.value(),
        _ => {
            return syn::Error::new_spanned(&spec_expr, "einsum! spec must be a string literal")
                .to_compile_error()
                .into()
        }
    };

    let parsed = match ferric_einsum_spec::parse(&spec) {
        Ok(p) => p,
        Err(msg) => {
            return syn::Error::new_spanned(&spec_expr, msg)
                .to_compile_error()
                .into()
        }
    };

    let l_batch_v: Vec<usize> = parsed.left_batch.clone();
    let l_free_v: Vec<usize> = parsed.left_free.clone();
    let l_contr_v: Vec<usize> = parsed.left_contr.clone();
    let r_batch_v: Vec<usize> = parsed.right_batch.clone();
    let r_free_v: Vec<usize> = parsed.right_free.clone();
    let r_contr_v: Vec<usize> = parsed.right_contr.clone();
    let out_from_batch_v: Vec<usize> = parsed.out_from_batch.clone();
    let out_from_left_v: Vec<usize> = parsed.out_from_left.clone();
    let out_from_right_v: Vec<usize> = parsed.out_from_right.clone();
    let scalar = parsed.out.is_empty();

    // For the debug axis-label check, batch axes are also paired across operands.
    let contr_l: Vec<usize> = parsed.left_contr.clone();
    let contr_r: Vec<usize> = parsed.right_contr.clone();
    let batch_l: Vec<usize> = parsed.left_batch.clone();
    let batch_r: Vec<usize> = parsed.right_batch.clone();

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
        }}
        .into()
    } else {
        core.into()
    }
}

fn compile_err(msg: &str) -> TokenStream {
    syn::Error::new(proc_macro2::Span::call_site(), msg)
        .to_compile_error()
        .into()
}
