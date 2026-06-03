//! Compile-time `einsum!` macro for ferric tensor contractions.
//!
//! `einsum!("ijcd,abcd->ijab", &a, &b)` parses the spec, derives the free and
//! contracted axis positions for each operand, and emits a call to
//! `ferric_tensors::einsum::einsum_binary`. A scalar spec (`"ij,ij->"`) emits
//! code that returns the single element as `f64`.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse::Parser, punctuated::Punctuated, Expr, Token};

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
    if it.next().is_some() {
        return syn::Error::new_spanned(&spec_expr, "einsum! takes exactly 2 operands (binary)")
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

    let l_free_v: Vec<usize> = parsed.left_free.iter().map(|&x| x as usize).collect();
    let l_contr_v: Vec<usize> = parsed.left_contr.iter().map(|&x| x as usize).collect();
    let r_free_v: Vec<usize> = parsed.right_free.iter().map(|&x| x as usize).collect();
    let r_contr_v: Vec<usize> = parsed.right_contr.iter().map(|&x| x as usize).collect();
    let out_from_left_v: Vec<usize> = parsed.out_from_left.iter().map(|&x| x as usize).collect();
    let out_from_right_v: Vec<usize> = parsed.out_from_right.iter().map(|&x| x as usize).collect();
    let scalar = parsed.out.is_empty();

    let core = quote! {{
        let __l = (#lhs).view();
        let __r = (#rhs).view();
        let mut __out_shape: ::std::vec::Vec<usize> = ::std::vec::Vec::new();
        #( __out_shape.push(__l.shape()[#out_from_left_v]); )*
        #( __out_shape.push(__r.shape()[#out_from_right_v]); )*
        ::ferric_tensors::einsum::einsum_binary(
            __l,
            &[#(#l_free_v),*],
            &[#(#l_contr_v),*],
            __r,
            &[#(#r_free_v),*],
            &[#(#r_contr_v),*],
            &__out_shape,
        ).expect("einsum_binary failed")
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
    left_free: Vec<u8>,
    left_contr: Vec<u8>,
    right_free: Vec<u8>,
    right_contr: Vec<u8>,
    out: Vec<char>,
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

    let mut left_free = Vec::new();
    let mut left_contr = Vec::new();
    for (pos, &c) in lchars.iter().enumerate() {
        if in_right(c) && !in_out(c) { left_contr.push((c, pos as u8)); } else { left_free.push((c, pos as u8)); }
    }
    let mut right_free = Vec::new();
    let mut right_contr = Vec::new();
    for (pos, &c) in rchars.iter().enumerate() {
        if in_left(c) && !in_out(c) { right_contr.push((c, pos as u8)); } else { right_free.push((c, pos as u8)); }
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

    let mut out_from_left = Vec::new();
    let mut out_from_right = Vec::new();
    for &oc in &ochars {
        if let Some((_, p)) = left_free.iter().find(|(c, _)| *c == oc) {
            out_from_left.push(*p);
        } else if let Some((_, p)) = right_free.iter().find(|(c, _)| *c == oc) {
            out_from_right.push(*p);
        } else {
            return Err(format!("output index '{oc}' is not a free index of either input"));
        }
    }

    Ok(Parsed {
        left_free: left_free.iter().map(|(_, p)| *p).collect(),
        left_contr: left_contr.iter().map(|(_, p)| *p).collect(),
        right_free: right_free.iter().map(|(_, p)| *p).collect(),
        right_contr: right_contr_ordered.iter().map(|(_, p)| *p).collect(),
        out: ochars,
        out_from_left,
        out_from_right,
    })
}
