//! Compile-time `einsum!` macro for ferric tensor contractions.
//!
//! Parses and validates an Einstein-summation index spec at compile time and
//! emits a call to `ferric_tensors::einsum::einsum_binary`. The macro contains
//! no arithmetic — only spec parsing/validation and code emission.

use proc_macro::TokenStream;

#[proc_macro]
pub fn einsum(_input: TokenStream) -> TokenStream {
    // Filled in Task 5.
    TokenStream::new()
}
