//! Orbital Padé MP3: the occupied-space matrices `W^(2)`, `W^(3)`, the matrix
//! Padé `P(A, B) = A (A − B)⁻¹ A`, and the OP-MP3, SOP-MP3 and SSD-OPMP3
//! models of Bystrom, Berkelbach and Hait (arXiv:2609.25330).
//!
//! From the spin-blocked amplitudes of [`crate::u_mp3`] (αα `[i,j,a,b]`, ββ
//! `[I,J,A,B]`, αβ `[i,J,a,B]`), each spin block of
//! `W^(n)_ij = ⅛ Σ_kab c^(n−1)_ikab ⟨jk‖ab⟩ + (i↔j)` splits by the spin of
//! `k` into a same-spin (SS) and an opposite-spin (OS) channel:
//!
//! ```text
//! W^SS_σ,ij = ⅛ Σ_{k a b ∈ σ}  c^σσ_ikab ⟨jk‖ab⟩     + (i↔j)
//! W^OS_α,ij = ¼ Σ_{K a B}       c^αβ_iKaB (ja|KB)    + (i↔j)
//! W^OS_β,IJ = ¼ Σ_{k a B}       c^αβ_kIaB (ka|JB)    + (I↔J)
//! ```
//!
//! so `Tr W^SS = E^(n)_αα + E^(n)_ββ` and `Tr W^OS = E^(n)_αβ`. The models
//! resum each matrix pair with [`pade_trace`](crate::opmp3::pade_trace),
//! spin block by spin block (equations and proofs:
//! `book/src/10-mp2/11-opmp3.md#opmp3`,
//! `scripts/opmp3_identities_sympy.py`).

use crate::u_mp3::{SpinBlockedDoubles, UMp3Result};
use ferric_core::FerricError;
use ndarray::{Array2, Array4, ArrayView2};
use ndarray_linalg::{Eigh, UPLO};

/// Spin index of α in the `[α, β]` arrays.
pub const ALPHA: usize = 0;
/// Spin index of β in the `[α, β]` arrays.
pub const BETA: usize = 1;

/// SSD-OPMP3 same-spin damping printed in the paper (Fig. 3 and the
/// analysis README; W4-17 nonMR TAE fits rounded to 0.05), by reference.
/// The names are lower case; `"hf"` is the Hartree–Fock reference.
pub const PAPER_X_SS: &[(&str, f64)] = &[
    ("hf", 0.0),
    ("pbe0", 0.50),
    ("pbe", 0.70),
    ("spw92", 0.75),
    ("r2scan", 0.60),
    ("wb97x-v", 0.50),
    ("wb97m-v", 0.50),
    ("koomp2", 0.45),
];

/// OP-MP3(x) scale fitted on κOOMP2 references (paper Fig. 2), printed value.
pub const PAPER_OP_MP3_X: f64 = 0.7890;

/// The paper's `x_ss` for a reference name (case-insensitive), if printed.
pub fn paper_x_ss(reference: &str) -> Option<f64> {
    let r = reference.to_ascii_lowercase();
    PAPER_X_SS.iter().find(|(k, _)| *k == r).map(|&(_, x)| x)
}

/// The SS and OS channels of one perturbation order, per spin `[α, β]`.
#[derive(Debug, Clone, PartialEq)]
pub struct WChannels {
    /// Same-spin `W^SS_σ`, `(no_σ, no_σ)`.
    pub ss: [Array2<f64>; 2],
    /// Opposite-spin `W^OS_σ`, `(no_σ, no_σ)`.
    pub os: [Array2<f64>; 2],
}

impl WChannels {
    /// `W^SS_σ + W^OS_σ`, the full spin block.
    pub fn full(&self, spin: usize) -> Array2<f64> {
        &self.ss[spin] + &self.os[spin]
    }

    /// `Tr W` over both spins and both channels.
    pub fn trace(&self) -> f64 {
        (0..2)
            .map(|s| self.ss[s].diag().sum() + self.os[s].diag().sum())
            .sum()
    }

    /// The same channels with every matrix multiplied by `x`.
    pub fn scaled(&self, x: f64) -> WChannels {
        WChannels {
            ss: [&self.ss[0] * x, &self.ss[1] * x],
            os: [&self.os[0] * x, &self.os[1] * x],
        }
    }
}

/// `W^(2)` (from `c^(1)`) and `W^(3)` (from `c^(2)`).
#[derive(Debug, Clone, PartialEq)]
pub struct WMatrices {
    /// Second order.
    pub w2: WChannels,
    /// Third order.
    pub w3: WChannels,
}

/// `X (rows × rest)` from a 4-index block `[r, ., ., .]` in standard layout.
fn as_rows(t: &Array4<f64>) -> ArrayView2<'_, f64> {
    let n = t.shape()[0];
    let rest = t.len().checked_div(n).unwrap_or(0);
    t.view()
        .into_shape_with_order((n, rest))
        .expect("u_mp3 blocks are in standard layout")
}

/// `f · (Y + Yᵀ)` with `Y_ij = Σ_rest c[i, rest] k[j, rest]`.
fn symmetrized_contraction(c: &Array4<f64>, k: &Array4<f64>, f: f64) -> Array2<f64> {
    let y = as_rows(c).dot(&as_rows(k).t());
    (&y + &y.t()) * f
}

/// The αβ block `[i, J, a, B]` reordered to `[J, i, a, B]`, standard layout.
fn beta_first(t: &Array4<f64>) -> Array4<f64> {
    t.view()
        .permuted_axes([1, 0, 2, 3])
        .as_standard_layout()
        .into_owned()
}

/// The SS and OS channels of `W` built from amplitudes `c` and integrals
/// `k` (the `⟨ij‖ab⟩` αα / ββ and `(ia|JB)` αβ blocks of [`UMp3Result::ints`]).
///
/// A spin with fewer than two occupied orbitals has no same-spin pair, so its
/// `W^SS` is set to exact zeros (its `c^σσ_iiab` vanish only to rounding).
pub fn w_channels(c: &SpinBlockedDoubles, k: &SpinBlockedDoubles) -> WChannels {
    let same = |c: &Array4<f64>, k: &Array4<f64>| {
        let n = c.shape()[0];
        if n < 2 {
            Array2::zeros((n, n))
        } else {
            symmetrized_contraction(c, k, 0.125)
        }
    };
    WChannels {
        ss: [same(&c.aa, &k.aa), same(&c.bb, &k.bb)],
        os: [
            symmetrized_contraction(&c.ab, &k.ab, 0.25),
            symmetrized_contraction(&beta_first(&c.ab), &beta_first(&k.ab), 0.25),
        ],
    }
}

/// `W^(2)` and `W^(3)` of a [`crate::u_mp3::u_mp3`] result.
pub fn w_matrices(r: &UMp3Result) -> WMatrices {
    WMatrices {
        w2: w_channels(&r.c1, &r.ints),
        w3: w_channels(&r.c2, &r.ints),
    }
}

/// Indices of the rows of `a` that are not identically zero.
fn nonzero_rows(a: &Array2<f64>) -> Vec<usize> {
    (0..a.nrows())
        .filter(|&i| a.row(i).iter().any(|&v| v != 0.0))
        .collect()
}

fn submatrix(m: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((idx.len(), idx.len()), |(i, j)| m[[idx[i], idx[j]]])
}

/// Relative size below which an eigenvalue of `A − B` counts as zero.
const SINGULAR_REL: f64 = 1e-12;

/// The matrix Padé `P(A, B) = A (A − B)⁻¹ A` of symmetric `A`, `B` (paper
/// Eq. 13), with `A = W^(2)` and `B = x W^(3)`.
///
/// Rows and columns where `A` is identically zero (a channel without pairs)
/// are left out of the solve and give zero rows of `P` (`P(0, 0) = 0`, the
/// paper's convention). `A − B` is diagonalized; an eigenvalue below
/// `1e-12 · max|λ|` is a singular denominator and an error.
pub fn pade(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let n = a.nrows();
    if a.dim() != (n, n) || b.dim() != (n, n) {
        return Err(FerricError::General(format!(
            "pade: A is {:?} and B is {:?}; both must be the same square shape",
            a.dim(),
            b.dim()
        )));
    }
    let keep = nonzero_rows(a);
    let mut out = Array2::zeros((n, n));
    if keep.is_empty() {
        return Ok(out);
    }
    let ak = submatrix(a, &keep);
    let d = &ak - &submatrix(b, &keep);
    let d = (&d + &d.t()) * 0.5;
    let (lam, v) = d
        .eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("pade: eigh of A - B failed: {e}")))?;
    let scale = lam.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    if let Some(&l) = lam.iter().find(|&&l| l.abs() <= SINGULAR_REL * scale) {
        return Err(FerricError::General(format!(
            "pade: A - B is singular (eigenvalue {l:.3e} against max {scale:.3e}); the \
             resummed energy is undefined"
        )));
    }
    let av = ak.dot(&v);
    let mut scaled = av.clone();
    for (mut col, &l) in scaled.columns_mut().into_iter().zip(lam.iter()) {
        col /= l;
    }
    let pk = scaled.dot(&av.t());
    for (r, &i) in keep.iter().enumerate() {
        for (c, &j) in keep.iter().enumerate() {
            out[[i, j]] = pk[[r, c]];
        }
    }
    Ok(out)
}

/// `Tr P(A, B)`.
pub fn pade_trace(a: &Array2<f64>, b: &Array2<f64>) -> Result<f64, FerricError> {
    Ok(pade(a, b)?.diag().sum())
}

/// OP-MP3(x), Eqs. 14–15: `Σ_σ Tr P(W^(2)_σ, x W^(3)_σ)` on the full spin
/// blocks (`x = 1` is OP-MP3).
pub fn op_mp3(w: &WMatrices, x: f64) -> Result<f64, FerricError> {
    (0..2).try_fold(0.0, |e, s| {
        Ok(e + pade_trace(&w.w2.full(s), &(w.w3.full(s) * x))?)
    })
}

/// SOP-MP3(x_ss, x_os), Eqs. 19–21: the SS and OS channels of each spin
/// block resummed separately.
pub fn sop_mp3(w: &WMatrices, x_ss: f64, x_os: f64) -> Result<f64, FerricError> {
    (0..2).try_fold(0.0, |e, s| {
        let ss = pade_trace(&w.w2.ss[s], &(&w.w3.ss[s] * x_ss))?;
        let os = pade_trace(&w.w2.os[s], &(&w.w3.os[s] * x_os))?;
        Ok(e + ss + os)
    })
}

/// SSD-OPMP3(x_ss), Eq. 22: SOP-MP3 with the opposite-spin channel unscaled.
pub fn ssd_opmp3(w: &WMatrices, x_ss: f64) -> Result<f64, FerricError> {
    sop_mp3(w, x_ss, 1.0)
}

/// The scalar (energy-level) Padé of Eq. 1, `E2² / (E2 − E3)` (the paper's
/// P-MP3). It is not size-consistent; it is here as the negative control.
/// A system without correlation (`E2 = 0`) gives 0.
pub fn scalar_pade(w: &WMatrices) -> f64 {
    let (e2, e3) = (w.w2.trace(), w.w3.trace());
    if e2 == 0.0 {
        0.0
    } else {
        e2 * e2 / (e2 - e3)
    }
}

/// Correlation energies of every model (Hartree), without `E_HF` or `E_NBS`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Opmp3Models {
    /// MP2 doubles, `Tr W^(2)`.
    pub mp2: f64,
    /// MP3, `Tr W^(2) + Tr W^(3)`.
    pub mp3: f64,
    /// Scalar Padé (Eq. 1, P-MP3).
    pub p_mp3: f64,
    /// OP-MP3 (Eq. 14).
    pub op_mp3: f64,
    /// SOP-MP3 at `x_ss = x_os = 1` (Eq. 21).
    pub sop_mp3: f64,
    /// SSD-OPMP3 at [`Self::x_ss`] (Eq. 22).
    pub ssd_opmp3: f64,
    /// The same-spin damping used for [`Self::ssd_opmp3`].
    pub x_ss: f64,
}

impl Opmp3Models {
    /// Every model of `w`, SSD-OPMP3 at `x_ss`.
    pub fn evaluate(w: &WMatrices, x_ss: f64) -> Result<Self, FerricError> {
        let (e2, e3) = (w.w2.trace(), w.w3.trace());
        Ok(Opmp3Models {
            mp2: e2,
            mp3: e2 + e3,
            p_mp3: scalar_pade(w),
            op_mp3: op_mp3(w, 1.0)?,
            sop_mp3: sop_mp3(w, 1.0, 1.0)?,
            ssd_opmp3: ssd_opmp3(w, x_ss)?,
            x_ss,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Exactness anchor: with `B = 0` the Padé is `A` itself, and with
    /// commuting (here diagonal) `A`, `B` each eigenvalue gets the scalar
    /// Padé `a²/(a − b)`.
    #[test]
    fn pade_is_a_at_b_zero_and_scalar_on_commuting_blocks() {
        let a = array![[-0.3, 0.05], [0.05, -0.2]];
        let p = pade(&a, &Array2::zeros((2, 2))).unwrap();
        let d = (&p - &a).iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(d < 1e-15, "P(A, 0) - A = {d:.2e}");
        let (ad, bd) = (
            array![[-0.3, 0.0], [0.0, -0.2]],
            array![[-0.05, 0.0], [0.0, 0.02]],
        );
        let want = 0.09 / (-0.3 + 0.05) + 0.04 / (-0.2 - 0.02);
        let got = pade_trace(&ad, &bd).unwrap();
        assert!((got - want).abs() < 1e-15, "{got} vs {want}");
    }

    /// A zero channel contributes zero, and only its rows are dropped.
    #[test]
    fn zero_rows_are_dropped_and_contribute_zero() {
        let z = Array2::<f64>::zeros((1, 1));
        assert_eq!(pade_trace(&z, &z).unwrap(), 0.0);
        let a = array![[-0.3, 0.0], [0.0, 0.0]];
        let b = array![[-0.05, 0.0], [0.0, 0.0]];
        let got = pade_trace(&a, &b).unwrap();
        assert!((got - 0.09 / (-0.25)).abs() < 1e-15, "{got}");
    }

    #[test]
    fn a_singular_denominator_is_an_error() {
        let a = array![[-0.3]];
        assert!(pade(&a, &a).is_err());
    }

    #[test]
    fn paper_x_ss_is_case_insensitive_and_strict() {
        assert_eq!(paper_x_ss("PBE0"), Some(0.5));
        assert_eq!(paper_x_ss("hf"), Some(0.0));
        assert_eq!(paper_x_ss("b3lyp"), None);
    }
}
