//! Matrix-free products of the singlet response matrices on a block of
//! vectors, for the Davidson solvers.
//!
//! With `J(x)_ia = Σ_jb (ia|jb) x_jb`, `K_A(x)_ia = Σ_jb (ij|ab) x_jb`,
//! `K_B(x)_ia = Σ_jb (ib|aj) x_jb` (all RI, Coulomb metric) and the f_xc
//! block `F` ([`ferric_dft::lr_kernel::SingletOvFxc`]):
//!
//! ```text
//!   A x     = Δ x + 2 J(x) − c_HF K_A(x) + F x
//!   (A+B) x = Δ x + 4 J(x) − c_HF (K_A + K_B)(x) + 2 F x
//!   (A−B) x = Δ x − c_HF (K_A − K_B)(x)
//! ```
//!
//! with `Δ_ia = ε_a − ε_i` — term for term the dense `A` and `B` of
//! [`crate::tddft`]. `A − B` needs neither J nor F.
//!
//! # The exchange contractions
//!
//! Per auxiliary function P, with X in `[j][n][b]` order (`(nocc, nvec·nvir)`):
//!
//! * `K_A`: `T = X_{(jn),b} B^P_vvᵀ` (a `(nocc·nvec, nvir)` GEMM, read back as
//!   `[j][n][a]`), then `out += B^P_oo T`.
//! * `K_B`: `M_n = X_nᵀ B^P_ov` (one GEMM for all n, `[n][b][a]`), permuted to
//!   `[b][n][a]`, then `out += B^P_ov M`.
//!
//! The P range is split into contiguous groups fixed by `naux` alone and the
//! group partials are summed in order, so a product does not depend on the
//! rayon thread count.

use ferric_core::timing::PhaseAccumulator;
use ferric_core::FerricError;
use ferric_dft::lr_kernel::SingletOvFxc;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2};
use rayon::prelude::*;

/// Most partial sums the exchange reduction keeps (see the module docs).
const EXCHANGE_GROUPS: usize = 32;

/// The response operator of one closed-shell reference.
pub(crate) struct ResponseOperator<'a> {
    /// `ε_a − ε_i`, `ia = i·nvir + a`.
    pub(crate) diag: Array1<f64>,
    pub(crate) b_ov: &'a Array2<f64>,
    pub(crate) b_oo: &'a Array2<f64>,
    pub(crate) b_vv: &'a Array2<f64>,
    pub(crate) nocc: usize,
    pub(crate) nvir: usize,
    pub(crate) c_hf: f64,
    /// `None` for an HF reference.
    pub(crate) fxc: Option<SingletOvFxc>,
}

/// Which exchange contractions to form.
#[derive(Clone, Copy)]
struct Want {
    a: bool,
    b: bool,
}

impl ResponseOperator<'_> {
    fn delta(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut y = x.clone();
        for (mut row, &d) in y.rows_mut().into_iter().zip(&self.diag) {
            row *= d;
        }
        y
    }

    fn coulomb(&self, x: &Array2<f64>) -> Array2<f64> {
        let w = self.b_ov.dot(x);
        self.b_ov.t().dot(&w)
    }

    /// `(K_A x, K_B x)`; a contraction not wanted comes back as zeros.
    fn exchange(&self, x: &Array2<f64>, want: Want) -> (Array2<f64>, Array2<f64>) {
        let (nocc, nvir) = (self.nocc, self.nvir);
        let nvec = x.ncols();
        let naux = self.b_ov.nrows();
        let width = nvec * nvir;
        let xj = Array2::from_shape_fn((nocc, width), |(j, nb)| {
            x[(j * nvir + nb % nvir, nb / nvir)]
        });
        let ngroups = naux.clamp(1, EXCHANGE_GROUPS);
        let parts: Vec<(Array2<f64>, Array2<f64>)> = (0..ngroups)
            .into_par_iter()
            .map(|g| {
                let mut oa = Array2::<f64>::zeros((nocc, if want.a { width } else { 0 }));
                let mut ob = Array2::<f64>::zeros((nocc, if want.b { width } else { 0 }));
                for p in g * naux / ngroups..(g + 1) * naux / ngroups {
                    if want.a {
                        self.exchange_a_aux(p, &xj, nvec, &mut oa);
                    }
                    if want.b {
                        self.exchange_b_aux(p, &xj, nvec, &mut ob);
                    }
                }
                (oa, ob)
            })
            .collect();
        let mut ka = Array2::<f64>::zeros((nocc, width));
        let mut kb = Array2::<f64>::zeros((nocc, width));
        for (oa, ob) in &parts {
            if want.a {
                ka += oa;
            }
            if want.b {
                kb += ob;
            }
        }
        (self.unpack_inb(&ka, nvec), self.unpack_inb(&kb, nvec))
    }

    /// `[i][n][a]` (`(nocc, nvec·nvir)`) to `(nocc·nvir, nvec)`.
    fn unpack_inb(&self, m: &Array2<f64>, nvec: usize) -> Array2<f64> {
        let nvir = self.nvir;
        Array2::from_shape_fn((self.nocc * nvir, nvec), |(ia, n)| {
            m[(ia / nvir, n * nvir + ia % nvir)]
        })
    }

    fn exchange_a_aux(&self, p: usize, xj: &Array2<f64>, nvec: usize, out: &mut Array2<f64>) {
        let (nocc, nvir) = (self.nocc, self.nvir);
        let xr = xj
            .view()
            .into_shape_with_order((nocc * nvec, nvir))
            .expect("standard layout");
        let bvv = self
            .b_vv
            .row(p)
            .into_shape_with_order((nvir, nvir))
            .expect("contiguous row");
        let boo = self
            .b_oo
            .row(p)
            .into_shape_with_order((nocc, nocc))
            .expect("contiguous row");
        let t = xr.dot(&bvv.t());
        let t = t
            .into_shape_with_order((nocc, nvec * nvir))
            .expect("standard layout");
        general_mat_mul(1.0, &boo, &t, 1.0, out);
    }

    fn exchange_b_aux(&self, p: usize, xj: &Array2<f64>, nvec: usize, out: &mut Array2<f64>) {
        let (nocc, nvir) = (self.nocc, self.nvir);
        let bp = self
            .b_ov
            .row(p)
            .into_shape_with_order((nocc, nvir))
            .expect("contiguous row");
        // M[(n·nvir + b), a] = Σ_j X_n[j, b] B^P[j, a].
        let m = xj.t().dot(&bp);
        let mp = Array2::from_shape_fn((nvir, nvec * nvir), |(b, na)| {
            m[((na / nvir) * nvir + b, na % nvir)]
        });
        general_mat_mul(1.0, &bp, &mp, 1.0, out);
    }

    fn fxc(&self, x: &Array2<f64>) -> Option<Array2<f64>> {
        self.fxc.as_ref().map(|f| f.apply(x))
    }

    /// `A x`.
    pub(crate) fn apply_a(
        &self,
        x: &Array2<f64>,
        acc: &mut PhaseAccumulator,
    ) -> Result<Array2<f64>, FerricError> {
        let mut y = self.delta(x);
        y.scaled_add(2.0, &acc.time("product: RI Coulomb", || self.coulomb(x)));
        if self.c_hf != 0.0 {
            let want = Want { a: true, b: false };
            let (ka, _) = acc.time("product: RI exchange", || self.exchange(x, want));
            y.scaled_add(-self.c_hf, &ka);
        }
        if let Some(f) = acc.time("product: f_xc", || self.fxc(x)) {
            y += &f;
        }
        Ok(y)
    }

    /// `((A+B) x, (A−B) x)`.
    pub(crate) fn apply_pm(
        &self,
        x: &Array2<f64>,
        acc: &mut PhaseAccumulator,
    ) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
        let d = self.delta(x);
        let mut p = d.clone();
        let mut m = d;
        p.scaled_add(4.0, &acc.time("product: RI Coulomb", || self.coulomb(x)));
        if self.c_hf != 0.0 {
            let want = Want { a: true, b: true };
            let (ka, kb) = acc.time("product: RI exchange", || self.exchange(x, want));
            p.scaled_add(-self.c_hf, &(&ka + &kb));
            m.scaled_add(-self.c_hf, &(&ka - &kb));
        }
        if let Some(f) = acc.time("product: f_xc", || self.fxc(x)) {
            p.scaled_add(2.0, &f);
        }
        Ok((p, m))
    }
}
