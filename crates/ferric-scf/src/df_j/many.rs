//! Coulomb matrices of many densities from one pass over the RI-J tensor.
//!
//! The coupled-perturbed solves of the Hessian need `J[D1_x]` for every
//! coordinate `x` of a lockstep block at each iteration. [`DfJ::build`] streams
//! the whole `(P|μν)` tensor twice per density (two GEMVs); [`DfJ::build_many`]
//! streams it twice per BLOCK, with the densities as the columns of one GEMM:
//!
//! - pass 1: `d_P = Σ_k B_Pk W_k` for the weight columns `W` of all densities,
//! - the metric solve `c = V⁻¹ d` per column (the same Cholesky solve),
//! - pass 2: `J_k = Σ_P B_Pk c_P` for all densities, through the ordered column
//!   fold, so the sum order of every element is fixed by the chunk edges.
//!
//! Every f64 tier takes this path: in-core (full `nbf²` columns), packed,
//! spilled or recomputed (the packed lower triangle, streamed by block) and
//! pair-screened (the kept pairs). The f32 and device tiers build one density
//! at a time, as does a single density, so a one-density call is
//! [`DfJ::build`] exactly.

use super::{chunk_width, packed_density_weights, pair_density_weights, scatter_pair_cols};
use super::{unpack_symmetric, DfJ};
use crate::fock::JBuilder;
use ferric_core::FerricError;
use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2};
use rayon::prelude::*;

/// Densities per lockstep pass for `cols` stored tensor columns and `naux`
/// auxiliary functions under `band_bytes`: each density holds its weight
/// column, its accumulator row and [`crate::reduce::MAX_FOLD_PARTIALS`] pass-2
/// partial rows (`cols` each) plus `d_P` and `c_P` (`naux` each). At least one.
/// A pure function of the sizes and the budget, never of the thread count.
pub(super) fn lockstep_width(cols: usize, naux: usize, band_bytes: usize) -> usize {
    let per_density =
        ((2 + crate::reduce::MAX_FOLD_PARTIALS) * cols + 2 * naux) * std::mem::size_of::<f64>();
    (band_bytes / per_density.max(1)).max(1)
}

/// Pass-2 partials held at once for `n` densities of `cols` columns.
fn lockstep_band(n: usize, cols: usize, band_bytes: usize) -> usize {
    let per_partial = (n * cols).max(1) * std::mem::size_of::<f64>();
    (band_bytes / per_partial).clamp(1, crate::reduce::MAX_FOLD_PARTIALS)
}

impl DfJ<'_> {
    /// The Coulomb matrices of every density of `ds`, in order: the in-core f64
    /// tensor in lockstep passes of at most `lockstep_width` densities, any
    /// other tier (or a single density) one [`DfJ::build`] per density.
    pub fn build_many(&mut self, ds: &[&Array2<f64>]) -> Result<Vec<Array2<f64>>, FerricError> {
        if ds.len() < 2 || !self.lockstep_supported() {
            return ds.iter().map(|d| self.build_one(d)).collect();
        }
        let band_bytes = crate::reduce::resolve_band_bytes(self.budget_bytes);
        let width = lockstep_width(self.stored_cols(), self.source.naux(), band_bytes);
        let mut out = Vec::with_capacity(ds.len());
        for pass in ds.chunks(width) {
            out.extend(self.build_lockstep(pass, band_bytes)?);
        }
        Ok(out)
    }

    fn build_one(&mut self, d: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        let mut j = Array2::<f64>::zeros(d.dim());
        self.build(d, &mut j)?;
        Ok(j)
    }

    /// Whether the tensor is an f64 tier the lockstep pass reads (the device
    /// path, when compiled in, keeps its one-density builds).
    pub(crate) fn lockstep_supported(&self) -> bool {
        if cfg!(feature = "gpu") || self.source.is_ri_j_f32() {
            return false;
        }
        self.source.is_incore()
            || self.source.pair_screened().is_some()
            || self.source.supports_packed_stream()
    }

    /// Stored columns per auxiliary row: all `n²` in core, the kept pairs, or
    /// the packed `n(n+1)/2`.
    fn stored_cols(&self) -> usize {
        let n = self.source.nao();
        if self.source.is_incore() {
            return n * n;
        }
        match self.source.pair_screened() {
            Some((data, _)) => data.ncols(),
            None => n * (n + 1) / 2,
        }
    }

    /// The weight columns `(cols, n)` of the densities for the stored columns.
    fn lockstep_weights(&self, ds: &[&Array2<f64>]) -> Array2<f64> {
        let mut w = Array2::<f64>::zeros((self.stored_cols(), ds.len()));
        for (i, d) in ds.iter().enumerate() {
            if self.source.is_incore() {
                let flat: ndarray::Array1<f64> = d.iter().copied().collect();
                w.column_mut(i).assign(&flat);
                continue;
            }
            let col = match self.source.pair_screened() {
                Some((_, screen)) => pair_density_weights(d, screen.cols()),
                None => packed_density_weights(d, self.source.nao()),
            };
            w.column_mut(i).assign(&col);
        }
        w
    }

    /// Write one density's accumulator row into its full symmetric `j`.
    fn lockstep_scatter(&self, acc: &[f64], j: &mut Array2<f64>) {
        if self.source.is_incore() {
            j.iter_mut().zip(acc).for_each(|(x, a)| *x = *a);
            return;
        }
        match self.source.pair_screened() {
            Some((_, screen)) => scatter_pair_cols(acc, screen.cols(), j),
            None => unpack_symmetric(acc, self.source.nao(), j),
        }
    }

    /// Run `f(p0, rows)` over the stored tensor's blocks of auxiliary rows.
    fn for_each_lockstep_block(
        &mut self,
        chunk: usize,
        mut f: impl FnMut(usize, ArrayView2<'_, f64>) -> Result<(), FerricError>,
    ) -> Result<(), FerricError> {
        if let Some(data) = self.source.incore_flat() {
            return f(self.source.band().0, data);
        }
        if let Some((data, _)) = self.source.pair_screened() {
            return f(self.source.band().0, data);
        }
        self.source
            .for_each_packed_block(chunk, |blk| f(blk.p0, blk.data))
    }

    /// One lockstep pass over the densities `ds`.
    fn build_lockstep(
        &mut self,
        ds: &[&Array2<f64>],
        band_bytes: usize,
    ) -> Result<Vec<Array2<f64>>, FerricError> {
        let w = self.lockstep_weights(ds);
        let d_p = self.lockstep_pass1(&w)?;
        let c_p = self.lockstep_metric_solve(d_p)?;
        let cols = w.nrows();
        let acc = self.lockstep_pass2(&c_p, cols, band_bytes)?;
        let nbf = self.source.nao();
        acc.chunks(cols.max(1))
            .take(ds.len())
            .map(|row| {
                let mut j = Array2::<f64>::zeros((nbf, nbf));
                self.lockstep_scatter(row, &mut j);
                self.reduce_j(&mut j);
                Ok(j)
            })
            .collect()
    }

    /// Pass 1: `d_P = Σ_k B_Pk W_k`, `(naux, n)`, one GEMM per fixed aux chunk
    /// (disjoint row slices, so no reduction and no thread dependence).
    fn lockstep_pass1(&mut self, w: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        let chunk = chunk_width(self.source.nao());
        let mut d_p = Array2::<f64>::zeros((self.source.naux(), w.ncols()));
        self.for_each_lockstep_block(chunk, |p0, data| {
            let b = data.nrows();
            let parts: Vec<Array2<f64>> = (0..b.div_ceil(chunk))
                .into_par_iter()
                .map(|ci| {
                    data.slice(s![ci * chunk..((ci + 1) * chunk).min(b), ..])
                        .dot(w)
                })
                .collect();
            for (ci, part) in parts.iter().enumerate() {
                let q0 = p0 + ci * chunk;
                d_p.slice_mut(s![q0..q0 + part.nrows(), ..]).assign(part);
            }
            Ok(())
        })?;
        Ok(d_p)
    }

    /// `c = V⁻¹ d_P` per density column, after the cross-rank `d_P` sum.
    fn lockstep_metric_solve(&self, mut d_p: Array2<f64>) -> Result<Array2<f64>, FerricError> {
        for mut col in d_p.columns_mut() {
            let mut d = col.to_owned();
            self.reduce_d_p(&mut d);
            col.assign(&self.solve_metric(&d)?);
        }
        Ok(d_p)
    }

    /// Pass 2: `acc_ik = Σ_P c_Pi B_Pk`, row-major `(n, cols)`, through the
    /// ordered column fold over the same fixed aux chunks as pass 1.
    fn lockstep_pass2(
        &mut self,
        c_p: &Array2<f64>,
        cols: usize,
        band_bytes: usize,
    ) -> Result<Vec<f64>, FerricError> {
        let n = c_p.ncols();
        let chunk = chunk_width(self.source.nao());
        let band = lockstep_band(n, cols, band_bytes);
        let mut acc = vec![0.0_f64; n * cols];
        let mut scratch = Vec::new();
        self.for_each_lockstep_block(chunk, |p0, data| {
            let b = data.nrows();
            crate::reduce::grouped_deterministic_fold_columns(
                &mut acc,
                b.div_ceil(chunk),
                band,
                &mut scratch,
                |ci, y| {
                    let (q0, q1) = (ci * chunk, ((ci + 1) * chunk).min(b));
                    let c_sub = c_p.slice(s![p0 + q0..p0 + q1, ..]);
                    let mut y = ArrayViewMut2::from_shape((n, cols), y).expect("n x cols partial");
                    // beta = 0: `y`'s previous contents are not read.
                    general_mat_mul(1.0, &c_sub.t(), &data.slice(s![q0..q1, ..]), 0.0, &mut y);
                },
            );
            Ok(())
        })?;
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_testkit::compare::max_abs;

    /// The pass width reads the sizes and the budget only: the same under
    /// every pool, at least one, and its live set within the budget whenever
    /// more than one density fits.
    #[test]
    fn lockstep_width_is_charged_and_ignores_the_pool() {
        for threads in [1usize, 2, 12] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mib = 1usize << 20;
                assert_eq!(lockstep_width(3403, 600, 512 * mib), 748);
                assert_eq!(lockstep_width(3403, 600, 1), 1);
                let w = lockstep_width(100_000, 5000, 64 * mib);
                let live = w * ((2 + crate::reduce::MAX_FOLD_PARTIALS) * 100_000 + 2 * 5000) * 8;
                assert!(w >= 1 && live <= 64 * mib);
                assert_eq!(
                    lockstep_band(40, 3403, 512 * mib),
                    crate::reduce::MAX_FOLD_PARTIALS
                );
                assert_eq!(lockstep_band(40, 3403, 1), 1);
            });
        }
    }

    /// A symmetric, unstructured density for slot `k`.
    fn density(n: usize, k: usize) -> Array2<f64> {
        let x = Array2::from_shape_fn((n, n), |(i, j)| {
            (((i * 7 + j * 3 + k * 11) % 17) as f64 - 8.0) / 9.0
        });
        &x + &x.t()
    }

    /// `build_many` equals one `build` per density on the in-core and the
    /// packed (or spilled) tier. The routes differ only in summation order:
    /// pass 1 (`cols` terms per `d_P`, whose error the metric solve amplifies
    /// by up to `cond(V)`, the metric's eigenvalue ratio) and pass 2 (`naux`
    /// terms per element). Bar: `(cols + naux)·ε·cond(V)·max|J|`.
    #[test]
    fn many_densities_equal_one_build_each() {
        use ferric_core::basis;
        use ferric_core::mol::Molecule;
        use ferric_integrals::basis_bridge::PreparedBasis;
        use ferric_integrals::operator::Operator;
        use ndarray_linalg::{EighInto, UPLO};
        let mol = Molecule::load_xyz(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/molecules/water.xyz"
        ))
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
        let aux =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let (n, naux) = (obs.nbasis(), aux.nbasis());
        let op = Operator::coulomb();
        let (eigs, _) = ferric_integrals::threeindex::coulomb_metric_2c(op, &aux)
            .unwrap()
            .eigh_into(UPLO::Lower)
            .unwrap();
        let cond = eigs[naux - 1] / eigs[0];
        let ds: Vec<Array2<f64>> = (0..5).map(|k| density(n, k)).collect();
        let refs: Vec<&Array2<f64>> = ds.iter().collect();
        let unpacked = naux * n * n * 8;
        let mut tiers = Vec::new();
        for budget in [1usize << 30, unpacked - 1] {
            let mut df = DfJ::new(op, &obs, &aux, budget).unwrap();
            assert!(df.lockstep_supported());
            tiers.push(df.source.is_incore());
            let cols = df.stored_cols();
            let many = df.build_many(&refs).unwrap();
            for (k, d) in ds.iter().enumerate() {
                let one = df.build_one(d).unwrap();
                let diff = max_abs(&(&many[k] - &one));
                let bar = (cols + naux) as f64 * f64::EPSILON * cond * max_abs(&one);
                eprintln!(
                    "in-core {}: density {k}: {diff:.3e} (bar {bar:.3e})",
                    tiers.last().unwrap()
                );
                assert!(diff <= bar, "density {k}: {diff:.3e} > {bar:.3e}");
            }
        }
        assert_eq!(tiers, [true, false], "both tiers must be reached");
    }
}
