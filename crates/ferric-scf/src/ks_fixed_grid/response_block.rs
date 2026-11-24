//! The response kernel applied to a block of first-order densities at once.
//!
//! The coupled-perturbed Kohn-Sham solve needs `V1_xc[D1_x]` for every nuclear
//! coordinate `x` of a lockstep block at each iteration. Per point chunk of the
//! resident AO table this module
//!
//! 1. puts every density of the block on the points with ONE product
//!    `[D1_1; …; D1_n] χ` against the table ([`TableChunk::densities`]),
//! 2. lets the caller apply its kernel pointwise to each density's `(ρ1, ∇ρ1)`
//!    (closed shell: [`super::XcResponse`], both spins:
//!    [`super::PolarizedXcResponse`]),
//! 3. contracts the weights back to one `nbf × nbf` matrix per density, again
//!    with one stacked product ([`TableChunk::potentials`]).
//!
//! Each density's matrix is the sum of its chunk contributions added in chunk
//! order, as the one-density application adds them, so a block differs from
//! one-density applications only by the rounding of the wider products.
//!
//! Memory: a chunk holds, per density, one `nbf × chunk` plane and two
//! `nbf × nbf` matrices, and at most [`MAX_BATCH_BAND`] chunks are in flight
//! ([`fold_batches`]). [`response_block_width`] caps the densities per pass so
//! that this live set stays within [`RESPONSE_BLOCK_BYTES`]; it reads `nbf`
//! only, never the thread count.

use super::{fold_batches, response_chunks, MAX_BATCH_BAND, RESPONSE_CHUNK};
use ndarray::{s, Array1, Array2, Array3, ArrayView2, ArrayView3, Axis, Zip};

/// Ceiling (bytes) of the chunk working set of one lockstep pass of the
/// response kernel: every in-flight chunk's density planes and matrices.
pub(super) const RESPONSE_BLOCK_BYTES: usize = 512 << 20;

/// Bytes one point chunk holds per density: the `nbf × RESPONSE_CHUNK` plane
/// (`D1 χ`, then the weighted AOs in the same buffer) and two `nbf × nbf`
/// matrices (the product `M` and the chunk's potential `M + Mᵀ`).
fn per_density_chunk_bytes(nbf: usize) -> usize {
    let nbf = nbf.max(1);
    (nbf * RESPONSE_CHUNK + 2 * nbf * nbf) * std::mem::size_of::<f64>()
}

/// Densities per lockstep pass for `nbf` basis functions, in whole groups of
/// `group` densities (2 for the two spins of one coordinate), at least one
/// group: [`MAX_BATCH_BAND`] chunks of [`per_density_chunk_bytes`] each within
/// [`RESPONSE_BLOCK_BYTES`]. A pure function of `nbf` and `group`.
pub(super) fn response_block_width(nbf: usize, group: usize) -> usize {
    let group = group.max(1);
    let per_group = MAX_BATCH_BAND * group * per_density_chunk_bytes(nbf);
    (RESPONSE_BLOCK_BYTES / per_group).max(1) * group
}

/// One point chunk of the resident AO table.
#[derive(Clone, Copy)]
pub(super) struct TableChunk<'t> {
    /// `χ`, `(nbf, nc)`.
    pub chi: ArrayView2<'t, f64>,
    /// `∂γχ`, `(3, nbf, nc)` (GGA only).
    pub dchi: Option<ArrayView3<'t, f64>>,
}

/// First-order densities of a block on the points of one chunk.
pub(super) struct PointDensities {
    /// `ρ1_i`, `(n, nc)`.
    pub rho1: Array2<f64>,
    /// `∇ρ1_i`, `(n, 3, nc)` (GGA only).
    pub g1: Option<Array3<f64>>,
}

impl PointDensities {
    /// `ρ1` of density `i`.
    pub fn rho1(&self, i: usize) -> Array1<f64> {
        self.rho1.row(i).to_owned()
    }

    /// `∇ρ1` of density `i`, `(3, nc)`.
    pub fn g1(&self, i: usize) -> Option<Array2<f64>> {
        self.g1
            .as_ref()
            .map(|g| g.index_axis(Axis(0), i).to_owned())
    }
}

/// The pointwise weights `(F, G)` of one density's potential
/// `V[F, G]_μν = Σ_g F χ_μ χ_ν + G_γ (∂γχ_μ χ_ν + χ_μ ∂γχ_ν)`.
pub(super) type PotentialWeights = (Array1<f64>, Option<Array2<f64>>);

impl TableChunk<'_> {
    fn nbf(&self) -> usize {
        self.chi.nrows()
    }

    /// `ρ1_i = Σ_μ χ_μ (D1_i χ)_μ` and `∇ρ1_i = 2 Σ_μ ∂γχ_μ (D1_i χ)_μ` of the
    /// stacked densities `[D1_1; …; D1_n]` (`(n·nbf, nbf)`), from one product.
    pub fn densities(&self, stacked: &Array2<f64>) -> PointDensities {
        let nbf = self.nbf();
        let n = stacked.nrows() / nbf;
        let pd = stacked.dot(&self.chi);
        let nc = self.chi.ncols();
        let mut rho1 = Array2::<f64>::zeros((n, nc));
        let mut g1 = self.dchi.map(|_| Array3::<f64>::zeros((n, 3, nc)));
        for i in 0..n {
            let pdi = pd.slice(s![i * nbf..(i + 1) * nbf, ..]);
            column_dots(self.chi, pdi, rho1.row_mut(i), 1.0);
            if let (Some(dchi), Some(g1)) = (self.dchi, g1.as_mut()) {
                for gam in 0..3 {
                    let plane = dchi.index_axis(Axis(0), gam);
                    column_dots(plane, pdi, g1.slice_mut(s![i, gam, ..]), 2.0);
                }
            }
        }
        PointDensities { rho1, g1 }
    }

    /// `V[F_i, G_i]` of every density, as `M_i + M_iᵀ` with
    /// `M_i = (½ F_i χ + Σ_γ G_iγ ∂γχ) χᵀ`: the weighted AOs of all densities
    /// stacked into ONE product with `χᵀ`. `G` is read only on a GGA table.
    pub fn potentials(&self, weights: &[PotentialWeights]) -> Vec<Array2<f64>> {
        let nbf = self.nbf();
        let mut a = Array2::<f64>::zeros((weights.len() * nbf, self.chi.ncols()));
        for (i, (fw, gw)) in weights.iter().enumerate() {
            let mut ai = a.slice_mut(s![i * nbf..(i + 1) * nbf, ..]);
            Zip::from(&mut ai)
                .and(&self.chi)
                .and_broadcast(&fw.view().insert_axis(Axis(0)))
                .for_each(|a, &c, &f| *a = 0.5 * f * c);
            if let (Some(gw), Some(dchi)) = (gw, self.dchi) {
                for gam in 0..3 {
                    Zip::from(&mut ai)
                        .and(&dchi.index_axis(Axis(0), gam))
                        .and_broadcast(&gw.row(gam).insert_axis(Axis(0)))
                        .for_each(|a, &d, &g| *a += g * d);
                }
            }
        }
        let m = a.dot(&self.chi.t());
        (0..weights.len())
            .map(|i| {
                let mi = m.slice(s![i * nbf..(i + 1) * nbf, ..]);
                &mi + &mi.t()
            })
            .collect()
    }
}

/// `out_g += Σ_μ scale · a_μg b_μg` (a column-wise dot product), accumulated
/// over `μ` in ascending order.
pub(super) fn column_dots(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    mut out: ndarray::ArrayViewMut1<'_, f64>,
    scale: f64,
) {
    for (ar, br) in a.outer_iter().zip(b.outer_iter()) {
        Zip::from(&mut out)
            .and(&ar)
            .and(&br)
            .for_each(|o, &x, &y| *o += scale * x * y);
    }
}

/// `[D_1; …; D_n]`, `(n·nbf, nbf)`.
pub(super) fn stack_densities(ds: &[&Array2<f64>]) -> Array2<f64> {
    let nbf = ds.first().map_or(0, |d| d.nrows());
    let mut out = Array2::<f64>::zeros((ds.len() * nbf, nbf));
    for (i, d) in ds.iter().enumerate() {
        out.slice_mut(s![i * nbf..(i + 1) * nbf, ..]).assign(*d);
    }
    out
}

/// The response table the kernel keeps: `χ` and, for a GGA, `∂γχ` at every
/// grid point, filled one point batch at a time.
pub(super) struct AoTable {
    chi: Array2<f64>,
    dchi: Option<Array3<f64>>,
}

impl AoTable {
    /// An empty table of `nbf` AOs on `npts` points (gradient planes iff `gga`).
    pub fn zeros(nbf: usize, npts: usize, gga: bool) -> Self {
        Self {
            chi: Array2::zeros((nbf, npts)),
            dchi: gga.then(|| Array3::zeros((3, nbf, npts))),
        }
    }

    /// Store a batch's AO values (and gradients) from point `g0` on.
    pub fn store(&mut self, g0: usize, chi: &Array2<f64>, dchi: Option<&Array3<f64>>) {
        let g1 = g0 + chi.ncols();
        self.chi.slice_mut(s![.., g0..g1]).assign(chi);
        if let (Some(dst), Some(src)) = (self.dchi.as_mut(), dchi) {
            dst.slice_mut(s![.., .., g0..g1]).assign(src);
        }
    }

    /// The table as the borrowed view [`ResidentTable::apply`] reads.
    pub fn resident(&self) -> ResidentTable<'_> {
        ResidentTable {
            chi: &self.chi,
            dchi: self.dchi.as_ref(),
        }
    }
}

/// The resident response table: `χ` and, for a GGA, `∂γχ` at every point.
pub(super) struct ResidentTable<'t> {
    pub chi: &'t Array2<f64>,
    pub dchi: Option<&'t Array3<f64>>,
}

impl<'t> ResidentTable<'t> {
    fn chunk(&self, c0: usize, c1: usize) -> TableChunk<'t> {
        TableChunk {
            chi: self.chi.slice(s![.., c0..c1]),
            dchi: self.dchi.map(|d| d.slice(s![.., .., c0..c1])),
        }
    }

    /// The response matrices of the densities `ds` (whole groups of `group`):
    /// passes of at most [`response_block_width`] densities, each over the
    /// point chunks in parallel with every chunk's contributions added in
    /// chunk order. `weights(c0, c1, pd)` maps the chunk's point densities to
    /// one [`PotentialWeights`] per density, in order.
    pub fn apply<W>(&self, ds: &[&Array2<f64>], group: usize, weights: W) -> Vec<Array2<f64>>
    where
        W: Fn(usize, usize, &PointDensities) -> Vec<PotentialWeights> + Sync,
    {
        let nbf = self.chi.nrows();
        let width = response_block_width(nbf, group);
        let chunks = response_chunks(self.chi.ncols());
        let mut out = Vec::with_capacity(ds.len());
        for pass in ds.chunks(width) {
            let stacked = stack_densities(pass);
            let mut v: Vec<Array2<f64>> = pass
                .iter()
                .map(|_| Array2::<f64>::zeros((nbf, nbf)))
                .collect();
            fold_batches(
                &chunks,
                pass.len() * nbf * nbf,
                |&(c0, c1)| {
                    let table = self.chunk(c0, c1);
                    let pd = table.densities(&stacked);
                    Ok(table.potentials(&weights(c0, c1, &pd)))
                },
                |parts| v.iter_mut().zip(parts).for_each(|(v, p)| *v += &p),
            )
            .expect("the response kernel chunks are infallible");
            out.extend(v);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pass width is a function of the basis size alone: the same under
    /// every pool, whole spin pairs, at least one group, and the live set it
    /// admits is within the ceiling whenever more than one group fits.
    #[test]
    fn block_width_is_charged_and_ignores_the_pool() {
        for threads in [1usize, 2, 12] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for nbf in [10usize, 82, 106, 500, 4000] {
                    for group in [1usize, 2] {
                        let w = response_block_width(nbf, group);
                        assert_eq!(w % group, 0);
                        assert!(w >= group);
                        let live = MAX_BATCH_BAND * w * per_density_chunk_bytes(nbf);
                        assert!(w == group || live <= RESPONSE_BLOCK_BYTES, "nbf {nbf}");
                    }
                }
                assert_eq!(response_block_width(82, 1), 28);
                assert_eq!(response_block_width(4000, 2), 2);
            });
        }
    }

    /// One stacked product equals one product per density (the GEMM rows of
    /// a density do not depend on the rows stacked around it, up to rounding).
    #[test]
    fn stacked_densities_and_potentials_match_one_at_a_time() {
        let (nbf, nc) = (7usize, 13usize);
        let f = |i: usize, j: usize, s: f64| ((i * 31 + j * 17) as f64 * s).sin();
        let chi = Array2::from_shape_fn((nbf, nc), |(i, j)| f(i, j, 0.37));
        let dchi = Array3::from_shape_fn((3, nbf, nc), |(g, i, j)| f(i + g, j, 0.11));
        let sym = |s: f64| {
            let m = Array2::from_shape_fn((nbf, nbf), |(i, j)| f(i, j, s));
            &m + &m.t()
        };
        let ds = [sym(0.5), sym(0.9), sym(1.3)];
        let refs: Vec<&Array2<f64>> = ds.iter().collect();
        let table = TableChunk {
            chi: chi.view(),
            dchi: Some(dchi.view()),
        };
        let all = table.densities(&stack_densities(&refs));
        let ws: Vec<PotentialWeights> = (0..3).map(|i| (all.rho1(i), all.g1(i))).collect();
        let v_all = table.potentials(&ws);
        for i in 0..3 {
            let one = table.densities(&stack_densities(&[refs[i]]));
            let d_rho = (&one.rho1(0) - &all.rho1(i)).mapv(f64::abs).sum();
            let d_g = (&one.g1(0).unwrap() - &all.g1(i).unwrap())
                .mapv(f64::abs)
                .sum();
            let v_one = table.potentials(&ws[i..i + 1]);
            let d_v = (&v_one[0] - &v_all[i]).mapv(f64::abs).sum();
            assert!(
                d_rho + d_g + d_v <= 1e-12,
                "density {i}: {d_rho} {d_g} {d_v}"
            );
        }
    }
}
