//! The response kernel in the molecular-orbital form the CPKS iterates in.
//!
//! A CPKS trial vector is a rotation `U` (rows: a set of MOs `r`, columns:
//! the occupied `i`) whose AO density is
//! `D1 = occ (C_r U C_oᵀ + C_o Uᵀ C_rᵀ)`. With the MOs on the grid,
//! `φ_p = Σ_μ C_μp χ_μ`, evaluated once per Hessian ([`MoTable`]), its
//! density and gradient on the points are
//!
//! ```text
//!   ρ1  = 2 occ Σ_i φ_i φ̃_i,                     φ̃_i = Σ_r U_ri φ_r
//!   ∇ρ1 = 2 occ Σ_i (∇φ_i φ̃_i + φ_i ∇φ̃_i),       ∇φ̃_i = Σ_r U_ri ∇φ_r
//! ```
//!
//! and the potential `V[F, G]` (the same pointwise weights the AO route
//! computes) projected to MO rows `p` of a chosen set and occupied columns is
//!
//! ```text
//!   (Cᵀ V C_o)_pi = Σ_g φ_p (F φ_i + G·∇φ_i) + Σ_γ Σ_g ∂γφ_p (G_γ φ_i).
//! ```
//!
//! Per point and density that is eight `nocc × n_r` products for a GGA (two
//! for an LDA) where the AO route ([`super::response_block`]) does two
//! `nbf × nbf` ones, and the pointwise work runs over `nocc` rows, not `nbf`.
//! The AO route is the oracle (`mo_product_equals_the_projected_ao_product`).
//!
//! Every density of a block is put on the points of a chunk with one product
//! per plane (`[U_1ᵀ; …; U_nᵀ] Φ_r`), and contracted back with one product per
//! plane; each density's result is the sum of its chunk contributions added
//! in chunk order.
//!
//! Memory: the tables hold `nmo × npts` doubles per plane (1 for an LDA, 4 for
//! a GGA) per spin, the size of the AO table they replace for a closed shell
//! (`nmo ≤ nbf`). A chunk holds, per density, `8 nocc × RESPONSE_CHUNK`
//! doubles of planes and two `nmo × nocc` results; [`mo_block_width`] caps the
//! densities per pass so that [`MAX_BATCH_BAND`] chunks of them stay within
//! [`super::response_block::RESPONSE_BLOCK_BYTES`]. Neither reads the thread
//! count.

use super::response_block::{column_dots, PointDensities, PotentialWeights, RESPONSE_BLOCK_BYTES};
use super::{fold_batches, response_chunks, MAX_BATCH_BAND, RESPONSE_CHUNK};
use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3, Axis, Zip};
use std::ops::Range;

/// Which MOs a rotation's rows, or a result's rows, run over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoRows {
    /// The occupied MOs `0..nocc`.
    Occupied,
    /// The virtual MOs `nocc..nmo`.
    Virtual,
    /// Every MO.
    All,
}

/// The MOs of one spin (or the restricted set) on every grid point.
pub(super) struct MoTable {
    /// `φ_p`, `(nmo, npts)`.
    phi: Array2<f64>,
    /// `∂γφ_p`, `(3, nmo, npts)` (GGA only).
    dphi: Option<Array3<f64>>,
    nocc: usize,
}

impl MoTable {
    /// An empty table of `nmo` MOs (the first `nocc` occupied) on `npts`
    /// points.
    pub fn zeros(nmo: usize, nocc: usize, npts: usize, gga: bool) -> Self {
        MoTable {
            phi: Array2::zeros((nmo, npts)),
            dphi: gga.then(|| Array3::zeros((3, nmo, npts))),
            nocc,
        }
    }

    /// Store `Cᵀ χ` (and `Cᵀ ∂γχ`) of the AO batch at points `g0..g1`.
    pub fn store(
        &mut self,
        c: &Array2<f64>,
        g0: usize,
        chi: &Array2<f64>,
        dchi: Option<&Array3<f64>>,
    ) {
        let g1 = g0 + chi.ncols();
        self.phi.slice_mut(s![.., g0..g1]).assign(&c.t().dot(chi));
        if let (Some(dst), Some(dchi)) = (self.dphi.as_mut(), dchi) {
            for gam in 0..3 {
                let plane = c.t().dot(&dchi.index_axis(Axis(0), gam));
                dst.slice_mut(s![gam, .., g0..g1]).assign(&plane);
            }
        }
    }

    pub fn nocc(&self) -> usize {
        self.nocc
    }

    fn nmo(&self) -> usize {
        self.phi.nrows()
    }

    /// The MO index range of `rows`.
    pub fn range(&self, rows: MoRows) -> Range<usize> {
        match rows {
            MoRows::Occupied => 0..self.nocc,
            MoRows::Virtual => self.nocc..self.nmo(),
            MoRows::All => 0..self.nmo(),
        }
    }

    fn chunk(&self, c0: usize, c1: usize) -> MoChunk<'_> {
        MoChunk {
            phi: self.phi.slice(s![.., c0..c1]),
            dphi: self.dphi.as_ref().map(|d| d.slice(s![.., .., c0..c1])),
        }
    }
}

/// Some MOs of a table on one point chunk.
#[derive(Clone, Copy)]
struct MoChunk<'t> {
    phi: ArrayView2<'t, f64>,
    dphi: Option<ArrayView3<'t, f64>>,
}

impl<'t> MoChunk<'t> {
    fn rows(&self, r: Range<usize>) -> MoChunk<'t> {
        MoChunk {
            phi: self.phi.slice_move(s![r.clone(), ..]),
            dphi: self.dphi.map(|d| d.slice_move(s![.., r, ..])),
        }
    }

    fn plane(&self, gam: usize) -> ArrayView2<'t, f64> {
        self.dphi.expect("GGA table").index_axis_move(Axis(0), gam)
    }
}

/// `[U_1ᵀ; …; U_nᵀ]`, `(n·nocc, n_r)`, from rotations `(n_r, nocc)`.
fn stack_transposed(us: &[&Array2<f64>]) -> Array2<f64> {
    let (nr, nocc) = us.first().map_or((0, 0), |u| u.dim());
    let mut out = Array2::<f64>::zeros((us.len() * nocc, nr));
    for (x, u) in us.iter().enumerate() {
        out.slice_mut(s![x * nocc..(x + 1) * nocc, ..])
            .assign(&u.t());
    }
    out
}

/// The `n` densities on a chunk of the rotations stacked in `ut` (rows over
/// the MOs `input`), each scaled by `scale` = `2 occ`. `n` is passed, not read
/// from `ut`: a spin with no occupied MOs stacks zero rows for every density,
/// and each of its densities is exactly zero.
fn chunk_densities(
    table: MoChunk<'_>,
    nocc: usize,
    ut: &Array2<f64>,
    n: usize,
    input: Range<usize>,
    scale: f64,
) -> PointDensities {
    let (occ, rin) = (table.rows(0..nocc), table.rows(input));
    debug_assert_eq!(ut.nrows(), n * nocc);
    let nc = table.phi.ncols();
    let tilde = ut.dot(&rin.phi);
    let dtilde = rin
        .dphi
        .map(|_| [0, 1, 2].map(|gam| ut.dot(&rin.plane(gam))));
    let mut rho1 = Array2::<f64>::zeros((n, nc));
    let mut g1 = dtilde.as_ref().map(|_| Array3::<f64>::zeros((n, 3, nc)));
    for x in 0..n {
        let rows = s![x * nocc..(x + 1) * nocc, ..];
        let tx = tilde.slice(rows);
        column_dots(occ.phi, tx, rho1.row_mut(x), scale);
        if let (Some(dt), Some(g1)) = (dtilde.as_ref(), g1.as_mut()) {
            for gam in 0..3 {
                let mut out = g1.slice_mut(s![x, gam, ..]);
                column_dots(occ.plane(gam), tx, out.view_mut(), scale);
                column_dots(occ.phi, dt[gam].slice(rows), out, scale);
            }
        }
    }
    PointDensities { rho1, g1 }
}

/// `(Cᵀ V[F_x, G_x] C_o)` on the MO rows `output` for every density's weights,
/// `(n_out, nocc)` each: `L0 = F φ_o + G·∇φ_o` and `L_γ = G_γ φ_o` stacked
/// over the densities, then `L0 Φ_outᵀ + Σ_γ L_γ ∂γΦ_outᵀ`.
fn chunk_potentials(
    table: MoChunk<'_>,
    nocc: usize,
    weights: &[PotentialWeights],
    output: Range<usize>,
) -> Vec<Array2<f64>> {
    let (occ, out) = (table.rows(0..nocc), table.rows(output));
    let n = weights.len();
    let nc = table.phi.ncols();
    let gga = weights.iter().any(|(_, g)| g.is_some()) && table.dphi.is_some();
    let mut l0 = Array2::<f64>::zeros((n * nocc, nc));
    let mut lg = gga.then(|| [0, 1, 2].map(|_| Array2::<f64>::zeros((n * nocc, nc))));
    for (x, (fw, gw)) in weights.iter().enumerate() {
        let rows = s![x * nocc..(x + 1) * nocc, ..];
        let mut l0x = l0.slice_mut(rows);
        Zip::from(&mut l0x)
            .and(&occ.phi)
            .and_broadcast(&fw.view().insert_axis(Axis(0)))
            .for_each(|l, &p, &f| *l = f * p);
        if let (Some(gw), Some(lg)) = (gw, lg.as_mut()) {
            for gam in 0..3 {
                let g = gw.row(gam).insert_axis(Axis(0));
                Zip::from(&mut l0x)
                    .and(&occ.plane(gam))
                    .and_broadcast(&g)
                    .for_each(|l, &d, &g| *l += g * d);
                Zip::from(&mut lg[gam].slice_mut(rows))
                    .and(&occ.phi)
                    .and_broadcast(&g)
                    .for_each(|l, &p, &g| *l = g * p);
            }
        }
    }
    let mut acc = l0.dot(&out.phi.t());
    if let Some(lg) = &lg {
        for (gam, l) in lg.iter().enumerate() {
            ndarray::linalg::general_mat_mul(1.0, l, &out.plane(gam).t(), 1.0, &mut acc);
        }
    }
    (0..n)
        .map(|x| acc.slice(s![x * nocc..(x + 1) * nocc, ..]).t().to_owned())
        .collect()
}

/// Bytes one point chunk holds per density with `nocc` occupied and `nmo`
/// MOs: eight `nocc × RESPONSE_CHUNK` planes (`φ̃`, `∇φ̃`, `L0`, `L_γ`) and two
/// `nmo × nocc` results (the stacked product and the density's matrix).
fn per_density_chunk_bytes(nmo: usize, nocc: usize) -> usize {
    (8 * nocc * RESPONSE_CHUNK + 2 * nmo * nocc).max(1) * std::mem::size_of::<f64>()
}

/// Densities per lockstep pass, in whole groups of `group`, at least one
/// group: [`MAX_BATCH_BAND`] chunks of [`per_density_chunk_bytes`] each within
/// [`RESPONSE_BLOCK_BYTES`]. A pure function of the sizes.
pub(super) fn mo_block_width(nmo: usize, nocc: usize, group: usize) -> usize {
    let group = group.max(1);
    let per_group = MAX_BATCH_BAND * group * per_density_chunk_bytes(nmo, nocc);
    (RESPONSE_BLOCK_BYTES / per_group).max(1) * group
}

/// One MO-form product: rotations of the rows `input`, results on the rows
/// `output`, each density scaled by `2 occ` (`scale`).
pub(super) struct MoProduct<'a> {
    /// One table per spin; density `x` reads `tables[x % tables.len()]`.
    pub tables: &'a [&'a MoTable],
    pub input: MoRows,
    pub output: MoRows,
    pub scale: f64,
}

impl MoProduct<'_> {
    /// The projected responses to every rotation of `us` (spins interleaved
    /// when there are two tables), in passes of at most [`mo_block_width`]
    /// densities over the point chunks in parallel. `weights(c0, c1, pd)` maps
    /// the chunk's point densities to one [`PotentialWeights`] per density.
    pub fn apply<W>(&self, us: &[&Array2<f64>], weights: W) -> Vec<Array2<f64>>
    where
        W: Fn(usize, usize, &PointDensities) -> Vec<PotentialWeights> + Sync,
    {
        let nspin = self.tables.len();
        let nmo = self.tables.iter().map(|t| t.nmo()).max().unwrap_or(0);
        let nocc = self.tables.iter().map(|t| t.nocc()).max().unwrap_or(0);
        let width = mo_block_width(nmo, nocc, nspin);
        let chunks = response_chunks(self.tables.first().map_or(0, |t| t.phi.ncols()));
        let mut out = Vec::with_capacity(us.len());
        for pass in us.chunks(width) {
            let stacked: Vec<Array2<f64>> = (0..nspin)
                .map(|s| stack_transposed(&spin_members(pass, s, nspin)))
                .collect();
            let elems: usize = (0..pass.len())
                .map(|x| self.out_len(x % nspin) * self.tables[x % nspin].nocc())
                .sum();
            let mut v: Vec<Array2<f64>> = (0..pass.len())
                .map(|x| Array2::zeros((self.out_len(x % nspin), self.tables[x % nspin].nocc())))
                .collect();
            fold_batches(
                &chunks,
                elems,
                |&(c0, c1)| Ok(self.chunk(c0, c1, &stacked, pass.len(), &weights)),
                |parts| v.iter_mut().zip(parts).for_each(|(v, p)| *v += &p),
            )
            .expect("the response kernel chunks are infallible");
            out.extend(v);
        }
        out
    }

    fn out_len(&self, spin: usize) -> usize {
        self.tables[spin].range(self.output).len()
    }

    /// One chunk: the densities of every spin, the caller's pointwise kernel,
    /// the projected potentials, all in the pass's density order.
    fn chunk<W>(
        &self,
        c0: usize,
        c1: usize,
        stacked: &[Array2<f64>],
        n: usize,
        weights: &W,
    ) -> Vec<Array2<f64>>
    where
        W: Fn(usize, usize, &PointDensities) -> Vec<PotentialWeights>,
    {
        let nspin = self.tables.len();
        let per_spin: Vec<PointDensities> = (0..nspin)
            .map(|s| {
                let t = self.tables[s];
                let input = t.range(self.input);
                // Densities of spin `s` among the pass's `n` (interleaved).
                let members = (n + nspin - 1 - s) / nspin;
                let chunk = t.chunk(c0, c1);
                chunk_densities(chunk, t.nocc(), &stacked[s], members, input, self.scale)
            })
            .collect();
        let pd = interleave(per_spin, n);
        let ws = weights(c0, c1, &pd);
        let mut per_spin_out: Vec<std::vec::IntoIter<Array2<f64>>> = (0..nspin)
            .map(|s| {
                let t = self.tables[s];
                let mine: Vec<PotentialWeights> =
                    ws.iter().skip(s).step_by(nspin).cloned().collect();
                chunk_potentials(t.chunk(c0, c1), t.nocc(), &mine, t.range(self.output)).into_iter()
            })
            .collect();
        (0..n)
            .map(|x| {
                per_spin_out[x % nspin]
                    .next()
                    .expect("one result per density")
            })
            .collect()
    }
}

/// The rotations of spin `s` among `nspin` interleaved spins.
fn spin_members<'a>(us: &[&'a Array2<f64>], s: usize, nspin: usize) -> Vec<&'a Array2<f64>> {
    us.iter().skip(s).step_by(nspin).copied().collect()
}

/// Point densities of `n` densities in pass order from the per-spin ones
/// (density `x` is row `x / nspin` of spin `x % nspin`).
fn interleave(mut per_spin: Vec<PointDensities>, n: usize) -> PointDensities {
    let nspin = per_spin.len();
    if nspin == 1 {
        return per_spin.remove(0);
    }
    let nc = per_spin[0].rho1.ncols();
    let mut rho1 = Array2::<f64>::zeros((n, nc));
    let mut g1 = per_spin[0]
        .g1
        .as_ref()
        .map(|_| Array3::<f64>::zeros((n, 3, nc)));
    for x in 0..n {
        let src = &per_spin[x % nspin];
        rho1.row_mut(x).assign(&src.rho1.row(x / nspin));
        if let (Some(dst), Some(g)) = (g1.as_mut(), src.g1.as_ref()) {
            dst.index_axis_mut(Axis(0), x)
                .assign(&g.index_axis(Axis(0), x / nspin));
        }
    }
    PointDensities { rho1, g1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pass width is a function of the sizes alone: the same under every
    /// pool, whole spin pairs, at least one group, and its live set within the
    /// ceiling whenever more than one group fits.
    #[test]
    fn mo_block_width_is_charged_and_ignores_the_pool() {
        for threads in [1usize, 2, 12] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for (nmo, nocc) in [(10usize, 5usize), (106, 17), (500, 80), (4000, 2000)] {
                    for group in [1usize, 2] {
                        let w = mo_block_width(nmo, nocc, group);
                        assert!(w.is_multiple_of(group) && w >= group);
                        let live = MAX_BATCH_BAND * w * per_density_chunk_bytes(nmo, nocc);
                        assert!(w == group || live <= RESPONSE_BLOCK_BYTES, "{nmo} {nocc}");
                    }
                }
                assert_eq!(mo_block_width(106, 17, 1), 19);
            });
        }
    }
}
