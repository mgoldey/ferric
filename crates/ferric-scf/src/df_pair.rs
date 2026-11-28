//! Pair-major, Schwarz-sparse storage of the dressed three-index tensor: one
//! tensor for RI-J and DF-K (the default RI-JK route where it qualifies;
//! `FERRIC_DF_LAYOUT=aux` opts out).
//!
//! The aux-major RI-JK route ([`crate::df_j`], [`crate::df_k`]) holds two
//! tensors, the raw `(P|μν)` for J and the dressed `B = L⁻¹(P|μν)` for K, each
//! `[P][μν]` over every AO pair. This module holds ONE, AO-major:
//!
//! ```text
//!   B_μ[ν][P]   for ν ∈ sig(μ) = { ν : shell pair (μ, ν) survives Schwarz }
//! ```
//!
//! with `P` the fastest index, so the block of one `μ` is the column-major
//! `naux × |sig(μ)|` matrix BLAS wants and no contraction unpacks or repacks.
//!
//! * **Integrals** are generated pair-outer ([`PairMajor::fill`]): only kept
//!   shell pairs, only `ν ≤ μ`.
//! * **Dressing** is in place ([`PairMajor::dress`]): one `DTRMM` per `μ` by
//!   the Cholesky `L⁻¹` on the columns `ν ≤ μ`, then a mirror to `ν > μ`.
//!   `naux² · (kept lower pairs)` flops and no second tensor.
//! * **J** ([`PairMajor::build_j`]) is two `DGEMV` per `μ` on the dressed
//!   tensor, `J = Σ_P B_P (B_P · D)`: the same fit as `(P|μν) V⁻¹ (Q|λσ)`
//!   without a raw tensor or a metric solve per build.
//! * **K** ([`PairMajor::build_k_occ`]) half-transforms with one `DGEMM` per
//!   `μ` against the gathered rows `C[sig(μ)]`, then reduces `K = T Tᵀ` over
//!   the `(n, naux·nocc)` intermediate, one `DSYRK` per fixed chunk of its
//!   `(P, i)` index, the partials folded in chunk order.
//!
//! # Error of the screen
//!
//! A dropped shell pair has `Q_μν · max_P Q_P ≤ thresh`
//! ([`PairLayout::schwarz`], the rule of
//! [`PairScreen`](ferric_integrals::three_index_source::PairScreen)), so every
//! raw `(P|μν)` of it is at most `thresh`; it is stored as zero. Kept pairs
//! are the unscreened build's integrals (same engine call and precision).
//!
//! # Determinism
//!
//! Every output element is produced by one BLAS call whose shape depends on
//! the layout only (`μ`, the fixed [`K_REDUCE_ROWS`], the aux block of
//! [`half_transform_rows`]) and partial sums are folded in a fixed order, so
//! J and K are bit-identical across rayon thread counts at one BLAS thread.
//!
//! [`PairMajor::fill`]: crate::df_pair::PairMajor::fill
//! [`PairMajor::dress`]: crate::df_pair::PairMajor::dress
//! [`PairMajor::build_j`]: crate::df_pair::PairMajor::build_j
//! [`PairMajor::build_k_occ`]: crate::df_pair::PairMajor::build_k_occ
//! [`PairLayout::schwarz`]: crate::df_pair::PairLayout::schwarz
//! [`K_REDUCE_ROWS`]: crate::df_pair::K_REDUCE_ROWS
//! [`half_transform_rows`]: crate::df_pair::half_transform_rows

use ferric_core::memory::pool::Reservation;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::engine_pool::for_each_with_engine;
use ferric_integrals::operator::Operator;
use ndarray::{Array2, Array3};
use rayon::prelude::*;

extern "C" {
    fn dtrmm_(
        side: *const u8,
        uplo: *const u8,
        transa: *const u8,
        diag: *const u8,
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
    );
    fn dgemm_(
        transa: *const u8,
        transb: *const u8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
    fn dsyrk_(
        uplo: *const u8,
        trans: *const u8,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
    fn dgemv_(
        trans: *const u8,
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        x: *const f64,
        incx: *const i32,
        beta: *const f64,
        y: *mut f64,
        incy: *const i32,
    );
}

/// Pool label of the resident tensor.
const PAIR_TENSOR_LABEL: &str = "DF pair-major dressed B[mu][nu][P]";

/// Rows of the half-transformed intermediate one K-reduction DSYRK reads.
/// Fixed (never the thread count): the chunks set K's summation order.
pub const K_REDUCE_ROWS: usize = 2048;

/// Bytes the half-transformed intermediate `T` of one aux block may take.
/// Fixed, so the aux blocks (and with them K's summation order) depend on
/// `(n, nocc)` only.
const HALF_TRANSFORM_BYTES: usize = 256 * 1024 * 1024;

/// Aux rows per half-transform block for an `n × nocc` factor: as many as
/// keep `T` (`n · nocc · rows` doubles) within a fixed 256 MiB, at
/// least one.
pub fn half_transform_rows(n: usize, nocc: usize, naux: usize) -> usize {
    (HALF_TRANSFORM_BYTES / (n * nocc * 8).max(1)).clamp(1, naux.max(1))
}

/// Which AO pairs the pair-major tensor stores: per function `μ` the
/// ascending list `sig(μ)` of partners, decided per shell pair.
#[derive(Debug, Clone)]
pub struct PairLayout {
    nao: usize,
    /// `sig[sig_start[μ]..sig_start[μ + 1]]`: the partners of `μ`, ascending.
    sig: Vec<u32>,
    sig_start: Vec<usize>,
    /// Partners `ν ≤ μ` of each `μ` (a prefix of its list).
    n_lower: Vec<u32>,
    /// Per shell `s1`: its kept partner shells, ascending, and for each the
    /// position of its first function inside `sig(μ)` of any `μ ∈ s1`.
    shell_partners: Vec<Vec<(u32, u32)>>,
    /// The shell sizes and AO offsets the layout was built for.
    shell_dims: Vec<usize>,
    shell_offs: Vec<usize>,
    kept_shell_pairs: usize,
    thresh: f64,
}

impl PairLayout {
    /// Keep shell pair `(s1, s2)` iff `Q[s1, s2] · q_aux_max > thresh`, with
    /// `Q` the operator's Schwarz table of `obs` and `q_aux_max` the largest
    /// `√(P|P)` of the same operator's aux metric. `thresh ≤ 0` keeps all.
    pub fn schwarz(
        op: Operator,
        obs: &PreparedBasis,
        q_aux_max: f64,
        thresh: f64,
    ) -> Result<Self, FerricError> {
        let q = ferric_integrals::schwarz::schwarz(op, obs)?;
        Ok(Self::from_shell_bounds(
            &q,
            obs.shell_dims(),
            obs.shell_offsets(),
            q_aux_max,
            thresh,
        ))
    }

    /// The pure part of [`Self::schwarz`]: `q` is read at `(max, min)`.
    pub fn from_shell_bounds(
        q: &Array2<f64>,
        dims: &[usize],
        offs: &[usize],
        q_aux_max: f64,
        thresh: f64,
    ) -> Self {
        let nsh = dims.len();
        let nao = (0..nsh).map(|s| offs[s] + dims[s]).max().unwrap_or(0);
        let kept = |a: usize, b: usize| {
            let (hi, lo) = (a.max(b), a.min(b));
            thresh <= 0.0 || q[(hi, lo)] * q_aux_max > thresh
        };
        let mut shell_partners = Vec::with_capacity(nsh);
        let mut sig = Vec::new();
        let mut sig_start = vec![0usize; nao + 1];
        let mut n_lower = vec![0u32; nao];
        let mut kept_shell_pairs = 0usize;
        for s1 in 0..nsh {
            let mut partners = Vec::new();
            let mut pos = 0u32;
            for s2 in (0..nsh).filter(|&s2| kept(s1, s2)) {
                partners.push((s2 as u32, pos));
                pos += dims[s2] as u32;
                kept_shell_pairs += usize::from(s2 <= s1);
            }
            for i in 0..dims[s1] {
                let mu = offs[s1] + i;
                let mut lower = 0u32;
                for &(s2, _) in &partners {
                    let s2 = s2 as usize;
                    for j in 0..dims[s2] {
                        let nu = offs[s2] + j;
                        lower += u32::from(nu <= mu);
                        sig.push(nu as u32);
                    }
                }
                n_lower[mu] = lower;
                sig_start[mu + 1] = sig.len();
            }
            shell_partners.push(partners);
        }
        Self {
            nao,
            sig,
            sig_start,
            n_lower,
            shell_partners,
            shell_dims: dims.to_vec(),
            shell_offs: offs[..nsh].to_vec(),
            kept_shell_pairs,
            thresh,
        }
    }

    /// Whether this layout was built for exactly `obs`'s shell partition
    /// (sizes and AO offsets, shell by shell): what makes every column
    /// position [`PairMajor::fill`] writes lie inside its function's list.
    pub fn matches(&self, obs: &PreparedBasis) -> bool {
        let nsh = obs.nshells();
        self.shell_dims.as_slice() == obs.shell_dims()
            && obs.shell_offsets().len() >= nsh
            && self.shell_offs.as_slice() == &obs.shell_offsets()[..nsh]
    }

    /// AO functions.
    pub fn nao(&self) -> usize {
        self.nao
    }
    /// Stored `(μ, ν)` entries, both triangles (the tensor's column count).
    pub fn stored(&self) -> usize {
        self.sig.len()
    }
    /// Stored entries with `ν ≤ μ`: what is computed and dressed.
    pub fn stored_lower(&self) -> usize {
        self.n_lower.iter().map(|&c| c as usize).sum()
    }
    /// Kept shell pairs `(s1, s2 ≤ s1)`.
    pub fn kept_shell_pairs(&self) -> usize {
        self.kept_shell_pairs
    }
    /// The screening threshold this layout was built with.
    pub fn thresh(&self) -> f64 {
        self.thresh
    }
    /// Bytes of the f64 tensor for `naux` auxiliary functions.
    pub fn tensor_bytes(&self, naux: usize) -> usize {
        self.stored().saturating_mul(naux).saturating_mul(8)
    }
    /// The partners of `mu`, ascending.
    pub fn partners(&self, mu: usize) -> &[u32] {
        &self.sig[self.sig_start[mu]..self.sig_start[mu + 1]]
    }
    /// How many of [`Self::partners`] are `≤ mu` (its leading entries).
    pub fn lower_count(&self, mu: usize) -> usize {
        self.n_lower[mu] as usize
    }
}

/// The dressed three-index tensor in the pair-major layout (module docs).
pub struct PairMajor {
    naux: usize,
    layout: PairLayout,
    /// `data[(sig_start[μ] + k)·naux + P] = B[P, μ, sig(μ)[k]]`.
    data: Vec<f64>,
    /// The half-transformed intermediate of [`Self::build_k_occ`], reused.
    k_scratch: std::sync::Mutex<KScratch>,
    /// Set when the tensor is held on column-rotated shells: every public
    /// entry point takes and returns parent-basis matrices.
    rot: Option<Rotation>,
    _charge: Reservation,
}

/// The retained K workspace and the pool charge for its capacity.
#[derive(Default)]
struct KScratch {
    buf: Vec<f64>,
    charge: Option<Reservation>,
}

/// The column rotation a [`PairMajor`] tensor is stored under: `χ = T χ'`.
struct Rotation {
    basis: ferric_integrals::contraction_rotation::RotatedBasis,
    /// `D' = Tᵀ D T`.
    forward: ferric_integrals::contraction_rotation::AoTranspose,
    /// Dense `T`, for `C' = Tᵀ C`.
    t: Array2<f64>,
}

impl std::fmt::Debug for PairMajor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairMajor")
            .field("naux", &self.naux)
            .field("nao", &self.layout.nao)
            .field("stored", &self.layout.stored())
            .field("thresh", &self.layout.thresh)
            .finish()
    }
}

/// A `*mut f64` shared with rayon workers that write disjoint regions.
#[derive(Clone, Copy)]
struct SharedMut(*mut f64);

impl SharedMut {
    /// The pointer. A method, so a closure captures the whole (`Sync`)
    /// wrapper and not its raw-pointer field.
    fn ptr(self) -> *mut f64 {
        self.0
    }
}
// SAFETY: every user documents why its writes are disjoint.
unsafe impl Send for SharedMut {}
unsafe impl Sync for SharedMut {}

impl PairMajor {
    /// Raw `(P|μν)` for the kept pairs with `ν ≤ μ`; the `ν > μ` columns stay
    /// zero until [`Self::dress`] mirrors them. Parallel over the orbital
    /// shell `s1`, which owns the columns of its functions.
    pub fn fill(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        layout: PairLayout,
    ) -> Result<Self, FerricError> {
        let naux = dfbs.nbasis();
        let charge = ferric_core::memory::pool::reserve_global(
            PAIR_TENSOR_LABEL,
            layout.tensor_bytes(naux),
        )?;
        Self::fill_charged(op, obs, dfbs, layout, charge)
    }

    /// [`Self::fill`] with the tensor's pool charge already taken.
    fn fill_charged(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        layout: PairLayout,
        charge: Reservation,
    ) -> Result<Self, FerricError> {
        let naux = dfbs.nbasis();
        // The raw-pointer scatter below trusts the layout's column positions,
        // so a layout built for another shell partition (equal AO counts are
        // not enough) is refused before any worker starts.
        if layout.nao != obs.nbasis() || !layout.matches(obs) {
            return Err(FerricError::General(format!(
                "pair-major fill: the layout ({} functions, {} shells) was not built for \
                 this orbital basis ({} functions, {} shells)",
                layout.nao,
                layout.shell_dims.len(),
                obs.nbasis(),
                obs.nshells()
            )));
        }
        let mut data = vec![0.0f64; layout.stored() * naux];
        let base = SharedMut(data.as_mut_ptr());
        let (dims, offs) = (obs.shell_dims(), obs.shell_offsets());
        let (dims_df, offs_df) = (dfbs.shell_dims(), dfbs.shell_offsets());
        let nsh = obs.nshells();
        let triples = dfbs.nshells() * layout.kept_shell_pairs;
        let parallel = triples >= ferric_integrals::threeindex::PAR_ERI3_TRIPLE_THRESHOLD;
        let layout_ref = &layout;
        for_each_with_engine(
            parallel,
            nsh,
            || Engine::new_3center(op, obs, dfbs, 1e-14),
            |eng, s1| -> Result<(), FerricError> {
                let (n1, m0) = (dims[s1], offs[s1]);
                for &(s2, pos) in &layout_ref.shell_partners[s1] {
                    let (s2, pos) = (s2 as usize, pos as usize);
                    if s2 > s1 {
                        break;
                    }
                    let n2 = dims[s2];
                    for sp in 0..dfbs.nshells() {
                        let Some(block) = eng.compute_eri3(obs, dfbs, sp, s1, s2) else {
                            continue;
                        };
                        let (np, pbase) = (dims_df[sp], offs_df[sp]);
                        for i in 0..n1 {
                            let col0 = layout_ref.sig_start[m0 + i] + pos;
                            let jmax = if s2 == s1 { i + 1 } else { n2 };
                            for j in 0..jmax {
                                // SAFETY: shell `s1` owns the columns of its
                                // functions (`sig_start[m0 + i]..`), one task
                                // per `s1`; `col0 + j` is inside that list and
                                // `pbase + p < naux`.
                                let dst = unsafe { base.ptr().add((col0 + j) * naux + pbase) };
                                for p in 0..np {
                                    unsafe { *dst.add(p) = block[(p * n1 + i) * n2 + j] };
                                }
                            }
                        }
                    }
                }
                Ok(())
            },
        )?;
        Ok(Self {
            naux,
            layout,
            data,
            k_scratch: std::sync::Mutex::new(KScratch::default()),
            rot: None,
            _charge: charge,
        })
    }

    /// Dress in place: `B_μ[:, ν ≤ μ] ← L⁻¹ B_μ[:, ν ≤ μ]`, then copy every
    /// `ν > μ` column from its mirror. `linv` is the row-major lower
    /// triangular `L⁻¹` of `V = L Lᵀ`
    /// ([`ferric_core::linalg::cholesky_inverse_lower`]).
    #[allow(clippy::manual_c_str_literals)] // byte literals match the *const u8 BLAS args
    pub fn dress(&mut self, linv: &Array2<f64>) -> Result<(), FerricError> {
        let naux = self.naux;
        if linv.dim() != (naux, naux) {
            return Err(FerricError::General(format!(
                "pair-major dress: the metric factor is {:?}, expected ({naux}, {naux})",
                linv.dim()
            )));
        }
        let linv = linv.as_standard_layout();
        let lptr = linv.as_ptr() as usize;
        let base = SharedMut(self.data.as_mut_ptr());
        let layout = &self.layout;
        let t = ferric_core::timing::Stage::start("dfpair:dress-trmm");
        (0..layout.nao).into_par_iter().for_each(|mu| {
            let (m, n) = (naux as i32, layout.n_lower[mu] as i32);
            if n == 0 {
                return;
            }
            let one = 1.0f64;
            // The row-major lower `L⁻¹` is the column-major UPPER `(L⁻¹)ᵀ`, so
            // `L⁻¹ X` is `Aᵀ X` with `A` upper.
            // SAFETY: the block of `mu` is `naux × n_lower[mu]` column-major at
            // `sig_start[mu]·naux`, disjoint from every other `mu`'s.
            unsafe {
                dtrmm_(
                    b"L\0".as_ptr(),
                    b"U\0".as_ptr(),
                    b"T\0".as_ptr(),
                    b"N\0".as_ptr(),
                    &m,
                    &n,
                    &one,
                    lptr as *const f64,
                    &m,
                    base.ptr().add(layout.sig_start[mu] * naux),
                    &m,
                );
            }
        });
        t.end();
        let t = ferric_core::timing::Stage::start("dfpair:dress-mirror");
        (0..layout.nao).into_par_iter().for_each(|mu| {
            let partners = layout.partners(mu);
            for (k, &nu) in partners.iter().enumerate().skip(layout.lower_count(mu)) {
                let nu = nu as usize;
                let lower = &layout.partners(nu)[..layout.lower_count(nu)];
                let pos = lower
                    .binary_search(&(mu as u32))
                    .expect("the pair layout is symmetric");
                // SAFETY: the source is a `ν ≤ μ'` column (never written
                // here), the destination a `ν > μ` column of `mu` (written by
                // this task only); both are `naux` long and in bounds.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        base.ptr().add((layout.sig_start[nu] + pos) * naux) as *const f64,
                        base.ptr().add((layout.sig_start[mu] + k) * naux),
                        naux,
                    );
                }
            }
        });
        t.end();
        Ok(())
    }

    /// Auxiliary functions.
    pub fn naux(&self) -> usize {
        self.naux
    }
    /// AO functions.
    pub fn nao(&self) -> usize {
        self.layout.nao
    }
    /// The stored pairs.
    pub fn layout(&self) -> &PairLayout {
        &self.layout
    }
    /// Bytes of the resident tensor.
    pub fn bytes(&self) -> usize {
        self.data.len() * 8
    }

    fn block_ptr(&self, mu: usize) -> *const f64 {
        // In bounds: `sig_start[mu] ≤ stored`, and the tensor is `stored·naux`.
        unsafe {
            self.data
                .as_ptr()
                .add(self.layout.sig_start[mu] * self.naux)
        }
    }

    /// Write aux rows `[p0, p1)` as the dense symmetric `(p1 − p0, n, n)`
    /// block the aux-major consumers read, in the PARENT AO basis; dropped
    /// pairs are zero (before the back-transform, when the tensor is held on
    /// rotated shells).
    pub fn unpack_square(
        &self,
        p0: usize,
        p1: usize,
        out: &mut Array3<f64>,
    ) -> Result<(), FerricError> {
        let Some(rot) = &self.rot else {
            self.unpack_square_stored(p0, p1, out);
            return Ok(());
        };
        let (n, w) = (self.layout.nao, p1 - p0);
        assert_eq!(out.dim(), (w, n, n), "pair-major unpack: block shape");
        // Pair rows `(μ n + ν, P)`, the layout the rotation's own
        // back-transform takes: `B_P = T B'_P Tᵀ` for every P at once.
        let mut rows = vec![0.0f64; n * n * w];
        for mu in 0..n {
            let col0 = self.layout.sig_start[mu];
            for (k, &nu) in self.layout.partners(mu).iter().enumerate() {
                let src = &self.data[(col0 + k) * self.naux + p0..(col0 + k) * self.naux + p1];
                let dst = (mu * n + nu as usize) * w;
                rows[dst..dst + w].copy_from_slice(src);
            }
        }
        rot.basis.back_transform_pair_rows(&mut rows, w)?;
        for mu in 0..n {
            for nu in 0..n {
                let src = (mu * n + nu) * w;
                for p in 0..w {
                    out[(p, mu, nu)] = rows[src + p];
                }
            }
        }
        Ok(())
    }

    /// `J[μν] = Σ_P B[P,μν] g_P`, `g_P = Σ_λσ B[P,λσ] D[λσ]`, for any `D`
    /// (only its symmetric part contributes). `j` is overwritten. With the
    /// tensor on rotated shells: `J = T J'(Tᵀ D T) Tᵀ`.
    pub fn build_j(&self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<(), FerricError> {
        let Some(rot) = &self.rot else {
            return self.build_j_stored(d, j);
        };
        let d_rot = rot.forward.matrix(d);
        let mut j_rot = Array2::<f64>::zeros(j.raw_dim());
        self.build_j_stored(&d_rot, &mut j_rot)?;
        j.assign(&rot.basis.back_transform_matrix(&j_rot)?);
        Ok(())
    }

    /// `K = Σ_P (B_P C)(B_P C)ᵀ` for an `n × m` factor `C` (occupied orbitals
    /// with `D = C Cᵀ`, or any factor of a positive semidefinite density).
    /// `k` is overwritten and exactly symmetric. With the tensor on rotated
    /// shells: `K = T K'(Tᵀ C) Tᵀ`.
    pub fn build_k_occ(&self, c: &Array2<f64>, k: &mut Array2<f64>) -> Result<(), FerricError> {
        self.build_k_occ_banded(c, k, crate::reduce::default_band_bytes())
    }

    /// [`Self::build_k_occ`] with the live-set budget of the reduction's
    /// partials given (`reduce::resolve_band_bytes` of the builder's budget,
    /// as the aux-major occupied path sizes its band). K does not depend on
    /// it.
    pub fn build_k_occ_banded(
        &self,
        c: &Array2<f64>,
        k: &mut Array2<f64>,
        band_bytes: usize,
    ) -> Result<(), FerricError> {
        let Some(rot) = &self.rot else {
            return self.build_k_occ_stored(c, k, band_bytes);
        };
        if c.nrows() != self.layout.nao {
            return self.build_k_occ_stored(c, k, band_bytes); // reports the shape error
        }
        let c_rot = rot.t.t().dot(c);
        let mut k_rot = Array2::<f64>::zeros(k.raw_dim());
        self.build_k_occ_stored(&c_rot, &mut k_rot, band_bytes)?;
        k.assign(&rot.basis.back_transform_matrix(&k_rot)?);
        Ok(())
    }

    /// Whether the tensor is held on column-rotated shells.
    pub fn is_rotated(&self) -> bool {
        self.rot.is_some()
    }

    /// The whole set-up: the pair screen, the fill and the dressing, on the
    /// column-rotated shells of `obs` when `level` is given and something
    /// rotates (`ferric_integrals::contraction_rotation`), else on `obs`
    /// itself. `linv` and `q_aux_max` as in [`PairMajor::dress`] and
    /// [`PairLayout::schwarz`]. `None` when the tensor would exceed
    /// `budget_bytes`.
    pub fn build(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        linv: &Array2<f64>,
        (q_aux_max, thresh): (f64, f64),
        level: Option<ferric_integrals::contraction_rotation::RotationLevel>,
        budget_bytes: usize,
    ) -> Result<Option<Self>, FerricError> {
        use ferric_core::timing::Stage;
        use ferric_integrals::contraction_rotation::{DerivRotMutant, RotatedBasis};
        let rot = match level {
            Some(level) => RotatedBasis::detect_prepared(obs, level)?,
            None => None,
        };
        let ints = rot.as_ref().map_or(obs, |r| &r.prep);
        let t = Stage::start("dfpair:schwarz");
        let layout = PairLayout::schwarz(op, ints, q_aux_max, thresh)?;
        t.end();
        if layout.tensor_bytes(dfbs.nbasis()) > budget_bytes {
            return Ok(None);
        }
        // A pool that cannot hold the tensor is a reason to take the
        // aux-major route (which can spill or recompute), not to end the run.
        let Some(charge) = ferric_core::memory::pool::try_reserve_global(
            PAIR_TENSOR_LABEL,
            layout.tensor_bytes(dfbs.nbasis()),
        ) else {
            return Ok(None);
        };
        let t = Stage::start("dfpair:fill");
        let mut tensor = Self::fill_charged(op, ints, dfbs, layout, charge)?;
        t.end();
        let t = Stage::start("dfpair:dress");
        tensor.dress(linv)?;
        t.end();
        tensor.rot = rot.map(|basis| Rotation {
            forward: basis.ao_transpose(DerivRotMutant::None),
            t: basis.dense_t(),
            basis,
        });
        Ok(Some(tensor))
    }

    /// The stored tensor's own rows `[p0, p1)`, unpacked (rotated shells
    /// when [`Self::is_rotated`]).
    fn unpack_square_stored(&self, p0: usize, p1: usize, out: &mut Array3<f64>) {
        let n = self.layout.nao;
        assert_eq!(out.dim(), (p1 - p0, n, n), "pair-major unpack: block shape");
        out.fill(0.0);
        for mu in 0..n {
            let col0 = self.layout.sig_start[mu];
            for (k, &nu) in self.layout.partners(mu).iter().enumerate() {
                let col = &self.data[(col0 + k) * self.naux..(col0 + k + 1) * self.naux];
                for p in p0..p1 {
                    out[(p - p0, mu, nu as usize)] = col[p];
                }
            }
        }
    }

    /// `J[μν] = Σ_P B[P,μν] g_P`, `g_P = Σ_λσ B[P,λσ] D[λσ]`, for any `D`
    /// (only its symmetric part contributes). `j` is overwritten; a dropped
    /// pair's element is zero.
    #[allow(clippy::manual_c_str_literals)]
    fn build_j_stored(&self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<(), FerricError> {
        let (n, naux) = (self.layout.nao, self.naux);
        if d.dim() != (n, n) || j.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "pair-major J: D is {:?} and J is {:?}, expected ({n}, {n})",
                d.dim(),
                j.dim()
            )));
        }
        let (one, zero, inc, m) = (1.0f64, 0.0f64, 1i32, naux as i32);
        // g_P: one GEMV per μ over its ν ≤ μ columns, weights D_μν + D_νμ.
        let partials: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|mu| {
                let nl = self.layout.lower_count(mu);
                let mut y = vec![0.0f64; naux];
                if nl == 0 {
                    return y;
                }
                let w: Vec<f64> = self.layout.partners(mu)[..nl]
                    .iter()
                    .map(|&nu| {
                        let nu = nu as usize;
                        if nu == mu {
                            d[(mu, mu)]
                        } else {
                            d[(mu, nu)] + d[(nu, mu)]
                        }
                    })
                    .collect();
                let cols = nl as i32;
                // SAFETY: the block is `naux × nl` column-major; `w` has `nl`
                // and `y` has `naux` elements.
                unsafe {
                    dgemv_(
                        b"N\0".as_ptr(),
                        &m,
                        &cols,
                        &one,
                        self.block_ptr(mu),
                        &m,
                        w.as_ptr(),
                        &inc,
                        &zero,
                        y.as_mut_ptr(),
                        &inc,
                    );
                }
                y
            })
            .collect();
        // Fold in μ order: the determinism anchor (see `reduce`).
        let mut g = vec![0.0f64; naux];
        for y in &partials {
            for (a, b) in g.iter_mut().zip(y) {
                *a += *b;
            }
        }
        drop(partials);
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|mu| {
                let nl = self.layout.lower_count(mu);
                let mut out = vec![0.0f64; nl];
                if nl == 0 {
                    return out;
                }
                let cols = nl as i32;
                // SAFETY: as above, with `g` (`naux`) in and `out` (`nl`) out.
                unsafe {
                    dgemv_(
                        b"T\0".as_ptr(),
                        &m,
                        &cols,
                        &one,
                        self.block_ptr(mu),
                        &m,
                        g.as_ptr(),
                        &inc,
                        &zero,
                        out.as_mut_ptr(),
                        &inc,
                    );
                }
                out
            })
            .collect();
        j.fill(0.0);
        for (mu, row) in rows.iter().enumerate() {
            for (&nu, &v) in self.layout.partners(mu).iter().zip(row) {
                j[(mu, nu as usize)] = v;
                j[(nu as usize, mu)] = v;
            }
        }
        Ok(())
    }

    /// `K = Σ_P (B_P C)(B_P C)ᵀ` for an `n × m` factor `C` (occupied orbitals
    /// with `D = C Cᵀ`, or any factor of a positive semidefinite density).
    /// `k` is overwritten and exactly symmetric.
    #[allow(clippy::manual_c_str_literals)]
    fn build_k_occ_stored(
        &self,
        c: &Array2<f64>,
        k: &mut Array2<f64>,
        band_bytes: usize,
    ) -> Result<(), FerricError> {
        let (n, naux) = (self.layout.nao, self.naux);
        if c.nrows() != n || k.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "pair-major K: C has {} rows and K is {:?}, expected {n} and ({n}, {n})",
                c.nrows(),
                k.dim()
            )));
        }
        let nocc = c.ncols();
        k.fill(0.0);
        if nocc == 0 || n == 0 {
            return Ok(());
        }
        let rows = half_transform_rows(n, nocc, naux);
        // Kept between builds: a fresh 100+ MB zeroed buffer per build cost a
        // fifth of a steady K at n-dodecane / cc-pVDZ. Every element read is
        // written first (each block's GEMMs fill `t[..n·ld]`). The pool charge
        // lives with the buffer, for as long as it is retained (soft: a
        // refused charge leaves the buffer uncounted, as the aux-major
        // scratch charges do, rather than failing a K build).
        let mut guard = self
            .k_scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let KScratch { buf, charge } = &mut *guard;
        let need = n * nocc * rows;
        if buf.len() < need {
            *charge = None; // release before asking for the larger size
                            // `reserve_exact`, then charge what the allocator actually gave:
                            // `resize` alone may over-allocate past the charged length.
            buf.reserve_exact(need - buf.len());
            buf.resize(need, 0.0);
            *charge = ferric_core::memory::pool::try_reserve_global(
                "DF pair-major half-transformed T[mu][i][P]",
                buf.capacity() * 8,
            );
        }
        let t: &mut [f64] = buf;
        let (one, zero) = (1.0f64, 0.0f64);
        let mut p0 = 0usize;
        while p0 < naux {
            let pb = rows.min(naux - p0);
            let ld = pb * nocc;
            // T column μ (column-major, `ld` long) = B_μ[p0.., sig] · C[sig, :].
            t[..n * ld]
                .par_chunks_mut(ld)
                .enumerate()
                .for_each(|(mu, tcol)| {
                    let partners = self.layout.partners(mu);
                    let ns = partners.len();
                    if ns == 0 {
                        tcol.fill(0.0);
                        return;
                    }
                    // C[sig, :] column-major (ns × nocc).
                    let mut csig = vec![0.0f64; ns * nocc];
                    for (r, &nu) in partners.iter().enumerate() {
                        for i in 0..nocc {
                            csig[i * ns + r] = c[(nu as usize, i)];
                        }
                    }
                    let (m, nn, kk, lda) = (pb as i32, nocc as i32, ns as i32, naux as i32);
                    // SAFETY: A is rows [p0, p0+pb) of the `naux × ns` block
                    // (lda = naux), B is `ns × nocc`, C is `pb × nocc` = `tcol`.
                    unsafe {
                        dgemm_(
                            b"N\0".as_ptr(),
                            b"N\0".as_ptr(),
                            &m,
                            &nn,
                            &kk,
                            &one,
                            self.block_ptr(mu).add(p0),
                            &lda,
                            csig.as_ptr(),
                            &kk,
                            &zero,
                            tcol.as_mut_ptr(),
                            &m,
                        );
                    }
                });
            // K += Tᵀ T as one DSYRK per fixed chunk of T's rows (the (P, i)
            // index), the partials folded in chunk order.
            let tptr = t.as_ptr() as usize;
            let n_chunks = ld.div_ceil(K_REDUCE_ROWS);
            crate::reduce::grouped_deterministic_sum(k, n_chunks, n, band_bytes, |ci| {
                let r0 = ci * K_REDUCE_ROWS;
                let (ni, kk, lda) = (n as i32, K_REDUCE_ROWS.min(ld - r0) as i32, ld as i32);
                let mut kc = Array2::<f64>::zeros((n, n));
                // SAFETY: A is rows [r0, r0 + kk) of the column-major
                // `ld × n` T (lda = ld); the output is `n × n`.
                unsafe {
                    dsyrk_(
                        b"U\0".as_ptr(),
                        b"T\0".as_ptr(),
                        &ni,
                        &kk,
                        &one,
                        (tptr as *const f64).add(r0),
                        &lda,
                        &zero,
                        kc.as_mut_ptr(),
                        &ni,
                    );
                }
                Ok(kc)
            })?;
            p0 += pb;
        }
        // DSYRK 'U' wrote the column-major upper triangle, which is the
        // row-major LOWER one: mirror it up.
        for mu in 0..n {
            for nu in 0..mu {
                k[(nu, mu)] = k[(mu, nu)];
            }
        }
        Ok(())
    }
}

/// `FERRIC_DF_LAYOUT`: `pair` (default: this module, for every RI-JK set-up
/// that qualifies; see `DfK::try_pair_major`) or `aux` (always the aux-major
/// tensors of [`crate::df_j`] / [`crate::df_k`], the only route before
/// 2026-11-27). The two agree to the screening and summation-order level
/// (measured `|ΔE| ≤ 6e-11` Ha on C4–C20 alkane RHF), not bit for bit.
static DF_LAYOUT: ferric_core::config::ConfigVar<bool> = ferric_core::config::ConfigVar {
    env_name: "FERRIC_DF_LAYOUT",
    default: true,
    parse: |s| match s.trim().to_ascii_lowercase().as_str() {
        "pair" => Ok(true),
        "aux" => Ok(false),
        _ => Err(format!(
            "invalid value {s:?}: expected \"aux\" or \"pair\" (unset means \"pair\")"
        )),
    },
    validate: ferric_core::config::accept_any,
};

thread_local! {
    static DF_LAYOUT_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Restores the previous [`override_pair_layout_for_test`] setting on drop.
#[doc(hidden)]
#[must_use = "the override ends when the guard is dropped"]
pub struct PairLayoutOverrideGuard(Option<bool>);

impl Drop for PairLayoutOverrideGuard {
    fn drop(&mut self) {
        let prev = self.0.take();
        DF_LAYOUT_OVERRIDE.with(|c| c.set(prev));
    }
}

/// Use `on` in place of `FERRIC_DF_LAYOUT` for every RI-JK set-up on THIS
/// thread until the guard drops. Test hook.
#[doc(hidden)]
pub fn override_pair_layout_for_test(on: bool) -> PairLayoutOverrideGuard {
    PairLayoutOverrideGuard(DF_LAYOUT_OVERRIDE.with(|c| c.replace(Some(on))))
}

/// Whether the pair-major layout is asked for (the override, else
/// `FERRIC_DF_LAYOUT`). A malformed value is an input error naming the
/// variable.
pub fn pair_layout_requested() -> Result<bool, FerricError> {
    if let Some(on) = DF_LAYOUT_OVERRIDE.with(|c| c.get()) {
        return Ok(on);
    }
    DF_LAYOUT
        .get()
        .map(|r| r.value)
        .map_err(|e| FerricError::General(format!("FERRIC_DF_LAYOUT: {e}")))
}

thread_local! {
    /// RI-JK set-ups on THIS thread that took the pair-major route.
    /// Thread-local, like [`override_pair_layout_for_test`], so a test reads
    /// only its own solves and never a sibling test's.
    static PAIR_MAJOR_TAKEN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Record that an RI-JK set-up on this thread took the pair-major route.
pub(crate) fn note_pair_major_taken() {
    PAIR_MAJOR_TAKEN.with(|c| c.set(c.get() + 1));
}

/// How many RI-JK set-ups on this thread took the pair-major route. Test hook.
#[doc(hidden)]
pub fn pair_major_count_for_test() -> usize {
    PAIR_MAJOR_TAKEN.with(std::cell::Cell::get)
}

/// RI-J on the shared pair-major tensor.
#[derive(Debug)]
pub struct PairJ {
    tensor: std::sync::Arc<PairMajor>,
}

impl PairJ {
    /// J from `tensor` (shared with the run's DF-K).
    pub fn new(tensor: std::sync::Arc<PairMajor>) -> Self {
        Self { tensor }
    }
    /// Auxiliary functions.
    pub fn naux(&self) -> usize {
        self.tensor.naux()
    }
}

impl crate::fock::JBuilder for PairJ {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        self.tensor.build_j(d, j)?;
        Ok(0)
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df_k::DfK;
    use crate::fock::{JBuilder, KBuilder};
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::threeindex::{coulomb_metric_2c, eri3_tensor};

    const AUX: &str = "def2-universal-jkfit";
    /// Two waters 6 Å apart: far enough that a 1e-5 Schwarz screen drops the
    /// intermolecular pairs of a minimal basis, close enough to keep some.
    const TWO_WATERS: &str = "6\n\n\
        O 0.000 0.000 0.117\nH 0.000 0.757 -0.467\nH 0.000 -0.757 -0.467\n\
        O 6.000 0.000 0.117\nH 6.000 0.757 -0.467\nH 6.000 -0.757 -0.467\n";

    struct System {
        obs: PreparedBasis,
        dfbs: PreparedBasis,
        linv: Array2<f64>,
        q_aux_max: f64,
    }

    fn system(obs: &str) -> System {
        let mol = Molecule::parse_xyz(TWO_WATERS, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled(obs).unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
        let v = coulomb_metric_2c(Operator::coulomb(), &dfbs).unwrap();
        let linv = ferric_core::linalg::cholesky_inverse_lower(&v).unwrap();
        let q_aux_max = v.diag().iter().fold(0.0f64, |m, x| m.max(x.sqrt()));
        System {
            obs,
            dfbs,
            linv,
            q_aux_max,
        }
    }

    fn tensor(sys: &System, thresh: f64) -> PairMajor {
        let op = Operator::coulomb();
        let layout = PairLayout::schwarz(op, &sys.obs, sys.q_aux_max, thresh).unwrap();
        let mut t = PairMajor::fill(op, &sys.obs, &sys.dfbs, layout).unwrap();
        t.dress(&sys.linv).unwrap();
        t
    }

    /// The test's own dressed tensor `(naux, n, n)`: the unscreened integrals
    /// with every pair `layout` drops set to zero, times `L⁻¹` by ndarray.
    fn reference_b(sys: &System, layout: &PairLayout) -> Array3<f64> {
        let mut raw = eri3_tensor(Operator::coulomb(), &sys.obs, &sys.dfbs).unwrap();
        let (naux, n, _) = raw.dim();
        for mu in 0..n {
            for nu in 0..n {
                if layout.partners(mu).binary_search(&(nu as u32)).is_err() {
                    raw.slice_mut(ndarray::s![.., mu, nu]).fill(0.0);
                }
            }
        }
        let flat = raw.into_shape_with_order((naux, n * n)).unwrap();
        sys.linv
            .dot(&flat)
            .into_shape_with_order((naux, n, n))
            .unwrap()
    }

    /// A deterministic, non-orthogonal factor with mixed signs.
    fn factor(n: usize, m: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, m), |(i, j)| {
            let x = ((i * 7 + j * 13 + 3) % 17) as f64 / 17.0 - 0.45;
            x * (1.0 + 0.1 * j as f64)
        })
    }

    /// The largest |element|, or infinity when any element is not finite
    /// (`f64::max` would drop a NaN and let a comparison pass).
    fn max_abs(a: &Array2<f64>) -> f64 {
        if a.iter().any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    /// Panics unless `got` matches a finite, non-vacuous `want` to `rel` of
    /// its largest element. A non-finite element on EITHER side fails (an
    /// infinite error against an infinite scale must not pass).
    fn assert_close(what: &str, got: &Array2<f64>, want: &Array2<f64>, rel: f64) {
        let (scale, err) = (max_abs(want), max_abs(&(got - want)));
        assert!(scale.is_finite(), "{what}: non-finite reference");
        assert!(scale > 1e-2, "{what}: vacuous reference ({scale:e})");
        assert!(
            err.is_finite() && err <= rel * scale,
            "{what}: off by {err:e} (largest element {scale:e})"
        );
    }

    /// Catches: a wrong column position in the fill, a dressing by `L⁻ᵀ` or
    /// the wrong triangle, a mirror from the wrong column, a J that drops the
    /// off-diagonal weight, a K reduction that reads the wrong block of `T`,
    /// and a screen that perturbs kept pairs. Bar: 1e-11 of the largest
    /// element (the two sides differ by GEMM reassociation only).
    #[test]
    fn j_and_k_match_an_independent_dressing_with_and_without_dropped_pairs() {
        let sys = system("sto-3g");
        let n = sys.obs.nbasis();
        for (thresh, drops) in [(0.0, false), (1e-5, true)] {
            let t = tensor(&sys, thresh);
            let stored = t.layout().stored();
            assert_eq!(
                stored < n * n,
                drops,
                "thresh {thresh:e}: {stored} of {}",
                n * n
            );
            for mu in 0..n {
                for &nu in t.layout().partners(mu) {
                    let back = t.layout().partners(nu as usize).binary_search(&(mu as u32));
                    assert!(back.is_ok(), "({mu}, {nu}) kept without its mirror");
                }
            }
            let b = reference_b(&sys, t.layout());
            let naux = b.dim().0;
            let c = factor(n, 4);
            // An asymmetric density: J must see only its symmetric part.
            let d = c.dot(&c.t()) + &factor(n, n) * 0.01;
            let (mut j_ref, mut k_ref) = (Array2::<f64>::zeros((n, n)), Array2::zeros((n, n)));
            for p in 0..naux {
                let bp = b.index_axis(ndarray::Axis(0), p);
                let g: f64 = bp.iter().zip(d.iter()).map(|(x, y)| x * y).sum();
                j_ref.scaled_add(g, &bp);
                let m = bp.dot(&c);
                k_ref += &m.dot(&m.t());
            }
            let (mut j, mut k) = (Array2::<f64>::zeros((n, n)), Array2::zeros((n, n)));
            t.build_j(&d, &mut j).unwrap();
            t.build_k_occ(&c, &mut k).unwrap();
            assert_close(&format!("thresh {thresh:e}: J"), &j, &j_ref, 1e-11);
            assert_close(&format!("thresh {thresh:e}: K"), &k, &k_ref, 1e-11);
            assert!(k == k.t(), "K is not exactly symmetric");
            assert!(j == j.t(), "J is not exactly symmetric");
        }
    }

    /// A layout built for another shell partition is refused before the
    /// raw-pointer fill, including one with the SAME number of functions
    /// (shell sizes permuted): its column positions would land outside the
    /// lists of this basis's functions.
    #[test]
    fn a_layout_for_another_shell_partition_is_refused() {
        let sys = system("sto-3g");
        let op = Operator::coulomb();
        let dims = sys.obs.shell_dims();
        let nsh = dims.len();
        // Two waters in STO-3G are [1, 1, 3, 1, 1] twice (a palindrome), so
        // rotate the shell sizes instead of reversing them.
        let mut permuted: Vec<usize> = dims.to_vec();
        permuted.rotate_left(2);
        assert_ne!(
            permuted.as_slice(),
            dims,
            "fixture needs unequal shell sizes"
        );
        let mut offs = vec![0usize; nsh];
        for s in 1..nsh {
            offs[s] = offs[s - 1] + permuted[s - 1];
        }
        let q = Array2::from_elem((nsh, nsh), 1.0);
        let wrong = PairLayout::from_shell_bounds(&q, &permuted, &offs, 1.0, 0.0);
        assert_eq!(
            wrong.nao(),
            sys.obs.nbasis(),
            "same AO count by construction"
        );
        assert!(!wrong.matches(&sys.obs));
        let err = PairMajor::fill(op, &sys.obs, &sys.dfbs, wrong)
            .expect_err("must be refused")
            .to_string();
        assert!(err.contains("was not built for"), "{err}");
        let right = PairLayout::schwarz(op, &sys.obs, sys.q_aux_max, 0.0).unwrap();
        assert!(right.matches(&sys.obs));
    }

    /// Catches a K (or J) whose bits depend on the rayon pool: a reduction
    /// block or a partial-sum order derived from the thread count.
    #[test]
    fn j_and_k_are_bit_identical_across_thread_counts() {
        let sys = system("def2-svp");
        let n = sys.obs.nbasis();
        let t = tensor(&sys, 1e-12);
        let c = factor(n, 7);
        let d = c.dot(&c.t());
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let (mut j, mut k) = (Array2::<f64>::zeros((n, n)), Array2::zeros((n, n)));
                t.build_j(&d, &mut j).unwrap();
                t.build_k_occ(&c, &mut k).unwrap();
                (j, k)
            })
        };
        let (one, four) = (run(1), run(4));
        assert!(one.0 == four.0, "J differs between 1 and 4 threads");
        assert!(one.1 == four.1, "K differs between 1 and 4 threads");
    }

    /// A DF-K on the pair-major tensor serves the density path (a general,
    /// asymmetric D) through the unpacked blocks, and RI-J reads the same
    /// tensor. Catches a wrong unpack (transposed, shifted aux row, a missing
    /// mirror). Bar: 1e-11 of the largest element, against the aux-major
    /// builders at f64.
    #[test]
    fn the_density_path_and_ri_j_agree_with_the_aux_major_builders() {
        let sys = system("def2-svp");
        let n = sys.obs.nbasis();
        let op = Operator::coulomb();
        let t = std::sync::Arc::new(tensor(&sys, 0.0));
        let mut pair_k = DfK::from_pair_major(t.clone(), usize::MAX);
        let mut pair_j = PairJ::new(t);
        let mut aux_k = DfK::new(op, &sys.obs, &sys.dfbs, usize::MAX).unwrap();
        let mut aux_j = crate::df_j::DfJ::new_with_precision(
            op,
            &sys.obs,
            &sys.dfbs,
            usize::MAX,
            crate::df_j::RiJPrecision::F64,
        )
        .unwrap();
        assert!(pair_k.is_pair_major() && !aux_k.is_pair_major());
        let c = factor(n, 5);
        let d = c.dot(&c.t()) + &factor(n, n) * 0.01;
        let build_k = |b: &mut DfK<'_>, occ: bool| {
            let mut k = Array2::<f64>::zeros((n, n));
            if occ {
                b.build_from_occ(&c, &mut k).unwrap();
            } else {
                b.build(&d, &mut k).unwrap();
            }
            k
        };
        for occ in [false, true] {
            let (a, b) = (build_k(&mut aux_k, occ), build_k(&mut pair_k, occ));
            assert_close(&format!("occ={occ}: K"), &b, &a, 1e-11);
        }
        let (mut ja, mut jp) = (Array2::<f64>::zeros((n, n)), Array2::zeros((n, n)));
        aux_j.build(&d, &mut ja).unwrap();
        pair_j.build(&d, &mut jp).unwrap();
        assert_close("J", &jp, &ja, 1e-11);
    }

    /// A tensor held on column-rotated shells (cc-pVDZ: the O s and p
    /// groups rotate) gives the parent-basis J, K and unpacked blocks.
    /// Catches a missing or one-sided transform of D, C, J or K, `T` used
    /// where `Tᵀ` belongs, and an unpack that returns rotated-basis blocks.
    /// Bar 1e-10 of the largest element: nothing is screened (thresh 0), so
    /// the two tensors span the same integrals and differ by the rounding of
    /// `T` (condition number below 2) on top of GEMM reassociation.
    #[test]
    fn a_rotated_tensor_gives_the_parent_basis_j_and_k() {
        use ferric_integrals::contraction_rotation::RotationLevel;
        let sys = system("cc-pvdz");
        let n = sys.obs.nbasis();
        let op = Operator::coulomb();
        let build = |level| {
            PairMajor::build(
                op,
                &sys.obs,
                &sys.dfbs,
                &sys.linv,
                (sys.q_aux_max, 0.0),
                level,
                usize::MAX,
            )
            .unwrap()
            .expect("fits an unbounded budget")
        };
        let plain = std::sync::Arc::new(build(None));
        let c = factor(n, 6);
        let d = c.dot(&c.t()) + &factor(n, n) * 0.01;
        let jk = |t: &PairMajor| {
            let (mut j, mut k) = (Array2::<f64>::zeros((n, n)), Array2::zeros((n, n)));
            t.build_j(&d, &mut j).unwrap();
            t.build_k_occ(&c, &mut k).unwrap();
            (j, k)
        };
        let density_k = |t: &std::sync::Arc<PairMajor>| {
            let mut k = Array2::<f64>::zeros((n, n));
            DfK::from_pair_major(t.clone(), usize::MAX)
                .build(&d, &mut k)
                .unwrap();
            k
        };
        let (j0, k0) = jk(&plain);
        let kd0 = density_k(&plain);
        assert!(!plain.is_rotated());
        for level in [RotationLevel::Rot1, RotationLevel::Ogc] {
            let rot = std::sync::Arc::new(build(Some(level)));
            assert!(rot.is_rotated(), "{level:?}: nothing rotated in cc-pVDZ");
            let (j, k) = jk(&rot);
            let kd = density_k(&rot);
            for (what, got, want) in [("J", &j, &j0), ("K", &k, &k0), ("density K", &kd, &kd0)] {
                assert_close(&format!("{level:?}: {what}"), got, want, 1e-10);
            }
            assert!(k == k.t() && j == j.t(), "{level:?}: J or K not symmetric");
        }
    }

    /// The whole RI-JK RHF on the pair-major route reaches the aux-major
    /// energy, on a segmented basis and on a generally contracted one (the
    /// tensor is then held on rotated shells). Catches a route that is not
    /// taken (the counter), a J or K off by a factor, and a first build (the
    /// factored guess density) that disagrees. Bar 1e-9 Ha: both runs stop at
    /// density_conv 1e-9, whose energy error is second order, and at
    /// integral_thresh 1e-12 no pair of this system is dropped.
    #[test]
    fn an_ri_jk_rhf_reaches_the_aux_major_energy() {
        for name in ["def2-svp", "cc-pvdz"] {
            rhf_matches_the_aux_major_route(name);
        }
    }

    fn rhf_matches_the_aux_major_route(name: &str) {
        use crate::rhf::{solve_rhf, RhfConfig};
        let mol = Molecule::parse_xyz(TWO_WATERS, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled(name).unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ferric_core::parallel::ParallelContext::default();
        let bounds = crate::screening::SchwarzBounds::compute(op, &obs).unwrap();
        let cfg = RhfConfig::default()
            .with_energy_conv(1e-3)
            .with_density_conv(1e-9)
            .with_max_iter(100)
            .with_df_j_aux(AUX)
            .with_df_k_aux(AUX)
            .with_ri_j_precision(crate::df_j::RiJPrecision::F64);
        let run = |pair: bool| {
            let _g = override_pair_layout_for_test(pair);
            let before = pair_major_count_for_test();
            let r = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();
            (r.energy, pair_major_count_for_test() - before)
        };
        let (e_aux, taken_aux) = run(false);
        let (e_pair, taken_pair) = run(true);
        assert_eq!(
            taken_aux, 0,
            "{name}: the aux-major run took the pair route"
        );
        assert!(
            taken_pair >= 1,
            "{name}: the pair-major route was not taken"
        );
        assert!(
            (e_aux - e_pair).abs() <= 1e-9,
            "{name}: pair-major {e_pair} vs aux-major {e_aux}"
        );
    }
}
