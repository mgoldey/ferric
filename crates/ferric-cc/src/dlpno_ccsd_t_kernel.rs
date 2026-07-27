//! DLPNO-CCSD(T) — the `(T)` kernel **evaluated inside** each triple's TNO
//! basis, so truncation changes the COST and not only the energy.
//!
//! [`crate::dlpno_ccsd_t_virtual`] built and proved the substrate: the union
//! PNO basis, its semicanonicalization, the `[nv,nv,nv]` block round trip and a
//! per-triple contribution [`crate::dlpno_ccsd_t_virtual::tno_triple_contribution`]
//! that is exact at `t_cut_tno = 0`. Its own docs state the limitation this
//! module exists to remove: that function consumes a **dense-built** `W̃`/`V`
//! and rotates them in afterwards, so a smaller TNO dimension buys nothing —
//! every GEMM still runs at `nvir`.
//!
//! Here the *inputs* are transformed first and the kernel then runs entirely at
//! dimension `ntno`:
//!
//! ```text
//!   dense path :  ovvv[i] (nv³) ⊗ t2 (nv²)  ->  W (nv³)  ->  Σ W̃V/D  (nv³)
//!   TNO   path :  ovvṽ[i] (n³)  ⊗ t2̃ (n²)   ->  W̃ (n³)   ->  Σ W̃V/D  (n³)
//! ```
//!
//! # What is transformed, and why that is the exact thing to do
//!
//! Write `Q` for the triple's `(nvir × ntno)` semicanonical transform. Every
//! virtual index appearing in the per-triple algebra is projected with the SAME
//! `Q`:
//!
//! ```text
//!   ovvv[i,a,b,d] -> Σ_abd Q_aã Q_bb̃ Q_dd̃ ovvv[i,a,b,d]      (3 virtual axes)
//!   ovoo[i,a,l,j] -> Σ_a   Q_aã            ovoo[i,a,l,j]      (1 virtual axis)
//!   ovov[i,a,j,b] -> Σ_ab  Q_aã Q_bb̃       ovov[i,a,j,b]      (2 virtual axes)
//!   t2[l,m,a,b]   -> Σ_ab  Q_aã Q_bb̃       t2[l,m,a,b]        (2 virtual axes)
//!   t1[l,a]       -> Σ_a   Q_aã            t1[l,a]            (1 virtual axis)
//! ```
//!
//! The `d` and the two `t2` axes are *summed* indices, not free ones. Projecting
//! them is the substantive approximation this module makes and it is the
//! standard DLPNO one: the triple's correlation is expanded in its own domain,
//! so the internal summation runs over that domain too. At `t_cut_tno = 0` it is
//! not an approximation at all — `Q` is square orthogonal, `Σ_d̃ Q_dd̃ Q_d'd̃ =
//! δ_dd'`, and every projected contraction is an identity insertion. That is
//! precisely why the exactness gate below is the load-bearing test.
//!
//! `l`, `m`, `j`, `k` are occupied and are NOT touched: this module is the
//! virtual-side kernel, and the occupied side is
//! [`crate::dlpno_ccsd_t`]'s triple screen.
//!
//! # THE MULTIPLICITY / DIVISOR INVARIANT
//!
//! [`triple_contribution_in_tno`] returns the **unweighted** per-triple sum
//! `Σ_ãb̃c̃ W̃·V/D`. It applies neither the multiplicity `m ∈ {1,3,6}` nor the
//! divisor `3`: those belong to
//! [`crate::dlpno_ccsd_t::screened_triple_energy`], and keeping them there is
//! what stops the spin-orbital `i<j<k` + `/6` convention leaking in (a silent
//! >50% error — see [`crate::ccsd_t_closed_shell`]'s "Combinatorial factor").
//!
//! The six-permutation `W` is built over the SAME `Q` for all six orderings of
//! `(i,j,k)`, because [`TripleTno`] is a function of the unordered multiset
//! (pinned bit-for-bit by `dlpno_ccsd_t_virtual`). A basis that varied with the
//! ordering would break the banded-weight identity silently.
//!
//! # SEMICANONICALIZATION
//!
//! The denominator uses [`TripleTno::eps`] — the eigenvalues of `Qᵀ diag(ε_v) Q`
//! — never the canonical `ε_v` and never the diagonal-only shortcut
//! `f_ãã = Σ_c Q_cã² ε_c`. `W̃` and `V` are honest tensors and rotate; `D` is the
//! diagonal of a Fock-like operator and only stays diagonal because `ε̃` really
//! are the Fock eigenvalues in this basis. The shortcut is measurably wrong
//! (`dlpno_ccsd_t_virtual::tests::stage3_diagonal_only_fock_shortcut_is_measurably_wrong`)
//! and is not reachable from here: [`TripleTno`] only ever exposes the
//! semicanonical composite.
//!
//! # THE COST CLAIM — structural, not a wall clock
//!
//! No timing is reported anywhere in this module, by design (the development box
//! is shared and contested; a wall clock there is not evidence). Cost is instead
//! reported as two quantities that are *pure functions of the dimensions*:
//!
//! * [`TripleCost::working_set_elements`] — the `[n,n,n]` block footprint, `n³`
//!   against the dense `nvir³`;
//! * [`TripleCost::kernel_flops`] — the multiply-add count of the GEMMs the
//!   kernel actually issues, a closed-form polynomial in `(no, n)`.
//!
//! Both are computed by [`triple_cost`] from `ntno` alone and are asserted to
//! decrease strictly with `t_cut_tno` by
//! [`tests::cost_strictly_decreases_with_truncation`].
//!
//! ## And the honest other half of that claim: the transform is not free
//!
//! Projecting `ovvv[i]` costs `O(no · nv³ · n)`-class work — *more* than the
//! `O(nv³ · n)`-class per-triple GEMM it feeds when the triple count is small.
//! [`TripleCost::transform_flops`] counts it separately and
//! [`TripleCost::total_flops`] adds it, so the break-even is visible rather than
//! hidden.
//!
//! Two things follow, and both are measured in the tests rather than asserted:
//!
//! 1. The `ovvv` projection depends only on `(i, Q)`, not on the full triple, so
//!    it is hoisted per-triple and reused across all six permutations of
//!    `(i,j,k)` (three distinct `i` values at most). [`TripleWorkspace`] is that
//!    cache.
//! 2. Even hoisted, at ferric's system sizes the transform DOMINATES.
//!    Even hoisted, at ferric's system sizes the transform DOMINATES.
//!    [`tests::transform_overhead_is_reported_honestly`] prints the measured
//!    table: at `(no=5, nv=19)` the transform is already **1.76×** the kernel at
//!    zero truncation, and truncating to half the virtuals makes it **6.67×**,
//!    because the kernel falls as `n⁴` while the transform's leading `nv³·n`
//!    term falls only as `n`.
//!
//!    So the honest bottom line, in the brief's own words: **at these sizes the
//!    transform cost exceeds the GEMM saving.** The `total_flops` ratio against
//!    a dense sweep is 0.254 at half-truncation on `(5,19)`, not the 0.09 the
//!    kernel count alone would suggest — a real saving, but a much smaller one,
//!    and one that only exists because the `ovvv` projection is hoisted per
//!    distinct occupied index. Whether it beats simply running the dense kernel
//!    is NOT established here and would need a measurement this module refuses
//!    to fake with a wall clock on a contested box.
//!
//! # Exactness contract
//!
//! | Gate | What | Test |
//! |------|------|------|
//! | 1 | per-triple contribution ≡ `dense_triple_contribution` at `t_cut_tno = 0` | [`tests::exactness_matches_dense_triple_contribution`] |
//! | 2 | banded energy ≡ dense band through `screened_triple_energy` | [`tests::exactness_matches_dense_band`] |
//! | 3 | end-to-end ≡ [`crate::ccsd_t_closed_shell`] on a real molecule | [`tests::exactness_matches_ccsd_t_closed_shell_h2o_sto3g`] |

use ferric_core::FerricError;
use ndarray::{Array2, Array3, Array4};

use crate::dlpno_ccsd_t_virtual::TripleTno;

/// The 6 permutations of `(0,1,2)`, in the same fixed order
/// [`crate::ccsd_t_closed_shell`] uses, so the accumulation sequence of `W`
/// matches the dense path term for term.
const PERMS3: [[usize; 3]; 6] =
    [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];

/// Chemist-notation integral blocks and amplitudes in the **canonical** virtual
/// basis — the inputs [`triple_contribution_in_tno`] projects.
///
/// Field for field this is [`crate::ccsd_t_closed_shell`]'s private
/// `TripleIntegrals` plus the amplitudes, gathered into one public struct so a
/// caller can drive the TNO kernel from blocks it already has.
#[derive(Debug, Clone)]
pub struct CanonicalBlocks<'a> {
    /// `(ia|bd)` at `[i,a,b,d]`.
    pub ovvv: &'a Array4<f64>,
    /// `(ia|lj)` at `[i,a,l,j]`.
    pub ovoo: &'a Array4<f64>,
    /// `(ia|jb)` at `[i,a,j,b]`.
    pub ovov: &'a Array4<f64>,
    /// Singles amplitudes `[no,nv]`.
    pub t1: &'a Array2<f64>,
    /// Doubles amplitudes `[no,no,nv,nv]`.
    pub t2: &'a Array4<f64>,
}

impl CanonicalBlocks<'_> {
    /// `(no, nv)` implied by the block shapes, after checking they agree.
    ///
    /// # Errors
    ///
    /// [`FerricError::General`] when the five blocks are not mutually
    /// consistent — a mis-shaped input here would otherwise surface as a
    /// plausible wrong energy rather than a failure.
    pub fn dims(&self) -> Result<(usize, usize), FerricError> {
        let (no, nv) = (self.t1.dim().0, self.t1.dim().1);
        let ok = self.ovvv.dim() == (no, nv, nv, nv)
            && self.ovoo.dim() == (no, nv, no, no)
            && self.ovov.dim() == (no, nv, no, nv)
            && self.t2.dim() == (no, no, nv, nv);
        if !ok {
            return Err(FerricError::General(format!(
                "CanonicalBlocks: inconsistent shapes for (no={no}, nv={nv}) — \
                 ovvv{:?} ovoo{:?} ovov{:?} t2{:?}",
                self.ovvv.dim(),
                self.ovoo.dim(),
                self.ovov.dim(),
                self.t2.dim()
            )));
        }
        Ok((no, nv))
    }
}

/// The canonical blocks **projected into one triple's TNO basis** — the objects
/// the kernel GEMMs actually run on.
///
/// Every virtual axis has been contracted with the triple's `Q`, so every
/// dimension below is `ntno` rather than `nvir`. Built once per triple by
/// [`TripleWorkspace::new`] and reused across all six permutations of
/// `(i,j,k)`, which is what keeps the (dominant) transform cost from being paid
/// six times over.
#[derive(Debug, Clone)]
pub struct TripleWorkspace {
    /// `ntno` for this triple.
    pub n: usize,
    /// `no` — the occupied dimension, untouched by the TNO projection.
    pub no: usize,
    /// `ovvv` projected on `(a,b,d)`, only for the occupied indices this triple
    /// needs: `ovvv_t[slot]` is `[ã,b̃,d̃]` for `occ_used[slot]`.
    ovvv_t: Vec<Array3<f64>>,
    /// The distinct occupied indices of the triple, sorted — the `i` values
    /// `ovvv_t` is indexed by. At most 3 entries.
    occ_used: Vec<usize>,
    /// `ovoo` projected on `a`, for each `occ_used` slot: `[ã,l,j]`.
    ovoo_t: Vec<Array3<f64>>,
    /// `ovov` projected on `(a,b)`, for each `occ_used` slot: `[ã,j,b̃]`.
    ovov_t: Vec<Array3<f64>>,
    /// `t2` projected on `(a,b)`, full occupied range: `[l,m,ã,b̃]`.
    t2_t: Array4<f64>,
    /// `t1` projected on `a`: `[l,ã]`.
    t1_t: Array2<f64>,
}

/// Contract the trailing virtual axis of a flattened block with `Q`.
///
/// `x` is viewed as `(rows, nvir)`, right-multiplied by `Q` `(nvir, n)`.
fn rot_last(x: ndarray::ArrayView2<'_, f64>, q: &Array2<f64>) -> Array2<f64> {
    x.dot(q)
}

impl TripleWorkspace {
    /// Project every block this triple needs into its TNO basis.
    ///
    /// The `ovvv` projection is the expensive one and is done ONLY for the
    /// distinct occupied indices in `{i,j,k}` (1, 2 or 3 of them), which is the
    /// whole reason the workspace exists: the six-permutation `W` below reaches
    /// for `ovvv[i_π]` six times but there are at most three distinct `i_π`.
    ///
    /// # Errors
    ///
    /// [`FerricError::General`] on inconsistent input shapes, a transform whose
    /// row count is not `nvir`, or an out-of-range occupied index.
    pub fn new(blocks: &CanonicalBlocks<'_>, tno: &TripleTno) -> Result<Self, FerricError> {
        let (no, nv) = blocks.dims()?;
        let q = &tno.transform;
        if q.nrows() != nv {
            return Err(FerricError::General(format!(
                "TripleWorkspace: transform is {:?} but the blocks carry nvir = {nv}",
                q.dim()
            )));
        }
        let n = q.ncols();
        let (i, j, k) = tno.ijk;
        if i >= no || j >= no || k >= no {
            return Err(FerricError::General(format!(
                "TripleWorkspace: triple ({i},{j},{k}) out of range for no = {no}"
            )));
        }

        let mut occ_used = vec![i, j, k];
        occ_used.sort_unstable();
        occ_used.dedup();

        let mut ovvv_t = Vec::with_capacity(occ_used.len());
        let mut ovoo_t = Vec::with_capacity(occ_used.len());
        let mut ovov_t = Vec::with_capacity(occ_used.len());
        for &p in &occ_used {
            // --- ovvv[p] : [a,b,d] -> [ã,b̃,d̃], three successive GEMMs with an
            // axis cycle between them (the same idiom as w3_to_tno, but the
            // intermediate dimensions shrink as we go, which is the point). ---
            let g = blocks.ovvv.index_axis(ndarray::Axis(0), p); // [a,b,d]
            // pass 1: contract d.  (a*b, nv) . (nv, n) -> [a,b,d̃]
            let f1 = g.to_shape((nv * nv, nv)).map_err(reshape_err("ovvv[p] (ab,d)"))?;
            let r1 = rot_last(f1.view(), q);
            let r1 = r1.to_shape((nv, nv, n)).map_err(reshape_err("ovvv r1"))?.to_owned();
            // cycle -> [b,d̃,a], contract a
            let c1 = cycle(&r1);
            let f2 = c1.to_shape((nv * n, nv)).map_err(reshape_err("ovvv (bd,a)"))?;
            let r2 = rot_last(f2.view(), q);
            let r2 = r2.to_shape((nv, n, n)).map_err(reshape_err("ovvv r2"))?.to_owned();
            // cycle -> [d̃,ã,b], contract b
            let c2 = cycle(&r2);
            let f3 = c2.to_shape((n * n, nv)).map_err(reshape_err("ovvv (da,b)"))?;
            let r3 = rot_last(f3.view(), q);
            let r3 = r3.to_shape((n, n, n)).map_err(reshape_err("ovvv r3"))?.to_owned();
            // cycle -> [ã,b̃,d̃]
            ovvv_t.push(cycle(&r3));

            // --- ovoo[p] : [a,l,j] -> [ã,l,j]. Contract the leading virtual by
            // transposing into (l*j, a) then rotating. ---
            let h = blocks.ovoo.index_axis(ndarray::Axis(0), p); // [a,l,j]
            let ht = h.permuted_axes([1, 2, 0]); // [l,j,a]
            let hts = ht.as_standard_layout().into_owned();
            let fh = hts.to_shape((no * no, nv)).map_err(reshape_err("ovoo (lj,a)"))?;
            let rh = rot_last(fh.view(), q);
            let rh = rh.to_shape((no, no, n)).map_err(reshape_err("ovoo rot"))?.to_owned();
            // [l,j,ã] -> [ã,l,j]
            ovoo_t.push(rh.view().permuted_axes([2, 0, 1]).as_standard_layout().into_owned());

            // --- ovov[p] : [a,j,b] -> [ã,j,b̃]. ---
            let o = blocks.ovov.index_axis(ndarray::Axis(0), p); // [a,j,b]
            let fo = o.to_shape((nv * no, nv)).map_err(reshape_err("ovov (aj,b)"))?;
            let ro = rot_last(fo.view(), q);
            let ro = ro.to_shape((nv, no, n)).map_err(reshape_err("ovov rot b"))?.to_owned();
            // [a,j,b̃] -> [j,b̃,a], contract a
            let oc = cycle(&ro);
            let fo2 = oc.to_shape((no * n, nv)).map_err(reshape_err("ovov (jb,a)"))?;
            let ro2 = rot_last(fo2.view(), q);
            let ro2 = ro2.to_shape((no, n, n)).map_err(reshape_err("ovov rot a"))?.to_owned();
            // [j,b̃,ã] -> [ã,j,b̃]
            ovov_t.push(ro2.view().permuted_axes([2, 0, 1]).as_standard_layout().into_owned());
        }

        // --- t2 : [l,m,a,b] -> [l,m,ã,b̃]. ---
        let f2 = blocks.t2.to_shape((no * no * nv, nv)).map_err(reshape_err("t2 (lma,b)"))?;
        let r = rot_last(f2.view(), q);
        let r = r.to_shape((no * no, nv, n)).map_err(reshape_err("t2 mid"))?.to_owned();
        // [(lm),a,b̃] -> [(lm),b̃,a]
        let rt = r.view().permuted_axes([0, 2, 1]).as_standard_layout().into_owned();
        let f2b = rt.to_shape((no * no * n, nv)).map_err(reshape_err("t2 (lmb,a)"))?;
        let r2 = rot_last(f2b.view(), q);
        let r2 = r2.to_shape((no, no, n, n)).map_err(reshape_err("t2 out"))?.to_owned();
        // currently [l,m,b̃,ã]; swap the last two axes back to [l,m,ã,b̃]
        let t2_t = r2.view().permuted_axes([0, 1, 3, 2]).as_standard_layout().into_owned();

        // --- t1 : [l,a] -> [l,ã]. ---
        let t1_t = blocks.t1.dot(q);

        Ok(Self { n, no, ovvv_t, occ_used, ovoo_t, ovov_t, t2_t, t1_t })
    }

    /// Slot of occupied index `p` in the per-`i` projected blocks.
    fn slot(&self, p: usize) -> Result<usize, FerricError> {
        self.occ_used.iter().position(|&x| x == p).ok_or_else(|| {
            FerricError::General(format!(
                "TripleWorkspace: occupied index {p} is not one of {:?}",
                self.occ_used
            ))
        })
    }
}

fn reshape_err(what: &'static str) -> impl Fn(ndarray::ShapeError) -> FerricError {
    move |e| FerricError::General(format!("dlpno_ccsd_t_kernel: {what} reshape: {e}"))
}

/// Cycle a 3-tensor's axes: `[0,1,2] -> [1,2,0]`, materialized standard-layout.
fn cycle(x: &Array3<f64>) -> Array3<f64> {
    x.view().permuted_axes([1, 2, 0]).as_standard_layout().into_owned()
}

/// Permute a 3-tensor's axes, materialized standard-layout.
fn permute3(x: &Array3<f64>, axes: [usize; 3]) -> Array3<f64> {
    x.view().permuted_axes(axes).as_standard_layout().into_owned()
}

/// `w0[ã,b̃,c̃] = Σ_d̃ (i ã|b̃ d̃)·t2̃[k,j,c̃,d̃] − Σ_l (i ã|l j)·t2̃[l,k,b̃,c̃]`, the
/// TNO-basis analogue of [`crate::ccsd_t_closed_shell`]'s `raw_w_block`.
///
/// Both terms are BLAS3 GEMMs at dimension `n = ntno`, not `nvir`:
/// * term 1: `(ãb̃, d̃) · (d̃, c̃)` — `n³·n` multiply-adds;
/// * term 2: `(ã, l) · (l, b̃c̃)` — `n·no·n²`.
///
/// This is the function whose cost the whole module is about.
fn raw_w_tno(
    ws: &TripleWorkspace,
    i: usize,
    j: usize,
    k: usize,
) -> Result<Array3<f64>, FerricError> {
    let n = ws.n;
    let no = ws.no;
    let si = ws.slot(i)?;

    // Term 1: (ab, d) . (d, c) -> (ab, c) == [ã,b̃,c̃].
    let g = &ws.ovvv_t[si]; // [ã,b̃,d̃]
    let g2 = g.to_shape((n * n, n)).map_err(reshape_err("tno ovvv (ab,d)"))?;
    let t2_k = ws.t2_t.index_axis(ndarray::Axis(0), k);
    let t2_kj = t2_k.index_axis(ndarray::Axis(0), j); // [c̃,d̃]
    let term1 = g2.dot(&t2_kj.t()); // (ab, c)
    let mut out = term1.to_shape((n, n, n)).map_err(reshape_err("tno term1"))?.to_owned();

    // Term 2: (a, l) . (l, bc) -> (a, bc) == [ã,b̃,c̃].
    let hh = &ws.ovoo_t[si]; // [ã,l,j]
    let h = hh.index_axis(ndarray::Axis(2), j); // [ã,l]
    let t2_bk = ws.t2_t.index_axis(ndarray::Axis(1), k); // [l,b̃,c̃]
    let t2_k2 = t2_bk.to_shape((no, n * n)).map_err(reshape_err("tno t2[:,k]"))?;
    let term2 = h.dot(&t2_k2); // (a, bc)
    let term2 = term2.to_shape((n, n, n)).map_err(reshape_err("tno term2"))?.to_owned();

    out -= &term2;
    Ok(out)
}

/// Fully pair-symmetrized `W[ã,b̃,c̃]` for the triple `(i,j,k)`, in the TNO basis.
///
/// Identical structure to [`crate::ccsd_t_closed_shell`]'s `w_block`: sum over
/// the 6 simultaneous permutations of the pairs `(i,ã) (j,b̃) (k,c̃)`, each
/// permuted raw block having its virtual axes permuted back by `π⁻¹` before
/// accumulation. The same `Q` serves all six — [`TripleTno`] is a function of
/// the unordered multiset, so this is well defined.
fn w_block_tno(
    ws: &TripleWorkspace,
    i: usize,
    j: usize,
    k: usize,
) -> Result<Array3<f64>, FerricError> {
    let occ = [i, j, k];
    let n = ws.n;
    let mut out = Array3::<f64>::zeros((n, n, n));
    for p in PERMS3 {
        let r = raw_w_tno(ws, occ[p[0]], occ[p[1]], occ[p[2]])?;
        let mut inv = [0usize; 3];
        for (t, &pt) in p.iter().enumerate() {
            inv[pt] = t;
        }
        out += &permute3(&r, inv);
    }
    Ok(out)
}

/// `V[ã,b̃,c̃] += (j b̃|k c̃)·t1̃[i,ã] + (i ã|k c̃)·t1̃[j,b̃] + (i ã|j b̃)·t1̃[k,c̃]`,
/// in the TNO basis.
///
/// The `fock[v,o]·t2` term is dropped for the same reason as the dense path:
/// it vanishes for canonical HF orbitals.
fn add_v_singles_tno(
    v: &mut Array3<f64>,
    ws: &TripleWorkspace,
    i: usize,
    j: usize,
    k: usize,
) -> Result<(), FerricError> {
    let n = ws.n;
    let (si, sj, _sk) = (ws.slot(i)?, ws.slot(j)?, ws.slot(k)?);
    // ovov_t[slot] is [ã,j,b̃] for occupied index occ_used[slot].
    let g_jk = ws.ovov_t[sj].index_axis(ndarray::Axis(1), k); // (j b̃|k c̃) -> [b̃,c̃]
    let g_ik = ws.ovov_t[si].index_axis(ndarray::Axis(1), k); // [ã,c̃]
    let g_ij = ws.ovov_t[si].index_axis(ndarray::Axis(1), j); // [ã,b̃]
    for a in 0..n {
        let t1ia = ws.t1_t[[i, a]];
        for b in 0..n {
            let t1jb = ws.t1_t[[j, b]];
            let gij_ab = g_ij[[a, b]];
            for c in 0..n {
                v[[a, b, c]] +=
                    g_jk[[b, c]] * t1ia + g_ik[[a, c]] * t1jb + gij_ab * ws.t1_t[[k, c]];
            }
        }
    }
    Ok(())
}

/// The **unweighted** per-triple `(T)` contribution, with `W̃` and `V` built
/// DIRECTLY in this triple's TNO basis.
///
/// ```text
///   contribution = Σ_ãb̃c̃  W̃[ã,b̃,c̃] · V[ã,b̃,c̃] / D[ã,b̃,c̃]
///   D[ã,b̃,c̃]    = e_ijk − ε̃_ã − ε̃_b̃ − ε̃_c̃          (ε̃ = TripleTno::eps)
/// ```
///
/// This is a drop-in replacement for
/// [`crate::dlpno_ccsd_t_virtual::tno_triple_contribution`] with one decisive
/// difference: that function takes DENSE `[nv,nv,nv]` `W̃`/`V` and rotates them,
/// so its GEMMs are all at `nvir`. This one never forms an `nvir`-dimensional
/// working block at all — every buffer is `[n,n,n]`.
///
/// Returns the raw sum: NO multiplicity weight, NO `/3` divisor. Feed it to
/// [`crate::dlpno_ccsd_t::screened_triple_energy`], which owns both.
///
/// `e_ijk = ε_i + ε_j + ε_k` from the canonical occupied energies (occupieds are
/// untouched by the TNO projection).
///
/// # Errors
///
/// [`FerricError::General`] on shape disagreements, an occupied index outside
/// the triple, or a denominator below `1e-10` in magnitude.
pub fn triple_contribution_in_tno(
    ws: &TripleWorkspace,
    tno: &TripleTno,
    e_ijk: f64,
) -> Result<f64, FerricError> {
    let (i, j, k) = tno.ijk;
    let n = ws.n;
    if tno.eps.len() != n {
        return Err(FerricError::General(format!(
            "triple_contribution_in_tno: {} TNO energies for ntno = {n}",
            tno.eps.len()
        )));
    }

    let w = w_block_tno(ws, i, j, k)?;
    let mut v = w.clone();
    add_v_singles_tno(&mut v, ws, i, j, k)?;

    // W̃ = 4W + W(bca) + W(cab) − 2W(cba) − 2W(acb) − 2W(bac), permutations on
    // the virtual axes (PySCF `r3`). Same term order as the dense path.
    let mut wt = 4.0 * &w;
    wt += &permute3(&w, [1, 2, 0]);
    wt += &permute3(&w, [2, 0, 1]);
    wt.scaled_add(-2.0, &permute3(&w, [2, 1, 0]));
    wt.scaled_add(-2.0, &permute3(&w, [0, 2, 1]));
    wt.scaled_add(-2.0, &permute3(&w, [1, 0, 2]));

    let eps = &tno.eps;
    let mut acc = 0.0f64;
    for a in 0..n {
        for b in 0..n {
            let dab = e_ijk - eps[a] - eps[b];
            for c in 0..n {
                let d = dab - eps[c];
                if d.abs() < 1e-10 {
                    return Err(FerricError::General(format!(
                        "triple_contribution_in_tno: vanishing denominator {d:.3e} at \
                         TNO ({a},{b},{c}) of triple {:?}",
                        tno.ijk
                    )));
                }
                acc += wt[[a, b, c]] * v[[a, b, c]] / d;
            }
        }
    }
    Ok(acc)
}

/// Convenience: build the workspace and evaluate one triple in one call.
///
/// Equivalent to [`TripleWorkspace::new`] followed by
/// [`triple_contribution_in_tno`]. Prefer the two-step form when the same
/// triple is evaluated more than once; this exists so a caller driving
/// [`crate::dlpno_ccsd_t::screened_triple_energy`] can write a one-line closure.
///
/// # Errors
///
/// Propagates both steps' errors.
pub fn triple_contribution(
    blocks: &CanonicalBlocks<'_>,
    tno: &TripleTno,
    e_ijk: f64,
) -> Result<f64, FerricError> {
    let ws = TripleWorkspace::new(blocks, tno)?;
    triple_contribution_in_tno(&ws, tno, e_ijk)
}

// =====================================================================
//  COST MODEL — pure functions of the dimensions. No wall clocks.
// =====================================================================

/// Structural cost of evaluating ONE triple, in elements and multiply-adds.
///
/// Every field is a closed-form polynomial in `(no, nvir, ntno, n_distinct_occ)`
/// — computable without running anything, which is the point: the development
/// box is contested and a wall clock on it is not evidence, but a FLOP count and
/// a working-set size are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TripleCost {
    /// `ntno` this cost was computed at.
    pub ntno: usize,
    /// `nvir` — the dense dimension it is measured against.
    pub nvir: usize,
    /// f64 elements in the largest per-triple working block, `ntno³`.
    ///
    /// The dense kernel's equivalent is `nvir³`
    /// ([`TripleCost::dense_working_set_elements`]).
    pub working_set_elements: usize,
    /// Multiply-adds in the GEMMs [`triple_contribution_in_tno`] issues, i.e.
    /// the part that shrinks with truncation.
    ///
    /// Per raw block: term 1 is `(n²×n)·(n×n)` = `n⁴`, term 2 is `(n×no)·(no×n²)`
    /// = `no·n³`. Six permutations, plus the `n³` contraction of `W̃·V/D`.
    pub kernel_flops: usize,
    /// Multiply-adds spent projecting the canonical blocks into the TNO basis
    /// ([`TripleWorkspace::new`]) — the cost truncation does NOT remove, and the
    /// reason this module refuses to quote `kernel_flops` alone.
    pub transform_flops: usize,
    /// Distinct occupied indices in the triple (1, 2 or 3) — how many times the
    /// `ovvv` projection is paid.
    pub n_distinct_occ: usize,
}

impl TripleCost {
    /// `nvir³` — what the dense kernel's per-triple block costs in elements.
    pub fn dense_working_set_elements(&self) -> usize {
        self.nvir.saturating_pow(3)
    }

    /// `kernel_flops + transform_flops` — the honest total for this path.
    pub fn total_flops(&self) -> usize {
        self.kernel_flops.saturating_add(self.transform_flops)
    }

    /// Fraction of the dense working set this triple occupies, `(n/nv)³`.
    pub fn working_set_ratio(&self) -> f64 {
        if self.nvir == 0 {
            return 1.0;
        }
        self.working_set_elements as f64 / self.dense_working_set_elements().max(1) as f64
    }
}

/// Cost of one triple at TNO dimension `n`, for the dense dimensions
/// `(no, nvir)` and a triple with `n_distinct_occ ∈ {1,2,3}` distinct occupied
/// indices.
///
/// Counted term by term against the code above; the correspondence is checked by
/// [`tests::cost_model_matches_the_dense_formula_at_zero_cut`], which asserts
/// that at `n = nvir` the kernel count equals the dense kernel's own count.
///
/// # Errors
///
/// [`FerricError::General`] when `ntno > nvir` (a TNO basis cannot exceed the
/// space it is drawn from) or `n_distinct_occ` is not in `1..=3`.
pub fn triple_cost(
    no: usize,
    nvir: usize,
    ntno: usize,
    n_distinct_occ: usize,
) -> Result<TripleCost, FerricError> {
    if ntno > nvir {
        return Err(FerricError::General(format!(
            "triple_cost: ntno = {ntno} exceeds nvir = {nvir}"
        )));
    }
    if !(1..=3).contains(&n_distinct_occ) {
        return Err(FerricError::General(format!(
            "triple_cost: n_distinct_occ = {n_distinct_occ} must be 1, 2 or 3"
        )));
    }
    let n = ntno;
    let nv = nvir;

    // --- kernel: 6 raw blocks + the W̃·V/D reduction ---
    // term 1: (n*n, n) . (n, n)      -> n^4
    // term 2: (n, no) . (no, n*n)    -> no * n^3
    let per_raw = n.saturating_pow(4).saturating_add(no.saturating_mul(n.saturating_pow(3)));
    let kernel = per_raw.saturating_mul(6).saturating_add(n.saturating_pow(3));

    // --- transform (TripleWorkspace::new) ---
    // ovvv[p]: nv^3*n + nv^2*n^2 + nv*n^3, per distinct occupied index
    let ovvv_one = nv
        .saturating_pow(3)
        .saturating_mul(n)
        .saturating_add(nv.saturating_pow(2).saturating_mul(n.saturating_pow(2)))
        .saturating_add(nv.saturating_mul(n.saturating_pow(3)));
    // ovoo[p]: no^2 * nv * n         (one virtual axis)
    let ovoo_one = no.saturating_pow(2).saturating_mul(nv).saturating_mul(n);
    // ovov[p]: nv*no*nv*n + no*n*nv*n (two virtual axes)
    let ovov_one = nv
        .saturating_mul(no)
        .saturating_mul(nv)
        .saturating_mul(n)
        .saturating_add(no.saturating_mul(n).saturating_mul(nv).saturating_mul(n));
    let per_i = ovvv_one.saturating_add(ovoo_one).saturating_add(ovov_one);
    // t2: no^2*nv^2*n + no^2*nv*n^2 ; t1: no*nv*n
    let t2_cost = no
        .saturating_pow(2)
        .saturating_mul(nv.saturating_pow(2))
        .saturating_mul(n)
        .saturating_add(no.saturating_pow(2).saturating_mul(nv).saturating_mul(n.saturating_pow(2)));
    let t1_cost = no.saturating_mul(nv).saturating_mul(n);
    let transform = per_i
        .saturating_mul(n_distinct_occ)
        .saturating_add(t2_cost)
        .saturating_add(t1_cost);

    Ok(TripleCost {
        ntno: n,
        nvir: nv,
        working_set_elements: n.saturating_pow(3),
        kernel_flops: kernel,
        transform_flops: transform,
        n_distinct_occ,
    })
}

/// Cost of an entire [`crate::dlpno_ccsd_t_virtual::TripleTnoBasis`] sweep:
/// summed over its triples, against the dense sweep over the same triple list.
///
/// Returns `(tno, dense)` [`SweepCost`]s so a caller can quote the ratio without
/// hand-summing.
///
/// # Errors
///
/// Propagates [`triple_cost`].
pub fn sweep_cost(
    no: usize,
    basis: &crate::dlpno_ccsd_t_virtual::TripleTnoBasis,
) -> Result<(SweepCost, SweepCost), FerricError> {
    let nv = basis.nvir;
    let mut tno = SweepCost::default();
    let mut dense = SweepCost::default();
    for t in &basis.triples {
        let (i, j, k) = t.ijk;
        let mut occ = [i, j, k];
        occ.sort_unstable();
        let nd = { let mut v = occ.to_vec(); v.dedup(); v.len() };

        let c = triple_cost(no, nv, t.ntno(), nd)?;
        tno.add(&c);
        let d = triple_cost(no, nv, nv, nd)?;
        dense.add(&d);
    }
    Ok((tno, dense))
}

/// Summed [`TripleCost`] over a triple list.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SweepCost {
    /// Σ `ntno³` — total per-triple block elements touched.
    pub working_set_elements: usize,
    /// Σ kernel multiply-adds.
    pub kernel_flops: usize,
    /// Σ transform multiply-adds.
    pub transform_flops: usize,
    /// Largest single-triple working block, the memory-ceiling driver.
    pub max_working_set_elements: usize,
}

impl SweepCost {
    fn add(&mut self, c: &TripleCost) {
        self.working_set_elements =
            self.working_set_elements.saturating_add(c.working_set_elements);
        self.kernel_flops = self.kernel_flops.saturating_add(c.kernel_flops);
        self.transform_flops = self.transform_flops.saturating_add(c.transform_flops);
        self.max_working_set_elements =
            self.max_working_set_elements.max(c.working_set_elements);
    }

    /// `kernel_flops + transform_flops`.
    pub fn total_flops(&self) -> usize {
        self.kernel_flops.saturating_add(self.transform_flops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dlpno_ccsd_t::{complete_triple_domains, screened_triple_energy};
    use crate::dlpno_ccsd_t_virtual::{dense_triple_contribution, TripleTnoBasis};

    // ------------------------------------------------------------------
    // Deterministic toy system, same shape as dlpno_ccsd_t_virtual's.
    // Small on purpose: every claim here is about EXACTNESS or about a
    // COUNT, neither of which needs a large system.
    // ------------------------------------------------------------------

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    }

    fn eps_vir(nvir: usize) -> Vec<f64> {
        (0..nvir).map(|a| 0.5 + 0.13 * a as f64).collect()
    }

    fn eps_occ(nocc: usize) -> Vec<f64> {
        (0..nocc).map(|i| -1.0 - 0.11 * i as f64).collect()
    }

    fn ovov_block(nocc: usize, nvir: usize) -> Array4<f64> {
        let n = nocc * nvir;
        let mut s = 0x2545F4914F6CDD1Du64;
        let mut m = Array2::<f64>::zeros((n, n));
        for p in 0..n {
            for q in p..n {
                let v = lcg(&mut s);
                m[(p, q)] = v;
                m[(q, p)] = v;
            }
        }
        Array4::from_shape_fn((nocc, nvir, nocc, nvir), |(i, a, j, b)| {
            m[(i * nvir + a, j * nvir + b)]
        })
    }

    fn mp2_t2(ovov: &Array4<f64>, eo: &[f64], ev: &[f64]) -> Array4<f64> {
        let (nocc, nvir, _, _) = ovov.dim();
        Array4::from_shape_fn((nocc, nocc, nvir, nvir), |(i, j, a, b)| {
            ovov[[i, a, j, b]] / (eo[i] + eo[j] - ev[a] - ev[b])
        })
    }

    fn line_centers(nocc: usize, spacing: f64) -> Array2<f64> {
        Array2::from_shape_fn((nocc, 3), |(i, ax)| if ax == 0 { i as f64 * spacing } else { 0.0 })
    }

    /// Deterministic pseudo-random blocks of every shape the kernel consumes.
    struct Toy {
        no: usize,
        nv: usize,
        ovvv: Array4<f64>,
        ovoo: Array4<f64>,
        ovov: Array4<f64>,
        t1: Array2<f64>,
        t2: Array4<f64>,
        eo: Vec<f64>,
        ev: Vec<f64>,
    }

    impl Toy {
        fn new(no: usize, nv: usize) -> Self {
            let (eo, ev) = (eps_occ(no), eps_vir(nv));
            let ovov = ovov_block(no, nv);
            let t2 = mp2_t2(&ovov, &eo, &ev);
            let mut s = 0x9E3779B97F4A7C15u64;
            let ovvv = Array4::from_shape_fn((no, nv, nv, nv), |_| lcg(&mut s));
            let ovoo = Array4::from_shape_fn((no, nv, no, no), |_| lcg(&mut s));
            let t1 = Array2::from_shape_fn((no, nv), |_| 0.05 * lcg(&mut s));
            Self { no, nv, ovvv, ovoo, ovov, t1, t2, eo, ev }
        }

        fn blocks(&self) -> CanonicalBlocks<'_> {
            CanonicalBlocks {
                ovvv: &self.ovvv,
                ovoo: &self.ovoo,
                ovov: &self.ovov,
                t1: &self.t1,
                t2: &self.t2,
            }
        }

        fn tno_basis(&self, t_cut: f64) -> TripleTnoBasis {
            let d = complete_triple_domains(&line_centers(self.no, 1.5)).unwrap();
            TripleTnoBasis::build(&d, self.nv, &self.ev, t_cut, |i, j| {
                Array2::from_shape_fn((self.nv, self.nv), |(a, b)| self.t2[[i, j, a, b]])
            })
            .unwrap()
        }

        /// The DENSE `W̃`/`V` for a triple, written out here independently of
        /// this module's TNO path — the oracle. Term for term
        /// `ccsd_t_closed_shell`'s `w_block` + `add_v_singles` + `r3`.
        fn dense_wt_v(&self, i: usize, j: usize, k: usize) -> (Array3<f64>, Array3<f64>) {
            let (no, nv) = (self.no, self.nv);
            let raw = |i: usize, j: usize, k: usize| -> Array3<f64> {
                let g = self.ovvv.index_axis(ndarray::Axis(0), i);
                let g2 = g.to_shape((nv * nv, nv)).unwrap();
                let t2_k = self.t2.index_axis(ndarray::Axis(0), k);
                let t2_kj = t2_k.index_axis(ndarray::Axis(0), j);
                let mut out =
                    g2.dot(&t2_kj.t()).to_shape((nv, nv, nv)).unwrap().to_owned();
                let ovoo_j = self.ovoo.index_axis(ndarray::Axis(3), j);
                let h = ovoo_j.index_axis(ndarray::Axis(0), i);
                let t2_bk = self.t2.index_axis(ndarray::Axis(1), k);
                let t2_k2 = t2_bk.to_shape((no, nv * nv)).unwrap();
                let term2 = h.dot(&t2_k2).to_shape((nv, nv, nv)).unwrap().to_owned();
                out -= &term2;
                out
            };
            let occ = [i, j, k];
            let mut w = Array3::<f64>::zeros((nv, nv, nv));
            for p in PERMS3 {
                let r = raw(occ[p[0]], occ[p[1]], occ[p[2]]);
                let mut inv = [0usize; 3];
                for (t, &pt) in p.iter().enumerate() {
                    inv[pt] = t;
                }
                w += &permute3(&r, inv);
            }
            let mut v = w.clone();
            let ovov_k = self.ovov.index_axis(ndarray::Axis(2), k);
            let ovov_j = self.ovov.index_axis(ndarray::Axis(2), j);
            let g_jk = ovov_k.index_axis(ndarray::Axis(0), j);
            let g_ik = ovov_k.index_axis(ndarray::Axis(0), i);
            let g_ij = ovov_j.index_axis(ndarray::Axis(0), i);
            for a in 0..nv {
                let t1ia = self.t1[[i, a]];
                for b in 0..nv {
                    let t1jb = self.t1[[j, b]];
                    let gij_ab = g_ij[[a, b]];
                    for c in 0..nv {
                        v[[a, b, c]] +=
                            g_jk[[b, c]] * t1ia + g_ik[[a, c]] * t1jb + gij_ab * self.t1[[k, c]];
                    }
                }
            }
            let mut wt = 4.0 * &w;
            wt += &permute3(&w, [1, 2, 0]);
            wt += &permute3(&w, [2, 0, 1]);
            wt.scaled_add(-2.0, &permute3(&w, [2, 1, 0]));
            wt.scaled_add(-2.0, &permute3(&w, [0, 2, 1]));
            wt.scaled_add(-2.0, &permute3(&w, [1, 0, 2]));
            (wt, v)
        }
    }

    // ================= GATE 1: per-triple exactness ========================

    /// **THE EXACTNESS GATE.** At `t_cut_tno = 0` the kernel built entirely
    /// inside the TNO basis must reproduce the dense per-triple contribution, on
    /// EVERY triple and in every multiplicity class.
    ///
    /// The transforms are square orthogonal there, so projecting the *summed*
    /// indices (`d` in term 1, the two `t2` virtual axes, the `ovov` axes) is an
    /// identity insertion rather than an approximation. If this fails the whole
    /// module is wrong: a truncated answer would be a truncation of the wrong
    /// quantity.
    #[test]
    fn exactness_matches_dense_triple_contribution() {
        let toy = Toy::new(4, 5);
        let basis = toy.tno_basis(0.0);
        assert!(basis.is_complete(), "t_cut_tno = 0 must keep every virtual");
        let blocks = toy.blocks();

        let mut worst_abs = 0.0f64;
        let mut worst_scaled = 0.0f64;
        let mut scale = 0.0f64;
        let (mut n1, mut n3, mut n6) = (0usize, 0usize, 0usize);
        let mut per_triple = Vec::new();
        for t in &basis.triples {
            let (i, j, k) = t.ijk;
            match (i == k, i == j || j == k) {
                (true, _) => n1 += 1,
                (false, true) => n3 += 1,
                _ => n6 += 1,
            }
            let e_ijk = toy.eo[i] + toy.eo[j] + toy.eo[k];
            let (wt, v) = toy.dense_wt_v(i, j, k);
            let dense = dense_triple_contribution(&wt, &v, &toy.ev, e_ijk).unwrap();
            let got = triple_contribution(&blocks, t, e_ijk).unwrap();
            worst_abs = worst_abs.max((got - dense).abs());
            scale = scale.max(dense.abs());
            per_triple.push((t.ijk, dense, got));
        }
        // Normalize by the LARGEST contribution in the band, not by each
        // triple's own value. A per-triple relative error is the wrong metric
        // here: individual contributions pass through zero by cancellation
        // (the `W̃·V/D` sum has both signs), so dividing by a near-zero
        // reference reports a huge "relative error" for a deviation that is
        // pure f64 rounding and is irrelevant to the band. See
        // `near_zero_triples_make_per_triple_relative_error_meaningless`, which
        // MEASURES that this is what is happening rather than asserting it.
        for &(_, dense, got) in &per_triple {
            worst_scaled = worst_scaled.max((got - dense).abs() / scale);
        }
        eprintln!(
            "gate 1: max |E_TNO-kernel - E_dense| = {worst_abs:.3e}, max deviation \
             relative to the band scale {scale:.3e} = {worst_scaled:.3e}; \
             classes m=1:{n1} m=3:{n3} m=6:{n6}"
        );
        assert!(scale > 1e-3, "contributions are ~zero — the gate would be vacuous");
        assert!(n1 > 0 && n3 > 0 && n6 > 0, "all three multiplicity classes must be exercised");
        assert!(
            worst_scaled < 1e-12,
            "TNO-basis kernel must reproduce the dense contribution: {worst_scaled:.3e} \
             of the band scale (abs {worst_abs:.3e})"
        );
    }

    /// **Why gate 1 is normalized by the band scale and not per triple.**
    ///
    /// This test exists because the per-triple relative error LOOKED like a
    /// failure (1.1e1 on the nocc=4/nvir=5 fixture) when the absolute deviation
    /// was 1.9e-14. It measures the cause instead of assuming it: some triples'
    /// dense contributions are themselves ~1e-15, because `Σ W̃·V/D` is a sum of
    /// both signs that can cancel to essentially nothing on synthetic blocks.
    /// Dividing a rounding-level deviation by a rounding-level reference gives a
    /// meaningless O(1) "relative error".
    ///
    /// The assertion is the diagnosis: there must EXIST a triple whose dense
    /// contribution is negligible against the band scale. If that ever stops
    /// being true, gate 1's normalization should be revisited rather than
    /// trusted.
    #[test]
    fn near_zero_triples_make_per_triple_relative_error_meaningless() {
        let toy = Toy::new(4, 5);
        let basis = toy.tno_basis(0.0);
        let blocks = toy.blocks();
        let mut vals = Vec::new();
        for t in &basis.triples {
            let (i, j, k) = t.ijk;
            let e_ijk = toy.eo[i] + toy.eo[j] + toy.eo[k];
            let (wt, v) = toy.dense_wt_v(i, j, k);
            let dense = dense_triple_contribution(&wt, &v, &toy.ev, e_ijk).unwrap();
            let got = triple_contribution(&blocks, t, e_ijk).unwrap();
            vals.push((t.ijk, dense, (got - dense).abs()));
        }
        let scale = vals.iter().map(|v| v.1.abs()).fold(0.0f64, f64::max);
        let (worst_ijk, worst_dense, worst_dev) = vals
            .iter()
            .max_by(|a, b| {
                (a.2 / a.1.abs().max(1e-300))
                    .partial_cmp(&(b.2 / b.1.abs().max(1e-300)))
                    .unwrap()
            })
            .copied()
            .unwrap();
        eprintln!(
            "gate-1 metric: band scale {scale:.3e}; the worst per-triple RELATIVE \
             error is on triple {worst_ijk:?} where E_dense = {worst_dense:.3e} \
             (|E_dense|/scale = {:.3e}) and the absolute deviation is {worst_dev:.3e}",
            worst_dense.abs() / scale
        );
        assert!(
            worst_dense.abs() / scale < 1e-12,
            "the worst per-triple relative error is NOT on a near-zero triple \
             (|E_dense|/scale = {:.3e}) — gate 1's band-scale normalization needs \
             re-examining, it may be masking a real error",
            worst_dense.abs() / scale
        );
        assert!(
            worst_dev / scale < 1e-12,
            "and its absolute deviation must still be rounding-level: {:.3e} of scale",
            worst_dev / scale
        );
    }

    /// The same gate on differently-shaped fixtures — more virtuals than
    /// occupieds, and more occupieds than virtuals, are different index regimes
    /// and a shape bug can hide in one and not the other.
    ///
    /// `(no=1, nv=4)` is deliberately EXCLUDED and covered by
    /// [`edge_single_occupied_is_shape_only`] instead: its only triple is
    /// `(0,0,0)`, whose contribution cancels to ~1e-16 on this synthetic
    /// fixture, so it cannot discriminate a working kernel from a broken one.
    /// Running it here would be a vacuous pass dressed as a gate.
    #[test]
    fn exactness_holds_on_other_shapes() {
        for &(no, nv) in &[(3usize, 6usize), (5, 4), (4, 7)] {
            let toy = Toy::new(no, nv);
            let basis = toy.tno_basis(0.0);
            let blocks = toy.blocks();
            let mut devs = Vec::new();
            let mut scale = 0.0f64;
            for t in &basis.triples {
                let (i, j, k) = t.ijk;
                let e_ijk = toy.eo[i] + toy.eo[j] + toy.eo[k];
                let (wt, v) = toy.dense_wt_v(i, j, k);
                let dense = dense_triple_contribution(&wt, &v, &toy.ev, e_ijk).unwrap();
                let got = triple_contribution(&blocks, t, e_ijk).unwrap();
                scale = scale.max(dense.abs());
                devs.push((got - dense).abs());
            }
            let worst = devs.iter().fold(0.0f64, |a, &b| a.max(b)) / scale;
            eprintln!(
                "gate 1 (no={no}, nv={nv}): max deviation / band scale ({scale:.3e}) = {worst:.3e}"
            );
            assert!(scale > 1e-6, "no={no} nv={nv}: contributions ~zero, gate vacuous");
            assert!(worst < 1e-12, "no={no} nv={nv}: {worst:.3e} of the band scale");
        }
    }

    /// The single-occupied system — one triple `(0,0,0)`, one constituent pair —
    /// must RUN and produce the same number as the dense path, but this is a
    /// SHAPE test and explicitly not a correctness gate: the contribution
    /// cancels to ~1e-16 on the synthetic fixture, so it cannot distinguish a
    /// working kernel from a stub. Same caveat
    /// [`crate::ccsd_t_closed_shell`]'s `closed_shell_t_h2_sto3g_is_small`
    /// carries, for the same reason.
    #[test]
    fn edge_single_occupied_is_shape_only() {
        let toy = Toy::new(1, 4);
        let basis = toy.tno_basis(0.0);
        assert_eq!(basis.triples.len(), 1);
        let t = &basis.triples[0];
        assert_eq!(t.ijk, (0, 0, 0));
        let ws = TripleWorkspace::new(&toy.blocks(), t).unwrap();
        assert_eq!(ws.ovvv_t.len(), 1, "one distinct occupied index");
        let e_ijk = 3.0 * toy.eo[0];
        let (wt, v) = toy.dense_wt_v(0, 0, 0);
        let dense = dense_triple_contribution(&wt, &v, &toy.ev, e_ijk).unwrap();
        let got = triple_contribution(&toy.blocks(), t, e_ijk).unwrap();
        eprintln!(
            "edge (no=1, nv=4): dense = {dense:.3e}, TNO kernel = {got:.3e} \
             (both ~zero — shape check only, NOT a correctness gate)"
        );
        assert!((got - dense).abs() < 1e-12);
    }

    /// The kernel must agree with
    /// [`crate::dlpno_ccsd_t_virtual::tno_triple_contribution`] — the
    /// rotate-after-the-fact path — at zero cut. Both are exact there, so this
    /// is a cross-check of two independent routes to the same number, and it
    /// pins that this module's *input* projection is the same operation as that
    /// module's *output* rotation.
    #[test]
    fn agrees_with_the_rotate_afterwards_path() {
        use crate::dlpno_ccsd_t_virtual::tno_triple_contribution;
        let toy = Toy::new(4, 5);
        let basis = toy.tno_basis(0.0);
        let blocks = toy.blocks();
        let mut devs = Vec::new();
        let mut scale = 0.0f64;
        for t in &basis.triples {
            let (i, j, k) = t.ijk;
            let e_ijk = toy.eo[i] + toy.eo[j] + toy.eo[k];
            let (wt, v) = toy.dense_wt_v(i, j, k);
            let rotate_after = tno_triple_contribution(&wt, &v, t, toy.nv, e_ijk).unwrap();
            let build_inside = triple_contribution(&blocks, t, e_ijk).unwrap();
            scale = scale.max(rotate_after.abs());
            devs.push((rotate_after - build_inside).abs());
        }
        // Band-scale normalized, for the same reason gate 1 is — see
        // `near_zero_triples_make_per_triple_relative_error_meaningless`.
        let worst = devs.iter().fold(0.0f64, |a, &b| a.max(b)) / scale;
        eprintln!(
            "gate 1: max |rotate-after - build-inside| / band scale ({scale:.3e}) = {worst:.3e}"
        );
        assert!(worst < 1e-12);
    }

    // ================= GATE 2: the banded energy ===========================

    /// **THE MULTIPLICITY / DIVISOR INVARIANT, end to end.** Driving
    /// [`screened_triple_energy`] with this kernel at `t_cut_tno = 0` must
    /// reproduce the dense weighted band with its `m ∈ {1,3,6}` weights and `/3`
    /// divisor.
    ///
    /// The kernel returns the UNWEIGHTED sum and applies neither the weight nor
    /// the divisor; `screened_triple_energy` owns both. If either leaked into
    /// this module the band would come out wrong by a factor near 3 or worse —
    /// and it would still look like a plausible triples energy.
    #[test]
    fn exactness_matches_dense_band() {
        let toy = Toy::new(4, 5);
        let basis = toy.tno_basis(0.0);
        let blocks = toy.blocks();
        let domains = complete_triple_domains(&line_centers(toy.no, 1.5)).unwrap();

        let e_tno = screened_triple_energy(&domains, |i, j, k| {
            let t = basis.triples.iter().find(|t| t.ijk == (i, j, k)).unwrap();
            triple_contribution(&blocks, t, toy.eo[i] + toy.eo[j] + toy.eo[k])
        })
        .unwrap();

        let e_dense = screened_triple_energy(&domains, |i, j, k| {
            let (wt, v) = toy.dense_wt_v(i, j, k);
            dense_triple_contribution(&wt, &v, &toy.ev, toy.eo[i] + toy.eo[j] + toy.eo[k])
        })
        .unwrap();

        let rel = (e_tno - e_dense).abs() / e_dense.abs();
        eprintln!("gate 2: banded E = {e_tno:.12} (TNO kernel) vs {e_dense:.12} (dense), rel {rel:.3e}");
        assert!(e_dense.abs() > 1e-3, "banded energy is ~zero — the gate is vacuous");
        assert!(rel < 1e-10, "TNO-kernel band must reproduce the dense band: rel {rel:.3e}");
    }

    /// A guard against the divisor trap being satisfied by accident: applying
    /// the spin-orbital `/6` + `i<j<k` convention instead must give a
    /// MEASURABLY different number, so gate 2 is discriminating.
    #[test]
    fn the_spin_orbital_convention_would_be_measurably_wrong() {
        let toy = Toy::new(4, 5);
        let basis = toy.tno_basis(0.0);
        let blocks = toy.blocks();
        let domains = complete_triple_domains(&line_centers(toy.no, 1.5)).unwrap();

        let right = screened_triple_energy(&domains, |i, j, k| {
            let t = basis.triples.iter().find(|t| t.ijk == (i, j, k)).unwrap();
            triple_contribution(&blocks, t, toy.eo[i] + toy.eo[j] + toy.eo[k])
        })
        .unwrap();

        // The WRONG convention: i<j<k only, weight 6, divisor 6.
        let mut wrong = 0.0f64;
        for i in 0..toy.no {
            for j in (i + 1)..toy.no {
                for k in (j + 1)..toy.no {
                    let t = basis.triples.iter().find(|t| t.ijk == (i, j, k)).unwrap();
                    wrong += 6.0
                        * triple_contribution(&blocks, t, toy.eo[i] + toy.eo[j] + toy.eo[k])
                            .unwrap();
                }
            }
        }
        wrong /= 6.0;
        let rel = (right - wrong).abs() / right.abs();
        eprintln!("gate 2: closed-shell band {right:.12} vs spin-orbital convention {wrong:.12} (rel {rel:.3e})");
        assert!(
            rel > 0.1,
            "the two conventions agree to {rel:.3e} — gate 2 would not discriminate"
        );
    }

    // ================= GATE 3: end-to-end on a real molecule ===============

    /// **THE END-TO-END GATE.** On real water/STO-3G integrals and converged
    /// CCSD amplitudes, driving this kernel through
    /// [`screened_triple_energy`] at `t_cut_tno = 0` must reproduce
    /// [`crate::ccsd_t_closed_shell`]'s energy.
    ///
    /// The toy gates use synthetic blocks with no physical symmetry; this one
    /// uses the actual RI-derived chemist blocks, the actual `t1`/`t2`, and the
    /// actual orbital energies — including near-degenerate virtuals, which is
    /// where a semicanonicalization bug would show.
    #[test]
    fn exactness_matches_ccsd_t_closed_shell_h2o_sto3g() {
        let (e_ref, e_tno, retention) = real_h2o("sto-3g", 0.0);
        let diff = (e_tno - e_ref).abs();
        eprintln!(
            "gate 3: H2O/STO-3G (T) — ccsd_t_closed_shell = {e_ref:.12}, TNO kernel \
             (t_cut = 0, retention {retention:.3}) = {e_tno:.12}, diff = {diff:.3e}"
        );
        assert!(e_ref < -1e-6, "reference (T) = {e_ref} is not a plausible triples energy");
        assert_eq!(retention, 1.0, "t_cut_tno = 0 must keep every virtual");
        assert!(
            diff / e_ref.abs() < 1e-9,
            "TNO kernel must reproduce ccsd_t_closed_shell: {e_tno:.12} vs {e_ref:.12}"
        );
    }

    /// Gate 3 on a **second basis**, water/6-31G — 13 basis functions and 8
    /// virtuals against STO-3G's 7 and 2, so the `[n,n,n]` blocks and the
    /// `ovvv` projection run at a genuinely different shape, and the (T) energy
    /// is two orders of magnitude larger (a real number to be exact against
    /// rather than a near-zero one).
    #[test]
    fn exactness_matches_ccsd_t_closed_shell_h2o_631g() {
        let (e_ref, e_tno, retention) = real_h2o("6-31g", 0.0);
        let diff = (e_tno - e_ref).abs();
        eprintln!(
            "gate 3: H2O/6-31G (T) — ccsd_t_closed_shell = {e_ref:.12}, TNO kernel \
             (t_cut = 0, retention {retention:.3}) = {e_tno:.12}, diff = {diff:.3e}"
        );
        assert!(e_ref < -1e-5, "reference (T) = {e_ref} is not a plausible triples energy");
        assert_eq!(retention, 1.0);
        assert!(
            diff / e_ref.abs() < 1e-9,
            "TNO kernel must reproduce ccsd_t_closed_shell: {e_tno:.12} vs {e_ref:.12}"
        );
    }

    /// Truncation on the REAL system: a threshold sweep, reported rather than
    /// asserted at one value.
    ///
    /// The first attempt at this test asserted that `t_cut_tno = 1e-4` must move
    /// the H2O/STO-3G energy. It does not — measured `dE = -1.4e-20` at
    /// retention 0.986. That is not an inert knob; it is ferric's known
    /// small-molecule result showing up again: STO-3G water has 7 virtuals with
    /// no redundancy, so the first TNOs a threshold discards carry essentially
    /// no amplitude weight, and the energy only moves once the threshold starts
    /// cutting into occupied directions.
    ///
    /// So this test sweeps and REPORTS, and asserts only the two things that
    /// must hold for the module's claims to mean anything:
    ///
    /// * somewhere in the sweep the energy does move (the knob is not inert);
    /// * where it moves, the retention has genuinely dropped.
    ///
    /// Measured sweep (H2O/STO-3G, 5 occupied / 2 virtual after the SCF's
    /// negative-eigenvalue occupied count — see the printed retention):
    /// the energy is flat to ~1e-20 until the threshold bites, then moves.
    #[test]
    fn truncation_on_the_real_system_is_swept_not_assumed() {
        let (e_ref, e0, r0) = real_h2o("sto-3g", 0.0);
        assert_eq!(r0, 1.0);
        eprintln!("H2O/STO-3G (T): ccsd_t_closed_shell {e_ref:.12} | t_cut=0 {e0:.12} (ret {r0:.3})");

        let mut moved = None;
        for &cut in &[1e-6f64, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1] {
            let (_, e, r) = real_h2o("sto-3g", cut);
            let de = e - e0;
            eprintln!("  t_cut {cut:8.0e}: retention {r:.4}  E = {e:.12}  dE = {de:.3e}");
            if de.abs() > 1e-12 && moved.is_none() {
                moved = Some((cut, r, de));
            }
        }
        let (cut, r, de) = moved.expect(
            "no threshold in the sweep moved the (T) energy — the TNO knob would be \
             inert on this system and the cost table would be measuring nothing",
        );
        eprintln!(
            "H2O/STO-3G: the TNO threshold first bites at t_cut = {cut:.0e} \
             (retention {r:.4}, dE = {de:.3e})"
        );
        assert!(
            r < 1.0,
            "the energy moved at t_cut = {cut:.0e} while retention was still 1.0 — \
             that would mean the change came from something other than truncation"
        );
    }

    /// Real water run in basis `obs_name`: returns `(E_(T) from
    /// `ccsd_t_closed_shell`, E_(T) from this kernel at `t_cut`, TNO virtual
    /// retention)`. Small bases only — this runs in a unit test on a shared box.
    fn real_h2o(obs_name: &str, t_cut: f64) -> (f64, f64, f64) {
        use crate::ccsd_closed_shell::ccsd_closed_shell;
        use crate::ccsd_t_closed_shell::ccsd_t_closed_shell;
        use crate::{CcConfig, CcResult};
        use ferric_core::basis;
        use ferric_core::mol::Molecule;
        use ferric_core::parallel::ParallelContext;
        use ferric_integrals::basis_bridge::PreparedBasis;
        use ferric_integrals::operator::Operator;
        use ferric_mp2::mo_transform::{
            transform_3center_oo, transform_3center_ov, transform_3center_vv,
        };
        use ferric_mp2::rimp2::cholesky_inverse_sqrt;
        use ferric_mp2::spinorbital::build_b;
        use ferric_scf::rhf::{solve_rhf, RhfConfig};
        use ferric_scf::screening::SchwarzBounds;
        use ferric_tensors::{einsum, Axis};

        const H2O: &str = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
        let mol = Molecule::parse_xyz(H2O, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ctx,
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig { energy_conv: 1e-11, ..Default::default() },
        )
        .unwrap();
        let cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-10, ..Default::default() };
        let cc: CcResult = ccsd_closed_shell(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        let e_ref = ccsd_t_closed_shell(&mol, &obs, &dfbs, op, &rhf, &cc, &cfg).unwrap();

        // Rebuild the SAME chemist blocks ccsd_t_closed_shell builds internally.
        let eps = rhf.eps_r();
        let c = rhf.mos_r();
        let nbas = obs.nbasis();
        let nocc_total = eps.iter().filter(|&&e| e < 0.0).count();
        let no = nocc_total;
        let nv = nbas - nocc_total;
        let c_occ = c.slice(ndarray::s![.., ..nocc_total]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();
        let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let v_inv_sqrt = cholesky_inverse_sqrt(&v2c).unwrap();
        let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        use Axis::{O, V};
        let b_ov = build_b(&transform_3center_ov(&eri3_ao, &c_occ, &c_vir), &v_inv_sqrt, O, V);
        let b_oo = build_b(&transform_3center_oo(&eri3_ao, &c_occ), &v_inv_sqrt, O, O);
        let b_vv = build_b(&transform_3center_vv(&eri3_ao, &c_vir), &v_inv_sqrt, V, V);
        drop(eri3_ao);
        let ovvv: Array4<f64> =
            einsum!("Pia,Pbd->iabd", &b_ov, &b_vv).into_dimensionality().unwrap();
        let ovoo: Array4<f64> =
            einsum!("Pia,Plj->ialj", &b_ov, &b_oo).into_dimensionality().unwrap();
        let ovov: Array4<f64> =
            einsum!("Pia,Pjb->iajb", &b_ov, &b_ov).into_dimensionality().unwrap();

        let t1 = cc.t1.as_ref().unwrap().clone();
        let t2 = cc.t2.clone();
        let blocks = CanonicalBlocks { ovvv: &ovvv, ovoo: &ovoo, ovov: &ovov, t1: &t1, t2: &t2 };
        let eo: Vec<f64> = (0..no).map(|i| eps[i]).collect();
        let ev: Vec<f64> = (0..nv).map(|a| eps[nocc_total + a]).collect();

        // Boys-center stand-in: the occupied orbitals' charge centroids are not
        // needed here (the triple screen is OFF — complete domains), so any
        // consistent center array does. Use a line, as the toy fixtures do.
        let domains = complete_triple_domains(&line_centers(no, 1.5)).unwrap();
        let basis = TripleTnoBasis::build(&domains, nv, &ev, t_cut, |i, j| {
            Array2::from_shape_fn((nv, nv), |(a, b)| t2[[i, j, a, b]])
        })
        .unwrap();

        let e_tno = screened_triple_energy(&domains, |i, j, k| {
            let t = basis.triples.iter().find(|t| t.ijk == (i, j, k)).unwrap();
            triple_contribution(&blocks, t, eo[i] + eo[j] + eo[k])
        })
        .unwrap();

        (e_ref, e_tno, basis.virtual_retention())
    }

    // ================= COST: structural, no wall clocks ====================

    /// **THE COST CLAIM.** Both the per-triple working set and the kernel FLOP
    /// count must decrease STRICTLY as `t_cut_tno` rises — that is what
    /// "truncation changes the cost, not just the energy" means, and it is
    /// asserted on counts, never on a timing (the box is shared; a wall clock
    /// there is not evidence).
    ///
    /// The measured table on the (nocc=4, nvir=6) fixture is printed by this
    /// test; representative numbers:
    ///
    /// ```text
    ///   t_cut   retention  Σ working set (×dense)  Σ kernel FLOPs (×dense)
    ///   0.0       1.000     4320  (1.000)          263520  (1.000)
    ///   1e-3      0.975     4047  (0.937)          244617  (0.928)
    ///   1e-2      0.950     3834  (0.887)          230838  (0.876)
    ///   3e-2      0.925     3675  (0.851)          221385  (0.840)
    /// ```
    ///
    /// Note the working set and the FLOPs fall together and slightly FASTER
    /// than the retention (0.851 and 0.840 against a retention of 0.925),
    /// because both are superlinear in `ntno` while retention is linear.
    #[test]
    fn cost_strictly_decreases_with_truncation() {
        let toy = Toy::new(4, 6);
        let cuts = [0.0f64, 1e-3, 1e-2, 3e-2];
        let mut rows = Vec::new();
        for &cut in &cuts {
            let basis = toy.tno_basis(cut);
            let (tno, dense) = sweep_cost(toy.no, &basis).unwrap();
            eprintln!(
                "cost: t_cut {cut:9.0e}  retention {:.3}  max ntno {:2}  \
                 Σ working set {:8} (dense {:8}, {:.3}×)  Σ kernel FLOPs {:10} \
                 (dense {:10}, {:.3}×)  Σ transform FLOPs {:10}",
                basis.virtual_retention(),
                basis.max_ntno(),
                tno.working_set_elements,
                dense.working_set_elements,
                tno.working_set_elements as f64 / dense.working_set_elements as f64,
                tno.kernel_flops,
                dense.kernel_flops,
                tno.kernel_flops as f64 / dense.kernel_flops as f64,
                tno.transform_flops,
            );
            rows.push((cut, basis.virtual_retention(), tno));
        }

        // Strict monotone decrease in BOTH reported cost quantities.
        for w in rows.windows(2) {
            let (c0, r0, a) = &w[0];
            let (c1, r1, b) = &w[1];
            assert!(r1 <= r0, "retention rose from {r0} at {c0:.0e} to {r1} at {c1:.0e}");
            assert!(
                b.working_set_elements < a.working_set_elements,
                "working set did not strictly decrease from t_cut {c0:.0e} ({}) to \
                 {c1:.0e} ({})",
                a.working_set_elements,
                b.working_set_elements
            );
            assert!(
                b.kernel_flops < a.kernel_flops,
                "kernel FLOPs did not strictly decrease from t_cut {c0:.0e} ({}) to \
                 {c1:.0e} ({})",
                a.kernel_flops,
                b.kernel_flops
            );
        }
        // And the untruncated end must report NO saving at all.
        let basis0 = toy.tno_basis(0.0);
        let (t0, d0) = sweep_cost(toy.no, &basis0).unwrap();
        assert_eq!(t0, d0, "t_cut_tno = 0 must cost exactly the dense sweep");
    }

    /// The cost model must be the code's own arithmetic, not a wish: at
    /// `ntno = nvir` the kernel count must equal the dense kernel's count for
    /// the same triple, and the working set must be exactly `nvir³`.
    #[test]
    fn cost_model_matches_the_dense_formula_at_zero_cut() {
        for &(no, nv) in &[(4usize, 6usize), (5, 19), (2, 3)] {
            for nd in 1..=3usize {
                let c = triple_cost(no, nv, nv, nd).unwrap();
                assert_eq!(c.working_set_elements, nv.pow(3));
                assert_eq!(c.dense_working_set_elements(), nv.pow(3));
                assert_eq!(c.working_set_ratio(), 1.0);
                // 6 raw blocks of (nv^4 + no*nv^3) plus the nv^3 reduction.
                let want = 6 * (nv.pow(4) + no * nv.pow(3)) + nv.pow(3);
                assert_eq!(c.kernel_flops, want, "no={no} nv={nv} nd={nd}");
            }
        }
        // ntno > nvir is impossible and must error, not produce a number.
        assert!(triple_cost(4, 5, 6, 3).is_err());
        assert!(triple_cost(4, 5, 5, 0).is_err());
        assert!(triple_cost(4, 5, 5, 4).is_err());
    }

    /// Working set and kernel FLOPs must be strictly monotone in `ntno` at fixed
    /// dimensions — pinned directly on the model, independent of whether any
    /// particular threshold happens to truncate.
    #[test]
    fn cost_model_is_monotone_in_ntno() {
        let (no, nv) = (5usize, 19usize);
        let mut prev: Option<TripleCost> = None;
        for n in 1..=nv {
            let c = triple_cost(no, nv, n, 3).unwrap();
            if let Some(p) = prev {
                assert!(c.working_set_elements > p.working_set_elements);
                assert!(c.kernel_flops > p.kernel_flops);
                assert!(c.transform_flops > p.transform_flops);
            }
            prev = Some(c);
        }
    }

    /// **The honest other half of the cost claim.** The transform is NOT free,
    /// and at ferric's sizes it dominates: report the measured ratio rather than
    /// quoting `kernel_flops` alone.
    ///
    /// Representative output (per triple, `n_distinct_occ = 3`):
    ///
    /// ```text
    ///   no  nv  ntno   kernel FLOPs   transform FLOPs   transform/kernel
    ///    5  19    19         994555           1750489           1.76
    ///    5  19    10          91000            606670           6.67
    ///    5  19     5           7625            235885          30.94
    ///   20  50    50       52625000         174300000           3.31
    ///   20  50    25        4234375          61056250          14.42
    ///   20  50    13         437203          26102050          59.70
    /// ```
    ///
    /// The kernel shrinks as `n⁴`; the transform only as `n` (its leading term
    /// is `nv³·n`, and `nv` is fixed). So truncation makes the transform a
    /// LARGER fraction of the total, and the net `total_flops` saving is far
    /// below the `kernel_flops` saving. This is stated here, in the tests, so it
    /// cannot be quietly dropped from a summary.
    #[test]
    fn transform_overhead_is_reported_honestly() {
        eprintln!("  no  nv  ntno   kernel FLOPs   transform FLOPs   transform/kernel   total/dense-total");
        let mut ratios = Vec::new();
        for &(no, nv) in &[(5usize, 19usize), (20, 50)] {
            let dense = triple_cost(no, nv, nv, 3).unwrap();
            for &frac in &[1.0f64, 0.5, 0.25] {
                let n = ((nv as f64 * frac).round() as usize).max(1);
                let c = triple_cost(no, nv, n, 3).unwrap();
                let r = c.transform_flops as f64 / c.kernel_flops as f64;
                eprintln!(
                    "  {no:2}  {nv:2}  {n:4}   {:12}   {:15}   {r:16.2}   {:17.3}",
                    c.kernel_flops,
                    c.transform_flops,
                    c.total_flops() as f64 / dense.total_flops() as f64
                );
                ratios.push((nv, n, r));
            }
        }
        // The finding, pinned: at every size tested the transform EXCEEDS the
        // kernel, and the ratio gets WORSE under truncation.
        for &(nv, n, r) in &ratios {
            assert!(
                r > 1.0,
                "transform/kernel = {r:.3} at nv={nv}, ntno={n} — if this ever drops \
                 below 1 the module doc's honesty caveat is stale and must be updated"
            );
        }
        // Monotonicity of the pathology: half the TNOs, worse ratio.
        assert!(ratios[1].2 > ratios[0].2 && ratios[2].2 > ratios[1].2);
    }

    /// [`sweep_cost`] must agree with a hand-summed loop over the same triples —
    /// the aggregation is where an off-by-one in the distinct-occupied count
    /// would hide.
    #[test]
    fn sweep_cost_is_the_sum_of_its_triples() {
        let toy = Toy::new(4, 6);
        let basis = toy.tno_basis(1e-2);
        let (tno, _) = sweep_cost(toy.no, &basis).unwrap();

        let mut want_ws = 0usize;
        let mut want_k = 0usize;
        let mut max_ws = 0usize;
        for t in &basis.triples {
            let (i, j, k) = t.ijk;
            let nd = if i == k { 1 } else if i == j || j == k { 2 } else { 3 };
            let c = triple_cost(toy.no, toy.nv, t.ntno(), nd).unwrap();
            want_ws += c.working_set_elements;
            want_k += c.kernel_flops;
            max_ws = max_ws.max(c.working_set_elements);
        }
        assert_eq!(tno.working_set_elements, want_ws);
        assert_eq!(tno.kernel_flops, want_k);
        assert_eq!(tno.max_working_set_elements, max_ws);
    }

    /// The working block the kernel ALLOCATES must really be `[n,n,n]`, not a
    /// dense block that was rotated — otherwise `working_set_elements` is a
    /// fiction. Checked by inspecting the intermediate `W` directly.
    #[test]
    fn the_kernel_never_allocates_a_dense_block() {
        let toy = Toy::new(4, 6);
        let basis = toy.tno_basis(3e-2);
        let blocks = toy.blocks();
        let mut n_small = 0usize;
        for t in &basis.triples {
            let ws = TripleWorkspace::new(&blocks, t).unwrap();
            let (i, j, k) = t.ijk;
            let w = w_block_tno(&ws, i, j, k).unwrap();
            assert_eq!(
                w.dim(),
                (t.ntno(), t.ntno(), t.ntno()),
                "triple {:?}: W is {:?}, not [ntno]³",
                t.ijk,
                w.dim()
            );
            // The projected ovvv is the largest transformed buffer.
            for g in &ws.ovvv_t {
                assert_eq!(g.dim(), (t.ntno(), t.ntno(), t.ntno()));
            }
            if t.ntno() < toy.nv {
                n_small += 1;
            }
        }
        assert!(n_small > 0, "test premise: something must have truncated");
        eprintln!("cost: {n_small}/{} triples ran on a strictly sub-dense block", basis.triples.len());
    }

    // ================= input validation ====================================

    /// Malformed inputs are caller bugs and must error, not produce a plausible
    /// wrong energy.
    #[test]
    fn invalid_inputs_are_rejected() {
        let toy = Toy::new(3, 4);
        let basis = toy.tno_basis(0.0);
        let t = &basis.triples[0];

        // Inconsistent block shapes.
        let bad_ovvv = Array4::<f64>::zeros((toy.no, toy.nv, toy.nv, toy.nv + 1));
        let bad = CanonicalBlocks {
            ovvv: &bad_ovvv,
            ovoo: &toy.ovoo,
            ovov: &toy.ovov,
            t1: &toy.t1,
            t2: &toy.t2,
        };
        assert!(bad.dims().is_err());
        assert!(TripleWorkspace::new(&bad, t).is_err());

        // A transform whose row count is not nvir.
        let mut wrong = t.clone();
        wrong.transform = Array2::<f64>::zeros((toy.nv + 1, 2));
        wrong.eps = vec![0.0, 1.0];
        assert!(TripleWorkspace::new(&toy.blocks(), &wrong).is_err());

        // eps of the wrong length for ntno.
        let ws = TripleWorkspace::new(&toy.blocks(), t).unwrap();
        let mut short = t.clone();
        short.eps.pop();
        assert!(triple_contribution_in_tno(&ws, &short, -3.0).is_err());

        // A vanishing denominator must error, not divide through.
        assert!(triple_contribution_in_tno(&ws, t, 3.0 * t.eps[0]).is_err());
    }

    /// A triple whose occupied index exceeds the block dimensions must error.
    #[test]
    fn out_of_range_triple_is_rejected() {
        let toy = Toy::new(3, 4);
        let basis = toy.tno_basis(0.0);
        let mut t = basis.triples[0].clone();
        t.ijk = (0, 1, 99);
        assert!(TripleWorkspace::new(&toy.blocks(), &t).is_err());
    }

    /// The workspace must project `ovvv` once per DISTINCT occupied index, not
    /// once per permutation — the hoist that keeps the (dominant) transform from
    /// being paid six times.
    #[test]
    fn ovvv_is_projected_once_per_distinct_occupied_index() {
        let toy = Toy::new(4, 5);
        let basis = toy.tno_basis(0.0);
        let blocks = toy.blocks();
        for t in &basis.triples {
            let (i, j, k) = t.ijk;
            let mut occ = vec![i, j, k];
            occ.sort_unstable();
            occ.dedup();
            let ws = TripleWorkspace::new(&blocks, t).unwrap();
            assert_eq!(
                ws.ovvv_t.len(),
                occ.len(),
                "triple {:?}: {} ovvv projections for {} distinct occupieds",
                t.ijk,
                ws.ovvv_t.len(),
                occ.len()
            );
            assert_eq!(ws.occ_used, occ);
        }
    }
}
