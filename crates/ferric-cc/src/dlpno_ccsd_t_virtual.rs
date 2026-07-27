//! DLPNO-CCSD(T) — the **per-triple TNO virtual basis**, the half
//! [`crate::dlpno_ccsd_t`] explicitly defers.
//!
//! [`crate::dlpno_ccsd_t`] screens *which occupied triples* `(i,j,k)` are
//! evaluated. This module is the other axis: *which virtuals* a retained triple
//! is evaluated in. It is the triples analogue of
//! [`crate::dlpno_ccsd_virtual`], and the two are deliberately structured the
//! same way so they can be read side by side.
//!
//! # What a TNO is
//!
//! A pair `(i,j)` gets a PNO basis from its own pair density. A *triple* has no
//! single pair density — it couples three pairs at once — so the standard
//! construction (Riplinger/Neese) gives the triple ONE shared virtual basis:
//!
//! ```text
//!   TNO(i,j,k) = orthonormal basis for  span{ PNO(i,j) ∪ PNO(i,k) ∪ PNO(j,k) }
//! ```
//!
//! followed by **semicanonicalization**. Three steps, in this order:
//!
//! 1. **Union.** Form `M^{ijk} = Σ_{(p,q) ⊂ {i,j,k}} P^{pq} (P^{pq})ᵀ`, a sum of
//!    orthogonal projectors, and take the range. This is the technique
//!    [`ferric_rpa::dlpno_rpa`] already uses for its per-orbital unions and it
//!    is chosen for a numerical reason, not a stylistic one: a sum of orthogonal
//!    projectors has every nonzero eigenvalue `≥ 1` and every out-of-span
//!    eigenvalue exactly `0`, so the rank test sits in an **O(1) gap** rather
//!    than resolving the fuzzy singular-value tail a QR of the wide
//!    concatenation `[P^{ij} | P^{ik} | P^{jk}]` would leave.
//! 2. **Orthogonalize.** Automatic — the eigenvectors of a symmetric `M` are
//!    orthonormal by construction.
//! 3. **Semicanonicalize.** See below. NOT optional.
//!
//! # SEMICANONICALIZATION IS MANDATORY
//!
//! TNOs diagonalize a sum of pair-density projectors, which has nothing to do
//! with the virtual Fock matrix. After rotating into the union basis, `F_vir` is
//! **not** diagonal, and the `(T)` energy denominator
//!
//! ```text
//!   D[a,b,c] = ε_i + ε_j + ε_k − ε_a − ε_b − ε_c
//! ```
//!
//! is only meaningful in a Fock-diagonal basis. [`TripleTnoBasis::build`]
//! therefore forms `F^{ijk} = Q ᵀ diag(ε_v) Q`, **re-diagonalizes it** with
//! [`eigh_dc`], and stores the composite `Q̃ = Q·U` together with the resulting
//! TNO orbital energies. Callers never see a non-semicanonical basis.
//!
//! Taking only the diagonal `f_ãã = Σ_c Q_cã² ε_c` is WRONG and fails
//! **silently** — the energy stays plausible. That is not hypothetical: it broke
//! DLPNO-MP2's exactness contract by 0.117 Ha before an exactness test caught
//! it. [`tests::stage1_semicanonical_fock_is_diagonal`] pins the fix and
//! [`tests::stage1_naive_diagonal_fock_would_be_wrong`] pins its premise by
//! MEASURING the off-diagonal magnitude in the raw TNO basis, so the shortcut is
//! demonstrably not an approximation *here* either.
//!
//! Note the union makes this **worse**, not better, than the pair case: a union
//! of three differently-oriented PNO subspaces is further from any Fock
//! eigenbasis than a single pair's PNOs are.
//!
//! # THE MULTIPLICITY / DIVISOR INVARIANT
//!
//! [`crate::ccsd_t_closed_shell`] bands over `i<=j<=k` with weights
//! `m ∈ {1,3,6}` and divisor **3** — NOT the spin-orbital `i<j<k` + `/6`. That
//! identity holds only because every banded triple stands for `m` orderings that
//! the unrestricted sum would evaluate *identically*.
//!
//! A TNO basis that depended on the ORDER of `(i,j,k)` would break this
//! silently: the `m` orderings would be evaluated in `m` different virtual
//! bases, the banded weighted sum would stop equalling the unrestricted sum, and
//! the `/3` divisor would become wrong with no symptom other than a wrong
//! number.
//!
//! The construction here is a function of the **unordered multiset** `{i,j,k}`
//! by design:
//!
//! * the three constituent pairs `{(i,j), (i,k), (j,k)}` are collected as
//!   sorted, deduplicated index pairs, so permuting `(i,j,k)` permutes the
//!   *summands* of `M^{ijk}` and matrix addition is commutative;
//! * `M` is accumulated in a canonical (sorted) pair order regardless of the
//!   caller's argument order, so even the floating-point summation order is
//!   permutation-independent — the resulting basis is **bit-identical** across
//!   all 6 orderings, not merely equal to tolerance.
//!
//! [`tests::stage1_tno_basis_is_permutation_invariant`] pins that over all 6
//! orderings of every triple, on bits.
//!
//! # Exactness contract
//!
//! At `t_cut_tno = 0` every constituent PNO transform is square orthogonal, so
//! every projector `P Pᵀ` is the identity, `M = 3·I` (or `1·I`/`2·I` for
//! repeated-index triples with fewer distinct pairs) — full rank, and `Q̃` is a
//! square orthogonal `nvir × nvir` rotation.
//!
//! Under such a rotation the `(T)` per-triple contribution
//! `Σ_abc W̃[a,b,c] V[a,b,c] / D[a,b,c]` is invariant, but **only** because the
//! denominator is diagonal in the *semicanonical* basis: `D` is not a tensor
//! that rotates, it is the diagonal of a Fock-like operator, and the identity
//! `Σ W̃V/D` holds in the new basis exactly when the new `ε̃` really are the Fock
//! eigenvalues there. This is the concrete reason step 3 is load-bearing rather
//! than cosmetic, and [`tests::stage3_tno_triple_energy_matches_dense`] is what
//! demonstrates it.
//!
//! | Stage | What | Exactness test |
//! |-------|------|----------------|
//! | 1 | [`TripleTnoBasis`] — union + semicanonical transform | [`tests::stage1_transforms_are_square_orthogonal_at_zero_cut`] |
//! | 2 | [`w3_to_tno`] / [`w3_from_tno`] — `[nv,nv,nv]` block round trip | [`tests::stage2_w3_round_trip_is_exact_at_zero_cut`] |
//! | 3 | [`tno_triple_contribution`] — the per-triple `Σ W̃V/D` | [`tests::stage3_tno_triple_energy_matches_dense`] |
//!
//! # What is NOT claimed
//!
//! * **No timing, no speedup, no cost model.** Truncating here changes the
//!   *evaluation*; it does not make [`crate::ccsd_t_closed_shell`]'s kernel
//!   cheaper, because that kernel still builds its `W` blocks densely. What is
//!   reported is COUNTS ([`TripleTnoBasis::virtual_retention`],
//!   [`TripleTnoBasis::block_elements`]) and ENERGIES.
//! * **The `(T)` kernel is not rewritten in the TNO basis.** Doing that means
//!   re-deriving [`crate::ccsd_t_closed_shell`]'s two `raw_w_block` GEMMs, the
//!   six-permutation `W`, the `r3` pattern and the `V` singles terms in a basis
//!   that differs per triple, with cross-triple overlap matrices wherever a
//!   contraction couples two triples. That is a much larger change whose failure
//!   mode is a plausible-but-wrong energy. What is here is the tested substrate
//!   that change needs, and [`tno_triple_contribution`] consumes a
//!   *dense-built* `W`/`V` — honest, and exact, but not yet the cheap path.
//! * **No claim that this compresses on ferric's systems.** ferric has a
//!   MEASURED negative result for virtual truncation at small N (the OSV sweep:
//!   100% retention at accurate thresholds, 48–76 mHa at loose ones). A union
//!   can only ever be **at least as large** as its largest constituent pair, so
//!   a per-pair PNO retention is never a valid proxy for a triple's dimension;
//!   [`TripleTnoBasis::virtual_retention`] is the number to quote.
//!
//!   How much *worse* than the per-pair numbers is threshold-dependent, and
//!   [`tests::stage1_union_is_at_least_as_large_as_its_pairs`] measures it
//!   rather than asserting the analogy with [`ferric_rpa::dlpno_rpa`]'s
//!   re-inflating per-orbital unions. On the toy fixture (nocc=4, nvir=6) the
//!   three constituent PNO subspaces are still **nested** at `t_cut = 1e-3` — no
//!   triple's union exceeds its largest pair — and re-inflation only switches on
//!   near `1e-2`, where 12/20 unions become strictly larger. Both regimes are
//!   pinned so neither gets quoted as the general case.

use ferric_core::linalg::{eigh_dc, Uplo};
use ferric_core::FerricError;
use ferric_mp2::local_pno::{build_pno_transforms, PnoTransforms};
use ferric_mp2::pair_domains::complete_pair_domains;
use ndarray::{Array2, Array3};

use crate::dlpno_ccsd_t::TripleDomains;

/// Tolerance on the union-projector eigenvalues deciding which directions lie in
/// the span of a triple's three PNO subspaces.
///
/// `M^{ijk} = Σ P^{pq} (P^{pq})ᵀ` is a sum of orthogonal projectors: a direction
/// in at least one subspace has eigenvalue `≥ 1`, one orthogonal to all of them
/// has eigenvalue exactly `0`. The gap is O(1), not O(ε) — this threshold sits
/// deep inside it and is **not** a tunable knob. Identical reasoning (and value)
/// to `ferric_rpa::dlpno_rpa`'s `UNION_RANK_TOL`.
const UNION_RANK_TOL: f64 = 1e-8;

/// The semicanonical TNO virtual basis of a single occupied triple `(i,j,k)`.
#[derive(Debug, Clone)]
pub struct TripleTno {
    /// The occupied triple, spatial indices, stored **sorted** `i <= j <= k`
    /// regardless of the order it was requested in.
    ///
    /// Storing the sorted form is part of the permutation-invariance contract:
    /// a consumer that keys on this field cannot end up with two different
    /// bases for the same multiset.
    pub ijk: (usize, usize, usize),
    /// Semicanonical transform `Q̃ = Q·U`, `(nvir × ntno)`, orthonormal columns.
    ///
    /// `Q` is the union basis (eigenvectors of the projector sum) and `U`
    /// diagonalizes the virtual Fock matrix *within* that union. Only the
    /// composite is exposed, so a caller cannot accidentally hold a
    /// non-semicanonical basis.
    pub transform: Array2<f64>,
    /// Virtual-orbital energies in this triple's semicanonical TNO basis,
    /// ascending — the eigenvalues of `Q ᵀ diag(ε_v) Q`.
    ///
    /// At `t_cut_tno = 0` these are the canonical `ε_v` up to ordering, because
    /// a square `Q` makes the rediagonalization recover the original spectrum.
    /// These are the ONLY legitimate source for a `(T)` denominator in this
    /// basis.
    pub eps: Vec<f64>,
    /// The distinct constituent pairs `{(i,j), (i,k), (j,k)}` that were unioned,
    /// sorted and deduplicated. One entry for `i=j=k`, two for a two-equal
    /// triple, three for an all-distinct one.
    pub source_pairs: Vec<(usize, usize)>,
    /// PNO dimension of each entry of `source_pairs`, same order — the
    /// per-pair numbers the union is built from, kept so the re-inflation of
    /// the union relative to its pairs is measurable rather than assumed.
    pub source_pair_dims: Vec<usize>,
}

impl TripleTno {
    /// Number of TNOs retained for this triple.
    pub fn ntno(&self) -> usize {
        self.transform.ncols()
    }
}

/// Per-triple semicanonical TNO bases for every retained triple.
#[derive(Debug, Clone)]
pub struct TripleTnoBasis {
    /// One entry per triple in [`TripleDomains::triples`], in the same order.
    pub triples: Vec<TripleTno>,
    /// Full canonical virtual count the transforms map *from*.
    pub nvir: usize,
    /// PNO occupation threshold the constituent pair PNOs were built at.
    pub t_cut_tno: f64,
}

impl TripleTnoBasis {
    /// Build the semicanonical TNO basis of every retained triple.
    ///
    /// `t2_pair(i,j)` returns the `(nvir × nvir)` first-order amplitude block
    /// `T^{ij}_{ab}` defining the pair density — the same closure contract as
    /// [`build_pno_transforms`] and [`crate::dlpno_ccsd_virtual::PairPnoBasis::build`].
    /// `eps_vir` are the canonical virtual orbital energies (length `nvir`).
    ///
    /// PNOs are built for **all** `i <= j` pairs of the system rather than only
    /// the ones a pair screen retained, because a triple's union needs all three
    /// of its constituent pairs and the triple screen
    /// ([`crate::dlpno_ccsd_t::triple_is_retained`]) already guarantees a
    /// retained triple's pairs would themselves have been retained. Taking the
    /// pair list from the triple list instead would make the basis depend on the
    /// pair cutoff in a way the exactness contract does not control.
    ///
    /// # Errors
    ///
    /// [`FerricError::General`] when `eps_vir.len() != nvir`, when a triple's
    /// index exceeds `domains.nocc`, and propagates any error from the PNO
    /// construction or either eigensolve, with the offending triple named.
    pub fn build<F>(
        domains: &TripleDomains,
        nvir: usize,
        eps_vir: &[f64],
        t_cut_tno: f64,
        t2_pair: F,
    ) -> Result<Self, FerricError>
    where
        F: FnMut(usize, usize) -> Array2<f64>,
    {
        if eps_vir.len() != nvir {
            return Err(FerricError::General(format!(
                "TripleTnoBasis::build: eps_vir has {} entries, expected nvir = {nvir}",
                eps_vir.len()
            )));
        }
        for &(i, j, k) in &domains.triples {
            if i >= domains.nocc || j >= domains.nocc || k >= domains.nocc {
                return Err(FerricError::General(format!(
                    "TripleTnoBasis::build: triple ({i},{j},{k}) out of range for \
                     nocc = {}",
                    domains.nocc
                )));
            }
        }

        // Per-pair PNOs over the COMPLETE i <= j pair list (see the doc above).
        let pair_domains = complete_pair_domains(&domains.centers)?;
        let pnos: PnoTransforms =
            build_pno_transforms(&pair_domains, nvir, t_cut_tno, t2_pair)?;

        let mut triples = Vec::with_capacity(domains.triples.len());
        for &(i, j, k) in &domains.triples {
            triples.push(build_one_tno(&pnos, nvir, eps_vir, i, j, k)?);
        }
        Ok(Self { triples, nvir, t_cut_tno })
    }

    /// True when nothing was truncated: every triple kept all `nvir` virtuals.
    ///
    /// The exactness precondition. At `t_cut_tno = 0` this must hold, which
    /// [`tests::stage1_transforms_are_square_orthogonal_at_zero_cut`] pins.
    pub fn is_complete(&self) -> bool {
        self.triples.iter().all(|t| t.ntno() == self.nvir)
    }

    /// Retained TNOs summed over triples, divided by `n_triples · nvir`.
    ///
    /// 1.0 means no compression at all. Report this, never a constituent pair's
    /// retention — a union is at least as large as any of its parts, so quoting
    /// the per-pair number for a triple would understate the dimension.
    pub fn virtual_retention(&self) -> f64 {
        if self.triples.is_empty() || self.nvir == 0 {
            return 1.0;
        }
        let kept: usize = self.triples.iter().map(TripleTno::ntno).sum();
        kept as f64 / (self.triples.len() * self.nvir) as f64
    }

    /// Largest TNO dimension over all triples — the working-set driver, since
    /// the `(T)` kernel allocates `[ntno]³` blocks.
    pub fn max_ntno(&self) -> usize {
        self.triples.iter().map(TripleTno::ntno).max().unwrap_or(0)
    }

    /// `(Σ_triples ntno³, n_triples · nvir³)` — TNO versus dense element counts
    /// for the per-triple `[nv,nv,nv]` working blocks.
    ///
    /// A COUNT, not a timing. See the module's "what is NOT claimed".
    pub fn block_elements(&self) -> (usize, usize) {
        let tno: usize = self.triples.iter().map(|t| t.ntno().pow(3)).sum();
        (tno, self.triples.len() * nvir_cubed(self.nvir))
    }
}

fn nvir_cubed(nvir: usize) -> usize {
    nvir * nvir * nvir
}

/// The distinct constituent pairs of an occupied triple, as sorted deduplicated
/// index pairs in ascending order.
///
/// **This function is the multiplicity invariant.** It is a function of the
/// unordered multiset `{i,j,k}` only: each pair is sorted internally, the list
/// is then sorted and deduplicated, so all 6 orderings of a triple produce the
/// identical `Vec` — same entries in the same order, which makes even the
/// floating-point accumulation of `M` below permutation-independent.
///
/// Counts: 1 pair for `i=j=k`, 2 for a two-equal triple, 3 for all-distinct.
fn constituent_pairs(i: usize, j: usize, k: usize) -> Vec<(usize, usize)> {
    let ord = |a: usize, b: usize| if a <= b { (a, b) } else { (b, a) };
    let mut p = vec![ord(i, j), ord(i, k), ord(j, k)];
    p.sort_unstable();
    p.dedup();
    p
}

/// Build one triple's semicanonical TNO basis. See [`TripleTnoBasis::build`].
fn build_one_tno(
    pnos: &PnoTransforms,
    nvir: usize,
    eps_vir: &[f64],
    i: usize,
    j: usize,
    k: usize,
) -> Result<TripleTno, FerricError> {
    let pairs = constituent_pairs(i, j, k);

    // --- STEP 1: UNION. M = Σ P Pᵀ, a sum of orthogonal projectors, summed in
    // the canonical sorted pair order so the result is bit-identical under any
    // permutation of (i,j,k). ---
    let mut m = Array2::<f64>::zeros((nvir, nvir));
    let mut source_pair_dims = Vec::with_capacity(pairs.len());
    for &(p, q) in &pairs {
        let entry = pnos
            .pairs
            .iter()
            .find(|e| e.ij == (p, q))
            .ok_or_else(|| {
                FerricError::General(format!(
                    "TNO for triple ({i},{j},{k}): no PNO basis for constituent \
                     pair ({p},{q})"
                ))
            })?;
        source_pair_dims.push(entry.transform.ncols());
        m = m + entry.transform.dot(&entry.transform.t());
    }

    let (eigs, vecs) = eigh_dc(&m, Uplo::Upper).map_err(|e| {
        FerricError::General(format!("TNO union eigh failed for triple ({i},{j},{k}): {e}"))
    })?;
    let keep: Vec<usize> = (0..nvir).filter(|&c| eigs[c] > UNION_RANK_TOL).collect();
    if keep.is_empty() {
        return Err(FerricError::General(format!(
            "TNO union subspace for triple ({i},{j},{k}) came out empty (largest \
             projector eigenvalue {:.3e}); impossible for a sum of orthogonal \
             projectors over {} pairs",
            eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            pairs.len()
        )));
    }
    let mut q = Array2::<f64>::zeros((nvir, keep.len()));
    for (slot, &c) in keep.iter().enumerate() {
        for a in 0..nvir {
            q[(a, slot)] = vecs[(a, c)];
        }
    }

    // --- STEP 3: SEMICANONICALIZE (step 2, orthogonality, is automatic — the
    // eigenvectors of a symmetric M are orthonormal). NOT optional: the (T)
    // denominator D = e_ijk - eps_a - eps_b - eps_c is only meaningful in a
    // Fock-diagonal basis, and the union basis is not one. Taking the diagonal
    // of F alone is the silent-failure mode that cost DLPNO-MP2 0.117 Ha. ---
    let ntno = q.ncols();
    let mut f_tno = Array2::<f64>::zeros((ntno, ntno));
    for a in 0..ntno {
        for b in 0..ntno {
            f_tno[(a, b)] = (0..nvir).map(|c| q[(c, a)] * q[(c, b)] * eps_vir[c]).sum();
        }
    }
    let (eps, u) = eigh_dc(&f_tno, Uplo::Upper).map_err(|e| {
        FerricError::General(format!(
            "TNO semicanonicalization failed for triple ({i},{j},{k}): {e}"
        ))
    })?;

    // Store the SORTED triple: the basis is a function of the multiset, so the
    // key must be too.
    let mut s = [i, j, k];
    s.sort_unstable();
    Ok(TripleTno {
        ijk: (s[0], s[1], s[2]),
        transform: q.dot(&u),
        eps,
        source_pairs: pairs,
        source_pair_dims,
    })
}

/// Rotate a dense `[nvir,nvir,nvir]` triple block into a triple's TNO basis.
///
/// `out[ã,b̃,c̃] = Σ_abc Q̃_aã Q̃_bb̃ Q̃_cc̃ · x[a,b,c]` — the same orthogonal `Q̃` on
/// all three virtual axes, which is what makes the `(T)` contraction invariant
/// at zero truncation.
///
/// Done as three successive `(nvir² × nvir) · (nvir × ntno)` GEMMs with an axis
/// rotation between them, rather than a naive `O(nv⁶)` sextuple loop.
///
/// # Errors
///
/// [`FerricError::General`] when `x` is not `(nvir, nvir, nvir)` for `tno`'s
/// parent basis dimension.
pub fn w3_to_tno(x: &Array3<f64>, tno: &TripleTno, nvir: usize) -> Result<Array3<f64>, FerricError> {
    if x.dim() != (nvir, nvir, nvir) {
        return Err(FerricError::General(format!(
            "w3_to_tno: block is {:?}, expected ({nvir}, {nvir}, {nvir})",
            x.dim()
        )));
    }
    let q = &tno.transform; // (nvir x ntno)
    let n = tno.ntno();

    // Contract the LAST axis, then cycle: [a,b,c] -> [b,c,ã] -> [c,ã,b̃] ->
    // [ã,b̃,c̃]. Three passes return the axes to their original roles.
    let mut cur = x.clone();
    let mut dims = (nvir, nvir, nvir);
    for _ in 0..3 {
        let (d0, d1, d2) = dims;
        let flat = cur
            .to_shape((d0 * d1, d2))
            .map_err(|e| FerricError::General(format!("w3_to_tno reshape: {e}")))?;
        let red = flat.dot(q); // (d0*d1, n)
        let red3 = red
            .to_shape((d0, d1, n))
            .map_err(|e| FerricError::General(format!("w3_to_tno reshape back: {e}")))?
            .to_owned();
        // Cycle axes so the next pass contracts what is now the last axis.
        cur = red3.view().permuted_axes([1, 2, 0]).as_standard_layout().into_owned();
        dims = (d1, n, d0);
    }
    Ok(cur)
}

/// Back-transform a TNO-basis `[ntno,ntno,ntno]` block into the canonical
/// virtual space: `out[a,b,c] = Σ_ãb̃c̃ Q̃_aã Q̃_bb̃ Q̃_cc̃ · y[ã,b̃,c̃]`.
///
/// The exact inverse of [`w3_to_tno`] **only when nothing was truncated** —
/// `Q̃ Q̃ᵀ` is the projector onto the retained TNO space, which is the identity
/// exactly when `Q̃` is square. Under truncation this is the projected block,
/// which is the intended lossy step, not a bug; hence the round-trip test is
/// pinned at `t_cut_tno = 0`.
///
/// # Errors
///
/// [`FerricError::General`] when `y` is not `(ntno, ntno, ntno)`.
pub fn w3_from_tno(
    y: &Array3<f64>,
    tno: &TripleTno,
    nvir: usize,
) -> Result<Array3<f64>, FerricError> {
    let n = tno.ntno();
    if y.dim() != (n, n, n) {
        return Err(FerricError::General(format!(
            "w3_from_tno: block is {:?}, expected ({n}, {n}, {n})",
            y.dim()
        )));
    }
    let qt = tno.transform.t().to_owned(); // (ntno x nvir)
    let mut cur = y.clone();
    let mut dims = (n, n, n);
    for _ in 0..3 {
        let (d0, d1, d2) = dims;
        let flat = cur
            .to_shape((d0 * d1, d2))
            .map_err(|e| FerricError::General(format!("w3_from_tno reshape: {e}")))?;
        let up = flat.dot(&qt); // (d0*d1, nvir)
        let up3 = up
            .to_shape((d0, d1, nvir))
            .map_err(|e| FerricError::General(format!("w3_from_tno reshape back: {e}")))?
            .to_owned();
        cur = up3.view().permuted_axes([1, 2, 0]).as_standard_layout().into_owned();
        dims = (d1, nvir, d0);
    }
    Ok(cur)
}

/// The **unweighted** per-triple `(T)` contribution, evaluated in this triple's
/// semicanonical TNO basis.
///
/// ```text
///   contribution = Σ_ãb̃c̃  W̃[ã,b̃,c̃] · V[ã,b̃,c̃] / D[ã,b̃,c̃]
///   D[ã,b̃,c̃]    = e_ijk − ε̃_ã − ε̃_b̃ − ε̃_c̃
/// ```
///
/// This is exactly the `partial` local of [`crate::ccsd_t_closed_shell`]'s
/// parallel map, *before* its `mult * partial` — i.e. precisely what
/// [`crate::dlpno_ccsd_t::screened_triple_energy`]'s callback contract asks for.
/// It deliberately does NOT apply the multiplicity weight or the `/3` divisor:
/// those belong to `screened_triple_energy`, and keeping them there is what
/// stops a caller reintroducing the spin-orbital `/6` convention.
///
/// `w_tilde` and `v` are the DENSE `[nvir,nvir,nvir]` blocks the existing kernel
/// already builds (`W̃ = 4W + W(bca) + W(cab) − 2W(cba) − 2W(acb) − 2W(bac)` and
/// `V = W + singles`); they are rotated in here. `e_ijk = ε_i + ε_j + ε_k`.
///
/// # Why the denominator is the load-bearing part
///
/// `W̃` and `V` are honest 3-index tensors: rotating both by the same orthogonal
/// `Q̃` leaves their contraction invariant. `D` is **not** — it is the diagonal
/// of `e_ijk − F_a − F_b − F_c`, and it stays diagonal in the new basis only
/// because `ε̃` are the Fock eigenvalues *there*. Feed this function the raw
/// union basis, or the diagonal-only `f_ãã`, and it returns a plausible wrong
/// number. That is the whole argument for
/// [`TripleTnoBasis::build`]'s rediagonalization step.
///
/// # Errors
///
/// [`FerricError::General`] on a shape disagreement, or when a denominator is
/// smaller than `1e-10` in magnitude (a vanishing `(T)` denominator is a
/// physically meaningless input, not something to silently divide by).
pub fn tno_triple_contribution(
    w_tilde: &Array3<f64>,
    v: &Array3<f64>,
    tno: &TripleTno,
    nvir: usize,
    e_ijk: f64,
) -> Result<f64, FerricError> {
    if w_tilde.dim() != (nvir, nvir, nvir) || v.dim() != (nvir, nvir, nvir) {
        return Err(FerricError::General(format!(
            "tno_triple_contribution: blocks are {:?} / {:?}, expected \
             ({nvir}, {nvir}, {nvir})",
            w_tilde.dim(),
            v.dim()
        )));
    }
    let wt = w3_to_tno(w_tilde, tno, nvir)?;
    let vt = w3_to_tno(v, tno, nvir)?;
    let n = tno.ntno();
    let eps = &tno.eps;

    let mut acc = 0.0f64;
    for a in 0..n {
        for b in 0..n {
            let dab = e_ijk - eps[a] - eps[b];
            for c in 0..n {
                let d = dab - eps[c];
                if d.abs() < 1e-10 {
                    return Err(FerricError::General(format!(
                        "tno_triple_contribution: vanishing denominator {d:.3e} at \
                         TNO ({a},{b},{c}) of triple {:?}",
                        tno.ijk
                    )));
                }
                acc += wt[[a, b, c]] * vt[[a, b, c]] / d;
            }
        }
    }
    Ok(acc)
}

/// The same per-triple contribution evaluated **densely** in the canonical
/// virtual basis — the oracle [`tno_triple_contribution`] must reproduce at
/// `t_cut_tno = 0`.
///
/// Term for term this is [`crate::ccsd_t_closed_shell`]'s inner `partial` loop.
/// Public because it is the honest way for a caller to check its own TNO
/// threshold against the untruncated answer on its own system rather than
/// trusting the threshold.
///
/// # Errors
///
/// [`FerricError::General`] on a shape disagreement, a wrong-length `eps_vir`,
/// or a vanishing denominator (same rule as the TNO path).
pub fn dense_triple_contribution(
    w_tilde: &Array3<f64>,
    v: &Array3<f64>,
    eps_vir: &[f64],
    e_ijk: f64,
) -> Result<f64, FerricError> {
    let (n0, n1, n2) = w_tilde.dim();
    if v.dim() != (n0, n1, n2) || n0 != n1 || n1 != n2 {
        return Err(FerricError::General(format!(
            "dense_triple_contribution: blocks are {:?} / {:?}, expected equal cubes",
            w_tilde.dim(),
            v.dim()
        )));
    }
    if eps_vir.len() != n0 {
        return Err(FerricError::General(format!(
            "dense_triple_contribution: eps_vir has {} entries, expected {n0}",
            eps_vir.len()
        )));
    }
    let mut acc = 0.0f64;
    for a in 0..n0 {
        for b in 0..n0 {
            let dab = e_ijk - eps_vir[a] - eps_vir[b];
            for c in 0..n0 {
                let d = dab - eps_vir[c];
                if d.abs() < 1e-10 {
                    return Err(FerricError::General(format!(
                        "dense_triple_contribution: vanishing denominator {d:.3e} at \
                         ({a},{b},{c})"
                    )));
                }
                acc += w_tilde[[a, b, c]] * v[[a, b, c]] / d;
            }
        }
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dlpno_ccsd_t::{build_triple_domains, complete_triple_domains, screened_triple_energy};
    use ndarray::Array4;

    // ------------------------------------------------------------------
    // Deterministic toy system. Small on purpose: this file runs on a
    // shared, heavily loaded box, and every claim here is about EXACTNESS.
    // ------------------------------------------------------------------

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    }

    /// Virtual orbital energies, well separated so no denominator is small AND
    /// no two are degenerate (degeneracy would make the semicanonical
    /// eigenvector orientation arbitrary and the permutation test flaky).
    fn eps_vir(nvir: usize) -> Vec<f64> {
        (0..nvir).map(|a| 0.5 + 0.13 * a as f64).collect()
    }

    fn eps_occ(nocc: usize) -> Vec<f64> {
        (0..nocc).map(|i| -1.0 - 0.11 * i as f64).collect()
    }

    /// Chemist `(ia|jb)` at `[i,a,j,b]`, symmetric under `(ia) <-> (jb)`.
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

    /// First-order amplitudes — the object the constituent PNOs are built from.
    fn mp2_t2(ovov: &Array4<f64>, eo: &[f64], ev: &[f64]) -> Array4<f64> {
        let (nocc, nvir, _, _) = ovov.dim();
        Array4::from_shape_fn((nocc, nocc, nvir, nvir), |(i, j, a, b)| {
            ovov[[i, a, j, b]] / (eo[i] + eo[j] - ev[a] - ev[b])
        })
    }

    fn line_centers(nocc: usize, spacing: f64) -> Array2<f64> {
        Array2::from_shape_fn((nocc, 3), |(i, ax)| if ax == 0 { i as f64 * spacing } else { 0.0 })
    }

    fn toy(nocc: usize, nvir: usize, t_cut: f64, spacing: f64) -> (TripleTnoBasis, Array4<f64>) {
        let (eo, ev) = (eps_occ(nocc), eps_vir(nvir));
        let ovov = ovov_block(nocc, nvir);
        let t2 = mp2_t2(&ovov, &eo, &ev);
        let d = complete_triple_domains(&line_centers(nocc, spacing)).unwrap();
        let basis = TripleTnoBasis::build(&d, nvir, &ev, t_cut, |i, j| {
            Array2::from_shape_fn((nvir, nvir), |(a, b)| t2[[i, j, a, b]])
        })
        .unwrap();
        (basis, t2)
    }

    /// A stand-in for the kernel's `W̃` and `V` blocks: dense, deterministic,
    /// with no symmetry the rotation could accidentally exploit.
    fn cube(nvir: usize, seed: u64) -> Array3<f64> {
        let mut s = seed;
        Array3::from_shape_fn((nvir, nvir, nvir), |_| lcg(&mut s))
    }

    // ============ STAGE 1: the union + semicanonical basis =================

    /// THE EXACTNESS PRECONDITION: at `t_cut_tno = 0` every TNO transform must
    /// be a SQUARE ORTHOGONAL matrix.
    ///
    /// At zero truncation each constituent PNO transform is square orthogonal,
    /// so each projector `P Pᵀ = I` and `M` is a positive multiple of the
    /// identity — full rank. Everything downstream rests on this: a square
    /// orthogonal rotation of `W̃`, `V` and the Fock operator cancels out of the
    /// `(T)` contraction. A non-orthonormal transform would silently rescale the
    /// energy instead of failing.
    #[test]
    fn stage1_transforms_are_square_orthogonal_at_zero_cut() {
        let (nocc, nvir) = (4, 5);
        let (basis, _) = toy(nocc, nvir, 0.0, 1.5);

        assert_eq!(basis.triples.len(), 20, "C(4+2,3) = 20 banded triples");
        assert!(basis.is_complete(), "t_cut_tno = 0 must keep every virtual");
        assert_eq!(basis.virtual_retention(), 1.0);

        let mut worst = 0.0f64;
        for t in &basis.triples {
            assert_eq!(t.ntno(), nvir, "triple {:?} is not square", t.ijk);
            let gram = t.transform.t().dot(&t.transform);
            for a in 0..nvir {
                for b in 0..nvir {
                    let want = if a == b { 1.0 } else { 0.0 };
                    worst = worst.max((gram[(a, b)] - want).abs());
                }
            }
        }
        eprintln!("stage 1: max |Q^T Q - I| over 20 triples = {worst:.3e}");
        assert!(worst < 1e-10, "TNO transforms are not orthogonal: {worst:.3e}");
    }

    /// Orthonormality must hold under TRUNCATION too, where the transform is
    /// rectangular — `Q̃ᵀ Q̃ = I_ntno` still, even though `Q̃ Q̃ᵀ != I_nvir`.
    #[test]
    fn stage1_truncated_transforms_are_still_orthonormal() {
        let (nocc, nvir) = (4, 6);
        let (basis, _) = toy(nocc, nvir, 1e-3, 1.5);
        assert!(!basis.is_complete(), "test premise: something must truncate");

        let mut worst = 0.0f64;
        for t in &basis.triples {
            let n = t.ntno();
            let gram = t.transform.t().dot(&t.transform);
            for a in 0..n {
                for b in 0..n {
                    let want = if a == b { 1.0 } else { 0.0 };
                    worst = worst.max((gram[(a, b)] - want).abs());
                }
            }
        }
        eprintln!("stage 1 (truncated): max |Q^T Q - I| = {worst:.3e}");
        assert!(worst < 1e-10);
    }

    /// **THE MULTIPLICITY INVARIANT, pinned.** All 6 orderings of a triple must
    /// give the SAME TNO basis.
    ///
    /// If the basis depended on which index came first, the `m` orderings a
    /// banded triple stands for would be evaluated in `m` different bases, the
    /// weighted `i<=j<=k` band would stop equalling the unrestricted `nocc³`
    /// sum, and [`crate::ccsd_t_closed_shell`]'s `/3` divisor would silently
    /// become wrong — a plausible but wrong energy, with no crash.
    ///
    /// Asserted **on bits**, not to a tolerance: the construction sums `M`'s
    /// projector terms in a canonical sorted pair order precisely so that even
    /// floating-point summation order is permutation-independent. Anything
    /// weaker than bit-identity would let an ordering-dependent summation
    /// through.
    #[test]
    fn stage1_tno_basis_is_permutation_invariant() {
        let (nocc, nvir) = (4, 5);
        let ev = eps_vir(nvir);
        let ovov = ovov_block(nocc, nvir);
        let t2 = mp2_t2(&ovov, &eps_occ(nocc), &ev);
        let amp =
            |i: usize, j: usize| Array2::from_shape_fn((nvir, nvir), |(a, b)| t2[[i, j, a, b]]);
        let centers = line_centers(nocc, 1.5);
        let pnos = build_pno_transforms(
            &complete_pair_domains(&centers).unwrap(),
            nvir,
            0.0,
            amp,
        )
        .unwrap();

        let mut n_checked = 0usize;
        for i in 0..nocc {
            for j in 0..nocc {
                for k in 0..nocc {
                    let want = build_one_tno(&pnos, nvir, &ev, i, j, k).unwrap();
                    for p in [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)] {
                        let got = build_one_tno(&pnos, nvir, &ev, p.0, p.1, p.2).unwrap();
                        assert_eq!(got.ijk, want.ijk, "sorted key differs at {p:?}");
                        assert_eq!(
                            got.source_pairs, want.source_pairs,
                            "constituent pair list differs at {p:?}"
                        );
                        assert_eq!(got.ntno(), want.ntno(), "TNO count differs at {p:?}");
                        for (a, b) in got.eps.iter().zip(want.eps.iter()) {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "TNO eps not bit-identical at {p:?}: {a} vs {b}"
                            );
                        }
                        for (a, b) in got.transform.iter().zip(want.transform.iter()) {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "TNO transform not bit-identical for ({i},{j},{k}) \
                                 vs {p:?}: {a} vs {b}"
                            );
                        }
                        n_checked += 1;
                    }
                }
            }
        }
        eprintln!("stage 1: {n_checked} permuted TNO bases bit-identical");
        assert!(n_checked > 0);
    }

    /// Permutation invariance must survive TRUNCATION, where the retained
    /// dimension itself could otherwise become order-dependent.
    #[test]
    fn stage1_permutation_invariance_survives_truncation() {
        let (nocc, nvir) = (4, 6);
        let ev = eps_vir(nvir);
        let ovov = ovov_block(nocc, nvir);
        let t2 = mp2_t2(&ovov, &eps_occ(nocc), &ev);
        let amp =
            |i: usize, j: usize| Array2::from_shape_fn((nvir, nvir), |(a, b)| t2[[i, j, a, b]]);
        let pnos = build_pno_transforms(
            &complete_pair_domains(&line_centers(nocc, 1.5)).unwrap(),
            nvir,
            1e-3,
            amp,
        )
        .unwrap();
        assert!(
            pnos.pairs.iter().any(|p| p.transform.ncols() < nvir),
            "test premise: the PNOs must actually truncate"
        );

        for i in 0..nocc {
            for j in 0..nocc {
                for k in 0..nocc {
                    let want = build_one_tno(&pnos, nvir, &ev, i, j, k).unwrap();
                    for p in [(i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)] {
                        let got = build_one_tno(&pnos, nvir, &ev, p.0, p.1, p.2).unwrap();
                        assert_eq!(got.ntno(), want.ntno(), "dim differs at {p:?}");
                        for (a, b) in got.transform.iter().zip(want.transform.iter()) {
                            assert_eq!(a.to_bits(), b.to_bits(), "transform differs at {p:?}");
                        }
                    }
                }
            }
        }
    }

    /// SEMICANONICALIZATION, pinned directly: the virtual Fock matrix in the
    /// STORED basis must come back DIAGONAL, carrying exactly the stored `eps`.
    ///
    /// If this fails, every `(T)` denominator built from `TripleTno::eps` is
    /// wrong and the failure is silent — the energy stays plausible.
    #[test]
    fn stage1_semicanonical_fock_is_diagonal() {
        let (nocc, nvir) = (4, 5);
        let ev = eps_vir(nvir);
        let (basis, _) = toy(nocc, nvir, 0.0, 1.5);

        let mut worst_off = 0.0f64;
        let mut worst_diag = 0.0f64;
        for t in &basis.triples {
            let q = &t.transform;
            let n = t.ntno();
            let f = Array2::from_shape_fn((n, n), |(a, b)| {
                (0..nvir).map(|c| q[(c, a)] * q[(c, b)] * ev[c]).sum::<f64>()
            });
            for a in 0..n {
                for b in 0..n {
                    if a == b {
                        worst_diag = worst_diag.max((f[(a, b)] - t.eps[a]).abs());
                    } else {
                        worst_off = worst_off.max(f[(a, b)].abs());
                    }
                }
            }
        }
        eprintln!(
            "stage 1: max off-diag F in the SEMICANONICAL TNO basis = {worst_off:.3e}, \
             max |F_aa - eps_a| = {worst_diag:.3e}"
        );
        assert!(worst_off < 1e-10, "Fock is not diagonal in the stored basis: {worst_off:.3e}");
        assert!(worst_diag < 1e-10, "stored eps disagree with F_aa: {worst_diag:.3e}");
    }

    /// **The PREMISE of the previous test**, and the reason the diagonal-only
    /// shortcut is not an option here: in the RAW (un-rediagonalized) union
    /// basis the virtual Fock matrix has LARGE off-diagonal elements.
    ///
    /// Without this measurement `stage1_semicanonical_fock_is_diagonal` would be
    /// vacuous — it would pass trivially if the union basis happened to already
    /// diagonalize `F`. It does not, and the magnitude printed here is precisely
    /// the error the shortcut would inject into every `(T)` denominator.
    #[test]
    fn stage1_naive_diagonal_fock_would_be_wrong() {
        let (nocc, nvir) = (4, 5);
        let ev = eps_vir(nvir);
        let ovov = ovov_block(nocc, nvir);
        let t2 = mp2_t2(&ovov, &eps_occ(nocc), &ev);
        let amp =
            |i: usize, j: usize| Array2::from_shape_fn((nvir, nvir), |(a, b)| t2[[i, j, a, b]]);
        let pnos = build_pno_transforms(
            &complete_pair_domains(&line_centers(nocc, 1.5)).unwrap(),
            nvir,
            0.0,
            amp,
        )
        .unwrap();

        // Rebuild the RAW union basis (step 1 only, no semicanonicalization).
        let mut worst_off = 0.0f64;
        let mut n_triples = 0usize;
        for i in 0..nocc {
            for j in i..nocc {
                for k in j..nocc {
                    let mut m = Array2::<f64>::zeros((nvir, nvir));
                    for &(p, q) in &constituent_pairs(i, j, k) {
                        let e = pnos.pairs.iter().find(|e| e.ij == (p, q)).unwrap();
                        m = m + e.transform.dot(&e.transform.t());
                    }
                    let (eigs, vecs) = eigh_dc(&m, Uplo::Upper).unwrap();
                    let keep: Vec<usize> =
                        (0..nvir).filter(|&c| eigs[c] > UNION_RANK_TOL).collect();
                    let q_raw = Array2::from_shape_fn((nvir, keep.len()), |(a, slot)| {
                        vecs[(a, keep[slot])]
                    });
                    let n = q_raw.ncols();
                    for a in 0..n {
                        for b in 0..n {
                            if a != b {
                                let f_ab: f64 = (0..nvir)
                                    .map(|c| q_raw[(c, a)] * q_raw[(c, b)] * ev[c])
                                    .sum();
                                worst_off = worst_off.max(f_ab.abs());
                            }
                        }
                    }
                    n_triples += 1;
                }
            }
        }
        eprintln!(
            "stage 1: max off-diag F in the RAW TNO basis over {n_triples} triples = \
             {worst_off:.3e}  <-- the error the diagonal-only shortcut would inject"
        );
        assert!(
            worst_off > 1e-3,
            "premise failed: raw TNOs already diagonalize F ({worst_off:.3e}), so the \
             semicanonicalization test would be vacuous"
        );
    }

    /// The constituent-pair decomposition must have the right cardinality per
    /// multiplicity class: 1 pair for `i=j=k`, 2 for two-equal, 3 for
    /// all-distinct. A triple that lost a pair would union too small a space.
    #[test]
    fn stage1_constituent_pair_counts_match_the_multiplicity_class() {
        for i in 0..4usize {
            for j in i..4 {
                for k in j..4 {
                    let p = constituent_pairs(i, j, k);
                    let want = if i == k {
                        1 // i == j == k
                    } else if i == j || j == k {
                        2
                    } else {
                        3
                    };
                    assert_eq!(p.len(), want, "triple ({i},{j},{k}) gave pairs {p:?}");
                    assert!(p.windows(2).all(|w| w[0] < w[1]), "pairs not sorted/unique: {p:?}");
                    assert!(p.iter().all(|&(a, b)| a <= b), "pair not internally sorted: {p:?}");
                }
            }
        }
    }

    /// **A union is at least as large as any of its parts** — the invariant that
    /// stops anyone quoting a flattering per-pair PNO retention as if it were
    /// the triple's dimension.
    ///
    /// The inequality `ntno >= max_a(dim P^a)` is unconditional (a span contains
    /// each of its summands) and is asserted at every threshold below.
    ///
    /// Whether the union is *strictly* larger is an empirical question, and this
    /// test MEASURES it across a threshold sweep rather than assuming it. The
    /// measured answer on this fixture (nocc=4, nvir=6) is worth recording,
    /// because it is not the one the analogy with `ferric_rpa::dlpno_rpa`'s
    /// per-orbital unions would predict:
    ///
    /// ```text
    ///   t_cut   TNO retention   triples with ntno > max pair dim
    ///   1e-4        1.000                 0 / 20
    ///   1e-3        0.975                 0 / 20      <- pairs still NESTED
    ///   1e-2        0.950                12 / 20
    ///   3e-2        0.925                16 / 20      <- re-inflation is real
    /// ```
    ///
    /// At mild truncation the three constituent pairs discard the *same*
    /// directions, so their subspaces are nested and the union costs nothing
    /// over the largest pair. Re-inflation only switches on once the threshold
    /// is aggressive enough that different pairs keep different virtuals. Both
    /// regimes are asserted here so neither can be quoted as the general case.
    #[test]
    fn stage1_union_is_at_least_as_large_as_its_pairs() {
        let (nocc, nvir) = (4, 6);
        let mut strict_by_cut = Vec::new();
        for &cut in &[1e-3f64, 1e-2, 3e-2] {
            let (basis, _) = toy(nocc, nvir, cut, 1.5);
            assert!(!basis.is_complete(), "test premise: PNOs must truncate at {cut:.0e}");

            let mut n_strict = 0usize;
            for t in &basis.triples {
                let max_pair = t.source_pair_dims.iter().copied().max().unwrap();
                // THE INVARIANT — unconditional, at every threshold.
                assert!(
                    t.ntno() >= max_pair,
                    "triple {:?} at cut {cut:.0e}: union dim {} < largest constituent \
                     pair dim {max_pair}",
                    t.ijk,
                    t.ntno()
                );
                if t.ntno() > max_pair {
                    n_strict += 1;
                }
            }
            eprintln!(
                "stage 1: cut {cut:.0e} -> TNO retention {:.3}, max ntno {}, \
                 {n_strict}/{} triples strictly larger than their biggest pair",
                basis.virtual_retention(),
                basis.max_ntno(),
                basis.triples.len()
            );
            strict_by_cut.push(n_strict);
        }
        // Nested at mild truncation, re-inflating at aggressive truncation. The
        // first is the surprising half and is pinned so a future change that
        // made unions inflate everywhere is noticed rather than welcomed.
        assert_eq!(
            strict_by_cut[0], 0,
            "at 1e-3 the constituent PNO subspaces were nested when measured; a \
             change here means the union step now costs dimension where it did not"
        );
        assert!(
            strict_by_cut[2] > 0,
            "at 3e-2 the union must strictly exceed its largest pair on some \
             triple, else the union step is inert and the whole construction \
             reduces to picking one pair's PNOs"
        );
        assert!(
            strict_by_cut[2] >= strict_by_cut[0],
            "re-inflation must not decrease as truncation tightens: {strict_by_cut:?}"
        );
    }

    /// A loose threshold must actually compress and report a smaller working
    /// set — otherwise the knob is inert.
    #[test]
    fn stage1_truncation_compresses() {
        let (nocc, nvir) = (4, 6);
        let (tight, _) = toy(nocc, nvir, 0.0, 1.5);
        let (loose, _) = toy(nocc, nvir, 1e-3, 1.5);

        assert!(tight.is_complete());
        assert!(!loose.is_complete(), "a loose threshold should truncate something");
        assert!(loose.virtual_retention() < 1.0);
        let (tno_el, dense_el) = loose.block_elements();
        eprintln!("stage 1: block elements {tno_el} (TNO) vs {dense_el} (dense)");
        assert!(tno_el < dense_el, "TNO block count {tno_el} not below dense {dense_el}");
        // The exact-limit basis must report NO compression.
        let (t_el, t_dense) = tight.block_elements();
        assert_eq!(t_el, t_dense, "t_cut_tno = 0 must report full block counts");
    }

    /// The TRIPLE SCREEN composes: dropping triples drops TNO blocks, and the
    /// blocks that remain are exactly the retained triples, in order.
    #[test]
    fn stage1_composes_with_triple_screening() {
        let (nocc, nvir) = (4, 5);
        let ev = eps_vir(nvir);
        let ovov = ovov_block(nocc, nvir);
        let t2 = mp2_t2(&ovov, &eps_occ(nocc), &ev);
        let amp =
            |i: usize, j: usize| Array2::from_shape_fn((nvir, nvir), |(a, b)| t2[[i, j, a, b]]);
        // Two clusters: 0,2 Bohr and 20,22 Bohr — the dlpno_ccsd_t fixture.
        let centers = ndarray::array![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [22.0, 0.0, 0.0]
        ];
        let all = complete_triple_domains(&centers).unwrap();
        let screened = build_triple_domains(&centers, 5.0).unwrap();
        assert!(screened.triples.len() < all.triples.len(), "test premise");

        let b_all = TripleTnoBasis::build(&all, nvir, &ev, 0.0, amp).unwrap();
        let b_scr = TripleTnoBasis::build(&screened, nvir, &ev, 0.0, amp).unwrap();
        assert!(b_scr.triples.len() < b_all.triples.len());
        let keys: Vec<_> = b_scr.triples.iter().map(|t| t.ijk).collect();
        assert_eq!(keys, screened.triples, "TNO blocks must track the retained triple list");
        assert_eq!(nocc, 4);
    }

    /// Bad inputs must error, not produce a plausible wrong number.
    #[test]
    fn stage1_invalid_inputs_are_rejected() {
        let (nocc, nvir) = (3, 4);
        let d = complete_triple_domains(&line_centers(nocc, 1.0)).unwrap();
        let amp = |_i: usize, _j: usize| Array2::<f64>::zeros((nvir, nvir));
        // eps_vir of the wrong length.
        assert!(TripleTnoBasis::build(&d, nvir, &eps_vir(nvir + 1), 0.0, amp).is_err());
        // Negative threshold (propagated from build_pno_transforms).
        assert!(TripleTnoBasis::build(&d, nvir, &eps_vir(nvir), -1.0, amp).is_err());
        // Amplitude block of the wrong size.
        let bad = |_i: usize, _j: usize| Array2::<f64>::zeros((nvir - 1, nvir - 1));
        assert!(TripleTnoBasis::build(&d, nvir, &eps_vir(nvir), 0.0, bad).is_err());
    }

    // ======== STAGE 2: the triple-block round trip (LOAD-BEARING) ==========

    /// **THE LOAD-BEARING TEST**: a `[nv,nv,nv]` block through the TNO basis and
    /// back must be the identity at `t_cut_tno = 0`.
    ///
    /// The transform is then square orthogonal on all three axes, so
    /// `Q̃ Q̃ᵀ = I` and the round trip is exact. Any deviation means the
    /// transform, its orientation (`Q̃ᵀ·` vs `·Q̃`), or the axis cycling in
    /// [`w3_to_tno`] is wrong — and every stage-3 claim rests on this one, so it
    /// is checked elementwise on the whole cube rather than through a scalar.
    #[test]
    fn stage2_w3_round_trip_is_exact_at_zero_cut() {
        let (nocc, nvir) = (4, 5);
        let (basis, _) = toy(nocc, nvir, 0.0, 1.5);
        let x = cube(nvir, 0xA5A5_1234_DEAD_BEEF);
        let scale = x.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(scale > 1e-3, "block is ~zero — the round trip check would be vacuous");

        let mut worst = 0.0f64;
        for t in &basis.triples {
            let y = w3_to_tno(&x, t, nvir).unwrap();
            assert_eq!(y.dim(), (nvir, nvir, nvir), "triple {:?} block truncated", t.ijk);
            let back = w3_from_tno(&y, t, nvir).unwrap();
            worst = worst.max(
                x.iter().zip(back.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max),
            );
        }
        eprintln!("stage 2: max |x - roundtrip(x)| over 20 triples = {worst:.3e} (max |x| = {scale:.3e})");
        assert!(worst < 1e-10, "triple-block round trip is not exact: {worst:.3e}");
    }

    /// The forward transform must be the honest three-axis contraction. Checked
    /// against a naive `O(nv⁶)` reference sum on one triple, so a mis-cycled
    /// axis in the GEMM path (which the round trip alone could not detect,
    /// because a consistent wrong permutation still inverts) is caught.
    #[test]
    fn stage2_transform_matches_a_naive_sextuple_sum() {
        let (nocc, nvir) = (3, 4);
        let (basis, _) = toy(nocc, nvir, 0.0, 1.5);
        let t = &basis.triples[basis.triples.len() - 1];
        let q = &t.transform;
        let x = cube(nvir, 0x1357_9BDF_2468_ACE0);

        let got = w3_to_tno(&x, t, nvir).unwrap();
        let mut want = Array3::<f64>::zeros((nvir, nvir, nvir));
        for ap in 0..nvir {
            for bp in 0..nvir {
                for cp in 0..nvir {
                    let mut acc = 0.0;
                    for a in 0..nvir {
                        for b in 0..nvir {
                            for c in 0..nvir {
                                acc += q[(a, ap)] * q[(b, bp)] * q[(c, cp)] * x[[a, b, c]];
                            }
                        }
                    }
                    want[[ap, bp, cp]] = acc;
                }
            }
        }
        let worst =
            got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        eprintln!("stage 2: max |GEMM path - naive sextuple sum| = {worst:.3e}");
        assert!(worst < 1e-10, "the three-axis transform is mis-cycled: {worst:.3e}");
    }

    /// Truncation must make the round trip LOSSY — otherwise the threshold is
    /// inert and any later accuracy/cost curve would be measuring nothing.
    #[test]
    fn stage2_truncation_makes_the_round_trip_lossy() {
        let (nocc, nvir) = (4, 6);
        let (basis, _) = toy(nocc, nvir, 1e-3, 1.5);
        assert!(!basis.is_complete(), "test premise");
        let x = cube(nvir, 0x0BAD_F00D_1122_3344);

        let mut worst = 0.0f64;
        for t in &basis.triples {
            if t.ntno() == nvir {
                continue; // this triple's union happened to stay full rank
            }
            let back = w3_from_tno(&w3_to_tno(&x, t, nvir).unwrap(), t, nvir).unwrap();
            worst = worst.max(
                x.iter().zip(back.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max),
            );
        }
        eprintln!("stage 2: truncated round-trip max deviation = {worst:.3e}");
        assert!(worst > 1e-12, "truncation had no effect on the round trip");
    }

    /// Shape disagreements are caller bugs and must error.
    #[test]
    fn stage2_shape_mismatches_are_rejected() {
        let (nocc, nvir) = (3, 4);
        let (basis, _) = toy(nocc, nvir, 0.0, 1.5);
        let t = &basis.triples[0];
        assert!(w3_to_tno(&cube(nvir + 1, 1), t, nvir).is_err());
        assert!(w3_from_tno(&cube(nvir + 1, 1), t, nvir).is_err());
    }

    // ================= STAGE 3: the per-triple energy ======================

    /// **THE EXACTNESS CONTRACT for the energy.** At `t_cut_tno = 0` the TNO
    /// evaluation of `Σ W̃V/D` must reproduce the dense canonical expression, on
    /// every triple.
    ///
    /// `W̃` and `V` are tensors — rotating both by the same orthogonal `Q̃`
    /// leaves their contraction invariant. `D` is NOT a tensor: it is the
    /// diagonal of `e_ijk − F_a − F_b − F_c`, and it survives the rotation only
    /// because the stored `ε̃` really are the Fock eigenvalues in the rotated
    /// basis. So this test is simultaneously the exactness contract AND the
    /// strongest available check that semicanonicalization was done right —
    /// feed it un-rediagonalized `eps` and it fails, which
    /// `stage3_diagonal_only_fock_shortcut_is_measurably_wrong` demonstrates.
    #[test]
    fn stage3_tno_triple_energy_matches_dense() {
        let (nocc, nvir) = (4, 5);
        let ev = eps_vir(nvir);
        let eo = eps_occ(nocc);
        let (basis, _) = toy(nocc, nvir, 0.0, 1.5);
        let w = cube(nvir, 0xFEED_FACE_CAFE_0001);
        let v = cube(nvir, 0xFEED_FACE_CAFE_0002);

        let mut worst_abs = 0.0f64;
        let mut worst_rel = 0.0f64;
        let mut scale = 0.0f64;
        for t in &basis.triples {
            let (i, j, k) = t.ijk;
            let e_ijk = eo[i] + eo[j] + eo[k];
            let dense = dense_triple_contribution(&w, &v, &ev, e_ijk).unwrap();
            let tno = tno_triple_contribution(&w, &v, t, nvir, e_ijk).unwrap();
            worst_abs = worst_abs.max((tno - dense).abs());
            worst_rel = worst_rel.max((tno - dense).abs() / dense.abs().max(1e-30));
            scale = scale.max(dense.abs());
        }
        eprintln!(
            "stage 3: max |E_TNO - E_dense| = {worst_abs:.3e} (max |E_dense| = \
             {scale:.3e}, max rel = {worst_rel:.3e})"
        );
        assert!(scale > 1e-3, "contributions are ~zero — the check would be vacuous");
        assert!(
            worst_rel < 1e-10,
            "untruncated TNO contribution must reproduce dense: rel {worst_rel:.3e}"
        );
    }

    /// **The diagonal-only Fock shortcut, measured to be wrong.**
    ///
    /// This is the test the brief asked for and it is the sharpest one in the
    /// file: it takes the SAME orthogonal union basis but replaces the
    /// semicanonical `ε̃` with the naive diagonal `f_ãã = Σ_c Q_cã² ε_c` — the
    /// shortcut that broke DLPNO-MP2 by 0.117 Ha — and shows the resulting
    /// contribution is measurably wrong on a triple where the correct one is
    /// exact to 1e-10.
    ///
    /// Note the failure mode: a plausible finite number, no crash, no NaN.
    #[test]
    fn stage3_diagonal_only_fock_shortcut_is_measurably_wrong() {
        let (nocc, nvir) = (4, 5);
        let ev = eps_vir(nvir);
        let eo = eps_occ(nocc);
        let ovov = ovov_block(nocc, nvir);
        let t2 = mp2_t2(&ovov, &eo, &ev);
        let amp =
            |i: usize, j: usize| Array2::from_shape_fn((nvir, nvir), |(a, b)| t2[[i, j, a, b]]);
        let pnos = build_pno_transforms(
            &complete_pair_domains(&line_centers(nocc, 1.5)).unwrap(),
            nvir,
            0.0,
            amp,
        )
        .unwrap();

        let w = cube(nvir, 0xFEED_FACE_CAFE_0001);
        let v = cube(nvir, 0xFEED_FACE_CAFE_0002);
        let (i, j, k) = (0usize, 1usize, 2usize);
        let e_ijk = eo[i] + eo[j] + eo[k];
        let dense = dense_triple_contribution(&w, &v, &ev, e_ijk).unwrap();

        // The CORRECT (semicanonical) basis.
        let good = build_one_tno(&pnos, nvir, &ev, i, j, k).unwrap();
        let e_good = tno_triple_contribution(&w, &v, &good, nvir, e_ijk).unwrap();

        // The SHORTCUT: same raw union basis, diagonal-only "orbital energies".
        let mut m = Array2::<f64>::zeros((nvir, nvir));
        for &(p, q) in &constituent_pairs(i, j, k) {
            let e = pnos.pairs.iter().find(|e| e.ij == (p, q)).unwrap();
            m = m + e.transform.dot(&e.transform.t());
        }
        let (eigs, vecs) = eigh_dc(&m, Uplo::Upper).unwrap();
        let keep: Vec<usize> = (0..nvir).filter(|&c| eigs[c] > UNION_RANK_TOL).collect();
        let q_raw =
            Array2::from_shape_fn((nvir, keep.len()), |(a, slot)| vecs[(a, keep[slot])]);
        let eps_naive: Vec<f64> = (0..q_raw.ncols())
            .map(|a| (0..nvir).map(|c| q_raw[(c, a)] * q_raw[(c, a)] * ev[c]).sum())
            .collect();
        let bad = TripleTno {
            ijk: (i, j, k),
            transform: q_raw,
            eps: eps_naive,
            source_pairs: constituent_pairs(i, j, k),
            source_pair_dims: vec![],
        };
        let e_bad = tno_triple_contribution(&w, &v, &bad, nvir, e_ijk).unwrap();

        eprintln!(
            "stage 3: dense = {dense:.12}, semicanonical = {e_good:.12} (d = {:.3e}), \
             diagonal-only shortcut = {e_bad:.12} (d = {:.3e})",
            e_good - dense,
            e_bad - dense
        );
        assert!(
            (e_good - dense).abs() / dense.abs() < 1e-10,
            "the semicanonical path must be exact"
        );
        assert!(
            (e_bad - dense).abs() / dense.abs() > 1e-6,
            "premise failed: the diagonal-only shortcut agreed with dense to \
             {:.3e} relative — it must be measurably wrong, or this test proves \
             nothing",
            (e_bad - dense).abs() / dense.abs()
        );
    }

    /// **The multiplicity/divisor invariant, end to end.** Feeding the TNO
    /// contribution into [`screened_triple_energy`] at `t_cut_tno = 0` must
    /// reproduce the dense weighted band with its `/3` divisor.
    ///
    /// This is the composition test: it exercises the retained-triple list, the
    /// `m ∈ {1,3,6}` weights and the per-triple TNO evaluation together, so an
    /// ordering-dependent basis or a reintroduced `/6` would show up as an
    /// energy discrepancy rather than staying latent.
    #[test]
    fn stage3_composes_with_screened_triple_energy_exactly() {
        let (nocc, nvir) = (4, 5);
        let ev = eps_vir(nvir);
        let eo = eps_occ(nocc);
        let centers = line_centers(nocc, 1.5);
        let (basis, _) = toy(nocc, nvir, 0.0, 1.5);
        let domains = complete_triple_domains(&centers).unwrap();
        let w = cube(nvir, 0xFEED_FACE_CAFE_0001);
        let v = cube(nvir, 0xFEED_FACE_CAFE_0002);

        let e_tno = screened_triple_energy(&domains, |i, j, k| {
            let t = basis
                .triples
                .iter()
                .find(|t| t.ijk == (i, j, k))
                .expect("every retained triple must have a TNO basis");
            tno_triple_contribution(&w, &v, t, nvir, eo[i] + eo[j] + eo[k])
        })
        .unwrap();

        let e_dense = screened_triple_energy(&domains, |i, j, k| {
            dense_triple_contribution(&w, &v, &ev, eo[i] + eo[j] + eo[k])
        })
        .unwrap();

        let rel = (e_tno - e_dense).abs() / e_dense.abs();
        eprintln!("stage 3: banded E = {e_tno:.12} (TNO) vs {e_dense:.12} (dense), rel {rel:.3e}");
        assert!(e_dense.abs() > 1e-3, "banded energy is ~zero — the check is vacuous");
        assert!(rel < 1e-10, "TNO band must reproduce the dense band: rel {rel:.3e}");
    }

    /// Truncation must actually change the per-triple contribution — a knob that
    /// leaves the energy alone is inert and no accuracy/cost claim about it
    /// would mean anything.
    #[test]
    fn stage3_truncation_changes_the_contribution() {
        let (nocc, nvir) = (4, 6);
        let eo = eps_occ(nocc);
        let (b0, _) = toy(nocc, nvir, 0.0, 1.5);
        let (bt, _) = toy(nocc, nvir, 1e-3, 1.5);
        assert!(!bt.is_complete(), "test premise");
        let w = cube(nvir, 0xFEED_FACE_CAFE_0001);
        let v = cube(nvir, 0xFEED_FACE_CAFE_0002);

        let mut n_changed = 0usize;
        let mut worst = 0.0f64;
        for (t0, tt) in b0.triples.iter().zip(bt.triples.iter()) {
            assert_eq!(t0.ijk, tt.ijk, "triple lists must be aligned");
            let (i, j, k) = t0.ijk;
            let e_ijk = eo[i] + eo[j] + eo[k];
            let e0 = tno_triple_contribution(&w, &v, t0, nvir, e_ijk).unwrap();
            let et = tno_triple_contribution(&w, &v, tt, nvir, e_ijk).unwrap();
            if (e0 - et).abs() > 1e-12 {
                n_changed += 1;
            }
            worst = worst.max((e0 - et).abs());
        }
        eprintln!(
            "stage 3: truncation (retention {:.3}) changed {n_changed}/{} triples, max \
             |dE| = {worst:.3e}",
            bt.virtual_retention(),
            b0.triples.len()
        );
        assert!(n_changed > 0, "truncation had no effect on any triple contribution");
    }

    /// Shape and input errors must be reported, not divided through.
    #[test]
    fn stage3_invalid_inputs_are_rejected() {
        let (nocc, nvir) = (3, 4);
        let ev = eps_vir(nvir);
        let (basis, _) = toy(nocc, nvir, 0.0, 1.5);
        let t = &basis.triples[0];
        let w = cube(nvir, 1);
        let v = cube(nvir, 2);
        let bad = cube(nvir + 1, 3);

        assert!(tno_triple_contribution(&bad, &v, t, nvir, -3.0).is_err());
        assert!(tno_triple_contribution(&w, &bad, t, nvir, -3.0).is_err());
        assert!(dense_triple_contribution(&w, &bad, &ev, -3.0).is_err());
        // eps_vir of the wrong length.
        assert!(dense_triple_contribution(&w, &v, &eps_vir(nvir + 1), -3.0).is_err());
        // A vanishing denominator must error rather than produce an infinity:
        // e_ijk chosen so e_ijk - 3*eps = 0 for the lowest virtual triple.
        let e_sing = 3.0 * ev[0];
        assert!(dense_triple_contribution(&w, &v, &ev, e_sing).is_err());
    }

    /// A single-occupied system yields the one triple `(0,0,0)`, whose union has
    /// exactly ONE constituent pair. The degenerate case
    /// [`crate::ccsd_t_closed_shell`] explicitly supports.
    #[test]
    fn edge_single_occupied_orbital() {
        let nvir = 4;
        let (basis, _) = toy(1, nvir, 0.0, 1.0);
        assert_eq!(basis.triples.len(), 1);
        let t = &basis.triples[0];
        assert_eq!(t.ijk, (0, 0, 0));
        assert_eq!(t.source_pairs, vec![(0, 0)]);
        assert_eq!(t.ntno(), nvir, "zero cut must keep the full virtual space");
    }


    /// An empty occupied space produces no TNO blocks and must not panic.
    #[test]
    fn edge_empty_system() {
        let d = complete_triple_domains(&Array2::<f64>::zeros((0, 3))).unwrap();
        let b = TripleTnoBasis::build(&d, 3, &eps_vir(3), 0.0, |_, _| Array2::zeros((3, 3)))
            .unwrap();
        assert!(b.triples.is_empty());
        assert!(b.is_complete());
        assert_eq!(b.virtual_retention(), 1.0);
        assert_eq!(b.max_ntno(), 0);
    }
}

