//! DLPNO-CCSD — making truncation change the **cost of the iteration**.
//!
//! [`crate::dlpno_ccsd_virtual`] built the exact per-pair PNO substrate and said
//! plainly what it did not do: the 74 residual contractions of
//! [`crate::ccsd_closed_shell`] are still dense, so raising `t_cut_pno` changes
//! the *energy* and nothing else. This module is the attack on that gap.
//!
//! # The one structural difficulty, stated first
//!
//! In a per-pair PNO basis different pairs span **different** virtual subspaces.
//! Any contraction that consumes `t̃^{il}` while summing over virtuals that also
//! index a quantity living in pair `(k,l)`'s basis must convert between the two,
//! and the converter is the **pair-pair overlap**
//!
//! ```text
//!   S^{P,Q} = Q_P ᵀ Q_Q          (npno_P × npno_Q)
//! ```
//!
//! At `t_cut_pno = 0` every `Q` is square orthogonal, so every `S` is an
//! *orthogonal* matrix (the identity only when the two pairs' PNOs coincide;
//! generally a rotation). [`pair_overlap`] builds it and
//! [`tests::stage_a_overlaps_are_orthogonal_at_zero_truncation`] pins that
//! property — it is the invariant that makes the whole rewrite well-posed,
//! because a contraction written with `S` inserted collapses back to the dense
//! one exactly when `S Sᵀ = 1`.
//!
//! # Staging, and where this module actually got to
//!
//! | Stage | What | Status |
//! |-------|------|--------|
//! | A | [`PairIndex`] + [`pair_overlap`] + [`PairOverlaps`] | done, tested |
//! | B | `kcld,ilcd->ki` (`F_oo`) in the PNO basis, S-inserted | done, exact to ~1e-15 |
//! | C | the remaining pair-coupling contractions: `F_vv`, `W_oooo`, `W_voov`/`W_vovo`, `L`-driven `t2` terms | done, each tested |
//! | D | a full PNO-basis residual + iteration reproducing dense CCSD | **NOT done** — see "What is not here" |
//!
//! # The cost claim, and how it is made
//!
//! **No wall-clock number appears anywhere in this module.** The box these were
//! developed on is contested and a timing there would be noise dressed as
//! evidence. The cost claim is made *structurally* instead: every PNO contraction
//! here has a FLOP count that is a pure function of the per-pair `npno`, computed
//! by [`FlopCount`] from the basis alone with no arithmetic performed, and
//! [`tests::cost_strictly_decreases_with_truncation`] asserts that count strictly
//! decreases as `t_cut_pno` rises while the dense count stays fixed. That is a
//! statement about the *algorithm*, which is what "truncation changes the cost"
//! actually means; a wall-clock measurement would additionally be a statement
//! about BLAS shapes and a loaded machine.
//!
//! # SEMICANONICALIZATION
//!
//! Denominators use [`PairPno::eps`] — the eigenvalues of `Q ᵀ diag(ε_v) Q`,
//! i.e. the orbital energies *in the pair's own basis*. The diagonal-only
//! shortcut `f_aa = Σ_c Q_ca² ε_c` is a MEASURED 0.117 Ha error in the MP2
//! sibling and is not used here. [`pno_denominators`] is the only place
//! denominators are formed.
//!
//! # What is not here, and why
//!
//! Stage D — a closed PNO-basis iteration — is **not** in this module. Stages A-C
//! prove the S-insertion is exact contraction-by-contraction, which is the
//! question worth answering first: if it had failed, the remaining rewrite would
//! have been worthless. It did not fail. What a full iteration additionally needs
//! and this module does not supply is a PNO-basis form for the *singles* channel
//! and the `ovvv`/`vvvv` blocks, whose per-pair storage is the actual engineering
//! bulk of a production DLPNO-CCSD. Shipping a full-looking residual that was
//! only spot-checked would be worse than shipping this, because its failure mode
//! is a plausible wrong energy.

use crate::dlpno_ccsd_virtual::{PairPno, PairPnoBasis};
use ferric_core::FerricError;
use ndarray::{Array2, Array4};
use std::collections::HashMap;

// =====================================================================
// STAGE A — pair indexing and pair-pair overlaps
// =====================================================================

/// Lookup from an occupied pair `(i,j)` to its entry in [`PairPnoBasis::pairs`].
///
/// [`PairDomains`](ferric_mp2::pair_domains::PairDomains) stores each pair once
/// with `i <= j`, but the CCSD residual is indexed over the full `nocc²` grid.
/// The mirror `(j,i)` resolves to the **same** basis, exactly as
/// [`crate::dlpno_ccsd_virtual::pno_ccsd_energy`] treats it: the pair density
/// `D^{ji}` is the symmetrized transpose of `D^{ij}`, so the two share PNOs.
///
/// Screened-out pairs simply have no entry — [`PairIndex::get`] returns `None`
/// and every contraction here skips them, which is the same zero-amplitude
/// convention as [`crate::dlpno_ccsd::apply_pair_mask`].
#[derive(Debug, Clone)]
pub struct PairIndex {
    map: HashMap<(usize, usize), usize>,
    nocc: usize,
}

impl PairIndex {
    /// Build the index. Registers both `(i,j)` and `(j,i)` for every pair.
    pub fn new(basis: &PairPnoBasis, nocc: usize) -> Result<Self, FerricError> {
        let mut map = HashMap::new();
        for (p, pair) in basis.pairs.iter().enumerate() {
            let (i, j) = pair.ij;
            if i >= nocc || j >= nocc {
                return Err(FerricError::General(format!(
                    "PairIndex::new: pair ({i},{j}) out of range for nocc = {nocc}"
                )));
            }
            map.insert((i, j), p);
            map.insert((j, i), p);
        }
        Ok(Self { map, nocc })
    }

    /// Index into [`PairPnoBasis::pairs`] for occupied pair `(i,j)`, or `None`
    /// when the pair was screened out.
    pub fn get(&self, i: usize, j: usize) -> Option<usize> {
        self.map.get(&(i, j)).copied()
    }

    /// Number of occupied orbitals this index spans.
    pub fn nocc(&self) -> usize {
        self.nocc
    }
}

/// The pair-pair overlap `S^{P,Q} = Q_P ᵀ Q_Q`, shape `(npno_P × npno_Q)`.
///
/// This is the object that makes a PNO-basis contraction *mean* the same thing as
/// its dense original: a quantity carried in pair `Q`'s basis is converted into
/// pair `P`'s by `S^{P,Q} · (…) · S^{P,Q}ᵀ` on the virtual indices it shares.
///
/// At `t_cut_pno = 0` both transforms are square orthogonal, so `S` is orthogonal
/// and every such conversion is invertible and lossless — that is the exactness
/// contract. Under truncation `S Sᵀ ≠ 1` and the conversion projects, which is
/// the intended lossy step.
pub fn pair_overlap(p: &PairPno, q: &PairPno) -> Array2<f64> {
    p.transform.t().dot(&q.transform)
}

/// All pair-pair overlaps of a basis, cached.
///
/// Stored densely over the `n_pairs²` grid because every contraction below wants
/// random access by `(P,Q)`. A production run would build these only for pairs the
/// domain screen actually couples; here the whole block is kept so the exactness
/// tests cannot accidentally pass by never touching an off-diagonal `S`.
#[derive(Debug, Clone)]
pub struct PairOverlaps {
    s: Vec<Array2<f64>>,
    n: usize,
}

impl PairOverlaps {
    /// Exact element count of the cache this basis would produce:
    /// `Σ_P Σ_Q npno_P · npno_Q  =  (Σ_P npno_P)²`.
    ///
    /// This is not an estimate. It is the same sum the build loop performs, in
    /// closed form, so it cannot drift from what is allocated — which is the
    /// whole point: the historical failures in this repo came from a *second*,
    /// hand-written implementation of an allocation's shape silently diverging
    /// from the allocator.
    ///
    /// Note the asymptotics this number hides: with complete domains
    /// `n_pairs ~ nocc²` and `npno ~ nvir`, so this is `O(nocc⁴ · nvir²)` — the
    /// worst scaling in the crate, and the only unbounded allocation on the
    /// DLPNO path.
    pub fn plan_elements(basis: &PairPnoBasis) -> usize {
        let total_pno: usize =
            basis.pairs.iter().map(|p| p.transform.ncols()).fold(0usize, usize::saturating_add);
        total_pno.saturating_mul(total_pno)
    }

    /// Build every `S^{P,Q}`, refusing up front if the cache cannot fit
    /// `budget_bytes`.
    ///
    /// WHY THIS GUARD EXISTS. [`Self::build`] materializes the **whole**
    /// `n_pairs² × npno²` block. That is deliberate and stays deliberate — the
    /// retention policy is documented on the type: keeping every `S`, including
    /// the off-diagonals a production screen would drop, is what stops the
    /// exactness tests from passing by never touching one. Nothing here changes
    /// the numerics or which overlaps are kept.
    ///
    /// What it changes is the failure mode. `Σ npno` grows like `nocc²·nvir`, so
    /// this cache grows like its square, faster than anything else on the DLPNO
    /// path, and it was allocated with no reference to
    /// [`crate::CcConfig::memory_budget_bytes`] at all. That is precisely the
    /// shape of the LNO-coupled incident: the reported working set was the
    /// compressed pair-shaped one (0.055 GB predicted) while the allocator
    /// served the dense setup (7.3 GB peak), and the first signal a user got was
    /// the OOM killer. A job that cannot fit should say so, in one line, naming
    /// the term — before it starts allocating.
    ///
    /// # Errors
    ///
    /// [`FerricError::General`] when the cache exceeds `budget_bytes`, carrying
    /// the per-reservation breakdown.
    pub fn build_within_budget(
        basis: &PairPnoBasis,
        budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        use ferric_core::memory::plan::{Lifetime, MemoryPlan};
        let n = basis.pairs.len();
        let mut plan = MemoryPlan::with_budget_bytes(
            budget_bytes,
            format!("PairOverlaps S^(P,Q) cache ({n} pairs)"),
        );
        plan.reserve("S^(P,Q) dense pair-pair overlap cache", Self::plan_elements(basis), Lifetime::Resident);
        plan.check()?;
        Ok(Self::build(basis))
    }

    /// Build every `S^{P,Q}`.
    ///
    /// Unguarded. Prefer [`Self::build_within_budget`] on any path that has a
    /// [`crate::CcConfig`] to hand; this remains for the exactness tests and
    /// small fixtures, where the cache is kilobytes.
    pub fn build(basis: &PairPnoBasis) -> Self {
        let n = basis.pairs.len();
        let mut s = Vec::with_capacity(n * n);
        for p in &basis.pairs {
            for q in &basis.pairs {
                s.push(pair_overlap(p, q));
            }
        }
        Self { s, n }
    }

    /// `S^{P,Q}`, shape `(npno_P × npno_Q)`.
    pub fn get(&self, p: usize, q: usize) -> &Array2<f64> {
        &self.s[p * self.n + q]
    }

    /// Number of pairs.
    pub fn n_pairs(&self) -> usize {
        self.n
    }

    /// Largest `‖S^{P,Q} S^{P,Q}ᵀ − 1‖_∞` over all pairs — zero (to round-off)
    /// exactly when nothing was truncated.
    ///
    /// This is the honest diagnostic for "how far from exact is this basis": it
    /// measures the *only* approximation the S-insertion introduces.
    pub fn max_nonorthogonality(&self) -> f64 {
        let mut worst = 0.0f64;
        for p in 0..self.n {
            for q in 0..self.n {
                let s = self.get(p, q);
                let g = s.dot(&s.t());
                for a in 0..g.nrows() {
                    for b in 0..g.ncols() {
                        let want = if a == b { 1.0 } else { 0.0 };
                        worst = worst.max((g[(a, b)] - want).abs());
                    }
                }
            }
        }
        worst
    }
}

// =====================================================================
// Per-pair integral blocks — the storage that makes truncation cheap
// =====================================================================

/// The `ovov` integral `(kc|ld)` stored **per occupied pair**, each block rotated
/// into that pair's own PNO basis.
///
/// This is the storage change that turns truncation into a cost reduction. The
/// dense block is `no²·nv²`; here pair `P = (k,l)` carries only
/// `npno_P²` numbers:
///
/// ```text
///   ĝ^{kl}[c̄,d̄] = Σ_cd Q_kl[c,c̄] (kc|ld) Q_kl[d,d̄]
/// ```
///
/// Every contraction in this module reads `ĝ` and never the dense block, so the
/// dense `nv²` per occupied pair is genuinely never materialized in the
/// iteration — it appears only once, at build time, and a production build would
/// form it per-pair straight from the RI `B` tensors.
#[derive(Debug, Clone)]
pub struct PairOvov {
    blocks: Vec<Array2<f64>>,
}

impl PairOvov {
    /// Rotate `(kc|ld)` into each pair's PNO basis.
    ///
    /// `ovov` is the chemist block stored at `[k,c,l,d]` exactly as
    /// [`crate::ccsd_closed_shell`] builds it.
    ///
    /// # Errors
    ///
    /// [`FerricError::General`] on any shape disagreement with the basis.
    pub fn build(ovov: &Array4<f64>, basis: &PairPnoBasis) -> Result<Self, FerricError> {
        let (no_k, nv_c, no_l, nv_d) = ovov.dim();
        if nv_c != basis.nvir || nv_d != basis.nvir {
            return Err(FerricError::General(format!(
                "PairOvov::build: ovov virtual dims ({nv_c}, {nv_d}) disagree with nvir = {}",
                basis.nvir
            )));
        }
        if no_k != no_l {
            return Err(FerricError::General(format!(
                "PairOvov::build: ovov occupied dims ({no_k}, {no_l}) are not square"
            )));
        }
        let nvir = basis.nvir;
        let mut blocks = Vec::with_capacity(basis.pairs.len());
        for p in &basis.pairs {
            let (k, l) = p.ij;
            if k >= no_k || l >= no_k {
                return Err(FerricError::General(format!(
                    "PairOvov::build: pair ({k},{l}) out of range for nocc = {no_k}"
                )));
            }
            let g = Array2::from_shape_fn((nvir, nvir), |(c, d)| ovov[[k, c, l, d]]);
            blocks.push(p.transform.t().dot(&g).dot(&p.transform));
        }
        Ok(Self { blocks })
    }

    /// `ĝ^{P}` for pair index `P`.
    pub fn get(&self, p: usize) -> &Array2<f64> {
        &self.blocks[p]
    }

    /// Elements stored, versus the dense `n_pairs · nvir²` the same pair list
    /// would need. A count, not a timing.
    pub fn elements(&self, nvir: usize) -> (usize, usize) {
        let pno: usize = self.blocks.iter().map(|b| b.len()).sum();
        (pno, self.blocks.len() * nvir * nvir)
    }
}

/// A general per-occupied-pair `(k,l)` integral block rotated into a *chosen*
/// pair's basis — used where the natural home of an integral is not the pair
/// whose amplitudes it multiplies.
///
/// Kept separate from [`PairOvov`] so that the two roles cannot be silently
/// swapped: `PairOvov` rotates `(kc|ld)` into pair `(k,l)`'s basis (the pair the
/// *integral's* occupied labels name), which is the storage a production DLPNO
/// code uses.
#[allow(dead_code)] // kept: documents the production-DLPNO storage rotation this doc describes
fn rotate_block(g: &Array2<f64>, q: &Array2<f64>) -> Array2<f64> {
    q.t().dot(g).dot(q)
}

// =====================================================================
// Denominators — SEMICANONICAL, from PairPno::eps
// =====================================================================

/// The `t2` denominator `D^{ij}[ã,b̃] = ε_i + ε_j − ε_ã − ε_b̃` in pair `(i,j)`'s
/// **semicanonical** PNO basis.
///
/// `eps_occ` are the canonical occupied orbital energies; the virtual energies
/// come from [`PairPno::eps`], which are the eigenvalues of `Q ᵀ diag(ε_v) Q`.
/// Using the raw `Σ_c Q_ca² ε_c` diagonal instead is the silent 0.117 Ha failure
/// documented in [`crate::dlpno_ccsd_virtual`]; it is not available here.
///
/// # Errors
///
/// [`FerricError::General`] when a pair's occupied index exceeds `eps_occ`.
pub fn pno_denominators(
    basis: &PairPnoBasis,
    eps_occ: &[f64],
) -> Result<Vec<Array2<f64>>, FerricError> {
    let mut out = Vec::with_capacity(basis.pairs.len());
    for p in &basis.pairs {
        let (i, j) = p.ij;
        if i >= eps_occ.len() || j >= eps_occ.len() {
            return Err(FerricError::General(format!(
                "pno_denominators: pair ({i},{j}) out of range for {} occupied energies",
                eps_occ.len()
            )));
        }
        let npno = p.eps.len();
        let e_ij = eps_occ[i] + eps_occ[j];
        out.push(Array2::from_shape_fn((npno, npno), |(a, b)| {
            e_ij - p.eps[a] - p.eps[b]
        }));
    }
    Ok(out)
}

// =====================================================================
// STAGE B — the proof-of-concept pair-coupling contraction
// =====================================================================

/// `F_oo` direct term: `A[k,i] = Σ_{l,c,d} (kc|ld) · t2[i,l,c,d]`, evaluated
/// entirely in per-pair PNO bases.
///
/// **This is the proof of concept for the whole module.** It is the archetype of
/// the ~10 contractions that couple two *different* occupied pairs: the integral
/// is naturally indexed by pair `(k,l)` and the amplitude by pair `(i,l)`, and
/// they share the summed virtuals `c,d`. In a per-pair basis those two objects
/// live in different subspaces, so the sum is only meaningful with the pair-pair
/// overlap inserted:
///
/// ```text
///   A[k,i] = Σ_l  tr[ ĝ^{kl} · S^{kl,il} · t̃^{il} · (S^{kl,il})ᵀ ]
/// ```
///
/// Derivation, so the insertion is checkable rather than asserted. Writing the
/// dense objects through their PNO representations,
/// `t2[i,l,c,d] = Σ_{c̃d̃} Q_il[c,c̃] t̃^{il}[c̃,d̃] Q_il[d,d̃]` and
/// `(kc|ld) = Σ_{c̄d̄} Q_kl[c,c̄] ĝ^{kl}[c̄,d̄] Q_kl[d,d̄]` (exact when `Q_kl` is
/// square), the `c` sum gives `Σ_c Q_kl[c,c̄] Q_il[c,c̃] = S^{kl,il}[c̄,c̃]` and
/// likewise for `d`. That is where `S` comes from — it is not a fudge factor, it
/// is the resolution of identity between the two pairs' virtual spaces.
///
/// Pairs `(i,l)` or `(k,l)` absent from the basis contribute **zero**, matching
/// the screened-amplitude convention.
///
/// # Errors
///
/// [`FerricError::General`] on shape disagreements.
pub fn pno_foo_direct(
    t2_pno: &[Array2<f64>],
    g_pno: &PairOvov,
    basis: &PairPnoBasis,
    overlaps: &PairOverlaps,
    index: &PairIndex,
) -> Result<Array2<f64>, FerricError> {
    if t2_pno.len() != basis.pairs.len() {
        return Err(FerricError::General(format!(
            "pno_foo_direct: got {} amplitude blocks for {} pairs",
            t2_pno.len(),
            basis.pairs.len()
        )));
    }
    let nocc = index.nocc();
    let mut a = Array2::<f64>::zeros((nocc, nocc));
    for k in 0..nocc {
        for i in 0..nocc {
            let mut acc = 0.0;
            for l in 0..nocc {
                let (Some(p_kl), Some(p_il)) = (index.get(k, l), index.get(i, l)) else {
                    continue;
                };
                // The integral block ĝ^{kl} is stored for the ORDERED pair as it
                // sits in `basis.pairs`; (k,l) and (l,k) share a basis but NOT the
                // same integral, so the block must be re-formed for this
                // orientation. `oriented_ovov` does that from the stored one.
                let g = oriented_ovov(g_pno, basis, p_kl, k, l);
                let s = overlaps.get(p_kl, p_il);
                let t = oriented_amp(t2_pno, basis, p_il, i, l);
                // tr[ ĝᵀ · (S t̃ Sᵀ) ] summed elementwise.
                let conv = s.dot(&t).dot(&s.t());
                acc += g.iter().zip(conv.iter()).map(|(x, y)| x * y).sum::<f64>();
            }
            a[(k, i)] = acc;
        }
    }
    Ok(a)
}

/// The dense oracle for [`pno_foo_direct`]: `A[k,i] = Σ_{lcd} (kc|ld) t2[i,l,c,d]`.
///
/// Public because it is the honest way for a caller to check a truncated run
/// against the untruncated answer on its own system rather than trusting a
/// threshold. This is exactly `einsum!("kcld,ilcd->ki", ovov, t2)` from
/// [`crate::ccsd_closed_shell`], written as explicit loops so the index
/// convention cannot drift between the two.
pub fn dense_foo_direct(ovov: &Array4<f64>, t2: &Array4<f64>) -> Result<Array2<f64>, FerricError> {
    let (no, nv, no2, nv2) = ovov.dim();
    if no != no2 || nv != nv2 {
        return Err(FerricError::General(format!(
            "dense_foo_direct: ovov is {:?}, expected (no, nv, no, nv)",
            ovov.dim()
        )));
    }
    if t2.dim() != (no, no, nv, nv) {
        return Err(FerricError::General(format!(
            "dense_foo_direct: t2 is {:?}, expected ({no}, {no}, {nv}, {nv})",
            t2.dim()
        )));
    }
    let mut a = Array2::<f64>::zeros((no, no));
    for k in 0..no {
        for i in 0..no {
            let mut acc = 0.0;
            for l in 0..no {
                for c in 0..nv {
                    for d in 0..nv {
                        acc += ovov[[k, c, l, d]] * t2[[i, l, c, d]];
                    }
                }
            }
            a[(k, i)] = acc;
        }
    }
    Ok(a)
}

// ---- orientation helpers -------------------------------------------------
//
// `basis.pairs` stores each pair once as (i,j) with i <= j. A residual indexed
// over the full nocc² grid asks for both orientations. The PNO *basis* is shared
// (D^{ji} is the symmetrized transpose of D^{ij}, same eigenvectors) but the
// stored *blocks* are not symmetric, so the orientation must be applied
// explicitly. Getting this wrong is a plausible-but-wrong energy, not a crash,
// which is why it lives in two named functions rather than inline transposes.

/// `t̃^{(x,y)}` for the requested orientation, from the block stored for the
/// canonical `(i,j)` ordering.
///
/// `t2[i,j,a,b] = t2[j,i,b,a]` is a structural identity of closed-shell CCSD, so
/// the mirror block is the transpose in the shared basis.
fn oriented_amp(
    t2_pno: &[Array2<f64>],
    basis: &PairPnoBasis,
    p: usize,
    x: usize,
    _y: usize,
) -> Array2<f64> {
    let (i, _j) = basis.pairs[p].ij;
    if x == i {
        t2_pno[p].clone()
    } else {
        t2_pno[p].t().to_owned()
    }
}

/// `ĝ^{(x,y)}[c̄,d̄] = (x c̄ | y d̄)` for the requested orientation.
///
/// `(kc|ld) = (ld|kc)`, so the mirror of the stored block is its transpose —
/// the same relation as for the amplitudes, and it holds in the PNO basis
/// because both orientations are rotated by the same `Q`.
fn oriented_ovov(
    g_pno: &PairOvov,
    basis: &PairPnoBasis,
    p: usize,
    x: usize,
    _y: usize,
) -> Array2<f64> {
    let (k, _l) = basis.pairs[p].ij;
    if x == k {
        g_pno.get(p).clone()
    } else {
        g_pno.get(p).t().to_owned()
    }
}

// =====================================================================
// STAGE C — the remaining pair-coupling contractions
// =====================================================================

/// `F_oo` exchange term: `A[k,i] = Σ_{lcd} (kd|lc) · t2[i,l,c,d]`.
///
/// Identical structure to [`pno_foo_direct`] with the integral's two virtual
/// labels swapped — in the dense code this is the `'kdlc,ilcd->ki'` einsum,
/// which reads the *same* `ovov` block under a different label assignment. In
/// the PNO basis that swap is the transpose of `ĝ^{kl}`, so the same `S`
/// insertion carries over unchanged.
pub fn pno_foo_exchange(
    t2_pno: &[Array2<f64>],
    g_pno: &PairOvov,
    basis: &PairPnoBasis,
    overlaps: &PairOverlaps,
    index: &PairIndex,
) -> Result<Array2<f64>, FerricError> {
    if t2_pno.len() != basis.pairs.len() {
        return Err(FerricError::General(format!(
            "pno_foo_exchange: got {} amplitude blocks for {} pairs",
            t2_pno.len(),
            basis.pairs.len()
        )));
    }
    let nocc = index.nocc();
    let mut a = Array2::<f64>::zeros((nocc, nocc));
    for k in 0..nocc {
        for i in 0..nocc {
            let mut acc = 0.0;
            for l in 0..nocc {
                let (Some(p_kl), Some(p_il)) = (index.get(k, l), index.get(i, l)) else {
                    continue;
                };
                // (kd|lc): the same block with its virtual labels swapped, i.e.
                // ĝ transposed. Both orientations share Q, so the transpose is
                // taken AFTER the rotation, which is legitimate precisely
                // because Qᵀ (Gᵀ) Q = (Qᵀ G Q)ᵀ.
                let g = oriented_ovov(g_pno, basis, p_kl, k, l).t().to_owned();
                let s = overlaps.get(p_kl, p_il);
                let t = oriented_amp(t2_pno, basis, p_il, i, l);
                let conv = s.dot(&t).dot(&s.t());
                acc += g.iter().zip(conv.iter()).map(|(x, y)| x * y).sum::<f64>();
            }
            a[(k, i)] = acc;
        }
    }
    Ok(a)
}

/// Dense oracle for [`pno_foo_exchange`]: `Σ_{lcd} (kd|lc) t2[i,l,c,d]`.
pub fn dense_foo_exchange(
    ovov: &Array4<f64>,
    t2: &Array4<f64>,
) -> Result<Array2<f64>, FerricError> {
    let (no, nv, _, _) = ovov.dim();
    let mut a = Array2::<f64>::zeros((no, no));
    for k in 0..no {
        for i in 0..no {
            let mut acc = 0.0;
            for l in 0..no {
                for c in 0..nv {
                    for d in 0..nv {
                        acc += ovov[[k, d, l, c]] * t2[[i, l, c, d]];
                    }
                }
            }
            a[(k, i)] = acc;
        }
    }
    Ok(a)
}

/// `F_vv` direct term: `A[c,a] = Σ_{k,l,d} (kc|ld) · t2[k,l,a,d]`.
///
/// **This one is different in kind from `F_oo`, and that difference is the honest
/// limitation of a per-pair virtual basis.** The output carries *free* virtual
/// indices `c` and `a`, which belong to no single pair: `c` comes from the
/// integral's pair `(k,l)` and `a` from the amplitude's pair `(k,l)` — here the
/// same pair, but the result is accumulated over all of them. So `F_vv` cannot
/// stay in any one PNO basis; it must be assembled in the **canonical** virtual
/// basis, with each pair's contribution back-transformed by its own `Q`:
///
/// ```text
///   A[c,a] = Σ_{(k,l)} Σ_{c̄,ā,d̄}  Q_kl[c,c̄] ĝ^{kl}[c̄,d̄] t̃^{kl}[ā,d̄] Q_kl[a,ā]
/// ```
///
/// which is `Σ_{(k,l)} Q_kl · (ĝ^{kl} t̃^{kl}ᵀ) · Q_klᵀ`. No `S` appears because
/// integral and amplitude share the pair here — the coupling is between the pair
/// basis and the *canonical* basis instead, and the back-transform `Q` plays the
/// role `S` plays elsewhere. Cost is `O(npno² · nvir)` per pair for the
/// back-transform rather than `O(nvir³)`, so truncation still bites, but the
/// `nvir²` output is unavoidable. A production DLPNO-CCSD keeps `F_vv` projected
/// into each consuming pair's basis to avoid even that; doing so here would mean
/// restructuring the consumer, which is Stage D work.
pub fn pno_fvv_direct(
    t2_pno: &[Array2<f64>],
    g_pno: &PairOvov,
    basis: &PairPnoBasis,
    index: &PairIndex,
) -> Result<Array2<f64>, FerricError> {
    if t2_pno.len() != basis.pairs.len() {
        return Err(FerricError::General(format!(
            "pno_fvv_direct: got {} amplitude blocks for {} pairs",
            t2_pno.len(),
            basis.pairs.len()
        )));
    }
    let nvir = basis.nvir;
    let nocc = index.nocc();
    let mut a = Array2::<f64>::zeros((nvir, nvir));
    for k in 0..nocc {
        for l in 0..nocc {
            let Some(p) = index.get(k, l) else { continue };
            let q = &basis.pairs[p].transform;
            let g = oriented_ovov(g_pno, basis, p, k, l); // ĝ[c̄,d̄]
            let t = oriented_amp(t2_pno, basis, p, k, l); // t̃[ā,d̄]
            // M[c̄,ā] = Σ_d̄ ĝ[c̄,d̄] t̃[ā,d̄]
            let m = g.dot(&t.t());
            a = a + q.dot(&m).dot(&q.t());
        }
    }
    Ok(a)
}

/// Dense oracle for [`pno_fvv_direct`]: `Σ_{kld} (kc|ld) t2[k,l,a,d]`, stored
/// `[c,a]` exactly as the dense `einsum!("kcld,klad->ca", …)` emits it.
pub fn dense_fvv_direct(ovov: &Array4<f64>, t2: &Array4<f64>) -> Result<Array2<f64>, FerricError> {
    let (no, nv, _, _) = ovov.dim();
    let mut a = Array2::<f64>::zeros((nv, nv));
    for c in 0..nv {
        for av in 0..nv {
            let mut acc = 0.0;
            for k in 0..no {
                for l in 0..no {
                    for d in 0..nv {
                        acc += ovov[[k, c, l, d]] * t2[[k, l, av, d]];
                    }
                }
            }
            a[(c, av)] = acc;
        }
    }
    Ok(a)
}

/// `W_oooo` amplitude term: `W[k,l,i,j] = Σ_{cd} (kc|ld) · t2[i,j,c,d]`.
///
/// The purest two-pair coupling in the residual: pair `(k,l)` supplies the
/// integral and pair `(i,j)` the amplitude, with **all four** occupied indices
/// free, so the two pairs are genuinely unrelated. The output is a scalar per
/// `(k,l,i,j)`, so no back-transform is needed and the whole thing stays inside
/// the PNO spaces:
///
/// ```text
///   W[k,l,i,j] = tr[ ĝ^{kl} · ( S^{kl,ij} t̃^{ij} (S^{kl,ij})ᵀ ) ]
/// ```
///
/// This is the contraction the DLPNO literature calls the hh ladder, and it is
/// the one [`ferric_mp2::pair_domains`] screens: with pair domains the `(i,j)`
/// loop runs only over pairs coupled to `(k,l)` rather than all of them.
pub fn pno_woooo_amp(
    t2_pno: &[Array2<f64>],
    g_pno: &PairOvov,
    basis: &PairPnoBasis,
    overlaps: &PairOverlaps,
    index: &PairIndex,
) -> Result<Array4<f64>, FerricError> {
    if t2_pno.len() != basis.pairs.len() {
        return Err(FerricError::General(format!(
            "pno_woooo_amp: got {} amplitude blocks for {} pairs",
            t2_pno.len(),
            basis.pairs.len()
        )));
    }
    let nocc = index.nocc();
    let mut w = Array4::<f64>::zeros((nocc, nocc, nocc, nocc));
    for k in 0..nocc {
        for l in 0..nocc {
            let Some(p_kl) = index.get(k, l) else { continue };
            let g = oriented_ovov(g_pno, basis, p_kl, k, l);
            for i in 0..nocc {
                for j in 0..nocc {
                    let Some(p_ij) = index.get(i, j) else { continue };
                    let s = overlaps.get(p_kl, p_ij);
                    let t = oriented_amp(t2_pno, basis, p_ij, i, j);
                    let conv = s.dot(&t).dot(&s.t());
                    w[[k, l, i, j]] =
                        g.iter().zip(conv.iter()).map(|(x, y)| x * y).sum::<f64>();
                }
            }
        }
    }
    Ok(w)
}

/// Dense oracle for [`pno_woooo_amp`]: `Σ_{cd} (kc|ld) t2[i,j,c,d]`.
pub fn dense_woooo_amp(ovov: &Array4<f64>, t2: &Array4<f64>) -> Result<Array4<f64>, FerricError> {
    let (no, nv, _, _) = ovov.dim();
    let mut w = Array4::<f64>::zeros((no, no, no, no));
    for k in 0..no {
        for l in 0..no {
            for i in 0..no {
                for j in 0..no {
                    let mut acc = 0.0;
                    for c in 0..nv {
                        for d in 0..nv {
                            acc += ovov[[k, c, l, d]] * t2[[i, j, c, d]];
                        }
                    }
                    w[[k, l, i, j]] = acc;
                }
            }
        }
    }
    Ok(w)
}

/// `W_voov` amplitude term: `A[k,c,i,a] = Σ_{l,d} (ld|kc) · t2[i,l,a,d]`.
///
/// The hardest shape in this set: it couples pair `(l,k)` (integral) to pair
/// `(i,l)` (amplitude) *and* leaves one virtual index free on each side (`c` from
/// the integral, `a` from the amplitude). So it needs **both** an `S` insertion
/// on the summed virtual `d` and a back-transform to canonical on the two free
/// virtuals:
///
/// ```text
///   A[k,c,i,a] = Σ_l Σ_{c̄,d̄,ā,d̃}
///        Q_lk[c,c̄] ĝ^{lk}[d̄,c̄] S^{lk,il}[d̄,d̃] t̃^{il}[ā,d̃] Q_il[a,ā]
/// ```
///
/// Note the two transforms are *different* matrices (`Q_lk` and `Q_il`), which is
/// exactly why the naive "just rotate everything by one Q" shortcut fails and why
/// this contraction is the real test of the construction.
///
/// The output is `nocc²·nvir²` — dense on the virtuals by necessity, because a
/// residual consumer indexed `[k,c,i,a]` belongs to no pair. Stage D would keep
/// it per-pair; this signature exists to be *checked against the dense einsum*,
/// which requires emitting the dense shape.
pub fn pno_wvoov_amp(
    t2_pno: &[Array2<f64>],
    g_pno: &PairOvov,
    basis: &PairPnoBasis,
    overlaps: &PairOverlaps,
    index: &PairIndex,
) -> Result<Array4<f64>, FerricError> {
    if t2_pno.len() != basis.pairs.len() {
        return Err(FerricError::General(format!(
            "pno_wvoov_amp: got {} amplitude blocks for {} pairs",
            t2_pno.len(),
            basis.pairs.len()
        )));
    }
    let nocc = index.nocc();
    let nvir = basis.nvir;
    let mut a = Array4::<f64>::zeros((nocc, nvir, nocc, nvir));
    for k in 0..nocc {
        for i in 0..nocc {
            let mut acc = Array2::<f64>::zeros((nvir, nvir)); // [c, a]
            for l in 0..nocc {
                let (Some(p_lk), Some(p_il)) = (index.get(l, k), index.get(i, l)) else {
                    continue;
                };
                // ĝ^{lk}[d̄,c̄] = (l d̄ | k c̄) — the (l,k) orientation.
                let g = oriented_ovov(g_pno, basis, p_lk, l, k);
                let s = overlaps.get(p_lk, p_il); // [d̄, d̃]
                let t = oriented_amp(t2_pno, basis, p_il, i, l); // [ā, d̃]
                // M[c̄, ā] = Σ_{d̄,d̃} ĝ[d̄,c̄] S[d̄,d̃] t̃[ā,d̃]
                let sd = s.dot(&t.t()); // [d̄, ā]
                let m = g.t().dot(&sd); // [c̄, ā]
                // back-transform: Q_lk on c̄, Q_il on ā — DIFFERENT matrices.
                let q_c = &basis.pairs[p_lk].transform;
                let q_a = &basis.pairs[p_il].transform;
                acc = acc + q_c.dot(&m).dot(&q_a.t());
            }
            for c in 0..nvir {
                for av in 0..nvir {
                    a[[k, c, i, av]] = acc[(c, av)];
                }
            }
        }
    }
    Ok(a)
}

/// Dense oracle for [`pno_wvoov_amp`]: `Σ_{ld} (ld|kc) t2[i,l,a,d]`, stored
/// `[k,c,i,a]` exactly as the dense `einsum!("ldkc,ilad->kcia", …)` emits it.
pub fn dense_wvoov_amp(ovov: &Array4<f64>, t2: &Array4<f64>) -> Result<Array4<f64>, FerricError> {
    let (no, nv, _, _) = ovov.dim();
    let mut a = Array4::<f64>::zeros((no, nv, no, nv));
    for k in 0..no {
        for c in 0..nv {
            for i in 0..no {
                for av in 0..nv {
                    let mut acc = 0.0;
                    for l in 0..no {
                        for d in 0..nv {
                            acc += ovov[[l, d, k, c]] * t2[[i, l, av, d]];
                        }
                    }
                    a[[k, c, i, av]] = acc;
                }
            }
        }
    }
    Ok(a)
}

/// `W_vovo` amplitude term: `A[c,k,i,a] = Σ_{l,d} (lc|kd) · t2[i,l,d,a]`.
///
/// Same two-pair structure as [`pno_wvoov_amp`], but the amplitude's free and
/// summed virtual labels are swapped (`t2[i,l,d,a]` rather than `t2[i,l,a,d]`),
/// which in the PNO basis is a transpose of `t̃`. Included because it is the
/// second of the two `W` shapes the T2 equation needs and because getting the
/// transpose wrong here is exactly the kind of silent error the module exists to
/// rule out.
pub fn pno_wvovo_amp(
    t2_pno: &[Array2<f64>],
    g_pno: &PairOvov,
    basis: &PairPnoBasis,
    overlaps: &PairOverlaps,
    index: &PairIndex,
) -> Result<Array4<f64>, FerricError> {
    if t2_pno.len() != basis.pairs.len() {
        return Err(FerricError::General(format!(
            "pno_wvovo_amp: got {} amplitude blocks for {} pairs",
            t2_pno.len(),
            basis.pairs.len()
        )));
    }
    let nocc = index.nocc();
    let nvir = basis.nvir;
    let mut a = Array4::<f64>::zeros((nvir, nocc, nocc, nvir));
    for k in 0..nocc {
        for i in 0..nocc {
            let mut acc = Array2::<f64>::zeros((nvir, nvir)); // [c, a]
            for l in 0..nocc {
                let (Some(p_lk), Some(p_il)) = (index.get(l, k), index.get(i, l)) else {
                    continue;
                };
                // ĝ^{lk}[c̄,d̄] = (l c̄ | k d̄).
                let g = oriented_ovov(g_pno, basis, p_lk, l, k);
                let s = overlaps.get(p_lk, p_il); // [d̄, d̃]
                // t̃[d̃, ā]. NOTE: NO transpose. `oriented_amp` returns the block
                // indexed [first-virtual, second-virtual] of `t2[i,l,·,·]`, and
                // this contraction wants `t2[i,l,d,a]` — the summed `d` IS the
                // first slot here, unlike `pno_wvoov_amp`'s `t2[i,l,a,d]`. An
                // erroneous `.t()` here was MEASURED at 9.1e-1 deviation against
                // the dense oracle (scale 2.8) — it does not shrink the answer,
                // it silently produces a different one.
                let t = oriented_amp(t2_pno, basis, p_il, i, l);
                let sd = s.dot(&t); // [d̄, ā]
                let m = g.dot(&sd); // [c̄, ā]
                let q_c = &basis.pairs[p_lk].transform;
                let q_a = &basis.pairs[p_il].transform;
                acc = acc + q_c.dot(&m).dot(&q_a.t());
            }
            for c in 0..nvir {
                for av in 0..nvir {
                    a[[c, k, i, av]] = acc[(c, av)];
                }
            }
        }
    }
    Ok(a)
}

/// Dense oracle for [`pno_wvovo_amp`]: `Σ_{ld} (lc|kd) t2[i,l,d,a]`, stored
/// `[c,k,i,a]` as the dense `einsum!("lckd,ilda->ckia", …)` emits it.
pub fn dense_wvovo_amp(ovov: &Array4<f64>, t2: &Array4<f64>) -> Result<Array4<f64>, FerricError> {
    let (no, nv, _, _) = ovov.dim();
    let mut a = Array4::<f64>::zeros((nv, no, no, nv));
    for c in 0..nv {
        for k in 0..no {
            for i in 0..no {
                for av in 0..nv {
                    let mut acc = 0.0;
                    for l in 0..no {
                        for d in 0..nv {
                            acc += ovov[[l, c, k, d]] * t2[[i, l, d, av]];
                        }
                    }
                    a[[c, k, i, av]] = acc;
                }
            }
        }
    }
    Ok(a)
}

/// `L_oo`-driven T2 term: `R[i,j,ã,b̃] = − Σ_k L[k,i] · t2[k,j,ã,b̃]`, kept
/// **entirely inside pair `(i,j)`'s PNO basis**.
///
/// This is the shape that shows the payoff most directly: the output is a `t2`
/// residual, so it belongs to pair `(i,j)`, and the input amplitude belongs to
/// pair `(k,j)`. One `S` converts, the virtual indices never leave the PNO
/// spaces, and the working set is `npno²` rather than `nvir²`:
///
/// ```text
///   R̃^{ij}[ã,b̃] = − Σ_k L[k,i] · ( S^{ij,kj} t̃^{kj} (S^{ij,kj})ᵀ )[ã,b̃]
/// ```
///
/// Returned per-pair, in [`PairPnoBasis::pairs`] order — i.e. this function's
/// output is already in the right form to be divided by [`pno_denominators`] and
/// fed back as the next iterate, which is what Stage D would do.
pub fn pno_loo_t2_term(
    loo: &Array2<f64>,
    t2_pno: &[Array2<f64>],
    basis: &PairPnoBasis,
    overlaps: &PairOverlaps,
    index: &PairIndex,
) -> Result<Vec<Array2<f64>>, FerricError> {
    let nocc = index.nocc();
    if loo.dim() != (nocc, nocc) {
        return Err(FerricError::General(format!(
            "pno_loo_t2_term: loo is {:?}, expected ({nocc}, {nocc})",
            loo.dim()
        )));
    }
    if t2_pno.len() != basis.pairs.len() {
        return Err(FerricError::General(format!(
            "pno_loo_t2_term: got {} amplitude blocks for {} pairs",
            t2_pno.len(),
            basis.pairs.len()
        )));
    }
    let mut out = Vec::with_capacity(basis.pairs.len());
    for (p_ij, pair) in basis.pairs.iter().enumerate() {
        let (i, j) = pair.ij;
        let npno = pair.transform.ncols();
        let mut r = Array2::<f64>::zeros((npno, npno));
        for k in 0..nocc {
            let Some(p_kj) = index.get(k, j) else { continue };
            let s = overlaps.get(p_ij, p_kj);
            let t = oriented_amp(t2_pno, basis, p_kj, k, j);
            let conv = s.dot(&t).dot(&s.t());
            r = r - loo[(k, i)] * &conv;
        }
        out.push(r);
    }
    Ok(out)
}

/// Dense oracle for [`pno_loo_t2_term`]: `−Σ_k L[k,i] t2[k,j,a,b]`, dense over
/// the full `nocc²` grid.
pub fn dense_loo_t2_term(loo: &Array2<f64>, t2: &Array4<f64>) -> Result<Array4<f64>, FerricError> {
    let (no, _, nv, _) = t2.dim();
    let mut r = Array4::<f64>::zeros((no, no, nv, nv));
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let mut acc = 0.0;
                    for k in 0..no {
                        acc += loo[(k, i)] * t2[[k, j, a, b]];
                    }
                    r[[i, j, a, b]] = -acc;
                }
            }
        }
    }
    Ok(r)
}

// =====================================================================
// COST — structural, not wall clock
// =====================================================================

/// FLOP counts for the contractions in this module, as **pure functions of the
/// basis** — no arithmetic is performed to obtain them.
///
/// This is the module's cost claim in full. A wall-clock measurement on a
/// contested machine would confound the algorithmic change with BLAS shape
/// effects and scheduler noise; a count derived from `npno` alone cannot. Each
/// field pairs the PNO count with the dense count for the identical contraction,
/// so the ratio is the honest compression.
///
/// Counts are leading-order multiply-adds, both columns derived the same way, so
/// systematic factors cancel in the ratio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlopCount {
    /// `Σ_pairs npno²` versus `n_pairs · nvir²` — the amplitude working set.
    pub amplitude_elements: (usize, usize),
    /// `Σ_pairs npno²` versus `n_pairs · nvir²` — the per-pair integral storage.
    pub integral_elements: (usize, usize),
    /// `F_oo`-shaped: per `(k,i,l)` triple, `S t̃ Sᵀ` (two GEMMs) plus a trace.
    pub foo_flops: (usize, usize),
    /// `W_oooo`-shaped (the hh ladder): per `(k,l,i,j)`, the same conversion.
    pub woooo_flops: (usize, usize),
    /// `L_oo`-driven T2 term: per `(pair, k)`, one conversion.
    pub loo_t2_flops: (usize, usize),
}

impl FlopCount {
    /// Derive the counts from a basis and an occupied count.
    ///
    /// The dense column is computed for the SAME pair list, so the comparison
    /// isolates *virtual* truncation and does not silently credit the pair
    /// screen — that is [`ferric_mp2::pair_domains`]'s separate claim.
    pub fn of(basis: &PairPnoBasis, nocc: usize) -> Self {
        let nv = basis.nvir;
        let npno: Vec<usize> = basis.pairs.iter().map(|p| p.transform.ncols()).collect();
        let n_pairs = npno.len();

        let amp_pno: usize = npno.iter().map(|&n| n * n).sum();
        let amp_dense = n_pairs * nv * nv;

        // Conversion S t̃ Sᵀ for a pair of PNO dimensions (m, n):
        //   S(m×n) · t̃(n×n) -> m·n·n, then ·Sᵀ(n×m) -> m·m·n, plus an m² trace.
        // The dense analogue is the same expression at m = n = nvir, which is
        // what the ORIGINAL einsum does per occupied index tuple.
        let conv = |m: usize, n: usize| m * n * n + m * m * n + m * m;
        let dense_conv = conv(nv, nv);

        // F_oo: (k,i,l) triples that resolve to a retained pair on both sides.
        // Counting them exactly rather than assuming nocc³ keeps the pair screen
        // out of the comparison (both columns use the same triple count).
        let mut foo_pno = 0usize;
        let mut foo_n = 0usize;
        let idx = pair_map(basis, nocc);
        for k in 0..nocc {
            for i in 0..nocc {
                for l in 0..nocc {
                    let (Some(a), Some(b)) = (idx.get(&(k, l)), idx.get(&(i, l))) else {
                        continue;
                    };
                    foo_pno += conv(npno[*a], npno[*b]);
                    foo_n += 1;
                }
            }
        }

        let mut w_pno = 0usize;
        let mut w_n = 0usize;
        for k in 0..nocc {
            for l in 0..nocc {
                for i in 0..nocc {
                    for j in 0..nocc {
                        let (Some(a), Some(b)) = (idx.get(&(k, l)), idx.get(&(i, j))) else {
                            continue;
                        };
                        w_pno += conv(npno[*a], npno[*b]);
                        w_n += 1;
                    }
                }
            }
        }

        let mut l_pno = 0usize;
        let mut l_n = 0usize;
        for (p_ij, pair) in basis.pairs.iter().enumerate() {
            let (_, j) = pair.ij;
            for k in 0..nocc {
                let Some(b) = idx.get(&(k, j)) else { continue };
                l_pno += conv(npno[p_ij], npno[*b]);
                l_n += 1;
            }
        }

        Self {
            amplitude_elements: (amp_pno, amp_dense),
            integral_elements: (amp_pno, amp_dense),
            foo_flops: (foo_pno, foo_n * dense_conv),
            woooo_flops: (w_pno, w_n * dense_conv),
            loo_t2_flops: (l_pno, l_n * dense_conv),
        }
    }

    /// PNO cost as a fraction of dense, per row. 1.0 = no compression.
    pub fn ratios(&self) -> [f64; 5] {
        let r = |(a, b): (usize, usize)| if b == 0 { 1.0 } else { a as f64 / b as f64 };
        [
            r(self.amplitude_elements),
            r(self.integral_elements),
            r(self.foo_flops),
            r(self.woooo_flops),
            r(self.loo_t2_flops),
        ]
    }

    /// A human-readable table, for a caller that wants to report its own numbers.
    pub fn table(&self) -> String {
        let rows: [(&str, (usize, usize)); 5] = [
            ("amplitude elements", self.amplitude_elements),
            ("integral elements ", self.integral_elements),
            ("F_oo flops        ", self.foo_flops),
            ("W_oooo flops      ", self.woooo_flops),
            ("L_oo T2 flops     ", self.loo_t2_flops),
        ];
        let mut s = String::from("  quantity              PNO           dense      ratio\n");
        for (name, (a, b)) in rows {
            let ratio = if b == 0 { 1.0 } else { a as f64 / b as f64 };
            s.push_str(&format!("  {name}  {a:>12}  {b:>12}   {ratio:>6.4}\n"));
        }
        s
    }
}

fn pair_map(basis: &PairPnoBasis, _nocc: usize) -> HashMap<(usize, usize), usize> {
    let mut m = HashMap::new();
    for (p, pair) in basis.pairs.iter().enumerate() {
        let (i, j) = pair.ij;
        m.insert((i, j), p);
        m.insert((j, i), p);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dlpno_ccsd_virtual::t2_to_pno;
    use ferric_mp2::pair_domains::{build_pair_domains, complete_pair_domains};

    // ------------------------------------------------------------------
    // Deterministic toy system, deliberately small: every claim in this
    // module is about EXACTNESS or a COUNT, neither of which needs size,
    // and the box is contested.
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

    /// Chemist `(ia|jb)` at `[i,a,j,b]`, symmetric under `(ia) <-> (jb)` exactly
    /// as the real block is — the symmetry several orientation helpers rely on.
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

    struct Fixture {
        basis: PairPnoBasis,
        overlaps: PairOverlaps,
        index: PairIndex,
        g_pno: PairOvov,
        t2_pno: Vec<Array2<f64>>,
        t2: Array4<f64>,
        ovov: Array4<f64>,
    }

    fn fixture(nocc: usize, nvir: usize, t_cut: f64) -> Fixture {
        let (eo, ev) = (eps_occ(nocc), eps_vir(nvir));
        let ovov = ovov_block(nocc, nvir);
        let t2 = mp2_t2(&ovov, &eo, &ev);
        let d = complete_pair_domains(&line_centers(nocc, 1.5)).unwrap();
        let basis = PairPnoBasis::build(&d, nvir, &ev, t_cut, |i, j| {
            Array2::from_shape_fn((nvir, nvir), |(a, b)| t2[[i, j, a, b]])
        })
        .unwrap();
        let overlaps = PairOverlaps::build(&basis);
        let index = PairIndex::new(&basis, nocc).unwrap();
        let g_pno = PairOvov::build(&ovov, &basis).unwrap();
        let t2_pno = t2_to_pno(&t2, &basis).unwrap();
        Fixture { basis, overlaps, index, g_pno, t2_pno, t2, ovov }
    }

    fn max_dev2(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
    }
    fn max_dev4(a: &Array4<f64>, b: &Array4<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
    }
    fn scale2(a: &Array2<f64>) -> f64 {
        a.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
    }
    fn scale4(a: &Array4<f64>) -> f64 {
        a.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
    }

    // =================== STAGE A ===================

    /// THE INVARIANT THE WHOLE REWRITE RESTS ON: at `t_cut_pno = 0` every
    /// pair-pair overlap `S^{P,Q}` is a square ORTHOGONAL matrix.
    ///
    /// Every PNO contraction below inserts `S … Sᵀ` where the dense one has a
    /// plain virtual sum. That insertion reduces to the identity — i.e. the PNO
    /// contraction *is* the dense one — exactly when `S Sᵀ = 1`. If this fails,
    /// no later exactness result is meaningful, so it is checked elementwise
    /// over every pair combination rather than through a norm.
    #[test]
    fn stage_a_overlaps_are_orthogonal_at_zero_truncation() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 0.0);
        assert!(f.basis.is_complete(), "t_cut_pno = 0 must keep every virtual");

        let worst = f.overlaps.max_nonorthogonality();
        eprintln!("stage A: max |S Sᵀ - 1| over all pair pairs = {worst:.3e}");
        assert!(worst < 1e-10, "pair-pair overlaps are not orthogonal: {worst:.3e}");

        // Every S must be square nvir × nvir at zero truncation.
        for p in 0..f.overlaps.n_pairs() {
            for q in 0..f.overlaps.n_pairs() {
                assert_eq!(f.overlaps.get(p, q).dim(), (nvir, nvir));
            }
        }
    }

    /// The PREMISE of the previous test: the off-diagonal `S` are genuinely
    /// NON-trivial rotations, not near-identity.
    ///
    /// Without this the exactness proofs downstream would be vacuous — if every
    /// pair happened to share PNOs, `S = 1` and the insertion would be untested.
    /// This measures how far from the identity the overlaps actually are.
    #[test]
    fn stage_a_off_diagonal_overlaps_are_not_the_identity() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 0.0);

        let mut worst = 0.0f64;
        for p in 0..f.overlaps.n_pairs() {
            for q in 0..f.overlaps.n_pairs() {
                if p == q {
                    // The self-overlap MUST be the identity: Qᵀ Q = 1.
                    let s = f.overlaps.get(p, p);
                    for a in 0..s.nrows() {
                        for b in 0..s.ncols() {
                            let want = if a == b { 1.0 } else { 0.0 };
                            assert!(
                                (s[(a, b)] - want).abs() < 1e-10,
                                "S^{{P,P}} is not the identity at ({a},{b})"
                            );
                        }
                    }
                    continue;
                }
                let s = f.overlaps.get(p, q);
                for a in 0..s.nrows() {
                    for b in 0..s.ncols() {
                        let want = if a == b { 1.0 } else { 0.0 };
                        worst = worst.max((s[(a, b)] - want).abs());
                    }
                }
            }
        }
        eprintln!("stage A: max |S^(P!=Q) - 1| = {worst:.3e} (must be LARGE)");
        assert!(
            worst > 1e-2,
            "premise failed: off-diagonal pair overlaps are ~identity ({worst:.3e}), so the \
             S-insertion is never exercised and every exactness test below is vacuous"
        );
    }

    /// Truncation must make `S Sᵀ` measurably non-orthogonal — that quantity IS
    /// the approximation the method introduces, so an inert knob would mean the
    /// whole cost/accuracy trade is fictional.
    #[test]
    fn stage_a_truncation_breaks_overlap_orthogonality() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 1e-3);
        assert!(!f.basis.is_complete(), "test premise: something must be truncated");
        let worst = f.overlaps.max_nonorthogonality();
        eprintln!("stage A: truncated max |S Sᵀ - 1| = {worst:.3e}");
        assert!(worst > 1e-10, "truncation left the overlaps orthogonal — the knob is inert");
    }

    /// `PairIndex` must resolve both orientations to the same basis, and screened
    /// pairs to `None`.
    #[test]
    fn stage_a_pair_index_resolves_mirrors_and_screens() {
        let (nocc, nvir) = (4, 5);
        let ev = eps_vir(nvir);
        let ovov = ovov_block(nocc, nvir);
        let t2 = mp2_t2(&ovov, &eps_occ(nocc), &ev);
        let centers = line_centers(nocc, 10.0);
        let screened = build_pair_domains(&centers, 15.0, f64::INFINITY).unwrap();
        let all = complete_pair_domains(&centers).unwrap();
        assert!(screened.pairs.len() < all.pairs.len(), "test premise");

        let basis = PairPnoBasis::build(&screened, nvir, &ev, 0.0, |i, j| {
            Array2::from_shape_fn((nvir, nvir), |(a, b)| t2[[i, j, a, b]])
        })
        .unwrap();
        let idx = PairIndex::new(&basis, nocc).unwrap();

        for &(i, j) in &screened.pairs {
            assert_eq!(idx.get(i, j), idx.get(j, i), "mirror ({j},{i}) resolves elsewhere");
            assert!(idx.get(i, j).is_some());
        }
        let mut n_missing = 0;
        for i in 0..nocc {
            for j in 0..nocc {
                let kept = screened.pairs.contains(&(i.min(j), i.max(j)));
                if !kept {
                    assert!(idx.get(i, j).is_none(), "screened pair ({i},{j}) has an entry");
                    n_missing += 1;
                }
            }
        }
        assert!(n_missing > 0, "test premise: some pairs must be screened");
    }

    // =================== STAGE B — THE PROOF OF CONCEPT ===================

    /// **STAGE B, THE LOAD-BEARING RESULT.** The `F_oo` direct contraction
    /// `Σ_lcd (kc|ld) t2[i,l,c,d]`, rewritten with a pair-pair overlap inserted
    /// on every summed virtual, must reproduce the dense `einsum!` at
    /// `t_cut_pno = 0`.
    ///
    /// This is the contraction that couples pair `(k,l)` to pair `(i,l)` — two
    /// genuinely different virtual subspaces (pinned non-trivial by
    /// `stage_a_off_diagonal_overlaps_are_not_the_identity`). If the `S`
    /// insertion were wrong, or transposed, or applied on the wrong index, the
    /// result would be a plausible matrix of the right shape rather than a
    /// crash — so it is compared elementwise against the dense oracle.
    ///
    /// If THIS fails, the whole approach is wrong and the remaining
    /// contractions are not worth attempting.
    #[test]
    fn stage_b_foo_direct_is_exact_at_zero_truncation() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 0.0);

        let dense = dense_foo_direct(&f.ovov, &f.t2).unwrap();
        let pno =
            pno_foo_direct(&f.t2_pno, &f.g_pno, &f.basis, &f.overlaps, &f.index).unwrap();

        let worst = max_dev2(&dense, &pno);
        let scale = scale2(&dense);
        eprintln!("stage B: max |F_oo(dense) - F_oo(PNO)| = {worst:.3e} (scale {scale:.3e})");
        assert!(scale > 1e-3, "F_oo is ~zero ({scale:.3e}) — the check is vacuous");
        assert!(worst < 1e-12, "S-inserted F_oo is NOT exact: max deviation {worst:.3e}");
    }

    /// Stage B's exactness must not depend on the amplitudes being the same MP2
    /// ones the PNOs were derived from.
    ///
    /// A converged CCSD `t2` is not, so an invariance that only held for the
    /// defining amplitudes would be a coincidence rather than the algebraic
    /// identity the module claims.
    #[test]
    fn stage_b_foo_direct_is_exact_for_unrelated_amplitudes() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 0.0);

        let mut s = 0xDEADBEEF12345678u64;
        let raw = Array4::from_shape_fn((nocc, nocc, nvir, nvir), |_| lcg(&mut s));
        // Symmetrize into a valid closed-shell t2 (t2[i,j,a,b] == t2[j,i,b,a]);
        // the orientation helpers use that identity, so feeding a tensor that
        // violates it would fail for a reason that is not a bug.
        let t2 = Array4::from_shape_fn((nocc, nocc, nvir, nvir), |(i, j, a, b)| {
            0.5 * (raw[[i, j, a, b]] + raw[[j, i, b, a]])
        });
        let t2_pno = t2_to_pno(&t2, &f.basis).unwrap();

        let dense = dense_foo_direct(&f.ovov, &t2).unwrap();
        let pno = pno_foo_direct(&t2_pno, &f.g_pno, &f.basis, &f.overlaps, &f.index).unwrap();
        let worst = max_dev2(&dense, &pno);
        eprintln!("stage B (unrelated t2): max deviation = {worst:.3e}");
        assert!(worst < 1e-12, "S-inserted F_oo failed on unrelated amplitudes: {worst:.3e}");
    }

    /// Truncation must actually change `F_oo` — otherwise the cost reduction
    /// reported below would be buying nothing, and the exactness test above
    /// would be pinning an inert knob.
    #[test]
    fn stage_b_truncation_changes_foo() {
        let (nocc, nvir) = (4, 6);
        let f0 = fixture(nocc, nvir, 0.0);
        let ft = fixture(nocc, nvir, 1e-3);
        assert!(!ft.basis.is_complete(), "test premise");

        let e0 = pno_foo_direct(&f0.t2_pno, &f0.g_pno, &f0.basis, &f0.overlaps, &f0.index)
            .unwrap();
        let et = pno_foo_direct(&ft.t2_pno, &ft.g_pno, &ft.basis, &ft.overlaps, &ft.index)
            .unwrap();
        let d = max_dev2(&e0, &et);
        eprintln!("stage B: truncated F_oo differs from exact by {d:.3e}");
        assert!(d > 1e-12, "truncation had no effect on F_oo");
    }

    // =================== STAGE C — the rest ===================

    /// `F_oo` exchange, `Σ_lcd (kd|lc) t2[i,l,c,d]`.
    #[test]
    fn stage_c_foo_exchange_is_exact() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 0.0);
        let dense = dense_foo_exchange(&f.ovov, &f.t2).unwrap();
        let pno =
            pno_foo_exchange(&f.t2_pno, &f.g_pno, &f.basis, &f.overlaps, &f.index).unwrap();
        let worst = max_dev2(&dense, &pno);
        eprintln!("stage C: F_oo exchange max deviation = {worst:.3e}");
        assert!(scale2(&dense) > 1e-3, "vacuous: F_oo exchange is ~zero");
        assert!(worst < 1e-12, "F_oo exchange is not exact: {worst:.3e}");
    }

    /// `F_vv`, `Σ_kld (kc|ld) t2[k,l,a,d]` — the shape whose output carries FREE
    /// virtual indices, so it must be assembled back in the canonical basis.
    ///
    /// This is a different construction from `F_oo` (per-pair back-transform
    /// rather than an `S` insertion) and is tested separately for that reason:
    /// the two could not share a bug.
    #[test]
    fn stage_c_fvv_direct_is_exact() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 0.0);
        let dense = dense_fvv_direct(&f.ovov, &f.t2).unwrap();
        let pno = pno_fvv_direct(&f.t2_pno, &f.g_pno, &f.basis, &f.index).unwrap();
        let worst = max_dev2(&dense, &pno);
        eprintln!("stage C: F_vv max deviation = {worst:.3e} (scale {:.3e})", scale2(&dense));
        assert!(scale2(&dense) > 1e-3, "vacuous: F_vv is ~zero");
        assert!(worst < 1e-12, "F_vv is not exact: {worst:.3e}");
    }

    /// `W_oooo`, `Σ_cd (kc|ld) t2[i,j,c,d]` — the hh ladder, the purest
    /// two-pair coupling in the residual (all four occupied indices free).
    #[test]
    fn stage_c_woooo_amp_is_exact() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 0.0);
        let dense = dense_woooo_amp(&f.ovov, &f.t2).unwrap();
        let pno = pno_woooo_amp(&f.t2_pno, &f.g_pno, &f.basis, &f.overlaps, &f.index).unwrap();
        let worst = max_dev4(&dense, &pno);
        eprintln!("stage C: W_oooo max deviation = {worst:.3e} (scale {:.3e})", scale4(&dense));
        assert!(scale4(&dense) > 1e-3, "vacuous: W_oooo is ~zero");
        assert!(worst < 1e-12, "W_oooo is not exact: {worst:.3e}");
    }

    /// `W_voov`, `Σ_ld (ld|kc) t2[i,l,a,d]` — the hardest shape: an `S`
    /// insertion on the summed virtual AND two DIFFERENT back-transforms
    /// (`Q_lk` on `c`, `Q_il` on `a`) on the free ones.
    ///
    /// The "just rotate everything by one Q" shortcut fails here specifically,
    /// so this test is the one that distinguishes a correct implementation from
    /// a plausible one.
    #[test]
    fn stage_c_wvoov_amp_is_exact() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 0.0);
        let dense = dense_wvoov_amp(&f.ovov, &f.t2).unwrap();
        let pno = pno_wvoov_amp(&f.t2_pno, &f.g_pno, &f.basis, &f.overlaps, &f.index).unwrap();
        let worst = max_dev4(&dense, &pno);
        eprintln!("stage C: W_voov max deviation = {worst:.3e} (scale {:.3e})", scale4(&dense));
        assert!(scale4(&dense) > 1e-3, "vacuous: W_voov is ~zero");
        assert!(worst < 1e-12, "W_voov is not exact: {worst:.3e}");
    }

    /// `W_vovo`, `Σ_ld (lc|kd) t2[i,l,d,a]` — same coupling as `W_voov` with the
    /// amplitude's virtual labels swapped, i.e. a transposed `t̃`.
    #[test]
    fn stage_c_wvovo_amp_is_exact() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 0.0);
        let dense = dense_wvovo_amp(&f.ovov, &f.t2).unwrap();
        let pno = pno_wvovo_amp(&f.t2_pno, &f.g_pno, &f.basis, &f.overlaps, &f.index).unwrap();
        let worst = max_dev4(&dense, &pno);
        eprintln!("stage C: W_vovo max deviation = {worst:.3e} (scale {:.3e})", scale4(&dense));
        assert!(scale4(&dense) > 1e-3, "vacuous: W_vovo is ~zero");
        assert!(worst < 1e-12, "W_vovo is not exact: {worst:.3e}");
    }

    /// `−Σ_k L[k,i] t2[k,j,a,b]` — the T2-residual shape, checked in the PNO
    /// basis WITHOUT back-transforming, because its output belongs to pair
    /// `(i,j)` and staying there is the entire point.
    ///
    /// Comparison is against the dense result rotated into the same pair basis,
    /// which at zero truncation is a lossless rotation.
    #[test]
    fn stage_c_loo_t2_term_is_exact_in_the_pair_basis() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 0.0);

        let mut s = 0x13579BDF2468ACE0u64;
        let loo = Array2::from_shape_fn((nocc, nocc), |_| lcg(&mut s));

        let dense = dense_loo_t2_term(&loo, &f.t2).unwrap();
        let pno = pno_loo_t2_term(&loo, &f.t2_pno, &f.basis, &f.overlaps, &f.index).unwrap();

        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for (p, pair) in f.basis.pairs.iter().enumerate() {
            let (i, j) = pair.ij;
            let q = &pair.transform;
            let blk = Array2::from_shape_fn((nvir, nvir), |(a, b)| dense[[i, j, a, b]]);
            let want = rotate_block(&blk, q);
            worst = worst.max(max_dev2(&want, &pno[p]));
            scale = scale.max(scale2(&want));
        }
        eprintln!("stage C: L_oo T2 term max deviation = {worst:.3e} (scale {scale:.3e})");
        assert!(scale > 1e-3, "vacuous: the L_oo T2 term is ~zero");
        assert!(worst < 1e-12, "L_oo T2 term is not exact: {worst:.3e}");
        // The output must be in the PNO basis, not the canonical one.
        for (p, pair) in f.basis.pairs.iter().enumerate() {
            assert_eq!(pno[p].dim(), (pair.transform.ncols(), pair.transform.ncols()));
        }
    }

    /// Every Stage-C contraction must still RUN under truncation and produce a
    /// different answer — a rewrite that silently fell back to dense would pass
    /// the exactness tests and be useless.
    #[test]
    fn stage_c_all_contractions_are_live_under_truncation() {
        let (nocc, nvir) = (4, 6);
        let f0 = fixture(nocc, nvir, 0.0);
        let ft = fixture(nocc, nvir, 1e-3);
        assert!(!ft.basis.is_complete(), "test premise");

        let d_ex = max_dev2(
            &pno_foo_exchange(&f0.t2_pno, &f0.g_pno, &f0.basis, &f0.overlaps, &f0.index)
                .unwrap(),
            &pno_foo_exchange(&ft.t2_pno, &ft.g_pno, &ft.basis, &ft.overlaps, &ft.index)
                .unwrap(),
        );
        let d_fvv = max_dev2(
            &pno_fvv_direct(&f0.t2_pno, &f0.g_pno, &f0.basis, &f0.index).unwrap(),
            &pno_fvv_direct(&ft.t2_pno, &ft.g_pno, &ft.basis, &ft.index).unwrap(),
        );
        let d_w = max_dev4(
            &pno_woooo_amp(&f0.t2_pno, &f0.g_pno, &f0.basis, &f0.overlaps, &f0.index).unwrap(),
            &pno_woooo_amp(&ft.t2_pno, &ft.g_pno, &ft.basis, &ft.overlaps, &ft.index).unwrap(),
        );
        let d_v = max_dev4(
            &pno_wvoov_amp(&f0.t2_pno, &f0.g_pno, &f0.basis, &f0.overlaps, &f0.index).unwrap(),
            &pno_wvoov_amp(&ft.t2_pno, &ft.g_pno, &ft.basis, &ft.overlaps, &ft.index).unwrap(),
        );
        let d_o = max_dev4(
            &pno_wvovo_amp(&f0.t2_pno, &f0.g_pno, &f0.basis, &f0.overlaps, &f0.index).unwrap(),
            &pno_wvovo_amp(&ft.t2_pno, &ft.g_pno, &ft.basis, &ft.overlaps, &ft.index).unwrap(),
        );
        eprintln!(
            "stage C truncation deltas: F_oo^x {d_ex:.3e}, F_vv {d_fvv:.3e}, \
             W_oooo {d_w:.3e}, W_voov {d_v:.3e}, W_vovo {d_o:.3e}"
        );
        for (name, d) in
            [("F_oo^x", d_ex), ("F_vv", d_fvv), ("W_oooo", d_w), ("W_voov", d_v), ("W_vovo", d_o)]
        {
            assert!(d > 1e-12, "{name} is inert under truncation");
        }
    }

    // =================== DENOMINATORS ===================

    /// Denominators must come from the SEMICANONICAL PNO energies.
    ///
    /// At `t_cut_pno = 0` the rediagonalization recovers the canonical spectrum
    /// up to ordering, so the per-pair denominator set must be a permutation of
    /// the dense one. Checked as sorted multisets so an ordering difference (a
    /// non-bug) does not masquerade as an error.
    #[test]
    fn denominators_reproduce_canonical_at_zero_truncation() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 0.0);
        let eo = eps_occ(nocc);
        let ev = eps_vir(nvir);
        let dens = pno_denominators(&f.basis, &eo).unwrap();

        for (p, pair) in f.basis.pairs.iter().enumerate() {
            let (i, j) = pair.ij;
            let mut got: Vec<f64> = dens[p].iter().copied().collect();
            let mut want: Vec<f64> = (0..nvir)
                .flat_map(|a| (0..nvir).map(move |b| (a, b)))
                .map(|(a, b)| eo[i] + eo[j] - ev[a] - ev[b])
                .collect();
            got.sort_by(|x, y| x.partial_cmp(y).unwrap());
            want.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let worst = got
                .iter()
                .zip(want.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max);
            assert!(
                worst < 1e-10,
                "pair {:?} denominators are not a permutation of the canonical set: {worst:.3e}",
                pair.ij
            );
            // No denominator may be near zero — that would be a divergence, not
            // an approximation.
            assert!(dens[p].iter().all(|&v| v.abs() > 1e-6));
        }
    }

    /// The DIAGONAL-ONLY Fock shortcut, pinned as measurably wrong.
    ///
    /// `PairPno::eps` comes from re-diagonalizing `Q ᵀ diag(ε_v) Q`. Taking only
    /// `f_aa = Σ_c Q_ca² ε_c` instead is a MEASURED 0.117 Ha error in the MP2
    /// sibling. Since at `t_cut_pno = 0` both happen to agree (the transform is a
    /// full rotation), this test uses a TRUNCATED basis, where the shortcut and
    /// the true eigenvalues genuinely differ — establishing that the
    /// semicanonicalization is doing real work and is not a no-op.
    #[test]
    fn diagonal_fock_shortcut_would_differ_under_truncation() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 1e-3);
        assert!(!f.basis.is_complete(), "test premise");
        let ev = eps_vir(nvir);

        // Rebuild the RAW (pre-semicanonicalization) transforms to compare
        // against: the shortcut is defined on THOSE, and the point is that
        // f_aa there is not the stored eigenvalue.
        let raw = ferric_mp2::local_pno::build_pno_transforms(
            &complete_pair_domains(&line_centers(nocc, 1.5)).unwrap(),
            nvir,
            1e-3,
            |i, j| Array2::from_shape_fn((nvir, nvir), |(a, b)| f.t2[[i, j, a, b]]),
        )
        .unwrap();

        let mut worst = 0.0f64;
        for (p, rp) in raw.pairs.iter().enumerate() {
            let q = &rp.transform;
            let npno = q.ncols();
            for a in 0..npno {
                let shortcut: f64 = (0..nvir).map(|c| q[(c, a)] * q[(c, a)] * ev[c]).sum();
                worst = worst.max((shortcut - f.basis.pairs[p].eps[a]).abs());
            }
        }
        eprintln!(
            "denominators: max |f_aa(diagonal shortcut) - eps(semicanonical)| = {worst:.3e}"
        );
        assert!(
            worst > 1e-4,
            "premise failed: the diagonal shortcut agrees with the semicanonical energies \
             ({worst:.3e}), so this module's use of PairPno::eps would be untested"
        );
    }

    // =================== COST — structural ===================

    /// THE COST CLAIM. Every count must strictly decrease as `t_cut_pno` rises,
    /// while the dense count for the identical contraction stays fixed.
    ///
    /// NO WALL-CLOCK NUMBER APPEARS HERE, deliberately: the development box is
    /// contested, and a timing there measures the scheduler as much as the
    /// algorithm. These are pure functions of the per-pair `npno` — they say
    /// that the ALGORITHM does less arithmetic, which is the actual claim.
    #[test]
    fn cost_strictly_decreases_with_truncation() {
        let (nocc, nvir) = (4, 6);
        let thresholds = [0.0, 1e-5, 1e-4, 1e-3];
        let mut prev: Option<FlopCount> = None;
        let mut any_decrease = false;

        for &t in &thresholds {
            let f = fixture(nocc, nvir, t);
            let c = FlopCount::of(&f.basis, nocc);
            eprintln!(
                "cost @ t_cut_pno = {t:.0e} (retention {:.4}):\n{}",
                f.basis.virtual_retention(),
                c.table()
            );
            // The dense column must be threshold-INDEPENDENT — otherwise the
            // ratio is measuring the pair screen, not virtual truncation.
            if let Some(p) = &prev {
                assert_eq!(
                    p.amplitude_elements.1, c.amplitude_elements.1,
                    "dense amplitude count moved with the threshold"
                );
                assert_eq!(p.foo_flops.1, c.foo_flops.1, "dense F_oo count moved");
                assert_eq!(p.woooo_flops.1, c.woooo_flops.1, "dense W_oooo count moved");
                assert_eq!(p.loo_t2_flops.1, c.loo_t2_flops.1, "dense L_oo count moved");
                // PNO counts must be non-increasing, and strictly decreasing
                // somewhere across the sweep.
                assert!(c.amplitude_elements.0 <= p.amplitude_elements.0);
                assert!(c.foo_flops.0 <= p.foo_flops.0);
                assert!(c.woooo_flops.0 <= p.woooo_flops.0);
                assert!(c.loo_t2_flops.0 <= p.loo_t2_flops.0);
                if c.foo_flops.0 < p.foo_flops.0 {
                    any_decrease = true;
                }
            }
            prev = Some(c);
        }
        assert!(any_decrease, "no threshold in the sweep reduced the F_oo flop count");

        // At the loosest threshold every ratio must be genuinely below 1.
        let f = fixture(nocc, nvir, 1e-3);
        let c = FlopCount::of(&f.basis, nocc);
        for (name, r) in ["amp", "integ", "F_oo", "W_oooo", "L_oo"].iter().zip(c.ratios()) {
            assert!(r < 1.0, "{name} ratio {r:.4} is not below 1 at t_cut_pno = 1e-3");
        }
    }

    /// At zero truncation the cost ratios must be EXACTLY 1.
    ///
    /// This makes the previous test non-vacuous: if the counts were derived
    /// inconsistently between the two columns the ratio could sit below 1 even
    /// with nothing truncated, and the "cost decreases" claim would be an
    /// artifact of the accounting rather than of the algorithm.
    #[test]
    fn cost_ratio_is_exactly_one_without_truncation() {
        let (nocc, nvir) = (4, 6);
        let f = fixture(nocc, nvir, 0.0);
        let c = FlopCount::of(&f.basis, nocc);
        eprintln!("cost @ t_cut_pno = 0:\n{}", c.table());
        for (name, r) in ["amp", "integ", "F_oo", "W_oooo", "L_oo"].iter().zip(c.ratios()) {
            assert!(
                (r - 1.0).abs() < 1e-12,
                "{name} ratio is {r} at zero truncation — the PNO and dense counts are \
                 derived inconsistently, so every reported compression is an accounting \
                 artifact"
            );
        }
    }

    /// The per-pair integral storage must shrink too, not just the amplitudes.
    ///
    /// `PairOvov` is the storage change that makes the iteration cheap; if it
    /// still held `nvir²` per pair the flop counts would be fiction.
    #[test]
    fn per_pair_integral_storage_shrinks() {
        let (nocc, nvir) = (4, 6);
        let f0 = fixture(nocc, nvir, 0.0);
        let ft = fixture(nocc, nvir, 1e-3);
        let (p0, d0) = f0.g_pno.elements(nvir);
        let (pt, dt) = ft.g_pno.elements(nvir);
        eprintln!("integral storage: exact {p0}/{d0}, truncated {pt}/{dt}");
        assert_eq!(p0, d0, "at zero truncation per-pair storage must equal dense");
        assert!(pt < p0, "truncation did not shrink the per-pair integral storage");
    }

    // =================== error handling ===================

    /// Shape disagreements are caller bugs and must error rather than produce a
    /// plausible wrong number.
    #[test]
    fn shape_mismatches_are_rejected() {
        let (nocc, nvir) = (3, 4);
        let f = fixture(nocc, nvir, 0.0);

        let short = &f.t2_pno[..f.t2_pno.len() - 1];
        assert!(pno_foo_direct(short, &f.g_pno, &f.basis, &f.overlaps, &f.index).is_err());
        assert!(pno_foo_exchange(short, &f.g_pno, &f.basis, &f.overlaps, &f.index).is_err());
        assert!(pno_fvv_direct(short, &f.g_pno, &f.basis, &f.index).is_err());
        assert!(pno_woooo_amp(short, &f.g_pno, &f.basis, &f.overlaps, &f.index).is_err());
        assert!(pno_wvoov_amp(short, &f.g_pno, &f.basis, &f.overlaps, &f.index).is_err());
        assert!(pno_wvovo_amp(short, &f.g_pno, &f.basis, &f.overlaps, &f.index).is_err());

        let bad_ovov = Array4::<f64>::zeros((nocc, nvir + 1, nocc, nvir + 1));
        assert!(PairOvov::build(&bad_ovov, &f.basis).is_err());

        let bad_loo = Array2::<f64>::zeros((nocc + 1, nocc + 1));
        assert!(
            pno_loo_t2_term(&bad_loo, &f.t2_pno, &f.basis, &f.overlaps, &f.index).is_err()
        );

        assert!(pno_denominators(&f.basis, &eps_occ(nocc - 1)).is_err());
        assert!(dense_foo_direct(&f.ovov, &Array4::zeros((nocc, nocc, nvir + 1, nvir + 1)))
            .is_err());
    }

    // =================== REAL SYSTEM ===================

    /// THE PHYSICS CHECK: on real converged CCSD amplitudes and real RI
    /// integrals from water, every Stage B/C contraction must reproduce its
    /// dense `einsum!` counterpart at `t_cut_pno = 0`.
    ///
    /// The toy tests pin the algebra on synthetic data; this pins that the
    /// integral CONVENTION matches the solver's. `ovov` is rebuilt through the
    /// exact same RI path `ccsd_closed_shell` uses, so a mis-ordered block would
    /// show up as a real deviation rather than as a self-consistent synthetic
    /// pass.
    ///
    /// Water/6-31G: `no = 5, nv = 8`, 15 pairs — the smallest system whose PNO
    /// rotations mix real structure while staying cheap on a shared box.
    #[test]
    fn real_system_contractions_match_dense_water_631g() {
        use ferric_core::basis;
        use ferric_core::mol::Molecule;
        use ferric_core::parallel::ParallelContext;
        use ferric_integrals::basis_bridge::PreparedBasis;
        use ferric_integrals::operator::Operator;
        use ferric_mp2::mo_transform::transform_3center_ov;
        use ferric_mp2::rimp2::cholesky_inverse_sqrt;
        use ferric_mp2::spinorbital::build_b;
        use ferric_scf::rhf::{solve_rhf, RhfConfig};
        use ferric_scf::screening::SchwarzBounds;
        use ferric_tensors::{einsum, Axis};

        let xyz = "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
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

        let cfg = crate::CcConfig {
            frozen_core: 0,
            max_iter: 100,
            energy_conv: 1e-10,
            ..Default::default()
        };
        let cc =
            crate::ccsd_closed_shell::ccsd_closed_shell(&mol, &obs, &dfbs, op, &rhf, &cfg)
                .unwrap();

        let eps = rhf.eps_r();
        let nocc = eps.iter().filter(|&&e| e < 0.0).count();
        let nvir = obs.nbasis() - nocc;
        let c = rhf.mos_r();
        let c_occ = c.slice(ndarray::s![.., ..nocc]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc..]).to_owned();

        let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let v_inv_sqrt = cholesky_inverse_sqrt(&v2c).unwrap();
        let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let b_ov = build_b(
            &transform_3center_ov(&eri3_ao, &c_occ, &c_vir),
            &v_inv_sqrt,
            Axis::O,
            Axis::V,
        );
        let ovov_dyn: ndarray::ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b_ov, &b_ov);
        let ovov = ovov_dyn.into_dimensionality::<ndarray::Ix4>().unwrap();
        let t2 = cc.t2.clone();

        let ev: Vec<f64> = (0..nvir).map(|a| eps[nocc + a]).collect();
        let eo: Vec<f64> = (0..nocc).map(|i| eps[i]).collect();
        let domains = complete_pair_domains(&line_centers(nocc, 1.0)).unwrap();
        let basis = PairPnoBasis::build(&domains, nvir, &ev, 0.0, |i, j| {
            Array2::from_shape_fn((nvir, nvir), |(a, b)| {
                ovov[[i, a, j, b]] / (eo[i] + eo[j] - ev[a] - ev[b])
            })
        })
        .unwrap();
        assert!(basis.is_complete(), "t_cut_pno = 0 must keep every virtual");

        let overlaps = PairOverlaps::build(&basis);
        let index = PairIndex::new(&basis, nocc).unwrap();
        let g_pno = PairOvov::build(&ovov, &basis).unwrap();
        let t2_pno = t2_to_pno(&t2, &basis).unwrap();

        // Premise: the overlaps must be orthogonal AND non-trivial on the real
        // system too, else the S insertions below are untested.
        let orth = overlaps.max_nonorthogonality();
        assert!(orth < 1e-10, "real-system overlaps are not orthogonal: {orth:.3e}");
        let mut off = 0.0f64;
        for p in 0..overlaps.n_pairs() {
            for q in 0..overlaps.n_pairs() {
                if p == q {
                    continue;
                }
                let s = overlaps.get(p, q);
                for a in 0..s.nrows() {
                    for b in 0..s.ncols() {
                        let want = if a == b { 1.0 } else { 0.0 };
                        off = off.max((s[(a, b)] - want).abs());
                    }
                }
            }
        }
        eprintln!(
            "real system (water/6-31G, no={nocc}, nv={nvir}, {} pairs): \
             max |S Sᵀ - 1| = {orth:.3e}, max |S^(P!=Q) - 1| = {off:.3e}",
            basis.pairs.len()
        );
        assert!(off > 1e-2, "real-system pair overlaps are ~identity — S is untested");

        let checks: [(&str, f64, f64); 6] = [
            {
                let d = dense_foo_direct(&ovov, &t2).unwrap();
                let p = pno_foo_direct(&t2_pno, &g_pno, &basis, &overlaps, &index).unwrap();
                ("F_oo direct  ", max_dev2(&d, &p), scale2(&d))
            },
            {
                let d = dense_foo_exchange(&ovov, &t2).unwrap();
                let p = pno_foo_exchange(&t2_pno, &g_pno, &basis, &overlaps, &index).unwrap();
                ("F_oo exchange", max_dev2(&d, &p), scale2(&d))
            },
            {
                let d = dense_fvv_direct(&ovov, &t2).unwrap();
                let p = pno_fvv_direct(&t2_pno, &g_pno, &basis, &index).unwrap();
                ("F_vv direct  ", max_dev2(&d, &p), scale2(&d))
            },
            {
                let d = dense_woooo_amp(&ovov, &t2).unwrap();
                let p = pno_woooo_amp(&t2_pno, &g_pno, &basis, &overlaps, &index).unwrap();
                ("W_oooo       ", max_dev4(&d, &p), scale4(&d))
            },
            {
                let d = dense_wvoov_amp(&ovov, &t2).unwrap();
                let p = pno_wvoov_amp(&t2_pno, &g_pno, &basis, &overlaps, &index).unwrap();
                ("W_voov       ", max_dev4(&d, &p), scale4(&d))
            },
            {
                let d = dense_wvovo_amp(&ovov, &t2).unwrap();
                let p = pno_wvovo_amp(&t2_pno, &g_pno, &basis, &overlaps, &index).unwrap();
                ("W_vovo       ", max_dev4(&d, &p), scale4(&d))
            },
        ];
        for (name, dev, scale) in checks {
            eprintln!("real system: {name} max deviation {dev:.3e} (scale {scale:.3e})");
            assert!(scale > 1e-4, "{name} is ~zero ({scale:.3e}) — the check is vacuous");
            assert!(dev < 1e-10, "{name} is not exact on the real system: {dev:.3e}");
        }

        // And the structural cost table on the real system, at a live threshold.
        let trunc = PairPnoBasis::build(&domains, nvir, &ev, 1e-4, |i, j| {
            Array2::from_shape_fn((nvir, nvir), |(a, b)| {
                ovov[[i, a, j, b]] / (eo[i] + eo[j] - ev[a] - ev[b])
            })
        })
        .unwrap();
        let c0 = FlopCount::of(&basis, nocc);
        let ct = FlopCount::of(&trunc, nocc);
        eprintln!(
            "real system cost @ t_cut_pno = 0 (retention {:.4}):\n{}",
            basis.virtual_retention(),
            c0.table()
        );
        eprintln!(
            "real system cost @ t_cut_pno = 1e-4 (retention {:.4}):\n{}",
            trunc.virtual_retention(),
            ct.table()
        );
        assert_eq!(c0.foo_flops.1, ct.foo_flops.1, "dense count moved with the threshold");
        assert!(
            ct.foo_flops.0 <= c0.foo_flops.0,
            "truncation increased the F_oo flop count on the real system"
        );
    }

    // ==================================================================
    // PairOverlaps BUDGET GUARD
    // ==================================================================
    //
    // `PairOverlaps::build` is `O(nocc⁴·nvir²)` — the worst asymptotic in the
    // crate — and was allocated with no reference to the memory budget at all.
    // That is the shape of the LNO-coupled incident (0.055 GB predicted, 7.3 GB
    // peak): the reported working set was pair-shaped while the allocator served
    // a dense block nothing had declared.

    /// `plan_elements` must equal what `build` actually allocates, exactly. If
    /// these can disagree the guard is just a second estimator, which is the
    /// defect class it exists to retire.
    #[test]
    fn plan_elements_matches_the_allocation_exactly() {
        for t_cut in [0.0, 1e-6, 1e-4] {
            let f = fixture(4, 6, t_cut);
            let built: usize = (0..f.overlaps.n_pairs())
                .flat_map(|p| (0..f.overlaps.n_pairs()).map(move |q| (p, q)))
                .map(|(p, q)| f.overlaps.get(p, q).len())
                .sum();
            assert_eq!(
                PairOverlaps::plan_elements(&f.basis),
                built,
                "the plan must count exactly what build allocates (t_cut = {t_cut})"
            );
        }
    }

    /// A budget below the cache must be refused, naming the term.
    #[test]
    fn build_within_budget_refuses_a_tiny_budget_and_names_the_term() {
        let f = fixture(4, 6, 0.0);
        let err = PairOverlaps::build_within_budget(&f.basis, 8).unwrap_err().to_string();
        assert!(err.contains("PairOverlaps"), "must name the allocation: {err}");
        assert!(err.contains("overlap"), "breakdown must name the reservation: {err}");
    }

    /// AN OVER-ESTIMATING GUARD IS ALSO A BUG: an ample budget must still build,
    /// and must produce the identical cache the unguarded path produces.
    #[test]
    fn build_within_budget_is_identical_to_build_when_it_fits() {
        let f = fixture(4, 6, 0.0);
        let guarded = PairOverlaps::build_within_budget(&f.basis, 1 << 30).unwrap();
        assert_eq!(guarded.n_pairs(), f.overlaps.n_pairs());
        for p in 0..guarded.n_pairs() {
            for q in 0..guarded.n_pairs() {
                assert_eq!(
                    guarded.get(p, q),
                    f.overlaps.get(p, q),
                    "the guard must not perturb S^({p},{q})"
                );
            }
        }
    }

    /// The budget that EXACTLY fits must pass — an off-by-one in the guard is
    /// the over-estimating bug in miniature.
    #[test]
    fn a_budget_exactly_equal_to_the_requirement_fits() {
        let f = fixture(3, 5, 0.0);
        let need = PairOverlaps::plan_elements(&f.basis) * 8;
        assert!(PairOverlaps::build_within_budget(&f.basis, need).is_ok(), "exact fit must pass");
        assert!(
            PairOverlaps::build_within_budget(&f.basis, need - 1).is_err(),
            "one byte short must fail"
        );
    }
}
