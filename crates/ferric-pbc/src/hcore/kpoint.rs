//! Stage 3: per-k complex one-electron matrices `S(k)`, `T(k)`, `V(k)`,
//! `h(k)` — the SAME shifted-shell lattice sums as
//! [`periodic_hcore`], each image weighted by the
//! Bloch phase `e^{ik·L}` (PySCF's convention, `KPointMesh` module doc):
//!
//! ```text
//! S(k)   = Σ_L e^{ik·L} ⟨μ_0|ν_L⟩          T(k) likewise
//! V_SR(k)= −Σ_L e^{ik·L} Σ_{C,M} Z_C ⟨μ_0| erfc(ω|r−R_C−M|)/|r−R_C−M| |ν_L⟩   (nuclei periodic: no phase on M)
//! V_LR(k)= −(1/Ω) Σ_{G≠0} (4π/G²) e^{−G²/4ω²} P^k_μν(G) conj(S(G)),   S(G) = Σ_C Z_C e^{−iG·R_C}
//! V_G0(k)= + π Z_tot S(k) / (ω² Ω)
//! ```
//!
//! `P^k(G) = Σ_L e^{ik·L} ∫ φ_μ φ_ν(·−L) e^{−iG·r}` comes from the
//! residue-resolved pair FT ([`crate::pair_ft::residues`]). The LR sum runs
//! over the FULL G sphere: `P^k(−G) = conj(P^{−k}(G))`, not `conj(P^k(G))`,
//! so the Gamma half-sphere trick does not apply. Every matrix is
//! Hermitian by construction and is Hermitised (`½(X + X^H)`) to remove
//! roundoff, as the Gamma path symmetrises; `X(−k) = X(k)*` holds EXACTLY
//! because the phases are exact conjugates ([`crate::kpts`]).
//!
//! Truncation, screening, Gaussian nuclei and memory gating are exactly
//! those of `periodic_hcore` (same `PeriodicHcoreConfig`, same image and
//! nucleus-candidate sets, same `SrBound::Derived` screen), so at Gamma
//! (1×1×1 mesh) this reproduces `periodic_hcore` up to the order of the LR
//! sum (full vs half sphere).
//!
//! # Orbital-pair symmetry (k-point s2)
//!
//! For `O` = S, T and the SR attraction (translation invariant; nuclei at
//! every lattice image, pair-image and candidate sets closed under the map,
//! as at Gamma: `super` module doc "Orbital-pair symmetry"),
//! `(ν_0|O|μ_L) = (μ_0|O|ν_{−L})`, and the image set is closed under
//! `L → −L` with `e^{−ik·L} = conj e^{ik·L}`, so
//!
//! ```text
//! X(k)[νμ] = Σ_L e^{ik·L} (ν_0|O|μ_L) = Σ_L e^{−ik·L} (μ_0|O|ν_L) = conj X(k)[μν]
//! ```
//!
//! (the blocks are real). There are no residue bins here: the `e^{ik·L}` ↔
//! `e^{−ik·L}` pairing is the conjugation itself. Each UNORDERED shell pair
//! `i1 ≤ i2` is therefore evaluated once over every image, with the ordered
//! loop's per-pair body and per-element addend sequence, and ONE task writes
//! `x` into `(μ, ν)` and `conj(x)` into `(ν, μ)` for every k; a diagonal pair
//! writes its `i ≤ j` elements (the `i = j` element once, unconjugated, so
//! its round-off imaginary part is the ordered loop's, removed by the
//! Hermitisation as before). Every element is bitwise the ordered loop's
//! `μ ≤ ν` value or its conjugate, so the matrices are EXACTLY Hermitian off
//! the diagonal, the trailing `hermitize` is the identity there
//! (`½(x + conj conj x) = x`), and each element is still written by exactly
//! one task (bitwise across thread counts). `n_sr_triplets` counts the
//! triplets COMPUTED; `n_sr_triplets_ordered` (off-diagonal × 2) is the
//! pre-s2 ordered count. [`periodic_hcore_kpts_pair_s1_oracle`](crate::hcore::kpoint::periodic_hcore_kpts_pair_s1_oracle) is the pre-s2
//! build (the FROZEN serial ordered loops), bit for bit. A transposed write
//! WITHOUT the conjugation ([`KHcorePairMutant::NoConj`](crate::hcore::kpoint::KHcorePairMutant::NoConj)) is an identity on a
//! mesh whose phases are all ±1 (every k TRIM), so it is only visible on a
//! non-TRIM mesh (`tests/pbc_kpair_symmetry.rs`).
//!
//! # Column rotation at k
//!
//! [`PeriodicHcoreConfig::sr_column_rotation`] = `Auto` (default) or `On`
//! runs the `V_SR(k)` walk on the column-rotated orbital shells of
//! [`crate::sr_rotation`] (nucleus candidates, sites and pair images from
//! the PARENT shells; the screen is the rotated pairs' own) and transforms
//! each finished `V'(k)` back, `V_SR(k) = T V'(k) Tᵀ`: `T` is real and
//! k-independent, so it commutes with the Bloch phases. `S(k)`, `T(k)`, the
//! LR and G = 0 parts and the ECP stay in the parent basis. The result
//! matches the unrotated build to the screening precision (not bitwise);
//! `PeriodicHcoreK::sr_rotated_columns` counts the rotated columns (0 =
//! unrotated, bit for bit). The frozen s1 oracle runs unrotated under
//! `Auto`; the k-point forces (`kgrad`) differentiate the unrotated walk.

use super::{
    gvector_list_bytes, max_pair_exponent, nonzero_nuclei, nucleus_radius_m, pair_bound,
    pair_images, pair_radius, prim_shells, segment_distance, sr_candidates, sr_engine_pool,
    st_engine_pools, NucCand, PeriodicHcoreConfig, PrimShell, SrBound, ERI3_ENGINE_PRECISION,
    G_CHUNK_BYTES, ONE_E_ENGINE_PRECISION,
};
use crate::budget::{bytes_of, Ledger};
use crate::ewald::{default_ewald_omega, ewald_nuclear_repulsion};
use crate::kpts::{lattice_coords, KPointMesh};
use crate::lattice::Cell;
use crate::pair_ft::residues::{pair_ft_residues_chunked, residue_coords};
use crate::rsgdf::unordered_pairs;
use crate::sr_rotation::{ColumnRotationMutant, RotatedBasis};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::engine_pool::EnginePool;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::Mutex;

/// Output of [`periodic_hcore_kpts`]: one `(nbasis, nbasis)` complex
/// Hermitian matrix per mesh k-point (mesh order).
#[derive(Debug, Clone)]
pub struct PeriodicHcoreK {
    /// `S(k)`.
    pub s: Vec<Array2<Complex64>>,
    /// `T(k)`.
    pub t: Vec<Array2<Complex64>>,
    /// `V(k) = V_SR + V_LR + V_G0` (nuclear attraction with Z_eff; no ECP).
    pub v: Vec<Array2<Complex64>>,
    /// `V_ECP(k) = Σ_L e^{ik·L} V_L` ([`crate::ecp`]), Hermitised; `None` for
    /// an all-electron basis.
    pub v_ecp: Option<Vec<Array2<Complex64>>>,
    /// `h(k) = T(k) + V(k) (+ V_ECP(k))`.
    pub h: Vec<Array2<Complex64>>,
    /// Ewald nuclear repulsion per cell.
    pub enn: f64,
    /// The ω used.
    pub omega: f64,
    /// Pair images summed.
    pub n_images: usize,
    /// Shifted 3-centre calls in the SR attraction (each unordered shell
    /// pair once; module doc "Orbital-pair symmetry").
    pub n_sr_triplets: usize,
    /// `n_sr_triplets` in ordered-pair units (off-diagonal pairs × 2): the
    /// pre-s2 count, the number to compare benchmarks on.
    pub n_sr_triplets_ordered: usize,
    /// Kept (shell, shell, ECP-image) triples in `v_ecp` (0 without ECP).
    pub n_ecp_triples: usize,
    /// Full-sphere G vectors in the LR attraction.
    pub n_g_lr: usize,
    /// Orbital columns the SR attraction walk ran rotated (module doc
    /// "Column rotation at k"); 0 = the unrotated walk.
    pub sr_rotated_columns: usize,
    /// Resolved memory budget (bytes).
    pub budget_bytes: usize,
}

/// Zeroed `(n, n)` complex matrix.
fn czero(n: usize) -> Array2<Complex64> {
    Array2::<Complex64>::zeros((n, n))
}

/// TEST-ONLY defects of the k-point s2 one-electron walks (module doc
/// "Orbital-pair symmetry"). [`KHcorePairMutant::NoConj`] is an identity
/// when every phase is real (a mesh of TRIM points only).
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KHcorePairMutant {
    /// `x` instead of `conj(x)` into `(ν, μ)`.
    NoConj,
    /// The transposed write dropped.
    DropTransposed,
    /// Diagonal shell pairs skipped.
    SkipDiagonal,
}

/// COPY unordered pair `(a, b)`'s finished `(N_k, dim_a, dim_b)` block `acc`
/// into `ms[k][(μ, ν)]` and its CONJUGATE into `ms[k][(ν, μ)]` (module doc
/// "Orbital-pair symmetry"). A diagonal pair (`same`) writes only `i <= j`,
/// the `i == j` element once, so every element receives exactly one value.
fn copy_block_herm(
    ms: &mut [Array2<Complex64>],
    acc: &[Complex64],
    a: &PrimShell,
    b: &PrimShell,
    same: bool,
    mutant: Option<KHcorePairMutant>,
) {
    let bl = a.dim * b.dim;
    for (k, m) in ms.iter_mut().enumerate() {
        for i in 0..a.dim {
            for j in 0..b.dim {
                if same && i > j {
                    continue;
                }
                let x = acc[k * bl + i * b.dim + j];
                m[(a.off + i, b.off + j)] = x;
                if same && i == j {
                    continue;
                }
                match mutant {
                    Some(KHcorePairMutant::DropTransposed) => {}
                    Some(KHcorePairMutant::NoConj) => m[(b.off + j, a.off + i)] = x,
                    _ => m[(b.off + j, a.off + i)] = x.conj(),
                }
            }
        }
    }
}

/// Sum per-unordered-pair counts in pair order: `(computed,
/// ordered-equivalent)` (off-diagonal pairs weighted 2); the first error in
/// pair order is returned instead.
fn sum_s2_pair_counts(
    pairs: &[(usize, usize)],
    counts: Vec<Result<usize, FerricError>>,
) -> Result<(usize, usize), FerricError> {
    let (mut n, mut n_ordered) = (0usize, 0usize);
    for (&(i1, i2), c) in pairs.iter().zip(counts) {
        let c = c?;
        n += c;
        n_ordered += if i1 == i2 { c } else { 2 * c };
    }
    Ok((n, n_ordered))
}

/// `m_k[o1+i, o2+j] += ph_k · (f · blk[i, j])` for every k.
#[allow(clippy::too_many_arguments)]
fn add_block_k(
    ms: &mut [Array2<Complex64>],
    ph: &[Complex64],
    blk: &[f64],
    o1: usize,
    n1: usize,
    o2: usize,
    n2: usize,
    f: f64,
) {
    for (m, p) in ms.iter_mut().zip(ph) {
        for i in 0..n1 {
            for j in 0..n2 {
                m[(o1 + i, o2 + j)] += *p * (f * blk[i * n2 + j]);
            }
        }
    }
}

/// `½(X + X^H)`.
pub(crate) fn hermitize(m: &Array2<Complex64>) -> Array2<Complex64> {
    let n = m.nrows();
    Array2::from_shape_fn((n, n), |(i, j)| 0.5 * (m[(i, j)] + m[(j, i)].conj()))
}

/// Build `S(k)`, `T(k)`, `V(k)`, `h(k)` on every point of `mesh`, and
/// `E_nn`. `prep` must be built from `cell.mol()`; `mesh` from `cell`.
pub fn periodic_hcore_kpts(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    cfg: &PeriodicHcoreConfig,
) -> Result<PeriodicHcoreK, FerricError> {
    periodic_hcore_kpts_impl(cell, prep, mesh, cfg, false)
}

/// TEST ORACLE (FROZEN; do not "improve"): [`periodic_hcore_kpts`] with the
/// pre-s2 `S(k)`/`T(k)` and `V_SR(k)` — the FROZEN serial ORDERED loops
/// (`overlap_kinetic_kpts_serial_oracle`, `SrKCtx::serial_oracle`, bitwise
/// the pre-s2 parallel production) followed by the same Hermitisation. Bit
/// for bit the build before s2 (its `n_sr_triplets` =
/// `n_sr_triplets_ordered` = the ordered count). Serial: small test cells
/// only.
#[doc(hidden)]
pub fn periodic_hcore_kpts_pair_s1_oracle(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    cfg: &PeriodicHcoreConfig,
) -> Result<PeriodicHcoreK, FerricError> {
    periodic_hcore_kpts_impl(cell, prep, mesh, cfg, true)
}

/// Shared body of [`periodic_hcore_kpts`] and the `s1_oracle` variant (`true` = frozen pre-s2
/// pair loop). The column rotation is not implemented here: explicit `On` is refused.
fn periodic_hcore_kpts_impl(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    cfg: &PeriodicHcoreConfig,
    s1_oracle: bool,
) -> Result<PeriodicHcoreK, FerricError> {
    cfg.validate()?;
    let rotation = kpoint_sr_rotation(cell, prep, cfg, s1_oracle)?;
    let sr_rotated_columns = rotated_columns(rotation.as_ref());
    // Z_eff guard first: a bare Z is silent for the k-mesh ≡ supercell anchor.
    crate::ecp::check_ecp_applied(cell, prep.basis_set())?;
    let shells = prim_shells(cell, prep)?;
    let n = prep.nbasis();
    let nk = mesh.nk();
    let omega = cfg.omega;
    let thresh = cfg.precision;
    let pair_thresh = 0.1 * thresh;
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    ledger.reserve(
        &format!(
            "periodic_hcore_kpts complex n×n matrices (n = {n}, N_k = {nk}: S, T, V_SR, V_LR, V, h + 2 temporaries)"
        ),
        bytes_of((nk * n * n) as u64, 16 * 8),
    )?;

    let (images, rpair, ph) = images_and_phases(cell, &shells, mesh, pair_thresh, &mut ledger)?;

    // --- S(k), T(k) (parallel over unordered shell pairs, each element
    // bitwise the serial ordered loop's μ ≤ ν value or its conjugate:
    // `overlap_kinetic_kpts`; module doc "Orbital-pair symmetry").
    let (s, t) = if s1_oracle {
        overlap_kinetic_kpts_serial_oracle(prep, &shells, &images, &ph, nk)?
    } else {
        overlap_kinetic_kpts(prep, &shells, &images, &ph, nk, None)?
    };
    let s: Vec<Array2<Complex64>> = s.iter().map(hermitize).collect();
    let t: Vec<Array2<Complex64>> = t.iter().map(hermitize).collect();

    // --- V_SR(k): the production SR loop of `sr_attraction` (Derived bound,
    // no tracking), phase-weighted by the ν image L; parallel over unordered
    // shell pairs, each element the serial ordered loop's μ ≤ ν value or its
    // conjugate (`SrKCtx::parallel`).
    let zs = cell.nuclear_charges();
    let inp = SrKInputs {
        cell,
        prep,
        cfg,
        shells: &shells,
        images: &images,
        ph: &ph,
        rpair,
        nk: mesh.nk(),
    };
    let (v_sr, n_sr_triplets, n_sr_triplets_ordered) = match SrKCtx::new(&inp, &mut ledger)? {
        Some(ctx) if s1_oracle => {
            let (v, c) = ctx.serial_oracle()?;
            (v, c, c)
        }
        Some(ctx) => ctx.parallel_maybe_rotated(&ledger, cell, rotation.as_ref())?,
        None => ((0..nk).map(|_| czero(n)).collect(), 0, 0),
    };
    let v_sr: Vec<Array2<Complex64>> = v_sr.iter().map(hermitize).collect();

    // --- V_LR(k): full G sphere, residue-resolved pair FT at q = 0.
    let smooth = (1.0 / thresh).ln().sqrt();
    let gcut = (2.0 * omega).min(2.0 * max_pair_exponent(prep).sqrt()) * smooth;
    ledger.reserve(
        &format!("periodic_hcore_kpts LR G list (|G| <= {gcut:.3})"),
        gvector_list_bytes(cell, gcut)?,
    )?;
    let gv: Vec<[f64; 3]> = cell
        .gvectors(gcut)?
        .into_iter()
        .filter(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2] > 0.0)
        .collect();
    let moduli = mesh.residue_moduli();
    let nr = moduli[0] * moduli[1] * moduli[2];
    let phr: Vec<Vec<Complex64>> = (0..nk)
        .map(|k| {
            (0..nr)
                .map(|r| mesh.phase(k, residue_coords(r, moduli)))
                .collect()
        })
        .collect();
    let pos = cell.positions();
    let vol = cell.volume();
    let mut v_lr: Vec<Array2<Complex64>> = (0..nk).map(|_| czero(n)).collect();
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let accumulate =
        |_g0: usize, gs: &[[f64; 3]], q: &[Array3<Complex64>]| -> Result<(), FerricError> {
            for (g, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let kern = 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp();
                // S(G) = Σ_C Z_C e^{−iG·R_C}
                let mut sg = Complex64::new(0.0, 0.0);
                for (zc, r) in zs.iter().zip(&pos) {
                    let a = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                    sg += Complex64::new(zc * a.cos(), -zc * a.sin());
                }
                let w = sg.conj() * (-kern / vol);
                for (vk, pk) in v_lr.iter_mut().zip(&phr) {
                    for m in 0..n {
                        for nu in 0..n {
                            let mut p = Complex64::new(0.0, 0.0);
                            for (qr, ph) in q.iter().zip(pk) {
                                p += *ph * qr[[m, nu, g]];
                            }
                            vk[(m, nu)] += p * w;
                        }
                    }
                }
            }
            Ok(())
        };
    pair_ft_residues_chunked(
        cell,
        prep,
        &gv,
        moduli,
        pair_thresh,
        chunk_budget,
        0,
        accumulate,
    )?;
    let ztot: f64 = zs.iter().sum();
    let c0 = PI / (omega * omega * vol);
    // --- V_ECP(k) = Σ_L e^{ik·L} V_L (`None` for an all-electron basis).
    let ecp = crate::ecp::periodic_ecp_images_on(cell, prep, &cfg.ecp_config(), &mut ledger)?;
    let n_ecp_triples = ecp.as_ref().map_or(0, |e| e.n_triples);
    let v_ecp = match &ecp {
        Some(e) => Some(e.at_kpts(cell, mesh)?),
        None => None,
    };
    drop(ecp);
    let mut v = Vec::with_capacity(nk);
    let mut h = Vec::with_capacity(nk);
    for k in 0..nk {
        let vk = &(&v_sr[k] + &hermitize(&v_lr[k])) + &s[k].mapv(|z| z * (c0 * ztot));
        let mut hk = &t[k] + &vk;
        if let Some(ve) = &v_ecp {
            hk += &ve[k];
        }
        h.push(hk);
        v.push(vk);
    }
    let enn = ewald_nuclear_repulsion(cell, default_ewald_omega(cell))?;
    ferric_core::memory::warn_if_rss_over("ferric-pbc periodic_hcore_kpts", ledger.budget(), 1.1);
    Ok(PeriodicHcoreK {
        s,
        t,
        v,
        v_ecp,
        h,
        enn,
        omega,
        n_images: images.len(),
        n_sr_triplets,
        n_sr_triplets_ordered,
        n_ecp_triples,
        n_g_lr: gv.len(),
        sr_rotated_columns,
        budget_bytes: ledger.budget(),
    })
}

/// Rotated columns of a rotation (0 without one).
fn rotated_columns(rot: Option<&RotatedBasis>) -> usize {
    rot.map_or(0, |r| r.n_rotated_columns)
}

/// The column rotation of the k-point SR attraction (module doc "Column
/// rotation at k"): `Auto`/`On` rotate, `Off` and the frozen s1 oracle (which
/// refuses an explicit `On`) do not; `None` also when nothing in the basis
/// rotates (the identity: today's walk bit for bit). The `RotateAux` mutant
/// has no aux basis here and is refused.
fn kpoint_sr_rotation(
    cell: &Cell,
    prep: &PreparedBasis,
    cfg: &PeriodicHcoreConfig,
    s1_oracle: bool,
) -> Result<Option<RotatedBasis>, FerricError> {
    if s1_oracle {
        cfg.sr_column_rotation.refuse_explicit(
            "periodic_hcore_kpts_pair_s1_oracle",
            "the frozen s1 oracle walks the unrotated shells",
        )?;
        return Ok(None);
    }
    let Some(rot) = cfg.sr_column_rotation.resolve_supported() else {
        return Ok(None);
    };
    if rot.mutant == ColumnRotationMutant::RotateAux {
        return Err(FerricError::General(format!(
            "periodic_hcore_kpts: sr_column_rotation {:?} is not available on the SR \
             attraction (no aux basis; build without it)",
            rot.mutant
        )));
    }
    RotatedBasis::detect(cell, prep, rot, "periodic_hcore_kpts")
}

/// Unhermitised `S(k)`, `T(k)` (one per mesh point).
type StK = (Vec<Array2<Complex64>>, Vec<Array2<Complex64>>);

/// Unhermitised `S(k) = Σ_L e^{ik·L} (μ_0|ν_L)` and `T(k)` over `images`
/// (`ph[il][k] = e^{ik·L_il}`), EXACTLY Hermitian off the diagonal (module
/// doc "Orbital-pair symmetry").
///
/// PARALLEL over UNORDERED shell pairs `i1 <= i2`: element `(k, μ, ν)` with
/// `μ` in shell `i1 <= ν`'s shell `i2` receives `e^{ik·L} (1.0 · block)`
/// ONLY from pair `(i1, i2)`, in `images` order (k only selects the phase),
/// so it is BITWISE the serial ordered `L → i1 → i2` loop's element
/// ([`overlap_kinetic_kpts_serial_oracle`]); `(k, ν, μ)` receives its
/// conjugate. Each task sums its two `(N_k, dim_i1, dim_i2)` blocks from zero
/// with `add_block_k`'s expression and COPIES them into both places of the
/// zeroed outputs under a mutex ([`copy_block_herm`]); distinct unordered
/// pairs own disjoint elements, so task finishing order cannot change a bit.
/// One overlap + one kinetic engine per rayon worker; errors are returned
/// in pair order. `mutant`: a test-only defect (`None` in production).
fn overlap_kinetic_kpts(
    prep: &PreparedBasis,
    shells: &[PrimShell],
    images: &[[f64; 3]],
    ph: &[Vec<Complex64>],
    nk: usize,
    mutant: Option<KHcorePairMutant>,
) -> Result<StK, FerricError> {
    let n = prep.nbasis();
    let nsh = shells.len();
    let [pool_s, pool_t] = st_engine_pools(prep)?;
    let zero = Complex64::new(0.0, 0.0);
    let out = Mutex::new((
        (0..nk).map(|_| czero(n)).collect::<Vec<_>>(),
        (0..nk).map(|_| czero(n)).collect::<Vec<_>>(),
    ));
    let skip_diag = mutant == Some(KHcorePairMutant::SkipDiagonal);
    let per_pair: Vec<Result<(), FerricError>> = unordered_pairs(nsh)
        .into_par_iter()
        .map(|(i1, i2)| {
            if skip_diag && i1 == i2 {
                return Ok(());
            }
            let (a, b) = (&shells[i1], &shells[i2]);
            let bl = a.dim * b.dim;
            let (mut acc_s, mut acc_t) = (vec![zero; nk * bl], vec![zero; nk * bl]);
            pool_s.with(|es| {
                pool_t.with(|et| -> Result<(), FerricError> {
                    for (il, l) in images.iter().enumerate() {
                        let blk = es.compute_1e_block_shifted(prep, i1, i2, *l)?;
                        add_phased(&mut acc_s, &ph[il], &blk[..bl]);
                        let blk = et.compute_1e_block_shifted(prep, i1, i2, *l)?;
                        add_phased(&mut acc_t, &ph[il], &blk[..bl]);
                    }
                    Ok(())
                })
            })?;
            let mut guard = out.lock().unwrap_or_else(|e| e.into_inner());
            let (s, t) = &mut *guard;
            copy_block_herm(s, &acc_s, a, b, i1 == i2, mutant);
            copy_block_herm(t, &acc_t, a, b, i1 == i2, mutant);
            Ok(())
        })
        .collect();
    for r in per_pair {
        r?;
    }
    Ok(out.into_inner().unwrap_or_else(|e| e.into_inner()))
}

/// `acc[k] += ph[k] · (1.0 · blk)` for every k: `add_block_k`'s expression
/// (f = 1, and `1.0 · y` is `y` exactly) on one pair's `(N_k, bl)` running
/// block.
fn add_phased(acc: &mut [Complex64], ph: &[Complex64], blk: &[f64]) {
    let bl = blk.len();
    for (k, p) in ph.iter().enumerate() {
        for (x, y) in acc[k * bl..(k + 1) * bl].iter_mut().zip(blk) {
            *x += *p * *y;
        }
    }
}

/// The serial `S(k)`/`T(k)` loop as it was before the pair-parallel
/// rewrite of [`overlap_kinetic_kpts`] (FROZEN; oracle only — do not
/// "improve").
fn overlap_kinetic_kpts_serial_oracle(
    prep: &PreparedBasis,
    shells: &[PrimShell],
    images: &[[f64; 3]],
    ph: &[Vec<Complex64>],
    nk: usize,
) -> Result<StK, FerricError> {
    let n = prep.nbasis();
    let mut eng_s = Engine::new_1e(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
    let mut eng_t = Engine::new_1e(ffi::OP_KINETIC, prep, ONE_E_ENGINE_PRECISION)?;
    let mut s: Vec<Array2<Complex64>> = (0..nk).map(|_| czero(n)).collect();
    let mut t: Vec<Array2<Complex64>> = (0..nk).map(|_| czero(n)).collect();
    for (il, l) in images.iter().enumerate() {
        for (i1, a) in shells.iter().enumerate() {
            for (i2, bsh) in shells.iter().enumerate() {
                let blk = eng_s.compute_1e_block_shifted(prep, i1, i2, *l)?;
                add_block_k(&mut s, &ph[il], blk, a.off, a.dim, bsh.off, bsh.dim, 1.0);
                let blk = eng_t.compute_1e_block_shifted(prep, i1, i2, *l)?;
                add_block_k(&mut t, &ph[il], blk, a.off, a.dim, bsh.off, bsh.dim, 1.0);
            }
        }
    }
    Ok((s, t))
}

/// TEST ORACLE for the parallel `S(k)`/`T(k)`: `[parallel, serial]`
/// unhermitised `(S(k), T(k))` over the pair images and phases
/// [`periodic_hcore_kpts`] uses at `cfg` on `mesh`, plus the image count.
/// `serial` is the pre-parallel ordered loop, FROZEN verbatim; `parallel`
/// is the s2 walk, whose `(μ, ν)` must be BITWISE `serial[(μ, ν)]` for
/// `μ ≤ ν` and `conj(serial[(ν, μ)])` for `μ > ν` (module doc
/// "Orbital-pair symmetry"; `tests/pbc_parallel_bitwise.rs`).
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn overlap_kinetic_kpts_parallel_and_serial(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    cfg: &PeriodicHcoreConfig,
) -> Result<([(Vec<Array2<Complex64>>, Vec<Array2<Complex64>>); 2], usize), FerricError> {
    cfg.validate()?;
    let shells = prim_shells(cell, prep)?;
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    let (images, _, ph) = images_and_phases(cell, &shells, mesh, 0.1 * cfg.precision, &mut ledger)?;
    let nk = mesh.nk();
    Ok((
        [
            overlap_kinetic_kpts(prep, &shells, &images, &ph, nk, None)?,
            overlap_kinetic_kpts_serial_oracle(prep, &shells, &images, &ph, nk)?,
        ],
        images.len(),
    ))
}

/// Pair images `L`, `r_pair` and the Bloch phases `ph[il][k] = e^{ik·L}`
/// (both lists reserved on `ledger`): what `S(k)`, `T(k)` and `V_SR(k)`
/// share.
#[allow(clippy::type_complexity)]
fn images_and_phases(
    cell: &Cell,
    shells: &[PrimShell],
    mesh: &KPointMesh,
    pair_thresh: f64,
    ledger: &mut Ledger,
) -> Result<(Vec<[f64; 3]>, f64, Vec<Vec<Complex64>>), FerricError> {
    let nk = mesh.nk();
    let images = pair_images(cell, shells, pair_thresh, ledger)?;
    let rpair = pair_radius(shells, pair_thresh);
    ledger.reserve(
        &format!(
            "periodic_hcore_kpts image phases ({} images × {nk} k)",
            images.len()
        ),
        bytes_of((images.len() * nk) as u64, 16),
    )?;
    let b = cell.reciprocal();
    let ph: Vec<Vec<Complex64>> = images
        .iter()
        .map(|l| {
            let nl = lattice_coords(&b, l);
            (0..nk).map(|k| mesh.phase(k, nl)).collect()
        })
        .collect();
    Ok((images, rpair, ph))
}

/// What the `V_SR(k)` walk shares with `S(k)`/`T(k)`.
struct SrKInputs<'a> {
    cell: &'a Cell,
    prep: &'a PreparedBasis,
    cfg: &'a PeriodicHcoreConfig,
    shells: &'a [PrimShell],
    images: &'a [[f64; 3]],
    /// `ph[il][k] = e^{ik·L_il}`.
    ph: &'a [Vec<Complex64>],
    rpair: f64,
    nk: usize,
}

/// The loop invariants of the `V_SR(k)` walk: nucleus candidates and sites.
struct SrKCtx<'a> {
    inp: &'a SrKInputs<'a>,
    cands: Vec<NucCand>,
    nuc: Vec<(f64, [f64; 3])>,
    site: SiteBasis,
    zmax: f64,
}

/// Unhermitised `V_SR(k)` (one per mesh point) and the triplet count.
type SrK = (Vec<Array2<Complex64>>, usize);

/// Unhermitised s2 `V_SR(k)`, triplets computed and ordered-equivalent.
type SrKS2 = (Vec<Array2<Complex64>>, usize, usize);

impl<'a> SrKCtx<'a> {
    /// Candidates (reserved on `ledger`) and Gaussian-nucleus sites; `None`
    /// when every nucleus has `Z = 0` (then `V_SR(k) = 0`).
    fn new(inp: &'a SrKInputs<'a>, ledger: &mut Ledger) -> Result<Option<Self>, FerricError> {
        let (nuc, zmax) = nonzero_nuclei(inp.cell);
        if nuc.is_empty() {
            return Ok(None);
        }
        let cfg = inp.cfg;
        let cands = sr_candidates(
            inp.cell,
            inp.shells,
            &nuc,
            cfg.omega,
            zmax,
            cfg.precision,
            inp.rpair,
            ledger,
        )?;
        let sites: Vec<[f64; 4]> = nuc
            .iter()
            .map(|(_, r)| [r[0], r[1], r[2], cfg.nucleus_exponent])
            .collect();
        let site = SiteBasis::new(&sites, 0)?;
        Ok(Some(Self {
            inp,
            cands,
            nuc,
            site,
            zmax,
        }))
    }

    /// Number of k-points in the mesh.
    fn nk(&self) -> usize {
        self.inp.nk
    }

    /// PARALLEL over UNORDERED shell pairs `i1 <= i2` (module doc
    /// "Orbital-pair symmetry"): element `(k, μ, ν)` with `μ` in shell `i1 <=
    /// ν`'s shell `i2` receives addends `e^{ik·L} (f · block)` ONLY from pair
    /// `(i1, i2)`, in the order "L ascending (the `images` order), then
    /// candidate order" — the k index only selects the phase, it never
    /// reorders — so it is BITWISE the serial ordered `L → i1 → i2 →
    /// candidate` loop's element ([`SrKCtx::serial_oracle`]); `(k, ν, μ)`
    /// receives its conjugate. Each task accumulates its `(N_k, dim_i1,
    /// dim_i2)` block from zero with the same `+=` expression and COPIES it
    /// into both places of the zeroed output under a mutex
    /// ([`copy_block_herm`]); distinct unordered pairs own disjoint elements,
    /// so task finishing order cannot change a bit. One erfc 3-centre engine
    /// per rayon worker; the counts are integer sums (computed,
    /// ordered-equivalent) and errors are returned in pair order. The
    /// per-thread scratch is CHECKED on `ledger` (width independent of the
    /// thread count). `mutant`: a test-only defect (`None` in production).
    fn parallel(
        &self,
        ledger: &Ledger,
        mutant: Option<KHcorePairMutant>,
    ) -> Result<SrKS2, FerricError> {
        let inp = self.inp;
        let (n, nk, nsh) = (inp.prep.nbasis(), self.nk(), inp.shells.len());
        let dmax = inp.shells.iter().map(|s| s.dim).max().unwrap_or(0) as u64;
        let threads = rayon::current_num_threads();
        ledger.check(
            &format!("periodic_hcore_kpts SR per-thread scratch ({threads} threads, N_k = {nk})"),
            bytes_of((nk as u64).saturating_mul(dmax * dmax), 16)
                .saturating_mul(threads.saturating_add(1)),
        )?;
        let pool = sr_engine_pool(inp.prep, &self.site, inp.cfg.omega)?;
        let out = Mutex::new((0..nk).map(|_| czero(n)).collect::<Vec<_>>());
        let skip_diag = mutant == Some(KHcorePairMutant::SkipDiagonal);
        let pairs = unordered_pairs(nsh);
        let counts: Vec<Result<usize, FerricError>> = pairs
            .par_iter()
            .map(|&(i1, i2)| {
                if skip_diag && i1 == i2 {
                    return Ok(0);
                }
                self.pair_task(&pool, i1, i2, &out, mutant)
            })
            .collect();
        let (n_triplets, n_ordered) = sum_s2_pair_counts(&pairs, counts)?;
        Ok((
            out.into_inner().unwrap_or_else(|e| e.into_inner()),
            n_triplets,
            n_ordered,
        ))
    }

    /// [`SrKCtx::parallel`] when `rot` is `None`, else
    /// [`SrKCtx::parallel_rotated`].
    fn parallel_maybe_rotated(
        self,
        ledger: &Ledger,
        cell: &Cell,
        rot: Option<&RotatedBasis>,
    ) -> Result<SrKS2, FerricError> {
        match rot {
            None => self.parallel(ledger, None),
            Some(r) => self.parallel_rotated(ledger, cell, r),
        }
    }

    /// [`SrKCtx::parallel`] on the column-rotated shells of `rot`, transformed
    /// back into the parent AO basis: `V_SR(k) = T V'(k) Tᵀ` per mesh point
    /// (`T` is k-independent and real, so it commutes with the Bloch phase
    /// sum). The nucleus candidates, sites and pair images stay the parent's
    /// (every rotated primitive set is a subset of its parent's); the
    /// screen is the rotated shells' own, so the result matches the
    /// unrotated one to the screening precision, not bitwise. The matrices
    /// are returned unhermitised-in-the-roundoff sense, as
    /// [`SrKCtx::parallel`]'s (the caller Hermitises).
    fn parallel_rotated(
        self,
        ledger: &Ledger,
        cell: &Cell,
        rot: &RotatedBasis,
    ) -> Result<SrKS2, FerricError> {
        let rot_shells = prim_shells(cell, &rot.prep)?;
        let inp = self.inp;
        let winp = SrKInputs {
            cell: inp.cell,
            prep: &rot.prep,
            cfg: inp.cfg,
            shells: &rot_shells,
            images: inp.images,
            ph: inp.ph,
            rpair: inp.rpair,
            nk: inp.nk,
        };
        let walk = SrKCtx {
            inp: &winp,
            cands: self.cands,
            nuc: self.nuc,
            site: self.site,
            zmax: self.zmax,
        };
        let (mut v, n_triplets, n_ordered) = walk.parallel(ledger, None)?;
        let n = rot.prep.nbasis();
        for m in &mut v {
            let data = m.as_slice_mut().ok_or_else(|| {
                FerricError::General("periodic_hcore_kpts: V_SR(k) is not contiguous".into())
            })?;
            rot.back_transform_complex_rows(data, 1)?;
            debug_assert_eq!(m.dim(), (n, n));
        }
        Ok((v, n_triplets, n_ordered))
    }

    /// One [`SrKCtx::parallel`] task: unordered pair `(i1 <= i2)` over every
    /// image `L` ascending, then its finished `(N_k, dim_i1, dim_i2)` block
    /// COPIED into `out` at `(μ, ν)` and, conjugated, at `(ν, μ)`
    /// ([`copy_block_herm`]). Returns the triplet count.
    fn pair_task(
        &self,
        pool: &EnginePool,
        i1: usize,
        i2: usize,
        out: &Mutex<Vec<Array2<Complex64>>>,
        mutant: Option<KHcorePairMutant>,
    ) -> Result<usize, FerricError> {
        let (a, b) = (&self.inp.shells[i1], &self.inp.shells[i2]);
        let bl = a.dim * b.dim;
        let mut acc = vec![Complex64::new(0.0, 0.0); self.nk() * bl];
        let mut count = 0usize;
        pool.with(|eng| -> Result<(), FerricError> {
            for il in 0..self.inp.images.len() {
                self.pair_image(eng, (i1, i2), il, &mut acc, &mut count)?;
            }
            Ok(())
        })?;
        if count > 0 {
            let mut vs = out.lock().unwrap_or_else(|e| e.into_inner());
            copy_block_herm(&mut vs[..], &acc, a, b, i1 == i2, mutant);
        }
        Ok(count)
    }

    /// Pair `(i1, i2)` at image `il`: screen, then every candidate in order,
    /// each kept block added into `acc[k]` with its phase (the serial loop's
    /// body, `add_block_k`'s expression).
    fn pair_image(
        &self,
        eng: &mut Engine,
        (i1, i2): (usize, usize),
        il: usize,
        acc: &mut [Complex64],
        count: &mut usize,
    ) -> Result<(), FerricError> {
        let inp = self.inp;
        let (a, bsh) = (&inp.shells[i1], &inp.shells[i2]);
        let l = &inp.images[il];
        let bound = SrBound::Derived;
        let bc = [
            bsh.center[0] + l[0],
            bsh.center[1] + l[1],
            bsh.center[2] + l[2],
        ];
        let r2 = (a.center[0] - bc[0]).powi(2)
            + (a.center[1] - bc[1]).powi(2)
            + (a.center[2] - bc[2]).powi(2);
        let (q, pmin, pmax) = pair_bound(a, bsh, r2);
        let wp = bound.omega_p(inp.cfg.omega, pmin);
        let Some(rad) = nucleus_radius_m(q, self.zmax, pmax, wp, inp.cfg.precision, bound.margin())
        else {
            return Ok(());
        };
        let bl = a.dim * bsh.dim;
        for (kc, m, x) in &self.cands {
            if segment_distance(*x, a.center, bc) > rad {
                continue;
            }
            *count += 1;
            let f = -self.nuc[*kc].0 / self.site.norm_int[*kc];
            if let Some(blk) = eng.compute_eri3_shifted(
                inp.prep,
                &self.site.prep,
                self.site.site_shell[*kc],
                i1,
                i2,
                [*m, [0.0; 3], *l],
            )? {
                // == add_block_k(v_sr, ph[il], blk, a.off, a.dim, b.off, b.dim, f)
                for (k, p) in inp.ph[il].iter().enumerate() {
                    let dst = &mut acc[k * bl..(k + 1) * bl];
                    for i in 0..a.dim {
                        for j in 0..bsh.dim {
                            dst[i * bsh.dim + j] += *p * (f * blk[i * bsh.dim + j]);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The serial `V_SR(k)` loop as it was before the pair-parallel rewrite
    /// (FROZEN; oracle only — do not "improve").
    fn serial_oracle(&self) -> Result<SrK, FerricError> {
        let inp = self.inp;
        let (prep, images, ph, shells) = (inp.prep, inp.images, inp.ph, inp.shells);
        let (omega, thresh, zmax) = (inp.cfg.omega, inp.cfg.precision, self.zmax);
        let (nuc, site, cands) = (&self.nuc, &self.site, &self.cands);
        let n = prep.nbasis();
        let mut v_sr: Vec<Array2<Complex64>> = (0..self.nk()).map(|_| czero(n)).collect();
        let mut n_sr_triplets = 0usize;
        let mut eng = Engine::new_3center(
            Operator::erfc(omega),
            prep,
            &site.prep,
            ERI3_ENGINE_PRECISION,
        )?;
        let bound = SrBound::Derived;
        for (il, l) in images.iter().enumerate() {
            for (i1, a) in shells.iter().enumerate() {
                for (i2, bsh) in shells.iter().enumerate() {
                    let bc = [
                        bsh.center[0] + l[0],
                        bsh.center[1] + l[1],
                        bsh.center[2] + l[2],
                    ];
                    let r2 = (a.center[0] - bc[0]).powi(2)
                        + (a.center[1] - bc[1]).powi(2)
                        + (a.center[2] - bc[2]).powi(2);
                    let (q, pmin, pmax) = pair_bound(a, bsh, r2);
                    let wp = bound.omega_p(omega, pmin);
                    let Some(rad) = nucleus_radius_m(q, zmax, pmax, wp, thresh, bound.margin())
                    else {
                        continue;
                    };
                    for (kc, m, x) in cands {
                        if segment_distance(*x, a.center, bc) > rad {
                            continue;
                        }
                        n_sr_triplets += 1;
                        let f = -nuc[*kc].0 / site.norm_int[*kc];
                        if let Some(blk) = eng.compute_eri3_shifted(
                            prep,
                            &site.prep,
                            site.site_shell[*kc],
                            i1,
                            i2,
                            [*m, [0.0; 3], *l],
                        )? {
                            add_block_k(&mut v_sr, &ph[il], blk, a.off, a.dim, bsh.off, bsh.dim, f);
                        }
                    }
                }
            }
        }
        Ok((v_sr, n_sr_triplets))
    }
}

/// TEST ORACLE for the parallel `V_SR(k)`: `[parallel, serial]`
/// unhermitised `V_SR(k)` and triplet counts of [`periodic_hcore_kpts`] at
/// `cfg` on `mesh` (same pair images, phases, nucleus candidates and
/// screen). `serial` is the pre-parallel ORDERED `L → i1 → i2 → candidate`
/// loop, FROZEN verbatim, with its ordered count; `parallel` is the s2
/// walk with its ORDERED-EQUIVALENT count, whose `(μ, ν)` must be BITWISE
/// `serial[(μ, ν)]` for `μ ≤ ν` and `conj(serial[(ν, μ)])` for `μ > ν`
/// (module doc "Orbital-pair symmetry"; `tests/pbc_parallel_bitwise.rs`).
#[doc(hidden)]
pub fn sr_attraction_kpts_parallel_and_serial(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    cfg: &PeriodicHcoreConfig,
) -> Result<[(Vec<Array2<Complex64>>, usize); 2], FerricError> {
    cfg.validate()?;
    let shells = prim_shells(cell, prep)?;
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    let (images, rpair, ph) =
        images_and_phases(cell, &shells, mesh, 0.1 * cfg.precision, &mut ledger)?;
    let inp = SrKInputs {
        cell,
        prep,
        cfg,
        shells: &shells,
        images: &images,
        ph: &ph,
        rpair,
        nk: mesh.nk(),
    };
    let ctx = SrKCtx::new(&inp, &mut ledger)?.ok_or_else(|| {
        FerricError::General("sr_attraction_kpts_parallel_and_serial: no nuclei".into())
    })?;
    let (v, _, n_ordered) = ctx.parallel(&ledger, None)?;
    Ok([(v, n_ordered), ctx.serial_oracle()?])
}

/// One s2 k-point one-electron result of [`hcore_kpts_s2_mutant`]:
/// unhermitised `(S(k), T(k), V_SR(k), SR triplets computed,
/// ordered-equivalent)`.
pub type HcoreKS2Parts = (
    Vec<Array2<Complex64>>,
    Vec<Array2<Complex64>>,
    Vec<Array2<Complex64>>,
    usize,
    usize,
);

/// TEST ORACLE for the k-point s2 walks under a test-only defect
/// (module doc "Orbital-pair symmetry"; `None`: production): the
/// unhermitised `S(k)`, `T(k)`, `V_SR(k)` over the pair images, phases and
/// nucleus candidates [`periodic_hcore_kpts`] uses at `cfg` on `mesh`
/// (`tests/pbc_kpair_symmetry.rs`).
#[doc(hidden)]
pub fn hcore_kpts_s2_mutant(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    cfg: &PeriodicHcoreConfig,
    mutant: Option<KHcorePairMutant>,
) -> Result<HcoreKS2Parts, FerricError> {
    cfg.validate()?;
    let shells = prim_shells(cell, prep)?;
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    let (images, rpair, ph) =
        images_and_phases(cell, &shells, mesh, 0.1 * cfg.precision, &mut ledger)?;
    let nk = mesh.nk();
    let (s, t) = overlap_kinetic_kpts(prep, &shells, &images, &ph, nk, mutant)?;
    let inp = SrKInputs {
        cell,
        prep,
        cfg,
        shells: &shells,
        images: &images,
        ph: &ph,
        rpair,
        nk,
    };
    let ctx = SrKCtx::new(&inp, &mut ledger)?
        .ok_or_else(|| FerricError::General("hcore_kpts_s2_mutant: no nuclei".into()))?;
    let (v, n, n_ordered) = ctx.parallel(&ledger, mutant)?;
    Ok((s, t, v, n, n_ordered))
}
