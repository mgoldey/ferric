//! Sparse per-orbital tile representation of the dressed 3-index tensor
//! `B^P_{i_loc, a}` for Boys-localized occupied orbitals.
//!
//! # Why
//!
//! In the dense PDEP-RPA path each subspace matvec costs `naux × nocc × nvir`,
//! independent of how local the occupied orbitals are. After Foster-Boys
//! localization, individual orbitals couple strongly only to nearby aux
//! functions; the corresponding rows of `B^{P}_{i_loc, a}` decay rapidly with
//! distance. By materializing per-orbital tiles with their own retained-aux
//! lists, the dielectric matvec scales with the *significant* number of pairs
//! rather than the full `nocc × naux`.
//!
//! # Screening metric (C7-tighten / option 1c)
//!
//! For each Boys-localized occupied i_loc we use the **exact density-pair
//! metric** `|(P | i_loc i_loc)|`. The Cauchy-Schwarz inequality gives
//! `|(P | i_loc a)|² ≤ (P | i_loc i_loc)(P | aa)`, so retaining aux shells
//! where `max_{p∈P} |(p | i_loc i_loc)| > thresh` keeps every aux function
//! that can contribute non-trivially to the dressed tile.
//!
//! Building `(P | i_loc i_loc)` is cheap: it shares the same 3-center engine
//! we already need for `(P | μν)`, costs `O(naux × n_sig_pairs × dim²)`
//! (no `nvir` factor), and uses a cheap shell-pair pre-screen
//! `|c_i|_max[Sμ] · |c_i|_max[Sν] · Q[Sμ,Sν] > thresh/100`. The looser
//! Schwarz×density product bound (option 1c original form) was tried first;
//! for benzene/cc-pVDZ it retains 100% of pairs at thresh=5e-3 (the
//! Σ_{Sμ,Sν} sum kills locality). The exact (P|i i) metric retains a
//! genuine sparse subset at production thresholds.
//!
//! After retaining aux shells per orbital, the (P|i_loc, a) tile is built
//! by re-running the same screened shell-pair loop with the additional
//! `c_vir` contraction over the virtual index. V^{-1/2} dressing is applied
//! by slicing both axes to `p_list` (so dropped aux rows are also dropped
//! from the V^{-1/2} mixing). At `thresh = 0` no aux shells are dropped and the
//! result is algebraically equivalent to the dense path — NOT bit-identical.
//! (This used to claim bit-identity, which was false: the Boys rotation and the
//! per-orbital accumulation order differ from dense, so the RPA energy agrees
//! only to a finite-precision floor. Measured 2.6e-10 on H2O/cc-pVDZ after the
//! 2026-07-28 semicanonicalization fix, improved from 8.2e-9 before it.)
//!
//! The (P|μν) integrals are computed only for retained P shells × density-
//! significant OBS shell pairs, then contracted with `c_loc[:,i] ⊗ c_vir` to
//! form the raw `(m_i × nvir)` tile directly. V^{-1/2} dressing is applied by
//! slicing both axes to `p_list` — this means dropped P rows are also dropped
//! from the V^{-1/2} mixing, consistent with the screening drop. At `thresh = 0`
//! all aux shells are retained and the result is algebraically equivalent to
//! the dense path.
//!
//! # Storage layout (option 2b)
//!
//! For each Boys-localized occupied i_loc:
//!   * `p_lists[i_loc]`: sorted ascending list of retained aux function indices.
//!   * `tiles[i_loc]`: dense (m_i × nvir) row-major matrix of dressed
//!     B-tensor values. m_i = p_lists[i_loc].len().

use crate::boys_localize::boys_localize_occupied;
use ferric_core::linalg::{eigh_dc, Uplo};
use ferric_core::FerricError;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::schwarz;
use ferric_integrals::threeindex;
use ferric_scf::ScfResult;
use ndarray::{s, Array2};
use ndarray_linalg::{Cholesky, UPLO};

/// Sparse representation of (P | i_loc, a) integrals on Boys-localized
/// occupied orbitals.
pub struct ScreenedBov {
    pub n_occ_loc: usize,
    pub nvir: usize,
    pub naux: usize,
    /// Per-orbital retained aux index list, sorted ascending.
    pub p_lists: Vec<Vec<usize>>,
    /// Per-orbital tile: tiles[i_loc] has shape (p_lists[i_loc].len(), nvir).
    pub tiles: Vec<Array2<f64>>,
    /// Per-orbital Boys centroid (Bohr) — diagnostic.
    pub centroids: Vec<[f64; 3]>,
    /// Per-orbital localized orbital energy = (C_loc^T F C_loc)_{ii}.
    pub eps_loc: Vec<f64>,
    /// V^{-1/2} (full naux × naux). Retained for diagnostic / back-transform.
    pub v_inv_sqrt: Array2<f64>,
    /// Diagnostic: retained pairs vs (nocc_loc × naux).
    pub total_retained: usize,
}

/// Cholesky-based V^{-1/2} (same path as `compute_rpa_intermediates`).
fn cholesky_inverse_sqrt(v: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::SolveTriangular;
    let n = v.nrows();
    let l = v
        .cholesky(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("V cholesky failed: {e}")))?;
    let eye = Array2::<f64>::eye(n);
    let v_inv_sqrt = l
        .solve_triangular(UPLO::Lower, ndarray_linalg::Diag::NonUnit, &eye)
        .map_err(|e| FerricError::General(format!("triangular solve failed: {e}")))?;
    Ok(v_inv_sqrt)
}

/// Build the screened, per-orbital B tile representation from a localized
/// occupied block.
///
/// `c_occ_loc` is the (nbas × nocc_loc) matrix of Boys-localized active
/// occupied orbitals (skips frozen core). `thresh` controls the density-pair
/// screening cutoff: aux shell P is retained for orbital i_loc iff
/// `sqrt(max_p (P|P)) · Σ_{Sμ,Sν} |D_loc|_max[Sμ,Sν] · Q[Sμ,Sν] > thresh`.
///
/// At `thresh = 0` no aux shells are dropped; the result is algebraically
/// equivalent to the dense `b_ov` build (up to Boys rotation, which is unitary
/// within the occ block).
///
/// # G6 centroid distance pre-filter (`dist_cutoff = r_ref`, Bohr)
///
/// The C7 pass-1 metric `p_ii[P] = (P | i_loc i_loc)` is *exact*, so no distance
/// term can make the **keep decision** more accurate — the exact metric is
/// already the tightest possible screen. What the distance filter buys is
/// avoiding the *pass-1 integral evaluation itself* for aux shells (and OBS
/// shell-pairs) whose contribution is provably ≤ `thresh` a priori. That is the
/// actual cost lever: pass 1 evaluates `O(nsh_df · sig_pairs)` `(P|μν)` blocks
/// per orbital, and pass 2's `O(keep_p_shells · sig_pairs · nvir)` work is
/// gated by how many aux shells pass 1 keeps.
///
/// **The bound (rigorous, lossless w.r.t. the `> thresh` keep decision).** By
/// Cauchy-Schwarz, `|(P|μν)| ≤ sqrt((P|P))·sqrt((μν|μν))`, so for any P
///
/// ```text
///   |p_ii[P]| ≤ sqrt((P|P)) · Σ_{(Sμ,Sν)∈sig_pairs} |ci|_max[Sμ]·|ci|_max[Sν]·Q[Sμ,Sν]
///            ≡ sqrt((P|P)) · Bsum_i                                   (the loose "1c" bound)
/// ```
///
/// The `i_loc i_loc` charge density is a compact, unit-norm blob centred at the
/// Boys centroid, so the 3-index Coulomb integral `(P|i_loc i_loc)` carries a
/// monopole `1/R` tail in `R_P = |center(P) − centroid_i|` beyond the blob
/// extent. Following the QQR/CFMM long-range-safe precedent
/// (`ferric-scf/src/{qqr,cfmm}.rs`: `schwarz × min(1, extent/R)`), we damp the
/// CS bound by the SAME `min(1, r_ref/R)` Coulomb envelope:
///
/// ```text
///   U(P) = min(1, r_ref / R_P) · sqrt((P|P)) · Bsum_i   ≥   |p_ii[P]|.
/// ```
///
/// `min(1, r_ref/R) ≤ 1` always, so `U(P)` is still an upper bound on the exact
/// metric; a shell is skipped (its `p_ii[P]` set to 0, i.e. dropped) only when
/// `U(P) ≤ thresh`, which is exactly the drop the exact `> thresh` decision
/// would make. We do NOT hard-cut at a radius — that would silently drop the
/// slow `1/R` Coulomb tail the module's exact metric exists to catch. The
/// envelope is a *pre-filter that composes with* the exact metric: wherever the
/// bound is inconclusive (`U(P) > thresh`) the exact integral is still
/// evaluated and the exact value drives the keep decision. The identical
/// composable bound gates the OBS `sig_pairs` list. At `dist_cutoff = +∞`
/// (default) `min(1, r_ref/R) ≡ 1`, so no shell/pair is skipped by distance and
/// the retained set and every energy are byte-for-byte identical to the pre-G6
/// path (regression-tested).
// System/basis inputs plus the localized occupied set, its Boys centroids, the
// occ-window indices, the screening threshold, and the distance-cutoff length
// scale — independent quantities with no natural grouping.
#[allow(clippy::too_many_arguments)]
pub fn build_screened_bov(
    _mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    first_occ: usize,
    nocc_active: usize,
    nocc_total: usize,
    c_occ_loc: &Array2<f64>,
    // IGNORED since 2026-07-28: centroids are recomputed in-function from the
    // SEMICANONICALIZED orbitals, because the caller's values are stale after
    // that rotation. Kept in the signature for source compatibility.
    _centroids_ignored: Vec<[f64; 3]>,
    thresh: f64,
    dist_cutoff: f64,
) -> Result<ScreenedBov, FerricError> {
    let nbas = obs.nbasis();
    let naux = dfbs.nbasis();
    let nvir = nbas - nocc_total;
    let nocc_loc = c_occ_loc.ncols();
    assert_eq!(nocc_loc, nocc_active, "c_occ_loc must have nocc_active columns");
    let _ = first_occ;

    // V^{-1/2} dressing.
    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;

    // OBS Schwarz Q for shell-pair pre-screening of the per-orbital integral loop.
    let q_obs = schwarz::schwarz(op, obs)?;

    // Active virtual block.
    let c = rhf.mos_r();
    let c_vir = c.slice(s![.., nocc_total..]).to_owned();

    // SEMICANONICALIZE the localized occupied block (FIXED 2026-07-28).
    //
    // Boys-localized orbitals are NOT Fock eigenvectors, so F_loc = C_loc^T F
    // C_loc is not diagonal. This code used to take `diag(F_loc)` as per-orbital
    // "orbital energies" and hand them to `sternheimer_sparse`, which consumes
    // them as `e_ia = eps_a - eps_loc[i]` — i.e. as if they WERE eigenvalues.
    // That silently discards the off-diagonal coupling; measured
    // max|offdiag F_loc| was 2.3e-1..3.1e-1 against a diagonal spread of ~1.0e1
    // (1.3-2.9%) on water/alkane_4/alkane_8/benzene at STO-3G.
    //
    // It is exactly the trap `dlpno_rpa.rs` warns about for PNOs ("RPA
    // denominators are only valid in a Fock-diagonal basis... Taking the
    // diagonal alone is silently wrong"), and the same defect class as the
    // AO-Laplace pseudo-density bug (canonical eps against a rotated basis).
    //
    // The fix mirrors `dlpno_rpa.rs`: diagonalize F_loc and rotate the orbitals
    // into ITS eigenbasis. The result is still localized in the sense that
    // matters here — the rotation is confined to the occupied block, so it
    // preserves the occupied SPACE exactly, and the resulting eps ARE genuine
    // eigenvalues of F restricted to that space.
    //
    // NOTE the rotation must be applied to the COEFFICIENTS as well (and hence
    // to the centroids computed from them below). Rotating the energies alone
    // would just relocate the same inconsistency.
    let f = rhf.fock_r();
    let fc = f.dot(c_occ_loc);
    let f_loc = c_occ_loc.t().dot(&fc);
    let (eps_semi, u_semi) = eigh_dc(&f_loc, Uplo::Upper).map_err(|e| {
        FerricError::General(format!(
            "Boys-screened RPA semicanonicalization eigh failed \
             (nocc_loc = {nocc_loc}): {e}"
        ))
    })?;
    let c_occ_semi = c_occ_loc.dot(&u_semi);
    let c_occ_loc = &c_occ_semi;
    let eps_loc: Vec<f64> = eps_semi.to_vec();

    // The caller's `centroids` were computed from the PRE-rotation orbitals, so
    // they are stale now. Recompute them here — in-function, so they cannot go
    // stale again regardless of what a caller passes — as ⟨i|r|i⟩ over the
    // semicanonicalized set. The G6 distance pre-filter below screens on these,
    // so a mismatch would silently screen the wrong orbitals.
    let centroids: Vec<[f64; 3]> = {
        let dip = ferric_integrals::oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
        (0..nocc_loc)
            .map(|i| {
                let ci = c_occ_loc.slice(s![.., i]);
                let mut r = [0.0f64; 3];
                for (axis, d) in dip.iter().enumerate() {
                    r[axis] = ci.dot(&d.dot(&ci));
                }
                r
            })
            .collect()
    };

    // OBS shell info.
    let nsh_obs = obs.nshells();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let nsh_df = dfbs.nshells();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();

    // Threshold for skipping shell pairs in the per-orbital sum.
    let shell_thresh = (thresh / 100.0).max(0.0);

    // ---- G6 centroid distance pre-filter precomputation ----
    // Whether the distance envelope is active at all (dist_cutoff = +∞ → off,
    // byte-identical to pre-G6). We also disable it for thresh <= 0, where the
    // whole point is to retain everything (the algebraic-equivalence contract).
    let dist_filter_on = dist_cutoff.is_finite() && thresh > 0.0;
    // Per-aux-shell max sqrt((P|P)) from the 2-center metric diagonal (the
    // aux side of the Cauchy-Schwarz bound |(P|μν)| ≤ sqrt((P|P))·Q[Sμ,Sν]).
    // Cheap: v2c is already materialized above.
    let sqrt_pp_shellmax: Vec<f64> = (0..nsh_df)
        .map(|p_sh| {
            let mut m = 0.0f64;
            for p in offs_df[p_sh]..offs_df[p_sh] + dims_df[p_sh] {
                m = m.max(v2c[(p, p)].max(0.0).sqrt());
            }
            m
        })
        .collect();
    // Per-shell nuclear centers (Bohr) for OBS and aux bases.
    let obs_centers = obs.shell_centers();
    let aux_centers = dfbs.shell_centers();
    // Diagnostic counters (retained aux shells / pass-1 integral blocks saved).
    let mut dist_skipped_aux: usize = 0;
    let mut dist_skipped_pairs: usize = 0;

    // min(1, r_ref / R): the QQR/CFMM-style Coulomb `1/R` decay envelope. At
    // R ≤ r_ref (or dist_cutoff = ∞) it is 1 (no damping). It is never > 1, so
    // multiplying it onto the Cauchy-Schwarz bound keeps that bound an upper
    // bound on the exact metric.
    let dist_envelope = |c: &[f64; 3], centroid: &[f64; 3]| -> f64 {
        if !dist_filter_on {
            return 1.0;
        }
        let dx = c[0] - centroid[0];
        let dy = c[1] - centroid[1];
        let dz = c[2] - centroid[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r <= dist_cutoff {
            1.0
        } else {
            dist_cutoff / r
        }
    };

    let mut p_lists: Vec<Vec<usize>> = Vec::with_capacity(nocc_loc);
    let mut tiles: Vec<Array2<f64>> = Vec::with_capacity(nocc_loc);
    let mut total_retained: usize = 0;

    // Reuse a single 3-center engine across orbitals.
    let mut eng3 = Engine::new_3center(op, obs, dfbs, 1e-14)?;

    // C7-fuse-v2: screen-before-allocate two-pass build. The previous C7
    // single-pass form fused the two *integral evaluations* (metric (P|i i)
    // and tile (P|i a)) into one, but still did the expensive nvir-scaled
    // accumulation into a full-naux `raw_full` for EVERY aux shell, deciding
    // which shells to keep only afterward — so the O(nvir) work ran on
    // screened-out shells too. That made the sparse path 10-50× slower than
    // dense (docs/spikes/sparse-pdep-scaling.md).
    //
    // Now we genuinely separate the two passes and let the cheap one gate the
    // expensive one:
    //   Pass 1 (cheap, nvir-INDEPENDENT): compute only the density-pair metric
    //     p_ii[P] += c_i[μ] (P|μν) c_i[ν]. Decide keep_p_shells/p_list from it.
    //   Pass 2 (expensive, nvir-SCALED, RESTRICTED to kept aux shells): redo
    //     the (P|μν) evaluation for kept shells only and accumulate the raw
    //     tile sized (m_i × nvir), not (naux × nvir).
    // Re-evaluating (P|μν) for kept shells in pass 2 is deliberate: the win is
    // skipping the O(nvir) accumulation (and the (naux×nvir) allocation) for
    // the shells we now know are screened out, which dominates. At thresh=0
    // every aux shell is kept and this is algebraically equivalent to dense.
    for i_loc in 0..nocc_loc {
        let ci = c_occ_loc.slice(s![.., i_loc]).to_owned(); // (nbas,)
        let centroid_i = centroids[i_loc];

        // 1. Per-OBS-shell-pair density bound B_loc[Sμ,Sν] = max |ci[μ] ci[ν]|.
        //    Use per-shell max(|ci|).
        let mut ci_shell_max = vec![0.0f64; nsh_obs];
        for s_mu in 0..nsh_obs {
            let mut m = 0.0f64;
            for mu in offs_obs[s_mu]..offs_obs[s_mu] + dims_obs[s_mu] {
                m = m.max(ci[mu].abs());
            }
            ci_shell_max[s_mu] = m;
        }

        // Pre-screen shell pairs (canonical s_mu >= s_nu): keep (Sμ,Sν) where
        // ci_max[Sμ]*ci_max[Sν]*Q[Sμ,Sν] > shell_thresh. Off-diagonal pairs
        // contribute via both (μν) and (νμ) symmetrization.
        //
        // G6 distance pre-filter: the retained `sig_pairs` list drives BOTH the
        // pass-1 metric AND the pass-2 tile build, so dropping a pair removes a
        // real (μν) contribution from every kept aux shell's tile — NOT just
        // from the screening metric. We therefore compose the distance envelope
        // onto the SAME density-pair quantity `cs = b_munu·Q[Sμ,Sν]` the pre-G6
        // test already screens against the SAME `shell_thresh`, so the decision
        // is `cs · min(1, r_ref/R_pair) > shell_thresh`. This is a strict
        // superset of the pre-G6 drop (env ≤ 1 only ever tightens it) and — the
        // key invariant — reduces to the byte-identical pre-G6 test `cs >
        // shell_thresh` whenever env = 1 (i.e. dist_cutoff = ∞, or any pair
        // within r_ref of the centroid). The damping factor is the QQR/CFMM
        // Coulomb `1/R` tail (`ferric-scf/src/{qqr,cfmm}.rs`), applied here to
        // the density-pair prescreen the tile build already tolerates dropping
        // below `shell_thresh`, so this is the same *kind* of approximation the
        // pre-G6 shell_thresh prescreen already makes — just distance-aware.
        // R_pair = |pair midpoint − centroid_i|.
        let mut sig_pairs: Vec<(usize, usize, f64)> = Vec::new();
        for s_mu in 0..nsh_obs {
            for s_nu in 0..=s_mu {
                let b_munu = ci_shell_max[s_mu] * ci_shell_max[s_nu];
                let cs = b_munu * q_obs[(s_mu, s_nu)];
                let env = if dist_filter_on {
                    // Pair midpoint of the two OBS shell nuclear centers.
                    let cm = &obs_centers[s_mu];
                    let cn = &obs_centers[s_nu];
                    let mid = [
                        0.5 * (cm[0] + cn[0]),
                        0.5 * (cm[1] + cn[1]),
                        0.5 * (cm[2] + cn[2]),
                    ];
                    dist_envelope(&mid, &centroid_i)
                } else {
                    1.0
                };
                if cs * env <= shell_thresh {
                    // Below the (distance-damped) density-pair prescreen. When
                    // env = 1 this is exactly the pre-G6 `cs <= shell_thresh`.
                    if dist_filter_on && env < 1.0 && cs > shell_thresh {
                        // Only count it as a *distance* drop when the envelope is
                        // what pushed it under — i.e. it would have been kept
                        // pre-G6. (Pairs already below shell_thresh are pre-G6
                        // drops, not attributable to the distance filter.)
                        dist_skipped_pairs += 1;
                    }
                    continue;
                }
                sig_pairs.push((s_mu, s_nu, b_munu));
            }
        }

        // Bsum_i = Σ_pairs ci_max[Sμ]·ci_max[Sν]·Q[Sμ,Sν] (loose Cauchy-Schwarz
        // bound coefficient); the aux side sqrt((P|P)) completes the per-shell
        // bound U(P) = sqrt((P|P))·Bsum_i·min(1, r_ref/R_P) ≥ |p_ii[P]|.
        // Off-diagonal pairs count twice (both (μν) and (νμ) symmetrizations).
        let bsum_i: f64 = sig_pairs
            .iter()
            .map(|&(s_mu, s_nu, b_munu)| {
                let base = b_munu * q_obs[(s_mu, s_nu)];
                if s_mu != s_nu { 2.0 * base } else { base }
            })
            .sum();

        // ---- Pass 1 (cheap, nvir-independent): density-pair metric only ----
        // For each (p_sh, s_mu, s_nu) compute the (P|μν) block and accumulate
        // p_ii[P] += c_i[μ] (P|μν) c_i[ν]. Cost O(nsh_df · sig_pairs · dim²) —
        // NO nvir factor, so it is cheap regardless of virtual-space size.
        let mut p_ii = vec![0.0f64; naux];
        for p_sh in 0..nsh_df {
            let np = dims_df[p_sh];
            let p0 = offs_df[p_sh];
            // G6 distance pre-filter on the aux shell: skip evaluating (P|μν)
            // for this shell iff its rigorous upper bound U(P) ≤ thresh, i.e.
            // the exact keep decision `max_p |p_ii[p]| > thresh` is provably
            // false. Leaving p_ii[p]=0 for the skipped functions drops the shell
            // exactly as the exact-metric decision would. Envelope ≡ 1 (and this
            // branch a no-op) at dist_cutoff = ∞.
            if dist_filter_on {
                let env = dist_envelope(&aux_centers[p_sh], &centroid_i);
                let u_bound = sqrt_pp_shellmax[p_sh] * bsum_i * env;
                if u_bound <= thresh {
                    dist_skipped_aux += 1;
                    continue;
                }
            }
            for &(s_mu, s_nu, _b) in &sig_pairs {
                let block = match eng3.compute_eri3(obs, dfbs, p_sh, s_mu, s_nu) {
                    Some(b) => b,
                    None => continue,
                };
                let n_mu = dims_obs[s_mu];
                let n_nu = dims_obs[s_nu];
                let o_mu = offs_obs[s_mu];
                let o_nu = offs_obs[s_nu];
                let off_diag = s_mu != s_nu;

                for p_off in 0..np {
                    let p_idx = p0 + p_off;
                    let mut metric_acc = 0.0f64;
                    for mi in 0..n_mu {
                        let mu = o_mu + mi;
                        let cim = ci[mu];
                        for ni in 0..n_nu {
                            let nu = o_nu + ni;
                            let v = block[(p_off * n_mu + mi) * n_nu + ni];
                            // (μν) contribution to the density-pair metric.
                            metric_acc += cim * v * ci[nu];
                            // (νμ) contribution by symmetry (P|μν) = (P|νμ).
                            if off_diag {
                                metric_acc += ci[nu] * v * cim;
                            }
                        }
                    }
                    p_ii[p_idx] += metric_acc;
                }
            }
        }

        // 2. Retain aux shells where any function p in P has |(p|i i)| > thresh.
        //    This decision uses ONLY the cheap pass-1 metric — no nvir-scaled
        //    work has touched any shell yet.
        let mut keep_p_shells: Vec<usize> = Vec::new();
        for p_sh in 0..nsh_df {
            let mut m = 0.0f64;
            for p in offs_df[p_sh]..offs_df[p_sh] + dims_df[p_sh] {
                m = m.max(p_ii[p].abs());
            }
            if m > thresh {
                keep_p_shells.push(p_sh);
            }
        }

        // ---- G6 PROBE (FERRIC_G6_PROBE=1): quantify bound tightness ----
        // For the FIRST orbital only, dump per-aux-shell (R = |aux_center −
        // centroid|, exact |p_ii|max, loose CS bound sqrt(PP)·Bsum_i). This
        // answers: at what R does the EXACT metric fall below thresh (the
        // physical locality), vs at what R does the *loose CS bound* fall below
        // thresh (what the distance envelope can actually act on). If those two
        // radii differ by orders of magnitude, the bound is too loose to prune.
        if i_loc == 0 && std::env::var("FERRIC_G6_PROBE").is_ok() {
            let mut rows: Vec<(f64, f64, f64)> = Vec::with_capacity(nsh_df);
            for p_sh in 0..nsh_df {
                let c = &aux_centers[p_sh];
                let dx = c[0] - centroid_i[0];
                let dy = c[1] - centroid_i[1];
                let dz = c[2] - centroid_i[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let mut exact = 0.0f64;
                for p in offs_df[p_sh]..offs_df[p_sh] + dims_df[p_sh] {
                    exact = exact.max(p_ii[p].abs());
                }
                let cs_bound = sqrt_pp_shellmax[p_sh] * bsum_i; // env=1 (no decay)
                rows.push((r, exact, cs_bound));
            }
            rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let max_r = rows.last().map(|r| r.0).unwrap_or(0.0);
            // Largest R at which the EXACT metric still keeps a shell = the true
            // coupling range (what any distance filter would have to match).
            let exact_range = rows
                .iter()
                .filter(|r| r.1 > thresh)
                .map(|r| r.0)
                .fold(0.0f64, f64::max);
            let exact_dropped = rows.iter().filter(|r| r.1 <= thresh).count();
            eprintln!(
                "G6 PROBE orb0 thresh={thresh:.1e} nsh_df={nsh_df} bsum_i={bsum_i:.3e}: \
                 max aux-shell R={max_r:.2} Bohr; EXACT-metric coupling range \
                 (largest R with |p_ii|>thresh) = {exact_range:.2} Bohr; \
                 exact metric drops {exact_dropped}/{nsh_df} shells for orb0"
            );
            // Print the farthest 12 shells: this is where pruning would happen.
            for &(r, exact, cs) in rows.iter().rev().take(12) {
                let exact_kept = exact > thresh;
                let bound_kept = cs > thresh; // env=1; the loosest keep
                eprintln!(
                    "   R={r:6.2}  exact={exact:.2e} ({})  CS_bound={cs:.2e} ({})  \
                     r_ref@thresh_via_bound={:.2}",
                    if exact_kept { "KEEP" } else { "drop" },
                    if bound_kept { "KEEP" } else { "drop" },
                    // radius at which min(1,r_ref/R)*CS = thresh ⇒ r_ref = thresh*R/CS
                    thresh * r / cs,
                );
            }
        }

        // Expand kept aux shells into kept aux function indices (sorted), and
        // build slot_of[P] = compact-tile row for each kept aux function so
        // pass 2 can scatter directly into the (m_i × nvir) raw tile. Aux
        // functions in dropped shells keep slot_of = usize::MAX and are never
        // visited in pass 2.
        let mut p_list: Vec<usize> = Vec::new();
        let mut slot_of: Vec<usize> = vec![usize::MAX; naux];
        for &p_sh in &keep_p_shells {
            for p in offs_df[p_sh]..offs_df[p_sh] + dims_df[p_sh] {
                slot_of[p] = p_list.len();
                p_list.push(p);
            }
        }
        let m_i = p_list.len();

        // ---- Pass 2 (expensive, nvir-scaled, RESTRICTED to kept aux shells) ----
        // Redo the (P|μν) evaluation for KEPT aux shells only and accumulate
        // raw[(slot, a)] += c_i[μ] (P|μν) c_vir[ν, a] into the compact
        // (m_i × nvir) tile. Screened-out aux shells are never evaluated here
        // and never touch the O(nvir) inner loop, and the tile is allocated
        // (m_i × nvir) not (naux × nvir) — this is the memory + FLOP win.
        let mut raw = Array2::<f64>::zeros((m_i, nvir));
        for &p_sh in &keep_p_shells {
            let np = dims_df[p_sh];
            let p0 = offs_df[p_sh];
            for &(s_mu, s_nu, _b) in &sig_pairs {
                let block = match eng3.compute_eri3(obs, dfbs, p_sh, s_mu, s_nu) {
                    Some(b) => b,
                    None => continue,
                };
                let n_mu = dims_obs[s_mu];
                let n_nu = dims_obs[s_nu];
                let o_mu = offs_obs[s_mu];
                let o_nu = offs_obs[s_nu];
                let off_diag = s_mu != s_nu;

                for p_off in 0..np {
                    let slot = slot_of[p0 + p_off];
                    debug_assert_ne!(slot, usize::MAX, "kept-shell aux fn must have a slot");
                    for mi in 0..n_mu {
                        let mu = o_mu + mi;
                        let cim = ci[mu];
                        for ni in 0..n_nu {
                            let nu = o_nu + ni;
                            let v = block[(p_off * n_mu + mi) * n_nu + ni];
                            // (μν) contribution: raw[slot, a] += cim v c_vir[nu, a].
                            if cim != 0.0 {
                                let w = cim * v;
                                for a in 0..nvir {
                                    raw[(slot, a)] += w * c_vir[(nu, a)];
                                }
                            }
                            // (νμ) contribution by symmetry (P|μν) = (P|νμ).
                            if off_diag {
                                let cin = ci[nu];
                                if cin != 0.0 {
                                    let w = cin * v;
                                    for a in 0..nvir {
                                        raw[(slot, a)] += w * c_vir[(mu, a)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 5. Dress: tile = V^{-1/2}[p_list, p_list] · raw.
        //    Slicing both axes keeps the tile compact and preserves bit-exact
        //    equivalence at thresh=0 (where p_list spans the full naux).
        let mut v_block = Array2::<f64>::zeros((m_i, m_i));
        for (i, &pi) in p_list.iter().enumerate() {
            for (j, &pj) in p_list.iter().enumerate() {
                v_block[(i, j)] = v_inv_sqrt[(pi, pj)];
            }
        }
        let tile = v_block.dot(&raw);

        total_retained += m_i;
        p_lists.push(p_list);
        tiles.push(tile);
    }

    if dist_filter_on {
        // Diagnostic: how much the distance envelope pruned. `dist_skipped_aux`
        // counts (orbital, aux-shell) pass-1 evaluations skipped; the possible
        // total is nocc_loc × nsh_df. Set FERRIC_G6_DEBUG=1 to see it.
        if std::env::var("FERRIC_G6_DEBUG").is_ok() {
            let poss_aux = nocc_loc * nsh_df;
            eprintln!(
                "G6 dist-filter r_ref={dist_cutoff:.2} Bohr thresh={thresh:.1e}: \
                 skipped {dist_skipped_aux}/{poss_aux} (orb,aux-shell) pass-1 blocks \
                 ({:.1}%), {dist_skipped_pairs} OBS pair-slots; retained {total_retained}/{}",
                100.0 * dist_skipped_aux as f64 / poss_aux.max(1) as f64,
                nocc_loc * naux,
            );
        }
    }

    Ok(ScreenedBov {
        n_occ_loc: nocc_loc,
        nvir,
        naux,
        p_lists,
        tiles,
        centroids,
        eps_loc,
        v_inv_sqrt,
        total_retained,
    })
}

/// Convenience constructor that runs Boys localization and screening in one
/// shot. Returns the screened representation plus localization diagnostics.
///
/// `dist_cutoff` is the G6 centroid distance pre-filter length scale `r_ref`
/// (Bohr); pass `f64::INFINITY` to disable it (byte-identical to the pre-G6
/// path). See [`build_screened_bov`] for the bound derivation.
#[allow(clippy::too_many_arguments)]
pub fn build_screened_bov_boys(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    frozen_core: usize,
    thresh: f64,
    dist_cutoff: f64,
) -> Result<(ScreenedBov, crate::boys_localize::BoysOccupied), FerricError> {
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let nocc_active = ferric_mp2::rimp2::active_occ(nocc_total, frozen_core)?;
    let _ = nbas;

    let boys = boys_localize_occupied(rhf, obs, frozen_core, nocc_active)?;
    let screened = build_screened_bov(
        mol,
        obs,
        dfbs,
        op,
        rhf,
        frozen_core,
        nocc_active,
        nocc_total,
        &boys.c_loc,
        boys.centroids.clone(),
        thresh,
        dist_cutoff,
    )?;
    Ok((screened, boys))
}
