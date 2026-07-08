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
//! from the V^{-1/2} mixing). At `thresh = 0` no aux shells are dropped
//! and the result is bit-identical to the dense path.
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
// System/basis inputs plus the localized occupied set, its Boys centroids, the
// occ-window indices, and the screening threshold — independent quantities with
// no natural grouping.
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
    centroids: Vec<[f64; 3]>,
    thresh: f64,
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

    // Localized orbital energies: diagonal of C_loc^T F C_loc.
    let f = rhf.fock_r();
    let fc = f.dot(c_occ_loc);
    let f_loc = c_occ_loc.t().dot(&fc);
    let eps_loc: Vec<f64> = (0..nocc_loc).map(|i| f_loc[(i, i)]).collect();

    // OBS shell info.
    let nsh_obs = obs.nshells();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let nsh_df = dfbs.nshells();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();

    // Threshold for skipping shell pairs in the per-orbital sum.
    let shell_thresh = (thresh / 100.0).max(0.0);

    let mut p_lists: Vec<Vec<usize>> = Vec::with_capacity(nocc_loc);
    let mut tiles: Vec<Array2<f64>> = Vec::with_capacity(nocc_loc);
    let mut total_retained: usize = 0;

    // Reuse a single 3-center engine across orbitals.
    let mut eng3 = Engine::new_3center(op, obs, dfbs, 1e-14)?;

    // C7-fuse: single-pass screen + tile build. Previously each orbital ran
    // two integral passes (metric (P|i i), then tile (P|i a)) — every libint2
    // shell triple was evaluated twice. Now we evaluate each triple once and
    // accumulate both quantities from the same integral block. The transient
    // full-naux raw tile costs ~naux·nvir·8 B (~14 MB at danuglipron scale),
    // well within budget. At thresh=0 the result is bit-identical to the
    // two-pass form.
    for i_loc in 0..nocc_loc {
        let ci = c_occ_loc.slice(s![.., i_loc]).to_owned(); // (nbas,)

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
        let mut sig_pairs: Vec<(usize, usize, f64)> = Vec::new();
        for s_mu in 0..nsh_obs {
            for s_nu in 0..=s_mu {
                let b_munu = ci_shell_max[s_mu] * ci_shell_max[s_nu];
                if b_munu * q_obs[(s_mu, s_nu)] > shell_thresh {
                    sig_pairs.push((s_mu, s_nu, b_munu));
                }
            }
        }

        // 2. Single integral pass: for each (p_sh, s_mu, s_nu) compute the
        //    (P|μν) block ONCE and accumulate both:
        //      - p_ii[P] += c_i[μ] (P|μν) c_i[ν]   (density-pair metric)
        //      - raw_full[P, a] += c_i[μ] (P|μν) c_vir[ν, a]   (raw tile)
        //    Loops are structured (p_sh, sig_pair) — same iteration count as
        //    each individual pass in the original two-pass code, but only
        //    one integral evaluation per triple.
        let mut p_ii = vec![0.0f64; naux];
        let mut raw_full = Array2::<f64>::zeros((naux, nvir));

        for p_sh in 0..nsh_df {
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
                    let p_idx = p0 + p_off;
                    let mut metric_acc = 0.0f64;
                    for mi in 0..n_mu {
                        let mu = o_mu + mi;
                        let cim = ci[mu];
                        for ni in 0..n_nu {
                            let nu = o_nu + ni;
                            let v = block[(p_off * n_mu + mi) * n_nu + ni];
                            // (μν) contribution.
                            //   metric: cim * v * ci[nu]
                            //   tile:   raw_full[p_idx, a] += cim * v * c_vir[nu, a]
                            metric_acc += cim * v * ci[nu];
                            if cim != 0.0 {
                                let w = cim * v;
                                for a in 0..nvir {
                                    raw_full[(p_idx, a)] += w * c_vir[(nu, a)];
                                }
                            }
                            // (νμ) contribution by symmetry (P|μν) = (P|νμ).
                            if off_diag {
                                let cin = ci[nu];
                                metric_acc += cin * v * cim;
                                if cin != 0.0 {
                                    let w = cin * v;
                                    for a in 0..nvir {
                                        raw_full[(p_idx, a)] += w * c_vir[(mu, a)];
                                    }
                                }
                            }
                        }
                    }
                    p_ii[p_idx] += metric_acc;
                }
            }
        }

        // 3. Retain aux shells where any function p in P has |(p|i i)| > thresh.
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

        // Expand kept aux shells into kept aux function indices (sorted).
        let mut p_list: Vec<usize> = Vec::new();
        for &p_sh in &keep_p_shells {
            for p in offs_df[p_sh]..offs_df[p_sh] + dims_df[p_sh] {
                p_list.push(p);
            }
        }
        let m_i = p_list.len();

        // 4. Compact: extract retained rows from raw_full into raw (m_i × nvir).
        let mut raw = Array2::<f64>::zeros((m_i, nvir));
        for (slot, &p_idx) in p_list.iter().enumerate() {
            for a in 0..nvir {
                raw[(slot, a)] = raw_full[(p_idx, a)];
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
pub fn build_screened_bov_boys(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    frozen_core: usize,
    thresh: f64,
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
    )?;
    Ok((screened, boys))
}
