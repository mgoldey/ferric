//! Amplitude-space single-threshold local MP2 (Wang/Shen/Head-Gordon,
//! JCTC 19, 7577 (2023)) — Rust Phase 2 of the prototype in
//! `scripts/amplitude_lmp2_proto.py` (measurement history in
//! `wiki/amplitude-threshold-lmp2.md` §1–§18).
//!
//! Pipeline: Boys-localized active occupieds + VV-HV orthogonal localized
//! virtuals → RI-assembled (ia|jb) in the localized basis → single threshold
//! ε gating integral magnitudes (keep (i,a,j,b) iff |(ia|jb)|>ε or
//! |(ib|ja)|>ε — the symmetric Eq-8 test, closed under the exchange swap so
//! the Hylleraas energy retains full same-spin exchange) → ragged per-pair
//! domain-block preconditioned-CG solve → Hylleraas energy.
//!
//! # Exactness anchor
//!
//! ε=0 (full mask) must reproduce [`crate::rimp2::ri_mp2`] — canonical
//! orbitals + closed-form denominators vs localized orbitals + ragged CG is
//! an independent-construction pair sharing only the RI integrals, so the
//! bar is CG tolerance (1e-9), not the RI floor. Enforced in
//! `tests/lmp2_amplitude.rs` together with the construction mutation test
//! (dropping one hard virtual must break the anchor).
//!
//! # Documented deviations from the paper (match the Python rig's wiki note)
//!
//! - Hard virtuals: spread-weighted pivoted-Cholesky selection of
//!   projected-AO candidates + Löwdin (same construction as the Python
//!   rig, using the emultipole2 second-moment integrals), instead of the
//!   paper's weighted symmetric orthogonalization of the full redundant
//!   set. Anchors are invariant to the selection weighting (the selected
//!   set spans the same space); only domain compactness depends on it.
//! - J is assembled dense from global-metric RI B tensors. The measured
//!   per-pair domain-local fit and the pre-assembly pair gate
//!   (prototype steps 4–5) are NOT yet ported; nothing here is
//!   integral-direct and no scaling claim attaches to this module yet.

use ndarray::{s, Array2};
use std::collections::HashMap;

use ferric_core::basis::BasisSet;
use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::{dipole, overlap, r2_moment};
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;

use crate::boys::boys_localize;
use crate::rimp2::{active_occ, eri3_mo_ov_blocked, eri3_budget_bytes, metric_inverse_sqrt, ri_mp2, RiMp2Config};
use ferric_integrals::threeindex::coulomb_metric_2c;

#[derive(Debug, Clone)]
pub struct AmplitudeLmp2Config {
    /// The single threshold ε on |(ia|jb)| in the localized basis (Eq. 8).
    /// ε = 0 keeps everything (the exactness-anchor limit).
    pub eps: f64,
    pub frozen_core: usize,
    pub cg_rtol: f64,
    pub cg_max_iter: usize,
    /// 3-index working-memory budget; `None` → the shared rimp2 default.
    pub eri3_budget_bytes: Option<usize>,
    /// Integral-free pair gate: `Some(cal)` skips pair blocks whose
    /// London-type estimate `cal·(σᵢσⱼ)³/R⁶` falls below θ = 1e-2·eps
    /// (the paper's linked gate) BEFORE any solver work — and before any
    /// per-pair fit when `fit_radius_bohr` is set. `cal` is the
    /// conservatively calibrated constant from the Python rig (p95:
    /// ~0.7 Coulomb, ~0.02 erfc ω=1, stable C8→C10). Diagonal pairs are
    /// never gated; eps = 0 makes the gate inert (anchor limit).
    pub pair_gate_cal: Option<f64>,
    /// `Some(r)`: per-pair domain-local same-kernel RI fit — for pair
    /// (i,j) only aux functions within `r` Bohr of either Boys centroid
    /// enter the fit (J_ij = A_D V_DD⁻¹ A_D; with one shared domain and
    /// the same-kernel metric the Dunlap robust correction cancels
    /// identically). `None`: global-metric fit (byte-identical to the
    /// original path). r ≥ 1e5 must reproduce the global fit to machine
    /// precision (trivial-limit anchor, tested).
    pub fit_radius_bohr: Option<f64>,
}

impl Default for AmplitudeLmp2Config {
    fn default() -> Self {
        Self {
            eps: 1e-4,
            frozen_core: 0,
            cg_rtol: 1e-11,
            cg_max_iter: 400,
            eri3_budget_bytes: None,
            pair_gate_cal: None,
            fit_radius_bohr: None,
        }
    }
}

#[derive(Debug)]
pub struct AmplitudeLmp2Result {
    pub e_corr: f64,
    pub e_total: f64,
    /// Canonical RI-MP2 on the same (mol, obs, dfbs, op, frozen_core) —
    /// the independent-construction reference; at ε=0 `e_corr` must match
    /// this to CG tolerance.
    pub e_corr_canonical_ri: f64,
    pub keep_fraction: f64,
    pub pair_fraction: f64,
    pub dom_mean: f64,
    pub dom_max: usize,
    pub cg_iterations: usize,
    pub cg_relres: f64,
    pub cg_converged: bool,
    pub n_valence_virt: usize,
    pub n_hard_virt: usize,
    /// Multiply-add count of one ragged matvec vs its dense equivalent
    /// (2·(no²·nv³ + no³·nv²)) — the counter pair the prototype's scaling
    /// table quotes.
    pub ragged_flops_per_matvec: u64,
    pub dense_flops_per_matvec: u64,
    /// Unique off-diagonal pairs removed by the integral-free gate
    /// (0 when the gate is off or eps = 0).
    pub n_pairs_gated: usize,
    pub timings: StageTimings,
}

/// Wall-clock per pipeline stage, seconds. `t_reference_s` is the canonical
/// ri_mp2 reference — NOT part of the method cost, reported separately so
/// benchmark tables can exclude it.
#[derive(Debug, Clone, Default)]
pub struct StageTimings {
    pub t_spaces_s: f64,
    pub t_assembly_s: f64,
    pub t_solve_s: f64,
    pub t_reference_s: f64,
}

/// VV-HV orthogonal localized virtual space.
#[derive(Debug, Clone)]
pub struct VvHv {
    /// (nao, nvir) — valence virtuals first, then hard virtuals.
    pub c_vloc: Array2<f64>,
    pub n_valence: usize,
    pub n_hard: usize,
}

// ---------------------------------------------------------------------------
// small dense-linear-algebra helpers (nao-scale eigh via ndarray-linalg)
// ---------------------------------------------------------------------------

fn eigh(m: &Array2<f64>) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    use ndarray_linalg::Eigh;
    let (w, v) = m
        .eigh(ndarray_linalg::UPLO::Lower)
        .map_err(|e| FerricError::General(format!("lmp2_amplitude eigh failed: {e}")))?;
    Ok((w.to_vec(), v))
}

/// Symmetric (Löwdin) orthonormalization of the columns of `c` w.r.t. `s`.
fn lowdin(c: &Array2<f64>, s: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let o = c.t().dot(&s.dot(c));
    let (w, v) = eigh(&o)?;
    let wmin = w.iter().cloned().fold(f64::INFINITY, f64::min);
    if wmin < 1e-10 {
        return Err(FerricError::General(format!(
            "lmp2_amplitude: Löwdin set near-singular (min eig {wmin:.2e})"
        )));
    }
    let mut vs = v.clone();
    for (k, wk) in w.iter().enumerate() {
        let f = 1.0 / wk.sqrt();
        vs.column_mut(k).mapv_inplace(|x| x * f);
    }
    Ok(c.dot(&vs.dot(&v.t())))
}

/// Canonical orthonormalization keeping the `rank` largest-eigenvalue
/// directions; errors if the kept spectrum dips below the lindep floor.
fn canonical_orth(c: &Array2<f64>, s: &Array2<f64>, rank: usize) -> Result<Array2<f64>, FerricError> {
    const LINDEP: f64 = 1e-8;
    let o = c.t().dot(&s.dot(c));
    let (w, v) = eigh(&o)?;
    // eigh returns ascending; take the top `rank`.
    let n = w.len();
    if rank > n {
        return Err(FerricError::General(format!(
            "lmp2_amplitude canonical_orth: rank {rank} > candidate dim {n}"
        )));
    }
    let kept: Vec<usize> = (n - rank..n).collect();
    let wmin = kept.iter().map(|&k| w[k]).fold(f64::INFINITY, f64::min);
    if wmin < LINDEP {
        return Err(FerricError::General(format!(
            "lmp2_amplitude canonical_orth: rank {rank} unreachable (min kept eig {wmin:.2e})"
        )));
    }
    let nao = c.nrows();
    let mut out = Array2::<f64>::zeros((nao, rank));
    for (col, &k) in kept.iter().enumerate() {
        let f = 1.0 / w[k].sqrt();
        let dir = c.dot(&v.column(k).to_owned());
        out.column_mut(col).assign(&(dir.mapv(|x| x * f)));
    }
    Ok(out)
}

/// Greedy pivoted Cholesky on a PSD matrix; returns `rank` pivot indices.
fn pivoted_cholesky_order(m: &Array2<f64>, rank: usize) -> Result<Vec<usize>, FerricError> {
    let n = m.nrows();
    let mut d: Vec<f64> = (0..n).map(|i| m[(i, i)]).collect();
    let mut l = Array2::<f64>::zeros((rank, n));
    let mut piv = Vec::with_capacity(rank);
    for k in 0..rank {
        let (j, &dj) = d
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("NaN in pivoted Cholesky diagonal"))
            .expect("empty diagonal");
        if dj <= 0.0 {
            return Err(FerricError::General(format!(
                "lmp2_amplitude: pivoted Cholesky broke down at k={k}"
            )));
        }
        piv.push(j);
        let ljk_prev = l.slice(s![..k, j]).to_owned();
        let mrow = m.row(j).to_owned();
        let proj = l.slice(s![..k, ..]).t().dot(&ljk_prev);
        let scale = 1.0 / dj.sqrt();
        for c in 0..n {
            l[(k, c)] = (mrow[c] - proj[c]) * scale;
        }
        for c in 0..n {
            d[c] -= l[(k, c)] * l[(k, c)];
        }
        for &p in &piv {
            d[p] = f64::NEG_INFINITY;
        }
    }
    Ok(piv)
}

// ---------------------------------------------------------------------------
// projected-minimal cross overlap via a combined basis
// ---------------------------------------------------------------------------

/// Overlap block ⟨obs AO | minimal AO⟩ computed by preparing ONE combined
/// basis (obs shells then STO-3G shells per element) and slicing the full
/// overlap — ferric has no cross-basis 1e path, and per-atom shell order
/// follows the merged `Vec<Shell>` order (basis_bridge walks
/// `bs.for_element` per atom), so membership is reconstructible exactly.
fn cross_overlap_with_minimal(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
) -> Result<(Array2<f64>, usize), FerricError> {
    let min_bs = ferric_core::basis::bundled("sto-3g")
        .map_err(|e| FerricError::General(format!("lmp2_amplitude: bundled sto-3g: {e}")))?;
    let mut merged: HashMap<i32, Vec<ferric_core::basis::Shell>> = HashMap::new();
    let mut n_obs_shells: HashMap<i32, usize> = HashMap::new();
    for atom in &mol.atoms {
        let z = atom.z;
        if merged.contains_key(&z) {
            continue;
        }
        let o = obs_bs
            .for_element(z)
            .ok_or_else(|| FerricError::Basis(format!("no obs shells for Z={z}")))?;
        let m = min_bs
            .for_element(z)
            .ok_or_else(|| FerricError::Basis(format!("no sto-3g shells for Z={z}")))?;
        let mut v: Vec<ferric_core::basis::Shell> = o.to_vec();
        v.extend(m.iter().cloned());
        n_obs_shells.insert(z, o.len());
        merged.insert(z, v);
    }
    let merged_bs = BasisSet { name: "lmp2-obs+sto3g".to_string(), shells: merged, ecps: obs_bs.ecps.clone() };
    let comb = PreparedBasis::new(mol, &merged_bs)?;
    let s_comb = overlap(&comb);

    // classify each combined AO as obs or minimal by walking shells per atom
    let mut obs_idx = Vec::new();
    let mut min_idx = Vec::new();
    let mut atom_shell_count: Vec<usize> = vec![0; mol.atoms.len()];
    for (sh, &ai) in comb.shell_to_atom().iter().enumerate() {
        let z = mol.atoms[ai].z;
        let k = atom_shell_count[ai];
        let dim = comb.shell_dims()[sh];
        let off = comb.shell_offsets()[sh];
        let is_obs = k < n_obs_shells[&z];
        for f in 0..dim {
            if is_obs {
                obs_idx.push(off + f);
            } else {
                min_idx.push(off + f);
            }
        }
        atom_shell_count[ai] += 1;
    }
    if obs_idx.len() != obs.nbasis() {
        return Err(FerricError::General(format!(
            "lmp2_amplitude: combined-basis bookkeeping mismatch ({} obs AOs vs {})",
            obs_idx.len(),
            obs.nbasis()
        )));
    }
    let nmin = min_idx.len();
    let mut s_x = Array2::<f64>::zeros((obs_idx.len(), nmin));
    for (r, &ir) in obs_idx.iter().enumerate() {
        for (c, &ic) in min_idx.iter().enumerate() {
            s_x[(r, c)] = s_comb[(ir, ic)];
        }
    }
    Ok((s_x, nmin))
}

/// Per-AO parent-atom map for the orbital basis.
fn ao_to_atom(prep: &PreparedBasis) -> Vec<usize> {
    let mut out = vec![0usize; prep.nbasis()];
    for (sh, &ai) in prep.shell_to_atom().iter().enumerate() {
        let off = prep.shell_offsets()[sh];
        for f in 0..prep.shell_dims()[sh] {
            out[off + f] = ai;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// VV-HV construction
// ---------------------------------------------------------------------------

/// Build the VV-HV orthogonal localized virtual space (paper §2.1; see the
/// module doc for the documented HV-selection deviation).
pub fn build_vvhv(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    rhf: &ScfResult,
) -> Result<VvHv, FerricError> {
    let s = overlap(obs);
    let nao = obs.nbasis();
    let nocc_total = (mol.nelec() as usize) / 2;
    let c_occ_all = rhf.mos_r().slice(s![.., ..nocc_total]).to_owned();

    // valence virtuals: projected STO-3G minus the occupied span
    let (s_x, nmin) = cross_overlap_with_minimal(mol, obs, obs_bs)?;
    let t = solve_spd(&s, &s_x)?; // S⁻¹ S_x : projected minimal in obs AO space
    let q_occ = c_occ_all.dot(&c_occ_all.t().dot(&s));
    let tv = &t - &q_occ.dot(&t);
    let n_l = nmin.checked_sub(nocc_total).ok_or_else(|| {
        FerricError::General(format!(
            "lmp2_amplitude: minimal basis ({nmin}) smaller than nocc ({nocc_total})"
        ))
    })?;
    let dip = dipole(obs, [0.0, 0.0, 0.0])?;
    let c_l = if n_l > 0 {
        let cl = canonical_orth(&tv, &s, n_l)?;
        boys_localize(&cl, &dip, 200).c_loc
    } else {
        Array2::<f64>::zeros((nao, 0))
    };

    // hard virtuals: project occ+valence out of each AO, select by pivoted
    // Cholesky, Löwdin, per-atom pseudo-canonicalize
    let n_h = nao - nocc_total - n_l;
    let c_h = if n_h > 0 {
        let mut c_e = Array2::<f64>::zeros((nao, nocc_total + n_l));
        c_e.slice_mut(s![.., ..nocc_total]).assign(&c_occ_all);
        c_e.slice_mut(s![.., nocc_total..]).assign(&c_l);
        let proj = c_e.dot(&c_e.t().dot(&s));
        // candidates: unit vectors minus their occ+valence projection
        let mut x = Array2::<f64>::eye(nao);
        x -= &proj; // columns are the projected AOs
        let sx = s.dot(&x);
        let mut keep_cols = Vec::new();
        for c in 0..nao {
            let nrm2 = x.column(c).dot(&sx.column(c));
            if nrm2 > 1e-8 {
                keep_cols.push((c, nrm2.sqrt()));
            }
        }
        let mut xn = Array2::<f64>::zeros((nao, keep_cols.len()));
        let mut parents = Vec::with_capacity(keep_cols.len());
        let a2a = ao_to_atom(obs);
        for (k, &(c, nrm)) in keep_cols.iter().enumerate() {
            xn.column_mut(k).assign(&(x.column(c).mapv(|v| v / nrm)));
            parents.push(a2a[c]);
        }
        // spread-weighted pivots (matches the Python rig): compact candidates
        // first, via w = 1/spread with spread = ⟨r²⟩ − |⟨r⟩|² per candidate
        let r2 = r2_moment(obs, [0.0; 3])?;
        let ncand = xn.ncols();
        let mut w = vec![0.0f64; ncand];
        for k in 0..ncand {
            let col = xn.column(k);
            let r2v = col.dot(&r2.dot(&col));
            let mut c2 = 0.0;
            for dm in dip.iter() {
                let d = col.dot(&dm.dot(&col));
                c2 += d * d;
            }
            w[k] = 1.0 / (r2v - c2).max(1e-6);
        }
        let wmax2 = w.iter().fold(0.0f64, |m, &x| m.max(x)).powi(2);
        let mut ov = xn.t().dot(&s.dot(&xn));
        for r in 0..ncand {
            for c in 0..ncand {
                ov[(r, c)] *= w[r] * w[c] / wmax2;
            }
        }
        let piv = pivoted_cholesky_order(&ov, n_h)?;
        let mut sel = Array2::<f64>::zeros((nao, n_h));
        let mut sel_parents = Vec::with_capacity(n_h);
        for (k, &p) in piv.iter().enumerate() {
            sel.column_mut(k).assign(&xn.column(p));
            sel_parents.push(parents[p]);
        }
        let mut ch = lowdin(&sel, &s)?;
        // per-atom pseudo-canonicalization (block-diagonal — cannot
        // delocalize, unlike full semicanonicalization)
        let f_ao = rhf.fock_r();
        let fh = ch.t().dot(&f_ao.dot(&ch));
        let mut by_atom: HashMap<usize, Vec<usize>> = HashMap::new();
        for (k, &a) in sel_parents.iter().enumerate() {
            by_atom.entry(a).or_default().push(k);
        }
        for idx in by_atom.values() {
            let m = idx.len();
            let mut blk = Array2::<f64>::zeros((m, m));
            for (r, &ir) in idx.iter().enumerate() {
                for (c, &ic) in idx.iter().enumerate() {
                    blk[(r, c)] = fh[(ir, ic)];
                }
            }
            let (_, u) = eigh(&blk)?;
            let cols: Vec<ndarray::Array1<f64>> =
                idx.iter().map(|&ic| ch.column(ic).to_owned()).collect();
            for (cnew, &ic) in idx.iter().enumerate() {
                let mut acc = ndarray::Array1::<f64>::zeros(nao);
                for (r, col) in cols.iter().enumerate() {
                    acc.scaled_add(u[(r, cnew)], col);
                }
                ch.column_mut(ic).assign(&acc);
            }
        }
        ch
    } else {
        Array2::<f64>::zeros((nao, 0))
    };

    let mut c_vloc = Array2::<f64>::zeros((nao, n_l + n_h));
    c_vloc.slice_mut(s![.., ..n_l]).assign(&c_l);
    c_vloc.slice_mut(s![.., n_l..]).assign(&c_h);
    Ok(VvHv { c_vloc, n_valence: n_l, n_hard: n_h })
}

fn solve_spd(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::Solve;
    let n = b.ncols();
    let mut out = Array2::<f64>::zeros((a.nrows(), n));
    for c in 0..n {
        let col = a
            .solve(&b.column(c).to_owned())
            .map_err(|e| FerricError::General(format!("lmp2_amplitude solve: {e}")))?;
        out.column_mut(c).assign(&col);
    }
    Ok(out)
}

/// Orthonormality + span check against the canonical virtuals (the
/// construction half of the exactness anchor). Returns (dev_orth, dev_span).
pub fn check_vvhv(
    obs: &PreparedBasis,
    rhf: &ScfResult,
    nocc_total: usize,
    c_vloc: &Array2<f64>,
) -> (f64, f64) {
    let s = overlap(obs);
    let o = c_vloc.t().dot(&s.dot(c_vloc));
    let n = o.nrows();
    let mut dev_orth: f64 = 0.0;
    for r in 0..n {
        for c in 0..n {
            let target = if r == c { 1.0 } else { 0.0 };
            dev_orth = dev_orth.max((o[(r, c)] - target).abs());
        }
    }
    let c_vcan = rhf.mos_r().slice(s![.., nocc_total..]).to_owned();
    let u = c_vcan.t().dot(&s.dot(c_vloc));
    let uut = u.dot(&u.t());
    let utu = u.t().dot(&u);
    let mut dev_span: f64 = 0.0;
    for (m, dim) in [(&uut, uut.nrows()), (&utu, utu.nrows())] {
        for r in 0..dim {
            for c in 0..dim {
                let target = if r == c { 1.0 } else { 0.0 };
                dev_span = dev_span.max((m[(r, c)] - target).abs());
            }
        }
    }
    (dev_orth, dev_span)
}

// ---------------------------------------------------------------------------
// ragged per-pair domain-block PCG solver
// ---------------------------------------------------------------------------

struct PairBlock {
    i: usize,
    j: usize,
    da: Vec<usize>,
    db: Vec<usize>,
    /// row-major (da.len() × db.len()) pattern
    pat: Vec<bool>,
    fvv_aa: Array2<f64>,
    fvv_bb: Array2<f64>,
    j_blk: Array2<f64>,
    denom: Array2<f64>,
    /// inverse maps: pos_da[v] = column of v in da, usize::MAX if absent
    pos_da: Vec<usize>,
    pos_db: Vec<usize>,
}

struct Ragged {
    pairs: Vec<PairBlock>,
    by_i: HashMap<usize, Vec<usize>>,
    by_j: HashMap<usize, Vec<usize>>,
}

/// Build one pair's ragged block from its dense (nv, nv) J block `g`
/// (g[a, b] = (ia|jb)). Returns None when the Eq-8 test retains nothing.
/// The swap partner (ib|ja) is g[b, a] — in-block, so the symmetric test
/// never needs any other pair's integrals.
#[allow(clippy::too_many_arguments)]
fn pair_block_from_g(
    i: usize,
    j: usize,
    g: &Array2<f64>,
    _f_oo: &Array2<f64>,
    f_vv: &Array2<f64>,
    fo: &[f64],
    fv: &[f64],
    eps: f64,
) -> Option<PairBlock> {
    let nv = g.nrows();
    let cand: Vec<usize> = (0..nv).collect();
    pair_block_from_g_cand(i, j, g, &cand, nv, f_vv, fo, fv, eps)
}

/// [`pair_block_from_g`] over a CANDIDATE index subset: `g` is
/// (cand.len(), cand.len()) with local indices mapping to the global
/// virtual indices `cand[..]`. Every Eq-8-retained element is guaranteed
/// inside the candidate square when `cand` comes from the Schwarz screen
/// (the bound never underestimates), so this is exact, not approximate.
#[allow(clippy::too_many_arguments)]
fn pair_block_from_g_cand(
    i: usize,
    j: usize,
    g: &Array2<f64>,
    cand: &[usize],
    nv_full: usize,
    f_vv: &Array2<f64>,
    fo: &[f64],
    fv: &[f64],
    eps: f64,
) -> Option<PairBlock> {
    let nv = cand.len();
    let _ = nv_full;
    let mut any_a = vec![false; nv];
    let mut any_b = vec![false; nv];
    let mut any = false;
    for a in 0..nv {
        for b in 0..nv {
            let jd = g[(a, b)].abs();
            let kd = g[(b, a)].abs();
            if eps == 0.0 || jd > eps || kd > eps {
                any_a[a] = true;
                any_b[b] = true;
                any = true;
            }
        }
    }
    if !any {
        return None;
    }
    // local (candidate-space) retained axes -> GLOBAL virtual indices
    let da_loc: Vec<usize> = (0..nv).filter(|&a| any_a[a]).collect();
    let db_loc: Vec<usize> = (0..nv).filter(|&b| any_b[b]).collect();
    let da: Vec<usize> = da_loc.iter().map(|&a| cand[a]).collect();
    let db: Vec<usize> = db_loc.iter().map(|&b| cand[b]).collect();
    let (na, nb) = (da.len(), db.len());
    let mut pat = vec![false; na * nb];
    let mut j_blk = Array2::<f64>::zeros((na, nb));
    let mut denom = Array2::<f64>::zeros((na, nb));
    for (r, &al) in da_loc.iter().enumerate() {
        for (c, &bl) in db_loc.iter().enumerate() {
            let jd = g[(al, bl)];
            let kd = g[(bl, al)];
            if eps == 0.0 || jd.abs() > eps || kd.abs() > eps {
                pat[r * nb + c] = true;
                j_blk[(r, c)] = jd;
            }
            denom[(r, c)] = fv[da[r]] + fv[db[c]] - fo[i] - fo[j];
        }
    }
    let mut fvv_aa = Array2::<f64>::zeros((na, na));
    for (r, &a) in da.iter().enumerate() {
        for (c, &a2) in da.iter().enumerate() {
            fvv_aa[(r, c)] = f_vv[(a, a2)];
        }
    }
    let mut fvv_bb = Array2::<f64>::zeros((nb, nb));
    for (r, &b) in db.iter().enumerate() {
        for (c, &b2) in db.iter().enumerate() {
            fvv_bb[(r, c)] = f_vv[(b, b2)];
        }
    }
    let nv_glob = fv.len();
    let mut pos_da = vec![usize::MAX; nv_glob];
    for (k, &a) in da.iter().enumerate() {
        pos_da[a] = k;
    }
    let mut pos_db = vec![usize::MAX; nv_glob];
    for (k, &b) in db.iter().enumerate() {
        pos_db[b] = k;
    }
    Some(PairBlock { i, j, da, db, pat, fvv_aa, fvv_bb, j_blk, denom, pos_da, pos_db })
}


fn apply_pattern(x: &mut Array2<f64>, pat: &[bool], nb: usize) {
    for (r, mut row) in x.rows_mut().into_iter().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            if !pat[r * nb + c] {
                *v = 0.0;
            }
        }
    }
}

fn dot_blocks(x: &[Array2<f64>], y: &[Array2<f64>]) -> f64 {
    x.iter().zip(y).map(|(a, b)| (a * b).sum()).sum()
}

/// Ragged PCG on the fixed pattern: solve P A P t = −P J.
fn solve_ragged(
    rg: &Ragged,
    f_oo: &Array2<f64>,
    rtol: f64,
    max_iter: usize,
) -> (Vec<Array2<f64>>, usize, f64, bool, u64) {
    let npair = rg.pairs.len();
    let zeros: Vec<Array2<f64>> = rg
        .pairs
        .iter()
        .map(|pb| Array2::<f64>::zeros((pb.da.len(), pb.db.len())))
        .collect();
    let mut rhs: Vec<Array2<f64>> = Vec::with_capacity(npair);
    for pb in &rg.pairs {
        let mut b = -pb.j_blk.clone();
        apply_pattern(&mut b, &pb.pat, pb.db.len());
        rhs.push(b);
    }
    let bnorm = dot_blocks(&rhs, &rhs).sqrt();
    if bnorm == 0.0 {
        return (zeros, 0, 0.0, true, 0);
    }
    let precond = |r: &[Array2<f64>]| -> Vec<Array2<f64>> {
        r.iter()
            .zip(&rg.pairs)
            .map(|(rb, pb)| rb / &pb.denom)
            .collect()
    };
    let mut flops_total = 0u64;
    let mut t = zeros.clone();
    let mut r = rhs.clone();
    let mut z = precond(&r);
    let mut p: Vec<Array2<f64>> = z.to_vec();
    let mut rz = dot_blocks(&r, &z);
    let mut it = 0;
    let mut relres = 1.0;
    let mut converged = false;
    while it < max_iter {
        it += 1;
        let ap = matvec_indexed(rg, f_oo, &p, &mut flops_total);
        let alpha = rz / dot_blocks(&p, &ap);
        for k in 0..npair {
            t[k].scaled_add(alpha, &p[k]);
            r[k].scaled_add(-alpha, &ap[k]);
        }
        relres = dot_blocks(&r, &r).sqrt() / bnorm;
        if relres < rtol {
            converged = true;
            break;
        }
        z = precond(&r);
        let rz_new = dot_blocks(&r, &z);
        let beta = rz_new / rz;
        for k in 0..npair {
            let mut pk = z[k].clone();
            pk.scaled_add(beta, &p[k]);
            p[k] = pk;
        }
        rz = rz_new;
    }
    let flops_per_mv = if it > 0 { flops_total / it as u64 } else { 0 };
    (t, it, relres, converged, flops_per_mv)
}

/// Index-passing form of the ragged matvec (the one actually used).
fn matvec_indexed(
    rg: &Ragged,
    f_oo: &Array2<f64>,
    t: &[Array2<f64>],
    flops: &mut u64,
) -> Vec<Array2<f64>> {
    let mut out: Vec<Array2<f64>> = Vec::with_capacity(rg.pairs.len());
    for (p, pb) in rg.pairs.iter().enumerate() {
        let (na, nb) = (pb.da.len(), pb.db.len());
        let mut r = pb.fvv_aa.dot(&t[p]);
        r += &t[p].dot(&pb.fvv_bb);
        *flops += (na * na * nb + na * nb * nb) as u64;
        out.push(r);
    }
    for (p_idx, pb) in rg.pairs.iter().enumerate() {
        for &q in &rg.by_j[&pb.j] {
            let qb = &rg.pairs[q];
            let f = f_oo[(pb.i, qb.i)];
            if f == 0.0 {
                continue;
            }
            gather_into(&mut out[p_idx], qb, pb, -f, &t[q], flops);
        }
        for &q in &rg.by_i[&pb.i] {
            let qb = &rg.pairs[q];
            let f = f_oo[(qb.j, pb.j)];
            if f == 0.0 {
                continue;
            }
            gather_into(&mut out[p_idx], qb, pb, -f, &t[q], flops);
        }
    }
    for (p_idx, pb) in rg.pairs.iter().enumerate() {
        apply_pattern(&mut out[p_idx], &pb.pat, pb.db.len());
    }
    out
}

fn gather_into(
    out: &mut Array2<f64>,
    src: &PairBlock,
    dst: &PairBlock,
    coeff: f64,
    tq: &Array2<f64>,
    flops: &mut u64,
) {
    let mut n = 0u64;
    for (r, &a) in dst.da.iter().enumerate() {
        let sr = src.pos_da[a];
        if sr == usize::MAX {
            continue;
        }
        for (c, &b) in dst.db.iter().enumerate() {
            let sc = src.pos_db[b];
            if sc == usize::MAX {
                continue;
            }
            out[(r, c)] += coeff * tq[(sr, sc)];
            n += 1;
        }
    }
    *flops += n;
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

/// Amplitude-threshold local MP2 with the VV-HV space built internally.
pub fn amplitude_lmp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLmp2Config,
) -> Result<AmplitudeLmp2Result, FerricError> {
    let vvhv = build_vvhv(mol, obs, obs_bs, rhf)?;
    let (dev_orth, dev_span) = {
        let nocc_total = (mol.nelec() as usize) / 2;
        check_vvhv(obs, rhf, nocc_total, &vvhv.c_vloc)
    };
    if dev_orth > 1e-8 || dev_span > 1e-8 {
        return Err(FerricError::General(format!(
            "lmp2_amplitude: VV-HV construction check failed (orth {dev_orth:.2e}, span {dev_span:.2e})"
        )));
    }
    amplitude_lmp2_with_virtuals(mol, obs, dfbs, op, rhf, cfg, &vvhv)
}

/// The assembled localized-basis problem: everything the solver consumes.
/// Public so tests can run an INDEPENDENT naive dense reference on the
/// exact same inputs (the ragged-vs-dense cross-check).
#[derive(Debug)]
pub struct LocalizedProblem {
    /// (no·nv, no·nv), row index i·nv+a — (ia|jb) in the localized basis.
    pub j_dense: Array2<f64>,
    pub f_oo: Array2<f64>,
    pub f_vv: Array2<f64>,
    pub no: usize,
    pub nv: usize,
    /// Boys-localized active occupied coefficients, (nao, no) — exposed so
    /// downstream methods (LinLCCD ladders) can transform additional blocks
    /// in the SAME localized basis.
    pub c_locc: Array2<f64>,
    /// Boys centroids of the active localized occupieds, (no, 3) Bohr.
    pub occ_centers: Array2<f64>,
    /// Orbital spreads σᵢ = sqrt(⟨r²⟩ − |⟨r⟩|²), Bohr.
    pub occ_spreads: Vec<f64>,
}

/// The basis-stage products (everything BEFORE any 4-index work): localized
/// coefficients, Fock blocks, centroids/spreads (the pair gate's inputs),
/// and the raw + whitening RI pieces.
pub struct LocalizedBasis {
    pub c_locc: Array2<f64>,
    pub f_oo: Array2<f64>,
    pub f_vv: Array2<f64>,
    pub occ_centers: Array2<f64>,
    pub occ_spreads: Vec<f64>,
    pub no: usize,
    pub nv: usize,
    /// UNWHITENED (P|ia), (naux, no·nv).
    pub b_flat: Array2<f64>,
    /// 2-center metric (P|Q).
    pub v2c: Array2<f64>,
}

/// Basis stage of the assembly — no (no·nv)² object is ever formed here.
pub fn assemble_basis(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLmp2Config,
    vvhv: &VvHv,
) -> Result<LocalizedBasis, FerricError> {
    let nocc_total = (mol.nelec() as usize) / 2;
    let no = active_occ(nocc_total, cfg.frozen_core)?;
    let first_occ = cfg.frozen_core;
    let c_occ_can = rhf.mos_r().slice(s![.., first_occ..nocc_total]).to_owned();
    let dip = dipole(obs, [0.0, 0.0, 0.0])?;
    let boys = boys_localize(&c_occ_can, &dip, 200);
    let c_locc = boys.c_loc;
    let occ_centers = boys.centers;
    let r2 = r2_moment(obs, [0.0; 3])?;
    let mut occ_spreads = Vec::with_capacity(no);
    for i in 0..no {
        let col = c_locc.column(i);
        let r2v = col.dot(&r2.dot(&col));
        let c2: f64 = (0..3).map(|x| occ_centers[(i, x)].powi(2)).sum();
        occ_spreads.push((r2v - c2).max(1e-10).sqrt());
    }
    let c_vloc = &vvhv.c_vloc;
    let nv = c_vloc.ncols();
    let f_ao = rhf.fock_r();
    let f_oo = c_locc.t().dot(&f_ao.dot(&c_locc));
    let f_vv = c_vloc.t().dot(&f_ao.dot(c_vloc));
    let budget = eri3_budget_bytes(cfg.eri3_budget_bytes);
    let b3 = eri3_mo_ov_blocked(op, obs, dfbs, &c_locc, c_vloc, budget)?;
    let naux = b3.shape()[0];
    let v2c = coulomb_metric_2c(op, dfbs)?;
    let b_flat = b3
        .into_shape_with_order((naux, no * nv))
        .map_err(|e| FerricError::General(format!("lmp2_amplitude reshape: {e}")))?;
    Ok(LocalizedBasis { c_locc, f_oo, f_vv, occ_centers, occ_spreads, no, nv, b_flat, v2c })
}

/// Assemble the localized-basis problem (Boys occupieds, caller's virtuals,
/// RI J via the same helpers the canonical reference uses).
pub fn assemble_localized(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLmp2Config,
    vvhv: &VvHv,
) -> Result<LocalizedProblem, FerricError> {
    let lb = assemble_basis(mol, obs, dfbs, op, rhf, cfg, vvhv)?;
    let (no, nv) = (lb.no, lb.nv);
    let j_dense = match cfg.fit_radius_bohr {
        None => {
            let vis = metric_inverse_sqrt(&lb.v2c, op)?;
            let btilde = vis.dot(&lb.b_flat); // (naux, no*nv)
            btilde.t().dot(&btilde) // (no*nv, no*nv)
        }
        Some(r) => domain_fit_j(&lb.b_flat, &lb.v2c, mol, dfbs, &lb.occ_centers, no, nv, r)?,
    };
    Ok(LocalizedProblem {
        j_dense,
        f_oo: lb.f_oo,
        f_vv: lb.f_vv,
        no,
        nv,
        c_locc: lb.c_locc,
        occ_centers: lb.occ_centers,
        occ_spreads: lb.occ_spreads,
    })
}

/// Per-pair domain-local same-kernel RI fit: J_ij = A_{i,D} V_DD⁻¹ A_{j,D}
/// with D = aux functions within `radius` Bohr of either pair centroid.
/// With one shared domain and the same-kernel metric the Dunlap robust
/// correction cancels identically (first-order fit error is zero by
/// construction) — the formulation the ferric domain-fitting lane validated.
#[allow(clippy::too_many_arguments)]
fn domain_fit_j(
    b_flat: &Array2<f64>, // (naux, no*nv), UNWHITENED (P|ia)
    v: &Array2<f64>,
    mol: &Molecule,
    dfbs: &PreparedBasis,
    occ_centers: &Array2<f64>,
    no: usize,
    nv: usize,
    radius: f64,
) -> Result<Array2<f64>, FerricError> {
    let naux = b_flat.nrows();
    // aux function -> parent atom position
    let aux_atom = ao_to_atom(dfbs);
    let mut aux_xyz = Array2::<f64>::zeros((naux, 3));
    for (p, &ai) in aux_atom.iter().enumerate() {
        let at = &mol.atoms[ai];
        aux_xyz[(p, 0)] = at.x;
        aux_xyz[(p, 1)] = at.y;
        aux_xyz[(p, 2)] = at.zpos;
    }
    let r2 = radius * radius;
    let mut j_dense = Array2::<f64>::zeros((no * nv, no * nv));
    for i in 0..no {
        for j in i..no {
            let blk = domain_fit_pair(b_flat, v, &aux_xyz, occ_centers, i, j, nv, r2, radius)?;
            for a in 0..nv {
                for b in 0..nv {
                    j_dense[(i * nv + a, j * nv + b)] = blk[(a, b)];
                    j_dense[(j * nv + b, i * nv + a)] = blk[(a, b)];
                }
            }
        }
    }
    Ok(j_dense)
}

/// One pair's domain-local same-kernel fit block J_ij (nv, nv) — see
/// [`domain_fit_j`] for the formulation notes.
#[allow(clippy::too_many_arguments)]
fn domain_fit_pair(
    b_flat: &Array2<f64>,
    v: &Array2<f64>,
    aux_xyz: &Array2<f64>,
    occ_centers: &Array2<f64>,
    i: usize,
    j: usize,
    nv: usize,
    r2: f64,
    radius: f64,
) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::Inverse;
    let naux = b_flat.nrows();
    let dist2 = |p: usize, k: usize| -> f64 {
        (0..3).map(|x| (aux_xyz[(p, x)] - occ_centers[(k, x)]).powi(2)).sum()
    };
    let dom: Vec<usize> = (0..naux).filter(|&p| dist2(p, i) <= r2 || dist2(p, j) <= r2).collect();
    if dom.is_empty() {
        return Err(FerricError::General(format!(
            "lmp2_amplitude: empty aux domain for pair ({i},{j}) at radius {radius} Bohr"
        )));
    }
    let d = dom.len();
    let mut vdd = Array2::<f64>::zeros((d, d));
    for (r_, &pp) in dom.iter().enumerate() {
        for (c_, &qq) in dom.iter().enumerate() {
            vdd[(r_, c_)] = v[(pp, qq)];
        }
    }
    let vdd_inv = vdd
        .inv()
        .map_err(|e| FerricError::General(format!("lmp2_amplitude V_DD inverse: {e}")))?;
    let mut a_i = Array2::<f64>::zeros((d, nv));
    let mut a_j = Array2::<f64>::zeros((d, nv));
    for (r_, &pp) in dom.iter().enumerate() {
        for a in 0..nv {
            a_i[(r_, a)] = b_flat[(pp, i * nv + a)];
            a_j[(r_, a)] = b_flat[(pp, j * nv + a)];
        }
    }
    Ok(a_i.t().dot(&vdd_inv.dot(&a_j)))
}

/// Same as [`amplitude_lmp2`], with a caller-supplied virtual space — the
/// mutation-test entry point (a deliberately broken space must fail the
/// ε=0 anchor).
pub fn amplitude_lmp2_with_virtuals(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &AmplitudeLmp2Config,
    vvhv: &VvHv,
) -> Result<AmplitudeLmp2Result, FerricError> {
    // ---- basis stage: NO 4-index object is formed here ----
    let t0 = std::time::Instant::now();
    let lb = assemble_basis(mol, obs, dfbs, op, rhf, cfg, vvhv)?;
    let (no, nv) = (lb.no, lb.nv);
    let (f_oo, f_vv) = (&lb.f_oo, &lb.f_vv);

    // ---- integral-free pair gate BEFORE any pair-block assembly ----
    // (theta = 1e-2*eps, the paper's linked gate; gated pairs get no GEMM,
    // no fit, no block — the "gate before assembly" structure the Python
    // rig measured, now real in Rust)
    let mut n_pairs_gated = 0usize;
    let keep_pair: Option<Vec<bool>> = cfg.pair_gate_cal.map(|cal| {
        let theta = 1e-2 * cfg.eps;
        let mut keep = vec![true; no * no];
        for i in 0..no {
            for j in 0..no {
                if i == j {
                    continue;
                }
                let rij2: f64 = (0..3)
                    .map(|x| (lb.occ_centers[(i, x)] - lb.occ_centers[(j, x)]).powi(2))
                    .sum();
                let est = cal * (lb.occ_spreads[i] * lb.occ_spreads[j]).powi(3)
                    / rij2.powi(3).max(1e-12);
                if est < theta {
                    keep[i * no + j] = false;
                    if i < j {
                        n_pairs_gated += 1;
                    }
                }
            }
        }
        keep
    });

    // ---- per-pair block assembly: the dense (no·nv)² J tensor is NEVER
    // formed. For each surviving unique pair (i<=j) one (nv, naux)x(naux, nv)
    // GEMM (or one domain-local fit) produces the block; its transpose is
    // the (j,i) block (J_jbia = J_iajb). Assembly cost therefore scales
    // with SURVIVING pairs, not no². ----
    let fo: Vec<f64> = (0..no).map(|i| f_oo[(i, i)]).collect();
    let fv: Vec<f64> = (0..nv).map(|a| f_vv[(a, a)]).collect();
    let (btilde, aux_xyz) = match cfg.fit_radius_bohr {
        None => {
            let vis = metric_inverse_sqrt(&lb.v2c, op)?;
            (Some(vis.dot(&lb.b_flat)), None)
        }
        Some(_) => {
            let aux_atom = ao_to_atom(dfbs);
            let naux = lb.b_flat.nrows();
            let mut xyz = Array2::<f64>::zeros((naux, 3));
            for (pp, &ai) in aux_atom.iter().enumerate() {
                let at = &mol.atoms[ai];
                xyz[(pp, 0)] = at.x;
                xyz[(pp, 1)] = at.y;
                xyz[(pp, 2)] = at.zpos;
            }
            (None, Some(xyz))
        }
    };
    // Schwarz column norms q[i·nv+a] = ||B̃_ia|| (whitened path): the exact
    // bound |J_iajb| <= q_ia q_jb lets each pair's GEMM run over a CANDIDATE
    // virtual subset that provably contains every Eq-8-retained element —
    // per-pair assembly cost becomes ~|C_ij|²·naux, tracking the saturated
    // domains instead of nv². eps = 0 keeps the full set (anchor path).
    let q: Option<(Vec<f64>, Vec<f64>)> = btilde.as_ref().map(|bt| {
        let mut qv = vec![0.0f64; no * nv];
        for (col, qslot) in qv.iter_mut().enumerate() {
            *qslot = bt.column(col).dot(&bt.column(col)).sqrt();
        }
        let qmax: Vec<f64> = (0..no)
            .map(|i| qv[i * nv..(i + 1) * nv].iter().cloned().fold(0.0f64, f64::max))
            .collect();
        (qv, qmax)
    });
    let mut pairs: Vec<PairBlock> = Vec::new();
    for i in 0..no {
        for j in i..no {
            if let Some(kp) = &keep_pair {
                if !kp[i * no + j] {
                    continue;
                }
            }
            // symmetric candidate set: a kept iff EITHER orientation's
            // Schwarz bound clears eps (covers the Eq-8 swap test — see
            // the module tests' screen-exactness anchor)
            let cand: Vec<usize> = match (&q, cfg.eps > 0.0) {
                (Some((qv, qmax)), true) => (0..nv)
                    .filter(|&a| {
                        qv[i * nv + a] * qmax[j] >= cfg.eps
                            || qv[j * nv + a] * qmax[i] >= cfg.eps
                    })
                    .collect(),
                _ => (0..nv).collect(),
            };
            if cand.is_empty() {
                continue;
            }
            let g: Array2<f64> = match (&btilde, &aux_xyz, cfg.fit_radius_bohr) {
                (Some(bt), _, _) => {
                    let naux = bt.nrows();
                    let nc = cand.len();
                    if nc == nv {
                        let bi = bt.slice(s![.., i * nv..(i + 1) * nv]);
                        let bj = bt.slice(s![.., j * nv..(j + 1) * nv]);
                        bi.t().dot(&bj)
                    } else {
                        let mut bi = Array2::<f64>::zeros((naux, nc));
                        let mut bj = Array2::<f64>::zeros((naux, nc));
                        for (k, &a) in cand.iter().enumerate() {
                            bi.column_mut(k).assign(&bt.column(i * nv + a));
                            bj.column_mut(k).assign(&bt.column(j * nv + a));
                        }
                        bi.t().dot(&bj)
                    }
                }
                (_, Some(xyz), Some(radius)) => domain_fit_pair(
                    &lb.b_flat,
                    &lb.v2c,
                    xyz,
                    &lb.occ_centers,
                    i,
                    j,
                    nv,
                    radius * radius,
                    radius,
                )?,
                _ => unreachable!("btilde/aux_xyz exactly one is Some"),
            };
            // domain-fit path returns a FULL (nv, nv) block; candidate
            // subsets apply to the whitened Gram path only
            let cand_used: Vec<usize> = if g.nrows() == nv && cand.len() != nv {
                unreachable!("full block with reduced candidates")
            } else if g.nrows() == nv {
                (0..nv).collect()
            } else {
                cand.clone()
            };
            if let Some(pb) =
                pair_block_from_g_cand(i, j, &g, &cand_used, nv, f_vv, &fo, &fv, cfg.eps)
            {
                pairs.push(pb);
            }
            if i != j {
                let gt = g.t().to_owned();
                if let Some(pb) =
                    pair_block_from_g_cand(j, i, &gt, &cand_used, nv, f_vv, &fo, &fv, cfg.eps)
                {
                    pairs.push(pb);
                }
            }
        }
    }
    let mut by_i: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut by_j: HashMap<usize, Vec<usize>> = HashMap::new();
    for (pidx, pb) in pairs.iter().enumerate() {
        by_i.entry(pb.i).or_default().push(pidx);
        by_j.entry(pb.j).or_default().push(pidx);
    }
    let rg = Ragged { pairs, by_i, by_j };
    let t_assembly_s = t0.elapsed().as_secs_f64();

    // canonical reference on the same (mol, basis, aux, op, frozen core)
    let t0 = std::time::Instant::now();
    let e_ref = ri_mp2(
        mol,
        obs,
        dfbs,
        op,
        rhf,
        &RiMp2Config { frozen_core: cfg.frozen_core, memory_budget_bytes: cfg.eri3_budget_bytes, ..Default::default() },
    )?
    .mp2_corr;
    let t_reference_s = t0.elapsed().as_secs_f64();

    // ---- ragged solve ----
    let t0 = std::time::Instant::now();
    let (t, iters, relres, converged, flops_mv) = solve_ragged(&rg, f_oo, cfg.cg_rtol, cfg.cg_max_iter);
    if !converged {
        return Err(FerricError::General(format!(
            "lmp2_amplitude: ragged CG failed to converge (relres {relres:.2e} after {iters} iters)"
        )));
    }

    // Hylleraas energy E = Σ (2 t_iajb − t_ibja) J_iajb over the pattern
    let mut e_dir = 0.0;
    let mut e_exx = 0.0;
    for (p, pb) in rg.pairs.iter().enumerate() {
        let nb = pb.db.len();
        for (r, &a) in pb.da.iter().enumerate() {
            for (c, &b) in pb.db.iter().enumerate() {
                if !pb.pat[r * nb + c] {
                    continue;
                }
                let jv = pb.j_blk[(r, c)];
                e_dir += t[p][(r, c)] * jv;
                // t_ibja lives on the SAME pair block at swapped positions;
                // zero (skip) when outside the union domains
                let sr = pb.pos_da[b];
                let sc = pb.pos_db[a];
                if sr != usize::MAX && sc != usize::MAX {
                    e_exx += t[p][(sr, sc)] * jv;
                }
            }
        }
    }
    let e_corr = 2.0 * e_dir - e_exx;

    // counters
    let total_el = (no * nv) as u64 * (no * nv) as u64;
    let kept: u64 = rg
        .pairs
        .iter()
        .map(|pb| pb.pat.iter().filter(|&&x| x).count() as u64)
        .sum();
    let dom: Vec<usize> = {
        // per-LMO union over j of Da
        let mut per_i: Vec<Vec<bool>> = vec![vec![false; nv]; no];
        for pb in &rg.pairs {
            for &a in &pb.da {
                per_i[pb.i][a] = true;
            }
        }
        per_i.iter().map(|v| v.iter().filter(|&&x| x).count()).collect()
    };
    let dom_max = dom.iter().copied().max().unwrap_or(0);
    let dom_mean = if no > 0 { dom.iter().sum::<usize>() as f64 / no as f64 } else { 0.0 };
    let dense_flops = 2 * ((no * no) as u64 * (nv as u64).pow(3) + (no as u64).pow(3) * (nv * nv) as u64);

    let t_solve_s = t0.elapsed().as_secs_f64();
    Ok(AmplitudeLmp2Result {
        e_corr,
        e_total: rhf.energy + e_corr,
        e_corr_canonical_ri: e_ref,
        keep_fraction: kept as f64 / total_el as f64,
        pair_fraction: rg.pairs.len() as f64 / (no * no) as f64,
        dom_mean,
        dom_max,
        cg_iterations: iters,
        cg_relres: relres,
        cg_converged: converged,
        n_valence_virt: vvhv.n_valence,
        n_hard_virt: vvhv.n_hard,
        ragged_flops_per_matvec: flops_mv,
        dense_flops_per_matvec: dense_flops,
        n_pairs_gated,
        timings: StageTimings { t_spaces_s: 0.0, t_assembly_s, t_solve_s, t_reference_s },
    })
}
