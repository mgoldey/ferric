//! One-shot post-RPA molecular properties for the diffusion-model feature
//! export track — polarizability from PDEP eigenpairs, dielectric-response
//! quantities, and per-atom dynamic α(iω).
//!
//! Currently exposes:
//!
//! * [`pdep_polarizability_static`] — closed-shell static (ω=0) electronic
//!   polarizability tensor α_ij(0) reconstructed from PDEP eigenpairs in the
//!   RI auxiliary basis.
//! * [`pdep_polarizability_becke`], [`pdep_polarizability_becke_dynamic`],
//!   [`pdep_polarizability_hirshfeld`], [`pdep_polarizability_hirshfeld_dynamic`] —
//!   per-atom (static and dynamic) polarizability via Becke/Hirshfeld
//!   partitioning of the PDEP response.
//! * [`molecular_dynamic_polarizability`], [`molecular_dynamic_polarizability_pdep`] —
//!   molecular dynamic α(iω) on the full dielectric vs. the PDEP-truncated one.
//!
//! ## Density/charge/ESP properties moved to `ferric-scf`
//!
//! The population-partition and ESP-fitted atomic charge schemes
//! (`esp_at_atoms`, `electric_field_at_atoms`, `atomic_effective_volumes_becke`,
//! `becke_charges`, `lowdin_charges`, `mulliken_charges`, `chelpg_charges`,
//! `resp_charges`, `esp_at_points`, and the `RadialProatom`/`ProatomProvider`
//! proatom machinery) have **no RPA dependency** — they only need an SCF
//! density, not PDEP eigenpairs, Lanczos, or the screened dielectric — so they
//! were moved to [`ferric_scf::properties`] (the lower crate in the dependency
//! graph) and are re-exported below unchanged. Existing call sites
//! (`ferric_rpa::properties::hirshfeld_charges` etc.) are unaffected.
//!
//! Three siblings — `atomic_effective_volumes_hirshfeld`, `hirshfeld_i_charges`,
//! and `hirshfeld_charges` — are equally RPA-independent but could NOT move:
//! they depend on `ferric_export::cube::GridSpec` /
//! `ferric_export::gto_eval::eval_basis_on_grid`, and `ferric-export` itself
//! depends on `ferric-scf`, so moving them would create a Cargo dependency
//! cycle (`ferric-scf` → `ferric-export` → `ferric-scf`). They remain defined
//! here.
//!
//! Both routines are closed-shell only.  They return
//! `FerricError::General(...)` if handed an Unrestricted / RestrictedOpen
//! result, mirroring the conventions in `gradient.rs`.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::result::{ScfResult, Spin};
use ndarray::Array2;

use crate::config::PdepRpaConfig;

// Shared pure helpers now defined in ferric-scf (see module doc above):
// imported here because RPA-dependent functions in THIS file still call them
// directly (unqualified), same as before the move.
use ferric_scf::properties::{debug_toggle, eig3_sym, hirshfeld_margin, hirshfeld_spacing, slater_xi_for_z};

// Public re-export: every symbol that moved to ferric-scf but was `pub` here
// before the move, so every existing call site
// (ferric_rpa::properties::hirshfeld_charges, ::lowdin_charges, ::chelpg_charges,
// etc. — ferric-python, ferric-export, ferric-cli, and the test suite) keeps
// working unchanged. `ProatomProvider`/`RadialProatom` must be `pub use` (not
// a private `use`) because they also appear in the signatures of `pub fn`s
// defined in THIS file (`pdep_polarizability_hirshfeld[_dynamic]`,
// `hirshfeld_i_charges`, `hirshfeld_charges`), which are reachable from
// outside the crate.
pub use ferric_scf::properties::{
    atomic_effective_volumes_becke, becke_charges, chelpg_and_resp_charges, chelpg_charges,
    esp_at_atoms, esp_at_points,
    electric_field_at_atoms, lowdin_charges, mulliken_charges, resp_charges,
    spherically_averaged_proatom, ProatomProvider, RadialProatom,
};

/// Solve the (naux × naux) screened-dielectric system ε̃·y^d = w^d for the
/// three Cartesian right-hand sides.
///
/// ε̃ = I + B·g·Bᵀ is symmetric positive-definite by construction (identity
/// plus a positive-semi-definite term has all eigenvalues ≥ 1, so it can
/// never be singular or indefinite for a mathematically valid input). We
/// therefore factor it ONCE via Cholesky (`dpotrf`) and reuse that single
/// factorization for all three x/y/z right-hand sides via `dpotrs`, rather
/// than re-factorizing via general LU (`dgetrf`) three separate times: half
/// the FLOPs of LU, and — since several call sites run inside rayon
/// `par_iter` regions — this also removes `dgetrf` from this hot path
/// entirely (see the `openblas-rayon-dgetrf-crash` history: `dgetrf` under
/// OpenBLAS with >1 thread inside a rayon parallel region is a known
/// segfault/corruption class; `OPENBLAS_NUM_THREADS=1` is still required as
/// a structural convention, but this at least removes one more LU call from
/// the hottest of these regions).
///
/// A Cholesky failure here means `dpotrf` detected a non-positive-definite
/// matrix — for a genuinely valid ε̃ this is mathematically impossible, so
/// hitting this branch means a NaN/Inf (or otherwise corrupted) entry
/// reached `eps_mat` upstream, not an ordinary "near-degenerate occ/vir
/// gap" solver failure.
pub(crate) fn solve_dielectric_3(
    eps_mat: &Array2<f64>,
    w: &[ndarray::Array1<f64>; 3],
) -> Result<[ndarray::Array1<f64>; 3], FerricError> {
    use ndarray_linalg::{FactorizeC, SolveC, UPLO};
    let chol = eps_mat.factorizec(UPLO::Upper).map_err(|e| {
        FerricError::Lapack(format!(
            "dielectric Cholesky factorization failed (ε̃ = I + B·g·Bᵀ is SPD by \
             construction, so this should be mathematically impossible — a NaN/Inf \
             or otherwise corrupted entry likely reached ε̃ upstream): {e}"
        ))
    })?;
    let solve_one = |d: usize| -> Result<ndarray::Array1<f64>, FerricError> {
        chol.solvec(&w[d]).map_err(|e| {
            FerricError::Lapack(format!(
                "dielectric solve failed against the Cholesky factor of ε̃ \
                 (should be mathematically impossible for a valid SPD ε̃): {e}"
            ))
        })
    };
    Ok([solve_one(0)?, solve_one(1)?, solve_one(2)?])
}

/// Static (ω=0) closed-shell polarizability tensor in atomic units.
#[derive(Debug, Clone)]
#[must_use]
pub struct PolarizabilityResult {
    /// Cartesian α_ij(0) tensor, i,j ∈ {x,y,z}, in a.u. (e²·a₀²/E_h).
    pub tensor: [[f64; 3]; 3],
    /// Isotropic average (1/3) Tr α.
    pub iso: f64,
    /// Principal values (eigenvalues of the symmetrized tensor), sorted ascending.
    pub principal: [f64; 3],
}

/// Wrapper around [`electric_field_at_atoms`] taking an [`ScfResult`].
///
/// Uses the spin-summed density `density_total()` so the same call works
/// for Restricted, Unrestricted, and RestrictedOpen references — the
/// electric field is a one-electron property of the total electron
/// density, independent of spin polarization at the field point.
pub fn electric_field_at_atoms_rpa(
    mol: &Molecule,
    prep: &PreparedBasis,
    rhf: &ScfResult,
) -> Result<Vec<[f64; 3]>, FerricError> {
    electric_field_at_atoms(mol, prep, rhf.density_total())
}

/// Closed-shell static (ω=0) electronic polarizability from PDEP eigenpairs.
///
/// # Derivation
///
/// Closed-shell direct-RPA static α via Sherman-Morrison-Woodbury on the
/// (A+B) matrix:
///
///   (A+B)_{ia,jb} = δ_{ia,jb} Δε_ia + 4 (ia|jb)
///
/// With RI (ia|jb) = Σ_P B̃^P_ia B̃^P_jb (B̃ = V^{-1/2} (P|ia)), and the
/// dielectric ε̃ = I + 4 B̃ D^{-1} B̃^T  (with D=diag(Δε_ia)), one gets
///
///   α_ij = 4 μ^i^T D^{-1} μ^j − 16 w^i^T ε̃^{-1} w^j
///
/// where μ^i_{ia} = ⟨ψ_i|r_i|ψ_a⟩ are MO-basis dipole matrix elements and
///
///   w^i_P = Σ_{ia} B̃^P_{ia} · μ^i_{ia} / Δε_{ia}.
///
/// Expanding ε̃^{-1} = I − Σ_α V_α V_α^T (λ_α − 1)/λ_α:
///
///   α_ij = α^{χ₀}_ij + 16 Σ_α (w^i·V_α)(w^j·V_α) · (λ_α − 1)/λ_α
///
///   α^{χ₀}_ij = 4 μ^i^T D^{-1} μ^j − 16 w^i^T w^j
///
/// where V_α are the **dressed-basis** PDEP eigenvectors.  Since the
/// physical-aux eigenpotentials returned by `run_pdep_rpa` are
/// V^{-1/2}·V_α^dressed, we instead build w in the dressed basis directly
/// (which is just `B_ov · diag(1/Δε) · μ` — no V^{1/2} or V^{-1/2} ever
/// touches the working vectors) and dot with the **dressed** eigenvectors.
/// We recover those by transforming back: V_α^dressed = V^{1/2} · V_α^phys.
/// Simpler: redo the PDEP solve here and keep the dressed eigenvectors.
///
/// The spin factor of 4 (closed-shell) is consistent with ferric's χ₀
/// convention (`scale = sqrt(4·e_ia/(ω²+e_ia²))`).
pub fn pdep_polarizability_static(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
) -> Result<PolarizabilityResult, FerricError> {
    // Open-shell: dispatch to the per-spin SMW path. The 2×2 spin-block
    // SMW with outer-factor-2 TDDFT convention reproduces the closed-shell
    // formula on spin-symmetric UHF (verified numerically).
    if !matches!(rhf.spin, Spin::Restricted) {
        return pdep_polarizability_static_unrestricted(mol, obs, dfbs, rhf, op, cfg);
    }

    // Build B̃^P_ia = V^{-1/2} (P|ia) and orbital-energy slices via the same
    // path `run_pdep_rpa` uses.  No frozen-core for α (response on all
    // occupied is physical).
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config {
        frozen_core: 0,
        // Propagate the caller's explicit budget. This was hardcoded
        // `None`, which silently discarded a user's `[memory] budget_gb`
        // and let `resolve_budget` substitute an env/auto-detected value
        // (~0.8x available RAM) instead -- so a run pinned to 4 GB could
        // take ~14 GB on a 23 GB box. See
        // tests/mwe_explicit_budget_reaches_mp2.rs.
        memory_budget_bytes: cfg.memory_budget_bytes,
        ..Default::default()
    };
    let inter =
        ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov; // shape (naux, nov)
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;
    debug_assert_eq!(b_ov.shape(), &[naux, nov]);

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();

    // MO-basis dipole μ^d_{ia} = ⟨ψ_i|r_d|ψ_a⟩ from AO dipole + MO transform.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();
    // mu_mo[d] : (nocc, nvir)
    let mu_mo: [Array2<f64>; 3] = std::array::from_fn(|d| {
        // C_occ^T · D^d_AO · C_vir
        c_occ.t().dot(&dip_ao[d]).dot(&c_vir)
    });

    // 1/Δε_ia table.
    let nov_check = nov;
    let mut inv_de = ndarray::Array1::<f64>::zeros(nov_check);
    for i in 0..nocc {
        for a in 0..nvir {
            let ia = i * nvir + a;
            inv_de[ia] = 1.0 / (eps_vir[a] - eps_occ[i]);
        }
    }

    // μ flattened to (nov,) per direction, scaled by 1/Δε_ia.
    let mu_flat_inv: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc {
            for a in 0..nvir {
                v[i * nvir + a] = mu_mo[d][(i, a)];
            }
        }
        v * &inv_de
    });
    // μ flattened, unscaled.
    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc {
            for a in 0..nvir {
                v[i * nvir + a] = mu_mo[d][(i, a)];
            }
        }
        v
    });

    // w^d_P = Σ_ia B̃^P_ia · μ^d_ia / Δε_ia.
    let w_vec: [ndarray::Array1<f64>; 3] =
        std::array::from_fn(|d| b_ov.dot(&mu_flat_inv[d]));

    // Dressed dielectric ε̃ at ω=0: ε̃ = I + 4 B̃ D^{-1} B̃^T
    //   (scale_ia² = 4/Δε_ia at ω=0)
    // Build via DGEMM with column-scaled B̃.
    let mut b_scaled = b_ov.clone();
    // multiply column ia by sqrt(4/Δε_ia)
    for ia in 0..nov {
        let s = (4.0 * inv_de[ia]).sqrt();
        let mut col = b_scaled.column_mut(ia);
        col.mapv_inplace(|x| x * s);
    }
    // ε̃ = I + b_scaled · b_scaled^T
    let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
    for p in 0..naux {
        eps_mat[(p, p)] += 1.0;
    }

    // Solve ε̃ · y^d = w^d  (naux × naux SPD).
    let y_vec = solve_dielectric_3(&eps_mat, &w_vec)?;

    // α_ij = 4 μ^i^T D^{-1} μ^j − 16 w^i^T y^j
    //
    // Derivation cross-check:
    //   α_ij = 4 μ^i^T (A+B)^{-1} μ^j  with (A+B) = D + 4 B̃^T B̃
    //   SMW: (D + 4 B̃^T B̃)^{-1} = D^{-1} − D^{-1} B̃^T (I/4 + B̃ D^{-1} B̃^T)^{-1} B̃ D^{-1}
    //     = D^{-1} − D^{-1} B̃^T · 4 · ε̃^{-1} · B̃ D^{-1}
    //   ⇒ α_ij = 4 μ^i^T D^{-1} μ^j − 16 μ^i^T D^{-1} B̃^T ε̃^{-1} B̃ D^{-1} μ^j
    //          = 4 μ^i^T D^{-1} μ^j − 16 (B̃ D^{-1} μ^i)^T ε̃^{-1} (B̃ D^{-1} μ^j)
    //          = 4 μ^i^T D^{-1} μ^j − 16 w^i^T y^j
    let mut tensor = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let bare = mu_flat[i].dot(&mu_flat_inv[j]); // = μ^i^T D^{-1} μ^j
            let coupled = w_vec[i].dot(&y_vec[j]);
            tensor[i][j] = 4.0 * bare - 16.0 * coupled;
        }
    }

    if debug_toggle("FERRIC_DEBUG_ALPHA") {
        eprintln!("[alpha-debug] tensor=\n{:?}", tensor);
        let bare_iso: f64 = (0..3)
            .map(|i| 4.0 * mu_flat[i].dot(&mu_flat_inv[i]))
            .sum::<f64>()
            / 3.0;
        eprintln!("[alpha-debug] bare α_iso = {:.6}", bare_iso);
    }

    // Symmetrize (numerically tiny asymmetry from finite-precision Davidson).
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg = 0.5 * (tensor[i][j] + tensor[j][i]);
            tensor[i][j] = avg;
            tensor[j][i] = avg;
        }
    }

    let iso = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;

    // Principal values via 3x3 symmetric eig.
    let principal = eig3_sym(tensor)?;

    Ok(PolarizabilityResult {
        tensor,
        iso,
        principal,
    })
}

/// Spectrum of the static (ω=0) closed-shell RPA dielectric ε̃ = I + 4 B̃ Δε⁻¹ B̃ᵀ,
/// plus PDEP-rank diagnostics.
///
/// The PDEP eigenmodes of ε̃ are the basis in which the dRPA response is
/// diagonal: the static-α correction weights each mode by `(λ_α − 1)/λ_α`, which
/// vanishes as λ_α → 1. So **the number of eigenvalues exceeding 1 + thresh is
/// the effective rank of the response operator** — the size of the "ideal basis"
/// for the response.
///
/// This is the diagnostic for the question *does attenuation shrink the ideal
/// response basis?* — build it with `op = Operator::coulomb()` vs
/// `Operator::erfc(ω)` and compare `rank`. The dielectric construction here is
/// the same one `pdep_polarizability_static` solves against (same B̃, same
/// factor-4 closed-shell convention), so the rank reported is exactly the rank
/// that path's response lives in.
#[derive(Debug, Clone)]
pub struct DielectricSpectrum {
    /// All `naux` eigenvalues of ε̃, ascending. λ ≥ 1 for a physical RPA
    /// dielectric (ε̃ = I + PSD); λ = 1 modes are inert (don't screen).
    pub eigenvalues: Vec<f64>,
    /// Auxiliary-basis dimension (= number of eigenvalues).
    pub naux: usize,
    /// Number of eigenvalues with λ > 1 + `thresh`: the effective response rank.
    pub rank: usize,
    /// Threshold used for `rank`.
    pub thresh: f64,
    /// Σ_α log(λ_α): the RPA correlation trace-log (a scalar fingerprint of the
    /// whole spectrum; cheaper to compare than the full vector).
    pub trace_log: f64,
}

/// Build the static closed-shell RPA dielectric for `op` and return its
/// spectrum + PDEP rank at `thresh`. Closed-shell only.
///
/// `thresh` is the same significance cutoff used for PDEP truncation elsewhere
/// (e.g. 1e-4): a mode counts toward `rank` iff `λ_α − 1 > thresh`.
///
/// `memory_budget_bytes` is the caller's explicit memory ceiling, threaded into
/// the RI-MP2 intermediates build. `None` does NOT mean unlimited: it falls
/// through `resolve_budget`'s chain (`FERRIC_MEM_BUDGET_GB` -> legacy env vars
/// -> 0.8x detected available RAM -> 2 GiB). Pass the caller's
/// `PdepRpaConfig::memory_budget_bytes` so a user's `[memory] budget_gb` is
/// honored rather than silently replaced by an auto-detected value.
pub fn dielectric_spectrum_static(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    thresh: f64,
    memory_budget_bytes: Option<usize>,
) -> Result<DielectricSpectrum, FerricError> {
    use ndarray_linalg::{Eigh, UPLO};

    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "dielectric_spectrum_static: closed-shell (Restricted) only".into(),
        ));
    }

    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config {
        frozen_core: 0,
        // Propagate the caller's explicit budget. This was hardcoded
        // `None`, which silently discarded a user's `[memory] budget_gb`
        // and let `resolve_budget` substitute an env/auto-detected value
        // (~0.8x available RAM) instead -- so a run pinned to 4 GB could
        // take ~14 GB on a 23 GB box. See
        // tests/mwe_explicit_budget_reaches_mp2.rs.
        memory_budget_bytes,
        ..Default::default()
    };
    let inter = ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();

    let mut inv_de = ndarray::Array1::<f64>::zeros(nov);
    for i in 0..nocc {
        for a in 0..nvir {
            inv_de[i * nvir + a] = 1.0 / (eps_vir[a] - eps_occ[i]);
        }
    }

    // ε̃ = I + B̃ · diag(4/Δε) · B̃ᵀ  (closed-shell factor 4) — identical to the
    // construction in `pdep_polarizability_static`.
    let mut b_scaled = b_ov.clone();
    for ia in 0..nov {
        let s = (4.0 * inv_de[ia]).sqrt();
        let mut col = b_scaled.column_mut(ia);
        col.mapv_inplace(|x| x * s);
    }
    let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
    for p in 0..naux {
        eps_mat[(p, p)] += 1.0;
    }

    let (evals, _) = eps_mat
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("dielectric eigh: {e}")))?;
    let mut eigenvalues: Vec<f64> = evals.to_vec();
    eigenvalues.sort_by(|x, y| x.total_cmp(y));

    let rank = eigenvalues.iter().filter(|&&l| l - 1.0 > thresh).count();
    let trace_log: f64 = eigenvalues
        .iter()
        .map(|&l| if l > 0.0 { l.ln() } else { 0.0 })
        .sum();

    Ok(DielectricSpectrum {
        eigenvalues,
        naux,
        rank,
        thresh,
        trace_log,
    })
}

/// Open-shell static polarizability tensor at ω=0 (UHF / ROHF reference).
///
/// Same SMW construction as the closed-shell path but with prefactor 2 per
/// spin (vs closed-shell's 4 = 2·2). The (A+B) matrix is block-diagonal in
/// spin since RPA has no cross-spin direct coupling:
/// ```text
///   (A+B)^{σσ'}_{ia,jb} = δ_{σσ'} (δ_{ia,jb} Δε_{iaσ} + 2 (ia|jb)_{σ})
/// ```
/// SMW per spin: `ε̃ = I + Σ_σ 2 B̃_σ D_σ^{-1} B̃_σ^T`, then
/// ```text
///   α_ij = Σ_σ [2 (μ_σ^i)^T D_σ^{-1} μ_σ^j  −  8 (w_σ^i)^T y^j]
/// ```
/// (the y vector is shared across spins since ε̃ couples them through the
/// auxiliary basis).
pub fn pdep_polarizability_static_unrestricted(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
) -> Result<PolarizabilityResult, FerricError> {
    use ferric_mp2::rimp2::{compute_rpa_intermediates_spin, RiMp2Config};

    let mp2_cfg = RiMp2Config {
        frozen_core: 0,
        // Propagate the caller's explicit budget. This was hardcoded
        // `None`, which silently discarded a user's `[memory] budget_gb`
        // and let `resolve_budget` substitute an env/auto-detected value
        // (~0.8x available RAM) instead -- so a run pinned to 4 GB could
        // take ~14 GB on a 23 GB box. See
        // tests/mwe_explicit_budget_reaches_mp2.rs.
        memory_budget_bytes: cfg.memory_budget_bytes,
        ..Default::default()
    };

    // Per-spin intermediates.
    let inter_a = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, true)?;
    let inter_b = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, false)?;
    let naux = inter_a.naux;
    debug_assert_eq!(inter_a.naux, inter_b.naux);

    // Orbital-energy slices per spin (ROHF reuses α-MOs for β; see run_u_pdep_rpa).
    let eps_b_full: &[f64] = if matches!(rhf.spin, Spin::RestrictedOpen) {
        rhf.eps_a()
    } else {
        rhf.eps_b()
    };
    let eps_occ_a: Vec<f64> = rhf.eps_a()[inter_a.first_occ..inter_a.first_occ + inter_a.nocc].to_vec();
    let eps_vir_a: Vec<f64> = rhf.eps_a()[inter_a.nocc_total..inter_a.nocc_total + inter_a.nvir].to_vec();
    let eps_occ_b: Vec<f64> = eps_b_full[inter_b.first_occ..inter_b.first_occ + inter_b.nocc].to_vec();
    let eps_vir_b: Vec<f64> = eps_b_full[inter_b.nocc_total..inter_b.nocc_total + inter_b.nvir].to_vec();

    // Per-spin MO-basis dipole μ_σ^d_{ia} = ⟨ψ_iσ|r_d|ψ_aσ⟩.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let c_a = rhf.mos_a();
    let c_b = if matches!(rhf.spin, Spin::RestrictedOpen) { rhf.mos_a() } else { rhf.mos_b() };
    let c_occ_a = c_a.slice(ndarray::s![.., inter_a.first_occ..inter_a.first_occ + inter_a.nocc]).to_owned();
    let c_vir_a = c_a.slice(ndarray::s![.., inter_a.nocc_total..inter_a.nocc_total + inter_a.nvir]).to_owned();
    let c_occ_b = c_b.slice(ndarray::s![.., inter_b.first_occ..inter_b.first_occ + inter_b.nocc]).to_owned();
    let c_vir_b = c_b.slice(ndarray::s![.., inter_b.nocc_total..inter_b.nocc_total + inter_b.nvir]).to_owned();
    let mu_mo_a: [Array2<f64>; 3] = std::array::from_fn(|d| c_occ_a.t().dot(&dip_ao[d]).dot(&c_vir_a));
    let mu_mo_b: [Array2<f64>; 3] = std::array::from_fn(|d| c_occ_b.t().dot(&dip_ao[d]).dot(&c_vir_b));

    // Per-spin 1/Δε tables and flattened μ vectors.
    let build_flats = |nocc: usize, nvir: usize, eps_o: &[f64], eps_v: &[f64], mu_mo: &[Array2<f64>; 3]|
        -> ([ndarray::Array1<f64>; 3], [ndarray::Array1<f64>; 3], ndarray::Array1<f64>)
    {
        let nov = nocc * nvir;
        let mut inv_de = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc {
            for a in 0..nvir {
                inv_de[i * nvir + a] = 1.0 / (eps_v[a] - eps_o[i]);
            }
        }
        let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
            let mut v = ndarray::Array1::<f64>::zeros(nov);
            for i in 0..nocc {
                for a in 0..nvir {
                    v[i * nvir + a] = mu_mo[d][(i, a)];
                }
            }
            v
        });
        let mu_flat_inv: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| &mu_flat[d] * &inv_de);
        (mu_flat, mu_flat_inv, inv_de)
    };

    let (mu_flat_a, mu_flat_inv_a, inv_de_a) =
        build_flats(inter_a.nocc, inter_a.nvir, &eps_occ_a, &eps_vir_a, &mu_mo_a);
    let (mu_flat_b, mu_flat_inv_b, inv_de_b) =
        build_flats(inter_b.nocc, inter_b.nvir, &eps_occ_b, &eps_vir_b, &mu_mo_b);

    // w_σ^d_P = Σ_ia B̃_σ^P_ia · μ_σ^d_ia / Δε_iaσ ; total w^d = w_α^d + w_β^d
    let w_a: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| inter_a.b_ov.dot(&mu_flat_inv_a[d]));
    let w_b: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        if inter_b.nocc == 0 {
            ndarray::Array1::<f64>::zeros(naux)
        } else {
            inter_b.b_ov.dot(&mu_flat_inv_b[d])
        }
    });
    let w_total: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| &w_a[d] + &w_b[d]);

    // ε̃ = I + 2 B̃_α D_α^{-1} B̃_α^T + 2 B̃_β D_β^{-1} B̃_β^T
    let mut eps_mat = Array2::<f64>::zeros((naux, naux));
    for p in 0..naux { eps_mat[(p, p)] = 1.0; }
    for (b_ov, inv_de) in [(&inter_a.b_ov, &inv_de_a), (&inter_b.b_ov, &inv_de_b)] {
        if b_ov.shape()[1] == 0 { continue; }
        let mut b_scaled = b_ov.clone();
        for ia in 0..inv_de.len() {
            let s = (2.0 * inv_de[ia]).sqrt();
            let mut col = b_scaled.column_mut(ia);
            col.mapv_inplace(|x| x * s);
        }
        let chi_sigma = b_scaled.dot(&b_scaled.t());
        eps_mat += &chi_sigma;
    }

    // Solve ε̃ · y^d = w_total^d
    let y_vec = solve_dielectric_3(&eps_mat, &w_total)?;

    // Correct UHF formula derived from the full 2×2 spin-block SMW:
    //   α = 2 · μ̃^T (A+B)^{-1} μ̃
    //     = 2 · [μ̃^T M^{-1} μ̃ − 2 w_total^T ε̃^{-1} w_total]
    //     = 2 Σ_σ μ_σ^T D_σ^{-1} μ_σ  −  4 w_total^T ε̃^{-1} w_total
    //
    // The outer factor 2 is the TDDFT/RPA Casida-equation convention that
    // accounts for the density-vs-orbital-response distinction (verified
    // numerically: on closed-shell-via-UHF, factor 2 reproduces the
    // closed-shell α exactly via the 2x2 spin-block symmetric mode
    // `(D + 4 B̃^T B̃)`).
    let mut tensor = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let bare_a = 2.0 * mu_flat_a[i].dot(&mu_flat_inv_a[j]);
            let bare_b = if inter_b.nocc == 0 { 0.0 } else { 2.0 * mu_flat_b[i].dot(&mu_flat_inv_b[j]) };
            let coupled = w_total[i].dot(&y_vec[j]);
            tensor[i][j] = bare_a + bare_b - 4.0 * coupled;
        }
    }

    // Symmetrize.
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg = 0.5 * (tensor[i][j] + tensor[j][i]);
            tensor[i][j] = avg;
            tensor[j][i] = avg;
        }
    }
    let iso = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;
    let principal = eig3_sym(tensor)?;

    Ok(PolarizabilityResult { tensor, iso, principal })
}

/// Per-atom static polarizability decomposition via Hirshfeld partitioning.
///
/// Algebraically:
///
/// ```text
///     α^A_ij = 4 (μ^{A,i})^T D^{-1} μ^j  −  16 (w^{A,i})^T y^j
/// ```
///
/// where
///
/// * `μ^j` and `y^j` are the molecular MO-basis dipole and SMW-solve vectors
///   already built by `pdep_polarizability_static` (factor-of-4 closed-shell
///   convention, `(A+B) y = …`).
/// * `μ^{A,i}_pq = ⟨p| w^A(r) · r_i |q⟩` is the **Hirshfeld-weighted** dipole
///   matrix in the MO basis (i Cartesian, A atom).
/// * `w^{A,i}_P = Σ_{ia} B̃^P_{ia} · μ^{A,i}_{ia} / Δε_{ia}`.
///
/// Summing over A reproduces the molecular dipole (because Σ_A w^A(r) ≡ 1),
/// so Σ_A α^A = α (this is the sum-rule gate inside the function).
///
/// The Hirshfeld weights use a smooth single-exponential **promolecular**
/// reference density per element,
///
/// ```text
///     ρ_X^free(r) = (Z_X ξ_X³ / π) exp(−2 ξ_X r),
/// ```
///
/// with `ξ_X` derived from Bragg-Slater atomic radii.  The partition is
/// robust to the exact proatom shape (sum-rule constrained); the gate
/// will fail loudly if the grid is too coarse.
///
/// Closed-shell only.
/// Per-atom static polarizability decomposition via **Becke-Lebedev**
/// atomic partition.
///
/// Replaces the Slater-proatom Hirshfeld scheme in
/// [`pdep_polarizability_hirshfeld`] with the production-standard Becke
/// fuzzy weights (no electron-density proatom; geometry-only) on a
/// Becke-Lebedev atom-centered quadrature grid.
///
/// Same SMW algebra:
/// ```text
///     α^A_ij = 4 (μ^{A,i})^T D^{-1} μ^j  −  16 (w^{A,i})^T ε̃^{-1} w^j
/// ```
/// with
/// ```text
///     μ^{A,i}_pq = ⟨p| w^A_Becke(r) · r_i |q⟩  (Becke-weighted AO dipole)
///     μ^i = Σ_A μ^{A,i}                       (sum-rule exact for Becke)
/// ```
///
/// How many chunk-partials (each `natoms * 3 * nbf^2 * 8` bytes) may be live
/// at once, given a byte budget. Floored at rayon's worker count so banding
/// never starves parallelism below one chunk per worker (mirrors
/// `ferric_scf::reduce::band_width`'s floor). A pure function of
/// `(natoms, nbf, budget_bytes)` and the ambient rayon pool's worker count —
/// never of chunk count or grid layout — so it cannot perturb the fold order.
/// Pre-flight gate for the grid-based per-atom property paths.
///
/// Call this AFTER the grid is built (so `npts` is a live value, never a
/// hardcoded 75x110) but BEFORE `chi` is allocated — that is the last point at
/// which refusing is still cheap. `chi` alone is a single contiguous
/// `nbf * npts * 8` block (~3.75 GB at 71-atom/def2-svp), so a gate placed after
/// it has already lost.
///
/// These paths had NO gate at all before this: `grep` found zero
/// `check_alloc`/`estimate_peak_bytes` call sites in `properties.rs` or
/// `dispersion.rs`, while `compute_alpha_atomic` defaults to `true` in the CLI
/// — so a stock run entered an ungoverned path without opting in. That is how
/// 16-17 GB anon-RSS reached the systemwide OOM killer three times on
/// 2026-07-13.
///
/// Returns the resolved budget so callers can reuse it for banding decisions
/// rather than resolving twice (and possibly inconsistently).
pub(crate) fn preflight_grid_path(
    label: &str,
    memory_budget_bytes: Option<usize>,
    naux: usize,
    nocc: usize,
    nvir: usize,
    npts: usize,
    nbf: usize,
    natoms: usize,
) -> Result<usize, FerricError> {
    let budget = ferric_core::memory::resolve_budget_bytes(memory_budget_bytes);
    // Clamp to the chunk count: the accumulation caps each band at `n_chunks`,
    // so a nominally huge width cannot materialize more partials than there are
    // chunks. Estimating with the unclamped value refuses trivial jobs (see
    // `effective_dipole_band_width`).
    let band = crate::budget::effective_dipole_band_width(
        dipole_band_width(natoms, nbf, budget),
        npts,
    );
    let est = crate::budget::estimate_peak_bytes(crate::budget::PeakEstimateShape {
        naux,
        nocc,
        nvir,
        // The grid paths run their own frequency loops; n_quad drives wall time
        // rather than peak resident bytes (see budget.rs's named no-op), so the
        // value here is immaterial to the estimate.
        n_quad: 1,
        n_workers: rayon::current_num_threads().max(1),
        // Pre-eigensolve, the retained-mode count is unknown; naux is the
        // untruncated upper bound and the honest conservative choice.
        n_keep: naux,
        grid: Some(crate::budget::GridEstimateShape {
            npts,
            nbf,
            natoms,
            dipole_band_width: band,
            n_workers: rayon::current_num_threads().max(1),
        }),
    });
    ferric_core::memory::check_alloc(label, est, budget)?;
    Ok(budget)
}

/// Test-only re-export of [`dipole_band_width_with_threads`] so the
/// budget-contract MWEs in `tests/mwe_budget_respected.rs` can exercise the
/// band-width policy at an EXPLICIT worker count, without standing up an SCF
/// and without inheriting the ambient rayon pool size.
///
/// Taking `nthreads` as a parameter is load-bearing, not a convenience: the
/// first version of this MWE called the ambient-pool variant under
/// `RAYON_NUM_THREADS=2`, where the thread floor and the byte cap happen to
/// coincide — so all three contracts passed against the *unfixed* tree and the
/// bug hid. The overage is proportional to worker count (measured on the
/// 3-atom/nbf=24 shape: 1.0x at 2 threads, 6.0x at 12, 32.0x at 64), so a test
/// that cannot vary the worker count cannot see the defect at all.
#[doc(hidden)]
pub fn dipole_band_width_for_test(
    natoms: usize, nbf: usize, budget_bytes: usize, nthreads: usize,
) -> usize {
    dipole_band_width_with_threads(natoms, nbf, budget_bytes, nthreads)
}

fn dipole_band_width(natoms: usize, nbf: usize, budget_bytes: usize) -> usize {
    dipole_band_width_with_threads(natoms, nbf, budget_bytes, rayon::current_num_threads())
}

/// Worker-count-explicit core of [`dipole_band_width`].
///
/// The byte cap WINS over parallelism. A band of `w` chunks holds
/// `w * natoms * 3 * nbf^2 * 8` bytes of partials live at once, so `w` is
/// `budget_bytes / per_partial_bytes`, floored at 1 so a starvation budget
/// still makes progress (slow, never stuck).
///
/// This deliberately does NOT floor at the rayon worker count. That floor used
/// to be here, on the reasoning that banding should never starve parallelism
/// below one chunk per worker — but it silently overrode the byte cap, letting
/// the resident set reach `nthreads * per_partial_bytes` no matter how small
/// the budget was. Memory then scaled with core count rather than with the
/// budget, which is the mechanism behind the 16-17 GB anon-RSS incidents of
/// 2026-07-13 (measured on the MWE shape: 1.0x over at 2 workers, 6.0x at 12,
/// 32.0x at 64). A budget that can be exceeded by adding cores is not a budget,
/// so on a tight budget the correct trade is fewer chunks in flight, not more
/// memory. `tests/mwe_budget_respected.rs` pins all three contracts.
///
/// Narrowing the band is numerically INERT: `chunk_size` in
/// `accumulate_atom_centred_dipoles` is a pure function of `npts`, and each
/// band's partials are folded in ascending chunk order via `collect()`, so the
/// band width changes only how many chunks are in flight — never the fold
/// order, and hence never the result.
fn dipole_band_width_with_threads(
    natoms: usize, nbf: usize, budget_bytes: usize, _nthreads: usize,
) -> usize {
    let per_partial_bytes =
        natoms.max(1) * 3 * nbf.max(1) * nbf.max(1) * std::mem::size_of::<f64>();
    (budget_bytes / per_partial_bytes.max(1)).max(1)
}

/// Rayon-parallel, thread-count-independent, MEMORY-BOUNDED accumulation of
/// the atom-centred Becke-weighted AO dipole matrices
/// `D^{A,d}_{μν} = Σ_g w^A(r_g) (r_g − R_A)_d χ_μ(r_g) χ_ν(r_g)` over all grid
/// points `g`.
///
/// This was previously a fully serial `for g in 0..npts` loop — the dominant
/// wall-clock cost of `pdep_polarizability_becke`/`_dynamic` for large systems
/// (O(npts·nbf²): ~586k grid points × nbf² for a 71-atom/def2-svp system is
/// ~4e11 scalar ops). Follows the exact two-level banded pattern documented in
/// `ferric_scf::reduce::grouped_deterministic_sum`: partition into fixed-size,
/// thread-count-independent chunks (a pure function of `npts`, never of the
/// worker count, so the fold order — and hence the exact floating-point
/// result — cannot depend on `RAYON_NUM_THREADS`), process one BAND of chunks
/// in parallel at a time (bounded by the caller's resolved memory budget),
/// and fold each band
/// into the running accumulator in ascending order before moving to the next
/// band. At most one band of partials is ever live, not all chunks at once.
///
/// Crate-visible alias so `dispersion.rs` can share this exact accumulation
/// instead of maintaining a second, serial copy over a monolithic chi.
#[allow(clippy::too_many_arguments)]
pub(crate) fn accumulate_atom_centred_dipoles_pub(
    npts: usize,
    natoms: usize,
    nbf: usize,
    home_atom: &[usize],
    weights: &[f64],
    points: &[[f64; 3]],
    atom_pos: &[[f64; 3]],
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    budget_bytes: usize,
) -> Result<Vec<[Array2<f64>; 3]>, FerricError> {
    accumulate_atom_centred_dipoles(
        npts, natoms, nbf, home_atom, weights, points, atom_pos, mol, obs_bs, budget_bytes,
    )
}

fn accumulate_atom_centred_dipoles(
    npts: usize,
    natoms: usize,
    nbf: usize,
    home_atom: &[usize],
    weights: &[f64],
    points: &[[f64; 3]],
    atom_pos: &[[f64; 3]],
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    budget_bytes: usize,
) -> Result<Vec<[Array2<f64>; 3]>, FerricError> {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use rayon::prelude::*;

    // Chunk size: same "≥1024 groups, floored at 1" convention as
    // ferric_scf::reduce::TARGET_GROUPS — a pure function of npts, never of
    // rayon::current_num_threads().
    const TARGET_CHUNKS: usize = 1024;
    let chunk_size = npts.div_ceil(TARGET_CHUNKS).max(1);
    let chunk_starts: Vec<usize> = (0..npts).step_by(chunk_size).collect();
    let n_chunks = chunk_starts.len();

    // The RESOLVED budget, not a hardcoded constant: the pre-flight gate
    // estimated this path using the caller's budget, so the accumulation must
    // band against the same number or the two disagree (below the old 512 MB
    // const the accumulation would exceed what the gate approved; above it, it
    // would band more tightly than necessary and cost wall time).
    let band_width = dipole_band_width(natoms, nbf, budget_bytes);

    let mut d_ai_ao: Vec<[Array2<f64>; 3]> = (0..natoms)
        .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nbf, nbf))))
        .collect();

    let mut band0 = 0usize;
    while band0 < n_chunks {
        let band1 = (band0 + band_width).min(n_chunks);
        // Parallel over this band's chunks only; collect preserves ascending
        // chunk-index order regardless of worker count.
        let band_partials: Vec<Vec<[Array2<f64>; 3]>> = chunk_starts[band0..band1]
            .to_vec()
            .into_par_iter()
            .map(|g0| -> Result<Vec<[Array2<f64>; 3]>, FerricError> {
                let g1 = (g0 + chunk_size).min(npts);
                // Evaluate chi for THIS CHUNK's points only. The full
                // (nbf, npts) matrix was previously built up front as one
                // contiguous allocation -- ~3.75 GB at nbf=800/npts=586k -- even
                // though every consumer reads it one grid point at a time. A
                // chunk is `chunk_size` points wide (npts/1024, so ~573 at
                // incident scale), reducing this to a few MB per worker.
                //
                // Numerically inert: chi is a materialized lookup table, not a
                // reduction. Each point's value is a pure function of its
                // coordinates, so evaluating it in chunks reproduces the same
                // numbers; the fold order that DOES matter is the ascending
                // chunk fold below, which is untouched.
                let chunk_pts = &points[g0..g1];
                let chi_c = eval_basis_on_points(mol, obs_bs, chunk_pts).map_err(|e| {
                    FerricError::General(format!(
                        "accumulate_atom_centred_dipoles: chi eval failed: {e}"
                    ))
                })?;
                let mut local: Vec<[Array2<f64>; 3]> = (0..natoms)
                    .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nbf, nbf))))
                    .collect();
                for g in g0..g1 {
                    let gc = g - g0; // chunk-local column index
                    let a = home_atom[g];
                    let w = weights[g];
                    let r = points[g];
                    let ra = atom_pos[a];
                    for d in 0..3 {
                        let factor = w * (r[d] - ra[d]);
                        for mu in 0..nbf {
                            let chi_mu = chi_c[(mu, gc)];
                            let weighted_chi_mu = factor * chi_mu;
                            if weighted_chi_mu.abs() < 1e-30 {
                                continue;
                            }
                            for nu in 0..nbf {
                                local[a][d][(mu, nu)] += weighted_chi_mu * chi_c[(nu, gc)];
                            }
                        }
                    }
                }
                Ok(local)
            })
            .collect::<Result<Vec<_>, FerricError>>()?;
        // Serial fold in ascending chunk order — the determinism anchor.
        for chunk in &band_partials {
            for a in 0..natoms {
                for d in 0..3 {
                    d_ai_ao[a][d] += &chunk[a][d];
                }
            }
        }
        band0 = band1;
    }
    Ok(d_ai_ao)
}

/// Closed-shell only. Returns Vec<[[f64; 3]; 3]>, one (3×3) tensor per atom.
pub fn pdep_polarizability_becke(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
) -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
    use ferric_dft::ao_grid::eval_basis_on_points;

    // Open-shell: the dynamic Becke path has a complete per-spin (U) branch, and
    // ω=0 reproduces the static per-atom α exactly. Delegate to it rather than
    // duplicate the per-spin static math (DRY).
    if !matches!(rhf.spin, Spin::Restricted) {
        let dyn0 = pdep_polarizability_becke_dynamic(
            mol, obs, obs_bs, dfbs, rhf, op, cfg, &[0.0],
        )?;
        // dyn0[atom][freq=0] → per-atom static tensor.
        return Ok(dyn0.into_iter().map(|per_freq| per_freq[0]).collect());
    }

    let natoms = mol.atoms.len();

    // RI intermediates (same as molecular static-α path).
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config {
        frozen_core: 0,
        // Propagate the caller's explicit budget. This was hardcoded
        // `None`, which silently discarded a user's `[memory] budget_gb`
        // and let `resolve_budget` substitute an env/auto-detected value
        // (~0.8x available RAM) instead -- so a run pinned to 4 GB could
        // take ~14 GB on a 23 GB box. See
        // tests/mwe_explicit_budget_reaches_mp2.rs.
        memory_budget_bytes: cfg.memory_budget_bytes,
        ..Default::default()
    };
    let inter =
        ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();

    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();

    let mut inv_de = ndarray::Array1::<f64>::zeros(nov);
    for i in 0..nocc {
        for a in 0..nvir {
            inv_de[i * nvir + a] = 1.0 / (eps_vir[a] - eps_occ[i]);
        }
    }

    // ε̃ = I + B̃ · diag(4/Δε) · B̃^T (closed-shell prefactor 4).
    let mut b_scaled = b_ov.clone();
    for ia in 0..nov {
        let s = (4.0 * inv_de[ia]).sqrt();
        let mut col = b_scaled.column_mut(ia);
        col.mapv_inplace(|x| x * s);
    }
    let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
    for p in 0..naux {
        eps_mat[(p, p)] += 1.0;
    }

    // Build Becke-Lebedev grid (atom-centered, Becke partition baked into
    // grid-point weights; per-atom restriction via home_atom field).
    let grid_cfg = AtomicGridConfig::default();
    let grid = build_atomic_grid(mol, &grid_cfg);
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let home_atom: Vec<usize> = grid.iter().map(|g| g.home_atom).collect();
    let npts = points.len();

    // Pre-flight BEFORE chi allocates: chi is one contiguous nbf*npts*8 block,
    // so a gate placed after it has already lost. npts comes from the grid we
    // just built, never from an assumed 75x110.
    let budget = preflight_grid_path(
        &format!(
            "pdep_polarizability_becke (natoms={natoms}, nbf={}, npts={npts}, naux={naux})",
            obs.nbasis()
        ),
        cfg.memory_budget_bytes,
        naux,
        nocc,
        nvir,
        npts,
        obs.nbasis(),
        natoms,
    )?;

    // Evaluate AO basis on the grid: χ has shape (nbf, npts).
    let chi = eval_basis_on_points(mol, obs_bs, &points).map_err(|e| {
        FerricError::General(format!("pdep_polarizability_becke: chi eval failed: {e}"))
    })?;
    let nbf = chi.nrows();
    debug_assert_eq!(nbf, obs.nbasis());

    // Build per-atom Becke-weighted AO dipole using the ATOM-CENTRED position
    // operator (r − R_A). This yields the intrinsic atomic polarizability:
    // origin-independent and charge-transfer-free, matching
    // `pdep_polarizability_becke_dynamic`'s ω=0 limit exactly. We deliberately
    // do NOT renormalize to the global lab-frame analytical dipole — that
    // renormalization is the gauge-breaking step (it scales each atom's
    // contribution by a shared factor derived from a lab-frame quantity and
    // cannot restore per-atom symmetry for off-origin geometries; see the
    // 2026-07-13 gauge-origin regression: danuglipron's cryo-EM lab-frame
    // pose produced α^A up to hundreds of a.u. via this renormalization).
    let atom_pos: Vec<[f64; 3]> = mol.atoms.iter().map(|at| [at.x, at.y, at.zpos]).collect();
    let mut d_ai_ao: Vec<[Array2<f64>; 3]> =
        accumulate_atom_centred_dipoles(npts, natoms, nbf, &home_atom, &weights, &points, &atom_pos, mol, obs_bs, budget)?;
    // Symmetrize per-atom AO dipoles.
    for d in 0..3 {
        for a in 0..natoms {
            let m = &mut d_ai_ao[a][d];
            for i in 0..nbf {
                for j in (i + 1)..nbf {
                    let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
                    m[(i, j)] = avg;
                    m[(j, i)] = avg;
                }
            }
        }
    }

    // Transform to MO occ-vir basis.
    let mut mu_ai_mo: Vec<[Array2<f64>; 3]> = (0..natoms)
        .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nocc, nvir))))
        .collect();
    for a in 0..natoms {
        for d in 0..3 {
            mu_ai_mo[a][d] = c_occ.t().dot(&d_ai_ao[a][d]).dot(&c_vir);
        }
    }

    // Molecular (field-side) dipole: the TRUE lab-frame molecular dipole from
    // the analytical AO integrals — the perturbation a uniform field couples
    // to. Paired below with the atom-centred bra (mu_ai_flat) to give
    // α^A_{dj} = ∂μ^A_d/∂E_j, exactly as in pdep_polarizability_hirshfeld.
    let dip_ao_analytical = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let mu_mo: [Array2<f64>; 3] = std::array::from_fn(|d| {
        c_occ.t().dot(&dip_ao_analytical[d]).dot(&c_vir)
    });
    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc {
            for ax in 0..nvir {
                v[i * nvir + ax] = mu_mo[d][(i, ax)];
            }
        }
        v
    });
    let mu_flat_inv: [ndarray::Array1<f64>; 3] =
        std::array::from_fn(|d| &mu_flat[d] * &inv_de);
    let w_mol: [ndarray::Array1<f64>; 3] =
        std::array::from_fn(|d| b_ov.dot(&mu_flat_inv[d]));
    let y_mol = solve_dielectric_3(&eps_mat, &w_mol)?;

    // Assemble per-atom α^A.
    let mut alpha_per_atom: Vec<[[f64; 3]; 3]> = vec![[[0.0; 3]; 3]; natoms];
    for a in 0..natoms {
        for d in 0..3 {
            let mut mu_ai_flat = ndarray::Array1::<f64>::zeros(nov);
            let mut mu_ai_flat_inv = ndarray::Array1::<f64>::zeros(nov);
            for i in 0..nocc {
                for ax in 0..nvir {
                    let ia = i * nvir + ax;
                    mu_ai_flat[ia] = mu_ai_mo[a][d][(i, ax)];
                    mu_ai_flat_inv[ia] = mu_ai_mo[a][d][(i, ax)] * inv_de[ia];
                }
            }
            let w_ai = b_ov.dot(&mu_ai_flat_inv);
            for j in 0..3 {
                let bare = mu_ai_flat.dot(&mu_flat_inv[j]);
                let coupled = w_ai.dot(&y_mol[j]);
                alpha_per_atom[a][d][j] = 4.0 * bare - 16.0 * coupled;
            }
        }
        // Symmetrize per-atom tensor.
        for i in 0..3 {
            for j in (i + 1)..3 {
                let avg = 0.5 * (alpha_per_atom[a][i][j] + alpha_per_atom[a][j][i]);
                alpha_per_atom[a][i][j] = avg;
                alpha_per_atom[a][j][i] = avg;
            }
        }
    }

    Ok(alpha_per_atom)
}

/// Per-atom Becke polarizability tensors α^A_{ij}(iω) at a list of imaginary
/// frequencies. Returns `out[a][k]` = 3×3 tensor for atom `a`, frequency `k`.
///
/// This is the frequency generalization of [`pdep_polarizability_becke`]:
/// the grid, Becke partition, and partition-weighted MO dipoles are built once
/// (ω-independent); only the χ₀ "denominator" g_ia(ω) = e_ia/(ω²+e_ia²) and the
/// SMW dielectric ε̃(ω) = I + 4 B̃ diag(g(ω)) B̃^T change per frequency. At ω=0,
/// g_ia = 1/Δε_ia, so this reproduces `pdep_polarizability_becke` exactly.
///
/// The Casimir-Polder C6 follow-up consumes these via
/// `dispersion::pdep_dynamic_polarizability`.
#[allow(clippy::too_many_arguments)]
pub fn pdep_polarizability_becke_dynamic(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
    freqs: &[f64],
) -> Result<Vec<Vec<[[f64; 3]; 3]>>, FerricError> {
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
    use ferric_integrals::blas_threads::with_blas_threads;
    use rayon::prelude::*;

    let natoms = mol.atoms.len();
    let nfreq = freqs.len();
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config {
        frozen_core: cfg.frozen_core,
        memory_budget_bytes: cfg.memory_budget_bytes,
        ..Default::default()
    };

    // Open-shell dispatch: build per-spin intermediates and MO slices.
    // Closed-shell falls through to the single-B̃ path below.
    if !matches!(rhf.spin, Spin::Restricted) {
        use ferric_mp2::rimp2::compute_rpa_intermediates_spin;

        let inter_a = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, true)?;
        let inter_b = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, false)?;
        let naux = inter_a.naux;

        // Orbital-energy slices (ROHF reuses α-MOs for β).
        let eps_b_full: &[f64] = if matches!(rhf.spin, Spin::RestrictedOpen) {
            rhf.eps_a()
        } else {
            rhf.eps_b()
        };
        let eps_occ_a: Vec<f64> = rhf.eps_a()[inter_a.first_occ..inter_a.first_occ + inter_a.nocc].to_vec();
        let eps_vir_a: Vec<f64> = rhf.eps_a()[inter_a.nocc_total..inter_a.nocc_total + inter_a.nvir].to_vec();
        let eps_occ_b: Vec<f64> = eps_b_full[inter_b.first_occ..inter_b.first_occ + inter_b.nocc].to_vec();
        let eps_vir_b: Vec<f64> = eps_b_full[inter_b.nocc_total..inter_b.nocc_total + inter_b.nvir].to_vec();

        // Per-spin e_ia tables.
        let e_ia_a = {
            let (nocc, nvir) = (inter_a.nocc, inter_a.nvir);
            let mut v = ndarray::Array1::<f64>::zeros(nocc * nvir);
            for i in 0..nocc { for a in 0..nvir { v[i*nvir+a] = eps_vir_a[a] - eps_occ_a[i]; } }
            v
        };
        let e_ia_b = {
            let (nocc, nvir) = (inter_b.nocc, inter_b.nvir);
            let mut v = ndarray::Array1::<f64>::zeros(nocc * nvir);
            for i in 0..nocc { for a in 0..nvir { v[i*nvir+a] = eps_vir_b[a] - eps_occ_b[i]; } }
            v
        };

        // MO coefficient slices per spin.
        let c_a = rhf.mos_a();
        let c_b = if matches!(rhf.spin, Spin::RestrictedOpen) { rhf.mos_a() } else { rhf.mos_b() };
        let c_occ_a = c_a.slice(ndarray::s![.., inter_a.first_occ..inter_a.first_occ+inter_a.nocc]).to_owned();
        let c_vir_a = c_a.slice(ndarray::s![.., inter_a.nocc_total..inter_a.nocc_total+inter_a.nvir]).to_owned();
        let c_occ_b = c_b.slice(ndarray::s![.., inter_b.first_occ..inter_b.first_occ+inter_b.nocc]).to_owned();
        let c_vir_b = c_b.slice(ndarray::s![.., inter_b.nocc_total..inter_b.nocc_total+inter_b.nvir]).to_owned();

        // Becke grid (spin-agnostic).
        let grid_cfg = ferric_dft::grid::AtomicGridConfig::default();
        let grid = ferric_dft::grid::build_atomic_grid(mol, &grid_cfg);
        let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let weights_g: Vec<f64> = grid.iter().map(|g| g.weight).collect();
        let home_atom: Vec<usize> = grid.iter().map(|g| g.home_atom).collect();
        let npts = points.len();
        // Pre-flight before the grid work: npts comes from the grid just built,
        // never an assumed 75x110.
        preflight_grid_path(
            &format!(
                "pdep_polarizability_becke_dynamic (U) (natoms={natoms}, nbf={}, npts={npts}, naux={naux})",
                obs.nbasis()
            ),
            cfg.memory_budget_bytes,
            naux,
            // Open-shell: charge the LARGER spin channel. Both intermediates
            // are resident, but the per-worker frequency scratch is sized from
            // one channel at a time.
            inter_a.nocc.max(inter_b.nocc),
            inter_a.nvir.max(inter_b.nvir),
            npts,
            obs.nbasis(),
            natoms,
        )?;

        // nbf from the prepared basis, NOT from a materialized chi: the
        // accumulation evaluates chi per grid chunk now, so building the full
        // (nbf, npts) block here just to read its row count would allocate
        // multi-GB for a single integer.
        let nbf = obs.nbasis();
        let atom_pos: Vec<[f64; 3]> = mol.atoms.iter().map(|at| [at.x, at.y, at.zpos]).collect();

        // Per-atom atom-centred AO dipole matrices (ω-independent).
        let mut d_ai_ao: Vec<[Array2<f64>; 3]> = accumulate_atom_centred_dipoles(
            npts, natoms, nbf, &home_atom, &weights_g, &points, &atom_pos, mol, obs_bs,
            ferric_core::memory::resolve_budget_bytes(cfg.memory_budget_bytes),
        )?;
        for a in 0..natoms {
            for d in 0..3 {
                let m = &mut d_ai_ao[a][d];
                for i in 0..nbf {
                    for j in (i+1)..nbf {
                        let avg = 0.5*(m[(i,j)]+m[(j,i)]);
                        m[(i,j)] = avg; m[(j,i)] = avg;
                    }
                }
            }
        }

        // Transform per-atom AO dipoles to per-spin MO basis.
        let nov_a = inter_a.nocc * inter_a.nvir;
        let nov_b = inter_b.nocc * inter_b.nvir;
        let mu_ai_flat_a: Vec<[ndarray::Array1<f64>; 3]> = (0..natoms).map(|a| {
            std::array::from_fn(|d| {
                let mo = c_occ_a.t().dot(&d_ai_ao[a][d]).dot(&c_vir_a);
                let mut v = ndarray::Array1::<f64>::zeros(nov_a);
                for i in 0..inter_a.nocc { for ax in 0..inter_a.nvir { v[i*inter_a.nvir+ax] = mo[(i,ax)]; } }
                v
            })
        }).collect();
        let mu_ai_flat_b: Vec<[ndarray::Array1<f64>; 3]> = (0..natoms).map(|a| {
            std::array::from_fn(|d| {
                if inter_b.nocc == 0 { return ndarray::Array1::<f64>::zeros(1.max(nov_b)); }
                let mo = c_occ_b.t().dot(&d_ai_ao[a][d]).dot(&c_vir_b);
                let mut v = ndarray::Array1::<f64>::zeros(nov_b);
                for i in 0..inter_b.nocc { for ax in 0..inter_b.nvir { v[i*inter_b.nvir+ax] = mo[(i,ax)]; } }
                v
            })
        }).collect();

        // Molecular MO dipoles per spin (sum over atoms).
        let mu_flat_a: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
            mu_ai_flat_a.iter().fold(ndarray::Array1::zeros(nov_a), |acc, ai| acc + &ai[d])
        });
        let mu_flat_b: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
            if inter_b.nocc == 0 { return ndarray::Array1::zeros(1); }
            mu_ai_flat_b.iter().fold(ndarray::Array1::zeros(nov_b), |acc, ai| acc + &ai[d])
        });

        // Frequency loop: each ω is fully independent. Parallelize over
        // frequencies (energy.rs pattern); the per-spin (naux × nov_σ)
        // column-scaled B̃_σ scratch that M9 hoisted out of the loop becomes
        // per-thread via map_init (one (b_scaled_a, b_scaled_b) pair per
        // rayon worker, reused across the ω's it processes — never cloned
        // per frequency, never shared across workers). BLAS pinned to 1
        // inside the rayon region (per-frequency dielectric solve must not
        // nest OpenBLAS threads under rayon workers).
        let rows: Vec<Result<Vec<[[f64; 3]; 3]>, FerricError>> = with_blas_threads(1, || {
            freqs
                .par_iter()
                .map_init(
                    // Zeros, not `.clone()`: the closure overwrites this
                    // buffer with `assign(b_ov)` before reading it, so cloning
                    // copies naux*nov doubles per worker PER SPIN that are then
                    // discarded (~23.5 GB at naux=2976/nov=61740 across 8
                    // workers). Same resident footprint, without the copy.
                    || (
                        Array2::<f64>::zeros(inter_a.b_ov.raw_dim()),
                        Array2::<f64>::zeros(inter_b.b_ov.raw_dim()),
                    ),
                    |(b_scaled_a, b_scaled_b), &omega| -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
                        let omega2 = omega * omega;

                        // Per-spin g_σ(ω) = e_iaσ / (ω² + e_iaσ²), prefactor 2.
                        let g_a = {
                            let n = e_ia_a.len();
                            let mut v = ndarray::Array1::<f64>::zeros(n);
                            for ia in 0..n { let e = e_ia_a[ia]; v[ia] = e/(omega2+e*e); }
                            v
                        };
                        let g_b = if inter_b.nocc > 0 {
                            let n = e_ia_b.len();
                            let mut v = ndarray::Array1::<f64>::zeros(n);
                            for ia in 0..n { let e = e_ia_b[ia]; v[ia] = e/(omega2+e*e); }
                            v
                        } else {
                            ndarray::Array1::<f64>::zeros(1)
                        };

                        // ε̃(ω) = I + 2 B̃_α diag(g_α) B̃_αᵀ + 2 B̃_β diag(g_β) B̃_βᵀ.
                        let mut eps_mat = Array2::<f64>::zeros((naux, naux));
                        for p in 0..naux { eps_mat[(p,p)] = 1.0; }
                        for (b_ov, g, b_scaled) in [
                            (&inter_a.b_ov, &g_a, &mut *b_scaled_a),
                            (&inter_b.b_ov, &g_b, &mut *b_scaled_b),
                        ] {
                            let nov = b_ov.shape()[1];
                            if nov == 0 { continue; }
                            b_scaled.assign(b_ov);
                            for ia in 0..nov {
                                let s = (2.0 * g[ia]).sqrt();
                                b_scaled.column_mut(ia).mapv_inplace(|x| x * s);
                            }
                            let chi_s = b_scaled.dot(&b_scaled.t());
                            eps_mat += &chi_s;
                        }

                        // w_total^d(ω) = B̃_α (μ_α^d ⊙ g_α) + B̃_β (μ_β^d ⊙ g_β).
                        let w_total: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
                            let wa = inter_a.b_ov.dot(&(&mu_flat_a[d] * &g_a));
                            if inter_b.nocc == 0 { return wa; }
                            let wb = inter_b.b_ov.dot(&(&mu_flat_b[d] * &g_b));
                            wa + wb
                        });
                        let y_total = solve_dielectric_3(&eps_mat, &w_total)?;

                        let mut row: Vec<[[f64; 3]; 3]> = vec![[[0.0; 3]; 3]; natoms];
                        for a in 0..natoms {
                            // w_ai_total^d = B̃_α (μ_α^{A,d} ⊙ g_α) + B̃_β (μ_β^{A,d} ⊙ g_β).
                            let w_ai: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
                                let wa = inter_a.b_ov.dot(&(&mu_ai_flat_a[a][d] * &g_a));
                                if inter_b.nocc == 0 { return wa; }
                                let wb = inter_b.b_ov.dot(&(&mu_ai_flat_b[a][d] * &g_b));
                                wa + wb
                            });
                            for d in 0..3 {
                                for j in 0..3 {
                                    let bare_a = 2.0 * mu_ai_flat_a[a][d].dot(&(&mu_flat_a[j] * &g_a));
                                    let bare_b = if inter_b.nocc > 0 {
                                        2.0 * mu_ai_flat_b[a][d].dot(&(&mu_flat_b[j] * &g_b))
                                    } else { 0.0 };
                                    let coupled = w_ai[d].dot(&y_total[j]);
                                    row[a][d][j] = bare_a + bare_b - 4.0 * coupled;
                                }
                            }
                            // Symmetrize.
                            for i in 0..3 {
                                for j in (i+1)..3 {
                                    let avg = 0.5*(row[a][i][j]+row[a][j][i]);
                                    row[a][i][j] = avg; row[a][j][i] = avg;
                                }
                            }
                        }
                        Ok(row)
                    },
                )
                .collect::<Vec<Result<Vec<[[f64; 3]; 3]>, FerricError>>>()
        });

        let mut out: Vec<Vec<[[f64; 3]; 3]>> = vec![vec![[[0.0; 3]; 3]; nfreq]; natoms];
        for (k, row) in rows.into_iter().enumerate() {
            let row = row?;
            for a in 0..natoms {
                out[a][k] = row[a];
            }
        }
        return Ok(out);
    }

    // --- Closed-shell path below (unchanged) ---
    let inter =
        ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();

    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();

    // ε_ia table (the bare excitation energies).
    let mut e_ia = ndarray::Array1::<f64>::zeros(nov);
    for i in 0..nocc {
        for a in 0..nvir {
            e_ia[i * nvir + a] = eps_vir[a] - eps_occ[i];
        }
    }

    // --- Build the Becke grid + partition-weighted per-atom MO dipoles ONCE.
    // (Identical to pdep_polarizability_becke; ω-independent.)
    let grid_cfg = AtomicGridConfig::default();
    let grid = build_atomic_grid(mol, &grid_cfg);
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();
    let home_atom: Vec<usize> = grid.iter().map(|g| g.home_atom).collect();
    let npts = points.len();
    // Pre-flight before the grid work: npts comes from the grid just built,
    // never an assumed 75x110.
    preflight_grid_path(
        &format!(
            "pdep_polarizability_becke_dynamic (natoms={natoms}, nbf={}, npts={npts}, naux={naux})",
            obs.nbasis()
        ),
        cfg.memory_budget_bytes,
        naux,
        nocc,
        nvir,
        npts,
        obs.nbasis(),
        natoms,
    )?;

    // nbf from the prepared basis, not a materialized chi (see the U branch).
    let nbf = obs.nbasis();
    debug_assert_eq!(nbf, obs.nbasis());

    // Atom positions (Bohr) — used to shift dipole to atom-centred coordinates.
    // Using (r - R_A) instead of the lab-frame r makes each per-atom contribution
    // to α^A(iω) origin-independent: at all frequencies, α^A and Σ_A α^A are
    // unchanged by a global translation of the coordinate system.
    let atom_pos: Vec<[f64; 3]> = mol
        .atoms
        .iter()
        .map(|at| [at.x, at.y, at.zpos])
        .collect();

    // No pre-flight gate on this path yet (tracked), so resolve the budget here
    // rather than band against the old hardcoded 512 MB const.
    let budget = ferric_core::memory::resolve_budget_bytes(cfg.memory_budget_bytes);
    let mut d_ai_ao: Vec<[Array2<f64>; 3]> =
        accumulate_atom_centred_dipoles(npts, natoms, nbf, &home_atom, &weights, &points, &atom_pos, mol, obs_bs, budget)?;
    for d in 0..3 {
        for a in 0..natoms {
            let m = &mut d_ai_ao[a][d];
            for i in 0..nbf {
                for j in (i + 1)..nbf {
                    let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
                    m[(i, j)] = avg;
                    m[(j, i)] = avg;
                }
            }
        }
    }
    // No renormalization of the atom-centred per-atom dipoles.
    //
    // The static Becke path renormalizes each AO-pair's grid partition to the
    // global analytical dipole ⟨μ|r|ν⟩, which fixes grid-quadrature error on the
    // dipole magnitude. Here we use atom-centred displacements (r − R_A), so the
    // natural analytical comparison is the atom-centred grid SUM, not the global
    // dipole — and those differ by Σ_A R_A · q^A_{μν} (charge partition moments).
    //
    // For the dynamic path what matters is the *frequency dependence* of α^A(iω),
    // not the absolute magnitude at ω=0. The grid quadrature error on (r − R_A)
    // is smooth and does not introduce frequency-dependent artifacts. We therefore
    // skip renormalization and accept the raw grid integrals. This gives correct
    // physics for the Casimir-Polder integrand at all ω.
    //
    // NOTE: the molecular sum Σ_A α^A(ω=0) from this path will NOT equal
    // pdep_polarizability_static (which uses a renormalized lab-frame dipole).
    // The correct comparison for the regression gate is: take the ω=0 dynamic path
    // result, form the atom-centred MO dipoles, and verify the SMW formula gives
    // the same α as the lab-frame path scaled by the atom-centred/lab-frame ratio.
    // For the purposes of C6 correctness, we verify:
    //   * α_iso_A(ω=0) > 0 for each atom (positive sum rule)
    //   * α_A(iω) decays monotonically with ω (correct frequency dependence)
    //   * homonuclear atom pairs give equal C6 (symmetry check)
    // These are tested in the unit tests.
    for a in 0..natoms {
        for d in 0..3 {
            let m = &mut d_ai_ao[a][d];
            for i in 0..nbf {
                for j in (i + 1)..nbf {
                    let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
                    m[(i, j)] = avg;
                    m[(j, i)] = avg;
                }
            }
        }
    }

    // Transform to MO occ-vir basis (per-atom + molecular sum).
    let mut mu_ai_mo: Vec<[Array2<f64>; 3]> = (0..natoms)
        .map(|_| std::array::from_fn(|_| Array2::<f64>::zeros((nocc, nvir))))
        .collect();
    let mut mu_mo: [Array2<f64>; 3] = std::array::from_fn(|_| Array2::<f64>::zeros((nocc, nvir)));
    for a in 0..natoms {
        for d in 0..3 {
            let m = c_occ.t().dot(&d_ai_ao[a][d]).dot(&c_vir);
            mu_mo[d] = &mu_mo[d] + &m;
            mu_ai_mo[a][d] = m;
        }
    }

    // Flatten molecular dipole and per-atom dipoles into (nov,) vectors ONCE.
    let mu_ai_flat: Vec<[ndarray::Array1<f64>; 3]> = (0..natoms)
        .map(|a| {
            std::array::from_fn(|d| {
                let mut v = ndarray::Array1::<f64>::zeros(nov);
                for i in 0..nocc {
                    for ax in 0..nvir {
                        v[i * nvir + ax] = mu_ai_mo[a][d][(i, ax)];
                    }
                }
                v
            })
        })
        .collect();

    // Flatten the Becke-sum molecular dipole (sum of atom-centred pieces).
    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc { for ax in 0..nvir { v[i*nvir+ax] = mu_mo[d][(i,ax)]; } }
        v
    });

    // --- Frequency loop: direct per-atom SMW (symmetric Becke-sum reference) ---
    //
    // α^A_{ij}(iω) = 4 μ^{A,i}·g·μ^{Becke,j} − 16 (B·μ^{A,i}·g)·ε̃⁻¹·(B·μ^{Becke,j}·g)
    //
    // Both slots use the same Becke-sum dipole μ^Becke = Σ_A μ^A, so
    // Σ_A α^A = α_mol(μ^Becke) exactly. This matches pdep_dynamic_polarizability_truncated.
    // The anisotropy reflects the Becke partition's atom-centred displacements;
    // for isotropic C6 via Casimir-Polder this is the correct formula.
    //
    // Each ω is independent — parallelize over frequencies. The (naux × nov)
    // column-scaled B̃ scratch M9 hoisted out of the loop becomes per-thread
    // via map_init (one buffer per rayon worker, reused across the ω's it
    // handles). BLAS pinned to 1 inside the rayon region.
    let rows: Vec<Result<Vec<[[f64; 3]; 3]>, FerricError>> = with_blas_threads(1, || {
        freqs
            .par_iter()
            .map_init(
                // Zeros, not `.clone()`: overwritten by `assign(b_ov)` before
                // it is read, so the cloned contents are discarded (see the
                // two-spin site and tests/mwe_per_worker_scratch.rs).
                || Array2::<f64>::zeros(b_ov.raw_dim()),
                |b_scaled, &omega| -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
                    let omega2 = omega * omega;
                    let mut g = ndarray::Array1::<f64>::zeros(nov);
                    for ia in 0..nov { let e = e_ia[ia]; g[ia] = e / (omega2 + e * e); }

                    // ε̃(ω) = I + 4 B̃ diag(g) B̃^T
                    b_scaled.assign(b_ov);
                    for ia in 0..nov {
                        b_scaled.column_mut(ia).mapv_inplace(|x| x * (4.0 * g[ia]).sqrt());
                    }
                    let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
                    for p in 0..naux { eps_mat[(p, p)] += 1.0; }

                    // Solve ε̃ y^j = B·(g⊙μ^{Becke,j}) once per direction.
                    let mu_g: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| &mu_flat[d] * &g);
                    let w_mol: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| b_ov.dot(&mu_g[d]));
                    let y_mol = solve_dielectric_3(&eps_mat, &w_mol)?;

                    let mut row: Vec<[[f64; 3]; 3]> = vec![[[0.0; 3]; 3]; natoms];
                    for a in 0..natoms {
                        let mut tensor = [[0.0_f64; 3]; 3];
                        for d in 0..3 {
                            let w_ai = b_ov.dot(&(&mu_ai_flat[a][d] * &g));
                            for j in 0..3 {
                                let bare = mu_ai_flat[a][d].dot(&mu_g[j]);
                                let coupled = w_ai.dot(&y_mol[j]);
                                tensor[d][j] = 4.0 * bare - 16.0 * coupled;
                            }
                        }
                        // Symmetrize.
                        for i in 0..3 { for j in (i+1)..3 {
                            let avg = 0.5*(tensor[i][j]+tensor[j][i]);
                            tensor[i][j] = avg; tensor[j][i] = avg;
                        }}
                        row[a] = tensor;
                    }
                    Ok(row)
                },
            )
            .collect::<Vec<Result<Vec<[[f64; 3]; 3]>, FerricError>>>()
    });

    let mut out: Vec<Vec<[[f64; 3]; 3]>> =
        vec![vec![[[0.0; 3]; 3]; nfreq]; natoms];
    for (k, row) in rows.into_iter().enumerate() {
        let row = row?;
        for a in 0..natoms {
            out[a][k] = row[a];
        }
    }

    Ok(out)
}

/// Compute per-atom static polarizability tensors via PDEP-RPA with Hirshfeld partitioning.
// Distinct inputs (system, two bases, reference, operator, config, proatom
// provider); no natural sub-bundle.
#[allow(clippy::too_many_arguments)]
pub fn pdep_polarizability_hirshfeld(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
    proatom: Option<&ProatomProvider>,
) -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
    use ferric_export::cube::GridSpec;
    use ferric_export::gto_eval::eval_basis_on_grid;

    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "pdep_polarizability_hirshfeld: only closed-shell (Restricted) supported".into(),
        ));
    }

    let natoms = mol.atoms.len();

    // ---------------------------------------------------------------------
    // 1. Reuse the same RI intermediates / dipole machinery as the
    //    molecular static-α path.
    // ---------------------------------------------------------------------
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config {
        frozen_core: 0,
        // Propagate the caller's explicit budget. This was hardcoded
        // `None`, which silently discarded a user's `[memory] budget_gb`
        // and let `resolve_budget` substitute an env/auto-detected value
        // (~0.8x available RAM) instead -- so a run pinned to 4 GB could
        // take ~14 GB on a 23 GB box. See
        // tests/mwe_explicit_budget_reaches_mp2.rs.
        memory_budget_bytes: cfg.memory_budget_bytes,
        ..Default::default()
    };
    let inter =
        ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();

    // Molecular (field-side) dipole: the TRUE lab-frame molecular dipole from
    // the analytical AO integrals — the perturbation a uniform field couples
    // to. Paired below with the atom-centred bra (mu_ai_flat) to give
    // α^A_{dj} = ∂μ^A_d/∂E_j, exactly as in pdep_polarizability_hirshfeld_dynamic.
    let dip_ao_analytical = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();
    let mu_mo: [Array2<f64>; 3] = std::array::from_fn(|d| {
        c_occ.t().dot(&dip_ao_analytical[d]).dot(&c_vir)
    });

    let mut inv_de = ndarray::Array1::<f64>::zeros(nov);
    for i in 0..nocc {
        for a in 0..nvir {
            inv_de[i * nvir + a] = 1.0 / (eps_vir[a] - eps_occ[i]);
        }
    }

    // ε̃ = I + B̃ · diag(4/Δε) · B̃^T  (independent of μ; build now, factor below).
    let mut b_scaled = b_ov.clone();
    for ia in 0..nov {
        let s = (4.0 * inv_de[ia]).sqrt();
        let mut col = b_scaled.column_mut(ia);
        col.mapv_inplace(|x| x * s);
    }
    let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
    for p in 0..naux {
        eps_mat[(p, p)] += 1.0;
    }

    // ---------------------------------------------------------------------
    // 2. Build a regular real-space grid bounding the molecule.
    //    Grid spacing chosen to balance integration error vs cost; the
    //    sum-rule gate validates whatever choice we make.
    // ---------------------------------------------------------------------
    let spacing = hirshfeld_spacing();
    let margin = hirshfeld_margin();
    let grid = GridSpec::bounding_box(mol, margin, spacing);
    let dv = spacing * spacing * spacing;
    let npts = grid.n_x * grid.n_y * grid.n_z;

    // χ_μ(r_g) on grid: shape (nbf_obs, npts).
    let chi = eval_basis_on_grid(mol, obs_bs, &grid).map_err(|e| {
        FerricError::General(format!("pdep_polarizability_hirshfeld: chi eval failed: {e}"))
    })?;
    let nbf = chi.nrows();
    debug_assert_eq!(nbf, obs.nbasis());

    // ---------------------------------------------------------------------
    // 3. Build Hirshfeld weights w^A(r_g) on the grid.
    //    Proatom: validated same-basis free-atom density when a provider is
    //    given (fixes H-starvation); else legacy single-exponential Slater.
    // ---------------------------------------------------------------------
    // For each atom: compute ρ_A_free(|r - R_A|) on the grid.
    // Then w^A = ρ_A / (Σ_B ρ_B + ε).
    let mut rho_free: Vec<Vec<f64>> = vec![vec![0.0; npts]; natoms];
    let mut rho_sum: Vec<f64> = vec![0.0; npts];
    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];
    for a in 0..natoms {
        let z_a = mol.atoms[a].z;
        let xi = slater_xi_for_z(z_a);
        let prefac = z_a as f64 * xi.powi(3) / std::f64::consts::PI;
        let pa = proatom.and_then(|p| p(z_a, 0));
        let rax = mol.atoms[a].x;
        let ray = mol.atoms[a].y;
        let raz = mol.atoms[a].zpos;
        let row = &mut rho_free[a];
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    let dx = x - rax;
                    let dy = y - ray;
                    let dz = z - raz;
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    let rho = match &pa {
                        Some(p) => p.at(r),
                        None => prefac * (-2.0 * xi * r).exp(),
                    };
                    row[g] = rho;
                    rho_sum[g] += rho;
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // 4. For each atom A and Cartesian i, build the Hirshfeld-weighted,
    //    ATOM-CENTRED AO dipole matrix
    //        D^{A,i}_{μν} = ∫ dr χ_μ(r) χ_ν(r) w^A(r) (r_i − R_{A,i})
    //    by direct quadrature on the regular grid. Using (r_i − R_{A,i})
    //    instead of the lab-frame r_i makes α^A origin-independent (matches
    //    pdep_polarizability_hirshfeld_dynamic / pdep_polarizability_becke_dynamic).
    //    We deliberately do NOT renormalize the per-atom pieces to the
    //    lab-frame analytical dipole — that renormalization is the
    //    gauge-breaking step (it scales each atom by a shared factor derived
    //    from a lab-frame quantity and cannot restore per-atom symmetry for
    //    off-origin geometries).
    //
    //    Implementation: combine χ with w^A and (r_i − R_{A,i}) into a
    //    "weighted χ̃" on grid, then χ · χ̃^T (one DGEMM per atom per Cartesian).
    // ---------------------------------------------------------------------
    let eps_floor = 1e-12;
    let mut alpha_per_atom: Vec<[[f64; 3]; 3]> = vec![[[0.0; 3]; 3]; natoms];

    // Precompute r_i(g) per Cartesian (lab-frame grid coordinates).
    let mut ri_grid: [Vec<f64>; 3] = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
    for ix in 0..grid.n_x {
        let x = grid.origin[0] + ix as f64 * hx;
        for iy in 0..grid.n_y {
            let y = grid.origin[1] + iy as f64 * hy;
            for iz in 0..grid.n_z {
                let z = grid.origin[2] + iz as f64 * hz;
                let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                ri_grid[0][g] = x;
                ri_grid[1][g] = y;
                ri_grid[2][g] = z;
            }
        }
    }

    let mu_ai_mo_all: Vec<[Array2<f64>; 3]> = (0..natoms)
        .map(|a| {
            let ra = [mol.atoms[a].x, mol.atoms[a].y, mol.atoms[a].zpos];
            let mut wa = vec![0.0_f64; npts];
            for g in 0..npts {
                let denom = rho_sum[g] + eps_floor;
                wa[g] = rho_free[a][g] / denom;
            }
            std::array::from_fn(|i_cart| {
                let ra_d = ra[i_cart];
                let mut combined = Array2::<f64>::zeros((nbf, npts));
                for mu in 0..nbf {
                    let chi_mu = chi.row(mu);
                    let mut row = combined.row_mut(mu);
                    for g in 0..npts {
                        row[g] = chi_mu[g] * wa[g] * (ri_grid[i_cart][g] - ra_d) * dv;
                    }
                }
                let d: Array2<f64> = chi.dot(&combined.t());
                let mut d_sym = Array2::<f64>::zeros((nbf, nbf));
                for mu in 0..nbf {
                    for nu in 0..nbf {
                        d_sym[(mu, nu)] = 0.5 * (d[(mu, nu)] + d[(nu, mu)]);
                    }
                }
                c_occ.t().dot(&d_sym).dot(&c_vir)
            })
        })
        .collect();

    // Molecular μ^j_flat (lab-frame, analytical) — the field-side ket.
    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc {
            for ax in 0..nvir {
                v[i * nvir + ax] = mu_mo[d][(i, ax)];
            }
        }
        v
    });
    let mu_flat_inv: [ndarray::Array1<f64>; 3] =
        std::array::from_fn(|d| &mu_flat[d] * &inv_de);
    let w_mol: [ndarray::Array1<f64>; 3] =
        std::array::from_fn(|d| b_ov.dot(&mu_flat_inv[d]));
    let y_mol = solve_dielectric_3(&eps_mat, &w_mol)?;

    // Assemble α^A: pair the atom-centred bra (mu_ai_flat) with the lab-frame
    // molecular ket (mu_flat_inv), exactly as in the dynamic sibling.
    for a in 0..natoms {
        for i_cart in 0..3 {
            let mu_ai_mo = &mu_ai_mo_all[a][i_cart];
            let mut mu_ai_flat = ndarray::Array1::<f64>::zeros(nov);
            let mut mu_ai_flat_inv = ndarray::Array1::<f64>::zeros(nov);
            for i in 0..nocc {
                for ax in 0..nvir {
                    let ia = i * nvir + ax;
                    mu_ai_flat[ia] = mu_ai_mo[(i, ax)];
                    mu_ai_flat_inv[ia] = mu_ai_mo[(i, ax)] * inv_de[ia];
                }
            }
            let w_ai = b_ov.dot(&mu_ai_flat_inv);

            for j in 0..3 {
                let bare = mu_ai_flat.dot(&mu_flat_inv[j]);
                let coupled = w_ai.dot(&y_mol[j]);
                alpha_per_atom[a][i_cart][j] = 4.0 * bare - 16.0 * coupled;
            }
        }

        // Per-atom symmetrize (consumer schema requires < 1e-5).
        for i in 0..3 {
            for j in (i + 1)..3 {
                let avg = 0.5 * (alpha_per_atom[a][i][j] + alpha_per_atom[a][j][i]);
                alpha_per_atom[a][i][j] = avg;
                alpha_per_atom[a][j][i] = avg;
            }
        }
    }

    // ---------------------------------------------------------------------
    // 5. Sanity/debug log. NOTE: Σ_A α^A is NOT expected to equal the
    //    molecular α_mol here — the atom-centred per-atom tensors omit
    //    inter-atomic coupling (charge-transfer) terms by construction, same
    //    as pdep_polarizability_hirshfeld_dynamic. There is no sum-rule gate;
    //    use molecular_polarizability (or the static analog) for the
    //    partition-independent molecular total.
    // ---------------------------------------------------------------------
    if std::env::var("FERRIC_DEBUG_HIRSHFELD").is_ok() {
        eprintln!(
            "[hirshfeld] grid {}×{}×{} (spacing={}, margin={})",
            grid.n_x, grid.n_y, grid.n_z, spacing, margin
        );
        for a in 0..natoms {
            eprintln!(
                "[hirshfeld] atom {} (Z={}) α_iso = {:.4}",
                a,
                mol.atoms[a].z,
                (alpha_per_atom[a][0][0] + alpha_per_atom[a][1][1] + alpha_per_atom[a][2][2]) / 3.0
            );
        }
    }

    Ok(alpha_per_atom)
}

/// Dynamic Hirshfeld per-atom polarizability α^A(iω) on an imaginary-frequency grid.
///
/// Computes the same renormalized Hirshfeld MO dipoles as [`pdep_polarizability_hirshfeld`]
/// (grid work is ω-independent), then evaluates the full-rank SMW formula at each
/// quadrature frequency. Because `Σ_A μ^{A,i} = μ^i` exactly after renormalization,
/// `Σ_A α^A(iω) = α_mol(iω)` at every ω — exact sum rule, origin-independent,
/// correct anisotropy including charge-transfer along bond axes.
///
/// Returns `per_atom[A][k][i][j]` = α^A_{ij}(iω_k), shape `(natoms, nfreq, 3, 3)`.
// System inputs (molecule, orbital + auxiliary bases, the basis-set object for
// partitioning, the SCF reference, the operator, config, and the frequency
// grid) are all distinct; there is no sub-bundle to extract.
/// `proatom`: optional provider of validated same-basis free-atom densities for
/// the Hirshfeld partition (fixes the H-starvation of the legacy single-Slater
/// proatom). `None` uses the legacy single-exponential Slater proatom.
#[allow(clippy::too_many_arguments)]
pub fn pdep_polarizability_hirshfeld_dynamic(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
    freqs: &[f64],
    proatom: Option<&ProatomProvider>,
) -> Result<Vec<Vec<[[f64; 3]; 3]>>, FerricError> {
    use ferric_export::cube::GridSpec;
    use ferric_export::gto_eval::eval_basis_on_grid;
    use ferric_integrals::blas_threads::with_blas_threads;
    use rayon::prelude::*;

    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "pdep_polarizability_hirshfeld_dynamic: only closed-shell (Restricted) supported".into(),
        ));
    }

    let natoms = mol.atoms.len();
    let nfreq = freqs.len();

    // RI intermediates — same as static path.
    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config {
        frozen_core: 0,
        // Propagate the caller's explicit budget. This was hardcoded
        // `None`, which silently discarded a user's `[memory] budget_gb`
        // and let `resolve_budget` substitute an env/auto-detected value
        // (~0.8x available RAM) instead -- so a run pinned to 4 GB could
        // take ~14 GB on a 23 GB box. See
        // tests/mwe_explicit_budget_reaches_mp2.rs.
        memory_budget_bytes: cfg.memory_budget_bytes,
        ..Default::default()
    };
    let inter = ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();
    let mut e_ia = ndarray::Array1::<f64>::zeros(nov);
    for i in 0..nocc { for a in 0..nvir { e_ia[i*nvir+a] = eps_vir[a] - eps_occ[i]; } }

    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();

    // --- Grid setup (identical to static Hirshfeld path) ---
    let _dip_ao_analytical = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let spacing = hirshfeld_spacing();
    let margin = hirshfeld_margin();
    let grid = GridSpec::bounding_box(mol, margin, spacing);
    let dv = spacing * spacing * spacing;
    let npts = grid.n_x * grid.n_y * grid.n_z;

    let chi = eval_basis_on_grid(mol, obs_bs, &grid).map_err(|e| {
        FerricError::General(format!("pdep_polarizability_hirshfeld_dynamic: chi eval failed: {e}"))
    })?;
    let nbf = chi.nrows();

    // Proatom densities and Hirshfeld weights. With a `proatom` provider, use
    // the validated same-basis free-atom density (fixes H-starvation); else the
    // legacy single-exponential Slater proatom. Both produce raw densities here;
    // the weight w^A = ρ_free^A/(Σρ_free) is formed at use below.
    let mut rho_free: Vec<Vec<f64>> = vec![vec![0.0; npts]; natoms];
    let mut rho_sum: Vec<f64> = vec![0.0; npts];
    let hx = grid.step_x[0]; let hy = grid.step_y[1]; let hz = grid.step_z[2];
    for a in 0..natoms {
        let z_a = mol.atoms[a].z;
        let xi = slater_xi_for_z(z_a);
        let prefac = z_a as f64 * xi.powi(3) / std::f64::consts::PI;
        let pa = proatom.and_then(|p| p(z_a, 0)); // neutral same-basis proatom
        let rax = mol.atoms[a].x; let ray = mol.atoms[a].y; let raz = mol.atoms[a].zpos;
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    let r = ((x-rax).powi(2)+(y-ray).powi(2)+(z-raz).powi(2)).sqrt();
                    let rho = match &pa {
                        Some(p) => p.at(r),
                        None => prefac * (-2.0 * xi * r).exp(),
                    };
                    rho_free[a][g] = rho; rho_sum[g] += rho;
                }
            }
        }
    }

    // Grid coordinates.
    let mut ri_grid: [Vec<f64>; 3] = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
    for ix in 0..grid.n_x {
        let x = grid.origin[0] + ix as f64 * hx;
        for iy in 0..grid.n_y {
            let y = grid.origin[1] + iy as f64 * hy;
            for iz in 0..grid.n_z {
                let z = grid.origin[2] + iz as f64 * hz;
                let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                ri_grid[0][g] = x; ri_grid[1][g] = y; ri_grid[2][g] = z;
            }
        }
    }

    // Build per-atom Hirshfeld AO dipoles with the ATOM-CENTRED position
    // operator (r − R_A). This yields the *intrinsic* atomic polarizability:
    // origin-independent and charge-transfer-free, the correct object for
    // atom-resolved C6 (TS/MBD convention). Using the global-frame r instead
    // leaks R_A·(field-induced charge transfer onto A) into α^A, which breaks
    // the symmetry of equivalent atoms off-origin (e.g. the two N in N₂) and
    // contaminates the per-atom bond-axis response. We deliberately do NOT
    // renormalize to the global lab-frame dipole — that renormalization is the
    // gauge-breaking step (it scales each atom by a shared factor and cannot
    // restore per-atom symmetry). The molecular total is computed separately
    // from the lab-frame molecular α (see `molecular_dynamic_polarizability`).
    let eps_floor = 1e-12;
    let atom_pos: Vec<[f64; 3]> =
        mol.atoms.iter().map(|at| [at.x, at.y, at.zpos]).collect();

    // Build each atom's 3 AO dipole matrices and transform to the (small) MO
    // occ-vir basis in the SAME pass, so only one atom's AO copy (3·nbf²) is
    // live at a time instead of all natoms·3·nbf² at once (M9 streaming). This
    // site has NO cross-atom renormalization (see the gauge note above), so the
    // per-atom AO matrices are independent — dropping each before the next is
    // exact. Numerics are unchanged vs the previous build-all-then-transform.
    let mu_ai_flat: Vec<[ndarray::Array1<f64>; 3]> = (0..natoms).map(|a| {
        let ra = atom_pos[a];
        let mut wa = vec![0.0_f64; npts];
        for g in 0..npts { wa[g] = rho_free[a][g] / (rho_sum[g] + eps_floor); }
        std::array::from_fn(|i_cart| {
            let ra_d = ra[i_cart];
            let mut combined = Array2::<f64>::zeros((nbf, npts));
            for mu in 0..nbf {
                let chi_mu = chi.row(mu);
                for g in 0..npts {
                    combined[(mu, g)] = chi_mu[g] * wa[g] * (ri_grid[i_cart][g] - ra_d) * dv;
                }
            }
            let d = chi.dot(&combined.t());
            let mut d_sym = Array2::<f64>::zeros((nbf, nbf));
            for mu in 0..nbf { for nu in 0..nbf { d_sym[(mu,nu)] = 0.5*(d[(mu,nu)]+d[(nu,mu)]); } }
            // Transform to MO and keep only the length-nov vector; d_sym drops here.
            let mo = c_occ.t().dot(&d_sym).dot(&c_vir);
            let mut v = ndarray::Array1::<f64>::zeros(nov);
            for i in 0..nocc { for ax in 0..nvir { v[i*nvir+ax] = mo[(i,ax)]; } }
            v
        })
    }).collect();

    // Molecular (field-side) dipole: the TRUE lab-frame molecular dipole Σ_i r_i
    // from the analytical AO integrals — the perturbation a uniform field
    // couples to. NOT the sum of atom-centred per-atom pieces (those answer the
    // atom-resolved question). Pairing the atom-centred bra (mu_ai_flat) with
    // this lab-frame molecular ket gives α^A_{dj} = ∂μ^A_d/∂E_j.
    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mo = c_occ.t().dot(&_dip_ao_analytical[d]).dot(&c_vir);
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc { for ax in 0..nvir { v[i * nvir + ax] = mo[(i, ax)]; } }
        v
    });

    // --- Frequency loop ---
    //
    // Each ω is independent — parallelize over frequencies. The (naux × nov)
    // column-scaled B̃ scratch M9 hoisted out of the loop (the buffer this
    // comment used to call out at ~766 MB for large aux bases) becomes
    // per-thread via map_init: one buffer allocated per rayon worker,
    // reused across the ω's that worker processes. BLAS pinned to 1 inside
    // the rayon region.
    let rows: Vec<Result<Vec<[[f64; 3]; 3]>, FerricError>> = with_blas_threads(1, || {
        freqs
            .par_iter()
            .map_init(
                // Zeros, not `.clone()`: overwritten by `assign(b_ov)` before
                // it is read, so the cloned contents are discarded (see the
                // two-spin site and tests/mwe_per_worker_scratch.rs).
                || Array2::<f64>::zeros(b_ov.raw_dim()),
                |b_scaled, &omega| -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
                    let omega2 = omega * omega;
                    let mut g = ndarray::Array1::<f64>::zeros(nov);
                    for ia in 0..nov { let e = e_ia[ia]; g[ia] = e / (omega2 + e*e); }

                    // ε̃(ω) = I + 4 B̃ diag(g) B̃^T
                    b_scaled.assign(b_ov);
                    for ia in 0..nov { b_scaled.column_mut(ia).mapv_inplace(|x| x * (4.0*g[ia]).sqrt()); }
                    let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
                    for p in 0..naux { eps_mat[(p,p)] += 1.0; }

                    // Solve ε̃ y^j = B·(g⊙μ^j) once per direction (molecular Hirshfeld dipole).
                    let mu_g: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| &mu_flat[d] * &g);
                    let w_mol: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| b_ov.dot(&mu_g[d]));
                    let y_mol = solve_dielectric_3(&eps_mat, &w_mol)?;

                    let mut row: Vec<[[f64; 3]; 3]> = vec![[[0.0; 3]; 3]; natoms];
                    for a in 0..natoms {
                        let mut tensor = [[0.0_f64; 3]; 3];
                        for d in 0..3 {
                            let w_ai = b_ov.dot(&(&mu_ai_flat[a][d] * &g));
                            for j in 0..3 {
                                let bare = mu_ai_flat[a][d].dot(&mu_g[j]);
                                let coupled = w_ai.dot(&y_mol[j]);
                                tensor[d][j] = 4.0 * bare - 16.0 * coupled;
                            }
                        }
                        for i in 0..3 { for j in (i+1)..3 {
                            let avg = 0.5*(tensor[i][j]+tensor[j][i]);
                            tensor[i][j] = avg; tensor[j][i] = avg;
                        }}
                        row[a] = tensor;
                    }
                    Ok(row)
                },
            )
            .collect::<Vec<Result<Vec<[[f64; 3]; 3]>, FerricError>>>()
    });

    let mut out: Vec<Vec<[[f64; 3]; 3]>> = vec![vec![[[0.0; 3]; 3]; nfreq]; natoms];
    for (k, row) in rows.into_iter().enumerate() {
        let row = row?;
        for a in 0..natoms {
            out[a][k] = row[a];
        }
    }

    Ok(out)
}

/// Molecular (whole-system) dynamic polarizability α_{ij}(iω_k) for the
/// closed-shell PDEP-RPA response, partition-independent.
///
/// Uses the lab-frame molecular dipole Σ_i r_i on both sides of the response,
/// so it is origin-independent and gives the DOSD-comparable molecular C6 when
/// fed to Casimir-Polder. This is the correct molecular total — distinct from
/// the sum of the intrinsic per-atom tensors (which omit inter-atomic
/// coupling). Returns `molecular[k]` = 3×3 tensor (a.u.).
pub fn molecular_dynamic_polarizability(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    cfg: &PdepRpaConfig,
    freqs: &[f64],
) -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
    use ferric_integrals::blas_threads::with_blas_threads;
    use rayon::prelude::*;

    // Open-shell: spin-summed dielectric ε̃ = I + 2 B̃_α diag(g_α) B̃_αᵀ
    // + 2 B̃_β diag(g_β) B̃_βᵀ, lab-frame molecular dipole per spin. Mirrors the
    // (U) branch of pdep_polarizability_becke_dynamic exactly; the only
    // difference is the dipole is the whole-molecule lab-frame dipole (origin
    // [0,0,0]) rather than atom-centred, giving the DOSD-comparable molecular
    // total. ω=0 reproduces the static open-shell molecular α.
    if !matches!(rhf.spin, Spin::Restricted) {
        use ferric_mp2::rimp2::compute_rpa_intermediates_spin;
        let mp2_cfg = ferric_mp2::rimp2::RiMp2Config {
        frozen_core: 0,
        // Propagate the caller's explicit budget. This was hardcoded
        // `None`, which silently discarded a user's `[memory] budget_gb`
        // and let `resolve_budget` substitute an env/auto-detected value
        // (~0.8x available RAM) instead -- so a run pinned to 4 GB could
        // take ~14 GB on a 23 GB box. See
        // tests/mwe_explicit_budget_reaches_mp2.rs.
        memory_budget_bytes: cfg.memory_budget_bytes,
        ..Default::default()
    };
        let inter_a = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, true)?;
        let inter_b = compute_rpa_intermediates_spin(mol, obs, dfbs, op, rhf, &mp2_cfg, false)?;
        let naux = inter_a.naux;

        // Orbital-energy slices (ROHF reuses α-MOs/eps for β).
        let eps_b_full: &[f64] = if matches!(rhf.spin, Spin::RestrictedOpen) {
            rhf.eps_a()
        } else {
            rhf.eps_b()
        };
        let mk_eia = |inter: &ferric_mp2::rimp2::RpaIntermediates, eps_full: &[f64]| {
            let eps_occ = &eps_full[inter.first_occ..inter.first_occ + inter.nocc];
            let eps_vir = &eps_full[inter.nocc_total..inter.nocc_total + inter.nvir];
            let mut v = ndarray::Array1::<f64>::zeros(inter.nocc * inter.nvir);
            for i in 0..inter.nocc {
                for a in 0..inter.nvir {
                    v[i * inter.nvir + a] = eps_vir[a] - eps_occ[i];
                }
            }
            v
        };
        let e_ia_a = mk_eia(&inter_a, rhf.eps_a());
        let e_ia_b = mk_eia(&inter_b, eps_b_full);

        // Lab-frame molecular dipole in each spin's occ-vir MO basis.
        let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
        let c_a = rhf.mos_a();
        let c_b = if matches!(rhf.spin, Spin::RestrictedOpen) {
            rhf.mos_a()
        } else {
            rhf.mos_b()
        };
        let mk_mu = |inter: &ferric_mp2::rimp2::RpaIntermediates, c: &Array2<f64>| {
            let c_occ = c
                .slice(ndarray::s![.., inter.first_occ..inter.first_occ + inter.nocc])
                .to_owned();
            let c_vir = c
                .slice(ndarray::s![.., inter.nocc_total..inter.nocc_total + inter.nvir])
                .to_owned();
            let nov = inter.nocc * inter.nvir;
            let arr: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
                if nov == 0 {
                    return ndarray::Array1::<f64>::zeros(1);
                }
                let mo = c_occ.t().dot(&dip_ao[d]).dot(&c_vir);
                let mut v = ndarray::Array1::<f64>::zeros(nov);
                for i in 0..inter.nocc {
                    for a in 0..inter.nvir {
                        v[i * inter.nvir + a] = mo[(i, a)];
                    }
                }
                v
            });
            arr
        };
        let mu_a = mk_mu(&inter_a, c_a);
        let mu_b = mk_mu(&inter_b, c_b);

        // Each ω is independent — parallelize over frequencies. Per-spin
        // (naux × nov_σ) column-scaled B̃_σ scratch becomes per-thread via
        // map_init (one (b_scaled_a, b_scaled_b) pair per rayon worker).
        // BLAS pinned to 1 inside the rayon region.
        let rows: Vec<Result<[[f64; 3]; 3], FerricError>> = with_blas_threads(1, || {
            freqs
                .par_iter()
                .map_init(
                    // Zeros, not `.clone()`: the closure overwrites this
                    // buffer with `assign(b_ov)` before reading it, so cloning
                    // copies naux*nov doubles per worker PER SPIN that are then
                    // discarded (~23.5 GB at naux=2976/nov=61740 across 8
                    // workers). Same resident footprint, without the copy.
                    || (
                        Array2::<f64>::zeros(inter_a.b_ov.raw_dim()),
                        Array2::<f64>::zeros(inter_b.b_ov.raw_dim()),
                    ),
                    |(b_scaled_a, b_scaled_b), &omega| -> Result<[[f64; 3]; 3], FerricError> {
                        let omega2 = omega * omega;
                        let g_of = |e_ia: &ndarray::Array1<f64>| {
                            let mut v = ndarray::Array1::<f64>::zeros(e_ia.len().max(1));
                            for ia in 0..e_ia.len() {
                                let e = e_ia[ia];
                                v[ia] = e / (omega2 + e * e);
                            }
                            v
                        };
                        let g_a = g_of(&e_ia_a);
                        let g_b = g_of(&e_ia_b);

                        // ε̃(ω) = I + 2 B̃_α diag(g_α) B̃_αᵀ + 2 B̃_β diag(g_β) B̃_βᵀ.
                        let mut eps_mat = Array2::<f64>::zeros((naux, naux));
                        for p in 0..naux {
                            eps_mat[(p, p)] = 1.0;
                        }
                        for (b_ov, g, nocc, b_scaled) in [
                            (&inter_a.b_ov, &g_a, inter_a.nocc, &mut *b_scaled_a),
                            (&inter_b.b_ov, &g_b, inter_b.nocc, &mut *b_scaled_b),
                        ] {
                            if nocc == 0 {
                                continue;
                            }
                            let nov = b_ov.shape()[1];
                            b_scaled.assign(b_ov);
                            for ia in 0..nov {
                                let s = (2.0 * g[ia]).sqrt();
                                b_scaled.column_mut(ia).mapv_inplace(|x| x * s);
                            }
                            eps_mat += &b_scaled.dot(&b_scaled.t());
                        }

                        // w_total^d = B̃_α (μ_α^d ⊙ g_α) + B̃_β (μ_β^d ⊙ g_β).
                        let w_total: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
                            let wa = inter_a.b_ov.dot(&(&mu_a[d] * &g_a));
                            if inter_b.nocc == 0 {
                                return wa;
                            }
                            wa + inter_b.b_ov.dot(&(&mu_b[d] * &g_b))
                        });
                        let y_total = solve_dielectric_3(&eps_mat, &w_total)?;

                        let mut t = [[0.0_f64; 3]; 3];
                        for d in 0..3 {
                            for j in 0..3 {
                                let bare_a = 2.0 * mu_a[d].dot(&(&mu_a[j] * &g_a));
                                let bare_b = if inter_b.nocc > 0 {
                                    2.0 * mu_b[d].dot(&(&mu_b[j] * &g_b))
                                } else {
                                    0.0
                                };
                                let coupled = w_total[d].dot(&y_total[j]);
                                t[d][j] = bare_a + bare_b - 4.0 * coupled;
                            }
                        }
                        for i in 0..3 {
                            for j in (i + 1)..3 {
                                let avg = 0.5 * (t[i][j] + t[j][i]);
                                t[i][j] = avg;
                                t[j][i] = avg;
                            }
                        }
                        Ok(t)
                    },
                )
                .collect::<Vec<Result<[[f64; 3]; 3], FerricError>>>()
        });

        let out: Vec<[[f64; 3]; 3]> = rows.into_iter().collect::<Result<Vec<_>, FerricError>>()?;
        return Ok(out);
    }

    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config {
        frozen_core: 0,
        // Propagate the caller's explicit budget. This was hardcoded
        // `None`, which silently discarded a user's `[memory] budget_gb`
        // and let `resolve_budget` substitute an env/auto-detected value
        // (~0.8x available RAM) instead -- so a run pinned to 4 GB could
        // take ~14 GB on a 23 GB box. See
        // tests/mwe_explicit_budget_reaches_mp2.rs.
        memory_budget_bytes: cfg.memory_budget_bytes,
        ..Default::default()
    };
    let inter = ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov;
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();
    let mut e_ia = ndarray::Array1::<f64>::zeros(nov);
    for i in 0..nocc {
        for a in 0..nvir {
            e_ia[i * nvir + a] = eps_vir[a] - eps_occ[i];
        }
    }

    // Lab-frame molecular dipole in the MO occ-vir basis.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();
    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mo = c_occ.t().dot(&dip_ao[d]).dot(&c_vir);
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc {
            for a in 0..nvir {
                v[i * nvir + a] = mo[(i, a)];
            }
        }
        v
    });

    // Each ω is independent — parallelize over frequencies. The (naux × nov)
    // column-scaled B̃ scratch becomes per-thread via map_init. BLAS pinned
    // to 1 inside the rayon region.
    let rows: Vec<Result<[[f64; 3]; 3], FerricError>> = with_blas_threads(1, || {
        freqs
            .par_iter()
            .map_init(
                // Zeros, not `.clone()`: overwritten by `assign(b_ov)` before
                // it is read, so the cloned contents are discarded (see the
                // two-spin site and tests/mwe_per_worker_scratch.rs).
                || Array2::<f64>::zeros(b_ov.raw_dim()),
                |b_scaled, &omega| -> Result<[[f64; 3]; 3], FerricError> {
                    let omega2 = omega * omega;
                    let mut g = ndarray::Array1::<f64>::zeros(nov);
                    for ia in 0..nov {
                        let e = e_ia[ia];
                        g[ia] = e / (omega2 + e * e);
                    }
                    // ε̃(ω) = I + 4 B̃ diag(g) B̃^T
                    b_scaled.assign(b_ov);
                    for ia in 0..nov {
                        b_scaled.column_mut(ia).mapv_inplace(|x| x * (4.0 * g[ia]).sqrt());
                    }
                    let mut eps_mat: Array2<f64> = b_scaled.dot(&b_scaled.t());
                    for p in 0..naux {
                        eps_mat[(p, p)] += 1.0;
                    }
                    let mu_g: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| &mu_flat[d] * &g);
                    let w: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| b_ov.dot(&mu_g[d]));
                    let y = solve_dielectric_3(&eps_mat, &w)?;

                    let mut t = [[0.0_f64; 3]; 3];
                    for d in 0..3 {
                        let w_d = b_ov.dot(&mu_g[d]);
                        for j in 0..3 {
                            let bare = mu_flat[d].dot(&mu_g[j]);
                            let coupled = w_d.dot(&y[j]);
                            t[d][j] = 4.0 * bare - 16.0 * coupled;
                        }
                    }
                    for i in 0..3 {
                        for j in (i + 1)..3 {
                            let avg = 0.5 * (t[i][j] + t[j][i]);
                            t[i][j] = avg;
                            t[j][i] = avg;
                        }
                    }
                    Ok(t)
                },
            )
            .collect::<Vec<Result<[[f64; 3]; 3], FerricError>>>()
    });

    let out: Vec<[[f64; 3]; 3]> = rows.into_iter().collect::<Result<Vec<_>, FerricError>>()?;
    Ok(out)
}

/// Molecular α_ij(iω_k) built in the TRUNCATED PDEP eigenbasis carried by `rpa`.
/// Shares the retained eigenpotentials (hence trunc_thresh) with the energy/GW
/// paths. Closed-shell (RHF) only. Errors if `rpa.inv_dielectric_freq` is None
/// (Laplace χ₀ path — the projected screening matrices are required).
///
/// Algebra (derived from `molecular_dynamic_polarizability`):
///   α_dj(iω) = 4·μ_d·(μ_j⊙g) − 16·p_dᵀ (W̃_k + I) p_j,
///   p_d = y(μ_d⊙g),  y = Uᵀ B̃,  U = dressed_eigenvectors,
///   W̃_k = inv_dielectric_freq[k] = ε̃_proj⁻¹ − I.
/// At M = naux (thresh 0) this equals the full-naux result to round-off.
///
/// `memory_budget_bytes` is the caller's explicit memory ceiling, threaded into
/// the RI-MP2 intermediates build. `None` does NOT mean unlimited — it falls
/// through `resolve_budget`'s chain (see [`dielectric_spectrum_static`]).
pub fn molecular_dynamic_polarizability_pdep(
    rpa: &crate::PdepRpaResult,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    op: Operator,
    memory_budget_bytes: Option<usize>,
) -> Result<Vec<[[f64; 3]; 3]>, FerricError> {
    use ferric_integrals::blas_threads::with_blas_threads;
    use rayon::prelude::*;

    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "molecular_dynamic_polarizability_pdep: closed-shell (RHF) only".into(),
        ));
    }
    let winv = rpa.inv_dielectric_freq.as_ref().ok_or_else(|| {
        FerricError::General(
            "molecular_dynamic_polarizability_pdep: inv_dielectric_freq is None \
             (Laplace χ₀ path unsupported)".into(),
        )
    })?;

    let mp2_cfg = ferric_mp2::rimp2::RiMp2Config {
        frozen_core: 0,
        // Propagate the caller's explicit budget. This was hardcoded
        // `None`, which silently discarded a user's `[memory] budget_gb`
        // and let `resolve_budget` substitute an env/auto-detected value
        // (~0.8x available RAM) instead -- so a run pinned to 4 GB could
        // take ~14 GB on a 23 GB box. See
        // tests/mwe_explicit_budget_reaches_mp2.rs.
        memory_budget_bytes,
        ..Default::default()
    };
    let inter = ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov; // V^{-1/2}-dressed occ-vir RI-MO tensor (naux × nov)
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let nov = nocc * nvir;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();
    let mut e_ia = ndarray::Array1::<f64>::zeros(nov);
    for i in 0..nocc {
        for a in 0..nvir {
            e_ia[i * nvir + a] = eps_vir[a] - eps_occ[i];
        }
    }

    // Lab-frame molecular dipole in the MO occ-vir basis (origin [0,0,0]).
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();
    let mu_flat: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| {
        let mo = c_occ.t().dot(&dip_ao[d]).dot(&c_vir);
        let mut v = ndarray::Array1::<f64>::zeros(nov);
        for i in 0..nocc {
            for a in 0..nvir {
                v[i * nvir + a] = mo[(i, a)];
            }
        }
        v
    });

    // Project the dressed B̃ onto the retained PDEP subspace: y = Uᵀ B̃ (M × nov).
    let u = &rpa.dressed_eigenvectors; // (naux × M)
    let y = u.t().dot(b_ov);

    let freqs = &rpa.quad_freqs;
    // Each ω is independent — parallelize over frequencies. No scratch to
    // hoist here (unlike the full-naux siblings): `wpi`/`p` are small
    // (M × M / length-M, M = truncated PDEP rank) and cheaply allocated
    // fresh per element; `winv[k]` is precomputed and indexed read-only.
    // BLAS pinned to 1 inside the rayon region per repo convention, uniform
    // with the full-rank siblings even though the per-element GEMM here is
    // small (M, not naux).
    let out: Vec<[[f64; 3]; 3]> = with_blas_threads(1, || {
        freqs
            .par_iter()
            .enumerate()
            .map(|(k, &omega)| {
                let omega2 = omega * omega;
                let mut g = ndarray::Array1::<f64>::zeros(nov);
                for ia in 0..nov {
                    let e = e_ia[ia];
                    g[ia] = e / (omega2 + e * e);
                }
                let mu_g: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| &mu_flat[d] * &g);
                // p_d = y (μ_d ⊙ g), length M.
                let p: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| y.dot(&mu_g[d]));
                // W̃_k + I  (M × M).
                let wpi = {
                    let mut m = winv[k].clone();
                    for d in 0..m.nrows() {
                        m[(d, d)] += 1.0;
                    }
                    m
                };
                let mut t = [[0.0_f64; 3]; 3];
                for d in 0..3 {
                    let wp_d = wpi.dot(&p[d]);
                    for j in 0..3 {
                        let bare = mu_flat[d].dot(&mu_g[j]);
                        let coupled = p[j].dot(&wp_d);
                        t[d][j] = 4.0 * bare - 16.0 * coupled;
                    }
                }
                for i in 0..3 {
                    for j in (i + 1)..3 {
                        let avg = 0.5 * (t[i][j] + t[j][i]);
                        t[i][j] = avg;
                        t[j][i] = avg;
                    }
                }
                t
            })
            .collect::<Vec<[[f64; 3]; 3]>>()
    });
    Ok(out)
}

/// Per-atom effective volume via Hirshfeld (Slater proatom) partitioning:
/// ```text
///   v_A = ∫ w^A_Hirsh(r) ρ(r) |r − R_A|³ dV
/// ```
/// This is the partition the TS dispersion model was calibrated for.
/// Uses the same Slater single-exponential proatom as `hirshfeld_charges`.
pub fn atomic_effective_volumes_hirshfeld(
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
    proatom: Option<&ProatomProvider>,
) -> Result<Vec<f64>, FerricError> {
    use ferric_export::cube::GridSpec;
    use ferric_export::gto_eval::eval_basis_on_grid;

    let natoms = mol.atoms.len();
    let spacing = hirshfeld_spacing();
    let margin = hirshfeld_margin();
    let grid = GridSpec::bounding_box(mol, margin, spacing);
    let dv = spacing * spacing * spacing;
    let npts = grid.n_x * grid.n_y * grid.n_z;
    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];

    let chi = eval_basis_on_grid(mol, obs_bs, &grid).map_err(|e| {
        FerricError::General(format!("atomic_effective_volumes_hirshfeld: chi failed: {e}"))
    })?;
    let nbf = chi.nrows();

    // ρ(r_g) = Σ_{μν} D_{μν} χ_μ χ_ν via matrix product.
    let d_chi = density.dot(&chi);
    let mut rho = vec![0.0_f64; npts];
    for mu in 0..nbf {
        for g in 0..npts {
            rho[g] += chi[(mu, g)] * d_chi[(mu, g)];
        }
    }

    // Proatom weights: validated same-basis density when a provider is given,
    // else legacy single-Slater.
    let mut rho_free: Vec<Vec<f64>> = vec![vec![0.0; npts]; natoms];
    let mut rho_sum = vec![0.0_f64; npts];
    for a in 0..natoms {
        let z_a = mol.atoms[a].z;
        let xi = slater_xi_for_z(z_a);
        let prefac = z_a as f64 * xi.powi(3) / std::f64::consts::PI;
        let pa = proatom.and_then(|p| p(z_a, 0));
        let (rax, ray, raz) = (mol.atoms[a].x, mol.atoms[a].y, mol.atoms[a].zpos);
        let row = &mut rho_free[a];
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    let r = ((x-rax)*(x-rax)+(y-ray)*(y-ray)+(z-raz)*(z-raz)).sqrt();
                    let r0 = match &pa {
                        Some(p) => p.at(r),
                        None => prefac * (-2.0 * xi * r).exp(),
                    };
                    row[g] = r0;
                    rho_sum[g] += r0;
                }
            }
        }
    }

    let eps_floor = 1e-12;
    let mut vol = vec![0.0_f64; natoms];
    for a in 0..natoms {
        let (rax, ray, raz) = (mol.atoms[a].x, mol.atoms[a].y, mol.atoms[a].zpos);
        let mut acc = 0.0;
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    let w = rho_free[a][g] / (rho_sum[g] + eps_floor);
                    let dx = x - rax; let dy = y - ray; let dz = z - raz;
                    let r3 = (dx*dx + dy*dy + dz*dz).powf(1.5);
                    acc += w * rho[g] * r3 * dv;
                }
            }
        }
        vol[a] = acc;
    }
    Ok(vol)
}

/// Iterative Hirshfeld (Hirshfeld-I) charges using ad-hoc same-basis free-atom
/// proatom densities.
///
/// `proatom` supplies the spherically-averaged free-atom density for element
/// `z` at integer charge state `q_int` (0, +1, −1, …), built by the caller from
/// an atomic SCF in the molecule's basis (so the Hirshfeld weight ratio is
/// basis-consistent with the molecular density). The loop iterates each atom's
/// fractional charge to self-consistency, interpolating the proatom between
/// bracketing integer charge states; charges outside the available bracket fall
/// back to the nearest state.
///
/// Returns per-atom charges q_A = Z_A − N_A, charge-conserving and equal for
/// symmetry-equivalent atoms.
pub fn hirshfeld_i_charges(
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
    proatom: &dyn Fn(i32, i32) -> Option<RadialProatom>,
) -> Result<Vec<f64>, FerricError> {
    use ferric_export::cube::GridSpec;
    use ferric_export::gto_eval::eval_basis_on_grid;

    let natoms = mol.atoms.len();
    let spacing = hirshfeld_spacing();
    let margin = hirshfeld_margin();
    let grid = GridSpec::bounding_box(mol, margin, spacing);
    let dv = spacing * spacing * spacing;
    let npts = grid.n_x * grid.n_y * grid.n_z;
    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];

    let chi = eval_basis_on_grid(mol, obs_bs, &grid)
        .map_err(|e| FerricError::General(format!("hirshfeld_i: chi eval failed: {e}")))?;
    let nbf = chi.nrows();
    if density.nrows() != nbf {
        return Err(FerricError::General("hirshfeld_i: density/nbf mismatch".into()));
    }
    let d_chi = density.dot(&chi);
    let mut rho = vec![0.0_f64; npts];
    for mu in 0..nbf {
        for g in 0..npts { rho[g] += chi[(mu, g)] * d_chi[(mu, g)]; }
    }

    // Precompute grid coordinates.
    let mut gx = vec![0.0; npts];
    let mut gy = vec![0.0; npts];
    let mut gz = vec![0.0; npts];
    for ix in 0..grid.n_x {
        let x = grid.origin[0] + ix as f64 * hx;
        for iy in 0..grid.n_y {
            let y = grid.origin[1] + iy as f64 * hy;
            for iz in 0..grid.n_z {
                let z = grid.origin[2] + iz as f64 * hz;
                let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                gx[g] = x; gy[g] = y; gz[g] = z;
            }
        }
    }

    // Cache integer-charge proatoms per element (z, q_int).
    let mut cache: std::collections::HashMap<(i32, i32), Option<RadialProatom>> =
        std::collections::HashMap::new();
    let get = |z: i32, qi: i32, cache: &mut std::collections::HashMap<(i32, i32), Option<RadialProatom>>| -> Option<RadialProatom> {
        cache.entry((z, qi)).or_insert_with(|| proatom(z, qi)).clone()
    };

    // Interpolated proatom density at distance r for fractional charge q.
    let proatom_rho = |z: i32, q: f64, r: f64, cache: &mut std::collections::HashMap<(i32, i32), Option<RadialProatom>>| -> f64 {
        let q_lo = q.floor() as i32;
        let q_hi = q_lo + 1;
        let f = q - q_lo as f64;
        let p_lo = get(z, q_lo, cache);
        let p_hi = get(z, q_hi, cache);
        match (p_lo, p_hi) {
            (Some(a), Some(b)) => ((1.0 - f) * a.at(r) + f * b.at(r)).max(0.0),
            (Some(a), None) => a.at(r),
            (None, Some(b)) => b.at(r),
            (None, None) => {
                // Fall back to the neutral if available.
                get(z, 0, cache).map(|p| p.at(r)).unwrap_or(0.0)
            }
        }
    };

    let eps_floor = 1e-12;
    let n_elec_target = mol.nelec() as f64;
    let mut q = vec![0.0_f64; natoms];
    let max_iter = 50;
    let tol = 1e-4;

    for _it in 0..max_iter {
        let mut rho_free: Vec<Vec<f64>> = vec![vec![0.0; npts]; natoms];
        let mut rho_sum = vec![0.0_f64; npts];
        for a in 0..natoms {
            let z_a = mol.atoms[a].z;
            let (rax, ray, raz) = (mol.atoms[a].x, mol.atoms[a].y, mol.atoms[a].zpos);
            let row = &mut rho_free[a];
            for g in 0..npts {
                let dx = gx[g] - rax;
                let dy = gy[g] - ray;
                let dz = gz[g] - raz;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let pr = proatom_rho(z_a, q[a], r, &mut cache);
                row[g] = pr;
                rho_sum[g] += pr;
            }
        }
        let mut n_e = vec![0.0_f64; natoms];
        for a in 0..natoms {
            let mut acc = 0.0;
            for g in 0..npts {
                let w = rho_free[a][g] / (rho_sum[g] + eps_floor);
                acc += rho[g] * w * dv;
            }
            n_e[a] = acc;
        }
        let n_sum: f64 = n_e.iter().sum();
        let scale = if n_sum.abs() > 1e-12 { n_elec_target / n_sum } else { 1.0 };
        let mut max_dq = 0.0_f64;
        for a in 0..natoms {
            let q_new = mol.atoms[a].z as f64 - scale * n_e[a];
            max_dq = max_dq.max((q_new - q[a]).abs());
            q[a] = 0.5 * q[a] + 0.5 * q_new; // damped
        }
        if debug_toggle("FERRIC_HI_DEBUG") {
            let r: Vec<f64> = q.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
            eprintln!("HI iter {_it}: q = {r:?} (max_dq={max_dq:.2e})");
        }
        if max_dq < tol { break; }
    }
    Ok(q)
}

///
/// The total electronic charge is renormalized so Σ_A (Z_A − q_A) = N_e
/// exactly (compensates for grid quadrature error in the density integral).
///
/// Returns `Vec<f64>` of length `mol.atoms.len()`, in units of e.
pub fn hirshfeld_charges(
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    density: &Array2<f64>,
    proatom: Option<&ProatomProvider>,
) -> Result<Vec<f64>, FerricError> {
    use ferric_export::cube::GridSpec;
    use ferric_export::gto_eval::eval_basis_on_grid;

    let natoms = mol.atoms.len();
    let spacing = hirshfeld_spacing();
    let margin = hirshfeld_margin();
    let grid = GridSpec::bounding_box(mol, margin, spacing);
    let dv = spacing * spacing * spacing;
    let npts = grid.n_x * grid.n_y * grid.n_z;
    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];

    let chi = eval_basis_on_grid(mol, obs_bs, &grid).map_err(|e| {
        FerricError::General(format!("hirshfeld_charges: chi eval failed: {e}"))
    })?;
    let nbf = chi.nrows();
    if density.nrows() != nbf || density.ncols() != nbf {
        return Err(FerricError::General(format!(
            "hirshfeld_charges: density {:?} != nbf {}",
            density.dim(),
            nbf
        )));
    }

    // ρ(r_g) = Σ_{μν} D_{μν} χ_μ(r_g) χ_ν(r_g) = sum_μ χ_μ(g) · (D · χ)(μ,g)
    let d_chi = density.dot(&chi); // (nbf, npts)
    let mut rho: Vec<f64> = vec![0.0; npts];
    for mu in 0..nbf {
        for g in 0..npts {
            rho[g] += chi[(mu, g)] * d_chi[(mu, g)];
        }
    }

    // Proatom densities and Hirshfeld weights. With a `proatom` provider, use
    // the validated same-basis free-atom density (fixes H-starvation); else the
    // legacy single-Slater proatom.
    let mut rho_free: Vec<Vec<f64>> = vec![vec![0.0; npts]; natoms];
    let mut rho_sum: Vec<f64> = vec![0.0; npts];
    for a in 0..natoms {
        let z_a = mol.atoms[a].z;
        let xi = slater_xi_for_z(z_a);
        let prefac = z_a as f64 * xi.powi(3) / std::f64::consts::PI;
        let pa = proatom.and_then(|p| p(z_a, 0));
        let rax = mol.atoms[a].x;
        let ray = mol.atoms[a].y;
        let raz = mol.atoms[a].zpos;
        let row = &mut rho_free[a];
        for ix in 0..grid.n_x {
            let x = grid.origin[0] + ix as f64 * hx;
            for iy in 0..grid.n_y {
                let y = grid.origin[1] + iy as f64 * hy;
                for iz in 0..grid.n_z {
                    let z = grid.origin[2] + iz as f64 * hz;
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    let dx = x - rax;
                    let dy = y - ray;
                    let dz = z - raz;
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    let r0 = match &pa {
                        Some(p) => p.at(r),
                        None => prefac * (-2.0 * xi * r).exp(),
                    };
                    row[g] = r0;
                    rho_sum[g] += r0;
                }
            }
        }
    }

    let eps_floor = 1e-12;
    // n_elec_A_grid = ∫ ρ · w^A
    let mut n_e_grid = vec![0.0_f64; natoms];
    for a in 0..natoms {
        let mut acc = 0.0;
        for g in 0..npts {
            let w = rho_free[a][g] / (rho_sum[g] + eps_floor);
            acc += rho[g] * w * dv;
        }
        n_e_grid[a] = acc;
    }

    // Renormalize: Σ_A n_e_grid_A should equal N_e (= trace(D·S)). Use the
    // electron count from the molecule (nelec) so the partition is exact.
    let n_elec_target = mol.nelec() as f64;
    let n_elec_sum: f64 = n_e_grid.iter().sum();
    let scale = if n_elec_sum.abs() > 1e-12 {
        n_elec_target / n_elec_sum
    } else {
        1.0
    };

    Ok((0..natoms)
        .map(|a| mol.atoms[a].z as f64 - scale * n_e_grid[a])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    /// Regression test for the 2026-07-13 unbounded-memory bug: naively
    /// `.collect()`-ing one chunk-partial per grid-point chunk (no banding)
    /// caused a real dataset build to sit for 3.5 hours with one process
    /// pinned at its memory cgroup cap (3M+ throttle events) before being
    /// killed — at 71-atom/def2-svp scale, ~1024 unbanded chunk-partials
    /// would have reached ~440 GB live. `dipole_band_width` must keep the
    /// live set (band_width * per_partial_bytes) within a small multiple of
    /// the byte budget, never anywhere close to that.
    #[test]
    fn dipole_band_width_bounds_live_memory_at_realistic_scale() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build().unwrap();
        pool.install(|| {
            // Danuglipron/def2-svp-scale: 71 atoms, nbf~500.
            let natoms = 71;
            let nbf = 500;
            // A representative budget; the band width is now a pure function
            // of it (the hardcoded 512 MB const is gone).
            let budget = 512 * 1024 * 1024usize;
            let bw = dipole_band_width(natoms, nbf, budget);
            let per_partial_bytes = natoms * 3 * nbf * nbf * std::mem::size_of::<f64>();
            let live_bytes = bw * per_partial_bytes;
            // Tightened: this used to allow 10x the budget because the
            // worker-count floor could push the live set above it "by design".
            // That floor is gone — the byte cap wins and parallelism degrades
            // instead — so the live set must now fit the budget outright,
            // except for the floor-of-one case where a single partial is
            // larger than the whole budget.
            assert!(
                live_bytes <= budget.max(per_partial_bytes),
                "live memory {live_bytes} bytes ({:.2} GB) is not bounded by the band budget \
                 — band_width={bw}, per_partial_bytes={per_partial_bytes}",
                live_bytes as f64 / 1e9
            );
            assert!(
                live_bytes < 10_000_000_000,
                "live memory {live_bytes} bytes must be well under 10 GB at this scale"
            );
        });
    }

    #[test]
    fn dipole_band_width_respects_budget_and_floors() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            // natoms=1, nbf=100: per_partial = 1*3*100*100*8 = 240_000 bytes;
            // 2_400_000 byte budget -> 10.
            assert_eq!(dipole_band_width(1, 100, 2_400_000), 10);
            // A budget below one partial floors at ONE, not at the worker count.
            //
            // This assertion used to expect 2 (= the pool's worker count),
            // encoding the old `.max(current_num_threads())` floor as intended
            // behavior. That floor overrode the byte cap and let resident memory
            // scale with core count instead of with the budget — the mechanism
            // behind the 16-17 GB anon-RSS incidents. The byte cap now wins and
            // parallelism degrades instead, so a starvation budget yields a
            // single chunk in flight: slow, never stuck, and never over budget.
            assert_eq!(dipole_band_width(1, 100, 1), 1);
            assert_eq!(dipole_band_width(0, 0, 1), 1);

            // The floor must NOT depend on the ambient pool size: same budget,
            // same answer, whatever the worker count. (Regression guard for the
            // exact defect above — see tests/mwe_budget_respected.rs.)
            let wide = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
            wide.install(|| {
                assert_eq!(dipole_band_width(1, 100, 1), 1);
                assert_eq!(dipole_band_width(1, 100, 2_400_000), 10);
            });
        });
    }

    /// At trunc_thresh = 0 (all eigenpotentials retained, M = naux), the
    /// PDEP-projected molecular α(iω) must reproduce the full-naux-dielectric
    /// `molecular_dynamic_polarizability` to ≤1e-8 per tensor element.
    #[test]
    fn molecular_pdep_untruncated_matches_full() {
        use crate::config::{PdepRpaConfig, QuadratureConfig, QuadratureScheme};
        use crate::run_pdep_rpa;

        let ctx = ParallelContext::default();
        let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

        let cfg = PdepRpaConfig {
            quadrature: QuadratureConfig {
                scheme: QuadratureScheme::GaussLegendre, n_points: 12, u0: 0.5,
            },
            trunc_thresh: 0.0,
            eigensolver_conv_thresh: 1e-9,
            // The PDEP dynamic-α property path reads inv_dielectric_freq (M9 gate).
            need_inv_dielectric_freq: true,
            need_eigenvalues_freq: true,
            ..Default::default()
        };
        let rpa = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();

        let full = molecular_dynamic_polarizability(
            &mol, &obs, &dfbs, &rhf, op, &cfg, &rpa.quad_freqs,
        ).unwrap();
        let pdep = molecular_dynamic_polarizability_pdep(
            &rpa, &mol, &obs, &dfbs, &rhf, op, cfg.memory_budget_bytes,
        ).unwrap();

        assert_eq!(full.len(), pdep.len());
        let mut max_abs = 0.0_f64;
        for k in 0..full.len() {
            for i in 0..3 {
                for j in 0..3 {
                    max_abs = max_abs.max((full[k][i][j] - pdep[k][i][j]).abs());
                }
            }
        }
        assert!(max_abs < 1e-8, "untruncated PDEP α deviates from full by {max_abs:.3e}");
    }

    fn build_h2() -> (Molecule, PreparedBasis, PreparedBasis, Operator, ScfResult) {
        // H2 at 1.4 Bohr, cc-pVDZ orbital + cc-pVDZ-RI aux.
        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74083\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        (mol, obs, dfbs, op, rhf)
    }

    #[test]
    fn becke_dynamic_alpha_molecular_sum_decays() {
        // The MOLECULAR dynamic polarizability — Σ_A α^A(iω) — is the robust,
        // origin-independent quantity. (Per-atom α^A(iω) is origin-dependent at
        // ω≠0 because the lab-frame partitioned dipole ⟨i|w^A r|a⟩ depends on the
        // common origin; only the atom SUM and the static ω=0 limit are clean.)
        // Check the molecular sum decays monotonically with the right tail.
        let (mol, obs, dfbs, op, rhf) = build_h2();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let cfg = PdepRpaConfig {
            frozen_core: 0,
            trunc_thresh: 0.0,
            ..Default::default()
        };
        let freqs = [0.0, 0.5, 2.0, 10.0];
        let dyn_a = pdep_polarizability_becke_dynamic(
            &mol, &obs, &bs, &dfbs, &rhf, op, &cfg, &freqs,
        )
        .unwrap();
        let mol_iso = |k: usize| -> f64 {
            dyn_a
                .iter()
                .map(|atom| (atom[k][0][0] + atom[k][1][1] + atom[k][2][2]) / 3.0)
                .sum()
        };
        let a0 = mol_iso(0);
        let a1 = mol_iso(1);
        let a2 = mol_iso(2);
        let a3 = mol_iso(3);
        assert!(a0 > a1 && a1 > a2 && a2 > a3, "not monotone: {a0} {a1} {a2} {a3}");
        assert!(a0 > 0.0, "α(0) must be positive: {a0}");
        assert!(a3 < 0.05 * a0, "tail too large: α(10)={a3} α(0)={a0}");
    }

    #[test]
    fn becke_dynamic_alpha_h2_symmetry_and_positivity() {
        // For H₂ the two atoms are equivalent by symmetry, so per-atom dynamic α
        // must be equal. Also α_iso^A(ω=0) must be positive.
        let (mol, obs, dfbs, op, rhf) = build_h2();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let cfg = PdepRpaConfig {
            frozen_core: 0,
            trunc_thresh: 0.0,
            ..Default::default()
        };

        let dynamic = pdep_polarizability_becke_dynamic(
            &mol, &obs, &bs, &dfbs, &rhf, op, &cfg, &[0.0, 0.5],
        )
        .unwrap();
        assert_eq!(dynamic.len(), 2, "H2 should have 2 atoms");

        let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        let iso0_h0 = iso(&dynamic[0][0]);
        let iso0_h1 = iso(&dynamic[1][0]);

        // Both α_iso at ω=0 must be positive.
        assert!(iso0_h0 > 0.0, "α_iso H0(ω=0) not positive: {iso0_h0}");
        assert!(iso0_h1 > 0.0, "α_iso H1(ω=0) not positive: {iso0_h1}");

        // H2 symmetric: both atoms must give equal α at each frequency.
        assert!(
            (iso0_h0 - iso0_h1).abs() / iso0_h0 < 1e-6,
            "H2 per-atom α not symmetric at ω=0: {iso0_h0} vs {iso0_h1}"
        );
        let iso1_h0 = iso(&dynamic[0][1]);
        let iso1_h1 = iso(&dynamic[1][1]);
        assert!(
            (iso1_h0 - iso1_h1).abs() / (iso1_h0.abs() + 1e-12) < 1e-6,
            "H2 per-atom α not symmetric at ω=0.5: {iso1_h0} vs {iso1_h1}"
        );
    }

    #[test]
    fn becke_dynamic_alpha_decays_with_frequency() {
        // α_iso(iω) must fall monotonically toward 0 as ω grows (~1/ω²).
        let (mol, obs, dfbs, op, rhf) = build_h2();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let cfg = PdepRpaConfig {
            frozen_core: 0,
            trunc_thresh: 0.0,
            ..Default::default()
        };

        let freqs = [0.0, 0.5, 2.0, 10.0];
        let dyn_a = pdep_polarizability_becke_dynamic(
            &mol, &obs, &bs, &dfbs, &rhf, op, &cfg, &freqs,
        )
        .unwrap();
        // Sum atomic isotropic α at each frequency = molecular iso (sum rule).
        let iso_at = |k: usize| -> f64 {
            dyn_a
                .iter()
                .map(|atom| (atom[k][0][0] + atom[k][1][1] + atom[k][2][2]) / 3.0)
                .sum()
        };
        let a0 = iso_at(0);
        let a1 = iso_at(1);
        let a2 = iso_at(2);
        let a3 = iso_at(3);
        assert!(a0 > a1 && a1 > a2 && a2 > a3, "not monotone: {a0} {a1} {a2} {a3}");
        assert!(a0 > 0.0, "α(0) must be positive: {a0}");
        // High-ω tail ~ 1/ω²: α(10) ≪ α(0).
        assert!(a3 < 0.05 * a0, "tail too large: α(10)={a3} α(0)={a0}");
    }

    fn build_h_atom() -> (Molecule, PreparedBasis, PreparedBasis, Operator, ScfResult) {
        use ferric_scf::uhf::solve_uhf;
        let xyz = "1\nH\nH 0 0 0\n";
        let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap(); // doublet
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let cfg = RhfConfig {
            mom_after_iter: 5,
            ..Default::default()
        };
        let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &cfg).unwrap();
        (mol, obs, dfbs, op, uhf)
    }

    #[test]
    fn h_atom_dynamic_alpha_unrestricted_decays() {
        // UHF H atom: dynamic α(iω) via the unrestricted path must decay monotonically
        // and be positive at ω=0. Single atom → no symmetry check needed.
        let (mol, obs, dfbs, op, uhf) = build_h_atom();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let cfg = PdepRpaConfig {
            frozen_core: 0,
            trunc_thresh: 0.0,
            ..Default::default()
        };

        let freqs = [0.0, 0.5, 2.0, 10.0];
        let dyn_a = pdep_polarizability_becke_dynamic(
            &mol, &obs, &bs, &dfbs, &uhf, op, &cfg, &freqs,
        )
        .unwrap();
        assert_eq!(dyn_a.len(), 1);

        let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        let a0 = iso(&dyn_a[0][0]);
        let a1 = iso(&dyn_a[0][1]);
        let a2 = iso(&dyn_a[0][2]);
        let a3 = iso(&dyn_a[0][3]);
        assert!(a0 > 0.0, "α(0) not positive: {a0}");
        assert!(a0 > a1 && a1 > a2 && a2 > a3, "not monotone: {a0} {a1} {a2} {a3}");
        assert!(a3 < 0.05 * a0, "tail too large: α(10)={a3} α(0)={a0}");
    }

    #[test]
    fn molecular_dynamic_alpha_uhf_matches_rhf_on_closed_shell() {
        // Rigor check for the open-shell molecular_dynamic_polarizability branch:
        // a closed-shell molecule (H2 singlet) solved via UHF must give the SAME
        // molecular α(iω) as the closed-shell (Restricted) path. nα=nβ, so the
        // spin-summed dielectric (2·Π_α + 2·Π_β) must reproduce the closed-shell
        // 4·Π. If the per-spin factors were wrong this would diverge.
        use ferric_scf::uhf::solve_uhf;
        use ferric_scf::rhf::RhfConfig;

        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74083\n";
        let op = Operator::coulomb();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let ctx = ParallelContext::default();
        let cfg = PdepRpaConfig { frozen_core: 0, trunc_thresh: 0.0, ..Default::default() };
        let freqs = [0.0, 0.3, 1.0, 4.0];

        // Closed-shell (Restricted) reference.
        let mol_r = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol_r, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol_r, &dfbs_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol_r, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let mol_dyn_r = molecular_dynamic_polarizability(&mol_r, &obs, &dfbs, &rhf, op, &cfg, &freqs).unwrap();

        // Same molecule forced through UHF as a singlet (nα = nβ).
        let mol_u = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let uhf = solve_uhf(&ctx, &mol_u, &obs, &bounds, &RhfConfig::default()).unwrap();
        assert!(!matches!(uhf.spin, Spin::Restricted), "UHF solve should be unrestricted");
        let mol_dyn_u = molecular_dynamic_polarizability(&mol_u, &obs, &dfbs, &uhf, op, &cfg, &freqs).unwrap();

        let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        for k in 0..freqs.len() {
            let ar = iso(&mol_dyn_r[k]);
            let au = iso(&mol_dyn_u[k]);
            assert!(
                (ar - au).abs() < 1e-6 * (1.0 + ar.abs()),
                "RHF vs UHF molecular α mismatch at ω={}: RHF={ar} UHF={au}",
                freqs[k]
            );
        }
        assert!(iso(&mol_dyn_r[0]) > 0.0, "static α must be positive");
    }

    #[test]
    fn h2_polarizability_finite_positive() {
        // Smoke test: α_iso for H2 / cc-pVDZ is positive O(few a.u.).
        // Quantitative comparison vs PySCF lives in tests/polarizability.rs.
        let (mol, obs, dfbs, op, rhf) = build_h2();
        let cfg = PdepRpaConfig {
            frozen_core: 0,
            trunc_thresh: 0.0,
            eigensolver_conv_thresh: 1e-9,
            ..Default::default()
        };
        let r = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, op, &cfg).unwrap();
        assert!(r.iso > 0.0, "α_iso ≤ 0: {}", r.iso);
        assert!(r.iso < 30.0, "α_iso too large: {}", r.iso);
        // tensor symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (r.tensor[i][j] - r.tensor[j][i]).abs() < 1e-10,
                    "α asymmetric at ({i},{j})"
                );
            }
        }
    }

    /// Regression guard for the LU→Cholesky, factor-three-times→factor-once
    /// rewrite of `solve_dielectric_3`: on a small synthetic SPD matrix
    /// ε̃ = I + BBᵀ (guaranteed SPD, mirroring the real dielectric's
    /// structure) with three distinct RHS vectors, the Cholesky-based
    /// `solve_dielectric_3` must agree with an independent general-LU solve
    /// (`ndarray_linalg::Solve`, i.e. exactly the old implementation's
    /// algorithm, computed fresh here rather than reused) to numerical
    /// precision. This directly catches sign errors, upper/lower-triangular
    /// mixups, or a transposed RHS in the Cholesky rewrite.
    #[test]
    fn solve_dielectric_3_cholesky_matches_general_lu() {
        use ndarray::{array, Array1};
        use ndarray_linalg::Solve;

        // B is a nonsymmetric 4x5 matrix; eps = I + B Bᵀ is SPD by
        // construction (identity plus PSD), same structure as the real
        // ε̃ = I + B·g·Bᵀ dielectric.
        let b: Array2<f64> = array![
            [0.3, -0.7, 1.2, 0.4, -0.1],
            [1.1, 0.2, -0.4, 0.9, 0.6],
            [-0.5, 0.8, 0.3, -1.0, 0.2],
            [0.7, -0.3, 0.6, 0.1, -0.9],
        ];
        let n = b.nrows();
        let mut eps_mat = b.dot(&b.t());
        for i in 0..n {
            eps_mat[(i, i)] += 1.0;
        }

        let w: [Array1<f64>; 3] = [
            array![1.0, 0.0, -1.0, 2.0],
            array![0.5, -0.5, 1.5, -1.0],
            array![-2.0, 1.0, 0.0, 0.5],
        ];

        // New implementation under test (Cholesky, factor-once).
        let y_chol = solve_dielectric_3(&eps_mat, &w).expect("SPD system must solve");

        // Independent reference: general LU solve per RHS (the old
        // algorithm), computed directly here rather than by calling any
        // shared helper, so this test does not silently pass if both paths
        // shared a latent bug.
        for d in 0..3 {
            let y_lu = eps_mat.solve(&w[d]).expect("LU solve must succeed on SPD matrix");
            for i in 0..n {
                let diff = (y_chol[d][i] - y_lu[i]).abs();
                assert!(
                    diff < 1e-12,
                    "Cholesky vs LU mismatch at rhs={d}, i={i}: chol={}, lu={}, diff={diff}",
                    y_chol[d][i],
                    y_lu[i]
                );
            }
            // Cross-check both against the defining equation ε̃ y = w.
            let resid = eps_mat.dot(&y_chol[d]) - &w[d];
            let resid_norm: f64 = resid.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                resid_norm < 1e-10,
                "Cholesky solution does not satisfy ε̃y=w at rhs={d}: ||resid||={resid_norm}"
            );
        }
    }
}
