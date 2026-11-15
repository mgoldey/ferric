//! One-shot post-RPA molecular properties for the diffusion-model feature
//! export track — polarizability from PDEP eigenpairs, dielectric-response
//! quantities, and per-atom dynamic α(iω).
//!
//! Currently exposes:
//!
//! * [`crate::properties::pdep_polarizability_static`] — closed-shell static (ω=0) electronic
//!   polarizability tensor α_ij(0) reconstructed from PDEP eigenpairs in the
//!   RI auxiliary basis.
//! * [`crate::properties::pdep_polarizability_becke`], [`crate::properties::pdep_polarizability_becke_dynamic`],
//!   [`crate::properties::pdep_polarizability_hirshfeld`], [`crate::properties::pdep_polarizability_hirshfeld_dynamic`] —
//!   per-atom (static and dynamic) polarizability via Becke/Hirshfeld
//!   partitioning of the PDEP response.
//! * [`crate::properties::molecular_dynamic_polarizability`], [`crate::properties::molecular_dynamic_polarizability_pdep`] —
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
//! and `hirshfeld_charges` — are equally RPA-independent and are defined here.
//! `hirshfeld_i_charges` integrates on the uniform Cartesian lattice of
//! `ferric_integrals::ao_grid::GridSpec`; the other two integrate on the
//! atom-centred Becke–Lebedev grid of `ferric_dft::grid::build_atomic_grid`. Their proatoms come from a caller-supplied
//! [`ProatomProvider`](ferric_scf::properties::ProatomProvider) — normally
//! [`ferric_scf::properties::scf_proatom_provider`] — with the single-Slater
//! proatom of [`slater_xi_for_z`](ferric_scf::properties::slater_xi_for_z) as the
//! fallback.
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
#[cfg(test)]
use ferric_scf::properties::slater_xi_for_z;
use ferric_scf::properties::{debug_toggle, eig3_sym};

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
    electric_field_at_atoms, esp_at_atoms, esp_at_points, lowdin_charges, mulliken_charges,
    resp_charges, spherically_averaged_proatom, ProatomProvider, RadialProatom,
};
// The RPA-independent Hirshfeld volumes and charges live beside their siblings in
// `ferric_scf::properties::hirshfeld_grid`; the paths below stay valid.
pub use ferric_scf::properties::{
    atomic_effective_volumes_hirshfeld, atomic_effective_volumes_hirshfeld_on_grid,
    estimate_hirshfeld_grid_scan_bytes, estimate_hirshfeld_volume_gradient_bytes,
    hirshfeld_charges, hirshfeld_i_charges, hirshfeld_volume_density_derivative,
    hirshfeld_volume_gradient, hirshfeld_volume_grid, hirshfeld_volume_grid_config,
    mbd_volume_grid, nbf_for_basis, preflight_hirshfeld_grid_scan,
    preflight_hirshfeld_grid_scan_reserved, HIRSHFELD_GRAD_CHUNK_POINTS,
    HIRSHFELD_VOLUME_CHUNK_POINTS,
};

mod per_atom;
pub(crate) use per_atom::accumulate_atom_centred_dipoles_pub;
use per_atom::dipole_band_width;
pub use per_atom::{
    charge_transfer_remainder, charge_transfer_remainder_dynamic, dipole_band_width_for_test,
    pdep_polarizability_becke, pdep_polarizability_becke_dynamic, pdep_polarizability_hirshfeld,
    pdep_polarizability_hirshfeld_dynamic,
};
#[cfg(test)]
use per_atom::{BAND_BUDGET_DENOM, BAND_BUDGET_NUMER};

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
/// entirely (`dgetrf` under
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
/// where V_α are the **dressed-basis** PDEP eigenvectors (not the physical
/// aux coefficients `PdepRpaResult::eigenpotentials`). w is built in the
/// dressed basis directly (`B_ov · diag(1/Δε) · μ` — no metric factor ever
/// touches the working vectors), and the dressed eigenvectors come from the
/// PDEP solve run here.
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
    let inter = ferric_mp2::rimp2::compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let b_ov = &inter.b_ov; // shape (naux, nov)
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let nocc_total = inter.nocc_total;
    let first_occ = inter.first_occ;
    let naux = inter.naux;
    let nov = nocc * nvir;
    debug_assert_eq!(b_ov.shape(), &[naux, nov]);

    // Pre-flight before `b_scaled` and `eps_mat` allocate. A public entry
    // point with no ceiling of any kind until now: it is not a grid path, so
    // it fell outside the 2026-07-13 sweep, but it still builds a full
    // `naux x nov` scaled copy plus an `naux x naux` dielectric.
    // `_charge` is bound at FUNCTION scope: the planes it covers (`b_scaled`,
    // `eps_mat`) are allocated below and live to the end of this call.
    let (_budget, _charge) = preflight_molecular_path(
        &format!("pdep_polarizability_static (naux={naux}, nocc={nocc}, nvir={nvir})"),
        cfg.memory_budget_bytes,
        naux,
        nov,
        1,
    )?;

    let eps = rhf.eps_r();
    let eps_occ: Vec<f64> = eps[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + nvir].to_vec();

    // MO-basis dipole μ^d_{ia} = ⟨ψ_i|r_d|ψ_a⟩ from AO dipole + MO transform.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let c = rhf.mos_r();
    let c_occ = c
        .slice(ndarray::s![.., first_occ..first_occ + nocc])
        .to_owned();
    let c_vir = c
        .slice(ndarray::s![.., nocc_total..nocc_total + nvir])
        .to_owned();
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
    let w_vec: [ndarray::Array1<f64>; 3] = std::array::from_fn(|d| b_ov.dot(&mu_flat_inv[d]));

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

    // Pre-flight before `b_scaled` and `eps_mat` allocate. Another public
    // entry point that had no ceiling. It additionally runs an `eigh` on the
    // `naux x naux` dielectric, whose eigenvector output is a second `naux²`
    // buffer co-resident with the input — charged here as the `n_spin = 2`
    // arm's second dielectric-sized term is not, so this is deliberately the
    // conservative side of the estimate.
    let (_budget, _charge) = preflight_molecular_path(
        &format!("dielectric_spectrum_static (naux={naux}, nocc={nocc}, nvir={nvir})"),
        memory_budget_bytes,
        naux,
        nov,
        1,
    )?;

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
    // ROHF/ROKS -> per-spin semi-canonical orbitals and energies (the same view
    // run_u_pdep_rpa uses); UHF borrowed.
    let rhf_view = ferric_scf::semicanonical::unrestricted_reference(mol, rhf)?;
    let rhf: &ScfResult = &rhf_view;

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

    // Pre-flight before the per-spin `b_scaled` copies and the shared
    // `eps_mat` allocate. Ungated until now, and the heaviest of the three
    // molecular entry points: it holds BOTH spin channels' intermediates
    // resident simultaneously, so every `naux x nov` term counts twice.
    let (_budget, _charge) = preflight_molecular_path(
        &format!(
            "pdep_polarizability_static_unrestricted (naux={naux}, nov_a={}, nov_b={})",
            inter_a.nocc * inter_a.nvir,
            inter_b.nocc * inter_b.nvir,
        ),
        cfg.memory_budget_bytes,
        naux,
        (inter_a.nocc * inter_a.nvir).max(inter_b.nocc * inter_b.nvir),
        2,
    )?;

    // Orbital-energy slices per spin.
    let eps_b_full: &[f64] = rhf.eps_b();
    let eps_occ_a: Vec<f64> =
        rhf.eps_a()[inter_a.first_occ..inter_a.first_occ + inter_a.nocc].to_vec();
    let eps_vir_a: Vec<f64> =
        rhf.eps_a()[inter_a.nocc_total..inter_a.nocc_total + inter_a.nvir].to_vec();
    let eps_occ_b: Vec<f64> =
        eps_b_full[inter_b.first_occ..inter_b.first_occ + inter_b.nocc].to_vec();
    let eps_vir_b: Vec<f64> =
        eps_b_full[inter_b.nocc_total..inter_b.nocc_total + inter_b.nvir].to_vec();

    // Per-spin MO-basis dipole μ_σ^d_{ia} = ⟨ψ_iσ|r_d|ψ_aσ⟩.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let c_a = rhf.mos_a();
    let c_b = rhf.mos_b();
    let c_occ_a = c_a
        .slice(ndarray::s![
            ..,
            inter_a.first_occ..inter_a.first_occ + inter_a.nocc
        ])
        .to_owned();
    let c_vir_a = c_a
        .slice(ndarray::s![
            ..,
            inter_a.nocc_total..inter_a.nocc_total + inter_a.nvir
        ])
        .to_owned();
    let c_occ_b = c_b
        .slice(ndarray::s![
            ..,
            inter_b.first_occ..inter_b.first_occ + inter_b.nocc
        ])
        .to_owned();
    let c_vir_b = c_b
        .slice(ndarray::s![
            ..,
            inter_b.nocc_total..inter_b.nocc_total + inter_b.nvir
        ])
        .to_owned();
    let mu_mo_a: [Array2<f64>; 3] =
        std::array::from_fn(|d| c_occ_a.t().dot(&dip_ao[d]).dot(&c_vir_a));
    let mu_mo_b: [Array2<f64>; 3] =
        std::array::from_fn(|d| c_occ_b.t().dot(&dip_ao[d]).dot(&c_vir_b));

    // Per-spin 1/Δε tables and flattened μ vectors.
    let build_flats = |nocc: usize,
                       nvir: usize,
                       eps_o: &[f64],
                       eps_v: &[f64],
                       mu_mo: &[Array2<f64>; 3]|
     -> (
        [ndarray::Array1<f64>; 3],
        [ndarray::Array1<f64>; 3],
        ndarray::Array1<f64>,
    ) {
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
    let w_a: [ndarray::Array1<f64>; 3] =
        std::array::from_fn(|d| inter_a.b_ov.dot(&mu_flat_inv_a[d]));
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
    for p in 0..naux {
        eps_mat[(p, p)] = 1.0;
    }
    for (b_ov, inv_de) in [(&inter_a.b_ov, &inv_de_a), (&inter_b.b_ov, &inv_de_b)] {
        if b_ov.shape()[1] == 0 {
            continue;
        }
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
            let bare_b = if inter_b.nocc == 0 {
                0.0
            } else {
                2.0 * mu_flat_b[i].dot(&mu_flat_inv_b[j])
            };
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

    Ok(PolarizabilityResult {
        tensor,
        iso,
        principal,
    })
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
/// Extra co-resident bytes the open-shell branch of
/// [`pdep_polarizability_becke_dynamic`] holds beyond what
/// `preflight_grid_path` charges when called with `(nocc, nvir)` set to the
/// LARGER spin channel (as that call site does — see the comment there).
///
/// `preflight_grid_path`/`estimate_peak_bytes` charge exactly one resident
/// `naux·nov` buffer (the b_ov term) and one per-worker `naux·nov + naux²`
/// scratch pair. The open-shell frequency loop actually holds BOTH spins'
/// `b_ov` resident (properties.rs builds `inter_a` and `inter_b` and never
/// drops either before the frequency loop runs) and, inside each worker's
/// `map_init` closure, both `b_scaled_a` (naux·nov_a) and `b_scaled_b`
/// (naux·nov_b) simultaneously, PLUS the accumulator `eps_mat` (naux²) and the
/// transient `chi_s` (naux²) that is added into it once per spin — so a
/// second live `naux²` buffer, not one. A prior version of this preflight's
/// comment claimed "the per-worker frequency scratch is sized from one
/// channel at a time"; that is contradicted by the `map_init` seed a few
/// lines below it, which allocates a `(b_scaled_a, b_scaled_b)` PAIR — both
/// buffers are written and read in the same loop body, not sequentially.
///
/// Measured at naux=1000, nov≈20000/spin, 12 workers (8 bytes/f64): charged
/// total ≈2.18 GB, actual resident ≈4.35 GB — almost exactly double, not a
/// rounding term.
///
/// Takes the SMALLER spin's `nov` (the larger one is already counted once by
/// `preflight_grid_path`'s own `nov_max` term), so this is additive, not a
/// double-count: total resident b_ov = `naux*(nov_max+nov_min)`, and the
/// per-worker scratch is `naux*(nov_max+nov_min) + 2*naux²`.
pub fn open_shell_dynamic_extra_bytes(naux: usize, nov_min: usize, n_workers: usize) -> usize {
    const F64_BYTES: usize = 8;
    let n_workers = n_workers.max(1);
    // Second spin's resident b_ov (the first is already charged by the base
    // estimate's `nov_max` term).
    let extra_bov = naux.saturating_mul(nov_min).saturating_mul(F64_BYTES);
    // Per worker: second spin's b_scaled (naux*nov_min) + the second live
    // naux² buffer (chi_s, co-resident with eps_mat during the `+=`).
    let extra_per_worker = naux
        .saturating_mul(nov_min)
        .saturating_add(naux.saturating_mul(naux))
        .saturating_mul(F64_BYTES);
    extra_bov.saturating_add(extra_per_worker.saturating_mul(n_workers))
}

/// Test-only re-export of [`preflight_grid_path`].
///
/// `pub(crate)` is right for production -- no other crate has business
/// building this gate -- but the guard's LIFETIME contract (that the returned
/// `Reservation` keeps the bytes debited until the CALLER drops it) is not
/// observable through any end-to-end path: on `pdep_polarizability_becke` the
/// RI intermediates are built BEFORE this gate runs, so nothing asks the pool
/// again while the guard is held, and dropping the guard early leaves the
/// high-water mark unchanged. Measured: mutation M4b (release early) left all
/// six end-to-end pool tests green.
///
/// A contract no test can see is an assumption, so expose the gate and assert
/// the contract directly: hold the guard, ask for the rest of the pool, and
/// require the refusal. See
/// `tests/mwe_rpa_planes_compose_in_one_pool.rs::
/// the_grid_preflight_guard_keeps_its_bytes_debited_until_the_caller_drops_it`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn preflight_grid_path_for_test(
    label: &str,
    memory_budget_bytes: Option<usize>,
    naux: usize,
    nocc: usize,
    nvir: usize,
    npts: usize,
    nbf: usize,
    natoms: usize,
) -> Result<(usize, ferric_core::memory::pool::Reservation), FerricError> {
    preflight_grid_path(
        label,
        memory_budget_bytes,
        naux,
        nocc,
        nvir,
        npts,
        nbf,
        natoms,
    )
}

/// Returns the resolved budget so callers can reuse it for banding decisions
/// rather than resolving twice (and possibly inconsistently), and an RAII
/// [`Reservation`](ferric_core::memory::pool::Reservation) debiting the shared pool for the projected peak.
///
/// # Why the guard is returned rather than dropped here
///
/// The planes this gate covers (`chi`, the per-atom dipole band partials, the
/// resident `b_ov`) are live in the CALLER, for the whole accumulation — not
/// across this function. A guard dropped at the end of `preflight_grid_path`
/// would credit the bytes back before a single one of them was allocated,
/// which makes the gate decoration: a second plane would then be admitted
/// against bytes this job is about to spend. Callers MUST bind the guard for
/// as long as the grid work runs. `mutation M4` in the migration report is the
/// test that pins this.
///
/// # Hard, not soft
///
/// There is no streaming fallback on this path: `accumulate_atom_centred_
/// dipoles` already chunks the grid, and the width it chunks at is derived
/// from this same budget. A `try_reserve` whose `None` branch cannot narrow
/// anything would be a lie (see the shared brief), so this refuses.
pub(crate) fn preflight_grid_path(
    label: &str,
    memory_budget_bytes: Option<usize>,
    naux: usize,
    nocc: usize,
    nvir: usize,
    npts: usize,
    nbf: usize,
    natoms: usize,
) -> Result<(usize, ferric_core::memory::pool::Reservation), FerricError> {
    let budget = ferric_core::memory::resolve_budget_bytes(memory_budget_bytes);
    // Clamp to the chunk count: the accumulation caps each band at `n_chunks`,
    // so a nominally huge width cannot materialize more partials than there are
    // chunks. Estimating with the unclamped value refuses trivial jobs (see
    // `effective_dipole_band_width`).
    let band =
        crate::budget::effective_dipole_band_width(dipole_band_width(natoms, nbf, budget), npts);
    let est = crate::budget::estimate_peak_bytes(crate::budget::PeakEstimateShape {
        naux,
        nocc,
        nvir,
        // The grid paths run their own frequency loops; n_quad drives wall time
        // rather than peak resident bytes, so the value here is immaterial to
        // the estimate — see `need_inv_dielectric` below, which is the one
        // condition that changes that.
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
        // This preflight guards the grid ACCUMULATION, which consumes the
        // per-frequency dielectric and drops it. Callers that additionally
        // retain `inv_dielectric_freq` (the PDEP dynamic-alpha path) run
        // `run_pdep_rpa` first, and ITS preflight charges the stack from the
        // live config flag — so charging it again here would double-count.
        need_inv_dielectric: false,
        // The AO tensor belongs to the INTERMEDIATES stage, which
        // `run_pdep_rpa`'s preflight already charges from the same budget
        // before this grid accumulation runs. Charging it again here would
        // double-count and refuse property jobs that fit.
        nao: 0,
    });
    ferric_core::memory::check_alloc(label, est, budget)?;
    // Debit the shared ledger for the same `est` the ceiling check just
    // approved. With no pool installed this is an inert guard and the whole
    // call is byte-for-byte what it was before -- the trivial limit pinned by
    // `mwe_rpa_pool_is_inert_without_a_pool.rs`.
    let charge = ferric_core::memory::pool::reserve_global(label, est)?;
    Ok((budget, charge))
}

/// Pre-flight gate for the **Hirshfeld** per-atom property paths.
///
/// Separate from [`preflight_grid_path`] because the two model different
/// grids and different allocations, and conflating them would be the
/// estimator-drift defect all over again:
///
/// * [`preflight_grid_path`] models the **Becke** path, whose `npts` comes from
///   an atom-centred Lebedev grid (grows with atom COUNT) and whose `chi` is
///   evaluated per chunk inside `accumulate_atom_centred_dipoles`.
/// * This models the **Hirshfeld** path, whose `npts` comes from
///   `GridSpec::bounding_box` — a regular real-space lattice, so `npts` grows
///   with molecular VOLUME and is typically far larger — and whose `chi` is a
///   single monolithic `(nbf, npts)` block that is genuinely READ, point by
///   point, for the whole method. It cannot be chunked away the way the Becke
///   path's was; it has to be charged.
///
/// Both Hirshfeld entry points had NO gate whatsoever: `grep` for
/// `check_alloc`/`preflight_grid_path` found the Becke path and nothing else,
/// even though these paths hold strictly MORE memory than the path that was
/// judged dangerous enough to need a gate on 2026-07-13.
///
/// Terms charged (all live simultaneously at the peak — the AO-dipole
/// assembly reads `chi` while `combined` is being filled, and both sit on top
/// of the resident RI intermediates):
///
/// | term | shape | why |
/// |---|---|---|
/// | `b_ov` | `naux · nov` | resident RI intermediate |
/// | `b_scaled` | `naux · nov` | column-scaled copy; a genuine copy, not aliasing (the scaling must not clobber `b_ov`, which is read again) |
/// | `eps_mat` | `naux²` | the dressed dielectric |
/// | `chi` | `nbf · npts` | AO-on-grid, monolithic and genuinely read |
/// | `combined` | `nbf · npts` | per-Cartesian scratch built from `chi`, co-resident with it |
/// | `rho_free` | `natoms · npts` | free-atom densities, one row per atom |
/// | `rho_sum`, `ri_grid` | `4 · npts` | grid side arrays |
///
/// `per_worker` covers the `_dynamic` variant's `map_init` scratch, which
/// seeds a full `b_ov`-shaped `(naux, nov)` buffer AND forms a `(naux, naux)`
/// `eps_mat` inside the rayon closure — the same worker-count-scaled shape as
/// the 2026-07-13 incident. Pass 1 for the static path, which runs serially.
pub(crate) fn preflight_hirshfeld_path(
    label: &str,
    memory_budget_bytes: Option<usize>,
    naux: usize,
    nov: usize,
    npts: usize,
    nbf: usize,
    natoms: usize,
    n_workers: usize,
) -> Result<(usize, ferric_core::memory::pool::Reservation), FerricError> {
    use ferric_core::memory::plan::{Lifetime, MemoryPlan};

    let budget = ferric_core::memory::resolve_budget_bytes(memory_budget_bytes);
    let mut plan = MemoryPlan::with_budget_bytes(budget, label);

    // Resident RI intermediates and the dressed dielectric.
    plan.reserve(
        "b_ov (naux,nov)",
        naux.saturating_mul(nov),
        Lifetime::Resident,
    );
    plan.reserve(
        "b_scaled (naux,nov)",
        naux.saturating_mul(nov),
        Lifetime::Resident,
    );
    plan.reserve(
        "eps_mat (naux,naux)",
        naux.saturating_mul(naux),
        Lifetime::Resident,
    );

    // Grid-side terms. `chi` and `combined` are the two that make this path
    // heavier than the Becke one, and they are co-resident by construction.
    plan.reserve(
        "chi (nbf,npts)",
        nbf.saturating_mul(npts),
        Lifetime::Resident,
    );
    plan.reserve(
        "combined (nbf,npts)",
        nbf.saturating_mul(npts),
        Lifetime::Resident,
    );
    plan.reserve(
        "rho_free (natoms,npts)",
        natoms.saturating_mul(npts),
        Lifetime::Resident,
    );
    plan.reserve(
        "rho_sum + ri_grid (4,npts)",
        npts.saturating_mul(4),
        Lifetime::Resident,
    );

    // The `_dynamic` variant's per-frequency rayon scratch. `n_workers == 1`
    // makes this the static path's (serial) footprint without special-casing.
    if n_workers > 1 {
        plan.reserve_per_worker(
            "per-omega b_scaled + eps_mat",
            naux.saturating_mul(nov)
                .saturating_add(naux.saturating_mul(naux)),
            n_workers,
        );
    }

    // `with_pool` re-bases the ceiling on the pool's REMAINING headroom, so a
    // Hirshfeld grid asked for after the RPA intermediates are already charged
    // sees only what those left. `commit` = `check` + debit + RAII guard; with
    // no pool installed it is `check()` plus an inert guard, i.e. exactly the
    // pre-migration behaviour.
    //
    // HARD: `chi` and `combined` are monolithic `(nbf, npts)` blocks that this
    // path genuinely reads point-by-point for the whole method (see the table
    // above). There is no chunked variant to fall back to, so a soft gate's
    // `None` branch would have nothing to do -- a lie, per the shared brief.
    let charge = match ferric_core::memory::pool::global() {
        Some(pool) => plan.with_pool(&pool).commit()?,
        None => plan.commit()?,
    };
    Ok((budget, charge))
}

/// Pre-flight gate for the **molecular** (non-grid) static property paths:
/// [`pdep_polarizability_static`], [`dielectric_spectrum_static`], and
/// [`pdep_polarizability_static_unrestricted`].
///
/// These are public entry points that touch no grid at all, which is why they
/// escaped the 2026-07-13 grid-path sweep — but "no grid" is not "no memory":
/// each builds a full column-scaled copy of `b_ov` (`naux · nov`) plus an
/// `naux²` dielectric on top of the resident RI intermediates, and each was
/// reachable from Python and the CLI with no ceiling of any kind.
///
/// `n_spin` is 1 for the restricted paths and 2 for the unrestricted one,
/// which builds a complete per-spin intermediate set.
pub(crate) fn preflight_molecular_path(
    label: &str,
    memory_budget_bytes: Option<usize>,
    naux: usize,
    nov: usize,
    n_spin: usize,
) -> Result<(usize, ferric_core::memory::pool::Reservation), FerricError> {
    use ferric_core::memory::plan::{Lifetime, MemoryPlan};

    let budget = ferric_core::memory::resolve_budget_bytes(memory_budget_bytes);
    let mut plan = MemoryPlan::with_budget_bytes(budget, label);
    let n_spin = n_spin.max(1);

    plan.reserve(
        "b_ov (naux,nov) per spin",
        naux.saturating_mul(nov).saturating_mul(n_spin),
        Lifetime::Resident,
    );
    // The column-scaled copy is a real second buffer: `b_ov` is read again
    // after the scaling, so the scaling cannot be done in place.
    plan.reserve(
        "b_scaled (naux,nov) per spin",
        naux.saturating_mul(nov).saturating_mul(n_spin),
        Lifetime::Resident,
    );
    plan.reserve(
        "eps_mat (naux,naux)",
        naux.saturating_mul(naux),
        Lifetime::Resident,
    );

    // HARD, guard returned to the caller: `b_ov` / `b_scaled` / `eps_mat` live
    // in the CALLER for the whole solve, and none of them has a streamed
    // variant (the scaling cannot be in place -- `b_ov` is read again).
    let charge = match ferric_core::memory::pool::global() {
        Some(pool) => plan.with_pool(&pool).commit()?,
        None => plan.commit()?,
    };
    Ok((budget, charge))
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
        // ROHF/ROKS -> per-spin semi-canonical orbitals and energies (the same view
        // run_u_pdep_rpa uses); UHF borrowed.
        let rhf_view = ferric_scf::semicanonical::unrestricted_reference(mol, rhf)?;
        let rhf: &ScfResult = &rhf_view;
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

        // Pre-flight: this open-shell branch had NO gate at all (unlike its
        // per-atom sibling `pdep_polarizability_becke_dynamic`, which at
        // least had a single-spin `preflight_grid_path` call). It holds both
        // spins' `b_ov` resident plus, per rayon worker, both `b_scaled_{a,b}`
        // (naux·nov_σ each) and TWO live naux² buffers (`eps_mat` and the
        // `b_scaled.dot(&b_scaled.t())` temporary added into it via `+=` a few
        // lines below) — the same two-spin/two-naux² shape documented on
        // `open_shell_dynamic_extra_bytes`. `preflight_molecular_path`'s
        // `n_spin` multiplies both the b_ov and b_scaled terms uniformly, so
        // pass the LARGER spin's `nov` (conservative, over- not
        // under-estimating for asymmetric α/β spaces) and add the second
        // naux² buffer the helper does not model.
        let nov_a = inter_a.nocc.saturating_mul(inter_a.nvir);
        let nov_b = inter_b.nocc.saturating_mul(inter_b.nvir);
        let (mol_budget, _charge) = preflight_molecular_path(
            &format!(
                "molecular_dynamic_polarizability (U) (naux={naux}, nov_a={nov_a}, nov_b={nov_b})"
            ),
            cfg.memory_budget_bytes,
            naux,
            nov_a.max(nov_b),
            2,
        )?;
        ferric_core::memory::check_alloc(
            &format!(
                "molecular_dynamic_polarizability (U) second naux² dielectric buffer (naux={naux})"
            ),
            naux.saturating_mul(naux).saturating_mul(8),
            mol_budget,
        )?;

        // Orbital-energy slices per spin.
        let eps_b_full: &[f64] = rhf.eps_b();
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
        let c_b = rhf.mos_b();
        let mk_mu = |inter: &ferric_mp2::rimp2::RpaIntermediates, c: &Array2<f64>| {
            let c_occ = c
                .slice(ndarray::s![
                    ..,
                    inter.first_occ..inter.first_occ + inter.nocc
                ])
                .to_owned();
            let c_vir = c
                .slice(ndarray::s![
                    ..,
                    inter.nocc_total..inter.nocc_total + inter.nvir
                ])
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
                    || {
                        (
                            Array2::<f64>::zeros(inter_a.b_ov.raw_dim()),
                            Array2::<f64>::zeros(inter_b.b_ov.raw_dim()),
                        )
                    },
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

    // Pre-flight: same shape as `pdep_polarizability_static` (resident b_ov +
    // per-worker b_scaled + eps_mat), but this closed-shell branch had no gate
    // of any kind until now — it is not a grid path, so the 2026-07-13 sweep
    // (which only covered `properties.rs`'s grid-based per-atom paths) missed
    // it, same as its open-shell sibling above.
    let (_budget, _charge) = preflight_molecular_path(
        &format!("molecular_dynamic_polarizability (naux={naux}, nocc={nocc}, nvir={nvir})"),
        cfg.memory_budget_bytes,
        naux,
        nov,
        1,
    )?;

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
    let c_occ = c
        .slice(ndarray::s![.., first_occ..first_occ + nocc])
        .to_owned();
    let c_vir = c
        .slice(ndarray::s![.., nocc_total..nocc_total + nvir])
        .to_owned();
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
                        b_scaled
                            .column_mut(ia)
                            .mapv_inplace(|x| x * (4.0 * g[ia]).sqrt());
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
///   W̃_k = inv_dielectric_freq\[k\] = ε̃_proj⁻¹ − I.
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
             (Laplace χ₀ path unsupported)"
                .into(),
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
    let c_occ = c
        .slice(ndarray::s![.., first_occ..first_occ + nocc])
        .to_owned();
    let c_vir = c
        .slice(ndarray::s![.., nocc_total..nocc_total + nvir])
        .to_owned();
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

#[cfg(test)]
mod tests {

    /// EXACTNESS ANCHOR for the Becke–Lebedev volume quadrature of
    /// [`atomic_effective_volumes_hirshfeld`], written BEFORE any accuracy
    /// sweep was run (CLAUDE.md Experimental Protocol: "EXACTNESS ANCHOR
    /// FIRST").
    ///
    /// The trivial limit of the Hirshfeld partition is ONE atom: `w^A =
    /// ρ⁰_A/(ρ⁰_A + 1e-12) = 1` everywhere ρ⁰_A ≫ 1e-12, so the volume
    /// collapses to the plain moment `∫ ρ r³ dV` with no partition left in it.
    /// Feeding the single-Slater proatom `ρ = Z ξ³/π e^{−2ξr}` as the density
    /// makes that moment analytic:
    /// ```text
    ///   ∫ ρ r³ dV = (Z ξ³/π)·4π·∫₀^∞ r⁵ e^{−2ξr} dr
    ///             = 4 Z ξ³ · 5!/(2ξ)⁶ = 7.5 Z / ξ³
    /// ```
    /// This exercises exactly the pieces an implementation error would break —
    /// the grid weights, the `r³` factor, and the `Σ_B ρ⁰_B` denominator — with
    /// no AO evaluation and no SCF in the way. It is deliberately NOT a
    /// comparison against another of ferric's own quadratures: a construction
    /// bug reproduces across quadratures (CLAUDE.md "CONSISTENCY IS NOT
    /// CORROBORATION"), an analytic value does not.
    ///
    /// The quadrature is the production one:
    /// `build_atomic_grid(&AtomicGridConfig::default())`, the 75×110 TA-M4 ×
    /// Lebedev grid, with the home-atom Becke factor folded into each weight.
    ///
    /// Bar DERIVED FROM MEASUREMENT, not guessed: the max relative deviation
    /// over Z ∈ {1, 6, 8} is 8.2e-7 (H 2.8e-7, C 8.2e-7, O 5.2e-7). That floor
    /// is the 75-point Treutler–Ahlrichs M4 radial rule against an
    /// exponential's cusp — the grid is tuned for a contracted Gaussian
    /// density, and a Slater is a harder radial integrand than anything a
    /// basis set produces — so it is the quadrature's accuracy, not an error.
    /// The bar is 1e-5, ~12x that.
    ///
    /// The bar is checked from the other side too: replacing this loop's
    /// `r * r * r` with `r * r` puts the sum 15% (Z=1), 70% (Z=6) and 65%
    /// (Z=8) off — four to five orders outside the bar.
    ///
    /// Two of the three mutations the issue asks for are INVISIBLE to a
    /// one-atom anchor, by construction, and that is recorded here rather than
    /// papered over:
    ///
    /// * the proatom of atom 0 used for every atom — there is only atom 0, so
    ///   the mutation is the identity;
    /// * the home-atom Becke factor not multiplied into the grid weight — on
    ///   one atom that factor IS 1 (`becke_weights_all` returns `[1.0]`), so
    ///   again the identity.
    ///
    /// Both are carried by the molecular rows of
    /// `tests/validation_hirshfeld.rs`; see that file's MUTATION ledger for
    /// the run log. An anchor cannot see a defect proportional to what it
    /// zeroes (memory `anchor-blind-spots`), and the trivial limit that makes
    /// this test exact is exactly what zeroes those two.
    #[test]
    fn hirshfeld_volume_quadrature_matches_the_closed_form_slater_moment() {
        use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};

        /// Measured max 8.2e-7 (C); 12x headroom. See the doc above for the
        /// mutations that must — and do — exceed it.
        const BAR: f64 = 1e-5;
        let mut worst = 0.0_f64;
        for z in [1_i32, 6, 8] {
            let sym = ferric_core::elements::z_to_symbol(z).expect("symbol");
            // Multiplicity only has to be CONSISTENT with Z here (the grid
            // depends on the nuclear charge and position, not on the state),
            // so take the parity-legal minimum: a singlet for even Z, a
            // doublet for odd.
            let mult = if z % 2 == 0 { 1 } else { 2 };
            let mol = Molecule::parse_xyz(&format!("1\n{sym}\n{sym} 0 0 0\n"), 0, mult)
                .expect("one-atom molecule");
            let xi = slater_xi_for_z(z);
            let prefac = z as f64 * xi.powi(3) / std::f64::consts::PI;
            let grid = build_atomic_grid(&mol, &AtomicGridConfig::default());
            // Σ_g w_g · ρ(r_g) · w^A(r_g) · r_g³ with ρ = ρ⁰ and w^A = 1 by
            // construction: the one-atom Hirshfeld weight, written out rather
            // than assumed, so a weight that was NOT 1 would show up here.
            let mut acc = 0.0_f64;
            let mut worst_weight_dev = 0.0_f64;
            for g in &grid {
                let r = (g.xyz[0] * g.xyz[0] + g.xyz[1] * g.xyz[1] + g.xyz[2] * g.xyz[2]).sqrt();
                let rho0 = prefac * (-2.0 * xi * r).exp();
                let w_hirsh = rho0 / (rho0 + 1e-12);
                // Only where ρ⁰ is above the 1e-12 floor is w == 1; the tail
                // contributes nothing to the moment either way.
                if rho0 > 1e-8 {
                    worst_weight_dev = worst_weight_dev.max((w_hirsh - 1.0).abs());
                }
                acc += g.weight * rho0 * w_hirsh * r * r * r;
            }
            assert!(
                worst_weight_dev < 1e-4,
                "Z={z}: one-atom Hirshfeld weight deviates from 1 by {worst_weight_dev:.2e} \
                 where ρ⁰ > 1e-8 — the anchor's premise is broken, not the quadrature"
            );
            let exact = 7.5 * z as f64 / xi.powi(3);
            let rel = (acc - exact).abs() / exact;
            eprintln!("Z={z} xi={xi:.4}: quadrature {acc:.12e} exact {exact:.12e} rel {rel:.2e}");
            worst = worst.max(rel);
        }
        assert!(
            worst < BAR,
            "Becke-Lebedev volume quadrature vs the closed-form Slater r³ moment: \
             max rel {worst:.2e} >= {BAR:.0e}"
        );
    }
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
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(6)
            .build()
            .unwrap();
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
            // The live set fits the budget outright: no worker-count floor can
            // push it above, because the byte cap wins and parallelism degrades
            // instead,
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
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            // natoms=1, nbf=100: per_partial = 1*3*100*100*8 = 240_000 bytes.
            //
            // The band gets BAND_BUDGET_NUMER/BAND_BUDGET_DENOM of the budget,
            // not all of it, so a 2_400_000-byte budget yields
            // (2_400_000/4)/240_000 = 2 partials in flight.
            //
            // The band is a FRACTION of the budget, so this assertion is stated
            // against the constants themselves and tracks the implementation
            // instead of encoding one value of it.
            let per_partial = 3 * 100 * 100 * std::mem::size_of::<f64>();
            let budget = 2_400_000usize;
            let expected = (budget / BAND_BUDGET_DENOM * BAND_BUDGET_NUMER) / per_partial;
            assert_eq!(expected, 2, "arithmetic guard: the worked example is 2");
            assert_eq!(dipole_band_width(1, 100, budget), expected);
            // A budget below one partial floors at ONE, not at the worker count.
            //
            // A `.max(current_num_threads())` floor (an expected 2 = the pool's worker
            // count) would override the byte cap and let resident memory scale
            // with core count instead of with the budget — the mechanism behind
            // the 16-17 GB anon-RSS incidents. The byte cap wins and
            // parallelism degrades instead, so a starvation budget yields a
            // single chunk in flight: slow, never stuck, and never over budget.
            assert_eq!(dipole_band_width(1, 100, 1), 1);
            assert_eq!(dipole_band_width(0, 0, 1), 1);

            // The floor must NOT depend on the ambient pool size: same budget,
            // same answer, whatever the worker count. (Regression guard for the
            // exact defect above — see tests/mwe_budget_respected.rs.)
            let wide = rayon::ThreadPoolBuilder::new()
                .num_threads(8)
                .build()
                .unwrap();
            wide.install(|| {
                assert_eq!(dipole_band_width(1, 100, 1), 1);
                assert_eq!(dipole_band_width(1, 100, budget), expected);
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
        let xyz =
            "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
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
                scheme: QuadratureScheme::GaussLegendre,
                n_points: 12,
                u0: 0.5,
            },
            trunc_thresh: 0.0,
            eigensolver_conv_thresh: 1e-9,
            // The PDEP dynamic-α property path reads inv_dielectric_freq (M9 gate).
            need_inv_dielectric_freq: true,
            need_eigenvalues_freq: true,
            ..Default::default()
        };
        let rpa = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();

        let full =
            molecular_dynamic_polarizability(&mol, &obs, &dfbs, &rhf, op, &cfg, &rpa.quad_freqs)
                .unwrap();
        let pdep = molecular_dynamic_polarizability_pdep(
            &rpa,
            &mol,
            &obs,
            &dfbs,
            &rhf,
            op,
            cfg.memory_budget_bytes,
        )
        .unwrap();

        assert_eq!(full.len(), pdep.len());
        let mut max_abs = 0.0_f64;
        for k in 0..full.len() {
            for i in 0..3 {
                for j in 0..3 {
                    max_abs = max_abs.max((full[k][i][j] - pdep[k][i][j]).abs());
                }
            }
        }
        assert!(
            max_abs < 1e-8,
            "untruncated PDEP α deviates from full by {max_abs:.3e}"
        );
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
        // The atom sum Σ_A α^A(iω) of the intrinsic per-atom tensors decays
        // monotonically with the right tail.
        let (mol, obs, dfbs, op, rhf) = build_h2();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let cfg = PdepRpaConfig {
            frozen_core: 0,
            trunc_thresh: 0.0,
            ..Default::default()
        };
        let freqs = [0.0, 0.5, 2.0, 10.0];
        let dyn_a =
            pdep_polarizability_becke_dynamic(&mol, &obs, &bs, &dfbs, &rhf, op, &cfg, &freqs)
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
        assert!(
            a0 > a1 && a1 > a2 && a2 > a3,
            "not monotone: {a0} {a1} {a2} {a3}"
        );
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

        let dynamic =
            pdep_polarizability_becke_dynamic(&mol, &obs, &bs, &dfbs, &rhf, op, &cfg, &[0.0, 0.5])
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
        let dyn_a =
            pdep_polarizability_becke_dynamic(&mol, &obs, &bs, &dfbs, &rhf, op, &cfg, &freqs)
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
        assert!(
            a0 > a1 && a1 > a2 && a2 > a3,
            "not monotone: {a0} {a1} {a2} {a3}"
        );
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
        let dyn_a =
            pdep_polarizability_becke_dynamic(&mol, &obs, &bs, &dfbs, &uhf, op, &cfg, &freqs)
                .unwrap();
        assert_eq!(dyn_a.len(), 1);

        let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        let a0 = iso(&dyn_a[0][0]);
        let a1 = iso(&dyn_a[0][1]);
        let a2 = iso(&dyn_a[0][2]);
        let a3 = iso(&dyn_a[0][3]);
        assert!(a0 > 0.0, "α(0) not positive: {a0}");
        assert!(
            a0 > a1 && a1 > a2 && a2 > a3,
            "not monotone: {a0} {a1} {a2} {a3}"
        );
        assert!(a3 < 0.05 * a0, "tail too large: α(10)={a3} α(0)={a0}");
    }

    #[test]
    fn molecular_dynamic_alpha_uhf_matches_rhf_on_closed_shell() {
        // Rigor check for the open-shell molecular_dynamic_polarizability branch:
        // a closed-shell molecule (H2 singlet) solved via UHF must give the SAME
        // molecular α(iω) as the closed-shell (Restricted) path. nα=nβ, so the
        // spin-summed dielectric (2·Π_α + 2·Π_β) must reproduce the closed-shell
        // 4·Π. If the per-spin factors were wrong this would diverge.
        use ferric_scf::rhf::RhfConfig;
        use ferric_scf::uhf::solve_uhf;

        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74083\n";
        let op = Operator::coulomb();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let ctx = ParallelContext::default();
        let cfg = PdepRpaConfig {
            frozen_core: 0,
            trunc_thresh: 0.0,
            ..Default::default()
        };
        let freqs = [0.0, 0.3, 1.0, 4.0];

        // Closed-shell (Restricted) reference.
        let mol_r = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol_r, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol_r, &dfbs_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol_r, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let mol_dyn_r =
            molecular_dynamic_polarizability(&mol_r, &obs, &dfbs, &rhf, op, &cfg, &freqs).unwrap();

        // Same molecule forced through UHF as a singlet (nα = nβ).
        let mol_u = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let uhf = solve_uhf(&ctx, &mol_u, &obs, &bounds, &RhfConfig::default()).unwrap();
        assert!(
            !matches!(uhf.spin, Spin::Restricted),
            "UHF solve should be unrestricted"
        );
        let mol_dyn_u =
            molecular_dynamic_polarizability(&mol_u, &obs, &dfbs, &uhf, op, &cfg, &freqs).unwrap();

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
    fn becke_dynamic_at_zero_frequency_is_the_static_per_atom_alpha() {
        // Exactness anchor: one Krishtal intrinsic definition, two code paths.
        // The dynamic path evaluated AT ω = 0 must reproduce the static one,
        // and α_CT closes Σ_A α^A onto the molecular static α.
        let (mol, obs, dfbs, op, rhf) = build_h2();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let cfg = PdepRpaConfig::default();
        let dyn0 =
            pdep_polarizability_becke_dynamic(&mol, &obs, &bs, &dfbs, &rhf, op, &cfg, &[0.0])
                .unwrap();
        let st = pdep_polarizability_becke(&mol, &obs, &bs, &dfbs, &rhf, op, &cfg).unwrap();
        let scale = st
            .iter()
            .flatten()
            .flatten()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut d = 0.0_f64;
        for a in 0..st.len() {
            for i in 0..3 {
                for j in 0..3 {
                    d = d.max((dyn0[a][0][i][j] - st[a][i][j]).abs());
                }
            }
        }
        assert!(
            d / scale < 1e-11,
            "per-atom dynamic(ω=0) vs static: rel {:.2e}",
            d / scale
        );

        let mol_a = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, op, &cfg)
            .unwrap()
            .tensor;
        let ct = charge_transfer_remainder(&mol_a, &st);
        // H2 along z: charge flows only along the bond, so α_CT is zz-dominated
        // and the remainder is a real (non-negligible) fraction of α_zz.
        assert!(
            ct[2][2] > 0.05 * mol_a[2][2],
            "α_CT,zz = {} vs α_zz = {}",
            ct[2][2],
            mol_a[2][2]
        );
        assert!(
            ct[0][0].abs() < 1e-3 * mol_a[0][0],
            "α_CT,xx = {} should vanish (no charge flow perpendicular to the bond)",
            ct[0][0]
        );
        for i in 0..3 {
            for j in 0..3 {
                let s_ij: f64 = st.iter().map(|t| t[i][j]).sum();
                assert!(
                    (s_ij + ct[i][j] - mol_a[i][j]).abs() < 1e-13 * mol_a[2][2],
                    "Σ_A α^A + α_CT != α_mol at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn becke_dynamic_uhf_matches_rhf_per_atom_on_closed_shell() {
        // The open-shell (U) branch's ket is the analytic dipole per spin, the
        // same operator as the closed-shell branch. A singlet forced through
        // UHF must therefore give the SAME per-atom α^A(iω).
        use ferric_scf::uhf::solve_uhf;
        let (mol, obs, dfbs, op, rhf) = build_h2();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let cfg = PdepRpaConfig::default();
        let freqs = [0.0, 0.7];
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &RhfConfig::default()).unwrap();
        assert!(!matches!(uhf.spin, Spin::Restricted));
        let r = pdep_polarizability_becke_dynamic(&mol, &obs, &bs, &dfbs, &rhf, op, &cfg, &freqs)
            .unwrap();
        let u = pdep_polarizability_becke_dynamic(&mol, &obs, &bs, &dfbs, &uhf, op, &cfg, &freqs)
            .unwrap();
        for a in 0..r.len() {
            for k in 0..freqs.len() {
                for i in 0..3 {
                    for j in 0..3 {
                        let (x, y) = (r[a][k][i][j], u[a][k][i][j]);
                        assert!(
                            (x - y).abs() < 1e-6 * (1.0 + x.abs()),
                            "atom {a} ω={} [{i}][{j}]: RHF {x} vs UHF {y}",
                            freqs[k]
                        );
                    }
                }
            }
        }
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
    /// (`ndarray_linalg::Solve`, i.e. a general-LU
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
            let y_lu = eps_mat
                .solve(&w[d])
                .expect("LU solve must succeed on SPD matrix");
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
