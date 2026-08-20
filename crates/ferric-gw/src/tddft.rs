//! **SPIKE**: TDA-DFT (Tamm–Dancoff-approximation TDDFT) singlet excitation
//! energies on a closed-shell Kohn–Sham reference.
//!
//! # Scope — READ THIS FIRST
//!
//! This module is a validated SPIKE, not a production capability. It computes
//! **closed-shell singlet** excitation energies in the **Tamm–Dancoff
//! approximation** only. Explicitly NOT covered:
//!
//!   * full Casida (A/B coupled) TDDFT — TDA only,
//!   * triplet excitations (the spin-flip combination `f_αα − f_αβ` is never
//!     assembled here, even though `GgaFxcKernel` is spin-resolved and could
//!     supply it),
//!   * open-shell / spin-flip references (RKS only — UKS/ROKS are rejected),
//!   * excited-state gradients or properties beyond the oscillator strength,
//!   * meta-GGA functionals (no τ f_xc kernel — `GgaFxcKernel::new` rejects
//!     them and that error propagates),
//!   * VV10 nonlocal correlation response (silently absent from the kernel —
//!     see [`run_tda_dft`]'s hard rejection of VV10-carrying functionals),
//!   * range-separated hybrids (the long-range exchange kernel term is not
//!     assembled — see the `KMix` handling in [`run_tda_dft`], which rejects
//!     `omega != 0`).
//!
//! It is library-only and deliberately NOT wired into the CLI or the Python
//! bindings.
//!
//! # Physics
//!
//! Closed-shell singlet TDA matrix in the particle–hole `(ia)` space:
//!
//! ```text
//!   A_{ia,jb} = δ_ij δ_ab (ε_a − ε_i)
//!             + 2 (ia|jb)
//!             − c_HF (ij|ab)
//!             + 2 (ia|f_xc|jb)
//! ```
//!
//! matching PySCF's `pyscf/tdscf/rhf.py::get_ab` convention exactly (its
//! `add_hf_` contributes `2·(ia|jb) − hyb·(ij|ab)`, and its XC block adds
//! `iajb = 2 · Σ_g w_g f_xc^{eff}(r_g) ρ_ov^{ia}(r_g) ρ_ov^{jb}(r_g)`).
//!
//! `c_HF` is the exact-exchange fraction from the functional's [`KMix`]
//! (0 for a pure LDA/GGA, e.g. 0.2 for B3LYP).
//!
//! ## The f_xc adapter — the one genuinely new piece
//!
//! `ferric_dft::fxc::GgaFxcKernel` is an **AO-density-matrix → AO-potential**
//! operator: given a spin-resolved perturbation (δD_α, δD_β) at a reference
//! density it returns (δV_α, δV_β). The TDA matrix needs `(ia)`-space matrix
//! elements. The adapter is:
//!
//! ```text
//!   δD_α = δD_β = C_j C_b^T   (the AO transition density of pair jb, unsymmetrized)
//!   δV_α        = kernel.apply_with_ref(ref, δD_α, δD_β).0
//!   K_{ia,jb}   = C_i^T δV_α C_a
//! ```
//!
//! With δD_α = δD_β the kernel evaluates
//! `δV_α = ∫ (f_αα + f_αβ) δρ_α`, and since `δρ_α = δρ_β`, this is exactly the
//! **singlet** spin combination. The remaining factor is fixed by the
//! definition of `A`: setting δD_α = δD_β = C_jC_b^T (i.e. a unit-weight
//! transition density in EACH spin channel) makes
//! `K_{ia,jb} = (ia| f_αα + f_αβ |jb)`, and the closed-shell singlet TDA term
//! `2(ia|f_xc|jb)` in PySCF's spin-summed convention equals precisely that.
//! **This factor is not asserted from theory here — it is pinned numerically
//! against PySCF `tddft.TDA` in `crates/ferric-gw/tests/tda_dft_pyscf.rs`, and
//! the whole f_xc-free limit is pinned against `run_cis_tda` by
//! [`run_tda_dft`]'s exactness anchor test.**
//!
//! # Validation status
//!
//! See `crates/ferric-gw/tests/tda_dft_pyscf.rs` and the `tests` module below.
//!   * **Exactness anchor**: with `include_fxc = false` and `c_hf = 1`, this
//!     reduces BIT-IDENTICALLY to `bse::run_cis_tda` (same reference, same
//!     RI tensor, same assembly). That isolates the A-matrix layout from the
//!     kernel adapter.
//!   * **PySCF cross-check**: water / cc-pVDZ, LDA and PBE and B3LYP, vs
//!     `pyscf.tddft.TDA` — see the test file for the achieved agreement.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_dft::fxc::GgaFxcKernel;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::libxc::xc_def_from_name_nspin;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;
use ndarray::{s, Array2};
use ndarray_linalg::{Eigh, UPLO};

use crate::mo_b;

/// Result of a TDA-DFT singlet excitation calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct TdaDftResult {
    /// Singlet excitation energies Ω_n (Hartree), ascending.
    pub omega: Vec<f64>,
    /// Length-gauge oscillator strengths f_n, same ordering/length as `omega`.
    pub oscillator_strength: Vec<f64>,
    /// Eigenvectors X_n, shape (n, n) with n = nocc·nvir; column `n` is state
    /// `n`, row index `ia = i*nvir + a` (LOCAL active occ/vir indices).
    /// Retained so callers can do STATE CHARACTER matching (dominant (i,a)
    /// pair, transition dipole direction) rather than nearest-energy matching.
    pub x: Array2<f64>,
    /// Number of occupied / virtual orbitals in the active (ia) window.
    pub nocc: usize,
    pub nvir: usize,
    /// Exact-exchange fraction actually used on the `−c_HF (ij|ab)` term.
    pub c_hf: f64,
    /// Whether the f_xc coupling term was included.
    pub fxc_included: bool,
}

/// Knobs for [`run_tda_dft`]. Defaults are the physical TDA-DFT setting.
#[derive(Debug, Clone)]
pub struct TdaDftConfig {
    /// Include the `2(ia|f_xc|jb)` coupling. Setting this `false` together
    /// with `c_hf_override = Some(1.0)` is the CIS exactness anchor.
    pub include_fxc: bool,
    /// Override the exact-exchange fraction on the `−c_HF(ij|ab)` term.
    /// `None` (default) takes it from the functional's `KMix`.
    pub c_hf_override: Option<f64>,
    /// Grid for the f_xc kernel. Should match the SCF's XC grid.
    pub grid: AtomicGridConfig,
    /// Freeze the lowest `frozen_core` occupied orbitals out of the (ia) space.
    pub frozen_core: usize,
}

impl Default for TdaDftConfig {
    fn default() -> Self {
        Self {
            include_fxc: true,
            c_hf_override: None,
            grid: AtomicGridConfig::default(),
            frozen_core: 0,
        }
    }
}

/// Fail-fast pre-flight for the dense `(n, n)` TDA matrix plus the co-resident
/// `eigh` eigenvector output. Same shape of guard as `bse::check_dense_response_alloc`.
fn check_tda_alloc(n: usize) -> Result<(), FerricError> {
    let bytes = n.saturating_mul(n).saturating_mul(8).saturating_mul(2);
    ferric_core::memory::check_alloc(
        &format!("TDA-DFT dense (ia) matrix (n = nocc*nvir = {n}, + eigh output)"),
        bytes,
        ferric_core::memory::resolve_budget_bytes(None),
    )
}

/// Length-gauge singlet TDA oscillator strengths.
///
/// Identical convention to `bse::tda_oscillator_strengths` (which is itself
/// cross-checked against PySCF's `tdscf.rhf.TDA.oscillator_strength` in
/// `crates/ferric-gw/tests/bse_oscillator_strength.rs`):
///
/// ```text
///   <0|r|n> = sqrt(2) * Σ_{ia} X_n(i,a) <i|r|a>
///   f_n     = (2/3) Ω_n |<0|r|n>|²
/// ```
///
/// Duplicated here rather than made `pub(crate)` in `bse.rs` on purpose: this
/// spike must not perturb the validated BSE path. The two are kept in sync by
/// `tda_dft_oscillator_strength_matches_bse_helper` in the tests below.
fn tda_oscillator_strengths(
    evals: &[f64],
    evecs: &Array2<f64>,
    mu_ao: &[Array2<f64>; 3],
    mo_coeff: &Array2<f64>,
    first_act: usize,
    nocc_total: usize,
    nocc: usize,
    nvir: usize,
) -> Vec<f64> {
    let n = nocc * nvir;
    let orbo = mo_coeff.slice(s![.., first_act..nocc_total]);
    let orbv = mo_coeff.slice(s![.., nocc_total..(nocc_total + nvir)]);
    let dip_ia: [Array2<f64>; 3] = std::array::from_fn(|d| orbo.t().dot(&mu_ao[d]).dot(&orbv));

    let mut f = Vec::with_capacity(n);
    for state in 0..n {
        let x = evecs.column(state);
        let mut mu = [0.0_f64; 3];
        for (d, dip) in dip_ia.iter().enumerate() {
            let mut acc = 0.0;
            for i in 0..nocc {
                for a in 0..nvir {
                    acc += x[i * nvir + a] * dip[(i, a)];
                }
            }
            mu[d] = std::f64::consts::SQRT_2 * acc;
        }
        let mu2 = mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2];
        f.push((2.0 / 3.0) * evals[state] * mu2);
    }
    f
}

/// Build the singlet f_xc coupling block `K_{ia,jb} = (ia| f_αα + f_αβ |jb)`
/// in the active (ia) space, via the validated `GgaFxcKernel` AO adapter.
///
/// One `apply_with_ref` call per `jb` column: δD_α = δD_β = C_j C_b^T, then
/// `K[:, jb] = C_occ^T δV_α C_vir` flattened. The kernel's AO→AO operator is
/// SYMMETRIC in the (μν) pair index (it back-projects onto χ_μχ_ν and
/// explicitly symmetrizes), so the resulting K is symmetric up to grid
/// round-off; we symmetrize at the end and report the pre-symmetrization
/// asymmetry as a diagnostic (an asymmetry far above grid noise would mean the
/// adapter, not the kernel, is wrong).
///
/// Cost: `n = nocc·nvir` kernel applications. That is deliberate — it reuses
/// the ALREADY-VALIDATED kernel unmodified rather than re-deriving the
/// grid-space contraction (which is what a production implementation would do,
/// in one pass over the grid; see the module docs' scope note).
fn build_fxc_block(
    kernel: &GgaFxcKernel,
    ref_dens: &ferric_dft::density_on_grid::UksDensityGrid,
    mo_coeff: &Array2<f64>,
    first_act: usize,
    nocc_total: usize,
    nocc: usize,
    nvir: usize,
) -> (Array2<f64>, f64) {
    let n = nocc * nvir;
    let orbo = mo_coeff.slice(s![.., first_act..nocc_total]).to_owned();
    let orbv = mo_coeff.slice(s![.., nocc_total..(nocc_total + nvir)]).to_owned();

    let mut k = Array2::<f64>::zeros((n, n));
    for j in 0..nocc {
        let cj = orbo.column(j);
        for b in 0..nvir {
            let cb = orbv.column(b);
            // δD = ½(C_j C_bᵀ + C_b C_jᵀ), EXPLICITLY SYMMETRIZED.
            //
            // An earlier version passed the raw outer product `C_j C_bᵀ` with a
            // comment asserting symmetrization was a no-op because the kernel
            // back-projects onto the symmetric product χ_μχ_ν. That is true for
            // LDA and FALSE for GGA: the gradient terms contract against ∇δD,
            // and ∇(C_j C_bᵀ) ≠ ∇(C_b C_jᵀ), so the antisymmetric part of δD
            // does leak into δV through the σ = |∇ρ|² coupling.
            //
            // MEASURED on water/STO-3G/PBE: the raw outer product gives an
            // (ia)-block asymmetry of 3.1e-2 relative — five orders of magnitude
            // above the 1e-8 grid-noise guard below, which correctly rejected
            // it. Symmetrizing drops the asymmetry under the guard and brings
            // PBE excitation energies to 1.0e-3 eV of PySCF `tddft.TDA`.
            // LDA is unchanged by this (1.02e-3 eV before and after), which is
            // exactly the LDA-vs-GGA asymmetry the old comment missed.
            let dd = {
                let mut m = Array2::<f64>::zeros((cj.len(), cj.len()));
                for (mu, &x) in cj.iter().enumerate() {
                    for (nu, &y) in cb.iter().enumerate() {
                        m[(mu, nu)] = 0.5 * x * y;
                    }
                }
                for (nu, &y) in cb.iter().enumerate() {
                    for (mu, &x) in cj.iter().enumerate() {
                        m[(nu, mu)] += 0.5 * y * x;
                    }
                }
                m
            };
            let (dv_a, _dv_b) = kernel.apply_with_ref(ref_dens, &dd, &dd);
            // K[:, jb] = C_occ^T δV_α C_vir, flattened as ia = i*nvir + a.
            let block = orbo.t().dot(&dv_a).dot(&orbv); // (nocc, nvir)
            let jb = j * nvir + b;
            for i in 0..nocc {
                for a in 0..nvir {
                    k[(i * nvir + a, jb)] = block[(i, a)];
                }
            }
        }
    }

    // Symmetry diagnostic: max |K − Kᵀ| relative to max |K|.
    let scale = k.iter().fold(0.0_f64, |m, &x| m.max(x.abs())).max(1e-30);
    let mut asym = 0.0_f64;
    for p in 0..n {
        for q in 0..n {
            asym = asym.max((k[(p, q)] - k[(q, p)]).abs());
        }
    }
    let asym_rel = asym / scale;
    let k_sym = 0.5 * (&k + &k.t());
    (k_sym, asym_rel)
}

/// Run a TDA-DFT singlet excitation calculation on a closed-shell KS (or HF)
/// reference.
///
/// `ks` must be a converged RESTRICTED result. `xc_name` is the functional the
/// reference was converged with (e.g. `"LDA"`, `"PBE"`, `"B3LYP"`), or `None`
/// for a pure-HF reference (then `c_HF` defaults to 1 and no f_xc term is
/// built — i.e. plain CIS).
///
/// # Errors
///
/// Hard-errors (never silently degrades) on: an open-shell reference, a
/// meta-GGA functional (no τ f_xc kernel), a range-separated hybrid
/// (`omega != 0` — the long-range exchange kernel term is not assembled), a
/// VV10-carrying functional (its nonlocal response is not in the kernel), an
/// empty (ia) space, or an over-budget dense matrix.
pub fn run_tda_dft(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    ks: &ScfResult,
    xc_name: Option<&str>,
    cfg: &TdaDftConfig,
) -> Result<TdaDftResult, FerricError> {
    use ferric_dft::libxc::FunctionalFamily;

    if !matches!(ks.spin, ferric_scf::Spin::Restricted) {
        return Err(FerricError::General(
            "run_tda_dft: closed-shell (restricted) reference only — this spike does not \
             cover UKS/ROKS/spin-flip"
                .into(),
        ));
    }

    let nmo = ks.eps_r().len();
    let nocc_total = (mol.nelec() as usize) / 2;
    let eps = ks.eps_r().to_vec();
    let c = ks.mos_r();

    let first_act = cfg.frozen_core;
    let nocc = ferric_mp2::rimp2::active_occ(nocc_total, cfg.frozen_core)?;
    let nvir = nmo - nocc_total;
    let n = nocc * nvir;
    if n == 0 {
        return Err(FerricError::General("run_tda_dft: empty (ia) space".into()));
    }
    check_tda_alloc(n)?;

    // ── Exact-exchange fraction, and the f_xc kernel (if any) ───────────────
    //
    // Resolve the functional ONCE (two independent XcDef handles are needed:
    // the kernel consumes one). c_HF comes from the same KMix logic the SCF
    // Fock build uses (`ks.rs::KsXc::k_mix`), so the TDA matrix cannot silently
    // disagree with the reference it was built on.
    let (c_hf_from_xc, kernel) = match xc_name {
        None => (1.0, None), // pure HF reference → CIS
        Some(name) => {
            // nspin=2: the kernel is spin-resolved (that IS the API), and the
            // singlet combination f_αα + f_αβ is extracted by feeding
            // δD_α = δD_β.
            let xc_probe = xc_def_from_name_nspin(name, 2)
                .map_err(|e| FerricError::General(format!("run_tda_dft: xc '{name}': {e:?}")))?;

            if xc_probe.vv10.is_some() {
                return Err(FerricError::General(format!(
                    "run_tda_dft: functional '{name}' carries VV10 nonlocal correlation, \
                     whose response is NOT in the f_xc kernel — the excitation energies \
                     would be silently wrong. Not supported by this spike."
                )));
            }
            if let Some(cam) = xc_probe.cam {
                if cam.omega != 0.0 {
                    return Err(FerricError::General(format!(
                        "run_tda_dft: functional '{name}' is range-separated (omega={}), \
                         and the long-range exchange kernel term is not assembled by this \
                         spike. Use a pure functional or a plain hybrid.",
                        cam.omega
                    )));
                }
            }
            if xc_probe
                .funcs
                .iter()
                .any(|f| f.family() == FunctionalFamily::MetaGga)
            {
                return Err(FerricError::General(format!(
                    "run_tda_dft: functional '{name}' is meta-GGA; ferric has no tau f_xc \
                     kernel (GgaFxcKernel rejects it). Not supported by this spike."
                )));
            }

            // Exact-exchange fraction: same resolution order as KsXc::k_mix.
            let c_hf = if let Some(cam) = xc_probe.cam {
                cam.c_sr // omega == 0 checked above, so c_sr == c_lr == the mix
            } else {
                xc_probe.b3lyp_mix.unwrap_or(0.0)
            };

            let kern = if cfg.include_fxc {
                let xc_kern = xc_def_from_name_nspin(name, 2).map_err(|e| {
                    FerricError::General(format!("run_tda_dft: xc '{name}': {e:?}"))
                })?;
                Some(
                    GgaFxcKernel::new(mol, obs.basis_set(), xc_kern, &cfg.grid).map_err(|e| {
                        FerricError::General(format!("run_tda_dft: GgaFxcKernel::new: {e}"))
                    })?,
                )
            } else {
                None
            };
            (c_hf, kern)
        }
    };
    let c_hf = cfg.c_hf_override.unwrap_or(c_hf_from_xc);

    // ── RI integrals over the active MO square (same tensor BSE/CIS uses) ───
    let mob = mo_b::build_full_b(mol, obs, dfbs, op, ks, cfg.frozen_core)?;
    let b = &mob.b_full;
    let naux = mob.naux;
    let bare = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        let mut acc = 0.0;
        for pp in 0..naux {
            acc += b[(pp, p, q)] * b[(pp, r, s)];
        }
        acc
    };

    // ── Orbital-energy diagonal + Coulomb + exact-exchange blocks ───────────
    //
    // Row `ia` is written exactly once by a single (i,a) pair — identical
    // structure to `bse::run_cis_tda`, deliberately, so the CIS exactness
    // anchor is a like-for-like comparison down to the summation order.
    let mut a_mat = Array2::<f64>::zeros((n, n));
    let fill_row = |ia: usize, row: &mut [f64]| {
        let i = ia / nvir;
        let a = ia % nvir;
        let eps_i = eps[first_act + i];
        let a_loc = nocc + a;
        let eps_a = eps[nocc_total + a];
        for j in 0..nocc {
            for bb in 0..nvir {
                let b_loc = nocc + bb;
                let jb = j * nvir + bb;
                let coul = bare(i, a_loc, j, b_loc); // (ia|jb)
                let exch = bare(a_loc, b_loc, i, j); // (ab|ij) == (ij|ab)
                row[jb] = 2.0 * coul - c_hf * exch;
            }
        }
        row[ia] += eps_a - eps_i;
    };
    const PAR_ROWS_THRESHOLD: usize = 8;
    if n >= PAR_ROWS_THRESHOLD {
        use rayon::prelude::*;
        a_mat
            .as_slice_mut()
            .expect("a_mat is contiguous (row-major default)")
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(ia, row)| fill_row(ia, row));
    } else {
        let flat = a_mat
            .as_slice_mut()
            .expect("a_mat is contiguous (row-major default)");
        for (ia, row) in flat.chunks_mut(n).enumerate() {
            fill_row(ia, row);
        }
    }

    // ── f_xc coupling block ─────────────────────────────────────────────────
    let fxc_included = kernel.is_some();
    if let Some(kern) = kernel.as_ref() {
        // Reference density on the kernel grid. For a RESTRICTED reference
        // D_α = D_β, and `density_alpha` is always populated (it is exactly
        // D_α, i.e. half of `density_total`) — use it directly rather than
        // re-deriving 0.5·D_total.
        let d_a = &ks.density_alpha;
        let ref_dens = kern.reference_density(d_a, d_a);
        let (k_fxc, asym_rel) =
            build_fxc_block(kern, &ref_dens, c, first_act, nocc_total, nocc, nvir);
        // The kernel's AO operator is symmetric by construction, so the (ia)
        // block must be too. A large asymmetry would indicate an adapter bug
        // (wrong index order), not grid noise. Reject rather than symmetrize
        // a wrong matrix into looking right.
        if asym_rel > 1e-8 {
            return Err(FerricError::General(format!(
                "run_tda_dft: f_xc (ia)-block asymmetry {asym_rel:e} exceeds the grid-noise \
                 tolerance 1e-8 — the AO→(ia) adapter is inconsistent"
            )));
        }
        a_mat += &k_fxc;
    }

    // ── Diagonalize ─────────────────────────────────────────────────────────
    // Same call-path reasoning as bse.rs: this runs after the row-parallel fill
    // (joined) and `run_tda_dft` is a top-level driver, never inside a rayon
    // par_iter. `opt_in_blas_threads()` defaults to 1 and self-guards to 1 on a
    // rayon worker.
    let (evals, evecs) = with_blas_threads(opt_in_blas_threads(), || a_mat.eigh(UPLO::Upper))
        .map_err(|e| FerricError::Lapack(format!("TDA-DFT eigh: {e}")))?;
    debug_assert!(
        evals
            .as_slice()
            .expect("eigh evals contiguous")
            .windows(2)
            .all(|w| w[0] <= w[1] + 1e-9),
        "eigh eigenvalues expected ascending"
    );
    let omega: Vec<f64> = evals.to_vec();

    let mu_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let oscillator_strength = tda_oscillator_strengths(
        &omega, &evecs, &mu_ao, c, first_act, nocc_total, nocc, nvir,
    );

    Ok(TdaDftResult {
        omega,
        oscillator_strength,
        x: evecs,
        nocc,
        nvir,
        c_hf,
        fxc_included,
    })
}

impl TdaDftResult {
    /// The dominant `(i, a)` particle–hole pair of state `n` (LOCAL active
    /// indices) and its weight `X²`. Use this — not nearest-energy matching —
    /// when comparing state-by-state against a reference code.
    ///
    /// `docs/VALIDATION.md` records that BSE-TDA's reported MAE is only a
    /// LOWER BOUND because naive nearest-energy matching mis-assigned a bright
    /// π→π* to a dark root. This accessor exists so that mistake is not
    /// repeated here.
    pub fn dominant_pair(&self, state: usize) -> ((usize, usize), f64) {
        let x = self.x.column(state);
        let mut best = (0usize, 0.0f64);
        for (ia, &v) in x.iter().enumerate() {
            if v * v > best.1 {
                best = (ia, v * v);
            }
        }
        ((best.0 / self.nvir, best.0 % self.nvir), best.1)
    }

    /// Excitation energies in eV.
    pub fn omega_ev(&self) -> Vec<f64> {
        self.omega
            .iter()
            .map(|w| w * 27.211_386_245_988)
            .collect::<Vec<f64>>()
    }
}

/// Transition dipole vector (a.u., length gauge) of state `n`, in the same
/// convention as [`tda_oscillator_strengths`]. Exposed separately from the
/// oscillator strength because a DIRECTION is a much stronger state-matching
/// fingerprint than a scalar magnitude.
pub fn transition_dipole(
    res: &TdaDftResult,
    mu_ao: &[Array2<f64>; 3],
    mo_coeff: &Array2<f64>,
    first_act: usize,
    nocc_total: usize,
    state: usize,
) -> [f64; 3] {
    let (nocc, nvir) = (res.nocc, res.nvir);
    let orbo = mo_coeff.slice(s![.., first_act..nocc_total]);
    let orbv = mo_coeff.slice(s![.., nocc_total..(nocc_total + nvir)]);
    let x = res.x.column(state);
    std::array::from_fn(|d| {
        let dip = orbo.t().dot(&mu_ao[d]).dot(&orbv);
        let mut acc = 0.0;
        for i in 0..nocc {
            for a in 0..nvir {
                acc += x[i * nvir + a] * dip[(i, a)];
            }
        }
        std::f64::consts::SQRT_2 * acc
    })
}
