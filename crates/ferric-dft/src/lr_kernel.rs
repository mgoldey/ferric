//! Closed-shell SINGLET linear-response XC kernel in the particle–hole `(ia)`
//! space — the `(ia|f_xc|jb)` block that TDA-DFT and Casida TDDFT add to their
//! A (and, for Casida, B) matrices.
//!
//! This is the ONE implementation of that block. It lives here, in the crate
//! that owns the kernels, so that every response driver shares it:
//!
//!   * `ferric_gw::tddft::run_tda_dft` — the library-only TDA-DFT path whose
//!     result is cross-checked against PySCF `tddft.TDA`
//!     (`crates/ferric-gw/tests/tda_dft_vs_pyscf.rs`); this code was moved here
//!     from that module unchanged;
//!   * `ferric_tddft::run_tddft` — the user-facing CLI `tda`/`tddft` and Python
//!     `run_tddft` path, which until this module existed had a TODO where this
//!     block belongs and returned kernel-less numbers for KS references.
//!
//! # The two pieces
//!
//! [`resolve_singlet_response_xc`] decides whether a functional HAS a kernel
//! ferric can apply, and what exact-exchange fraction goes on the `−c_HF` K-like
//! terms. It hard-refuses (never silently degrades) every functional whose
//! response would be incomplete: meta-GGA (no τ f_xc kernel), VV10-carrying
//! functionals (the nonlocal response is not in the kernel), and
//! range-separated hybrids (the long-range exchange term needs attenuated
//! integrals the drivers do not build).
//!
//! [`singlet_fxc_ov_block`] builds `K_{ia,jb} = (ia| f_αα + f_αβ |jb)` from the
//! validated [`GgaFxcKernel`](crate::fxc::GgaFxcKernel) AO operator. With real orbitals
//! `(ia|f_xc|jb) = (ia|f_xc|bj)`, so the SAME block enters A and B:
//!
//! ```text
//!   A_{ia,jb} += K_{ia,jb}
//!   B_{ia,jb} += K_{ia,jb}
//! ```
//!
//! which is exactly PySCF `tdscf.rhf.get_ab`'s `a += iajb; b += iajb` (its
//! `iajb` carries the spin-summation factor 2, and K here equals that `iajb` —
//! pinned numerically against PySCF, see below).
//!
//! # The adapter
//!
//! `GgaFxcKernel` is an AO-density-matrix → AO-potential operator: given a
//! spin-resolved perturbation (δD_α, δD_β) at a reference density it returns
//! (δV_α, δV_β). Feeding δD_α = δD_β = the (symmetrized) AO transition density
//! of pair `jb` makes δV_α = ∫(f_αα + f_αβ) δρ, the singlet combination, and
//! `K[:, jb] = C_occᵀ δV_α C_vir`. The factor is not asserted from theory — it
//! is pinned against PySCF `tddft.TDA` by `ferric-gw/tests/tda_dft_vs_pyscf.rs`
//! (LDA/PBE/B3LYP water, 1.0e-3 eV) and against PySCF `TDA`/`TDDFT` by
//! `ferric-tddft/tests/validation_tddft.rs`.

use crate::density_on_grid::UksDensityGrid;
use crate::fxc::GgaFxcKernel;
use crate::libxc::{xc_def_from_name_nspin, FunctionalFamily, XcDef};
use ferric_core::FerricError;
use ndarray::{s, Array2};

/// Relative `(ia)`-block asymmetry above which [`singlet_fxc_ov_block`]'s
/// result must be REJECTED rather than symmetrized.
///
/// The kernel's AO operator is symmetric by construction, so the `(ia)` block
/// must be too up to grid round-off. The GGA transition-density bug this guard
/// caught showed up at 3.1e-2 relative, five orders above it.
pub const FXC_BLOCK_ASYMMETRY_TOL: f64 = 1e-8;

/// A functional that passed [`resolve_singlet_response_xc`]: its exact-exchange
/// fraction and the (nspin = 2) XC definition for [`GgaFxcKernel::new`](crate::fxc::GgaFxcKernel::new).
#[derive(Debug)]
pub struct SingletResponseXc {
    /// Exact-exchange fraction for the `−c_HF (ij|ab)` (A) and `−c_HF (ib|aj)`
    /// (B) terms. Same resolution order as the SCF's `KMix`: the CAM
    /// short-range coefficient (ω = 0 is enforced, so it equals the long-range
    /// one), else the global-hybrid mix, else 0.
    pub c_hf: f64,
    /// Spin-resolved XC definition, ready for [`GgaFxcKernel::new`](crate::fxc::GgaFxcKernel::new) (which
    /// consumes it).
    pub xc_kernel: XcDef,
}

/// Decide whether `name` has a complete singlet response kernel in ferric, and
/// resolve its exact-exchange fraction.
///
/// `caller` prefixes every error message so the user sees which driver
/// refused.
///
/// # Errors
///
/// A hard error, never a warning, for:
///   * an unknown functional name;
///   * a functional carrying VV10 nonlocal correlation (its response is not in
///     the kernel — the excitation energies would silently omit it);
///   * a range-separated hybrid (`omega != 0`) — the long-range exact-exchange
///     response needs erf-attenuated integrals no driver builds;
///   * any meta-GGA component — ferric has no τ f_xc kernel.
pub fn resolve_singlet_response_xc(
    name: &str,
    caller: &str,
) -> Result<SingletResponseXc, FerricError> {
    // nspin = 2: the kernel is spin-resolved (that IS its API), and the
    // singlet combination f_αα + f_αβ is extracted by feeding δD_α = δD_β.
    let probe = xc_def_from_name_nspin(name, 2)
        .map_err(|e| FerricError::General(format!("{caller}: xc '{name}': {e:?}")))?;

    if probe.vv10.is_some() {
        return Err(FerricError::General(format!(
            "{caller}: functional '{name}' carries VV10 nonlocal correlation, whose \
             linear response is NOT in ferric's f_xc kernel — the excitation energies \
             would silently omit it. Refused; use a functional without VV10."
        )));
    }
    if let Some(cam) = probe.cam {
        if cam.omega != 0.0 {
            return Err(FerricError::General(format!(
                "{caller}: functional '{name}' is range-separated (omega = {}); the \
                 long-range exact-exchange response term is not assembled, so the \
                 excitation energies would be wrong. Refused; use a pure functional or \
                 a global hybrid.",
                cam.omega
            )));
        }
    }
    if probe
        .funcs
        .iter()
        .any(|f| f.family() == FunctionalFamily::MetaGga)
    {
        return Err(FerricError::General(format!(
            "{caller}: functional '{name}' is meta-GGA; ferric has no tau f_xc kernel, so \
             the XC response term cannot be built. Refused rather than returning \
             excitation energies without it."
        )));
    }

    let c_hf = if let Some(cam) = probe.cam {
        cam.c_sr // omega == 0 checked above, so c_sr == c_lr == the mix
    } else {
        probe.b3lyp_mix.unwrap_or(0.0)
    };

    Ok(SingletResponseXc {
        c_hf,
        xc_kernel: probe,
    })
}

/// Build the singlet f_xc coupling block `K_{ia,jb} = (ia| f_αα + f_αβ |jb)` in
/// the active `(ia)` space, via the validated `GgaFxcKernel` AO adapter.
///
/// Index convention: row/column `ia = i*nvir + a` with `i` in
/// `first_act..nocc_total` (local `0..nocc`) and `a` in
/// `nocc_total..nocc_total+nvir` (local `0..nvir`) of `mo_coeff`'s columns.
///
/// Returns `(K_sym, asym_rel)`: the symmetrized block and the relative
/// asymmetry of the raw one. Callers MUST reject `asym_rel >
/// FXC_BLOCK_ASYMMETRY_TOL` — a large asymmetry means the adapter, not the
/// grid, is wrong, and symmetrizing would hide it.
///
/// Cost: `nocc·nvir` kernel applications. That is deliberate — it reuses the
/// already-validated kernel unmodified rather than re-deriving the grid-space
/// contraction. Peak memory is two `(n, n)` matrices (raw + symmetrized).
pub fn singlet_fxc_ov_block(
    kernel: &GgaFxcKernel,
    ref_dens: &UksDensityGrid,
    mo_coeff: &Array2<f64>,
    first_act: usize,
    nocc_total: usize,
    nocc: usize,
    nvir: usize,
) -> (Array2<f64>, f64) {
    let n = nocc * nvir;
    let orbo = mo_coeff.slice(s![.., first_act..nocc_total]).to_owned();
    let orbv = mo_coeff
        .slice(s![.., nocc_total..(nocc_total + nvir)])
        .to_owned();

    let mut k = Array2::<f64>::zeros((n, n));
    for j in 0..nocc {
        let cj = orbo.column(j);
        for b in 0..nvir {
            let cb = orbv.column(b);
            // δD = ½(C_j C_bᵀ + C_b C_jᵀ), EXPLICITLY SYMMETRIZED.
            //
            // The raw outer product `C_j C_bᵀ` is fine for LDA and WRONG for
            // GGA: the gradient terms contract against ∇δD, and
            // ∇(C_j C_bᵀ) ≠ ∇(C_b C_jᵀ), so the antisymmetric part of δD leaks
            // into δV through the σ = |∇ρ|² coupling.
            //
            // MEASURED on water/STO-3G/PBE: the raw outer product gives an
            // (ia)-block asymmetry of 3.1e-2 relative — five orders of
            // magnitude above FXC_BLOCK_ASYMMETRY_TOL. Symmetrizing drops the
            // asymmetry under the guard and brings PBE excitation energies to
            // 1.0e-3 eV of PySCF `tddft.TDA`. LDA is unchanged by this.
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

#[cfg(test)]
mod tests {
    use super::*;

    const WHO: &str = "test";

    #[test]
    fn pure_functionals_resolve_with_no_exact_exchange() {
        for name in ["LDA", "PBE"] {
            let r = resolve_singlet_response_xc(name, WHO)
                .unwrap_or_else(|e| panic!("{name} must have a kernel: {e}"));
            assert_eq!(r.c_hf, 0.0, "{name}: a pure functional has c_HF = 0");
        }
    }

    #[test]
    fn b3lyp_resolves_with_its_20_percent_exact_exchange() {
        let r = resolve_singlet_response_xc("B3LYP", WHO).expect("B3LYP has a GGA kernel");
        assert!(
            (r.c_hf - 0.2).abs() < 1e-12,
            "B3LYP c_HF must be 0.20, got {}",
            r.c_hf
        );
    }

    #[test]
    fn meta_gga_is_refused_not_approximated() {
        for name in ["SCAN", "r2SCAN"] {
            let err = resolve_singlet_response_xc(name, WHO)
                .expect_err("meta-GGA has no tau f_xc kernel and must be refused");
            assert!(
                err.to_string().contains("meta-GGA"),
                "{name}: refusal must name the reason, got: {err}"
            );
        }
    }

    #[test]
    fn range_separated_and_vv10_functionals_are_refused() {
        // wB97X-V is BOTH range-separated and VV10-carrying; either refusal is
        // correct, and both are hard errors.
        assert!(resolve_singlet_response_xc("wB97X-V", WHO).is_err());
        // Canonical libxc identifier (no friendly alias exists for CAM-B3LYP).
        assert!(resolve_singlet_response_xc("HYB_GGA_XC_CAM_B3LYP", WHO).is_err());
    }
}
