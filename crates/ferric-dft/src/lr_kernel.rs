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
//! [`resolve_singlet_response_xc`](crate::lr_kernel::resolve_singlet_response_xc) decides whether a functional HAS a kernel
//! ferric can apply, and what exact-exchange fraction goes on the `−c_HF` K-like
//! terms. It hard-refuses (never silently degrades) every functional whose
//! response would be incomplete: VV10-carrying functionals (the nonlocal
//! response is not in the kernel) and range-separated hybrids (the long-range
//! exchange term needs attenuated integrals the drivers do not build).
//! LDA, GGA, global hybrids and τ-only meta-GGAs (SCAN / r2SCAN / TPSS) pass.
//!
//! [`singlet_fxc_ov_block`](crate::lr_kernel::singlet_fxc_ov_block) builds `K_{ia,jb} = (ia| f_αα + f_αβ |jb)` from a
//! [`SingletKernel`](crate::lr_kernel::SingletKernel): the validated
//! [`GgaFxcKernel`](crate::fxc::GgaFxcKernel) AO operator, or the τ-only
//! [`MggaFxcKernel`](crate::fxc::MggaFxcKernel) for a meta-GGA. With real orbitals
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
//! Each kernel is an AO-density-matrix → AO-potential operator: given a
//! spin-resolved perturbation (δD_α, δD_β) at a reference density it returns
//! (δV_α, δV_β). Feeding δD_α = δD_β = the (symmetrized) AO transition density
//! of pair `jb` makes δV_α = ∫(f_αα + f_αβ) δρ, the singlet combination, and
//! `K[:, jb] = C_occᵀ δV_α C_vir`. The factor is not asserted from theory — it
//! is pinned against PySCF `tddft.TDA` by `ferric-gw/tests/tda_dft_vs_pyscf.rs`
//! (LDA/PBE/B3LYP water, 1.0e-3 eV) and against PySCF `TDA`/`TDDFT` by
//! `ferric-tddft/tests/validation_tddft.rs`.

use crate::density_on_grid::{eval_density_uks, eval_tau_closed, UksDensityGrid};
use crate::fxc::{
    point_hessians, restricted_coefficients, GgaFxcKernel, MggaFxcKernel, MggaFxcReference,
};
use crate::grid::{AtomicGridConfig, GridPoint};
use crate::libxc::{xc_def_from_name_nspin, FunctionalFamily, XcDef};
use crate::vxc::{above_density_floor, DensityFloor};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ndarray::{s, Array2, Array3};
use rayon::prelude::*;

/// Relative `(ia)`-block asymmetry above which [`singlet_fxc_ov_block`]'s
/// result must be REJECTED rather than symmetrized.
///
/// The kernel's AO operator is symmetric by construction, so the `(ia)` block
/// must be too up to grid round-off. The GGA transition-density bug this guard
/// caught showed up at 3.1e-2 relative, five orders above it.
pub const FXC_BLOCK_ASYMMETRY_TOL: f64 = 1e-8;

/// A functional that passed [`resolve_singlet_response_xc`]: its exact-exchange
/// fraction and the (nspin = 2) XC definition for [`SingletKernel::new`].
#[derive(Debug)]
pub struct SingletResponseXc {
    /// Exact-exchange fraction for the `−c_HF (ij|ab)` (A) and `−c_HF (ib|aj)`
    /// (B) terms. Same resolution order as the SCF's `KMix`: the CAM
    /// short-range coefficient (ω = 0 is enforced, so it equals the long-range
    /// one), else the global-hybrid mix, else 0.
    pub c_hf: f64,
    /// Spin-resolved XC definition, ready for [`SingletKernel::new`] (which
    /// consumes it).
    pub xc_kernel: XcDef,
}

/// The spin-resolved f_xc kernel of a restricted reference, with its
/// reference state: GGA-family (LDA / GGA / global hybrid) or τ-only
/// meta-GGA. Built at D_α = D_β = ½D.
#[derive(Debug)]
pub enum SingletKernel {
    /// No meta-GGA component: the GGA kernel (which also covers LDA and
    /// global-hybrid semilocal parts).
    Gga {
        /// The kernel and its grid / AO data.
        kernel: GgaFxcKernel,
        /// Reference α/β densities at D_α = D_β.
        reference: Box<UksDensityGrid>,
    },
    /// At least one meta-GGA component: the τ-only meta-GGA kernel.
    Mgga {
        /// The kernel and its grid / AO data.
        kernel: MggaFxcKernel,
        /// Reference per-point Hessians, gradients and density-floor mask at
        /// D_α = D_β.
        reference: Box<MggaFxcReference>,
    },
}

impl SingletKernel {
    /// Build the kernel for `def` (from [`resolve_singlet_response_xc`]) on
    /// `grid` — the reference SCF's grid — at the restricted reference
    /// `d_alpha` (= D_α = D_β).
    pub fn new(
        mol: &Molecule,
        bs: &BasisSet,
        def: XcDef,
        grid: &AtomicGridConfig,
        d_alpha: &Array2<f64>,
    ) -> Result<Self, String> {
        if def
            .funcs
            .iter()
            .any(|f| f.family() == FunctionalFamily::MetaGga)
        {
            let kernel = MggaFxcKernel::new(mol, bs, def, grid)?;
            let reference = Box::new(kernel.reference_restricted(d_alpha, d_alpha)?);
            Ok(Self::Mgga { kernel, reference })
        } else {
            let kernel = GgaFxcKernel::new(mol, bs, def, grid)?;
            let reference = Box::new(kernel.reference_density(d_alpha, d_alpha));
            Ok(Self::Gga { kernel, reference })
        }
    }

    /// The functional, grid and AO values (`(nbf, npts)`) and gradients
    /// (`(3, nbf, npts)`) this kernel integrates with.
    fn grid_data(&self) -> (&XcDef, &[GridPoint], &Array2<f64>, &Array3<f64>) {
        match self {
            Self::Gga { kernel, .. } => (&kernel.xc, &kernel.grid, &kernel.chi, &kernel.dchi),
            Self::Mgga { kernel, .. } => (&kernel.xc, &kernel.grid, &kernel.chi, &kernel.dchi),
        }
    }

    /// (δV_α, δV_β) for the perturbation (δD_α, δD_β).
    pub fn apply(&self, dd_a: &Array2<f64>, dd_b: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        match self {
            Self::Gga { kernel, reference } => {
                kernel.apply_with_ref_restricted(reference, dd_a, dd_b)
            }
            Self::Mgga { kernel, reference } => kernel.apply_with_ref(reference, dd_a, dd_b),
        }
    }
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
///   * a meta-GGA component libxc has no second derivative for.
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
        .any(|f| f.family() == FunctionalFamily::MetaGga && !f.has_fxc())
    {
        return Err(FerricError::General(format!(
            "{caller}: functional '{name}' is a meta-GGA without a libxc second \
             derivative, so the tau f_xc response term cannot be built. Refused rather \
             than returning excitation energies without it."
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
/// the active `(ia)` space, via the [`SingletKernel`] AO adapter.
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
    kernel: &SingletKernel,
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
            let (dv_a, _dv_b) = kernel.apply(&dd, &dd);
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

/// Grid points per block of [`SingletOvFxc`]. A block's MO values and its
/// per-vector scratch are a few `(nvec·nocc, 256)` arrays, small enough to stay
/// in cache-sized GEMM panels.
const OV_BLOCK_PTS: usize = 256;

/// Most partial sums the [`SingletOvFxc::apply`] reduction keeps. The groups
/// are contiguous block ranges fixed by the block count alone, and are summed
/// in group order, so the product does not depend on the rayon thread count.
const OV_REDUCE_GROUPS: usize = 64;

/// One block of grid points: occupied and virtual MO values (and gradients)
/// there, and the restricted kernel coefficients of
/// [`restricted_coefficients`](crate::fxc::restricted_coefficients), quadrature
/// weight folded in and zero below the density floor.
struct OvBlock {
    /// `φ_i(r_g)`, `(nocc, nb)`.
    phi_o: Array2<f64>,
    /// `φ_a(r_g)`, `(nvir, nb)`.
    phi_v: Array2<f64>,
    /// `∂_x φ_i`, `∂_y φ_i`, `∂_z φ_i`; empty for a pure LDA.
    dphi_o: Vec<Array2<f64>>,
    /// The same for the virtuals.
    dphi_v: Vec<Array2<f64>>,
    /// `(δρ, g·∇δρ, δτ)` coefficients of u.
    cu: Vec<[f64; 3]>,
    /// `(δρ, g·∇δρ, δτ)` coefficients of `2δv_σαα + δv_σαβ`.
    cs: Vec<[f64; 3]>,
    /// `(δρ, g·∇δρ, δτ)` coefficients of t.
    ct: Vec<[f64; 3]>,
    /// ∇δρ coefficient of w, `2v_σαα + v_σαβ`.
    gam: Vec<f64>,
    /// Reference `g = ∇ρ_α`.
    grad: Vec<[f64; 3]>,
}

/// Matrix-free singlet f_xc product in the `(ia)` space: `y = K x` with `K`
/// the block [`singlet_fxc_ov_block`] builds, for a block of vectors at once.
///
/// # Same kernel, contracted in the MO basis
///
/// For `x` the AO perturbation is `δD = Σ_jb x_jb ½(C_j C_bᵀ + C_b C_jᵀ)`, and
/// `(K x)_ia = (C_occᵀ δV[δD] C_vir)_ia` by linearity, `δV` the
/// [`SingletKernel`]'s α response at δD_α = δD_β = δD. With MO values on the
/// grid, `φ_p = Σ_μ C_μp χ_μ`, the perturbation density and its derivatives
/// are
///
/// ```text
///   δρ = Σ_jb x_jb φ_j φ_b,   ∇δρ = Σ_jb x_jb ∇(φ_j φ_b),   δτ = ½ Σ_jb x_jb ∇φ_j·∇φ_b
/// ```
///
/// and the kernel's back-projection `Σ_g [u χχ + w·∇(χχ) + ½ t ∇χ·∇χ]`
/// transforms to
///
/// ```text
///   (K x)_ia = Σ_g [u φ_i φ_a + w·(φ_i ∇φ_a + ∇φ_i φ_a) + ½ t ∇φ_i·∇φ_a]
/// ```
///
/// with the point coefficients `u = c_u·(δρ, g·∇δρ, δτ)`, `t = c_t·(...)`,
/// `w = (c_s·(...)) g + γ ∇δρ` of the restricted regrouping
/// `crate::fxc::restricted_coefficients` (pinned against the spin-resolved
/// `linearize_point` that both dense kernels apply). The Hessians come from
/// `crate::fxc::point_hessians`, the meta-GGA kernel's evaluator, which on a
/// τ-free functional is the GGA kernel (`mgga_kernel_on_gga_functional_equals_gga_kernel`).
/// The density floor is the restricted one both dense kernels use: a point is
/// live where `ρ_α + ρ_β` is above the functional's floor.
///
/// # Cost
///
/// Per block and product: `Ψ = X φ_v` (and `X ∇φ_v` for a GGA), the
/// point-local linearisation, then `L φ_vᵀ` (and `L_x ∇φ_vᵀ`): about
/// `8·nvec·nocc·nvir·npts` flops for a GGA, against `nocc·nvir` full AO
/// kernel applications for the dense block. Memory: the MO values,
/// `(nocc + nvir)·npts` doubles (four times that for a GGA).
pub struct SingletOvFxc {
    blocks: Vec<OvBlock>,
    nocc: usize,
    nvir: usize,
    /// The functional has a σ (or τ) dependence: gradient terms are live.
    gradients: bool,
    /// The functional has a τ dependence.
    tau: bool,
}

impl std::fmt::Debug for SingletOvFxc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingletOvFxc")
            .field("nblocks", &self.blocks.len())
            .field("nocc", &self.nocc)
            .field("nvir", &self.nvir)
            .field("gradients", &self.gradients)
            .field("tau", &self.tau)
            .finish()
    }
}

/// Reference ρ_α, σ and τ on the whole grid, libxc-interleaved (2, 3 and 2
/// per point), and the reference ∇ρ_α as `(3, npts)`.
struct RestrictedReference {
    rho: Vec<f64>,
    sigma: Vec<f64>,
    tau: Vec<f64>,
    grad: Array2<f64>,
}

fn restricted_reference(
    d_alpha: &Array2<f64>,
    chi: &Array2<f64>,
    dchi: &Array3<f64>,
    with_tau: bool,
) -> RestrictedReference {
    let dens = eval_density_uks(d_alpha, d_alpha, chi, dchi);
    let npts = chi.ncols();
    let tau_a = with_tau.then(|| eval_tau_closed(d_alpha, dchi));
    let mut rho = vec![0.0; 2 * npts];
    let mut sigma = vec![0.0; 3 * npts];
    let mut tau = vec![0.0; 2 * npts];
    for g in 0..npts {
        rho[2 * g] = dens.rho_a[g];
        rho[2 * g + 1] = dens.rho_b[g];
        for c in 0..3 {
            sigma[3 * g + c] = dens.sigma[(c, g)];
        }
        if let Some(t) = &tau_a {
            tau[2 * g] = t[g];
            tau[2 * g + 1] = t[g];
        }
    }
    RestrictedReference {
        rho,
        sigma,
        tau,
        grad: dens.grad_a,
    }
}

impl SingletOvFxc {
    /// Doubles held by an operator over `npts` points: the MO values (and
    /// gradients when `gradients`) plus 13 coefficients per point. For a
    /// memory plan, before construction.
    pub fn resident_doubles(nocc: usize, nvir: usize, npts: usize, gradients: bool) -> usize {
        let per_mo = if gradients { 4 } else { 1 };
        (nocc + nvir)
            .saturating_mul(per_mo)
            .saturating_add(13)
            .saturating_mul(npts)
    }

    /// Whether `kernel`'s functional has gradient (σ or τ) terms, i.e. the
    /// `gradients` argument of [`Self::resident_doubles`].
    pub fn kernel_has_gradients(kernel: &SingletKernel) -> bool {
        kernel
            .grid_data()
            .0
            .funcs
            .iter()
            .any(|f| f.family() != FunctionalFamily::Lda)
    }

    /// Grid size of `kernel` (for [`Self::resident_doubles`]).
    pub fn kernel_npts(kernel: &SingletKernel) -> usize {
        kernel.grid_data().1.len()
    }

    /// Build the operator for `kernel` at the restricted reference `d_alpha`
    /// (= D_α = D_β, the density the kernel was built at), over the same
    /// `(ia)` space as [`singlet_fxc_ov_block`] with the same arguments.
    ///
    /// # Errors
    ///
    /// A libxc meta-GGA second-derivative failure (as the dense kernel).
    pub fn new(
        kernel: &SingletKernel,
        d_alpha: &Array2<f64>,
        mo_coeff: &Array2<f64>,
        first_act: usize,
        nocc_total: usize,
        nocc: usize,
        nvir: usize,
    ) -> Result<Self, String> {
        let (xc, grid, chi, dchi) = kernel.grid_data();
        let tau = xc
            .funcs
            .iter()
            .any(|f| f.family() == FunctionalFamily::MetaGga);
        let gradients = Self::kernel_has_gradients(kernel);
        let reference = restricted_reference(d_alpha, chi, dchi, tau);
        let orbo = mo_coeff.slice(s![.., first_act..nocc_total]).to_owned();
        let orbv = mo_coeff
            .slice(s![.., nocc_total..(nocc_total + nvir)])
            .to_owned();
        let floor = DensityFloor::for_xc(xc);
        let npts = grid.len();
        let starts: Vec<usize> = (0..npts).step_by(OV_BLOCK_PTS).collect();
        let blocks = starts
            .into_par_iter()
            .map(|lo| {
                let hi = (lo + OV_BLOCK_PTS).min(npts);
                let ctx = BlockInputs {
                    xc,
                    grid,
                    chi,
                    dchi,
                    reference: &reference,
                    orbo: &orbo,
                    orbv: &orbv,
                    floor,
                    gradients,
                };
                ctx.block(lo, hi)
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            blocks,
            nocc,
            nvir,
            gradients,
            tau,
        })
    }

    /// `K x` for every column of `x` (`(nocc·nvir, nvec)`, rows `ia =
    /// i·nvir + a`). Deterministic across rayon thread counts.
    pub fn apply(&self, x: &Array2<f64>) -> Array2<f64> {
        let (nocc, nvir) = (self.nocc, self.nvir);
        let nvec = x.ncols();
        assert_eq!(
            x.nrows(),
            nocc * nvir,
            "SingletOvFxc: x has the wrong length"
        );
        // Xs[(n·nocc + i), a] = x[ia, n]: one row-block per vector.
        let xs = Array2::from_shape_fn((nvec * nocc, nvir), |(r, a)| {
            x[((r % nocc) * nvir + a, r / nocc)]
        });
        let nb = self.blocks.len();
        let ngroups = nb.clamp(1, OV_REDUCE_GROUPS);
        let partials: Vec<Array2<f64>> = (0..ngroups)
            .into_par_iter()
            .map(|grp| {
                let mut acc = Array2::<f64>::zeros((nvec * nocc, nvir));
                for b in &self.blocks[grp * nb / ngroups..(grp + 1) * nb / ngroups] {
                    self.block_product(b, &xs, nvec, &mut acc);
                }
                acc
            })
            .collect();
        let mut acc = Array2::<f64>::zeros((nvec * nocc, nvir));
        for p in &partials {
            acc += p;
        }
        Array2::from_shape_fn((nocc * nvir, nvec), |(ia, n)| {
            acc[(n * nocc + ia / nvir, ia % nvir)]
        })
    }

    /// One block's contribution to `acc` (`(nvec·nocc, nvir)`, rows
    /// `n·nocc + i`).
    fn block_product(&self, b: &OvBlock, xs: &Array2<f64>, nvec: usize, acc: &mut Array2<f64>) {
        let nocc = self.nocc;
        let nb = b.phi_o.ncols();
        let psi = xs.dot(&b.phi_v);
        let psi_d: Vec<Array2<f64>> = b.dphi_v.iter().map(|d| xs.dot(d)).collect();
        let mut l0 = Array2::<f64>::zeros((nvec * nocc, nb));
        let mut lax: Vec<Array2<f64>> = (0..b.dphi_o.len())
            .map(|_| Array2::<f64>::zeros((nvec * nocc, nb)))
            .collect();
        let mut drho = vec![0.0; nb];
        let mut dgrad = [vec![0.0; nb], vec![0.0; nb], vec![0.0; nb]];
        let mut dtau = vec![0.0; nb];
        let mut u = vec![0.0; nb];
        let mut t = vec![0.0; nb];
        let mut w = [vec![0.0; nb], vec![0.0; nb], vec![0.0; nb]];
        for n in 0..nvec {
            let rows = n * nocc..(n + 1) * nocc;
            // Perturbation density, gradient and τ at the block's points.
            drho.fill(0.0);
            dgrad.iter_mut().for_each(|v| v.fill(0.0));
            dtau.fill(0.0);
            for (i, r) in rows.clone().enumerate() {
                let (po, ps) = (b.phi_o.row(i), psi.row(r));
                for g in 0..nb {
                    drho[g] += po[g] * ps[g];
                }
                if !self.gradients {
                    continue;
                }
                for ax in 0..3 {
                    let (dpo, dps) = (b.dphi_o[ax].row(i), psi_d[ax].row(r));
                    for g in 0..nb {
                        dgrad[ax][g] += dpo[g] * ps[g] + po[g] * dps[g];
                    }
                    if self.tau {
                        for g in 0..nb {
                            dtau[g] += 0.5 * dpo[g] * dps[g];
                        }
                    }
                }
            }
            // Point-local linearised potential.
            for g in 0..nb {
                let gd = [dgrad[0][g], dgrad[1][g], dgrad[2][g]];
                let gr = b.grad[g];
                let bb = gr[0] * gd[0] + gr[1] * gd[1] + gr[2] * gd[2];
                let lin = |k: [f64; 3]| k[0] * drho[g] + k[1] * bb + k[2] * dtau[g];
                u[g] = lin(b.cu[g]);
                if self.gradients {
                    let sc = lin(b.cs[g]);
                    t[g] = lin(b.ct[g]);
                    for ax in 0..3 {
                        w[ax][g] = sc * gr[ax] + b.gam[g] * gd[ax];
                    }
                }
            }
            // L0 = u φ_o + w·∇φ_o,  L_x = w_x φ_o + ½ t ∂_x φ_o.
            for (i, r) in rows.enumerate() {
                let po = b.phi_o.row(i);
                let mut l0r = l0.row_mut(r);
                for g in 0..nb {
                    l0r[g] = u[g] * po[g];
                }
                if !self.gradients {
                    continue;
                }
                for ax in 0..3 {
                    let dpo = b.dphi_o[ax].row(i);
                    for g in 0..nb {
                        l0r[g] += w[ax][g] * dpo[g];
                    }
                    let mut lr = lax[ax].row_mut(r);
                    for g in 0..nb {
                        lr[g] = w[ax][g] * po[g] + 0.5 * t[g] * dpo[g];
                    }
                }
            }
        }
        ndarray::linalg::general_mat_mul(1.0, &l0, &b.phi_v.t(), 1.0, acc);
        for (l, dv) in lax.iter().zip(&b.dphi_v) {
            ndarray::linalg::general_mat_mul(1.0, l, &dv.t(), 1.0, acc);
        }
    }
}

/// Borrowed inputs of one [`OvBlock`] build.
struct BlockInputs<'a> {
    xc: &'a XcDef,
    grid: &'a [GridPoint],
    chi: &'a Array2<f64>,
    dchi: &'a Array3<f64>,
    reference: &'a RestrictedReference,
    orbo: &'a Array2<f64>,
    orbv: &'a Array2<f64>,
    floor: DensityFloor,
    gradients: bool,
}

impl BlockInputs<'_> {
    /// MO values and kernel coefficients on points `lo..hi`.
    fn block(&self, lo: usize, hi: usize) -> Result<OvBlock, String> {
        let r = self.reference;
        let mo = |c: &Array2<f64>, ao: ndarray::ArrayView2<f64>| c.t().dot(&ao);
        let chi_b = self.chi.slice(s![.., lo..hi]);
        let (dphi_o, dphi_v) = if self.gradients {
            (0..3)
                .map(|ax| {
                    let d = self.dchi.slice(s![ax, .., lo..hi]);
                    (mo(self.orbo, d), mo(self.orbv, d))
                })
                .unzip()
        } else {
            (Vec::new(), Vec::new())
        };
        let (hess, vsigma) = point_hessians(
            self.xc,
            &r.rho[2 * lo..2 * hi],
            &r.sigma[3 * lo..3 * hi],
            &r.tau[2 * lo..2 * hi],
        )?;
        let nb = hi - lo;
        let mut blk = OvBlock {
            phi_o: mo(self.orbo, chi_b),
            phi_v: mo(self.orbv, chi_b),
            dphi_o,
            dphi_v,
            cu: vec![[0.0; 3]; nb],
            cs: vec![[0.0; 3]; nb],
            ct: vec![[0.0; 3]; nb],
            gam: vec![0.0; nb],
            grad: vec![[0.0; 3]; nb],
        };
        for k in 0..nb {
            let g = lo + k;
            if !above_density_floor(r.rho[2 * g] + r.rho[2 * g + 1], self.floor) {
                continue;
            }
            let (cu, cs, ct, gam) =
                restricted_coefficients(&hess[k], &vsigma[k], self.grid[g].weight);
            blk.cu[k] = cu;
            blk.cs[k] = cs;
            blk.ct[k] = ct;
            blk.gam[k] = gam;
            blk.grad[k] = [r.grad[(0, g)], r.grad[(1, g)], r.grad[(2, g)]];
        }
        Ok(blk)
    }
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
    fn tau_only_meta_ggas_resolve_with_no_exact_exchange() {
        for name in ["SCAN", "r2SCAN", "TPSS"] {
            let r = resolve_singlet_response_xc(name, WHO)
                .unwrap_or_else(|e| panic!("{name} has a tau f_xc kernel: {e}"));
            assert_eq!(r.c_hf, 0.0, "{name}: a pure meta-GGA has c_HF = 0");
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

    /// [`SingletOvFxc`] (MO-space, batched) applied to the identity equals the
    /// dense AO-route block [`singlet_fxc_ov_block`], water/6-31G at an HF
    /// density, for an LDA, a GGA, a global hybrid and a τ-only meta-GGA.
    ///
    /// Bar: `FXC_BLOCK_ASYMMETRY_TOL · max|K|`, the noise floor the dense
    /// route is accepted at (its raw block, analytically symmetric, is
    /// rejected above that asymmetry). The two routes contract the same grid
    /// sum in different orders (AO then MO against MO directly), so they
    /// agree to that floor or better; a wrong term (a dropped gradient or τ
    /// piece, a lost ½) is an O(1) relative change.
    ///
    /// Mutations (applied and reverted, 2026-11-13; the loop stops at the
    /// first failing functional): dropping the `½ t ∂φ_o` τ term passes LDA,
    /// PBE and B3LYP and fails r2SCAN (2.9e-2 against a 1.3e-9 bar); dropping
    /// `γ ∇δρ` from w fails PBE (2.6e-2); dropping the u term fails LDA
    /// (7.9e-2).
    #[test]
    fn ov_operator_matches_the_dense_block() {
        use ferric_core::basis;
        use ferric_core::parallel::ParallelContext;
        use ferric_integrals::basis_bridge::PreparedBasis;
        use ferric_integrals::operator::Operator;
        let _env_guard = crate::TEST_BUDGET_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let xyz = format!(
            "{}/../../testdata/molecules/water.xyz",
            env!("CARGO_MANIFEST_DIR")
        );
        let mol = Molecule::load_xyz(&xyz).unwrap();
        let bs = basis::bundled("6-31g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &prep).unwrap();
        let scf = ferric_scf::rhf::solve_rhf(
            &ParallelContext::default(),
            &mol,
            &prep,
            op,
            &bounds,
            &ferric_scf::rhf::RhfConfig::default(),
        )
        .unwrap();
        let nocc = (mol.nelec() / 2) as usize;
        let nvir = scf.eps_r().len() - nocc;
        let dim = nocc * nvir;
        for name in ["LDA", "PBE", "B3LYP", "r2SCAN"] {
            let spec = resolve_singlet_response_xc(name, WHO).unwrap();
            let grid = AtomicGridConfig::default();
            let kern =
                SingletKernel::new(&mol, &bs, spec.xc_kernel, &grid, &scf.density_alpha).unwrap();
            let (dense, asym) = singlet_fxc_ov_block(&kern, scf.mos_r(), 0, nocc, nocc, nvir);
            assert!(
                asym < FXC_BLOCK_ASYMMETRY_TOL,
                "{name}: dense asymmetry {asym:e}"
            );
            let ov = SingletOvFxc::new(&kern, &scf.density_alpha, scf.mos_r(), 0, nocc, nocc, nvir)
                .unwrap();
            let mf = ov.apply(&Array2::eye(dim));
            let scale = dense.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
            let diff = (&mf - &dense).iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
            eprintln!("{name}: max|K_mf - K_dense| = {diff:.3e} (max|K| = {scale:.3e})");
            assert!(
                diff <= FXC_BLOCK_ASYMMETRY_TOL * scale,
                "{name}: matrix-free f_xc block differs from the dense one by {diff:e} \
                 (bar {:e})",
                FXC_BLOCK_ASYMMETRY_TOL * scale
            );
        }
    }
}
