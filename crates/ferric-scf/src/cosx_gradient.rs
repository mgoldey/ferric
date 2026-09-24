//! Analytic nuclear gradient of the COSX (seminumerical) exchange energy.
//!
//! When `k_builder = "cosx"`, the SCF's exchange energy is the one
//! [`crate::cosx_k::CosxK`] computes, not the exact four-centre one, so the
//! exchange part of the nuclear gradient must be the derivative of THAT
//! energy. Differentiating exact exchange instead (what every gradient path did
//! before this module) leaves a residual of the size of the COSX grid error:
//! measured 2e-5..3e-4 Ha/Bohr on water at a (30,110) grid
//! (`scripts/cosx_gradient_proto.py`).
//!
//! # The energy (read from `cosx_k.rs`, overlap fit OFF)
//!
//! ```text
//!     tr[D K(D)] = sum_g w_g e_g,   e_g = F_g^T A^g F_g,   F_g = D phi(r_g)
//!     A^g_{mu nu} = \int chi_mu chi_nu / |r - r_g|
//! ```
//!
//! `K = sym(X G^T)` with `X = sqrt(w) phi`, `G = A^g D X`, so
//! `tr[D K] = sum_g w_g phi_g^T D A^g D phi_g` — a SYMMETRIC quadratic form in
//! `D` whose `D`-derivative is `2 K(D)`, i.e. exactly the exchange part of the
//! Fock matrix the SCF used. The plain-COSX SCF energy is therefore
//! variational, and its gradient is the usual Hellmann–Feynman + Pulay (`−W dS`,
//! carried by [`crate::gradient::oneelectron_gradient`]) form with the
//! four-centre exchange derivative REPLACED by the explicit fixed-`D` derivative
//! of `sum_g w_g e_g`, computed here.
//!
//! The grid is ferric's atom-centred Becke–Lebedev grid (`build_atomic_grid`,
//! the same one `CosxK::new` builds for an unpruned `CosxConfig::grid`); every
//! point rides rigidly with its home atom `h(g)`. Three terms:
//!
//! ```text
//!  d/dR_X [w_g e_g] = e_g dw_g/dR_X                                           (c)
//!    + w_g sum_mu 2 (D G_g)_mu grad phi_mu(r_g) . (delta_{X,h(g)} - delta_{X,atom(mu)})  (a)
//!    + w_g sum_{lam sig} F_lam F_sig dA^g_{lam sig}/dR_X                      (b)
//! ```
//!
//! * (a) AO values on the grid: `phi_mu(r_g)` moves with its own centre (−grad)
//!   and with the point (+grad).
//! * (b) the ESP-like integrals: `dA/dR_X` = bra-centre + ket-centre + charge-
//!   position derivatives, the last one landing on `h(g)`. libint2's nuclear
//!   first-derivative engine with ONE unit charge at `r_g` returns exactly these
//!   three blocks `[bra xyz, ket xyz, charge xyz]` (the layout
//!   `oneelectron_gradient` consumes) for `V = −A`, hence the sign flips below.
//! * (c) the Becke weight response `weight1` of
//!   [`ferric_dft::grid::build_atomic_grid_with_response`] (total derivative
//!   with the point riding its atom — the PySCF `grids_response_cc`
//!   convention).
//!
//! This is the structure of Plessow & Weigend, J. Comput. Chem. 33, 810 (2012)
//! (seminumerical-exchange gradients on a moving, weight-responsive grid) and of
//! PySCF's `sgx/grad/rhf.py` grid-response branch; the Python prototype
//! `scripts/cosx_gradient_proto.py` validated this exact term set against
//! central finite differences to <= 2.1e-9 Ha/Bohr (water, STO-3G and 6-31G,
//! RHF and B3LYP) and showed each of (a), (b), (c) to be O(0.1) on its own.
//!
//! # The overlap fit (`overlap_fit = true`, the energy default): Z-vector
//!
//! With the fit, `K_f(D) = sym(Q Kt(D))`, `Q = S S_num^{-1}`,
//! `S_num = sum_g w_g phi_g phi_g^T`, and `K_f` is NOT self-adjoint:
//! `tr[Y K_f(B)] = tr[B L(Y)]` with `L(Y) = sym(Kt_builder(Y Q))`. So
//! `dE/dD = F + Delta` with `Delta = (c/4)(K_f(D) - L(D))` (RHF) and the fitted
//! SCF energy is not stationary in the orbitals. The exact gradient is the
//! Handy–Schaefer Lagrangian one (`fitted_exchange_response`):
//!
//! ```text
//!   (e_a - e_i) z_ai + p [C^T (J(Zs) - c_L L(Zs)) C]_ai = -p Delta_ai     (Z-vector)
//!   Zs = sym(C_v z C_o^T)          RHF: p = 4, c_L = c/2   UHF: p = 2, c_L = c (per spin)
//!   W  = w_o C_o (e_o + Delta_oo + R(Zs)_oo) C_o^T + sym(C_v (z e_o) C_o^T)
//!   g  = V_nn' + tr[(D + Zs) h'] - tr[S' W] + J'(D, D + 2 Zs)/2
//!        - (c/4) dT(D, D) - (c/2) dT(Zs, D)                               (RHF)
//! ```
//!
//! where `T(Y, B) = tr[Y Q Kt(B)]` and `dT` is its explicit derivative at fixed
//! matrices, INCLUDING `dQ = dS S_num^{-1} - Q dS_num S_num^{-1}`
//! (`cosx_exchange_gradient_bilinear`). The prototype
//! `scripts/cosx_fit_gradient_proto.py` validated this against FD of the fitted
//! energy (RHF water STO-3G/6-31G 1.7e-9/1.9e-9, vs 1.1e-6/1.4e-6 without the
//! response) and the `Q = I` limit (response identically zero, plain gradient
//! reproduced to 2e-16).
//!
//! # What is NOT supported, and why (hard errors, never a silent fallback)
//!
//! * **Fitted COSX with a KS functional.** The Z-vector term then also needs
//!   `tr[Zs V_xc'(D)]` — the nuclear derivative of the XC Fock matrix at fixed
//!   density, with grid response — which ferric does not have (`hessian.rs` is a
//!   stub). Fit-off COSX with a hybrid is exact (the energy is variational).
//! * **A pruned COSX grid.** The weight response exists only for the flat grid
//!   (see `build_atomic_grid_with_response`).
//!
//! # Screening
//!
//! The energy's density-driven pair screen (`screen_thresh`, default 1e-7) and
//! shell-sparse half transforms (`eps` 1e-10) drop contributions below their
//! thresholds; this gradient evaluates every shell pair at every point, i.e. it
//! differentiates the UNSCREENED energy. The difference is the screen error,
//! measured (not assumed) by `tests/cosx_gradient.rs`. It is also why this
//! routine is O(npts · nshell²) with no screening: correct first, fast later.

use std::os::raw::c_int;

use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_dft::grid::{build_atomic_grid_with_response, GridPoint};
use ferric_integrals::ao_grid::{
    collect_shells, eval_basis_and_grad_on_points_unchecked, LocatedShell,
};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi::{self, CAtom};
use ndarray::Array2;
use ndarray_linalg::{Cholesky, Diag, SolveTriangular, UPLO};
use rayon::prelude::*;

use crate::cosx_k::{validate_grid, CosxConfig, CosxHalfTransform, CosxK};
use crate::fock::{JBuilder, KBuilder};
use crate::screening::SchwarzBounds;

/// Grid points per parallel work item. Fixed (a pure function of the point
/// count), and the per-chunk partials are summed in chunk order, so the result
/// is bit-identical across thread counts. Each chunk owns two libint2 engines
/// and `(nbf, CHUNK)` AO planes (~4 · nbf · 128 · 8 B) per density term.
pub const COSX_GRAD_CHUNK_POINTS: usize = 128;

/// Chunks evaluated concurrently before their partials are folded (in chunk
/// order) into the running sums. Bounds the resident per-chunk `nbf²` partials
/// of the fitted path to `COSX_GRAD_CHUNK_GROUP` per term, independent of the
/// thread count (a fixed constant, so the fold order — and the result — does not
/// depend on it).
const COSX_GRAD_CHUNK_GROUP: usize = 16;

/// libint2 precision for the per-point 3c1e value and derivative engines — the
/// same value the 1e gradient engines in `gradient.rs` use.
const COSX_GRAD_PRECISION: f64 = 1e-14;

/// Largest `max|D − D^T|` (relative to `max(1, max|D|)`) accepted. The AO term
/// uses `d e_g / d phi` formulas that hold only for symmetric matrices.
const DENSITY_SYMMETRY_TOL: f64 = 1e-8;

/// Z-vector GMRES: relative residual target, restart length, iteration cap.
/// `1e-10` relative puts the response error ~1e-10 x |Delta|-scale on the
/// gradient, two decades under the 1e-9 FD floor the prototype reached.
const ZVEC_RTOL: f64 = 1e-10;
const ZVEC_RESTART: usize = 30;
const ZVEC_MAX_ITER: usize = 300;

/// Integral threshold of the exact Coulomb builds inside the Z-vector solve.
const ZVEC_J_THRESH: f64 = 1e-14;

/// Which `Q` the fitted machinery uses. `Configured` is production (`Q = S
/// S_num^{-1}` when `overlap_fit`, `I` otherwise). `Identity` forces `Q = I`
/// through the FITTED code path — the fit's trivial limit, where `K_f = L`,
/// `Delta = 0`, the Z-vector vanishes and the Lagrangian gradient must reduce
/// to the plain one. It exists for that exactness anchor
/// (`tests/cosx_gradient.rs`); it does not correspond to any SCF energy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitQ {
    /// Production: the `Q` of the configuration.
    Configured,
    /// Anchor only: `Q = I` on the fitted code path.
    Identity,
}

/// Refuse the COSX configurations this gradient cannot differentiate exactly
/// (see the module doc). Public so a caller can fail BEFORE a long SCF.
///
/// The overlap fit is accepted: its non-variational response is supplied by
/// `fitted_exchange_response`. KS + fit is refused separately by
/// [`check_fitted_ks_supported`], which needs to know the functional.
pub fn check_gradient_supported(cfg: &CosxConfig) -> Result<(), FerricError> {
    if cfg.grid.prune.is_some() {
        return Err(FerricError::General(
            "COSX analytic gradient: a pruned COSX grid is not supported (the Becke weight \
             response is built for the flat grid only); set the cosx grid's prune = None"
                .into(),
        ));
    }
    validate_grid(&cfg.grid)
}

/// Refuse the overlap-fitted COSX gradient for a Kohn–Sham functional.
///
/// The fitted energy's Z-vector term needs `tr[Zs V_xc'(D)]`, the nuclear
/// derivative of the XC Fock matrix at fixed density (grid response included),
/// which ferric does not implement for any functional family. Returning a
/// gradient without it would miss FD by the same ~1e-6 Ha/Bohr the response
/// exists to remove, so it is an error. `xc = None` (Hartree–Fock) and
/// `overlap_fit = false` pass.
pub fn check_fitted_ks_supported(cfg: &CosxConfig, xc: Option<&str>) -> Result<(), FerricError> {
    if let (true, Some(name)) = (cfg.overlap_fit, xc) {
        return Err(FerricError::General(format!(
            "COSX analytic gradient: overlap_fit = true with the functional '{name}' is not \
             supported. The overlap-fitted COSX energy is not variational, and its orbital- \
             response (Z-vector) term needs the nuclear derivative of the XC Fock matrix, which \
             ferric does not implement. Set cosx_overlap_fit = false for KS gradient tasks \
             (exact), or use Hartree-Fock (the fitted HF gradient is exact)."
        )));
    }
    Ok(())
}

/// The refusal for a reference whose SCF can use COSX exchange but whose
/// gradient path has no COSX derivative wired in (UKS, ROHF, ROKS): an error,
/// never an exact-exchange gradient paired with a COSX energy.
pub fn unsupported_reference_error(reference: &str) -> FerricError {
    FerricError::General(format!(
        "k_builder = \"cosx\": the COSX exchange gradient is implemented for RHF, RKS and UHF \
         only, not {reference}; the exact-exchange gradient would not be the derivative of the \
         COSX energy. Use k_builder = \"direct\" for {reference} gradient tasks."
    ))
}

/// Whether an SCF run with `config` builds its exchange with COSX — i.e.
/// whether its gradient must come from this module.
///
/// `k_builder = "cosx"` alone is not enough: the solvers IGNORE it (with a
/// warning) when density-fitted J/K is active, when the functional uses no
/// exact exchange, and for range-separated functionals. This mirrors those
/// rules so the gradient differentiates the exchange the SCF actually used:
///
/// * restricted (`solve_rhf`, `open_shell = false`): a functional
///   AUTO-DEFAULTS RI-J (and RI-K for a hybrid) unless the caller opts out with
///   `df_j_aux = Some("")` / `df_k_aux = Some("")`; any active DF skips COSX.
/// * open shell (`solve_uhf` / `solve_rohf`, `open_shell = true`): no
///   auto-default; a NON-EMPTY `df_j_aux` or `df_k_aux` skips COSX (`Some("")`
///   is the exact-J/K sentinel, filtered out by `fock_assembly::build_df_jk`).
///
/// A mismatch between this predicate and the solvers would show up as a
/// finite-difference failure in `tests/cosx_gradient.rs`, which runs the
/// dispatch end to end.
pub fn scf_exchange_is_cosx(
    config: &crate::rhf::RhfConfig,
    open_shell: bool,
) -> Result<bool, FerricError> {
    if config.k_builder.as_deref() != Some("cosx") {
        return Ok(false);
    }
    let has_xc = config.xc.is_some();
    let k_mix = match config.xc.as_deref() {
        None => ferric_dft::xc_trait::KMix::default(),
        Some(name) => ferric_dft::libxc::k_mix_from_xc_def(
            &ferric_dft::libxc::xc_def_from_name(name)
                .map_err(|e| FerricError::General(format!("libxc: {e:?}")))?,
        ),
    };
    // Exact exchange consumed from a K matrix at all (RSH exchange comes from
    // the SR/LR fitters and has no COSX form; the SCF refuses or skips it).
    let need_k = !has_xc || k_mix.sr != 0.0;
    if !need_k || k_mix.omega > 0.0 {
        return Ok(false);
    }
    let df_active = if open_shell {
        // `fock_assembly::build_df_jk` drops an empty name (`Some("")` is the
        // explicit exact-J/K sentinel), so only a NON-EMPTY name activates DF.
        let named = |aux: &Option<String>| aux.as_deref().is_some_and(|s| !s.is_empty());
        named(&config.df_j_aux) || named(&config.df_k_aux)
    } else {
        // `solve_rhf`'s `resolve_aux`: None = auto-default when needed,
        // Some("") = explicit opt-out, Some(name) = on.
        let requested = |aux: &Option<String>, auto: bool| match aux.as_deref() {
            None => auto,
            Some("") => false,
            Some(_) => true,
        };
        requested(&config.df_j_aux, has_xc) || requested(&config.df_k_aux, has_xc)
    };
    Ok(!df_active)
}

/// `d/dR sum_i c_i tr[D_i K_COSX(D_i)]` at FIXED densities — the EXPLICIT part.
///
/// `dens` pairs each (symmetric) AO density with its coefficient in the
/// energy. The exchange energy is `-1/4 c_x tr[D K(D)]` for a closed shell
/// (total `D`) and `-1/2 c_x sum_s tr[D_s K(D_s)]` for a spin-unrestricted or
/// restricted-open one, so callers pass `[(D, -c_x/4)]` or
/// `[(D_a, -c_x/2), (D_b, -c_x/2)]`. Returns `(natoms, 3)`.
///
/// With `cfg.overlap_fit = false` this IS the exchange gradient (the energy is
/// variational). With the fit it is only the explicit part (all of `dQ`
/// included); the exact fitted gradient adds the Z-vector terms — see
/// `fitted_exchange_response` and `gradient::rhf_gradient_cosx`.
pub fn cosx_exchange_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    cfg: &CosxConfig,
    dens: &[(&Array2<f64>, f64)],
) -> Result<Array2<f64>, FerricError> {
    let terms: Vec<(&Array2<f64>, &Array2<f64>, f64)> =
        dens.iter().map(|&(d, c)| (d, d, c)).collect();
    cosx_exchange_gradient_bilinear(mol, prep, cfg, &terms)
}

/// Explicit nuclear derivative of `sum_k c_k T(Y_k, B_k)` at fixed matrices,
///
/// ```text
///   T(Y, B) = tr[Y Q Kt(B)] = sum_g w_g (B phi_g)^T A^g (Y Q phi_g),
///   Q = S S_num^{-1} (overlap_fit)  or  I (fit off),
/// ```
///
/// i.e. `tr[Y K_COSX(B)]` for symmetric `Y`. `T(D, D)` is the exchange energy's
/// trace; `T(Zs, D)` is the fitted Z-vector's exchange term. Every `Y_k`, `B_k`
/// must be symmetric. With the fit, the `dQ` terms need `Kt(B_k)` over the
/// whole grid, so the routine makes one integral pass (AO values/gradients,
/// 3c1e values and derivatives, Becke weight response) and one cheap AO-only
/// pass for `d S_num`.
pub fn cosx_exchange_gradient_bilinear(
    mol: &Molecule,
    prep: &PreparedBasis,
    cfg: &CosxConfig,
    terms: &[(&Array2<f64>, &Array2<f64>, f64)],
) -> Result<Array2<f64>, FerricError> {
    cosx_exchange_gradient_bilinear_with_q(mol, prep, cfg, terms, FitQ::Configured)
}

/// `cosx_exchange_gradient_bilinear` with an explicit [`FitQ`]. With
/// `FitQ::Identity` it is the fit-off derivative evaluated through the general
/// (`H != F`) kernel — the anchor's view of the fitted path.
pub fn cosx_exchange_gradient_bilinear_with_q(
    mol: &Molecule,
    prep: &PreparedBasis,
    cfg: &CosxConfig,
    terms: &[(&Array2<f64>, &Array2<f64>, f64)],
    q: FitQ,
) -> Result<Array2<f64>, FerricError> {
    check_gradient_supported(cfg)?;
    let natoms = mol.atoms.len();
    let nbf = prep.nbasis();
    for (y, b, _) in terms {
        check_symmetric(y, nbf)?;
        check_symmetric(b, nbf)?;
    }
    let active: Vec<(&Array2<f64>, &Array2<f64>, f64)> =
        terms.iter().copied().filter(|t| t.2 != 0.0).collect();
    if active.is_empty() || natoms == 0 {
        return Ok(Array2::zeros((natoms, 3)));
    }
    let fit = cfg.overlap_fit && q == FitQ::Configured;
    let identity_path = q == FitQ::Identity;

    // The SAME points and weights `CosxK::new` builds (`build_atomic_grid` for
    // an unpruned config), plus d w_g / d R.
    let (grid, weight1) = build_atomic_grid_with_response(mol, &cfg.grid)?;
    let shells = collect_shells(mol, prep.basis_set())
        .map_err(|e| FerricError::General(format!("cosx_exchange_gradient AO shells: {e:?}")))?;
    let ao_atom = ao_to_atom(prep);

    // Fit: S, the Cholesky factor of S_num, and M_k = Y_k Q = (S_num^{-1} S Y_k)^T.
    let fitdata = if fit {
        let s = ferric_integrals::oneelectron::overlap(prep);
        let snum = numeric_overlap(&shells, nbf, &grid)?;
        let fac = SnumChol::new(&snum)?;
        Some((s, fac))
    } else {
        None
    };
    let ms: Vec<Option<Array2<f64>>> = active
        .iter()
        .map(|&(y, b, _)| -> Result<Option<Array2<f64>>, FerricError> {
            match &fitdata {
                Some((s, fac)) => Ok(Some(
                    fac.solve(&s.dot(y))?.t().as_standard_layout().to_owned(),
                )),
                // Plain: M = Y; `None` when Y and B are the same matrix (H = F),
                // except on the anchor's Identity path, which must exercise the
                // general kernel.
                None if std::ptr::eq(y, b) && !identity_path => Ok(None),
                None => Ok(Some(y.clone())),
            }
        })
        .collect::<Result<_, _>>()?;
    let kterms: Vec<KTerm> = active
        .iter()
        .zip(&ms)
        .map(|(&(_, b, c), m)| KTerm {
            b,
            m: m.as_ref(),
            c,
        })
        .collect();

    // Fit: the per-chunk Kt partials (a fixed group of them live at once),
    // their running sums and the S / S_num / P1 / P2 work matrices, declared
    // to the global memory pool like the rest of the gradient's planes.
    let _fit_guard = if fit {
        let mut plan = MemoryPlan::from_global_pool(None, "COSX fitted-gradient dQ terms");
        plan.reserve(
            "Kt partials + sums",
            (COSX_GRAD_CHUNK_GROUP + 1) * kterms.len() * nbf * nbf,
            Lifetime::Resident,
        );
        plan.reserve("S, S_num factor, P1, P2", 8 * nbf * nbf, Lifetime::Resident);
        Some(plan.commit()?)
    } else {
        None
    };

    // Pass 1: the integral pass, in fixed chunk groups folded in chunk order.
    let npts = grid.len();
    let starts: Vec<usize> = (0..npts).step_by(COSX_GRAD_CHUNK_POINTS).collect();
    let mut grad = Array2::<f64>::zeros((natoms, 3));
    let mut kts: Vec<Array2<f64>> = if fit {
        (0..kterms.len())
            .map(|_| Array2::zeros((nbf, nbf)))
            .collect()
    } else {
        Vec::new()
    };
    for group in starts.chunks(COSX_GRAD_CHUNK_GROUP) {
        let outs: Vec<ChunkOut> = group
            .par_iter()
            .map(|&g0| {
                let g1 = (g0 + COSX_GRAD_CHUNK_POINTS).min(npts);
                chunk_gradient(
                    prep, &shells, &grid, &weight1, &ao_atom, &kterms, fit, g0, g1,
                )
            })
            .collect::<Result<_, _>>()?;
        for o in outs {
            grad += &o.grad;
            for (acc, kt) in kts.iter_mut().zip(o.kt) {
                *acc += &kt;
            }
        }
    }

    // Fit: dQ terms. tr[dQ Mtot] with Mtot = sum_k c_k Kt(B_k) Y_k and
    // dQ = dS S_num^{-1} - S S_num^{-1} dS_num S_num^{-1}:
    //   + tr[dS P1],      P1 = S_num^{-1} Mtot
    //   - tr[dS_num P2],  P2 = S_num^{-1} Mtot S S_num^{-1}
    if let Some((s, fac)) = &fitdata {
        let mut mtot = Array2::<f64>::zeros((nbf, nbf));
        for (kt, &(y, _, c)) in kts.iter().zip(&active) {
            mtot.scaled_add(c, &kt.dot(y));
        }
        let p1 = fac.solve(&mtot)?;
        // P2 = (S_num^{-1} Mtot) (S S_num^{-1}) = P1 · (S_num^{-1} S)^T.
        let sns = fac.solve(s)?;
        let p2 = p1.dot(&sns.t());
        // tr[dS P1] = sum_{mu nu} dS_{mu nu} P1_{nu mu}.
        let ds = crate::gradient::overlap_deriv_contract(prep, &p1.t().to_owned())?;
        if ds.dim() != (natoms, 3) {
            return Err(FerricError::General(format!(
                "cosx_exchange_gradient: overlap derivative has shape {:?}, expected ({natoms}, 3)",
                ds.dim()
            )));
        }
        grad += &ds;
        grad -= &snum_derivative(&shells, nbf, natoms, &grid, &weight1, &ao_atom, &p2)?;
    }
    Ok(grad)
}

/// One density term of `cosx_exchange_gradient_bilinear`: `F = B phi`,
/// `H = M phi` with `M = Y Q` (`None`: `H = F`, the plain `Y == B` case).
struct KTerm<'a> {
    b: &'a Array2<f64>,
    m: Option<&'a Array2<f64>>,
    c: f64,
}

struct ChunkOut {
    grad: Array2<f64>,
    /// Per term, `sum_{g in chunk} w_g phi_g (A^g F_g)^T` (fit only).
    kt: Vec<Array2<f64>>,
}

fn check_symmetric(d: &Array2<f64>, nbf: usize) -> Result<(), FerricError> {
    if d.dim() != (nbf, nbf) {
        return Err(FerricError::General(format!(
            "cosx_exchange_gradient: matrix shape {:?} != ({nbf}, {nbf})",
            d.dim()
        )));
    }
    let scale = d.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    let asym = d
        .iter()
        .zip(d.t().iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    if asym > DENSITY_SYMMETRY_TOL * scale {
        return Err(FerricError::General(format!(
            "cosx_exchange_gradient: matrix is not symmetric (max|D - D^T| = {asym:.3e})"
        )));
    }
    Ok(())
}

fn ao_to_atom(prep: &PreparedBasis) -> Vec<usize> {
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();
    let mut ao_atom = vec![0usize; prep.nbasis()];
    for s in 0..prep.nshells() {
        for i in 0..dims[s] {
            ao_atom[offs[s] + i] = sh2at[s];
        }
    }
    ao_atom
}

/// Cholesky factor of `S_num` with two-triangular-solve application of
/// `S_num^{-1}` (never an explicit inverse), as in `cosx_k::finalize_fitted`.
struct SnumChol {
    l: Array2<f64>,
    lt: Array2<f64>,
}

impl SnumChol {
    fn new(snum: &Array2<f64>) -> Result<Self, FerricError> {
        let l = snum.cholesky(UPLO::Lower).map_err(|e| {
            FerricError::General(format!(
                "COSX gradient: S_num is not positive definite on this grid ({e})"
            ))
        })?;
        let lt = l.t().as_standard_layout().to_owned();
        Ok(Self { l, lt })
    }

    /// `S_num^{-1} m`.
    fn solve(&self, m: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        let m = m.as_standard_layout().to_owned();
        let y = self
            .l
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &m)
            .map_err(|e| FerricError::General(format!("COSX gradient: S_num solve: {e}")))?;
        self.lt
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &y)
            .map_err(|e| FerricError::General(format!("COSX gradient: S_num solve: {e}")))
    }
}

/// `S_num = sum_g |w_g| phi_g phi_g^T` on `grid` — the fit's Gram matrix, as
/// `CosxK` forms it (`X X^T`, `X = sqrt|w| phi`).
fn numeric_overlap(
    shells: &[LocatedShell],
    nbf: usize,
    grid: &[GridPoint],
) -> Result<Array2<f64>, FerricError> {
    let npts = grid.len();
    let starts: Vec<usize> = (0..npts).step_by(COSX_GRAD_CHUNK_POINTS).collect();
    let mut snum = Array2::<f64>::zeros((nbf, nbf));
    for group in starts.chunks(COSX_GRAD_CHUNK_GROUP) {
        let parts: Vec<Array2<f64>> = group
            .par_iter()
            .map(|&g0| -> Result<Array2<f64>, FerricError> {
                let g1 = (g0 + COSX_GRAD_CHUNK_POINTS).min(npts);
                let pts: Vec<[f64; 3]> = grid[g0..g1].iter().map(|p| p.xyz).collect();
                let (chi, _) = eval_basis_and_grad_on_points_unchecked(shells, nbf, &pts)
                    .map_err(|e| FerricError::General(format!("COSX S_num AO eval: {e:?}")))?;
                let mut x = chi;
                for (j, p) in grid[g0..g1].iter().enumerate() {
                    let sw = p.weight.abs().sqrt();
                    x.column_mut(j).mapv_inplace(|v| v * sw);
                }
                Ok(x.dot(&x.t()))
            })
            .collect::<Result<_, _>>()?;
        for p in parts {
            snum += &p;
        }
    }
    Ok(snum)
}

/// `d/dR tr[S_num P2]` at fixed `P2`, `S_num = sum_g w_g phi_g phi_g^T` on the
/// moving grid: weight response + AO values riding the point (+grad, home) and
/// their own centre (-grad).
#[allow(clippy::too_many_arguments)]
fn snum_derivative(
    shells: &[LocatedShell],
    nbf: usize,
    natoms: usize,
    grid: &[GridPoint],
    weight1: &[Vec<[f64; 3]>],
    ao_atom: &[usize],
    p2: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let npts = grid.len();
    let p2s = p2 + &p2.t();
    let starts: Vec<usize> = (0..npts).step_by(COSX_GRAD_CHUNK_POINTS).collect();
    let mut grad = Array2::<f64>::zeros((natoms, 3));
    for group in starts.chunks(COSX_GRAD_CHUNK_GROUP) {
        let parts: Vec<Array2<f64>> = group
            .par_iter()
            .map(|&g0| -> Result<Array2<f64>, FerricError> {
                let g1 = (g0 + COSX_GRAD_CHUNK_POINTS).min(npts);
                let pts: Vec<[f64; 3]> = grid[g0..g1].iter().map(|p| p.xyz).collect();
                let (chi, dchi) = eval_basis_and_grad_on_points_unchecked(shells, nbf, &pts)
                    .map_err(|e| FerricError::General(format!("COSX dS_num AO eval: {e:?}")))?;
                let p2chi = p2.dot(&chi); // (nbf, n)
                let vchi = p2s.dot(&chi); // (P2 + P2^T) phi
                let mut local = Array2::<f64>::zeros((natoms, 3));
                for (j, gp) in grid[g0..g1].iter().enumerate() {
                    let (w, home) = (gp.weight, gp.home_atom);
                    let mut e2 = 0.0;
                    for mu in 0..nbf {
                        e2 += chi[(mu, j)] * p2chi[(mu, j)];
                    }
                    for (b, row) in weight1[g0 + j].iter().enumerate() {
                        for c in 0..3 {
                            local[(b, c)] += e2 * row[c];
                        }
                    }
                    for mu in 0..nbf {
                        let t = w * vchi[(mu, j)];
                        if t == 0.0 {
                            continue;
                        }
                        for c in 0..3 {
                            let v = t * dchi[(c, mu, j)];
                            local[(home, c)] += v;
                            local[(ao_atom[mu], c)] -= v;
                        }
                    }
                }
                Ok(local)
            })
            .collect::<Result<_, _>>()?;
        for p in parts {
            grad += &p;
        }
    }
    Ok(grad)
}

/// Install a single unit point charge at `r` (libint2 then evaluates `-1/|r - r_g|`).
fn set_probe(eng: &mut Engine, r: [f64; 3]) -> Result<(), FerricError> {
    let probe = [CAtom {
        atomic_number: 1.0,
        x: r[0],
        y: r[1],
        z: r[2],
    }];
    // SAFETY: `probe` is a stack-local CAtom slice alive for the call and
    // `handle_mut()` is the live engine pointer (same pattern as
    // `ferric_integrals::cosx_a::a_matrix_at_point_with`). The shim copies the
    // data and catches C++ exceptions, returning a negative status.
    let rc = unsafe {
        ffi::scf_engine_set_point_charges(eng.handle_mut(), probe.as_ptr(), probe.len() as c_int)
    };
    if rc < 0 {
        return Err(FerricError::General(format!(
            "cosx_exchange_gradient: set_point_charges failed (rc={rc})"
        )));
    }
    Ok(())
}

/// Integral-pass contribution of grid points `g0..g1` (see the module doc for
/// the three terms), for every term `c_k T(Y_k, B_k)`:
///
/// ```text
///   e_g = F^T A H,  F = B phi,  H = M phi (M = Y Q)
///   (a) AO:        d e/d phi = B (A H) + M^T (A F)   (= 2 B A F when H = F)
///   (b) integrals: sum_{lam sig} F_lam H_sig dA_{lam sig}
///   (c) weights:   e_g dw_g/dR
/// ```
#[allow(clippy::too_many_arguments)]
fn chunk_gradient(
    prep: &PreparedBasis,
    shells: &[LocatedShell],
    grid: &[GridPoint],
    weight1: &[Vec<[f64; 3]>],
    ao_atom: &[usize],
    terms: &[KTerm],
    fit: bool,
    g0: usize,
    g1: usize,
) -> Result<ChunkOut, FerricError> {
    let nbf = prep.nbasis();
    let natoms = weight1.first().map_or(0, |r| r.len());
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();
    let n = g1 - g0;
    let nk = terms.len();

    // Chunk-sized planes (4 · nbf · 128 doubles + 4 per term): not
    // budget-checked, like the per-block scratch of the other fixed-size grid
    // loops.
    let pts: Vec<[f64; 3]> = grid[g0..g1].iter().map(|p| p.xyz).collect();
    let (chi, dchi) = eval_basis_and_grad_on_points_unchecked(shells, nbf, &pts)
        .map_err(|e| FerricError::General(format!("cosx_exchange_gradient AO eval: {e:?}")))?;
    let fs: Vec<Array2<f64>> = terms
        .iter()
        .map(|t| t.b.dot(&chi).as_standard_layout().to_owned())
        .collect();
    let hs: Vec<Option<Array2<f64>>> = terms
        .iter()
        .map(|t| t.m.map(|m| m.dot(&chi).as_standard_layout().to_owned()))
        .collect();
    let mut gfs: Vec<Array2<f64>> = (0..nk).map(|_| Array2::zeros((nbf, n))).collect();
    let mut ghs: Vec<Option<Array2<f64>>> = hs
        .iter()
        .map(|h| h.as_ref().map(|_| Array2::zeros((nbf, n))))
        .collect();

    let mut veng = Engine::new_1e(ffi::OP_NUCLEAR, prep, COSX_GRAD_PRECISION)?;
    let mut deng = Engine::new_1e_deriv(ffi::OP_NUCLEAR, prep, COSX_GRAD_PRECISION)?;

    let mut local = Array2::<f64>::zeros((natoms, 3));
    let mut fcol = vec![vec![0.0_f64; nbf]; nk];
    let mut hcol = vec![vec![0.0_f64; nbf]; nk];
    let mut gfcol = vec![vec![0.0_f64; nbf]; nk];
    let mut ghcol = vec![vec![0.0_f64; nbf]; nk];

    for j in 0..n {
        let gp = &grid[g0 + j];
        let w = gp.weight;
        let home = gp.home_atom;
        set_probe(&mut veng, gp.xyz)?;
        set_probe(&mut deng, gp.xyz)?;
        for k in 0..nk {
            for mu in 0..nbf {
                fcol[k][mu] = fs[k][(mu, j)];
            }
            if let Some(h) = &hs[k] {
                for mu in 0..nbf {
                    hcol[k][mu] = h[(mu, j)];
                }
            }
            gfcol[k].fill(0.0);
            ghcol[k].fill(0.0);
        }

        for s1 in 0..nsh {
            let (n1, o1, a1) = (dims[s1], offs[s1], sh2at[s1]);
            for s2 in 0..=s1 {
                let (n2, o2, a2) = (dims[s2], offs[s2], sh2at[s2]);
                let bsz = n1 * n2;
                let off = s1 != s2;

                // Values: A = -V (libint2's unit probe charge is attractive).
                // A F and A H, both orientations of the (s1, s2) block.
                {
                    let blk = veng.compute_1e_block(prep, s1, s2);
                    if blk.len() < bsz {
                        return Err(FerricError::General(format!(
                            "cosx_exchange_gradient: 3c1e value block ({s1},{s2}) has {} values, \
                             expected {bsz}",
                            blk.len()
                        )));
                    }
                    for k in 0..nk {
                        let has_h = hs[k].is_some();
                        let (f, h) = (&fcol[k], &hcol[k]);
                        let (gf, gh) = (&mut gfcol[k], &mut ghcol[k]);
                        for i in 0..n1 {
                            for jj in 0..n2 {
                                let a = -blk[i * n2 + jj];
                                gf[o1 + i] += a * f[o2 + jj];
                                if off {
                                    gf[o2 + jj] += a * f[o1 + i];
                                }
                                if has_h {
                                    gh[o1 + i] += a * h[o2 + jj];
                                    if off {
                                        gh[o2 + jj] += a * h[o1 + i];
                                    }
                                }
                            }
                        }
                    }
                }

                // (b) integral derivatives: blocks [bra xyz, ket xyz, charge xyz]
                // of dV = -dA. The charge moves with the home atom.
                if let Some(der) = deng.compute_1e_deriv_block_n(prep, s1, s2, 1) {
                    if der.len() < 9 * bsz {
                        return Err(FerricError::General(format!(
                            "cosx_exchange_gradient: nuclear deriv block ({s1},{s2}) has {} \
                             values, expected {} (9 blocks x {bsz})",
                            der.len(),
                            9 * bsz
                        )));
                    }
                    for (k, t) in terms.iter().enumerate() {
                        let f = &fcol[k];
                        let h = if hs[k].is_some() { &hcol[k] } else { &fcol[k] };
                        let pref = w * t.c;
                        for i in 0..n1 {
                            for jj in 0..n2 {
                                // F_lam H_sig over both orientations of the block.
                                let mut p = f[o1 + i] * h[o2 + jj];
                                if off {
                                    p += f[o2 + jj] * h[o1 + i];
                                }
                                let p = pref * p;
                                let idx = i * n2 + jj;
                                for c in 0..3 {
                                    local[(a1, c)] -= p * der[c * bsz + idx];
                                    local[(a2, c)] -= p * der[(3 + c) * bsz + idx];
                                    local[(home, c)] -= p * der[(6 + c) * bsz + idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        for k in 0..nk {
            for mu in 0..nbf {
                gfs[k][(mu, j)] = gfcol[k][mu];
            }
            if let Some(gh) = ghs[k].as_mut() {
                for mu in 0..nbf {
                    gh[(mu, j)] = ghcol[k][mu];
                }
            }
        }
    }

    let mut kt = Vec::with_capacity(if fit { nk } else { 0 });
    for (k, t) in terms.iter().enumerate() {
        let gf = &gfs[k];
        // A H (= A F when H = F).
        let gh = ghs[k].as_ref().unwrap_or(gf);
        // d e_g / d phi: B (A H) + M^T (A F); 2 B (A F) when H = F.
        let dg = match t.m {
            Some(m) => t.b.dot(gh) + m.t().dot(gf),
            None => t.b.dot(gf) * 2.0,
        };
        for j in 0..n {
            let gp = &grid[g0 + j];
            let (w, home) = (gp.weight, gp.home_atom);
            // (c) weight response: c e_g dw_g/dR, e_g = F . (A H) = H . (A F).
            let mut e = 0.0;
            for mu in 0..nbf {
                e += fs[k][(mu, j)] * gh[(mu, j)];
            }
            for (b, row) in weight1[g0 + j].iter().enumerate() {
                for c in 0..3 {
                    local[(b, c)] += t.c * e * row[c];
                }
            }
            // (a) AO term: the point drags phi (+grad on home), the centre
            // drags phi (-grad on atom(mu)).
            for mu in 0..nbf {
                let tv = w * t.c * dg[(mu, j)];
                if tv == 0.0 {
                    continue;
                }
                let am = ao_atom[mu];
                for c in 0..3 {
                    let v = tv * dchi[(c, mu, j)];
                    local[(home, c)] += v;
                    local[(am, c)] -= v;
                }
            }
        }
        if fit {
            // Kt partial: sum_j w_j phi_j (A F_j)^T.
            let mut chiw = chi.clone();
            for (j, gp) in grid[g0..g1].iter().enumerate() {
                let w = gp.weight;
                chiw.column_mut(j).mapv_inplace(|v| v * w);
            }
            kt.push(chiw.dot(&gf.t()));
        }
    }
    Ok(ChunkOut { grad: local, kt })
}

/// Converged orbitals of one spin channel (RHF: the single, doubly occupied
/// channel with `d` the TOTAL density; UHF: one spin, `d` = `D_s`).
pub struct SpinOrbitals<'a> {
    /// MO coefficients `(nbf, nmo)`, canonical (the Fock matrix is diagonal).
    pub c: &'a Array2<f64>,
    /// Orbital energies (length `nmo`).
    pub eps: &'a [f64],
    /// Occupied count (the first `nocc` columns).
    pub nocc: usize,
    /// The density this channel's exchange is built from.
    pub d: &'a Array2<f64>,
}

/// Solution of the fitted-COSX Z-vector problem, in AO form.
#[derive(Debug, Clone)]
pub struct FittedResponse {
    /// `Zs_s = sym(C_v z_s C_o^T)` per spin channel (one for RHF).
    pub zs: Vec<Array2<f64>>,
    /// The Lagrangian energy-weighted density `W` (replaces the usual
    /// `sum w_o C_o e_o C_o^T`; spin-summed for UHF).
    pub w: Array2<f64>,
    /// GMRES iterations and final relative residual.
    pub iterations: usize,
    pub rel_residual: f64,
}

/// The orbital-response (Z-vector) solve that makes the overlap-fitted COSX
/// Hartree–Fock gradient exact (module doc, "The overlap fit").
///
/// `spins` has ONE entry for RHF (`d` = total density, doubly occupied) and
/// TWO for UHF. `c_x` is the exact-exchange fraction (1 for HF). The Fock
/// response uses exact four-centre Coulomb (`DirectJ`, as the SCF's `J`) and
/// the COSX adjoint `L(Y) = sym(Kt_builder(Y Q))` on the SAME grid, unscreened
/// and dense (the gradient differentiates the unscreened energy; the screens'
/// footprint is measured in `tests/cosx_gradient.rs`).
///
/// The response operator is NOT symmetric (`L != K_f`), so the linear solve is
/// restarted GMRES with the orbital-energy-gap diagonal as right
/// preconditioner, not CG. Each GMRES iteration costs one exact `J` build and
/// one (per spin) COSX `L` build; memory: the Krylov basis
/// `(restart + 2) * n_vo` plus ~16 `nbf²` matrices, declared to the global
/// memory pool, plus the builders' own (budget-checked) block scratch.
pub fn fitted_exchange_response(
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    cfg: &CosxConfig,
    spins: &[SpinOrbitals],
    c_x: f64,
) -> Result<FittedResponse, FerricError> {
    fitted_exchange_response_with_q(mol, prep, bounds, cfg, spins, c_x, FitQ::Configured)
}

/// `fitted_exchange_response` with an explicit [`FitQ`]. `FitQ::Identity`
/// builds `K_f` and `L` both with `Q = I` (the plain builder), so `Delta`,
/// the right-hand side and the solution are exactly zero and `W` must equal
/// the ordinary energy-weighted density — the fit's trivial-limit anchor.
#[allow(clippy::too_many_arguments)]
pub fn fitted_exchange_response_with_q(
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    cfg: &CosxConfig,
    spins: &[SpinOrbitals],
    c_x: f64,
    q: FitQ,
) -> Result<FittedResponse, FerricError> {
    let identity = q == FitQ::Identity;
    if !cfg.overlap_fit && !identity {
        return Err(FerricError::General(
            "fitted_exchange_response: the Z-vector exists only for overlap_fit = true".into(),
        ));
    }
    check_gradient_supported(cfg)?;
    let restricted = match spins.len() {
        1 => true,
        2 => false,
        n => {
            return Err(FerricError::General(format!(
                "fitted_exchange_response: expected 1 (RHF) or 2 (UHF) spin channels, got {n}"
            )))
        }
    };
    // RHF: p = 4, c_L = c/2, Delta = (c/4)(K_f - L)(D), W occ weight 2.
    // UHF: p = 2, c_L = c,   Delta = (c/2)(K_f - L)(D_s), W occ weight 1.
    let (pz, cl, cd, wocc) = if restricted {
        (4.0, 0.5 * c_x, 0.25 * c_x, 2.0)
    } else {
        (2.0, c_x, 0.5 * c_x, 1.0)
    };
    let nbf = prep.nbasis();
    for sp in spins {
        check_symmetric(sp.d, nbf)?;
        if sp.c.nrows() != nbf || sp.eps.len() != sp.c.ncols() || sp.nocc > sp.c.ncols() {
            return Err(FerricError::General(format!(
                "fitted_exchange_response: inconsistent orbitals (C {:?}, {} eps, nocc {})",
                sp.c.dim(),
                sp.eps.len(),
                sp.nocc
            )));
        }
    }
    let blocks: Vec<(Array2<f64>, Array2<f64>, Vec<f64>, Vec<f64>)> = spins
        .iter()
        .map(|sp| {
            let co = sp.c.slice(ndarray::s![.., ..sp.nocc]).to_owned();
            let cv = sp.c.slice(ndarray::s![.., sp.nocc..]).to_owned();
            (
                co,
                cv,
                sp.eps[..sp.nocc].to_vec(),
                sp.eps[sp.nocc..].to_vec(),
            )
        })
        .collect();
    let sizes: Vec<usize> = blocks.iter().map(|b| b.0.ncols() * b.1.ncols()).collect();
    let nvec: usize = sizes.iter().sum();

    let mut plan = MemoryPlan::from_global_pool(None, "COSX fitted-gradient Z-vector");
    plan.reserve(
        "GMRES Krylov basis",
        (ZVEC_RESTART + 2) * nvec,
        Lifetime::Resident,
    );
    plan.reserve(
        "response nbf^2 matrices",
        16 * nbf * nbf,
        Lifetime::Resident,
    );
    let _guard = plan.commit()?;

    // Q pieces on the same grid as the energy.
    let grid = ferric_dft::grid::build_atomic_grid(mol, &cfg.grid);
    let shells = collect_shells(mol, prep.basis_set())
        .map_err(|e| FerricError::General(format!("COSX response AO shells: {e:?}")))?;
    let s = ferric_integrals::oneelectron::overlap(prep);
    let fac = SnumChol::new(&numeric_overlap(&shells, nbf, &grid)?)?;
    // Y Q = Y S S_num^{-1} = (S_num^{-1} S Y)^T for symmetric Y.
    let times_q = |y: &Array2<f64>| -> Result<Array2<f64>, FerricError> {
        if identity {
            return Ok(y.clone());
        }
        Ok(fac.solve(&s.dot(y))?.t().as_standard_layout().to_owned())
    };

    let ctx = ParallelContext::default();
    let exact = CosxConfig {
        screen_thresh: None,
        half_transform: CosxHalfTransform::Dense,
        ..cfg.clone()
    };
    // K_f: the fitted builder (Identity: the plain one, so K_f == L exactly).
    let mut kf_b = CosxK::new(
        &ctx,
        mol,
        prep,
        CosxConfig {
            overlap_fit: !identity,
            ..exact.clone()
        },
        0,
    )?;
    let mut l_b = CosxK::new(
        &ctx,
        mol,
        prep,
        CosxConfig {
            overlap_fit: false,
            ..exact
        },
        0,
    )?;
    let mut jb = crate::direct_j::DirectJ::new(
        &ctx,
        prep,
        bounds,
        ZVEC_J_THRESH,
        ferric_core::memory::resolve_budget_bytes(None),
    );

    let mut l_of = |y: &Array2<f64>| -> Result<Array2<f64>, FerricError> {
        let mut out = Array2::<f64>::zeros((nbf, nbf));
        l_b.build(&times_q(y)?, &mut out)?;
        Ok(out)
    };

    // Delta_s = cd (K_f(D_s) - L(D_s)) and the right-hand side -p C_v^T Delta C_o.
    let mut deltas = Vec::with_capacity(spins.len());
    for sp in spins {
        let mut kf = Array2::<f64>::zeros((nbf, nbf));
        kf_b.build(sp.d, &mut kf)?;
        let l = l_of(sp.d)?;
        deltas.push((kf - l) * cd);
    }
    let mut rhs = Vec::with_capacity(nvec);
    for ((co, cv, _, _), delta) in blocks.iter().zip(&deltas) {
        let r = cv.t().dot(&delta.dot(co)) * (-pz);
        rhs.extend(r.iter().copied());
    }
    let mut diag = Vec::with_capacity(nvec);
    for (_, _, eo, ev) in &blocks {
        for a in ev {
            for i in eo {
                diag.push(a - i);
            }
        }
    }

    let unpack = |v: &[f64]| -> Vec<Array2<f64>> {
        let mut out = Vec::with_capacity(blocks.len());
        let mut o = 0;
        for (co, cv, _, _) in &blocks {
            let (nv, no) = (cv.ncols(), co.ncols());
            out.push(
                Array2::from_shape_vec((nv, no), v[o..o + nv * no].to_vec())
                    .expect("slice length is nv * no"),
            );
            o += nv * no;
        }
        out
    };
    let zsym = |zz: &[Array2<f64>]| -> Vec<Array2<f64>> {
        blocks
            .iter()
            .zip(zz)
            .map(|((co, cv, _, _), z)| {
                let m = cv.dot(&z.dot(&co.t()));
                (&m + &m.t()) * 0.5
            })
            .collect()
    };
    // R_s(Zs) = J(sum_t Zs_t) - c_L L(Zs_s).
    let mut response = |zs: &[Array2<f64>]| -> Result<Vec<Array2<f64>>, FerricError> {
        let mut zt = Array2::<f64>::zeros((nbf, nbf));
        for z in zs {
            zt += z;
        }
        let mut j = Array2::<f64>::zeros((nbf, nbf));
        jb.build(&zt, &mut j)?;
        zs.iter()
            .map(|z| -> Result<Array2<f64>, FerricError> {
                let l = l_of(z)?;
                Ok(&j - &(l * cl))
            })
            .collect()
    };

    let mut apply = |v: &[f64]| -> Result<Vec<f64>, FerricError> {
        let zz = unpack(v);
        let rr = response(&zsym(&zz))?;
        let mut out = Vec::with_capacity(nvec);
        let mut o = 0;
        for ((co, cv, _, _), r) in blocks.iter().zip(&rr) {
            let t = cv.t().dot(&r.dot(co));
            for (idx, tv) in t.iter().enumerate() {
                out.push(diag[o + idx] * v[o + idx] + pz * tv);
            }
            o += t.len();
        }
        Ok(out)
    };

    let (zvec, iterations, rel_residual) = gmres(&mut apply, &rhs, &diag)?;

    let zz = unpack(&zvec);
    let zs = zsym(&zz);
    let rr = response(&zs)?;
    let mut w = Array2::<f64>::zeros((nbf, nbf));
    for ((((co, cv, eo, _), delta), r), z) in blocks.iter().zip(&deltas).zip(&rr).zip(&zz) {
        // w_o C_o (e_o + Delta_oo + R_oo) C_o^T
        let mut occ = co.t().dot(&(delta + r).dot(co));
        for (i, e) in eo.iter().enumerate() {
            occ[(i, i)] += e;
        }
        w.scaled_add(wocc, &co.dot(&occ.dot(&co.t())));
        // sym(C_v (z e_o) C_o^T)
        let mut ze = z.clone();
        for (i, e) in eo.iter().enumerate() {
            ze.column_mut(i).mapv_inplace(|v| v * e);
        }
        let m = cv.dot(&ze.dot(&co.t()));
        w.scaled_add(0.5, &m);
        w.scaled_add(0.5, &m.t());
    }
    Ok(FittedResponse {
        zs,
        w,
        iterations,
        rel_residual,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Restarted GMRES(`ZVEC_RESTART`) for `A x = b`, right-preconditioned with
/// `diag` (the orbital-energy gaps). Hard error if `ZVEC_RTOL` is not reached
/// in `ZVEC_MAX_ITER` iterations — an unconverged Z-vector is an inexact
/// gradient. Returns `(x, iterations, relative residual)`.
fn gmres(
    apply: &mut dyn FnMut(&[f64]) -> Result<Vec<f64>, FerricError>,
    b: &[f64],
    diag: &[f64],
) -> Result<(Vec<f64>, usize, f64), FerricError> {
    let n = b.len();
    let bnorm = norm(b);
    if bnorm == 0.0 {
        return Ok((vec![0.0; n], 0, 0.0));
    }
    // Padded (lindep-removed) columns carry a sentinel gap; any non-positive
    // gap would mean an unconverged or non-aufbau reference.
    if let Some(bad) = diag.iter().find(|&&d| d <= 0.0 || d.is_nan()) {
        return Err(FerricError::General(format!(
            "COSX fitted-gradient Z-vector: non-positive orbital gap {bad:.3e} (non-aufbau or \
             unconverged reference)"
        )));
    }
    let precond = |y: &[f64]| -> Vec<f64> { y.iter().zip(diag).map(|(v, d)| v / d).collect() };
    let m = ZVEC_RESTART;
    let mut x = vec![0.0; n];
    let mut total = 0usize;
    loop {
        let ax = apply(&x)?;
        let r: Vec<f64> = b.iter().zip(&ax).map(|(bi, ai)| bi - ai).collect();
        let beta = norm(&r);
        let relres = beta / bnorm;
        if relres < ZVEC_RTOL {
            return Ok((x, total, relres));
        }
        if total >= ZVEC_MAX_ITER {
            return Err(FerricError::General(format!(
                "COSX fitted-gradient Z-vector did not converge: relative residual {relres:.3e} \
                 after {total} GMRES iterations (target {ZVEC_RTOL:e})"
            )));
        }
        let mut v: Vec<Vec<f64>> = vec![r.iter().map(|ri| ri / beta).collect()];
        let mut h = vec![vec![0.0_f64; m]; m + 1];
        let (mut cs, mut sn) = (vec![0.0_f64; m], vec![0.0_f64; m]);
        let mut g = vec![0.0_f64; m + 1];
        g[0] = beta;
        let mut used = 0usize;
        for k in 0..m {
            if total >= ZVEC_MAX_ITER {
                break;
            }
            total += 1;
            used = k + 1;
            let mut w = apply(&precond(&v[k]))?;
            for i in 0..=k {
                h[i][k] = dot(&w, &v[i]);
                let hik = h[i][k];
                for (wj, vj) in w.iter_mut().zip(&v[i]) {
                    *wj -= hik * vj;
                }
            }
            h[k + 1][k] = norm(&w);
            let breakdown = h[k + 1][k] <= 1e-14 * beta;
            if !breakdown {
                let inv = 1.0 / h[k + 1][k];
                v.push(w.iter().map(|wj| wj * inv).collect());
            }
            for i in 0..k {
                let t = cs[i] * h[i][k] + sn[i] * h[i + 1][k];
                h[i + 1][k] = -sn[i] * h[i][k] + cs[i] * h[i + 1][k];
                h[i][k] = t;
            }
            let den = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
            if den > 0.0 {
                cs[k] = h[k][k] / den;
                sn[k] = h[k + 1][k] / den;
                h[k][k] = den;
                h[k + 1][k] = 0.0;
                g[k + 1] = -sn[k] * g[k];
                g[k] *= cs[k];
            }
            if g[k + 1].abs() / bnorm < ZVEC_RTOL || breakdown {
                break;
            }
        }
        // Back-substitute H y = g, then x += M^{-1} V y.
        let mut y = vec![0.0_f64; used];
        for i in (0..used).rev() {
            let mut s = g[i];
            for j in i + 1..used {
                s -= h[i][j] * y[j];
            }
            y[i] = if h[i][i] != 0.0 { s / h[i][i] } else { 0.0 };
        }
        let mut upd = vec![0.0_f64; n];
        for (yi, vi) in y.iter().zip(&v) {
            for (u, vv) in upd.iter_mut().zip(vi) {
                *u += yi * vv;
            }
        }
        for (xi, ui) in x.iter_mut().zip(precond(&upd)) {
            *xi += ui;
        }
    }
}
