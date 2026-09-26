//! Analytic RHF second derivatives (Hessian).
//!
//! The full (3N × 3N) Cartesian Hessian, Hartree/Bohr², row/column index
//! `3·atom + xyz`, is assembled from five terms, following PySCF
//! `hessian/rhf.py` term by term (ferric conventions: Bohr, closed-shell
//! `D = 2 C_occ C_occᵀ`, `W = 2 Σ_i ε_i C_i C_iᵀ`, `G(D) = J(D) − ½K(D)`):
//!
//! | term | formula | PySCF |
//! |---|---|---|
//! | 1 nuclear repulsion | `∂²V_nn/∂x∂y` | `hess_nuc` |
//! | 2 skeleton 1e | `Σ D_μν ∂²(T+V)_μν/∂x∂y`, V differentiated w.r.t. both basis centres AND every nucleus | `_partial_hess_ejk` (`hcore_deriv` part of `e1`) |
//! | 3 overlap | `−Σ W_μν ∂²S_μν/∂x∂y` | `_partial_hess_ejk` (`s1aa`/`s1ab` part of `e1`) |
//! | 4 skeleton 2e | `Σ Γ_μνλσ ∂²(μν|λσ)/∂x∂y`, `Γ = ½D_μνD_λσ − ¼D_μλD_νσ` | `_partial_hess_ejk` (`ej − ek`) |
//! | 5 response | `4 Σ_pi F^x_pi U^y_pi − 4 Σ_pi S^x_pi ε_i U^y_pi − 2 Σ_ij S^x_ij ε^y_ij` | `hess_elec` |
//!
//! Terms 1–4 are the "skeleton": the second derivative at FIXED AO density
//! and energy-weighted density. Term 5 is the orbital relaxation, with MO-basis
//! first-order matrices `F^x = Cᵀ (∂h/∂x + G^x(D)) C_occ` (`make_h1`: the
//! first-derivative Fock matrix at fixed D, `G^x` from ERI first derivatives),
//! `S^x = Cᵀ ∂S/∂x C_occ`, and the coupled-perturbed HF solution (`solve_mo1`
//! → `cphf.solve_withs1`):
//!
//! ```text
//!   U^x_ij = −½ S^x_ij                                    (occupied block, fixed)
//!   (ε_a − ε_i) U^x_ai + G(D1(U^x))_ai = −(F^x_ai − ε_i S^x_ai + G(D_oo^x)_ai)
//!   D1(U) = 2 (C_v U_vo C_oᵀ + h.c.),   D_oo^x = −2 C_o S^x_oo C_oᵀ
//!   ε^x_ij = F^x_ij + G(D^x)_ij − ½ S^x_ij (ε_i + ε_j),  D^x = D_oo^x + D1(U^x_vo)
//! ```
//!
//! solved per perturbation by preconditioned conjugate gradient on the
//! (symmetric positive definite at a stable RHF minimum) orbital Hessian, with
//! `G` from the exact four-centre [`crate::rhf::build_jk_with_pool`].
//!
//! Scope: closed-shell RHF, exact four-centre Coulomb J/K. Everything else —
//! open shells, KS, RI, COSX, ECPs, ghost atoms, external potentials,
//! solvation, polarizable embedding, fractional occupations, cDFT, non-Coulomb
//! operators — is refused with a typed error naming the feature (the FD
//! Hessian in [`crate::frequencies::harmonic_frequencies`] covers those).
//!
//! Second-derivative integrals need libint2 generated with
//! `LIBINT2_MAX_DERIV_ORDER >= 2` (conda-forge 2.13.1: AM ≤ 3, i.e. up to f
//! functions). Against a first-derivative libint2 the engine constructors
//! return a typed error; nothing panics.

use crate::gradient::{
    build_energy_weighted_density, gamma, screened_quartets_with, sum_equivalent_perms,
    QuartetBlock,
};
use crate::result::{ScfResult, Spin};
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::{deriv2_pair_index, Engine};
use ferric_integrals::engine_pool::EnginePool;
use ferric_integrals::ffi;
use ferric_integrals::operator::{Operator, OperatorKind};
use ndarray::{s, Array1, Array2, ArrayView2};

/// Skeleton 2e screen: a canonical quartet is skipped when
/// `Q12·Q34·max|D|² < HESS_2E_SCREEN`. Conservative on purpose: the Schwarz
/// bound bounds the integral, not its second derivative, which carries extra
/// exponent factors.
const HESS_2E_SCREEN: f64 = 1e-15;

/// First-derivative Fock screen: skip when `Q12·Q34·max|D| < FOCK1_SCREEN`.
const FOCK1_SCREEN: f64 = 1e-14;

/// Integral threshold handed to `build_jk_with_pool` for the CPHF `G(D)` builds.
const CPHF_JK_THRESH: f64 = 1e-14;

/// CPHF convergence: max-abs residual of `A·U + b` (Hartree/Bohr units of the
/// right-hand side). The response term is linear in `U`, so the Hessian error
/// it leaves is ~`CPHF_TOL · |F^x| / gap`.
pub const CPHF_TOL: f64 = 1e-10;

/// CPHF iteration cap per perturbation; exceeding it is a typed error.
pub const CPHF_MAX_ITER: usize = 200;

/// Per-term breakdown of the analytic RHF Hessian. Every matrix is 3N × 3N,
/// Hartree/Bohr², index `3·atom + xyz`.
#[derive(Debug, Clone)]
pub struct RhfHessianParts {
    /// Term 1, `∂²V_nn`.
    pub nuclear: Array2<f64>,
    /// Term 2, `Σ D·∂²(T + V)` including the nuclear-centre derivatives of V.
    pub one_electron: Array2<f64>,
    /// Term 3, `−Σ W·∂²S`.
    pub overlap: Array2<f64>,
    /// Term 4, `Σ Γ(D)·∂²(μν|λσ)`.
    pub two_electron: Array2<f64>,
    /// Term 5, the CPHF orbital response, symmetrized.
    pub response: Array2<f64>,
    /// `max |R − Rᵀ|` of the response term BEFORE symmetrization. Terms 1–4 are
    /// symmetric by construction; the response is not (PySCF assembles one
    /// triangle and mirrors it), so this is zero only in exact arithmetic with
    /// a converged CPHF and is the analytic Hessian's internal consistency probe.
    pub response_asymmetry: f64,
    /// CPHF iterations per perturbation (length 3N).
    pub cphf_iterations: Vec<usize>,
    /// Largest final CPHF residual over all perturbations.
    pub cphf_max_residual: f64,
}

impl RhfHessianParts {
    /// Terms 1–4: the second derivative at fixed D and W.
    pub fn skeleton(&self) -> Array2<f64> {
        &self.nuclear + &self.one_electron + &self.overlap + &self.two_electron
    }

    /// The full Hessian, terms 1–5.
    pub fn total(&self) -> Array2<f64> {
        self.skeleton() + &self.response
    }
}

/// Compute the full analytic RHF Hessian, (3N, 3N) in Hartree/Bohr².
///
/// `bounds` must be the Schwarz table for `op` on `prep`; `config` is the
/// configuration the SCF `rhf` was converged with (it is read only to refuse
/// unsupported features). See the module doc for scope.
pub fn rhf_hessian(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    config: &RhfConfig,
) -> Result<Array2<f64>, FerricError> {
    Ok(rhf_hessian_parts(ctx, mol, prep, op, bounds, rhf, config)?.total())
}

/// [`rhf_hessian`] with the per-term breakdown and CPHF diagnostics.
pub fn rhf_hessian_parts(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    config: &RhfConfig,
) -> Result<RhfHessianParts, FerricError> {
    check_supported(mol, prep, op, bounds, rhf, config)?;
    let natoms = mol.atoms.len();
    let nocc = (mol.nelec() / 2) as usize;
    let d = rhf.density_r();
    let w = build_energy_weighted_density(rhf, nocc);

    let nuclear = hess_nuclear_repulsion(mol);
    let one_electron = skeleton_hess_1e(prep, natoms, d)?;
    let overlap = skeleton_hess_overlap(prep, natoms, &w)?;
    let two_electron = skeleton_hess_2e(prep, op, bounds, natoms, d)?;

    let first = first_order_ao(prep, op, bounds, natoms, d)?;
    let resp = cpks_response(ctx, prep, bounds, rhf, nocc, &first)?;
    let (response, response_asymmetry) = symmetrize(&resp.raw);

    Ok(RhfHessianParts {
        nuclear,
        one_electron,
        overlap,
        two_electron,
        response,
        response_asymmetry,
        cphf_iterations: resp.iterations,
        cphf_max_residual: resp.max_residual,
    })
}

/// Refuse, before any SCF, a configuration or operator the analytic RHF
/// Hessian does not implement. [`rhf_hessian`] repeats this and adds the
/// checks that need the molecule, basis and converged result.
pub fn rhf_hessian_preflight(op: Operator, config: &RhfConfig) -> Result<(), FerricError> {
    if let Some(what) = operator_refusal(op) {
        return Err(refusal(&what));
    }
    if let Some(what) = config_refusal(config)? {
        return Err(refusal(what));
    }
    Ok(())
}

fn refusal(what: &str) -> FerricError {
    FerricError::General(format!(
        "analytic RHF Hessian: {what} is not supported (closed-shell RHF with exact \
         four-centre Coulomb J/K only); use ferric_scf::frequencies::harmonic_frequencies, \
         which differentiates analytic gradients numerically"
    ))
}

fn check_supported(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    config: &RhfConfig,
) -> Result<(), FerricError> {
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(refusal("an open-shell (UHF/ROHF) reference"));
    }
    if !rhf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        });
    }
    rhf_hessian_preflight(op, config)?;
    if bounds.op != op {
        return Err(refusal("Schwarz bounds built for a different operator"));
    }
    if let Some(what) = system_refusal(mol, prep) {
        return Err(refusal(what));
    }
    if crate::gradient::active_df_route(rhf).is_some() {
        return Err(refusal("a density-fitted SCF (RI J/K)"));
    }
    Ok(())
}

/// Whether [`rhf_hessian`] can run for this molecule, basis, operator and SCF
/// configuration, decided BEFORE any SCF: the configuration checks of
/// [`rhf_hessian_preflight`], the system checks (ghost atoms, ECPs, odd
/// electron count), and that libint2 can build the second-derivative engines
/// for this basis — which fails for a library generated without second
/// derivatives (the mpqc4 2.7.2 export) and for shells above the library's
/// second-derivative angular momentum (f for conda-forge 2.13.1), since libint2
/// checks the basis's maximum L when it constructs an engine.
pub fn analytic_hessian_available(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    config: &RhfConfig,
) -> Result<(), FerricError> {
    rhf_hessian_preflight(op, config)?;
    if mol.multiplicity != 1 {
        return Err(refusal("an open-shell (multiplicity > 1) molecule"));
    }
    if let Some(what) = system_refusal(mol, prep) {
        return Err(refusal(what));
    }
    let engines = Engine::new_1e_deriv2(ffi::OP_NUCLEAR, prep, 1e-14)
        .and_then(|_| Engine::new_1e_deriv2(ffi::OP_OVERLAP, prep, 1e-14))
        .and_then(|_| Engine::new_2e_deriv2(op, prep, 1e-14));
    engines.map(|_| ()).map_err(|e| {
        FerricError::General(format!(
            "analytic RHF Hessian: libint2 cannot build second-derivative engines for \
             this basis (a library without second derivatives, or shells above its \
             second-derivative angular momentum): {e}"
        ))
    })
}

fn operator_refusal(op: Operator) -> Option<String> {
    (op.is_composite() || op.kind != OperatorKind::Coulomb)
        .then(|| format!("the {:?} two-electron operator", op.kind))
}

fn config_refusal(config: &RhfConfig) -> Result<Option<&'static str>, FerricError> {
    let named = |s: &Option<String>| s.as_deref().is_some_and(|v| !v.is_empty());
    let checks = [
        (config.xc.is_some(), "Kohn-Sham DFT (xc)"),
        (
            crate::cosx_gradient::scf_exchange_is_cosx(config, false)?,
            "COSX seminumerical exchange",
        ),
        (
            named(&config.df_j_aux) || named(&config.df_k_aux),
            "density-fitted J/K (RI)",
        ),
        (
            config
                .external_potential
                .as_ref()
                .is_some_and(|e| !e.is_empty()),
            "an external potential",
        ),
        (
            config.cosmo.is_some() || config.pcm.is_some(),
            "implicit solvation",
        ),
        (config.polarizable.is_some(), "polarizable embedding"),
        (
            config.smearing_sigma.is_some() || config.fractional_occ,
            "fractional occupations",
        ),
        (!config.constraints.is_empty(), "constrained DFT"),
    ];
    Ok(checks.iter().find(|(bad, _)| *bad).map(|(_, what)| *what))
}

fn system_refusal(mol: &Molecule, prep: &PreparedBasis) -> Option<&'static str> {
    let bs = prep.basis_set();
    let checks = [
        (mol.atoms.iter().any(|a| a.ghost), "ghost atoms"),
        (
            mol.atoms
                .iter()
                .any(|a| a.n_core_ecp > 0 || bs.ecp_for_element(a.z).is_some()),
            "an effective core potential",
        ),
        (mol.nelec() % 2 != 0, "an odd electron count"),
        (
            prep.atoms().len() != mol.atoms.len(),
            "a basis whose nuclear centres differ from the molecule's atoms",
        ),
    ];
    checks.iter().find(|(bad, _)| *bad).map(|(_, what)| *what)
}

/// Symmetrize `m`, returning `(½(m + mᵀ), max |m − mᵀ|)`.
fn symmetrize(m: &Array2<f64>) -> (Array2<f64>, f64) {
    let asym = (m - &m.t()).iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    (0.5 * (m + &m.t()), asym)
}

// ---------------------------------------------------------------------------
// Term 1: Nuclear repulsion second derivatives
// ---------------------------------------------------------------------------

/// Nuclear repulsion Hessian: `d²V_nn / dR_{A,x} dR_{B,y}`.
///
/// Shape: (3*natom, 3*natom). This is a pure geometry term — no integrals.
///
/// For A ≠ B:
///   d²(Z_A Z_B / r_AB) / dR_{Ax} dR_{By}
///     = Z_A Z_B * (−3 (R_Ax − R_Bx)(R_Ay − R_By) / r^5 + δ_{xy} / r^3)
///
/// Diagonal blocks (A = A) are the negative sum of all off-diagonal blocks.
pub fn hess_nuclear_repulsion(mol: &Molecule) -> Array2<f64> {
    let natoms = mol.atoms.len();
    let n3 = 3 * natoms;
    let mut h = Array2::<f64>::zeros((n3, n3));

    for i in 0..natoms {
        let ai = &mol.atoms[i];
        if ai.ghost {
            continue;
        }
        let za = ai.effective_z() as f64;
        let ri = [ai.x, ai.y, ai.zpos];

        for j in 0..natoms {
            if i == j {
                continue;
            }
            let aj = &mol.atoms[j];
            if aj.ghost {
                continue;
            }
            let zb = aj.effective_z() as f64;
            let rj = [aj.x, aj.y, aj.zpos];

            let d = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let r = r2.sqrt();
            let r3 = r * r2;
            let r5 = r3 * r2;
            let zz = za * zb;

            // Off-diagonal block (i, j)
            for x in 0..3 {
                for y in 0..3 {
                    let val = zz * (-3.0 * d[x] * d[y] / r5 + if x == y { 1.0 / r3 } else { 0.0 });
                    h[(3 * i + x, 3 * j + y)] = val;
                    // Diagonal block accumulation: h[i,i] -= h[i,j]
                    h[(3 * i + x, 3 * i + y)] -= val;
                }
            }
        }
    }

    h
}

// ---------------------------------------------------------------------------
// Shared scatter of unique second-derivative blocks
// ---------------------------------------------------------------------------

/// Add libint2's unique second-derivative values to the Cartesian Hessian.
///
/// `vals[deriv2_pair_index(ncoord, i, j)]` is `∂²X/∂q_i∂q_j` (i ≤ j) over
/// `ncoord = 3·centre_atoms.len()` coordinates, `q_{3k+a}` being axis `a` of
/// centre `k`, which sits on atom `centre_atoms[k]`. The Hessian element for
/// atoms (A, B) is the sum over every centre pair on (A, B) of BOTH orders,
/// so an off-diagonal `i < j` value goes to `(row_i, row_j)` and
/// `(row_j, row_i)` — which, when two centres share an atom, correctly lands
/// twice on the same element (`∂²/∂A² f(A₁,A₂)|_{A₁=A₂=A} = f₁₁ + 2f₁₂ + f₂₂`).
fn scatter_unique_pairs(h: &mut Array2<f64>, vals: &[f64], centre_atoms: &[usize]) {
    let ncoord = 3 * centre_atoms.len();
    debug_assert_eq!(vals.len(), ncoord * (ncoord + 1) / 2);
    let row = |q: usize| 3 * centre_atoms[q / 3] + q % 3;
    for i in 0..ncoord {
        for j in i..ncoord {
            let v = vals[deriv2_pair_index(ncoord, i, j)];
            if v == 0.0 {
                continue;
            }
            let (r, c) = (row(i), row(j));
            h[(r, c)] += v;
            if i != j {
                h[(c, r)] += v;
            }
        }
    }
}

/// `out[b] = Σ_k weights[k] · blocks[b·n + k]` for every block `b`, `n = weights.len()`.
fn contract_blocks(blocks: &[f64], weights: &[f64]) -> Vec<f64> {
    let n = weights.len();
    blocks
        .chunks_exact(n)
        .map(|blk| blk.iter().zip(weights).map(|(x, w)| x * w).sum())
        .collect()
}

/// AO weights of one canonical shell pair (s1 ≥ s2): `w_μν`, plus `w_νμ` when
/// the pair stands for both orders (s1 ≠ s2).
fn pair_weights(
    w: &Array2<f64>,
    o1: usize,
    n1: usize,
    o2: usize,
    n2: usize,
    both: bool,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(n1 * n2);
    for i in 0..n1 {
        for j in 0..n2 {
            let (mu, nu) = (o1 + i, o2 + j);
            out.push(if both {
                w[(mu, nu)] + w[(nu, mu)]
            } else {
                w[(mu, nu)]
            });
        }
    }
    out
}

/// Number of point charges a 1e engine of `op_kind` differentiates (nuclear
/// only), and the centre→atom table for its coordinates: bra centre, ket
/// centre, then one centre per nucleus in `prep.atoms()` order (= molecule
/// order; [`system_refusal`] rejects a mismatch).
fn one_electron_centres(op_kind: std::os::raw::c_int, prep: &PreparedBasis) -> (usize, Vec<usize>) {
    let n_charges = if op_kind == ffi::OP_NUCLEAR {
        prep.atoms().len()
    } else {
        0
    };
    let mut centres = vec![0usize; 2 + n_charges];
    for (c, slot) in centres[2..].iter_mut().enumerate() {
        *slot = c;
    }
    (n_charges, centres)
}

// ---------------------------------------------------------------------------
// Terms 2 and 3: skeleton one-electron and overlap
// ---------------------------------------------------------------------------

/// Term 2: `Σ_μν D_μν ∂²(T + V)_μν/∂x∂y`.
///
/// The nuclear-attraction engine differentiates w.r.t. both basis-function
/// centres AND every nucleus (libint2 rebuilds the operator-centre blocks by
/// translational invariance, `engine.impl.h` Engine::compute1), so the
/// Hellmann-Feynman `∂²/∂R_C²` of `−Z_C/|r − R_C|` and the mixed basis/nucleus
/// terms are all included. PySCF: `hcore_deriv(ia, ja)` in `_partial_hess_ejk`.
fn skeleton_hess_1e(
    prep: &PreparedBasis,
    natoms: usize,
    d: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let mut h = contract_1e_deriv2(prep, natoms, ffi::OP_KINETIC, d)?;
    h += &contract_1e_deriv2(prep, natoms, ffi::OP_NUCLEAR, d)?;
    Ok(h)
}

/// Term 3: `−Σ_μν W_μν ∂²S_μν/∂x∂y`. PySCF: the `s1aa`/`s1ab` · `dme0` part
/// of `e1` in `_partial_hess_ejk`.
fn skeleton_hess_overlap(
    prep: &PreparedBasis,
    natoms: usize,
    w: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    Ok(-contract_1e_deriv2(prep, natoms, ffi::OP_OVERLAP, w)?)
}

/// `Σ_μν weight_μν ∂²O_μν/∂x∂y` for the 1e operator `op_kind`, over canonical
/// shell pairs.
fn contract_1e_deriv2(
    prep: &PreparedBasis,
    natoms: usize,
    op_kind: std::os::raw::c_int,
    weight: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let (n_charges, mut centres) = one_electron_centres(op_kind, prep);
    let mut eng = Engine::new_1e_deriv2(op_kind, prep, 1e-14)?;
    if n_charges > 0 {
        eng.set_point_charges(prep)?;
    }
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();
    let mut h = Array2::<f64>::zeros((3 * natoms, 3 * natoms));
    for s1 in 0..prep.nshells() {
        for s2 in 0..=s1 {
            centres[0] = sh2at[s1];
            centres[1] = sh2at[s2];
            let wv = pair_weights(weight, offs[s1], dims[s1], offs[s2], dims[s2], s1 != s2);
            let blocks = eng.compute_1e_deriv2_block(prep, s1, s2, n_charges);
            let vals = contract_blocks(blocks, &wv);
            scatter_unique_pairs(&mut h, &vals, &centres);
        }
    }
    Ok(h)
}

// ---------------------------------------------------------------------------
// Term 4: skeleton two-electron
// ---------------------------------------------------------------------------

/// Term 4: `Σ_μνλσ Γ_μνλσ ∂²(μν|λσ)/∂x∂y`, `Γ = ½D_μνD_λσ − ¼D_μλD_νσ`
/// (the same Γ as [`crate::gradient::twoelectron_gradient`], so this is the
/// derivative of that gradient term at fixed D). PySCF: `ej − ek` of
/// `_partial_hess_ejk` (int2e_ipip1 / ip1ip2 / ipvip1 contractions).
///
/// Serial canonical-quartet loop with a conservative Schwarz screen
/// ([`HESS_2E_SCREEN`] against `max|D|²`).
fn skeleton_hess_2e(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    natoms: usize,
    d: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let max_d = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let quads = screened_quartets_with(prep.nshells(), bounds, max_d * max_d, HESS_2E_SCREEN);
    let _quads_charge = ferric_core::memory::pool::reserve_global(
        "RHF Hessian 2e skeleton screened quartet list",
        quads
            .len()
            .saturating_mul(std::mem::size_of::<(usize, usize, usize, usize)>()),
    )?;
    let (dims, offs, sh2at) = (
        prep.shell_dims(),
        prep.shell_offsets(),
        prep.shell_to_atom(),
    );
    let gam = |mu, nu, la, sg| gamma(d, mu, nu, la, sg);
    let mut eng = Engine::new_2e_deriv2(op, prep, 1e-14)?;
    let mut h = Array2::<f64>::zeros((3 * natoms, 3 * natoms));
    let mut g = Vec::new();
    for &(s1, s2, s3, s4) in &quads {
        let blk = QuartetBlock::new(dims, offs, sh2at, s1, s2, s3, s4);
        perm_summed_gamma(&blk, &gam, &mut g);
        let Some(dq) = eng.compute_eri_deriv2_quartet(prep, s1, s2, s3, s4) else {
            continue;
        };
        let vals = contract_blocks(dq, &g);
        scatter_unique_pairs(&mut h, &vals, &blk.atoms);
    }
    Ok(h)
}

/// `out[idx] = Σ_perms Γ` for every AO quartet of `blk`, in the row-major
/// `[n1][n2][n3][n4]` order of libint2's blocks.
fn perm_summed_gamma<G>(blk: &QuartetBlock, gam: &G, out: &mut Vec<f64>)
where
    G: Fn(usize, usize, usize, usize) -> f64,
{
    let [n1, n2, n3, n4] = blk.n;
    let [o1, o2, o3, o4] = blk.o;
    out.clear();
    for a in 0..n1 {
        for b in 0..n2 {
            for c in 0..n3 {
                for dd in 0..n4 {
                    out.push(sum_equivalent_perms(
                        gam,
                        o1 + a,
                        o2 + b,
                        o3 + c,
                        o4 + dd,
                        blk.sym12,
                        blk.sym34,
                        blk.sym1234,
                    ));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// First-order AO matrices for the response
// ---------------------------------------------------------------------------

/// `∂S/∂x` and the fixed-density first-order Fock matrix
/// `∂F/∂x|_D = ∂h/∂x + G^x(D)` for every nuclear coordinate x (length 3N,
/// each nbf × nbf, symmetric). PySCF: `make_h1` (`h1ao`) and the `s1ao`
/// assembled in `hess_elec` / `solve_mo1`.
struct FirstOrderAo {
    s1: Vec<Array2<f64>>,
    f1: Vec<Array2<f64>>,
}

fn first_order_ao(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    natoms: usize,
    d: &Array2<f64>,
) -> Result<FirstOrderAo, FerricError> {
    let s1 = deriv1_1e_matrices(prep, natoms, ffi::OP_OVERLAP)?;
    let mut f1 = deriv1_1e_matrices(prep, natoms, ffi::OP_KINETIC)?;
    let parts = [
        deriv1_1e_matrices(prep, natoms, ffi::OP_NUCLEAR)?,
        deriv1_g_matrices(prep, op, bounds, natoms, d)?,
    ];
    for part in &parts {
        for (f, p) in f1.iter_mut().zip(part) {
            *f += p;
        }
    }
    Ok(FirstOrderAo { s1, f1 })
}

/// Full AO matrices `∂O/∂x` of a 1e operator for every nuclear coordinate,
/// including (nuclear attraction) the operator-centre derivatives.
fn deriv1_1e_matrices(
    prep: &PreparedBasis,
    natoms: usize,
    op_kind: std::os::raw::c_int,
) -> Result<Vec<Array2<f64>>, FerricError> {
    let (n_charges, mut centres) = one_electron_centres(op_kind, prep);
    let mut eng = Engine::new_1e_deriv(op_kind, prep, 1e-14)?;
    if n_charges > 0 {
        eng.set_point_charges(prep)?;
    }
    let nbf = prep.nbasis();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();
    let mut mats = vec![Array2::<f64>::zeros((nbf, nbf)); 3 * natoms];
    for s1 in 0..prep.nshells() {
        for s2 in 0..=s1 {
            centres[0] = sh2at[s1];
            centres[1] = sh2at[s2];
            let Some(blocks) = eng.compute_1e_deriv_block_n(prep, s1, s2, n_charges) else {
                continue;
            };
            let n = dims[s1] * dims[s2];
            for (k, &atom) in centres.iter().enumerate() {
                for a in 0..3 {
                    let src = &blocks[(3 * k + a) * n..(3 * k + a + 1) * n];
                    let pair = (offs[s1], dims[s1], offs[s2], dims[s2]);
                    scatter_pair_block(&mut mats[3 * atom + a], src, pair, s1 != s2);
                }
            }
        }
    }
    Ok(mats)
}

/// Add a shell-pair block (row-major n1 × n2) at (o1, o2), and its transpose
/// at (o2, o1) when the canonical pair stands for both orders.
fn scatter_pair_block(
    m: &mut Array2<f64>,
    src: &[f64],
    (o1, n1, o2, n2): (usize, usize, usize, usize),
    both: bool,
) {
    for i in 0..n1 {
        for j in 0..n2 {
            let v = src[i * n2 + j];
            m[(o1 + i, o2 + j)] += v;
            if both {
                m[(o2 + j, o1 + i)] += v;
            }
        }
    }
}

/// `G^x(D)_pq = Σ_rs [∂(pq|rs)/∂x − ½ ∂(pr|qs)/∂x] D_rs` for every nuclear
/// coordinate x, from ERI first derivatives (PySCF `make_h1`'s `vj1 − ½vk1`
/// + transpose).
fn deriv1_g_matrices(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    natoms: usize,
    d: &Array2<f64>,
) -> Result<Vec<Array2<f64>>, FerricError> {
    let max_d = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let quads = screened_quartets_with(prep.nshells(), bounds, max_d, FOCK1_SCREEN);
    let _quads_charge = ferric_core::memory::pool::reserve_global(
        "RHF Hessian first-order Fock screened quartet list",
        quads
            .len()
            .saturating_mul(std::mem::size_of::<(usize, usize, usize, usize)>()),
    )?;
    let nbf = prep.nbasis();
    let (dims, offs, sh2at) = (
        prep.shell_dims(),
        prep.shell_offsets(),
        prep.shell_to_atom(),
    );
    let mut eng = Engine::new_2e_deriv(op, prep, 1e-14)?;
    let mut g = vec![Array2::<f64>::zeros((nbf, nbf)); 3 * natoms];
    let mut v = Vec::new();
    for &(s1, s2, s3, s4) in &quads {
        let blk = QuartetBlock::new(dims, offs, sh2at, s1, s2, s3, s4);
        let Some(dq) = eng.compute_eri_deriv_quartet(prep, s1, s2, s3, s4) else {
            continue;
        };
        for atom in distinct_atoms(&blk.atoms) {
            for a in 0..3 {
                atom_summed_deriv(dq, &blk, atom, a, &mut v);
                scatter_g(&mut g[3 * atom + a], &v, &blk, d);
            }
        }
    }
    Ok(g)
}

/// The distinct atoms among a quartet's four shell centres.
fn distinct_atoms(atoms: &[usize; 4]) -> Vec<usize> {
    let mut out: Vec<usize> = atoms.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

/// `out[idx] = Σ_{k: atom(k) = atom} dq[(3k + a)·bs + idx]` — the derivative of
/// the quartet w.r.t. axis `a` of `atom` (all shell centres on that atom move).
fn atom_summed_deriv(dq: &[f64], blk: &QuartetBlock, atom: usize, a: usize, out: &mut Vec<f64>) {
    let bs = blk.block_sz;
    out.clear();
    out.resize(bs, 0.0);
    for k in (0..4).filter(|&k| blk.atoms[k] == atom) {
        let src = &dq[(3 * k + a) * bs..(3 * k + a + 1) * bs];
        for (o, s) in out.iter_mut().zip(src) {
            *o += s;
        }
    }
}

/// Scatter one quartet's (derivative) integrals into `G = J − ½K` with the
/// symmetric density `d`: every distinct index permutation (p,q,r,s) of the
/// canonical quartet adds `v·D_rs` to `G_pq` and `−½ v·D_qs` to `G_pr`, so
/// `J_pq = Σ_rs (pq|rs) D_rs` and `K_pr = Σ_qs (pq|rs) D_qs` over all ordered
/// AO tuples (same permutation bookkeeping as the gradient's Γ sum).
fn scatter_g(gx: &mut Array2<f64>, v: &[f64], blk: &QuartetBlock, d: &Array2<f64>) {
    let [n1, n2, n3, n4] = blk.n;
    let [o1, o2, o3, o4] = blk.o;
    let mut idx = 0;
    for a in 0..n1 {
        for b in 0..n2 {
            for c in 0..n3 {
                for dd in 0..n4 {
                    let val = v[idx];
                    idx += 1;
                    if val == 0.0 {
                        continue;
                    }
                    let (perms, np) = equivalent_perms([o1 + a, o2 + b, o3 + c, o4 + dd], blk);
                    for &[p, q, r, t] in &perms[..np] {
                        gx[(p, q)] += val * d[(r, t)];
                        gx[(p, r)] -= 0.5 * val * d[(q, t)];
                    }
                }
            }
        }
    }
}

/// The distinct index permutations of `(μν|λσ)` a canonical quartet stands
/// for — the same set, in the same order, that
/// [`crate::gradient::sum_equivalent_perms`] sums Γ over.
fn equivalent_perms(i: [usize; 4], blk: &QuartetBlock) -> ([[usize; 4]; 8], usize) {
    let [mu, nu, la, sg] = i;
    let (s12, s34, s1234) = (blk.sym12, blk.sym34, blk.sym1234);
    let candidates = [
        (true, [mu, nu, la, sg]),
        (s12, [nu, mu, la, sg]),
        (s34, [mu, nu, sg, la]),
        (s12 && s34, [nu, mu, sg, la]),
        (s1234, [la, sg, mu, nu]),
        (s1234 && s12, [la, sg, nu, mu]),
        (s1234 && s34, [sg, la, mu, nu]),
        (s1234 && s12 && s34, [sg, la, nu, mu]),
    ];
    let mut out = [[0usize; 4]; 8];
    let mut n = 0;
    for (keep, perm) in candidates {
        if keep {
            out[n] = perm;
            n += 1;
        }
    }
    (out, n)
}

// ---------------------------------------------------------------------------
// Term 5: CPHF orbital response
// ---------------------------------------------------------------------------

/// Unsymmetrized response term plus CPHF diagnostics.
struct ResponseTerm {
    raw: Array2<f64>,
    iterations: Vec<usize>,
    max_residual: f64,
}

/// Converged canonical MOs split into occupied/virtual blocks.
struct MoFrame<'a> {
    c_occ: ArrayView2<'a, f64>,
    c_vir: ArrayView2<'a, f64>,
    eps_occ: Array1<f64>,
    /// `ε_a − ε_i`, nvir × nocc (the CPHF diagonal and preconditioner).
    denom: Array2<f64>,
}

impl<'a> MoFrame<'a> {
    fn new(rhf: &'a ScfResult, nocc: usize) -> Self {
        let c = rhf.mos_r();
        let eps = rhf.eps_r();
        let nmo = c.ncols();
        let eps_occ = Array1::from(eps[..nocc].to_vec());
        let denom = Array2::from_shape_fn((nmo - nocc, nocc), |(a, i)| eps[nocc + a] - eps[i]);
        MoFrame {
            c_occ: c.slice(s![.., ..nocc]),
            c_vir: c.slice(s![.., nocc..]),
            eps_occ,
            denom,
        }
    }

    fn nocc(&self) -> usize {
        self.c_occ.ncols()
    }

    /// `D1(U) = 2 (C_v U C_oᵀ + C_o Uᵀ C_vᵀ)` for a virtual–occupied rotation U.
    fn response_density(&self, u_vo: &Array2<f64>) -> Array2<f64> {
        let half = self.c_vir.dot(u_vo).dot(&self.c_occ.t());
        2.0 * (&half + &half.t())
    }
}

/// `G(D) = J(D) − ½K(D)` from exact four-centre integrals, one engine pool
/// reused across every CPHF iteration.
struct TwoElectronBuilder<'a> {
    ctx: &'a ParallelContext,
    prep: &'a PreparedBasis,
    bounds: &'a SchwarzBounds,
    pool: EnginePool,
}

impl<'a> TwoElectronBuilder<'a> {
    fn new(
        ctx: &'a ParallelContext,
        prep: &'a PreparedBasis,
        bounds: &'a SchwarzBounds,
    ) -> Result<Self, FerricError> {
        let pool = EnginePool::new(bounds.op, prep, 1e-14)?;
        Ok(TwoElectronBuilder {
            ctx,
            prep,
            bounds,
            pool,
        })
    }

    fn g(&self, d: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        let n = d.nrows();
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        crate::rhf::build_jk_with_pool(
            self.ctx,
            self.prep,
            self.bounds,
            CPHF_JK_THRESH,
            d,
            &mut j,
            &mut k,
            &self.pool,
            crate::reduce::default_band_bytes(),
        )?;
        j.scaled_add(-0.5, &k);
        Ok(j)
    }
}

/// Term 5 (PySCF `hess_elec` after `solve_mo1`), unsymmetrized:
///
/// ```text
///   R[x, y] = 4 Σ_pi F^x_pi U^y_pi − 4 Σ_pi S^x_pi ε_i U^y_pi − 2 Σ_ij S^x_ij ε^y_ij
/// ```
///
/// with `p` over all MOs, `i, j` occupied, MO-basis `F^x = Cᵀ ∂F/∂x|_D C_occ`
/// and `S^x = Cᵀ ∂S/∂x C_occ`.
fn cpks_response(
    ctx: &ParallelContext,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    nocc: usize,
    first: &FirstOrderAo,
) -> Result<ResponseTerm, FerricError> {
    let c = rhf.mos_r();
    let c_occ = c.slice(s![.., ..nocc]);
    let to_mo = |m: &Array2<f64>| c.t().dot(m).dot(&c_occ);
    let h1_mo: Vec<Array2<f64>> = first.f1.iter().map(&to_mo).collect();
    let s1_mo: Vec<Array2<f64>> = first.s1.iter().map(&to_mo).collect();

    let frame = MoFrame::new(rhf, nocc);
    let g = TwoElectronBuilder::new(ctx, prep, bounds)?;
    let mut sols = Vec::with_capacity(h1_mo.len());
    for (x, (h1, s1)) in h1_mo.iter().zip(&s1_mo).enumerate() {
        let sol = solve_cphf_one(&g, &frame, h1, s1).map_err(|e| match e {
            FerricError::Convergence(msg) => FerricError::Convergence(format!(
                "CPHF for nuclear coordinate {x} (atom {}, axis {}): {msg}",
                x / 3,
                x % 3
            )),
            other => other,
        })?;
        sols.push(sol);
    }
    let raw = assemble_response(&h1_mo, &s1_mo, &sols, &frame);
    Ok(ResponseTerm {
        raw,
        iterations: sols.iter().map(|s| s.iterations).collect(),
        max_residual: sols.iter().fold(0.0f64, |m, s| m.max(s.residual)),
    })
}

/// One perturbation's CPHF solution: the full first-order MO coefficients
/// `U` (nmo × nocc, occupied block `−½S_oo`) and the first-order occupied
/// orbital-energy matrix `ε^x` (nocc × nocc).
struct CphfSolution {
    u: Array2<f64>,
    e1: Array2<f64>,
    iterations: usize,
    residual: f64,
}

/// PySCF `cphf.solve_withs1` for one perturbation, with conjugate gradient in
/// place of its Krylov solver (see the module doc for the equations).
fn solve_cphf_one(
    g: &TwoElectronBuilder<'_>,
    frame: &MoFrame<'_>,
    h1: &Array2<f64>,
    s1: &Array2<f64>,
) -> Result<CphfSolution, FerricError> {
    let nocc = frame.nocc();
    let s1_oo = s1.slice(s![..nocc, ..]).to_owned();
    let d_oo = -2.0 * frame.c_occ.dot(&s1_oo).dot(&frame.c_occ.t());
    let g_oo = g.g(&d_oo)?;

    // b = F_vo − S_vo·ε_i + G(D_oo)_vo ; solve A U = −b.
    let mut rhs = &s1.slice(s![nocc.., ..]) * &frame.eps_occ - h1.slice(s![nocc.., ..]);
    rhs -= &frame.c_vir.t().dot(&g_oo).dot(&frame.c_occ);
    let matvec = |u: &Array2<f64>| -> Result<Array2<f64>, FerricError> {
        let gu = g.g(&frame.response_density(u))?;
        Ok(&frame.denom * u + frame.c_vir.t().dot(&gu).dot(&frame.c_occ))
    };
    let (u_vo, iterations, residual) = pcg(matvec, &rhs, &frame.denom)?;

    let g_tot = g_oo + g.g(&frame.response_density(&u_vo))?;
    let mut u = Array2::<f64>::zeros((h1.nrows(), nocc));
    u.slice_mut(s![..nocc, ..]).assign(&(-0.5 * &s1_oo));
    u.slice_mut(s![nocc.., ..]).assign(&u_vo);

    let eps = &frame.eps_occ;
    let mut e1 =
        h1.slice(s![..nocc, ..]).to_owned() + frame.c_occ.t().dot(&g_tot).dot(&frame.c_occ);
    for ((i, j), v) in e1.indexed_iter_mut() {
        *v -= 0.5 * s1_oo[(i, j)] * (eps[i] + eps[j]);
    }
    Ok(CphfSolution {
        u,
        e1,
        iterations,
        residual,
    })
}

/// Jacobi-preconditioned conjugate gradient for `A x = rhs` with `A` symmetric
/// positive definite (`precond` = diag of A). Returns `(x, iterations, final
/// max |residual|)`; a non-positive curvature `pᵀAp` or `CPHF_MAX_ITER`
/// iterations without reaching `CPHF_TOL` is a typed error.
fn pcg<F>(
    mut apply: F,
    rhs: &Array2<f64>,
    precond: &Array2<f64>,
) -> Result<(Array2<f64>, usize, f64), FerricError>
where
    F: FnMut(&Array2<f64>) -> Result<Array2<f64>, FerricError>,
{
    let dot = |a: &Array2<f64>, b: &Array2<f64>| (a * b).sum();
    let max_abs = |a: &Array2<f64>| a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut x = rhs / precond;
    let mut r = rhs - &apply(&x)?;
    let mut res = max_abs(&r);
    if res < CPHF_TOL {
        return Ok((x, 0, res));
    }
    let mut z = &r / precond;
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    for it in 1..=CPHF_MAX_ITER {
        let ap = apply(&p)?;
        let pap = dot(&p, &ap);
        if pap.is_nan() || pap <= 0.0 {
            return Err(FerricError::Convergence(format!(
                "orbital Hessian is not positive definite (pᵀAp = {pap:.3e} at iteration \
                 {it}); the RHF reference is not a stable minimum"
            )));
        }
        let alpha = rz / pap;
        x.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &ap);
        res = max_abs(&r);
        if res < CPHF_TOL {
            return Ok((x, it, res));
        }
        z = &r / precond;
        let rz_new = dot(&r, &z);
        p = &z + &(rz_new / rz * &p);
        rz = rz_new;
    }
    Err(FerricError::Convergence(format!(
        "CPHF did not converge in {CPHF_MAX_ITER} iterations (max |residual| {res:.3e} > \
         {CPHF_TOL:.0e})"
    )))
}

/// `R[x, y]` of [`cpks_response`] from the per-perturbation solutions.
fn assemble_response(
    h1_mo: &[Array2<f64>],
    s1_mo: &[Array2<f64>],
    sols: &[CphfSolution],
    frame: &MoFrame<'_>,
) -> Array2<f64> {
    let n3 = h1_mo.len();
    let nocc = frame.nocc();
    let mut r = Array2::<f64>::zeros((n3, n3));
    for x in 0..n3 {
        // 4 (F^x − S^x ε) contracted with U^y over all (p, i).
        let hs = 4.0 * (&h1_mo[x] - &(&s1_mo[x] * &frame.eps_occ));
        let s1_oo = s1_mo[x].slice(s![..nocc, ..]);
        for (y, sol) in sols.iter().enumerate() {
            r[(x, y)] = (&hs * &sol.u).sum() - 2.0 * (&s1_oo * &sol.e1).sum();
        }
    }
    r
}

// ---------------------------------------------------------------------------
// Frequencies from the analytic Hessian
// ---------------------------------------------------------------------------

/// Harmonic frequencies from [`rhf_hessian`] at an already-converged SCF.
///
/// The Hessian goes through the same mass-weighting, translation/rotation
/// projection and diagonalization as the finite-difference path
/// ([`super::frequencies::frequencies_from_cartesian_hessian`]).
/// `asymmetry` carries [`RhfHessianParts::response_asymmetry`] (the only
/// non-symmetric term), `n_gradient_evaluations` is 0 and `energy` is
/// `rhf.energy`. The driver that also runs the SCF is
/// [`super::frequencies::harmonic_frequencies_analytic`].
pub fn analytic_frequencies(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    config: &RhfConfig,
) -> Result<super::frequencies::FrequencyResult, FerricError> {
    let masses = super::frequencies::atom_masses(mol)?;
    let parts = rhf_hessian_parts(ctx, mol, prep, op, bounds, rhf, config)?;
    let mut result =
        super::frequencies::frequencies_from_cartesian_hessian(mol, &parts.total(), &masses)?;
    result.asymmetry = parts.response_asymmetry;
    result.n_gradient_evaluations = 0;
    result.hessian_source = super::frequencies::HessianSource::Analytic;
    result.energy = rhf.energy;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn water() -> Molecule {
        Molecule::parse_xyz(
            "3\nwater\nO 0.000000 0.000000 0.117300\n\
             H 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n",
            0,
            1,
        )
        .unwrap()
    }

    fn h2() -> Molecule {
        Molecule::parse_xyz("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n", 0, 1).unwrap()
    }

    #[test]
    fn nuclear_hessian_h2_symmetry() {
        let m = h2();
        let h = hess_nuclear_repulsion(&m);
        assert_eq!(h.dim(), (6, 6));

        // H2 along z: only zz blocks are nonzero
        // d²(1/r)/dz1 dz2 = d²(1/|z1-z2|)/dz1 dz2 = -2/r³ (off-diag)
        // d²(1/r)/dz1 dz1 = +2/r³ (diagonal, from -sum rule)
        let r = 0.74 * 1.8897259886; // Angstrom -> Bohr
        let expected = 2.0 / (r * r * r);

        // h[z1,z1] should be positive (restoring force)
        assert!(
            (h[(2, 2)] - expected).abs() < 1e-6,
            "h[z1,z1] = {}, expected {}",
            h[(2, 2)],
            expected
        );
        // h[z1,z2] should be negative
        assert!(
            (h[(2, 5)] + expected).abs() < 1e-6,
            "h[z1,z2] = {}, expected {}",
            h[(2, 5)],
            -expected
        );

        // Perpendicular (x) blocks: h[x1,x2] = ZZ*(−3·0·0/r⁵ + 1/r³) = 1/r³
        // Diagonal sum rule: h[x1,x1] = −h[x1,x2] = −1/r³
        let expected_diag_perp = -1.0 / (r * r * r);
        assert!(
            (h[(0, 0)] - expected_diag_perp).abs() < 1e-6,
            "h[x1,x1] = {}, expected {}",
            h[(0, 0)],
            expected_diag_perp
        );
    }

    #[test]
    fn nuclear_hessian_symmetric() {
        let m = water();
        let h = hess_nuclear_repulsion(&m);
        let n3 = 3 * m.atoms.len();
        for i in 0..n3 {
            for j in 0..n3 {
                assert!(
                    (h[(i, j)] - h[(j, i)]).abs() < 1e-12,
                    "nuclear Hessian not symmetric at ({i},{j}): {} vs {}",
                    h[(i, j)],
                    h[(j, i)]
                );
            }
        }
    }

    #[test]
    fn nuclear_hessian_translational_invariance() {
        let m = water();
        let h = hess_nuclear_repulsion(&m);
        let natoms = m.atoms.len();

        // Translational invariance: Σ_B d²V/dR_A dR_B = 0 for all A
        for a in 0..natoms {
            for x in 0..3 {
                let mut sum = [0.0; 3];
                for b in 0..natoms {
                    for y in 0..3 {
                        sum[y] += h[(3 * a + x, 3 * b + y)];
                    }
                }
                for y in 0..3 {
                    assert!(
                        sum[y].abs() < 1e-10,
                        "translational invariance violated: Σ_B h[{a}{x},{b}{y}] = {}",
                        sum[y],
                        b = "B",
                        y = y
                    );
                }
            }
        }
    }
}
