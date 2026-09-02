//! Analytical nuclear gradients.
//!
//! Provides both the RHF gradient and a density-parameterized core
//! ([`crate::gradient::hf_gradient_with_density`]) that correlated methods reuse with relaxed densities.

use crate::result::{ScfResult, Spin};
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use ndarray::Array2;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Parallel derivative-loop infrastructure (P1)
//
// Every SCF/KS 2e nuclear-gradient path is the same shape: a Schwarz-screened
// canonical shell-quartet loop contracting ERI first derivatives with a
// two-particle density Γ built from one or more one-particle densities. The
// 1e paths are canonical shell-pair loops. Both were serial — the dominant
// cost of every RHF/UHF/KS geometry step ran on one core.
//
// Design (mirrors direct_jk.rs + ferric-mp2 P7):
//  * enumerate the flat screened work list up front — a pure function of
//    (prep, bounds, max|D|), never of the thread count;
//  * partition it with reduce::deterministic_group_size (pure function of the
//    list length) and reduce per-group (natoms,3) partials with
//    reduce::grouped_deterministic_sum — fold order is ascending group index,
//    so the gradient is bit-identical across RAYON_NUM_THREADS;
//  * one derivative engine per rayon worker via a small pool (engine_pool.rs
//    pattern: libint2 engine ctors are serialized behind a global mutex, so
//    construct them O(threads) times, not O(groups));
//  * below a small work threshold (free-atom rule) run the plain serial loop —
//    the threshold is a pure function of the work-list length, so the
//    serial/parallel path choice can never depend on the thread count either.
// ---------------------------------------------------------------------------

/// Below this many screened quartets the 2e-derivative loop runs serially
/// (pool construction + rayon fan-out overhead beats the win on tiny jobs,
/// e.g. free atoms / diatomics in minimal bases). Pure function of the
/// screened-list length — never of the thread count.
const PAR_2E_QUARTET_THRESHOLD: usize = 512;

/// Below this many shell pairs the 1e-derivative loops run serially.
const PAR_1E_PAIR_THRESHOLD: usize = 64;

/// One derivative engine per rayon worker (+1 spare for non-pool threads),
/// same rationale as `engine_pool::EnginePool`: engine construction is
/// expensive and serialized behind a global libint2 ctor mutex, so it must
/// happen O(threads) times, not once per rayon work chunk. Generic over the
/// constructor so the same pool serves 2e-deriv and 1e-deriv engines.
struct GradEnginePool {
    engines: Vec<Mutex<Engine>>,
}

impl GradEnginePool {
    fn new(mk: &(dyn Fn() -> Result<Engine, FerricError> + Sync)) -> Result<Self, FerricError> {
        let n = rayon::current_num_threads().max(1) + 1;
        let mut engines = Vec::with_capacity(n);
        for _ in 0..n {
            engines.push(Mutex::new(mk()?));
        }
        Ok(GradEnginePool { engines })
    }

    /// Run `f` with this thread's engine (index by `current_thread_index()`,
    /// spare slot for non-pool threads). The per-slot mutex is uncontended —
    /// it only satisfies `&mut Engine` borrowing.
    #[inline]
    fn with<R>(&self, f: impl FnOnce(&mut Engine) -> R) -> R {
        let idx = rayon::current_thread_index().unwrap_or(self.engines.len() - 1);
        let slot = idx.min(self.engines.len() - 1);
        let mut eng = self.engines[slot].lock().unwrap();
        f(&mut eng)
    }
}

/// Flat screened canonical quartet list — identical enumeration order and
/// screen (`Q12·Q34·max|D| < 1e-12`) to the old serial loops, and a pure
/// function of (nsh, bounds, max_d): group boundaries derived from it fix the
/// floating-point association of the reduction, so it must never depend on
/// the thread count.
fn screened_quartets(
    nsh: usize,
    bounds: &SchwarzBounds,
    max_d: f64,
) -> Vec<(usize, usize, usize, usize)> {
    let mut quads = Vec::new();
    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let b12 = bounds.q[(s1, s2)];
            for s3 in 0..=s1 {
                let s4max = if s3 == s1 { s2 } else { s3 };
                for s4 in 0..=s4max {
                    let b34 = bounds.q[(s3, s4)];
                    if b12 * b34 * max_d < 1e-12 {
                        continue;
                    }
                    quads.push((s1, s2, s3, s4));
                }
            }
        }
    }
    quads
}

/// Shared driver for every 4-center 2e-derivative gradient contribution:
/// `grad[atom,coord] += Σ_quartets Σ_perms Γ(μ,ν,λ,σ) · d(μν|λσ)/dR`.
///
/// `gamma` is the two-particle density for one AO index permutation; the
/// permutational symmetry sums (8-fold canonical) are handled here. All six
/// former serial twins (RHF, UHF, KS scaled-K, K-only, UKS scaled-K, UKS
/// K-only) are thin wrappers differing only in `gamma` and `max_d`.
pub(crate) fn par_twoelectron_gradient<G>(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    max_d: f64,
    gamma: G,
) -> Result<Array2<f64>, FerricError>
where
    G: Fn(usize, usize, usize, usize) -> f64 + Sync,
{
    let natoms = prep.shell_to_atom().iter().copied().max().unwrap_or(0) + 1;
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();

    let quads = screened_quartets(nsh, bounds, max_d);
    let n_quads = quads.len();
    let mut grad = Array2::zeros((natoms, 3));
    if n_quads == 0 {
        return Ok(grad);
    }

    if n_quads < PAR_2E_QUARTET_THRESHOLD {
        // Serial fallback (free-atom rule). Path choice is a pure function of
        // the screened-list length, so it cannot vary with thread count.
        let mut eng = Engine::new_2e_deriv(op, prep, 1e-14)?;
        for &(s1, s2, s3, s4) in &quads {
            if let Some(dq) = eng.compute_eri_deriv_quartet(prep, s1, s2, s3, s4) {
                let blk = QuartetBlock::new(dims, offs, sh2at, s1, s2, s3, s4);
                accum_2e_grad_gamma(&mut grad, dq, &blk, &gamma);
            }
        }
        return Ok(grad);
    }

    let pool = GradEnginePool::new(&|| Engine::new_2e_deriv(op, prep, 1e-14))?;
    let group_size = crate::reduce::deterministic_group_size(n_quads);
    let n_groups = n_quads.div_ceil(group_size);
    // Per-group partials are (natoms,3) — tiny — so the band budget in
    // grouped_deterministic_sum is effectively unbounded here; what we use it
    // for is the ascending-group-order fold (bit-identical across threads).
    crate::reduce::grouped_deterministic_sum(&mut grad, n_groups, natoms.max(2), crate::reduce::default_band_bytes(), |g| {
        let lo = g * group_size;
        let hi = (lo + group_size).min(n_quads);
        let mut local = Array2::<f64>::zeros((natoms, 3));
        for &(s1, s2, s3, s4) in &quads[lo..hi] {
            pool.with(|eng| {
                if let Some(dq) = eng.compute_eri_deriv_quartet(prep, s1, s2, s3, s4) {
                    let blk = QuartetBlock::new(dims, offs, sh2at, s1, s2, s3, s4);
                    accum_2e_grad_gamma(&mut local, dq, &blk, &gamma);
                }
            });
        }
        Ok(local)
    })?;
    Ok(grad)
}

/// Shared driver for the 1e-derivative shell-pair loops (overlap, kinetic,
/// nuclear). `accum` computes one canonical pair's contribution into the
/// per-group partial using the worker's engine; reduction is the same
/// deterministic grouped sum as the 2e path.
fn par_pair_gradient<M, F>(
    prep: &PreparedBasis,
    natoms: usize,
    mk_engine: M,
    accum: F,
) -> Result<Array2<f64>, FerricError>
where
    M: Fn() -> Result<Engine, FerricError> + Sync,
    F: Fn(&mut Array2<f64>, &mut Engine, usize, usize) + Sync,
{
    let nsh = prep.nshells();
    let pairs: Vec<(usize, usize)> = (0..nsh)
        .flat_map(|s1| (0..=s1).map(move |s2| (s1, s2)))
        .collect();
    let n_pairs = pairs.len();
    let mut grad = Array2::zeros((natoms, 3));
    if n_pairs == 0 {
        return Ok(grad);
    }

    if n_pairs < PAR_1E_PAIR_THRESHOLD {
        let mut eng = mk_engine()?;
        for &(s1, s2) in &pairs {
            accum(&mut grad, &mut eng, s1, s2);
        }
        return Ok(grad);
    }

    let pool = GradEnginePool::new(&mk_engine)?;
    let group_size = crate::reduce::deterministic_group_size(n_pairs);
    let n_groups = n_pairs.div_ceil(group_size);
    crate::reduce::grouped_deterministic_sum(&mut grad, n_groups, natoms.max(2), crate::reduce::default_band_bytes(), |g| {
        let lo = g * group_size;
        let hi = (lo + group_size).min(n_pairs);
        let mut local = Array2::<f64>::zeros((natoms, 3));
        for &(s1, s2) in &pairs[lo..hi] {
            pool.with(|eng| accum(&mut local, eng, s1, s2));
        }
        Ok(local)
    })?;
    Ok(grad)
}

/// Build the HF energy-weighted density: W_μν = 2 Σ_i^occ ε_i C_μi C_νi.
/// Contract an arbitrary weight matrix with the overlap first derivative:
/// returns `(natoms, 3)` with `grad[A,c] = Σ_μν W_μν · ∂S_μν/∂R_{A,c}`.
///
/// Unlike [`oneelectron_gradient`]'s Pulay term (which assumes a symmetric `W`
/// and folds a `−` sign in), this is the raw, sign-free `Σ W·dS/dR` for a general
/// (possibly asymmetric) `W`. Both bra- and ket-center derivatives are summed, so
/// `W` is used exactly as given — no implicit symmetrization or factor. The
/// correlated-gradient assembly needs this to contract the asymmetric `im1`/`zeta`
/// Lagrangian matrices with `dS/dR` under PySCF's explicit sign conventions.
pub fn overlap_deriv_contract(
    prep: &PreparedBasis,
    w: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = prep.shell_to_atom().iter().copied().max().unwrap_or(0) + 1;
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();
    // Canonical shell-pair loop (s1 >= s2). For the off-diagonal pair we add both
    // (μ,ν) and (ν,μ) contributions explicitly, so an asymmetric W is contracted
    // exactly. dS deriv layout: [dx_bra, dy_bra, dz_bra, dx_ket, dy_ket, dz_ket].
    par_pair_gradient(
        prep,
        natoms,
        || Engine::new_1e_deriv(ffi::OP_OVERLAP, prep, 1e-14),
        |local, eng, s1, s2| {
            if let Some(deriv) = eng.compute_1e_deriv_block(prep, s1, s2) {
                let n1 = dims[s1];
                let n2 = dims[s2];
                let block_sz = n1 * n2;
                let a1 = sh2at[s1];
                let a2 = sh2at[s2];
                for i in 0..n1 {
                    for j in 0..n2 {
                        let mu = offs[s1] + i;
                        let nu = offs[s2] + j;
                        let idx = i * n2 + j;
                        // W contribution for this (μ,ν) block. For s1 != s2 the
                        // canonical loop only visits (s1,s2) once, so include the
                        // transposed (ν,μ) weight against the same symmetric dS.
                        let wval = if s1 == s2 {
                            w[(mu, nu)]
                        } else {
                            w[(mu, nu)] + w[(nu, mu)]
                        };
                        for c in 0..3 {
                            let d1 = deriv[c * block_sz + idx];
                            let d2 = deriv[(3 + c) * block_sz + idx];
                            local[(a1, c)] += wval * d1;
                            local[(a2, c)] += wval * d2;
                        }
                    }
                }
            }
        },
    )
}

/// Build the energy-weighted density matrix W = 2 C_occ diag(ε_occ) C_occ^T for gradient evaluation.
pub fn build_energy_weighted_density(result: &ScfResult, nocc: usize) -> Array2<f64> {
    let c = result.mos_r();
    let eps = result.eps_r();
    // W = 2 C_occ diag(ε) C_occ^T. Scale the columns of C_occ by ε first
    // (cw[μ,i] = ε_i C_μi), then one GEMM C_occ · cw^T.
    let c_occ = c.slice(ndarray::s![.., ..nocc]);
    let eps_occ = ndarray::ArrayView1::from(&eps[..nocc]);
    let cw = &c_occ * &eps_occ;
    c_occ.dot(&cw.t()) * 2.0
}

/// Compute the RHF analytical nuclear gradient.
/// Returns a (natoms, 3) array of dE/dR_Ax, dE/dR_Ay, dE/dR_Az per atom.
///
/// Returns `Err(FerricError::Libint(...))` if the molecule contains ghost atoms —
/// gradients with respect to ghost centers are not implemented (they involve a
/// different 1e nuclear derivative structure and are out of scope for CP use-cases
/// where only the real-atom geometry matters).
///
/// # ECPs
///
/// ECP molecules are supported: the `dV_ECP/dR` term is added via
/// [`ecp_gradient`] (libecpint first derivatives), and the nuclear-repulsion
/// derivative uses `effective_z()` to match the energy it differentiates. Both
/// halves are required — before 2026-07-28 the term was missing *and* the
/// nuclear derivative used the bare `z`, so ECP gradients were silently wrong
/// and were temporarily rejected outright. Validated against central finite
/// difference of the total energy on HI/def2-SVP (see
/// `ferric-scf/tests/ecp_rhf.rs`).
pub fn rhf_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    if mol.atoms.iter().any(|a| a.ghost) {
        return Err(FerricError::Libint(
            "rhf_gradient is not implemented for molecules containing ghost atoms \
             (CP gradient requires special treatment of ghost-center derivatives)"
                .into(),
        ));
    }
    let nocc = (mol.nelec() / 2) as usize;
    let w = build_energy_weighted_density(result, nocc);
    let mut grad = hf_gradient_with_density(mol, prep, op, bounds, result.density_r(), &w, ext)?;
    // ECP term: dE/dR gains Σ_μν D_μν dV_ECP_μν/dR whenever the basis carries
    // ECPs. `ecp_gradient` is a no-op (zero work) otherwise, so the
    // all-electron path is unchanged.
    grad += &ecp_gradient(mol, prep, result.density_r())?;
    Ok(grad)
}

/// [`rhf_gradient`] plus the QM-atom-centre Fock-term contribution of
/// Thole-damped polarizable embedding (Lane B, Task B3) — a thin wrapper
/// rather than a new parameter on `rhf_gradient` itself, so the other
/// (numerous) existing call sites of `rhf_gradient` are completely
/// unaffected (zero risk of a signature-change regression outside this
/// lane). `sites`/`dipoles` are typically `config.polarizable.as_ref()`
/// and `result.induced_dipoles.as_ref()`; when either is `None` this is
/// EXACTLY `rhf_gradient` (see [`crate::polarizable::qm_gradient_contribution`]
/// for why the fixed-mu QM-centre term is a plain Hellmann-Feynman
/// contraction, no `W`-vs-naive subtlety unlike [`crate::polarizable::site_gradient`]).
pub fn rhf_gradient_with_polarizable(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    sites: Option<&crate::polarizable::PolarizableSites>,
    dipoles: Option<&Array2<f64>>,
) -> Result<Array2<f64>, FerricError> {
    let mut grad = rhf_gradient(mol, prep, op, bounds, result, ext)?;
    if let (Some(sites), Some(dipoles)) = (sites, dipoles) {
        grad += &crate::polarizable::polarizable_gradient_term(mol, prep, sites, dipoles, result.density_r())?;
    }
    Ok(grad)
}

/// [`uhf_gradient`] plus the QM-atom-centre Fock-term contribution of
/// Thole-damped polarizable embedding — the UHF sibling of
/// [`rhf_gradient_with_polarizable`]. `sites`/`dipoles` are typically
/// `config.polarizable.as_ref()` and `result.induced_dipoles.as_ref()`;
/// when either is `None`, or `sites.sites` is empty, this is EXACTLY
/// `uhf_gradient` (`polarizable_gradient_term` returns an all-zero array
/// in that case, added unconditionally so both branches share one code
/// path rather than an `if`/`else` that could silently diverge — pinned
/// by `uhf_gradient_with_polarizable_none_matches_plain_uhf_gradient`).
/// The added term uses `result.density_total()` (`= density_alpha +
/// density_beta` for UHF), matching
/// [`crate::polarizable::qm_gradient_contribution`]'s documented
/// total-density-only dependence — see that function's doc for why this
/// term needs no UHF-specific derivation.
pub fn uhf_gradient_with_polarizable(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    sites: Option<&crate::polarizable::PolarizableSites>,
    dipoles: Option<&Array2<f64>>,
) -> Result<Array2<f64>, FerricError> {
    let mut grad = uhf_gradient(mol, prep, op, bounds, result, ext)?;
    if let (Some(sites), Some(dipoles)) = (sites, dipoles) {
        grad += &crate::polarizable::polarizable_gradient_term(mol, prep, sites, dipoles, result.density_total())?;
    }
    Ok(grad)
}

/// ECP contribution to the nuclear gradient:
/// `dE_ECP/dR_{A,c} = Σ_μν D_μν dV_ECP_μν/dR_{A,c}`.
///
/// Returns a zero `(natoms, 3)` array when the basis carries no ECPs, so callers
/// can add it unconditionally without changing the all-electron result.
///
/// `V_ECP` enters the energy through `hcore_ecp` as a plain one-electron
/// operator, so its gradient is the same simple density contraction as `dT/dR`
/// and `dV_nuc/dR` — there is no Pulay-type term beyond the `−W·dS/dR` already
/// carried by [`oneelectron_gradient`], because the AO basis derivative is
/// already accounted for there. libecpint sums the bra-, ket-, and
/// ECP-center contributions per atom internally.
///
/// The density passed must be the SPIN-SUMMED (total) AO density, matching the
/// convention `oneelectron_gradient` uses for the other 1e terms.
pub fn ecp_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    d: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = mol.atoms.len();
    let Some(derivs) =
        ferric_integrals::oneelectron::ecp_potential_deriv(mol, prep.basis_set())?
    else {
        return Ok(Array2::zeros((natoms, 3)));
    };
    if derivs.len() != natoms {
        return Err(FerricError::Libint(format!(
            "ecp_gradient: got {} atom blocks, expected {natoms}",
            derivs.len()
        )));
    }
    let mut grad = Array2::zeros((natoms, 3));
    for (a, block) in derivs.iter().enumerate() {
        for (c, dv) in block.iter().enumerate() {
            if dv.dim() != d.dim() {
                return Err(FerricError::Libint(format!(
                    "ecp_gradient: dV_ECP shape {:?} != density shape {:?}",
                    dv.dim(),
                    d.dim()
                )));
            }
            // Σ_μν D_μν (dV_ECP/dR)_μν
            grad[(a, c)] = d.iter().zip(dv.iter()).map(|(x, y)| x * y).sum::<f64>();
        }
    }
    Ok(grad)
}

/// Compute nuclear gradient using provided density and energy-weighted density.
///
/// Shared core of both the RHF gradient and correlated gradients (which pass
/// relaxed densities). Includes nuclear repulsion, 1e (dS, dT, dV), and
/// 4-center 2e contributions.
pub fn hf_gradient_with_density(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d: &Array2<f64>,
    w: &Array2<f64>,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    let mut grad = oneelectron_gradient(mol, prep, d, w, ext)?;
    grad += &twoelectron_gradient(prep, op, bounds, d)?;
    Ok(grad)
}

/// Compute one-electron gradient contributions: nuclear repulsion + dS, dT, dV.
///
/// Takes the density `d` (for kinetic + nuclear attraction derivatives) and the
/// energy-weighted density `w` (for overlap / Pulay force). Returns a `(natoms, 3)`
/// gradient array.
///
/// `ext`, when `Some`, adds the external-potential gradient contributions:
/// charge-electron (Hellmann-Feynman, via the shared nuclear-attraction
/// derivative engine extended with `set_point_charges_extra`/
/// `compute_1e_deriv_block_n`), charge-nuclear and field-nuclear (classical
/// Coulomb terms, via `ExternalPotential::charge_nuclear_gradient`/
/// `field_nuclear_gradient`), and — when a field is present — the
/// field-density Hellmann-Feynman term `d/dR_A [Σ_μν D_μν (E·<μ|r|ν>)]`.
///
/// The field-density term is evaluated by **finite difference** of
/// `oneelectron::field_hcore_term` contracted with the fixed density `D`,
/// not via a native libint2 analytical derivative. This was a deliberate
/// choice, not an oversight: the shim's derivative dispatcher
/// (`op_for_kind` in `shim.cc`, shared by `scf_engine_create_deriv`) has no
/// case for the `emultipole1` operator (dipole integrals in this codebase
/// go through a separate, non-derivative-capable `scf_compute_dipole` path
/// built directly from a hardcoded `Operator::emultipole1` engine — see
/// `shim.cc`'s "Electric dipole integrals via emultipole1" section). Passing
/// `ffi::OP_EMULTIPOLE1` (103) to `scf_engine_create_deriv` therefore returns
/// NULL. Adding a native emultipole1-derivative shim function is out of
/// scope for this plan (would require new C++), so this term uses the
/// finite-difference fallback at fixed `D` (cheap: O(natoms) hcore rebuilds,
/// not full SCF re-solves) with `h = 1e-4`, matching this file's other
/// finite-difference gradient checks.
///
/// `ext = None` reproduces the pre-external-potential gradient exactly.
pub fn oneelectron_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    d: &Array2<f64>,
    w: &Array2<f64>,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    let natoms = mol.atoms.len();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let sh2at = prep.shell_to_atom();

    let mut grad = Array2::zeros((natoms, 3));

    // 1. Nuclear repulsion gradient (ghost atoms: zero charge, skip)
    for i in 0..natoms {
        if mol.atoms[i].ghost { continue; }
        for j in (i + 1)..natoms {
            if mol.atoms[j].ghost { continue; }
            let a = &mol.atoms[i];
            let b = &mol.atoms[j];
            let dx = a.x - b.x;
            let dy = a.y - b.y;
            let dz = a.zpos - b.zpos;
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            // MUST be effective_z(), not the bare z: the ENERGY being
            // differentiated (`Molecule::nuclear_repulsion`) uses effective_z(),
            // which is `z − n_core_ecp` for an ECP atom. Using the bare z here
            // made the classical term disagree with its own energy on every ECP
            // molecule. Identical to `z` for all-electron atoms, so the
            // all-electron gradient is unchanged.
            let za = a.effective_z() as f64;
            let zb = b.effective_z() as f64;
            let f_over_r3 = za * zb / (r * r2);
            grad[(i, 0)] -= f_over_r3 * dx;
            grad[(i, 1)] -= f_over_r3 * dy;
            grad[(i, 2)] -= f_over_r3 * dz;
            grad[(j, 0)] += f_over_r3 * dx;
            grad[(j, 1)] += f_over_r3 * dy;
            grad[(j, 2)] += f_over_r3 * dz;
        }
    }

    // 2. One-electron gradient: Σ_μν D_μν dH_μν/dR - Σ_μν W_μν dS_μν/dR
    // H = T + V, so we need dT/dR, dV/dR, dS/dR
    //
    // Each piece is a canonical shell-pair loop fanned out through
    // par_pair_gradient (deterministic grouped reduction, per-worker engines).

    // 2a. Overlap derivative (Pulay force): -W_μν dS_μν/dR
    grad += &par_pair_gradient(
        prep,
        natoms,
        || Engine::new_1e_deriv(ffi::OP_OVERLAP, prep, 1e-14),
        |local, eng, s1, s2| {
            if let Some(deriv) = eng.compute_1e_deriv_block(prep, s1, s2) {
                let n1 = dims[s1];
                let n2 = dims[s2];
                let block_sz = n1 * n2;
                let a1 = sh2at[s1];
                let a2 = sh2at[s2];
                for i in 0..n1 {
                    for j in 0..n2 {
                        let mu = offs[s1] + i;
                        let nu = offs[s2] + j;
                        let idx = i * n2 + j;
                        let wval = if s1 == s2 { w[(mu, nu)] } else { 2.0 * w[(mu, nu)] };
                        for c in 0..3 {
                            // deriv layout: [dx1, dy1, dz1, dx2, dy2, dz2]
                            let d1 = deriv[c * block_sz + idx];
                            let d2 = deriv[(3 + c) * block_sz + idx];
                            local[(a1, c)] -= wval * d1;
                            local[(a2, c)] -= wval * d2;
                        }
                    }
                }
            }
        },
    )?;

    // 2b. Kinetic derivative: D_μν dT_μν/dR
    grad += &par_pair_gradient(
        prep,
        natoms,
        || Engine::new_1e_deriv(ffi::OP_KINETIC, prep, 1e-14),
        |local, eng, s1, s2| {
            if let Some(deriv) = eng.compute_1e_deriv_block(prep, s1, s2) {
                let n1 = dims[s1];
                let n2 = dims[s2];
                let block_sz = n1 * n2;
                let a1 = sh2at[s1];
                let a2 = sh2at[s2];
                for i in 0..n1 {
                    for j in 0..n2 {
                        let mu = offs[s1] + i;
                        let nu = offs[s2] + j;
                        let idx = i * n2 + j;
                        let dval = if s1 == s2 { d[(mu, nu)] } else { 2.0 * d[(mu, nu)] };
                        for c in 0..3 {
                            let d1 = deriv[c * block_sz + idx];
                            let d2 = deriv[(3 + c) * block_sz + idx];
                            local[(a1, c)] += dval * d1;
                            local[(a2, c)] += dval * d2;
                        }
                    }
                }
            }
        },
    )?;

    // 2c. Nuclear attraction derivative: D_μν dV_μν/dR
    // Nuclear has 2 shell centers + natoms nuclear centers = 2+natoms centers for derivatives
    // But libint2 only returns derivatives w.r.t. the 2 shell centers for nuclear operator.
    // The nuclear center derivatives need separate handling.
    // Actually for nuclear integrals, libint2 returns derivatives w.r.t. all centers:
    // For 2 shell centers, results has 3*(2+natoms) entries when using the nuclear operator.
    // This is complex — let me use a simpler approach: finite difference on V for now,
    // and use the Hellmann-Feynman term for the nuclear derivative.
    //
    // The proper approach for dV/dR:
    // dV/dR_A = Σ_μν D_μν [dV_μν/dR_A(shell) + dV_μν/dR_A(nuclear)]
    // The shell derivative part is the Pulay-like term from shells on atom A.
    // The nuclear (Hellmann-Feynman) part is: Σ_μν D_μν d/dR_A [Σ_C Z_C/|r-R_C|]
    // which only contributes when A is one of the nuclear centers.
    //
    // For the Hellmann-Feynman nuclear attraction gradient, we need:
    // dV/dR_A = -Z_A Σ_μν D_μν <μ| (r-R_A)/|r-R_A|^3 |ν>
    // This is equivalent to the electric field integral at nucleus A, contracted with D.
    //
    // However, libint2's nuclear operator derivative engine handles all of this in one shot:
    // for deriv_order=1 with nuclear operator, it returns 3*(2 + N_charges) derivative blocks.
    // Blocks 0-5 are d/d(shell center 1) and d/d(shell center 2) as usual.
    // Blocks 6, 7, 8 are d/d(charge center 0) (x, y, z).
    // Blocks 9, 10, 11 are d/d(charge center 1), etc.
    //
    // So we need to handle this specially.
    let extra_charges: &[ferric_core::external_potential::PointCharge] =
        ext.map(|e| e.point_charges.as_slice()).unwrap_or(&[]);
    let n_charges = natoms + extra_charges.len();

    grad += &par_pair_gradient(
        prep,
        natoms,
        || {
            let mut eng = Engine::new_1e_deriv(ffi::OP_NUCLEAR, prep, 1e-14)?;
            eng.set_point_charges_extra(prep, extra_charges)?;
            Ok(eng)
        },
        |local, eng, s1, s2| {
            // Engine::compute_1e_deriv_block_n sizes its buffer for
            // 3*(2 + n_charges) blocks: blocks 0-5 are the two shell-center
            // derivatives, blocks 6..6+3*natoms are the real-atom nuclear
            // centers, and blocks 6+3*natoms..6+3*n_charges (when extra
            // charges are present) are the external-charge centers.
            if let Some(deriv) = eng.compute_1e_deriv_block_n(prep, s1, s2, n_charges) {
                let n1 = dims[s1];
                let n2 = dims[s2];
                let block_sz = n1 * n2;
                let a1 = sh2at[s1];
                let a2 = sh2at[s2];
                for i in 0..n1 {
                    for j in 0..n2 {
                        let mu = offs[s1] + i;
                        let nu = offs[s2] + j;
                        let idx = i * n2 + j;
                        let dval = if s1 == s2 { d[(mu, nu)] } else { 2.0 * d[(mu, nu)] };
                        // Shell center derivatives (first 6 blocks)
                        for c in 0..3 {
                            local[(a1, c)] += dval * deriv[c * block_sz + idx];
                            local[(a2, c)] += dval * deriv[(3 + c) * block_sz + idx];
                        }
                        // Real-atom nuclear-center derivatives only (blocks
                        // 6..6+3*natoms) — external-charge blocks
                        // (6+3*natoms..6+3*n_charges) are written by libint2
                        // but deliberately NOT accumulated into `local`,
                        // since `local`/`grad` are sized (natoms, 3): no
                        // force is reported on the external charges
                        // themselves (they are fixed, not gradient
                        // variables).
                        for atom_c in 0..natoms {
                            for c in 0..3 {
                                let blk = 6 + atom_c * 3 + c;
                                local[(atom_c, c)] += dval * deriv[blk * block_sz + idx];
                            }
                        }
                    }
                }
            }
        },
    )?;

    // External-potential classical + Hellmann-Feynman terms that don't come
    // from the shared nuclear-attraction engine above.
    if let Some(ext) = ext {
        grad += &ext.charge_nuclear_gradient(mol);
        grad += &ext.field_nuclear_gradient(mol);
        if let Some(field) = ext.field {
            // Field-density Hellmann-Feynman term — see the doc comment on
            // this function for why finite difference (not a native libint2
            // derivative) is used here.
            let bs = prep.basis_set();
            grad += &field_density_gradient_fd(mol, bs, d, field)?;
        }
        if !ext.smeared_charges.is_empty() {
            grad += &smeared_charge_qm_gradient(prep, d, &ext.smeared_charges)?;
        }
    }

    Ok(grad)
}

/// QM-atom-centre Hellmann-Feynman gradient contribution of the
/// Gaussian-smeared MM charges' hcore term
/// `V_μν = -Σ_i q_i (μν|g_i) / norm_i` (see
/// `oneelectron::smeared_attraction`): `dE/dR_A = Σ_μν D_μν dV_μν/dR_A`.
///
/// Uses `Engine::compute_eri3_deriv`'s 9-block layout — `[d/d(site), d/d(sh1),
/// d/d(sh2)] × [x, y, z]`, each block `np*n1*n2` doubles (`np = 1` here, one
/// function per site s-shell) — and accumulates ONLY the `sh1`/`sh2` (blocks
/// 1, 2) obs-centre derivatives into `grad`, which is `(natoms, 3)` over the
/// QM molecule. The site-centre derivative (block 0) is NOT accumulated here
/// (the sites are not QM atoms); [`smeared_site_forces`] uses translational
/// invariance (block 0 = −(block 1 + block 2), proved by
/// `engine.rs::test_eri3_deriv_translational_invariance`) to get the site's
/// own force from the SAME two blocks this function reads, without a
/// separate integral evaluation.
fn smeared_charge_qm_gradient(
    prep: &PreparedBasis,
    d: &Array2<f64>,
    smeared: &[ferric_core::external_potential::SmearedCharge],
) -> Result<Array2<f64>, FerricError> {
    use ferric_core::external_potential::SmearedCharge;
    use ferric_integrals::operator::Operator;
    use ferric_integrals::site_basis::SiteBasis;

    let natoms = prep.shell_to_atom().iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut grad = Array2::<f64>::zeros((natoms, 3));
    if smeared.is_empty() {
        return Ok(grad);
    }

    let sites: Vec<[f64; 4]> = smeared
        .iter()
        .map(|s: &SmearedCharge| [s.x, s.y, s.z, 1.0 / (s.width * s.width)])
        .collect();
    let site_basis = SiteBasis::new(&sites, 0)?;

    let mut eng = Engine::new_3center_deriv(Operator::coulomb(), prep, &site_basis.prep, 1e-14)?;
    let offs = prep.shell_offsets();
    let dims = prep.shell_dims();
    let sh2at = prep.shell_to_atom();
    let nsh = prep.nshells();

    for (i, sc) in smeared.iter().enumerate() {
        let sh_p = site_basis.site_shell[i];
        let norm = site_basis.norm_int[i];
        for s1 in 0..nsh {
            for s2 in 0..nsh {
                let Some(deriv) = eng.compute_eri3_deriv(prep, &site_basis.prep, sh_p, s1, s2) else { continue };
                let n1 = dims[s1];
                let n2 = dims[s2];
                let block_sz = n1 * n2;
                let a1 = sh2at[s1];
                let a2 = sh2at[s2];
                for a in 0..n1 {
                    for b in 0..n2 {
                        let mu = offs[s1] + a;
                        let nu = offs[s2] + b;
                        let idx = a * n2 + b;
                        let dval = d[(mu, nu)];
                        // dV_μν/dR = -q/norm * d(μν|g)/dR
                        let coeff = -sc.q / norm * dval;
                        for c in 0..3 {
                            // block 1 = d/d(sh1 center), block 2 = d/d(sh2 center)
                            let d1 = deriv[(3 + c) * block_sz + idx];
                            let d2 = deriv[(6 + c) * block_sz + idx];
                            grad[(a1, c)] += coeff * d1;
                            grad[(a2, c)] += coeff * d2;
                        }
                    }
                }
            }
        }
    }
    Ok(grad)
}

/// dE/dR of the SCF energy with respect to each Gaussian-smeared MM site's
/// OWN position (i.e. the force convention `mm_forces` needs is `-` this),
/// from the same `V_μν = -Σ_i q_i (μν|g_i)/norm_i` term
/// `smeared_charge_qm_gradient` differentiates.
///
/// By translational invariance of the 3-centre integral `(μν|g_i)` (site
/// center + the two obs shell centers, three centers total — proved by
/// `engine.rs::test_eri3_deriv_translational_invariance`, block 0 + block 1 +
/// block 2 = 0 elementwise), `dE/dR_site = -(dE/dR_sh1 + dE/dR_sh2)` for
/// every shell-pair contribution: the site derivative never needs to be
/// evaluated on its own, it falls out of the SAME two obs-centre blocks
/// `smeared_charge_qm_gradient` already reads.
///
/// Returns one `[f64; 3]` per site, in `smeared` order (this is `dE/dR`, NOT
/// the force `qmmm::mm_forces` returns for point charges — negate it for a
/// force, matching that function's documented sign convention).
pub fn smeared_site_forces(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    smeared: &[ferric_core::external_potential::SmearedCharge],
) -> Result<Vec<[f64; 3]>, FerricError> {
    use ferric_core::external_potential::SmearedCharge;
    use ferric_integrals::operator::Operator;
    use ferric_integrals::site_basis::SiteBasis;

    if smeared.is_empty() {
        return Ok(Vec::new());
    }

    let sites: Vec<[f64; 4]> = smeared
        .iter()
        .map(|s: &SmearedCharge| [s.x, s.y, s.z, 1.0 / (s.width * s.width)])
        .collect();
    let site_basis = SiteBasis::new(&sites, 0)?;

    let mut eng = Engine::new_3center_deriv(Operator::coulomb(), prep, &site_basis.prep, 1e-14)?;
    let offs = prep.shell_offsets();
    let dims = prep.shell_dims();
    let nsh = prep.nshells();

    let mut out = vec![[0.0_f64; 3]; smeared.len()];
    for (i, sc) in smeared.iter().enumerate() {
        let sh_p = site_basis.site_shell[i];
        let norm = site_basis.norm_int[i];
        let mut site_grad = [0.0_f64; 3];
        for s1 in 0..nsh {
            for s2 in 0..nsh {
                let Some(deriv) = eng.compute_eri3_deriv(prep, &site_basis.prep, sh_p, s1, s2) else { continue };
                let n1 = dims[s1];
                let n2 = dims[s2];
                let block_sz = n1 * n2;
                for a in 0..n1 {
                    for b in 0..n2 {
                        let mu = offs[s1] + a;
                        let nu = offs[s2] + b;
                        let idx = a * n2 + b;
                        let coeff = -sc.q / norm * density[(mu, nu)];
                        for c in 0..3 {
                            let d1 = deriv[(3 + c) * block_sz + idx];
                            let d2 = deriv[(6 + c) * block_sz + idx];
                            // dE/dR_site = -(dE/dR_sh1 + dE/dR_sh2) by
                            // translational invariance (see doc comment).
                            site_grad[c] += -coeff * (d1 + d2);
                        }
                    }
                }
            }
        }
        out[i] = site_grad;
    }
    // Classical nuclear-charge/site-charge term: E = sum_A Z_A q_i erf/R
    // depends on the site position exactly as it does on each atom, so
    // ferric_core::external_potential::ExternalPotential::smeared_charge_nuclear_site_gradient
    // (dE/dR_site) must be added -- the electronic Hellmann-Feynman term
    // above accounts only for the one-electron hcore contraction, not the
    // classical Z_A*q_i term also present in ExternalPotential::charge_nuclear_energy.
    let classical = ferric_core::external_potential::ExternalPotential {
        point_charges: Vec::new(),
        smeared_charges: smeared.to_vec(),
        field: None,
    }
    .smeared_charge_nuclear_site_gradient(mol);
    for (o, c) in out.iter_mut().zip(classical.iter()) {
        o[0] += c[0];
        o[1] += c[1];
        o[2] += c[2];
    }
    Ok(out)
}

/// Field-density Hellmann-Feynman gradient via finite difference of the
/// field one-electron term, contracted with the fixed density `D`.
///
/// This is O(natoms) hcore rebuilds at fixed `D` (not a full SCF re-solve),
/// cheap relative to the SCF itself. See the doc comment on
/// [`oneelectron_gradient`] for why this term does not use a native libint2
/// analytical derivative (no `emultipole1`-derivative shim function exists).
fn field_density_gradient_fd(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    d: &Array2<f64>,
    field: [f64; 3],
) -> Result<Array2<f64>, FerricError> {
    let natoms = mol.atoms.len();
    let mut grad = Array2::zeros((natoms, 3));
    let h = 1e-4;
    for a in 0..natoms {
        for c in 0..3 {
            let mut mol_p = mol.clone();
            let mut mol_m = mol.clone();
            match c {
                0 => { mol_p.atoms[a].x += h; mol_m.atoms[a].x -= h; }
                1 => { mol_p.atoms[a].y += h; mol_m.atoms[a].y -= h; }
                _ => { mol_p.atoms[a].zpos += h; mol_m.atoms[a].zpos -= h; }
            }
            let prep_p = match PreparedBasis::new(&mol_p, bs) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let prep_m = match PreparedBasis::new(&mol_m, bs) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let e_p: f64 = (d * &ferric_integrals::oneelectron::field_hcore_term(&prep_p, field)?).sum();
            let e_m: f64 = (d * &ferric_integrals::oneelectron::field_hcore_term(&prep_m, field)?).sum();
            grad[(a, c)] = (e_p - e_m) / (2.0 * h);
        }
    }
    Ok(grad)
}

/// Compute the 4-center two-electron gradient contribution: Σ Γ_μνλσ d(μν|λσ)/dR.
///
/// Uses the two-particle density Γ built from the provided one-particle density `d`:
///   Γ_μνλσ = 0.5 D_μν D_λσ - 0.25 D_μλ D_νσ
///
/// Returns a `(natoms, 3)` gradient array.
pub fn twoelectron_gradient(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    // The 2e gradient contribution from each integral (μν|λσ) is:
    //   Γ_μνλσ * d(μν|λσ)/dR
    // where Γ_μνλσ = 0.5*D_μν*D_λσ - 0.25*D_μλ*D_νσ.
    // Canonical-quartet enumeration, permutational symmetry, and the parallel
    // deterministic reduction all live in par_twoelectron_gradient.
    let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    par_twoelectron_gradient(prep, op, bounds, max_d, |mu, nu, la, sg| {
        gamma(d, mu, nu, la, sg)
    })
}

/// Bilinear two-electron gradient: `Σ Γ(D1,D2)_μνλσ · d(μν|λσ)/dR`.
///
/// Uses the symmetrized bilinear 2-particle density
/// `Γ(D1,D2)_μνλσ = 0.25(D1_μν D2_λσ + D2_μν D1_λσ) − 0.125(D1_μλ D2_νσ + D2_μλ D1_νσ)`,
/// the polarization of the quadratic RHF `Γ(D)` (so `Γ(D,D) ≡ gamma(D)` and this
/// reduces to [`twoelectron_gradient`] when `D1 == D2`).
///
/// This is the exact form the correlated (MP2) gradient needs for the
/// two-electron-integral-derivative energy term: PySCF's `grad/mp2.py::grad_elec`
/// contracts `∂veff(hf_dm1)/∂R` with `dm1p = hf_dm1 + 2·dm1_corr` (line 184), i.e.
/// `Γ(hf_dm1, hf_dm1 + 2·dm1_corr)` — NOT `Γ(P_relax, P_relax)`. Passing
/// `d1 = hf_dm1`, `d2 = hf_dm1 + 2·dm1_corr` reproduces that term.
pub fn twoelectron_gradient_bilinear(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d1: &Array2<f64>,
    d2: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let m1 = d1.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let m2 = d2.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    // Screening bound: the bilinear Γ magnitude scales with max|D1|·max|D2|; use
    // the geometric-mean-free product-safe bound sqrt(m1·m2) is too loose — pass
    // the larger so screening never drops a significant quartet.
    let max_d = m1.max(m2);
    par_twoelectron_gradient(prep, op, bounds, max_d, |mu, nu, la, sg| {
        0.25 * (d1[(mu, nu)] * d2[(la, sg)] + d2[(mu, nu)] * d1[(la, sg)])
            - 0.125 * (d1[(mu, la)] * d2[(nu, sg)] + d2[(mu, la)] * d1[(nu, sg)])
    })
}

/// Geometry and permutational-symmetry bundle for one shell quartet.
///
/// Groups the per-quartet scalars that the 2e-gradient kernels need: the AO
/// dimensions `n` and offsets `o` of the four shells, the atom each shell sits
/// on, the derivative block size, and which index permutations are distinct.
#[derive(Clone, Copy)]
struct QuartetBlock {
    /// AO dimensions (n1,n2,n3,n4).
    n: [usize; 4],
    /// AO offsets (o1,o2,o3,o4).
    o: [usize; 4],
    /// Atom index for each of the four shells.
    atoms: [usize; 4],
    /// Number of AO quartets in one derivative block (n1*n2*n3*n4).
    block_sz: usize,
    /// s1 != s2 — the (1,2) pair has a distinct transpose.
    sym12: bool,
    /// s3 != s4 — the (3,4) pair has a distinct transpose.
    sym34: bool,
    /// (s1,s2) != (s3,s4) — bra and ket are distinct.
    sym1234: bool,
}

impl QuartetBlock {
    /// Build the bundle for shell quartet (s1,s2,s3,s4) from the prepared-basis
    /// per-shell tables.
    fn new(
        dims: &[usize], offs: &[usize], sh2at: &[usize],
        s1: usize, s2: usize, s3: usize, s4: usize,
    ) -> Self {
        let n = [dims[s1], dims[s2], dims[s3], dims[s4]];
        Self {
            n,
            o: [offs[s1], offs[s2], offs[s3], offs[s4]],
            atoms: [sh2at[s1], sh2at[s2], sh2at[s3], sh2at[s4]],
            block_sz: n[0] * n[1] * n[2] * n[3],
            sym12: s1 != s2,
            sym34: s3 != s4,
            sym1234: (s1, s2) != (s3, s4),
        }
    }
}

/// Accumulate the 2e gradient for one shell quartet, handling all permutational
/// symmetry, for an arbitrary two-particle density `gamma(μ,ν,λ,σ)`.
///
/// dq layout: 12 blocks of block_sz each. Block ordering:
///   [dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3, dx4,dy4,dz4]
/// where centers 1,2,3,4 correspond to shells s1,s2,s3,s4.
///
/// The derivative blocks always correspond to the physical shell centers s1..s4.
/// For each permutation of the integral indices, the Γ prefactor changes but the
/// derivative blocks remain the same (they are derivatives w.r.t. the shell centers).
///
/// The total Γ prefactor for each derivative block is the sum of Γ over all
/// equivalent permutations.
fn accum_2e_grad_gamma<G>(grad: &mut Array2<f64>, dq: &[f64], blk: &QuartetBlock, gamma: &G)
where
    G: Fn(usize, usize, usize, usize) -> f64,
{
    let [n1, n2, n3, n4] = blk.n;
    let [o1, o2, o3, o4] = blk.o;
    let QuartetBlock { block_sz, atoms, sym12, sym34, sym1234, .. } = *blk;
    for a in 0..n1 {
        for b in 0..n2 {
            for c in 0..n3 {
                for dd in 0..n4 {
                    let idx = ((a * n2 + b) * n3 + c) * n4 + dd;
                    let mu = o1 + a;
                    let nu = o2 + b;
                    let la = o3 + c;
                    let sg = o4 + dd;

                    // Sum Γ over all equivalent permutations of (μ,ν,λ,σ).
                    let mut g = gamma(mu, nu, la, sg);

                    if sym12 {
                        g += gamma(nu, mu, la, sg);
                    }
                    if sym34 {
                        g += gamma(mu, nu, sg, la);
                    }
                    if sym12 && sym34 {
                        g += gamma(nu, mu, sg, la);
                    }
                    if sym1234 {
                        g += gamma(la, sg, mu, nu);
                        if sym12 {
                            g += gamma(la, sg, nu, mu);
                        }
                        if sym34 {
                            g += gamma(sg, la, mu, nu);
                        }
                        if sym12 && sym34 {
                            g += gamma(sg, la, nu, mu);
                        }
                    }

                    // Accumulate into gradient for each center
                    for center in 0..4 {
                        let atom = atoms[center];
                        for coord in 0..3 {
                            let dv = dq[(center * 3 + coord) * block_sz + idx];
                            grad[(atom, coord)] += g * dv;
                        }
                    }
                }
            }
        }
    }
}

/// RHF-Γ specialization of [`accum_2e_grad_gamma`], kept for the
/// component-breakdown test harness.
#[cfg(test)]
fn accum_2e_grad(grad: &mut Array2<f64>, d: &Array2<f64>, dq: &[f64], blk: &QuartetBlock) {
    accum_2e_grad_gamma(grad, dq, blk, &|mu, nu, la, sg| gamma(d, mu, nu, la, sg));
}

/// Two-particle density matrix element: Γ_μνλσ = 0.5*D_μν*D_λσ - 0.25*D_μλ*D_νσ
#[inline]
fn gamma(d: &Array2<f64>, mu: usize, nu: usize, la: usize, sg: usize) -> f64 {
    0.5 * d[(mu, nu)] * d[(la, sg)] - 0.25 * d[(mu, la)] * d[(nu, sg)]
}

/// UHF two-particle density: Γ_μνλσ = 0.5*D_μν D_λσ − 0.5*(D_α,μλ D_α,νσ + D_β,μλ D_β,νσ)
/// where D = D_α + D_β.
#[inline]
fn gamma_uhf(
    d: &Array2<f64>,
    da: &Array2<f64>,
    db: &Array2<f64>,
    mu: usize,
    nu: usize,
    la: usize,
    sg: usize,
) -> f64 {
    0.5 * d[(mu, nu)] * d[(la, sg)]
        - 0.5 * (da[(mu, la)] * da[(nu, sg)] + db[(mu, la)] * db[(nu, sg)])
}

/// Build the UHF energy-weighted density: W = W_α + W_β with
/// W_σ = Σ_i^occ_σ ε_σi C_σi C_σi^T.
pub fn build_energy_weighted_density_uhf(
    result: &ScfResult,
    nocc_a: usize,
    nocc_b: usize,
) -> Array2<f64> {
    // W = Σ_spin C_occ diag(ε) C_occ^T (per-spin, unit weight — UHF, not ×2).
    let spin_block = |c: &Array2<f64>, eps: &[f64], nocc: usize| {
        let c_occ = c.slice(ndarray::s![.., ..nocc]);
        let eps_occ = ndarray::ArrayView1::from(&eps[..nocc]);
        let cw = &c_occ * &eps_occ;
        c_occ.dot(&cw.t())
    };
    let mut w = spin_block(&result.mos_alpha, &result.eps_alpha, nocc_a);
    if let (Some(cb), Some(epsb)) = (result.mos_beta.as_ref(), result.eps_beta.as_ref()) {
        w = w + spin_block(cb, epsb, nocc_b);
    }
    w
}

/// Compute the UHF analytical nuclear gradient.
pub fn uhf_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    if mol.atoms.iter().any(|a| a.ghost) {
        return Err(FerricError::Libint(
            "uhf_gradient is not implemented for molecules containing ghost atoms".into(),
        ));
    }
    assert!(matches!(result.spin, Spin::Unrestricted), "uhf_gradient: ScfResult.spin must be Unrestricted");
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;

    let d_total = &result.density_alpha
        + result
            .density_beta
            .as_ref()
            .expect("uhf_gradient: missing density_beta");
    let w = build_energy_weighted_density_uhf(result, nocc_a, nocc_b);
    let mut grad = oneelectron_gradient(mol, prep, &d_total, &w, ext)?;
    grad += &twoelectron_gradient_uhf(
        prep,
        op,
        bounds,
        &d_total,
        &result.density_alpha,
        result.density_beta.as_ref().unwrap(),
    )?;
    Ok(grad)
}

/// Compute the ROHF analytical nuclear gradient.
///
/// ROHF has a single set of MOs partitioned into doubly-occupied (closed),
/// singly-α-occupied (open), and virtual blocks. The total/α/β densities are:
///   D_β = Σ_i^closed C_i C_i^T,   D_α = D_β + Σ_j^open C_j C_j^T,
///   D_total = D_α + D_β = 2 D_β + D_open.
///
/// Energy-weighted density:
///   W = 2 Σ_i^closed ε_i C_i C_i^T + Σ_j^open ε_j C_j C_j^T
/// where ε are the eigenvalues of the Roothaan effective Fock.
///
/// The two-electron piece uses the UHF Γ form with D_α/D_β as above.
pub fn rohf_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    if mol.atoms.iter().any(|a| a.ghost) {
        return Err(FerricError::Libint(
            "rohf_gradient is not implemented for molecules containing ghost atoms".into(),
        ));
    }
    assert!(
        matches!(result.spin, Spin::RestrictedOpen),
        "rohf_gradient: ScfResult.spin must be RestrictedOpen"
    );
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let nocc_open = two_s as usize;
    let nocc_double = ((nelec - two_s) / 2) as usize;

    let d_alpha = &result.density_alpha;
    let d_beta = result
        .density_beta
        .as_ref()
        .expect("rohf_gradient: missing density_beta");
    let d_total = d_alpha + d_beta;

    // Energy-weighted density: closed orbitals weighted 2 ε_i, open weighted ε_j.
    let n = result.mos_alpha.nrows();
    let c = &result.mos_alpha;
    let eps = &result.eps_alpha;
    let mut w = Array2::<f64>::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for i in 0..nocc_double {
                sum += 2.0 * eps[i] * c[(mu, i)] * c[(nu, i)];
            }
            for j in nocc_double..nocc_double + nocc_open {
                sum += eps[j] * c[(mu, j)] * c[(nu, j)];
            }
            w[(mu, nu)] = sum;
        }
    }

    let mut grad = oneelectron_gradient(mol, prep, &d_total, &w, ext)?;
    grad += &twoelectron_gradient_uhf(prep, op, bounds, &d_total, d_alpha, d_beta)?;
    Ok(grad)
}

/// UHF four-center two-electron gradient: Σ Γ_uhf_μνλσ d(μν|λσ)/dR.
pub fn twoelectron_gradient_uhf(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d_total: &Array2<f64>,
    d_alpha: &Array2<f64>,
    d_beta: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let max_d = d_total.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    par_twoelectron_gradient(prep, op, bounds, max_d, |mu, nu, la, sg| {
        gamma_uhf(d_total, d_alpha, d_beta, mu, nu, la, sg)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhf::{solve_rhf, RhfConfig};
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::external_potential::{ExternalPotential, PointCharge};
    use ferric_integrals::basis_bridge::PreparedBasis;

    #[test]
    fn oneelectron_gradient_none_matches_prior_behavior() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ferric_core::parallel::ParallelContext::default();
        let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        let nocc = (mol.nelec() / 2) as usize;
        let d = result.density_r();
        let w = build_energy_weighted_density(&result, nocc);

        let g_orig = oneelectron_gradient(&mol, &prep, d, &w, None).unwrap();
        let ext = ExternalPotential::default();
        let g_new = oneelectron_gradient(&mol, &prep, d, &w, Some(&ext)).unwrap();
        assert_eq!(g_orig, g_new);
    }

    #[test]
    fn oneelectron_gradient_external_charge_matches_finite_difference() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ferric_core::parallel::ParallelContext::default();
        let ext = ExternalPotential {
            point_charges: vec![PointCharge { q: 1.0, x: 0.0, y: 0.0, z: 15.0 }],
            smeared_charges: Vec::new(),
            field: None,
        };
        let config = RhfConfig { external_potential: Some(ext.clone()), ..Default::default() };

        // Analytic gradient at the equilibrium geometry (converged SCF density/W).
        // NOTE: the finite difference below differentiates the *total* SCF
        // energy (1e + 2e + nuclear/classical-external repulsion), so the
        // comparable analytic quantity is the full `hf_gradient_with_density`
        // (1e + 2e), not the bare `oneelectron_gradient` (1e only) — comparing
        // the 1e-only piece against a total-energy FD is an apples-to-oranges
        // mismatch (verified: the 1e-only gradient is off from the FD by
        // ~2.94 Hartree/Bohr here, exactly accounted for by the missing 2e
        // contribution; the 1e-only nuclear+extra-charge term itself was
        // independently verified exact against a fixed-density FD probe).
        let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        let nocc = (mol.nelec() / 2) as usize;
        let w = build_energy_weighted_density(&result, nocc);
        let analytic =
            hf_gradient_with_density(&mol, &prep, op, &bounds, result.density_r(), &w, Some(&ext)).unwrap();

        // Finite-difference check on atom 0 (O), z-component: perturb the
        // molecule's geometry by +/- h and re-run solve_rhf + hcore/vnn.
        let h = 1e-4;
        let mut mol_plus = mol.clone();
        mol_plus.atoms[0].zpos += h;
        let prep_plus = PreparedBasis::new(&mol_plus, &bs).unwrap();
        let bounds_plus = SchwarzBounds::compute(op, &prep_plus).unwrap();
        let e_plus = solve_rhf(&ctx, &mol_plus, &prep_plus, op, &bounds_plus, &config).unwrap().energy;

        let mut mol_minus = mol.clone();
        mol_minus.atoms[0].zpos -= h;
        let prep_minus = PreparedBasis::new(&mol_minus, &bs).unwrap();
        let bounds_minus = SchwarzBounds::compute(op, &prep_minus).unwrap();
        let e_minus = solve_rhf(&ctx, &mol_minus, &prep_minus, op, &bounds_minus, &config).unwrap().energy;

        let fd = (e_plus - e_minus) / (2.0 * h);
        assert!((analytic[(0, 2)] - fd).abs() < 1e-5, "analytic={}, fd={}", analytic[(0, 2)], fd);
    }

    #[test]
    fn oneelectron_gradient_uniform_field_matches_finite_difference() {
        // Covers the field-density Hellmann-Feynman term (finite-difference
        // fallback in `field_density_gradient_fd`), which the point-charge
        // test above does not exercise at all (its `ext.field` is `None`).
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ferric_core::parallel::ParallelContext::default();
        let ext = ExternalPotential {
            point_charges: vec![],
            smeared_charges: Vec::new(),
            field: Some([0.0, 0.0, 0.01]),
        };
        let config = RhfConfig { external_potential: Some(ext.clone()), ..Default::default() };

        let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        let nocc = (mol.nelec() / 2) as usize;
        let w = build_energy_weighted_density(&result, nocc);
        let analytic =
            hf_gradient_with_density(&mol, &prep, op, &bounds, result.density_r(), &w, Some(&ext)).unwrap();

        let h = 1e-4;
        let mut mol_plus = mol.clone();
        mol_plus.atoms[0].zpos += h;
        let prep_plus = PreparedBasis::new(&mol_plus, &bs).unwrap();
        let bounds_plus = SchwarzBounds::compute(op, &prep_plus).unwrap();
        let e_plus = solve_rhf(&ctx, &mol_plus, &prep_plus, op, &bounds_plus, &config).unwrap().energy;

        let mut mol_minus = mol.clone();
        mol_minus.atoms[0].zpos -= h;
        let prep_minus = PreparedBasis::new(&mol_minus, &bs).unwrap();
        let bounds_minus = SchwarzBounds::compute(op, &prep_minus).unwrap();
        let e_minus = solve_rhf(&ctx, &mol_minus, &prep_minus, op, &bounds_minus, &config).unwrap().energy;

        let fd = (e_plus - e_minus) / (2.0 * h);
        assert!((analytic[(0, 2)] - fd).abs() < 1e-5, "analytic={}, fd={}", analytic[(0, 2)], fd);
    }

    /// Compute individual gradient components for debugging.
    /// Returns (vnn_grad, overlap_grad, kinetic_grad, nuclear_grad, twoelec_grad, total_grad)
    #[allow(clippy::type_complexity)] // 6 named gradient-component arrays for FD validation
    fn gradient_components(
        mol: &Molecule,
        prep: &PreparedBasis,
        op: Operator,
        bounds: &SchwarzBounds,
        result: &ScfResult,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let natoms = mol.atoms.len();
        let n = prep.nbasis();
        let nsh = prep.nshells();
        let dims = prep.shell_dims();
        let offs = prep.shell_offsets();
        let sh2at = prep.shell_to_atom();
        let d = result.density_r();
        let nocc = (mol.nelec() / 2) as usize;
        let c = result.mos_r();
        let eps = result.eps_r();
        let mut w = Array2::zeros((n, n));
        for mu in 0..n {
            for nu in 0..n {
                let mut sum = 0.0;
                for i in 0..nocc {
                    sum += eps[i] * c[(mu, i)] * c[(nu, i)];
                }
                w[(mu, nu)] = 2.0 * sum;
            }
        }

        let mut vnn_grad = Array2::zeros((natoms, 3));
        let mut overlap_grad = Array2::zeros((natoms, 3));
        let mut kinetic_grad = Array2::zeros((natoms, 3));
        let mut nuclear_grad = Array2::zeros((natoms, 3));
        let mut twoelec_grad = Array2::zeros((natoms, 3));

        // Vnn gradient (ghost atoms: zero charge, skip)
        for i in 0..natoms {
            if mol.atoms[i].ghost { continue; }
            for j in (i + 1)..natoms {
                if mol.atoms[j].ghost { continue; }
                let a = &mol.atoms[i];
                let b = &mol.atoms[j];
                let dx = a.x - b.x;
                let dy = a.y - b.y;
                let dz = a.zpos - b.zpos;
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = r2.sqrt();
                let za = a.z as f64;
                let zb = b.z as f64;
                let f_over_r3 = za * zb / (r * r2);
                vnn_grad[(i, 0)] -= f_over_r3 * dx;
                vnn_grad[(i, 1)] -= f_over_r3 * dy;
                vnn_grad[(i, 2)] -= f_over_r3 * dz;
                vnn_grad[(j, 0)] += f_over_r3 * dx;
                vnn_grad[(j, 1)] += f_over_r3 * dy;
                vnn_grad[(j, 2)] += f_over_r3 * dz;
            }
        }

        // Overlap (Pulay) gradient
        {
            let mut eng = Engine::new_1e_deriv(ffi::OP_OVERLAP, prep, 1e-14).unwrap();
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    if let Some(deriv) = eng.compute_1e_deriv_block(prep, s1, s2) {
                        let n1 = dims[s1];
                        let n2 = dims[s2];
                        let block_sz = n1 * n2;
                        let a1 = sh2at[s1];
                        let a2 = sh2at[s2];
                        for i in 0..n1 {
                            for j in 0..n2 {
                                let mu = offs[s1] + i;
                                let nu = offs[s2] + j;
                                let idx = i * n2 + j;
                                let wval = if s1 == s2 { w[(mu, nu)] } else { 2.0 * w[(mu, nu)] };
                                for cc in 0..3 {
                                    let d1 = deriv[cc * block_sz + idx];
                                    let d2 = deriv[(3 + cc) * block_sz + idx];
                                    overlap_grad[(a1, cc)] -= wval * d1;
                                    overlap_grad[(a2, cc)] -= wval * d2;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Kinetic gradient
        {
            let mut eng = Engine::new_1e_deriv(ffi::OP_KINETIC, prep, 1e-14).unwrap();
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    if let Some(deriv) = eng.compute_1e_deriv_block(prep, s1, s2) {
                        let n1 = dims[s1];
                        let n2 = dims[s2];
                        let block_sz = n1 * n2;
                        let a1 = sh2at[s1];
                        let a2 = sh2at[s2];
                        for i in 0..n1 {
                            for j in 0..n2 {
                                let mu = offs[s1] + i;
                                let nu = offs[s2] + j;
                                let idx = i * n2 + j;
                                let dval = if s1 == s2 { d[(mu, nu)] } else { 2.0 * d[(mu, nu)] };
                                for cc in 0..3 {
                                    let d1 = deriv[cc * block_sz + idx];
                                    let d2 = deriv[(3 + cc) * block_sz + idx];
                                    kinetic_grad[(a1, cc)] += dval * d1;
                                    kinetic_grad[(a2, cc)] += dval * d2;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Nuclear attraction gradient
        {
            let mut eng = Engine::new_1e_deriv(ffi::OP_NUCLEAR, prep, 1e-14).unwrap();
            eng.set_point_charges(prep).unwrap();
            let max_fn = prep.shell_dims().iter().copied().max().unwrap_or(1);
            let nderiv_nuclear = 6 + 3 * natoms;
            let max_block = max_fn * max_fn;
            let mut nbuf = vec![0.0f64; nderiv_nuclear * max_block];
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    let n1 = dims[s1];
                    let n2 = dims[s2];
                    let block_sz = n1 * n2;
                    let total = nderiv_nuclear * block_sz;
                    if nbuf.len() < total { nbuf.resize(total, 0.0); }
                    // SAFETY: nbuf is pre-sized to nderiv_nuclear * block_sz; handle_mut()/handle()
                    // are live pointers; shell indices are in range. Shim returns written >= 0.
                    let written = unsafe {
                        ffi::scf_compute_1e_deriv_block(
                            eng.handle_mut(), prep.handle(),
                            s1 as std::os::raw::c_int, s2 as std::os::raw::c_int,
                            nbuf.as_mut_ptr(),
                        )
                    };
                    assert!(written >= 0, "libint2 internal error in nuclear deriv block ({s1},{s2}): status {written}");
                    if written == 0 { continue; }
                    let a1 = sh2at[s1];
                    let a2 = sh2at[s2];
                    for i in 0..n1 {
                        for j in 0..n2 {
                            let mu = offs[s1] + i;
                            let nu = offs[s2] + j;
                            let idx = i * n2 + j;
                            let dval = if s1 == s2 { d[(mu, nu)] } else { 2.0 * d[(mu, nu)] };
                            for cc in 0..3 {
                                nuclear_grad[(a1, cc)] += dval * nbuf[cc * block_sz + idx];
                                nuclear_grad[(a2, cc)] += dval * nbuf[(3 + cc) * block_sz + idx];
                            }
                            for atom_c in 0..natoms {
                                for cc in 0..3 {
                                    let blk = 6 + atom_c * 3 + cc;
                                    nuclear_grad[(atom_c, cc)] += dval * nbuf[blk * block_sz + idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2e gradient
        {
            let mut eng = Engine::new_2e_deriv(op, prep, 1e-14).unwrap();
            let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    let b12 = bounds.q[(s1, s2)];
                    for s3 in 0..=s1 {
                        let s4max = if s3 == s1 { s2 } else { s3 };
                        for s4 in 0..=s4max {
                            let b34 = bounds.q[(s3, s4)];
                            if b12 * b34 * max_d < 1e-12 { continue; }
                            let deriv = eng.compute_eri_deriv_quartet(prep, s1, s2, s3, s4);
                            if let Some(dq) = deriv {
                                let blk = QuartetBlock::new(dims, offs, sh2at, s1, s2, s3, s4);
                                accum_2e_grad(&mut twoelec_grad, d, dq, &blk);
                            }
                        }
                    }
                }
            }
        }

        let mut total = Array2::zeros((natoms, 3));
        for i in 0..natoms {
            for c in 0..3 {
                total[(i, c)] = vnn_grad[(i, c)] + overlap_grad[(i, c)]
                    + kinetic_grad[(i, c)] + nuclear_grad[(i, c)] + twoelec_grad[(i, c)];
            }
        }

        (vnn_grad, overlap_grad, kinetic_grad, nuclear_grad, twoelec_grad, total)
    }

    #[test]
    fn test_gradient_components_h2() {
        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig { energy_conv: 1e-10, ..Default::default() };
        let result = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &prep, op, &bounds, &config).unwrap();

        let (vnn, overlap, kinetic, nuclear, twoelec, total) =
            gradient_components(&mol, &prep, op, &bounds, &result);

        // Also compute FD of individual energy components
        let delta = 1e-5;
        let natoms = mol.atoms.len();
        let mut fd_vnn = Array2::<f64>::zeros((natoms, 3));
        let mut fd_total = Array2::<f64>::zeros((natoms, 3));
        let config2 = RhfConfig { energy_conv: 1e-10, ..Default::default() };
        for atom in 0..natoms {
            for coord in 0..3 {
                let mut mol_p = mol.clone();
                let mut mol_m = mol.clone();
                match coord {
                    0 => { mol_p.atoms[atom].x += delta; mol_m.atoms[atom].x -= delta; }
                    1 => { mol_p.atoms[atom].y += delta; mol_m.atoms[atom].y -= delta; }
                    _ => { mol_p.atoms[atom].zpos += delta; mol_m.atoms[atom].zpos -= delta; }
                }
                fd_vnn[(atom, coord)] = (mol_p.nuclear_repulsion() - mol_m.nuclear_repulsion()) / (2.0 * delta);

                let bs2 = basis::bundled("sto-3g").unwrap();
                let prep_p = PreparedBasis::new(&mol_p, &bs2).unwrap();
                let bounds_p = SchwarzBounds::compute(Operator::coulomb(), &prep_p).unwrap();
                let res_p = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol_p, &prep_p, Operator::coulomb(), &bounds_p, &config2).unwrap();

                let prep_m = PreparedBasis::new(&mol_m, &bs2).unwrap();
                let bounds_m = SchwarzBounds::compute(Operator::coulomb(), &prep_m).unwrap();
                let res_m = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol_m, &prep_m, Operator::coulomb(), &bounds_m, &config2).unwrap();

                fd_total[(atom, coord)] = (res_p.energy - res_m.energy) / (2.0 * delta);
            }
        }

        eprintln!("=== H2 Gradient Component Breakdown ===");
        for atom in 0..natoms {
            for c in 0..3 {
                eprintln!(
                    "atom={} coord={}: vnn={:12.8} overlap={:12.8} kinetic={:12.8} nuclear={:12.8} 2e={:12.8} total={:12.8} fd_vnn={:12.8} fd_total={:12.8}",
                    atom, c, vnn[(atom, c)], overlap[(atom, c)], kinetic[(atom, c)],
                    nuclear[(atom, c)], twoelec[(atom, c)], total[(atom, c)],
                    fd_vnn[(atom, c)], fd_total[(atom, c)]
                );
            }
        }
        // Verify total matches fd_total
        for atom in 0..natoms {
            for c in 0..3 {
                let diff = (total[(atom, c)] - fd_total[(atom, c)]).abs();
                assert!(diff < 1e-5,
                    "atom={atom} coord={c}: total={:.8} fd={:.8} diff={:.2e}",
                    total[(atom, c)], fd_total[(atom, c)], diff);
            }
        }
    }

    fn finite_difference_gradient(xyz: &str, basis_name: &str, delta: f64) -> Array2<f64> {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let natoms = mol.atoms.len();
        let mut grad = Array2::zeros((natoms, 3));
        let config = RhfConfig { energy_conv: 1e-10, ..Default::default() };

        for atom in 0..natoms {
            for coord in 0..3 {
                let mut mol_plus = mol.clone();
                let mut mol_minus = mol.clone();
                match coord {
                    0 => { mol_plus.atoms[atom].x += delta; mol_minus.atoms[atom].x -= delta; }
                    1 => { mol_plus.atoms[atom].y += delta; mol_minus.atoms[atom].y -= delta; }
                    _ => { mol_plus.atoms[atom].zpos += delta; mol_minus.atoms[atom].zpos -= delta; }
                }
                let bs = basis::bundled(basis_name).unwrap();
                let prep_p = PreparedBasis::new(&mol_plus, &bs).unwrap();
                let bounds_p = SchwarzBounds::compute(Operator::coulomb(), &prep_p).unwrap();
                let res_p = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol_plus, &prep_p, Operator::coulomb(), &bounds_p, &config).unwrap();

                let prep_m = PreparedBasis::new(&mol_minus, &bs).unwrap();
                let bounds_m = SchwarzBounds::compute(Operator::coulomb(), &prep_m).unwrap();
                let res_m = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol_minus, &prep_m, Operator::coulomb(), &bounds_m, &config).unwrap();

                grad[(atom, coord)] = (res_p.energy - res_m.energy) / (2.0 * delta);
            }
        }
        grad
    }

    #[test]
    fn test_gradient_h2_sto3g_vs_finite_diff() {
        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig { energy_conv: 1e-10, ..Default::default() };
        let result = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &prep, op, &bounds, &config).unwrap();

        let analytic = match rhf_gradient(&mol, &prep, op, &bounds, &result, None) {
            Ok(g) => g,
            Err(FerricError::Libint(msg)) if msg.contains("derivative engine not available") => {
                eprintln!("SKIPPED: {msg}");
                return;
            }
            Err(e) => panic!("unexpected error: {e}"),
        };
        let fd = finite_difference_gradient(xyz, "sto-3g", 1e-5);

        eprintln!("=== H2/STO-3G Gradient ===");
        for atom in 0..2 {
            for c in 0..3 {
                let diff = (analytic[(atom, c)] - fd[(atom, c)]).abs();
                eprintln!("atom={atom} coord={c}: analytic={:.8} fd={:.8} diff={:.2e}",
                    analytic[(atom, c)], fd[(atom, c)], diff);
                assert!(
                    diff < 1e-5,
                    "atom={atom} coord={c}: analytic={:.8} fd={:.8} diff={:.2e}",
                    analytic[(atom, c)], fd[(atom, c)], diff
                );
            }
        }
    }

    /// P1 regression guard: the full RHF gradient (1e pair loops + 2e quartet
    /// loop, both parallelized) must be bit-identical across rayon thread
    /// counts. Water/cc-pVDZ has 78 shell pairs and ~3k screened quartets —
    /// above both serial-fallback thresholds, so the parallel grouped-reduction
    /// path is exercised in both pools.
    #[test]
    fn rhf_gradient_bit_identical_across_thread_counts() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig { energy_conv: 1e-10, ..Default::default() };
        // One SCF solve outside the pools: the same ScfResult feeds both
        // gradient evaluations, so any difference is the gradient's own.
        let result = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol, &prep, op, &bounds, &config,
        )
        .unwrap();

        let run = |threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| rhf_gradient(&mol, &prep, op, &bounds, &result, None).unwrap())
        };
        let g1 = run(1);
        let g4 = run(4);
        for (a, (v1, v4)) in g1.iter().zip(g4.iter()).enumerate() {
            assert_eq!(
                v1.to_bits(),
                v4.to_bits(),
                "gradient element {a} differs across thread counts: {v1:e} vs {v4:e}"
            );
        }
    }

    #[test]
    fn test_gradient_h2o_sto3g_vs_finite_diff() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig { energy_conv: 1e-10, ..Default::default() };
        let result = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &prep, op, &bounds, &config).unwrap();

        let analytic = match rhf_gradient(&mol, &prep, op, &bounds, &result, None) {
            Ok(g) => g,
            Err(FerricError::Libint(msg)) if msg.contains("derivative engine not available") => {
                eprintln!("SKIPPED: {msg}");
                return;
            }
            Err(e) => panic!("unexpected error: {e}"),
        };
        let fd = finite_difference_gradient(xyz, "sto-3g", 1e-5);

        eprintln!("=== H2O/STO-3G Gradient ===");
        for atom in 0..3 {
            for c in 0..3 {
                let diff = (analytic[(atom, c)] - fd[(atom, c)]).abs();
                eprintln!("atom={atom} coord={c}: analytic={:.8} fd={:.8} diff={:.2e}",
                    analytic[(atom, c)], fd[(atom, c)], diff);
                assert!(
                    diff < 1e-5,
                    "atom={atom} coord={c}: analytic={:.8} fd={:.8} diff={:.2e}",
                    analytic[(atom, c)], fd[(atom, c)], diff
                );
            }
        }
    }
}
