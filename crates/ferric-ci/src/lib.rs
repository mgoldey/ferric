//! # ferric-ci — CAS-CI, RDMs and two-step CASSCF
//!
//! Full CI within a small user-specified active space, on top of a converged
//! restricted (closed-shell) RHF reference. Given `n_active` active spatial
//! orbitals and `n_elec_active` active electrons, this:
//!
//! 1. Transforms the AO integrals to the active-space MO basis (a one-time
//!    quarter-transform of the dense AO ERIs, plus the folded inactive-core
//!    effective potential and core energy) — see [`integrals`].
//! 2. Enumerates the α- and β-strings and forms the full determinant list as
//!    their outer product — see [`strings`].
//! 3. Builds `sigma = H c` by the string-driven direct algorithm
//!    (Knowles–Handy / Olsen σ1/σ2/σ3 over single-excitation string lists) —
//!    see [`sigma`]. The Slater–Condon rules in [`hamiltonian`] give the
//!    diagonal and an `O(N_det²)` pair-loop oracle used by the tests.
//! 4. Diagonalizes for the lowest root via a CI-native Davidson solver — see
//!    [`davidson`]. When Davidson stops above `conv_thresh` and the space has
//!    at most [`DENSE_SOLVE_MAX_DETS`] determinants, the dense Hamiltonian is
//!    diagonalized instead, so thresholds below Davidson's reach are met on
//!    small spaces.
//! 5. Optionally ([`CasCiConfig::compute_rdms`]) builds the spin-summed 1- and
//!    2-particle reduced density matrices of that root from the same string
//!    lists — see [`rdm`]. [`rdm3`] adds the 3-RDM, the two `(aa|aa)`-
//!    contracted 4-RDM intermediates and the hole RDMs used by NEVPT2.
//!
//! [`casscf`] adds two-step CASSCF on top: [`run_cas_ci_on_integrals`] with
//! RDMs in the current orbitals, the generalized-Fock orbital gradient and an
//! inverse-BFGS step `C ← C exp(K)`, single root.
//!
//! The total energy is `E_CASCI = E_core + lambda_min`, where `E_core` is the
//! nuclear repulsion plus the inactive (frozen-core) electronic energy, and
//! `lambda_min` is the lowest eigenvalue of the active-space CI Hamiltonian.
//! When the active space is the full MO space (no inactive orbitals), CAS-CI is
//! exactly FCI in that basis.
//!
//! ## Not implemented
//!
//! * **RAS1 / RAS3 hole/particle restrictions** (Restricted Active Space): the
//!   *complete* active space is enumerated with no occupation restrictions on
//!   any orbital subspace.
//! * **Open-shell / non-singlet references, S² handling, multiple roots**: the
//!   reference is a closed-shell RHF; `n_alpha == n_beta` is not checked, and
//!   only the single lowest root is extracted.
//! * **NEVPT2 energy.** [`rdm3`] provides its RDM inputs only.
//! * **CLI wiring.** Python exposes `ferric.run_casci` and `ferric.run_casscf`
//!   (`crates/ferric-python/src/ci.rs`); there is no CLI `method.kind`.
//!
//! ## Scaling
//!
//! One sigma call costs `O(N_det · n_exc_α · n_exc_β)` (σ3) with
//! `n_exc = k(n−k) + k` single replacements per string; it never visits
//! determinant pairs. The production memory peak is the Davidson subspace,
//! `2 · max_subspace · N_det` doubles. The dense Hamiltonian
//! ([`hamiltonian::dense_hamiltonian`]) and [`hamiltonian::sigma_naive`] are
//! `O(N_det²)` and are reached only from tests and benchmarks.

/// Two-step CASSCF on top of the CAS-CI solver and its RDMs.
pub mod casscf;
/// CI-native Davidson eigensolver for the lowest root.
pub mod davidson;
/// Slater-Condon matrix elements and sigma-vector build.
pub mod hamiltonian;
/// Active-space MO integral transform.
pub mod integrals;
/// Spin-summed 1- and 2-particle reduced density matrices.
pub mod rdm;
/// 3-RDM, contracted 4-RDM intermediates and hole RDMs (NEVPT2 inputs).
pub mod rdm3;
/// String-driven direct sigma build (σ1/σ2/σ3).
pub mod sigma;
/// Determinant-string enumeration (bitmask α/β strings).
pub mod strings;

pub use casscf::{run_casscf, CasScfConfig, CasScfResult};
pub use rdm::CasCiRdms;
pub use rdm3::{hole_rdms, nevpt_rdms, HoleRdms, NevptRdms};

use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::ScfResult;

/// Configuration for a CAS-CI calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CasCiConfig {
    /// Number of active spatial orbitals.
    pub n_active: usize,
    /// Active electrons split as `(n_alpha, n_beta)`. For a closed-shell
    /// singlet, `n_alpha == n_beta == n_elec_active / 2`.
    pub n_elec_active: (usize, usize),
    /// Index of the first active MO. MOs `0..active_start` are the
    /// doubly-occupied inactive (frozen-core) orbitals.
    pub active_start: usize,
    /// Residual convergence threshold (Davidson, or the dense eigensolve; see
    /// [`DENSE_SOLVE_MAX_DETS`]).
    pub conv_thresh: f64,
    /// Maximum Davidson subspace dimension before collapse/restart.
    pub max_subspace: usize,
    /// Hard Davidson iteration cap.
    pub max_iter: usize,
    /// Memory ceiling in bytes, resolved through
    /// [`ferric_core::memory::resolve_budget_bytes`]. `None` does NOT mean
    /// unlimited — it falls through to `FERRIC_MEM_BUDGET_GB`, then to
    /// `0.8 x` detected available RAM, then to a 2 GiB floor.
    ///
    /// This bounds the Davidson subspace, which is the production peak:
    /// `2 * max_subspace * N_det * 8` bytes (two bases, `v_basis` and
    /// `hv_basis`). N_det grows combinatorially with the active space, so the
    /// peak crosses this box between CAS(12,14) (7.2 GB) and CAS(14,16)
    /// (105 GB). See `tests/mwe_casci_has_no_guard.rs`.
    pub memory_budget_bytes: Option<usize>,
    /// Build the spin-summed 1- and 2-RDMs of the converged root
    /// ([`CasCiResult::rdms`]). Their scratch is charged to the same memory
    /// plan as the Davidson subspace; see [`rdm::rdm_memory_elems`].
    pub compute_rdms: bool,
}

impl Default for CasCiConfig {
    fn default() -> Self {
        CasCiConfig {
            n_active: 0,
            n_elec_active: (0, 0),
            active_start: 0,
            conv_thresh: 1e-9,
            max_subspace: 50,
            max_iter: 200,
            memory_budget_bytes: None,
            compute_rdms: false,
        }
    }
}

/// Result of a CAS-CI calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct CasCiResult {
    /// Total CAS-CI energy: `e_core + e_active`.
    pub e_total: f64,
    /// Active-space energy `e_total − e_core` (PySCF `CASCI.e_cas`). The CI
    /// Hamiltonian carries `e_core` on its diagonal, so its lowest eigenvalue
    /// is `e_total`; this field removes it.
    pub e_active: f64,
    /// Additive core (nuclear + inactive electronic) energy.
    pub e_core: f64,
    /// Number of determinants in the CI expansion.
    pub n_determinants: usize,
    /// Normalized CI coefficient vector, indexed `d = ia * n_beta + ib`.
    pub ci_vector: Vec<f64>,
    /// Whether the returned root meets `conv_thresh`: Davidson's flag, or,
    /// after the dense fallback ([`DENSE_SOLVE_MAX_DETS`]), `residual_norm <
    /// conv_thresh`.
    pub converged: bool,
    /// Davidson iterations taken (the dense fallback adds none).
    pub iterations: usize,
    /// `‖H c − E c‖` of the returned (normalized) `ci_vector` and `e_total`,
    /// from one extra sigma build.
    pub residual_norm: f64,
    /// Spin-summed RDMs of `ci_vector` over the active orbitals, when
    /// [`CasCiConfig::compute_rdms`] is set.
    pub rdms: Option<CasCiRdms>,
    /// The active-space integrals the CI Hamiltonian was built from (`h`
    /// with the inactive-core potential, `(pq|rs)`, `e_core`), so the energy
    /// can be rebuilt from the RDMs ([`rdm::rdm_energy`]).
    pub integrals: integrals::ActiveSpaceIntegrals,
}

impl std::fmt::Display for CasCiResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CAS-CI energy: {:.10} Ha ({} dets, {} iters, converged: {})",
            self.e_total, self.n_determinants, self.iterations, self.converged
        )
    }
}

/// Run a plain CAS-CI calculation from a converged restricted RHF reference.
///
/// The Hamiltonian is `h = T + V_nuc + V_ECP` (the ECP of `prep`'s basis, as
/// the RHF builds it) and the exact AO ERIs. An external potential, solvation
/// or embedding term of the reference is not recorded on [`ScfResult`] and is
/// not included, so the reference must be a vacuum RHF.
///
/// Validates the active-space configuration against the molecule/basis up front
/// (electron count, orbital window) and returns a clean [`FerricError`] on any
/// inconsistency — it never panics deep inside the transform or matvec.
pub fn run_cas_ci(
    mol: &Molecule,
    prep: &PreparedBasis,
    rhf: &ScfResult,
    config: CasCiConfig,
) -> Result<CasCiResult, FerricError> {
    let (n_alpha, n_beta) = config.n_elec_active;
    let n_active = config.n_active;
    let active_start = config.active_start;

    // ---- Validation ----------------------------------------------------
    if !matches!(rhf.spin, ferric_scf::Spin::Restricted) {
        return Err(FerricError::General(
            "CAS-CI requires a restricted (closed-shell) RHF reference".to_string(),
        ));
    }
    check_active_config(&config)?;
    // Consistency: inactive orbitals are doubly occupied; the active electrons
    // plus 2*inactive must equal the molecule's electron count.
    let n_inactive = active_start;
    let nelec_mol = mol.nelec() as usize;
    let nelec_accounted = 2 * n_inactive + n_alpha + n_beta;
    if nelec_accounted != nelec_mol {
        return Err(FerricError::General(format!(
            "CAS-CI: electron bookkeeping mismatch — 2*inactive({n_inactive}) + \
             active(α={n_alpha}+β={n_beta}) = {nelec_accounted} != molecule nelec {nelec_mol}"
        )));
    }

    // ---- Active-space integrals ---------------------------------------
    let ints = integrals::build_active_space_integrals(
        mol,
        prep,
        rhf,
        active_start,
        n_active,
        config.memory_budget_bytes,
    )?;
    run_cas_ci_on_integrals(ints, &config)
}

/// Orbital-window checks shared by [`run_cas_ci`] and
/// [`run_cas_ci_on_integrals`].
fn check_active_config(config: &CasCiConfig) -> Result<(), FerricError> {
    let (n_alpha, n_beta) = config.n_elec_active;
    let n_active = config.n_active;
    if n_active == 0 {
        return Err(FerricError::General(
            "CAS-CI: n_active must be >= 1".to_string(),
        ));
    }
    if n_active > strings::MAX_ACTIVE_ORBITALS {
        return Err(FerricError::General(format!(
            "CAS-CI: n_active = {n_active} exceeds the {}-orbital u64-string limit \
             (spike-scale only)",
            strings::MAX_ACTIVE_ORBITALS
        )));
    }
    if n_alpha > n_active || n_beta > n_active {
        return Err(FerricError::General(format!(
            "CAS-CI: active electrons per spin (α={n_alpha}, β={n_beta}) cannot exceed \
             n_active ({n_active})"
        )));
    }
    Ok(())
}

/// CAS-CI on caller-supplied active-space integrals: the lowest root of the
/// `config.n_elec_active` sector of the Hamiltonian `(ints.h, ints.eri,
/// ints.e_core)`, with RDMs when [`CasCiConfig::compute_rdms`] is set.
///
/// [`CasCiConfig::active_start`] is not read: the integrals already fold the
/// inactive orbitals into `ints.h` and `ints.e_core`. Two-step CASSCF
/// ([`casscf`]) calls this once per macro iteration with integrals in the
/// current orbitals.
pub fn run_cas_ci_on_integrals(
    ints: integrals::ActiveSpaceIntegrals,
    config: &CasCiConfig,
) -> Result<CasCiResult, FerricError> {
    check_active_config(config)?;
    let (n_alpha, n_beta) = config.n_elec_active;
    let n_active = config.n_active;
    let n4 = n_active.pow(4);
    if ints.n_active != n_active || ints.h.dim() != (n_active, n_active) || ints.eri.len() != n4 {
        return Err(FerricError::General(format!(
            "CAS-CI: integrals for {} active orbitals (h {:?}, eri len {}) do not match \
             n_active = {n_active}",
            ints.n_active,
            ints.h.dim(),
            ints.eri.len()
        )));
    }

    // ---- Determinant-space size --------------------------------------
    // The string counts are binomials, computed without enumerating, so the
    // pre-flight below refuses an oversized space before the string lists
    // exist. A count that overflows `usize` saturates and fails the budget.
    let na_str = strings::count_strings(n_active, n_alpha).unwrap_or(usize::MAX);
    let nb_str = strings::count_strings(n_active, n_beta).unwrap_or(usize::MAX);
    if na_str == 0 || nb_str == 0 {
        return Err(FerricError::General(
            "CAS-CI: empty determinant space (check n_active / electron counts)".to_string(),
        ));
    }
    let ndet = na_str.saturating_mul(nb_str);

    // ---- Memory pre-flight --------------------------------------------
    // Davidson holds TWO bases (v_basis + hv_basis), each up to max_subspace
    // vectors of ndet doubles, plus seven ndet work vectors (counted below).
    // ndet grows combinatorially with the active space, so this crosses a
    // 23 GB box between CAS(12,14) (7.2 GB) and CAS(14,16) (105 GB) -- and
    // before this gate nothing stopped it. max_subspace is read from config,
    // not assumed: it is a user-settable memory knob whether or not it was
    // designed as one. It is clamped with `.max(2)` exactly as
    // `davidson_lowest` clamps it.
    //
    // `MemoryPlan::from_global_pool` re-bases the ceiling on what the
    // process-global pool has LEFT (a no-op, resolving exactly as before,
    // when no pool is installed) — the whole point being that this gate must
    // not pass while believing it owns the entire budget when another
    // subsystem is already holding some of it.
    let mut plan = MemoryPlan::from_global_pool(
        config.memory_budget_bytes,
        format!(
            "CAS-CI Davidson subspace (n_active={n_active}, n_elec=({n_alpha},{n_beta}), \
             N_det={ndet}, max_subspace={})",
            config.max_subspace
        ),
    );
    plan.reserve_sized(
        "alpha and beta determinant string lists",
        na_str.saturating_add(nb_str),
        std::mem::size_of::<u64>(),
        Lifetime::Resident,
        1,
    );
    plan.reserve(
        "Davidson v_basis + hv_basis",
        ndet.saturating_mul(config.max_subspace.max(2))
            .saturating_mul(2),
        Lifetime::Resident,
    );
    // Live alongside the bases at the restart branch of `davidson_lowest`:
    // diag, x, hx, resid, best_vec, t, x_unit. `guess` is moved into
    // v_basis, so it is already counted there.
    plan.reserve(
        "Davidson diag/x/hx/resid/best_vec/t/x_unit work vectors",
        ndet.saturating_mul(7),
        Lifetime::Resident,
    );
    // Direct sigma (`sigma::DirectSigma::apply`): the output vector, the
    // transposed C and its σ1 image, and the σ3 gather C' (at most
    // n_alpha_str x n_beta_str) are live together at worst.
    plan.reserve(
        "direct sigma output/transpose/sigma1/gather vectors",
        ndet.saturating_mul(4),
        Lifetime::Resident,
    );
    // Excitation lists: `n_str * (k(n-k) + k)` 16-byte entries per channel,
    // the β list stored twice (per string and grouped by pair).
    let n_exc = |k: usize| k * (n_active - k) + k;
    plan.reserve(
        "direct sigma excitation lists",
        2usize.saturating_mul(
            na_str
                .saturating_mul(n_exc(n_alpha))
                .saturating_add(nb_str.saturating_mul(n_exc(n_beta)).saturating_mul(2)),
        ),
        Lifetime::Resident,
    );
    if config.compute_rdms {
        // Built after the Davidson solve, but charged on top of it: the
        // plan has no stage ordering between Resident reservations.
        let (per_worker, transient, resident) = rdm::rdm_memory_elems(n_active, na_str, nb_str);
        plan.reserve_per_worker(
            "RDM block D = E_rs C",
            per_worker,
            rdm::rdm_concurrent_blocks(na_str, nb_str, rayon::current_num_threads()),
        );
        plan.reserve(
            "RDM block results (one batch)",
            transient,
            Lifetime::Transient,
        );
        plan.reserve("RDM accumulators and dm1/dm2", resident, Lifetime::Resident);
    }
    // Not charged: per-rayon-worker scratch of `O(n_str)` doubles (σ1/σ2
    // row coefficients, the σ3 accumulator), under 0.3 MB per worker even at
    // CAS(14,16) where the subspace above is 105 GB.
    // Held for the whole solve below: both bases and the work vectors are
    // live for davidson_lowest's entire call, and are dropped when it
    // returns, so the guard's scope matches exactly.
    //
    // `_mem` MUST stay a named binding. `let _ = plan.commit()?` drops the
    // reservation immediately, crediting the pool before Davidson allocates,
    // and no test catches it: nothing else reserves from the pool during the
    // solve, and outstanding bytes are not observable mid-run (the pool's
    // high-water mark survives the drop).
    let _mem = plan.commit()?;

    // ---- Determinant space --------------------------------------------
    let space = hamiltonian::DeterminantSpace {
        alpha_strings: strings::enumerate_strings(n_active, n_alpha),
        beta_strings: strings::enumerate_strings(n_active, n_beta),
    };
    debug_assert_eq!(space.n_det(), ndet);

    // ---- Diagonal + Davidson solve ------------------------------------
    let diag = hamiltonian::hamiltonian_diagonal(&ints, &space);
    let direct = sigma::DirectSigma::new(&ints, &space);
    let matvec = |c: &[f64]| direct.apply(c);
    let mut dav = davidson::davidson_lowest(
        ndet,
        matvec,
        &diag,
        config.conv_thresh,
        config.max_subspace,
        config.max_iter,
    )?;
    let mut residual_norm = eigen_residual(&direct, &dav);
    if !(dav.converged && residual_norm < config.conv_thresh) && ndet <= DENSE_SOLVE_MAX_DETS {
        // Davidson's residual floor is set by its Gram-Schmidt round-off and
        // can sit above a tight threshold; a direct eigensolve reaches the
        // rounding floor of `H` itself.
        let mut dense_plan = MemoryPlan::from_global_pool(
            config.memory_budget_bytes,
            format!("CAS-CI dense eigensolve (N_det={ndet})"),
        );
        dense_plan.reserve(
            "dense CI Hamiltonian, eigenvectors and eigenvalues",
            ndet.saturating_mul(ndet)
                .saturating_mul(2)
                .saturating_add(ndet),
            Lifetime::Transient,
        );
        let _dense_mem = dense_plan.commit()?;
        let dense = dense_lowest(&ints, &space, dav.iterations, config.memory_budget_bytes)?;
        let dense_residual = eigen_residual(&direct, &dense);
        if dense_residual < residual_norm {
            dav = dense;
            residual_norm = dense_residual;
            dav.converged = residual_norm < config.conv_thresh;
        }
    }

    let rdms = if config.compute_rdms {
        Some(rdm::spin_summed_rdms(&direct, &dav.eigenvector)?)
    } else {
        None
    };
    drop(direct);

    Ok(CasCiResult {
        e_total: dav.eigenvalue,
        e_active: dav.eigenvalue - ints.e_core,
        e_core: ints.e_core,
        n_determinants: ndet,
        ci_vector: dav.eigenvector,
        converged: dav.converged,
        iterations: dav.iterations,
        residual_norm,
        rdms,
        integrals: ints,
    })
}

/// Largest determinant space diagonalized densely when Davidson stops above
/// `conv_thresh`: the dense `H` and its eigenvectors are `2 · 1024² · 8` B =
/// 16 MiB, and the eigensolve costs about `10 N_det³` = 1e10 flop at the
/// limit.
pub const DENSE_SOLVE_MAX_DETS: usize = 1024;

/// `‖H x − θ x‖` of an eigenpair from the direct sigma.
fn eigen_residual(direct: &sigma::DirectSigma<'_>, pair: &davidson::CiDavidsonResult) -> f64 {
    let hc = direct.apply(&pair.eigenvector);
    hc.iter()
        .zip(&pair.eigenvector)
        .map(|(h, c)| (h - pair.eigenvalue * c).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Lowest eigenpair of the dense CI Hamiltonian (LAPACK `eigh`), with the
/// sign fixed so the largest-magnitude coefficient is positive. `iterations`
/// is carried through from the preceding Davidson solve; `converged` is set
/// by the caller from the residual.
pub fn dense_lowest(
    ints: &integrals::ActiveSpaceIntegrals,
    space: &hamiltonian::DeterminantSpace,
    iterations: usize,
    memory_budget_bytes: Option<usize>,
) -> Result<davidson::CiDavidsonResult, FerricError> {
    use ndarray_linalg::{Eigh, UPLO};
    let h = hamiltonian::dense_hamiltonian(ints, space, memory_budget_bytes)?;
    let (w, v) = h
        .eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("CAS-CI dense eigh failed: {e}")))?;
    let mut x: Vec<f64> = v.column(0).to_vec();
    let lead = x
        .iter()
        .fold(0.0f64, |m, &c| if c.abs() > m.abs() { c } else { m });
    if lead < 0.0 {
        x.iter_mut().for_each(|c| *c = -*c);
    }
    Ok(davidson::CiDavidsonResult {
        eigenvalue: w[0],
        eigenvector: x,
        iterations,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    // ---- PySCF reference generation --------------------------------------
    // All reference energies below were generated with PySCF 2.13.0, using the
    // *same* geometries (in Angstrom) and basis (STO-3G) as the ferric runs.
    // The exact PySCF calls are cited per-test. Command template:
    //
    //   from pyscf import gto, scf, fci, mcscf
    //   mol = gto.M(atom=..., basis="sto-3g", unit="Angstrom")
    //   mf  = scf.RHF(mol); mf.kernel()
    //   e_fci, _ = fci.FCI(mf).kernel()               # full-space FCI
    //   e_cas, *_ = mcscf.CASCI(mf, ncas, nelecas).kernel()  # frozen-core CASCI

    fn rhf_for(xyz: &str) -> (Molecule, PreparedBasis, ScfResult) {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &prep,
            op,
            &bounds,
            &RhfConfig {
                // density_conv (ΔP RMS) is the primary/tight SCF gate in ferric
                // (energy_conv is only a loose "still descending" sanity bound —
                // see rhf.rs). Tighten it so the reference RHF energy matches
                // PySCF's to well under the 1e-8 CAS-CI comparison tolerance.
                energy_conv: 1e-10,
                density_conv: 1e-9,
                ..Default::default()
            },
        )
        .unwrap();
        (mol, prep, rhf)
    }

    // ---- On the two comparison tolerances used below ---------------------
    //
    // Two numbers are checked against PySCF:
    //
    //  (a) the *correlation* energy  E_CASCI - E_RHF  vs  E_FCI - E_RHF(PySCF).
    //      This isolates the CI machinery from the mean-field floor and is the
    //      real test of this crate. It matches PySCF to ~1e-9 (H2 to ~1e-10).
    //
    //  (b) the absolute total energy, checked to a looser 1e-7. ferric's own
    //      RHF total energy sits ~2.4e-8 below PySCF's for H2O/STO-3G (a
    //      pre-existing ferric-vs-PySCF integral/SCF-precision offset, NOT a
    //      CAS-CI artifact: the offset is identical in RHF and in FCI, and the
    //      full-FCI energy is orbital-invariant, so it can only come from the
    //      shared mean-field floor). That same offset rides through to the CAS
    //      total energy unchanged; the 1e-7 window accommodates it while still
    //      pinning the absolute number. For H2 the RHF floor matches PySCF to
    //      ~1e-10, so H2's total energy is asserted at the tight 1e-8.

    /// H2 / STO-3G, full active space (2 orbitals, 2 electrons) = FCI = exact
    /// diagonalization of the 2-electron problem in the minimal basis.
    ///
    /// PySCF reference:
    ///   mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Angstrom")
    ///   mf  = scf.RHF(mol); mf.kernel()         # -1.1167593073964255
    ///   e_fci, _ = fci.FCI(mf).kernel()          # -1.1372838344885023
    /// The Davidson gate must REFUSE a starved budget rather than allocate.
    ///
    /// H2/STO-3G is the smallest possible CAS-CI (N_det = 4), so a 1-byte
    /// budget is the only way to force a refusal at this scale -- but the point
    /// is structural: before this gate nothing bounded
    /// `2 * max_subspace * N_det * 8`, which reaches 105 GB at CAS(14,16) on a
    /// 23 GB box. See tests/mwe_casci_has_no_guard.rs for the scaling.
    #[test]
    fn casci_refuses_a_starved_budget() {
        let (mol, prep, rhf) = rhf_for("2\nH2\nH 0 0 0\nH 0 0 0.74\n");
        let cfg = CasCiConfig {
            n_active: 2,
            n_elec_active: (1, 1),
            active_start: 0,
            memory_budget_bytes: Some(1),
            ..Default::default()
        };
        // `match` rather than `expect_err`: CasCiResult does not implement Debug.
        let msg = match run_cas_ci(&mol, &prep, &rhf, cfg) {
            Ok(_) => panic!("a 1-byte budget must be refused before allocating"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("CAS-CI"),
            "message must name the method: {msg}"
        );
        assert!(
            msg.contains("budget is"),
            "message must name the budget: {msg}"
        );
        // NOT asserted here: that the message names `N_det`. At H2/STO-3G the
        // dense-AO-ERI gate is reached first, so a 1-byte budget can only ever
        // refuse there -- the Davidson N_det gate is downstream of it and never
        // runs. Pinning N_det at this scale would assert on a path this system
        // cannot take. The N_det term is covered where it actually bites, in
        // tests/mwe_casci_has_no_guard.rs.
    }

    /// The complement: an ample budget must still RUN. A gate that only ever
    /// refuses is as broken as none at all -- it becomes a wall and trains
    /// users to inflate budgets until it stops complaining.
    #[test]
    fn casci_runs_under_an_ample_budget() {
        let (mol, prep, rhf) = rhf_for("2\nH2\nH 0 0 0\nH 0 0 0.74\n");
        let cfg = CasCiConfig {
            n_active: 2,
            n_elec_active: (1, 1),
            active_start: 0,
            memory_budget_bytes: Some(4 * 1024 * 1024 * 1024),
            ..Default::default()
        };
        let res = match run_cas_ci(&mol, &prep, &rhf, cfg) {
            Ok(r) => r,
            Err(e) => panic!("H2/STO-3G CAS-CI must fit a 4 GiB budget: {e}"),
        };
        assert!(res.e_total.is_finite(), "energy must be finite");
        assert!(res.converged, "must converge");
    }

    #[test]
    fn h2_sto3g_full_fci() {
        const PYSCF_RHF: f64 = -1.1167593073964255;
        const PYSCF_FCI: f64 = -1.1372838344885023;
        const PYSCF_CORR: f64 = PYSCF_FCI - PYSCF_RHF;
        let (mol, prep, rhf) = rhf_for("2\nH2\nH 0 0 0\nH 0 0 0.74\n");
        let cfg = CasCiConfig {
            n_active: 2,
            n_elec_active: (1, 1),
            active_start: 0,
            ..Default::default()
        };
        let res = run_cas_ci(&mol, &prep, &rhf, cfg).unwrap();
        let corr = res.e_total - rhf.energy;
        eprintln!(
            "H2/STO-3G  RHF={:.12}  CAS-CI={:.12}  corr={:.12}  ndet={}  (PySCF FCI {PYSCF_FCI:.12}, corr {PYSCF_CORR:.12})",
            rhf.energy, res.e_total, corr, res.n_determinants
        );
        assert_eq!(res.n_determinants, 4);
        assert!(res.converged, "Davidson did not converge");
        // Correlation energy (the CI observable): tight.
        assert!(
            (corr - PYSCF_CORR).abs() < 1e-8,
            "H2 corr {corr} vs PySCF {PYSCF_CORR} (diff {:.2e})",
            (corr - PYSCF_CORR).abs()
        );
        // H2 RHF floor matches PySCF to ~1e-10, so total energy is tight too.
        assert!(
            (res.e_total - PYSCF_FCI).abs() < 1e-8,
            "H2 CAS-CI {} vs PySCF FCI {PYSCF_FCI} (diff {:.2e})",
            res.e_total,
            (res.e_total - PYSCF_FCI).abs()
        );
    }

    /// H2O / STO-3G, full active space (7 orbitals, 10 electrons) = FCI.
    /// C(7,5)^2 = 441 determinants. The full-space FCI energy is invariant to
    /// the specific orbital rotation, so this is a robust exactness check that
    /// does not depend on matching PySCF's orbitals.
    ///
    /// PySCF reference:
    ///   mol = gto.M(atom="O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692;
    ///                     H 0.0 -0.7572 -0.4692", basis="sto-3g", unit="Angstrom")
    ///   mf  = scf.RHF(mol); mf.kernel()          # -74.96302313846127
    ///   e_fci, _ = fci.FCI(mf).kernel()           # -75.0125782410909
    #[test]
    fn h2o_sto3g_full_fci() {
        const PYSCF_RHF: f64 = -74.96302313846127;
        const PYSCF_FCI: f64 = -75.0125782410909;
        const PYSCF_CORR: f64 = PYSCF_FCI - PYSCF_RHF;
        let xyz = "3\nH2O\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
        let (mol, prep, rhf) = rhf_for(xyz);
        let cfg = CasCiConfig {
            n_active: 7,
            n_elec_active: (5, 5),
            active_start: 0,
            ..Default::default()
        };
        let res = run_cas_ci(&mol, &prep, &rhf, cfg).unwrap();
        let corr = res.e_total - rhf.energy;
        eprintln!(
            "H2O/STO-3G RHF={:.12}  CAS-CI(full FCI)={:.12}  corr={:.12}  ndet={}  (PySCF FCI {PYSCF_FCI:.12}, corr {PYSCF_CORR:.12})",
            rhf.energy, res.e_total, corr, res.n_determinants
        );
        assert_eq!(res.n_determinants, 441);
        assert!(res.converged, "Davidson did not converge");
        // Correlation energy is the CI observable — matches PySCF to ~1e-9.
        assert!(
            (corr - PYSCF_CORR).abs() < 1e-8,
            "H2O full-FCI corr {corr} vs PySCF {PYSCF_CORR} (diff {:.2e})",
            (corr - PYSCF_CORR).abs()
        );
        // Absolute total energy: 1e-7 window (ferric mean-field floor sits
        // ~2.4e-8 below PySCF's; see the tolerance note above).
        assert!(
            (res.e_total - PYSCF_FCI).abs() < 1e-7,
            "H2O full FCI {} vs PySCF FCI {PYSCF_FCI} (diff {:.2e})",
            res.e_total,
            (res.e_total - PYSCF_FCI).abs()
        );
    }

    /// LiH / STO-3G, full active space (6 orbitals, 4 electrons) = FCI.
    /// C(6,2)^2 = 225 determinants. Third system (widens past H2/H2O):
    /// heteronuclear, one heavy atom (Li, unlike H2's two identical light
    /// atoms or H2O's single O), full-space FCI so this is orbital-
    /// rotation-invariant like the H2O full-FCI test above.
    ///
    /// PySCF reference:
    ///   mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", unit="Angstrom")
    ///   mf  = scf.RHF(mol); mf.kernel(conv_tol=1e-12)  # -7.861864769808649
    ///   e_fci, _ = fci.FCI(mf).kernel()                 # -7.882324378883495
    #[test]
    fn lih_sto3g_full_fci() {
        const PYSCF_RHF: f64 = -7.861864769808649;
        const PYSCF_FCI: f64 = -7.882324378883495;
        const PYSCF_CORR: f64 = PYSCF_FCI - PYSCF_RHF;
        let xyz = "2\nLiH\nLi 0.0 0.0 0.0\nH 0.0 0.0 1.6\n";
        let (mol, prep, rhf) = rhf_for(xyz);
        let cfg = CasCiConfig {
            n_active: 6,
            n_elec_active: (2, 2),
            active_start: 0,
            ..Default::default()
        };
        let res = run_cas_ci(&mol, &prep, &rhf, cfg).unwrap();
        let corr = res.e_total - rhf.energy;
        eprintln!(
            "LiH/STO-3G RHF={:.12}  CAS-CI(full FCI)={:.12}  corr={:.12}  ndet={}  (PySCF FCI {PYSCF_FCI:.12}, corr {PYSCF_CORR:.12})",
            rhf.energy, res.e_total, corr, res.n_determinants
        );
        assert_eq!(res.n_determinants, 225);
        assert!(res.converged, "Davidson did not converge");
        assert!(
            (corr - PYSCF_CORR).abs() < 1e-8,
            "LiH full-FCI corr {corr} vs PySCF {PYSCF_CORR} (diff {:.2e})",
            (corr - PYSCF_CORR).abs()
        );
        assert!(
            (res.e_total - PYSCF_FCI).abs() < 1e-7,
            "LiH full FCI {} vs PySCF FCI {PYSCF_FCI} (diff {:.2e})",
            res.e_total,
            (res.e_total - PYSCF_FCI).abs()
        );
    }

    /// H2O / STO-3G, frozen-core CASCI(6,8): freeze the lowest (O 1s) orbital,
    /// active = 6 orbitals / 8 electrons. Unlike full FCI, this DOES depend on
    /// the RHF orbitals, so it also confirms ferric's RHF matches PySCF's.
    /// C(6,4)^2 = 225 determinants.
    ///
    /// PySCF reference:
    ///   mf  = scf.RHF(mol); mf.kernel()
    ///   e_cas, *_ = mcscf.CASCI(mf, 6, 8).kernel()  # -75.01250015394263
    #[test]
    fn h2o_sto3g_casci_frozen_core() {
        const PYSCF_RHF: f64 = -74.96302313846127;
        const PYSCF_CASCI: f64 = -75.01250015394263;
        // CASCI correlation relative to the *same* RHF reference PySCF used.
        const PYSCF_CORR: f64 = PYSCF_CASCI - PYSCF_RHF;
        let xyz = "3\nH2O\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
        let (mol, prep, rhf) = rhf_for(xyz);
        let cfg = CasCiConfig {
            n_active: 6,
            n_elec_active: (4, 4),
            active_start: 1, // freeze O 1s
            ..Default::default()
        };
        let res = run_cas_ci(&mol, &prep, &rhf, cfg).unwrap();
        let corr = res.e_total - rhf.energy;
        eprintln!(
            "H2O/STO-3G RHF={:.12}  CASCI(6,8) fc={:.12}  corr={:.12}  ndet={}  (PySCF CASCI {PYSCF_CASCI:.12}, corr {PYSCF_CORR:.12})",
            rhf.energy, res.e_total, corr, res.n_determinants
        );
        assert_eq!(res.n_determinants, 225);
        assert!(res.converged, "Davidson did not converge");
        // Correlation energy (CI observable): matches PySCF to ~1e-9. This also
        // confirms ferric's frozen-core partitioning matches PySCF's CASCI,
        // since a frozen-core correlation energy is orbital-dependent.
        assert!(
            (corr - PYSCF_CORR).abs() < 1e-8,
            "H2O CASCI(6,8) corr {corr} vs PySCF {PYSCF_CORR} (diff {:.2e})",
            (corr - PYSCF_CORR).abs()
        );
        // Absolute total energy: 1e-7 window (ferric mean-field floor offset).
        assert!(
            (res.e_total - PYSCF_CASCI).abs() < 1e-7,
            "H2O CASCI(6,8) {} vs PySCF {PYSCF_CASCI} (diff {:.2e})",
            res.e_total,
            (res.e_total - PYSCF_CASCI).abs()
        );
    }

    /// Cross-check: the dense-Hamiltonian eigensolve and the Davidson solve
    /// must agree, and both must equal e_core + lowest eigenvalue. This guards
    /// the Davidson path against the dense reference independently of PySCF.
    #[test]
    fn dense_vs_davidson_h2o_casci() {
        use ndarray_linalg::Eigh;
        let xyz = "3\nH2O\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
        let (mol, prep, rhf) = rhf_for(xyz);
        let cfg = CasCiConfig {
            n_active: 6,
            n_elec_active: (4, 4),
            active_start: 1,
            ..Default::default()
        };
        let ints = integrals::build_active_space_integrals(&mol, &prep, &rhf, 1, 6, None).unwrap();
        let space = hamiltonian::DeterminantSpace {
            alpha_strings: strings::enumerate_strings(6, 4),
            beta_strings: strings::enumerate_strings(6, 4),
        };
        let hmat = hamiltonian::dense_hamiltonian(&ints, &space, None).unwrap();
        let (evals, _) = hmat.eigh(ndarray_linalg::UPLO::Upper).unwrap();
        let e_dense = evals[0];

        let res = run_cas_ci(&mol, &prep, &rhf, cfg).unwrap();
        eprintln!(
            "dense min eig = {:.12}, davidson = {:.12}, diff = {:.2e}",
            e_dense,
            res.e_total,
            (e_dense - res.e_total).abs()
        );
        assert!(
            (e_dense - res.e_total).abs() < 1e-9,
            "dense {e_dense} vs davidson {}",
            res.e_total
        );
    }

    /// Validation: malformed config returns a clean Err rather than panicking.
    #[test]
    fn bad_config_errors_cleanly() {
        let (mol, prep, rhf) = rhf_for("2\nH2\nH 0 0 0\nH 0 0 0.74\n");
        // Active window past the MO count.
        let cfg = CasCiConfig {
            n_active: 99,
            n_elec_active: (1, 1),
            active_start: 0,
            ..Default::default()
        };
        assert!(run_cas_ci(&mol, &prep, &rhf, cfg).is_err());

        // Electron bookkeeping mismatch (H2 has 2 electrons; claim 4 active).
        let cfg2 = CasCiConfig {
            n_active: 2,
            n_elec_active: (2, 2),
            active_start: 0,
            ..Default::default()
        };
        assert!(run_cas_ci(&mol, &prep, &rhf, cfg2).is_err());
    }
}
