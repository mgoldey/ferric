//! # ferric-ci — CAS-CI (Complete Active Space Configuration Interaction)
//!
//! ## Phase A (this crate): plain CAS-CI
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
//! 3. Applies the Slater–Condon rules to build `sigma = H c` (and, for the
//!    tiny spike systems, a dense H) — see [`hamiltonian`].
//! 4. Diagonalizes for the lowest root via a CI-native Davidson solver (with a
//!    dense-eigensolve fallback used as an independent cross-check in tests) —
//!    see [`davidson`].
//!
//! The total energy is `E_CASCI = E_core + lambda_min`, where `E_core` is the
//! nuclear repulsion plus the inactive (frozen-core) electronic energy, and
//! `lambda_min` is the lowest eigenvalue of the active-space CI Hamiltonian.
//! When the active space is the full MO space (no inactive orbitals), CAS-CI is
//! exactly FCI in that basis.
//!
//! ## Explicitly NOT implemented in Phase A
//!
//! * **RAS1 / RAS3 hole/particle restrictions** (Restricted Active Space) —
//!   Phase B. Phase A enumerates the *complete* active space with no occupation
//!   restrictions on any orbital subspace.
//! * **Open-shell / non-singlet references, S² handling, multiple roots** —
//!   Phase C. Phase A assumes a closed-shell singlet (`n_alpha == n_beta`) and
//!   extracts only the single lowest root.
//! * **CLI / Python wiring** — Phase C. Phase A is a library-only spike with
//!   `#[test]` validation against PySCF FCI/CASCI.
//!
//! ## Known scaling limitation
//!
//! The sigma build ([`hamiltonian::sigma`]) and the dense-H build
//! ([`hamiltonian::dense_hamiltonian`]) are both **O(N_det²)** naive pair loops
//! over the determinant list. This is correct and simple but does not scale:
//! the determinant count grows combinatorially with the active space. A
//! production FCI needs the string-based *direct* sigma algorithm
//! (Knowles–Handy / Olsen: the `sigma1`/`sigma2`/`sigma3` α/β-string
//! contractions). Replacing the pair loop with that algorithm is the first item
//! for Phase B/C. For the STO-3G spike systems here (≤441 determinants) the
//! O(N_det²) loop is fast and unambiguously correct, which is what Phase A is
//! meant to prove.

pub mod davidson;
pub mod hamiltonian;
pub mod integrals;
pub mod strings;

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::ScfResult;

/// Configuration for a CAS-CI calculation.
#[derive(Debug, Clone, Copy)]
pub struct CasCiConfig {
    /// Number of active spatial orbitals.
    pub n_active: usize,
    /// Active electrons split as `(n_alpha, n_beta)`. For a closed-shell
    /// singlet, `n_alpha == n_beta == n_elec_active / 2`.
    pub n_elec_active: (usize, usize),
    /// Index of the first active MO. MOs `0..active_start` are the
    /// doubly-occupied inactive (frozen-core) orbitals.
    pub active_start: usize,
    /// Davidson residual convergence threshold.
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
        }
    }
}

/// Result of a CAS-CI calculation.
pub struct CasCiResult {
    /// Total CAS-CI energy: `e_core + e_active`.
    pub e_total: f64,
    /// Lowest eigenvalue of the active-space CI Hamiltonian (includes e_core,
    /// since e_core is folded onto the Hamiltonian diagonal). Equal to
    /// `e_total`; retained for clarity/debugging.
    pub e_active: f64,
    /// Additive core (nuclear + inactive electronic) energy.
    pub e_core: f64,
    /// Number of determinants in the CI expansion.
    pub n_determinants: usize,
    /// Normalized CI coefficient vector, indexed `d = ia * n_beta + ib`.
    pub ci_vector: Vec<f64>,
    /// Whether the Davidson solve converged to `conv_thresh`.
    pub converged: bool,
    /// Davidson iterations taken.
    pub iterations: usize,
}

/// Run a plain CAS-CI calculation from a converged restricted RHF reference.
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
            "CAS-CI Phase A requires a restricted (closed-shell) RHF reference".to_string(),
        ));
    }
    if n_active == 0 {
        return Err(FerricError::General(
            "CAS-CI: n_active must be >= 1".to_string(),
        ));
    }
    if n_active > 63 {
        return Err(FerricError::General(format!(
            "CAS-CI: n_active = {n_active} exceeds the 63-orbital u64-string limit \
             (spike-scale only)"
        )));
    }
    if n_alpha > n_active || n_beta > n_active {
        return Err(FerricError::General(format!(
            "CAS-CI: active electrons per spin (α={n_alpha}, β={n_beta}) cannot exceed \
             n_active ({n_active})"
        )));
    }
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
    let ints = integrals::build_active_space_integrals(mol, prep, rhf, active_start, n_active)?;

    // ---- Determinant space --------------------------------------------
    let alpha_strings = strings::enumerate_strings(n_active, n_alpha);
    let beta_strings = strings::enumerate_strings(n_active, n_beta);
    if alpha_strings.is_empty() || beta_strings.is_empty() {
        return Err(FerricError::General(
            "CAS-CI: empty determinant space (check n_active / electron counts)".to_string(),
        ));
    }
    let space = hamiltonian::DeterminantSpace {
        alpha_strings,
        beta_strings,
    };
    let ndet = space.n_det();

    // ---- Memory pre-flight --------------------------------------------
    // Davidson holds TWO bases (v_basis + hv_basis), each up to max_subspace
    // vectors of ndet doubles, plus the diagonal and a few ndet work vectors.
    // ndet grows combinatorially with the active space, so this crosses a
    // 23 GB box between CAS(12,14) (7.2 GB) and CAS(14,16) (105 GB) -- and
    // before this gate nothing stopped it. max_subspace is read from config,
    // not assumed: it is a user-settable memory knob whether or not it was
    // designed as one.
    let budget = ferric_core::memory::resolve_budget_bytes(config.memory_budget_bytes);
    let subspace_bytes = ndet
        .saturating_mul(config.max_subspace.max(1))
        .saturating_mul(2)
        .saturating_mul(8);
    // diag + guess + x + hx + t: five ndet-sized vectors alongside the bases.
    let work_bytes = ndet.saturating_mul(5).saturating_mul(8);
    ferric_core::memory::check_alloc(
        &format!(
            "CAS-CI Davidson subspace (n_active={n_active}, n_elec=({n_alpha},{n_beta}),              N_det={ndet}, max_subspace={})",
            config.max_subspace
        ),
        subspace_bytes.saturating_add(work_bytes),
        budget,
    )?;

    // ---- Diagonal + Davidson solve ------------------------------------
    let diag = hamiltonian::hamiltonian_diagonal(&ints, &space);
    let matvec = |c: &[f64]| hamiltonian::sigma(&ints, &space, c);
    let dav = davidson::davidson_lowest(
        ndet,
        matvec,
        &diag,
        config.conv_thresh,
        config.max_subspace,
        config.max_iter,
    )?;

    Ok(CasCiResult {
        e_total: dav.eigenvalue,
        e_active: dav.eigenvalue,
        e_core: ints.e_core,
        n_determinants: ndet,
        ci_vector: dav.eigenvector,
        converged: dav.converged,
        iterations: dav.iterations,
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
        assert!(msg.contains("CAS-CI"), "message must name the method: {msg}");
        assert!(msg.contains("N_det="), "message must name N_det: {msg}");
        assert!(msg.contains("budget is"), "message must name the budget: {msg}");
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
        let ints =
            integrals::build_active_space_integrals(&mol, &prep, &rhf, 1, 6).unwrap();
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
