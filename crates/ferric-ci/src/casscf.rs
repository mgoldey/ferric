//! Two-step CASSCF: CAS-CI with RDMs, then a quasi-Newton orbital step.
//!
//! Single root, spin-summed RDMs, real orbitals, chemist notation. The start
//! is a closed-shell RHF reference; the active space is the `n_active`
//! orbitals above the `n_core` doubly occupied ones. Equations and the
//! validation record: `crates/ferric-ci/theory.md` ("Two-step CASSCF").
//!
//! One macro iteration at orbitals `C = [C_c | C_a | C_v]`:
//!
//! ```text
//!   D^c   = 2 C_c C_cᵀ          F^I = h + J[D^c] − ½ K[D^c]
//!   E_core = E_nuc + ½ Tr[D^c (h + F^I)]
//!   CAS-CI on  h_eff = C_aᵀ F^I C_a,  (tu|vw)        → γ, Γ, E
//!   D^a   = C_a γ C_aᵀ          F^A = J[D^a] − ½ K[D^a]
//!   F_pi  = 2 (F^I_pi + F^A_pi)                       (core column i)
//!   F_pt  = Σ_u F^I_pu γ_ut + Σ_uvw (pu|vw) Γ_tuvw    (active column t)
//!   g_pq  = 2 (F_pq − F_qp),   p > q nonredundant
//! ```
//!
//! with `F^I`, `F^A` in the MO basis on the right-hand sides. The rotation is
//! `C' = C exp(K)`, `K_pq = κ_pq = −K_qp` for `p > q`, and `g_pq = ∂E/∂κ_pq`.
//! The nonredundant pairs are core-active `(t, i)`, core-virtual `(a, i)` and
//! active-virtual `(a, t)`.
//!
//! The step is inverse BFGS in the current orbital frame (no transport of
//! earlier steps), started from the reciprocal of an approximate diagonal
//! Hessian ([`crate::casscf::CasScfProblem::approx_hessian_diag`], floored), with the
//! largest step element capped and a reset to the diagonal whenever the
//! energy rises. This is the optimizer of `scripts/prototypes/casscf_proto.py`.
//!
//! Optionally ([`CasScfConfig::canonicalize`]) the returned orbitals are
//! canonical: the core and virtual blocks of `F^I + F^A` are diagonal and the
//! active orbitals are natural orbitals (see [`crate::casscf::canonicalize`]).
//!
//! The active space may have `n_alpha != n_beta`: the energy and gradient
//! depend on the spin-summed RDMs only, and the CI root is the lowest of the
//! `(n_alpha, n_beta)` sector. The orbitals stay one restricted set.
//!
//! The AO two-electron integrals are held densely (`nbas^4`), and `J`, `K` and
//! the `(pu|vw)` transform are built from that block every iteration. The
//! method is limited to bases small enough for that block.

use crate::integrals::{dense_ao_eri, transform_eri_mixed, ActiveSpaceIntegrals};
use crate::{run_cas_ci_on_integrals, CasCiConfig, CasCiRdms, CasCiResult};
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::memory::pool::Reservation;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::hcore_ecp;
use ferric_scf::ScfResult;
use ndarray::{s, Array2, ArrayView2};
use ndarray_linalg::{Eigh, UPLO};

/// Settings of a two-step CASSCF run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CasScfConfig {
    /// Number of active spatial orbitals.
    pub n_active: usize,
    /// Active electrons `(n_alpha, n_beta)`. The CI root is the lowest state
    /// of that `M_s` sector.
    pub n_elec_active: (usize, usize),
    /// Doubly occupied inactive orbitals below the active space.
    pub n_core: usize,
    /// Convergence: `‖g‖₂` below this ...
    pub grad_tol: f64,
    /// ... and `|E_k − E_{k−1}|` below this (Ha), both at the same iteration.
    pub energy_tol: f64,
    /// Macro-iteration cap. A run that reaches it returns with
    /// [`CasScfResult::converged`] false.
    pub max_iter: usize,
    /// Cap on `max_k |s_k|` of each orbital step (radians).
    pub max_step: f64,
    /// Floor on `|H_diag|` in the initial inverse Hessian.
    pub hdiag_floor: f64,
    /// Residual threshold of every CAS-CI solve (Davidson, or the dense
    /// eigensolve on spaces of at most [`crate::DENSE_SOLVE_MAX_DETS`]
    /// determinants).
    pub ci_conv_thresh: f64,
    /// Davidson subspace size.
    pub ci_max_subspace: usize,
    /// Davidson iteration cap. A CI solve that misses `ci_conv_thresh` is an
    /// error at any macro iteration.
    pub ci_max_iter: usize,
    /// Memory ceiling in bytes (see [`CasCiConfig::memory_budget_bytes`]).
    /// The dense AO block, the transform chain and the CI are charged to it.
    pub memory_budget_bytes: Option<usize>,
    /// Return canonical orbitals ([`canonicalize()`]) and
    /// [`CasScfResult::mo_energy`].
    pub canonicalize: bool,
}

impl Default for CasScfConfig {
    fn default() -> Self {
        CasScfConfig {
            n_active: 0,
            n_elec_active: (0, 0),
            n_core: 0,
            grad_tol: 1e-7,
            energy_tol: 1e-12,
            max_iter: 300,
            max_step: 0.2,
            hdiag_floor: 0.05,
            ci_conv_thresh: 1e-10,
            ci_max_subspace: 50,
            ci_max_iter: 200,
            memory_budget_bytes: None,
            canonicalize: false,
        }
    }
}

/// Class of a nonredundant orbital rotation `(p, q)`, `p > q`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationClass {
    /// `(t, i)`: active `t`, core `i`.
    CoreActive,
    /// `(a, i)`: virtual `a`, core `i`.
    CoreVirtual,
    /// `(a, t)`: virtual `a`, active `t`.
    ActiveVirtual,
}

/// Everything evaluated at one set of orbitals.
#[derive(Debug, Clone)]
pub struct CasScfPoint {
    /// CAS-CI energy `E_core + E_active` (Ha).
    pub e_total: f64,
    /// Orbital gradient over [`CasScfProblem::pairs`], `g_k = ∂E/∂κ_k`.
    pub gradient: Vec<f64>,
    /// The full matrix `2 (F − Fᵀ)` (`nmo x nmo`), including the redundant
    /// core-core and active-active blocks.
    pub gradient_matrix: Array2<f64>,
    /// Inactive Fock matrix `F^I` in the MO basis.
    pub f_inactive: Array2<f64>,
    /// Active Fock matrix `F^A` in the MO basis.
    pub f_active: Array2<f64>,
    /// Generalized Fock matrix `F` (columns of virtual orbitals are zero).
    pub f_general: Array2<f64>,
    /// The CAS-CI solve at these orbitals (CI vector, RDMs, residual).
    pub ci: CasCiResult,
}

impl CasScfPoint {
    /// The spin-summed RDMs of the CI root.
    pub fn rdms(&self) -> &CasCiRdms {
        // `evaluate` always requests RDMs.
        self.ci
            .rdms
            .as_ref()
            .expect("CasScfPoint is built with compute_rdms = true")
    }
}

/// The CAS energy functional of one molecule and active-space partition:
/// AO integrals, partition and the nonredundant rotation pairs.
pub struct CasScfProblem {
    h_ao: Array2<f64>,
    eri_ao: Vec<f64>,
    nbas: usize,
    nmo: usize,
    e_nuc: f64,
    n_core: usize,
    n_active: usize,
    ci_config: CasCiConfig,
    /// The settings the problem was built from (`optimize` checks that its
    /// own config agrees on the partition and the CI settings).
    config: CasScfConfig,
    pairs: Vec<(usize, usize)>,
    classes: Vec<RotationClass>,
    _mem: Reservation,
}

impl CasScfProblem {
    /// Set up the functional for `nmo` orbitals (the column count of the
    /// orbitals later passed in). Validates the partition and the electron
    /// count, charges the dense AO block and the transform chain to the
    /// memory budget, then builds `h` and the AO ERIs.
    pub fn new(
        mol: &Molecule,
        prep: &PreparedBasis,
        nmo: usize,
        config: &CasScfConfig,
    ) -> Result<Self, FerricError> {
        let (na, nb) = config.n_elec_active;
        let (nc, nact) = (config.n_core, config.n_active);
        if nact == 0 {
            return Err(FerricError::General(
                "CASSCF: n_active must be >= 1".to_string(),
            ));
        }
        if nact > crate::strings::MAX_ACTIVE_ORBITALS {
            return Err(FerricError::General(format!(
                "CASSCF: n_active = {nact} exceeds the {}-orbital CI string limit",
                crate::strings::MAX_ACTIVE_ORBITALS
            )));
        }
        check_step_controls(config)?;
        if na > nact || nb > nact {
            return Err(FerricError::General(format!(
                "CASSCF: active electrons per spin (α={na}, β={nb}) exceed n_active ({nact})"
            )));
        }
        if nc + nact > nmo {
            return Err(FerricError::General(format!(
                "CASSCF: n_core({nc}) + n_active({nact}) exceeds the number of MOs ({nmo})"
            )));
        }
        let nelec = mol.nelec() as usize;
        if 2 * nc + na + nb != nelec {
            return Err(FerricError::General(format!(
                "CASSCF: 2*n_core({nc}) + active(α={na}+β={nb}) = {} != molecule nelec {nelec}",
                2 * nc + na + nb
            )));
        }
        let nbas = prep.nbasis();
        let n4 = nbas.saturating_pow(4);
        let npair = nact * nc + (nmo - nc - nact) * (nc + nact);
        let mut plan = MemoryPlan::from_global_pool(
            config.memory_budget_bytes,
            format!("CASSCF dense AO ERIs + (pu|vw) transform (nbas={nbas}, n_active={nact})"),
        );
        plan.reserve("dense AO ERIs", n4, Lifetime::Resident);
        plan.reserve(
            "transform chain (first two quarter outputs)",
            nact.saturating_mul(nbas.saturating_pow(3))
                .saturating_add(nact.saturating_pow(2).saturating_mul(nbas * nbas)),
            Lifetime::Transient,
        );
        plan.reserve(
            "(pu|vw), MO Fock matrices, inverse BFGS Hessian",
            nmo.saturating_mul(nact.saturating_pow(3))
                .saturating_add(8 * nbas * nbas)
                .saturating_add(npair.saturating_mul(npair)),
            Lifetime::Resident,
        );
        let mem = plan.commit()?;
        let h_ao = hcore_ecp(prep, mol, prep.basis_set());
        let eri_ao = dense_ao_eri(prep)?;

        let mut pairs = Vec::with_capacity(npair);
        let mut classes = Vec::with_capacity(npair);
        for t in nc..nc + nact {
            for i in 0..nc {
                pairs.push((t, i));
                classes.push(RotationClass::CoreActive);
            }
        }
        for a in nc + nact..nmo {
            for i in 0..nc {
                pairs.push((a, i));
                classes.push(RotationClass::CoreVirtual);
            }
        }
        for a in nc + nact..nmo {
            for t in nc..nc + nact {
                pairs.push((a, t));
                classes.push(RotationClass::ActiveVirtual);
            }
        }
        let ci_config = CasCiConfig {
            n_active: nact,
            n_elec_active: config.n_elec_active,
            active_start: nc,
            conv_thresh: config.ci_conv_thresh,
            max_subspace: config.ci_max_subspace,
            max_iter: config.ci_max_iter,
            // The AO reservation stays held for the problem's lifetime; the CI
            // gets what is left of the same ceiling, so the two stages cannot
            // each claim the whole budget (with no pool installed `commit`
            // debits nothing, and with a pool an explicit budget still binds).
            memory_budget_bytes: Some(plan.budget_bytes().saturating_sub(plan.peak_bytes())),
            compute_rdms: true,
        };
        Ok(CasScfProblem {
            h_ao,
            eri_ao,
            nbas,
            nmo,
            e_nuc: mol.nuclear_repulsion(),
            n_core: nc,
            n_active: nact,
            ci_config,
            config: *config,
            pairs,
            classes,
            _mem: mem,
        })
    }

    /// Nonredundant rotation pairs `(p, q)`, `p > q`: core-active, then
    /// core-virtual, then active-virtual.
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Class of each entry of [`Self::pairs`].
    pub fn classes(&self) -> &[RotationClass] {
        &self.classes
    }

    fn check_orbitals(&self, c: &Array2<f64>) -> Result<(), FerricError> {
        if c.dim() != (self.nbas, self.nmo) {
            return Err(FerricError::General(format!(
                "CASSCF: orbitals have shape {:?}, expected ({}, {})",
                c.dim(),
                self.nbas,
                self.nmo
            )));
        }
        Ok(())
    }

    /// `J[D]` and `K[D]` from the dense AO block, `J_mn = Σ (mn|ls) D_ls`,
    /// `K_mn = Σ (ml|ns) D_ls`.
    fn jk(&self, d: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        dense_jk(&self.eri_ao, self.nbas, d)
    }

    /// `F^I` (AO basis) and `E_core` at orbitals `c`.
    fn inactive(&self, c: &Array2<f64>) -> (Array2<f64>, f64) {
        let nc = self.n_core;
        if nc == 0 {
            return (self.h_ao.clone(), self.e_nuc);
        }
        let cc = c.slice(s![.., 0..nc]);
        let dc = 2.0 * cc.dot(&cc.t());
        let (j, k) = self.jk(&dc);
        let fi = &self.h_ao + &j - &(0.5 * &k);
        let e_core = self.e_nuc + 0.5 * (&dc * &(&self.h_ao + &fi)).sum();
        (fi, e_core)
    }

    fn solve_ci(
        &self,
        ca: ArrayView2<'_, f64>,
        fi_ao: &Array2<f64>,
        eri_act: Vec<f64>,
        e_core: f64,
        compute_rdms: bool,
    ) -> Result<CasCiResult, FerricError> {
        let ints = ActiveSpaceIntegrals {
            n_active: self.n_active,
            h: ca.t().dot(fi_ao).dot(&ca),
            eri: eri_act,
            e_core,
        };
        let cfg = CasCiConfig {
            compute_rdms,
            ..self.ci_config
        };
        let ci = run_cas_ci_on_integrals(ints, &cfg)?;
        // The CI flag is the residual criterion ‖Hc − Ec‖ < ci_conv_thresh
        // (Davidson relaxes it to 10x when its correction vector vanishes;
        // the dense fallback sets it from the residual alone). The
        // gradient error is first order in the residual, so the returned
        // root must itself meet ci_conv_thresh.
        if !ci_root_accepted(ci.converged, ci.residual_norm, self.ci_config.conv_thresh) {
            return Err(FerricError::General(format!(
                "CASSCF: CAS-CI residual {:.2e} did not reach ci_conv_thresh {:e} \
                 in {} iterations (raise ci_max_iter or ci_max_subspace)",
                ci.residual_norm, self.ci_config.conv_thresh, ci.iterations
            )));
        }
        Ok(ci)
    }

    /// Active Fock matrix `F^A = J[D^a] − ½ K[D^a]`, `D^a = C_a γ C_aᵀ`, in
    /// the MO basis of `c`, for any symmetric active 1-RDM `gamma`. Linear in
    /// `gamma`.
    pub fn active_fock(
        &self,
        c: &Array2<f64>,
        gamma: &Array2<f64>,
    ) -> Result<Array2<f64>, FerricError> {
        self.check_orbitals(c)?;
        let (nc, na) = (self.n_core, self.n_active);
        if gamma.dim() != (na, na) {
            return Err(FerricError::General(format!(
                "CASSCF: active 1-RDM has shape {:?}, expected ({na}, {na})",
                gamma.dim()
            )));
        }
        let ca = c.slice(s![.., nc..nc + na]);
        let da = ca.dot(gamma).dot(&ca.t());
        let (j, k) = self.jk(&da);
        let fa_ao = &j - &(0.5 * &k);
        Ok(c.t().dot(&fa_ao).dot(c))
    }

    /// CI-relaxed CAS-CI energy at orbitals `c` (no RDMs, no gradient).
    pub fn energy(&self, c: &Array2<f64>) -> Result<f64, FerricError> {
        self.check_orbitals(c)?;
        let (nc, na) = (self.n_core, self.n_active);
        let (fi_ao, e_core) = self.inactive(c);
        let ca = c.slice(s![.., nc..nc + na]);
        let eri = transform_eri_mixed(&self.eri_ao, self.nbas, [ca, ca, ca, ca])?;
        Ok(self.solve_ci(ca, &fi_ao, eri, e_core, false)?.e_total)
    }

    /// Energy, orbital gradient and intermediates at orbitals `c`.
    pub fn evaluate(&self, c: &Array2<f64>) -> Result<CasScfPoint, FerricError> {
        self.check_orbitals(c)?;
        let (nc, na, nmo) = (self.n_core, self.n_active, self.nmo);
        let na3 = na * na * na;
        let (fi_ao, e_core) = self.inactive(c);
        let ca = c.slice(s![.., nc..nc + na]);
        // (pu|vw), p over all MOs: the active rows are the CI integrals.
        let paaa = transform_eri_mixed(&self.eri_ao, self.nbas, [c.view(), ca, ca, ca])?;
        let eri_act = paaa[nc * na3..(nc + na) * na3].to_vec();
        let ci = self.solve_ci(ca, &fi_ao, eri_act, e_core, true)?;
        let rdms = ci
            .rdms
            .as_ref()
            .ok_or_else(|| FerricError::General("CASSCF: CAS-CI returned no RDMs".to_string()))?;
        let gamma = &rdms.dm1;
        let fi = c.t().dot(&fi_ao).dot(c);
        let fa = self.active_fock(c, gamma)?;

        let mut f = Array2::<f64>::zeros((nmo, nmo));
        for p in 0..nmo {
            for i in 0..nc {
                f[(p, i)] = 2.0 * (fi[(p, i)] + fa[(p, i)]);
            }
        }
        // Active columns: F^I γ plus the (pu|vw) Γ_tuvw contraction, the
        // latter one GEMM of (nmo x na³) by (na³ x na).
        let one = fi.slice(s![.., nc..nc + na]).dot(gamma);
        let p_mat = ArrayView2::from_shape((nmo, na3), &paaa)
            .map_err(|e| FerricError::General(format!("CASSCF (pu|vw) shape: {e}")))?;
        let g2 = rdms
            .dm2
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((na, na3))
            .map_err(|e| FerricError::General(format!("CASSCF dm2 shape: {e}")))?;
        let two = p_mat.dot(&g2.t());
        f.slice_mut(s![.., nc..nc + na]).assign(&(&one + &two));

        let gmat = 2.0 * (&f - &f.t());
        let gradient = self.pairs.iter().map(|&(p, q)| gmat[(p, q)]).collect();
        Ok(CasScfPoint {
            e_total: ci.e_total,
            gradient,
            gradient_matrix: gmat,
            f_inactive: fi,
            f_active: fa,
            f_general: f,
            ci,
        })
    }

    /// `C exp(K)` with `K_pq = κ_k = −K_qp` for `(p, q) = pairs[k]`.
    pub fn rotate(&self, c: &Array2<f64>, kappa: &[f64]) -> Result<Array2<f64>, FerricError> {
        self.check_orbitals(c)?;
        if kappa.len() != self.pairs.len() {
            return Err(FerricError::General(format!(
                "CASSCF: rotation vector has {} entries, expected {}",
                kappa.len(),
                self.pairs.len()
            )));
        }
        let mut k = Array2::<f64>::zeros((self.nmo, self.nmo));
        for (&(p, q), &x) in self.pairs.iter().zip(kappa) {
            k[(p, q)] = x;
            k[(q, p)] = -x;
        }
        Ok(c.dot(&expm_antisymmetric(&k)))
    }

    /// Approximate diagonal orbital Hessian over [`Self::pairs`]
    /// (Chaban, Schmidt and Gordon, Theor. Chem. Acc. 97, 88 (1997)), with
    /// `F^T = F^I + F^A`:
    ///
    /// ```text
    ///   (a, i): 4 (F^T_aa − F^T_ii)
    ///   (a, t): 2 γ_tt F^T_aa − 2 F_tt
    ///   (t, i): 4 (F^T_tt − F^T_ii) + 2 γ_tt F^T_ii − 2 F_tt
    /// ```
    ///
    /// A preconditioner only: it is not an anchored Hessian.
    pub fn approx_hessian_diag(&self, pt: &CasScfPoint) -> Vec<f64> {
        let ft = &pt.f_inactive + &pt.f_active;
        let fg = &pt.f_general;
        let gamma = &pt.rdms().dm1;
        let nc = self.n_core;
        self.pairs
            .iter()
            .zip(&self.classes)
            .map(|(&(p, q), cl)| match cl {
                RotationClass::CoreVirtual => 4.0 * (ft[(p, p)] - ft[(q, q)]),
                RotationClass::ActiveVirtual => {
                    let gtt = gamma[(q - nc, q - nc)];
                    2.0 * gtt * ft[(p, p)] - 2.0 * fg[(q, q)]
                }
                RotationClass::CoreActive => {
                    let gtt = gamma[(p - nc, p - nc)];
                    4.0 * (ft[(p, p)] - ft[(q, q)]) + 2.0 * gtt * ft[(q, q)] - 2.0 * fg[(p, p)]
                }
            })
            .collect()
    }
}

/// `J[D]` and `K[D]` from a dense row-major AO block `eri` of `n^4`
/// elements, `J_mn = Σ (mn|ls) D_ls`, `K_mn = Σ (ml|ns) D_ls`.
pub(crate) fn dense_jk(eri: &[f64], n: usize, d: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let mut j = Array2::<f64>::zeros((n, n));
    let mut k = Array2::<f64>::zeros((n, n));
    for m in 0..n {
        for l in 0..n {
            let base = (m * n + l) * n * n;
            let mut jml = 0.0;
            for a in 0..n {
                for b in 0..n {
                    let v = eri[base + a * n + b];
                    jml += v * d[(a, b)];
                    // (m l|a b) D[l b] -> K[m, a]
                    k[(m, a)] += v * d[(l, b)];
                }
            }
            j[(m, l)] = jml;
        }
    }
    (j, k)
}

/// `exp(K)` for a real antisymmetric `K`, by scaling and squaring a Taylor
/// series: `K / 2^s` has ∞-norm at most 1/2, where the series reaches
/// machine precision within 20 terms. The result is orthogonal to rounding.
pub fn expm_antisymmetric(k: &Array2<f64>) -> Array2<f64> {
    let n = k.nrows();
    let norm = k
        .rows()
        .into_iter()
        .map(|r| r.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let mut squarings = 0u32;
    while norm / f64::powi(2.0, squarings as i32) > 0.5 {
        squarings += 1;
    }
    let a = k / f64::powi(2.0, squarings as i32);
    let mut sum = Array2::<f64>::eye(n);
    let mut term = Array2::<f64>::eye(n);
    for j in 1..=30 {
        term = term.dot(&a) / j as f64;
        sum += &term;
        if term.iter().all(|x| x.abs() < 1e-18) {
            break;
        }
    }
    for _ in 0..squarings {
        sum = sum.dot(&sum);
    }
    sum
}

/// Result of a two-step CASSCF run.
#[derive(Debug, Clone)]
#[must_use]
pub struct CasScfResult {
    /// CASSCF energy at [`Self::mo_coeff`] (Ha).
    pub e_total: f64,
    /// Core energy `E_nuc + ½ Tr[D^c (h + F^I)]` at those orbitals.
    pub e_core: f64,
    /// Optimized orbitals, `nbas x nmo`: core, active, virtual columns.
    /// Canonical ([`canonicalize()`]) when [`CasScfConfig::canonicalize`] is
    /// set, otherwise not canonicalized within any block.
    pub mo_coeff: Array2<f64>,
    /// With [`CasScfConfig::canonicalize`], the diagonal of `F^I + F^A` in
    /// [`Self::mo_coeff`] (Ha): orbital energies of the core and virtual
    /// orbitals, and the Fock diagonal of the natural active orbitals.
    /// `None` otherwise.
    pub mo_energy: Option<Vec<f64>>,
    /// CI vector of the root in [`Self::mo_coeff`], indexed
    /// `ia * n_beta_str + ib`.
    pub ci_vector: Vec<f64>,
    /// Spin-summed RDMs of the root over the active orbitals.
    pub rdms: CasCiRdms,
    /// Determinants in the CI expansion.
    pub n_determinants: usize,
    /// Macro iterations (gradient evaluations) taken.
    pub iterations: usize,
    /// `‖g‖₂` at the returned orbitals.
    pub grad_norm: f64,
    /// Both convergence criteria met before [`CasScfConfig::max_iter`].
    pub converged: bool,
    /// `‖Hc − Ec‖` of the final CI vector.
    pub ci_residual_norm: f64,
}

/// Canonical orbitals at the point `pt` evaluated at orbitals `c`.
///
/// The core and virtual blocks of `F^T = F^I + F^A` (MO basis) are
/// diagonalized, eigenvalues ascending, and the active orbitals become the
/// eigenvectors of the 1-RDM `γ`, occupations descending. Each new column
/// is signed by [`sign_reference`]: its first AO coefficient within a
/// relative 1e-6 of the largest magnitude is positive (the first index
/// breaks the ties between symmetry-equivalent atoms). All three rotations
/// are within the redundant blocks, so the energy and the norm of the
/// nonredundant gradient are unchanged.
///
/// The CAS-CI is solved again in the new orbitals (one
/// [`CasScfProblem::evaluate`]): the returned point carries the CI vector,
/// RDMs and Fock matrices in the new orbitals, with the CI vector signed the
/// same way. `F^I` depends on the core block only through `D^c`, which the
/// core rotation leaves unchanged, and `F^A` through `D^a = C_a γ C_aᵀ`,
/// which the active rotation and the transformed CI vector leave unchanged;
/// so `F^T` in the new orbitals is the rotated `F^T` up to the CI error of
/// the second solve.
pub fn canonicalize(
    problem: &CasScfProblem,
    c: &Array2<f64>,
    pt: &CasScfPoint,
) -> Result<(Array2<f64>, CasScfPoint), FerricError> {
    problem.check_orbitals(c)?;
    let (nc, na, nmo) = (problem.n_core, problem.n_active, problem.nmo);
    let ft = &pt.f_inactive + &pt.f_active;
    let mut u = Array2::<f64>::zeros((nmo, nmo));
    let eig_block = |m: ArrayView2<'_, f64>| -> Result<(Vec<f64>, Array2<f64>), FerricError> {
        let sym = 0.5 * (&m + &m.t());
        let (w, v) = sym
            .eigh(UPLO::Lower)
            .map_err(|e| FerricError::General(format!("CASSCF canonicalize: eigh: {e}")))?;
        Ok((w.to_vec(), v))
    };
    for (lo, hi) in [(0, nc), (nc + na, nmo)] {
        if hi > lo {
            let (_, v) = eig_block(ft.slice(s![lo..hi, lo..hi]))?;
            u.slice_mut(s![lo..hi, lo..hi]).assign(&v);
        }
    }
    let (occ, v) = eig_block(pt.rdms().dm1.view())?;
    let mut order: Vec<usize> = (0..na).collect();
    order.sort_by(|&a, &b| occ[b].total_cmp(&occ[a]));
    for (k, &j) in order.iter().enumerate() {
        u.slice_mut(s![nc..nc + na, nc + k]).assign(&v.column(j));
    }
    let mut c_new = c.dot(&u);
    for mut col in c_new.columns_mut() {
        if sign_reference(col.iter().copied()) < 0.0 {
            col.mapv_inplace(|x| -x);
        }
    }
    let mut pt_new = problem.evaluate(&c_new)?;
    if sign_reference(pt_new.ci.ci_vector.iter().copied()) < 0.0 {
        pt_new.ci.ci_vector.iter_mut().for_each(|x| *x = -*x);
    }
    Ok((c_new, pt_new))
}

/// The first element whose magnitude is within a relative 1e-6 of the
/// largest; its sign fixes the sign of a vector. Elements equal in magnitude
/// by symmetry (an orbital on two equivalent atoms) resolve to the first
/// index, so the choice does not depend on rounding.
pub fn sign_reference(v: impl Iterator<Item = f64> + Clone) -> f64 {
    let big = v.clone().fold(0.0f64, |m, x| m.max(x.abs()));
    v.into_iter()
        .find(|x| x.abs() >= big * (1.0 - 1e-6))
        .unwrap_or(0.0)
}

/// Two-step CASSCF from the canonical orbitals of a restricted RHF result.
///
/// The Hamiltonian is the molecule's: `h = T + V_nuc + V_ECP` (the ECP of
/// `prep`'s basis, as the RHF builds it; apply it to `mol` with
/// `Molecule::apply_ecp` first) and the exact AO ERIs. An external potential,
/// solvation or embedding term of the reference is not part of it and is not
/// recorded on [`ScfResult`], so the reference must be a vacuum RHF.
pub fn run_casscf(
    mol: &Molecule,
    prep: &PreparedBasis,
    rhf: &ScfResult,
    config: &CasScfConfig,
) -> Result<CasScfResult, FerricError> {
    if !matches!(rhf.spin, ferric_scf::Spin::Restricted) {
        return Err(FerricError::General(
            "CASSCF requires a restricted (closed-shell) RHF reference".to_string(),
        ));
    }
    let c0 = rhf.mos_r().clone();
    let problem = CasScfProblem::new(mol, prep, c0.ncols(), config)?;
    optimize(&problem, c0, config)
}

/// The two-step macro-iteration loop from starting orbitals `c0`.
///
/// `config` supplies the loop controls (`grad_tol`, `energy_tol`,
/// `max_iter`, `max_step`, `hdiag_floor`, `canonicalize`). Its active
/// partition (`n_active`, `n_elec_active`, `n_core`), CI settings
/// (`ci_conv_thresh`, `ci_max_subspace`, `ci_max_iter`) and
/// `memory_budget_bytes` are fixed when `problem` is built; a config that
/// differs in any of them is an error before the first evaluation.
pub fn optimize(
    problem: &CasScfProblem,
    c0: Array2<f64>,
    config: &CasScfConfig,
) -> Result<CasScfResult, FerricError> {
    check_step_controls(config)?;
    check_problem_settings(&problem.config, config)?;
    let npair = problem.pairs().len();
    let finish = |pt: CasScfPoint, c: Array2<f64>, it: usize, converged: bool| {
        let (c, pt, mo_energy) = if config.canonicalize {
            let (c, pt) = canonicalize(problem, &c, &pt)?;
            let ft = &pt.f_inactive + &pt.f_active;
            let eps = ft.diag().to_vec();
            (c, pt, Some(eps))
        } else {
            (c, pt, None)
        };
        let grad_norm = norm(&pt.gradient);
        let ci = pt.ci;
        let rdms = ci
            .rdms
            .ok_or_else(|| FerricError::General("CASSCF: CAS-CI returned no RDMs".to_string()))?;
        Ok(CasScfResult {
            e_total: ci.e_total,
            e_core: ci.e_core,
            mo_coeff: c,
            mo_energy,
            ci_vector: ci.ci_vector,
            rdms,
            n_determinants: ci.n_determinants,
            iterations: it,
            grad_norm,
            converged,
            ci_residual_norm: ci.residual_norm,
        })
    };

    let mut c = c0;
    let mut e_old: Option<f64> = None;
    let mut g_old: Vec<f64> = Vec::new();
    let mut s_old: Vec<f64> = Vec::new();
    let mut hinv: Option<Array2<f64>> = None;
    let mut it = 0usize;
    loop {
        it += 1;
        let pt = problem.evaluate(&c)?;
        let e = pt.e_total;
        let g = pt.gradient.clone();
        let gn = norm(&g);
        let done = npair == 0
            || e_old.is_some_and(|eo| (e - eo).abs() < config.energy_tol && gn < config.grad_tol);
        if done {
            return finish(pt, c, it, true);
        }
        if it >= config.max_iter {
            return finish(pt, c, it, false);
        }
        let hd: Vec<f64> = problem
            .approx_hessian_diag(&pt)
            .into_iter()
            .map(|h| h.abs().max(config.hdiag_floor))
            .collect();
        let rose = e_old.is_some_and(|eo| e > eo + 1e-10);
        let b = match hinv.take() {
            Some(mut b) if !rose => {
                bfgs_inverse_update(&mut b, &s_old, &g, &g_old);
                b
            }
            _ => Array2::from_diag(&ndarray::Array1::from_iter(hd.iter().map(|h| 1.0 / h))),
        };
        let mut step: Vec<f64> = (0..npair)
            .map(|r| -(0..npair).map(|k| b[(r, k)] * g[k]).sum::<f64>())
            .collect();
        let smax = step.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        if smax > config.max_step {
            let f = config.max_step / smax;
            step.iter_mut().for_each(|x| *x *= f);
        }
        c = problem.rotate(&c, &step)?;
        hinv = Some(b);
        e_old = Some(e);
        g_old = g;
        s_old = step;
    }
}

/// The settings `CasScfProblem::new` consumed must match `config`: the
/// problem evaluates with its own partition and CI settings, so a differing
/// config would report convergence for a calculation it did not describe.
/// Floats compare by bit pattern, so a NaN is a mismatch only against a
/// different value.
fn check_problem_settings(built: &CasScfConfig, config: &CasScfConfig) -> Result<(), FerricError> {
    let mut diffs: Vec<String> = Vec::new();
    let mut usize_field = |name: &str, a: usize, b: usize| {
        if a != b {
            diffs.push(format!("{name}: problem {a}, config {b}"));
        }
    };
    usize_field("n_active", built.n_active, config.n_active);
    usize_field("n_core", built.n_core, config.n_core);
    usize_field(
        "ci_max_subspace",
        built.ci_max_subspace,
        config.ci_max_subspace,
    );
    usize_field("ci_max_iter", built.ci_max_iter, config.ci_max_iter);
    if built.n_elec_active != config.n_elec_active {
        diffs.push(format!(
            "n_elec_active: problem {:?}, config {:?}",
            built.n_elec_active, config.n_elec_active
        ));
    }
    if built.ci_conv_thresh.to_bits() != config.ci_conv_thresh.to_bits() {
        diffs.push(format!(
            "ci_conv_thresh: problem {:e}, config {:e}",
            built.ci_conv_thresh, config.ci_conv_thresh
        ));
    }
    if built.memory_budget_bytes != config.memory_budget_bytes {
        diffs.push(format!(
            "memory_budget_bytes: problem {:?}, config {:?}",
            built.memory_budget_bytes, config.memory_budget_bytes
        ));
    }
    if diffs.is_empty() {
        Ok(())
    } else {
        Err(FerricError::General(format!(
            "CASSCF: optimize config differs from the one the problem was built with ({})",
            diffs.join("; ")
        )))
    }
}

/// A CI root is used only when Davidson converged and its recomputed
/// residual `‖Hc − Ec‖` is below the requested threshold (Davidson tests the
/// Ritz residual, which differs from the recomputed one at round-off).
fn ci_root_accepted(converged: bool, residual_norm: f64, conv_thresh: f64) -> bool {
    converged && residual_norm < conv_thresh
}

/// A step cap or Hessian floor that is not a positive finite number makes
/// every step zero (or NaN): an infinite floor gives a zero inverse Hessian,
/// a zero or NaN floor a NaN step where `H_diag` vanishes. The loop would
/// then repeat the same CI solve until `max_iter`.
fn check_step_controls(config: &CasScfConfig) -> Result<(), FerricError> {
    if !(config.max_step.is_finite() && config.max_step > 0.0) {
        return Err(FerricError::General(format!(
            "CASSCF: max_step must be a positive finite number of radians, got {}",
            config.max_step
        )));
    }
    if !(config.hdiag_floor.is_finite() && config.hdiag_floor > 0.0) {
        return Err(FerricError::General(format!(
            "CASSCF: hdiag_floor must be a positive finite number, got {}",
            config.hdiag_floor
        )));
    }
    Ok(())
}

/// Inverse BFGS update `B⁺ = V B Vᵀ + ρ s sᵀ`, `V = I − ρ s yᵀ`,
/// `ρ = 1/(sᵀy)`, with `y = g − g_old`; skipped when `sᵀy <= 1e-14`.
///
/// Applied in place as the expanded rank-two form
/// `B⁺ = B − ρ u sᵀ − ρ s wᵀ + (ρ² yᵀu + ρ) s sᵀ`, `u = B y`, `w = Bᵀ y`,
/// so the only `npair × npair` buffer is `b` itself (the one the memory
/// plan reserves).
fn bfgs_inverse_update(b: &mut Array2<f64>, s: &[f64], g: &[f64], g_old: &[f64]) {
    let n = s.len();
    let y: Vec<f64> = g.iter().zip(g_old).map(|(a, b)| a - b).collect();
    let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
    if sy <= 1e-14 {
        return;
    }
    let rho = 1.0 / sy;
    let u: Vec<f64> = (0..n)
        .map(|r| (0..n).map(|k| b[(r, k)] * y[k]).sum())
        .collect();
    let w: Vec<f64> = (0..n)
        .map(|r| (0..n).map(|k| b[(k, r)] * y[k]).sum())
        .collect();
    let yu: f64 = y.iter().zip(&u).map(|(a, b)| a * b).sum();
    let css = rho * rho * yu + rho;
    for r in 0..n {
        for k in 0..n {
            b[(r, k)] += -rho * (u[r] * s[k] + s[r] * w[k]) + css * s[r] * s[k];
        }
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn antisym(n: usize, seed: u64) -> Array2<f64> {
        let mut st = seed;
        let mut k = Array2::<f64>::zeros((n, n));
        for p in 0..n {
            for q in 0..p {
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let x = ((st >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
                k[(p, q)] = x;
                k[(q, p)] = -x;
            }
        }
        k
    }

    /// The CI stage is charged against what the AO reservation leaves of the
    /// same ceiling: with no pool installed the AO guard debits nothing, so
    /// handing the CI the full budget would let the two stages sum past it.
    #[test]
    fn ci_budget_is_the_ceiling_minus_the_ao_reservation() {
        let mol = Molecule::parse_xyz(
            "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n",
            0,
            1,
        )
        .unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let budget = 64 << 20;
        let cfg = CasScfConfig {
            n_active: 4,
            n_elec_active: (2, 2),
            n_core: 3,
            memory_budget_bytes: Some(budget),
            ..Default::default()
        };
        let problem = CasScfProblem::new(&mol, &prep, prep.nbasis(), &cfg).unwrap();
        let ci = problem.ci_config.memory_budget_bytes.unwrap();
        let ao_bytes = prep.nbasis().pow(4) * std::mem::size_of::<f64>();
        assert!(
            ci + ao_bytes <= budget,
            "CI budget {ci} + dense AO block {ao_bytes} exceeds the {budget}-byte ceiling"
        );
    }

    /// The in-place update equals the explicit `V B Vᵀ + ρ s sᵀ` on a
    /// non-symmetric `B` (so both the `B y` and `Bᵀ y` terms are exercised).
    /// Bar: entries are O(1) sums of O(n) products, rounding ~ n ε ≈ 1e-14.
    #[test]
    fn bfgs_inverse_update_in_place_matches_explicit_form() {
        let n = 7;
        let b0 = antisym(n, 3) + Array2::<f64>::eye(n) * 2.0 + antisym(n, 11).mapv(f64::abs);
        let v = |seed: u64| antisym(n + 1, seed).row(n).slice(s![..n]).to_vec();
        let (s, g, g_old) = (v(5), v(9), v(13));
        let y: Vec<f64> = g.iter().zip(&g_old).map(|(a, b)| a - b).collect();
        let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
        let (s, sy) = if sy < 0.0 {
            (s.iter().map(|x| -x).collect::<Vec<_>>(), -sy)
        } else {
            (s, sy)
        };
        assert!(sy > 1e-3, "test vectors must give a well-defined update");
        let rho = 1.0 / sy;
        let mut vm = Array2::<f64>::eye(n);
        for r in 0..n {
            for k in 0..n {
                vm[(r, k)] -= rho * s[r] * y[k];
            }
        }
        let mut want = vm.dot(&b0).dot(&vm.t());
        for r in 0..n {
            for k in 0..n {
                want[(r, k)] += rho * s[r] * s[k];
            }
        }
        let mut got = b0.clone();
        bfgs_inverse_update(&mut got, &s, &g, &g_old);
        let d = (&got - &want).iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(
            d < 1e-12,
            "in-place vs explicit inverse BFGS differ by {d:e}"
        );
    }

    /// A root flagged converged with a recomputed residual at or above the
    /// threshold is refused; one below the threshold is kept; an unconverged
    /// or NaN residual is refused.
    #[test]
    fn ci_root_must_meet_the_requested_residual() {
        let t = 1e-10;
        assert!(ci_root_accepted(true, 0.5 * t, t));
        assert!(!ci_root_accepted(true, 5.0 * t, t));
        assert!(!ci_root_accepted(true, t, t));
        assert!(!ci_root_accepted(false, 0.5 * t, t));
        assert!(!ci_root_accepted(true, f64::NAN, t));
    }

    /// `exp(K)` is orthogonal, `exp(K) exp(−K) = I`, and two half steps
    /// compose to the full step. A truncated series or a missing squaring
    /// fails the composition check at the 1e-6 level.
    #[test]
    fn expm_antisymmetric_is_orthogonal_and_composes() {
        for (n, scale) in [(5usize, 0.1), (9, 1.0), (13, 3.0)] {
            let k = antisym(n, 7 + n as u64) * scale;
            let u = expm_antisymmetric(&k);
            let ortho = (u.t().dot(&u) - Array2::<f64>::eye(n))
                .iter()
                .fold(0.0f64, |m, x| m.max(x.abs()));
            let half = expm_antisymmetric(&(&k * 0.5));
            let comp = (half.dot(&half) - &u)
                .iter()
                .fold(0.0f64, |m, x| m.max(x.abs()));
            let inv = (u.dot(&expm_antisymmetric(&(-&k))) - Array2::<f64>::eye(n))
                .iter()
                .fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(ortho < 1e-13, "n={n}: |UᵀU − I| = {ortho:e}");
            assert!(comp < 1e-13, "n={n}: |U(K/2)² − U(K)| = {comp:e}");
            assert!(inv < 1e-13, "n={n}: |U(K)U(−K) − I| = {inv:e}");
        }
        // 2x2 closed form: exp([[0, −θ], [θ, 0]]) is the rotation by θ.
        let th = 0.7f64;
        let mut k = Array2::<f64>::zeros((2, 2));
        k[(1, 0)] = th;
        k[(0, 1)] = -th;
        let u = expm_antisymmetric(&k);
        assert!((u[(0, 0)] - th.cos()).abs() < 1e-15);
        assert!((u[(1, 0)] - th.sin()).abs() < 1e-15);
    }
}
