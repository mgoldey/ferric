//! Coupled-Kohn–Sham (CKS) dispersion — the SAPT Casimir–Polder term MP2C needs.
//!
//! Pitoňák & Heßelmann, *J. Chem. Theory Comput.* **6**, 168 (2010).
//!
//! **EXPERIMENTAL — the coupled response is not yet usable.** The uncoupled
//! (UCHF) limit is validated against an independent construction and the
//! trace decays as `R⁻⁶`. The coupled (CKS) response of a monomer in the dimer
//! basis, however, has a negative static Dyson eigenvalue on every complex
//! measured (RHF references, ALDAX kernel), so the coupled integrand crosses a
//! pole and `E_disp^CKS` is not meaningful; see [`crate::mp2c_cks`] for the
//! numbers and [`CksDispersionResult::static_dyson_min_eigenvalue`] for the
//! per-run diagnostic. Do not use the coupled numbers for production results.
//!
//! # What this computes
//!
//! The second-order SAPT dispersion energy, as a Casimir–Polder integral over
//! the two **monomers'** frequency-dependent density–density response functions:
//!
//! ```text
//!   E_disp^(2) = −(1/2π) ∫dω ∫∫∫∫ dr dr' dr'' dr'''
//!                  χ_A(r,r';iω) (1/|r−r''|) χ_B(r'',r''';iω) (1/|r'−r'''|)   (1)
//! ```
//!
//! Expanded in an auxiliary basis {P} with the Coulomb metric
//! `V_PQ = (P|1/r₁₂|Q)`, the quadruple real-space integral collapses to a matrix
//! trace. Writing `χ^X_PQ` for the response in the *bare* aux basis,
//!
//! ```text
//!   E_disp = −(1/2π) ∫dω Tr[ χ_A V χ_B V ] .                               (2)
//! ```
//!
//! ferric never stores a bare `χ`. Its RI tensor is already
//! **symmetrically dressed**, `b_ov = V^{-1/2}(P|ia)` (see
//! `rimp2::compute_rpa_intermediates`), so every response built from it is
//! `χ̃ = V^{1/2} χ V^{1/2}`… — careful: what the dressed tensor produces is
//! `Π̃ = V^{-1/2} (…) V^{-1/2}` contracted through `b_ov b_ovᵀ`, which is exactly
//! the object for which the two explicit `V` factors in (2) are already absorbed.
//! In that representation (2) is a plain trace of two matrices:
//!
//! ```text
//!   E_disp = −(1/2π) ∫dω Tr[ Π̃_A(iω) Π̃_B(iω) ] .                          (3)
//! ```
//!
//! This is the same mathematical shape as `dispersion::casimir_polder_c6`'s
//! `(3/π) Σ_k w_k α^A:α^B`, but at the *full* density–density level rather than
//! the dipole (point-polarizability) truncation, so it carries no multipole
//! expansion and no `1/R⁶` asymptotic assumption.
//!
//! # Uncoupled vs coupled
//!
//! The **uncoupled** (UCHF/KS) response is the bare particle–hole bubble,
//! which is exactly what `sternheimer::dielectric_matrix_with_scale` already
//! builds (ferric stores it positive, `Π⁰ = −χ⁰ ≥ 0`):
//!
//! ```text
//!   Π⁰_PQ(iω) = 4 Σ_ia  e_ia/(ω² + e_ia²)  b_P,ia b_Q,ia ,   e_ia = ε_a−ε_i . (4)
//! ```
//!
//! The **coupled** (CKS/TDDFT) response resums the Dyson series in the
//! Hartree-plus-XC kernel. In the dressed basis the Coulomb (Hartree) part of
//! the kernel *is the identity*, so with `F̃` the exchange–correlation kernel
//! projected into the same dressed aux basis,
//!
//! ```text
//!   Π = Π⁰ − Π⁰ (I + F̃) Π + O(…)  ⟹  Π = [I + Π⁰(I + F̃)]^{-1} Π⁰ .        (5)
//! ```
//!
//! (The sign follows ferric's positive-`Π⁰` storage convention: the physical
//! Dyson equation is `χ = χ⁰ + χ⁰ K χ` with `χ = −Π`, `K = I + F̃`.)
//!
//! Setting `F̃ = 0` and *additionally* dropping the Hartree identity recovers
//! `Π = Π⁰` — the uncoupled response — which is why
//! [`CksDispersionConfig::coupling`] has three settings rather than two.
//!
//! # The sharp internal test, and what it does and does not say
//!
//! At the uncoupled level, second-order perturbation theory in the *inter*monomer
//! Coulomb interaction gives the UCHF dispersion. For two monomers whose orbitals
//! come from the **same dimer-basis SCF**, the trace (3) with `Π⁰` reproduces the
//! non-additive part of the direct-MP2 energy, i.e. of `2·E_OS` (the direct
//! ring term of the second-order energy).
//!
//! **That identity is intermolecular and asymptotic, not absolute.** `2·E_OS`
//! evaluated on a single system is a supermolecular correlation energy
//! containing intramonomer correlation; `E_disp^(2)` is by construction
//! interaction-only, so only the dimer-minus-monomers difference is
//! comparable. Even that difference differs at finite `R`: the
//! supermolecular `2·E_OS` is built on dimer-HF orbitals, which carry
//! intermolecular exchange and orbital deformation, while the trace uses each
//! monomer's own orbitals. Both sides share the dimer basis and the frequency
//! grid, so this is the whole mismatch. `tests/cks_dispersion_uncoupled.rs`
//! measures it on He₂/cc-pVDZ: 5.2e-3 relative at 4 Å, 3.0e-8 at 8 Å, and at
//! 16 Å an absolute residual of 2.1e-16 Ha (the arithmetic floor).
//!
//! Switching the coupling off must recover the uncoupled number *bit-for-bit*
//! through the same code path — that one **is** exact and is asserted.
//!
//! # Scope
//!
//! - Closed-shell restricted monomer references only.
//! - The XC kernel is **adiabatic LDA** (ALDA / ALDAX), the kernel the
//!   published MP2C dispersion correction uses. GGA and meta-GGA kernels are
//!   refused (see [`build_monomer_response`]).
//! - The aux-basis projection of `f_xc` is done on the DFT quadrature grid (see
//!   `project_fxc_to_aux`); it inherits that grid's accuracy.
//! - Library-only: no CLI or Python surface.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_mp2::rimp2::RpaIntermediates;
use ndarray::{Array1, Array2};

/// Which response the dispersion trace is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CksCoupling {
    /// Bare particle–hole bubble `Π⁰`. This is the **UCHF/uncoupled** response —
    /// the one MP2 already contains, and the term MP2C *subtracts*.
    ///
    /// No Dyson resummation at all: neither the Hartree nor the XC kernel.
    Uncoupled,
    /// Dyson-resummed with the **Hartree kernel only** (`F̃ = 0`), i.e. the
    /// coupled-Hartree / RPA-level response.
    ///
    /// Useful as a diagnostic middle rung: it isolates how much of the
    /// coupled−uncoupled difference is Coulomb screening versus XC.
    CoupledHartree,
    /// Full **CKS**: Dyson-resummed with Hartree *and* the adiabatic XC kernel.
    /// This is the response the published MP2C uses.
    #[default]
    Cks,
}

/// Configuration for [`cks_dispersion_energy`].
#[derive(Debug, Clone)]
pub struct CksDispersionConfig {
    /// Which response to build. Default [`CksCoupling::Cks`].
    pub coupling: CksCoupling,
    /// Number of imaginary-frequency quadrature points. Default 12.
    ///
    /// The Casimir–Polder integrand is smooth and decays like `ω^{-4}`, so a
    /// modest Gauss–Legendre grid on the `tan` substitution converges quickly;
    /// 12 points reproduces 16 to well under a microhartree on the systems
    /// tested. Convergence is the caller's to verify on a new system —
    /// [`CksDispersionResult::per_frequency`] is exposed for exactly that.
    pub n_freq: usize,
    /// Frequency-grid scale `u₀` (Hartree) for the `tan` substitution. Default
    /// 1.0, which places the nodes across the range where valence excitation
    /// energies live.
    pub u0: f64,
    /// Frozen-core orbitals, forwarded to the response build. Must match
    /// whatever the term this is differenced against used, or the cancellation
    /// is polluted.
    pub frozen_core: usize,
}

impl Default for CksDispersionConfig {
    fn default() -> Self {
        Self {
            coupling: CksCoupling::Cks,
            n_freq: 12,
            u0: 1.0,
            frozen_core: 0,
        }
    }
}

impl CksDispersionConfig {
    fn validate(&self) -> Result<(), FerricError> {
        if self.n_freq == 0 {
            return Err(FerricError::General(
                "cks_dispersion: n_freq must be >= 1 (got 0)".into(),
            ));
        }
        if self.u0 <= 0.0 || !self.u0.is_finite() {
            return Err(FerricError::General(format!(
                "cks_dispersion: u0 must be a finite positive scale (got {})",
                self.u0
            )));
        }
        Ok(())
    }
}

/// Result of a Casimir–Polder dispersion evaluation.
#[derive(Debug, Clone)]
pub struct CksDispersionResult {
    /// `E_disp^(2)` in Hartree. Negative for a physical (attractive) pair.
    pub e_disp: f64,
    /// Which response was used, recorded so a consumer can tell an uncoupled
    /// from a coupled result after the fact.
    pub coupling: CksCoupling,
    /// The imaginary-frequency nodes `ω_k` (Hartree).
    pub freqs: Vec<f64>,
    /// Quadrature weights `w_k`.
    pub weights: Vec<f64>,
    /// Per-frequency integrand `Tr[Π̃_A(iω_k) Π̃_B(iω_k)]`, **before** the
    /// `−1/2π` prefactor and before weighting.
    ///
    /// Exposed so a caller can check frequency-grid convergence without
    /// re-running: a well-resolved integrand decays smoothly and monotonically
    /// in `ω`. A ragged or non-decaying tail means `n_freq`/`u0` are wrong for
    /// the system, and that is a silent accuracy loss otherwise.
    pub per_frequency: Vec<f64>,
    /// Smallest eigenvalue of the static (`ω = 0`) Dyson matrix
    /// `I + Π⁰^{1/2} K Π⁰^{1/2}` over both monomers, or `None` for
    /// [`CksCoupling::Uncoupled`]. `≤ 0` means the coupled response has a
    /// negative mode: the integrand crosses a pole and `e_disp` is not
    /// meaningful (a warning is printed). In every case measured the minimum
    /// rises monotonically with `ω`, so `ω = 0` is where it is lowest.
    pub static_dyson_min_eigenvalue: Option<f64>,
}

/// `min eig M(0)` over both monomers, `None` when the coupling has no Dyson step.
fn static_dyson_min(
    a: &MonomerResponse,
    b: &MonomerResponse,
    coupling: CksCoupling,
) -> Result<Option<f64>, FerricError> {
    let ma = a.dyson_min_eigenvalue(0.0, coupling)?;
    let mb = b.dyson_min_eigenvalue(0.0, coupling)?;
    Ok(match (ma, mb) {
        (Some(x), Some(y)) => Some(x.min(y)),
        _ => None,
    })
}

/// Frequency-dependent response of one monomer, in the dressed auxiliary basis.
///
/// Holds everything needed to rebuild `Π̃(iω)` at an arbitrary frequency. The
/// expensive, frequency-independent work (the RI transform, and the aux-basis
/// projection of `f_xc`) is done once at construction.
#[derive(Debug, Clone)]
pub struct MonomerResponse {
    /// Dressed RI tensor `V^{-1/2}(P|ia)`, shape `(naux, nocc·nvir)`.
    b_ov: Array2<f64>,
    /// Active occupied orbital energies.
    eps_occ: Vec<f64>,
    /// Virtual orbital energies.
    eps_vir: Vec<f64>,
    /// Adiabatic XC kernel projected into the dressed aux basis, `(naux, naux)`.
    /// `None` for a coupling level that does not use it.
    fxc_aux: Option<Array2<f64>>,
    naux: usize,
    /// Cartesian center (Bohr) of every auxiliary basis function, in aux-index
    /// order.
    ///
    /// Carried solely so [`cks_dispersion_energy`] can verify that aux index `P`
    /// denotes the **same point in space** for both monomers. See that
    /// function's alignment check — this was a live, silent, 130×-magnitude bug.
    aux_centers: Vec<[f64; 3]>,
}

impl MonomerResponse {
    /// Number of auxiliary functions — both monomers must agree, since they are
    /// traced against each other in a shared (dimer) auxiliary basis.
    pub fn naux(&self) -> usize {
        self.naux
    }

    /// Zero the XC kernel in place, leaving everything else identical.
    ///
    /// Exists for one test — `cks_with_null_fxc_equals_coupled_hartree_bitwise`
    /// — which pins that `Cks` and `CoupledHartree` differ *only* by the kernel
    /// matrix and share every other step of the Dyson solve. Doing that by
    /// choosing a functional with a vanishing `f_xc` would test libxc rather
    /// than this module; zeroing the matrix directly tests the claim itself.
    pub fn zero_fxc_for_test(&mut self) {
        if let Some(f) = &mut self.fxc_aux {
            f.fill(0.0);
        }
    }

    /// Eigenvalues of the full Dyson kernel `K = I + F̃`, ascending.
    ///
    /// # Why this is exposed
    ///
    /// If `K ≻ 0` then `Π = [I + Π⁰K]⁻¹Π⁰` is a strict contraction of every
    /// `Π⁰` eigenvalue, so `|E_disp^CKS| < |E_disp^UCHF|` and the MP2C
    /// correction `E^CKS − E^UCHF` is positive for every system. A negative
    /// correction (which the published MP2C table reports for the methane and
    /// ethene dimers) needs `K` to have a negative eigenvalue. Measured on
    /// water/cc-pVDZ, `K` is strongly indefinite (ALDAX: min eig −5.59, 50 of
    /// 84 below zero; `tests/cks_kernel_spectrum.rs`), so a negative correction
    /// is reachable and the positive corrections ferric reports are not
    /// explained by the kernel's spectrum.
    ///
    /// Returns `None` at a coupling level that carries no XC kernel, where
    /// `K = I` trivially and the question does not arise.
    pub fn kernel_eigenvalues(&self) -> Option<Vec<f64>> {
        use ndarray_linalg::Eigh;
        let f = self.fxc_aux.as_ref()?;
        let k = Array2::<f64>::eye(self.naux) + f;
        // K is symmetric by construction (F̃ = V^{-1/2} F V^{-1/2} with F
        // symmetric), so a symmetric eigensolve is the right tool and its
        // eigenvalues are real.
        let (vals, _) = k.eigh(ndarray_linalg::UPLO::Lower).ok()?;
        Some(vals.to_vec())
    }

    /// Build `Π⁰(iω)` — the bare bubble, ferric's positive-storage convention.
    ///
    /// Delegates to the same `sternheimer` builder the RPA energy path uses, so
    /// the two cannot drift apart in convention.
    fn pi0(&self, omega: f64) -> Array2<f64> {
        crate::sternheimer::dielectric_matrix_with_scale(
            &Array2::<f64>::eye(self.naux),
            &self.b_ov,
            &build_scale(&self.eps_occ, &self.eps_vir, omega),
        )
        // `dielectric_matrix_with_scale` returns I + Π⁰; strip the identity.
        .map(|x| *x)
            - Array2::<f64>::eye(self.naux)
    }

    /// The Dyson kernel `K` at a coupling level: `I` (Hartree, absorbed by the
    /// `V^{-1/2}` dressing) plus `F̃` for [`CksCoupling::Cks`]. `None` for
    /// [`CksCoupling::Uncoupled`], which has no Dyson step.
    fn dyson_kernel(&self, coupling: CksCoupling) -> Option<Array2<f64>> {
        match coupling {
            CksCoupling::Uncoupled => None,
            CksCoupling::CoupledHartree | CksCoupling::Cks => {
                let mut kernel = Array2::<f64>::eye(self.naux);
                if coupling == CksCoupling::Cks {
                    if let Some(f) = &self.fxc_aux {
                        kernel += f;
                    }
                }
                Some(kernel)
            }
        }
    }

    /// Build the response at the requested coupling level.
    ///
    /// For the coupled levels this solves the Dyson equation (5) as a linear
    /// system rather than forming an explicit inverse.
    fn response(&self, omega: f64, coupling: CksCoupling) -> Result<Array2<f64>, FerricError> {
        let pi0 = self.pi0(omega);
        match self.dyson_kernel(coupling) {
            None => Ok(pi0),
            Some(kernel) => {
                // Solve [I + Π⁰ K] Π = Π⁰ for Π.
                let lhs = Array2::<f64>::eye(self.naux) + pi0.dot(&kernel);
                solve_linear_system(&lhs, &pi0)
            }
        }
    }

    /// Smallest eigenvalue of the symmetrized Dyson matrix
    /// `M(iω) = I + Π⁰^{1/2} K Π⁰^{1/2}` at one imaginary frequency.
    ///
    /// `I + Π⁰K` is similar to `M` (`Π⁰ ⪰ 0`), so its eigenvalues are real and
    /// equal `M`'s. `Π = Π⁰^{1/2} M⁻¹ Π⁰^{1/2}`, so:
    ///
    /// * `min eig M > 0` at every node: the coupled response is positive
    ///   semi-definite and the Dyson step is pole-free on the grid;
    /// * `min eig M` near 0: that node sits near a pole and the integrand is
    ///   amplified there;
    /// * `min eig M < 0`: the response has a negative mode — the reference is
    ///   unstable under this kernel (a triplet-like instability of the
    ///   orbitals with respect to ALDA), and `Π` is no longer a physical
    ///   density–density response.
    ///
    /// This is the per-frequency diagnostic for the open MP2C sign
    /// discrepancy (see the module docs). Returns `None` for
    /// [`CksCoupling::Uncoupled`], where `M = I`.
    pub fn dyson_min_eigenvalue(
        &self,
        omega: f64,
        coupling: CksCoupling,
    ) -> Result<Option<f64>, FerricError> {
        use ndarray_linalg::Eigh;
        let Some(kernel) = self.dyson_kernel(coupling) else {
            return Ok(None);
        };
        let pi0 = self.pi0(omega);
        let (vals, vecs) = pi0.eigh(ndarray_linalg::UPLO::Lower).map_err(|e| {
            FerricError::Lapack(format!(
                "cks_dispersion: eigh of the bare bubble failed: {e}"
            ))
        })?;
        // Π⁰^{1/2}; Π⁰ is PSD up to roundoff, so clamp tiny negatives to zero.
        let mut scaled = vecs.clone();
        for (j, &v) in vals.iter().enumerate() {
            let r = v.max(0.0).sqrt();
            scaled.column_mut(j).mapv_inplace(|x| x * r);
        }
        let sqrt_pi0 = scaled.dot(&vecs.t());
        let m = Array2::<f64>::eye(self.naux) + sqrt_pi0.dot(&kernel).dot(&sqrt_pi0);
        let (mv, _) = m.eigh(ndarray_linalg::UPLO::Lower).map_err(|e| {
            FerricError::Lapack(format!(
                "cks_dispersion: eigh of the Dyson matrix failed: {e}"
            ))
        })?;
        Ok(mv.iter().cloned().reduce(f64::min))
    }
}

/// `sqrt(4·e_ia/(ω²+e_ia²))` scale factors — the same convention
/// `sternheimer::build_scale_factors` uses (prefactor 4 for a closed shell).
fn build_scale(eps_occ: &[f64], eps_vir: &[f64], omega: f64) -> Array1<f64> {
    let omega2 = omega * omega;
    let mut s = Array1::<f64>::zeros(eps_occ.len() * eps_vir.len());
    for (i, &ei) in eps_occ.iter().enumerate() {
        for (a, &ea) in eps_vir.iter().enumerate() {
            let e_ia = ea - ei;
            s[i * eps_vir.len() + a] = (4.0 * e_ia / (omega2 + e_ia * e_ia)).sqrt();
        }
    }
    s
}

/// Solve `A X = B` for `X`, mapping a singular/NaN factorization to a clean
/// `Err` rather than a panic (reliability convention: a numerically dead
/// dielectric must not abort the process).
fn solve_linear_system(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::{FactorizeInto, Solve};
    let n = a.nrows();
    let mut out = Array2::<f64>::zeros((n, b.ncols()));
    let factored = a.clone().factorize_into().map_err(|e| {
        FerricError::Lapack(format!(
            "cks_dispersion: Dyson system factorization failed \
             (singular or NaN response — near-degenerate reference?): {e}"
        ))
    })?;
    for (j, col) in b.columns().into_iter().enumerate() {
        let x = factored.solve(&col.to_owned()).map_err(|e| {
            FerricError::Lapack(format!(
                "cks_dispersion: Dyson solve failed on column {j}: {e}"
            ))
        })?;
        out.column_mut(j).assign(&x);
    }
    Ok(out)
}

/// Project the adiabatic XC kernel into the **dressed** auxiliary basis.
///
/// # What is being projected
///
/// The XC kernel is a real-space multiplicative object,
/// `f_xc(r,r') = δ²E_xc/δρ(r)δρ(r')`, which for a (semi)local functional is
/// `f_xc(r) δ(r−r')` plus gradient corrections. Its aux-basis matrix element is
///
/// ```text
///   F_PQ = ∫dr  P(r) f_xc(r) Q(r)                                        (6)
/// ```
///
/// evaluated on the DFT quadrature grid: `F_PQ ≈ Σ_g w_g P(r_g) f_xc(r_g) Q(r_g)`.
///
/// # Why the dressing matters, and the trap it hides
///
/// (6) is in the **bare** aux basis, but `Π⁰` from `b_ov` lives in the
/// `V^{-1/2}`-dressed one. The Dyson equation (5) mixes the two, so the kernel
/// must be dressed to match:
///
/// ```text
///   F̃ = V^{-1/2} F V^{-1/2} .                                            (7)
/// ```
///
/// Skipping (7) is the natural mistake here and it is **not** loud: the result
/// stays finite, symmetric, and the right order of magnitude — it is simply the
/// wrong number, by whatever the aux metric's condition number happens to be.
/// `v_inv_sqrt` is carried on [`RpaIntermediates`] precisely so this step has
/// no excuse to be skipped. The uncoupled-limit identity in
/// `tests/cks_dispersion_uncoupled.rs` does not exercise `F̃`; the pinned ALDA
/// values in `cross_spin_fxc_term_is_included` do.
///
/// # Spin convention — the factor that was wrong once
///
/// `Π⁰` here is the **spin-summed density–density** response (ferric's
/// prefactor 4 in (4) is `2` for spin × `2` for the closed-shell double
/// occupation; that convention is fixed by the dRPA dielectric `ε = I + Π⁰`,
/// which is validated against PySCF). The kernel entering the Dyson equation
/// must therefore be the matching spin-summed density–density second
/// derivative:
///
/// ```text
///   f_ρρ = δ²E_xc/δρ δρ = ¼(f_αα + 2f_αβ + f_ββ) = ½(f_αα + f_αβ)          (8)
/// ```
///
/// the last equality using `f_αα = f_ββ` for a closed shell. Equivalently: a
/// total perturbation `δρ` splits as `δρ_α = δρ_β = δρ/2`, so the potential it
/// induces is `δV = f_αα δρ_α + f_αβ δρ_β = ½(f_αα + f_αβ) δρ`.
///
/// **The `½` is load-bearing and its absence is not obvious from the output.**
/// The first version of this function omitted it. The result stayed finite,
/// attractive, and the right order of magnitude — but the doubled kernel
/// over-cancelled the Hartree screening so thoroughly that the CKS dispersion
/// came out *larger* in magnitude than the uncoupled one (measured on He₂:
/// −1.0234e-6 vs −1.0219e-6 Ha), which would have inverted the sign of the
/// entire MP2C correction. `coupling_screens_dispersion` is the test that
/// catches it, and it catches it only because it asserts an *ordering* between
/// the three coupling levels rather than merely that each is negative.
fn project_fxc_to_aux(
    mol: &Molecule,
    dfbs: &PreparedBasis,
    scf_density: &Array2<f64>,
    obs_bs: &ferric_core::basis::BasisSet,
    v_inv_sqrt: &Array2<f64>,
    xc_name: &str,
    grid_cfg: &ferric_dft::grid::AtomicGridConfig,
) -> Result<Array2<f64>, FerricError> {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::density_on_grid::eval_density_closed;
    use ferric_dft::grid::build_atomic_grid;
    use ferric_dft::libxc::{xc_def_from_name_nspin, FunctionalFamily};

    let xc = xc_def_from_name_nspin(xc_name, 2).map_err(|e| {
        FerricError::General(format!("cks_dispersion: bad functional '{xc_name}': {e:?}"))
    })?;

    // FAMILY GATE — every accepted functional must be one whose f_xc this
    // routine actually evaluates.
    //
    // This gate exists because of a live bug caught by mutation testing, and the
    // failure mode is worth recording. The first version of this function called
    // `eval_lda_fxc_polarized` unconditionally. On a GGA that is not an
    // approximation — libxc's `xc_lda_fxc` on a GGA functional writes **zeros**,
    // so `Cks` silently degenerated into `CoupledHartree`: same number to the
    // last bit, no warning, no error, and every test at the time still passed.
    // `tests/cks_dispersion_uncoupled.rs::xc_kernel_changes_the_answer` now pins
    // this directly.
    //
    // ALDA (LDA f_xc) is a legitimate and widely used CKS/TDDFT kernel, so the
    // LDA path is a real capability rather than a fallback. GGA is REFUSED, not
    // silently reduced to its ρρ block: the σ-coupling terms (v2rhosigma,
    // v2sigma2) require ∇χ_P in the auxiliary basis and a corresponding
    // gradient-projected kernel, which is genuine unimplemented work — see the
    // module docs' scope section. Truncating them would be exactly the
    // silently-plausible-but-wrong number this gate is here to prevent.
    for f in &xc.funcs {
        match f.family() {
            FunctionalFamily::Lda => {}
            FunctionalFamily::Gga | FunctionalFamily::HybridGga | FunctionalFamily::RangeSepGga => {
                return Err(FerricError::General(format!(
                    "cks_dispersion: functional '{xc_name}' is GGA-family. The CKS kernel here \
                     is ALDA (LDA f_xc) only; the GGA σ-coupling terms (v2rhosigma, v2sigma2) \
                     need gradient-projected auxiliary functions, which is not implemented. \
                     Refusing rather than silently dropping them. Use an LDA functional \
                     (e.g. 'LDA', 'SVWN') for an ALDA CKS response."
                )));
            }
            FunctionalFamily::MetaGga => {
                return Err(FerricError::General(format!(
                    "cks_dispersion: functional '{xc_name}' is meta-GGA; ferric has no meta-GGA \
                     f_xc kernel (the τ second derivative is not implemented), so a CKS response \
                     cannot be built from it. Use an LDA functional for an ALDA CKS response."
                )));
            }
        }
    }

    let grid = build_atomic_grid(mol, grid_cfg);
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let npts = pts.len();

    // Aux functions on the grid: (naux, npts).
    let aux_chi = eval_basis_on_points(mol, dfbs.basis_set(), &pts).map_err(|e| {
        FerricError::General(format!("cks_dispersion: aux grid eval failed: {e:?}"))
    })?;

    // Reference density on the grid, from the ORBITAL basis (the density is an
    // orbital-basis object; the aux basis only carries the perturbation).
    let (obs_chi, obs_dchi) = ferric_dft::ao_grid::eval_basis_and_grad_on_points(mol, obs_bs, &pts)
        .map_err(|e| {
            FerricError::General(format!("cks_dispersion: obs grid eval failed: {e:?}"))
        })?;
    let rho = eval_density_closed(scf_density, &obs_chi, &obs_dchi).rho;

    // Per-grid-point spin-summed LDA kernel f = f_αα + f_αβ. `scf_density` is
    // the total closed-shell D (trace = nelec), so ρ_α = ρ_β = ρ/2.
    let mut rho_packed = vec![0.0f64; 2 * npts];
    for g in 0..npts {
        rho_packed[2 * g] = 0.5 * rho[g];
        rho_packed[2 * g + 1] = 0.5 * rho[g];
    }
    let mut v2 = vec![0.0f64; 3 * npts];
    let mut v2_tmp = vec![0.0f64; 3 * npts];
    for f in &xc.funcs {
        for x in v2_tmp.iter_mut() {
            *x = 0.0;
        }
        f.eval_lda_fxc_polarized(&rho_packed, &mut v2_tmp);
        for (a, b) in v2.iter_mut().zip(v2_tmp.iter()) {
            *a += *b;
        }
    }

    // F_PQ = Σ_g w_g P(r_g) [f_αα + f_αβ](r_g) Q(r_g).
    let mut scaled = aux_chi.clone();
    for g in 0..npts {
        let fac = grid[g].weight * 0.5 * (v2[3 * g] + v2[3 * g + 1]);
        for p in 0..scaled.nrows() {
            scaled[(p, g)] *= fac;
        }
    }
    let f_bare = scaled.dot(&aux_chi.t());

    // Dress: F̃ = V^{-1/2} F V^{-1/2}. See the doc comment — omitting this is
    // silent and wrong.
    Ok(v_inv_sqrt.dot(&f_bare).dot(&v_inv_sqrt.t()))
}

/// Build a monomer's frequency-dependent response in a **given** (dimer)
/// auxiliary basis.
///
/// # The monomer-in-dimer-basis construction
///
/// `mol` must be the monomer *with ghost centers* at the partner fragment's
/// atomic positions (`@C`, `@H`, … in the XYZ). ferric's ghost handling makes
/// this work with no special-casing here: a ghost contributes basis functions
/// and zero electrons, so `nbasis` (hence `nvir`) sees the full dimer basis
/// while `nelec` (hence `nocc`) sees only the monomer. That is precisely the
/// counterpoise/monomer-in-dimer-basis prescription, and it is what makes the
/// two monomers' responses live in a **common** auxiliary space so the trace in
/// (3) is well defined.
///
/// Both monomers must therefore be prepared with the *same* `dfbs` (same aux
/// basis over the same union of centers), **listed in the same order**.
/// [`cks_dispersion_energy`] checks both the dimension and the per-function
/// center alignment and errors if either fails.
///
/// The ordering requirement is not pedantry. Writing the two CP fragments as
///
/// ```text
///   A:  He 0 0 0     @He 0 0 R          B:  He 0 0 R     @He 0 0 0
/// ```
///
/// gives both monomers the same `naux` and a completely permuted aux index. The
/// trace then multiplies A's on-A block by B's on-B block and returns a monomer
/// self-response — flat in `R`, ~10⁴× too large, and entirely silent. Write them
/// as
///
/// ```text
///   A:  He 0 0 0     @He 0 0 R          B:  @He 0 0 0     He 0 0 R
/// ```
///
/// instead: one common center ordering, flipping only which centers are ghosts.
///
/// # Arguments
///
/// * `scf` — a converged **closed-shell restricted** SCF for the monomer in the
///   dimer basis. For a genuine CKS response this should be a *Kohn–Sham*
///   reference (see [`cks_dispersion_energy`]'s functional discussion); an RHF
///   reference gives the coupled-Hartree–Fock response instead, which is a
///   different (and, for MP2C, incorrect) object.
/// * `xc_name` — the functional whose adiabatic `f_xc` supplies the XC kernel.
///   Ignored when `coupling` needs no kernel.
pub fn build_monomer_response(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    scf: &ferric_scf::ScfResult,
    xc_name: &str,
    cfg: &CksDispersionConfig,
) -> Result<MonomerResponse, FerricError> {
    use ferric_mp2::rimp2::{active_occ, compute_rpa_intermediates, RiMp2Config};
    use ferric_scf::result::Spin;

    if !matches!(scf.spin, Spin::Restricted) {
        return Err(FerricError::General(format!(
            "build_monomer_response requires a closed-shell restricted reference, got {:?}. \
             The factor-4 closed-shell spin summation in the bubble (4) does not hold for an \
             unrestricted reference and there is no open-shell CKS path in ferric.",
            scf.spin
        )));
    }
    cfg.validate()?;

    let ri_cfg = RiMp2Config {
        frozen_core: cfg.frozen_core,
        ..Default::default()
    };
    let inter: RpaIntermediates = compute_rpa_intermediates(
        mol,
        obs,
        dfbs,
        ferric_integrals::operator::Operator::coulomb(),
        scf,
        &ri_cfg,
    )?;

    let eps = scf.eps_r();
    let nocc_total = inter.nocc_total;
    let nocc = active_occ(nocc_total, cfg.frozen_core)?;
    let first = inter.first_occ;
    let eps_occ: Vec<f64> = eps[first..first + nocc].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..nocc_total + inter.nvir].to_vec();

    let fxc_aux = if cfg.coupling == CksCoupling::Cks {
        Some(project_fxc_to_aux(
            mol,
            dfbs,
            &scf.density_total,
            obs.basis_set(),
            &inter.v_inv_sqrt,
            xc_name,
            &ferric_dft::grid::AtomicGridConfig::default(),
        )?)
    } else {
        None
    };

    let naux = inter.b_ov.nrows();

    // Per-aux-function centers, so the trace can verify the two monomers index
    // the SAME auxiliary functions. `shell_centers()` is per shell; expand it to
    // per basis function with `shell_dims()`.
    let mut aux_centers = Vec::with_capacity(naux);
    for (si, &dim) in dfbs.shell_dims().iter().enumerate() {
        let c = dfbs.shell_centers()[si];
        for _ in 0..dim {
            aux_centers.push(c);
        }
    }
    if aux_centers.len() != naux {
        return Err(FerricError::General(format!(
            "build_monomer_response: auxiliary center map has {} entries but the RI tensor has \
             {naux} auxiliary functions — the aux basis and the RI intermediates disagree.",
            aux_centers.len()
        )));
    }

    Ok(MonomerResponse {
        b_ov: inter.b_ov,
        eps_occ,
        eps_vir,
        fxc_aux,
        naux,
        aux_centers,
    })
}

/// The Casimir–Polder dispersion energy between two monomer responses.
///
/// ```text
///   E_disp = −(1/2π) Σ_k w_k Tr[ Π̃_A(iω_k) Π̃_B(iω_k) ]
/// ```
///
/// # Functional choice
///
/// MP2C's coupled term is a *Kohn–Sham* response, so `resp_a`/`resp_b` should
/// have been built on KS references with an LDA-family kernel (`"LDA_X"` is
/// the published ALDAX choice). The kernel itself is chosen when the responses
/// are built, in [`build_monomer_response`], which refuses GGA and meta-GGA.
///
/// # Errors
///
/// The two responses must share an auxiliary basis dimension. They will if both
/// monomers were prepared in the same dimer basis, which is the only correct way
/// to use this function; a mismatch is a hard error rather than a meaningless
/// trace over incompatible spaces.
pub fn cks_dispersion_energy(
    resp_a: &MonomerResponse,
    resp_b: &MonomerResponse,
    cfg: &CksDispersionConfig,
) -> Result<CksDispersionResult, FerricError> {
    use ferric_integrals::blas_threads::with_blas_threads;
    use rayon::prelude::*;

    cfg.validate()?;
    if resp_a.naux != resp_b.naux {
        return Err(FerricError::General(format!(
            "cks_dispersion_energy: monomer responses live in different auxiliary spaces \
             (naux_A = {}, naux_B = {}). Both monomers must be prepared in the SAME dimer \
             auxiliary basis (ghost centers at the partner's positions) — the Casimir–Polder \
             trace is otherwise meaningless.",
            resp_a.naux, resp_b.naux
        )));
    }

    // ALIGNMENT — matching `naux` is necessary but nowhere near sufficient.
    //
    // `Tr[Π_A Π_B] = Σ_PQ Π_A,PQ Π_B,QP` pairs aux index `P` of monomer A with
    // aux index `P` of monomer B. That is only the Casimir–Polder integral if
    // index `P` denotes the *same auxiliary function at the same point in space*
    // for both. Two monomers built from XYZ files that list their atoms in
    // different orders — the natural thing to write, and what
    // `benchmarks/grid/geoms/meth_m{A,B}_cp.xyz` actually did — have identical
    // `naux` and completely permuted aux indices.
    //
    // The failure is silent and severe. With the aux blocks positionally
    // reversed, A's on-A block multiplies B's on-B block, so the cross-trace
    // becomes numerically equal to A's *self*-trace `Tr[Π_A Π_A]` — a monomer
    // property with no R dependence at all. Measured on He₂/cc-pVDZ, `E_disp`
    // came out flat at −64.82 kcal/mol from 3 Å to 24 Å (correct value at 3 Å:
    // −3.9e-3 kcal/mol, decaying as a clean R⁻⁶). The methane and ethene dimers
    // reported −270 and −343 kcal/mol for the same reason, while a24-12 — whose
    // CP fragment files *do* share an atom ordering — was correct throughout.
    // That contrast was misread as a property of the systems; it was this bug.
    //
    // Comparing centers rather than requiring identical `Molecule`s is
    // deliberate: the monomers legitimately differ in which centers are ghosts,
    // and only the geometric layout of the auxiliary basis has to agree.
    const ALIGN_TOL: f64 = 1e-8;
    for (p, (ca, cb)) in resp_a
        .aux_centers
        .iter()
        .zip(resp_b.aux_centers.iter())
        .enumerate()
    {
        let d2 = (0..3).map(|k| (ca[k] - cb[k]).powi(2)).sum::<f64>();
        if d2 > ALIGN_TOL * ALIGN_TOL {
            return Err(FerricError::General(format!(
                "cks_dispersion_energy: auxiliary basis functions are not aligned between the \
                 two monomers. Aux index {p} is centered at ({:.6}, {:.6}, {:.6}) for monomer A \
                 but ({:.6}, {:.6}, {:.6}) for monomer B (Bohr, {:.3e} apart). The \
                 Casimir-Polder trace pairs aux index P of A with aux index P of B, so the two \
                 monomers must list their centers — real AND ghost — in the SAME order. Rebuild \
                 the CP fragments from one common atom ordering, flipping only which centers \
                 carry nuclear charge. Proceeding would silently compute a monomer self-response \
                 instead of an interaction energy.",
                ca[0],
                ca[1],
                ca[2],
                cb[0],
                cb[1],
                cb[2],
                d2.sqrt()
            )));
        }
    }

    let (freqs, weights) = crate::quadrature::gauss_legendre_nodes(cfg.n_freq, cfg.u0);

    // Per-frequency work is an independent Dyson solve + one trace. BLAS pinned
    // to 1 inside the rayon region (nesting OpenBLAS under rayon is the known
    // stack-overflow crash site); order-preserving collect into Result so a
    // singular response Errs rather than aborting a worker.
    let per_frequency: Vec<f64> = with_blas_threads(1, || {
        freqs
            .par_iter()
            .map(|&omega| {
                let pa = resp_a.response(omega, cfg.coupling)?;
                let pb = resp_b.response(omega, cfg.coupling)?;
                // Tr[A B] = Σ_ij A_ij B_ji; both are symmetric, so this is a
                // plain elementwise dot — no GEMM needed for a trace.
                let t: f64 = pa.iter().zip(pb.iter()).map(|(x, y)| x * y).sum();
                Ok(t)
            })
            .collect::<Result<Vec<f64>, FerricError>>()
    })?;

    // E = −(1/2π) Σ_k w_k Tr[...]. ferric stores Π = −χ positive, and the two
    // sign flips (one per monomer) cancel, so the overall sign is carried
    // entirely by this prefactor: a physical pair gives E_disp < 0.
    let e_disp = -(1.0 / (2.0 * std::f64::consts::PI))
        * weights
            .iter()
            .zip(per_frequency.iter())
            .map(|(w, t)| w * t)
            .sum::<f64>();

    let static_dyson_min_eigenvalue = static_dyson_min(resp_a, resp_b, cfg.coupling)?;
    if let Some(m) = static_dyson_min_eigenvalue {
        if m <= 0.0 {
            eprintln!(
                "[warning] cks_dispersion: the static Dyson matrix I + Pi0^(1/2) K Pi0^(1/2) has \
                 a negative eigenvalue ({m:.4}); the coupled response crosses a pole on the \
                 imaginary-frequency axis and E_disp = {e_disp:.6e} Ha is not meaningful."
            );
        }
    }

    Ok(CksDispersionResult {
        e_disp,
        coupling: cfg.coupling,
        freqs,
        weights,
        per_frequency,
        static_dyson_min_eigenvalue,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic monomer: deterministic pseudo-random `b_ov`, a gapped
    /// spectrum, and a symmetric indefinite `F̃` (no integrals needed).
    fn synthetic(naux: usize, nocc: usize, nvir: usize, fxc_scale: f64) -> MonomerResponse {
        let mut seed = 0x9e37_79b9_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 2001) as f64 / 1000.0 - 1.0
        };
        let b_ov = Array2::from_shape_fn((naux, nocc * nvir), |_| next());
        let raw = Array2::from_shape_fn((naux, naux), |_| next());
        let fxc = (&raw + &raw.t()) * (0.5 * fxc_scale);
        MonomerResponse {
            b_ov,
            eps_occ: (0..nocc).map(|i| -1.0 - 0.3 * i as f64).collect(),
            eps_vir: (0..nvir).map(|a| 0.2 + 0.25 * a as f64).collect(),
            fxc_aux: Some(fxc),
            naux,
            aux_centers: vec![[0.0; 3]; naux],
        }
    }

    /// `dyson_min_eigenvalue` symmetrizes `I + Π⁰K` as `I + Π⁰^{1/2}KΠ⁰^{1/2}`.
    /// Check that against an INDEPENDENT construction: the general
    /// (non-symmetric) eigensolve of `I + Π⁰K` itself, whose spectrum must be
    /// real and equal. Exercised with an indefinite kernel strong enough to
    /// push the minimum below zero, the regime the diagnostic exists for.
    #[test]
    fn dyson_min_eigenvalue_matches_the_unsymmetrized_spectrum() {
        use ndarray_linalg::Eig;
        for &(scale, omega) in &[(0.3, 0.0), (3.0, 0.0), (3.0, 0.7), (0.0, 0.2)] {
            let r = synthetic(12, 3, 5, scale);
            let sym = r
                .dyson_min_eigenvalue(omega, CksCoupling::Cks)
                .unwrap()
                .unwrap();
            let pi0 = r.pi0(omega);
            let lhs =
                Array2::<f64>::eye(r.naux) + pi0.dot(&r.dyson_kernel(CksCoupling::Cks).unwrap());
            let (vals, _) = lhs.eig().unwrap();
            let max_im = vals.iter().map(|z| z.im.abs()).fold(0.0, f64::max);
            let min_re = vals.iter().map(|z| z.re).fold(f64::INFINITY, f64::min);
            assert!(
                max_im < 1e-8,
                "I + Pi0 K must have a real spectrum, max |Im| = {max_im:.2e}"
            );
            assert!(
                (sym - min_re).abs() < 1e-9 * min_re.abs().max(1.0),
                "scale {scale}, omega {omega}: symmetrized min eig {sym:.12e} vs direct {min_re:.12e}"
            );
        }
        // The strong kernel must actually reach the negative regime, or the
        // check above never exercised it.
        let strong = synthetic(12, 3, 5, 3.0)
            .dyson_min_eigenvalue(0.0, CksCoupling::Cks)
            .unwrap();
        assert!(
            strong.unwrap() < 0.0,
            "synthetic strong kernel did not go negative: {strong:?}"
        );
    }
}
