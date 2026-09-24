//! Linear-response TDDFT for closed-shell references.
//!
//! Implements the Tamm–Dancoff approximation (TDA, equivalent to CIS for HF)
//! and the full Casida equations for singlet excitations:
//!
//! TDA:    A X = ω X
//! Casida: (A−B)^{1/2} (A+B) (A−B)^{1/2} Z = Ω² Z
//!
//! where A_{ia,jb} = δ_{ij}δ_{ab}(ε_a − ε_i) + 2(ia|jb) − c_HF(ij|ab) + K_{ia,jb}
//!       B_{ia,jb} = 2(ia|bj) − c_HF(ib|aj) + K_{ia,jb}
//!
//! and `K_{ia,jb} = (ia| f_αα + f_αβ |jb)` is the singlet XC-kernel block
//! (PySCF `get_ab`'s `iajb`). With real orbitals `(ia|f_xc|jb) = (ia|f_xc|bj)`,
//! so the same block enters A and B. It is built by
//! [`ferric_dft::lr_kernel::singlet_fxc_ov_block`] — the SAME code the
//! PySCF-validated `ferric_gw::tddft::run_tda_dft` uses — and the functional
//! gate [`ferric_dft::lr_kernel::resolve_singlet_response_xc`] hard-refuses
//! every functional without a complete kernel (meta-GGA, VV10, range-separated
//! hybrids). There is no kernel-less DFT path: a KS reference either gets its
//! f_xc term or an error.
//!
//! Scope: closed-shell (restricted) references, singlet excitations. An
//! open-shell reference is refused — there is no spin-resolved A/B assembly
//! here, for HF or KS.

use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_dft::fxc::GgaFxcKernel;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::lr_kernel::{
    resolve_singlet_response_xc, singlet_fxc_ov_block, FXC_BLOCK_ASYMMETRY_TOL,
};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex;
use ferric_mp2::rimp2::{eri3_budget_bytes, metric_inverse_sqrt, stream_dressed_mo_band};
use ferric_scf::ScfResult;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

/// TDDFT solution method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TddftMethod {
    Tda,
    Casida,
}

/// Configuration for a TDDFT calculation.
#[derive(Debug, Clone)]
pub struct TddftConfig {
    pub n_roots: usize,
    pub method: TddftMethod,
    /// Optional resident-bytes ceiling for the dense `(ia, jb)` matrices and
    /// the RI 3-index build. `None` → resolved via
    /// [`ferric_core::memory::resolve_budget_bytes`] (env / cgroup / RAM
    /// auto-detect).
    ///
    /// This field did not exist before: the crate had NO budget machinery at
    /// all, so `[memory] budget_gb` and a caller-supplied ceiling were both
    /// unreachable from here, and `build_b_tensors` hardcoded
    /// `eri3_budget_bytes(None)` — the same "silently discarded budget" defect
    /// ferric-rpa fixed three times (see `ferric_rpa::properties`). Threading
    /// it makes a pinned budget actually reach the allocations; see
    /// `caller_budget_is_honoured_not_discarded` in the tests below.
    pub memory_budget_bytes: Option<usize>,
    /// The functional the reference was converged with (e.g. `"PBE"`,
    /// `"B3LYP"`), or `None` for a Hartree–Fock reference (CIS / TDHF).
    ///
    /// It selects BOTH the exact-exchange fraction and the f_xc kernel, so it
    /// must name the reference's functional: `run_tddft` cannot check that
    /// (an `ScfResult` does not record its functional). A functional without a
    /// complete kernel (meta-GGA, VV10, range-separated) is refused.
    pub xc: Option<String>,
    /// Integration grid for the f_xc kernel. Must match the grid the reference
    /// SCF used (`RhfConfig::dft_grid`; `AtomicGridConfig::default()` when that
    /// is `None`), or the kernel is evaluated on a different quadrature than
    /// the density it linearizes. Ignored for an HF reference.
    pub grid: AtomicGridConfig,
    /// Override the exact-exchange fraction on the `−c_HF` terms. `None`
    /// (default) takes it from `xc` (1 for an HF reference). An override is a
    /// deliberate model change and is only accepted WITH a functional: an HF
    /// reference with `c_HF != 1` is neither CIS/TDHF nor any TDDFT, and is
    /// refused.
    pub c_hf_override: Option<f64>,
}

impl TddftConfig {
    /// `n_roots` singlets by `method` on an HF reference, default grid, no
    /// budget pin and no `c_HF` override. Set `xc` for a KS reference.
    pub fn new(n_roots: usize, method: TddftMethod) -> Self {
        Self {
            n_roots,
            method,
            memory_budget_bytes: None,
            xc: None,
            grid: AtomicGridConfig::default(),
            c_hf_override: None,
        }
    }
}

impl Default for TddftConfig {
    fn default() -> Self {
        Self::new(3, TddftMethod::Tda)
    }
}

/// Result of a TDDFT calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct TddftResult {
    /// Lowest `n_roots` positive singlet excitation energies (Hartree), ascending.
    pub excitation_energies: Vec<f64>,
    /// Length-gauge oscillator strengths, PySCF convention
    /// (`f = (2/3) Ω |<0|r|n>|²`, `<0|r|n> = √2 Σ_ia (X+Y)_ia <i|r|a>` with
    /// `|X|² − |Y|² = 1`; `Y = 0` for TDA).
    pub oscillator_strengths: Vec<f64>,
    /// `<0|r|n>` (a.u., origin at 0), same ordering as the energies.
    pub transition_dipoles: Vec<[f64; 3]>,
    pub method: TddftMethod,
    /// Exact-exchange fraction actually used on the `−c_HF` terms.
    pub c_hf: f64,
    /// Whether the `(ia|f_xc|jb)` kernel block was added to A (and B). True for
    /// every KS reference — there is no kernel-less DFT path — and false only
    /// for an HF reference, where the kernel is identically zero.
    pub fxc_included: bool,
}

impl std::fmt::Display for TddftResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ha_to_ev = 27.211_386_245_988;
        writeln!(
            f,
            "TDDFT {:?} — {} roots:",
            self.method,
            self.excitation_energies.len()
        )?;
        for (i, (&e, &osc)) in self
            .excitation_energies
            .iter()
            .zip(&self.oscillator_strengths)
            .enumerate()
        {
            writeln!(
                f,
                "  {}: {:.6} Ha  ({:.4} eV)  f = {:.6}",
                i + 1,
                e,
                e * ha_to_ev,
                osc
            )?;
        }
        Ok(())
    }
}

/// Build the RI-dressed 3-index tensors needed for TDDFT.
///
/// Returns `(b_ov, b_oo, b_vv)`:
///   - `b_ov`: (naux, nocc*nvir)  — `(ia|jb) = Σ_P b_ov[P,ia] * b_ov[P,jb]`
///   - `b_oo`: (naux, nocc*nocc)  — for exchange `(ij|ab)`
///   - `b_vv`: (naux, nvir*nvir)  — for exchange `(ij|ab)`
///
/// All three are returned to the caller and stay live for the whole of
/// [`run_tddft`], so all three are `Resident` in `plan`. `b_vv` is the largest
/// (nvir ≫ nocc at any production basis), and it is the term that shows up
/// first in the breakdown when this gate fires.
///
/// The `naux×nao²` AO source is streamed aux-blocked under the SAME budget
/// (`ThreeIndexSource::build` spills to disk rather than allocating past it),
/// so it is not charged here — but its ceiling now comes from
/// `eri3_budget_bytes(memory_budget_bytes)` rather than the hardcoded
/// `eri3_budget_bytes(None)` that used to discard a caller's budget outright.
fn build_b_tensors(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
    plan: &mut MemoryPlan,
    memory_budget_bytes: Option<usize>,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), FerricError> {
    let op = Operator::coulomb();
    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v2c_inv_sqrt = metric_inverse_sqrt(&v2c, op)?;
    let budget = eri3_budget_bytes(memory_budget_bytes);
    let mut src = ThreeIndexSource::build(op, obs, dfbs, budget)?;

    // naux is only known once the metric is built, so the RI reservations are
    // declared here rather than at the entry point — still BEFORE the first
    // large allocation (`stream_dressed_mo_band` below), which is the property
    // that matters.
    let naux = v2c_inv_sqrt.nrows();
    let (nocc, nvir) = (c_occ.ncols(), c_vir.ncols());
    plan.reserve(
        "B(P|ia) [b_ov]",
        naux.saturating_mul(nocc * nvir),
        Lifetime::Resident,
    );
    plan.reserve(
        "B(P|ij) [b_oo]",
        naux.saturating_mul(nocc * nocc),
        Lifetime::Resident,
    );
    plan.reserve(
        "B(P|ab) [b_vv]",
        naux.saturating_mul(nvir * nvir),
        Lifetime::Resident,
    );
    plan.check()?;
    let b_ov = stream_dressed_mo_band(&mut src, &v2c_inv_sqrt, c_occ, c_vir, None)?;
    let b_oo = stream_dressed_mo_band(&mut src, &v2c_inv_sqrt, c_occ, c_occ, None)?;
    let b_vv = stream_dressed_mo_band(&mut src, &v2c_inv_sqrt, c_vir, c_vir, None)?;
    Ok((b_ov, b_oo, b_vv))
}

/// Declare every dense `(dim, dim)` matrix the requested method holds, with the
/// lifetimes read off the source rather than guessed.
///
/// This is the single place the peak is written down, and it is deliberately
/// verbose about WHICH matrices coexist, because co-residency is the part that
/// goes stale silently. Getting it wrong in EITHER direction is a defect:
/// under-counting OOMs, over-counting refuses jobs that would have fit.
///
/// # The scratch inside `build_a_matrix` / `build_b_matrix`
///
/// Two `dim²` buffers are live at once, regardless of `c_hf`:
///
/// * `coulomb = b_ovᵀ·b_ov` — `(dim, dim)`, in scope for the whole function.
/// * plus EITHER the `2.0 * &coulomb` temporary in `a += &(2.0 * &coulomb)`
///   (a full `dim²` materialization, live for that one statement), OR
///   `k_ij_ab = b_ooᵀ·b_vv` when `c_hf != 0` — which is `(nocc², nvir²)`, i.e.
///   exactly `dim²` elements. The scaled temporary is dropped at the end of its
///   statement, before `k_ij_ab` is built, so they never make three.
///
/// So `2·dim²` of scratch, charged as ONE `Transient`. It is not charged
/// per-worker: neither builder uses rayon.
///
/// # TDA
///
/// Stage 1 (`build_a_matrix`): `a` + `2·dim²` scratch = `3·dim²`.
/// Stage 2 (`a.eigh`): `a` (still live — `eigh` takes `&self`) + the
/// eigenvector output = `2·dim²`.
///
/// (A KS reference adds a third alternative, see [`reserve_fxc_block`].)
///
/// Peak is stage 1. Declared as `a` resident plus the *alternatives* as
/// transients, so the plan takes `a` + max(scratch, eigenvectors) = `3·dim²`
/// and NOT `4·dim²` — charging the eigenvectors as resident on top of the
/// scratch would over-estimate by a whole matrix.
///
/// # Casida
///
/// Nine `dim²` matrices are simultaneously live at the widest point (just after
/// `m.eigh`), because nothing in the arm is dropped early: `a`, `b_mat`,
/// `a_plus_b`, `a_minus_b`, `amb_vecs`, `amb_sqrt`, `m`, `z` (the eigenvector
/// output) and `x = amb_sqrt·z`.
///
/// That is stage 3, and it dominates: the `build_a`/`build_b` stage peaks at
/// `a` + `b_mat` + `2·dim²` scratch = `4·dim²`, well under nine. So the nine
/// are declared resident and the earlier scratch adds NOTHING on top — it is
/// long dropped by the time the peak is reached, and adding it would
/// over-estimate by two matrices.
fn reserve_dense_response(plan: &mut MemoryPlan, method: TddftMethod, dim: usize) {
    let d2 = dim.saturating_mul(dim);
    let build_scratch = 2usize.saturating_mul(d2);

    match method {
        TddftMethod::Tda => {
            plan.reserve("A (ia,jb)", d2, Lifetime::Resident);
            // These two are the alternating occupants of the second slot, one
            // per stage — never simultaneous, so the plan takes the larger.
            plan.reserve("build_a_matrix scratch", build_scratch, Lifetime::Transient);
            plan.reserve("eigh eigenvectors", d2, Lifetime::Transient);
        }
        TddftMethod::Casida => {
            for label in [
                "A (ia,jb)",
                "B (ia,jb)",
                "A+B",
                "A-B",
                "(A-B) eigenvectors",
                "(A-B)^1/2",
                "M = (A-B)^1/2 (A+B) (A-B)^1/2",
                "M eigenvectors Z",
                "X = (A-B)^1/2 Z",
            ] {
                plan.reserve(label, d2, Lifetime::Resident);
            }
            // Deliberately NOT declared: the build-stage scratch is dropped
            // long before the nine-matrix peak above, so charging it would
            // over-estimate. See this function's docs.
        }
    }
}

/// Declare the f_xc block's scratch (KS reference only).
///
/// [`add_fxc_block`] holds the raw and the symmetrized `K` (`2·dim²`) while
/// the matrices it adds into are live.
///
/// * TDA: `a` + K + K_sym = `3·dim²` — a third alternating occupant of the
///   second slot next to [`reserve_dense_response`]'s build scratch and
///   eigenvectors, dropped before `eigh`, so it is `Transient`.
/// * Casida: `a` + `b_mat` + K + K_sym = `4·dim²`, well under the nine-matrix
///   peak, so nothing is added.
///
/// Not called for an HF reference: over-charging would refuse jobs that fit.
fn reserve_fxc_block(plan: &mut MemoryPlan, method: TddftMethod, dim: usize) {
    if method == TddftMethod::Tda {
        let k_pair = 2usize.saturating_mul(dim.saturating_mul(dim));
        plan.reserve("f_xc block (K + K_sym)", k_pair, Lifetime::Transient);
    }
}

/// Build the TDA A matrix in the (ia, jb) compound-index space.
///
/// A_{ia,jb} = δ_{ij}δ_{ab}(ε_a − ε_i) + 2(ia|jb) − c_HF(ij|ab)
///
/// (the `+ K_{ia,jb}` f_xc block is added by the caller).
fn build_a_matrix(
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    b_ov: &Array2<f64>,
    b_oo: &Array2<f64>,
    b_vv: &Array2<f64>,
    c_hf: f64,
) -> Array2<f64> {
    let dim = nocc * nvir;
    let mut a = Array2::<f64>::zeros((dim, dim));

    // Diagonal: orbital energy differences
    for i in 0..nocc {
        for a_idx in 0..nvir {
            let ia = i * nvir + a_idx;
            a[(ia, ia)] = eps[nocc + a_idx] - eps[i];
        }
    }

    // Coulomb: 2*(ia|jb) = 2 * Σ_P B_ov[P,ia] * B_ov[P,jb]
    // This is 2 * B_ov^T . B_ov
    let b_ov_t = b_ov.t();
    let coulomb = b_ov_t.dot(b_ov);
    a += &(2.0 * &coulomb);

    // Exchange: -c_HF * (ij|ab)
    // (ij|ab) = Σ_P B_oo[P,ij] * B_vv[P,ab]
    // Mapping: compound index ia → (i,a), compound jb → (j,b)
    //   (ij|ab) uses B_oo[P, i*nocc+j] * B_vv[P, a*nvir+b]
    if c_hf != 0.0 {
        let b_oo_t = b_oo.t(); // (nocc*nocc, naux)
        let k_ij_ab = b_oo_t.dot(b_vv); // (nocc*nocc, nvir*nvir)
        for i in 0..nocc {
            for a_idx in 0..nvir {
                let ia = i * nvir + a_idx;
                for j in 0..nocc {
                    for b_idx in 0..nvir {
                        let jb = j * nvir + b_idx;
                        let ij = i * nocc + j;
                        let ab = a_idx * nvir + b_idx;
                        a[(ia, jb)] -= c_hf * k_ij_ab[(ij, ab)];
                    }
                }
            }
        }
    }

    // The (ia|f_xc|jb) block is added by `run_tddft` after this returns (it
    // needs the kernel and the reference density, and the SAME block also
    // goes into B). For an HF reference it is identically zero: CIS.

    a
}

/// Build the TDDFT B matrix in the (ia, jb) compound-index space.
///
/// B_{ia,jb} = 2(ia|bj) − c_HF(ib|aj)
///
/// (the `+ K_{ia,jb}` f_xc block is added by the caller).
///
/// Note: (ia|bj) = (ia|jb) by 4-fold symmetry of real ERIs, so the Coulomb
/// part is identical to A's Coulomb part. The exchange part differs: (ib|aj).
fn build_b_matrix(
    nocc: usize,
    nvir: usize,
    b_ov: &Array2<f64>,
    _b_oo: &Array2<f64>,
    _b_vv: &Array2<f64>,
    c_hf: f64,
) -> Array2<f64> {
    let dim = nocc * nvir;
    let mut b = Array2::<f64>::zeros((dim, dim));

    // Coulomb: 2*(ia|bj) = 2*(ia|jb) = 2 * B_ov^T . B_ov
    let b_ov_t = b_ov.t();
    let coulomb = b_ov_t.dot(b_ov);
    b += &(2.0 * &coulomb);

    // Exchange: -c_HF * (ib|aj)
    // (ib|aj) = Σ_P B_ov[P, i*nvir+b] * B_ov[P, a_idx ... wait]
    // Actually: (ib|aj) with chemist notation = <ia|bj> in physicist
    // In chemist notation: (ib|aj) = Σ_P B_{ib}^P * B_{aj}^P
    // where B_{ib} uses occ i, vir b and B_{aj} uses vir a, occ j
    // So (ib|aj) = Σ_P B_ov[P, i*nvir+b_idx] * B_ov[P, j ... ]
    // Wait — B_ov has layout [P, i*nvir+a], so B_{aj} would need vir-occ ordering.
    // Let's use B_oo and B_vv instead:
    // (ib|aj) = Σ_P B_{ib}^P * B_{aj}^P
    // where ib is (occ, vir) = B_ov, and aj is also (vir, occ) — but B_ov is (occ, vir).
    // Actually: (ib|aj) uses first index pair (i,b) and second pair (a,j).
    // In our RI: first pair = B_ov[P, i*nvir + b_idx]
    //            second pair = we need B_{vo}[P, a_idx*nocc + j]
    // We don't have B_vo but B_ov[P, j*nvir + a_idx] = B_{ja}. So:
    // (ib|aj) = Σ_P B_ov[P, i*nvir+b_idx] * B_ov[P, j*nvir+a_idx]
    // That IS just the (ib, ja) element of B_ov^T . B_ov!
    if c_hf != 0.0 {
        // k_ov = B_ov^T . B_ov has indices (ia, jb) giving (ia|jb).
        // We need (ib|aj) = k_ov[(i*nvir+b_idx), (j*nvir+a_idx)]
        // = coulomb[(i*nvir+b_idx), (j*nvir+a_idx)]
        // We already computed `coulomb` above.
        for i in 0..nocc {
            for a_idx in 0..nvir {
                let ia = i * nvir + a_idx;
                for j in 0..nocc {
                    for b_idx in 0..nvir {
                        let jb = j * nvir + b_idx;
                        let ib = i * nvir + b_idx;
                        let ja = j * nvir + a_idx;
                        b[(ia, jb)] -= c_hf * coulomb[(ib, ja)];
                    }
                }
            }
        }
    }

    b
}

/// Transition dipoles and length-gauge oscillator strengths for the states in
/// `cols` (columns of `amplitudes`), in PySCF's convention:
///
/// ```text
///   <0|r|n> = √2 Σ_ia (X+Y)_{ia,n} <i|r|a>,   f_n = (2/3) Ω_n |<0|r|n>|²
/// ```
///
/// with `|X|² − |Y|² = 1` (TDA: `Y = 0`, `|X| = 1`). The `√2` is the singlet
/// spin-adaptation factor — the same one `ferric_gw::tddft` and `bse.rs` use,
/// both cross-checked against PySCF's `oscillator_strength`. This helper
/// previously omitted it (so TDA `f` was half PySCF's) and fed the Casida
/// branch an un-normalized `(A−B)^{1/2} Z`; both are fixed here.
fn oscillator_strengths(
    obs: &PreparedBasis,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
    nocc: usize,
    nvir: usize,
    omegas: &Array1<f64>,
    amplitudes: &Array2<f64>,
    cols: &[usize],
) -> Result<(Vec<f64>, Vec<[f64; 3]>), FerricError> {
    let origin = [0.0, 0.0, 0.0];
    let mu_ao = oneelectron::dipole(obs, origin)?;

    // Transform dipole integrals to MO basis: μ^(MO)_{ia} = C_occ^T μ^(AO) C_vir
    let mu_ov: Vec<Array2<f64>> = (0..3)
        .map(|xyz| c_occ.t().dot(&mu_ao[xyz]).dot(c_vir))
        .collect();

    let mut osc = Vec::with_capacity(cols.len());
    let mut tdips = Vec::with_capacity(cols.len());
    for &n in cols {
        let omega_n = omegas[n];
        let mut tdip = [0.0_f64; 3];
        for xyz in 0..3 {
            let mut val = 0.0;
            for i in 0..nocc {
                for a in 0..nvir {
                    val += amplitudes[(i * nvir + a, n)] * mu_ov[xyz][(i, a)];
                }
            }
            tdip[xyz] = std::f64::consts::SQRT_2 * val;
        }
        let mu_sq = tdip[0] * tdip[0] + tdip[1] * tdip[1] + tdip[2] * tdip[2];
        osc.push((2.0 / 3.0) * omega_n * mu_sq);
        tdips.push(tdip);
    }
    Ok((osc, tdips))
}

/// Resolve the exact-exchange fraction and (for a KS reference) the XC
/// definition the f_xc kernel will be built from.
///
/// Split out of [`run_tddft`] so the refusals are testable without an SCF, and
/// so they fire BEFORE any integral work: a functional with no complete kernel
/// must cost the user nothing but the error.
fn resolve_response_xc(
    xc: Option<&str>,
    c_hf_override: Option<f64>,
) -> Result<(f64, Option<ferric_dft::libxc::XcDef>), FerricError> {
    if let Some(c) = c_hf_override {
        if !c.is_finite() {
            return Err(FerricError::General(format!(
                "run_tddft: c_hf override must be finite, got {c}"
            )));
        }
    }
    match xc {
        None => match c_hf_override {
            Some(c) if c != 1.0 => Err(FerricError::General(format!(
                "run_tddft: c_hf = {c} on a Hartree-Fock reference (no functional) is \
                 neither CIS/TDHF (c_hf = 1) nor any TDDFT — there is no f_xc kernel to \
                 pair it with. Name the reference functional (config xc / [tddft] xc) or \
                 drop the override."
            ))),
            _ => Ok((1.0, None)),
        },
        Some(name) => {
            let spec = resolve_singlet_response_xc(name, "run_tddft")?;
            Ok((c_hf_override.unwrap_or(spec.c_hf), Some(spec.xc_kernel)))
        }
    }
}

/// Refuse any reference `run_tddft` cannot treat, and return `nocc`.
///
/// There is no spin-resolved A/B assembly here: `eps_r`/`mos_r` and
/// `nocc = nelec/2` are closed-shell by construction (and `eps_r`/`mos_r`
/// ASSERT a restricted reference, so an open-shell input used to panic).
/// Refused for HF as well as KS — neither has an open-shell path here.
fn closed_shell_nocc(mol: &Molecule, scf: &ScfResult) -> Result<usize, FerricError> {
    if !scf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: scf.iterations,
            last_energy: scf.energy,
        });
    }
    if !matches!(scf.spin, ferric_scf::Spin::Restricted) {
        return Err(FerricError::General(format!(
            "run_tddft: closed-shell (restricted) references only; got a {:?} reference. \
             Open-shell TDA/TDDFT (spin-resolved A/B and the spin-flip kernel) is not \
             implemented.",
            scf.spin
        )));
    }
    let nelec = mol.nelec() as usize;
    if !nelec.is_multiple_of(2) {
        return Err(FerricError::General(format!(
            "run_tddft: closed-shell references only; {nelec} electrons is odd"
        )));
    }
    Ok(nelec / 2)
}

/// The f_xc kernel for a KS reference (`None` for HF), on `grid` — which must
/// be the reference SCF's grid. Its AO-on-grid cache is budget-checked inside
/// `GgaFxcKernel::new`.
fn build_kernel(
    mol: &Molecule,
    obs: &PreparedBasis,
    xc_def: Option<ferric_dft::libxc::XcDef>,
    grid: &AtomicGridConfig,
) -> Result<Option<GgaFxcKernel>, FerricError> {
    xc_def
        .map(|def| {
            GgaFxcKernel::new(mol, obs.basis_set(), def, grid).map_err(|e| {
                FerricError::General(format!("run_tddft: f_xc kernel construction: {e}"))
            })
        })
        .transpose()
}

/// Add the singlet `K_{ia,jb} = (ia|f_αα + f_αβ|jb)` block to every matrix in
/// `targets` (A for TDA; A and B for Casida — with real orbitals
/// `(ia|f_xc|jb) = (ia|f_xc|bj)`, PySCF `get_ab`'s `a += iajb; b += iajb`).
/// A no-op without a kernel (HF reference).
///
/// K is built here and dropped on return, so it is live exactly as long as
/// [`reserve_fxc_block`] assumes.
fn add_fxc_block(
    kernel: Option<&GgaFxcKernel>,
    scf: &ScfResult,
    nocc: usize,
    nvir: usize,
    targets: &mut [&mut Array2<f64>],
) -> Result<(), FerricError> {
    let Some(kern) = kernel else {
        return Ok(());
    };
    // Restricted reference: D_α = D_β = `density_alpha`.
    let d_a = &scf.density_alpha;
    let ref_dens = kern.reference_density(d_a, d_a);
    let (k_fxc, asym_rel) = singlet_fxc_ov_block(kern, &ref_dens, scf.mos_r(), 0, nocc, nocc, nvir);
    // Symmetric by construction up to grid noise; a large asymmetry is an
    // adapter bug, and symmetrizing would hide it.
    if asym_rel > FXC_BLOCK_ASYMMETRY_TOL {
        return Err(FerricError::General(format!(
            "run_tddft: f_xc (ia)-block asymmetry {asym_rel:e} exceeds the grid-noise \
             tolerance {FXC_BLOCK_ASYMMETRY_TOL:e} — the AO→(ia) adapter is inconsistent"
        )));
    }
    for t in targets.iter_mut() {
        **t += &k_fxc;
    }
    Ok(())
}

/// TDA: eigenpairs of A; the eigenvectors are the unit-norm X.
fn solve_tda(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), FerricError> {
    a.eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("TDDFT TDA diagonalization failed: {e}")))
}

/// Casida: `(A−B)^{1/2}(A+B)(A−B)^{1/2} Z = Ω² Z`. Returns Ω and
/// `(X+Y)_n = Ω_n^{-1/2} (A−B)^{1/2} Z_n`, which carries `|X|² − |Y|² = 1`
/// for unit `Z_n`.
fn solve_casida(
    a: &Array2<f64>,
    b_mat: &Array2<f64>,
) -> Result<(Array1<f64>, Array2<f64>), FerricError> {
    let dim = a.nrows();
    let a_plus_b = a + b_mat;
    let a_minus_b = a - b_mat;

    // (A−B) should be positive-definite for stable systems
    let (amb_vals, amb_vecs) = a_minus_b
        .eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("TDDFT: (A-B) diagonalization failed: {e}")))?;

    // (A−B)^{1/2} = U diag(√λ) U^T
    let mut amb_sqrt = Array2::<f64>::zeros((dim, dim));
    for k in 0..dim {
        let s = amb_vals[k].sqrt();
        if !s.is_finite() {
            return Err(FerricError::General(format!(
                "TDDFT: (A-B) has negative eigenvalue {:.6e} — RPA instability",
                amb_vals[k]
            )));
        }
        for i in 0..dim {
            for j in 0..dim {
                amb_sqrt[(i, j)] += s * amb_vecs[(i, k)] * amb_vecs[(j, k)];
            }
        }
    }

    // M = (A−B)^{1/2} (A+B) (A−B)^{1/2}
    let m = amb_sqrt.dot(&a_plus_b).dot(&amb_sqrt);
    let (omega_sq, z) = m
        .eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("TDDFT Casida diagonalization failed: {e}")))?;
    let omega = omega_sq.mapv(|v| if v > 0.0 { v.sqrt() } else { 0.0 });

    let mut xpy = amb_sqrt.dot(&z);
    for (n, mut col) in xpy.columns_mut().into_iter().enumerate() {
        let scale = if omega[n] > 0.0 {
            1.0 / omega[n].sqrt()
        } else {
            0.0
        };
        col.mapv_inplace(|v| v * scale);
    }
    Ok((omega, xpy))
}

/// Indices of the lowest `n_roots` positive eigenvalues, ascending.
fn lowest_positive_roots(eigenvalues: &Array1<f64>, n_roots: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..eigenvalues.len())
        .filter(|&i| eigenvalues[i] > 1e-10)
        .collect();
    indices.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));
    indices.truncate(n_roots);
    indices
}

/// Run a TDDFT (TDA or Casida) singlet calculation on a closed-shell reference.
///
/// `config.xc` names the functional the reference was converged with (`None`
/// for Hartree–Fock). It fixes both the exact-exchange fraction on the
/// `−c_HF` terms and the `(ia|f_xc|jb)` kernel block added to A and B:
///
///   * HF reference: `c_HF = 1`, no kernel — CIS (TDA) or TDHF (Casida).
///   * LDA / GGA / global hybrid: `c_HF` from the functional, kernel from
///     [`ferric_dft::lr_kernel`] on `config.grid`.
///
/// Coulomb and exchange integrals are RI (Coulomb metric) over `dfbs`.
///
/// # Memory
///
/// `dim = nocc*nvir` grows as N², so every dense matrix here is N⁴; a
/// water/cc-pVTZ Casida run holds nine co-resident dim² matrices. The plan
/// (`reserve_dense_response`, `reserve_fxc_block`, and the RI tensors in
/// `build_b_tensors`) is counted from the actual live sets in BOTH directions:
/// under-counting OOMs, over-counting refuses jobs that would have fit.
///
/// # Errors
///
/// Hard errors — never a warning plus a number — for: an unconverged or
/// open-shell reference; a functional without a complete kernel (meta-GGA,
/// VV10, range-separated hybrid); a `c_hf` override on an HF reference; an
/// empty `(ia)` space; an over-budget dense matrix; an f_xc block whose
/// asymmetry exceeds the grid-noise tolerance; an RPA instability (Casida).
pub fn run_tddft(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    scf: &ScfResult,
    config: &TddftConfig,
) -> Result<TddftResult, FerricError> {
    let nocc = closed_shell_nocc(mol, scf)?;
    // Functional gate before any integral work.
    let (c_hf, xc_def) = resolve_response_xc(config.xc.as_deref(), config.c_hf_override)?;

    let eps = scf.eps_r();
    // nmo, not nbasis: the SCF may have dropped near-linearly-dependent
    // combinations, and the virtual space is what the MOs span.
    let nmo = eps.len();
    let nvir = nmo.saturating_sub(nocc);
    let dim = nocc * nvir;
    if dim == 0 {
        return Err(FerricError::General(
            "TDDFT: no excitations possible (nocc*nvir = 0)".to_string(),
        ));
    }
    let c = scf.mos_r();
    let c_occ = c.slice(ndarray::s![.., ..nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc..nmo]).to_owned();

    let label = match config.method {
        TddftMethod::Tda => "TDDFT/TDA",
        TddftMethod::Casida => "TDDFT/Casida",
    };
    let mut plan = MemoryPlan::resolve(config.memory_budget_bytes, label);
    reserve_dense_response(&mut plan, config.method, dim);
    if xc_def.is_some() {
        reserve_fxc_block(&mut plan, config.method, dim);
    }
    let budget = config.memory_budget_bytes;
    let (b_ov, b_oo, b_vv) = build_b_tensors(obs, dfbs, &c_occ, &c_vir, &mut plan, budget)?;

    let kernel = build_kernel(mol, obs, xc_def, &config.grid)?;
    let mut a = build_a_matrix(eps, nocc, nvir, &b_ov, &b_oo, &b_vv, c_hf);
    // `amplitudes` column n: TDA → X_n; Casida → (X+Y)_n.
    let (eigenvalues, amplitudes) = match config.method {
        TddftMethod::Tda => {
            add_fxc_block(kernel.as_ref(), scf, nocc, nvir, &mut [&mut a])?;
            solve_tda(&a)?
        }
        TddftMethod::Casida => {
            let mut b_mat = build_b_matrix(nocc, nvir, &b_ov, &b_oo, &b_vv, c_hf);
            add_fxc_block(kernel.as_ref(), scf, nocc, nvir, &mut [&mut a, &mut b_mat])?;
            solve_casida(&a, &b_mat)?
        }
    };

    let indices = lowest_positive_roots(&eigenvalues, config.n_roots.min(dim));
    let (oscillator_strengths, transition_dipoles) = oscillator_strengths(
        obs,
        &c_occ,
        &c_vir,
        nocc,
        nvir,
        &eigenvalues,
        &amplitudes,
        &indices,
    )?;

    Ok(TddftResult {
        excitation_energies: indices.iter().map(|&i| eigenvalues[i]).collect(),
        oscillator_strengths,
        transition_dipoles,
        method: config.method,
        c_hf,
        fxc_included: kernel.is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_reference_is_cis_tdhf_with_no_kernel() {
        let (c_hf, def) = resolve_response_xc(None, None).unwrap();
        assert_eq!(c_hf, 1.0);
        assert!(def.is_none(), "an HF reference has no f_xc kernel");
    }

    #[test]
    fn ks_references_always_get_a_kernel() {
        // The defect this pins: a KS reference used to run with NO kernel
        // (a warning and a kernel-less number). Every accepted functional must
        // now come back with a kernel definition.
        for (name, want_c_hf) in [("LDA", 0.0), ("PBE", 0.0), ("B3LYP", 0.2)] {
            let (c_hf, def) = resolve_response_xc(Some(name), None)
                .unwrap_or_else(|e| panic!("{name} must be accepted: {e}"));
            assert!(def.is_some(), "{name}: accepted without an f_xc kernel");
            assert!(
                (c_hf - want_c_hf).abs() < 1e-12,
                "{name}: c_HF {c_hf} != {want_c_hf}"
            );
        }
    }

    #[test]
    fn meta_gga_is_refused_with_an_error_not_a_number() {
        let err = resolve_response_xc(Some("SCAN"), None)
            .expect_err("meta-GGA has no tau f_xc kernel; it must be refused");
        let msg = err.to_string();
        assert!(msg.contains("meta-GGA"), "unhelpful refusal: {msg}");
        assert!(
            msg.contains("run_tddft"),
            "refusal must name the driver: {msg}"
        );
    }

    #[test]
    fn range_separated_and_vv10_functionals_are_refused() {
        assert!(resolve_response_xc(Some("wB97X-V"), None).is_err());
        assert!(resolve_response_xc(Some("HYB_GGA_XC_CAM_B3LYP"), None).is_err());
    }

    #[test]
    fn c_hf_override_on_an_hf_reference_is_refused() {
        assert!(resolve_response_xc(None, Some(0.2)).is_err());
        // c_hf = 1 on an HF reference is just CIS/TDHF spelled out.
        assert_eq!(resolve_response_xc(None, Some(1.0)).unwrap().0, 1.0);
        // With a functional, an override is an explicit model change.
        assert_eq!(
            resolve_response_xc(Some("PBE"), Some(0.25)).unwrap().0,
            0.25
        );
        assert!(resolve_response_xc(Some("PBE"), Some(f64::NAN)).is_err());
    }
}
