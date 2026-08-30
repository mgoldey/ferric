//! Linear-response TDDFT for closed-shell references.
//!
//! Implements the Tamm–Dancoff approximation (TDA, equivalent to CIS for HF)
//! and the full Casida equations for singlet excitations:
//!
//! TDA:    A X = ω X
//! Casida: (A−B)^{1/2} (A+B) (A−B)^{1/2} Z = Ω² Z
//!
//! where A_{ia,jb} = δ_{ij}δ_{ab}(ε_a − ε_i) + 2(ia|jb) − c_HF(ij|ab) + (ia|f_xc|jb)
//!       B_{ia,jb} = 2(ia|bj) − c_HF(ib|aj) + (ia|f_xc|bj)

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex;
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
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
}

impl Default for TddftConfig {
    fn default() -> Self {
        Self { n_roots: 3, method: TddftMethod::Tda, memory_budget_bytes: None }
    }
}

/// Result of a TDDFT calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct TddftResult {
    pub excitation_energies: Vec<f64>,
    pub oscillator_strengths: Vec<f64>,
    pub transition_dipoles: Vec<[f64; 3]>,
    pub method: TddftMethod,
}

impl std::fmt::Display for TddftResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ha_to_ev = 27.211_386_245_988;
        writeln!(f, "TDDFT {:?} — {} roots:", self.method, self.excitation_energies.len())?;
        for (i, (&e, &osc)) in self
            .excitation_energies
            .iter()
            .zip(&self.oscillator_strengths)
            .enumerate()
        {
            writeln!(f, "  {}: {:.6} Ha  ({:.4} eV)  f = {:.6}", i + 1, e, e * ha_to_ev, osc)?;
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
    plan.reserve("B(P|ia) [b_ov]", naux.saturating_mul(nocc * nvir), Lifetime::Resident);
    plan.reserve("B(P|ij) [b_oo]", naux.saturating_mul(nocc * nocc), Lifetime::Resident);
    plan.reserve("B(P|ab) [b_vv]", naux.saturating_mul(nvir * nvir), Lifetime::Resident);
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
/// Peak is stage 1. Declared as `a` resident plus the two *alternatives* as
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

/// Build the TDA A matrix in the (ia, jb) compound-index space.
///
/// A_{ia,jb} = δ_{ij}δ_{ab}(ε_a − ε_i) + 2(ia|jb) − c_HF(ij|ab) + (ia|f_xc|jb)
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

    // TODO: add (ia|f_xc|jb) contribution from ferric-dft's fxc.rs kernel.
    // For pure HF references (c_hf = 1.0) this is zero and the result is CIS.
    // For DFT references, the missing f_xc term means excitation energies will
    // be approximate (missing the XC kernel response).

    a
}

/// Build the TDDFT B matrix in the (ia, jb) compound-index space.
///
/// B_{ia,jb} = 2(ia|bj) − c_HF(ib|aj)
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

    // TODO: add (ia|f_xc|bj) contribution (same caveat as A matrix).

    b
}

/// Compute transition dipole moments and oscillator strengths.
///
/// f_n = (2/3) * ω_n * |μ_n|² where μ_n = Σ_{ia} X_{ia,n} * μ_{ia}^(MO)
fn oscillator_strengths(
    _mol: &Molecule,
    obs: &PreparedBasis,
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
    nocc: usize,
    nvir: usize,
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
    n_roots: usize,
) -> Result<(Vec<f64>, Vec<[f64; 3]>), FerricError> {
    let origin = [0.0, 0.0, 0.0];
    let mu_ao = oneelectron::dipole(obs, origin)?;

    // Transform dipole integrals to MO basis: μ^(MO)_{ia} = C_occ^T μ^(AO) C_vir
    let mut mu_ov = Vec::with_capacity(3);
    for xyz in 0..3 {
        let tmp = c_occ.t().dot(&mu_ao[xyz]).dot(c_vir); // (nocc, nvir)
        mu_ov.push(tmp);
    }

    let mut osc = Vec::with_capacity(n_roots);
    let mut tdips = Vec::with_capacity(n_roots);

    for n in 0..n_roots {
        if n >= eigenvalues.len() {
            break;
        }
        let omega_n = eigenvalues[n];
        if omega_n <= 0.0 {
            osc.push(0.0);
            tdips.push([0.0; 3]);
            continue;
        }

        let mut tdip = [0.0_f64; 3];
        for xyz in 0..3 {
            let mut val = 0.0;
            for i in 0..nocc {
                for a in 0..nvir {
                    let ia = i * nvir + a;
                    val += eigenvectors[(ia, n)] * mu_ov[xyz][(i, a)];
                }
            }
            tdip[xyz] = val;
        }

        let mu_sq = tdip[0] * tdip[0] + tdip[1] * tdip[1] + tdip[2] * tdip[2];
        osc.push((2.0 / 3.0) * omega_n * mu_sq);
        tdips.push(tdip);
    }

    Ok((osc, tdips))
}

/// Run a TDDFT calculation on a closed-shell reference.
///
/// `c_hf` is the fraction of exact (Hartree–Fock) exchange in the functional.
/// For pure HF, `c_hf = 1.0` gives CIS (TDA) or TDHF (Casida).
/// For pure DFT (LDA/GGA), `c_hf = 0.0`.
/// For hybrids (B3LYP, PBE0), the appropriate fraction (0.20, 0.25, etc.).
pub fn run_tddft(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &TddftConfig,
    c_hf: f64,
) -> Result<TddftResult, FerricError> {
    if !rhf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        });
    }

    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc = nelec / 2;
    let nvir = nbas - nocc;
    let dim = nocc * nvir;

    if dim == 0 {
        return Err(FerricError::General(
            "TDDFT: no excitations possible (nocc*nvir = 0)".to_string(),
        ));
    }

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., ..nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc..]).to_owned();

    // ── Memory pre-flight ───────────────────────────────────────────────────
    //
    // WHY this gate exists. `dim = nocc*nvir` grows as N², so every dense
    // matrix here is N⁴ — the steepest term in the whole crate — and until this
    // guard landed the crate had NO budget machinery whatsoever: not one
    // `check_alloc`, not one `resolve_budget`, and no budget field on
    // `TddftConfig`, so a caller who WANTED to constrain it could not. A
    // water/cc-pVTZ Casida run holds nine co-resident dim² matrices; at
    // dim = 20k that is 9 × 3.2 GB with nothing between it and the OOM killer.
    //
    // The reservations below are counted from the actual live sets, not from a
    // round number, in BOTH directions: under-counting OOMs, and over-counting
    // refuses jobs that would have fit (this repo has been bitten by an
    // over-estimating guard too — see `ample_budget_still_runs_to_completion`).
    let mut plan = MemoryPlan::resolve(
        config.memory_budget_bytes,
        match config.method {
            TddftMethod::Tda => "TDDFT/TDA",
            TddftMethod::Casida => "TDDFT/Casida",
        },
    );
    reserve_dense_response(&mut plan, config.method, dim);

    let (b_ov, b_oo, b_vv) =
        build_b_tensors(obs, dfbs, &c_occ, &c_vir, &mut plan, config.memory_budget_bytes)?;

    let n_roots = config.n_roots.min(dim);

    let (eigenvalues, eigenvectors) = match config.method {
        TddftMethod::Tda => {
            let a = build_a_matrix(eps, nocc, nvir, &b_ov, &b_oo, &b_vv, c_hf);
            let (vals, vecs) = a.eigh(UPLO::Lower).map_err(|e| {
                FerricError::General(format!("TDDFT TDA diagonalization failed: {e}"))
            })?;
            (vals, vecs)
        }
        TddftMethod::Casida => {
            let a = build_a_matrix(eps, nocc, nvir, &b_ov, &b_oo, &b_vv, c_hf);
            let b_mat = build_b_matrix(nocc, nvir, &b_ov, &b_oo, &b_vv, c_hf);

            // Hermitian eigenvalue problem: (A−B)^{1/2}(A+B)(A−B)^{1/2} Z = Ω² Z
            let a_plus_b = &a + &b_mat;
            let a_minus_b = &a - &b_mat;

            // (A−B) should be positive-definite for stable systems
            let (amb_vals, amb_vecs) = a_minus_b.eigh(UPLO::Lower).map_err(|e| {
                FerricError::General(format!("TDDFT: (A-B) diagonalization failed: {e}"))
            })?;

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

            let (omega_sq, z) = m.eigh(UPLO::Lower).map_err(|e| {
                FerricError::General(format!("TDDFT Casida diagonalization failed: {e}"))
            })?;

            // ω = √(Ω²), recover X from Z: X ∝ (A−B)^{1/2} Z / √ω
            let omega = omega_sq.mapv(|v| {
                if v > 0.0 { v.sqrt() } else { 0.0 }
            });

            // The eigenvectors of the original problem: X ∝ (A−B)^{1/2} Z
            let x = amb_sqrt.dot(&z);

            (omega, x)
        }
    };

    let (osc, tdips) = oscillator_strengths(
        mol, obs, &c_occ, &c_vir, nocc, nvir, &eigenvalues, &eigenvectors, n_roots,
    )?;

    // Take the lowest n_roots positive excitations
    let mut indices: Vec<usize> = (0..eigenvalues.len())
        .filter(|&i| eigenvalues[i] > 1e-10)
        .collect();
    indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());
    indices.truncate(n_roots);

    let excitation_energies: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();

    // Re-map oscillator strengths to sorted order
    let mut sorted_osc = Vec::with_capacity(n_roots);
    let mut sorted_tdips = Vec::with_capacity(n_roots);
    for &idx in &indices {
        if idx < osc.len() {
            sorted_osc.push(osc[idx]);
            sorted_tdips.push(tdips[idx]);
        } else {
            sorted_osc.push(0.0);
            sorted_tdips.push([0.0; 3]);
        }
    }

    Ok(TddftResult {
        excitation_energies,
        oscillator_strengths: sorted_osc,
        transition_dipoles: sorted_tdips,
        method: config.method,
    })
}
