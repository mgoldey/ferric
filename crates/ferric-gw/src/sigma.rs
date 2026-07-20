//! G0W0 correlation self-energy via imaginary-axis quadrature + Padé AC.
//!
//! Working formula (closed-shell, RHF reference, MO basis, PDEP-W
//! representation):
//!
//!   Σ_c(m, z) = − (1/π) Σ_n Σ_α  M²_{mn,α}
//!                 ∫₀^∞ dω' w_α(iω')
//!                  ·  (z − ε_n) / ((z − ε_n)² + ω'²)
//!
//! where w_α(iω') = 1/λ_α(iω') − 1. The integrand is even in ω', so the
//! ∫₋∞^∞ → 2·∫₀^∞ factor of 2 has been absorbed; the prefactor is then
//! 1/(2π) · 2 = 1/π.
//!
//! For z = iω (imaginary axis), the integrand is purely imaginary in the
//! conventional split, and the resulting Σ_c(m, iω) is real-imaginary
//! — for our purposes we evaluate at complex z and let num-complex do
//! the bookkeeping.
//!
//! The quadrature {(ω_k, w_k)} is reused verbatim from PDEP
//! (`pdep.quad_freqs`, `pdep.quad_weights`). This guarantees that
//! Σ_c at z=0 is consistent with the inverse-dielectric values used in
//! COHSEX.
//!
//! After evaluating Σ_c(m, iω_k) at the PDEP quadrature nodes, we fit a
//! Padé continued-fraction model (`crate::pade::PadeCF`) on the sampled
//! values and evaluate at real ω = ε^QP_m via Newton iteration.

use crate::cohsex::{project_b_into_pdep, sigma_x_diag};
use crate::mo_b::MoB;
use crate::pade::PadeCF;
use crate::w_pdep;
use crate::{GwConfig, GwResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_rpa::PdepRpaResult;
use ferric_scf::ScfResult;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};

/// Sub-sample `npts` node indices from an `nw`-point evaluation grid with a
/// *decreasing* step size — a direct port of PySCF `gw_ac._get_ac_idx`
/// (pyscf/gw/utils/ac_grid.py). `steps = linspace(1, step_ratio, npts)`,
/// normalized, cumulatively summed × nw, offset so the first index is
/// `idx_start`, then rounded. Denser near ω=0 (where Σc varies fastest) and
/// sparser at large ω — the node distribution that makes the Thiele–Padé fit
/// stable for the long ε_HOMO→ε_F extrapolation that a small-gap (KS/PBE)
/// reference requires. Returns strictly-increasing, in-range indices.
pub(crate) fn pade_node_indices(nw: usize, npts: usize, step_ratio: f64, idx_start: usize) -> Vec<usize> {
    // PySCF requires nw > npts; if the caller passes a grid too small, fall back
    // to using every available node (still ascending) rather than erroring.
    if nw <= npts {
        return (0..nw).collect();
    }
    let n = npts as f64;
    // steps[k] = 1 + (step_ratio − 1)·k/(npts−1)   (== linspace(1, step_ratio, npts))
    let raw: Vec<f64> = (0..npts)
        .map(|k| 1.0 + (step_ratio - 1.0) * (k as f64) / (n - 1.0))
        .collect();
    let s: f64 = raw.iter().sum();
    // cumsum(steps/sum · nw), then shift so the first entry lands on idx_start.
    let mut cum = 0.0;
    let mut out: Vec<usize> = Vec::with_capacity(npts);
    let mut first = None;
    for r in &raw {
        cum += r / s * (nw as f64);
        if first.is_none() {
            first = Some(cum);
        }
        let shifted = cum + idx_start as f64 - first.unwrap();
        out.push(shifted.round() as usize);
    }
    // Clamp into range and dedup (rounding can collide at the dense end).
    out.iter_mut().for_each(|i| *i = (*i).min(nw - 1));
    out.dedup();
    out
}
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rayon::prelude::*;

/// Emit one stderr warning (per call) if any per-state Newton QP solve failed
/// to converge. Pure observability: does not affect returned energies. `label`
/// identifies the calling method (e.g. "G0W0", "evGW0") for the message.
pub(crate) fn warn_if_unconverged(label: &str, mo_indices: &[usize], qp_converged: &[bool]) {
    let n_bad = qp_converged.iter().filter(|&&c| !c).count();
    if n_bad == 0 {
        return;
    }
    let bad_mos: Vec<usize> = mo_indices
        .iter()
        .zip(qp_converged.iter())
        .filter(|(_, &c)| !c)
        .map(|(&m, _)| m)
        .collect();
    eprintln!(
        "ferric-gw WARNING: {label} QP Newton solve did not converge for {n_bad}/{} \
         state(s) (MO indices {bad_mos:?}); returned ε_qp for these states is the last \
         Newton iterate, not a converged root.",
        mo_indices.len(),
    );
}

/// Fermi level (mid-gap) of the active-space spectrum: midpoint between the
/// active HOMO and active LUMO. Used to center the analytic-continuation
/// support line `z = ef + iω`, matching PySCF gw_ac. For an all-occupied or
/// all-virtual active block (degenerate edge cases) we fall back to the band
/// edge so `ef` stays finite.
pub(crate) fn fermi_level(eps_act: &[f64], n_occ_act: usize) -> f64 {
    let n = eps_act.len();
    if n == 0 {
        return 0.0;
    }
    if n_occ_act == 0 {
        return eps_act[0];
    }
    if n_occ_act >= n {
        return eps_act[n - 1];
    }
    0.5 * (eps_act[n_occ_act - 1] + eps_act[n_occ_act])
}

/// Evaluate Σ_c(m, z) at a single (possibly complex) z given the projected
/// matrix elements M[(α,m,n)], inverse-dielectric weights w_α(iω_k), and
/// quadrature data. `n_occ_act` is used only for completeness (sum runs
/// over all n).
pub(crate) fn sigma_c_at_z(
    m_idx: usize,
    z: Complex64,
    m_proj: &ndarray::Array3<f64>,
    inv_diel_freq: &[Array2<f64>], // length N_quad, each (M, M): W̃_d(iω_k) = ε̃⁻¹ − I
    quad_weights: &[f64],
    quad_freqs: &[f64],
    eps_act: &[f64],
) -> Complex64 {
    let n_quad = quad_freqs.len();
    let n_act = eps_act.len();

    // The screened matrix element W_{mn}(iω_k) = Mᵀ_{mn} · W̃_d(iω_k) · M_{mn}
    // (FULL inverse-dielectric matrix in the PDEP basis; the earlier diagonal-only
    // form was wrong — the scalar eigenvalues λ_α(iω) live in a per-ω rotated
    // eigenbasis, inconsistent with the static B̃ projection M).
    //
    // For a fixed m_idx, let V = M[:, m_idx, :] be the (m_modes × n_act) matrix of
    // projected elements. W̃_d(iω_k) depends only on k and V only on n, so the
    // per-(n,k) quadratic form vᵀ·W̃_d·v is one BLAS3 GEMM per frequency:
    //   WV_k = W̃_d(iω_k) · V            (m_modes × n_act)
    //   W_{n,k} = Σ_α V[α,n] · WV_k[α,n]  (column-wise dot)
    // replacing n_act·n_quad scalar O(m_modes²) forms with n_quad GEMMs. The
    // summation order differs from the old scalar loop (BLAS vs nested-scalar), so
    // results agree to machine precision, not bit-identically.
    let v = m_proj.index_axis(ndarray::Axis(1), m_idx); // (m_modes, n_act)
    let v = v.as_standard_layout(); // ensure contiguous for GEMM

    // w_nk[(n, k)] = screened element W_{mn}(iω_k), real.
    let mut w_nk = Array2::<f64>::zeros((n_act, n_quad));
    for k in 0..n_quad {
        let wv = with_blas_threads(opt_in_blas_threads(), || inv_diel_freq[k].dot(&v)); // (m_modes, n_act)
        // Column-wise dot: W_{n,k} = Σ_α V[α,n]·WV[α,n].
        let col = (&v.to_owned() * &wv).sum_axis(ndarray::Axis(0)); // (n_act,)
        w_nk.column_mut(k).assign(&col);
    }

    let mut sigma = Complex64::new(0.0, 0.0);
    for n_idx in 0..n_act {
        let eps_n = eps_act[n_idx];
        let diff = z - Complex64::new(eps_n, 0.0);
        let mut inner = Complex64::new(0.0, 0.0);
        for k in 0..n_quad {
            let omk = quad_freqs[k];
            let den = diff * diff + Complex64::new(omk * omk, 0.0);
            inner += Complex64::new(quad_weights[k] * w_nk[(n_idx, k)], 0.0) / den;
        }
        sigma += diff * inner;
    }
    -Complex64::new(1.0 / std::f64::consts::PI, 0.0) * sigma
}

/// Per-MO QP solve given a fixed (m_proj, w_α(iω), quadrature, propagator-ε)
/// snapshot. Returns (ε_qp, Σ_c at ε_qp, Z, was Newton-converged).
///
/// "Converged" means the Newton iteration hit its `|step| < 1e-7` stopping
/// criterion within the 30-iteration budget. If the loop instead exits because
/// `|f'| < 1e-3` (near a Σ_c pole) or because it ran out of iterations without
/// the step shrinking below tolerance, the returned `ε_qp` is the *last
/// iterate*, not a converged root — this flag is what lets callers detect that.
pub(crate) fn solve_qp_for_mo(
    m_loc: usize,
    eps_m_mf: f64,
    m_proj: &ndarray::Array3<f64>,
    inv_diel_freq: &[Array2<f64>],
    quad_weights: &[f64],
    quad_freqs: &[f64],
    eps_prop: &[f64],
    pade_npts: usize,
    newton_damp: f64,
    ef: f64,
    // Static (energy-independent) self-energy shift Σ_x − v_xc, added INSIDE the
    // QP self-consistency. Zero for HF references (Σ_x is already in ε_mf); for a
    // KS reference it is Σ_x − v_xc and is large (~−7 eV), moving the QP root by
    // several eV — so Σ_c must be evaluated at the shifted energy, not post-hoc.
    static_shift: f64,
) -> Result<(f64, f64, f64, bool), FerricError> {
    // Padé analytic-continuation nodes, matching PySCF gw_ac.
    //
    // Σc(z) is analytic (closed-form ω-integral over the PDEP quadrature, see
    // sigma_c_at_z), so the AC nodes are INDEPENDENT of the quad_freqs used for
    // that integral — they are just where we sample Σc(iω') to fit the Padé.
    // PySCF samples on `ef + i·[0, scaled-Legendre freqs]` sub-sampled with a
    // decreasing step (_get_ac_idx: npts=18, step_ratio=2/3), NOT on a fixed
    // log grid. The old bespoke log grid (0.05 → 3·ω_max, non-ef-aware) gave the
    // right answer for @HF (ε_HOMO≈ε_F ⇒ short extrapolation) but swung @PBE by
    // eV per molecule (small gap ⇒ ε_HOMO far from ε_F ⇒ long extrapolation on
    // the wrong nodes). See the gw-ks-sigma-c-offset diagnosis.
    let npts = if pade_npts == 0 { 18 } else { pade_npts };
    const STEP_RATIO: f64 = 2.0 / 3.0;
    // Dense scaled-Legendre evaluation grid (u0=0.5 matches PySCF
    // _get_scaled_legendre_roots and ferric's own GW quadrature), prepended with
    // ω=0, then sub-sampled. Use a grid comfortably larger than npts so the
    // decreasing-step selection is meaningful (PySCF's default nw is 100).
    let nw_eval = 100usize;
    let (leg_freqs, _leg_wts) = ferric_rpa::quadrature::gauss_legendre_nodes(nw_eval, 0.5);
    let mut eval_grid: Vec<f64> = Vec::with_capacity(nw_eval + 1);
    eval_grid.push(0.0);
    eval_grid.extend_from_slice(&leg_freqs);
    // idx_start = 1 (PySCF default): skip the ω=0 node as the first fit point but
    // keep it available; the decreasing step still lands nodes near it.
    let idx = pade_node_indices(eval_grid.len(), npts, STEP_RATIO, 1);
    // Sample Σ_c on the Fermi-shifted imaginary axis z = ef + iω'.
    let z_nodes: Vec<Complex64> = idx
        .iter()
        .map(|&i| Complex64::new(ef, eval_grid[i]))
        .collect();
    let f_vals: Vec<Complex64> = z_nodes
        .iter()
        .map(|&z| {
            sigma_c_at_z(
                m_loc, z, m_proj, inv_diel_freq, quad_weights, quad_freqs, eps_prop,
            )
        })
        .collect();
    let pade = PadeCF::fit(z_nodes, &f_vals)?;
    let h = 0.05;
    let sc_at_ref = pade.eval(Complex64::new(eps_m_mf, 0.0)).re;
    let dsig = pade.deriv_real(eps_m_mf, h).re;
    let z_renorm = (1.0 / (1.0 - dsig)).clamp(0.0, 1.5);
    // Linearized: ε_qp = ε_mf + Z·(Σc(ε_mf) + static_shift). The static shift
    // (Σx−vxc, KS only) enters the QP equation alongside Σc (PySCF gw_ac form
    // zn·(sigmaR + vk − v_mf)); it is large for KS and moves the root by eV, so
    // the self-consistent Σc below is evaluated at the correctly-shifted energy.
    let eps_qp_lin = eps_m_mf + z_renorm * (sc_at_ref + static_shift);
    let mut eps_curr = eps_qp_lin;
    let damp = newton_damp.clamp(0.1, 1.0);
    // The doc always promised this flag; it was computed and then dropped.
    // false = the 30-step cap was exhausted or Σc'(ε) ≈ 1 (singular Newton
    // slope, e.g. a QP root near a pole of the Padé model) — the returned
    // ε_qp is best-effort, not a solved fixed point.
    let mut newton_converged = false;
    for _ in 0..30 {
        let sc = pade.eval(Complex64::new(eps_curr, 0.0)).re;
        // QP residual: ε − ε_mf − static_shift − Σc(ε) = 0.
        let f = eps_curr - eps_m_mf - static_shift - sc;
        let dsc = pade.deriv_real(eps_curr, h).re;
        let fprime = 1.0 - dsc;
        if fprime.abs() < 1e-3 {
            break;
        }
        let step = -damp * f / fprime;
        eps_curr += step;
        if step.abs() < 1e-7 {
            newton_converged = true;
            break;
        }
    }
    let sc_final = pade.eval(Complex64::new(eps_curr, 0.0)).re;
    Ok((eps_curr, sc_final, z_renorm, newton_converged))
}

/// Run G0W0. `vxc_diag`, if given (KS reference), is the absolute-MO-indexed
/// diagonal v_xc; the QP equation then includes Σ_x − v_xc inside the
/// self-consistency. `None` ⇒ HF reference (no shift, unchanged behavior).
pub fn run_g0w0(
    mol: &Molecule,
    rhf: &ScfResult,
    mo_b: &MoB,
    v_dressed: &Array2<f64>,
    pdep: PdepRpaResult,
    qp_range: std::ops::Range<usize>,
    gw_cfg: &GwConfig,
    vxc_diag: Option<&Array1<f64>>,
) -> Result<GwResult, FerricError> {
    let _ = (mol, rhf);
    let _n_act = mo_b.n_act;
    let first_act = mo_b.first_act;
    let m_modes = v_dressed.ncols();
    if pdep.eigenvalues_freq.ncols() != m_modes {
        return Err(FerricError::General(
            "pdep eigenvalues_freq mode count does not match dressed eigenpotentials".into(),
        ));
    }

    // Σ_x for all active MOs.
    let sigma_x_all = sigma_x_diag(mo_b);

    // Project B̃ → M[(α,m,n)] (shape M × n_act × n_act).
    let m_proj = project_b_into_pdep(mo_b, v_dressed);

    // Full per-frequency inverse-dielectric matrices W̃_d(iω_k) in the PDEP basis.
    let inv_diel_freq = pdep.inv_dielectric_freq.as_ref().ok_or_else(|| {
        FerricError::General(
            "PDEP result missing inv_dielectric_freq (GW requires the dense χ₀ path)".into(),
        )
    })?;

    let quad_freqs = pdep.quad_freqs.clone();
    let quad_weights = pdep.quad_weights.clone();

    // Fermi level: mid-gap of the active-space mean-field spectrum.
    let ef = fermi_level(&mo_b.eps_act, mo_b.n_occ_act);

    // For each MO in qp_range, sample Σ_c on the imaginary axis at the PDEP
    // quadrature nodes, fit Padé, solve QP via Newton.
    let mo_indices: Vec<usize> = qp_range.clone().collect();
    let mut eps_qp = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_out = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_out = Array1::<f64>::zeros(mo_indices.len());
    let mut z_out = Array1::<f64>::ones(mo_indices.len());
    let mut qp_converged = vec![true; mo_indices.len()];

    // Each MO's QP solve is independent (scalar math only — no BLAS inside),
    // so parallelize over the QP index; per-state summation order is unchanged.
    let qp_rows = mo_indices
        .par_iter()
        .map(|&mo_abs| {
            if mo_abs < first_act {
                return Err(FerricError::General(format!(
                    "qp_mos index {mo_abs} is in the frozen-core block"
                )));
            }
            let m_loc = mo_abs - first_act;
            let eps_m = mo_b.eps_act[m_loc];

            // KS reference: shift = Σ_x − v_xc (inside the QP self-consistency).
            let shift = vxc_diag.map(|v| sigma_x_all[m_loc] - v[mo_abs]).unwrap_or(0.0);
            let (eps_qp_m, sc_final, z_renorm, converged) = solve_qp_for_mo(
                m_loc,
                eps_m,
                &m_proj,
                inv_diel_freq,
                &quad_weights,
                &quad_freqs,
                &mo_b.eps_act,
                gw_cfg.pade_npts,
                gw_cfg.qp_newton_damp,
                ef,
                shift,
            )?;
            Ok((eps_m, sigma_x_all[m_loc], eps_qp_m, sc_final, z_renorm, converged))
        })
        .collect::<Result<Vec<_>, FerricError>>()?;
    for (idx, &(eps_m, sx, eps_qp_m, sc_final, z_renorm, converged)) in qp_rows.iter().enumerate() {
        eps_mf[idx] = eps_m;
        sx_out[idx] = sx;
        eps_qp[idx] = eps_qp_m;
        sc_out[idx] = sc_final;
        z_out[idx] = z_renorm;
        qp_converged[idx] = converged;
    }
    warn_if_unconverged("G0W0", &mo_indices, &qp_converged);

    Ok(GwResult {
        mo_indices,
        eps_mf,
        eps_qp,
        sigma_x: sx_out,
        sigma_c: sc_out,
        z_factor: z_out,
        qp_converged,
        n_ev_iter: 0,
        outer_converged: true,
        pdep,
    })
}

/// evGW₀: eigenvalue self-consistency on G only.
///
/// Iterate: at each step k, replace ε_n in the propagator denominator by
/// the current QP estimate. W stays frozen at iteration 0 (so PDEP is not
/// re-run). Converges in a few iterations for closed-shell molecules.
///
/// `vxc_diag`, if given (KS reference), is the absolute-MO-indexed diagonal
/// v_xc — same convention as `run_g0w0`. The static shift Σ_x − v_xc for
/// each QP state is computed ONCE, before the outer eigenvalue
/// self-consistency loop starts, from the frozen mean-field `sigma_x_all`/
/// `vxc_diag` snapshot, and is held FIXED across every outer iteration: it
/// does not get recomputed as ε^QP updates. This matches `run_g0w0`'s
/// treatment of the same quantity — `vxc_diag` is a property of the
/// *starting* KS orbitals, not something that evolves with QP
/// self-consistency, so re-deriving it per outer iteration would be wrong
/// (and would just reproduce the same fixed value at extra cost, since it
/// does not depend on `eps_prop`). `None` ⇒ HF reference (no shift,
/// unchanged behavior).
pub fn run_evgw0(
    mol: &Molecule,
    rhf: &ScfResult,
    mo_b: &MoB,
    v_dressed: &Array2<f64>,
    pdep: PdepRpaResult,
    qp_range: std::ops::Range<usize>,
    gw_cfg: &GwConfig,
    vxc_diag: Option<&Array1<f64>>,
) -> Result<GwResult, FerricError> {
    let _ = (mol, rhf);
    let first_act = mo_b.first_act;
    let m_modes = v_dressed.ncols();
    if pdep.eigenvalues_freq.ncols() != m_modes {
        return Err(FerricError::General(
            "pdep eigenvalues_freq mode count does not match dressed eigenpotentials".into(),
        ));
    }
    let sigma_x_all = sigma_x_diag(mo_b);
    let m_proj = project_b_into_pdep(mo_b, v_dressed);
    let inv_diel_freq = pdep.inv_dielectric_freq.as_ref().ok_or_else(|| {
        FerricError::General(
            "PDEP result missing inv_dielectric_freq (GW requires the dense χ₀ path)".into(),
        )
    })?;

    let quad_freqs = pdep.quad_freqs.clone();
    let quad_weights = pdep.quad_weights.clone();

    let ef = fermi_level(&mo_b.eps_act, mo_b.n_occ_act);

    let mo_indices: Vec<usize> = qp_range.clone().collect();
    let mut eps_qp = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_out = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_out = Array1::<f64>::zeros(mo_indices.len());
    let mut z_out = Array1::<f64>::ones(mo_indices.len());
    let mut qp_converged = vec![true; mo_indices.len()];
    // KS static shift Σ_x − v_xc, one value per QP MO. Computed ONCE here
    // (outside/before the outer self-consistency loop below) from the fixed
    // mean-field sigma_x_all/vxc_diag — see the doc comment on this function
    // for why it must NOT be recomputed per outer iteration.
    let static_shifts: Vec<f64> = mo_indices
        .iter()
        .map(|&mo_abs| {
            let m_loc = mo_abs - first_act;
            vxc_diag.map(|v| sigma_x_all[m_loc] - v[mo_abs]).unwrap_or(0.0)
        })
        .collect();
    for (idx, &mo_abs) in mo_indices.iter().enumerate() {
        let m_loc = mo_abs - first_act;
        eps_mf[idx] = mo_b.eps_act[m_loc];
        sx_out[idx] = sigma_x_all[m_loc];
        eps_qp[idx] = eps_mf[idx];
    }

    // Outer loop.
    let mut eps_prop = mo_b.eps_act.clone();
    let mut iter_done = 0usize;
    let mut outer_converged = gw_cfg.max_ev_iter == 0;
    for it in 0..gw_cfg.max_ev_iter {
        let mut max_dev = 0.0_f64;
        // First update eps_prop using the *previous* QP energies.
        for (idx, &mo_abs) in mo_indices.iter().enumerate() {
            let m_loc = mo_abs - first_act;
            eps_prop[m_loc] = eps_qp[idx];
        }
        // eps_prop is a frozen snapshot for this iteration (Jacobi-style
        // update), so each QP state's solve is independent — parallelize.
        let qp_new: Vec<(f64, f64, f64, bool)> = mo_indices
            .par_iter()
            .zip(static_shifts.par_iter())
            .map(|(&mo_abs, &shift)| {
                let m_loc = mo_abs - first_act;
                let eps_m_mf = mo_b.eps_act[m_loc];
                solve_qp_for_mo(
                    m_loc,
                    eps_m_mf,
                    &m_proj,
                    inv_diel_freq,
                    &quad_weights,
                    &quad_freqs,
                    &eps_prop,
                    gw_cfg.pade_npts,
                    gw_cfg.qp_newton_damp,
                    ef,
                    shift,
                )
            })
            .collect::<Result<Vec<_>, FerricError>>()?;
        for (idx, &(eps_new, sc_new, z_new, converged)) in qp_new.iter().enumerate() {
            max_dev = max_dev.max((eps_new - eps_qp[idx]).abs());
            eps_qp[idx] = eps_new;
            sc_out[idx] = sc_new;
            z_out[idx] = z_new;
            qp_converged[idx] = converged;
        }
        iter_done = it + 1;
        // Live per-iteration progress (see RhfConfig.verbose's doc for the
        // full rationale). STDOUT, opt-in via `gw_cfg.verbose`.
        if gw_cfg.verbose {
            println!(
                "evGW0 iter={it:4}  max|d_eps_qp|={max_dev:.3e}  ev_conv_thresh={:.3e}",
                gw_cfg.ev_conv_thresh
            );
        }
        if max_dev < gw_cfg.ev_conv_thresh {
            outer_converged = true;
            break;
        }
    }
    warn_if_unconverged("evGW0", &mo_indices, &qp_converged);
    if !outer_converged {
        eprintln!(
            "ferric-gw WARNING: evGW0 outer loop did not converge within max_ev_iter={} \
             (ev_conv_thresh={:.3e}); returned energies are the last iterate, not \
             self-consistent.",
            gw_cfg.max_ev_iter, gw_cfg.ev_conv_thresh
        );
    }

    Ok(GwResult {
        mo_indices,
        eps_mf,
        eps_qp,
        sigma_x: sx_out,
        sigma_c: sc_out,
        z_factor: z_out,
        qp_converged,
        n_ev_iter: iter_done,
        outer_converged,
        pdep,
    })
}

/// evGW: eigenvalue self-consistency on G *and* W.
///
/// Outer loop re-runs PDEP-RPA with QP-shifted ε's on every iteration.
/// Inside each PDEP rebuild we override the occupied/virtual orbital
/// energies in the χ₀ denominators using the current QP estimates for the
/// QP block and the mean-field ε's elsewhere.
///
/// **Spike note**: the cleanest way to override the ε's used by PDEP is
/// to shift the RHF eigenvalue vector before re-calling `run_pdep_rpa`.
/// We mutate a `ScfResult` clone so it presents the QP-shifted ε's. All
/// other RHF data (orbitals, density, energy) stays.
///
/// `vxc_diag`, if given (KS reference), is the absolute-MO-indexed diagonal
/// v_xc — same convention as `run_g0w0`/`run_evgw0`. The static shift
/// Σ_x − v_xc for each QP state is computed ONCE, before the outer G+W
/// self-consistency loop, from the frozen mean-field `sigma_x_all`/
/// `vxc_diag` snapshot, and held FIXED across every outer iteration (which
/// here also rebuilds W via a fresh PDEP-RPA run) — it is never recomputed
/// from the QP-shifted propagator. `vxc_diag` is a property of the
/// *starting* KS orbitals, not of the evolving QP/W self-consistency, so
/// this matches `run_g0w0`/`run_evgw0`'s treatment. `None` ⇒ HF reference
/// (no shift, unchanged behavior).
pub fn run_evgw(
    mol: &Molecule,
    obs: &ferric_integrals::basis_bridge::PreparedBasis,
    dfbs: &ferric_integrals::basis_bridge::PreparedBasis,
    op: ferric_integrals::operator::Operator,
    rhf: &ScfResult,
    pdep_cfg: &ferric_rpa::config::PdepRpaConfig,
    mo_b: &MoB,
    pdep0: PdepRpaResult,
    qp_range: std::ops::Range<usize>,
    gw_cfg: &GwConfig,
    vxc_diag: Option<&Array1<f64>>,
) -> Result<GwResult, FerricError> {
    // We rebuild PDEP from a *modified* ScfResult that has QP-shifted
    // eigenvalues for the QP block. mo_b.eps_act stays as the original
    // MF eigenvalues — only the PDEP χ₀ denominators see the shifted ε's.
    let mut shifted_rhf = rhf.clone();
    let mut current_pdep = pdep0;
    let mut current_v_dressed =
        w_pdep::redress_eigenpotentials(&mo_b.v_inv_sqrt, &current_pdep.eigenpotentials)?;

    let mo_indices: Vec<usize> = qp_range.clone().collect();
    let mut eps_qp = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_out = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_out = Array1::<f64>::zeros(mo_indices.len());
    let mut z_out = Array1::<f64>::ones(mo_indices.len());
    let mut qp_converged = vec![true; mo_indices.len()];

    let first_act = mo_b.first_act;
    let sigma_x_all = sigma_x_diag(mo_b);
    let ef = fermi_level(&mo_b.eps_act, mo_b.n_occ_act);
    // KS static shift Σ_x − v_xc, one value per QP MO. Computed ONCE here
    // (outside/before the outer G+W self-consistency loop below) from the
    // fixed mean-field sigma_x_all/vxc_diag — see the doc comment on this
    // function for why it must NOT be recomputed per outer iteration.
    let static_shifts: Vec<f64> = mo_indices
        .iter()
        .map(|&mo_abs| {
            let m_loc = mo_abs - first_act;
            vxc_diag.map(|v| sigma_x_all[m_loc] - v[mo_abs]).unwrap_or(0.0)
        })
        .collect();
    for (idx, &mo_abs) in mo_indices.iter().enumerate() {
        let m_loc = mo_abs - first_act;
        eps_mf[idx] = mo_b.eps_act[m_loc];
        sx_out[idx] = sigma_x_all[m_loc];
        eps_qp[idx] = eps_mf[idx];
    }
    let mut iter_done = 0usize;
    let mut outer_converged = false;
    for it in 0..gw_cfg.max_ev_iter {
        // Update shifted_rhf.eps_alpha by overlaying QP energies for the
        // QP block (in absolute MO indices).
        for (idx, &mo_abs) in mo_indices.iter().enumerate() {
            shifted_rhf.eps_alpha[mo_abs] = eps_qp[idx];
        }
        // Rebuild PDEP on iterations > 0.
        if it > 0 {
            current_pdep = ferric_rpa::run_pdep_rpa(mol, obs, dfbs, op, &shifted_rhf, pdep_cfg)?;
            current_v_dressed = w_pdep::redress_eigenpotentials(
                &mo_b.v_inv_sqrt,
                &current_pdep.eigenpotentials,
            )?;
        }
        let m_proj = project_b_into_pdep(mo_b, &current_v_dressed);
        let inv_diel_freq = current_pdep.inv_dielectric_freq.as_ref().ok_or_else(|| {
            FerricError::General(
                "PDEP result missing inv_dielectric_freq (GW requires the dense χ₀ path)".into(),
            )
        })?;

        // For the propagator denominator we also use QP-shifted ε's
        // (consistent with the rebuilt W).
        let mut eps_prop = mo_b.eps_act.clone();
        for (idx, &mo_abs) in mo_indices.iter().enumerate() {
            let m_loc = mo_abs - first_act;
            eps_prop[m_loc] = eps_qp[idx];
        }
        let mut max_dev = 0.0_f64;
        // Frozen (m_proj, W, eps_prop) snapshot ⇒ independent per-state solves.
        let qp_new: Vec<(f64, f64, f64, bool)> = mo_indices
            .par_iter()
            .zip(static_shifts.par_iter())
            .map(|(&mo_abs, &shift)| {
                let m_loc = mo_abs - first_act;
                solve_qp_for_mo(
                    m_loc,
                    mo_b.eps_act[m_loc],
                    &m_proj,
                    inv_diel_freq,
                    &current_pdep.quad_weights,
                    &current_pdep.quad_freqs,
                    &eps_prop,
                    gw_cfg.pade_npts,
                    gw_cfg.qp_newton_damp,
                    ef,
                    shift,
                )
            })
            .collect::<Result<Vec<_>, FerricError>>()?;
        for (idx, &(eps_new, sc_new, z_new, converged)) in qp_new.iter().enumerate() {
            max_dev = max_dev.max((eps_new - eps_qp[idx]).abs());
            eps_qp[idx] = eps_new;
            sc_out[idx] = sc_new;
            z_out[idx] = z_new;
            qp_converged[idx] = converged;
        }
        iter_done = it + 1;
        // Live per-iteration progress (see RhfConfig.verbose's doc for the
        // full rationale). STDOUT, opt-in via `gw_cfg.verbose`.
        if gw_cfg.verbose {
            println!(
                "evGW  iter={it:4}  max|d_eps_qp|={max_dev:.3e}  ev_conv_thresh={:.3e}",
                gw_cfg.ev_conv_thresh
            );
        }
        if max_dev < gw_cfg.ev_conv_thresh && it > 0 {
            outer_converged = true;
            break;
        }
    }
    warn_if_unconverged("evGW", &mo_indices, &qp_converged);
    if !outer_converged {
        eprintln!(
            "ferric-gw WARNING: evGW outer loop did not converge within max_ev_iter={} \
             (ev_conv_thresh={:.3e}); returned energies (and W) are the last iterate, \
             not self-consistent.",
            gw_cfg.max_ev_iter, gw_cfg.ev_conv_thresh
        );
    }

    Ok(GwResult {
        mo_indices,
        eps_mf,
        eps_qp,
        sigma_x: sx_out,
        sigma_c: sc_out,
        z_factor: z_out,
        qp_converged,
        n_ev_iter: iter_done,
        outer_converged,
        pdep: current_pdep,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pade_node_indices_matches_pyscf_get_ac_idx() {
        // Reference values from PySCF gw_ac._get_ac_idx (npts=18, step_ratio=2/3,
        // idx_start=1), the convention this ports. Computed against the local
        // pyscf checkout — see the fix commit message.
        let got = pade_node_indices(101, 18, 2.0 / 3.0, 1);
        let expect = vec![1, 8, 14, 20, 27, 33, 39, 44, 50, 56, 61, 66, 72, 77, 81, 86, 91, 95];
        assert_eq!(got, expect, "must reproduce PySCF _get_ac_idx(101,18,2/3,1)");

        let got2 = pade_node_indices(50, 18, 2.0 / 3.0, 1);
        let expect2 = vec![1, 4, 7, 11, 14, 17, 20, 23, 25, 28, 31, 33, 36, 38, 41, 43, 45, 48];
        assert_eq!(got2, expect2, "must reproduce PySCF _get_ac_idx(50,18,2/3,1)");

        // Degenerate guard: nw <= npts falls back to all indices (ascending).
        assert_eq!(pade_node_indices(10, 18, 2.0 / 3.0, 1), (0..10).collect::<Vec<_>>());
    }

    /// Reference scalar implementation of sigma_c_at_z (the pre-BLAS3 loop nest),
    /// kept in the test module to pin the optimized version's numerics.
    fn sigma_c_at_z_scalar_ref(
        m_idx: usize,
        z: Complex64,
        m_proj: &ndarray::Array3<f64>,
        inv_diel_freq: &[Array2<f64>],
        quad_weights: &[f64],
        quad_freqs: &[f64],
        eps_act: &[f64],
    ) -> Complex64 {
        let m_modes = m_proj.shape()[0];
        let n_quad = quad_freqs.len();
        let n_act = eps_act.len();
        let mut sigma = Complex64::new(0.0, 0.0);
        for n_idx in 0..n_act {
            let diff = z - Complex64::new(eps_act[n_idx], 0.0);
            let mut inner = Complex64::new(0.0, 0.0);
            for k in 0..n_quad {
                let omk = quad_freqs[k];
                let den = diff * diff + Complex64::new(omk * omk, 0.0);
                let wd = &inv_diel_freq[k];
                let mut w_mn = 0.0_f64;
                for a in 0..m_modes {
                    let ma = m_proj[(a, m_idx, n_idx)];
                    let mut row = 0.0_f64;
                    for b in 0..m_modes {
                        row += wd[(a, b)] * m_proj[(b, m_idx, n_idx)];
                    }
                    w_mn += ma * row;
                }
                inner += Complex64::new(quad_weights[k] * w_mn, 0.0) / den;
            }
            sigma += diff * inner;
        }
        -Complex64::new(1.0 / std::f64::consts::PI, 0.0) * sigma
    }

    #[test]
    fn sigma_c_blas3_matches_scalar_reference() {
        // The BLAS3 rewrite reorders the α,β summation (GEMM vs nested scalar),
        // so it agrees with the reference to machine precision, not bit-identically.
        // Use a non-trivial, dense, non-symmetric-in-(m,n) m_proj and a symmetric
        // inv_diel to exercise every path.
        let m_modes = 7;
        let n_act = 5;
        let n_quad = 4;
        // Deterministic pseudo-random fill (no rng dep): a cheap LCG-ish hash.
        let val = |i: usize| ((i.wrapping_mul(2654435761) % 1000) as f64) / 500.0 - 1.0;
        let mut m_proj = ndarray::Array3::<f64>::zeros((m_modes, n_act, n_act));
        let mut c = 0usize;
        for a in 0..m_modes {
            for m in 0..n_act {
                for n in 0..n_act {
                    m_proj[(a, m, n)] = val(c);
                    c += 1;
                }
            }
        }
        // Symmetric inv-dielectric matrices per frequency (W̃_d is symmetric).
        let mut inv_diel_freq = Vec::new();
        for _k in 0..n_quad {
            let mut d = Array2::<f64>::zeros((m_modes, m_modes));
            for a in 0..m_modes {
                for b in a..m_modes {
                    let x = val(c);
                    c += 1;
                    d[(a, b)] = x;
                    d[(b, a)] = x;
                }
            }
            inv_diel_freq.push(d);
        }
        let quad_freqs: Vec<f64> = (0..n_quad).map(|k| 0.2 + 0.3 * k as f64).collect();
        let quad_weights: Vec<f64> = (0..n_quad).map(|k| 1.0 / (k as f64 + 1.0)).collect();
        let eps_act: Vec<f64> = (0..n_act).map(|n| -0.6 + 0.25 * n as f64).collect();

        for &m_idx in &[0usize, 2, 4] {
            for &z in &[
                Complex64::new(0.1, 0.4),
                Complex64::new(-0.3, 0.0),
                Complex64::new(0.05, 1.2),
            ] {
                let got = sigma_c_at_z(
                    m_idx, z, &m_proj, &inv_diel_freq, &quad_weights, &quad_freqs, &eps_act,
                );
                let want = sigma_c_at_z_scalar_ref(
                    m_idx, z, &m_proj, &inv_diel_freq, &quad_weights, &quad_freqs, &eps_act,
                );
                assert!(
                    (got.re - want.re).abs() < 1e-12 && (got.im - want.im).abs() < 1e-12,
                    "m_idx={m_idx} z={z}: BLAS3 {got} vs scalar {want}"
                );
            }
        }
    }

    #[test]
    fn qp_newton_converges_for_a_mild_self_energy() {
        // Small screened-exchange weight ⇒ Σc is a gentle, well-behaved
        // function near ε_mf ⇒ Newton should converge well within budget.
        let eps_act = vec![-0.5_f64, 0.3_f64];
        let mut m_proj = ndarray::Array3::<f64>::zeros((1, 2, 2));
        m_proj[(0, 0, 1)] = 1.0;
        m_proj[(0, 1, 0)] = 1.0;
        let quad_freqs = vec![0.5_f64];
        let quad_weights = vec![1.0_f64];
        let mut d = Array2::<f64>::zeros((1, 1));
        d[(0, 0)] = 0.05; // mild coupling
        let inv_diel_freq = vec![d];
        let ef = fermi_level(&eps_act, 1);

        let (eps_qp, _sc, _z, converged) = solve_qp_for_mo(
            0, eps_act[0], &m_proj, &inv_diel_freq, &quad_weights, &quad_freqs,
            &eps_act, 0, 1.0, ef, 0.0,
        )
        .expect("well-conditioned Padé fit must not error");
        assert!(converged, "expected Newton to converge for a mild self-energy");
        assert!(eps_qp.is_finite());
    }

    #[test]
    fn qp_newton_flags_nonconvergence_when_iteration_budget_exhausted() {
        // Drive the fixed 30-iteration Newton budget to exhaustion — the
        // "max_iter=1"-style probe requested by the TD-CONV brief. The Newton
        // loop has no external max_iter knob, but its per-step damping is the
        // public `qp_newton_damp` config. At the clamp minimum (0.1) each
        // damped step removes only 10% of the remaining distance to the root,
        // so the distance shrinks by at most 0.9 per iteration: starting more
        // than ~1e-4 from the root, |step| = 0.1·dist ≥ 0.1·1e-4·0.9³⁰ ≈ 4e-7
        // > 1e-7 on every one of the 30 passes — the `|step| < 1e-7` success
        // criterion is unreachable within budget, deterministically.
        //
        // A moderate coupling (d = 0.5, Σc(ε_mf) ≈ 0.14 Ha with curvature)
        // guarantees the Z-linearized starting guess is displaced well beyond
        // 1e-4 from the self-consistent root.
        let eps_act = vec![-0.5_f64, 0.3_f64];
        let mut m_proj = ndarray::Array3::<f64>::zeros((1, 2, 2));
        m_proj[(0, 0, 1)] = 1.0;
        m_proj[(0, 1, 0)] = 1.0;
        let quad_freqs = vec![0.5_f64];
        let quad_weights = vec![1.0_f64];
        let mut d = Array2::<f64>::zeros((1, 1));
        d[(0, 0)] = 0.5; // moderate coupling: sizeable, curved Σc
        let inv_diel_freq = vec![d];
        let ef = fermi_level(&eps_act, 1);

        // pade_npts = 8 (NOT 0): with a single quadrature node, npts=0 would
        // collapse to a 1-point Padé — a *constant* Σc model whose linearized
        // start is exactly the root (Z=1), converging in zero steps. Eight
        // support points give the model genuine curvature so the linearized
        // start is displaced ~1e-3 from the self-consistent root.
        let (eps_qp, _sc, _z, converged) = solve_qp_for_mo(
            0, eps_act[0], &m_proj, &inv_diel_freq, &quad_weights, &quad_freqs,
            &eps_act, 8, 0.1, // qp_newton_damp at clamp minimum
            ef, 0.0,
        )
        .expect("well-conditioned Padé fit must not error");
        assert!(
            !converged,
            "expected the heavily-damped Newton to exhaust its 30-iteration budget \
             without meeting |step| < 1e-7 (got apparently-converged eps_qp={eps_qp})"
        );
        assert!(eps_qp.is_finite());

        // Same system, undamped: must converge — proves the flag tracks the
        // solve outcome and not the system construction.
        let (_e, _s, _z2, converged_full) = solve_qp_for_mo(
            0, eps_act[0], &m_proj, &inv_diel_freq, &quad_weights, &quad_freqs,
            &eps_act, 8, 1.0, ef, 0.0,
        )
        .expect("well-conditioned Padé fit must not error");
        assert!(converged_full, "undamped Newton on the same system must converge");
    }

    // The degenerate-Padé-node guard (repeated support point -> Err instead
    // of silent NaN/Inf) is exercised directly on `PadeCF::fit` in
    // `pade::tests::fit_errors_on_repeated_support_node` — `solve_qp_for_mo`
    // builds its support nodes by sub-sampling a strictly-increasing
    // scaled-Legendre eval grid via `pade_node_indices` (PySCF `_get_ac_idx`),
    // which is independent of `quad_freqs`'s *values*, so duplicate `quad_freqs`
    // entries cannot reach this call chain to produce duplicate `z_nodes`.
    // The node-index port is pinned against PySCF in
    // `pade_node_indices_matches_pyscf_get_ac_idx`.
}
