//! Restricted Open-Shell Hartree-Fock (ROHF) solver.
//!
//! Spin-pure open-shell HF: a single set of MOs partitioned into doubly
//! occupied (closed), singly occupied (open, α-only), and virtual blocks.
//! ⟨S²⟩ is exact S(S+1) by construction.
//!
//! Coupling: **Guest-Saunders** (PySCF default, hard-coded — no knob).
//! Implementation mirrors PySCF's `get_roothaan_fock` (`pyscf/scf/rohf.py`):
//! the effective Fock is built via density-based projectors
//!   P_c = D_β · S,   P_o = (D_α − D_β) · S,   P_v = I − D_α · S
//! and then assembled from F_c = (F_α + F_β)/2, F_α, F_β according to the
//! Roothaan block table:
//! ```text
//! ========  ======== ====== =========
//! space      closed   open   virtual
//! ========  ======== ====== =========
//! closed       Fc      Fb     Fc
//! open         Fb      Fc     Fa
//! virtual      Fc      Fa     Fc
//! ========  ======== ====== =========
//! ```
//! Per Guest-Saunders (a_cc = a_oo = a_vv = 1/2, b = -1/2, c = 3/2 in the
//! a/b/c parametrisation), the diagonal blocks reduce to F_c — exactly what
//! the projector form above produces.

use crate::diis::Diis;
use crate::direct_j::DirectJ;
use crate::direct_k::DirectK;
use crate::fock::{JBuilder, KBuilder};
use crate::guess::hcore_guess;
use crate::result::{ScfResult, Spin};
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ndarray::Array2;
use ndarray_linalg::Eigh;

/// `FERRIC_ROHF_TRACE` descriptor: per-iteration ROHF plateau-dynamics trace
/// (env-only debug toggle). NOTE behavior change: previously read via `.is_ok()`,
/// so ANY value (incl. `=0`) enabled it; now `=0`/`false`/`off` disable it,
/// consistent with every other `FERRIC_*_TRACE` flag.
static ROHF_TRACE: ferric_core::config::ConfigVar<bool> = ferric_core::config::ConfigVar {
    env_name: "FERRIC_ROHF_TRACE",
    default: false,
    parse: ferric_core::config::parse_toggle,
    validate: ferric_core::config::accept_any,
};

/// Whether the per-iteration ROHF trace is on. Malformed value → warn + off.
fn rohf_trace() -> bool {
    ROHF_TRACE.toggle()
}

/// ROHF configuration mirrors RHF.
pub type RohfConfig = RhfConfig;

/// Solve restricted open-shell Hartree-Fock equations.
///
/// Uses `mol.charge` and `mol.multiplicity` to determine doubly/singly
/// occupied orbital counts:
///   - nocc_open   = mult − 1                (singly α-occupied)
///   - nocc_double = (nelec − nocc_open) / 2 (doubly occupied)
pub fn solve_rohf(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    _op: Operator,
    bounds: &SchwarzBounds,
    config: &RohfConfig,
) -> Result<ScfResult, FerricError> {
    use ferric_dft::ks::KsXcUks;
    use ferric_dft::xc_trait::{KMix, UksXcContribution};

    // Build UKS XC contribution once. None for pure ROHF.
    let xc_contrib: Option<Box<dyn UksXcContribution>> = if let Some(name) = config.xc.as_deref() {
        let main = config.dft_grid.clone().unwrap_or_default();
        let nlc = config.nlc_grid.clone()
            .unwrap_or(ferric_dft::grid::AtomicGridConfig { n_radial: 50, n_angular: 50 });
        let ks = KsXcUks::new(mol, prep.basis_set(), name, &main, &nlc)
            .map_err(|e| FerricError::General(format!("KsXcUks init for {name}: {e:?}")))?;
        Some(Box::new(ks) as Box<dyn UksXcContribution>)
    } else {
        None
    };

    let k_mix: KMix = xc_contrib.as_ref().map(|x| x.k_mix()).unwrap_or_default();
    let c_k: f64 = if xc_contrib.is_some() { k_mix.sr } else { 1.0 };

    // RSH path (ω > 0): per-spin K from c_SR · K[erfc(ω)] + c_LR · K[erf(ω)]
    // via two DfK fitters (geometry-only — built once, contracted per spin
    // per iter). Mirrors solve_uhf's RSH path.
    let (mut dfk_sr, mut dfk_lr) = if k_mix.omega > 0.0 {
        use ferric_integrals::basis_bridge::PreparedBasis as _PB;
        let aux_name = config.df_k_aux.as_deref().unwrap_or("def2-universal-jkfit");
        let dfbs_set = ferric_core::basis::bundled(aux_name)?;
        let dfbs_prep = _PB::new(mol, &dfbs_set)?;
        let ooc_budget = crate::rhf::resolve_three_index_budget(config.three_index_budget_bytes);
        (
            Some(crate::df_k::DfK::new_banded(
                ferric_integrals::operator::Operator::erfc(k_mix.omega),
                prep, &dfbs_prep, ooc_budget, Some(ctx),
            )?),
            Some(crate::df_k::DfK::new_banded(
                ferric_integrals::operator::Operator::erf(k_mix.omega),
                prep, &dfbs_prep, ooc_budget, Some(ctx),
            )?),
        )
    } else {
        (None, None)
    };
    let s = oneelectron::overlap(prep);
    let h = oneelectron::hcore_with_external(prep, config.external_potential.as_ref())?;
    let n = prep.nbasis();
    let nelec = mol.nelec() as i64;
    let mult = mol.multiplicity as i64;
    if mult < 1 {
        return Err(FerricError::General(
            "ROHF: multiplicity must be >= 1".into(),
        ));
    }
    let two_s = mult - 1; // 2S = number of singly-occupied (α) orbitals
    if (nelec - two_s) % 2 != 0 || nelec < two_s {
        return Err(FerricError::General(format!(
            "ROHF: incompatible nelec={nelec} and multiplicity={mult}"
        )));
    }
    let nocc_open = two_s as usize;
    let nocc_double = ((nelec - two_s) / 2) as usize;
    let nocc_a = nocc_double + nocc_open;
    let nocc_b = nocc_double;
    if nocc_a + nocc_b != nelec as usize {
        return Err(FerricError::General(
            "ROHF: nocc_a + nocc_b != nelec".into(),
        ));
    }
    let vnn = mol.nuclear_repulsion()
        + config.external_potential.as_ref().map_or(0.0, |ext| {
            ext.charge_nuclear_energy(mol) + ext.field_nuclear_energy(mol)
        });

    // S^{-1/2}
    let (s_evals, s_evecs) = s
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("S diag: {e}")))?;
    let mut u_scaled = s_evecs.clone();
    for i in 0..n {
        let scale = 1.0 / s_evals[i].sqrt();
        for mu in 0..n {
            u_scaled[(mu, i)] *= scale;
        }
    }
    let s_inv_sqrt = u_scaled.dot(&s_evecs.t());

    // hcore guess MOs
    let _ = hcore_guess(&s, &h, nocc_a.max(1))?;
    let h_prime = s_inv_sqrt.dot(&h).dot(&s_inv_sqrt);
    let (_, c_prime) = h_prime
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("H' diag: {e}")))?;
    let mut c = s_inv_sqrt.dot(&c_prime);

    // ROHF densities (AO):
    //   D_c (closed/doubly-occupied) = 2 Σ_i C_i C_i^T  (i = 0..nocc_double)
    //   D_o (open/singly-α-occupied) = Σ_j C_j C_j^T    (j = nocc_double..nocc_a)
    // D_α = D_c/2 + D_o,  D_β = D_c/2  → (D_α + D_β) = D_c + D_o (total).
    // We track D_α and D_β internally to feed J/K builders (matching UHF JK).
    let (mut d_a, mut d_b) = build_rohf_densities(&c, nocc_double, nocc_open);

    let mut j_buf = Array2::<f64>::zeros((n, n));
    let mut k_a_buf = Array2::<f64>::zeros((n, n));
    let mut k_b_buf = Array2::<f64>::zeros((n, n));

    let mut diis = Diis::new(config.diis_size);
    let mut prev_e = 0.0;
    // MOM reference: (closed-MO AO block, open-MO AO block) from the most
    // recently accepted iter. None until iter `config.mom_after_iter`.
    let mut mom_ref: Option<(Array2<f64>, Array2<f64>)> = None;
    let mut total_quartets = 0usize;
    // Previous iteration's total density, for the ΔP convergence signal (shared
    // with solve_rhf via rhf::scf_converged). None on iter 1 → dp = INFINITY, so
    // the gate can't fire before a real density change exists.
    let mut prev_d_total: Option<Array2<f64>> = None;

    // K built per spin (same convention as solve_uhf's RSH path). Builders
    // hoisted out of the loop: each lazily builds a per-thread libint2
    // EnginePool on first use (ctors serialized behind a global mutex), so a
    // loop-local builder would pay that construction every iteration.
    let need_k = c_k != 0.0 || k_mix.omega > 0.0;
    let mut direct_j = DirectJ::new(ctx, prep, bounds, config.integral_thresh);
    let mut direct_k: Option<DirectK> = if need_k && k_mix.omega == 0.0 {
        Some(DirectK::new(ctx, prep, bounds, config.integral_thresh))
    } else {
        None
    };

    for iter in 1..=config.max_iter {
        ctx.check_interrupted()?;
        j_buf.fill(0.0);
        k_a_buf.fill(0.0);
        k_b_buf.fill(0.0);
        let d_total = &d_a + &d_b;
        // ΔP vs the previous iteration's total density — the primary convergence
        // signal (see rhf::scf_converged). INFINITY on iter 1.
        let (dp_rms, dp_max) = match prev_d_total.as_ref() {
            Some(prev) => {
                let diff = &d_total - prev;
                let n2 = (diff.len() as f64).max(1.0);
                (
                    (diff.iter().map(|v| v * v).sum::<f64>() / n2).sqrt(),
                    diff.iter().map(|v| v.abs()).fold(0.0f64, f64::max),
                )
            }
            None => (f64::INFINITY, f64::INFINITY),
        };
        prev_d_total = Some(d_total.clone());

        // J from D_total
        total_quartets += direct_j.build(&d_total, &mut j_buf)?;
        let mut k_a_total = Array2::<f64>::zeros((n, n));
        let mut k_b_total = Array2::<f64>::zeros((n, n));
        if k_mix.omega > 0.0 {
            let dfk_sr = dfk_sr.as_mut().expect("dfk_sr built when omega>0");
            let dfk_lr = dfk_lr.as_mut().expect("dfk_lr built when omega>0");
            let mut k_sr_a = Array2::<f64>::zeros((n, n));
            let mut k_lr_a = Array2::<f64>::zeros((n, n));
            let mut k_sr_b = Array2::<f64>::zeros((n, n));
            let mut k_lr_b = Array2::<f64>::zeros((n, n));
            dfk_sr.build(&d_a, &mut k_sr_a)?;
            dfk_lr.build(&d_a, &mut k_lr_a)?;
            dfk_sr.build(&d_b, &mut k_sr_b)?;
            dfk_lr.build(&d_b, &mut k_lr_b)?;
            k_a_total = k_mix.sr * &k_sr_a + k_mix.lr * &k_lr_a;
            k_b_total = k_mix.sr * &k_sr_b + k_mix.lr * &k_lr_b;
        } else if need_k {
            let dk = direct_k.as_mut().expect("DirectK built before loop");
            total_quartets += <DirectK as KBuilder>::build(dk, &d_a, &mut k_a_buf)?;
            total_quartets += <DirectK as KBuilder>::build(dk, &d_b, &mut k_b_buf)?;
            k_a_total = c_k * &k_a_buf;
            k_b_total = c_k * &k_b_buf;
        }

        // F_σ = H + J − K_σ_total  (then + V^σ_xc below for ROKS)
        let mut f_a: Array2<f64> = &h + &j_buf;
        let mut f_b: Array2<f64> = &h + &j_buf;
        if need_k {
            f_a -= &k_a_total;
            f_b -= &k_b_total;
        }

        // Pre-XC electronic energy.
        let e_elec_no_xc: f64 =
            0.5 * ((&(&h + &f_a) * &d_a).sum() + (&(&h + &f_b) * &d_b).sum());
        let e_xc = if let Some(x) = xc_contrib.as_ref() {
            x.add_xc_uks(&d_a, &d_b, &mut f_a, &mut f_b)
        } else {
            0.0
        };
        let energy = e_elec_no_xc + e_xc + vnn;

        // Build Roothaan effective Fock (Guest-Saunders, via PySCF projector form).
        let f_eff = roothaan_fock(&f_a, &f_b, &d_a, &d_b, &s);

        // DIIS error: the proper ROHF orbital-rotation gradient (PySCF
        // `get_grad`). In MO basis the gradient has only three nonzero
        // off-diagonal blocks — the *unique* orbital rotations:
        //   g[v,c] = f_α[v,c] + f_β[v,c]    (closed → virtual)
        //   g[v,o] = f_α[v,o]               (open → virtual; only α occupies open)
        //   g[o,c] = f_β[o,c]               (closed → open; only β leaves open)
        // We then antisymmetrize (g - g^T) and project back to AO basis so
        // DIIS still operates on (n × n) error matrices. This eliminates the
        // within-class-rotation ambiguity that causes the LDA/PBE doublet-OH
        // plateau in the FDS-SDF formulation.
        let f_a_mo: Array2<f64> = c.t().dot(&f_a).dot(&c);
        let f_b_mo: Array2<f64> = c.t().dot(&f_b).dot(&c);
        let mut g_mo: Array2<f64> = Array2::zeros((n, n));
        // closed → virtual block: rows = virtual, cols = closed
        for p in nocc_a..n {
            for q in 0..nocc_double {
                g_mo[(p, q)] = f_a_mo[(p, q)] + f_b_mo[(p, q)];
            }
        }
        // open → virtual block: rows = virtual, cols = open
        for p in nocc_a..n {
            for q in nocc_double..nocc_a {
                g_mo[(p, q)] = f_a_mo[(p, q)];
            }
        }
        // closed → open block: rows = open, cols = closed
        for p in nocc_double..nocc_a {
            for q in 0..nocc_double {
                g_mo[(p, q)] = f_b_mo[(p, q)];
            }
        }
        // Antisymmetrize in MO basis, then transform back to AO.
        let g_mo_anti: Array2<f64> = &g_mo - &g_mo.t();
        let err: Array2<f64> = s.dot(&c).dot(&g_mo_anti).dot(&c.t()).dot(&s);

        let de = (energy - prev_e).abs();
        let err_max = err.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        // Converge on ΔP + loose ΔE, the same gate as solve_rhf/solve_uhf — the
        // ROHF cation path (used as a gw100 fallback) hits the same RI energy/
        // gradient noise floor at aTZ, so the old `err_max < density_conv` gate
        // would grind to MaxIter there too. See rhf::scf_converged.
        let conv_exit = crate::rhf::scf_converged(
            crate::rhf::ConvergenceSignals { de, dp_rms, dp_max },
            config.energy_conv,
            config.density_conv,
        );
        let converged = conv_exit.is_some();

        // FERRIC_ROHF_TRACE=1: per-iter diagnostic of plateau dynamics.
        // Logs (iter, energy, ΔE, err_max, per-block gradient max,
        // eigenvalues straddling SOMO, and the occupied-α↔virt overlap).
        if rohf_trace() {
            let (eps_now, _c_now) = diagonalize(&f_eff, &s_inv_sqrt)?;
            // Eigenvalues around the SOMO. With nocc_double β-pairs and
            // nocc_open singly-α-occupied orbitals, the SOMO index range is
            // [nocc_double .. nocc_double+nocc_open) and the LUMO starts at
            // nocc_a = nocc_double + nocc_open.
            let nlow = nocc_double.saturating_sub(2);
            let nhi = (nocc_a + 2).min(n);
            let eps_window: Vec<String> = (nlow..nhi).map(|i| {
                let tag = if i < nocc_double { " D " }
                    else if i < nocc_a { " S " }
                    else { " V " };
                format!("[{i}{tag}{:.4}]", eps_now[i])
            }).collect();
            // Per-block gradient maxima from g_mo (pre-antisymmetrize).
            let mut g_vc_max = 0.0f64;
            let mut g_vo_max = 0.0f64;
            let mut g_oc_max = 0.0f64;
            for p in nocc_a..n {
                for q in 0..nocc_double { g_vc_max = g_vc_max.max(g_mo[(p, q)].abs()); }
                for q in nocc_double..nocc_a { g_vo_max = g_vo_max.max(g_mo[(p, q)].abs()); }
            }
            for p in nocc_double..nocc_a {
                for q in 0..nocc_double { g_oc_max = g_oc_max.max(g_mo[(p, q)].abs()); }
            }
            eprintln!(
                "ROHFTRACE it={iter:>3} E={energy:.10} dE={de:.3e} err={err_max:.3e} |g|vc={g_vc_max:.3e} |g|vo={g_vo_max:.3e} |g|oc={g_oc_max:.3e}  eps:{}",
                eps_window.join(" ")
            );
        }

        if iter > 1 && converged {
            let (eps, c_f) = diagonalize(&f_eff, &s_inv_sqrt)?;
            let (d_a_f, d_b_f) = build_rohf_densities(&c_f, nocc_double, nocc_open);
            let density_total = &d_a_f + &d_b_f;
            return Ok(ScfResult {
                spin: Spin::RestrictedOpen,
                energy,
                density_total,
                density_alpha: d_a_f,
                density_beta: Some(d_b_f),
                mos_alpha: c_f,
                mos_beta: None,
                eps_alpha: eps,
                eps_beta: None,
                fock_alpha: f_eff,
                fock_beta: None,
                converged: true,
                exit: crate::result::ScfExit::Converged,
                iterations: iter,
                computed_quartets: total_quartets,
            });
        }
        prev_e = energy;

        // Pick update strategy. Newton step (if enabled and below trigger)
        // uses per-spin diagonal Fock entries to precondition a PCG solve of
        // H·κ = −g, then rotates C via the Cayley unitary.
        //   - HF (xc=None): full Newton with HF orbital Hessian only.
        //   - LDA ROKS: Newton + LDA f_xc kernel response.
        //   - GGA / hybrid / RSH ROKS: not supported yet, falls through to DIIS.
        let xc_is_lda = matches!(
            config.xc.as_deref(),
            Some("LDA") | Some("lda")
        );
        // Augmented-Hessian Newton — used when err_max is below ah_trigger.
        // Handles vanishing Hessian eigenvalues (e.g., doublet OH at LDA with
        // near-degenerate SOMO/HOMO) that PCG can't resolve.
        let use_ah = config.ah_trigger > 0.0
            && iter > 3
            && err_max < config.ah_trigger
            && (xc_contrib.is_none() || xc_is_lda)
            && k_mix.omega == 0.0;
        if use_ah {
            let (_, c_now) = diagonalize(&f_eff, &s_inv_sqrt)?;
            let f_a_mo = c_now.t().dot(&f_a).dot(&c_now);
            let f_b_mo = c_now.t().dot(&f_b).dot(&c_now);
            let eps_dummy: Vec<f64> = (0..n).map(|i| f_a_mo[(i, i)] + f_b_mo[(i, i)]).collect();

            let lda_kernel_and_ref = if xc_is_lda {
                let main = config.dft_grid.clone().unwrap_or_default();
                let xc_def = ferric_dft::libxc::xc_def_from_name_nspin("LDA", 2)
                    .map_err(|e| FerricError::General(format!("LDA fxc def: {e:?}")))?;
                let kernel = ferric_dft::fxc::LdaFxcKernel::new(
                    mol, prep.basis_set(), xc_def, &main,
                ).map_err(|e| FerricError::General(format!("LdaFxcKernel: {e}")))?;
                let (rho_a0, rho_b0) = kernel.reference_density(&d_a, &d_b);
                Some((kernel, rho_a0, rho_b0))
            } else {
                None
            };
            #[allow(clippy::type_complexity)]
            let fxc_storage: Option<Box<dyn Fn(&Array2<f64>, &Array2<f64>) -> (Array2<f64>, Array2<f64>) + Sync + '_>> =
                match lda_kernel_and_ref.as_ref() {
                    Some((k, ra, rb)) => Some(Box::new(
                        move |dd_a: &Array2<f64>, dd_b: &Array2<f64>| k.apply_with_ref(ra, rb, dd_a, dd_b)
                    )),
                    None => None,
                };
            let fxc_ref: Option<&crate::rohf_newton::FxcResponse<'_>> = fxc_storage.as_deref();

            let inputs = crate::rohf_newton::RohfNewtonInputs {
                prep,
                bounds,
                c: &c_now,
                eps: &eps_dummy,
                f_a_mo: &f_a_mo,
                f_b_mo: &f_b_mo,
                nocc_double,
                nocc_open,
                k_mix_sr: if k_mix.omega > 0.0 { 0.0 } else { c_k },
                fxc: fxc_ref,
                thresh: config.integral_thresh,
            };
            let ah_inputs = crate::rohf_ah::RohfAhInputs { base: &inputs };
            let (c_new, _kmax) = crate::rohf_ah::rohf_ah_step(
                ctx, &ah_inputs,
                /*max_step=*/0.2,
                /*davidson_conv=*/1e-7,
                /*davidson_max_vecs=*/50,
            )?;
            c = c_new;
            let (da_n, db_n) = build_rohf_densities(&c, nocc_double, nocc_open);
            d_a = da_n;
            d_b = db_n;
            continue;
        }
        let use_newton = config.newton_trigger > 0.0
            && iter > 3
            && err_max < config.newton_trigger
            && (xc_contrib.is_none() || xc_is_lda);
        if use_newton {
            let (_, c_now) = diagonalize(&f_eff, &s_inv_sqrt)?;
            let f_a_mo = c_now.t().dot(&f_a).dot(&c_now);
            let f_b_mo = c_now.t().dot(&f_b).dot(&c_now);
            let eps_dummy: Vec<f64> = (0..n).map(|i| f_a_mo[(i, i)] + f_b_mo[(i, i)]).collect();

            // Build LDA f_xc kernel + reference densities (closure target).
            // We keep the kernel + reference ρ here so the closure borrows them.
            let lda_kernel_and_ref = if xc_is_lda {
                let main = config.dft_grid.clone().unwrap_or_default();
                let xc_def = ferric_dft::libxc::xc_def_from_name_nspin("LDA", 2)
                    .map_err(|e| FerricError::General(format!("LDA fxc def: {e:?}")))?;
                let kernel = ferric_dft::fxc::LdaFxcKernel::new(
                    mol, prep.basis_set(), xc_def, &main,
                )
                .map_err(|e| FerricError::General(format!("LdaFxcKernel: {e}")))?;
                let (rho_a0, rho_b0) = kernel.reference_density(&d_a, &d_b);
                Some((kernel, rho_a0, rho_b0))
            } else {
                None
            };
            // Stack-local closure storage so its lifetime matches
            // `lda_kernel_and_ref`. Two branches (Some / None) to avoid the
            // Option::map inference that forces 'static on the trait object.
            #[allow(clippy::type_complexity)]
            let fxc_storage: Option<Box<dyn Fn(&Array2<f64>, &Array2<f64>) -> (Array2<f64>, Array2<f64>) + Sync + '_>> =
                match lda_kernel_and_ref.as_ref() {
                    Some((k, ra, rb)) => Some(Box::new(
                        move |dd_a: &Array2<f64>, dd_b: &Array2<f64>| k.apply_with_ref(ra, rb, dd_a, dd_b)
                    )),
                    None => None,
                };
            let fxc_ref: Option<&crate::rohf_newton::FxcResponse<'_>> = fxc_storage
                .as_deref();

            let inputs = crate::rohf_newton::RohfNewtonInputs {
                prep,
                bounds,
                c: &c_now,
                eps: &eps_dummy,
                f_a_mo: &f_a_mo,
                f_b_mo: &f_b_mo,
                nocc_double,
                nocc_open,
                k_mix_sr: if k_mix.omega > 0.0 { 0.0 } else { c_k },
                fxc: fxc_ref,
                thresh: config.integral_thresh,
            };
            let (c_new, _kmax) = crate::rohf_newton::rohf_newton_step(
                ctx, &inputs,
                config.level_shift.max(1e-6),
                0.1,  // trust radius (conservative — ROKS hessians are stiff)
                20,
                1e-7,
            )?;
            c = c_new;
            let (da_n, db_n) = build_rohf_densities(&c, nocc_double, nocc_open);
            d_a = da_n;
            d_b = db_n;
        } else {
            // DIIS extrapolate effective Fock, then optionally level-shift the
            // virtual–virtual block in MO basis to damp open-shell oscillations.
            let mut f_new = diis.step(&f_eff, &err);
            if config.level_shift > 0.0 && iter > 1 {
                const SHIFT_DAMP_ERR: f64 = 1e-3;
                let damp = err_max / (err_max + SHIFT_DAMP_ERR);
                let shift_eff = config.level_shift * damp;
                if shift_eff > 1e-10 {
                    let c_virt = c.slice(ndarray::s![.., nocc_a..]);
                    let p_virt: Array2<f64> = c_virt.dot(&c_virt.t());
                    let shift_term: Array2<f64> = shift_eff * s.dot(&p_virt).dot(&s);
                    f_new += &shift_term;
                }
            }
            let (_, c_new) = diagonalize(&f_new, &s_inv_sqrt)?;
            let c_after_mom = if config.mom_after_iter > 0 && iter > config.mom_after_iter {
                match mom_ref.as_ref() {
                    Some((ref_closed, ref_open)) => crate::mom::mom_reorder(
                        &c_new,
                        &s,
                        ref_closed,
                        ref_open,
                        nocc_double,
                        nocc_open,
                    ),
                    None => c_new,
                }
            } else {
                c_new
            };
            c = c_after_mom;
            if config.mom_after_iter > 0 && iter >= config.mom_after_iter {
                let ref_closed = c.slice(ndarray::s![.., ..nocc_double]).to_owned();
                let ref_open = c
                    .slice(ndarray::s![.., nocc_double..nocc_double + nocc_open])
                    .to_owned();
                mom_ref = Some((ref_closed, ref_open));
            }
            let (da_n, db_n) = build_rohf_densities(&c, nocc_double, nocc_open);
            d_a = da_n;
            d_b = db_n;
        }
    }
    Err(FerricError::ScfConvergence {
        iterations: config.max_iter,
        last_energy: prev_e,
    })
}

/// Build ROHF α/β densities from MO coefficients:
///   D_β = Σ_{i<nocc_double} C_i C_i^T
///   D_α = D_β + Σ_{j∈open} C_j C_j^T
fn build_rohf_densities(
    c: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
) -> (Array2<f64>, Array2<f64>) {
    let n = c.nrows();
    let mut d_b = Array2::<f64>::zeros((n, n));
    if nocc_double > 0 {
        let cd = c.slice(ndarray::s![.., ..nocc_double]);
        d_b = cd.dot(&cd.t());
    }
    let mut d_a = d_b.clone();
    if nocc_open > 0 {
        let co = c.slice(ndarray::s![.., nocc_double..nocc_double + nocc_open]);
        d_a = &d_a + &co.dot(&co.t());
    }
    (d_a, d_b)
}

/// Roothaan effective Fock (Guest-Saunders coupling).
/// Mirrors `pyscf.scf.rohf.get_roothaan_fock`.
fn roothaan_fock(
    f_a: &Array2<f64>,
    f_b: &Array2<f64>,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    s: &Array2<f64>,
) -> Array2<f64> {
    let n = s.shape()[0];
    let f_c = 0.5 * (f_a + f_b);
    // Projectors: P_c = D_β S, P_o = (D_α − D_β) S, P_v = I − D_α S
    let p_c = d_b.dot(s);
    let do_diff: Array2<f64> = d_a - d_b;
    let p_o = do_diff.dot(s);
    let mut p_v = Array2::<f64>::eye(n);
    p_v = &p_v - &d_a.dot(s);

    // Upper-triangle pieces (PySCF builds half then symmetrises by F + F^T).
    let p_c_t = p_c.t();
    let p_o_t = p_o.t();
    let p_v_t = p_v.t();

    let mut f = 0.5 * p_c_t.dot(&f_c).dot(&p_c);
    f = &f + &(0.5 * p_o_t.dot(&f_c).dot(&p_o));
    f = &f + &(0.5 * p_v_t.dot(&f_c).dot(&p_v));
    f = &f + &p_o_t.dot(f_b).dot(&p_c);
    f = &f + &p_o_t.dot(f_a).dot(&p_v);
    f = &f + &p_v_t.dot(&f_c).dot(&p_c);
    let f_sym = &f + &f.t();
    f_sym
}

fn diagonalize(
    f: &Array2<f64>,
    s_inv_sqrt: &Array2<f64>,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    let f_prime = s_inv_sqrt.dot(f).dot(s_inv_sqrt);
    let (evals, evecs) = f_prime
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("F diag: {e}")))?;
    let c = s_inv_sqrt.dot(&evecs);
    Ok((evals.to_vec(), c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    #[test]
    fn test_rohf_h_atom_sto3g() {
        // Single H atom — trivial 1-electron case.
        let mol = Molecule::parse_xyz("1\nH\nH 0 0 0\n", 0, 2).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let cfg = RohfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-9,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let res = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        assert!(res.converged);
        assert!(
            (res.energy + 0.466581850).abs() < 1e-5,
            "H atom energy = {}",
            res.energy
        );
    }
}
