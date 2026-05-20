//! Closed-shell restricted Hartree-Fock (RHF) solver.
//!
//! Implements the Roothaan-Hall SCF procedure with DIIS convergence acceleration
//! and Schwarz-screened two-electron integral evaluation.

use crate::df_j::DfJ;
use crate::df_k::DfK;
use crate::diis::Diis;
use crate::direct_j::DirectJ;
use crate::direct_jk::DirectJK;
use crate::direct_k::DirectK;
use crate::fock::{JBuilder, KBuilder};
use crate::guess::hcore_guess;
use crate::result::{ScfResult, Spin};

use crate::link_k::LinkK;
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_core::parallel::ParallelContext;
use ndarray::Array2;
use ndarray_linalg::Eigh;

/// Configuration parameters for the RHF solver.
#[derive(Debug, Clone)]
pub struct RhfConfig {
    pub max_iter: usize,
    pub energy_conv: f64,
    pub density_conv: f64,
    pub diis_size: usize,
    pub integral_thresh: f64,
    /// Choose K matrix builder: "direct" (default) or "link".
    pub k_builder: Option<String>,
    /// Optional auxiliary basis for density-fitted Coulomb (RI-J). When set, J is
    /// built from precomputed 3-center ERIs in O(N^2 · naux) per iteration instead
    /// of contracting full 4-index ERIs.
    pub df_j_aux: Option<String>,
    /// Optional auxiliary basis for density-fitted exchange (RI-K). When set, K is
    /// built from the V^{-1/2}-dressed 3-center tensor in O(N^3 · naux) GEMMs.
    /// Should be a JK-fit basis (e.g. `def2-universal-jkfit`), not an RI/MP2-fit
    /// basis, which would introduce mHa-scale error in K.
    pub df_k_aux: Option<String>,
    /// XC functional name (None = pure HF; e.g. "LDA", "PBE", "B3LYP", "wB97X-V").
    pub xc: Option<String>,
    /// Main DFT grid spec. Default (75, 110) when xc.is_some().
    pub dft_grid: Option<ferric_dft::grid::AtomicGridConfig>,
    /// NLC (VV10) grid spec. Default (50, 50) when XC requires VV10.
    pub nlc_grid: Option<ferric_dft::grid::AtomicGridConfig>,
    /// Level shift (Ha) applied to the virtual–virtual block of the Fock
    /// matrix in MO basis. Defaults to 0 (no shift). Used to damp oscillations
    /// in open-shell SCF (ROHF/UHF/ROKS) where DIIS plateaus near a near-
    /// degenerate transition. A shift of 0.1–0.5 Ha is typical.
    pub level_shift: f64,
}

impl Default for RhfConfig {
    fn default() -> Self {
        Self {
            // Tightened from (1e-8, 1e-7, 100) after H2O+ false-convergence
            // diagnosis: with the looser tolerances, UHF would report
            // converged=true at a state 85 mHa above the true minimum.
            max_iter: 200,
            energy_conv: 1e-10,
            density_conv: 1e-8,
            diis_size: 8,
            integral_thresh: 1e-12,
            k_builder: None,
            df_j_aux: None,
            df_k_aux: None,
            xc: None,
            dft_grid: None,
            nlc_grid: None,
            level_shift: 0.0,
        }
    }
}

/// Solve the closed-shell RHF equations for a molecule.
///
/// Uses the Roothaan-Hall procedure: build Fock matrix from density, diagonalize,
/// rebuild density, iterate until convergence. DIIS extrapolation accelerates
/// convergence. Returns [`ScfResult`] on success or [`FerricError::ScfConvergence`]
/// if `max_iter` is exceeded.
pub fn solve_rhf(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
) -> Result<ScfResult, FerricError> {
    let s = oneelectron::overlap(prep);
    let h = oneelectron::hcore(prep);
    let n = prep.nbasis();
    let nelec = mol.nelec();
    if nelec % 2 != 0 {
        return Err(FerricError::ScfConvergence {
            iterations: 0,
            last_energy: 0.0,
        });
    }
    let nocc = (nelec / 2) as usize;
    let vnn = mol.nuclear_repulsion();

    let mut d = hcore_guess(&s, &h, nocc)?;
    let mut f = Array2::zeros((n, n));
    let mut j_buf = Array2::<f64>::zeros((n, n));
    let mut k_buf = Array2::<f64>::zeros((n, n));
    let mut diis = Diis::new(config.diis_size);
    let mut prev_e = 0.0;
    let mut total_quartets = 0;

    if let Some(kb) = config.k_builder.as_deref() {
        if kb != "direct" && kb != "link" {
            return Err(FerricError::General(format!("unknown k_builder '{kb}': valid options are 'direct', 'link'")));
        }
    }

    // Build the XC contribution once. None for pure HF. Done before df_j/df_k
    // setup so we can read k_mix and apply auto JK-aux defaults for hybrid /
    // RSH functionals without forcing the user to set them explicitly.
    use ferric_dft::ks::KsXc;
    use ferric_dft::xc_trait::{XcContribution, KMix};

    let xc_contrib: Option<Box<dyn XcContribution>> = if let Some(name) = config.xc.as_deref() {
        let main = config.dft_grid.clone().unwrap_or_default();
        let nlc = config.nlc_grid.clone()
            .unwrap_or(ferric_dft::grid::AtomicGridConfig { n_radial: 50, n_angular: 50 });
        let ks = KsXc::new(mol, prep.basis_set(), name, &main, &nlc)
            .map_err(|e| FerricError::General(format!("KsXc init for {name}: {e:?}")))?;
        Some(Box::new(ks) as Box<dyn XcContribution>)
    } else {
        None
    };

    let k_mix: KMix = xc_contrib.as_ref().map(|x| x.k_mix()).unwrap_or_default();

    // Auto-default JK aux bases when the functional needs exact exchange but
    // the caller hasn't explicitly set df_j_aux / df_k_aux. This makes
    // `cfg.xc = Some("B3LYP")` (or any hybrid/RSH) work out of the box.
    // Pure HF (no xc) keeps the historical behavior of no auto-default.
    let needs_k = xc_contrib.is_some() && (k_mix.sr > 0.0 || k_mix.omega > 0.0);
    let needs_j = xc_contrib.is_some();
    const DEFAULT_JK_AUX: &str = "def2-universal-jkfit";
    let df_j_aux_eff: Option<String> = config.df_j_aux.clone()
        .or_else(|| needs_j.then(|| DEFAULT_JK_AUX.into()));
    let df_k_aux_eff: Option<String> = config.df_k_aux.clone()
        .or_else(|| needs_k.then(|| DEFAULT_JK_AUX.into()));

    // Density-fitted Coulomb (RI-J). Builds 3-center tensor + inverse metric once.
    let mut df_j: Option<DfJ> = if let Some(aux_name) = df_j_aux_eff.as_deref() {
        let dfbs_set = ferric_core::basis::bundled(aux_name)?;
        let dfbs = PreparedBasis::new(mol, &dfbs_set)?;
        Some(DfJ::new(op, prep, &dfbs)?)
    } else {
        None
    };

    // Density-fitted exchange (RI-K). Builds V^{-1/2}-dressed 3-center tensor once.
    let mut df_k: Option<DfK> = if let Some(aux_name) = df_k_aux_eff.as_deref() {
        let dfbs_set = ferric_core::basis::bundled(aux_name)?;
        let dfbs = PreparedBasis::new(mol, &dfbs_set)?;
        Some(DfK::new(op, prep, &dfbs)?)
    } else {
        None
    };

    // RSH path: pre-build the SR/LR DfK fitters once. Their 3-center B[P,μ,ν]
    // tensors are geometry-only — only the D-dependent contraction happens per
    // SCF iteration. Building inside the loop was a major hot-spot for wB97X-V.
    let (mut dfk_sr, mut dfk_lr) = if k_mix.omega > 0.0 {
        let aux_name = df_k_aux_eff.as_deref().ok_or_else(|| {
            FerricError::General(
                "Range-separated hybrid requires RhfConfig.df_k_aux (e.g. \"def2-universal-jkfit\")".into()
            )
        })?;
        let dfbs_set = ferric_core::basis::bundled(aux_name)?;
        let dfbs_prep = PreparedBasis::new(mol, &dfbs_set)?;
        (
            Some(DfK::new(Operator::erfc(k_mix.omega), prep, &dfbs_prep)?),
            Some(DfK::new(Operator::erf(k_mix.omega), prep, &dfbs_prep)?),
        )
    } else {
        (None, None)
    };

    // Build LinkK once — SignificantPairs is geometry-dependent and expensive per iteration.
    // When using the "link" builder, compute a fresh SchwarzBounds to own the lifetime.
    let link_schwarz_opt = if config.k_builder.as_deref() == Some("link") {
        Some(SchwarzBounds::compute(op, prep)?)
    } else {
        None
    };
    let mut k_builder: Option<Box<dyn KBuilder>> = link_schwarz_opt.as_ref().map(|sb| {
        let mut lk = LinkK::new(ctx, prep, sb, op, config.integral_thresh);
        lk.update_density(&d);
        Box::new(lk) as Box<dyn KBuilder>
    });

    // Precompute S^{-1/2} = U * diag(1/sqrt(λ)) * U^T  (BLAS dgemm)
    let (s_evals, s_evecs) = s
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("S diag: {e}")))?;
    // Scale each column i of U by 1/sqrt(λ_i)
    let mut u_scaled = s_evecs.clone();
    for i in 0..n {
        let scale = 1.0 / s_evals[i].sqrt();
        for mu in 0..n {
            u_scaled[(mu, i)] *= scale;
        }
    }
    let s_inv_sqrt = u_scaled.dot(&s_evecs.t());

    for iter in 1..=config.max_iter {
        ctx.check_interrupted()?;
        // Build J and K using selected builder (reuse pre-allocated buffers)
        j_buf.fill(0.0);
        k_buf.fill(0.0);
        // Build J: DF-J if configured, else fall through to combined direct path below.
        // Build K: DF-K > LinkK > combined DirectJK, in priority order.
        if df_j.is_some() || df_k.is_some() {
            if let Some(dfj) = df_j.as_mut() {
                dfj.build(&d, &mut j_buf)?;
            } else {
                let mut direct_j = DirectJ::new(ctx, prep, bounds, config.integral_thresh);
                total_quartets += direct_j.build(&d, &mut j_buf)?;
            }
            if let Some(dfk) = df_k.as_mut() {
                dfk.build(&d, &mut k_buf)?;
            } else {
                let mut direct_k = DirectK::new(ctx, prep, bounds, config.integral_thresh);
                total_quartets += <DirectK as KBuilder>::build(&mut direct_k, &d, &mut k_buf)?;
            }
        } else if let Some(lk) = k_builder.as_mut() {
            let mut direct_j = DirectJ::new(ctx, prep, bounds, config.integral_thresh);
            total_quartets += direct_j.build(&d, &mut j_buf)?;
            lk.update_density(&d);
            total_quartets += lk.build(&d, &mut k_buf)?;
        } else {
            let mut direct_jk = DirectJK::new(ctx, prep, bounds, config.integral_thresh);
            total_quartets += direct_jk.build(&d, &mut j_buf, &mut k_buf)?;
        }

        // k_total accumulates the exact-exchange contribution to be subtracted from F
        // as ½ · k_total. Convention:
        //   pure HF (xc=None):    k_mix = {1, 1, 0}      → k_total = k_buf (existing)
        //   pure DFT (LDA/PBE):   k_mix = {0, 0, 0}      → k_total = 0 (skip K)
        //   plain hybrid (B3LYP): k_mix = {α, α, 0}      → k_total = α · k_buf
        //   RSH (wB97X-V):        k_mix = {sr, lr, ω>0}  → k_total = sr·K_SR + lr·K_LR
        let k_total: Array2<f64> = if k_mix.omega > 0.0 {
            // Range-separated: SR/LR DfK fitters were built once before the
            // loop (geometry-only). Only the D-dependent contraction runs here.
            let dfk_sr = dfk_sr.as_mut().expect("dfk_sr built when omega>0");
            let dfk_lr = dfk_lr.as_mut().expect("dfk_lr built when omega>0");

            let mut k_sr = Array2::<f64>::zeros((n, n));
            dfk_sr.build(&d, &mut k_sr)?;

            let mut k_lr = Array2::<f64>::zeros((n, n));
            dfk_lr.build(&d, &mut k_lr)?;

            k_mix.sr * &k_sr + k_mix.lr * &k_lr
        } else if k_mix.sr > 0.0 {
            // Plain hybrid or pure HF: use the K already built by the existing builder path.
            k_mix.sr * &k_buf
        } else {
            // Pure DFT: no exact exchange.
            Array2::<f64>::zeros((n, n))
        };

        // F = H + J − ½ K_total  (V_xc added below)
        f.assign(&(&h + &j_buf - &(0.5 * &k_total)));

        // Electronic energy BEFORE adding V_xc (V_xc is one-body in F but
        // E_xc is its own integral).
        let e_elec_no_xc: f64 = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| 0.5 * d[(i, j)] * (h[(i, j)] + f[(i, j)]))
            .sum();
        let e_xc = if let Some(x) = xc_contrib.as_ref() {
            x.add_xc(&d, &mut f)
        } else {
            0.0
        };
        let energy = e_elec_no_xc + e_xc + vnn;

        // DIIS error: e = FDS - SDF
        let fds = f.dot(&d).dot(&s);
        let sdf = s.dot(&d).dot(&f);
        let err = &fds - &sdf;

        let de = (energy - prev_e).abs();
        let err_max = err.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        if std::env::var("FERRIC_SCF_TRACE").ok().as_deref() == Some("1") {
            eprintln!("SCF iter={:4}  E={:.12}  dE={:.3e}  err_max={:.3e}", iter, energy, de, err_max);
        }

        // DF builds introduce O(1e-6) Ha noise in the K matrix per iteration that
        // breaks strict energy variational convergence even when orbitals have
        // fully converged. When DF is active, accept on |FDS-SDF| (orbital gradient)
        // alone — the same criterion PySCF uses for DF-SCF.
        //
        // For large polyaromatic molecules with near-degenerate π orbitals the DF
        // noise floor can park err_max at ~1-5×density_conv indefinitely (H1
        // diagnosis: plateau, not oscillation). The fallback below accepts when
        // the gradient is within a 10× factor of the threshold AND the energy
        // change is below 1e-5 Ha (safely in the noise floor plateau, not still
        // descending toward the minimum).
        let df_active = df_j.is_some() || df_k.is_some();
        let energy_ok = de < config.energy_conv;
        let grad_ok = err_max < config.density_conv;
        let df_noise_floor_ok = df_active
            && err_max < 10.0 * config.density_conv
            && de < 1e-5;
        let converged = if df_active { grad_ok || df_noise_floor_ok } else { energy_ok && grad_ok };

        if iter > 1 && converged {
            let (orb_e, c) = diagonalize(&f, &s_inv_sqrt)?;
            let density_alpha = 0.5 * &d;
            return Ok(ScfResult {
                spin: Spin::Restricted,
                energy,
                density_total: d,
                density_alpha,
                density_beta: None,
                mos_alpha: c,
                mos_beta: None,
                eps_alpha: orb_e,
                eps_beta: None,
                fock_alpha: f,
                fock_beta: None,
                converged: true,
                iterations: iter,
                computed_quartets: total_quartets,
            });
        }
        prev_e = energy;

        let f_new = diis.step(&f, &err);
        let (_, c) = diagonalize(&f_new, &s_inv_sqrt)?;

        // Rebuild density: D = 2 * C_occ @ C_occ^T  (BLAS dgemm)
        let c_occ = c.slice(ndarray::s![.., ..nocc]);
        d.assign(&(2.0 * c_occ.dot(&c_occ.t())));
    }
    Err(FerricError::ScfConvergence {
        iterations: config.max_iter,
        last_energy: prev_e,
    })
}

/// Build the Coulomb (J) and exchange (K) matrices from the density matrix.
///
/// Uses Schwarz screening and 8-fold permutational symmetry of the ERIs.
pub fn build_jk(
    ctx: &ParallelContext,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    thresh: f64,
    d: &Array2<f64>,
    j: &mut Array2<f64>,
    k: &mut Array2<f64>,
) -> Result<usize, FerricError> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    ctx.check_interrupted()?;

    let nsh = prep.nshells();
    let nbf = prep.nbasis();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let computed_quartets = AtomicUsize::new(0);

    // Shell-blocked density-max table for Häser-Ahlrichs pair-wise screening.
    let mut d_max_shell = Array2::<f64>::zeros((nsh, nsh));
    for si in 0..nsh {
        for sj in 0..nsh {
            let (oi, ni) = (offs[si], dims[si]);
            let (oj, nj) = (offs[sj], dims[sj]);
            let mut m = 0.0f64;
            for a in 0..ni {
                for b in 0..nj {
                    let v = unsafe { d.uget((oi + a, oj + b)).abs() };
                    if v > m { m = v; }
                }
            }
            d_max_shell[(si, sj)] = m;
        }
    }

    let shell_pairs: Vec<_> = (0..nsh)
        .flat_map(|s1| (0..=s1).map(move |s2| (s1, s2)))
        .collect();

    let total_jk = shell_pairs.into_par_iter().fold(
        || (
            Engine::new_2e(bounds.op, prep, 1e-14).unwrap(),
            Array2::zeros((nbf, nbf)),
            Array2::zeros((nbf, nbf)),
            0usize
        ),
        |(mut engine, mut local_j, mut local_k, mut local_count), (s1, s2)| {
            if ferric_core::INTERRUPT.load(std::sync::atomic::Ordering::Relaxed) {
                return (engine, local_j, local_k, local_count);
            }
            let b12 = bounds.q[(s1, s2)];
            let d12 = d_max_shell[(s1, s2)];
            let (n1, n2) = (dims[s1], dims[s2]);
            let (o1, o2) = (offs[s1], offs[s2]);
            let sym12 = s1 != s2;

            for s3 in 0..=s1 {
                if s3 % 100 == 0 && ferric_core::INTERRUPT.load(Ordering::Relaxed) {
                    return (engine, local_j, local_k, local_count);
                }
                let s4max = if s3 == s1 { s2 } else { s3 };
                let d13 = d_max_shell[(s1, s3)];
                let d23 = d_max_shell[(s2, s3)];
                for s4 in 0..=s4max {
                    let b34 = bounds.q[(s3, s4)];
                    let d34 = d_max_shell[(s3, s4)];
                    let d14 = d_max_shell[(s1, s4)];
                    let d24 = d_max_shell[(s2, s4)];
                    let dmax = d12.max(d34).max(d13).max(d14).max(d23).max(d24);
                    if b12 * b34 * dmax < thresh {
                        continue;
                    }

                    if let Some(q) = engine.compute_quartet(prep, s1, s2, s3, s4) {
                        local_count += 1;
                        let (n3, n4) = (dims[s3], dims[s4]);
                        let (o3, o4) = (offs[s3], offs[s4]);
                        let sym34 = s3 != s4;
                        let sym1234 = (s1, s2) != (s3, s4);

                        // Fast-path for STO-3G / small shells (n=1)
                        if n1 == 1 && n2 == 1 && n3 == 1 && n4 == 1 {
                            let v = unsafe { *q.get_unchecked(0) };
                            unsafe {
                                *local_j.uget_mut((o1, o2)) += d.uget((o3, o4)) * v;
                                *local_k.uget_mut((o1, o3)) += d.uget((o2, o4)) * v;
                                if sym12 {
                                    *local_j.uget_mut((o2, o1)) += d.uget((o3, o4)) * v;
                                    *local_k.uget_mut((o2, o3)) += d.uget((o1, o4)) * v;
                                }
                                if sym34 {
                                    *local_j.uget_mut((o1, o2)) += d.uget((o4, o3)) * v;
                                    *local_k.uget_mut((o1, o4)) += d.uget((o2, o3)) * v;
                                }
                                if sym12 && sym34 {
                                    *local_j.uget_mut((o2, o1)) += d.uget((o4, o3)) * v;
                                    *local_k.uget_mut((o2, o4)) += d.uget((o1, o3)) * v;
                                }
                                if sym1234 {
                                    *local_j.uget_mut((o3, o4)) += d.uget((o1, o2)) * v;
                                    *local_k.uget_mut((o3, o1)) += d.uget((o4, o2)) * v;
                                    if sym12 {
                                        *local_j.uget_mut((o3, o4)) += d.uget((o2, o1)) * v;
                                        *local_k.uget_mut((o3, o2)) += d.uget((o4, o1)) * v;
                                    }
                                    if sym34 {
                                        *local_j.uget_mut((o4, o3)) += d.uget((o1, o2)) * v;
                                        *local_k.uget_mut((o4, o1)) += d.uget((o3, o2)) * v;
                                    }
                                    if sym12 && sym34 {
                                        *local_j.uget_mut((o4, o3)) += d.uget((o2, o1)) * v;
                                        *local_k.uget_mut((o4, o2)) += d.uget((o3, o1)) * v;
                                    }
                                }
                            }
                            continue;
                        }

                        // General path for larger shells
                        for a in 0..n1 {
                            for b in 0..n2 {
                                for c in 0..n3 {
                                    for dd in 0..n4 {
                                        let v = unsafe { *q.get_unchecked(((a * n2 + b) * n3 + c) * n4 + dd) };
                                        let mu = o1 + a;
                                        let nu = o2 + b;
                                        let la = o3 + c;
                                        let sg = o4 + dd;

                                        unsafe {
                                            *local_j.uget_mut((mu, nu)) += d.uget((la, sg)) * v;
                                            *local_k.uget_mut((mu, la)) += d.uget((nu, sg)) * v;

                                            if sym12 {
                                                *local_j.uget_mut((nu, mu)) += d.uget((la, sg)) * v;
                                                *local_k.uget_mut((nu, la)) += d.uget((mu, sg)) * v;
                                            }
                                            if sym34 {
                                                *local_j.uget_mut((mu, nu)) += d.uget((sg, la)) * v;
                                                *local_k.uget_mut((mu, sg)) += d.uget((nu, la)) * v;
                                            }
                                            if sym12 && sym34 {
                                                *local_j.uget_mut((nu, mu)) += d.uget((sg, la)) * v;
                                                *local_k.uget_mut((nu, sg)) += d.uget((mu, la)) * v;
                                            }
                                            if sym1234 {
                                                *local_j.uget_mut((la, sg)) += d.uget((mu, nu)) * v;
                                                *local_k.uget_mut((la, mu)) += d.uget((sg, nu)) * v;
                                                if sym12 {
                                                    *local_j.uget_mut((la, sg)) += d.uget((nu, mu)) * v;
                                                    *local_k.uget_mut((la, nu)) += d.uget((sg, mu)) * v;
                                                }
                                                if sym34 {
                                                    *local_j.uget_mut((sg, la)) += d.uget((mu, nu)) * v;
                                                    *local_k.uget_mut((sg, mu)) += d.uget((la, nu)) * v;
                                                }
                                                if sym12 && sym34 {
                                                    *local_j.uget_mut((sg, la)) += d.uget((nu, mu)) * v;
                                                    *local_k.uget_mut((sg, nu)) += d.uget((la, mu)) * v;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (engine, local_j, local_k, local_count)
        }
    ).map(|(_, j, k, count)| {
        computed_quartets.fetch_add(count, Ordering::Relaxed);
        (j, k)
    }).reduce(
        || (Array2::zeros((nbf, nbf)), Array2::zeros((nbf, nbf))),
        |mut acc, next| {
            acc.0 += &next.0;
            acc.1 += &next.1;
            acc
        }
    );

    *j += &total_jk.0;
    *k += &total_jk.1;

    #[cfg(feature = "mpi")]
    if let Some(world) = &ctx.world {
        let mut j_global = Array2::zeros(j.dim());
        let mut k_global = Array2::zeros(k.dim());
        world.all_reduce_into(j.as_slice().unwrap(), j_global.as_slice_mut().unwrap(), mpi::collective::SystemOperation::sum());
        world.all_reduce_into(k.as_slice().unwrap(), k_global.as_slice_mut().unwrap(), mpi::collective::SystemOperation::sum());
        *j = j_global;
        *k = k_global;
    }

    Ok(computed_quartets.load(Ordering::SeqCst))
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
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;

    fn run_rhf_test(xyz: &str, basis_name: &str, ref_slug: &str, tol: f64) {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig {
            energy_conv: 1e-12,
            density_conv: 1e-10,
            integral_thresh: 1e-14,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(result.converged, "RHF did not converge");
        eprintln!(
            "{ref_slug}: energy={:.12}, iters={}, vnn={:.12}",
            result.energy,
            result.iterations,
            mol.nuclear_repulsion()
        );
        let ref_path = format!("../../testdata/reference/{ref_slug}");
        if let Ok(text) = std::fs::read_to_string(&ref_path) {
            let ref_data: serde_json::Value = serde_json::from_str(&text).unwrap();
            let ref_energy = ref_data["energy"].as_f64().unwrap();
            assert!(
                (result.energy - ref_energy).abs() < tol,
                "{ref_slug}: got {:.10}, ref {:.10}",
                result.energy,
                ref_energy
            );
        }
    }

    #[test]
    fn test_rhf_h2_sto3g() {
        run_rhf_test(
            "2\nH2\nH 0 0 0\nH 0 0 0.74\n",
            "sto-3g",
            "h2_sto-3g_rhf.json",
            1e-8,
        );
    }

    #[test]
    fn test_rhf_h2o_sto3g() {
        // Tolerance 5e-8 due to libint2 vs libcint integral differences
        run_rhf_test(
            "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
            "sto-3g",
            "h2o_sto-3g_rhf.json",
            5e-8,
        );
    }

    #[test]
    fn test_rhf_h2o_631g() {
        run_rhf_test(
            "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
            "6-31g",
            "h2o_6-31g_rhf.json",
            1e-8,
        );
    }
}
