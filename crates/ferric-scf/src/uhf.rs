//! Unrestricted Hartree-Fock (UHF) solver.
//!
//! Parallels `rhf.rs` but tracks independent α/β densities, Fock matrices, and
//! DIIS streams. Uses J built from D_total = D_α + D_β and K built per spin.

use crate::diis::Diis;
use crate::direct_j::DirectJ;
use crate::direct_k::DirectK;
use crate::fock::{JBuilder, KBuilder};
use crate::guess::hcore_guess;
use crate::result::{ScfExit, ScfResult, Spin};
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;

/// UHF configuration mirrors RHF (separate type for forward extensibility).
pub type UhfConfig = RhfConfig;

/// In-place modifier applied to the (α, β) Fock matrices each SCF iteration —
/// the hook cDFT uses to add its constraint potential. `None` for plain UHF.
pub type UhfFockMod<'a> = &'a dyn Fn(&mut Array2<f64>, &mut Array2<f64>);

/// Solve unrestricted Hartree-Fock equations for a molecule.
///
/// Uses `mol.charge` and `mol.multiplicity` to determine α/β electron counts.
/// The initial guess is built from a single hcore diagonalization; symmetry is
/// broken by occupying fewer β orbitals than α (or by a small HOMO/LUMO mixing
/// when nocc_a == nocc_b).
pub fn solve_uhf(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &UhfConfig,
) -> Result<ScfResult, FerricError> {
    solve_uhf_with_guess(ctx, mol, prep, bounds, config, None)
}

/// UHF with optional caller-supplied initial MOs.
///
/// `initial_mos` lets the caller provide a directed starting point (e.g.
/// neutral RHF MOs for a cation calculation, to avoid landing in a
/// doublet-excited basin from the symmetric hcore guess). Pass `None`
/// for the default hcore guess.
///
/// The provided `c_a`/`c_b` must have shape (nbasis, nbasis) and span
/// the AO basis; only the first `nocc_α`/`nocc_β` columns are used as
/// the occupied set.
pub fn solve_uhf_with_guess(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &UhfConfig,
    initial_mos: Option<(&Array2<f64>, &Array2<f64>)>,
) -> Result<ScfResult, FerricError> {
    solve_uhf_fockmod(ctx, mol, prep, bounds, config, initial_mos, None)
}

/// UHF with an optional per-iteration Fock modifier.
///
/// `fock_mod`, if given, is called as `fock_mod(&mut f_a, &mut f_b)` each
/// iteration immediately after the XC potential is added and before the DIIS
/// error is formed, so the added potential is part of the converged Fock and
/// of the DIIS condition. `None` reproduces ordinary UHF/UKS exactly. cDFT uses
/// this to add Σ_C λ_C W^C.
///
/// The two-electron operator (Coulomb or an attenuated erf/erfc kernel for
/// short-range correlation) is taken from `bounds.op`: the J/K builders need
/// both the operator and its matching Schwarz screening table, and the bounds
/// carry both. There is therefore no separate `op` argument — `bounds` is the
/// single source of truth, which makes an operator/screening mismatch
/// unrepresentable.
pub fn solve_uhf_fockmod(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &UhfConfig,
    initial_mos: Option<(&Array2<f64>, &Array2<f64>)>,
    fock_mod: Option<UhfFockMod>,
) -> Result<ScfResult, FerricError> {
    use ferric_dft::ks::KsXcUks;
    use ferric_dft::xc_trait::{KMix, UksXcContribution};

    // Build UKS XC contribution once. None for pure UHF.
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

    // Meta-GGA (SCAN / r2SCAN) UKS is stiffer than LDA/GGA and limit-cycles under
    // plain DIIS; apply the same modest default virtual-block level shift the
    // closed-shell RKS path uses (see solve_rhf) when the user hasn't set one.
    // Ramped to zero as the gradient converges, so the final energy is
    // unperturbed. LDA/GGA/hybrid keep the user's shift (default 0).
    let effective_level_shift = crate::driver::effective_level_shift(config);

    // K coefficient on the full Coulomb K, per-spin. For pure HF (no xc),
    // KMix::default() gives sr = 1.0, lr = 1.0 → c_k = 1.0 (original UHF behavior).
    let c_k: f64 = if xc_contrib.is_some() { k_mix.sr } else { 1.0 };

    // RSH path (ω > 0): build per-spin K from c_SR · K[erfc(ω)] + c_LR · K[erf(ω)]
    // via two DfK fitters. Each fitter's geometry-only B[P,μ,ν] tensor is built
    // once; only the D-dependent contraction runs per iteration per spin.
    // Shared geometry-only environment: S, hcore(+external+ECP — V_ECP folds
    // into hcore once, byte-identical to plain hcore for all-electron bases),
    // V_nn(+external), COSMO/PCM contexts, resolved memory budget, RSH
    // fitters. One construction serving all six SCF variants (crate::driver).
    let crate::driver::ScfEnv { s, h, vnn, ooc_budget, cosmo_cavity, pcm_ctx, mut dfk_sr, mut dfk_lr } =
        crate::driver::prepare(ctx, mol, prep, config, &k_mix)?;
    let n = prep.nbasis();
    let nelec = mol.nelec() as i64;
    let mult = mol.multiplicity as i64;
    if mult < 1 {
        return Err(FerricError::General(
            "UHF: multiplicity must be >= 1".into(),
        ));
    }
    let two_s = mult - 1; // 2S
    if (nelec - two_s) % 2 != 0 || nelec < two_s {
        return Err(FerricError::General(format!(
            "UHF: incompatible nelec={nelec} and multiplicity={mult}"
        )));
    }
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;
    if nocc_a + nocc_b != nelec as usize {
        return Err(FerricError::General(
            "UHF: nocc_a + nocc_b != nelec".into(),
        ));
    }
    if nocc_b > nocc_a {
        return Err(FerricError::General("UHF: nocc_b > nocc_a".into()));
    }
    // Canonical orthogonalizer X (n × m): drops eigenvectors of S below
    // LINDEP_THRESH to handle near-linear-dependent basis sets (Na clusters in
    // aug-cc-pVDZ). m == n for well-conditioned S. See crate::rhf for details.
    let x = crate::rhf::canonical_orthogonalizer(&s)?;

    // Initial guess: caller-supplied MOs if provided, else hcore.
    let (mut c_a, mut c_b) = if let Some((ca0, cb0)) = initial_mos {
        if ca0.dim() != (n, n) || cb0.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "solve_uhf_with_guess: initial MO shape mismatch (got {:?}/{:?}, want ({n},{n}))",
                ca0.dim(), cb0.dim()
            )));
        }
        (ca0.clone(), cb0.clone())
    } else {
        // hcore guess: get MO coefficients from H' = Xᵀ H X (canonical-orthog).
        // diagonalize() pads to (n × n), keeping the guess shape historical.
        let _ = hcore_guess(&s, &h, nocc_a.max(1))?; // sanity check it succeeds
        let (_, c) = diagonalize(&h, &x)?;
        (c.clone(), c)
    };

    // For genuine open shell (nocc_a > nocc_b), occupying the lowest nocc_σ
    // orbitals per spin is already symmetry-broken. For "forced" UHF on a
    // closed-shell, mix HOMO/LUMO in β with a small angle to break symmetry.
    if nocc_a == nocc_b && nocc_a > 0 && nocc_a < n {
        let theta = 0.1f64;
        let (cs, sn) = (theta.cos(), theta.sin());
        let homo = nocc_b - 1;
        let lumo = nocc_b;
        for mu in 0..n {
            let h_val = c_b[(mu, homo)];
            let l_val = c_b[(mu, lumo)];
            c_b[(mu, homo)] = cs * h_val + sn * l_val;
            c_b[(mu, lumo)] = -sn * h_val + cs * l_val;
        }
    }

    let mut d_a = density(&c_a, nocc_a);
    let mut d_b = density(&c_b, nocc_b);

    let mut j_buf = Array2::<f64>::zeros((n, n));
    let mut k_a_buf = Array2::<f64>::zeros((n, n));
    let mut k_b_buf = Array2::<f64>::zeros((n, n));

    // Coupled α/β DIIS — single subspace, joint error norm. PySCF-style.
    // Independent per-spin DIIS desyncs α and β on cations (e.g. H2O+ took
    // 421 iterations to converge with independent DIIS; coupled converges
    // in ~15-25 cycles).
    let mut diis = Diis::new(config.diis_size);
    // Convergence bookkeeping (prev energy, ΔP over the TOTAL α+β density,
    // divergence streak, stall history) — shared driver::ScfMonitor; ΔP is
    // INFINITY until the first rebuild (see rhf::scf_converged).
    let mut mon = crate::driver::ScfMonitor::new();
    let mut total_quartets = 0usize;

    // Maximum-Overlap Method (MOM) references: the previously-accepted occupied
    // α/β MO blocks. From iter `mom_after_iter + 1` onward we pick each spin's
    // occupied set by AO-overlap with these (Gilbert-Besley-Gill), instead of
    // pure aufbau. This pins the open-shell occupation through SCF and converges
    // open-shell atoms (e.g. S/O ³P) whose near-degenerate p-shell otherwise
    // makes plain DIIS oscillate forever. Empty open block for UHF (each spin is
    // a pure closed set). Mirrors the rohf.rs MOM wiring.
    let mut mom_ref_a: Option<Array2<f64>> = None;
    let mut mom_ref_b: Option<Array2<f64>> = None;
    let empty_open: Array2<f64> = Array2::<f64>::zeros((n, 0));

    // K built per spin:
    //   * RSH (ω > 0): K_σ = c_SR · K_SR[D_σ] + c_LR · K_LR[D_σ] via DfK
    //   * Plain hybrid / pure HF (ω = 0): K_σ = c_K · K[D_σ] via DirectK, or DfK
    //     when config.df_k_aux is set (density-fitted exchange).
    //   * Pure DFT (c_K = 0): K skipped
    // J (ω = 0 path only — RSH doesn't touch J here) is DirectJ, or DfJ when
    // config.df_j_aux is set (density-fitted Coulomb).
    // Builders hoisted out of the loop: each lazily builds a per-thread libint2
    // EnginePool on first use (ctors serialized behind a global mutex), so a
    // loop-local builder would pay that construction every iteration.
    let need_k = c_k != 0.0 || k_mix.omega > 0.0;
    let coulomb_op = bounds.op;
    // Effective aux names under the existing gates (ω=0 for both; DF-K
    // additionally needs `need_k`) — `None` reproduces the prior "skip this
    // builder entirely" branch exactly. `build_df_jk` shares one
    // `PreparedBasis` when both names are set and identical (see its doc);
    // it independently gates each output on its own `Option`, so this is
    // byte-identical to the previous two-independent-`if` structure.
    let j_aux_eff = if k_mix.omega == 0.0 { config.df_j_aux.as_deref() } else { None };
    let k_aux_eff = if need_k && k_mix.omega == 0.0 { config.df_k_aux.as_deref() } else { None };
    let (mut df_j, mut df_k) = crate::fock_assembly::build_df_jk(
        ctx, mol, coulomb_op, prep, j_aux_eff, k_aux_eff, ooc_budget,
    )?;
    let mut direct_j: Option<DirectJ> = if df_j.is_none() {
        Some(DirectJ::new(ctx, prep, bounds, config.integral_thresh, ooc_budget))
    } else {
        None
    };
    let mut direct_k: Option<DirectK> = if need_k && k_mix.omega == 0.0 && df_k.is_none() {
        Some(DirectK::new(ctx, prep, bounds, config.integral_thresh, ooc_budget))
    } else {
        None
    };

    for iter in 1..=config.max_iter {
        ctx.check_interrupted()?;
        j_buf.fill(0.0);
        k_a_buf.fill(0.0);
        k_b_buf.fill(0.0);
        let d_total = &d_a + &d_b;

        // J built from total density (one call). DF-J if configured, else direct.
        if let Some(dfj) = df_j.as_mut() {
            dfj.build(&d_total, &mut j_buf)?;
        } else {
            let dj = direct_j.as_mut().expect("DirectJ built before loop");
            total_quartets += dj.build(&d_total, &mut j_buf)?;
        }
        // F_σ = H + J − K_σ_total  (then + V_xc^σ below for UKS path),
        // assembled in place — no k_σ_total clones for HF/plain hybrids and no
        // zeros allocations for pure DFT. `f_σ += (−c)·K` is bit-identical to
        // the former `f_σ −= c·&K` clone path (sign flip is exact); the RSH
        // branch keeps the explicit SR/LR combination for the same reason.
        let mut f_a: Array2<f64> = &h + &j_buf;
        let mut f_b: Array2<f64> = &h + &j_buf;
        if k_mix.omega > 0.0 {
            let dfk_sr = dfk_sr.as_mut().expect("dfk_sr built when omega>0");
            let dfk_lr = dfk_lr.as_mut().expect("dfk_lr built when omega>0");
            crate::fock_assembly::subtract_rsh_exchange(
                dfk_sr, dfk_lr, &d_a, &mut f_a, k_mix.sr, k_mix.lr, 1.0,
            )?;
            crate::fock_assembly::subtract_rsh_exchange(
                dfk_sr, dfk_lr, &d_b, &mut f_b, k_mix.sr, k_mix.lr, 1.0,
            )?;
        } else if need_k {
            if let Some(dfk) = df_k.as_mut() {
                dfk.build(&d_a, &mut k_a_buf)?;
                dfk.build(&d_b, &mut k_b_buf)?;
            } else {
                let dk = direct_k.as_mut().expect("DirectK built before loop");
                total_quartets += <DirectK as KBuilder>::build(dk, &d_a, &mut k_a_buf)?;
                total_quartets += <DirectK as KBuilder>::build(dk, &d_b, &mut k_b_buf)?;
            }
            f_a.scaled_add(-c_k, &k_a_buf);
            f_b.scaled_add(-c_k, &k_b_buf);
        }

        // Electronic energy BEFORE adding V_xc (V_xc is one-body in F_σ but
        // E_xc is its own integral).
        let e_elec_no_xc: f64 =
            0.5 * ((&(&h + &f_a) * &d_a).sum() + (&(&h + &f_b) * &d_b).sum());
        let e_xc = if let Some(x) = xc_contrib.as_ref() {
            x.add_xc_uks(&d_a, &d_b, &mut f_a, &mut f_b)
        } else {
            0.0
        };
        // cDFT (or any external) Fock modifier: add a fixed AO potential to
        // both spin Focks before DIIS sees them. The constraint energy term is
        // accounted for by the outer driver, not here, so `energy` below is the
        // ordinary KS energy at the current (constrained) density.
        if let Some(fm) = fock_mod {
            fm(&mut f_a, &mut f_b);
        }

        // COSMO reaction field: built from the TOTAL density (D_a + D_b),
        // recomputed every iteration (see rhf::solve_rhf / crate::cosmo for
        // the full derivation). The reaction-field potential is spin-
        // independent (a classical electrostatic term), so it is added
        // identically to both spin Focks. Its energy is added directly to
        // `energy`, not via the e_elec_no_xc trace (matches PySCF's
        // e_solvent-added-on-top convention).
        let (e_cosmo, e_pcm) = crate::driver::solvent_terms(
            mol,
            prep,
            config,
            cosmo_cavity.as_ref(),
            pcm_ctx.as_ref(),
            &d_total,
            &mut [&mut f_a, &mut f_b],
        )?;

        let energy = e_elec_no_xc + e_xc + e_cosmo + e_pcm + vnn;

        // DIIS errors per spin: F_σ D_σ S − S D_σ F_σ
        let err_a = f_a.dot(&d_a).dot(&s) - s.dot(&d_a).dot(&f_a);
        let err_b = f_b.dot(&d_b).dot(&s) - s.dot(&d_b).dot(&f_b);

        let sig = mon.signals(energy);
        let de = sig.de;
        let err_max_a = err_a.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let err_max_b = err_b.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let err_max = err_max_a.max(err_max_b);

        if crate::rhf::scf_trace() {
            eprintln!(
                "UHF iter={iter:4}  E={energy:.12}  dE={de:.3e}  \
                 dp_rms={:.3e}  dp_max={:.3e}  err_max={err_max:.3e}",
                mon.dp_rms, mon.dp_max
            );
        }

        // Live per-iteration progress (see solve_rhf's identical block for the
        // full rationale). STDOUT, opt-in via `config.verbose`, rank-0-only.
        if config.verbose && ctx.is_root() {
            println!(
                "UHF iter={iter:4}  E={energy:.10}  dE={de:.3e}  dp_rms={:.3e}  err_max={err_max:.3e}",
                mon.dp_rms
            );
        }

        // Convergence: energy + total-density change (ΔP), the same
        // ORCA/PySCF gate as solve_rhf — NOT the DIIS commutator, which parks on
        // the naux-dependent RI noise floor and never drains. See
        // rhf::scf_converged. This replaces the old df_noise_floor_ok hack; the
        // `df_active` distinction is gone (ΔP handles DF and direct uniformly).
        let conv_exit = crate::rhf::scf_converged(sig, config.energy_conv, config.density_conv);

        // Divergence / stall early exits (shared driver::ScfMonitor; both are
        // no-ops at the None defaults — UHF previously ignored these knobs).
        if mon.diverging(energy, config.divergence_tol) || mon.stalled(err_max, config.stall_window) {
            return Err(FerricError::ScfConvergence {
                iterations: iter,
                last_energy: mon.prev_e,
            });
        }

        if iter > 1 && conv_exit.is_some() {
            let (eps_a, c_a_f) = diagonalize(&f_a, &x)?;
            let (eps_b, c_b_f) = diagonalize(&f_b, &x)?;
            // ⟨S²⟩ diagnostic
            let s2 = expectation_s_squared(&c_a_f, &c_b_f, &s, nocc_a, nocc_b);
            let s_true = 0.5 * (nocc_a as f64 - nocc_b as f64);
            let s_ideal = s_true * (s_true + 1.0);
            if s2 > s_ideal + 0.1 {
                eprintln!(
                    "UHF warning: spin contamination ⟨S²⟩ = {:.4} (ideal {:.4})",
                    s2, s_ideal
                );
            }
            let density_total = &d_a + &d_b;
            return Ok(ScfResult {
                spin: Spin::Unrestricted,
                energy,
                density_total,
                density_alpha: d_a,
                density_beta: Some(d_b),
                mos_alpha: c_a_f,
                mos_beta: Some(c_b_f),
                eps_alpha: eps_a,
                eps_beta: Some(eps_b),
                fock_alpha: f_a,
                fock_beta: Some(f_b),
                converged: true,
                exit: ScfExit::Converged,
                iterations: iter,
                computed_quartets: total_quartets,
            });
        }
        mon.note_energy(energy);

        // ── Second-order (Newton) update, UHF/UKS ────────────────────────────
        // When enabled (newton_trigger > 0) and err_max has dropped below the
        // trigger, take a damped-Newton step on the α/β orbital rotations
        // instead of DIIS. For UKS this uses the SAME LDA/GGA f_xc kernel the
        // ROKS Newton path uses (via FxcKernelStore), so PBE/B3LYP/etc. UKS now
        // gets real second-order acceleration, not just LDA. Gated to the
        // non-RSH case (ω = 0): the Newton matvec's K comes from the plain
        // Coulomb `build_jk`, so range-separated K would be inconsistent — RSH
        // keeps the DIIS path.
        // Meta-GGA (SCAN / r2SCAN) has no τ-dependent f_xc kernel in Phase A —
        // exclude it from the Newton path so it falls back to DIIS (energy-only).
        let use_newton = config.newton_trigger > 0.0
            && iter > 3
            && err_max < config.newton_trigger
            && k_mix.omega == 0.0
            && !crate::rohf::xc_is_metagga(config.xc.as_deref());
        if use_newton {
            let f_a_mo = c_a.t().dot(&f_a).dot(&c_a);
            let f_b_mo = c_b.t().dot(&f_b).dot(&c_b);

            // Build the f_xc kernel (LDA or GGA) + reference density once per
            // Newton step (None for pure UHF). The response closure borrows it.
            let fxc_store = if xc_contrib.is_some() {
                let main = config.dft_grid.clone().unwrap_or_default();
                let name = config.xc.as_deref().expect("xc_contrib implies Some(xc)");
                Some(crate::rohf::FxcKernelStore::build(mol, prep, &main, name, &d_a, &d_b)?)
            } else {
                None
            };
            let fxc_storage = fxc_store.as_ref().map(|s| s.response());
            let fxc_ref: Option<&crate::rohf_newton::FxcResponse<'_>> = fxc_storage.as_deref();

            let inputs = crate::uhf_newton::UhfNewtonInputs {
                prep,
                bounds,
                c_a: &c_a,
                c_b: &c_b,
                f_a_mo: &f_a_mo,
                f_b_mo: &f_b_mo,
                nocc_a,
                nocc_b,
                k_mix_sr: if xc_contrib.is_some() { c_k } else { 1.0 },
                fxc: fxc_ref,
                thresh: config.integral_thresh,
                ooc_budget,
            };
            let (c_a_n, c_b_n, _kmax) = crate::uhf_newton::uhf_newton_step(
                ctx,
                &inputs,
                config.level_shift.max(1e-6),
                0.2, // trust radius
                20,
                1e-7,
            )?;
            c_a = c_a_n;
            c_b = c_b_n;
            let d_tot_old = &d_a + &d_b;
            d_a = density(&c_a, nocc_a);
            d_b = density(&c_b, nocc_b);
            let d_tot_new = &d_a + &d_b;
            mon.record_density_change(&d_tot_new, &d_tot_old);
            continue;
        }

        // Coupled DIIS extrapolation: one set of coefficients applied to
        // both spin Fock histories.
        let (mut f_a_new, mut f_b_new) = diis.step_pair(&f_a, &f_b, &err_a, &err_b);
        // Optional level shift on each spin's virtual–virtual block, applied
        // after DIIS so the DIIS error / convergence criterion remains
        // anchored to the unshifted Fock. Rational-damped by err_max so the
        // converged Fock is the unshifted stationary point (see solve_rohf
        // for the same formula and rationale).
        if effective_level_shift > 0.0 && iter > 1 {
            const SHIFT_DAMP_ERR: f64 = 1e-3;
            let err_max = err_a
                .iter()
                .chain(err_b.iter())
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max);
            let damp = err_max / (err_max + SHIFT_DAMP_ERR);
            let shift_eff = effective_level_shift * damp;
            if shift_eff > 1e-10 {
                let c_av = c_a.slice(ndarray::s![.., nocc_a..]);
                let c_bv = c_b.slice(ndarray::s![.., nocc_b..]);
                let p_av: Array2<f64> = c_av.dot(&c_av.t());
                let p_bv: Array2<f64> = c_bv.dot(&c_bv.t());
                f_a_new += &(shift_eff * s.dot(&p_av).dot(&s));
                f_b_new += &(shift_eff * s.dot(&p_bv).dot(&s));
            }
        }
        let (eps_a_new, mut c_a_new) = diagonalize(&f_a_new, &x)?;
        let (eps_b_new, mut c_b_new) = diagonalize(&f_b_new, &x)?;

        // MOM occupied-orbital selection (per spin) from iter mom_after_iter+1.
        if config.mom_after_iter > 0 && iter > config.mom_after_iter {
            if let Some(ref_a) = mom_ref_a.as_ref() {
                if nocc_a > 0 {
                    c_a_new =
                        crate::mom::mom_reorder(&c_a_new, &s, ref_a, &empty_open, nocc_a, 0);
                }
            }
            if let Some(ref_b) = mom_ref_b.as_ref() {
                if nocc_b > 0 {
                    c_b_new =
                        crate::mom::mom_reorder(&c_b_new, &s, ref_b, &empty_open, nocc_b, 0);
                }
            }
        }
        c_a = c_a_new;
        c_b = c_b_new;
        // Update MOM references to the (possibly reordered) occupied blocks, so
        // the next iter pins against the most recent accepted occupation.
        if config.mom_after_iter > 0 && iter >= config.mom_after_iter {
            mom_ref_a = Some(c_a.slice(ndarray::s![.., ..nocc_a]).to_owned());
            mom_ref_b = Some(c_b.slice(ndarray::s![.., ..nocc_b]).to_owned());
        }
        let d_tot_old = &d_a + &d_b;
        if config.fractional_occ {
            d_a = density_fractional(&c_a, &eps_a_new, nocc_a);
            d_b = density_fractional(&c_b, &eps_b_new, nocc_b);
        } else {
            d_a = density(&c_a, nocc_a);
            d_b = density(&c_b, nocc_b);
        }
        // ΔP over the total density (α+β) — consumed at the top of the next
        // iteration by rhf::scf_converged. Drains to zero at the RI fixed point
        // even when the DIIS commutator parks on the naux noise floor.
        let d_tot_new = &d_a + &d_b;
        mon.record_density_change(&d_tot_new, &d_tot_old);
    }
    Err(FerricError::ScfConvergence {
        iterations: config.max_iter,
        last_energy: mon.prev_e,
    })
}

fn density(c: &Array2<f64>, nocc: usize) -> Array2<f64> {
    let n = c.nrows();
    if nocc == 0 {
        return Array2::zeros((n, n));
    }
    let c_occ = c.slice(ndarray::s![.., ..nocc]);
    c_occ.dot(&c_occ.t())
}

/// Density with fixed fractional (ensemble) occupation of a degenerate frontier
/// shell, for one spin channel: `D = Σ_p f_p c_p c_pᵀ`.
///
/// Builds integer occupation `f = 1` for orbitals fully below the frontier, then
/// detects the group of orbitals near-degenerate with the HOMO (energies within
/// `EPS_TOL` of `eps[nocc-1]`) that straddle the occupation boundary, and spreads
/// the remaining electrons of that shell *equally* across the whole group. For a
/// ³P atom this puts 2/3 of an electron in each of the three degenerate p
/// orbitals, restoring spherical symmetry so the GGA potential stops oscillating.
///
/// Falls back to plain integer `density()` when the HOMO is non-degenerate (the
/// common case), so it is a no-op for ordinary molecules.
fn density_fractional(c: &Array2<f64>, eps: &[f64], nocc: usize) -> Array2<f64> {
    let n = c.nrows();
    if nocc == 0 {
        return Array2::zeros((n, n));
    }
    if nocc >= n {
        return density(c, nocc);
    }
    // Tolerance must be loose enough to capture a frontier shell that the
    // *oscillating* SCF has artificially split. For a free ³P atom the three p
    // orbitals can be ~0.01–0.02 Ha apart mid-oscillation, so a tight 1e-3 tol
    // catches only 2 of 3 and the fix fails. 0.05 Ha reliably groups them; this
    // path is opt-in (free atoms only), so a loose tol cannot affect molecules.
    const EPS_TOL: f64 = 0.05;
    let e_homo = eps[nocc - 1];
    // Grow the group around BOTH the HOMO (nocc-1) and the LUMO (nocc): the
    // degenerate frontier shell straddles the occupation boundary.
    let mut lo = nocc - 1;
    while lo > 0 && (eps[lo - 1] - e_homo).abs() < EPS_TOL {
        lo -= 1;
    }
    // Extend upward from the LUMO using the LUMO energy as the anchor (it may
    // differ from the HOMO by the artificial split, but be within EPS_TOL).
    let e_lumo = eps[nocc];
    let mut hi = nocc; // start at LUMO
    while hi + 1 < n && (eps[hi + 1] - e_lumo).abs() < EPS_TOL {
        hi += 1;
    }
    // Only act if HOMO and LUMO are within tol (genuine straddling degeneracy).
    if (e_lumo - e_homo).abs() >= EPS_TOL {
        return density(c, nocc);
    }
    let group_size = hi - lo + 1;
    if group_size <= 1 {
        return density(c, nocc);
    }
    // Electrons to distribute over the group = (occupied orbitals in group).
    let n_in_group_occupied = nocc - lo; // how many of the group are below the boundary
    let frac = n_in_group_occupied as f64 / group_size as f64;

    let mut d = Array2::<f64>::zeros((n, n));
    // Fully-occupied orbitals below the degenerate group: f = 1.
    if lo > 0 {
        let c_core = c.slice(ndarray::s![.., ..lo]);
        d = c_core.dot(&c_core.t());
    }
    // Degenerate group: f = frac each.
    for p in lo..=hi {
        let cp = c.slice(ndarray::s![.., p]);
        let outer = {
            let col = cp.to_owned();
            let mut m = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    m[(i, j)] = frac * col[i] * col[j];
                }
            }
            m
        };
        d += &outer;
    }
    d
}

/// Diagonalize F in the canonical-orthogonal basis X (n × m), padding the
/// result back to (n × n) with sentinel-energy zero columns for any dropped
/// near-singular modes. Mirrors `crate::rhf::diagonalize`. See that function for
/// the shape/padding rationale.
fn diagonalize(
    f: &Array2<f64>,
    x: &Array2<f64>,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    crate::driver::diagonalize_rect(f, x)
}

/// ⟨S²⟩ for a UHF determinant:
/// ⟨S²⟩ = S(S+1) + N_β − Σ_{i∈α-occ, j∈β-occ} |⟨α_i|β_j⟩|²
fn expectation_s_squared(
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    s: &Array2<f64>,
    nocc_a: usize,
    nocc_b: usize,
) -> f64 {
    let s_true = 0.5 * (nocc_a as f64 - nocc_b as f64);
    let s_ideal = s_true * (s_true + 1.0);
    if nocc_a == 0 || nocc_b == 0 {
        return s_ideal;
    }
    let c_a_occ = c_a.slice(ndarray::s![.., ..nocc_a]);
    let c_b_occ = c_b.slice(ndarray::s![.., ..nocc_b]);
    // overlap_ab[i,j] = (C_α^T S C_β)[i,j]
    let overlap_ab = c_a_occ.t().dot(s).dot(&c_b_occ);
    let sum_sq: f64 = overlap_ab.iter().map(|v| v * v).sum();
    s_ideal + (nocc_b as f64) - sum_sq
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_integrals::oneelectron;
    use ferric_core::basis;
    use ferric_core::external_potential::{ExternalPotential, PointCharge};
    use ferric_core::parallel::ParallelContext;

    #[test]
    fn test_uhf_h_atom_sto3g() {
        // Single H atom, doublet. Energy = -0.466581 in STO-3G (one electron, no e-e).
        let mol = Molecule::parse_xyz("1\nH\nH 0 0 0\n", 0, 2).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = ferric_integrals::operator::Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let cfg = UhfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-9,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
        assert!(res.converged);
        // STO-3G H atom: H = -0.46658185 (one electron, -ζ_1s).
        assert!(
            (res.energy + 0.466581850).abs() < 1e-5,
            "H atom energy = {}",
            res.energy
        );
        // ⟨S²⟩ exact = 0.75 for doublet, single electron.
        let s2 = expectation_s_squared(
            &res.mos_alpha,
            res.mos_beta.as_ref().unwrap(),
            &oneelectron::overlap(&prep),
            1,
            0,
        );
        assert!((s2 - 0.75).abs() < 1e-10, "⟨S²⟩ = {}", s2);
    }

    #[test]
    fn test_uhf_oxygen_atom_mom_converges() {
        // Oxygen atom ground state is ³P (triplet): nα=5, nβ=3. The three
        // near-degenerate 2p orbitals make the open-shell occupation ambiguous,
        // and plain aufbau DIIS can oscillate on which p is the SOMO. MOM
        // (mom_after_iter) pins the occupation by AO-overlap and converges it.
        // This is the regression for "UHF had no MOM" — the S/O free-atom solves
        // in the C6 TS path failed without it.
        let mol = Molecule::parse_xyz("1\nO\nO 0 0 0\n", 0, 3).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = ferric_integrals::operator::Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let cfg = UhfConfig {
            energy_conv: 1e-9,
            density_conv: 1e-8,
            mom_after_iter: 5,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
        assert!(res.converged, "O atom UHF did not converge");
        // ⟨S²⟩ for a clean triplet = S(S+1) = 1·2 = 2.0 (allow mild contamination).
        let s2 = expectation_s_squared(
            &res.mos_alpha,
            res.mos_beta.as_ref().unwrap(),
            &oneelectron::overlap(&prep),
            5,
            3,
        );
        assert!((s2 - 2.0).abs() < 0.05, "O atom ⟨S²⟩ = {} (expected ≈2.0)", s2);
    }

    #[test]
    fn uhf_ecp_xe_matches_rhf() {
        // Xe atom, def2-SVP (ECP-valence basis: 28-core def2 ECP, nelec 54→26).
        // Closed-shell singlet, so UHF must collapse to the RHF solution. This
        // is the regression for the open-shell ECP gap: driver::prepare used to
        // build UHF/ROHF hcore WITHOUT the V_ECP projector, so this comparison
        // was off by the full ECP energy (~hundreds of Ha). For all-electron
        // bases the ECP fold-in is a byte-identical no-op, so every existing
        // all-electron UHF/ROHF result is unchanged.
        let mut mol = Molecule::parse_xyz("1\nXe\nXe 0 0 0\n", 0, 1).unwrap();
        let bs = basis::bundled("def2-svp").unwrap();
        mol.apply_ecp(&bs);
        assert_eq!(mol.nelec(), 26, "def2 Xe ECP should remove 28 core electrons");
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = ferric_integrals::operator::Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let cfg = UhfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-9,
            max_iter: 200,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let uhf = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
        assert!(uhf.converged, "Xe/ECP UHF did not converge");
        let rhf = crate::rhf::solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        assert!(rhf.converged, "Xe/ECP RHF did not converge");
        assert!(
            (uhf.energy - rhf.energy).abs() < 1e-8,
            "closed-shell UHF ({}) != RHF ({}) on ECP basis",
            uhf.energy,
            rhf.energy
        );
    }

    #[test]
    fn test_uks_pbe_bromine_atom_converges() {
        // Free Br atom ground state is ²P (doublet): 4s²4p⁵, one hole in the p shell.
        // nα=18, nβ=17. Without fractional occupation the degenerate 4p shell
        // makes the GGA XC potential orientation-dependent and the UKS-PBE SCF
        // oscillates forever (same failure as O/S/Si ³P). Fractional/ensemble
        // occupation spreads the hole equally over the three 4p orbitals, restoring
        // spherical symmetry and converging the SCF.
        // Regression for the HBr TS/MBD aug-cc-pVTZ hang in the free-atom proatom
        // solve (proatom closure in ferric-cli/src/main.rs).
        let mol = Molecule::parse_xyz("1\nBr\nBr 0 0 0\n", 0, 2).unwrap();
        let bs = basis::bundled("aug-cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = ferric_integrals::operator::Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let cfg = UhfConfig {
            xc: Some("PBE".to_string()),
            fractional_occ: true,
            mom_after_iter: 5,
            max_iter: 200,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg)
            .expect("UKS-PBE Br atom with fractional occ should converge");
        assert!(res.converged, "UKS-PBE Br atom did not converge with fractional occ");
        // Sanity: ⟨S²⟩ for a clean doublet (S=1/2) = 0.75; allow mild contamination.
        let s2 = expectation_s_squared(
            &res.mos_alpha,
            res.mos_beta.as_ref().unwrap(),
            &oneelectron::overlap(&prep),
            18,
            17,
        );
        assert!((s2 - 0.75).abs() < 0.1, "Br atom ⟨S²⟩ = {} (expected ≈0.75)", s2);
    }

    #[test]
    fn external_point_charge_changes_uks_pbe_energy() {
        // Reuses the test_uks_pbe_bromine_atom_converges fixture verbatim (free
        // Br atom, doublet, aug-cc-pvdz, fractional_occ, mom_after_iter: 5),
        // adding only the external point charge. Proves the external_potential
        // wiring from Task 5 composes with UKS (xc.is_some() + open-shell).
        let mol = Molecule::parse_xyz("1\nBr\nBr 0 0 0\n", 0, 2).unwrap();
        let bs = basis::bundled("aug-cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = ferric_integrals::operator::Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let base_cfg = UhfConfig {
            xc: Some("PBE".to_string()),
            fractional_occ: true,
            mom_after_iter: 5,
            max_iter: 200,
            ..Default::default()
        };
        let base = solve_uhf(&ctx, &mol, &prep, &bounds, &base_cfg)
            .expect("UKS-PBE Br atom baseline should converge");

        let ext = ExternalPotential {
            point_charges: vec![PointCharge { q: 1.0, x: 0.0, y: 0.0, z: 20.0 }],
            field: None,
        };
        let perturbed_cfg = UhfConfig {
            xc: Some("PBE".to_string()),
            fractional_occ: true,
            mom_after_iter: 5,
            max_iter: 200,
            external_potential: Some(ext),
            ..Default::default()
        };
        let perturbed = solve_uhf(&ctx, &mol, &prep, &bounds, &perturbed_cfg)
            .expect("UKS-PBE Br atom + external potential should converge");

        assert!(perturbed.converged);
        assert!((perturbed.energy - base.energy).abs() > 1e-8);
    }

    #[test]
    fn test_uks_pbe_oxygen_atom_fractional_occ_converges() {
        // Free O atom (³P) via UKS-PBE. With INTEGER occupation the degenerate
        // 2p shell makes the GGA potential orientation-dependent and the SCF
        // oscillates forever (no convergence at 200 iters). Fractional/ensemble
        // occupation spreads the open-shell electrons equally over the
        // degenerate p orbitals, restoring spherical symmetry and converging it.
        // Regression for the TS free-atom-volume residual (commit chain on
        // feat/tensor-einsum-framework). sto-3g keeps it fast and still has the
        // 3-fold-degenerate 2p shell that triggers the pathology.
        let mol = Molecule::parse_xyz("1\nO\nO 0 0 0\n", 0, 3).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = ferric_integrals::operator::Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let cfg = UhfConfig {
            xc: Some("PBE".to_string()),
            fractional_occ: true,
            mom_after_iter: 0,
            max_iter: 200,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg)
            .expect("UKS-PBE O atom with fractional occ should solve");
        assert!(res.converged, "UKS-PBE O atom did not converge with fractional occ");
    }

    #[test]
    fn uhf_dfjk_matches_direct_small() {
        // Water cation (H2O+), doublet, def2-svp: standard small open-shell
        // UHF benchmark. Compares direct J/K against DF-JK
        // (def2-universal-jkfit) for plain HF (omega=0).
        let mol = Molecule::parse_xyz(
            "3\nwater cation\nO 0.000000 0.000000 0.117300\nH 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n",
            1,
            2,
        )
        .unwrap();
        let bs = basis::bundled("def2-svp").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = ferric_integrals::operator::Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let cfg_direct = UhfConfig {
            max_iter: 100,
            ..Default::default()
        };
        let r_direct = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg_direct).unwrap();

        let cfg_df = UhfConfig {
            max_iter: 100,
            df_j_aux: Some("def2-universal-jkfit".to_string()),
            df_k_aux: Some("def2-universal-jkfit".to_string()),
            ..Default::default()
        };
        let r_df = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg_df).unwrap();

        assert!(
            (r_direct.energy - r_df.energy).abs() < 2e-4,
            "UHF DF-JK {} vs direct {} differ by {:.2e}",
            r_df.energy,
            r_direct.energy,
            (r_df.energy - r_direct.energy).abs()
        );
        assert!(r_df.converged);
        // DF-JK must actually be used (not silently ignored): it builds J/K from
        // 3-center integrals, not 4-center quartets, so it should report strictly
        // fewer computed direct quartets than the fully-direct path.
        assert!(
            r_df.computed_quartets < r_direct.computed_quartets,
            "DF-JK computed_quartets={} should be less than direct's {} \
             (df_j_aux/df_k_aux appear to be ignored)",
            r_df.computed_quartets,
            r_direct.computed_quartets
        );
    }
}
