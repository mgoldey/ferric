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
use crate::rhf::{validate_injected, InjectedConfigError, PeriodicInjection, RhfConfig};
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
    err_if_unconverged(solve_uhf_best_effort(
        ctx,
        mol,
        prep,
        bounds,
        config,
        initial_mos,
    )?)
}

/// Reject every `UhfConfig` feature [`solve_uhf_injected`] cannot honour,
/// naming the first offending field: everything [`validate_injected`] refuses
/// for RHF (DF/pluggable K, XC, Newton/TRAH/AURORA, the MINAO guess without an
/// explicit density, external potential, COSMO/PCM/polarizable,
/// `check_stability`), plus the UHF-only `scf_stability_descent` (the descent
/// re-runs the MOLECULAR solver from rotated MOs and needs the molecular
/// stability Hessian). Checked after the RHF list, so a config that sets both
/// reports the RHF field.
pub fn validate_injected_uhf(config: &UhfConfig) -> Result<(), InjectedConfigError> {
    validate_injected(config)?;
    if config.scf_stability_descent {
        return Err(InjectedConfigError {
            field: "scf_stability_descent",
            reason: "the descent re-solves with the molecular J/K and needs the molecular \
                     stability Hessian (prep/bounds)",
        });
    }
    Ok(())
}

/// Open-shell UHF on caller-supplied `(S, h, V_nn)` and J/K builders — the
/// Gamma-point periodic entry, Stage 4 (`reference/pbc/FINDINGS.md`
/// "Iteration 6"). Mirrors [`crate::rhf::solve_rhf_injected`]:
///
/// * `mol` supplies only the electron count, charge and multiplicity; `prep`
///   only `nbasis`; `bounds` is never read for integrals on this path (it is
///   required by the shared body's signature, like `solve_rhf_injected`'s).
/// * J = `inj.j` on `D_α + D_β`; K_σ = the SAME `inj.k`, called per spin as
///   `update_density(D_σ)` then `build(D_σ)` (the pluggable-K sequence of
///   the molecular path). F_σ = h + J − K_σ. Any Madelung term lives in the
///   K builder and is linear in D, so NO per-spin factor is applied here.
///   `build_from_occ` is never called on this path.
/// * Guess: `initial_mos` (per-spin MOs, as [`solve_uhf_with_guess`]) >
///   `config.init_guess_density` (spin-summed; split `D/2` per spin and the
///   guess Fock `h + J[D] − K[D/2]` is built with the INJECTED builders) >
///   the core guess (diagonalize the injected `h`, exactly the molecular
///   `use_sad_guess = false` path). The molecular `rhf::build_jk` / MINAO
///   guess is never called. The closed-shell β HOMO/LUMO mixing (0.1 rad when
///   `nocc_α == nocc_β`) is applied as on the molecular path.
/// * Kohn-Sham (Stage 5, FINDINGS "Iteration 10"): with `inj.xc = Some(x)`
///   (which must report [`crate::rhf::XcBuilder::supports_polarized`]) and
///   `a = x.exact_exchange_fraction()` ∈ [0, 1],
///
///   ```text
///   F_σ = h + J[D_α + D_β] − a·K_inj(D_σ) + V_σ,   (E_xc, V_α, V_β) = x.build_polarized(D_α, D_β)
///   E   = ½ Σ_σ Tr[D_σ (h + F_σ − V_σ)] + E_xc + V_nn
///   ```
///
///   — the molecular `UksXcContribution::add_xc_uks` convention. The injected
///   K carries any Madelung term and is linear in D_σ, so it rides on the
///   exact-exchange part only, with NO per-spin ½. For `a = 0` the injected K
///   is never built. The injected guess Fock has no XC (it is a guess).
///
/// Errors (typed, by name) on any config field [`validate_injected_uhf`]
/// refuses, on an `xc` builder without spin-polarized support or with `a`
/// outside [0, 1], on wrong-shape `s`/`h` or non-finite `vnn`, and — like
/// [`solve_uhf`] — when the SCF does not converge.
///
/// Exchange-divergence caveat (Ewald trap, FINDINGS Iteration 6): at Gamma
/// `v_M S D_σ S` lowers every occupied level by `v_M`, so a state that is
/// non-aufbau without the Madelung term can be self-consistent with it. This
/// function does not know `v_M`; `ferric_pbc::uhf::gamma_uhf` reports the
/// per-spin gap against it and offers the staged (none → ewald) start. With a
/// hybrid the shift (and the trap window) is `a·v_M`; `ferric_pbc::dft::gamma_uks`
/// checks the occupation-aware per-spin gap against it.
pub fn solve_uhf_injected<'a>(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &UhfConfig,
    inj: PeriodicInjection<'a>,
    initial_mos: Option<(&Array2<f64>, &Array2<f64>)>,
) -> Result<ScfResult, FerricError> {
    validate_injected_uhf(config).map_err(|e| {
        FerricError::General(format!(
            "solve_uhf_injected: RhfConfig.{} is not supported with injected S/h/J/K: {}",
            e.field, e.reason
        ))
    })?;
    crate::rhf::check_injection_shapes("solve_uhf_injected", &inj, prep.nbasis())?;
    if let Some(x) = inj.xc.as_ref() {
        if !x.supports_polarized() {
            return Err(FerricError::General(
                "solve_uhf_injected: PeriodicInjection.xc does not support spin-polarized \
                 evaluation (XcBuilder::supports_polarized() is false): the injected UKS path \
                 needs XcBuilder::build_polarized (e.g. ferric_pbc::dft::PeriodicXc)"
                    .into(),
            ));
        }
    }
    err_if_unconverged(solve_uhf_impl(
        ctx,
        mol,
        prep,
        bounds,
        config,
        initial_mos,
        None,
        Some(inj),
        None,
    )?)
}

/// The error variant carries no text: say why a run whose density settled
/// still failed (the orbital-gradient veto refused it), when it did.
fn note_vetoed_nonconvergence(r: &ScfResult) {
    let Some(v) = r.convergence.and_then(|c| c.gradient_vetoes) else {
        return;
    };
    eprintln!(
        "UHF: not converged after {} iterations; the density settled but the \
         orbital-gradient veto refused it {} time(s) (last at iteration {}, \
         gradient {:.3e} > {:.1e})",
        r.iterations, v.count, v.last_iteration, v.last_orbital_gradient, v.bound
    );
}

/// Convert a non-converged best-effort result into the historical hard error.
fn err_if_unconverged(r: ScfResult) -> Result<ScfResult, FerricError> {
    if r.converged {
        Ok(r)
    } else {
        note_vetoed_nonconvergence(&r);
        Err(FerricError::ScfConvergence {
            iterations: r.iterations,
            last_energy: r.energy,
        })
    }
}

/// UHF that returns its best-effort state instead of erroring when it fails.
///
/// Returns `Ok` with `converged: false` (and `exit` describing why) when the SCF does
/// not converge, keeping the final density and MOs. This is what an open-shell
/// convergence ladder needs: a failed rung's density becomes the next rung's guess.
///
/// **Callers must check `converged`.** Prefer [`solve_uhf`] unless you are
/// specifically implementing restart/escalation logic.
pub fn solve_uhf_best_effort(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &UhfConfig,
    initial_mos: Option<(&Array2<f64>, &Array2<f64>)>,
) -> Result<ScfResult, FerricError> {
    let first = solve_uhf_fockmod(ctx, mol, prep, bounds, config, initial_mos, None)?;
    if !config.scf_stability_descent {
        return Ok(first);
    }
    Ok(stability_descent(ctx, mol, prep, bounds, config, first))
}

/// Step sizes tried when following a downhill eigenvector, in radians.
///
/// NOT a guess. Measured in the sibling cDFT lane on
/// `test/cdft-constrained-stability`: steps ε ≤ 0.5 rad fall straight back into
/// the saddle's own DIIS basin even though those rotations demonstrably LOWER
/// the energy, and only ~1 rad escapes. A ladder probing only small steps would
/// wrongly conclude the eigenvector is useless. The smaller entries are kept so
/// a system where a gentler step suffices is not over-rotated past its minimum,
/// and the sweep takes the LOWEST result rather than the first success.
pub(crate) const DESCENT_STEPS: [f64; 3] = [0.4, 0.8, 1.2];

/// Maximum descent rounds. Each round is one Davidson eigensolve plus up to
/// `DESCENT_STEPS.len()` full SCF re-converges, bounding the worst case at a
/// small multiple of the undescended solve.
pub(crate) const MAX_DESCENT_ROUNDS: usize = 3;

/// **Unconstrained UHF state selection.** Given a converged UHF solution, check
/// whether it is a SADDLE of the orbital Hessian and, if so, follow the
/// downhill eigenvector and re-converge from there, keeping the lowest result.
///
/// Gated by [`crate::rhf::RhfConfig::scf_stability_descent`], which defaults
/// **off** — see that field's doc for why this default differs from
/// `cdft_stability_descent`'s.
///
/// # Failure policy
///
/// Every failure mode returns the INPUT solution unchanged, after printing why.
/// A descent that cannot be computed, does not re-converge, or lands HIGHER
/// must never make the answer worse than not having tried, so this function can
/// only ever improve on `first` or leave it exactly alone.
fn stability_descent(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &UhfConfig,
    first: ScfResult,
) -> ScfResult {
    if !first.converged {
        eprintln!(
            "SCF stability descent: SKIPPED — the SCF did not converge, so the orbital \
             Hessian would be evaluated at a non-stationary point and its eigenvalues \
             would not be a stability verdict at all."
        );
        return first;
    }
    // The descent needs a stability verdict, and the verdict is only computed
    // when `check_stability` is set. Rather than silently computing one behind
    // the user's back (which would make the cost of this knob invisible), say
    // so: the two knobs are meant to be set together.
    let mut best = first;
    for round in 1..=MAX_DESCENT_ROUNDS {
        let Some(st) = best.stability.as_ref() else {
            eprintln!(
                "SCF stability descent: SKIPPED — no stability verdict is available \
                 (RhfConfig::check_stability is off, or the analysis was skipped for a \
                 stated reason). Set check_stability = true alongside \
                 scf_stability_descent. The solution is returned as converged, which does \
                 NOT mean it is the lowest state."
            );
            return best;
        };
        let verdict = st.verdict();
        let lmin = st.lowest_eigenvalue;
        if verdict != crate::stability::StabilityVerdict::Unstable {
            if round == 1 && config.verbose {
                eprintln!(
                    "SCF stability descent: the converged solution is {} (lambda_min = \
                     {lmin:+.4e}); no descent taken.",
                    verdict.label()
                );
            }
            return best;
        }
        let va = st.eigenvector_alpha.clone();
        let Some(vb) = st.eigenvector_beta.clone() else {
            eprintln!(
                "SCF stability descent: the verdict is UNSTABLE (lambda_min = {lmin:+.4e}) \
                 but carries no beta eigenvector block, so the UHF rotation cannot be \
                 built. Returning the saddle."
            );
            return best;
        };
        eprintln!(
            "SCF stability descent (round {round}): the converged solution at E = {:.8} is \
             a SADDLE (lambda_min = {lmin:+.4e}); following the downhill eigenvector.",
            best.energy
        );

        let (nocc_a, nocc_b) = match nocc_ab(mol) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("SCF stability descent: cannot resolve occupations ({e:?})");
                return best;
            }
        };
        let Some(cb0) = best.mos_beta.clone() else {
            eprintln!("SCF stability descent: the solution carries no beta MOs; returning it.");
            return best;
        };

        let mut improved: Option<ScfResult> = None;
        for &step in &DESCENT_STEPS {
            let g_a = rotate_mos(&best.mos_alpha, &va, nocc_a, step);
            let g_b = rotate_mos(&cb0, &vb, nocc_b, step);
            let cand =
                match solve_uhf_fockmod(ctx, mol, prep, bounds, config, Some((&g_a, &g_b)), None) {
                    Ok(c) if c.converged => c,
                    Ok(c) => {
                        eprintln!(
                            "SCF stability descent: step {step} did not converge (E = {:.8}); \
                         discarded.",
                            c.energy
                        );
                        continue;
                    }
                    Err(e) => {
                        eprintln!("SCF stability descent: step {step} failed ({e:?}); discarded.");
                        continue;
                    }
                };
            if accepts_candidate(
                cand.energy,
                best.energy,
                improved.as_ref().map(|b| b.energy),
            ) {
                improved = Some(cand);
            }
        }

        match improved {
            Some(c) => {
                eprintln!(
                    "SCF stability descent (round {round}): reached a LOWER state, \
                     E = {:.8} (was {:.8}, dE = {:.8} Ha = {:.4} eV)",
                    c.energy,
                    best.energy,
                    best.energy - c.energy,
                    (best.energy - c.energy) * 27.211_386_245_988
                );
                best = c;
            }
            None => {
                eprintln!(
                    "SCF stability descent (round {round}): the solution is a saddle \
                     (lambda_min = {lmin:+.4e}) but NO step reached a lower state. Returning \
                     the saddle, which is therefore NOT established as the lowest state."
                );
                return best;
            }
        }
    }
    eprintln!(
        "SCF stability descent: still descending after {MAX_DESCENT_ROUNDS} rounds; \
         returning the lowest found (E = {:.8}). It is NOT established as the bottom.",
        best.energy
    );
    best
}

/// Should a descended candidate replace the best-so-far?
///
/// Only if it is strictly BELOW both the incumbent and any better candidate
/// already found this round. Strict `<` throughout, so an exactly equal energy
/// does not churn the answer.
///
/// Extracted rather than inlined for the reason the sibling cDFT lane
/// documented: inline, this guard is UNREACHABLE by the suite, because on the
/// systems tested every descended candidate happens to be lower — so deleting
/// it leaves everything GREEN. It is the entire reason the descent cannot make
/// an answer worse than not having tried, so it is tested directly. See
/// `descent_never_accepts_a_higher_state`.
pub(crate) fn accepts_candidate(cand_e: f64, best_e: f64, improved_e: Option<f64>) -> bool {
    cand_e < best_e && improved_e.is_none_or(|b| cand_e < b)
}

/// Test-visible alias for [`accepts_candidate`], so the guard can be exercised
/// from an integration test rather than only through a path that never reaches
/// its `false` branch. Behaviour is the function itself, not a copy of it.
#[doc(hidden)]
pub fn accepts_candidate_for_test(cand_e: f64, best_e: f64, improved_e: Option<f64>) -> bool {
    accepts_candidate(cand_e, best_e, improved_e)
}

/// (nocc_α, nocc_β) from the molecule's charge and multiplicity.
fn nocc_ab(mol: &Molecule) -> Result<(usize, usize), FerricError> {
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    if two_s < 0 || nelec < two_s || (nelec - two_s) % 2 != 0 {
        return Err(FerricError::General(format!(
            "UHF descent: incompatible nelec={nelec} and multiplicity={}",
            mol.multiplicity
        )));
    }
    Ok((
        ((nelec + two_s) / 2) as usize,
        ((nelec - two_s) / 2) as usize,
    ))
}

/// Rotate MOs by `exp(κ)` for the antisymmetric κ built from the occ→virt
/// block `k_ov` scaled by `eps`, via the Cayley transform
/// `(I − κ/2)⁻¹ (I + κ/2)` — orthogonality-preserving to machine precision.
pub(crate) fn rotate_mos(
    c: &Array2<f64>,
    k_ov: &Array2<f64>,
    nocc: usize,
    eps: f64,
) -> Array2<f64> {
    use ndarray_linalg::Solve;
    let n = c.nrows();
    let mut kappa = Array2::<f64>::zeros((n, n));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            if ir >= k_ov.nrows() || i >= k_ov.ncols() {
                continue;
            }
            let v = eps * k_ov[(ir, i)];
            kappa[(a, i)] = v;
            kappa[(i, a)] = -v;
        }
    }
    let half = 0.5 * &kappa;
    let eye = Array2::<f64>::eye(n);
    let am = &eye - &half;
    let bm = &eye + &half;
    let mut u = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        match am.solve(&bm.column(col).to_owned()) {
            Ok(sol) => {
                for row in 0..n {
                    u[(row, col)] = sol[row];
                }
            }
            // A singular (I − κ/2) cannot happen for antisymmetric κ (its
            // eigenvalues are 1 ± i·imag), but the solve is fallible, so fall
            // back to the identity column rather than panicking inside a
            // best-effort descent.
            Err(_) => u[(col, col)] = 1.0,
        }
    }
    c.dot(&u)
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
    solve_uhf_molecular(ctx, mol, prep, bounds, config, initial_mos, fock_mod)
}

/// The molecular (non-injected, unblocked) solver: refuses the symmetry fields,
/// which only [`solve_uhf_blocked`] honours.
fn solve_uhf_molecular(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &UhfConfig,
    initial_mos: Option<(&Array2<f64>, &Array2<f64>)>,
    fock_mod: Option<UhfFockMod>,
) -> Result<ScfResult, FerricError> {
    crate::symmetric::require_symmetry_off(config, "solve_uhf")?;
    solve_uhf_impl(
        ctx,
        mol,
        prep,
        bounds,
        config,
        initial_mos,
        fock_mod,
        None,
        None,
    )
}

/// The blocked solve of [`crate::symmetric_open::solve_uhf_symmetric`]: `mol`,
/// `prep` and `bounds` are already in the symmetry frame, one spec per spin.
pub(crate) fn solve_uhf_blocked(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &UhfConfig,
    specs: (
        &crate::symmetric::SymmetrySpec,
        &crate::symmetric::SymmetrySpec,
    ),
) -> Result<ScfResult, FerricError> {
    err_if_unconverged(solve_uhf_impl(
        ctx,
        mol,
        prep,
        bounds,
        config,
        None,
        None,
        None,
        Some(specs),
    )?)
}

/// Shared body of [`solve_uhf_fockmod`] (`inj = None`: the molecular solver,
/// byte-identical to the pre-injection code — every injected-path branch is
/// gated on `inj_jk.is_some()`) and [`solve_uhf_injected`] (`inj = Some`,
/// config already validated).
#[allow(clippy::too_many_arguments)]
fn solve_uhf_impl(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &UhfConfig,
    initial_mos: Option<(&Array2<f64>, &Array2<f64>)>,
    fock_mod: Option<UhfFockMod>,
    inj: Option<PeriodicInjection<'_>>,
    sym: Option<(
        &crate::symmetric::SymmetrySpec,
        &crate::symmetric::SymmetrySpec,
    )>,
) -> Result<ScfResult, FerricError> {
    use ferric_dft::ks::KsXcUks;
    use ferric_dft::xc_trait::{KMix, UksXcContribution};

    // Split the injection: (S, h, V_nn) go to the shared env builder, the J/K
    // builders are driven in the iteration loop (as in rhf::solve_rhf_impl).
    // Both `None` on the molecular path.
    // `inj_xc` is the injected (periodic) spin-polarized XC; its
    // `supports_polarized()` was checked by solve_uhf_injected.
    let (pre_env, mut inj_jk, mut inj_xc) = match inj {
        Some(PeriodicInjection {
            s,
            h,
            vnn,
            j,
            k,
            xc,
        }) => (Some((s, h, vnn)), Some((j, k)), xc),
        None => (None, None, None),
    };
    let injected = inj_jk.is_some();

    // Build UKS XC contribution once. None for pure UHF.
    let xc_contrib: Option<Box<dyn UksXcContribution>> = if let Some(name) = config.xc.as_deref() {
        let main = config.dft_grid.clone().unwrap_or_default();
        let nlc = config
            .nlc_grid
            .clone()
            .unwrap_or_else(ferric_dft::grid::AtomicGridConfig::default_nlc);
        // Thread the caller's `[memory] budget_gb` into the grid AO cache --
        // the largest single allocation in a DFT job. This used to call the
        // UNbudgeted `new_with_omega`, which resolves from env/auto-detect
        // and so silently DISCARDED `config.three_index_budget_bytes`. The
        // budgeted constructor existed for exactly this and had ZERO
        // production callers; its own doc records the symptom ("Setting
        // `budget_gb = 4` on a 64 GB box still sized the grid cache against
        // ~51 GB"), i.e. the documented primary knob did nothing while
        // FERRIC_MEM_BUDGET_GB worked. 0 means unset, matching
        // `rhf::resolve_three_index_budget`.
        let ks = KsXcUks::new_with_omega_budgeted(
            mol,
            prep.basis_set(),
            name,
            &main,
            &nlc,
            config.xc_omega,
            (config.three_index_budget_bytes != 0).then_some(config.three_index_budget_bytes),
        )
        .map_err(|e| FerricError::General(format!("KsXcUks init for {name}: {e:?}")))?;
        Some(Box::new(ks) as Box<dyn UksXcContribution>)
    } else {
        None
    };

    // Injected (periodic) UKS: a global hybrid scales the injected K by its
    // exact-exchange fraction (validate_injected_uhf guarantees `xc_contrib`
    // is None there, so the two sources never coexist). Molecular path:
    // `inj_xc` is None and this is the historical expression.
    let k_mix: KMix = match inj_xc.as_ref() {
        Some(x) => {
            let a = x.exact_exchange_fraction();
            if !(a.is_finite() && (0.0..=1.0).contains(&a)) {
                return Err(FerricError::General(format!(
                    "solve_uhf_injected: XcBuilder exact-exchange fraction must be in [0, 1], \
                     got {a}"
                )));
            }
            KMix {
                sr: a,
                lr: a,
                omega: 0.0,
            }
        }
        None => xc_contrib.as_ref().map(|x| x.k_mix()).unwrap_or_default(),
    };

    // Meta-GGA (SCAN / r2SCAN) UKS is stiffer than LDA/GGA and limit-cycles under
    // plain DIIS; apply the same modest default virtual-block level shift the
    // closed-shell RKS path uses (see solve_rhf) when the user hasn't set one.
    // Ramped to zero as the gradient converges, so the final energy is
    // unperturbed. LDA/GGA/hybrid keep the user's shift (default 0).
    let effective_level_shift = crate::driver::effective_level_shift(config);

    // K coefficient on the full Coulomb K, per-spin. For pure HF (no xc),
    // KMix::default() gives sr = 1.0, lr = 1.0 → c_k = 1.0 (original UHF behavior).
    // Injected UKS: c_k = a (need_k below is then false for a = 0, so the
    // injected K is never built for a pure functional).
    let c_k: f64 = if xc_contrib.is_some() || inj_xc.is_some() {
        k_mix.sr
    } else {
        1.0
    };

    // RSH path (ω > 0): build per-spin K from c_SR · K[erfc(ω)] + c_LR · K[erf(ω)]
    // via two DfK fitters. Each fitter's geometry-only B[P,μ,ν] tensor is built
    // once; only the D-dependent contraction runs per iteration per spin.
    // Shared geometry-only environment: S, hcore(+external+ECP — V_ECP folds
    // into hcore once, byte-identical to plain hcore for all-electron bases),
    // V_nn(+external), COSMO/PCM contexts, resolved memory budget, RSH
    // fitters. One construction serving all six SCF variants (crate::driver).
    let crate::driver::ScfEnv {
        s,
        h,
        vnn,
        ooc_budget,
        cosmo_cavity,
        pcm_ctx,
        polarizable_site_basis,
        dfk_sr,
        dfk_lr,
    } = crate::driver::prepare(ctx, mol, prep, config, &k_mix, pre_env)?;

    // The ONE SR/LR exchange response every Hessian matvec in this solve uses —
    // Newton steps, TRAH steps and the post-SCF stability eigensolve alike. It
    // borrows the SAME two fitters `subtract_rsh_exchange` assembles the
    // converged Fock from and reads the SAME `k_mix`, so the Fock and the
    // Hessian cannot disagree about the kernel (the #314 defect). `None` at
    // omega == 0, where the matvecs keep their untouched plain-Coulomb path.
    let rsh_response = match (dfk_sr.as_ref(), dfk_lr.as_ref()) {
        (Some(sr), Some(lr)) => Some(crate::rsh_response::RshResponse::new(
            sr,
            lr,
            k_mix.sr,
            k_mix.lr,
            k_mix.omega,
        )),
        _ => None,
    };
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
        return Err(FerricError::General("UHF: nocc_a + nocc_b != nelec".into()));
    }
    if nocc_b > nocc_a {
        return Err(FerricError::General("UHF: nocc_b > nocc_a".into()));
    }
    // Canonical orthogonalizer X (n × m): drops eigenvectors of S below
    // LINDEP_THRESH to handle near-linear-dependent basis sets (Na clusters in
    // aug-cc-pVDZ). m == n for well-conditioned S. See crate::rhf for details.
    let x = crate::rhf::canonical_orthogonalizer(&s)?;
    // Per-spin Fock diagonalizers: global by default, per-irrep under
    // point-group blocking (each spin has its own occupation request).
    let plain = crate::symmetric::PlainDiag::Rect(&x);
    let diag_a = crate::symmetric::FockDiag::new(sym.map(|p| p.0), &s, plain)?;
    let diag_b = crate::symmetric::FockDiag::new(sym.map(|p| p.1), &s, plain)?;

    // Initial guess: caller-supplied MOs if provided, else the configured
    // density guess (MINAO/SAD by default), else hcore.
    //
    // # Why this is not just hcore any more
    //
    // Until 2026-09-16 this branch ALWAYS used hcore: it called `hcore_guess`
    // purely as a "sanity check it succeeds" and threw the density away, then
    // diagonalized bare `h`. `RhfConfig::init_guess_density` and
    // `use_sad_guess` — which `rhf.rs` honours, and which default to the MINAO
    // projection — were referenced NOWHERE in `uhf.rs` or `rohf.rs`, so a
    // caller who explicitly asked for a better open-shell guess silently got
    // hcore.
    //
    // That is not cosmetic. MEASURED on the systems in
    // `tests/scf_state_selection.rs`, against PySCF 2.13.0 / ORCA 6.1.1 /
    // NWChem 7.2.2 at the same geometry and basis: from hcore, UHF converged to
    // a state ABOVE the reference on 4 of 7 open-shell diatomics, every one of
    // them flagged UNSTABLE by ferric's own stability check —
    //
    //   HeNe⁺/def2-SVP  0.136 eV     HeNe⁺/6-31G  0.126 eV
    //   OH/6-31G        4.222 eV     N₂⁺/6-31G   10.373 eV
    //
    // From the MINAO density both HeNe⁺ rows and OH reach the external
    // reference (~1e-10 Ha for HeNe⁺, 3.5e-9 Ha for OH) and HeNe⁺ reports
    // STABLE; N₂⁺ improves by 9.58 eV. OH was not in the original sweep — it
    // surfaced because `ferric-dft`'s own f_xc finite-difference test builds
    // its reference density with `solve_uhf` on OH/6-31G and began failing, so
    // the sweep in `scf_state_selection.rs` was an UNDER-count.
    //
    // The guess is NOT sufficient on its own: N₂⁺/6-31G from MINAO still lands
    // 0.79 eV high and UNSTABLE — on exactly the state PySCF's OWN default
    // guess finds before it follows its own instability. That residue is what
    // `config.scf_stability_descent` exists for; see `stability_descent` below.
    let (mut c_a, mut c_b) = if let Some((ca0, cb0)) = initial_mos {
        if ca0.dim() != (n, n) || cb0.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "solve_uhf_with_guess: initial MO shape mismatch (got {:?}/{:?}, want ({n},{n}))",
                ca0.dim(),
                cb0.dim()
            )));
        }
        (ca0.clone(), cb0.clone())
    } else if let Some((ij, ik)) = inj_jk.as_mut() {
        // Injected (periodic) path: never the molecular build_jk / MINAO.
        // An explicit density builds the guess Fock from the INJECTED J/K;
        // otherwise the core guess, identical to the molecular hcore branch.
        match config.init_guess_density.as_ref() {
            Some(d0) => injected_guess_mos(ij.as_mut(), ik.as_mut(), d0, &h, &x)?,
            None => {
                let _ = hcore_guess(&s, &h, nocc_a.max(1))?;
                let (_, c) = diagonalize(&h, &x)?;
                (c.clone(), c)
            }
        }
    } else if let Some((ga, gb)) =
        uhf_guess_mos(ctx, mol, prep, bounds, config, &h, (&diag_a, &diag_b))?
    {
        (ga, gb)
    } else {
        // hcore guess: get MO coefficients from H' = Xᵀ H X (canonical-orthog).
        // diagonalize() pads to (n × n), keeping the guess shape historical.
        let _ = hcore_guess(&s, &h, nocc_a.max(1))?; // sanity check it succeeds
        (diag_a.run(&h)?.1, diag_b.run(&h)?.1)
    };

    // For genuine open shell (nocc_a > nocc_b), occupying the lowest nocc_σ
    // orbitals per spin is already symmetry-broken. For "forced" UHF on a
    // closed-shell, mix HOMO/LUMO in β with a small angle to break symmetry.
    mix_beta_homo_lumo(&mut c_b, nocc_a, nocc_b, sym.is_some());

    let mut d_a = density(&c_a, nocc_a);
    let mut d_b = density(&c_b, nocc_b);

    // ── Trust-region augmented-Hessian state (opt-in; None = TRAH disabled) ──
    // See the RHF counterpart in `rhf.rs` and `crate::trah::TrahState` for why
    // the ρ assessment spans two SCF iterations. `trah_undo` holds the α/β MOs
    // the pending step departed from, so a rejected step is undone exactly.
    // Both stay `None` when `config.trah_trigger` is `None`, which is what
    // makes the disabled path bit-identical.
    let mut trah_state: Option<crate::trah::TrahState> = config
        .trah_trigger
        .map(|_| crate::trah::TrahState::new(config.trah));
    let mut trah_undo: Option<(Array2<f64>, Array2<f64>)> = None;
    // Set on the iteration a TRAH step is rejected; consumed at the next
    // iteration's arming gate (`crate::trah::settle_engaged_latch`).
    let mut trah_last_rejected = false;
    // TRAH keeps control from its first step until it declines a step as
    // worthless or its radius collapses -- see the arming gate.
    let mut trah_engaged = false;

    // Occupied MO coefficients driving the O(naux·n²·nocc) DF-K half-transform
    // (KBuilder::build_from_occ) in place of the O(naux·n³) density
    // contraction. Unlike RHF, each UHF spin density is a bare
    // D_σ = C_σ·C_σᵀ (see `density`), so no factor-2 rescale is needed here —
    // dropping that RHF-only factor was what made this path look like a
    // convergence bug and get disabled (see the note in rhf.rs).
    let occ_or_none = |c: &Array2<f64>, nocc: usize| -> Option<Array2<f64>> {
        (nocc > 0).then(|| c.slice(ndarray::s![.., ..nocc]).to_owned())
    };
    let mut d_occ_a: Option<Array2<f64>> = occ_or_none(&c_a, nocc_a);
    let mut d_occ_b: Option<Array2<f64>> = occ_or_none(&c_b, nocc_b);

    let mut j_buf = Array2::<f64>::zeros((n, n));
    let mut k_a_buf = Array2::<f64>::zeros((n, n));
    let mut k_b_buf = Array2::<f64>::zeros((n, n));

    // Coupled α/β DIIS — single subspace, joint error norm. PySCF-style.
    // Independent per-spin DIIS desyncs α and β on cations (e.g. H2O+ took
    // 421 iterations to converge with independent DIIS; coupled converges
    // in ~15-25 cycles).
    let mut diis = Diis::new(config.diis_size);
    // Orbital-gradient exit gate (`UhfExitGate`): veto state + swap history.
    let mut exit_gate = UhfExitGate::default();
    // UHF extrapolates with `step_pair`, which fills the β histories as well —
    // FOUR n×n matrices per subspace entry, twice RHF's. This projection was
    // missing entirely until 2026-08-30: the warning was called from RHF only,
    // so the variant that holds the most DIIS memory reported none of it.
    crate::driver::warn_if_diis_history_large(
        "UHF",
        n,
        config.diis_size,
        crate::diis::DiisHistoryShape::PairSpin,
        false,
        ooc_budget,
    );
    // Convergence bookkeeping (prev energy, ΔP over the TOTAL α+β density,
    // divergence streak, stall history) — shared driver::ScfMonitor; ΔP is
    // INFINITY until the first rebuild (see rhf::scf_converged).
    // Most recent Thole-damped polarizable-embedding dipoles (None when
    // `config.polarizable` is off/empty) — threaded into both ScfResult
    // constructors below (converged early-exit and MaxIter).
    let mut last_induced_dipoles: Option<Array2<f64>> = None;
    let mut mon = crate::driver::ScfMonitor::new();
    let mut last_report: Option<crate::result::ConvergenceReport> = None;
    let mut total_quartets = 0usize;

    // Maximum-Overlap Method (MOM) references: the previously-accepted occupied
    // α/β MO blocks. From iter `mom_after_iter + 1` onward we pick each spin's
    // occupied set by AO-overlap with these (Gilbert-Besley-Gill), instead of
    // pure aufbau. This pins the open-shell occupation through SCF and converges
    // open-shell atoms (e.g. S/O ³P) whose near-degenerate p-shell otherwise
    // makes plain DIIS oscillate forever. Empty open block for UHF (each spin is
    // a pure closed set, see `mom_pin`). Mirrors the rohf.rs MOM wiring.
    let mut mom_ref_a: Option<Array2<f64>> = None;
    let mut mom_ref_b: Option<Array2<f64>> = None;

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
    let j_aux_eff = if k_mix.omega == 0.0 {
        config.df_j_aux.as_deref()
    } else {
        None
    };
    // RIJCOSX: `k_builder = "cosx"` replaces the ω = 0 DF-K (an explicitly
    // named `df_k_aux` alongside it is refused); DF-J is unaffected.
    let cosx_k = crate::fock_assembly::cosx_replaces_df_k(
        config.k_builder.as_deref(),
        config.df_k_aux.as_deref(),
        need_k && k_mix.omega == 0.0,
    )?;
    let k_aux_eff = if need_k && k_mix.omega == 0.0 && !cosx_k {
        config.df_k_aux.as_deref()
    } else {
        None
    };
    let (mut df_j, mut df_k) = crate::fock_assembly::build_df_jk(
        ctx, mol, coulomb_op, prep, j_aux_eff, k_aux_eff, ooc_budget,
    )?;
    // Record the builders that produced the energy (same effective names as
    // `build_df_jk` above and `driver::prepare`'s RSH pair) so the analytic
    // gradient differentiates THIS energy. `None` when nothing is fitted.
    let df_jk_route = crate::result::DfJkRoute::from_scf(
        j_aux_eff,
        k_aux_eff,
        (k_mix.omega > 0.0).then(|| {
            (
                config
                    .df_k_aux
                    .as_deref()
                    .unwrap_or(crate::fock_assembly::DEFAULT_JK_AUX),
                k_mix.omega,
            )
        }),
        coulomb_op,
        ooc_budget,
    );
    // ── Combined open-shell direct J+K (single quartet pass) ────────────────
    // When BOTH J and K come from the direct (non-DF) path with an ordinary
    // ω = 0 kernel, one `DirectJK::build_uhf` pass produces J[D_α+D_β], K[D_α]
    // and K[D_β] together. This replaces three separate quartet traversals
    // (`DirectJ(D_total)` + `DirectK(D_α)` + `DirectK(D_β)`), each of which
    // re-evaluated the same integrals and screened on the loose global max|D|
    // scalar instead of the tight six-pairwise shell table.
    //
    // Gated to exactly the case where all three matrices are direct and share
    // the Coulomb operator: `df_j.is_none() && df_k.is_none() && need_k &&
    // k_mix.omega == 0.0`. Every other combination (DF-J, DF-K, RSH, pure DFT
    // with no K at all) keeps the previous per-matrix builders untouched.
    //
    // Escape hatch (test + debugging, mirroring `FERRIC_SCF_INCREMENTAL`):
    // `FERRIC_SCF_COMBINED_JK=0` (or `off`/`false`) restores the historical
    // three-pass open-shell build, so the combined path can be A/B'd for
    // correctness and timing without a rebuild.
    // ── Pluggable exchange builder ("link" / "cosx") ────────────────────────
    // Same semantics, warnings and hard errors as `solve_rhf` (shared helper —
    // see `fock_assembly::resolve_k_builder`). Until 2026-09-08 `solve_uhf`
    // never read `config.k_builder`, so both values were silently ignored for
    // every open-shell run.
    //
    // ONE builder instance serves BOTH spins. LinK's density-pair list is
    // density-dependent, so `build_open_shell_pluggable_k` calls
    // `update_density(D_σ)` immediately before each `build(D_σ)` — per-spin
    // lists, not one list from D_α + D_β (exactness over a cheap shortcut; see
    // that helper's doc). COSX carries no density state at all.
    //
    // Only consumed on the non-DF, ω = 0, exact-exchange path — exactly where
    // the combined `DirectJK` would otherwise run. J then comes from `DirectJ`
    // (as it does in `solve_rhf`'s pluggable branch), since the combined pass
    // is what supplied J for free.
    let pluggable_k_kind = crate::fock_assembly::resolve_k_builder(
        config.k_builder.as_deref(),
        df_j.is_some(),
        df_k.is_some(),
        df_k.is_some(),
        need_k,
        k_mix.omega,
    )?;
    let pluggable_k_kind =
        crate::fock_assembly::narrow_k_builder_to_supported(pluggable_k_kind, need_k, k_mix.omega);
    // `LinkBound::SchwarzRef` applies whichever refinement the caller's
    // `bounds` carries — CSB (Eq. (8) of Thompson & Ochsenfeld, JCP 147,
    // 144101 (2017)) for a `csb_m` table, the non-rigorous CSAM estimate for a
    // `csam_x` one — i.e. whichever
    // `SchwarzBounds::compute_for_screening(.., kind)` attached. With no table
    // it is byte-identical to passing `bounds` directly, so the default path is
    // unchanged. Without this wrapper `[scf] screening = "csb"`/`"csam"` would
    // reach every direct builder and closed-shell LinK but silently NOT
    // open-shell LinK, since `solve_uhf` has no bound-construction site of its
    // own to consult a config field at.
    let link_bound = crate::screening::LinkBound::SchwarzRef(bounds);
    // Optional COSX grid schedule (default off; see `crate::cosx_schedule`).
    let k_args = crate::cosx_schedule::KBuilderArgs {
        kind: pluggable_k_kind,
        ctx,
        mol,
        prep,
        link_bound: &link_bound,
        op: coulomb_op,
        cosx: &config.cosx,
        integral_thresh: config.integral_thresh,
        ooc_budget,
    };
    let (mut cosx_sched, mut pluggable_k): (_, Option<Box<dyn KBuilder>>) =
        crate::cosx_schedule::start_with_builder(&k_args)?;

    let combined_direct_jk = df_j.is_none()
        && df_k.is_none()
        && need_k
        && k_mix.omega == 0.0
        && pluggable_k.is_none()
        && !injected
        && crate::direct_jk::combined_open_shell_jk_enabled();
    let mut direct_jk: Option<crate::direct_jk::DirectJK> = if combined_direct_jk {
        Some(crate::direct_jk::DirectJK::new(
            ctx,
            prep,
            bounds,
            config.integral_thresh,
            ooc_budget,
        ))
    } else {
        None
    };
    // No molecular direct builders on the injected path (J/K come from
    // `inj_jk`); `!injected` is `true` on the molecular path, so the gates are
    // unchanged there.
    let mut direct_j: Option<DirectJ> = if df_j.is_none() && !combined_direct_jk && !injected {
        Some(DirectJ::new(
            ctx,
            prep,
            bounds,
            config.integral_thresh,
            ooc_budget,
        ))
    } else {
        None
    };
    let mut direct_k: Option<DirectK> = if need_k
        && k_mix.omega == 0.0
        && df_k.is_none()
        && !combined_direct_jk
        && pluggable_k.is_none()
        && !injected
    {
        Some(DirectK::new(
            ctx,
            prep,
            bounds,
            config.integral_thresh,
            ooc_budget,
        ))
    } else {
        None
    };

    // ── Incremental (differential) Fock build, combined DirectJK path only ──
    // Same scheme (and the same `FERRIC_SCF_INCREMENTAL` kill switch and
    // 8-iteration periodic full rebuild) as `solve_rhf`; see that function for
    // the full rationale. Restricted to the combined direct path for the same
    // reason: the DF/RI paths carry a naux-dependent fitted-Fock noise floor
    // that ΔD accumulation would compound.
    //
    // The open-shell specifics — screening on |ΔD_α| + |ΔD_β| rather than
    // |ΔD_total|, and why — are documented on
    // `DirectJK::build_uhf_incremental`.
    // NOTE: opt-IN (default off), unlike the closed-shell path. See
    // `direct_jk::open_shell_incremental_enabled` for the measurements behind
    // that choice — the scheme is correct but measured ~1.00-1.05x here.
    //
    // INTERACTION WITH A PLUGGABLE K (`k_builder = "link"` / `"cosx"`):
    // structurally impossible, not merely discouraged. `incremental_direct`
    // requires `direct_jk.is_some()`, and `combined_direct_jk` is now gated on
    // `pluggable_k.is_none()` — so when a pluggable builder is active there is
    // no `DirectJK` and the ΔD path is off regardless of
    // `FERRIC_SCF_UHF_INCREMENTAL`. A pluggable K is therefore never handed a
    // ΔD it cannot interpret: LinK would build its density-pair list from a
    // DIFFERENCE density (whose sparsity pattern and magnitudes are unrelated
    // to the density being screened for), and COSX's overlap fit is defined
    // against a true density. The user gets the full-rebuild pluggable path
    // with no error and no silent wrong answer; the env knob keeps applying to
    // the default (non-pluggable) open-shell path exactly as before.
    let incremental_direct =
        direct_jk.is_some() && crate::direct_jk::open_shell_incremental_enabled();
    // The (α, β) densities that produced the CURRENT contents of
    // j_buf/k_a_buf/k_b_buf. `None` until the first full build.
    let mut d_last_fock: Option<(Array2<f64>, Array2<f64>)> = None;
    const INCREMENTAL_FULL_REBUILD_EVERY: usize = 8;
    // Newton and TRAH rebuild an integer density after each step; under
    // fractional occupation they stay off (see second_order_blocked_by_fractional_occ).
    let fractional_blocks_second_order = second_order_blocked_by_fractional_occ(config);
    if let Some(x) = xc_contrib.as_ref() {
        x.start_scf();
    }

    for iter in 1..=config.max_iter {
        ctx.check_interrupted()?;
        // COSX grid schedule: a pending coarse -> production switch rebuilds
        // the builder and restarts DIIS here, before this iteration's K.
        crate::cosx_schedule::before_k_build(
            &mut cosx_sched,
            iter,
            &mut pluggable_k,
            &mut diis,
            &k_args,
        )?;
        let d_total = &d_a + &d_b;

        // On incremental iterations the buffers must KEEP the previous
        // iteration's J/K_σ (the delta is accumulated onto them); every other
        // path starts from zero exactly as before.
        let direct_full_rebuild = incremental_direct
            && (d_last_fock.is_none() || iter % INCREMENTAL_FULL_REBUILD_EVERY == 1);
        let direct_incremental = incremental_direct && !direct_full_rebuild;
        if !direct_incremental {
            j_buf.fill(0.0);
            k_a_buf.fill(0.0);
            k_b_buf.fill(0.0);
        }

        // Combined single-pass J + K_α + K_β when enabled; otherwise J alone
        // here (DF-J or DirectJ) and K below, as before. An injected
        // (periodic) J pre-empts all of them (validate_injected_uhf guarantees
        // no DF/pluggable builder was requested alongside it).
        if let Some((ij, _)) = inj_jk.as_mut() {
            total_quartets += ij.build(&d_total, &mut j_buf)?;
        } else if let Some(djk) = direct_jk.as_mut() {
            if direct_incremental {
                let (da_prev, db_prev) = d_last_fock
                    .as_ref()
                    .expect("d_last_fock set on full rebuild");
                let delta_a = &d_a - da_prev;
                let delta_b = &d_b - db_prev;
                let delta_total = &delta_a + &delta_b;
                total_quartets += djk.build_uhf_incremental(
                    &delta_total,
                    &delta_a,
                    &delta_b,
                    &mut j_buf,
                    &mut k_a_buf,
                    &mut k_b_buf,
                )?;
            } else {
                total_quartets +=
                    djk.build_uhf(&d_total, &d_a, &d_b, &mut j_buf, &mut k_a_buf, &mut k_b_buf)?;
            }
            d_last_fock = Some((d_a.clone(), d_b.clone()));
        } else if let Some(dfj) = df_j.as_mut() {
            // J built from total density (one call). DF-J if configured, else direct.
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
            let mut dfk_sr = dfk_sr
                .as_ref()
                .expect("dfk_sr built when omega>0")
                .borrow_mut();
            let mut dfk_lr = dfk_lr
                .as_ref()
                .expect("dfk_lr built when omega>0")
                .borrow_mut();
            let (dfk_sr, dfk_lr) = (&mut *dfk_sr, &mut *dfk_lr);
            crate::fock_assembly::subtract_rsh_exchange(
                dfk_sr,
                dfk_lr,
                &d_a,
                d_occ_a.as_ref(),
                1.0,
                &mut f_a,
                k_mix.sr,
                k_mix.lr,
                1.0,
            )?;
            crate::fock_assembly::subtract_rsh_exchange(
                dfk_sr,
                dfk_lr,
                &d_b,
                d_occ_b.as_ref(),
                1.0,
                &mut f_b,
                k_mix.sr,
                k_mix.lr,
                1.0,
            )?;
        } else if need_k {
            if let Some((_, ik)) = inj_jk.as_mut() {
                // Injected K per spin: update_density(D_σ) + build(D_σ) from the
                // one builder — the same sequence as the pluggable arm below.
                // The builder's Madelung term (if any) is linear in D_σ, so no
                // per-spin factor here (FINDINGS Iteration 6).
                total_quartets += crate::fock_assembly::build_open_shell_pluggable_k(
                    ik.as_mut(),
                    &d_a,
                    &d_b,
                    &mut k_a_buf,
                    &mut k_b_buf,
                )?;
            } else if direct_jk.is_some() {
                // K_α/K_β already filled by the combined single-pass build above.
            } else if let Some(kb) = pluggable_k.as_mut() {
                // Per-spin `update_density(D_σ)` + `build(D_σ)` from one shared
                // instance (see fock_assembly::build_open_shell_pluggable_k).
                total_quartets += crate::fock_assembly::build_open_shell_pluggable_k(
                    kb.as_mut(),
                    &d_a,
                    &d_b,
                    &mut k_a_buf,
                    &mut k_b_buf,
                )?;
            } else if let Some(dfk) = df_k.as_mut() {
                // C_occ half-transform per spin when available (D_σ = C_occ,σ·C_occ,σᵀ);
                // the fractional-occupation ensemble path falls back to build(D_σ).
                match d_occ_a.as_ref() {
                    Some(c) => dfk.build_from_occ(c, &mut k_a_buf)?,
                    None => dfk.build(&d_a, &mut k_a_buf)?,
                };
                match d_occ_b.as_ref() {
                    Some(c) => dfk.build_from_occ(c, &mut k_b_buf)?,
                    None => dfk.build(&d_b, &mut k_b_buf)?,
                };
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
        let e_elec_no_xc: f64 = 0.5 * ((&(&h + &f_a) * &d_a).sum() + (&(&h + &f_b) * &d_b).sum());
        let e_xc = if let Some(x) = xc_contrib.as_ref() {
            // Occupied-factored density pass per spin (D_σ = C_σ·C_σᵀ); the XC
            // side re-checks each pair and falls back to the dense D_σ, which
            // also covers fractional occupations.
            x.add_xc_uks_occ(
                (&d_a, &d_b),
                (d_occ_a.as_ref(), d_occ_b.as_ref()),
                &mut f_a,
                &mut f_b,
            )
        } else if let Some(ix) = inj_xc.as_mut() {
            // Injected (periodic) UKS: V_σ added after the exchange term,
            // E_xc outside the trace (the add_xc_uks convention above).
            let (e, v_a, v_b) = ix.build_polarized(&d_a, &d_b)?;
            if v_a.dim() != f_a.dim() || v_b.dim() != f_b.dim() {
                return Err(FerricError::General(format!(
                    "solve_uhf_injected: XcBuilder::build_polarized returned V_xc of shapes \
                     {:?}/{:?}, expected {:?}",
                    v_a.dim(),
                    v_b.dim(),
                    f_a.dim()
                )));
            }
            f_a += &v_a;
            f_b += &v_b;
            e
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
        let (e_cosmo, e_pcm, e_pol, iter_induced_dipoles) = crate::driver::solvent_terms(
            mol,
            prep,
            config,
            cosmo_cavity.as_ref(),
            pcm_ctx.as_ref(),
            polarizable_site_basis.as_ref(),
            &d_total,
            &mut [&mut f_a, &mut f_b],
        )?;
        last_induced_dipoles = iter_induced_dipoles;

        let energy = e_elec_no_xc + e_xc + e_cosmo + e_pcm + e_pol + vnn;

        // DIIS errors per spin: F_σ D_σ S − S D_σ F_σ
        let err_a = f_a.dot(&d_a).dot(&s) - s.dot(&d_a).dot(&f_a);
        let err_b = f_b.dot(&d_b).dot(&s) - s.dot(&d_b).dot(&f_b);

        let sig = mon.signals(energy);
        let de = sig.de;
        let err_max_a = err_a.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let err_max_b = err_b.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let err_max = err_max_a.max(err_max_b);
        last_report = Some(crate::result::ConvergenceReport {
            iteration: iter,
            de,
            dp_rms: mon.dp_rms,
            dp_max: mon.dp_max,
            err_max,
            energy_conv: config.energy_conv,
            density_conv: config.density_conv,
            // The open-shell solvers rebuild J/K from the full density every
            // iteration by default (measured: rounding-level dE). The opt-in
            // incremental path (`FERRIC_SCF_UHF_INCREMENTAL`) is not characterised.
            de_noise_floor: (!crate::direct_jk::open_shell_incremental_enabled())
                .then(|| {
                    crate::noise_floor::floor_for_config(config, energy.abs(), prep.nbasis(), false)
                })
                .flatten(),
            // UHF reads no `diis_flavor`: it always runs Pulay.
            diis_fallback: None,
            gradient_vetoes: exit_gate.record(),
        });

        if crate::rhf::scf_trace() {
            eprintln!(
                "UHF iter={iter:4}  E={energy:.12}  dE={de:.3e}  \
                 dp_rms={:.3e}  dp_max={:.3e}  err_max={err_max:.3e}",
                mon.dp_rms, mon.dp_max
            );
        }

        // Machine-readable per-iteration record, streamed and flushed NOW. See
        // the identical block in `solve_rhf` for the full rationale; every
        // value is one the loop already computed. `grad_rms` is `None`: unlike
        // `solve_rhf`, this loop forms only the per-spin commutator MAXIMA
        // (err_max_a/err_max_b), never an RMS — reporting a fabricated 0 would
        // be worse than reporting "not measured here".
        if ctx.is_root() {
            if let Some(rl) = crate::runlog::log() {
                rl.scf_iter(
                    if xc_contrib.is_some() || inj_xc.is_some() {
                        "uks"
                    } else {
                        "uhf"
                    },
                    crate::runlog::current_rung(),
                    iter,
                    energy,
                    de,
                    mon.dp_rms,
                    mon.dp_max,
                    err_max,
                    None,
                );
            }
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
        // Two orbital-gradient amendments (see `UhfExitGate`): a UKS
        // degenerate-swap stationary point is accepted, a stagnated
        // non-stationary state is refused.
        let (conv_exit, swapped) = exit_gate.decide(
            &mut cosx_sched,
            UhfIterGradients {
                iter,
                sig,
                err_max,
                err: [&err_a, &err_b],
                x: &x,
                xc_active: [xc_contrib.is_some(), inj_xc.is_some()],
            },
            config,
            &mut diis,
            ctx.is_root(),
            &mut last_report,
        );

        // Divergence / stall early exits (shared driver::ScfMonitor; both are
        // no-ops at the None defaults — UHF previously ignored these knobs).
        if mon.diverging(energy, config.divergence_tol) || mon.stalled(err_max, config.stall_window)
        {
            return Err(FerricError::ScfConvergence {
                iterations: iter,
                last_energy: mon.prev_e,
            });
        }

        if iter > 1 && conv_exit.is_some() {
            // A converged SCF returns f64 V_xc (see `UksXcContribution::refine_last_to_f64_uks`).
            if let Some(x) = xc_contrib.as_ref() {
                x.refine_last_to_f64_uks(&mut f_a, &mut f_b);
            }
            // Returned MOs must reproduce the returned density: after a
            // degenerate swap the aufbau eigenvectors of F hold the OTHER
            // member of the pair. Only then (see `occupied_by_density`): a MOM
            // state is non-aufbau by design and keeps main's C and eps. Per
            // spin, and only for an integer density (`reorder_exit_mos`).
            // A plain ΔP exit can also settle on an integer D that is not the
            // aufbau state of F[D]; then the spin gets D's semicanonical
            // orbitals instead (`exit_mos_building_density`; not under MOM or
            // symmetry blocking, see `exit_eigenpairs`).
            let mom = config.mom_after_iter > 0;
            let (eps_a, c_a_f) = exit_eigenpairs(
                &diag_a,
                ExitSpin {
                    f: &f_a,
                    s: &s,
                    x: &x,
                    d: &d_a,
                    d_occ: d_occ_a.as_ref(),
                },
                nocc_a,
                swapped,
                mom,
            )?;
            let (eps_b, c_b_f) = exit_eigenpairs(
                &diag_b,
                ExitSpin {
                    f: &f_b,
                    s: &s,
                    x: &x,
                    d: &d_b,
                    d_occ: d_occ_b.as_ref(),
                },
                nocc_b,
                swapped,
                mom,
            )?;
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

            // ── Opt-in internal stability analysis (UHF/UKS) ─────────────────
            // Runs ONLY at a converged exit and ONLY when the flag is set; with
            // `check_stability = false` (the default) nothing below is
            // constructed, so this branch is bit-identical to a build with no
            // stability support. Diagnostic: it warns, it never Errs.
            let stability = if config.check_stability {
                stability_uhf(
                    ctx,
                    mol,
                    prep,
                    bounds,
                    config,
                    &c_a_f,
                    &c_b_f,
                    &f_a,
                    &f_b,
                    &d_a,
                    &d_b,
                    nocc_a,
                    nocc_b,
                    xc_contrib.is_some(),
                    k_mix,
                    ooc_budget,
                    fock_mod.is_some(),
                    rsh_response.as_ref(),
                )
            } else {
                None
            };
            // COSX final-grid pass (opt-in, `CosxConfig::final_grid`):
            // `k_a_buf`/`k_b_buf` hold K_scf(D_σ) from this iteration's build.
            let (energy, cosx_final) = match (pluggable_k_kind, config.cosx.final_grid.as_ref()) {
                (Some("cosx"), Some(fg)) => {
                    let rec = crate::cosx_k::run_final_pass(
                        ctx,
                        mol,
                        prep,
                        &config.cosx,
                        fg,
                        ooc_budget,
                        energy,
                        &[(&d_a, &k_a_buf, -0.5 * c_k), (&d_b, &k_b_buf, -0.5 * c_k)],
                    )?;
                    (rec.e_final, Some(rec))
                }
                _ => (energy, None),
            };
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
                induced_dipoles: last_induced_dipoles,
                stability,
                // No external check on a UHF reference: the UHF-internal
                // Hessian already spans the independent alpha/beta rotations
                // that an RHF->UHF check would open up, so there is no
                // additional spin-symmetry-breaking channel to test. (UHF->GHF
                // is a further external channel and is not implemented.)
                stability_external: None,
                df_jk: df_jk_route.clone(),
                rohf_spin_focks: None,
                cosx_final,
                cosx_schedule: crate::cosx_schedule::record(&cosx_sched),
                convergence: last_report,
            });
        }
        mon.note_energy(energy);

        // ── Trust-region augmented-Hessian (TRAH) update, UHF/UKS ────────────
        //
        // Opt-in via `config.trah_trigger`; `None` (the default) skips the whole
        // block, so the iteration is bit-identical to a build with no TRAH
        // support (`tests/trah_off_is_bit_identical.rs`). Structure, gates and
        // the two-phase ρ scheme are identical to the RHF branch in `rhf.rs` —
        // see its comment for the reasoning. The α and β rotations are solved as
        // ONE coupled trust-region problem (see `crate::trah::uhf_trah_step`).
        // Armed below the trigger -- OR once TRAH has ENGAGED, until it declines
        // a step as worthless or its radius collapses. A step may legitimately
        // RAISE err_max (a hard-case escape moves along negative curvature,
        // away from a stationary point). Re-deciding on err_max alone handed
        // the next iteration to DIIS, which, with a history built at the
        // saddle, walked straight back into it: tail-mode TRAH cycled to
        // max_iter on N2/cc-pVDZ (pinned in `tests/trah_hard_case.rs`).
        // Keying on "a step is pending" closed that only for accepted steps:
        // a REJECTION consumes the pending prediction, so the restored point
        // (err_max still above the trigger) went to DIIS. Measured on
        // CoH/def2-SVP UHF (`tests/saddle_repair_compare.rs`), that threw a
        // 100 mHa escape back onto the saddle. Pinned by the N2/STO-3G RHF and
        // O2/STO-3G UHF hand-off-after-rejection cases in `trah_hard_case.rs`.
        let trah_armed = config
            .trah_trigger
            .is_some_and(|t| err_max < t || trah_engaged)
            && iter > 3
            && k_mix.omega == 0.0
            && !crate::rohf::xc_is_metagga(config.xc.as_deref())
            && !fractional_blocks_second_order;
        // One condition for "a TRAH step runs", used BOTH to gate the step and
        // to discard a stale prediction — see the RHF counterpart for why these
        // must not diverge (a radius that collapses mid-run would otherwise
        // leave a prediction pending across intervening DIIS steps).
        let trah_collapsed = trah_state.as_ref().is_some_and(|s| s.collapsed());
        let trah_runs = trah_armed && !trah_collapsed;
        let trah_after_rejection = crate::trah::settle_engaged_latch(
            &mut trah_engaged,
            &mut trah_last_rejected,
            trah_runs,
        );
        if let Some(st) = trah_state.as_mut() {
            if !trah_runs {
                if st.has_pending() {
                    crate::trah::note_trah_prediction_discarded();
                }
                st.clear_pending();
                trah_undo = None;
            }
        }
        let mut trah_took_step = false;
        if trah_runs {
            // Phase 1: score the pending step against the energy just computed.
            // `assess` returns None when NOTHING was pending — the first armed
            // iteration, and the iteration right after a rejection (which
            // consumed its pending step and skipped Phase 2). That is not an
            // acceptance: treating it as one and then printing the PERSISTENT
            // `last_rho` made the trace show a frozen ρ repeated across a
            // reject/"accept" alternation that was not happening. Keep the
            // Option so "no verdict" stays distinguishable from "accepted".
            let verdict: Option<crate::trah::TrahVerdict> =
                trah_state.as_mut().and_then(|st| st.assess(energy));
            if let (Some(_), Some(rho)) = (verdict, trah_state.as_ref().and_then(|s| s.last_rho)) {
                crate::trah::note_rho_uhf(rho);
            }
            if crate::rhf::scf_trace() {
                if let (Some(v), Some(rho)) =
                    (verdict, trah_state.as_ref().and_then(|s| s.last_rho))
                {
                    eprintln!(
                        "TRAH iter={iter}: rho={rho:.6} verdict={v:?} \
                         Delta={:.3e} acc={} rej={}",
                        trah_state.as_ref().map(|s| s.radius()).unwrap_or(0.0),
                        trah_state.as_ref().map(|s| s.accepted).unwrap_or(0),
                        trah_state.as_ref().map(|s| s.rejected).unwrap_or(0),
                    );
                }
            }
            if verdict == Some(crate::trah::TrahVerdict::Rejected) {
                crate::trah::note_trah_rejection();
                // `trah_engaged` is already set: the rejected step set it when
                // it was taken, and only a hand-off or a decline (both of which
                // clear the pending prediction) releases it. So the restored
                // point stays armed whatever its err_max.
                trah_last_rejected = true;
                if let Some((ca_undo, cb_undo)) = trah_undo.take() {
                    if crate::rhf::scf_trace() {
                        eprintln!(
                            "TRAH iter={iter}: REJECTED (ρ={:.3e}), restoring orbitals, Δ→{:.3e}",
                            trah_state.as_ref().and_then(|s| s.last_rho).unwrap_or(0.0),
                            trah_state.as_ref().map(|s| s.radius()).unwrap_or(0.0)
                        );
                    }
                    // Undo, recording the density motion so the convergence
                    // monitor sees it (an undo hidden from dp_rms could let the
                    // loop "converge" on a density it had just discarded).
                    let d_tot_old = &d_a + &d_b;
                    c_a = ca_undo;
                    c_b = cb_undo;
                    d_a = density(&c_a, nocc_a);
                    d_b = density(&c_b, nocc_b);
                    // Keep the DF-K C_occ cache on the restored orbitals (see
                    // the TRAH step below for why every C update needs this).
                    d_occ_a = occ_or_none(&c_a, nocc_a);
                    d_occ_b = occ_or_none(&c_b, nocc_b);
                    let d_tot_new = &d_a + &d_b;
                    mon.record_density_change(&d_tot_new, &d_tot_old);
                }
            }

            // Phase 2: step from the (possibly restored) point.
            //
            // After a REJECTION we do not step this iteration: `f` was built
            // from the density we just discarded, so an MO-basis Fock formed
            // from it would be a gradient at the wrong point. The loop
            // `continue`s, rebuilds F from the restored density, and steps on
            // the NEXT iteration at the already-contracted radius. That costs
            // one Fock build per rejection and is why rejection is not free.
            //
            // The pending assessment was consumed by `assess` above, so the
            // skipped iteration records nothing and the next ρ is formed from a
            // matched (energy_before, predicted) pair — not from a stale one.
            // Set when the step is declined as worthless (see the
            // `predicted_min` check below); the loop then falls through to DIIS.
            let mut declined = false;
            if verdict != Some(crate::trah::TrahVerdict::Rejected) {
                let f_a_mo = c_a.t().dot(&f_a).dot(&c_a);
                let f_b_mo = c_b.t().dot(&f_b).dot(&c_b);

                // Same duplicate-grid-cache accounting the Newton branch does —
                // `FxcKernelStore::build` allocates a SECOND chi+dchi cache on
                // the live KS grid, and `check_ao_grid_budget` cannot see it.
                let fxc_store = if xc_contrib.is_some() {
                    let main = config.dft_grid.clone().unwrap_or_default();
                    let name = config.xc.as_deref().expect("xc_contrib implies Some(xc)");
                    let needed = fxc_kernel_duplicate_bytes(n, mol.atoms.len(), &main);
                    let avail = ferric_core::memory::available_budget_now(
                        ferric_core::memory::resolve_budget_bytes(
                            (config.three_index_budget_bytes != 0)
                                .then_some(config.three_index_budget_bytes),
                        ),
                    );
                    ferric_core::memory::check_alloc(
                        "UHF/UKS TRAH f_xc kernel (duplicate chi+dchi cache on the live KS grid)",
                        needed,
                        avail,
                    )?;
                    Some(crate::rohf::FxcKernelStore::build(
                        mol,
                        prep,
                        &main,
                        name,
                        config.xc_omega,
                        &d_a,
                        &d_b,
                    )?)
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
                    rsh: rsh_response.as_ref(),
                    fxc: fxc_ref,
                    thresh: config.integral_thresh,
                    ooc_budget,
                };
                let radius = trah_state
                    .as_ref()
                    .map(|s| s.radius())
                    .expect("trah_state is Some inside the armed branch");
                let (ca_new, cb_new, step) =
                    crate::trah::uhf_trah_step(ctx, &inputs, radius, &config.trah)?;

                // Decline a step the model says is worthless, as the RHF loop
                // does (`TrahConfig::predicted_min`). Measured on UKS/PBE
                // OH/cc-pVDZ: from iteration 17 the model predicted -6.0e-15,
                // the energy moved by exactly 0, rho = 0 rejected it, and the
                // identical step was recomputed 28 times (74 iterations against
                // DIIS's 33). Declining falls through to DIIS, and recording a
                // zero density change lets the normal convergence test end the run.
                if step.predicted.abs() < config.trah.predicted_min {
                    if crate::rhf::scf_trace() {
                        eprintln!(
                            "TRAH iter={iter}: predicted |{:.3e}| < {:.0e}, \
                             nothing left to gain -- deferring to DIIS",
                            step.predicted, config.trah.predicted_min
                        );
                    }
                    if let Some(st) = trah_state.as_mut() {
                        st.clear_pending();
                    }
                    trah_undo = None;
                    crate::trah::note_trah_null_step_declined();
                    trah_engaged = false;
                    let d_tot = &d_a + &d_b;
                    mon.record_density_change(&d_tot, &d_tot);
                    declined = true;
                } else {
                    if crate::rhf::scf_trace() {
                        eprintln!(
                            "TRAH iter={iter}: ‖κ‖={:.3e} Δ={:.3e} μ={:.3e} α={:.1} \
                         ΔE_pred={:.3e} solves={} boundary={}",
                            step.norm,
                            radius,
                            step.level_shift,
                            step.alpha,
                            step.predicted,
                            step.shift_iterations,
                            step.on_boundary
                        );
                    }

                    trah_undo = Some((c_a.clone(), c_b.clone()));
                    if let Some(st) = trah_state.as_mut() {
                        st.record_step(energy, &step);
                    }
                    crate::trah::note_trah_step();
                    trah_engaged = true;

                    c_a = ca_new;
                    c_b = cb_new;
                    let d_tot_old = &d_a + &d_b;
                    d_a = density(&c_a, nocc_a);
                    d_b = density(&c_b, nocc_b);
                    // DF-K builds K_σ from this C_occ cache, J from d_σ. Before
                    // 2026-10-24 only the DIIS path refreshed it, so after a
                    // TRAH step (or restore, or Newton step) the next Fock had
                    // K at the last DIIS orbitals and J at the current ones --
                    // a wrong energy and a wrong ρ. RHF refreshes `d_occ` at
                    // every C update; this mirrors it.
                    d_occ_a = occ_or_none(&c_a, nocc_a);
                    d_occ_b = occ_or_none(&c_b, nocc_b);
                    let d_tot_new = &d_a + &d_b;
                    mon.record_density_change(&d_tot_new, &d_tot_old);
                }
            }
            trah_took_step = !declined;
        }
        if trah_took_step {
            continue;
        }
        // TRAH did not step this iteration (not run, or ran and declined): the
        // iteration goes to Newton/DIIS. Counted here, not at the gate, so a
        // decline right after a rejection is seen too.
        crate::trah::note_handoff_if_after_rejection(trah_after_rejection, trah_collapsed);

        // ── Second-order (Newton) update, UHF/UKS ────────────────────────────
        // When enabled (newton_trigger > 0) and err_max has dropped below the
        // trigger, take a damped-Newton step on the α/β orbital rotations
        // instead of DIIS. For UKS this uses the SAME LDA/GGA f_xc kernel the
        // ROKS Newton path uses (via FxcKernelStore), so PBE/B3LYP/etc. UKS now
        // gets real second-order acceleration, not just LDA. Gated to the
        // non-RSH case (ω = 0). The REASON is no longer the matvec's kernel:
        // since #292 the matvec builds `c_SR·δK[erfc(ω)] + c_LR·δK[erf(ω)]`
        // from the converged Fock's own DF-K fitters (`crate::rsh_response`,
        // threaded via `UhfNewtonInputs::rsh`), and that response is
        // finite-difference-validated at ω ≠ 0 to 5.3e-12 and matches PySCF's
        // dense UKS Hessian on O2/def2-SVP/ωB97X-V to 1.0e-7 relative. The gate
        // stays only because LIFTING it changes the SCF trajectory of every RSH
        // run, which needs its own convergence measurement (iteration counts,
        // no regressions across the RSH test set) that has not been made.
        // #292 lists the lift as optional and as a separate commit; it is not
        // a correctness blocker any more, and stability analysis — which drives
        // the same matvec through Davidson — is already unblocked.
        // Meta-GGA (SCAN / r2SCAN) has no τ-dependent f_xc kernel in Phase A —
        // exclude it from the Newton path so it falls back to DIIS (energy-only).
        let use_newton = config.newton_trigger > 0.0
            && iter > 3
            && err_max < config.newton_trigger
            && k_mix.omega == 0.0
            && !crate::rohf::xc_is_metagga(config.xc.as_deref())
            && !fractional_blocks_second_order;
        if use_newton {
            let f_a_mo = c_a.t().dot(&f_a).dot(&c_a);
            let f_b_mo = c_b.t().dot(&f_b).dot(&c_b);

            // Build the f_xc kernel (LDA or GGA) + reference density once per
            // Newton step (None for pure UHF). The response closure borrows it.
            //
            // `FxcKernelStore::build` -> `GgaFxcKernel::new`/`LdaFxcKernel::new`
            // (ferric-dft/src/fxc.rs) calls `eval_basis_and_grad_on_points` on
            // THIS SAME `main` grid config, i.e. it allocates a SECOND
            // (nbf, npts) chi + (3, nbf, npts) dchi cache — 4 planes — that
            // duplicates the one already resident in `xc_contrib` (the KS grid
            // cache built once above, at construction, and held live for the
            // whole SCF loop; it is not dropped before this branch runs).
            // Measured at the shapes this crate's other memory findings use:
            // benzene/aug-cc-pVTZ (nbf=414, ~12 atoms, npts=12*8250=99000) one
            // plane is 414*99000*8 = 328 MB, so the duplicate 4-plane kernel
            // costs 1.31 GB on top of the 1.31 GB KS cache already resident
            // (2.62 GB peak where either gate alone only ever saw 1.31 GB);
            // danuglipron/def2-SVP (nbf=700, 73 atoms, npts=73*8250=602250)
            // one plane is 700*602250*8 = 3.37 GB, so the duplicate kernel
            // costs 13.49 GB on top of the 13.49 GB KS cache (26.98 GB peak).
            // `eval_basis_and_grad_on_points`'s own `check_ao_grid_budget`
            // gate cannot see any of this: it re-resolves
            // `resolve_budget_bytes(None)` — the WHOLE ceiling, ignoring
            // `config.three_index_budget_bytes` AND the live KS cache — so it
            // independently re-approves the duplicate against 100% of the
            // budget the KS cache was already charged against.
            //
            // Gate it here instead, the way `ks.rs::new_with_omega_budgeted`
            // gates its OWN grid cache: `available_budget_now` reads this
            // process's live RSS (which by now already includes the resident
            // KS cache) and reserves the same 0.9 headroom fraction PySCF
            // uses, so a duplicate that would not actually fit is a clean
            // `Err` here rather than a silent OOM three calls deeper. This
            // does not fix the duplication itself (that requires teaching
            // `GgaFxcKernel`/`LdaFxcKernel` to borrow `xc_contrib`'s cache
            // instead of rebuilding it, which lives in ferric-dft, out of
            // scope for this file) — it only makes the ALREADY-EXISTING
            // second allocation honestly accounted for before it happens.
            let fxc_store = if xc_contrib.is_some() {
                let main = config.dft_grid.clone().unwrap_or_default();
                let name = config.xc.as_deref().expect("xc_contrib implies Some(xc)");
                let needed = fxc_kernel_duplicate_bytes(n, mol.atoms.len(), &main);
                let avail = ferric_core::memory::available_budget_now(
                    ferric_core::memory::resolve_budget_bytes(
                        (config.three_index_budget_bytes != 0)
                            .then_some(config.three_index_budget_bytes),
                    ),
                );
                ferric_core::memory::check_alloc(
                    "UHF/UKS Newton f_xc kernel (duplicate chi+dchi cache on the live KS grid)",
                    needed,
                    avail,
                )?;
                Some(crate::rohf::FxcKernelStore::build(
                    mol,
                    prep,
                    &main,
                    name,
                    config.xc_omega,
                    &d_a,
                    &d_b,
                )?)
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
                rsh: rsh_response.as_ref(),
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
            d_occ_a = occ_or_none(&c_a, nocc_a);
            d_occ_b = occ_or_none(&c_b, nocc_b);
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
        let (mut eps_a_new, mut c_a_new) = diag_a.run(&f_a_new)?;
        let (mut eps_b_new, mut c_b_new) = diag_b.run(&f_b_new)?;

        // MOM occupied-orbital selection (per spin) from iter mom_after_iter+1.
        if config.mom_after_iter > 0 && iter > config.mom_after_iter {
            (c_a_new, eps_a_new) = mom_pin(c_a_new, eps_a_new, &s, mom_ref_a.as_ref(), nocc_a);
            (c_b_new, eps_b_new) = mom_pin(c_b_new, eps_b_new, &s, mom_ref_b.as_ref(), nocc_b);
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
        // Density plus the cached occupied MO block for the next iteration's
        // DF-K half-transform; the cache is None whenever the ensemble density
        // is genuinely fractional (see spin_density_and_occ).
        (d_a, d_occ_a) = spin_density_and_occ(&c_a, &eps_a_new, nocc_a, config.fractional_occ);
        (d_b, d_occ_b) = spin_density_and_occ(&c_b, &eps_b_new, nocc_b, config.fractional_occ);
        // ΔP over the total density (α+β) — consumed at the top of the next
        // iteration by rhf::scf_converged. Drains to zero at the RI fixed point
        // even when the DIIS commutator parks on the naux noise floor.
        let d_tot_new = &d_a + &d_b;
        mon.record_density_change(&d_tot_new, &d_tot_old);
    }
    // Max iterations reached.
    //
    // Historically this discarded everything and returned a bare Err, which made an
    // open-shell convergence LADDER impossible: a ladder works by carrying the
    // best-effort density of a failed rung forward as the next rung's guess (exactly
    // what `solve_rhf_ladder` does via RHF's `build_nonconverged`).
    //
    // So the best-effort state is now BUILT here, and the caller decides what to do
    // with it. `solve_uhf` / `solve_uhf_with_guess` keep their historical Err contract
    // -- every existing caller unwraps or `?`s them, and silently handing those an
    // unconverged result would be a regression. `solve_uhf_best_effort` returns it.
    //
    // Orbital energies and Fock matrices are not retained across the loop (they are
    // rebuilt each iteration), so the core-Hamiltonian eigenvalues stand in. The
    // density and MOs -- the parts a ladder restart actually consumes -- are the
    // genuine converged-so-far values.
    let (eps_a_last, eps_b_last) = match diagonalize(&h, &x) {
        Ok((e, _)) => (e.clone(), e),
        Err(_) => (vec![0.0; c_a.ncols()], vec![0.0; c_a.ncols()]),
    };
    let density_total = &d_a + &d_b;
    Ok(ScfResult {
        spin: Spin::Unrestricted,
        energy: mon.prev_e,
        density_total,
        density_alpha: d_a,
        density_beta: Some(d_b),
        mos_alpha: c_a,
        mos_beta: Some(c_b),
        eps_alpha: eps_a_last,
        eps_beta: Some(eps_b_last),
        fock_alpha: h.clone(),
        fock_beta: Some(h.clone()),
        converged: false,
        exit: ScfExit::MaxIter,
        iterations: config.max_iter,
        computed_quartets: total_quartets,
        induced_dipoles: last_induced_dipoles,
        stability: None,
        stability_external: None,
        df_jk: df_jk_route,
        rohf_spin_focks: None,
        cosx_final: None,
        cosx_schedule: crate::cosx_schedule::record(&cosx_sched),
        convergence: last_report,
    })
}

/// Post-convergence internal stability analysis for a UHF/UKS solution.
///
/// Called ONLY from `solve_uhf_fockmod`'s converged exit and ONLY when
/// `config.check_stability` is set. Returns `None` — meaning "not checked", per
/// [`crate::result::ScfResult::stability`] — whenever the reference is not
/// analysable with the operator that exists, ALWAYS after printing why.
///
/// # The KS trap this function exists to avoid
///
/// [`crate::uhf_newton::hessian_matvec`] takes an OPTIONAL `fxc` response
/// closure. Passing `None` on a KS reference does not fail; it silently
/// analyses the **HF** orbital Hessian at the **KS** density, producing a
/// λ_min for an operator nobody asked about, presented with the same
/// confidence as a correct one. So for `xc.is_some()` this builds the SAME
/// [`crate::rohf::FxcKernelStore`] the UKS Newton branch a few hundred lines
/// above builds at the same (d_α, d_β) reference, and where that kernel cannot
/// be built — range-separated or meta-GGA — it SKIPS with a printed reason.
/// Those are exactly the gates the UKS Newton branch itself uses.
#[allow(clippy::too_many_arguments)]
fn stability_uhf<'r>(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &UhfConfig,
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    f_a: &Array2<f64>,
    f_b: &Array2<f64>,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    nocc_a: usize,
    nocc_b: usize,
    has_xc: bool,
    k_mix: ferric_dft::xc_trait::KMix,
    ooc_budget: usize,
    fock_modified: bool,
    rsh: Option<&'r crate::rsh_response::RshResponse<'r>>,
) -> Option<crate::stability::StabilityResult> {
    let skip = if fock_modified {
        Some(crate::stability::StabilitySkip::FockModified)
    } else {
        crate::stability::ks_reference_is_analysable(config.xc.as_deref(), k_mix.omega).err()
    };
    if let Some(skip) = skip {
        eprintln!(
            "SCF stability: check requested but SKIPPED — {}. \
             ScfResult::stability is None (not checked), which does NOT mean stable.",
            skip.reason()
        );
        return None;
    }

    let fxc_store = if has_xc {
        let grid = config.dft_grid.clone().unwrap_or_default();
        let name = config.xc.as_deref().expect("has_xc implies Some(xc)");
        match crate::rohf::FxcKernelStore::build(mol, prep, &grid, name, config.xc_omega, d_a, d_b)
        {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!(
                    "SCF stability: check requested but SKIPPED — the f_xc response kernel could \
                     not be built ({e}), and analysing the HF Hessian at a KS density instead \
                     would be a wrong-operator verdict. ScfResult::stability is None."
                );
                return None;
            }
        }
    } else {
        None
    };
    let fxc_storage = fxc_store.as_ref().map(|s| s.response());
    let fxc_ref: Option<&crate::rohf_newton::FxcResponse<'_>> = fxc_storage.as_deref();

    let f_a_mo = c_a.t().dot(f_a).dot(c_a);
    let f_b_mo = c_b.t().dot(f_b).dot(c_b);
    let inputs = crate::uhf_newton::UhfNewtonInputs {
        prep,
        bounds,
        c_a,
        c_b,
        f_a_mo: &f_a_mo,
        f_b_mo: &f_b_mo,
        nocc_a,
        nocc_b,
        k_mix_sr: if has_xc { k_mix.sr } else { 1.0 },
        rsh,
        fxc: fxc_ref,
        thresh: config.integral_thresh,
        ooc_budget,
    };
    match crate::stability::uhf_internal_stability(
        ctx,
        &inputs,
        &crate::stability::StabilityConfig::default(),
    ) {
        Ok(res) => {
            crate::stability::report_stability(&res, config.verbose);
            Some(res)
        }
        Err(e) => {
            eprintln!(
                "SCF stability: check requested but FAILED — {}: {e}. ScfResult::stability is \
                 None (not checked). The SCF result itself is unaffected.",
                crate::stability::StabilitySkip::AnalysisFailed.reason()
            );
            None
        }
    }
}

/// Bytes the Newton f_xc kernel's own (chi, dchi) cache costs: 4 planes
/// (`AoGridKind::ValueAndGrad` — 1 for chi, 3 for dchi's x/y/z) of
/// `nbf * npts` `f64`s, where `npts = natoms * cfg.n_radial * cfg.n_angular`
/// (the UNPRUNED atomic grid `GgaFxcKernel::new`/`LdaFxcKernel::new` build via
/// `build_atomic_grid`, not `build_atomic_grid_pruned`).
///
/// A pure function of shape only (no rayon/env ambient state) so it is
/// testable without constructing a molecule, basis, or grid. This is exactly
/// the allocation that duplicates the live `xc_contrib` KS grid cache — see
/// the call site's comment in `solve_uhf_fockmod` for the measured magnitude
/// at benzene/aug-cc-pVTZ and danuglipron/def2-SVP shapes.
pub fn fxc_kernel_duplicate_bytes(
    nbf: usize,
    natoms: usize,
    cfg: &ferric_dft::grid::AtomicGridConfig,
) -> usize {
    const PLANES: usize = 4; // chi (1) + dchi x/y/z (3)
    let npts = natoms
        .saturating_mul(cfg.n_radial)
        .saturating_mul(cfg.n_angular);
    PLANES
        .saturating_mul(nbf)
        .saturating_mul(npts)
        .saturating_mul(std::mem::size_of::<f64>())
}

/// For a "forced" UHF on a closed-shell occupation (`nocc_a == nocc_b`), mix the
/// beta HOMO and LUMO by a small angle so the solver can leave the restricted
/// saddle. Skipped under point-group blocking: a blocked closed-shell UHF has
/// identical alpha and beta blocks and cannot leave `alpha = beta`, so the mix
/// would only be projected away (symmetry-broken solutions are outside the
/// blocked space by construction).
fn mix_beta_homo_lumo(c_b: &mut Array2<f64>, nocc_a: usize, nocc_b: usize, blocked: bool) {
    let n = c_b.nrows();
    if blocked || nocc_a != nocc_b || nocc_a == 0 || nocc_a >= n {
        return;
    }
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

/// Largest energy change (Ha) between the two members of an accepted
/// degenerate swap, whatever `energy_conv` is (its default, 1e-3, would admit
/// two distinct states).
const SWAP_MAX_DE: f64 = 1e-5;

/// Lower limit of the gradient bound for a degenerate-swap acceptance:
/// `max(density_conv, 1e-6)`.
///
/// The bound cannot be `density_conv` itself: the swapping state's orbital
/// gradient floors on the arithmetic, and the floor straddles a tight
/// `density_conv`. MEASURED on OH UKS-LDA/STO-3G at the stationary ground
/// state: 9.1e-10 (OpenBLAS Nehalem kernels), 1.1e-9 (Prescott, the CI
/// kernels), 2.2e-9 (native FMA). At `density_conv` 1e-9 the Prescott run
/// therefore never accepted the swap, cycled ground ↔ stagnated state, and
/// ended at `max_iter` (Err) on the RIGHT energy. 1e-6 sits three orders
/// above those floors and three below the veto floor (1e-3). Accepting at
/// this bound loses nothing that `density_conv` could buy here: the swapping
/// density is only defined up to the degenerate rotation (it changes by O(1)
/// every iteration), and the energy error of a gradient g is O(g²); the ΔE
/// bound still applies at full strength.
const SWAP_GRADIENT_FLOOR: f64 = 1e-6;

/// One iteration's inputs to [`UhfExitGate::decide`].
struct UhfIterGradients<'a> {
    iter: usize,
    sig: crate::rhf::ConvergenceSignals,
    /// max over spins of |F_σD_σS − SD_σF_σ| (AO basis, pre-DIIS).
    err_max: f64,
    /// The per-spin AO commutators.
    err: [&'a Array2<f64>; 2],
    /// The lindep-filtered canonical orthogonalizer.
    x: &'a Array2<f64>,
    /// Whether an XC functional is active: (the SCF's own, an injected one).
    /// Either makes this Kohn-Sham.
    xc_active: [bool; 2],
}

/// The UHF convergence decision: the shared ΔP/ΔE gate
/// ([`crate::rhf::scf_converged`]) with two orbital-gradient amendments,
/// MEASURED on OH UKS-LDA/STO-3G (`tests/uhf_degenerate_swap.rs`):
/// 1. a UKS degenerate swap at a stationary point is accepted
///    ([`degenerate_swap_converged`]) — the OH β hole flips between π_x and
///    π_y every iteration, so ΔP_max = 1 and the plain gate never passes at
///    the true ground state (−74.0755553242, `err_max` ~2e-9);
/// 2. a ΔP/ΔE-settled state that is not stationary is refused and the DIIS
///    history cleared (`rhf::GradientVeto`, on the orthonormal-basis
///    gradient) — without it DIIS stagnated at that system into a
///    non-aufbau state 0.586 Ha high (`err_max` 0.136), froze D, and the
///    plain gate accepted it.
#[derive(Debug, Default)]
struct UhfExitGate {
    veto: crate::rhf::GradientVeto,
    /// The previous iteration's AO `err_max` (∞ before the first).
    prev_err_max: Option<f64>,
}

impl UhfExitGate {
    /// The exit to take, and whether it is a degenerate-swap acceptance (the
    /// only exit whose MOs may be reordered; per spin, [`reorder_exit_mos`]).
    /// Not refused wholesale under `fractional_occ`: when the frontier is not
    /// degenerate `density_fractional` spreads nothing and the density is the
    /// integer `C_occ·C_occᵀ`, which the reorder must still match.
    fn decide(
        &mut self,
        cosx_sched: &mut Option<crate::cosx_schedule::ScheduleRun>,
        it: UhfIterGradients<'_>,
        config: &UhfConfig,
        diis: &mut Diis,
        announce: bool,
        report: &mut Option<crate::result::ConvergenceReport>,
    ) -> (Option<ScfExit>, bool) {
        let (econv, dconv) = (config.energy_conv, config.density_conv);
        let prev_err_max = self
            .prev_err_max
            .replace(it.err_max)
            .unwrap_or(f64::INFINITY);
        let plain = crate::rhf::scf_converged(it.sig, econv, dconv);
        let swap = plain.is_none()
            && it.xc_active.contains(&true)
            && degenerate_swap_converged(it.sig, it.err_max, prev_err_max, it.iter, econv, dconv);
        let settled = if swap {
            Some(ScfExit::Converged)
        } else {
            plain
        };
        let gated = crate::cosx_schedule::gate(cosx_sched, it.iter, it.sig.dp_max, settled);
        let orbital_gradient = || {
            it.err
                .iter()
                .map(|e| crate::rhf::orthogonal_gradient_max(it.x, e))
                .fold(0.0f64, f64::max)
        };
        let (exit, vetoed) = self.veto.apply(
            gated,
            orbital_gradient,
            it.iter,
            dconv,
            "UHF",
            announce,
            report,
        );
        if vetoed {
            diis.reset();
        }
        (exit, swap && exit.is_some())
    }

    fn record(&self) -> Option<crate::result::GradientVetoes> {
        self.veto.record()
    }
}

/// A stationary point whose occupied set swaps between degenerate orbitals on
/// every iteration.
///
/// OH doublet: the β hole flips between π_x and π_y each step, so ΔP_max = 1
/// and [`crate::rhf::scf_converged`] can never pass, although each density is
/// stationary (`err_max` ~1e-9 MEASURED at −74.0755553242). Accept when the
/// density moved by an orbital-sized amount (`dp_max > 0.1`) while the
/// orbital gradient is below `max(density_conv, SWAP_GRADIENT_FLOOR)` at BOTH
/// ends of the swap (this iteration and the previous one) and the energy
/// moved by less than
/// `min(energy_conv, 1e-5)`: a swap between degenerate orbitals at a
/// stationary point, not a descending run or a hop between distinct states.
/// The caller restricts it to Kohn-Sham: a UHF (Hartree-Fock) minimiser is
/// aufbau, so a swapping HF state is not one to accept.
fn degenerate_swap_converged(
    sig: crate::rhf::ConvergenceSignals,
    err_max: f64,
    prev_err_max: f64,
    iter: usize,
    energy_conv: f64,
    density_conv: f64,
) -> bool {
    let bound = density_conv.max(SWAP_GRADIENT_FLOOR);
    iter > 2
        && sig.dp_max > 0.1
        && err_max < bound
        && prev_err_max < bound
        && sig.de < energy_conv.min(SWAP_MAX_DE)
}

/// Whether one spin's exit MOs are reordered to build its density: after a
/// degenerate-swap acceptance (`swapped`, from [`UhfExitGate::decide`]) and
/// only when that spin's density is integer. `d_occ` is the occupied-block
/// cache that built the exit density, `Some` exactly when `D = C_occ·C_occᵀ`
/// (see `spin_density_and_occ`); a fractional ensemble density has no integer
/// occupied set to identify, so it keeps the aufbau eigenpairs. A MOM exit is
/// never a swap acceptance, so it is never reordered.
fn reorder_exit_mos(swapped: bool, d_occ: Option<&Array2<f64>>) -> bool {
    swapped && d_occ.is_some()
}

/// Order one spin's eigenpairs so the first `nocc` columns are the
/// eigenvectors carrying the most weight in `d` (`w_p = c_pᵀ S D S c_p`),
/// each block kept in ascending eigenvalue order with `eps` permuted to match.
///
/// Applied (`apply`) only to an exit accepted as a degenerate swap
/// ([`degenerate_swap_converged`]): there `d` occupies the other member of
/// the degenerate pair, and returning the aufbau `C` would hand downstream
/// code (gradients, post-HF) occupied orbitals that do not build the returned
/// density. Every other exit returns the aufbau eigenpairs unchanged — in
/// particular a MOM state, which is non-aufbau BY DESIGN and whose consumers
/// (and the Hessian's non-aufbau refusal) rely on main's C and eps. Not
/// `mom::mom_reorder`, which scrambles the occupied order. The reordered
/// result has an occupied eigenvalue above a virtual one, which is how
/// `hessian::uhf_state_refusal` recognises it.
fn occupied_by_density(
    (eps, c): (Vec<f64>, Array2<f64>),
    s: &Array2<f64>,
    d: &Array2<f64>,
    nocc: usize,
    apply: bool,
) -> (Vec<f64>, Array2<f64>) {
    let n = c.ncols();
    if !apply || nocc == 0 || nocc >= n {
        return (eps, c);
    }
    let sc = s.dot(&c);
    let w = (sc.t().dot(d) * sc.t()).sum_axis(ndarray::Axis(1));
    let mut by_weight: Vec<usize> = (0..n).collect();
    by_weight.sort_by(|&p, &q| w[q].total_cmp(&w[p]).then(p.cmp(&q)));
    let mut occ = by_weight[..nocc].to_vec();
    occ.sort_unstable();
    if occ.iter().enumerate().all(|(k, &p)| k == p) {
        return (eps, c);
    }
    let mut vir = by_weight[nocc..].to_vec();
    vir.sort_unstable();
    let order: Vec<usize> = occ.into_iter().chain(vir).collect();
    let mut c_out = Array2::<f64>::zeros(c.raw_dim());
    for (k, &p) in order.iter().enumerate() {
        c_out.column_mut(k).assign(&c.column(p));
    }
    (order.iter().map(|&p| eps[p]).collect(), c_out)
}

/// One spin's returned eigenpairs at a converged exit: F[D] diagonalised,
/// reordered after a degenerate swap (`occupied_by_density`), then replaced
/// by D's semicanonical orbitals when D is not the aufbau state of F[D]
/// (`exit_mos_building_density`). `mom` is whether MOM is configured.
///
/// A symmetry-blocked run (`FockDiag` with a `BlockedDiag`) is never
/// replaced: the blocked diagonaliser records each column's irrep on the
/// spec as it runs (`symmetric_open::solve_uhf_symmetric` reads them back as
/// `alpha_irreps`/`beta_irreps`), so new columns would carry stale labels, and
/// its per-irrep `U_h X_h` bases are not the global X the construction
/// projects in. The blocked solver fixes occupations per irrep, which keeps
/// the two π of a linear radical in different irreps in the first place.
fn exit_eigenpairs(
    diag: &crate::symmetric::FockDiag<'_>,
    spin: ExitSpin<'_>,
    nocc: usize,
    swapped: bool,
    mom: bool,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    let eig = occupied_by_density(
        diag.run(spin.f)?,
        spin.s,
        spin.d,
        nocc,
        reorder_exit_mos(swapped, spin.d_occ),
    );
    if diag.is_blocked() {
        return Ok(eig);
    }
    exit_mos_building_density(eig, spin, mom)
}

/// Largest `max|C_occ C_occᵀ − D|` an exit's eigenpairs may show before the
/// exit even checks whether D is the aufbau state of F[D].
///
/// A cheap prefilter, not the decision: `rhf::scf_converged` accepts
/// `dp_max < 10·density_conv`, and D is built from the PREVIOUS iteration's
/// MOs while C diagonalises F[D], so an ordinary exit misses D by up to about
/// that (MEASURED 3.5e-8 for NH2 UKS-PBE at density_conv 1e-6, 6e-6..9e-6 for
/// the aufbau α of SH/HCl+ UKS-LDA, and above 1e-3 for SH UHF at density_conv
/// 1e-4). Below the prefilter nothing is computed, so tight runs stay
/// bit-identical at no cost; above it, the decision is the aufbau test in
/// `exit_mos_building_density`, which does not depend on `density_conv`.
const EXIT_MO_DENSITY_GATE: f64 = 1e-3;

/// One spin's exit inputs to `exit_mos_building_density`: its Fock F[D],
/// the AO overlap, the canonical orthogonaliser X (n × m), the exit density,
/// and its occupied-block cache (`Some` exactly when `D = C_occ C_occᵀ`, see
/// `spin_density_and_occ`).
struct ExitSpin<'a> {
    f: &'a Array2<f64>,
    s: &'a Array2<f64>,
    x: &'a Array2<f64>,
    d: &'a Array2<f64>,
    d_occ: Option<&'a Array2<f64>>,
}

/// Make one spin's returned MOs build its returned integer density when D is
/// not the aufbau state of F[D].
///
/// `eig` are the exit eigenpairs (aufbau, or swap-reordered by
/// `occupied_by_density`). A plain ΔP/ΔE exit can settle on a state whose D
/// is the integer density of the previous iteration's MOs but whose occupied
/// space is not the aufbau space of F[D] (UKS-LDA SH/HCl+: no integer β state
/// is aufbau, DIIS stagnates under the gradient-veto floor). The aufbau
/// eigenvectors then occupy a DIFFERENT orbital (the other π) and hand
/// gradients and post-SCF code MOs of another state.
///
/// Decision: when `eig` miss D by more than `EXIT_MO_DENSITY_GATE`, build D's
/// semicanonical orbitals (`semicanonical_orbitals`) and return them only if
/// their eigenvalues CROSS (an occupied one above a virtual one), i.e. D is
/// not the aufbau state of F[D]. A loosely converged aufbau state (a moderate
/// rotation of the aufbau space, eigenvalues still ordered) keeps F's
/// eigenpairs: its miss is the convergence scale, and the analytic Hessian
/// refuses it through its MO-density consistency check. A replaced result
/// rebuilds D to rounding with D's energy unchanged, and the crossing is what
/// `hessian::uhf_state_refusal` refuses it on (its stationarity check is a
/// second, independent backstop).
///
/// Unchanged: a fractional (ensemble) density, which has no integer occupied
/// space (`d_occ` is `None`); and every run with MOM configured (`mom`),
/// including one that exits before MOM engages (iter ≤ `mom_after_iter`),
/// which keeps F's eigenpairs. A MOM state is non-aufbau by design and its
/// consumers and Hessian refusal rely on the aufbau eigenpairs of F
/// (`uhf_degenerate_swap.rs`,
/// `mom_state_keeps_aufbau_eigenpairs_and_the_hessian_refuses_it`).
fn exit_mos_building_density(
    eig: (Vec<f64>, Array2<f64>),
    spin: ExitSpin<'_>,
    mom: bool,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    let Some(c_occ) = spin.d_occ else {
        return Ok(eig);
    };
    let nocc = c_occ.ncols();
    if mom || nocc == 0 || nocc >= spin.x.ncols() {
        return Ok(eig);
    }
    let occ = eig.1.slice(ndarray::s![.., ..nocc]);
    let miss = (occ.dot(&occ.t()) - spin.d)
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    if miss <= EXIT_MO_DENSITY_GATE {
        return Ok(eig);
    }
    let (eps, c) = semicanonical_orbitals(spin.f, spin.s, spin.x, c_occ)?;
    let homo = eps[..nocc]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let lumo = eps[nocc..].iter().copied().fold(f64::INFINITY, f64::min);
    Ok(if homo > lumo { (eps, c) } else { eig })
}

/// The semicanonical orbitals of the space spanned by `c_occ` (n × nocc,
/// `0 < nocc < m`, S-orthonormal: the occupied block of an earlier
/// `diagonalize` in the same X basis): F diagonalised separately inside that
/// space and inside its complement within span(X), with no shift and no
/// mixing between the two.
///
/// Occupied: `C_occ U_o` with `U_o` the eigenvectors of `C_occᵀ F C_occ`, so
/// the occupied block rebuilds `C_occ C_occᵀ` up to rounding. Virtual: with
/// `W = Xᵀ S C_occ` (orthonormal columns in the m-dimensional X basis), the
/// complement is the eigenvalue-1 eigenspace of the projector `I − W Wᵀ`
/// (its other eigenvalue is 0, a gap of 1); F is diagonalised in it. Returned
/// occupied block first, each block ascending, padded to n × n like
/// `driver::diagonalize_rect` (zero columns, sentinel eigenvalue 1e6).
fn semicanonical_orbitals(
    f: &Array2<f64>,
    s: &Array2<f64>,
    x: &Array2<f64>,
    c_occ: &Array2<f64>,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    use ndarray_linalg::{Eigh, UPLO};
    let lapack = |e: ndarray_linalg::error::LinalgError| {
        FerricError::Lapack(format!("exit semicanonical orbitals: {e}"))
    };
    let (n, m, nocc) = (x.nrows(), x.ncols(), c_occ.ncols());
    let (e_o, u_o) = c_occ
        .t()
        .dot(f)
        .dot(c_occ)
        .eigh(UPLO::Upper)
        .map_err(lapack)?;
    let w = x.t().dot(s).dot(c_occ);
    let (p_vals, p_vecs) = (Array2::<f64>::eye(m) - w.dot(&w.t()))
        .eigh(UPLO::Upper)
        .map_err(lapack)?;
    // C_occ lies in span(X) and is S-orthonormal, so the projector's spectrum
    // is exactly nocc zeros and m − nocc ones, up to rounding.
    debug_assert!(
        p_vals.iter().take(nocc).all(|v| v.abs() < 1e-8)
            && p_vals.iter().skip(nocc).all(|v| (v - 1.0).abs() < 1e-8),
        "C_occ is not an S-orthonormal set in span(X): projector spectrum {p_vals:?}"
    );
    let xq = x.dot(&p_vecs.slice(ndarray::s![.., nocc..]));
    let (e_v, u_v) = xq.t().dot(f).dot(&xq).eigh(UPLO::Upper).map_err(lapack)?;
    let mut c = Array2::<f64>::zeros((n, n));
    c.slice_mut(ndarray::s![.., ..nocc])
        .assign(&c_occ.dot(&u_o));
    c.slice_mut(ndarray::s![.., nocc..m]).assign(&xq.dot(&u_v));
    let mut eps: Vec<f64> = e_o.iter().chain(e_v.iter()).copied().collect();
    eps.resize(n, 1e6);
    Ok((eps, c))
}

fn density(c: &Array2<f64>, nocc: usize) -> Array2<f64> {
    let n = c.nrows();
    if nocc == 0 {
        return Array2::zeros((n, n));
    }
    let c_occ = c.slice(ndarray::s![.., ..nocc]);
    c_occ.dot(&c_occ.t())
}

/// One spin's MOM step: reorder the new eigenpairs so the first `nocc` columns
/// overlap most with `mom_ref` (the previous occupied block). A no-op without a
/// reference or with nothing occupied.
///
/// The eigenvalues are permuted WITH the columns. `density_fractional` reads
/// `eps[p]` as the energy of column `p`; before 2026-10-24 only C was
/// reordered, so after a MOM swap the frontier was grouped by the energies of
/// other orbitals and the ensemble could spread an electron into a virtual.
fn mom_pin(
    c: Array2<f64>,
    eps: Vec<f64>,
    s: &Array2<f64>,
    mom_ref: Option<&Array2<f64>>,
    nocc: usize,
) -> (Array2<f64>, Vec<f64>) {
    match mom_ref {
        Some(r) if nocc > 0 => {
            let empty_open = Array2::<f64>::zeros((c.nrows(), 0));
            crate::mom::mom_reorder(&c, &eps, s, r, &empty_open, nocc, 0)
        }
        _ => (c, eps),
    }
}

/// The degenerate frontier group that [`density_fractional`] spreads: the
/// columns in `members` (ascending) each carry occupation `frac`, the other
/// occupied columns (`p < nocc`) carry 1 and the other virtuals 0.
struct FrontierGroup {
    members: Vec<usize>,
    frac: f64,
}

/// Find the fractional frontier group, or `None` when the occupation stays
/// integer. `eps[p]` must be the energy of column `p`.
///
/// Membership is by ENERGY, not by index: the HOMO is the highest occupied
/// energy, the LUMO the lowest virtual one, and the group is every occupied
/// column within `EPS_TOL` of the HOMO plus every virtual within `EPS_TOL` of
/// the LUMO. For ε-sorted columns this is exactly the contiguous walk down
/// from `nocc − 1` and up from `nocc` that this function used before
/// 2026-10-24. It also holds after a MOM step, whose occupied block is ordered
/// by overlap score (measured on Br/aug-cc-pVDZ: the 4s at −0.73 Ha lands in
/// column `nocc − 3` or `nocc − 2` of the α block), where the index walk
/// would anchor on whichever occupied orbital landed last.
fn fractional_frontier_group(eps: &[f64], nocc: usize, n: usize) -> Option<FrontierGroup> {
    if nocc == 0 || nocc >= n {
        return None;
    }
    // Tolerance must be loose enough to capture a frontier shell that the
    // *oscillating* SCF has artificially split. For a free ³P atom the three p
    // orbitals can be ~0.01–0.02 Ha apart mid-oscillation, so a tight 1e-3 tol
    // catches only 2 of 3 and the fix fails. 0.05 Ha reliably groups them; this
    // path is opt-in (free atoms only), so a loose tol cannot affect molecules.
    const EPS_TOL: f64 = 0.05;
    let e_homo = eps[..nocc]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let e_lumo = eps[nocc..n].iter().copied().fold(f64::INFINITY, f64::min);
    // Only act if HOMO and LUMO are within tol (genuine straddling degeneracy).
    if (e_lumo - e_homo).abs() >= EPS_TOL {
        return None;
    }
    // The degenerate frontier shell straddles the occupation boundary: grow
    // it around the HOMO on the occupied side and around the LUMO (which may
    // differ from the HOMO by the artificial split) on the virtual side.
    let members: Vec<usize> = (0..n)
        .filter(|&p| {
            let anchor = if p < nocc { e_homo } else { e_lumo };
            (eps[p] - anchor).abs() < EPS_TOL
        })
        .collect();
    // Electrons to distribute over the group = (occupied orbitals in group).
    let n_in_group_occupied = members.iter().filter(|&&p| p < nocc).count();
    let frac = n_in_group_occupied as f64 / members.len() as f64;
    Some(FrontierGroup { members, frac })
}

/// The per-column occupation vector `f` (length `n`) that [`density_fractional`]
/// uses, so `D = Σ_p f_p c_p c_pᵀ`; `None` when the occupation stays integer.
/// The gradient's energy-weighted density `W = Σ_p f_p ε_p c_p c_pᵀ` is built
/// from this same vector (`gradient::build_energy_weighted_density_uhf`).
pub(crate) fn fractional_occupations(eps: &[f64], nocc: usize, n: usize) -> Option<Vec<f64>> {
    let g = fractional_frontier_group(eps, nocc, n)?;
    let mut f: Vec<f64> = (0..n).map(|p| if p < nocc { 1.0 } else { 0.0 }).collect();
    for &p in &g.members {
        f[p] = g.frac;
    }
    Some(f)
}

/// Density with fixed fractional (ensemble) occupation of a degenerate frontier
/// shell, for one spin channel: `D = Σ_p f_p c_p c_pᵀ`.
///
/// Builds integer occupation `f = 1` for orbitals fully below the frontier, then
/// detects the group of orbitals near-degenerate with the HOMO (energies within
/// `EPS_TOL` of the highest occupied energy, see [`fractional_frontier_group`])
/// that straddle the occupation boundary, and spreads
/// the remaining electrons of that shell *equally* across the whole group. For a
/// ³P atom this puts 2/3 of an electron in each of the three degenerate p
/// orbitals, restoring spherical symmetry so the GGA potential stops oscillating.
///
/// Returns `None` when the occupation stays integer (HOMO non-degenerate — the
/// common case — or nothing to spread); the caller then uses plain `density()`,
/// so it is a no-op for ordinary molecules.
fn density_fractional(c: &Array2<f64>, eps: &[f64], nocc: usize) -> Option<Array2<f64>> {
    let n = c.ncols();
    debug_assert_eq!(c.nrows(), n, "density_fractional: C must be square");
    let FrontierGroup { members, frac } = fractional_frontier_group(eps, nocc, n)?;
    let core: Vec<usize> = (0..nocc).filter(|p| !members.contains(p)).collect();
    let mut d = Array2::<f64>::zeros((n, n));
    // Fully-occupied orbitals outside the degenerate group: f = 1. For
    // ε-sorted columns they are the leading block, contracted as a slice
    // exactly as before; after a MOM reorder they are gathered.
    if core.iter().enumerate().all(|(k, &p)| k == p) {
        if !core.is_empty() {
            let c_core = c.slice(ndarray::s![.., ..core.len()]);
            d = c_core.dot(&c_core.t());
        }
    } else {
        let c_core = c.select(ndarray::Axis(1), &core);
        d = c_core.dot(&c_core.t());
    }
    // Degenerate group: f = frac each.
    for &p in &members {
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
    Some(d)
}

/// Whether `fractional_occ` keeps the second-order (Newton / TRAH) steps off.
///
/// Both steps rotate the orbitals and rebuild the integer density
/// `C_occ·C_occᵀ`, then skip the DIIS branch, which is the only place
/// `density_fractional` runs. Under `fractional_occ` they therefore dropped
/// the ensemble occupation: before 2026-10-24, Br/def2-SVP/O3LYP with
/// `newton_trigger = 0.1` converged to the integer state, 16 mHa from the
/// fractional DIIS answer. So they stay off, and the SCF uses DIIS, the same
/// way the analytic Hessian and the KS z-vector refuse fractional occupation.
/// An explicit request is reported once per solve instead of skipped silently.
fn second_order_blocked_by_fractional_occ(config: &UhfConfig) -> bool {
    if !config.fractional_occ {
        return false;
    }
    let requested: Vec<&str> = [
        (config.newton_trigger > 0.0, "newton_trigger"),
        (config.trah_trigger.is_some(), "trah_trigger"),
    ]
    .into_iter()
    .filter_map(|(on, name)| on.then_some(name))
    .collect();
    if !requested.is_empty() {
        eprintln!(
            "[warning] solve_uhf: {} ignored: fractional occupation. Newton/TRAH steps rebuild \
             an integer-occupied density, so this SCF uses DIIS only",
            requested.join(" and ")
        );
    }
    true
}

/// One spin's density after an SCF step, paired with the occupied MO block
/// that drives the DF-K half-transform (`KBuilder::build_from_occ`, which
/// returns K for `C_occ·C_occᵀ`).
///
/// The cache is `Some(C_occ)` exactly when `D = C_occ·C_occᵀ`. A genuinely
/// fractional ensemble density (`D = Σ_p f_p c_p c_pᵀ`, 0 < f < 1 on the
/// frontier group) is not that product, so it returns `None` and K is built
/// from `D` itself — the same convention the RHF Fermi-smearing path uses
/// (`d_occ = None` in `rhf.rs`). Before 2026-10-24 the cache was always the
/// integer C_occ, so under `fractional_occ` + DF-K, K came from the integer
/// occupation while J, E and V_xc came from the fractional D. When
/// `fractional_occ` finds nothing to spread, `D` and the cache are the
/// integer ones, bit-identical to `fractional_occ = false`.
fn spin_density_and_occ(
    c: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    fractional_occ: bool,
) -> (Array2<f64>, Option<Array2<f64>>) {
    if fractional_occ {
        if let Some(d) = density_fractional(c, eps, nocc) {
            return (d, None);
        }
    }
    let occ = (nocc > 0).then(|| c.slice(ndarray::s![.., ..nocc]).to_owned());
    (density(c, nocc), occ)
}

/// Diagonalize F in the canonical-orthogonal basis X (n × m), padding the
/// result back to (n × n) with sentinel-energy zero columns for any dropped
/// near-singular modes. Mirrors `crate::rhf::diagonalize`. See that function for
/// the shape/padding rationale.
fn diagonalize(f: &Array2<f64>, x: &Array2<f64>) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    crate::driver::diagonalize_rect(f, x)
}

/// Build the open-shell initial MOs from the CONFIGURED density guess.
///
/// Returns `Ok(None)` — meaning "use hcore" — when the caller asked for the
/// bare hcore guess (`use_sad_guess = false` with no explicit density), and
/// also whenever the configured guess cannot be built. A guess that fails is
/// never fatal: hcore is what this path did unconditionally until 2026-09-16,
/// so falling back to it can only reproduce the old behavior, never break a
/// system that used to work.
///
/// # How a density becomes MOs
///
/// The SCF is MO-driven, so a guess DENSITY has to be turned into occupied
/// orbitals. That is done the only way it can be: build the Fock AT the guess
/// density and diagonalize it. The guess density is spin-summed (both `SAD`
/// and the MINAO projection return `D_total`), so it is split evenly between
/// the spins — `D_α = D_β = D/2` — and the resulting `F_α = F_β` are
/// diagonalized to give the same starting MOs for both spins. The α/β symmetry
/// is then broken exactly as before by the occupation itself (`nocc_α >
/// nocc_β`) or, for a forced-UHF closed shell, by the HOMO/LUMO mixing a few
/// lines below the call site — that code is untouched.
///
/// # Why the guess Fock is built with plain Coulomb J/K even under RSH/DFT
///
/// This is a GUESS. `build_jk` uses `bounds.op`, which is the operator the
/// whole SCF was set up with, and adds no XC. Under a range-separated or DFT
/// reference the guess Fock is therefore not the converged Fock — which is
/// fine and is what every SAD-style guess in every code does — but it means
/// this function must never be mistaken for a converged-Fock builder.
fn uhf_guess_mos(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &UhfConfig,
    h: &Array2<f64>,
    diags: (
        &crate::symmetric::FockDiag<'_>,
        &crate::symmetric::FockDiag<'_>,
    ),
) -> Result<Option<(Array2<f64>, Array2<f64>)>, FerricError> {
    let n = prep.nbasis();

    // Which density? An explicit one wins; otherwise the MINAO projection,
    // which is exactly what `rhf.rs` resolves for the same two config fields.
    let d_total = if let Some(d0) = config.init_guess_density.as_ref() {
        if d0.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "UHF: init_guess_density shape {:?} != ({n},{n})",
                d0.dim()
            )));
        }
        d0.clone()
    } else if config.use_sad_guess {
        match crate::guess::minao_projection_guess(mol, prep, prep.basis_set()) {
            Ok(d) => d,
            Err(e) => {
                if crate::rhf::scf_trace() {
                    eprintln!("UHF guess: MINAO projection failed ({e:?}); falling back to hcore");
                }
                return Ok(None);
            }
        }
    } else {
        return Ok(None);
    };

    // Split the spin-summed guess density evenly and build F_α = F_β at it.
    let d_spin = &d_total * 0.5;
    let mut j = Array2::<f64>::zeros((n, n));
    let mut k = Array2::<f64>::zeros((n, n));
    if let Err(e) = crate::rhf::build_jk(
        ctx,
        prep,
        bounds,
        config.integral_thresh,
        &d_total,
        &mut j,
        &mut k,
    ) {
        if crate::rhf::scf_trace() {
            eprintln!("UHF guess: J build at the guess density failed ({e:?}); using hcore");
        }
        return Ok(None);
    }
    let mut k_spin = Array2::<f64>::zeros((n, n));
    if let Err(e) = crate::rhf::build_jk(
        ctx,
        prep,
        bounds,
        config.integral_thresh,
        &d_spin,
        &mut j.clone(),
        &mut k_spin,
    ) {
        if crate::rhf::scf_trace() {
            eprintln!("UHF guess: K build at the guess density failed ({e:?}); using hcore");
        }
        return Ok(None);
    }

    // The UHF Fock convention, matching this file's own assembly below:
    //   F_σ = h + J[D_α + D_β] − K[D_σ].
    let f_guess = h + &j - &k_spin;
    match (diags.0.run(&f_guess), diags.1.run(&f_guess)) {
        (Ok((_, ca)), Ok((_, cb))) => Ok(Some((ca, cb))),
        (Err(e), _) | (_, Err(e)) => {
            if crate::rhf::scf_trace() {
                eprintln!("UHF guess: diagonalizing the guess Fock failed ({e:?}); using hcore");
            }
            Ok(None)
        }
    }
}

/// Injected-path counterpart of [`uhf_guess_mos`] for an explicit
/// spin-summed guess density: `D_α = D_β = D/2`, guess Fock
/// `h + J[D] − K[D/2]` from the INJECTED builders (so any Madelung term is
/// included consistently), diagonalized for the same starting MOs on both
/// spins. Unlike the molecular helper, a failing build is an error, not a
/// silent fallback: on the injected path the builders ARE the integrals.
fn injected_guess_mos(
    j: &mut dyn JBuilder,
    k: &mut dyn KBuilder,
    d_total: &Array2<f64>,
    h: &Array2<f64>,
    x: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let n = h.nrows();
    if d_total.dim() != (n, n) {
        return Err(FerricError::General(format!(
            "solve_uhf_injected: init_guess_density shape {:?} != ({n},{n})",
            d_total.dim()
        )));
    }
    let d_spin = d_total * 0.5;
    let mut jm = Array2::<f64>::zeros((n, n));
    j.build(d_total, &mut jm)?;
    let mut k_spin = Array2::<f64>::zeros((n, n));
    k.update_density(&d_spin);
    k.build(&d_spin, &mut k_spin)?;
    let f_guess = &(h + &jm) - &k_spin;
    let (_, c) = diagonalize(&f_guess, x)?;
    Ok((c.clone(), c))
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
    use ferric_core::basis;
    use ferric_core::external_potential::{ExternalPotential, PointCharge};
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::oneelectron;

    fn signals(de: f64, dp: f64) -> crate::rhf::ConvergenceSignals {
        crate::rhf::ConvergenceSignals {
            de,
            dp_rms: dp,
            dp_max: dp,
        }
    }

    /// The UHF veto wiring, independent of SCF arithmetic: a ΔP/ΔE-settled
    /// state with a large orbital gradient is refused, the refusal recorded,
    /// and the DIIS history cleared (the next extrapolation sees one entry and
    /// returns its input unchanged). End to end the stagnation is
    /// arithmetic-sensitive (it is reached on some OpenBLAS kernels and not
    /// others), so the wiring is pinned here.
    #[test]
    fn exit_gate_vetoes_a_settled_nonstationary_state_and_clears_diis() {
        let n = 2;
        let x = Array2::<f64>::eye(n);
        let f = |v: f64| Array2::from_elem((n, n), v);
        let mut diis = Diis::new(6);
        diis.step_pair(&f(1.0), &f(1.0), &f(0.3), &f(0.1));
        diis.step_pair(&f(2.0), &f(2.0), &f(0.2), &f(0.05));
        let err = ndarray::array![[0.0, 0.136], [-0.136, 0.0]];
        let config = UhfConfig {
            density_conv: 1e-6,
            ..Default::default()
        };
        let mut gate = UhfExitGate::default();
        let mut report = None;
        let (exit, swapped) = gate.decide(
            &mut None,
            UhfIterGradients {
                iter: 12,
                sig: signals(2.4e-12, 3.4e-8),
                err_max: 0.136,
                err: [&err, &err],
                x: &x,
                xc_active: [true, false],
            },
            &config,
            &mut diis,
            false,
            &mut report,
        );
        assert!(
            exit.is_none() && !swapped,
            "a non-stationary state was accepted"
        );
        let rec = gate.record().expect("the veto must be recorded");
        assert_eq!((rec.count, rec.last_iteration), (1, 12));
        let (fa, _) = diis.step_pair(&f(5.0), &f(5.0), &f(0.4), &f(0.4));
        assert_eq!(fa, f(5.0), "DIIS history was not cleared by the veto");
    }

    /// A degenerate swap under `fractional_occ` whose density is integer (no
    /// degenerate frontier to spread) must still reorder the exit MOs, per
    /// spin: before, `decide` refused any reorder under `fractional_occ`, so
    /// the returned aufbau C occupied the other member of the swapped pair
    /// and did not build the returned integer D. A fractional spin keeps its
    /// aufbau eigenpairs (no integer occupied set exists).
    #[test]
    fn swap_under_fractional_occ_reorders_each_integer_spin() {
        let n = 3;
        let x = Array2::<f64>::eye(n);
        let err = Array2::<f64>::zeros((n, n));
        let config = UhfConfig {
            density_conv: 1e-6,
            fractional_occ: true,
            ..Default::default()
        };
        let mut gate = UhfExitGate::default();
        let mut diis = Diis::new(6);
        let mut report = None;
        let mut step = |iter: usize| {
            gate.decide(
                &mut None,
                UhfIterGradients {
                    iter,
                    sig: signals(1e-9, 1.0),
                    err_max: 1e-9,
                    err: [&err, &err],
                    x: &x,
                    xc_active: [true, false],
                },
                &config,
                &mut diis,
                false,
                &mut report,
            )
        };
        let _ = step(5);
        let (exit, swapped) = step(6);
        assert!(exit.is_some(), "the degenerate swap was not accepted");
        assert!(swapped, "a swap under fractional_occ was not flagged");

        // Aufbau C = I with eps ascending; D occupies orbital 1, the other
        // member of a swapped pair, as an integer density.
        let c = Array2::<f64>::eye(n);
        let eps = vec![-0.5, -0.4, 0.3];
        let mut d = Array2::<f64>::zeros((n, n));
        d[(1, 1)] = 1.0;
        let d_occ = Some(c.slice(ndarray::s![.., 1..2]).to_owned());
        let (_, c_int) = occupied_by_density(
            (eps.clone(), c.clone()),
            &x,
            &d,
            1,
            reorder_exit_mos(swapped, d_occ.as_ref()),
        );
        let occ = c_int.slice(ndarray::s![.., ..1]);
        assert_eq!(
            occ.dot(&occ.t()),
            d,
            "integer-density exit MOs do not build D"
        );
        // A fractional spin (no cache) keeps the aufbau eigenpairs.
        assert!(!reorder_exit_mos(swapped, None));
        let (eps_f, c_f) = occupied_by_density((eps.clone(), c.clone()), &x, &d, 1, false);
        assert_eq!((eps_f, c_f), (eps, c));
        // No swap (a plain or MOM exit): never reordered.
        assert!(!reorder_exit_mos(false, d_occ.as_ref()));
    }

    /// `exit_mos_building_density`: the gate and the construction, on a
    /// synthetic F with a non-identity S (square X) and with a rectangular X
    /// (a dropped near-singular mode, padded like `diagonalize_rect`).
    ///
    /// - eigenpairs missing D by less than the prefilter come back
    ///   bit-identical, as do MOM and fractional (no cache) exits;
    /// - above it, a D that is not the aufbau state of F[D] is replaced: the
    ///   occupied block rebuilds D to rounding, C stays S-orthonormal (the
    ///   blocks never mix), F is diagonal inside each block, each block
    ///   ascends, and the result is non-aufbau;
    /// - a moderate rotation of the aufbau space (eigenvalues still ordered)
    ///   keeps the eigenpairs.
    #[test]
    fn exit_mos_building_density_gates_and_semicanonicalises() {
        let n = 5;
        let f = Array2::from_shape_fn((n, n), |(p, q)| {
            let (a, b) = (p.min(q) as f64, p.max(q) as f64);
            if p == q {
                -1.0 + 0.4 * a
            } else {
                0.05 * (a + 1.0) / (b + 2.0)
            }
        });
        let s_full = Array2::from_shape_fn((n, n), |(p, q)| {
            if p == q {
                1.0
            } else {
                0.2 / (1.0 + (p as f64 - q as f64).abs())
            }
        });
        let x_full = crate::rhf::canonical_orthogonalizer(&s_full).unwrap();
        assert_eq!(x_full.ncols(), n);
        let s_id = Array2::<f64>::eye(n);
        let x_rect = s_id.slice(ndarray::s![.., ..n - 1]).to_owned();
        let max_abs = |a: &Array2<f64>| a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (s, x) in [(&s_full, &x_full), (&s_id, &x_rect)] {
            let m = x.ncols();
            let eig = diagonalize(&f, x).unwrap();
            // A non-aufbau integer D: orbitals 0 and 2 occupied, 1 empty.
            let c_occ = ndarray::concatenate(
                ndarray::Axis(1),
                &[
                    eig.1.slice(ndarray::s![.., 0..1]),
                    eig.1.slice(ndarray::s![.., 2..3]),
                ],
            )
            .unwrap();
            let d = c_occ.dot(&c_occ.t());
            let spin = |d_occ| ExitSpin {
                f: &f,
                s,
                x,
                d: &d,
                d_occ,
            };

            // MOM and fractional exits are never touched.
            assert_eq!(
                exit_mos_building_density(eig.clone(), spin(Some(&c_occ)), true).unwrap(),
                eig
            );
            assert_eq!(
                exit_mos_building_density(eig.clone(), spin(None), false).unwrap(),
                eig
            );

            let (eps, c) =
                exit_mos_building_density(eig.clone(), spin(Some(&c_occ)), false).unwrap();
            let occ = c.slice(ndarray::s![.., ..2]);
            assert!(
                max_abs(&(occ.dot(&occ.t()) - &d)) < 1e-14,
                "occupied MOs miss D"
            );
            let ck = c.slice(ndarray::s![.., ..m]);
            let orth = max_abs(&(ck.t().dot(s).dot(&ck) - Array2::<f64>::eye(m)));
            assert!(orth < 1e-13, "C^T S C - I = {orth:.2e}");
            let g = ck.t().dot(&f).dot(&ck);
            for p in 0..m {
                assert!(
                    (g[(p, p)] - eps[p]).abs() < 1e-13,
                    "eps[{p}] is not diag(C^T F C)"
                );
                for q in 0..m {
                    if p != q && (p < 2) == (q < 2) {
                        assert!(
                            g[(p, q)].abs() < 1e-13,
                            "F not diagonal in a block ({p},{q})"
                        );
                    }
                }
            }
            assert!(eps[0] <= eps[1] && eps[2..m].windows(2).all(|w| w[0] <= w[1]));
            assert!(
                eps[1] > eps[2],
                "the occupied orbital above a virtual one is lost"
            );
            // Padding matches `diagonalize_rect`.
            assert!(eps[m..].iter().all(|&e| e == 1e6));
            assert!(c.slice(ndarray::s![.., m..]).iter().all(|&v| v == 0.0));

            // The prefilter: a miss below 1e-3 returns `eig` bit-identically
            // without looking further, even for a non-aufbau `d_occ`.
            let aufbau = eig.1.slice(ndarray::s![.., ..2]).to_owned();
            let mut d_small = aufbau.dot(&aufbau.t());
            d_small[(0, 0)] += 5e-4;
            let small = ExitSpin {
                f: &f,
                s,
                x,
                d: &d_small,
                d_occ: Some(&c_occ),
            };
            assert_eq!(
                exit_mos_building_density(eig.clone(), small, false).unwrap(),
                eig
            );
            // The decision: occupied space = {φ0, cos θ φ1 + sin θ φ2}. Its
            // semicanonical eigenvalues cross iff θ > π/4. A moderate
            // rotation of the aufbau space (θ = 0.3, a loosely converged
            // aufbau state: miss ~0.3, above the prefilter) keeps `eig`; past
            // π/4 (θ = 1.2) D is not the aufbau state of F[D] and is replaced.
            for (theta, replaced) in [(0.3f64, false), (1.2, true)] {
                let mut rot = aufbau.clone();
                let mixed = &eig.1.column(1) * theta.cos() + &eig.1.column(2) * theta.sin();
                rot.column_mut(1).assign(&mixed);
                let d_rot = rot.dot(&rot.t());
                let spin_rot = ExitSpin {
                    f: &f,
                    s,
                    x,
                    d: &d_rot,
                    d_occ: Some(&rot),
                };
                let out = exit_mos_building_density(eig.clone(), spin_rot, false).unwrap();
                assert_eq!(out != eig, replaced, "decision wrong at theta {theta}");
                if replaced {
                    let occ = out.1.slice(ndarray::s![.., ..2]);
                    assert!(max_abs(&(occ.dot(&occ.t()) - &d_rot)) < 1e-14);
                }
            }
        }
    }

    /// The swap gradient bound is `max(density_conv, 1e-6)`: at a tight
    /// `density_conv` a swap at the measured arithmetic floor (~1e-9) is
    /// accepted; a still-descending gradient, or an energy hop, is not.
    #[test]
    fn swap_acceptance_bound_is_floored_above_the_arithmetic_floor() {
        let swap = signals(1.4e-14, 0.986);
        // OH/STO-3G floor on the CI (Prescott) kernels, dconv 1e-9.
        assert!(degenerate_swap_converged(
            swap, 1.12e-9, 1.12e-9, 153, 1e-7, 1e-9
        ));
        // dconv 1e-4: the bound is dconv (OH: iterations 5 and 6).
        assert!(degenerate_swap_converged(
            swap, 8.9e-7, 6.8e-5, 6, 1e-7, 1e-4
        ));
        // One end above the floor: still descending.
        assert!(!degenerate_swap_converged(
            swap, 1.9e-8, 4.6e-6, 151, 1e-7, 1e-9
        ));
        // Energy moved: not a degenerate swap.
        assert!(!degenerate_swap_converged(
            signals(2e-5, 0.986),
            1e-9,
            1e-9,
            9,
            1e-3,
            1e-9
        ));
        // Density did not move by an orbital: the plain gate's business.
        assert!(!degenerate_swap_converged(
            signals(1e-14, 1e-3),
            1e-9,
            1e-9,
            9,
            1e-3,
            1e-9
        ));
    }

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
        assert!(
            (s2 - 2.0).abs() < 0.05,
            "O atom ⟨S²⟩ = {} (expected ≈2.0)",
            s2
        );
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
        assert_eq!(
            mol.nelec(),
            26,
            "def2 Xe ECP should remove 28 core electrons"
        );
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
        // solve (proatom closure in ferric-cli/src/lib.rs).
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
        assert!(
            res.converged,
            "UKS-PBE Br atom did not converge with fractional occ"
        );
        // Sanity: ⟨S²⟩ for a clean doublet (S=1/2) = 0.75; allow mild contamination.
        let s2 = expectation_s_squared(
            &res.mos_alpha,
            res.mos_beta.as_ref().unwrap(),
            &oneelectron::overlap(&prep),
            18,
            17,
        );
        assert!(
            (s2 - 0.75).abs() < 0.1,
            "Br atom ⟨S²⟩ = {} (expected ≈0.75)",
            s2
        );
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
            point_charges: vec![PointCharge {
                q: 1.0,
                x: 0.0,
                y: 0.0,
                z: 20.0,
            }],
            smeared_charges: Vec::new(),
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
        assert!(
            res.converged,
            "UKS-PBE O atom did not converge with fractional occ"
        );
    }

    /// Max-abs deviation from idempotency, `max|D·S·D − D|`: zero for an
    /// integer-occupied spin density, nonzero once the ensemble path spreads.
    fn idempotency_defect(d: &Array2<f64>, s: &Array2<f64>) -> f64 {
        (&d.dot(s).dot(d) - d)
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    #[test]
    fn fractional_occ_dfk_is_bit_identical_when_occupations_stay_integer() {
        // Exactness anchor for the fractional-occupation DF-K cache: H2O+ has a
        // non-degenerate frontier in both spins, so `fractional_occ` finds
        // nothing to spread and must reproduce `fractional_occ = false` — the
        // same density, the same C_occ half-transform, the same bits. This
        // guards against a NEEDLESS dense fallback (or any other change to the
        // integer path); it cannot see the bug itself, which needs a genuinely
        // fractional density (see fractional_occ_dfk_matches_exact_k).
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
        let run = |fractional_occ: bool| {
            let cfg = UhfConfig {
                max_iter: 100,
                df_k_aux: Some("def2-universal-jkfit".to_string()),
                fractional_occ,
                ..Default::default()
            };
            solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap()
        };
        let (plain, frac) = (run(false), run(true));
        assert!(plain.converged && frac.converged);
        let s = oneelectron::overlap(&prep);
        assert!(
            idempotency_defect(frac.density_beta.as_ref().unwrap(), &s) < 1e-8,
            "fixture must stay integer-occupied for this anchor"
        );
        assert_eq!(frac.iterations, plain.iterations);
        assert_eq!(
            frac.energy.to_bits(),
            plain.energy.to_bits(),
            "integer-occupation fractional_occ run {} != plain {}",
            frac.energy,
            plain.energy
        );
    }

    #[test]
    fn fractional_occ_dfk_matches_exact_k() {
        // Free Br atom (²P), UKS-O3LYP (11.6 % exact exchange) with
        // `fractional_occ`: the β 4p hole is spread 1/3 over px/py/pz, so D_β is
        // NOT C_occ·C_occᵀ. (With ≥20 % exchange, PBE0/B3LYP/HF, the exchange
        // splitting of the p shell exceeds the 0.05 Ha grouping tolerance and
        // the occupations stay integer — hence this functional.) DF-K must then
        // build K_β from D_β itself; before the fix it used the integer C_occ
        // half-transform, so K described a different density from J, E and
        // V_xc. Reference: the same calculation with exact (direct) K; J is
        // exact in both arms, so the only difference left is the K fit.
        // Measured 2026-10-24: fixed DF-K −2574.3315855 vs exact-K
        // −2574.3315936 Ha (8.0e-6, the fit). The old cache broke the p-shell
        // symmetry through K, the ensemble collapsed to integer occupation, and
        // DF-K converged to a different state at −2574.3475260 Ha (1.59e-2 Ha
        // away); both asserts below fail on that behaviour.
        let mol = Molecule::parse_xyz("1\nBr\nBr 0 0 0\n", 0, 2).unwrap();
        let bs = basis::bundled("def2-svp").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = ferric_integrals::operator::Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let s = oneelectron::overlap(&prep);
        let run = |df_k_aux: Option<&str>| {
            let cfg = UhfConfig {
                xc: Some("HYB_GGA_XC_O3LYP".to_string()),
                fractional_occ: true,
                max_iter: 200,
                df_k_aux: df_k_aux.map(str::to_string),
                ..Default::default()
            };
            solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap()
        };
        let (exact, df) = (run(None), run(Some("def2-universal-jkfit")));
        assert!(exact.converged && df.converged, "Br/O3LYP did not converge");
        for (name, r) in [("exact-K", &exact), ("DF-K", &df)] {
            let defect = idempotency_defect(r.density_beta.as_ref().unwrap(), &s);
            assert!(
                defect > 1e-2,
                "{name}: β density is idempotent ({defect:.1e}) — fractional path not taken"
            );
        }
        let de = (df.energy - exact.energy).abs();
        eprintln!(
            "Br/def2-SVP O3LYP fractional: DF-K {:.10} exact-K {:.10} |dE| {de:.3e}",
            df.energy, exact.energy
        );
        assert!(
            de < 1e-4,
            "DF-K {} vs exact-K {} differ by {de:.2e} (> DF-K fitting error)",
            df.energy,
            exact.energy
        );
    }

    #[test]
    fn mom_scrambled_occupied_block_keeps_the_fractional_group() {
        // A Br-like β channel: n = 5, S = I, C = I, ε-sorted
        // [-0.700 (s), -0.276, -0.275, -0.274 (p shell), 0.030], nocc = 3, so
        // the ensemble is s full and 2/3 on each p. MOM orders the occupied
        // block by overlap SCORE: with the reference below the scores are
        // u1 = u2 = 1 and u0 = 0.81, so the block becomes [u1, u2, u0] and the
        // s orbital sits in the HOMO slot `nocc − 1`.
        //
        // Measured failure modes this pins:
        // * ε left in diagonalisation order (pre-fix): the group is columns
        //   1..=3 = [u2, u0, u3], so the s orbital got 2/3 and u1 a full 1.
        // * ε permuted but the group found by the old index walk anchored on
        //   eps[nocc − 1] = −0.700: no group, the ensemble collapses to integer.
        let n = 5;
        let s = Array2::<f64>::eye(n);
        let c = Array2::<f64>::eye(n);
        let eps = vec![-0.700, -0.276, -0.275, -0.274, 0.030];
        let mut mom_ref = Array2::<f64>::zeros((n, 3));
        mom_ref[(1, 0)] = 1.0;
        mom_ref[(2, 1)] = 1.0;
        mom_ref[(0, 2)] = 0.9;
        mom_ref[(4, 2)] = (1.0f64 - 0.81).sqrt();
        let (c_m, eps_m) = mom_pin(c, eps, &s, Some(&mom_ref), 3);
        let (d, occ) = spin_density_and_occ(&c_m, &eps_m, 3, true);
        let f = 2.0 / 3.0;
        let expected = Array2::from_diag(&ndarray::arr1(&[1.0, f, f, f, 0.0]));
        let err = (&d - &expected).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(err < 1e-15, "wrong ensemble after the MOM reorder: {d:?}");
        assert!(occ.is_none(), "fractional density must not cache C_occ");
        assert_eq!(eps_m, vec![-0.276, -0.275, -0.700, -0.274, 0.030]);
    }

    #[test]
    fn mom_swap_groups_the_fractional_frontier_by_the_moved_energies() {
        // Four orthonormal orbitals (S = I, C = I), ε-sorted by the
        // diagonaliser: [-1.00, -0.30, -0.28, -0.20], nocc = 2. MOM keeps the
        // previously occupied set {u0, u3}: u3 is pinned although it rose
        // above u1, u2. After the swap the column order is [u0, u3, u1, u2] and
        // their energies are [-1.00, -0.20, -0.30, -0.28]: the HOMO (u3,
        // -0.20) and LUMO (u1, -0.30) are 0.10 Ha apart, outside the 0.05 Ha
        // grouping tolerance, so the occupation stays integer, D = u0 u0ᵀ +
        // u3 u3ᵀ.
        //
        // With the eigenvalues left in diagonalisation order (the pre-fix
        // code), column 1 (u3) was given -0.30 and column 2 (u1) -0.28, so the
        // ensemble spread the pinned electron half onto u1, a virtual.
        let n = 4;
        let s = Array2::<f64>::eye(n);
        let c = Array2::<f64>::eye(n);
        let eps = vec![-1.00, -0.30, -0.28, -0.20];
        let mut mom_ref = Array2::<f64>::zeros((n, 2));
        mom_ref[(0, 0)] = 1.0;
        mom_ref[(3, 1)] = 1.0;
        let (c_m, eps_m) = mom_pin(c, eps, &s, Some(&mom_ref), 2);
        let (d, occ) = spin_density_and_occ(&c_m, &eps_m, 2, true);
        let mut expected = Array2::<f64>::zeros((n, n));
        expected[(0, 0)] = 1.0;
        expected[(3, 3)] = 1.0;
        assert_eq!(d, expected, "MOM-pinned occupation was spread: {d:?}");
        assert!(occ.is_some(), "integer density must keep its C_occ cache");
        assert_eq!(eps_m, vec![-1.00, -0.20, -0.30, -0.28]);
    }

    #[test]
    fn spin_density_and_occ_caches_c_occ_only_for_integer_densities() {
        // Synthetic orthonormal-column C (identity, 6 AOs), eps with a 3-fold
        // degenerate group straddling nocc=3 (orbitals 2..=4), and a
        // non-degenerate control.
        let n = 6;
        let c = Array2::<f64>::eye(n);
        let degenerate = [-1.0, -0.5, -0.2, -0.2, -0.2, 0.5];
        let gapped = [-1.0, -0.5, -0.2, 0.3, 0.4, 0.5];
        let c_ct = |o: &Array2<f64>| o.dot(&o.t());
        // Degenerate + fractional_occ: D spreads 1/3 over the group, no cache.
        let (d, occ) = spin_density_and_occ(&c, &degenerate, 3, true);
        assert!(occ.is_none(), "fractional D must not carry a C_occ cache");
        for (p, want) in [1.0, 1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]
            .iter()
            .enumerate()
        {
            assert!((d[(p, p)] - want).abs() < 1e-15, "f_{p} = {}", d[(p, p)]);
        }
        // Every integer case caches exactly the C_occ with C_occ·C_occᵀ == D.
        for (eps, frac) in [(&degenerate, false), (&gapped, true), (&gapped, false)] {
            for nocc in [0, 3, n] {
                let (d, occ) = spin_density_and_occ(&c, eps, nocc, frac);
                match occ {
                    Some(o) => assert_eq!(c_ct(&o), d, "C_occ·C_occᵀ != D (nocc {nocc})"),
                    None => {
                        assert_eq!(nocc, 0, "integer D lost its cache (nocc {nocc})");
                        assert!(d.iter().all(|v| *v == 0.0));
                    }
                }
            }
        }
    }

    #[test]
    fn fractional_occ_keeps_newton_and_trah_off() {
        // Newton and TRAH rebuild an integer density after every step, so with
        // them armed the Br/O3LYP ensemble used to collapse onto the integer
        // state (−2574.3475339 Ha exact-K at newton_trigger = 0.1, 16 mHa from
        // the fractional DIIS answer). With fractional_occ they must stay off
        // and the run must land on the fractional DIIS state.
        let mol = Molecule::parse_xyz("1\nBr\nBr 0 0 0\n", 0, 2).unwrap();
        let bs = basis::bundled("def2-svp").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = ferric_integrals::operator::Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let s = oneelectron::overlap(&prep);
        let run = |newton_trigger: f64, trah_trigger: Option<f64>| {
            let cfg = UhfConfig {
                xc: Some("HYB_GGA_XC_O3LYP".to_string()),
                fractional_occ: true,
                max_iter: 200,
                df_k_aux: Some("def2-universal-jkfit".to_string()),
                newton_trigger,
                trah_trigger,
                ..Default::default()
            };
            solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap()
        };
        let diis = run(0.0, None);
        assert!(diis.converged);
        for (name, newton_trigger, trah_trigger) in
            [("newton", 0.1, None), ("trah", 0.0, Some(0.1))]
        {
            let r = run(newton_trigger, trah_trigger);
            assert!(r.converged, "{name}: not converged");
            let defect = idempotency_defect(r.density_beta.as_ref().unwrap(), &s);
            let de = (r.energy - diis.energy).abs();
            eprintln!(
                "{name}: E {:.10} vs DIIS {:.10} |dE| {de:.3e} defect {defect:.2e}",
                r.energy, diis.energy
            );
            assert!(
                defect > 1e-2,
                "{name}: β density is idempotent ({defect:.1e}) — fractional occupation dropped"
            );
            assert!(
                de < 1e-4,
                "{name}: E {} vs fractional DIIS {} ({de:.2e})",
                r.energy,
                diis.energy
            );
        }
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
