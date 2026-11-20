# Exits, the ladder and opt-in solvers {#scf-exits}

<!-- status: edit; registry: rhf, uhf, rohf -->

Every SCF loop ends in one of five ways, and only one of them is a converged
result. This section gives the stopping test, the exit reasons, the two
vetoes that refuse an exit, and the convergence ladder that reruns a failed
SCF with stronger settings. It closes with two opt-in solvers for closed
shells that replace parts of the loop: AURORA, a quasi-Newton step in place
of DIIS, and the symmetry-blocked solve, which diagonalizes the Fock matrix
one irrep at a time.

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/rhf.rs::{scf_converged,GradientVeto,veto_stalled_energy_diis,stall_detected}`, `ferric-scf/src/result.rs::ScfExit`, `ferric-scf/src/ladder.rs`, `ferric-scf/src/aurora.rs`, `ferric-scf/src/symmetric.rs`, `ferric-scf/src/symmetric_open.rs` |
| CLI | `[scf] density_conv`, `energy_conv`, `max_iter`, `[[scf.ladder]]`, `aurora`, `symmetry`, `irrep_nelec` |
| Python | `run_rhf`, `run_uhf`, `run_rohf`, `run_dft`: `density_conv`, `energy_conv`, `max_iter`, `aurora=`, `symmetry=`, `irrep_nelec=` |

## The stopping test {#scf-exits-stopping-test}

The exit test is `ferric-scf/src/rhf.rs::scf_converged`, shown whole:

```rust
let dp_rms_ok = sig.dp_rms < density_conv;          // Delta D_rms < density_conv
let dp_max_ok = sig.dp_max < 10.0 * density_conv;   // Delta D_max < 10 density_conv
let energy_not_descending = sig.de < energy_conv;   // |Delta E| < energy_conv
if dp_rms_ok && dp_max_ok && energy_not_descending {
    Some(crate::result::ScfExit::Converged)
} else {
    None
}
```

| Test | Default |
|---|---|
| $\Delta D_{\rm rms}$ between successive densities | `density_conv` $= 10^{-6}$ |
| $\Delta D_{\max}$ | $10 \times$ `density_conv` |
| $\lvert\Delta E\rvert$ | `energy_conv` $= 10^{-3}$, a sanity bound, not the target |

The test runs from iteration 2 on, and the exit diagonalizes the
unextrapolated, unshifted $F$.

Why is the density the gate and not the commutator? With density fitting
the commutator stops falling at a floor set by the fit, while the density
change keeps draining to zero at the fitted fixed point. The $\Delta D_{\max}$
companion catches a single density element that is still moving while the
root-mean-square change is small.

## Exit reasons {#scf-exits-reasons}

`ferric-scf/src/result.rs::ScfExit` records why the loop ended.
`ScfResult::converged` is true only for `Converged`.

| Exit | Condition | Produced by |
|---|---|---|
| `Converged` | the stopping test passes and no veto refuses the exit | every solver |
| `Stalled` | `stall_window` $= n$ is set, at least $2n$ iterations have run, $\max\lvert e\rvert \ge 10^{-4}$, and the smallest $\max\lvert e\rvert$ of the last $n$ iterations is not below $0.9\times$ the smallest of the $n$ before (`stall_detected`) | RHF and RKS; off by default |
| `Diverged` | `divergence_tol` $= f$ is set and the energy rose by more than $f$ Ha on 3 consecutive iterations | RHF and RKS; off by default |
| `MaxIter` | `max_iter` iterations without any of the above | every solver |
| `NotCertified` | ROHF and ROKS: the SCF converged, but the swap witness found a one-electron move to a lower state and the restart budget ran out (`ferric-scf/src/rohf_occupation.rs`) | ROHF, ROKS |

`Stalled` compares running minima, not single values, so a limit cycle with
a wide band trips it while a slow but steady descent does not. Below the
$10^{-4}$ floor it never fires. Both early exits exist for the ladder below,
which sets `stall_window` $= 15$ and `divergence_tol` $= 0.5$ Ha on every
rung, so a stuck rung hands over after about 15 iterations instead of
running to `max_iter`. Below the floor the commutator can park on a fitting
or near-degeneracy floor while the density settles, and the stopping test
decides.

## Vetoes {#scf-exits-vetoes}

The density test alone can accept a point that is not stationary. A
stagnated extrapolation can return the same density on every step without
that density being stationary: the doc comment of `GradientVeto` records OH
UKS-LDA/STO-3G frozen at a non-aufbau state 0.586 Ha above the stationary
ground state. Three checks run after the test has passed.

| Veto | Refuses an exit when | Then |
|---|---|---|
| Orbital gradient (`GradientVeto`, RHF, RKS, UHF, UKS) | $\max\lvert X^{\mathsf T}eX\rvert > \max(\sqrt{\texttt{density\_conv}}, 10^{-3})$ | the DIIS history is cleared and the loop continues from a fresh diagonalization |
| Energy-DIIS stall (`veto_stalled_energy_diis`, RHF and RKS under `diis = "adiis"` or `"ediis"`) | $\max\lvert e\rvert \ge$ `diis_switch_thresh` ($10^{-2}$ when the switch is infinite) | the driver drops the energy model for the rest of the run and restarts on Pulay from a clean history, once |
| Mixed precision (`jk_ladder::veto_mixed_exit`, RHF and RKS) | the iteration that passed ran a mixed-precision $J$/$K$ kernel | the build is latched to f64 for the rest of the SCF and the iteration is redone at the same density, with a clean DIIS history |

The orbital-gradient veto uses $X^{\mathsf T}eX$, the error in the
orthonormal basis $X$ ($X^{\mathsf T}SX = 1$), which has the same Frobenius
norm as $C^{\mathsf T}eC$ and the same $2f_{ai}$ block (claim N6 of
`scripts/scf_diis_sympy.py`). At the default `density_conv` the bound is
$\max(10^{-3}, 10^{-3}) = 10^{-3}$. The gradient is evaluated only once the
test has passed, so an iteration that does not exit pays nothing. ROHF has
its own guard, the swap witness behind `NotCertified`.

The energy-DIIS veto exists because the energy model of section
@sec:diis-variants can settle the density far from a stationary point: the
simplex minimum stops moving while the commutator is still large (issue
#422). With the exact EDIIS model none of 72 EDIIS runs and 24 ADIIS runs
of water, CH3Br and benzene in point-charge fields stalls; the veto is kept
as a guard and tested directly (unit test on `veto_stalled_energy_diis`). Pulay
alone has no veto bound, so the default path is untouched. The
doc comment of `DiisDriver::commutator_veto_bound` records the price at a
user switch at or below the density-fitting commutator floor (about
$10^{-6}$): the veto can fire on a converged run and cost 3 extra Pulay
iterations, measured at $10^{-6}$ and $10^{-8}$, with the energy unchanged.

## The convergence ladder {#scf-exits-ladder}

`ferric-scf/src/ladder.rs::solve_rhf_ladder` runs a list of rungs, each a
full `RhfConfig`. It carries each rung's final density into the next as the
starting guess (unless the rung sets `restart`), stops at the first rung
that converges, and otherwise returns the lowest-energy result with
`converged = false`. Each rung's `max_iter` is its own budget, not a share
of a total. `solve_rohf_ladder`, the counterpart for ROHF and ROKS, has no
caller in the CLI or Python.

The built-in ladder (`default_ladder_from`) escalates convergence settings
only. It inherits the caller's $J$/$K$ route and every other field:

| Rung | DIIS | Level shift (Ha) | Newton trigger | Smearing $\sigma$ (Ha) | `max_iter` |
|---|---|---|---|---|---|
| 0 | Pulay | 0 | off | none | 60 |
| 1 | ADIIS, then Pulay | 0 | off | none | 60 |
| 2 | ADIIS, then Pulay | 0.5 | off | none | 60 |
| 3 | ADIIS, then Pulay | 1.0 | $10^{-3}$ | none | 80 |
| 4 | Pulay | 0.5 | $10^{-3}$ | 0.01 | 100 |

Every rung sets `stall_window` $= 15$ and `divergence_tol` $= 0.5$ Ha. The
level shift is section @sec:level-shift-mom-smearing, the Newton tail
section @sec:second-order, and smearing the last resort for a
near-degenerate frontier.

`ksdft_ladder` has the same shape for Kohn–Sham. Rung 0 keeps the caller's
`level_shift` and `max_iter`; rungs 2 to 4 use at least 0.5 Ha (rung 3 adds
0.5 to that). With a functional set and no auxiliary basis given it fits $J$
and $K$ with def2-universal-jkfit, except under COSX, where it fits $J$ only.

Who runs the ladder: the CLI kinds `rhf` and `ksdft` (an explicit
`[[scf.ladder]]` list, or the built-in one for the run's kind), the
closed-shell Kohn–Sham reference of the double hybrids, and Python `run_dft`
and the closed-shell TDDFT reference. Python `run_rhf` and the other CLI
kinds call `solve_rhf` once.
The ladder anchor is the N2 row of `site/src/reference/validation.md`: on
N2 at 1.60 Å / def2-SVP rung 0 lands on a saddle 1.906e-2 Ha above the
minimum, which the stability descent of section @sec:stability then reaches.

## AURORA {#scf-exits-aurora}

AURORA (`ferric-scf/src/aurora.rs`, `aurora = true`) replaces the DIIS
extrapolation with a quasi-Newton orbital rotation, after Bao and Shi,
arXiv:2608.07354. The energy, the gradient and the stopping test come from
the target Hamiltonian; the curvature comes from a cheaper model, the full
Coulomb and exchange orbital response with STO-3G as the fitting basis,
corrected on the fly by an L-BFGS history of target-level secant pairs. The
step is the exact orbital exponential along the geodesic.

What it is and is not:

- **Scope.** Closed-shell RHF and hybrid RKS. The CLI admits it for kind
  `rhf` with multiplicity 1 only; Python takes `aurora=`. Open-shell
  references are not implemented.
- **No exchange–correlation curvature.** The paper names a Kohn–Sham
  diagonal correction and never defines it, so the model carries none. With
  little exact exchange the model understates the curvature and the step
  overshoots, so such functionals stay on DIIS unless
  `allow_low_exchange_ks` is set.
- **No trust region.** The step is clipped to a fixed length; there is no
  trial rejection or radius reduction.
- **Off by default.** With it off the SCF is bit-identical to a build
  without the module (`aurora_off_bit_identity.rs`).

The tests check the construction, not the speed. `aurora_math_anchors.rs`
checks the geodesic against the matrix exponential, the transport of the
secant pairs, and the auxiliary response against finite differences.
`aurora_same_fixed_point.rs` checks that AURORA and DIIS converge to the
same energy (bar $10^{-9}$ Ha), as an accelerator must, and
`aurora_step_control.rs` and `zz_aurora_hybrid.rs` cover the step clip and
the hybrid-RKS gate. The cost benchmark, `aurora_benchmark.rs`, is ignored
by default and no measured result is recorded with it. AURORA is unanchored:
nothing in the repository shows that it converges in fewer iterations or
less time than DIIS on any system.

## The symmetry-blocked solve {#scf-exits-symmetry}

`ferric-scf/src/symmetric.rs::solve_rhf_symmetric` (`symmetry = true`)
detects the largest Abelian subgroup of the molecule's point group, moves the
molecule into that group's frame, symmetrizes it, and diagonalizes the Fock
matrix one irrep at a time. With $U_h$ the symmetry-adapted AO combinations
of irrep $h$,

$$ F_h = U_h^{\mathsf T}FU_h, \qquad X_h = \text{canonical orthogonalizer of } U_h^{\mathsf T}SU_h, \qquad (U_hX_h)^{\mathsf T}F(U_hX_h) = V_h\varepsilon_hV_h^{\mathsf T}, \qquad C_h = U_hX_hV_h . $$

Each block has its own linear-dependence filter (section
@sec:orthogonalization-and-guesses). The eigenpairs of all blocks are merged
into one list sorted by orbital energy, so aufbau across irreps, the density,
DIIS and the energy are the unblocked code. Energies within $10^{-10}$ Ha are
ordered by irrep, totally symmetric first, so an exact degeneracy between
irreps resolves the same way every run.

`irrep_nelec` fixes the electron count of the listed irreps (each even for
RHF; the lowest orbitals of the irrep are filled whatever their energy), and
the unlisted irreps share the rest by aufbau. That is how an excited
configuration of a different symmetry is targeted.

The result is a `SymmetricRhf`, not a plain `ScfResult`, because its AO
matrices belong to the symmetrized, reoriented molecule. About sixty public
functions take an `ScfResult` with the caller's molecule; a frame-rotated
result paired with the input molecule gives a silent wrong answer (RI-MP2 on
a blocked NH3 gave $-7.89$ Ha of correlation instead of $-0.189$). The
symmetrized geometry differs from the input by at most the detection
tolerance, $10^{-4}$ Bohr. A C1 molecule runs the unblocked solver on the
original input.

| Reference | Entry point | Reached from |
|---|---|---|
| RHF, RKS | `solve_rhf_symmetric` | CLI `[scf] symmetry` with kind `rhf` or `ksdft`, energy only, closed-shell singlet; Python `run_rhf(symmetry=True)`, `run_dft(symmetry=True)` |
| UHF, ROHF | `ferric-scf/src/symmetric_open.rs::{solve_uhf_symmetric,solve_rohf_symmetric}`, `irrep_nelec` as $(n_\alpha, n_\beta)$ per irrep | the library only |

Refused, with a typed error: MOM, smearing, fractional occupations, Newton,
AH, TRAH, AURORA, stability analysis and descent, cDFT constraints, a user
starting density (it would be in the input frame), COSX, external
potentials, COSMO, PCM, polarizable embedding and the periodic path. The
open-shell entry points also refuse Kohn–Sham and, for ROHF, the swap
witness. Blocked RKS also refuses a main grid below 110 angular points and a
partly filled degenerate shell split across irreps, because the Lebedev grid
is not invariant under C3 and C6 and would choose the occupation by grid
noise.

The anchors are the two point-group rows of
`site/src/reference/validation.md`: blocked RHF matches PySCF with symmetry
to $1.0\times10^{-11}$ Ha over 10 blocked and 2 `irrep_nelec` cases, and
blocked against unblocked in the same frame agrees to $1.1\times10^{-13}$
Ha; blocked RKS matches PySCF to $3.6\times10^{-13}$ Ha. On stretched H2,
where the unblocked SCF does not converge, the blocked solve reaches
$\sigma_g^2$.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/src/rhf.rs`, `result.rs`, `driver.rs` | the stopping test, exits and vetoes |
| `crates/ferric-scf/src/diis.rs` | the energy-DIIS veto bound |
| `crates/ferric-scf/src/ladder.rs` | the ladders |
| `crates/ferric-scf/src/aurora.rs` and `crates/ferric-scf/tests/aurora_*.rs` | AURORA and its tests |
| `crates/ferric-scf/src/symmetric.rs`, `symmetric_open.rs` | the blocked solve |
| `scripts/scf_diis_sympy.py` | claim N6 (the orthonormal-basis gradient) |
| `site/src/reference/validation.md` | the ladder and point-group anchors |
| X. Bao and Y. Shi, arXiv:2608.07354 (2026) | AURORA |
