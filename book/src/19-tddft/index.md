# Time-dependent density functional theory {#tddft}

Shine light of frequency $\nu$ on a molecule and its electron density
oscillates at that frequency. For most $\nu$ the oscillation is small.
When $\hbar\nu$ matches the energy of an excited state, the linear
response grows without bound: the excitation energies are the poles of the density
response. Linear-response TDDFT finds those poles from the converged
ground-state orbitals, without ever writing down an excited-state
wavefunction.

The chapter builds the response problem for a converged SCF reference.
Section @sec:response derives the response matrices $\mathbf A$ and $\mathbf B$, the
Tamm–Dancoff approximation and the Casida equations, and the dense and
matrix-free solvers. Section @sec:oscillator-strengths turns the amplitudes into transition
dipoles and oscillator strengths. Section @sec:kernel covers the
exchange-correlation kernel and which functionals are accepted, section
@sec:rsh range-separated hybrids, and section @sec:triplets triplets. Section @sec:unrestricted
solves the spin-conserving problem of UHF and UKS references and gives
$\langle S^2\rangle$ for each root.

The kernel itself is derived in section @sec:fxc, and the matrix
$\mathbf A + \mathbf B$ is the orbital Hessian of sections @sec:second-order and @sec:stability.

| CLI kind | Python | Sections | Grade |
|---|---|---|---|
| `tda` | `run_tddft(method="tda")` | @sec:response to @sec:unrestricted | Proven (narrow) |
| `tddft` (Casida) | `run_tddft(method="casida")` | @sec:response to @sec:unrestricted | Proven (narrow) |

The grades and their envelopes are the `tda` and `tddft` rows of
`site/src/reference/validation.md`. A functional or spin outside the
validated set runs and is certified Smoke.

The $\omega$ of this chapter is the range-separation parameter of a
range-separated hybrid (section @sec:rsh); a driving frequency is $\nu$.

**Known limits (2026-11-19).**

- Restricted (RHF, RKS) and unrestricted (UHF, UKS) references only. ROHF
  and ROKS references are refused (section @sec:unrestricted-occupations-refusals).
- Spin-conserving excitations only. A closed shell gives singlets and the
  $M_S = 0$ triplet; $M_S = \pm1$ is not computed, and there is no
  spin-flip TDDFT. On an unrestricted reference a triplet request is
  refused.
- VV10 functionals are refused: the nonlocal kernel is not built. A
  meta-GGA with no libxc second derivative is refused.
- SCAN is not grid-converged at $10^{-4}$ eV on any grid tested; r2SCAN
  is (section @sec:kernel-grid-sensitivity).
- Excitation energies, oscillator strengths and $\langle S^2\rangle$ only.
  There are no excited-state gradients, no excited-state densities and no
  transition properties beyond the length-gauge dipole.
- The dense solver holds several $d\times d$ matrices, $d = n_on_v$, so its
  memory grows as $N^4$. The Davidson solver avoids them (section
  @sec:response-matrix-free-solvers); the RI tensors remain.
- The OH radical runs TDA only in the validation: the zero mode of its
  degenerate ground state leaves the sign of its lowest Casida $\Omega^2$
  undetermined.

## What is proven and what is not {#tddft-proven}

The chapter's tests and their reference data:

| Check | Where | Reference |
|---|---|---|
| TDA and Casida roots against PySCF `TDA`/`TDDFT` | `crates/ferric-tddft/tests/validation_tddft.rs` | generator `scripts/validation/gen_tddft_refs.py`; grades in `site/src/reference/validation.md`, rows `tda` and `tddft` |
| range-separated hybrids and triplets against PySCF `get_ab` | `crates/ferric-tddft/tests/validation_tddft_rsh.rs` | generator `scripts/validation/gen_tddft_rsh_refs.py`; RHF blocks for singlets, the UKS spin blocks for triplets; the response integrals, the SCF fit and the grid are matched to PySCF |
| unrestricted roots and $\langle S^2\rangle$ against PySCF `tdscf.uhf.get_ab` | `crates/ferric-tddft/tests/validation_tddft_open_shell.rs` | generator `scripts/validation/gen_tddft_open_shell_refs.py` |
| the exact limits of section @sec:rsh and the singlet + triplet identity of section @sec:triplets | unit tests in `crates/ferric-tddft/src` | exact identities |
| the Davidson solvers against the dense solve | unit tests in `crates/ferric-tddft/src/davidson.rs` | the dense roots |
| refusals | `crates/ferric-tddft/tests/refusals.rs` | |
| memory plan | `crates/ferric-tddft/tests/memory_budget.rs` | |

`scripts/tddft_response_sympy.py` proves, symbolically: the singlet and
triplet spin adaptation of $\mathbf A$ and $\mathbf B$, the kernel factor 2,
the scalar Casida reduction and its normalization, the range-separated
exchange split, Olsen's correction, the Casida subspace solution and the
triplet linearization of the libxc variables. Numerically, on random
models with RI tensors and a grid kernel, it checks the spin-orbital
$\mathbf A$ and $\mathbf B$ against the code's singlet and triplet blocks;
CIS by brute force over determinants (Jordan–Wigner) against $\mathbf A$,
closed and open shell; the Casida roots and $X+Y$ against the full
non-Hermitian problem; the oscillator-strength sum rule; the unrestricted
blocks and their closed-shell limit; the $\langle S^2\rangle$ formula
against a Fock-space $\lVert S_+\Psi\rVert^2$; and the Casida $X$
extraction against the full eigenvector. With `DERIV_CROSSCHECK=1` and
PySCF importable, it also compares H$_2$O/STO-3G RHF (TDA and TDHF,
singlet and triplet) and H$_2$O$^+$/STO-3G UHF (TDA and TDHF) energies and
oscillator strengths with PySCF; without that variable this part is
skipped with a message.
