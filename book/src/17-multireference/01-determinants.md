# 17.1 Determinants, strings and the phase convention

<!-- section: 17.1; status: ready; registry: casci;
     theory: crates/ferric-ci/theory.md#"Determinants and the phase convention", "Hamiltonian" -->

A CAS-CI wavefunction is a linear combination of every determinant with
$n_\alpha$ and $n_\beta$ electrons in $n_a$ active orbitals. ferric stores a
determinant as two `u64` bit strings, one per spin, and the CI vector as the
matrix $C(I_\alpha, I_\beta)$ over α strings and β strings. A sign
convention is part of that representation: it fixes which matrix elements of
$H$ are positive, and a convention mixed up between two parts of the code
gives a wrong Hamiltonian with the right spectrum on symmetric test
molecules.

| Symbol | Meaning | Code name |
|---|---|---|
| $A$, $B$ | α and β occupation strings, bit $p$ = active orbital $p$ | `u64` |
| ${\rm pb}(s, p)$ | $(-1)^{\#\{q<p:\,q\in s\}}$ | `strings::phase_below` |
| $N_\text{det}$ | $\binom{n_a}{n_\alpha}\binom{n_a}{n_\beta}$ | `DeterminantSpace::n_det` |

| Entry point | Location |
|---|---|
| Rust | `ferric-ci/src/strings.rs::{enumerate_strings,phase_below,count_strings}` |
| CLI | used by `method.kind = "casscf"` and `"nevpt2"` |
| Python | `run_casci`, `run_casscf`, `run_nevpt2` |

## 17.1.1 The convention

<!-- include: crates/ferric-ci/theory.md#"Determinants and the phase convention" -->

## 17.1.2 The Hamiltonian

<!-- include: crates/ferric-ci/theory.md#"Hamiltonian" -->

$h$ already contains the potential of the doubly occupied core, and
$E_\text{core}$ its energy plus the nuclear repulsion; section 17.5 gives
both for a CASSCF orbital set.

> **Verified equation 17.1** · bit strings are determinants with sign $+1$
>
> On Jordan–Wigner matrices for eight spin orbitals: the annihilators obey
> the canonical anticommutation relations; the ascending product of
> creators applied to $|0\rangle$ is the computational basis vector of the
> bit string with coefficient $+1$; and $a_P$ on it carries
> $(-1)^{\#\{Q<P\}}$ over occupied modes. A single-channel phase is
> therefore the full Jordan–Wigner phase for α, and for β up to a factor
> $(-1)^{N_\alpha}$ that cancels in every term of $H$.
>
> | | |
> |---|---|
> | Proof | `scripts/casci_slater_condon_phases_sympy.py`, claims 1 and 2 |
> | Negative control | claim 4c: a β single excitation with the $(-1)^{N_\alpha}$ applied once |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/strings.rs::{phase_below,enumerate_strings}` |
> | Code test | `brute_force_reference_is_hermitian_and_number_conserving` in `crates/ferric-ci/tests/slater_condon_brute_force.rs` |
>
> <!-- verified: id=ci.strings.convention; script=scripts/casci_slater_condon_phases_sympy.py; claim=2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/strings.rs::{phase_below,enumerate_strings}; rust_test=ferric-ci/tests/slater_condon_brute_force.rs::brute_force_reference_is_hermitian_and_number_conserving; registry=casci -->

## 17.1.3 Strings in the code

`enumerate_strings(n, k)` lists all $k$-electron strings in ascending
numeric order (Gosper's hack). The order only indexes the coefficient
array, so any fixed order is correct. The `u64` holds at most
`MAX_ACTIVE_ORBITALS` = 63 active orbitals; CAS-CI and CASSCF refuse a
larger space before any integral work. The real limit is memory: the
Davidson subspace grows with $N_\text{det}$, which is combinatorial in
$n_a$, and the CAS-CI memory plan refuses a space that does not fit
(section 17.3).

**Known limits (2026-11-15).** One fixed convention, α before β. An
interleaved order ($p\alpha, p\beta$) would change $H_{IJ}$ by a diagonal
$\pm1$ similarity; ferric never uses it, and the proof script shows it fails
the element-wise check while keeping the spectrum.
