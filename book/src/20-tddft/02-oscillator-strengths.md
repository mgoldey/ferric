# 20.2 Transition dipoles and oscillator strengths

<!-- section: 20.2; status: ready; registry: tddft;
     theory: crates/ferric-tddft/theory.md#"4. Transition dipoles and oscillator strengths" -->

The transition dipole of a root is a contraction of its amplitudes with the
occupied–virtual dipole integrals, and the oscillator strength follows in the
length gauge. ferric uses PySCF's convention, so the numbers compare
directly.

| Symbol | Meaning | Code name |
|---|---|---|
| $\langle i|\mathbf r|a\rangle$ | MO dipole integrals, origin at the coordinate origin | `mu_ov` |
| $\langle 0|\mathbf r|n\rangle$ | transition dipole of root $n$ | `tdip` |
| $f_n$ | oscillator strength | `osc` |

| Entry point | Location |
|---|---|
| Rust | `ferric-tddft/src/tddft.rs::{oscillator_strengths,transition_properties}` |
| Python | `run_tddft(...)` result: `excitation_energies`, `oscillator_strengths` |

## 20.2.1 The length-gauge formula

<!-- include: crates/ferric-tddft/theory.md#"4. Transition dipoles and oscillator strengths" -->

$$ \langle 0|\mathbf r|n\rangle = \sqrt2\sum_{ia}(X+Y)_{ia,n}\langle i|\mathbf r|a\rangle,\qquad
f_n = \tfrac23\,\Omega_n\,|\langle 0|\mathbf r|n\rangle|^2 , \tag{20.6} $$

with $Y = 0$ and $|X| = 1$ in TDA. The $\sqrt2$ is the spin sum: a singlet
has $X_{ia\alpha} = X_{ia\beta} = X_{ia}/\sqrt2$, and the spin-free dipole adds
both spins. The result does not depend on the origin, because
$\langle i|a\rangle = 0$.

`oscillator_strengths` forms $\langle i|\mathbf r|a\rangle = C_{\rm occ}^{\mathsf T}\boldsymbol\mu^{\rm AO}C_{\rm vir}$
once, contracts it with each amplitude column in the `i * nvir + a` order,
multiplies by `SQRT_2` and pushes $\tfrac23\Omega_n|\langle 0|\mathbf r|n\rangle|^2$.

Triplet roots are reported with zero transition dipole and oscillator
strength (`transition_properties`): $X_\alpha = -X_\beta$ cancels in the
spin-free dipole (section 20.5).

> **Verified equation 20.6** · the $\sqrt2$ and the $\Omega^{-1/2}$ are fixed by the brute-force moment and the sum rule
>
> For the TDA singlets of a random HF model, $\langle\Phi_0|\mathbf r|\Psi_n\rangle$
> evaluated over Jordan–Wigner determinants equals
> $\sqrt2\sum_{ia}X_{ia}\langle i|\mathbf r|a\rangle$. With all Casida
> singlet roots,
> $\sum_n f_n = \tfrac43\sum_x\mathbf r_x^{\mathsf T}(\mathbf A-\mathbf B)\mathbf r_x$,
> which follows from $\sum_n\Omega_n(X+Y)_n(X+Y)_n^{\mathsf T} = \mathbf A - \mathbf B$
> and so holds only with both factors right.
>
> | | |
> |---|---|
> | Proof | `scripts/tddft_response_sympy.py`, claims B2, sumrule |
> | Negative control | the runner drops the $\sqrt2$ from the transcribed `oscillator_strengths` (B2 fails) and scales $X+Y$ by $1/\Omega$ (B4 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-tddft/src/tddft.rs::oscillator_strengths` |
> | Code test | `crates/ferric-tddft/tests/validation_tddft.rs` (oscillator strengths against PySCF) |
>
> <!-- verified: id=td.osc.sum-rule; script=scripts/tddft_response_sympy.py; claim=sumrule; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-tddft/src/tddft.rs::oscillator_strengths; registry=tddft -->

## 20.2.2 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-tddft/theory.md`, section 4 | the formula and the sum rule |
| `crates/ferric-tddft/theory.md`, section 10 | the unrestricted form, without the $\sqrt2$ (section 20.6) |
