# Transition dipoles and oscillator strengths {#oscillator-strengths}

<!-- status: ready; registry: tddft;
     theory: crates/ferric-tddft/theory.md#"4. Transition dipoles and oscillator strengths" -->

An excitation energy says where a molecule absorbs. The oscillator strength
says how much. Light couples to the electrons through the dipole operator,
so a transition absorbs in proportion to the square of its transition
dipole $\langle 0|\mathbf r|n\rangle$, the dipole matrix element between the
ground and the excited state. A root whose transition density
$\phi_i\phi_a$ has no dipole moment is dark however low it lies. Response
theory gives the transition dipole as a contraction of the root's
amplitudes with the occupied–virtual dipole integrals. ferric uses PySCF's
convention, so the numbers compare directly.

| Symbol | Meaning | Code name |
|---|---|---|
| $\langle i|\mathbf r|a\rangle$ | MO dipole integrals, origin at the coordinate origin | `mu_ov` |
| $\langle 0|\mathbf r|n\rangle$ | transition dipole of root $n$ | `tdip` |
| $f_n$ | oscillator strength | `osc` |

| Entry point | Location |
|---|---|
| Rust | `ferric-tddft/src/tddft.rs::{oscillator_strengths,transition_properties}` |
| Python | `run_tddft(...)` result: `excitation_energies`, `oscillator_strengths` |

## The length-gauge formula {#oscillator-strengths-length-gauge-formula}

In the length gauge the transition dipole and the oscillator strength are

$$ \langle 0|\mathbf r|n\rangle = \sqrt2\sum_{ia}(X+Y)_{ia,n}\langle i|\mathbf r|a\rangle,\qquad
f_n = \tfrac23\,\Omega_n\,|\langle 0|\mathbf r|n\rangle|^2 , \label{eq:td-osc-sum-rule} $$

with $Y = 0$ and $|X| = 1$ in TDA.

Where do the two factors come from?

- **The $\sqrt2$ is the spin sum.** A singlet root moves both spins equally,
  $X_{ia\alpha} = X_{ia\beta} = X_{ia}/\sqrt2$, so that the spin-orbital
  vector has unit length. The dipole operator does not act on spin, so it
  adds the two spin contributions:
  $2\cdot X_{ia}/\sqrt2 = \sqrt2\,X_{ia}$.
- **$X+Y$, not $X$, enters for Casida.** The dipole is a real operator, so
  it couples to the excitation and the de-excitation parts with the same
  sign. The $\Omega_n^{-1/2}$ in equation @eq:td-casida-reduction fixes the scale of $X+Y$
  through $|X|^2 - |Y|^2 = 1$.

The result does not depend on where the dipole origin sits. Moving it by
$\mathbf R$ changes each integral by $\mathbf R\langle i|a\rangle$, and
$\langle i|a\rangle = 0$ for orthogonal occupied and virtual orbitals.

Connect this to the poles of section @sec:response-tda-casida-reduction. Take the perturbation to be
one Cartesian component of the dipole, $v_{ia} = \langle i|x|a\rangle$. The
numerator of pole $n$ in the sum-over-states expression is then
$2\Omega_n\big(v\cdot(X+Y)_n\big)^2 = \Omega_n|\langle 0|x|n\rangle|^2$, and
$f_n$ is $\tfrac23$ of that numerator summed over $x$, $y$ and $z$. The
oscillator strength measures how strongly the field drives root $n$.

Sum $f_n$ over all Casida roots. With $\mathbf r_x$ the vector of
$\langle i|x|a\rangle$ and equation @eq:td-casida-reduction,

$$ \sum_n\Omega_n(X+Y)_n(X+Y)_n^{\mathsf T} = T\Big(\sum_n Z_nZ_n^{\mathsf T}\Big)T = \mathbf A-\mathbf B , $$

because the $Z_n$ form an orthonormal basis. Substitute into the sum of
equation @eq:td-osc-sum-rule:

$$ \sum_n f_n = \tfrac23\sum_x\sum_n\Omega_n\cdot2\big(\mathbf r_x\cdot(X+Y)_n\big)^2
= \tfrac43\sum_x\mathbf r_x^{\mathsf T}(\mathbf A-\mathbf B)\,\mathbf r_x . $$

This holds only when both the $\sqrt2$ and the $\Omega_n^{-1/2}$ are right,
which makes it a test of both factors.

`ferric-tddft/src/tddft.rs::oscillator_strengths` forms
$\langle i|\mathbf r|a\rangle = C_{\rm occ}^{\mathsf T}\boldsymbol\mu^{\rm AO}C_{\rm vir}$
once, then contracts it with each amplitude column in the loop over
roots:

<!-- excerpt: ferric-tddft/src/tddft.rs::oscillator_strengths -->
```rust
for &n in cols {
    let omega_n = omegas[n];
    let mut tdip = [0.0_f64; 3];
    for xyz in 0..3 {
        let mut val = 0.0;
        for i in 0..nocc {
            for a in 0..nvir {
                val += amplitudes[(i * nvir + a, n)] * mu_ov[xyz][(i, a)];
            }
        }
        tdip[xyz] = std::f64::consts::SQRT_2 * val;
    }
    let mu_sq = tdip[0] * tdip[0] + tdip[1] * tdip[1] + tdip[2] * tdip[2];
    osc.push((2.0 / 3.0) * omega_n * mu_sq);
    tdips.push(tdip);
}
```

| Code | Equation @eq:td-osc-sum-rule |
|---|---|
| `amplitudes[(i * nvir + a, n)]` | $X_{ia,n}$ (TDA) or $(X+Y)_{ia,n}$ (Casida), in the compound order of section @sec:response |
| `mu_ov[xyz][(i, a)]` | $\langle i|\mathbf r|a\rangle$, origin `[0.0, 0.0, 0.0]` |
| `SQRT_2 * val` | the spin sum |
| `(2.0 / 3.0) * omega_n * mu_sq` | $f_n$, with $\Omega_n$ in Hartree |

The same loop serves TDA and Casida: the solvers hand it $X$ or $X+Y$, both
already normalized.

Triplet roots are reported with zero transition dipole and oscillator
strength (`ferric-tddft/src/tddft.rs::transition_properties`). A triplet has
$X_\alpha = -X_\beta$, and the spin-free dipole adds the two spins, so they
cancel (section @sec:triplets). On an unrestricted reference each spin is its own
block and the formula has no $\sqrt2$ (section @sec:unrestricted).

> **Verified equation @eq:td-osc-sum-rule** · the $\sqrt2$ and the $\Omega^{-1/2}$ are fixed by the brute-force moment and the sum rule
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

## Sources {#oscillator-strengths-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-tddft/theory.md`, section 4 | the formula and the sum rule |
| `crates/ferric-tddft/theory.md`, section 10 | the unrestricted form, without the $\sqrt2$ (section @sec:unrestricted) |
