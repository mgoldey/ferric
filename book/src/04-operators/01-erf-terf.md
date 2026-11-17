# 4.1 erf and erfc; terf and terfc; the curvature constraint

<!-- section: 4.1; status: edit; registry: none;
     theory: crates/ferric-mp2/theory.md#"4. Spin-component scaling", "5. Attenuated MP2" -->

The Coulomb kernel $1/r$ does two different jobs in a correlation
calculation. At short range it is large and singular: two electrons close
together repel strongly, and the wavefunction has a cusp where they meet.
At long range it decays slowly: two molecules several ångström apart still
feel each other through it, and that is where dispersion comes from. A
method can be good at one of these jobs and poor at the other. MP2 in a
finite basis is one example. Its long-range part overestimates dispersion,
and basis set superposition error adds more spurious binding. The
dissertation's abstract states the remedy that chapter 2 develops:
"attenuation of the long-range helps to cancel both BSSE and intrinsic MP2
errors".

An attenuated operator replaces the Coulomb kernel $1/r$ in a two-electron
integral by a short-range or a long-range part of it. ferric uses two
families. The error-function pair is the common one. The tempered
error-function pair, terf and terfc, adds a distance $r_0$ at which the
switch happens. Every attenuated method in the book (attenuated MP2,
SCS-MP2(2terfc), MP2-V, RS-MP2 + LR-RPA, range-separated hybrids) uses one of
these four kernels.

The plan for the section:

1. Define the two pairs and check their limits (4.1.1).
2. Expand the long-range kernels at $r = 0$ and find the one value of
   $\omega$ that makes terf flat there (4.1.2).
3. Derive two consequences for how the integrals can be used: the energy is
   not additive in the split, and terf is not positive definite (4.1.3).
4. List the parameters, their units and their sources (4.1.4 to 4.1.7).

| Symbol | Meaning | Code name |
|---|---|---|
| $r$ | electron-electron distance $r_{12}$ | |
| $\omega$ | sharpness of the switch, Bohr$^{-1}$ in the Rust API | `omega` |
| $r_0$ | terf/terfc switching distance, Bohr in the Rust API | `distance` (the `r0` argument) |
| $\hat w(r)$ | the kernel of an integral $(pq\vert\hat w\vert rs)$ | `Operator` |

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/operator.rs::{Operator,coulomb,erf,erfc,terf,terfc,terfc_with_omega,terf_with_omega}` |
| CLI | `[mp2] omega` (Å$^{-1}$), `att_operator`, `att_r0` and `att_omega` (Å, Å$^{-1}$), `r0`, `terf_omega`, `r0_bonded`, `r0_nonbonded`; the keys and units are in `site/src/reference/input.md` |
| Python | the `omega=`, `r0=` and `terf_omega=` keywords of the MP2-family `run_*` functions (`crates/ferric-python/ferric.pyi`) |

## 4.1.1 The two pairs

$\operatorname{erf}(\omega r)$ rises from 0 at $r = 0$ to 1 for
$r \gg 1/\omega$. Split the number 1
into $\operatorname{erf}$ and its complement, then divide by $r$:

$$ 1 = \operatorname{erf}(\omega r) + \operatorname{erfc}(\omega r), \qquad \frac1r = \frac{\operatorname{erfc}(\omega r)}{r} + \frac{\operatorname{erf}(\omega r)}{r} . \tag{4.1} $$

The tempered pair moves the rise out to $r_0$. Average two error functions,
one centred at $+r_0$ and one at $-r_0$:

$$ \operatorname{terf}(r; r_0, \omega) = \tfrac12\Big[\operatorname{erf}\big(\omega(r - r_0)\big) + \operatorname{erf}\big(\omega(r + r_0)\big)\Big], \qquad \operatorname{terfc} = 1 - \operatorname{terf} . \tag{4.2} $$

For $r \ge 0$ the second error function is already near 1 when $r_0 \gg
1/\omega$, so terf rises from 0 to 1 near $r = r_0$, not near the origin.
The mirror term at $-r_0$ makes terf an odd function of $r$, which 4.1.2
uses.

In each pair the "c" member is the short-range part: erfc and terfc go to 1
at $r = 0$, so the short-range kernel keeps the full Coulomb singularity, and
to 0 as $r \to \infty$. erf and terf carry the long range. The limits match
between the pairs: as $r_0 \to \infty$ ($\omega \to 0$ at the constraint
below) terf $\to 0$, and as $r_0 \to 0$ terf $\to$ erf with $\omega \to \infty$,
which is the full Coulomb operator.

Check those two limits by substitution. At $r_0 = 0$ the two terms of (4.2)
are equal and terf$(r; 0, \omega) = \operatorname{erf}(\omega r)$. With the
constraint $\omega = 1/(\sqrt2\,r_0)$ of 4.1.2, the arguments of (4.2) are
$(r \mp r_0)/(\sqrt2\,r_0)$. As $r_0 \to \infty$ at fixed $r$ they tend to
$\mp 1/\sqrt2$, the two error functions cancel, and terf $\to 0$: the whole
of $1/r$ is short range. As $r_0 \to 0$ both arguments go to $+\infty$ for
any $r > 0$, terf $\to 1$, and the long-range kernel is all of $1/r$. The
dissertation reports the same two ends for attenuated MP2 itself: it
"converges to the unattenuated result" as $r_0$ grows and "approaches HF
results" as $r_0 \to 0$ (chapter 3, figure captions).

Range separation of the correlation energy with the erf pair, and the terf
pair of Dutoi and Head-Gordon, are introduced in the dissertation's chapter 2;
chapter 4, Eqs. 4.1–4.2, restates terf and terfc in
the form (4.2).

![Left: $r\hat w(r)$ for the four kernels; each pair sums to 1. Right: the long-range kernels themselves, normalized at $r = 0$. Script: `book/src/figures/04-operators/kernels.py`.](../figures/04-operators/kernels.svg)

## 4.1.2 The curvature constraint

terf has two parameters, $r_0$ and $\omega$, and this subsection shows
what fixes the second. Look at the long-range kernel near $r = 0$. A
long-range kernel should be smooth there, so that it does not take over
any of the short-range physics. What matters is how many derivatives of
$1/r$ the short-range kernel keeps.

Start with erf. Expand $\operatorname{erf}(x) = (2/\sqrt\pi)(x - x^3/3 +
x^5/10 - \dots)$ and divide by $r$:

$$ \frac{\operatorname{erf}(\omega r)}{r} = \frac{2\omega}{\sqrt\pi} - \frac{2\omega^3}{3\sqrt\pi}\,r^2 + O(r^4) . $$

The $r^2$ coefficient is negative for every $\omega > 0$. So
$\operatorname{erfc}(\omega r)/r = 1/r - 2\omega/\sqrt\pi +
2\omega^3 r^2/(3\sqrt\pi) + \dots$ differs from $1/r$ in its curvature at
the origin, however $\omega$ is chosen.

Now terf. Write $g(r) = \operatorname{erf}\big(\omega(r + r_0)\big)$.
Because erf is odd, the first term of (4.2) is
$\operatorname{erf}\big(\omega(r - r_0)\big) = -g(-r)$. Substitute:

$$ \operatorname{terf}(r) = \tfrac12\big[g(r) - g(-r)\big], $$

the odd part of $g$. The Taylor series of an odd part keeps only the odd
powers, so divide by $r$ and read off the even powers of the kernel:

$$ \frac{\operatorname{terf}(r)}{r} = g'(0) + \frac{g'''(0)}{6}\,r^2 + O(r^4) . $$

Differentiate $g$. With $g'(r) = (2\omega/\sqrt\pi)\,e^{-\omega^2(r +
r_0)^2}$, two more derivatives of the Gaussian give

$$ g'(0) = \frac{2\omega}{\sqrt\pi}\,e^{-\omega^2 r_0^2}, \qquad g'''(0) = \frac{2\omega}{\sqrt\pi}\big(4\omega^4 r_0^2 - 2\omega^2\big)\,e^{-\omega^2 r_0^2} . $$

Divide $g'''(0)$ by 6 and collect $\omega^3$. The long-range kernel is even
and analytic at the origin:

$$ \frac{\operatorname{terf}(r; r_0, \omega)}{r} = \tau_1 + \tau_3 r^2 + O(r^4), \qquad \tau_3 = \frac{\omega^3}{3\sqrt\pi}\big(4\omega^2 r_0^2 - 2\big)\,e^{-\omega^2 r_0^2} . \tag{4.3} $$

with $\tau_1 = g'(0)$. Check the limit $r_0 = 0$: $\tau_3 \to
-2\omega^3/(3\sqrt\pi)$, the erf coefficient above.

$\tau_3$ has one positive root, $\omega = 1/(\sqrt2\,r_0)$. At that $\omega$
the long-range kernel is flat to fourth order at the origin, so the
short-range kernel $\operatorname{terfc}/r = 1/r - \tau_1 - O(r^4)$ has the
same first, second and third derivatives as $1/r$. The dissertation states
the constraint as $r_0\omega = 1/\sqrt2$ and the property as "the same
derivative as the Coulomb operator in the short-range" (chapter 2). The erf
kernel has no such freedom: $\operatorname{erf}(\omega r)/r$ has a nonzero
$r^2$ term for every $\omega > 0$, which the right panel of the figure shows.

What does this mean physically? Near the origin, terfc$/r$ is $1/r$ shifted
down by the constant $\tau_1$. A constant shift exerts no force, so the
slope of the interaction between two close electrons is that of $1/r$ up to
terms in $r^3$. erfc$/r$ adds an $r^2$ term as well, so its slope differs
from that of $1/r$ already at first order in $r$. With the constraint, terfc removes the long range
and leaves the short range as close to Coulomb as a smooth subtraction can.
The constraint also turns a two-parameter family into a one-parameter one,
so the fitted methods of 4.1.5 have a single length $r_0$.

> **Verified equation 4.3** · the curvature constraint
>
> The $r^2$ coefficient of $\operatorname{terf}(r)/r$ is $\tau_3$ of (4.3), its
> only positive root is $\omega = 1/(\sqrt2\,r_0)$, terf + terfc = 1, and
> terfc$(0) = 1$.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_family_sympy.py`, part D |
> | Negative control | the runner's mutation of the linked width $1/(r_0\sqrt2) \to 1/(2r_0)$ turns part D red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-integrals/src/operator.rs::terfc`, `ferric-integrals/src/operator.rs::terf` |
> | Code test | `terf_plus_terfc_reconstructs_coulomb` in `crates/ferric-integrals/tests/operator_positivity_gate.rs` (needs the untracked terf tables; skips without them) |
>
> <!-- verified: id=op.terf.curvature-constraint; script=scripts/mp2_family_sympy.py; claim=D; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-integrals/src/operator.rs::{terfc,terf}; rust_test=ferric-integrals/tests/operator_positivity_gate.rs::terf_plus_terfc_reconstructs_coulomb -->

The proof script finds $\tau_3$ with a sympy series and solves for its
roots. The intermediate steps above ($g'(0)$, $g'''(0)$ and the $r_0 = 0$
check) were checked separately with sympy.

`Operator::terfc` and `Operator::terf` apply the constraint, so $r_0$ is their
only argument:

<!-- excerpt: ferric-integrals/src/operator.rs::terfc -->
```rust
let omega = 1.0 / (r0 * std::f64::consts::SQRT_2);
Self::primitive(OperatorKind::Terfc, omega, r0)
```

Two points about these two lines:

- **Both parameters are stored.** The operator carries `omega` and
  `distance` (= $r_0$, in Bohr) side by side. The integral engine reads
  both, so it never has to know whether the constraint was applied.
- **`std::f64::consts::SQRT_2` keeps the constant exact to double
  precision.** The linked $\omega$ is $1/(r_0\sqrt2)$ to rounding, the
  value at which (4.3) has its root.

`Operator::terfc_with_omega` and `Operator::terf_with_omega` take $r_0$ and
$\omega$ independently. The sum rule terf + terfc = 1 holds for every
$(r_0, \omega)$, so either form splits $1/r$ exactly.

## 4.1.3 Two properties that decide how the integrals are used

**Splitting is not additive in the energy.** MP2 is quadratic in the
integrals. With $(ia|jb) = s + l$ for the short- and long-range parts,

$$ E[s + l] = E[s] + E[l] + \text{(cross term in } s\,l\text{)}, $$

so $E[\operatorname{erfc}] + E[\operatorname{erf}] \ne E[1/r]$. To see the
cross term, write the closed-shell energy (10.3) as
$E[g] = -\sum_{ijab} g_{iajb}(2g_{iajb} - g_{ibja})/\Delta_{ij}^{ab}$,
substitute $g = s + l$ and expand the product. The $s\,s$ terms give
$E[s]$, the $l\,l$ terms give $E[l]$, and the mixed terms remain:

$$ E[s+l] - E[s] - E[l] = -\sum_{ijab}\frac{4s_{iajb}l_{iajb} - s_{iajb}l_{ibja} - l_{iajb}s_{ibja}}{\Delta_{ij}^{ab}} . $$

The cross term is what RS-MP2 + LR-RPA is built on (chapter 26). It is checked by
claim 4 of `scripts/mp2_variant_identities.py`.

**terf is not positive definite.** A kernel $\hat w$ is positive definite
when $(\rho|\hat w|\rho) > 0$ for every nonzero charge distribution $\rho$.
In Fourier space $(\rho|\hat w|\rho) = \int |\tilde\rho(k)|^2\,\tilde w(k)\,
d\mathbf k$, so the kernel is positive definite when its transform is
positive (Bochner's theorem; section 4.2.1 derives the transforms). The
transforms of the kernels are

| Kernel | $\tilde w(k)\cdot 2\pi^2 k^2$ | Positive definite |
|---|---|---|
| erf | $e^{-k^2/4\omega^2}$ | yes |
| erfc | $1 - e^{-k^2/4\omega^2}$ | yes |
| terfc | $1 - \cos(k r_0)\,e^{-k^2/4\omega^2}$ | yes |
| terf | $\cos(k r_0)\,e^{-k^2/4\omega^2}$ | no: negative at $k r_0 = \pi$ |

The factor $\cos(kr_0)$ comes from the shell. terf switches on at a finite
distance $r_0$, and a charge distribution modulated at a wavevector with
$kr_0$ near $\pi$ (wavelength near $2r_0$) has negative interaction energy
through it. The erf
kernel is long range too, but its switch is centred on $r = 0$ and its
transform has no oscillating factor.

A kernel that is not positive definite has no Cauchy–Schwarz bound and can
give a fitting metric with negative eigenvalues. Two consequences follow in
the code. Schwarz screening is refused for terf. The fitting metric of the erf
and terf operators goes through the eigendecomposition branch of
`ferric-mp2/src/rimp2.rs::metric_inverse_sqrt`, which drops small and negative
eigenvalues instead of failing a Cholesky factorization (section 6.1). The
erf metric is positive definite in exact arithmetic; it takes the same branch
because it loses rank numerically in a Coulomb-fitted auxiliary basis. The
table in this section is the one pinned by
`crates/ferric-integrals/tests/operator_positivity_gate.rs`, whose test
`terf_is_indefinite_and_terfc_is_not` asserts a negative terf diagonal and a
non-negative terfc one on water / cc-pVDZ.

The dispatch is one `match` on the operator kind:

```rust
if matches!(op.kind, OperatorKind::ErfCoulomb | OperatorKind::Terf) {
    eigh_inverse_sqrt(v)       // drop small/negative eigenvalues of V
} else {
    cholesky_inverse_sqrt(v)   // V = L L^T, M = L^{-1}
}
```

This is the body of `metric_inverse_sqrt` with only the comments added.
terfc takes the Cholesky branch, as Coulomb and erfc do: its transform is
positive, and a Cholesky failure on it would point at a real defect in the
integrals.

## 4.1.4 Options and units

The Rust API takes $\omega$ in Bohr$^{-1}$ and $r_0$ in Bohr. The CLI and the
MP2-family Python entry points take Å$^{-1}$ and Å. `[dft] omega` is the
exception: it is in Bohr$^{-1}$ (chapter 0, section 8).

| Parameter | Surface | Default | Used by |
|---|---|---|---|
| $\omega$, erfc | `[mp2] omega`, `run_attenuated_rimp2(omega=)` | 0.420 Å$^{-1}$ (0.2223 Bohr$^{-1}$) | `att-rimp2` with erfc |
| $r_0$, terfc | `[mp2] att_r0` | 1.05 Å | `att-rimp2` with `att_operator = "terfc"` |
| free $\omega$, terfc | `[mp2] att_omega` | linked, $1/(\sqrt2 r_0)$ | same, `Operator::terfc_with_omega` |
| $r_0^{(1)}, r_0^{(2)}$ | `[mp2] r0_bonded`, `r0_nonbonded` | 0.75 Å, 1.05 Å | `scs-mp2-2terfc` |
| $r_0$, terf | `[mp2] r0` | 1.6828 Å (3.18 Bohr) | `rs-mp2-rpa` with terf |

The Bohr values in the table are the Å values times 0.529177 Bohr/Å for
$\omega$ and divided by it for $r_0$.

The terf and terfc integrals come from interpolation tables that are not in
the repository; the run needs `FERRIC_TERF_TABLE_DIR` (the input reference
says where).

## 4.1.5 Parameters from the dissertation

The attenuated MP2 methods were defined and fitted in the dissertation. The
values below are the method definitions, quoted with their source; the
benchmark statistics behind them are not reproduced.

| Method | Kernel | Basis | Parameter | Source |
|---|---|---|---|---|
| MP2(erfc, aDZ) | erfc | aug-cc-pVDZ | $\omega = 0.420$ Å$^{-1}$ | dissertation, chapter 2 |
| MP2(terfc, aDZ) | terfc | aug-cc-pVDZ | $r_0 = 1.05$ Å ($\omega = 0.673$ Å$^{-1}$ at the constraint) | dissertation, chapter 2 |
| MP2(terfc, aTZ) | terfc | aug-cc-pVTZ | $r_0 = 1.35$ Å | dissertation, chapter 3 |
| SCS-MP2(2terfc) | terfc at two cutoffs | aug-cc-pVTZ | $r_0^{(1)} = 0.75$ Å, $r_0^{(2)} = 1.05$ Å | dissertation, chapter 5 |

The ferric defaults in section 4.1.4 are these values. The 0.673 Å$^{-1}$
is $1/(\sqrt2 \times 1.05)$.

## 4.1.6 Numerical anchors

| Quantity | Value | Source |
|---|---|---|
| terf + terfc against the Coulomb quartet | machine precision | `crates/ferric-integrals/tests/eri4_terfc_split.rs` (`terf4_plus_terfc4_reconstructs_the_coulomb_quartet`) |
| terf + terfc with a free $\omega$ | machine precision | `crates/ferric-integrals/tests/terfc_base_validation.rs` (`free_omega_terf_plus_terfc_is_coulomb`) |
| erf + erfc against Coulomb | machine precision | `test_erf_plus_erfc_equals_coulomb` in `crates/ferric-integrals/src/engine.rs` |

## 4.1.7 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-mp2/theory.md`, "4. Spin-component scaling" | the terfc kernel, (4.3) and its root |
| `crates/ferric-mp2/theory.md`, "5. Attenuated MP2" | units, the range-split cross term |
| `scripts/mp2_family_sympy.py`, part D | the proof of (4.3) |
| `crates/ferric-integrals/tests/operator_positivity_gate.rs` | the Fourier-transform table and its measurement |
| Dissertation, chapters 2–5 | the terf pair, the constraint, the fitted parameters |

**Known limits (2026-11-15).** The terf/terfc tests skip when the
interpolation tables are absent, so a run without `FERRIC_TERF_TABLE_DIR`
checks only the erf pair. Yukawa, Slater-geminal and fitted-geminal kernels
are section 4.3.
