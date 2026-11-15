# 4.1 erf and erfc; terf and terfc; the curvature constraint

<!-- section: 4.1; status: edit; registry: none;
     theory: crates/ferric-mp2/theory.md#"4. Spin-component scaling", "5. Attenuated MP2" -->

An attenuated operator replaces the Coulomb kernel $1/r$ in a two-electron
integral by a short-range or a long-range part of it. ferric uses two
families. The error-function pair is the common one. The tempered
error-function pair, terf and terfc, adds a distance $r_0$ at which the
switch happens. Every attenuated method in the book (attenuated MP2,
SCS-MP2(2terfc), MP2-V, RS-MP2 + LR-RPA, range-separated hybrids) uses one of
these four kernels.

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

$$ 1 = \operatorname{erf}(\omega r) + \operatorname{erfc}(\omega r), \qquad \frac1r = \frac{\operatorname{erfc}(\omega r)}{r} + \frac{\operatorname{erf}(\omega r)}{r} . \tag{4.1} $$

$$ \operatorname{terf}(r; r_0, \omega) = \tfrac12\Big[\operatorname{erf}\big(\omega(r - r_0)\big) + \operatorname{erf}\big(\omega(r + r_0)\big)\Big], \qquad \operatorname{terfc} = 1 - \operatorname{terf} . \tag{4.2} $$

In each pair the "c" member is the short-range part: erfc and terfc go to 1
at $r = 0$, so the short-range kernel keeps the full Coulomb singularity, and
to 0 as $r \to \infty$. erf and terf carry the long range. The limits match
between the pairs: as $r_0 \to \infty$ ($\omega \to 0$ at the constraint
below) terf $\to 0$, and as $r_0 \to 0$ terf $\to$ erf with $\omega \to \infty$,
which is the full Coulomb operator.

Range separation of the correlation energy with the erf pair, and the terf
pair of Dutoi and Head-Gordon, are introduced in the dissertation's chapter 2;
chapter 4, Eqs. 4.1–4.2, restates terf and terfc in
the form (4.2).

![Left: $r\hat w(r)$ for the four kernels; each pair sums to 1. Right: the long-range kernels themselves, normalized at $r = 0$. Script: `book/src/figures/04-operators/kernels.py`.](../figures/04-operators/kernels.svg)

## 4.1.2 The curvature constraint

terf is odd in $r$, so its long-range kernel is even and analytic at the
origin:

$$ \frac{\operatorname{terf}(r; r_0, \omega)}{r} = \tau_1 + \tau_3 r^2 + O(r^4), \qquad \tau_3 = \frac{\omega^3}{3\sqrt\pi}\big(4\omega^2 r_0^2 - 2\big)\,e^{-\omega^2 r_0^2} . \tag{4.3} $$

$\tau_3$ has one positive root, $\omega = 1/(\sqrt2\,r_0)$. At that $\omega$
the long-range kernel is flat to fourth order at the origin, so the
short-range kernel $\operatorname{terfc}/r = 1/r - \tau_1 - O(r^4)$ has the
same first, second and third derivatives as $1/r$. The dissertation states
the constraint as $r_0\omega = 1/\sqrt2$ and the property as "the same
derivative as the Coulomb operator in the short-range" (chapter 2). The erf
kernel has no such freedom: $\operatorname{erf}(\omega r)/r$ has a nonzero
$r^2$ term for every $\omega > 0$, which the right panel of the figure shows.

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

`Operator::terfc` and `Operator::terf` apply the constraint, so $r_0$ is their
only argument:

<!-- excerpt: ferric-integrals/src/operator.rs::terfc -->
```rust
let omega = 1.0 / (r0 * std::f64::consts::SQRT_2);
Self::primitive(OperatorKind::Terfc, omega, r0)
```

`Operator::terfc_with_omega` and `Operator::terf_with_omega` take $r_0$ and
$\omega$ independently. The sum rule terf + terfc = 1 holds for every
$(r_0, \omega)$, so either form splits $1/r$ exactly.

## 4.1.3 Two properties that decide how the integrals are used

**Splitting is not additive in the energy.** MP2 is quadratic in the
integrals. With $(ia|jb) = s + l$ for the short- and long-range parts,

$$ E[s + l] = E[s] + E[l] + \text{(cross term in } s\,l\text{)}, $$

so $E[\operatorname{erfc}] + E[\operatorname{erf}] \ne E[1/r]$. The
cross term is what RS-MP2 + LR-RPA is built on (chapter 26). It is checked by
claim 4 of `scripts/mp2_variant_identities.py`.

**terf is not positive definite.** The Fourier transforms of the kernels are

| Kernel | $\tilde w(k)\cdot 2\pi^2 k^2$ | Positive definite |
|---|---|---|
| erf | $e^{-k^2/4\omega^2}$ | yes |
| erfc | $1 - e^{-k^2/4\omega^2}$ | yes |
| terfc | $1 - \cos(k r_0)\,e^{-k^2/4\omega^2}$ | yes |
| terf | $\cos(k r_0)\,e^{-k^2/4\omega^2}$ | no: negative at $k r_0 = \pi$ |

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

The ferric defaults in section 4.1.4 are these values.

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
