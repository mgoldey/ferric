# 10.4 Attenuated MP2: erfc and terfc

<!-- section: 10.4; status: edit; registry: attenuated-mp2;
     theory: crates/ferric-mp2/theory.md#"5. Attenuated MP2", "4. Spin-component scaling" -->

Attenuated MP2 keeps the Hartree–Fock energy and replaces $1/r_{12}$ in the
MP2 correlation integrals with a short-range kernel. The long-range
correlation it removes is, in a finite basis, mostly error: MP2 overbinds
dispersion-dominated complexes (uncoupled Hartree–Fock polarizabilities),
and basis-set superposition error (BSSE) overbinds them further. Both
errors grow with distance and have the same sign, so cutting the
correlation off at a fitted length removes much of both (Goldey and
Head-Gordon, J. Phys. Chem. Lett. 3, 3592 (2012); Goldey, Dutoi and
Head-Gordon, Phys. Chem. Chem. Phys. 15, 15869 (2013); Goldey dissertation,
chapters 2 and 3).

The price: attenuated MP2 has no long-range dispersion at all. Its $C_6$
coefficients are zero. Section 10.8 adds a model tail back.

| Entry point | Location |
|---|---|
| Rust (erfc) | `ferric-mp2/src/attenuated.rs::attenuated_ri_mp2` |
| Rust (terfc) | `ferric-mp2/src/rimp2.rs::ri_mp2` with `Operator::terfc` (`ferric-integrals/src/operator.rs`) |
| CLI | `method.kind = "att-rimp2"`, `[mp2] att_operator = "erfc"` (default) or `"terfc"` |
| Python | `run_attenuated_rimp2(omega=)` (erfc); `run_terfc_rimp2(r0=)` (terfc); `run_rimp2(att_operator=, omega=, att_r0=, att_omega=)` |

## 10.4.1 The two kernels

The erfc kernel is the short-range half of the usual range split:

$$ \frac{1}{r} = \frac{\operatorname{erfc}(\omega r)}{r} + \frac{\operatorname{erf}(\omega r)}{r} . \tag{10.11} $$

The tempered kernel terfc is built from two error functions centred at
$\pm r_0$ (Dutoi and Head-Gordon, J. Phys. Chem. A 112, 2110 (2008);
dissertation Eq. 6.3):

$$ \operatorname{terfc}(r; r_0) = \tfrac12\Big[\operatorname{erfc}\big(\omega(r - r_0)\big) + \operatorname{erfc}\big(\omega(r + r_0)\big)\Big], \qquad \omega = \frac{1}{\sqrt2\,r_0} , \tag{10.12} $$

and $\operatorname{terf} = 1 - \operatorname{terfc}$. terfc$(0) = 1$, so
terfc$(r)/r$ keeps the full Coulomb singularity at the origin.

The link $\omega = 1/(\sqrt2\,r_0)$ is the curvature constraint. terf is odd
in $r$, so $\operatorname{terf}(r)/r = \tau_1 + \tau_3 r^2 + O(r^4)$ with

$$ \tau_3 = \frac{\omega^3}{3\sqrt\pi}\big(4\omega^2 r_0^2 - 2\big)\,e^{-\omega^2 r_0^2} , \tag{10.13} $$

whose only positive root is $\omega = 1/(\sqrt2\,r_0)$. At that $\omega$ the
long-range part terf$(r)/r$ is constant near the origin up to an $r^4$ term, so terfc
reproduces the short-range Coulomb shape as closely as one parameter
allows. With the link, terfc has one parameter, $r_0$.

> **Verified equation 10.12** · the terfc kernel and its curvature constraint
>
> terf + terfc = 1; terfc$(0) = 1$; terf is odd; the $r^2$ coefficient of
> terf$(r)/r$ vanishes at, and only at, $\omega = 1/(\sqrt2\,r_0)$;
> terfc → 0 as $r \to \infty$ and as $r_0 \to 0$ at fixed $r > 0$.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_family_sympy.py`, part D |
> | Negative control | the runner replaces $1/(\sqrt2\,r_0)$ by $1/(2 r_0)$ and requires part D to fail |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-integrals/src/operator.rs::terfc` |
>
> <!-- verified: id=mp2.att.terfc-kernel; script=scripts/mp2_family_sympy.py; claim=D; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-integrals/src/operator.rs::terfc -->

![Short-range multipliers of 1/r used in chapter 10, and the MP2-V VV10 damping. Script: `book/src/figures/10-mp2/attenuators.py`.](../figures/10-mp2/attenuators.svg)

The figure shows the two kernels at their default parameters. At
$\omega = 0.420$ Å⁻¹ and $r_0 = 1.05$ Å they nearly coincide below about 1.5 Å.
Beyond that terfc falls faster: its tail is Gaussian in $r - r_0$ rather
than in $r$.

## 10.4.2 The energy

Attenuated RI-MP2 is section 10.1 with $\hat w(r)$ (either kernel divided
by $r$) in place of $1/r$, in both the 3-index integrals and the fitting
metric:

$$ (ia|jb)_{\hat w} \approx \sum_P B^P_{ia} B^P_{jb}, \qquad B^P_{ia} = \sum_Q M_{PQ}\,(Q|\hat w|ia), \qquad M^{\mathsf T} M = V_{\hat w}^{-1}, \quad (V_{\hat w})_{PQ} = (P|\hat w|Q) . \tag{10.14} $$

Fitting in the attenuated metric keeps the fit variational for the
attenuated integrals. The energy is (10.3) on these integrals:

$$ E = E_{\mathrm{HF}} + s\,E^{(2)}\big[\hat w\big] . \tag{10.15} $$

The Hartree–Fock part is never attenuated. $s$ scales the correlation
energy only.

| Parameter | Rust | CLI `[mp2]` | Python | Default |
|---|---|---|---|---|
| erfc $\omega$ | `AttenuatedMp2Config::omega`, Bohr⁻¹ | `omega`, Å⁻¹ | `omega`, Å⁻¹ | 0.420 Å⁻¹ (0.2223 Bohr⁻¹) |
| terfc $r_0$ | `Operator::terfc(r0)`, Bohr | `att_r0`, Å | `r0` / `att_r0`, Å | 1.05 Å (1.984 Bohr) |
| terfc free $\omega$ | `Operator::terfc_with_omega(r0, omega)` | `att_omega`, Å⁻¹ | `att_omega`, Å⁻¹ | unset: the curvature link |
| scaling $s$ | `AttenuatedMp2Config::scaling` | not exposed (1) | not exposed (1) | 1.0 |

**Units.** An inverse length in Å⁻¹ is multiplied by 0.529177 (Bohr per Å)
to give Bohr⁻¹; a length in Å is divided by 0.529177. So 0.420 Å⁻¹ is
0.2223 Bohr⁻¹, and the linked terfc width at $r_0 = 1.05$ Å is
$\omega = 0.673$ Å⁻¹ (0.356 Bohr⁻¹). The conversion constant is
`BOHR_INV_PER_ANG_INV` in `crates/ferric-mp2/src/attenuated.rs`.

Keys are strict. `att_r0` or `att_omega` with the erfc operator is an
error, and so is `omega` with terfc. A free `att_omega` breaks the
curvature link; the tables cover $r_0\omega$ up to about 8.9.

**Runtime tables.** terfc and terf integrals come from the Dutoi–Goldey
two-dimensional interpolation tables, located by `FERRIC_TERF_TABLE_DIR`.
erfc needs no tables.

## 10.4.3 Published parameterizations

The parameters are fitted to S66 interaction energies without counterpoise
correction, and they belong to the basis they were fitted in.

| Method | Kernel | Parameter | $s$ | Basis | S66 RMSD (kcal/mol) | Source |
|---|---|---|---|---|---|---|
| MP2 | Coulomb | | 1 | aug-cc-pVDZ | 2.67 | dissertation Table 2.1 |
| MP2/CBS | Coulomb | | 1 | CBS | 0.73 | dissertation Table 2.1 |
| MP2(terfc, aDZ) | terfc | $r_0$ = 1.05 Å | 1 | aug-cc-pVDZ | 0.46 | dissertation ch. 2, Table 2.1 |
| scaled terfc | terfc | $r_0$ = 1.00 Å | 1.06 | aug-cc-pVDZ | 0.47 | dissertation ch. 2, Table 2.1 |
| MP2(erfc, aDZ) | erfc | $\omega$ = 0.420 Å⁻¹ | 1 | aug-cc-pVDZ | 0.48 | dissertation ch. 2, Table 2.1 |
| scaled erfc | erfc | $\omega$ = 0.420 Å⁻¹ | 0.99 | aug-cc-pVDZ | 0.48 | dissertation ch. 2, Table 2.1 |
| MP2 | Coulomb | | 1 | aug-cc-pVTZ | 1.53 | dissertation Table 3.1 |
| MP2(terfc, aTZ) | terfc | $r_0$ = 1.35 Å | 1 | aug-cc-pVTZ | 0.25 | dissertation ch. 3, Table 3.1 |

Two readings of the table. First, at aug-cc-pVDZ, erfc and terfc perform
alike; the attenuation length matters more than the kernel shape. Second,
the optimal $r_0$ grows with the basis: a larger basis has less BSSE to
cancel, so less correlation is cut. The aug-cc-pVTZ value is 1.35 Å, not
the 1.05 Å default. Section 10.6 tabulates $r_0$ across 24 basis sets.

The CLI and Python default, $r_0 = 1.05$ Å, is the aug-cc-pVDZ value; it
is used whatever the basis. Set `att_r0 = 1.35` for aug-cc-pVTZ.

## 10.4.4 How the code evaluates it

The erfc path (`attenuated_ri_mp2`) builds the erfc 3-index integrals and
metric, then runs the RI-MP2 pair loop of section 10.1. A screened variant
skips shell triples by the QQR-3 bound, which takes the attenuation into
account; Schwarz bounds alone do not see it. The terfc path is `ri_mp2`
with the operator swapped; its metric is positive definite and takes the
Cholesky route.

`attenuated_ri_mp2_long_range` evaluates the erf (long-range) half, and
`rs_mp2_decomposition` returns the short-range, long-range and full
Coulomb energies together (section 10.7).

## 10.4.5 What is anchored

| Quantity | Systems | Agreement | Bar | Test |
|---|---|---|---|---|
| erfc RI-MP2 $E_{\mathrm{corr}}$ vs numpy RI-MP2 on PySCF `with_range_coulomb(-ω)` integrals, ω = 0.2, 0.222, 0.42, 1.0 Bohr⁻¹ | H2O, NH3, CH4 / aug-cc-pVDZ | ≤ 5.9e-12 Ha at every ω; ω → 0 gives Coulomb RI-MP2 to 1.6e-15 | 1e-10 Ha | `crates/ferric-mp2/tests/validation_attenuated_mp2.rs` |
| terfc $E_{\mathrm{OS}}$, $E_{\mathrm{SS}}$ at $r_0$ = 0.75, 1.00, 1.05 Å vs numpy RI-MP2 on Fourier-space terfc integrals | H2O, NH3 / cc-pVDZ; H2O / aug-cc-pVDZ | ≤ 3.3e-12 Ha | 5e-11 Ha | `crates/ferric-mp2/tests/validation_scs_mp2_2terfc.rs` |
| terfc 3-index and 2-index integrals, every shell class s–f | as above | ≤ 1.4e-13 (3-index), ≤ 2.6e-12 (2-index) | 2e-12, 3e-11 | `crates/ferric-integrals/tests/validation_terfc_integrals.rs` |

The terfc energies are the spin components of section 10.5's anchor; the
terfc energy of this section is their sum. Figures from
`site/src/reference/validation.md`.

## Known limits (2026-11-15)

- The scaled variants ($s \ne 1$) are reachable from Rust only.
- The run certificate anchors the erfc operator only (frozen core 0, $s = 1$,
  $\omega$ at the default, 0.2, 0.42 or 1.0 Bohr⁻¹ as listed in
  `site/src/reference/validation.md`); a terfc run is anchored by the
  section 10.5 tests, not by its certificate.
- No gradient for either kernel.
- The CLI terfc default is the aug-cc-pVDZ $r_0$ for every basis.
- Parameters were fitted without counterpoise; applying them to
  counterpoise-corrected interaction energies is a different method.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, "4. Spin-component scaling" (terfc kernel), "5. Attenuated MP2" | equations 10.11–10.15, units |
| `crates/ferric-mp2/src/attenuated.rs`, `crates/ferric-integrals/src/operator.rs`, `crates/ferric-cli/src/config.rs` | defaults, keys, metric, screening |
| Goldey dissertation, chapter 2 (Table 2.1), chapter 3 (Eqs. 3.1–3.2, Table 3.1), chapter 6 (Eq. 6.3) | definitions, parameters, summary table |
| `scripts/mp2_family_sympy.py`, part D | verified box |
| `site/src/reference/validation.md` | anchors |

Changes from the sources: the dissertation writes terfc with
$(r \mp r_0)/(r_0\sqrt2)$; (10.12) writes the same thing with
$\omega = 1/(\sqrt2\,r_0)$ to match the code.
