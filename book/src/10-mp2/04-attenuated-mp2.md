# Attenuated MP2: erfc and terfc {#attenuated-mp2}

<!-- status: edit; registry: attenuated-mp2, u-attenuated-mp2 -->

Take two molecules a few ångström apart and compute their MP2 interaction
energy in a small basis. Two errors push it the same way. First, MP2
describes dispersion with uncoupled Hartree–Fock polarizabilities and
overbinds many dispersion-dominated complexes, π-stacked ones most of all.
Second, each monomer borrows basis functions from its partner, which lowers
the dimer energy artificially: basis set superposition error (BSSE). Both
errors come from correlation at long range, and both overbind. Attenuated
MP2 keeps the Hartree–Fock energy and replaces $1/r$ in the correlation
integrals by a short-range kernel, so correlation is cut off beyond a
fitted length. In the dissertation's words, the role of attenuation is "to
remove part of the over-binding associated with BSSE in small basis sets,
as well as part of the over-binding associated with MP2 itself for some
types of dispersion interactions" (chapter 2). (Goldey and Head-Gordon,
J. Phys. Chem. Lett. 3, 3592 (2012); Goldey, Dutoi and Head-Gordon, Phys.
Chem. Chem. Phys. 15, 15869 (2013); Goldey dissertation, chapters 2 and 3.)

The price: attenuated MP2 has no long-range dispersion at all. Its $C_6$
coefficients are zero. Section @sec:mp2-v adds a model tail back.

| Entry point | Location |
|---|---|
| Rust (erfc) | `ferric-mp2/src/attenuated.rs::attenuated_ri_mp2` |
| Rust (terfc) | `ferric-mp2/src/rimp2.rs::ri_mp2` with `Operator::terfc` (`ferric-integrals/src/operator.rs`) |
| CLI | `method.kind = "att-rimp2"`, `[mp2] att_operator = "erfc"` (default) or `"terfc"` |
| Python | `run_attenuated_rimp2(omega=)` (erfc); `run_terfc_rimp2(r0=)` (terfc); `run_rimp2(att_operator=, omega=, att_r0=, att_omega=)` for the local MP2 of chapter @sec:local-mp2 only (a `ValueError` on the exact method) |

## The two kernels {#attenuated-mp2-kernels}

Chapter @sec:operators defines both kernels and derives their properties; this section
uses the results. The erfc kernel is the short-range half of the usual
range split:

$$ \frac{1}{r} = \frac{\operatorname{erfc}(\omega r)}{r} + \frac{\operatorname{erf}(\omega r)}{r} . \label{eq:attenuated-mp2-1} $$

The tempered kernel terfc is built from two error functions centred at
$\pm r_0$ (Dutoi and Head-Gordon, J. Phys. Chem. A 112, 2110 (2008);
dissertation Eq. 6.3):

$$ \operatorname{terfc}(r; r_0) = \tfrac12\Big[\operatorname{erfc}\big(\omega(r - r_0)\big) + \operatorname{erfc}\big(\omega(r + r_0)\big)\Big], \qquad \omega = \frac{1}{\sqrt2\,r_0} , \label{eq:mp2-att-terfc-kernel} $$

and $\operatorname{terf} = 1 - \operatorname{terfc}$. terfc$(0) = 1$, so
terfc$(r)/r$ keeps the full Coulomb singularity at the origin.

The link $\omega = 1/(\sqrt2\,r_0)$ is the curvature constraint. terf is odd
in $r$, so $\operatorname{terf}(r)/r = \tau_1 + \tau_3 r^2 + O(r^4)$ with

$$ \tau_3 = \frac{\omega^3}{3\sqrt\pi}\big(4\omega^2 r_0^2 - 2\big)\,e^{-\omega^2 r_0^2} , \label{eq:attenuated-mp2-3} $$

whose only positive root is $\omega = 1/(\sqrt2\,r_0)$. At that $\omega$ the
long-range part terf$(r)/r$ is constant near the origin up to an $r^4$ term, so terfc
reproduces the short-range Coulomb shape as closely as one parameter
allows. With the link, terfc has one parameter, $r_0$. Section @sec:erf-terf-curvature-constraint
derives (@eq:attenuated-mp2-3) step by step from the Taylor series of the odd part of
$\operatorname{erf}\big(\omega(r + r_0)\big)$.

> **Verified equation @eq:mp2-att-terfc-kernel** · the terfc kernel and its curvature constraint
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

![Short-range multipliers of 1/r used in chapter @sec:mp2, and the MP2-V VV10 damping. Script: `book/src/figures/10-mp2/attenuators.py`.](../figures/10-mp2/attenuators.svg)

The figure shows the two kernels at their default parameters. At
$\omega = 0.420$ Å⁻¹ and $r_0 = 1.05$ Å they nearly coincide below about 1.5 Å.
Beyond that terfc falls faster: its tail is Gaussian in $r - r_0$ rather
than in $r$.

Why are the $C_6$ coefficients zero? Both kernels decay like a Gaussian at
large $r$. An integral $(ia|jb)$ with $\phi_i\phi_a$ on one molecule and
$\phi_j\phi_b$ on another therefore falls off faster than any power of the
separation $R$, and the $R^{-6}$ term that MP2 dispersion comes from never
appears.

## The energy {#attenuated-mp2-energy}

Attenuated RI-MP2 is section @sec:closed-shell-ri-mp2 with $\hat w(r)$ (either kernel divided
by $r$) in place of $1/r$, in both the 3-index integrals and the fitting
metric:

$$ (ia|jb)_{\hat w} \approx \sum_P B^P_{ia} B^P_{jb}, \qquad B^P_{ia} = \sum_Q M_{PQ}\,(Q|\hat w|ia), \qquad M^{\mathsf T} M = V_{\hat w}^{-1}, \quad (V_{\hat w})_{PQ} = (P|\hat w|Q) . \label{eq:attenuated-mp2-4} $$

Why the attenuated metric as well? Section @sec:closed-shell-ri-mp2-density-fitting-making-affordable showed that the fitting
error is quadratic in the fitting residual (equation @eq:df-quadratic-error) when the metric
and the 3-index integrals share one operator. Fitting in the attenuated
metric keeps the fit variational for the attenuated integrals. The energy is
(@eq:closed-shell-ri-mp2-3) on these integrals:

$$ E = E_{\mathrm{HF}} + s\,E^{(2)}\big[\hat w\big] . \label{eq:attenuated-mp2-5} $$

The Hartree–Fock part is never attenuated. $s$ scales the correlation
energy only; the dissertation introduced it to absorb "systematic errors in
the correlation energy due to severe truncation in the strong attenuation
limit and BSSE in the weak attenuation limit" (chapter 2).

Check the limits of (@eq:attenuated-mp2-5) at $s = 1$.

- $\omega \to 0$ for erfc, or $r_0 \to \infty$ for terfc: the kernel goes
  to 1 at every $r$ (chapter @sec:operators, @sec:erf-terf-pairs), every integral is the Coulomb one,
  and (@eq:attenuated-mp2-5) is RI-MP2.
- $\omega \to \infty$ for erfc, or $r_0 \to 0$ for terfc: the kernel goes to
  0 at every $r > 0$, the correlation energy vanishes, and (@eq:attenuated-mp2-5) is
  Hartree–Fock.

Both limits on water in aug-cc-pVDZ, from the `18375bead` build. The
`os.environ` line points `FERRIC_TERF_TABLE_DIR` at the repository's
`terf-tables/`, which terfc needs (see "Runtime tables" below). erfc runs from
$\omega = 10^{-4}$ Å⁻¹, where it is RI-MP2, to 5 Å⁻¹, where 2% of the
correlation is left; terfc does the same as $r_0$ shrinks from 10 Å to 0.5 Å:

```python
import os

os.environ.setdefault("FERRIC_TERF_TABLE_DIR", "terf-tables")  # the terfc interpolation tables
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("aug-cc-pvdz")
aux = ferric.BasisSet.bundled("aug-cc-pvdz-ri")

mp2 = ferric.run_rimp2(mol, bs, aux)
print("RI-MP2                E_c = %.8f" % mp2.mp2_corr)
for omega in (1e-4, 0.1, 0.420, 1.0, 5.0):          # Angstrom^-1
    r = ferric.run_attenuated_rimp2(mol, bs, aux, omega=omega)
    print("erfc  omega = %6.4f  E_c = %.8f  fraction of MP2 = %.4f"
          % (omega, r.mp2_corr, r.mp2_corr / mp2.mp2_corr))
# terfc needs FERRIC_TERF_TABLE_DIR
t = ferric.run_terfc_rimp2(mol, bs, aux)           # r0 from the basis table: 1.05 A at aug-cc-pVDZ
print("terfc r0 = table      E_c = %.8f  fraction of MP2 = %.4f"
      % (t.mp2_corr, t.mp2_corr / mp2.mp2_corr))
for r0 in (0.5, 2.0, 10.0):                         # Angstrom
    t = ferric.run_terfc_rimp2(mol, bs, aux, r0=r0)
    print("terfc r0 = %5.2f      E_c = %.8f  fraction of MP2 = %.4f"
          % (r0, t.mp2_corr, t.mp2_corr / mp2.mp2_corr))
```

```text
terfc r0 = 1.05 Å (default for aug-cc-pvdz; Goldey & Head-Gordon, J. Phys. Chem. B 2014; dissertation Table 6.2)
RI-MP2                E_c = -0.22190484
erfc  omega = 0.0001  E_c = -0.22190484  fraction of MP2 = 1.0000
erfc  omega = 0.1000  E_c = -0.22180076  fraction of MP2 = 0.9995
erfc  omega = 0.4200  E_c = -0.21461209  fraction of MP2 = 0.9671
erfc  omega = 1.0000  E_c = -0.15913105  fraction of MP2 = 0.7171
erfc  omega = 5.0000  E_c = -0.00494427  fraction of MP2 = 0.0223
terfc r0 = table      E_c = -0.21998892  fraction of MP2 = 0.9914
terfc r0 =  0.50      E_c = -0.18057945  fraction of MP2 = 0.8138
terfc r0 =  2.00      E_c = -0.22187158  fraction of MP2 = 0.9999
terfc r0 = 10.00      E_c = -0.22190485  fraction of MP2 = 1.0000
```

The dissertation fitted between these two ends. At aug-cc-pVDZ on S66 the
Hartree–Fock end has an RMSD of 4.0 kcal/mol and the MP2 end 2.67 kcal/mol
(chapter 2, Table 2.1); the fitted lengths of @sec:attenuated-mp2-published-parameterizations sit in between and do
better than either.

| Parameter | Rust | CLI `[mp2]` | Python | Default |
|---|---|---|---|---|
| erfc $\omega$ | `AttenuatedMp2Config::omega`, Bohr⁻¹ | `omega`, Å⁻¹ | `omega`, Å⁻¹ | 0.420 Å⁻¹ (0.2223 Bohr⁻¹) |
| terfc $r_0$ | `Operator::terfc(r0)`, Bohr | `att_r0`, Å | `r0` / `att_r0`, Å | per basis from `ferric-mp2/src/terfc_r0.rs::TERFC_R0_TABLE` (1.05 Å, 1.984 Bohr, at aug-cc-pVDZ); an untabulated basis needs an explicit value |
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
The terfc kernel is evaluated in the C++ shim,
`crates/ferric-integrals/shim/shim.cc`, as the Coulomb integral minus the
terf integral (section @sec:erf-terf-curvature-constraint). erfc needs no tables.

## Published parameterizations {#attenuated-mp2-published-parameterizations}

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
alike: the attenuation length matters more than the kernel shape. Second,
the optimal $r_0$ grows with the basis. A larger basis has less BSSE to
cancel, so less correlation is cut. The aug-cc-pVTZ value is 1.35 Å, not
the aug-cc-pVDZ 1.05 Å. Section @sec:attenuation-basis-limit tabulates $r_0$ across 24 basis sets.

Because $r_0$ belongs to a basis, the CLI and Python read the terfc default
from a per-basis table, `ferric-mp2/src/terfc_r0.rs::resolve_terfc_r0`: the
13 bundled sets of section @sec:attenuation-basis-limit (1.05 Å at aug-cc-pVDZ, 1.35 Å at
aug-cc-pVTZ). An explicit `att_r0` always wins, a run that used the table
prints the value and its source, and a basis with no row is an error that
asks for `att_r0`. The erfc default, 0.420 Å⁻¹, is the aug-cc-pVDZ value and
is used whatever the basis.

## How the code evaluates it {#attenuated-mp2-code-evaluates}

The erfc path builds the erfc 3-index integrals and metric, then runs the
RI-MP2 pair loop of section @sec:closed-shell-ri-mp2. This is the body of
`ferric-mp2/src/attenuated.rs::attenuated_ri_mp2`, with the comments
replaced:

```rust
let op = Operator::erfc(config.omega);               // w = erfc(omega r)/r
let sc = if let Some(thresh) = config.screen_thresh {
    attenuated_spin_components_screened(mol, obs, dfbs, op, rhf, config.frozen_core, thresh)?
} else {
    let ri_config = RiMp2Config {
        frozen_core: config.frozen_core,
        memory_budget_bytes: config.memory_budget_bytes,
        ..Default::default()
    };
    ri_mp2_spin_components_any(mol, obs, dfbs, op, rhf, &ri_config)?   // fitted integrals, then the RI-MP2 energy
};
let scaled_corr = config.scaling * sc.e_total;      // s E^(2)[w]
Ok(AttenuatedMp2Result {
    mp2_corr: scaled_corr,
    total_energy: rhf.energy + scaled_corr,         // E_HF + s E^(2)[w]: HF unattenuated
    spin_components: sc,
})
```

Three points about it:

- **The operator is the only change.** `op` goes into the same
  `ri_mp2_spin_components_any` that plain RI-MP2 uses, and that function
  builds both the metric and the 3-index integrals with it. So (@eq:attenuated-mp2-4) holds
  by construction.
- **The metric takes the Cholesky route.** The erfc and terfc kernels have
  positive Fourier transforms (section @sec:erf-terf-properties-decide-integrals-used), so their metrics are positive
  definite and `ferric-mp2/src/rimp2.rs::metric_inverse_sqrt` factorizes
  them by Cholesky, as for Coulomb. The terfc path is `ri_mp2` with the
  operator swapped and goes the same way.
- **`spin_components` stays unscaled.** Only `mp2_corr` and
  `total_energy` carry $s$, so the reported $E_{\mathrm{OS}}$ and
  $E_{\mathrm{SS}}$ are the attenuated components themselves.

The screened variant skips shell triples by the QQR-3 bound
(`ferric-integrals/src/qqr3.rs::QqrBounds3`). That bound multiplies the
Schwarz product by a distance envelope, and the envelope is the Coulomb one
for every operator. The attenuation enters only through the Schwarz
factors, which are computed with the erfc kernel. Those factors are
self-interactions, $(P|P)$ and $(\mu\nu|\mu\nu)$, in which both electrons
sit in the same charge cloud, so they shrink little. Measured on decane /
cc-pVDZ at a threshold of 1e-8, the screen keeps 79.1% of the shell triples
for Coulomb and 78.5% for erfc at $\omega = 0.222$ Bohr⁻¹ (module
documentation of `attenuated.rs`).

`attenuated_ri_mp2_long_range` evaluates the erf (long-range) half, and
`rs_mp2_decomposition` returns the short-range, long-range and full
Coulomb energies together (section @sec:range-split-cross-term). The two halves do not add up to
the Coulomb energy, because MP2 is quadratic in the integrals (section
@sec:erf-terf-properties-decide-integrals-used).

## What is anchored {#attenuated-mp2-anchored}

| Quantity | Systems | Agreement | Bar | Test |
|---|---|---|---|---|
| erfc RI-MP2 $E_{\mathrm{corr}}$ vs numpy RI-MP2 on PySCF `with_range_coulomb(-ω)` integrals, ω = 0.2, 0.222, 0.42, 1.0 Bohr⁻¹ | H2O, NH3, CH4 / aug-cc-pVDZ | ≤ 5.9e-12 Ha at every ω; ω → 0 gives Coulomb RI-MP2 to 1.6e-15 | 1e-10 Ha | `crates/ferric-mp2/tests/validation_attenuated_mp2.rs` |
| terfc $E_{\mathrm{OS}}$, $E_{\mathrm{SS}}$ at $r_0$ = 0.75, 1.00, 1.05 Å vs numpy RI-MP2 on Fourier-space terfc integrals | H2O, NH3 / cc-pVDZ; H2O / aug-cc-pVDZ | ≤ 3.3e-12 Ha | 5e-11 Ha | `crates/ferric-mp2/tests/validation_scs_mp2_2terfc.rs` |
| terfc 3-index and 2-index integrals, every shell class s–f | as above | ≤ 1.4e-13 (3-index), ≤ 2.6e-12 (2-index) | 2e-12, 3e-11 | `crates/ferric-integrals/tests/validation_terfc_integrals.rs` |

The terfc energies are the spin components of section @sec:scs-mp2-2terfc's anchor; the
terfc energy of this section is their sum. Figures from
`site/src/reference/validation.md`.

## Known limits (2026-11-15)

- The scaled variants ($s \ne 1$) are reachable from Rust only.
- The run certificate anchors the erfc operator only (frozen core 0, $s = 1$,
  $\omega$ at the default, 0.2, 0.42 or 1.0 Bohr⁻¹ as listed in
  `site/src/reference/validation.md`); a terfc run is anchored by the
  section @sec:scs-mp2-2terfc tests, not by its certificate.
- This section covers the energy. The analytic nuclear gradient (erfc and
  terfc, RHF and UHF; `task = "optimize"` with `att-rimp2`,
  `run_optimize(method="att-rimp2")`) is derived in sections
  @sec:rimp2-gradient-attenuated-erfc-ri-mp2,
  @sec:u-rimp2-gradient-attenuated-erfc-variant and
  @sec:u-rimp2-gradient-terfc-scs-mp2-2terfc, and graded on the validation
  page.
- The terfc default $r_0$ exists only for the 13 tabulated bases; the erfc
  default $\omega$ is the aug-cc-pVDZ value for every basis.
- Parameters were fitted without counterpoise; applying them to
  counterpoise-corrected interaction energies is a different method.

## Open shell

With `multiplicity > 1` the CLI kinds `att-rimp2`, `scs-mp2` and
`scs-mp2-2terfc`, and the Python drivers `run_attenuated_rimp2`,
`run_terfc_rimp2`, `run_scs_mp2` and `run_scs_mp2_2terfc`, take a UHF
reference. All of them, and `attenuated_ri_mp2_long_range`, read their spin
components from one function,
`ferric-mp2/src/u_rimp2.rs::ri_mp2_spin_components_any`. On a restricted
reference it runs the closed-shell components unchanged. On a UHF reference,
or a ROHF one, it runs `ferric-mp2/src/u_rimp2.rs::u_ri_mp2`, which
semicanonicalizes a ROHF reference per spin first. The spin components are
then the blocks of unrestricted RI-MP2 (section @sec:unrestricted-ri-mp2)
under the same operator:
$E_{\rm OS} = E_{\alpha\beta}$ and $E_{\rm SS} = E_{\alpha\alpha} + E_{\beta\beta}$.
A `kappa` regularization on an open-shell reference is an error: the
$\kappa$ kernel is closed shell only.

The spin-scaled energies of sections @sec:scs-and-kappa and @sec:scs-mp2-2terfc
keep their form, with the blocks in place of the components:

$$ E_{\rm SCS} = c_{\rm OS}E_{\alpha\beta}[1/r] + c_{\rm SS}\big(E_{\alpha\alpha} + E_{\beta\beta}\big)[1/r], $$

$$ E_{\rm SCS(2terfc)} = c_{\rm OS}E_{\alpha\beta}\big[r_0^{(1)}\big] + c_{\rm SS}\Big(\big(E_{\alpha\alpha}+E_{\beta\beta}\big)\big[r_0^{(2)}\big] - \big(E_{\alpha\alpha}+E_{\beta\beta}\big)\big[r_0^{(1)}\big]\Big), $$

each without $E_{\rm HF}$, and attenuated MP2 is
$E_{\alpha\alpha} + E_{\beta\beta} + E_{\alpha\beta}$ under erfc, erf or
terfc.

Why does the derivation of section @sec:unrestricted-ri-mp2 carry over? It used one property of $1/r$: it
does not act on spin, so an integral is nonzero only when each electron
keeps its spin. erfc$(\omega r)/r$ and terfc$(r)/r$ are spin-free too. The
exchange integral needs both electrons in one spin, so it appears only in
the same-spin blocks, and no new kernel is needed. The operator limits hold
block by block: $\omega \to 0$ gives U-RI-MP2 and $\omega \to \infty$ sends
every block to zero.

With $\alpha = \beta$ orbitals the blocks reduce to the closed-shell
components (@sec:unrestricted-ri-mp2-energy), and a closed-shell molecule run through the UHF path
gives the restricted energies to rounding (≤ 6e-17 Ha on water/cc-pVDZ for
erfc, erf, terfc, SCS and SCS(2terfc)). On OH, NH2 and O2 at cc-pVDZ the erfc
energy and the SCS energy match PySCF `DFUMP2` on the same fits to ≤ 4e-15 Ha.
The QQR-3 screened build (`screen_thresh`) is closed shell only and refuses
an open-shell reference.

`scripts/u_attenuated_sympy.py` checks the block split. On an open-shell
model with different $\alpha$ and $\beta$ orbitals and different
occupation counts, $E_{\alpha\alpha} + E_{\beta\beta} + E_{\alpha\beta}$
equals the spin-orbital sum $\tfrac14\sum|\langle ij\|ab\rangle|^2/D$,
which uses no spin blocks; the pairs with $\sigma_i \ne \sigma_j$ give
$E_{\alpha\beta}$ and those with $\sigma_i = \sigma_j$ give
$E_{\alpha\alpha} + E_{\beta\beta}$. The singlet check sets
$\alpha = \beta$ and recovers $E_{\rm OS}$ and $E_{\rm SS}$ of the
closed-shell kernel.

## Sources

| Source | Used for |
|---|---|
| section @sec:scs-and-kappa-scs-mp2 (terfc kernel) | equations @eq:attenuated-mp2-1–@eq:attenuated-mp2-5, units, the per-basis $r_0$ table, open shell |
| `crates/ferric-mp2/src/attenuated.rs`, `crates/ferric-mp2/src/terfc_r0.rs`, `crates/ferric-integrals/src/operator.rs`, `crates/ferric-integrals/src/qqr3.rs`, `crates/ferric-cli/src/config.rs` | defaults, keys, metric, screening |
| Goldey and Head-Gordon, Chem. Phys. Lett. 608, 249 (2014) | the per-basis terfc $r_0$ table (`ferric-mp2/src/terfc_r0.rs::TERFC_R0_TABLE`) |
| Goldey dissertation, chapter 2 (Table 2.1), chapter 3 (Eqs. 3.1–3.2, Table 3.1), chapter 5 (section 5.1), chapter 6 (Eq. 6.3) | definitions, the two same-sign errors, the scaling factor, parameters, summary table |
| `scripts/mp2_family_sympy.py`, part D | verified box |
| `site/src/reference/validation.md` | anchors, gradient grade |

Changes from the sources: the dissertation writes terfc with
$(r \mp r_0)/(r_0\sqrt2)$; (@eq:mp2-att-terfc-kernel) writes the same thing with
$\omega = 1/(\sqrt2\,r_0)$ to match the code.
