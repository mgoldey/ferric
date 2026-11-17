# SCS-MP2(2terfc) {#scs-mp2-2terfc}

<!-- status: ready; registry: scs-mp2-2terfc -->

SCS-MP2 (section @sec:scs-and-kappa) needs one pair of coefficients for reaction energies
and a nearly reversed pair for intermolecular interactions. Attenuated MP2
(section @sec:attenuated-mp2) is good for intermolecular interactions and poor for bonded
ones: in a finite basis thermochemistry needs more correlation than MP2
gives, not less, which is why scaling-all-correlation (SAC) methods use factors
above 1 (dissertation, chapter 5, section 5.1). SCS-MP2(2terfc) gives each spin component its own attenuation
length, so that one parameter set serves both (Goldey and Head-Gordon,
J. Phys. Chem. B 118, 6519 (2014); Goldey dissertation, chapter 5).

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/scs.rs::scs_mp2_2terfc`, `ScsMp2TerfcConfig` |
| CLI | `method.kind = "scs-mp2-2terfc"`, `[mp2] r0_bonded`, `r0_nonbonded` (Å), `c_os`, `c_ss` |
| Python | `run_scs_mp2_2terfc(r0_bonded=, r0_nonbonded=, c_os=, c_ss=)`, lengths in Å |

## The energy {#scs-mp2-2terfc-energy}

The dissertation (chapter 5, section 5.2) builds the method from three
observations.

1. Bonded interactions act on a shorter length scale than non-bonded ones,
   so they get a shorter attenuation length, $r_0^{(1)} < r_0^{(2)}$.
2. SOS-MP2, which drops the same-spin part, does about as well as SCS-MP2
   for thermochemistry. So the opposite-spin part can carry the bonded
   correlation, attenuated at $r_0^{(1)}$.
3. SCS(MI)-MP2 weights the same-spin part more heavily, and a same-spin-only
   variant does about as well for non-bonded interactions. So the
   same-spin part carries the non-bonded correlation, attenuated at
   $r_0^{(2)}$. The same-spin correlation already present at $r_0^{(1)}$ is
   subtracted, so that the bonded length scale is not counted twice.

Write $E_x[r_0]$ for the spin component $x$ of (@eq:mp2-cs-spin-adaptation), evaluated with the
terfc kernel (@eq:mp2-att-terfc-kernel) at length $r_0$. The three observations give

$$ E = E_{\mathrm{HF}} + c_{\mathrm{OS}}\,E_{\mathrm{OS}}\big[r_0^{(1)}\big] + c_{\mathrm{SS}}\Big(E_{\mathrm{SS}}\big[r_0^{(2)}\big] - E_{\mathrm{SS}}\big[r_0^{(1)}\big]\Big) . \label{eq:mp2-scs2terfc-limits} $$

This is dissertation Eq. 5.6 and the code's assembly. The opposite-spin
part is short-range MP2 at the shorter length. The same-spin part is a
window: the same-spin correlation between $r_0^{(1)}$ and $r_0^{(2)}$, with
the shortest-range same-spin correlation removed.

The window is a difference of two energies, not the energy of a difference
kernel. $E_{\mathrm{SS}}$ is quadratic in the integrals. Write the
$r_0^{(2)}$ integrals as $g_1 + d$, with $g_1$ the $r_0^{(1)}$ integrals and
$d$ the integrals of $\operatorname{terfc}(r; r_0^{(2)}) - \operatorname{terfc}(r; r_0^{(1)})$.
Expand $E_{\mathrm{SS}}[g_1 + d]$ as in section @sec:erf-terf-properties-decide-integrals-used. The window is
$E_{\mathrm{SS}}[d]$ plus a cross term between $g_1$ and $d$.

The limits make the construction clear. terfc$(r; r_0) \to 0$ for $r > 0$
as $r_0 \to 0$ and $\to 1$ as $r_0 \to \infty$, and each spin component is
quadratic in the integrals, so

| Limit | (@eq:mp2-scs2terfc-limits) reduces to |
|---|---|
| $r_0^{(1)} \to 0$, $r_0^{(2)} \to \infty$ | $c_{\mathrm{SS}}\,E_{\mathrm{SS}}[1/r]$ |
| $r_0^{(1)},\ r_0^{(2)} \to \infty$ | $c_{\mathrm{OS}}\,E_{\mathrm{OS}}[1/r]$ |

Read the rows one at a time. In the first, the opposite-spin part is
attenuated away completely and the window opens to the whole Coulomb
same-spin energy. In the second, the window closes, because both same-spin
terms are the same Coulomb energy, and what is left is SOS-MP2 with
coefficient $c_{\mathrm{OS}}$. The correlation energy (without
$E_{\mathrm{HF}}$) is what the rows list.

> **Verified equation @eq:mp2-scs2terfc-limits** · the operator limits of SCS-MP2(2terfc)
>
> With symbolic integrals, (@eq:mp2-scs2terfc-limits) equals $c_{\mathrm{SS}}E_{\mathrm{SS}}$
> of the Coulomb operator when $r_0^{(1)} \to 0$, $r_0^{(2)} \to \infty$, and
> $c_{\mathrm{OS}}E_{\mathrm{OS}}$ when both lengths go to infinity.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_family_sympy.py`, part C (kernel limits in part D) |
> | Negative control | the runner reverses the window, $E_{\mathrm{SS}}[r_0^{(1)}] - E_{\mathrm{SS}}[r_0^{(2)}]$, and requires part C to fail |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/scs.rs::scs_mp2_2terfc` |
>
> <!-- verified: id=mp2.scs2terfc.limits; script=scripts/mp2_family_sympy.py; claim=C; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/scs.rs::scs_mp2_2terfc; rust_test=ferric-mp2/src/scs.rs::scs_2terfc_matches_eq12_assembled_by_hand -->

## Parameters {#scs-mp2-2terfc-parameters}

| Parameter | Value | Rust field (unit) | CLI / Python (unit) |
|---|---|---|---|
| $r_0^{(1)}$ | 0.75 Å (1.417 Bohr) | `r0_bonded` (Bohr) | `r0_bonded` (Å) |
| $r_0^{(2)}$ | 1.05 Å (1.984 Bohr) | `r0_nonbonded` (Bohr) | `r0_nonbonded` (Å) |
| $c_{\mathrm{OS}}$ | 1.27 | `c_os` | `c_os` |
| $c_{\mathrm{SS}}$ | 4.05 | `c_ss` | `c_ss` |

These are the non-VV10 fit, the parameters for the method without a VV10
dispersion tail. Fitted at aug-cc-pVTZ with frozen core and no counterpoise correction,
against a weighted objective over S66 (intermolecular) and W4-11
(thermochemistry) (dissertation Eq. 5.7). $r_0^{(2)} \le r_0^{(1)}$, or
either length non-finite or not positive, is an error: the window would be
empty or inverted.

The values are the defaults of `ScsMp2TerfcConfig` in Rust. The CLI and
Python resolve them through
`ferric-mp2/src/terfc_r0.rs::resolve_scs_2terfc`, which has one row,
aug-cc-pVTZ. In that basis an omitted value comes from the row. In any
other basis both lengths must be given (an error otherwise), an omitted
$c_{\mathrm{OS}}$ or $c_{\mathrm{SS}}$ takes the aug-cc-pVTZ value, and the
run prints a note saying so.

Each length uses the curvature link (@eq:mp2-att-terfc-kernel); there is no free-$\omega$
option here.

$c_{\mathrm{SS}} = 4.05$ is large because the same-spin window is small: it
carries only the same-spin correlation between the two lengths.

## Summary from the dissertation {#scs-mp2-2terfc-summary-dissertation}

RMSD in kcal/mol, aug-cc-pVTZ, no counterpoise (dissertation Tables 5.1,
5.2):

| Method | S66 | W4-11 |
|---|---|---|
| MP2 | 1.533 | 7.29 |
| SCS-MP2 | 0.632 | 5.16 |
| MP2(terfc, aTZ) | 0.251 | 6.97 |
| SCS-MP2(2terfc, aTZ) | 0.228 | 4.79 |

The single parameter set improves on SCS-MP2 for thermochemistry and on
attenuated MP2 for intermolecular interactions.

## How the code evaluates it {#scs-mp2-2terfc-code-evaluates}

This is `ferric-mp2/src/scs.rs::scs_mp2_2terfc` after the length check and
the `ri_config` setup, with each call joined onto one line and the comments
replaced:

```rust
// E_x[r0(1)], x = OS, SS
let sc1 = ri_mp2_spin_components_any(mol, obs, dfbs, Operator::terfc(config.r0_bonded), rhf, &ri_config)?;
// E_x[r0(2)]
let sc2 = ri_mp2_spin_components_any(mol, obs, dfbs, Operator::terfc(config.r0_nonbonded), rhf, &ri_config)?;
let e_ss = sc2.e_ss - sc1.e_ss;                              // the same-spin window
let scs_corr = config.c_os * sc1.e_os + config.c_ss * e_ss; // the energy above without E_HF
Ok(ScsMp2Result {
    total_energy: rhf.energy + scs_corr,
    scs_corr,
    e_os: sc1.e_os,
    e_ss,
})
```

Two calls, each with its own terfc metric, so the cost is twice an RI-MP2.
`sc2.e_os` is computed and not used. The reported $E_{\mathrm{SS}}$ is the
window difference, so the reported components combine with
$c_{\mathrm{OS}}$, $c_{\mathrm{SS}}$ directly. `Operator::terfc` applies the
curvature link of (@eq:mp2-att-terfc-kernel). On a UHF reference
`ferric-mp2/src/u_rimp2.rs::ri_mp2_spin_components_any` returns the blocks
of section @sec:unrestricted-ri-mp2 at both lengths, so the same lines assemble the open-shell
energy.

A one-pair molecule (H2, one occupied orbital) has $E_{\mathrm{SS}} = 0$ at
every length: with $i = j$ the bracket $(ia|ib) - (ib|ia)$ of (@eq:mp2-cs-spin-adaptation) is
zero. Tests of the window use water.

## What is anchored {#scs-mp2-2terfc-anchored}

| Quantity | Systems | Agreement | Bar | Test |
|---|---|---|---|---|
| terf/terfc 3-index and 2-index integrals vs Fourier-space quadrature on PySCF `ft_ao` | H2O, NH3 / cc-pVDZ; H2O / aug-cc-pVDZ; shell classes s–f | ≤ 1.4e-13 (3-index), ≤ 2.6e-12 (2-index) | 2e-12, 3e-11 | `crates/ferric-integrals/tests/validation_terfc_integrals.rs` |
| $E_{\mathrm{OS}}$, $E_{\mathrm{SS}}$ at 0.75, 1.00, 1.05 Å and the (@eq:mp2-scs2terfc-limits) correlation energy | same; frozen core 0 and 1 | ≤ 3.3e-12 Ha | 5e-11 Ha | `crates/ferric-mp2/tests/validation_scs_mp2_2terfc.rs` |
| Controls | | terfc at 0.75 Å misses the 1.05 Å reference by ≥ 5.2e-3 Ha; erfc at the same curvature misses terfc by ≥ 4.1e-2 Ha | | as above |

The reference shares none of ferric's integral code. The published
parameters at aug-cc-pVTZ are exercised only on a five-dimer S66 subset,
with no external anchor for the counterpoise path. Figures from
`site/src/reference/validation.md`.

## Known limits (2026-11-15)

- This section covers the energy. Open shells run on a UHF reference (see
  "Open shell" in section @sec:attenuated-mp2). The analytic nuclear gradient, closed shell
  and UHF (`run_optimize(method="scs-mp2-2terfc")`, CLI
  `task = "optimize"`), is derived in sections
  @sec:rimp2-gradient and @sec:u-rimp2-gradient-terfc-scs-mp2-2terfc, and graded on the validation
  page.
- The parameters are aug-cc-pVTZ, no-counterpoise values; other bases are
  unparameterized.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, "4. Spin-component scaling", "5. Attenuated MP2", "5.1 Open-shell attenuated and spin-scaled MP2" | equation @eq:mp2-scs2terfc-limits, limits, the one-row table, open shell |
| `crates/ferric-mp2/src/scs.rs`, `crates/ferric-mp2/src/terfc_r0.rs` | defaults, validation of the lengths, assembly, the basis rules |
| Goldey dissertation, chapter 5 (sections 5.1, 5.2, Eqs. 5.6, 5.7, Tables 5.1, 5.2) | motivation, definition, fit, summary table |
| `scripts/mp2_family_sympy.py`, parts C and D | verified box |
| `site/src/reference/validation.md` | anchors, gradient grade |

The statement that the window equals $E_{\mathrm{SS}}[d]$ plus a cross term
was checked numerically on a random model for this section; it is the
identity of section @sec:erf-terf-properties-decide-integrals-used with $g_1$ and $d$ in place of the short- and
long-range integrals.
