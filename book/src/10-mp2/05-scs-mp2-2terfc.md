# 10.5 SCS-MP2(2terfc)

<!-- section: 10.5; status: ready; registry: scs-mp2-2terfc;
     theory: crates/ferric-mp2/theory.md#"4. Spin-component scaling" -->

SCS-MP2 (section 10.3) needs one pair of coefficients for reaction energies
and a different pair for intermolecular interactions. Attenuated MP2
(section 10.4) is good for intermolecular interactions and poor for bonded
ones. SCS-MP2(2terfc) combines the two so that one parameter set serves
both (Goldey and Head-Gordon, J. Phys. Chem. B 118, 6519 (2014); Goldey
dissertation, chapter 5).

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/scs.rs::scs_mp2_2terfc`, `ScsMp2TerfcConfig` |
| CLI | `method.kind = "scs-mp2-2terfc"`, `[mp2] r0_bonded`, `r0_nonbonded` (Å), `c_os`, `c_ss` |
| Python | `run_scs_mp2_2terfc(r0_bonded=, r0_nonbonded=, c_os=, c_ss=)`, lengths in Å |

## 10.5.1 The energy

The spin components are evaluated twice, with the terfc kernel (10.12) at
two lengths $r_0^{(1)} < r_0^{(2)}$:

$$ E = E_{\mathrm{HF}} + c_{\mathrm{OS}}\,E_{\mathrm{OS}}\big[r_0^{(1)}\big] + c_{\mathrm{SS}}\Big(E_{\mathrm{SS}}\big[r_0^{(2)}\big] - E_{\mathrm{SS}}\big[r_0^{(1)}\big]\Big) . \tag{10.16} $$

This is dissertation Eq. 5.6 and the code's assembly. The opposite-spin
part is short-range MP2 at the shorter length. The same-spin part is a
window: the same-spin correlation between $r_0^{(1)}$ and $r_0^{(2)}$, with
the shortest-range same-spin correlation removed.

The limits make the construction clear. terfc$(r; r_0) \to 0$ for $r > 0$
as $r_0 \to 0$ and $\to 1$ as $r_0 \to \infty$, and each spin component is
quadratic in the integrals, so

| Limit | (10.16) reduces to |
|---|---|
| $r_0^{(1)} \to 0$, $r_0^{(2)} \to \infty$ | $c_{\mathrm{SS}}\,E_{\mathrm{SS}}[1/r]$ |
| $r_0^{(1)},\ r_0^{(2)} \to \infty$ | $c_{\mathrm{OS}}\,E_{\mathrm{OS}}[1/r]$ |

> **Verified equation 10.16** · the operator limits of SCS-MP2(2terfc)
>
> With symbolic integrals, (10.16) equals $c_{\mathrm{SS}}E_{\mathrm{SS}}$
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

## 10.5.2 Parameters

| Parameter | Value | Rust field (unit) | CLI / Python (unit) |
|---|---|---|---|
| $r_0^{(1)}$ | 0.75 Å (1.417 Bohr) | `r0_bonded` (Bohr) | `r0_bonded` (Å) |
| $r_0^{(2)}$ | 1.05 Å (1.984 Bohr) | `r0_nonbonded` (Bohr) | `r0_nonbonded` (Å) |
| $c_{\mathrm{OS}}$ | 1.27 | `c_os` | `c_os` |
| $c_{\mathrm{SS}}$ | 4.05 | `c_ss` | `c_ss` |

Fitted at aug-cc-pVTZ with frozen core and no counterpoise correction,
against a weighted objective over S66 (intermolecular) and W4-11
(thermochemistry) (dissertation Eq. 5.7). $r_0^{(2)} \le r_0^{(1)}$, or
either length non-finite or not positive, is an error: the window would be
empty or inverted.

Each length uses the curvature link (10.12); there is no free-$\omega$
option here.

The large $c_{\mathrm{SS}}$ is not a typo. The same-spin window is small:
it carries only the same-spin correlation between the two lengths.

## 10.5.3 Summary from the dissertation

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

## 10.5.4 How the code evaluates it

Two calls to `ri_mp2_spin_components` with `Operator::terfc` at the two
lengths, each with its own terfc metric. The cost is twice an RI-MP2. The
reported $E_{\mathrm{SS}}$ is the window difference, so the reported
components combine with $c_{\mathrm{OS}}$, $c_{\mathrm{SS}}$ directly.

A one-pair molecule (H2, one occupied orbital) has $E_{\mathrm{SS}} = 0$ at
every length; tests of the window use water.

## 10.5.5 What is anchored

| Quantity | Systems | Agreement | Bar | Test |
|---|---|---|---|---|
| terf/terfc 3-index and 2-index integrals vs Fourier-space quadrature on PySCF `ft_ao` | H2O, NH3 / cc-pVDZ; H2O / aug-cc-pVDZ; shell classes s–f | ≤ 1.4e-13 (3-index), ≤ 2.6e-12 (2-index) | 2e-12, 3e-11 | `crates/ferric-integrals/tests/validation_terfc_integrals.rs` |
| $E_{\mathrm{OS}}$, $E_{\mathrm{SS}}$ at 0.75, 1.00, 1.05 Å and the (10.16) correlation energy | same; frozen core 0 and 1 | ≤ 3.3e-12 Ha | 5e-11 Ha | `crates/ferric-mp2/tests/validation_scs_mp2_2terfc.rs` |
| Controls | | terfc at 0.75 Å misses the 1.05 Å reference by ≥ 5.2e-3 Ha; erfc at the same curvature misses terfc by ≥ 4.1e-2 Ha | | as above |

The reference shares none of ferric's integral code. The published
parameters at aug-cc-pVTZ are exercised only on a five-dimer S66 subset,
with no external anchor for the counterpoise path. Figures from
`site/src/reference/validation.md`.

## Known limits (2026-11-15)

- Closed-shell only; no gradient.
- The parameters are aug-cc-pVTZ, no-counterpoise values; other bases are
  unparameterized.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, "4. Spin-component scaling" | equation 10.16, limits |
| `crates/ferric-mp2/src/scs.rs` | defaults, validation of the lengths, assembly |
| Goldey dissertation, chapter 5 (Eqs. 5.6, 5.7, Tables 5.1, 5.2) | definition, fit, summary table |
| `scripts/mp2_family_sympy.py`, parts C and D | verified box |
| `site/src/reference/validation.md` | anchors |
