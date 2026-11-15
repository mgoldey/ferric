# 10.7 The range-split cross term

<!-- section: 10.7; status: ready; registry: attenuated-mp2;
     theory: crates/ferric-mp2/theory.md#"5. Attenuated MP2" -->

Splitting the operator does not split the MP2 energy. This section derives
the term that is left over. It matters for every method that pairs a
short-range MP2 with a separate long-range treatment: MP2-V (section 10.8)
and RS-MP2 + LR-RPA (`ferric-rpa/src/rs_mp2_rpa.rs`).

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/attenuated.rs::rs_mp2_decomposition` (returns the erfc, erf and Coulomb correlation energies) |
| CLI, Python | none directly; `rs-mp2-rpa` (`run_rs_mp2_rpa`) is built on it |

## 10.7.1 The identity

Write the Coulomb integral as a short-range part and a long-range part,
$(ia|jb) = s_{ijab} + l_{ijab}$, for example
$s$ from $\operatorname{erfc}(\omega r)/r$ and $l$ from
$\operatorname{erf}(\omega r)/r$ as in (10.11). Write $x^\times_{ijab} = x_{ijba}$
for the exchange partner. The closed-shell energy (10.3) is
$E[g] = -\sum g\,(2g - g^\times)/\Delta$, quadratic in $g$, so

$$ E[s + l] = E[s] + E[l] - \sum_{ijab} \frac{4\,s_{ijab}\,l_{ijab} - s_{ijab}\,l^\times_{ijab} - l_{ijab}\,s^\times_{ijab}}{\Delta_{ij}^{ab}} . \tag{10.18} $$

The last sum is the cross term $X$: correlation in which one interaction
of the pair is short-range and the other long-range. In general
$X \ne 0$, so $E[\operatorname{erfc}] + E[\operatorname{erf}] \ne E[1/r]$.

> **Verified equation 10.18** · MP2 is not additive in a split operator
>
> For random factor tensors $s$, $l$ and orbital energies,
> $E[s+l] = E[s] + E[l] + X$ holds, and $E[s] + E[l]$ alone differs from
> $E[s+l]$.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 4 |
> | Negative control | the runner changes the $4 s l$ coefficient to 2 and requires the claim to fail |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/attenuated.rs::rs_mp2_decomposition` |
>
> <!-- verified: id=mp2.rs.cross-term; script=scripts/mp2_variant_identities.py; claim=4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/attenuated.rs::rs_mp2_decomposition, ferric-mp2/src/attenuated.rs::attenuated_ri_mp2_long_range -->

## 10.7.2 Consequences

| Construction | What it does with $X$ |
|---|---|
| Attenuated MP2 (section 10.4) | drops $E[l]$ and $X$ together; the fitted length absorbs the loss |
| MP2-V (section 10.8) | drops $E[l]$ and $X$, then adds a model long-range tail (damped VV10) |
| RS-MP2 + LR-RPA, formulation B (`delta-lr`) | keeps full-Coulomb MP2 and corrects the long range by $E_{\mathrm{dRPA}}[\operatorname{erf}] - 2E_{\mathrm{OS}}[\operatorname{erf}]$, so $X$ is never separated |

Adding a short-range MP2 to a long-range MP2 is therefore not MP2. A method
that wants full MP2 at short range and something else at long range has to
say where $X$ goes.

## 10.7.3 In the code

`rs_mp2_decomposition` runs three RI-MP2 energies: erfc, erf and Coulomb,
each fitted in its own metric. In RI, (10.18) holds exactly only for
integrals in one common fit; with three fits the residual
$E[1/r] - E[\operatorname{erfc}] - E[\operatorname{erf}]$ is $-X$ plus the
difference of three fitting errors.

The unit test
`ferric-mp2/src/attenuated.rs::tests::test_rs_mp2_decomposition_sums_approximately_to_full`
bounds that residual by 0.05 Ha for H2 at $\omega = 0.5$ Bohr⁻¹. That is a
smoke check that the three energies are wired, not a measurement of $X$.

## Known limits (2026-11-15)

- No entry point returns $X$ itself.
- No analytic anchor for the residual beyond the proof's identity.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, "5. Attenuated MP2" ("Range split") | equation 10.18 |
| `crates/ferric-mp2/src/attenuated.rs` | the decomposition, the test |
| `scripts/mp2_variant_identities.py`, claim 4 | verified box |

Changes from the sources: none; theory.md and (10.18) both use the positive
$\Delta$ with an explicit minus sign.
