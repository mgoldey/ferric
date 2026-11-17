# The range-split cross term {#range-split-cross-term}

<!-- status: ready; registry: attenuated-mp2;
     theory: crates/ferric-mp2/theory.md#"5. Attenuated MP2" -->

MP2 correlates two electrons through two interactions: one excites the
pair, the other returns it to the ground state. Split the Coulomb operator
into a short-range and a long-range piece, and each of those two
interactions can be either piece. Short with short is short-range MP2, long
with long is long-range MP2, and one of each is a third contribution, the
cross term $X$, that belongs to neither. Every method that pairs a
short-range MP2 with a separate long-range treatment has to decide what to
do with $X$: MP2-V (section @sec:mp2-v) and RS-MP2 + LR-RPA
(`ferric-rpa/src/rs_mp2_rpa.rs`) decide differently.

The plan: derive the identity (@sec:range-split-cross-term-identity), list what each construction does
with $X$ (@sec:range-split-cross-term-consequences), and read the code that evaluates the three energies
(@sec:range-split-cross-term-code).

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/attenuated.rs::rs_mp2_decomposition` (returns the erfc, erf and Coulomb correlation energies) |
| CLI, Python | none directly; `rs-mp2-rpa` (`run_rs_mp2_rpa`) is built on it |

## The identity {#range-split-cross-term-identity}

Write the Coulomb integral as a short-range part and a long-range part,
$(ia|jb) = s_{ijab} + l_{ijab}$. The usual choice takes $s$ from
$\operatorname{erfc}(\omega r)/r$ and $l$ from $\operatorname{erf}(\omega r)/r$,
the split (@eq:attenuated-mp2-1) of section @sec:attenuated-mp2 (equation @eq:erf-terf-1 of section @sec:erf-terf-pairs).
Write $x^\times_{ijab} = x_{ijba}$ for the exchange partner of any
four-index quantity.

The closed-shell energy (@eq:closed-shell-ri-mp2-3) is a function of the integrals alone:

$$ E[g] = -\sum_{ijab} \frac{g_{ijab}\,\big(2g_{ijab} - g^\times_{ijab}\big)}{\Delta_{ij}^{ab}} . $$

It is quadratic in $g$. Substitute $g = s + l$ and expand the numerator:

$$ (s + l)\big(2s + 2l - s^\times - l^\times\big) = s\,(2s - s^\times) + l\,(2l - l^\times) + 4\,s\,l - s\,l^\times - l\,s^\times . $$

The first two products are the numerators of $E[s]$ and $E[l]$. Divide by
$\Delta$, sum, and attach the minus sign:

$$ E[s + l] = E[s] + E[l] - \sum_{ijab} \frac{4\,s_{ijab}\,l_{ijab} - s_{ijab}\,l^\times_{ijab} - l_{ijab}\,s^\times_{ijab}}{\Delta_{ij}^{ab}} . \label{eq:mp2-rs-cross-term} $$

The last term, minus sign included, is the cross term $X$, so
$E[s+l] = E[s] + E[l] + X$. It is the correlation in which one of the two
interactions is short-range and the other long-range.

The two exchange products in $X$ are equal after summation. Relabel
$a \leftrightarrow b$ in $\sum s\,l^\times/\Delta$: $\Delta_{ij}^{ab}$ is
symmetric in $a, b$, so the sum becomes $\sum s^\times l/\Delta$. With the
bilinear form

$$ B(g, h) = -\sum_{ijab} \frac{g_{ijab}\,\big(2h_{ijab} - h^\times_{ijab}\big)}{\Delta_{ij}^{ab}} , $$

which is symmetric by the same relabelling, $E[g] = B(g, g)$ and
$X = 2B(s, l)$. The cross term is twice an MP2 energy evaluated with one
short-range and one long-range factor. (Both forms checked numerically on
random tensors, to $10^{-15}$.)

Check the limits. As $\omega \to 0$, $\operatorname{erf}(\omega r) \to 0$,
so $l \to 0$, $X \to 0$ and $E[s] \to E[1/r]$. As $\omega \to \infty$,
$s \to 0$ and again $X \to 0$. Between the two limits $X \ne 0$ in general,
so $E[\operatorname{erfc}] + E[\operatorname{erf}] \ne E[1/r]$.

> **Verified equation @eq:mp2-rs-cross-term** · MP2 is not additive in a split operator
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

## Consequences {#range-split-cross-term-consequences}

Adding a short-range MP2 to a long-range MP2 is therefore not MP2. A method
that wants full MP2 at short range and something else at long range has to
say where $X$ goes. The three constructions in ferric answer differently:

| Construction | What it does with $X$ |
|---|---|
| Attenuated MP2 (section @sec:attenuated-mp2) | drops $E[l]$ and $X$ together; the fitted length absorbs the loss |
| MP2-V (section @sec:mp2-v) | drops $E[l]$ and $X$, then adds a model long-range tail (damped VV10) |
| RS-MP2 + LR-RPA, formulation B (`delta-lr`) | keeps full-Coulomb MP2 and corrects the long range by $E_{\mathrm{dRPA}}[\operatorname{erf}] - 2E_{\mathrm{OS}}[\operatorname{erf}]$, so $X$ is never separated |

$X$ is not small next to $E[l]$. For the S66 dimers at the terfc length
of MP2(terfc, aTZ), $r_0 = 1.35$ Å, Goldey, Belzunces and Head-Gordon
(J. Chem. Theory Comput. 11, 4159 (2015), Fig. 1) report average
correlation binding energies of −0.62 kcal/mol from the long-range term and
−0.54 kcal/mol from the mixed term, against about −4 kcal/mol from the
short-range term.

Why drop $E[l]$ and $X$ together? Both contain the long-range interaction.
The same paper gives the reason: the long-range and mixed terms both carry
the long-range perturbation, which in a finite basis is contaminated with
basis-set superposition error and gives incorrect $C_6$ coefficients, so
MP2-V replaces them together with one long-range model.

## In the code {#range-split-cross-term-code}

`rs_mp2_decomposition` runs three RI-MP2 energies, erfc, erf and Coulomb,
each fitted in its own metric. The function body in full:

```rust
let sr = erfc_attenuated_ri_mp2(mol, obs, dfbs, rhf, config)?.mp2_corr;            // E[erfc]
let lr = attenuated_ri_mp2_long_range(mol, obs, dfbs, rhf, config)?.mp2_corr;     // E[erf]
let op = Operator::coulomb();
let ri_config = RiMp2Config {
    frozen_core: config.frozen_core,
    memory_budget_bytes: config.memory_budget_bytes,
    ..Default::default()
};
let sc = ri_mp2_spin_components_any(mol, obs, dfbs, op, rhf, &ri_config)?;        // E[1/r]
Ok((sr, lr, sc.e_total))
```

Points about it:

- **It returns the three energies, not $X$.** The caller forms
  $E[1/r] - E[\operatorname{erfc}] - E[\operatorname{erf}]$ if it wants the
  residual.
- **Each energy has its own fit.** In RI, (@eq:mp2-rs-cross-term) holds exactly only for
  integrals fitted in one common metric, because the identity needs
  $s + l$ to equal the fitted Coulomb integral. With three fits the
  residual $E[1/r] - E[\operatorname{erfc}] - E[\operatorname{erf}]$ is $X$
  plus the difference of three fitting errors.

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
| `crates/ferric-mp2/theory.md`, "5. Attenuated MP2" ("Range split") | equation @eq:mp2-rs-cross-term |
| `crates/ferric-mp2/src/attenuated.rs` | the decomposition, the test |
| `scripts/mp2_variant_identities.py`, claim 4 | verified box |
| Goldey, Belzunces and Head-Gordon, JCTC 11, 4159 (2015), Fig. 1 and section 2 | the S66 sizes of the long-range and mixed terms; the reason they are replaced together |

Changes from the sources: none; theory.md and (@eq:mp2-rs-cross-term) both use the positive
$\Delta$ with an explicit minus sign.
