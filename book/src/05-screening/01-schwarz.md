# 5.1 Schwarz bounds, and why they do not see attenuation

<!-- section: 5.1; status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"The Schwarz bound", "The combined Schwarz bound (CSB)" -->

The Schwarz inequality bounds a four-center integral by the product of two
diagonal integrals. It is the screen behind direct J and K, LinK and the
three-index tensors. This section states when it holds, which kernels ferric
admits, how the table is built so that it never undershoots, and why it
screens erfc and terfc integrals almost exactly as it screens Coulomb ones.

| Entry point | Location |
|---|---|
| Rust, tables | `ferric-integrals/src/schwarz.rs::{schwarz,schwarz3_aux,bound_inflation,check_operator_supported}` |
| Rust, the bound trait | `ferric-scf/src/screening.rs::{Bound,SchwarzBounds,CsbBounds}` |
| Rust, CSB | `ferric-integrals/src/csb.rs::csb_m_table` |
| terfc validity box | `ferric-integrals/src/schwarz.rs::check_terfc_parameters_validated` |

## 5.1.1 The bound and the admitted kernels

<!-- include: crates/ferric-integrals/theory.md#"The Schwarz bound" -->

> **Verified equation 5.1** · terf is not positive definite
>
> $\hat K_{\mathrm{terf}}$ is negative on $kr_0 \in (\pi/2, 3\pi/2)$, with its first
> zero at $kr_0 = \pi/2$. On $s$ Gaussians on one centre the terf Gram matrix has
> a negative eigenvalue, while erfc and terfc stay positive on the same set,
> and a terf integral exceeds its "Schwarz bound" outright.
>
> | | |
> |---|---|
> | Proof | `scripts/screening_bounds_sympy.py`, claims sign and indefinite |
> | Negative control | dropping the $e^{-\phi^2r_0^2}$ factor of the same-centre terf integral must fail (recorded mutation) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-integrals/src/schwarz.rs::check_operator_supported` |
> | Code test | `crates/ferric-integrals/tests/operator_positivity_gate.rs` |
>
> <!-- verified: id=screening.terf-indefinite; script=scripts/screening_bounds_sympy.py; claim=indefinite; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-integrals/src/schwarz.rs::check_operator_supported; rust_test=ferric-integrals/tests/operator_positivity_gate.rs::positive_definite_operators_have_nonnegative_diagonals -->

> **Verified equation 5.2** · Schwarz holds for erf, erfc and terfc, and does not see attenuation
>
> On random $s$ quartets $|(ab|cd)| \le Q_{ab}Q_{cd}$ for erf, erfc and terfc.
> $Q^2_{\mathrm{erfc}}/Q^2_{\mathrm{Coulomb}} = 1 - \phi/\theta \to 1$ for compact
> pairs, and the bound is independent of the distance between the pairs while
> the true erfc integral decays.
>
> | | |
> |---|---|
> | Proof | `scripts/screening_bounds_sympy.py`, claims schwarz and blind |
> | Negative control | $Q$ without the square root must fail (recorded mutation) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-integrals/src/schwarz.rs::schwarz` |
> | Code test | `crates/ferric-integrals/tests/terfc_schwarz_screening.rs` (terfc never undershoots, with $d$ shells, diffuse shells and quartets; mutants caught) |
>
> <!-- verified: id=screening.schwarz-blind; script=scripts/screening_bounds_sympy.py; claim=blind; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-integrals/src/schwarz.rs::schwarz; rust_test=ferric-integrals/tests/terfc_schwarz_screening.rs::terfc_schwarz_never_undershoots_quartets -->

![Two s pair densities: the true Coulomb and erfc integrals against their Schwarz and QQR bounds as a function of distance. The erfc Schwarz bound is flat while the integral falls by many orders. Script: book/src/figures/05-screening/schwarz_vs_true.py](../figures/05-screening/schwarz_vs_true.svg)

## 5.1.2 Building a table that never undershoots

| Choice | Value | Why |
|---|---|---|
| engine precision for the diagonals | $0$ (`SCHWARZ_TABLE_PRECISION`) | libint2's own skip would return no block for a small diagonal |
| floor on every factor | $10^{-100}$ (`SCHWARZ_Q_FLOOR`) | a zero factor would make a threshold-$0$ screen drop pairs |
| terfc factors | $\times(1 + 10^{-5})$ (`bound_inflation`) | the table engine is interpolated; the margin is empirical |
| terfc parameters | $r_0 \in [0.3, 2.5]$ Å, $0 < r_0\omega \le 8.9$, $l \le 4$ | the box the margin was measured in; outside it the call is refused |
| shell-pair factor | the largest $\lvert(\mu\nu\vert\mu\nu)\rvert^{1/2}$ over the pair's functions | an element-wise bound for every function in the block |

The trivial limit is threshold $0$: every pair is kept, and a screened build
equals the unscreened one (`crates/ferric-scf/tests/screening_exactness.rs`,
`crates/ferric-scf/tests/eri_precision_matches_unscreened.rs`).

## 5.1.3 The combined Schwarz bound

<!-- include: crates/ferric-integrals/theory.md#"The combined Schwarz bound (CSB)" -->

> **Verified equation 5.3** · the $MM$ terms need only $w \ge 0$
>
> $\operatorname{terf}(r) \ge 0$ for $r \ge 0$, and
> $|(ab|cd)| \le \min(M_{ac}M_{bd}, M_{ad}M_{bc})$ holds on random terf and terfc
> $s$ quartets, although terf violates Schwarz.
>
> | | |
> |---|---|
> | Proof | `scripts/screening_bounds_sympy.py`, claim csb |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-integrals/src/csb.rs::csb_m_table` |
> | Code test | `crates/ferric-scf/tests/csb_screening.rs` (Coulomb and erfc: valid, never looser than Schwarz, trivial limit) |
>
> <!-- verified: id=screening.csb-pointwise; script=scripts/screening_bounds_sympy.py; claim=csb; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-integrals/src/csb.rs::csb_m_table; rust_test=ferric-scf/tests/csb_screening.rs::csb_is_a_valid_upper_bound -->

The code refuses CSB for terf with the reason that the $M$ terms "come from
the same Cauchy–Schwarz argument as Schwarz" and so need positive
definiteness. The refusal is right, because the full minimum contains the
$QQ$ term; the stated reason is stronger than the derivation needs. An
$MM$-only bound would be valid for terf.

## 5.1.4 Numerical anchors

| What | Test | Bar |
|---|---|---|
| terfc Schwarz never undershoots a true $(P\vert ab)$ or quartet over the validated box (methane, water; cc-pVDZ to def2-QZVP) | `crates/ferric-integrals/tests/terfc_schwarz_screening.rs` | bound/true $\ge 1$ |
| a halved bound and the Coulomb-diagonal substitute are caught | same file, `mutants_are_caught_by_the_validity_check` | must fail |
| terfc quartet bound on water / cc-pVDZ | `crates/ferric-integrals/tests/eri4_terfc_schwarz_validity.rs` | bound/true $\ge 1$ |
| positive-definite kernels have nonnegative diagonals; terf does not | `crates/ferric-integrals/tests/operator_positivity_gate.rs` | sign |
| CSB valid, never looser than Schwarz, equal to it in the trivial limit, RHF energy unchanged | `crates/ferric-scf/tests/csb_screening.rs` | per test |

## Sources

- M. Häser and R. Ahlrichs, *J. Comput. Chem.* **10**, 104 (1989).
- T. H. Thompson and C. Ochsenfeld, *J. Chem. Phys.* **147**, 144101 (2017).
- S. Bochner, *Math. Ann.* **108**, 378 (1933).
