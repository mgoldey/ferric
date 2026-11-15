# 5.2 Distance-dependent bounds (QQR)

<!-- section: 5.2; status: edit; registry: none;
     theory: crates/ferric-integrals/theory.md#"Distance-dependent bounds (QQR)" -->

Schwarz cannot tell two distant pairs from two touching ones. QQR multiplies
it by an envelope that falls as $1/R$ with the distance between the pair
centres, which is the leading behaviour of the Coulomb interaction between two
compact charge clouds.

| Entry point | Location |
|---|---|
| four-index | `ferric-scf/src/qqr.rs::QqrBounds` (implements `ferric-scf/src/screening.rs::Bound`) |
| three-index | `ferric-integrals/src/qqr3.rs::QqrBounds3` (`estimate3`) |

<!-- include: crates/ferric-integrals/theory.md#"Distance-dependent bounds (QQR)" -->

> **Verified equation 5.4** · the envelope bounds primitive $s$ pairs
>
> For primitive $s$ pairs, $|(ab|cd)|/(Q_{ab}Q_{cd}) \le \sqrt{\pi/8}\,(\epsilon_{ab} + \epsilon_{cd})/R$,
> so the coded envelope with factor $1.10$ holds; the same envelope over erfc
> Schwarz factors holds on the same sweep.
>
> | | |
> |---|---|
> | Proof | `scripts/screening_bounds_sympy.py`, claims qqr-s and qqr-erfc |
> | Negative control | factor $1.10 \to 0.50$ must fail (recorded mutation) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/qqr.rs::QqrBounds` |
> | Code test | `crates/ferric-scf/tests/qqr_diagonal_noop.rs` |
>
> <!-- verified: id=screening.qqr-s-pairs; script=scripts/screening_bounds_sympy.py; claim=qqr-s; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/qqr.rs::QqrBounds; rust_test=ferric-scf/tests/qqr_diagonal_noop.rs::qqr_equals_schwarz_on_every_diagonal_quartet -->

## 5.2.1 Where QQR is used

| Consumer | Effect |
|---|---|
| LinK pair lists | none: the lists use diagonal estimates, where QQR equals Schwarz |
| per-quartet screens built on `Bound` | the envelope removes distant quartets |
| three-index builds (`build_band_screened`) | QQR-3 removes distant $(P\vert\mu\nu)$ triples; the screen is fixed for the life of a source (section 6.2) |

The RHF driver does not build `QqrBounds` itself; a caller passes it in as the
`Bound`. The envelope never reads $\omega$, so for erfc the attenuation enters
only through the Schwarz factors (section 5.1), and the gain is geometric.

## 5.2.2 Numerical anchors

| What | Test | Bar |
|---|---|---|
| QQR bounds every quartet on water and benzene, Coulomb and erfc | `ferric-scf/src/qqr.rs` unit tests (`test_estimate_is_valid_upper_bound_*`) | bound/true $\ge 1$ |
| QQR equals Schwarz on every diagonal quartet | `crates/ferric-scf/tests/qqr_diagonal_noop.rs` | exact |
| LinK K at a threshold sweep with QQR against the dense K | `crates/ferric-scf/tests/screening_exactness.rs` (`link_k_thresh_sweep_alkane_8_qqr`) | per test |
| QQR-3 on terfc three-index tensors | `crates/ferric-integrals/tests/qqr3_terfc_screening.rs` | per test |

## Sources

- S. A. Maurer, D. S. Lambrecht, D. Flaig and C. Ochsenfeld, *J. Chem. Phys.*
  **136**, 144107 (2012).
