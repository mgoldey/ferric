# 1.2 Wick contractions as the proof scripts run them

<!-- section: 1.2; status: edit; registry: ;
     theory: -->

Many derivation proofs in `scripts/` have the same shape: build the operators
of a method with `sympy.physics.secondquant`, let Wick's theorem produce the
amplitude equation or the energy, and compare the result with an independent
route. This section describes that recipe, so that a reader can read those
scripts, and writes down the three prefactors that it has settled.

| Index | Range | Meaning |
|---|---|---|
| $i, j, k, l$ | occupied spin orbitals | below the Fermi level in sympy |
| $a, b, c, d$ | virtual spin orbitals | above the Fermi level in sympy |
| $t_{ij}^{ab}$ | doubles amplitude | `t[a, b, i, j]` in the scripts |

| Entry point | Location |
|---|---|
| Scripts | `scripts/oomp2_mp3_identities_sympy.py` (parts A and B), `scripts/ump3_spin_blocks_sympy.py` |
| Code the results are checked against | `ferric-mp2/src/mp3.rs::mp3_energy`, `ferric-mp2/src/u_mp3.rs::u_mp3` |

## 1.2.1 Wick's theorem for an expectation value

For a product of creation and annihilation operators, Wick's theorem writes
the product as a sum of normal-ordered terms with every possible set of
contractions. The only nonzero contractions relative to $|0\rangle$ are

$$ \langle a^\dagger_i a_j\rangle = \delta_{ij}\ (i, j \text{ occupied}), \qquad \langle a_a a^\dagger_b\rangle = \delta_{ab}\ (a, b \text{ virtual}). $$

A vacuum expectation value keeps only the fully contracted terms, because
every normal-ordered remainder has zero expectation value. Each contraction
sets two orbital labels equal and removes one sum. A sign comes from the
number of operator crossings.

## 1.2.2 The recipe

| Step | What the script does | Why |
|---|---|---|
| 1 | Build each vertex as a sum of antisymmetrized tensors times a string of operators, for example $\tfrac14 t^{ab}_{ij}\,a^\dagger_a a^\dagger_b a_j a_i$ for the doubles operator $\hat T_2$ | The tensors keep their symmetry, so a result is a sum of tensor products |
| 2 | Give the bra, each vertex and the ket **distinct dummy indices** | Sharing a symbol between two factors merges terms that are different. It produced a spurious factor of $\tfrac14$ on the MP3 ring term in an earlier wiki derivation |
| 3 | Call `wicks(..., keep_only_fully_contracted=True)` and collect terms with `substitute_dummies` | One canonical representative of each term |
| 4 | Evaluate each term as a sum over a random antisymmetrized model with the same tensor shapes the code uses | The coefficient of a given contraction pattern is then a number |
| 5 | Compare with a route that uses no Wick algebra: finite differences of an energy, or Rayleigh–Schrödinger perturbation theory in the full determinant space of a few spin orbitals | A bug in step 1 to 3 shows as a mismatch |

The independent route in step 5 is what makes the proof more than a
transcription. For MP3 the script runs second- and third-order
perturbation theory in the 35 determinants of 7 spin orbitals and 3
electrons, and compares with the formulas the code contracts.

## 1.2.3 The second-order energy

The first-order doubles amplitudes of canonical Hartree–Fock orbitals are

$$ t_{ij}^{ab} = -\,\frac{\langle ij\|ab\rangle}{\Delta_{ij}^{ab}}, \qquad \hat T_2 = \frac14\sum_{ijab} t_{ij}^{ab}\, a^\dagger_a a^\dagger_b a_j a_i , \tag{1.7} $$

and the second-order energy is the contraction of $\hat V_{\mathrm N}$ with
$\hat T_2$:

$$ E^{(2)} = \langle 0|\hat V_{\mathrm N}\hat T_2|0\rangle = \frac14\sum_{ijab}\langle ij\|ab\rangle\,t_{ij}^{ab} = -\frac14\sum_{ijab}\frac{|\langle ij\|ab\rangle|^2}{\Delta_{ij}^{ab}} . \tag{1.8} $$

The scripts write the amplitude with the code's negative denominator,
$t = \langle ab\|ij\rangle/D$ and $D = -\Delta$. The two forms are the same
number.

> **Verified equation 1.8** · the second-order energy from Wick
>
> The Hylleraas functional built by Wick contraction is stationary at the
> amplitudes of (1.7), and its value there is $\tfrac14\sum\langle ij\|ab\rangle t$. An
> independent Rayleigh–Schrödinger calculation in the determinant space gives
> the same second-order energy.
>
> | | |
> |---|---|
> | Proof | `scripts/oomp2_mp3_identities_sympy.py`, the checks "B1 J(t*) == E2" and "A3 RS E2" |
> | Negative control | none specific to this identity; the mutations change the MP3 ring coefficient, the sign of $\kappa$ and the two-particle density |
> | Runner | `crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py` |
> | Code | `ferric-mp2/src/mp3.rs::mp3_energy` |
>
> <!-- verified: id=so.e2-wick; script=scripts/oomp2_mp3_identities_sympy.py; claim=A3 RS E2 == 1/4 <ij||ab> t (singles vanish by the diagonal Fock operator); runner=crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py; code=ferric-mp2/src/mp3.rs::mp3_energy -->

The closed-shell form of (1.8), the one RI-MP2 evaluates, follows in
section 1.3.

## 1.2.4 Three prefactors the recipe settled

The third-order energy $E^{(3)} = \langle \hat T_2^\dagger \hat V_{\mathrm N}\hat T_2\rangle$
has three connected parts. The script finds, on the full unrestricted
sums over spin orbitals:

| Part | Coefficient on the full sum | Where it matters |
|---|---|---|
| particle–particle ladder | $\tfrac18$ | `e_pp` in `ferric-mp2/src/mp3.rs::mp3_energy` |
| hole–hole ladder | $\tfrac18$ | `e_hh` in the same function |
| particle–hole ring | exactly $1$ | the ring contraction in the same function |

The ring coefficient is $1$ and not $\tfrac14$ because the ring vertex has
four orderings of one hole and one particle in the creators and in the
annihilators. They are equal by antisymmetry, and each carries $\tfrac14$,
so the vertex is $4\cdot\tfrac14 = 1$. A derivation that keeps a single
ordering gets $\tfrac14$ of the right value. The script keeps that variant
as a negative control and asserts that it differs.

> **Verified equation 1.9** · MP3 prefactors
>
> | | |
> |---|---|
> | Proof | `scripts/oomp2_mp3_identities_sympy.py`, the "A2 ring" check (four orderings give coefficient 1; one ordering gives $\tfrac14$) |
> | Negative control | the script's `ring_quarter` mutation sets the coefficient to $\tfrac14$; the script must turn red, and the runner asserts it does |
> | Runner | `crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py` |
> | Code | `ferric-mp2/src/mp3.rs::mp3_energy` |
>
> <!-- verified: id=so.mp3-prefactors; script=scripts/oomp2_mp3_identities_sympy.py; claim=A2 ring: wicks (four orderings) == coefficient 1 on the mp3.rs contraction; runner=crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py; code=ferric-mp2/src/mp3.rs::mp3_energy -->

The unrestricted code works in spin blocks, not in spin orbitals. Its
equivalence to the spin-orbital form is the subject of
`scripts/ump3_spin_blocks_sympy.py`, which treats every spatial integral as an
independent symbol, and chapter 10.9 covers it.

## Known limits

- The recipe proves an identity between expressions. It does not show that the
  Rust loop contracts the expression. That is the job of the Rust tests and
  the numerical anchors named in each chapter.
- Wick's theorem as used here assumes one reference determinant with
  integer occupations. Multireference methods (chapter 17) use density
  matrices instead.

## Sources

| Source | Used for |
|---|---|
| `scripts/oomp2_mp3_identities_sympy.py`, header | the recipe, the three prefactors and the negative controls |
| `crates/ferric-mp2/src/mp3.rs` | the contraction the prefactors are checked against |
| Dissertation, chapter 1, section 1.1.3, Eqs. 1.19–1.23 | the perturbation expansion (the text extraction of Eq. 1.23 garbles a denominator sign; the book uses $\Delta > 0$) |

Changes from the sources: amplitudes are written with the positive
denominator $\Delta$ and an explicit minus sign. The script keeps the code's
negative $D$.
