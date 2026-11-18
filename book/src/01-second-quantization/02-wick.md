# Wick contractions as the proof scripts run them {#wick}

<!-- status: edit; registry: -->

Deriving a correlation energy by hand means expanding products of operator
strings, pairing creators with annihilators in every allowed way, and
tracking a sign for each pairing. The result is a handful of tensor
contractions with prefactors such as $\tfrac14$, $\tfrac18$ or 1. A wrong
prefactor still gives a finite, plausible energy, so nothing downstream
flags it. The MP3 ring term in this code base is an example: a single
ordering of its vertex gives $\tfrac14$ of the right value (section @sec:wick-three-prefactors-recipe-settled).

Many derivation proofs in `scripts/` therefore hand the bookkeeping to
sympy. They build the operators of a method with
`sympy.physics.secondquant`, let Wick's theorem produce the amplitude
equation or the energy, and compare the result with an independent route.
This section describes that recipe, so that you can read those scripts,
works the second-order energy through it, and writes down the three
prefactors it has settled.

| Index | Range | Meaning |
|---|---|---|
| $i, j, k, l$ | occupied spin orbitals | below the Fermi level in sympy |
| $a, b, c, d$ | virtual spin orbitals | above the Fermi level in sympy |
| $t_{ij}^{ab}$ | doubles amplitude | `t[a, b, i, j]` in the scripts |

| Entry point | Location |
|---|---|
| Scripts | `scripts/oomp2_mp3_identities_sympy.py` (parts A and B), `scripts/ump3_spin_blocks_sympy.py` |
| Code the results are checked against | `ferric-mp2/src/mp3.rs::mp3_energy`, `ferric-mp2/src/u_mp3.rs::u_mp3` |

## Wick's theorem for an expectation value {#wick-wick-s-theorem-expectation}

For a product of creation and annihilation operators, Wick's theorem writes
the product as a sum of normal-ordered terms with every possible set of
contractions. A contraction of two operators is their expectation value in
$|0\rangle$. The only nonzero contractions relative to $|0\rangle$ are

$$ \langle a^\dagger_i a_j\rangle = \delta_{ij}\ (i, j \text{ occupied}), \qquad \langle a_a a^\dagger_b\rangle = \delta_{ab}\ (a, b \text{ virtual}). $$

The first removes an electron from an occupied orbital and puts it back.
The second adds an electron to an empty orbital and takes it out again.
Every other order gives zero: you cannot remove an electron from an empty
orbital, or add one to a full orbital.

A vacuum expectation value keeps only the fully contracted terms, because
every normal-ordered remainder has zero expectation value (section @sec:hamiltonian-reference-determinant-normal-ordering).
Each contraction sets two orbital labels equal and removes one sum. A sign
comes from the number of operator crossings: draw a line between each
contracted pair, and the term carries $(-1)$ for every crossing of two
lines.

You have already seen the theorem at work. The two expectation values that
gave $E_{\mathrm{ref}}$ in (@eq:so-ref-energy) are the full contractions of
$a^\dagger_p a_q$ and of $a^\dagger_p a^\dagger_q a_s a_r$. In the second,
pairing $p$ with $r$ and $q$ with $s$ draws two nested lines, which do not
cross ($+$), and pairing $p$ with $s$ and $q$ with $r$ draws two lines that
cross ($-$).

## The recipe {#wick-recipe}

| Step | What the script does | Why |
|---|---|---|
| 1 | Build each vertex as a sum of antisymmetrized tensors times a string of operators, for example $\tfrac14 t^{ab}_{ij}\,a^\dagger_a a^\dagger_b a_j a_i$ for the doubles operator $\hat T_2$ | The tensors keep their symmetry, so a result is a sum of tensor products |
| 2 | Give the bra, each vertex and the ket **distinct dummy indices** | Sharing a symbol between two factors merges terms that are different. On the MP3 ring term it produces a spurious factor of $\tfrac14$ |
| 3 | Call `wicks(..., keep_only_fully_contracted=True)` and collect terms with `substitute_dummies` | One canonical representative of each term |
| 4 | Evaluate each term as a sum over a random antisymmetrized model with the same tensor shapes the code uses | The coefficient of a given contraction pattern is then a number |
| 5 | Compare with a route that uses no Wick algebra: finite differences of an energy, or Rayleigh–Schrödinger perturbation theory in the full determinant space of a few spin orbitals | A bug in step 1 to 3 shows as a mismatch |

Here are steps 1 to 3 as `scripts/oomp2_mp3_identities_sympy.py` writes
them for the MP3 hole–hole ladder, in `part_a`, with the particle–particle
vertex and the dummy-index declarations removed:

```python
    q = Rational(1, 4)
    t_bra = q * AntiSymmetricTensor("t", (a, b), (i, j)) * Fd(i) * Fd(j) * F(b) * F(a)
    t_ket = (                                 # same operator, fresh labels (step 2)
        q
        * AntiSymmetricTensor("t", (a2, b2), (i2, j2))
        * Fd(a2)
        * Fd(b2)
        * F(j2)
        * F(i2)
    )
    v_hh = (                                  # hole-hole block of V_N, labels o1..o4
        q
        * AntiSymmetricTensor("g", (o1, o2), (o3, o4))
        * NO(Fd(o1) * Fd(o2) * F(o4) * F(o3))
    )
    hh = wick_expect(t_bra * v_hh * t_ket)  # step 3
```

`wick_expect` is the script's three-line wrapper:
`wicks(op, keep_only_fully_contracted=True)`, then `evaluate_deltas` to
remove the summed Kronecker deltas, then `substitute_dummies`.

`AntiSymmetricTensor` is sympy's tensor that changes sign under a swap of
its two upper or two lower indices, the symmetry of
$\langle pq\|rs\rangle$ and $t^{ab}_{ij}$. `NO(...)` marks a normal-ordered
string, so contractions inside the vertex are excluded, as in (@eq:so-normal-order-split). The
bra is $\hat T_2^\dagger$: the same tensor with the operator string
reversed and daggered.

The independent route in step 5 is what makes the proof more than a
transcription. For MP3 the script runs second- and third-order
perturbation theory in the 35 determinants of 7 spin orbitals and 3
electrons, and compares with the formulas the code contracts. That route
uses only the Jordan–Wigner matrices of the creators and annihilators and
the textbook perturbation formulas, so an error in steps 1 to 3 cannot
repeat itself there.

## The second-order energy {#wick-second-order-energy}

The first-order doubles amplitudes of canonical Hartree–Fock orbitals are

$$ t_{ij}^{ab} = -\,\frac{\langle ij\|ab\rangle}{\Delta_{ij}^{ab}}, \qquad \hat T_2 = \frac14\sum_{ijab} t_{ij}^{ab}\, a^\dagger_a a^\dagger_b a_j a_i , \label{eq:wick-1} $$

and the second-order energy is the contraction of $\hat V_{\mathrm N}$ with
$\hat T_2$:

$$ E^{(2)} = \langle 0|\hat V_{\mathrm N}\hat T_2|0\rangle = \frac14\sum_{ijab}\langle ij\|ab\rangle\,t_{ij}^{ab} = -\frac14\sum_{ijab}\frac{|\langle ij\|ab\rangle|^2}{\Delta_{ij}^{ab}} . \label{eq:so-e2-wick} $$

Physically, $\hat T_2$ lifts two electrons from $i, j$ into $a, b$, and
$\hat V_{\mathrm N}$ drops them back. The amplitude is how much of that
double excitation the perturbation mixes in: the coupling divided by the
energy cost $\Delta_{ij}^{ab} = \varepsilon_a + \varepsilon_b -
\varepsilon_i - \varepsilon_j$.

### Where the $\tfrac14$ comes from

Only one block of $\hat V_{\mathrm N}$ can give a nonzero result: the one
that returns two electrons from virtuals to holes,
$\tfrac14\sum\langle kl\|cd\rangle\{a^\dagger_k a^\dagger_l a_d a_c\}$.
Fully contract it with $\hat T_2$. The holes $k, l$ pair with $i, j$ in two
ways, and the particles $c, d$ pair with $a, b$ in two ways, so there are
four full contractions. In each, a swap of two labels relative to the first
pairing costs one crossing sign and one antisymmetry sign of a tensor, and
the two cancel. All four equal
$\tfrac14\cdot\tfrac14\sum\langle ij\|ab\rangle t^{ab}_{ij}$, and together
they give the $\tfrac14$ of (@eq:so-e2-wick). `wicks` returns exactly this one term.

Check one pair. With one occupied pair and one virtual pair the sum
over $ijab$ has four terms, $(ij,ab)$, $(ji,ab)$, $(ij,ba)$ and $(ji,ba)$.
They are equal by antisymmetry, so the $\tfrac14$ cancels and
$E^{(2)} = -|\langle ij\|ab\rangle|^2/\Delta$: second-order perturbation
theory for two levels coupled by $\langle ij\|ab\rangle$ and separated by
$\Delta$.

### Why the amplitudes are (@eq:wick-1)

The scripts do not assume (@eq:wick-1). They build the Hylleraas functional, also
by Wick contraction,

$$ J(t) = \langle 0|\hat T_2^\dagger\hat V_{\mathrm N}|0\rangle + \langle 0|\hat V_{\mathrm N}\hat T_2|0\rangle + \langle 0|\hat T_2^\dagger\hat F_{\mathrm N}\hat T_2|0\rangle , $$

and check that it is stationary at the amplitudes the code uses. In
canonical orbitals the three terms evaluate to

$$ J(t) = \frac12\sum_{ijab}\langle ij\|ab\rangle\,t_{ij}^{ab} + \frac14\sum_{ijab}\Delta_{ij}^{ab}\,\big(t_{ij}^{ab}\big)^2 . $$

The first term is the two equal $\hat V_{\mathrm N}$ contractions of (@eq:so-e2-wick)
added together. The second is the Fock operator measuring the excitation
energy of each doubly excited determinant. Each unique amplitude
($i<j$, $a<b$) appears four times in each sum, so

$$ \frac{\partial J}{\partial t_{ij}^{ab}} = 2\langle ij\|ab\rangle + 2\Delta_{ij}^{ab}\,t_{ij}^{ab} = 0 \quad\Longrightarrow\quad t_{ij}^{ab} = -\frac{\langle ij\|ab\rangle}{\Delta_{ij}^{ab}} . $$

Substitute back:
$J = \tfrac12\sum\langle ij\|ab\rangle t + \tfrac14\sum\Delta t^2 = -\tfrac12\sum|\langle ij\|ab\rangle|^2/\Delta + \tfrac14\sum|\langle ij\|ab\rangle|^2/\Delta$,
which is $E^{(2)}$ of (@eq:so-e2-wick). The stationary point is a minimum, because
$\Delta > 0$ makes the quadratic term positive. Chapter @sec:local-mp2 uses the same
functional when the Fock matrix is not diagonal.

### Signs in the code

The scripts write the amplitude with the code's negative denominator,
$t = \langle ab\|ij\rangle/D$ and $D = -\Delta$. The two forms are the same
number, because for real orbitals $\langle ab\|ij\rangle = \langle
ij\|ab\rangle$. The amplitude and energy are three lines of
`ferric-mp2/src/mp3.rs::mp3_energy_from_spin_orbital_blocks`, the
spin-orbital back end of `mp3_energy`, shown here with the memory-management
comments removed:

```rust
let d = eo[i] + eo[j] - ev[a] - ev[b];          // D = -Delta < 0
t[[i, j, a, b]] = asym_oovv_t[[i, j, a, b]] / d; // t = <ij||ab> / D, Eq. (1.7)
// ...
let v = 0.25 * einsum!("ijab,ijab->", &t_t, &oovv_t);  // E2 = 1/4 sum t <ij||ab>, Eq. (1.8)
```

There is no leading minus anywhere, because the sign sits in `d`. A port
that writes $\Delta$ with the textbook sign has to add the minus sign to
both lines.

> **Verified equation @eq:so-e2-wick** · the second-order energy from Wick
>
> The Hylleraas functional built by Wick contraction is stationary at the
> amplitudes of (@eq:wick-1), and its value there is $\tfrac14\sum\langle ij\|ab\rangle t$. An
> independent Rayleigh–Schrödinger calculation in the determinant space gives
> the same second-order energy.
>
> | | |
> |---|---|
> | Proof | `scripts/oomp2_mp3_identities_sympy.py`, the checks "B1 J(t*) == E2" and the one labelled "claim A3" (Rayleigh–Schrödinger $E_2$) |
> | Negative control | none specific to this identity; the mutations change the MP3 ring coefficient, the sign of $\kappa$ and the two-particle density |
> | Runner | `crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py` |
> | Code | `ferric-mp2/src/mp3.rs::mp3_energy` |
>
> <!-- verified: id=so.e2-wick; script=scripts/oomp2_mp3_identities_sympy.py; claim=A3; runner=crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py; code=ferric-mp2/src/mp3.rs::mp3_energy -->

The script asserts stationarity and the value at the stationary point.
It does not assert the explicit canonical form of $J(t)$ written above.

The closed-shell form of (@eq:so-e2-wick), the one RI-MP2 evaluates, follows in
section @sec:spin-adaptation.

## Three prefactors the recipe settled {#wick-three-prefactors-recipe-settled}

The third-order energy $E^{(3)} = \langle \hat T_2^\dagger \hat V_{\mathrm N}\hat T_2\rangle$
has three connected parts. In each, $\hat T_2$ excites a pair,
$\hat V_{\mathrm N}$ scatters it once, and $\hat T_2^\dagger$ returns it.
The particle–particle ladder scatters the two particles, the hole–hole
ladder the two holes, and the ring scatters one hole and one particle. The
script finds, on the full unrestricted sums over spin orbitals:

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

The ladders count the same way as (@eq:so-e2-wick). The bra, the vertex and the ket
each carry $\tfrac14$, so $\tfrac1{64}$ in all. For the particle–particle
ladder, the two holes of $\hat T_2$ pair with the two holes of
$\hat T_2^\dagger$ in 2 ways, its two particles pair with the vertex
annihilators in 2 ways, and the vertex creators pair with the particles of
$\hat T_2^\dagger$ in 2 ways. That makes $2^3 = 8$ equal contractions and
a coefficient of $8/64 = \tfrac18$. The hole–hole ladder is the same with
holes and particles exchanged.

The code applies the three coefficients as literals. With the scoping
blocks and tensor wrapping removed:

```rust
let x: ArrayD<f64> = einsum!("ijab,abcd->ijcd", &t_t, &vvvv_t);  // sum_ab t_ij^ab <ab||cd>
0.125 * einsum!("ijcd,ijcd->", &x_t, &t_t)                         // e_pp: 1/8
// ...
let y: ArrayD<f64> = einsum!("klij,ijab->klab", &oooo_t, &t_t);  // sum_ij <kl||ij> t_ij^ab
0.125 * einsum!("klab,klab->", &y_t, &t_t)                         // e_hh: 1/8
// ...
let z: ArrayD<f64> = einsum!("ijab,kbcj->iakc", &t_t, &ovvo_t);  // sum_jb t_ij^ab <kb||cj>
einsum!("iakc,iakc->", &z_t, &t_iakc_t)                            // e_ph: coefficient 1
```

Each term is two contractions: a vertex applied to one amplitude tensor,
then a full contraction with the other. The first step of the
particle–particle ladder costs $O(n_o^2 n_v^4)$ in spin orbitals; the ring's
costs $O(n_o^3 n_v^3)$. `t_iakc_t` is the amplitude tensor
with its middle two axes permuted, so that the ring's final contraction is
a plain elementwise sum.

> **Verified equation {#eq:so-mp3-prefactors}** · MP3 prefactors
>
> | | |
> |---|---|
> | Proof | `scripts/oomp2_mp3_identities_sympy.py`, the check labelled "claim A2" (four orderings give coefficient 1; one ordering gives $\tfrac14$) |
> | Negative control | the script's `ring_quarter` mutation sets the coefficient to $\tfrac14$; the script must turn red, and the runner asserts it does |
> | Runner | `crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py` |
> | Code | `ferric-mp2/src/mp3.rs::mp3_energy` |
>
> <!-- verified: id=so.mp3-prefactors; script=scripts/oomp2_mp3_identities_sympy.py; claim=A2; runner=crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py; code=ferric-mp2/src/mp3.rs::mp3_energy -->

The unrestricted code works in spin blocks, not in spin orbitals. Its
equivalence to the spin-orbital form is the subject of
`scripts/ump3_spin_blocks_sympy.py`, which treats every spatial integral as an
independent symbol, and chapter @sec:mp3 covers it.

## Known limits

- The recipe proves an identity between expressions. It does not show that the
  Rust loop contracts the expression. That is the job of the Rust tests and
  the numerical anchors named in each chapter.
- Wick's theorem as used here assumes one reference determinant with
  integer occupations. Multireference methods (chapter @sec:multireference) use density
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
