# Hamiltonian, spin orbitals and normal ordering {#hamiltonian}

<!-- status: edit; registry: -->

Electrons are identical fermions. A many-electron wavefunction has to change
sign when you swap any two of them, and a Slater determinant is the simplest
function that does. Writing determinants and integrals out in first
quantization means carrying that antisymmetry by hand through every
derivation: $N!$ terms per determinant, and a sign to track in each.

Second quantization moves the antisymmetry into the operators. A creation
operator $a^\dagger_p$ puts an electron in spin orbital $p$, an annihilation
operator $a_p$ removes one, and the anticommutation rule
$a^\dagger_p a^\dagger_q = -a^\dagger_q a^\dagger_p$ carries the Pauli
principle. A determinant becomes a list of occupied orbitals, which in the
code is a bit string. The Hamiltonian becomes one operator that does not
depend on how many electrons there are.

Every method in this book starts from that operator, the electronic
Hamiltonian in a finite orbital basis. The plan of this section:

1. Write the Hamiltonian in second quantization, once with spatial orbitals
   and once with spin orbitals, and show the two are the same operator.
2. Take its expectation value in a single determinant.
3. Split it, relative to that determinant, into a constant, a Fock operator
   and a fluctuation potential.

Section @sec:wick shows how the proof scripts use that split. Section @sec:spin-adaptation reduces
spin orbitals to spatial orbitals.

| Index | Range in this chapter | Code name |
|---|---|---|
| $p, q, r, s$ | spin orbitals ($2n$ of them) in a spin-orbital expression | `0..2*n`, interleaved |
| $i, j, k, l$ | occupied spin orbitals | `0..2*no` |
| $a, b, c, d$ | virtual spin orbitals | `0..2*nv` |
| $[p]$ | the spatial orbital of spin orbital $p$ | `spat(p)` = `p >> 1` |
| $\sigma(p)$ | the spin of spin orbital $p$, $\alpha$ or $\beta$ | `spin(p)` = `p & 1` |

In a spin-orbital expression the letters above run over spin orbitals. In a
spatial expression, from section @sec:spin-adaptation on, the same letters run over spatial
orbitals, and the section says so. Where both appear in one equation, a spin
orbital is written $p_\sigma$.

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/spinorbital.rs::{spin,spat,asym_oovv,asym_same,asym_phys}` |
| CLI, Python | none; this section is background for every chapter |

## The electronic Hamiltonian {#hamiltonian-electronic-hamiltonian}

The nuclei are fixed (the Born–Oppenheimer approximation). The electrons
move in the field of point charges $Z_A$ at positions $\mathbf R_A$, in
atomic units. The dissertation writes the electronic Hamiltonian as the sum
of three terms (chapter 1, Eqs. 1.3–1.5): the kinetic energy of the
electrons, their attraction to the nuclei, and their repulsion from each
other. The nuclear repulsion
$E_{\mathrm{nn}} = \sum_{A<B} Z_A Z_B / |\mathbf R_A - \mathbf R_B|$ does not
depend on the electrons, so it is a constant added at the end.

The first two terms act on one electron at a time. Together they are the
one-electron operator
$\hat h = -\tfrac12\nabla^2 - \sum_A Z_A/|\mathbf r - \mathbf R_A|$. In the
code the same name covers more: `hcore` also holds an effective core
potential, an external point-charge potential or a field when one is set
(chapters @sec:integrals and @sec:embedding). None of the algebra below cares what is inside
$\hat h$, only that it is a one-electron operator.

Orbitals are real and orthonormal. A spin orbital is a spatial orbital times
a spin function. The spatial orbitals $\phi_p$ are molecular orbitals,
expanded in the AO basis as $\phi_p = \sum_\mu C_{\mu p}\chi_\mu$. Their
integrals are

$$ h_{pq} = \int \phi_p\,\hat h\,\phi_q\,d\mathbf r, \qquad (pq|rs) = \iint \phi_p(\mathbf r_1)\phi_q(\mathbf r_1)\,\frac{1}{r_{12}}\,\phi_r(\mathbf r_2)\phi_s(\mathbf r_2)\,d\mathbf r_1 d\mathbf r_2 . $$

Read $(pq|rs)$ as the Coulomb repulsion between two charge distributions,
$\phi_p\phi_q$ for electron 1 and $\phi_r\phi_s$ for electron 2. With real
orbitals each distribution is symmetric in its two labels, and the Coulomb
interaction is symmetric between the two electrons. That gives the
symmetry the notation chapter fixes:
$(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)$.

## Second quantization {#hamiltonian-second-quantization}

With $a^\dagger_{p\sigma}$ creating an electron of spin $\sigma$ in spatial
orbital $p$, the Hamiltonian is

$$ \hat H = E_{\mathrm{nn}} + \sum_{pq}\sum_{\sigma} h_{pq}\,a^\dagger_{p\sigma} a_{q\sigma} + \frac12 \sum_{pqrs}\sum_{\sigma\tau} (pq|rs)\,a^\dagger_{p\sigma} a^\dagger_{r\tau} a_{s\tau} a_{q\sigma} . \label{eq:hamiltonian-1} $$

Read the two-electron term from right to left. $a_{q\sigma}$ removes electron
1 from $q$, $a_{s\tau}$ removes electron 2 from $s$, and the creators put
them back in $r$ and $p$. Each electron keeps its spin, because the Coulomb
operator does not act on spin. The $\tfrac12$ corrects for counting each
pair of electrons twice.

The CAS-CI code builds this form, and its density matrices follow
the same pairing of spin labels: $\gamma_{pq} = \langle E_{pq}\rangle$ and
$\Gamma_{pqrs} = \sum_{\sigma\tau}\langle a^\dagger_{p\sigma} a^\dagger_{r\tau}
a_{s\tau} a_{q\sigma}\rangle$ (section @sec:rdms-1-2-rdm).

### From spatial to spin-orbital form

MP3 and coupled cluster are derived in spin orbitals, so the same operator
is also written with antisymmetrized integrals:

$$ \hat H = E_{\mathrm{nn}} + \sum_{pq} h_{pq}\,a^\dagger_p a_q + \frac14 \sum_{pqrs} \langle pq\|rs\rangle\, a^\dagger_p a^\dagger_q a_s a_r , \label{eq:so-operator-forms} $$

$$ h_{pq} = h_{[p][q]}\,\delta_{\sigma(p)\sigma(q)}, \qquad \langle pq\|rs\rangle = ([p][r]|[q][s])\,\delta_{\sigma(p)\sigma(r)}\delta_{\sigma(q)\sigma(s)} - ([p][s]|[q][r])\,\delta_{\sigma(p)\sigma(s)}\delta_{\sigma(q)\sigma(r)} . \label{eq:hamiltonian-3} $$

Here is how (@eq:hamiltonian-1) becomes (@eq:so-operator-forms). The one-electron term only needs the
spin label folded into the orbital index: $h_{pq}$ is zero between
different spins, which is the delta in (@eq:hamiltonian-3).

For the two-electron term, rename the labels of (@eq:hamiltonian-1) as spin orbitals.
Call the first creator $P = (p,\sigma)$, the second $Q = (r,\tau)$, and
the annihilators $R = (q,\sigma)$ and $S = (s,\tau)$. The operator string
is then $a^\dagger_P a^\dagger_Q a_S a_R$, and the integral is the direct
(physicists') integral

$$ \langle PQ|RS\rangle = ([P][R]|[Q][S])\,\delta_{\sigma(P)\sigma(R)}\,\delta_{\sigma(Q)\sigma(S)} , $$

so the two-electron term of (@eq:hamiltonian-1) is
$\tfrac12\sum_{PQRS}\langle PQ|RS\rangle\,a^\dagger_P a^\dagger_Q a_S a_R$.
The deltas carry the spin conservation that (@eq:hamiltonian-1) wrote as shared labels
$\sigma$ and $\tau$.

Now use the anticommutation of the two annihilators,
$a_R a_S = -a_S a_R$. Swap the dummy labels $R \leftrightarrow S$ in the
sum, then anticommute:

$$ \frac12\sum_{PQRS}\langle PQ|RS\rangle\,a^\dagger_P a^\dagger_Q a_S a_R = \frac12\sum_{PQRS}\langle PQ|SR\rangle\,a^\dagger_P a^\dagger_Q a_R a_S = -\frac12\sum_{PQRS}\langle PQ|SR\rangle\,a^\dagger_P a^\dagger_Q a_S a_R . $$

The left side equals the right side, so it also equals their average:

$$ \frac14\sum_{PQRS}\big(\langle PQ|RS\rangle - \langle PQ|SR\rangle\big)\,a^\dagger_P a^\dagger_Q a_S a_R = \frac14\sum_{PQRS}\langle PQ\|RS\rangle\,a^\dagger_P a^\dagger_Q a_S a_R . $$

The exchange integral $\langle PQ|SR\rangle = ([P][S]|[Q][R])\,\delta_{\sigma(P)\sigma(S)}\delta_{\sigma(Q)\sigma(R)}$
is the second term of (@eq:hamiltonian-3). Nothing was approximated: the antisymmetrized
integral holds the same information as the direct one. The $\tfrac14$ is the
$\tfrac12$ of (@eq:hamiltonian-1) times the $\tfrac12$ of the average.

What do you gain? $\langle pq\|rs\rangle$ changes sign under $p \leftrightarrow q$
and under $r \leftrightarrow s$, the same symmetry as the operator string.
Contractions in section @sec:wick can then treat every ordering of a pair as
the same term, which is what keeps the prefactors simple. The price is a
tensor $2^4 = 16$ times the size of the spatial one. Ten of its 16 spin
blocks are zero (the table in section @sec:spin-adaptation-integral-selection-rule lists the six that are not).

> **Verified equation @eq:so-operator-forms** · the two operator forms are equal
>
> Equation (@eq:hamiltonian-1) with the spin sums written out equals (@eq:so-operator-forms) with (@eq:hamiltonian-3), as
> operators on the Fock space. The proof is the antisymmetry of the operator
> product $a^\dagger_p a^\dagger_q a_s a_r$ under $r \leftrightarrow s$, which
> turns the $\tfrac12$ and the direct term into the $\tfrac14$ and the
> antisymmetrized integral.
>
> | | |
> |---|---|
> | Proof | `scripts/casci_slater_condon_phases_sympy.py`, claim 5: both operators as 64 × 64 Jordan–Wigner matrices on 3 spatial orbitals, with random symmetric $h$, a random 8-fold symmetric $(pq\vert rs)$ and a random $E_{\mathrm{nn}}$ |
> | Negative control | the $\tfrac14$ replaced by $\tfrac12$, and a $+$ sign on the exchange term of (@eq:hamiltonian-3), must each differ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py`, which also applies the recorded mutation (exchange sign) |
> | Code | `ferric-mp2/src/spinorbital.rs::asym_phys` builds the tensor (@eq:hamiltonian-3) |
>
> <!-- verified: id=so.operator-forms; script=scripts/casci_slater_condon_phases_sympy.py; claim=5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/spinorbital.rs::asym_phys -->

### The code that builds (@eq:hamiltonian-3)

The code builds exactly this tensor from spatial chemist integrals:
`ferric-mp2/src/spinorbital.rs::asym_oovv` for the
occupied–occupied–virtual–virtual block,
`ferric-mp2/src/spinorbital.rs::asym_same` for blocks with all four indices
in one space, and the general `ferric-mp2/src/spinorbital.rs::asym_phys`.
The excerpt below is the body of the four nested loops of `asym_phys`, copied unchanged
(the loops over `p, q, r, s` are omitted):

<!-- excerpt: ferric-mp2/src/spinorbital.rs::asym_phys -->
```rust
                    // <pq|rs> = (pr|qs)
                    let dir = if spin(p) == spin(r) && spin(q) == spin(s) {
                        g_dir[[spat(p), spat(r), spat(q), spat(s)]]
                    } else {
                        0.0
                    };
                    // <pq|sr> = (ps|qr)
                    let exc = if spin(p) == spin(s) && spin(q) == spin(r) {
                        g_exc[[spat(p), spat(s), spat(q), spat(r)]]
                    } else {
                        0.0
                    };
                    out[[p, q, r, s]] = dir - exc;
```

Each line is a piece of (@eq:hamiltonian-3):

- **`dir` is the first term.** It is nonzero only when the spins pair as
  $\sigma(p) = \sigma(r)$ and $\sigma(q) = \sigma(s)$, and then reads the
  chemist integral $([p][r]|[q][s])$.
- **`exc` is the second term.** It is nonzero when the spins pair crosswise,
  and reads $([p][s]|[q][r])$.
- **`spin` and `spat` are the index map.** `spin(p) = p & 1` and
  `spat(p) = p >> 1`, so spin orbital $2[p]$ is $\alpha$ and $2[p]+1$ is
  $\beta$. Every spin-orbital tensor in `ferric-mp2` and `ferric-cc` uses this
  interleaved order.

The function takes two spatial arrays, `g_dir` and `g_exc`, because for a
block such as $\langle kb\|cj\rangle$ the direct and exchange terms are
different spatial blocks ($(kc|bj)$ and $(kj|bc)$). For
$\langle ij\|ab\rangle$ both come from one $(ia|jb)$ array, which is what
`asym_oovv` exploits.

These loops visit every element of a $(2n)^4$ tensor, with no blocking.
They serve the dense spin-orbital MP3, CCD and CCSD codes. The closed-shell
RI-MP2 and spin-adapted CCSD codes never build this tensor: section @sec:spin-adaptation
does the spin sums by hand, once, so that they work on spatial integrals
only.

## The reference determinant and normal ordering {#hamiltonian-reference-determinant-normal-ordering}

Let $|0\rangle$ be a single determinant with spin orbitals $i, j, k, \dots$
occupied. Most of the methods in this book start from such a determinant,
usually Hartree–Fock, and describe everything else as a correction to it.
So every operator is measured relative to $|0\rangle$.

### The reference energy

Two expectation values in a determinant do all the work. With $n_p = 1$
if $p$ is occupied and 0 otherwise,

$$ \langle 0|a^\dagger_p a_q|0\rangle = n_p\,\delta_{pq}, \qquad \langle 0|a^\dagger_p a^\dagger_q a_s a_r|0\rangle = n_p n_q\,\big(\delta_{pr}\delta_{qs} - \delta_{ps}\delta_{qr}\big) . $$

The first says you can only remove and replace an electron that is there.
The second says you can remove two electrons and put them back in the same
two orbitals, either in the same order (+) or swapped (−). Substitute both
into (@eq:so-operator-forms). The one-electron sum collapses to $\sum_i h_{ii}$. The
two-electron sum becomes
$\tfrac14\sum_{ij}\big(\langle ij\|ij\rangle - \langle ij\|ji\rangle\big)$,
and antisymmetry gives $\langle ij\|ji\rangle = -\langle ij\|ij\rangle$, so

$$ E_{\mathrm{ref}} = \langle 0|\hat H|0\rangle = E_{\mathrm{nn}} + \sum_i h_{ii} + \frac12\sum_{ij}\langle ij\|ij\rangle . \label{eq:so-ref-energy} $$

Check one electron. The two-electron term is
$\tfrac12\langle ii\|ii\rangle$, and an antisymmetrized integral with two
equal labels is zero. So $E_{\mathrm{ref}} = E_{\mathrm{nn}} + h_{ii}$: the
electron does not repel itself. In (@eq:so-ref-energy) the exchange half of
$\langle ii\|ii\rangle$ cancels the Coulomb half exactly.

> **Verified equation @eq:so-ref-energy** · the reference energy by Wick's theorem
>
> The fully contracted terms of $\langle 0|\hat H|0\rangle$, evaluated with
> `sympy.physics.secondquant.wicks`, equal $\sum_i h_{ii} +
> \tfrac12\sum_{ij}\langle ij\|ij\rangle$ on a random antisymmetrized model.
>
> | | |
> |---|---|
> | Proof | `scripts/oomp2_mp3_identities_sympy.py`, the check labelled "claim B1" ($E_{\mathrm{HF}}$ from Wick's theorem) |
> | Negative control | none for this check; the script's mutations act on the MP3 ring coefficient and the rotation sign |
> | Runner | `crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py` |
> | Code | `ferric-mp2/src/spinorbital.rs::asym_same` builds the $\langle pq\Vert rs\rangle$ the sum reads |
>
> <!-- verified: id=so.ref-energy; script=scripts/oomp2_mp3_identities_sympy.py; claim=B1; runner=crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py; code=ferric-mp2/src/spinorbital.rs::asym_same -->

The CAS-CI code evaluates (@eq:so-ref-energy) for every determinant it handles, as the
diagonal of its Hamiltonian. In `ferric-ci/src/hamiltonian.rs::diagonal_element`
the determinant is a list of occupied $\alpha$ and $\beta$ spatial orbitals,
so the spin deltas of (@eq:hamiltonian-3) are resolved by which loop a pair sits in.
With the $\beta$ one-electron loop and the $\beta\beta$ pair loop removed
(they repeat the $\alpha$ ones):

```rust
for &p in a_occ {
    e += h1(ints, p, p);                   // sum_i h_ii (alpha; beta likewise)
}
// ...
for i in 0..a_occ.len() {
    for j in (i + 1)..a_occ.len() {        // same-spin pairs, i < j
        let p = a_occ[i];
        let q = a_occ[j];
        e += h2(ints, p, p, q, q) - h2(ints, p, q, q, p);  // (pp|qq) - (pq|qp)
    }
}
// ...
for &p in a_occ {
    for &q in b_occ {
        e += h2(ints, p, p, q, q);         // opposite spins: Coulomb only
    }
}
```

Two points connect it to (@eq:so-ref-energy):

- **$\tfrac12\sum_{ij}$ became $\sum_{i<j}$.** The summand is symmetric in
  $i, j$ and zero at $i = j$, so half the full sum is the sum over $i < j$.
  The loops skip the zero diagonal.
- **The exchange term appears only for same-spin pairs.** That is the
  $\delta_{\sigma(p)\sigma(s)}\delta_{\sigma(q)\sigma(r)}$ of (@eq:hamiltonian-3). The
  $\alpha\beta$ loop has the Coulomb integral alone.

`diagonal_element` returns the electronic part only. The constant
`e_core`, which holds $E_{\mathrm{nn}}$ and the frozen inactive core, is
added by the caller.

### Normal ordering and the split of $\hat H$

Normal ordering $\{\cdots\}$ is relative to $|0\rangle$. Inside the braces,
an operator that would give zero on $|0\rangle$ goes to the right: the
annihilator of a virtual orbital, and the creator of an occupied one
(you cannot add a second electron to a full orbital). The operators that
act on $|0\rangle$ without killing it go to the left. Each swap costs a
sign. The point of the definition is that the expectation value of any
normal-ordered string in $|0\rangle$ is zero.

Wick's theorem (section @sec:wick) relates a plain string to normal-ordered
ones. A plain string is its normal-ordered form plus every way of
contracting pairs of its operators, where a contraction of $a^\dagger_p$
with $a_q$ is $n_p\delta_{pq}$. For one creator and one annihilator,

$$ a^\dagger_p a_q = \{a^\dagger_p a_q\} + n_p\,\delta_{pq} . $$

For the two-electron string there are four single contractions and two
full ones:

$$ a^\dagger_p a^\dagger_q a_s a_r = \{a^\dagger_p a^\dagger_q a_s a_r\} + n_p\delta_{pr}\{a^\dagger_q a_s\} - n_p\delta_{ps}\{a^\dagger_q a_r\} - n_q\delta_{qr}\{a^\dagger_p a_s\} + n_q\delta_{qs}\{a^\dagger_p a_r\} + n_p n_q\big(\delta_{pr}\delta_{qs} - \delta_{ps}\delta_{qr}\big) . $$

Each minus sign counts one operator crossing. Substitute both expansions
into (@eq:so-operator-forms):

- The full contractions are the expectation values above, and give the
  constant $E_{\mathrm{ref}}$ of (@eq:so-ref-energy).
- The four single contractions of the $\tfrac14\langle pq\|rs\rangle$
  term are equal. Relabel each so the surviving pair is $\{a^\dagger_p
  a_q\}$ and the contracted label is $k$, and use
  $\langle pq\|rs\rangle = -\langle qp\|rs\rangle = -\langle pq\|sr\rangle$
  to fix the signs. Each gives
  $\tfrac14\sum_k\langle pk\|qk\rangle\{a^\dagger_p a_q\}$, and four of
  them give $\sum_k\langle pk\|qk\rangle\{a^\dagger_p a_q\}$. Add the
  one-electron term $h_{pq}\{a^\dagger_p a_q\}$ and you have the Fock
  operator.
- The uncontracted string stays as it is.

So the Fock operator has matrix elements

$$ f_{pq} = h_{pq} + \sum_k \langle pk\|qk\rangle , \label{eq:hamiltonian-5} $$

and the Hamiltonian splits into a constant, a one-body part and a two-body
fluctuation part:

$$ \hat H = E_{\mathrm{ref}} + \sum_{pq} f_{pq}\,\{a^\dagger_p a_q\} + \frac14\sum_{pqrs}\langle pq\|rs\rangle\,\{a^\dagger_p a^\dagger_q a_s a_r\} . \label{eq:so-normal-order-split} $$

What does each piece mean? $f_{pq}$ is the energy of one electron in the
bare one-electron potential plus the averaged Coulomb and exchange field
of all the electrons in $|0\rangle$; that is the mean field of
Hartree–Fock. The last term, the fluctuation potential
$\hat V_{\mathrm N}$, is what the mean field leaves out: the instantaneous
repulsion minus its average. Every perturbation and coupled-cluster
derivation in the book starts from this form, with $|0\rangle$ the
Hartree–Fock determinant, and treats $\hat V_{\mathrm N}$ as the part to be
corrected for.

For canonical orbitals $f_{pq} = \varepsilon_p\delta_{pq}$, and the
denominators of the notation chapter are built from $\varepsilon$. Check
the trivial case: with no electrons in $|0\rangle$ every $n_p$ is zero,
there are no contractions, $f = h$ and $E_{\mathrm{ref}} = E_{\mathrm{nn}}$,
and (@eq:so-normal-order-split) is (@eq:so-operator-forms) again.

> **Verified equation @eq:so-normal-order-split** · the split of the Hamiltonian
>
> Equations (@eq:so-ref-energy), (@eq:hamiltonian-5) and (@eq:so-normal-order-split) together say $\hat H = E_{\mathrm{ref}} +
> \hat F_{\mathrm N} + \hat V_{\mathrm N}$ as operators.
>
> | | |
> |---|---|
> | Proof | `scripts/casci_slater_condon_phases_sympy.py`, claim 6: $\hat H$ and $E_{\mathrm{ref}} + \hat F_{\mathrm N} + \hat V_{\mathrm N}$ as Jordan–Wigner matrices, for a closed-shell and an open-shell reference. The normal-ordered products come from the quasi-particle operators of the reference and the permutation sign alone, so the check does not use Wick's theorem |
> | Negative control | $\hat F_{\mathrm N}$ and $\hat V_{\mathrm N}$ without normal ordering, and $f_{pq}$ without the sum over $k$, must each differ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py`, which also applies the recorded mutation (the $k$ sum dropped from $f$) |
> | Code | `ferric-mp2/src/spinorbital.rs::asym_same` |
>
> <!-- verified: id=so.normal-order-split; script=scripts/casci_slater_condon_phases_sympy.py; claim=6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/spinorbital.rs::asym_same -->

## What a determinant is, in this code {#hamiltonian-determinant-code}

The Hartree–Fock determinant is $|\chi_1\chi_2\cdots\chi_n\rangle$, an
antisymmetrized product of occupied spin orbitals (dissertation, chapter 1,
Eq. 1.7). Its energy is Eq. (@eq:so-ref-energy). Section @sec:spin-adaptation gives it in spatial orbitals
and chapter @sec:hartree-fock minimizes it.

In second quantization the determinant is a product of creators acting on
the vacuum, $a^\dagger_{p_1}a^\dagger_{p_2}\cdots|\mathrm{vac}\rangle$.
Which orbitals are occupied is a set, and the code stores the set as bits.
The order of the creators fixes the sign of the determinant, so the code
has to choose one order and keep it.

Two phase conventions for ordering spin orbitals exist in the code, and
section @sec:spin-adaptation states what each changes. Spin-orbital tensors for MP3 and
coupled cluster are interleaved ($p_\alpha$ is $2[p]$, $p_\beta$ is
$2[p]+1$). CAS-CI determinants put every $\alpha$ before every $\beta$.

CAS-CI keeps a determinant as two `u64` strings, one per spin, with bit $p$
set when spatial orbital $p$ is occupied in that spin. Removing or adding an
electron at $p$ costs $(-1)$ for every occupied orbital below $p$, because
the operator has to anticommute past each of their creators to reach its
own. That is all of `ferric-ci/src/strings.rs::phase_below`:

<!-- excerpt: ferric-ci/src/strings.rs::phase_below -->
```rust
    // Count set bits strictly below position p.
    let mask = (1u64 << p) - 1;
    if (s & mask).count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
```

It counts only within one spin's string. For an $\alpha$ operator that is
the full sign. A $\beta$ operator would also cross all $N_\alpha$ $\alpha$
creators, but $\beta$ operators come in creator–annihilator pairs in every
term of $\hat H$, so that factor appears an even number of times and
cancels (section @sec:determinants-convention).

## Sources

| Source | Used for |
|---|---|
| Dissertation, chapter 1, section 1.1, Eqs. 1.3–1.7 and 1.10 | the Born–Oppenheimer separation, the determinant and the Hartree–Fock energy |
| sections @sec:determinants-convention and @sec:rdms-1-2-rdm | form (@eq:hamiltonian-1) and its density matrices |
| `crates/ferric-mp2/src/spinorbital.rs` | the construction (@eq:hamiltonian-3) |
| `scripts/oomp2_mp3_identities_sympy.py` | verified equation @eq:so-ref-energy |

Changes from the sources: spin orbitals are labeled by lowercase letters with
the helper functions $[p]$ and $\sigma(p)$, and the Hamiltonian is written
with both forms so that section @sec:spin-adaptation can reduce one to the other.
