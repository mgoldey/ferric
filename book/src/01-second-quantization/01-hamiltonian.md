# 1.1 Hamiltonian, spin orbitals and normal ordering

<!-- section: 1.1; status: edit; registry: ;
     theory: -->

Every method in this book starts from one operator, the electronic
Hamiltonian in a finite orbital basis. This section writes it in second
quantization, fixes what a spin orbital is, and splits it relative to a
reference determinant. Chapter 1.2 shows how the proof scripts use that split.
Chapter 1.3 reduces spin orbitals to spatial orbitals.

| Index | Range in this chapter | Code name |
|---|---|---|
| $p, q, r, s$ | spin orbitals ($2n$ of them) in a spin-orbital expression | `0..2*n`, interleaved |
| $i, j, k, l$ | occupied spin orbitals | `0..2*no` |
| $a, b, c, d$ | virtual spin orbitals | `0..2*nv` |
| $[p]$ | the spatial orbital of spin orbital $p$ | `spat(p)` = `p >> 1` |
| $\sigma(p)$ | the spin of spin orbital $p$, $\alpha$ or $\beta$ | `spin(p)` = `p & 1` |

In a spin-orbital expression the letters above run over spin orbitals. In a
spatial expression, from section 1.3 on, the same letters run over spatial
orbitals, and the section says so. Where both appear in one equation, a spin
orbital is written $p_\sigma$.

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/spinorbital.rs::{spin,spat,asym_oovv,asym_same,asym_phys}` |
| CLI, Python | none; this section is background for every chapter |

## 1.1.1 The electronic Hamiltonian

The nuclei are fixed (the Born–Oppenheimer approximation). The electrons
move in the field of point charges $Z_A$ at positions $\mathbf R_A$, in
atomic units. The dissertation writes the electronic Hamiltonian as the sum of
the electronic kinetic energy, the electron–nucleus attraction and the
electron–electron repulsion (chapter 1, Eqs. 1.3–1.5). The nuclear repulsion
$E_{\mathrm{nn}} = \sum_{A<B} Z_A Z_B / |\mathbf R_A - \mathbf R_B|$ is a
constant added at the end.

The one-electron operator is
$\hat h = -\tfrac12\nabla^2 - \sum_A Z_A/|\mathbf r - \mathbf R_A|$. In the code
the same name covers more: `hcore` also holds an effective core potential, an
external point-charge potential or a field when one is set (chapters 3 and
23).

Orbitals are real and orthonormal. A spin orbital is a spatial orbital times a
spin function. With spatial orbitals $\phi_p$ (molecular orbitals, expanded in
the AO basis as $\phi_p = \sum_\mu C_{\mu p}\chi_\mu$) the integrals are

$$ h_{pq} = \int \phi_p\,\hat h\,\phi_q\,d\mathbf r, \qquad (pq|rs) = \iint \phi_p(\mathbf r_1)\phi_q(\mathbf r_1)\,\frac{1}{r_{12}}\,\phi_r(\mathbf r_2)\phi_s(\mathbf r_2)\,d\mathbf r_1 d\mathbf r_2 . $$

The notation chapter fixes the symmetry $(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)$.

## 1.1.2 Second quantization

With $a^\dagger_{p\sigma}$ creating an electron of spin $\sigma$ in spatial
orbital $p$,

$$ \hat H = E_{\mathrm{nn}} + \sum_{pq}\sum_{\sigma} h_{pq}\,a^\dagger_{p\sigma} a_{q\sigma} + \frac12 \sum_{pqrs}\sum_{\sigma\tau} (pq|rs)\,a^\dagger_{p\sigma} a^\dagger_{r\tau} a_{s\tau} a_{q\sigma} . \tag{1.1} $$

This is the form the CAS-CI code builds, and its density matrices follow
the same pairing of spin labels: $\gamma_{pq} = \langle E_{pq}\rangle$ and
$\Gamma_{pqrs} = \sum_{\sigma\tau}\langle a^\dagger_{p\sigma} a^\dagger_{r\tau}
a_{s\tau} a_{q\sigma}\rangle$ (`crates/ferric-ci/theory.md`, "Reduced density
matrices").

In spin orbitals the same operator is written with antisymmetrized
integrals,

$$ \hat H = E_{\mathrm{nn}} + \sum_{pq} h_{pq}\,a^\dagger_p a_q + \frac14 \sum_{pqrs} \langle pq\|rs\rangle\, a^\dagger_p a^\dagger_q a_s a_r , \tag{1.2} $$

$$ h_{pq} = h_{[p][q]}\,\delta_{\sigma(p)\sigma(q)}, \qquad \langle pq\|rs\rangle = ([p][r]|[q][s])\,\delta_{\sigma(p)\sigma(r)}\delta_{\sigma(q)\sigma(s)} - ([p][s]|[q][r])\,\delta_{\sigma(p)\sigma(s)}\delta_{\sigma(q)\sigma(r)} . \tag{1.3} $$

The code builds exactly this tensor from spatial chemist integrals:
`ferric-mp2/src/spinorbital.rs::asym_oovv` for the occupied–occupied–virtual–virtual
block, `ferric-mp2/src/spinorbital.rs::asym_same` for blocks with all four
indices in one space, and the general
`ferric-mp2/src/spinorbital.rs::asym_phys`. Each takes the direct term when the
spins pair as $\sigma(p) = \sigma(r)$ and $\sigma(q) = \sigma(s)$, and the
exchange term when they pair crosswise.

> **Verified equation 1.2** · the two operator forms are equal
>
> Equation (1.1) with the spin sums written out equals (1.2) with (1.3), as
> operators on the Fock space. The proof is the antisymmetry of the operator
> product $a^\dagger_p a^\dagger_q a_s a_r$ under $r \leftrightarrow s$, which
> turns the $\tfrac12$ and the direct term into the $\tfrac14$ and the
> antisymmetrized integral.
>
> | | |
> |---|---|
> | Proof | `scripts/casci_slater_condon_phases_sympy.py`, claim 5: both operators as 64 × 64 Jordan–Wigner matrices on 3 spatial orbitals, with random symmetric $h$, a random 8-fold symmetric $(pq\vert rs)$ and a random $E_{\mathrm{nn}}$ |
> | Negative control | the $\tfrac14$ replaced by $\tfrac12$, and a $+$ sign on the exchange term of (1.3), must each differ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py`, which also applies the recorded mutation (exchange sign) |
> | Code | `ferric-mp2/src/spinorbital.rs::asym_phys` builds the tensor (1.3) |
>
> <!-- verified: id=so.operator-forms; script=scripts/casci_slater_condon_phases_sympy.py; claim=5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/spinorbital.rs::asym_phys -->

## 1.1.3 The reference determinant and normal ordering

Let $|0\rangle$ be a single determinant with spin orbitals $i, j, k, \dots$
occupied. Normal ordering $\{\cdots\}$ is relative to it: a creation operator
of an occupied orbital and an annihilation operator of an occupied orbital
are exchanged to the other side, so that the expectation value of any
normal-ordered string in $|0\rangle$ is zero.

The reference energy is the expectation value of (1.2):

$$ E_{\mathrm{ref}} = \langle 0|\hat H|0\rangle = E_{\mathrm{nn}} + \sum_i h_{ii} + \frac12\sum_{ij}\langle ij\|ij\rangle . \tag{1.4} $$

> **Verified equation 1.4** · the reference energy by Wick's theorem
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

The Fock operator has matrix elements

$$ f_{pq} = h_{pq} + \sum_k \langle pk\|qk\rangle , \tag{1.5} $$

and the Hamiltonian splits into a constant, a one-body part and a two-body
fluctuation part:

$$ \hat H = E_{\mathrm{ref}} + \sum_{pq} f_{pq}\,\{a^\dagger_p a_q\} + \frac14\sum_{pqrs}\langle pq\|rs\rangle\,\{a^\dagger_p a^\dagger_q a_s a_r\} . \tag{1.6} $$

The last term is the fluctuation potential $\hat V_{\mathrm N}$. Every
perturbation and coupled-cluster derivation in the book starts from this
form, with $|0\rangle$ the Hartree–Fock determinant. For canonical orbitals
$f_{pq} = \varepsilon_p\delta_{pq}$, and the denominators of the notation
chapter are built from $\varepsilon$.

> **Verified equation 1.6** · the split of the Hamiltonian
>
> Equations (1.4), (1.5) and (1.6) together say $\hat H = E_{\mathrm{ref}} +
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

## 1.1.4 What a determinant is, in this code

The Hartree–Fock determinant is $|\chi_1\chi_2\cdots\chi_n\rangle$, an
antisymmetrized product of occupied spin orbitals (dissertation, chapter 1,
Eq. 1.7). Its energy is Eq. (1.4). Section 1.3 gives it in spatial orbitals
and chapter 7 minimizes it.

Two phase conventions for ordering spin orbitals exist in the code, and
section 1.3 states what each changes. Spin-orbital tensors for MP3 and
coupled cluster are interleaved ($p_\alpha$ is $2[p]$, $p_\beta$ is
$2[p]+1$). CAS-CI determinants put every $\alpha$ before every $\beta$.

## Sources

| Source | Used for |
|---|---|
| Dissertation, chapter 1, section 1.1, Eqs. 1.3–1.7 and 1.10 | the Born–Oppenheimer separation, the determinant and the Hartree–Fock energy |
| `crates/ferric-ci/theory.md`, "Determinants and the phase convention", "Reduced density matrices" | form (1.1) and its density matrices |
| `crates/ferric-mp2/src/spinorbital.rs` | the construction (1.3) |
| `scripts/oomp2_mp3_identities_sympy.py` | verified equation 1.4 |

Changes from the sources: spin orbitals are labeled by lowercase letters with
the helper functions $[p]$ and $\sigma(p)$, and the Hamiltonian is written
with both forms so that section 1.3 can reduce one to the other.
