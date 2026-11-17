# Slater–Condon rules as implemented {#slater-condon}

<!-- status: ready; registry: casci;
     theory: crates/ferric-ci/theory.md#"Slater–Condon rules as implemented" -->

$H$ moves at most two electrons, so two determinants that differ in more
than two spin orbitals do not interact, and the matrix element of the rest
depends only on which orbitals differ. The textbook Slater–Condon rules give
the integrals for each case. The sign in front of them comes from
reordering operators, depends on the convention of section @sec:determinants, and is
what the code has to get right. ferric's production CI never visits a
determinant pair (section @sec:direct-sigma-davidson), so `general_element` serves as the
reference that the direct sigma is tested against.

| Class | Condition | Function |
|---|---|---|
| diagonal | $d_\alpha = d_\beta = 0$ | `diagonal_element` |
| single | $d = 2$ in one channel | `single_excitation_same_spin` |
| same-spin double | $d = 4$ in one channel | `double_same_spin`, `double_phase_same_spin` |
| opposite-spin double | $d_\alpha = d_\beta = 2$ | `double_opposite_spin` |

| Entry point | Location |
|---|---|
| Rust | `ferric-ci/src/hamiltonian.rs::{general_element,diagonal_element,dense_hamiltonian,sigma_naive}` |

## The rules {#slater-condon-rules}

`ferric-ci/src/hamiltonian.rs::general_element` takes the bra strings
$(A', B')$ and the ket strings $(A, B)$ and returns
$\langle A',B'|H - E_\text{core}|A,B\rangle$. It classifies the pair by the
number of differing bits in each channel,
$d_\alpha = |A'\oplus A|$ and $d_\beta = |B'\oplus B|$:

```rust
let da = (a_bra ^ a_ket).count_ones();   // d_α: bits set in one α string only
let db = (b_bra ^ b_ket).count_ones();   // d_β
```

Moving one electron flips two bits in its channel, one emptied and one
filled. So $d = 2$ in one channel is a single excitation and $d = 4$ in one
channel, or $d_\alpha = d_\beta = 2$, is a double. Every orbital here is
active, so the labels are the active letters: $t, u$ are vacated and $v, w$
filled.

### Diagonal

For $d_\alpha = d_\beta = 0$ (`diagonal_element`):

$$
\sum_{p\in A\cup B} h_{pp}
+ \sum_{\substack{p<q\\ p,q\in A}} \big[(pp|qq)-(pq|qp)\big]
+ \sum_{\substack{p<q\\ p,q\in B}} \big[(pp|qq)-(pq|qp)\big]
+ \sum_{p\in A,\,q\in B} (pp|qq).
$$

Each electron contributes its one-electron energy. Each same-spin pair
contributes Coulomb minus exchange, and each opposite-spin pair Coulomb
only: exchange needs the two electrons to have the same spin. The sum over
$p \in A\cup B$ runs over occupied spin orbitals, so a doubly occupied
orbital counts $h_{pp}$ twice.

### Single excitation

For $d = 2$ in one channel (`single_excitation_same_spin`), an electron
moves $t\to v$ in channel $\sigma$. Call $s$ the ket string of that channel
and $\tilde s$ the string of the other channel. Then

$$
\langle D'|H|D\rangle = \epsilon_1
\Big[h_{vt} + \sum_{p\in s\setminus\{t\}} \big((vt|pp)-(vp|pt)\big)
     + \sum_{p\in\tilde s} (vt|pp)\Big],
\qquad
\epsilon_1 = \text{pb}(s,t)\;\text{pb}(s\setminus t,\,v).
$$

The bracket is the one-electron hop plus its interaction with the
spectator electrons: Coulomb minus exchange with the same-spin ones,
Coulomb with the others. The sign $\epsilon_1 = \langle D'|a^\dagger_v a_t|D\rangle$
comes from applying the operators right to left. Annihilate $t$ in $s$:
${\rm pb}(s, t)$. Then create $v$ in the string that no longer holds $t$:
${\rm pb}(s\setminus t, v)$. Taking the second phase in $s$ instead is
wrong whenever $t < v$, and is one of the mutations the proof script
rejects.

In the code, with the lines that build the spectator string elided (`i` is
$t$ and `a` is $v$):

```rust
let phase_i = phase_below(ket, i);                       // pb(s, t)
let ket_minus = ket & !(1u64 << i);
let phase_a = phase_below(ket_minus, a);                 // pb(s \ t, v)
let phase = phase_i * phase_a;                           // ε₁

let mut val = h1(ints, a, i);                            // h_vt
// ...
for p in crate::strings::occupied_orbitals(same_spectators) {
    val += h2(ints, a, i, p, p) - h2(ints, a, p, p, i);  // (vt|pp) − (vp|pt)
}
// opposite-spin occupied: Coulomb only.
for p in crate::strings::occupied_orbitals(other_occ_str) {
    val += h2(ints, a, i, p, p);                         // (vt|pp)
}
```

### Same-spin double excitation

For $d = 4$ in one channel (`double_same_spin`), $t<u$ are removed and
$v<w$ added. Within one spin, the term
$\tfrac12\sum(pq|rs)\,a^\dagger_p a^\dagger_r a_s a_q$ connects the two
determinants through four index choices: $(p, r)$ is $(v, w)$ or $(w, v)$,
and $(q, s)$ is $(t, u)$ or $(u, t)$. Bring each to the operator order
$a^\dagger_v a^\dagger_w a_u a_t$:

- $p=v, r=w, q=t, s=u$: $(vt|wu)$, already in that order.
- $p=w, r=v, q=u, s=t$: $(wu|vt)\,a^\dagger_w a^\dagger_v a_t a_u$. Swap the
  creators and swap the annihilators (two signs), and use
  $(wu|vt) = (vt|wu)$: $+(vt|wu)$.
- $p=v, r=w, q=u, s=t$: $(vu|wt)\,a^\dagger_v a^\dagger_w a_t a_u$. Swap the
  annihilators: $-(vu|wt)$.
- $p=w, r=v, q=t, s=u$: $(wt|vu)\,a^\dagger_w a^\dagger_v a_u a_t$. Swap the
  creators and use $(wt|vu) = (vu|wt)$: $-(vu|wt)$.

The four terms times $\tfrac12$ give

$$
\langle D'|H|D\rangle = \epsilon_2\big[(vt|wu)-(vu|wt)\big],
\qquad
\epsilon_2 = \langle D'|a^\dagger_v a^\dagger_w a_u a_t|D\rangle .
$$

`double_phase_same_spin` evaluates $\epsilon_2$ as successive
single-channel parities, applying the operators right to left. In code
names $(i, j, a, b) = (t, u, v, w)$:

```rust
let mut s = ket;
let mut phase = phase_below(s, i);   // annihilate t
s &= !(1u64 << i);
phase *= phase_below(s, j);          // annihilate u, in the string without t
s &= !(1u64 << j);
// ...
phase *= phase_below(s, b);          // create w
s |= 1u64 << b;
phase *= phase_below(s, a);          // create v, in the string that holds w
phase
```

Only the comments have been removed. Creating $v$ before $w$ would apply
$a^\dagger_w a^\dagger_v = -a^\dagger_v a^\dagger_w$ and flip the sign of
every same-spin double; that is mutation 4a of the proof script.

### Opposite-spin double excitation

For $d_\alpha = d_\beta = 2$ (`double_opposite_spin`), $t\to v$ in α and
$u\to w$ in β. Two terms of $H$ connect the determinants. The
$\sigma\tau = \alpha\beta$ term is
$\tfrac12(vt|wu)\,a^\dagger_{v\alpha}a^\dagger_{w\beta}a_{u\beta}a_{t\alpha}$.
The $\beta\alpha$ term is
$\tfrac12(wu|vt)\,a^\dagger_{w\beta}a^\dagger_{v\alpha}a_{t\alpha}a_{u\beta}$;
swap both pairs and use $(wu|vt) = (vt|wu)$, and it equals the first. Their
sum is $(vt|wu)\,a^\dagger_{v\alpha}a^\dagger_{w\beta}a_{u\beta}a_{t\alpha}$.
Move $a_{t\alpha}$ left past $a_{u\beta}$ and $a^\dagger_{w\beta}$. That costs
two signs, and no contraction term appears because the spins differ. The
operator is then $E^\alpha_{vt}E^\beta_{wu}$, a product of one excitation
per channel:

$$
\langle D'|H|D\rangle = \epsilon_\alpha\,\epsilon_\beta\,(vt|wu),
\qquad
\epsilon_\alpha = \text{pb}(A,t)\,\text{pb}(A\setminus t,v),\;
\epsilon_\beta = \text{pb}(B,u)\,\text{pb}(B\setminus u,w).
$$

There is no exchange term: exchange would move the α electron from $t$ into
$w$, which is a β orbital. The code carries one phase per channel:

<!-- excerpt: ferric-ci/src/hamiltonian.rs::double_opposite_spin -->
```rust
// α phase
let pa_i = phase_below(a_ket, i);
let pa_a = phase_below(a_ket & !(1u64 << i), a);
// β phase
let pb_j = phase_below(b_ket, j);
let pb_b = phase_below(b_ket & !(1u64 << j), b);

pa_i * pa_a * pb_j * pb_b * h2(ints, a, i, b, j)
```

| Symbol | Code |
|---|---|
| $\epsilon_\alpha = {\rm pb}(A,t)\,{\rm pb}(A\setminus t, v)$ | `pa_i * pa_a` |
| $\epsilon_\beta = {\rm pb}(B,u)\,{\rm pb}(B\setminus u, w)$ | `pb_j * pb_b` |
| $(vt|wu)$ | `h2(ints, a, i, b, j)` |

### Everything else, and a (2,2) check

All other classes are zero: $d_\alpha + d_\beta > 4$, or $d = 2$ split
across the channels. `sigma_naive` and `dense_hamiltonian` apply
`general_element` to every pair and add $E_\text{core}$ on the diagonal;
`hamiltonian_diagonal` uses `diagonal_element` plus $E_\text{core}$.
`run_cas_ci` calls neither pair loop. They are test and benchmark oracles.

Take H2 in a CAS(2,2) with orbitals 0 ($\sigma_g$) and 1 ($\sigma_u$). The
opposite-spin double from $|0\alpha\,0\beta\rangle$ to
$|1\alpha\,1\beta\rangle$ has $t = u = 0$, $v = w = 1$, and every phase
${\rm pb}$ counts an empty set below orbital 0 or 1, so the element is
$+(10|10)$, the exchange integral $K$ of the two orbitals. The singles to
$|0\alpha\,1\beta\rangle$ and $|1\alpha\,0\beta\rangle$ need $h_{10}$,
$(10|00)$ and $(10|11)$, which have an odd number of $\sigma_u$ indices and
vanish by inversion symmetry. What is left is the $2\times2$ problem

$$
\begin{pmatrix} 2h_{00} + (00|00) & K \\ K & 2h_{11} + (11|11) \end{pmatrix}
\begin{pmatrix} c_1 \\ c_2\end{pmatrix} = E\begin{pmatrix} c_1 \\ c_2\end{pmatrix}.
$$

With $K > 0$ the lowest root has $c_1c_2 < 0$. In the dissociation limit
of a minimal basis the two diagonal elements become equal, and then
$|c_1| = |c_2| = 1/\sqrt2$. The element $K$ and its sign
were checked on Jordan–Wigner matrices in scratch.

> **Verified equation {#eq:ci-slater-condon-elements}** · every element against the Jordan–Wigner $H$
>
> For every bra–ket pair in the sectors $(n, N_\alpha, N_\beta) =$
> (4,2,2), (5,3,2), (5,2,3) and (5,3,3), a line-for-line port of
> `general_element` equals the matrix element of the explicit Jordan–Wigner
> Hamiltonian as an exact linear form in symbolic $h_{pq}$ and
> $(pq|rs)$. The odd-$N_\alpha$ sectors are the ones that test the omitted
> α parity on β operators. Four sign mutations each mismatch, and the
> interleaved order mismatches element-wise with the same spectrum.
>
> | | |
> |---|---|
> | Proof | `scripts/casci_slater_condon_phases_sympy.py`, claim 3 |
> | Negative control | claim 4, cases a–e |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/hamiltonian.rs::{general_element,diagonal_element,double_opposite_spin}` |
> | Code test | `general_element_matches_second_quantization_on_random_integrals` in `crates/ferric-ci/tests/slater_condon_brute_force.rs` |
>
> <!-- verified: id=ci.slater-condon.elements; script=scripts/casci_slater_condon_phases_sympy.py; claim=3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/hamiltonian.rs::{general_element,diagonal_element,double_opposite_spin}; rust_test=ferric-ci/tests/slater_condon_brute_force.rs::general_element_matches_second_quantization_on_random_integrals; registry=casci -->

## Why an energy test is not enough {#slater-condon-energy-test-enough}

An FCI energy against PySCF checks the spectrum only. A $\pm1$ similarity
inserted into `general_element` passes every energy test, and H2 in a
minimal basis has no nontrivial sign at all. The element-wise checks above,
and the Rust test against an independent bit-loop second-quantized $H$ on
random asymmetric integrals, are what pin the convention.

## Numerical anchors {#slater-condon-numerical-anchors}

| Quantity | Systems | Error | Test |
|---|---|---|---|
| `general_element`, `sigma` (direct) and `hamiltonian_diagonal` vs an independent bit-loop second-quantized $H$ on random asymmetric integrals, element-wise, every excitation class exercised | CAS(4,4) to CAS(6,6), odd and even $N_\alpha$ | $\le 8.9\times10^{-15}$ measured, bar $10^{-12}$ | `crates/ferric-ci/tests/slater_condon_brute_force.rs` |

## Sources {#slater-condon-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-ci/theory.md`, "Slater–Condon rules as implemented" | the four classes and their phases |
| `scripts/casci_slater_condon_phases_sympy.py` | claims 1–4 |
| `crates/ferric-ci/theory.md`, "Checks" | what each test constrains |

**Known limits (2026-11-15).** `dense_hamiltonian` and `sigma_naive` visit
every determinant pair and are test and benchmark oracles; `run_cas_ci`
calls neither.
