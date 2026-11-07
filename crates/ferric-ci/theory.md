# ferric-ci: determinant phases and Slater–Condon rules

Scope: the CAS-CI Hamiltonian in `src/strings.rs`, `src/hamiltonian.rs` and
the direct sigma build in `src/sigma.rs`.
Exact check: `scripts/casci_slater_condon_phases_sympy.py`. Numerical check:
`tests/slater_condon_brute_force.rs`.

## Determinants and the phase convention

A determinant is a pair of `u64` strings $(A, B)$ over $n$ active spatial
orbitals; bit $p$ of $A$ ($B$) set means $p\alpha$ ($p\beta$) is occupied.
Spin-orbitals are ordered **α before β**: mode $p$ is $p\alpha$, mode
$n+p$ is $p\beta$. The determinant is

$$
|A,B\rangle = \prod_{p\in A}^{\uparrow} a^\dagger_{p\alpha}
              \prod_{q\in B}^{\uparrow} a^\dagger_{q\beta}\,|0\rangle ,
$$

creators in ascending index order from left to right. In this order a
Jordan–Wigner annihilator on mode $P$ acting on the bit string $s$ gives

$$
a_P\,|s\rangle = (-1)^{\#\{Q<P\,:\,Q\in s\}}\,|s\setminus P\rangle ,
\qquad
a^\dagger_P\,|s\rangle = (-1)^{\#\{Q<P\,:\,Q\in s\}}\,|s\cup P\rangle .
$$

`strings::phase_below(s, p)` evaluates $(-1)^{\#\{q<p\,:\,q\in s\}}$ on one
spin channel's string. For an α operator it is the full JW sign. For a β
operator the full sign carries an extra $(-1)^{N_\alpha}$; every term of $H$
contains β operators in annihilator/creator pairs, so the factor appears an
even number of times and cancels. The code therefore never counts α
electrons for a β operator.

Any other determinant phase convention (e.g. interleaved
$p\alpha, p\beta$ ordering) changes $H_{IJ}$ by $S_I S_J$, $S_I=\pm1$: a
diagonal similarity with the same spectrum but different matrix elements.

## Hamiltonian

$$
H = \sum_{pq,\sigma} h_{pq}\, a^\dagger_{p\sigma} a_{q\sigma}
  + \tfrac12 \sum_{pqrs,\sigma\tau} (pq|rs)\,
    a^\dagger_{p\sigma} a^\dagger_{r\tau} a_{s\tau} a_{q\sigma}
  + E_\text{core},
$$

$h$ symmetric (inactive-core potential folded in), $(pq|rs)$ chemist
notation with 8-fold symmetry, $E_\text{core}$ added on the diagonal only.

## Slater–Condon rules as implemented

`general_element(ints, A', B', A, B)` returns $\langle A',B'|H - E_\text{core}|A,B\rangle$
(ket $= (A,B)$) classified by the bit differences $d_\alpha = |A'\oplus A|$,
$d_\beta = |B'\oplus B|$.

**Diagonal** ($d_\alpha=d_\beta=0$, `diagonal_element`):

$$
\sum_{p\in A\cup B} h_{pp}
+ \sum_{\substack{p<q\\ p,q\in A}} \big[(pp|qq)-(pq|qp)\big]
+ \sum_{\substack{p<q\\ p,q\in B}} \big[(pp|qq)-(pq|qp)\big]
+ \sum_{p\in A,\,q\in B} (pp|qq).
$$

**Single** ($d=2$ in one channel, `single_excitation_same_spin`): electron
$i\to a$ in channel $\sigma$, ket string $s$ of that channel, other-channel
string $\bar s$:

$$
\langle D'|H|D\rangle = \epsilon_1
\Big[h_{ai} + \sum_{p\in s\setminus\{i\}} \big((ai|pp)-(ap|pi)\big)
     + \sum_{p\in\bar s} (ai|pp)\Big],
\qquad
\epsilon_1 = \text{pb}(s,i)\;\text{pb}(s\setminus i,\,a),
$$

with $\text{pb}$ = `phase_below`; $\epsilon_1 = \langle D'|a^\dagger_a a_i|D\rangle$.

**Same-spin double** ($d=4$ in one channel, `double_same_spin`): $i<j$
removed, $a<b$ added:

$$
\langle D'|H|D\rangle = \epsilon_2\big[(ai|bj)-(aj|bi)\big],
\qquad
\epsilon_2 = \langle D'|a^\dagger_a a^\dagger_b a_j a_i|D\rangle ,
$$

evaluated by `double_phase_same_spin` as successive single-channel parities:
annihilate $i$, then $j$, then create $b$, then $a$ (right-to-left operator
application). The four terms of $\tfrac12\sum(pq|rs)\ldots$ that connect $D$
and $D'$ reduce to this pair by $a^\dagger_a a^\dagger_b = -a^\dagger_b a^\dagger_a$
and $a_j a_i = -a_i a_j$.

**Opposite-spin double** ($d_\alpha=d_\beta=2$, `double_opposite_spin`):
$i\to a$ in α, $j\to b$ in β:

$$
\langle D'|H|D\rangle = \epsilon_\alpha\,\epsilon_\beta\,(ai|bj),
\qquad
\epsilon_\alpha = \text{pb}(A,i)\,\text{pb}(A\setminus i,a),\;
\epsilon_\beta = \text{pb}(B,j)\,\text{pb}(B\setminus j,b).
$$

The $\sigma\tau = \alpha\beta$ and $\beta\alpha$ terms of $H$ each give
$\tfrac12(ai|bj)$; no exchange term exists.

All other classes ($d_\alpha+d_\beta>4$, or $d=2$ split across channels) are
zero. `sigma_naive` and `dense_hamiltonian` use `general_element` for every
pair and add $E_\text{core}$ on the diagonal; `hamiltonian_diagonal` uses
`diagonal_element` $+E_\text{core}$. Neither pair loop is called by
`run_cas_ci`; they are test and benchmark oracles.

## Direct sigma (`src/sigma.rs`)

With $E^\sigma_{pq} = a^\dagger_{p\sigma} a_{q\sigma}$ and the
anticommutator $a_{q\sigma} a^\dagger_{r\tau} = \delta_{qr}\delta_{\sigma\tau}
- a^\dagger_{r\tau} a_{q\sigma}$,

$$
H - E_\text{core} = \sum_{pq,\sigma} g_{pq} E^\sigma_{pq}
  + \tfrac12 \sum_{pqrs,\sigma} (pq|rs)\, E^\sigma_{pq} E^\sigma_{rs}
  + \sum_{pqrs} (pq|rs)\, E^\alpha_{pq} E^\beta_{rs},
\qquad
g_{pq} = h_{pq} - \tfrac12 \sum_r (pr|rq).
$$

The $\alpha\beta$ and $\beta\alpha$ products are merged using
$[E^\alpha, E^\beta]=0$ and $(pq|rs)=(rs|pq)$. Writing the CI vector as the
matrix $C(I_\alpha, I_\beta)$, $\sigma = \sigma_1 + \sigma_2 + \sigma_3 +
E_\text{core} C$ with

$$
\begin{aligned}
\sigma_1(I_\alpha,I_\beta) &= \sum_{J_\beta} \langle I_\beta|H_\beta|J_\beta\rangle\, C(I_\alpha,J_\beta), \\
\sigma_2(I_\alpha,I_\beta) &= \sum_{J_\alpha} \langle I_\alpha|H_\alpha|J_\alpha\rangle\, C(J_\alpha,I_\beta), \\
\sigma_3(I_\alpha,I_\beta) &= \sum_{pqrs} (pq|rs) \sum_{J_\alpha J_\beta}
  \langle I_\alpha|E_{pq}|J_\alpha\rangle \langle I_\beta|E_{rs}|J_\beta\rangle\, C(J_\alpha,J_\beta),
\end{aligned}
$$

$H_\sigma$ being the first two terms restricted to one spin. The
factorization $\langle I_\alpha I_\beta|E^\alpha_{pq}E^\beta_{rs}|J_\alpha J_\beta\rangle
= \langle I_\alpha|E_{pq}|J_\alpha\rangle\langle I_\beta|E_{rs}|J_\beta\rangle$ holds in the
α-before-β order because $E^\beta_{rs}$ carries $(-1)^{N_\alpha}$ twice.

Every factor comes from one table per spin channel: for each string $S$
and each occupied $q$ and each $p$ empty in $S$ or equal to $q$,

$$
E_{pq}|S\rangle = \epsilon\,|T\rangle,\qquad
\epsilon = \text{pb}(S,q)\;\text{pb}(S\setminus q,\,p),\qquad
T = (S\setminus q)\cup p,
$$

the same single-channel phase as $\epsilon_1$ above; there are
$k(n-k)+k$ entries per string with $k$ electrons. With real symmetric
integrals $\langle J|E_{pq}|I\rangle = \langle I|E_{qp}|J\rangle$ and $H$ is
symmetric, so all three contractions run over the excitations OF the output
string:

* $\sigma_2$, per row $I_\alpha$: $F(K) \mathrel{+}= \epsilon_1 g_{kl}$ for each
  $E_{kl}|I_\alpha\rangle=\epsilon_1|K\rangle$, and
  $F(J) \mathrel{+}= \tfrac12\epsilon_1\epsilon_2 (ij|kl)$ for each
  $E_{ij}|K\rangle=\epsilon_2|J\rangle$; then $\sigma(I_\alpha,:) \mathrel{+}= \sum_J F(J)\, C(J,:)$.
  $F(J) = \langle J|H_\alpha|I_\alpha\rangle$.
* $\sigma_1$: the same on $C^\mathsf{T}$ with the β table.
* $\sigma_3$, per β pair $rs$: gather $C'(J_\alpha,m) = \epsilon_m C(J_\alpha, J_{\beta,m})$
  over the β strings with $E_{rs}|J_{\beta,m}\rangle = \epsilon_m|I_{\beta,m}\rangle$; per row
  $I_\alpha$, $V(m) = \sum \epsilon\,(pq|rs)\,C'(J_\alpha,m)$ over
  $E_{pq}|I_\alpha\rangle = \epsilon|J_\alpha\rangle$ (which is $\langle I_\alpha|E_{qp}|J_\alpha\rangle = \epsilon$,
  and $(qp|rs)=(pq|rs)$); then $\sigma(I_\alpha, I_{\beta,m}) \mathrel{+}= V(m)$.

Cost per call is $O(N_\text{det}\, n_\text{exc}^\alpha n_\text{exc}^\beta)$
for $\sigma_3$, with $n_\text{exc} = k(n-k)+k$; no determinant pair is
visited. Rows of $\sigma$ are owned by one rayon task each, so the result
does not depend on the thread count. Scratch beyond the Davidson vectors:
four $N_\text{det}$ vectors (output, $C^\mathsf{T}$, $\sigma_1^\mathsf{T}$,
$C'$) and the tables, $16(n_\text{str}^\alpha n^\alpha_\text{exc} +
2 n_\text{str}^\beta n^\beta_\text{exc})$ bytes.

## Checks

| Check | What it constrains |
|---|---|
| `scripts/casci_slater_condon_phases_sympy.py` | JW matrices obey the CAR; the ascending-creation product is the bit-string basis vector with sign $+1$; a line-for-line port of `general_element` equals the explicit JW $H$ as an exact linear form in symbolic $h_{pq}$, $(pq\|rs)$ for every pair in $(n,N_\alpha,N_\beta)=(4,2,2),(5,3,2),(5,2,3),(5,3,3)$. Four sign mutations and the interleaved order each mismatch; the interleaved $H$ has the same spectrum. |
| `tests/direct_sigma_vs_naive.rs` | The direct `sigma` against `sigma_naive` (the `general_element` pair loop) on random vectors, element-wise to $10^{-12}$: random 8-fold-symmetric integrals in ten sectors with $N_\alpha \ne N_\beta$, empty and full channels; real H2O/STO-3G integrals for full CI (441 det), 8 electrons in 6 orbitals above a frozen O 1s (225 det) and four unequal-spin sectors; `hamiltonian_diagonal` against the dense diagonal; unit vectors against dense columns. |
| `tests/slater_condon_brute_force.rs` | The Rust `general_element`, `sigma` (direct) and `hamiltonian_diagonal` against an independent bit-loop second-quantized $H$ on random asymmetric integrals, element-wise to $10^{-12}$, sectors with odd and even $N_\alpha$, every excitation class exercised. |
| FCI energies vs PySCF (`src/lib.rs` tests, `tests/validation_casci.rs`) | Spectrum only: blind to sign conventions that are diagonal similarities, and to errors in elements that vanish by point-group symmetry in the test molecules. A $\pm1$ similarity inserted into `general_element` passes all of them; H2/STO-3G has no nontrivial sign at all. |
