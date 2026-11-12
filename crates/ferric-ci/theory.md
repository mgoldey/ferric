# ferric-ci: CAS-CI phases, Slater–Condon rules, RDMs and CASSCF

Scope: the CAS-CI Hamiltonian in `src/strings.rs`, `src/hamiltonian.rs`, the
direct sigma build in `src/sigma.rs`, the RDMs in `src/rdm.rs`, the NEVPT RDMs
in `src/rdm3.rs` and two-step CASSCF in `src/casscf.rs`.
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

## Reduced density matrices (`src/rdm.rs`)

Spin-summed RDMs of the real, normalized CI vector $C$ over the active
orbitals, in the index convention of PySCF `fci.direct_spin1.make_rdm12`:

$$
\gamma_{pq} = \langle C|E_{pq}|C\rangle ,\qquad
\Gamma_{pqrs} = \sum_{\sigma\tau}\langle C|a^\dagger_{p\sigma} a^\dagger_{r\tau}
                 a_{s\tau} a_{q\sigma}|C\rangle
              = \langle C|E_{pq}E_{rs}|C\rangle - \delta_{qr}\gamma_{ps},
$$

with $E_{pq} = E^\alpha_{pq} + E^\beta_{pq}$; the second form follows from
$a_{q\sigma}a^\dagger_{r\tau} = \delta_{qr}\delta_{\sigma\tau} - a^\dagger_{r\tau}a_{q\sigma}$.
`dm1[p,q]` $=\gamma_{pq}$ and `dm2[p,q,r,s]` $=\Gamma_{pqrs}$. With the
Hamiltonian above,

$$
E = E_\text{core} + \sum_{pq} h_{pq}\gamma_{pq}
  + \tfrac12\sum_{pqrs} (pq|rs)\,\Gamma_{pqrs}.
$$

Identities for an $N$-electron state: $\operatorname{tr}\gamma = N$,
$\sum_{pr}\Gamma_{pprr} = N(N-1)$, $\sum_r \Gamma_{pqrr} = (N-1)\gamma_{pq}$,
$\Gamma_{pqrs} = \Gamma_{rspq}$, and for a real state $\gamma = \gamma^\mathsf{T}$,
$\Gamma_{pqrs} = \Gamma_{qpsr}$; the eigenvalues of $\gamma$ lie in $[0,2]$.

Construction. For each pair $rs$ form $D_{rs} = E_{rs}C$ from the excitation
tables of the direct sigma: a table entry $E_{pq}|I\rangle = \epsilon|T\rangle$
gives $\langle I|E_{qp}|T\rangle = \epsilon$, so

$$
D_{qp}(I_\alpha,I_\beta) \mathrel{+}= \epsilon\, C(T,I_\beta)\ \ (\alpha\text{ entry of }I_\alpha),
\qquad
D_{qp}(I_\alpha,I_\beta) \mathrel{+}= \epsilon\, C(I_\alpha,T)\ \ (\beta\text{ entry of }I_\beta).
$$

A single $E^\beta$ carries $(-1)^{N_\alpha}$ twice, so the β phase is the
single-channel one. Then, using $E_{pq}^\dagger = E_{qp}$,

$$
\gamma_{rs} = \sum_I C(I)\,D_{rs}(I),\qquad
\langle C|E_{pq}E_{rs}|C\rangle = \sum_I D_{qp}(I)\,D_{rs}(I),
$$

one $n^2\times n^2$ GEMM $DD^\mathsf{T}$. Writing $D = D^\alpha + D^\beta$, the
αα, ββ and αβ (cross) parts of $\Gamma$ are the blocks of that product; none is
assembled separately. The determinants are processed in fixed blocks of α rows
(about 4096 determinants), each block's $D$ built, contracted and dropped, and
block results summed in block order, so the result does not depend on the
thread count. Cost $O(N_\text{det}\,n^4)$ for the GEMM plus
$O(N_\text{det}(n^\alpha_\text{exc}+n^\beta_\text{exc}))$ for $D$; scratch one
$n^2\times$block $D$ per worker and 64 block results of $n^4+n^2$ doubles,
charged to the CAS-CI memory plan when `compute_rdms` is set.

## NEVPT RDMs (`src/rdm3.rs`)

Inputs of strongly contracted NEVPT2: the normal-ordered 3-RDM, two
contractions of the 4-RDM with the active $(aa|aa)$ block, and hole RDMs.
Definitions (PySCF's, so that `pyscf.mrpt.nevpt2` formulas port directly):

$$
\Gamma^{(3)}_{p_1q_1p_2q_2p_3q_3} = \sum_{\sigma\tau\upsilon}
  \langle a^\dagger_{p_1\sigma}a^\dagger_{p_2\tau}a^\dagger_{p_3\upsilon}
  a_{q_3\upsilon}a_{q_2\tau}a_{q_1\sigma}\rangle ,
$$

$$
f^\text{ca}_{abcdef} = \sum_{ijk}(ki|fj)\,\langle E_{ab}E_{cd}E_{ej}E_{ki}\rangle ,\qquad
f^\text{ac}_{abcdef} = \sum_{ijk}(ik|je)\,\langle E_{ab}E_{cd}E_{jf}E_{ik}\rangle ,
$$

$$
\eta^{(k)}_{p_1..p_k,a_1..a_k} = \sum_\text{spins}
  \langle a_{p_1}\cdots a_{p_k}\,a^\dagger_{a_k}\cdots a^\dagger_{a_1}\rangle .
$$

$\Gamma^{(3)}$ is `fci.rdm.make_dm1234` + `reorder_dm1234`. $f^\text{ca}$,
$f^\text{ac}$ are `mrpt.nevpt2._contract4pdm` with kernels
`NEVPTkern_cedf_aedf`, `NEVPTkern_aedf_ecdf`: contractions of the E-product
4-RDM, not the normal-ordered one. The einsum comments in PySCF's `make_a16`
and `make_a22` that list the 4-RDM in the argument order above agree with the
kernels to $4\times10^{-14}$ on random states; the comment `kbij,rpqjkiac`
above the first `f3ca` use in `make_a16` does not (difference $\sim 40$).
$\eta^{(k)}$ is `make_hdm1/2/3` fed with the E-product RDMs of
`fci.rdm.make_dm123` (PySCF's NEVPT2 driver passes those, not the reordered
ones); fed with the reordered ones it matches no operator ordering. The
operator forms above were identified by exhaustive index permutation against
dense Fock-space expectation values.

Construction. With $X_{rs} = E_{rs}C$ (the $D$ of the 2-RDM build) and
$\langle C|E_{ab}E_{cd} = (E_{dc}E_{ba}C)^\mathsf{T}$,

$$
\langle E_{ab}E_{cd}E_{ef}\rangle = \langle E_{dc}X_{ba}|X_{ef}\rangle ,\quad
f^\text{ca}_{abcdef} = \langle E_{dc}X_{ba}|R_{ef}\rangle ,\quad
f^\text{ac}_{abcdef} = \langle E_{dc}X_{ba}|R'_{ef}\rangle ,
$$

with $W_{fj} = \sum_{ki}(ki|fj)X_{ki}$, $R_{ef} = \sum_j E_{ej}W_{fj}$,
$R'_{ef} = \sum_j E_{jf}W_{ej}$. $X$, $W$, $R$, $R'$ are $n^2$ vectors over all
determinants ($W$ one GEMM); $L_{dc,ba} = E_{dc}X_{ba}$ is built for one block
of α rows at a time (about $2^{21}/n^4$ determinants) and contracted as
$L^\mathsf{T}[X|R|R']$, $n^4\times3n^2$. Blocks are summed in block order.
No 4-RDM is formed. Then (PySCF `reorder_dm123`)

$$
\Gamma^{(3)}_{pqrstu} = \langle E_{pq}E_{rs}E_{tu}\rangle
 - \delta_{qr}\Gamma_{pstu} - \delta_{st}\Gamma_{pqru} - \delta_{qt}\Gamma_{purs}
 - \delta_{qr}\delta_{st}\gamma_{up}.
$$

Cost $O(3N_\text{det}n^6)$; memory $4n^2N_\text{det} + 6n^6$ resident plus one
$L$ block and its $3n^6$ result per worker, reserved through a `MemoryPlan`
before any allocation (`nevpt_rdm_memory_elems`).

Identities: $\sum_x\Gamma^{(3)}_{pqrsxx} = (N-2)\Gamma_{pqrs}$,
$\sum_x\eta^{(3)}_{pqx,abx} = (2n-N-2)\eta^{(2)}_{pq,ab}$,
$\sum_x\eta^{(2)}_{px,ax} = (2n-N-1)\eta^{(1)}_{pa}$; for one determinant
every RDM is the antisymmetrized product of the spin-resolved $\gamma$
(particles) or $\eta = 1-\gamma^\mathsf{T}$ (holes). Both are blind to a
wrong sign or index in the cumulant, which vanishes for one determinant and
is summed away by a trace.

Dense fallback (`src/lib.rs`). When Davidson stops with $\|r\|\ge$
`conv_thresh` and $N_\text{det}\le$ `DENSE_SOLVE_MAX_DETS` (1024), the dense
$H$ is diagonalized (LAPACK `eigh`) and its root kept if its residual is
smaller; `converged` is then $\|r\| <$ `conv_thresh`.

## Two-step CASSCF (`src/casscf.rs`)

Orbitals $C = [C_c\,|\,C_a\,|\,C_v]$: $n_c$ doubly occupied core orbitals
$i,j$, $n_a$ active $t,u,v,w$, virtual $a$; $p,q,r,s$ run over all MOs.
Real orbitals, single root, spin-summed RDMs of the section above.
Reference implementation: `scripts/prototypes/casscf_proto.py` with
`scripts/prototypes/casscf_derivation.md`.

**Energy.** With $D^c = 2C_cC_c^\mathsf{T}$ and $D^a = C_a\gamma C_a^\mathsf{T}$,

$$
F^I = h + J[D^c] - \tfrac12 K[D^c],\qquad
F^A = J[D^a] - \tfrac12 K[D^a],\qquad
E_\text{core} = E_\text{nuc} + \tfrac12\operatorname{Tr}[D^c(h + F^I)],
$$

the CAS-CI runs on $h_\text{eff} = C_a^\mathsf{T}F^IC_a$, $(tu|vw)$ and
$E_\text{core}$ (`run_cas_ci_on_integrals`). $J$ and $K$ are contracted
from a dense AO block, $(pu|vw)$ for all $p$ by four quarter transforms of
that block (`integrals::transform_eri_mixed`).

**Gradient.** The rotation is $C' = C\exp(K)$, $K_{pq} = \kappa_{pq} = -K_{qp}$
for $p > q$. With the generalized Fock matrix (MO basis, $F^I$ and $F^A$
transformed with $C$)

$$
F_{pi} = 2(F^I_{pi} + F^A_{pi}),\qquad
F_{pt} = \sum_u F^I_{pu}\gamma_{ut} + \sum_{uvw}(pu|vw)\,\Gamma_{tuvw},\qquad
F_{pa} = 0,
$$

$\partial E/\partial\kappa_{pq} = 2(F_{pq} - F_{qp})$. The nonredundant pairs
are core-active $(t,i)$, core-virtual $(a,i)$, where the gradient is
$4(F^I_{ai} + F^A_{ai})$, and active-virtual $(a,t)$, where it is $2F_{at}$.
For any normalized CI vector $c$ with residual $r = Hc - \theta c$,
$2(F_{pq} - F_{qp}) = \langle c|[H, E_{pq} - E_{qp}]|c\rangle = -2\,c^\mathsf{T}(E_{pq}-E_{qp})\,r$,
so a redundant element is bounded by $8\lVert r\rVert$ ($\lVert E_{pq}\rVert \le 2$
for $p\ne q$).

**Step.** Inverse BFGS in the current orbital frame, no transport of earlier
steps. The initial and reset inverse Hessian is $1/\max(|H_\text{diag}|, 0.05)$
with the approximate diagonal of Chaban, Schmidt and Gordon, Theor. Chem.
Acc. 97, 88 (1997), $F^T = F^I + F^A$:

| Pair | $H_\text{diag}$ |
|---|---|
| $(a,i)$ | $4(F^T_{aa} - F^T_{ii})$ |
| $(a,t)$ | $2\gamma_{tt}F^T_{aa} - 2F_{tt}$ |
| $(t,i)$ | $4(F^T_{tt} - F^T_{ii}) + 2\gamma_{tt}F^T_{ii} - 2F_{tt}$ |

Its median relative error against the finite-difference diagonal is 24-28 %
(prototype measurement), so it is a preconditioner, not a Hessian. The update
skips when $s^\mathsf{T}y \le 10^{-14}$; the inverse resets to the diagonal
when the energy rises by more than $10^{-10}$ Ha; $\max_k|s_k|$ is capped at
0.2. $\exp(K)$ is a Taylor series after scaling $K$ to $\infty$-norm $\le \tfrac12$,
then squaring (`expm_antisymmetric`). Convergence: $\lVert g\rVert_2 < 10^{-7}$
and $|\Delta E| < 10^{-12}$ Ha at the same iteration (defaults). Every CI solve
must reach the CI residual threshold ($10^{-10}$ by default); the
gradient error is first order in that residual, so a solve that misses it
is an error, not a warning.

**Open-shell active spaces.** $H$ is spin-free, so the energy and the
gradient above depend on the spin-summed $\gamma$ and $\Gamma$ only, for any
sector $(n_\alpha, n_\beta)$. The CI root is the lowest of that $M_s$ sector,
and the orbitals stay one restricted set with a doubly occupied core. Swapping
$n_\alpha$ and $n_\beta$ gives the same energy and RDMs ($M_s \to -M_s$).

**Canonical orbitals** (`canonicalize`, `CasScfConfig::canonicalize`). With
$F^T = F^I + F^A$ in the MO basis, $U$ is block diagonal: the eigenvectors of
the core block of $F^T$ (ascending), of $\gamma$ (occupations descending) and
of the virtual block of $F^T$ (ascending). $C' = CU$, each column signed by its
first AO coefficient within a relative $10^{-6}$ of the largest magnitude
(the first index breaks ties between symmetry-equivalent atoms). Every
rotation in $U$ is redundant, so $E$ and $\lVert g\rVert$ are unchanged. The
CAS-CI is solved again in $C'$, which carries the CI vector into the new
orbitals: $D^c$ is unchanged by the core rotation and $D^a = C_a\gamma
C_a^\mathsf{T}$ by the active rotation, so $F^I$ and $F^A$ in $C'$ are the
rotated matrices up to the CI error of the second solve. `mo_energy` is the
diagonal of $F^T$ in $C'$.

**Limits.** Lowest root, no state averaging; dense AO block,
$n_\text{bas}^4$ doubles, charged to the memory budget; the start is a
restricted RHF (an even electron count). No exact orbital Hessian,
orbital-CI coupling or augmented-Hessian step.

**Measured (2026-11-11, `tests/casscf.rs`).** Geometries
`testdata/molecules/water.xyz` and `n2.xyz`, 6-31G, ferric exact-J/K RHF
start, Davidson threshold $10^{-10}$ ($10^{-11}$ for the FD anchor).

Anchor 1, active space = full space (STO-3G), against `run_cas_ci` over the
full window in the same $(n_\alpha, n_\beta)$ sector; the RDMs of the sector
also rebuild the energy (bar of `tests/casci_rdms.rs`). The H2O $M_s = 1$
sector's lowest root is a triplet above the singlet ground state, so a CI run
in another sector misses `run_cas_ci` by the gap:

| System | Sector | $N_\text{det}$ | $\Delta E$ (Ha) | max $\lvert 2(F - F^\mathsf{T})\rvert$ | bar $8\lVert r\rVert$ + rounding |
|---|---|---:|---:|---:|---:|
| H2O | (5, 5) | 441 | 9.9e-14 | 7.6e-11 | 1.9e-9 |
| H2O | (6, 4) | 245 | 2.8e-14 | 7.7e-11 | 1.2e-9 |
| N2 | (7, 7) | 14400 | 9.9e-14 | 6.2e-12 | 4.5e-10 |
| O2 | (9, 7) | 1200 | -2.8e-14 | 1.8e-11 | 3.6e-10 |

Anchor 2, analytic gradient against a Richardson-extrapolated 5-point central
difference ($h = 10^{-3}$, $h/2$) at RHF orbitals rotated by a random
$\kappa$ (scale 0.05), max over each class, bar $10^{-8}$:

| System | Row | core-active | core-virtual | active-virtual |
|---|---|---:|---:|---:|
| H2O CAS(4,4) | max $\lvert g\rvert$ | 6.28 | 5.58 | 0.834 |
| | FD($h$) − FD($h/2$) | 2.0e-7 | 1.7e-10 | 1.2e-10 |
| | analytic − FD | 2.4e-10 | 1.2e-10 | 1.5e-10 |
| N2 CAS(6,6) | max $\lvert g\rvert$ | 2.07 | 6.98 | 0.800 |
| | FD($h$) − FD($h/2$) | 1.4e-9 | 2.0e-10 | 1.9e-10 |
| | analytic − FD | 1.8e-10 | 2.0e-10 | 1.6e-10 |
| H2O CAS(4,4), (3, 1) | max $\lvert g\rvert$ | 4.41 | 5.62 | 0.942 |
| | FD($h$) − FD($h/2$) | 3.7e-8 | 2.5e-10 | 1.3e-10 |
| | analytic − FD | 2.6e-10 | 1.8e-10 | 1.1e-10 |
| O2 CAS(8,6), (5, 3) | max $\lvert g\rvert$ | 2.47 | 9.13 | 0.991 |
| | FD($h$) − FD($h/2$) | 2.5e-9 | 4.1e-10 | 5.3e-10 |
| | analytic − FD | 4.2e-10 | 4.6e-10 | 4.2e-10 |

The finite-difference check uses the same CI sector as the analytic gradient,
so it cannot see a CI run in the wrong sector; anchor 1 and the PySCF
comparison below do.

Anchor 3, converged energies against the prototype's committed values, bar
$(\lVert g\rVert^2 + \lVert g_\text{proto}\rVert^2)/(2\lambda_\text{min}) + 8\varepsilon|E|\sqrt{N_\text{det}}$,
with $\lambda_\text{min}$ the lowest eigenvalue of the finite-difference
CI-relaxed orbital Hessian at the converged orbitals:

| Quantity | H2O CAS(4,4)/6-31G | N2 CAS(6,6)/6-31G |
|---|---:|---:|
| $E$ (Ha) | -76.0369690965018 | -109.0155467897005 |
| $E$ prototype (Ha) | -76.0369690965018 | -109.0155467897007 |
| $\Delta E$ (Ha) | -4.3e-14 | 2.3e-13 |
| bar (Ha) | 1.1e-12 | 3.9e-12 |
| $\lVert g\rVert$ / prototype | 4.2e-8 / 9.5e-8 | 3.4e-8 / 3.4e-8 |
| $\lambda_\text{min}$ (Ha) / note | 0.0195 / 0.0195 | 0.0402 / 0.0402 |
| macro iterations / prototype | 60 / 62 | 21 / 21 |

The prototype's energies match `pyscf.mcscf.CASSCF` on ferric's AO integrals
to 1.4e-12 Ha (H2O) and 1.8e-13 Ha (N2).

Anchor 4 (`tests/casscf_vs_pyscf.rs`), converged energies against
`pyscf.mcscf.CASSCF` run on ferric's AO integrals from PySCF's own RHF
orbitals (`scripts/gen_casscf_refs.py`, PySCF 2.13.1, `conv_tol` 1e-12,
`conv_tol_grad` 1e-6, FCI `conv_tol` 1e-14; references in
`testdata/reference/casscf_*_pyscf.json`). Ferric runs from its own RHF with
`canonicalize`. Bar $(\lVert g\rVert^2 + \lVert g_\text{ref}\rVert^2)/(2\lambda_\text{min})$
+ `conv_tol` + rounding, $\lVert g_\text{ref}\rVert$ = twice PySCF's packed
norm. Natural occupations and the sorted eigenvalues of the core and virtual
blocks of $F^I + F^A$ are compared and not gated; they move to first order in
the orbital error, which PySCF's gradient sets.

| System | $E$ (Ha) | $\Delta E$ vs PySCF (Ha) | bar (Ha) | $\lVert g\rVert$ / $2\lVert g_\text{PySCF}\rVert$ | $\lambda_\text{min}$ | iterations | natural occupations | max $\Delta$ occupation / core / virtual $\varepsilon$ |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| H2O CAS(4,4) | -76.0369690965018 | -1.5e-11 | 1.0e-10 | 4.2e-8 / 2.0e-6 | 0.0195 | 60 | 1.978, 1.975, 0.024, 0.023 | 2.2e-7 / 1.6e-5 / 1.6e-6 |
| N2 CAS(6,6), 1.0977 Å | -109.0155467897006 | -9.9e-14 | 5.2e-12 | 3.4e-8 / 1.5e-7 | 0.0402 | 21 | 1.981, 1.935 (2), 0.065 (2), 0.019 | 8.9e-8 / 8.7e-7 / 1.9e-7 |
| N2 CAS(6,6), 1.6 Å | -108.8482292478625 | -5.8e-13 | 5.1e-12 | 7.8e-8 / 2.5e-7 | 0.1485 | 12 | 1.897, 1.723 (2), 0.277 (2), 0.102 | 7.0e-7 / 2.1e-7 / 6.1e-8 |
| O2 CAS(8,6), (5, 3) | -149.6367298537082 | -2.6e-13 | 4.0e-12 | 5.9e-8 / 1.3e-7 | 0.0958 | 14 | 1.960 (2), 1.955, 1.039 (2), 0.046 | 5.5e-8 / 1.9e-7 / 9.3e-8 |

All four reach the same minimum as PySCF. PySCF's O2 state has
$\langle S^2\rangle = 2$. The H2O bar is set by PySCF's gradient
(`conv_tol_grad` 1e-7 does not converge H2O in 300 macro iterations).

Canonicalization (`tests/casscf_canonical.rs`) at ferric's converged
orbitals $C_0$; $C_1$ the canonical orbitals, the CI solved again in $C_1$;
$\delta\gamma = \gamma_1 - U_a^\mathsf{T}\gamma_0U_a$; bars derived in the
test's module documentation:

| Quantity | H2O CAS(4,4) | N2 CAS(6,6), 1.6 Å | O2 CAS(8,6), (5, 3) |
|---|---:|---:|---:|
| $\lvert E_1 - E_0\rvert$ (bar) | 1.4e-14 (8.1e-11) | 5.7e-14 (5.7e-11) | 2.8e-14 (1.1e-10) |
| $\lvert E_\text{RDM} - E_1\rvert$ (bar) | 1.4e-14 (1.4e-11) | 2.8e-14 (3.1e-11) | 2.8e-14 (7.6e-11) |
| $\lVert H_1c - E_1c\rVert$ of the returned CI vector | 1.2e-11 | 2.7e-11 | 7.0e-11 |
| max $\lvert\delta\gamma\rvert$ (bar $2N(\lVert r_0\rVert+\lVert r_1\rVert)/\Delta$) | 4.4e-11 (8.2e-10) | 5.2e-12 (9.8e-9) | 2.4e-11 (7.5e-9) |
| max off-diagonal $F^T$, core and virtual (bar) | 9.1e-13 (9.2e-11) | 3.7e-13 (2.1e-11) | 4.5e-11 (6.3e-11) |
| idempotence, worst $\lvert O_{pq}\rvert$ / bar; unconstrained pairs | 0.50; 0 | 0.46; 8 | 0.50; 0 |

**Mutations** (applied to `src/casscf.rs` one at a time, 2026-11-11; max
$\lvert$analytic − FD$\rvert$ per class, core-active / core-virtual /
active-virtual):

| Mutation | H2O | N2 | Anchor 1 | Anchor 3 |
|---|---|---|---|---|
| $F_{pi} = 2F^I_{pi}$ ($F^A$ dropped) | 1.94 / 1.57 / 1.5e-10 | 1.39 / 1.34 / 1.6e-10 | pass (no core) | not converged in 300 |
| $\Gamma$ term of $F_{pt}$ dropped | 0.567 / 1.2e-10 / 1.91 | 1.17 / 2.0e-10 / 1.71 | fail (6.7, 1.7) | not converged in 300 |
| core-virtual gradient sign flipped | 2.4e-10 / 11.2 / 1.5e-10 | 1.8e-10 / 14.0 / 1.6e-10 | pass (no core) | not converged in 300 |

Each mutation fails the classes whose formula it changes and leaves the
others at the unmutated error.

Mutations of canonicalization and of the open-shell path (applied to
`src/casscf.rs` one at a time, 2026-11-11):

| Mutation | Result |
|---|---|
| active orbitals rotated, CI vector and RDMs of the old orbitals kept | `casscf_canonical.rs` fails for all three systems: $\lvert E_\text{RDM} - E_1\rvert$ = 3.9e-2 / 0.29 / 0.16 Ha, $\lVert H_1c - E_1c\rVert$ = 0.31 / 0.63 / 0.52, max $\lvert\delta\gamma\rvert$ = 1.7e-2 / 1.3e-10 / 5.1e-3 (H2O / N2 1.6 Å / O2) |
| $F^A$ dropped from the canonical Fock ($F^I$ only) | `casscf_canonical.rs` fails for all three: max off-diagonal $F^T$ 0.26 / 0.20 / 0.37 against bars near 1e-10; the idempotence check passes (both calls use the same matrix) |
| CI in the sector $(n_\alpha - 1, n_\beta + 1)$ when $n_\alpha \ne n_\beta$ | H2O (6, 4) full space fails, $\Delta E$ = -0.398 Ha (the singlet). O2 (9, 7) full space, O2 against PySCF and the H2O (3, 1) finite-difference check pass: the O2 $M_s = 0$ sector's lowest root is the $M_s = 0$ component of the same triplet, and the finite difference runs in the same wrong sector. The O2 finite-difference test errors (a Davidson solve in the near-degenerate $M_s = 0$ sector misses its threshold) and the O2 canonicalization test panics on the CI vector length; neither is an assertion |

The spin-sector mutation is caught only where the requested sector's lowest
state is not the lowest state of the neighbouring sector; the H2O (6, 4)
full-space case is that check.

## Checks

| Check | What it constrains |
|---|---|
| `scripts/casci_slater_condon_phases_sympy.py` | JW matrices obey the CAR; the ascending-creation product is the bit-string basis vector with sign $+1$; a line-for-line port of `general_element` equals the explicit JW $H$ as an exact linear form in symbolic $h_{pq}$, $(pq\|rs)$ for every pair in $(n,N_\alpha,N_\beta)=(4,2,2),(5,3,2),(5,2,3),(5,3,3)$. Four sign mutations and the interleaved order each mismatch; the interleaved $H$ has the same spectrum. |
| `tests/direct_sigma_vs_naive.rs` | The direct `sigma` against `sigma_naive` (the `general_element` pair loop) on random vectors, element-wise to $10^{-12}$: random 8-fold-symmetric integrals in ten sectors with $N_\alpha \ne N_\beta$, empty and full channels; real H2O/STO-3G integrals for full CI (441 det), 8 electrons in 6 orbitals above a frozen O 1s (225 det) and four unequal-spin sectors; `hamiltonian_diagonal` against the dense diagonal; unit vectors against dense columns. |
| `tests/slater_condon_brute_force.rs` | The Rust `general_element`, `sigma` (direct) and `hamiltonian_diagonal` against an independent bit-loop second-quantized $H$ on random asymmetric integrals, element-wise to $10^{-12}$, sectors with odd and even $N_\alpha$, every excitation class exercised. |
| `tests/casci_rdms.rs` | The RDM energy equals $c\cdot Hc$ from the direct sigma to rounding and the Davidson eigenvalue to rounding plus the residual norm; the traces, partial trace and symmetries above; `make_rdm12` (PySCF) of PySCF's dense ground state on ferric's integrals, rotated into the run's orbitals. A dropped β phase, the missing αβ cross terms and a doubled diagonal each fail it. |
| `tests/casci_rdm3.rs` | NEVPT RDMs on ferric's CI vectors for H2O/6-31G CAS(4,4), N2/cc-pVDZ CAS(6,6) at 1.10 and 2.20 Å (`scripts/gen_casci_rdm3_refs.py`: `make_dm1234` + `reorder_dm1234`, `_contract4pdm`, `make_hdm*` on PySCF's side of the same vector). Anchors first: partial traces (dm3→dm2 ≤ 8.9e-15, hdm3→hdm2 ≤ 1.7e-14) and single determinants (exact zero difference). Against PySCF, element-wise at bar $100N_\text{det}\varepsilon\max|\cdot|$: dm1..dm3 and hdm1..hdm3 ≤ 6e-15 (bars 1.3e-12..6.8e-11); `f3ca`/`f3ac` element-wise 7.1e-15 (H2O, bar 1.8e-11) and through 32 random ±1 projections ≤ 2.8e-12 (bars 7e-8..1e-5). An independent dense Fock-space construction (H2O) gives dm3 5e-16, hdm3 3.5e-15, and `f3ca`/`f3ac` as explicit contractions of a stored E-product 4-RDM to 1.1e-14. Dense fallback: Davidson stopped at max_iter 1 (residual 3.9e-2), conv_thresh 1e-13 met at residual 6.1e-14; dense vs Davidson root on N2 1.10 Å $|\Delta E|$ 1.4e-13 (bar 4.5e-10). Mutations that fail it: β Jordan–Wigner phase forced to +1 (n2_r2.20 dm3 2.32), one index swap in the reorder (0.82), same-spin block omitted (2.84), a swapped destination index of $R'$ (n2_r2.20 f3ac projections 1.7e2), hdm3 coefficient 2→1, an uncommitted MemoryPlan (starved budget not refused), the dense fallback disabled (residual stays 3.9e-2). |
| `tests/casscf.rs` | CASSCF: full-space limit (zero $2(F-F^\mathsf{T})$, FCI energy, RDM energy) in closed- and open-shell sectors, the analytic orbital gradient against finite differences per rotation class, converged energies against the prototype. Mutation record in the CASSCF section above. |
| `tests/casscf_vs_pyscf.rs` | Converged CASSCF energies against `pyscf.mcscf.CASSCF` on ferric's AO integrals from PySCF's own RHF (`scripts/gen_casscf_refs.py`), four systems including an open-shell sector. |
| `tests/casscf_canonical.rs` | Canonical orbitals: energy, RDM energy and CI eigenvector in the new orbitals, natural orbitals, diagonal core and virtual Fock blocks, idempotence. |
| FCI energies vs PySCF (`src/lib.rs` tests, `tests/validation_casci.rs`) | Spectrum only: blind to sign conventions that are diagonal similarities, and to errors in elements that vanish by point-group symmetry in the test molecules. A $\pm1$ similarity inserted into `general_element` passes all of them; H2/STO-3G has no nontrivial sign at all. |
