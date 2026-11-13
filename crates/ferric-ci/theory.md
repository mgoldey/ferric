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
$E_\text{core}$ (`run_cas_ci_on_integrals`). The integrals come from
`CasScfConfig::integrals` (`casscf_ints.rs`). The default, `Auto`, takes
the dense route when its $n_\text{bas}^4$ block and transients fit in half of
the memory budget and the direct route otherwise. The direct route
builds $J$ and $K$ with ferric-scf's direct builder (`build_jk_with_pool`,
Schwarz and density screened) and $(pu|vw)$ for all $p$ by a direct half
transform: per canonical bra shell pair, every ket shell pair with
$Q_\text{bra}Q_\text{ket} \ge$ `thresh` is contracted with $C_a$ on both ket
indices, $X_{mn,vw} = \sum_{ls}(mn|ls)C_{lv}C_{sw}$, then
$(pu|vw) = \sum_{mn}C_{mp}C_{nu}X_{mn,vw}$ by GEMMs. Peak memory is
$X$, $n_\text{bas}^2n_a^2$ doubles. The dense route holds the
$n_\text{bas}^4$ AO block, contracts $J$, $K$ from it and transforms it by four
quarter transforms (`integrals::transform_eri_mixed`); it is the test oracle
(`tests/casscf_direct.rs`: at `thresh = 0` the routes agree to
$10^{-12}$ in $F$, $(pu|vw)$, $E$ and $g$).

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

**Augmented-Hessian step** (`optimizer = "ah"`, `src/casscf_newton.rs`).
The exact orbital Hessian at fixed RDMs, applied as a product. For
$C(\varepsilon) = C\exp(\varepsilon X)$, every quantity above in the
rotated orbitals has the frame derivative

$$
\begin{aligned}
dF^I &= F^IX - XF^I + C^\mathsf{T}(J - \tfrac12K)[dD^c]C, &
dD^c &= 2(CX)_cC_c^\mathsf{T} + \text{transpose},\\
dF^A &= F^AX - XF^A + C^\mathsf{T}(J - \tfrac12K)[dD^a]C, &
dD^a &= (CX)_a\gamma C_a^\mathsf{T} + \text{transpose},
\end{aligned}
$$

$d(pu|vw) = \sum_r X_{rp}(ru|vw) + X_{ru}(pr|vw) + X_{rv}(pu|rw) +
X_{rw}(pu|rv)$, and $H_fx = \operatorname{pack}2(dF - dF^\mathsf{T})$ is the
derivative of the fixed-RDM gradient along $x$ in the rotated frame. The
$(pr|vw)$ and $(pu|rw)$ terms are contracted with $\Gamma$ once per macro
iteration into

$$
W_{(p,t),(r,s)} = \sum_{vw}(pr|vw)\Gamma_{tsvw}
 + \sum_{uw}(pu|rw)(\Gamma_{tusw} + \Gamma_{tuws}),
$$

the first sum being the active-active coupling through $(pr|vw)$. Since
$e^{\varepsilon X}e^{\delta A} = \exp(\varepsilon X + \delta A +
\tfrac12\varepsilon\delta[X, A] + \dots)$, the symmetric Hessian of
$E(C\exp K)$ at $K = 0$ is $Hx = H_fx - \tfrac12\operatorname{pack}[G, X]$,
$G = 2(F - F^\mathsf{T})$. Both density responses have the form
$dD = C(Y + Y^\mathsf{T})C^\mathsf{T}$ with $Y$ nonzero only in occupied
columns ($Y_{ri} = 2X_{ri}$ for core $i$, $Y_{rt} = (X_a\gamma)_{rt}$ for
active $t$), so in the MO basis
$C^\mathsf{T}(J - \tfrac12K)[dD]C_\text{occ} = L\,\mathrm{vec}(Y)$ with the
occupied kernel
$L_{(pj),(rk)} = 2(pj|rk) - \tfrac12(pr|jk) - \tfrac12(pk|rj)$, $j, k$ over
the core and active orbitals. $(pq|jk)$ and $(pj|qk)$ are built once per
macro iteration (dense: one first quarter shared by both chains, the
$(pj|qk)$ chain contracting $\nu$ with $C_\text{occ}$ second through the
$(\mu\nu|\lambda\sigma) = (\mu\nu|\sigma\lambda)$ symmetry of the stored
block, `transform_ppaa_papa`; direct: one half transform per bra shell
pair, all ket pairs); their active blocks give $W$ and
$\sum(pu|vw)\Gamma$. A product is then one GEMM of $L$
($(n_\text{mo}n_\text{occ})^2$) with two columns. The AO form, one $J/K$
pass over $dD^c$, $dD^a$ per product (`HessianProducts::AoJk`), is kept as
its oracle. The step is the lowest
eigenpair of $\begin{pmatrix}0 & g^\mathsf{T}\\ g & H\end{pmatrix}$ by
Davidson (preconditioner $\max(|H_\text{diag}|, \texttt{hdiag\_floor}) -
\lambda$, at most 40 products), $s = v/v_0$, stopped at
$\lVert(H - \lambda)s + g\rVert \le \eta\lVert g\rVert$, $\eta =
\min(0.1, \lVert g\rVert)$ by default (`CasScfConfig::ah_forcing`;
`AhForcing::Constant` holds $\eta$ fixed);
then scaled to a trust radius (2-norm; start 0.4, at most 1.0) and to
`max_step` ($\infty$-norm). A step that raises $E$ by more than
$10^{-10}$ Ha is halved and evaluated again (a CI solve each). The radius
halves below an actual/predicted ratio of 0.25 and doubles above 0.75 for a
step at the boundary, the predicted change being
$g\cdot s + \tfrac12 s\cdot Hs/\beta$. The Hessian lacks the CI response,
which lowers the curvature: near convergence the plain AH step undershoots
by a factor of about 1.6 (linear convergence), and on N2 CAS(10,10) it
stalls by orders of magnitude along one direction. Each accepted step
therefore measures the ratio of the model to the actual curvature along
itself, $\beta = s\cdot Hs / (2(\Delta E - g\cdot s))$ (100 when the actual
curvature is not positive), and the next AH step is multiplied by $\beta$
clamped to $[1, 100]$ before the trust and `max_step` scaling
(`curvature_stretch`). The prototype's `scale` variant used the energy ratio
clamped to $[1, 2]$ instead; in Rust that rule stalled N2 CAS(10,10) (below).
Multi-secant
SR1 corrections of $H$ from earlier steps were tried and were unstable for a
memory of three or more steps (water 62 iterations with 16 rejected steps).

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

**Limits.** Lowest root, no state averaging; the direct route recomputes
every shell quartet each macro iteration (three passes: $J/K[D^c]$,
$(pu|vw)$, $J/K[D^a]$); the start is a
restricted RHF (an even electron count). No orbital-CI coupling: both
optimizers are two-step, and the augmented-Hessian step uses the orbital
Hessian at fixed RDMs.

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

All four reach the same minimum as PySCF. The test asserts that the
point is a minimum: the lowest eigenvalue of the central-difference
CI-relaxed orbital Hessian at $h = 10^{-4}$ must exceed
$\tfrac23\lvert\lambda(2h) - \lambda(h)\rvert + n_\text{pair}\,\delta g/h$,
with $\delta g = 8r + 4\varepsilon n_\text{mo}^2 \max\lvert F\rvert$
($r$ the CI convergence threshold). Measured $\lambda(h)$ and $\lambda(2h)$
agree to six digits: 0.019457 (bar 4.4e-4), 0.040210 (8.4e-4), 0.148488
(8.4e-4) and 0.095834 (8.4e-4) for the four rows in order; no anchor is a
saddle. PySCF's O2 state has
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

**Integral routes (2026-11-13, `tests/casscf_direct.rs`).** H2O CAS(4,4),
N2 CAS(6,6) and O2 CAS(8,6) $(5, 3)$ on 6-31G, at RHF orbitals rotated by a
random $\kappa$ (scale 0.05), CI threshold $10^{-12}$. The direct route is
compared with the dense oracle at `thresh = 0` and at the default $10^{-12}$.
The bars on $F$ and $(pu|vw)$ are a priori: summation rounding of both
routes plus the screened quartets, carried through the coefficients by their
column 1-norms. The bars on $E$ and $g$ propagate the measured $\delta F$ and
$\delta P$, including the CI response through the dense CI gap. $(pu|vw)$ is
checked before any CI runs on it.

| System | thresh | $\delta F$ (bar) | $\delta(pu\vert vw)$ (bar) | $\lvert\Delta E\rvert$ (bar) | $\lvert\Delta g\rvert$ (bar) |
|---|---:|---:|---:|---:|---:|
| H2O CAS(4,4) | 0 | 9.4e-14 (1.1e-8) | 1.5e-14 (7.7e-11) | 2.8e-14 (1.8e-11) | 2.5e-13 (1.5e-6) |
| H2O CAS(4,4) | 1e-12 | 9.4e-14 (2.6e-8) | 1.5e-14 (7.8e-10) | 2.8e-14 (1.8e-11) | 2.5e-13 (1.5e-6) |
| N2 CAS(6,6) | 0 | 2.3e-13 (1.4e-6) | 1.7e-13 (6.5e-9) | 2.6e-13 (2.7e-10) | 7.7e-13 (7.9e-4) |
| N2 CAS(6,6) | 1e-12 | 2.3e-13 (1.6e-6) | 1.7e-13 (5.6e-8) | 2.6e-13 (2.7e-10) | 7.7e-13 (7.9e-4) |
| O2 CAS(8,6) | 0 | 3.3e-13 (2.6e-7) | 7.3e-14 (8.4e-10) | 4.3e-13 (1.6e-10) | 1.0e-12 (1.3e-3) |

The gradient bars are loose: they multiply worst-case element bounds over
every rotation class. The measured differences are at rounding.

Mutations (one at a time, 2026-11-13; each fails on an assertion):

| Mutation | Caught by |
|---|---|
| $F^A = J - \tfrac14K$ (half the exchange in the active Fock) | `casscf.rs`: analytic vs FD, core-active 0.245 (H2O) against $10^{-8}$; prototype energy $\lvert\Delta E\rvert$ = 4.7e-2 Ha (H2O). `casscf_direct.rs` passes: both routes share the formula |
| direct $K$ halved | `casscf_direct.rs`, 5 of 6 tests: $\delta F$ = 2.32 against $\le 1.6\times10^{-6}$ |
| direct $(pu\vert vw)$ without the $s_3 \ne s_4$ ket mirror | `casscf_direct.rs`, 5 of 6 tests: $\delta(pu\vert vw)$ = 0.86 to 8.5 against $\le 5.6\times10^{-8}$ |
| dense $K$ contracted with $D^\mathsf{T}$ (`column(l)`) | unit test `dense_jk_gemv_matches_the_scalar_loops` (nonsymmetric $D$): $\lvert\Delta K\rvert$ = 0.84 against 5.3e-11 |

The sixth `casscf_direct.rs` test checks that a negative or NaN threshold is
refused.

**Benzene CAS(6,6)/cc-pVDZ (2026-11-13).** `testdata/molecules/benzene.xyz`,
114 basis functions, 18 core orbitals, ferric RHF start, default settings
(`integrals = "auto"` resolves to dense at `budget_gb = 10`). CLI wall times,
paired and interleaved (A B B A A B), 12 rayon threads, box load average
2.5 to 10.4 during the runs:

| Build | Wall (s), 3 runs | mean ± sd (s) | macro iterations | wall per iteration (s) | peak RSS |
|---|---|---:|---:|---:|---:|
| before (dense scalar $J/K$, single-threaded transform) | 117.3, 110.5, 110.3 | 112.7 ± 4.0 | 90 | 1.25 ± 0.04 | 1.79 GB |
| dense route | 37.4, 35.9, 35.6 | 36.3 ± 1.0 | 89 | 0.41 ± 0.01 | 1.78 GB |

Both reach $E = -230.7814586093$ Ha. The wall times include the RHF
(about 9 s). The iteration counts differ by one because the $J/K$ sums round
differently. Phase totals of the dense route (`FERRIC_TIMING=1`, the median
run) and of one direct-route run (`thresh` $10^{-12}$, 88 iterations, peak
RSS 0.56 GB):

| Phase | dense (s) | busy cores | direct (s) | busy cores |
|---|---:|---:|---:|---:|
| inactive Fock ($J/K$ of $D^c$) | 6.0 | 11.3 | 93.2 | 11.2 |
| $(pu\vert vw)$ transform | 7.0 | 11.5 | 169.6 | 11.2 |
| CI solve | 2.4 | 11.7 | 1.6 | 11 |
| active Fock ($J/K$ of $D^a$) | 6.3 | 11.2 | 93.0 | 11.2 |
| BFGS update + step | 3.7 | 1.0 | 3.8 | 1.0 |
| other | 0.1 | | 0.1 | |
| total, per iteration | 25.5, 0.29 | | 361.4, 4.1 | |

**Optimizer choice (2026-11-13, round-2 task 14).** Macro iterations
(one CAS-CI each) and Hessian-vector products, from
`scripts/prototypes/casscf_newton_proto.py` on ferric's integrals and RHF
start; PySCF 2.13.1 `mcscf.CASSCF` (`conv_tol` 1e-12, `conv_tol_grad` 1e-6,
FCI `conv_tol` 1e-14, the settings of `scripts/gen_casscf_refs.py`) from
ferric's RHF and from its own. Water, N2 and O2 as in `tests/casscf.rs`
(6-31G); benzene CAS(6,6)/cc-pVDZ with 18 core orbitals; N2
CAS(10,10)/cc-pVDZ with 2 core orbitals.

| System | BFGS | AH | AH, ratio stretch | AH, η = 0.1 | AH, ratio stretch, η = 0.3 | Rust AH (shipped) | PySCF macro (from ferric / own RHF) | PySCF micro, inner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| water CAS(4,4) | 61 | 21 / 197 | 22 / 190 | 27 / 134 | 27 / 111 | 22 | 19 / 19 | 67, 312 |
| N2 CAS(6,6) | 21 | 10 / 57 | 11 / 55 | 12 / 32 | 13 / 24 | 11 | 8 / 8 | 25, 95 |
| O2 CAS(8,6) $M_s = 1$ | 14 | 7 / 31 | 8 / 35 | 8 / 19 | 10 / 15 | 8 | 8 / 8 | 23, 82 |
| benzene CAS(6,6) | 90 | 27 / 470 | 19 / 352 | 29 / 314 | 28 / 223 | 18 | 16 / 19 | 57, 285 |
| N2 CAS(10,10) | 36 | 39 / 343 | 26 / 191 | | 29 / 79 | 25 | 15 / 15 | 51, 230 |

Prototype entries are macro iterations / Hessian products; "AH" is the
step without a stretch, "ratio stretch" multiplies an unclipped step by the
previous energy ratio clamped to $[1, 2]$, with $\eta = \min(0.1, \lVert g\rVert)$ forcing; the η
columns stop the Davidson at the constant relative residual η. "Rust AH"
is the shipped step with the curvature stretch (macro iterations; BFGS in
Rust takes 60, 21, 14, 89 and 36). PySCF's macro iteration contains several
orbital micro-steps with approximate CI updates, so its macro count is not
the CI-solve count; ferric's two-step AH needs 1.2 to 1.7 times PySCF's macro
count. All AH variants on benzene stay at ferric's minimum
$-230.7814586093$ Ha; none reaches PySCF's $-230.7873347336$. The
$\eta = 0.1$ runs (with and without the stretch) stop at $-230.7814587989$,
$1.9\times10^{-7}$ Ha lower with $\lVert g\rVert = 8\times10^{-8}$, a nearby
stationary point that was not examined further.

**AH anchors (`tests/casscf_newton.rs`, `tests/casscf_direct.rs`,
2026-11-13).** 6-31G, RHF orbitals rotated by a random $\kappa$ (scale 0.05),
random unit $x$. $H_fx$ against the central difference of the fixed-RDM
gradient at $h/2 = 5\times10^{-4}$: worst error 0.50 of the bar on water,
N2 and O2 (the bar is $2|\Delta|/3$ plus rounding and the error is
$|\Delta|/3$, the leading truncation term, because $H_fx$ matches the
Richardson extrapolate to $1.5\times10^{-11}$, $1.2\times10^{-10}$ and
$2.3\times10^{-11}$). Symmetry of $H$: $|y^\mathsf{T}Hx - x^\mathsf{T}Hy|$
$7\times10^{-16}$ to $4\times10^{-15}$ against bars of $3$-$6\times10^{-11}$,
while the frame derivative alone is asymmetric by 0.06-0.79. AH and BFGS
minima: water $|\Delta E| = 1.4\times10^{-13}$ (bar $1.1\times10^{-12}$), O2
$1.4\times10^{-13}$ (bar $2.9\times10^{-12}$). Direct against dense $Hx$:
$8.4\times10^{-14}$, $2.6\times10^{-13}$ (O2, N2 at `thresh` $10^{-12}$)
against a priori bars of $2.8\times10^{-6}$ to $2.2\times10^{-3}$. With the
default switched to AH every existing CASSCF test passes (`casscf.rs`,
`casscf_vs_pyscf.rs`, `casscf_canonical.rs`, `casscf_direct.rs`,
`cas_ecp.rs`): water, N2, O2 and stretched N2 against PySCF within
$1.5\times10^{-11}$, $4\times10^{-14}$, $3\times10^{-13}$, $5\times10^{-13}$ Ha
in 22, 11, 8 and 7 iterations.

Mutations, each failing on an assertion: the active-active coupling term of
$W$ with its sign flipped (`&b - &a`) fails all three FD anchors and both
same-minimum tests (AH no longer converges); dropping the $-\tfrac12[G, X]$
correction fails the three symmetry assertions; $K$ replaced by $J$ in
`dense_jk_multi` fails its unit test; the direct $(pu|qv)$ scaled by
$1 + 10^{-4}$ fails the water direct-vs-dense test ($3.4\times10^{-5}$
against $2.8\times10^{-6}$; $1 + 10^{-6}$ is below the bar); the CLI
`optimizer` key ignored fails `cli_keys_reach_the_config`.

**AH timing (2026-11-13, AO J/K products, c74da329d).** CLI wall times
(RHF included), paired and interleaved: base build 1db597bf6, build
c74da329d with `optimizer = "bfgs"`
(then the default) and with `"ah"`, order base, bfgs, ah, ah, bfgs, base, base,
bfgs, ah; 12 rayon threads, default `[memory]`, dense route. The box was
shared with other jobs; the load average during each set is given.

| System, load | base (s) | BFGS (s) | AH (s) | iterations base / BFGS / AH |
|---|---|---|---|---|
| benzene CAS(6,6), load 15.5-20.5 | 63.47, 65.66, 64.23: 64.5 ± 1.1 | 63.27, 56.84, 65.13: 61.7 ± 4.4 | 87.04, 83.69, 85.09: 85.3 ± 1.7 | 89 / 89 / 18 |
| N2 CAS(10,10), load 17-20 | 28.70, 28.50, 28.84: 28.7 ± 0.2 | 28.58, 30.43, 29.03: 29.4 ± 1.0 | 20.53, 19.60, 20.72: 20.3 ± 0.6 | 36 / 36 / 25 |
| benzene CAS(6,6), load 5.6-15.5 | 37.70, 36.76, 36.26: 36.9 ± 0.7 | 37.04, 35.61, 35.54: 36.1 ± 0.9 | 48.72, 47.65, 47.91: 48.1 ± 0.6 (ratio stretch) | 89 / 89 / 18 |

Every run reaches the energy of the base build to the printed 10 decimals
(benzene $-230.7814586093$, N2 $-109.1294674756$ Ha). BFGS in this build
takes the same iterations as the base build. AH costs about 1.3 times the
BFGS wall time on benzene, where one Hessian product (a $J/K$ pass over the
dense block) is about a third of a macro iteration and each AH step takes
10-20 products, and saves about 30 % on N2 CAS(10,10), where the CI solve
dominates. The third row was measured with the ratio stretch of the
prototype instead of the shipped curvature stretch, at the same 18
iterations. With the ratio stretch N2 CAS(10,10) did not converge: it
passed the $-109.1294674756$ Ha point with $\lVert g\rVert \approx 5\times10^{-4}$
and steps of $10^{-4}$, actual/predicted ratios of $10^5$-$10^6$, then
descended to $-109.1347620086$ Ha, 5.3 mHa lower, and stalled there at
$\lVert g\rVert = 9.2\times10^{-5}$ after 1000 macro iterations. That lower
energy was not examined further (whether it is a minimum of the same active
space class, and whether BFGS's and PySCF's point is a saddle).

**Davidson forcing and occupied-kernel products (2026-11-13, round-2
task 14 follow-up).** Harness `examples/casscf_ah_bench.rs` (exact-J/K RHF
of `driver::rhf_start`, `CasScfConfig` defaults, `FERRIC_TIMING=1` for the
product count); base build 8833755a8 (same harness without the forcing
argument) in a separate target directory; 12 rayon threads, dense route.
Wall times are of `run_casscf` only. Iterations / Hessian products of the
AH step from single runs (identical on repeats):

| System | base AH (quadratic) | constant $\eta = 0.1$ | $\eta = 0.3$ | $\eta = 0.5$ |
|---|---:|---:|---:|---:|
| water CAS(4,4) | 22 / 193 | 24 / 133 | 28 / 112 | 31 / 101 |
| N2 CAS(6,6) | 11 / 55 | 11 / 27 | 13 / 23 | 16 / 21 |
| O2 CAS(8,6) $M_s = 1$ | 8 / 35 | 8 / 17 | 10 / 15 | 12 / 13 |
| N2 CAS(10,10) | 25 / 176 | 25 / 80 | 29 / 65 | |
| benzene CAS(6,6) | 18 / 335 | 24 / 235 | 30 / 230 | |

Anchor: `AhForcing::Quadratic` in the build with the forcing option and
the AO J/K products reproduces 8833755a8's AH run on all five systems in
the final energy to the last printed digit (17 significant), $\lVert
g\rVert$, the iteration count and the product count. Constant forcing
alone left benzene slower than BFGS (CASSCF 33.5 s at $\eta = 0.1$,
38.6 s at 0.3, BFGS 27.4 s, single runs at load 7-13).

The occupied kernel removes the $J/K$ pass from the product. Anchors
(`tests/casscf_newton.rs`, dense route, rotated RHF orbitals, unit $x$):
kernel against AO $J/K$ product $3.6\times10^{-15}$, $1.6\times10^{-14}$,
$3.6\times10^{-15}$ (water, N2, O2; a priori bars $2.9\times10^{-6}$,
$6.7\times10^{-4}$, $1.1\times10^{-4}$); `transform_ppaa_papa` against two
`transform_eri_mixed` chains: $(pq|vw)$ bitwise, $(pu|qv)$
$8.1\times10^{-16}$ (bar $1.1\times10^{-10}$, water, 5 columns); direct
against dense $Hx$ (both on the kernel) $8.5\times10^{-14}$,
$2.6\times10^{-13}$, $2.5\times10^{-13}$ (bars $1.1\times10^{-6}$,
$3.7\times10^{-5}$, $2.1\times10^{-3}$). Benzene phases (one run, load
9-12): orbital Hessian integrals 0.71 s per macro iteration, products
1.4 s in total for 349 (3.4 s before the product GEMM ran at the
serial-section BLAS thread count). With products this cheap, constant
forcing costs more in extra macro iterations (each with a Hessian
integral build) than it saves: benzene CASSCF 31.6 ± 8.9 s at
$\eta = 0.1$ against 27.3 ± 6.0 s quadratic, so quadratic stays the
default. At $\eta = 0.1$ benzene stops at $-230.7814587989$ Ha, the nearby
stationary point of the optimizer comparison above.

Mutations, each failing on an assertion: $-\tfrac12(pr|jk) \to
+\tfrac12(pr|jk)$ in $L$ (kernel against AO $J/K$ 1.2-1.6 against bars
$\le 6.7\times10^{-4}$, the FD anchors, the water same-minimum tests); the
$(pu|qv)$ active block taken at $u$ instead of $n_c + u$ (kernel tests
0.30-1.09, FD anchors, all four same-minimum tests); the axis swap of
`transform_ppaa_papa` dropped (1.01 against $1.1\times10^{-10}$); the two
outputs of the shared-first-quarter transform swapped (bitwise check); the
direct $(pu|qv)$ scaled by $1 + 10^{-4}$ (direct against dense $Hx$,
water $7.7\times10^{-5}$ and O2 $1.1\times10^{-4}$ against
$1.1\times10^{-6}$, $3.7\times10^{-5}$).

Wall times with the kernel (`run_casscf`, s), paired and interleaved. The
small systems: 10 repetitions in alternating order (base BFGS, base AH,
new AH; then reversed), load 13-17. N2 CAS(10,10) and benzene: three
repetitions in the orders (bB bA nA nE), (nE nA bA bB), (bB nA nE bA)
(b base, n this build, B BFGS, A AH quadratic, E AH $\eta = 0.1$), load
given per system:

| System | base BFGS | base AH | AH, kernel | iterations BFGS / AH | load |
|---|---|---|---|---|---|
| water | 0.478 ± 0.048 | 0.240 ± 0.057 | 0.202 ± 0.040 | 61 / 22 | 13-14 |
| N2 CAS(6,6) | 0.859 ± 0.247 | 0.498 ± 0.158 | 0.474 ± 0.144 | 21 / 11 | 14-17 |
| O2 CAS(8,6) | 0.343 ± 0.096 | 0.234 ± 0.087 | 0.264 ± 0.096 | 14 / 8 | 13-15 |
| N2 CAS(10,10) | 27.23, 26.70, 26.90: 26.95 ± 0.27 | 18.62, 18.12, 18.57: 18.44 ± 0.28 | 18.63, 19.79, 18.56: 18.99 ± 0.69 | 36 / 25 | 15-24 |
| benzene CAS(6,6) | 38.75, 47.11, 41.27: 42.4 ± 4.3 | 62.19, 39.46, 56.15: 52.6 ± 11.8 | 30.76, 20.33, 30.75: 27.3 ± 6.0 | 89 / 19 | 8-25 |

The benzene load swung between 8 and 25 within the set; within each
repetition the kernel AH run was faster than the base BFGS run
(30.8/38.8, 20.3/47.1, 30.8/41.3 s). On N2 CAS(10,10) the CI solve
dominates and the kernel changes nothing measurable. O2 base AH and
kernel AH are inside the noise; both beat BFGS in fewer iterations. BFGS
in this build reproduces the base BFGS run (energy, $\lVert g\rVert$,
iterations) on all five systems. With AH the default these pass:
ferric-ci `--lib`, `casscf`, `casscf_vs_pyscf`, `casscf_canonical`,
`casscf_direct`, `casscf_newton`, `cas_ecp`, `nevpt2`; ferric-cli
`cas_kinds`; and the Python `test_casscf.py`, `test_cas_cli.py`,
`test_nevpt2.py`, `test_casci.py` (slow markers included) and the `.pyi`
coverage tests.

**N2 CAS(10,10)/cc-pVDZ stationary points (2026-11-13).** Harness
`examples/casscf_stationary_points.rs`: BFGS from the RHF orbitals to
$\lVert g\rVert < 10^{-9}$, canonicalized (natural active orbitals); the
CI-relaxed orbital Hessian by central differences of the analytic gradient
($h = 10^{-4}$, all 212 pairs, symmetrized) and its lowest eigenvalues; each
active natural orbital classified by its dominant $|m|$ (Fourier power on
circles about the bond axis, 64 points) and its inversion parity
$\langle\phi(r), \phi(-r)\rangle/\langle\phi,\phi\rangle$ about the bond
midpoint. While the lowest eigenvalue is below $-10^{-4}$, the orbitals are
rotated by 0.3 along its eigenvector and BFGS converges again.

| Point | $E$ (Ha) | $\lVert g\rVert$ | lowest Hessian eigenvalues | natural occupations | active orbitals beyond the 8 valence |
|---|---:|---:|---|---|---|
| start (= BFGS, AH, PySCF) | -109.1294674756 | 9.4e-10 | -0.00999, -0.00814, 0.00000, 0.00263 | 1.9938 1.9803 1.9790 1.9422 1.9407 0.0679 0.0641 0.0203 0.0076 0.0040 | $\sigma_g$, one component of a $\pi_u$ pair |
| one step down | -109.1300807964 | 7.8e-10 | -0.00457, 0.00000, 0.00219 | 1.9933 1.9811 1.9782 1.9418 1.9404 0.0681 0.0643 0.0203 0.0085 0.0039 | $\sigma$ orbitals of mixed parity (0.03 to 2.7), one $\pi_u$ component |
| two steps down | -109.1347620089 | 6.2e-10 | 0.01087, 0.01087, 0.01928, 0.02000 | 1.9905 1.9811 1.9795 1.9423 1.9423 0.0636 0.0636 0.0202 0.0100 0.0070 | $\sigma_g$, $\sigma_u$ (parities $\pm1.000$) |

The point that BFGS, AH (both forcings) and PySCF reach is a second-order
saddle: its active space holds a single component of a degenerate
$\pi_u$ pair, which breaks the axial symmetry (the valence $\pi$
occupations differ by $1.5\times10^{-3}$ and $3.8\times10^{-3}$). The zero
eigenvalue is the rotation of that component about the axis. Two steps
down the negative modes reach a local minimum 5.29 mHa lower whose extra
active orbitals are one $\sigma_g$ and one $\sigma_u$, with exactly
degenerate $\pi$ pairs; this is the $-109.1347620086$ Ha point the ratio
stretch descended to (above), to $3\times10^{-10}$ Ha. Measurement only:
the optimizers are first-order convergent to stationary points and nothing
here checks the Hessian at convergence; why the RHF start leads to the
saddle was not examined.

**Benzene stationary point.** `pyscf.mcscf.CASSCF` (PySCF 2.13.1) on ferric's
AO integrals reaches $E = -230.7873347336$ Ha, 5.876 mHa below ferric, from
ferric's RHF orbitals (36 macro iterations) and from PySCF's own RHF. Each
code's energy functional reproduces the other's at the other's orbitals:

| Orbitals | ferric $E$ (Ha) | ferric $\lVert g\rVert$ | PySCF CASCI $E$ (Ha) | PySCF $\lVert g\rVert$ |
|---|---:|---:|---:|---:|
| ferric's converged | -230.7814586093 | 4.8e-8 | -230.7814586093 | 2.4e-8 |
| PySCF's converged | -230.7873347336 | 1.0e-6 | -230.7873347336 | |

Both points are stationary in both codes. Lowest eigenvalues of the
CI-relaxed orbital Hessian (Lanczos on central-difference gradient products,
$h = 10^{-4}$, on $D^{-1/2}HD^{-1/2}$ with $D$ the approximate diagonal
floored at 0.05, which has the inertia of $H$):

| Point | Lanczos steps | lowest three | Ritz residual of the lowest |
|---|---:|---|---:|
| ferric's | 80 | 0.3645, 0.5082, 0.5132 | 2.9e-11 |
| PySCF's | 40 | 0.3999, 0.5125, 0.5143 | not recorded |

Ferric's point is therefore a local minimum, not a saddle. Under $z \to -z$
(the molecular plane) every orbital in all three sets (RHF, ferric, PySCF)
stays purely $\sigma$ or purely $\pi$. Both active spaces hold two $\sigma$ and
four $\pi$ orbitals, with the sixth $\pi$ ($a_{2u}$) in the core. In the RHF
the $a_{2u}$ $\pi$ ($-0.498$ Ha) lies below the $e_{2g}$ $\sigma$ pair
($-0.491$ Ha), and the 18-orbital core splits that pair. Natural occupations:
ferric 1.981, 1.938, 1.936, 0.064, 0.063, 0.019; PySCF 1.982, 1.941, 1.936,
0.065, 0.058, 0.018. The two codes sit at two minima of the same
$2\sigma + 4\pi$ active-space class, and neither is the all-$\pi$ CAS(6,6). A
$\pi$ CAS(6,6) needs an active window placed on the six $\pi$ orbitals, which neither
`[cas]` nor `run_casscf` offers: the active orbitals are the `n_active`
orbitals after the first `n_core` in the RHF order.

## SC-NEVPT2 (`src/nevpt2.rs`)

Strongly contracted NEVPT2 (Angeli, Cimiraglia and Malrieu, JCP 117, 9138
(2002)) on one CAS root. Definitions, conventions and the prototype's
measurements: `scripts/prototypes/nevpt2_derivation.md` (cited below as
"derivation"), whose numpy construction `scripts/prototypes/nevpt2_proto.py`
is the reference implementation.

Zeroth order (derivation, "Definition"): Dyall's Hamiltonian
$H_D = \sum_i \varepsilon_i E_{ii} + \sum_r \varepsilon_r E_{rr} + H_\text{act}$,
$H_\text{act} = \sum_{tu} F^I_{tu}E_{tu} + \tfrac12\sum_{tuvw}(tu|vw)(E_{tu}E_{vw} - \delta_{uv}E_{tw})$.
For a label $l$ (its external orbitals) of a class with $n_v$ virtual
particles and $n_c$ core holes, $\Psi_l = P_l H\Psi_0$,
$N_l = \langle\Psi_l|\Psi_l\rangle$,
$K_l = \langle\Psi_l|H_\text{act} - E_0|\Psi_l\rangle$,
$\Delta_l = \sum_{r\in l}\varepsilon_r - \sum_{i\in l}\varepsilon_i$ and

$$
E_\text{class} = -\sum_l \frac{N_l}{\Delta_l + K_l/N_l},
$$

over labels with $|N_l| > 10^{-14}$ (PySCF `NUMERICAL_ZERO`). The eight
classes are those of the derivation's table: `Sr` (r), `Si` (i), `Sijrs`,
`Sijr`, `Srsi`, `Srs`, `Sij`, `Sir`.

Orbitals. $F^I = h + J[D^c] - \tfrac12 K[D^c]$, $D^c = 2C_cC_c^\mathsf{T}$;
$F = F^I + F^A$ with $F^A$ from $D^a = C_a\gamma C_a^\mathsf{T}$. The core and
virtual orbitals are the eigenvectors of the core and virtual blocks of $F$
and $\varepsilon$ their eigenvalues: PySCF `mcscf.CASCI.canonicalize`, which
`mrpt.NEVPT` calls (derivation, "Orbitals"). The active orbitals are used as
given: every active index is summed, so $E$ does not depend on the active
basis, and the RDMs are built in that basis from the reference CI vector.
PySCF rotates to natural orbitals instead. With `canonicalize = false` the
given core and virtual orbitals are used, $\varepsilon = \operatorname{diag}F$,
and an off-diagonal element above $10^{-8}$ Ha is an error. Inside a group of
degenerate core or virtual orbitals the eigenvectors are not unique and the
pair classes are not invariant to rotations in it (derivation, "Comparison
with pyscf.mrpt.NEVPT"); N2's $\pi$ pairs move `Srs` by up to 1.7e-7 Ha.

Evaluation. $N_l$ and $K_l$ reduce (derivation, "Construction A", steps
1-5; $K_l$ through $\langle V^\dagger P_l [H_\text{act}, V]\rangle$) to
contractions of integrals with RDMs up to rank 4, the 4-RDM only contracted
with $(aa|aa)$. The module evaluates them with the factorization of PySCF
`mrpt.nevpt2` (`Sr` ... `Sir`, `make_a16`, `make_a17`, `make_a19`,
`make_a22`, `make_a23`, `make_a25`, `make_a3`, `make_k27`, `make_a7`,
`make_a9`, `make_a12`, `make_a13`), transcribed spec for spec into an
`einsum` over owned arrays. The RDM inputs are PySCF's: the E-product
$\langle E_{pq}E_{rs}\rangle$ and $\langle E_{pq}E_{rs}E_{tu}\rangle$ (from the
normal-ordered ones by `rdm3::e_product_rdms`, the inverse of the reordering
in "NEVPT RDMs"), the hole RDMs and $f^\text{ca}$, $f^\text{ac}$ of that
section. In PySCF's layout `h1e` is the active block of $F^I$ and
$\texttt{h2e}_{pqrs} = (pr|qs)$; the external integral blocks are
$(va|aa)$, $(ac|aa)$, $(cv|cv)$, $(ca|cv)$, $(va|cv)$, $(va|va)$, $(ca|ca)$,
$(vc|aa)$, $(va|ca)$ and the $F^I$ blocks $va$, $ac$, $vc$ (derivation,
"What a Rust port needs"), each transformed from the dense AO block with its
narrowest orbital set contracted first.

Reference checks before any class: $\lVert Hc - Ec\rVert \le$
`max_ci_residual` (default $10^{-8}$; the formulas assume
$H_\text{act}\Psi_0 = E_0\Psi_0$), $\lVert c\rVert = 1$ to $10^{-10}$,
$n_\alpha = n_\beta$ (open-shell active spaces are not anchored and are
refused), and a converged CASSCF or CAS-CI run.

Cost: $O(n^8)$ for the $a_{16}$/$a_{22}$ intermediates, $O(n_v n^6)$ (r),
$O(n_c n^6)$ (i), $O(n_v^2 n^4)$ (rs), $O(n_c^2 n^4)$ (ij), $O(n_c n_v n^4)$ (ir),
$O(n_c^2 n_v^2)$ (ijrs). The `einsum` is a direct loop over every label
combination (one sequential sum per output element, parallel over output
elements), so its cost is the product of all label extents of a spec.
Memory: the dense AO block $n_\text{bas}^4$, the MO blocks (twice, transformed
and permuted), twelve $n^6$ intermediates and the per-label tensors, reserved
through one `MemoryPlan` before the AO block is built.

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
| `tests/nevpt2.rs` | SC-NEVPT2. Exact zeros of every class whose external space is empty (H2 CAS(2,4) and CAS(2,2), LiH core 1 CAS(2,5)), with the non-empty classes nonzero; a one-orbital closed-shell active space on H2O/6-31G RHF orbitals equals the MP2 pair sums and the closed-form `Srs` to 8.1e-15 (bar 1e-10); the numpy prototype's Wick construction on the same ferric CASSCF orbitals and CI vector (H2O CAS(4,4), N2 CAS(6,6) at 1.0977 and 2.2 Å, 6-31G) to 6.2e-14 (bars 2.3e-10, 4.6e-9); `pyscf.mrpt.NEVPT` on PySCF's own CASSCF to 1.3e-8 (H2O), 3.2e-8 (N2), 2.1e-8 Ha (stretched N2), within the propagated CASSCF-convergence bar of each class (largest ratio 0.88, stretched N2 `Sr`, where PySCF's gradient is 9.8e-7). Mutations that fail it: a sign flip in `make_a12` (`Sir`), a $p\leftrightarrow q$ swap in one `make_a12` RDM index, the dropped `f3ca` term of `make_a16` (`Sr`), $F^I$ for the orbital energies (MP2 anchor, prototype, PySCF), `Sr` gated on the core space (zeros anchor). |
| FCI energies vs PySCF (`src/lib.rs` tests, `tests/validation_casci.rs`) | Spectrum only: blind to sign conventions that are diagonal similarities, and to errors in elements that vanish by point-group symmetry in the test molecules. A $\pm1$ similarity inserted into `general_element` passes all of them; H2/STO-3G has no nontrivial sign at all. |
