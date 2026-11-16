# Coupled Cluster Theory

Equations as `crates/ferric-cc/src` implements them, with the function that
implements each. Two conventions are used:

- **Spin-orbital** (`ccd`, `ccsd`, `ccsd_t`, `linlccd`, `linlccd_u`,
  `linlccd_exact`, `dlpno_linlccd`): indices run over $2n_o$ / $2n_v$ spin
  orbitals, interleaved (even = $\alpha$, odd = $\beta$);
  $\langle pq\|rs\rangle = (pr|qs) - (ps|qr)$, with chemist integrals
  $(pq|rs) = \sum_P B^P_{pq} B^P_{rs}$ from the dressed RI factors
  (`ferric_mp2::spinorbital::asym_*`).
- **Spin blocks, UHF** (`u_ccsd`, `u_ccsd_t`): separate $\alpha$ and $\beta$
  occupied and virtual spaces; see "Unrestricted CCSD and CCSD(T)".
- **Spatial, closed shell** (`ccsd_closed_shell`, `ccsd_t_closed_shell`,
  `linlccd_amplitude`, `dlpno_ccsd*`): indices run over $n_o$ / $n_v$
  spatial orbitals and amplitudes satisfy $t_{ij}^{ab} = t_{ji}^{ba}$.

Denominators are positive, $\Delta_{ij}^{ab} = \varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j$
and $\Delta_{ijk}^{abc} = \varepsilon_a+\varepsilon_b+\varepsilon_c-\varepsilon_i-\varepsilon_j-\varepsilon_k$
for a gapped reference, and every energy carries its minus sign
explicitly. The code stores the negative $-\Delta$ (the `d` arrays,
`pno_denominators`) and divides by it. Canonical HF orbitals are assumed, so
$f_{ia} = 0$ and the occupied/virtual Fock blocks are diagonal.

## Exponential ansatz

$$|\Psi_{CC}\rangle = e^{\hat{T}} |\Phi_0\rangle,\qquad \hat{T} = \hat{T}_1 + \hat{T}_2 + \dots$$

## CCD

$\langle\Phi_{ij}^{ab}| e^{-\hat T_2}\hat H e^{\hat T_2} |\Phi_0\rangle = 0$ in
spin orbitals, $\hat T_2 = \tfrac14\sum t_{ij}^{ab}\, a^\dagger_a a^\dagger_b a_j a_i$,
$t$ antisymmetric in $ij$ and in $ab$ (`ccd.rs::ccd_spinorbital`; `ccd` delegates
to it). With canonical HF orbitals the Fock terms reduce to
$\Delta_{ij}^{ab} t_{ij}^{ab}$ and the projection is
$R_{ij}^{ab} = \Delta_{ij}^{ab} t_{ij}^{ab} + r_{ij}^{ab}$, where

$$
\begin{aligned}
r_{ij}^{ab} ={}& \langle ij\|ab\rangle
 + \tfrac12\sum_{cd}\langle ab\|cd\rangle t_{ij}^{cd}
 + \tfrac12\sum_{kl}\langle kl\|ij\rangle t_{kl}^{ab}
 + P(ij)P(ab)\sum_{kc}\langle kb\|cj\rangle t_{ik}^{ac} \\
 &+ \tfrac14\sum_{klcd}\langle kl\|cd\rangle t_{ij}^{cd} t_{kl}^{ab}
 + \tfrac12 P(ij)P(ab)\sum_{klcd}\langle kl\|cd\rangle t_{ik}^{ac} t_{jl}^{bd} \\
 &- \tfrac12 P(ab)\sum_{klcd}\langle kl\|cd\rangle t_{ij}^{ac} t_{kl}^{bd}
 - \tfrac12 P(ij)\sum_{klcd}\langle kl\|cd\rangle t_{ik}^{ab} t_{jl}^{cd},
\end{aligned}
$$

$P(ij)x = x - x_{i\leftrightarrow j}$ (`helpers.rs::{p_ij,p_ab,p_ij_ab}`). The
loop sets $t \leftarrow r/(-\Delta)$ (the stored `d` is $-\Delta$), with DIIS
on the increment, from the MP2 guess $t = \langle ij\|ab\rangle/(-\Delta)$, and
stops when the energy changes by less than $10^{-10}$:

$$E_c = \tfrac14 \sum_{ijab} \langle ij\|ab\rangle\, t_{ij}^{ab}.$$

The closed-shell spatial form of the same energy is
$E_c = \sum_{ijab} (ia|jb)\,(2 t_{ij}^{ab} - t_{ij}^{ba})$.

**Check.** `scripts/cc_ccsd_eom_brute_force_sympy.py` (`ccd-residual`) builds
$H$, $\hat T_2$ and $e^{-\hat T_2}\hat He^{\hat T_2}$ as Jordan–Wigner
matrices on a ten-spin-orbital model whose Fock matrix is diagonal by
construction, and compares $r + \Delta t$ and $E_c$ with the projections at
random antisymmetric $t$. That route uses no diagram or intermediate.

## CCSD

$R_i^a = \langle\Phi_i^a| e^{-\hat T}\hat H e^{\hat T} |\Phi_0\rangle = 0$ and
$R_{ij}^{ab} = \langle\Phi_{ij}^{ab}| e^{-\hat T}\hat H e^{\hat T} |\Phi_0\rangle = 0$,
$\hat T = \hat T_1 + \hat T_2$, $\hat T_1 = \sum t_i^a a^\dagger_a a_i$.

### CCSD in spin orbitals

`ccsd.rs::ccsd` follows Stanton, Gauss, Watts and Bartlett,
JCP 94, 4334 (1991), with $f_{ia} = 0$ and the diagonal Fock moved into
$\Delta$. With
$\tilde\tau_{ij}^{ab} = t_{ij}^{ab} + \tfrac12(t_i^at_j^b - t_j^at_i^b)$ and
$\tau_{ij}^{ab} = t_{ij}^{ab} + t_i^at_j^b - t_j^at_i^b$,

$$
\begin{aligned}
F_{ae} &= \sum_{mf} t_m^f\langle ma\|fe\rangle - \tfrac12\sum_{mnf}\tilde\tau_{mn}^{af}\langle mn\|ef\rangle, &
F_{mi} &= \sum_{ne} t_n^e\langle mn\|ie\rangle + \tfrac12\sum_{nef}\tilde\tau_{in}^{ef}\langle mn\|ef\rangle, \\
F_{me} &= \sum_{nf} t_n^f\langle mn\|ef\rangle, &
W_{mnij} &= \langle mn\|ij\rangle + P(ij)\sum_e t_j^e\langle mn\|ie\rangle + \tfrac14\sum_{ef}\tau_{ij}^{ef}\langle mn\|ef\rangle,
\end{aligned}
$$

$$
W_{abef} = \langle ab\|ef\rangle - P(ab)\sum_m t_m^b\langle am\|ef\rangle + \tfrac14\sum_{mn}\tau_{mn}^{ab}\langle mn\|ef\rangle,
$$

$$
W_{mbej} = \langle mb\|ej\rangle + \sum_f t_j^f\langle mb\|ef\rangle - \sum_n t_n^b\langle mn\|ej\rangle
 - \sum_{nf}\big(\tfrac12 t_{jn}^{fb} + t_j^ft_n^b\big)\langle mn\|ef\rangle ,
$$

the singles and doubles are $R = \Delta t + r$ with

$$
\begin{aligned}
r_i^a ={}& \sum_e t_i^eF_{ae} - \sum_m t_m^aF_{mi} + \sum_{me} t_{im}^{ae}F_{me}
 - \sum_{nf} t_n^f\langle na\|if\rangle
 - \tfrac12\sum_{mef} t_{im}^{ef}\langle ma\|ef\rangle
 - \tfrac12\sum_{men} t_{mn}^{ae}\langle nm\|ei\rangle , \\
r_{ij}^{ab} ={}& \langle ij\|ab\rangle
 + P(ab)\sum_e t_{ij}^{ae}\big(F_{be} - \tfrac12\sum_m t_m^bF_{me}\big)
 - P(ij)\sum_m t_{im}^{ab}\big(F_{mj} + \tfrac12\sum_e t_j^eF_{me}\big) \\
 &+ \tfrac12\sum_{mn}\tau_{mn}^{ab}W_{mnij} + \tfrac12\sum_{ef}\tau_{ij}^{ef}W_{abef}
 + P(ij)P(ab)\sum_{me}\big(t_{im}^{ae}W_{mbej} - t_i^et_m^a\langle mb\|ej\rangle\big) \\
 &+ P(ij)\sum_e t_i^e\langle ab\|ej\rangle - P(ab)\sum_m t_m^a\langle mb\|ij\rangle ,
\end{aligned}
$$

$t \leftarrow r/(-\Delta)$ for both ranks (DIIS on $t_2$ only), and

$$E_c = \tfrac14\sum_{ijab}\langle ij\|ab\rangle t_{ij}^{ab} + \tfrac12\sum_{ijab}\langle ij\|ab\rangle t_i^at_j^b .$$

At $t_1 = 0$ the doubles residual is the CCD one above.

**Checks.** `scripts/cc_ccsd_eom_brute_force_sympy.py`: `ccsd-residual` and
`ccsd-energy` compare $r + \Delta t$ and $E_c$ with the brute-force
projections at random $t_1$, $t_2$; `ccsd-ccd-limit` checks the $t_1 = 0$ limit.

### CCSD, closed shell

`ccsd_closed_shell.rs::ccsd_closed_shell`. For an RHF
reference the spin-orbital amplitudes are fixed by spatial ones
(`expand_amplitudes_to_spin_orbital`):
$t_{i\sigma}^{a\tau} = \delta_{\sigma\tau}t_i^a$ and
$t_{i\sigma j\tau}^{a\sigma' b\tau'} = \delta_{\sigma\sigma'}\delta_{\tau\tau'}t_{ij}^{ab} - \delta_{\sigma\tau'}\delta_{\tau\sigma'}t_{ij}^{ba}$,
with $t_{ij}^{ab} = t_{ji}^{ba}$. On such amplitudes the spin-orbital
residual is spin-adapted: the $\alpha$ and $\beta$ singles are equal, the
opposite-spin block $R_{ij}^{ab} = R_{i\alpha j\beta}^{a\alpha b\beta}$ has
$R_{ij}^{ab} = R_{ji}^{ba}$, and the same-spin block is $R_{ij}^{ab} - R_{ij}^{ba}$.
Solving the spatial equations $R_i^a = 0$, $R_{ij}^{ab} = 0$ is therefore
solving CCSD. The code evaluates them in the factorization of Hirata,
Podeszwa, Tobita and Bartlett, JCP 120, 2581 (2004), Eqs. 35–45 (PySCF
`rccsd.update_amps` and `rintermediates`): the $F_{oo}$, $F_{vv}$, $F_{ov}$
and $L_{oo}$, $L_{vv}$ intermediates (Eqs. 37–41, no orbital-energy shift, so
$\varepsilon$ enters only through $\Delta$), $W_{oooo}$, $W_{voov}$, $W_{vovo}$
(Eqs. 42–45) and the T1 and T2 equations (Eqs. 35, 36), the T2 part
symmetrized as $x + x^{(ij)(ab)}$. The $\sum_{cd}W_{abcd}\tau_{ij}^{cd}$ ladder
is never formed as $W_{abcd}$: the dense route contracts $(ac|bd)$ and the
$t_1$ terms with $\tau$ separately (`dense_ladder`), the RI route rebuilds
the integrals in panels (`ri_ladder.rs::ri_ladder_contract`,
$W_{abcd} = \sum_P[(B_{vv}+B_x)^P_{ac}B^P_{vv,bd} + B^P_{vv,ac}B^P_{x,bd}]$ with
$B^P_{x,bd} = -\sum_k B^P_{kd}t_k^b$). The energy is

$$E_c = \sum_{ijab}\big(2(ia|jb) - (ib|ja)\big)\big(t_{ij}^{ab} + t_i^at_j^b\big),$$

and $(t_1, t_2)$ are extrapolated jointly by DIIS (`diis_step_two`).

**Checks.** `scripts/cc_ccsd_eom_brute_force_sympy.py`, `spin-adaptation`:
the block structure above and the closed-shell energy on expanded spatial
amplitudes, against the brute-force projections. The Hirata factorization itself is checked in
Rust: `ccsd_closed_shell` against the spin-orbital `ccsd` on the same
integrals and against PySCF `cc.CCSD` (`tests/validation_cc.rs`).

## CCSD(T), spin-orbital

`ccsd_t.rs` (module doc; PySCF `gccsd_t_slow` convention):

$$W_{ijk}^{abc} = P(i/jk)\,P(a/bc)\Big[\sum_e t_{jk}^{ae}\langle bc\|ei\rangle - \sum_m t_{im}^{bc}\langle ma\|jk\rangle\Big],\qquad
V_{ijk}^{abc} = P(i/jk)\,P(a/bc)\big[t_i^a \langle bc\|jk\rangle\big],$$

$$E_{(T)} = -\frac{1}{36}\sum_{ijkabc} \frac{(W + V)_{ijk}^{abc}\, W_{ijk}^{abc}}{\Delta_{ijk}^{abc}},$$

with $P(i/jk)x = x - x_{i\leftrightarrow j} - x_{i\leftrightarrow k}$. The
summand is invariant under any simultaneous relabelling of $(i,j,k)$ and of
$(a,b,c)$, so the streaming loop visits each $i<j<k$ once with $a,b,c$ over
the full axis and divides by 6 instead of 36.

## CCSD(T), closed shell

`ccsd_t_closed_shell.rs` computes the perturbative triples of a closed-shell
RHF-reference CCSD in spatial orbitals (Rendell, Lee & Komornicki, CPL 178,
462 (1991); index convention of PySCF `cc/ccsd_t_slow.py`). It needs the
spatial amplitudes of `ccsd_closed_shell` and rejects spin-orbital ones.
Indices $i,j,k,l$ run over the $n_o$ active occupied orbitals, $a,b,c,d$ over
the $n_v$ virtuals, and all integrals are chemist RI integrals.

**Definition.** Write $o=(i,j,k)$, $v=(a,b,c)$ and, for a permutation
$\pi\in S_3$, $o_\pi = (o_{\pi(0)},o_{\pi(1)},o_{\pi(2)})$, likewise $v_\pi$.
The raw connected block is

$$w^{(0)}_{ijk}{}^{abc} = \sum_d (ia|bd)\,t_{kj}^{cd} - \sum_l (ia|lj)\,t_{lk}^{bc}.$$

$W$ sums it over the six simultaneous permutations of the (occupied, virtual)
pairs $(i,a),(j,b),(k,c)$, and $V$ adds the singles:

$$W_{o}^{v} = \sum_{\pi\in S_3} w^{(0)}{}_{o_\pi}^{v_\pi},\qquad
V_{ijk}^{abc} = W_{ijk}^{abc} + (jb|kc)\,t_i^a + (ia|kc)\,t_j^b + (ia|jb)\,t_k^c .$$

The $f_{ai}\,t_2$ piece of PySCF's `get_v` is absent because $f_{ai}=0$ for
canonical HF orbitals. The spin-summed weight is PySCF's `r3`, acting on the
virtual labels at fixed $o$:

$$\tilde W_{ijk}^{abc} = 4W_{ijk}^{abc} + W_{ijk}^{bca} + W_{ijk}^{cab}
 - 2W_{ijk}^{cba} - 2W_{ijk}^{acb} - 2W_{ijk}^{bac},$$

and the energy is the unrestricted sum

$$E_{(T)} = \frac13 \sum_{ijk}\sum_{abc} f_{ijk}^{abc},\qquad
f_{ijk}^{abc} = -\frac{\tilde W_{ijk}^{abc}\,V_{ijk}^{abc}}{\Delta_{ijk}^{abc}} .$$

The divisor 3 is not $1/36$ or $1/6$: the spin summation behind the
$4,1,1,-2,-2,-2$ coefficients has already absorbed most of the spin-orbital
antisymmetrizer's redundancy. This sum equals the spin-orbital $(T)$ of the
previous section on the spin-orbital expansion of the same amplitudes and
integrals.

**Pair invariance.** Let $\sigma\in S_3$ act on both triples at once.

- $W_{o_\sigma}^{v_\sigma} = W_o^v$, because $\pi\mapsto\pi\circ\sigma$ is a
  bijection of $S_3$.
- $V$: the three singles terms map onto each other, using $(pq|rs)=(rs|pq)$.
  For $(i,a)\leftrightarrow(j,b)$, $(jb|kc)t_i^a \leftrightarrow (ia|kc)t_j^b$
  and $(ia|jb)t_k^c$ is fixed.
- $\Delta$ is symmetric in $o$ and in $v$ separately.
- $\tilde W_o^v = \sum_\tau c_\tau W_o^{v_\tau}$ with $c_\tau = 4$ for the
  identity, $1$ for the two 3-cycles and $-2$ for the three transpositions.
  Then $\tilde W_{o_\sigma}^{v_\sigma} = \sum_\tau c_\tau W_o^{v_{\sigma\tau\sigma^{-1}}}$
  by pair invariance of $W$, and this is $\tilde W_o^v$ because $c_\tau$ is
  constant on conjugacy classes.

So $f_{o_\sigma}^{v_\sigma} = f_o^v$. Pair invariance also moves the `r3`
pattern from the virtual labels to the occupied ones, which is the form the
kernel uses at a fixed virtual triple:
$W_{ijk}^{bca} = W_{kij}^{abc}$, $W_{ijk}^{cab} = W_{jki}^{abc}$,
$W_{ijk}^{cba} = W_{kji}^{abc}$, $W_{ijk}^{acb} = W_{ikj}^{abc}$,
$W_{ijk}^{bac} = W_{jik}^{abc}$, so

$$\tilde W_{ijk}^{abc} = 4W_{ijk}^{abc} + W_{kij}^{abc} + W_{jki}^{abc}
 - 2W_{kji}^{abc} - 2W_{ikj}^{abc} - 2W_{jik}^{abc}.$$

**Virtual-triple banding.** Define $g(a,b,c) = \sum_{ijk} f_{ijk}^{abc}$, with
$i,j,k$ over the full occupied axes. Relabelling the dummy $(i,j,k)$ by the
same $\sigma$ shows that $g$ is symmetric in $(a,b,c)$ alone. Each multiset
$\{a,b,c\}$ therefore contributes $m(a,b,c)$ equal terms to $\sum_{abc} g$,
where $m$ counts its distinct orderings:

$$\sum_{abc} g(a,b,c) = \sum_{a\le b\le c} m(a,b,c)\, g(a,b,c),\qquad
m(a,b,c) = \begin{cases} 1 & a=b=c \\ 3 & \text{exactly two equal} \\ 6 & \text{all distinct,}\end{cases}$$

and $\sum_{a\le b\le c<n_v} m = n_v^3$. Hence

$$E_{(T)} = -\frac13 \sum_{a\le b\le c} m(a,b,c) \sum_{ijk}
\frac{\tilde W_{ijk}^{abc}\,V_{ijk}^{abc}}{\Delta_{ijk}^{abc}} .$$

Repeated indices are physical. Here $a,b,c$ (and $i,j,k$) label spatial
orbitals that each hold two electrons, so $a=b$ or $i=j$ gives a nonzero
contribution. In the spin-orbital formulation the antisymmetrized amplitude
vanishes for a repeated index, which is why that path can band over $i<j<k$
with a single weight. Transplanting that banding here (strict inequalities,
weight 6) drops these terms. On a repeated virtual triple the kernel still
sums all six permutations in $W$, some of them identical, as the definition
requires; $m$ only counts how many orderings of the multiset the unrestricted
sum would visit.

**Kernel.** The code maps onto these equations as follows.

- Layouts (`ccsd_t_closed_shell`, built once as `TripleIntegrals`):
  `ovvv[i,a,b,d]` $=(ia|bd)$, `vooo[a,i,j,l]` $=(ia|lj)$,
  `ovov[i,a,j,b]` $=(ia|jb)$, `t2x[c,d,j,k]` $=t_{kj}^{cd}$,
  `t1t[a,i]` $=t_i^a$. The denominators use the canonical HF orbital energies
  of the active occupied and virtual orbitals.
- `raw_w_pair` evaluates $r_\pi[n] = w^{(0)}{}_n^{v_\pi}$ for every occupied
  $n=(n_0,n_1,n_2)$ at once, for two permutations that share the third virtual
  label (`PERM_PAIRS`). Term 1 is one GEMM of the stacked
  `[ovvv[:,x0,y0,:]; ovvv[:,x1,y1,:]]` with `t2x[z]`; term 2 is
  `vooo[x]` times the transpose of `t2x[y,z]`, scaled by $-1$.
- `triple_energy(a,b,c)` accumulates $W_n = \sum_\pi r_\pi[n_\pi]$, the raw
  block read with its occupied axes permuted by $\pi^{-1}$. One fused pass
  over $(i,j,k)$ then forms $V$, the occupied-axis $\tilde W$ above and
  $-\tilde W V/\Delta$ (as $\tilde W V$ over the stored $-\Delta$), and returns $g(a,b,c)$.
- `pair_job_energy(a,b)` sums `triple_multiplicity(a,b,c)` $\times\,g(a,b,c)$
  over $c=b,\dots,n_v-1$. Jobs are the pairs $a\le b$ (`virtual_pairs`),
  folded serially in ascending order, so the result does not depend on the
  thread count.
- `ccsd_t_closed_shell` returns the folded sum divided by 3.

**DLPNO-(T).** `dlpno_ccsd_t.rs` keeps the occupied-banded form of the same
identity: it screens occupied triples, so it needs $f$ summed per occupied
triple. With $h(i,j,k) = \sum_{abc} f_{ijk}^{abc}$ (the `r3` pattern on the
virtual labels; `dlpno_ccsd_t_kernel.rs` evaluates it per triple), $h$ is
symmetric in $(i,j,k)$ by the same argument, and `screened_triple_energy` forms
$\frac13\sum_{i\le j\le k} m(i,j,k)\,h(i,j,k)$ over the retained triples, with
`occ_triple_multiplicity` giving the same $1,3,6$ weights. The screen retains
or drops whole multisets $\{i,j,k\}$ and always keeps the repeated-index
classes, so the weights stay exact.

**Checks.** `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`
proves $\sum_{a\le b\le c} m = n^3$ symbolically (claim 1). On random
closed-shell amplitudes and 8-fold-symmetric RI integrals it checks that $h$
is the same for all orderings of $(i,j,k)$ (claim 2), that the occupied
banding equals the unrestricted sum (claim 3), and that the unrestricted sum
over 3 equals the dense spin-orbital $(T)$ (claim 4); together these cover the
DLPNO-(T) form. Claim 4b ports the dense kernel (virtual banding, $\tilde W$
and singles on the occupied axes, divisor 3) and matches the spin-orbital
$(T)$ to a few $10^{-16}$ relative. Claim 5 shows that the
spin-orbital banding, divisor 6, unit weights and a $\tilde W$ without the
cyclic terms all fail. In Rust, the `ccsd_t_closed_shell` tests compare with
the spin-orbital `ccsd_t` given the same amplitudes and with PySCF.

## Unrestricted CCSD and CCSD(T)

`u_ccsd.rs::u_ccsd` and `u_ccsd_t.rs::u_ccsd_t` solve CCSD and its $(T)$
correction on a UHF reference. The $\alpha$ and $\beta$ orbitals are
different, so no spin adaptation is possible, but the spin-orbital
amplitudes vanish unless spin is conserved. The solver keeps only the
nonzero blocks. Lower-case indices are $\alpha$, upper-case $\beta$:

$$t_i^a,\quad t_I^A,\quad t_{ij}^{ab},\quad t_{iJ}^{aB} = t_{i\alpha J\beta}^{a\alpha B\beta},\quad t_{IJ}^{AB}.$$

`t2aa` and `t2bb` are antisymmetric in $(i,j)$ and in $(a,b)$. These are
PySCF's `uccsd` conventions. The other mixed-spin spin-orbital elements
follow from antisymmetry:

| spins of $(q,r,s,e)$ | $t_{qr}^{se}$ |
|---|---|
| $\alpha\alpha\alpha\alpha$ | $t_{qr}^{se}$ (`t2aa`) |
| $\beta\beta\beta\beta$ | $t_{QR}^{SE}$ (`t2bb`) |
| $\alpha\beta\alpha\beta$ | $t_{qR}^{sE}$ |
| $\alpha\beta\beta\alpha$ | $-t_{qR}^{eS}$ |
| $\beta\alpha\alpha\beta$ | $-t_{rQ}^{sE}$ |
| $\beta\alpha\beta\alpha$ | $t_{rQ}^{eS}$ |

Every other spin pattern is zero.

**Integrals.** Each spin has its own dressed RI factors
$B^P_{pq,\sigma}$ over its active occupied and virtual orbitals
(`URiBlocks`), so $(pq|rs)$ with $p,q$ of spin $\sigma$ and $r,s$ of spin
$\tau$ is $\sum_P B^P_{pq,\sigma}B^P_{rs,\tau}$. A chemist integral is zero
unless each bra and each ket pair shares a spin, and
$\langle pq\|rs\rangle = (pr|qs)\delta_{\sigma_p\sigma_r}\delta_{\sigma_q\sigma_s} - (ps|qr)\delta_{\sigma_p\sigma_s}\delta_{\sigma_q\sigma_r}$.
`UEris` holds the chemist blocks of PySCF's `_ChemistsERIs` for UCCSD, all
three `vvvv` blocks dense. The Fock matrix is taken as diagonal (canonical
UHF orbital energies), as on the restricted route.

**Amplitude equations.** The residuals are the spin-orbital CCSD residuals
of the section above with $p,q,\dots$ restricted to the spin-allowed
blocks. `update_amps` evaluates them in the factorization of PySCF
`uccsd.update_amps` (with $f_{ia}=0$), line for line: the $F_{oo}$,
$F_{vv}$, $F_{ov}$ intermediates per spin, the six $W_{ovvo}$ ring blocks
(`wovvo`, `wOVVO`, `woVvO`, `woVVo`, `wOvVo`, `wOvvO`), the hole-hole and
particle-particle ladders on $\tau$, and the singles. The update is
$t \leftarrow r/(-\Delta)$ with the per-spin denominators, and DIIS runs
on all five blocks jointly.

**Energy.** With $\tau_{ij}^{ab} = t_{ij}^{ab} + t_i^at_j^b - t_j^at_i^b$
the spin-orbital energy $\tfrac14\sum\langle ij\|ab\rangle\tau_{ij}^{ab}$
splits by the spins of $(i,j)$:

$$
\begin{aligned}
E_{\alpha\alpha} &= \tfrac14\sum_{ijab}\big[(ia|jb)-(ib|ja)\big]t_{ij}^{ab}
 + \tfrac12\sum_{ijab}\big[(ia|jb)-(ib|ja)\big]t_i^at_j^b, \\
E_{\alpha\beta} &= \sum_{iJaB}(ia|JB)\big(t_{iJ}^{aB} + t_i^at_J^B\big),
\end{aligned}
$$

and $E_{\beta\beta}$ is $E_{\alpha\alpha}$ with $\beta$ blocks. The $\alpha\beta$
term already counts both $(i\alpha,J\beta)$ and $(J\beta,i\alpha)$ orderings.

**(T).** `u_ccsd_t` evaluates the spin-orbital formula of the section
"CCSD(T), spin-orbital" on the spin blocks. The summand
$f = (W+V)W/\Delta$ is invariant under any permutation of $(i,j,k)$ and,
separately, any permutation of $(a,b,c)$. It vanishes unless the occupied
and virtual triples carry the same spins, and unless $i,j,k$ are distinct.
Classify the occupied triple by its number of $\beta$ indices,
$n_\beta = 0,1,2,3$ (classes aaa, aab, abb, bbb). For each class the loop
visits canonical occupied triples ($i<j<k$ within one spin) and one
canonical virtual spin layout ($\alpha$ positions first), with each
same-spin virtual index running over its full range:

| class | occupied loop | virtual layout | weight |
|---|---|---|---|
| aaa | $i<j<k$ | $(\alpha,\alpha,\alpha)$ | $1/6$ |
| aab | $i<j$, $K$ | $(\alpha,\alpha,\beta)$ | $1/2$ |
| abb | $i$, $J<K$ | $(\alpha,\beta,\beta)$ | $1/2$ |
| bbb | $I<J<K$ | $(\beta,\beta,\beta)$ | $1/6$ |

A canonical occupied triple stands for $3!$ ordered ones, and a canonical
virtual layout stands for $\binom{3}{n_\beta}$ placements of its $\beta$
labels. The weight is therefore

$$w(n_\beta) = \frac{3!\binom{3}{n_\beta}}{36},$$

and $E_{(T)} = -\sum_{\text{class}} w \sum_{\text{canonical}} (W+V)W/\Delta$.
For each occupied triple the kernel builds $X$ and $Y$ block by block: a
term enters only when the spins of its $t_2$ element (table above) and its
integral both match, and $P(i/jk)P(a/bc)$ acts on the three occupied
labels and on the three virtual positions with their spins. The loop runs
over canonical triples of all four classes in parallel and folds them
serially in a fixed order. The class energies are reported.

**Frozen core.** `frozen_core` drops the lowest `frozen_core` orbitals of
each spin, as `rimp2::active_occ` does for U-RI-MP2.

**Limits.**

- Closed-shell molecule, RHF orbitals given to both spins: $t_{ij}^{ab}$
  (`t2aa`) $= t_{iJ}^{aB} - t_{iJ}^{bA}$, $E_{\alpha\alpha} = E_{\beta\beta}$,
  aaa $=$ bbb and aab $=$ abb. The CCSD energy, $t_{iJ}^{aB}$ and
  $E_{(T)}$ equal those of `ccsd_closed_shell` and `ccsd_t_closed_shell`.
- On the same UHF orbitals, the spin-orbital solver of `ccsd.rs` and
  `ccsd_t.rs`, fed the $\alpha$-then-$\beta$ spin-orbital space
  (`ccsd_uhf_spin_orbital`, `ccsd_t_uhf_spin_orbital`), has no spin
  bookkeeping and must give the same amplitudes, energy and $(T)$ classes.

**Checks.** `scripts/cc_uccsd_spin_blocks_sympy.py` proves the class
weights with sympy and a brute-force count (claim 1). On random spin-blocked
amplitudes and per-spin RI factors with unequal $\alpha$ and $\beta$ spaces it
checks that the transcribed spin-blocked $(T)$ equals the dense spin-orbital
$(T)$ per class (claim 2), that the energy split above equals the dense
spin-orbital energy (claim 3) and the closed-shell limit (claim 4). Claim 5
shows that an aab weight of $1/6$, a wrong sign in the $t_2$ table, swapped
$\alpha$/$\beta$ virtual energies, a dropped $t_I^A$ and a dropped exchange term
each fail. In Rust, `tests/uccsd_anchors.rs` runs both limits above and
`tests/validation_uccsd.rs` compares with PySCF DF-UCCSD and UCCSD(T). The
two Rust solvers share the per-spin RI factors and orbital energies, so a
fault in those inputs is caught only by the PySCF comparison.

## EE-EOM-CCSD singlets, closed shell

`eom_ccsd.rs` (Stanton & Bartlett 1993; closed-shell intermediates and
product of PySCF `eom_rccsd._IMDS.make_ee` / `eeccsd_matvec_singlet`). The
excitation energies are the eigenvalues of $\bar H - E_{CC}$,
$\bar H = e^{-T} H e^{T}$, on singlet singles and doubles
$r_i^a$, $r_{ij}^{ab} = r_{ji}^{ba}$. At converged amplitudes that matrix is
the CC Jacobian:

$$(\bar H - E_{CC})\, r = \frac{\partial \Omega(t)}{\partial t}\, r,$$

where $\Omega$ is the closed-shell CCSD residual. The prototype
(`scripts/prototypes/eom_ccsd_proto.py`) checks the product against this
Jacobian by a 5-point central difference, which is exact for the quartic
$\Omega$.

The product (`sigma`) uses the dressed intermediates of `intermediates`: $F_{oo}, F_{vv}, F_{ov}$, $W_{mnij}$, $W_{mbij}$, $W_{ejab}$ and the
two $W_{mbej}$ couplings. $\tau$ with a right vector is
`make_tau`, $\theta = 2r_{ij}^{ab} - r_{ij}^{ba}$ is `theta_of`,
and the doubles part is symmetrized at the end, $\sigma_{ijab} \leftarrow
\sigma_{ijab} + \sigma_{jiba}$. The particle–particle ladder takes the dense
$(ae|bf)$ block or RI panels, the same choice as the CCSD ladder.

At $t = 0$ the singles–singles block is the CIS matrix
$\delta_{ij}\delta_{ab}(\varepsilon_a - \varepsilon_i) + 2(ia|jb) - (ij|ab)$.

The packed vector (`pack`) is $r_1$ then the lower triangle of
$M_{ia,jb} = r_{ij}^{ab}$. The reported singles weight uses the spin-summed
norm (`spin_norm`):

$$\|R\|^2 = 2\|r_1\|^2 + \|r_2\|^2 + \tfrac12 \|r_2 - r_2^{(ab)}\|^2,$$

with $r_2^{(ab)}$ the vector with $a,b$ swapped.

The solver (`eom_davidson.rs::davidson_nonsymmetric`) diagonalizes
the projected matrix with `dgeev`, keeps the lowest $n$ roots by real part
above $10^{-10}$, and adds the Olsen correction (`olsen_correction`)

$$\delta = -(\Delta - \theta)^{-1} r + \epsilon (\Delta - \theta)^{-1} x,\qquad
\epsilon = \frac{x^\top (\Delta-\theta)^{-1} r}{x^\top (\Delta-\theta)^{-1} x},$$

where $\Delta$ is the diagonal $F_{vv,aa} - F_{oo,ii}$ (doubles: the sum over both
pairs). For a non-symmetric matrix the eigenvalue error is first order in the
residual $\|r\|$, not second.

**Checks.** `scripts/cc_ccsd_eom_brute_force_sympy.py`: `eom-jacobian`
builds $e^{-T}He^{T}$ as a matrix at converged CCSD amplitudes of a
ten-spin-orbital model and shows that its singles-and-doubles block minus
$E_{CC}$ equals the 5-point Jacobian of the spin-orbital residual of the CCSD
section; `cis-limit` the singlet singles block at $t = 0$; `spin-norm` that
the norm above is the norm of the spin-orbital expansion of $(r_1, r_2)$ over
distinct determinants. The closed-shell intermediates of `sigma` are a
transcription of PySCF and are checked against it
(`tests/validation_eom_ccsd.rs`) and against the CIS block at $t=0$
(`tests/eom_ccsd_anchors.rs`).

## LinLCCD and LinLCCD(hh)

Carter-Fenk, JPCA 129, 7251 (2025). Linearized ladder CCD keeps the driver,
Fock and ladder terms of CCD and drops rings, crossed rings and all
quadratic terms (eq. 7):

$$0 = \langle ij\|ab\rangle + \Delta_{ij}^{ab}\, t_{ij}^{ab}
+ \tfrac12 \sum_{kl}\langle kl\|ij\rangle\, t_{kl}^{ab}
+ \tfrac12 \sum_{cd}\langle ab\|cd\rangle\, t_{ij}^{cd}.$$

`LadderVariant` (`linlccd.rs`) selects the ladder terms:

| Variant | Terms kept | Blocks built |
|---|---|---|
| `DriversOnly` | driver + Fock (= MP2 amplitude equations) | OOVV |
| `Hh` — LinLCCD(hh), eq. 14 | + hole–hole ladder; $O(n_o^4 n_v^2)$ | OOVV, OOOO |
| `Full` — LinLCCD | + particle–particle ladder; $O(n_o^2 n_v^4)$ | OOVV, OOOO, VVVV |

The hh truncation is the absence of the pp term: `Hh` never forms the VVVV
block (`needs_vvvv`, `linlccd.rs`).

**Spin-orbital solver** (`linlccd.rs`, restricted orbitals; `linlccd_u.rs`,
unrestricted). Jacobi iteration with DIIS on
$t \leftarrow -R/\Delta$, starting from the MP2 guess $t = -\langle ij\|ab\rangle/\Delta$:

$$R_{ij}^{ab} = \langle ij\|ab\rangle + \tfrac12\sum_{kl}\langle kl\|ij\rangle t_{kl}^{ab}
+ \tfrac12\sum_{cd} t_{ij}^{cd}\langle ab\|cd\rangle\ (\text{Full only}),$$

$$E_c = \tfrac14\sum_{ijab}\langle ij\|ab\rangle\, t_{ij}^{ab}.$$

`linlccd_u.rs` builds the same blocks from per-spin orbitals
(`ferric_mp2::spinorbital_u`); a slot where one spin has fewer orbitals than
the other gets $\varepsilon_i = 10^6$, $\varepsilon_a = -10^6$, so its
zero integrals give a zero amplitude without a $0/0$. A raw ROHF reference is
rejected; it must be semicanonicalized first. `linlccd_exact.rs` solves the
same equations from exact four-centre integrals.

**Spatial closed-shell form** (`linlccd_amplitude.rs`, module doc). With
$T_{iajb} = t_{i\alpha j\beta}^{a\alpha b\beta}$,

$$0 = (ia|jb) + \Delta_{ij}^{ab} T_{iajb} + \sum_{kl}(ik|jl)\, T_{kalb} + \sum_{cd}(ac|bd)\, T_{icjd},\qquad
E_c = \sum_{iajb}(ia|jb)\,(2T_{iajb} - T_{ibja}).$$

It gives the same $E_c$ and amplitudes as the spin-orbital equations for all
three variants (`scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`, claim 6).

### ωB97X-L-V correlation

`double_hybrid.rs` (Ransford & Carter-Fenk 2026, eqns 22 and 27). $\lambda$
multiplies the driver and the hh ladder; the Fock terms stay unscaled. The
solver therefore runs LinLCCD(hh) once under the operator
$\lambda\,\mathrm{erfc}(\omega r)/r$ (`lambda_scaled_sr_operator`): the
three-centre integrals and the RI metric both scale by $\lambda$, so
$B^P_{pq}$ scales by $\sqrt\lambda$ and every four-index integral by $\lambda$.
The returned energy $\tfrac14\sum(\lambda\langle ij\|ab\rangle^{sr})t$ is
$\lambda E_c^{sr,\omega,\lambda}$, quadratic in $\lambda$ at leading order
(exactly $\lambda^2 E_{c,\mathrm{MP2}}^{sr}$ with the ladder off).

**The whole functional** (Ransford and Carter-Fenk, Phys. Chem. Chem. Phys.
28, 14428 (2026), eqn 27) is

$$ E = E_{\mathrm{KS}}[\omega\mathrm{B97X\text{-}L}] + E_{c,\mathrm{VV10}} + \lambda E_c^{sr,\omega,\lambda}, \qquad \lambda = 0.6,\ \omega = 0.1\ \mathrm{Bohr}^{-1}. $$

The KS half (`ferric-dft/src/libxc.rs` `wb97x_l_v_def`) is libxc's
`HYB_GGA_XC_WB97X_V` form with the paper's Table 2 coefficients set through
the external-parameter interface (`WB97X_L_V_EXT_PARAMS`, checked by name
against libxc's order) and its own VV10 ($b = 10.0$, $C = 0.01$,
`WB97X_L_V_VV10`). Exact exchange is full at long range and $\lambda$-scaled at
short range: in libxc's CAM convention $\alpha = 1$, $\beta = -0.4$, so
$c_{\mathrm{lr}} = \alpha = 1$ and $c_{\mathrm{sr}} = \alpha + \beta = 0.6 = \lambda$,
with $\omega = 0.1$; the complementary semilocal exchange carries
$c_{x,0} = 1 - \lambda = 0.4$ and the same- and opposite-spin correlation
$c_{ss,0} = c_{os,0} = 1 - \lambda^2 = 0.64$, the uniform-electron-gas
constraints; the other twelve B97 coefficients are fitted:

| | $c_{\cdot,0}$ | $c_{\cdot,1}$ | $c_{\cdot,2}$ | $c_{\cdot,3}$ | $c_{\cdot,4}$ |
|---|---|---|---|---|---|
| exchange $c_x$ | 0.4 | 0.154 | 3.884 | −11.300 | 8.425 |
| same-spin $c_{ss}$ | 0.64 | −1.417 | 4.716 | −2.956 | −0.861 |
| opposite-spin $c_{os}$ | 0.64 | −1.194 | −1.348 | 11.869 | −9.571 |

The SCF converges the KS half alone (`run_wb97x_l_v`: the KS convergence
ladder, RI-J and RI-K on def2-universal-jkfit unless the input names another
fit, an error if no rung converges); the LinLCCD(hh) term is evaluated once
on its orbitals and orbital energies, with the RI auxiliary basis of `[mp2]
auxbasis`.

**Overrides.** `DoubleHybridConfig` carries $\lambda$ and $\omega$ for the
correlation term only, and the CLI's `[dft] lambda` and `[dft] omega` set
those two fields. `wb97x_l_v_def` takes no $\lambda$ or $\omega$: its
$c_{\mathrm{sr}}$, $\omega$ and the constrained $c_{x,0}$, $c_{ss,0}$,
$c_{os,0}$ stay at the published 0.6 and 0.1. An override therefore changes
the correlation term on the published KS reference, not the functional as a
whole. $\lambda$ must lie in $[0, 1]$ and $\omega$ must be positive; at
$\lambda = 0$ no correlation is computed (the operator, and with it the RI
metric, vanishes).

[`scripts/double_hybrid_sympy.py`](https://github.com/mgoldey/ferric/blob/main/scripts/double_hybrid_sympy.py)
proves (P) that the Rust tables equal Table 2 and the constraints above, and
that `wb97x_l_v_def` takes only `nspin`; and (L) on a model, that LinLCCD(hh)
with the three-centre integrals and metric both scaled by $\lambda$ equals
$\lambda E_c$ of eqn 22 with $\lambda$ on the driver and the hh ladder, that
with the ladder off it is $\lambda^2E_{\mathrm{MP2}}$, that the linear form
$\lambda E_c(\lambda = 1)$ misses it, and that at $\lambda = 1$ with the ladder
off it is the brute-force MP2 doubles sum.

## Amplitude-threshold LinLCCD

`linlccd_amplitude.rs`, closed shell, in Boys-localized occupieds and VV-HV
localized virtuals (`ferric_mp2::lmp2_amplitude`). The amplitude pattern
keeps $(i,a,j,b)$ iff

$$|(ia|jb)| > \epsilon \quad\text{or}\quad |(ib|ja)| > \epsilon,$$

the swap-closed test of `ferric_mp2::lmp2_amplitude`. The spatial equations above are solved by
pattern-projected CG on the ragged per-pair blocks (`linlccd_masked_solve`): the Fock part in the localized basis, the hh ladder as a gather
over source pairs with coefficient $(ik|jl)$, and for `Full` the pp
block $\sum_P B^P_{ac} B^P_{bd}$ restricted to each pair's virtual domain. The energy is the linear form $\sum (ia|jb)(2T - \tilde T)$ over the
pattern, so its error is first order in the CG residual
(`cg_rtol` default $10^{-11}$). $\epsilon = 0$ keeps every amplitude and
reproduces exact LinLCCD of the same variant. The optional integral-free pair
gate (`pair_gate_cal`, direct path only) drops pair $(i,j)$, $i\ne j$, when
$c\,(s_i s_j)^3/R_{ij}^6 < 10^{-2}\epsilon$, with $s$ the orbital spreads.

## DLPNO-LinLCCD(hh)

`dlpno_linlccd.rs`. The LinLCCD(hh) iteration is closed in per-pair PNO
bases; spin-orbital indices are used throughout, so pairs run over $2n_o$
occupied spin orbitals.

**PNOs.** From the canonical MP2 guess
$t_{ij}^{ab} = -\langle ij\|ab\rangle/\Delta_{ij}^{ab}$, each retained pair
gets the pair density (`ferric_mp2::local_pno::pno_for_pair`, local_pno.rs)

$$D^{ij} = T^{ij} T^{ij\,\mathsf T} + T^{ij\,\mathsf T} T^{ij},\qquad (T^{ij})_{ab} = t_{ij}^{ab},$$

and keeps the eigenvectors with $|n| \ge t_{\mathrm{cut,pno}}$ (`local_pno.rs`);
if none survives, the largest one is kept. $D^{ij}$ carries no
$1/(1+\delta_{ij})$ normalization and, in the spin-orbital basis, no
closed-shell $\tilde T = 2T - T^{\mathsf T}$, so $t_{\mathrm{cut,pno}}$ values
are not interchangeable with ORCA's `TCutPNO`. The kept columns $Q_{ij}$ are
semicanonicalized: $F^{ij} = Q_{ij}^{\mathsf T}\,\mathrm{diag}(\varepsilon_v)\,Q_{ij}$
is diagonalized and $\tilde Q_{ij} = Q_{ij} U$, giving PNO energies
$\tilde\varepsilon^{ij}_{\tilde a}$ (`PairPnoBasis::build`, dlpno_ccsd_virtual.rs).
Denominators are $\tilde\Delta^{ij}_{\tilde a\tilde b} = \tilde\varepsilon^{ij}_{\tilde a}+\tilde\varepsilon^{ij}_{\tilde b}-\varepsilon_i-\varepsilon_j$
(`pno_denominators`, dlpno_ccsd_kernel.rs, stores $-\tilde\Delta$).

**Iteration** (`dlpno_linlccd_hh`). With
$S^{ij,kl} = \tilde Q_{ij}^{\mathsf T}\tilde Q_{kl}$ (`pair_overlap`,
dlpno_ccsd_kernel.rs),

$$\tilde R^{ij} = \tilde v^{ij} + \tfrac12\sum_{kl}\langle kl\|ij\rangle\, S^{ij,kl}\,\tilde t^{kl}\,(S^{ij,kl})^{\mathsf T},\qquad
\tilde t^{ij} \leftarrow -\tilde R^{ij} / \tilde\Delta^{ij},$$

$$E_c = \tfrac14\sum_{i\le j} w_{ij} \sum_{\tilde a\tilde b} \tilde v^{ij}_{\tilde a\tilde b}\tilde t^{ij}_{\tilde a\tilde b},\qquad w_{ii}=1,\ w_{i<j}=2.$$

Only the $i\le j$ block is stored; $t_{lk} = -t_{kl}$ with $a,b$ in place
(`so_oriented_amp`). At $t_{\mathrm{cut,pno}} = 0$ every $S$ is
orthogonal and the iteration reproduces `linlccd(.., Hh)`.

## DLPNO-CCSD: pair screen and PNO substrate

**Pair lists** (`ferric_mp2`). Two criteria build the retained pairs
$i\le j$; diagonal pairs are always kept:

- distance (`pair_domains::build_pair_domains`, pair_domains.rs): keep
  $(i,j)$ if $|\mathbf r_i - \mathbf r_j| \le R_{\mathrm{pair}}$ (Boys centres, Bohr);
- estimated MP2 pair energy (`pair_energy_screen`): keep $(i,j)$ if
  $|e_{ij}| \ge T_{\mathrm{cut,pairs}}$, with
  $e_{ij} = -(2-\delta_{ij})\sum_{ab} (2(ia|jb) - (ib|ja))(ia|jb)/\Delta_{ij}^{ab}$.

`dlpno_ccsd::apply_pair_mask` (dlpno_ccsd.rs) zeroes $t_{ij}$ and
$t_{ji}$ of every screened pair in the closed-shell CCSD amplitudes; with
complete domains it is the identity.

**PNO basis** (`dlpno_ccsd_virtual.rs`): the same `pno_for_pair` criterion on
the closed-shell spatial $T^{ij}$, the same semicanonicalization, and the
closed-shell energy
$E = \sum_{ij}\sum_{\tilde a\tilde b}\tilde\tau^{ij}_{\tilde a\tilde b}(2\tilde g^{ij}_{\tilde a\tilde b} - \tilde x^{ij}_{\tilde a\tilde b})$
with $g_{ab} = (ia|jb)$, $x_{ab} = (ib|ja)$ rotated by $\tilde Q_{ij}$
(`pno_ccsd_energy`). `dlpno_ccsd_kernel.rs` evaluates individual residual
contractions in PNO bases with $S$ inserted; a closed PNO-basis CCSD
iteration does not exist.

## DLPNO-CCSD(T): triple screen and TNOs

**Triple list** (`dlpno_ccsd_t.rs`). A banded triple $i\le j\le k$ is kept if
any two indices coincide, or else if all three pairwise Boys distances are
$\le R_{\mathrm{triple}}$ (`triple_is_retained`). The predicate depends
only on the multiset $\{i,j,k\}$, so retained triples keep the dense weights
$m\in\{1,3,6\}$, and
$E_{(T)} = \tfrac13\sum_{\text{kept}} m\, f(i,j,k)$
(`screened_triple_energy`). Only $m=6$ triples can be screened.

**TNOs** (`dlpno_ccsd_t_virtual.rs`). Each constituent pair
$(p,q)\subset\{i,j,k\}$ gets a PNO basis $P^{pq}$ at $t_{\mathrm{cut,tno}}$
(same `pno_for_pair` criterion). The union is the range of the
projector sum

$$M^{ijk} = \sum_{(p,q)} P^{pq} P^{pq\,\mathsf T},$$

taking eigenvectors with eigenvalue $> 10^{-8}$ (`UNION_RANK_TOL`); in-span eigenvalues are $\ge 1$, out-of-span ones are 0. The pairs are
summed in sorted order, so the basis is bit-identical for all orderings of
$(i,j,k)$. The union is then semicanonicalized as for PNOs, and the per-triple
$-\sum \tilde W V/\Delta$ is evaluated with every virtual index (free and summed)
projected onto that basis (`dlpno_ccsd_t_kernel::triple_contribution_in_tno`). The kernel returns the unweighted $f(i,j,k)$; $m$ and the divisor 3 are
applied only in `screened_triple_energy`. At $t_{\mathrm{cut,tno}} = 0$ every
transform is square orthogonal and the dense $f$ is recovered.
