# Coupled Cluster Theory

Equations as `crates/ferric-cc/src` implements them, with the file and line
of each. Two conventions are used:

- **Spin-orbital** (`ccd`, `ccsd`, `ccsd_t`, `linlccd`, `linlccd_u`,
  `linlccd_exact`, `dlpno_linlccd`): indices run over $2n_o$ / $2n_v$ spin
  orbitals, interleaved (even = $\alpha$, odd = $\beta$);
  $\langle pq\|rs\rangle = (pr|qs) - (ps|qr)$, with chemist integrals
  $(pq|rs) = \sum_P B^P_{pq} B^P_{rs}$ from the dressed RI factors
  (`ferric_mp2::spinorbital::asym_*`).
- **Spatial, closed shell** (`ccsd_closed_shell`, `ccsd_t_closed_shell`,
  `linlccd_amplitude`, `dlpno_ccsd*`): indices run over $n_o$ / $n_v$
  spatial orbitals and amplitudes satisfy $t_{ij}^{ab} = t_{ji}^{ba}$.

Denominators are $D_{ij}^{ab} = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b$
(negative for a gapped reference). Canonical HF orbitals are assumed, so
$f_{ia} = 0$ and the occupied/virtual Fock blocks are diagonal.

## Exponential ansatz

$$|\Psi_{CC}\rangle = e^{\hat{T}} |\Phi_0\rangle,\qquad \hat{T} = \hat{T}_1 + \hat{T}_2 + \dots$$

## CCD

$\langle ab | \hat{H}_N e^{\hat{T}_2} | ij \rangle_C = 0$, solved by Jacobi
iteration plus DIIS. Energy (`ccd.rs:271`):

$$E_c = \tfrac14 \sum_{ijab} \langle ij\|ab\rangle\, t_{ij}^{ab}.$$

The closed-shell spatial form of the same energy is
$E_c = \sum_{ijab} (ia|jb)\,(2 t_{ij}^{ab} - t_{ij}^{ba})$.

## CCSD

$R_i^a = \langle a | \hat{H}_N e^{\hat{T}_1 + \hat{T}_2} | i \rangle_C = 0$ and
$R_{ij}^{ab} = \langle ab | \hat{H}_N e^{\hat{T}_1 + \hat{T}_2} | ij \rangle_C = 0$,
built from effective Fock blocks $F_{oo}, F_{vv}, F_{ov}$ and dressed
two-electron intermediates $W_{oooo}, W_{vvvv}, W_{ovvo}$. The closed-shell
solver (`ccsd_closed_shell.rs`) transcribes Hirata, JCP 120, 2581 (2004),
Eqs. 35–45; the spin-orbital solver (`ccsd.rs`) is its cross-check.

## CCSD(T), spin-orbital

`ccsd_t.rs` (module doc; PySCF `gccsd_t_slow` convention):

$$W_{ijk}^{abc} = P(i/jk)\,P(a/bc)\Big[\sum_e t_{jk}^{ae}\langle bc\|ei\rangle - \sum_m t_{im}^{bc}\langle ma\|jk\rangle\Big],\qquad
V_{ijk}^{abc} = P(i/jk)\,P(a/bc)\big[t_i^a \langle bc\|jk\rangle\big],$$

$$E_{(T)} = \frac{1}{36}\sum_{ijkabc} \frac{(W + V)_{ijk}^{abc}\, W_{ijk}^{abc}}{D_{ijk}^{abc}},\qquad
D_{ijk}^{abc} = \varepsilon_i+\varepsilon_j+\varepsilon_k-\varepsilon_a-\varepsilon_b-\varepsilon_c,$$

with $P(i/jk)x = x - x_{i\leftrightarrow j} - x_{i\leftrightarrow k}$. The
summand is invariant under any simultaneous relabelling of $(i,j,k)$ and of
$(a,b,c)$, so the streaming loop visits each $i<j<k$ once with $a,b,c$ over
the full axis and divides by 6 instead of 36.

## CCSD(T), closed shell

`ccsd_t_closed_shell.rs` (Rendell, Lee & Komornicki 1991; index convention of
PySCF `ccsd_t_slow`). For an ordered occupied triple $(i,j,k)$
(`raw_w_block`, :297):

$$w^{(0)}_{abc}(i,j,k) = \sum_d (ia|bd)\, t_{kj}^{cd} - \sum_l (ia|lj)\, t_{lk}^{bc}.$$

$W$ sums $w^{(0)}$ over the six simultaneous permutations $\pi$ of the
(occupied, virtual) pairs $(i,a),(j,b),(k,c)$ (`w_block`, :348):

$$W_{abc} = \sum_{\pi\in S_3} w^{(0)}(i_\pi, j_\pi, k_\pi)_{a_\pi b_\pi c_\pi}.$$

Singles (`add_v_singles`, :376):

$$V_{abc} = W_{abc} + (jb|kc)\, t_i^a + (ia|kc)\, t_j^b + (ia|jb)\, t_k^c.$$

Spin-summed weight (PySCF `r3`, :601–607) and energy (:611–634):

$$\tilde W_{abc} = 4W_{abc} + W_{bca} + W_{cab} - 2W_{cba} - 2W_{acb} - 2W_{bac},$$

$$E_{(T)} = \frac13 \sum_{i\le j\le k} m(i,j,k) \sum_{abc} \frac{\tilde W_{abc} V_{abc}}{D_{ijk}^{abc}},\qquad
m(i,j,k) = \begin{cases} 1 & i=j=k \\ 3 & \text{exactly two equal} \\ 6 & \text{all distinct}\end{cases}$$

(`occ_triple_multiplicity`, :171). The per-triple sum
$f(i,j,k) = \sum_{abc}\tilde W V/D$ is the same for all orderings of
$(i,j,k)$, and $\sum_{i\le j\le k} m = n_o^3$, so the band equals the
unrestricted sum $\sum_{ijk} f$, which is $3E_{(T)}$. Repeated spatial indices
($i=j$, $i=j=k$) are physical here, unlike the spin-orbital case where the
antisymmetrized amplitude vanishes: the spin-orbital banding ($i<j<k$, weight 6)
drops them. Checked in `wiki/sympy_derivations.md` §23.

## LinLCCD and LinLCCD(hh)

Carter-Fenk, JPCA 129, 7251 (2025). Linearized ladder CCD keeps the driver,
Fock and ladder terms of CCD and drops rings, crossed rings and all
quadratic terms (eq. 7):

$$0 = \langle ij\|ab\rangle + \Delta_{ij}^{ab}\, t_{ij}^{ab}
+ \tfrac12 \sum_{kl}\langle kl\|ij\rangle\, t_{kl}^{ab}
+ \tfrac12 \sum_{cd}\langle ab\|cd\rangle\, t_{ij}^{cd},\qquad \Delta_{ij}^{ab} = -D_{ij}^{ab}.$$

`LadderVariant` (`linlccd.rs`) selects the ladder terms:

| Variant | Terms kept | Blocks built |
|---|---|---|
| `DriversOnly` | driver + Fock (= MP2 amplitude equations) | OOVV |
| `Hh` — LinLCCD(hh), eq. 14 | + hole–hole ladder; $O(n_o^4 n_v^2)$ | OOVV, OOOO |
| `Full` — LinLCCD | + particle–particle ladder; $O(n_o^2 n_v^4)$ | OOVV, OOOO, VVVV |

The hh truncation is the absence of the pp term: `Hh` never forms the VVVV
block (`needs_vvvv`, `linlccd.rs:112`).

**Spin-orbital solver** (`linlccd.rs`, restricted orbitals; `linlccd_u.rs`,
unrestricted). Jacobi iteration with DIIS on
$t \leftarrow R/D$ (:388), starting from the MP2 guess $t = \langle ij\|ab\rangle/D$ (:345):

$$R_{ij}^{ab} = \langle ij\|ab\rangle + \tfrac12\sum_{kl}\langle kl\|ij\rangle t_{kl}^{ab}\ (\text{:378})
+ \tfrac12\sum_{cd} t_{ij}^{cd}\langle ab\|cd\rangle\ (\text{:383, Full only}),$$

$$E_c = \tfrac14\sum_{ijab}\langle ij\|ab\rangle\, t_{ij}^{ab}\quad(\text{:358}).$$

`linlccd_u.rs` builds the same blocks from per-spin orbitals
(`ferric_mp2::spinorbital_u`); a slot where one spin has fewer orbitals than
the other gets $\varepsilon_i = 10^6$, $\varepsilon_a = -10^6$ (:311), so its
zero integrals give a zero amplitude without a $0/0$. A raw ROHF reference is
rejected; it must be semicanonicalized first. `linlccd_exact.rs` solves the
same equations from exact four-centre integrals.

**Spatial closed-shell form** (`linlccd_amplitude.rs`, module doc). With
$T_{iajb} = t_{i\alpha j\beta}^{a\alpha b\beta}$,

$$0 = (ia|jb) + \Delta_{ij}^{ab} T_{iajb} + \sum_{kl}(ik|jl)\, T_{kalb} + \sum_{cd}(ac|bd)\, T_{icjd},\qquad
E_c = \sum_{iajb}(ia|jb)\,(2T_{iajb} - T_{ibja}).$$

It gives the same $E_c$ and amplitudes as the spin-orbital equations for all
three variants (§23 claim 6).

### ωB97X-L-V correlation

`double_hybrid.rs` (Ransford & Carter-Fenk 2026, eqns 22 and 27). $\lambda$
multiplies the driver and the hh ladder; the Fock terms stay unscaled. The
solver therefore runs LinLCCD(hh) once under the operator
$\lambda\,\mathrm{erfc}(\omega r)/r$ (`lambda_scaled_sr_operator`, :127): the
three-centre integrals and the RI metric both scale by $\lambda$, so
$B^P_{pq}$ scales by $\sqrt\lambda$ and every four-index integral by $\lambda$.
The returned energy $\tfrac14\sum(\lambda\langle ij\|ab\rangle^{sr})t$ is
$\lambda E_c^{sr,\omega,\lambda}$, quadratic in $\lambda$ at leading order
(exactly $\lambda^2 E_{c,\mathrm{MP2}}^{sr}$ with the ladder off).

## Amplitude-threshold LinLCCD

`linlccd_amplitude.rs`, closed shell, in Boys-localized occupieds and VV-HV
localized virtuals (`ferric_mp2::lmp2_amplitude`). The amplitude pattern
keeps $(i,a,j,b)$ iff

$$|(ia|jb)| > \epsilon \quad\text{or}\quad |(ib|ja)| > \epsilon,$$

the swap-closed test of `ferric_mp2::lmp2_amplitude` (derivation
`wiki/sympy_derivations.md` §14). The spatial equations above are solved by
pattern-projected CG on the ragged per-pair blocks (`linlccd_masked_solve`,
:299): the Fock part in the localized basis (:316), the hh ladder as a gather
over source pairs with coefficient $(ik|jl)$ (:331), and for `Full` the pp
block $\sum_P B^P_{ac} B^P_{bd}$ restricted to each pair's virtual domain
(:356). The energy is the linear form $\sum (ia|jb)(2T - \tilde T)$ over the
pattern (:407), so its error is first order in the CG residual
(`cg_rtol` default $10^{-11}$). $\epsilon = 0$ keeps every amplitude and
reproduces exact LinLCCD of the same variant. The optional integral-free pair
gate (`pair_gate_cal`, direct path only) drops pair $(i,j)$, $i\ne j$, when
$c\,(s_i s_j)^3/R_{ij}^6 < 10^{-2}\epsilon$, with $s$ the orbital spreads.

## DLPNO-LinLCCD(hh)

`dlpno_linlccd.rs`. The LinLCCD(hh) iteration is closed in per-pair PNO
bases; spin-orbital indices are used throughout, so pairs run over $2n_o$
occupied spin orbitals.

**PNOs.** From the canonical MP2 guess
$t_{ij}^{ab} = \langle ij\|ab\rangle/D_{ij}^{ab}$ (:584), each retained pair
gets the pair density (`ferric_mp2::local_pno::pno_for_pair`, local_pno.rs:147)

$$D^{ij} = T^{ij} T^{ij\,\mathsf T} + T^{ij\,\mathsf T} T^{ij},\qquad (T^{ij})_{ab} = t_{ij}^{ab},$$

and keeps the eigenvectors with $|n| \ge t_{\mathrm{cut,pno}}$ (local_pno.rs:172);
if none survives, the largest one is kept (:177). $D^{ij}$ carries no
$1/(1+\delta_{ij})$ normalization and, in the spin-orbital basis, no
closed-shell $\tilde T = 2T - T^{\mathsf T}$, so $t_{\mathrm{cut,pno}}$ values
are not interchangeable with ORCA's `TCutPNO`. The kept columns $Q_{ij}$ are
semicanonicalized: $F^{ij} = Q_{ij}^{\mathsf T}\,\mathrm{diag}(\varepsilon_v)\,Q_{ij}$
is diagonalized and $\tilde Q_{ij} = Q_{ij} U$, giving PNO energies
$\tilde\varepsilon^{ij}_{\tilde a}$ (`PairPnoBasis::build`, dlpno_ccsd_virtual.rs:138).
Denominators are $\tilde D^{ij}_{\tilde a\tilde b} = \varepsilon_i+\varepsilon_j-\tilde\varepsilon^{ij}_{\tilde a}-\tilde\varepsilon^{ij}_{\tilde b}$
(`pno_denominators`, dlpno_ccsd_kernel.rs:381).

**Iteration** (`dlpno_linlccd_hh`, :623–700). With
$S^{ij,kl} = \tilde Q_{ij}^{\mathsf T}\tilde Q_{kl}$ (`pair_overlap`,
dlpno_ccsd_kernel.rs:134),

$$\tilde R^{ij} = \tilde v^{ij} + \tfrac12\sum_{kl}\langle kl\|ij\rangle\, S^{ij,kl}\,\tilde t^{kl}\,(S^{ij,kl})^{\mathsf T}\quad(\text{:325, :673}),\qquad
\tilde t^{ij} \leftarrow \tilde R^{ij} / \tilde D^{ij},$$

$$E_c = \tfrac14\sum_{i\le j} w_{ij} \sum_{\tilde a\tilde b} \tilde v^{ij}_{\tilde a\tilde b}\tilde t^{ij}_{\tilde a\tilde b},\qquad w_{ii}=1,\ w_{i<j}=2\quad(\text{:659, :728}).$$

Only the $i\le j$ block is stored; $t_{lk} = -t_{kl}$ with $a,b$ in place
(`so_oriented_amp`, :246). At $t_{\mathrm{cut,pno}} = 0$ every $S$ is
orthogonal and the iteration reproduces `linlccd(.., Hh)`.

## DLPNO-CCSD: pair screen and PNO substrate

**Pair lists** (`ferric_mp2`). Two criteria build the retained pairs
$i\le j$; diagonal pairs are always kept:

- distance (`pair_domains::build_pair_domains`, pair_domains.rs:152): keep
  $(i,j)$ if $|\mathbf r_i - \mathbf r_j| \le R_{\mathrm{pair}}$ (Boys centres, Bohr);
- estimated MP2 pair energy (`pair_energy_screen`): keep $(i,j)$ if
  $|e_{ij}| \ge T_{\mathrm{cut,pairs}}$, with
  $e_{ij} = (2-\delta_{ij})\sum_{ab} (2(ia|jb) - (ib|ja))(ia|jb)/D_{ij}^{ab}$.

`dlpno_ccsd::apply_pair_mask` (dlpno_ccsd.rs:49) zeroes $t_{ij}$ and
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
$\le R_{\mathrm{triple}}$ (`triple_is_retained`, :186). The predicate depends
only on the multiset $\{i,j,k\}$, so retained triples keep the dense weights
$m\in\{1,3,6\}$, and
$E_{(T)} = \tfrac13\sum_{\text{kept}} m\, f(i,j,k)$
(`screened_triple_energy`, :353–374). Only $m=6$ triples can be screened.

**TNOs** (`dlpno_ccsd_t_virtual.rs`). Each constituent pair
$(p,q)\subset\{i,j,k\}$ gets a PNO basis $P^{pq}$ at $t_{\mathrm{cut,tno}}$
(same `pno_for_pair` criterion, :315). The union is the range of the
projector sum

$$M^{ijk} = \sum_{(p,q)} P^{pq} P^{pq\,\mathsf T}\quad(\text{:439}),$$

taking eigenvectors with eigenvalue $> 10^{-8}$ (`UNION_RANK_TOL`, :157,
:447); in-span eigenvalues are $\ge 1$, out-of-span ones are 0. The pairs are
summed in sorted order, so the basis is bit-identical for all orderings of
$(i,j,k)$. The union is then semicanonicalized as for PNOs, and the per-triple
$\sum \tilde W V/D$ is evaluated with every virtual index (free and summed)
projected onto that basis (`dlpno_ccsd_t_kernel::triple_contribution_in_tno`,
:558). The kernel returns the unweighted $f(i,j,k)$; $m$ and the divisor 3 are
applied only in `screened_triple_energy`. At $t_{\mathrm{cut,tno}} = 0$ every
transform is square orthogonal and the dense $f$ is recovered.
