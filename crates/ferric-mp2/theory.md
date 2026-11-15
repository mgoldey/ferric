# MP2 Family Theory

This crate implements density-fitted second-order Møller–Plesset theory and
its variants: spin-component scaling, operator attenuation, orbital
optimization, Laplace quadrature, MP3, the analytic RI-MP2 gradient and
static polarizability, and the CABS construction for F12. Every equation
below is the one the code executes; references are `path:line` relative to
`crates/ferric-mp2/src/` unless another crate is named.

Identities marked **[check n]** are verified by claim *n* of
[`scripts/mp2_variant_identities.py`](../../scripts/mp2_variant_identities.py),
which exits nonzero on any mismatch and carries a negative control per claim.
Identities marked **[family X]** are checked by part X of
[`scripts/mp2_family_sympy.py`](../../scripts/mp2_family_sympy.py) and
**[gradient X]** by part X of
[`scripts/mp2_gradient_sympy.py`](../../scripts/mp2_gradient_sympy.py); both
compare against an independent route (brute-force spin-orbital MP2, adaptive
quadrature, or finite differences of a fully re-solved model) and exit nonzero
on a failure. Derivations that live in the project notes are cited as
`wiki/sympy_derivations.md` §*n*.

## Conventions

- Hartree atomic units. Real orbitals. $i,j,k$ index active occupied, $a,b,c$
  virtual, $P,Q$ auxiliary, $\mu,\nu$ AO functions; $\sigma$ labels spin.
- Denominators carry the code's sign,
  $D_{ij}^{ab} = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b < 0$,
  so every correlation energy is a sum of numerator$/D$ and is negative.
  $\Delta_{ij}^{ab} = -D_{ij}^{ab} > 0$.
- $(ia|jb)$ is a chemist-notation integral; $(ia|jb)^\times \equiv (ib|ja)$.
- **Frozen core.** The active occupied orbitals are
  `first_occ = frozen_core` through `nocc_total − 1`;
  `active_occ(nocc_total, frozen_core)` (`rimp2.rs:96`) returns their count
  and errors when the freeze covers every occupied orbital. Virtuals always
  start at `nocc_total`. Unrestricted paths freeze the same count in each spin
  (`u_rimp2.rs:149`).
- **RHF spin factors.** Closed-shell expressions are spin-summed over spatial
  orbitals; spin-orbital expressions (MP3) interleave
  $\alpha,\beta$ as spin orbital $2p+\sigma$ (`mp3.rs:250`).

## 1. Density fitting

The 4-index integrals are factorized through the auxiliary basis
(`rimp2.rs:997` `ri_mp2_spin_components`):

$$ (ia|\hat w|jb) \approx \sum_P B^P_{ia} B^P_{jb}, \qquad
B^P_{ia} = \sum_Q [L^{-1}]_{PQ}\,(Q|\hat w|ia), \qquad V_{PQ} = (P|\hat w|Q) = (LL^{\mathsf T})_{PQ}. $$

$L^{-1}$ is the inverse Cholesky factor (`rimp2.rs:1975`
`cholesky_inverse_sqrt`), so $L^{-1} V L^{-\mathsf T} = I$. The fitting
metric uses the same operator $\hat w$ as the 3-index integrals
(`rimp2.rs:1029`, `metric_op = None`). For $\hat w = \operatorname{erf}$ or
`terf` the metric is not positive definite in a Coulomb-fitted auxiliary
basis and goes through a regularized eigendecomposition instead
(`rimp2.rs:2112` `metric_inverse_sqrt`). A metric operator different from
$\hat w$ (`RiMp2Config::metric_op`) is a measurement option; its robust
(Dunlap) form is `ri_mp2_robust_attenuated_metric` (`rimp2.rs:784`).

## 2. Closed-shell RI-MP2

The energy is accumulated as opposite-spin and same-spin parts
(`rimp2.rs:1291` `pair_energy`):

$$ E_{\mathrm{OS}} = \sum_{ijab} \frac{(ia|jb)^2}{D_{ij}^{ab}}, \qquad
E_{\mathrm{SS}} = \sum_{ijab} \frac{(ia|jb)\left[(ia|jb) - (ia|jb)^\times\right]}{D_{ij}^{ab}}, \qquad
E^{(2)} = E_{\mathrm{OS}} + E_{\mathrm{SS}} = \sum_{ijab} \frac{(ia|jb)\left[2(ia|jb) - (ib|ja)\right]}{D_{ij}^{ab}} . $$

$E_{\mathrm{OS}}$ is the $\alpha\beta+\beta\alpha$ part and $E_{\mathrm{SS}}$
the $\alpha\alpha+\beta\beta$ part of the spin-orbital
$\tfrac14\sum |\langle ij\|ab\rangle|^2/D$ **[check 1]**. The loop visits
unique pairs $j \ge i$ and weights off-diagonal pairs by 2 (`rimp2.rs:1417`),
which equals the full $(i,j)$ sum because each summand is invariant under the
joint swap $(i,a)\leftrightarrow(j,b)$ **[check 1]**.

**κ-regularized MP2** (`RiMp2Config::kappa`, Lee & Head-Gordon, JCTC 14, 5203
(2018)) multiplies every term by $(1 - e^{-\kappa\Delta_{ij}^{ab}})^2$
(`rimp2.rs:1324`). $\kappa\to\infty$ recovers MP2 and $\kappa\to 0$ removes
correlation; $\kappa$ is in Ha$^{-1}$ and must be finite and positive.

**Spin adaptation.** With spin orbitals $p\sigma$,
$\langle IJ\|AB\rangle = (IA|JB)\,\delta_{\sigma_I\sigma_A}\delta_{\sigma_J\sigma_B} - (IB|JA)\,\delta_{\sigma_I\sigma_B}\delta_{\sigma_J\sigma_A}$.
An occupied $\alpha\beta$ pair keeps only the direct term when the virtuals
are $\alpha\beta$ and only the exchange term when they are $\beta\alpha$, so it
contributes $\tfrac14\sum_{ab}[(ia|jb)^2 + (ib|ja)^2]/D = \tfrac12\sum_{ab}(ia|jb)^2/D$;
with the $\beta\alpha$ pair this is $E_{\mathrm{OS}}$. An $\alpha\alpha$ or
$\beta\beta$ pair keeps both terms, and the two blocks give
$2\cdot\tfrac14\sum[(ia|jb)-(ib|ja)]^2/D = E_{\mathrm{SS}}$ after the
$a\leftrightarrow b$ relabelling of the cross term. Both identities hold for
symbolic integrals **[family A]**.

## 3. Unrestricted RI-MP2

`u_ri_mp2` (`u_rimp2.rs:159`) takes a UHF reference, or semicanonicalizes a
ROHF/ROKS one so each spin uses the eigenvectors of its own occupied and
virtual Fock blocks (`u_rimp2.rs:174`). The ROHF-MBPT singles term
$\sum_{ia}|f^\sigma_{ia}|^2/(\varepsilon_i-\varepsilon_a)$ is not included. With
$K_{iajb} = (ia|jb)_{\sigma\sigma} - (ib|ja)_{\sigma\sigma}$ and one shared
auxiliary metric,

$$ E_{\sigma\sigma} = \tfrac14 \sum_{j\ge i} f_{ij} \sum_{ab}\frac{K_{iajb}^2}{D_{ij}^{ab}} \quad (f_{ii}=1,\ f_{i\ne j}=2), \qquad
E_{\alpha\beta} = \sum_{iJaB}\frac{(ia|JB)^2}{\varepsilon_i+\varepsilon_J-\varepsilon_a-\varepsilon_B} $$

(`u_rimp2.rs:586` `same_spin_block_energy`, `u_rimp2.rs:624`
`opposite_spin_block_energy`; capitals index $\beta$). These are the
spin-orbital $\alpha\alpha$ and $\alpha\beta$ components, and with
$\alpha=\beta$ orbitals $E_{\alpha\alpha}+E_{\beta\beta} = E_{\mathrm{SS}}$ and
$E_{\alpha\beta} = E_{\mathrm{OS}}$ of section 2 **[check 2]**.

## 4. Spin-component scaling

**SCS-MP2** (`scs.rs:77` `scs_mp2`, Grimme 2003):
$E = c_{\mathrm{OS}}E_{\mathrm{OS}} + c_{\mathrm{SS}}E_{\mathrm{SS}}$ with the
Coulomb components of section 2 (`scs.rs:90`). Defaults
$c_{\mathrm{OS}} = 6/5$, $c_{\mathrm{SS}} = 1/3$ (`scs.rs:41`).

**SCS-MP2(2terfc)** (`scs.rs:165` `scs_mp2_2terfc`, Goldey & Head-Gordon,
JPCB 118, 6519 (2014)) evaluates the spin components twice, with the terfc
operator at two cutoffs $r_0^{(1)} < r_0^{(2)}$ (`scs.rs:202`–`:219`):

$$ E = c_{\mathrm{OS}}\,E_{\mathrm{OS}}\big[r_0^{(1)}\big] + c_{\mathrm{SS}}\Big(E_{\mathrm{SS}}\big[r_0^{(2)}\big] - E_{\mathrm{SS}}\big[r_0^{(1)}\big]\Big) \qquad (\texttt{scs.rs:221}). $$

The same-spin term is a difference, so $r_0^{(2)} \le r_0^{(1)}$ is an error
(`scs.rs:188`). Defaults (`scs.rs:147`): $r_0^{(1)} = 0.75$ Å,
$r_0^{(2)} = 1.05$ Å, $c_{\mathrm{OS}} = 1.27$, $c_{\mathrm{SS}} = 4.05$, the
non-VV10 aug-cc-pVTZ fit. The code's "thesis Eq 5.6" label is not checked
against the dissertation text (`scs.rs:105`); the formula above is the one
pinned by `scs_2terfc_matches_eq12_assembled_by_hand`.

The terfc kernel (`ferric-integrals/shim/shim.cc:1641`) is

$$ \frac{\operatorname{terfc}(r;r_0)}{r} = \frac{1}{r}\Big[1 - \tfrac12\operatorname{erf}\big(\omega(r-r_0)\big) - \tfrac12\operatorname{erf}\big(\omega(r+r_0)\big)\Big], \qquad \omega = \frac{1}{\sqrt2\,r_0} $$

with the curvature constraint applied by `Operator::terfc`
(`ferric-integrals/src/operator.rs:219`); `terf + terfc = 1/r` exactly.

**Operator limits of SCS-MP2(2terfc).** terfc$(r;r_0)\to 0$ for $r>0$ as
$r_0\to 0$ and $\to 1$ as $r_0\to\infty$ **[family D]**, and each spin
component is quadratic in the integrals, so
$E[r_0^{(1)}\to 0,\ r_0^{(2)}\to\infty] = c_{\mathrm{SS}}E_{\mathrm{SS}}[1/r]$
and $E[r_0^{(1)},r_0^{(2)}\to\infty] = c_{\mathrm{OS}}E_{\mathrm{OS}}[1/r]$:
the same-spin term is the window between the two cutoffs **[family C]**.

**The terfc kernel.** terf is odd in $r$, so
$\operatorname{terf}(r)/r = \tau_1 + \tau_3 r^2 + O(r^4)$ with

$$ \tau_3 = \frac{\omega^3}{3\sqrt\pi}\left(4\omega^2r_0^2 - 2\right)e^{-\omega^2r_0^2}, $$

whose only positive root is $\omega = 1/(\sqrt2\,r_0)$: the curvature
constraint of `Operator::terfc` is a flat long-range kernel at the origin. The
real-space forms in `ferric-dft` (`vv10.rs` `terfc_scalar`, linked, and
`terfc_scalar_decoupled`, free $\omega$) coincide at the linked $\omega$, and
terfc$(0) = 1$, so terfc$(r)/r$ keeps the full Coulomb singularity
**[family D]**. `run_terfc_rimp2` is section 2 with $\hat w =$ terfc$(r;r_0)/r$,
$r_0$ in Å converted to Bohr (default 1.05 Å,
`ferric-python/src/mp2.rs`); its metric is positive definite and takes the
Cholesky path (`metric_inverse_sqrt` sends only erf and terf to the
eigendecomposition).

## 5. Attenuated MP2

`attenuated_ri_mp2` (`attenuated.rs:247`) is section 2 with
$\hat w = \operatorname{erfc}(\omega r)/r$ in both the 3-index integrals and the
fitting metric, so the fit is variational in the attenuated metric
(Goldey & Head-Gordon, JPCL 3, 3592 (2012)). `scaling` multiplies the
correlation energy. `attenuated_ri_mp2_long_range` (`attenuated.rs:219`) uses
$\operatorname{erf}(\omega r)/r$ and the eigendecomposition metric path.

**Units.** $\omega$ is in Bohr$^{-1}$ in the Rust API. The CLI and Python take
Å$^{-1}$ and multiply by `BOHR_INV_PER_ANG_INV` = 0.529177 Bohr/Å
(`attenuated.rs:160`; e.g. `ferric-cli/src/lib.rs:3104`). The default
0.420 Å$^{-1}$ is 0.222254 Bohr$^{-1}$ (`attenuated.rs:165`).

**Default terfc cutoff by basis.** `terfc_r0.rs` holds the S66-trained
optimal $r_0$ per basis set (no counterpoise), read by the CLI (`[mp2] att_r0`)
and Python (`run_terfc_rimp2`, `run_rimp2(att_r0=)`) when $r_0$ is omitted; an
explicit value wins and an untabulated basis is an error. Source: Goldey and
Head-Gordon, Chem. Phys. Lett. 608, 249 (2014), and dissertation Tables 6.2
(Dunning) and 6.4 (Pople, Karlsruhe):

| Basis | $r_0$ (Å) | Basis | $r_0$ (Å) |
|---|---|---|---|
| cc-pVDZ | 1.55 | 6-31G\* | 1.75 |
| aug-cc-pVDZ | 1.05 | 6-31+G\* | 1.45 |
| cc-pVTZ | 1.50 | 6-31++G\*\* | 1.35 |
| aug-cc-pVTZ | 1.35 | 6-311++G\*\* | 1.35 |
| cc-pVQZ | 1.55 | def2-SVPD | 0.75 |
| aug-cc-pVQZ | 1.50 | def2-TZVPD | 1.30 |
| | | def2-TZVPPD | 1.35 |

The calendar sets of Table 6.2 (jun-DZ 1.50, jul-DZ 1.25, may-TZ 1.60, jun-TZ
1.45, jul-TZ 1.45, apr-QZ 1.65, may-QZ 1.60, jun-QZ 1.55, jul-QZ 1.55) are not
bundled and have no row. SCS-MP2(2terfc) has a one-row table: the four-parameter
fit of dissertation Eq. 5.6 exists only in aug-cc-pVTZ ($r_0^{(1)} = 0.75$ Å,
$r_0^{(2)} = 1.05$ Å, $c_{\mathrm{OS}} = 1.27$, $c_{\mathrm{SS}} = 4.05$). No new equations.

**Range split.** MP2 is quadratic in the integrals. Writing
$(ia|jb) = s + l$ for the erfc and erf parts,

$$ E[s+l] = E[s] + E[l] + \sum_{ijab}\frac{4sl - s\,l^\times - l\,s^\times}{D_{ij}^{ab}} , $$

so $E[\operatorname{erfc}] + E[\operatorname{erf}] \ne E[1/r]$ **[check 4]**.
`rs_mp2_decomposition` (`attenuated.rs:347`) returns the three energies, and
RS-MP2 + LR-RPA (`ferric-rpa::rs_mp2_rpa`) is built on this cross term.

## 6. MP3

`mp3_energy` (`mp3.rs:143`) works in spin orbitals with
$t_{ij}^{ab} = \langle ij\|ab\rangle/D_{ij}^{ab}$ (`mp3.rs:295`) built from RI
integrals:

$$ E^{(2)} = \tfrac14\sum t_{ij}^{ab}\langle ij\|ab\rangle, \qquad
E^{(3)} = \tfrac18\sum t_{ij}^{ab}\langle ab\|cd\rangle t_{ij}^{cd}
+ \tfrac18\sum \langle kl\|ij\rangle t_{ij}^{ab} t_{kl}^{ab}
+ \sum t_{ij}^{ab}\langle kb\|cj\rangle t_{ik}^{ac} $$

(`mp3.rs:310`, `:318`, `:326`, `:334`; the contraction is
`mp3_energy_from_spin_orbital_blocks`, `mp3.rs:279`). The ladder prefactors and the
ring-term convention are derived in `wiki/sympy_derivations.md` §2b.

### 6.1 Spin-blocked unrestricted MP3

`u_mp3` (`u_mp3.rs:492`) evaluates the same energies on an RHF, UHF or
semicanonicalized ROHF reference (`reference_orbitals`, `u_mp3.rs:247`;
doubles only, no non-Brillouin singles) and keeps both amplitude orders. The
spin-orbital sums of §6 are split by the spin of every index. Lower case is
α, upper case β; a tensor carries its spin block as a superscript, and the
αβ block is stored α first, $c^{\alpha\beta}_{iJaB}$. With
$D^{\sigma\tau}_{ij,ab} = \varepsilon_{i\sigma}+\varepsilon_{j\tau}-\varepsilon_{a\sigma}-\varepsilon_{b\tau}$:

$$ K^{\sigma\sigma}_{ijab} = (ia|jb)-(ib|ja), \qquad K^{\alpha\beta}_{iJaB} = (ia|JB), \qquad
c^{(1)} = K/D, \qquad c^{(2)} = R/D $$

(`antisymmetrize` `u_mp3.rs:411`, `divide_blocks` `u_mp3.rs:430`,
`amplitudes` `u_mp3.rs:462`). The spin-orbital amplitudes are recovered as
$t_{ijab}=c^{\sigma\sigma}_{ijab}$ for four indices of one spin and
$t_{i\bar J a\bar B} = -t_{\bar J i a \bar B} = -t_{i \bar J \bar B a} = t_{\bar J i \bar B a} = c^{\alpha\beta}_{iJaB}$,
the four orderings that give the αβ factor 1 below. Writing $t=c^{(1)}$, the
residual $R = D\,c^{(2)}$ of the spin-orbital
$R_{ijab} = \tfrac12\sum\langle ab\|cd\rangle t_{ijcd} + \tfrac12\sum\langle kl\|ij\rangle t_{klab} + P(ij)P(ab)\sum\langle kb\|cj\rangle t_{ikac}$
is, block by block (`mp3_residual`, `u_mp3_residual.rs:215`):

$$ R^{\alpha\alpha}_{ijab} = \sum_{cd}(ac|bd)\,t^{\alpha\alpha}_{ijcd} + \sum_{kl}(ki|lj)\,t^{\alpha\alpha}_{klab}
+ P(ij)P(ab)\,X_{iajb}, $$
$$ X_{iajb} = \sum_{kc}\big[(kc|jb)-(kj|bc)\big]\,t^{\alpha\alpha}_{ikac} + \sum_{KC}(KC|jb)\,t^{\alpha\beta}_{iKaC}, $$

$R^{\beta\beta}$ the same with α↔β (its mixed ring is $\sum_{kc}(kc|JB)\,t^{\alpha\beta}_{kIcA}$), and

$$ \begin{aligned} R^{\alpha\beta}_{iJaB} ={}& \sum_{cD}(ac|BD)\,t^{\alpha\beta}_{iJcD} + \sum_{kL}(ki|LJ)\,t^{\alpha\beta}_{kLaB}
+ \sum_{kc}(kc|JB)\,t^{\alpha\alpha}_{ikac} + \sum_{KC}\big[(KC|JB)-(KJ|BC)\big]\,t^{\alpha\beta}_{iKaC} \\
&+ \sum_{kc}\big[(ia|kc)-(ik|ac)\big]\,t^{\alpha\beta}_{kJcB} + \sum_{KC}(ia|KC)\,t^{\beta\beta}_{JKBC}
- \sum_{kC}(ki|BC)\,t^{\alpha\beta}_{kJaC} - \sum_{Kc}(KJ|ac)\,t^{\alpha\beta}_{iKcB}. \end{aligned} $$

The same-spin ladders lose the ½ of the spin-orbital form because
$\tfrac12\sum_{cd}\langle ab\|cd\rangle t_{ijcd} = \sum_{cd}(ac|bd)\,t_{ijcd}$ for
$t$ antisymmetric in $cd$; the αβ ladders pick up the two equal orderings
$(c\,\bar D)$, $(\bar D\, c)$. The last two αβ terms are the exchange halves of
$\langle k\bar B\|c\bar J\rangle$ when $k,c$ and $\bar B,\bar J$ have opposite
spins, which only the αβ block has. Implementations: same-spin and mixed
rings as GEMMs over compound $(ov)$ indices with
$M^\sigma_{(kc),(jb)} = (kc|jb)-(kj|bc)$ (`ring_terms`,
`u_mp3_residual.rs:103`; $P(ij)P(ab)$ in `scatter_same_spin`, `:82`), the two
αβ exchange rings (`mixed_exchange_rings`, `:144`), hole ladders
(`hole_ladder`, `:173`) and particle ladders one $(ac|\cdot\,\cdot)$ slab per
$a$ (`particle_ladder`, `:187`; slabs from `for_each_vvvv_slab`,
`u_mp3_eri.rs:252`). The energies (`channel_energy`, `u_mp3.rs:421`) are

$$ E^{(n)} = \tfrac14\sum c^{(n-1),\alpha\alpha}K^{\alpha\alpha} + \tfrac14\sum c^{(n-1),\beta\beta}K^{\beta\beta} + \sum c^{(n-1),\alpha\beta}K^{\alpha\beta}, $$

so the αα + ββ and αβ parts of $E^{(2)}$ are the same-spin and opposite-spin
energies of §3, and $E^{(3)}$ is split the same way. Summing $\tfrac14 K R/D$
over the blocks reproduces the three spin-orbital terms of §6 with their
coefficients ⅛, ⅛ and 1 term class by term class:
[`scripts/ump3_spin_blocks_sympy.py`](../../scripts/ump3_spin_blocks_sympy.py)
proves this as exact polynomial identities in symbolic integrals and checks
the energies against brute-force Rayleigh–Schrödinger theory
(runner `crates/ferric-python/tests/test_ump3_spin_blocks_sympy.py`). For an
RHF input the blocks satisfy
$c^{\alpha\alpha}_{ijab} = c^{\alpha\beta}_{ijab} - c^{\alpha\beta}_{ijba}$ and
$c^{\beta\beta} = c^{\alpha\alpha}$ at both orders, and the energies equal
`mp3_energy` (RI) and `mp3_energy_from_spin_orbital_blocks` (`mp3.rs:279`,
exact) (`tests/u_mp3.rs`).

Integrals are RI (the §1 build, per spin) or exact (`exact_mo_block`,
`u_mp3_eri.rs:96`, four quarter transforms of the dense AO tensor). The
largest VVVV piece is one $n_{v\sigma}^2 n_{v\tau}^2$ block (exact) or one
$n_{v\sigma} n_{v\tau}^2$ slab (RI), against $(2n_v)^4$ in §6. The memory
peak `u_mp3_peak_bytes` (`u_mp3.rs:373`) is pre-flighted and charged; it
matches the measured heap high-water to < 0.5 KiB
(`tests/u_mp3_peak_matches_the_heap.rs`).

### 6.2 Orbital-partitioned Padé MP3 (OP-MP3)

`ferric_mp2::opmp3` resums the MP3 series orbital by orbital (Bystrom,
Berkelbach, Hait, arXiv:2609.25330). The pair energies of §6.1 are
collected into a symmetric matrix over occupied spin orbitals,

$$ W^{(n)}_{ij} = \tfrac18\sum_{kab}\big(t^{(n-1)}_{ikab}\langle jk\|ab\rangle + t^{(n-1)}_{jkab}\langle ik\|ab\rangle\big), \qquad \operatorname{Tr}W^{(n)} = E^{(n)} $$

(Eq. 11a; $t^{(1)} = c^{(1)}$, $t^{(2)} = c^{(2)}$). Spin conserves the
pair, so $W$ is block diagonal in the spin of $i,j$. Splitting each block by
the spin of the summed $k$ gives the same-spin and opposite-spin channels
(`w_channels`, `opmp3.rs:122`):

$$ W^{SS,\sigma}_{ij} = \tfrac18\sum_{kab}c^{\sigma\sigma}_{ikab}K^{\sigma\sigma}_{jkab} + (i\leftrightarrow j), $$
$$ W^{OS,\alpha}_{ij} = \tfrac14\sum_{KaB}c^{\alpha\beta}_{iKaB}K^{\alpha\beta}_{jKaB} + (i\leftrightarrow j), \qquad
W^{OS,\beta}_{IJ} = \tfrac14\sum_{kaB}c^{\alpha\beta}_{kIaB}K^{\alpha\beta}_{kJaB} + (I\leftrightarrow J). $$

The opposite-spin prefactor is ¼, not ⅛: with $k$ of the other spin, the
orderings $(a\bar B)$ and $(\bar B a)$ of the virtual pair both map onto the
stored αβ block with the same sign, which doubles the sum (Eq. 18). Then
$\operatorname{Tr}W^{SS} = E_{\alpha\alpha}+E_{\beta\beta}$ and
$\operatorname{Tr}W^{OS} = E_{\alpha\beta}$ at both orders; the traces of
$W^{OS,\alpha}$ and $W^{OS,\beta}$ are each half of $E_{\alpha\beta}$.
The β OS block is built from the αβ tensors with their first two axes
swapped (`beta_first`, `opmp3.rs:110`). A spin with fewer than two occupied
orbitals has no same-spin pair, and its $W^{SS}$ is set to exact zeros.

The matrix Padé of symmetric $A=W^{(2)}$ and $B=xW^{(3)}$ (Eq. 13;
`pade`, `opmp3.rs:169`) is

$$ P(A,B) = A\,(A-B)^{-1}A, \qquad \operatorname{Tr}P(A,tB) = \operatorname{Tr}A + t\operatorname{Tr}B + t^2\operatorname{Tr}\big[B A^{-1} B\big] + O(t^3). $$

Its first two Taylor coefficients in $t$ are $E^{(2)}$ and $E^{(3)}$, so the
series limit is MP3. For commuting $A$ and $B$, each eigenvalue gets the
scalar Padé $a^2/(a-b)$. The code diagonalizes $A-B = V\Lambda V^{\mathsf T}$ and
forms $P = (AV)\Lambda^{-1}(AV)^{\mathsf T}$. An eigenvalue below
$10^{-12}\max|\lambda|$ is a singular denominator and an error. Rows where $A$
is identically zero (a channel with no pairs) are left out, so $P(0,0)=0$.
$P$ of a block-diagonal pair is the block-diagonal $P$ of the blocks, so
$\operatorname{Tr}P$ of two non-interacting fragments is the sum of the
fragments' traces, and every model below is size-consistent. The scalar Padé
$E_2^2/(E_2-E_3)$ (P-MP3, `scalar_pade`, `opmp3.rs:241`) is not, since
$(a_1+a_2)^2/(a_1+a_2-b_1-b_2) \ne a_1^2/(a_1-b_1)+a_2^2/(a_2-b_2)$ in general.
It is kept as the negative control.

The models (`Opmp3Models::evaluate`, `opmp3.rs:271`) are

$$ E_c^{\text{OP-MP3}(x)} = \sum_\sigma \operatorname{Tr}P\big(W^{(2)}_\sigma, xW^{(3)}_\sigma\big), $$
$$ E_c^{\text{SOP-MP3}(x_{ss},x_{os})} = \sum_\sigma \Big[\operatorname{Tr}P\big(W^{(2),SS}_\sigma, x_{ss}W^{(3),SS}_\sigma\big) + \operatorname{Tr}P\big(W^{(2),OS}_\sigma, x_{os}W^{(3),OS}_\sigma\big)\Big], $$
$$ E_c^{\text{SSD-OPMP3}(x_{ss})} = E_c^{\text{SOP-MP3}(x_{ss},1)}, $$

with $W_\sigma = W^{SS}_\sigma + W^{OS}_\sigma$ (Eqs. 14, 19–22). OP-MP3 and
SOP-MP3 are reported at $x = 1$. SSD-OPMP3 uses the same-spin damping the
paper prints for each reference (`PAPER_X_SS`, `opmp3.rs:34`): HF 0, PBE0
0.5, PBE 0.7, SPW92 0.75, r²SCAN 0.6, ωB97X-V and ωB97M-V 0.5, κOOMP2 0.45.

**Reference.** For a DFT reference, the perturbation theory is built on the
HF Fock operator of the KS density (`opmp3_reference`,
`opmp3_reference.rs:187`). From the converged spin densities $D_\sigma$,

$$ F_\sigma = h + J[D_\alpha + D_\beta] - K[D_\sigma], \qquad
E_{\text{HF}}[\rho] = E_{nn} + \tfrac12\sum_\sigma \operatorname{Tr}\big[D_\sigma(h + F_\sigma)\big], $$

with $h$ the SCF's own one-electron operator (including ECP and X2C when
used). The occupied and virtual blocks of each $F_\sigma$ are diagonalized
separately in the SCF orbitals (semicanonicalization; `semicanonical_spin`,
`opmp3_reference.rs:111`). This leaves the density, and so $E_{\text{HF}}[\rho]$,
unchanged, and gives the orbital energies $\varepsilon_{p\sigma}$ for the
denominators of §6.1. The remaining $F^\sigma_{ia}$, zero only for a UHF
reference, enter through the non-Brillouin singles energy

$$ E_{\text{NBS}} = \sum_\sigma\sum_{i\,\text{active},\,a}\frac{(F^\sigma_{ia})^2}{\varepsilon_{i\sigma}-\varepsilon_{a\sigma}} $$

(`SemicanonicalSpin::singles_energy`, `opmp3_reference.rs:53`; frozen-core
orbitals are excluded). The total is
$E = E_{\text{HF}}[\rho] + E_{\text{NBS}} + E_c$ (`Opmp3Result::total`,
`opmp3_driver.rs:161`). The paper uses spin-restricted references: RHF or RKS
for singlets, ROHF or ROKS otherwise (`opmp3_scf`, `opmp3_driver.rs:168`). A
UHF result is also accepted, with $E_{\text{NBS}}=0$.

[`scripts/opmp3_identities_sympy.py`](../../scripts/opmp3_identities_sympy.py)
proves four identities: (a) the Eq. 18 SS/OS split of the spin-orbital
Eq. 11a, with the ¼ OS prefactor; (b) the Padé Taylor coefficients and its
commuting-block scalar limit; (c) block additivity of $\operatorname{Tr}P$, and
the scalar Padé's failure of it; (d) $\operatorname{Tr}W^{(n)} = E^{(n)}$, checked
against brute-force Rayleigh–Schrödinger theory. Each identity has a recorded
mutation that turns it red (runner
`crates/ferric-python/tests/test_opmp3_identities_sympy.py`).

## 7. Orbital-optimized RI-MP2

**Functional.** `oo_ri_mp2` minimizes $E(C) = E_{\mathrm{HF}}(C) + E^{(2)}_{\mathrm{RI}}(C)$
with $E^{(2)}$ evaluated in the semicanonical frame of $C$: the active
occupied–occupied and virtual–virtual blocks of $C^{\mathsf T}F(C)C$ are
diagonalized first (`oo_rimp2.rs:425` `semicanonical_rotation`). That is the
full-Fock Hylleraas functional of OMP2 (Bozkaya et al., JCP 135, 104103
(2011)); it is invariant to occupied–occupied and virtual–virtual rotations,
so occupied–virtual stationarity is full stationarity.

**Rotation.** $C \leftarrow C\,U(\kappa)$ with the Cayley transform
(`orbital_rotation.rs:53`)

$$ U(\kappa) = \left(I - \tfrac12\kappa\right)^{-1}\left(I + \tfrac12\kappa\right), \qquad \kappa^{\mathsf T} = -\kappa,\quad \kappa_{ai} = -\kappa_{ia} , $$

which is exactly orthogonal and equals $e^{\kappa}$ to second order
**[check 6]**. The gradient is $g_{ai} = +\partial E/\partial\kappa_{ai}$.

**Gradient** (closed shell, `oo_rimp2.rs:1045`):

$$ g_{ck} = 4F_{ck} + 2\sum_{ijab}\tilde T_{ij}^{ab}\,\frac{\partial (ia|jb)}{\partial\kappa_{ck}} + \sum_{pq} W_{pq}\frac{\partial F_{pq}}{\partial\kappa_{ck}}, \qquad \tilde T = 2t - t^{\mathrm{swap}}, $$

$$ \frac{\partial (ia|jb)}{\partial\kappa_{ck}} = \delta_{ik}(ca|jb) + \delta_{jk}(ia|cb) - \delta_{ac}(ik|jb) - \delta_{bc}(ia|jk), $$

at fixed denominators, with $W$ the unrelaxed MP2 density of section 9 and the
last term the Fock (orbital-energy) response. The HF term is $+4F_{ai}$
closed-shell (`oo_rimp2.rs:1236`) and $+2F^\sigma_{ai}$ per spin unrestricted
(`u_oo_rimp2.rs:333`). The gradient derivation is in
`wiki/sympy_derivations.md` §3.

**Step.** A level-shifted diagonal Newton step in the solver's frame
(`oo_rimp2.rs:1561`):

$$ \kappa_{ai} = -\frac{g_{ai}}{\varepsilon_a - \varepsilon_i + \mu}, \qquad \mu = \texttt{level\_shift} = 0.1\ \mathrm{Ha}, $$

with $\varepsilon = \operatorname{diag}(C^{\mathsf T}FC)$, uniformly rescaled
so $\max|\kappa_{ai}| \le 0.3$ rad (`oo_rimp2.rs:1450`). DIIS extrapolates the
MO coefficients with error vector $C\,G\,C^{\mathsf T}$, $G$ the
antisymmetric matrix of $g$ (`oo_rimp2.rs:1611`), followed by Löwdin
reorthonormalization. A trial step that raises the energy by more than
$10^{-4}$ Ha (`oo_rimp2.rs:1624`) is replaced by up to ten halvings of
$\kappa$ without DIIS, the first with $E \le E_{\mathrm{old}} + 10^{-12}$ Ha
accepted (`oo_rimp2.rs:1649`), and the DIIS subspace is reset.

**Unrestricted** (`u_oo_rimp2.rs`) applies the same step per spin with
independent $\kappa^\alpha, \kappa^\beta$ and the per-spin cap
`max_kappa` = 0.3. When ten halvings fail, $\mu$ doubles up to 5 Ha
(`u_oo_rimp2.rs:931`, `:758`).

The OO-RI-MP2 nuclear gradient (`oo_rimp2_gradient.rs`) needs no Z-vector,
because the converged orbitals are stationary.

**J/K route** (closed shell, `OoJk` in `oo_rimp2.rs`). $F = h + G[D]$ and
the Fock response $\sum_{pq}W_{pq}\,\partial F_{pq}/\partial\kappa$ use one
symmetric linear $G[X] = J[X] - \tfrac12K[X]$: four-centre, or the reference
SCF's RI-J + RI-K when it fitted both on one auxiliary basis. The gradient
formula above only uses that $G$ is linear and symmetric
($\sum X\,G[Y] = \sum Y\,G[X]$), which the Coulomb-metric fit satisfies, so it
is unchanged on the fitted route; the optimised functional is then the fitted
one, and at $\kappa = 0$ its HF part is the fitted SCF energy. The nuclear
gradient follows the same route through `ResponseJk` (fitted bilinear
two-electron derivative, §9.1). Unrestricted OO-RI-MP2 (`UOoJk` in
`u_oo_rimp2.rs`) takes the same route per spin: $F^\sigma = h + J[D^\alpha +
D^\beta] - K[D^\sigma]$ and its Fock response, with $J$ and $K$ of each spin
density (or spin response density) from the reference's fit and
$J[D^\alpha + D^\beta] = J[D^\alpha] + J[D^\beta]$ by linearity.

## 8. Laplace-transform MP2

With $x = \Delta_{ij}^{ab} > 0$,
$1/x = \int_0^\infty e^{-tx}\,dt \approx \sum_k w_k e^{-t_k x}$ on a minimax
grid (`ferric_quadrature::LaplaceQuadrature`) fitted on
$[2(\varepsilon_{\mathrm{LUMO}}-\varepsilon_{\mathrm{HOMO}}),\ 2(\varepsilon_{\max}-\varepsilon_{\min})]$
(`laplace.rs:762`). `n_quad` must be 3, 5 or 7. Because $x$ is a sum of
one-orbital energies, $e^{-tx}$ splits over the $(i,a)$ and $(j,b)$ pairs:

$$ B^P_{ia}(t) = B^P_{ia}\,e^{-t(\varepsilon_a-\varepsilon_i)/2}\quad(\texttt{laplace.rs:666}), \qquad
J_{PQ}(t) = \sum_{ia} B^P_{ia}(t)B^Q_{ia}(t), \qquad
G_{PQ}[i,j](t) = \sum_a B^P_{ia}(t)B^Q_{ja}(t), $$

$$ \sum_{PQ}J_{PQ}(t)^2 = \sum_{ijab}(ia|jb)^2e^{-tx}, \qquad
\sum_{PQij}G_{PQ}[i,j]\,G_{PQ}[j,i] = \sum_{ijab}(ia|jb)(ib|ja)\,e^{-tx} $$

**[check 3]**. The RI-Laplace MP2 energy (`laplace.rs:818`, `:1073`) is

$$ E^{(2)} \approx -\sum_k w_k\Big[2\sum_{PQ}J_{PQ}(t_k)^2 - \sum_{PQij}G_{PQ}[i,j](t_k)\,G_{PQ}[j,i](t_k)\Big] , $$

with $E_{\mathrm{OS}}$ the $J^2$ term and $E_{\mathrm{SS}}$ the $J^2 - G\cdot G$
term. `compute_mo` (`laplace.rs:745`) evaluates both in the MO basis.
`compute_ao` (`laplace.rs:836`, the `laplace-mp2` path) evaluates the Coulomb
term from AO pseudo-densities

$$ P(t)_{\mu\nu} = \sum_i C_{\mu i}e^{t\varepsilon_i}C_{\nu i}, \qquad
Q(t)_{\mu\nu} = \sum_a C_{\mu a}e^{-t\varepsilon_a}C_{\nu a}, \qquad
J_{PQ}(t) = \operatorname{Tr}\!\left[B^P P(t)\,B^Q Q(t)\right] $$

(`laplace.rs:1500`, `:1522`, `:568`; $B^P_{\mu\nu}$ the dressed AO factor)
**[check 3]**, and the exchange term in the MO basis
(`laplace.rs:479` `laplace_exchange_energy`). The exchange term has no AO
trace of the same form: $\sum_{PQ}\operatorname{Tr}(B^PP\,B^QQ\,B^QP\,B^PQ)$
contracts the wrong index pairs **[check 3, control]**.

**Laplace SOS-MP2** (`laplace.rs:1461` `laplace_sos_mp2`) keeps only the
Coulomb term, $E_{\mathrm{SOS}} = c_{\mathrm{OS}}E_{\mathrm{OS}}$ with
$E_{\mathrm{OS}} \approx -\sum_k w_k\sum_{PQ}J_{PQ}(t_k)^2$ and default
$c_{\mathrm{OS}} = 1.3$. Dropping $E_{\mathrm{SS}}$ is what leaves a product
of independent $(ia)$ and $(jb)$ contractions; the derivation is
`wiki/sympy_derivations.md` §11. Formulations (`laplace.rs:1381`
`SosFormulation`):

| Formulation | Evaluates | Exact |
|---|---|---|
| `Mo` (default) | $J = B(t)B(t)^{\mathsf T}$ (`laplace.rs:1157`) | yes |
| `Ao` | $\operatorname{Tr}[B^P P(t) B^Q Q(t)]$ (`laplace.rs:1249`) | yes |
| `AoSparse(r)` | `Ao` with Boys-localized occupied orbitals and $P(t)$ restricted to AO domains of radius $r$ Bohr | no |

`AoSparse` requires $r$; passing $r$ with `Mo`/`Ao` is an error
(`laplace.rs:1410`). The algebra is dense in all three.

**Sign and quadrature, checked.** $\int_0^\infty e^{-tx}dt = 1/x$ for $x>0$,
so a summand $N/D$ with $D = -x$ is $-N\int_0^\infty e^{-tx}dt$; the factor
$e^{-t(\varepsilon_a-\varepsilon_i)/2}$ on $B^P_{ia}$ squares to
$e^{-t(\varepsilon_a-\varepsilon_i)}$ in $J_{PQ}$, and $P(t)Q(t)$ carries the
same unhalved weight **[family F]**. Integrated by adaptive quadrature instead
of the minimax grid, the MO and AO forms reproduce brute-force spin-orbital
$E_{\mathrm{OS}}+E_{\mathrm{SS}}$ and $E_{\mathrm{OS}}$ to $2\times10^{-15}$
relative **[family F]**; the minimax grid's own error is a property of
`ferric_quadrature`, not of these identities.

## 9. Analytic RI-MP2 gradient and Z-vector

**Unrelaxed density** (`oo_rimp2.rs:990` `build_mp2_density`), with
$t_{ij}^{ab} = (ia|jb)/D_{ij}^{ab}$:

$$ P_{ij} = -\sum_{kab} t_{ik}^{ab}\left(2t_{jk}^{ab} - t_{jk}^{ba}\right), \qquad
P_{ab} = \sum_{ijc} t_{ij}^{ac}\left(2t_{ij}^{bc} - t_{ij}^{cb}\right). $$

**Lagrangian matrix** (`zvector.rs:440`), with
$x^P_{ia} = \sum_{jb}(2t_{ij}^{ab} - t_{ij}^{ba})B^P_{jb}$ (`zvector.rs:381`)
and the full-MO factor $B^P_{pq}$:

$$ I_{qi} = -2\sum_{Pa} x^P_{ia}B^P_{qa}, \qquad I_{qa} = -2\sum_{Pi} x^P_{ia}B^P_{qi}. $$

Frozen-core columns are zero; frozen-core rows are not.

**Frozen core.** With core orbitals $k$ frozen, the correlation density gains
the core–active block (`zvector.rs:226` `frozen_core_block`, PySCF `dco`)

$$ D^{\mathrm{corr}}_{kI} = D^{\mathrm{corr}}_{Ik} = \frac{I_{kI}}{\varepsilon_k - \varepsilon_I}, $$

which enters $D^{\mathrm{corr}}$ below, and $X$, $z$, the $\varepsilon$-weighted
density and the $I$ overlap term span every occupied orbital, core included.

**Z-vector** (`zvector.rs:65` `solve_zvector`). The right-hand side follows
PySCF `grad/mp2.py`:

$$ X_{ai} = \left[C^{\mathsf T}(2J - K)[D^{\mathrm{corr}}]\,C\right]_{ai} + I_{ia} - I_{ai}, \qquad D^{\mathrm{corr}} = C\,(P + P^{\mathsf T} + D_{kI})\,C^{\mathsf T}, $$

and the CPHF equation is

$$ (\varepsilon_a - \varepsilon_i)\,z_{ai} + \sum_{bj}\left[4(ai|bj) - (ab|ij) - (aj|bi)\right] z_{bj} = -X_{ai}, $$

solved by Jacobi-preconditioned DIIS to $\max|r| < 10^{-8}$; a solve that
reaches the 50-iteration cap first is an error on the gradient path
(`ZvectorSolution::require_converged`). The coupling is
$\tfrac12$ of `compute_az_product` (`zvector.rs:174`), which returns the vo
block of $(4J - K - K^{\mathsf T})[D^z]$ with
$D^z = \sum_{ai}z_{ai}(C_aC_i^{\mathsf T} + C_iC_a^{\mathsf T})$, twice the
bracket above **[check 5]**.

**Relaxed density** (`zvector.rs:566`): $2\delta_{ij} + P_{ij}$ on the
occupied block, $P_{ab}$ on the virtual block, $z_{ai}$ on both off-diagonal
blocks, for whatever $P$ the caller passes. The density whose dipole is
$-\partial E/\partial F$ takes $P + P^{\mathsf T}$.

**Gradient** (`gradient.rs:138` `rimp2_gradient_analytical`): overlap terms
from $I$, the orbital-energy-weighted relaxed density and the
$J - \tfrac12K$ potential of the correlation density; the core-Hamiltonian
term with the full relaxed density; the bilinear two-electron term
$\Gamma(D^{\mathrm{HF}}, D^{\mathrm{HF}} + 2D^{\mathrm{corr}})$; and the RI
3-centre/2-centre response with the closed-shell 2-PDM weight
$\Gamma_2 = 2(2t - t^{\mathsf T})$ (`gradient.rs:93`–`:127` lists each block
with its PySCF line).

**J/K of the correlation density, built once.** The Lagrangian's potential
term needs $(J,K)[2D^{\mathrm{corr}}_{\mathrm{rel}}]$ of the *relaxed*
correlation density $D^{\mathrm{corr}}_{\mathrm{rel}} = D^{\mathrm{corr}} + D^z$,
while the right-hand side needs $(J,K)[D^{\mathrm{corr}}]$ of the unrelaxed one
and every CPHF matvec $(J,K)[D^z]$. By linearity

$$ (J,K)[2D^{\mathrm{corr}}_{\mathrm{rel}}] = 2\,(J,K)[D^{\mathrm{corr}}] + 2\,(J,K)[D^z], $$

with the second pair taken from the matvec of the converged $z$ (the iterate
the solver returns). No further J/K build follows the Z-vector. The sum
differs from a direct build of $2D^{\mathrm{corr}}_{\mathrm{rel}}$ only by
which quartets the absolute screen ($\text{bound}\cdot\max|D| < 10^{-12}$)
drops in each of the three builds.

**Working assembly, proven end to end.** For a parameter $x$ on which $S$, $h$,
the four-centre ERIs and the RI factors $(P|\mu\nu)$, $V$ all depend,
`mp2_relaxed_lagrangian_gradient` plus `integral_response_gradient_3c2c`
evaluate

$$ \frac{dE}{dx} = \operatorname{Tr}[D^{\mathrm{tot}}h'] + \sum_{\mu\nu}\big(\mathrm{im1} - \zeta - v^{\mathrm{s1occ}}\big)_{\mu\nu}S'_{\mu\nu}
+ \partial_x B\big(D^{\mathrm{HF}}, D^{\mathrm{HF}} + 2D^{\mathrm{corr}}_{\mathrm{rel}}\big)
+ 4\sum_{Pia} y^P_{ia}\,(P|ia)' - 2\sum_{PQ}V'_{PQ}\sum_{ia}(V^{-1}u)^P_{ia}\,y^Q_{ia}, $$

with $D^{\mathrm{tot}} = D^{\mathrm{HF}} + D^{\mathrm{corr}}_{\mathrm{rel}}$,
$\mathrm{im1} = C\,\tilde I\,C^{\mathsf T}$ ($\tilde I$ = $I$ with its
vir–occ block replaced by the transpose of its occ–vir block),
$\zeta = C\,(\zeta_{pq}D^{\mathrm{corr}}_{pq})\,C^{\mathsf T} + \sum_i 2\varepsilon_iC_iC_i^{\mathsf T}$
with $\zeta_{pq} = \tfrac12(\varepsilon_p+\varepsilon_q)$ except
$\zeta_{ai} = \zeta_{ia} = \varepsilon_i$,
$v^{\mathrm{s1occ}} = P_{\mathrm{occ}}(2J-K)[D^{\mathrm{corr}}_{\mathrm{rel}}]P_{\mathrm{occ}}$
($P_{\mathrm{occ}} = C_oC_o^{\mathsf T}$),
$B(D_1,D_2) = \tfrac12\operatorname{Tr}[D_1(J-\tfrac12K)[D_2]]$
differentiated through the ERIs only, $u^P_{ia} = (P|ia)$,
$y = V^{-1/2}x$ and $(P|ia)'$ the AO derivative at fixed $C$. On a model with
all five inputs moving, the right-hand side equals Richardson differences of
the re-solved $E_{\mathrm{HF}} + E^{(2)}$ to $10^{-11}$ relative for MP2 and
SCS-MP2 $(6/5, 1/3)$, with and without a frozen core, and
$\operatorname{Tr}[(D^{\mathrm{HF}}+D^{\mathrm{corr}}_{\mathrm{rel}})R] = dE/d\lambda$
for $h\to h+\lambda R$ to $10^{-12}$ **[gradient G]**. The model's MO-level
factors 4 and $-2$ in the RI terms are the energy derivative; the AO-level
bookkeeping of `integral_response_gradient_3c2c` (its $2y$ and symmetrized
$\Gamma$) is not transcribed.

### 9.1 Fitted (RI-JK) reference

With `[scf] jk = "rijk"` the reference SCF minimises the fitted energy

$$ E_{\mathrm{HF}} = \operatorname{Tr}[hD] + Q(D) + V_{nn}, \qquad
Q(D) = \tfrac12\operatorname{Tr}[D\,G(D)], \qquad
G(D) = J^{\mathrm{fit}}[D] - \tfrac12 K^{\mathrm{fit}}[D], $$

$$ J^{\mathrm{fit}}[D]_{\mu\nu} = \sum_{PQ}(\mu\nu|P)\,(V^{-1})_{PQ}\,(Q|\lambda\sigma)D_{\lambda\sigma}, \qquad
K^{\mathrm{fit}}[D]_{\mu\nu} = \sum_{PQ}(\mu\lambda|P)\,M_{PQ}\,(Q|\nu\sigma)D_{\lambda\sigma}, $$

on one JK-fitting basis, $V_{PQ} = (P|Q)$, and $M$ the eigenvalue-cut
$V^{-1/2}V^{-1/2}$ of `ferric_scf::df_k` (`DfJ` / `DfK`). $G$ is linear and
self-adjoint ($\operatorname{Tr}[D_1G(D_2)] = \operatorname{Tr}[D_2G(D_1)]$, both
fits use the same factor on each side), so $\partial E_{\mathrm{HF}}/\partial D
= h + G(D) = F$, the fitted Fock matrix whose eigenpairs are the reference
orbitals.

The MP2 Lagrangian above meets the reference only through $F$: the orbital
energies, the stationarity condition $F_{ai} = 0$ the multipliers $z_{ai}$
enforce, and orthonormality. Its derivation is therefore unchanged with
$G$ in place of the four-centre $J - \tfrac12K$, and the fitted gradient
replaces each four-centre object by the fitted one:

| term | four-centre | fitted |
|---|---|---|
| right-hand side | $(2J-K)[D^{\mathrm{corr}}]$ | $2G(D^{\mathrm{corr}})$ |
| CPHF coupling | $(4J-K-K^{\mathsf T})[D^z]$ | $4G(D^z)$ |
| potential term | $(J - \tfrac12K)[2D^{\mathrm{corr}}_{\mathrm{rel}}]$ | $G(2D^{\mathrm{corr}}_{\mathrm{rel}})$ |
| explicit 2e derivative | $\sum\Gamma(D_1,D_2)\,\partial(\mu\nu|\lambda\sigma)$ | $\partial_R B(D_1,D_2)$ at fixed $D_1,D_2$ |

with $D_1 = D^{\mathrm{HF}}$, $D_2 = D^{\mathrm{HF}} + 2D^{\mathrm{corr}}_{\mathrm{rel}}$
and $B(D_1,D_2) = \tfrac12\operatorname{Tr}[D_1G(D_2)]$ the polarization of $Q$
($B(D,D) = Q(D)$). The MP2 integral response (its own RI basis, $\Gamma_2$
above) and the one-electron and overlap terms are unchanged.

**The fitted separable term.** $\partial_R B$ at fixed densities is
assembled from derivatives of the quadratic form, which
`ferric_scf::df_gradient` already provides:

$$ B(D_1,D_2) = Q(u) - Q(v), \qquad u = \tfrac12(D_1 + D_2) = D^{\mathrm{HF}} + D^{\mathrm{corr}}_{\mathrm{rel}},
\qquad v = \tfrac12(D_1 - D_2) = -D^{\mathrm{corr}}_{\mathrm{rel}}, $$

an identity of any symmetric bilinear form. The Coulomb part of
$\partial_R Q(D)$ is the RI-J gradient
$c^{\mathsf T}\partial(P|\mu\nu)D - \tfrac12 c^{\mathsf T}\partial V c$, $c = V^{-1}d$,
$d_P = (P|\mu\nu)D_{\mu\nu}$; the exchange part is the RI-K gradient of
the channel $(D, -\tfrac14)$ (Weigend, Kattannek & Ahlrichs 2009, with the
Daleckii–Krein derivative of $M$; `df_gradient.rs` module doc). The RI-K
gradient factors each channel density as $XX^{\mathsf T}$, so an indefinite
$D$ (always $v$; $u$ when a natural occupation of the relaxed density is
negative) is split by eigenvalue sign, $D = P - N$ with $P, N \succeq 0$, and

$$ Q_K(P - N) = 2Q_K(P) + 2Q_K(N) - Q_K(P + N), $$

again exact for any quadratic form. Both $u$ and $v$ are full rank, so this
term costs a few full-rank RI-K gradient channels rather than the
occupied-rank one of the SCF gradient.

**The CLI default.** With `[scf] jk` omitted, a closed-shell run of the
MP2-family kinds takes this fitted reference on the JK-fit set matched to
the orbital basis (`ferric_scf::scf_jk::matched_jk_aux`; Weigend's
cc-pVTZ-JKFIT for cc-pVDZ, since no cc-pVDZ-JKFIT exists). The fit moves
$E_{\mathrm{HF}}$ by $\delta E_{\mathrm{HF}} = O(\|\delta G\|)$, and, through the
fitted orbitals and orbital energies, $E^{(2)}$ by a shift of the same
order; neither is variational in the fit, so their sum is the quantity to
compare with the RI-MP2 fit error of $E^{(2)}$ itself. Measured against the
four-centre reference on n-alkanes (closed shell, `cc-pvXz-ri` for MP2;
frozen core at cc-pVTZ):

| System | Basis (functions) | JK aux | ΔE(HF) | ΔE(corr) | ΔE(total) | max \|Δg\| | RI-MP2 fit error |
|---|---|---|---|---|---|---|---|
| propane | cc-pVTZ (202), fc | def2-universal-jkfit | +29.8 | +70.0 | +99.8 | 37.5 | +90.8 |
| | | cc-pvtz-jkfit | +15.4 | +48.8 | +64.2 | 10.1 | |
| hexane | cc-pVDZ (154) | def2-universal-jkfit | +19.1 | +93.3 | +112 | 11.3 | |
| | | cc-pvtz-jkfit | +26.9 | +29.4 | +56.3 | 7.4 | |
| octane | cc-pVDZ (202) | def2-universal-jkfit | +31.7 | +123 | +155 | 11.1 | +280 (23.2) |
| | | cc-pvtz-jkfit | +39.1 | +39.1 | +78.2 | 7.6 | |
| dodecane | cc-pVDZ (298) | def2-universal-jkfit | +56.4 | +184 | +240 | 11.1 | |
| | | cc-pvtz-jkfit | +63.6 | +58.5 | +122 | 7.6 | |
| octane | cc-pVTZ (492), fc | def2-universal-jkfit | +104 | +171 | +275 | — | |
| | | cc-pvtz-jkfit | +62.6 | +127 | +189 | — | |

ΔE in µHa, max $|\Delta g|$ in µHa/Bohr (f64 RI-J); the RI-MP2 fit error is PySCF DF-MP2 minus
conventional MP2 at the same orbitals (gradient in parentheses).

The matched set halves the total shift relative to def2-universal-jkfit,
and its shift is below the RI-MP2 fit error where that was measured (0.28 of
it at octane/cc-pVDZ, 0.71 at propane/cc-pVTZ, where def2-universal-jkfit
exceeds it); the two have the same sign and add. An open shell keeps the
four-centre reference when `jk` is omitted (no fitted default was measured
for an open shell), as does an input that sets a key only the four-centre
route reads. A closed-shell `rimp2`, `scs-mp2` or `oo-rimp2` optimization
takes the fitted default too. Its gradient's peak RSS is a multiple of the
four-centre route's (2.1 times at dodecane/cc-pVDZ, 8.3 against 4.0 GB), so
it declares that peak before the reference SCF and refuses a budget below
it (`gradient_memory.rs`; the model is in `crates/ferric-scf/theory.md`,
"Peak memory of the fitted MP2 gradient").

### 9.2 Spin-component-scaled (SCS-MP2) gradient

The closed-shell spin components of section 4 are
$E_{\mathrm{OS}} = \sum K t$ and $E_{\mathrm{SS}} = \sum K(t - \bar t)$, with
$K_{ij}^{ab} = (ia|jb)$, $t = K/\Delta$, $\bar t_{ij}^{ab} = t_{ij}^{ba}$ and
$\Delta_{ij}^{ab} = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b$. So

$$ E_{\mathrm{SCS}} = c_{\mathrm{OS}}E_{\mathrm{OS}} + c_{\mathrm{SS}}E_{\mathrm{SS}}
= \sum_{ijab} K_{ij}^{ab}\,\tilde T_{ij}^{ab}, \qquad
\tilde T = (c_{\mathrm{OS}} + c_{\mathrm{SS}})\,t - c_{\mathrm{SS}}\,\bar t, $$

which is MP2's $2t - \bar t$ at $(1, 1)$. The gradient of section 9 uses the
amplitudes in two places, and both carry over with $\tilde T$ in place of
$2t - \bar t$:

* **Integral derivative.** $E_{\mathrm{SCS}} = \sum K\,W K/\Delta$ with
  $W = (c_{\mathrm{OS}} + c_{\mathrm{SS}}) - c_{\mathrm{SS}}(a \leftrightarrow b)$,
  symmetric because $\Delta$ is invariant under $a \leftrightarrow b$, so
  $\partial E/\partial K_{ij}^{ab} = 2\tilde T_{ij}^{ab}$. This 2-PDM reaches
  the Lagrangian matrix $I$ and the 3-centre/2-centre RI response only
  through $x^P_{ia} = \sum_{jb}\tilde T_{ij}^{ab}B^P_{jb}$
  (`zvector.rs` `build_x_ov_weighted`).
* **Denominator derivative.** $\partial E/\partial \Delta_{ij}^{ab} =
  -t_{ij}^{ab}\tilde T_{ij}^{ab}$, so the unrelaxed densities are those above
  with $\tilde T$ in the second slot:

  $$ P_{ij} = -\sum_{kab} t_{ik}^{ab}\tilde T_{jk}^{ab}, \qquad
  P_{ab} = \sum_{ijc} t_{ij}^{ac}\tilde T_{ij}^{bc}. $$

  Bilinear in $(t, \tilde T)$, they are
  $c_{\mathrm{SS}}P^{\mathrm{MP2}} + (c_{\mathrm{OS}} - c_{\mathrm{SS}})P^{\mathrm{OS}}$
  with $P^{\mathrm{OS}} = P(t, t)$ (`rimp2.rs` `opposite_spin_density`).

Everything else — the frozen core–active block, the Z-vector right-hand
side, the relaxed and energy-weighted densities, the fitted (RI-JK)
reference terms of 9.1 — is built from $I$ and $P$ and needs no change.
`Mp2Intermediates::apply_spin_scaling` applies both substitutions;
$(c_{\mathrm{OS}}, c_{\mathrm{SS}}) = (1, 1)$ leaves the intermediates
untouched, so the RI-MP2 gradient runs through the same code bit for bit.
Checked against central finite differences of `scs_mp2` (H2O STO-3G and
cc-pVDZ, NH3 6-31G; $(6/5, 1/3)$ and $(1, 0)$; frozen core; RI-JK) in
`tests/scs_mp2_gradient.rs`.

**SCS weights, checked.** Symbolically, $\partial E/\partial K_{ij}^{ab}$
summed over both slots of a symmetric integral is $2\tilde T$, and
$\partial E/\partial\varepsilon_k = (P+P^{\mathsf T})_{kk}$,
$\partial E/\partial\varepsilon_c = (P+P^{\mathsf T})_{cc}$ with $\tilde T$ in
the second slot of $P$ **[gradient S]**; the full SCS gradient is part of
**[gradient G]**.

### 9.3 OO-RI-MP2 nuclear gradient

At converged OO-RI-MP2 orbitals the energy is stationary in every rotation
(section 7), so `oo_rimp2_gradient.rs` sets $z = 0$ and uses the
assembly above with the unrelaxed $W = (P+P^{\mathsf T})_{oo}\oplus(P+P^{\mathsf T})_{vv}$
of the semicanonical frame, $D^{\mathrm{corr}} = CWC^{\mathsf T}$, and two
changes in $\zeta$, because the orbitals are not canonical and $F_{ia}\ne0$:

$$ \zeta^{\mathrm{MO}} = \tfrac12\big(FW + (FW)^{\mathsf T}\big), \quad
\zeta^{\mathrm{MO}}_{ia} = \zeta^{\mathrm{MO}}_{ai} = (FW)_{ia}, \qquad
\zeta = C\zeta^{\mathrm{MO}}C^{\mathsf T} + 2\,C_oF_{oo}C_o^{\mathsf T}
\quad(\texttt{energy\_weighted\_density}), $$

$F$ the MO Fock matrix of the OO orbitals, and
$v^{\mathrm{s1occ}} = P_{\mathrm{occ}}\,(J-\tfrac12K)[2CWC^{\mathsf T}]\,P_{\mathrm{occ}}$
(`vhf_s1occ`). The occ–vir entry is the one-sided $(FW)_{ia}$ that $\tilde I$
uses. On the model of section 9, with OO-MP2 re-converged to
$\max|g| < 10^{-11}$ at every displacement, this $z=0$ gradient equals the
finite difference to $8\times10^{-12}$ relative; dropping $v^{\mathrm{s1occ}}$
breaks it **[gradient O]**. Frozen core is not covered (the core rotations are
not optimized, so the $z=0$ form is not exact there).

## 10. Static polarizability

**Finite field** (`ff_polar.rs:390` `mp2_polarizability_static`). The core
Hamiltonian gains $-\sum_d F_d\,r_d$; at each field the RHF is reconverged and
the relaxed MP2 density $P^{\mathrm{rel}}(F)$ built. Then

$$ \mu_d(F) = -\operatorname{Tr}[P^{\mathrm{rel}}(F)\,r_d] + \sum_A Z_AR_{A,d}, \qquad
\alpha_{de} = -\frac{\mu_d(+F_e) - \mu_d(-F_e)}{2|F|} , $$

symmetrized. $P^{\mathrm{rel}}$ is `static_relaxed_density_ao`
(`cpks_polar.rs:483`): $2\delta_{ij} + (P+P^{\mathsf T}) + z$ with
$X_{ai} = L_{ai} + [(2J-K)[P+P^{\mathsf T}]]_{ai}$ ($L$ = `build_lagrangian`) and
the section-9 operator, $(\Delta\varepsilon + \tfrac12$`compute_az_product`$)\,z = -X$.
This $z$ equals `solve_zvector`'s, and $\mu_d(0) = -\partial E/\partial F_d$
of the RI-MP2 energy. `DensityMode::Unrelaxed` sets $z = 0$.

**Analytic** (`mp2_polarizability_analytic`, delegating to
`analytic_alpha_relaxed`). The field enters $h$ as $+F\,r$. Per axis $x$, the
dipole CPHF $(A+B)\,U^x = -r^x_{vo}$ gives the first-order Fock response
(`first_order_fock_response`)

$$ \partial F = r^x + J[\partial D] - \tfrac12 K[\partial D], \qquad
\partial D = 2\,(C_v U^x C_o^{\mathsf T} + C_o U^{x\mathsf T} C_v^{\mathsf T}) , $$

and the first-order reference at field $s$ is the set of canonical orbitals of
$F + s\,\partial F$ (`reference_at_field`). Its occupied space, orbital energies
and oo/vv Fock blocks equal the field-$s$ SCF's to $O(s^2)$. Then

$$ \alpha_{xy} = -\operatorname{Tr}\!\left[\frac{P^{\mathrm{rel}}(+h) - P^{\mathrm{rel}}(-h)}{2h}\, r_y\right],
\qquad h = 10^{-4}, $$

with $P^{\mathrm{rel}}$ = `static_relaxed_density_ao` on that reference. The
same difference of the SCF density reproduces the CPHF
$\alpha^{\mathrm{HF}} = 4\mu^{\mathsf T}(A+B)^{-1}\mu$ to $4\times10^{-9}$
relative. A hand assembly $\partial P + \partial z + 2U$ with the
canonical-denominator $\partial t_2$ omits the off-diagonal oo/vv Fock response
and the rotation $\Theta D^{\mathrm{corr}} + D^{\mathrm{corr}}\Theta^{\mathsf T}$
of the correlated density $D^{\mathrm{corr}} = P + P^{\mathsf T} + z$; on
water/STO-3G these contribute 0.0009 and $-0.0054$ to $\alpha_{zz}$, and 0.048
and $-0.040$ to $\alpha_{yy}$.

The SCF response (CPHF, $\partial F$, the $(2J-K)[P+P^{\mathsf T}]$ term and
the Z-vector Hessian) uses `bounds.op`, the operator the RHF was converged
with; the MP2 amplitudes and the Lagrangian $L$ use `op`. With a Coulomb RHF
and `op` $=\operatorname{erfc}(\omega r)$ this is $-\partial^2/\partial F^2$ of
$E_{\mathrm{RHF}} + E_{\mathrm{MP2}}[\operatorname{erfc}]$ (att-MP2), checked
against a 5-point finite field of that energy
(`attenuated_relaxed_alpha_matches_energy_finite_field`).

**Frozen-amplitude C6** (`cpks_polar.rs:1128` `frozen_mp2_c6_molecular`):
$\alpha^{\mathrm{MP2}}(i\omega) = \alpha^{\mathrm{MP2}}_{\mathrm{iso}}(0)\,
\alpha^{\mathrm{HF}}_{\mathrm{iso}}(i\omega)/\alpha^{\mathrm{HF}}_{\mathrm{iso}}(0)$,
the static magnitude from `analytic_alpha_relaxed` and the HF shape on
`bounds.op`; $C_6 = \tfrac{3}{\pi}\sum_k w_k\,\alpha^{\mathrm{MP2}}(i\omega_k)^2$.

The singlet response matrices are (`cpks_polar.rs:846` `build_apb_amb`)

$$ (A+B)_{ai,bj} = \Delta\varepsilon\,\delta + 4(ai|bj) - (ab|ij) - (aj|bi), \qquad
(A-B)_{ai,bj} = \Delta\varepsilon\,\delta + (aj|bi) - (ab|ij), $$

and the HF-level polarizabilities are
$\alpha = 4\mu^{\mathsf T}(A+B)^{-1}\mu$ and
$\alpha(i\omega) = 4\mu^{\mathsf T}(A-B)\left[(A-B)(A+B) + \omega^2\right]^{-1}\mu$
(`cpks_polar.rs:1000` `dynamic_cphf_alpha_iw`). `solve_cphf_cg` applies
$(A+B) = \Delta\varepsilon + \tfrac12$`compute_az_product`.

**Relaxed dipole.** The identity $\mu_d(0) = -\partial E/\partial F_d$ is
the one-electron case $h \to h + \lambda R$ of **[gradient G]** (G1) for the
`solve_zvector` relaxed density. That `static_relaxed_density_ao`'s $z$ equals
`solve_zvector`'s, and the analytic field derivative, are not checked by a
script.

## 11. CABS for F12

`build_cabs` (`f12.rs`) forms the RI basis as the union OBS ∪ aux
(`union_basis_sets`: per element, the OBS shells then the aux shells). The
union functions are therefore atom-major and interleaved: on each atom the
OBS functions come first and that atom's aux functions follow, so neither set
is contiguous in the union. A provenance mask (`union_obs_mask`) splits the
union indices into the OBS set $O$ and the aux set $a$, and the overlap
blocks are taken in that partition. With $S_{OO}$ inverted by
eigendecomposition (an error if it is not positive definite),

$$ Q = S_{aa} - S_{aO}S_{OO}^{-1}S_{Oa} = V\,w\,V^{\mathsf T}, \qquad
c_k = \frac{1}{\sqrt{w_k}}\begin{pmatrix} -S_{OO}^{-1}S_{Oa}v_k \\ v_k \end{pmatrix}, \quad w_k > 10^{-8} \ (\texttt{CABS\_NULL\_THRESH}) , $$

written in the partitioned order $[O\,|\,a]$. The rows of each $c_k$ are then
scattered back to the interleaved union order, so `Cabs::coeffs` is
$n_{\rm RI}\times n_{\rm CABS}$ over the union functions as
`PreparedBasis` orders them, `Cabs::s_rr` is the union overlap in that order,
and `Cabs::s_or` is its OBS rows. A permutation of the union functions leaves
both properties unchanged, $c^{\mathsf T}S_{RR}\,c = I$ and $S_{O,R}\,c = 0$
**[check 7]**. The crate builds CABS only; there is no F12 energy.

## 12. Local MP2

Amplitude-threshold local MP2 (`lmp2_amplitude.rs`, `lmp2_direct.rs`) is
derived in `wiki/sympy_derivations.md` §14; user-facing settings are in
`site/src/methods/mp2.md`.

## 13. MP2-V

`att_mp2_vv10` (closed shell) and `u_att_mp2_vv10` (UHF or ROHF) in
`att_vv10.rs` evaluate the post-HF MP2-V energy of Goldey, Belzunces and
Head-Gordon (JCTC 11, 4159 (2015)), CLI `method.kind = "mp2-v"`, Python
`run_mp2_v`:

$$ E_{\mathrm{MP2\text{-}V}} = E_{\mathrm{HF}} + E_c^{\mathrm{att}}[r_0] + E_{\mathrm{nl}}[\rho_{\mathrm{HF}}; b, C, r_0] \qquad (\texttt{assemble}). $$

**Correlation half.** $E_c^{\mathrm{att}}$ is RI-MP2 (section 2,
`ri_mp2_spin_components`) or U-RI-MP2 (section 3, `u_ri_mp2`) with
$\hat w = \operatorname{terfc}(r;r_0,\omega)/r$ (`mp2_operator`):
`Operator::terfc` at the linked $\omega = 1/(\sqrt2 r_0)$ when
`omega` is `None`, `Operator::terfc_with_omega` otherwise. The erfc control
uses $\operatorname{erfc}(\omega r)/r$ with the same linked $\omega$.

**Dispersion half.** VV10 (`ferric-dft` `vv10.rs`) on the HF total density
$\rho = \rho_\alpha + \rho_\beta$ (`density_total`, spin-free in both
paths), written as $E_{\mathrm{nl}} = \sum_g w_g\rho_g(\beta + \tfrac12 f_g)$
with $\beta = (3/b^2)^{3/4}/32$ local and
$f_g = \sum_{g'}w_{g'}\rho_{g'}\Phi(g,g')\,d(R_{gg'})$ the pair term. Eq. 11
of the paper damps only the pair kernel:

$$ d(R) = 1 - \operatorname{terfc}(R; r_0, \omega)^2 \qquad (\texttt{Vv10Damping::factor\_from\_r2}), $$

with the same $(r_0,\omega)$ as the MP2 operator (`effective_vv10_damping`
errors on a mismatch). $d(0) = 0$ and $d \to 1$ at large $R$, the complement of
what attenuated MP2 keeps; beyond
$r_{\mathrm{one}} = r_0 + 5/\omega$ (`damping_r_one`) terfc $< 10^{-12}$ and
$1 - \operatorname{terfc}^2$ rounds to exactly 1.0, so skipping the erf pair
there is bit-exact **[family E]**. The VV10 kernel $\Phi$ and $\beta$ are
`ferric-dft`'s.

**Spin.** The closed-shell path takes $E_{\mathrm{OS}} + E_{\mathrm{SS}}$, the
open-shell path $E_{\alpha\alpha} + E_{\beta\beta} + E_{\alpha\beta}$; both
are the spin-orbital MP2 energy of the attenuated integrals
(**[family A]**, **[family B]**, **[check 1]**, **[check 2]**).

## 14. Unrestricted OO-RI-MP2 orbital gradient

`u_oo_ri_mp2` minimizes $E = E_{\mathrm{UHF}}(C^\alpha,C^\beta) +
E_{\alpha\alpha} + E_{\beta\beta} + E_{\alpha\beta}$ (section 3) evaluated in
each spin's semicanonical frame, over $C^\tau \leftarrow C^\tau U(\kappa^\tau)$
(section 7). With $F^\tau = h + J[D^\alpha + D^\beta] - K[D^\tau]$ and
$D^\tau = C^\tau_oC^{\tau\mathsf T}_o$, a rotation $\kappa^\tau_{ck}$ moves
$\delta C_k = C_c$, $\delta C_c = -C_k$, so $\delta C = CX$ with
$X_{ck} = 1 = -X_{kc}$, and

$$ \delta E_{\mathrm{UHF}} = \operatorname{Tr}[F^\tau\delta D^\tau] = 2F^\tau_{ck}, \qquad
\delta F^\sigma_{\mathrm{MO}} = F^\sigma X - XF^\sigma + C^{\sigma\mathsf T}\big(J[\delta D^\tau] - \delta_{\sigma\tau}K[\delta D^\tau]\big)C^\sigma . $$

The U-MP2 energy depends on the orbitals through the integrals and, at fixed
integrals, on $F^\sigma$ through $W^\sigma = \partial E/\partial F^\sigma$,
the unrelaxed density of the full-Fock functional (`build_u_mp2_density`):

$$ W^\alpha_{ij} = -\tfrac12\sum_{kab}t^{\alpha\alpha}_{ikab}t^{\alpha\alpha}_{jkab} - \sum_{KaB}t^{\alpha\beta}_{iKaB}t^{\alpha\beta}_{jKaB}, \qquad
W^\alpha_{ab} = \tfrac12\sum_{ijc}t^{\alpha\alpha}_{ijac}t^{\alpha\alpha}_{ijbc} + \sum_{iJC}t^{\alpha\beta}_{iJaC}t^{\alpha\beta}_{iJbC}, $$

$t^{\alpha\alpha} = [(ia|jb)-(ib|ja)]/D$, $t^{\alpha\beta} = (ia|JB)/D$, and
$\beta$ by exchanging the spin labels. Contracting $W$ with $\delta F$, using
$\sum_\sigma\operatorname{Tr}[P^\sigma G^\sigma[\delta D]] = \operatorname{Tr}[\delta D\,G[P]]$
($P^\sigma = C^\sigma W^\sigma C^{\sigma\mathsf T}$) and the antisymmetry of
$WF - FW$, gives the gradient `u_gradient_solver_frame` assembles:

$$ g^\tau_{ck} = 2F^\tau_{ck} + \Big[\frac{\partial E_{\mathrm{MP2}}}{\partial\kappa^\tau_{ck}}\Big]_{\varepsilon}
+ 2\big[C^{\tau\mathsf T}(J[P^\alpha + P^\beta] - K[P^\tau])C^\tau\big]_{ck} + 2\big(F^\tau W^\tau - W^\tau F^\tau\big)_{ck} , $$

all in the semicanonical frame $C^\tau u^\tau$, carried back by
$g^\tau = u^\tau_v\,g^\tau_{\mathrm{sc}}\,u^{\tau\mathsf T}_o$. The integral
term at fixed denominators is
$\sum t^{\sigma\sigma}_{ijab}\,\delta(ia|jb) + 2\sum t^{\alpha\beta}_{iJaB}\,\delta(ia|JB)$ with
$\delta(ia|jb) = \delta_{ik}(ca|jb) + \delta_{jk}(ia|cb) - \delta_{ac}(ik|jb) - \delta_{bc}(ia|jk)$
for the rotated spin (`compute_u_mp2_orbital_gradient`). At non-stationary,
non-semicanonical orbitals of a 3$\alpha$/2$\beta$ model with separate
four-centre and RI factors, this equals Richardson differences of the
brute-force spin-orbital energy to $5\times10^{-12}$ for both spins
**[gradient U]**; the closed-shell restriction ($C^\alpha = C^\beta$,
$g = g^\alpha + g^\beta$) is the optimizer of **[gradient O]**. The integral
term in the proof is the derivation above, not a transcription of
`compute_u_mp2_orbital_gradient`'s loop.

## 15. Unrestricted RI-MP2 nuclear gradient

`u_rimp2_gradient.rs` `u_rimp2_gradient_analytical`; UHF reference, both J/K
routes of §9.1 (four-centre and RI-JK on one auxiliary basis). Proof:
`scripts/u_rimp2_gradient_sympy.py` (parts S, G, C below).

**Energy and its weights.** With $B^{\sigma}_{P,ia}$ the dressed factors of
spin $\sigma$ ($i,a$ of that spin; capital indices for $\beta$),
$t^{\alpha\alpha}_{ijab} = [(ia|jb) - (ib|ja)]/D_{ijab}$ and
$t^{\alpha\beta}_{iJaB} = (ia|JB)/D_{iJaB}$,

$$ E^{(2)} = \tfrac14\sum |\langle ij\Vert ab\rangle|^2/D \,(\alpha\alpha)
 + \tfrac14\sum |\langle IJ\Vert AB\rangle|^2/D \,(\beta\beta)
 + \sum_{iJaB} (ia|JB)\,t^{\alpha\beta}_{iJaB}, $$

and at fixed orbital energies $\partial E^{(2)}/\partial B^{\sigma} = 2x^{\sigma}$ with

$$ x^{\alpha}_{P,ia} = \sum_{jb} t^{\alpha\alpha}_{ijab}B^{\alpha}_{P,jb} + \sum_{JB} t^{\alpha\beta}_{iJaB}B^{\beta}_{P,JB}, \qquad
 x^{\beta}_{P,JB} = \sum_{KC} t^{\beta\beta}_{JKBC}B^{\beta}_{P,KC} + \sum_{ia} t^{\alpha\beta}_{iJaB}B^{\alpha}_{P,ia}. $$

The opposite-spin block enters both weights **[check S]**; $E^{(2)}$ is
homogeneous of degree 4 in $B$, so $\sum_\sigma B^{\sigma}\cdot x^{\sigma} = 2E^{(2)}$
(the in-module test `energy_weights_at_alpha_equals_beta_are_the_closed_weights`).

**Per-spin Lagrangian pieces.** Each is the closed-shell piece of §9 built
from one spin, with the closed factors halved where the closed form carries
both spins:

$$ I^{\sigma}_{qi} = -\sum_{Pa} x^{\sigma}_{P,ia}B^{\sigma}_{P,qa}, \qquad I^{\sigma}_{qa} = -\sum_{Pi} x^{\sigma}_{P,ia}B^{\sigma}_{P,qi}, $$

$P^{\sigma}$ the unrelaxed spin density (`build_u_mp2_density`) symmetrized,
$P^{\sigma}_{kI} = I^{\sigma}_{kI}/(\varepsilon_k-\varepsilon_I)$ on a frozen
core, $v^{\sigma}[X] = J[X^{\alpha}+X^{\beta}] - K[X^{\sigma}]$, and

$$ X^{\sigma}_{ai} = \big[C_\sigma^{\mathsf T} v^{\sigma}[P]\,C_\sigma\big]_{ai} + I^{\sigma}_{ia} - I^{\sigma}_{ai}. $$

**Coupled Z-vector.** The UHF orbital Hessian couples the spins through
Coulomb only:

$$ (\varepsilon^{\sigma}_a - \varepsilon^{\sigma}_i)\,z^{\sigma}_{ai} + \big[C_\sigma^{\mathsf T}\big(J[D^{z,\alpha}+D^{z,\beta}] - K[D^{z,\sigma}]\big)C_\sigma\big]_{ai} = -X^{\sigma}_{ai}, \qquad
 D^{z,\sigma} = C_{v\sigma}z^{\sigma}C_{o\sigma}^{\mathsf T} + \text{h.c.}, $$

solved for the pair by DIIS to $\max|r| < 10^{-8}$; a solve that reaches its
cap is an error.

**Assembly.** With $P^{\sigma}_{\mathrm{rel}} = P^{\sigma} + z^{\sigma}$ (both
off-diagonal blocks), $D^{\sigma}$ the UHF spin densities,

$$ \frac{dE}{dx} = \operatorname{Tr}\big[(D+P_{\mathrm{rel}})_{\mathrm T}h'\big]
 + \sum_{\mu\nu}\Big(\sum_\sigma \big[\mathrm{im1}^{\sigma} - \zeta^{\sigma} - W^{\sigma}_{\mathrm{HF}} - P^{\sigma}_{\mathrm{occ}}v^{\sigma}[P_{\mathrm{rel}}]P^{\sigma}_{\mathrm{occ}}\big]\Big)_{\mu\nu}S'_{\mu\nu}
 + \partial_x U\big((D^{\alpha},D^{\beta}),(D^{\alpha}+2P^{\alpha}_{\mathrm{rel}},D^{\beta}+2P^{\beta}_{\mathrm{rel}})\big)
 + \sum_\sigma\Big[2\sum y^{\sigma}(P|ia)'_\sigma - \sum_{PQ}V'_{PQ}\sum_{ia}(V^{-1}u^{\sigma})^P_{ia}y^{\sigma,Q}_{ia}\Big], $$

$\mathrm{im1}$, $\zeta$ and $P_{\mathrm{occ}}$ per spin as in §9 (unit
occupation, $W^{\sigma}_{\mathrm{HF}} = C_{o\sigma}\varepsilon_{o}C_{o\sigma}^{\mathsf T}$),
$y^{\sigma} = V^{-1/2}x^{\sigma}$ and

$$ U(A,B) = \tfrac12\operatorname{Tr}\big[A_{\mathrm T}J[B_{\mathrm T}]\big] - \tfrac12\sum_\sigma\operatorname{Tr}\big[A^{\sigma}K[B^{\sigma}]\big] $$

differentiated through the ERIs only (`twoelectron_gradient_bilinear_uhf`;
on RI-JK `fitted_bilinear_gradient_open`, the polarization of §9.1 per spin).
On a model with $S$, $h$, the ERIs, $u$ and $V$ all moving, the right-hand side
equals Richardson differences of the re-solved $E_{\mathrm{UHF}}+E^{(2)}$
($n_\alpha=3$, $n_\beta=2$) to $1.5\times10^{-11}$ relative with and without a
frozen core, and $\operatorname{Tr}[(D+P_{\mathrm{rel}})_{\mathrm T}R]$ equals
$dE/d\lambda$ for $h\to h+\lambda R$ to $10^{-11}$ **[check G]**.

**Closed-shell limit.** For $\alpha\equiv\beta$: $t^{\alpha\alpha} = t - t^{\mathrm{swap}}$,
$t^{\alpha\beta} = t$, so $x^{\sigma} = x$ of §9; $I^{\sigma} = I/2$,
$P^{\alpha}+P^{\beta} = P+P^{\mathsf T}$, $X^{\sigma} = X/2$, $z^{\sigma} = z/2$,
$U = B$ of §9, and the RI response is the closed one (weight 1 per spin
against 2). The model reproduces the closed assembly to $2\times10^{-16}$
**[check C]**; in Rust, a closed-shell RHF written as a UHF gives the
restricted gradient on both routes (`closed_shell_limit_matches_restricted_*`).

**ROHF is refused.** A ROHF reference is not stationary under independent
$\alpha$/$\beta$ rotations and its MP2 is evaluated on semicanonical orbitals,
so its gradient needs the ROHF Z-vector and the semicanonicalization
response; neither is built (`ROHF_REFUSAL`).

**No external reference.** PySCF has no density-fitted UMP2 gradient
(`pyscf/grad/ump2.py` is conventional), so the molecular anchor is the finite
difference of ferric's own U-RI-MP2 energy (`tests/u_rimp2_gradient_fd.rs`,
OH, NH2, O2 in 6-31G, the bar derived in the file header).

**Memory.** The RI-JK route's peak is
`gradient_memory::fitted_gradient_plan_unrestricted`: §9.1's terms with the
three amplitude blocks, two $B^{\sigma}$ and two $x^{\sigma}$, one full-MO
$b^{\sigma}(P|pq)$ at a time, and four exchange-channel densities
($8\,n_{\mathrm{ao}}^2$ in place of $4\,n_{\mathrm{ao}}^2$); checked before the
UHF.
