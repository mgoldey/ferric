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
Derivations that live in the project notes are cited as
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

**Range split.** MP2 is quadratic in the integrals. Writing
$(ia|jb) = s + l$ for the erfc and erf parts,

$$ E[s+l] = E[s] + E[l] + \sum_{ijab}\frac{4sl - s\,l^\times - l\,s^\times}{D_{ij}^{ab}} , $$

so $E[\operatorname{erfc}] + E[\operatorname{erf}] \ne E[1/r]$ **[check 4]**.
`rs_mp2_decomposition` (`attenuated.rs:347`) returns the three energies, and
RS-MP2 + LR-RPA (`ferric-rpa::rs_mp2_rpa`) is built on this cross term.

## 6. MP3

`mp3_energy` (`mp3.rs:143`) works in spin orbitals with
$t_{ij}^{ab} = \langle ij\|ab\rangle/D_{ij}^{ab}$ (`mp3.rs:265`) built from RI
integrals:

$$ E^{(2)} = \tfrac14\sum t_{ij}^{ab}\langle ij\|ab\rangle, \qquad
E^{(3)} = \tfrac18\sum t_{ij}^{ab}\langle ab\|cd\rangle t_{ij}^{cd}
+ \tfrac18\sum \langle kl\|ij\rangle t_{ij}^{ab} t_{kl}^{ab}
+ \sum t_{ij}^{ab}\langle kb\|cj\rangle t_{ik}^{ac} $$

(`mp3.rs:281`, `:292`, `:300`, `:314`). The ladder prefactors and the
ring-term convention are derived in `wiki/sympy_derivations.md` §2b.

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

## 9. Analytic RI-MP2 gradient and Z-vector

**Unrelaxed density** (`oo_rimp2.rs:990` `build_mp2_density`), with
$t_{ij}^{ab} = (ia|jb)/D_{ij}^{ab}$:

$$ P_{ij} = -\sum_{kab} t_{ik}^{ab}\left(2t_{jk}^{ab} - t_{jk}^{ba}\right), \qquad
P_{ab} = \sum_{ijc} t_{ij}^{ac}\left(2t_{ij}^{bc} - t_{ij}^{cb}\right). $$

**Lagrangian matrix** (`zvector.rs:393`), with
$x^P_{ia} = \sum_{jb}(2t_{ij}^{ab} - t_{ij}^{ba})B^P_{jb}$ (`zvector.rs:335`)
and the full-MO factor $B^P_{pq}$:

$$ I_{qi} = -2\sum_{Pa} x^P_{ia}B^P_{qa}, \qquad I_{qa} = -2\sum_{Pi} x^P_{ia}B^P_{qi}. $$

Frozen-core rows and columns stay zero (`zvector.rs:387`).

**Z-vector** (`zvector.rs:58` `solve_zvector`). The right-hand side follows
PySCF `grad/mp2.py`:

$$ X_{ai} = \left[C^{\mathsf T}(2J - K)[D^{\mathrm{corr}}]\,C\right]_{ai} + I_{ia} - I_{ai}, \qquad D^{\mathrm{corr}} = C\,(P + P^{\mathsf T})\,C^{\mathsf T}, $$

and the CPHF equation is

$$ (\varepsilon_a - \varepsilon_i)\,z_{ai} + \sum_{bj}\left[4(ai|bj) - (ab|ij) - (aj|bi)\right] z_{bj} = -X_{ai}, $$

solved by Jacobi-preconditioned DIIS to $\max|r| < 10^{-8}$. The coupling is
$\tfrac12$ of `compute_az_product` (`zvector.rs:159`), which returns the vo
block of $(4J - K - K^{\mathsf T})[D^z]$ with
$D^z = \sum_{ai}z_{ai}(C_aC_i^{\mathsf T} + C_iC_a^{\mathsf T})$, twice the
bracket above **[check 5]**.

**Relaxed density** (`zvector.rs:519`): $2\delta_{ij} + P_{ij}$ on the
occupied block, $P_{ab}$ on the virtual block, $z_{ai}$ on both off-diagonal
blocks.

**Gradient** (`gradient.rs:132` `rimp2_gradient_analytical`): overlap terms
from $I$, the orbital-energy-weighted relaxed density and the
$J - \tfrac12K$ potential of the correlation density; the core-Hamiltonian
term with the full relaxed density; the bilinear two-electron term
$\Gamma(D^{\mathrm{HF}}, D^{\mathrm{HF}} + 2D^{\mathrm{corr}})$; and the RI
3-centre/2-centre response with the closed-shell 2-PDM weight
$\Gamma_2 = 2(2t - t^{\mathsf T})$ (`gradient.rs:93`–`:127` lists each block
with its PySCF line). Frozen core enters only through the active-orbital
ranges: there is no frozen–active orbital-response block.

## 10. Static polarizability

**Finite field** (`ff_polar.rs:523` `mp2_polarizability_static`). The core
Hamiltonian gains $-\sum_d F_d\,r_d$; at each field the RHF is reconverged and
the relaxed MP2 density $P^{\mathrm{rel}}(F)$ built. Then

$$ \mu_d(F) = -\operatorname{Tr}[P^{\mathrm{rel}}(F)\,r_d] + \sum_A Z_AR_{A,d}, \qquad
\alpha_{de} = -\frac{\mu_d(+F_e) - \mu_d(-F_e)}{2|F|} , $$

symmetrized. Its Z-vector RHS is `build_lagrangian` and its CG operator is
$\Delta\varepsilon + $ `compute_az_product` without the $\tfrac12$
(`ff_polar.rs:409`), so the two-electron coupling is twice that of section 9.
`DensityMode::Unrelaxed` sets $z = 0$.

**Analytic** (`cpks_polar.rs:73` `mp2_polarizability_analytic`, delegating to
`analytic_alpha_full`, `cpks_polar.rs:1578`) solves the dipole CPHF
$(A+B)\,U^x = -\mu^x_{vo}$ densely in the full MO basis, rotates the MO
integrals with the generator $\Theta_{ai} = U_{ai}$, $\Theta_{ia} = -U_{ai}$,
differentiates the relaxed density by central difference in that rotation,
and contracts $\alpha_{pq} = -\sum \partial D\cdot r$. The singlet response
matrices are (`cpks_polar.rs:1736` `build_apb_amb`)

$$ (A+B)_{ai,bj} = \Delta\varepsilon\,\delta + 4(ai|bj) - (ab|ij) - (aj|bi), \qquad
(A-B)_{ai,bj} = \Delta\varepsilon\,\delta + (aj|bi) - (ab|ij), $$

and the HF-level polarizabilities are
$\alpha = 4\mu^{\mathsf T}(A+B)^{-1}\mu$ and
$\alpha(i\omega) = 4\mu^{\mathsf T}(A-B)\left[(A-B)(A+B) + \omega^2\right]^{-1}\mu$
(`cpks_polar.rs:1890` `dynamic_cphf_alpha_iw`). In `solve_cphf_cg_scaled`
the operator is $\Delta\varepsilon + s\cdot$`compute_az_product`; $s = \tfrac12$
is $(A+B)$.

## 11. CABS for F12

`build_cabs` (`f12.rs:99`) forms the RI basis as the union OBS ∪ aux, ordered
$[\mathrm{O}\,|\,a]$, and projects OBS out of the auxiliary block:

$$ Q = S_{aa} - S_{aO}S_{OO}^{-1}S_{Oa} = V\,w\,V^{\mathsf T}, \qquad
c_k = \frac{1}{\sqrt{w_k}}\begin{pmatrix} -S_{OO}^{-1}S_{Oa}v_k \\ v_k \end{pmatrix}, \quad w_k > 10^{-8} , $$

so $c^{\mathsf T}S_{RR}\,c = I$ and $S_{O,R}\,c = 0$ **[check 7]**. The crate
builds CABS only; there is no F12 energy.

## 12. Local MP2

Amplitude-threshold local MP2 (`lmp2_amplitude.rs`, `lmp2_direct.rs`) is
derived in `wiki/sympy_derivations.md` §14; user-facing settings are in
`site/src/methods/mp2.md`.
