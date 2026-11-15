# 1.3 Spin adaptation: from spin orbitals to spatial orbitals

<!-- section: 1.3; status: edit; registry: ;
     theory: crates/ferric-mp2/theory.md#"Conventions" -->

A derivation is cleanest in spin orbitals. A working equation is cheapest in
spatial orbitals, because the spin sums are done once, by hand, and the code
never stores a spin-orbital tensor. This section does the reduction for the
quantities every later chapter reuses: the reference energy, the Fock matrix,
the second-order energy, the unrestricted spin blocks and the closed-shell
coupled-cluster amplitudes. Each identity has a verified box or says it has
not.

| Index | Range in this section | Code name |
|---|---|---|
| $i, j$ | occupied **spatial** orbitals | `0..nocc` |
| $a, b$ | virtual **spatial** orbitals | `0..nvir` |
| $i_\sigma$ | an occupied spin orbital, spatial part $i$, spin $\sigma$ | `2*i + s` |
| $\bar\imath, \bar a$ | the $\beta$ spatial orbital in an unrestricted expression | `b`-labeled arrays |

| Entry point | Location |
|---|---|
| Spin-orbital tensors | `ferric-mp2/src/spinorbital.rs::{spin,spat,asym_oovv,asym_phys}` |
| Closed-shell to spin-orbital amplitudes | `ferric-cc/src/ccsd_closed_shell.rs::expand_amplitudes_to_spin_orbital` |
| Closed-shell pair kernel | `ferric-mp2/src/rimp2.rs::pair_energy` |
| Unrestricted kernels | `ferric-mp2/src/u_rimp2.rs::{same_spin_pair_kernel,opposite_spin_pair_kernel}` |

## 1.3.1 The integral selection rule

Equation (1.3) says that an antisymmetrized integral is a direct term, nonzero
when the spins pair as $(p,r)$ and $(q,s)$, minus an exchange term, nonzero
when they pair as $(p,s)$ and $(q,r)$. Between occupied pair $(i_\sigma, j_\tau)$
and virtual pair $(a_\rho, b_\kappa)$:

$$ \langle i_\sigma j_\tau\|a_\rho b_\kappa\rangle = (ia|jb)\,\delta_{\sigma\rho}\delta_{\tau\kappa} - (ib|ja)\,\delta_{\sigma\kappa}\delta_{\tau\rho} . $$

| Spins of $(i, j, a, b)$ | Direct term | Exchange term | Value |
|---|---|---|---|
| $\alpha\alpha\alpha\alpha$ or $\beta\beta\beta\beta$ | yes | yes | $(ia\vert jb) - (ib\vert ja)$ |
| $\alpha\beta\alpha\beta$ or $\beta\alpha\beta\alpha$ | yes | no | $(ia\vert jb)$ |
| $\alpha\beta\beta\alpha$ or $\beta\alpha\alpha\beta$ | no | yes | $-(ib\vert ja)$ |
| any other | no | no | 0 |

Every spin reduction in this section is a sum over this table.

## 1.3.2 A closed-shell reference

A closed-shell reference occupies each of $n_o$ spatial orbitals with an
$\alpha$ and a $\beta$ electron in the same spatial function. The occupied
spin orbitals are $i_\alpha$ and $i_\beta$. Applying the table to (1.4), with
$\sum_{\sigma\tau}1 = 4$ for the direct term and $\sum_{\sigma\tau}\delta_{\sigma\tau} = 2$
for the exchange term,

$$ E_{\mathrm{ref}} - E_{\mathrm{nn}} = 2\sum_i h_{ii} + \sum_{ij}\big[\,2(ii|jj) - (ij|ij)\,\big] . \tag{1.10} $$

> **Verified equation 1.10** · the closed-shell reference energy
>
> $\langle 0|\hat H|0\rangle$ expanded from spatial integrals with the spin
> selection rule of (1.3) equals $2\sum_i h_{ii} + \sum_{ij}[2(ii|jj) - (ij|ij)]$,
> symbolically on 3 occupied orbitals and numerically against the AO route
> $\mathrm{tr}[D(h+F)]/2$ with $D = 2C_oC_o^{\mathsf T}$ at random orbitals.
>
> | | |
> |---|---|
> | Proof | `scripts/hf_energy_sympy.py`, the check labelled "claim N8". The converged-SCF check labelled "claim N2" and the UHF-to-RHF check labelled "claim S3" are the neighbouring anchors |
> | Negative control | exchange without the spin delta must differ from the AO energy; the runner applies that edit to the script and requires a red run |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/rhf.rs::solve_rhf` |
>
> <!-- verified: id=so.closed-shell-energy; script=scripts/hf_energy_sympy.py; claim=N8; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/rhf.rs::solve_rhf -->

The Fock matrix of (1.5) reduces the same way. For occupied $k_\sigma$ summed
over both spins, $f_{pq} = \big(h + 2J - K\big)_{[p][q]}\,\delta_{\sigma(p)\sigma(q)}$,
with $J_{pq} = \sum_k (pq|kk)$ and $K_{pq} = \sum_k (pk|qk)$ over the occupied
spatial orbitals. In the AO basis and with the total density
$D = 2 C_o C_o^{\mathsf T}$ this is the $F = h + J[D] - \tfrac12 K[D]$ that the
code builds (notation chapter, section 5).

> **Verified equation 1.11** · the spin-orbital Fock matrix is the spatial one
>
> On a random model, the spin-orbital Fock matrix built from the
> antisymmetrized integrals (1.3) equals $F\otimes 1_{\mathrm{spin}}$, with $F = h + 2J - K$.
>
> | | |
> |---|---|
> | Proof | `scripts/oomp2_mp3_identities_sympy.py`, the check labelled "claim B0" |
> | Negative control | none for this check |
> | Runner | `crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py` |
> | Code | `ferric-mp2/src/spinorbital.rs::asym_oovv` |
>
> <!-- verified: id=so.fock-closed-shell; script=scripts/oomp2_mp3_identities_sympy.py; claim=B0; runner=crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py; code=ferric-mp2/src/spinorbital.rs::asym_oovv -->

## 1.3.3 The closed-shell second-order energy

Apply the table to (1.8). Pairs with opposite spin in the occupied indices
are the $\alpha\beta$ and $\beta\alpha$ pairs. Each of the four spin
patterns that survive contributes $\tfrac14$ of $(ia|jb)^2/\Delta$, so the
opposite-spin part is $\sum (ia|jb)^2/\Delta$ without a factor. Same-spin pairs
carry the full antisymmetrized integral and a factor $\tfrac14$ for each of
$\alpha\alpha$ and $\beta\beta$. Summing the two gives $\tfrac12\sum[(ia|jb) - (ib|ja)]^2/\Delta$.
Relabeling $a\leftrightarrow b$ shows $\sum (ib|ja)^2 = \sum (ia|jb)^2$, so
this equals $\sum (ia|jb)[(ia|jb) - (ib|ja)]/\Delta$.

$$ E_{\mathrm{OS}} = -\sum_{ijab}\frac{(ia|jb)^2}{\Delta_{ij}^{ab}}, \qquad E_{\mathrm{SS}} = -\sum_{ijab}\frac{(ia|jb)\,[(ia|jb) - (ib|ja)]}{\Delta_{ij}^{ab}}, \qquad E^{(2)} = E_{\mathrm{OS}} + E_{\mathrm{SS}} . \tag{1.12} $$

> **Verified equation 1.12** · spin adaptation of the MP2 energy
>
> The spatial sums of (1.12) equal the opposite-spin and same-spin parts of
> the spin-orbital sum (1.8), and their sum equals (1.8). The unique-pair sum
> over $j \ge i$ with weight 2 off the diagonal equals the full sum.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 1; the same identity is proved symbolically, with every $(ia\vert jb)$ a free symbol, in `scripts/mp2_family_sympy.py`, part A |
> | Negative control | $E_{\mathrm{SS}}$ without the exchange term must differ; the unique-pair sum with off-diagonal weight 1 must differ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/rimp2.rs::pair_energy` |
>
> <!-- verified: id=so.mp2-spin-adaptation; script=scripts/mp2_variant_identities.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/rimp2.rs::pair_energy -->

Chapter 10.1 uses (1.12) as its working equation and shows the code loop.

## 1.3.4 An unrestricted reference

With independent $\alpha$ and $\beta$ orbitals and energies, the sum splits
into three spin blocks, not two. Write $\bar\imath, \bar a$ for $\beta$
spatial orbitals.

$$ E_{\alpha\alpha} = -\frac14\sum_{ijab}\frac{[(ia|jb) - (ib|ja)]^2}{\Delta_{ij}^{ab}}, \qquad E_{\alpha\beta} = -\sum_{ia\bar\jmath\bar b}\frac{(ia|\bar\jmath\bar b)^2}{\Delta_{i\bar\jmath}^{a\bar b}}, \tag{1.13} $$

with $E_{\beta\beta}$ the same as $E_{\alpha\alpha}$ over the $\beta$ orbitals.
The $\alpha\beta$ block has no factor, by the count of section 1.3.3. When
$\alpha$ and $\beta$ orbitals coincide, $E_{\alpha\alpha} + E_{\beta\beta} = E_{\mathrm{SS}}$
and $E_{\alpha\beta} = E_{\mathrm{OS}}$.

> **Verified equation 1.13** · unrestricted MP2 spin blocks
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 2: independent $\alpha$ and $\beta$ orbitals give the spin-orbital $\alpha\alpha$ and $\alpha\beta$ parts, and $\alpha = \beta$ reduces to claim 1 |
> | Negative control | the same-spin kernel without its $\tfrac14$ must differ; the mutation sets the prefactor to $\tfrac12$ and turns the script red |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/u_rimp2.rs::same_spin_pair_kernel` |
>
> <!-- verified: id=so.ump2-blocks; script=scripts/mp2_variant_identities.py; claim=2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/u_rimp2.rs::same_spin_pair_kernel -->

The same blockwise reduction for third order, including the ring terms that
mix spins, is `scripts/ump3_spin_blocks_sympy.py` (chapter 10.9).

## 1.3.5 Closed-shell amplitudes

Closed-shell coupled-cluster codes store spatial amplitudes $t_{ij}^{ab}$ over
$n_o, n_v$, with $t_{ij}^{ab} = t_{ji}^{ba}$. Expanding to spin orbitals applies
the selection rule of section 1.3.1 to the amplitude:

$$ t_{i_\sigma j_\tau}^{a_\rho b_\kappa} = t_{ij}^{ab}\,\delta_{\sigma\rho}\delta_{\tau\kappa} - t_{ij}^{ba}\,\delta_{\sigma\kappa}\delta_{\tau\rho}, \qquad t_{i_\sigma}^{a_\rho} = t_i^a\,\delta_{\sigma\rho} . \tag{1.14} $$

`ferric-cc/src/ccsd_closed_shell.rs::expand_amplitudes_to_spin_orbital` is this
expansion. The $\alpha\beta\alpha\beta$ block of (1.14) is $t_{ij}^{ab}$ itself.
The correlation energy then reduces, by the same count as (1.12), to

$$ E_{\mathrm{c}} = \frac14\sum \langle ij\|ab\rangle\,t_{ij}^{ab}\Big|_{\mathrm{spin\ orbital}} = \sum_{ijab}(ia|jb)\,\big(2t_{ij}^{ab} - t_{ij}^{ba}\big) . \tag{1.15} $$

> **Verified equation 1.15** · closed-shell amplitudes and energy
>
> On random amplitudes with the closed-shell structure, the spin-orbital
> solution of the linearized LinLCCD equations has its $\alpha\beta\alpha\beta$
> block equal to the spatial solution, and the two energies agree. The
> expansion (1.14) is the one the closed-shell $(T)$ check builds as its
> spin-orbital reference.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`, claim 6 (the energy and the block identity) and claim 4 (the expansion (1.14), asserted antisymmetric in both index pairs) |
> | Negative control | claim 7: a hole–hole coefficient of 1 in the spin-orbital equations, and $(ij\vert kl)$ in place of $(ik\vert jl)$ in the spatial ones, must each fail |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/ccsd_closed_shell.rs::expand_amplitudes_to_spin_orbital` |
>
> <!-- verified: id=so.cc-amplitude-expansion; script=scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py; claim=6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/ccsd_closed_shell.rs::expand_amplitudes_to_spin_orbital -->

In the CCSD code the energy loop evaluates (1.15) with $t_{ij}^{ab}$
replaced by $\tau_{ij}^{ab} = t_{ij}^{ab} + t_i^a t_j^b$ and assumes
$f_{ia} = 0$. That singles term is not covered by claim 6, which is
doubles only. Chapter 16 takes it up.

## 1.3.6 The two spin-orbital orderings

The ordering of spin orbitals changes no energy and changes determinant
signs. The interleaved order ($2[p]+\sigma$) is used by the spin-orbital
integral code above. The blocked order is used by CAS-CI. The two Jordan–Wigner
Hamiltonians differ by a diagonal similarity of $\pm1$ entries.

> **Verified equation 1.16** · orderings differ by a sign similarity
>
> | | |
> |---|---|
> | Proof | `scripts/casci_slater_condon_phases_sympy.py`, claim 4, control e: the blocked-order Slater–Condon port mismatches the interleaved matrix element by element, the absolute values of the elements agree, and the spectra agree |
> | Negative control | the claim is itself the comparison against the other ordering; claim 3 holds only for the blocked order |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/hamiltonian.rs::general_element` |
>
> <!-- verified: id=so.ordering-similarity; script=scripts/casci_slater_condon_phases_sympy.py; claim=4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/hamiltonian.rs::general_element -->

## Known limits

- There is no open-shell CCSD or $(T)$ in the library. Every coupled-cluster
  solver reads a restricted reference (see the validation page).
- ROHF references use one spatial set with unequal occupations. The spin
  blocks above hold for them through the UHF functional at the ROHF
  densities (chapter 7), and are not restated here.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, "Conventions" and "2. Closed-shell RI-MP2" | the interleaved ordering and the closed-shell pair sums |
| `crates/ferric-cc/theory.md`, header and "CCD" | the spatial and spin-orbital conventions; Eq. (1.15) in the CCD section |
| `scripts/mp2_variant_identities.py`, `scripts/mp2_family_sympy.py` | verified equations 1.12 and 1.13 |
| `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py` | verified equation 1.15 |
| `scripts/casci_slater_condon_phases_sympy.py` | verified equation 1.16 |

Changes from the sources: denominators are the positive $\Delta$ with an
explicit minus sign, and the exchange partner is written $(ib|ja)$.
`crates/ferric-mp2/theory.md` writes the negative $D$ and the shorthand
$(ia|jb)^\times$.
