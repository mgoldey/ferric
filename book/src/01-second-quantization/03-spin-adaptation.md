# Spin adaptation: from spin orbitals to spatial orbitals {#spin-adaptation}

<!-- status: edit; registry: none -->

In a closed-shell molecule every occupied spatial orbital holds one
$\alpha$ and one $\beta$ electron, and the $\alpha$ and $\beta$ orbitals are
the same functions. A spin-orbital tensor ignores that. It doubles every
index, so a four-index tensor is 16 times larger than the spatial one, and
10 of its 16 spin blocks are zero by spin conservation. The other six are
built from the spatial direct and exchange integrals.

A derivation is cleanest in spin orbitals. A working equation is cheapest in
spatial orbitals, because the spin sums are done once, by hand, and the code
never stores a spin-orbital tensor. This section does the reduction for the
quantities every later chapter reuses:

1. the reference energy and the Fock matrix (@sec:spin-adaptation-closed-shell-reference);
2. the second-order energy, closed shell (@sec:spin-adaptation-closed-shell-second-order) and unrestricted (@sec:spin-adaptation-unrestricted-reference);
3. the closed-shell coupled-cluster amplitudes and energy (@sec:spin-adaptation-closed-shell-amplitudes);
4. what the two spin-orbital orderings in the code change (@sec:spin-adaptation-spin-orbital-orderings).

Every reduction applies one table, the selection rule of section @sec:spin-adaptation-integral-selection-rule.
Each identity has a verified box or says it has not.

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

## The integral selection rule {#spin-adaptation-integral-selection-rule}

The Coulomb operator does not act on spin. So in $\langle pq|rs\rangle$
electron 1 must have the same spin in $p$ and $r$, and electron 2 the same
spin in $q$ and $s$. Equation (@eq:hamiltonian-3) says this with deltas: an
antisymmetrized integral is a direct term, nonzero when the spins pair as
$(p,r)$ and $(q,s)$, minus an exchange term, nonzero when they pair as
$(p,s)$ and $(q,r)$. Between occupied pair $(i_\sigma, j_\tau)$ and virtual
pair $(a_{\sigma'}, b_{\tau'})$:

$$ \langle i_\sigma j_\tau\|a_{\sigma'} b_{\tau'}\rangle = (ia|jb)\,\delta_{\sigma\sigma'}\delta_{\tau\tau'} - (ib|ja)\,\delta_{\sigma\tau'}\delta_{\tau\sigma'} . $$

Go through the 16 spin assignments of $(i, j, a, b)$ and keep those where
at least one delta pair is satisfied:

| Spins of $(i, j, a, b)$ | Direct term | Exchange term | Value |
|---|---|---|---|
| $\alpha\alpha\alpha\alpha$ or $\beta\beta\beta\beta$ | yes | yes | $(ia\vert jb) - (ib\vert ja)$ |
| $\alpha\beta\alpha\beta$ or $\beta\alpha\beta\alpha$ | yes | no | $(ia\vert jb)$ |
| $\alpha\beta\beta\alpha$ or $\beta\alpha\alpha\beta$ | no | yes | $-(ib\vert ja)$ |
| any other | no | no | 0 |

Six of the 16 survive. When both electrons have the same spin, they are
indistinguishable in every respect, and the integral carries both the
Coulomb and the exchange term. When the spins differ, the spin labels tell
the electrons apart, and only one of the two terms can connect them.

Every spin reduction in this section is a sum over this table.

## A closed-shell reference {#spin-adaptation-closed-shell-reference}

A closed-shell reference occupies each of $n_o$ spatial orbitals with an
$\alpha$ and a $\beta$ electron in the same spatial function. The occupied
spin orbitals are $i_\alpha$ and $i_\beta$.

Apply the table to (@eq:so-ref-energy). The one-electron sum $\sum h_{ii}$ over spin
orbitals visits each spatial $h_{ii}$ twice, once per spin. In the
two-electron sum, $\langle i_\sigma j_\tau\|i_\sigma j_\tau\rangle = (ii|jj) -
(ij|ji)\,\delta_{\sigma\tau}$. Sum over the four spin pairs: the direct term
appears for all of them, $\sum_{\sigma\tau}1 = 4$, and the exchange term
only for the two with equal spins, $\sum_{\sigma\tau}\delta_{\sigma\tau} = 2$.
With the $\tfrac12$ in front, and $(ij|ji) = (ij|ij)$ for real orbitals,

$$ E_{\mathrm{ref}} - E_{\mathrm{nn}} = 2\sum_i h_{ii} + \sum_{ij}\big[\,2(ii|jj) - (ij|ij)\,\big] . \label{eq:so-closed-shell-energy} $$

Check one doubly occupied orbital, as in He or H$_2$ in a minimal basis.
Then $E_{\mathrm{ref}} - E_{\mathrm{nn}} = 2h_{11} + (11|11)$: two
electrons, each with its one-electron energy, and one Coulomb repulsion
between them. Of the $2(11|11)$, one $(11|11)$ is the repulsion between the
two electrons, counted from both sides with the $\tfrac12$, and the other
is each electron's repulsion with itself. The exchange term $-(11|11)$
removes that self-interaction.

> **Verified equation @eq:so-closed-shell-energy** · the closed-shell reference energy
>
> $\langle 0|\hat H|0\rangle$ expanded from spatial integrals with the spin
> selection rule of (@eq:hamiltonian-3) equals $2\sum_i h_{ii} + \sum_{ij}[2(ii|jj) - (ij|ij)]$,
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

### The Fock matrix

The Fock matrix of (@eq:hamiltonian-5) reduces the same way. Take $p$ and $q$ with the
same spin $\sigma$ (otherwise $f_{pq} = 0$) and sum over the occupied
$k_\tau$:

$$ f_{p_\sigma q_\sigma} = h_{pq} + \sum_k\sum_\tau\big[(pq|kk) - (pk|kq)\,\delta_{\sigma\tau}\big] = h_{pq} + 2\sum_k (pq|kk) - \sum_k (pk|qk) . $$

So $f_{pq} = \big(h + 2J - K\big)_{[p][q]}\,\delta_{\sigma(p)\sigma(q)}$,
with $J_{pq} = \sum_k (pq|kk)$ and $K_{pq} = \sum_k (pk|qk)$ over the occupied
spatial orbitals. An electron feels the Coulomb field of both electrons in
each occupied orbital, and exchange only with the one of its own spin.

In the AO basis and with the total density $D = 2 C_o C_o^{\mathsf T}$
this gives the $F = h + J[D] - \tfrac12 K[D]$ that the code builds (notation
chapter, section @sec:notation-density-matrices-fock-matrix). The factor 2 in $D$ makes $J[D] = 2J$ and $K[D] =
2K$, which is where the $\tfrac12$ on exchange comes from.

> **Verified equation {#eq:so-fock-closed-shell}** · the spin-orbital Fock matrix is the spatial one
>
> On a random model, the spin-orbital Fock matrix built from the
> antisymmetrized integrals (@eq:hamiltonian-3) equals $F\otimes 1_{\mathrm{spin}}$, with $F = h + 2J - K$.
>
> | | |
> |---|---|
> | Proof | `scripts/oomp2_mp3_identities_sympy.py`, the check labelled "claim B0" |
> | Negative control | the spatial model built as $h + 2J - 2K$ (mutation `fock_k_double`) must fail the check |
> | Runner | `crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py`, which applies the mutation |
> | Code | `ferric-scf/src/rhf.rs::RhfJk::build_fock` assembles $F = h + J[D] - \tfrac12 K[D]$ |
>
> <!-- verified: id=so.fock-closed-shell; script=scripts/oomp2_mp3_identities_sympy.py; claim=B0; runner=crates/ferric-python/tests/test_oomp2_mp3_identities_sympy.py; code=ferric-scf/src/rhf.rs::RhfJk::build_fock -->

### The energy as the code computes it

The SCF code never loops over orbitals to evaluate (@eq:so-closed-shell-energy). It uses the
trace form $E_{\mathrm{ref}} - E_{\mathrm{nn}} = \tfrac12\mathrm{tr}[D(h+F)]$.
To see that it is the same number, expand the trace with
$D = 2C_oC_o^{\mathsf T}$, which turns a trace over AOs into a sum over
occupied orbitals: $\mathrm{tr}[Dh] = 2\sum_i h_{ii}$ and
$\mathrm{tr}[DF] = 2\sum_i f_{ii} = 2\sum_i h_{ii} + 2\sum_{ij}[2(ii|jj) - (ij|ij)]$.
Half their sum is (@eq:so-closed-shell-energy). `ferric-scf/src/rhf.rs::RhfState::energy`
evaluates it in one line:

<!-- excerpt: ferric-scf/src/rhf.rs::RhfState::energy -->
```rust
let e_elec_no_xc: f64 = 0.5 * (d * &(&env.h + &*f)).sum();  // 1/2 tr[D (h + F)]
```

`d * &(...)` is an elementwise product, and `.sum()` adds every element.
For symmetric matrices that is the trace of the matrix product, at $n^2$
cost instead of $n^3$. The rest of the function adds terms this chapter
does not have (exchange–correlation, a Fock hook, solvent and polarization
terms) and $E_{\mathrm{nn}}$.

## The closed-shell second-order energy {#spin-adaptation-closed-shell-second-order}

Apply the table to (@eq:so-e2-wick). The sum runs over the 16 spin assignments of
$(i, j, a, b)$, and the six that survive split into two groups.

**Opposite spins in the occupied pair** ($\alpha\beta$ or $\beta\alpha$).
Four spin patterns survive. Two, $\alpha\beta\alpha\beta$ and
$\beta\alpha\beta\alpha$, have the value $(ia|jb)$. The other two,
$\alpha\beta\beta\alpha$ and $\beta\alpha\alpha\beta$, have the value
$-(ib|ja)$. Each enters (@eq:so-e2-wick) squared and with the factor $\tfrac14$:

$$ \frac14\sum_{ijab}\frac{2(ia|jb)^2 + 2(ib|ja)^2}{\Delta_{ij}^{ab}} = \sum_{ijab}\frac{(ia|jb)^2}{\Delta_{ij}^{ab}} . $$

The step uses a relabeling. Swap the dummy labels $a \leftrightarrow b$ in
the $(ib|ja)^2$ sum. $\Delta_{ij}^{ab}$ is symmetric in $a, b$, so that sum
equals the $(ia|jb)^2$ sum. The four patterns therefore cancel the
$\tfrac14$, and the opposite-spin part has no factor.

**Same spins** ($\alpha\alpha$ or $\beta\beta$). Two patterns survive, each
with the full antisymmetrized integral and the factor $\tfrac14$:

$$ \frac14\cdot 2\sum_{ijab}\frac{[(ia|jb) - (ib|ja)]^2}{\Delta_{ij}^{ab}} = \frac12\sum_{ijab}\frac{(ia|jb)^2 - 2(ia|jb)(ib|ja) + (ib|ja)^2}{\Delta_{ij}^{ab}} = \sum_{ijab}\frac{(ia|jb)\,[(ia|jb) - (ib|ja)]}{\Delta_{ij}^{ab}} . $$

The second step expands the square. The third relabels $a \leftrightarrow b$
in the last term, which turns $(ib|ja)^2$ into $(ia|jb)^2$, and divides by 2.
With the minus sign of (@eq:so-e2-wick):

$$ E_{\mathrm{OS}} = -\sum_{ijab}\frac{(ia|jb)^2}{\Delta_{ij}^{ab}}, \qquad E_{\mathrm{SS}} = -\sum_{ijab}\frac{(ia|jb)\,[(ia|jb) - (ib|ja)]}{\Delta_{ij}^{ab}}, \qquad E^{(2)} = E_{\mathrm{OS}} + E_{\mathrm{SS}} . \label{eq:so-mp2-spin-adaptation} $$

Check one doubly occupied orbital again. Then $i = j$, and
$(ia|ib) - (ib|ia) = 0$ for every $a, b$, so $E_{\mathrm{SS}} = 0$. That is
right: two electrons in one spatial orbital have opposite spins, and
there is no same-spin pair to correlate. All of the MP2 energy of He or of
H$_2$ in any basis is opposite-spin. For larger molecules section @sec:closed-shell-ri-mp2
shows the split on water: about three quarters of the correlation energy is
opposite-spin.

> **Verified equation @eq:so-mp2-spin-adaptation** · spin adaptation of the MP2 energy
>
> The spatial sums of (@eq:so-mp2-spin-adaptation) equal the opposite-spin and same-spin parts of
> the spin-orbital sum (@eq:so-e2-wick), and their sum equals (@eq:so-e2-wick). The unique-pair sum
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

### The unique-pair sum

Section @sec:closed-shell-ri-mp2 uses (@eq:so-mp2-spin-adaptation) as its working equation and shows the inner loop,
`pair_energy`, which evaluates one $(i,j)$ block. The loop that calls it,
in `ferric-mp2/src/rimp2.rs::spin_components_from_blocks`, visits
only $j \ge i$:

<!-- excerpt: ferric-mp2/src/rimp2.rs::spin_components_from_blocks -->
```rust
for j in i..nocc {
    // Symmetry weight: off-diagonal pairs stand in for their mirror.
    let fac = if i == j { 1.0 } else { 2.0 };
    let (e_os_ij, e_ss_ij) =
        pair_energy(&g_i.view(), i, j, nvir, eps, first_occ, nocc_total, kappa);
    e_os_i += fac * e_os_ij;   // weight 2 counts the (j, i) block
    e_ss_i += fac * e_ss_ij;
}
```

Why is the $(j,i)$ block equal to the $(i,j)$ block? Swap $i \leftrightarrow j$
and, at the same time, the dummy labels $a \leftrightarrow b$. Then
$(ia|jb) \to (jb|ia) = (ia|jb)$ and $(ib|ja) \to (ja|ib) = (ib|ja)$, by the
electron-exchange symmetry of the chemist integral, and $\Delta$ is
unchanged. Each block's sum over $a, b$ is therefore the same, and the
code does half the pair work. The second claim of the verified box above,
with its weight-1 negative control, pins this factor of 2.

## An unrestricted reference {#spin-adaptation-unrestricted-reference}

With independent $\alpha$ and $\beta$ orbitals and energies, the
$\alpha\alpha$ and $\beta\beta$ sums are no longer equal, and the sum
splits into three spin blocks, not two. Write $\bar\imath, \bar a$ for
$\beta$ spatial orbitals.

$$ E_{\alpha\alpha} = -\frac14\sum_{ijab}\frac{[(ia|jb) - (ib|ja)]^2}{\Delta_{ij}^{ab}}, \qquad E_{\alpha\beta} = -\sum_{ia\bar\jmath\bar b}\frac{(ia|\bar\jmath\bar b)^2}{\Delta_{i\bar\jmath}^{a\bar b}}, \label{eq:so-ump2-blocks} $$

with $E_{\beta\beta}$ the same as $E_{\alpha\alpha}$ over the $\beta$ orbitals.
The $\alpha\beta$ block has no factor, by the count of section @sec:spin-adaptation-closed-shell-second-order: the
four opposite-spin patterns of the spin-orbital sum each give the same
spatial sum. No exchange integral appears, because $(i\bar b|\bar\jmath a)$
would need electron 1 to change spin.

The same-spin block keeps its $\tfrac14$. In section @sec:spin-adaptation-closed-shell-second-order two equal
same-spin blocks were added and the square was expanded. Here each block
stands alone, with the full antisymmetrized integral squared. When $\alpha$
and $\beta$ orbitals coincide, $E_{\alpha\alpha} + E_{\beta\beta} = E_{\mathrm{SS}}$
and $E_{\alpha\beta} = E_{\mathrm{OS}}$.

The same-spin kernel `ferric-mp2/src/u_rimp2.rs::same_spin_pair_kernel`
is (@eq:so-ump2-blocks) term by term. This is its inner loop, from the branch that also
stores amplitudes, with the loop headers and orbital-energy lookups
removed:

<!-- excerpt: ferric-mp2/src/u_rimp2.rs::same_spin_pair_kernel -->
```rust
let g_ab = g_i[(a, j * nvir + b_idx)]; // (ia|jb)
let g_ba = g_i[(b_idx, j * nvir + a)]; // (ib|ja)
let k = g_ab - g_ba;                   // <ij||ab> for a same-spin pair
let denom = eps_i + eps_j - eps_a - eps_b;   // -Delta
let t_val = k / denom;                 // t = -<ij||ab> / Delta
energy_i += t_val * k;
t_i[(j, a, b_idx)] = t_val;
// ... after the j, a, b loops:
*energy_i_slot = 0.25 * energy_i;      // the same-spin 1/4
```

The last line is the $\tfrac14$ of $E_{\alpha\alpha}$ in (@eq:so-ump2-blocks).

The opposite-spin kernel, `opposite_spin_pair_kernel`, has the same shape
with `k` replaced by the single integral $(ia|\bar\jmath\bar b)$ and no
$\tfrac14$. Its $i$ and $\bar\jmath$ come from different orbital sets with
different sizes, so the two spins carry separate `nocc` and `nvir`.

> **Verified equation @eq:so-ump2-blocks** · unrestricted MP2 spin blocks
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
mix spins, is `scripts/ump3_spin_blocks_sympy.py` (section @sec:mp3).

## Closed-shell amplitudes {#spin-adaptation-closed-shell-amplitudes}

Closed-shell coupled-cluster codes store spatial amplitudes $t_{ij}^{ab}$ over
$n_o, n_v$, with $t_{ij}^{ab} = t_{ji}^{ba}$. That symmetry says pair $(i,j)$
going to $(a,b)$ is the same excitation as pair $(j,i)$ going to $(b,a)$:
only the order in which the two electrons are named changed. Expanding to
spin orbitals applies the selection rule of section @sec:spin-adaptation-integral-selection-rule to the
amplitude:

$$ t_{i_\sigma j_\tau}^{a_{\sigma'} b_{\tau'}} = t_{ij}^{ab}\,\delta_{\sigma\sigma'}\delta_{\tau\tau'} - t_{ij}^{ba}\,\delta_{\sigma\tau'}\delta_{\tau\sigma'}, \qquad t_{i_\sigma}^{a_{\sigma'}} = t_i^a\,\delta_{\sigma\sigma'} . \label{eq:spin-adaptation-4} $$

`ferric-cc/src/ccsd_closed_shell.rs::expand_amplitudes_to_spin_orbital` is this
expansion. Its doubles loop, with the loop headers removed:

<!-- excerpt: ferric-cc/src/ccsd_closed_shell.rs::expand_amplitudes_to_spin_orbital -->
```rust
let mut v = 0.0;
if spin(i) == spin(a) && spin(j) == spin(b) {
    v += t2[[spat(i), spat(j), spat(a), spat(b)]];   // + t_ij^ab (direct spins)
}
if spin(i) == spin(b) && spin(j) == spin(a) {
    v -= t2[[spat(i), spat(j), spat(b), spat(a)]];   // - t_ij^ba (crossed spins)
}
t2_so[[i, j, a, b]] = v;
```

It is `asym_oovv` of section @sec:hamiltonian with $t$ in place of $(ia|jb)$, and the
same `spin`/`spat` convention. The $\alpha\beta\alpha\beta$ block of (@eq:spin-adaptation-4)
is $t_{ij}^{ab}$ itself: with $\sigma = \sigma' = \alpha$ and
$\tau = \tau' = \beta$ only the first delta pair holds.

The correlation energy is $\tfrac14\sum\langle ij\|ab\rangle t_{ij}^{ab}$ in
spin orbitals. Substitute the table for the integral and (@eq:spin-adaptation-4) for the
amplitude, and count as in section @sec:spin-adaptation-closed-shell-second-order:

- The four opposite-spin patterns each give $\tfrac14\sum(ia|jb)\,t_{ij}^{ab}$.
  For the two crossed patterns the integral is $-(ib|ja)$ and the amplitude
  $-t_{ij}^{ba}$; the signs cancel, and relabeling $a \leftrightarrow b$
  gives the same sum. Together: $\sum(ia|jb)\,t_{ij}^{ab}$.
- The two same-spin patterns each give
  $\tfrac14\sum[(ia|jb) - (ib|ja)][t_{ij}^{ab} - t_{ij}^{ba}]$. Expand the
  product and relabel $a \leftrightarrow b$ in the two terms that carry
  $(ib|ja)$. Together: $\sum(ia|jb)\,(t_{ij}^{ab} - t_{ij}^{ba})$.

Adding the two groups,

$$ E_{\mathrm{c}} = \frac14\sum \langle ij\|ab\rangle\,t_{ij}^{ab}\Big|_{\mathrm{spin\ orbital}} = \sum_{ijab}(ia|jb)\,\big(2t_{ij}^{ab} - t_{ij}^{ba}\big) . \label{eq:so-cc-amplitude-expansion} $$

Check the MP2 limit. Put in the first-order amplitudes
$t_{ij}^{ab} = -(ia|jb)/\Delta_{ij}^{ab}$; then
$t_{ij}^{ba} = -(ib|ja)/\Delta_{ij}^{ab}$, and (@eq:so-cc-amplitude-expansion) becomes
$-\sum(ia|jb)[2(ia|jb) - (ib|ja)]/\Delta$, which is
$E_{\mathrm{OS}} + E_{\mathrm{SS}}$ of (@eq:so-mp2-spin-adaptation). The combination
$2t^{ab} - t^{ba}$ is the closed-shell form of "direct minus exchange", and
it is the same $2(ia|jb) - (ib|ja)$ pattern as in section @sec:closed-shell-ri-mp2.

> **Verified equation @eq:so-cc-amplitude-expansion** · closed-shell amplitudes and energy
>
> On random amplitudes with the closed-shell structure, the spin-orbital
> solution of the linearized LinLCCD equations has its $\alpha\beta\alpha\beta$
> block equal to the spatial solution, and the two energies agree. The
> expansion (@eq:spin-adaptation-4) is the one the closed-shell $(T)$ check builds as its
> spin-orbital reference.
>
> | | |
> |---|---|
> | Proof | `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py`, claim 6 (the energy and the block identity) and claim 4 (the expansion (@eq:spin-adaptation-4), asserted antisymmetric in both index pairs) |
> | Negative control | claim 7: a hole–hole coefficient of 1 in the spin-orbital equations, and $(ij\vert kl)$ in place of $(ik\vert jl)$ in the spatial ones, must each fail |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-cc/src/ccsd_closed_shell.rs::expand_amplitudes_to_spin_orbital` |
>
> <!-- verified: id=so.cc-amplitude-expansion; script=scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py; claim=6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-cc/src/ccsd_closed_shell.rs::expand_amplitudes_to_spin_orbital -->

In the CCSD code the energy loop evaluates (@eq:so-cc-amplitude-expansion) with $t_{ij}^{ab}$
replaced by $\tau_{ij}^{ab} = t_{ij}^{ab} + t_i^a t_j^b$ and assumes
$f_{ia} = 0$. That singles term is not covered by claim 6, which is
doubles only. Chapter @sec:coupled-cluster takes it up.

Why does the code expand at all, if the closed-shell solver never needs
spin orbitals? The spin-orbital $(T)$ code, `ferric-cc/src/ccsd_t.rs::ccsd_t`,
takes spin-orbital amplitudes. Expanding the closed-shell solution with
(@eq:spin-adaptation-4) lets the spin-adapted CCSD feed that $(T)$ unchanged, and gives the
spin-adapted $(T)$ a spin-orbital reference to be compared against
(chapter @sec:coupled-cluster). The expansion costs $O((2n_o\cdot 2n_v)^2)$ in memory and
time, small next to the triples that consume it.

## The two spin-orbital orderings {#spin-adaptation-spin-orbital-orderings}

The ordering of spin orbitals changes no energy and changes determinant
signs. The interleaved order ($2[p]+\sigma$) is used by the spin-orbital
integral code above. The blocked order is used by CAS-CI. The two Jordan–Wigner
Hamiltonians differ by a diagonal similarity of $\pm1$ entries.

Where does the sign come from? A determinant is a product of the same
creators in either convention, applied in ascending mode order. Converting
one order to the other is a permutation of those creators, and each swap
of two creators costs $-1$. Two $\alpha$ creators, or two $\beta$ creators,
keep their relative order in both conventions. An $\alpha$ creator at
$q$ and a $\beta$ creator at $p$ swap places exactly when $p < q$: in the
blocked order every $\alpha$ comes first, while in the interleaved order
$p_\beta$ (mode $2p+1$) comes before $q_\alpha$ (mode $2q$). So a
determinant with $\alpha$ set $A$ and $\beta$ set $B$ changes sign by

$$ S_{AB} = (-1)^{\#\{(p, q)\,:\,p \in B,\ q \in A,\ p < q\}} , $$

and every matrix element changes as $H_{IJ} \to S_I H_{IJ} S_J$. The
spectrum is unchanged. A CI coefficient changes as $c_I \to S_I c_I$, and
a single off-diagonal matrix element can change sign, so a comparison of
CI vectors or matrix elements between two codes has to use one convention
for both.

> **Verified equation {#eq:so-ordering-similarity}** · orderings differ by a sign similarity
>
> | | |
> |---|---|
> | Proof | `scripts/casci_slater_condon_phases_sympy.py`, claim 4, control e: the blocked-order Slater–Condon port mismatches the interleaved matrix element by element, the blocked matrix equals $S H S$ of the interleaved one with $S_{AB}$ in the explicit form above, and the spectra agree |
> | Negative control | the two matrices compared without $S_{AB}$ differ; claim 3 holds only for the blocked order |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-ci/src/hamiltonian.rs::general_element` |
>
> <!-- verified: id=so.ordering-similarity; script=scripts/casci_slater_condon_phases_sympy.py; claim=4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-ci/src/hamiltonian.rs::general_element -->

## Known limits

Checked 2026-11-19.

- There is no open-shell CCSD or $(T)$ in the library. Every coupled-cluster
  solver reads a restricted reference (see the validation page).
- ROHF references are semi-canonicalized per spin on entry, after which
  (@eq:so-ump2-blocks) applies. The single-excitation term
  $-\sum_{ia}|f^\sigma_{ia}|^2/\Delta_i^a$, nonzero for ROHF, is not included
  (`ferric-mp2/src/u_rimp2.rs`, module documentation).

## Sources

| Source | Used for |
|---|---|
| chapter @sec:coupled-cluster and section @sec:ccd-ccsd-spin-orbital-equations | the spatial and spin-orbital conventions; Eq. (@eq:so-cc-amplitude-expansion) in the CCD equations |
| `scripts/mp2_variant_identities.py`, `scripts/mp2_family_sympy.py` | verified equations @eq:so-mp2-spin-adaptation and @eq:so-ump2-blocks |
| `scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py` | verified equation @eq:so-cc-amplitude-expansion |
| `scripts/casci_slater_condon_phases_sympy.py` | verified equation @eq:so-ordering-similarity |

Changes from the sources: denominators are the positive $\Delta$ with an
explicit minus sign, and the exchange partner is written $(ib|ja)$.
