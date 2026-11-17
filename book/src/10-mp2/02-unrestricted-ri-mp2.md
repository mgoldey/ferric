# Unrestricted RI-MP2 {#unrestricted-ri-mp2}

<!-- status: ready; registry: u-ri-mp2 -->

A radical such as OH has one more $\alpha$ electron than $\beta$. Exchange
acts only between electrons of the same spin, so an $\alpha$ electron and a
$\beta$ electron feel different Fock operators, and UHF gives them
different spatial orbitals. Section @sec:closed-shell-ri-mp2 relied on one set of spatial
orbitals for both spins, so it does not apply. Unrestricted RI-MP2 (UMP2)
keeps the two sets apart and sums the correlation energy in three spin
blocks. The UHF determinant need not be an eigenfunction of $\hat S^2$,
and UMP2 inherits that spin contamination: MP2 "cannot be recommended for
open shell systems when there is significant spin contamination"
(dissertation, chapter 5, section 5.1).

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/u_rimp2.rs::u_ri_mp2` |
| CLI | `method.kind = "rimp2"` on a molecule with multiplicity > 1 (UHF reference) |
| Python | `run_rimp2` (an open-shell molecule runs UMP2; `result.reference` says which) |

Index convention: $i, j, a, b$ are $\alpha$ orbitals and
$\bar\imath, \bar\jmath, \bar a, \bar b$ are $\beta$ orbitals. The code names
the $\beta$ indices with capitals (`I`, `J`, `A`, `B`). The factors
$B^P_{ia}$ of (@eq:closed-shell-ri-mp2-4) are built once per spin, each from its own orbitals,
with one shared auxiliary metric.

## The energy {#unrestricted-ri-mp2-energy}

Start again from the spin-orbital energy (@eq:closed-shell-ri-mp2-1), now with per-spin
canonical orbitals. The operator $1/r_{12}$ does not act on spin, so
$\langle ij|ab\rangle = (ia|jb)$ is nonzero only when $i$ and $a$ have the
same spin, and $j$ and $b$ have the same spin. Split the sum by the spins of
the two occupied orbitals.

**Both $\alpha$.** Both virtuals must be $\alpha$ too, and both the direct
and the exchange integral survive. The antisymmetrized integral is

$$ K_{iajb} = (ia|jb) - (ib|ja) , $$

and the block keeps the $\tfrac14$ of (@eq:closed-shell-ri-mp2-1). The $\beta\beta$ block is the
same with every index barred.

**One $\alpha$, one $\beta$.** Take $i$ to be $\alpha$ and $\bar\jmath$ to
be $\beta$. The exchange integral $(i\bar b|\bar\jmath a)$ would need $i$
and $\bar b$ to share a spin, so it vanishes. The direct term survives when
the virtuals are $(a, \bar b)$. When they are $(\bar b, a)$, only the
exchange term survives, and it squares to the same number. So this ordering
of the occupied pair gives $\tfrac14 \cdot 2 \sum (ia|\bar\jmath\bar b)^2/\Delta$.
The ordering with the $\beta$ orbital first gives the same again, which
cancels the $\tfrac14$.

Collect the blocks:

$$ E^{(2)} = E_{\alpha\alpha} + E_{\beta\beta} + E_{\alpha\beta}, \label{eq:unrestricted-ri-mp2-1} $$

$$ E_{\alpha\alpha} = -\frac14 \sum_{ijab} \frac{\big[(ia|jb) - (ib|ja)\big]^2}{\Delta_{ij}^{ab}}, \qquad E_{\alpha\beta} = -\sum_{i\bar\jmath a\bar b} \frac{(ia|\bar\jmath\bar b)^2}{\Delta_{i\bar\jmath}^{a\bar b}} , \label{eq:mp2-u-spin-blocks} $$

and $E_{\beta\beta}$ is $E_{\alpha\alpha}$ with every index barred. Each
denominator uses the orbital energies of the spins it names:
$\Delta_{i\bar\jmath}^{a\bar b} = \varepsilon_a + \varepsilon_{\bar b} - \varepsilon_i - \varepsilon_{\bar\jmath}$.

The factor $\tfrac14$ in $E_{\alpha\alpha}$ counts each antisymmetrized
pair four times: once per ordering of $(i,j)$ and once per ordering of
$(a,b)$. The sum over $i<j$, $a<b$ without the $\tfrac14$ is the same
number. The opposite-spin block has no exchange integral and no double
counting.

Check the closed-shell limit. Give both spins the same orbitals. The
opposite-spin block is then $E_{\mathrm{OS}}$ of (@eq:mp2-cs-spin-adaptation) term by term. The
two same-spin blocks add to

$$ E_{\alpha\alpha} + E_{\beta\beta} = -\frac12 \sum_{ijab} \frac{(ia|jb)^2 - 2(ia|jb)(ib|ja) + (ib|ja)^2}{\Delta_{ij}^{ab}} . $$

Relabel $a \leftrightarrow b$ in the last term. The denominator does not
change, and $(ib|ja)^2$ becomes $(ia|jb)^2$. The bracket becomes
$2(ia|jb)^2 - 2(ia|jb)(ib|ja)$, and the sum is $E_{\mathrm{SS}}$ of (@eq:mp2-cs-spin-adaptation).
So the grouping

$$ E_{\mathrm{SS}} = E_{\alpha\alpha} + E_{\beta\beta}, \qquad E_{\mathrm{OS}} = E_{\alpha\beta} $$

carries over to open shells, and the SCS weights of section @sec:scs-and-kappa apply to
UMP2 unchanged.

> **Verified equation @eq:mp2-u-spin-blocks** · unrestricted MP2 spin blocks
>
> The three blocks of (@eq:mp2-u-spin-blocks) sum to the spin-orbital energy (@eq:closed-shell-ri-mp2-1), and the
> $\tfrac14$ sum over all $(i,j,a,b)$ of the squared antisymmetrized
> integral equals the sum over $i<j$, $a<b$.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 2; `scripts/mp2_family_sympy.py`, part B |
> | Negative control | the runner sets the same-spin kernel's scale to 1/2 and requires the claim to fail |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/u_rimp2.rs::same_spin_block_energy`, `ferric-mp2/src/u_rimp2.rs::opposite_spin_block_energy` |
>
> <!-- verified: id=mp2.u.spin-blocks; script=scripts/mp2_variant_identities.py; claim=2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/u_rimp2.rs::same_spin_block_energy, ferric-mp2/src/u_rimp2.rs::opposite_spin_block_energy; rust_test=ferric-mp2/tests/validation_u_rimp2.rs::u_rimp2_oh_ccpvdz_vs_pyscf_dfump2 -->

> **Verified equation @eq:mp2-u-spin-blocks** · the quarter factor
>
> $\tfrac14\sum_{ijab} K_{ijab}^2/\Delta$ with $K_{ijab} = (ia|jb) - (ib|ja)$
> equals $\sum_{i<j,\,a<b} |\langle ij\|ab\rangle|^2/\Delta$.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_family_sympy.py`, part B |
> | Negative control | the runner replaces the quarter by 1/2 and requires part B to fail |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/u_rimp2.rs::same_spin_block_energy` |
>
> <!-- verified: id=mp2.u.quarter-factor; script=scripts/mp2_family_sympy.py; claim=B; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/u_rimp2.rs::same_spin_block_energy -->

## How the code evaluates it {#unrestricted-ri-mp2-code-evaluates}

The same-spin kernel uses the symmetry of $K$ under the joint swap
$(i,a) \leftrightarrow (j,b)$:

$$ K_{jbia} = (jb|ia) - (ja|ib) = (ia|jb) - (ib|ja) = K_{iajb} , $$

using $(pq|rs) = (rs|pq)$ for real orbitals. The denominator is symmetric
under the same swap. So the $(j,i)$ block equals the $(i,j)$ block, as in
equation (@eq:mp2-cs-unique-pairs). The kernel loops over $j \ge i$ and weights the
off-diagonal pairs by 2, which halves the work. This is
`ferric-mp2/src/u_rimp2.rs::same_spin_block_energy` with the destructuring
of the spin channel removed and the comments replaced:

```rust
let eps_i = eps[first_occ + i];
let mut energy_i = 0.0;
for j in i..nocc {                                   // unique pairs j >= i
    let fac = if i == j { 1.0 } else { 2.0 };        // (j,i) block equals (i,j)
    let jcol = (j - i) * nvir;
    let eps_j = eps[first_occ + j];
    let mut energy_ij = 0.0;
    for a in 0..nvir {
        let eps_a = eps[nocc_total + a];
        for b_idx in 0..nvir {
            let eps_b = eps[nocc_total + b_idx];
            let g_ab = g_i[(a, jcol + b_idx)]; // (ia|jb)
            let g_ba = g_i[(b_idx, jcol + a)]; // (ib|ja)
            let k = g_ab - g_ba;                         // K_iajb
            let denom = eps_i + eps_j - eps_a - eps_b;   // -Delta
            let t_val = k / denom;                       // -K/Delta
            energy_ij += t_val * k;                      // -K^2/Delta
        }
    }
    energy_i += fac * energy_ij;
}
0.25 * energy_i                                      // the 1/4 of (10.8)
```

The block `g_i` is built exactly as in @sec:closed-shell-ri-mp2-algorithm, one wide product per $i$.
The sign convention is the one of 10.1: `denom` is $-\Delta$, and the
explicit minus of (@eq:mp2-u-spin-blocks) is absent.

`ferric-mp2/src/u_rimp2.rs::opposite_spin_block_energy` has no such
symmetry. Its two indices belong to different spins, $(i, \bar\jmath)$ and
$(\bar\jmath, i)$ are not the same pair, and the loop runs over every
$i$ and every $\bar\jmath$. Its block is
$G^{(i)}_{a,\bar\jmath\bar b} = \sum_P B^P_{ia} B^P_{\bar\jmath\bar b}$, a
product of an $\alpha$ factor and a $\beta$ factor.

| Step | What happens |
|---|---|
| Reference check | a restricted (RHF) result is an error; use section @sec:closed-shell-ri-mp2 |
| ROHF or ROKS input | semicanonicalized first: the occupied–occupied and virtual–virtual blocks of each spin's Fock matrix are diagonalized (`ferric_scf::semicanonical::unrestricted_reference`) |
| Frozen core | the same number of core orbitals is dropped in each spin |
| Memory | one admission check on the MO-side tensors of both spins before the intermediates are built |
| Output | `URiMp2Components` with the $\alpha\alpha$, $\beta\beta$ and $\alpha\beta$ energies |

Why semicanonicalize? The energy (@eq:mp2-u-spin-blocks) assumes that the occupied and the
virtual blocks of each spin's Fock matrix are diagonal, so that every
denominator is a difference of orbital energies. ROHF orbitals diagonalize
one effective Fock operator, not the separate $\alpha$ and $\beta$ Fock
matrices. Rotating within the occupied and within the
virtual space of each spin makes the two blocks diagonal and leaves the
reference determinant unchanged.

The rotation does not remove the occupied–virtual block. On an ROHF
reference the energy is the doubles part only. The singles term
$\sum_{ia} |f_{ia}|^2/(\varepsilon_i - \varepsilon_a)$, which is nonzero
for semicanonical ROHF orbitals, is not added. The CLI runs UMP2 on a UHF
reference; the ROHF route is reached from Rust.

The frozen-core count applies to both spins, so a count larger than
$n_o^\beta$ is an error even when it fits in $n_o^\alpha$.

## What is anchored {#unrestricted-ri-mp2-anchored}

| Quantity | Systems | Agreement | Bar | Test |
|---|---|---|---|---|
| UMP2 correlation energy vs PySCF `DFUMP2` | OH, CH3, NH2, O2, HO2 / cc-pVDZ | ≤ 9.2e-10 Ha (HO2), ≤ 5.8e-11 Ha elsewhere | 1e-8 Ha | `crates/ferric-mp2/tests/validation_u_rimp2.rs` |
| UHF energy, same runs | as above | ≤ 2.8e-12 Ha | 3e-11 Ha | as above |
| U-MP2 $E_c$ (doubles only) on a semicanonical ROHF reference vs numpy UMP2 | OH, CH3 / cc-pVDZ | ≤ 8.7e-12 Ha | 1e-10 Ha | `crates/ferric-gw/tests/validation_rohf_reference.rs` |

Figures from `site/src/reference/validation.md`.

## Known limits (2026-11-15)

- κ-regularization (section @sec:scs-and-kappa) is closed-shell only on the CLI and in
  Python; `[mp2] kappa` on an open-shell molecule is refused.
- No ROHF singles term.
- The local (amplitude-threshold) scheme of `[local]` is closed-shell only.

## Sources

| Source | Used for |
|---|---|
| sections @sec:unrestricted-ri-mp2-energy and @sec:attenuated-mp2 | equations @eq:unrestricted-ri-mp2-1, @eq:mp2-u-spin-blocks, the spin-block derivation |
| `crates/ferric-mp2/src/u_rimp2.rs` | kernels, reference handling, frozen core |
| `scripts/mp2_variant_identities.py` claim 2, `scripts/mp2_family_sympy.py` part B | verified boxes |
| Goldey dissertation, chapter 5, section 5.1 | the spin-contamination caveat |
| `site/src/reference/validation.md` | anchors |

Changes from the sources: barred $\beta$ indices instead of capitals; positive
$\Delta$ with an explicit minus sign.
