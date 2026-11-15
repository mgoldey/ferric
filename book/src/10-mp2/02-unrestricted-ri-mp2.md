# 10.2 Unrestricted RI-MP2

<!-- section: 10.2; status: ready; registry: u-ri-mp2;
     theory: crates/ferric-mp2/theory.md#"3. Unrestricted RI-MP2" -->

Unrestricted RI-MP2 (UMP2) is the open-shell form of section 10.1. The
$\alpha$ and $\beta$ orbitals differ, so the energy splits into three spin
blocks instead of two spatial sums. The factors $B^P_{ia}$ of (10.4) are
built once per spin.

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/u_rimp2.rs::u_ri_mp2` |
| CLI | `method.kind = "rimp2"` on a molecule with multiplicity > 1 (UHF reference) |
| Python | `run_rimp2` (an open-shell molecule runs UMP2; `result.reference` says which) |

Index convention: $i, j, a, b$ are $\alpha$ orbitals and
$\bar\imath, \bar\jmath, \bar a, \bar b$ are $\beta$ orbitals. The code names
the $\beta$ indices with capitals (`I`, `J`, `A`, `B`).

## 10.2.1 The energy

With per-spin canonical orbitals,

$$ E^{(2)} = E_{\alpha\alpha} + E_{\beta\beta} + E_{\alpha\beta}, \tag{10.7} $$

$$ E_{\alpha\alpha} = -\frac14 \sum_{ijab} \frac{\big[(ia|jb) - (ib|ja)\big]^2}{\Delta_{ij}^{ab}}, \qquad E_{\alpha\beta} = -\sum_{i\bar\jmath a\bar b} \frac{(ia|\bar\jmath\bar b)^2}{\Delta_{i\bar\jmath}^{a\bar b}} , \tag{10.8} $$

and $E_{\beta\beta}$ is $E_{\alpha\alpha}$ with every index barred. The
same-spin part is $E_{\mathrm{SS}} = E_{\alpha\alpha} + E_{\beta\beta}$ and the
opposite-spin part is $E_{\mathrm{OS}} = E_{\alpha\beta}$, so SCS weights
(section 10.3) apply to UMP2 unchanged.

The factor $\tfrac14$ in $E_{\alpha\alpha}$ counts each antisymmetrized pair
four times: once per ordering of $(i,j)$ and once per ordering of $(a,b)$.
The opposite-spin block has no exchange integral and no double counting.

> **Verified equation 10.8** · unrestricted MP2 spin blocks
>
> The three blocks of (10.8) sum to the spin-orbital energy (10.1), and the
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

> **Verified equation 10.8** · the quarter factor
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

## 10.2.2 How the code evaluates it

The same-spin kernel uses the symmetry of $K$ under the joint swap
$(i,a) \leftrightarrow (j,b)$: $K_{jbia} = K_{iajb}$. It loops over $j \ge i$
and weights the off-diagonal pairs by 2, which halves the work. The
opposite-spin kernel has no such symmetry and runs over every
$(i, \bar\jmath)$.

| Step | What happens |
|---|---|
| Reference check | a restricted (RHF) result is an error; use section 10.1 |
| ROHF or ROKS input | semicanonicalized first: the occupied–occupied and virtual–virtual blocks of each spin's Fock matrix are diagonalized (`ferric_scf::semicanonical::unrestricted_reference`) |
| Frozen core | the same number of core orbitals is dropped in each spin |
| Memory | one admission check on the MO-side tensors of both spins before the intermediates are built |
| Output | `URiMp2Components` with the $\alpha\alpha$, $\beta\beta$ and $\alpha\beta$ energies |

On an ROHF reference the energy is the doubles part only. The singles term
$\sum_{ia} |f_{ia}|^2/(\varepsilon_i - \varepsilon_a)$, which is nonzero
for semicanonical ROHF orbitals, is not added. The CLI runs UMP2 on a UHF
reference; the ROHF route is reached from Rust.

A frozen-core count larger than $n_o^\beta$ is an error even when it fits
in $n_o^\alpha$.

## 10.2.3 What is anchored

| Quantity | Systems | Agreement | Bar | Test |
|---|---|---|---|---|
| UMP2 correlation energy vs PySCF `DFUMP2` | OH, CH3, NH2, O2, HO2 / cc-pVDZ | ≤ 9.2e-10 Ha (HO2), ≤ 5.8e-11 Ha elsewhere | 1e-8 Ha | `crates/ferric-mp2/tests/validation_u_rimp2.rs` |
| UHF energy, same runs | as above | ≤ 2.8e-12 Ha | 3e-11 Ha | as above |
| U-MP2 $E_c$ (doubles only) on a semicanonical ROHF reference vs numpy UMP2 | OH, CH3 / cc-pVDZ | ≤ 8.7e-12 Ha | 1e-10 Ha | `crates/ferric-gw/tests/validation_rohf_reference.rs` |

Figures from `site/src/reference/validation.md`.

## Known limits (2026-11-15)

- κ-regularization (section 10.3) is closed-shell only on the CLI and in
  Python; `[mp2] kappa` on an open-shell molecule is refused.
- No ROHF singles term.
- The local (amplitude-threshold) scheme of `[local]` is closed-shell only.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-mp2/theory.md`, "3. Unrestricted RI-MP2" | equations 10.7, 10.8 |
| `crates/ferric-mp2/src/u_rimp2.rs` | kernels, reference handling, frozen core |
| `scripts/mp2_variant_identities.py` claim 2, `scripts/mp2_family_sympy.py` part B | verified boxes |
| `site/src/reference/validation.md` | anchors |

Changes from the sources: barred $\beta$ indices instead of capitals; positive
$\Delta$ with an explicit minus sign.
