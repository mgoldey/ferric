# 6.4 Complementary auxiliary basis (CABS)

<!-- section: 6.4; status: edit; registry: none;
     theory: crates/ferric-mp2/theory.md#"11. CABS for F12" -->

Explicitly correlated (F12) methods resolve products of orbitals with the
correlation factor through a basis that is orthogonal to the orbital basis.
The complementary auxiliary basis is built from the union of the orbital
basis and an auxiliary set by projecting the orbital span out. ferric builds
CABS; it has no F12 energy.

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/f12.rs::{build_cabs,Cabs}` |
| Cross overlaps | `ferric-integrals/src/cabs.rs::cross_overlap` |

<!-- include: crates/ferric-mp2/theory.md#"11. CABS for F12" -->

> **Verified equation 6.5** · CABS from the Schur complement
>
> With $Q = S_{aa} - S_{aO}S_{OO}^{-1}S_{Oa} = VwV^{\mathsf T}$ and
> $c_k = w_k^{-1/2}[-S_{OO}^{-1}S_{Oa}v_k;\ v_k]$ for $w_k > 10^{-8}$,
> $c^{\mathsf T}S_{RR}c = I$ and $S_{O,R}c = 0$; the unprojected eigenvectors
> are neither orthogonal to the orbital basis nor orthonormal.
>
> | | |
> |---|---|
> | Proof | `scripts/mp2_variant_identities.py`, claim 7 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mp2/src/f12.rs::build_cabs` |
> | Code test | `ferric-mp2/src/f12.rs::cabs_is_orthonormal_and_complementary` |
>
> <!-- verified: id=df.cabs.schur; script=scripts/mp2_variant_identities.py; claim=7; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mp2/src/f12.rs::build_cabs; rust_test=ferric-mp2/src/f12.rs::cabs_is_orthonormal_and_complementary -->

## 6.4.1 The construction in the code

| Step | What | Detail |
|---|---|---|
| 1 | RIBS = OBS $\cup$ aux | atom-major: per atom, the orbital shells then the auxiliary shells, so the two sets interleave |
| 2 | $S$ over the union, then split by provenance mask | the blocks $S_{OO}$, $S_{Oa}$, $S_{aa}$ are taken in $[O\,\vert\,a]$ order |
| 3 | $Q$, its eigenvectors, keep $w_k > 10^{-8}$ | `CABS_NULL_THRESH`; drops the orbital-spanned directions and union dependencies |
| 4 | scatter $[-S_{OO}^{-1}S_{Oa}c_2;\ c_2]$ back to the interleaved order | coefficients over the union functions |

`crates/ferric-mp2/theory.md` writes the union as "ordered $[O\,|\,a]$"; the
functions are in fact interleaved by atom and reordered for the algebra in
step 2. The result is the same.

## 6.4.2 Numerical anchors

| What | Test | Bar |
|---|---|---|
| CABS orthonormal and orthogonal to the orbital basis | `ferric-mp2/src/f12.rs::cabs_is_orthonormal_and_complementary` | rounding |
| CABS dimension equals the complement's | `ferric-mp2/src/f12.rs::cabs_dimension_is_complement` | exact |
| the orbital basis as its own auxiliary set gives an empty CABS | `ferric-mp2/src/f12.rs::cabs_self_complement_is_empty` | exact |

## Sources

- E. F. Valeev, *Chem. Phys. Lett.* **395**, 190 (2004).
- W. Klopper and C. C. M. Samson, *J. Chem. Phys.* **116**, 6397 (2002).
