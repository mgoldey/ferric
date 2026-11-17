# Complementary auxiliary basis (CABS) {#cabs}

<!-- status: edit; registry: none;
     theory: crates/ferric-mp2/theory.md#"11. CABS for F12" -->

Two electrons repel as $1/r_{12}$, so the exact wavefunction has a cusp where
they meet. A product of smooth orbitals cannot form a cusp, and the
correlation energy converges slowly with the orbital basis. Explicitly
correlated (F12) methods add pair functions that contain a correlation factor
$f(r_{12})$ directly. Their integrals contain products of orbitals with that
factor, and those products are resolved through a one-particle basis larger
than the orbital basis.

That basis has to exclude what the orbital basis already spans. The
conventional doubles already describe excitations within the orbital basis,
and resolving the F12 pair functions through the same functions again would
count those excitations twice. The complementary auxiliary basis (CABS) is
the part of a large resolution-of-identity basis that is orthogonal to the
orbital basis. ferric builds it from the union of the orbital basis (OBS) and
an auxiliary set by projecting the orbital span out. ferric builds CABS; it
has no F12 energy.

The plan: project the auxiliary functions, orthonormalize what is left, and
read the code that does both.

| Entry point | Location |
|---|---|
| Rust | `ferric-mp2/src/f12.rs::{build_cabs,Cabs}` |
| Cross overlaps | `ferric-integrals/src/cabs.rs::cross_overlap` |

### The construction

Start from the union basis, RIBS = OBS $\cup$ aux. Write $O$ for the
orbital-basis functions and $a$ for the auxiliary functions, and split the
union overlap into blocks $S_{OO}$, $S_{Oa}$, $S_{aa}$.

**Project.** Remove from each auxiliary function its component in the
orbital span. The projector onto the span of $O$ in a non-orthogonal basis is
$|O\rangle S_{OO}^{-1}\langle O|$, so the projected functions are

$$ |a_\perp\rangle = |a\rangle - |O\rangle\,S_{OO}^{-1}S_{Oa} . $$

Their overlap is the Schur complement of the auxiliary block. Multiply out
$\langle a_\perp|a_\perp\rangle$; the two cross terms and the
$S_{aO}S_{OO}^{-1}S_{OO}S_{OO}^{-1}S_{Oa}$ term combine into one:

$$ Q = \langle a_\perp|a_\perp\rangle = S_{aa} - S_{aO}S_{OO}^{-1}S_{Oa} . $$

**Orthonormalize.** The projected functions overlap, and some combinations
of them vanish. Diagonalize $Q = VwV^{\mathsf T}$. A zero eigenvalue is a
combination of auxiliary functions that lies inside the orbital span, or a
linear dependency within the union; both are dropped. Each kept eigenvector
$v_k$, scaled by $w_k^{-1/2}$, is one CABS function. Its coefficients over
the union are the projection coefficients of $|a_\perp\rangle$ applied to
$v_k$:

$$ Q = S_{aa} - S_{aO}S_{OO}^{-1}S_{Oa} = V\,w\,V^{\mathsf T}, \qquad
c_k = \frac{1}{\sqrt{w_k}}\begin{pmatrix} -S_{OO}^{-1}S_{Oa}v_k \\ v_k \end{pmatrix}, \quad w_k > 10^{-8} \ (\texttt{CABS\_NULL\_THRESH}) , $$

written in the partitioned order $[O\,|\,a]$.

**Check the two properties.** Orthogonality to the orbital basis: the $O$
rows of the union overlap are $[S_{OO}\ S_{Oa}]$, and

$$ S_{O,R}\,c_k = \frac{1}{\sqrt{w_k}}\big(-S_{OO}S_{OO}^{-1}S_{Oa}v_k + S_{Oa}v_k\big) = 0 . $$

Orthonormality: $c_k^{\mathsf T}S_{RR}\,c_l$ is the overlap of the projected
combinations, $v_k^{\mathsf T}Qv_l/\sqrt{w_kw_l} = w_k\delta_{kl}/w_k =
\delta_{kl}$. So $c^{\mathsf T}S_{RR}\,c = I$ and $S_{O,R}\,c = 0$.

The $O$ rows are what make the first property hold. The eigenvector $v_k$
alone, placed in the auxiliary rows with zero $O$ rows, is neither orthogonal
to the orbital basis nor orthonormal.

**Check a limit.** Use the orbital basis as its own auxiliary set. Then
$S_{aa} = S_{Oa} = S_{OO}$, $Q = S_{OO} - S_{OO}S_{OO}^{-1}S_{OO} = 0$, every
eigenvalue is dropped, and CABS is empty. In general the number of kept
functions is the dimension of the complement, about $n_{\mathrm{aux}}$ when
the auxiliary set adds directions the orbital basis lacks.

> **Verified equation {#eq:df-cabs-schur}** · CABS from the Schur complement
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

## The construction in the code {#cabs-construction-code}

`ferric-mp2/src/f12.rs::build_cabs` forms the union with `union_basis_sets`:
per element, the orbital shells and then the auxiliary shells. The union
functions are therefore atom-major and interleaved. On each atom the orbital
functions come first and that atom's auxiliary functions follow, so neither
set is contiguous in the union. A provenance mask (`union_obs_mask`) splits
the union indices into the orbital set $O$ and the auxiliary set $a$, and the
overlap blocks are taken in that partition.

| Step | What | Detail |
|---|---|---|
| 1 | RIBS = OBS $\cup$ aux | atom-major: per atom, the orbital shells then the auxiliary shells, so the two sets interleave |
| 2 | $S$ over the union, then split by provenance mask | the blocks $S_{OO}$, $S_{Oa}$, $S_{aa}$ are taken in $[O\,\vert\,a]$ order |
| 3 | $Q$, its eigenvectors, keep $w_k > 10^{-8}$ | `CABS_NULL_THRESH`; drops the orbital-spanned directions and union dependencies |
| 4 | scatter $[-S_{OO}^{-1}S_{Oa}c_2;\ c_2]$ back to the interleaved order | coefficients over the union functions |

Here are steps 3 and 4, from the inverse of $S_{OO}$ to the scatter, with
the source comments and blank lines removed and line comments added:

```rust
let s_oo_inv = sym_inverse(&s_oo)?;                 // S_OO^{-1}, error if not PD
let ls12 = s_oo_inv.dot(&s_oa);                     // S_OO^{-1} S_Oa
let q = &s_aa - &s_oa.t().dot(&ls12);               // Q = S_aa - S_aO S_OO^{-1} S_Oa
let (q_evals, q_evecs) = q
    .eigh(UPLO::Upper)                              // Q = V w V^T
    .map_err(|e| FerricError::General(format!("CABS Schur-complement eigh failed: {e}")))?;
let keep: Vec<usize> = (0..naux)
    .filter(|&i| q_evals[i] > CABS_NULL_THRESH)     // w_k > 1e-8
    .collect();
// ... ncabs = keep.len(); c2[:, col] = v_k / sqrt(w_k) for each kept k ...
let c1 = ls12.dot(&c2);                             // S_OO^{-1} S_Oa v_k / sqrt(w_k)
let mut coeffs = Array2::zeros((nri, ncabs));       // union AO order
for col in 0..ncabs {
    for (a, &gi) in obs_idx.iter().enumerate() {
        coeffs[(gi, col)] = -c1[(a, col)];           // O rows: -S_OO^{-1} S_Oa v_k / sqrt(w_k)
    }
    for (b, &gi) in aux_idx.iter().enumerate() {
        coeffs[(gi, col)] = c2[(b, col)];            // a rows: v_k / sqrt(w_k)
    }
}
```

The loop that fills `c2` is summarized in the `// ...` comment.

Points about it:

- **$S_{OO}$ is inverted by eigendecomposition** (`sym_inverse`), and a
  non-positive eigenvalue is an error. $S_{OO}$ is the orbital-basis overlap,
  small and positive definite for a sound basis.
- **The algebra runs in $[O\,|\,a]$ order, the result is stored in union
  order.** `obs_idx` and `aux_idx` hold the union index of each orbital and
  auxiliary function, and the last loop scatters the rows of each $c_k$ back
  through them. `Cabs::coeffs` is $n_{\rm RI}\times n_{\rm CABS}$ over the
  union functions as `PreparedBasis` orders them, `Cabs::s_rr` is the union
  overlap in that order, and `Cabs::s_or` is its orbital rows.
- **The order does not matter for the result.** A permutation of the union
  functions permutes the rows of $c$ and the rows and columns of $S_{RR}$
  together, so it leaves both properties unchanged,
  $c^{\mathsf T}S_{RR}\,c = I$ and $S_{O,R}\,c = 0$ (`crates/ferric-mp2/theory.md`,
  check 7).

## Numerical anchors {#cabs-numerical-anchors}

| What | Test | Bar |
|---|---|---|
| CABS orthonormal and orthogonal to the orbital basis | `ferric-mp2/src/f12.rs::cabs_is_orthonormal_and_complementary` | rounding |
| CABS dimension equals the complement's | `ferric-mp2/src/f12.rs::cabs_dimension_is_complement` | exact |
| the orbital basis as its own auxiliary set gives an empty CABS | `ferric-mp2/src/f12.rs::cabs_self_complement_is_empty` | exact |

## Sources

- E. F. Valeev, *Chem. Phys. Lett.* **395**, 190 (2004).
- W. Klopper and C. C. M. Samson, *J. Chem. Phys.* **116**, 6397 (2002).
