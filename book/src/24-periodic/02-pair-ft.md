# 24.2 Fourier transform of lattice-summed Gaussian pairs

<!-- section: 24.2; status: ready; registry: pbc-ewald-pairft-stress;
     theory: crates/ferric-pbc/theory.md#"2. Fourier transform of lattice-summed Gaussian pairs" -->

In a crystal, a basis function in the home cell overlaps every periodic copy
of its partner, so the charge density of an AO pair is a sum over lattice
translations. The long-range parts of the Coulomb problem are evaluated in
reciprocal space, so they need that density's Fourier coefficients at the
reciprocal vectors $\mathbf G$. For Gaussian basis functions the transform
has a closed form: a Gaussian in $\mathbf G$ times a phase times a short
polynomial. The Coulomb tensor of the dense builder, the long-range nuclear
attraction, the long-range part of the density fit, the $k$-point kernels and
the strain derivative all read this one object.

Section 24.2.1 derives the transform, 24.2.2 shows the kernel that evaluates
it, and 24.2.3 lists the anchors.

| Symbol | Meaning | Code name |
|---|---|---|
| $P_{\mu\nu}(\mathbf G)$ | pair transform, PySCF `ft_aopair` sign | `pair_ft` output |
| $\alpha, \beta$, $p = \alpha + \beta$ | primitive exponents | |
| $\mathbf P_c$ | product centre | |
| $E_t^{ij}$ | McMurchie–Davidson coefficients | `ferric_integrals::md3c1e::e_table` |

| Entry point | Location |
|---|---|
| Rust | `ferric-pbc/src/pair_ft.rs::{pair_ft,pair_ft_chunked}` |
| Plan and kernels | `crates/ferric-pbc/src/pair_ft/plan.rs` (`walk`, `site_terms`, `f_rows`) |
| $k$-point residues | `crates/ferric-pbc/src/pair_ft/residues.rs` |

## 24.2.1 The transform in closed form

**The definition.** Sum the partner over all translations and transform:

$$
P_{\mu\nu}(\mathbf G) = \sum_{\mathbf L}\int \phi_\mu(\mathbf r)\,\phi_\nu(\mathbf r - \mathbf L)\,
 e^{-i\mathbf G\cdot\mathbf r}\,d^3r .
$$

The sign of the exponent is PySCF's `ft_aopair`
(`ferric-pbc/src/pair_ft.rs::pair_ft`). Two symmetries follow from the
definition for real AOs. Conjugate the integrand: $P_{\mu\nu}(-\mathbf G) = P_{\mu\nu}(\mathbf G)^*$,
so the code sums over half of the $\mathbf G$ sphere with weight 2. Swap the
indices and shift $\mathbf r \to \mathbf r + \mathbf L$: the phase picks up
$e^{-i\mathbf G\cdot\mathbf L}$, which is 1 for a reciprocal-lattice vector,
so $P_{\nu\mu}(\mathbf G) = P_{\mu\nu}(\mathbf G)$ there. Off the lattice, at
the $\mathbf G + q$ of a $k$-point mesh (section 24.6), the second symmetry
does not hold.

**One primitive pair.** Take primitives of exponent $\alpha$ at $\mathbf A$
and $\beta$ at $\mathbf B' = \mathbf B + \mathbf L$, and let
$\mathbf P_c = (\alpha\mathbf A + \beta\mathbf B')/p$. Complete the square in
the exponent (the Gaussian product theorem):

$$
-\alpha|\mathbf r - \mathbf A|^2 - \beta|\mathbf r - \mathbf B'|^2
 = -p|\mathbf r - \mathbf P_c|^2 - \frac{\alpha\beta}{p}|\mathbf A - \mathbf B'|^2 .
$$

The product is one Gaussian at $\mathbf P_c$, scaled by the overlap Gaussian
$e^{-\alpha\beta|\mathbf A - \mathbf B'|^2/p}$, which depends only on the
distance between the two centres.

**Angular momentum.** The Cartesian prefactors $x_A^{i}x_B^{j}$ ride along.
The McMurchie–Davidson expansion writes them, times the product Gaussian, as
derivatives of that Gaussian with respect to its centre:

$$ x_A^{i}x_B^{j}e^{-\alpha x_A^2 - \beta x_B^2} = e^{-\alpha\beta X_{AB}^2/p}\sum_t E_t^{ij}\,\partial_{P_x}^t e^{-p(x-P_x)^2}, $$

with $x_A = x - A_x$, $X_{PA} = P_x - A_x$, $X_{AB} = A_x - B_x$ and the overlap Gaussian kept
outside the coefficients. Check the first order: $\partial_{P_x}$ of the
product Gaussian is $2p(x - P_x)$ times it, so
$X_{PA} + \tfrac{1}{2p}\cdot 2p(x - P_x) = x - A_x$. Low orders:
$E^{00}_0 = 1$; $E^{10} = (X_{PA},\,1/2p)$;
$E^{11} = (X_{PA}X_{PB} + 1/2p,\,(X_{PA}+X_{PB})/2p,\,1/4p^2)$;
$E^{20} = (X_{PA}^2 + 1/2p,\,X_{PA}/p,\,1/4p^2)$.

**Transform one dimension.** The Fourier transform of a centred Gaussian is
a Gaussian:

$$ \int e^{-p(x - P_x)^2}e^{-iG_xx}\,dx = \sqrt{\frac{\pi}{p}}\,e^{-G_x^2/4p}\,e^{-iG_xP_x} . $$

The centre $P_x$ appears only in the phase. A derivative with respect to it
therefore brings down $-iG_x$ and nothing else: under the transform,
$\partial_{P_d} \to (-iG_d)$. Apply this to every term of the
McMurchie–Davidson sum and multiply the three dimensions:

$$
P^\text{prim}(\mathbf G) = c_a c_b \Big(\frac{\pi}{p}\Big)^{3/2}
 e^{-\alpha\beta|\mathbf A - \mathbf B'|^2/p}\;
 e^{-G^2/4p}\; e^{-i\mathbf G\cdot\mathbf P_c}
 \prod_{d=x,y,z}\sum_t E_t^{i_d j_d}(-iG_d)^t .
$$

$P_{\mu\nu}(\mathbf G)$ is the sum of this over the primitives and over
$\mathbf L$.

Read the factors from left to right: contraction coefficients, the overlap
of two s primitives, a Gaussian decay in $G$ whose width is set by the
product exponent $p$, a phase that locates the product, and a polynomial in
$\mathbf G$ of degree $l_a + l_b$ per dimension. A tight pair (large $p$) has
a wide transform and needs a large $\mathbf G$ sphere; this is why the
reciprocal-space cost of the chapter grows with the steepest exponent.

**Check $\mathbf G = 0$.** The phase is 1 and only $E_0$ survives, so
$P_{\mu\nu}(0)$ is the overlap $\sum_{\mathbf L}\langle\mu_0|\nu_{\mathbf L}\rangle = S^\text{latt}_{\mu\nu}$.
For two s primitives in one image it is
$c_ac_b(\pi/p)^{3/2}e^{-\alpha\beta|\mathbf A - \mathbf B'|^2/p}$, the
textbook s–s overlap. The test below compares every element of $P(0)$ with
an independent libint2 lattice overlap.

> **Verified equation 24.3** · the McMurchie–Davidson pair transform
>
> The Gaussian product theorem holds as an exact exponent identity, and the
> expansion $\prod_d\sum_tE_t^{i_dj_d}(-iG_d)^t$ for
> $(l_a, l_b) = (0,0), (1,0), (1,1), (2,0)$ equals an mpmath quadrature of the
> complex one-dimensional Fourier integral.
>
> | | |
> |---|---|
> | Proof | `scripts/pbc_ewald_pairft_stress_sympy.py`, part S2, claim `pairft-md` |
> | Negative control | none in the script; the code tests compare with PySCF |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pbc/src/pair_ft.rs::pair_ft` |
> | Code test | `pair_ft_at_nonzero_g_matches_pyscf_ft_aopair` in `crates/ferric-pbc/tests/pbc_stage0.rs` |
>
> <!-- verified: id=pbc.pairft.md-expansion; script=scripts/pbc_ewald_pairft_stress_sympy.py; claim=pairft-md; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pbc/src/pair_ft.rs::pair_ft; rust_test=ferric-pbc/tests/pbc_stage0.rs::pair_ft_at_nonzero_g_matches_pyscf_ft_aopair; registry=pbc-ewald-pairft-stress -->

## 24.2.2 The algorithm

| Step | Quantity | Cost |
|---|---|---|
| 1 | pair images $\mathbf L$ whose overlap Gaussian $e^{-\alpha\beta|\mathbf A - \mathbf B'|^2/p}$ passes the screen | $N_\text{shell}^2N_{\mathbf L}$ |
| 2 | per pair and image: prefactor, $E_t$ tables, phase $e^{-i\mathbf G\cdot\mathbf P_c}$ | |
| 3 | per $\mathbf G$ chunk: rows $\sum_tE_t(-iG_d)^t$, accumulate into $P_{\mu\nu}(\mathbf G)$ | $N_\text{pair}N_{\mathbf L}N_{\mathbf G}$ |

The prefactor and the overlap Gaussian are formed in the plan's walk
(`pair_ft/plan.rs`, $c_ac_b(\pi/p)^{3/2}$ and
$e^{-\alpha\beta|\mathbf A - \mathbf B'|^2/p}$), the phase in `site_terms`.
The polynomial rows are `ferric-pbc/src/pair_ft/plan.rs::f_rows`. This is
its loop over dimensions:

<!-- excerpt: ferric-pbc/src/pair_ft/plan.rs::f_rows -->
```rust
    for d in 0..3 {
        e_table(la, lb, a, b, ab[d], &mut ebuf[d]);
        let (e, f, w) = (&ebuf[d], &mut fbuf[d], &ch.pw[d]);
        for ij in 0..nij {
            let frow = &mut f[ij * ng..ij * ng + n];
            frow.fill(zero);
            for t in 0..st {
                let et = e[ij * st + t];
                if et == 0.0 {
                    continue;
                }
                for (x, wv) in frow.iter_mut().zip(&w[t * ng..t * ng + n]) {
                    *x += *wv * et;
                }
            }
        }
    }
```

| Code | Equation |
|---|---|
| `d` | the dimension $x$, $y$ or $z$ of the product $\prod_d$ |
| `e_table(...)` | $E_t^{ij}$ for every $(i, j)$ with $i \le l_a$, $j \le l_b$ |
| `ij` | one Cartesian pair $(i_d, j_d)$ in this dimension |
| `w[t * ng + g]` | $(-iG_d)^t$ at the chunk's $\mathbf G_g$ (`ChunkG::pw`) |
| `frow[g]` | $\sum_tE_t^{i_dj_d}(-iG_d)^t$ |

- **The powers are precomputed.** `ChunkG` forms $(-iG_d)^t$ for
  $t < 2l_\text{max} + 1$ once per chunk, so the inner loop is a scaled
  vector add over the chunk's $\mathbf G$ vectors.
- **One row per dimension, not per Cartesian function.** A Cartesian product
  function needs one $x$ row, one $y$ row and one $z$ row, and a shell pair
  needs $(l_a + 1)(l_b + 1)$ rows per dimension. The product of the three
  rows is formed afterwards for each Cartesian pair (`accumulate_fused`).
- **Streamed in $\mathbf G$.** The transform is evaluated in $\mathbf G$
  chunks (`ferric-pbc/src/pair_ft.rs::pair_ft_chunked`), and no caller holds
  all $\mathbf G$ at once.

At a $k$-point mesh the images are binned by residue $\mathbf L \bmod N$,
and one residue-resolved transform per momentum-transfer class gives
$P^{k'}(\mathbf G + q)$ for every $k'$ by a phase sum (section 24.6).

## 24.2.3 Numerical anchors

| Quantity | Reference | Bar | Test |
|---|---|---|---|
| $P_{\mu\nu}(0)$, every element | an independent libint2 lattice overlap | $10^{-10}$ | `crates/ferric-pbc/tests/pbc_stage0.rs` `pair_ft_at_g0_equals_independent_libint_lattice_overlap` |
| $P_{\mu\nu}(\mathbf G)$ for s, p, pure d at lattice and off-lattice $\mathbf G$ | PySCF `ft_aopair` | $10^{-10}$; the test asserts the imaginary part is exercised | `pair_ft_at_nonzero_g_matches_pyscf_ft_aopair` |

## 24.2.4 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pbc/theory.md`, §2 | the transform and its coefficients |
| `scripts/pbc_ewald_pairft_stress_sympy.py`, part S2 | the proof |
| L. E. McMurchie and E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978) | the Hermite expansion |
