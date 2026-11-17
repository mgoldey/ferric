# 9.3 $V_{xc}$ matrix elements: LDA, GGA, meta-GGA

<!-- section: 9.3; status: ready; registry: ks-semilocal, ks-mgga;
     theory: crates/ferric-dft/theory.md#"Numerical Integration" "Semilocal $V_{xc}$ matrix elements" -->

A semilocal functional looks at the electrons one point at a time. An LDA
sees only the density there. A GGA also sees how fast the density changes,
and a meta-GGA sees the kinetic energy of the orbitals at that point. The
SCF needs the matrix that tells each pair of basis functions how the energy
changes when the density matrix does. This section derives that matrix
directly from the grid energy, so the Fock matrix is the exact derivative of
the energy the SCF reports, whatever the grid. The derivation needs the
density and its first derivatives on the grid, and no second derivatives.

The plan: put the density and its derivatives on the grid (9.3.1),
differentiate the grid energy with respect to $D$ (9.3.2), then read the
batched loop that builds the result (9.3.3).

| Symbol | Meaning | Code name |
|---|---|---|
| $\chi_\mu(\mathbf r_g)$, $\partial_b\chi_\mu$ | AO values and gradients on the grid | `chi`, `dchi` |
| $\rho$, $\nabla\rho$, $\sigma$, $\tau$ | density, gradient, $\lvert\nabla\rho\rvert^2$, kinetic-energy density | `rho`, `grad`, `sigma`, `tau` |
| $\epsilon$ | XC energy per particle, from libxc | `exc` |
| $v_\rho, v_\sigma, v_\tau$ | $\partial(\rho\epsilon)/\partial(\rho, \sigma, \tau)$, from libxc | `vrho`, `vsigma`, `vtau` |
| $\mathbf G^\sigma$ | polarized gradient coefficient | |

| Entry point | Location |
|---|---|
| Rust, production | `ferric-dft/src/xc_batch.rs` behind `ferric-dft/src/ks.rs::KsXc` |
| Rust, dense reference | `ferric-dft/src/vxc.rs::{semilocal_vxc_closed,semilocal_vxc_polarized}`, `ferric-dft/src/density_on_grid.rs::{eval_density_closed,eval_tau_closed}` |
| Python | `AoIntegrals.veff`, the XC blocks tested in `crates/ferric-python/tests/test_xc_blocks.py` |

## 9.3.1 Grid quantities

The XC energy is a quadrature over the grid points $\{\mathbf r_g, w_g\}$ of
section 9.2:

$$ E_{xc} = \int \rho\,\epsilon_{xc}(\rho, \sigma, \tau)\,d\mathbf{r} \approx \sum_g w_g\,\rho_g\,\epsilon_{xc}(\rho_g, \sigma_g, \tau_g) . $$

For a closed shell with total density matrix $D$ and real AOs $\chi_\mu$,
the three variables are

$$ \rho = \sum_{\mu\nu} D_{\mu\nu}\chi_\mu\chi_\nu, \qquad \sigma = |\nabla\rho|^2, \qquad \tau = \tfrac12\sum_b\sum_{\mu\nu} D_{\mu\nu}\,\partial_b\chi_\mu\,\partial_b\chi_\nu . $$

Substitute $D = 2\mathbf C_o\mathbf C_o^{\mathsf T}$ into $\tau$ and the AO
sums collapse onto orbitals: $\tau = \sum_i\lvert\nabla\phi_i\rvert^2$ over
occupied spatial orbitals, which is $\tfrac12\sum_i n_i\lvert\nabla\phi_i\rvert^2$
with occupation $n_i = 2$. $\tau$ is the kinetic energy density of the
Kohn–Sham orbitals. Building it from $D$ needs no MO coefficients:
`ferric-dft/src/density_on_grid.rs::eval_tau_closed` does that, and
`density_on_grid::tests::tau_from_density_matrix_matches_explicit_mo_sum`
checks it against the explicit MO sum.

A one-orbital check shows what $\tau$ adds. With one doubly occupied orbital
$\phi$, $\rho = 2\phi^2$ and $\nabla\rho = 4\phi\nabla\phi$, so
$\sigma = 16\phi^2\lvert\nabla\phi\rvert^2$ and

$$ \tau = \lvert\nabla\phi\rvert^2 = \frac{\sigma}{8\rho} . $$

That is the von Weizsäcker value $\tau_W$. One orbital is fixed by its
density, so $\tau$ carries nothing new there. With several orbitals,
$\tau \ge \tau_W$ (the Cauchy–Schwarz inequality on
$\nabla\rho = 4\sum_i\phi_i\nabla\phi_i$), so a meta-GGA can tell a
one-orbital region from one where several orbitals overlap. ferric uses no
density Laplacian: meta-GGAs are evaluated $\tau$-only.

**Spin-polarized packing.** Polarized quantities follow libxc's interleaved
order: $\rho = (\rho_\alpha, \rho_\beta)$,
$\sigma = (\sigma_{\alpha\alpha}, \sigma_{\alpha\beta}, \sigma_{\beta\beta})$
with $\sigma_{\alpha\beta} = \nabla\rho_\alpha\cdot\nabla\rho_\beta$, and
$\tau = (\tau_\alpha, \tau_\beta)$ with
$\tau_\sigma = \tfrac12\sum_b\sum D^\sigma_{\mu\nu}\partial_b\chi_\mu\partial_b\chi_\nu$.

**What libxc returns.** Per point, libxc returns
$v_\rho = \partial(\rho\epsilon)/\partial\rho$,
$v_\sigma = \partial(\rho\epsilon)/\partial\sigma$ and
$v_\tau = \partial(\rho\epsilon)/\partial\tau$. $v_\sigma$ is the derivative
with respect to $\sigma$, not $2\times$ it; the factor 2 in the GGA term
below comes from the chain rule.

## 9.3.2 The matrix elements

Write $e = \rho\epsilon$ so that $E_{xc} = \sum_g w_g\,e(\rho_g, \sigma_g, \tau_g)$.
The grid weights do not depend on $D$, so the derivative is the chain rule
at each point:

$$ \frac{\partial E_{xc}}{\partial D_{\mu\nu}} = \sum_g w_g\Bigl[v_\rho\frac{\partial\rho_g}{\partial D_{\mu\nu}} + v_\sigma\frac{\partial\sigma_g}{\partial D_{\mu\nu}} + v_\tau\frac{\partial\tau_g}{\partial D_{\mu\nu}}\Bigr] . $$

Treat $D_{\mu\nu}$ and $D_{\nu\mu}$ as independent variables and compute
the three partials.

- $\rho$ is linear in $D$: $\partial\rho/\partial D_{\mu\nu} = \chi_\mu\chi_\nu$.
- $\nabla\rho = \sum D_{\kappa\lambda}\nabla(\chi_\kappa\chi_\lambda)$ is linear
  too, with $\partial\nabla\rho/\partial D_{\mu\nu} = \chi_\mu\nabla\chi_\nu + \chi_\nu\nabla\chi_\mu$.
  $\sigma = \nabla\rho\cdot\nabla\rho$ is quadratic in $\nabla\rho$, so the
  chain rule brings down a 2:
  $\partial\sigma/\partial D_{\mu\nu} = 2\nabla\rho\cdot(\chi_\mu\nabla\chi_\nu + \chi_\nu\nabla\chi_\mu)$.
- $\tau$ is linear in $D$, and the $\tfrac12$ of its definition comes along:
  $\partial\tau/\partial D_{\mu\nu} = \tfrac12\sum_b\partial_b\chi_\mu\,\partial_b\chi_\nu$.

Substitute:

$$ \frac{\partial E_{xc}}{\partial D_{\mu\nu}} = \sum_g w_g\Bigl[v_\rho\,\chi_\mu\chi_\nu + 2v_\sigma\,\nabla\rho\cdot(\chi_\mu\nabla\chi_\nu + \chi_\nu\nabla\chi_\mu) + \tfrac12 v_\tau \sum_b \partial_b\chi_\mu\,\partial_b\chi_\nu\Bigr] . $$

An LDA has $v_\sigma = v_\tau = 0$ and keeps only the first term; a GGA has
$v_\tau = 0$. In a uniform density $\nabla\rho = 0$, so the GGA term
vanishes and a GGA's matrix has the LDA form.

Where does the $\tfrac12$ of $\tau$ go? In $V_{xc}$ it stays: each matrix
element carries $\tfrac12v_\tau$. In the nuclear gradient (section 14.3) it
cancels: moving a nucleus differentiates both $\partial_b\chi_\mu$ and
$\partial_b\chi_\nu$, which for symmetric $D$ gives two equal terms, and the
2 cancels the $\tfrac12$.

**The trace against $D$.** Contract the result with $D$:
$\sum D_{\mu\nu}\chi_\mu\chi_\nu = \rho$,
$\sum D_{\mu\nu}\nabla(\chi_\mu\chi_\nu) = \nabla\rho$ and
$\sum D_{\mu\nu}\tfrac12\nabla\chi_\mu\cdot\nabla\chi_\nu = \tau$, so

$$ \mathrm{Tr}[DV_{xc}] = \sum_g w_g\,(\rho v_\rho + 2\sigma v_\sigma + \tau v_\tau) . $$

This is the identity behind section 9.1.2's statement that
$\mathrm{Tr}[DV_{xc}] \ne E_{xc}$ in general.

**Spin-polarized.** The energy is a function of seven variables. The same
chain rule on $D^\alpha$ gives

$$ \frac{\partial E_{xc}}{\partial D^\alpha_{\mu\nu}} = \sum_g w_g\Bigl[v_{\rho_\alpha}\chi_\mu\chi_\nu + \mathbf{G}^\alpha\cdot(\chi_\mu\nabla\chi_\nu+\chi_\nu\nabla\chi_\mu) + \tfrac12 v_{\tau_\alpha}\sum_b\partial_b\chi_\mu\partial_b\chi_\nu\Bigr], \qquad \mathbf{G}^\alpha = 2v_{\sigma_{\alpha\alpha}}\nabla\rho_\alpha + v_{\sigma_{\alpha\beta}}\nabla\rho_\beta, $$

and the same with $\alpha\leftrightarrow\beta$. $\mathbf G^\alpha$ is
$\partial e/\partial\nabla\rho_\alpha$. The coefficient of
$v_{\sigma_{\alpha\alpha}}$ is 2 because $\sigma_{\alpha\alpha}$ is quadratic
in $\nabla\rho_\alpha$. The cross coefficient is 1 because
$\sigma_{\alpha\beta}$ is linear in it:
$\partial\sigma_{\alpha\beta}/\partial\nabla\rho_\alpha = \nabla\rho_\beta$.
At $\rho_\alpha = \rho_\beta$ the polarized matrix reduces to the closed-shell
one (`test_polarized_equals_closed_at_equal_spin_densities`).

**The matrix element and the potential.** The textbook Kohn–Sham potential
is the functional derivative $\delta E_{xc}/\delta\rho$, a function of
position. Vary $\rho$ by $\delta\rho$. Then $\delta\sigma = 2\nabla\rho\cdot\nabla\delta\rho$
and

$$ \delta E_{xc} = \int\bigl(v_\rho\,\delta\rho + 2v_\sigma\nabla\rho\cdot\nabla\delta\rho\bigr)\,d\mathbf r . $$

Integrate the second term by parts. The boundary term vanishes for a
decaying density, and

$$ v_{xc} = \frac{\delta E_{xc}}{\delta\rho} = v_\rho - 2\nabla\cdot(v_\sigma\nabla\rho) . $$

This potential needs the second derivatives of $\rho$. The matrix-element
form above does not. The two differ by a pure divergence. By the product
rule,

$$ v_\rho\chi_\mu\chi_\nu + 2v_\sigma\nabla\rho\cdot\nabla(\chi_\mu\chi_\nu) - v_{xc}\chi_\mu\chi_\nu = \nabla\cdot\bigl(2v_\sigma\nabla\rho\,\chi_\mu\chi_\nu\bigr), $$

which integrates to zero for decaying AOs. So both define the same Fock
matrix, and the code uses the left form.

> **Verified equation 9.4** · $V_{xc} = \partial E_{xc}/\partial D$, closed shell and polarized
>
> For a generic $e(\rho, \sigma, \tau)$ (closed shell) and a generic
> seven-variable polarized $e$, the code's matrix elements, with the GGA
> factor 2, the cross coefficient 1 on $v_{\sigma_{\alpha\beta}}$ and the
> $\tfrac12$ on $v_\tau$, equal $\partial E/\partial D_{\mu\nu}$
> symbolically; on a 4-point, 3-AO model with a concrete meta-GGA-like $e$,
> a central difference of $E_{xc}(D)$ equals $\mathrm{Tr}[V X]$ within its
> derived truncation bar; the matrix-element and potential forms differ by a
> divergence.
>
> | | |
> |---|---|
> | Proof | `scripts/dft_vxc_matrix_elements_sympy.py`, claims closed, polarized, numeric, ibp |
> | Negative control | the runner drops the $\tfrac12$ on $v_\tau$ (numeric closed fails), sets the polarized cross coefficient to 2, and halves the GGA factor; the script's own controls fail the same way symbolically |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` (validation tier: about 12 minutes) |
> | Code | `ferric-dft/src/vxc.rs::{semilocal_vxc_closed_scratch,semilocal_vxc_polarized_scratch}` |
> | Code test | `tau_from_density_matrix_matches_explicit_mo_sum` in `crates/ferric-dft/src/density_on_grid.rs` |
>
> <!-- verified: id=ks.vxc.matrix-elements; script=scripts/dft_vxc_matrix_elements_sympy.py; claim=closed; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-dft/src/vxc.rs::{semilocal_vxc_closed_scratch,semilocal_vxc_polarized_scratch}; rust_test=ferric-dft/src/density_on_grid.rs::tau_from_density_matrix_matches_explicit_mo_sum; registry=ks-mgga -->

The numeric claim's residual on the model is $\le 7\times10^{-17}$, closed and
polarized. The production check of the same identity is end to end: a
central difference of ferric's own $E_{xc}$ along a random symmetric $X$
against $\mathrm{Tr}[V_{xc}X]$ for PBE, SCAN, ωB97X and ωB97X-V, closed and
polarized, to $10^{-9}$ relative (`test_vxc_is_the_derivative_of_exc` in
`crates/ferric-python/tests/test_xc_blocks.py`).
`crates/ferric-dft/tests/mgga_polarized_packing.rs` checks the polarized
packing, the independence of the two $\tau$ channels, and the values
against PySCF's `eval_xc`.

**The dense reference in code.** `ferric-dft/src/vxc.rs::semilocal_vxc_closed_scratch`
holds $\chi$ and $\nabla\chi$ over the whole grid and builds each term as a
scaled GEMM. The energy is the grouped sum:

<!-- excerpt: ferric-dft/src/vxc.rs::semilocal_vxc_closed_scratch -->
```rust
let e_xc: f64 = deterministic_point_sum(npts, |g| w[g] * dens.rho[g] * exc_total[g]);
```

The GGA term is built as $M_a + M_a^{\mathsf T}$ per axis $a$, with
$M_{a,\mu\nu} = \sum_g f_{a,g}\chi_\mu\partial_a\chi_\nu$ and
$f_a = 2wv_\sigma\partial_a\rho$. This closure computes $f_a$ at one point,
zero below the density floor:

<!-- excerpt: ferric-dft/src/vxc.rs::semilocal_vxc_closed_scratch -->
```rust
if above_density_floor(r, floor) {
    2.0 * w * v * gr
} else {
    0.0
}
```

| Symbol | Code |
|---|---|
| $w_g$, $\rho_g$, $\epsilon_g$ | `w`, `dens.rho`, `exc_total` |
| $v_\sigma$, $\partial_a\rho$ | `v`, `gr` |
| $f_a$ | the closure's value |

The $\tau$ term is built with the factor $\tfrac12wv_\tau$ and is symmetric
on its own. The function ends with $\tfrac12(V + V^{\mathsf T})$, which leaves
the symmetric pieces unchanged. Points with $\rho$ at or below the
functional's `DensityFloor` are dropped: `DensityFloor::for_xc` is
$10^{-14}$ for LDA, GGA and their hybrids, $10^{-10}$ when any component is a
meta-GGA. The polarized function,
`ferric-dft/src/vxc.rs::semilocal_vxc_polarized_scratch`, floors each spin
block on its own $\rho_\sigma$.

## 9.3.3 The batched algorithm

The dense functions above hold $\chi$ and $\nabla\chi$ over the whole grid.
The SCF uses `xc_batch.rs`, which computes the same matrix in spatial
batches and calls the same libxc kernels, so the two cannot drift.

| Step | What | Cost |
|---|---|---|
| 1 | bisect the grid into compact batches of at most 128 points (pure function of the coordinates) | once per grid |
| 2 | per batch, keep a shell if any $\lvert\chi\rvert$ or $\lvert\partial\chi\rvert$ on the batch exceeds $10^{-10}$ (actual values, not a bound) | once per grid |
| 3 | per batch, $\Phi = D_{\rm sub}[\chi\,\vert\,\partial_x\chi\,\vert\,\partial_y\chi\,\vert\,\partial_z\chi]$; $\rho$, $\nabla\rho$, $\tau$ | one GEMM, $O(n_{\rm act}^2 n_b)$ |
| 4 | libxc once over the whole grid: $\epsilon$, $v_\rho$, $v_\sigma$, $v_\tau$ | $O(N_{\rm grid})$ |
| 5 | $E_{xc} = \sum_g w_g\rho_g\epsilon_g$, fixed grouping | $O(N_{\rm grid})$ |
| 6 | per batch, $V_b = A B^{\mathsf T}$, $B = [\chi\,\vert\,\partial\chi]$, $A = [\tfrac12 s\chi + \sum_a f_a\partial_a\chi\,\vert\,\tfrac12t\,\partial\chi]$; $V_{\rm sub} \mathrel{+}= V_b + V_b^{\mathsf T}$ | one GEMM |

with $s = wv_\rho$, $f_a = 2wv_\sigma\partial_a\rho$, $t = \tfrac12wv_\tau$
(below the density floor all three are zero).

Step 6 folds all three terms into one GEMM. Read off what
$V_b + V_b^{\mathsf T}$ contains, term by term:

- LDA: $\tfrac12s\chi_\mu\chi_\nu$ plus its transpose is $s\chi_\mu\chi_\nu$.
- GGA: $\sum_af_a\partial_a\chi_\mu\,\chi_\nu$ plus its transpose is
  $\sum_af_a(\chi_\mu\partial_a\chi_\nu + \chi_\nu\partial_a\chi_\mu)$, the
  dense $\sum_a(M_a + M_a^{\mathsf T})$.
- $\tau$: $\tfrac12t\,\partial\chi_\mu\cdot\partial\chi_\nu$ plus its
  transpose is $t\,\partial\chi_\mu\cdot\partial\chi_\nu = \tfrac12wv_\tau\,\partial\chi_\mu\cdot\partial\chi_\nu$.

So $V_b + V_b^{\mathsf T}$ is exactly the dense $\tfrac12(V + V^{\mathsf T})$:
the extra $\tfrac12$ on the LDA and $\tau$ columns of $A$ is cancelled by the
transpose, and the GGA column needs no $\tfrac12$ because its two halves are
the two orderings. The per-point factors,
`ferric-dft/src/xc_batch.rs::closed_point_factors`, carry those halves:

<!-- excerpt: ferric-dft/src/xc_batch.rs::closed_point_factors -->
```rust
if above_density_floor(r, floor) {
    fac.s_half[j] = 0.5 * (w * vrho);
    if has_gga {
        for (ax, ga) in grad.iter().enumerate() {
            fac.f[ax][j] = 2.0 * w * vsigma * ga;
        }
    }
    if has_mgga {
        fac.t_half[j] = 0.5 * (0.5 * w * vtau);
    }
}
```

| Symbol | Code |
|---|---|
| $\tfrac12s = \tfrac12wv_\rho$ | `fac.s_half[j]` |
| $f_a = 2wv_\sigma\partial_a\rho$ | `fac.f[ax][j]` |
| $\tfrac12t = \tfrac12\cdot\tfrac12wv_\tau$ | `fac.t_half[j]` |

The expressions inside are the dense path's, then an exact $\times\tfrac12$
(a power of two), so the batched and dense factors differ by no rounding.

**Reproducibility.** Batch boundaries depend only on the coordinates;
inside a batch everything is serial; the batch contributions are summed in a
fixed grouping that depends on the problem shape, never on the thread
count. The result is bit-identical across thread counts. Setting the
screening threshold to zero keeps every shell and is the exactness anchor
of the screen.

## 9.3.4 Options

| Option | Effect |
|---|---|
| the functional family (section 9.1) | which of $v_\sigma$, $v_\tau$ are nonzero |
| `DensityFloor::for_xc` | $10^{-14}$, or $10^{-10}$ with a meta-GGA component; not user-set |
| resident or recompute AO batches | chosen from the memory budget; the same bits |

## 9.3.5 Numerical anchors

| Quantity | Value | Source |
|---|---|---|
| FD of $E_{xc}$ vs $\mathrm{Tr}[V_{xc}X]$, PBE, SCAN, ωB97X-V | $10^{-9}$ relative | `test_vxc_is_the_derivative_of_exc`, `crates/ferric-python/tests/test_xc_blocks.py` |
| polarized evaluation vs PySCF `eval_xc`, packing | bit-for-bit | `crates/ferric-dft/tests/mgga_polarized_packing.rs` |
| SCAN, r2SCAN energies vs PySCF | $\le 4.0\times10^{-12}$ Ha (RKS) | `crates/ferric-scf/tests/validation_mgga_energies.rs` |

## 9.3.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-dft/theory.md`, "Numerical Integration", "Semilocal $V_{xc}$ matrix elements" | the reference statement of the equations |
| `crates/ferric-dft/src/xc_batch.rs` module documentation | the batched design, its measurements and the determinism rules |
| Pople, Gill and Johnson, Chem. Phys. Lett. 199, 557 (1992) | the matrix-element form of the GGA potential |
