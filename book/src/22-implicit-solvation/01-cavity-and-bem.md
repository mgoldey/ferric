# 22.1 The cavity and the boundary-element matrices

<!-- section: 22.1; status: ready; registry: pcm;
     theory: crates/ferric-pcm/theory.md#"1. Cavity", "2. Boundary-element matrices $S$ and $D$" -->

A polarizable continuum model replaces the solvent by a dielectric outside a
surface that wraps the solute. The unknown is a charge density on that
surface, found by solving a boundary integral equation. To put the equation
on a computer the surface is cut into small elements, tesserae, each carrying
one unknown charge, and the integral operators become dense matrices over the
tesserae. This section builds the tesserae and two of those matrices. Section
22.2 combines them into the solver.

| Symbol | Meaning | Code name |
|---|---|---|
| $i, j, k$ | tesserae, $1\ldots N$ (no occupied-orbital index appears in this chapter, so the letters are reused) | rows and columns of `s`, `d`, `k`, `r` |
| $A, B$ | atoms | `Molecule::atoms` |
| $\mu, \nu$ | AO basis functions | `0..nbas` |
| $\mathbf r_k$, $\hat{\mathbf n}_k$, $a_k$ | tessera position, outward unit normal, area | `Tessera::{position, normal, area}` |
| $R_A$ | sphere radius of atom $A$ | `Tessera::sphere_radius` |
| $\xi_k$ | Gaussian width of tessera $k$ | `Tessera::charge_exp` |
| $\mathrm{swf}_k$ | switching-function value at tessera $k$ | `Tessera::switch_fun` |
| $S$, $D$ | single-layer and double-layer matrices | `s`, `d` |

| Entry point | Location |
|---|---|
| Rust | `ferric-pcm/src/cavity.rs::build_cavity`, `ferric-pcm/src/matrices.rs::{build_s_d,build_s_d_kind,gaussian_xi_table}`, `ferric-pcm/src/lib.rs::PcmContext` |
| CLI | `[pcm]` with `solvent` or `epsilon`, and `lebedev_order` (`examples/water-pcm.toml`) |
| Python | `solvent=` and `pcm_lebedev_order=` on `run_rhf` and the other SCF drivers (`crates/ferric-python/ferric.pyi`) |

## 22.1.1 The cavity

<!-- include: crates/ferric-pcm/theory.md#"1. Cavity" -->

The cavity is a union of spheres, so a tessera on one sphere that lies inside
another is not on the surface. A hard keep-or-discard test would make the area
jump as an atom moves, and the gradient would then miss a term. The code uses
the smooth switch of the theory text instead: a point's area is multiplied by
the product of $h$ over every other sphere, which falls from 1 to 0 across a
band whose width scales with the grid spacing. Section 22.4 differentiates
that product.

The kernel of `ferric-pcm/src/cavity.rs::build_cavity` is four lines per
tessera. The width $\xi_k$ is set by the Lebedev weight alone, in PySCF's
convention where the weights of a sphere sum to $4\pi$:

<!-- excerpt: ferric-pcm/src/cavity.rs::build_cavity -->
```rust
let w_pyscf = w * 4.0 * std::f64::consts::PI;
let charge_exp = xi_prefactor / (r_a * w_pyscf.sqrt());
```

The switch is accumulated over the other atoms and the point is dropped at the
numerical floor:

<!-- excerpt: ferric-pcm/src/cavity.rs::build_cavity -->
```rust
swf *= switch_h(geom.switch_arg(b_idx, dist));
if swf == 0.0 {
    break;
}
```

<!-- excerpt: ferric-pcm/src/cavity.rs::build_cavity -->
```rust
let area = 4.0 * std::f64::consts::PI * r_a * r_a * w * swf;
```

| Symbol | Code |
|---|---|
| $w_g$ (sum 1) | `w` |
| $4\pi w_g$ | `w_pyscf` |
| $\Xi(N_{\mathrm{Leb}})$ | `xi_prefactor`, from `gaussian_xi_table` |
| $\xi_k$ | `charge_exp` |
| $\prod_{B\neq A} h$ | `swf` |
| $a_k$ | `area` |

## 22.1.2 The matrices $S$ and $D$

$S_{ij}$ is the Coulomb potential at tessera $i$ of a unit charge spread over
tessera $j$. $D_{ij}$ is the potential at $i$ of a unit dipole layer at $j$,
with the normal taken at the source. The default treats each tessera charge as
a normalized Gaussian, which makes the diagonal finite and keeps the matrix
well conditioned on a dense grid.

<!-- include: crates/ferric-pcm/theory.md#"2. Boundary-element matrices $S$ and $D$" -->

Both diagonal formulas, and the off-diagonal $D$, are in
`ferric-pcm/src/matrices.rs::build_s_d_kind`:

<!-- excerpt: ferric-pcm/src/matrices.rs::build_s_d_kind -->
```rust
s[(i, i)] = xi_i * sqrt_2_over_pi / ti.switch_fun;
d[(i, i)] = -xi_i * sqrt_2_over_pi / (2.0 * ti.sphere_radius);
```

<!-- excerpt: ferric-pcm/src/matrices.rs::build_s_d_kind -->
```rust
d[(i, j)] = s_ij * dot_j / r2 - gauss_term * dot_j;
```

| Symbol | Code |
|---|---|
| $\xi_i$ | `xi_i` |
| $\sqrt{2/\pi}$ | `sqrt_2_over_pi` |
| $\mathrm{swf}_i$, $R_i$ | `ti.switch_fun`, `ti.sphere_radius` |
| $S_{ij}$ | `s_ij` |
| $\nu_{ij} = \mathbf r_{ij}\cdot\hat{\mathbf n}_j$ | `dot_j` |
| $r_{ij}^2$ | `r2` |
| $\tfrac{2}{\sqrt\pi}\,\xi_{ij}r_{ij}e^{-\xi_{ij}^2r_{ij}^2}/r_{ij}^3$ | `gauss_term` |

The normal is the source's, so $D_{ij}$ uses `tj.normal` and $D_{ji}$ uses
`ti.normal`. $D$ is therefore not symmetric, and section 22.2 shows that this
is why the charge solve needs a symmetrization.

> **Verified equation 22.1** · the double-layer diagonal on a sphere
>
> On a sphere of radius $R$ the double-layer kernel with the source normal
> equals $-1/(2R)$ times the single-layer kernel,
> $(\mathbf r-\mathbf r')\cdot\hat n'/|\mathbf r-\mathbf r'|^3
> = -\tfrac{1}{2R}\,1/|\mathbf r-\mathbf r'|$. Both diagonal formulas apply
> this identity to the tessera's own sphere, which is why $D_{ii} = -S_{ii}/(2R_i)$.
>
> | | |
> |---|---|
> | Proof | `scripts/pcm_ief_sympy.py`, claim 1 |
> | Negative control | the field-point normal gives the opposite sign, $+\tfrac{1}{2R}$ times the single-layer kernel |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pcm/src/matrices.rs::build_s_d_kind` |
> | Code test | `point_charge_in_sphere_matches_born_kirkwood_and_gauss_law` in `crates/ferric-pcm/tests/ief_limits.rs`, whose sphere cases exercise the diagonal |
>
> <!-- verified: id=pcm.bem.diagonal; script=scripts/pcm_ief_sympy.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pcm/src/matrices.rs::build_s_d_kind; rust_test=ferric-pcm/tests/ief_limits.rs::point_charge_in_sphere_matches_born_kirkwood_and_gauss_law; registry=pcm -->

> **Verified equation 22.2** · the continuum spectra the discretization must reproduce
>
> On a sphere the continuum operators are diagonal in the spherical
> harmonics: $\hat S Y_{lm} = \tfrac{4\pi R}{2l+1}Y_{lm}$ and
> $\hat D Y_{lm} = -\tfrac{2\pi}{2l+1}Y_{lm}$. These are the channel scalars
> section 22.2 inserts into $K$ and $R$.
>
> | | |
> |---|---|
> | Proof | `scripts/pcm_ief_sympy.py`, claim 2 (Funk–Hecke, exact integrals for $l = 0$ to $6$) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pcm/src/matrices.rs::build_s_d_kind` |
> | Code test | `point_charge_in_sphere_matches_born_kirkwood_and_gauss_law` in `crates/ferric-pcm/tests/ief_limits.rs`, which converges to the continuum only if the discrete $S$ and $D$ approach these spectra |
>
> <!-- verified: id=pcm.bem.funk-hecke; script=scripts/pcm_ief_sympy.py; claim=2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pcm/src/matrices.rs::build_s_d_kind; rust_test=ferric-pcm/tests/ief_limits.rs::point_charge_in_sphere_matches_born_kirkwood_and_gauss_law; registry=pcm -->

## 22.1.3 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | sphere radii $R_A$, switching band $R^{\mathrm{sw}}_A$ and inner radius $R^{\mathrm{in}}_A$ | $N_{\mathrm{at}}$ | `ferric-pcm/src/cavity.rs::SphereGeometry` |
| 2 | Lebedev rule of order $N_{\mathrm{Leb}}$, width constant $\Xi$ | table lookup | `ferric_quadrature::lebedev`, `gaussian_xi_table` |
| 3 | tesserae: position, normal, area, $\xi_k$, $\mathrm{swf}_k$; drop $w_g\,\mathrm{swf}_k \le 10^{-16}$ | $N_{\mathrm{at}}^2N_{\mathrm{Leb}}$ | `build_cavity` |
| 4 | $S$ and $D$ over all tessera pairs | $N^2$ | `build_s_d_kind` |
| 5 | $K$ and $R$ (section 22.2) | $N^3$ | `ferric-pcm/src/matrices.rs::build_k_r` |

The cavity, $S$, $D$, $K$ and $R$ depend only on the geometry and the
dielectric constant. `ferric-pcm/src/lib.rs::PcmContext` builds them once
before the SCF loop; the loop re-solves only the charges.

## 22.1.4 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[pcm] solvent`, `solvent=` | a name from `ferric_pcm::NAMED_SOLVENTS`, or a number $> 1$ in Python | none | sets $\varepsilon$; an unknown name is an error, never vacuum |
| `[pcm] epsilon` | float $> 1$ | none | exclusive with `solvent`; $\varepsilon = 1$ is refused |
| `[pcm] lebedev_order`, `pcm_lebedev_order=` | a positive-weight Lebedev order (`SUPPORTED_LEBEDEV_ORDERS`) | 110 | tesserae per sphere |
| `PcmConfig::vdw_scale` | float | 1.2 | scale on the modified Bondi radii; library only |
| `PcmConfig::sd_kind` | `GaussianSmeared`, `PointCharge` | `GaussianSmeared` | the $S$ and $D$ formulation; library only |
| `PcmConfig::probe` | `Point`, `GaussianSmeared` | `Point` | section 22.3; library only |
| `PcmConfig::cavity` | an explicit tessera list | none | replaces the construction; no gradient, not settable from TOML |

The Lebedev orders with negative weights (74, 230, 266) are excluded, because
a negative weight would give a tessera of negative area.

## 22.1.5 Numerical anchors

| Quantity | System | Result | Source |
|---|---|---|---|
| ferric's own cavity against PySCF's, solvation energy | water, NH3 / STO-3G, cc-pVDZ; $\varepsilon = 78.4, 4.7$ | 0.07 to 0.61 % apart | `site/src/reference/validation.md`, IEF-PCM row |
| the 302-point cavity against PySCF's `gen_surface` | water, methanol / STO-3G | tessera count, total area and area centroid equal to $10^{-10}$ | `crates/ferric-pcm/theory.md`, section 8.3 |
| one isolated sphere | any | total area exactly $4\pi R_A^2$ | `crates/ferric-pcm/theory.md`, section 1 |

The first row is a measured difference, not a bound: the radii (modified
Bondi), the default Lebedev order (110 against PySCF's 302), the probe and the
fallback radius of 2.0 Å for untabulated elements all differ.

## 22.1.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-pcm/theory.md`, sections 1 and 2 | the cavity and the $S$, $D$ formulas |
| `scripts/pcm_ief_sympy.py`, claims 1 and 2 | the diagonal identity and the sphere spectra |
| `crates/ferric-pcm/tests/ief_limits.rs` | the sphere limits of section 22.2 |

**Known limits (2026-11-15).**

- No interstitial spheres (GEPOL "added spheres"): the cavity is the union of
  the atomic spheres only.
- No cavitation, dispersion or repulsion terms.
- The Lebedev grid of every sphere is fixed in the lab frame, so the torque on
  a rigid rotation is not zero in general (`crates/ferric-pcm/theory.md`,
  section 8.3).
- `PcmContext::from_tesserae` rejects a tessera with a non-finite position, a
  normal that is not unit length to $10^{-8}$, or a non-positive area, radius,
  width or switching value, because each would give a NaN in $S$ or $D$.
