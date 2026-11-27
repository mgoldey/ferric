# The cavity and the boundary-element matrices {#cavity-and-bem}

<!-- status: ready; registry: pcm -->

A polarizable continuum model replaces the solvent by a dielectric outside a
surface that wraps the solute. When the solute's field polarizes the
dielectric, the effect on the solute is the same as that of a charge density
on the surface, and that density is the unknown of a boundary integral
equation. To put the equation on a computer the surface is cut into small
elements, tesserae, each carrying one unknown charge, and the integral
operators become dense matrices over the tesserae. This section builds the
tesserae (@sec:cavity-and-bem-cavity) and two of those matrices, $S$ and $D$ (@sec:cavity-and-bem-matrices); section
@sec:ief-operators combines them into the solver.

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
| Python | `solvent=` and `pcm_lebedev_order=` on `run_rhf` and the other SCF drivers (`crates/ferric-python/python/ferric/ferric.pyi`) |

Units are Hartree atomic units throughout the chapter: Bohr, Hartree, and
charge in units of $e$. $N$ is the number of tesserae.

## The cavity {#cavity-and-bem-cavity}

The cavity is a union of spheres, one per atom that is not a ghost. Atom $A$
gets the radius

$$ R_A = s_{\mathrm{vdW}}\, r_{\mathrm{mB}}(Z_A), \qquad s_{\mathrm{vdW}} = 1.2 , $$

where $r_{\mathrm{mB}}$ is the modified Bondi radius: Bondi's 1964 values
except H = 1.10 Å and B = 1.92 Å. These are PySCF's `modified_Bondi` values
(`ferric-pcm/src/radii.rs::modified_bondi_radius_bohr`), and the scale 1.2 is
the default of `ferric-pcm/src/cavity.rs::CavityConfig`. An element with no
tabulated radius falls back to 2.0 Å.

Each sphere carries a Lebedev grid of directions $\hat n_g$ and weights
$w_g$, with the weights normalized to $\sum_g w_g = 1$
(`ferric_quadrature::lebedev`). Put one tessera at every grid point of
sphere $A$. Its position, normal and area are

$$ \mathbf r_k = \mathbf R_A + R_A \hat n_g, \qquad \hat{\mathbf n}_k = \hat n_g \ \text{(outward)}, \qquad a_k = 4\pi R_A^2\, w_g\, \mathrm{swf}_k . $$

Without the factor $\mathrm{swf}_k$ the areas of one sphere add up to
$4\pi R_A^2\sum_g w_g = 4\pi R_A^2$. A lone sphere has $\mathrm{swf}_k = 1$
everywhere, so its total area is exactly $4\pi R_A^2$.

Now the overlap between spheres. A tessera on sphere $A$ that lies inside
sphere $B$ is not on the surface. A hard keep-or-discard test would make the
area jump as an atom moves, and the energy would then not be differentiable.
The code uses a smooth switch instead. Each other sphere $B$ multiplies the
area by a factor that goes from 0 (deep inside $B$) to 1 (outside it):

$$ \mathrm{swf}_k = \prod_{B\ne A} h_{\rm sw}\!\left(\frac{|\mathbf r_k-\mathbf R_B| - R^{\mathrm{in}}_B}{R^{\mathrm{sw}}_B}\right), \qquad h_{\rm sw}(x) = x^3(10 - 15x + 6x^2) \ \text{on}\ [0,1] , $$

with $h_{\rm sw} = 0$ below the band and $h_{\rm sw} = 1$ above it. $h_{\rm sw}$ is the quintic
smoothstep of Lange and Herbert (JCP 133, 244111 (2010), eq. 3.19;
`ferric-pcm/src/cavity.rs::switch_h`). Its derivative is
$h_{\rm sw}'(x) = 30x^2(1-x)^2$ (`ferric-pcm/src/cavity.rs::switch_h_deriv`), and
both $h_{\rm sw}'$ and $h_{\rm sw}''$ vanish at $x = 0$ and $x = 1$. So the switch joins the
constant pieces with a continuous second derivative, and section @sec:gradient can
differentiate it. The band width scales with the grid spacing,

$$ R^{\mathrm{sw}}_B = R_B\sqrt{14/N_{\mathrm{Leb}}} , $$

and the inner radius is $R^{\mathrm{in}}_B = R_B - \alpha R^{\mathrm{sw}}_B$
with $\alpha = \tfrac12 + R_B/R^{\mathrm{sw}}_B - \sqrt{(R_B/R^{\mathrm{sw}}_B)^2 - 1/28}$
(Lange and Herbert, eqs. 3.19 to 3.21; PySCF `gen_surface`). Both come from
`SphereGeometry::new` in `ferric-pcm/src/cavity.rs`, which `build_cavity` and
the gradient of section @sec:gradient share.

A tessera with $w_g\,\mathrm{swf}_k \le 10^{-16}$ is dropped: it would add a
negligible area and one more unknown. The cavity has no interstitial spheres
(GEPOL "added spheres"), so the surface is the union of the atomic spheres
only.

Each tessera also carries a Gaussian width, used by the matrices of @sec:cavity-and-bem-matrices:

$$ \xi_k = \frac{\Xi(N_{\mathrm{Leb}})}{R_A\sqrt{4\pi w_g}} . $$

$\Xi$ is a per-order constant (`ferric-pcm/src/matrices.rs::gaussian_xi_table`),
copied from PySCF's `pcm.py` table `XI`, which cites J. Chem. Phys. 122,
194110 (2005), Table II. Since $R_A\sqrt{4\pi w_g} = \sqrt{4\pi R_A^2 w_g}$,
the width is $\Xi$ over the square root of the unswitched tessera area: a
tessera on a coarse grid is a broad Gaussian, one on a fine grid a narrow
one. The width does not depend on where the other atoms are.

The kernel of `build_cavity` follows these formulas line for line. The width
is set by the Lebedev weight alone, after converting to PySCF's convention in
which the weights of a sphere sum to $4\pi$ (the excerpt is unchanged):

<!-- excerpt: ferric-pcm/src/cavity.rs::build_cavity -->
```rust
let w_pyscf = w * 4.0 * std::f64::consts::PI;
let charge_exp = xi_prefactor / (r_a * w_pyscf.sqrt());
```

The switch is accumulated over the other atoms, with an early exit once the
product is zero:

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

`switch_arg` clamps an argument below $10^{-8}$ in magnitude to zero, as
PySCF does. `PcmConfig::cavity` (`ferric-pcm/src/config.rs`) replaces the
whole construction with an explicit tessera list.
`crates/ferric-scf/tests/validation_pcm.rs` uses it to feed PySCF's cavity
through ferric's solver.

## The matrices $S$ and $D$ {#cavity-and-bem-matrices}

$S_{ij}$ is the Coulomb potential at tessera $i$ of a unit charge spread
over tessera $j$. $D_{ij}$ is the potential at $i$ of a unit dipole layer at
$j$, with the dipole along the normal at the source $j$. With
$\mathbf r_{ij} = \mathbf r_i - \mathbf r_j$ and $r_{ij} = |\mathbf r_{ij}|$,
the source normal enters through

$$ \nu_{ij} = \mathbf r_{ij}\cdot\hat{\mathbf n}_j . $$

For point charges, $D_{ij}$ is the normal derivative of $1/r_{ij}$ at the
source: $D_{ij} = \hat{\mathbf n}_j\cdot\nabla_{\mathbf r_j}(1/|\mathbf r_i-\mathbf r_j|) = \nu_{ij}/r_{ij}^3$.
$\nu_{ij}$ uses $\hat{\mathbf n}_j$ and $\nu_{ji}$ uses $\hat{\mathbf n}_i$, so
$D$ is not symmetric. With this sign a closed surface gives
$\sum_j D_{ij}a_j \to -2\pi$ for $i$ on the surface, which is Gauss's law
for a dipole layer.

**`SdKind::GaussianSmeared` (the default).** Each tessera charge is a
normalized Gaussian $(\xi_k^2/\pi)^{3/2}e^{-\xi_k^2|\mathbf r-\mathbf r_k|^2}$.
The Coulomb energy of two such Gaussians is $\operatorname{erf}(\xi_{ij}r_{ij})/r_{ij}$,
with the widths combined as $1/\xi_{ij}^2 = 1/\xi_i^2 + 1/\xi_j^2$:

$$ S_{ij} = \frac{\operatorname{erf}(\xi_{ij} r_{ij})}{r_{ij}}, \qquad \xi_{ij} = \frac{\xi_i\xi_j}{\sqrt{\xi_i^2+\xi_j^2}} \qquad (i\ne j) . $$

Take $D_{ij}$ as the source-normal derivative of $S_{ij}$, as for point
charges. $S_{ij}$ depends on $r_{ij}$ only, and
$\nabla_{\mathbf r_j} r_{ij} = -\mathbf r_{ij}/r_{ij}$, so
$\hat{\mathbf n}_j\cdot\nabla_{\mathbf r_j}S_{ij} = -S'(r_{ij})\,\nu_{ij}/r_{ij}$.
Differentiate the erf:

$$ D_{ij} = \frac{S_{ij}\,\nu_{ij}}{r_{ij}^2} - \frac{2}{\sqrt\pi}\,\frac{\xi_{ij} r_{ij}\, e^{-\xi_{ij}^2 r_{ij}^2}\,\nu_{ij}}{r_{ij}^3} \qquad (i\ne j) . $$

As $\xi_{ij}\to\infty$ the erf goes to 1 and the Gaussian term to 0, which
recovers the point-charge $1/r_{ij}$ and $\nu_{ij}/r_{ij}^3$.

The diagonal is where the Gaussians help. A point charge has an infinite
self-energy. A Gaussian of width $\xi_i$ interacting with itself has
$\xi_{ii} = \xi_i/\sqrt2$, and the $r\to 0$ limit of
$\operatorname{erf}(\xi_i r/\sqrt2)/r$ is $\xi_i\sqrt{2/\pi}$. The code
divides this self-energy by the switching value:

$$ S_{ii} = \frac{\xi_i\sqrt{2/\pi}}{\mathrm{swf}_i}, \qquad D_{ii} = -\frac{\xi_i\sqrt{2/\pi}}{2R_i} . $$

Dividing by $\mathrm{swf}_i$ makes a tessera that is fading out of the
surface expensive to charge. The finite diagonal also keeps $S$ well conditioned on a dense grid.
$D_{ii}$ keeps the unswitched self-term. That is PySCF's convention, which
ferric copies so that the two codes solve the same equations on the same
cavity (section @sec:ief-operators), and section @sec:gradient differentiates
the two diagonals accordingly: $S_{ii}$ depends on the switch, $D_{ii}$ does
not.

**`SdKind::PointCharge`.** Bare point charges: $S_{ij} = 1/r_{ij}$,
$D_{ij} = \nu_{ij}/r_{ij}^3$, $S_{ii} = 1.0694\sqrt{4\pi/a_i}$ (the GEPOL
self-energy constant `XI_SELF` in `ferric-pcm/src/matrices.rs`) and
$D_{ii} = -S_{ii}/(2R_i)$.

Both diagonal $D$ formulas are $-1/(2R_i)$ times a single-layer self-term:
the unswitched $\xi_i\sqrt{2/\pi}$ for `GaussianSmeared`, and $S_{ii}$ itself
for `PointCharge`. So $D_{ii} = -\mathrm{swf}_iS_{ii}/(2R_i)$ for the default
kind, and $D_{ii} = -S_{ii}/(2R_i)$ only for `PointCharge`. Why the factor? On a sphere of radius
$R$ the two kernels are proportional. Let $\cos\gamma = \hat n\cdot\hat n'$
for two points on the sphere. Then
$|\mathbf r-\mathbf r'|^2 = 2R^2(1-\cos\gamma)$ and
$(\mathbf r-\mathbf r')\cdot\hat n' = R\hat n\cdot\hat n' - R = -R(1-\cos\gamma)$.
Divide the second by the cube of the square root of the first:

$$ \frac{(\mathbf r-\mathbf r')\cdot\hat n'}{|\mathbf r-\mathbf r'|^3} = -\frac{1}{2R}\,\frac{1}{|\mathbf r-\mathbf r'|} . $$

The self-term of a tessera is an integral over a small patch of its own
sphere, so the double-layer self-term is $-1/(2R_i)$ times the single-layer
one.

> **Verified equation {#eq:pcm-bem-diagonal}** · the double-layer diagonal on a sphere
>
> On a sphere of radius $R$ the double-layer kernel with the source normal
> equals $-1/(2R)$ times the single-layer kernel,
> $(\mathbf r-\mathbf r')\cdot\hat n'/|\mathbf r-\mathbf r'|^3
> = -\tfrac{1}{2R}\,1/|\mathbf r-\mathbf r'|$. Both diagonal formulas apply
> this identity to the tessera's own sphere, which is why $D_{ii}$ is $-1/(2R_i)$ times the self-term.
>
> | | |
> |---|---|
> | Proof | `scripts/pcm_ief_sympy.py`, claim 1 |
> | Negative control | the field-point normal gives the opposite sign, $+\tfrac{1}{2R}$ times the single-layer kernel; the runner transcribes the kernel with $+R(1-\cos\gamma)$ (claim 1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pcm/src/matrices.rs::build_s_d_kind` |
> | Code test | `point_charge_in_sphere_matches_born_kirkwood_and_gauss_law` in `crates/ferric-pcm/tests/ief_limits.rs`, whose sphere cases exercise the diagonal |
>
> <!-- verified: id=pcm.bem.diagonal; script=scripts/pcm_ief_sympy.py; claim=1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pcm/src/matrices.rs::build_s_d_kind; rust_test=ferric-pcm/tests/ief_limits.rs::point_charge_in_sphere_matches_born_kirkwood_and_gauss_law; registry=pcm -->

The same proportionality makes $D$ symmetric on a sphere:
$\nu_{ij} = \nu_{ji} = -R(1-\hat n_i\cdot\hat n_j)$. Section @sec:ief-operators uses this
when it checks the operators on a sphere.

The matrices are discrete forms of two integral operators,
$\hat S\sigma = \int \sigma(\mathbf r')/|\mathbf r-\mathbf r'|\,dS'$ and
$\hat D$, the same integral with the double-layer kernel. A surface integral
becomes a sum over tesserae weighted by area, so
$\hat S \leftrightarrow SA$ and $\hat D \leftrightarrow DA$ with
$A = \operatorname{diag}(a_k)$. On a sphere both operators are diagonal in
the spherical harmonics (the Funk–Hecke theorem), with the eigenvalues of
box @eq:pcm-bem-funk-hecke. At $l = 0$ the $\hat D$ eigenvalue is $-2\pi$, the sum rule
$\sum_j D_{ij}a_j \to -2\pi$ above.

> **Verified equation {#eq:pcm-bem-funk-hecke}** · the continuum spectra the discretization must reproduce
>
> On a sphere the continuum operators are diagonal in the spherical
> harmonics: $\hat S Y_{lm} = \tfrac{4\pi R}{2l+1}Y_{lm}$ and
> $\hat D Y_{lm} = -\tfrac{2\pi}{2l+1}Y_{lm}$. These are the channel scalars
> section @sec:ief-operators inserts into $K$ and $R$.
>
> | | |
> |---|---|
> | Proof | `scripts/pcm_ief_sympy.py`, claim 2 (Funk–Hecke, exact integrals for $l = 0$ to $6$) |
> | Negative control | the runner sets $D_l = -S_l/R$ in place of $-S_l/(2R)$ (claim 2 fails at every $l$) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pcm/src/matrices.rs::build_s_d_kind` |
> | Code test | `point_charge_in_sphere_matches_born_kirkwood_and_gauss_law` in `crates/ferric-pcm/tests/ief_limits.rs`, which converges to the continuum only if the discrete $S$ and $D$ approach these spectra |
>
> <!-- verified: id=pcm.bem.funk-hecke; script=scripts/pcm_ief_sympy.py; claim=2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pcm/src/matrices.rs::build_s_d_kind; rust_test=ferric-pcm/tests/ief_limits.rs::point_charge_in_sphere_matches_born_kirkwood_and_gauss_law; registry=pcm -->

The `GaussianSmeared` branch of `build_s_d_kind` fills the diagonal first:

<!-- excerpt: ferric-pcm/src/matrices.rs::build_s_d_kind -->
```rust
s[(i, i)] = xi_i * sqrt_2_over_pi / ti.switch_fun;
d[(i, i)] = -xi_i * sqrt_2_over_pi / (2.0 * ti.sphere_radius);
```

It then visits each unordered pair $i < j$ once, writes $S_{ij}$ into both
triangles, and computes $D_{ij}$ and $D_{ji}$ with their own source normals.
The two pieces below are unchanged:

<!-- excerpt: ferric-pcm/src/matrices.rs::build_s_d_kind -->
```rust
let xi_ij = xi_i * xi_j / (xi_i * xi_i + xi_j * xi_j).sqrt();
let xi_r = xi_ij * r;
// SAFETY: erf is the C standard math library function; xi_r is a finite f64.
let s_ij = erf(xi_r) / r;
s[(i, j)] = s_ij;
s[(j, i)] = s_ij; // S is symmetric.
```

<!-- excerpt: ferric-pcm/src/matrices.rs::build_s_d_kind -->
```rust
let gauss_term =
    2.0 / std::f64::consts::PI.sqrt() * xi_r * (-xi_r * xi_r).exp() / (r2 * r);

let dot_j = dx * tj.normal[0] + dy * tj.normal[1] + dz * tj.normal[2];
d[(i, j)] = s_ij * dot_j / r2 - gauss_term * dot_j;

let dot_i = (-dx) * ti.normal[0] + (-dy) * ti.normal[1] + (-dz) * ti.normal[2];
d[(j, i)] = s_ij * dot_i / r2 - gauss_term * dot_i;
```

| Symbol | Code |
|---|---|
| $\xi_i$ | `xi_i` |
| $\sqrt{2/\pi}$ | `sqrt_2_over_pi` |
| $\mathrm{swf}_i$, $R_i$ | `ti.switch_fun`, `ti.sphere_radius` |
| $\xi_{ij}$, $\xi_{ij}r_{ij}$ | `xi_ij`, `xi_r` |
| $S_{ij}$ | `s_ij` |
| $\nu_{ij} = \mathbf r_{ij}\cdot\hat{\mathbf n}_j$, $\nu_{ji}$ | `dot_j`, `dot_i` (`dx` is $x_i - x_j$) |
| $r_{ij}^2$ | `r2` |
| $\tfrac{2}{\sqrt\pi}\,\xi_{ij}r_{ij}e^{-\xi_{ij}^2r_{ij}^2}/r_{ij}^3$ | `gauss_term` |

`d[(i, j)]` uses `tj.normal` and `d[(j, i)]` uses `ti.normal`. If you port
this loop, keep the source normal: swapping it for the field normal gives
$D^{\mathsf T}$, and section @sec:ief-operators shows that the order of $D$ and its
transpose changes the energy.

## The algorithm {#cavity-and-bem-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | sphere radii $R_A$, switching band $R^{\mathrm{sw}}_A$ and inner radius $R^{\mathrm{in}}_A$ | $N_{\mathrm{at}}$ | `ferric-pcm/src/cavity.rs::SphereGeometry` |
| 2 | Lebedev rule of order $N_{\mathrm{Leb}}$, width constant $\Xi$ | table lookup | `ferric_quadrature::lebedev`, `gaussian_xi_table` |
| 3 | tesserae: position, normal, area, $\xi_k$, $\mathrm{swf}_k$; drop $w_g\,\mathrm{swf}_k \le 10^{-16}$ | $N_{\mathrm{at}}^2N_{\mathrm{Leb}}$ | `build_cavity` |
| 4 | $S$ and $D$ over all tessera pairs | $N^2$ | `build_s_d_kind` |
| 5 | $K$ and $R$ (section @sec:ief-operators) | $N^3$ | `ferric-pcm/src/matrices.rs::build_k_r` |

The cavity, $S$, $D$, $K$ and $R$ depend only on the geometry and the
dielectric constant. `PcmContext` builds them once before the SCF loop, and
the loop re-solves only the charges.

## Options {#cavity-and-bem-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[pcm] solvent`, `solvent=` | a name from `ferric_pcm::NAMED_SOLVENTS`, or a number $> 1$ in Python | none | sets $\varepsilon$; an unknown name is an error, never vacuum |
| `[pcm] epsilon` | float $> 1$ | none | exclusive with `solvent`; $\varepsilon = 1$ is refused |
| `[pcm] lebedev_order`, `pcm_lebedev_order=` | a positive-weight Lebedev order (`SUPPORTED_LEBEDEV_ORDERS`) | 110 | tesserae per sphere |
| `PcmConfig::vdw_scale` | float | 1.2 | scale on the modified Bondi radii; library only |
| `PcmConfig::sd_kind` | `GaussianSmeared`, `PointCharge` | `GaussianSmeared` | the $S$ and $D$ formulation; library only |
| `PcmConfig::probe` | `Point`, `GaussianSmeared` | `Point` | section @sec:reaction-field; library only |
| `PcmConfig::cavity` | an explicit tessera list | none | replaces the construction; no gradient, not settable from TOML |

The Lebedev orders with negative weights (74, 230, 266) are excluded,
because a negative weight would give a tessera of negative area.

The solvation energy of water / cc-pVDZ in water at three Lebedev orders, from
the `589b50832` build. It moves by less than 0.1 kcal/mol from 50 to 302
points per sphere; 110 is the default.

```python
import ferric

mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
bs = ferric.BasisSet.bundled("cc-pvdz")

e_vac = ferric.run_rhf(mol, bs).energy
for order in (50, 110, 302):
    e_sol = ferric.run_rhf(mol, bs, solvent="water", pcm_lebedev_order=order).energy
    d = e_sol - e_vac
    print(f"lebedev {order:4d}: E(solv) - E(vac) = {d:.8f} Ha = {d * 627.5095:.4f} kcal/mol")
```

```text
lebedev   50: E(solv) - E(vac) = -0.00983213 Ha = -6.1698 kcal/mol
lebedev  110: E(solv) - E(vac) = -0.00992420 Ha = -6.2275 kcal/mol
lebedev  302: E(solv) - E(vac) = -0.00997667 Ha = -6.2605 kcal/mol
```

## Numerical anchors {#cavity-and-bem-numerical-anchors}

| Quantity | System | Result | Source |
|---|---|---|---|
| ferric's own cavity against PySCF's, solvation energy | water, NH3 / STO-3G, cc-pVDZ; $\varepsilon = 78.4, 4.7$ | 0.07 to 0.61 % apart | `site/src/reference/validation.md`, IEF-PCM row |
| the 302-point cavity against PySCF's `gen_surface` | water, methanol / STO-3G | tessera count, total area and area centroid equal to $10^{-10}$ | section @sec:gradient-numerical-anchors |

The first row is a measured difference between two models, not a bound on
an error. The radii (modified Bondi), the default Lebedev order (110 against
PySCF's 302), the probe and the fallback radius of 2.0 Å for untabulated
elements all differ. At order 302 ferric's cavity reproduces PySCF's
(second row), which is the setting of the gradient comparison in section
@sec:gradient.

## Sources {#cavity-and-bem-sources}

| Source | What it gives |
|---|---|
| `scripts/pcm_ief_sympy.py`, claims 1 and 2 | the diagonal identity and the sphere spectra |
| `crates/ferric-pcm/tests/ief_limits.rs` | the sphere limits of section @sec:ief-operators |

**Known limits (2026-11-15).**

- No interstitial spheres (GEPOL "added spheres"): the cavity is the union of
  the atomic spheres only.
- No cavitation, dispersion or repulsion terms.
- The Lebedev grid of every sphere is fixed in the lab frame, so the torque on
  a rigid rotation is not zero in general (section @sec:gradient-numerical-anchors).
- `PcmContext::from_tesserae` rejects a tessera with a non-finite position, a
  normal that is not unit length to $10^{-8}$, or a non-positive area, radius,
  width or switching value, because each would give a NaN in $S$ or $D$.
