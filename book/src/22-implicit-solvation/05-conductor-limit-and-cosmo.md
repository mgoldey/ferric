# The conductor limit and COSMO {#conductor-limit-and-cosmo}

<!-- status: edit; registry: cosmo -->

A perfect conductor screens a charge completely: the potential on its
surface is constant, and with the right reference, zero. Replacing the
dielectric by a conductor removes the double-layer operator from the
problem and leaves one symmetric linear system for the surface charges.
COSMO, the conductor-like screening model, solves that system and rescales
the answer by a function of $\varepsilon$ to account for a real, finite
dielectric. The plan: derive the limit from the IEF-PCM operators (@sec:conductor-limit-and-cosmo-limit-ief-pcm),
then the COSMO driver of `ferric-scf` (@sec:conductor-limit-and-cosmo-cosmo-driver), its algorithm (@sec:conductor-limit-and-cosmo-algorithm) and
its options (@sec:conductor-limit-and-cosmo-options).

The limit is proved in `ferric-pcm` by `scripts/pcm_ief_sympy.py`. The COSMO
driver's equations are in `crates/ferric-scf/theory.md` ("COSMO reaction
field") and are proved in `scripts/cosmo_sympy.py`.

| Symbol | Meaning | Code name |
|---|---|---|
| $S$ | segment interaction matrix, symmetric | `build_s_matrix`, `s_mat` |
| $q_k$ | apparent charge on segment $k$ | `charges` |
| $v_k$ | solute potential at segment $k$ | `v` |
| $f_x(\varepsilon)$ | $(\varepsilon-1)/(\varepsilon+x)$, with $x = \tfrac12$ here | `CosmoConfig::f_epsilon` |
| $M_k$ | $\langle\mu\vert -1/\vert\mathbf r-\mathbf s_k\vert\vert\nu\rangle$ | the engine block with a $+1$ charge at the segment |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/cosmo.rs::{CosmoCavity,cosmo_reaction_field,CosmoConfig}`, `ferric-scf/src/driver.rs::solvent_terms` |
| CLI | `[cosmo]` with `epsilon`, `radius_scale`, `lebedev_order`, `s_matrix_kind`, for `task = "energy"` only |
| Python | none |

## The limit of IEF-PCM {#conductor-limit-and-cosmo-limit-ief-pcm}

Take $\varepsilon\to\infty$, so $f = (\varepsilon-1)/(\varepsilon+1)\to 1$.
Set $f = 1$ in the operators of section @sec:ief-operators:

$$ K = \left(I - \tfrac{1}{2\pi}DA\right)S, \qquad R = -\left(I - \tfrac{1}{2\pi}DA\right) . $$

The same factor $I - DA/2\pi$ now stands in front of both. Cancel it in
$K^{-1}R = -S^{-1}(I - DA/2\pi)^{-1}(I - DA/2\pi)$:

$$ K^{-1}R = -S^{-1}, \qquad S q = -v . $$

This holds on any discretization, for any $S$, $D$ and $A$. The double
layer has dropped out, and what is left is the conductor-like (C-PCM or
COSMO) equation. $S$ is symmetric, so $-S^{-1}$ is too, and the
symmetrization of section @sec:ief-operators changes nothing there.

How fast does IEF-PCM approach this? $K^{-1}R$ is smooth in $f$, so the gap
is first order in $1 - f = 2/(\varepsilon+1)$. The test
`ief_reduces_to_conductor_equation_as_eps_goes_to_infinity` measures it on
water's cavity with a model potential: the relative gap between the IEF
and conductor charges is $1.649\times10^{-6}$ at $\varepsilon = 10^6$ and
$1.649\times10^{-8}$ at $\varepsilon = 10^8$, a ratio of 100.000, and
$2.1\times10^{-2}$ at $\varepsilon = 78.4$.

> **Verified equation {#eq:pcm-conductor-limit}** · IEF-PCM at $f = 1$ is the conductor equation
>
> At $f = 1$, $K = (I - DA/2\pi)S$ and $R = -(I - DA/2\pi)$ for any $S$, $D$
> and $A$, so $K^{-1}R = -S^{-1}$ exactly and $Kq = Rv$ becomes $Sq = -v$.
> The check is exact rational arithmetic on a random symmetric positive
> definite $S$ and a non-symmetric $D$.
>
> | | |
> |---|---|
> | Proof | `scripts/pcm_ief_sympy.py`, claim 4 |
> | Negative control | $f < 1$ is not the conductor equation |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pcm/src/matrices.rs::build_k_r` |
> | Code test | `ief_reduces_to_conductor_equation_as_eps_goes_to_infinity` in `crates/ferric-pcm/tests/ief_limits.rs` |
>
> <!-- verified: id=pcm.conductor.limit; script=scripts/pcm_ief_sympy.py; claim=4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pcm/src/matrices.rs::build_k_r; rust_test=ferric-pcm/tests/ief_limits.rs::ief_reduces_to_conductor_equation_as_eps_goes_to_infinity; registry=pcm -->

At finite $\varepsilon$ the conductor-like models scale $Sq = -v$ by one
factor, and this is where they differ from IEF-PCM. C-PCM uses
$(\varepsilon-1)/\varepsilon$, which is Kirkwood's $k_0$ of section @sec:ief-operators:
exact for a central charge. COSMO uses $(\varepsilon-1)/(\varepsilon+\tfrac12)$,
which is Kirkwood's $k_1 = 2(\varepsilon-1)/(2\varepsilon+1)$ with the 2
cancelled: exact for a central dipole. Each is exact in one channel and
approximate in the others, and IEF-PCM is exact in all of them (the control
rows of box @eq:pcm-sphere-kirkwood). For water, $\varepsilon = 78.4$, the three factors
$f = 0.9748$, $k_0 = 0.9872$ and $k_1 = 0.9810$ differ in the second or
third digit.

## The COSMO driver {#conductor-limit-and-cosmo-cosmo-driver}

`ferric-scf/src/cosmo.rs` implements the conductor-like screening model of
Klamt and Schüürmann (J. Chem. Soc. Perkin Trans. 2, 799 (1993)) as a
density-dependent term folded into the Fock matrix every iteration, as
`solvent_terms` does for PCM (section @sec:reaction-field). It uses the IEF-PCM
machinery's conventions where it can. Per SCF iteration it computes the
segment potential, solves the symmetric system, and returns the operator
and energy.

**Segment potential.** For segments at $\mathbf s_k$ and the total AO
density $P = \sum_\sigma D_\sigma$, the potential is the point-probe one of
section @sec:reaction-field:

$$ v_k = \sum_{A} \frac{Z^{\mathrm{eff}}_A}{|\mathbf{s}_k - \mathbf{R}_A|} + \sum_{\mu\nu} P_{\mu\nu} M_{k,\mu\nu},
\qquad
M_{k,\mu\nu} = \Big\langle \mu \Big| \frac{-1}{|\mathbf{r} - \mathbf{s}_k|} \Big| \nu \Big\rangle . $$

$M_k$ is the nuclear-attraction engine block with a $+1$ probe at
$\mathbf s_k$. $Z^{\mathrm{eff}}_A$ is `Atom::effective_z`, the valence
charge on an ECP atom, and ghost atoms are skipped. The probe is a point:
the potential is not smeared over the segment's Gaussian, so PySCF's
Gaussian probe is not used.

**Charges.** With the symmetric segment matrix $S$ and the dielectric factor

$$ f_{1/2}(\varepsilon) = \frac{\varepsilon - 1}{\varepsilon + \tfrac12} = \frac{2(\varepsilon-1)}{2\varepsilon+1} , $$

the code solves $Sq = -f_{1/2}\,v$ by one LU solve
(`CosmoConfig::f_epsilon`). Check the limits: $f_{1/2}(1) = 0$, so vacuum
gives no charge, and $f_{1/2}\to 1$ as $\varepsilon\to\infty$, which is the
conductor condition $Sq = -v$ of @sec:conductor-limit-and-cosmo-limit-ief-pcm. C-PCM's $(\varepsilon-1)/\varepsilon$
is not used.

**Energy and Fock term.** With $Q = -f_{1/2}S^{-1}$ the energy is

$$ E_{\mathrm{cosmo}} = \tfrac12\,q\cdot v = -\tfrac{f_{1/2}}{2}\,v^{\mathsf T}S^{-1}v . $$

For a positive definite $S$, $v^{\mathsf T}S^{-1}v \ge 0$, so
$E_{\mathrm{cosmo}} \le 0$ at every density.
$E_{\mathrm{cosmo}}$ includes the nuclear part of $v$, so it is the full
electrostatic solvation term of the solute. Differentiate with respect to
$P_{\mu\nu}$, using $\partial v_k/\partial P_{\mu\nu} = M_{k,\mu\nu}$:

$$ V^{\mathrm{cosmo}}_{\mu\nu} = \frac{\partial E_{\mathrm{cosmo}}}{\partial P_{\mu\nu}} = -f_{1/2}\,(S^{-1}v)\cdot\frac{\partial v}{\partial P_{\mu\nu}} = \sum_k q_k M_{k,\mu\nu} . $$

The middle step uses $S = S^{\mathsf T}$. For a non-symmetric matrix the
derivative would be
$-\tfrac{f_{1/2}}{2}[(S^{-1} + S^{-\mathsf T})v]\cdot\partial v/\partial P$,
and $\sum_k q_kM_k$ would not be it. There is no factor $\tfrac12$ in
$V^{\mathrm{cosmo}}$: the $\tfrac12$ of the energy cancels against the two
factors of $v$. Because $S$ is symmetric, $q$ is already the symmetric
response and COSMO needs no symmetrization. $E_{\mathrm{cosmo}}$ depends on
the spins only through $P$, so the same $V^{\mathrm{cosmo}}$ is added to
every spin Fock matrix. $E_{\mathrm{cosmo}}$ is added to the SCF energy
outside the $\tfrac12\operatorname{Tr}[P(h+F)]$ trace, which is taken before
$V^{\mathrm{cosmo}}$ enters $F$, for the reasons given for PCM in
section @sec:reaction-field.

> **Verified equation {#eq:cosmo-reaction-field}** · the COSMO charges, energy and operator
>
> $$ S\,q = -f_{1/2}(\varepsilon)\,v, \qquad f_{1/2}(\varepsilon) = \frac{\varepsilon-1}{\varepsilon+\tfrac12}, \qquad E_{\mathrm{cosmo}} = \tfrac12\,q\cdot v, \qquad V^{\mathrm{cosmo}}_{\mu\nu} = \sum_k q_k M_{k,\mu\nu} = \frac{\partial E_{\mathrm{cosmo}}}{\partial P_{\mu\nu}} . $$
>
> The Fock term is the density derivative of the energy with no factor
> $\tfrac12$, because $S$ is symmetric: in symbols on two segments and two
> AOs (claim C1), and against a central difference of $E_{\mathrm{cosmo}}(P)$
> on a Gaussian-smeared $S$ (claim C2, where $V/2$ and a non-symmetric $S$
> miss). $f_{1/2}$ is Kirkwood's $k_1$, zero at $\varepsilon = 1$ and $1$ in
> the conductor limit (claim C3). The Gaussian $S_{kl}$ and $S_{kk}$ are the
> Coulomb energies of unit Gaussians, checked against a Fourier-space
> quadrature, and tend to the point-charge $1/r_{kl}$ (claim C4).
>
> | | |
> |---|---|
> | Proof | `scripts/cosmo_sympy.py`, claims C1 to C4 |
> | Negative control | the runner halves the Fock term (C2 fails), uses C-PCM's $(\varepsilon-1)/\varepsilon$ (C3 fails) and combines $\xi_{kl}$ wrongly (C4 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/cosmo.rs::cosmo_reaction_field` |
> | Code test | `cosmo_matches_vacuum_in_the_trivial_limit` in `crates/ferric-scf/tests/validation_cosmo.rs` |
>
> <!-- verified: id=cosmo.reaction.field; script=scripts/cosmo_sympy.py; claim=C2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/cosmo.rs::cosmo_reaction_field; rust_test=ferric-scf/tests/validation_cosmo.rs::cosmo_matches_vacuum_in_the_trivial_limit; registry=cosmo -->

The solve and the energy are three lines of `cosmo_reaction_field`:

<!-- excerpt: ferric-scf/src/cosmo.rs::cosmo_reaction_field -->
```rust
let s_mat = cavity.build_s_matrix(config.s_matrix_kind);
let rhs = -config.f_epsilon() * &v;
```

<!-- excerpt: ferric-scf/src/cosmo.rs::cosmo_reaction_field -->
```rust
let e_cosmo = 0.5 * charges.dot(&v);
```

| Symbol | Code |
|---|---|
| $S$ | `s_mat` |
| $-f_{1/2}v$ | `rhs` |
| $q$ | `charges`, from `s_mat.solve(&rhs)` |
| $E_{\mathrm{cosmo}}$ | `e_cosmo` |

The operator is built in a second pass over the segments, once $q$ is
known. Each segment's block is recomputed with a $+1$ probe and weighted by
$q_k$; the lower triangle of shell pairs is mirrored into the upper (excerpt
unchanged, from inside the shell-pair loop):

<!-- excerpt: ferric-scf/src/cosmo.rs::cosmo_reaction_field -->
```rust
let val = qk * block[i * n2 + j];
v_reaction[(o1 + i, o2 + j)] += val;
if s1 != s2 {
    v_reaction[(o2 + j, o1 + i)] += val;
}
```

`val` is $q_kM_{k,\mu\nu}$ for one $(\mu, \nu)$ of the shell pair, and
`v_reaction` is $V^{\mathrm{cosmo}}$.

### The segment matrix

**`SMatrixKind::GaussianSmeared`** (the default, PySCF's `pcm.py` form)
treats segment $k$ as the unit Gaussian
$(\xi_k^2/\pi)^{3/2}e^{-\xi_k^2|\mathbf r-\mathbf s_k|^2}$ with
$\xi_k = \Xi_{N_{\mathrm{Leb}}}/(R_A\sqrt{4\pi w_k})$ (`gaussian_xi_table`,
$w_k$ the ferric Lebedev weight, which sums to 1):

$$ S_{kl} = \frac{\operatorname{erf}(\xi_{kl} r_{kl})}{r_{kl}}, \quad \xi_{kl} = \frac{\xi_k\xi_l}{\sqrt{\xi_k^2+\xi_l^2}}, \qquad S_{kk} = \frac{\xi_k\sqrt{2/\pi}}{\mathrm{swf}_k} . $$

These are the formulas of section @sec:cavity-and-bem-matrices. The off-diagonal element is the
Coulomb energy of two Gaussians ($1/\xi_{kl}^2 = 1/\xi_k^2 + 1/\xi_l^2$), and
$\xi_k\sqrt{2/\pi}$ is a Gaussian's self-energy, the $r\to 0$ limit of
$\operatorname{erf}(\xi_k r/\sqrt2)/r$. The self-term is divided by the
segment's switching weight $\mathrm{swf}_k$, the product of the smooth
switching functions $h$ of every other atom's sphere
(`ferric_pcm::cavity::switch_h`), so a segment fading out of the surface
costs more to charge.

**`SMatrixKind::PointCharge`** has $S_{kl} = 1/r_{kl}$, the
$\xi\to\infty$ limit of the Gaussian form, and the original Klamt–Schüürmann
diagonal $S_{kk} = 3.8\sqrt{4\pi/a_k}$ (`COSMO_XI`) with the switched area
$a_k = 4\pi R_A^2 w_k\,\mathrm{swf}_k$. The constant 3.8 is a model constant,
not derived here.

### The cavity

`CosmoCavity::build` is a second implementation of the cavity of section
@sec:cavity-and-bem. It shares `ferric_pcm::cavity::switch_h` and
`ferric_pcm::matrices::gaussian_xi_table` and differs in what it feeds them.
Segments with $w_k\,\mathrm{swf}_k \le 10^{-16}$ are dropped, as for PCM.

| | IEF-PCM (`ferric-pcm`) | COSMO (`ferric-scf`) |
|---|---|---|
| radii | modified Bondi (H 1.10 Å, B 1.92 Å), untabulated elements fall back to 2.0 Å | Bondi, H 1.20 Å; an element with no entry is an error |
| scale | 1.2 | 1.17 (`DEFAULT_RADIUS_SCALE`) |
| Lebedev order | 110 | 110 |
| switching | SWIG, quintic, $\sqrt{14/N_{\mathrm{Leb}}}$ band | the same |
| $S$ | `GaussianSmeared` or `PointCharge` | `GaussianSmeared` or `PointCharge` (`SMatrixKind`) |

### Why there is no gradient

There is no COSMO nuclear gradient.
`ferric-scf/src/gradient.rs::refuse_cosmo_gradient` stops every gradient
dispatcher, geometry optimization, saddle search, IRC and
finite-difference-of-gradient frequency run on a COSMO SCF. Without it each
would follow the vacuum-form gradient of a solvated energy. The derivative
of $E_{\mathrm{cosmo}}$ with respect to the nuclei would need
$\partial S/\partial\mathbf R$ (segments move with their atoms, and
$\mathrm{swf}_k$ changes), the derivative of $M_k$ with respect to both the
basis centres and the segment positions, and the derivative of the nuclear
part of $v$. None of it is implemented. Section @sec:gradient has all of these terms
for IEF-PCM, which is the model to use for a geometry.

## The algorithm {#conductor-limit-and-cosmo-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | segments: position, area, $\xi_k$, $\mathrm{swf}_k$; once per SCF | $N_{\mathrm{at}}^2N_{\mathrm{Leb}}$ | `CosmoCavity::build` |
| 2 | $v_k$: one nuclear-attraction engine pass per segment, contracted with $P$ | $N\,N^2_{\mathrm{bf}}$ | `cosmo_reaction_field`, pass 1 |
| 3 | $S$, then $q$ from one LU solve | $N^3$ | `build_s_matrix`, `s_mat.solve` |
| 4 | $V^{\mathrm{cosmo}}$: the engine again, per segment, weighted by $q_k$ | $N\,N^2_{\mathrm{bf}}$ | `cosmo_reaction_field`, pass 2 |
| 5 | $V^{\mathrm{cosmo}}$ added to every spin Fock matrix, $E_{\mathrm{cosmo}}$ to the energy | | `solvent_terms` |

The two passes recompute the integrals rather than cache one
$N_{\mathrm{bf}}^2$ matrix per segment, which would cost $N\,N^2_{\mathrm{bf}}$
memory. The loops in `cosmo_reaction_field` are serial, with none of the
thread parallelism of the PCM operator in section @sec:reaction-field, and $S$ is rebuilt
every iteration.

## Options {#conductor-limit-and-cosmo-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[cosmo] epsilon` | float $> 1$ | 78.39 | $\varepsilon$; $\le 1$ or non-finite is an error |
| `[cosmo] radius_scale` | float $> 0$ | 1.17 | scale on the Bondi radii |
| `[cosmo] lebedev_order` | a positive-weight Lebedev order | 110 | segments per sphere |
| `[cosmo] s_matrix_kind` | `GaussianSmeared`, `PointCharge` | `GaussianSmeared` | the $S$ formulation |
| `[cosmo]` with `[pcm]` | | | refused: two models of one solvent would be added |
| `[cosmo]` with a task other than `energy` | | | refused: there is no gradient |

## Numerical anchors {#conductor-limit-and-cosmo-numerical-anchors}

| Quantity | Systems | Result | Source |
|---|---|---|---|
| solvated energy and $E_{\mathrm{cosmo}}$ against PySCF's COSMO driver with ferric's radii, cavity and point-charge potential | H2O, NH3, CH3OH, acetate($-$) / STO-3G, cc-pVDZ; $\varepsilon = 4.7, 78.4$; HO2 (UHF) / cc-pVDZ | $\le 1.9\times10^{-11}$ Ha and $\le 9.7\times10^{-11}$ Ha | `site/src/reference/validation.md`, COSMO row |
| against stock PySCF COSMO (point against Gaussian surface charges) | same | 0.05 to 0.51 % | same |
| solvation energy | water / cc-pVDZ, $\varepsilon = 78.39$ | $-5.955$ kcal/mol against PySCF COSMO $-5.94$ (the Gaussian-smeared $S$; the bare point-charge $S$ gives $-3.39$) | `crates/ferric-scf/tests/cosmo_water.rs`, `site/src/methods/scf.md` |

The first row compares the same model on both sides. The second measures
the formulation difference to stock PySCF, whose potential is Gaussian. The
third shows how much the $S$ formulation matters: the point-charge $S$,
with the Klamt–Schüürmann diagonal, gives about 57 % of the
Gaussian-smeared solvation energy.

## Sources {#conductor-limit-and-cosmo-sources}

| Source | What it gives |
|---|---|
| `crates/ferric-pcm/theory.md`, section 6 | the conductor limit and the finite-$\varepsilon$ factors |
| `scripts/pcm_ief_sympy.py`, claims 3 and 4 | the limit and the Kirkwood comparison |
| `crates/ferric-scf/theory.md`, "COSMO reaction field" | the driver's equations, the $S$ matrices and why there is no gradient |
| `scripts/cosmo_sympy.py`, claims C1 to C4 | the Fock term, $f(\varepsilon)$ and the Gaussian $S$ |
| `crates/ferric-scf/src/cosmo.rs` | the driver |
| `crates/ferric-scf/tests/validation_cosmo.rs`, `crates/ferric-scf/tests/cosmo_gradient_refusal.rs` | the PySCF comparison and the gradient refusal |

**Known limits (2026-11-15).**

- Energy only. `refuse_cosmo_gradient` stops every gradient dispatcher,
  geometry optimization, saddle search, IRC and finite-difference frequency
  run on a COSMO SCF, because each would follow the vacuum-form gradient of
  a solvated energy.
- No outlying-charge correction, no cavitation, dispersion or repulsion
  terms, and no re-tessellation of partly buried spheres beyond the smooth
  switch.
- COSMO is reachable from the CLI and the Rust library only; the Python
  drivers take `solvent=` for IEF-PCM.
