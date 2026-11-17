# The nuclear gradient of the PCM energy {#gradient}

<!-- status: ready; registry: pcm-gradient -->

A geometry optimization in a continuum needs the force on every nucleus,
including the part that comes from the cavity moving with the atoms. The
SCF energy is variational in the density, so the solvated gradient is the
vacuum SCF gradient of section @sec:hf-gradient, evaluated with the solvated orbitals
and energy-weighted density, plus one term: the derivative of
$E_{\mathrm{pcm}}$ at fixed AO density $P$. The solvated orbitals already
carry $V^{\mathrm{pcm}}$ through the Fock matrix, so nothing else changes.
This section is that one term.

The plan: split the derivative into a potential part and a response part
(@sec:gradient-derivative), derive each, then the algorithm (@sec:gradient-algorithm), the refusals (@sec:gradient-options-refusals)
and the anchors (@sec:gradient-numerical-anchors).

| Symbol | Meaning | Code name |
|---|---|---|
| $x$, $y$ | $K^{-1}Rv$ and $K^{-\mathsf T}v$ | `x`, `y` |
| $q$ | $\tfrac12(x + R^{\mathsf T}y)$ | `q` |
| $w$, $u$, $z$ | $Sx+v$, $Aw$, $y - \tfrac{f}{2\pi}AD^{\mathsf T}y$ | `w`, `u`, `z` |
| $p$ | $f/2\pi$ | `pref` |
| $A(k)$ | atom that carries tessera $k$ | `Tessera::atom_index` |
| $h_{\mathrm{sw}}$ | the quintic switch of section @sec:cavity-and-bem | `switch_h`, `switch_h_deriv` |

| Entry point | Location |
|---|---|
| Rust | `ferric-pcm/src/gradient.rs::{pcm_gradient,pcm_gradient_parts}`, summed into the SCF gradient by `ferric-scf/src/gradient.rs::with_pcm_gradient` |
| CLI | `[pcm]` with `task = "optimize"` on `rhf`, `uhf`, `rohf` and `ksdft` |
| Python | `run_optimize(solvent=...)` |

## The derivative {#gradient-derivative}

Write the PCM energy as a function of a nuclear coordinate $R_{A\alpha}$ at
fixed $P$: $E_{\mathrm{pcm}} = \tfrac12 v^{\mathsf T}Q_{\mathrm{sym}}v$, with
both $v$ and $Q_{\mathrm{sym}}$ depending on the geometry. Differentiate
with the product rule. The $v$ factors give $(Q_{\mathrm{sym}}v)\cdot\partial v = q\cdot\partial v$,
as in section @sec:reaction-field. The middle factor gives
$\tfrac12 v^{\mathsf T}\partial Q_{\mathrm{sym}}v$, which equals
$\tfrac12 v^{\mathsf T}\partial Q\,v$ for the same reason the energies of
section @sec:ief-operators agree. With $x = K^{-1}Rv$, $y = K^{-\mathsf T}v$ and
$q = \tfrac12(x + R^{\mathsf T}y)$ (`pcm_gradient_parts`):

$$ \frac{\partial E_{\mathrm{pcm}}}{\partial R_{A\alpha}} = q^{\mathsf T}\frac{\partial v}{\partial R_{A\alpha}} + \tfrac12\, v^{\mathsf T}\frac{\partial (K^{-1}R)}{\partial R_{A\alpha}}\, v . $$

The first term is the force on the fixed charges $q_k$ from the solute's
nuclei and electrons. The second is the change of the operator $K^{-1}R$
itself, because $S$, $D$ and the areas depend on where the tesserae are.

How do the tesserae move? Each one rides rigidly on the atom of its sphere:
$\partial\mathbf r_k/\partial\mathbf R_B = \delta_{A(k)B}\,I$. The normals
$\hat{\mathbf n}_k$, the widths $\xi_k$ and the sphere radii do not depend
on the geometry. The areas and the diagonal $S_{kk}$ depend on it only
through $\mathrm{swf}_k$, the switch that measures how deep the tessera sits
inside its neighbours' spheres.

### The potential term

$q^{\mathsf T}\partial v$ is the gradient of the interaction of fixed
charges $q_k$ at $\mathbf r_k$ with the solute. It has an electronic and a
classical part.

For point probes (`potential_term_point`) the electronic part is
$\sum_{\mu\nu}P_{\mu\nu}\,\partial V^{\mathrm{pcm}}_{\mu\nu}(q)$. Libint's
nuclear-attraction derivative engine computes it with the tesserae as the
only charges. The derivative blocks of the two shell centres go to the
atoms of the AOs, and the block of charge $k$ goes to $A(k)$. The classical
part is

$$ \sum_k q_k \sum_{B\ne A(k)} Z_B\,\phi(|\mathbf r_k-\mathbf R_B|), \qquad \phi(r) = 1/r , $$

and its derivative moves $A(k)$ against $B$ (`nuclear_tessera_term`). The
own atom $B = A(k)$ is left out: the tessera stays at the fixed distance
$R_{A(k)}$ from it, so that term does not change. For Gaussian probes
(`potential_term_smeared`) $\phi(r) = \operatorname{erf}(\xi_k r)/r$, and the
electronic part uses the derivative three-centre integrals
$\partial(\mu\nu|g_k)/N_k$, with the site block going to $A(k)$.

### The response term

Differentiate the inverse first: $\partial(K^{-1}) = -K^{-1}(\partial K)K^{-1}$,
from $\partial(K^{-1}K) = 0$. So

$$ \tfrac12 v^{\mathsf T}\partial(K^{-1}R)\,v = \tfrac12\left[-y^{\mathsf T}(\partial K)\,x + y^{\mathsf T}(\partial R)\,v\right] , $$

where $y^{\mathsf T} = v^{\mathsf T}K^{-1}$ and $x = K^{-1}Rv$ absorbed the
outer factors. Write $p = f/2\pi$, so that $K = (I - pDA)S$ and
$R = -f(I - DA/2\pi)$. By the product rule,

$$ \partial K = (I - pDA)\,\partial S - p\,\partial(DA)\,S, \qquad \partial R = p\,\partial(DA) . $$

Substitute. The $\partial(DA)$ terms collect into
$p\,y^{\mathsf T}\partial(DA)(Sx + v)$. Then expand
$\partial(DA) = (\partial D)A + D\,\partial A$. With the intermediates

$$ w = Sx + v, \qquad u = Aw, \qquad z = y - p\,A D^{\mathsf T} y , $$

where $z^{\mathsf T} = y^{\mathsf T}(I - pDA)$, the result is
(`ferric-pcm/src/gradient.rs::response_term`)

$$ \tfrac12 v^{\mathsf T}\partial(K^{-1}R)v = \tfrac12\Big[ -z^{\mathsf T}(\partial S)\,x + p\, y^{\mathsf T}(\partial D)\,u + p\sum_j (D^{\mathsf T}y)_j\, w_j\, \partial a_j \Big] . $$

Box @eq:pcm-gradient-response certifies this formula.

> **Verified equation {#eq:pcm-gradient-response}** · the response derivative of $\tfrac12v^{\mathsf T}K^{-1}Rv$
>
> For $S(t)$, $D(t)$ and $A(t)$ depending on a nuclear coordinate $t$ at
> fixed $v$, $E = \tfrac12v^{\mathsf T}K^{-1}Rv$ has derivative
> $\tfrac12\big[-z^{\mathsf T}S'x + p\,y^{\mathsf T}D'u + p\sum_j(D^{\mathsf T}y)_jw_jA'_{jj}\big]$,
> with $x$, $y$, $w$, $u$, $z$ as in the table above. The check is exact
> ($n = 3$) against the $t$-derivative of the energy.
>
> | | |
> |---|---|
> | Proof | `scripts/pcm_ief_sympy.py`, claim 8 |
> | Negative control | dropping the area term, or using $q_{\mathrm{sym}}$ for $x$, breaks it; the runner flips the sign in $z$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-pcm/src/gradient.rs::response_term` |
> | Code test | `fixed_density_gradient_matches_fd_water_default` in `crates/ferric-pcm/tests/pcm_gradient.rs` |
>
> <!-- verified: id=pcm.gradient.response; script=scripts/pcm_ief_sympy.py; claim=8; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-pcm/src/gradient.rs::response_term; rust_test=ferric-pcm/tests/pcm_gradient.rs::fixed_density_gradient_matches_fd_water_default; registry=pcm-gradient -->

Check the vacuum limit: at $\varepsilon\to 1$, $f\to 0$, so $R = 0$, $x = 0$,
$q = 0$ and $p = 0$. Both terms vanish and the solvated gradient becomes the
vacuum one. The anchor at $\varepsilon = 1 + 2\times10^{-9}$ in @sec:gradient-numerical-anchors tests
exactly this.

What is left is the derivative of each matrix element. The off-diagonal
elements depend on $\mathbf r_{ij}$ alone. Write $S_{ij} = S(r_{ij})$ and
$D_{ij} = h(r_{ij})\,\nu_{ij}$ with $h(r) = -S'(r)/r$, the source-normal
derivative of section @sec:cavity-and-bem (`sd_kernel`). The gradients with respect to the
field point follow from $\nabla_{\mathbf r_i}r_{ij} = \hat{\mathbf r}_{ij}$
and $\nabla_{\mathbf r_i}\nu_{ij} = \hat{\mathbf n}_j$:

$$ \nabla_{\mathbf r_i} S_{ij} = S'(r)\,\hat{\mathbf r}_{ij}, \qquad
   \nabla_{\mathbf r_i} D_{ij} = h'(r)\,\nu_{ij}\,\hat{\mathbf r}_{ij} + h(r)\,\hat{\mathbf n}_j , $$

and $\nabla_{\mathbf r_j} = -\nabla_{\mathbf r_i}$, since the normal does
not move. A pair on one sphere moves rigidly with one atom, so its
contributions cancel and the code skips it. For `GaussianSmeared`, with
$c = 2/\sqrt\pi$ and $\xi = \xi_{ij}$, differentiate the erf kernel:

$$ S' = \frac{c\,\xi e^{-\xi^2r^2}}{r} - \frac{\operatorname{erf}(\xi r)}{r^2}, \qquad
   h' = \frac{3c\,\xi e^{-\xi^2r^2}}{r^3} - \frac{3\operatorname{erf}(\xi r)}{r^4} + \frac{2c\,\xi^3 e^{-\xi^2r^2}}{r} . $$

For `PointCharge`, $S' = -1/r^2$, $h = 1/r^3$ and $h' = -3/r^4$.

The diagonal and area terms are all proportional to
$\partial\ln\mathrm{swf}_j$. The area is $a_j \propto \mathrm{swf}_j$, so
$\partial a_j = a_j\,\partial\ln\mathrm{swf}_j$. For `GaussianSmeared`,
$S_{jj} \propto 1/\mathrm{swf}_j$ gives
$\partial S_{jj} = -S_{jj}\,\partial\ln\mathrm{swf}_j$, and $D_{jj}$ is
constant. For `PointCharge`, $S_{jj} \propto a_j^{-1/2}$ gives
$\partial S_{jj} = -\tfrac12 S_{jj}\,\partial\ln\mathrm{swf}_j$, and likewise
$\partial D_{jj} = -\tfrac12 D_{jj}\,\partial\ln\mathrm{swf}_j$. The switch is
a product, so its logarithm is a sum, and each factor differentiates by the
chain rule:

$$ \nabla_{\mathbf r_j}\ln\mathrm{swf}_j = \sum_{B\ne A(j)} \frac{h_{\mathrm{sw}}'(t_{jB})}{h_{\mathrm{sw}}(t_{jB})}\,\frac{1}{R^{\mathrm{sw}}_B}\,\hat{\mathbf r}_{jB},
   \qquad t_{jB} = \frac{|\mathbf r_j-\mathbf R_B| - R^{\mathrm{in}}_B}{R^{\mathrm{sw}}_B} , $$

with $h_{\mathrm{sw}}'(t) = 30t^2(1-t)^2$ (`switch_h_deriv`). Atom $A(j)$
receives $+$ and atom $B$ receives $-$. $R^{\mathrm{sw}}$ and $R^{\mathrm{in}}$
come from the same `SphereGeometry::new` that `build_cavity` uses, so the
cavity and its derivative cannot disagree about the band. The switch plays
the role of the Becke weight of section @sec:xc-gradient-becke-weight-derivative: a smooth partition function
whose derivative moves two atoms in opposite directions.

### The code

`response_term` builds the four vectors once (excerpt unchanged):

<!-- excerpt: ferric-pcm/src/gradient.rs::response_term -->
```rust
let pref = f_eps / (2.0 * PI);
let area = Array1::from_iter(tess.iter().map(|t| t.area));
let w = s.dot(x) + v;
let u = &area * &w;
let dty = d.t().dot(y);
let z = y - &(pref * &area * &dty);
```

The first loop runs over every ordered pair $(i, j)$ on different spheres
and contracts the kernel derivatives. The inner part, unchanged:

<!-- excerpt: ferric-pcm/src/gradient.rs::response_term -->
```rust
let (sp, h, hp) = sd_kernel(kind, xi_ij, rr);
let nu = rij[0] * tj.normal[0] + rij[1] * tj.normal[1] + rij[2] * tj.normal[2];
let cs = -0.5 * z[i] * x[j];
let cd = 0.5 * pref * y[i] * u[j];
// ∂/∂r_i of cs·S_ij + cd·D_ij.
let mut gi = [0.0; 3];
for ax in 0..3 {
    let rhat = rij[ax] / rr;
    gi[ax] = cs * sp * rhat + cd * (hp * nu * rhat + h * tj.normal[ax]);
}
push_pair(&mut g, ti.atom_index, tj.atom_index, 1.0, gi);
```

The second loop gathers every term proportional to $\partial\ln\mathrm{swf}_j$
into one coefficient per tessera:

<!-- excerpt: ferric-pcm/src/gradient.rs::response_term -->
```rust
let diag = match kind {
    // S_jj = ξ_j √(2/π) / swf_j.
    SdKind::GaussianSmeared => 0.5 * z[j] * x[j] * s[(j, j)],
    // S_jj = 1.0694 √(4π / a_j) and D_jj = −S_jj / (2 R_j), a_j ∝ swf_j.
    SdKind::PointCharge => {
        0.25 * z[j] * x[j] * s[(j, j)] - 0.25 * pref * y[j] * u[j] * d[(j, j)]
    }
};
let c = diag + 0.5 * pref * dty[j] * w[j] * tj.area;
```

| Code | Equation |
|---|---|
| `pref` | $p = f/2\pi$ |
| `w`, `u`, `dty`, `z` | $w = Sx + v$, $u = Aw$, $D^{\mathsf T}y$, $z = y - pAD^{\mathsf T}y$ |
| `sp`, `h`, `hp` | $S'(r)$, $h(r)$, $h'(r)$ |
| `nu` | $\nu_{ij}$ |
| `cs` | weight of $\partial S_{ij}$: $-\tfrac12 z_ix_j$ |
| `cd` | weight of $\partial D_{ij}$: $\tfrac12 p\,y_iu_j$ |
| `gi` | $\nabla_{\mathbf r_i}$ of $c_s S_{ij} + c_d D_{ij}$ |
| `diag` (Gaussian) | $-\tfrac12 z_jx_j\,\partial S_{jj}$ per unit $\partial\ln\mathrm{swf}_j$: $+\tfrac12 z_jx_jS_{jj}$ |
| `diag` (point) | $+\tfrac14 z_jx_jS_{jj} - \tfrac14 p\,y_ju_jD_{jj}$ |
| `c` | `diag` plus the area term $\tfrac12 p\,(D^{\mathsf T}y)_jw_ja_j$ |

Three points about this code:

- `push_pair` adds a vector to one atom and subtracts it from the other.
  Every contribution is a function of a difference of two positions, so
  the gradient sums to zero over the atoms pair by pair, cavity included.
- `cs` and `cd` multiply the derivatives with respect to $\mathbf r_i$; the
  $\mathbf r_j$ derivative is the same vector with the opposite sign, which
  is what `push_pair` applies to `tj.atom_index`.
- A tessera outside atom $B$'s switching band has $h_{\mathrm{sw}}' = 0$
  for $B$ and contributes nothing for that atom; the loop skips it before
  dividing by $h_{\mathrm{sw}}$.

The assembled gradient is added to the SCF gradient of the solvated
orbitals by `with_pcm_gradient`, which `restricted_scf_gradient`,
`unrestricted_scf_gradient` and `restricted_open_scf_gradient` call:

<!-- excerpt: ferric-scf/src/gradient.rs::with_pcm_gradient -->
```rust
grad += &ferric_pcm::pcm_gradient(mol, prep, result.density_total(), pcm)?;
```

Box @eq:pcm-gradient-response proves the response formula on a model with $S$, $D$ and $A$ as
functions of one coordinate. It does not prove the kernel derivatives
$S'(r)$, $h(r)$, $h'(r)$ or the switching derivative, which are transcribed
from the code. Those are covered by the finite-difference tests in the
anchors table, and by nine single-term mutants of the Rust code that each
fail them (section @sec:gradient-numerical-anchors).

## The algorithm {#gradient-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | cavity, $S$, $D$, $K$, $R$, $v$ at the current geometry and density | as one SCF iteration | `pcm_gradient_parts` |
| 2 | $x$, $y$, $q$ by two LU solves | $N^3$ | `pcm_gradient_parts` |
| 3 | potential term $q^{\mathsf T}\partial v$: nuclei, AO centres, tessera positions | $N\,N^2_{\mathrm{bf}}$ with libint derivative engines | `potential_term_point`, `potential_term_smeared`, `nuclear_tessera_term` |
| 4 | response term: kernel derivatives over all off-sphere ordered pairs | $N^2$ | `response_term` |
| 5 | switching derivative $\nabla\ln\mathrm{swf}_j$ over tesserae and atoms | $N\,N_{\mathrm{at}}$ | `response_term`, `switch_h_deriv` |
| 6 | add to the solvated SCF gradient | | `with_pcm_gradient` |

`pcm_gradient_parts` rebuilds the cavity from the molecule exactly as
`PcmContext::new` does, and re-solves the charges at the converged density.
It does not reuse the SCF's factorization.

## Options and refusals {#gradient-options-refusals}

| Case | Behaviour |
|---|---|
| `PcmConfig::cavity` set (an injected tessera list) | error: the list carries no record of how it moves |
| `sd_kind`, `probe` | both `PointCharge` and `GaussianSmeared` $S$/$D$, both point and Gaussian probes |
| restricted, unrestricted and restricted-open SCF, and KS | the same term is added by `restricted_scf_gradient`, `unrestricted_scf_gradient` and `restricted_open_scf_gradient` |
| COSMO | `ferric-scf/src/gradient.rs::refuse_cosmo_gradient` refuses (section @sec:conductor-limit-and-cosmo) |
| frequencies with `[pcm]` | refused by the CLI; there is no analytic Hessian |

## Numerical anchors {#gradient-numerical-anchors}

The finite-difference tests are `crates/ferric-pcm/tests/pcm_gradient.rs`
(fixed $P$, central differences of `pcm_step`'s energy) and
`crates/ferric-scf/tests/pcm_gradient.rs` (the solvated SCF energy). The
bar at fixed $P$ is the change between steps $h$ and $2h$ plus a round-off
floor; the SCF-level bar adds $4\,\delta E/h$.

| Check | System | Max error (Ha/Bohr) | Source |
|---|---|---|---|
| PySCF 2.13 `nuc_grad_method()`, IEF-PCM, 302 points, Gaussian probes | water, methanol / STO-3G | $4.2\times10^{-11}$, $7.2\times10^{-11}$ | `site/src/reference/validation.md`, IEF-PCM gradient row |
| central FD at fixed density | water, C1 methanol / STO-3G | $\le 2.0\times10^{-7}$ | same |
| central FD of the solvated SCF energy | same, RHF, RKS, UHF, ROHF | $\le 2.9\times10^{-7}$ | same |
| torque against FD of a rigid rotation | C1 methanol / STO-3G | $8\times10^{-9}$ (torque $1.36\times10^{-4}$) | same |
| $\varepsilon = 1 + 2\times10^{-9}$ against the vacuum gradient | same | $3.3\times10^{-12}$ | same |

The individual finite-difference cases behind the summary rows:

| Check | System | Max error (Ha/Bohr) |
|---|---|---:|
| FD, fixed $P$, default kinds, Lebedev 110 | water/STO-3G | 5.7e-8 |
| FD, fixed $P$, point / Gaussian probes, Lebedev 50 | C1 methanol/STO-3G | 2.0e-7 / 2.0e-7 |
| FD, fixed $P$, `PointCharge` $S$/$D$, $h = 2.5\times10^{-4}$ | water/STO-3G | 8.6e-7 |
| FD of the solvated RHF energy | water/STO-3G | 1.4e-7 |
| FD of the solvated RHF energy, Lebedev 50 | C1 methanol/STO-3G | 2.9e-7 |
| FD of the solvated RKS (SVWN) energy, Gaussian probes | water/STO-3G | 1.4e-7 |
| FD of the solvated UHF / ROHF energy | NH2/STO-3G | 1.3e-7 / 1.3e-7 |
| torque against FD rigid rotation, Lebedev 50 | C1 methanol/STO-3G | 8.0e-9 |

Each part of the gradient also sums to zero over the atoms, cavity
included. The torque is not zero, because the Lebedev frame is fixed in the
lab; the test checks that it equals the finite-difference derivative under
a rigid rotation. The PySCF comparison runs on ferric's own cavity at
Lebedev order 302, which reproduces PySCF's `gen_surface` (tessera count,
total area and area centroid to $10^{-10}$, section @sec:cavity-and-bem). Nine single-term
mutants (the switching, area, $S$ and $D$ diagonal and off-diagonal,
tessera-site and nuclear-tessera terms) each fail the FD tests.

## Sources {#gradient-sources}

| Source | What it gives |
|---|---|
| `scripts/pcm_ief_sympy.py`, claim 8 | the response formula on a model |
| `crates/ferric-pcm/tests/pcm_gradient.rs`, `crates/ferric-scf/tests/pcm_gradient.rs` | FD at fixed density and at SCF level |

**Known limits (2026-11-15).**

- Analytic first derivatives only: no analytic nuclear Hessian, and the CLI
  refuses frequencies with `[pcm]`.
- `PointCharge` $S$/$D$ energies are far rougher in the switching zone than
  the default. On C1 methanol at Lebedev 50 (fixed model density) gradient
  components reach about 10 Ha/Bohr, and central differences are not yet
  in their $O(h^2)$ regime at $h = 2.5\times10^{-4}$
  (section @sec:gradient-numerical-anchors).
