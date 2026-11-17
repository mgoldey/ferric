# Geometry optimization, saddle points and reaction paths {#geometry-search}

<!-- status: edit; registry: geometry-search;
     theory: crates/ferric-scf/theory.md#"Harmonic frequencies"; crates/ferric-scf/theory.md#"Geometry searches" -->

A chemist asks three questions of a potential energy surface. Where are the
stable structures, the minima, where every small displacement raises the
energy? Where is the transition state, the first-order saddle point that is
the highest point on the lowest path between two minima? Which two minima
does a given saddle connect, found by walking downhill from it both ways
along the intrinsic reaction coordinate? Each question is a search driven by
the gradient of section @sec:hf-gradient and, for saddles, the Hessian of section @sec:hf-hessian.

The plan: minimization (@sec:geometry-search-minimization), its starting Hessian and internal
coordinates (@sec:geometry-search-lindh-model-hessian), saddles (@sec:geometry-search-saddle-points-p-rfo), reaction paths (@sec:geometry-search-reaction-paths-mass-weighted), then
reproducibility, options and anchors.

A saddle is a maximum along the reaction direction and a minimum along
every other. A minimizer follows the gradient downhill with a quasi-Newton
model that stays positive definite. A saddle search must break exactly that
property: it climbs along one Hessian mode and descends along all others. A
reaction-path walk follows the mass-weighted gradient down from the saddle.
The step formulas are machine-checked by
`scripts/geometry_search_steps_sympy.py`, which carries its own mutations in
the runner.

| Symbol | Meaning | Code name |
|---|---|---|
| $x$, $g$ | Cartesian coordinates and gradient, flattened to length $3N$ | `x`, `grad` |
| $H$ | inverse Hessian estimate of the minimizer | `h`, `h_inv` |
| $s$, $y$ | achieved step and gradient change | `s`, `y` |
| $B$ | Hessian estimate of the saddle search, Hartree/Bohr²/u | `hess` |
| $b_i$, $g_i$ | projected mass-weighted Hessian eigenvalue and gradient component | `evals`, `g_q` |
| $\lambda_p$, $\lambda_n$ | P-RFO shifts for the followed mode and for the rest | `lambda_p`, `lambda_n` |
| $q = \sqrt m\,x$ | mass-weighted coordinates | `q`, `sm` |

| Entry point | Location |
|---|---|
| Rust, minimum | `ferric-scf/src/optimize.rs::{optimize_geometry,optimize_coordinates}`; open shell `optimize_geometry_uhf`, `optimize_geometry_rohf` |
| Rust, saddle | `ferric-scf/src/saddle.rs::find_saddle` |
| Rust, path | `ferric-scf/src/irc.rs::follow_irc` |
| CLI | `task = "optimize"`, section `[optimize]` (`max_steps`, `g_max_thresh`, `g_rms_thresh`, `e_conv`, `trust_radius`, `coordinates`, `initial_hessian`, `trust_update`, `scf_warm_start`) |
| Python | `run_optimize`, `run_saddle`, `run_irc` |

## Minimization {#geometry-search-minimization}

### The quasi-Newton step

Near a minimum the energy is close to a quadratic,
$E(x + s) \approx E + g^{\mathsf T}s + \tfrac12 s^{\mathsf T}Bs$, with $B$ the
Hessian. Setting the derivative of that model to zero gives the Newton step
$s = -B^{-1}g$. An exact Hessian costs $6N+1$ gradients by finite
differences, so a minimizer does not compute one. It builds an estimate from
the gradients it has already paid for. After a step $s = x_{k+1}-x_k$ the
gradient changed by $y = g_{k+1} - g_k$. A quadratic with Hessian $B'$ would
change the gradient by $B's$, so the new estimate is required to satisfy the
secant condition $B's = y$, or, for the inverse $H = B^{-1}$, $H'y = s$.

ferric runs BFGS on the inverse Hessian $H$, so that each step is a matrix
product and never a linear solve. The default starts from the inverse of
the Lindh model Hessian (`initial_hessian = "lindh"`, section @sec:geometry-search-lindh-model-hessian).
`"simple"` starts from the identity. There is no line search. Each step is
$p = -Hg$, scaled uniformly onto the trust radius $\Delta$ when it is longer:

$$ s = \alpha p,\qquad \alpha = \min\!\Big(1, \frac{\Delta}{\lVert p\rVert}\Big) . $$

### The trust radius

The quadratic model is only good near the current point. The trust radius
$\Delta$ is how far the search believes it. With `trust_update = "fixed"`,
$\Delta = r_{\rm trust}$ and the new point is always accepted. With
`"ratio"`, the default, `ferric-scf/src/optimize.rs::ratio_trust_update`
compares the energy change the model predicted with the one the SCF
measured (Nocedal and Wright, *Numerical Optimization*, Algorithm 4.1 with
$\eta = 0$).

The predicted change needs no linear solve. The model is
$m(s) = g^{\mathsf T}s + \tfrac12 s^{\mathsf T}Bs$ with $B = H^{-1}$. Substitute
$s = \alpha p = -\alpha Hg$: then $Bs = -\alpha BHg = -\alpha g$, so
$s^{\mathsf T}Bs = -\alpha^2 g^{\mathsf T}p$, and

$$ \Delta E_{\rm pred} = \alpha\, g^{\mathsf T}p + \tfrac12(-\alpha^2 g^{\mathsf T}p) = \alpha\, g^{\mathsf T}p\,\big(1 - \tfrac{\alpha}{2}\big) < 0 . $$

It is negative because $g^{\mathsf T}p = -g^{\mathsf T}Hg < 0$ for positive
definite $H$ and $0 < \alpha \le 1$.

With $r = \Delta E/\Delta E_{\rm pred}$ the radius moves as follows:

- $r < \tfrac14$: the model was poor; the radius becomes $\tfrac14\lVert s\rVert$;
- $r > \tfrac34$ and $\lVert s\rVert \ge 0.99\,\Delta$ (the step hit the
  boundary): the radius doubles to $2\Delta$;
- otherwise it stays;
- always clamped to $[0.01, 10]\,r_{\rm trust}$.

The step is kept when $\Delta E < 10^{-8}$ Ha: the energy fell, or rose by
less than the noise of an SCF energy. A rejected step costs its evaluation,
the search stays at the old point, and its $(s, y)$ pair still updates $H$.
When both $\lvert\Delta E\rvert$ and $\lvert\Delta E_{\rm pred}\rvert$ are
below $10^{-8}$ the ratio is noise and $\Delta$ is kept. The radius is also
shrunk, never grown, when the step is rejected or when
$\Delta E_{\rm pred} \ge 0$.
A rise predicted as a rise has $r > 0$ and would otherwise double the radius
on a rejected step. In Cartesians $\Delta E_{\rm pred} < 0$ always (above), so
the $\Delta E_{\rm pred} \ge 0$ clause can only fire in internal coordinates
(@sec:geometry-search-lindh-model-hessian).

<!-- excerpt: ferric-scf/src/optimize.rs::ratio_trust_update -->
```rust
let accept = actual < TRUST_ENERGY_NOISE;
if predicted.abs() < TRUST_ENERGY_NOISE && actual.abs() < TRUST_ENERGY_NOISE {
    return (accept, radius);
}
let ratio = actual / predicted;
let next = if !accept || predicted >= 0.0 || ratio < TRUST_SHRINK_RATIO {
    TRUST_SHRINK_RATIO * step_norm
} else if ratio > TRUST_GROW_RATIO && step_norm >= 0.99 * radius {
    2.0 * radius
} else {
    radius
};
(accept, next.clamp(r_min, r_max))
```

| Code | Rule |
|---|---|
| `TRUST_ENERGY_NOISE` | $10^{-8}$ Ha, the accept threshold and the noise floor |
| `TRUST_SHRINK_RATIO` | $\tfrac14$, both the ratio threshold and the shrink factor on $\lVert s\rVert$ |
| `TRUST_GROW_RATIO` | $\tfrac34$ |
| `r_min`, `r_max` | $0.01\,r_{\rm trust}$ and $10\,r_{\rm trust}$ in Cartesians |

### The BFGS update

With $\rho = 1/(y^{\mathsf T}s)$ the inverse update is

$$ p = -H g, \qquad s = p\,\min\!\Big(1, \frac{r_{\mathrm{trust}}}{\lVert p\rVert}\Big), \qquad
H' = H + \frac{y^{\mathsf T}s + y^{\mathsf T}Hy}{(y^{\mathsf T}s)^2}\,s\,s^{\mathsf T} - \frac{H y\,s^{\mathsf T} + s\,y^{\mathsf T}H}{y^{\mathsf T}s} . \label{eq:geom-bfgs} $$

It is the inverse of the direct BFGS update of $B = H^{-1}$,

$$ B' = B - \frac{Bss^{\mathsf T}B}{s^{\mathsf T}Bs} + \frac{yy^{\mathsf T}}{y^{\mathsf T}s} , $$

by the Sherman–Morrison–Woodbury formula. It satisfies the secant condition
$H'y = s$, and it keeps $H$ positive definite when $y^{\mathsf T}s > 0$. A
positive-definite $H$ is what makes $p = -Hg$ a descent direction:
$g^{\mathsf T}p = -g^{\mathsf T}Hg < 0$.

### The curvature guard

What if $y^{\mathsf T}s \le 0$? That happens when the step crossed a region
of negative curvature, for example a start on the wrong side of an
inflection. Multiply the secant condition $H'y = s$ on the left by
$y^{\mathsf T}$: every update that satisfies it has $y^{\mathsf T}H'y = y^{\mathsf T}s$.
With $y^{\mathsf T}s \le 0$ the matrix $H'$ has a non-positive value of
$y^{\mathsf T}H'y$, so no positive-definite $H'$ satisfies the condition, and
a later step $-H'g$ can point uphill. Under `"fixed"` every step is
accepted, so nothing downstream would catch that.

The update is therefore guarded
(`ferric-scf/src/optimize.rs::guarded_inverse_bfgs_update`):

$$ H' = \begin{cases} H & \lvert y^{\mathsf T}s\rvert \le 10^{-12}:\ \text{reset to the seed instead},\\ H & y^{\mathsf T}s \le \varepsilon\,\lVert y\rVert\,\lVert s\rVert,\ \ \varepsilon = 10^{-8}:\ \text{skipped},\\ \text{the update above} & \text{otherwise.} \end{cases} $$

When $\lvert y^{\mathsf T}s\rvert \le 10^{-12}$ the update is skipped and $H$
is reset to the starting Hessian. When $y^{\mathsf T}s \le
10^{-8}\lVert y\rVert\lVert s\rVert$, negative curvature along the step, the
update is skipped and $H$ is kept, with a printed note. A skipped update
leaves $H$ positive definite by induction from the starting Hessian, so
every step is a descent direction. Above the threshold the guard changes
nothing, bit for bit.

Skipping was chosen over Powell damping. Damping replaces $y$ by
$\theta y + (1-\theta)Bs$ so that $y^{\mathsf T}s$ becomes
$0.2\,s^{\mathsf T}Bs$. On the triplet-water optimizations where
$y^{\mathsf T}s < 0$ occurs, damping converged in the same or slightly more
steps (58 vs 58 UMP2, 91 vs 88 UHF/STO-3G). It also needs $Bs = H^{-1}s$, a
linear solve with the carried inverse. The saddle search does not use this
update: it carries an indefinite $B$ with the Bofill update of @sec:geometry-search-saddle-points-p-rfo. The
IRC walk carries no Hessian.

The guard and the update are the body of `guarded_inverse_bfgs_update`:

<!-- excerpt: ferric-scf/src/optimize.rs::guarded_inverse_bfgs_update -->
```rust
let ys = y.dot(s);
if ys.abs() <= 1e-12 {
    return BfgsUpdate::Degenerate;
}
if ys <= CURVATURE_EPS * y.dot(y).sqrt() * s.dot(s).sqrt() {
    return BfgsUpdate::SkippedNonPositiveCurvature;
}
let n = s.len();
let hy = h.dot(y);
let yhy = y.dot(&hy);
let rho = 1.0 / ys;
let term1 = (ys + yhy) * rho * rho;
for i in 0..n {
    for j in 0..n {
        h[(i, j)] += term1 * s[i] * s[j] - rho * (hy[i] * s[j] + s[i] * hy[j]);
    }
}
```

| Code | Equation |
|---|---|
| `ys` | $y^{\mathsf T}s = 1/\rho$ |
| `BfgsUpdate::Degenerate` | the caller resets $H$ to the seed |
| `CURVATURE_EPS` | $\varepsilon = 10^{-8}$ |
| `term1` | $(y^{\mathsf T}s + y^{\mathsf T}Hy)\rho^2$, the weight of $ss^{\mathsf T}$ |
| `hy[i] * s[j] + s[i] * hy[j]` | $Hys^{\mathsf T} + sy^{\mathsf T}H$, using $H$ symmetric so $(Hy)^{\mathsf T} = y^{\mathsf T}H$ |

> **Verified equation @eq:geom-bfgs** · the BFGS inverse update
>
> The update satisfies the secant condition $H'y = s$ in symbols for a generic
> symmetric $H$, equals the inverse of the direct BFGS update of $B = H^{-1}$,
> is symmetric, and stays positive definite when $y^{\mathsf T}s > 0$. A
> positive-definite $H$ is what makes the step a descent direction. Claim B3 covers the
> curvature guard: the guarded update stays positive definite for every pair, the unguarded
> one loses it whenever $y^{\mathsf T}s < 0$, and above the threshold the two are bitwise equal.
>
> | | |
> |---|---|
> | Proof | `scripts/geometry_search_steps_sympy.py`, claims B1, B2 and B3 |
> | Negative control | the runner drops the $y^{\mathsf T}s$ term of the weight $(y^{\mathsf T}s + y^{\mathsf T}Hy)/(y^{\mathsf T}s)^2$ (claim B2 fails), and removes the curvature guard (claim B3 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/optimize.rs::optimize_coordinates` |
> | Code test | `test_optimize_h2_sto3g_step_energy_anchor` in `crates/ferric-scf/src/optimize.rs` (the Cartesian path, pinned bit for bit); `negative_curvature_step_does_not_send_the_minimizer_uphill` (a double well started in its negative-curvature region; without the guard the search climbs to the maximum and reports converged there) |
>
> <!-- verified: id=geom.bfgs; script=scripts/geometry_search_steps_sympy.py; claim=B2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/optimize.rs::optimize_coordinates; rust_test=ferric-scf/src/optimize.rs::test_optimize_h2_sto3g_step_energy_anchor; registry=geometry-search -->

### Convergence

Convergence needs at least one step taken and all three of
$\lvert\Delta E\rvert < e_{\mathrm{conv}}$, $\max\lvert g\rvert < g_{\max}$
and $\mathrm{rms}(g) < g_{\mathrm{rms}}$ together. The defaults are
$10^{-6}$ Ha, $4.5\times10^{-4}$ and $3.0\times10^{-4}$ Ha/Bohr, a trust
radius of 0.1 Bohr and 100 steps.

## Lindh model Hessian {#geometry-search-lindh-model-hessian}

BFGS learns the curvature one step at a time. Started from the identity, it
treats a stiff C–H stretch and a soft torsion alike, and the first steps
are poor. The Lindh model (R. Lindh, A. Bernhardsson, G. Karlström and
P.-Å. Malmqvist, Chem. Phys. Lett. 241, 423 (1995)) gives a better start for
the cost of a few exponentials: a force constant for every stretch, bend and
torsion, decaying with distance. It is
`ferric-core/src/model_hessian.rs::lindh_cartesian_hessian`.

### The model

For atoms $i, j$ in periods $p_i, p_j$ (H–He, Li–Ne, Na and heavier), a
decay factor measures how bonded the pair is:

$$ \rho_{ij} = \exp\!\big[\alpha_{p_ip_j}\big(r_{{\rm ref},p_ip_j}^2 - r_{ij}^2\big)\big] , $$

with $\alpha$ = (1.0000, 0.3949, 0.3949; 0.3949, 0.2800, 0.2800; 0.3949,
0.2800, 0.2800) Bohr$^{-2}$ and $r_{\rm ref}$ = (1.35, 2.10, 2.53; 2.10, 2.87,
3.40; 2.53, 3.40, 3.40) Bohr (Table 1 of the paper). The force constants are
products of the factors along each internal coordinate:

$$ k_{ij} = k_r\rho_{ij},\qquad k_{ijk} = k_\varphi\rho_{ij}\rho_{jk},\qquad k_{ijkl} = k_\tau\rho_{ij}\rho_{jk}\rho_{kl} , $$

with $k_r = 0.45$ Ha/Bohr$^2$, $k_\varphi = 0.15$ and $k_\tau = 0.005$ Ha/rad$^2$.
At the reference distance $\rho = 1$ and $k = k_r$. Differentiating the
exponent gives $\partial\ln\rho_{ij}/\partial r_{ij} = -2\alpha r_{ij}$.

The Cartesian model is a sum over every stretch, bend and torsion of the
molecule, bonded or not. Each internal coordinate $q$ contributes its force
constant times the outer product of its Wilson vector
$b_q = \partial q/\partial x$ (from `ferric-core/src/internal_coords.rs::primitive_derivatives`):

$$ H^{\rm L} = \sum_q k_q\, b_q b_q^{\mathsf T} . $$

Two properties follow term by term. Each term is a rank-one positive
semi-definite matrix, so $H^{\rm L}$ is symmetric and positive
semi-definite. Each primitive is invariant under a rigid motion $t$, so
$b_q^{\mathsf T}t = 0$ and $H^{\rm L}t = 0$: the three translations and three
rotations are null vectors. Bends within 5° of linear and torsions that
contain one are dropped, because their $b_q$ diverge, and so are terms with
a $\rho$ product below $10^{-8}$. Dropping a positive semi-definite term keeps
the sum positive semi-definite.

### The seed

The optimizer needs an inverse, and $H^{\rm L}$ is singular on the rigid
motions. Let $T$ be the orthonormal rigid-body basis and $Q$ its orthonormal
complement. Diagonalize the model on the complement,
$Q^{\mathsf T}H^{\rm L}Q = VwV^{\mathsf T}$, and invert there:

$$ H_0 = QV\,{\rm diag}\big(1/\max(w_i, 10^{-3})\big)V^{\mathsf T}Q^{\mathsf T} + TT^{\mathsf T} . $$

The floor covers directions with no model curvature, such as a linear
molecule's bends. $10^{-3}$ is below the curvature of one bonded torsion,
about $k_\tau\lVert b\rVert^2 \approx 3\times10^{-3}$. The identity block on
$T$ matches the simple start, so a gradient with a rigid-body component (an
external field) is treated as before. In redundant internals the seed is the
diagonal $1/\max(k_q, 10^{-3})$ over the bonded primitives. The
coordinate-only search `optimize_coordinates` (periodic cells) has no
molecule to build the model from. It starts from the identity, refuses an
explicit `lindh`, and uses the trust rule it is given.

**QM/MM** (`optimize_qmmm`). The search moves the free atoms $f$ (the QM
real atoms, plus the MM atoms that `move_mm` frees) and holds the others
fixed, so its Hessian is the free–free block $H^{\rm L}_{ff}$ of the model
of the whole structure. That means QM real atoms and MM atoms, but not link
atoms, whose positions follow from their frontier and host atoms. A QM-only
model would drop the boundary bonds that tie a frontier atom to its fixed
host and, under `move_mm`, every MM curvature.

$H^{\rm L}_{ff}$ only needs the terms that contain a free atom. Each factor
of a term is at most $\rho_{\max} = \max(1, \max_{i<j}\rho_{ij})$, so every
link of a term above the cutoff has $\rho \ge 10^{-8}/\rho_{\max}^2$, and
every atom of such a term is within three such links of a free atom. The
block is taken from the model of that environment
(`ferric-scf/src/optimize.rs::lindh_environment`), which gives the same block
to rounding without the $(3N)^2$ matrix of a large MM region. Fixed atoms
make the rigid motions of the free atoms physical, but the model gives them
curvature only through bonds to fixed atoms, so the seed keeps the identity
on them:

$$ H_0 = Q_fV\,{\rm diag}\big(1/\max(w_i, 10^{-3})\big)V^{\mathsf T}Q_f^{\mathsf T} + T_fT_f^{\mathsf T} , $$

with $T_f$ the rigid-body basis of the free atoms and
$Q_f^{\mathsf T}H^{\rm L}_{ff}Q_f = VwV^{\mathsf T}$. With every atom free
it reduces to the molecular seed, bit for bit. The search is Cartesian;
`initial_hessian = "simple"` restores the identity start.

`scripts/lindh_hessian_sympy.py` proves the model: L1 the force constants
and their distance derivatives in symbols, and the constants and tables read
from the Rust against Table 1; L2 the Wilson vectors against sympy gradients
of the primitive values; L3 symmetry, positive semi-definiteness and the six
null vectors; L4 a brute-force numpy sum over every term, which the Rust
test `lindh_hessian_matches_the_numpy_reference` matches through
`testdata/reference/lindh_model_hessian.json`.

> **Verified equation** · the Lindh model Hessian
>
> The force constants and their distance derivatives hold in symbols, the
> constants in the Rust equal Table 1 of the paper, the closed-form Wilson
> vectors equal sympy gradients of the primitive values, and the Cartesian sum
> is symmetric and positive semi-definite with exactly six rigid-body null
> vectors. An independent brute-force numpy sum over every stretch, bend and
> torsion gives the stored reference that the Rust matches to 1e-12.
>
> | | |
> |---|---|
> | Proof | `scripts/lindh_hessian_sympy.py`, claims L1 to L4 |
> | Negative control | the runner changes one α entry, the bend constant, a Wilson torsion term and the sign of the ρ exponent (claims L4, L4, L2 and L1 fail) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-core/src/model_hessian.rs::lindh_cartesian_hessian` |
> | Code test | `lindh_hessian_matches_the_numpy_reference` in `crates/ferric-scf/tests/optimizer_model_hessian.rs` (fails when an α entry changes from 0.3949 to 0.28) |
>
> <!-- verified: id=geom.lindh; script=scripts/lindh_hessian_sympy.py; claim=L4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-core/src/model_hessian.rs::lindh_cartesian_hessian; rust_test=ferric-scf/tests/optimizer_model_hessian.rs::lindh_hessian_matches_the_numpy_reference; registry=geometry-search -->

### Measured step counts

Steps to convergence at HF/STO-3G from RDKit/MMFF starting geometries, with
the default thresholds (`g_max` 4.5e-4, `g_rms` 3e-4, `e_conv` 1e-6). "Old"
is `simple` + `fixed`, "new" is `lindh` + `ratio`. geomeTRIC 1.1 in TRIC
coordinates, run on ferric gradients through `ferric.ase`, is an outside
reference:

| molecule (atoms) | Cartesian old | Cartesian new | internal old | internal new | geomeTRIC |
|---|---|---|---|---|---|
| water (3) | 5 | 3 | 5 | 3 | 4 |
| ethanol (9) | 28 | 4 | 16 | 12 | 5 |
| butane (14) | 19 | 5 | 16 | 12 | 8 |
| paracetamol (20) | 60 | 33 | 30 | 23 | 29 |
| alanine dipeptide (22) | 118 | 39 | 40 | 27 | 22 |
| aspirin (21) | 58 | 16 | 32 | 17 | 21 |
| ibuprofen (33) | 80 | 19 | 67 | 21 | 16 |
| sulfamethoxazole (28) | 170 | 92 | 46 | 24 | 25 |
| total | 538 | 211 | 252 | 139 | 130 |
| pregabalin (28) | | 21 | | 30 | 17 |
| procaine (37) | | 13 | | 17 | 18 |
| lidocaine (39) | | 78 | | 34 | 42 |
| metoprolol (44) | | 43 | | 31 | 30 |
| total, 12 molecules | | 366 | | 251 | 237 |
| rejected steps, 12 molecules | | 18 | | 24 | |

Which change does the work? `lindh` alone (Cartesian, `fixed`) gives 4, 5,
49, 30 and 26 steps on ethanol, butane, alanine dipeptide, aspirin and
ibuprofen. `ratio` alone reproduces the old trajectories exactly, since with
the identity seed no step reaches the trust boundary. So the seed gives most
of the gain on these molecules.

The searches also differ in where they stop. The old Cartesian search stops
0.01 to 0.10 mHa above the minimum the others reach (0.03 to 0.21 Å RMSD):
with the identity seed a soft mode meets the gradient thresholds before it
is relaxed. The new searches end within 4e-3 mHa of each other, except
Cartesian new on sulfamethoxazole. Its long early steps turn the sulfonamide
C–C–N–S torsion from 13° to −64° (internals and geomeTRIC stop at +59°), and
it ends in a rotamer 3.3 mHa lower. Both end points are minima: re-optimizing
either one with the other search moves the energy by less than 0.02 mHa.

### Redundant internal coordinates

`coordinates = "internal"` (opt-in, `CoordSystem::Internal`) runs the same
update on bonds, angles and proper dihedrals:

| Step | Quantity | Function |
|---|---|---|
| 1 | start from the empirical diagonal Hessian (Schlegel), so $H_{ii} = 1/k_i$ | `ferric-core/src/internal_coords.rs::InternalCoords::initial_hessian_diagonal` |
| 2 | $g_q = G^{-}B\,g_x$ with the generalized inverse of $G = BB^{\mathsf T}$ | `ferric_core::internal_coords::gradient_to_internal` |
| 3 | $p = -Hg_q$, clipped per coordinate type: bonds against the Cartesian trust radius, angles and dihedrals against $0.3$ rad. A mixed norm would add Bohr² to rad² | `ferric-scf/src/optimize.rs::clip_internal_step` |
| 4 | back-transform to Cartesians iteratively, to $10^{-9}$ Bohr, in at most 25 passes. If it fails the step is halved, up to five attempts, and the best iterate is taken with a printed warning | `ferric_core::internal_coords::step_to_cartesian` |
| 5 | update $H$ with the **achieved** displacement $s$ read from the two geometries, not the requested one. The requested step can differ from the achieved one by more than its own size in a redundant set. A dihedral crossing $\pm\pi$ is wrapped to the short way round | `ferric-scf/src/optimize.rs::bfgs_secant_step` |
| 6 | on an unusable update, reset to the empirical diagonal, not the identity, which has no meaning when coordinates have different units | same loop |

The update uses the same curvature guard as in Cartesians. The convergence
test is the Cartesian one in both systems, so "converged" means the same
thing in both. The coordinates are built once from the starting geometry,
because the Hessian history is indexed against that primitive list; if they
cannot be built the run falls back to Cartesians with a printed warning.

### The internal step under the ratio rule

Under `"ratio"` the internal search scales both radii by one factor
$f \in [0.01, 4]$. Write $\rho_d = f r_{\rm trust}$ for bonds and
$\rho_a = 0.3 f$ for angles, and split $p = -Hg_q$ into its distance and
angular blocks $p_d$, $p_a$. The step is clipped by one factor
(`ferric-scf/src/optimize.rs::clip_internal_step_uniform`):

$$ \Delta q = \alpha p,\qquad \alpha = \min\!\Big(1, \frac{\rho_d}{\lVert p_d\rVert}, \frac{\rho_a}{\lVert p_a\rVert}\Big) . $$

Each block stays inside its own radius and the units are never added. The
first-order change is $g_q^{\mathsf T}\Delta q = -\alpha\, g_q^{\mathsf T}Hg_q < 0$
for $H$ positive definite, and the model change is
$\alpha g_q^{\mathsf T}p\,(1 - \alpha/2) < 0$ for $0 < \alpha \le 1$, as in
Cartesians. If rounding leaves $g_q^{\mathsf T}p \ge 0$, the step is taken
along $-H_0g_q$ instead, with $H_0$ the positive diagonal seed
(`ferric-scf/src/optimize.rs::internal_model_direction`).

The predicted change is $g_q^{\mathsf T}s + \tfrac12 s^{\mathsf T}H^{-1}s$ on
the achieved internal step $s$. That needs a linear solve, because a
redundant step is not $\alpha p$. The step length that enters the radius
rule is $\max(\lVert s_{\rm dist}\rVert/r_{\rm trust}, \lVert s_{\rm ang}\rVert/0.3)$,
in the units of $f$.

Why one factor? Under `"fixed"` the two blocks are clipped by their own
factors $a$ and $b$ (`clip_internal_step`, the search before the ratio rule,
kept bit for bit). Then

$$ g_q^{\mathsf T}\Delta q = a\,g_d^{\mathsf T}p_d + b\,g_a^{\mathsf T}p_a , $$

which is not sign-definite: $g_d^{\mathsf T}p_d$ can be positive while the
sum $g_q^{\mathsf T}p$ is negative. With both blocks clipped,
$a/b = (\rho_d\lVert p_a\rVert)/(\rho_a\lVert p_d\rVert)$ does not depend on
$f$, so the clipped direction is the same at every radius, and shrinking
the radius cannot restore descent. With
$\Delta E(\Delta q) \approx g^{\mathsf T}\Delta q + \tfrac12\Delta q^{\mathsf T}B\Delta q$,
a direction whose first-order descent is small against its curvature rises
until the radius is below $2\lvert g^{\mathsf T}u\rvert/(u^{\mathsf T}Bu)$
along the unit direction $u$, which can be below the floor $f = 0.01$.

*The achieved step.* The back-transformation returns $s = P\Delta q$ to first
order, with $P = GG^{-}$ the projector onto the range of $G = BB^{\mathsf T}$.
The internal gradient $g_q = G^{-}Bg_x$ lies in that range, so
$g_q^{\mathsf T}s = g_q^{\mathsf T}\Delta q$. That is also the Cartesian change
$g_x^{\mathsf T}\Delta x$ with $\Delta x = B^{\mathsf T}G^{-}\Delta q$. A descent
$\Delta q$ therefore gives a descent $s$ for a short enough step.

*The radius.* `ratio_trust_update` sets the radius to $\tfrac14\lVert s\rVert$
when the step is rejected, when $\Delta E_{\rm pred} \ge 0$, or when
$r < \tfrac14$. It doubles only for an accepted step with
$\Delta E_{\rm pred} < 0$, $r > \tfrac34$ and $\lVert s\rVert \ge 0.99\Delta$.

*The stall this removes.* Alanine dipeptide at HF/STO-3G (internal, Lindh,
`"ratio"`), with the radius rule fixed and the per-block clip kept, fails in
300 steps, 284 of them at $f = 0.01$. Each logged step there has
$a = 3.1\times10^{-2}$, $b = 2.9\times10^{-3}$ and
$g_q^{\mathsf T}p = -3.9\times10^{-3}$, but
$g_q^{\mathsf T}\Delta q = -4.7\times10^{-8}$ and
$g_q^{\mathsf T}s = -4.6\times10^{-8}$. The model predicts
$+1.300\times10^{-7}$ Ha against $+1.302\times10^{-7}$ Ha measured. So the
model is right ($r = 1.00$), the back-transformation is not the cause
($g_q^{\mathsf T}s \approx g_q^{\mathsf T}\Delta q$), and the energy rise is far
above SCF noise: the per-block direction is barely downhill and its
curvature term wins at the floor radius. One factor would have given
$g_q^{\mathsf T}\Delta q = b\,g_q^{\mathsf T}p = -1.1\times10^{-5}$.

### Which coordinates are the default

Internals are not the default, because they add ways to fail (bond
perception, back-transformation) that Cartesians do not have. The default
was chosen on three criteria, in order: the search stays in the starting
basin, it does not fail, and then the step count. The four flexible
molecules in the lower rows of the step-count table (RDKit ETKDG seed 7 +
MMFF starts) were added for this choice. The basin test is the largest
change of any heavy-atom dihedral between start and end point.

Before the runs the artifact hypothesis was written down. A broken internal
back-transformation would show as many rejected steps and
back-transformation warnings, as dihedrals that stay near the start only
because steps are suppressed, and as an end point that differs from
geomeTRIC's and moves when a Cartesian search restarts from it. A real
coordinate effect gives the same end point as geomeTRIC. The end-point
comparison tells the two apart; the step count does not.

| molecule | largest dihedral change: Cartesian / internal / geomeTRIC | rejected steps: Cartesian / internal |
|---|---|---|
| paracetamol | 22° / 22° (opposite signs, same energy) / – | 2 / 3 |
| sulfamethoxazole | 77° (C–C–N–S, +13° → −64°) / 46° (→ +59°) / – | 7 / 0 |
| pregabalin | 15.6° / 16.5° / 16.1° | 1 / 3 |
| procaine | 4.7° / 4.7° / 4.7° | 0 / 1 |
| lidocaine | 80° / 80° / 80° (N–C–C–N, −107° → +172°) | 5 / 2 |
| metoprolol | 22.7° / 24.3° / 24.5° | 2 / 4 |

The other molecules change no dihedral by more than 17° in either search.
On the four new molecules the internal search ends within 0.4° and 0.003 Å
RMSD of geomeTRIC's end point and within 1.7° and 0.024 Å of the Cartesian
one, with no back-transformation warnings and no fallback steps. Lidocaine
turns by 80° in all three optimizers: that is a downhill relaxation from the
MMFF start, not a barrier crossing. Only Cartesian sulfamethoxazole leaves
the starting basin. On this set internals lead on the first criterion (the
one sulfamethoxazole case), tie on the second (no failures; 24 rejected
steps against 18) and lead on steps in total (251 against 366), while
taking more steps than Cartesians on six of the twelve molecules (ethanol 12
against 4, butane 12 against 5, pregabalin 30 against 21).

The default also governs searches outside this set, and two of them go
against internals.

- Pregabalin at PBE/def2-SVP (RI-J/RI-K, f64), same start: Cartesian 34
  steps (2 rejected), internal 21 (1 rejected). The internal search stops
  0.44 mHa above the Cartesian end point, with the C5–C8–C9–O10 torsion at
  +109° where the Cartesian search reaches +129°. Its last steps still lower
  the energy by about 1e-6 Ha each, and $\Delta E$ falls below `e_conv`
  while the gradient is under its thresholds. A Cartesian search restarted
  from the internal end point stops after one step, so the default
  thresholds do not tell the two end points apart.
- Water at HF/STO-3G in a uniform field of 0.02 a.u. along $x$
  (`external_field`): Cartesian converges in 22 steps; internal does not
  converge in 100, 14 mHa higher, with the largest gradient component fixed
  at 1.1e-2. A field exerts a torque. The internal step cannot rotate the
  molecule, and the convergence test is on the Cartesian gradient, which
  contains the torque. Fixed point or smeared charges (QM/MM MM atoms) and
  polarizable sites break the invariance the same way (a net force and
  torque). PCM and COSMO cavities are built from the atoms and move with
  them, so they keep it.

An explicit `coordinates = "internal"` with any of those terms is refused
before the first SCF, with an error that names the cause and suggests
`"cartesian"`. The check is
`ferric-scf/src/optimize.rs::refuse_internal_coordinates_in_fixed_frame`,
called by every SCF and OO-RI-MP2 optimizer, the CLI and `run_optimize`.
Adding the six rigid-body coordinates to the internal set was not chosen: it
changes the $B$ matrix, the model Hessian and every recorded internal path,
and Cartesian already handles these systems. A `fock_hook` is opaque and is
not inspected.

The default therefore stays Cartesian. `coordinates = "internal"` is the
choice for an isolated molecule where a soft torsion near a barrier matters.

## Saddle points: P-RFO {#geometry-search-saddle-points-p-rfo}

### Why the minimizer cannot find a saddle

At a transition state the Hessian has exactly one negative eigenvalue. The
search must go uphill along that mode and downhill along every other. BFGS
keeps its Hessian positive definite by construction, so it would remove the
one negative eigenvalue the search depends on. The saddle search of
`ferric-scf/src/saddle.rs::find_saddle` is therefore a separate code. It
works in mass-weighted Cartesians, in the eigenbasis of the Hessian, which
is why it is called eigenvector following.

1. Mass weight, $\tilde B_{ij} = B_{ij}/\sqrt{m_im_j}$, symmetrize, and project
   out translations and rotations with the projector of section @sec:frequencies. The six
   rigid modes are then shifted to $+10^{3}$, not removed, so indices stay
   aligned and a rigid mode that comes out slightly negative can never be
   followed.
2. Diagonalize. The gradient components in the eigenbasis are
   $g_i = (E^{\mathsf T}M^{-1/2}g)_i$, with $E$ the eigenvectors.
3. Take the P-RFO step below, back to Cartesians, scale it uniformly to the
   trust radius (0.05 Bohr by default), and evaluate energy and gradient.
4. Carry $B$ forward with a Bofill update, or recompute it every
   `hessian_recalc_every` steps (default 0, never).
5. At convergence build a final Hessian and count negative projected
   eigenvalues.

### The P-RFO step

In the eigenbasis the Newton step is $\Delta q_i = -g_i/b_i$. Along a mode
with $b_i > 0$ it goes downhill; along a mode with $b_i < 0$ it goes uphill.
So the Newton step goes to whatever stationary point the local quadratic
has, whatever its character. Rational-function optimization (RFO) gives
control over the direction. It solves the augmented-Hessian eigenproblem

$$ \begin{pmatrix} \mathrm{diag}(b) & g\\ g^{\mathsf T} & 0\end{pmatrix}\begin{pmatrix}\Delta q\\ 1\end{pmatrix} = \lambda\begin{pmatrix}\Delta q\\ 1\end{pmatrix} . $$

Read off the rows. The top rows say $(b_i - \lambda)\Delta q_i + g_i = 0$, so
$\Delta q_i = -g_i/(b_i - \lambda)$: a Newton step with every curvature
shifted by $\lambda$. The last row says $\lambda = g^{\mathsf T}\Delta q$.
Substituting the first into the second gives the secular equation
$\lambda = \sum_i g_i^2/(\lambda - b_i)$.

Which root to take decides the direction. Along mode $i$ the step goes
downhill when $g_i\Delta q_i < 0$, that is when $b_i - \lambda > 0$. A shift
below every $b_i$ makes every step downhill.

Partitioned RFO (P-RFO) solves a separate problem in each partition. For the
followed mode $k$, with curvature $b_k$ and gradient $g_k$, the problem is
$2\times2$: $\left[\begin{smallmatrix} b_k & g_k\\ g_k & 0\end{smallmatrix}\right]$,
whose characteristic polynomial is $\lambda^2 - b_k\lambda - g_k^2 = 0$. Take
the larger root:

$$ \lambda_p = \tfrac12\big(b_k + \sqrt{b_k^2 + 4g_k^2}\big) > \max(b_k, 0) . $$

The bound holds because the square root exceeds $\lvert b_k\rvert$ when
$g_k \ne 0$. So $b_k - \lambda_p < 0$, and $\Delta q_k = -g_k/(b_k - \lambda_p)$
has the sign of $g_k$: the step climbs whatever the sign of $b_k$. For the
other modes one shift $\lambda_n$ is the most negative root of their
secular equation, the lowest eigenvalue of their augmented Hessian, below
$\min_{i\ne k}b_i$, and every one of those steps is downhill:

$$ \lambda_p = \tfrac12\big(b_k + \sqrt{b_k^2 + 4g_k^2}\big), \quad \Delta q_k = -\frac{g_k}{b_k - \lambda_p}; \qquad
\lambda_n = \sum_{i\ne k}\frac{g_i^2}{\lambda_n - b_i}\ \ (\lambda_n < \min_{i\neq k}b_i), \quad \Delta q_i = -\frac{g_i}{b_i - \lambda_n} . \label{eq:geom-prfo} $$

Check the trivial limit. At a stationary point every $g_i = 0$, so
$\Delta q = 0$ in both partitions and the search takes no step
(`prfo_at_a_stationary_point_takes_no_step`).

The code, `ferric-scf/src/saddle.rs::prfo_step`, solves the second equation
by Newton iteration on $h(\lambda) = \lambda - \sum g_i^2/(\lambda - b_i)$
from $\lambda_n = \min(b_{\min}-1, 0)$, at most 100 iterations, and falls
back to $b_{\min}-10^{-3}$ if the iterate leaves the bracket. The followed
mode is four lines:

<!-- excerpt: ferric-scf/src/saddle.rs::prfo_step -->
```rust
let b_p = evals[follow];
let g_p = g_q[follow];
let lambda_p = 0.5 * (b_p + (b_p * b_p + 4.0 * g_p * g_p).sqrt());
let den_p = b_p - lambda_p;
```

`den_p` is $b_k - \lambda_p \le 0$. It is zero only when $g_k = 0$ and
$b_k \ge 0$, and the code then sets the step on that mode to zero instead of
dividing.

> **Verified equation @eq:geom-prfo** · the P-RFO step
>
> The followed-mode shift is the largest root of
> $\lambda^2 - b\lambda - g^2 = 0$, which is the characteristic equation of the
> $2\times2$ augmented Hessian, and $\Delta q = -g/(b-\lambda)$ solves
> $(b-\lambda)\Delta q + g = 0$ with $\lambda = g\,\Delta q$ (symbols). The
> code's step for both partitions equals the eigenvector of the augmented
> Hessian, largest eigenpair for the followed mode and smallest for the rest,
> computed by `numpy.linalg.eigh` on random spectra with one negative
> eigenvalue (numerics, to rounding).
>
> | | |
> |---|---|
> | Proof | `scripts/geometry_search_steps_sympy.py`, claims P1 and P2 |
> | Negative control | the runner takes the smaller root for the followed mode (claim P2 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/saddle.rs::prfo_step` |
> | Code test | `prfo_climbs_even_when_the_followed_mode_has_positive_curvature` in `crates/ferric-scf/src/saddle.rs` |
>
> <!-- verified: id=geom.prfo; script=scripts/geometry_search_steps_sympy.py; claim=P2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/saddle.rs::prfo_step; rust_test=ferric-scf/src/saddle.rs::prfo_climbs_even_when_the_followed_mode_has_positive_curvature; registry=geometry-search -->

The Rust test is the discriminating one. On a mode that is already
negative, the Newton step $-g/b$ and the RFO step point the same way, so a
test built there cannot tell P-RFO from a minimizer. With positive curvature
$b>0$ the shift $\lambda_p > b$ flips the denominator and the step keeps
climbing, while Newton steps downhill. That is why the test uses
all-positive eigenvalues.

### The Hessian update

Between steps the saddle search updates $B$ from the step $\Delta x$ and the
gradient change $\Delta g$, as BFGS does, but with an update that does not
force $B$ positive definite. Bofill's update mixes the Murtagh–Sargent and
Powell symmetric-Broyden corrections. With $e = \Delta g - B\Delta x$, the
part of the gradient change the current $B$ did not predict, and
$\varphi = (e\!\cdot\!\Delta x)^2/(e\!\cdot\! e\;\Delta x\!\cdot\!\Delta x)$:

$$ B' = B + \varphi\,\frac{e\,e^{\mathsf T}}{e\cdot\Delta x} + (1-\varphi)\Big[\frac{e\,\Delta x^{\mathsf T} + \Delta x\,e^{\mathsf T}}{\Delta x\cdot\Delta x} - \frac{(e\cdot\Delta x)\,\Delta x\,\Delta x^{\mathsf T}}{(\Delta x\cdot\Delta x)^2}\Big] . \label{eq:geom-bofill} $$

Both corrections satisfy $\Delta B\,\Delta x = e$. For Murtagh–Sargent,
$ee^{\mathsf T}\Delta x/(e\cdot\Delta x) = e$. For Powell, multiply out:
$e + \Delta x\,(e\cdot\Delta x)/(\Delta x\cdot\Delta x) - \Delta x\,(e\cdot\Delta x)/(\Delta x\cdot\Delta x) = e$.
So every mixture satisfies $B'\Delta x = B\Delta x + e = \Delta g$, the
secant condition. The weight $\varphi$ is 1 when $e$ is parallel to
$\Delta x$, where Murtagh–Sargent is exact, and 0 when they are orthogonal,
where Murtagh–Sargent divides by zero and Powell does not
(`ferric-scf/src/saddle.rs::bofill_update`).

> **Verified equation @eq:geom-bofill** · the Bofill update
>
> Each of the two corrections satisfies the secant condition
> $\Delta B\,\Delta x = e$ in symbols, so every mixture $\varphi$ does. The code's
> update satisfies $B'\Delta x = \Delta g$, is symmetric, and keeps a negative
> eigenvalue of an indefinite $B$ that the BFGS update of 14.20 would drive out.
>
> | | |
> |---|---|
> | Proof | `scripts/geometry_search_steps_sympy.py`, claims F1 and F2 |
> | Negative control | the runner drops the projector term of the Powell correction (claim F2 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/saddle.rs::bofill_update` |
> | Code test | `bofill_reproduces_the_secant_condition` in `crates/ferric-scf/src/saddle.rs` |
>
> <!-- verified: id=geom.bofill; script=scripts/geometry_search_steps_sympy.py; claim=F2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/saddle.rs::bofill_update; rust_test=ferric-scf/src/saddle.rs::bofill_reproduces_the_secant_condition; registry=geometry-search -->

### What the search refuses

- It will not start from a geometry whose projected Hessian has no negative
  eigenvalue (below $-10^{-6}$): there is no uphill direction, and a result
  from there would be a minimum labelled as a transition state. The error
  names the lowest eigenvalue.
- It does not call a stationary point converged on the gradient alone.
  `SaddleResult::is_transition_state` needs `converged` and exactly one
  imaginary frequency in a final Hessian. Zero means a minimum and two or
  more a higher-order saddle.
- It does not check that the imaginary mode is the reaction you meant. The
  mode is returned in `SaddleResult::imaginary_mode` for the caller, which
  is also the input of the reaction-path walk below.

Masses in this module are in u, so Hessian eigenvalues are in Ha/Bohr²/u and
the $10^{3}$ shift and $10^{-6}$ tolerance are in those units, unlike the
electron-mass units of section @sec:frequencies. The variable is named `mass_au` in the
code.

**Cost.** A finite-difference Hessian costs $6N+1$ gradients
(`a_finite_difference_hessian_costs_6n_plus_one_gradients` in
`crates/ferric-scf/tests/saddle_cost.rs`), and the default builds two
Hessians: one at the start and one for the final character check. The
Python default is `hessian = "fd"`; `"analytic"` and `"auto"` use section
@sec:hf-hessian, with zero displaced gradients, for closed-shell references that the
analytic Hessian accepts. Recomputing every step is available and, on the
system tested, did not reduce the step count.

## Reaction paths: mass-weighted descent {#geometry-search-reaction-paths-mass-weighted}

`ferric-scf/src/irc.rs::follow_irc` displaces the saddle along the imaginary
mode in each direction and follows the mass-weighted gradient downhill. The
path is the classical trajectory of infinitesimal kinetic energy, which is
the definition of the intrinsic reaction coordinate. In mass-weighted
coordinates the kinetic energy is $\tfrac12\lVert\dot q\rVert^2$, with the same
weight on every coordinate, and that path is steepest descent in $q$. The
weighting is therefore part of the definition: unweighted steepest descent
gives a different path.

With $q_i = \sqrt{m_i}\,x_i$ the chain rule gives
$\partial E/\partial q_i = g_i/\sqrt{m_i}$. Each step moves a length $\ell$
down that gradient, and $\Delta x_i = \Delta q_i/\sqrt{m_i}$ converts it back:

$$ \Delta q = -\ell\,\frac{g_q}{\lVert g_q\rVert}, \qquad \ell = \min(\ell_0,\,2\lVert g_q\rVert), \qquad
\Delta x_i = -\,\ell\,\frac{g_i}{m_i\,\lVert g_q\rVert}, \label{eq:geom-irc-step} $$

with $\ell_0 = 0.1$ $\sqrt{\mathrm u}$ Bohr. The walk starts from
$q_{\mathrm{saddle}} \pm 0.2\,\hat q$, where $\hat q$ is the mass-weighted,
normalized mode, and stops when $\max\lvert g\rvert < 10^{-4}$ Ha/Bohr or
after 200 steps per branch. The Cartesian step divides by $m_i$, not
$\sqrt{m_i}$: a hydrogen moves about twelve times farther than a carbon
under the same force. With all masses equal the factor is common to every coordinate
and the direction is that of Cartesian steepest descent.

The mode is normalized after mass weighting, not in Cartesians. Otherwise
the initial displacement would depend on which atoms the mode moves: a mode
on hydrogens would step much farther in mass-weighted space than one on
carbons for the same nominal length. The step itself is two lines of
`ferric-scf/src/irc.rs::walk`:

<!-- excerpt: ferric-scf/src/irc.rs::walk -->
```rust
let len = config.step.min(DAMPING * gn);
for i in 0..n3 {
    q[i] -= len * gq[i] / gn;
}
```

Here `gq` is $g_q$ (the Cartesian gradient divided by `sm`, which holds
$\sqrt{m_i}$), `gn` is $\lVert g_q\rVert$, `DAMPING` is 2 and `config.step`
is $\ell_0$. The geometry is set from $q$ at the top of the next iteration
by dividing by $\sqrt{m_i}$, which completes the two-stage form of (@eq:geom-irc-step).

> **Verified equation @eq:geom-irc-step** · the mass-weighted step
>
> $\partial E/\partial q_i = (\partial E/\partial x_i)/\sqrt{m_i}$ by the chain rule
> (symbols), the Cartesian step of (@eq:geom-irc-step) equals the code's two-stage form
> through $q$, and it differs from unweighted steepest descent unless all masses
> are equal.
>
> | | |
> |---|---|
> | Proof | `scripts/geometry_search_steps_sympy.py`, claim I1 |
> | Negative control | the runner divides by $\sqrt m$ instead of $m$ in the closed form (claim I1 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/irc.rs::walk` |
> | Code test | `the_two_branches_land_on_opposite_sides_of_the_plane` in `crates/ferric-scf/tests/irc_connects_minima.rs` |
>
> <!-- verified: id=geom.irc-step; script=scripts/geometry_search_steps_sympy.py; claim=I1; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/irc.rs::walk; rust_test=ferric-scf/tests/irc_connects_minima.rs::the_two_branches_land_on_opposite_sides_of_the_plane; registry=geometry-search -->

**The damping.** A fixed-length step cannot settle into a basin: near a
minimum it overshoots and orbits. The source records NH3 inversion at the
default threshold: steps of 0.15 and 0.05 both ran to the step budget (400
and 800 steps) while 0.02 converged, in 63. Capping the step at
$2\lVert g_q\rVert$ shrinks it with the gradient. Far from the minimum
$\ell_0$ governs and the path is unchanged. The cap only binds at the end,
where the path shape no longer matters because the endpoint is what is
being identified.

**What it does not do.** There is no higher-order integrator
(Gonzalez–Schlegel, local quadratic approximation), so the path length and
the energy profile are not quantitative. There is no automatic check that
an endpoint is a minimum. Each `IrcBranch::converged` says only whether the
gradient threshold was met, and a `false` branch identifies no basin.
`IrcResult::both_converged` is the condition for reading the endpoints as a
reaction. The step, initial displacement and threshold must be finite and
positive. Infinities would fail without an error (an infinite threshold
reports the first point as converged), so they are refused at the boundary.

## Reproducibility {#geometry-search-reproducibility}

- The Cartesian minimizer path is pinned bit for bit by
  `cartesian_path_is_reproducible_and_matches_the_recorded_path`, and
  selecting internals does not perturb it
  (`selecting_internals_does_not_perturb_the_cartesian_path`), both in
  `crates/ferric-scf/tests/internal_coord_optimize.rs`.
- Each geometry's SCF starts from the previous geometry's converged orbitals,
  re-orthonormalized (`scf_warm_start`, default true), which agrees with a cold
  start to the SCF's own convergence, not bitwise.
- An unconverged restricted SCF is an error at every step
  (`ferric-scf/src/optimize.rs::require_converged`): a search must not walk on
  energies and gradients from a density that never settled.
- An energy correction without a gradient is a hard error, so a search never
  walks one surface while reporting another.

## Options {#geometry-search-options}

| Option | Effect |
|---|---|
| `coordinates` | `cartesian` (default) or `internal`; strict parse, no silent default |
| `initial_hessian` | `lindh` (default) or `simple`; strict parse. The coordinate-only search (periodic cells, QM/MM) has no model, starts from the identity and refuses `lindh` |
| `trust_update` | `ratio` (default) or `fixed`; strict parse. `simple` with `fixed` reproduces the earlier search bit for bit (`simple_hessian_and_fixed_trust_reproduce_the_recorded_trajectories`) |
| energy correction (dispersion) | added to energy and gradient at every step by a caller closure. With the default no-correction closure the path is byte-identical (`a_zero_correction_is_byte_identical_to_no_correction`) |
| `follow_mode` | which projected mode to climb, by ascending eigenvalue; 0 follows the lowest |
| `hessian_recalc_every` | rebuild the exact Hessian every $N$ steps instead of carrying the Bofill update |
| open shell | `run_optimize` covers UHF and ROHF; `run_saddle` and `run_irc` are closed shell only |
| QM/MM | `run_optimize_qmmm` has its own boundary handling; see chapter @sec:embedding |

## Numerical anchors {#geometry-search-numerical-anchors}

| Check | Reference | Agreement | Test |
|---|---|---|---|
| Geometry optimization, RHF and RKS: H2O, NH3, CH2O from distorted starts / RHF and B3LYP 6-31G, PBE cc-pVDZ | PySCF analytic gradient driven to $10^{-6}$ Ha/Bohr by scipy BFGS | distances at most 2.6e-6 Bohr, angles at most 1.1e-4°, energies at most 9.3e-13 Ha; bars 2e-5 Bohr, 1e-3°, 1e-10 Ha | `crates/ferric-scf/tests/validation_geometry_optimization.rs` |
| Geometry optimization, UHF, ROHF and UKS: HO2, CH3, NH2 / UHF, ROHF, UKS-PBE 6-31G | the same | the same distance, angle and energy bounds | same |
| NH3 inversion saddle, RHF/STO-3G: FD and analytic Hessians | each other | both searches end on a transition state with one imaginary frequency; geometries within 5e-3 Bohr and energies within 1e-6 Ha (asserted). The FD route builds two Hessians of 50 gradients each, the analytic route none | `crates/ferric-scf/tests/saddle_analytic_hessian.rs` |
| IRC of NH3 inversion | the geometry itself | the two branches land on opposite sides of the plane; both endpoints are downhill from the saddle | `crates/ferric-scf/tests/irc_connects_minima.rs` |
| P-RFO against a surface with a known saddle | closed form | the discriminating positive-curvature case and the zero-gradient trivial limit | `crates/ferric-scf/src/saddle.rs` unit tests |

<!-- figures: site/src/reference/validation.md "Geometry optimization, RHF and RKS"; site/src/reference/validation.md "Geometry optimization, UHF, ROHF and UKS" -->

The first two rows compare end points with an independent optimizer on an
independent gradient, so they test the search and the gradient of section
@sec:hf-gradient together. The NH3 row needs no outside code: the finite-difference and
analytic Hessians must lead the same search to the same saddle.

On the validation page, `run_saddle` is stated to cost $2(6N+1) + (\mathrm{steps}+1)$
gradients with a finite-difference Hessian and to raise if the start has no
negative mode.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/theory.md`, "Geometry searches" and "Lindh model Hessian" | the derivations, the measured step counts and the choice of default coordinates |
| `crates/ferric-scf/src/optimize.rs`, `saddle.rs`, `irc.rs` module documentation and loops | every formula and refusal in this section |
| `scripts/geometry_search_steps_sympy.py` | verified equations @eq:geom-bfgs to @eq:geom-irc-step |
| `scripts/lindh_hessian_sympy.py` | the Lindh model box |
| Banerjee, Adams, Simons and Shepard, J. Phys. Chem. 89, 52 (1985); Bofill, J. Comput. Chem. 15, 1 (1994) | P-RFO and the Hessian update, as cited in `saddle.rs` |
| Pulay and Fogarasi, J. Chem. Phys. 96, 2856 (1992); Schlegel, Theor. Chim. Acta 66, 333 (1984) | redundant internals and the initial Hessian, as cited in `ferric-core/src/internal_coords.rs` |
| `site/src/reference/validation.md` | the optimization rows and the saddle and IRC capability rows |
