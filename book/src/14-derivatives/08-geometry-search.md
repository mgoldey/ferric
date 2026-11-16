# 14.8 Geometry optimization, saddle points and reaction paths

<!-- section: 14.8; status: edit; registry: geometry-search;
     theory: crates/ferric-scf/theory.md#"Harmonic frequencies"; crates/ferric-scf/theory.md#"Geometry searches" -->

Three searches use the gradient of section 14.1 and, for saddles, the
Hessian of section 14.4. A minimizer follows the gradient downhill with a
quasi-Newton update that stays positive definite. A saddle search must break
exactly that property: it climbs along one Hessian mode and descends along
all others. A reaction-path walk answers a different question: which two
minima does a saddle connect?

The derivation is in `crates/ferric-scf/theory.md`, "Geometry searches", and
the step formulas are machine-checked by
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
| CLI | `task = "optimize"`, section `[optimize]` (`max_steps`, `g_max_thresh`, `g_rms_thresh`, `e_conv`, `trust_radius`, `coordinates`, `scf_warm_start`) |
| Python | `run_optimize`, `run_saddle`, `run_irc` |

<!-- include: crates/ferric-scf/theory.md#"Geometry searches" -->

## 14.8.1 Minimization

The default is Cartesian BFGS on the inverse Hessian, started from the
identity, with a trust-radius clip, no line search and no step rejection. With $s = x_{k+1}-x_k$
and $y = g_{k+1}-g_k$:

$$ p = -H g, \qquad s = p\,\min\!\Big(1, \frac{r_{\mathrm{trust}}}{\lVert p\rVert}\Big), \qquad
H' = H + \frac{y^{\mathsf T}s + y^{\mathsf T}Hy}{(y^{\mathsf T}s)^2}\,s\,s^{\mathsf T} - \frac{H y\,s^{\mathsf T} + s\,y^{\mathsf T}H}{y^{\mathsf T}s} . \tag{14.20} $$

If $\lvert y^{\mathsf T}s\rvert \le 10^{-12}$ the update is skipped and $H$ is reset to the
identity. If $y^{\mathsf T}s \le 10^{-8}\lVert y\rVert\lVert s\rVert$, which means negative
curvature along the step, the update is skipped and $H$ is kept, with a printed note. Any
update that satisfies $H'y = s$ has $y^{\mathsf T}H'y = y^{\mathsf T}s$, so applying it there
would make $H$ indefinite and could send the next step uphill. With the guard, $H$ stays
positive definite and every step is a descent direction. Convergence needs at least one step taken and all three of
$\lvert\Delta E\rvert < e_{\mathrm{conv}}$, $\max\lvert g\rvert < g_{\max}$,
$\mathrm{rms}(g) < g_{\mathrm{rms}}$; the defaults are $10^{-6}$ Ha,
$4.5\times10^{-4}$ and $3.0\times10^{-4}$ Ha/Bohr, a trust radius of 0.1 Bohr and 100
steps.

> **Verified equation 14.20** · the BFGS inverse update
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

**Redundant internal coordinates** (`coordinates = "internal"`, opt-in) run the
same update on bonds, angles and proper dihedrals:

| Step | Quantity | Function |
|---|---|---|
| 1 | start from the empirical diagonal Hessian (Schlegel), so $H_{ii} = 1/k_i$ | `ferric-core/src/internal_coords.rs::InternalCoords::initial_hessian_diagonal` |
| 2 | $g_q = G^{-}B\,g_x$ with the generalized inverse of $G = BB^{\mathsf T}$ | `ferric_core::internal_coords::gradient_to_internal` |
| 3 | $p = -Hg_q$, clipped per coordinate type: bonds against the Cartesian trust radius, angles and dihedrals against $0.3$ rad. A mixed norm would add Bohr² to rad² | `ferric-scf/src/optimize.rs::clip_internal_step` |
| 4 | back-transform to Cartesians iteratively, to $10^{-9}$ Bohr, in at most 25 passes. If it fails the step is halved, up to five attempts, and the best iterate is taken with a printed warning | `ferric_core::internal_coords::step_to_cartesian` |
| 5 | update $H$ with the **achieved** displacement $s$ read from the two geometries, not the requested one. The requested step can differ from the achieved one by more than its own size in a redundant set. A dihedral crossing $\pm\pi$ is wrapped to the short way round | `ferric-scf/src/optimize.rs::bfgs_secant_step` |
| 6 | on an unusable update, reset to the empirical diagonal, not the identity, which has no meaning when coordinates have different units | same loop |

The convergence test is the Cartesian one in both systems, so "converged"
means the same thing in both. The coordinates are built once from the starting
geometry, because the Hessian history is indexed against that primitive list;
if they cannot be built the run falls back to Cartesians with a printed
warning. Internals are not the default because they add ways to fail
(bond perception, back-transformation) that Cartesians do not have.

## 14.8.2 Saddle points: P-RFO

The saddle search of `ferric-scf/src/saddle.rs::find_saddle` works in mass-weighted
Cartesians. It does not reuse the minimizer, because a positive-definite
update would remove the one negative eigenvalue the search depends on.

1. Mass weight, $\tilde B_{ij} = B_{ij}/\sqrt{m_im_j}$, symmetrize, and project out
   translations and rotations with the projector of section 14.6. The six rigid modes are
   then shifted to $+10^{3}$ rather than removed, so indices stay aligned and a
   rigid mode that comes out slightly negative can never be followed.
2. Diagonalize. Gradient components in the eigenbasis are
   $g_i = (E^{\mathsf T}M^{-1/2}g)_i$.
3. Take the P-RFO step below, back to Cartesians, scale it uniformly to the trust
   radius (0.05 Bohr by default), and evaluate energy and gradient.
4. Carry $B$ forward with a Bofill update, or recompute it every `hessian_recalc_every`
   steps (default 0, never).
5. At convergence build a final Hessian and count negative projected eigenvalues.

**The step.** Partitioned rational-function optimization solves a separate
augmented-Hessian problem in each partition. For the followed mode $k$, with
curvature $b$ and gradient $g$, the maximizing step uses the larger root of the
secular equation of $\left[\begin{smallmatrix} b & g\\ g & 0\end{smallmatrix}\right]$. For all other modes one shift
$\lambda_n$ is the most negative root of the secular equation of the remaining
augmented matrix, and every step is downhill:

$$ \lambda_p = \tfrac12\big(b_k + \sqrt{b_k^2 + 4g_k^2}\big), \quad \Delta q_k = -\frac{g_k}{b_k - \lambda_p}; \qquad
\lambda_n = \sum_{i\ne k}\frac{g_i^2}{\lambda_n - b_i}\ \ (\lambda_n < \min_{i\neq k}b_i), \quad \Delta q_i = -\frac{g_i}{b_i - \lambda_n} . \tag{14.21} $$

The code solves the second equation by Newton iteration from
$\lambda_n = \min(b_{\min}-1, 0)$, at most 100 iterations, and falls back to
$b_{\min}-10^{-3}$ if the iterate leaves the bracket.

> **Verified equation 14.21** · the P-RFO step
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

The Rust test is the discriminating one. On a mode that is already negative
the Newton step $-g/b$ and the RFO step point the same way, so a test built
there cannot tell P-RFO from a minimizer. With positive curvature $b>0$ the
shift $\lambda_p > b$ flips the denominator and the step keeps climbing, while
Newton steps downhill. That is why the test uses all-positive eigenvalues.

**The Hessian update.** Bofill mixes the Murtagh–Sargent and Powell
symmetric-Broyden corrections, with $e = \Delta g - B\Delta x$ and
$\varphi = (e\!\cdot\!\Delta x)^2/(e\!\cdot\! e\;\Delta x\!\cdot\!\Delta x)$:

$$ B' = B + \varphi\,\frac{e\,e^{\mathsf T}}{e\cdot\Delta x} + (1-\varphi)\Big[\frac{e\,\Delta x^{\mathsf T} + \Delta x\,e^{\mathsf T}}{\Delta x\cdot\Delta x} - \frac{(e\cdot\Delta x)\,\Delta x\,\Delta x^{\mathsf T}}{(\Delta x\cdot\Delta x)^2}\Big] . \tag{14.22} $$

> **Verified equation 14.22** · the Bofill update
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

**What the search refuses.**

- It will not start from a geometry whose projected Hessian has no negative
  eigenvalue (below $-10^{-6}$): there is no uphill direction, and a result from
  there would be a minimum labelled as a transition state. The error names the
  lowest eigenvalue.
- It does not call a stationary point converged on the gradient alone.
  `SaddleResult::is_transition_state` needs `converged` and exactly one
  imaginary frequency in a final Hessian. Zero means a minimum and two or more
  a higher-order saddle.
- It does not check that the imaginary mode is the reaction you meant. The mode
  is returned in `SaddleResult::imaginary_mode` for the caller, which is
  also the input of the reaction-path walk below.

Masses in this module are in u, so Hessian eigenvalues are in Ha/Bohr²/u and the
$10^{3}$ shift and $10^{-6}$ tolerance are in those units, unlike the electron-mass
units of section 14.6. The variable is named `mass_au` in the code.

**Cost.** A finite-difference Hessian costs $6N+1$ gradients
(`a_finite_difference_hessian_costs_6n_plus_one_gradients` in
`crates/ferric-scf/tests/saddle_cost.rs`), and the default builds two Hessians: one at the
start and one for the final character check. The Python default is
`hessian = "fd"`; `"analytic"` and `"auto"` use section 14.4, with zero
displaced gradients, for closed-shell references that the analytic Hessian
accepts. Recomputing every step is available and, on the system tested,
did not reduce the step count.

## 14.8.3 Reaction paths: mass-weighted descent

`ferric-scf/src/irc.rs::follow_irc` displaces the saddle along the imaginary mode in
each direction and follows the mass-weighted gradient downhill. The path is the
classical trajectory of infinitesimal kinetic energy, which is the definition
of the intrinsic reaction coordinate, so the weighting is not optional:
unweighted steepest descent gives a different path.

With $q_i = \sqrt{m_i}\,x_i$ and $g_q = g/\sqrt m$, each step moves by

$$ \Delta q = -\ell\,\frac{g_q}{\lVert g_q\rVert}, \qquad \ell = \min(\ell_0,\,2\lVert g_q\rVert), \qquad
\Delta x_i = -\,\ell\,\frac{g_i}{m_i\,\lVert g_q\rVert}, \tag{14.23} $$

with $\ell_0 = 0.1$ $\sqrt{\mathrm u}$ Bohr, starting from $q_{\mathrm{saddle}} \pm 0.2\,\hat q$ where $\hat q$ is the mass-weighted,
normalized mode, and stopping when $\max\lvert g\rvert < 10^{-4}$ Ha/Bohr or after 200
steps per branch.

> **Verified equation 14.23** · the mass-weighted step
>
> $\partial E/\partial q_i = (\partial E/\partial x_i)/\sqrt{m_i}$ by the chain rule
> (symbols), the Cartesian step of (14.23) equals the code's two-stage form
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

**The damping.** A fixed-length step cannot settle into a basin: near a minimum it
overshoots and orbits. The source records NH3 inversion at the default
threshold, where steps of 0.15 and 0.05 both ran to the step budget while 0.02
converged. Capping the step at $2\lVert g_q\rVert$ shrinks it with the gradient.
Far from the minimum $\ell_0$ governs and the path is unchanged. The cap only binds
at the end, where the path shape no longer matters because the endpoint is
what is being identified.

**What it does not do.** There is no higher-order integrator (Gonzalez–Schlegel,
local quadratic approximation), so the path length and the energy profile are
not quantitative. There is no automatic check that an endpoint is a minimum; each
`IrcBranch::converged` says only whether the gradient threshold was met, and a
`false` branch identifies no basin. `IrcResult::both_converged` is the condition
for reading the endpoints as a reaction. The step, initial displacement and
threshold must be finite and positive: infinities fail silently (an infinite
threshold reports the first point as converged), so they are refused at the
boundary.

## 14.8.4 Reproducibility

- The Cartesian minimizer path is pinned bit for bit by
  `cartesian_path_is_reproducible_and_matches_the_recorded_path` and the
  selecting of internals does not perturb it
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

## 14.8.5 Options

| Option | Effect |
|---|---|
| `coordinates` | `cartesian` (default) or `internal`; strict parse, no silent default |
| energy correction (dispersion) | added to energy and gradient at every step by a caller closure. With the default no-correction closure the path is byte-identical (`a_zero_correction_is_byte_identical_to_no_correction`) |
| `follow_mode` | which projected mode to climb, by ascending eigenvalue; 0 follows the lowest |
| `hessian_recalc_every` | rebuild the exact Hessian every $N$ steps instead of carrying the Bofill update |
| open shell | `run_optimize` covers UHF and ROHF; `run_saddle` and `run_irc` are closed shell only |
| QM/MM | `run_optimize_qmmm` has its own boundary handling; see chapter 23 |

## 14.8.6 Numerical anchors

| Check | Reference | Agreement | Test |
|---|---|---|---|
| Geometry optimization, RHF and RKS: H2O, NH3, CH2O from distorted starts / RHF and B3LYP 6-31G, PBE cc-pVDZ | PySCF analytic gradient driven to $10^{-6}$ Ha/Bohr by scipy BFGS | distances at most 2.6e-6 Bohr, angles at most 1.1e-4°, energies at most 9.3e-13 Ha; bars 2e-5 Bohr, 1e-3°, 1e-10 Ha | `crates/ferric-scf/tests/validation_geometry_optimization.rs` |
| Geometry optimization, UHF, ROHF and UKS: HO2, CH3, NH2 / UHF, ROHF, UKS-PBE 6-31G | the same | the same distance, angle and energy bounds | same |
| NH3 inversion saddle, RHF/STO-3G: FD and analytic Hessians | each other | both searches end on a transition state with one imaginary frequency; geometries within 5e-3 Bohr and energies within 1e-6 Ha (asserted). The FD route builds two Hessians of 50 gradients each, the analytic route none | `crates/ferric-scf/tests/saddle_analytic_hessian.rs` |
| IRC of NH3 inversion | the geometry itself | the two branches land on opposite sides of the plane; both endpoints are downhill from the saddle | `crates/ferric-scf/tests/irc_connects_minima.rs` |
| P-RFO against a surface with a known saddle | closed form | the discriminating positive-curvature case and the zero-gradient trivial limit | `crates/ferric-scf/src/saddle.rs` unit tests |

<!-- figures: site/src/reference/validation.md "Geometry optimization, RHF and RKS"; site/src/reference/validation.md "Geometry optimization, UHF, ROHF and UKS" -->

On the validation page, `run_saddle` is stated to cost $2(6N+1) + (\mathrm{steps}+1)$
gradients with a finite-difference Hessian and to raise if the start has no
negative mode.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/src/optimize.rs`, `saddle.rs`, `irc.rs` module documentation and loops | every formula and refusal in this section |
| `scripts/geometry_search_steps_sympy.py` | verified equations 14.20 to 14.23 |
| Banerjee, Adams, Simons and Shepard, J. Phys. Chem. 89, 52 (1985); Bofill, J. Comput. Chem. 15, 1 (1994) | P-RFO and the Hessian update, as cited in `saddle.rs` |
| Pulay and Fogarasi, J. Chem. Phys. 96, 2856 (1992); Schlegel, Theor. Chim. Acta 66, 333 (1984) | redundant internals and the initial Hessian, as cited in `ferric-core/src/internal_coords.rs` |
| `site/src/reference/validation.md` | the optimization rows and the saddle and IRC capability rows |
