# ferric-mm Theory

AMBER-form molecular mechanics: bonded terms, periodic torsions, and a
pairwise Lennard-Jones plus Coulomb nonbonded sum with exclusions and 1-4
scaling. The crate assigns no parameters; every equation below is the one the
code executes. References are `path:line` relative to `crates/ferric-mm/`.

Identities marked **[sympy n]** are checked by claim *n* of
[`scripts/mm_dihedral_gradient_sympy.py`](../../scripts/mm_dihedral_gradient_sympy.py)
(60-digit arithmetic, negative control per claim, exits nonzero on mismatch).
Finite-difference tests are in `tests/terms_fd.rs`; agreement with OpenMM's
Reference platform is in `tests/vs_openmm.rs` and `tests/validation_mm.rs`.

Units are Hartree atomic units (Hartree, Bohr, radians, charge in $e$). Force
fields published in kcal/mol, Å and degrees are converted once in
`MmTopology::from_amber_units` (`src/topology.rs:271`, `src/units.rs`).
Gradients are $\partial E/\partial\mathbf R$ (not forces). $\mathbf R_a$ is the
position of atom $a$; $\mathbf r_{ab} = \mathbf R_a - \mathbf R_b$.

Total energy (`energy.rs:85`, `MmEnergy` fields):

$$E = E_\text{bond} + E_\text{angle} + E_\text{torsion} + E_\text{LJ} + E_\text{Coul}.$$

## 1. Bonds

$$E_\text{bond} = \sum_{\text{bonds}} k\,(r - r_0)^2,\qquad r = |\mathbf r_{ij}|.$$

No leading $\tfrac12$ (AMBER convention; OpenMM's `HarmonicBondForce` uses
$\tfrac12 k_\text{omm}(r-r_0)^2$, so $k_\text{omm} = 2k$; `src/topology.rs:31`).
Gradient (`energy.rs:94-112`):

$$\frac{\partial E}{\partial\mathbf R_i} = 2k(r-r_0)\,\frac{\mathbf r_{ij}}{r} = -\frac{\partial E}{\partial\mathbf R_j}.$$

## 2. Angles

$$E_\text{angle} = \sum_{\text{angles}} k_\theta\,(\theta - \theta_0)^2,\qquad
\cos\theta = \frac{\mathbf r_{ij}\cdot\mathbf r_{kj}}{|\mathbf r_{ij}|\,|\mathbf r_{kj}|},$$

with $\theta\in[0,\pi]$ the valence angle at the central atom $j$ (no leading
$\tfrac12$; `energy.rs:114-155`). With $\mathbf a=\mathbf r_{ij}$,
$\mathbf b=\mathbf r_{kj}$ and $\partial E/\partial\theta = 2k_\theta(\theta-\theta_0)$:

$$\frac{\partial\theta}{\partial\mathbf R_i} = -\frac{1}{\sin\theta}\Big(\frac{\mathbf b}{|\mathbf a||\mathbf b|} - \cos\theta\,\frac{\mathbf a}{|\mathbf a|^2}\Big),\quad
\frac{\partial\theta}{\partial\mathbf R_k} = -\frac{1}{\sin\theta}\Big(\frac{\mathbf a}{|\mathbf a||\mathbf b|} - \cos\theta\,\frac{\mathbf b}{|\mathbf b|^2}\Big),\quad
\frac{\partial\theta}{\partial\mathbf R_j} = -\frac{\partial\theta}{\partial\mathbf R_i}-\frac{\partial\theta}{\partial\mathbf R_k}$$

(`energy.rs:138-147`). **[sympy 4]** The factor $1/\sin\theta$ is a genuine
coordinate singularity at $\theta = 0,\pi$, where the force direction is
undefined (for $\theta_0\ne\pi$, $E$ is a cone, not differentiable, in the
perpendicular displacement). If $|\mathbf a\times\mathbf b|^2\le10^{-20}$ the
code keeps the finite energy and adds no gradient for that angle, the same
convention as the degenerate torsion (section 4.4). This is the symmetric
subgradient (ferric's convention; not claimed to match any other code). Pinned by `exactly_linear_valence_angle_gradient_is_finite_and_zero`
(`tests/terms_fd.rs`); near-linear angles (0.6 deg) still match central
differences (`torsion_gradient_vs_fd_near_phi_0_pi_and_near_linear_angle`).

## 3. Periodic torsions: proper and improper

$$E_\text{torsion} = \sum_{\text{torsions}} k_\phi\,\big[1 + \cos(n\phi - \delta)\big],\qquad n\in\mathbb Z_{\ge1},$$

$\delta$ in radians (`energy.rs:157-200`; `src/topology.rs:53`; $n\ge1$ is
enforced in `MmTopology::new`). Impropers are torsions with a caller-chosen
atom order; nothing distinguishes them in the energy. Several terms on one
$i$-$j$-$k$-$l$ quartet are separate list entries. By the chain rule

$$\frac{\partial E}{\partial\mathbf R_a} = -k_\phi\,n\sin(n\phi-\delta)\;\frac{\partial\phi}{\partial\mathbf R_a}\qquad (a=i,j,k,l)$$

(`energy.rs:163`). Because $n$ is an integer, $E$ is $2\pi$-periodic in $\phi$,
so the $\pm\pi$ branch cut of $\phi$ is invisible to the energy and gradient.

## 4. The dihedral angle $\phi$ and its gradient (Bekker form)

### 4.1 Definition and sign

For the quartet $i$-$j$-$k$-$l$ (`energy.rs:263-310`, `dihedral_and_gradient`):

$$\mathbf r_{ij}=\mathbf R_i-\mathbf R_j,\quad \mathbf r_{kj}=\mathbf R_k-\mathbf R_j,\quad \mathbf r_{kl}=\mathbf R_k-\mathbf R_l,$$
$$\mathbf m = \mathbf r_{ij}\times\mathbf r_{kj},\qquad \mathbf n = \mathbf r_{kj}\times\mathbf r_{kl},$$
$$\cos\phi = \frac{\mathbf m\cdot\mathbf n}{|\mathbf m||\mathbf n|},\qquad
\operatorname{sign}\phi = \operatorname{sign}(\mathbf r_{ij}\cdot\mathbf n),\qquad \phi\in(-\pi,\pi].$$

$\mathbf m$ and $\mathbf n$ are normals to the planes $ijk$ and $jkl$. $\phi=0$
is the cis (eclipsed) arrangement, $\phi=\pm\pi$ trans. The code evaluates
$\cos\phi$ with `clamp(-1,1)` then `acos` (`energy.rs:283-291`), and flips the
sign when $\mathbf r_{ij}\cdot\mathbf n<0$.

**Equivalent atan2 form [sympy 1].** The identity
$\mathbf m\times\mathbf n = \mathbf r_{kj}\,(\mathbf r_{ij}\cdot\mathbf n)$
(exact, symbolic) gives $|\mathbf m||\mathbf n|\sin\phi = |\mathbf r_{kj}|\,(\mathbf r_{ij}\cdot\mathbf n)$, so

$$\phi = \operatorname{atan2}\big(|\mathbf r_{kj}|\,(\mathbf r_{ij}\cdot\mathbf n),\ \mathbf m\cdot\mathbf n\big).$$

The gradient below is the exact derivative of this function.

### 4.2 Gradient

With $s=|\mathbf r_{kj}|$ (Bekker et al.; Blondel and Karplus, J. Comput. Chem.
17, 1132 (1996), as in GROMACS; here as $\partial\phi/\partial\mathbf R$, the
negative of the published force direction), `energy.rs:299-309`:

$$\frac{\partial\phi}{\partial\mathbf R_i} = \frac{s}{|\mathbf m|^2}\,\mathbf m,\qquad
\frac{\partial\phi}{\partial\mathbf R_l} = -\frac{s}{|\mathbf n|^2}\,\mathbf n,$$
$$p = \frac{\mathbf r_{ij}\cdot\mathbf r_{kj}}{s^2},\quad q = \frac{\mathbf r_{kl}\cdot\mathbf r_{kj}}{s^2},$$
$$\frac{\partial\phi}{\partial\mathbf R_j} = -\frac{\partial\phi}{\partial\mathbf R_i} + p\,\frac{\partial\phi}{\partial\mathbf R_i} - q\,\frac{\partial\phi}{\partial\mathbf R_l},\qquad
\frac{\partial\phi}{\partial\mathbf R_k} = -\frac{\partial\phi}{\partial\mathbf R_l} - p\,\frac{\partial\phi}{\partial\mathbf R_i} + q\,\frac{\partial\phi}{\partial\mathbf R_l}.$$

**[sympy 2]** These twelve components equal the symbolic derivative of the
atan2 form at three rational non-planar geometries (one within 0.01 Bohr of
$\phi=0$), residual below $10^{-50}$. Controls: negating $\partial\phi/\partial\mathbf R_i$
(max deviation 2.0) and swapping $p\leftrightarrow q$ (0.45) are both detected.

**[sympy 3]** Translation invariance, $\sum_a\partial\phi/\partial\mathbf R_a = 0$,
holds identically in $p,q$; rotation invariance,
$\sum_a\mathbf R_a\times\partial\phi/\partial\mathbf R_a = 0$, holds at a generic geometry.

### 4.3 Conditioning at $\phi\approx0,\pi$

$\mathbf m,\mathbf n$ are nonzero at $\phi=0,\pi$ (they vanish only for linear
valence angles), so the gradient is smooth there. Only the *value* of $\phi$
loses accuracy: $\arccos$ near $\cos\phi=\pm1$ has absolute error
$\sim\epsilon/\sin\phi$. This is invisible in $E$ and $\partial E/\partial\phi$
when the phase is $\delta\in\{0,\pi\}$ ($E$ is flat in $\phi$) and enters a
phase $\delta=\pi/2$ energy linearly. The gradient does not use $\phi$'s value
except through $\sin(n\phi-\delta)$.
`torsion_gradient_vs_fd_near_phi_0_pi_and_near_linear_angle`
(`tests/terms_fd.rs`) checks $\phi\in\{10^{-3},10^{-6},\pi-10^{-3},\pi-10^{-6}\}$
and a valence angle 0.6 deg from linear, at $n=1,3$ with $\delta=\pi/2$, against
central differences with a tolerance derived from the step.

### 4.4 Degenerate geometry

If $|\mathbf m|^2$ or $|\mathbf n|^2\le10^{-20}$ (i-j-k or j-k-l linear), the
dihedral plane is undefined and $\partial\phi/\partial\mathbf R$ is singular.
The code takes $\phi=0$ (finite energy) and returns a zero gradient for that
torsion (`energy.rs:281-285`, `energy.rs:299`). Pinned by
`collinear_valence_angle_torsion_gradient_is_finite_and_zero`
(`tests/terms_fd.rs`).

## 5. Nonbonded: Lennard-Jones and Coulomb

For every pair $i<j$ that is not excluded (`energy.rs:202-250`):

$$E_\text{LJ} = \sum_{i<j} c^\text{LJ}_{ij}\;4\varepsilon_{ij}\Big[\Big(\frac{\sigma_{ij}}{r_{ij}}\Big)^{12}-\Big(\frac{\sigma_{ij}}{r_{ij}}\Big)^{6}\Big],\qquad
E_\text{Coul} = \sum_{i<j} c^\text{Coul}_{ij}\,\frac{q_iq_j}{r_{ij}}.$$

Lorentz-Berthelot mixing (`energy.rs:253`):
$\sigma_{ij}=\tfrac12(\sigma_i+\sigma_j)$, $\varepsilon_{ij}=\sqrt{\varepsilon_i\varepsilon_j}$.
Pairs with $\varepsilon_{ij}=0$ or $q_iq_j=0$ (or a zero scale) are skipped.
All pairs are summed directly: no cutoff, no periodic images, no long-range
correction. Coulomb is in atomic units (no $1/4\pi\epsilon_0$ factor).

With $x=(\sigma_{ij}/r)^6$ and $\hat{\mathbf r}=\mathbf r_{ij}/r$ (`energy.rs:225,237`):

$$\frac{\partial E_\text{LJ}}{\partial\mathbf R_i}=c^\text{LJ}\,\frac{4\varepsilon}{r}\,(6x-12x^2)\,\hat{\mathbf r},\qquad
\frac{\partial E_\text{Coul}}{\partial\mathbf R_i}=-c^\text{Coul}\,\frac{q_iq_j}{r^2}\,\hat{\mathbf r},$$

and the opposite sign on atom $j$.

## 6. Exclusions and 1-4 scaling

Both sets are derived from the bond graph in `MmTopology::new`
(`src/topology.rs:130-245`) by breadth-first search to depth 3, recording each
atom's shortest bond-path length $d$ from the start atom:

| $d$ | class | $c^\text{LJ}_{ij}$ | $c^\text{Coul}_{ij}$ |
|---|---|---|---|
| 1, 2 (1-2, 1-3) | excluded | 0 | 0 |
| 3 (1-4) | scaled | `scale_lj_14` | `scale_coul_14` |
| other | normal | 1 | 1 |

Defaults are the AMBER values $c^\text{LJ}_{14}=0.5$ and
$c^\text{Coul}_{14}=1/1.2$ (`src/topology.rs:68-70`); override with
`MmTopology::with_scales` (`src/topology.rs:259`). Because $d$ is the *shortest*
path, a pair reachable by both a path of length $\le2$ and a path of length 3
(e.g. meta carbons in a six-membered ring, any pair in rings of size 3-5) is
excluded, never scaled; the two sets are disjoint (`tests/ring_exclusions.rs`).
The term skip is applied identically to LJ and Coulomb (`energy.rs:206-215`).

## 7. QM-MM Lennard-Jones

`qm_mm_lj_energy_gradient_scaled` (`energy.rs:340`) sums the section 5 LJ term
over all QM atom $a$ and MM atom $b$ pairs with the same mixing rule and a
caller-supplied per-pair factor $c_{ab}$ (`0` skips the pair, `1` is
bit-identical to the unscaled `qm_mm_lj_energy_gradient`, `energy.rs:323`).
There are no exclusions unless the caller encodes them in $c_{ab}$.
