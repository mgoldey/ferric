# The molecular-mechanics force field {#molecular-mechanics}

<!-- status: ready; registry: mm -->

Point charges give the QM region the electrostatics of its surroundings and
nothing else. They do not hold the MM atoms together, they do not keep atoms
from overlapping, and they carry no dispersion. A force field supplies those
terms with simple functions of the geometry: springs for bonds and angles, a
periodic potential for rotation about bonds, and a pairwise Lennard-Jones and
Coulomb sum between atoms that are not bonded to each other.

`ferric-mm` is arithmetic only. It takes a topology supplied entirely by the
caller and returns an AMBER-form energy and its analytic gradient. It assigns
no parameters. Turning a protein into a force field is a separate step; the
repository's way of doing it is `scripts/validation/mm_topology.py` (the
function `topology_from_openmm`), which reads the numbers out of a real OpenMM
system. In QM/MM the force field supplies the terms the wavefunction does not
describe: the MM–MM energy and the Lennard-Jones between the QM and MM atoms
(section @sec:qmmm-boundaries-additive-mm-terms).

The section goes term by term: bonds, angles and torsions (@sec:molecular-mechanics-bonds-angles-torsions), the
dihedral gradient (@sec:molecular-mechanics-dihedral-angle-gradient), the nonbonded sum and its exclusions (@sec:molecular-mechanics-nonbonded-terms-exclusions-1),
the QM–MM Lennard-Jones (@sec:molecular-mechanics-qm-mm-lennard-jones) and the optional cutoff (@sec:molecular-mechanics-opt-nonbonded-cutoff), then a
Python example (@sec:molecular-mechanics-running-it). The
proofs are in `scripts/mm_dihedral_gradient_sympy.py` (60-digit arithmetic,
a negative control per claim) and `scripts/mm_cutoff_sympy.py` (the cutoff
identities, and the bond and pair gradients in claim 6).

| Symbol | Meaning | Code name |
|---|---|---|
| $\mathbf R_a$ | position of atom $a$, Bohr | rows of `coords` |
| $\mathbf r_{ab}$ | $\mathbf R_a - \mathbf R_b$ | `diff(coords, a, b)` |
| $k$, $r_0$ | bond constant and length | `Bond::{k,r0}` |
| $k_\theta$, $\theta_0$ | angle constant and equilibrium angle | `Angle::{k_theta,theta0}` |
| $k_\phi$, $n$, $\delta$ | torsion constant, periodicity, phase | `Torsion` |
| $\sigma$, $\varepsilon$ | Lennard-Jones size and depth | `LjParams` |
| $q_i$ | partial charge | `MmTopology::charges` |
| $c^{\mathrm{LJ}}_{ij}$, $c^{\mathrm{Coul}}_{ij}$ | pair scale factors | `scale_lj_14`, `scale_coul_14` |
| $r_c$, $\varepsilon_{\mathrm{rf}}$, $r_{\mathrm{sw}}$ | cutoff, reaction-field dielectric, LJ switch start | `CutoffParams` |

| Entry point | Location |
|---|---|
| Rust | `ferric-mm/src/energy.rs::{energy,gradient,qm_mm_lj_energy_gradient,qm_mm_lj_energy_gradient_scaled}`, `ferric-mm/src/cutoff.rs::{energy_with,gradient_with,neighbor_pairs}`, `ferric-mm/src/topology.rs::MmTopology` |
| CLI | none |
| Python | `MmTopology.from_amber_units`, used through `run_qmmm(mm_topology=)` and `run_optimize_qmmm(mm_topology=)`; `QmmmResult.mm_energy()` |

Units are Hartree atomic units: Hartree, Bohr, radians, charge in $e$.
`MmTopology::from_amber_units` converts once from kcal/mol, Å and degrees.
Gradients are $\partial E/\partial\mathbf R$, not forces. The total energy is
the sum of five terms, the fields of `MmEnergy`:

$$ E = E_{\mathrm{bond}} + E_{\mathrm{angle}} + E_{\mathrm{torsion}} + E_{\mathrm{LJ}} + E_{\mathrm{Coul}} . $$

## Bonds, angles and torsions {#molecular-mechanics-bonds-angles-torsions}

**Bonds.** A bond is a harmonic spring about its equilibrium length:

$$ E_{\mathrm{bond}} = \sum_{\text{bonds}} k\,(r - r_0)^2, \qquad r = |\mathbf r_{ij}| . $$

There is no leading $\tfrac12$. That is the AMBER convention, and it means
the constants are half of those a $\tfrac12k(r-r_0)^2$ code uses: OpenMM's
`HarmonicBondForce` has $k_{\mathrm{omm}} = 2k$. For the gradient,
differentiate $(r-r_0)^2$ to $2(r-r_0)$ and use $\partial r/\partial\mathbf R_i = \mathbf r_{ij}/r$:

$$ \frac{\partial E}{\partial\mathbf R_i} = 2k(r-r_0)\,\frac{\mathbf r_{ij}}{r} = -\frac{\partial E}{\partial\mathbf R_j} . $$

> **Verified equation {#eq:mm-bond-gradient}** · the bond gradient
>
> The rows $\pm 2k(r-r_0)\,\mathbf r_{ij}/r$ equal the symbolic gradient of
> $k(r-r_0)^2$ with respect to all six coordinates of the two atoms.
>
> | | |
> |---|---|
> | Proof | `scripts/mm_cutoff_sympy.py`, claim 6 |
> | Negative control | the runner drops the factor 2 from `bond_dedr` |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mm/src/energy.rs::bonded_terms` |
> | Code test | `bond_energy_hand_computed_and_gradient_vs_fd` in `crates/ferric-mm/tests/terms_fd.rs` |
>
> <!-- verified: id=mm.bond.gradient; script=scripts/mm_cutoff_sympy.py; claim=6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mm/src/energy.rs::bonded_terms; rust_test=ferric-mm/tests/terms_fd.rs::bond_energy_hand_computed_and_gradient_vs_fd; registry=mm -->

**Angles.** The valence angle $\theta\in[0,\pi]$ at the central atom $j$ is
also harmonic, again with no leading $\tfrac12$:

$$ E_{\mathrm{angle}} = \sum_{\text{angles}} k_\theta\,(\theta - \theta_0)^2, \qquad \cos\theta = \frac{\mathbf r_{ij}\cdot\mathbf r_{kj}}{|\mathbf r_{ij}|\,|\mathbf r_{kj}|} . $$

Write $\mathbf a = \mathbf r_{ij}$ and $\mathbf b = \mathbf r_{kj}$. The chain
rule gives $\partial E/\partial\theta = 2k_\theta(\theta-\theta_0)$ times
$\partial\theta/\partial\mathbf R$. To get the angle derivative, differentiate
$\cos\theta$ with respect to $\mathbf a$,
$\partial\cos\theta/\partial\mathbf a = \mathbf b/(|\mathbf a||\mathbf b|) - \cos\theta\,\mathbf a/|\mathbf a|^2$,
and divide by $d\cos\theta/d\theta = -\sin\theta$. Atom $k$ follows by
swapping $\mathbf a$ and $\mathbf b$, and atom $j$ by translation invariance:

$$ \frac{\partial\theta}{\partial\mathbf R_i} = -\frac{1}{\sin\theta}\Big(\frac{\mathbf b}{|\mathbf a||\mathbf b|} - \cos\theta\,\frac{\mathbf a}{|\mathbf a|^2}\Big),\qquad \frac{\partial\theta}{\partial\mathbf R_k} = -\frac{1}{\sin\theta}\Big(\frac{\mathbf a}{|\mathbf a||\mathbf b|} - \cos\theta\,\frac{\mathbf b}{|\mathbf b|^2}\Big),\qquad \frac{\partial\theta}{\partial\mathbf R_j} = -\frac{\partial\theta}{\partial\mathbf R_i}-\frac{\partial\theta}{\partial\mathbf R_k} . $$

The kernel in `ferric-mm/src/energy.rs::bonded_terms` computes the first two
for each Cartesian component; `rji` is $\mathbf R_i - \mathbf R_j = \mathbf a$:

<!-- excerpt: ferric-mm/src/energy.rs::bonded_terms -->
```rust
for c in 0..3 {
    dti[c] = -inv_sin * (rjk[c] / (rji_n * rjk_n) - cos_t * rji[c] / (rji_n * rji_n));
    dtk[c] = -inv_sin * (rji[c] / (rji_n * rjk_n) - cos_t * rjk[c] / (rjk_n * rjk_n));
}
let dtj = [-(dti[0] + dtk[0]), -(dti[1] + dtk[1]), -(dti[2] + dtk[2])];
```

| Symbol | Code |
|---|---|
| $-1/\sin\theta$ | `-inv_sin` |
| $\mathbf b/(\lvert\mathbf a\rvert\lvert\mathbf b\rvert)$ | `rjk[c] / (rji_n * rjk_n)` |
| $\cos\theta$ | `cos_t`, clamped to $[-1, 1]$ before `acos` |
| $\mathbf a/\lvert\mathbf a\rvert^2$ | `rji[c] / (rji_n * rji_n)` |
| $\partial\theta/\partial\mathbf R_i$, $\partial\theta/\partial\mathbf R_k$, $\partial\theta/\partial\mathbf R_j$ | `dti`, `dtk`, `dtj` |

> **Verified equation {#eq:mm-angle-gradient}** · the bond-angle gradient
>
> The closed form above equals the exact derivative of
> $\theta = \arccos\big(\mathbf r_{ji}\cdot\mathbf r_{jk}/(|\mathbf r_{ji}||\mathbf r_{jk}|)\big)$
> with respect to the three atoms, at concrete rational geometries to 60
> digits.
>
> | | |
> |---|---|
> | Proof | `scripts/mm_dihedral_gradient_sympy.py`, claim 4 |
> | Negative control | dropping the $\cos\theta$ term; the runner normalizes it by $\lvert\mathbf r_{ji}\rvert^2$ instead of $\lvert\mathbf r_{jk}\rvert^2$ for atom $k$ |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mm/src/energy.rs::bonded_terms` |
> | Code test | `angle_energy_hand_computed_and_gradient_vs_fd` in `crates/ferric-mm/tests/terms_fd.rs` |
>
> <!-- verified: id=mm.angle.gradient; script=scripts/mm_dihedral_gradient_sympy.py; claim=4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mm/src/energy.rs::bonded_terms; rust_test=ferric-mm/tests/terms_fd.rs::angle_energy_hand_computed_and_gradient_vs_fd; registry=mm -->

What happens at a linear angle? At $\theta = 0$ or $\pi$ the factor
$1/\sin\theta$ is a coordinate singularity, and the direction of the force is
undefined. For $\theta_0\ne\pi$ the energy is a cone in the perpendicular
displacement, not differentiable at its tip. When
$|\mathbf a\times\mathbf b|^2\le10^{-20}$ the code keeps the finite energy and
adds no gradient for that angle, the same convention as the degenerate
torsion of @sec:molecular-mechanics-dihedral-angle-gradient. This zero is the symmetric subgradient. It is ferric's
convention, and no claim is made that other codes match it.
`exactly_linear_valence_angle_gradient_is_finite_and_zero` in
`crates/ferric-mm/tests/terms_fd.rs` pins it. Angles near linear but not
exactly linear (0.6° from linear) still match central differences
(`torsion_gradient_vs_fd_near_phi_0_pi_and_near_linear_angle`).

**Torsions.** Rotation about a bond has a periodic barrier, modelled as a
cosine series in the dihedral angle $\phi$:

$$ E_{\mathrm{torsion}} = \sum_{\text{torsions}} k_\phi\,\big[1 + \cos(n\phi - \delta)\big], \qquad n\in\mathbb Z_{\ge1}, $$

with $\delta$ in radians. `MmTopology::new` enforces $n\ge1$. An improper
torsion is a torsion with a caller-chosen atom order; nothing distinguishes
the two in the energy. Several terms on one $i$-$j$-$k$-$l$ quartet are
separate list entries. By the chain rule,

$$ \frac{\partial E}{\partial\mathbf R_a} = -k_\phi\,n\sin(n\phi-\delta)\;\frac{\partial\phi}{\partial\mathbf R_a}\qquad (a=i,j,k,l) . $$

$n$ is an integer, so $E$ is $2\pi$-periodic in $\phi$, and the $\pm\pi$
branch cut of $\phi$ is invisible to the energy and the gradient. All that
remains is $\partial\phi/\partial\mathbf R$.

## The dihedral angle and its gradient {#molecular-mechanics-dihedral-angle-gradient}

For the quartet $i$-$j$-$k$-$l$ take the three bond vectors and the normals
to the planes $ijk$ and $jkl$:

$$ \mathbf r_{ij}=\mathbf R_i-\mathbf R_j,\quad \mathbf r_{kj}=\mathbf R_k-\mathbf R_j,\quad \mathbf r_{kl}=\mathbf R_k-\mathbf R_l,\qquad \mathbf m = \mathbf r_{ij}\times\mathbf r_{kj},\quad \mathbf n = \mathbf r_{kj}\times\mathbf r_{kl} . $$

The dihedral is the angle between the normals, with a sign:

$$ \cos\phi = \frac{\mathbf m\cdot\mathbf n}{|\mathbf m||\mathbf n|},\qquad \operatorname{sign}\phi = \operatorname{sign}(\mathbf r_{ij}\cdot\mathbf n),\qquad \phi\in(-\pi,\pi] . $$

$\phi = 0$ is the cis (eclipsed) arrangement and $\phi = \pm\pi$ is trans. The
code evaluates $\cos\phi$, clamps it to $[-1, 1]$, takes `acos`, and flips
the sign when $\mathbf r_{ij}\cdot\mathbf n<0$.

An equivalent form is easier to differentiate. The vector identity
$\mathbf m\times\mathbf n = \mathbf r_{kj}\,(\mathbf r_{ij}\cdot\mathbf n)$ holds
exactly (claim 1). Take its magnitude:
$|\mathbf m||\mathbf n|\sin\phi = |\mathbf r_{kj}|\,(\mathbf r_{ij}\cdot\mathbf n)$.
Together with the cosine this gives

$$ \phi = \operatorname{atan2}\big(|\mathbf r_{kj}|\,(\mathbf r_{ij}\cdot\mathbf n),\ \mathbf m\cdot\mathbf n\big), $$

and the gradient below is the exact derivative of this function. With
$s = |\mathbf r_{kj}|$, the two end atoms move only their own plane, so their
derivatives point along the normals:

$$ \frac{\partial\phi}{\partial\mathbf R_i} = \frac{s}{|\mathbf m|^2}\,\mathbf m,\qquad \frac{\partial\phi}{\partial\mathbf R_l} = -\frac{s}{|\mathbf n|^2}\,\mathbf n . $$

The two middle atoms follow from translation invariance and from the
projections of the outer bonds onto the central one,

$$ p = \frac{\mathbf r_{ij}\cdot\mathbf r_{kj}}{s^2},\quad q = \frac{\mathbf r_{kl}\cdot\mathbf r_{kj}}{s^2},\qquad \frac{\partial\phi}{\partial\mathbf R_j} = -\frac{\partial\phi}{\partial\mathbf R_i} + p\,\frac{\partial\phi}{\partial\mathbf R_i} - q\,\frac{\partial\phi}{\partial\mathbf R_l},\qquad \frac{\partial\phi}{\partial\mathbf R_k} = -\frac{\partial\phi}{\partial\mathbf R_l} - p\,\frac{\partial\phi}{\partial\mathbf R_i} + q\,\frac{\partial\phi}{\partial\mathbf R_l} . $$

These are the Bekker formulas (Blondel and Karplus, J. Comput. Chem. 17, 1132
(1996), as in GROMACS), written here as $\partial\phi/\partial\mathbf R$, the
negative of the published force direction. Check translation invariance in
your head: add the four rows, and the $p$ and $q$ terms cancel pairwise,
leaving $\partial\phi/\partial\mathbf R_i - \partial\phi/\partial\mathbf R_i + \partial\phi/\partial\mathbf R_l - \partial\phi/\partial\mathbf R_l = 0$
for any $p$ and $q$ (claim 3).

The twelve components are `ferric-mm/src/energy.rs::dihedral_and_gradient`:

<!-- excerpt: ferric-mm/src/energy.rs::dihedral_and_gradient -->
```rust
if m_sq > 1e-20 && n_sq > 1e-20 && kj_norm > 1e-20 {
    for c in 0..3 {
        grad_i[c] = (kj_norm / m_sq) * m[c];
        grad_l[c] = -(kj_norm / n_sq) * n[c];
    }
    let p = dot(r_ij, r_kj) / (kj_norm * kj_norm);
    let q = dot(r_kl, r_kj) / (kj_norm * kj_norm);
    for c in 0..3 {
        grad_j[c] = -grad_i[c] + p * grad_i[c] - q * grad_l[c];
        grad_k[c] = -grad_l[c] - p * grad_i[c] + q * grad_l[c];
    }
}
```

| Symbol | Code |
|---|---|
| $s = \lvert\mathbf r_{kj}\rvert$ | `kj_norm` |
| $\lvert\mathbf m\rvert^2$, $\lvert\mathbf n\rvert^2$ | `m_sq`, `n_sq` |
| $\partial\phi/\partial\mathbf R_i$, $\partial\phi/\partial\mathbf R_l$ | `grad_i`, `grad_l` |
| $p$, $q$ | `p`, `q` |
| $\partial\phi/\partial\mathbf R_j$, $\partial\phi/\partial\mathbf R_k$ | `grad_j`, `grad_k` |

> **Verified equation {#eq:mm-dihedral-gradient}** · the dihedral gradient
>
> The code computes $\phi$ by $\arccos$ with the sign of
> $\mathbf r_{ij}\cdot\mathbf n$. The identity
> $\mathbf m\times\mathbf n = \mathbf r_{kj}(\mathbf r_{ij}\cdot\mathbf n)$ makes that equal to
> $\operatorname{atan2}(|\mathbf r_{kj}|\,\mathbf r_{ij}\cdot\mathbf n,\ \mathbf m\cdot\mathbf n)$
> (claim 1). The twelve gradient components equal the symbolic derivative of
> the atan2 form at three rational non-planar geometries (one within 0.01 Bohr
> of $\phi = 0$), residual below $10^{-50}$ (claim 2). They sum to zero over
> the atoms, and the rotational sum $\sum_a\mathbf R_a\times\partial\phi/\partial\mathbf R_a$
> vanishes at a generic geometry (claim 3).
>
> | | |
> |---|---|
> | Proof | `scripts/mm_dihedral_gradient_sympy.py`, claims 1, 2 and 3 |
> | Negative control | negating $\partial\phi/\partial\mathbf R_i$ (max deviation 2.0) and swapping $p\leftrightarrow q$ (0.45) are both detected |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mm/src/energy.rs::dihedral_and_gradient` |
> | Code test | `torsion_gradient_vs_fd_near_phi_0_pi_and_near_linear_angle` in `crates/ferric-mm/tests/terms_fd.rs` |
>
> <!-- verified: id=mm.dihedral.gradient; script=scripts/mm_dihedral_gradient_sympy.py; claim=2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mm/src/energy.rs::dihedral_and_gradient; rust_test=ferric-mm/tests/terms_fd.rs::torsion_gradient_vs_fd_near_phi_0_pi_and_near_linear_angle; registry=mm -->

**Near cis and trans.** $\mathbf m$ and $\mathbf n$ are nonzero at
$\phi = 0, \pi$; they vanish only for linear valence angles. So the gradient
is smooth there. Only the value of $\phi$ loses accuracy: $\arccos$ near
$\cos\phi = \pm1$ has absolute error about $\epsilon/\sin\phi$. For a phase
$\delta\in\{0,\pi\}$ that error does not show in $E$ or $\partial E/\partial\phi$,
because $E$ is flat in $\phi$ there. For $\delta = \pi/2$ it enters the
energy linearly. The gradient uses the value of $\phi$ only through
$\sin(n\phi-\delta)$. `torsion_gradient_vs_fd_near_phi_0_pi_and_near_linear_angle`
checks $\phi\in\{10^{-3},10^{-6},\pi-10^{-3},\pi-10^{-6}\}$ and a valence angle
0.6° from linear, at $n = 1, 3$ with $\delta = \pi/2$, against central
differences with a tolerance derived from the step.

**Degenerate geometry.** When $|\mathbf m|^2$ or $|\mathbf n|^2$ is at most
$10^{-20}$ (a linear $i$-$j$-$k$ or $j$-$k$-$l$) the plane is undefined and
$\partial\phi/\partial\mathbf R$ is singular. The code takes $\phi = 0$, a finite
energy and a zero gradient for that torsion (the guard at the top of the
excerpt); `collinear_valence_angle_torsion_gradient_is_finite_and_zero` in
`crates/ferric-mm/tests/terms_fd.rs` pins it.

## Nonbonded terms, exclusions and 1–4 scaling {#molecular-mechanics-nonbonded-terms-exclusions-1}

Atoms that are not bonded to each other interact through two pair terms. The
Lennard-Jones term is repulsive at short range, where electron clouds
overlap, and attractive as $-r^{-6}$ at long range, the dispersion
interaction. The Coulomb term is the interaction of the partial charges. For
every pair $i<j$ that is not excluded:

$$ E_{\mathrm{LJ}} = \sum_{i<j} c^{\mathrm{LJ}}_{ij}\;4\varepsilon_{ij}\Big[\Big(\frac{\sigma_{ij}}{r_{ij}}\Big)^{12}-\Big(\frac{\sigma_{ij}}{r_{ij}}\Big)^{6}\Big],\qquad E_{\mathrm{Coul}} = \sum_{i<j} c^{\mathrm{Coul}}_{ij}\,\frac{q_iq_j}{r_{ij}} . $$

Set the derivative of the bracket to zero and the minimum is at
$r = 2^{1/6}\sigma_{ij}$, with energy $-\varepsilon_{ij}$: $\sigma$ sets the
size and $\varepsilon$ the depth. The pair parameters come from
Lorentz–Berthelot mixing (`mix`), $\sigma_{ij}=\tfrac12(\sigma_i+\sigma_j)$ and
$\varepsilon_{ij}=\sqrt{\varepsilon_i\varepsilon_j}$. A pair with
$\varepsilon_{ij}=0$, $q_iq_j=0$ or a zero scale is skipped. Coulomb is in
atomic units, with no $1/4\pi\epsilon_0$. By default
(`NonbondedMethod::NoCutoff`) all pairs are summed directly: no cutoff, no
periodic images, no long-range correction. The opt-in cutoff is section
@sec:molecular-mechanics-opt-nonbonded-cutoff.

With $x=(\sigma_{ij}/r)^6$ and $\hat{\mathbf r}=\mathbf r_{ij}/r$, differentiate
each term in $r$ and multiply by $\partial r/\partial\mathbf R_i = \hat{\mathbf r}$:

$$ \frac{\partial E_{\mathrm{LJ}}}{\partial\mathbf R_i}=c^{\mathrm{LJ}}\,\frac{4\varepsilon}{r}\,(6x-12x^2)\,\hat{\mathbf r},\qquad \frac{\partial E_{\mathrm{Coul}}}{\partial\mathbf R_i}=-c^{\mathrm{Coul}}\,\frac{q_iq_j}{r^2}\,\hat{\mathbf r}, $$

with the opposite sign on atom $j$. One function evaluates the pair, and the
all-pairs sum and the 1–4 pairs of the cutoff path both call it
(`plain_pair`):

<!-- excerpt: ferric-mm/src/energy.rs::plain_pair -->
```rust
let mixed = mix(top.lj[i], top.lj[j]);
if mixed.epsilon != 0.0 && scale_lj != 0.0 {
    let sr6 = (mixed.sigma / r).powi(6);
    let sr12 = sr6 * sr6;
    let e_lj = scale_lj * 4.0 * mixed.epsilon * (sr12 - sr6);
    e.lj += e_lj;
    let dedr = scale_lj * 4.0 * mixed.epsilon * (-12.0 * sr12 + 6.0 * sr6) / r;
```

| Symbol | Code |
|---|---|
| $\sigma_{ij}$, $\varepsilon_{ij}$ (Lorentz–Berthelot) | `mixed.sigma`, `mixed.epsilon`, from `mix` |
| $x = (\sigma_{ij}/r)^6$ | `sr6` |
| $x^2 = (\sigma_{ij}/r)^{12}$ | `sr12` |
| $c^{\mathrm{LJ}}_{ij}$ | `scale_lj` |
| $E_{\mathrm{LJ}}$ of the pair | `e_lj` |
| $\partial E_{\mathrm{LJ}}/\partial r$ | `dedr`, then times $\pm\hat{\mathbf r}$ on the two atoms |

> **Verified equation {#eq:mm-pair-gradient}** · the Lennard-Jones and Coulomb pair gradients
>
> With the 1–4 scale factors left symbolic, the rows
> $\pm(\partial E/\partial r)\,\hat{\mathbf r}$ built from the code's `dedr`
> expressions equal the symbolic gradient of each pair energy with respect to
> both atoms.
>
> | | |
> |---|---|
> | Proof | `scripts/mm_cutoff_sympy.py`, claim 6 |
> | Negative control | a Coulomb `dedr` of $+q_iq_j/r^2$ fails the same check |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mm/src/energy.rs::plain_pair` |
> | Code test | `nonbonded_hand_computed_and_gradient_vs_fd` in `crates/ferric-mm/tests/terms_fd.rs` |
>
> <!-- verified: id=mm.pair.gradient; script=scripts/mm_cutoff_sympy.py; claim=6; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mm/src/energy.rs::plain_pair; rust_test=ferric-mm/tests/terms_fd.rs::nonbonded_hand_computed_and_gradient_vs_fd; registry=mm -->

### Which pairs count

Atoms one or two bonds apart already interact through the bond and angle
terms, so their nonbonded terms would double-count. Atoms three bonds apart,
the ends of a torsion, get a reduced nonbonded term. `MmTopology::new` finds
both sets once, from the bond graph, by a breadth-first search to depth 3
that records each atom's shortest bond-path length $d$ from the start atom:

| $d$ | class | $c^{\mathrm{LJ}}_{ij}$ | $c^{\mathrm{Coul}}_{ij}$ |
|---|---|---|---|
| 1, 2 (1–2, 1–3) | excluded | 0 | 0 |
| 3 (1–4) | scaled | `scale_lj_14` | `scale_coul_14` |
| other | normal | 1 | 1 |

The defaults are the AMBER values $c^{\mathrm{LJ}}_{14}=0.5$ and
$c^{\mathrm{Coul}}_{14}=1/1.2$ (`DEFAULT_SCALE_LJ_14`, `DEFAULT_SCALE_COUL_14`);
`MmTopology::with_scales` overrides them. The skip applies to LJ and Coulomb
alike (`nonbonded_all_pairs`).

Because $d$ is the shortest path, the two sets never overlap. A pair reachable
both by a path of length at most 2 and by a path of length 3 is excluded,
never scaled. That covers the meta carbons of a six-membered ring and any pair
in a ring of three to five atoms (`crates/ferric-mm/tests/ring_exclusions.rs`).
Editing `bonds` after construction does not update the sets.

## The QM–MM Lennard-Jones term {#molecular-mechanics-qm-mm-lennard-jones}

The QM atoms have no MM charges in QM/MM (the wavefunction carries their
electrostatics), but the wavefunction has no dispersion or overlap repulsion
toward the MM atoms. The QM–MM Lennard-Jones term supplies it.
`qm_mm_lj_energy_gradient_scaled` sums the LJ term of @sec:molecular-mechanics-nonbonded-terms-exclusions-1 over every pair
of a QM atom $a$ and an MM atom $b$, with the same mixing rule and a
caller-supplied factor $c_{ab}$ per pair. $c_{ab} = 0$ skips the pair, and
$c_{ab} = 1$ everywhere is bit-identical to the unscaled
`qm_mm_lj_energy_gradient`. There are no exclusions unless the caller encodes
them in $c_{ab}$.

`qmmm_mm_terms` (section @sec:qmmm-boundaries-additive-mm-terms) does encode them. It takes the factor from the
topology's own exclusions, so a QM–MM pair that is 1–2 or 1–3 through the
bond graph, the cut bond included, contributes nothing, and a 1–4 pair is
scaled by `scale_lj_14`.

## The opt-in nonbonded cutoff {#molecular-mechanics-opt-nonbonded-cutoff}

The all-pairs sum costs $N^2$. `gradient_with(top, coords, &NonbondedMethod::CutoffNonPeriodic(p))`
(`ferric-mm/src/cutoff.rs`) replaces it with the form of OpenMM's
`NonbondedForce` method `CutoffNonPeriodic`. The bonded terms are unchanged.
The parameters (`CutoffParams`) are a cutoff $r_c>0$, a reaction-field
dielectric $\varepsilon_{\mathrm{rf}}>0$ (default 78.3, OpenMM's), and an
optional switching distance $0\le r_{\mathrm{sw}}<r_c$. Invalid values are
errors.

The three pair classes of section @sec:molecular-mechanics-nonbonded-terms-exclusions-1 are treated differently
(`nonbonded_cutoff`):

| class | range | Coulomb | LJ |
|---|---|---|---|
| 1–2, 1–3 | any | 0 | 0 |
| 1–4 | every $r$ | $c^{\mathrm{Coul}}_{14}\,q_iq_j/r$ | $c^{\mathrm{LJ}}_{14}\,E_{\mathrm{LJ}}(r)$ |
| ordinary | $r<r_c$ | $E_{\mathrm{rf}}(r)$ | $S(r)\,E_{\mathrm{LJ}}(r)$ |

The 1–4 pairs reuse `plain_pair`, because OpenMM evaluates these exceptions
with no cutoff and no reaction field. Excluded pairs get no reaction-field
correction. Ordinary pairs at $r\ge r_c$ contribute nothing.

### Reaction-field Coulomb

Truncating $1/r$ at $r_c$ would make the energy jump. The reaction-field form
treats everything beyond $r_c$ as a dielectric continuum of permittivity
$\varepsilon_{\mathrm{rf}}$, which adds a term in $r^2$, and shifts the pair
energy so that it reaches zero at the cutoff:

$$ E_{\mathrm{rf}}(r) = q_iq_j\Big(\frac1r + k_{\mathrm{rf}}\,r^2 - c_{\mathrm{rf}}\Big),\qquad k_{\mathrm{rf}} = \frac{\varepsilon_{\mathrm{rf}}-1}{(2\varepsilon_{\mathrm{rf}}+1)\,r_c^3} . $$

Require $E_{\mathrm{rf}}(r_c) = 0$ and solve for the shift:

$$ c_{\mathrm{rf}} = \frac1{r_c} + k_{\mathrm{rf}}r_c^2 = \frac{3\varepsilon_{\mathrm{rf}}}{(2\varepsilon_{\mathrm{rf}}+1)\,r_c} $$

(`CutoffParams::reaction_field_constants`). The energy is continuous at the
cutoff; the force is not. Differentiate:

$$ \frac{\partial E_{\mathrm{rf}}}{\partial r} = q_iq_j\Big(-\frac1{r^2} + 2k_{\mathrm{rf}}\,r\Big),\qquad \frac{\partial E_{\mathrm{rf}}}{\partial r}\Big|_{r_c} = -\frac{3\,q_iq_j}{(2\varepsilon_{\mathrm{rf}}+1)\,r_c^2} . $$

Check $\varepsilon_{\mathrm{rf}} = 1$: then $k_{\mathrm{rf}} = 0$, and
$E_{\mathrm{rf}}$ is plain Coulomb shifted by $-q_iq_j/r_c$.

The Coulomb part of the cutoff loop adds the unshifted part and accumulates
$\sum q_iq_j$ over ordinary pairs, to multiply by $c_{\mathrm{rf}}$ and
subtract once after the loop:

<!-- excerpt: ferric-mm/src/cutoff.rs::nonbonded_cutoff -->
```rust
e.coulomb += qq / r + qq * k_rf * r * r;
qq_shift += qq;
let dedr = -qq / (r * r) + 2.0 * qq * k_rf * r;
```

| Symbol | Code |
|---|---|
| $q_iq_j$ | `qq` |
| $q_iq_j(1/r + k_{\mathrm{rf}}r^2)$ | `qq / r + qq * k_rf * r * r` |
| $\sum q_iq_j$ over ordinary pairs | `qq_shift`, multiplied by $c_{\mathrm{rf}}$ after the loop |
| $\partial E_{\mathrm{rf}}/\partial r$ | `dedr` |

> **Verified equation {#eq:mm-cutoff-reaction-field}** · the reaction-field constants and force
>
> $c_{\mathrm{rf}} = 3\varepsilon_{\mathrm{rf}}/((2\varepsilon_{\mathrm{rf}}+1)r_c)$ equals
> $1/r_c + k_{\mathrm{rf}}r_c^2$ with
> $k_{\mathrm{rf}} = (\varepsilon_{\mathrm{rf}}-1)/((2\varepsilon_{\mathrm{rf}}+1)r_c^3)$ (claim 1);
> $E_{\mathrm{rf}}(r_c) = 0$, its slope there is
> $-3q_iq_j/((2\varepsilon_{\mathrm{rf}}+1)r_c^2)$, and the code's
> $dE/dr = q_iq_j(-1/r^2 + 2k_{\mathrm{rf}}r)$ is its derivative (claim 2). The energy is
> continuous at the cutoff and the force is not.
>
> | | |
> |---|---|
> | Proof | `scripts/mm_cutoff_sympy.py`, claims 1 and 2 |
> | Negative control | $c_{\mathrm{rf}}$ with $2\varepsilon$ in the numerator; $dE/dr$ without the factor 2 |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mm/src/cutoff.rs::nonbonded_cutoff`, `ferric-mm/src/cutoff.rs::reaction_field_constants` |
> | Code test | `reaction_field_and_switched_lj_vanish_continuously_at_the_cutoff` in `crates/ferric-mm/tests/cutoff.rs` |
>
> <!-- verified: id=mm.cutoff.reaction-field; script=scripts/mm_cutoff_sympy.py; claim=2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mm/src/cutoff.rs::nonbonded_cutoff; rust_test=ferric-mm/tests/cutoff.rs::reaction_field_and_switched_lj_vanish_continuously_at_the_cutoff; registry=mm -->

### Lennard-Jones truncation and switching

Without $r_{\mathrm{sw}}$, $S=1$ for $r<r_c$: LJ is truncated, not shifted, and
jumps by $E_{\mathrm{LJ}}(r_c)$ at the cutoff. That is OpenMM's convention.
With $r_{\mathrm{sw}}$, a polynomial switch takes LJ smoothly to zero over the
window $r_{\mathrm{sw}}<r<r_c$. Put $x=(r-r_{\mathrm{sw}})/(r_c-r_{\mathrm{sw}})$:

$$ S = 1 - 10x^3 + 15x^4 - 6x^5,\qquad \frac{dS}{dr} = \frac{-30x^2 + 60x^3 - 30x^4}{r_c-r_{\mathrm{sw}}},\qquad \frac{\partial (S E_{\mathrm{LJ}})}{\partial r} = S\,\frac{\partial E_{\mathrm{LJ}}}{\partial r} + E_{\mathrm{LJ}}\,\frac{dS}{dr} . $$

Check the ends. $S(0)=1$ and $S(1)=0$, and $S'$ and $S''$ vanish at both ends,
so the switched LJ energy and force are continuous. The switch applies to LJ
only. The code evaluates $S$ in Horner form:

<!-- excerpt: ferric-mm/src/cutoff.rs::nonbonded_cutoff -->
```rust
let w = rc - rsw;
let x = (r - rsw) / w;
let x2 = x * x;
let s = 1.0 + x2 * x * (-10.0 + x * (15.0 - 6.0 * x));
let ds = x2 * (-30.0 + x * (60.0 - 30.0 * x)) / w;
dedr = dedr * s + e_lj * ds;
e_lj *= s;
```

| Symbol | Code |
|---|---|
| $r_c - r_{\mathrm{sw}}$ | `w` |
| $x = (r-r_{\mathrm{sw}})/(r_c-r_{\mathrm{sw}})$ | `x`, with `x2 = x * x` |
| $S$ | `s` |
| $dS/dr$ | `ds` |
| $\partial(SE_{\mathrm{LJ}})/\partial r$ | `dedr * s + e_lj * ds`, formed before `e_lj` is scaled |

> **Verified equation {#eq:mm-cutoff-switch}** · the switch and the switched derivative
>
> The Horner form equals $1 - 10x^3 + 15x^4 - 6x^5$, with $S(0) = 1$,
> $S(1) = 0$ and $S'$, $S''$ zero at both ends (claim 3). The code's
> $dS/dr$ is the derivative for $x = (r-r_{\mathrm{sw}})/w$, and
> $d(SE_{\mathrm{LJ}})/dr = S\,dE_{\mathrm{LJ}}/dr + E_{\mathrm{LJ}}\,dS/dr$ (claim 4). The
> transcribed floating-point pair energy and $dE/dr$ agree with a central finite
> difference of that energy in 50-digit arithmetic ($h = 10^{-12}$) below $r_{\mathrm{sw}}$
> and inside the window, and $E_{\mathrm{rf}}(r_c) = 0$ numerically (claim 5).
>
> | | |
> |---|---|
> | Proof | `scripts/mm_cutoff_sympy.py`, claims 3, 4 and 5 |
> | Negative control | $dS/dr$ without the $1/w$; the switched derivative with the wrong sign; the runner drops the $1/w$ and the factor 2 of the transcribed code |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-mm/src/cutoff.rs::nonbonded_cutoff` |
> | Code test | `gradient_matches_fd_with_pairs_straddling_cutoff_and_switch` in `crates/ferric-mm/tests/cutoff.rs` |
>
> <!-- verified: id=mm.cutoff.switch; script=scripts/mm_cutoff_sympy.py; claim=4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-mm/src/cutoff.rs::nonbonded_cutoff; rust_test=ferric-mm/tests/cutoff.rs::gradient_matches_fd_with_pairs_straddling_cutoff_and_switch; registry=mm -->

### Finding the pairs

`neighbor_pairs` bins the atoms into cubic cells of edge $r_c$, hashed so that
empty space costs nothing, and compares each atom with the 27 cells around it,
keeping pairs with $|\mathbf r_{ij}|^2<r_c^2$. The ordinary candidates and all
1–4 pairs are merged and visited in lexicographic $(i,j)$ order, the order of
the all-pairs loop. That ordering gives an exact limit. With $r_c$ beyond the
system diameter, $\varepsilon_{\mathrm{rf}}=1$ and no active switching, the LJ
energy and the full gradient are bit-identical to `gradient`, and the Coulomb
energy differs by exactly $-c_{\mathrm{rf}}\sum q_iq_j$, the constant shift
(`crates/ferric-mm/tests/cutoff.rs`).

## Running it {#molecular-mechanics-running-it}

From Python the force field enters through `run_qmmm(mm_topology=)`. The
topology covers every atom of the structure, QM atoms included, because the
exclusions come from the full bond graph. This water dimer puts one water in
the QM region and gives the other the TIP3P parameters with flexible bonds:

```python
import ferric

# Water dimer: atoms 0-2 are the QM water, atoms 3-5 the MM water (TIP3P).
symbols = ["O", "H", "H", "O", "H", "H"]
coords = [
    (0.000, 0.000, 0.000), (0.757, 0.586, 0.000), (-0.757, 0.586, 0.000),
    (0.000, -0.300, 2.900), (0.757, 0.286, 2.900), (-0.757, 0.286, 2.900),
]
charges = [-0.834, 0.417, 0.417] * 2
system = ferric.QmmmSystem(symbols, coords, charges, qm_indices=[0, 1, 2])

# AMBER units: kcal/mol, Angstrom, degrees. TIP3P with flexible bonds.
topology = ferric.MmTopology.from_amber_units(
    charges,
    [3.15061, 0.0, 0.0] * 2,
    [0.1521, 0.0, 0.0] * 2,
    [(0, 1, 553.0, 0.9572), (0, 2, 553.0, 0.9572),
     (3, 4, 553.0, 0.9572), (3, 5, 553.0, 0.9572)],
    [(1, 0, 2, 100.0, 104.52), (4, 3, 5, 100.0, 104.52)],
    [],
)
result = ferric.run_qmmm(system, "cc-pvdz", mm_topology=topology)
print(f"E_total = {result.energy:.10f} Ha")
for name, value in result.mm_energy.items():
    print(f"  {name:8s} {value: .6e}")
```

At build `3f612e0b3` this prints:

```text
E_total = -76.0217877933 Ha
  bond      2.184894e-08
  angle     2.733705e-09
  torsion   0.000000e+00
  lj        9.150682e-04
  coulomb   0.000000e+00
  total     9.150928e-04
```

The components follow the additive rules of section
@sec:qmmm-boundaries-additive-mm-terms. The QM water's bonds and angle are
dropped, so the small bond and angle energies are the MM water's, whose O–H
length differs from $r_0$ by about $10^{-4}$ Å. The MM–MM Coulomb term is
zero because the only MM–MM pairs are 1–2 and 1–3 within one water, both
excluded. The QM–MM Coulomb interaction is inside the SCF energy, not here.
The Lennard-Jones energy is the oxygen–oxygen repulsion across the cut at
2.92 Å, about $9.2\times10^{-4}$ Ha (0.57 kcal/mol).

## The algorithm {#molecular-mechanics-algorithm}

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | exclusions and 1–4 sets from the bond graph, once | bonds | `MmTopology::new` |
| 2 | bond, angle and torsion energies and gradients | terms | `bonded_terms` |
| 3 | nonbonded sum: all non-excluded pairs, or the cutoff form | $N^2$, or $N$ with the cell list | `nonbonded_all_pairs`, `nonbonded_cutoff` |
| 4 | with a cutoff: cell list of edge $r_c$, 27 neighbour cells, hashed | $N$ | `neighbor_pairs` |
| 5 | QM–MM Lennard-Jones across the cut, with the topology's pair scales | $N_{\mathrm{QM}}N_{\mathrm{MM}}$ | `qm_mm_lj_energy_gradient_scaled` |

## Options {#molecular-mechanics-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `MmTopology::with_scales(lj, coul)` | floats | 0.5, $1/1.2$ | the 1–4 factors |
| `NonbondedMethod` | `NoCutoff`, `CutoffNonPeriodic(CutoffParams)` | `NoCutoff` | all pairs, or the reaction-field cutoff; Rust library only |
| `CutoffParams::new(r_c)` | $r_c > 0$ | | the cutoff; invalid values are errors |
| `with_reaction_field_dielectric` | $\varepsilon_{\mathrm{rf}} > 0$ | 78.3 (OpenMM's) | the reaction-field dielectric |
| `with_switch_distance` | $0 \le r_{\mathrm{sw}} < r_c$ | none | LJ switching start |

The QM/MM paths (`qmmm_mm_terms`, Python `run_qmmm(mm_topology=)`) take no MM
options and always use the all-pairs sum.

## Numerical anchors {#molecular-mechanics-numerical-anchors}

| Quantity | System | Result | Source |
|---|---|---|---|
| bond, angle, torsion, Coulomb and LJ energies and gradients against OpenMM (Reference platform, ff14SB, flexible TIP3P, no cutoff) | ALA-ALA, ACE-PHE-NME, TRP-PRO-ASP (charge $-1$), ACE-PHE-NME + 3 TIP3P waters; relaxed and two displaced geometries; one case with every torsion phase shifted by 37° | energies $\le 1.75\times10^{-12}$ relative; gradients $\le 8.5\times10^{-11}$ relative (after OpenMM's Coulomb constant, which differs by $6.6\times10^{-11}$ relative); pair sets identical | `site/src/reference/validation.md`, MM force field row |
| the cutoff form against OpenMM 8.6 `CutoffNonPeriodic` (Reference platform, `scripts/gen_openmm_mm_cutoff_refs.py`, charges rescaled to OpenMM's Coulomb constant) | ethanol + 12 TIP3P-charged waters, 45 atoms, diameter 14.2 Å; $r_c$ 5 Å with $\varepsilon_{\mathrm{rf}}$ 78.3, with a 4 Å switch, and with 1; $r_c$ 2.9 Å with a 2.2 Å switch (two 1–4 pairs beyond $r_c$) | energy $\le 2.4\times10^{-14}$, gradient $\le 2.4\times10^{-15}$ relative; FD with pairs inside, in the switching region and beyond the cutoff $\le 1.8\times10^{-11}$ Ha/Bohr | same, MM nonbonded cutoff row |
| QM–MM LJ across a cut against OpenMM's exclusions and exceptions | ethanol cut at C–C | energy $\le 8.7\times10^{-19}$ Ha; gradient $\le 2.0\times10^{-18}$ Ha/Bohr | same, QM–MM LJ row |

## Sources {#molecular-mechanics-sources}

| Source | What it gives |
|---|---|
| `scripts/mm_dihedral_gradient_sympy.py`, `scripts/mm_cutoff_sympy.py` | the dihedral and angle gradients, the cutoff identities, the bond and pair gradients |
| `crates/ferric-mm/tests/vs_openmm.rs`, `crates/ferric-mm/tests/validation_mm.rs`, `crates/ferric-mm/tests/cutoff.rs`, `crates/ferric-mm/tests/terms_fd.rs`, `crates/ferric-mm/tests/ring_exclusions.rs` | OpenMM comparisons, FD, and the ring rule |

**Known limits (2026-11-15).**

- No periodic boundary conditions, no PME or Ewald summation, and no
  long-range dispersion correction.
- Parameters are never assigned and no `prmtop` reader exists. Go through
  OpenMM or build the arrays.
- The cutoff is a Rust-library option only; the Python and CLI QM/MM paths
  use the all-pairs sum.
- Near-linear valence angles keep the symmetric subgradient convention (finite
  energy, zero gradient at exact linearity); near-linear but not exact angles
  (0.6° from linear) match central differences.
