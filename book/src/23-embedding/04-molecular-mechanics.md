# 23.4 The molecular-mechanics force field

<!-- section: 23.4; status: ready; registry: mm;
     theory: crates/ferric-mm/theory.md#"1. Bonds", "2. Angles", "3. Periodic torsions: proper and improper", "4. The dihedral angle $\phi$ and its gradient (Bekker form)", "5. Nonbonded: Lennard-Jones and Coulomb", "6. Exclusions and 1-4 scaling", "7. QM-MM Lennard-Jones", "8. Opt-in nonbonded cutoff (OpenMM `CutoffNonPeriodic`)" -->

`ferric-mm` is arithmetic only. It takes a topology, supplied entirely by the
caller, and returns an AMBER-form energy and its analytic gradient: harmonic
bonds and angles, periodic torsions, Lennard-Jones and Coulomb. It assigns no
parameters. Turning a protein into a force field is a separate step; the
repository's way of doing it is
`scripts/validation/mm_topology.py` (the function `topology_from_openmm`), which
reads the numbers out of a real OpenMM system.

In QM/MM the force field supplies the terms the wavefunction does not
describe: the MM–MM energy and the Lennard-Jones between the QM and MM atoms
(section 23.2.5).

| Symbol | Meaning | Code name |
|---|---|---|
| $\mathbf R_a$ | position of atom $a$, Bohr | rows of `coords` |
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

Units are Hartree atomic units. `MmTopology::from_amber_units` converts once
from kcal/mol, Å and degrees. Gradients are $\partial E/\partial\mathbf R$,
not forces.

## 23.4.1 Bonds, angles and torsions

<!-- include: crates/ferric-mm/theory.md#"1. Bonds", "2. Angles", "3. Periodic torsions: proper and improper" -->

The bond and angle terms carry no leading $\tfrac12$, the AMBER convention, so
the constants are half of those a $\tfrac12k(r-r_0)^2$ code (OpenMM) uses. The angle gradient is
the closed form of the table; its kernel in `bonded_terms` is

<!-- excerpt: ferric-mm/src/energy.rs::bonded_terms -->
```rust
dti[c] = -inv_sin * (rjk[c] / (rji_n * rjk_n) - cos_t * rji[c] / (rji_n * rji_n));
```

| Symbol | Code |
|---|---|
| $-1/\sin\theta$ | `-inv_sin` |
| $\mathbf b/(\lvert\mathbf a\rvert\lvert\mathbf b\rvert)$ | `rjk[c] / (rji_n * rjk_n)` |
| $\cos\theta$ | `cos_t` |
| $\mathbf a/\lvert\mathbf a\rvert^2$ | `rji[c] / (rji_n * rji_n)` |
| $\partial\theta/\partial\mathbf R_i$ | `dti[c]` |

> **Verified equation 23.6** · the bond-angle gradient
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

At exactly linear $\theta = 0$ or $\pi$ the factor $1/\sin\theta$ is a true
singularity and the direction of the force is undefined. The code keeps the
finite energy and adds no gradient for that angle, the same convention as the
degenerate torsion below.

## 23.4.2 The dihedral angle and its gradient

<!-- include: crates/ferric-mm/theory.md#"4. The dihedral angle $\phi$ and its gradient (Bekker form)" -->

The twelve components are `ferric-mm/src/energy.rs::dihedral_and_gradient`. The
two end atoms are read off the plane normals, and the two middle atoms are
fixed by translation invariance and the two projections $p$ and $q$:

<!-- excerpt: ferric-mm/src/energy.rs::dihedral_and_gradient -->
```rust
grad_j[c] = -grad_i[c] + p * grad_i[c] - q * grad_l[c];
grad_k[c] = -grad_l[c] - p * grad_i[c] + q * grad_l[c];
```

| Symbol | Code |
|---|---|
| $\partial\phi/\partial\mathbf R_i$, $\partial\phi/\partial\mathbf R_l$ | `grad_i`, `grad_l` |
| $p = \mathbf r_{ij}\cdot\mathbf r_{kj}/s^2$ | `p` |
| $q = \mathbf r_{kl}\cdot\mathbf r_{kj}/s^2$ | `q` |
| $\partial\phi/\partial\mathbf R_j$, $\partial\phi/\partial\mathbf R_k$ | `grad_j`, `grad_k` |

> **Verified equation 23.7** · the dihedral gradient
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

When $\mathbf m$ or $\mathbf n$ vanishes (a linear $i$-$j$-$k$ or $j$-$k$-$l$) the plane is
undefined. The code takes $\phi = 0$, a finite energy and a zero gradient for
that torsion; `collinear_valence_angle_torsion_gradient_is_finite_and_zero` in
`crates/ferric-mm/tests/terms_fd.rs` pins it.

## 23.4.3 Nonbonded terms, exclusions and 1–4 scaling

<!-- include: crates/ferric-mm/theory.md#"5. Nonbonded: Lennard-Jones and Coulomb", "6. Exclusions and 1-4 scaling" -->

The pair term is shared by the all-pairs sum and by the 1–4 pairs of the cutoff
path, so both evaluate the same operations (`plain_pair`):

<!-- excerpt: ferric-mm/src/energy.rs::plain_pair -->
```rust
let sr6 = (mixed.sigma / r).powi(6);
let sr12 = sr6 * sr6;
let e_lj = scale_lj * 4.0 * mixed.epsilon * (sr12 - sr6);
```

| Symbol | Code |
|---|---|
| $(\sigma_{ij}/r)^6$ | `sr6` |
| $(\sigma_{ij}/r)^{12}$ | `sr12` |
| $\sigma_{ij}$, $\varepsilon_{ij}$ (Lorentz–Berthelot) | `mixed.sigma`, `mixed.epsilon`, from `mix` |
| $c^{\mathrm{LJ}}_{ij}$ | `scale_lj` |
| $E_{\mathrm{LJ}}$ of the pair | `e_lj` |

The pair sets come from the bond graph once, in `MmTopology::new`, by a
breadth-first search to depth 3 that records each atom's shortest bond path.
Because the path is the shortest one, the two sets never overlap: a pair that is
both 1–3 and 1–4 (the meta carbons of a six-membered ring, any pair in a ring of
three to five) is excluded, never scaled. Editing `bonds` after construction
does not update the sets.

## 23.4.4 The QM–MM Lennard-Jones term

<!-- include: crates/ferric-mm/theory.md#"7. QM-MM Lennard-Jones" -->

The all-pairs and scaled QM–MM forms are `qm_mm_lj_energy_gradient` and
`qm_mm_lj_energy_gradient_scaled`. `qmmm_mm_terms` (section 23.2.5) supplies the
per-pair factor from the topology's own exclusions, so a QM–MM pair that is 1–2
or 1–3 through the bond graph, the cut bond included, contributes nothing.

## 23.4.5 The opt-in nonbonded cutoff

<!-- include: crates/ferric-mm/theory.md#"8. Opt-in nonbonded cutoff (OpenMM `CutoffNonPeriodic`)" -->

The Coulomb part of the cutoff path adds the shifted reaction-field energy and
accumulates the constant $c_{\mathrm{rf}}\sum q_iq_j$ to subtract once at the end
(`nonbonded_cutoff`):

<!-- excerpt: ferric-mm/src/cutoff.rs::nonbonded_cutoff -->
```rust
e.coulomb += qq / r + qq * k_rf * r * r;
qq_shift += qq;
```

| Symbol | Code |
|---|---|
| $q_iq_j$ | `qq` |
| $1/r + k_{\mathrm{rf}}r^2$ | `qq / r + qq * k_rf * r * r`, with the factor $q_iq_j$ outside |
| $\sum q_iq_j$ over ordinary pairs | `qq_shift`, multiplied by $c_{\mathrm{rf}}$ after the loop |

> **Verified equation 23.8** · the reaction-field constants and force
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

The Lennard-Jones switch is a quintic polynomial in Horner form:

<!-- excerpt: ferric-mm/src/cutoff.rs::nonbonded_cutoff -->
```rust
let s = 1.0 + x2 * x * (-10.0 + x * (15.0 - 6.0 * x));
let ds = x2 * (-30.0 + x * (60.0 - 30.0 * x)) / w;
```

| Symbol | Code |
|---|---|
| $x = (r-r_{\mathrm{sw}})/(r_c-r_{\mathrm{sw}})$ | `x`, with `x2 = x * x` |
| $r_c - r_{\mathrm{sw}}$ | `w` |
| $S$ | `s` |
| $dS/dr$ | `ds` |

> **Verified equation 23.9** · the switch and the switched derivative
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

## 23.4.6 The algorithm

| Step | Quantity | Cost | Function |
|---|---|---|---|
| 1 | exclusions and 1–4 sets from the bond graph, once | bonds | `MmTopology::new` |
| 2 | bond, angle and torsion energies and gradients | terms | `bonded_terms` |
| 3 | nonbonded sum: all non-excluded pairs, or the cutoff form | $N^2$, or $N$ with the cell list | `nonbonded_all_pairs`, `nonbonded_cutoff` |
| 4 | with a cutoff: cell list of edge $r_c$, 27 neighbour cells, hashed | $N$ | `neighbor_pairs` |
| 5 | QM–MM Lennard-Jones across the cut, with the topology's pair scales | $N_{\mathrm{QM}}N_{\mathrm{MM}}$ | `qm_mm_lj_energy_gradient_scaled` |

With the cutoff beyond the system diameter, $\varepsilon_{\mathrm{rf}} = 1$ and no
switch, the LJ energy and the whole gradient are bit-identical to the all-pairs
path, because the cell list visits pairs in the same lexicographic order; the
Coulomb energy then differs by exactly $-c_{\mathrm{rf}}\sum q_iq_j$.

## 23.4.7 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `MmTopology::with_scales(lj, coul)` | floats | 0.5, $1/1.2$ | the 1–4 factors |
| `NonbondedMethod` | `NoCutoff`, `CutoffNonPeriodic(CutoffParams)` | `NoCutoff` | all pairs, or the reaction-field cutoff; Rust library only |
| `CutoffParams::new(r_c)` | $r_c > 0$ | | the cutoff; invalid values are errors |
| `with_reaction_field_dielectric` | $\varepsilon_{\mathrm{rf}} > 0$ | 78.3 (OpenMM's) | the reaction-field dielectric |
| `with_switch_distance` | $0 \le r_{\mathrm{sw}} < r_c$ | none | LJ switching start |

The QM/MM paths (`qmmm_mm_terms`, Python `run_qmmm(mm_topology=)`) take no MM
options and always use the all-pairs sum.

## 23.4.8 Numerical anchors

| Quantity | System | Result | Source |
|---|---|---|---|
| bond, angle, torsion, Coulomb and LJ energies and gradients against OpenMM (Reference platform, ff14SB, flexible TIP3P, no cutoff) | ALA-ALA, ACE-PHE-NME, TRP-PRO-ASP (charge $-1$), ACE-PHE-NME + 3 TIP3P waters; relaxed and two displaced geometries; one case with every torsion phase shifted by 37° | energies $\le 1.75\times10^{-12}$ relative; gradients $\le 8.5\times10^{-11}$ relative (after OpenMM's Coulomb constant, which differs by $6.6\times10^{-11}$ relative); pair sets identical | `site/src/reference/validation.md`, MM force field row |
| the cutoff form against OpenMM `CutoffNonPeriodic` | ethanol + 12 TIP3P-charged waters, 45 atoms, diameter 14.2 Å; $r_c$ 5 Å with $\varepsilon_{\mathrm{rf}}$ 78.3, with a 4 Å switch, and with 1; $r_c$ 2.9 Å with a 2.2 Å switch | energy $\le 2.4\times10^{-14}$, gradient $\le 2.4\times10^{-15}$ relative; FD with pairs inside, in the switching region and beyond the cutoff $\le 1.8\times10^{-11}$ Ha/Bohr | same, MM nonbonded cutoff row |
| QM–MM LJ across a cut against OpenMM's exclusions and exceptions | ethanol cut at C–C | energy $\le 8.7\times10^{-19}$ Ha; gradient $\le 2.0\times10^{-18}$ Ha/Bohr | same, QM–MM LJ row |

## 23.4.9 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-mm/theory.md` | all of 23.4.1 to 23.4.5 |
| `scripts/mm_dihedral_gradient_sympy.py`, `scripts/mm_cutoff_sympy.py` | the dihedral and angle gradients, the cutoff identities |
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
