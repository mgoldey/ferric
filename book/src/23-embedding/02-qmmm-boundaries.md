# QM/MM boundaries, link atoms and the full gradient {#qmmm-boundaries}

<!-- status: edit; registry: qmmm -->

A ligand in a protein pocket, the reacting site of an enzyme, or a solute in
explicit solvent is too large to treat quantum mechanically in full. The
chemistry happens in a small region: bonds break there and the density
rearranges there. The rest of the structure matters mainly through its
electrostatics, so QM/MM solves the small region as a molecule and lets the
rest act on it as fixed point charges (the operator of section @sec:external-potentials), with an
optional force field for the classical terms (section @sec:molecular-mechanics).

`ferric-scf/src/qmmm.rs` is a bookkeeping layer around that idea. It adds no
new quantum mechanics. It decides which atoms are quantum, caps the covalent
bonds the split cuts, treats the charge next to the cut, and maps the
gradient of the embedded SCF back onto the atoms of the full structure. In
order: the partition (@sec:qmmm-boundaries-partition), the cut bond (@sec:qmmm-boundaries-cutting-covalent-bond), energies and forces
(@sec:qmmm-boundaries-energies-forces), the full gradient (@sec:qmmm-boundaries-gradient-over-full-structure), the force-field terms (@sec:qmmm-boundaries-additive-mm-terms), and
optimization (@sec:qmmm-boundaries-geometry-optimization), then the algorithm, options, anchors and sources. The
proofs are in `scripts/qmmm_boundary_sympy.py`, claims Q1 to Q4.

| Symbol | Meaning | Code name |
|---|---|---|
| $g$ | link scale, a fraction of the cut bond | `LinkAtomSpec::scale`, `DEFAULT_LINK_SCALE` |
| $\mathbf R_Q$, $\mathbf R_M$, $\mathbf R_L$ | QM frontier atom, MM host of the cut bond, link hydrogen | `qm_atom`, `mm_atom_full_index`, `position` |
| M1, M2 | the MM host of a cut bond and its MM neighbours | `hosts`, `m2s` in `with_boundary_charges` |
| $q_0$ | $q_{\mathrm{M1}}/n$, with $n$ the number of M2 neighbours | `q0` |
| $\mathbf F_i$ | force on embedding charge $i$ | rows of `mm_forces` |
| $\partial E/\partial\mathbf R$ | gradient over the full structure | `full_gradient` |

| Entry point | Location |
|---|---|
| Rust | `ferric-scf/src/qmmm.rs::{QmmmSystem,QmSelection,BoundaryChargeScheme,mm_forces,electric_field_at_points,full_gradient,full_gradient_with_mm,qmmm_mm_terms,optimize_qmmm}` |
| CLI | `[qmmm]`, reading a PQR (`examples/water-qmmm.toml`); fixed point charges and, for the energy task, polarizable sites |
| Python | `QmmmSystem`, `run_qmmm`, `run_optimize_qmmm`; see `site/src/using/qmmm.md` |

## The partition {#qmmm-boundaries-partition}

`QmmmSystem::new` takes the full structure and a rule for the QM region.

| `QmSelection` | QM region | Python |
|---|---|---|
| `Indices` | exactly the listed atoms | `qm_indices=` |
| `WithinRadius` | the seed atoms plus every atom within a radius (Bohr) of any seed, by atom | `qm_seeds=`, `qm_radius_angstrom=` |
| `WithinRadiusWholeResidues` | as above, then every residue with at least one QM atom joins whole | the same, plus `residue_ids=` |

A radius alone selects atom by atom and can slice a residue mid-chain. The
whole-residue variant avoids that.

What happens to the MM charge of an atom selected into the QM region? It is
ignored. The wavefunction already describes that atom's electrons and
nucleus, and keeping its charge would count it twice. A MM atom with exactly
zero charge is recorded as MM but adds nothing to the potential. A partition
with no MM atoms, or only zero charges, gives
`to_external_potential() == None`, which is the literal gas-phase code path.
That makes the empty-environment limit exact, bit for bit, and not merely
small.

## Cutting a covalent bond {#qmmm-boundaries-cutting-covalent-bond}

Cut a C–C bond and the QM carbon is left with an unpaired valence electron.
The QM region would then be a radical that does not exist in the real
molecule. The scaled-position link atom caps the dangling valence with a
hydrogen on the bond vector:

<!-- excerpt: ferric-scf/src/qmmm.rs::with_link_atoms -->
```rust
position: [p.x + scale * dx, p.y + scale * dy, p.z_pos + scale * dz],
```

| Symbol | Code |
|---|---|
| $\mathbf R_L$ | `position` |
| $\mathbf R_Q$ | `p.{x, y, z_pos}`, the QM frontier atom |
| $g$ | `scale` |
| $\mathbf R_M - \mathbf R_Q$ | `(dx, dy, dz)` |

So

$$ \mathbf R_L = \mathbf R_Q + g(\mathbf R_M - \mathbf R_Q) = (1-g)\mathbf R_Q + g\mathbf R_M, \qquad 0 < g < 1 . $$

The default $g = 1.09/1.53$ (`DEFAULT_LINK_SCALE`) puts the cap at a typical
C–H distance along a typical C–C bond. It is the right choice only for a cut
nonpolar single bond. A scale outside $(0, 1)$ is an error, and so is a cut
bond whose two atoms coincide (there is no bond vector). The link hydrogen
is a real QM atom: it has $Z = 1$ and basis functions, it enters the SCF and
the classical charge–nuclear energy, and it follows the real QM atoms in the
QM molecule (`to_qm_molecule`).

$\mathbf R_L$ is not a free coordinate. It moves with its two hosts, and
that fixes how its gradient row is shared out (section @sec:qmmm-boundaries-gradient-over-full-structure).

### The charge next to the cut

The cap creates a second problem. The link hydrogen sits $(1-g)$ of the bond
length from the MM host, $1.53 - 1.09 = 0.44$ Å for the default scale on a
C–C bond. The host's point charge at that distance pulls the QM density
toward it far more than the real, bonded atom would. `with_boundary_charges`
acts on that host (M1) and the atoms bonded to it (M2), following the
nomenclature of Lin and Truhlar, J. Phys. Chem. A 109, 3991 (2005):

| `BoundaryChargeScheme` | $q(\mathrm{M1})$ | $q(\mathrm{M2})$ | added at each M1–M2 midpoint | Conserves |
|---|---|---|---|---|
| `Keep` | $q$ | unchanged | nothing | everything (the untreated boundary) |
| `DeleteHost` (Z1) | $0$ | unchanged | nothing | nothing: the total MM charge changes by $-q_{\mathrm{M1}}$ |
| `RedistributedCharge` (RC) | $0$ | unchanged | $q_0$ | total charge |
| `RedistributedChargeDipole` (RCD) | $0$ | $q_{\mathrm{M2}} - q_0$ | $2q_0$ | total charge and the dipole of {M1, M2} |

Deleting the host is the bluntest fix. RC and RCD move the host charge
instead of discarding it. Do they keep the electrostatics of the group? Put
M1 at $\mathbf R_1$ with charge $q_1$, and its $n$ M2
neighbours at $\mathbf R_{2,m}$ with charges $q_{2,m}$. Set $q_0 = q_1/n$
and the midpoints at $\mathbf M_m = (\mathbf R_1+\mathbf R_{2,m})/2$.

**RC.** The new charges are $q_0$ at each $\mathbf M_m$ and $q_{2,m}$
unchanged. Sum them: $nq_0 + \sum_mq_{2,m} = q_1 + \sum_mq_{2,m}$, the
original total. Now the dipole. Subtract the original dipole and substitute
$\mathbf M_m$:

$$ \Delta\mathbf d_{\mathrm{RC}} = \sum_m q_0\mathbf M_m - q_1\mathbf R_1 = \frac{q_0}{2}\sum_m(\mathbf R_{2,m}-\mathbf R_1) = \frac{q_1}{2}\Big(\frac1n\sum_m\mathbf R_{2,m} - \mathbf R_1\Big) . $$

This is zero only when M1 sits at the centroid of its neighbours, so RC
moves the dipole in general.

**RCD.** Put $2q_0$ at each $\mathbf M_m$ and subtract $q_0$ from each M2.
The total is again $2nq_0 - nq_0 + \sum_mq_{2,m} = q_1 + \sum_mq_{2,m}$.
For the dipole, substitute $\mathbf M_m$ and collect: the $\mathbf R_{2,m}$
terms cancel,

$$ \sum_m 2q_0\mathbf M_m + \sum_m(q_{2,m}-q_0)\mathbf R_{2,m} = nq_0\mathbf R_1 + \sum_mq_{2,m}\mathbf R_{2,m} = q_1\mathbf R_1 + \sum_mq_{2,m}\mathbf R_{2,m} , $$

which is the original dipole of {M1, M2}, exactly, for any $n$ and about any
origin. Check $n = 1$ in your head: M1 and one M2 with charges $q_1$, $q_2$
become $2q_1$ at the midpoint and $q_2 - q_1$ on M2, whose dipole is
$q_1(\mathbf R_1 + \mathbf R_2) + (q_2 - q_1)\mathbf R_2 = q_1\mathbf R_1 + q_2\mathbf R_2$.

> **Verified equation {#eq:qmmm-boundary-rcd}** · what Z1, RC and RCD conserve
>
> For an M1 of charge $q$ with $n = 1, \ldots, 5$ M2 neighbours at symbolic
> positions, the transcribed `with_boundary_charges` gives: RCD conserves the
> total charge and the dipole of {M1, M2} about any origin; RC conserves the
> charge and moves the dipole by $(q/2)(\bar{\mathbf R}_{\mathrm{M2}} - \mathbf R_{\mathrm{M1}})$,
> nonzero in general; Z1 changes the charge by $-q$ (claim Q2).
>
> | | |
> |---|---|
> | Proof | `scripts/qmmm_boundary_sympy.py`, claim Q2 |
> | Negative control | the runner puts $q_0$ instead of $2q_0$ on the RCD midpoints (claim Q2 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/qmmm.rs::with_boundary_charges` |
> | Code test | `redistributed_charge_dipole_conserves_charge_and_dipole` in `crates/ferric-scf/tests/qmmm.rs` |
>
> <!-- verified: id=qmmm.boundary.rcd; script=scripts/qmmm_boundary_sympy.py; claim=Q2; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/qmmm.rs::with_boundary_charges; rust_test=ferric-scf/tests/qmmm.rs::redistributed_charge_dipole_conserves_charge_and_dipole; registry=qmmm -->

The RC/RCD branch of `with_boundary_charges` runs from the split of
$q_{\mathrm{M1}}$ to the choice of the midpoint charge:

<!-- excerpt: ferric-scf/src/qmmm.rs::with_boundary_charges -->
```rust
let q0 = q_m1 / m2s.len() as f64;
self.effective_charges[m1] = 0.0;
let p1 = &self.atoms[m1];
for &m2 in &m2s {
    let p2 = &self.atoms[m2];
    let position = [
        0.5 * (p1.x + p2.x),
        0.5 * (p1.y + p2.y),
        0.5 * (p1.z_pos + p2.z_pos),
    ];
    let q = if scheme == BoundaryChargeScheme::RedistributedChargeDipole {
        // The M2 charge moves toward M1 by q0 to keep the
        // M1–M2 bond dipole: 2q0 at the midpoint, −q0 on M2.
        self.effective_charges[m2] -= q0;
        2.0 * q0
    } else {
        q0
    };
```

| Symbol | Code |
|---|---|
| $q_0 = q_1/n$ | `q0` |
| M1 charge set to 0 | `effective_charges[m1] = 0.0` |
| $\mathbf M_m$ | `position` |
| $q_{2,m} - q_0$ (RCD) | `effective_charges[m2] -= q0` |
| midpoint charge, $2q_0$ (RCD) or $q_0$ (RC) | `q` |

Four points about this code:

- **The M2 shell.** `m2s` holds the MM neighbours of M1 in the bond list
  that are not hosts themselves. RC and RCD refuse an M1 with no MM
  neighbour: the charge would have nowhere to go, and a silent fall back to
  deletion would change the total charge. They also refuse two cut-bond
  hosts bonded to each other, because the two redistributions would
  interfere.
- **Idempotence.** Each call starts again from the raw charges, so calling
  `with_boundary_charges` twice replaces the first scheme. With no cut bond
  every scheme is an exact no-op.
- **Midpoints are point charges.** They are never smeared, since there is no
  smearing model for them. Each lies on the midpoint of two real atoms, which
  is why its force is split between them in section @sec:qmmm-boundaries-gradient-over-full-structure.
- **Only positions move.** The charges are constants; only the midpoint
  positions depend on the geometry.

`min_link_to_charge_distance` reports the shortest distance from a link
hydrogen to a nonzero MM charge. It is a diagnostic and acts on nothing.

## Energies and forces {#qmmm-boundaries-energies-forces}

The embedded SCF is the vacuum solver with `RhfConfig::external_potential`
set to `QmmmSystem::to_external_potential` (section @sec:external-potentials). The QM molecule is
the QM atoms plus the link hydrogens. The MM charges, point or smeared, are
the effective charges after the boundary scheme, in one fixed order: the
atom-centred charges with nonzero effective charge in ascending atom index,
then the RC/RCD midpoints (`mm_charge_positions`, from `active_charges`).
The result is $E_{\mathrm{QM}}$ in the field of the charges. There is no QM–MM
Coulomb term beyond this, and adding the force field's charges for the QM
atoms would count the interaction twice.

The force on an embedding point charge is the charge times the field of the
QM nuclei and electron density at the charge, the Hellmann–Feynman result of
section @sec:external-potentials-gradient. It is a force, $-\partial E/\partial\mathbf R$; the other
gradient routines return $\partial E/\partial\mathbf R$.

<!-- excerpt: ferric-scf/src/qmmm.rs::mm_forces -->
```rust
.map(|(e, &q)| [q * e[0], q * e[1], q * e[2]])
```

| Symbol | Code |
|---|---|
| $\mathbf E^{\mathrm{QM}}(\mathbf R_i)$ | `e`, one row of `electric_field_at_points` |
| $q_i$ | `q` |
| $\mathbf F_i = q_i\mathbf E^{\mathrm{QM}}(\mathbf R_i)$ | the mapped row |

`electric_field_at_points` evaluates $\mathbf E$ with the same
nuclear-attraction derivative blocks that the field at an atom uses, with a
unit positive probe at the point and the nuclear sum over all QM atoms (the
probe is not one of them, so there is no self-term to skip). It needs the
total spin-summed density, and it errors if a point coincides with a QM
nucleus. A Gaussian-smeared charge uses its own formula,
`smeared_site_forces` (section @sec:external-potentials-gradient), and `mm_forces` merges the two kinds
back into the canonical order.

## The gradient over the full structure {#qmmm-boundaries-gradient-over-full-structure}

The SCF gradient lives on the QM molecule, link hydrogens included. What a
geometry optimizer needs is the gradient on the real atoms of the whole
structure. Which atoms does each row belong to? Three chain rules answer
that.

**Link rows.** $\mathbf R_L = (1-g)\mathbf R_Q + g\mathbf R_M$, so
$\partial\mathbf R_L/\partial\mathbf R_Q = (1-g)\mathbf 1$ and
$\partial\mathbf R_L/\partial\mathbf R_M = g\mathbf 1$. Apply the chain rule to
any energy that depends on $\mathbf R_L$:

$$ \frac{\partial E}{\partial\mathbf R_Q} \mathrel{+}= (1-g)\frac{\partial E}{\partial\mathbf R_L}, \qquad \frac{\partial E}{\partial\mathbf R_M} \mathrel{+}= g\,\frac{\partial E}{\partial\mathbf R_L} . $$

The two shares add up to $\partial E/\partial\mathbf R_L$, so moving the whole
structure rigidly still gives a zero net gradient: a rigid translation moves
the cap rigidly.

**Atom-centred charges.** $\partial E/\partial\mathbf R_i \mathrel{+}= -\mathbf F_i$
on that MM atom.

**Midpoint charges.** A midpoint at $\mathbf m = (\mathbf R_{M1}+\mathbf R_{M2})/2$
has $\partial\mathbf m/\partial\mathbf R_{M1} = \partial\mathbf m/\partial\mathbf R_{M2} = \tfrac12\mathbf 1$,
so each host gets $-\tfrac12\mathbf F$.

An MM atom whose charge is zero and that hosts nothing has a zero row. The
result is $dE/d\mathbf R$ of the embedded SCF energy with the partition
rebuilt at each geometry, and its rows sum to zero for an energy that
depends only on relative positions.

> **Verified equation {#eq:qmmm-full-gradient}** · the full gradient across the cut
>
> The link chain rule $(1-g, g)$ follows from $\mathbf R_L = (1-g)\mathbf R_Q + g\mathbf R_M$
> for any energy of $\mathbf R_L$ (claim Q1). The transcribed `full_gradient`
> equals a central difference of a model embedded energy over every
> coordinate of a capped ethane-like structure, with the link atom and the
> RC/RCD midpoints rebuilt at each displacement, for Keep, Z1, RC and RCD
> (claim Q3), and its rows sum to zero (claim Q4).
>
> | | |
> |---|---|
> | Proof | `scripts/qmmm_boundary_sympy.py`, claims Q1, Q3 and Q4 |
> | Negative control | the runner folds the link row onto the frontier atom with $g$ instead of $1-g$, and puts a midpoint force wholly on M1 (claim Q3 fails each time) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/qmmm.rs::full_gradient` |
> | Code test | `full_gradient_matches_finite_difference_across_the_boundary` in `crates/ferric-scf/tests/qmmm.rs` |
>
> <!-- verified: id=qmmm.full.gradient; script=scripts/qmmm_boundary_sympy.py; claim=Q3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/qmmm.rs::full_gradient; rust_test=ferric-scf/tests/qmmm.rs::full_gradient_matches_finite_difference_across_the_boundary; registry=qmmm -->

The model in claim Q3 is a smooth, translation-invariant function standing
in for the SCF, so the check is about the bookkeeping, not the quantum
mechanics. The central differences use an outer step $h = 10^{-4}$ and an
inner step $h = 10^{-5}$; their $O(h^2)$ truncation on derivatives of order
one is about $10^{-8}$, and the bar is $10^{-7}$. The Rust test does the same
with the real SCF on the RCD-treated ethane cut.

The link fold in `ferric-scf/src/qmmm.rs::full_gradient` is a loop over the
link atoms:

<!-- excerpt: ferric-scf/src/qmmm.rs::full_gradient -->
```rust
for (l, link) in system.link_atoms.iter().enumerate() {
    let row = n_qm + l;
    let q_full = system.qm_indices[link.qm_atom];
    let g = link.scale;
    for k in 0..3 {
        let d = qm_gradient[(row, k)];
        full[(q_full, k)] += (1.0 - g) * d;
        full[(link.mm_atom_full_index, k)] += g * d;
    }
}
```

| Symbol | Code |
|---|---|
| $\partial E/\partial\mathbf R_L$ | `d`, one component of the link row of `qm_gradient` |
| $g$ | `g`, from `link.scale` |
| $\partial E/\partial\mathbf R_Q$ | `full[(q_full, k)]` |
| $\partial E/\partial\mathbf R_M$ | `full[(link.mm_atom_full_index, k)]` |

The link rows sit after the real QM rows (`row = n_qm + l`), the order
`to_qm_molecule` builds. The midpoint split follows the atom-centred charges,
again in the order of `active_charges`:

<!-- excerpt: ferric-scf/src/qmmm.rs::full_gradient -->
```rust
for (b, f) in boundary_rows.zip(mm_forces[n_atom_centred..].iter()) {
    let (m1, m2) = b.hosts;
    for k in 0..3 {
        full[(m1, k)] -= 0.5 * f[k];
        full[(m2, k)] -= 0.5 * f[k];
    }
}
```

`full_gradient` carries no MM force-field term and treats the MM charges as
constants. `full_gradient_with_mm` adds the force-field gradient of
`qmmm_mm_terms` (section @sec:molecular-mechanics) row by row. The polarizable terms of section
@sec:polarizable-embedding have their own variant, `full_gradient_with_polarizable`, because the
induced dipoles exert an extra force on every site and charge.

## The additive MM terms {#qmmm-boundaries-additive-mm-terms}

With a force field, `qmmm_mm_terms` adds the MM energy under the additive
convention. The rule is to count each interaction once, either in the
wavefunction or in the force field:

- A bond, angle or torsion that touches at least one MM atom is kept in
  full, including those that straddle the cut. A term whose atoms are all QM
  is dropped, because the wavefunction already describes it.
- Nonbonded terms are summed over MM–MM pairs only, with the topology's own
  exclusions and 1–4 scaling, derived from the full bond graph.
- Lennard-Jones between every QM real atom and every MM atom is added, with
  the same exclusions applied across the cut: 1–2 and 1–3 pairs through the
  topology's bond list (the cut bond included) are skipped, and 1–4 pairs are
  scaled by the topology's `scale_lj_14`. The wavefunction has no dispersion
  or Pauli repulsion toward the point charges, so this term supplies them.
- There is no QM–MM Coulomb term; the embedding already holds it. Link atoms
  and midpoint charges carry no MM terms.

The MM–MM pass zeroes the charges and Lennard-Jones parameters of the QM
atoms but keeps the full bond graph for the exclusions. An MM–MM pair that is
1–2, 1–3 or 1–4 through a path crossing the QM region is still bonded in the
topology, and is excluded or scaled the same way.

## Geometry optimization {#qmmm-boundaries-geometry-optimization}

`optimize_qmmm` runs the BFGS driver
`ferric-scf/src/optimize.rs::optimize_coordinates` over the free atoms. At
every step it calls `QmmmSystem::with_coordinates` to rebuild the link atoms
and boundary charges at the new geometry. Reusing the old link and midpoint
positions would make the analytic gradient wrong at any step that moved a
frontier or boundary atom; with the rebuild, the gradient is the exact
derivative of the energy evaluated. Only the free atoms' gradient rows reach
the optimizer.

| `MoveMm` | MM atoms that move |
|---|---|
| `None` | none; the only value valid without `mm_topology` |
| `WithinRadius(r)` | MM atoms within $r$ Bohr of a QM real atom, measured once at the starting geometry |
| `Residues(ids)` | the listed residues; needs the whole-residue selection |
| `All` | every MM atom |

Any value other than `None` needs an `mm_topology`. Without a force field
nothing holds the MM geometry together, so moving MM atoms would be
meaningless. `QmmmMethod` selects RHF, UHF, or RKS/UKS with a named functional
(those two set RI-J and RI-K with `def2-universal-jkfit`). `optimize_qmmm`
refuses a COSMO SCF (section @sec:conductor-limit-and-cosmo), a relativistic SCF, and a polarizable
embedding (section @sec:polarizable-embedding): the polarizable sites would be copied unchanged
from step to step and go stale, and the step gradient would also need the
polarizable site and charge rows. It accepts IEF-PCM, because that gradient
exists.

## The algorithm {#qmmm-boundaries-algorithm}

| Step | Quantity | Function |
|---|---|---|
| 1 | partition, link atoms, boundary charges | `QmmmSystem::new`, `with_link_atoms`, `with_boundary_charges` |
| 2 | QM molecule, external potential | `to_qm_molecule`, `to_external_potential` |
| 3 | embedded SCF and its gradient on the QM molecule | the SCF drivers with `external_potential` set |
| 4 | field at the charges, forces | `electric_field_at_points`, `mm_forces` |
| 5 | MM terms, if a topology is given | `qmmm_mm_terms` |
| 6 | fold onto the full structure | `full_gradient`, `full_gradient_with_mm` |

## Options {#qmmm-boundaries-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[qmmm] pqr` | path | required | geometry and MM partial charges |
| `[qmmm] qm_indices` or `qm_seeds` with `qm_radius_angstrom` | zero-based indices; seeds and a radius in Å | one is required | the QM region |
| `[qmmm] link_bonds` | `[[qm_atom, mm_atom], ...]` | none | required when the cut is covalent |
| `[qmmm] boundary_scheme` | `keep`, `delete-host`, `rc`, `rcd` | `delete-host` | the host charge treatment; an unknown name is an error |
| Python `with_boundary_charges(bonds, scheme)` | the same names | `keep` if never called | the same |
| Python `with_link_atoms(bonds, scale=None)` | $0 < g < 1$ | $1.09/1.53$ | the cap position |
| Python `run_optimize_qmmm(move_mm=...)` | `"none"`, `("within", r)`, `("residues", ids)`, `"all"` | `"none"` | the free MM atoms |

Why do the two defaults differ? A bare `keep` leaves the host charge inside
the link bond length (0.443 Å measured on an ethane C–C cut), and an
optimization in that field diverges, so the CLI defaults to `delete-host`. In
Python, `with_boundary_charges` is a separate call, and a system that never
makes it has the untreated boundary, `keep`.

## Numerical anchors {#qmmm-boundaries-numerical-anchors}

| Quantity | System | Result | Source |
|---|---|---|---|
| QM/MM energy, embedding shift, QM gradient, MM forces against PySCF `mm_charge` | H2O + 10 charges / cc-pVDZ; CH3OH + 501 TIP3P charges / 6-31G | $\le 4.8\times10^{-12}$ Ha; $\le 1.2\times10^{-10}$; $\le 4.4\times10^{-12}$ Ha/Bohr | `site/src/reference/validation.md`, QM/MM embedding row |
| KS QM/MM (B3LYP, PBE; PySCF with grid response) | CH3OH + 20 charges / def2-SVP | energy $\le 4.1\times10^{-12}$ Ha; QM gradient $\le 2.9\times10^{-9}$; MM forces $\le 1.8\times10^{-10}$ Ha/Bohr | same, KS QM/MM row |
| QM–MM Lennard-Jones across a cut against OpenMM | ethanol cut at C–C, 1-2 to 1-5 pairs | energy $\le 8.7\times10^{-19}$ Ha; gradient $\le 2.0\times10^{-18}$ Ha/Bohr; the every-pair sum misses by 6839 kcal/mol | same, QM–MM LJ row |
| optimization gradient against FD of the minimized energy, partition rebuilt at each displacement | capped ethane (Z1, RC, Keep, RCD), ethyl radical, RKS/UKS PBE, residue selection | $\le 1.8\times10^{-8}$ Ha/Bohr at the minimum, $\le 5.8\times10^{-7}$ relative off it; a gradient of the wrong scheme, or with the MM-row fold broken, misses by $\ge 3.5\times10^{-3}$ relative | same, QM/MM optimization row |

The Lennard-Jones row shows why the exclusions across the cut matter: summing
every QM–MM pair, bonded neighbours included, misses by 6839 kcal/mol.

The optimization row has no external reference. No other code defines this
energy surface, because the partition, link scale and boundary scheme are not
standardized. The check is internal: a central difference of the energy the
optimizer minimizes, against its analytic gradient, so both paths share the
SCF and the integrals. Partition-construction errors are covered by
charge-bookkeeping assertions and scheme-swap controls.

## Sources {#qmmm-boundaries-sources}

| Source | What it gives |
|---|---|
| `site/src/using/qmmm.md` | the user-facing description and the CLI section |
| `scripts/qmmm_boundary_sympy.py`, claims Q1 to Q4 | the proof |
| `crates/ferric-scf/src/qmmm.rs` | the module |
| `crates/ferric-scf/tests/qmmm.rs`, `crates/ferric-scf/tests/qmmm_mm.rs` | partition, boundary, gradient tests |
| `crates/ferric-scf/tests/validation_optimize_qmmm.rs` | the optimization anchors |

**Known limits (2026-11-15).**

- The additive rule of section @sec:qmmm-boundaries-additive-mm-terms (which MM terms are kept across the
  cut) is described from `qmmm_mm_terms` and has no derivation.
- Link atoms are an approximation with known artifacts. Cut only nonpolar
  single bonds (C–C, never a C=O, an amide C–N, or anything conjugated or
  charged), and keep the cut two or three bonds from anything chemically
  interesting. Properties near the boundary are the least trustworthy.
- Double link atoms and Gaussian smearing of the boundary charges are not
  implemented, and no periodic boundary conditions exist.
- `QmSelection::WithinRadius` selects by atom and can cut a residue mid-chain;
  use the whole-residue variant for proteins.
- `QmmmSystem::with_coordinates` re-runs link placement and the boundary scheme
  but not the selection, so it cannot change which atoms are quantum.
- The CLI `[qmmm]` section reads a PQR only. Smeared charges, the MM force
  field and MM relaxation are Python and Rust only.
- Combining a boundary scheme with a polarizable site (section @sec:polarizable-embedding) has no
  finite-difference cross-check yet (`crates/ferric-python/ferric.pyi`,
  `run_qmmm`).
