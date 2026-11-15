# 23.2 QM/MM boundaries, link atoms and the full gradient

<!-- section: 23.2; status: edit; registry: qmmm;
     theory: crates/ferric-scf/theory.md#"QM/MM boundary and the full gradient" -->

A ligand in a protein pocket, a reacting site in an enzyme, or a solute in
explicit solvent is too large to treat quantum mechanically in full. QM/MM
splits the structure in two. The QM region is solved as a molecule. The MM
region becomes fixed point charges that polarize it (the operator of section
23.1), and an optional force field supplies the MM terms (section 23.4).

`ferric-scf/src/qmmm.rs` is a bookkeeping layer. It adds no new quantum
mechanics: it decides which atoms are quantum, caps the covalent bonds the
split cuts, treats the charge next to the cut, and maps the gradient of the
embedded SCF back onto the atoms of the full structure.

The boundary bookkeeping and the chain rules are derived in
`crates/ferric-scf/theory.md` ("QM/MM boundary and the full gradient") and
proved in `scripts/qmmm_boundary_sympy.py`.

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

## 23.2.1 The partition

`QmmmSystem::new` takes the full structure and a rule for the QM region.

| `QmSelection` | QM region | Python |
|---|---|---|
| `Indices` | exactly the listed atoms | `qm_indices=` |
| `WithinRadius` | the seed atoms plus every atom within a radius (Bohr) of any seed, by atom | `qm_seeds=`, `qm_radius_angstrom=` |
| `WithinRadiusWholeResidues` | as above, then every residue with at least one QM atom joins whole | the same, plus `residue_ids=` |

A residue-by-residue selection avoids slicing a residue mid-chain, which a
radius alone would do. The MM charge of an atom selected into the QM region is
ignored: the wavefunction already describes it, and keeping it would count it
twice. A MM atom with exactly zero charge is recorded as MM but adds nothing to
the potential, and a partition with no MM atoms, or only zero charges, gives
`to_external_potential() == None`, the literal gas-phase path.

## 23.2.2 Cutting a covalent bond

A bond with one end in each region leaves the QM atom with a dangling
valence. The scaled-position link atom caps it with a hydrogen on the bond
vector:

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

so $\mathbf R_L = \mathbf R_Q + g(\mathbf R_M - \mathbf R_Q) = (1-g)\mathbf R_Q + g\mathbf R_M$, with
$0 < g < 1$. The default $g = 1.09/1.53$ places the cap at a typical C–H
distance along a typical C–C bond; it is the right choice only for a cut
nonpolar single bond. The link hydrogens follow the real QM atoms in the QM
molecule (`to_qm_molecule`).

The link hydrogen sits about half a bond length from the MM host, whose point
charge then over-polarizes the QM density. `with_boundary_charges` acts on
that host (M1) and the atoms bonded to it (M2). Nomenclature follows Lin and
Truhlar, J. Phys. Chem. A 109, 3991 (2005).

| `BoundaryChargeScheme` | Effect | Conserves |
|---|---|---|
| `Keep` | nothing | everything (the untreated boundary) |
| `DeleteHost` (Z1) | $q_{\mathrm{M1}} \to 0$ | nothing: the total MM charge changes by $-q_{\mathrm{M1}}$ |
| `RedistributedCharge` (RC) | $q_{\mathrm{M1}} \to 0$; $q_0$ on the midpoint of each M1–M2 bond | total charge |
| `RedistributedChargeDipole` (RCD) | $q_{\mathrm{M1}} \to 0$; $2q_0$ on each midpoint; $-q_0$ on each M2 | total charge and the dipole of {M1, M2} |

<!-- include: crates/ferric-scf/theory.md#"QM/MM boundary and the full gradient" -->

> **Verified equation 23.2** · what Z1, RC and RCD conserve
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

For M1 at $\mathbf R_1$ with charge $q_1$ and $n$ M2 atoms at $\mathbf R_{2,m}$
with charges $q_{2,m}$, put $q_0 = q_1/n$ and the midpoints at
$\mathbf M_m = (\mathbf R_1+\mathbf R_{2,m})/2$.

RC: the new charges are $q_0$ at each $\mathbf M_m$ and $q_{2,m}$ unchanged.
The total charge is $nq_0 + \sum_mq_{2,m} = q_1 + \sum_mq_{2,m}$. The dipole
changes by $(q_0/2)\sum_m(\mathbf R_{2,m}-\mathbf R_1)$.

RCD: $2q_0$ at each $\mathbf M_m$ and $q_{2,m} - q_0$ on each M2. The total
charge is again $q_1 + \sum_mq_{2,m}$, and the dipole is
$\sum_m2q_0\mathbf M_m + \sum_m(q_{2,m}-q_0)\mathbf R_{2,m} = nq_0\mathbf R_1 + \sum_mq_{2,m}\mathbf R_{2,m} = q_1\mathbf R_1 + \sum_mq_{2,m}\mathbf R_{2,m}$,
the original dipole of {M1, M2}, exactly and for any $n$.

The RCD branch is two lines in `with_boundary_charges`:

<!-- excerpt: ferric-scf/src/qmmm.rs::with_boundary_charges -->
```rust
self.effective_charges[m2] -= q0;
2.0 * q0
```

| Symbol | Code |
|---|---|
| $q_{2,m} - q_0$ | `effective_charges[m2] -= q0` |
| $2q_0$ | the value returned as the midpoint charge |
| M1 charge | `effective_charges[m1] = 0.0`, set before the loop |

The midpoint charges are point charges, never smeared. Each lies on the
midpoint of two real atoms, which is why the force on it is split between them
in section 23.2.4. RC and RCD refuse an M1 with no MM neighbour in the bond
list (the charge would have nowhere to go, and a silent fall back to deletion
would change the total charge), and refuse two cut-bond hosts bonded to each
other.

`min_link_to_charge_distance` reports the shortest distance from a link
hydrogen to a nonzero MM charge. It is a diagnostic and acts on nothing.

## 23.2.3 Energies and forces

The embedded SCF is the vacuum solver with `RhfConfig::external_potential`
set to `QmmmSystem::to_external_potential` (section 23.1): the QM molecule is
the QM atoms plus the link hydrogens, and the MM charges, point or smeared, are
the effective charges after the boundary scheme, in one fixed order, the
atom-centred charges in ascending atom index and then the RC/RCD midpoints
(`mm_charge_positions`). The result is $E_{\mathrm{QM}}$ in the field of the
charges. There is no QM–MM Coulomb term beyond this: adding the force field's
charges for the QM atoms would count the interaction twice.

The force on an embedding charge is the charge times the field of the QM
nuclei and electron density at the charge. It is a force, $-\partial E/\partial\mathbf R$,
and the other gradient routines return $\partial E/\partial\mathbf R$.

<!-- excerpt: ferric-scf/src/qmmm.rs::mm_forces -->
```rust
.map(|(e, &q)| [q * e[0], q * e[1], q * e[2]])
```

`electric_field_at_points` evaluates $\mathbf E$ with the same nuclear-attraction
derivative blocks that the atomic field uses, with the nuclear sum over all QM
atoms (the probe is not one of them). It needs the total spin-summed density. A
Gaussian-smeared charge is not $q\mathbf E(\mathbf R)$: its force is the
derivative of the three-index integral against its own exponent
(`smeared_site_forces`), and `mm_forces` merges the two in the canonical order.

## 23.2.4 The gradient over the full structure

The SCF gradient lives on the QM molecule, link hydrogens included. The full
gradient is on the real atoms of the whole structure. Three chain rules map one
to the other.

$$ \frac{\partial E}{\partial\mathbf R_Q} \mathrel{+}= (1-g)\frac{\partial E}{\partial\mathbf R_L}, \qquad \frac{\partial E}{\partial\mathbf R_M} \mathrel{+}= g\,\frac{\partial E}{\partial\mathbf R_L}, \qquad \frac{\partial E}{\partial\mathbf R_i} \mathrel{+}= -\mathbf F_i , $$

and for a midpoint charge at $(\mathbf R_{M1}+\mathbf R_{M2})/2$ the force is
split in halves, $\partial E/\partial\mathbf R_{M1}$ and $\partial E/\partial\mathbf R_{M2}$ each
$\mathrel{+}= -\tfrac12\mathbf F$. The first pair is the chain rule through
$\mathbf R_L = (1-g)\mathbf R_Q + g\mathbf R_M$. The sum of each column over
atoms is unchanged by the fold, because a rigid translation moves the cap
rigidly.

> **Verified equation 23.3** · the full gradient across the cut
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

The link fold is two lines of `ferric-scf/src/qmmm.rs::full_gradient`:

<!-- excerpt: ferric-scf/src/qmmm.rs::full_gradient -->
```rust
full[(q_full, k)] += (1.0 - g) * d;
full[(link.mm_atom_full_index, k)] += g * d;
```

| Symbol | Code |
|---|---|
| $\partial E/\partial\mathbf R_L$ | `d`, one component of the link row of `qm_gradient` |
| $g$ | `g`, from `link.scale` |
| $\partial E/\partial\mathbf R_Q$ | `full[(q_full, k)]` |
| $\partial E/\partial\mathbf R_M$ | `full[(link.mm_atom_full_index, k)]` |

`full_gradient` carries no MM force-field term and does not know that the MM
charges are constants. `full_gradient_with_mm` adds the force-field gradient of
`qmmm_mm_terms` (section 23.4) row by row. The polarizable terms of section
23.3 have their own variant, `full_gradient_with_polarizable`, because the
induced dipoles exert an extra force on every site and charge.

## 23.2.5 The additive MM terms

With a force field, `qmmm_mm_terms` adds the MM energy under the additive
convention. A bond, angle or torsion that touches at least one MM atom is kept
in full, including those that straddle the cut; a term whose atoms are all QM is
dropped, because the wavefunction already describes it. Nonbonded terms are
summed over MM–MM pairs only, with the topology's own exclusions and 1–4
scaling, derived from the full bond graph. Lennard-Jones between every QM real
atom and every MM atom is added with the same exclusions applied across the cut:
1–2 and 1–3 pairs through the topology's bond list (the cut bond included) are
skipped, and 1–4 pairs are scaled by the topology's `scale_lj_14`. There is no
QM–MM Coulomb term. Link atoms and midpoint charges carry no MM terms.

## 23.2.6 Geometry optimization

`optimize_qmmm` runs the BFGS driver
`ferric-scf/src/optimize.rs::optimize_coordinates` over the free atoms. At every step it calls `QmmmSystem::with_coordinates` to rebuild the
link atoms and boundary charges at the new geometry, so the gradient is the
exact derivative of the energy evaluated. Only the free atoms' gradient rows
reach the optimizer.

| `MoveMm` | MM atoms that move |
|---|---|
| `None` | none; the only value valid without `mm_topology` |
| `WithinRadius(r)` | MM atoms within $r$ Bohr of a QM real atom, measured once at the starting geometry |
| `Residues(ids)` | the listed residues; needs the whole-residue selection |
| `All` | every MM atom |

Any value other than `None` needs an `mm_topology`: moving MM atoms with no
force field to hold their geometry is meaningless. `QmmmMethod` selects RHF,
UHF, or RKS/UKS with a named functional (those two set RI-J and RI-K with
`def2-universal-jkfit`). `optimize_qmmm` refuses a COSMO SCF (section 22.5),
a relativistic SCF, and a polarizable embedding (section 23.3). It accepts
IEF-PCM, because that gradient exists.

## 23.2.7 The algorithm

| Step | Quantity | Function |
|---|---|---|
| 1 | partition, link atoms, boundary charges | `QmmmSystem::new`, `with_link_atoms`, `with_boundary_charges` |
| 2 | QM molecule, external potential | `to_qm_molecule`, `to_external_potential` |
| 3 | embedded SCF and its gradient on the QM molecule | the SCF drivers with `external_potential` set |
| 4 | field at the charges, forces | `electric_field_at_points`, `mm_forces` |
| 5 | MM terms, if a topology is given | `qmmm_mm_terms` |
| 6 | fold onto the full structure | `full_gradient`, `full_gradient_with_mm` |

## 23.2.8 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[qmmm] pqr` | path | required | geometry and MM partial charges |
| `[qmmm] qm_indices` or `qm_seeds` with `qm_radius_angstrom` | zero-based indices; seeds and a radius in Å | one is required | the QM region |
| `[qmmm] link_bonds` | `[[qm_atom, mm_atom], ...]` | none | required when the cut is covalent |
| `[qmmm] boundary_scheme` | `keep`, `delete-host`, `rc`, `rcd` | `delete-host` | the host charge treatment; an unknown name is an error |
| Python `with_boundary_charges(bonds, scheme)` | the same names | `keep` if never called | the same |
| Python `with_link_atoms(bonds, scale=None)` | $0 < g < 1$ | $1.09/1.53$ | the cap position |
| Python `run_optimize_qmmm(move_mm=...)` | `"none"`, `("within", r)`, `("residues", ids)`, `"all"` | `"none"` | the free MM atoms |

The CLI default is `delete-host` because a bare `keep` puts the host charge
inside the link bond length (0.443 Å measured on an ethane C–C cut), and an
optimization in that field diverges. Python's default is `keep`, the untreated
boundary, because `with_boundary_charges` is a separate call.

## 23.2.9 Numerical anchors

| Quantity | System | Result | Source |
|---|---|---|---|
| QM/MM energy, embedding shift, QM gradient, MM forces against PySCF `mm_charge` | H2O + 10 charges / cc-pVDZ; CH3OH + 501 TIP3P charges / 6-31G | $\le 4.8\times10^{-12}$ Ha; $\le 1.2\times10^{-10}$; $\le 4.4\times10^{-12}$ Ha/Bohr | `site/src/reference/validation.md`, QM/MM embedding row |
| KS QM/MM (B3LYP, PBE; PySCF with grid response) | CH3OH + 20 charges / def2-SVP | energy $\le 4.1\times10^{-12}$ Ha; QM gradient $\le 2.9\times10^{-9}$; MM forces $\le 1.8\times10^{-10}$ Ha/Bohr | same, KS QM/MM row |
| QM–MM Lennard-Jones across a cut against OpenMM | ethanol cut at C–C, 1-2 to 1-5 pairs | energy $\le 8.7\times10^{-19}$ Ha; gradient $\le 2.0\times10^{-18}$ Ha/Bohr; the every-pair sum misses by 6839 kcal/mol | same, QM–MM LJ row |
| optimization gradient against FD of the minimized energy, partition rebuilt at each displacement | capped ethane (Z1, RC, Keep, RCD), ethyl radical, RKS/UKS PBE, residue selection | $\le 1.8\times10^{-8}$ Ha/Bohr at the minimum, $\le 5.8\times10^{-7}$ relative off it; a gradient of the wrong scheme, or with the MM-row fold broken, misses by $\ge 3.5\times10^{-3}$ relative | same, QM/MM optimization row |

The optimization row has no external reference: no other code defines this
energy surface, because the partition, link scale and boundary scheme are not
standardized. The check is internal, a central difference of the energy
the optimizer minimizes against its analytic gradient, so both paths share the
SCF and the integrals; partition-construction errors are covered by
charge-bookkeeping assertions and scheme-swap controls.

## 23.2.10 Sources

| Source | What it gives |
|---|---|
| `site/src/using/qmmm.md` | the user-facing description and the CLI section |
| `crates/ferric-scf/theory.md`, "QM/MM boundary and the full gradient" | link atoms, boundary schemes and the chain rules |
| `scripts/qmmm_boundary_sympy.py`, claims Q1 to Q4 | the proof |
| `crates/ferric-scf/src/qmmm.rs` | the module |
| `crates/ferric-scf/tests/qmmm.rs`, `crates/ferric-scf/tests/qmmm_mm.rs` | partition, boundary, gradient tests |
| `crates/ferric-scf/tests/validation_optimize_qmmm.rs` | the optimization anchors |

**Known limits (2026-11-15).**

- The additive rule of section 23.2.5 (which MM terms are kept across the
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
- Combining a boundary scheme with a polarizable site (section 23.3) has no
  finite-difference cross-check yet (`crates/ferric-python/ferric.pyi`,
  `run_qmmm`).
