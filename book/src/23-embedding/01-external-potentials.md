# Point charges, smeared charges and uniform fields {#external-potentials}

<!-- status: edit; registry: qmmm -->

A molecule in a crystal, a protein or a solvent shell feels the charges
around it. The simplest model of that environment is a set of fixed
classical charges, plus a uniform electric field if the molecule sits in an
applied field. The charges are not atoms: they have no basis functions, no
spin and no wavefunction, and their values can be fractional. They act on
the electrons through one extra one-electron potential and on the nuclei
through one extra classical energy.

Everything else in this chapter is built on this operator. The QM/MM charges
of section @sec:qmmm-boundaries are an `ExternalPotential`, and the polarizable sites of
section @sec:polarizable-embedding add a density-dependent term on top of it. The section derives
the operator and the classical energy (@sec:external-potentials-operator-classical-energy), shows where they enter the
SCF (@sec:external-potentials-terms-enter-scf), derives the gradient (@sec:external-potentials-gradient), runs the limits on water
(@sec:external-potentials-running-it), and lists the options and
anchors (@sec:external-potentials-options to @sec:external-potentials-sources). The proofs are in
`scripts/external_potential_sympy.py`, claims E1 to E5.

| Symbol | Meaning | Code name |
|---|---|---|
| $q_i$, $\mathbf R_i$ | point charge and position, Bohr | `PointCharge::{q,x,y,z}` |
| $q_j$, $\mathbf R_j$, $w_j$ | smeared charge, centre, Gaussian width (Bohr) | `SmearedCharge::{q,x,y,z,width}` |
| $\zeta_j$ | $1/w_j^2$ | `1.0 / (s.width * s.width)` |
| $\mathbf E$ | uniform field, atomic units, lab frame | `ExternalPotential::field` |
| $Z_A$, $\mathbf R_A$ | effective nuclear charge (valence charge for an ECP atom) and position | `effective_z`, `atom.{x,y,zpos}` |
| $\mu, \nu$ | AO basis functions | `0..nbas` |
| $P$ | total (spin-summed) AO density | `density`, `d_total` |

| Entry point | Location |
|---|---|
| Rust | `ferric-core/src/external_potential.rs::ExternalPotential`, `ferric-integrals/src/oneelectron.rs::{hcore_with_external,hcore_ecp_with_external,nuclear_with_external,smeared_attraction,field_hcore_term}`, `ferric-scf/src/rhf.rs::RhfConfig` field `external_potential` |
| CLI | `[external_potential]` with `[[external_potential.point_charges]]` (`q`, `x`, `y`, `z`, optional `width`) and `field` (`examples/water-rhf-smeared-charge.toml`) |
| Python | `point_charges=` and `external_field=` on `run_rhf`, `run_uhf`, `run_rohf`, `run_dft`, `run_optimize`, `run_frequencies`, `run_saddle`, `run_irc` and `run_pdep_rpa`; `smeared_charges=` on `run_rhf` only |

## The operator and the classical energy {#external-potentials-operator-classical-energy}

Start with one fixed point charge $q_i$ at $\mathbf R_i$. Its electrostatic
potential at $\mathbf r$ is $q_i/|\mathbf r-\mathbf R_i|$. An electron has
charge $-1$, so its energy in that potential is $-q_i/|\mathbf r-\mathbf R_i|$:
a positive charge attracts electrons. Summed over electrons this is a
one-electron operator, and its matrix in the AO basis is a nuclear-attraction
integral with $q_i$ in place of a nuclear charge.

A point charge placed close to the electron density pulls on it too hard,
because its potential diverges at the charge. A Gaussian-smeared charge
removes the divergence. Spread $q_j$ as the normalized density
$q_j(\zeta_j/\pi)^{3/2}e^{-\zeta_j|\mathbf r-\mathbf R_j|^2}$ with
$\zeta_j = 1/w_j^2$. Solving Poisson's equation for that density gives the
potential $q_j\operatorname{erf}(|\mathbf r-\mathbf R_j|/w_j)/|\mathbf r-\mathbf R_j|$
(claim E1). Two limits fix what this function does. At the centre it is
finite, $2q_j/(\sqrt\pi\,w_j)$. As $w_j \to 0$, $\operatorname{erf}\to 1$ and
it becomes the point-charge potential. A width of exactly zero is therefore
not a valid smeared charge: the point charge is its own representation.

A uniform field $\mathbf E$ has potential $\phi(\mathbf r) = -\mathbf E\cdot\mathbf r$,
since $\mathbf E = -\nabla\phi$. A charge $q$ at $\mathbf R$ has energy
$q\phi(\mathbf R) = -q\,\mathbf E\cdot\mathbf R$. For an electron, $q = -1$,
so each electron gains $+\mathbf E\cdot\mathbf r$, and each nucleus gains
$-Z_A\,\mathbf E\cdot\mathbf R_A$. The code folds the electron's charge into the
operator, so the matrix is $+\mathbf E$ contracted with the dipole integrals
$\langle\mu|\mathbf r|\nu\rangle$ about the origin.

Collect the three electronic pieces into one matrix:

$$ H^{\mathrm{ext}}_{\mu\nu} = -\sum_i q_i\Big\langle\mu\Big|\tfrac{1}{|\mathbf r-\mathbf R_i|}\Big|\nu\Big\rangle - \sum_j q_j\Big\langle\mu\Big|\tfrac{\operatorname{erf}(|\mathbf r-\mathbf R_j|/w_j)}{|\mathbf r-\mathbf R_j|}\Big|\nu\Big\rangle + \mathbf E\cdot\langle\mu|\mathbf r|\nu\rangle . $$

The nuclei see the same charges classically. The nuclei stay points and only
the environment charge is smeared, so the nucleus–smeared-charge energy is
$Z_A$ times the potential of the cloud at the point nucleus. Sum over the
real atoms (ghost atoms are skipped):

$$ E^{\mathrm{ext}}_{\mathrm{nuc}} = \sum_A Z_A\Big[\sum_i \frac{q_i}{|\mathbf R_A-\mathbf R_i|} + \sum_j q_j\frac{\operatorname{erf}(|\mathbf R_A-\mathbf R_j|/w_j)}{|\mathbf R_A-\mathbf R_j|}\Big] - \mathbf E\cdot\sum_A Z_A\mathbf R_A . $$

$Z_A$ is `Atom::effective_z`, the valence charge on an ECP atom, the same
charge the nuclear-attraction integrals use.

The SCF energy is the vacuum functional with $h \to h + H^{\mathrm{ext}}$ and
$V_{nn} \to V_{nn} + E^{\mathrm{ext}}_{\mathrm{nuc}}$. Neither term depends on
the density. Differentiating the energy with respect to $P$ gives the Fock
matrix $F = h + H^{\mathrm{ext}} + G(P)$, so the environment reaches $F$
through $h$ alone and needs no update inside the iterations. The interaction
of the external charges with each other is left out. It is a constant of the
electronic problem; section @sec:molecular-mechanics supplies it from a force field.

For the gradient and Hessian we need the radial derivatives of the smeared
potential. Write $f(R) = \operatorname{erf}(R/w)/R$ and $g = \operatorname{erf}(R/w)$.
Differentiate $g$, then apply the quotient rule twice:

$$ g' = \frac{2e^{-R^2/w^2}}{\sqrt\pi\,w},\qquad f' = \frac{g'}{R} - \frac{g}{R^2},\qquad f'' = -g'\Big(\frac{2}{w^2} + \frac{2}{R^2}\Big) + \frac{2g}{R^3} $$

(`smeared_radial_derivs`, claim E2). With $\mathbf d = \mathbf R_A - \mathbf R_j$
and $R = |\mathbf d|$, the chain rule $\partial R/\partial\mathbf R_A = \mathbf d/R$
gives the classical gradient $Z_Aq_jf'(R)\,\mathbf d/R$ on the atom. The
site enters through $-\mathbf d$, so the gradient on the site is the negative
(`smeared_charge_nuclear_site_gradient`). Differentiate once more for the
Hessian block on atom $A$'s diagonal:

$$ Z_Aq_j\Big[f''\frac{\mathbf d\mathbf d^{\mathsf T}}{R^2} + \frac{f'}{R}\Big(\mathbf 1 - \frac{\mathbf d\mathbf d^{\mathsf T}}{R^2}\Big)\Big] . $$

Check the point limit. As $w\to0$, $f\to 1/R$, $f' \to -1/R^2$ and
$f'' \to 2/R^3$, and the block becomes the point-charge block
$Z_Aq_i(3\mathbf d\mathbf d^{\mathsf T}/R^5 - \mathbf 1/R^3)$. The charges do not move
with the atoms, so the classical Hessian is block diagonal. The field term is
linear in $\mathbf R_A$ and contributes nothing to it.

> **Verified equation {#eq:ext-smeared-classical}** · the smeared charge and the classical terms
>
> $\operatorname{erf}(r/w)/r$ solves Poisson's equation for the unit Gaussian
> $(\zeta/\pi)^{3/2}e^{-\zeta r^2}$, $\zeta = 1/w^2$, and tends to $1/r$ as
> $w \to 0$ (claim E1). The radial derivatives of `smeared_radial_derivs`
> are those of $\operatorname{erf}(r/w)/r$ (claim E2), and the classical
> charge–nucleus gradient, site gradient and Hessian blocks as coded are the
> derivatives of the second line, with the point blocks as the $w\to0$
> limit (claim E3).
>
> | | |
> |---|---|
> | Proof | `scripts/external_potential_sympy.py`, claims E1 to E3 |
> | Negative control | the runner changes $2/w^2$ to $1/w^2$ in $f''$ (claims E2 and E3 fail) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-core/src/external_potential.rs::{charge_nuclear_energy,charge_nuclear_hessian}` |
> | Code test | `smeared_charge_nuclear_hessian_matches_gradient_difference` in `crates/ferric-core/src/external_potential.rs` |
>
> <!-- verified: id=ext.smeared.classical; script=scripts/external_potential_sympy.py; claim=E3; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-core/src/external_potential.rs::charge_nuclear_hessian; rust_test=ferric-core/src/external_potential.rs::smeared_charge_nuclear_hessian_matches_gradient_difference; registry=qmmm -->

### Where is the origin?

The field term uses dipole integrals about one point, and the code puts it
at the coordinate origin, $\mathbf O = 0$. Does the energy depend on that
choice? Move the origin to $\mathbf O$ and expand both terms. The electronic
term becomes
$\mathbf E\cdot\mathrm{Tr}[P\langle\mathbf r\rangle] - (\mathbf E\cdot\mathbf O)\,\mathrm{Tr}[PS]$,
and $\mathrm{Tr}[PS] = N_e$. The nuclear term $-\mathbf E\cdot\sum_AZ_A(\mathbf R_A-\mathbf O)$
gains $+(\mathbf E\cdot\mathbf O)\sum_AZ_A$. Add the two changes:

$$ E(\mathbf O) - E(0) = (\mathbf E\cdot\mathbf O)\Big(\sum_AZ_A - N_e\Big) = (\mathbf E\cdot\mathbf O)\,Q . $$

A neutral molecule's energy does not depend on the origin. A charged one
shifts by a constant. The shift adds $-(\mathbf E\cdot\mathbf O)S$ to $h$,
which moves every orbital energy by the same amount, so neither the orbitals
nor the gradient change (claim E4). The code's $\mathbf O = 0$ fixes the
constant.

> **Verified equation {#eq:ext-field-origin}** · the field term does not depend on the origin
>
> With $H' = +\mathbf E\cdot(\mathbf r-\mathbf O)$ and
> $E_{\rm nuc} = -\mathbf E\cdot\sum_AZ_A(\mathbf R_A-\mathbf O)$, moving the
> origin changes the total energy by $(\mathbf E\cdot\mathbf O)Q$, a constant
> that vanishes for a neutral molecule (claim E4).
>
> | | |
> |---|---|
> | Proof | `scripts/external_potential_sympy.py`, claim E4 |
> | Negative control | the un-negated electronic term does not cancel; the runner flips the sign of the electronic field term (claim E4 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-integrals/src/oneelectron.rs::field_hcore_term` |
>
> <!-- verified: id=ext.field.origin; script=scripts/external_potential_sympy.py; claim=E4; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-integrals/src/oneelectron.rs::field_hcore_term; registry=qmmm -->

## Where the terms enter the SCF {#external-potentials-terms-enter-scf}

The charges and the field go into the core Hamiltonian once, before the
iteration loop, and the classical energy goes into $V_{nn}$ once. Neither is
recomputed. The density-dependent terms of sections @sec:reaction-field and @sec:polarizable-embedding differ:
they are rebuilt every iteration.

| Term | Where | Function |
|---|---|---|
| point-charge attraction | appended to the real atoms as extra nuclei of charge $q_i$ in the nuclear-attraction engine | `nuclear_with_external` |
| smeared-charge attraction | three-index Coulomb integral against one normalized Gaussian per site | `smeared_attraction` |
| field | dipole integrals about the origin, contracted with $\mathbf E$ | `field_hcore_term` |
| the sum | $T + V + $ the three terms above | `hcore_with_external` |
| the ECP projector, when the basis carries one | added on top | `hcore_ecp_with_external` |
| $E^{\mathrm{ext}}_{\mathrm{nuc}}$ | added to $V_{nn}$ before the loop | `ferric-scf/src/driver.rs::prepare` |

`hcore_with_external` in full:

<!-- excerpt: ferric-integrals/src/oneelectron.rs::hcore_with_external -->
```rust
let t = kinetic(prep);
let Some(ext) = ext else {
    return Ok(t + nuclear(prep));
};
if ext.is_empty() {
    return Ok(t + nuclear(prep));
}
let v = if ext.point_charges.is_empty() {
    nuclear(prep)
} else {
    nuclear_with_external(prep, ext)
};
let mut h = t + v;
if !ext.smeared_charges.is_empty() {
    h += &smeared_attraction(prep, &ext.smeared_charges)?;
}
if let Some(field) = ext.field {
    h += &field_hcore_term(prep, field)?;
}
Ok(h)
```

| Symbol | Code |
|---|---|
| $T$ | `t` |
| $V_{\mathrm{nuc}}$, the vacuum attraction | `nuclear(prep)` |
| $V_{\mathrm{nuc}} + V_{\text{point charges}}$ | `nuclear_with_external(prep, ext)` |
| $\sum_j(\ldots)$, smeared | `smeared_attraction(...)` |
| $\mathbf E\cdot\langle\mu\vert\mathbf r\vert\nu\rangle$ | `field_hcore_term(...)` |

Three points about this function:

- **Two early returns.** `ext = None` and an empty `ExternalPotential` (no
  point charges, no smeared charges, no field) both return `t + nuclear(prep)`,
  the vacuum expression. The SCF energy is then bit-identical to a run with
  `external_potential = None`
  (`external_potential_none_matches_default_exactly` in `ferric-scf/src/rhf.rs`).
- **Point charges cost nothing extra.** They are appended to the list of
  nuclear centres, so one call of the nuclear-attraction engine builds
  $V_{\mathrm{nuc}}$ and the charge attraction together.
- **The field term is a contraction.** `field_hcore_term` returns the zero
  matrix for a zero field, and otherwise sums the three dipole matrices
  about the origin weighted by the field components:

<!-- excerpt: ferric-integrals/src/oneelectron.rs::field_hcore_term -->
```rust
let dip = dipole(prep, [0.0, 0.0, 0.0])?;
Ok(field[0] * &dip[0] + field[1] * &dip[1] + field[2] * &dip[2])
```

The smeared integral is the Coulomb integral of the basis pair against a
normalized s Gaussian of exponent $\zeta_j$ at the site, divided by that
Gaussian's integral so that the charge is $q_j$:

<!-- excerpt: ferric-integrals/src/oneelectron.rs::smeared_attraction -->
```rust
v[(o1 + a, o2 + b)] += -sc.q * block[a * n2 + b] / norm;
```

| Symbol | Code |
|---|---|
| $-q_j$ | `-sc.q` |
| $(\mu\nu\vert g_j)$ | `block[a * n2 + b]` |
| $\int g_j\,d\mathbf r$ | `norm`, from `SiteBasis::norm_int` |

Dividing by $\int g_j$ turns a three-index integral against a basis
function into the potential of a cloud of total charge one.
`tight_site_gaussian_reproduces_point_charge_attraction` in
`crates/ferric-integrals/src/site_basis.rs` pins the $w\to0$ limit of this
matrix against the point-charge attraction.

The classical energy goes into $V_{nn}$ in `prepare`:

<!-- excerpt: ferric-scf/src/driver.rs::prepare -->
```rust
let vnn = mol.nuclear_repulsion()
    + config.external_potential.as_ref().map_or(0.0, |ext| {
        ext.charge_nuclear_energy(mol) + ext.field_nuclear_energy(mol)
    });
```

and the smeared part of `charge_nuclear_energy` is the erf term of
$E^{\mathrm{ext}}_{\mathrm{nuc}}$, with `sqrt_zeta` $= 1/w_j$:

<!-- excerpt: ferric-core/src/external_potential.rs::charge_nuclear_energy -->
```rust
e += za * sc.q * erf(sqrt_zeta * r) / r;
```

## The gradient {#external-potentials-gradient}

The charges are fixed in space. The gradient is taken over the QM atoms
only, and `ferric-scf/src/gradient.rs::oneelectron_gradient` adds five
pieces to the plain gradient. The Pulay and two-electron terms are those of
the plain gradient, evaluated at the embedded density.

| Piece | Source | Method |
|---|---|---|
| electron–point-charge, Hellmann–Feynman | the nuclear-attraction derivative engine with the charges appended (`set_point_charges_extra`, `compute_1e_deriv_block_n`, $3(2 + N_{\mathrm{at}} + N_c)$ blocks); the two shell-centre blocks and the $N_{\mathrm{at}}$ real-atom blocks are contracted with $P$, and the $N_c$ charge-centre blocks are computed and discarded, because the charges are not variables | analytic, libint |
| smeared-charge attraction | $\sum_{\mu\nu}P_{\mu\nu}\,\partial H^{\mathrm{ext}}_{\mu\nu}/\partial\mathbf R_A$ from the three-index derivative engine, blocks 1 and 2 (the two orbital centres) | analytic, `smeared_charge_qm_gradient` |
| classical charge–nucleus | $\partial E^{\mathrm{ext}}_{\mathrm{nuc}}/\partial\mathbf R_A$: $-Z_Aq_i\mathbf d/R^3$ for a point charge and the $f'$ form above for a smeared one | analytic, `ExternalPotential::charge_nuclear_gradient` |
| field–nucleus | $-Z_A\mathbf E$ per atom | analytic, `ExternalPotential::field_nuclear_gradient` |
| field–electron, $\sum_{\mu\nu}P_{\mu\nu}\,\partial\langle\mu|\mathbf E\cdot\mathbf r|\nu\rangle/\partial\mathbf R_A$ at fixed $P$ | central difference of `field_hcore_term` rebuilt at displaced basis centres, $h = 10^{-4}$ Bohr | finite difference, `field_density_gradient_fd` |

Why is the last term a finite difference? The integral shim has no
derivative engine for the dipole operator (`emultipole1`). The truncation
error is $O(h^2)$, about $10^{-10}$ on the model of the proof (claim E5).
Two exact properties survive the finite difference to that error. Translate
every basis centre by $\mathbf t$: $\langle\mu|\mathbf r|\nu\rangle$ changes
by $\mathbf t S_{\mu\nu}$, so the atom sum of the field–electron term is
$\mathbf E\,\mathrm{Tr}[PS] = N_e\mathbf E$. Add the field–nucleus rows,
$-\sum_AZ_A\mathbf E$. The atom sum of the two is $-Q\mathbf E$, which is a
net force of $Q\mathbf E$ on the molecule. A neutral molecule feels no net
force from a uniform field, as it should (claim E5).

> **Verified equation {#eq:ext-field-fd-gradient}** · the finite-difference field–electron term and the net force
>
> On an s-Gaussian model the central difference ($h = 10^{-4}$) of
> $\mathrm{Tr}[P\,\mathbf E\cdot\langle\mathbf r\rangle]$ equals the analytic
> derivative of the dipole integrals to the Richardson-estimated $O(h^2)$
> error, its atom sum is $\mathbf E\,\mathrm{Tr}[PS]$, and with the
> field–nucleus rows the net force is $Q\mathbf E$ (claim E5).
>
> | | |
> |---|---|
> | Proof | `scripts/external_potential_sympy.py`, claim E5 |
> | Negative control | the runner divides the central difference by $h$ in place of $2h$ (claim E5 fails) |
> | Runner | `crates/ferric-python/tests/test_derivation_scripts.py` |
> | Code | `ferric-scf/src/gradient.rs::field_density_gradient_fd` |
> | Code test | `oneelectron_gradient_uniform_field_matches_finite_difference` in `crates/ferric-scf/src/gradient.rs` |
>
> <!-- verified: id=ext.field.fd-gradient; script=scripts/external_potential_sympy.py; claim=E5; runner=crates/ferric-python/tests/test_derivation_scripts.py; code=ferric-scf/src/gradient.rs::field_density_gradient_fd; rust_test=ferric-scf/src/gradient.rs::oneelectron_gradient_uniform_field_matches_finite_difference; registry=qmmm -->

### The force on a charge

The force on a charge itself is not part of this gradient, but QM/MM needs
it (section @sec:qmmm-boundaries). The basis functions do not
move with a point charge, so there is no Pulay term, and the SCF density is
stationary, so only the explicit dependence of the energy on $\mathbf R_i$
counts. That dependence is $q_i$ times the potential of the QM region at
the charge,
$q_i\big[\sum_AZ_A/|\mathbf R_A-\mathbf R_i| - \int\rho(\mathbf r)/|\mathbf r-\mathbf R_i|\,d\mathbf r\big]$.
Take minus its gradient: the force is $q_i\,\mathbf E^{\mathrm{QM}}(\mathbf R_i)$,
the field of the QM nuclei and electron density at the charge. Its
electronic part is the contraction of $P$ with the charge-centre blocks that
`oneelectron_gradient` discards; QM/MM takes it from
`ferric-scf/src/qmmm.rs::electric_field_at_points` instead.

A smeared charge is different, because its cloud has a width over which
the QM field varies. The site derivative of the electronic term comes from
the three-index derivative blocks: the integral $(\mu\nu|g_j)$ is unchanged
when all three centres move together, so the site block is
$-(\text{block 1} + \text{block 2})$, contracted with $P$. Add the classical
site term of @sec:external-potentials-operator-classical-energy, and the force is minus the sum. That is
`ferric-scf/src/gradient.rs::smeared_site_forces`, and it is not
$q_j\mathbf E(\mathbf R_j)$. `ferric-scf/src/qmmm.rs::mm_forces` (section
@sec:qmmm-boundaries) merges the two kinds of charge.

The analytic Hessian with external charges (QM–QM block, charges fixed) is
derived with the RHF Hessian (section @sec:hf-hessian-options-change-equation).
The smeared-charge second derivatives there are a fourth-order central
stencil of the first-derivative three-centre blocks
(`ferric-scf/src/hessian_smeared.rs`); the anchors table gives its agreement
with PySCF.


## Running it {#external-potentials-running-it}

Here are the limits of this page on water in cc-pVDZ, from the `589b50832`
build: the empty charge list against vacuum, a smeared charge as its width
shrinks to the point charge, the origin of the field term, and the net force
of a uniform field on a neutral molecule.

```python
import numpy as np
import ferric

bs = ferric.BasisSet.bundled("cc-pvdz")
mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")

# 1. Exactness: no environment vs an empty charge list (both take the vacuum path).
e_vac = ferric.run_rhf(mol, bs).energy
e_empty = ferric.run_rhf(mol, bs, point_charges=[]).energy
print(f"E(vacuum)               = {e_vac:.12f}")
print(f"E(empty charge list)    = {e_empty:.12f}   diff = {e_empty - e_vac:.1e}")

# 2. One charge 5 Bohr off the oxygen side (Bohr), point vs smeared with shrinking width (the w -> 0 limit).
q = (-0.8, 0.0, 0.0, -5.0)
e_pt = ferric.run_rhf(mol, bs, point_charges=[q]).energy
print(f"E(point charge)         = {e_pt:.12f}   shift = {e_pt - e_vac:.6f} Ha")
for w in (2.0, 1.0, 0.5, 0.1, 0.01):
    e_sm = ferric.run_rhf(mol, bs, smeared_charges=[q + (w,)]).energy
    print(f"  smeared w = {w:5.2f} Bohr: E - E(point) = {e_sm - e_pt:+.3e}")

# 3. Field: a neutral molecule's energy does not depend on the origin (claim E4).
#    Translating the molecule by t is the same as moving the origin by -t.
field = (0.0, 0.0, 0.01)
r0 = ferric.run_rhf(mol, bs, external_field=field, with_gradient=True)
xyz = open("testdata/molecules/water.xyz").read().splitlines()
shifted = xyz[:2] + [
    " ".join([ln.split()[0]] + [f"{float(c) + 3.0:.10f}" for c in ln.split()[1:4]]) for ln in xyz[2:] if ln.strip()
]
mol_t = ferric.Molecule.from_xyz_string("\n".join(shifted) + "\n")
e_t = ferric.run_rhf(mol_t, bs, external_field=field).energy
print(f"E(field) = {r0.energy:.12f}; translated by 3 A: diff = {e_t - r0.energy:.1e}")

# 4. Net force from a uniform field is Q E: zero for neutral water; the
#    field-electron term is a finite difference, so the sum is zero to O(h^2).
g = np.asarray(r0.gradient())
print("sum of gradient rows (neutral, expect ~0):", g.sum(axis=0))
```

```text
E(vacuum)               = -76.026767997377
E(empty charge list)    = -76.026767997377   diff = 0.0e+00
E(point charge)         = -76.051032944123   shift = -0.024265 Ha
  smeared w =  2.00 Bohr: E - E(point) = -1.883e-03
  smeared w =  1.00 Bohr: E - E(point) = -2.485e-05
  smeared w =  0.50 Bohr: E - E(point) = -1.629e-06
  smeared w =  0.10 Bohr: E - E(point) = -4.475e-08
  smeared w =  0.01 Bohr: E - E(point) = -4.410e-10
E(field) = -76.018909023183; translated by 3 A: diff = -1.1e-13
sum of gradient rows (neutral, expect ~0): [-2.09048140e-29 -4.69277395e-14 -3.86668475e-11]
```

An empty charge list gives the vacuum energy bit for bit. The smeared charge
approaches the point charge as $w$ shrinks, a factor of about $10^2$ per
decade of $w$ at the small widths. That is a $w^2$ law. For a site outside
the nuclei the first-order shift is $\Delta E = q_j\,\pi w_j^2\,\rho(\mathbf R_j)$,
because $\int[1/r - \operatorname{erf}(r/w)/r]\,d\mathbf r = \pi w^2$: the
smeared potential differs from the point one only within a few $w$ of the
site, where the density is nearly constant. The shift is negative for
$q = -0.8$, and the last row gives $\rho \approx 1.8\times10^{-6}$ at the
site. Moving the molecule by 3 Å in the field
changes the energy by $10^{-13}$ Ha, and the gradient rows sum to
$4\times10^{-11}$ Ha/Bohr, inside the $O(h^2)$ error of the finite-difference
field term.

## Options {#external-potentials-options}

| Option | Values | Default | Effect |
|---|---|---|---|
| `[[external_potential.point_charges]]` | `q`, `x`, `y`, `z` in Bohr; `width` in Bohr optional | none | a charge; a `width` makes it smeared, and `width` $\le 0$ is an error |
| `[external_potential] field` | $[E_x, E_y, E_z]$, atomic units | none | the uniform field |
| `[qmmm]` with `[external_potential]` | | | refused: the MM region is an external potential, so both would count it |
| Python `point_charges=`, `smeared_charges=`, `external_field=` | `(q,x,y,z)`, `(q,x,y,z,width)`, `(Ex,Ey,Ez)` | none | the same, in Bohr and atomic units; `smeared_charges=` on `run_rhf` only |

An omitted `[external_potential]` section and an empty one both take the
vacuum path. The MP2 and coupled-cluster Python drivers take no
external-potential arguments; the RI-MP2 gradient in Rust threads one through
(third row of the anchors).

## Numerical anchors {#external-potentials-numerical-anchors}

| Quantity | System | Result | Source |
|---|---|---|---|
| energy and embedding shift against PySCF `qmmm.mm_charge`; QM gradient; forces on the charges | H2O + 10 charges / cc-pVDZ, aug-cc-pVDZ; CH3OH + 501 TIP3P charges / 6-31G | energy and shift $\le 4.8\times10^{-12}$ Ha; gradient $\le 1.2\times10^{-10}$; forces $\le 4.4\times10^{-12}$ Ha/Bohr | `site/src/reference/validation.md`, QM/MM embedding row |
| Gaussian-smeared charges against `mm_charge(radii=)` | H2O + 10 charges / cc-pVDZ, widths 0.5, 1, 2 Bohr | energy $\le 4.1\times10^{-12}$ Ha; gradient $\le 1.2\times10^{-10}$ Ha/Bohr; width $\to0$ reproduces point charges to $3.0\times10^{-12}$ Ha | same, smeared-charge row |
| the RI-MP2 analytic gradient with an external potential | H2O + 10 charges / cc-pVDZ; CH3OH + 20 charges / 6-31G | energies $\le 8.4\times10^{-12}$ Ha; gradient $\le 4.8\times10^{-9}$ Ha/Bohr against PySCF FD | same, external-potential-in-RI-MP2 row |
| analytic RHF/UHF Hessian with point and smeared charges | H2O, CH3OH, H2O$^+$ + charges | up to $8.0\times10^{-7}$ Ha/Bohr$^2$ against FD of PySCF's gradient; the gas-phase Hessian misses by $7.7\times10^{-3}$ to $1.6\times10^{-2}$ | same, QM/MM Hessian row |

The Hessian row also gives the size of the effect. The charges change the
Hessian by at least $7.7\times10^{-3}$ Ha/Bohr$^2$, about $10^4$ times the
largest residual against PySCF.

## Sources {#external-potentials-sources}

| Source | What it gives |
|---|---|
| `scripts/external_potential_sympy.py`, claims E1 to E5 | the proof |
| `crates/ferric-core/src/external_potential.rs` | the struct, the classical energy, gradient and Hessian |
| `crates/ferric-integrals/src/oneelectron.rs` | the one-electron terms |
| `crates/ferric-scf/tests/qmmm_vs_pyscf.rs`, `crates/ferric-scf/tests/qmmm_smeared.rs`, `crates/ferric-scf/tests/validation_qmmm_hessian.rs` | the PySCF comparisons |

**Known limits (2026-11-15).**

- The field–electron gradient term is a finite difference, not analytic.
- No force on the charges appears in `oneelectron_gradient`; use section @sec:qmmm-boundaries.
- The MP2 and coupled-cluster Python drivers take no external potential, and
  analytic Hessians exclude some reference and functional combinations
  (see the validation matrix on `site/src/reference/validation.md`).
