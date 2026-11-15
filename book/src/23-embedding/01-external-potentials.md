# 23.1 Point charges, smeared charges and uniform fields

<!-- section: 23.1; status: gap; registry: none; theory: none -->

The simplest environment is a set of fixed classical charges and a uniform
electric field. The charges are not atoms: they have no basis functions, no
spin and no wavefunction, and their charges can be fractional. They enter the
quantum problem as one extra one-electron potential and one extra classical
energy, and nothing else in the SCF changes. Every other embedding in this
chapter, from QM/MM partial charges to the polarizable sites of section 23.3,
is built on this operator.

This section has no `theory.md` text and no proof script. The equations are
read from the code loops and are marked unverified; the tests that pin them
are named in each box.

| Symbol | Meaning | Code name |
|---|---|---|
| $q_i$, $\mathbf R_i$ | point charge and position, Bohr | `PointCharge::{q,x,y,z}` |
| $q_j$, $\mathbf R_j$, $w_j$ | smeared charge, centre, Gaussian width (Bohr) | `SmearedCharge::{q,x,y,z,width}` |
| $\zeta_j$ | $1/w_j^2$ | `1.0 / (s.width * s.width)` |
| $\mathbf E$ | uniform field, atomic units, lab frame | `ExternalPotential::field` |
| $Z_A$, $\mathbf R_A$ | effective nuclear charge (valence charge for an ECP atom) and position | `effective_z`, `atom.{x,y,zpos}` |
| $\mu, \nu$ | AO basis functions | `0..nbas` |

| Entry point | Location |
|---|---|
| Rust | `ferric-core/src/external_potential.rs::ExternalPotential`, `ferric-integrals/src/oneelectron.rs::{hcore_with_external,hcore_ecp_with_external,nuclear_with_external,smeared_attraction,field_hcore_term}`, `ferric-scf/src/rhf.rs::RhfConfig` field `external_potential` |
| CLI | `[external_potential]` with `[[external_potential.point_charges]]` (`q`, `x`, `y`, `z`, optional `width`) and `field` (`examples/water-rhf-smeared-charge.toml`) |
| Python | `point_charges=`, `smeared_charges=`, `external_field=` on `run_rhf`, `run_uhf`, `run_rohf`, `run_dft` and the geometry drivers |

## 23.1.1 The operator and the classical energy

A point charge is attractive to an electron through $-q_i/|\mathbf r-\mathbf R_i|$.
A Gaussian-smeared charge has density $q_j(\zeta_j/\pi)^{3/2}e^{-\zeta_j|\mathbf r-\mathbf R_j|^2}$,
whose potential is $q_j\operatorname{erf}(|\mathbf r-\mathbf R_j|/w_j)/|\mathbf r-\mathbf R_j|$; it
tends to the point-charge potential as $w_j \to 0$, which is why a width of
exactly zero is not a valid smeared charge and a point charge is its own
representation. The field adds $+\mathbf E\cdot\mathbf r$ for each electron,
because the electron charge is $-1$.

**Equation 23.1, the embedding operator and the classical energy.** (Transcribed from the code loop; see Known limits.)

$$ H^{\mathrm{ext}}_{\mu\nu} = -\sum_i q_i\Big\langle\mu\Big|\tfrac{1}{|\mathbf r-\mathbf R_i|}\Big|\nu\Big\rangle - \sum_j q_j\Big\langle\mu\Big|\tfrac{\operatorname{erf}(|\mathbf r-\mathbf R_j|/w_j)}{|\mathbf r-\mathbf R_j|}\Big|\nu\Big\rangle + \mathbf E\cdot\langle\mu|\mathbf r|\nu\rangle , $$

$$ E^{\mathrm{ext}}_{\mathrm{nuc}} = \sum_A Z_A\Big[\sum_i \frac{q_i}{|\mathbf R_A-\mathbf R_i|} + \sum_j q_j\frac{\operatorname{erf}(|\mathbf R_A-\mathbf R_j|/w_j)}{|\mathbf R_A-\mathbf R_j|}\Big] - \mathbf E\cdot\sum_A Z_A\mathbf R_A . $$

The SCF energy is the vacuum functional with $h \to h + H^{\mathrm{ext}}$ and
$V_{nn} \to V_{nn} + E^{\mathrm{ext}}_{\mathrm{nuc}}$. The interaction of
the external charges with each other is not included (it is a constant of
the electronic problem; section 23.4 supplies it from a force field).

Origin dependence of the field. The field term uses dipole integrals about the
coordinate origin, and $E^{\mathrm{ext}}_{\mathrm{nuc}}$ carries $-\mathbf E\cdot\sum_AZ_A\mathbf R_A$.
Shifting every coordinate by $\mathbf a$ changes the total by
$-\mathbf E\cdot\mathbf a\,(\sum_AZ_A - N_e)$, so the energy of a charged
molecule depends on where the origin is; for a neutral molecule it does not.
(This follows from the two terms above and is not separately tested.)

## 23.1.2 Where the terms enter the SCF

The charges and the field are in the core Hamiltonian once, before the
iteration loop, and the classical energy is in $V_{nn}$ once. Neither is
recomputed. This is different from the density-dependent terms of sections
22.3 and 23.3.

| Term | Where | Function |
|---|---|---|
| point-charge attraction | appended to the real atoms as extra nuclei of charge $q_i$ in the nuclear-attraction engine | `nuclear_with_external` |
| smeared-charge attraction | three-index Coulomb integral against one normalized Gaussian per site | `smeared_attraction` |
| field | dipole integrals about the origin, contracted with $\mathbf E$ | `field_hcore_term` |
| the sum | $T + V + $ the three terms above | `hcore_with_external` |
| the ECP projector, when the basis carries one | added on top | `hcore_ecp_with_external` |
| $E^{\mathrm{ext}}_{\mathrm{nuc}}$ | added to $V_{nn}$ before the loop | `ferric-scf/src/driver.rs::prepare` |

`hcore_with_external` assembles $h$ in four lines:

<!-- excerpt: ferric-integrals/src/oneelectron.rs::hcore_with_external -->
```rust
let mut h = t + v;
if !ext.smeared_charges.is_empty() {
    h += &smeared_attraction(prep, &ext.smeared_charges)?;
}
if let Some(field) = ext.field {
    h += &field_hcore_term(prep, field)?;
}
```

| Symbol | Code |
|---|---|
| $T + V_{\mathrm{nuc}} + V_{\text{point charges}}$ | `t + v` |
| $\sum_j(\ldots)$ smeared | `smeared_attraction(...)` |
| $\mathbf E\cdot\langle\mu\vert\mathbf r\vert\nu\rangle$ | `field_hcore_term(...)` |

The smeared integral is the Coulomb integral of the basis pair against a
normalized s Gaussian of exponent $\zeta_j$ at the site, divided by that
Gaussian's integral so the charge is $q_j$:

<!-- excerpt: ferric-integrals/src/oneelectron.rs::smeared_attraction -->
```rust
v[(o1 + a, o2 + b)] += -sc.q * block[a * n2 + b] / norm;
```

| Symbol | Code |
|---|---|
| $-q_j$ | `-sc.q` |
| $(\mu\nu\vert g_j)$ | `block[a * n2 + b]` |
| $\int g_j\,d\mathbf r$ | `norm`, from `SiteBasis::norm_int` |

The classical energy of the smeared charges against the nuclei is the erf form
of equation 23.1:

<!-- excerpt: ferric-core/src/external_potential.rs::charge_nuclear_energy -->
```rust
e += za * sc.q * erf(sqrt_zeta * r) / r;
```

**Exactness.** An empty `ExternalPotential` (no point charges, no smeared
charges, no field) takes the vacuum path in `hcore_with_external`, and the SCF
energy is bit-identical to a run with `external_potential = None`
(`external_potential_none_matches_default_exactly` in `ferric-scf/src/rhf.rs`).

## 23.1.3 The gradient

With the charges fixed in space, the gradient over the QM atoms has five
pieces. They are added in `ferric-scf/src/gradient.rs::oneelectron_gradient`.

| Piece | Source | Method |
|---|---|---|
| electron–point-charge, Hellmann–Feynman | the nuclear-attraction derivative engine with the charges as extra centres; the shell-centre and real-atom blocks are accumulated and the charge-centre blocks are written but not accumulated, since the charges are not variables | analytic, libint |
| classical charge–nucleus | $\partial E^{\mathrm{ext}}_{\mathrm{nuc}}/\partial\mathbf R_A$ | analytic, `ExternalPotential::charge_nuclear_gradient` |
| field–nucleus | $-Z_A\mathbf E$ per atom | analytic, `ExternalPotential::field_nuclear_gradient` |
| field–electron, $\sum_{\mu\nu}D_{\mu\nu}\,\partial\langle\mu|\mathbf E\cdot\mathbf r|\nu\rangle/\partial\mathbf R_A$ | central difference ($h = 10^{-4}$ Bohr) of the field one-electron term at fixed $D$; libint has no derivative of the dipole operator | `field_density_gradient_fd` |
| smeared-charge attraction | three-index derivative integrals, `smeared_charge_qm_gradient` | analytic |

The force on a charge itself is not part of this gradient. For a point charge
it is $q\,\mathbf E^{\mathrm{QM}}(\mathbf R_i)$, the field of the QM nuclei and
electron density at the charge, and for a smeared charge it needs the derivative
of $(\mu\nu|g_j)$ with respect to the site, obtained from the two basis-centre
blocks by translational invariance. Both are in section 23.2
(`ferric-scf/src/qmmm.rs::mm_forces`, `ferric-scf/src/gradient.rs::smeared_site_forces`).

The analytic Hessian with external charges (QM–QM block, charges fixed) and the
smeared-charge version are validated on the page named in the anchors table.

## 23.1.4 Options

| Option | Values | Default | Effect |
|---|---|---|---|
| `[[external_potential.point_charges]]` | `q`, `x`, `y`, `z` in Bohr; `width` in Bohr optional | none | a charge; a `width` makes it smeared, and `width` $\le 0$ is an error |
| `[external_potential] field` | $[E_x, E_y, E_z]$, atomic units | none | the uniform field |
| `[qmmm]` with `[external_potential]` | | | refused: the MM region is an external potential, so both would count it |
| Python `point_charges=`, `smeared_charges=`, `external_field=` | `(q,x,y,z)`, `(q,x,y,z,width)`, `(Ex,Ey,Ez)` | none | the same, in Bohr and atomic units |

An omitted `[external_potential]` section, and an empty one, both take the
vacuum path. The MP2 and coupled-cluster Python drivers take no
external-potential arguments; the RI-MP2 gradient in Rust threads one through
(third row of the anchors).

## 23.1.5 Numerical anchors

| Quantity | System | Result | Source |
|---|---|---|---|
| energy and embedding shift against PySCF `qmmm.mm_charge`; QM gradient; forces on the charges | H2O + 10 charges / cc-pVDZ, aug-cc-pVDZ; CH3OH + 501 TIP3P charges / 6-31G | energy and shift $\le 4.8\times10^{-12}$ Ha; gradient $\le 1.2\times10^{-10}$; forces $\le 4.4\times10^{-12}$ Ha/Bohr | `site/src/reference/validation.md`, QM/MM embedding row |
| Gaussian-smeared charges against `mm_charge(radii=)` | H2O + 10 charges / cc-pVDZ, widths 0.5, 1, 2 Bohr | energy $\le 4.1\times10^{-12}$ Ha; gradient $\le 1.2\times10^{-10}$ Ha/Bohr; width $\to0$ reproduces point charges to $3.0\times10^{-12}$ Ha | same, smeared-charge row |
| the RI-MP2 analytic gradient with an external potential | H2O + 10 charges / cc-pVDZ; CH3OH + 20 charges / 6-31G | energies $\le 8.4\times10^{-12}$ Ha; gradient $\le 4.8\times10^{-9}$ Ha/Bohr against PySCF FD | same, external-potential-in-RI-MP2 row |
| analytic RHF/UHF Hessian with point and smeared charges | H2O, CH3OH, H2O$^+$ + charges | up to $8.0\times10^{-7}$ Ha/Bohr$^2$ against FD of PySCF's gradient; the gas-phase Hessian misses by $7.7\times10^{-3}$ to $1.6\times10^{-2}$ | same, QM/MM Hessian row |

## 23.1.6 Sources

| Source | What it gives |
|---|---|
| `crates/ferric-core/src/external_potential.rs` | the struct, the classical energy, gradient and Hessian |
| `crates/ferric-integrals/src/oneelectron.rs` | the one-electron terms |
| `crates/ferric-scf/tests/qmmm_vs_pyscf.rs`, `crates/ferric-scf/tests/qmmm_smeared.rs`, `crates/ferric-scf/tests/validation_qmmm_hessian.rs` | the PySCF comparisons |

**Known limits (2026-11-15).**

- The displayed equation 23.1 are transcribed from the code loop and await a
  `theory.md` derivation and a sympy proof; they are not machine-checked.
- The field–electron gradient term is a finite difference, not analytic.
- No force on the charges appears in `oneelectron_gradient`; use section 23.2.
- The MP2 and coupled-cluster Python drivers take no external potential, and
  analytic Hessians exclude some reference and functional combinations
  (see the validation matrix on `site/src/reference/validation.md`).

**Theory text still needed in `crates/ferric-scf/theory.md`, which another pass is editing.**
Equation 23.1 with the sign conventions of the two Coulomb terms; the $w\to0$
limit of the smeared potential; the origin-shift identity of the field term;
and the classification of the five gradient pieces of 23.1.3 by whether
libint supplies the derivative.
