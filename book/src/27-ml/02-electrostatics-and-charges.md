# Electrostatic features and atomic charges {#electrostatics-and-charges}

<!-- status: gap; registry: none -->

What does a neighbouring molecule, or a model of one, need to know about this
molecule's electrons? At long range, mostly the electrostatics: the potential
the density makes, the field it exerts at each nucleus, and some way of
summing the density into a number per atom. These are the cheapest features
ferric exports that still carry information about the density, and they are
the inputs of ESP-conditioned models such as the one in section @sec:esp-conditioning.

All of them are functions of the reference density $D = 2C_oC_o^{\mathsf T}$
and of nothing else. This section states them as read from the loops in
`crates/ferric-scf/src/properties.rs`; no proof script covers them.
Section @sec:atomic-partitions derives the charge schemes; this section states what each
exported number is and how far it is checked.

The plan: the potential and field at the nuclei (@sec:electrostatics-and-charges-potential-field-nuclei), the potential at
other points (@sec:electrostatics-and-charges-potential-arbitrary-points-surface), multipoles (@sec:electrostatics-and-charges-multipoles-orbital-moments), the five charge schemes (@sec:electrostatics-and-charges-five-charge-schemes),
the checks (@sec:electrostatics-and-charges-well-these-checked) and the limits (@sec:electrostatics-and-charges-known-limits-2026-11).

| Quantity | Rust | Python |
|---|---|---|
| ESP at nuclei | `ferric-scf/src/properties.rs::esp_at_atoms` | `esp_at_atoms(mol, basis_set, result)` |
| ESP at points | `ferric-scf/src/properties.rs::esp_at_points` | `esp_at_points(mol, basis_set, result, points)`, points in Bohr |
| Surface ESP | `ferric-scf/src/properties.rs::esp_on_surface` | `esp_on_surface(mol, basis_set, result, vdw_scale=1.4, n_angular=110)`; NPZ `esp_surface` |
| Field at nuclei | `ferric-scf/src/properties.rs::electric_field_at_atoms` | NPZ `electric_field` |
| Dipole | `ferric-scf/src/properties.rs::dipole_moment` | NPZ `dipole` |
| Löwdin, Mulliken | `ferric-scf/src/properties.rs::{lowdin_charges,mulliken_charges}` | `lowdin_charges`, `mulliken_charges` |
| Hirshfeld | `ferric-scf/src/properties/hirshfeld_grid.rs::hirshfeld_charges` | `hirshfeld_charges(..., proatom="scf")` |
| CHELPG, RESP | `ferric-scf/src/properties.rs::{chelpg_charges,resp_charges,chelpg_and_resp_charges}` | `chelpg_charges`, `resp_charges` |

The Python functions take the molecule, the basis and an `RhfResult` or
`DftResult`, so they work on a Kohn–Sham density as well. Mulliken, Löwdin,
CHELPG and RESP are documented as closed-shell only.

Throughout this section $Z_A$ in the nuclear terms is the effective charge
$Z^{\mathrm{eff}}_A$ of equation @eq:molecule-1: the valence charge on an ECP atom, zero
on a ghost, and the bare $Z_A$ otherwise (section @sec:electrostatics-and-charges-known-limits-2026-11).

## Potential and field at the nuclei {#electrostatics-and-charges-potential-field-nuclei}

What potential does nucleus $A$ feel from the rest of the molecule? Every
other nucleus contributes $Z_B/r$, and the electrons contribute minus the
Coulomb integral of the density. $A$'s own nuclear term is infinite at
$\mathbf R_A$, so it is left out:

$$ V(\mathbf R_A) = \sum_{B\neq A}\frac{Z_B}{\lvert\mathbf R_A-\mathbf R_B\rvert} - \sum_{\mu\nu} D_{\mu\nu}\,\Big\langle\mu\Big\lvert\frac{1}{\lvert\mathbf r-\mathbf R_A\rvert}\Big\rvert\nu\Big\rangle . \label{eq:electrostatics-and-charges-1} $$

The electronic integral is libint2's nuclear-attraction operator with a unit
probe charge at $\mathbf R_A$. The field is minus the gradient of the same
potential with respect to the probe position, again without the self term:

$$ \mathbf E(\mathbf R_A) = \sum_{B\neq A} Z_B\frac{\mathbf R_A-\mathbf R_B}{\lvert\mathbf R_A-\mathbf R_B\rvert^3} - \sum_{\mu\nu} D_{\mu\nu}\,\Big\langle\mu\Big\lvert\frac{\mathbf R_A-\mathbf r}{\lvert\mathbf r-\mathbf R_A\rvert^3}\Big\rvert\nu\Big\rangle . \label{eq:electrostatics-and-charges-2} $$

Check the sign with one other nucleus $B$ and no electrons: (@eq:electrostatics-and-charges-2) reduces
to $Z_B(\mathbf R_A-\mathbf R_B)/\lvert\mathbf R_A-\mathbf R_B\rvert^3$,
which points from $B$ toward $A$, as the field of a positive charge should.

Both are exact functions of $D$: no grid and no fit. $V(\mathbf R_A)$ is a
rotation- and translation-invariant scalar per atom, and $\mathbf E$ is a
vector per atom. Two coincident atoms are an error, not an infinity. The
nuclear half of `esp_at_atoms` shows both points, with the distance setup
trimmed:

```rust
// Nuclear sum: Σ_{B ≠ A} Z_B / |R_A − R_B|
let mut v_nuc = 0.0_f64;
for b in 0..natoms {
    if b == a {
        continue;                                   // drop A's own Z_A / 0
    }
    // ...
    if r < 1e-12 {
        return Err(FerricError::General(format!(
            "esp_at_atoms: atoms {a} and {b} coincide"
        )));
    }
    v_nuc += atom_b.effective_z() as f64 / r;      // Z^eff_B / |R_A − R_B|
}
Ok(v_nuc + v_elec)                                  // v_elec: −Σ D_{μν}⟨μ|1/|r−R_A||ν⟩
```

- The electronic part `v_elec` is accumulated before this loop over shell
  pairs $s_1 \ge s_2$, with off-diagonal pairs counted twice. The sign of the
  electron term comes from libint2's nuclear-attraction operator, which
  already carries the minus.
- The charge is `effective_z()`, not `z`. On an ECP atom the density holds
  only valence electrons, so pairing it with the bare $Z$ would leave the
  core charge unscreened.

## Potential at arbitrary points and on a surface {#electrostatics-and-charges-potential-arbitrary-points-surface}

At a point $\mathbf r_g$ that is not a nucleus, every nucleus is included:

$$ V(\mathbf r_g) = \sum_{A}\frac{Z_A}{\lvert\mathbf r_g-\mathbf R_A\rvert} - \sum_{\mu\nu} D_{\mu\nu}\,\Big\langle\mu\Big\lvert\frac{1}{\lvert\mathbf r-\mathbf r_g\rvert}\Big\rvert\nu\Big\rangle . \label{eq:electrostatics-and-charges-3} $$

`esp_at_points` takes the points in **Bohr**. Because the self term is kept,
$V(\mathbf r_g)$ diverges as $\mathbf r_g$ approaches a nucleus; it does not
tend to `esp_atoms` there.

The NPZ surface (`compute_esp_surface = true`) and the Python
`esp_on_surface` sample (@eq:electrostatics-and-charges-3) on the outside of the molecule. Each atom gets
a Lebedev sphere at `esp_surface_vdw_scale` × its Bondi radius
(`ferric-pcm/src/radii.rs::bondi_radius_bohr`), and points that fall inside
another atom's scaled sphere are dropped. The defaults are a scale of 1.4 and
110 angular points. The Python function also returns how many points were
dropped as buried.

The Lebedev spheres have a fixed orientation in the lab frame. Rotating the
molecule therefore changes the point set, not only its coordinates, and the
values at corresponding points are not available after a rotation. Store
`esp_points` with `esp_surface` (the writer refuses one without the other)
and treat the pair as a point cloud, not as a fixed-length vector.

## Multipoles and orbital moments {#electrostatics-and-charges-multipoles-orbital-moments}

| Key | Definition | Note |
|---|---|---|
| `dipole` | $\boldsymbol\mu = \sum_A Z^{\mathrm{eff}}_A\mathbf R_A - \sum D_{\mu\nu}\langle\mu\lvert\mathbf r\rvert\nu\rangle$ | origin $(0,0,0)$; origin-dependent when the molecule is charged (@sec:npz-bundle-keys) |
| `density_second_moment` | $M_{cd} = \sum D_{\mu\nu}\langle\mu\lvert r_c r_d\rvert\nu\rangle$ | electronic only, about the origin; translation changes it |
| `orbital_centers` | $\langle\phi_p\lvert\mathbf r\rvert\phi_p\rangle$ | every retained MO, restricted references only |
| `orbital_spreads` | $\sqrt{\max(0,\langle r^2\rangle_p-\lvert\langle\mathbf r\rangle_p\rvert^2)}$ | canonical MOs, so delocalized; restricted only |

The dipole, like the ESP, field and charges, uses the ECP-reduced nuclear
charge $Z^{\mathrm{eff}}_A = Z_A - n^{\mathrm{core}}_A$. The spreads are the
root-mean-square radius of each orbital about its own centroid. The
$\max(0,\cdot)$ guards against a small negative variance from rounding.
Because the MOs are canonical, a valence orbital spread over the whole
molecule has a large spread; these are not localized-orbital sizes.

## Five charge schemes {#electrostatics-and-charges-five-charge-schemes}

A charge per atom is a common model feature, and there is no unique
one: the density is a property of the whole molecule. Each scheme below is
a rule for dividing it, and section @sec:atomic-partitions derives what each rule conserves.
This table is the consumer's summary.

| Scheme | Definition | Depends on |
|---|---|---|
| Mulliken | $q_A = Z_A - \sum_{\mu\in A}(DS)_{\mu\mu}$ ({#eq:electrostatics-and-charges-mulliken}) | basis; unstable with diffuse functions |
| Löwdin | $q_A = Z_A - \sum_{\mu\in A}(S^{1/2}DS^{1/2})_{\mu\mu}$ ({#eq:electrostatics-and-charges-l-wdin}) | basis, less strongly |
| Hirshfeld | $q_A = Z_A - s\!\int w_A\rho$, $w_A = \rho^0_A/(\sum_B\rho^0_B + 10^{-12})$ ({#eq:electrostatics-and-charges-hirshfeld}) | proatoms, grid |
| CHELPG | least-squares fit of point charges to $V(\mathbf r_g)$, $\sum_A q_A = $ molecular charge (@eq:electrostatics-and-charges-4) | point lattice |
| RESP | CHELPG plus a hyperbolic restraint on non-H charges | point lattice, restraint |

**Löwdin** forms $S^{1/2}$ from the eigendecomposition of $S$ and errors if an
eigenvalue is not positive.

**Hirshfeld** integrates on the flat 75 × 110 Becke–Lebedev grid
(`AtomicGridConfig::default()`). The proatom densities $\rho^0_A$ come from a
neutral free-atom SCF in the same basis, which is the default of both the CLI
and Python. `proatom="slater"` selects a single-Slater model instead, and the
code falls back to it, with a warning, when the free-atom SCF fails. The
factor $s = N_e/\sum_A\!\int w_A\rho$ rescales the integrated populations to
the exact electron count. So $\sum_A q_A$ equals the molecular charge by
construction, and the grid error goes into the individual charges only.

**CHELPG** fits charges to the potential instead of dividing the density.
It builds a cubic lattice in the lab frame with spacing 0.3 Bohr, extending
2.8 Bohr beyond the molecule. A point is kept if it lies outside every
atom's Bondi radius and within 2.8 Bohr of some atom's radius. Minimizing the
squared misfit to $V(\mathbf r_g)$ with the total charge held fixed by a
multiplier $\lambda$ gives the bordered normal equations (derived in section
@sec:atomic-partitions-esp-fitted-charges as equation @eq:charges-chelpg-bordered):

$$ \begin{pmatrix} A & \mathbf 1\\ \mathbf 1^{\mathsf T} & 0\end{pmatrix}\begin{pmatrix}\mathbf q\\ \lambda\end{pmatrix} = \begin{pmatrix}\mathbf b\\ Q\end{pmatrix}, \qquad A_{AB} = \sum_g \frac{1}{r_{gA}r_{gB}},\quad b_A = \sum_g \frac{V(\mathbf r_g)}{r_{gA}} , \label{eq:electrostatics-and-charges-4} $$

with $Q$ the molecular charge. The environment variables
`FERRIC_CHELPG_SPACING`, `FERRIC_CHELPG_MARGIN`, `FERRIC_CHELPG_VDW_SCALE` and
`FERRIC_CHELPG_OUTER_CUTOFF` override the four lattice parameters. A dataset
that sets them must record them, because the certificate does not.

**RESP** adds the single-stage hyperbolic restraint
$a\sum_{A\notin\mathrm H}(\sqrt{q_A^2+b^2}-b)$ with $a = 0.0005$ and
$b = 0.1\,e$ (`FERRIC_RESP_RESTRAINT_WEIGHT`, `FERRIC_RESP_RESTRAINT_B`).
It solves by fixed-point iteration: each pass adds $a/\sqrt{q_A^2+b^2}$ to
the diagonal of the bordered matrix of (@eq:electrostatics-and-charges-4), evaluated at the previous
charges. It is not two-stage or multi-conformer RESP. When both are
requested, CHELPG and RESP share one lattice and one set of ESP values.

**Rotation.** Because the lattice is fixed in the lab frame, CHELPG and RESP
charges move slightly when the molecule is rotated. Mulliken is invariant
under rotation. Löwdin is invariant for spherical-harmonic (`pure`) shells;
Cartesian d shells transform non-orthogonally and need not leave it
unchanged. Hirshfeld is invariant up to the error of its atom-centred grid. A
committed test holds symmetry-equivalent atoms equal, and the charges
unchanged under translation, to below $10^{-3}\,e$
(`ferric-rpa/tests/hirshfeld_symmetry.rs::equivalent_atoms_are_equal_in_a_rotated_frame`,
`ferric-rpa/tests/hirshfeld_symmetry.rs::charges_are_invariant_to_translation`).

![Charges by scheme and the polarizability split for water/cc-pVDZ](../figures/27-ml/charges_and_alpha.svg)

*Water/cc-pVDZ: (a) O and H charges in the five schemes;
(b) the isotropic static α as $\sum_A\alpha^A$ plus $\alpha_{\mathrm{CT}}$
(section @sec:polarizability-pdep-c6). Made by `book/src/figures/27-ml/charges_and_alpha.py`.*

Panel (a) shows five different numbers for the same oxygen. They are five
definitions, not five estimates of one quantity, so a model trained on a mix
of them fits the difference between definitions. Pick one scheme for a
dataset and record which; the NPZ key name is the only tag the charges carry.

## How well these are checked {#electrostatics-and-charges-well-these-checked}

| Quantity | Check | Agreement | Test |
|---|---|---|---|
| ESP at nuclei | PySCF `int1e_rinv` on PySCF's density | 4.8e-14 a.u. same density; 4.6e-9 own SCF | `ferric-scf/tests/validation_density_properties.rs::density_properties_h2o_vs_pyscf` |
| Field at nuclei | PySCF `int1e_iprinv`, same density | 1.5e-13 a.u.; 2.7e-9 own SCF | the same file |
| Hirshfeld charges | numpy on ferric's grid with PySCF density and proatoms | 1.9e-13 $e$ same density; 2.6e-9 $e$ own SCF; default grid within 2.0e-4 $e$ of a dense grid | `crates/ferric-rpa/tests/validation_hirshfeld.rs` |
| All NPZ keys | PySCF integrals contracted with the exported density | integral level ≤ 2.8e-14; property chain ≤ 5.9e-10 | `crates/ferric-cli/tests/validation_npz.rs` |

The figures are from `site/src/reference/validation.md`, "Anchors". Two
columns of agreement separate two error sources. "Same density" feeds
ferric the reference code's density, so it measures ferric's integrals and
contractions alone. "Own SCF" lets ferric converge its own density, so it
adds the SCF convergence error.

The CHELPG and RESP checks in `validation_npz.rs` are invariants only: the
charges sum to the molecular charge, respect the mirror symmetry of water,
and reproduce the surface ESP they were fitted to. No external code supplies
a reference value for them.

## Known limits (2026-11-15) {#electrostatics-and-charges-known-limits-2026-11}

- **Effective nuclear charges.** `esp_at_atoms`, `electric_field_at_atoms`,
  `esp_at_points`, `becke_charges`, `lowdin_charges`, `mulliken_charges`,
  `hirshfeld_charges` and `hirshfeld_i_charges` pair the density with
  $Z^{\mathrm{eff}}_A$ (@eq:molecule-1): the valence charge on an ECP atom and zero on a
  ghost, so $Z_A$ in (@eq:electrostatics-and-charges-3)–(@eq:electrostatics-and-charges-hirshfeld) reads as $Z^{\mathrm{eff}}_A$. Checked
  against PySCF RHF with ECPs on HI and CH3I / def2-SVP (max |d| 1.8e-9;
  `crates/ferric-scf/tests/properties_effective_z.rs`, and through the Python
  bindings, which apply the basis's ECP first). Hirshfeld on an ECP atom still
  divides by a proatom built for the full $Z$; that pairing is unchecked.
- CHELPG and RESP charges depend on molecular orientation (@sec:electrostatics-and-charges-five-charge-schemes).
- The surface-ESP point set depends on orientation (@sec:electrostatics-and-charges-potential-arbitrary-points-surface).

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-scf/src/properties.rs` | eqs. @eq:electrostatics-and-charges-1–@eq:electrostatics-and-charges-mulliken, @eq:electrostatics-and-charges-4, the lattice and restraint parameters, the effective-$Z$ pairing, the excerpt in @sec:electrostatics-and-charges-potential-field-nuclei |
| `crates/ferric-scf/src/properties/hirshfeld_grid.rs` | eq. @eq:electrostatics-and-charges-hirshfeld, the grid, the proatoms and the rescaling |
| `crates/ferric-pcm/src/radii.rs` | the Bondi radii |
| `crates/ferric-python/ferric.pyi` | the Python names, `esp_on_surface` and argument units |
| `site/src/reference/validation.md` | the agreement figures in @sec:electrostatics-and-charges-well-these-checked |

Changes from the sources: none; there is no theory text to change.
