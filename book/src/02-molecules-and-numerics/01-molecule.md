# 2.1 Molecule, charge, multiplicity and units

<!-- section: 2.1; status: edit; registry: ;
     theory: crates/ferric-core/theory.md#"1. Molecular Representation & Coordinate Conventions" -->

A calculation starts from a list of atoms, a net charge and a spin
multiplicity. From those the code derives the electron count, the numbers of
$\alpha$ and $\beta$ electrons, the nuclear repulsion and the number of
orbitals a frozen-core calculation leaves out.

| Quantity | Symbol | Code name |
|---|---|---|
| Atoms | $A, B$ | `Molecule::atoms`, in input order |
| Atomic number | $Z_A$ | `Atom::z` |
| Core electrons replaced by an ECP | $n^{\mathrm{ECP}}_A$ | `Atom::n_core_ecp` |
| Net charge | $q$ | `Molecule::charge` |
| Multiplicity | $M = 2S+1$ | `Molecule::multiplicity` |
| Electrons | $N$ | `Molecule::nelec()` |

| Entry point | Location |
|---|---|
| Rust | `ferric-core/src/mol.rs::{Molecule,Atom,parse_xyz,from_atoms,nelec,nuclear_repulsion,auto_frozen_core,apply_ecp}` |
| CLI | `[molecule]` section, keys `xyz`, `charge`, `multiplicity` (`examples/water-rhf.toml`) |
| Python | `ferric.Molecule.from_xyz` (`crates/ferric-python/ferric.pyi`) |

## 2.1.1 Coordinates and units

Inside the code every length is in Bohr and every energy in Hartree. The XYZ
parser reads Ångström and multiplies by one constant,
`ferric-core/src/units.rs::ANGSTROM_TO_BOHR`, the reciprocal of the Bohr
radius $0.52917721092$ Å (CODATA 2010, the value PySCF uses). Every crate
converts through it, and a test fails on any other Å–Bohr literal in the
source. The same constant converts an inverse length: a range-separation
parameter given in Å$^{-1}$ times `BOHR_TO_ANGSTROM` is in Bohr$^{-1}$
(notation chapter, sections 8 and 9, lists which surface takes which unit).

An XYZ file has an atom count, one comment line and one line per atom with
four fields: a symbol and three coordinates. The parser errors on a short
line, an unknown symbol or a non-numeric coordinate. The element table covers
$Z = 1$ to $54$ (H to Xe).

## 2.1.2 Ghost atoms and effective charge

A symbol prefixed with `@` is a ghost atom: it carries the full basis of its
element and no nuclear charge and no electrons. Counterpoise corrections use
it. The atomic number of a ghost stays the physical one, so basis lookup by
element works unchanged. The charge the electrons see is

$$ Z^{\mathrm{eff}}_A = \begin{cases} 0 & \text{ghost} \\ Z_A - n^{\mathrm{ECP}}_A & \text{otherwise} \end{cases} \tag{2.1} $$

(`ferric-core/src/mol.rs::Atom::effective_z`). `apply_ecp` sets
$n^{\mathrm{ECP}}_A$ from the ECP definitions of the basis set, and skips
ghosts. It is the single point where an ECP enters the molecule.

## 2.1.3 Electrons and spin

$$ N = \sum_A Z^{\mathrm{eff}}_A - q, \qquad N_\alpha = \frac{N + M - 1}{2}, \qquad N_\beta = N - N_\alpha . \tag{2.2} $$

The multiplicity is $2S+1$ with $2S = N_\alpha - N_\beta$, which gives
$N_\alpha$ in (2.2). The construction `from_atoms` refuses a combination for
which $N_\alpha$ is not an integer between $0$ and $N$: an odd electron count
needs an even multiplicity, and $M - 1$ may not exceed $N$. The error names
the three numbers. Without the check, an inconsistent molecule would fail
later in the SCF loop with a message that does not mention the molecule.

A restricted Hartree–Fock reference needs an even $N$ and $M = 1$. The CLI
kind `rhf` refuses $M > 1$ and points to `uhf` or `rohf` (validation page,
"Open shells in the CLI").

## 2.1.4 Nuclear repulsion

$$ E_{\mathrm{nn}} = \sum_{A<B} \frac{Z^{\mathrm{eff}}_A Z^{\mathrm{eff}}_B}{|\mathbf R_A - \mathbf R_B|} . \tag{2.3} $$

`ferric-core/src/mol.rs::Molecule::nuclear_repulsion` loops over pairs, skips
ghosts and uses the effective charges. For an ECP atom this is the repulsion
of the screened core.

## 2.1.5 Frozen core

`frozen_core = "auto"` freezes the small-core convention that Psi4, ORCA and
Q-Chem use. `ferric-core/src/mol.rs::core_orbitals` counts the orbitals per
element:

| Elements | Core orbitals |
|---|---|
| H–He | 0 |
| Li–Ne | 1 |
| Na–Ar | 5 |
| K–Zn | 9 |
| Ga–Kr | 14 |
| Rb–Cd | 18 |
| In–Xe | 23 |

The table is a convention, not a derived quantity. The $(n-1)$d shell stays
correlated across its own transition series and becomes core from the
following p-block on. The molecule's count is the sum over real atoms, minus
$n^{\mathrm{ECP}}_A/2$ for each ECP atom, since the potential has already
removed those orbitals. It must be read **after** `apply_ecp`, or an ECP run
freezes orbitals that are not in the calculation. Ghosts contribute zero.

Where the count enters an energy, chapters 10 and 16 use
`ferric-mp2/src/rimp2.rs::active_occ`, which returns an error rather than
underflow when the freeze covers every occupied orbital.

## Known limits

- The element table stops at xenon, so every XYZ symbol must be H to Xe.
- The auto-freeze count is not checked against the occupied count in this
  module. The check lives at the configuration boundary, where it can name the
  offending key.

## Sources

| Source | Used for |
|---|---|
| `crates/ferric-core/theory.md`, "1. Molecular Representation & Coordinate Conventions" | the $E_{\mathrm{nn}}$ formula and the charge and multiplicity definitions |
| `crates/ferric-core/src/mol.rs`, `crates/ferric-core/src/units.rs` | everything above, read against the code |

Changes from the sources: the Bohr radius is quoted with the 11 digits the
code holds. `theory.md` has 10 digits. Ghosts, effective charges, the
$N_\alpha$ rule and the frozen-core table are in the code and not in
`theory.md`.
