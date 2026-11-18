# Molecule, charge, multiplicity and units {#molecule}

<!-- status: edit; registry: -->

A calculation starts from a list of nuclei, a net charge and a spin
multiplicity. Everything else about the electrons follows from those three
inputs: how many there are, how many have spin up and how many spin down,
how strongly the bare nuclei repel each other, and how many core orbitals a
frozen-core calculation can leave out. This section derives each of these
and shows the code that computes it.

The plan:

1. Read coordinates and convert them to atomic units.
2. Assign each center a nuclear charge, which differs from $Z$ for ghost
   atoms and ECP atoms.
3. Count electrons and split them by spin, and refuse a combination that
   has no solution.
4. Compute the nuclear repulsion and the frozen-core count.

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

## Coordinates and units {#molecule-coordinates-units}

Inside the code every length is in Bohr and every energy in Hartree. In
these units the electron mass, the elementary charge, $\hbar$ and
$4\pi\varepsilon_0$ are 1, so the Coulomb energy of two unit charges a distance $r$ apart is $1/r$ with no
prefactor. Every integral in Chapter @sec:integrals is written this way.

Input geometries are in Ångström, so one conversion sits at the boundary.
The XYZ parser multiplies by one constant,
`ferric-core/src/units.rs::ANGSTROM_TO_BOHR`, the reciprocal of the Bohr
radius $0.52917721092$ Å (CODATA 2010, the value PySCF uses). These are the
three definitions in `units.rs`, with their doc comments replaced by short
ones:

```rust
pub const BOHR_RADIUS_ANGSTROM: f64 = 0.529_177_210_92;   // a_0 in Å
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / BOHR_RADIUS_ANGSTROM;   // 1 Å ≈ 1.889 Bohr
pub const BOHR_TO_ANGSTROM: f64 = BOHR_RADIUS_ANGSTROM;         // 1 Bohr in Å
```

Why one constant and not a literal in each crate? Every length given in Å,
whether a coordinate, an attenuation radius or a cavity radius, has to land
in the same frame as the geometry. PySCF and mctc-lib (DFT-D3) use this
value, and most of ferric's validation references are generated with them.
Every crate converts through this constant, and a test fails on any other
Å–Bohr literal in the source.

The same constant converts an inverse length. A range-separation parameter
$\omega$ given in Å$^{-1}$ times `BOHR_TO_ANGSTROM` is in Bohr$^{-1}$, since
$\omega\,[\text{Å}^{-1}] \times a_0\,[\text{Å/Bohr}] = \omega\,[\text{Bohr}^{-1}]$.
The notation chapter, sections @sec:notation-attenuated-coulomb-operators and @sec:notation-units, lists which surface takes which unit.

An XYZ file has an atom count, one comment line and one line per atom with
four fields: a symbol and three coordinates. The parser errors on a short
line, an unknown symbol or a non-numeric coordinate. The element table covers
$Z = 1$ to $54$ (H to Xe).

## Ghost atoms and effective charge {#molecule-ghost-atoms-effective-charge}

A counterpoise correction asks how much a fragment's energy drops when it
can borrow its partner's basis functions. To compute that, you place the
partner's basis functions in space without the partner's nucleus or
electrons. That center is a ghost atom.

In an XYZ file a symbol prefixed with `@` is a ghost atom. It carries the
full basis of its element and no nuclear charge and no electrons. Its atomic
number stays the physical one, so the basis lookup by element works
unchanged. Only the charge the electrons see changes:

$$ Z^{\mathrm{eff}}_A = \begin{cases} 0 & \text{ghost} \\ Z_A - n^{\mathrm{ECP}}_A & \text{otherwise} \end{cases} \label{eq:molecule-1} $$

The second case covers effective core potentials. An ECP replaces the
$n^{\mathrm{ECP}}_A$ innermost electrons of a heavy atom by a potential, so
the valence electrons see the nucleus screened by those core electrons:
charge $Z_A - n^{\mathrm{ECP}}_A$. For an all-electron atom
$n^{\mathrm{ECP}}_A = 0$ and $Z^{\mathrm{eff}}_A = Z_A$.

The code is (@eq:molecule-1) line for line, in
`ferric-core/src/mol.rs::Atom::effective_z`:

```rust
pub fn effective_z(&self) -> i32 {
    if self.ghost {
        0                            // ghost: basis functions only
    } else {
        self.z - self.n_core_ecp     // Z_A - n_A^ECP (= Z_A without an ECP)
    }
}
```

`apply_ecp` sets $n^{\mathrm{ECP}}_A$ from the ECP definitions of the basis
set, and skips ghosts. It is the single point where an ECP enters the
molecule. `nelec()` and `nuclear_repulsion()` both read `effective_z`, so
both see the ECP once `apply_ecp` has run.

## Electrons and spin {#molecule-electrons-spin}

Each real atom brings $Z^{\mathrm{eff}}_A$ electrons, and a net charge $q$
removes $q$ of them. Split the total into $N_\alpha$ spin-up and $N_\beta$
spin-down electrons. Two conditions fix the split: the counts add to $N$,
and the multiplicity counts the unpaired electrons, $M - 1 = 2S = N_\alpha -
N_\beta$:

$$ N_\alpha + N_\beta = N, \qquad N_\alpha - N_\beta = M - 1 . $$

Add the two equations and divide by 2 to get $N_\alpha$. Subtract the second
from the first to get $N_\beta$:

$$ N = \sum_A Z^{\mathrm{eff}}_A - q, \qquad N_\alpha = \frac{N + M - 1}{2}, \qquad N_\beta = N - N_\alpha . \label{eq:molecule-2} $$

Check it on water: $N = 8 + 1 + 1 = 10$ and $M = 1$ give
$N_\alpha = N_\beta = 5$. The cation $\mathrm{H_2O^+}$ has $N = 9$, and $M = 1$
gives $N_\alpha = 9/2$, which is not an electron count. A doublet, $M = 2$,
gives $N_\alpha = 5$ and $N_\beta = 4$.

So (@eq:molecule-2) has an integer solution only when $N + M - 1$ is even, which means an
odd electron count needs an even multiplicity and the reverse. It also needs
$N_\beta \ge 0$, so $M - 1$ may not exceed $N$. The construction `from_atoms`
checks both conditions in
`ferric-core/src/mol.rs::validate_electron_multiplicity_parity`:

```rust
let two_s = multiplicity as i64 - 1;   // M - 1 = 2S = N_alpha - N_beta
let numerator = nelec as i64 + two_s;  // 2 N_alpha = N + M - 1
// fail if N_alpha < 0, N_alpha is not an integer, or N_beta < 0 (M - 1 > N)
if numerator < 0 || numerator % 2 != 0 || numerator > 2 * nelec as i64 {
    return Err(FerricError::General(format!(
        // ...
    )));
}
```

The text of the error message has been removed. The message names the three
numbers. Without the check, an inconsistent
molecule would fail later in the SCF loop with a message that does not
mention the molecule.

A restricted Hartree–Fock reference needs an even $N$ and $M = 1$. The CLI
kind `rhf` refuses $M > 1$ and points to `uhf` or `rohf` (validation page,
"Open shells in the CLI").

## Nuclear repulsion {#molecule-nuclear-repulsion}

The Born–Oppenheimer energy includes the Coulomb repulsion of the fixed
nuclei. It does not depend on the electrons, so the code computes it once per
geometry:

$$ E_{\mathrm{nn}} = \sum_{A<B} \frac{Z^{\mathrm{eff}}_A Z^{\mathrm{eff}}_B}{|\mathbf R_A - \mathbf R_B|} . \label{eq:molecule-3} $$

The restriction $A<B$ counts each pair once. Two limits follow from (@eq:molecule-1).
A ghost has $Z^{\mathrm{eff}} = 0$, so every term that contains it vanishes and
adding a ghost does not change $E_{\mathrm{nn}}$. For an ECP atom the charge is
that of the screened core, which is the charge the valence electrons see in
the one-electron Hamiltonian, so the electronic and nuclear terms use the same
picture of the atom.

`ferric-core/src/mol.rs::Molecule::nuclear_repulsion` is the double loop of
(@eq:molecule-3):

```rust
for i in 0..self.atoms.len() {
    if self.atoms[i].ghost {
        continue;                                 // Z_eff = 0: the whole row vanishes
    }
    for j in (i + 1)..self.atoms.len() {          // A < B: each pair once
        if self.atoms[j].ghost {
            continue;
        }
        let a = &self.atoms[i];
        let b = &self.atoms[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dz = a.zpos - b.zpos;
        let r = (dx * dx + dy * dy + dz * dz).sqrt();   // |R_A - R_B| in Bohr
        v += (a.effective_z() as f64) * (b.effective_z() as f64) / r;
    }
}
```

Only the accumulator's declaration and the return have been removed.
Skipping ghosts gives the same sum as multiplying by $Z^{\mathrm{eff}} = 0$, and
avoids a division by zero when a ghost sits on top of a real atom.

## Frozen core {#molecule-frozen-core}

Core electrons sit close to the nucleus and barely change when bonds form.
Their correlation energy is large but nearly the same in a molecule and in
its fragments, so it cancels in reaction energies. A frozen-core
calculation leaves the core orbitals out of the correlation treatment, which
saves time, and the standard correlation-consistent basis sets are built to
describe valence correlation only.

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

The table is a convention, not a derived quantity. Each row adds the shells
that become core at that element: 1s at Li, 2s2p at Na, 3s3p at K. The
$(n-1)$d shell stays correlated across its own transition series and becomes
core from the following p-block on: 3d is valence for Sc–Zn and core from Ga,
which is why the count jumps by 5 between Zn and Ga. The d electrons of a
transition metal take part in its bonding, so freezing them would discard
correlation that does not cancel.

The molecule's count is the sum over real atoms, minus
$n^{\mathrm{ECP}}_A/2$ for each ECP atom, since the potential has already
removed those orbitals. From `ferric-core/src/mol.rs::Molecule::auto_frozen_core`:

```rust
self.atoms
    .iter()
    .filter(|a| !a.ghost)                         // ghosts have no core
    .map(|a| {
        // orbitals in the table, minus the n_ECP/2 the potential removed
        core_orbitals(a.z).saturating_sub((a.n_core_ecp.max(0) / 2) as usize)
    })
    .sum()
```

The source's comment inside `map` is replaced by the one shown.

For iodine in def2-ECP the table gives 23 and the ECP removes 28 electrons,
so $23 - 28/2 = 9$ orbitals (4s4p4d) are frozen. `saturating_sub` returns 0
for a large-core ECP that removes more than the table would freeze.

The count must be read **after** `apply_ecp`, or an ECP run freezes orbitals
that are not in the calculation. Ghosts contribute zero: a ghost has no
electrons, and counting its element's core would freeze valence orbitals of
the real atoms.

Where the count enters an energy, chapters @sec:mp2 and @sec:coupled-cluster use
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
| `crates/ferric-core/src/mol.rs`, `crates/ferric-core/src/units.rs` | everything above, read against the code |
