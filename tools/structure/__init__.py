"""Read a structure in whatever format you have, get a `ferric.Molecule`.

## Why this module exists

ferric's Rust core accepts exactly one input format. `Molecule` (in
`crates/ferric-core/src/mol.rs`) exposes `load_xyz`, `load_xyz_with_charge`
and `parse_xyz` -- and nothing else. There is no Rust-side PDB, SDF, mol2 or
SMILES reader, and adding one would mean reimplementing, in Rust, chemistry
that `gemmi` and `rdkit` already do correctly.

So this module converts *in Python* and hands the result to the Rust parser
that already exists. Every path below ends at the same place:

    <your format> --> (symbols, coords in Angstrom) --> Molecule.from_xyz_string

That single funnel is deliberate. It means a PDB and an XYZ of the same
molecule produce a **bit-identical** `Molecule`, because they go through the
same Angstrom->Bohr conversion in the same Rust code, rather than through two
parsers that could drift apart. `test_all_formats_agree_bitwise` pins it.

## What you need installed

| format             | backend | availability                       |
|--------------------|---------|------------------------------------|
| `.xyz`             | builtin | always                             |
| `.pdb`, `.cif`     | gemmi   | always (gemmi is a core dependency)|
| `.pqr`             | builtin | always                             |
| `.gro`             | builtin | always                             |
| `.sdf`, `.mol`     | rdkit   | `pip install 'ferric[docking]'`    |
| `.mol2`            | rdkit   | `pip install 'ferric[docking]'`    |
| SMILES (string)    | rdkit   | `pip install 'ferric[docking]'`    |

A missing backend raises `MissingBackend` naming the extra to install, rather
than a bare `ModuleNotFoundError` from three frames down -- the same
convention `tools/docking/vina_dock.py` uses.

## What `charge` and `multiplicity` mean here

`charge` is the **total** molecular charge in units of the elementary charge:
the net number of electrons removed. `-1` for an anion such as a deprotonated
carboxylate, `+1` for a protonated amine, `0` for a neutral molecule. It is
*not* a sum of per-atom formal charges from a drawing -- those can be nonzero
on a neutral zwitterion and would give the wrong electron count.

`multiplicity` is the **spin multiplicity 2S+1**, not the number of unpaired
electrons and not S itself. This is the convention ferric's Rust core uses
(`Molecule.multiplicity`, `crates/ferric-core/src/mol.rs`) and the one Gaussian,
Q-Chem, ORCA and PySCF's `spin=2S` all relate to, so the values you already
know carry over:

    multiplicity  2S+1   unpaired e-   name        typical case
    1             1      0             singlet     nearly every closed-shell molecule
    2             2      1             doublet     a radical; any odd-electron species
    3             3      2             triplet     O2 ground state, many carbenes
    4             4      3             quartet     e.g. high-spin d3 metal centres

**The default is 1 (singlet), which is right for most closed-shell organic
molecules and wrong for every radical.**

ferric checks this, so half the mistakes are caught for you.
`validate_electron_multiplicity_parity` (`crates/ferric-core/src/mol.rs:65`)
runs inside `parse_xyz` -- which is the entry point everything in this module
funnels into -- and requires `n_alpha = (nelec + multiplicity - 1) / 2` to be a
non-negative integer. Two ways to fail it:

- **Wrong parity.** `nelec` and `multiplicity` must be of *opposite* parity: an
  odd electron count demands an even multiplicity (doublet, quartet) and vice
  versa. A radical left at the default multiplicity 1 is arithmetically
  impossible and raises, rather than computing something meaningless.
- **More unpaired electrons than electrons.** `multiplicity - 1 > nelec` also
  raises.

**What the check cannot catch is the case that matters most.** An even-electron
species whose true ground state is a **triplet** -- O2 is the textbook example,
and many carbenes and transition-metal complexes follow it -- passes parity at
multiplicity 1 without complaint, and the SCF converges neatly to a state that
does not exist. No format records spin, and no validator can infer it. That one
is on you.

## Neither is guessed from the file

Every reader takes both explicitly. Formats differ wildly in how (and whether)
they record them: PDB records neither, SDF has a charge block that counts only
*formal* charges, and a SMILES string carries formal charges but says nothing
about spin state. **No common structure format records multiplicity at all**,
which is precisely why it must be an argument rather than something inferred.

Guessing charge would silently produce the wrong number of electrons, and an
SCF on the wrong charge converges happily to a meaningless answer.

The one exception is `from_smiles`, which reads the formal charge off the
RDKit molecule *because the string states it explicitly* -- and it still lets
you override, and still never guesses multiplicity.

## SMILES needs a geometry, and this gives you a crude one

`from_smiles` runs ETKDG embedding plus an MMFF cleanup. That is a *starting
geometry*, not an optimized one -- tier 2 of the cost hierarchy in
`tools/campaign/hierarchy.py`. Feeding it straight to DFT wastes the DFT. Run
it through xtb first (`tools/campaign/xtb_engine.py`); see
`wiki/golden-path-iteration-1.md` for the measured tier costs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MissingBackend",
    "StructureError",
    "Structure",
    "read",
    "from_smiles",
    "SUPPORTED_SUFFIXES",
    "element_from_pdb_atom_name",
]


class StructureError(ValueError):
    """A structure file could not be turned into a molecule."""


class MissingBackend(ImportError):
    """An optional reader backend is not installed.

    Carries the pip extra to install, because the bare `ModuleNotFoundError`
    that would otherwise surface names the transitive import, not the thing
    the user has to do about it.
    """

    def __init__(self, fmt: str, module: str, extra: str) -> None:
        super().__init__(
            f"reading {fmt} needs the `{module}` package, which is not installed. "
            f"Install it with:  pip install 'ferric[{extra}]'"
        )
        self.fmt = fmt
        self.module = module
        self.extra = extra


# Suffix -> the reader key. Lowercase, leading dot.
SUPPORTED_SUFFIXES: dict[str, str] = {
    ".xyz": "xyz",
    ".pdb": "pdb",
    ".cif": "pdb",
    ".mmcif": "pdb",
    ".ent": "pdb",
    ".pqr": "pqr",
    ".gro": "gro",
    ".sdf": "sdf",
    ".mol": "sdf",
    ".mol2": "mol2",
}


@dataclass(frozen=True)
class Structure:
    """Element symbols plus Angstrom coordinates -- the common currency.

    This is the intermediate every reader produces and the single thing that
    gets handed to Rust. Kept as a named type rather than a bare tuple so that
    `n_atoms` mismatches are caught at construction instead of surfacing as a
    confusing XYZ parse error later.
    """

    symbols: tuple[str, ...]
    coords: tuple[tuple[float, float, float], ...]
    #: Total molecular charge (net electrons removed): -1 anion, +1 cation.
    charge: int = 0
    #: Spin multiplicity 2S+1 -- 1 singlet, 2 doublet, 3 triplet. NOT the
    #: number of unpaired electrons (that is `multiplicity - 1`).
    multiplicity: int = 1
    source: str = "<memory>"

    def __post_init__(self) -> None:
        if len(self.symbols) != len(self.coords):
            raise StructureError(
                f"{self.source}: {len(self.symbols)} symbols but "
                f"{len(self.coords)} coordinate rows"
            )
        if not self.symbols:
            raise StructureError(f"{self.source}: no atoms found")
        for i, row in enumerate(self.coords):
            if len(row) != 3:
                raise StructureError(
                    f"{self.source}: atom {i} has {len(row)} coordinates, expected 3"
                )
            for v in row:
                if v != v or v in (float("inf"), float("-inf")):
                    raise StructureError(
                        f"{self.source}: atom {i} has a non-finite coordinate"
                    )
        if self.multiplicity < 1:
            raise StructureError(
                f"{self.source}: multiplicity is the spin multiplicity 2S+1 and "
                f"must be >= 1 (1 = singlet, 2 = doublet, 3 = triplet); got "
                f"{self.multiplicity}. If you meant the number of unpaired "
                f"electrons n, pass n + 1."
            )

    def to_xyz(self) -> str:
        """Render as an XYZ-format string (Angstrom), ready for the Rust parser.

        Uses `repr`-grade float formatting (17 significant digits) so the
        round trip through text is exact: a float64 survives `%.17g` and comes
        back bit-identical. A shorter format here would silently perturb
        coordinates.
        """
        lines = [str(len(self.symbols)), f"from {self.source}"]
        for sym, (x, y, z) in zip(self.symbols, self.coords):
            lines.append(f"{sym} {x!r} {y!r} {z!r}")
        return "\n".join(lines) + "\n"

    def to_molecule(self):
        """Build a `ferric.Molecule` via the one Rust entry point that exists.

        Deliberately routed through `Molecule.from_xyz_string` rather than
        `from_coordinates`: it is the same parser an `.xyz` file uses, so every
        format in this module converges on identical Rust code and identical
        unit handling.
        """
        import ferric

        return ferric.Molecule.from_xyz_string(
            self.to_xyz(), self.charge, self.multiplicity
        )


def _require(module: str, fmt: str, extra: str):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise MissingBackend(fmt, module, extra) from exc


# ── readers: each returns a Structure, none of them touch ferric ──


def _read_xyz(path: Path, charge: int, multiplicity: int) -> Structure:
    lines = path.read_text().splitlines()
    if not lines:
        raise StructureError(f"{path}: empty file")
    try:
        n = int(lines[0].strip())
    except ValueError as exc:
        raise StructureError(
            f"{path}: first line is not an atom count: {lines[0]!r}"
        ) from exc
    body = lines[2 : 2 + n]
    if len(body) != n:
        raise StructureError(f"{path}: header says {n} atoms, file has {len(body)}")
    symbols, coords = [], []
    for i, line in enumerate(body):
        parts = line.split()
        if len(parts) < 4:
            raise StructureError(
                f"{path}: atom {i}: expected 4 fields, got {len(parts)}"
            )
        symbols.append(parts[0])
        coords.append(tuple(float(v) for v in parts[1:4]))
    return Structure(tuple(symbols), tuple(coords), charge, multiplicity, str(path))


def _read_pdb(path: Path, charge: int, multiplicity: int) -> Structure:
    """PDB / mmCIF via gemmi.

    Takes model 1 only. A multi-model PDB (an NMR ensemble, or an MD frame
    dump) holds many geometries of the SAME molecule; silently averaging or
    concatenating them would be wrong, so this reads the first and says so in
    `Structure.source`. For a conformer set use
    `ferric.ConformerEnsemble.from_multi_xyz` instead.

    Hydrogens are NOT added. A crystallographic PDB usually has none, and an
    SCF on a dehydrogenated structure is meaningless -- but guessing
    protonation states is exactly what `pdb2pqr` exists to do properly (see
    `tools/active_site/pdb2pqr_runner.py`). This raises rather than guesses.
    """
    gemmi = _require("gemmi", "PDB/mmCIF", "docking")
    try:
        st = gemmi.read_structure(str(path))
    except Exception as exc:
        raise StructureError(f"{path}: gemmi could not read this file: {exc}") from exc
    if len(st) == 0:
        raise StructureError(f"{path}: contains no models")
    st.remove_alternative_conformations()
    symbols, coords = [], []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                symbols.append(atom.element.name)
                coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
    if not symbols:
        raise StructureError(f"{path}: model 1 contains no atoms")
    if not any(s == "H" for s in symbols):
        raise StructureError(
            f"{path}: no hydrogens. A quantum calculation on a structure with "
            f"no hydrogens is meaningless, and this reader will not invent "
            f"them -- protonation depends on pH and residue environment. Run "
            f"the structure through pdb2pqr first "
            f"(tools/active_site/pdb2pqr_runner.py), or pass a structure that "
            f"already carries hydrogens."
        )
    n_models = len(st)
    src = str(path) if n_models == 1 else f"{path} (model 1 of {n_models})"
    return Structure(tuple(symbols), tuple(coords), charge, multiplicity, src)


def _read_pqr(path: Path, charge: int, multiplicity: int) -> Structure:
    """PQR -- reusing the parser `tools/active_site` already has.

    PQR replaces the occupancy/B-factor columns with charge and radius. Those
    charges are MM point charges, not a QM charge state, so they are dropped
    here: `charge` remains the caller's explicit total. Use
    `tools.active_site.pqr_parser.parse_pqr` when you want the charges
    themselves (for QM/MM embedding).
    """
    from tools.active_site.pqr_parser import parse_pqr_atoms

    BOHR_TO_ANGSTROM = 0.52917721092
    atoms = parse_pqr_atoms(path)
    if not atoms:
        raise StructureError(f"{path}: no ATOM/HETATM records")
    symbols, coords = [], []
    for a in atoms:
        symbols.append(element_from_pdb_atom_name(a.name))
        coords.append(
            (a.x * BOHR_TO_ANGSTROM, a.y * BOHR_TO_ANGSTROM, a.z * BOHR_TO_ANGSTROM)
        )
    return Structure(tuple(symbols), tuple(coords), charge, multiplicity, str(path))


# Two-letter element symbols that appear in PDB/PQR atom names meaning the
# element itself rather than a carbon position. Kept short and explicit.
#
# `CA` is deliberately NOT here. It is genuinely ambiguous -- an alpha carbon
# in every protein residue, calcium as an ion -- and alpha carbons outnumber
# calcium ions by orders of magnitude in any real structure, so it resolves to
# carbon. `CO` (carbonyl carbon vs cobalt) and `NI` (a nitrogen vs nickel) are
# excluded for the same reason, and for the same reason they are absent from
# the Rust table.
# An ion-heavy system needs a format with a real element column.
# `crates/ferric-cli/src/config.rs::element_from_pqr_name` makes the same call
# for the same reason; these two must agree.
#: GROMACS stores coordinates in nanometres; everything else here is Angstrom.
NM_TO_ANGSTROM = 10.0

_TWO_LETTER_OK = frozenset({"CL", "BR", "ZN", "FE", "MG", "MN", "NA", "CU", "SE"})


def element_from_pdb_atom_name(name: str) -> str:
    """Element symbol from a PDB/PQR atom name (`CA`, `HB2`, `1HB`, `ZN`).

    Neither PDB's atom-name column nor PQR carries an element, so this is a
    heuristic on a naming convention: strip any leading digit (PDB puts one on
    some hydrogens, `1HB`), take the leading alphabetic run, and accept it as a
    two-letter element only when it is exactly one of `_TWO_LETTER_OK`.
    Otherwise the element is the first letter.

    The comparison is case-insensitive and the returned symbol is
    `Xx`-capitalised. That matters: an earlier version compared an already
    `Xx`-cased string against an all-uppercase set, so the two-letter branch
    was unreachable and `CL`/`ZN`/`NA` were silently read as carbon, `Z` and
    nitrogen -- wrong nuclear charges, not merely wrong labels.
    """
    raw = name.strip().lstrip("0123456789")
    alpha = "".join(itertools.takewhile(str.isalpha, raw))
    if not alpha:
        raise StructureError(f"atom name {name!r} carries no element letters")
    if alpha[:2].upper() in _TWO_LETTER_OK:
        return alpha[0].upper() + alpha[1].lower()
    return alpha[0].upper()


def _is_gro_box(line: str) -> bool:
    """A GRO frame ends with 3 (rectangular) or 9 (triclinic) numeric values."""
    fields = line.split()
    if len(fields) not in (3, 9):
        return False
    try:
        for f in fields:
            float(f)
    except ValueError:
        return False
    return True


def _gro_coord_width(path: Path, line: str) -> int:
    """Infer the coordinate field width from the decimal-point spacing.

    GROMACS writes `%(n+5).nf` per coordinate, so three decimals gives the
    common width 8 but four gives 9 and the `y`/`z` fields shift. Hardcoding 8
    reads a four-decimal file as garbage or rejects it outright.

    The width is the distance between consecutive decimal points, which is how
    OpenMM's own `GromacsGroFile` does it. (OpenMM is inconsistent here: its
    parser infers the width exactly this way, while its `_is_gro_coord`
    line-detector hardcodes width-8 column offsets and therefore REJECTS the
    four-decimal files its parser could read. We follow the parser.)
    """
    try:
        first = line.index(".", 20)
        second = line.index(".", first + 1)
    except ValueError as exc:
        raise StructureError(
            f"{path}: first atom line has no decimal-point coordinates after "
            f"column 20, so the field width cannot be determined: {line!r}"
        ) from exc
    width = second - first
    if width < 4 or width > 20:
        raise StructureError(
            f"{path}: implausible GRO coordinate width {width} inferred from "
            f"decimal spacing in {line!r}"
        )
    return width


def _read_gro(path: Path, charge: int, multiplicity: int) -> Structure:
    """GROMACS `.gro` -- fixed-column, nanometres, no element column.

    Three things make this its own reader rather than a whitespace split:

    1. **Fixed columns are mandatory, not a convention.** The format is
       `%5d%-5s%5s%5d%8.3f%8.3f%8.3f`. A residue name and atom name that both
       fill their five columns run together (`12SOLVENTOW1`), and a five-digit
       atom index touching an eight-wide coordinate does the same. Splitting on
       whitespace works on hand-written examples and silently mis-parses real
       trajectory output.
    2. **Coordinates are in nanometres.** Everything else here is Angstrom, so
       this is the one reader that scales by 10.
    3. **Only the first frame is read.** A `.gro` can hold a trajectory; those
       are frames of the SAME molecule, so concatenating them would invent
       atoms and averaging them would invent a geometry. This reads frame 1 and
       says so in `Structure.source`, matching `_read_pdb`'s model-1 rule.

    Velocities (optional columns 45-68) are ignored: a `Structure` is a
    geometry.

    Elements come from `element_from_pdb_atom_name`, because GRO carries no
    element column either and GROMACS atom names follow the same convention.
    Cross-checked against `openmm.app.GromacsGroFile` in the tests -- an
    independent reader, not a second reading of the spec by the same author.
    """
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise StructureError(
            f"{path}: a GRO file needs at least a title, a count and a box line"
        )
    try:
        n = int(lines[1].strip())
    except ValueError as exc:
        raise StructureError(
            f"{path}: line 2 is not an atom count: {lines[1]!r}"
        ) from exc
    if n <= 0:
        raise StructureError(f"{path}: atom count is {n}")
    body = lines[2 : 2 + n]
    if len(body) != n:
        raise StructureError(
            f"{path}: header says {n} atoms, file has {len(body)} atom lines"
        )
    symbols, coords = [], []
    width = _gro_coord_width(path, body[0])
    need = 20 + 3 * width
    for i, line in enumerate(body):
        if len(line) < need:
            raise StructureError(
                f"{path}:{i + 3}: atom line is {len(line)} characters, need at "
                f"least {need} for name and {width}-wide coordinates: {line!r}"
            )
        name = line[10:15].strip()
        if not name:
            raise StructureError(f"{path}:{i + 3}: no atom name in columns 11-15")
        fields = [line[20 + k * width : 20 + (k + 1) * width] for k in range(3)]
        try:
            x, y, z = (float(f) for f in fields)
        except ValueError as exc:
            raise StructureError(
                f"{path}:{i + 3}: could not read nm coordinates from {line[20:need]!r}"
            ) from exc
        symbols.append(element_from_pdb_atom_name(name))
        coords.append((x * NM_TO_ANGSTROM, y * NM_TO_ANGSTROM, z * NM_TO_ANGSTROM))
    if len(lines) <= 2 + n or not _is_gro_box(lines[2 + n]):
        raise StructureError(
            f"{path}: no box line after the {n} atom records. A GRO file ends "
            f"each frame with 3 or 9 numeric box vectors; without it the atom "
            f"count and the frame boundaries cannot be trusted."
        )
    n_frames = 1 + max(0, (len(lines) - (n + 3)) // (n + 3))
    src = str(path) if n_frames == 1 else f"{path} (frame 1 of {n_frames})"
    return Structure(tuple(symbols), tuple(coords), charge, multiplicity, src)


def _read_rdkit(path: Path, charge: int, multiplicity: int, fmt: str) -> Structure:
    """SDF / MOL / MOL2 via rdkit, preserving the file's 3-D coordinates."""
    _require("rdkit", fmt, "docking")
    from rdkit import Chem

    if fmt == "mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False)
    else:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mols = [m for m in supplier if m is not None]
        if len(mols) > 1:
            raise StructureError(
                f"{path}: contains {len(mols)} molecules. This reader returns one "
                f"structure; pick a record explicitly with rdkit, or use "
                f"ferric.ConformerEnsemble for a conformer set."
            )
        mol = mols[0] if mols else None
    if mol is None:
        raise StructureError(f"{path}: rdkit could not parse this file")
    if mol.GetNumConformers() == 0:
        raise StructureError(
            f"{path}: has no 3-D coordinates. Use `from_smiles` to generate a "
            f"geometry from connectivity."
        )
    return _from_rdkit_mol(mol, charge, multiplicity, str(path))


def _from_rdkit_mol(
    mol, charge: int | None, multiplicity: int, source: str
) -> Structure:
    conf = mol.GetConformer()
    symbols, coords = [], []
    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        symbols.append(atom.GetSymbol())
        coords.append((p.x, p.y, p.z))
    if charge is None:
        from rdkit import Chem

        charge = Chem.GetFormalCharge(mol)
    return Structure(tuple(symbols), tuple(coords), charge, multiplicity, source)


_READERS = {
    "xyz": _read_xyz,
    "pdb": _read_pdb,
    "pqr": _read_pqr,
    "gro": _read_gro,
}


def read(
    path: str | Path,
    charge: int = 0,
    multiplicity: int = 1,
    fmt: str | None = None,
):
    """Read any supported structure file and return a `ferric.Molecule`.

    `charge` is the total molecular charge (-1 anion, +1 cation, 0 neutral).
    `multiplicity` is the spin multiplicity **2S+1**: 1 = singlet (closed
    shell, the default), 2 = doublet (one unpaired electron, i.e. any radical),
    3 = triplet. It is not the count of unpaired electrons -- that is
    `multiplicity - 1`.

    Neither is inferred from the file: no common structure format records
    multiplicity at all. See the module docstring.

    `fmt` overrides suffix detection (for a PDB named `.txt`, say).

    Use `read_structure` instead if you want the intermediate `Structure`
    (symbols + Angstrom coords) without importing the compiled extension.
    """
    return read_structure(path, charge, multiplicity, fmt).to_molecule()


def read_structure(
    path: str | Path,
    charge: int = 0,
    multiplicity: int = 1,
    fmt: str | None = None,
) -> Structure:
    """Like `read`, but stops at the `Structure` and never imports `ferric`.

    Useful in tests and in tooling that only needs coordinates, and it keeps
    the format layer testable without a compiled extension present.
    """
    path = Path(path)
    if fmt is None:
        suffix = path.suffix.lower()
        if suffix == ".gz":
            raise StructureError(
                f"{path}: gzipped input is not supported; decompress it first"
            )
        if suffix not in SUPPORTED_SUFFIXES:
            known = ", ".join(sorted(SUPPORTED_SUFFIXES))
            raise StructureError(
                f"{path}: unrecognised suffix {suffix!r}. Known: {known}. "
                f"Pass fmt= to override detection."
            )
        key = SUPPORTED_SUFFIXES[suffix]
    else:
        key = fmt.lower().lstrip(".")
        if key in SUPPORTED_SUFFIXES:
            key = SUPPORTED_SUFFIXES[key]
        elif key not in ("xyz", "pdb", "pqr", "gro", "sdf", "mol2"):
            raise StructureError(f"unknown fmt={fmt!r}")
    if not path.exists():
        raise StructureError(f"{path}: no such file")

    if key in _READERS:
        return _READERS[key](path, charge, multiplicity)
    return _read_rdkit(path, charge, multiplicity, key)


def from_smiles(
    smiles: str,
    charge: int | None = None,
    multiplicity: int = 1,
    seed: int = 0xF00D,
    optimize: bool = True,
):
    """Build a `ferric.Molecule` from a SMILES string via RDKit ETKDG.

    `charge=None` (the default) reads the total formal charge off the parsed
    molecule, which a SMILES string states explicitly -- this is a read, not a
    guess. Pass an integer to override.

    `multiplicity` is the spin multiplicity **2S+1** (1 = singlet, 2 = doublet,
    3 = triplet), not the number of unpaired electrons, and it is never
    inferred: SMILES encodes connectivity and formal charge but carries no spin
    state. A radical written as `[CH3]` still arrives here as multiplicity 1
    unless you pass `multiplicity=2`; because it has an odd electron count,
    ferric's parity check rejects it rather than computing the wrong thing
    silently. An even-electron triplet gets no such protection.

    The geometry is ETKDG + (by default) an MMFF cleanup. That is a tier-2
    starting structure, NOT an optimized one; see the module docstring.

    `seed` is fixed so the same SMILES gives the same geometry on every call.
    ETKDG is stochastic, and an unseeded embedding would make every downstream
    energy irreproducible.
    """
    _require("rdkit", "SMILES", "docking")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise StructureError(f"rdkit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise StructureError(
            f"ETKDG could not embed a 3-D geometry for {smiles!r}. Strained "
            f"macrocycles and over-constrained ring systems are the usual "
            f"cause; try a different seed= or supply a structure file."
        )
    if optimize:
        # A failed MMFF cleanup is not fatal: the ETKDG geometry is still a
        # valid starting point, and silently returning an unrelaxed structure
        # is better than failing outright -- but the caller is not told, so
        # this must never be treated as an optimized geometry either way.
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
    return _from_rdkit_mol(mol, charge, multiplicity, f"SMILES {smiles}").to_molecule()
