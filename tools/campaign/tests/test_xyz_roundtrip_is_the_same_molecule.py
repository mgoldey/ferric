"""A geometry written out and read back must be the SAME MOLECULE.

This is the writer half of the audit that found the united-atom docking bug.
That one was a READER problem -- a PDBQT pose came back with its hydrogens
merged, and the quantum tiers scored a 14-atom fragment of a 21-atom molecule.
The same class exists on the way out: a writer that drops atoms produces a file
that reads back as something else, with no error anywhere.

AUDITED 2026-09-21 across every geometry writer in `tools/`. Two carried it:

  * `xtb_engine._write_xyz` wrote `len(symbols)` as the header while the body
    came from `zip(symbols, coords)`, which stops at the shorter one.
  * `morph.EmbeddedAnalogue.write_xyz` did the same.

And the protection was UNEVEN, exactly as it was for the tiers:
`tools.structure.read_structure` REFUSES a header/body mismatch, while
`xtb_engine._read_xyz` sliced `lines[2:2+n]` and silently returned however many
rows it found. Which reader a caller happens to use should not decide whether a
malformed file is caught.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import pytest


def _ethanol():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = [
        (
            conf.GetAtomPosition(i).x,
            conf.GetAtomPosition(i).y,
            conf.GetAtomPosition(i).z,
        )
        for i in range(mol.GetNumAtoms())
    ]
    return symbols, coords


def test_write_then_read_preserves_the_molecule():
    """The property that matters: formula, ORDER, and coordinates survive."""
    pytest.importorskip("rdkit")
    from tools.campaign.xtb_engine import _read_xyz, _write_xyz

    symbols, coords = _ethanol()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a.xyz"
        _write_xyz(path, symbols, coords)
        got_sym, got_crd = _read_xyz(path)

    assert Counter(got_sym) == Counter(symbols), "formula changed"
    assert got_sym == symbols, (
        "atom ORDER changed -- a reordering keeps the formula and pairs every "
        "coordinate with the wrong element"
    )
    worst = max(
        abs(a - b) for want, got in zip(coords, got_crd) for a, b in zip(want, got)
    )
    assert worst < 1e-7, f"coordinates moved by {worst:.2e} A (%14.8f gives ~1e-8)"


def test_a_symbol_coordinate_mismatch_is_REFUSED_by_the_writer():
    """`zip` truncates, so the header would disagree with the body.

    MEASURED before the fix: 22 in the header over 21 atom lines. That file
    reads back as a different molecule, and only one of the two readers in this
    repo noticed.
    """
    pytest.importorskip("rdkit")
    from tools.campaign.xtb_engine import _write_xyz

    symbols, coords = _ethanol()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a.xyz"
        with pytest.raises(ValueError, match="per-atom and must match"):
            _write_xyz(path, symbols + ["H"], coords)
        with pytest.raises(ValueError, match="per-atom and must match"):
            _write_xyz(path, symbols, coords[:-1])


def test_a_truncated_file_is_REFUSED_by_the_reader_too():
    """Both ends, because either alone leaves the hole open.

    A file can arrive truncated from outside this repo. `read_structure`
    already refused this; `_read_xyz` returned a short molecule, so the caller's
    choice of reader decided whether the corruption was caught.
    """
    from tools.campaign.xtb_engine import _read_xyz

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "short.xyz"
        body = "\n".join(f"C 0.0 0.0 {i}.0" for i in range(9))
        path.write_text(f"10\ncomment\n{body}\n")
        with pytest.raises(ValueError, match="header says 10 atoms"):
            _read_xyz(path)


def test_a_file_with_SURPLUS_rows_is_REFUSED_by_the_reader():
    """The other half of the mismatch, which the truncation check cannot see.

    `_read_xyz` sliced `lines[2 : 2 + n]`, so a header of 9 over a body of 10
    was capped to 9 and returned as a valid molecule with the last atom
    silently DROPPED -- the same class of bug as the united-atom pose
    (a plausible number for a molecule nobody asked about). Truncation raised;
    surplus did not, so testing one direction proved nothing about the other.
    """
    from tools.campaign.xtb_engine import _read_xyz

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "long.xyz"
        body = "\n".join(f"C 0.0 0.0 {i}.0" for i in range(10))
        path.write_text(f"9\ncomment\n{body}\n")
        with pytest.raises(ValueError, match="header says 9 atoms"):
            _read_xyz(path)


def test_both_readers_agree_about_a_malformed_file():
    """The uneven-protection check, stated directly.

    One consumer refusing and another accepting is how the united-atom bug hid:
    tier 4's parity check caught it while tier 3 scored it. Same shape here.
    """
    pytest.importorskip("rdkit")
    from tools.campaign.xtb_engine import _read_xyz
    from tools.structure import StructureError, read_structure

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "short.xyz"
        body = "\n".join(f"C 0.0 0.0 {i}.0" for i in range(9))
        path.write_text(f"10\ncomment\n{body}\n")

        with pytest.raises(StructureError):
            read_structure(path)
        with pytest.raises(ValueError):
            _read_xyz(path)


def test_every_xyz_writer_in_tools_guards_its_header_against_its_body():
    """A STRUCTURAL guard, so a new writer cannot reintroduce the class.

    The bug shape is specific and greppable: build an xyz header from
    `len(symbols)`, then fill the body with `zip(symbols, coords)`. `zip` stops
    at the shorter one, so any mismatch writes a file whose header disagrees
    with its contents -- and it reads back as a different molecule.

    AUDITED 2026-09-21: seven sites in `tools/` have this shape. Five were
    already safe (`Structure.__post_init__` validates, `align.py` checks
    explicitly, `tiers.py` reads through the now-formula-checked `_embedded`);
    two were not and are fixed. This finds any NEW one.

    The check is deliberately crude -- it looks for the two patterns near each
    other and requires a length comparison in the same function. A crude
    structural test that fires on a real pattern beats a precise one nobody
    writes.
    """
    import ast
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    tools = repo / "tools"
    assert tools.is_dir(), f"no {tools}"

    # Functions whose arrays are validated BEFORE they get here. Each entry is
    # a claim someone checked, not a silencer: if the upstream guard is
    # removed, the entry becomes wrong and this list is where a reviewer looks.
    KNOWN_SAFE = {
        # `Structure.__post_init__` refuses len(symbols) != len(coords), so a
        # Structure cannot exist in the broken state.
        "tools/structure/__init__.py::to_xyz",
        # Reads through `_embedded`, which now checks the geometry's FORMULA
        # against the candidate SMILES -- a stronger property than a length.
        "tools/pipeline/tiers.py::_tier4_dft_inner",
    }

    offenders = []
    for path in sorted(tools.rglob("*.py")):
        if "test" in path.parts or path.name.startswith("test_"):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - not ours to parse
            continue
        for fn in [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]:
            src = ast.get_source_segment(path.read_text(), fn) or ""
            # The shape: a header from len(<something>) AND a zip of two
            # per-atom sequences.
            has_header = "str(len(" in src or 'f"{len(' in src
            has_zip = "zip(" in src and (
                "coords" in src or "coord" in src or "crd" in src
            )
            if not (has_header and has_zip):
                continue

            # Safe only if the function compares the lengths of two sequences
            # that its own header/body actually use. A substring check on
            # "!= len(" accepted ANY comparison -- mutation-verified: replacing
            # a real guard with `len(_a) != len(_b)` over two unrelated lists
            # left the test green. Bind it to the named sequences instead.
            def _seq_name(node):
                """`x` / `self.x` / `p.x` / `tuple(x)` -> a comparable key."""
                if isinstance(node, ast.Name):
                    return node.id
                if isinstance(node, ast.Attribute):
                    return node.attr
                if isinstance(node, ast.Call) and node.args:
                    return _seq_name(node.args[0])
                return None

            # ONLY the sequences the writer itself names: the zip() that fills
            # the body. Collecting from every len() call too would be circular
            # -- a decoy `len(_a) != len(_b)` would supply its own evidence
            # (mutation-verified: that decoy survived until this was narrowed).
            used = set()
            for call in ast.walk(fn):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "zip"
                ):
                    used |= {n for n in (_seq_name(a) for a in call.args) if n}

            guarded = False
            for cmp_node in [c for c in ast.walk(fn) if isinstance(c, ast.Compare)]:
                if not all(
                    isinstance(o, ast.NotEq) or isinstance(o, ast.Eq)
                    for o in cmp_node.ops
                ):
                    continue
                lens = set()
                for op in (cmp_node.left, *cmp_node.comparators):
                    if (
                        isinstance(op, ast.Call)
                        and isinstance(op.func, ast.Name)
                        and op.func.id == "len"
                        and op.args
                    ):
                        name = _seq_name(op.args[0])
                        if name:
                            lens.add(name)
                # Two DISTINCT sequences, and at least one of them is a
                # sequence the body actually zips -- so the comparison is tied
                # to the data that fills the file, not to scratch variables.
                if len(lens) >= 2 and lens & used:
                    guarded = True
                    break

            key = f"{path.relative_to(repo)}::{fn.name}"
            if not guarded and key not in KNOWN_SAFE:
                offenders.append(key)

    assert not offenders, (
        "these build an xyz header from len(...) and fill the body with zip(), "
        "with no length check -- a mismatch writes a file whose header "
        "disagrees with its body and reads back as a DIFFERENT MOLECULE:\n  "
        + "\n  ".join(offenders)
    )
