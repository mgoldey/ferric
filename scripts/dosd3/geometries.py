#!/usr/bin/env python3
"""Idempotently create heavy-main-group geometries for the alpha-vs-CRC test.
Standard experimental structures (Angstrom)."""
from pathlib import Path

MOL_DIR = Path(__file__).resolve().parents[2] / "testdata" / "molecules"

GEOMS = {
    "sif4": ("silicon tetrafluoride Td (rSiF=1.554)", [
        ("Si", 0.000000, 0.000000, 0.000000),
        ("F",  0.897089, 0.897089, 0.897089),
        ("F", -0.897089,-0.897089, 0.897089),
        ("F", -0.897089, 0.897089,-0.897089),
        ("F",  0.897089,-0.897089,-0.897089)]),
    "geh4": ("germane Td (rGeH=1.525)", [
        ("Ge", 0.000000, 0.000000, 0.000000),
        ("H",  0.880463, 0.880463, 0.880463),
        ("H", -0.880463,-0.880463, 0.880463),
        ("H", -0.880463, 0.880463,-0.880463),
        ("H",  0.880463,-0.880463,-0.880463)]),
    "ph3": ("phosphine C3v (rPH=1.420, angle=93.5)", [
        ("P",  0.000000, 0.000000,  0.124847),
        ("H",  0.000000, 1.193609, -0.624235),
        ("H",  1.033633,-0.596805, -0.624235),
        ("H", -1.033633,-0.596805, -0.624235)]),
    "ch3br": ("methyl bromide C3v (rCBr=1.933, rCH=1.086)", [
        ("Br", 0.000000, 0.000000,  1.063000),
        ("C",  0.000000, 0.000000, -0.870000),
        ("H",  1.025000, 0.000000, -1.210000),
        ("H", -0.512500, 0.887700, -1.210000),
        ("H", -0.512500,-0.887700, -1.210000)]),
    "br2": ("bromine (r=2.281)", [
        ("Br", 0.0, 0.0, 0.000000),
        ("Br", 0.0, 0.0, 2.281000)]),
}


def xyz_text(comment, atoms):
    lines = [str(len(atoms)), comment]
    for s, x, y, z in atoms:
        lines.append(f"{s:<2} {x:>12.6f} {y:>12.6f} {z:>12.6f}")
    return "\n".join(lines) + "\n"


def main():
    for name, (comment, atoms) in GEOMS.items():
        path = MOL_DIR / f"{name}.xyz"
        text = xyz_text(comment, atoms)
        if path.exists() and path.read_text() == text:
            print(f"  ok {path.name}")
            continue
        path.write_text(text)
        print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
