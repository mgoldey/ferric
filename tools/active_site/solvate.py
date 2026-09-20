"""Explicit TIP3P water DROPLET around a solute — not a periodic box.

## Why a droplet and not a box

The usual "solvation box" is periodic: its electrostatics are only correct
under the minimum-image convention, and ferric's MM kernel has no PBC. Cutting
a periodic box out and treating it as isolated is not an approximation of that
box, it is a different system — the surface waters were oriented for a lattice
that is no longer there.

So this builds what ferric can actually evaluate: a finite spherical shell of
explicit waters around the solute, handed to `[qmmm]` as MM point charges.

## What a droplet is and is not

**Is:** explicit first-shell structure — the hydrogen bonds an implicit model
averages away, and the specific waters that bridge a ligand to a pocket.

**Is not:** bulk solvent. A droplet has a vacuum boundary, so its outermost
waters point inward and the dielectric response saturates at the surface rather
than continuing to infinity. For bulk electrostatics use COSMO or IEF-PCM
(`ferric_scf::cosmo`, `ferric_pcm`), which are validated against PySCF; for the
first shell, use this. They answer different questions and the honest setup for
many problems is both.

## ONE DROPLET IS NOT A RESULT

MEASURED, water/STO-3G with this packer, dE against vacuum over six seeds:

    r =  6 A:  mean -1.54, sd 2.28, range [-5.35, +0.96] kcal/mol
    r =  8 A:  mean +2.14, sd 4.55, range [-4.90, +7.30] kcal/mol

**The scatter is larger than the mean, and the sign is not even fixed.** The
orientations are random, so a single droplet samples one arbitrary
configuration of a distribution several kcal/mol wide -- and a solvation shift
of a few kcal/mol is exactly what a user is usually trying to resolve.

So a single droplet answers "what does explicit first-shell structure do to my
wavefunction" (a real question) and NOT "what is the solvation energy" (it
cannot, at this noise). Either average over seeds and report the SEM, or use
COSMO/PCM for the bulk term. `dE_statistics` below does the averaging and
returns the spread with it, because a mean without its scatter is the thing
that would get quoted.

## Packing

Waters are placed on a jittered cubic lattice inside the sphere and kept only
if every atom clears `min_dist` from every solute atom and from every already
accepted water. That is a hard-sphere packing, NOT an equilibrated one: the
density lands near but below bulk, and the orientations are random rather than
hydrogen-bonded. Equilibrating them is MD, which is out of scope here — this is
a starting structure, and it says so.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

#: TIP3P internal geometry (Angstrom / degrees) and charges (e).
TIP3P_ROH = 0.9572
TIP3P_HOH_DEG = 104.52
TIP3P_Q_O = -0.834
TIP3P_Q_H = 0.417
#: TIP3P Lennard-Jones sigma on oxygen (Angstrom); hydrogens carry none.
TIP3P_R_O = 1.7683
TIP3P_R_H = 0.0

#: Bulk water number density at 300 K, molecules per cubic Angstrom.
BULK_DENSITY = 0.0334

#: Any pair involving a hydrogen uses `H_SCALE * min_dist` instead of the
#: heavy-atom `min_dist`. Hydrogens are small and hydrogen bonds are short --
#: H...O reaches ~1.8 A and H...H ~2.0 A in bulk water -- so the heavy-atom
#: cutoff applied to them rejects the first-shell contacts the droplet exists
#: to represent. 0.75 x 2.4 = 1.8 A, the hydrogen-bond H...O distance.
H_SCALE = 0.75


@dataclass(frozen=True)
class Droplet:
    """The packed result. Coordinates are ANGSTROM, charges are e."""

    symbols: tuple[str, ...]
    coords: tuple[tuple[float, float, float], ...]
    charges: tuple[float, ...]
    radii: tuple[float, ...]
    n_waters: int
    radius_angstrom: float

    @property
    def density(self) -> float:
        """Packed number density (molecules / A^3), for comparison with
        `BULK_DENSITY`. A hard-sphere pack lands BELOW bulk; how far below is
        the honest measure of how crude the packing is."""
        v = 4.0 / 3.0 * math.pi * self.radius_angstrom**3
        return self.n_waters / v if v > 0 else 0.0


def _water_at(cx: float, cy: float, cz: float, rng: random.Random):
    """One randomly oriented TIP3P water with its oxygen at (cx, cy, cz)."""
    half = math.radians(TIP3P_HOH_DEG) / 2.0
    local = [
        (0.0, 0.0, 0.0),
        (TIP3P_ROH * math.sin(half), TIP3P_ROH * math.cos(half), 0.0),
        (-TIP3P_ROH * math.sin(half), TIP3P_ROH * math.cos(half), 0.0),
    ]
    # Uniform random rotation (Shoemake): a unit quaternion from 3 uniforms.
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    s1, s2 = math.sqrt(1 - u1), math.sqrt(u1)
    q = (
        s1 * math.sin(2 * math.pi * u2),
        s1 * math.cos(2 * math.pi * u2),
        s2 * math.sin(2 * math.pi * u3),
        s2 * math.cos(2 * math.pi * u3),
    )
    x, y, z, w = q
    rot = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    out = []
    for px, py, pz in local:
        out.append(
            (
                cx + rot[0][0] * px + rot[0][1] * py + rot[0][2] * pz,
                cy + rot[1][0] * px + rot[1][1] * py + rot[1][2] * pz,
                cz + rot[2][0] * px + rot[2][1] * py + rot[2][2] * pz,
            )
        )
    return out


def solvate(
    solute_symbols,
    solute_coords_angstrom,
    radius_angstrom: float,
    *,
    min_dist_angstrom: float = 2.4,
    seed: int = 0xF00D,
    center=None,
) -> Droplet:
    """Pack TIP3P waters into a sphere of `radius_angstrom` around the solute.

    `min_dist_angstrom` is the heavy-atom (O...O, O...solute-heavy) clash
    distance. 2.4 A sits a little under the 2.8 A hydrogen-bond O...O
    separation, so first-shell waters are allowed without hard overlaps.

    HYDROGENS GET A SMALLER CUTOFF, and this is not a tuning knob -- it is what
    makes the packing physical. A hydrogen bond puts H within ~1.8 A of an
    acceptor oxygen, and H...H in bulk water reaches ~2.0 A. Applying the
    heavy-atom 2.4 A to every pair rejects exactly the contacts that define
    first-shell structure: MEASURED, a single all-atom 2.4 A cutoff packs to
    0.0182 A^-3, 54% of bulk, because two lattice-adjacent waters can present
    H...H at 1.19 A and are thrown away. Per-element cutoffs
    (`H_SCALE` x min_dist for any pair involving H) recover it.

    Deterministic for a given `seed`: a droplet is a starting structure, and an
    irreproducible starting structure makes every number downstream
    irreproducible too.
    """
    if not (radius_angstrom > 0 and math.isfinite(radius_angstrom)):
        raise ValueError(
            f"radius_angstrom must be finite and > 0, got {radius_angstrom}"
        )
    if not (min_dist_angstrom > 0 and math.isfinite(min_dist_angstrom)):
        raise ValueError(
            f"min_dist_angstrom must be finite and > 0, got {min_dist_angstrom}"
        )
    solute = [tuple(float(v) for v in c) for c in solute_coords_angstrom]
    if len(solute) != len(tuple(solute_symbols)):
        raise ValueError(
            f"{len(solute)} coordinates for {len(tuple(solute_symbols))} symbols"
        )
    if center is None:
        if not solute:
            raise ValueError("no solute atoms and no explicit center")
        cx = sum(c[0] for c in solute) / len(solute)
        cy = sum(c[1] for c in solute) / len(solute)
        cz = sum(c[2] for c in solute) / len(solute)
    else:
        cx, cy, cz = (float(v) for v in center)

    rng = random.Random(seed)
    d2_heavy = min_dist_angstrom**2
    d2_h = (H_SCALE * min_dist_angstrom) ** 2
    # Lattice pitch from bulk density, so the candidate count is right without
    # a pitch constant nobody can justify.
    pitch = BULK_DENSITY ** (-1.0 / 3.0)
    n = int(math.ceil(radius_angstrom / pitch))

    placed: list[tuple[float, float, float]] = []
    placed_is_h: list[bool] = []
    solute_is_h = tuple(str(sy).strip().upper() == "H" for sy in solute_symbols)
    syms: list[str] = []
    charges: list[float] = []
    radii: list[float] = []
    n_waters = 0

    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            for k in range(-n, n + 1):
                ox = cx + i * pitch + rng.uniform(-0.25, 0.25)
                oy = cy + j * pitch + rng.uniform(-0.25, 0.25)
                oz = cz + k * pitch + rng.uniform(-0.25, 0.25)
                if (ox - cx) ** 2 + (oy - cy) ** 2 + (
                    oz - cz
                ) ** 2 > radius_angstrom**2:
                    continue
                atoms = _water_at(ox, oy, oz, rng)
                # Water atom order from `_water_at` is (O, H, H).
                w_is_h = (False, True, True)
                clash = False
                for (ax, ay, az), a_h in zip(atoms, w_is_h):
                    for (sx, sy, sz), s_h in zip(solute, solute_is_h):
                        lim = d2_h if (a_h or s_h) else d2_heavy
                        if (ax - sx) ** 2 + (ay - sy) ** 2 + (az - sz) ** 2 < lim:
                            clash = True
                            break
                    if clash:
                        break
                    for (px, py, pz), p_h in zip(placed, placed_is_h):
                        lim = d2_h if (a_h or p_h) else d2_heavy
                        if (ax - px) ** 2 + (ay - py) ** 2 + (az - pz) ** 2 < lim:
                            clash = True
                            break
                    if clash:
                        break
                if clash:
                    continue
                placed.extend(atoms)
                placed_is_h.extend(w_is_h)
                syms.extend(("O", "H", "H"))
                charges.extend((TIP3P_Q_O, TIP3P_Q_H, TIP3P_Q_H))
                radii.extend((TIP3P_R_O, TIP3P_R_H, TIP3P_R_H))
                n_waters += 1

    return Droplet(
        symbols=tuple(syms),
        coords=tuple(placed),
        charges=tuple(charges),
        radii=tuple(radii),
        n_waters=n_waters,
        radius_angstrom=radius_angstrom,
    )


def write_pqr(
    path: str | Path,
    solute_symbols,
    solute_coords_angstrom,
    solute_charges,
    droplet: Droplet,
    *,
    solute_radii=None,
) -> int:
    """Write solute + droplet as one PQR, the input `[qmmm]` reads.

    The solute comes FIRST, so its indices are `0 .. n_solute-1` and can be
    handed to `[qmmm] qm_indices` unchanged. Returns the total atom count.
    """
    solute = [tuple(float(v) for v in c) for c in solute_coords_angstrom]
    ssym = tuple(solute_symbols)
    sq = tuple(float(q) for q in solute_charges)
    if not (len(solute) == len(ssym) == len(sq)):
        raise ValueError(
            f"solute mismatch: {len(ssym)} symbols, {len(solute)} coords, {len(sq)} charges"
        )
    srad = tuple(solute_radii) if solute_radii is not None else (0.0,) * len(ssym)

    lines = [
        "REMARK  solute + explicit TIP3P DROPLET (finite, non-periodic).",
        f"REMARK  {droplet.n_waters} waters in a {droplet.radius_angstrom:g} A sphere; "
        f"packed density {droplet.density:.4f} vs bulk {BULK_DENSITY:.4f} A^-3.",
        "REMARK  Hard-sphere packing, random orientations -- a STARTING structure,",
        "REMARK  not an equilibrated one, and a droplet has a vacuum boundary.",
    ]
    serial = 0
    for sym, (x, y, z), q, r in zip(ssym, solute, sq, srad):
        serial += 1
        lines.append(
            f"ATOM  {serial:5d}  {sym:<3s} SOL  {1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f} {q:7.4f} {r:6.4f}"
        )
    for w in range(droplet.n_waters):
        for a in range(3):
            i = 3 * w + a
            serial += 1
            x, y, z = droplet.coords[i]
            lines.append(
                f"ATOM  {serial:5d}  {droplet.symbols[i]:<3s} WAT  {w + 2:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f} {droplet.charges[i]:7.4f} {droplet.radii[i]:6.4f}"
            )
    Path(path).write_text("\n".join(lines) + "\n")
    return serial


@dataclass(frozen=True)
class DropletStatistics:
    """Mean and spread of a property over independently packed droplets."""

    mean: float
    sd: float
    sem: float
    n: int
    values: tuple[float, ...]

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.mean:+.3f} +/- {self.sem:.3f} (sd {self.sd:.3f}, n={self.n})"


def dE_statistics(
    evaluate,
    solute_symbols,
    solute_coords_angstrom,
    radius_angstrom: float,
    *,
    n_seeds: int = 8,
    seed0: int = 0xF00D,
    **solvate_kwargs,
) -> DropletStatistics:
    """Average a droplet-dependent quantity over `n_seeds` independent packs.

    `evaluate(droplet) -> float` is whatever the caller wants measured -- a QM
    energy in the droplet's field, an ESP, a property difference.

    THE SEM IS THE POINT. A single droplet's value carries several kcal/mol of
    orientation noise (see the module docstring), so a bare mean would be
    quoted as though it were converged. Returning the spread alongside makes
    the precision visible at the call site instead of in a docstring nobody
    re-reads.
    """
    import statistics

    if n_seeds < 2:
        raise ValueError(
            f"n_seeds must be >= 2 to have a spread at all, got {n_seeds}. "
            "A single droplet has no measurable precision -- that is the whole "
            "reason this function exists."
        )
    vals = []
    for i in range(n_seeds):
        d = solvate(
            solute_symbols,
            solute_coords_angstrom,
            radius_angstrom,
            seed=seed0 + i,
            **solvate_kwargs,
        )
        vals.append(float(evaluate(d)))
    sd = statistics.stdev(vals)
    return DropletStatistics(
        mean=statistics.fmean(vals),
        sd=sd,
        sem=sd / math.sqrt(len(vals)),
        n=len(vals),
        values=tuple(vals),
    )
