import numpy as np
from math import cos, sin, radians

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

CC_BOND = 1.54      # Angstroms
CH_BOND = 1.09      # Angstroms
TETRA_ANGLE = 109.5


# ------------------------------------------------------------
# Vector helpers
# ------------------------------------------------------------

def normalize(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def rotation_matrix(axis, angle_deg):
    """
    Rodrigues rotation formula.
    """
    axis = normalize(axis)
    angle = radians(angle_deg)

    x, y, z = axis
    c = cos(angle)
    s = sin(angle)
    C = 1 - c

    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
    ])


# ------------------------------------------------------------
# Build carbon backbone
# ------------------------------------------------------------

def build_alkane_backbone(n_carbons):
    """
    Build carbon coordinates in a zig-zag tetrahedral arrangement.
    """

    coords = []

    # First carbon
    coords.append(np.array([0.0, 0.0, 0.0]))

    # Second carbon along x-axis
    coords.append(np.array([CC_BOND, 0.0, 0.0]))

    # Initial bond direction
    prev_dir = normalize(coords[1] - coords[0])

    # Alternating bend axis
    axis = np.array([0.0, 0.0, 1.0])

    # Supplementary angle creates tetrahedral zig-zag
    bend = 180.0 - TETRA_ANGLE

    sign = 1

    for i in range(2, n_carbons):

        R = rotation_matrix(axis, sign * bend)
        new_dir = normalize(R @ prev_dir)

        new_pos = coords[-1] + CC_BOND * new_dir
        coords.append(new_pos)

        prev_dir = new_dir

        # Alternate zig-zag direction
        sign *= -1

    return coords


# ------------------------------------------------------------
# Generate hydrogens
# ------------------------------------------------------------

def perpendicular_vector(v):
    """
    Find arbitrary perpendicular vector.
    """
    v = normalize(v)

    if abs(v[0]) < 0.9:
        other = np.array([1.0, 0.0, 0.0])
    else:
        other = np.array([0.0, 1.0, 0.0])

    perp = np.cross(v, other)
    return normalize(perp)



def generate_hydrogens(carbons):
    """
    Approximate tetrahedral hydrogen placement.
    """

    atoms = []

    for i, c in enumerate(carbons):
        atoms.append(("C", c))

    for i, c in enumerate(carbons):

        neighbors = []

        if i > 0:
            neighbors.append(carbons[i - 1] - c)

        if i < len(carbons) - 1:
            neighbors.append(carbons[i + 1] - c)

        neighbors = [normalize(v) for v in neighbors]

        if len(neighbors) == 1:
            # Terminal carbon: CH3
            bond = neighbors[0]

            perp1 = perpendicular_vector(bond)
            perp2 = normalize(np.cross(bond, perp1))

            theta = radians(109.5)

            for angle in [0, 120, 240]:
                phi = radians(angle)

                direction = (
                    -cos(theta) * bond
                    + sin(theta) * (
                        cos(phi) * perp1 +
                        sin(phi) * perp2
                    )
                )

                h = c + CH_BOND * normalize(direction)
                atoms.append(("H", h))

        elif len(neighbors) == 2:
            # Internal carbon: CH2
            b1 = neighbors[0]
            b2 = neighbors[1]

            bisector = normalize(-(b1 + b2))
            perp = normalize(np.cross(b1, b2))

            angle = 35.25

            R1 = rotation_matrix(perp, angle)
            R2 = rotation_matrix(perp, -angle)

            h1 = c + CH_BOND * normalize(R1 @ bisector)
            h2 = c + CH_BOND * normalize(R2 @ bisector)

            atoms.append(("H", h1))
            atoms.append(("H", h2))

    return atoms


# ------------------------------------------------------------
# XYZ export
# ------------------------------------------------------------

def write_xyz(filename, atoms):
    with open(filename, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write("Alkane generated with tetrahedral geometry\n")

        for element, pos in atoms:
            x, y, z = pos
            f.write(f"{element:2s} {x:10.4f} {y:10.4f} {z:10.4f}\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(n):
    carbons = build_alkane_backbone(n)
    atoms = generate_hydrogens(carbons)

    outfile = f"alkane_C{n}.xyz"
    write_xyz(outfile, atoms)

    print(f"Wrote {outfile}")

if __name__ == "__main__":
    from fire import Fire
    Fire(main)
