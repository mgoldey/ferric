#!/usr/bin/env python
from fire import Fire
def generate_alkane_xyz(n_carbon):
    """Generate XYZ coordinates for a linear alkane C_n H_{2n+2}."""
    cc_dist = 1.54 # Angstrom
    ch_dist = 1.09

    lines = [f"{3*n_carbon + 2}", f"Linear Alkane C{n_carbon}"]
    for i in range(n_carbon):
        x = i * cc_dist
        lines.append(f"C {x:10.6f} 0.000000 0.000000")
        # Add hydrogens
        lines.append(f"H {x:10.6f} {ch_dist:10.6f} 0.000000")
        lines.append(f"H {x:10.6f} {-ch_dist:10.6f} 0.000000")

    # End caps
    lines.append(f"H {-ch_dist:10.6f} 0.000000 0.000000")
    lines.append(f"H {n_carbon * cc_dist:10.6f} 0.000000 0.000000")

    return "\n".join(lines)
def main(n):
    with open(f"alkane_{n}.xyz",'w') as o:
        o.write(generate_alkane_xyz(n))

if __name__=="__main__":
    Fire(main)
