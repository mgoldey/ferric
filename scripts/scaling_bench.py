import ferric
import time
import numpy as np

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

def run_bench(n_carbon):
    xyz = generate_alkane_xyz(n_carbon)
    mol = ferric.Molecule.from_xyz_string(xyz)
    bs  = ferric.BasisSet.bundled("sto-3g")
    
    start = time.time()
    res = ferric.run_rhf(mol, bs,     k_builder="link" )  # linear k or direct
    end = time.time()
    
    return {
        "n_carbon": n_carbon,
        "time": end - start,
        "quartets": getattr(res, "computed_quartets", 0),
        "energy": res.energy
    }

if __name__ == "__main__":
    print("Linear Scaling Benchmark: Alkanes C10 to C50")
    print("-" * 60)
    print(f"{'N(C)':<5} | {'Time (s)':<10} | {'Quartets':<12} | {'Energy (Ha)':<15}")
    print("-" * 60)
    
    sizes = list(range(1,20))+list(range(20,40,5))
    
    for n in sizes:
        try:
            res = run_bench(n)
            print(f"{n:<5} | {res['time']:10.2f} | {res['quartets']:12} | {res['energy']:15.6f}")
        except Exception as e:
            print(f"{n:<5} | Error: {e}")
