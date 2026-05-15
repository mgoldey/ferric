import ferric
import time

def generate_alkane_xyz(n_carbon):
    cc_dist = 1.54
    ch_dist = 1.09
    lines = [f"{3*n_carbon + 2}", f"Linear Alkane C{n_carbon}"]
    for i in range(n_carbon):
        x = i * cc_dist
        lines.append(f"C {x:10.6f} 0.000000 0.000000")
        lines.append(f"H {x:10.6f} {ch_dist:10.6f} 0.000000")
        lines.append(f"H {x:10.6f} {-ch_dist:10.6f} 0.000000")
    lines.append(f"H {-ch_dist:10.6f} 0.000000 0.000000")
    lines.append(f"H {n_carbon * cc_dist:10.6f} 0.000000 0.000000")
    return "\n".join(lines)

def run_bench(n_carbon):
    xyz = generate_alkane_xyz(n_carbon)
    mol = ferric.Molecule.from_xyz_string(xyz)
    bs  = ferric.BasisSet.bundled("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    frozen_core = n_carbon  # freeze C 1s orbitals

    t0 = time.time()
    res = ferric.run_laplace_mp2(mol, bs, aux,
                                  n_quad=7,
                                  frozen_core=frozen_core,
                                  k_builder="link")
    elapsed = time.time() - t0
    return elapsed, res

if __name__ == "__main__":
    print("AO Laplace RI-MP2 Scaling Benchmark: Linear Alkanes, cc-pVDZ/cc-pVDZ-RI, 7-point quadrature")
    print("-" * 80)
    print(f"{'N(C)':<6} {'Time (s)':>10} {'RHF (Ha)':>18} {'MP2 corr (Ha)':>16}")
    print("-" * 80)

    for n in range(1, 8):
        try:
            elapsed, res = run_bench(n)
            print(f"{n:<6} {elapsed:>10.2f} {res.total_energy - res.mp2_corr:>18.8f} {res.mp2_corr:>16.8f}")
        except Exception as e:
            print(f"{n:<6} {'Error:':>10} {e}")
