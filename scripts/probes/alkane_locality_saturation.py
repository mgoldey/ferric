import math, sys, ferric
bs  = ferric.BasisSet.bundled("cc-pvdz")
aux = ferric.BasisSet.bundled("cc-pvdz-ri")
RADII = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0]
def diam(p):
    ls=[l.split() for l in open(p).read().strip().split("\n")[2:] if l.strip()]
    c=[(float(a[1]),float(a[2]),float(a[3])) for a in ls]
    return len(c), max(math.dist(a,b) for a in c for b in c)*1.8897259886
print("system      nat    diam | " + " ".join(f"r={r:<5.1f}" for r in RADII), flush=True)
rows=[]
for name in ("alkane_2","alkane_4","alkane_6","alkane_8","alkane_10"):
    p=f"testdata/molecules/{name}.xyz"
    nat,d = diam(p)
    m = ferric.Molecule.from_xyz(p)
    try:
        dense = ferric.run_laplace_sos_mp2(m,bs,aux,c_os=1.0,formulation="ao",
                                           memory_budget_gb=8.0).e_os
    except Exception as e:
        print(f"{name:10} {nat:>4} {d:>7.1f} | DENSE FAILED: {str(e)[:50]}", flush=True); continue
    cells=[]; r_chem=None
    for r in RADII:
        try:
            e = ferric.run_laplace_sos_mp2(m,bs,aux,c_os=1.0,formulation="ao-sparse",
                                           domain_cutoff_bohr=r,memory_budget_gb=8.0).e_os
            rel=abs(e-dense)/abs(dense); cells.append(f"{rel:7.1e}")
            if r_chem is None and abs(e-dense) < 1.6e-3: r_chem=r
        except Exception as ex:
            cells.append("  ERR  ")
    print(f"{name:10} {nat:>4} {d:>7.1f} | " + " ".join(cells) + f"   r(chem)={r_chem}", flush=True)
    rows.append((name,d,r_chem))
print("\n=== SATURATION: does r(chem) stay flat while diameter grows? ===", flush=True)
print(f"{'system':10} {'diam':>8} {'r(chem)':>9} {'r/diam':>8}", flush=True)
for name,d,rc in rows:
    print(f"{name:10} {d:>8.1f} {str(rc):>9} {(f'{rc/d:.2f}' if rc else 'n/a'):>8}", flush=True)
