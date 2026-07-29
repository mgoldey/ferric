#!/usr/bin/env python3
"""Density-matrix sparsity vs system size (n-alkanes, cc-pVDZ).

Measures whether the SCF density matrix actually becomes sparse as molecules
grow, and at what rate. Reports the LOCAL scaling exponent between successive
systems, not just a global fit -- a global fit averages in the pre-onset
systems and hides the effect entirely (1.90 global vs 1.68 last-three on the
same data).

Results and caveats: docs/ao-laplace-locality-saturation.md section 2.
Two limits worth repeating: this is |P| > 1e-6 (the >1e-8 tail does NOT thin),
and it is the SCF density, not the Laplace pseudo-densities the AO path
contracts.

Run:
  OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH \
    python -u scripts/probes/alkane_density_sparsity.py
"""
import math, numpy as np, ferric
bs=ferric.BasisSet.bundled("cc-pvdz")
print(f"{'system':11}{'nat':>4}{'nbas':>6}{'diam':>7}"
      f"{'f>1e-6':>9}{'f>1e-8':>9}{'nnz6':>10}{'pairs/atom':>12}", flush=True)
rows=[]
for n in ("alkane_4","alkane_8","alkane_12","alkane_16","alkane_18","alkane_20"):
    p=f"testdata/molecules/{n}.xyz"
    ls=[l.split() for l in open(p).read().strip().split("\n")[2:] if l.strip()]
    c=[(float(a[1]),float(a[2]),float(a[3])) for a in ls]
    nat=len(c); d=max(math.dist(x,y) for x in c for y in c)*1.8897259886
    m=ferric.Molecule.from_xyz(p)
    try: r=ferric.run_rhf(m,bs)
    except Exception as e:
        print(f"{n:11}{nat:>4} RHF FAIL {str(e)[:38]}",flush=True); continue
    a=np.abs(np.asarray(r.density())); nb=a.shape[0]
    n6=int((a>1e-6).sum()); f6=n6/a.size; f8=(a>1e-8).sum()/a.size
    print(f"{n:11}{nat:>4}{nb:>6}{d:>7.1f}{f6:>9.3f}{f8:>9.3f}{n6:>10}{n6/nat:>12.1f}",flush=True)
    rows.append((nat,nb,f6,n6))
if len(rows)>=3:
    nb=np.array([r[1] for r in rows],float); n6=np.array([r[3] for r in rows],float)
    print("\nLOCAL exponent between successive systems (the informative number):",flush=True)
    for i in range(len(nb)-1):
        le=np.log(n6[i+1]/n6[i])/np.log(nb[i+1]/nb[i])
        print(f"  nbas {int(nb[i])} -> {int(nb[i+1])}: {le:.2f}",flush=True)
    g,_=np.polyfit(np.log(nb),np.log(n6),1)
    print(f"\nglobal fit  nnz ~ nbas^{g:.2f}  <-- averages in pre-onset points, MISLEADING",flush=True)
    if len(nb)>=3:
        t,_=np.polyfit(np.log(nb[-3:]),np.log(n6[-3:]),1)
        print(f"last-3 fit  nnz ~ nbas^{t:.2f}  <-- use this one",flush=True)
    print("\n2.0 = fully dense, 1.0 = linear scaling. Falling local exponent = sparsity onset.",flush=True)
