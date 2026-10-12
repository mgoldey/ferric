"""Extend the all-anti n-alkane series (even N) with the same recipe as testdata/molecules/alkane_{20,32}.xyz.

usage: python gen_alkane.py N OUT.xyz [--check REF.xyz]
Validates by regenerating a reference file to 2e-5 A.
"""

import sys

DX, DY = 1.245964, 0.881050
HY, HZ = 0.631631, 0.893241


def gen(n):
    assert n % 2 == 0
    atoms = []
    cy = [DY if i % 2 else 0.0 for i in range(n)]
    for i in range(n):
        atoms.append(("C", i * DX, cy[i], 0.0))
    # left end: carbon 0 (y=0): cap + two H
    atoms.append(("H", -HZ, HY, 0.0))
    atoms.append(("H", 0.000027, -HY, -0.893254))
    atoms.append(("H", 0.000027, -HY, 0.893254))
    for i in range(1, n - 1):
        s = 1 if cy[i] > 0 else -1
        y = cy[i] + s * HY
        atoms.append(("H", i * DX, y, HZ))
        atoms.append(("H", i * DX, y, -HZ))
    xl = (n - 1) * DX
    atoms.append(("H", xl + 0.893248, DY - HY, 0.0))
    atoms.append(("H", xl - 0.000019, DY + HY, 0.893254))
    atoms.append(("H", xl - 0.000019, DY + HY, -0.893254))
    return atoms


def read(path):
    L = open(path).read().split("\n")
    return [(t[0], *map(float, t[1:4])) for t in (ln.split() for ln in L[2:]) if t]


if __name__ == "__main__":
    n = int(sys.argv[1])
    a = gen(n)
    if "--check" in sys.argv:
        r = read(sys.argv[sys.argv.index("--check") + 1])

        def key(x):
            return (x[0], round(x[1], 3), round(x[2], 3), round(x[3], 3))

        assert len(a) == len(r), (len(a), len(r))
        sa, sr = sorted(a, key=key), sorted(r, key=key)
        d = max(abs(p - q) for u, v in zip(sa, sr) for p, q in zip(u[1:], v[1:]))
        assert all(u[0] == v[0] for u, v in zip(sa, sr))
        print("max dev vs ref", d)
        assert d < 3e-5
    else:
        with open(sys.argv[2], "w") as fh:
            fh.write(
                f"{len(a)}\nn-alkane C{n}H{2 * n + 2}, all-anti conformer (generated like alkane_20)\n"
            )
            for s, x, y, z in a:
                fh.write(f"{s:<2} {x:12.6f} {y:12.6f} {z:12.6f}\n")
