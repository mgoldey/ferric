"""Clean-band ECP derivative discrepancy: libecpint (ferric shim, via ctypes) vs PySCF oracle.
Fixture = ferric-integrals/tests/ecp_deriv_block.rs shifted_block_deriv_matches_richardson_fd_per_triple_away_from_centres."""

import ctypes as C, numpy as np, sys, os, math
from pyscf import gto

S = os.path.dirname(os.path.abspath(__file__))
lib = C.CDLL(os.path.join(S, "libecpshim.so"))


class GS(C.Structure):
    _fields_ = [
        ("l", C.c_int),
        ("nprim", C.c_int),
        ("x", C.c_double),
        ("y", C.c_double),
        ("z", C.c_double),
        ("exponents", C.POINTER(C.c_double)),
        ("coefficients", C.POINTER(C.c_double)),
    ]


class EC(C.Structure):
    _fields_ = [
        ("x", C.c_double),
        ("y", C.c_double),
        ("z", C.c_double),
        ("nterm", C.c_int),
        ("ams", C.POINTER(C.c_int)),
        ("ns", C.POINTER(C.c_int)),
        ("exponents", C.POINTER(C.c_double)),
        ("coefficients", C.POINTER(C.c_double)),
    ]


lib.ferric_ecp_block.restype = C.c_int
lib.ferric_ecp_block_deriv.restype = C.c_int

H0 = np.array([0.3, 0.2, 0.4])
I0 = np.array([0.9, -0.4, 3.3])
LV = np.array([0.8, -0.6, 1.1])
H_STO3G = (0, [3.42525091, 0.62391373, 0.1688554], [0.15432897, 0.53532814, 0.44463454])
I_SH = [
    (0, [0.7242, 0.4653], [-2.9731048, 3.4827643]),
    (0, [0.1336], [1.0]),
    (1, [1.29, 0.318], [-0.2092377, 1.1035347]),
    (1, [0.1053], [1.0]),
]
NAMES = ["H s", "I s(2)", "I s(1)", "I p(2)", "I p(1)"]
ECP_I = [
    (
        3,
        [
            (0, 1.0715702, -0.0747621),
            (1, 44.1936028, -30.0811224),
            (2, 12.9367609, -75.3722721),
            (2, 3.1956412, -22.0563758),
            (2, 0.8589806, -1.6979585),
        ],
    ),
    (
        0,
        [
            (0, 127.9202670, 2.9380036),
            (1, 78.6211465, 41.2471267),
            (2, 36.5146237, 287.8680095),
            (2, 9.9065681, 114.3758506),
            (2, 1.9420086, 37.6547714),
        ],
    ),
    (
        1,
        [
            (0, 13.0035304, 2.2222630),
            (1, 76.0331404, 39.4090831),
            (2, 24.1961684, 177.4075002),
            (2, 6.4053433, 77.9889462),
            (2, 1.5851786, 25.7547641),
        ],
    ),
    (
        2,
        [
            (0, 40.4278108, 7.0524360),
            (1, 28.9084375, 33.3041635),
            (2, 15.6268936, 186.9453875),
            (2, 4.1442856, 71.9688361),
            (2, 0.9377235, 9.3630657),
        ],
    ),
]


def gto_norm(l, a):
    return math.sqrt(
        2 ** (2 * l + 3)
        * math.factorial(l + 1)
        * (2 * a) ** (l + 1.5)
        / (math.factorial(2 * l + 2) * math.sqrt(math.pi))
    )


def bare(l, e, c):
    s = sum(
        ca * cb * (2 * math.sqrt(a * b) / (a + b)) ** (l + 1.5)
        for a, ca in zip(e, c)
        for b, cb in zip(e, c)
    )
    return [cc / math.sqrt(s) * gto_norm(l, a) for a, cc in zip(e, c)]


C2S = {0: 0.282094791773878143, 1: 0.488602511902919921}
KEEP = []


def gshell(sh, pos):
    l, e, c = sh
    ea = (C.c_double * len(e))(*e)
    ca = (C.c_double * len(e))(*bare(l, e, c))
    KEEP.extend([ea, ca])
    return GS(l, len(e), *map(float, pos), ea, ca)


def ecenter(pos):
    ams, ns, ex, co = [], [], [], []
    for l, ts in ECP_I:
        for n, z, d in ts:
            ams.append(l)
            ns.append(n)
            ex.append(z)
            co.append(d)
    arrs = [
        (C.c_int * len(ams))(*ams),
        (C.c_int * len(ns))(*ns),
        (C.c_double * len(ex))(*ex),
        (C.c_double * len(co))(*co),
    ]
    KEEP.extend(arrs)
    return EC(*map(float, pos), len(ams), *arrs)


def nc(l):
    return (l + 1) * (l + 2) // 2


def lib_value(sa, pa, sb, pb, pc):
    a = (GS * 1)(gshell(sa, pa))
    b = (GS * 1)(gshell(sb, pb))
    u = (EC * 1)(ecenter(pc))
    n = nc(sa[0]) * nc(sb[0])
    out = (C.c_double * n)()
    st = lib.ferric_ecp_block(a, 1, b, 1, u, 1, None, out, C.c_longlong(n))
    assert st == 0, st
    return np.array(out[:]).reshape(nc(sa[0]), nc(sb[0])) * C2S[sa[0]] * C2S[sb[0]]


def lib_deriv(sa, pa, sb, pb, pc):
    a = (GS * 1)(gshell(sa, pa))
    b = (GS * 1)(gshell(sb, pb))
    u = (EC * 1)(ecenter(pc))
    n = nc(sa[0]) * nc(sb[0])
    ob, ok, oc = (
        (C.c_double * (3 * n))(),
        (C.c_double * (3 * n))(),
        (C.c_double * (3 * n))(),
    )
    g = (C.c_int * 1)(0)
    st = lib.ferric_ecp_block_deriv(
        a, 1, b, 1, u, 1, None, g, 1, ob, ok, oc, C.c_longlong(n)
    )
    assert st == 0, st
    f = C2S[sa[0]] * C2S[sb[0]]
    sh = (3, nc(sa[0]), nc(sb[0]))
    return [np.array(o[:]).reshape(sh) * f for o in (ob, ok, oc)]


AUG = 1e-3


def pyscf_value(sa, pa, sb, pb, pc, guard=True):
    def bas(sh):
        l, e, c = sh
        prims = [[x, y] for x, y in zip(e, c)]
        if guard:
            prims.append([AUG, 0.0])
        return [[l] + prims]

    mol = gto.M(
        atom=[("H@1", pa), ("H@2", pb), ("I", pc)],
        basis={"H@1": bas(sa), "H@2": bas(sb), "I": "lanl2dz"},
        ecp={"I": "lanl2dz"},
        unit="B",
        verbose=0,
        spin=1,
        cart=False,
    )
    v = mol.intor("ECPscalar_sph", shls_slice=(0, 1, 1, 2))
    return v


def rich(f, h=1e-4):
    fd = lambda hh: (f(hh) - f(-hh)) / (2 * hh)
    return (4 * fd(h / 2) - fd(h)) / 3


def main():
    bra = [(H_STO3G, H0)] + [(s, I0) for s in I_SH]
    ket = [(s, p + LV) for s, p in bra]
    cs = [
        I0,
        I0 + LV + np.array([0.3, 0.2, -0.25]),
        H0 + np.array([-0.5, 0.4, 0.6]),
        I0 + np.array([1.5, -1.2, 0.8]),
    ]
    rows = []
    for ia, (sa, pa) in enumerate(bra):
        for ib, (sb, pb) in enumerate(ket):
            for iu, pc in enumerate(cs):
                dac, dbc = np.linalg.norm(pa - pc), np.linalg.norm(pb - pc)
                dmin = min(dac, dbc)
                if dmin < 0.5:
                    continue
                an = lib_deriv(sa, pa, sb, pb, pc)
                for x in range(3):
                    e = np.eye(3)[x]
                    fs = {
                        "bra": lambda d: lib_value(sa, pa + d * e, sb, pb, pc),
                        "ket": lambda d: lib_value(sa, pa, sb, pb + d * e, pc),
                        "centre": lambda d: lib_value(sa, pa, sb, pb, pc + d * e),
                    }
                    for k, slot in enumerate(("bra", "ket", "centre")):
                        fd = rich(fs[slot])
                        diff = np.abs(an[k][x] - fd)
                        idx = np.unravel_index(np.argmax(diff), diff.shape)
                        rows.append(
                            dict(
                                dmin=dmin,
                                dac=dac,
                                dbc=dbc,
                                ia=ia,
                                ib=ib,
                                iu=iu,
                                slot=slot,
                                x=x,
                                idx=idx,
                                an=an[k][x][idx],
                                fd=fd[idx],
                                err=diff[idx],
                                scale=np.abs(an[k][x]).max(),
                            )
                        )
    for lo, hi, name in ((1.0, 1e9, "CLEAN"), (0.5, 1.0, "MID")):
        rs = [r for r in rows if lo <= r["dmin"] < hi]
        print(
            f"{name}: {len(rs)} checks, worst {max(r['err'] for r in rs):.3e}, scale {max(r['scale'] for r in rs):.3f}"
        )
    worst = {}
    for lo, hi, name in ((1.0, 1e9, "CLEAN"), (0.5, 1.0, "MID")):
        rs = sorted([r for r in rows if lo <= r["dmin"] < hi], key=lambda r: -r["err"])[
            :10
        ]
        worst[name] = rs
    import pickle

    pickle.dump((worst, bra, ket, cs), open(os.path.join(S, "worst.pkl"), "wb"))
    for name, rs in worst.items():
        print(
            f"--- {name} worst 10: libecpint analytic / libecpint FD / PySCF FD (guard) / PySCF FD (no guard)"
        )
        for r in rs:
            sa, pa = bra[r["ia"]]
            sb, pb = ket[r["ib"]]
            pc = cs[r["iu"]]
            e = np.eye(3)[r["x"]]
            if r["slot"] == "bra":
                f = lambda d, g=True: pyscf_value(sa, pa + d * e, sb, pb, pc, g)
            elif r["slot"] == "ket":
                f = lambda d, g=True: pyscf_value(sa, pa, sb, pb + d * e, pc, g)
            else:
                f = lambda d, g=True: pyscf_value(sa, pa, sb, pb, pc + d * e, g)
            o = rich(f)[r["idx"]]
            og = rich(lambda d: f(d, False))[r["idx"]]
            v_lib = lib_value(sa, pa, sb, pb, pc)[r["idx"]]
            v_py = pyscf_value(sa, pa, sb, pb, pc)[r["idx"]]
            print(
                f"{r['slot']:6s} x{r['x']} bra {NAMES[r['ia']]} @{pa.round(3).tolist()} ket {NAMES[r['ib']]} @{pb.round(3).tolist()} "
                f"U{r['iu']} @{pc.round(3).tolist()} elem {tuple(int(i) for i in r['idx'])} |A-C| {r['dac']:.2f} |B-C| {r['dbc']:.2f}\n"
                f"    an {r['an']:+.12e} libFD {r['fd']:+.12e} pyFD {o:+.12e} pyFD(noguard) {og:+.12e}\n"
                f"    |an-py| {abs(r['an'] - o):.2e}  |libFD-py| {abs(r['fd'] - o):.2e}  |an-libFD| {r['err']:.2e};  "
                f"value lib {v_lib:+.12e} py {v_py:+.12e} |dV| {abs(v_lib - v_py):.2e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
