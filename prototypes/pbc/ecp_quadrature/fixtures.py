"""Fixture data (basis/ECP DATA only — numbers copied from LANL2DZ / def2-SVP / def2-ECP, not code).

Shells are dicts {l, center, exponents, coefficients(BARE: contraction-normalised as PySCF does, gto_norm folded in)},
ECPs dicts {center, ams, ns, exponents, coefficients} with the local channel at max(ams) (ferric/libecpint)."""

import math

import numpy as np

from ecpq import gto_norm

# ---------------------------------------------------------------- HI / LANL2DZ (= reference/pbc/ecp_accuracy)
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
I_NAMES = ["I s(2)", "I s(1)", "I p(2)", "I p(1)"]
# (l, [(n, zeta, d)]), l = 3 is the local channel
ECP_I_LANL2DZ = [
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

# ---------------------------------------------------------------- Au / def2-SVP + def2-ECP (ECP60MWB), H / def2-SVP
AU_SVP = [
    (0, [20.115299, 12.193477], [-0.15910719389, 0.79105526778]),
    (0, [6.0735294368], [1.0]),
    (0, [1.3174451569], [1.0]),
    (0, [0.58596768244], [1.0]),
    (0, [0.13875427354], [1.0]),
    (0, [0.048876985527], [1.0]),
    (
        1,
        [8.609665, 7.335326, 1.6575296365, 0.78159310216],
        [0.50053018599, -0.72681584494, 0.57315511417, 0.49579068859],
    ),
    (1, [0.32384840661], [1.0]),
    (1, [0.054], [1.0]),
    (
        2,
        [4.143949, 3.568257, 1.234575713, 0.48190232338],
        [-0.37099566643, 0.40197233762, 0.46001988624, 0.46152130957882],
    ),
    (2, [0.16490636769], [1.0]),
    (3, [0.72482], [1.0]),
]
H_SVP = [
    (0, [13.010701, 1.9622572, 0.44453796], [0.019682158, 0.13796524, 0.47831935]),
    (0, [0.12194962], [1.0]),
    (1, [0.8], [1.0]),
]
# local channel written as l = 3 (max am) for the ferric convention; PySCF calls it -1
ECP_AU_DEF2 = [
    (3, [(2, 4.78982, 30.4900889), (2, 2.39491, 5.17107381)]),
    (
        0,
        [
            (2, 13.2051, 426.8466792),
            (2, 6.60255, 37.00708285),
            (2, 4.78982, -30.4900889),
            (2, 2.39491, -5.17107381),
        ],
    ),
    (
        1,
        [
            (2, 10.45202, 261.19958038),
            (2, 5.22601, 26.96249604),
            (2, 4.78982, -30.4900889),
            (2, 2.39491, -5.17107381),
        ],
    ),
    (
        2,
        [
            (2, 7.8511, 124.79066561),
            (2, 3.92555, 16.30072573),
            (2, 4.78982, -30.4900889),
            (2, 2.39491, -5.17107381),
        ],
    ),
]
AU0 = np.array([0.0, 0.0, 0.0])
H_AU = np.array([0.0, 0.0, 2.88])  # ~1.52 A


def bare(l, e, c):
    """PySCF contraction normalisation, then gto_norm folded in (ferric's bare-Cartesian convention)."""
    s = sum(
        ca * cb * (2 * math.sqrt(a * b) / (a + b)) ** (l + 1.5)
        for a, ca in zip(e, c)
        for b, cb in zip(e, c)
    )
    return [cc / math.sqrt(s) * gto_norm(l, a) for a, cc in zip(e, c)]


def shell(sh, pos):
    l, e, c = sh
    return dict(
        l=l,
        center=np.asarray(pos, float),
        exponents=list(e),
        coefficients=bare(l, e, c),
        raw=(l, e, c),
    )


def ecp(channels, pos, ncore, symbol):
    ams, ns, ex, co = [], [], [], []
    for l, ts in channels:
        for n, z, d in ts:
            ams.append(l)
            ns.append(n)
            ex.append(z)
            co.append(d)
    return dict(
        center=np.asarray(pos, float),
        ams=ams,
        ns=ns,
        exponents=ex,
        coefficients=co,
        ncore=ncore,
        symbol=symbol,
        channels=channels,
    )


def ecp_i(pos):
    return ecp(ECP_I_LANL2DZ, pos, 46, "I")


def ecp_au(pos):
    return ecp(ECP_AU_DEF2, pos, 60, "Au")


def hi_shells(h, i):
    return [shell(H_STO3G, h)] + [shell(s, i) for s in I_SH]


def auh_shells(au, h):
    return [shell(s, au) for s in AU_SVP] + [shell(s, h) for s in H_SVP]


def only_channels(e, keep):
    """Copy of ECP e keeping only the listed channel ls (others' coefficients zeroed; local stays the max-am tag)."""
    ch = [
        (l, [(n, z, (d if l in keep else 0.0)) for n, z, d in ts])
        for l, ts in e["channels"]
    ]
    return ecp(ch, e["center"], e["ncore"], e["symbol"])
