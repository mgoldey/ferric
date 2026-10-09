"""HI: H STO-3G (PySCF digits) + I LANL2DZ basis + LANL2DZ ECP (46 e-), as BSE JSON for ferric."""

import json, os

H = [[0, [3.42525091, 0.15432897], [0.62391373, 0.53532814], [0.1688554, 0.44463454]]]
I = [
    [0, [0.7242, -2.9731048], [0.4653, 3.4827643]],
    [0, [0.1336, 1.0]],
    [1, [1.29, -0.2092377], [0.318, 1.1035347]],
    [1, [0.1053, 1.0]],
]
ECP = [  # (l, [(n, zeta, d)]) with r^(n-2); l = 3 is the local (ul) term — as test_pbc_bindings.LANL2DZ_I_ECP
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


def shells(bas):
    return [
        {
            "function_type": "gto",
            "angular_momentum": [b[0]],
            "exponents": [repr(p[0]) for p in b[1:]],
            "coefficients": [[repr(p[1]) for p in b[1:]]],
        }
        for b in bas
    ]


def write(path):
    d = {
        "name": "HI-lanl2dz",
        "elements": {
            "1": {"electron_shells": shells(H)},
            "53": {
                "electron_shells": shells(I),
                "ecp_electrons": 46,
                "ecp_potentials": [
                    {
                        "ecp_type": "scalar_ecp",
                        "angular_momentum": [l],
                        "r_exponents": [t[0] for t in terms],
                        "gaussian_exponents": [repr(t[1]) for t in terms],
                        "coefficients": [[repr(t[2]) for t in terms]],
                    }
                    for l, terms in ECP
                ],
            },
        },
    }
    with open(path, "w") as f:
        json.dump(d, f)
    return path
